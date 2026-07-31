"""Training script V5 - Full physics pipeline with Jacobian-based ID loss.

Predicts 12 outputs:
- COP (4): [Right X, Right Z, Left X, Left Z] in ground-aligned calc frame
  (Physiologically normalized by height)
- GRF (6): [Right X, Right Y, Right Z, Left X, Left Y, Left Z] (Physiologically normalized by mass)
- Free Moments (2): [Right Z, Left Z] (Physiologically normalized by mass)

Normalization Workflow:
1. Physiological: Inputs/targets scaled by body height/mass.
2. Statistical: Inputs/targets Z-scored using dataset-wide mean/std (Normalizer class).
3. Physics Loss: Predictions unnormalized back to physical units (N, Nm) for Jacobian multiplication.
"""

import os


def _quiet_loader_logs() -> bool:
    return os.environ.get("MJX_DATALOADER_QUIET", "").strip().lower() in {"1", "true", "yes", "on"}


try:
    from wandb_utils import WandbLogger, configure_runtime_env
except ModuleNotFoundError:
    class WandbLogger:  # type: ignore[override]
        """No-op fallback when wandb_utils is unavailable."""

        def __init__(self, *args, **kwargs):
            self.is_active = False

        def log(self, *args, **kwargs):
            return None

        def log_artifact(self, *args, **kwargs):
            return None

        def save_file(self, *args, **kwargs):
            return None

        def set_summary(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    def configure_runtime_env():
        return {}

RUNTIME_ENV_APPLIED = configure_runtime_env()

import sys
import json
import hashlib
import argparse
import gc
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
import time
from glob import glob
import re
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import traverse_util
import optax
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt

# Import memory-efficient data loader
from data_loader import (
    TrialDataLoader,
    flatten_jacobian_components as flatten_jacobian_components_np,
    flatten_rotation_matrices as flatten_rotation_matrices_np,
    mocap_processed_dir,
    subject_group_id,
    select_pos_input_columns,
    video_processed_dir,
    unnormalize_qfrc_inverse_by_bw_height,
    validate_prediction_margin,
)

# Hardcoded minimum trial length (in frames)
MIN_TRIAL_LENGTH = 30

# Configuration
ENABLE_COP_TRIM = False  # Set to True to enable COP trimming logic
MAX_GRF_PERCENTAGE = 75.0 # Exclude trials with NonZero Vertical GRF % > 90% (Static Poses)
MAX_PELVIS_SLOPE = 0.0005 # Exclude trials with |Pelvis_Z_Slope| > 0.0003 (Drift)
MAX_AVERAGE_NORM_VGRF = 1.15 # Exclude trials with Average Normalized Vertical GRF > 1.2 (Abnormal Loads)
MIN_AVERAGE_NORM_VGRF = 0.85 # Exclude trials with Average Normalized Vertical GRF < 0.5 (Abnormal Loads)
OnlySuperviseStance = False # If True, only apply supervision during stance phases (contactBoolean)
AUXILIARY_MODEL_OUTPUTS_ENABLED = False

# Optional coupled presets for sweeping fixed "base" model configs.
# These presets intentionally control core architecture only.
BASE_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cfg01": {"dropout_rate": 0.362, "d_model": 256,  "num_layers": 4, "window_size": 100},
    "cfg02": {"dropout_rate": 0.28, "d_model": 512,  "num_layers": 4, "window_size": 120},
    "cfg03": {"dropout_rate": 0.233, "d_model": 128,  "num_layers": 6, "window_size": 125},
}
BASE_MODEL_CONFIG_KEYS = (
    "dropout_rate",
    "d_model",
    "num_layers",
    "window_size",
)


def _parse_optional_bool_arg(value):
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _parse_save_model_epochs_arg(value: object) -> List[int]:
    """Parse epoch lists like "(7,8,9)" or "7,8,9" into sorted unique positive ints."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parsed_values = [int(v) for v in value]
    else:
        text = str(value).strip()
        if not text:
            return []
        if text[:1] in "([{" and text[-1:] in ")]}":
            text = text[1:-1].strip()
        if not text:
            return []
        parsed_values = [int(part.strip()) for part in text.split(",") if part.strip()]

    invalid_values = [epoch for epoch in parsed_values if epoch <= 0]
    if invalid_values:
        raise argparse.ArgumentTypeError(
            f"All save-model epochs must be positive integers, got: {invalid_values}"
        )
    return sorted(set(parsed_values))


def load_wandb_key_from_env(repo_root: str = None) -> Optional[str]:
    """Load wandb API key from .env file at repo root."""
    if repo_root is None:
        # Find repo root (look for .env file going up from current dir)
        current = Path(__file__).parent.absolute()
        repo_root = current.parent  # GaitDynamics_jax -> GaitDynamics-JAX
    
    env_file = Path(repo_root) / ".env"
    if not env_file.exists():
        return None
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            # Match: wandb = "key" or wandb="key"
            match = re.search(r'wandb\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
    except Exception as e:
        print(f"Warning: Could not read .env file: {e}", flush=True)
    
    return None


def _format_sweep_value(value: object) -> str:
    """Format sweep values compactly for run names."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _as_numpy_debug(value: Any) -> Optional[np.ndarray]:
    """Best-effort conversion for lightweight NaN diagnostics."""
    try:
        return np.asarray(jax.device_get(value))
    except Exception:
        try:
            return np.asarray(value)
        except Exception:
            return None


def _format_nonfinite_detail(
    name: str,
    value: Any,
    *,
    trial_name_map: Optional[Dict[int, str]] = None,
    raw_batch: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return a compact description when a tensor contains NaN/Inf values."""
    arr = _as_numpy_debug(value)
    if arr is None:
        return f"{name}: unable to convert for diagnostics"
    if arr.size == 0 or np.isfinite(arr).all():
        return None

    bad_mask = ~np.isfinite(arr)
    first_idx = tuple(int(i) for i in np.argwhere(bad_mask)[0])
    first_val = arr[first_idx]
    finite_vals = arr[np.isfinite(arr)]
    finite_summary = "no finite values"
    if finite_vals.size > 0:
        finite_summary = (
            f"finite_min={float(np.min(finite_vals)):.6g}, "
            f"finite_max={float(np.max(finite_vals)):.6g}, "
            f"finite_mean={float(np.mean(finite_vals)):.6g}"
        )

    lines = [
        (
            f"{name}: first_nonfinite_idx={first_idx}, value={first_val}, "
            f"shape={arr.shape}, dtype={arr.dtype}, "
            f"nonfinite={int(np.count_nonzero(bad_mask))}/{int(arr.size)}, {finite_summary}"
        )
    ]

    if raw_batch is not None and len(first_idx) > 0:
        batch_row = first_idx[0]
        trial_idx_arr = _as_numpy_debug(raw_batch.get("trial_idx")) if "trial_idx" in raw_batch else None
        if trial_idx_arr is not None and trial_idx_arr.size > batch_row:
            trial_idx = int(trial_idx_arr.reshape(-1)[batch_row])
            trial_name = trial_name_map.get(trial_idx, f"trial_idx={trial_idx}") if trial_name_map else f"trial_idx={trial_idx}"
            lines.append(f"   batch_row={batch_row}, trial={trial_name}")
        window_arr = _as_numpy_debug(raw_batch.get("window_start_idx")) if "window_start_idx" in raw_batch else None
        if window_arr is not None and window_arr.size > batch_row:
            lines.append(f"   window_start_idx={int(window_arr.reshape(-1)[batch_row])}")

    return "\n".join(lines)


def _report_first_nonfinite_training_step(
    step_num: int,
    raw_batch: Dict[str, Any],
    batch_norm: Dict[str, Any],
    metrics: Dict[str, Any],
    pred: Any,
    normalizers: Dict[str, Any],
    *,
    trial_name_map: Optional[Dict[int, str]] = None,
) -> None:
    """Print the earliest non-finite tensor we can identify for a failed step."""
    print("\n❌ Non-finite values detected during training; stopping at first bad step.", flush=True)
    print(f"   Step: {step_num}", flush=True)

    metric_summary = ", ".join(
        f"{k}={float(metrics[k]):.6g}" for k in sorted(metrics.keys())
    )
    print(f"   Metrics: {metric_summary}", flush=True)

    qfrc_inverse_output_dim = _infer_qfrc_inverse_output_dim(batch=raw_batch, normalizers=normalizers)
    pred_np = _as_numpy_debug(pred)
    head_candidates: List[Tuple[str, Any]] = []
    if pred_np is not None:
        cop_pred, grf_pred, moments_pred, contact_pred, qfrc_inverse_pred, rotation_pred, jacobian_pred = split_model_predictions(
            pred_np,
            qfrc_inverse_output_dim=qfrc_inverse_output_dim,
            rotation_output_dim=0,
        )
        head_candidates = [
            ("pred/all", pred_np),
            ("pred/cop", cop_pred),
            ("pred/grf", grf_pred),
            ("pred/moments", moments_pred),
            ("pred/contact", contact_pred),
            ("pred/qfrc_inverse", qfrc_inverse_pred),
            ("pred/rotation", rotation_pred),
            ("pred/jacobian", jacobian_pred),
        ]

    candidates: List[Tuple[str, Any]] = [
        ("batch_norm/input", batch_norm.get("input")),
        ("batch_norm/static_context", batch_norm.get("static_context")),
        ("batch_norm/cop", batch_norm.get("cop")),
        ("batch_norm/grf", batch_norm.get("grf")),
        ("batch_norm/moments", batch_norm.get("moments")),
        ("batch_norm/qfrc_inverse_gt", batch_norm.get("qfrc_inverse_gt")),
        ("batch_norm/gt_rot_w_to_ga", batch_norm.get("gt_rot_w_to_ga")),
        ("batch_norm/jacobian_gt", batch_norm.get("jacobian_gt")),
    ] + head_candidates

    for name, value in candidates:
        if value is None:
            continue
        detail = _format_nonfinite_detail(
            name,
            value,
            trial_name_map=trial_name_map,
            raw_batch=raw_batch,
        )
        if detail is not None:
            print(detail, flush=True)
            return

    print("   No non-finite tensor was isolated in the checked batch/prediction tensors; the issue may be in an intermediate physics computation.", flush=True)


def _tree_first_nonfinite_detail(tree: Any, prefix: str) -> Optional[str]:
    """Return the first non-finite leaf/path found in a param or optimizer tree."""
    try:
        flat = traverse_util.flatten_dict(tree, keep_empty_nodes=False, sep="/")
    except Exception:
        return None
    for path, value in flat.items():
        detail = _format_nonfinite_detail(f"{prefix}/{path}", value)
        if detail is not None:
            return detail
    return None


def _extract_keys_from_sweep_yaml_text(text: str) -> List[str]:
    """Extract top-level parameter keys from a sweep yaml `parameters:` block."""
    keys: List[str] = []
    in_parameters = False
    parameters_indent: Optional[int] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if not in_parameters:
            if stripped == "parameters:":
                in_parameters = True
                parameters_indent = indent
            continue

        if parameters_indent is not None and indent <= parameters_indent:
            break

        if parameters_indent is not None and indent == parameters_indent + 2 and stripped.endswith(":"):
            key = stripped[:-1].strip()
            if key:
                keys.append(key)

    return keys


def _extract_sweep_param_names_from_env_or_repo() -> List[str]:
    """Best-effort detection of varied sweep parameter names."""
    # Optional explicit override.
    env_names = os.environ.get("WANDB_SWEEP_PARAM_NAMES")
    if env_names:
        parsed = [name.strip() for name in env_names.split(",") if name.strip()]
        if parsed:
            return parsed

    # W&B agent commonly writes selected sweep params to a file.
    param_path = os.environ.get("WANDB_SWEEP_PARAM_PATH")
    if param_path:
        try:
            text = Path(param_path).read_text(encoding="utf-8")
            try:
                parsed_json = json.loads(text)
                if isinstance(parsed_json, dict):
                    return [str(k) for k in parsed_json.keys()]
            except Exception:
                keys = _extract_keys_from_sweep_yaml_text(text)
                if keys:
                    return keys
        except Exception:
            pass

    # Repository fallback: if there's exactly one sweep yaml near this file, use it.
    sweep_files = sorted(Path(__file__).resolve().parent.glob("sweep*.y*ml"))
    if len(sweep_files) == 1:
        try:
            text = sweep_files[0].read_text(encoding="utf-8")
            keys = _extract_keys_from_sweep_yaml_text(text)
            if keys:
                return keys
        except Exception:
            pass

    return []


def build_wandb_run_name(args: argparse.Namespace) -> str:
    """Build readable W&B run names, emphasizing swept hyperparameters."""
    base_name = args.exp_name if str(args.exp_name).strip() else Path(args.output_dir).name
    if not os.environ.get("WANDB_SWEEP_ID"):
        return base_name

    # Abbreviations for all currently available tunable hyperparameters.
    short = {
        "epochs": "ep",
        "batch_size": "bs",
        "learning_rate": "lr",
        "dropout_rate": "dr",
        "weight_decay": "wd",
        "window_size": "ws",
        "stride": "st",
        "prediction_margin_frames": "pmf",
        "base_config_id": "bcfg",
        "d_model": "dm",
        "num_layers": "nl",
        "ff_dim": "ff",
        "log_interval": "li",
        "vis_interval": "vi",
        "scan_workers": "scw",
        "resume_checkpoint": "rcp",
        "refresh_cache": "rfc",
        "cop_weight": "cw",
        "grf_weight": "gw",
        "moments_weight": "mw",
        "contact_weight": "ctw",
        "contact_weight_multiplier": "ctwm",
        "magWeight": "mgw",
        "torque_weight": "tw",
        "qfrc_inverse_weight": "qiw",
        "qfrc_inverse_input_reg_weight": "qirw",
        "rotation_weight": "rtw",
        "rotation_input_reg_weight": "rirw",
        "grf_correction_weight": "gcw",
        "output_reg_weight": "orw",
        "hip_add_r_weight": "harw",
        "knee_r_weight": "krw",
        "ankle_r_weight": "arw",
        "subtalar_r_weight": "srw",
        "hip_add_l_weight": "halw",
        "knee_l_weight": "klw",
        "ankle_l_weight": "alw",
        "subtalar_l_weight": "slw",
        "lumbar_extension_weight": "lew",
        "lumbar_bending_weight": "lbw",
        "lumbar_rotation_weight": "lrw",
        "magOnOff": "mag",
        "contactOnOff": "con",
        "use_contact_weighting": "ucw",
        "trim_cop": "tcp",
        "UseNoised": "nz",
        "NoisedGT": "ngt",
        "jacobian_weight": "jw",
        "jacobian_input_reg_weight": "jirw",
        "cop_mask": "cm",
        "BestModelByTorque": "bmt",
        "BestModel_TorqueWeighting": "btw",
    }

    sweep_param_names = _extract_sweep_param_names_from_env_or_repo()
    if not sweep_param_names:
        # Conservative fallback: only include names present in argv and known to `short`.
        provided_flags = set()
        for token in sys.argv[1:]:
            if not token.startswith("--"):
                continue
            flag = token[2:].split("=", 1)[0].strip()
            if flag:
                provided_flags.add(flag)
        sweep_param_names = [name for name in short.keys() if name in provided_flags]

    pieces = []
    for name in sweep_param_names:
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None:
            continue
        pieces.append(f"{short.get(name, name)}={_format_sweep_value(value)}")

    if not pieces:
        return base_name
    return f"{base_name} | {','.join(pieces)}"


def resolve_output_dir_for_run(args: argparse.Namespace, run_name: str) -> str:
    """Resolve the run folder using the experiment name only.

    The caller may pass either a base output directory or a fully-resolved
    experiment folder. In both cases, we avoid appending generated suffixes.
    """
    output_dir = str(args.output_dir)
    exp_name = str(getattr(args, "exp_name", "") or "").strip()
    if not exp_name:
        return output_dir

    if Path(output_dir).name == exp_name:
        return output_dir
    return os.path.join(output_dir, exp_name)


def _yaml_scalar(value: Any) -> str:
    """Format a Python scalar as a YAML-friendly value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return format(value, ".16g")
        return json.dumps(str(value))
    return json.dumps(str(value))


def save_model_parameters_yaml(params: Dict[str, Any], output_path: str) -> None:
    """Write model parameters to YAML without requiring external dependencies."""
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {_yaml_scalar(item)}\n")
            else:
                f.write(f"{key}: {_yaml_scalar(value)}\n")


def _json_compatible(value: Any) -> Any:
    """Convert nested metrics/config payloads into strict JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_compatible(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    return None if not np.isfinite(as_float) else as_float


def apply_base_model_config(args: argparse.Namespace) -> Optional[str]:
    """Apply coupled preset hyperparameters when --base_config_id is provided."""
    cfg_id = str(getattr(args, "base_config_id", "") or "").strip()
    if not cfg_id:
        return None

    preset = BASE_MODEL_CONFIGS.get(cfg_id)
    if preset is None:
        valid_ids = ", ".join(sorted(BASE_MODEL_CONFIGS.keys()))
        raise ValueError(f"Unknown --base_config_id '{cfg_id}'. Valid options: {valid_ids}")

    for key in BASE_MODEL_CONFIG_KEYS:
        if key not in preset:
            raise ValueError(f"Base config '{cfg_id}' is missing required key '{key}'")
        setattr(args, key, preset[key])

    return cfg_id


def infer_input_feature_layout_from_loader(
    data_loader: "TrialDataLoader",
    include_pelvis_euler: bool,
    include_ankle_heights: bool = True,
    include_jacobian_input: bool = True,
    include_auxiliary_denoising_inputs: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Infer temporal input feature blocks from one successfully loaded trial.

    This is used as a guardrail to ensure the assembled model input vector
    matches what the loader actually concatenates.
    """
    sample_trial_data = None
    sample_trial_name = None
    for trial_info, _n_windows in data_loader.trial_window_counts:
        td = data_loader._load_trial(trial_info)
        if td is not None:
            sample_trial_data = td
            sample_trial_name = trial_info.get("trial_name", "unknown")
            break

    if sample_trial_data is None:
        return None

    def _dim(key: str) -> int:
        arr = np.asarray(sample_trial_data[key])
        return int(arr.shape[-1]) if arr.ndim >= 2 else 1

    pos_dim = int(select_pos_input_columns(sample_trial_data["pos"], include_pelvis_euler=include_pelvis_euler).shape[-1])

    blocks: List[Dict[str, Any]] = [
        {"name": "pelvis_rot", "dim": _dim("pelvis_rot")},
        {"name": "pos", "dim": pos_dim},
        {"name": "vel", "dim": _dim("vel")},
        {"name": "com_r", "dim": _dim("com_r")},
        {"name": "com_l", "dim": _dim("com_l")},
        {"name": "com_accel", "dim": _dim("com_accel")},
    ]
    if include_ankle_heights:
        blocks.append({"name": "ankle_heights", "dim": _dim("ankle_heights")})
    if include_jacobian_input:
        jacobian_dim = int(
            flatten_jacobian_components_np(
                np.asarray(sample_trial_data["jacp"][:1]),
                np.asarray(sample_trial_data["jacr"][:1]),
            ).shape[-1]
        )
        blocks.append({"name": "jacobian_input", "dim": jacobian_dim})

    blocks.extend([
        {"name": "foot_progression_angle", "dim": _dim("foot_progression_angle")},
        {"name": "calcn_to_floor_angle", "dim": _dim("calcn_to_floor_angle")},
    ])
    if include_auxiliary_denoising_inputs:
        blocks.append({"name": "qfrc_inverse_input", "dim": _dim("qfrc_inverse")})
        rot_dim = int(
            flatten_rotation_matrices_np(
                np.asarray(sample_trial_data["rot_w_to_ga"][:1], dtype=np.float32)
            ).shape[-1]
        )
        blocks.append({"name": "rot_w_to_ga_input_flat", "dim": rot_dim})

    total_dim = int(sum(int(b["dim"]) for b in blocks))
    return {
        "sample_trial": sample_trial_name,
        "contact_boolean_is_input": False,
        "direct_target_prediction": True,
        "blocks": blocks,
        "total_dim": total_dim,
    }


# =============================================================================
# Data Discovery and Loading
# =============================================================================

def discover_all_trials(
    data_dir: str,
    refresh_cache: bool = False,
    scan_workers: Optional[int] = None,
    layout: str = "trusted",
) -> List[Dict]:
    """Discover all valid trials.

    ``layout="trusted"`` is the standard training dataset layout:
    Subject/Trial_*/ProcessedData.

    ``layout="opencap"`` is used by LOSO/OpenCap validation:
    Subject/trial_*/Video/ProcessedData and Subject/trial_*/MoCap/ProcessedData.
    """
    data_dir = Path(data_dir)
    trials = []
    
    pos_name = "pos_inputs.npy"
    layout = str(layout or "trusted").strip().lower()
    if layout not in {"trusted", "opencap"}:
        raise ValueError(f"Unsupported trial discovery layout '{layout}'. Expected 'trusted' or 'opencap'.")
    layout_version = (
        "trusted_processed_v1"
        if layout == "trusted"
        else "opencap_video_mocap_processed_v1"
    )
    
    print(f"   🔍 Searching for trials in {data_dir} (layout={layout}, Parallel Scan enabled)...", flush=True)
    # Check if a cache file exists to speed up subsequent runs
    cache_file = data_dir / "trial_discovery_cache.json"
    if cache_file.exists() and not refresh_cache:
        print(f"   📂 Loading trials from cache: {cache_file}", flush=True)
        try:
            with open(cache_file, 'r') as f:
                cached_trials = json.load(f)
            if isinstance(cached_trials, dict):
                cached_layout = cached_trials.get("layout_version")
                cached_trials_list = cached_trials.get("trials", [])
            else:
                cached_layout = None
                cached_trials_list = cached_trials

            # Reject stale caches from legacy layouts (e.g., SecondaryProcessing/TrainingData).
            data_dir_resolved = data_dir.resolve()

            def _cache_entry_is_valid(entry: Dict) -> bool:
                if cached_layout != layout_version:
                    return False
                training_td = Path(entry.get("training_data_path", ""))
                trial_root = Path(entry.get("trial_root", ""))
                if layout == "opencap":
                    video_td = Path(entry.get("video_processed_path", entry.get("training_data_path", "")))
                    mocap_td = Path(entry.get("mocap_processed_path", ""))
                    if video_td.name != "ProcessedData" or mocap_td.name != "ProcessedData":
                        return False
                    if not (video_td.exists() and mocap_td.exists() and trial_root.exists()):
                        return False
                    if not ((video_td / pos_name).exists() and (mocap_td / pos_name).exists()):
                        return False
                else:
                    if training_td.name != "ProcessedData":
                        return False
                    if not (training_td.exists() and trial_root.exists()):
                        return False
                    if not (training_td / pos_name).exists():
                        return False
                try:
                    trial_root.resolve().relative_to(data_dir_resolved)
                except ValueError:
                    return False
                return True

            if isinstance(cached_trials_list, list) and all(_cache_entry_is_valid(t) for t in cached_trials_list):
                return cached_trials_list

            print(f"   ⚠️  Trial cache is stale (not {layout} layout). Re-scanning...", flush=True)
        except Exception as e:
            print(f"   ⚠️  Failed to load cache: {e}. Re-scanning...", flush=True)

    search_start = time.time()
    
    # Use iterdir for selective traversal (Subject -> Trial)
    try:
        subjects = [s for s in data_dir.iterdir() if s.is_dir() and not s.name.startswith('.')]
    except Exception as e:
        print(f"   ❌ Error accessing data directory: {e}", flush=True)
        return []

    def process_single_subject(subject_path):
        subject_trials = []
        subject_name = subject_path.name
        
        for trial_path in subject_path.iterdir():
            if not trial_path.is_dir():
                continue

            if layout == "opencap":
                if not trial_path.name.startswith("trial_"):
                    continue
                video_td = video_processed_dir(trial_path)
                mocap_td = mocap_processed_dir(trial_path)
                if not (
                    video_td.exists()
                    and mocap_td.exists()
                    and (video_td / pos_name).exists()
                    and (mocap_td / pos_name).exists()
                ):
                    continue

                try:
                    pos_shape = np.load(video_td / pos_name, mmap_mode='r').shape
                    length = pos_shape[0]

                    if length >= MIN_TRIAL_LENGTH:
                        subject_trials.append({
                            "subject": subject_name,
                            "subject_group": subject_group_id(subject_name),
                            "trial": trial_path.name,
                            "trial_name": f"{subject_name}/{trial_path.name}",
                            "dataset_root": str(data_dir),
                            "trial_root": str(trial_path),
                            "training_data_path": str(video_td),
                            "video_processed_path": str(video_td),
                            "mocap_processed_path": str(mocap_td),
                            "length": length
                        })
                except Exception:
                    continue
            else:
                processed_td = trial_path / "ProcessedData"
                if not (processed_td.exists() and (processed_td / pos_name).exists()):
                    continue

                try:
                    pos_shape = np.load(processed_td / pos_name, mmap_mode='r').shape
                    length = pos_shape[0]

                    if length >= MIN_TRIAL_LENGTH:
                        subject_trials.append({
                            "subject": subject_name,
                            "subject_group": subject_group_id(subject_name),
                            "trial": trial_path.name,
                            "trial_name": f"{subject_name}/{trial_path.name}",
                            "dataset_root": str(data_dir),
                            "trial_root": str(trial_path),
                            "training_data_path": str(processed_td),
                            "length": length
                        })
                except Exception:
                    continue
        return subject_trials

    # Process subjects in parallel
    if scan_workers is None:
        subject_scan_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    else:
        subject_scan_workers = max(1, int(scan_workers))
    if not _quiet_loader_logs():
        print(f"   ⚙️  Processing {len(subjects)} subjects using {subject_scan_workers} threads...")
    with ThreadPoolExecutor(max_workers=subject_scan_workers) as executor:
        futures = {executor.submit(process_single_subject, s): s for s in subjects}
        for future in as_completed(futures):
            trials.extend(future.result())
    
    search_time = time.time() - search_start
    if not _quiet_loader_logs():
        print(f"   ✓ Search complete in {search_time:.1f}s (found {len(trials)} trials)", flush=True)

    # Save to cache
    try:
        with open(cache_file, 'w') as f:
            json.dump({"layout_version": layout_version, "trials": trials}, f, indent=2)
        if not _quiet_loader_logs():
            print(f"   💾 Saved {len(trials)} trials to cache: {cache_file}", flush=True)
    except Exception as e:
        print(f"   ⚠️  Failed to save cache: {e}", flush=True)

    return trials


# =============================================================================
# Normalization
# =============================================================================

class Normalizer:
    def __init__(self, data: np.ndarray = None, eps: float = 1e-8, name: str = "unknown"):
        if data is not None:
            self.mean = np.mean(data, axis=0, keepdims=True)
            self.std = np.std(data, axis=0, keepdims=True)
            # Check for near-zero std and warn
            below_eps = self.std < eps
            if np.any(below_eps):
                bad_indices = np.where(below_eps.flatten())[0]
                bad_stds = self.std.flatten()[bad_indices]
                print(f"   ⚠️  Normalizer '{name}': {len(bad_indices)} dim(s) have std < {eps}. "
                      f"Indices: {bad_indices.tolist()}, Stds: {bad_stds.tolist()}. Clamping to {eps}.", flush=True)
            self.std = np.where(self.std < eps, eps, self.std)
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def unnormalize(self, x):
        return x * self.std + self.mean


def compute_normalizers_from_loader(data_loader: TrialDataLoader, max_batches: int = 100) -> Dict[str, Normalizer]:
    """Compute normalizers from a sample of batches from the data loader.
    
    Args:
        data_loader: TrialDataLoader instance
        max_batches: Maximum number of batches to sample for computing statistics
        
    Returns:
        Dictionary of normalizers for each data type
    """
    print(f"   Sampling up to {max_batches} batches for statistics...", flush=True)
    
    # Collect data from batches
    input_samples = []
    static_samples = []
    cop_samples = []
    grf_samples = []
    moments_samples = []
    tau_samples = []
    grf_res_samples = []
    jacobian_samples = []
    qfrc_inverse_samples = []
    
    batch_count = 0
    for batch in data_loader:
        input_samples.append(np.array(batch["input"]))
        static_samples.append(np.array(batch["static_context"]))
        cop_samples.append(np.array(batch["cop"])) # Height-normalized from data_loader
        
        # GRF is already mass-normalized from data_loader
        grf_samples.append(np.array(batch["grf"]))
        
        moments_samples.append(np.array(batch["moments"]))
        tau_samples.append(np.array(batch["qfrc_grf_contribution"]))
        if "qfrc_inverse_gt" in batch and batch["qfrc_inverse_gt"] is not None:
            qfrc_inverse_samples.append(np.array(batch["qfrc_inverse_gt"]))
        if "jacobian_gt" in batch:
            jacobian_samples.append(np.array(batch["jacobian_gt"]))
        
        # Compute ground truth GRF residuals for normalization
        if "com_accel" in batch:
            com_accel = np.array(batch["com_accel"]) # (batch, T, 3)
            grf_gt = np.array(batch["grf"])          # Already N/kg from data_loader
            mass = np.array(batch["static_context"][:, 1])[:, np.newaxis, np.newaxis] # (batch, 1, 1)
            
            # Use force-based residuals (N) to equate Net_force = M * Acc
            # MJ_X (Forward)
            fx_gt = (grf_gt[..., 0] + grf_gt[..., 3]) * mass * 9.8067
            res_x = mass * com_accel[..., 0] - fx_gt
            
            # MJ_Y (Lateral)
            fy_gt = (grf_gt[..., 1] + grf_gt[..., 4]) * mass * 9.8067
            res_y = mass * com_accel[..., 1] - fy_gt
            
            # MJ_Z (Vertical)
            fz_gt = (grf_gt[..., 2] + grf_gt[..., 5]) * mass * 9.8067
            gravity = 9.8067
            res_z = mass * (com_accel[..., 2] + gravity) - fz_gt
            
            res_gt = np.stack([res_x, res_y, res_z], axis=-1)
            grf_res_samples.append(res_gt)
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"   Loaded {batch_count}/{max_batches} batches...", flush=True)
        if batch_count >= max_batches:
            break
    
    # Concatenate and flatten for statistics
    print(f"   Computing statistics from {batch_count} batches...", flush=True)
    if batch_count == 0 or not input_samples:
        raise ValueError(
            "No training batches were produced while computing normalizers. "
            "This usually means the data loader skipped every trial because required files "
            "were missing or no windows could be formed for the requested settings."
        )
    input_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in input_samples], axis=0)
    static_flat = np.concatenate([x for x in static_samples], axis=0) # (Batch, static_dim)
    cop_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in cop_samples], axis=0)
    grf_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in grf_samples], axis=0)
    moments_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in moments_samples], axis=0)
    tau_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in tau_samples], axis=0)
    qfrc_inverse_flat = (
        np.concatenate([x.reshape(-1, x.shape[-1]) for x in qfrc_inverse_samples], axis=0)
        if qfrc_inverse_samples else None
    )

    print(f"   Creating normalizers (mean/std)...", flush=True)
    # Use eps=1e-8 for input/static (many features, small scales ok)
    # Use eps=1e-3 for output normalizers (cop, grf, moments, tau, grf_res) to prevent Z-score explosion
    OUTPUT_EPS = 1e-3
    INPUT_EPS = 1e-8
    normalizers = {
        "input": Normalizer(input_flat, eps=INPUT_EPS, name="input"),
        "static": Normalizer(static_flat, eps=INPUT_EPS, name="static"),
        "cop": Normalizer(cop_flat, eps=OUTPUT_EPS, name="cop"),
        "grf": Normalizer(grf_flat, eps=OUTPUT_EPS, name="grf"),
        "moments": Normalizer(moments_flat, eps=OUTPUT_EPS, name="moments"),
        "tau": Normalizer(tau_flat, eps=OUTPUT_EPS, name="tau"),
    }
    if qfrc_inverse_flat is not None:
        normalizers["qfrc_inverse"] = Normalizer(
            qfrc_inverse_flat,
            eps=OUTPUT_EPS,
            name="qfrc_inverse",
        )
    if jacobian_samples:
        jacobian_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in jacobian_samples], axis=0)
        normalizers["jacobian"] = Normalizer(jacobian_flat, eps=OUTPUT_EPS, name="jacobian")

    if grf_res_samples:
        grf_res_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in grf_res_samples], axis=0)
        normalizers["grf_res"] = Normalizer(grf_res_flat, eps=OUTPUT_EPS, name="grf_res")
        print(f"   ✅ Added grf_res normalizer (mean std={np.mean(normalizers['grf_res'].std):.2f}N)", flush=True)
    
    print(f"   ✅ Normalizers ready (from {input_flat.shape[0]} data points)", flush=True)
    return normalizers

def normalize_batch(batch: Dict, normalizers: Dict) -> Dict:
    """Apply normalization to a batch."""
    normalized = {}
    for key, val in batch.items():
        if key == "input" and "input" in normalizers:
            normalized[key] = normalizers["input"].normalize(val)

        elif key == "static_context" and "static" in normalizers:
            normalized[key] = normalizers["static"].normalize(val)

        # Z-score outputs
        elif key == "cop" and "cop" in normalizers:
            normalized[key] = normalizers["cop"].normalize(val)
        elif key == "grf" and "grf" in normalizers:
            normalized[key] = normalizers["grf"].normalize(val)
        elif key == "moments" and "moments" in normalizers:
            normalized[key] = normalizers["moments"].normalize(val)

    # Z-score reconstruction curves too; they stay available for diagnostics.
        elif key == "cop_recon" and "cop" in normalizers:
            normalized[key] = normalizers["cop"].normalize(val)
        elif key == "grf_recon" and "grf" in normalizers:
            normalized[key] = normalizers["grf"].normalize(val)
        elif key in ["moment_recon", "moments_recon"] and "moments" in normalizers:
            normalized[key] = normalizers["moments"].normalize(val)

        # Leave torque in Nm (raw)
        elif key == "qfrc_grf_contribution":
            normalized[key] = val
        elif key == "jacobian_gt" and "jacobian" in normalizers:
            normalized[key] = normalizers["jacobian"].normalize(val)

        else:
            # pass through jacp, jacr, ankle_heights, contactBoolean, body_ids, height, mass, etc.
            normalized[key] = val

    return normalized


def _extract_frame_mask(mask: Optional[np.ndarray], seq_len: int) -> np.ndarray:
    """Convert a stored supervision mask to a 1D boolean frame mask."""
    if mask is None:
        return np.ones((seq_len,), dtype=bool)
    mask_np = np.asarray(mask)
    if mask_np.ndim >= 2:
        mask_np = mask_np[..., 0]
    mask_np = mask_np.reshape(-1)
    if mask_np.shape[0] != seq_len:
        return np.ones((seq_len,), dtype=bool)
    return mask_np > 0.5


def _extract_batched_frame_mask(mask: Optional[np.ndarray], batch_size: int, seq_len: int) -> np.ndarray:
    """Convert a batched supervision mask to shape (batch, seq) boolean."""
    if mask is None:
        return np.ones((batch_size, seq_len), dtype=bool)
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np[..., 0]
    elif mask_np.ndim == 1:
        mask_np = np.broadcast_to(mask_np[np.newaxis, :], (batch_size, seq_len))
    if mask_np.shape != (batch_size, seq_len):
        return np.ones((batch_size, seq_len), dtype=bool)
    return mask_np > 0.5


def _masked_rmse(pred: np.ndarray, gt: np.ndarray, frame_mask: np.ndarray) -> float:
    """Compute RMSE over the valid frames only."""
    mask = _extract_frame_mask(frame_mask, np.asarray(pred).shape[0])
    if not np.any(mask):
        return float("nan")
    diff = np.asarray(pred)[mask] - np.asarray(gt)[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def _masked_mae(pred: np.ndarray, gt: np.ndarray, frame_mask: np.ndarray) -> float:
    """Compute MAE over the valid frames only."""
    mask = _extract_frame_mask(frame_mask, np.asarray(pred).shape[0])
    if not np.any(mask):
        return float("nan")
    diff = np.asarray(pred)[mask] - np.asarray(gt)[mask]
    return float(np.mean(np.abs(diff)))


def _masked_max_abs_err(pred: np.ndarray, gt: np.ndarray, frame_mask: np.ndarray) -> float:
    """Compute max absolute error over the valid frames only."""
    mask = _extract_frame_mask(frame_mask, np.asarray(pred).shape[0])
    if not np.any(mask):
        return float("nan")
    diff = np.asarray(pred)[mask] - np.asarray(gt)[mask]
    return float(np.max(np.abs(diff)))


def _format_named_metric_rows(
    metric_values: Dict[str, float],
    display_names: Dict[str, str],
    ordered_keys: List[str],
    values_per_row: int = 3,
    suffix: str = "%",
) -> List[str]:
    """Format ordered metric key/value pairs into terminal-friendly rows."""
    rows = []
    parts = [
        f"{display_names.get(key, key)}: {float(metric_values.get(key, float('nan'))):.2f}{suffix}"
        for key in ordered_keys
    ]
    for start_idx in range(0, len(parts), values_per_row):
        rows.append(" | ".join(parts[start_idx:start_idx + values_per_row]))
    return rows


def _torque_stance_mask_for_name(
    dof_name: str,
    stance_r: np.ndarray,
    stance_l: np.ndarray,
) -> np.ndarray:
    """Choose the stance mask used for a torque metric based on DOF naming."""
    if dof_name.startswith("R "):
        return stance_r
    if dof_name.startswith("L "):
        return stance_l
    return np.logical_or(stance_r, stance_l)


STANDARD_OUTPUT_DIM = 14
COP_SLICE = slice(0, 4)
GRF_SLICE = slice(4, 10)
MOMENTS_SLICE = slice(10, 12)
CONTACT_SLICE = slice(12, 14)
ROTATION_PARAMETERIZATION = "residual_axis_angle"
ROTATION_COMPOSE_ORDER = "left"
ROTATION_RESIDUAL_FEET = 2
ROTATION_RESIDUAL_AXIS_DIM = 3
ROTATION_OUTPUT_DIM = ROTATION_RESIDUAL_FEET * ROTATION_RESIDUAL_AXIS_DIM
DEFAULT_ROTATION_RESIDUAL_MAX_DEG = 15.0


def flatten_rotation_matrices(
    rot: Any,
    xp=jnp,
) -> Any:
    """Flatten (..., 2, 3, 3) rotation bundles to (..., 18)."""
    leading_shape = tuple(rot.shape[:-3])
    return xp.reshape(rot, leading_shape + (-1,))


def unflatten_rotation_matrices(
    rot_flat: Any,
    xp=jnp,
) -> Any:
    """Restore flattened (..., 18) rotation bundles to (..., 2, 3, 3)."""
    leading_shape = tuple(rot_flat.shape[:-1])
    return xp.reshape(rot_flat, leading_shape + (2, 3, 3))


def unflatten_rotation_residuals(
    rot_residual_flat: Any,
    xp=jnp,
) -> Any:
    """Restore flattened (..., 6) residual bundles to (..., 2, 3)."""
    leading_shape = tuple(rot_residual_flat.shape[:-1])
    return xp.reshape(
        rot_residual_flat,
        leading_shape + (ROTATION_RESIDUAL_FEET, ROTATION_RESIDUAL_AXIS_DIM),
    )


def _safe_vector_norm(
    vec: Any,
    *,
    xp=jnp,
    axis: int = -1,
    keepdims: bool = False,
    eps: float = 1e-6,
) -> Any:
    """Stable vector norm that never returns zero."""
    squared = xp.sum(xp.square(vec), axis=axis, keepdims=keepdims)
    return xp.sqrt(xp.maximum(squared, eps))


def _normalize_vector(
    vec: Any,
    *,
    xp=jnp,
    axis: int = -1,
    eps: float = 1e-6,
) -> Any:
    """Normalize a vector with epsilon protection."""
    return vec / _safe_vector_norm(vec, xp=xp, axis=axis, keepdims=True, eps=eps)


def _vector_norm(
    vec: Any,
    *,
    xp=jnp,
    axis: int = -1,
    keepdims: bool = False,
) -> Any:
    """True vector norm without epsilon flooring."""
    squared = xp.sum(xp.square(vec), axis=axis, keepdims=keepdims)
    return xp.sqrt(squared)


def _broadcast_rotation_identity(
    leading_shape: Tuple[int, ...],
    *,
    dtype: Any,
    xp=jnp,
) -> Any:
    identity = xp.eye(3, dtype=dtype)
    return xp.broadcast_to(identity, leading_shape + (3, 3))


def _skew_symmetric(
    vec: Any,
    *,
    xp=jnp,
) -> Any:
    """Return the skew-symmetric matrix for vectors shaped (..., 3)."""
    zeros = xp.zeros_like(vec[..., 0])
    x = vec[..., 0]
    y = vec[..., 1]
    z = vec[..., 2]
    return xp.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        axis=-1,
    ).reshape(vec.shape[:-1] + (3, 3))


def bound_rotation_residual_axis_angles(
    raw_residual: Any,
    *,
    max_residual_deg: float,
    xp=jnp,
    eps: float = 1e-6,
) -> Any:
    """Bound residual axis-angle vectors so their magnitude saturates smoothly."""
    residual = xp.asarray(raw_residual)
    max_residual_deg = float(max(max_residual_deg, 0.0))
    if max_residual_deg == 0.0:
        return xp.zeros_like(residual)

    max_residual_rad = xp.asarray(np.deg2rad(max_residual_deg), dtype=residual.dtype)
    raw_norm = _vector_norm(residual, xp=xp, axis=-1, keepdims=True)
    bounded_norm = max_residual_rad * xp.tanh(raw_norm / max_residual_rad)
    scale = bounded_norm / xp.maximum(raw_norm, xp.asarray(eps, dtype=residual.dtype))
    return residual * scale


def axis_angle_to_rotation_matrices(
    axis_angle: Any,
    *,
    xp=jnp,
    eps: float = 1e-6,
) -> Any:
    """Convert axis-angle vectors (..., 3) to rotation matrices (..., 3, 3)."""
    axis_angle = xp.asarray(axis_angle)
    theta_sq = xp.sum(xp.square(axis_angle), axis=-1, keepdims=True)
    theta = xp.sqrt(theta_sq)
    eps_sq = xp.asarray(eps * eps, dtype=axis_angle.dtype)

    sin_over_theta = xp.where(
        theta_sq > eps_sq,
        xp.sin(theta) / xp.maximum(theta, xp.asarray(eps, dtype=axis_angle.dtype)),
        1.0 - (theta_sq / 6.0) + (theta_sq * theta_sq / 120.0),
    )
    one_minus_cos_over_theta_sq = xp.where(
        theta_sq > eps_sq,
        (1.0 - xp.cos(theta)) / xp.maximum(theta_sq, eps_sq),
        0.5 - (theta_sq / 24.0) + (theta_sq * theta_sq / 720.0),
    )

    skew = _skew_symmetric(axis_angle, xp=xp)
    skew_sq = xp.matmul(skew, skew)
    identity = _broadcast_rotation_identity(
        tuple(axis_angle.shape[:-1]),
        dtype=axis_angle.dtype,
        xp=xp,
    )
    return (
        identity
        + sin_over_theta[..., None] * skew
        + one_minus_cos_over_theta_sq[..., None] * skew_sq
    )


def rotation_matrices_to_axis_angle(
    rot: Any,
    *,
    xp=jnp,
    eps: float = 1e-6,
) -> Any:
    """Convert rotation matrices (..., 3, 3) to axis-angle vectors (..., 3)."""
    rot = project_rotation_matrices(xp.asarray(rot), xp=xp)
    skew_vec = xp.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        axis=-1,
    )
    sin_theta = 0.5 * _vector_norm(skew_vec, xp=xp, axis=-1, keepdims=True)
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    cos_theta = xp.clip((trace[..., None] - 1.0) * 0.5, -1.0, 1.0)
    theta = xp.arctan2(sin_theta, cos_theta)

    scale = theta / xp.maximum(2.0 * sin_theta, xp.asarray(eps, dtype=rot.dtype))
    axis_angle = scale * skew_vec
    small_angle = sin_theta <= xp.asarray(1e-4, dtype=rot.dtype)
    first_order = 0.5 * skew_vec
    return xp.where(small_angle, first_order, axis_angle)


def project_rotation_matrices(
    rot: Any,
    xp=jnp,
) -> Any:
    """Project arbitrary 3x3 matrices to proper rotations with stable Gram-Schmidt."""
    leading_shape = tuple(rot.shape[:-2])
    rot_flat = xp.reshape(rot, (-1, 3, 3)).astype(rot.dtype)
    rot_flat = xp.where(xp.isfinite(rot_flat), rot_flat, xp.zeros_like(rot_flat))

    col1 = rot_flat[..., :, 0]
    col2 = rot_flat[..., :, 1]
    col3 = rot_flat[..., :, 2]

    basis1 = _normalize_vector(col1, xp=xp, eps=1e-6)

    cand2 = col2 - xp.sum(basis1 * col2, axis=-1, keepdims=True) * basis1
    cand2_norm = _safe_vector_norm(cand2, xp=xp, axis=-1, keepdims=True, eps=1e-12)

    fallback2 = col3 - xp.sum(basis1 * col3, axis=-1, keepdims=True) * basis1
    fallback2_norm = _safe_vector_norm(fallback2, xp=xp, axis=-1, keepdims=True, eps=1e-12)

    x_axis = xp.broadcast_to(
        xp.asarray([1.0, 0.0, 0.0], dtype=rot_flat.dtype),
        basis1.shape,
    )
    y_axis = xp.broadcast_to(
        xp.asarray([0.0, 1.0, 0.0], dtype=rot_flat.dtype),
        basis1.shape,
    )
    arbitrary_axis = xp.where(xp.abs(basis1[..., 0:1]) < 0.9, x_axis, y_axis)
    arbitrary2 = arbitrary_axis - xp.sum(basis1 * arbitrary_axis, axis=-1, keepdims=True) * basis1

    use_fallback2 = cand2_norm <= 1e-6
    cand2 = xp.where(use_fallback2, fallback2, cand2)
    cand2_norm = xp.where(use_fallback2, fallback2_norm, cand2_norm)
    cand2 = xp.where(cand2_norm <= 1e-6, arbitrary2, cand2)

    basis2 = _normalize_vector(cand2, xp=xp, eps=1e-6)
    basis3 = xp.cross(basis1, basis2)
    basis3 = _normalize_vector(basis3, xp=xp, eps=1e-6)
    basis2 = xp.cross(basis3, basis1)
    basis2 = _normalize_vector(basis2, xp=xp, eps=1e-6)

    projected = xp.stack([basis1, basis2, basis3], axis=-1)
    return xp.reshape(projected, leading_shape + (3, 3))


def compose_residual_rotation_predictions(
    rotation_residual_flat: Any,
    rotation_input: Any,
    *,
    xp=jnp,
) -> Tuple[Any, Any, Any]:
    """Decode a residual rotation head and compose it with the noised rotation input."""
    rotation_input_phys = project_rotation_matrices(xp.asarray(rotation_input), xp=xp)
    rotation_residual_axis_angle = unflatten_rotation_residuals(rotation_residual_flat, xp=xp)
    rotation_delta = axis_angle_to_rotation_matrices(rotation_residual_axis_angle, xp=xp)
    rotation_pred_phys = xp.matmul(rotation_delta, rotation_input_phys)
    return rotation_residual_axis_angle, rotation_delta, rotation_pred_phys


def rotation_geodesic_angle(
    rotation_a: Any,
    rotation_b: Any,
    *,
    xp=jnp,
) -> Any:
    """Return the SO(3) geodesic angle between rotation bundles."""
    rotation_a = xp.asarray(rotation_a)
    rotation_b = xp.asarray(rotation_b)
    rotation_err = xp.matmul(rotation_a, xp.swapaxes(rotation_b, -1, -2))
    trace = (
        rotation_err[..., 0, 0]
        + rotation_err[..., 1, 1]
        + rotation_err[..., 2, 2]
    )
    cos_theta = xp.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    skew_vec = xp.stack(
        [
            rotation_err[..., 2, 1] - rotation_err[..., 1, 2],
            rotation_err[..., 0, 2] - rotation_err[..., 2, 0],
            rotation_err[..., 1, 0] - rotation_err[..., 0, 1],
        ],
        axis=-1,
    )
    sin_theta = 0.5 * _vector_norm(skew_vec, xp=xp, axis=-1)
    return xp.arctan2(sin_theta, cos_theta)


def masked_mean(
    values: Any,
    mask: Optional[Any],
    *,
    xp=jnp,
) -> Any:
    """Compute a masked mean with broadcasting over trailing dimensions."""
    values = xp.asarray(values)
    if mask is None:
        return xp.mean(values)

    weights = xp.asarray(mask, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights[..., None]
    weights = xp.broadcast_to(weights, values.shape)
    denom = xp.maximum(xp.sum(weights), xp.asarray(1.0, dtype=values.dtype))
    return xp.sum(values * weights) / denom


def geodesic_rotation_mse(
    rotation_a: Any,
    rotation_b: Any,
    supervision_mask: Optional[Any],
    *,
    xp=jnp,
) -> Any:
    """Masked mean squared geodesic angle between rotation bundles."""
    angle = rotation_geodesic_angle(rotation_a, rotation_b, xp=xp)
    return masked_mean(xp.square(angle), supervision_mask, xp=xp)


def _infer_qfrc_inverse_output_dim(
    batch: Optional[Dict[str, Any]] = None,
    normalizers: Optional[Dict[str, Any]] = None,
) -> int:
    if not AUXILIARY_MODEL_OUTPUTS_ENABLED:
        return 0
    if batch is not None and batch.get("qfrc_inverse_gt") is not None:
        return int(batch["qfrc_inverse_gt"].shape[-1])
    if normalizers is not None and "qfrc_inverse" in normalizers:
        return int(np.asarray(normalizers["qfrc_inverse"].mean).shape[-1])
    return 0


def _qfrc_inverse_phys_from_scaled(
    qfrc_inverse_scaled: Any,
    batch: Dict[str, Any],
    xp=jnp,
) -> Any:
    """Convert qfrc_inverse from Nm / (BW * H) back to physical Nm."""
    qfrc_scaled = xp.asarray(qfrc_inverse_scaled)
    norm_factor = batch.get("qfrc_inverse_norm_factor")
    if norm_factor is not None:
        return qfrc_scaled * xp.asarray(norm_factor, dtype=qfrc_scaled.dtype)

    static_context = batch.get("static_context")
    if static_context is None:
        return qfrc_scaled

    static_arr = xp.asarray(static_context, dtype=qfrc_scaled.dtype)
    if static_arr.ndim == 2 and static_arr.shape[-1] >= 2:
        height_m = static_arr[:, 0:1, None]
        mass_kg = static_arr[:, 1:2, None]
        return unnormalize_qfrc_inverse_by_bw_height(
            qfrc_scaled,
            mass_kg,
            height_m,
            xp=xp,
        )

    return qfrc_scaled


def _residual_z_from_difference(
    difference: Any,
    normalizer: Any,
    *,
    xp=jnp,
) -> Any:
    """Scale a residual by the task std without subtracting a mean."""
    diff = xp.asarray(difference)
    std = xp.asarray(normalizer.std, dtype=diff.dtype)
    return diff / std


def _decode_residual_prediction(
    residual_pred_z: Any,
    input_value: Any,
    normalizer: Any,
    *,
    xp=jnp,
) -> Any:
    """Decode a residual prediction back into a full signal in input units."""
    residual_pred_z = xp.asarray(residual_pred_z)
    input_value = xp.asarray(input_value, dtype=residual_pred_z.dtype)
    std = xp.asarray(normalizer.std, dtype=residual_pred_z.dtype)
    return input_value + residual_pred_z * std


def decode_cop_signal_to_length(
    cop_signal: Any,
    grf_ratio: Any,
    height_m: Any,
    *,
    use_grf_norm_cop: bool = False,
    contact_probability: Any = None,
    contact_threshold: float = 0.5,
    xp=jnp,
    eps: float = 1e-6,
) -> Any:
    """
    Convert the model COP signal to length units.

    Default COP signal is COP/height. With UseGRFNormCOP it is
    (COP/height) * (|GRF|/BW), so decode by dividing by each foot's
    bodyweight-normalized GRF magnitude before multiplying by height.
    """
    cop_arr = xp.asarray(cop_signal)
    h = xp.asarray(height_m, dtype=cop_arr.dtype)
    if not use_grf_norm_cop:
        return cop_arr * h

    grf_arr = xp.asarray(grf_ratio, dtype=cop_arr.dtype)
    eps_arr = xp.asarray(eps, dtype=cop_arr.dtype)
    mag_r_sq = xp.sum(xp.square(grf_arr[..., 0:3]), axis=-1, keepdims=True)
    mag_l_sq = xp.sum(xp.square(grf_arr[..., 3:6]), axis=-1, keepdims=True)
    mag_r = xp.sqrt(xp.maximum(mag_r_sq, eps_arr * eps_arr))
    mag_l = xp.sqrt(xp.maximum(mag_l_sq, eps_arr * eps_arr))
    decoded = xp.concatenate([
        cop_arr[..., 0:2] * h / mag_r,
        cop_arr[..., 2:4] * h / mag_l,
    ], axis=-1)
    if contact_probability is None:
        return decoded

    contact = xp.asarray(contact_probability, dtype=cop_arr.dtype)
    threshold = xp.asarray(contact_threshold, dtype=cop_arr.dtype)
    mask_r = (contact[..., 0:1] >= threshold).astype(cop_arr.dtype)
    mask_l = (contact[..., 1:2] >= threshold).astype(cop_arr.dtype)
    return xp.concatenate([
        decoded[..., 0:2] * mask_r,
        decoded[..., 2:4] * mask_l,
    ], axis=-1)


def _full_id_target_from_batch(
    batch: Dict[str, Any],
    xp=jnp,
) -> Any:
    """Return the clean full-ID target in physical Nm units."""
    if batch.get("id_gt_mjx") is not None:
        return xp.asarray(batch["id_gt_mjx"])
    if batch.get("qfrc_inverse_gt_raw") is not None:
        qfrc_inverse_gt_phys = xp.asarray(batch["qfrc_inverse_gt_raw"])
    else:
        qfrc_inverse_gt_phys = _qfrc_inverse_phys_from_scaled(
            batch["qfrc_inverse_gt"],
            batch,
            xp=xp,
        )
    return qfrc_inverse_gt_phys - xp.asarray(batch["qfrc_grf_contribution"])


def decode_auxiliary_predictions(
    pred: Any,
    batch: Dict[str, Any],
    normalizers: Dict[str, Any],
    *,
    xp=jnp,
) -> Tuple[Optional[Any], Any]:
    """Return preprocessed physics arrays used for torque/ID bookkeeping.

    The model no longer predicts qfrc_inverse, rotation, or Jacobian residuals.
    Those files remain loaded from preprocessing and are used directly by the
    torque loss, diagnostics, and best-model-by-torque selection.
    """
    if batch.get("qfrc_inverse_input_raw") is not None:
        qfrc_inverse_phys = xp.asarray(batch["qfrc_inverse_input_raw"])
    elif batch.get("qfrc_inverse_gt_raw") is not None:
        qfrc_inverse_phys = xp.asarray(batch["qfrc_inverse_gt_raw"])
    elif batch.get("qfrc_inverse_input") is not None:
        qfrc_inverse_phys = _qfrc_inverse_phys_from_scaled(
            batch["qfrc_inverse_input"],
            batch,
            xp=xp,
        )
    elif batch.get("qfrc_inverse_gt") is not None:
        qfrc_inverse_phys = _qfrc_inverse_phys_from_scaled(
            batch["qfrc_inverse_gt"],
            batch,
            xp=xp,
        )
    else:
        qfrc_inverse_phys = None

    rotation_phys = project_rotation_matrices(xp.asarray(batch["rot_w_to_ga"]), xp=xp)
    return qfrc_inverse_phys, rotation_phys


def split_model_predictions(
    pred: Any,
    qfrc_inverse_output_dim: int = 0,
    rotation_output_dim: int = ROTATION_OUTPUT_DIM,
) -> Tuple[Any, Any, Any, Any, Optional[Any], Optional[Any], Optional[Any]]:
    """Split model outputs into standard channels plus any legacy auxiliary heads."""
    cop_pred = pred[..., COP_SLICE]
    grf_pred = pred[..., GRF_SLICE]
    moments_pred = pred[..., MOMENTS_SLICE]
    contact_pred = pred[..., CONTACT_SLICE]
    offset = STANDARD_OUTPUT_DIM
    qfrc_inverse_pred = None
    if qfrc_inverse_output_dim > 0:
        qfrc_inverse_pred = pred[..., offset:offset + qfrc_inverse_output_dim]
        offset += qfrc_inverse_output_dim
    rotation_pred = None
    if rotation_output_dim > 0:
        rotation_pred = pred[..., offset:offset + rotation_output_dim]
        offset += rotation_output_dim
    jacobian_pred = None
    return (
        cop_pred,
        grf_pred,
        moments_pred,
        contact_pred,
        qfrc_inverse_pred,
        rotation_pred,
        jacobian_pred,
    )


def flatten_jacobian_components(
    jacp: Any,
    jacr: Any,
    xp=jnp,
) -> Any:
    """Flatten jacp/jacr to [..., 12 * nv] using [jacp_flat, jacr_flat] ordering."""
    leading_shape = tuple(jacp.shape[:-3])
    jacp_flat = xp.reshape(jacp, leading_shape + (-1,))
    jacr_flat = xp.reshape(jacr, leading_shape + (-1,))
    return xp.concatenate([jacp_flat, jacr_flat], axis=-1)


def unflatten_jacobian_components(
    jacobian_flat: Any,
    nv: int,
    xp=jnp,
) -> Tuple[Any, Any]:
    """Restore flattened [jacp_flat, jacr_flat] to (..., 2, 3, nv) tensors."""
    leading_shape = tuple(jacobian_flat.shape[:-1])
    jacobian_block_dim = 2 * 3 * int(nv)
    jacp = xp.reshape(
        jacobian_flat[..., :jacobian_block_dim],
        leading_shape + (2, 3, int(nv)),
    )
    jacr = xp.reshape(
        jacobian_flat[..., jacobian_block_dim:2 * jacobian_block_dim],
        leading_shape + (2, 3, int(nv)),
    )
    return jacp, jacr


def select_torque_jacobians(
    pred: Any,
    batch: Dict[str, Any],
    normalizers: Dict[str, Any],
    xp=jnp,
) -> Tuple[Any, Any]:
    """Use preprocessed Jacobians for torque computation."""
    return batch["jacp"], batch["jacr"]


# =============================================================================
# Model Architecture 
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        position = jnp.arange(seq_len)
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = position[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return x + emb[None, :, :]


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x, x, deterministic=not train)
        x = residual + attn_out
        
        residual = x
        x = nn.LayerNorm()(x)
        ff_out = nn.Dense(self.ff_dim)(x)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
        x = residual + ff_out
        
        return x


class KinematicsToCOPGRFMoments(nn.Module):
    """Transformer-based model for gait dynamics prediction.
    
    Inputs:
        - Temporal feature vector (constructed in data_loader.py)
          Includes kinematics, reconstructed COP/GRF/GRM, and optional flattened Jacobians.
          NOTE: contactBoolean is NO LONGER an input; the model predicts it as output.
        - Static token:
          [height, mass, gender, PatientSize(4), forwardVel]
    
    Outputs:
        - COP (4): [rx, rz, lx, lz] in ground-aligned calc frame - Unit: m/h  (contact-masked)
        - GRF (6): [rx, ry, rz, lx, ly, lz] - Unit: N/m*9.806                 (contact-masked)
        - Moments (2): [rz, lz] - Unit: Nm/m*h*9.806
        - ContactBoolean (2): [right, left] - soft sigmoid, hard-thresholded for masking
    """
    input_dim: int = 54
    static_dim: int = 8 # height, mass, gender, PatientSize(4), forwardVel
    output_dim: int = STANDARD_OUTPUT_DIM
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # 1. Project temporal inputs directly into the transformer width.
        x = nn.Dense(self.d_model)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        # Positional Encoding
        x = SinusoidalPosEmb(dim=self.d_model)(x)

        # 2. Static Branch: MLP Layer
        s = nn.Dense(self.d_model)(static_context)
        s = nn.gelu(s)
        s = nn.LayerNorm()(s)

        # 3. Prepend Static Token
        s = jnp.expand_dims(s, axis=1)
        x = jnp.concatenate([s, x], axis=1)  # (batch, seq_len + 1, d_model)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        # 4. Remove Static Token
        x = x[:, 1:, :]
        x = nn.LayerNorm()(x)

        # 5. Predict the standard 14 outputs from the shared backbone.
        raw_out = nn.Dense(self.output_dim)(x)  # (batch, seq, 14)

        # 6. Predict contact probabilities (sigmoid)
        contact_logits = raw_out[..., CONTACT_SLICE]          # (batch, seq, 2) — right, left
        contact_prob   = nn.sigmoid(contact_logits)   # soft, used for BCE loss

        cop_raw = raw_out[..., COP_SLICE]    # [rx, rz, lx, lz]
        grf_raw = raw_out[..., GRF_SLICE]   # [rx, ry, rz, lx, ly, lz]
        mom_raw = raw_out[..., MOMENTS_SLICE]  # [rz, lz]

        # 8. Concatenate final raw (normalized) output: COP(4) + GRF(6) + Moments(2) + ContactProb(2)
        #    Contact hard-masking is now applied in physical space within compute_total_loss
        out = jnp.concatenate([cop_raw, grf_raw, mom_raw, contact_prob], axis=-1)
        return out


# =============================================================================
# Physics: Compute τ_grf from predictions using Jacobian
# =============================================================================

def compute_full_external_moments(
    cop_pred_unnorm: jnp.ndarray,  # (batch, seq, 4) [rx, rz, lx, lz] in ground-aligned calc frame
    grf_pred_unnorm: jnp.ndarray,  # (batch, seq, 6)
    free_moments_pred_unnorm: jnp.ndarray,  # (batch, seq, 2) [rz, lz]
    ankle_heights: jnp.ndarray,  # (batch, seq, 2) [right, left]
    rot_w_to_ga: jnp.ndarray,  # (batch, seq, 2, 3, 3) world->ground-aligned calc rotation
) -> jnp.ndarray:
    """
    Compute full external moment about each foot origin using COP, GRF, and free moments.
    """
    # Predicted COP channels are [X, Z] in the ground-aligned calc frame.
    # Build 3D ground-aligned vectors by inserting ankle height as Y.
    cop_r_ga = jnp.concatenate(
        [cop_pred_unnorm[..., 0:1], ankle_heights[..., 0:1], cop_pred_unnorm[..., 1:2]],
        axis=-1
    )
    cop_l_ga = jnp.concatenate(
        [cop_pred_unnorm[..., 2:3], ankle_heights[..., 1:2], cop_pred_unnorm[..., 3:4]],
        axis=-1
    )

    # Rotate ground-aligned COP vectors back to world:
    # R_ga->w = (R_w->ga)^T
    rot_w_to_ga_r = rot_w_to_ga[:, :, 0]  # (batch, seq, 3, 3)
    rot_w_to_ga_l = rot_w_to_ga[:, :, 1]
    rot_ga_to_w_r = jnp.swapaxes(rot_w_to_ga_r, -1, -2)
    rot_ga_to_w_l = jnp.swapaxes(rot_w_to_ga_l, -1, -2)
    cop_r = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_r, cop_r_ga)
    cop_l = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_l, cop_l_ga)

    grf_r = grf_pred_unnorm[..., :3]
    grf_l = grf_pred_unnorm[..., 3:6]
    
    # Reconstruct 3D moments from 1D predictions (assume Mx=My=0)
    mz_r = free_moments_pred_unnorm[..., 0:1]
    mz_l = free_moments_pred_unnorm[..., 1:2]
    
    mom_r = jnp.concatenate([jnp.zeros_like(mz_r), jnp.zeros_like(mz_r), mz_r], axis=-1)
    mom_l = jnp.concatenate([jnp.zeros_like(mz_l), jnp.zeros_like(mz_l), mz_l], axis=-1)

    # Cross product: r x F
    # r is the vector from the point of force application (COP) to the moment reference point.
    # Usually M_total = M_free + (r x F)
    # Here r is COP relative to ankle, expressed in world coordinates.
    
    m_r_induced = jnp.cross(cop_r, grf_r)
    m_l_induced = jnp.cross(cop_l, grf_l)
    
    m_r_total = m_r_induced + mom_r
    m_l_total = m_l_induced + mom_l
    
    return jnp.concatenate([m_r_total, m_l_total], axis=-1)


def compute_tau_grf_from_predictions(
    grf_pred: jnp.ndarray,  # (batch, seq, 6) [right_xyz, left_xyz]
    moments_pred: jnp.ndarray,  # (batch, seq, 6) - full moments computed from COP/GRF/free moment
    jacp: jnp.ndarray,  # (batch, seq, 2, 3, 39)
    jacr: jnp.ndarray,  # (batch, seq, 2, 3, 39)
) -> jnp.ndarray:
    """
    Compute τ_grf = Jp^T @ GRF + Jr^T @ M for each timestep.
    
    Args:
        grf_pred: Predicted GRF [right_xyz, left_xyz]
        moments_pred: Full external moments [right_xyz, left_xyz]
        jacp: Position Jacobian (batch, seq, 2 bodies, 3 spatial, 39 dofs)
        jacr: Rotation Jacobian
    
    Returns:
        tau_grf: Joint torques from GRF (batch, seq, 39)
    """
    # Split into right/left
    grf_r = grf_pred[..., :3]  # (batch, seq, 3)
    grf_l = grf_pred[..., 3:]  # (batch, seq, 3)
    moment_r = moments_pred[..., :3]  # (batch, seq, 3)
    moment_l = moments_pred[..., 3:]  # (batch, seq, 3)
    
    # Jp^T @ F: need to einsum over spatial dimension
    # jacp shape: (batch, seq, 2, 3, 39)
    # force shape: (batch, seq, 3)
    
    # For right foot (body 0? Check BatchDataProcessing): 
    # In BatchDataProcessing: External_Force = External_Force.at[calcn_l_id, ...].set(...)
    # jacobian_data['jacp'] has shape (T, 2, 3, nv). 
    # Usually index 0 is right, index 1 is left based on how it was saved?
    # Let's check BatchDataProcessing.py saving order.
    # "jacobian_data['jacp'] = np.stack([jacp_r, jacp_l], axis=1)" -> So 0 is Right, 1 is Left.
    
    tau_p_r = jnp.einsum('bsij,bsi->bsj', jacp[:, :, 0], grf_r)  # (batch, seq, 39)
    tau_p_l = jnp.einsum('bsij,bsi->bsj', jacp[:, :, 1], grf_l)  # (batch, seq, 39)
    
    # Jr^T @ M
    tau_r_r = jnp.einsum('bsij,bsi->bsj', jacr[:, :, 0], moment_r)  # (batch, seq, 39)
    tau_r_l = jnp.einsum('bsij,bsi->bsj', jacr[:, :, 1], moment_l)  # (batch, seq, 39)
    
    tau_grf = tau_p_r + tau_p_l + tau_r_r + tau_r_l
    
    return tau_grf


# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(pred: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray = 1.0) -> jnp.ndarray:
    """Compute weighted Mean Squared Error."""
    return jnp.mean(weights * jnp.square(pred - target))

def m3e_loss(pred: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray = 1.0) -> jnp.ndarray:
    "Mean Absolute Cubed Error"
    return jnp.sum((jnp.abs(pred - target) ** 3) * weights)

def compute_total_loss(
    pred: jnp.ndarray,
    batch: Dict[str, jnp.ndarray],
    normalizers: Dict,
    loss_weights: Dict[str, float],
    use_contact_weighting: bool,
    magOnOff: bool,
    contactOnOff: bool,
    only_supervise_stance: bool,
    contact_weight_multiplier: float = 1.5,
    magWeight: float = 3.0,
    dof_weights_dict: Optional[Dict[int, float]] = None,  # New parameter for DOF weights
    epoch: float = 1.0,
    total_epochs: float = 1.0,
    cop_mask: bool = True,
    use_grf_norm_cop: bool = False,
    use_gt_jacob_and_rot: bool = False,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute direct COP/GRF/moment/contact losses plus torque supervision."""
    cop_pred, grf_pred, moments_pred, contact_pred, qfrc_inverse_pred, rotation_pred, jacobian_pred = split_model_predictions(
        pred,
        qfrc_inverse_output_dim=0,
        rotation_output_dim=0,
    )

    # --- Contact Weighting ---
    contact_bool = batch["contactBoolean"]
    supervision_mask = batch.get("supervision_mask", None)
    if supervision_mask is None:
        supervision_mask = jnp.ones(contact_bool.shape[:-1] + (1,), dtype=pred.dtype)
    else:
        if supervision_mask.ndim == 2:
            supervision_mask = supervision_mask[..., None]
        supervision_mask = supervision_mask.astype(pred.dtype)

    # --- Per-window speed/gender balancing weight ---
    # Each window carries a scalar sample_weight (1.0 when balancing is disabled).
    # Folding it into the supervision mask scales that window's contribution to
    # every downstream loss component, yielding a weighted average across the batch.
    sample_weight = batch.get("sample_weight", None)
    if sample_weight is not None:
        sample_weight = sample_weight.astype(pred.dtype).reshape((-1,) + (1,) * (supervision_mask.ndim - 1))
        supervision_mask = supervision_mask * sample_weight

    contact_r = contact_bool[..., 0:1] # (batch, seq, 1)
    contact_l = contact_bool[..., 1:2] # (batch, seq, 1)
    
    # Create weight masks
    weight_r = 1.0 + (contact_weight_multiplier - 1.0) * contact_r
    weight_l = 1.0 + (contact_weight_multiplier - 1.0) * contact_l
    
    if not use_contact_weighting:
        weight_r = jnp.ones_like(weight_r)
        weight_l = jnp.ones_like(weight_l)

    cop_weights = jnp.concatenate([
        jnp.tile(weight_r, (1, 1, 2)),
        jnp.tile(weight_l, (1, 1, 2)),
    ], axis=-1) * supervision_mask
    grf_weights = jnp.concatenate([
        jnp.tile(weight_r, (1, 1, 3)),
        jnp.tile(weight_l, (1, 1, 3)),
    ], axis=-1) * supervision_mask
    moments_weights = jnp.concatenate([
        jnp.tile(weight_r, (1, 1, 1)),
        jnp.tile(weight_l, (1, 1, 1)),
    ], axis=-1) * supervision_mask

    output_mask_r = (contact_r > 0.5).astype(pred.dtype)
    output_mask_l = (contact_l > 0.5).astype(pred.dtype)
    if cop_mask:
        cop_weights = cop_weights * jnp.concatenate([
            jnp.tile(output_mask_r, (1, 1, 2)),
            jnp.tile(output_mask_l, (1, 1, 2)),
        ], axis=-1)
        grf_weights = grf_weights * jnp.concatenate([
            jnp.tile(output_mask_r, (1, 1, 3)),
            jnp.tile(output_mask_l, (1, 1, 3)),
        ], axis=-1)
        moments_weights = moments_weights * jnp.concatenate([
            output_mask_r,
            output_mask_l,
        ], axis=-1)

    cop_pred_abs = cop_pred
    grf_pred_abs = grf_pred
    moments_pred_abs = moments_pred

    if cop_mask:
        cop_abs_unnorm = normalizers["cop"].unnormalize(cop_pred_abs)
        grf_abs_unnorm = normalizers["grf"].unnormalize(grf_pred_abs)
        mom_abs_unnorm = normalizers["moments"].unnormalize(moments_pred_abs)

        cop_abs_unnorm_masked = jnp.concatenate([cop_abs_unnorm[..., 0:2] * output_mask_r, cop_abs_unnorm[..., 2:4] * output_mask_l], axis=-1)
        grf_abs_unnorm_masked = jnp.concatenate([grf_abs_unnorm[..., 0:3] * output_mask_r, grf_abs_unnorm[..., 3:6] * output_mask_l], axis=-1)
        mom_abs_unnorm_masked = jnp.concatenate([mom_abs_unnorm[..., 0:1] * output_mask_r, mom_abs_unnorm[..., 1:2] * output_mask_l], axis=-1)

        cop_pred_abs = normalizers["cop"].normalize(cop_abs_unnorm_masked)
        grf_pred_abs = normalizers["grf"].normalize(grf_abs_unnorm_masked)
        moments_pred_abs = normalizers["moments"].normalize(mom_abs_unnorm_masked)

    loss_type = "mse"
    if loss_type == "mse":
        cop_loss = mse_loss(cop_pred_abs, batch["cop"], cop_weights)
        grf_loss = mse_loss(grf_pred_abs, batch["grf"], grf_weights)
        moments_loss = mse_loss(moments_pred_abs, batch["moments"], moments_weights)
    elif loss_type == "m3e":
        cop_target_z = normalizers["cop"].normalize(batch["cop"])
        grf_target_z = normalizers["grf"].normalize(batch["grf"])
        moments_target_z = normalizers["moments"].normalize(batch["moments"])
        cop_loss = m3e_loss(cop_pred_abs, cop_target_z, cop_weights)
        grf_loss = m3e_loss(grf_pred_abs, grf_target_z, grf_weights)
        moments_loss = m3e_loss(moments_pred_abs, moments_target_z, moments_weights)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    jacobian_loss = jnp.zeros_like(cop_loss)
    jacobian_input_reg_loss = jnp.zeros_like(cop_loss)

    qfrc_inverse_loss = jnp.zeros_like(cop_loss)
    if qfrc_inverse_pred is not None and "qfrc_inverse" in normalizers and batch.get("qfrc_inverse_gt") is not None:
        qfrc_inverse_target_residual_z = _residual_z_from_difference(
            batch["qfrc_inverse_gt"] - batch["qfrc_inverse_input"],
            normalizers["qfrc_inverse"],
            xp=jnp,
        )
        qfrc_inverse_loss = mse_loss(qfrc_inverse_pred, qfrc_inverse_target_residual_z, supervision_mask)

    qfrc_inverse_input_reg_loss = jnp.zeros_like(cop_loss)
    if qfrc_inverse_pred is not None and "qfrc_inverse" in normalizers and batch.get("qfrc_inverse_input") is not None:
        qfrc_inverse_zero_residual_z = jnp.zeros_like(qfrc_inverse_pred)
        qfrc_inverse_input_reg_loss = mse_loss(qfrc_inverse_pred, qfrc_inverse_zero_residual_z, supervision_mask)

    rotation_loss = jnp.zeros_like(cop_loss)
    rotation_pred_phys = project_rotation_matrices(
        jnp.asarray(batch["rot_w_to_ga"], dtype=pred.dtype),
        xp=jnp,
    )
    rotation_delta = None
    rotation_residual_axis_angle = None
    if rotation_pred is not None and batch.get("rot_w_to_ga") is not None:
        rotation_residual_axis_angle, rotation_delta, rotation_pred_phys = compose_residual_rotation_predictions(
            rotation_pred,
            batch["rot_w_to_ga"],
            xp=jnp,
        )
    if batch.get("gt_rot_w_to_ga") is not None:
        rotation_target_phys = project_rotation_matrices(
            jnp.asarray(batch["gt_rot_w_to_ga"], dtype=rotation_pred_phys.dtype),
            xp=jnp,
        )
        if rotation_residual_axis_angle is not None:
            rotation_target_delta = jnp.matmul(
                rotation_target_phys,
                jnp.swapaxes(
                    project_rotation_matrices(jnp.asarray(batch["rot_w_to_ga"], dtype=rotation_pred_phys.dtype), xp=jnp),
                    -1,
                    -2,
                ),
            )
            rotation_target_residual = rotation_matrices_to_axis_angle(rotation_target_delta, xp=jnp)
            rotation_loss = masked_mean(
                jnp.square(rotation_residual_axis_angle - rotation_target_residual),
                supervision_mask,
                xp=jnp,
            )

    rotation_input_reg_loss = jnp.zeros_like(cop_loss)
    if rotation_residual_axis_angle is not None:
        rotation_input_reg_loss = masked_mean(
            jnp.square(rotation_residual_axis_angle),
            supervision_mask,
            xp=jnp,
        )

    eps = 1e-7
    contact_pred_clipped = jnp.clip(contact_pred, eps, 1.0 - eps)
    contact_bce = -(
        contact_bool * jnp.log(contact_pred_clipped) +
        (1.0 - contact_bool) * jnp.log(1.0 - contact_pred_clipped)
    )
    contact_loss = jnp.mean(contact_bce * supervision_mask)

    output_reg_loss = jnp.zeros_like(cop_loss)

    static_unnorm = normalizers["static"].unnormalize(batch["static_context"])
    height_m = static_unnorm[:, 0:1, None]
    mass_kg = static_unnorm[:, 1:2, None]

    cop_pred_phys = normalizers["cop"].unnormalize(cop_pred_abs)
    grf_pred_phys = normalizers["grf"].unnormalize(grf_pred_abs)
    moments_pred_phys = normalizers["moments"].unnormalize(moments_pred_abs)

    if cop_mask:
        mask_r_t = output_mask_r
        mask_l_t = output_mask_l
        cop_pred_phys = jnp.concatenate([
            cop_pred_phys[..., 0:2] * mask_r_t,
            cop_pred_phys[..., 2:4] * mask_l_t,
        ], axis=-1)
        grf_pred_phys = jnp.concatenate([
            grf_pred_phys[..., 0:3] * mask_r_t,
            grf_pred_phys[..., 3:6] * mask_l_t,
        ], axis=-1)
        moments_pred_phys = jnp.concatenate([
            moments_pred_phys[..., 0:1] * mask_r_t,
            moments_pred_phys[..., 1:2] * mask_l_t,
        ], axis=-1)

    grf_unnorm = grf_pred_phys * (mass_kg * 9.8067)
    cop_unnorm = decode_cop_signal_to_length(
        cop_pred_phys,
        grf_pred_phys,
        height_m,
        use_grf_norm_cop=use_grf_norm_cop,
        contact_probability=contact_pred if use_grf_norm_cop else None,
        xp=jnp,
    )
    moments_unnorm = moments_pred_phys * (mass_kg * 9.8067 * height_m)
    grf_pred_abs = grf_pred_phys

    torque_cop_unnorm = cop_unnorm
    torque_grf_unnorm = grf_unnorm
    torque_moments_unnorm = moments_unnorm
    # When use_gt_jacob_and_rot is set, the predicted COP is taken to world with the
    # ground-truth (MoCap) rotation and the wrench is mapped to joint torques with the
    # ground-truth (MoCap) Jacobian, instead of the video (ProcessedData) terms. The
    # model still consumes video inputs; only the torque reconstruction uses GT kinematics.
    if use_gt_jacob_and_rot:
        torque_rotation_pred_phys = project_rotation_matrices(
            jnp.asarray(batch["gt_rot_w_to_ga"], dtype=pred.dtype), xp=jnp
        )
    else:
        torque_rotation_pred_phys = rotation_pred_phys
    full_moments = compute_full_external_moments(
        torque_cop_unnorm,
        torque_grf_unnorm,
        torque_moments_unnorm,
        batch["ankle_heights"],
        torque_rotation_pred_phys,
    )

    if use_gt_jacob_and_rot:
        jacp_for_tau = jnp.asarray(batch["gt_jacp"], dtype=pred.dtype)
        jacr_for_tau = jnp.asarray(batch["gt_jacr"], dtype=pred.dtype)
    else:
        jacp_for_tau, jacr_for_tau = select_torque_jacobians(
            pred,
            batch,
            normalizers,
            xp=jnp,
        )
    tau_grf_pred = compute_tau_grf_from_predictions(
        torque_grf_unnorm, full_moments, jacp_for_tau, jacr_for_tau
    )
    target_tau_grf = jnp.asarray(batch["qfrc_grf_contribution"], dtype=pred.dtype)
    nv = int(target_tau_grf.shape[-1])
    cop_target_phys = normalizers["cop"].unnormalize(batch["cop"])
    grf_target_phys = normalizers["grf"].unnormalize(batch["grf"])
    moments_target_phys = normalizers["moments"].unnormalize(batch["moments"])
    target_grf_unnorm = grf_target_phys * (mass_kg * 9.8067)
    target_cop_unnorm = decode_cop_signal_to_length(
        cop_target_phys,
        grf_target_phys,
        height_m,
        use_grf_norm_cop=use_grf_norm_cop,
        xp=jnp,
    )
    target_moments_unnorm = moments_target_phys * (mass_kg * 9.8067 * height_m)

    active_indices = np.array([
        6, 7, 9, 10, 11,
        13, 14, 16, 17, 18,
        20, 21, 22,
    ], dtype=np.int32)
    active_indices = active_indices[active_indices < nv]
    active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
    torque_mask = jnp.zeros((nv,))
    torque_mask = torque_mask.at[active_indices].set(1.0)
    torque_mask = torque_mask[None, None, :]

    norm_factor = mass_kg * 9.8067 * height_m
    target_norm = target_tau_grf / norm_factor
    pred_norm = tau_grf_pred / norm_factor
    torque_diff = pred_norm - target_norm
    raw_error = jnp.square(torque_diff)

    contact_multiplier = jnp.ones((1, 1, nv))
    if use_contact_weighting:
        right_indices = np.array([6, 7, 8, 9, 10, 11, 12], dtype=np.int32)
        left_indices = np.array([13, 14, 15, 16, 17, 18, 19], dtype=np.int32)
        right_indices = jnp.asarray(right_indices[right_indices < nv], dtype=jnp.int32)
        left_indices = jnp.asarray(left_indices[left_indices < nv], dtype=jnp.int32)
        right_mask = jnp.zeros((nv,))
        right_mask = right_mask.at[right_indices].set(1.0)
        right_mask = right_mask[None, None, :]
        left_mask = jnp.zeros((nv,))
        left_mask = left_mask.at[left_indices].set(1.0)
        left_mask = left_mask[None, None, :]
        contact_multiplier = (
            1.0 * (1.0 - right_mask - left_mask) +
            weight_r * right_mask +
            weight_l * left_mask
        )

    magnitude_multiplier = jnp.abs(target_norm) / magWeight + 0.5

    if dof_weights_dict is None:
        dof_weights_dict = {
            6: 1.0,
            7: 1.0,
            9: 1.0,
            10: 1.0,
            11: 1.0,
            13: 1.0,
            14: 1.0,
            16: 1.0,
            17: 1.0,
            18: 1.0,
            20: 1.0,
            21: 1.0,
            22: 1.0,
        }

    individual_multiplier = jnp.ones((nv,))
    valid_weight_items = [
        (idx, val) for idx, val in dof_weights_dict.items()
        if int(idx) < nv
    ]
    if valid_weight_items:
        ind_indices = jnp.asarray([idx for idx, _val in valid_weight_items], dtype=jnp.int32)
        ind_values = jnp.asarray([val for _idx, val in valid_weight_items], dtype=pred.dtype)
        individual_multiplier = individual_multiplier.at[ind_indices].set(ind_values)
    individual_multiplier = individual_multiplier[None, None, :]

    if not magOnOff:
        magnitude_multiplier = jnp.ones_like(magnitude_multiplier)
    if not contactOnOff:
        contact_multiplier = jnp.ones_like(contact_multiplier)

    raw_weights = contact_multiplier * magnitude_multiplier * individual_multiplier
    raw_weights = raw_weights * torque_mask
    final_weights = raw_weights * supervision_mask
    num_active_dof = jnp.sum(torque_mask)

    def _weighted_torque_loss_from_prediction(tau_grf_pred_local: jnp.ndarray) -> jnp.ndarray:
        pred_norm_local = tau_grf_pred_local / norm_factor
        raw_error_local = jnp.square(pred_norm_local - target_norm)
        weighted_error_local = raw_error_local * final_weights
        return jnp.sum(weighted_error_local) / jnp.maximum(num_active_dof, 1.0)

    torque_loss = _weighted_torque_loss_from_prediction(tau_grf_pred)

    def _branch_outputs_to_unnorm(
        cop_pred_abs_local: jnp.ndarray,
        grf_pred_abs_local: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cop_pred_phys_local = normalizers["cop"].unnormalize(cop_pred_abs_local)
        grf_pred_phys_local = normalizers["grf"].unnormalize(grf_pred_abs_local)
        if cop_mask:
            cop_pred_phys_local = jnp.concatenate([
                cop_pred_phys_local[..., 0:2] * mask_r_t,
                cop_pred_phys_local[..., 2:4] * mask_l_t,
            ], axis=-1)
            grf_pred_phys_local = jnp.concatenate([
                grf_pred_phys_local[..., 0:3] * mask_r_t,
                grf_pred_phys_local[..., 3:6] * mask_l_t,
            ], axis=-1)
        return (
            decode_cop_signal_to_length(
                cop_pred_phys_local,
                grf_pred_phys_local,
                height_m,
                use_grf_norm_cop=use_grf_norm_cop,
                contact_probability=contact_pred if use_grf_norm_cop else None,
                xp=jnp,
            ),
            grf_pred_phys_local * (mass_kg * 9.8067),
        )

    def _torque_loss_from_branch_outputs(
        cop_pred_abs_local: jnp.ndarray,
        grf_pred_abs_local: jnp.ndarray,
    ) -> jnp.ndarray:
        cop_unnorm_local, grf_unnorm_local = _branch_outputs_to_unnorm(
            cop_pred_abs_local,
            grf_pred_abs_local,
        )
        full_moments_local = compute_full_external_moments(
            cop_unnorm_local,
            grf_unnorm_local,
            torque_moments_unnorm,
            batch["ankle_heights"],
            torque_rotation_pred_phys,
        )
        tau_grf_local = compute_tau_grf_from_predictions(
            grf_unnorm_local,
            full_moments_local,
            jacp_for_tau,
            jacr_for_tau,
        )
        return _weighted_torque_loss_from_prediction(tau_grf_local)

    def _cop_direct_loss_from_pred(cop_pred_abs_local: jnp.ndarray) -> jnp.ndarray:
        if loss_type == "mse":
            return mse_loss(cop_pred_abs_local, batch["cop"], cop_weights) / 4
        return m3e_loss(cop_pred_abs_local, cop_target_z, cop_weights) / 4

    def _grf_direct_loss_from_pred(grf_pred_abs_local: jnp.ndarray) -> jnp.ndarray:
        if loss_type == "mse":
            return mse_loss(grf_pred_abs_local, batch["grf"], grf_weights) / 6
        return m3e_loss(grf_pred_abs_local, grf_target_z, grf_weights) / 6

    def _grad_rms(grad_tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.mean(jnp.square(grad_tensor)))

    torque_cop_effect_loss = jnp.zeros_like(torque_loss)
    torque_grf_effect_loss = jnp.zeros_like(torque_loss)
    grad_eps = jnp.asarray(1e-8, dtype=pred.dtype)

    torque_cop_grad = jax.grad(
        lambda cop_pred_abs_local: _torque_loss_from_branch_outputs(
            cop_pred_abs_local,
            jax.lax.stop_gradient(grf_pred_abs),
        )
    )(cop_pred_abs)
    direct_cop_grad = jax.grad(_cop_direct_loss_from_pred)(cop_pred_abs)
    torque_cop_effect_loss = cop_loss * (
        _grad_rms(torque_cop_grad) / jnp.maximum(_grad_rms(direct_cop_grad), grad_eps)
    )

    torque_grf_grad = jax.grad(
        lambda grf_pred_abs_local: _torque_loss_from_branch_outputs(
            jax.lax.stop_gradient(cop_pred_abs),
            grf_pred_abs_local,
        )
    )(grf_pred_abs)
    direct_grf_grad = jax.grad(_grf_direct_loss_from_pred)(grf_pred_abs)
    torque_grf_effect_loss = grf_loss * (
        _grad_rms(torque_grf_grad) / jnp.maximum(_grad_rms(direct_grf_grad), grad_eps)
    )

    com_accel = batch["com_accel"]
    pred_fx = (grf_pred_abs[..., 0] + grf_pred_abs[..., 3]) * mass_kg * 9.8067
    res_x = mass_kg * com_accel[..., 0] - pred_fx
    pred_fy = (grf_pred_abs[..., 1] + grf_pred_abs[..., 4]) * mass_kg * 9.8067
    res_y = mass_kg * com_accel[..., 1] - pred_fy
    pred_fz = (grf_pred_abs[..., 2] + grf_pred_abs[..., 5]) * mass_kg * 9.8067
    gravity = 9.8067
    res_z = mass_kg * (com_accel[..., 2] + gravity) - pred_fz
    grf_res = jnp.stack([res_x, res_y, res_z], axis=-1)
    if "grf_res" in normalizers:
        grf_res_norm = normalizers["grf_res"].normalize(grf_res)
        grf_correction_loss = jnp.mean(jnp.square(grf_res_norm) * supervision_mask)
    else:
        grf_correction_loss = jnp.mean(jnp.square(grf_res) * supervision_mask)

    cop_loss = cop_loss / 4
    grf_loss = grf_loss / 6
    moments_loss = moments_loss / 2
    qfrc_inverse_loss = jnp.zeros_like(cop_loss)
    qfrc_inverse_input_reg_loss = jnp.zeros_like(cop_loss)
    rotation_loss = jnp.zeros_like(cop_loss)
    rotation_input_reg_loss = jnp.zeros_like(cop_loss)
    jacobian_loss = jnp.zeros_like(cop_loss)
    jacobian_input_reg_loss = jnp.zeros_like(cop_loss)
    contact_loss = contact_loss / 2
    grf_correction_loss = grf_correction_loss / 3

    epoch_f = jnp.asarray(epoch, dtype=jnp.float32)
    total_epochs_f = jnp.maximum(jnp.asarray(total_epochs, dtype=jnp.float32), 1.0)
    output_reg_multiplier = jnp.clip(1.0 - (epoch_f / total_epochs_f), 0.0, 1.0)
    total_loss = (
        loss_weights.get("cop", 1.0) * cop_loss +
        loss_weights.get("grf", 1.0) * grf_loss +
        loss_weights.get("moments", 1.0) * moments_loss +
        loss_weights.get("contact", 1.0) * contact_loss +
        loss_weights.get("torque", 1.0) * torque_loss +
        loss_weights.get("grf_correction", 1.0) * grf_correction_loss
    )
    
    return total_loss, {
        "cop_loss": cop_loss,
        "grf_loss": grf_loss,
        "moments_loss": moments_loss,
        "qfrc_inverse_loss": qfrc_inverse_loss,
        "qfrc_inverse_input_reg_loss": qfrc_inverse_input_reg_loss,
        "rotation_loss": rotation_loss,
        "rotation_input_reg_loss": rotation_input_reg_loss,
        "jacobian_loss": jacobian_loss,
        "jacobian_input_reg_loss": jacobian_input_reg_loss,
        "contact_loss": contact_loss,
        "torque_loss": torque_loss,
        "torque_cop_effect_loss": torque_cop_effect_loss,
        "torque_grf_effect_loss": torque_grf_effect_loss,
        "grf_correction_loss": grf_correction_loss,
        "output_reg_loss": output_reg_loss,
        "total_loss": total_loss,
    }


# =============================================================================
# Training Functions
# =============================================================================

def create_train_state(rng, model, input_shape, static_shape, learning_rate=1e-4, weight_decay=0.01):
    dummy_input = jnp.ones(input_shape)
    dummy_static = jnp.ones(static_shape)
    params = model.init(rng, dummy_input, dummy_static, train=False)["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(normalizers: Dict, use_contact_weighting: bool, magOnOff: bool, contactOnOff: bool, only_supervise_stance: bool, contact_weight_multiplier: float, magWeight: float, total_epochs: int, dof_weights_dict: Dict = None, cop_mask: bool = True, use_grf_norm_cop: bool = False, use_gt_jacob_and_rot: bool = False):
    def _tree_any_nonfinite(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return jnp.array(False)
        flags = [jnp.any(~jnp.isfinite(jnp.asarray(leaf))) for leaf in leaves]
        return jnp.any(jnp.stack(flags))

    def _sanitize_tree(tree):
        return jax.tree_util.tree_map(
            lambda leaf: jnp.where(
                jnp.isfinite(jnp.asarray(leaf)),
                jnp.asarray(leaf),
                jnp.zeros_like(jnp.asarray(leaf)),
            ),
            tree,
        )

    @jax.jit
    def train_step(state, batch, loss_weights, dropout_rng, epoch):
        def loss_fn(params):
            pred = state.apply_fn(
                {"params": params},
                batch["input"],
                batch["static_context"],
                train=True,
                rngs={"dropout": dropout_rng}
            )
            loss, metrics = compute_total_loss(pred, batch, normalizers, loss_weights,
                                             use_contact_weighting, magOnOff, contactOnOff, only_supervise_stance,
                                             contact_weight_multiplier, magWeight, dof_weights_dict, epoch, total_epochs,
                                             cop_mask=cop_mask,
                                             use_grf_norm_cop=use_grf_norm_cop,
                                             use_gt_jacob_and_rot=use_gt_jacob_and_rot)
            return loss, (metrics, pred)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, pred)), grads = grad_fn(state.params)
        loss_nonfinite = ~jnp.isfinite(loss)
        metrics_nonfinite = _tree_any_nonfinite(metrics)
        pred_nonfinite = _tree_any_nonfinite(pred)
        grads_nonfinite = _tree_any_nonfinite(grads)
        safe_grads = _sanitize_tree(grads)
        updated_state = state.apply_gradients(grads=safe_grads)
        updated_params_nonfinite = _tree_any_nonfinite(updated_state.params)
        updated_opt_state_nonfinite = _tree_any_nonfinite(updated_state.opt_state)
        skip_update = (
            loss_nonfinite
            | metrics_nonfinite
            | pred_nonfinite
            | grads_nonfinite
            | updated_params_nonfinite
            | updated_opt_state_nonfinite
        )
        debug = {
            "grad_global_norm": optax.global_norm(safe_grads),
            "params_nonfinite_before_update": _tree_any_nonfinite(state.params),
            "grads_nonfinite": grads_nonfinite,
            "loss_nonfinite": loss_nonfinite,
            "metrics_nonfinite": metrics_nonfinite,
            "pred_nonfinite": pred_nonfinite,
            "updated_params_nonfinite": updated_params_nonfinite,
            "updated_opt_state_nonfinite": updated_opt_state_nonfinite,
            "update_skipped": skip_update,
        }
        state = jax.lax.cond(skip_update, lambda _: state, lambda _: updated_state, operand=None)
        debug["params_nonfinite_after_update"] = _tree_any_nonfinite(state.params)
        debug["opt_state_nonfinite_after_update"] = _tree_any_nonfinite(state.opt_state)
        return state, metrics, pred, debug

    return train_step


def make_eval_step(normalizers: Dict, use_contact_weighting: bool, magOnOff: bool, contactOnOff: bool, only_supervise_stance: bool, contact_weight_multiplier: float, magWeight: float, total_epochs: int, dof_weights_dict: Dict = None, cop_mask: bool = True, use_grf_norm_cop: bool = False):
    @jax.jit
    def eval_step(state, batch, loss_weights, epoch):
        pred = state.apply_fn(
            {"params": state.params},
            batch["input"],
            batch["static_context"],
            train=False
        )
        _, metrics = compute_total_loss(pred, batch, normalizers, loss_weights,
                                      use_contact_weighting, magOnOff, contactOnOff, only_supervise_stance,
                                      contact_weight_multiplier, magWeight, dof_weights_dict, epoch, total_epochs,
                                      cop_mask=cop_mask,
                                      use_grf_norm_cop=use_grf_norm_cop)
        return metrics, pred

    return eval_step


# =============================================================================
# Visualization
# =============================================================================

def plot_predictions(train_batch, train_pred, val_batch, val_pred, normalizers, epoch, output_dir, 
                     train_sample_idx=0, val_sample_idx=0,
                     train_trial_names=None, val_trial_names=None,
                     train_metrics=None, val_metrics=None,
                     loss_weights=None, epoch_time=None,
                     cop_mask: bool = True,
                     use_grf_norm_cop: bool = False,
                     val_fullset_stats: dict = None):
    """Plot predictions vs ground truth for BOTH train and validation data.
    
    Layout (9 rows x 6 cols):
      Row 0: COP Right X/Z, COP Left X         (Train cols 0-2, Val cols 3-5)
      Row 1: COP Left Z, GRF Right X/Y
      Row 2: GRF Right Z, GRF Left X/Y
      Row 3: GRF Left Z, COP error hist, GRF error hist  (per split)
      Row 4: COP world-frame panel at col 0 + stats panel spanning cols 1-2 (per split)
      Row 5: Torque R Hip Add / R Knee / R Ankle
      Row 6: Torque R Subtalar / L Hip Add / L Knee
      Row 7: Torque L Ankle / L Subtalar
      Row 8: Contact Boolean — Right foot | Left foot   (per split)
    """
    
    # Get trial names for the samples
    train_trial = train_trial_names[train_sample_idx] if train_trial_names else "Unknown"
    val_trial = val_trial_names[val_sample_idx] if val_trial_names else "Unknown"
    
    def _loss_summary(metrics_dict, weights_dict):
        if metrics_dict is None:
            return None
        lw = weights_dict or {}
        comps = [
            ("cop_loss", "cop"),
            ("grf_loss", "grf"),
            ("moments_loss", "moments"),
            ("qfrc_inverse_loss", "qfrc_inverse"),
            ("qfrc_inverse_input_reg_loss", "qfrc_inverse_input_reg"),
            ("rotation_loss", "rotation"),
            ("rotation_input_reg_loss", "rotation_input_reg"),
            ("jacobian_loss", "jacobian"),
            ("jacobian_input_reg_loss", "jacobian_input_reg"),
            ("contact_loss", "contact"),
            ("torque_loss", "torque"),
            ("grf_correction_loss", "grf_correction"),
            ("output_reg_loss", "output_reg"),
        ]
        raw_total = float(metrics_dict.get("total_loss", float("nan")))
        scaled_total = 0.0
        for m_key, w_key in comps:
            raw_val = float(metrics_dict.get(m_key, 0.0))
            scaled_total += raw_val * float(lw.get(w_key, 1.0))
        return {"raw_total": raw_total, "scaled_total": scaled_total}

    train_loss_summary = _loss_summary(train_metrics, loss_weights)
    val_loss_summary = _loss_summary(val_metrics, loss_weights)
    loss_line = ""
    if train_loss_summary is not None and val_loss_summary is not None:
        loss_line = (
            f"Train Loss raw/scaled: {train_loss_summary['raw_total']:.5f}/{train_loss_summary['scaled_total']:.5f}  |  "
            f"Val Loss raw/scaled: {val_loss_summary['raw_total']:.5f}/{val_loss_summary['scaled_total']:.5f}"
        )

    # 9 rows: COP (rows 0-1), GRF (rows 1-3), Histograms (row 3), Stats (row 4), Torques (rows 5-7), Contact (row 8)
    fig, axes = plt.subplots(9, 6, figsize=(30, 49))
    fig.subplots_adjust(hspace=0.52, wspace=0.38, top=0.96, bottom=0.03, left=0.06, right=0.97)
    title = (
        f'Predictions vs Ground Truth — Epoch {epoch}\n'
        f'Blue: TRAIN ({train_trial}) | Red: VAL ({val_trial})'
    )
    if loss_line:
        title += f"\n{loss_line}"
    title += "\nTorque panels: Pred full ID (preprocessed qfrc_inverse - predicted tau_grf) vs clean ID_GT_MJX"
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # ── per-split data accumulator so we can build stats panels after both passes ──
    split_stats = {}

    for col_offset, (batch, predictions, split_name, sample_idx, trial_name) in enumerate([
        (train_batch, train_pred, "TRAIN", train_sample_idx, train_trial),
        (val_batch, val_pred, "VAL", val_sample_idx, val_trial)
    ]):
        col_start = col_offset * 3
        color_gt   = '#1f77b4'   # blue   for GT
        color_pred = '#d62728'   # red    for Pred

        # 1. Predictions: Z-score space -> physiological ratio
        qfrc_inverse_output_dim = _infer_qfrc_inverse_output_dim(batch=batch, normalizers=normalizers)
        cop_pred_z, grf_pred_z, moments_pred_z_raw, _contact_pred_prob, _qfrc_inverse_pred_z, _rotation_pred_z, _jacobian_pred = split_model_predictions(
            predictions,
            qfrc_inverse_output_dim=qfrc_inverse_output_dim,
            rotation_output_dim=0,
        )
        # Outputs are z-scored absolute targets.
        cop_pred_ratio = normalizers["cop"].unnormalize(cop_pred_z)
        grf_pred_ratio = normalizers["grf"].unnormalize(grf_pred_z)
        moments_pred_ratio = normalizers["moments"].unnormalize(moments_pred_z_raw)

        # 2. Static context → physical scalars
        h_batch = batch["static_context"][:, 0, None, None]
        m_batch = batch["static_context"][:, 1, None, None]
        qfrc_inverse_pred_phys, rotation_pred_phys = decode_auxiliary_predictions(
            predictions,
            batch,
            normalizers,
            xp=jnp,
        )

        # 3. Physiological ratio → physical units  (m, N, Nm)
        grf_unnorm      = grf_pred_ratio    * m_batch * 9.8067
        cop_unnorm = decode_cop_signal_to_length(
            cop_pred_ratio,
            grf_pred_ratio,
            h_batch,
            use_grf_norm_cop=use_grf_norm_cop,
            contact_probability=_contact_pred_prob if use_grf_norm_cop else None,
            xp=jnp,
        )
        moments_unnorm  = moments_pred_ratio* m_batch * h_batch * 9.8067

        # 4. Targets → physical units
        grf_target      = batch["grf"]     * m_batch * 9.8067
        cop_target = decode_cop_signal_to_length(
            batch["cop"],
            batch["grf"],
            h_batch,
            use_grf_norm_cop=use_grf_norm_cop,
            xp=jnp,
        )
        moments_target  = batch["moments"] * m_batch * h_batch * 9.8067

        if cop_mask and predictions.shape[-1] >= STANDARD_OUTPUT_DIM:
            contact_hard = (predictions[..., CONTACT_SLICE] > 0.5).astype(cop_unnorm.dtype)
            mask_r = contact_hard[..., 0:1]
            mask_l = contact_hard[..., 1:2]
            cop_unnorm = jnp.concatenate([
                cop_unnorm[..., 0:2] * mask_r,
                cop_unnorm[..., 2:4] * mask_l,
            ], axis=-1)
            grf_unnorm = jnp.concatenate([
                grf_unnorm[..., 0:3] * mask_r,
                grf_unnorm[..., 3:6] * mask_l,
            ], axis=-1)
            moments_unnorm = jnp.concatenate([
                moments_unnorm[..., 0:1] * mask_r,
                moments_unnorm[..., 1:2] * mask_l,
            ], axis=-1)

        # 5. Full ID from denoised qfrc_inverse and predicted GRF contribution
        full_moments = compute_full_external_moments(
            cop_unnorm, grf_unnorm, moments_unnorm, batch["ankle_heights"], rotation_pred_phys
        )
        jacp_for_tau, jacr_for_tau = select_torque_jacobians(
            predictions,
            batch,
            normalizers,
            xp=jnp,
        )
        tau_grf_pred = compute_tau_grf_from_predictions(
            grf_unnorm, full_moments, jacp_for_tau, jacr_for_tau
        )
        full_id_pred = qfrc_inverse_pred_phys - tau_grf_pred
        full_id_target = _full_id_target_from_batch(batch, xp=jnp)

        # 6. Extract single sample as numpy
        idx = sample_idx
        cop_pred_np    = np.array(cop_unnorm[idx])
        grf_pred_np    = np.array(grf_unnorm[idx])
        mom_pred_np    = np.array(moments_unnorm[idx])
        full_id_pred_np    = np.array(full_id_pred[idx])

        cop_target_np  = np.array(cop_target[idx])
        grf_target_np  = np.array(grf_target[idx])
        mom_target_np  = np.array(moments_target[idx])
        full_id_target_np  = np.array(full_id_target[idx])

        seq_len = cop_pred_np.shape[0]
        frames  = np.arange(seq_len)
        sample_frame_mask = _extract_frame_mask(
            np.array(batch["supervision_mask"][idx]) if "supervision_mask" in batch else None,
            seq_len,
        )

        # ── per-channel RMSE helpers ──────────────────────────────────────────
        def ch_rmse(pred, gt):
            return _masked_rmse(pred, gt, sample_frame_mask)

        cop_ch_rmse  = [ch_rmse(cop_pred_np[:, i],  cop_target_np[:, i])  for i in range(4)]
        grf_ch_rmse  = [ch_rmse(grf_pred_np[:, i],  grf_target_np[:, i])  for i in range(6)]
        mom_ch_rmse  = [ch_rmse(mom_pred_np[:, i],  mom_target_np[:, i])  for i in range(2)]
        tau_dof_indices = {
            'R Hip Flexion': 6,
            'R Hip Add': 7,
            'R Knee': 9,
            'R Ankle': 10,
            'R Subtalar': 11,
            'L Hip Flexion': 13,
            'L Hip Add': 14,
            'L Knee': 16,
            'L Ankle': 17,
            'L Subtalar': 18,
            'Lumbar Extension': 20,
            'Lumbar Bending': 21,
            'Lumbar Rotation': 22,
        }
        tau_ch_rmse_sel = {
            name: ch_rmse(full_id_pred_np[:, idx], full_id_target_np[:, idx])
            for name, idx in tau_dof_indices.items()
            if idx < full_id_pred_np.shape[1] and idx < full_id_target_np.shape[1]
        }

        def _tau_panel(row, col, name, title=None):
            idx = tau_dof_indices[name]
            if idx >= full_id_pred_np.shape[1] or idx >= full_id_target_np.shape[1]:
                return None
            return (
                row,
                col,
                full_id_target_np[:, idx],
                full_id_pred_np[:, idx],
                title or f"Full ID {name}",
                tau_ch_rmse_sel[name],
                "Nm",
            )

        cop_overall_rmse = _masked_rmse(cop_pred_np, cop_target_np, sample_frame_mask)
        grf_overall_rmse = _masked_rmse(grf_pred_np, grf_target_np, sample_frame_mask)
        tau_overall_rmse = _masked_rmse(full_id_pred_np, full_id_target_np, sample_frame_mask)
        cop_mae = _masked_mae(cop_pred_np, cop_target_np, sample_frame_mask)
        grf_mae = _masked_mae(grf_pred_np, grf_target_np, sample_frame_mask)
        tau_mae = _masked_mae(full_id_pred_np, full_id_target_np, sample_frame_mask)
        cop_max = _masked_max_abs_err(cop_pred_np, cop_target_np, sample_frame_mask)
        grf_max = _masked_max_abs_err(grf_pred_np, grf_target_np, sample_frame_mask)

        split_stats[split_name] = dict(
            cop_ch_rmse=cop_ch_rmse, grf_ch_rmse=grf_ch_rmse,
            mom_ch_rmse=mom_ch_rmse, tau_ch_rmse_sel=tau_ch_rmse_sel,
            cop_overall_rmse=cop_overall_rmse, grf_overall_rmse=grf_overall_rmse,
            tau_overall_rmse=tau_overall_rmse,
            cop_mae=cop_mae, grf_mae=grf_mae, tau_mae=tau_mae,
            cop_max=cop_max, grf_max=grf_max,
        )

        cop_labels = ['Right X', 'Right Z', 'Left X', 'Left Z']
        grf_labels = ['Right X', 'Right Y', 'Right Z', 'Left X', 'Left Y', 'Left Z']

        # ── plot_configs: (row, col_within_split, gt, pred, base_title, rmse) ──
        plot_configs = [
            # Row 0: COP
            (0, 0, cop_target_np[:, 0], cop_pred_np[:, 0], f'COP {cop_labels[0]}', cop_ch_rmse[0], 'm'),
            (0, 1, cop_target_np[:, 1], cop_pred_np[:, 1], f'COP {cop_labels[1]}', cop_ch_rmse[1], 'm'),
            (0, 2, cop_target_np[:, 2], cop_pred_np[:, 2], f'COP {cop_labels[2]}', cop_ch_rmse[2], 'm'),
            # Row 1: COP Lz + GRF Rx/Ry
            (1, 0, cop_target_np[:, 3], cop_pred_np[:, 3], f'COP {cop_labels[3]}', cop_ch_rmse[3], 'm'),
            (1, 1, grf_target_np[:, 0], grf_pred_np[:, 0], f'GRF {grf_labels[0]}', grf_ch_rmse[0], 'N'),
            (1, 2, grf_target_np[:, 1], grf_pred_np[:, 1], f'GRF {grf_labels[1]}', grf_ch_rmse[1], 'N'),
            # Row 2: GRF Rz/Lx/Ly
            (2, 0, grf_target_np[:, 2], grf_pred_np[:, 2], f'GRF {grf_labels[2]}', grf_ch_rmse[2], 'N'),
            (2, 1, grf_target_np[:, 3], grf_pred_np[:, 3], f'GRF {grf_labels[3]}', grf_ch_rmse[3], 'N'),
            (2, 2, grf_target_np[:, 4], grf_pred_np[:, 4], f'GRF {grf_labels[4]}', grf_ch_rmse[4], 'N'),
            # Row 3 col 0: GRF Lz
            (3, 0, grf_target_np[:, 5], grf_pred_np[:, 5], f'GRF {grf_labels[5]}', grf_ch_rmse[5], 'N'),
            # Torque rows use the trimmed 23-DOF independent coordinate layout.
            _tau_panel(5, 0, 'R Hip Add'),
            _tau_panel(5, 1, 'R Knee'),
            _tau_panel(5, 2, 'R Ankle'),
            _tau_panel(6, 0, 'R Subtalar'),
            _tau_panel(6, 1, 'L Hip Add'),
            _tau_panel(6, 2, 'L Knee'),
            _tau_panel(7, 0, 'L Ankle'),
            _tau_panel(7, 1, 'L Subtalar'),
        ]
        plot_configs = [cfg for cfg in plot_configs if cfg is not None]

        for r, c, gt, pred, base_title, rmse_val, unit in plot_configs:
            ax = axes[r, col_start + c]
            ax.plot(frames, gt,   color=color_gt,   linestyle='-',  linewidth=1.8, label='GT')
            ax.plot(frames, pred, color=color_pred,  linestyle='--', linewidth=1.8, label='Pred')
            ax.set_title(f'[{split_name}] {base_title}\nRMSE: {rmse_val:.4g} {unit}', fontsize=8)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)

        # ── Row 3: COP & GRF error histograms (cols 1-2 within split) ─────────
        hist_color = '#2196F3' if split_name == "TRAIN" else '#FF9800'

        cop_err_flat = (cop_pred_np - cop_target_np).flatten() * 100  # cm
        grf_err_flat = (grf_pred_np - grf_target_np).flatten()        # N

        ax_cop_hist = axes[3, col_start + 1]
        ax_cop_hist.hist(cop_err_flat, bins=40, color=hist_color, alpha=0.75, edgecolor='white', linewidth=0.4)
        ax_cop_hist.axvline(0, color='black', linewidth=1.2, linestyle='--')
        ax_cop_hist.axvline( cop_overall_rmse * 100, color='red',   linewidth=1.2, linestyle=':', label=f'+RMSE')
        ax_cop_hist.axvline(-cop_overall_rmse * 100, color='red',   linewidth=1.2, linestyle=':')
        ax_cop_hist.set_title(
            f'[{split_name}] COP Error Distribution\n'
            f'RMSE: {cop_overall_rmse*100:.2f} cm  MAE: {cop_mae*100:.2f} cm  Max: {cop_max*100:.2f} cm',
            fontsize=8)
        ax_cop_hist.set_xlabel('Error (cm)', fontsize=7)
        ax_cop_hist.set_ylabel('Count', fontsize=7)
        ax_cop_hist.legend(fontsize=6)
        ax_cop_hist.grid(True, alpha=0.3)

        ax_grf_hist = axes[3, col_start + 2]
        ax_grf_hist.hist(grf_err_flat, bins=40, color=hist_color, alpha=0.75, edgecolor='white', linewidth=0.4)
        ax_grf_hist.axvline(0, color='black', linewidth=1.2, linestyle='--')
        ax_grf_hist.axvline( grf_overall_rmse, color='red',   linewidth=1.2, linestyle=':', label=f'+RMSE')
        ax_grf_hist.axvline(-grf_overall_rmse, color='red',   linewidth=1.2, linestyle=':')
        ax_grf_hist.set_title(
            f'[{split_name}] GRF Error Distribution\n'
            f'RMSE: {grf_overall_rmse:.2f} N  MAE: {grf_mae:.2f} N  Max: {grf_max:.2f} N',
            fontsize=8)
        ax_grf_hist.set_xlabel('Error (N)', fontsize=7)
        ax_grf_hist.set_ylabel('Count', fontsize=7)
        ax_grf_hist.legend(fontsize=6)
        ax_grf_hist.grid(True, alpha=0.3)

        # ── Row 4 col 0: COP in world frame (relative to calcaneus) ──────────
        # Convert COP from ground-aligned calc frame [X, Z] (+ankle height as Y)
        # back to world using R_ga->w = (R_w->ga)^T for both feet.
        ax_world = axes[4, col_start + 0]
        try:
            rot_w_to_ga_pred_np = np.array(rotation_pred_phys[idx])
            rot_w_to_ga_gt_np = np.array(batch["gt_rot_w_to_ga"][idx] if batch.get("gt_rot_w_to_ga") is not None else batch["rot_w_to_ga"][idx])
            ankle_h_np = np.array(batch["ankle_heights"][idx])

            rot_ga_to_w_r_pred = np.transpose(rot_w_to_ga_pred_np[:, 0], (0, 2, 1))
            rot_ga_to_w_l_pred = np.transpose(rot_w_to_ga_pred_np[:, 1], (0, 2, 1))
            rot_ga_to_w_r_gt = np.transpose(rot_w_to_ga_gt_np[:, 0], (0, 2, 1))
            rot_ga_to_w_l_gt = np.transpose(rot_w_to_ga_gt_np[:, 1], (0, 2, 1))

            cop_r_ga_pred = np.stack([cop_pred_np[:, 0], ankle_h_np[:, 0], cop_pred_np[:, 1]], axis=1)
            cop_l_ga_pred = np.stack([cop_pred_np[:, 2], ankle_h_np[:, 1], cop_pred_np[:, 3]], axis=1)
            cop_r_ga_gt   = np.stack([cop_target_np[:, 0], ankle_h_np[:, 0], cop_target_np[:, 1]], axis=1)
            cop_l_ga_gt   = np.stack([cop_target_np[:, 2], ankle_h_np[:, 1], cop_target_np[:, 3]], axis=1)

            cop_r_world_pred = np.einsum("tij,tj->ti", rot_ga_to_w_r_pred, cop_r_ga_pred)
            cop_l_world_pred = np.einsum("tij,tj->ti", rot_ga_to_w_l_pred, cop_l_ga_pred)
            cop_r_world_gt   = np.einsum("tij,tj->ti", rot_ga_to_w_r_gt, cop_r_ga_gt)
            cop_l_world_gt   = np.einsum("tij,tj->ti", rot_ga_to_w_l_gt, cop_l_ga_gt)

            rmse_rwx = ch_rmse(cop_r_world_pred[:, 0], cop_r_world_gt[:, 0])
            rmse_lwx = ch_rmse(cop_l_world_pred[:, 0], cop_l_world_gt[:, 0])
            rmse_rwy = ch_rmse(cop_r_world_pred[:, 1], cop_r_world_gt[:, 1])
            rmse_lwy = ch_rmse(cop_l_world_pred[:, 1], cop_l_world_gt[:, 1])

            ax_world.plot(frames, cop_r_world_gt[:, 0], color="#1f77b4", lw=1.6, label="GT RwX")
            ax_world.plot(frames, cop_r_world_pred[:, 0], color="#1f77b4", lw=1.6, ls="--", label="Pred RwX")
            ax_world.plot(frames, cop_l_world_gt[:, 0], color="#2ca02c", lw=1.6, label="GT LwX")
            ax_world.plot(frames, cop_l_world_pred[:, 0], color="#2ca02c", lw=1.6, ls="--", label="Pred LwX")
            ax_world.plot(frames, cop_r_world_gt[:, 1], color="#9467bd", lw=1.2, alpha=0.8, label="GT RwY")
            ax_world.plot(frames, cop_r_world_pred[:, 1], color="#9467bd", lw=1.2, alpha=0.8, ls="--", label="Pred RwY")
            ax_world.plot(frames, cop_l_world_gt[:, 1], color="#ff7f0e", lw=1.2, alpha=0.8, label="GT LwY")
            ax_world.plot(frames, cop_l_world_pred[:, 1], color="#ff7f0e", lw=1.2, alpha=0.8, ls="--", label="Pred LwY")

            ax_world.set_title(
                f'[{split_name}] COP World (using residual-composed rotation)\n'
                f'RMSE RwX={rmse_rwx:.4g}m LwX={rmse_lwx:.4g}m | RwY={rmse_rwy:.4g}m LwY={rmse_lwy:.4g}m',
                fontsize=8
            )
            ax_world.set_xlabel("Frame", fontsize=7)
            ax_world.set_ylabel("COP (m)", fontsize=7)
            ax_world.grid(True, alpha=0.3)
            ax_world.tick_params(labelsize=6)
            ax_world.legend(fontsize=6, loc="upper right", ncol=2)
        except Exception:
            ax_world.set_visible(False)

    # ── Row 4, cols 1-2 (Train) and 4-5 (Val): comprehensive stats panels ─────
    for col_offset, split_name in enumerate(["TRAIN", "VAL"]):
        col_start   = col_offset * 3
        # For VAL, prefer full-dataset stats over single-sample stats when available.
        st          = (val_fullset_stats if (split_name == "VAL" and val_fullset_stats is not None)
                       else split_stats[split_name])
        epoch_metrics = train_metrics if split_name == "TRAIN" else val_metrics
        bg_color    = '#eaf4fb' if split_name == "TRAIN" else '#fef9e7'
        accent      = '#1a5276' if split_name == "TRAIN" else '#784212'

        # Span cols 1-2 within the split's column block using inset approach via merged axis.
        # We remove the two default axes and replace with one spanning subplot.
        axes[4, col_start + 1].remove()
        axes[4, col_start + 2].remove()
        ax_stats = fig.add_subplot(8, 6, (4 * 6 + col_start + 1) + 1)  # 1-indexed
        ax_stats.set_position(
            [axes[4, col_start + 1].get_position().x0,
             axes[4, col_start + 1].get_position().y0,
             axes[4, col_start + 2].get_position().x1 - axes[4, col_start + 1].get_position().x0,
             axes[4, col_start + 1].get_position().height]
        )
        ax_stats.set_facecolor(bg_color)
        ax_stats.set_xticks([]); ax_stats.set_yticks([])
        for spine in ax_stats.spines.values():
            spine.set_edgecolor(accent); spine.set_linewidth(1.5)

        cop_ch_lbl  = ['COP Rx', 'COP Rz', 'COP Lx', 'COP Lz']
        grf_ch_lbl  = ['GRF Rx', 'GRF Ry', 'GRF Rz', 'GRF Lx', 'GRF Ly', 'GRF Lz']
        mom_ch_lbl  = ['Mz R', 'Mz L']

        # Build text lines
        lines = []
        if split_name == "VAL" and val_fullset_stats is not None:
            lines.append((f'VAL FULL-DATASET STATISTICS', accent, 13.5, 'bold'))
            scope_note = 'Full validation set'
        elif split_name == "TRAIN":
            lines.append((f'TRAIN SAMPLE STATISTICS', accent, 13.5, 'bold'))
            scope_note = 'Sample window'
        else:
            lines.append((f'VAL SAMPLE STATISTICS (full-set N/A)', accent, 13.5, 'bold'))
            scope_note = 'Sample window'

        # Overall metrics
        lines.append((f'Overall ({scope_note})', accent, 11.5, 'bold'))
        lines.append((f'COP RMSE {st["cop_overall_rmse"]*100:.2f} cm | MAE {st["cop_mae"]*100:.2f} cm | MaxErr {st["cop_max"]*100:.2f} cm', 'black', 10.5, 'normal'))
        lines.append((f'GRF RMSE {st["grf_overall_rmse"]:.2f} N | MAE {st["grf_mae"]:.2f} N | MaxErr {st["grf_max"]:.2f} N', 'black', 10.5, 'normal'))
        lines.append((f'Full ID RMSE {st["tau_overall_rmse"]:.2f} Nm | MAE {st["tau_mae"]:.2f} Nm', 'black', 10.5, 'normal'))

        # Per-channel COP
        lines.append(('COP per-channel RMSE (cm)', accent, 10.5, 'bold'))
        cop_row = ' | '.join(f'{lbl}: {v*100:.2f}' for lbl, v in zip(cop_ch_lbl, st['cop_ch_rmse']))
        lines.append((cop_row, 'black', 9.5, 'normal'))

        # Per-channel GRF
        lines.append(('GRF per-channel RMSE (N)', accent, 10.5, 'bold'))
        grf_row1 = ' | '.join(f'{lbl}: {v:.2f}' for lbl, v in zip(grf_ch_lbl[:3], st['grf_ch_rmse'][:3]))
        grf_row2 = ' | '.join(f'{lbl}: {v:.2f}' for lbl, v in zip(grf_ch_lbl[3:], st['grf_ch_rmse'][3:]))
        lines.append((grf_row1, 'black', 9.5, 'normal'))
        lines.append((grf_row2, 'black', 9.5, 'normal'))

        # Free moments
        lines.append(('Free Moment per-channel RMSE (Nm)', accent, 10.5, 'bold'))
        mom_row = ' | '.join(f'{lbl}: {v:.3f}' for lbl, v in zip(mom_ch_lbl, st['mom_ch_rmse']))
        lines.append((mom_row, 'black', 9.5, 'normal'))

        # Tracked torques
        lines.append(('Tracked Full ID RMSE (Nm)', accent, 10.5, 'bold'))
        tau_names = list(st['tau_ch_rmse_sel'].keys())
        tau_vals  = list(st['tau_ch_rmse_sel'].values())
        for i in range(0, len(tau_names), 4):
            chunk = tau_names[i:i+4]
            cvals = tau_vals[i:i+4]
            lines.append((' | '.join(f'{n}: {v:.2f}' for n, v in zip(chunk, cvals)), 'black', 9.5, 'normal'))

        # Epoch-level loss breakdown (from metrics dicts)
        if epoch_metrics is not None:
            lines.append(('Epoch Loss (avg over batches)', accent, 11.0, 'bold'))
            lw = loss_weights if loss_weights else {}
            def _fmt_loss(key):
                raw = float(epoch_metrics.get(key, float('nan')))
                scaled = raw * float(lw.get(key.replace('_loss', ''), 1.0)) if lw else raw
                return raw, scaled
            total_raw = float(epoch_metrics.get("total_loss", float("nan")))
            total_scaled = sum(
                float(epoch_metrics.get(mk, 0.0)) * float(lw.get(wk, 1.0))
                for mk, wk in [
                    ("cop_loss", "cop"),
                    ("grf_loss", "grf"),
                    ("moments_loss", "moments"),
                    ("qfrc_inverse_loss", "qfrc_inverse"),
                    ("qfrc_inverse_input_reg_loss", "qfrc_inverse_input_reg"),
                    ("rotation_loss", "rotation"),
                    ("rotation_input_reg_loss", "rotation_input_reg"),
                    ("jacobian_loss", "jacobian"),
                    ("jacobian_input_reg_loss", "jacobian_input_reg"),
                    ("contact_loss", "contact"),
                    ("torque_loss", "torque"),
                    ("grf_correction_loss", "grf_correction"),
                    ("output_reg_loss", "output_reg"),
                ]
            )
            lines.append((f'Total raw/scaled: {total_raw:.5f} / {total_scaled:.5f}', 'black', 10.5, 'bold'))

            cop_raw, cop_scaled = _fmt_loss("cop_loss")
            grf_raw, grf_scaled = _fmt_loss("grf_loss")
            mom_raw, mom_scaled = _fmt_loss("moments_loss")
            qfrc_raw, qfrc_scaled = _fmt_loss("qfrc_inverse_loss")
            qfrc_in_raw, qfrc_in_scaled = _fmt_loss("qfrc_inverse_input_reg_loss")
            rot_raw, rot_scaled = _fmt_loss("rotation_loss")
            rot_in_raw, rot_in_scaled = _fmt_loss("rotation_input_reg_loss")
            jac_raw, jac_scaled = _fmt_loss("jacobian_loss")
            jac_in_raw, jac_in_scaled = _fmt_loss("jacobian_input_reg_loss")
            con_raw, con_scaled = _fmt_loss("contact_loss")
            trq_raw, trq_scaled = _fmt_loss("torque_loss")
            cor_raw, cor_scaled = _fmt_loss("grf_correction_loss")
            out_raw, out_scaled = _fmt_loss("output_reg_loss")
            lines.append((f'COP {cop_raw:.4f}/{cop_scaled:.4f} | GRF {grf_raw:.4f}/{grf_scaled:.4f} | Mom {mom_raw:.4f}/{mom_scaled:.4f}', 'black', 9.5, 'normal'))
            lines.append((f'QInv {qfrc_raw:.4f}/{qfrc_scaled:.4f} | Rot {rot_raw:.4f}/{rot_scaled:.4f} | Jac {jac_raw:.4f}/{jac_scaled:.4f}', 'black', 9.5, 'normal'))
            lines.append((f'QInvReg {qfrc_in_raw:.4f}/{qfrc_in_scaled:.4f} | RotReg {rot_in_raw:.4f}/{rot_in_scaled:.4f} | JacReg {jac_in_raw:.4f}/{jac_in_scaled:.4f}', 'black', 9.5, 'normal'))
            lines.append((f'Cont {con_raw:.4f}/{con_scaled:.4f} | TauGRF {trq_raw:.4f}/{trq_scaled:.4f}', 'black', 9.5, 'normal'))
            lines.append((f'Cor {cor_raw:.4f}/{cor_scaled:.4f} | OutReg {out_raw:.4f}/{out_scaled:.4f}', 'black', 9.5, 'normal'))
            if epoch_time is not None and split_name == "VAL":
                lines.append((f'Epoch time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)', accent, 9.5, 'normal'))

        # Render lines top-to-bottom
        y_cursor = 0.97
        for text, color, fsize, fweight in lines:
            ax_stats.text(0.02, y_cursor, text, transform=ax_stats.transAxes,
                          fontsize=fsize, color=color, fontweight=fweight,
                          verticalalignment='top', family='sans-serif')
            y_cursor -= 0.047
            if y_cursor < 0.01:
                break

    # ── Row 8: Contact Boolean plots (Right foot | Left foot) per split ──────
    for col_offset, (batch, predictions, split_name, sample_idx, _) in enumerate([
        (train_batch, train_pred, "TRAIN", train_sample_idx, train_trial),
        (val_batch, val_pred, "VAL", val_sample_idx, val_trial)
    ]):
        col_start = col_offset * 3
        if predictions.shape[-1] >= STANDARD_OUTPUT_DIM and "contactBoolean" in batch:
            contact_prob_np = np.array(predictions[sample_idx, :, CONTACT_SLICE])   # (seq, 2) sigmoid
            contact_gt_np   = np.array(batch["contactBoolean"][sample_idx])  # (seq, 2) binary
            frames = np.arange(contact_prob_np.shape[0])
            for foot_col, (foot_name, ch) in enumerate([("Right", 0), ("Left", 1)]):
                ax = axes[8, col_start + foot_col]
                ax.plot(frames, contact_gt_np[:, ch],   color='#1f77b4', lw=1.8, label='GT')
                ax.plot(frames, contact_prob_np[:, ch], color='#d62728', lw=1.8, ls='--', label='Pred')
                ax.axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f'[{split_name}] Contact {foot_name}', fontsize=8)
                ax.set_xlabel('Frame', fontsize=7)
                ax.set_ylabel('Contact', fontsize=7)
                ax.legend(fontsize=6, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=6)
        else:
            for c in range(2):
                axes[8, col_start + c].set_visible(False)
        # Hide the unused 3rd column of each split's contact row
        axes[8, col_start + 2].set_visible(False)

    plt.savefig(os.path.join(output_dir, f"predictions_epoch_{epoch:04d}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_validation_outlier_grid(
    epoch: int,
    output_dir: str,
    top_outliers_by_channel: Dict[int, List[Tuple[float, str, int]]],
    outlier_channel_defs: List[Tuple[str, str]],
    val_trial_series: Dict[int, Dict[str, List]],
    top_k: int = 3,
):
    """Save one large PNG with GT vs Pred for top outlier trials per channel."""
    if not top_outliers_by_channel:
        return

    def _reassemble_channel_windows(
        channel_windows: List[np.ndarray],
        window_starts: List[int],
        window_masks: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Reassemble a trial channel from overlapping windows using original start indices.
        Overlapping samples are averaged using only center-valid frames.
        """
        if not channel_windows:
            return np.array([], dtype=np.float32)
        if len(channel_windows) != len(window_starts):
            return np.concatenate(channel_windows, axis=0).astype(np.float32)
        if window_masks is not None and len(window_masks) != len(channel_windows):
            window_masks = None

        ordered = sorted(
            zip(
                window_starts,
                channel_windows,
                window_masks if window_masks is not None else [None] * len(channel_windows),
            ),
            key=lambda x: int(x[0]),
        )
        max_end = max(int(s) + int(w.shape[0]) for s, w, _m in ordered)
        if max_end <= 0:
            return np.array([], dtype=np.float32)

        accum = np.zeros((max_end,), dtype=np.float64)
        counts = np.zeros((max_end,), dtype=np.float64)
        for s, w, mask in ordered:
            s_i = int(s)
            vec = np.asarray(w, dtype=np.float64).reshape(-1)
            e_i = s_i + vec.shape[0]
            if s_i < 0 or e_i <= s_i:
                continue
            if mask is None:
                mask_vec = np.ones_like(vec, dtype=np.float64)
            else:
                mask_vec = np.asarray(mask, dtype=np.float64).reshape(-1)
                if mask_vec.shape[0] != vec.shape[0]:
                    mask_vec = np.ones_like(vec, dtype=np.float64)
            accum[s_i:e_i] += vec * mask_vec
            counts[s_i:e_i] += mask_vec

        valid = counts > 0.0
        if not np.any(valid):
            return np.array([], dtype=np.float32)

        out = np.full((max_end,), np.nan, dtype=np.float64)
        out[valid] = accum[valid] / counts[valid]
        return out.astype(np.float32)

    n_rows = len(outlier_channel_defs)
    n_cols = top_k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 2.8 * n_rows), squeeze=False)
    fig.suptitle(
        f"Validation Top-{top_k} Outliers Per Channel (Epoch {epoch})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for ch_idx, (ch_name, ch_unit) in enumerate(outlier_channel_defs):
        channel_top = top_outliers_by_channel.get(ch_idx, [])
        for rank_idx in range(top_k):
            ax = axes[ch_idx, rank_idx]

            if rank_idx >= len(channel_top):
                ax.axis("off")
                continue

            rmse, trial_name, trial_idx = channel_top[rank_idx]
            trial_series = val_trial_series.get(trial_idx, None)
            if trial_series is None or len(trial_series["pred"]) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.set_axis_off()
                continue

            starts = [int(s) for s in trial_series.get("starts", [])]
            masks = [
                np.asarray(m).reshape(-1)
                for m in trial_series.get("masks", [])
            ] if trial_series.get("masks") else None
            pred_windows = [w[:, ch_idx] for w in trial_series["pred"]]
            gt_windows = [w[:, ch_idx] for w in trial_series["gt"]]
            pred_concat = _reassemble_channel_windows(pred_windows, starts, masks)
            gt_concat = _reassemble_channel_windows(gt_windows, starts, masks)
            n = min(pred_concat.shape[0], gt_concat.shape[0])
            if n <= 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax.set_axis_off()
                continue
            pred_concat = pred_concat[:n]
            gt_concat = gt_concat[:n]
            x = np.arange(n)

            ax.plot(x, gt_concat, color="#1f77b4", linewidth=0.9, label="GT")
            ax.plot(x, pred_concat, color="#d62728", linewidth=0.9, alpha=0.9, label="Pred")
            ax.set_title(
                f"{ch_name} | #{rank_idx + 1} {trial_name}\nRMSE={rmse:.4f} {ch_unit}",
                fontsize=7.5,
            )
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25)

            if ch_idx == n_rows - 1:
                ax.set_xlabel("Frame (reassembled)", fontsize=6.5)
            if rank_idx == 0:
                ax.set_ylabel(ch_unit, fontsize=6.5)
            if ch_idx == 0 and rank_idx == 0:
                ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    out_path = os.path.join(output_dir, f"validation_outliers_epoch_{epoch:04d}.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_loss_history(
    train_losses,
    val_losses,
    output_dir,
    train_component_history=None,
    val_component_history=None,
    loss_weights=None,
):
    """Plot training and validation loss history with train/val component breakdowns.

    Args:
        train_losses: List of total training loss per epoch.
        val_losses:   List of total validation loss per epoch.
        output_dir:   Directory to save the figure.
        train_component_history: Dict mapping component name -> list of raw (unscaled) train losses.
        val_component_history: Dict mapping component name -> list of raw (unscaled) val losses.
            Expected keys: 'cop_loss', 'grf_loss', 'moments_loss', 'qfrc_inverse_loss',
                           'qfrc_inverse_input_reg_loss', 'rotation_loss',
                           'rotation_input_reg_loss', 'jacobian_loss',
                           'jacobian_input_reg_loss', 'torque_loss',
                           'grf_correction_loss', 'output_reg_loss'.
        loss_weights: Dict mapping weight keys to floats, used to scale each component.
            Expected keys: 'cop', 'grf', 'moments', 'qfrc_inverse',
                           'qfrc_inverse_input_reg', 'rotation',
                           'rotation_input_reg', 'jacobian',
                           'jacobian_input_reg', 'torque',
                           'grf_correction', 'output_reg'.
    """
    epochs = range(1, len(train_losses) + 1)
    epochs_arr = list(epochs)

    # Component display config: (history_key, weight_key, display_label, colour)
    COMPONENT_CFG = [
        ("cop_loss",             "cop",             "COP",         "#e41a1c"),
        ("grf_loss",             "grf",             "GRF",         "#377eb8"),
        ("moments_loss",         "moments",         "Moments",     "#4daf4a"),
        ("qfrc_inverse_loss",    "qfrc_inverse",    "QfrcInv",     "#984ea3"),
        ("qfrc_inverse_input_reg_loss", "qfrc_inverse_input_reg", "QInv Reg", "#cab2d6"),
        ("rotation_loss",        "rotation",        "Rot Geo",     "#ff7f00"),
        ("rotation_input_reg_loss", "rotation_input_reg", "Rot Resid", "#fdbf6f"),
        ("jacobian_loss",        "jacobian",        "Jacobian",    "#66c2a5"),
        ("jacobian_input_reg_loss", "jacobian_input_reg", "Jac Reg", "#b2df8a"),
        ("contact_loss",         "contact",         "Contact",     "#f781bf"),
        ("torque_loss",          "torque",          "Tau GRF",     "#a65628"),
        ("grf_correction_loss",  "grf_correction",  "GRF Corr",   "#999999"),
        ("output_reg_loss",      "output_reg",      "Out Reg",    "#6b6b6b"),
    ]

    def _build_scaled_stack(component_history, expected_len):
        if component_history is None or loss_weights is None:
            return None
        if len(component_history.get("cop_loss", [])) != expected_len:
            return None
        scaled = {}
        for hist_key, w_key, _label, _color in COMPONENT_CFG:
            raw = np.array(component_history.get(hist_key, [0.0] * expected_len), dtype=float)
            w = float(loss_weights.get(w_key, 1.0))
            scaled[hist_key] = raw * w
        stack_keys = [k for k, _, _, _ in COMPONENT_CFG]
        return np.array([scaled[k] for k in stack_keys])  # (n_components, n_epochs)

    train_stacked = _build_scaled_stack(train_component_history, len(train_losses))
    val_stacked = _build_scaled_stack(val_component_history, len(val_losses))

    # -------------------------------------------------------------------------
    # Figure layout: 2 panels
    #   Left  — train vs val total (linear)
    #   Right — same curves + stacked training component breakdown (linear)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes[0], axes[1]

    val_arr   = np.array(val_losses, dtype=float)
    train_arr = np.array(train_losses, dtype=float)

    # ---- Panel 1: Train vs Val total (linear) + val breakdown ----------------
    if val_stacked is not None:
        bottom = np.zeros(len(epochs_arr))
        for (_hist_key, _w_key, label, color), vals in zip(COMPONENT_CFG, val_stacked):
            ax1.fill_between(
                epochs_arr,
                bottom,
                bottom + vals,
                alpha=0.35,
                color=color,
                label=f'Val {label}',
            )
            bottom = bottom + vals

    ax1.plot(epochs_arr, train_arr, 'b-', linewidth=1.8, label='Train Total', zorder=5)
    ax1.plot(epochs_arr, val_arr,   'r-', linewidth=2.2, label='Val Total',   zorder=6)
    ax1.set_title('Loss History (Linear Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Train-component breakdown + total curves (linear) ----------
    if train_stacked is not None:
        bottom = np.zeros(len(epochs_arr))
        for (_hist_key, _w_key, label, color), vals in zip(COMPONENT_CFG, train_stacked):
            ax2.fill_between(epochs_arr, bottom, bottom + vals,
                             alpha=0.45, color=color, label=f'Train {label}')
            bottom = bottom + vals

    ax2.plot(epochs_arr, train_arr, 'b-', linewidth=1.8, label='Train Total', zorder=5)
    ax2.plot(epochs_arr, val_arr,   'r-', linewidth=2.2, label='Val Total',   zorder=6)
    ax2.set_title('Train Loss Composition (Linear Scale)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=150)
    plt.close()


def get_batch(data: Dict[str, jnp.ndarray], indices: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    return {k: v[indices] for k, v in data.items()}


def save_balance_report(stats: Optional[Dict], output_dir: str) -> None:
    """Print and save the speed/gender balancing report (console + JSON + histogram PNG)."""
    if not stats:
        print("⚠️  Balancing requested but no balance stats were produced.", flush=True)
        return

    gc_ = stats.get("gender_counts", {})
    gw_ = stats.get("gender_weights", {})
    ws_ = stats.get("weight_summary", {})
    bins = stats.get("speed_bins", [])

    print("\n⚖️  Speed/Gender window balancing", flush=True)
    print(f"   Gender windows  -> male: {gc_.get('male', 0):.0f}  "
          f"female: {gc_.get('female', 0):.0f}  unknown: {gc_.get('unknown', 0):.0f}", flush=True)
    print(f"   Gender weights  -> male: {gw_.get('male', 1):.3f}  "
          f"female: {gw_.get('female', 1):.3f}  unknown: {gw_.get('unknown', 1):.3f}", flush=True)
    print(f"   Speed histogram ({stats.get('config', {}).get('bin_width', 0.05)} m/s bins):", flush=True)
    for b in bins:
        print(f"     [{b['left_edge']:.2f}, {b['right_edge']:.2f}) m/s  "
              f"windows={b['n_windows']:.0f}  speed_w={b['speed_weight']:.3f}", flush=True)
    print(f"   Final per-window weights -> min: {ws_.get('min', 1):.3f}  "
          f"max: {ws_.get('max', 1):.3f}  mean: {ws_.get('mean', 1):.3f}  "
          f"clipped: {ws_.get('clipped_fraction', 0) * 100:.1f}%", flush=True)

    # JSON
    json_path = os.path.join(output_dir, "speed_gender_balance.json")
    try:
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"   📝 Saved balance stats: {json_path}", flush=True)
    except Exception as e:
        print(f"   ⚠️ Failed to save balance JSON: {e}", flush=True)

    # Histogram PNG: raw counts (left) vs weighted/effective mass (right),
    # both stacked by gender. The weighted panel should show flatter bins and a
    # tighter male/female balance — i.e. what the loss actually sees.
    if bins:
        try:
            lefts = [b["left_edge"] for b in bins]
            width = stats.get("config", {}).get("bin_width", 0.05)

            def _stacked(ax, mkey, fkey, ukey, title, ylabel):
                male = [b.get(mkey, 0.0) for b in bins]
                female = [b.get(fkey, 0.0) for b in bins]
                unknown = [b.get(ukey, 0.0) for b in bins]
                ax.bar(lefts, male, width=width * 0.95, align="edge", label="male", color="#1f77b4")
                ax.bar(lefts, female, width=width * 0.95, align="edge", bottom=male,
                       label="female", color="#d62728")
                bottom_fu = [m + f for m, f in zip(male, female)]
                ax.bar(lefts, unknown, width=width * 0.95, align="edge", bottom=bottom_fu,
                       label="unknown", color="#7f7f7f")
                ax.set_xlabel("Walking speed (m/s)")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend()

            fig, (ax_raw, ax_w) = plt.subplots(1, 2, figsize=(16, 5))
            _stacked(ax_raw, "n_windows_male", "n_windows_female", "n_windows_unknown",
                     "Raw window counts", "Number of windows")
            _stacked(ax_w, "w_windows_male", "w_windows_female", "w_windows_unknown",
                     "Weighted (effective) window mass", "Effective windows (Σ weights)")
            fig.suptitle("Training windows by walking speed and gender — raw vs. weighted")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            png_path = os.path.join(output_dir, "speed_histogram.png")
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"   📊 Saved speed histogram (raw + weighted): {png_path}", flush=True)
        except Exception as e:
            print(f"   ⚠️ Failed to save speed histogram: {e}", flush=True)


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    print("🩺 NaN diagnostics enabled: fail-fast on first non-finite training step", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../Data_Full_Cleaned")
    parser.add_argument(
        "--val_subjects_json",
        type=str,
        default="",
        help="Optional JSON file or JSON list specifying validation subjects; all other subjects are used for training.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for all transformer layers")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="L2 regularization (weight decay) strength")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--prediction_margin_frames",
        type=int,
        default=20,
        help="Use the full window as context, but supervise and score only the center frames [margin : window_size - margin).",
    )
    parser.add_argument(
        "--base_config_id",
        type=str,
        default="",
        help="Optional preset ID (cfg01..cfg03) that overrides dropout_rate, d_model, num_layers, and window_size.",
    )
    # parser.add_argument("--min_trial_length", type=int, default=300, help="Skip trials shorter than this length")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument(
        "--ff_dim",
        type=int,
        default=None,
        help="Feed-forward dimension. Defaults to 4 * d_model when omitted.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/v5")
    parser.add_argument(
        "--save_model_epochs",
        type=_parse_save_model_epochs_arg,
        default=[],
        help="Optional comma-separated epoch list, e.g. '(7,8,9)' or '7,8,9', to save model_epoch_####.pkl in addition to best_model.pkl.",
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument(
        "--save_final_predictions_only",
        action="store_true",
        help="Only save the final predictions panel for the run; skip periodic and best-epoch prediction plots.",
    )
    parser.add_argument(
        "--save_best_model_png_only",
        action="store_true",
        help="For HPO runs, keep only one prediction PNG named best_model.png for the best-torque/best-val epoch.",
    )
    parser.add_argument(
        "--disable_validation_outlier_plots",
        action="store_true",
        help="Disable validation outlier accumulation and skip validation_outliers_epoch_####.png outputs.",
    )
    parser.add_argument(
        "--scan_workers",
        type=int,
        default=4,
        help="Subject-scan thread count during trial discovery (lower helps multi-agent RAM/CPU contention).",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="Wandb API key")
    parser.add_argument("--exp_name", type=str, default="cop_grf_v5", help="Experiment name for wandb")
    parser.add_argument("--wandb_project", type=str, default="gait-dynamics-jax", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Optional WandB entity/team")
    parser.add_argument("--wandb_group", type=str, default=None, help="Optional WandB group name")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated WandB tags")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional WandB mode override",
    )
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Optional WandB run ID (for resume)")
    parser.add_argument(
        "--wandb_resume",
        type=str,
        default=None,
        choices=["allow", "must", "never", "auto"],
        help="Optional WandB resume policy",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Path to checkpoint to resume from. Leave empty to train from scratch.",
    )
    # Loss weights (Default_Weights)
    parser.add_argument("--cop_weight", type=float, default=1.0)
    parser.add_argument("--grf_weight", type=float, default=1.0)
    parser.add_argument("--moments_weight", type=float, default=.25)
    parser.add_argument("--contact_weight", type=float, default=1.0)
    parser.add_argument(
        "--contact_weight_multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to stance-contact frames when contact weighting is enabled.",
    )
    parser.add_argument(
        "--magWeight",
        type=float,
        default=3.0,
        help="Scaling factor for the torque-loss magnitude multiplier.",
    )
    parser.add_argument("--torque_weight", type=float, default=2.0, help="Weight for qfrc_GRF_contribution torque supervision.")
    parser.add_argument("--qfrc_inverse_weight", type=float, default=0.0, help="Deprecated: qfrc_inverse is loaded from preprocessing and is not predicted.")
    parser.add_argument(
        "--qfrc_inverse_input_reg_weight",
        type=float,
        default=None,
        help="Deprecated: qfrc_inverse is not a model input/output.",
    )
    parser.add_argument("--rotation_weight", type=float, default=0.0, help="Deprecated: rotation is loaded from preprocessing and is not predicted.")
    parser.add_argument(
        "--rotation_input_reg_weight",
        type=float,
        default=None,
        help="Deprecated: rotation is not a model input/output.",
    )
    parser.add_argument("--jacobian_weight", type=float, default=0.0, help="Deprecated: Jacobian is loaded from preprocessing and is not predicted.")
    parser.add_argument(
        "--jacobian_input_reg_weight",
        type=float,
        default=None,
        help="Deprecated: Jacobian is not a model input/output.",
    )
    parser.add_argument("--grf_correction_weight", type=float, default=0.0, help="Weight for physics residue loss: m*a = sum(GRF)")
    parser.add_argument(
        "--output_reg_weight",
        type=float,
        default=0.0,
        help="Weight for L1 regularizer on normalized model outputs (encourages small deviations).",
    )

    # Speed/gender window balancing (loss-weighting). Training only; validation is never weighted.
    parser.add_argument(
        "--balance_speed_gender",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="Enable per-window loss weighting to balance gender and up-weight under-represented walking speeds.",
    )
    parser.add_argument(
        "--gender_balance",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Equalize male vs female window mass when balancing (unknown sex stays neutral 1.0).",
    )
    parser.add_argument(
        "--speed_bin_width",
        type=float,
        default=0.05,
        help="Walking-speed histogram bin width in m/s used for speed balancing.",
    )
    parser.add_argument(
        "--speed_weight_power",
        type=float,
        default=0.5,
        help="Speed up-weight exponent: w ~ (1/bin_count)^power (0=off, 0.5=softened, 1=full inverse-frequency).",
    )
    parser.add_argument(
        "--weight_clip_ratio",
        type=float,
        default=3.0,
        help="Clip combined per-window weight to [1/ratio, ratio]; start conservative, raise to weight outlier speeds harder.",
    )

    # Individual DOF weights
    parser.add_argument("--hip_add_r_weight", type=float, default=None)
    parser.add_argument("--knee_r_weight", type=float, default=None)
    parser.add_argument("--ankle_r_weight", type=float, default=None)
    parser.add_argument("--subtalar_r_weight", type=float, default=None)
    parser.add_argument("--hip_add_l_weight", type=float, default=None)
    parser.add_argument("--knee_l_weight", type=float, default=None)
    parser.add_argument("--ankle_l_weight", type=float, default=None)
    parser.add_argument("--subtalar_l_weight", type=float, default=None)
    parser.add_argument("--lumbar_extension_weight", type=float, default=None)
    parser.add_argument("--lumbar_bending_weight", type=float, default=None)
    parser.add_argument("--lumbar_rotation_weight", type=float, default=None)
    
    # Ablation study flags
    parser.add_argument("--magOnOff", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable magnitude weighting in torque loss")
    parser.add_argument("--contactOnOff", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable contact weighting in torque loss")
    parser.add_argument("--use_contact_weighting", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable contact weighting for COP/GRF/Moments loss")
    parser.add_argument("--trim_cop", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable COP trimming")
    parser.add_argument("--UseNoised", nargs="?", const=True, default=False, type=_parse_optional_bool_arg,
                        help="Load noised kinematic input bundles (e.g. *_noised.npy) for model inputs and prediction-side physics while keeping clean files as ground truth.")
    parser.add_argument(
        "--includePelvisEuler",
        "--inlcudePelvisEuler",
        dest="includePelvisEuler",
        nargs="?",
        const=True,
        default=True,
        type=_parse_optional_bool_arg,
        help="If false, drop pelvis_tilt/list/rotation from pos_inputs before building the model input.",
    )
    parser.add_argument("--includeJacobianInput", nargs="?", const=True, default=True, type=_parse_optional_bool_arg,
                        help="Include flattened preprocessed Jacobian [jacp,jacr] as temporal model inputs.")
    parser.add_argument("--NoisedGT", nargs="?", const=True, default=False, type=_parse_optional_bool_arg,
                        help="Use noised ground-truth bundles for COP / calc-frame GT rotations / qfrc_inverse when available.")
    parser.add_argument("--UseGRFNormCOP", nargs="?", const=True, default=False, type=_parse_optional_bool_arg,
                        help="Train the COP head on COP_CalcFrame_GroundAligned_GRFNorm.npy, whose units are (COP/height)*(|GRF|/body_weight).")
    parser.add_argument("--UseOSFiltering", nargs="?", const=True, default=False, type=_parse_optional_bool_arg,
                        help="Train on the OpenSim-filtered (_OSfilt) inputs/targets produced by "
                             "ProcessData --os-filtering (GCVSpline vel/accel instead of 6 Hz Butterworth).")
    parser.add_argument(
        "--use_GRF_NoFilt",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help=(
            "Use ProcessedData/GRF_NoFilt_Trimmed.npy as the GRF training target. "
            "If omitted, the loader uses GRF_NoFilt_Trimmed.npy when present and otherwise falls back to GRF_Cleaned.npy. "
            "Pass false to force GRF_Cleaned.npy."
        ),
    )
    parser.add_argument("--cop_mask", type=lambda x: (str(x).lower() != 'false'), default=True, help="Apply contact-predicted mask to COP/GRF (default: True)")
    parser.add_argument("--refresh_cache", action="store_true", help="Refresh the trial discovery cache")
    parser.add_argument(
        "--exclude_trials",
        type=str,
        default=None,
        help="JSON list of trials/subjects to exclude from the entire run. Entries with a "
             "'/' (e.g. 'OA19/Trial_5') exclude that specific trial; bare entries (e.g. 'OA19') "
             "exclude every trial for that subject.",
    )
    parser.add_argument(
        "--exclude_prefixes",
        type=str,
        default=None,
        help="JSON list of subject-name prefixes to exclude from the entire run. Every trial "
             "whose subject (patient folder) name starts with one of these strings is dropped "
             "(e.g. '[\"SUBJ\", \"OA\", \"Y\"]' drops all SUBJ*, OA*, and Y* subjects).",
    )
    parser.add_argument("--BestModelByTorque", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="If True, best model is selected by weighted torque RMSE instead of val loss")
    parser.add_argument("--BestModel_TorqueWeighting", type=str, default=None,
                        help="JSON dict of torque-score weights for best-model selection. "
                             "Supported keys: hip_flexion, hip_add, knee, ankle, subtalar, "
                             "lumbar_extension, lumbar_bending, lumbar_rotation. "
                             "Example: '{\"hip_flexion\":1.0,\"knee\":2.0,\"lumbar_extension\":0.5}'")

    args = parser.parse_args()
    if not AUXILIARY_MODEL_OUTPUTS_ENABLED:
        args.qfrc_inverse_weight = 0.0
        args.qfrc_inverse_input_reg_weight = 0.0
        args.rotation_weight = 0.0
        args.rotation_input_reg_weight = 0.0
        args.jacobian_weight = 0.0
        args.jacobian_input_reg_weight = 0.0

    # When only one knee weight is provided (for example from a sweep),
    # mirror it so bilateral knee weighting stays synchronized.
    if args.knee_r_weight is None and args.knee_l_weight is not None:
        args.knee_r_weight = args.knee_l_weight
    if args.knee_l_weight is None and args.knee_r_weight is not None:
        args.knee_l_weight = args.knee_r_weight

    try:
        applied_base_config_id = apply_base_model_config(args)
    except ValueError as e:
        parser.error(str(e))
    try:
        validate_prediction_margin(args.window_size, args.prediction_margin_frames)
    except ValueError as e:
        parser.error(str(e))
    if applied_base_config_id:
        print(f"🎛️ Applied base model config: {applied_base_config_id}", flush=True)
        for _key in BASE_MODEL_CONFIG_KEYS:
            print(f"   {_key}={getattr(args, _key)}", flush=True)

    if args.ff_dim is None:
        args.ff_dim = int(args.d_model) * 4
        print(f"ℹ️  --ff_dim not provided; using default ff_dim=4*d_model={args.ff_dim}", flush=True)
    
    global ENABLE_COP_TRIM
    ENABLE_COP_TRIM = args.trim_cop

    wandb_run_name = build_wandb_run_name(args)
    args.output_dir = resolve_output_dir_for_run(args, wandb_run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    resolved_qfrc_inverse_input_reg_weight = (
        args.qfrc_inverse_input_reg_weight
        if args.qfrc_inverse_input_reg_weight is not None
        else args.qfrc_inverse_weight
    )
    resolved_rotation_input_reg_weight = (
        args.rotation_input_reg_weight
        if args.rotation_input_reg_weight is not None
        else args.rotation_weight
    )
    resolved_jacobian_input_reg_weight = 0.0

    if args.qfrc_inverse_input_reg_weight is None:
        print(
            f"ℹ️  qfrc_inverse_input_reg_weight not provided; using qfrc_inverse_weight={resolved_qfrc_inverse_input_reg_weight:g}.",
            flush=True,
        )
    if args.rotation_input_reg_weight is None:
        print(
            f"ℹ️  rotation_input_reg_weight not provided; using rotation_weight={resolved_rotation_input_reg_weight:g}.",
            flush=True,
        )
    if RUNTIME_ENV_APPLIED:
        print("🔒 Applied runtime safety env defaults:", flush=True)
        for _k in sorted(RUNTIME_ENV_APPLIED.keys()):
            print(f"   {_k}={os.environ.get(_k)}", flush=True)
    else:
        print("🔒 Runtime safety env defaults already provided by parent environment.", flush=True)

    # Load wandb key from .env if not provided
    if args.use_wandb and not args.wandb_api_key:
        wandb_key = load_wandb_key_from_env()
        if wandb_key:
            args.wandb_api_key = wandb_key
            print("✅ Loaded wandb key from .env file", flush=True)

    wandb_logger = WandbLogger(enabled=False, project=args.wandb_project, run_name=wandb_run_name)
    if args.use_wandb:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb_tags = [tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()]
        wandb_config = dict(vars(args))
        wandb_config["wandb_tags"] = wandb_tags
        wandb_config["qfrc_inverse_input_reg_weight"] = resolved_qfrc_inverse_input_reg_weight
        wandb_config["rotation_input_reg_weight"] = resolved_rotation_input_reg_weight
        wandb_config["jacobian_input_reg_weight"] = resolved_jacobian_input_reg_weight
        wandb_config["rotation_parameterization"] = ROTATION_PARAMETERIZATION
        wandb_config["rotation_compose_order"] = ROTATION_COMPOSE_ORDER
        wandb_config["rotation_output_dim"] = 0
        try:
            wandb_logger = WandbLogger(
                enabled=True,
                project=args.wandb_project,
                run_name=wandb_run_name,
                config=wandb_config,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                group=args.wandb_group,
                job_type="train",
                tags=wandb_tags,
                dir=args.output_dir,
                resume=args.wandb_resume,
                run_id=args.wandb_run_id,
            )
            if wandb_logger.is_active:
                print("✅ WandB logging enabled", flush=True)
            else:
                print("⚠️  WandB requested but run is inactive. Continuing without WandB.", flush=True)
        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}", flush=True)
            print("   Continuing without WandB logging...", flush=True)
            wandb_logger = WandbLogger(enabled=False, project=args.wandb_project, run_name=wandb_run_name)
    
    print("=" * 70, flush=True)
    print("Training COP/GRF (V5)", flush=True)
    print("=" * 70, flush=True)
    
    # Discover and load data
    print(f"\n🔍 Step 1: Discovering trials in: {args.data_dir}", flush=True)
    
    # Check if directory exists first
    if not os.path.exists(args.data_dir):
        print(f"❌ ERROR: Data directory does not exist: {args.data_dir}", flush=True)
        print(f"   Please check the path in train_single_model.py", flush=True)
        wandb_logger.set_summary({"status": "failed", "failure_reason": "missing_data_dir"})
        wandb_logger.finish()
        return
    
    print(f"   Data directory exists ✓", flush=True)
    print(f"   Looking for ProcessedData subdirectories...", flush=True)
    
    trials = discover_all_trials(
        args.data_dir,
        refresh_cache=args.refresh_cache,
        scan_workers=args.scan_workers,
    )
    print(f"✅ Found {len(trials)} potential trials", flush=True)

    # Exclude user-specified trials/subjects from the entire run (train + val).
    if args.exclude_trials:
        try:
            exclude_list = json.loads(args.exclude_trials)
        except Exception as e:
            print(f"⚠️ Could not parse --exclude_trials as JSON ({e}); ignoring.", flush=True)
            exclude_list = []
        exclude_trial_names = {str(x).strip() for x in exclude_list if "/" in str(x)}
        exclude_subjects = {str(x).strip() for x in exclude_list if "/" not in str(x)}
        if exclude_trial_names or exclude_subjects:
            all_trial_names = {t.get("trial_name") for t in trials}
            all_subjects = {t.get("subject") for t in trials}
            before = len(trials)
            kept, removed = [], []
            for t in trials:
                if t.get("trial_name") in exclude_trial_names or t.get("subject") in exclude_subjects:
                    removed.append(t.get("trial_name", t.get("subject")))
                else:
                    kept.append(t)
            trials = kept
            print(f"🚫 Excluded {before - len(trials)} trials via --exclude_trials "
                  f"({len(exclude_trial_names)} trial rules, {len(exclude_subjects)} subject rules)", flush=True)
            for name in removed:
                print(f"     - {name}", flush=True)
            # Surface exclusion entries that matched nothing (typos / renamed trials).
            unmatched = sorted((exclude_trial_names - all_trial_names)
                               | (exclude_subjects - all_subjects))
            if unmatched:
                print(f"   ⚠️ {len(unmatched)} exclusion entries matched no trials: {unmatched}", flush=True)

    # Exclude whole subject cohorts by name prefix (e.g. "SUBJ", "OA", "Y").
    if args.exclude_prefixes:
        try:
            prefix_list = json.loads(args.exclude_prefixes)
        except Exception as e:
            print(f"⚠️ Could not parse --exclude_prefixes as JSON ({e}); ignoring.", flush=True)
            prefix_list = []
        exclude_prefixes = tuple(str(p).strip() for p in prefix_list if str(p).strip())
        if exclude_prefixes:
            before = len(trials)
            kept, removed = [], []
            for t in trials:
                subject = str(t.get("subject", ""))
                if subject.startswith(exclude_prefixes):
                    removed.append(t.get("trial_name", subject))
                else:
                    kept.append(t)
            trials = kept
            print(f"🚫 Excluded {before - len(trials)} trials via --exclude_prefixes "
                  f"({list(exclude_prefixes)})", flush=True)
            removed_subjects = sorted({
                str(t).split('/')[0] for t in removed
            })
            for name in removed_subjects:
                print(f"     - {name}*", flush=True)

    # Filter trials by length
    print(f"\n🔍 Step 2: Filtering trials shorter than {args.window_size} frames...", flush=True)
    valid_trials = []
    _pos_name = "pos_inputs_noised.npy" if args.UseNoised else "pos_inputs.npy"
    for t in trials:
        # Use cached length if available, else load it (fallback)
        if "length" in t:
            if t["length"] > args.window_size:
                valid_trials.append(t)
        else:
            try:
                pos_path = os.path.join(t["training_data_path"], _pos_name)
                shape = np.load(pos_path, mmap_mode='r').shape
                if shape[0] > args.window_size:
                    valid_trials.append(t)
            except:
                continue
    
    trials = valid_trials
    print(f"✅ Retained {len(trials)} trials after length filtering", flush=True)
    
    if len(trials) == 0:
        print("❌ No valid trials found!", flush=True)
        wandb_logger.set_summary({"status": "failed", "failure_reason": "no_valid_trials"})
        wandb_logger.finish()
        return
    
    # Split into train/val by subject
    print(f"\n🔍 Step 3: Splitting data into train/val sets by SUBJECT...", flush=True)
    rng = jax.random.PRNGKey(42)
    
    # Group trials by subject
    subject_to_trials = {}
    for t in trials:
        s = t["subject"]
        if s not in subject_to_trials:
            subject_to_trials[s] = []
        subject_to_trials[s].append(t)
    
    subjects = sorted(list(subject_to_trials.keys()))
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    fixed_val_requested_subjects = []
    fixed_val_missing_subjects = []
    if args.val_subjects_json:
        val_subjects_source = args.val_subjects_json
        val_subjects_path = Path(val_subjects_source)
        if not val_subjects_path.is_absolute():
            candidate_paths = [
                Path.cwd() / val_subjects_path,
                Path(__file__).resolve().parent / val_subjects_path,
                Path(__file__).resolve().parent.parent / val_subjects_path,
            ]
            val_subjects_path = next((p for p in candidate_paths if p.exists()), candidate_paths[-1])
        try:
            if val_subjects_path.exists():
                with val_subjects_path.open("r", encoding="utf-8") as f:
                    val_subjects_payload = json.load(f)
            else:
                val_subjects_payload = json.loads(val_subjects_source)
        except Exception as exc:
            raise ValueError(f"Failed to load --val_subjects_json={val_subjects_source!r}: {exc}") from exc

        if isinstance(val_subjects_payload, dict):
            fixed_val_requested_subjects = list(val_subjects_payload.get("val_subjects", []))
        elif isinstance(val_subjects_payload, list):
            fixed_val_requested_subjects = list(val_subjects_payload)
        else:
            raise ValueError("--val_subjects_json must contain a JSON list or an object with a 'val_subjects' list.")

        fixed_val_requested_subjects = [str(s) for s in fixed_val_requested_subjects]

    if len(subjects) >= 2:
        if fixed_val_requested_subjects:
            requested_val_set = set(fixed_val_requested_subjects)
            available_subjects = set(subjects)
            fixed_val_missing_subjects = sorted(requested_val_set - available_subjects)
            val_subs = [s for s in subjects if s in requested_val_set]
            train_subs = [s for s in subjects if s not in requested_val_set]
            if not val_subs:
                raise ValueError(
                    "--val_subjects_json did not match any discovered subjects after filtering/exclusions."
                )
            if not train_subs:
                raise ValueError("--val_subjects_json matched every subject; no training subjects remain.")
            print(
                f"   Using fixed validation subject set: {len(train_subs)} train and {len(val_subs)} val subjects",
                flush=True,
            )
            if fixed_val_missing_subjects:
                print(
                    f"   ⚠️ {len(fixed_val_missing_subjects)} requested validation subjects were not present in this dataset: "
                    f"{', '.join(fixed_val_missing_subjects[:8])}"
                    f"{'...' if len(fixed_val_missing_subjects) > 8 else ''}",
                    flush=True,
                )
        else:
            n_train_subs = max(1, int(0.80 * len(subjects))) # 80/20 split on subjects
            train_subs = subjects[:n_train_subs]
            val_subs = subjects[n_train_subs:]
        
        train_trials = []
        for s in train_subs:
            train_trials.extend(subject_to_trials[s])
            
        val_trials = []
        for s in val_subs:
            val_trials.extend(subject_to_trials[s])
            
        print(f"   Split {len(subjects)} subjects into {len(train_subs)} train and {len(val_subs)} val", flush=True)
        print(f"   Train Subjects: {', '.join(train_subs[:5])}...", flush=True)
        print(f"   Val Subjects:   {', '.join(val_subs[:5])}...", flush=True)
    else:
        # Fallback if only 1 subject
        print("⚠️  Only 1 subject found. Splitting by individual trials instead.", flush=True)
        # Seed NumPy for consistency
        np.random.seed(42)
        np.random.shuffle(trials)
        n_train = max(1, int(0.75 * len(trials)))
        train_trials = trials[:n_train]
        val_trials = trials[n_train:]
        if len(val_trials) == 0 and len(train_trials) > 1:
            val_trials = [train_trials.pop()]
        elif len(val_trials) == 0:
            val_trials = train_trials
    
    print(f"✅ Train: {len(train_trials)} trials, Val: {len(val_trials)} trials", flush=True)
    
    # Save train/val split to file for inspection
    split_info = {
        "train_trials": train_trials,
        "val_trials": val_trials,
        "n_train": len(train_trials),
        "n_val": len(val_trials),
        "split_mode": "fixed_val_subjects" if fixed_val_requested_subjects else "subject_random_80_20",
        "fixed_val_requested_subjects": fixed_val_requested_subjects,
        "fixed_val_missing_subjects": fixed_val_missing_subjects,
    }
    split_path = os.path.join(args.output_dir, "train_val_split.json")
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"📝 Saved train/val split to: {split_path}", flush=True)
    
    # 🚀 MEMORY-EFFICIENT DATA LOADING: Use TrialDataLoader instead of loading all data
    print("\n Step 4: Creating memory-efficient data loaders...", flush=True)
    print("   This will scan trials to count windows (fast, doesn't load data yet)...", flush=True)
    
    train_loader = TrialDataLoader(
        train_trials,
        window_size=args.window_size, 
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=True,
        trim_cop=args.trim_cop,
        use_noised=args.UseNoised,
        noised_gt=args.NoisedGT,
        use_grf_norm_cop=args.UseGRFNormCOP,
        use_grf_nofilt=args.use_GRF_NoFilt,
        use_os_filtering=args.UseOSFiltering,
        include_pelvis_euler=args.includePelvisEuler,
        include_jacobian_input=args.includeJacobianInput,
        prediction_margin_frames=args.prediction_margin_frames,
        balance_speed_gender=args.balance_speed_gender,
        gender_balance=args.gender_balance,
        speed_bin_width=args.speed_bin_width,
        speed_weight_power=args.speed_weight_power,
        weight_clip_ratio=args.weight_clip_ratio,
    )
    print(f"✅ Train loader created: {train_loader.total_windows} windows", flush=True)
    if args.balance_speed_gender:
        save_balance_report(train_loader.balance_stats, args.output_dir)
    
    val_loader = TrialDataLoader(
        val_trials,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle validation
        trim_cop=args.trim_cop,
        use_noised=args.UseNoised,
        noised_gt=args.NoisedGT,
        use_grf_norm_cop=args.UseGRFNormCOP,
        use_grf_nofilt=args.use_GRF_NoFilt,
        use_os_filtering=args.UseOSFiltering,
        include_pelvis_euler=args.includePelvisEuler,
        include_jacobian_input=args.includeJacobianInput,
        prediction_margin_frames=args.prediction_margin_frames,
    )
    print(f"✅ Val loader created: {val_loader.total_windows} windows", flush=True)
    
    print(f"\n📊 Train: {train_loader.total_windows} windows from {len(train_trials)} trials", flush=True)
    print(f"📊 Val: {val_loader.total_windows} windows from {len(val_trials)} trials", flush=True)
    
    # Compute normalizers from training data sample
    print("\n� Step 5: Computing normalizers from training data sample...", flush=True)
    print("   Loading up to 100 batches to compute statistics...", flush=True)
    print("   (This is where actual data loading begins - may take a few minutes)", flush=True)
    normalizers = compute_normalizers_from_loader(train_loader, max_batches=100)
    print("✅ Normalizers computed successfully", flush=True)
    
    resume_params = None
    resume_best_val = None
    if args.resume_checkpoint:
        print(f"\n🔍 Step 6: Loading checkpoint from {args.resume_checkpoint}...", flush=True)
        if os.path.isfile(args.resume_checkpoint):
            try:
                with open(args.resume_checkpoint, "rb") as f:
                    ckpt = pickle.load(f)
                ckpt_rotation_output_dim = int(ckpt.get("rotation_output_dim", -1))
                ckpt_rotation_parameterization = ckpt.get("rotation_parameterization")
                ckpt_rotation_compose_order = ckpt.get("rotation_compose_order")
                if AUXILIARY_MODEL_OUTPUTS_ENABLED and (
                    ckpt_rotation_output_dim != ROTATION_OUTPUT_DIM
                    or ckpt_rotation_parameterization != ROTATION_PARAMETERIZATION
                    or ckpt_rotation_compose_order != ROTATION_COMPOSE_ORDER
                ):
                    raise ValueError(
                        "Checkpoint rotation head is incompatible with the current residual rotation setup. "
                        f"Expected output_dim={ROTATION_OUTPUT_DIM}, parameterization={ROTATION_PARAMETERIZATION}, "
                        f"compose_order={ROTATION_COMPOSE_ORDER}, but found output_dim={ckpt_rotation_output_dim}, "
                        f"parameterization={ckpt_rotation_parameterization}, compose_order={ckpt_rotation_compose_order}. "
                        "Please start from scratch or resume from a checkpoint created with the residual rotation model."
                    )
                resume_params = ckpt.get("params", None)
                if "normalizers" in ckpt:
                    loaded_normalizers = ckpt["normalizers"]
                    missing_norm_messages = []
                    if "qfrc_inverse" not in loaded_normalizers and "qfrc_inverse" in normalizers:
                        loaded_normalizers = dict(loaded_normalizers)
                        loaded_normalizers["qfrc_inverse"] = normalizers["qfrc_inverse"]
                        missing_norm_messages.append("qfrc_inverse")
                    if missing_norm_messages:
                        print(
                            "   ⚠️  Resume checkpoint was missing normalizers for "
                            + ", ".join(missing_norm_messages)
                            + "; using the freshly computed version(s).",
                            flush=True,
                        )
                    normalizers = loaded_normalizers
                    print(f"   ✅ Loaded normalizers from checkpoint", flush=True)
                resume_best_val = ckpt.get("best_val_loss", None)
                print(f"✅ Loaded checkpoint successfully", flush=True)
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}", flush=True)
        else:
            print(f"⚠️  Checkpoint file not found. Training from scratch.", flush=True)
    else:
        print(f"\n🔍 Step 6: No checkpoint specified - training from scratch", flush=True)
    
    # Determine input dimension from a sample batch
    print("\n� Step 7: Getting sample batch to determine input dimensions...", flush=True)
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["input"].shape[-1]
    static_dim = sample_batch["static_context"].shape[-1]
    print(f"✅ Sample batch obtained: input_dim = {input_dim}, static_dim = {static_dim}", flush=True)

    input_layout = infer_input_feature_layout_from_loader(
        train_loader,
        include_pelvis_euler=bool(args.includePelvisEuler),
        include_jacobian_input=bool(args.includeJacobianInput),
    )
    if input_layout is not None:
        print("   Temporal input blocks:", flush=True)
        for block in input_layout["blocks"]:
            print(f"      - {block['name']}: {block['dim']}", flush=True)
        print(f"   contactBoolean in model input: {input_layout['contact_boolean_is_input']}", flush=True)
        print(
            f"   Input layout total from blocks: {input_layout['total_dim']} "
            f"(sample trial: {input_layout['sample_trial']})",
            flush=True,
        )
        if int(input_layout["total_dim"]) != int(input_dim):
            raise ValueError(
                "Input feature layout mismatch: "
                f"loader blocks sum to {input_layout['total_dim']} but sampled input_dim is {input_dim}. "
                "Check TrialDataLoader input assembly and train.py assumptions."
            )
    
    # Model
    print("\n🔍 Step 8: Creating model...", flush=True)
    qfrc_inverse_output_dim = 0
    rotation_output_dim = 0
    jacobian_output_dim = 0
    total_output_dim = STANDARD_OUTPUT_DIM + qfrc_inverse_output_dim + rotation_output_dim + jacobian_output_dim
    model = KinematicsToCOPGRFMoments(
        input_dim=input_dim,
        static_dim=static_dim,
        output_dim=total_output_dim,
        d_model=args.d_model,
        num_heads=4,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout_rate=args.dropout_rate,
    )
    print(f"✅ Model: KinematicsToCOPGRFMoments (Transformer)", flush=True)
    print(f"   Input: {input_dim} (temporal features incl. non-auxiliary geometry; contact removed)", flush=True)
    print(f"   Output: {total_output_dim} (COP:4 + GRF:6 + Moments:2 + Contact:2)", flush=True)
    print(f"   Architecture: d_model={args.d_model}, layers={args.num_layers}, ff_dim={args.ff_dim}", flush=True)
    print(
        f"   Windowing: window_size={args.window_size}, stride={args.stride}, "
        f"prediction_margin_frames={args.prediction_margin_frames}",
        flush=True,
    )
    print(
        f"   Data source flags: UseNoised={args.UseNoised}, NoisedGT={args.NoisedGT}, "
        f"UseGRFNormCOP={args.UseGRFNormCOP}, use_GRF_NoFilt={args.use_GRF_NoFilt}, "
        f"includeJacobianInput={args.includeJacobianInput}",
        flush=True,
    )
    
    # Construct DOF weights dictionary if any are provided
    dof_weights_dict = None
    dof_args = {
        7: args.hip_add_r_weight,
        9: args.knee_r_weight,
        10: args.ankle_r_weight,
        11: args.subtalar_r_weight,
        14: args.hip_add_l_weight,
        16: args.knee_l_weight,
        17: args.ankle_l_weight,
        18: args.subtalar_l_weight,
        20: args.lumbar_extension_weight,
        21: args.lumbar_bending_weight,
        22: args.lumbar_rotation_weight,
    }
    
    if any(v is not None for v in dof_args.values()):
        dof_weights_dict = {
            6: 1.0, 7: 1.0, 9: 1.0, 10: 1.0, 11: 1.0,
            13: 1.0, 14: 1.0, 16: 1.0, 17: 1.0, 18: 1.0,
            20: 1.0, 21: 1.0, 22: 1.0,
        }
        for k, v in dof_args.items():
            if v is not None:
                dof_weights_dict[k] = v
        print(f"Using custom DOF weights: {dof_weights_dict}", flush=True)

    # Training setup
    print("\n🔍 Step 9: Initializing training state...", flush=True)
    rng, init_rng = jax.random.split(rng)
    print("   Creating train state (this compiles the model - may take a minute)...", flush=True)
    state = create_train_state(init_rng, model, (1, args.window_size, input_dim), (1, static_dim), args.learning_rate, args.weight_decay)
    if resume_params is not None:
        state = state.replace(params=resume_params)
        print("✅ Initialized train state from checkpoint parameters", flush=True)
    else:
        print("✅ Initialized fresh train state", flush=True)
    
    print("   Creating JIT-compiled train and eval functions...", flush=True)
    train_step = make_train_step(
        normalizers,
        args.use_contact_weighting,
        args.magOnOff,
        args.contactOnOff,
        OnlySuperviseStance,
        args.contact_weight_multiplier,
        args.magWeight,
        args.epochs,
        dof_weights_dict,
        cop_mask=args.cop_mask,
        use_grf_norm_cop=args.UseGRFNormCOP,
    )
    eval_step = make_eval_step(
        normalizers,
        args.use_contact_weighting,
        args.magOnOff,
        args.contactOnOff,
        OnlySuperviseStance,
        args.contact_weight_multiplier,
        args.magWeight,
        args.epochs,
        dof_weights_dict,
        cop_mask=args.cop_mask,
        use_grf_norm_cop=args.UseGRFNormCOP,
    )
    print("✅ Train and eval functions created", flush=True)
    
    loss_weights = {
        "cop": args.cop_weight,
        "grf": args.grf_weight,
        "moments": args.moments_weight,
        "qfrc_inverse": args.qfrc_inverse_weight,
        "qfrc_inverse_input_reg": resolved_qfrc_inverse_input_reg_weight,
        "rotation": args.rotation_weight,
        "rotation_input_reg": resolved_rotation_input_reg_weight,
        "jacobian": 0.0,
        "jacobian_input_reg": resolved_jacobian_input_reg_weight,
        "contact": args.contact_weight,
        "torque": args.torque_weight,
        "grf_correction": args.grf_correction_weight,
        "output_reg": 0.0,
    }
    if args.output_reg_weight != 0.0:
        print(
            "ℹ️  output_reg_weight is disabled because the model predicts direct targets.",
            flush=True,
        )
    
    steps_per_epoch = len(train_loader)
    val_steps = len(val_loader)
    startup_log = {
        "data/num_trials_total": int(len(trials)),
        "data/num_train_trials": int(len(train_trials)),
        "data/num_val_trials": int(len(val_trials)),
        "data/num_train_subjects": int(len({t.get("subject", "") for t in train_trials})),
        "data/num_val_subjects": int(len({t.get("subject", "") for t in val_trials})),
        "data/train_windows": int(train_loader.total_windows),
        "data/val_windows": int(val_loader.total_windows),
        "data/window_size": int(args.window_size),
        "data/stride": int(args.stride),
        "data/prediction_margin_frames": int(args.prediction_margin_frames),
        "model/input_dim": int(input_dim),
        "model/static_dim": int(static_dim),
        "model/output_dim": int(total_output_dim),
        "model/qfrc_inverse_output_dim": int(qfrc_inverse_output_dim),
        "model/rotation_output_dim": int(rotation_output_dim),
        "model/rotation_parameterization": ROTATION_PARAMETERIZATION,
        "model/rotation_compose_order": ROTATION_COMPOSE_ORDER,
        "model/jacobian_output_dim": int(jacobian_output_dim),
        "train/steps_per_epoch": int(steps_per_epoch),
        "val/steps_per_epoch": int(val_steps),
    }
    if input_layout is not None:
        startup_log["model/input_layout_total_dim"] = int(input_layout["total_dim"])
        startup_log["model/input_layout_sample_trial"] = str(input_layout["sample_trial"])
        for block in input_layout["blocks"]:
            startup_log[f"model/input_block_dim/{block['name']}"] = int(block["dim"])
    wandb_logger.log(startup_log, step=0)

    # Map loader trial indices to readable subject/trial names
    train_trial_name_map = {}
    for idx, (trial_info, _n_windows) in enumerate(train_loader.trial_window_counts):
        trial_name = trial_info.get("trial_name")
        if not trial_name:
            subject = trial_info.get("subject", "unknown_subject")
            trial = trial_info.get("trial", "unknown_trial")
            trial_name = f"{subject}/{trial}"
        train_trial_name_map[idx] = trial_name

    val_trial_name_map = {}
    for idx, (trial_info, _n_windows) in enumerate(val_loader.trial_window_counts):
        trial_name = trial_info.get("trial_name")
        if not trial_name:
            subject = trial_info.get("subject", "unknown_subject")
            trial = trial_info.get("trial", "unknown_trial")
            trial_name = f"{subject}/{trial}"
        val_trial_name_map[idx] = trial_name

    # Channels used for per-epoch validation outlier reporting
    outlier_channel_defs = [
        ("COP Rx", "m"),
        ("COP Rz", "m"),
        ("COP Lx", "m"),
        ("COP Lz", "m"),
        ("GRF Rx", "N"),
        ("GRF Ry", "N"),
        ("GRF Rz", "N"),
        ("GRF Lx", "N"),
        ("GRF Ly", "N"),
        ("GRF Lz", "N"),
        ("Moment Rz", "Nm"),
        ("Moment Lz", "Nm"),
    ]
    
    best_val_loss = resume_best_val if resume_best_val is not None else float("inf")

    # Parse torque-weighting dict for best-model-by-torque selection.
    # Bilateral groups average right/left RMSE; lumbar groups use single-DOF RMSE.
    _BILATERAL_TAU_MAP = {
        "hip_flexion": ("R Hip Flexion", "L Hip Flexion"),
        "hip_add":  ("R Hip Add", "L Hip Add"),
        "knee":     ("R Knee",    "L Knee"),
        "ankle":    ("R Ankle",   "L Ankle"),
        "subtalar": ("R Subtalar","L Subtalar"),
    }
    _BEST_MODEL_TAU_GROUPS = {
        **_BILATERAL_TAU_MAP,
        "lumbar_extension": ("Lumbar Extension",),
        "lumbar_bending": ("Lumbar Bending",),
        "lumbar_rotation": ("Lumbar Rotation",),
    }
    _BEST_MODEL_TAU_KEY_ALIASES = {
        "hip_adduction": "hip_add",
    }
    _STANCE_MAE_TAU_DOFS = {
        "R Hip Flexion": 6,
        "R Hip Adduction": 7,
        "R Hip Rotation": 8,
        "R Knee": 9,
        "R Ankle": 10,
        "R Subtalar": 11,
        "L Hip Flexion": 13,
        "L Hip Adduction": 14,
        "L Hip Rotation": 15,
        "L Knee": 16,
        "L Ankle": 17,
        "L Subtalar": 18,
    }
    _STANCE_MAE_BILATERAL_TAU_MAP = {
        "hip_flexion": ("R Hip Flexion", "L Hip Flexion"),
        "hip_adduction": ("R Hip Adduction", "L Hip Adduction"),
        "hip_rotation": ("R Hip Rotation", "L Hip Rotation"),
        "knee": ("R Knee", "L Knee"),
        "ankle": ("R Ankle", "L Ankle"),
        "subtalar": ("R Subtalar", "L Subtalar"),
    }
    _STANCE_MAE_DISPLAY = {
        "hip_flexion": "Hip Flexion",
        "hip_adduction": "Hip Adduction",
        "hip_rotation": "Hip Rotation",
        "knee": "Knee",
        "ankle": "Ankle",
        "subtalar": "Subtalar",
    }
    _BILATERAL_GRF_AXIS_MAP = {
        "x": (0, 3),  # Rx, Lx
        "y": (1, 4),  # Ry, Ly
        "z": (2, 5),  # Rz, Lz
    }
    _default_tau_weights = {key: 1.0 for key in _BEST_MODEL_TAU_GROUPS}
    if args.BestModel_TorqueWeighting:
        try:
            _tau_weights_raw = json.loads(args.BestModel_TorqueWeighting)
            if not isinstance(_tau_weights_raw, dict):
                raise ValueError("BestModel_TorqueWeighting must decode to a JSON object")
            _tau_weights = dict(_default_tau_weights)
            for _raw_key, _value in _tau_weights_raw.items():
                _canonical_key = _BEST_MODEL_TAU_KEY_ALIASES.get(str(_raw_key), str(_raw_key))
                if _canonical_key in _default_tau_weights:
                    _tau_weights[_canonical_key] = float(_value)
                else:
                    print(
                        f"⚠️  Ignoring unknown BestModel_TorqueWeighting key '{_raw_key}'.",
                        flush=True,
                    )
            print(f"   Torque weighting for best-model selection: {_tau_weights}", flush=True)
        except Exception as _e:
            print(f"⚠️  Could not parse BestModel_TorqueWeighting JSON: {_e}. Using equal weights.", flush=True)
            _tau_weights = _default_tau_weights
    else:
        _tau_weights = _default_tau_weights

    best_torque_score = float("inf")  # lower is better (weighted RMSE sum)
    
    # History for plotting
    train_loss_history = []
    val_loss_history = []
    train_component_history = {
        "cop_loss": [], "grf_loss": [], "moments_loss": [], "qfrc_inverse_loss": [], "qfrc_inverse_input_reg_loss": [],
        "rotation_loss": [], "rotation_input_reg_loss": [], "jacobian_loss": [], "jacobian_input_reg_loss": [], "contact_loss": [],
        "torque_loss": [], "torque_cop_effect_loss": [], "torque_grf_effect_loss": [], "grf_correction_loss": [], "output_reg_loss": [],
    }
    val_component_history = {
        "cop_loss": [], "grf_loss": [], "moments_loss": [], "qfrc_inverse_loss": [], "qfrc_inverse_input_reg_loss": [],
        "rotation_loss": [], "rotation_input_reg_loss": [], "jacobian_loss": [], "jacobian_input_reg_loss": [], "contact_loss": [],
        "torque_loss": [], "torque_cop_effect_loss": [], "torque_grf_effect_loss": [], "grf_correction_loss": [], "output_reg_loss": [],
    }
    train_rmse_history = {"cop": [], "grf": [], "moments": []}
    val_rmse_history = {"cop": [], "grf": [], "moments": []}
    best_model_epoch = None
    best_predictions_plot_path = None
    final_prediction_plot_path = None
    last_val_outlier_plot_path = None
    best_epoch_snapshot = None
    last_epoch_snapshot = None
    last_epoch_torque_score = float("nan")
    saved_epoch_checkpoints: List[str] = []
    
    print(f"\n{'='*70}", flush=True)
    print(f"🚀 STARTING TRAINING", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Steps per epoch: {steps_per_epoch}", flush=True)
    print(f"Total training steps: {steps_per_epoch * args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Learning rate: {args.learning_rate}", flush=True)
    print(f"Best val loss so far: {best_val_loss:.4f}" if best_val_loss != float("inf") else "Best val loss: N/A (fresh training)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n⏱️  First epoch will be slower (JIT compilation + data loading)", flush=True)
    print(f"⏱️  Subsequent epochs should be faster\n", flush=True)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'='*70}", flush=True)
        print(f"📊 EPOCH {epoch}/{args.epochs}", flush=True)
        print(f"{'='*70}", flush=True)
        
        train_metrics = {
            "cop_loss": 0, "grf_loss": 0, "moments_loss": 0, "qfrc_inverse_loss": 0, "qfrc_inverse_input_reg_loss": 0,
            "rotation_loss": 0, "rotation_input_reg_loss": 0, "jacobian_loss": 0, "jacobian_input_reg_loss": 0, "contact_loss": 0,
            "torque_loss": 0, "torque_cop_effect_loss": 0, "torque_grf_effect_loss": 0, "grf_correction_loss": 0, "output_reg_loss": 0, "total_loss": 0
        }
        _KEY_TAU_DOFS = {
            'R Hip Flexion': 6, 'R Hip Add': 7, 'R Knee': 9, 'R Ankle': 10, 'R Subtalar': 11,
            'L Hip Flexion': 13, 'L Hip Add': 14, 'L Knee': 16, 'L Ankle': 17, 'L Subtalar': 18,
            'Lumbar Extension': 20, 'Lumbar Bending': 21, 'Lumbar Rotation': 22,
        }
        _KEY_TAU_DISPLAY = {name: name for name in _KEY_TAU_DOFS}
        _train_tau_mae_pct_sum = {k: 0.0 for k in _KEY_TAU_DOFS}
        _train_tau_mae_pct_count = {k: 0 for k in _KEY_TAU_DOFS}
        _train_stance_tau_mae_pct_sum = {k: 0.0 for k in _STANCE_MAE_TAU_DOFS}
        _train_stance_tau_mae_pct_count = {k: 0 for k in _STANCE_MAE_TAU_DOFS}
        _train_grf_mae_pct_bw_sum = {axis: 0.0 for axis in _BILATERAL_GRF_AXIS_MAP}
        _train_grf_mae_pct_bw_count = {axis: 0 for axis in _BILATERAL_GRF_AXIS_MAP}
        _train_cop_sumsq = np.zeros(4, dtype=np.float64)
        _train_grf_sumsq = np.zeros(6, dtype=np.float64)
        _train_mom_sumsq = np.zeros(2, dtype=np.float64)
        _train_frames = 0
        
        # Train loop - iterate through data loader
        print(f"🔄 Training phase: Processing {steps_per_epoch} batches...", flush=True)
        train_step_count = 0
        last_train_batch = None
        last_train_pred = None
        nonfinite_reported = False
        
        # Performance tracking
        epoch_step_start = time.time()
        
        # Print progress every N steps
        # For large datasets, let's print more frequently (every 50 steps or 2% of epoch)
        log_every = min(50, max(1, steps_per_epoch // 50))
        
        for batch in train_loader:
            # Keep raw batch for visualization
            raw_batch = batch

            # Normalize batch
            batch = normalize_batch(batch, normalizers)
            
            # Record time for step duration tracking
            step_start = time.time()
            
            rng, dropout_rng = jax.random.split(rng)
            
            if train_step_count == 0:
                print("   (Note: First step includes JAX JIT compilation, which can take several minutes...)", flush=True)
            
            prev_params = state.params
            state, metrics, step_pred_train, step_debug = train_step(state, batch, loss_weights, dropout_rng, float(epoch))

            if bool(step_debug.get("update_skipped", False)):
                skip_reasons = {
                    "loss_nonfinite": bool(step_debug.get("loss_nonfinite", False)),
                    "metrics_nonfinite": bool(step_debug.get("metrics_nonfinite", False)),
                    "pred_nonfinite": bool(step_debug.get("pred_nonfinite", False)),
                    "grads_nonfinite": bool(step_debug.get("grads_nonfinite", False)),
                    "updated_params_nonfinite": bool(step_debug.get("updated_params_nonfinite", False)),
                    "updated_opt_state_nonfinite": bool(step_debug.get("updated_opt_state_nonfinite", False)),
                }
                print(
                    "   ⚠️  Skipped a parameter update to keep the model state finite: "
                    + ", ".join(f"{k}={v}" for k, v in skip_reasons.items()),
                    flush=True,
                )
            
            # Wait for computation to finish for accurate timing (optional but helpful for first step)
            if train_step_count == 0:
                jax.block_until_ready(state.params)
                jit_time = time.time() - step_start
                print(f"   (JIT compilation complete in {jit_time:.1f}s)", flush=True)

            metric_values = {k: float(metrics[k]) for k in train_metrics}
            if (not nonfinite_reported) and any(not np.isfinite(v) for v in metric_values.values()):
                nonfinite_reported = True
                _report_first_nonfinite_training_step(
                    train_step_count + 1,
                    raw_batch,
                    batch,
                    metrics,
                    step_pred_train,
                    normalizers,
                    trial_name_map=train_trial_name_map,
                )
                debug_values = {
                    "grad_global_norm": float(step_debug["grad_global_norm"]),
                    "loss_nonfinite": bool(step_debug["loss_nonfinite"]),
                    "metrics_nonfinite": bool(step_debug["metrics_nonfinite"]),
                    "pred_nonfinite": bool(step_debug["pred_nonfinite"]),
                    "params_nonfinite_before_update": bool(step_debug["params_nonfinite_before_update"]),
                    "grads_nonfinite": bool(step_debug["grads_nonfinite"]),
                    "updated_params_nonfinite": bool(step_debug["updated_params_nonfinite"]),
                    "updated_opt_state_nonfinite": bool(step_debug["updated_opt_state_nonfinite"]),
                    "update_skipped": bool(step_debug["update_skipped"]),
                    "params_nonfinite_after_update": bool(step_debug["params_nonfinite_after_update"]),
                    "opt_state_nonfinite_after_update": bool(step_debug["opt_state_nonfinite_after_update"]),
                }
                print(
                    "   Update diagnostics: "
                    + ", ".join(f"{k}={v}" for k, v in debug_values.items()),
                    flush=True,
                )
                prev_param_detail = _tree_first_nonfinite_detail(prev_params, "params_before_update")
                if prev_param_detail is not None:
                    print(prev_param_detail, flush=True)
                new_param_detail = _tree_first_nonfinite_detail(state.params, "params_after_update")
                if new_param_detail is not None:
                    print(new_param_detail, flush=True)
                opt_state_detail = _tree_first_nonfinite_detail(state.opt_state, "opt_state_after_update")
                if opt_state_detail is not None:
                    print(opt_state_detail, flush=True)
                raise FloatingPointError(
                    f"Non-finite training values detected at step {train_step_count + 1}. See diagnostics above."
                )
            
            for k in train_metrics:
                train_metrics[k] += metric_values[k]

            # --- Full-train-set stance-only diagnostics (GRF MAE%BW + torque MAE%) ---
            if "trial_idx" in raw_batch:
                try:
                    pred_np = np.array(step_pred_train)
                    static_np = np.array(raw_batch["static_context"])  # (B, 8)
                    h_batch = static_np[:, 0:1, None]  # (B, 1, 1)
                    m_batch = static_np[:, 1:2, None]  # (B, 1, 1)
                    qfrc_inverse_output_dim = _infer_qfrc_inverse_output_dim(batch=raw_batch, normalizers=normalizers)
                    cop_pred_raw, grf_pred_raw, moments_pred_raw, contact_pred_raw, _qfrc_inverse_pred_raw, _rotation_pred_raw, _jacobian_pred_raw = split_model_predictions(
                        pred_np,
                        qfrc_inverse_output_dim=qfrc_inverse_output_dim,
                        rotation_output_dim=0,
                    )
                    qfrc_inverse_pred_phys, rotation_pred_phys = decode_auxiliary_predictions(
                        pred_np,
                        raw_batch,
                        normalizers,
                        xp=np,
                    )

                    cop_pred_ratio = normalizers["cop"].unnormalize(cop_pred_raw)
                    grf_pred_ratio = normalizers["grf"].unnormalize(grf_pred_raw)
                    moments_pred_ratio = normalizers["moments"].unnormalize(moments_pred_raw)

                    grf_pred_phys = grf_pred_ratio * m_batch * 9.8067
                    cop_pred_phys = decode_cop_signal_to_length(
                        cop_pred_ratio,
                        grf_pred_ratio,
                        h_batch,
                        use_grf_norm_cop=args.UseGRFNormCOP,
                        contact_probability=contact_pred_raw if args.UseGRFNormCOP else None,
                        xp=np,
                    )
                    moments_pred_phys = moments_pred_ratio * m_batch * h_batch * 9.8067

                    grf_gt_phys = np.array(raw_batch["grf"]) * m_batch * 9.8067
                    cop_gt_phys = decode_cop_signal_to_length(
                        np.array(raw_batch["cop"]),
                        np.array(raw_batch["grf"]),
                        h_batch,
                        use_grf_norm_cop=args.UseGRFNormCOP,
                        xp=np,
                    )
                    moments_gt_phys = np.array(raw_batch["moments"]) * m_batch * h_batch * 9.8067
                    _valid_mask = _extract_batched_frame_mask(
                        np.array(raw_batch["supervision_mask"]) if "supervision_mask" in raw_batch else None,
                        cop_pred_phys.shape[0],
                        cop_pred_phys.shape[1],
                    )

                    if args.cop_mask and pred_np.shape[-1] >= STANDARD_OUTPUT_DIM:
                        contact_prob = pred_np[..., CONTACT_SLICE]
                        mask_r = (contact_prob[..., 0:1] > 0.5).astype(cop_pred_phys.dtype)
                        mask_l = (contact_prob[..., 1:2] > 0.5).astype(cop_pred_phys.dtype)
                        cop_pred_phys = np.concatenate([
                            cop_pred_phys[..., 0:2] * mask_r,
                            cop_pred_phys[..., 2:4] * mask_l,
                        ], axis=-1)
                        grf_pred_phys = np.concatenate([
                            grf_pred_phys[..., 0:3] * mask_r,
                            grf_pred_phys[..., 3:6] * mask_l,
                        ], axis=-1)
                        moments_pred_phys = np.concatenate([
                            moments_pred_phys[..., 0:1] * mask_r,
                            moments_pred_phys[..., 1:2] * mask_l,
                        ], axis=-1)

                    # Full-train-set COP/GRF/Moment RMSE accumulation
                    cop_err = cop_pred_phys - cop_gt_phys
                    grf_err = grf_pred_phys - grf_gt_phys
                    mom_err = moments_pred_phys - moments_gt_phys
                    _train_valid = _valid_mask[..., None]
                    _train_cop_sumsq += np.sum((cop_err ** 2) * _train_valid, axis=(0, 1))
                    _train_grf_sumsq += np.sum((grf_err ** 2) * _train_valid, axis=(0, 1))
                    _train_mom_sumsq += np.sum((mom_err ** 2) * _train_valid, axis=(0, 1))
                    _train_frames += int(np.sum(_valid_mask))

                    _contact_bool = np.array(raw_batch["contactBoolean"])  # (B, T, 2)
                    _stance_r = (_contact_bool[..., 0] > 0.5) & _valid_mask
                    _stance_l = (_contact_bool[..., 1] > 0.5) & _valid_mask

                    # GRF MAE in %BW: abs((pred/(m*g)) - (gt/(m*g))) * 100, stance-only
                    _norm_mg = np.maximum(m_batch * 9.8067, 1e-8)
                    _grf_abs_pct_bw_err = np.abs((grf_pred_phys / _norm_mg) - (grf_gt_phys / _norm_mg)) * 100.0
                    for _axis, (_ri, _li) in _BILATERAL_GRF_AXIS_MAP.items():
                        _train_grf_mae_pct_bw_sum[_axis] += float(
                            np.sum(_grf_abs_pct_bw_err[:, :, _ri] * _stance_r)
                            + np.sum(_grf_abs_pct_bw_err[:, :, _li] * _stance_l)
                        )
                        _train_grf_mae_pct_bw_count[_axis] += int(np.sum(_stance_r) + np.sum(_stance_l))

                    _full_mom_pred = np.array(compute_full_external_moments(
                        jnp.array(cop_pred_phys),
                        jnp.array(grf_pred_phys),
                        jnp.array(moments_pred_phys),
                        raw_batch["ankle_heights"],
                        jnp.array(rotation_pred_phys),
                    ))
                    _jacp_pred, _jacr_pred = select_torque_jacobians(
                        pred_np,
                        raw_batch,
                        normalizers,
                        xp=np,
                    )
                    _tau_grf_pred = np.array(compute_tau_grf_from_predictions(
                        jnp.array(grf_pred_phys),
                        jnp.array(_full_mom_pred),
                        jnp.array(_jacp_pred),
                        jnp.array(_jacr_pred),
                    ))
                    _tau_pred = np.array(qfrc_inverse_pred_phys) - _tau_grf_pred
                    _tau_gt = np.array(_full_id_target_from_batch(raw_batch, xp=np))

                    # Torque MAE% in BW*H-normalized space: abs((pred/(mgh)) - (gt/(mgh))) * 100.
                    _norm_mgh = np.maximum(m_batch * 9.8067 * h_batch, 1e-8)
                    _tau_abs_pct_err = np.abs((_tau_pred / _norm_mgh) - (_tau_gt / _norm_mgh)) * 100.0
                    for _name, _di in _KEY_TAU_DOFS.items():
                        _stance_mask = _torque_stance_mask_for_name(_name, _stance_r, _stance_l)
                        _train_tau_mae_pct_sum[_name] += float(np.sum(_tau_abs_pct_err[:, :, _di] * _stance_mask))
                        _train_tau_mae_pct_count[_name] += int(np.sum(_stance_mask))
                    for _name, _di in _STANCE_MAE_TAU_DOFS.items():
                        _stance_mask = _stance_r if _name.startswith("R ") else _stance_l
                        _train_stance_tau_mae_pct_sum[_name] += float(
                            np.sum(_tau_abs_pct_err[:, :, _di] * _stance_mask)
                        )
                        _train_stance_tau_mae_pct_count[_name] += int(np.sum(_stance_mask))
                except Exception:
                    pass
            
            train_step_count += 1
            
            # Progress update
            if train_step_count % log_every == 0 or train_step_count == 1:
                step_duration = time.time() - step_start
                progress_pct = (train_step_count / steps_per_epoch) * 100
                avg_loss = train_metrics["total_loss"] / train_step_count
                
                # Average components
                avg_cop = train_metrics["cop_loss"] / train_step_count
                avg_grf = train_metrics["grf_loss"] / train_step_count
                avg_mom = train_metrics["moments_loss"] / train_step_count
                avg_qinv = train_metrics["qfrc_inverse_loss"] / train_step_count
                avg_qinv_reg = train_metrics["qfrc_inverse_input_reg_loss"] / train_step_count
                avg_rot = train_metrics["rotation_loss"] / train_step_count
                avg_rot_reg = train_metrics["rotation_input_reg_loss"] / train_step_count
                avg_jac = train_metrics["jacobian_loss"] / train_step_count
                avg_jac_reg = train_metrics["jacobian_input_reg_loss"] / train_step_count
                avg_trq = train_metrics["torque_loss"] / train_step_count
                avg_trq_cop = train_metrics["torque_cop_effect_loss"] / train_step_count
                avg_trq_grf = train_metrics["torque_grf_effect_loss"] / train_step_count
                avg_grf_corr = train_metrics["grf_correction_loss"] / train_step_count
                avg_out_reg = train_metrics["output_reg_loss"] / train_step_count
                
                # Estimate time remaining in epoch
                elapsed = time.time() - epoch_step_start
                if train_step_count > 1:
                    # Exclude the first JIT step for a better estimate
                    avg_step_time = (elapsed - (jit_time if 'jit_time' in locals() else 0)) / (train_step_count - 1)
                    rem_steps = steps_per_epoch - train_step_count
                    eta_sec = rem_steps * avg_step_time
                    eta_str = f" | ETA: {eta_sec/60:.1f}m"
                else:
                    eta_str = ""

                print(f"   Step {train_step_count}/{steps_per_epoch} ({progress_pct:.1f}%) - Loss: {float(metrics['total_loss']):.4f} (Avg: {avg_loss:.4f}) | {step_duration:.2f}s/step{eta_str}")
                print(f"      [Avg Raw]    COP:{avg_cop:.4f}, GRF:{avg_grf:.4f}, Mom:{avg_mom:.4f}, Cont:{train_metrics['contact_loss']/train_step_count:.4f}, TauGRF:{avg_trq:.4f}, Cor:{avg_grf_corr:.4f}, OutReg:{avg_out_reg:.4f}")
                print(f"      [Avg Raw]    Tau->COP:{avg_trq_cop:.4f}, Tau->GRF:{avg_trq_grf:.4f}")
                print(f"      [Avg Scaled] COP:{avg_cop*loss_weights['cop']:.4f}, GRF:{avg_grf*loss_weights['grf']:.4f}, Mom:{avg_mom*loss_weights['moments']:.4f}, Cont:{train_metrics['contact_loss']/train_step_count*loss_weights['contact']:.4f}, TauGRF:{avg_trq*loss_weights['torque']:.4f}, Cor:{avg_grf_corr*loss_weights['grf_correction']:.4f}, OutReg:{avg_out_reg*loss_weights['output_reg']:.4f}")
                print(f"      [Avg Scaled] Tau->COP:{avg_trq_cop*loss_weights['torque']:.4f}, Tau->GRF:{avg_trq_grf*loss_weights['torque']:.4f}")
                print("", flush=True)
            
            # Keep last batch for visualization
            if train_step_count == steps_per_epoch:
                last_train_batch = raw_batch # Store RAW batch for visualization
                # batch is already normalized here
                last_train_pred = state.apply_fn(
                    {"params": state.params},
                    batch["input"],
                    batch["static_context"],
                    train=False
                )
        
        print(f"✅ Training phase complete: {train_step_count} batches processed", flush=True)
        
        # Average train metrics
        if train_step_count > 0:
            for k in train_metrics:
                train_metrics[k] /= train_step_count
        _train_tau_mae_pct_norm = {}
        for _name in _KEY_TAU_DOFS:
            _cnt = int(_train_tau_mae_pct_count.get(_name, 0))
            if _cnt > 0:
                _train_tau_mae_pct_norm[_name] = float(_train_tau_mae_pct_sum[_name] / _cnt)
            else:
                _train_tau_mae_pct_norm[_name] = float("nan")
        _train_tau_mae_pct_norm_bilateral = {}
        for _joint, (_dof_r, _dof_l) in _BILATERAL_TAU_MAP.items():
            _mae_sum_lr = float(_train_tau_mae_pct_sum.get(_dof_r, 0.0) + _train_tau_mae_pct_sum.get(_dof_l, 0.0))
            _mae_cnt_lr = int(_train_tau_mae_pct_count.get(_dof_r, 0) + _train_tau_mae_pct_count.get(_dof_l, 0))
            if _mae_cnt_lr > 0:
                _train_tau_mae_pct_norm_bilateral[_joint] = float(_mae_sum_lr / _mae_cnt_lr)
            else:
                _train_tau_mae_pct_norm_bilateral[_joint] = float("nan")
        _train_stance_tau_mae_pct_norm_bilateral = {}
        for _joint, (_dof_r, _dof_l) in _STANCE_MAE_BILATERAL_TAU_MAP.items():
            _mae_sum_lr = float(
                _train_stance_tau_mae_pct_sum.get(_dof_r, 0.0)
                + _train_stance_tau_mae_pct_sum.get(_dof_l, 0.0)
            )
            _mae_cnt_lr = int(
                _train_stance_tau_mae_pct_count.get(_dof_r, 0)
                + _train_stance_tau_mae_pct_count.get(_dof_l, 0)
            )
            if _mae_cnt_lr > 0:
                _train_stance_tau_mae_pct_norm_bilateral[_joint] = float(_mae_sum_lr / _mae_cnt_lr)
            else:
                _train_stance_tau_mae_pct_norm_bilateral[_joint] = float("nan")
        _train_grf_mae_pct_bw_bilateral = {}
        for _axis in _BILATERAL_GRF_AXIS_MAP.keys():
            _cnt = int(_train_grf_mae_pct_bw_count.get(_axis, 0))
            if _cnt > 0:
                _train_grf_mae_pct_bw_bilateral[_axis] = float(_train_grf_mae_pct_bw_sum[_axis] / _cnt)
            else:
                _train_grf_mae_pct_bw_bilateral[_axis] = float("nan")
        _train_cop_overall_rmse = float("nan")
        _train_grf_overall_rmse = float("nan")
        _train_mom_overall_rmse = float("nan")
        if _train_frames > 0:
            _n_train = float(_train_frames)
            _train_cop_overall_rmse = float(np.sqrt(np.mean(_train_cop_sumsq) / _n_train))
            _train_grf_overall_rmse = float(np.sqrt(np.mean(_train_grf_sumsq) / _n_train))
            _train_mom_overall_rmse = float(np.sqrt(np.mean(_train_mom_sumsq) / _n_train))
        
        # Validation phase: Processing val_steps batches...
        val_metrics = {
            "cop_loss": 0, "grf_loss": 0, "moments_loss": 0, "qfrc_inverse_loss": 0, "qfrc_inverse_input_reg_loss": 0,
            "rotation_loss": 0, "rotation_input_reg_loss": 0, "jacobian_loss": 0, "jacobian_input_reg_loss": 0, "contact_loss": 0,
            "torque_loss": 0, "torque_cop_effect_loss": 0, "torque_grf_effect_loss": 0, "grf_correction_loss": 0, "output_reg_loss": 0, "total_loss": 0
        }
        
        last_val_batch = None
        last_val_pred = None
        val_step_count = 0
        val_trial_channel_accum = {} if not args.disable_validation_outlier_plots else None
        val_trial_series = {} if not args.disable_validation_outlier_plots else None

        # Accumulators for full-validation-set RMSE (COP, GRF, moments, torque)
        # Per-channel SSE accumulators (index matches outlier_channel_defs / pred_channels)
        # COP: 4ch, GRF: 6ch, moments: 2ch
        _val_cop_sumsq   = np.zeros(4,  dtype=np.float64)   # per-channel
        _val_grf_sumsq   = np.zeros(6,  dtype=np.float64)
        _val_mom_sumsq   = np.zeros(2,  dtype=np.float64)
        _val_cop_sumae   = np.zeros(4,  dtype=np.float64)   # for MAE
        _val_grf_sumae   = np.zeros(6,  dtype=np.float64)
        _val_cop_maxae   = np.zeros(4,  dtype=np.float64)   # running max
        _val_grf_maxae   = np.zeros(6,  dtype=np.float64)
        _val_frames      = 0                                  # total frames accumulated
        _val_tau_sumsq   = {k: 0.0 for k in _KEY_TAU_DOFS}
        _val_tau_sumsq_all = 0.0   # overall across all torque DOFs
        _val_tau_count   = 0        # total frame-DOF samples (for overall)
        _val_tau_frames  = 0        # total frames (for per-DOF counts)
        _val_tau_mae_pct_sum = {k: 0.0 for k in _KEY_TAU_DOFS}
        _val_tau_mae_pct_count = {k: 0 for k in _KEY_TAU_DOFS}
        _val_stance_tau_mae_pct_sum = {k: 0.0 for k in _STANCE_MAE_TAU_DOFS}
        _val_stance_tau_mae_pct_count = {k: 0 for k in _STANCE_MAE_TAU_DOFS}
        _val_grf_mae_pct_bw_sum = {axis: 0.0 for axis in _BILATERAL_GRF_AXIS_MAP}
        _val_grf_mae_pct_bw_count = {axis: 0 for axis in _BILATERAL_GRF_AXIS_MAP}
        
        # Print progress for validation too
        val_log_every = max(1, val_steps // 5)  # Print ~5 times during validation

        for batch in val_loader:
            # Keep raw batch for visualization
            raw_batch = batch
            
            # Normalize batch for computation
            batch_norm = normalize_batch(batch, normalizers)
            
            step_metrics, step_pred = eval_step(state, batch_norm, loss_weights, float(epoch))
            
            for k in val_metrics:
                val_metrics[k] += float(step_metrics[k])

            # --- Per-trial/channel outlier accumulation (validation only) ---
            # Accumulate per-trial SSE per channel so we can print top outlier trials
            # by RMSE for each prediction channel at epoch end.
            if "trial_idx" in raw_batch:
                pred_np = np.array(step_pred)
                static_np = np.array(raw_batch["static_context"])  # (B, 8)
                h_batch = static_np[:, 0:1, None]  # (B, 1, 1)
                m_batch = static_np[:, 1:2, None]  # (B, 1, 1)
                qfrc_inverse_output_dim = _infer_qfrc_inverse_output_dim(batch=raw_batch, normalizers=normalizers)
                cop_pred_raw, grf_pred_raw, moments_pred_raw, contact_pred_raw, _qfrc_inverse_pred_raw, _rotation_pred_raw, _jacobian_pred_raw = split_model_predictions(
                    pred_np,
                    qfrc_inverse_output_dim=qfrc_inverse_output_dim,
                    rotation_output_dim=0,
                )
                qfrc_inverse_pred_phys, rotation_pred_phys = decode_auxiliary_predictions(
                    pred_np,
                    raw_batch,
                    normalizers,
                    xp=np,
                )

                cop_pred_ratio = normalizers["cop"].unnormalize(cop_pred_raw)
                grf_pred_ratio = normalizers["grf"].unnormalize(grf_pred_raw)
                moments_pred_ratio = normalizers["moments"].unnormalize(moments_pred_raw)

                grf_pred_phys = grf_pred_ratio * m_batch * 9.8067
                cop_pred_phys = decode_cop_signal_to_length(
                    cop_pred_ratio,
                    grf_pred_ratio,
                    h_batch,
                    use_grf_norm_cop=args.UseGRFNormCOP,
                    contact_probability=contact_pred_raw if args.UseGRFNormCOP else None,
                    xp=np,
                )
                moments_pred_phys = moments_pred_ratio * m_batch * h_batch * 9.8067

                grf_gt_phys = np.array(raw_batch["grf"]) * m_batch * 9.8067
                cop_gt_phys = decode_cop_signal_to_length(
                    np.array(raw_batch["cop"]),
                    np.array(raw_batch["grf"]),
                    h_batch,
                    use_grf_norm_cop=args.UseGRFNormCOP,
                    xp=np,
                )
                moments_gt_phys = np.array(raw_batch["moments"]) * m_batch * h_batch * 9.8067
                _valid_mask = _extract_batched_frame_mask(
                    np.array(raw_batch["supervision_mask"]) if "supervision_mask" in raw_batch else None,
                    cop_pred_phys.shape[0],
                    cop_pred_phys.shape[1],
                )

                if args.cop_mask and pred_np.shape[-1] >= STANDARD_OUTPUT_DIM:
                    contact_prob = pred_np[..., CONTACT_SLICE]
                    mask_r = (contact_prob[..., 0:1] > 0.5).astype(cop_pred_phys.dtype)
                    mask_l = (contact_prob[..., 1:2] > 0.5).astype(cop_pred_phys.dtype)
                    cop_pred_phys = np.concatenate([
                        cop_pred_phys[..., 0:2] * mask_r,
                        cop_pred_phys[..., 2:4] * mask_l,
                    ], axis=-1)
                    grf_pred_phys = np.concatenate([
                        grf_pred_phys[..., 0:3] * mask_r,
                        grf_pred_phys[..., 3:6] * mask_l,
                    ], axis=-1)
                    moments_pred_phys = np.concatenate([
                        moments_pred_phys[..., 0:1] * mask_r,
                        moments_pred_phys[..., 1:2] * mask_l,
                    ], axis=-1)

                _contact_bool = np.array(raw_batch["contactBoolean"])  # (B, T, 2)
                _stance_r = (_contact_bool[..., 0] > 0.5) & _valid_mask
                _stance_l = (_contact_bool[..., 1] > 0.5) & _valid_mask

                # GRF MAE in %BW: abs((pred/(m*g)) - (gt/(m*g))) * 100, stance-only
                _norm_mg = np.maximum(m_batch * 9.8067, 1e-8)
                _grf_abs_pct_bw_err = np.abs((grf_pred_phys / _norm_mg) - (grf_gt_phys / _norm_mg)) * 100.0
                for _axis, (_ri, _li) in _BILATERAL_GRF_AXIS_MAP.items():
                    _val_grf_mae_pct_bw_sum[_axis] += float(
                        np.sum(_grf_abs_pct_bw_err[:, :, _ri] * _stance_r)
                        + np.sum(_grf_abs_pct_bw_err[:, :, _li] * _stance_l)
                    )
                    _val_grf_mae_pct_bw_count[_axis] += int(np.sum(_stance_r) + np.sum(_stance_l))

                err_channels = np.concatenate([
                    cop_pred_phys - cop_gt_phys,        # 4
                    grf_pred_phys - grf_gt_phys,        # 6
                    moments_pred_phys - moments_gt_phys # 2
                ], axis=-1)  # (B, T, 12)
                pred_channels = np.concatenate([
                    cop_pred_phys,
                    grf_pred_phys,
                    moments_pred_phys
                ], axis=-1)  # (B, T, 12)
                gt_channels = np.concatenate([
                    cop_gt_phys,
                    grf_gt_phys,
                    moments_gt_phys
                ], axis=-1)  # (B, T, 12)

                if not args.disable_validation_outlier_plots:
                    trial_idx_batch = np.array(raw_batch["trial_idx"]).reshape(-1).astype(np.int32)
                    window_start_batch = (
                        np.array(raw_batch["window_start_idx"]).reshape(-1).astype(np.int32)
                        if "window_start_idx" in raw_batch
                        else None
                    )
                    for bi, trial_idx in enumerate(trial_idx_batch):
                        key = int(trial_idx)
                        if key not in val_trial_channel_accum:
                            val_trial_channel_accum[key] = {
                                "sumsq": np.zeros(len(outlier_channel_defs), dtype=np.float64),
                                "count": 0,
                            }
                        _trial_valid = _valid_mask[bi][:, None]
                        val_trial_channel_accum[key]["sumsq"] += np.sum((err_channels[bi] ** 2) * _trial_valid, axis=0)
                        val_trial_channel_accum[key]["count"] += int(np.sum(_valid_mask[bi]))
                        if key not in val_trial_series:
                            val_trial_series[key] = {"pred": [], "gt": [], "starts": [], "masks": []}
                        val_trial_series[key]["pred"].append(pred_channels[bi].copy())
                        val_trial_series[key]["gt"].append(gt_channels[bi].copy())
                        val_trial_series[key]["masks"].append(_valid_mask[bi].copy())
                        if window_start_batch is not None:
                            val_trial_series[key]["starts"].append(int(window_start_batch[bi]))
                        else:
                            # Backward-compatible fallback if start indices are unavailable.
                            val_trial_series[key]["starts"].append(int(len(val_trial_series[key]["starts"])))

                # --- Full-val-set COP / GRF / moment RMSE accumulation ---
                cop_err = cop_pred_phys - cop_gt_phys   # (B, T, 4)
                grf_err = grf_pred_phys - grf_gt_phys   # (B, T, 6)
                mom_err = moments_pred_phys - moments_gt_phys  # (B, T, 2)
                _val_valid = _valid_mask[..., None]
                _val_cop_sumsq += np.sum((cop_err ** 2) * _val_valid, axis=(0, 1))
                _val_grf_sumsq += np.sum((grf_err ** 2) * _val_valid, axis=(0, 1))
                _val_mom_sumsq += np.sum((mom_err ** 2) * _val_valid, axis=(0, 1))
                _val_cop_sumae += np.sum(np.abs(cop_err) * _val_valid, axis=(0, 1))
                _val_grf_sumae += np.sum(np.abs(grf_err) * _val_valid, axis=(0, 1))
                _val_cop_maxae = np.maximum(_val_cop_maxae, np.max(np.abs(cop_err) * _val_valid, axis=(0, 1)))
                _val_grf_maxae = np.maximum(_val_grf_maxae, np.max(np.abs(grf_err) * _val_valid, axis=(0, 1)))
                _val_frames += int(np.sum(_valid_mask))

                # --- Full-val-set torque RMSE accumulation ---
                # Reuse the already-computed physical GRF/COP/moments from above.
                # compute_full_external_moments / compute_tau_grf_from_predictions operate
                # on JAX arrays but accept numpy; we keep everything on CPU numpy here.
                try:
                    _full_mom_pred = np.array(compute_full_external_moments(
                        jnp.array(cop_pred_phys),
                        jnp.array(grf_pred_phys),
                        jnp.array(moments_pred_phys),
                        raw_batch["ankle_heights"],
                        jnp.array(rotation_pred_phys),
                    ))
                    _jacp_pred, _jacr_pred = select_torque_jacobians(
                        pred_np,
                        raw_batch,
                        normalizers,
                        xp=np,
                    )
                    _tau_grf_pred = np.array(compute_tau_grf_from_predictions(
                        jnp.array(grf_pred_phys),
                        jnp.array(_full_mom_pred),
                        jnp.array(_jacp_pred),
                        jnp.array(_jacr_pred),
                    ))
                    _tau_pred = np.array(qfrc_inverse_pred_phys) - _tau_grf_pred
                    _tau_gt = np.array(_full_id_target_from_batch(raw_batch, xp=np))
                    _tau_err = _tau_pred - _tau_gt  # (B, T, 39)
                    _B, _T, _D = _tau_err.shape
                    for _name, _di in _KEY_TAU_DOFS.items():
                        _val_tau_sumsq[_name] += float(np.sum((_tau_err[:, :, _di] ** 2) * _valid_mask))
                    _val_tau_sumsq_all += float(np.sum((_tau_err ** 2) * _val_valid))
                    _valid_frame_count = int(np.sum(_valid_mask))
                    _val_tau_count  += _valid_frame_count * _D
                    _val_tau_frames += _valid_frame_count

                    # Stance-only torque MAE% in BW*H-normalized space:
                    # abs((tau_pred/(mgh)) - (tau_gt/(mgh))) * 100.
                    _norm_mgh = np.maximum(m_batch * 9.8067 * h_batch, 1e-8)  # (B, 1, 1)
                    _tau_abs_pct_err = np.abs((_tau_pred / _norm_mgh) - (_tau_gt / _norm_mgh)) * 100.0

                    for _name, _di in _KEY_TAU_DOFS.items():
                        _stance_mask = _torque_stance_mask_for_name(_name, _stance_r, _stance_l)
                        _val_tau_mae_pct_sum[_name] += float(np.sum(_tau_abs_pct_err[:, :, _di] * _stance_mask))
                        _val_tau_mae_pct_count[_name] += int(np.sum(_stance_mask))
                    for _name, _di in _STANCE_MAE_TAU_DOFS.items():
                        _stance_mask = _stance_r if _name.startswith("R ") else _stance_l
                        _val_stance_tau_mae_pct_sum[_name] += float(
                            np.sum(_tau_abs_pct_err[:, :, _di] * _stance_mask)
                        )
                        _val_stance_tau_mae_pct_count[_name] += int(np.sum(_stance_mask))
                except Exception as _tau_ex:
                    pass  # never crash the training loop over diagnostics

            val_step_count += 1
            
            # Progress update
            if val_step_count % val_log_every == 0 or val_step_count == 1:
                progress_pct = (val_step_count / val_steps) * 100
                avg_loss = val_metrics["total_loss"] / val_step_count
                
                # Average components
                avg_cop = val_metrics["cop_loss"] / val_step_count
                avg_grf = val_metrics["grf_loss"] / val_step_count
                avg_mom = val_metrics["moments_loss"] / val_step_count
                avg_qinv = val_metrics["qfrc_inverse_loss"] / val_step_count
                avg_qinv_reg = val_metrics["qfrc_inverse_input_reg_loss"] / val_step_count
                avg_rot = val_metrics["rotation_loss"] / val_step_count
                avg_rot_reg = val_metrics["rotation_input_reg_loss"] / val_step_count
                avg_jac = val_metrics["jacobian_loss"] / val_step_count
                avg_jac_reg = val_metrics["jacobian_input_reg_loss"] / val_step_count
                avg_trq = val_metrics["torque_loss"] / val_step_count
                avg_trq_cop = val_metrics["torque_cop_effect_loss"] / val_step_count
                avg_trq_grf = val_metrics["torque_grf_effect_loss"] / val_step_count
                avg_grf_corr = val_metrics["grf_correction_loss"] / val_step_count
                avg_out_reg = val_metrics["output_reg_loss"] / val_step_count
                
                print(f"   Val Step {val_step_count}/{val_steps} ({progress_pct:.1f}%) - Loss: {float(step_metrics['total_loss']):.4f} (Avg: {avg_loss:.4f})")
                print(f"      [Avg Raw]    COP:{avg_cop:.4f}, GRF:{avg_grf:.4f}, Mom:{avg_mom:.4f}, Cont:{val_metrics['contact_loss']/val_step_count:.4f}, TauGRF:{avg_trq:.4f}, Cor:{avg_grf_corr:.4f}, OutReg:{avg_out_reg:.4f}")
                print(f"      [Avg Raw]    Tau->COP:{avg_trq_cop:.4f}, Tau->GRF:{avg_trq_grf:.4f}")
                print(f"      [Avg Scaled] COP:{avg_cop*loss_weights['cop']:.4f}, GRF:{avg_grf*loss_weights['grf']:.4f}, Mom:{avg_mom*loss_weights['moments']:.4f}, Cont:{val_metrics['contact_loss']/val_step_count*loss_weights['contact']:.4f}, TauGRF:{avg_trq*loss_weights['torque']:.4f}, Cor:{avg_grf_corr*loss_weights['grf_correction']:.4f}, OutReg:{avg_out_reg*loss_weights['output_reg']:.4f}")
                print(f"      [Avg Scaled] Tau->COP:{avg_trq_cop*loss_weights['torque']:.4f}, Tau->GRF:{avg_trq_grf*loss_weights['torque']:.4f}")
                print("", flush=True)
            
            # Keep last batch for visualization
            if val_step_count == val_steps:
                last_val_batch = raw_batch
                last_val_pred = step_pred

        print(f"✅ Validation phase complete: {val_step_count} batches processed", flush=True)

        # Average val metrics
        if val_step_count > 0:
            for k in val_metrics:
                val_metrics[k] /= val_step_count

        # Print top-3 validation outlier trials per prediction channel
        print("\n🔎 Validation Outliers (Top 3 Trials per Channel):", flush=True)
        top_outliers_by_channel = {}
        if args.disable_validation_outlier_plots:
            print("   Disabled for this run.", flush=True)
        elif not val_trial_channel_accum:
            print("   No validation outlier data collected.", flush=True)
        else:
            for ch_idx, (ch_name, ch_unit) in enumerate(outlier_channel_defs):
                channel_rows = []
                for trial_idx, acc in val_trial_channel_accum.items():
                    if acc["count"] <= 0:
                        continue
                    rmse = float(np.sqrt(acc["sumsq"][ch_idx] / max(acc["count"], 1)))
                    trial_name = val_trial_name_map.get(trial_idx, f"trial_idx={trial_idx}")
                    channel_rows.append((rmse, trial_name, trial_idx))

                channel_rows.sort(key=lambda x: x[0], reverse=True)
                top_rows = channel_rows[:3]
                top_outliers_by_channel[ch_idx] = top_rows
                print(f"   {ch_name}:", flush=True)
                if not top_rows:
                    print("      none", flush=True)
                    continue
                for rank, (rmse, trial_name, _trial_idx) in enumerate(top_rows, start=1):
                    print(f"      {rank}. {trial_name} -> RMSE {rmse:.5f} {ch_unit}", flush=True)

        # Save one large GT-vs-Pred outlier grid on epoch 1, then every 5 epochs.
        if (not args.disable_validation_outlier_plots) and top_outliers_by_channel and (epoch == 1 or epoch % 5 == 0):
            print(f"📊 Saving validation outlier grid for epoch {epoch}...", flush=True)
            plot_validation_outlier_grid(
                epoch=epoch,
                output_dir=args.output_dir,
                top_outliers_by_channel=top_outliers_by_channel,
                outlier_channel_defs=outlier_channel_defs,
                val_trial_series=val_trial_series,
                top_k=3,
            )
            last_val_outlier_plot_path = os.path.join(args.output_dir, f"validation_outliers_epoch_{epoch:04d}.png")
            print(f"✅ Saved {last_val_outlier_plot_path}", flush=True)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n📈 EPOCH {epoch} SUMMARY:", flush=True)
        print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)", flush=True)
        print(f"   Train Total: {train_metrics['total_loss']:.4f}")
        print(f"      [Raw]    COP:{train_metrics['cop_loss']:.4f}, GRF:{train_metrics['grf_loss']:.4f}, Mom:{train_metrics['moments_loss']:.4f}, Cont:{train_metrics['contact_loss']:.4f}, TauGRF:{train_metrics['torque_loss']:.4f}, Cor:{train_metrics['grf_correction_loss']:.4f}, OutReg:{train_metrics['output_reg_loss']:.4f}")
        print(f"      [Raw]    Tau->COP:{train_metrics['torque_cop_effect_loss']:.4f}, Tau->GRF:{train_metrics['torque_grf_effect_loss']:.4f}")
        print(f"      [Scaled] COP:{train_metrics['cop_loss']*loss_weights['cop']:.4f}, GRF:{train_metrics['grf_loss']*loss_weights['grf']:.4f}, Mom:{train_metrics['moments_loss']*loss_weights['moments']:.4f}, Cont:{train_metrics['contact_loss']*loss_weights['contact']:.4f}, TauGRF:{train_metrics['torque_loss']*loss_weights['torque']:.4f}, Cor:{train_metrics['grf_correction_loss']*loss_weights['grf_correction']:.4f}, OutReg:{train_metrics['output_reg_loss']*loss_weights['output_reg']:.4f}")
        print(f"      [Scaled] Tau->COP:{train_metrics['torque_cop_effect_loss']*loss_weights['torque']:.4f}, Tau->GRF:{train_metrics['torque_grf_effect_loss']*loss_weights['torque']:.4f}")
        print("")

        print(f"   Val Total:   {float(val_metrics['total_loss']):.4f}")
        print(f"      [Raw]    COP:{float(val_metrics['cop_loss']):.4f}, GRF:{float(val_metrics['grf_loss']):.4f}, Mom:{float(val_metrics['moments_loss']):.4f}, Cont:{float(val_metrics['contact_loss']):.4f}, TauGRF:{float(val_metrics['torque_loss']):.4f}, Cor:{float(val_metrics['grf_correction_loss']):.4f}, OutReg:{float(val_metrics['output_reg_loss']):.4f}")
        print(f"      [Raw]    Tau->COP:{float(val_metrics['torque_cop_effect_loss']):.4f}, Tau->GRF:{float(val_metrics['torque_grf_effect_loss']):.4f}")
        print(f"      [Scaled] COP:{val_metrics['cop_loss']*loss_weights['cop']:.4f}, GRF:{val_metrics['grf_loss']*loss_weights['grf']:.4f}, Mom:{val_metrics['moments_loss']*loss_weights['moments']:.4f}, Cont:{val_metrics['contact_loss']*loss_weights['contact']:.4f}, TauGRF:{val_metrics['torque_loss']*loss_weights['torque']:.4f}, Cor:{val_metrics['grf_correction_loss']*loss_weights['grf_correction']:.4f}, OutReg:{val_metrics['output_reg_loss']*loss_weights['output_reg']:.4f}")
        print(f"      [Scaled] Tau->COP:{val_metrics['torque_cop_effect_loss']*loss_weights['torque']:.4f}, Tau->GRF:{val_metrics['torque_grf_effect_loss']*loss_weights['torque']:.4f}")
        print("", flush=True)
        print(
            f"   Train RMSE (full train set): COP {_train_cop_overall_rmse*100:.2f} cm | "
            f"GRF {_train_grf_overall_rmse:.2f} N | Moments {_train_mom_overall_rmse:.2f} Nm",
            flush=True,
        )
        _train_stance_rows = _format_named_metric_rows(
            metric_values=_train_stance_tau_mae_pct_norm_bilateral,
            display_names=_STANCE_MAE_DISPLAY,
            ordered_keys=list(_STANCE_MAE_BILATERAL_TAU_MAP.keys()),
            values_per_row=3,
            suffix="%",
        )
        _train_tau_mae_rows = _format_named_metric_rows(
            metric_values=_train_tau_mae_pct_norm,
            display_names=_KEY_TAU_DISPLAY,
            ordered_keys=list(_KEY_TAU_DOFS.keys()),
            values_per_row=4,
            suffix="%",
        )
        print("   Train Torque MAE% (mgh-normalized, stance-only):", flush=True)
        for _row in _train_tau_mae_rows:
            print(f"      {_row}", flush=True)
        print("   Train Torque MAE %BW*H (stance-only, bilateral):", flush=True)
        for _row in _train_stance_rows:
            print(f"      {_row}", flush=True)

        # ── Full-val-set physical RMSE (COP, GRF, moments, torque) ───────────
        # Build a single dict that will be forwarded to plot_predictions so the
        # VAL stats panel shows whole-dataset numbers instead of a single sample.
        val_fullset_stats = None
        _val_tau_fullset_rmse = {}
        _val_tau_bilateral_rmse = {}
        _val_tau_mae_pct_norm = {}
        _val_tau_mae_pct_norm_bilateral = {}
        _val_stance_tau_mae_pct_norm_bilateral = {}
        _val_grf_mae_pct_bw_bilateral = {}
        _val_tau_overall_rmse = float('nan')
        _fs_cop_overall_rmse = float("nan")
        _fs_grf_overall_rmse = float("nan")
        _fs_mom_overall_rmse = float("nan")
        if _val_frames > 0:
            n = float(_val_frames)
            _fs_cop_ch_rmse  = [float(np.sqrt(v / n)) for v in _val_cop_sumsq]
            _fs_grf_ch_rmse  = [float(np.sqrt(v / n)) for v in _val_grf_sumsq]
            _fs_mom_ch_rmse  = [float(np.sqrt(v / n)) for v in _val_mom_sumsq]
            _fs_cop_ch_mae   = [float(v / n) for v in _val_cop_sumae]
            _fs_grf_ch_mae   = [float(v / n) for v in _val_grf_sumae]
            _fs_cop_overall_rmse = float(np.sqrt(np.mean(_val_cop_sumsq) / n))
            _fs_grf_overall_rmse = float(np.sqrt(np.mean(_val_grf_sumsq) / n))
            _fs_mom_overall_rmse = float(np.sqrt(np.mean(_val_mom_sumsq) / n))
            _fs_cop_mae = float(np.mean(_val_cop_sumae) / n)
            _fs_grf_mae = float(np.mean(_val_grf_sumae) / n)
            _fs_cop_max = float(np.max(_val_cop_maxae))
            _fs_grf_max = float(np.max(_val_grf_maxae))

            if _val_tau_frames > 0:
                _val_tau_overall_rmse = float(np.sqrt(_val_tau_sumsq_all / max(_val_tau_count, 1)))
                for _name in _KEY_TAU_DOFS:
                    _val_tau_fullset_rmse[_name] = float(np.sqrt(_val_tau_sumsq[_name] / max(_val_tau_frames, 1)))
                    _cnt = int(_val_tau_mae_pct_count.get(_name, 0))
                    if _cnt > 0:
                        _val_tau_mae_pct_norm[_name] = float(_val_tau_mae_pct_sum[_name] / _cnt)
                    else:
                        _val_tau_mae_pct_norm[_name] = float("nan")
                for _joint, (_dof_r, _dof_l) in _BILATERAL_TAU_MAP.items():
                    _sum_lr = float(_val_tau_sumsq.get(_dof_r, 0.0) + _val_tau_sumsq.get(_dof_l, 0.0))
                    # True bilateral RMSE across both sides for this joint:
                    # sqrt((SSE_R + SSE_L) / (2 * n_frames)).
                    _val_tau_bilateral_rmse[_joint] = float(np.sqrt(_sum_lr / max(2 * _val_tau_frames, 1)))
                    _mae_sum_lr = float(_val_tau_mae_pct_sum.get(_dof_r, 0.0) + _val_tau_mae_pct_sum.get(_dof_l, 0.0))
                    _mae_cnt_lr = int(_val_tau_mae_pct_count.get(_dof_r, 0) + _val_tau_mae_pct_count.get(_dof_l, 0))
                    if _mae_cnt_lr > 0:
                        _val_tau_mae_pct_norm_bilateral[_joint] = float(_mae_sum_lr / _mae_cnt_lr)
                    else:
                        _val_tau_mae_pct_norm_bilateral[_joint] = float("nan")
                for _joint, (_dof_r, _dof_l) in _STANCE_MAE_BILATERAL_TAU_MAP.items():
                    _mae_sum_lr = float(
                        _val_stance_tau_mae_pct_sum.get(_dof_r, 0.0)
                        + _val_stance_tau_mae_pct_sum.get(_dof_l, 0.0)
                    )
                    _mae_cnt_lr = int(
                        _val_stance_tau_mae_pct_count.get(_dof_r, 0)
                        + _val_stance_tau_mae_pct_count.get(_dof_l, 0)
                    )
                    if _mae_cnt_lr > 0:
                        _val_stance_tau_mae_pct_norm_bilateral[_joint] = float(_mae_sum_lr / _mae_cnt_lr)
                    else:
                        _val_stance_tau_mae_pct_norm_bilateral[_joint] = float("nan")
                _fs_tau_overall_rmse = _val_tau_overall_rmse
                _fs_tau_mae = float('nan')  # not tracked; tau MAE skipped for brevity
            else:
                _fs_tau_overall_rmse = float('nan')
                _fs_tau_mae = float('nan')

            for _axis in _BILATERAL_GRF_AXIS_MAP.keys():
                _cnt = int(_val_grf_mae_pct_bw_count.get(_axis, 0))
                if _cnt > 0:
                    _val_grf_mae_pct_bw_bilateral[_axis] = float(_val_grf_mae_pct_bw_sum[_axis] / _cnt)
                else:
                    _val_grf_mae_pct_bw_bilateral[_axis] = float("nan")

            val_fullset_stats = dict(
                cop_ch_rmse=_fs_cop_ch_rmse,
                grf_ch_rmse=_fs_grf_ch_rmse,
                mom_ch_rmse=_fs_mom_ch_rmse,
                tau_ch_rmse_sel=_val_tau_fullset_rmse,
                cop_overall_rmse=_fs_cop_overall_rmse,
                grf_overall_rmse=_fs_grf_overall_rmse,
                moments_overall_rmse=_fs_mom_overall_rmse,
                tau_overall_rmse=_fs_tau_overall_rmse,
                cop_mae=_fs_cop_mae,
                grf_mae=_fs_grf_mae,
                tau_mae=_fs_tau_mae,
                cop_max=_fs_cop_max,
                grf_max=_fs_grf_max,
            )

            _tau_rmse_rows = _format_named_metric_rows(
                metric_values=_val_tau_fullset_rmse,
                display_names=_KEY_TAU_DISPLAY,
                ordered_keys=list(_KEY_TAU_DOFS.keys()),
                values_per_row=4,
                suffix="",
            )
            print(f"   Val RMSE (full val set):", flush=True)
            print(
                f"      COP overall: {_fs_cop_overall_rmse*100:.2f} cm | "
                f"GRF overall: {_fs_grf_overall_rmse:.2f} N | "
                f"Moments overall: {_fs_mom_overall_rmse:.2f} Nm",
                flush=True,
            )
            if _val_tau_frames > 0:
                print(f"      Torque overall: {_val_tau_overall_rmse:.2f} Nm", flush=True)
                for _row in _tau_rmse_rows:
                    print(f"      {_row}", flush=True)
                _tau_mae_rows = _format_named_metric_rows(
                    metric_values=_val_tau_mae_pct_norm,
                    display_names=_KEY_TAU_DISPLAY,
                    ordered_keys=list(_KEY_TAU_DOFS.keys()),
                    values_per_row=4,
                    suffix="%",
                )
                print("      Torque MAE% (mgh-normalized, stance-only):", flush=True)
                for _row in _tau_mae_rows:
                    print(f"      {_row}", flush=True)
                _val_stance_rows = _format_named_metric_rows(
                    metric_values=_val_stance_tau_mae_pct_norm_bilateral,
                    display_names=_STANCE_MAE_DISPLAY,
                    ordered_keys=list(_STANCE_MAE_BILATERAL_TAU_MAP.keys()),
                    values_per_row=3,
                    suffix="%",
                )
                print("      Torque MAE %BW*H (stance-only, bilateral):", flush=True)
                for _row in _val_stance_rows:
                    print(f"      {_row}", flush=True)
                print(
                    "      GRF MAE%BW bilateral (stance-only): "
                    + " | ".join(f"{a.upper()}: {_val_grf_mae_pct_bw_bilateral.get(a, float('nan')):.2f}%" for a in ["x", "y", "z"]),
                    flush=True,
                )
        else:
            print("   Val RMSE (full val set): N/A (no trial_idx in batch)", flush=True)

        # Compute the epoch's torque score (if BestModelByTorque is on AND data is available).
        _epoch_torque_score = float("nan")
        if args.BestModelByTorque and _val_tau_fullset_rmse:
            _epoch_torque_score = 0.0
            for _joint, _metric_names in _BEST_MODEL_TAU_GROUPS.items():
                _w  = float(_tau_weights.get(_joint, 1.0))
                _group_rmse = [
                    float(_val_tau_fullset_rmse.get(_metric_name, float("nan")))
                    for _metric_name in _metric_names
                ]
                if not any(np.isnan(_rmse) for _rmse in _group_rmse):
                    _epoch_torque_score += _w * float(np.mean(_group_rmse))
            print(f"   Weighted torque score (best-model criterion): {_epoch_torque_score:.4f} Nm", flush=True)
        last_epoch_torque_score = float(_epoch_torque_score) if np.isfinite(_epoch_torque_score) else float("nan")
        
        # Track history
        train_loss_history.append(float(train_metrics['total_loss']))
        val_loss_history.append(float(val_metrics['total_loss']))
        for _comp_key in train_component_history:
            train_component_history[_comp_key].append(float(train_metrics[_comp_key]))
        for _comp_key in val_component_history:
            val_component_history[_comp_key].append(float(val_metrics[_comp_key]))
        train_rmse_history["cop"].append(float(_train_cop_overall_rmse))
        train_rmse_history["grf"].append(float(_train_grf_overall_rmse))
        train_rmse_history["moments"].append(float(_train_mom_overall_rmse))
        val_rmse_history["cop"].append(float(_fs_cop_overall_rmse))
        val_rmse_history["grf"].append(float(_fs_grf_overall_rmse))
        val_rmse_history["moments"].append(float(_fs_mom_overall_rmse))
        
        # Plot loss history
        plot_loss_history(train_loss_history, val_loss_history, args.output_dir,
                          train_component_history=train_component_history,
                          val_component_history=val_component_history,
                          loss_weights=loss_weights)
        
        # Phase-1 WandB logging: scalar epoch metrics only.
        _wandb_log = {
            "epoch": int(epoch),
            "epoch/time_s": float(epoch_time),
            "train/total_loss": float(train_metrics["total_loss"]),
            "train/cop_loss": float(train_metrics["cop_loss"]),
            "train/grf_loss": float(train_metrics["grf_loss"]),
            "train/moments_loss": float(train_metrics["moments_loss"]),
            "train/qfrc_inverse_loss": float(train_metrics["qfrc_inverse_loss"]),
            "train/qfrc_inverse_input_reg_loss": float(train_metrics["qfrc_inverse_input_reg_loss"]),
            "train/rotation_loss": float(train_metrics["rotation_loss"]),
            "train/rotation_input_reg_loss": float(train_metrics["rotation_input_reg_loss"]),
            "train/jacobian_loss": float(train_metrics["jacobian_loss"]),
            "train/jacobian_input_reg_loss": float(train_metrics["jacobian_input_reg_loss"]),
            "train/contact_loss": float(train_metrics["contact_loss"]),
            "train/torque_loss": float(train_metrics["torque_loss"]),
            "train/torque_cop_effect_loss": float(train_metrics["torque_cop_effect_loss"]),
            "train/torque_grf_effect_loss": float(train_metrics["torque_grf_effect_loss"]),
            "train/grf_correction_loss": float(train_metrics["grf_correction_loss"]),
            "train/output_reg_loss": float(train_metrics["output_reg_loss"]),
            "val/total_loss": float(val_metrics["total_loss"]),
            "val/cop_loss": float(val_metrics["cop_loss"]),
            "val/grf_loss": float(val_metrics["grf_loss"]),
            "val/moments_loss": float(val_metrics["moments_loss"]),
            "val/qfrc_inverse_loss": float(val_metrics["qfrc_inverse_loss"]),
            "val/qfrc_inverse_input_reg_loss": float(val_metrics["qfrc_inverse_input_reg_loss"]),
            "val/rotation_loss": float(val_metrics["rotation_loss"]),
            "val/rotation_input_reg_loss": float(val_metrics["rotation_input_reg_loss"]),
            "val/jacobian_loss": float(val_metrics["jacobian_loss"]),
            "val/jacobian_input_reg_loss": float(val_metrics["jacobian_input_reg_loss"]),
            "val/contact_loss": float(val_metrics["contact_loss"]),
            "val/torque_loss": float(val_metrics["torque_loss"]),
            "val/torque_cop_effect_loss": float(val_metrics["torque_cop_effect_loss"]),
            "val/torque_grf_effect_loss": float(val_metrics["torque_grf_effect_loss"]),
            "val/grf_correction_loss": float(val_metrics["grf_correction_loss"]),
            "val/output_reg_loss": float(val_metrics["output_reg_loss"]),
            "train/cop_rmse_fullset_overall_m": float(_train_cop_overall_rmse),
            "train/grf_rmse_fullset_overall_N": float(_train_grf_overall_rmse),
            "train/moments_rmse_fullset_overall_Nm": float(_train_mom_overall_rmse),
            "val/cop_rmse_fullset_overall_m": float(_fs_cop_overall_rmse),
            "val/grf_rmse_fullset_overall_N": float(_fs_grf_overall_rmse),
            "val/moments_rmse_fullset_overall_Nm": float(_fs_mom_overall_rmse),
            "val/torque_rmse_fullset_overall_Nm": float(_val_tau_overall_rmse),
        }
        _wandb_log["val/weighted_torque_score_Nm"] = float(_epoch_torque_score)
        _wandb_log["val/best_torque_score_Nm"] = (
            float(best_torque_score) if np.isfinite(best_torque_score) else float("nan")
        )
        for _name, _rmse in _val_tau_fullset_rmse.items():
            _wandb_log[f"val/torque_rmse_{_name.replace(' ', '_')}_Nm"] = float(_rmse)
        for _joint, _rmse in _val_tau_bilateral_rmse.items():
            _wandb_log[f"val/torque_rmse_bilateral_{_joint}_Nm"] = float(_rmse)
        for _name, _mae_pct in _val_tau_mae_pct_norm.items():
            _wandb_log[f"val/torque_mae_pct_norm_{_name.replace(' ', '_')}"] = float(_mae_pct)
        for _name, _mae_pct in _train_tau_mae_pct_norm.items():
            _wandb_log[f"train/torque_mae_pct_norm_{_name.replace(' ', '_')}"] = float(_mae_pct)
        for _joint, _mae_pct in _val_tau_mae_pct_norm_bilateral.items():
            _wandb_log[f"val/torque_mae_pct_norm_bilateral_{_joint}"] = float(_mae_pct)
            _wandb_log[f"val/torque_mae_percent_bilateral_{_joint}"] = float(_mae_pct)
        for _joint, _mae_pct in _val_stance_tau_mae_pct_norm_bilateral.items():
            _wandb_log[f"val/torque_mae_percent_bilateral_stance_{_joint}"] = float(_mae_pct)
        for _joint, _mae_pct in _train_tau_mae_pct_norm_bilateral.items():
            _wandb_log[f"train/torque_mae_pct_norm_bilateral_{_joint}"] = float(_mae_pct)
            _wandb_log[f"train/torque_mae_percent_bilateral_{_joint}"] = float(_mae_pct)
        for _joint, _mae_pct in _train_stance_tau_mae_pct_norm_bilateral.items():
            _wandb_log[f"train/torque_mae_percent_bilateral_stance_{_joint}"] = float(_mae_pct)
        for _axis, _mae_pct in _train_grf_mae_pct_bw_bilateral.items():
            _wandb_log[f"train/grf_mae_percent_bw_bilateral_{_axis}"] = float(_mae_pct)
        for _axis, _mae_pct in _val_grf_mae_pct_bw_bilateral.items():
            _wandb_log[f"val/grf_mae_percent_bw_bilateral_{_axis}"] = float(_mae_pct)
        
        # Visualization
        _should_plot_predictions = False
        if args.save_final_predictions_only:
            _should_plot_predictions = (epoch == args.epochs)
        elif args.vis_interval > 0:
            _should_plot_predictions = (epoch % args.vis_interval == 0)

        if _should_plot_predictions:
            print(f"\n📊 Generating visualizations for epoch {epoch}...", flush=True)
            
            # Use the last batches we processed for visualization
            if last_train_batch is not None and last_train_pred is not None:
                train_vis_batch = last_train_batch
                train_vis_pred = last_train_pred
            else:
                # Fallback: get a fresh batch from train loader
                train_vis_batch = next(iter(train_loader)) # Raw for plotter
                vis_norm = normalize_batch(train_vis_batch, normalizers)
                train_vis_pred = state.apply_fn(
                    {"params": state.params},
                    vis_norm["input"],
                    vis_norm["static_context"],
                    train=False
                )
            
            if last_val_batch is not None and last_val_pred is not None:
                val_vis_batch = last_val_batch
                val_vis_pred = last_val_pred
            else:
                val_vis_batch = next(iter(val_loader)) # Raw
                vis_norm = normalize_batch(val_vis_batch, normalizers)
                val_vis_pred = state.apply_fn(
                    {"params": state.params},
                    vis_norm["input"],
                    vis_norm["static_context"],
                    train=False
                )
            
            plot_predictions(
                train_vis_batch, train_vis_pred, 
                val_vis_batch, val_vis_pred, 
                normalizers, epoch, args.output_dir,
                train_sample_idx=0, val_sample_idx=0,
                train_trial_names=None,  # No trial names with data loader
                val_trial_names=None,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                loss_weights=loss_weights,
                epoch_time=epoch_time,
                cop_mask=args.cop_mask,
                use_grf_norm_cop=args.UseGRFNormCOP,
                val_fullset_stats=val_fullset_stats,
            )
            _candidate_final_pred_path = os.path.join(args.output_dir, f"predictions_epoch_{epoch:04d}.png")
            if os.path.exists(_candidate_final_pred_path):
                final_prediction_plot_path = _candidate_final_pred_path
            print(f"✅ Visualizations saved to {args.output_dir}", flush=True)

        _epoch_snapshot = _json_compatible(
            {
                "epoch": int(epoch),
                "epoch_time_s": float(epoch_time),
                "weighted_torque_score_Nm": _epoch_torque_score,
                "train_total_loss": float(train_metrics["total_loss"]),
                "val_total_loss": float(val_metrics["total_loss"]),
                "train": {
                    "cop_rmse_fullset_overall_m": float(_train_cop_overall_rmse),
                    "grf_rmse_fullset_overall_N": float(_train_grf_overall_rmse),
                    "moments_rmse_fullset_overall_Nm": float(_train_mom_overall_rmse),
                    "torque_mae_pct_norm_by_dof": dict(_train_tau_mae_pct_norm),
                    "torque_mae_pct_norm_bilateral": dict(_train_tau_mae_pct_norm_bilateral),
                    "torque_mae_percent_bilateral_stance": dict(_train_stance_tau_mae_pct_norm_bilateral),
                    "grf_mae_percent_bw_bilateral": dict(_train_grf_mae_pct_bw_bilateral),
                },
                "val": {
                    "cop_rmse_fullset_overall_m": float(_fs_cop_overall_rmse),
                    "grf_rmse_fullset_overall_N": float(_fs_grf_overall_rmse),
                    "moments_rmse_fullset_overall_Nm": float(_fs_mom_overall_rmse),
                    "torque_rmse_fullset_overall_Nm": float(_val_tau_overall_rmse),
                    "torque_rmse_by_dof_Nm": dict(_val_tau_fullset_rmse),
                    "torque_rmse_bilateral_Nm": dict(_val_tau_bilateral_rmse),
                    "torque_mae_pct_norm_by_dof": dict(_val_tau_mae_pct_norm),
                    "torque_mae_pct_norm_bilateral": dict(_val_tau_mae_pct_norm_bilateral),
                    "torque_mae_percent_bilateral_stance": dict(_val_stance_tau_mae_pct_norm_bilateral),
                    "grf_mae_percent_bw_bilateral": dict(_val_grf_mae_pct_bw_bilateral),
                },
                "loss_components_raw": {
                    "train": {k: float(train_metrics[k]) for k in train_metrics},
                    "val": {k: float(val_metrics[k]) for k in val_metrics},
                },
            }
        )
        last_epoch_snapshot = _epoch_snapshot

        if epoch in set(args.save_model_epochs):
            epoch_checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch:04d}.pkl")
            print(f"💾 Saving requested epoch checkpoint to {epoch_checkpoint_path}...", flush=True)
            with open(epoch_checkpoint_path, "wb") as f:
                pickle.dump({
                    "params": state.params,
                    "normalizers": normalizers,
                    "train_trials": train_trials,
                    "val_trials": val_trials,
                    "best_val_loss": best_val_loss,
                    "best_torque_score": best_torque_score,
                    "train_cop_rmse_m": _train_cop_overall_rmse,
                    "train_grf_rmse_N": _train_grf_overall_rmse,
                    "train_moments_rmse_Nm": _train_mom_overall_rmse,
                    "val_cop_rmse_m": _fs_cop_overall_rmse,
                    "val_grf_rmse_N": _fs_grf_overall_rmse,
                    "val_moments_rmse_Nm": _fs_mom_overall_rmse,
                    "input_dim": int(input_dim),
                    "static_dim": int(static_dim),
                    "output_dim": int(total_output_dim),
                    "qfrc_inverse_output_dim": int(qfrc_inverse_output_dim),
                    "rotation_output_dim": int(rotation_output_dim),
                    "rotation_parameterization": ROTATION_PARAMETERIZATION,
                    "rotation_compose_order": ROTATION_COMPOSE_ORDER,
                    "jacobian_output_dim": jacobian_output_dim,
                    "UseGRFNormCOP": bool(args.UseGRFNormCOP),
                    "use_GRF_NoFilt": args.use_GRF_NoFilt,
                    "includeJacobianInput": bool(args.includeJacobianInput),
                    "saved_epoch": int(epoch),
                }, f)
            saved_epoch_checkpoints.append(epoch_checkpoint_path)
            print("✅ Requested epoch checkpoint saved successfully", flush=True)
        
        # Decide whether this epoch is the new best.
        _is_new_best = False
        if args.BestModelByTorque and not np.isnan(_epoch_torque_score):
            # Torque-score mode: only update when torque RMSE is available.
            if _epoch_torque_score < best_torque_score:
                best_torque_score = _epoch_torque_score
                best_val_loss = float(val_metrics["total_loss"])  # keep for reference
                _is_new_best = True
                _best_label = f"torque score: {best_torque_score:.4f} Nm"
        else:
            # Default mode (also used as fallback when torque RMSE is unavailable).
            if float(val_metrics["total_loss"]) < best_val_loss:
                best_val_loss = float(val_metrics["total_loss"])
                _is_new_best = True
                _best_label = f"val loss: {best_val_loss:.4f}"

        if _is_new_best:
            print(f"\n🎯 NEW BEST MODEL! {_best_label}", flush=True)
            print(f"   Saving checkpoint to {args.output_dir}/best_model.pkl...", flush=True)
            best_epoch_snapshot = json.loads(json.dumps(_epoch_snapshot))
            with open(os.path.join(args.output_dir, "best_model.pkl"), "wb") as f:
                pickle.dump({
                    "params": state.params,
                    "normalizers": normalizers,
                    "train_trials": train_trials,
                    "val_trials": val_trials,
                    "best_val_loss": best_val_loss,
                    "best_torque_score": best_torque_score,
                    "train_cop_rmse_m": _train_cop_overall_rmse,
                    "train_grf_rmse_N": _train_grf_overall_rmse,
                    "train_moments_rmse_Nm": _train_mom_overall_rmse,
                    "val_cop_rmse_m": _fs_cop_overall_rmse,
                    "val_grf_rmse_N": _fs_grf_overall_rmse,
                    "val_moments_rmse_Nm": _fs_mom_overall_rmse,
                    "input_dim": int(input_dim),
                    "static_dim": int(static_dim),
                    "output_dim": int(total_output_dim),
                    "qfrc_inverse_output_dim": int(qfrc_inverse_output_dim),
                    "rotation_output_dim": int(rotation_output_dim),
                    "rotation_parameterization": ROTATION_PARAMETERIZATION,
                    "rotation_compose_order": ROTATION_COMPOSE_ORDER,
                    "jacobian_output_dim": jacobian_output_dim,
                    "UseGRFNormCOP": bool(args.UseGRFNormCOP),
                    "use_GRF_NoFilt": args.use_GRF_NoFilt,
                }, f)
            print(f"✅ Checkpoint saved successfully", flush=True)

            best_model_epoch = int(epoch)
            _candidate_best_pred_path = os.path.join(args.output_dir, f"predictions_epoch_{epoch:04d}.png")
            if (
                (not args.save_final_predictions_only)
                and (not os.path.exists(_candidate_best_pred_path))
            ):
                try:
                    # Ensure best-epoch prediction panel exists even when vis_interval skips this epoch.
                    if last_train_batch is not None and last_train_pred is not None:
                        train_vis_batch = last_train_batch
                        train_vis_pred = last_train_pred
                    else:
                        train_vis_batch = next(iter(train_loader))
                        _vis_norm = normalize_batch(train_vis_batch, normalizers)
                        train_vis_pred = state.apply_fn(
                            {"params": state.params},
                            _vis_norm["input"],
                            _vis_norm["static_context"],
                            train=False,
                        )

                    if last_val_batch is not None and last_val_pred is not None:
                        val_vis_batch = last_val_batch
                        val_vis_pred = last_val_pred
                    else:
                        val_vis_batch = next(iter(val_loader))
                        _vis_norm = normalize_batch(val_vis_batch, normalizers)
                        val_vis_pred = state.apply_fn(
                            {"params": state.params},
                            _vis_norm["input"],
                            _vis_norm["static_context"],
                            train=False,
                        )

                    plot_predictions(
                        train_vis_batch,
                        train_vis_pred,
                        val_vis_batch,
                        val_vis_pred,
                        normalizers,
                        epoch,
                        args.output_dir,
                        train_sample_idx=0,
                        val_sample_idx=0,
                        train_trial_names=None,
                        val_trial_names=None,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        loss_weights=loss_weights,
                        epoch_time=epoch_time,
                        cop_mask=args.cop_mask,
                        use_grf_norm_cop=args.UseGRFNormCOP,
                        val_fullset_stats=val_fullset_stats,
                    )
                    print(
                        f"✅ Saved best-epoch prediction plot to: {_candidate_best_pred_path}",
                        flush=True,
                    )
                except Exception as _best_plot_exc:
                    print(
                        f"⚠️  Could not generate best-epoch prediction plot: {_best_plot_exc}",
                        flush=True,
                    )
            if os.path.exists(_candidate_best_pred_path):
                best_predictions_plot_path = _candidate_best_pred_path
                if final_prediction_plot_path is None and epoch == args.epochs:
                    final_prediction_plot_path = _candidate_best_pred_path

        _wandb_log["val/best_torque_score_Nm"] = (
            float(best_torque_score) if np.isfinite(best_torque_score) else float("nan")
        )
        wandb_logger.log(_wandb_log, step=epoch)

        # Keep host RAM stable across long/multi-agent runs.
        gc.collect()
    
    print("\n" + "=" * 70, flush=True)
    print(f"🎉 TRAINING COMPLETE!", flush=True)
    
    # Save hyperparameters
    hyperparams = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
        "dropout_rate": args.dropout_rate,
        "base_config_id": args.base_config_id if str(args.base_config_id).strip() else None,
        "input_dim": int(input_dim),
        "static_dim": int(static_dim),
        "window_size": args.window_size,
        "stride": args.stride,
        "prediction_margin_frames": args.prediction_margin_frames,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "cop_weight": args.cop_weight,
        "grf_weight": args.grf_weight,
        "moments_weight": args.moments_weight,
        "contact_weight": args.contact_weight,
        "contact_weight_multiplier": args.contact_weight_multiplier,
        "magWeight": args.magWeight,
        "torque_weight": args.torque_weight,
        "qfrc_inverse_weight": args.qfrc_inverse_weight,
        "qfrc_inverse_input_reg_weight": resolved_qfrc_inverse_input_reg_weight,
        "rotation_weight": args.rotation_weight,
        "rotation_input_reg_weight": resolved_rotation_input_reg_weight,
        "jacobian_weight": 0.0,
        "jacobian_input_reg_weight": resolved_jacobian_input_reg_weight,
        "grf_correction_weight": args.grf_correction_weight,
        "output_reg_weight": 0.0,
        "hip_add_r_weight": args.hip_add_r_weight,
        "knee_r_weight": args.knee_r_weight,
        "ankle_r_weight": args.ankle_r_weight,
        "subtalar_r_weight": args.subtalar_r_weight,
        "hip_add_l_weight": args.hip_add_l_weight,
        "knee_l_weight": args.knee_l_weight,
        "ankle_l_weight": args.ankle_l_weight,
        "subtalar_l_weight": args.subtalar_l_weight,
        "lumbar_extension_weight": args.lumbar_extension_weight,
        "lumbar_bending_weight": args.lumbar_bending_weight,
        "lumbar_rotation_weight": args.lumbar_rotation_weight,
        "output_dim": int(total_output_dim),
        "qfrc_inverse_output_dim": int(qfrc_inverse_output_dim),
        "rotation_output_dim": int(rotation_output_dim),
        "rotation_parameterization": ROTATION_PARAMETERIZATION,
        "rotation_compose_order": ROTATION_COMPOSE_ORDER,
        "jacobian_output_dim": int(jacobian_output_dim),
        "cop_mask": args.cop_mask,
        "UseNoised": args.UseNoised,
        "NoisedGT": args.NoisedGT,
        "UseGRFNormCOP": args.UseGRFNormCOP,
        "UseOSFiltering": args.UseOSFiltering,
        "use_GRF_NoFilt": args.use_GRF_NoFilt,
        "includeJacobianInput": args.includeJacobianInput,
        "use_contact_weighting": args.use_contact_weighting,
        "magOnOff": args.magOnOff,
        "contactOnOff": args.contactOnOff,
        "trim_cop": args.trim_cop,
        "contact_boolean_is_model_input": False,
        "input_feature_blocks": input_layout["blocks"] if input_layout is not None else [],
        "input_feature_total_dim_from_blocks": int(input_layout["total_dim"]) if input_layout is not None else None,
        "input_layout_sample_trial": input_layout["sample_trial"] if input_layout is not None else None,
    }
    with open(os.path.join(args.output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)
    print(f"📝 Saved hyperparameters to: {os.path.join(args.output_dir, 'hyperparameters.json')}", flush=True)
    model_parameters_yaml_path = os.path.join(args.output_dir, "model_parameters.yaml")
    save_model_parameters_yaml(hyperparams, model_parameters_yaml_path)
    print(f"📝 Saved model parameters to: {model_parameters_yaml_path}", flush=True)
    rmse_history = {
        "train": {
            "cop_overall_rmse_m": train_rmse_history["cop"],
            "grf_overall_rmse_N": train_rmse_history["grf"],
            "moments_overall_rmse_Nm": train_rmse_history["moments"],
        },
        "val": {
            "cop_overall_rmse_m": val_rmse_history["cop"],
            "grf_overall_rmse_N": val_rmse_history["grf"],
            "moments_overall_rmse_Nm": val_rmse_history["moments"],
        },
    }
    rmse_history_path = os.path.join(args.output_dir, "rmse_history.json")
    with open(rmse_history_path, "w", encoding="utf-8") as f:
        json.dump(rmse_history, f, indent=2)
    print(f"📝 Saved RMSE history to: {rmse_history_path}", flush=True)

    # Phase-2 WandB artifacts: best prediction plot, final loss history, latest outlier plot.
    loss_history_plot_path = os.path.join(args.output_dir, "loss_history.png")
    best_model_checkpoint_path = os.path.join(args.output_dir, "best_model.pkl")
    _prediction_candidates = sorted(glob(os.path.join(args.output_dir, "predictions_epoch_*.png")))
    if _prediction_candidates:
        final_prediction_plot_path = _prediction_candidates[-1]
    if args.save_best_model_png_only:
        best_model_png_path = os.path.join(args.output_dir, "best_model.png")
        best_epoch_plot_path = None
        if best_model_epoch is not None:
            candidate = os.path.join(args.output_dir, f"predictions_epoch_{best_model_epoch:04d}.png")
            if os.path.exists(candidate):
                best_epoch_plot_path = candidate
        if best_epoch_plot_path is None and best_predictions_plot_path and os.path.exists(best_predictions_plot_path):
            best_epoch_plot_path = best_predictions_plot_path
        if best_epoch_plot_path is None and _prediction_candidates:
            best_epoch_plot_path = _prediction_candidates[-1]
        if best_epoch_plot_path is not None:
            try:
                shutil.copy2(best_epoch_plot_path, best_model_png_path)
                best_predictions_plot_path = best_model_png_path
                final_prediction_plot_path = best_model_png_path
                print(f"✅ Saved single best-model prediction PNG to: {best_model_png_path}", flush=True)
            except OSError as _copy_exc:
                print(f"⚠️  Could not create best_model.png: {_copy_exc}", flush=True)
        for _old_prediction_path in sorted(glob(os.path.join(args.output_dir, "predictions_epoch_*.png"))):
            try:
                os.remove(_old_prediction_path)
            except OSError as _rm_exc:
                print(f"⚠️  Could not remove old prediction plot {_old_prediction_path}: {_rm_exc}", flush=True)
        _prediction_candidates = [best_model_png_path] if os.path.exists(best_model_png_path) else []
    if args.save_final_predictions_only and len(_prediction_candidates) > 1:
        for _old_prediction_path in _prediction_candidates[:-1]:
            try:
                os.remove(_old_prediction_path)
            except OSError as _rm_exc:
                print(f"⚠️  Could not remove old prediction plot {_old_prediction_path}: {_rm_exc}", flush=True)
        _prediction_candidates = _prediction_candidates[-1:]
        final_prediction_plot_path = _prediction_candidates[-1]
    if best_predictions_plot_path is None and best_model_epoch is not None and not args.save_final_predictions_only:
        _candidate = os.path.join(args.output_dir, f"predictions_epoch_{best_model_epoch:04d}.png")
        if os.path.exists(_candidate):
            best_predictions_plot_path = _candidate
    if last_val_outlier_plot_path is None:
        _outlier_candidates = sorted(glob(os.path.join(args.output_dir, "validation_outliers_epoch_*.png")))
        if _outlier_candidates:
            last_val_outlier_plot_path = _outlier_candidates[-1]
    if args.save_final_predictions_only and best_predictions_plot_path is None and best_model_epoch == args.epochs:
        best_predictions_plot_path = final_prediction_plot_path

    training_summary = _json_compatible(
        {
            "status": "completed",
            "exp_name": args.exp_name,
            "output_dir": args.output_dir,
            "selection_mode": "best_torque_score" if args.BestModelByTorque else "val_total_loss",
            "selection_metric_name": "val/best_torque_score_Nm" if args.BestModelByTorque else "val/total_loss",
            "best_val_loss": best_val_loss,
            "best_torque_score": best_torque_score,
            "best_model_epoch": best_model_epoch,
            "epochs_requested": int(args.epochs),
            "epochs_completed": int(args.epochs),
            "saved_epoch_checkpoints": saved_epoch_checkpoints,
            "final_train_total_loss": float(train_loss_history[-1]) if train_loss_history else None,
            "final_val_total_loss": float(val_loss_history[-1]) if val_loss_history else None,
            "final_weighted_torque_score_Nm": last_epoch_torque_score,
            "final_train_cop_rmse_m": float(train_rmse_history["cop"][-1]) if train_rmse_history["cop"] else None,
            "final_train_grf_rmse_N": float(train_rmse_history["grf"][-1]) if train_rmse_history["grf"] else None,
            "final_train_moments_rmse_Nm": float(train_rmse_history["moments"][-1]) if train_rmse_history["moments"] else None,
            "final_val_cop_rmse_m": float(val_rmse_history["cop"][-1]) if val_rmse_history["cop"] else None,
            "final_val_grf_rmse_N": float(val_rmse_history["grf"][-1]) if val_rmse_history["grf"] else None,
            "final_val_moments_rmse_Nm": float(val_rmse_history["moments"][-1]) if val_rmse_history["moments"] else None,
            "final_val_torque_rmse_Nm": (
                None if last_epoch_snapshot is None else last_epoch_snapshot["val"].get("torque_rmse_fullset_overall_Nm")
            ),
            "best_epoch_metrics": best_epoch_snapshot,
            "final_epoch_metrics": last_epoch_snapshot,
            "artifacts": {
                "best_predictions_plot": best_predictions_plot_path if best_predictions_plot_path and os.path.exists(best_predictions_plot_path) else None,
                "final_prediction_plot": final_prediction_plot_path if final_prediction_plot_path and os.path.exists(final_prediction_plot_path) else None,
                "loss_history_plot": loss_history_plot_path if os.path.exists(loss_history_plot_path) else None,
                "latest_validation_outliers_plot": last_val_outlier_plot_path if last_val_outlier_plot_path and os.path.exists(last_val_outlier_plot_path) else None,
                "best_model_checkpoint": best_model_checkpoint_path if os.path.exists(best_model_checkpoint_path) else None,
                "hyperparameters_path": os.path.join(args.output_dir, "hyperparameters.json"),
                "model_parameters_yaml_path": model_parameters_yaml_path,
                "rmse_history_path": rmse_history_path if os.path.exists(rmse_history_path) else None,
            },
            "hpo_flags": {
                "save_final_predictions_only": bool(args.save_final_predictions_only),
                "save_best_model_png_only": bool(args.save_best_model_png_only),
                "disable_validation_outlier_plots": bool(args.disable_validation_outlier_plots),
            },
        }
    )
    training_summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(training_summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    print(f"📝 Saved training summary to: {training_summary_path}", flush=True)

    if best_predictions_plot_path and os.path.exists(best_predictions_plot_path):
        _aliases = ["best"]
        if best_model_epoch is not None:
            _aliases.append(f"epoch_{best_model_epoch:04d}")
        wandb_logger.log_artifact(
            best_predictions_plot_path,
            artifact_type="plot",
            name="predictions_best_epoch",
            aliases=_aliases,
        )
        wandb_logger.save_file(best_predictions_plot_path)
    if os.path.exists(loss_history_plot_path):
        wandb_logger.log_artifact(
            loss_history_plot_path,
            artifact_type="plot",
            name="loss_history_final",
            aliases=["final"],
        )
        wandb_logger.save_file(loss_history_plot_path)
    if last_val_outlier_plot_path and os.path.exists(last_val_outlier_plot_path):
        wandb_logger.log_artifact(
            last_val_outlier_plot_path,
            artifact_type="plot",
            name="validation_outliers_latest",
            aliases=["latest"],
        )
        wandb_logger.save_file(last_val_outlier_plot_path)
    if os.path.exists(best_model_checkpoint_path):
        _model_aliases = ["best"]
        if best_model_epoch is not None:
            _model_aliases.append(f"epoch_{best_model_epoch:04d}")
        wandb_logger.log_artifact(
            best_model_checkpoint_path,
            artifact_type="model",
            name="best_model_checkpoint",
            aliases=_model_aliases,
        )
        wandb_logger.save_file(best_model_checkpoint_path)
    if os.path.exists(model_parameters_yaml_path):
        wandb_logger.log_artifact(
            model_parameters_yaml_path,
            artifact_type="config",
            name="model_parameters_yaml",
            aliases=["final"],
        )
        wandb_logger.save_file(model_parameters_yaml_path)
    if os.path.exists(rmse_history_path):
        wandb_logger.log_artifact(
            rmse_history_path,
            artifact_type="metrics",
            name="rmse_history",
            aliases=["final"],
        )
        wandb_logger.save_file(rmse_history_path)
    
    wandb_logger.set_summary(
        {
            "status": "completed",
            "best_val_loss": None if not np.isfinite(best_val_loss) else float(best_val_loss),
            "best_torque_score": None if not np.isfinite(best_torque_score) else float(best_torque_score),
            "best_model_epoch": best_model_epoch,
            "final_train_total_loss": float(train_loss_history[-1]) if train_loss_history else None,
            "final_val_total_loss": float(val_loss_history[-1]) if val_loss_history else None,
            "final_train_cop_rmse_m": float(train_rmse_history["cop"][-1]) if train_rmse_history["cop"] else None,
            "final_train_grf_rmse_N": float(train_rmse_history["grf"][-1]) if train_rmse_history["grf"] else None,
            "final_train_moments_rmse_Nm": float(train_rmse_history["moments"][-1]) if train_rmse_history["moments"] else None,
            "final_val_cop_rmse_m": float(val_rmse_history["cop"][-1]) if val_rmse_history["cop"] else None,
            "final_val_grf_rmse_N": float(val_rmse_history["grf"][-1]) if val_rmse_history["grf"] else None,
            "final_val_moments_rmse_Nm": float(val_rmse_history["moments"][-1]) if val_rmse_history["moments"] else None,
            "epochs_completed": int(args.epochs),
            "output_dir": args.output_dir,
            "best_predictions_plot": best_predictions_plot_path,
            "final_prediction_plot": final_prediction_plot_path,
            "loss_history_plot": loss_history_plot_path if os.path.exists(loss_history_plot_path) else None,
            "latest_validation_outliers_plot": last_val_outlier_plot_path,
            "rmse_history_path": rmse_history_path if os.path.exists(rmse_history_path) else None,
            "training_summary_path": training_summary_path,
        }
    )
    wandb_logger.finish()


if __name__ == "__main__":
    main()
