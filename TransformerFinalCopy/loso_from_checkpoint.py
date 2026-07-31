"""Run LOSO fine-tuning from a pretrained Transformer checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import pickle
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Thread caps must be set BEFORE jax/BLAS import so per-fold workers stay lean and
# the inner-validation phase can run several folds in parallel without oversubscribing
# CPU cores. setdefault keeps any value the caller already exported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_LOG_COMPILES", "0")
os.environ.setdefault("MJX_DATALOADER_QUIET", "1")
os.environ.setdefault("JAX_CPU_THREADS", "1")
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import jax
import jax.numpy as jnp
import numpy as np

try:
    from wandb_utils import configure_runtime_env
except ModuleNotFoundError:
    def configure_runtime_env():
        return {}

RUNTIME_ENV_APPLIED = configure_runtime_env()

import train as train_module
import infer as infer_module
from data_loader import TrialDataLoader, load_single_trial, subject_group_id
import loso_adapters

# =============================================================================
# USER CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LOSO_CONFIG: Dict[str, Any] = {
    # Required: point this at the trained checkpoint you want to fine-tune/evaluate with LOSO.
    "checkpoint": str(PROJECT_ROOT / "outputs" / "TrustedDataSetNoised12Distributed" / "Sweeps" / "HPO_April24" / "local_hpo_20260424_232420" / "runs" / "April24_Sweep_trial_0049_dm256_nl7_lr0.000312653_dr0.326502_9827ea69" / "best_model.pkl"),
    # Optional paths
    "data_dir": None,    # Defaults to the OpenCapSubjects dataset used previously
    "output_dir": str(PROJECT_ROOT / "inference_results" / "Top_April24_Sweep_Trial49_KAM_OS_GTV2"),
    # Fine-tuning basics
    "epochs": 2,
    "learning_rate": 5e-5,
    "batch_size": None,
    "weight_decay": 0.001,
    "adapter_hidden_dim": None,
    "adapter_dropout_rate": None,
    "seed": 42,
    # ---- Inner-validation epoch selection (LOSO_HPO-style, epochs only) ----
    # When enabled, each outer fold first runs an inner leave-one-subject-out sweep
    # over the held-in subjects: every held-in subject becomes the validation subject
    # once (training on the rest), each inner model is trained ONCE up to
    # `inner_max_epochs` while being evaluated at every candidate epoch, the objective
    # is averaged across the inner validation subjects per epoch, and the epoch with
    # the best mean is used to train/test the final held-out model.
    "inner_epoch_selection": True,
    # Max epochs trained during the inner sweep. None -> falls back to `epochs`.
    "inner_max_epochs": None,
    # Explicit candidate epochs to evaluate (e.g. [1, 2, 4, 7]). None -> every epoch
    # from 1..inner_max_epochs.
    "inner_eval_epochs": None,
    # Objective used to rank epochs (lower is better), averaged across inner subjects.
    "inner_selection_objective": "selected_left_stance_moment_mae_percent_bwh_mean",
    # Penalize cross-subject variance like the HPO selection score: mean + w*std.
    "inner_selection_std_weight": 0.25,
    # Parallel inner training jobs (across outer x inner folds). Each worker is capped
    # to a single CPU thread via the env vars above.
    "max_parallel_splits": 3,
    # Housekeeping to bound memory during the long inner sweep.
    "jax_cache_clear_every_inner": 12,
    "jax_cache_clear_memory_fraction": 0.85,
    # Subjects excluded from training/validation roles across all folds.
    # Each listed subject still gets its OWN held-out fold (tested with all others as
    # the training set), but is never included in the training or inner-validation set
    # when another subject is being held out.  Use this for subjects whose data may
    # be noisy or structurally different enough to bias generalisation estimates.
    "training_excluded_subjects": [],
    # Loss weights / regularizers
    "cop_weight": 0,
    "grf_weight": 0,
    "moments_weight": None,
    "contact_weight": None,
    "torque_weight": 0.65,
    "qfrc_inverse_weight": 0,
    "qfrc_inverse_input_reg_weight": .1,
    "rotation_weight": 0,
    "rotation_input_reg_weight": .1,
    "rotation_residual_max_deg": None,
    "jacobian_weight": 0,
    "jacobian_input_reg_weight": .1,
    "grf_correction_weight": None,
    "output_reg_weight": None,
    # Behavior toggles
    "contact_weight_multiplier": None,
    "magWeight": None,
    "use_contact_weighting": None,
    "magOnOff": None,
    "contactOnOff": None,
    "cop_mask": None,
    "UseGRFNormCOP": None,
    "use_OpenSimID_GT": True,
    "use_recalculated_opensim_id_gt": False,
    # When True: the fine-tuned model is trained with the GT (MoCap) Jacobian and
    # rotation matrix for the torque reconstruction, the torque target is forced to the
    # MoCap qfrc_grf_contribution (even under recalc), and only COP/GRF/grf_contribution
    # losses are active. Validation/metric still uses the video (ProcessedData) terms.
    "useGTJacobAndRotForTraining": False,
    # When True: the fine-tuned OpenCap eval reconstructs torque with the GT (MoCap)
    # Jacobian, rotation, and qfrc_inverse (model still predicts COP/GRF from OpenCap
    # inputs). Isolates how much of the COP/GRF signal the OpenCap kinematics capture.
    "useGTJacobAndRotForEval": False,
    "BestModelByTorque": None,
    "BestModel_TorqueWeighting": None,  # dict or JSON string
    "torque_grad_through_jacob": None,
    # Per-DOF torque weights
    "hip_add_r_weight": None,
    "knee_r_weight": 1.5,
    "ankle_r_weight": 1.4,
    "subtalar_r_weight": None,
    "hip_add_l_weight": None,
    "knee_l_weight": 1.5,
    "ankle_l_weight": 1.4,
    "subtalar_l_weight": None,
    "lumbar_extension_weight": None,
    "lumbar_bending_weight": None,
    "lumbar_rotation_weight": None,
}


OPEN_CAP_LOSO_SUBJECTS: Tuple[str, ...] = (
    "subject2",
    "subject3",
    "subject4",
    "subject5",
    "subject6",
    "subject7",
    "subject8",
    "subject9",
    "subject10",
    "subject11",
)
KEY_TAU_DOFS = {
    "R Hip Add": 7,
    "R Knee": 9,
    "R Ankle": 10,
    "R Subtalar": 11,
    "L Hip Add": 14,
    "L Knee": 16,
    "L Ankle": 17,
    "L Subtalar": 18,
}
STANCE_MAE_TAU_DOFS = {
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
STANCE_MAE_BILATERAL_TAU_MAP = {
    "hip_flexion": ("R Hip Flexion", "L Hip Flexion"),
    "hip_adduction": ("R Hip Adduction", "L Hip Adduction"),
    "hip_rotation": ("R Hip Rotation", "L Hip Rotation"),
    "knee": ("R Knee", "L Knee"),
    "ankle": ("R Ankle", "L Ankle"),
    "subtalar": ("R Subtalar", "L Subtalar"),
}
BILATERAL_TAU_MAP = {
    "hip_add": ("R Hip Add", "L Hip Add"),
    "knee": ("R Knee", "L Knee"),
    "ankle": ("R Ankle", "L Ankle"),
    "subtalar": ("R Subtalar", "L Subtalar"),
}
BILATERAL_GRF_AXIS_MAP = {
    "x": (0, 3),
    "y": (1, 4),
    "z": (2, 5),
}


class TeeStream:
    """Mirror stdout/stderr to the terminal and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _subject_sort_key(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", str(name))
    if match:
        return int(match.group(1)), str(name)
    return (10**9, str(name))


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_coerce_jsonable(dict(payload)), indent=2), encoding="utf-8")


def _compute_source_average_mae_per_dof(
    mae_reports_by_source: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> Dict[str, Dict[str, float]]:
    return {
        str(source_label): infer_module._compute_average_mae_per_dof(dict(source_mae))
        for source_label, source_mae in mae_reports_by_source.items()
        if source_mae
    }


def _filter_joint_moment_mae_map(mae_map: Mapping[str, float]) -> Dict[str, float]:
    return {
        str(dof_name): float(value)
        for dof_name, value in mae_map.items()
        if np.isfinite(float(value))
        and not str(dof_name).upper().startswith("COP_")
        and "GRF" not in str(dof_name).upper()
    }


def _extract_bilateral_grf_mae_percent_bw(
    metrics: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    if not metrics:
        return {}

    direct_value = metrics.get("grf_mae_percent_bw_bilateral_stance")
    if isinstance(direct_value, Mapping):
        return {
            str(axis): float(value)
            for axis, value in direct_value.items()
            if value is not None and np.isfinite(float(value))
        }

    bilateral_report = metrics.get("bilateral_stance_mae_report")
    if not isinstance(bilateral_report, Mapping):
        return {}

    sides = bilateral_report.get("sides")
    if not isinstance(sides, Mapping):
        return {}

    channel_map = {
        "X": ("GRF_X_Right", "GRF_X_Left"),
        "Y": ("GRF_Y_Right", "GRF_Y_Left"),
        "Z": ("GRF_Z_Right", "GRF_Z_Left"),
    }
    result: Dict[str, float] = {}
    for axis, (right_key, left_key) in channel_map.items():
        axis_values: List[float] = []
        for side_name, channel_key in (("right", right_key), ("left", left_key)):
            side_payload = sides.get(side_name)
            if not isinstance(side_payload, Mapping):
                continue
            grf_payload = side_payload.get("grf_mae_percent_bw")
            if not isinstance(grf_payload, Mapping):
                continue
            value = grf_payload.get(channel_key)
            if value is None:
                continue
            numeric_value = float(value)
            if np.isfinite(numeric_value):
                axis_values.append(numeric_value)
        if axis_values:
            result[axis] = float(np.mean(axis_values))
    return result


def _average_metric_dicts(
    metrics_by_trial: Mapping[str, Mapping[str, float]]
) -> Dict[str, float]:
    collected: Dict[str, List[float]] = {}
    for metric_payload in metrics_by_trial.values():
        for key, value in metric_payload.items():
            numeric_value = float(value)
            if np.isfinite(numeric_value):
                collected.setdefault(str(key), []).append(numeric_value)
    return {
        key: float(np.mean(values))
        for key, values in collected.items()
        if values
    }


def _build_trial_detail_payloads(
    mae_reports_by_trial: Mapping[str, Mapping[str, float]],
    *,
    grf_by_trial: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for trial_name, mae_report in mae_reports_by_trial.items():
        normalized_mae = {
            str(dof_name): float(value)
            for dof_name, value in dict(mae_report).items()
            if np.isfinite(float(value))
        }
        torque_like_mae = {
            dof_name: value
            for dof_name, value in normalized_mae.items()
            if not dof_name.upper().startswith("COP_") and "GRF" not in dof_name.upper()
        }
        mae_values = list(torque_like_mae.values())
        trial_payload: Dict[str, Any] = {
            "torque_mae_bwh_percent": float(np.mean(mae_values)) if mae_values else None,
            "average_mae_per_dof": normalized_mae,
            "average_joint_moment_mae_per_dof": torque_like_mae,
        }
        if grf_by_trial and trial_name in grf_by_trial:
            trial_payload["grf_mae_percent_bw_bilateral_stance"] = {
                str(axis): float(value)
                for axis, value in dict(grf_by_trial[trial_name]).items()
                if np.isfinite(float(value))
            }
        payload[str(trial_name)] = trial_payload
    return payload


def _compute_subject_average_torque_mae_from_trial_details(
    trial_details: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    subject_values: Dict[str, List[float]] = {}
    for trial_name, trial_payload in trial_details.items():
        match = re.match(r"subject(\d+)/", str(trial_name))
        if match is None or not isinstance(trial_payload, Mapping):
            continue
        metric_value = trial_payload.get("torque_mae_bwh_percent")
        if metric_value is None:
            continue
        numeric_value = float(metric_value)
        if not np.isfinite(numeric_value):
            continue
        subject_name = f"subject{int(match.group(1))}"
        subject_values.setdefault(subject_name, []).append(numeric_value)

    per_subject = {
        subject_name: float(np.mean(values))
        for subject_name, values in sorted(subject_values.items(), key=lambda item: _subject_sort_key(item[0]))
        if values
    }
    subject_average_values = list(per_subject.values())
    overall_mean = float(np.mean(subject_average_values)) if subject_average_values else None
    overall_std = (
        float(np.std(subject_average_values, ddof=1))
        if len(subject_average_values) >= 2
        else 0.0 if subject_average_values else None
    )
    return {
        "subject_count": len(per_subject),
        "per_subject": per_subject,
        "mean": overall_mean,
        "std": overall_std,
    }


def _compute_subject_average_torque_mae_by_source(
    trial_details_by_source: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(source_label): _compute_subject_average_torque_mae_from_trial_details(trial_details)
        for source_label, trial_details in trial_details_by_source.items()
        if trial_details
    }


def _format_subject_average_summary_line(
    label: str,
    summary: Mapping[str, Any],
) -> Optional[str]:
    mean_value = summary.get("mean")
    if mean_value is None:
        return None
    std_value = summary.get("std")
    subject_count = int(summary.get("subject_count", 0))
    std_text = f" +/- {std_value:.3f}" if std_value is not None else ""
    count_text = f" across {subject_count} subjects" if subject_count else ""
    return f"{label}: {float(mean_value):.3f}{std_text} %BW*H{count_text}"


def _flatten_metrics(mapping: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_metrics(value, prefix=path))
        elif isinstance(value, (bool, np.bool_)):
            flat[path] = float(bool(value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if not math.isnan(numeric) and not math.isinf(numeric):
                flat[path] = numeric
    return flat


def _nested_from_flat(flat: Mapping[str, float]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for path, value in flat.items():
        cursor: MutableMapping[str, Any] = nested
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def _aggregate_metric_dicts(metrics_list: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    collected: Dict[str, List[float]] = {}
    for metrics in metrics_list:
        for key, value in _flatten_metrics(metrics).items():
            collected.setdefault(key, []).append(float(value))

    means = {key: float(np.mean(values)) for key, values in collected.items() if values}
    stds = {key: float(np.std(values)) for key, values in collected.items() if values}
    return _nested_from_flat(means), _nested_from_flat(stds)


def _write_summary_csv(path: Path, fold_rows: Sequence[Mapping[str, Any]]) -> None:
    flattened_rows = []
    all_keys: List[str] = []
    seen = set()
    for row in fold_rows:
        flat = _flatten_metrics(row)
        base = {
            "held_out_subject": row.get("held_out_subject"),
            "inner_val_subject": row.get("inner_val_subject"),
            "best_epoch": row.get("best_epoch"),
        }
        flat_row = {**base, **flat}
        flattened_rows.append(flat_row)
        for key in flat_row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in flattened_rows:
            writer.writerow({key: _coerce_jsonable(row.get(key)) for key in all_keys})


def _resolve_required_hyperparameters(checkpoint_path: Path) -> Mapping[str, Any]:
    hyperparams_path = checkpoint_path.parent / "hyperparameters.json"
    if not hyperparams_path.exists():
        raise FileNotFoundError(
            f"Required sibling hyperparameters.json was not found next to checkpoint: {hyperparams_path}"
        )
    return json.loads(hyperparams_path.read_text(encoding="utf-8"))


def _install_pickle_compat_shims() -> None:
    """Allow checkpoints pickled from train.py when it ran as __main__ to load here."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    if not hasattr(main_module, "Normalizer"):
        setattr(main_module, "Normalizer", train_module.Normalizer)


def _load_checkpoint_bundle(checkpoint_path: Path) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    _install_pickle_compat_shims()
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if "params" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'params': {checkpoint_path}")
    if "normalizers" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'normalizers': {checkpoint_path}")
    raw_hparams = _resolve_required_hyperparameters(checkpoint_path)
    normalized = loso_adapters.normalize_hyperparameters(raw_hparams, checkpoint_metadata=checkpoint)
    normalized["source_checkpoint"] = str(checkpoint_path)
    normalized["source_hyperparameters_path"] = str(checkpoint_path.parent / "hyperparameters.json")
    return checkpoint, normalized


def _build_dof_weights(config: Mapping[str, Any]) -> Optional[Dict[int, float]]:
    dof_args = {
        7: config.get("hip_add_r_weight"),
        11: config.get("knee_r_weight"),
        14: config.get("ankle_r_weight"),
        15: config.get("subtalar_r_weight"),
        18: config.get("hip_add_l_weight"),
        22: config.get("knee_l_weight"),
        25: config.get("ankle_l_weight"),
        26: config.get("subtalar_l_weight"),
    }
    if not any(value is not None for value in dof_args.values()):
        return None
    weights = {
        7: 1.0,
        11: 1.0,
        14: 1.0,
        15: 1.0,
        18: 1.0,
        22: 1.0,
        25: 1.0,
        26: 1.0,
    }
    for key, value in dof_args.items():
        if value is not None:
            weights[key] = float(value)
    return weights


def _build_loss_weights(config: Mapping[str, Any]) -> Dict[str, float]:
    # GT-Jacobian/rotation training mode supervises ONLY COP, GRF, and grf_contribution
    # (torque); every other term (free-moment, qfrc_inverse, rotation, jacobian, contact,
    # grf_correction, output_reg) is zeroed.
    if bool(config.get("use_gt_jacob_and_rot_for_training", False)):
        return {
            "cop": float(config["cop_weight"]),
            "grf": float(config["grf_weight"]),
            "moments": 0.0,
            "qfrc_inverse": 0.0,
            "qfrc_inverse_input_reg": 0.0,
            "rotation": 0.0,
            "rotation_input_reg": 0.0,
            "jacobian": 0.0,
            "jacobian_input_reg": 0.0,
            "contact": 0.0,
            "torque": float(config["torque_weight"]),
            "grf_correction": 0.0,
            "output_reg": 0.0,
        }
    return {
        "cop": float(config["cop_weight"]),
        "grf": float(config["grf_weight"]),
        "moments": float(config["moments_weight"]),
        "qfrc_inverse": float(config.get("qfrc_inverse_weight", 1.0)),
        "qfrc_inverse_input_reg": float(
            config.get("qfrc_inverse_input_reg_weight", config.get("qfrc_inverse_weight", 1.0))
        ),
        "rotation": float(config.get("rotation_weight", 1.0)),
        "rotation_input_reg": float(
            config.get("rotation_input_reg_weight", config.get("rotation_weight", 1.0))
        ),
        "jacobian": float(config.get("jacobian_weight", 1.0)) if bool(config.get("predict_jacobian", False)) else 0.0,
        "jacobian_input_reg": float(
            config.get("jacobian_input_reg_weight", config.get("jacobian_weight", 1.0))
        ) if bool(config.get("predict_jacobian", False)) else 0.0,
        "contact": float(config["contact_weight"]),
        "torque": float(config["torque_weight"]),
        "grf_correction": float(config["grf_correction_weight"]),
        "output_reg": float(config["output_reg_weight"]) if config["deviation_learning"] else 0.0,
    }


def _parse_optional_bool_arg(value):
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _normalize_cli_like_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in dict(config).items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    if isinstance(normalized.get("BestModel_TorqueWeighting"), Mapping):
        normalized["BestModel_TorqueWeighting"] = json.dumps(
            dict(normalized["BestModel_TorqueWeighting"])
        )
    return normalized


def _merge_config_into_args(args: argparse.Namespace, config: Mapping[str, Any]) -> argparse.Namespace:
    merged = argparse.Namespace(**vars(args))
    normalized_config = _normalize_cli_like_config(config)
    for key, value in normalized_config.items():
        if not hasattr(merged, key):
            continue
        if getattr(merged, key) is None and value is not None:
            setattr(merged, key, value)
    return merged


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    resolved = dict(config)
    scalar_override_map = {
        "cop_weight": "cop_weight",
        "grf_weight": "grf_weight",
        "moments_weight": "moments_weight",
        "contact_weight": "contact_weight",
        "torque_weight": "torque_weight",
        "qfrc_inverse_weight": "qfrc_inverse_weight",
        "qfrc_inverse_input_reg_weight": "qfrc_inverse_input_reg_weight",
        "rotation_weight": "rotation_weight",
        "rotation_input_reg_weight": "rotation_input_reg_weight",
        "rotation_residual_max_deg": "rotation_residual_max_deg",
        "jacobian_weight": "jacobian_weight",
        "jacobian_input_reg_weight": "jacobian_input_reg_weight",
        "grf_correction_weight": "grf_correction_weight",
        "output_reg_weight": "output_reg_weight",
        "contact_weight_multiplier": "contact_weight_multiplier",
        "magWeight": "mag_weight",
        "hip_add_r_weight": "hip_add_r_weight",
        "knee_r_weight": "knee_r_weight",
        "ankle_r_weight": "ankle_r_weight",
        "subtalar_r_weight": "subtalar_r_weight",
        "hip_add_l_weight": "hip_add_l_weight",
        "knee_l_weight": "knee_l_weight",
        "ankle_l_weight": "ankle_l_weight",
        "subtalar_l_weight": "subtalar_l_weight",
        "lumbar_extension_weight": "lumbar_extension_weight",
        "lumbar_bending_weight": "lumbar_bending_weight",
        "lumbar_rotation_weight": "lumbar_rotation_weight",
    }
    for arg_name, config_key in scalar_override_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            resolved[config_key] = value

    bool_override_map = {
        "use_contact_weighting": "use_contact_weighting",
        "magOnOff": "mag_on_off",
        "contactOnOff": "contact_on_off",
        "cop_mask": "cop_mask",
        "UseGRFNormCOP": "use_grf_norm_cop",
        "use_OpenSimID_GT": "use_OpenSimID_GT",
        "use_recalculated_opensim_id_gt": "use_recalculated_opensim_id_gt",
        "useGTJacobAndRotForTraining": "use_gt_jacob_and_rot_for_training",
        "useGTJacobAndRotForEval": "use_gt_jacob_and_rot_for_eval",
        "BestModelByTorque": "best_model_by_torque",
        "torque_grad_through_jacob": "torque_grad_through_jacob",
    }
    for arg_name, config_key in bool_override_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            resolved[config_key] = value

    if getattr(args, "BestModel_TorqueWeighting", None):
        resolved["best_model_torque_weighting"] = json.loads(args.BestModel_TorqueWeighting)

    return resolved


def _safe_trial_loader(
    trials: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
) -> TrialDataLoader:
    loader = TrialDataLoader(
        list(trials),
        window_size=int(config["window_size"]),
        stride=int(config["stride"]),
        batch_size=int(batch_size),
        shuffle=shuffle,
        trim_cop=bool(config["trim_cop"]),
        deviation_learning=bool(config["deviation_learning"]),
        use_noised=False, # OpenCapVal does not have noised inputs
        noised_gt=bool(config["noised_gt"]),
        predict_jacobian=bool(config.get("predict_jacobian", False)),
        opencap_val=True,
        input_source=str(config.get("loso_input_source", "processed")),
        include_pelvis_euler=bool(config["include_pelvis_euler"]),
        include_ankle_heights=bool(config.get("include_ankle_heights", True)),
        include_jacobian_input=bool(config.get("include_jacobian_input", True)),
        include_auxiliary_denoising_inputs=bool(
            config.get("include_auxiliary_denoising_inputs", True)
        ),
        prediction_margin_frames=int(config["prediction_margin_frames"]),
        use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
        use_recalculated_opensim_id_gt=bool(config.get("use_recalculated_opensim_id_gt", False)),
        force_gt_grf_contribution=bool(config.get("use_gt_jacob_and_rot_for_training", False)),
        grf_grm_from_processed=bool(config.get("loso_grf_grm_from_processed", False)),
        drop_last=False,
    )
    if loader.total_windows <= 0:
        raise ValueError("Loader has zero windows for the requested split.")
    safe_batch_size = min(int(batch_size), int(loader.total_windows))
    if safe_batch_size != int(batch_size):
        loader = TrialDataLoader(
            list(trials),
            window_size=int(config["window_size"]),
            stride=int(config["stride"]),
            batch_size=int(max(1, safe_batch_size)),
            shuffle=shuffle,
            trim_cop=bool(config["trim_cop"]),
            deviation_learning=bool(config["deviation_learning"]),
            use_noised=False, # OpenCapVal does not have noised inputs
            noised_gt=bool(config["noised_gt"]),
            predict_jacobian=bool(config.get("predict_jacobian", False)),
            opencap_val=True,
            input_source=str(config.get("loso_input_source", "processed")),
            include_pelvis_euler=bool(config["include_pelvis_euler"]),
            include_ankle_heights=bool(config.get("include_ankle_heights", True)),
            include_jacobian_input=bool(config.get("include_jacobian_input", True)),
            include_auxiliary_denoising_inputs=bool(
                config.get("include_auxiliary_denoising_inputs", True)
            ),
            prediction_margin_frames=int(config["prediction_margin_frames"]),
            use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
            use_recalculated_opensim_id_gt=bool(config.get("use_recalculated_opensim_id_gt", False)),
            force_gt_grf_contribution=bool(config.get("use_gt_jacob_and_rot_for_training", False)),
            grf_grm_from_processed=bool(config.get("loso_grf_grm_from_processed", False)),
            drop_last=False,
        )
    return loader


def _resolve_loso_data_dir(arg_data_dir: Optional[str]) -> Path:
    """Resolve the OpenCap LOSO dataset directory shared by both LOSO pipelines.

    An explicit ``--data_dir`` always wins. Otherwise prefer the filtered dataset
    (``OpenCapSubjects_Filt``) when present, falling back to the unfiltered
    AddBiomechanics export. Keeping this in one place ensures
    ``loso_from_checkpoint`` and ``loso_from_checkpoint_HPO`` train/validate on the
    same files.
    """
    if arg_data_dir:
        return Path(arg_data_dir).resolve()
    trunk_sway = PROJECT_ROOT / "OpenCapWalkingTrunkSwaySubjects"
    if trunk_sway.exists():
        return trunk_sway.resolve()
    filtered = PROJECT_ROOT / "OpenCapSubjects_Filt"
    if filtered.exists():
        return filtered.resolve()
    return (
        PROJECT_ROOT / "Datasets_NAS" / "AddBiomechanicsDataset_All_npy" / "OpenCapSubjects"
    ).resolve()


def _discover_subject_trials(
    data_dir: Path,
    *,
    include_trunk_sway: bool = False,
) -> Tuple[List[Mapping[str, Any]], List[str], List[str], Dict[str, List[Mapping[str, Any]]]]:
    all_subject_dirs = sorted(
        [path.name for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("subject")],
        key=_subject_sort_key,
    )
    trials = train_module.discover_all_trials(
        str(data_dir),
        refresh_cache=True,
        scan_workers=4,
        layout="opencap",
    )
    subject_to_trials: Dict[str, List[Mapping[str, Any]]] = {}
    for trial in trials:
        subject = str(trial["subject"])
        if not include_trunk_sway and subject.endswith("_TS"):
            continue
        group = str(trial.get("subject_group") or subject_group_id(str(trial["subject"])))
        trial_with_group = dict(trial)
        trial_with_group["subject_group"] = group
        subject_to_trials.setdefault(group, []).append(trial_with_group)

    valid_subjects = sorted(subject_to_trials.keys(), key=_subject_sort_key)
    all_subject_groups = sorted({subject_group_id(subject) for subject in all_subject_dirs}, key=_subject_sort_key)
    skipped_subjects = [subject for subject in all_subject_groups if subject not in valid_subjects]
    return trials, all_subject_dirs, valid_subjects, subject_to_trials


def _build_loso_folds(
    valid_subjects: Sequence[str],
    subject_to_trials: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    training_excluded_subjects: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    folds: List[Dict[str, Any]] = []
    subjects = list(sorted(valid_subjects, key=_subject_sort_key))
    if len(subjects) < 2:
        raise ValueError("LOSO evaluation requires at least 2 valid subjects.")
    excluded_set = {str(s) for s in (training_excluded_subjects or [])}

    for test_subject in subjects:
        # Excluded subjects are stripped from the training pool for every fold
        # EXCEPT when they are the held-out subject themselves (they still get
        # a full evaluation fold trained on all remaining subjects).
        train_subjects = [
            subject for subject in subjects
            if subject != test_subject
            and (subject not in excluded_set or subject == test_subject)
        ]
        folds.append(
            {
                "held_out_subject": test_subject,
                "inner_val_subject": None,
                "train_subjects": train_subjects,
                "train_trials": [trial for subject in train_subjects for trial in subject_to_trials[subject]],
                "inner_val_trials": [],
                "held_out_trials": list(subject_to_trials[test_subject]),
                "training_excluded_subjects": sorted(excluded_set - {test_subject}),
            }
        )
    return folds


def _trial_root_from_info(trial_info: Mapping[str, Any]) -> Path:
    if trial_info.get("trial_root"):
        return Path(str(trial_info["trial_root"]))
    trial_path = Path(str(trial_info["training_data_path"]))
    if trial_path.name == "ProcessedData" and trial_path.parent.name in {"Video", "MoCap"}:
        return trial_path.parent.parent
    return trial_path.parent if trial_path.name == "ProcessedData" else trial_path


def _build_static_context_from_trial_data(data: Mapping[str, Any]) -> np.ndarray:
    patient_size = np.asarray(data["patient_size"], dtype=np.float32).reshape(-1)
    static_context = np.array(
        [
            float(np.asarray(data["height"])[0, 0]),
            float(np.asarray(data["mass"])[0, 0]),
            float(data["gender"]),
            float(patient_size[0]) if patient_size.size > 0 else 0.0,
            float(patient_size[1]) if patient_size.size > 1 else 0.0,
            float(patient_size[2]) if patient_size.size > 2 else 0.0,
            float(patient_size[3]) if patient_size.size > 3 else 0.0,
            float(data["forward_vel"]),
        ],
        dtype=np.float32,
    )
    return static_context


def _resolve_fold_input_config(
    trials: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    expected_input_dim: int,
    expected_static_dim: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_config = dict(config)
    requested_include_pelvis_euler = bool(config.get("include_pelvis_euler", True))
    last_error: Optional[str] = None

    for trial_info in trials:
        trial_name = str(trial_info.get("trial_name", _trial_root_from_info(trial_info).name))
        trial_data = load_single_trial(
            _trial_root_from_info(trial_info),
            trim_cop=bool(config["trim_cop"]),
            deviation_learning=False,
            opencap_val=True,
            input_source=str(config.get("loso_input_source", "processed")),
            use_noised=False, # OpenCapVal does not have noised inputs
            noised_gt=bool(config["noised_gt"]),
            use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
            use_recalculated_opensim_id_gt=bool(config.get("use_recalculated_opensim_id_gt", False)),
        )
        if trial_data is None:
            last_error = f"failed to load sample trial data for {trial_name}"
            continue

        input_features, resolved_include_pelvis_euler, layout_name, input_blocks, input_diag = (
            infer_module._resolve_train_style_inputs(
                trial_data,
                requested_include_pelvis_euler=requested_include_pelvis_euler,
                expected_input_dim=int(expected_input_dim),
            )
        )
        resolved_input_dim = int(input_features.shape[-1])
        static_dim = int(_build_static_context_from_trial_data(trial_data).shape[-1])
        if resolved_input_dim != int(expected_input_dim):
            last_error = (
                f"sample trial {trial_name} resolved input_dim={resolved_input_dim}, "
                f"expected checkpoint input_dim={expected_input_dim}"
            )
            continue
        if expected_static_dim is not None and static_dim != int(expected_static_dim):
            last_error = (
                f"sample trial {trial_name} resolved static_dim={static_dim}, "
                f"expected checkpoint static_dim={expected_static_dim}"
            )
            continue

        resolved_config["deviation_learning"] = False
        resolved_config["include_pelvis_euler"] = bool(resolved_include_pelvis_euler)
        resolved_config["include_ankle_heights"] = any(
            str(block_name) == "ankle_heights" for block_name, _dim in input_blocks
        )
        resolved_config["include_jacobian_input"] = any(
            str(block_name) in {"jacobian_input", "jacobian_input_flat"}
            for block_name, _dim in input_blocks
        )
        resolved_config["include_auxiliary_denoising_inputs"] = any(
            str(block_name) in {"qfrc_inverse_input", "rot_w_to_ga_input_flat"}
            for block_name, _dim in input_blocks
        )
        resolved_config["input_dim"] = int(expected_input_dim)
        resolved_config["static_dim"] = int(static_dim)
        resolved_config["resolved_input_layout"] = str(layout_name)
        resolved_config["resolved_input_feature_blocks"] = [
            {"name": str(name), "dim": int(dim)} for name, dim in input_blocks
        ]
        resolved_config["resolved_input_layout_diagnostics"] = dict(input_diag)
        resolved_config["resolved_input_sample_trial"] = trial_name
        return resolved_config

    raise ValueError(
        "Unable to resolve checkpoint-compatible LOSO input layout from available trials. "
        f"Last error: {last_error}"
    )


def _augment_infer_metric_views(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    augmented = dict(metrics)
    dof_names = infer_module.get_dof_names()
    per_dof_metric_keys = (
        "torque_rmse_per_dof",
        "torque_rmse_bwh_per_dof",
        "torque_nrmse_per_dof",
    )
    for metric_key in per_dof_metric_keys:
        values = augmented.get(metric_key)
        if isinstance(values, (list, tuple, np.ndarray)):
            augmented[f"{metric_key}_named"] = {
                (dof_names[idx] if idx < len(dof_names) else f"DOF_{idx}"): float(value)
                for idx, value in enumerate(values)
            }

    cop_bias = augmented.get("cop_bias_per_channel")
    if isinstance(cop_bias, (list, tuple, np.ndarray)):
        cop_labels = ("cop_r_x", "cop_r_z", "cop_l_x", "cop_l_z")
        augmented["cop_bias_per_channel_named"] = {
            label: float(cop_bias[idx])
            for idx, label in enumerate(cop_labels)
            if idx < len(cop_bias)
        }

    grf_bias = augmented.get("grf_bias_per_channel")
    if isinstance(grf_bias, (list, tuple, np.ndarray)):
        grf_labels = ("grf_r_x", "grf_r_y", "grf_r_z", "grf_l_x", "grf_l_y", "grf_l_z")
        augmented["grf_bias_per_channel_named"] = {
            label: float(grf_bias[idx])
            for idx, label in enumerate(grf_labels)
            if idx < len(grf_bias)
        }

    return augmented


def _trim_array_mapping(mapping: Mapping[str, Any], max_len: int) -> Dict[str, Any]:
    trimmed: Dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None or isinstance(value, (str, bytes)):
            trimmed[key] = value
            continue
        try:
            value_np = np.asarray(value)
        except Exception:
            trimmed[key] = value
            continue
        if value_np.ndim >= 1 and value_np.shape[0] >= max_len:
            trimmed[key] = value_np[:max_len]
        else:
            trimmed[key] = value
    return trimmed


def _to_numpy_array(value: Any) -> np.ndarray:
    """Convert array-likes (including JAX DeviceArrays) to NumPy without copy= kwargs."""
    return np.asarray(value)


def _resolve_source_qfrc_inverse_for_plot(
    *,
    source_label: str,
    reference_ground_truth: Optional[Mapping[str, Any]],
    fallback_ground_truth: Mapping[str, Any],
    qfrc_key: str,
) -> Optional[np.ndarray]:
    """Pick the qfrc_inverse baseline used to reconstruct full-ID curves per source."""
    ref_gt = reference_ground_truth or {}
    if source_label == "Original Motion Capture":
        for key in ("qfrc_inverse_mocap", "qfrc_inverse"):
            value = ref_gt.get(key)
            if value is not None:
                return _to_numpy_array(value)
        fallback = fallback_ground_truth.get("qfrc_inverse_mocap")
        return None if fallback is None else _to_numpy_array(fallback)

    for key in ("qfrc_inverse_processed", "qfrc_inverse"):
        value = ref_gt.get(key)
        if value is not None:
            return _to_numpy_array(value)
    fallback = fallback_ground_truth.get(qfrc_key, fallback_ground_truth.get("qfrc_inverse"))
    return None if fallback is None else _to_numpy_array(fallback)


def _prepare_combined_plot_sources(
    time_axis: np.ndarray,
    ground_truth: Mapping[str, Any],
    source_entries: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    lengths = [len(time_axis), len(np.asarray(ground_truth["cop"]))]
    for source in source_entries:
        predictions = source.get("predictions")
        if predictions is None:
            continue
        lengths.append(len(np.asarray(predictions["cop"])))
    min_plot_len = min(lengths)

    plot_time_axis = np.asarray(time_axis)[:min_plot_len]
    plot_ground_truth = _trim_array_mapping(ground_truth, min_plot_len)
    prepared_sources: List[Dict[str, Any]] = []

    for source in source_entries:
        predictions = source.get("predictions")
        if predictions is None:
            continue
        evaluation_mask = source.get("evaluation_mask", predictions.get("_evaluation_mask"))
        evaluation_mask = infer_module._normalize_evaluation_mask(
            evaluation_mask,
            len(np.asarray(predictions["cop"])),
        )[:min_plot_len]
        raw_predictions = {
            key: value for key, value in predictions.items() if not str(key).startswith("_")
        }
        metric_predictions = predictions.get("_metric_view", raw_predictions)
        plot_predictions = _trim_array_mapping(
            infer_module._mask_prediction_dict_for_display(raw_predictions, evaluation_mask),
            min_plot_len,
        )
        plot_metric_predictions = _trim_array_mapping(
            infer_module._mask_prediction_dict_for_display(metric_predictions, evaluation_mask),
            min_plot_len,
        )
        prepared_sources.append(
            {
                **dict(source),
                "evaluation_mask": evaluation_mask,
                "plot_predictions": plot_predictions,
                "plot_metric_predictions": plot_metric_predictions,
                "plot_qfrc_inverse_pred": (
                    None
                    if source.get("qfrc_inverse_pred") is None
                    else np.asarray(source["qfrc_inverse_pred"])[:min_plot_len]
                ),
            }
        )
    return plot_time_axis, plot_ground_truth, prepared_sources


def _get_plot_dof_names_for_width(width: int) -> List[str]:
    """Return display names for plotted torque channels, appending KAM labels when present."""
    base_names = list(infer_module.get_dof_names())
    if width <= len(base_names):
        return base_names[:width]

    dof_names = list(base_names)
    extra_names = [
        infer_module.LEFT_STANCE_KAM_DOF_NAME,
        "knee_adduction_moment_r",
    ]
    for extra_name in extra_names:
        if len(dof_names) >= width:
            break
        dof_names.append(extra_name)
    while len(dof_names) < width:
        dof_names.append(f"DOF_{len(dof_names)}")
    return dof_names


def _resolve_ground_truth_plot_label(ground_truth: Mapping[str, Any]) -> str:
    if bool(ground_truth.get("use_recalculated_opensim_id_gt", False)):
        return "Recalculated OpenSim ID"
    if bool(ground_truth.get("use_OpenSimID_GT", False)):
        return "OpenSim ID (STO)"
    return "GT"


def _resolve_plot_torque_curves(
    source: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    *,
    use_metric_predictions: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    prediction_key = "plot_metric_predictions" if use_metric_predictions else "plot_predictions"
    predictions = source.get(prediction_key)
    if not isinstance(predictions, Mapping):
        return None, None, "missing_predictions"

    qfrc_inverse_pred = source.get("plot_qfrc_inverse_pred")
    full_id_pred, full_id_gt, id_source = infer_module.compute_full_id_curves(
        predictions,
        ground_truth,
        qfrc_inverse_override=qfrc_inverse_pred,
    )
    if full_id_pred is not None and full_id_gt is not None:
        return np.asarray(full_id_pred), np.asarray(full_id_gt), id_source

    tau_pred = predictions.get("tau_grf")
    tau_gt = ground_truth.get("tau_grf")
    if tau_pred is None or tau_gt is None:
        return None, None, "missing_tau_grf"
    return np.asarray(tau_pred), np.asarray(tau_gt), "tau_grf_contribution"


def _create_multi_prediction_timeseries_plot(
    time_axis: np.ndarray,
    ground_truth: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    trial_name: str,
    side: str,
    save_path: Path,
    prediction_margin_frames: int,
) -> None:
    fig = infer_module.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            f"COP {side} X", f"COP {side} Z", f"COP {side} Y (derived)",
            f"GRF {side} X", f"GRF {side} Y", f"GRF {side} Z",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    gt_color = "#2E86AB"
    mjx_color = "#6C757D"
    if side == "Right":
        cop_indices = [0, 1]
        grf_indices = [0, 1, 2]
    else:
        cop_indices = [2, 3]
        grf_indices = [3, 4, 5]

    metric_parts: List[str] = []
    default_mask = np.ones(len(time_axis), dtype=bool)
    for i in range(3):
        if i < 2:
            idx = cop_indices[i]
            gt_val = np.asarray(ground_truth["cop"])[:, idx]
        else:
            gt_val = np.zeros(len(time_axis), dtype=np.float32)
        fig.add_trace(
            infer_module.go.Scatter(
                x=time_axis,
                y=gt_val,
                name="Ground Truth",
                line=dict(color=gt_color, width=2),
                legendgroup="gt",
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )
        for source_index, source in enumerate(sources):
            pred_cop = np.asarray(source["plot_predictions"]["cop"])
            if i < 2:
                pred_val = pred_cop[:, idx]
            else:
                pred_val = np.zeros(len(time_axis), dtype=np.float32)
            fig.add_trace(
                infer_module.go.Scatter(
                    x=time_axis,
                    y=pred_val,
                    name=str(source["label"]),
                    line=dict(color=str(source["color"]), width=2, dash=str(source["dash"])),
                    legendgroup=f"src_{source_index}",
                    showlegend=(i == 0),
                ),
                row=1,
                col=i + 1,
            )
        fig.update_yaxes(title_text="Position (m)", row=1, col=i + 1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=i + 1)

    for i, idx in enumerate(grf_indices):
        fig.add_trace(
            infer_module.go.Scatter(
                x=time_axis,
                y=np.asarray(ground_truth["grf"])[:, idx],
                name="Ground Truth",
                line=dict(color=gt_color, width=2),
                legendgroup="gt",
                showlegend=False,
            ),
            row=2,
            col=i + 1,
        )
        for source_index, source in enumerate(sources):
            fig.add_trace(
                infer_module.go.Scatter(
                    x=time_axis,
                    y=np.asarray(source["plot_predictions"]["grf"])[:, idx],
                    name=str(source["label"]),
                    line=dict(color=str(source["color"]), width=2, dash=str(source["dash"])),
                    legendgroup=f"src_{source_index}",
                    showlegend=False,
                ),
                row=2,
                col=i + 1,
            )
        fig.update_yaxes(title_text="Force (N)", row=2, col=i + 1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=i + 1)

    for source in sources:
        mask = source.get("evaluation_mask", default_mask)
        cop_rmse = infer_module._masked_rmse(
            np.asarray(source["plot_metric_predictions"]["cop"])[:, cop_indices],
            np.asarray(ground_truth["cop"])[:, cop_indices],
            mask,
        )
        grf_rmse = infer_module._masked_rmse(
            np.asarray(source["plot_metric_predictions"]["grf"])[:, grf_indices],
            np.asarray(ground_truth["grf"])[:, grf_indices],
            mask,
        )
        metric_parts.append(f"{source['label']}: COP {cop_rmse:.4f} m, GRF {grf_rmse:.1f} N")

    fig.update_layout(
        title=dict(
            text=f"<b>{trial_name}</b><br><span style='font-size:12px'>{' | '.join(metric_parts)}</span>",
            x=0.5,
            y=0.97,
        ),
        height=700,
        width=1200,
        margin=dict(t=110, b=40, l=60, r=30),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        hovermode="x unified",
    )
    infer_module._add_prediction_margin_shading(fig, time_axis, prediction_margin_frames)
    fig.write_html(str(save_path))


def _create_multi_prediction_error_distribution_plot(
    ground_truth: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    trial_name: str,
    save_path: Path,
) -> None:
    fig = infer_module.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["COP (m)", "GRF (N)", "Moment (Nm)", "Torque (Nm)"],
    )
    subplot_specs = [
        ("COP (m)", "cop", 1, 1),
        ("GRF (N)", "grf", 1, 2),
        ("Moment (Nm)", "moments", 2, 1),
        ("Torque (Nm)", "tau_grf", 2, 2),
    ]
    annotation_lines: List[str] = []
    for title, key, row, col in subplot_specs:
        for source in sources:
            mask = np.asarray(source["evaluation_mask"], dtype=bool)
            if key == "tau_grf":
                qfrc_inverse_pred = source.get("plot_qfrc_inverse_pred")
                pred_full_id, gt_full_id, _source = infer_module.compute_full_id_curves(
                    source["plot_metric_predictions"],
                    ground_truth,
                    qfrc_inverse_override=qfrc_inverse_pred,
                )
                if pred_full_id is not None and gt_full_id is not None:
                    pred = np.asarray(pred_full_id)
                    gt = np.asarray(gt_full_id)
                    title = "Full ID Torque (Nm)"
                else:
                    pred = np.asarray(source["plot_metric_predictions"][key])
                    gt = np.asarray(ground_truth[key])
            else:
                pred = np.asarray(source["plot_metric_predictions"][key])
                gt = np.asarray(ground_truth[key])
            err = (pred[mask] - gt[mask]).reshape(-1)
            rmse = float(np.sqrt(np.mean(err ** 2))) if err.size > 0 else float("nan")
            annotation_lines.append(f"{title} {source['label']}: {rmse:.4f}")
            fig.add_trace(
                infer_module.go.Histogram(
                    x=err,
                    name=str(source["label"]),
                    marker_color=str(source["color"]),
                    opacity=0.55,
                    nbinsx=50,
                    legendgroup=str(source["label"]),
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(title_text="Count", row=row, col=col)

    fig.update_layout(
        title=dict(
            text=f"<b>Error Distributions: {trial_name}</b>",
            x=0.5,
            y=0.95,
        ),
        barmode="overlay",
        height=520,
        width=1100,
        margin=dict(t=100, b=60, l=60, r=60),
        template="plotly_white",
    )
    if annotation_lines:
        fig.add_annotation(
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            text="<br>".join(annotation_lines[:12]),
            showarrow=False,
            align="left",
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="gray",
            borderwidth=1,
        )
    fig.write_html(str(save_path))


def _create_multi_prediction_all_dofs_plot(
    time_axis: np.ndarray,
    ground_truth: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    trial_name: str,
    save_path: Path,
    prediction_margin_frames: int,
) -> None:
    gt_color = "#2E86AB"
    mjx_color = "#6C757D"
    rmse_parts: List[str] = []
    source_torque_curves: List[Tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    n_dofs: Optional[int] = None
    for source in sources:
        torque_pred, torque_gt, _torque_source = _resolve_plot_torque_curves(
            source,
            ground_truth,
            use_metric_predictions=False,
        )
        torque_metric_pred, torque_metric_gt, _metric_source = _resolve_plot_torque_curves(
            source,
            ground_truth,
            use_metric_predictions=True,
        )
        if (
            torque_pred is None
            or torque_gt is None
            or torque_metric_pred is None
            or torque_metric_gt is None
        ):
            continue
        current_width = min(
            int(torque_pred.shape[1]),
            int(torque_gt.shape[1]),
            int(torque_metric_pred.shape[1]),
            int(torque_metric_gt.shape[1]),
        )
        n_dofs = current_width if n_dofs is None else min(n_dofs, current_width)
        rmse = infer_module._masked_rmse(
            torque_metric_pred[:, :current_width],
            torque_metric_gt[:, :current_width],
            source["evaluation_mask"],
        )
        rmse_parts.append(f"{source['label']}: {rmse:.2f} Nm")
        source_torque_curves.append((source, torque_pred, torque_gt, torque_metric_pred))

    if not source_torque_curves or n_dofs is None or n_dofs <= 0:
        return

    dof_names = _get_plot_dof_names_for_width(int(n_dofs))
    n_cols = 8
    n_rows = (int(n_dofs) + n_cols - 1) // n_cols
    subplot_titles = [
        dof_names[idx][:20] + "..." if len(dof_names[idx]) > 20 else dof_names[idx]
        for idx in range(int(n_dofs))
    ]
    fig = infer_module.make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )

    _selected_gt, metric_reference_label, mjx_gt_full, opensim_gt_full, opensim_mask = (
        infer_module.resolve_full_id_reference_curves(ground_truth)
    )
    gt_plot_label = _resolve_ground_truth_plot_label(ground_truth)
    for dof_idx in range(int(n_dofs)):
        row = dof_idx // n_cols + 1
        col = dof_idx % n_cols + 1
        dof_name = dof_names[dof_idx]
        if opensim_gt_full is not None and (
            opensim_mask is None or (dof_idx < len(opensim_mask) and opensim_mask[dof_idx])
        ):
            fig.add_trace(
                infer_module.go.Scatter(
                    x=time_axis,
                    y=np.asarray(opensim_gt_full)[:, dof_idx],
                    name=gt_plot_label,
                    line=dict(color=gt_color, width=1.4),
                    legendgroup="gt",
                    showlegend=(dof_idx == 0),
                    hovertemplate=(
                        f"<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>{gt_plot_label}: %{{y:.2f}} Nm<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        if mjx_gt_full is not None:
            fig.add_trace(
                infer_module.go.Scatter(
                    x=time_axis,
                    y=np.asarray(mjx_gt_full)[:, dof_idx],
                    name="MJX_ID",
                    line=dict(color=mjx_color, width=1.4, dash="dot"),
                    legendgroup="mjx",
                    showlegend=(dof_idx == 0),
                    hovertemplate=f"<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>MJX_ID: %{{y:.2f}} Nm<extra></extra>",
                ),
                row=row,
                col=col,
            )
        for source_index, (source, torque_pred, _torque_gt, _torque_metric_pred) in enumerate(source_torque_curves):
            fig.add_trace(
                infer_module.go.Scatter(
                    x=time_axis,
                    y=torque_pred[:, dof_idx],
                    name=str(source["label"]),
                    line=dict(color=str(source["color"]), width=1.4, dash=str(source["dash"])),
                    legendgroup=f"src_{source_index}",
                    showlegend=(dof_idx == 0),
                    hovertemplate=f"<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>{source['label']}: %{{y:.2f}} Nm<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Time (s)", row=row, col=col, showticklabels=True)
        fig.update_yaxes(title_text="Torque (Nm)", row=row, col=col, showticklabels=True)

    fig.update_layout(
        title=dict(
            text=f"<b>{trial_name}</b><br><span style='font-size:12px'>{' | '.join(rmse_parts)} | Ref: {metric_reference_label}</span>",
            x=0.5,
            y=0.98,
        ),
        height=220 + 160 * n_rows,
        width=1700,
        margin=dict(t=110, b=40, l=60, r=30),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        hovermode="x unified",
    )
    infer_module._add_prediction_margin_shading(fig, time_axis, prediction_margin_frames)
    fig.write_html(str(save_path))

    dof_names = infer_module.get_dof_names()
    name_to_idx = {name: idx for idx, name in enumerate(dof_names[:n_dofs])}
    joint_groups = {
        "knee_joints": (
            "Knee Joints",
            [
                name_to_idx[name]
                for name in ("knee_angle_r", "knee_angle_l")
                if name in name_to_idx
            ],
        ),
        "ankle_joints": (
            "Ankle & Foot Joints",
            [
                name_to_idx[name]
                for name in (
                    "ankle_angle_r",
                    "ankle_angle_l",
                    "subtalar_angle_r",
                    "subtalar_angle_l",
                    "mtp_angle_r",
                    "mtp_angle_l",
                )
                if name in name_to_idx
            ],
        ),
        "hip_joints": (
            "Hip Joints",
            [
                name_to_idx[name]
                for name in (
                    "hip_flexion_r",
                    "hip_adduction_r",
                    "hip_rotation_r",
                    "hip_flexion_l",
                    "hip_adduction_l",
                    "hip_rotation_l",
                )
                if name in name_to_idx
            ],
        ),
    }
    base_path = Path(save_path)
    for file_stub, (group_name, joint_indices) in joint_groups.items():
        if not joint_indices:
            continue
        _create_multi_prediction_joint_group_plot(
            time_axis,
            ground_truth,
            sources,
            trial_name=trial_name,
            group_name=group_name,
            joint_indices=joint_indices,
            save_path=base_path.with_name(f"{base_path.stem}_{file_stub}.html"),
            prediction_margin_frames=prediction_margin_frames,
        )


def _create_multi_prediction_joint_group_plot(
    time_axis: np.ndarray,
    ground_truth: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    trial_name: str,
    group_name: str,
    joint_indices: Sequence[int],
    save_path: Path,
    prediction_margin_frames: int,
) -> None:
    if not joint_indices:
        return

    n_cols = 2
    n_rows = (len(joint_indices) + n_cols - 1) // n_cols
    gt_color = "#2E86AB"
    mjx_color = "#6C757D"
    source_torque_curves: List[Tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    group_metric_parts: List[str] = []
    subplot_titles: List[str] = []
    max_width = 0

    for source in sources:
        torque_pred, torque_gt, _torque_source = _resolve_plot_torque_curves(
            source,
            ground_truth,
            use_metric_predictions=False,
        )
        torque_metric_pred, torque_metric_gt, _metric_source = _resolve_plot_torque_curves(
            source,
            ground_truth,
            use_metric_predictions=True,
        )
        if (
            torque_pred is None
            or torque_gt is None
            or torque_metric_pred is None
            or torque_metric_gt is None
        ):
            continue
        current_width = min(
            int(torque_pred.shape[1]),
            int(torque_gt.shape[1]),
            int(torque_metric_pred.shape[1]),
            int(torque_metric_gt.shape[1]),
        )
        max_width = max(max_width, current_width)
        group_rmse = infer_module._masked_rmse(
            torque_metric_pred[:, joint_indices],
            torque_metric_gt[:, joint_indices],
            source["evaluation_mask"],
        )
        group_metric_parts.append(f"{source['label']}: {group_rmse:.2f} Nm")
        source_torque_curves.append((source, torque_pred, torque_gt, torque_metric_pred))

    if not source_torque_curves:
        return

    dof_names = _get_plot_dof_names_for_width(max_width)
    full_id_gt = source_torque_curves[0][2]
    _selected_gt, metric_reference_label, mjx_gt_full, opensim_gt_full, opensim_mask = (
        infer_module.resolve_full_id_reference_curves(ground_truth)
    )
    gt_plot_label = _resolve_ground_truth_plot_label(ground_truth)
    for dof_idx in joint_indices:
        dof_name = dof_names[dof_idx] if dof_idx < len(dof_names) else f"DOF_{dof_idx}"
        per_source_parts = []
        for source, _torque_pred, _torque_gt, torque_metric_pred in source_torque_curves:
            dof_rmse = infer_module._masked_rmse(
                torque_metric_pred[:, dof_idx],
                full_id_gt[:, dof_idx],
                source["evaluation_mask"],
            )
            per_source_parts.append(f"{source['label']}: {dof_rmse:.2f}")
        subtitle = " | ".join(per_source_parts)
        if subtitle:
            subplot_titles.append(
                f"{dof_name}<br><span style='font-size:10px;color:gray'>{subtitle} Nm</span>"
            )
        else:
            subplot_titles.append(dof_name)

    fig = infer_module.make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    for plot_idx, dof_idx in enumerate(joint_indices):
        row = plot_idx // n_cols + 1
        col = plot_idx % n_cols + 1
        dof_name = dof_names[dof_idx] if dof_idx < len(dof_names) else f"DOF_{dof_idx}"
        if opensim_gt_full is not None and (
            opensim_mask is None or (dof_idx < len(opensim_mask) and opensim_mask[dof_idx])
        ):
            fig.add_trace(
                infer_module.go.Scatter(
                    x=time_axis,
                    y=np.asarray(opensim_gt_full)[:, dof_idx],
                    name=gt_plot_label,
                    line=dict(color=gt_color, width=2),
                    legendgroup="gt",
                    showlegend=(plot_idx == 0),
                    hovertemplate=(
                        f"<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>{gt_plot_label}: %{{y:.2f}} Nm<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        if mjx_gt_full is not None:
            fig.add_trace(
                infer_module.go.Scatter(
                    x=time_axis,
                    y=np.asarray(mjx_gt_full)[:, dof_idx],
                    name="MJX_ID",
                    line=dict(color=mjx_color, width=2, dash="dot"),
                    legendgroup="mjx",
                    showlegend=(plot_idx == 0),
                    hovertemplate=f"<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>MJX_ID: %{{y:.2f}} Nm<extra></extra>",
                ),
                row=row,
                col=col,
            )
        for source_index, (source, torque_pred, _torque_gt, _torque_metric_pred) in enumerate(source_torque_curves):
            fig.add_trace(
                infer_module.go.Scatter(
                    x=time_axis,
                    y=torque_pred[:, dof_idx],
                    name=str(source["label"]),
                    line=dict(color=str(source["color"]), width=2, dash=str(source["dash"])),
                    legendgroup=f"src_{source_index}",
                    showlegend=(plot_idx == 0),
                    hovertemplate=f"<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>{source['label']}: %{{y:.2f}} Nm<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Time (s)", row=row, col=col, showticklabels=True)
        fig.update_yaxes(title_text="Torque (Nm)", row=row, col=col, showticklabels=True)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{group_name}: {trial_name}</b><br>"
                f"<span style='font-size:12px'>{' | '.join(group_metric_parts)} | Ref: {metric_reference_label}</span>"
            ),
            x=0.5,
            y=0.98,
        ),
        height=240 + 220 * n_rows,
        width=1200,
        margin=dict(t=110, b=40, l=60, r=30),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        hovermode="x unified",
    )
    infer_module._add_prediction_margin_shading(fig, time_axis, prediction_margin_frames)
    fig.write_html(str(save_path))


def _create_multi_source_stance_analysis_plot(
    *,
    output_dir: Path,
    trial_name: str,
    source_stance_results: Mapping[str, Optional[Mapping[str, Any]]],
    source_mae_reports: Mapping[str, Optional[Mapping[str, float]]],
    source_styles: Mapping[str, Mapping[str, str]],
) -> None:
    all_dofs = sorted(
        {
            dof_name
            for stance_results in source_stance_results.values()
            if stance_results
            for dof_name in stance_results.keys()
        }
    )
    if not all_dofs:
        return

    cols = 4
    rows = (len(all_dofs) + cols - 1) // cols
    fig, axes = infer_module.plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    legend_handles = []
    legend_labels = []
    for idx, dof_name in enumerate(all_dofs):
        ax = axes[idx]
        gt_drawn = False
        title_parts = []
        for source_label, stance_results in source_stance_results.items():
            if not stance_results:
                continue
            stance_entry = stance_results.get(dof_name)
            if stance_entry is None:
                continue
            style = source_styles[source_label]
            gt_mean = np.mean(np.asarray(stance_entry["gt"]), axis=0)
            pred_mean = np.mean(np.asarray(stance_entry["pred"]), axis=0)
            if not gt_drawn:
                gt_line, = ax.plot(gt_mean, color="#2E86AB", linewidth=2.4, label="Ground Truth")
                if "Ground Truth" not in legend_labels:
                    legend_handles.append(gt_line)
                    legend_labels.append("Ground Truth")
                gt_drawn = True
            pred_line, = ax.plot(
                pred_mean,
                color=style["color"],
                linewidth=2.0,
                linestyle=style["linestyle"],
                label=source_label,
            )
            if source_label not in legend_labels:
                legend_handles.append(pred_line)
                legend_labels.append(source_label)
            mae_report = source_mae_reports.get(source_label) or {}
            if dof_name in mae_report and np.isfinite(mae_report[dof_name]):
                title_parts.append(f"{source_label}: {float(mae_report[dof_name]):.2f}")

        ax.set_title(f"{dof_name}\n" + " | ".join(title_parts), fontsize=10)
        ax.set_xlabel("% Stance")
        if "COP" in dof_name:
            ax.set_ylabel("COP (% height)")
        elif "GRF" in dof_name:
            ax.set_ylabel("GRF (%BW)")
        else:
            ax.set_ylabel("Torque (%BW*H)")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    for idx in range(len(all_dofs), len(axes)):
        axes[idx].axis("off")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
    fig.suptitle(f"Stance Analysis Comparison: {trial_name}", fontsize=16, fontweight="bold")
    infer_module.plt.tight_layout(rect=[0, 0, 1, 0.93])
    infer_module.plt.savefig(output_dir / f"{trial_name}_stance_analysis.png", dpi=150)
    infer_module.plt.close(fig)


def _create_multi_source_mae_boxplots(
    mae_by_source: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_dir: Path,
    subject_average_mae_by_source: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    print("\n📦 Generating multi-source MAE box plots...", flush=True)
    source_order = [label for label, data in mae_by_source.items() if data]
    if not source_order:
        return

    all_dofs = set()
    for trial_mae_by_source in mae_by_source.values():
        for trial_mae in trial_mae_by_source.values():
            all_dofs.update(trial_mae.keys())

    cop_dofs = sorted(d for d in all_dofs if d.upper().startswith("COP_"))
    ankle_dofs = sorted(d for d in all_dofs if "ankle" in d.lower() or "subtalar" in d.lower() or "mtp" in d.lower())
    knee_dofs = sorted(d for d in all_dofs if "knee" in d.lower())
    grf_dofs = sorted(d for d in all_dofs if "grf" in d.upper())
    groups = {
        "COP DOFs": cop_dofs,
        "Ankle DOFs": ankle_dofs,
        "Knee DOFs": knee_dofs,
        "GRF DOFs": grf_dofs,
    }

    colors = {
        "LOSO Fine-Tuned": "#E94F37",
        "Original OpenCap PredInput": "#1B9E77",
        "Original OpenCap OCInput": "#8E44AD",
        "Original Motion Capture": "#264653",
    }
    linestyles = {
        "LOSO Fine-Tuned": "-",
        "Original OpenCap PredInput": "--",
        "Original OpenCap OCInput": ":",
        "Original Motion Capture": "-.",
    }

    fig, axes = infer_module.plt.subplots(1, len(groups), figsize=(8 * len(groups), 8))
    axes = np.atleast_1d(axes)
    legend_handles = []
    legend_labels = []

    for axis_index, (group_name, dofs) in enumerate(groups.items()):
        ax = axes[axis_index]
        if not dofs:
            ax.text(0.5, 0.5, f"No data for {group_name}", ha="center", va="center")
            continue

        labels = []
        per_source_data: Dict[str, List[List[float]]] = {label: [] for label in source_order}
        for dof in dofs:
            per_dof_values: Dict[str, List[float]] = {}
            source_has_any = False
            for source_label in source_order:
                values = [
                    float(trial_mae[dof])
                    for trial_mae in mae_by_source[source_label].values()
                    if dof in trial_mae and np.isfinite(trial_mae[dof])
                ]
                per_dof_values[source_label] = values if values else [np.nan]
                source_has_any = source_has_any or bool(values)
            if source_has_any:
                for source_label in source_order:
                    per_source_data[source_label].append(per_dof_values[source_label])
                labels.append(dof.replace("_", "\n"))

        if not labels:
            ax.text(0.5, 0.5, f"No valid MAE values for {group_name}", ha="center", va="center")
            continue

        positions = np.arange(1, len(labels) + 1, dtype=np.float32)
        width = min(0.7 / max(len(source_order), 1), 0.22)
        offsets = np.linspace(
            -width * (len(source_order) - 1) / 2.0,
            width * (len(source_order) - 1) / 2.0,
            len(source_order),
        )

        for source_idx, source_label in enumerate(source_order):
            box_positions = positions + offsets[source_idx]
            bp = ax.boxplot(
                per_source_data[source_label],
                positions=box_positions,
                widths=width,
                patch_artist=True,
                manage_ticks=False,
            )
            color = colors.get(source_label, "#555555")
            for patch in bp["boxes"]:
                patch.set(facecolor=color, alpha=0.55)
            for key in ("medians", "caps", "whiskers"):
                for artist in bp[key]:
                    artist.set(color=color)
                    if key != "medians":
                        artist.set_linestyle(linestyles.get(source_label, "-"))
            if source_label not in legend_labels and bp["boxes"]:
                legend_handles.append(bp["boxes"][0])
                legend_labels.append(source_label)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(group_name, fontsize=14, fontweight="bold")
        if group_name == "COP DOFs":
            ax.set_ylabel("MAE (% height)", fontsize=12)
        elif group_name == "GRF DOFs":
            ax.set_ylabel("MAE (%BW)", fontsize=12)
        else:
            ax.set_ylabel("MAE (%BW*H)", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(
        "Stance-Phase MAE Summary: LOSO Fine-Tuned vs Original OpenCap PredInput vs OCInput vs Motion Capture",
        fontsize=16,
        fontweight="bold",
    )
    summary_lines = []
    if subject_average_mae_by_source:
        for source_label in source_order:
            summary_line = _format_subject_average_summary_line(
                source_label,
                subject_average_mae_by_source.get(source_label, {}),
            )
            if summary_line:
                summary_lines.append(summary_line)
    if summary_lines:
        fig.text(
            0.5,
            0.94,
            "Subject-avg Torque MAE: " + " | ".join(summary_lines),
            ha="center",
            va="center",
            fontsize=11,
        )
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
    infer_module.plt.tight_layout(rect=[0, 0, 1, 0.90 if summary_lines else 0.92])
    infer_module.plt.savefig(output_dir / "mae_boxplots.png", dpi=300)
    infer_module.plt.close(fig)


def _compute_stance_summary_stats(
    aggregated_stance_data: Mapping[str, Mapping[str, List[np.ndarray]]]
) -> Dict[str, Dict[str, Any]]:
    summary_stats: Dict[str, Dict[str, Any]] = {}
    for dof_name, stance_data in aggregated_stance_data.items():
        pred_segments = list(stance_data.get("pred", []))
        gt_segments = list(stance_data.get("gt", []))
        if not pred_segments or not gt_segments:
            continue
        all_preds = np.vstack(pred_segments)
        all_gts = np.vstack(gt_segments)
        diff = all_preds - all_gts
        abs_error_sum = float(np.sum(np.abs(diff)))
        squared_error_sum = float(np.sum(diff ** 2))
        element_count = int(diff.size)
        segment_count = int(all_preds.shape[0])
        summary_stats[dof_name] = {
            "MAE": float(abs_error_sum / element_count) if element_count > 0 else float("nan"),
            "RMSE": float(np.sqrt(squared_error_sum / element_count)) if element_count > 0 else float("nan"),
            "Count": segment_count,
            "ElementCount": element_count,
            "AbsErrorSum": abs_error_sum,
            "SquaredErrorSum": squared_error_sum,
        }
    return summary_stats


def _merge_stance_summary_stats(
    summary_stats_by_group: Sequence[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for group_stats in summary_stats_by_group:
        for dof_name, stats in group_stats.items():
            if not isinstance(stats, Mapping):
                continue
            entry = merged.setdefault(
                str(dof_name),
                {
                    "Count": 0,
                    "ElementCount": 0,
                    "AbsErrorSum": 0.0,
                    "SquaredErrorSum": 0.0,
                },
            )
            entry["Count"] += int(stats.get("Count", 0) or 0)
            entry["ElementCount"] += int(stats.get("ElementCount", 0) or 0)
            entry["AbsErrorSum"] += float(stats.get("AbsErrorSum", 0.0) or 0.0)
            entry["SquaredErrorSum"] += float(stats.get("SquaredErrorSum", 0.0) or 0.0)

    for entry in merged.values():
        element_count = int(entry["ElementCount"])
        if element_count > 0:
            entry["MAE"] = float(entry["AbsErrorSum"] / element_count)
            entry["RMSE"] = float(np.sqrt(entry["SquaredErrorSum"] / element_count))
        else:
            entry["MAE"] = float("nan")
            entry["RMSE"] = float("nan")
    return merged


def _write_infer_style_summary_artifacts(
    output_dir: Path,
    trial_metrics: Sequence[Mapping[str, Any]],
    overall_mae: Mapping[str, Mapping[str, float]],
    aggregated_stance_data: Mapping[str, Mapping[str, List[np.ndarray]]],
    mae_by_source: Optional[Mapping[str, Mapping[str, Mapping[str, float]]]] = None,
    aggregated_stance_data_by_source: Optional[
        Mapping[str, Mapping[str, Mapping[str, List[np.ndarray]]]]
    ] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_means, metric_stds = _aggregate_metric_dicts(trial_metrics) if trial_metrics else ({}, {})
    summary_payload: Dict[str, Any] = {
        "trial_count": len(trial_metrics),
        "mae_trial_count": len(overall_mae),
        "metric_means": metric_means,
        "metric_stds": metric_stds,
    }

    if trial_metrics:
        _write_summary_csv(output_dir / "trial_metrics.csv", trial_metrics)
        infer_module.create_summary_dashboard(list(trial_metrics), str(output_dir))

    if overall_mae:
        dof_averages = infer_module._compute_average_mae_per_dof(dict(overall_mae))
        joint_moment_dof_averages = _filter_joint_moment_mae_map(dof_averages)
        subject_average_mae = _compute_subject_average_torque_mae_from_trial_details(
            _build_trial_detail_payloads(overall_mae)
        )
        mae_report_payload = {
            "average_mae_per_dof": dof_averages,
            "average_joint_moment_mae_per_dof": joint_moment_dof_averages,
            "average_mae_per_dof_opencap_input": dof_averages,
            "trial_details": overall_mae,
            "trial_details_opencap_input": overall_mae,
            "trial_details_motioncapture_input": {},
            "subject_average_torque_mae_bwh_percent": subject_average_mae,
            "subject_average_torque_mae_bwh_percent_opencap_input": subject_average_mae,
        }
        _save_json(output_dir / "overall_mae_report.json", mae_report_payload)
        summary_payload["average_mae_per_dof"] = dof_averages
        summary_payload["subject_average_torque_mae_bwh_percent"] = subject_average_mae

    if mae_by_source:
        source_trial_details = {
            str(source_label): _build_trial_detail_payloads(source_mae)
            for source_label, source_mae in mae_by_source.items()
            if source_mae
        }
        source_average_mae_per_dof = {
            str(source_label): infer_module._compute_average_mae_per_dof(dict(source_mae))
            for source_label, source_mae in mae_by_source.items()
            if source_mae
        }
        source_average_joint_moment_mae_per_dof = {
            str(source_label): _filter_joint_moment_mae_map(source_mae)
            for source_label, source_mae in source_average_mae_per_dof.items()
            if source_mae
        }
        subject_average_mae_by_source = _compute_subject_average_torque_mae_by_source(source_trial_details)
        primary_average_mae = source_average_mae_per_dof.get("LOSO Fine-Tuned", dict(dof_averages) if overall_mae else {})
        primary_average_joint_moment_mae = source_average_joint_moment_mae_per_dof.get("LOSO Fine-Tuned", {})
        original_predinput_average_mae = source_average_mae_per_dof.get("Original OpenCap PredInput", {})
        original_predinput_average_joint_moment_mae = source_average_joint_moment_mae_per_dof.get(
            "Original OpenCap PredInput", {}
        )
        original_ocinput_average_mae = source_average_mae_per_dof.get("Original OpenCap OCInput", {})
        original_ocinput_average_joint_moment_mae = source_average_joint_moment_mae_per_dof.get(
            "Original OpenCap OCInput", {}
        )
        motioncapture_average_mae = source_average_mae_per_dof.get("Original Motion Capture", {})
        motioncapture_average_joint_moment_mae = source_average_joint_moment_mae_per_dof.get(
            "Original Motion Capture", {}
        )
        primary_trial_details = source_trial_details.get("LOSO Fine-Tuned", _build_trial_detail_payloads(overall_mae))
        original_predinput_trial_details = source_trial_details.get("Original OpenCap PredInput", {})
        original_ocinput_trial_details = source_trial_details.get("Original OpenCap OCInput", {})
        motioncapture_trial_details = source_trial_details.get("Original Motion Capture", {})
        primary_subject_average = subject_average_mae_by_source.get("LOSO Fine-Tuned", {})
        original_predinput_subject_average = subject_average_mae_by_source.get("Original OpenCap PredInput", {})
        original_ocinput_subject_average = subject_average_mae_by_source.get("Original OpenCap OCInput", {})
        motioncapture_subject_average = subject_average_mae_by_source.get("Original Motion Capture", {})
        summary_payload["average_mae_per_dof_by_source"] = source_average_mae_per_dof
        summary_payload["subject_average_torque_mae_bwh_percent_by_source"] = subject_average_mae_by_source
        _save_json(
            output_dir / "overall_mae_report.json",
            {
                "average_mae_per_dof": primary_average_mae,
                "average_joint_moment_mae_per_dof": primary_average_joint_moment_mae,
                "average_mae_per_dof_opencap_input": original_predinput_average_mae,
                "average_joint_moment_mae_per_dof_opencap_input": original_predinput_average_joint_moment_mae,
                "average_mae_per_dof_original_opencap_predinput": original_predinput_average_mae,
                "average_joint_moment_mae_per_dof_original_opencap_predinput": original_predinput_average_joint_moment_mae,
                "average_mae_per_dof_original_opencap_ocinput": original_ocinput_average_mae,
                "average_joint_moment_mae_per_dof_original_opencap_ocinput": original_ocinput_average_joint_moment_mae,
                "average_mae_per_dof_fine_tuned_opencap_input": primary_average_mae,
                "average_joint_moment_mae_per_dof_fine_tuned_opencap_input": primary_average_joint_moment_mae,
                "average_mae_per_dof_motioncapture_input": motioncapture_average_mae,
                "average_joint_moment_mae_per_dof_motioncapture_input": motioncapture_average_joint_moment_mae,
                "average_mae_per_dof_by_source": source_average_mae_per_dof,
                "average_joint_moment_mae_per_dof_by_source": source_average_joint_moment_mae_per_dof,
                "trial_details": primary_trial_details,
                "trial_details_opencap_input": original_predinput_trial_details,
                "trial_details_original_opencap_predinput": original_predinput_trial_details,
                "trial_details_original_opencap_ocinput": original_ocinput_trial_details,
                "trial_details_fine_tuned_opencap_input": primary_trial_details,
                "trial_details_motioncapture_input": motioncapture_trial_details,
                "subject_average_torque_mae_bwh_percent": primary_subject_average,
                "subject_average_torque_mae_bwh_percent_opencap_input": original_predinput_subject_average,
                "subject_average_torque_mae_bwh_percent_original_opencap_predinput": original_predinput_subject_average,
                "subject_average_torque_mae_bwh_percent_original_opencap_ocinput": original_ocinput_subject_average,
                "subject_average_torque_mae_bwh_percent_fine_tuned_opencap_input": primary_subject_average,
                "subject_average_torque_mae_bwh_percent_motioncapture_input": motioncapture_subject_average,
                "subject_average_torque_mae_bwh_percent_by_source": subject_average_mae_by_source,
            },
        )
        _create_multi_source_mae_boxplots(
            mae_by_source,
            output_dir,
            subject_average_mae_by_source=subject_average_mae_by_source,
        )
        print("Subject-averaged torque MAE used by compareMAEAcrossSub:", flush=True)
        for source_label, source_summary in subject_average_mae_by_source.items():
            summary_line = _format_subject_average_summary_line(source_label, source_summary)
            if summary_line:
                print(f"  {summary_line}", flush=True)
    elif overall_mae:
        infer_module.create_mae_boxplots(dict(overall_mae), str(output_dir))

    if aggregated_stance_data:
        with (output_dir / "aggregated_stance_data.pkl").open("wb") as handle:
            pickle.dump(dict(aggregated_stance_data), handle)
        stance_summary_stats = _compute_stance_summary_stats(aggregated_stance_data)
        _save_json(output_dir / "aggregated_stance_statistics.json", stance_summary_stats)
        summary_payload["aggregated_stance_statistics"] = stance_summary_stats

    if aggregated_stance_data_by_source:
        stance_summary_stats_by_source = {
            str(source_label): _compute_stance_summary_stats(source_stance_data)
            for source_label, source_stance_data in aggregated_stance_data_by_source.items()
            if source_stance_data
        }
        if stance_summary_stats_by_source:
            _save_json(
                output_dir / "aggregated_stance_statistics_by_source.json",
                stance_summary_stats_by_source,
            )
            summary_payload["aggregated_stance_statistics_by_source"] = stance_summary_stats_by_source

    _save_json(output_dir / "infer_style_summary.json", summary_payload)
    return summary_payload


def _run_original_checkpoint_reference_outputs(
    *,
    trial_name: str,
    trial_output_dir: Path,
    data_dir: Path,
    original_data_dir: Optional[Path],
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    references_summary: Dict[str, Any] = {}
    references_eval: Dict[str, Any] = {}
    run_specs = (
        ("original_checkpoint_opencap_predinput", "processed", data_dir),
        ("original_checkpoint_opencap_ocinput", "processed", original_data_dir or data_dir),
        ("original_checkpoint_mocap", "mocap", original_data_dir or data_dir),
    )

    seen_input_specs: Dict[Tuple[str, Path], str] = {}
    for label, input_source, run_data_dir in run_specs:
        resolved_run_dir = Path(run_data_dir).resolve()
        input_spec_key = (str(input_source).strip().lower(), resolved_run_dir)
        duplicate_of = seen_input_specs.get(input_spec_key)
        if duplicate_of is not None:
            message = (
                f"Duplicate reference source skipped: {label} uses the same "
                f"input_source={input_source} and data_dir={resolved_run_dir} as {duplicate_of}."
            )
            print(f"   ℹ️ {message}", flush=True)
            references_summary[label] = {
                "input_source": input_source,
                "data_dir": str(resolved_run_dir),
                "skipped": True,
                "duplicate_of": duplicate_of,
                "reason": message,
            }
            references_eval[label] = {
                "input_source": input_source,
                "data_dir": str(resolved_run_dir),
                "skipped": True,
                "duplicate_of": duplicate_of,
                "reason": message,
            }
            continue
        seen_input_specs[input_spec_key] = label

        print(
            f"\n🔁 Running original checkpoint reference inference ({label}, input_source={input_source})",
            flush=True,
        )
        print(
            "   Reference config: "
            f"checkpoint={config['source_checkpoint']}, "
            f"data_dir={run_data_dir}, trial={trial_name}, output_dir={trial_output_dir}",
            flush=True,
        )
        print(
            "   Reference args: "
            f"window_size={config['window_size']}, stride={config['stride']}, "
            f"prediction_margin_frames={config['prediction_margin_frames']}, "
            f"include_pelvis_euler={config['include_pelvis_euler']}, "
            f"use_OpenSimID_GT={config.get('use_OpenSimID_GT', False)}, "
            f"use_recalculated_opensim_id_gt={config.get('use_recalculated_opensim_id_gt', False)}",
            flush=True,
        )
        try:
            mae_report, metrics, predictions, ground_truth, time_axis, stance_results, _secondary_mae_report = (
                infer_module.run_inference(
                    checkpoint_path=str(config["source_checkpoint"]),
                    data_dir=str(run_data_dir),
                    trial_name=trial_name,
                    output_dir=str(trial_output_dir),
                    window_size=int(config["window_size"]),
                    stride=int(config["stride"]),
                    prediction_margin_frames=int(config["prediction_margin_frames"]),
                    no_plots=True,
                    lightweight=True,
                    make_graph=False,
                    use_noised=False,
                    include_pelvis_euler=bool(config["include_pelvis_euler"]),
                    min_trial_length=30,
                    opencap_val_dataset=True,
                    input_source=input_source,
                    use_OpenSimID_GT=bool(config.get("use_OpenSimID_GT", False)),
                    use_recalculated_opensim_id_gt=bool(
                        config.get("use_recalculated_opensim_id_gt", False)
                    ),
                )
            )
            print(
                "   ✅ Reference inference returned: "
                f"metrics={'yes' if metrics is not None else 'no'}, "
                f"mae_report={'yes' if mae_report is not None else 'no'}, "
                f"predictions={'yes' if predictions is not None else 'no'}, "
                f"stance_results={0 if not stance_results else len(stance_results)} DOFs",
                flush=True,
            )
            references_summary[label] = {
                "input_source": input_source,
                "data_dir": str(run_data_dir),
                "output_dir": str(trial_output_dir),
                "metrics": metrics,
                "mae_report": mae_report,
            }
            references_eval[label] = {
                "input_source": input_source,
                "data_dir": str(run_data_dir),
                "metrics": metrics,
                "mae_report": mae_report,
                "predictions": predictions,
                "ground_truth": ground_truth,
                "time_axis": time_axis,
                "stance_results": stance_results,
            }
            if input_source == "mocap" and isinstance(ground_truth, Mapping):
                qfrc_mocap = ground_truth.get("qfrc_inverse_mocap")
                qfrc_active = ground_truth.get("qfrc_inverse")
                print(
                    "   MoCap reference qfrc sources: "
                    f"qfrc_inverse_mocap={'yes' if qfrc_mocap is not None else 'no'}, "
                    f"active qfrc_inverse={'yes' if qfrc_active is not None else 'no'}",
                    flush=True,
                )
        except Exception as exc:
            error_traceback = traceback.format_exc()
            print(
                f"   ⚠️ Original checkpoint reference inference failed for {label}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            print(error_traceback, flush=True)
            references_summary[label] = {
                "input_source": input_source,
                "data_dir": str(run_data_dir),
                "output_dir": str(trial_output_dir),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": error_traceback,
            }
            references_eval[label] = {
                "input_source": input_source,
                "data_dir": str(run_data_dir),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": error_traceback,
            }

    return references_summary, references_eval


def _evaluate_single_trial_infer_style(
    trial_info: Mapping[str, Any],
    *,
    predict_fn,
    params: Mapping[str, Any],
    output_root: Path,
    normalizers: Mapping[str, Any],
    config: Mapping[str, Any],
    held_out_subject: str,
    inner_val_subject: Optional[str],
) -> Dict[str, Any]:
    trial_name = str(trial_info.get("trial_name", _trial_root_from_info(trial_info).name))
    safe_trial_name = trial_name.replace("/", "_")
    trial_output_dir = output_root / safe_trial_name
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(str(trial_info.get("dataset_root", _trial_root_from_info(trial_info).parents[1])))
    original_trial_dir = trial_info.get("original_trial_dir")
    original_data_dir = None
    if original_trial_dir:
        original_data_dir = Path(str(original_trial_dir)).parents[1]

    print(f"\n📂 Infer-style held-out evaluation: {trial_name}", flush=True)
    trial_data = load_single_trial(
        _trial_root_from_info(trial_info),
        trim_cop=bool(config["trim_cop"]),
        deviation_learning=False,
        opencap_val=True,
        input_source=str(config.get("loso_input_source", "processed")),
        use_noised=False, # OpenCapVal does not have noised inputs
        noised_gt=bool(config["noised_gt"]),
        use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
        use_recalculated_opensim_id_gt=bool(config.get("use_recalculated_opensim_id_gt", False)),
        grf_grm_from_processed=bool(config.get("loso_grf_grm_from_processed", False)),
    )
    if trial_data is None:
        raise RuntimeError(f"Failed to load held-out trial data for infer-style evaluation: {trial_name}")

    expected_input_dim = int(np.asarray(normalizers["input"].mean).shape[-1])
    input_features, resolved_include_pelvis_euler, input_layout_name, input_blocks, input_diag = (
        infer_module._resolve_train_style_inputs(
            trial_data,
            requested_include_pelvis_euler=bool(config["include_pelvis_euler"]),
            expected_input_dim=expected_input_dim,
        )
    )
    if int(input_features.shape[-1]) != expected_input_dim:
        raise ValueError(
            f"Infer-style input dimension mismatch for {trial_name}: "
            f"resolved {input_features.shape[-1]} but checkpoint normalizer expects {expected_input_dim}."
        )

    input_combined = np.asarray(normalizers["input"].normalize(input_features), dtype=np.float32)
    static_context = _build_static_context_from_trial_data(trial_data)
    static_combined = np.asarray(normalizers["static"].normalize(static_context), dtype=np.float32)
    if static_combined.ndim == 2:
        static_combined = static_combined.squeeze()

    print(
        f"   Input layout={input_layout_name}, "
        f"include_pelvis_euler={resolved_include_pelvis_euler}, input_dim={input_features.shape[-1]}",
        flush=True,
    )

    start_time = time.perf_counter()
    output_np_kept, output_np_metric, evaluation_mask, window_meta = infer_module._predict_with_train_style_windows(
        predict_fn=predict_fn,
        params=params,
        input_features_z=input_combined,
        static_context_z=static_combined,
        window_size=int(config["window_size"]),
        stride=int(config["stride"]),
        output_dim=int(config.get("output_dim", infer_module.STANDARD_OUTPUT_DIM)),
        prediction_margin_frames=int(config["prediction_margin_frames"]),
    )
    inference_time_ms = float((time.perf_counter() - start_time) * 1000.0)
    print(
        f"   Inference time: {inference_time_ms:.2f} ms "
        f"({window_meta['num_windows']} windows, eval_frames={window_meta['evaluation_frame_count']})",
        flush=True,
    )

    predictions = infer_module._convert_output_to_physical_predictions(
        output_np=output_np_kept,
        data=trial_data,
        normalizers=normalizers,
        detected_output_dim=int(config.get("output_dim", infer_module.STANDARD_OUTPUT_DIM)),
        cop_mask=bool(config["cop_mask"]),
        use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
        qfrc_inverse_output_dim=int(config.get("qfrc_inverse_output_dim", 0)),
        rotation_output_dim=int(config.get("rotation_output_dim", 0)),
        jacobian_output_dim=int(
            config.get(
                "jacobian_output_dim",
                config.get("PredictedJacobianDim", config.get("predicted_jacobian_dim", infer_module.PREDICTED_JACOBIAN_FLAT_DIM)),
            )
        ),
        use_gt_jacob_and_rot=bool(config.get("use_gt_jacob_and_rot_for_eval", False)),
    )
    evaluation_predictions = infer_module._convert_output_to_physical_predictions(
        output_np=output_np_metric,
        data=trial_data,
        normalizers=normalizers,
        detected_output_dim=int(config.get("output_dim", infer_module.STANDARD_OUTPUT_DIM)),
        cop_mask=bool(config["cop_mask"]),
        use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
        qfrc_inverse_output_dim=int(config.get("qfrc_inverse_output_dim", 0)),
        rotation_output_dim=int(config.get("rotation_output_dim", 0)),
        jacobian_output_dim=int(
            config.get(
                "jacobian_output_dim",
                config.get("PredictedJacobianDim", config.get("predicted_jacobian_dim", infer_module.PREDICTED_JACOBIAN_FLAT_DIM)),
            )
        ),
        use_gt_jacob_and_rot=bool(config.get("use_gt_jacob_and_rot_for_eval", False)),
    )

    opensim_id_bundle = infer_module.load_opensim_id_ground_truth_bundle(
        _trial_root_from_info(trial_info),
        target_len=len(trial_data["pos"]),
        use_recalculated=bool(config.get("use_recalculated_opensim_id_gt", False)),
    )
    mjx_id_reference = (
        infer_module.load_mjx_id_reference_ground_truth(
            _trial_root_from_info(trial_info),
            target_len=len(trial_data["pos"]),
        )
        if bool(config.get("use_recalculated_opensim_id_gt", False))
        else None
    )

    ground_truth = {
        "cop": np.array(trial_data.get("cop_gt_raw", trial_data["cop_raw"]), copy=True),
        "grf": np.array(trial_data.get("grf_gt_raw", trial_data["grf_raw"]), copy=True),
        "moments": np.array(trial_data.get("moments_gt_raw", trial_data["moments_raw"]), copy=True),
        "tau_grf": np.array(trial_data.get("tau_grf_gt", trial_data["qfrc_grf_contribution"]), copy=True),
        "id_gt_mjx": None if trial_data.get("id_gt_mjx") is None else np.array(trial_data["id_gt_mjx"], copy=True),
        "opensim_id_gt": None if opensim_id_bundle is None else np.array(opensim_id_bundle["id"], copy=True),
        "opensim_id_available_mask": None if opensim_id_bundle is None else np.array(opensim_id_bundle["available_mask"], copy=True),
        "opensim_id_source_path": None if opensim_id_bundle is None else str(opensim_id_bundle["source_path"]),
        "use_OpenSimID_GT": bool(config.get("use_OpenSimID_GT", False))
        or bool(config.get("use_recalculated_opensim_id_gt", False)),
        "use_recalculated_opensim_id_gt": bool(config.get("use_recalculated_opensim_id_gt", False)),
        "mjx_id_reference": mjx_id_reference,
        "qfrc_inverse": None if trial_data.get("qfrc_inverse") is None else np.array(trial_data["qfrc_inverse"], copy=True),
        "qfrc_inverse_processed": (
            None
            if trial_data.get("qfrc_inverse_processed") is None
            else np.array(trial_data["qfrc_inverse_processed"], copy=True)
        ),
        "qfrc_inverse_mocap": (
            None
            if trial_data.get("qfrc_inverse_mocap") is None
            else np.array(trial_data["qfrc_inverse_mocap"], copy=True)
        ),
        "source": trial_data.get("ground_truth_source", "selected input source"),
    }

    swing_r_gt = np.abs(ground_truth["grf"][:, 2]) < 5.0
    swing_l_gt = np.abs(ground_truth["grf"][:, 5]) < 5.0
    ground_truth["cop"][swing_r_gt, 0:2] = 0.0
    ground_truth["grf"][swing_r_gt, 0:3] = 0.0
    ground_truth["moments"][swing_r_gt, 0:1] = 0.0
    ground_truth["cop"][swing_l_gt, 2:4] = 0.0
    ground_truth["grf"][swing_l_gt, 3:6] = 0.0
    ground_truth["moments"][swing_l_gt, 1:2] = 0.0

    evaluation_mask = infer_module._normalize_evaluation_mask(evaluation_mask, len(trial_data["pos"]))
    if infer_module.FilterPostInfer:
        predictions["cop"] = infer_module.apply_butterworth_filter_masked(predictions["cop"], evaluation_mask)
        predictions["grf"] = infer_module.apply_butterworth_filter_masked(predictions["grf"], evaluation_mask)
        predictions["moments"] = infer_module.apply_butterworth_filter_masked(predictions["moments"], evaluation_mask)
        predictions["tau_grf"] = infer_module.apply_butterworth_filter_masked(predictions["tau_grf"], evaluation_mask)
        predictions["qfrc_grf_contribution"] = predictions["tau_grf"]

        evaluation_predictions["cop"] = infer_module.apply_butterworth_filter_masked(
            evaluation_predictions["cop"], evaluation_mask
        )
        evaluation_predictions["grf"] = infer_module.apply_butterworth_filter_masked(
            evaluation_predictions["grf"], evaluation_mask
        )
        evaluation_predictions["moments"] = infer_module.apply_butterworth_filter_masked(
            evaluation_predictions["moments"], evaluation_mask
        )
        evaluation_predictions["tau_grf"] = infer_module.apply_butterworth_filter_masked(
            evaluation_predictions["tau_grf"], evaluation_mask
        )
        evaluation_predictions["qfrc_grf_contribution"] = evaluation_predictions["tau_grf"]

        ground_truth["cop"] = infer_module.apply_butterworth_filter_masked(ground_truth["cop"], evaluation_mask)
        ground_truth["grf"] = infer_module.apply_butterworth_filter_masked(ground_truth["grf"], evaluation_mask)
        ground_truth["moments"] = infer_module.apply_butterworth_filter_masked(ground_truth["moments"], evaluation_mask)
        ground_truth["tau_grf"] = infer_module.apply_butterworth_filter_masked(
            ground_truth["tau_grf"], evaluation_mask
        )
        if ground_truth["id_gt_mjx"] is not None:
            ground_truth["id_gt_mjx"] = infer_module.apply_butterworth_filter_masked(
                ground_truth["id_gt_mjx"], evaluation_mask
            )
        if ground_truth.get("opensim_id_gt") is not None:
            ground_truth["opensim_id_gt"] = infer_module.apply_butterworth_filter_masked(
                ground_truth["opensim_id_gt"], evaluation_mask
            )
        if ground_truth.get("mjx_id_reference") is not None:
            ground_truth["mjx_id_reference"] = infer_module.apply_butterworth_filter_masked(
                ground_truth["mjx_id_reference"], evaluation_mask
            )
        if ground_truth["qfrc_inverse"] is not None:
            ground_truth["qfrc_inverse"] = infer_module.apply_butterworth_filter_masked(
                ground_truth["qfrc_inverse"], evaluation_mask
            )
        if ground_truth["qfrc_inverse_processed"] is not None:
            ground_truth["qfrc_inverse_processed"] = infer_module.apply_butterworth_filter_masked(
                ground_truth["qfrc_inverse_processed"], evaluation_mask
            )
        if ground_truth["qfrc_inverse_mocap"] is not None:
            ground_truth["qfrc_inverse_mocap"] = infer_module.apply_butterworth_filter_masked(
                ground_truth["qfrc_inverse_mocap"], evaluation_mask
            )

    evaluation_frame_count = int(np.sum(evaluation_mask))
    print(
        f"   Evaluation region: {evaluation_frame_count}/{len(trial_data['pos'])} frames "
        f"(prediction_margin_frames={config['prediction_margin_frames']})",
        flush=True,
    )

    metrics: Optional[Dict[str, Any]] = None
    if evaluation_frame_count > 0:
        mass = float(np.asarray(trial_data["mass"])[0, 0])
        height = float(np.asarray(trial_data["height"])[0, 0])
        norm_factor = mass * height * 9.8067
        torque_metric_source = "tau_grf_contribution"
        torque_reference_label = "MJX_ID"
        if bool(config.get("use_OpenSimID_GT", False)) or bool(
            config.get("use_recalculated_opensim_id_gt", False)
        ):
            full_id_metric_pred, full_id_metric_gt, full_id_metric_source = infer_module.compute_full_id_curves(
                evaluation_predictions,
                ground_truth,
            )
            if full_id_metric_pred is not None and full_id_metric_gt is not None:
                torque_pred_eval = full_id_metric_pred
                torque_gt_eval = full_id_metric_gt
                torque_metric_source = full_id_metric_source
                torque_reference_label = infer_module.resolve_full_id_reference_curves(ground_truth)[1]
            else:
                torque_pred_eval = evaluation_predictions["tau_grf"]
                torque_gt_eval = ground_truth["tau_grf"]
        else:
            torque_pred_eval = evaluation_predictions["tau_grf"]
            torque_gt_eval = ground_truth["tau_grf"]

        torque_rmse_per_dof = infer_module._masked_rmse_per_channel(
            torque_pred_eval,
            torque_gt_eval,
            evaluation_mask,
        )
        cop_bias_per_channel = infer_module._masked_mean_diff(
            evaluation_predictions["cop"],
            ground_truth["cop"],
            evaluation_mask,
        )
        grf_bias_per_channel = infer_module._masked_mean_diff(
            evaluation_predictions["grf"],
            ground_truth["grf"],
            evaluation_mask,
        )
        torque_rmse = infer_module._masked_rmse(
            torque_pred_eval,
            torque_gt_eval,
            evaluation_mask,
        )
        torque_mae = infer_module._masked_mae(
            torque_pred_eval,
            torque_gt_eval,
            evaluation_mask,
        )
        torque_rmse_bwh = (torque_rmse / norm_factor) * 100.0
        torque_rmse_bwh_per_dof = (torque_rmse_per_dof / norm_factor) * 100.0

        gt_torque = torque_gt_eval[evaluation_mask]
        gt_std = np.std(gt_torque, axis=0)
        gt_std_safe = np.where(gt_std < 1e-6, 1.0, gt_std)
        torque_nrmse_per_dof = torque_rmse_per_dof / gt_std_safe

        metrics = {
            "trial_name": trial_name,
            "held_out_subject": held_out_subject,
            "inner_val_subject": inner_val_subject,
            "cop_rmse": infer_module._masked_rmse(
                evaluation_predictions["cop"], ground_truth["cop"], evaluation_mask
            ),
            "grf_rmse": infer_module._masked_rmse(
                evaluation_predictions["grf"], ground_truth["grf"], evaluation_mask
            ),
            "moments_rmse": infer_module._masked_rmse(
                evaluation_predictions["moments"], ground_truth["moments"], evaluation_mask
            ),
            "cop_bias_per_channel": cop_bias_per_channel.tolist(),
            "grf_bias_per_channel": grf_bias_per_channel.tolist(),
            "torque_rmse": float(torque_rmse),
            "torque_rmse_bwh": float(torque_rmse_bwh),
            "torque_nrmse": float(np.mean(torque_nrmse_per_dof)),
            "torque_rmse_per_dof": torque_rmse_per_dof.tolist(),
            "torque_rmse_bwh_per_dof": torque_rmse_bwh_per_dof.tolist(),
            "torque_nrmse_per_dof": torque_nrmse_per_dof.tolist(),
            "torque_mae": float(torque_mae),
            "torque_mae_bwh": float((torque_mae / norm_factor) * 100.0),
            "torque_metric_source": torque_metric_source,
            "torque_reference_label": torque_reference_label,
            "use_OpenSimID_GT": bool(config.get("use_OpenSimID_GT", False)),
            "use_recalculated_opensim_id_gt": bool(config.get("use_recalculated_opensim_id_gt", False)),
            "inference_time_ms": inference_time_ms,
            "num_frames": int(len(trial_data["pos"])),
            "evaluation_frame_count": evaluation_frame_count,
            "window_size": int(config["window_size"]),
            "stride": int(config["stride"]),
            "prediction_margin_frames": int(config["prediction_margin_frames"]),
            "input_source": "processed",
            "input_source_label": "OpenCap",
            "input_kinematics_source": trial_data.get("input_kinematics_source", "Pos"),
            "use_noised_inputs": bool(trial_data.get("use_noised_inputs", False)),
            "ground_truth_source": trial_data.get("ground_truth_source", "selected input source"),
            "split_status": "HELD_OUT",
            "input_feature_layout": input_layout_name,
            "input_feature_blocks": [{"name": name, "dim": int(dim)} for name, dim in input_blocks],
            "input_layout_diagnostics": input_diag,
            "restrict_max_vals": None,
        }
        print(
            f"   COP RMSE={metrics['cop_rmse']:.4f} m | GRF RMSE={metrics['grf_rmse']:.1f} N | "
            f"Moments RMSE={metrics['moments_rmse']:.2f} Nm | Torque RMSE={metrics['torque_rmse']:.2f} Nm | "
            f"Torque MAE BWH={metrics['torque_mae_bwh']:.3f} % | Ref={metrics['torque_reference_label']}",
            flush=True,
        )
    else:
        print("   ⚠️ No evaluation frames remained after center-window aggregation.", flush=True)

    predictions["_metric_view"] = evaluation_predictions
    predictions["_evaluation_mask"] = evaluation_mask
    trial_display = f"{trial_name} [HELD_OUT]"

    mae_report, stance_results = infer_module.analyze_stance_phase_torques(
        evaluation_predictions,
        ground_truth,
        trial_data,
        str(trial_output_dir),
        safe_trial_name,
        no_plots=True,
        lightweight=True,
        evaluation_mask=evaluation_mask,
    )

    reference_outputs_summary, reference_outputs_eval = _run_original_checkpoint_reference_outputs(
        trial_name=trial_name,
        trial_output_dir=trial_output_dir,
        data_dir=data_dir,
        original_data_dir=original_data_dir,
        config=config,
    )
    if metrics is not None:
        if mae_report:
            metrics["stance_cop_mae_percent_height"] = infer_module._extract_stance_cop_mae_percent_height(mae_report)
        metrics = _augment_infer_metric_views(metrics)

    source_styles = {
        "LOSO Fine-Tuned": {"color": "#E94F37", "dash": "solid", "linestyle": "-"},
        "Original OpenCap PredInput": {"color": "#1B9E77", "dash": "dash", "linestyle": "--"},
        "Original OpenCap OCInput": {"color": "#8E44AD", "dash": "longdash", "linestyle": ":"},
        "Original Motion Capture": {"color": "#264653", "dash": "dot", "linestyle": "-."},
    }
    source_entries: List[Dict[str, Any]] = [
        {
            "label": "LOSO Fine-Tuned",
            "color": source_styles["LOSO Fine-Tuned"]["color"],
            "dash": source_styles["LOSO Fine-Tuned"]["dash"],
            "predictions": predictions,
            "evaluation_mask": evaluation_mask,
            "qfrc_inverse_pred": ground_truth.get("qfrc_inverse_processed", ground_truth.get("qfrc_inverse")),
        }
    ]
    reference_label_map = {
        "original_checkpoint_opencap_predinput": ("Original OpenCap PredInput", "qfrc_inverse_processed"),
        "original_checkpoint_opencap_ocinput": ("Original OpenCap OCInput", "qfrc_inverse_processed"),
        "original_checkpoint_mocap": ("Original Motion Capture", "qfrc_inverse_mocap"),
    }
    for result_key, (source_label, qfrc_key) in reference_label_map.items():
        result = reference_outputs_eval.get(result_key, {})
        if result.get("predictions") is None:
            continue
        ref_ground_truth = result.get("ground_truth") or {}
        source_entries.append(
            {
                "label": source_label,
                "color": source_styles[source_label]["color"],
                "dash": source_styles[source_label]["dash"],
                "predictions": result["predictions"],
                "evaluation_mask": result["predictions"].get("_evaluation_mask"),
                "qfrc_inverse_pred": _resolve_source_qfrc_inverse_for_plot(
                    source_label=source_label,
                    reference_ground_truth=ref_ground_truth,
                    fallback_ground_truth=ground_truth,
                    qfrc_key=qfrc_key,
                ),
            }
        )

    plot_time_axis, plot_ground_truth, prepared_sources = _prepare_combined_plot_sources(
        np.arange(len(trial_data["pos"]), dtype=np.float32) / 100.0,
        ground_truth,
        source_entries,
    )
    if metrics is not None and prepared_sources:
        _create_multi_prediction_timeseries_plot(
            plot_time_axis,
            plot_ground_truth,
            prepared_sources,
            trial_name=trial_display,
            side="Right",
            save_path=trial_output_dir / "timeseries_right.html",
            prediction_margin_frames=int(config["prediction_margin_frames"]),
        )
        _create_multi_prediction_timeseries_plot(
            plot_time_axis,
            plot_ground_truth,
            prepared_sources,
            trial_name=trial_display,
            side="Left",
            save_path=trial_output_dir / "timeseries_left.html",
            prediction_margin_frames=int(config["prediction_margin_frames"]),
        )
        _create_multi_prediction_error_distribution_plot(
            plot_ground_truth,
            prepared_sources,
            trial_name=trial_display,
            save_path=trial_output_dir / "errors.html",
        )
        _create_multi_prediction_all_dofs_plot(
            plot_time_axis,
            plot_ground_truth,
            prepared_sources,
            trial_name=trial_display,
            save_path=trial_output_dir / "all_dofs.html",
            prediction_margin_frames=int(config["prediction_margin_frames"]),
        )
        _save_json(trial_output_dir / "metrics.json", metrics)

    source_stance_results = {
        "LOSO Fine-Tuned": stance_results,
        "Original OpenCap PredInput": (reference_outputs_eval.get("original_checkpoint_opencap_predinput", {}) or {}).get("stance_results"),
        "Original OpenCap OCInput": (reference_outputs_eval.get("original_checkpoint_opencap_ocinput", {}) or {}).get("stance_results"),
        "Original Motion Capture": (reference_outputs_eval.get("original_checkpoint_mocap", {}) or {}).get("stance_results"),
    }
    source_mae_reports = {
        "LOSO Fine-Tuned": mae_report,
        "Original OpenCap PredInput": (reference_outputs_eval.get("original_checkpoint_opencap_predinput", {}) or {}).get("mae_report"),
        "Original OpenCap OCInput": (reference_outputs_eval.get("original_checkpoint_opencap_ocinput", {}) or {}).get("mae_report"),
        "Original Motion Capture": (reference_outputs_eval.get("original_checkpoint_mocap", {}) or {}).get("mae_report"),
    }
    _create_multi_source_stance_analysis_plot(
        output_dir=trial_output_dir,
        trial_name=safe_trial_name,
        source_stance_results=source_stance_results,
        source_mae_reports=source_mae_reports,
        source_styles=source_styles,
    )
    if mae_report:
        _save_json(trial_output_dir / f"{safe_trial_name}_mae_report.json", mae_report)
    _save_json(
        trial_output_dir / "comparison_mae_report.json",
        {
            label: report
            for label, report in source_mae_reports.items()
            if report is not None
        },
    )

    comparison_payload = {
        "trial_name": trial_name,
        "held_out_subject": held_out_subject,
        "inner_val_subject": inner_val_subject,
        "ground_truth_source": trial_data.get("ground_truth_source", "MoCap"),
        "loso_fine_tuned": {
            "output_dir": str(trial_output_dir),
            "metrics": metrics,
            "mae_report": mae_report,
        },
        **reference_outputs_summary,
    }
    _save_json(trial_output_dir / "model_comparison.json", comparison_payload)

    return {
        "trial_name": trial_name,
        "trial_output_dir": str(trial_output_dir),
        "metrics": metrics,
        "mae_report": mae_report,
        "stance_results": stance_results,
        "reference_stance_results": {
            "Original OpenCap PredInput": (reference_outputs_eval.get("original_checkpoint_opencap_predinput", {}) or {}).get("stance_results"),
            "Original OpenCap OCInput": (reference_outputs_eval.get("original_checkpoint_opencap_ocinput", {}) or {}).get("stance_results"),
            "Original Motion Capture": (reference_outputs_eval.get("original_checkpoint_mocap", {}) or {}).get("stance_results"),
        },
        "reference_outputs": reference_outputs_summary,
        "reference_mae_reports": {
            "Original OpenCap PredInput": (reference_outputs_summary.get("original_checkpoint_opencap_predinput", {}) or {}).get("mae_report"),
            "Original OpenCap OCInput": (reference_outputs_summary.get("original_checkpoint_opencap_ocinput", {}) or {}).get("mae_report"),
            "Original Motion Capture": (reference_outputs_summary.get("original_checkpoint_mocap", {}) or {}).get("mae_report"),
        },
        "comparison_payload": comparison_payload,
        "window_meta": window_meta,
        "evaluation_frame_count": evaluation_frame_count,
        "inference_time_ms": inference_time_ms,
    }


def _run_infer_style_evaluation(
    fold: Mapping[str, Any],
    *,
    fold_dir: Path,
    model,
    params: Mapping[str, Any],
    normalizers: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    infer_output_dir = fold_dir / "infer_style_eval"
    infer_output_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def predict_fn(model_params, x_batch, static_batch):
        return model.apply({"params": model_params}, x_batch, static_batch, train=False)

    trial_metrics: List[Dict[str, Any]] = []
    overall_mae: Dict[str, Dict[str, float]] = {}
    overall_mae_original_opencap_predinput: Dict[str, Dict[str, float]] = {}
    overall_mae_original_opencap_ocinput: Dict[str, Dict[str, float]] = {}
    overall_mae_original_motion_capture: Dict[str, Dict[str, float]] = {}
    aggregated_stance_data: Dict[str, Dict[str, List[np.ndarray]]] = {}
    aggregated_stance_data_by_source: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {
        "LOSO Fine-Tuned": {},
        "Original OpenCap PredInput": {},
        "Original OpenCap OCInput": {},
        "Original Motion Capture": {},
    }
    comparison_rows: List[Dict[str, Any]] = []
    failed_trials: List[Dict[str, str]] = []
    trial_artifacts: List[Dict[str, Any]] = []

    for trial_info in fold["held_out_trials"]:
        trial_name = str(trial_info.get("trial_name", _trial_root_from_info(trial_info).name))
        try:
            trial_result = _evaluate_single_trial_infer_style(
                trial_info,
                predict_fn=predict_fn,
                params=params,
                output_root=infer_output_dir,
                normalizers=normalizers,
                config=config,
                held_out_subject=str(fold["held_out_subject"]),
                inner_val_subject=(
                    None
                    if fold.get("inner_val_subject") is None
                    else str(fold["inner_val_subject"])
                ),
            )
            if trial_result["metrics"] is not None:
                trial_metrics.append(dict(trial_result["metrics"]))
            if trial_result["mae_report"]:
                overall_mae[trial_result["trial_name"]] = {
                    key: float(value) for key, value in dict(trial_result["mae_report"]).items()
                }
            for source_label, source_report in (trial_result.get("reference_mae_reports") or {}).items():
                if not source_report:
                    continue
                normalized_report = {
                    key: float(value) for key, value in dict(source_report).items()
                }
                if source_label == "Original OpenCap PredInput":
                    overall_mae_original_opencap_predinput[trial_result["trial_name"]] = normalized_report
                elif source_label == "Original OpenCap OCInput":
                    overall_mae_original_opencap_ocinput[trial_result["trial_name"]] = normalized_report
                elif source_label == "Original Motion Capture":
                    overall_mae_original_motion_capture[trial_result["trial_name"]] = normalized_report
            if trial_result["stance_results"]:
                for dof_name, stance_result in trial_result["stance_results"].items():
                    if stance_result is None:
                        continue
                    bucket = aggregated_stance_data.setdefault(dof_name, {"pred": [], "gt": []})
                    bucket["pred"].append(np.asarray(stance_result["pred"]))
                    bucket["gt"].append(np.asarray(stance_result["gt"]))
                    source_bucket = aggregated_stance_data_by_source["LOSO Fine-Tuned"].setdefault(
                        dof_name,
                        {"pred": [], "gt": []},
                    )
                    source_bucket["pred"].append(np.asarray(stance_result["pred"]))
                    source_bucket["gt"].append(np.asarray(stance_result["gt"]))
            for source_label, source_stance_results in (trial_result.get("reference_stance_results") or {}).items():
                if not source_stance_results:
                    continue
                source_stance_bucket = aggregated_stance_data_by_source.setdefault(str(source_label), {})
                for dof_name, stance_result in source_stance_results.items():
                    if stance_result is None:
                        continue
                    dof_bucket = source_stance_bucket.setdefault(dof_name, {"pred": [], "gt": []})
                    dof_bucket["pred"].append(np.asarray(stance_result["pred"]))
                    dof_bucket["gt"].append(np.asarray(stance_result["gt"]))
            if trial_result.get("comparison_payload"):
                comparison_rows.append(dict(trial_result["comparison_payload"]))
            trial_artifacts.append(
                {
                    "trial_name": trial_result["trial_name"],
                    "trial_output_dir": trial_result["trial_output_dir"],
                    "evaluation_frame_count": trial_result["evaluation_frame_count"],
                    "inference_time_ms": trial_result["inference_time_ms"],
                }
            )
        except Exception as exc:
            print(f"   ❌ Infer-style evaluation failed for {trial_name}: {exc}", flush=True)
            failed_trials.append({"trial_name": trial_name, "error": str(exc)})
        finally:
            gc.collect()

    if not trial_metrics:
        raise RuntimeError(
            f"Infer-style evaluation produced no successful held-out trial metrics for {fold['held_out_subject']}."
        )

    summary_payload = _write_infer_style_summary_artifacts(
        infer_output_dir,
        trial_metrics,
        overall_mae,
        aggregated_stance_data,
        mae_by_source={
            "LOSO Fine-Tuned": overall_mae,
            "Original OpenCap PredInput": overall_mae_original_opencap_predinput,
            "Original OpenCap OCInput": overall_mae_original_opencap_ocinput,
            "Original Motion Capture": overall_mae_original_motion_capture,
        },
        aggregated_stance_data_by_source=aggregated_stance_data_by_source,
    )
    comparison_metric_means: Dict[str, Any] = {}
    comparison_metric_stds: Dict[str, Any] = {}
    if comparison_rows:
        comparison_metric_means, comparison_metric_stds = _aggregate_metric_dicts(comparison_rows)
        _save_json(
            infer_output_dir / "model_comparison_summary.json",
            {
                "trial_count": len(comparison_rows),
                "metric_means": comparison_metric_means,
                "metric_stds": comparison_metric_stds,
                "per_trial": comparison_rows,
            },
        )
    summary_payload.update(
        {
            "output_dir": str(infer_output_dir),
            "trial_metrics": trial_metrics,
            "mae_reports": overall_mae,
            "mae_reports_original_opencap_predinput": overall_mae_original_opencap_predinput,
            "mae_reports_original_opencap_ocinput": overall_mae_original_opencap_ocinput,
            "mae_reports_original_motion_capture": overall_mae_original_motion_capture,
            # Backward-compatible aliases.
            "mae_reports_original_opencap": overall_mae_original_opencap_predinput,
            "mae_reports_original_motioncapture": overall_mae_original_motion_capture,
            "comparison_rows": comparison_rows,
            "comparison_metric_means": comparison_metric_means,
            "comparison_metric_stds": comparison_metric_stds,
            "failed_trials": failed_trials,
            "trial_artifacts": trial_artifacts,
        }
    )
    return summary_payload


def _collect_kam_segments_from_trial_result(
    trial_result: Mapping[str, Any],
    *,
    source_label: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect left-stance KAM prediction and GT stance-normalized segments."""
    kam_name = infer_module.LEFT_STANCE_KAM_DOF_NAME
    if source_label == "LOSO Fine-Tuned":
        source_stance = trial_result.get("stance_results") or {}
    else:
        source_stance = (trial_result.get("reference_stance_results") or {}).get(source_label) or {}
    kam_entry = source_stance.get(kam_name) if isinstance(source_stance, Mapping) else None
    if not isinstance(kam_entry, Mapping):
        return [], []
    pred = kam_entry.get("pred")
    gt = kam_entry.get("gt")
    pred_segments = [np.asarray(seg, dtype=np.float32) for seg in np.asarray(pred)] if pred is not None else []
    gt_segments = [np.asarray(seg, dtype=np.float32) for seg in np.asarray(gt)] if gt is not None else []
    return pred_segments, gt_segments


def _stack_kam_segments(segments: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    clean_segments = []
    for segment in segments:
        arr = np.asarray(segment, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        if arr.size != 101:
            x_old = np.linspace(0.0, 100.0, arr.size, dtype=np.float32)
            x_new = np.linspace(0.0, 100.0, 101, dtype=np.float32)
            arr = np.interp(x_new, x_old, arr).astype(np.float32)
        clean_segments.append(arr)
    if not clean_segments:
        return None
    return np.stack(clean_segments, axis=0)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled_var = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2)
    if pooled_var <= 1e-12:
        return float("nan")
    return float((np.mean(b) - np.mean(a)) / np.sqrt(pooled_var))


def _summarize_kam_condition_effect(
    *,
    normal_segments: Sequence[np.ndarray],
    ts_segments: Sequence[np.ndarray],
) -> Optional[Dict[str, Any]]:
    normal_stack = _stack_kam_segments(normal_segments)
    ts_stack = _stack_kam_segments(ts_segments)
    if normal_stack is None or ts_stack is None:
        return None
    normal_curve = np.nanmean(normal_stack, axis=0)
    ts_curve = np.nanmean(ts_stack, axis=0)
    effect_curve = ts_curve - normal_curve
    normal_stride_means = np.nanmean(normal_stack, axis=1)
    ts_stride_means = np.nanmean(ts_stack, axis=1)
    return {
        "normal_curve": normal_curve,
        "trunk_sway_curve": ts_curve,
        "effect_curve": effect_curve,
        "normal_n_stances": int(normal_stack.shape[0]),
        "trunk_sway_n_stances": int(ts_stack.shape[0]),
        "normal_mean": float(np.nanmean(normal_stride_means)),
        "trunk_sway_mean": float(np.nanmean(ts_stride_means)),
        "mean_effect": float(np.nanmean(ts_stride_means) - np.nanmean(normal_stride_means)),
        "peak_abs_effect": float(np.nanmax(np.abs(effect_curve))),
        "cohens_d": _cohens_d(normal_stride_means, ts_stride_means),
    }


def _plot_trunk_sway_kam_effects(
    *,
    output_dir: Path,
    held_out_subject: str,
    source_summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stance_percent = np.linspace(0.0, 100.0, 101)
    styles = {
        "Ground Truth": {"color": "#2E86AB", "linestyle": "-", "dash": "solid"},
        "LOSO Fine-Tuned": {"color": "#E94F37", "linestyle": "--", "dash": "dash"},
        "Original OpenCap": {"color": "#1B9E77", "linestyle": ":", "dash": "dot"},
    }

    fig, axes = infer_module.plt.subplots(1, 2, figsize=(15, 5))
    for source_label, summary in source_summaries.items():
        style = styles.get(source_label, {"color": "#444444", "linestyle": "-", "dash": "solid"})
        axes[0].plot(
            stance_percent,
            np.asarray(summary["normal_curve"]),
            color=style["color"],
            linestyle="-",
            linewidth=2.2,
            label=f"{source_label} normal",
        )
        axes[0].plot(
            stance_percent,
            np.asarray(summary["trunk_sway_curve"]),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            label=f"{source_label} trunk sway",
        )
        axes[1].plot(
            stance_percent,
            np.asarray(summary["effect_curve"]),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            label=source_label,
        )
    axes[0].set_title(f"{held_out_subject}: knee adduction moment")
    axes[0].set_xlabel("% left stance")
    axes[0].set_ylabel("KAM (% BW*H)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0.0, color="#888888", linewidth=1.0)
    axes[1].set_title("Trunk sway effect (TS - normal)")
    axes[1].set_xlabel("% left stance")
    axes[1].set_ylabel("Delta KAM (% BW*H)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "kam_trunk_sway_timeseries.png", dpi=200)
    infer_module.plt.close(fig)

    labels = list(source_summaries.keys())
    mean_effects = [float(source_summaries[label]["mean_effect"]) for label in labels]
    effect_sizes = [float(source_summaries[label]["cohens_d"]) for label in labels]
    colors = [styles.get(label, {}).get("color", "#444444") for label in labels]
    fig, axes = infer_module.plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, mean_effects, color=colors)
    axes[0].axhline(0.0, color="#888888", linewidth=1.0)
    axes[0].set_title("Mean TS effect")
    axes[0].set_ylabel("Delta KAM (% BW*H)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, effect_sizes, color=colors)
    axes[1].axhline(0.0, color="#888888", linewidth=1.0)
    axes[1].set_title("Effect size")
    axes[1].set_ylabel("Cohen's d")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "kam_trunk_sway_effect_sizes.png", dpi=200)
    infer_module.plt.close(fig)

    plotly_fig = infer_module.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("KAM by condition", "Trunk sway effect (TS - normal)"),
    )
    for source_label, summary in source_summaries.items():
        style = styles.get(source_label, {"color": "#444444", "dash": "solid"})
        plotly_fig.add_trace(
            infer_module.go.Scatter(
                x=stance_percent,
                y=np.asarray(summary["normal_curve"]),
                name=f"{source_label} normal",
                line=dict(color=style["color"], dash="solid"),
            ),
            row=1,
            col=1,
        )
        plotly_fig.add_trace(
            infer_module.go.Scatter(
                x=stance_percent,
                y=np.asarray(summary["trunk_sway_curve"]),
                name=f"{source_label} trunk sway",
                line=dict(color=style["color"], dash=style["dash"]),
            ),
            row=1,
            col=1,
        )
        plotly_fig.add_trace(
            infer_module.go.Scatter(
                x=stance_percent,
                y=np.asarray(summary["effect_curve"]),
                name=f"{source_label} effect",
                line=dict(color=style["color"], dash=style["dash"]),
            ),
            row=1,
            col=2,
        )
    plotly_fig.update_xaxes(title_text="% left stance")
    plotly_fig.update_yaxes(title_text="KAM (% BW*H)", row=1, col=1)
    plotly_fig.update_yaxes(title_text="Delta KAM (% BW*H)", row=1, col=2)
    plotly_fig.update_layout(template="plotly_white", height=520, title=f"{held_out_subject}: trunk-sway KAM effect")
    plotly_fig.write_html(output_dir / "kam_trunk_sway_timeseries.html")


def _build_trunk_sway_effect_summary(
    *,
    normal_results: Sequence[Mapping[str, Any]],
    trunk_sway_results: Sequence[Mapping[str, Any]],
    output_dir: Path,
    held_out_subject: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_segments = {
        "Ground Truth": {"normal": [], "ts": []},
        "LOSO Fine-Tuned": {"normal": [], "ts": []},
        "Original OpenCap": {"normal": [], "ts": []},
    }
    for condition_name, results in (("normal", normal_results), ("ts", trunk_sway_results)):
        for trial_result in results:
            fine_pred, fine_gt = _collect_kam_segments_from_trial_result(
                trial_result,
                source_label="LOSO Fine-Tuned",
            )
            orig_pred, _orig_gt = _collect_kam_segments_from_trial_result(
                trial_result,
                source_label="Original OpenCap PredInput",
            )
            source_segments["LOSO Fine-Tuned"][condition_name].extend(fine_pred)
            source_segments["Ground Truth"][condition_name].extend(fine_gt)
            source_segments["Original OpenCap"][condition_name].extend(orig_pred)

    source_summaries: Dict[str, Mapping[str, Any]] = {}
    for source_label, condition_segments in source_segments.items():
        summary = _summarize_kam_condition_effect(
            normal_segments=condition_segments["normal"],
            ts_segments=condition_segments["ts"],
        )
        if summary is not None:
            source_summaries[source_label] = summary

    serializable = {
        source_label: {
            key: (np.asarray(value).tolist() if key.endswith("_curve") else value)
            for key, value in summary.items()
        }
        for source_label, summary in source_summaries.items()
    }
    payload = {
        "held_out_subject": held_out_subject,
        "normal_trial_count": len(normal_results),
        "trunk_sway_trial_count": len(trunk_sway_results),
        "kam_dof": infer_module.LEFT_STANCE_KAM_DOF_NAME,
        "units": "percent_BW_times_height",
        "sources": serializable,
    }
    _save_json(output_dir / "kam_trunk_sway_effect_summary.json", payload)
    if source_summaries:
        _plot_trunk_sway_kam_effects(
            output_dir=output_dir,
            held_out_subject=held_out_subject,
            source_summaries=source_summaries,
        )
    return payload


def _discover_trunk_sway_trials_for_subject(
    *,
    data_dir: Path,
    held_out_subject: str,
) -> List[Mapping[str, Any]]:
    ts_subject = f"{held_out_subject}_TS"
    trials = train_module.discover_all_trials(
        str(data_dir),
        refresh_cache=False,
        scan_workers=4,
        layout="opencap",
    )
    return [
        trial
        for trial in trials
        if str(trial.get("subject")) == ts_subject
    ]


def _run_trunk_sway_effect_evaluation(
    fold: Mapping[str, Any],
    *,
    fold_dir: Path,
    data_dir: Path,
    model,
    params: Mapping[str, Any],
    normalizers: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    held_out_subject = str(fold["held_out_subject"])
    output_dir = fold_dir / "trunk_sway_effect_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_trials = _discover_trunk_sway_trials_for_subject(
        data_dir=data_dir,
        held_out_subject=held_out_subject,
    )
    if not ts_trials:
        payload = {
            "held_out_subject": held_out_subject,
            "enabled": True,
            "skipped": True,
            "reason": f"No matching trunk-sway subject folder found for {held_out_subject}_TS.",
            "output_dir": str(output_dir),
        }
        _save_json(output_dir / "kam_trunk_sway_effect_summary.json", payload)
        return payload

    @jax.jit
    def predict_fn(model_params, x_batch, static_batch):
        return model.apply({"params": model_params}, x_batch, static_batch, train=False)

    normal_results: List[Mapping[str, Any]] = []
    ts_results: List[Mapping[str, Any]] = []
    failures: List[Dict[str, str]] = []
    normal_held_out_trials = [
        trial
        for trial in fold["held_out_trials"]
        if not str(trial.get("subject", "")).endswith("_TS")
    ]
    eval_specs = (
        ("normal", normal_held_out_trials, normal_results),
        ("trunk_sway", ts_trials, ts_results),
    )
    for condition_name, condition_trials, result_bucket in eval_specs:
        condition_output = output_dir / condition_name
        condition_output.mkdir(parents=True, exist_ok=True)
        for trial_info in condition_trials:
            trial_name = str(trial_info.get("trial_name", _trial_root_from_info(trial_info).name))
            try:
                result_bucket.append(
                    _evaluate_single_trial_infer_style(
                        trial_info,
                        predict_fn=predict_fn,
                        params=params,
                        output_root=condition_output,
                        normalizers=normalizers,
                        config=config,
                        held_out_subject=held_out_subject,
                        inner_val_subject=None,
                    )
                )
            except Exception as exc:
                print(
                    f"   ⚠️ Trunk-sway effect evaluation failed for {condition_name} {trial_name}: {exc}",
                    flush=True,
                )
                failures.append(
                    {
                        "condition": condition_name,
                        "trial_name": trial_name,
                        "error": str(exc),
                    }
                )
            finally:
                gc.collect()

    payload = _build_trunk_sway_effect_summary(
        normal_results=normal_results,
        trunk_sway_results=ts_results,
        output_dir=output_dir,
        held_out_subject=held_out_subject,
    )
    payload.update(
        {
            "enabled": True,
            "output_dir": str(output_dir),
            "normal_trials": [str(result.get("trial_name")) for result in normal_results],
            "trunk_sway_trials": [str(result.get("trial_name")) for result in ts_results],
            "failed_trials": failures,
        }
    )
    _save_json(output_dir / "kam_trunk_sway_effect_summary.json", payload)
    return payload


def _compute_epoch_torque_score(
    tau_rmse_by_dof: Mapping[str, float],
    torque_weights: Mapping[str, float],
) -> float:
    score = 0.0
    found_any = False
    for joint, (dof_r, dof_l) in BILATERAL_TAU_MAP.items():
        weight = float(torque_weights.get(joint, 1.0))
        right = tau_rmse_by_dof.get(dof_r, float("nan"))
        left = tau_rmse_by_dof.get(dof_l, float("nan"))
        if math.isnan(float(right)) or math.isnan(float(left)):
            continue
        score += weight * (float(right) + float(left)) / 2.0
        found_any = True
    return score if found_any else float("nan")


def _compute_selection_metric(
    inner_val: MutableMapping[str, Any],
    *,
    use_torque_selection: bool,
    torque_weights: Mapping[str, float],
) -> Tuple[float, str]:
    selection_metric = float(inner_val["metrics"]["total_loss"])
    selection_name = "val_total_loss"
    if use_torque_selection:
        torque_score = _compute_epoch_torque_score(
            inner_val["metrics"]["torque_rmse_selected_dofs_Nm"],
            torque_weights,
        )
        if not math.isnan(torque_score):
            selection_metric = float(torque_score)
            selection_name = "weighted_torque_score_Nm"
    inner_val["selection_metric"] = {
        "name": selection_name,
        "value": selection_metric,
    }
    return selection_metric, selection_name


def _run_train_epoch(
    state,
    train_loader: TrialDataLoader,
    *,
    train_step,
    normalizers: Mapping[str, Any],
    loss_weights: Mapping[str, float],
    rng: jax.Array,
    epoch: int,
) -> Tuple[Any, Dict[str, float], jax.Array]:
    metrics_sum = {
        "cop_loss": 0.0,
        "grf_loss": 0.0,
        "moments_loss": 0.0,
        "qfrc_inverse_loss": 0.0,
        "qfrc_inverse_input_reg_loss": 0.0,
        "rotation_loss": 0.0,
        "rotation_input_reg_loss": 0.0,
        "jacobian_loss": 0.0,
        "jacobian_input_reg_loss": 0.0,
        "contact_loss": 0.0,
        "torque_loss": 0.0,
        "torque_cop_effect_loss": 0.0,
        "torque_grf_effect_loss": 0.0,
        "grf_correction_loss": 0.0,
        "output_reg_loss": 0.0,
        "total_loss": 0.0,
    }
    steps = 0
    for batch in train_loader:
        batch_norm = train_module.normalize_batch(batch, normalizers)
        rng, dropout_rng = jax.random.split(rng)
        state, step_metrics, _pred, _debug = train_step(
            state,
            batch_norm,
            loss_weights,
            dropout_rng,
            float(epoch),
        )
        for key in metrics_sum:
            metrics_sum[key] += float(step_metrics[key])
        steps += 1
    if steps == 0:
        raise RuntimeError("Training loader yielded zero batches.")
    return state, {key: value / steps for key, value in metrics_sum.items()}, rng


def _evaluate_loader(
    state,
    loader: TrialDataLoader,
    *,
    eval_step,
    normalizers: Mapping[str, Any],
    loss_weights: Mapping[str, float],
    config: Mapping[str, Any],
    epoch: int,
    require_left_kam: bool = False,
) -> Dict[str, Any]:
    val_metrics = {
        "cop_loss": 0.0,
        "grf_loss": 0.0,
        "moments_loss": 0.0,
        "qfrc_inverse_loss": 0.0,
        "qfrc_inverse_input_reg_loss": 0.0,
        "rotation_loss": 0.0,
        "rotation_input_reg_loss": 0.0,
        "jacobian_loss": 0.0,
        "jacobian_input_reg_loss": 0.0,
        "contact_loss": 0.0,
        "torque_loss": 0.0,
        "torque_cop_effect_loss": 0.0,
        "torque_grf_effect_loss": 0.0,
        "grf_correction_loss": 0.0,
        "output_reg_loss": 0.0,
        "total_loss": 0.0,
    }
    val_steps = 0
    val_cop_sumsq = np.zeros(4, dtype=np.float64)
    val_grf_sumsq = np.zeros(6, dtype=np.float64)
    val_mom_sumsq = np.zeros(2, dtype=np.float64)
    val_frames = 0
    val_tau_sumsq = {key: 0.0 for key in KEY_TAU_DOFS}
    val_tau_sumsq_all = 0.0
    val_tau_count = 0
    val_tau_frames = 0
    val_tau_mae_pct_sum = {key: 0.0 for key in KEY_TAU_DOFS}
    val_tau_mae_pct_count = {key: 0 for key in KEY_TAU_DOFS}
    val_stance_tau_mae_pct_sum = {key: 0.0 for key in STANCE_MAE_TAU_DOFS}
    val_stance_tau_mae_pct_count = {key: 0 for key in STANCE_MAE_TAU_DOFS}
    selected_left_dof_names = list(infer_module.SELECTED_LEFT_STANCE_DOF_NAMES)
    selected_left_dof_indices = list(infer_module.get_selected_left_stance_dof_indices())
    selected_left_mae_pct_sum = {key: 0.0 for key in selected_left_dof_names}
    selected_left_mae_pct_count = {key: 0 for key in selected_left_dof_names}
    selected_left_kam_name = infer_module.LEFT_STANCE_KAM_DOF_NAME
    selected_left_kam_sum = 0.0
    selected_left_kam_count = 0
    saw_left_kam_vectors = False
    val_grf_mae_pct_bw_sum = {axis: 0.0 for axis in BILATERAL_GRF_AXIS_MAP}
    val_grf_mae_pct_bw_count = {axis: 0 for axis in BILATERAL_GRF_AXIS_MAP}

    for batch in loader:
        batch_norm = train_module.normalize_batch(batch, normalizers)
        step_metrics, step_pred = eval_step(state, batch_norm, loss_weights, float(epoch))
        for key in val_metrics:
            val_metrics[key] += float(step_metrics[key])
        val_steps += 1

        pred_np = np.array(step_pred)
        static_np = np.array(batch["static_context"])
        h_batch = static_np[:, 0:1, None]
        m_batch = static_np[:, 1:2, None]

        if config["deviation_learning"]:
            cop_pred_ratio = pred_np[..., :4] * normalizers["cop"].std + np.array(batch["cop_recon"])
            grf_pred_ratio = pred_np[..., 4:10] * normalizers["grf"].std + np.array(batch["grf_recon"])
            moments_pred_ratio = pred_np[..., 10:12] * normalizers["moments"].std + np.array(batch["moment_recon"])
        else:
            cop_pred_ratio = normalizers["cop"].unnormalize(pred_np[..., :4])
            grf_pred_ratio = normalizers["grf"].unnormalize(pred_np[..., 4:10])
            moments_pred_ratio = normalizers["moments"].unnormalize(pred_np[..., 10:12])

        grf_pred_phys = grf_pred_ratio * m_batch * 9.8067
        cop_pred_phys = train_module.decode_cop_signal_to_length(
            cop_pred_ratio,
            grf_pred_ratio,
            h_batch,
            use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
            xp=np,
        )
        moments_pred_phys = moments_pred_ratio * m_batch * h_batch * 9.8067

        grf_gt_phys = np.array(batch["grf"]) * m_batch * 9.8067
        cop_gt_phys = train_module.decode_cop_signal_to_length(
            np.array(batch["cop"]),
            np.array(batch["grf"]),
            h_batch,
            use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
            xp=np,
        )
        moments_gt_phys = np.array(batch["moments"]) * m_batch * h_batch * 9.8067

        valid_mask = train_module._extract_batched_frame_mask(
            np.array(batch["supervision_mask"]) if "supervision_mask" in batch else None,
            cop_pred_phys.shape[0],
            cop_pred_phys.shape[1],
        )
        hpo_metric_frame_mask = valid_mask
        if "window_start_idx" in batch and "trial_length" in batch:
            try:
                window_starts = np.asarray(batch["window_start_idx"]).reshape(-1)
                trial_lengths = np.asarray(batch["trial_length"]).reshape(-1)
                local_frame_idx = np.arange(cop_pred_phys.shape[1], dtype=np.int32)[None, :]
                absolute_frame_idx = window_starts[:, None] + local_frame_idx
                hpo_metric_frame_mask = absolute_frame_idx < trial_lengths[:, None]
            except Exception:
                hpo_metric_frame_mask = valid_mask

        if config["cop_mask"] and pred_np.shape[-1] >= 14:
            contact_prob = pred_np[..., 12:14]
            mask_r = (contact_prob[..., 0:1] > 0.5).astype(cop_pred_phys.dtype)
            mask_l = (contact_prob[..., 1:2] > 0.5).astype(cop_pred_phys.dtype)
            cop_pred_phys = np.concatenate(
                [cop_pred_phys[..., 0:2] * mask_r, cop_pred_phys[..., 2:4] * mask_l],
                axis=-1,
            )
            grf_pred_phys = np.concatenate(
                [grf_pred_phys[..., 0:3] * mask_r, grf_pred_phys[..., 3:6] * mask_l],
                axis=-1,
            )
            moments_pred_phys = np.concatenate(
                [moments_pred_phys[..., 0:1] * mask_r, moments_pred_phys[..., 1:2] * mask_l],
                axis=-1,
            )

        contact_bool = np.array(batch["contactBoolean"])
        stance_r = (contact_bool[..., 0] > 0.5) & valid_mask
        stance_l = (contact_bool[..., 1] > 0.5) & valid_mask
        hpo_metric_stance_l = (contact_bool[..., 1] > 0.5) & hpo_metric_frame_mask

        cop_err = cop_pred_phys - cop_gt_phys
        grf_err = grf_pred_phys - grf_gt_phys
        mom_err = moments_pred_phys - moments_gt_phys
        valid = valid_mask[..., None]
        val_cop_sumsq += np.sum((cop_err ** 2) * valid, axis=(0, 1))
        val_grf_sumsq += np.sum((grf_err ** 2) * valid, axis=(0, 1))
        val_mom_sumsq += np.sum((mom_err ** 2) * valid, axis=(0, 1))
        val_frames += int(np.sum(valid_mask))

        norm_mg = np.maximum(m_batch * 9.8067, 1e-8)
        grf_abs_pct_bw_err = np.abs((grf_pred_phys / norm_mg) - (grf_gt_phys / norm_mg)) * 100.0
        for axis, (right_idx, left_idx) in BILATERAL_GRF_AXIS_MAP.items():
            val_grf_mae_pct_bw_sum[axis] += float(
                np.sum(grf_abs_pct_bw_err[:, :, right_idx] * stance_r)
                + np.sum(grf_abs_pct_bw_err[:, :, left_idx] * stance_l)
            )
            val_grf_mae_pct_bw_count[axis] += int(np.sum(stance_r) + np.sum(stance_l))

        try:
            _eval_gt_kin = bool(config.get("use_gt_jacob_and_rot_for_eval", False))
            _rot_eval = batch["gt_rot_w_to_ga"] if (_eval_gt_kin and batch.get("gt_rot_w_to_ga") is not None) else batch["rot_w_to_ga"]
            _jacp_eval = batch["gt_jacp"] if (_eval_gt_kin and batch.get("gt_jacp") is not None) else batch["jacp"]
            _jacr_eval = batch["gt_jacr"] if (_eval_gt_kin and batch.get("gt_jacr") is not None) else batch["jacr"]
            full_mom_pred = np.array(
                train_module.compute_full_external_moments(
                    jnp.array(cop_pred_phys),
                    jnp.array(grf_pred_phys),
                    jnp.array(moments_pred_phys),
                    batch["ankle_heights"],
                    _rot_eval,
                )
            )
            tau_pred = np.array(
                train_module.compute_tau_grf_from_predictions(
                    jnp.array(grf_pred_phys),
                    jnp.array(full_mom_pred),
                    _jacp_eval,
                    _jacr_eval,
                )
            )
            tau_gt = np.array(batch["qfrc_grf_contribution"])
            tau_err = tau_pred - tau_gt
            _, _, dof_count = tau_err.shape

            for name, dof_index in KEY_TAU_DOFS.items():
                val_tau_sumsq[name] += float(np.sum((tau_err[:, :, dof_index] ** 2) * valid_mask))
            val_tau_sumsq_all += float(np.sum((tau_err ** 2) * valid))
            valid_frame_count = int(np.sum(valid_mask))
            val_tau_count += valid_frame_count * dof_count
            val_tau_frames += valid_frame_count

            norm_mgh = np.maximum(m_batch * 9.8067 * h_batch, 1e-8)
            tau_abs_pct_err = np.abs((tau_pred / norm_mgh) - (tau_gt / norm_mgh)) * 100.0
            for name, dof_index in KEY_TAU_DOFS.items():
                stance_mask = stance_r if name.startswith("R ") else stance_l
                val_tau_mae_pct_sum[name] += float(np.sum(tau_abs_pct_err[:, :, dof_index] * stance_mask))
                val_tau_mae_pct_count[name] += int(np.sum(stance_mask))
            for name, dof_index in STANCE_MAE_TAU_DOFS.items():
                stance_mask = stance_r if name.startswith("R ") else stance_l
                val_stance_tau_mae_pct_sum[name] += float(
                    np.sum(tau_abs_pct_err[:, :, dof_index] * stance_mask)
                )
                val_stance_tau_mae_pct_count[name] += int(np.sum(stance_mask))
            for name, dof_index in zip(selected_left_dof_names, selected_left_dof_indices):
                selected_left_mae_pct_sum[name] += float(
                    np.sum(tau_abs_pct_err[:, :, dof_index] * hpo_metric_stance_l)
                )
                selected_left_mae_pct_count[name] += int(np.sum(hpo_metric_stance_l))

            knee_to_cop_vectors = batch.get("knee_to_cop_vectors")
            if knee_to_cop_vectors is not None:
                saw_left_kam_vectors = True
                knee_to_cop = np.array(knee_to_cop_vectors)
                if knee_to_cop.shape[-1] < 6:
                    raise ValueError("knee_to_cop_vectors must have at least 6 columns for KAM computation.")
                z_vec_l = knee_to_cop[:, :, 5]
                y_vec_l = knee_to_cop[:, :, 4]
                kam_pred = z_vec_l * grf_pred_phys[:, :, 4] - y_vec_l * grf_pred_phys[:, :, 5]
                kam_gt = z_vec_l * grf_gt_phys[:, :, 4] - y_vec_l * grf_gt_phys[:, :, 5]
                kam_abs_pct_err = np.abs((kam_pred / norm_mgh[:, :, 0]) - (kam_gt / norm_mgh[:, :, 0])) * 100.0
                selected_left_kam_sum += float(np.sum(kam_abs_pct_err * hpo_metric_stance_l))
                selected_left_kam_count += int(np.sum(hpo_metric_stance_l))
        except Exception:
            if require_left_kam:
                raise
            pass

    if val_steps == 0:
        raise RuntimeError("Evaluation loader yielded zero batches.")

    averaged_losses = {key: value / val_steps for key, value in val_metrics.items()}
    cop_overall_rmse = float("nan")
    grf_overall_rmse = float("nan")
    moments_overall_rmse = float("nan")
    if val_frames > 0:
        frames_f = float(val_frames)
        cop_overall_rmse = float(np.sqrt(np.mean(val_cop_sumsq) / frames_f))
        grf_overall_rmse = float(np.sqrt(np.mean(val_grf_sumsq) / frames_f))
        moments_overall_rmse = float(np.sqrt(np.mean(val_mom_sumsq) / frames_f))

    tau_rmse_selected_dofs: Dict[str, float] = {}
    torque_overall_rmse = float("nan")
    if val_tau_frames > 0:
        for name in KEY_TAU_DOFS:
            tau_rmse_selected_dofs[name] = float(
                np.sqrt(val_tau_sumsq[name] / max(val_tau_frames, 1))
            )
        torque_overall_rmse = float(np.sqrt(val_tau_sumsq_all / max(val_tau_count, 1)))

    torque_mae_percent_bilateral_stance: Dict[str, float] = {}
    for joint, (dof_r, dof_l) in STANCE_MAE_BILATERAL_TAU_MAP.items():
        total = float(val_stance_tau_mae_pct_sum.get(dof_r, 0.0) + val_stance_tau_mae_pct_sum.get(dof_l, 0.0))
        count = int(val_stance_tau_mae_pct_count.get(dof_r, 0) + val_stance_tau_mae_pct_count.get(dof_l, 0))
        torque_mae_percent_bilateral_stance[joint] = float(total / count) if count > 0 else float("nan")

    grf_mae_percent_bw_bilateral_stance: Dict[str, float] = {}
    for axis in BILATERAL_GRF_AXIS_MAP.keys():
        count = int(val_grf_mae_pct_bw_count.get(axis, 0))
        grf_mae_percent_bw_bilateral_stance[axis] = (
            float(val_grf_mae_pct_bw_sum[axis] / count) if count > 0 else float("nan")
        )

    selected_left_stance_mae: Dict[str, float] = {}
    for name in selected_left_dof_names:
        count = int(selected_left_mae_pct_count.get(name, 0))
        selected_left_stance_mae[name] = (
            float(selected_left_mae_pct_sum[name] / count) if count > 0 else float("nan")
        )
    if require_left_kam and not saw_left_kam_vectors:
        raise ValueError(
            "KAM is required for metric-only LOSO HPO, but no knee_to_cop_vectors were found in evaluation batches."
        )
    if selected_left_kam_count > 0:
        selected_left_stance_mae[selected_left_kam_name] = float(selected_left_kam_sum / selected_left_kam_count)
    elif require_left_kam:
        raise ValueError("KAM is required for metric-only LOSO HPO, but no valid left-stance KAM frames were found.")

    selected_left_values = [
        float(value)
        for value in selected_left_stance_mae.values()
        if value is not None and np.isfinite(float(value))
    ]
    selected_left_mean = float(np.mean(selected_left_values)) if selected_left_values else float("nan")

    return {
        "losses": averaged_losses,
        "metrics": {
            "total_loss": float(averaged_losses["total_loss"]),
            "cop_rmse_fullset_overall_m": cop_overall_rmse,
            "grf_rmse_fullset_overall_N": grf_overall_rmse,
            "moments_rmse_fullset_overall_Nm": moments_overall_rmse,
            "torque_rmse_fullset_overall_Nm": torque_overall_rmse,
            "torque_rmse_selected_dofs_Nm": tau_rmse_selected_dofs,
            "torque_mae_percent_bilateral_stance": torque_mae_percent_bilateral_stance,
            "selected_left_stance_moment_mae_percent_bwh": selected_left_stance_mae,
            "selected_left_stance_moment_mae_percent_bwh_mean": selected_left_mean,
            "grf_mae_percent_bw_bilateral_stance": grf_mae_percent_bw_bilateral_stance,
        },
    }


# =============================================================================
# Inner-validation epoch selection (LOSO_HPO-style, epochs only)
# =============================================================================


def _finite_float(value: Any, default: float = float("inf")) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    return value_f if np.isfinite(value_f) else default


def _format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.2f}h"


def _system_memory_used_fraction() -> Optional[float]:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    values_kb: Dict[str, float] = {}
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                values_kb[parts[0].rstrip(":")] = float(parts[1])
    except OSError:
        return None
    total = values_kb.get("MemTotal")
    available = values_kb.get("MemAvailable")
    if not total or available is None or total <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - available / total))


def _maybe_clear_jax_caches(
    *,
    completed_count: int,
    clear_every: int,
    memory_fraction_threshold: float,
) -> bool:
    used_fraction = _system_memory_used_fraction()
    due_by_interval = clear_every > 0 and completed_count > 0 and completed_count % int(clear_every) == 0
    due_by_memory = used_fraction is not None and used_fraction >= float(memory_fraction_threshold)
    if not due_by_interval and not due_by_memory:
        return False
    try:
        jax.clear_caches()
    except Exception:
        return False
    reason_parts = []
    if due_by_interval:
        reason_parts.append(f"interval={clear_every}")
    if due_by_memory and used_fraction is not None:
        reason_parts.append(f"memory={used_fraction * 100:.1f}%")
    print(f"[cache] cleared JAX caches ({', '.join(reason_parts)})", flush=True)
    return True


def _mean_std_score(values: Sequence[float], std_weight: float) -> Tuple[float, float, float, int]:
    finite_values = [float(value) for value in values if np.isfinite(float(value))]
    if not finite_values:
        return float("inf"), float("inf"), float("inf"), 0
    mean_value = float(np.mean(finite_values))
    std_value = float(np.std(finite_values, ddof=0))
    return mean_value, std_value, float(mean_value + float(std_weight) * std_value), len(finite_values)


def _resolve_inner_eval_epochs(
    inner_eval_epochs: Optional[Sequence[int]],
    max_epochs: int,
) -> List[int]:
    if inner_eval_epochs:
        epochs = sorted({int(epoch) for epoch in inner_eval_epochs})
    else:
        epochs = list(range(1, int(max_epochs) + 1))
    epochs = [epoch for epoch in epochs if 1 <= epoch <= int(max_epochs)]
    if not epochs:
        raise ValueError(
            f"No valid inner evaluation epochs in 1..{max_epochs} (requested {list(inner_eval_epochs or [])})."
        )
    return epochs


def _build_inner_epoch_folds(
    fold: Mapping[str, Any],
    subject_to_trials: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """For one outer fold, build inner LOSO folds over the held-in subjects.

    Each held-in subject becomes the inner validation subject once; the model is
    trained on the remaining held-in subjects. The outer held-out (test) subject is
    never present in either split.
    """
    held_in_subjects = list(fold["train_subjects"])
    inner_folds: List[Dict[str, Any]] = []
    for inner_val_subject in held_in_subjects:
        inner_train_subjects = [s for s in held_in_subjects if s != inner_val_subject]
        inner_folds.append(
            {
                "inner_val_subject": inner_val_subject,
                "inner_train_subjects": inner_train_subjects,
                "inner_train_trials": [
                    trial for subject in inner_train_subjects for trial in subject_to_trials[subject]
                ],
                "inner_eval_trials": list(subject_to_trials[inner_val_subject]),
            }
        )
    return inner_folds


def _run_inner_epoch_validation_fold(
    *,
    test_subject: str,
    inner_val_subject: str,
    inner_train_subjects: Sequence[str],
    inner_train_trials: Sequence[Mapping[str, Any]],
    inner_eval_trials: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    max_epochs: int,
    eval_epochs: Sequence[int],
    objective_name: str,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    adapter_hidden_dim: int,
    adapter_dropout_rate: float,
    seed: int,
    split_dir: Path,
) -> Dict[str, Any]:
    """Train one inner fold once up to ``max_epochs``, scoring at each eval epoch.

    Metric-only (no infer-style plotting, no model pickle) so the inner sweep stays
    cheap. Safe to run inside a worker thread: every call builds its own loaders,
    model, and train state.
    """
    inner_start = time.time()
    split_dir.mkdir(parents=True, exist_ok=True)
    eval_epoch_set = {int(epoch) for epoch in eval_epochs}
    log_path = split_dir / "inner_training_log.txt"
    with log_path.open("w", encoding="utf-8") as log_handle:
        def _log(message: str) -> None:
            print(message, file=log_handle, flush=True)

        if inner_val_subject in set(inner_train_subjects):
            raise ValueError(
                f"Leakage: inner validation subject {inner_val_subject} appears in inner training subjects."
            )
        if test_subject in set(inner_train_subjects) or test_subject == inner_val_subject:
            raise ValueError(
                f"Leakage: outer test subject {test_subject} present in inner split for {inner_val_subject}."
            )

        _save_json(
            split_dir / "inner_split.json",
            {
                "outer_test_subject": test_subject,
                "inner_val_subject": inner_val_subject,
                "inner_train_subjects": list(inner_train_subjects),
                "max_epochs": int(max_epochs),
                "eval_epochs": list(eval_epochs),
                "objective_name": objective_name,
            },
        )

        checkpoint_input_dim = int(np.asarray(checkpoint["params"]["Dense_0"]["kernel"]).shape[0])
        checkpoint_static_dim = int(np.asarray(checkpoint["params"]["Dense_1"]["kernel"]).shape[0])
        fold_config = _resolve_fold_input_config(
            list(inner_train_trials) + list(inner_eval_trials),
            config,
            expected_input_dim=checkpoint_input_dim,
            expected_static_dim=checkpoint_static_dim,
        )
        _log(
            f"[{test_subject}/inner_{inner_val_subject}] layout={fold_config['resolved_input_layout']} "
            f"input_dim={fold_config['input_dim']} static_dim={fold_config['static_dim']} "
            f"max_epochs={max_epochs} eval_epochs={list(eval_epochs)} lr={learning_rate:g}"
        )

        train_loader = _safe_trial_loader(inner_train_trials, fold_config, batch_size=batch_size, shuffle=True)
        eval_loader = _safe_trial_loader(inner_eval_trials, fold_config, batch_size=batch_size, shuffle=False)
        sample_batch = next(iter(train_loader))
        input_dim = int(sample_batch["input"].shape[-1])
        static_dim = int(sample_batch["static_context"].shape[-1])
        if checkpoint_input_dim != input_dim:
            raise ValueError(f"Checkpoint input_dim={checkpoint_input_dim}, loader input_dim={input_dim}.")
        if checkpoint_static_dim != static_dim:
            raise ValueError(f"Checkpoint static_dim={checkpoint_static_dim}, loader static_dim={static_dim}.")

        model = loso_adapters.build_loso_model(
            fold_config,
            checkpoint["params"],
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout_rate=adapter_dropout_rate,
        )
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        state = loso_adapters.create_loso_train_state(
            init_rng,
            model,
            checkpoint["params"],
            input_shape=(1, int(fold_config["window_size"]), input_dim),
            static_shape=(1, static_dim),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        normalizers = checkpoint["normalizers"]
        dof_weights = _build_dof_weights(fold_config)
        loss_weights = _build_loss_weights(fold_config)
        train_step = train_module.make_train_step(
            normalizers,
            bool(fold_config["use_contact_weighting"]),
            bool(fold_config["mag_on_off"]),
            bool(fold_config["contact_on_off"]),
            False,
            float(fold_config["contact_weight_multiplier"]),
            float(fold_config["mag_weight"]),
            int(max_epochs),
            dof_weights,
            cop_mask=bool(fold_config["cop_mask"]),
            use_grf_norm_cop=bool(fold_config.get("use_grf_norm_cop", False)),
            use_gt_jacob_and_rot=bool(fold_config.get("use_gt_jacob_and_rot_for_training", False)),
        )
        eval_step = train_module.make_eval_step(
            normalizers,
            bool(fold_config["use_contact_weighting"]),
            bool(fold_config["mag_on_off"]),
            bool(fold_config["contact_on_off"]),
            False,
            float(fold_config["contact_weight_multiplier"]),
            float(fold_config["mag_weight"]),
            int(max_epochs),
            dof_weights,
            cop_mask=bool(fold_config["cop_mask"]),
            use_grf_norm_cop=bool(fold_config.get("use_grf_norm_cop", False)),
        )

        objectives_by_epoch: Dict[int, float] = {}
        epoch_records: List[Dict[str, Any]] = []
        for epoch in range(1, int(max_epochs) + 1):
            epoch_start = time.time()
            state, train_losses, rng = _run_train_epoch(
                state,
                train_loader,
                train_step=train_step,
                normalizers=normalizers,
                loss_weights=loss_weights,
                rng=rng,
                epoch=epoch,
            )
            _log(
                f"[{test_subject}/inner_{inner_val_subject}] epoch {epoch}/{max_epochs} "
                f"train_loss={train_losses['total_loss']:.4f} time={_format_duration(time.time() - epoch_start)}"
            )
            if int(epoch) in eval_epoch_set:
                # Match the final held-out evaluation (require_left_kam=False): KAM is
                # still folded into the objective when present, but a trial lacking KAM
                # vectors degrades gracefully instead of failing the whole inner fold.
                eval_payload = _evaluate_loader(
                    state,
                    eval_loader,
                    eval_step=eval_step,
                    normalizers=normalizers,
                    loss_weights=loss_weights,
                    config=fold_config,
                    epoch=int(epoch),
                    require_left_kam=False,
                )
                objective = _finite_float(eval_payload["metrics"].get(objective_name))
                objectives_by_epoch[int(epoch)] = float(objective)
                epoch_records.append(
                    {
                        "epoch": int(epoch),
                        "objective": float(objective),
                        "metrics": eval_payload["metrics"],
                    }
                )
                _log(
                    f"[{test_subject}/inner_{inner_val_subject}] eval epoch {epoch}/{max_epochs} "
                    f"{objective_name}={objective:.6f}"
                )

        missing = sorted(eval_epoch_set - set(objectives_by_epoch.keys()))
        if missing:
            raise RuntimeError(
                f"Inner fold {test_subject}/inner_{inner_val_subject} missing epoch evaluations {missing}."
            )

        inner_payload = {
            "outer_test_subject": test_subject,
            "inner_val_subject": inner_val_subject,
            "inner_train_subjects": list(inner_train_subjects),
            "objective_name": objective_name,
            "objectives_by_epoch": {int(k): float(v) for k, v in objectives_by_epoch.items()},
            "epoch_records": epoch_records,
            "duration_s": float(time.time() - inner_start),
        }
        _save_json(split_dir / "inner_metrics.json", inner_payload)

    return {
        "test_subject": str(test_subject),
        "inner_val_subject": str(inner_val_subject),
        "objectives_by_epoch": {int(k): float(v) for k, v in objectives_by_epoch.items()},
        "duration_s": float(time.time() - inner_start),
        "split_dir": str(split_dir),
    }


def _select_epochs_via_inner_validation(
    folds: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    subject_to_trials: Mapping[str, Sequence[Mapping[str, Any]]],
    max_epochs: int,
    eval_epochs: Sequence[int],
    objective_name: str,
    std_weight: float,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    adapter_hidden_dim: int,
    adapter_dropout_rate: float,
    base_seed: int,
    max_parallel_splits: int,
    cache_clear_every: int,
    cache_clear_memory_fraction: float,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Run the inner LOSO epoch sweep for every outer fold and pick the best epoch.

    Returns ``(selected_epoch_by_subject, summary)``. All inner trainings are
    dispatched to a single thread pool so outer x inner folds run concurrently.
    """
    inner_root = output_root / "inner_epoch_selection"
    inner_root.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for outer_index, fold in enumerate(folds):
        test_subject = str(fold["held_out_subject"])
        inner_folds = _build_inner_epoch_folds(fold, subject_to_trials)
        if len(inner_folds) < 2:
            print(
                f"[{test_subject}] only {len(inner_folds)} held-in subject(s); "
                f"skipping inner epoch selection for this fold.",
                flush=True,
            )
        for inner_index, inner_fold in enumerate(inner_folds):
            inner_val_subject = str(inner_fold["inner_val_subject"])
            split_dir = inner_root / test_subject / f"inner_{inner_val_subject}"
            jobs.append(
                {
                    "outer_index": int(outer_index),
                    "inner_index": int(inner_index),
                    "test_subject": test_subject,
                    "inner_val_subject": inner_val_subject,
                    "inner_train_subjects": list(inner_fold["inner_train_subjects"]),
                    "inner_train_trials": list(inner_fold["inner_train_trials"]),
                    "inner_eval_trials": list(inner_fold["inner_eval_trials"]),
                    "split_dir": split_dir,
                    "seed": int(base_seed + outer_index * 1000 + inner_index),
                }
            )

    print(
        f"\nInner epoch selection: {len(jobs)} inner trainings "
        f"({len(folds)} outer folds x up to {len(folds[0]['train_subjects']) if folds else 0} inner folds), "
        f"max_epochs={max_epochs}, eval_epochs={list(eval_epochs)}, "
        f"objective={objective_name}, score=mean+{std_weight:g}*std, "
        f"max_parallel_splits={max_parallel_splits}",
        flush=True,
    )

    results_by_subject: Dict[str, List[Dict[str, Any]]] = {
        str(fold["held_out_subject"]): [] for fold in folds
    }
    errors_by_subject: Dict[str, List[str]] = {
        str(fold["held_out_subject"]): [] for fold in folds
    }

    inner_phase_start = time.time()
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max(1, int(max_parallel_splits))) as executor:
        future_to_job = {
            executor.submit(
                _run_inner_epoch_validation_fold,
                test_subject=job["test_subject"],
                inner_val_subject=job["inner_val_subject"],
                inner_train_subjects=job["inner_train_subjects"],
                inner_train_trials=job["inner_train_trials"],
                inner_eval_trials=job["inner_eval_trials"],
                checkpoint=checkpoint,
                config=config,
                max_epochs=int(max_epochs),
                eval_epochs=list(eval_epochs),
                objective_name=objective_name,
                learning_rate=float(learning_rate),
                batch_size=int(batch_size),
                weight_decay=float(weight_decay),
                adapter_hidden_dim=int(adapter_hidden_dim),
                adapter_dropout_rate=float(adapter_dropout_rate),
                seed=int(job["seed"]),
                split_dir=job["split_dir"],
            ): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            test_subject = str(job["test_subject"])
            try:
                result = future.result()
            except Exception as exc:
                message = f"{test_subject}/inner_{job['inner_val_subject']}: {exc}"
                print(f"[{test_subject}] inner fold failed: {exc}", flush=True)
                errors_by_subject[test_subject].append(message)
                continue
            results_by_subject[test_subject].append(result)
            best_for_log = min(result["objectives_by_epoch"].items(), key=lambda kv: kv[1], default=(None, None))
            print(
                f"[{test_subject}] inner val={result['inner_val_subject']} done "
                f"(best epoch={best_for_log[0]} {objective_name}={best_for_log[1] if best_for_log[1] is not None else float('nan'):.4f}, "
                f"{_format_duration(float(result['duration_s']))})",
                flush=True,
            )
            completed_count += 1
            gc.collect()
            _maybe_clear_jax_caches(
                completed_count=completed_count,
                clear_every=int(cache_clear_every),
                memory_fraction_threshold=float(cache_clear_memory_fraction),
            )

    selected_epoch_by_subject: Dict[str, int] = {}
    per_fold_summary: List[Dict[str, Any]] = []
    eval_epochs = [int(epoch) for epoch in eval_epochs]
    for fold in folds:
        test_subject = str(fold["held_out_subject"])
        inner_results = results_by_subject[test_subject]
        expected_inner = len(fold["train_subjects"])
        per_epoch_scores: Dict[int, List[float]] = {int(epoch): [] for epoch in eval_epochs}
        for result in inner_results:
            for epoch in eval_epochs:
                value = result["objectives_by_epoch"].get(int(epoch))
                if value is not None:
                    per_epoch_scores[int(epoch)].append(float(value))

        epoch_stats: Dict[int, Dict[str, float]] = {}
        best_epoch: Optional[int] = None
        best_score = float("inf")
        for epoch in eval_epochs:
            mean_value, std_value, score, n_valid = _mean_std_score(per_epoch_scores[int(epoch)], std_weight)
            epoch_stats[int(epoch)] = {
                "mean": mean_value,
                "std": std_value,
                "selection_score": score,
                "num_inner_subjects": int(n_valid),
            }
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_epoch = int(epoch)

        if best_epoch is None:
            best_epoch = int(max(eval_epochs))
            print(
                f"[{test_subject}] WARNING: no finite inner scores "
                f"({len(inner_results)}/{expected_inner} inner folds, "
                f"{len(errors_by_subject[test_subject])} errors); "
                f"falling back to epochs={best_epoch}.",
                flush=True,
            )
        else:
            print(
                f"[{test_subject}] selected epoch={best_epoch} "
                f"(mean {objective_name}={epoch_stats[best_epoch]['mean']:.4f}, "
                f"std={epoch_stats[best_epoch]['std']:.4f}, score={best_score:.4f}) "
                f"from {len(inner_results)}/{expected_inner} inner folds",
                flush=True,
            )

        selected_epoch_by_subject[test_subject] = int(best_epoch)
        per_fold_summary.append(
            {
                "held_out_subject": test_subject,
                "held_in_subjects": list(fold["train_subjects"]),
                "completed_inner_folds": len(inner_results),
                "expected_inner_folds": int(expected_inner),
                "errors": list(errors_by_subject[test_subject]),
                "selected_epoch": int(best_epoch),
                "selection_score": float(best_score) if np.isfinite(best_score) else None,
                "epoch_stats": {str(epoch): epoch_stats[int(epoch)] for epoch in eval_epochs},
            }
        )

    summary = {
        "objective_name": objective_name,
        "selection_score_formula": f"mean + {std_weight} * std",
        "selection_std_weight": float(std_weight),
        "max_epochs": int(max_epochs),
        "eval_epochs": list(eval_epochs),
        "max_parallel_splits": int(max_parallel_splits),
        "inner_phase_duration_s": float(time.time() - inner_phase_start),
        "selected_epoch_by_subject": selected_epoch_by_subject,
        "per_fold": per_fold_summary,
    }
    _save_json(inner_root / "inner_epoch_selection_summary.json", summary)
    print(
        f"\nInner epoch selection complete in {_format_duration(summary['inner_phase_duration_s'])}; "
        f"selected epochs: {selected_epoch_by_subject}",
        flush=True,
    )
    return selected_epoch_by_subject, summary


def _run_fold(
    fold: Mapping[str, Any],
    *,
    fold_dir: Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    adapter_hidden_dim: int,
    adapter_dropout_rate: float,
    seed: int,
    epoch_selection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    split_payload = {
        "held_out_subject": fold["held_out_subject"],
        "inner_val_subject": fold.get("inner_val_subject"),
        "train_subjects": list(fold["train_subjects"]),
        "held_out_trials": list(fold["held_out_trials"]),
        "inner_val_trials": list(fold.get("inner_val_trials") or []),
        "train_trials": list(fold["train_trials"]),
        "training_split_mode": "all_non_held_out_no_inner_val",
    }
    _save_json(fold_dir / "split.json", split_payload)

    checkpoint_input_dim = int(np.asarray(checkpoint["params"]["Dense_0"]["kernel"]).shape[0])
    checkpoint_static_dim = int(np.asarray(checkpoint["params"]["Dense_1"]["kernel"]).shape[0])
    fold_config = _resolve_fold_input_config(
        list(fold["train_trials"]) + list(fold["held_out_trials"]),
        config,
        expected_input_dim=checkpoint_input_dim,
        expected_static_dim=checkpoint_static_dim,
    )
    print(
        f"[{fold['held_out_subject']}] resolved input layout="
        f"{fold_config['resolved_input_layout']} from {fold_config['resolved_input_sample_trial']} "
        f"(deviation_learning={fold_config['deviation_learning']}, "
        f"include_pelvis_euler={fold_config['include_pelvis_euler']}, "
        f"include_ankle_heights={fold_config.get('include_ankle_heights', True)}, "
        f"include_jacobian_input={fold_config.get('include_jacobian_input', True)}, "
        f"include_auxiliary_denoising_inputs={fold_config.get('include_auxiliary_denoising_inputs', True)}, "
        f"input_dim={fold_config['input_dim']})",
        flush=True,
    )

    train_loader = _safe_trial_loader(fold["train_trials"], fold_config, batch_size=batch_size, shuffle=True)
    test_loader = _safe_trial_loader(fold["held_out_trials"], fold_config, batch_size=batch_size, shuffle=False)

    sample_batch = next(iter(train_loader))
    input_dim = int(sample_batch["input"].shape[-1])
    static_dim = int(sample_batch["static_context"].shape[-1])
    if fold_config.get("input_dim") is not None and int(fold_config["input_dim"]) != input_dim:
        raise ValueError(
            f"Input dimension mismatch for fold {fold['held_out_subject']}: "
            f"checkpoint expects {fold_config['input_dim']} but loader produced {input_dim}."
        )
    if checkpoint_input_dim != input_dim:
        raise ValueError(
            f"Checkpoint Dense_0 expects input_dim={checkpoint_input_dim}, "
            f"but LOSO loader produced input_dim={input_dim}."
        )
    if checkpoint_static_dim != static_dim:
        raise ValueError(
            f"Checkpoint Dense_1 expects static_dim={checkpoint_static_dim}, "
            f"but LOSO loader produced static_dim={static_dim}."
        )

    model = loso_adapters.build_loso_model(
        fold_config,
        checkpoint["params"],
        adapter_hidden_dim=adapter_hidden_dim,
        adapter_dropout_rate=adapter_dropout_rate,
    )
    rng = jax.random.PRNGKey(int(seed))
    rng, init_rng = jax.random.split(rng)
    state = loso_adapters.create_loso_train_state(
        init_rng,
        model,
        checkpoint["params"],
        input_shape=(1, int(fold_config["window_size"]), input_dim),
        static_shape=(1, static_dim),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    normalizers = checkpoint["normalizers"]
    dof_weights = _build_dof_weights(fold_config)
    loss_weights = _build_loss_weights(fold_config)
    train_step = train_module.make_train_step(
        normalizers,
        bool(fold_config["use_contact_weighting"]),
        bool(fold_config["mag_on_off"]),
        bool(fold_config["contact_on_off"]),
        False,
        float(fold_config["contact_weight_multiplier"]),
        float(fold_config["mag_weight"]),
        int(epochs),
        dof_weights,
        cop_mask=bool(fold_config["cop_mask"]),
        use_grf_norm_cop=bool(fold_config.get("use_grf_norm_cop", False)),
        use_gt_jacob_and_rot=bool(fold_config.get("use_gt_jacob_and_rot_for_training", False)),
    )
    eval_step = train_module.make_eval_step(
        normalizers,
        bool(fold_config["use_contact_weighting"]),
        bool(fold_config["mag_on_off"]),
        bool(fold_config["contact_on_off"]),
        False,
        float(fold_config["contact_weight_multiplier"]),
        float(fold_config["mag_weight"]),
        int(epochs),
        dof_weights,
        cop_mask=bool(fold_config["cop_mask"]),
        use_grf_norm_cop=bool(fold_config.get("use_grf_norm_cop", False)),
    )

    initial_checkpoint_params = loso_adapters.extract_checkpoint_params(state.params)
    initial_adapter_params = loso_adapters.extract_adapter_params(state.params)

    history: List[Dict[str, Any]] = []
    history.append(
        {
            "epoch": 0,
            "epoch_time_s": 0.0,
            "train_losses": None,
            "inner_val": None,
        }
    )
    print(
        f"[{fold['held_out_subject']}] training on {len(fold['train_subjects'])} subjects "
        f"({len(fold['train_trials'])} trials), no inner validation subject; "
        f"using final epoch checkpoint after {epochs} training epoch(s)",
        flush=True,
    )
    if epochs <= 0:
        print(
            f"[{fold['held_out_subject']}] epochs={epochs}; keeping source checkpoint without fine-tuning",
            flush=True,
        )
    else:
        print(
            f"[{fold['held_out_subject']}] epoch 0/{epochs} train_loss=NA (source checkpoint)",
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        state, train_losses, rng = _run_train_epoch(
            state,
            train_loader,
            train_step=train_step,
            normalizers=normalizers,
            loss_weights=loss_weights,
            rng=rng,
            epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "epoch_time_s": float(time.time() - epoch_start),
                "train_losses": train_losses,
                "inner_val": None,
            }
        )
        print(
            f"[{fold['held_out_subject']}] epoch {epoch}/{epochs} "
            f"train_loss={train_losses['total_loss']:.4f}",
            flush=True,
        )

    selected_epoch = int(epochs)
    best_selection_name = "inner_validation_epoch" if epoch_selection else "final_epoch"
    best_selection_metric = float(
        history[-1]["train_losses"]["total_loss"]
        if epochs > 0 and history[-1].get("train_losses") is not None
        else float("nan")
    )
    final_state = state
    held_out_metrics = _evaluate_loader(
        final_state,
        test_loader,
        eval_step=eval_step,
        normalizers=normalizers,
        loss_weights=loss_weights,
        config=fold_config,
        epoch=selected_epoch,
    )
    held_out_infer_style = _run_infer_style_evaluation(
        fold,
        fold_dir=fold_dir,
        model=model,
        params=final_state.params,
        normalizers=normalizers,
        config=fold_config,
    )
    trunk_sway_effect_eval = None
    if bool(fold_config.get("evaluate_on_ts", False)):
        data_dir = Path(str(fold.get("dataset_root", config.get("data_dir", PROJECT_ROOT / "OpenCapWalkingTrunkSwaySubjects"))))
        trunk_sway_effect_eval = _run_trunk_sway_effect_evaluation(
            fold,
            fold_dir=fold_dir,
            data_dir=data_dir,
            model=model,
            params=final_state.params,
            normalizers=normalizers,
            config=fold_config,
        )

    final_checkpoint_params = loso_adapters.extract_checkpoint_params(final_state.params)
    final_adapter_params = loso_adapters.extract_adapter_params(final_state.params)
    checkpoint_params_changed = not loso_adapters.checkpoint_params_unchanged(
        initial_checkpoint_params,
        final_checkpoint_params,
    )
    adapter_params_present = bool(jax.tree_util.tree_leaves(final_adapter_params))

    fold_checkpoint_payload = {
        "params": final_state.params,
        "source_checkpoint": config["source_checkpoint"],
        "source_hyperparameters_path": config["source_hyperparameters_path"],
        "normalizers": normalizers,
        "held_out_subject": fold["held_out_subject"],
        "inner_val_subject": fold.get("inner_val_subject"),
        "train_subjects": list(fold["train_subjects"]),
        "training_split_mode": "all_non_held_out_no_inner_val",
        "best_epoch": selected_epoch,
        "checkpoint_selection_mode": "final_epoch",
        "best_selection_metric_name": best_selection_name,
        "best_selection_metric_value": best_selection_metric,
        "best_inner_val_metrics": None,
        "inner_epoch_selection": dict(epoch_selection) if epoch_selection else None,
        "held_out_metrics": held_out_metrics,
        "resolved_input_config": {
            "deviation_learning": bool(fold_config["deviation_learning"]),
            "include_pelvis_euler": bool(fold_config["include_pelvis_euler"]),
            "include_ankle_heights": bool(fold_config.get("include_ankle_heights", True)),
            "include_jacobian_input": bool(fold_config.get("include_jacobian_input", True)),
            "UseGRFNormCOP": bool(fold_config.get("use_grf_norm_cop", False)),
            "input_dim": int(fold_config["input_dim"]),
            "static_dim": int(fold_config["static_dim"]),
            "layout_name": fold_config.get("resolved_input_layout"),
            "sample_trial": fold_config.get("resolved_input_sample_trial"),
            "input_feature_blocks": fold_config.get("resolved_input_feature_blocks", []),
            "input_layout_diagnostics": fold_config.get("resolved_input_layout_diagnostics", {}),
        },
        "fine_tuning_epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    with (fold_dir / "best_model.pkl").open("wb") as handle:
        pickle.dump(fold_checkpoint_payload, handle)

    metrics_payload = {
        "held_out_subject": fold["held_out_subject"],
        "inner_val_subject": fold.get("inner_val_subject"),
        "train_subjects": list(fold["train_subjects"]),
        "training_split_mode": "all_non_held_out_no_inner_val",
        "best_epoch": selected_epoch,
        "checkpoint_selection_mode": "final_epoch",
        "best_selection_metric_name": best_selection_name,
        "best_selection_metric_value": best_selection_metric,
        "best_inner_val_metrics": None,
        "inner_epoch_selection": dict(epoch_selection) if epoch_selection else None,
        "held_out_metrics": held_out_metrics,
        "resolved_input_config": {
            "deviation_learning": bool(fold_config["deviation_learning"]),
            "include_pelvis_euler": bool(fold_config["include_pelvis_euler"]),
            "input_dim": int(fold_config["input_dim"]),
            "static_dim": int(fold_config["static_dim"]),
            "layout_name": fold_config.get("resolved_input_layout"),
            "sample_trial": fold_config.get("resolved_input_sample_trial"),
            "input_feature_blocks": fold_config.get("resolved_input_feature_blocks", []),
            "input_layout_diagnostics": fold_config.get("resolved_input_layout_diagnostics", {}),
        },
        "verification": {
            "checkpoint_params_changed": checkpoint_params_changed,
            "adapter_params_present": adapter_params_present,
        },
        "held_out_infer_style": held_out_infer_style,
        "trunk_sway_effect_eval": trunk_sway_effect_eval,
        "history": history,
    }
    _save_json(fold_dir / "metrics.json", metrics_payload)
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LOSO fine-tuning from a pretrained Transformer checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to source best_model.pkl")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Defaults to Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Optional LOSO output root")
    parser.add_argument("--epochs", type=int, default=None, help="Fine-tuning epochs per fold")
    parser.add_argument("--learning_rate", type=float, default=None, help="Fine-tuning learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Fine-tuning batch size")
    parser.add_argument("--weight_decay", type=float, default=None, help="Fine-tuning optimizer weight decay")
    parser.add_argument("--cop_weight", type=float, default=None, help="Override COP loss weight")
    parser.add_argument("--grf_weight", type=float, default=None, help="Override GRF loss weight")
    parser.add_argument("--moments_weight", type=float, default=None, help="Override free-moment loss weight")
    parser.add_argument("--contact_weight", type=float, default=None, help="Override contact loss weight")
    parser.add_argument("--torque_weight", type=float, default=None, help="Override torque loss weight")
    parser.add_argument("--qfrc_inverse_weight", type=float, default=None, help="Override qfrc_inverse loss weight")
    parser.add_argument("--qfrc_inverse_input_reg_weight", type=float, default=None, help="Override qfrc_inverse input regularization weight")
    parser.add_argument("--rotation_weight", type=float, default=None, help="Override rotation loss weight")
    parser.add_argument("--rotation_input_reg_weight", type=float, default=None, help="Override rotation input regularization weight")
    parser.add_argument("--rotation_residual_max_deg", type=float, default=None, help="Override residual rotation clamp in degrees")
    parser.add_argument("--jacobian_weight", type=float, default=None, help="Override Jacobian loss weight")
    parser.add_argument("--jacobian_input_reg_weight", type=float, default=None, help="Override Jacobian input regularization weight")
    parser.add_argument("--grf_correction_weight", type=float, default=None, help="Override GRF correction loss weight")
    parser.add_argument("--output_reg_weight", type=float, default=None, help="Override output regularization weight")
    parser.add_argument("--contact_weight_multiplier", type=float, default=None, help="Override stance contact weighting multiplier")
    parser.add_argument("--magWeight", type=float, default=None, help="Override torque magnitude weighting scale")
    parser.add_argument("--use_contact_weighting", nargs="?", const=True, default=None, type=_parse_optional_bool_arg, help="Override COP/GRF/Moments contact weighting")
    parser.add_argument("--magOnOff", nargs="?", const=True, default=None, type=_parse_optional_bool_arg, help="Override torque magnitude weighting on/off")
    parser.add_argument("--contactOnOff", nargs="?", const=True, default=None, type=_parse_optional_bool_arg, help="Override torque contact weighting on/off")
    parser.add_argument("--cop_mask", nargs="?", const=True, default=None, type=_parse_optional_bool_arg, help="Override predicted-contact masking on COP/GRF outputs")
    parser.add_argument("--UseGRFNormCOP", nargs="?", const=True, default=None, type=_parse_optional_bool_arg, help="Use COP_CalcFrame_GroundAligned_GRFNorm.npy as the COP target")
    parser.add_argument("--use_OpenSimID_GT", nargs="?", const=True, default=None, type=_parse_optional_bool_arg, help="If true, use aligned OpenSim ID STO torques as the torque/full-ID ground truth where available.")
    parser.add_argument(
        "--use_recalculated_opensim_id_gt",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help=(
            "If true, use MoCap/OpenSim_ID_recalculated.npy as the torque/full-ID ground truth "
            "for training, evaluation, and plots (with original MoCap/ID_GT_MJX.npy shown as reference)."
        ),
    )
    parser.add_argument("--BestModelByTorque", nargs="?", const=True, default=None, type=_parse_optional_bool_arg, help="Override best-checkpoint selection metric")
    parser.add_argument("--BestModel_TorqueWeighting", type=str, default=None, help="Optional JSON dict overriding grouped best-model torque weights")
    parser.add_argument("--torque_grad_through_jacob", nargs="?", const=True, default=None, type=_parse_optional_bool_arg, help="If False, block torque-loss gradients through predicted Jacobian/rotation branches")
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
    parser.add_argument(
        "--adapter_hidden_dim",
        type=int,
        default=None,
        help="Unused legacy adapter arg retained for CLI compatibility",
    )
    parser.add_argument(
        "--adapter_dropout_rate",
        type=float,
        default=None,
        help="Unused legacy adapter arg retained for CLI compatibility",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for LOSO fine-tuning")
    parser.add_argument(
        "--inner_epoch_selection",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help="Enable inner leave-one-subject-out epoch selection before each final fold.",
    )
    parser.add_argument(
        "--inner_max_epochs",
        type=int,
        default=None,
        help="Max epochs trained during the inner sweep (defaults to --epochs).",
    )
    parser.add_argument(
        "--inner_eval_epochs",
        type=str,
        default=None,
        help="Comma-separated candidate epochs to score (e.g. '1,2,4,7'); defaults to every epoch.",
    )
    parser.add_argument(
        "--inner_selection_objective",
        type=str,
        default=None,
        help="Metric (lower is better) averaged across inner subjects to rank epochs.",
    )
    parser.add_argument(
        "--inner_selection_std_weight",
        type=float,
        default=None,
        help="Cross-subject std penalty in the epoch selection score (mean + w*std).",
    )
    parser.add_argument(
        "--max_parallel_splits",
        type=int,
        default=None,
        help="Number of inner trainings to run concurrently.",
    )
    parser.add_argument(
        "--jax_cache_clear_every_inner",
        type=int,
        default=None,
        help="Clear JAX caches after this many completed inner folds (0 disables).",
    )
    parser.add_argument(
        "--jax_cache_clear_memory_fraction",
        type=float,
        default=None,
        help="Clear JAX caches when system memory usage exceeds this fraction.",
    )
    parser.add_argument(
        "--training_excluded_subjects",
        type=str,
        default=None,
        help=(
            "Comma-separated subjects excluded from training/validation roles in all folds "
            "(they still get their own held-out test fold). E.g. 'subject5' or 'subject5,subject7'."
        ),
    )
    parser.add_argument(
        "--exclude_subjects",
        type=str,
        default=None,
        help=(
            "Comma-separated subjects removed ENTIRELY from the run: no held-out fold AND never "
            "in any training/validation set. Unlike --training_excluded_subjects, these subjects "
            "get no evaluation fold at all. E.g. 'subject5'."
        ),
    )
    parser.add_argument(
        "--useGTJacobAndRotForTraining",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help=(
            "If true, train the fine-tuned model using the ground-truth (MoCap) Jacobian and "
            "rotation matrix for the torque reconstruction, force the torque target to the MoCap "
            "qfrc_grf_contribution (even with --use_recalculated_opensim_id_gt), and supervise only "
            "COP, GRF, and grf_contribution. Validation/metric still uses the video (ProcessedData) terms."
        ),
    )
    parser.add_argument(
        "--useGTJacobAndRotForEval",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help=(
            "If true, the fine-tuned OpenCap evaluation reconstructs torque using the "
            "ground-truth (MoCap) Jacobian, rotation matrix, and qfrc_inverse while the model "
            "still predicts COP/GRF from OpenCap inputs. Isolates how much of the COP/GRF signal "
            "is captured by the OpenCap kinematics (training is unaffected)."
        ),
    )
    parser.add_argument(
        "--skipEpochSelection",
        action="store_true",
        default=False,
        help=(
            "Skip inner epoch selection entirely and train each outer fold for inner_max_epochs "
            "epochs (or --epochs if inner_max_epochs is not set)."
        ),
    )
    parser.add_argument(
        "--MocapLoso",
        action="store_true",
        default=False,
        help=(
            "Run the LOSO on the Motion Capture input pipeline (input_source=mocap) instead of the "
            "OpenCap/ProcessedData pipeline: the model is fine-tuned and evaluated on MoCap "
            "kinematics. Output defaults to a 'MocapLoso/' folder at the repo root unless "
            "--output_dir is given."
        ),
    )
    parser.add_argument(
        "--includeTrunkSway",
        action="store_true",
        default=False,
        help=(
            "Include subject*_TS trunk-sway condition folders in each base subject's LOSO fold. "
            "By default, only the 10 non-TS subject folders are analyzed."
        ),
    )
    parser.add_argument(
        "--evaluateOnTS",
        action="store_true",
        default=False,
        help=(
            "After each normal held-out subject evaluation, also evaluate the matching subject*_TS "
            "trunk-sway trials with the fine-tuned fold model and original OpenCap checkpoint, "
            "then plot knee adduction moment condition effects."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _merge_config_into_args(parse_args(), LOSO_CONFIG)
    if args.checkpoint is None:
        raise ValueError(
            "No checkpoint specified. Set LOSO_CONFIG['checkpoint'] in "
            "TransformerFilesWithCNN/loso_from_checkpoint.py or pass --checkpoint."
        )
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint, config = _load_checkpoint_bundle(checkpoint_path)
    config = _apply_cli_overrides(config, args)
    data_dir = _resolve_loso_data_dir(args.data_dir)
    config["data_dir"] = str(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"OpenCapSubjects data directory not found: {data_dir}")

    epochs = int(args.epochs if args.epochs is not None else config["epochs"])
    learning_rate = float(
        args.learning_rate if args.learning_rate is not None else config.get("learning_rate", 1e-4)
    )
    batch_size = int(args.batch_size if args.batch_size is not None else config["batch_size"])
    weight_decay = float(args.weight_decay if args.weight_decay is not None else config["weight_decay"])
    adapter_hidden_dim = int(
        args.adapter_hidden_dim if args.adapter_hidden_dim is not None else config["ff_dim"]
    )
    adapter_dropout_rate = float(
        args.adapter_dropout_rate if args.adapter_dropout_rate is not None else config["dropout_rate"]
    )

    def _resolve(name: str, default: Any) -> Any:
        value = getattr(args, name, None)
        if value is not None:
            return value
        return config.get(name, default)

    inner_epoch_selection = bool(_resolve("inner_epoch_selection", True))
    skip_epoch_selection = bool(getattr(args, "skipEpochSelection", False))
    if skip_epoch_selection:
        inner_epoch_selection = False
    inner_max_epochs = int(_resolve("inner_max_epochs", None) or epochs)
    inner_selection_objective = str(
        _resolve("inner_selection_objective", "selected_left_stance_moment_mae_percent_bwh_mean")
    )
    inner_selection_std_weight = float(_resolve("inner_selection_std_weight", 0.25))
    max_parallel_splits = max(1, int(_resolve("max_parallel_splits", 3)))
    jax_cache_clear_every_inner = int(_resolve("jax_cache_clear_every_inner", 12))
    jax_cache_clear_memory_fraction = float(_resolve("jax_cache_clear_memory_fraction", 0.85))

    inner_eval_epochs_raw = _resolve("inner_eval_epochs", None)
    if isinstance(inner_eval_epochs_raw, str):
        inner_eval_epochs_request: Optional[List[int]] = [
            int(token) for token in inner_eval_epochs_raw.replace(",", " ").split() if token.strip()
        ]
    elif inner_eval_epochs_raw:
        inner_eval_epochs_request = [int(epoch) for epoch in inner_eval_epochs_raw]
    else:
        inner_eval_epochs_request = None

    training_excluded_raw = _resolve("training_excluded_subjects", None)
    if isinstance(training_excluded_raw, str):
        training_excluded_subjects: List[str] = [
            token.strip() for token in training_excluded_raw.replace(",", " ").split() if token.strip()
        ]
    elif training_excluded_raw:
        training_excluded_subjects = [str(s) for s in training_excluded_raw]
    else:
        training_excluded_subjects = []
    mocap_loso = bool(getattr(args, "MocapLoso", False))
    include_trunk_sway = bool(getattr(args, "includeTrunkSway", False))
    evaluate_on_ts = bool(getattr(args, "evaluateOnTS", False))
    # --MocapLoso runs the LOSO on the Motion Capture input pipeline instead of OpenCap.
    # GRF/GRM are force-plate (non-kinematic), so they stay sourced from ProcessedData/.
    config["loso_input_source"] = "mocap" if mocap_loso else "video"
    config["loso_grf_grm_from_processed"] = mocap_loso
    config["evaluate_on_ts"] = evaluate_on_ts
    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    elif mocap_loso:
        output_root = (PROJECT_ROOT / "MocapLoso").resolve()
    else:
        output_root = checkpoint_path.parent / "loso_finetune_eval"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "folds").mkdir(exist_ok=True)
    if mocap_loso:
        print("🎥➡️🦴 --MocapLoso: running LOSO on the MoCap input pipeline "
              f"(input_source=mocap), output -> {output_root}", flush=True)

    if RUNTIME_ENV_APPLIED:
        print("Applied runtime safety env defaults:", flush=True)
        for key in sorted(RUNTIME_ENV_APPLIED.keys()):
            print(f"  {key}={os.environ.get(key)}", flush=True)

    print(f"Source checkpoint: {checkpoint_path}", flush=True)
    print(f"Source hyperparameters: {config['source_hyperparameters_path']}", flush=True)
    print(f"OpenCap LOSO root: {data_dir}", flush=True)
    print(
        f"Fine-tuning overrides -> epochs={epochs}, lr={learning_rate}, batch_size={batch_size}, "
        f"weight_decay={weight_decay}, adapter_hidden_dim={adapter_hidden_dim}, "
        f"adapter_dropout_rate={adapter_dropout_rate}",
        flush=True,
    )

    _trials, all_subjects, valid_subjects, subject_to_trials = _discover_subject_trials(
        data_dir,
        include_trunk_sway=include_trunk_sway,
    )

    # Fully excluded subjects are dropped from the run entirely: they get no
    # held-out fold AND never appear in any training/validation set. This differs
    # from --training_excluded_subjects, which still runs the subject's own fold.
    fully_excluded_raw = getattr(args, "exclude_subjects", None)
    if isinstance(fully_excluded_raw, str):
        fully_excluded_subjects = [
            token.strip() for token in fully_excluded_raw.replace(",", " ").split() if token.strip()
        ]
    elif fully_excluded_raw:
        fully_excluded_subjects = [str(s) for s in fully_excluded_raw]
    else:
        fully_excluded_subjects = []
    if fully_excluded_subjects:
        excluded_present = [s for s in fully_excluded_subjects if s in valid_subjects]
        excluded_set = set(fully_excluded_subjects)
        valid_subjects = [s for s in valid_subjects if s not in excluded_set]
        for subject in fully_excluded_subjects:
            subject_to_trials.pop(subject, None)
        print(
            f"Fully excluded subjects (no held-out fold, no training/validation role): "
            f"{excluded_present}",
            flush=True,
        )

    print(f"All subject dirs: {all_subjects}", flush=True)
    print(f"includeTrunkSway: {include_trunk_sway}", flush=True)
    print(f"evaluateOnTS: {evaluate_on_ts}", flush=True)
    print("Requested LOSO subjects: all discovered subjects", flush=True)
    print(f"Valid LOSO subjects: {valid_subjects}", flush=True)
    all_subject_groups = sorted({subject_group_id(subject) for subject in all_subjects}, key=_subject_sort_key)
    skipped_subjects = sorted(
        set(subject for subject in all_subject_groups if subject not in valid_subjects),
        key=_subject_sort_key,
    )
    print(f"Skipped subjects: {skipped_subjects}", flush=True)
    if training_excluded_subjects:
        print(
            f"Training-excluded subjects (own fold still runs, excluded from others' training sets): "
            f"{training_excluded_subjects}",
            flush=True,
        )

    folds = _build_loso_folds(
        valid_subjects,
        subject_to_trials,
        training_excluded_subjects=training_excluded_subjects,
    )

    if skip_epoch_selection:
        print(
            f"--skipEpochSelection: inner epoch sweep disabled; each fold will train for "
            f"{inner_max_epochs} epoch(s).",
            flush=True,
        )

    selected_epoch_by_subject: Dict[str, int] = {}
    inner_selection_summary: Dict[str, Any] = {}
    if inner_epoch_selection and len(valid_subjects) >= 3:
        inner_eval_epochs = _resolve_inner_eval_epochs(inner_eval_epochs_request, inner_max_epochs)
        print(
            f"\n=== Inner-validation epoch selection enabled "
            f"(max_epochs={inner_max_epochs}, objective={inner_selection_objective}) ===",
            flush=True,
        )
        selected_epoch_by_subject, inner_selection_summary = _select_epochs_via_inner_validation(
            folds,
            output_root=output_root,
            checkpoint=checkpoint,
            config=config,
            subject_to_trials=subject_to_trials,
            max_epochs=inner_max_epochs,
            eval_epochs=inner_eval_epochs,
            objective_name=inner_selection_objective,
            std_weight=inner_selection_std_weight,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout_rate=adapter_dropout_rate,
            base_seed=int(args.seed),
            max_parallel_splits=max_parallel_splits,
            cache_clear_every=jax_cache_clear_every_inner,
            cache_clear_memory_fraction=jax_cache_clear_memory_fraction,
        )
    elif inner_epoch_selection:
        print(
            f"Inner epoch selection requested but only {len(valid_subjects)} valid subjects "
            f"(need >= 3); falling back to fixed epochs={epochs}.",
            flush=True,
        )

    fold_results: List[Dict[str, Any]] = []
    for fold in folds:
        held_out_subject = str(fold["held_out_subject"])
        fold_dir = output_root / "folds" / held_out_subject
        fold_dir.mkdir(parents=True, exist_ok=True)
        log_path = fold_dir / "training_log.txt"
        fold_epochs = int(selected_epoch_by_subject.get(held_out_subject, inner_max_epochs if skip_epoch_selection else epochs))
        epoch_selection_info: Optional[Dict[str, Any]] = None
        if held_out_subject in selected_epoch_by_subject:
            fold_inner_summary = next(
                (
                    item
                    for item in inner_selection_summary.get("per_fold", [])
                    if str(item.get("held_out_subject")) == held_out_subject
                ),
                None,
            )
            epoch_selection_info = {
                "enabled": True,
                "selected_epoch": fold_epochs,
                "default_epochs": int(epochs),
                "max_epochs": int(inner_max_epochs),
                "objective_name": inner_selection_objective,
                "selection_std_weight": float(inner_selection_std_weight),
                "fold_summary": fold_inner_summary,
            }
        print(
            f"\n=== Running fold for held-out subject: {held_out_subject} "
            f"(epochs={fold_epochs}) ===",
            flush=True,
        )

        with log_path.open("w", encoding="utf-8") as log_handle:
            tee = TeeStream(sys.__stdout__, log_handle)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                fold_result = _run_fold(
                    fold,
                    fold_dir=fold_dir,
                    checkpoint=checkpoint,
                    config=config,
                    epochs=fold_epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    adapter_hidden_dim=adapter_hidden_dim,
                    adapter_dropout_rate=adapter_dropout_rate,
                    seed=int(args.seed + _subject_sort_key(held_out_subject)[0]),
                    epoch_selection=epoch_selection_info,
                )
        fold_results.append(fold_result)

    infer_trial_metric_rows: List[Dict[str, Any]] = []
    infer_trial_mae_reports: Dict[str, Dict[str, float]] = {}
    infer_trial_mae_reports_original_predinput: Dict[str, Dict[str, float]] = {}
    infer_trial_mae_reports_original_ocinput: Dict[str, Dict[str, float]] = {}
    infer_trial_mae_reports_original_motion_capture: Dict[str, Dict[str, float]] = {}
    infer_comparison_rows: List[Dict[str, Any]] = []
    infer_aggregated_stance_stats_by_source_groups: Dict[str, List[Mapping[str, Mapping[str, Any]]]] = {}
    for result in fold_results:
        infer_summary = result.get("held_out_infer_style", {})
        for metric in infer_summary.get("trial_metrics", []):
            infer_trial_metric_rows.append(dict(metric))
        for trial_name, mae_report in infer_summary.get("mae_reports", {}).items():
            infer_trial_mae_reports[str(trial_name)] = {
                key: float(value) for key, value in dict(mae_report).items()
            }
        for trial_name, mae_report in infer_summary.get("mae_reports_original_opencap_predinput", {}).items():
            infer_trial_mae_reports_original_predinput[str(trial_name)] = {
                key: float(value) for key, value in dict(mae_report).items()
            }
        for trial_name, mae_report in infer_summary.get("mae_reports_original_opencap_ocinput", {}).items():
            infer_trial_mae_reports_original_ocinput[str(trial_name)] = {
                key: float(value) for key, value in dict(mae_report).items()
            }
        for trial_name, mae_report in infer_summary.get("mae_reports_original_motion_capture", {}).items():
            infer_trial_mae_reports_original_motion_capture[str(trial_name)] = {
                key: float(value) for key, value in dict(mae_report).items()
            }
        for comparison_row in infer_summary.get("comparison_rows", []):
            infer_comparison_rows.append(dict(comparison_row))
        for source_label, source_stats in infer_summary.get("aggregated_stance_statistics_by_source", {}).items():
            if isinstance(source_stats, Mapping) and source_stats:
                infer_aggregated_stance_stats_by_source_groups.setdefault(str(source_label), []).append(source_stats)

    infer_style_summary_dir = output_root / "infer_style_eval"
    infer_style_summary: Dict[str, Any] = {}
    if infer_trial_metric_rows:
        mae_reports_by_source = {
            "LOSO Fine-Tuned": infer_trial_mae_reports,
            "Original OpenCap PredInput": infer_trial_mae_reports_original_predinput,
            "Original OpenCap OCInput": infer_trial_mae_reports_original_ocinput,
            "Original Motion Capture": infer_trial_mae_reports_original_motion_capture,
        }
        infer_style_summary = _write_infer_style_summary_artifacts(
            infer_style_summary_dir,
            infer_trial_metric_rows,
            infer_trial_mae_reports,
            {},
            mae_by_source=mae_reports_by_source,
        )
        _write_summary_csv(output_root / "loso_infer_trial_metrics.csv", infer_trial_metric_rows)
        average_mae_per_dof_by_source = _compute_source_average_mae_per_dof(mae_reports_by_source)
        average_joint_moment_mae_per_dof_by_source = {
            source_label: _filter_joint_moment_mae_map(source_mae)
            for source_label, source_mae in average_mae_per_dof_by_source.items()
            if source_mae
        }
        grf_by_source_trial: Dict[str, Dict[str, Dict[str, float]]] = {
            "fine_tuned_opencap_input": {},
            "original_opencap_predinput": {},
            "original_opencap_ocinput": {},
            "motioncapture_input": {},
        }
        for comparison_row in infer_comparison_rows:
            trial_name = str(comparison_row.get("trial_name", "unknown_trial"))
            source_specs = (
                ("loso_fine_tuned", "fine_tuned_opencap_input"),
                ("original_checkpoint_opencap_predinput", "original_opencap_predinput"),
                ("original_checkpoint_opencap_ocinput", "original_opencap_ocinput"),
                ("original_checkpoint_mocap", "motioncapture_input"),
            )
            for comparison_key, output_key in source_specs:
                source_payload = comparison_row.get(comparison_key)
                if not isinstance(source_payload, Mapping):
                    continue
                grf_summary = _extract_bilateral_grf_mae_percent_bw(source_payload.get("metrics"))
                if grf_summary:
                    grf_by_source_trial[output_key][trial_name] = grf_summary

        fine_tuned_trial_details = _build_trial_detail_payloads(
            infer_trial_mae_reports,
            grf_by_trial=grf_by_source_trial["fine_tuned_opencap_input"],
        )
        original_predinput_trial_details = _build_trial_detail_payloads(
            infer_trial_mae_reports_original_predinput,
            grf_by_trial=grf_by_source_trial["original_opencap_predinput"],
        )
        original_ocinput_trial_details = _build_trial_detail_payloads(
            infer_trial_mae_reports_original_ocinput,
            grf_by_trial=grf_by_source_trial["original_opencap_ocinput"],
        )
        motioncapture_trial_details = _build_trial_detail_payloads(
            infer_trial_mae_reports_original_motion_capture,
            grf_by_trial=grf_by_source_trial["motioncapture_input"],
        )
        subject_average_torque_mae_by_source = _compute_subject_average_torque_mae_by_source(
            {
                "LOSO Fine-Tuned": fine_tuned_trial_details,
                "Original OpenCap PredInput": original_predinput_trial_details,
                "Original OpenCap OCInput": original_ocinput_trial_details,
                "Original Motion Capture": motioncapture_trial_details,
            }
        )
        aggregated_stance_statistics_by_source = {
            source_label: _merge_stance_summary_stats(source_stats_group)
            for source_label, source_stats_group in infer_aggregated_stance_stats_by_source_groups.items()
            if source_stats_group
        }
        compatible_overall_mae_report = {
            "average_mae_per_dof": average_mae_per_dof_by_source.get("LOSO Fine-Tuned", {}),
            "average_joint_moment_mae_per_dof": average_joint_moment_mae_per_dof_by_source.get("LOSO Fine-Tuned", {}),
            "average_mae_per_dof_opencap_input": average_mae_per_dof_by_source.get("Original OpenCap PredInput", {}),
            "average_joint_moment_mae_per_dof_opencap_input": average_joint_moment_mae_per_dof_by_source.get("Original OpenCap PredInput", {}),
            "average_mae_per_dof_original_opencap_predinput": average_mae_per_dof_by_source.get("Original OpenCap PredInput", {}),
            "average_joint_moment_mae_per_dof_original_opencap_predinput": average_joint_moment_mae_per_dof_by_source.get("Original OpenCap PredInput", {}),
            "average_mae_per_dof_original_opencap_ocinput": average_mae_per_dof_by_source.get("Original OpenCap OCInput", {}),
            "average_joint_moment_mae_per_dof_original_opencap_ocinput": average_joint_moment_mae_per_dof_by_source.get("Original OpenCap OCInput", {}),
            "average_mae_per_dof_fine_tuned_opencap_input": average_mae_per_dof_by_source.get("LOSO Fine-Tuned", {}),
            "average_joint_moment_mae_per_dof_fine_tuned_opencap_input": average_joint_moment_mae_per_dof_by_source.get("LOSO Fine-Tuned", {}),
            "average_mae_per_dof_motioncapture_input": average_mae_per_dof_by_source.get("Original Motion Capture", {}),
            "average_joint_moment_mae_per_dof_motioncapture_input": average_joint_moment_mae_per_dof_by_source.get("Original Motion Capture", {}),
            "average_mae_per_dof_by_source": average_mae_per_dof_by_source,
            "average_joint_moment_mae_per_dof_by_source": average_joint_moment_mae_per_dof_by_source,
            "trial_details": fine_tuned_trial_details,
            "trial_details_opencap_input": original_predinput_trial_details,
            "trial_details_original_opencap_predinput": original_predinput_trial_details,
            "trial_details_original_opencap_ocinput": original_ocinput_trial_details,
            "trial_details_fine_tuned_opencap_input": fine_tuned_trial_details,
            "trial_details_motioncapture_input": motioncapture_trial_details,
            "subject_average_torque_mae_bwh_percent": subject_average_torque_mae_by_source.get("LOSO Fine-Tuned", {}),
            "subject_average_torque_mae_bwh_percent_opencap_input": subject_average_torque_mae_by_source.get("Original OpenCap PredInput", {}),
            "subject_average_torque_mae_bwh_percent_original_opencap_predinput": subject_average_torque_mae_by_source.get("Original OpenCap PredInput", {}),
            "subject_average_torque_mae_bwh_percent_original_opencap_ocinput": subject_average_torque_mae_by_source.get("Original OpenCap OCInput", {}),
            "subject_average_torque_mae_bwh_percent_fine_tuned_opencap_input": subject_average_torque_mae_by_source.get("LOSO Fine-Tuned", {}),
            "subject_average_torque_mae_bwh_percent_motioncapture_input": subject_average_torque_mae_by_source.get("Original Motion Capture", {}),
            "subject_average_torque_mae_bwh_percent_by_source": subject_average_torque_mae_by_source,
            "grf_mae_percent_bw_bilateral_stance_by_source": {
                source_label: _average_metric_dicts(trial_metrics_by_source)
                for source_label, trial_metrics_by_source in grf_by_source_trial.items()
                if trial_metrics_by_source
            },
            "trial_grf_mae_percent_bw_bilateral_stance_by_source": grf_by_source_trial,
            "torque_metric_scope": "left_stance_selected_dofs",
            "torque_metric_side": "left",
            "torque_metric_phase": "stance",
            "torque_metric_dof_names": list(infer_module.SELECTED_LEFT_STANCE_DOF_NAMES)
            + [infer_module.LEFT_STANCE_KAM_DOF_NAME],
            "source": "loso_from_checkpoint",
            "source_summary_json": str(infer_style_summary_dir / "infer_style_summary.json"),
            "source_trial_metrics_csv": str(output_root / "loso_infer_trial_metrics.csv"),
        }
        if aggregated_stance_statistics_by_source:
            compatible_overall_mae_report["aggregated_stance_statistics_by_source"] = (
                aggregated_stance_statistics_by_source
            )
            compatible_overall_mae_report["aggregated_stance_statistics"] = (
                aggregated_stance_statistics_by_source.get("LOSO Fine-Tuned", {})
            )
        _save_json(output_root / "overall_mae_report.json", compatible_overall_mae_report)
        if aggregated_stance_statistics_by_source:
            infer_style_summary["aggregated_stance_statistics_by_source"] = aggregated_stance_statistics_by_source
            infer_style_summary["aggregated_stance_statistics"] = aggregated_stance_statistics_by_source.get(
                "LOSO Fine-Tuned",
                {},
            )
            _save_json(infer_style_summary_dir / "infer_style_summary.json", infer_style_summary)
            _save_json(
                output_root / "aggregated_stance_statistics_by_source.json",
                aggregated_stance_statistics_by_source,
            )
            _save_json(
                output_root / "aggregated_stance_statistics.json",
                aggregated_stance_statistics_by_source.get("LOSO Fine-Tuned", {}),
            )
    infer_comparison_metric_means: Dict[str, Any] = {}
    infer_comparison_metric_stds: Dict[str, Any] = {}
    if infer_comparison_rows:
        infer_comparison_metric_means, infer_comparison_metric_stds = _aggregate_metric_dicts(
            infer_comparison_rows
        )
        _save_json(
            output_root / "loso_model_comparison_summary.json",
            {
                "trial_count": len(infer_comparison_rows),
                "metric_means": infer_comparison_metric_means,
                "metric_stds": infer_comparison_metric_stds,
                "per_trial": infer_comparison_rows,
            },
        )

    held_out_metric_views = [
        {
            "held_out_subject": result["held_out_subject"],
            "inner_val_subject": result["inner_val_subject"],
            "best_epoch": result["best_epoch"],
            **result["held_out_metrics"]["metrics"],
        }
        for result in fold_results
    ]
    metric_means, metric_stds = _aggregate_metric_dicts(
        [result["held_out_metrics"]["metrics"] for result in fold_results]
    )
    summary_payload = {
        "source_checkpoint": str(checkpoint_path),
        "source_hyperparameters_path": config["source_hyperparameters_path"],
        "data_dir": str(data_dir),
        "requested_loso_subjects": "all_discovered_subjects",
        "valid_subjects": valid_subjects,
        "skipped_subjects": skipped_subjects,
        "completed_folds": len(fold_results),
        "adapter_settings": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "adapter_hidden_dim": adapter_hidden_dim,
            "adapter_dropout_rate": adapter_dropout_rate,
            "seed": int(args.seed),
        },
        "include_trunk_sway": include_trunk_sway,
        "evaluate_on_ts": evaluate_on_ts,
        "training_excluded_subjects": training_excluded_subjects,
        "inner_epoch_selection_enabled": bool(selected_epoch_by_subject),
        "selected_epoch_by_subject": selected_epoch_by_subject,
        "inner_epoch_selection_summary": inner_selection_summary,
        "per_fold": fold_results,
        "held_out_metric_rows": held_out_metric_views,
        "metric_means": metric_means,
        "metric_stds": metric_stds,
        "infer_style_eval_output_dir": str(infer_style_summary_dir) if infer_trial_metric_rows else None,
        "infer_style_trial_count": len(infer_trial_metric_rows),
        "infer_style_trial_metric_rows": infer_trial_metric_rows,
        "infer_style_metric_means": infer_style_summary.get("metric_means", {}),
        "infer_style_metric_stds": infer_style_summary.get("metric_stds", {}),
        "infer_style_average_mae_per_dof": infer_style_summary.get("average_mae_per_dof", {}),
        "infer_style_subject_average_torque_mae_bwh_percent": infer_style_summary.get(
            "subject_average_torque_mae_bwh_percent"
        ),
        "infer_style_subject_average_torque_mae_bwh_percent_by_source": infer_style_summary.get(
            "subject_average_torque_mae_bwh_percent_by_source", {}
        ),
        "infer_style_model_comparison_rows": infer_comparison_rows,
        "infer_style_model_comparison_metric_means": infer_comparison_metric_means,
        "infer_style_model_comparison_metric_stds": infer_comparison_metric_stds,
    }
    _save_json(output_root / "loso_summary.json", summary_payload)
    _write_summary_csv(output_root / "loso_summary.csv", held_out_metric_views)
    print(f"\nSaved LOSO summary to {output_root / 'loso_summary.json'}", flush=True)
    print(f"Saved LOSO CSV summary to {output_root / 'loso_summary.csv'}", flush=True)
    if infer_trial_metric_rows:
        print(f"Saved infer-style LOSO summary to {infer_style_summary_dir / 'infer_style_summary.json'}", flush=True)
        print(f"Saved infer-style LOSO trial CSV to {output_root / 'loso_infer_trial_metrics.csv'}", flush=True)
        print(f"Saved compareMAE-compatible summary to {output_root / 'overall_mae_report.json'}", flush=True)
        subject_average_by_source = infer_style_summary.get("subject_average_torque_mae_bwh_percent_by_source", {})
        if subject_average_by_source:
            print("Subject-averaged torque MAE used by compareMAEAcrossSub:", flush=True)
            for source_label, source_summary in subject_average_by_source.items():
                summary_line = _format_subject_average_summary_line(source_label, source_summary)
                if summary_line:
                    print(f"  {summary_line}", flush=True)
    if infer_comparison_rows:
        print(f"Saved LOSO model comparison summary to {output_root / 'loso_model_comparison_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
