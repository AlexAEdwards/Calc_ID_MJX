#!/usr/bin/env python3
"""Local hyperparameter search runner that mirrors the repository sweep YAML.

This script is intended as a WandB-free replacement for the current sweep file.
It reads the sweep definition, runs local training jobs sequentially, ranks the
results, and writes condensed summaries for the top-performing models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SWEEP_FILE = SCRIPT_DIR / "HPOApril24.yaml"
SUMMARY_DIR_NAME = "condensed_results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "local_hpo_searches"
DEFAULT_EXP_NAME = "LocalHPOSearch"

# Trials/subjects excluded from every HPO trial (forwarded to train.py via
# --exclude_trials / --exclude_prefixes). Entries with a "/" (e.g.
# "OA19/Trial_5") drop a single trial; bare entries drop every trial for that
# subject. Prefer loading the actively maintained lists from
# train_single_model.py so local HPO and single-model training stay aligned.
FALLBACK_EXCLUDE_PREFIXES = ["SUBJ"]
FALLBACK_EXCLUDE_FROM_TRAINING = [
    "OA19/Trial_5",
    "OA19/Trial_6",
    "OA18/Trial_3",
    "OA18/Trial_11",
    "OA10/Trial_6",
    "SUBJ12/Trial_1",
    "SUBJ12/Trial_2",
    "SUBJ44/Trial_1",
    "GaitRetraining_Subject125/Trial_1",
    "GaitRetraining_Subject138/Trial_4",
    "GaitRetraining_Subject138/Trial_28",
    "GaitRetraining_Subject153/Trial_5",
    "GaitRetraining_SubjectR583/Trial_12",
    "04/Trial_26",
    "Y21/Trial_14",
    "S11/Trial_1",
    "S11/Trial_2",
]
try:
    from train_single_model import EXCLUDE_FROM_TRAINING as _SINGLE_MODEL_EXCLUDES
    from train_single_model import EXCLUDE_PREFIXES as _SINGLE_MODEL_PREFIXES

    EXCLUDE_FROM_TRAINING = list(_SINGLE_MODEL_EXCLUDES)
    EXCLUDE_PREFIXES = list(_SINGLE_MODEL_PREFIXES)
except Exception:
    EXCLUDE_FROM_TRAINING = list(FALLBACK_EXCLUDE_FROM_TRAINING)
    EXCLUDE_PREFIXES = list(FALLBACK_EXCLUDE_PREFIXES)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    mode: str
    values: Optional[Tuple[Any, ...]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @classmethod
    def from_yaml(cls, name: str, payload: Dict[str, Any]) -> "ParameterSpec":
        if "values" in payload:
            return cls(name=name, mode="values", values=tuple(payload["values"]))
        distribution = str(payload.get("distribution", "uniform"))
        if distribution not in {"uniform", "log_uniform_values"}:
            raise ValueError(f"Unsupported sweep distribution for '{name}': {distribution}")
        return cls(
            name=name,
            mode=distribution,
            minimum=float(payload["min"]),
            maximum=float(payload["max"]),
        )

    def sample(self, rng: np.random.Generator) -> Any:
        if self.mode == "values":
            idx = int(rng.integers(0, len(self.values)))
            return self.values[idx]
        if self.mode == "uniform":
            return float(rng.uniform(self.minimum, self.maximum))
        if self.mode == "log_uniform_values":
            lo = math.log(self.minimum)
            hi = math.log(self.maximum)
            return float(math.exp(rng.uniform(lo, hi)))
        raise ValueError(f"Unsupported sampling mode: {self.mode}")

    def encode(self, value: Any) -> float:
        if self.mode == "values":
            numeric_values: List[Optional[float]] = []
            for item in self.values:
                try:
                    numeric_values.append(float(item))
                except (TypeError, ValueError):
                    numeric_values.append(None)
            if all(v is not None for v in numeric_values):
                lo = min(numeric_values)
                hi = max(numeric_values)
                if hi <= lo:
                    return 0.0
                return float((float(value) - lo) / (hi - lo))
            idx = self.values.index(value)
            if len(self.values) == 1:
                return 0.0
            return float(idx / (len(self.values) - 1))
        if self.mode == "uniform":
            if self.maximum <= self.minimum:
                return 0.0
            return float((float(value) - self.minimum) / (self.maximum - self.minimum))
        if self.mode == "log_uniform_values":
            lo = math.log(self.minimum)
            hi = math.log(self.maximum)
            if hi <= lo:
                return 0.0
            return float((math.log(float(value)) - lo) / (hi - lo))
        raise ValueError(f"Unsupported encoding mode: {self.mode}")

    def summary(self) -> str:
        if self.mode == "values":
            return "[" + ", ".join(_format_value(v) for v in self.values) + "]"
        if self.mode == "uniform":
            return f"uniform({self.minimum:g}, {self.maximum:g})"
        return f"log_uniform({self.minimum:g}, {self.maximum:g})"


@dataclass
class ActiveTrial:
    trial_index: int
    worker_slot: int
    run_name: str
    run_dir: Path
    log_path: Path
    log_handle: Any
    process: subprocess.Popen[Any]
    base_payload: Dict[str, Any]
    start_time: float
    assigned_cuda_visible_devices: Optional[str]
    jax_gpu_mem_fraction: Optional[float]
    jax_cpu_threads: int


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _slugify(value: str, max_len: int = 96) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value).strip())
    cleaned = cleaned.strip("._-") or "item"
    return cleaned[:max_len]


def _json_compatible(value: Any) -> Any:
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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_compatible(payload), f, indent=2)


def _tail_lines(path: Path, num_lines: int = 40) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    return [line.rstrip("\n") for line in lines[-num_lines:]]


def _read_sweep_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep file did not parse into a dictionary: {path}")
    return payload


def _normalize_command_list(command_items: Sequence[Any]) -> List[str]:
    normalized: List[str] = []
    for item in command_items:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in {"${env}", "${interpreter}", "${program}", "${args_no_boolean_flags}"}:
            continue
        if text == "--use_wandb" or text.startswith("--wandb_"):
            continue
        normalized.append(text)
    return normalized


def _apply_command_overrides(
    base_args: Sequence[str],
    *,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    exp_name: Optional[str] = None,
) -> List[str]:
    """Apply optional command-line overrides while preserving the sweep defaults."""
    override_names = {"data_dir", "output_dir", "exp_name"}
    normalized = _strip_args(base_args, override_names)
    if data_dir is not None:
        normalized.append(f"--data_dir={data_dir}")
    if output_dir is not None:
        normalized.append(f"--output_dir={output_dir}")
    if exp_name is not None:
        normalized.append(f"--exp_name={exp_name}")
    return normalized


def _get_arg_value(args: Sequence[str], name: str) -> Optional[str]:
    prefix = f"--{name}="
    flag = f"--{name}"
    for idx, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == flag and idx + 1 < len(args):
            return args[idx + 1]
    return None


def _strip_args(args: Sequence[str], names: Iterable[str]) -> List[str]:
    target_names = set(names)
    stripped: List[str] = []
    skip_next = False
    for idx, token in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if not token.startswith("--"):
            stripped.append(token)
            continue
        flag_name = token[2:].split("=", 1)[0]
        if flag_name in target_names:
            if "=" not in token and idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                skip_next = True
            continue
        stripped.append(token)
    return stripped


def _resolve_program_path(program_value: str) -> Path:
    program_path = Path(program_value)
    if not program_path.is_absolute():
        program_path = (PROJECT_ROOT / program_path).resolve()
    return program_path


def _parse_parameter_specs(sweep_payload: Dict[str, Any]) -> List[ParameterSpec]:
    params_block = sweep_payload.get("parameters", {})
    if not isinstance(params_block, dict) or not params_block:
        raise ValueError("Sweep file does not define any parameters.")
    return [ParameterSpec.from_yaml(name, payload) for name, payload in params_block.items()]


def _parameter_signature(params: Dict[str, Any]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _sample_unique_random_config(
    specs: Sequence[ParameterSpec],
    rng: np.random.Generator,
    existing_signatures: set[str],
) -> Dict[str, Any]:
    for _ in range(10000):
        candidate = {spec.name: spec.sample(rng) for spec in specs}
        signature = _parameter_signature(candidate)
        if signature not in existing_signatures:
            return candidate
    raise RuntimeError("Could not sample a unique hyperparameter configuration.")


def _encode_config(specs: Sequence[ParameterSpec], params: Dict[str, Any]) -> np.ndarray:
    return np.asarray([spec.encode(params[spec.name]) for spec in specs], dtype=np.float64)


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    improvement = best_y - mean - xi
    out = np.zeros_like(mean, dtype=np.float64)
    valid = std > 1e-12
    if np.any(valid):
        z = improvement[valid] / std[valid]
        out[valid] = improvement[valid] * norm.cdf(z) + std[valid] * norm.pdf(z)
    return out


def _propose_next_config(
    specs: Sequence[ParameterSpec],
    completed_results: Sequence[Dict[str, Any]],
    rng: np.random.Generator,
    startup_trials: int,
    candidate_pool_size: int,
    extra_excluded_signatures: Optional[set[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    existing_signatures = {result["parameter_signature"] for result in completed_results}
    if extra_excluded_signatures:
        existing_signatures.update(extra_excluded_signatures)
    usable = [result for result in completed_results if result["status"] == "completed" and result.get("objective") is not None]
    if len(usable) < startup_trials:
        return _sample_unique_random_config(specs, rng, existing_signatures), "random"

    x_train = np.asarray([_encode_config(specs, result["parameters"]) for result in usable], dtype=np.float64)
    y_train = np.asarray([float(result["objective"]) for result in usable], dtype=np.float64)
    if x_train.shape[0] < 2 or np.unique(y_train).size <= 1:
        return _sample_unique_random_config(specs, rng, existing_signatures), "random"

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(x_train.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=4,
        random_state=0,
    )
    try:
        model.fit(x_train, y_train)
    except Exception:
        return _sample_unique_random_config(specs, rng, existing_signatures), "random"

    candidate_configs: List[Dict[str, Any]] = []
    candidate_vectors: List[np.ndarray] = []
    local_signatures: set[str] = set()
    max_attempts = max(candidate_pool_size * 10, 5000)
    attempts = 0
    while len(candidate_configs) < candidate_pool_size and attempts < max_attempts:
        attempts += 1
        candidate = {spec.name: spec.sample(rng) for spec in specs}
        signature = _parameter_signature(candidate)
        if signature in existing_signatures or signature in local_signatures:
            continue
        local_signatures.add(signature)
        candidate_configs.append(candidate)
        candidate_vectors.append(_encode_config(specs, candidate))

    if not candidate_configs:
        return _sample_unique_random_config(specs, rng, existing_signatures), "random"

    x_candidates = np.asarray(candidate_vectors, dtype=np.float64)
    mean, std = model.predict(x_candidates, return_std=True)
    best_y = float(np.min(y_train))
    ei = _expected_improvement(mean, std, best_y)
    best_idx = int(np.argmax(ei))
    return candidate_configs[best_idx], "bayes"


def _build_trial_name(base_exp_name: str, trial_index: int, params: Dict[str, Any]) -> str:
    signature = _parameter_signature(params)[:8]
    dm = _format_value(params.get("d_model", "na"))
    nl = _format_value(params.get("num_layers", "na"))
    lr = _format_value(params.get("learning_rate", "na"))
    dr = _format_value(params.get("dropout_rate", "na"))
    return _slugify(f"{base_exp_name}_trial_{trial_index:04d}_dm{dm}_nl{nl}_lr{lr}_dr{dr}_{signature}", max_len=120)


def _resolve_train_params(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """Expand synthetic sweep params into the concrete train.py CLI arguments."""
    resolved = dict(raw_params)

    knee_weight = resolved.pop("knee_weight", None)
    if knee_weight is not None:
        resolved["knee_r_weight"] = float(knee_weight)
        resolved["knee_l_weight"] = float(knee_weight)

    if "d_model" in resolved and "ff_dim" not in resolved:
        resolved["ff_dim"] = int(resolved["d_model"]) * 4

    reg_ratio = resolved.pop("reg_ratio", None)
    if reg_ratio is not None:
        reg_ratio = float(reg_ratio)
        if reg_ratio <= 0.0:
            raise ValueError(f"reg_ratio must be positive, got {reg_ratio}")

        derived_pairs = [
            ("qfrc_inverse_weight", "qfrc_inverse_input_reg_weight"),
            ("rotation_weight", "rotation_input_reg_weight"),
            ("jacobian_weight", "jacobian_input_reg_weight"),
        ]
        for main_weight_name, reg_weight_name in derived_pairs:
            if reg_weight_name in resolved:
                continue
            main_weight_value = resolved.get(main_weight_name)
            if main_weight_value is None:
                continue
            resolved[reg_weight_name] = float(main_weight_value) * reg_ratio

    return resolved


def _build_trial_command(
    program_path: Path,
    base_args: Sequence[str],
    run_dir: Path,
    run_name: str,
    params: Dict[str, Any],
    epochs: int,
) -> List[str]:
    resolved_params = _resolve_train_params(params)
    override_names = {
        "output_dir",
        "exp_name",
        "vis_interval",
        "save_final_predictions_only",
        "save_best_model_png_only",
        "save_model_epochs",
        "BestModelByTorque",
        "exclude_trials",
        "exclude_prefixes",
    } | set(params.keys()) | set(resolved_params.keys())
    stripped_base_args = _strip_args(base_args, override_names)
    command = [sys.executable, str(program_path)]
    command.extend(stripped_base_args)
    command.extend(
        [
            f"--output_dir={run_dir}",
            f"--exp_name={run_name}",
            "--vis_interval=0",
            "--save_model_epochs=7,10,12,15,18,20,25,30,40",
            "--BestModelByTorque=true",
            "--save_best_model_png_only",
            "--disable_validation_outlier_plots",
        ]
    )
    # Forward the shared exclusion list to every HPO trial (balancing flags, if
    # any, pass through from the sweep's base command args).
    if EXCLUDE_FROM_TRAINING:
        command.append(f"--exclude_trials={json.dumps(EXCLUDE_FROM_TRAINING)}")
    if EXCLUDE_PREFIXES:
        command.append(f"--exclude_prefixes={json.dumps(EXCLUDE_PREFIXES)}")
    for key, value in resolved_params.items():
        command.append(f"--{key}={_format_value(value)}")
    return command


def _metric_alias_candidates(metric_key: str) -> List[str]:
    candidates = [metric_key]
    if metric_key.startswith("torque_mae_percent_bilateral_"):
        suffix = metric_key[len("torque_mae_percent_bilateral_"):]
        candidates.append(f"torque_mae_pct_norm_bilateral_{suffix}")
    elif metric_key.startswith("torque_mae_pct_norm_bilateral_"):
        suffix = metric_key[len("torque_mae_pct_norm_bilateral_"):]
        candidates.append(f"torque_mae_percent_bilateral_{suffix}")

    if metric_key.startswith("torque_mae_percent_bilateral_stance_"):
        suffix = metric_key[len("torque_mae_percent_bilateral_stance_"):]
        candidates.append(f"torque_mae_pct_norm_bilateral_stance_{suffix}")
    elif metric_key.startswith("torque_mae_pct_norm_bilateral_stance_"):
        suffix = metric_key[len("torque_mae_pct_norm_bilateral_stance_"):]
        candidates.append(f"torque_mae_percent_bilateral_stance_{suffix}")
    return candidates


def _coerce_metric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        metric_value = float(value)
    except (TypeError, ValueError):
        return None
    return metric_value if np.isfinite(metric_value) else None


def _mean_of_metric_map(value: Any) -> Optional[float]:
    if not isinstance(value, dict) or not value:
        return None
    values = [_coerce_metric_value(item) for item in value.values()]
    values = [item for item in values if item is not None]
    if not values:
        return None
    return float(np.mean(values))


def _lookup_epoch_metric(epoch_metrics: Dict[str, Any], split: str, metric_key: str) -> Optional[float]:
    split_metrics = epoch_metrics.get(split) or {}
    if not isinstance(split_metrics, dict):
        return None

    direct_value = _coerce_metric_value(split_metrics.get(metric_key))
    if direct_value is not None:
        return direct_value

    mean_metric_maps = {
        "torque_mae_pct_norm_bilateral_mean": "torque_mae_pct_norm_bilateral",
        "torque_mae_percent_bilateral_mean": "torque_mae_pct_norm_bilateral",
        "torque_mae_percent_bilateral_stance_mean": "torque_mae_percent_bilateral_stance",
        "torque_rmse_bilateral_mean_Nm": "torque_rmse_bilateral_Nm",
    }
    mean_metric_map_name = mean_metric_maps.get(metric_key)
    if mean_metric_map_name is not None:
        mean_value = _mean_of_metric_map(split_metrics.get(mean_metric_map_name))
        if mean_value is not None:
            return mean_value

    for candidate in _metric_alias_candidates(metric_key):
        for nested_prefix in ("torque_rmse_bilateral_",):
            suffix = "_Nm"
            if candidate.startswith(nested_prefix) and candidate.endswith(suffix):
                nested_map_name = f"{nested_prefix[:-1]}{suffix}"
                nested_map = split_metrics.get(nested_map_name)
                if isinstance(nested_map, dict):
                    nested_key = candidate[len(nested_prefix):-len(suffix)]
                    nested_value = _coerce_metric_value(nested_map.get(nested_key))
                    if nested_value is not None:
                        return nested_value

        for nested_prefix in (
            "torque_mae_pct_norm_bilateral_",
            "torque_mae_percent_bilateral_",
            "torque_mae_percent_bilateral_stance_",
            "grf_mae_percent_bw_bilateral_",
        ):
            if candidate.startswith(nested_prefix):
                nested_map_name = nested_prefix[:-1]
                nested_map = split_metrics.get(nested_map_name)
                if isinstance(nested_map, dict):
                    nested_key = candidate[len(nested_prefix):]
                    nested_value = _coerce_metric_value(nested_map.get(nested_key))
                    if nested_value is not None:
                        return nested_value

    return None


def _extract_metric_value(summary: Dict[str, Any], metric_name: str) -> Optional[float]:
    metric_map = {
        "val/best_torque_score_Nm": summary.get("best_torque_score"),
        "val/weighted_torque_score_Nm": (
            (summary.get("final_epoch_metrics") or {}).get("weighted_torque_score_Nm")
            if isinstance(summary.get("final_epoch_metrics"), dict)
            else None
        ),
        "val/torque_rmse_fullset_overall_Nm": (
            ((summary.get("final_epoch_metrics") or {}).get("val") or {}).get("torque_rmse_fullset_overall_Nm")
            if isinstance(summary.get("final_epoch_metrics"), dict)
            else None
        ),
        "val/total_loss": (
            (summary.get("final_epoch_metrics") or {}).get("val_total_loss")
            if isinstance(summary.get("final_epoch_metrics"), dict)
            else None
        ),
        "train/total_loss": (
            (summary.get("final_epoch_metrics") or {}).get("train_total_loss")
            if isinstance(summary.get("final_epoch_metrics"), dict)
            else None
        ),
    }
    if metric_name in metric_map:
        metric_value = _coerce_metric_value(metric_map[metric_name])
        if metric_value is not None:
            return metric_value

    split, sep, metric_key = metric_name.partition("/")
    if sep and split in {"train", "val"}:
        final_epoch = summary.get("final_epoch_metrics") or {}
        if isinstance(final_epoch, dict):
            metric_value = _lookup_epoch_metric(final_epoch, split, metric_key)
            if metric_value is not None:
                return metric_value

        best_epoch = summary.get("best_epoch_metrics") or {}
        if isinstance(best_epoch, dict):
            metric_value = _lookup_epoch_metric(best_epoch, split, metric_key)
            if metric_value is not None:
                return metric_value

    return None


def _trial_result_from_summary(
    summary: Dict[str, Any],
    metric_name: str,
    metric_goal: str,
    base_payload: Dict[str, Any],
) -> Dict[str, Any]:
    metric_value = _extract_metric_value(summary, metric_name)
    objective = None
    if metric_value is not None and np.isfinite(metric_value):
        objective = float(metric_value) if metric_goal == "minimize" else float(-metric_value)

    best_epoch_metrics = summary.get("best_epoch_metrics") or {}
    final_epoch_metrics = summary.get("final_epoch_metrics") or {}
    artifacts = summary.get("artifacts") or {}
    payload = dict(base_payload)
    payload.update(
        {
            "status": "completed" if base_payload["returncode"] == 0 and objective is not None else "failed",
            "metric_value": metric_value,
            "objective": objective,
            "best_torque_score": summary.get("best_torque_score"),
            "best_val_loss": summary.get("best_val_loss"),
            "best_model_epoch": summary.get("best_model_epoch"),
            "final_val_total_loss": summary.get("final_val_total_loss"),
            "final_val_torque_rmse_Nm": summary.get("final_val_torque_rmse_Nm"),
            "final_val_cop_rmse_m": summary.get("final_val_cop_rmse_m"),
            "final_val_grf_rmse_N": summary.get("final_val_grf_rmse_N"),
            "final_val_moments_rmse_Nm": summary.get("final_val_moments_rmse_Nm"),
            "final_weighted_torque_score_Nm": summary.get("final_weighted_torque_score_Nm"),
            "summary_path": str(Path(base_payload["output_dir"]) / "training_summary.json"),
            "hyperparameters_path": artifacts.get("hyperparameters_path"),
            "final_prediction_plot": artifacts.get("final_prediction_plot"),
            "best_prediction_plot": artifacts.get("best_predictions_plot"),
            "loss_history_plot": artifacts.get("loss_history_plot"),
            "best_epoch_val_torque_rmse_Nm": (best_epoch_metrics.get("val") or {}).get("torque_rmse_fullset_overall_Nm"),
            "best_epoch_weighted_torque_score_Nm": best_epoch_metrics.get("weighted_torque_score_Nm"),
            "final_epoch_val_torque_rmse_bilateral_Nm": (final_epoch_metrics.get("val") or {}).get("torque_rmse_bilateral_Nm"),
        }
    )
    return payload


def _parse_cuda_visible_devices_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    devices = [part.strip() for part in str(value).split(",") if part.strip()]
    return devices or None


def _select_worker_slot(active_trials: Sequence[ActiveTrial], max_parallel: int) -> int:
    used_slots = {trial.worker_slot for trial in active_trials}
    for slot in range(1, max_parallel + 1):
        if slot not in used_slots:
            return slot
    raise RuntimeError("No free worker slot was available.")


def _build_worker_env(
    worker_slot: int,
    *,
    jax_multi_agent_safe: bool,
    jax_gpu_mem_fraction: Optional[float],
    jax_cpu_threads: int,
    cuda_visible_devices: Optional[Sequence[str]],
) -> Tuple[Dict[str, str], Optional[str]]:
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    if jax_multi_agent_safe:
        env["JAX_MULTI_AGENT_SAFE"] = "true"
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        if jax_gpu_mem_fraction is not None:
            env["JAX_GPU_MEM_FRACTION"] = str(jax_gpu_mem_fraction)
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(jax_gpu_mem_fraction))
        env["JAX_CPU_THREADS"] = str(int(jax_cpu_threads))
        for thread_var in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]:
            env.setdefault(thread_var, str(int(jax_cpu_threads)))

    assigned_devices = None
    if cuda_visible_devices:
        assigned_devices = str(cuda_visible_devices[(worker_slot - 1) % len(cuda_visible_devices)])
        env["CUDA_VISIBLE_DEVICES"] = assigned_devices

    return env, assigned_devices


def _launch_trial(
    trial_index: int,
    program_path: Path,
    base_args: Sequence[str],
    base_exp_name: str,
    params: Dict[str, Any],
    epochs: int,
    search_root: Path,
    proposal_source: str,
    worker_slot: int,
    jax_multi_agent_safe: bool,
    jax_gpu_mem_fraction: Optional[float],
    jax_cpu_threads: int,
    cuda_visible_devices: Optional[Sequence[str]],
) -> ActiveTrial:
    run_name = _build_trial_name(base_exp_name, trial_index, params)
    run_dir = search_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "trial.log"
    command = _build_trial_command(program_path, base_args, run_dir, run_name, params, epochs)

    base_payload = {
        "trial_index": trial_index,
        "trial_name": run_name,
        "proposal_source": proposal_source,
        "parameter_signature": _parameter_signature(params),
        "parameters": _json_compatible(params),
        "resolved_parameters": _json_compatible(_resolve_train_params(params)),
        "worker_slot": worker_slot,
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "command": command,
        "returncode": None,
        "duration_s": 0.0,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "status": "pending",
        "metric_value": None,
        "objective": None,
        "assigned_cuda_visible_devices": None,
        "jax_gpu_mem_fraction": jax_gpu_mem_fraction if jax_multi_agent_safe else None,
        "jax_cpu_threads": int(jax_cpu_threads),
    }

    env, assigned_cuda_visible_devices = _build_worker_env(
        worker_slot,
        jax_multi_agent_safe=jax_multi_agent_safe,
        jax_gpu_mem_fraction=jax_gpu_mem_fraction,
        jax_cpu_threads=jax_cpu_threads,
        cuda_visible_devices=cuda_visible_devices,
    )
    base_payload["assigned_cuda_visible_devices"] = assigned_cuda_visible_devices

    print(f"\n[{trial_index}] Launching {run_name} on worker {worker_slot}", flush=True)
    print(f"    Output: {run_dir}", flush=True)
    print(f"    Log:    {log_path}", flush=True)
    if assigned_cuda_visible_devices is not None:
        print(f"    CUDA_VISIBLE_DEVICES={assigned_cuda_visible_devices}", flush=True)
    if jax_multi_agent_safe:
        print(
            f"    Conservative JAX runtime: gpu_mem_fraction={jax_gpu_mem_fraction}, cpu_threads={jax_cpu_threads}",
            flush=True,
        )

    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(f"WORKER_SLOT: {worker_slot}\n")
    if assigned_cuda_visible_devices is not None:
        log_handle.write(f"CUDA_VISIBLE_DEVICES: {assigned_cuda_visible_devices}\n")
    if jax_multi_agent_safe:
        log_handle.write(f"JAX_GPU_MEM_FRACTION: {jax_gpu_mem_fraction}\n")
        log_handle.write(f"JAX_CPU_THREADS: {jax_cpu_threads}\n")
    log_handle.write("COMMAND:\n")
    log_handle.write(" ".join(command) + "\n\n")
    log_handle.flush()

    process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    return ActiveTrial(
        trial_index=trial_index,
        worker_slot=worker_slot,
        run_name=run_name,
        run_dir=run_dir,
        log_path=log_path,
        log_handle=log_handle,
        process=process,
        base_payload=base_payload,
        start_time=time.time(),
        assigned_cuda_visible_devices=assigned_cuda_visible_devices,
        jax_gpu_mem_fraction=jax_gpu_mem_fraction if jax_multi_agent_safe else None,
        jax_cpu_threads=int(jax_cpu_threads),
    )


def _finalize_trial(
    active_trial: ActiveTrial,
    metric_name: str,
    metric_goal: str,
) -> Dict[str, Any]:
    returncode = active_trial.process.wait()
    active_trial.log_handle.flush()
    active_trial.log_handle.close()

    duration_s = time.time() - active_trial.start_time
    base_payload = dict(active_trial.base_payload)
    base_payload["returncode"] = int(returncode)
    base_payload["duration_s"] = float(duration_s)
    base_payload["finished_at"] = datetime.now().isoformat(timespec="seconds")

    summary_path = active_trial.run_dir / "training_summary.json"
    if summary_path.exists():
        try:
            summary = _load_json(summary_path)
            summary = _prune_trial_artifacts(active_trial.run_dir, summary)
            result = _trial_result_from_summary(summary, metric_name, metric_goal, base_payload)
        except Exception as exc:
            result = dict(base_payload)
            result["status"] = "failed"
            result["failure_reason"] = f"Could not parse training summary: {exc}"
            result["failure_log_tail"] = _tail_lines(active_trial.log_path, num_lines=30)
    else:
        result = dict(base_payload)
        result["status"] = "failed"
        result["failure_reason"] = "training_summary.json was not produced"
        result["failure_log_tail"] = _tail_lines(active_trial.log_path, num_lines=30)

    status_text = result["status"]
    metric_text = "n/a"
    if result.get("metric_value") is not None:
        metric_text = f"{float(result['metric_value']):.6f}"
    print(
        f"    Worker {active_trial.worker_slot} finished trial {active_trial.trial_index} "
        f"in {duration_s/60.0:.1f} min | status={status_text} | {metric_name}={metric_text}",
        flush=True,
    )
    return _json_compatible(result)


def _prediction_plot_path(run_dir: Path, epoch: Optional[int]) -> Optional[Path]:
    if epoch is None:
        return None
    path = run_dir / f"predictions_epoch_{int(epoch):04d}.png"
    return path if path.exists() else None


def _prune_trial_artifacts(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the best-model prediction plot for local HPO runs."""
    kept_summary = dict(summary)
    artifacts = dict(kept_summary.get("artifacts") or {})

    best_epoch = kept_summary.get("best_model_epoch")
    best_model_plot = run_dir / "best_model.png"
    best_plot = best_model_plot if best_model_plot.exists() else _prediction_plot_path(run_dir, best_epoch)
    if best_plot is not None and best_plot.name != "best_model.png":
        try:
            shutil.copy2(best_plot, best_model_plot)
            best_plot = best_model_plot
        except OSError:
            pass

    keep_paths = {
        path.resolve()
        for path in (best_plot,)
        if path is not None
    }
    for plot_path in run_dir.glob("predictions_epoch_*.png"):
        if plot_path.resolve() in keep_paths:
            continue
        try:
            plot_path.unlink()
        except OSError:
            pass
    for png_path in run_dir.glob("*.png"):
        if png_path.resolve() in keep_paths:
            continue
        try:
            png_path.unlink()
        except OSError:
            pass

    artifacts["first_prediction_plot"] = None
    artifacts["best_predictions_plot"] = str(best_plot) if best_plot is not None else None
    artifacts["final_prediction_plot"] = str(best_plot) if best_plot is not None else None
    artifacts["hpo_saved_prediction_plots"] = [str(best_plot)] if best_plot is not None else []
    artifacts["loss_history_plot"] = None
    artifacts["latest_validation_outliers_plot"] = None
    kept_summary["artifacts"] = artifacts
    _write_json(run_dir / "training_summary.json", kept_summary)
    return kept_summary


def _terminate_active_trials(active_trials: Sequence[ActiveTrial]) -> None:
    for active_trial in active_trials:
        if active_trial.process.poll() is not None:
            continue
        try:
            active_trial.process.terminate()
        except Exception:
            pass
    for active_trial in active_trials:
        if active_trial.process.poll() is not None:
            continue
        try:
            active_trial.process.wait(timeout=10)
        except Exception:
            try:
                active_trial.process.kill()
            except Exception:
                pass
        try:
            active_trial.log_handle.flush()
            active_trial.log_handle.close()
        except Exception:
            pass


def _sorted_results(results: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    successful = [r for r in results if r.get("status") == "completed" and r.get("objective") is not None]
    failed = [r for r in results if r.get("status") != "completed" or r.get("objective") is None]
    successful.sort(key=lambda item: float(item["objective"]))
    failed.sort(key=lambda item: (item.get("trial_index") or 0))
    return successful, failed


def _format_metric(value: Any, precision: int = 4, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def _flatten_result_row(
    result: Dict[str, Any],
    rank_lookup: Dict[str, int],
    parameter_names: Sequence[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "rank": rank_lookup.get(result["parameter_signature"], ""),
        "trial_index": result.get("trial_index"),
        "trial_name": result.get("trial_name"),
        "status": result.get("status"),
        "proposal_source": result.get("proposal_source"),
        "metric_value": result.get("metric_value"),
        "objective": result.get("objective"),
        "best_torque_score": result.get("best_torque_score"),
        "best_val_loss": result.get("best_val_loss"),
        "best_model_epoch": result.get("best_model_epoch"),
        "final_val_total_loss": result.get("final_val_total_loss"),
        "final_val_torque_rmse_Nm": result.get("final_val_torque_rmse_Nm"),
        "final_val_cop_rmse_m": result.get("final_val_cop_rmse_m"),
        "final_val_grf_rmse_N": result.get("final_val_grf_rmse_N"),
        "final_val_moments_rmse_Nm": result.get("final_val_moments_rmse_Nm"),
        "final_weighted_torque_score_Nm": result.get("final_weighted_torque_score_Nm"),
        "duration_s": result.get("duration_s"),
        "returncode": result.get("returncode"),
        "output_dir": result.get("output_dir"),
        "summary_path": result.get("summary_path"),
        "log_path": result.get("log_path"),
    }
    params = result.get("parameters", {})
    for name in parameter_names:
        row[name] = params.get(name)
    resolved_params = result.get("resolved_parameters", {})
    for name, value in resolved_params.items():
        if name not in params:
            row[f"resolved_{name}"] = value
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_summary_from_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary_path = result.get("summary_path")
    if not summary_path:
        return None
    path = Path(summary_path)
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _format_named_stats(metric_map: Optional[Dict[str, Any]], precision: int = 2, suffix: str = "") -> str:
    if not metric_map:
        return "n/a"
    pieces: List[str] = []
    for key in sorted(metric_map.keys()):
        pieces.append(f"{key}: {_format_metric(metric_map.get(key), precision=precision, suffix=suffix)}")
    return " | ".join(pieces)


def _top5_aggregate_stats(top_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not top_results:
        return {}
    metric_values = [float(r["metric_value"]) for r in top_results if r.get("metric_value") is not None]
    final_tau = [float(r["final_val_torque_rmse_Nm"]) for r in top_results if r.get("final_val_torque_rmse_Nm") is not None]
    d_model_counts = Counter(str(r.get("parameters", {}).get("d_model", "n/a")) for r in top_results)
    num_layer_counts = Counter(str(r.get("parameters", {}).get("num_layers", "n/a")) for r in top_results)
    return _json_compatible(
        {
            "count": len(top_results),
            "metric_mean": float(np.mean(metric_values)) if metric_values else None,
            "metric_std": float(np.std(metric_values)) if metric_values else None,
            "metric_min": float(np.min(metric_values)) if metric_values else None,
            "metric_max": float(np.max(metric_values)) if metric_values else None,
            "final_torque_rmse_mean": float(np.mean(final_tau)) if final_tau else None,
            "final_torque_rmse_std": float(np.std(final_tau)) if final_tau else None,
            "d_model_counts": dict(d_model_counts),
            "num_layers_counts": dict(num_layer_counts),
        }
    )


def _copy_if_exists(src: Optional[str], dst: Path) -> None:
    if not src:
        return
    src_path = Path(src)
    if not src_path.exists():
        return
    shutil.copy2(src_path, dst)


def _refresh_condensed_top_models(summary_dir: Path, top_results: Sequence[Dict[str, Any]]) -> None:
    top_root = summary_dir / "top_models"
    if top_root.exists():
        shutil.rmtree(top_root)
    top_root.mkdir(parents=True, exist_ok=True)

    for rank, result in enumerate(top_results, start=1):
        trial_dir = top_root / f"top_{rank:02d}_{_slugify(result['trial_name'], max_len=72)}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        summary = _load_summary_from_result(result) or {}
        artifacts = summary.get("artifacts") or {}

        _copy_if_exists(result.get("summary_path"), trial_dir / "training_summary.json")
        _copy_if_exists(result.get("hyperparameters_path"), trial_dir / "hyperparameters.json")
        _copy_if_exists(artifacts.get("model_parameters_yaml_path"), trial_dir / "model_parameters.yaml")
        _copy_if_exists(artifacts.get("final_prediction_plot"), trial_dir / "prediction_final.png")
        _copy_if_exists(artifacts.get("loss_history_plot"), trial_dir / "loss_history.png")
        _copy_if_exists(result.get("log_path"), trial_dir / "trial.log")

        _write_json(
            trial_dir / "artifact_paths.json",
            {
                "rank": rank,
                "trial_name": result.get("trial_name"),
                "output_dir": result.get("output_dir"),
                "summary_path": result.get("summary_path"),
                "hyperparameters_path": result.get("hyperparameters_path"),
                "log_path": result.get("log_path"),
                "artifacts": artifacts,
            },
        )


def _write_markdown_summary(
    path: Path,
    successful: Sequence[Dict[str, Any]],
    failed: Sequence[Dict[str, Any]],
    parameter_specs: Sequence[ParameterSpec],
    metric_name: str,
    metric_goal: str,
    sweep_path: Path,
    search_root: Path,
) -> None:
    top_results = list(successful[:5])
    aggregate = _top5_aggregate_stats(top_results)

    lines: List[str] = []
    lines.append("# Local HPO Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Sweep file: `{sweep_path}`")
    lines.append(f"- Search root: `{search_root}`")
    lines.append(f"- Metric: `{metric_name}` ({metric_goal})")
    lines.append(f"- Completed successful trials: {len(successful)}")
    lines.append(f"- Failed trials: {len(failed)}")
    lines.append("")
    lines.append("## Search Space")
    lines.append("")
    for spec in parameter_specs:
        lines.append(f"- `{spec.name}`: {spec.summary()}")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | Trial | Metric | Best Epoch | Final Torque RMSE (Nm) | d_model | num_layers | learning_rate | dropout_rate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, result in enumerate(successful, start=1):
        params = result.get("parameters", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    result.get("trial_name", "n/a"),
                    _format_metric(result.get("metric_value"), precision=5),
                    str(result.get("best_model_epoch", "n/a")),
                    _format_metric(result.get("final_val_torque_rmse_Nm"), precision=4),
                    str(params.get("d_model", "n/a")),
                    str(params.get("num_layers", "n/a")),
                    _format_metric(params.get("learning_rate"), precision=6),
                    _format_metric(params.get("dropout_rate"), precision=4),
                ]
            )
            + " |"
        )
    if failed:
        lines.append("")
        lines.append("## Failed Trials")
        lines.append("")
        lines.append("| Trial | Return Code | Reason |")
        lines.append("| --- | ---: | --- |")
        for result in failed:
            reason = result.get("failure_reason") or "training failed"
            lines.append(
                f"| {result.get('trial_name', 'n/a')} | {result.get('returncode', 'n/a')} | {str(reason).replace('|', '/')} |"
            )

    lines.append("")
    lines.append("## Top 5 Aggregate Stats")
    lines.append("")
    if aggregate:
        lines.append(f"- Mean metric: {_format_metric(aggregate.get('metric_mean'), precision=5)}")
        lines.append(f"- Metric std: {_format_metric(aggregate.get('metric_std'), precision=5)}")
        lines.append(f"- Final torque RMSE mean: {_format_metric(aggregate.get('final_torque_rmse_mean'), precision=4)} Nm")
        lines.append(f"- Final torque RMSE std: {_format_metric(aggregate.get('final_torque_rmse_std'), precision=4)} Nm")
        lines.append(
            "- d_model counts: "
            + ", ".join(f"{k}x{v}" for k, v in sorted((aggregate.get("d_model_counts") or {}).items()))
        )
        lines.append(
            "- num_layers counts: "
            + ", ".join(f"{k}x{v}" for k, v in sorted((aggregate.get("num_layers_counts") or {}).items()))
        )
    else:
        lines.append("- No completed trials yet.")

    lines.append("")
    lines.append("## Top 5 Detailed Stats")
    lines.append("")
    if not top_results:
        lines.append("No completed trials yet.")
    for rank, result in enumerate(top_results, start=1):
        summary = _load_summary_from_result(result) or {}
        best_epoch = summary.get("best_epoch_metrics") or {}
        best_val = best_epoch.get("val") or {}
        final_epoch = summary.get("final_epoch_metrics") or {}
        final_val = final_epoch.get("val") or {}
        params = result.get("parameters", {})
        resolved_params = result.get("resolved_parameters", {})
        display_params = dict(params)
        for key, value in resolved_params.items():
            if key not in display_params:
                display_params[key] = value
        lines.append(f"### {rank}. {result.get('trial_name', 'n/a')}")
        lines.append("")
        lines.append(f"- Metric value: {_format_metric(result.get('metric_value'), precision=5)}")
        lines.append(f"- Best model epoch: {result.get('best_model_epoch', 'n/a')}")
        lines.append(f"- Best torque score: {_format_metric(result.get('best_torque_score'), precision=5)}")
        lines.append(f"- Final val torque RMSE: {_format_metric(result.get('final_val_torque_rmse_Nm'), precision=4)} Nm")
        lines.append(f"- Final val total loss: {_format_metric(result.get('final_val_total_loss'), precision=5)}")
        lines.append(f"- Output dir: `{result.get('output_dir', '')}`")
        lines.append(f"- Final prediction plot: `{result.get('final_prediction_plot', '')}`")
        lines.append(
            "- Hyperparameters: "
            + ", ".join(f"{key}={_format_value(display_params.get(key))}" for key in sorted(display_params.keys()))
        )
        lines.append(
            "- Best epoch overall metrics: "
            + " | ".join(
                [
                    f"COP {_format_metric(best_val.get('cop_rmse_fullset_overall_m'), precision=4)} m",
                    f"GRF {_format_metric(best_val.get('grf_rmse_fullset_overall_N'), precision=3)} N",
                    f"Moments {_format_metric(best_val.get('moments_rmse_fullset_overall_Nm'), precision=3)} Nm",
                    f"Torque {_format_metric(best_val.get('torque_rmse_fullset_overall_Nm'), precision=3)} Nm",
                ]
            )
        )
        lines.append(
            "- Best epoch bilateral torque RMSE: "
            + _format_named_stats(best_val.get("torque_rmse_bilateral_Nm"), precision=3, suffix=" Nm")
        )
        lines.append(
            "- Best epoch stance torque MAE% bilateral: "
            + _format_named_stats(best_val.get("torque_mae_percent_bilateral_stance"), precision=2, suffix="%")
        )
        lines.append(
            "- Best epoch GRF MAE%BW bilateral: "
            + _format_named_stats(best_val.get("grf_mae_percent_bw_bilateral"), precision=2, suffix="%")
        )
        lines.append(
            "- Final epoch bilateral torque RMSE: "
            + _format_named_stats(final_val.get("torque_rmse_bilateral_Nm"), precision=3, suffix=" Nm")
        )
        lines.append(
            "- Final epoch stance torque MAE% bilateral: "
            + _format_named_stats(final_val.get("torque_mae_percent_bilateral_stance"), precision=2, suffix="%")
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_search_outputs(
    search_root: Path,
    manifest: Dict[str, Any],
    parameter_specs: Sequence[ParameterSpec],
    metric_name: str,
    metric_goal: str,
    sweep_path: Path,
) -> None:
    summary_dir = search_root / SUMMARY_DIR_NAME
    summary_dir.mkdir(parents=True, exist_ok=True)

    results = manifest.get("results", [])
    successful, failed = _sorted_results(results)
    rank_lookup = {result["parameter_signature"]: rank for rank, result in enumerate(successful, start=1)}
    parameter_names = [spec.name for spec in parameter_specs]

    flat_rows = [
        _flatten_result_row(result, rank_lookup=rank_lookup, parameter_names=parameter_names)
        for result in successful + failed
    ]

    _write_json(search_root / "search_state.json", manifest)
    _write_json(summary_dir / "hpo_results.json", {"results": successful + failed})
    _write_csv(summary_dir / "hpo_results.csv", flat_rows)

    top5_details = []
    for rank, result in enumerate(successful[:5], start=1):
        summary = _load_summary_from_result(result)
        top5_details.append(
            {
                "rank": rank,
                "trial_name": result.get("trial_name"),
                "metric_value": result.get("metric_value"),
                "output_dir": result.get("output_dir"),
                "parameters": result.get("parameters"),
                "summary": summary,
            }
        )
    _write_json(
        summary_dir / "top5_statistics.json",
        {
            "metric_name": metric_name,
            "metric_goal": metric_goal,
            "aggregate": _top5_aggregate_stats(successful[:5]),
            "top_models": top5_details,
        },
    )

    _write_markdown_summary(
        summary_dir / "hpo_summary.md",
        successful=successful,
        failed=failed,
        parameter_specs=parameter_specs,
        metric_name=metric_name,
        metric_goal=metric_goal,
        sweep_path=sweep_path,
        search_root=search_root,
    )
    _refresh_condensed_top_models(summary_dir, successful[:5])


def _dry_run_preview(
    program_path: Path,
    base_args: Sequence[str],
    base_exp_name: str,
    specs: Sequence[ParameterSpec],
    epochs: int,
    sweep_payload: Dict[str, Any],
    search_root: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    params = {spec.name: spec.sample(rng) for spec in specs}
    run_name = _build_trial_name(base_exp_name, 1, params)
    run_dir = search_root / "runs" / run_name
    command = _build_trial_command(program_path, base_args, run_dir, run_name, params, epochs)

    preview = {
        "program": str(program_path),
        "search_root": str(search_root),
        "metric": sweep_payload.get("metric", {}),
        "run_cap": sweep_payload.get("run_cap"),
        "sampled_trial_name": run_name,
        "sampled_parameters": params,
        "command": command,
    }
    print(json.dumps(_json_compatible(preview), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local HPO search defined by a sweep YAML.")
    parser.add_argument(
        "--sweep_file",
        type=str,
        default=str(DEFAULT_SWEEP_FILE),
        help="Path to the sweep YAML to mirror locally.",
    )
    parser.add_argument(
        "--search_name",
        type=str,
        default=None,
        help="Optional subfolder name created inside the sweep output_dir.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional override for the sweep data_dir passed through to train.py.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional root directory for local HPO outputs. Defaults to the sweep output_dir or a local fallback.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Optional override for the sweep exp_name passed through to train.py.",
    )
    parser.add_argument(
        "--run_cap",
        type=int,
        default=None,
        help="Override the sweep run_cap value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for local search proposal generation.",
    )
    parser.add_argument(
        "--startup_trials",
        type=int,
        default=None,
        help="Number of random warmup trials before the GP-based proposer takes over.",
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=2048,
        help="Number of random candidates scored by expected improvement on each Bayesian step.",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=2,
        help="Maximum number of local training workers to run at once. Defaults to 2 for conservative scheduling.",
    )
    parser.add_argument(
        "--poll_interval_s",
        type=float,
        default=10.0,
        help="How often to poll active workers for completion when no trial has finished yet.",
    )
    parser.add_argument(
        "--jax_gpu_mem_fraction",
        type=float,
        default=0.35,
        help="Per-worker JAX GPU memory cap, matching the conservative multi-agent setting from train_single_model.py.",
    )
    parser.add_argument(
        "--jax_cpu_threads",
        type=int,
        default=1,
        help="Per-worker BLAS/OpenMP thread cap, matching the conservative multi-agent setting from train_single_model.py.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Optional comma-separated GPU list. When provided, workers are assigned devices round-robin.",
    )
    parser.add_argument(
        "--disable_jax_multi_agent_safe",
        action="store_true",
        help="Disable the conservative multi-agent JAX environment overrides.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse the sweep and print one sampled local command without launching training.",
    )
    parser.add_argument(
        "--resume_search",
        action="store_true",
        help="Resume an existing --search_name folder by loading search_state.json and continuing after finalized trials.",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep_file).resolve()
    sweep_payload = _read_sweep_config(sweep_path)
    parameter_specs = _parse_parameter_specs(sweep_payload)

    metric_cfg = sweep_payload.get("metric", {}) or {}
    metric_name = str(metric_cfg.get("name", "val/best_torque_score_Nm"))
    metric_goal = str(metric_cfg.get("goal", "minimize")).lower()
    if metric_goal not in {"minimize", "maximize"}:
        raise ValueError(f"Unsupported metric goal: {metric_goal}")

    program_path = _resolve_program_path(str(sweep_payload["program"]))
    command_items = sweep_payload.get("command", [])
    original_base_args = _normalize_command_list(command_items)
    original_data_dir = _get_arg_value(original_base_args, "data_dir")
    original_output_dir = _get_arg_value(original_base_args, "output_dir")
    original_exp_name = _get_arg_value(original_base_args, "exp_name")
    if args.data_dir is not None:
        resolved_data_dir = Path(args.data_dir).expanduser().resolve()
    elif original_data_dir:
        resolved_data_dir = Path(original_data_dir).expanduser().resolve()
    else:
        resolved_data_dir = None
    resolved_output_root = None if args.output_root is None else Path(args.output_root).expanduser().resolve()
    resolved_exp_name = args.exp_name or original_exp_name or DEFAULT_EXP_NAME

    if resolved_output_root is None:
        if original_output_dir:
            resolved_output_root = Path(original_output_dir).expanduser().resolve()
        else:
            resolved_output_root = DEFAULT_OUTPUT_ROOT.resolve()

    base_args = _apply_command_overrides(
        original_base_args,
        data_dir=resolved_data_dir,
        output_dir=resolved_output_root,
        exp_name=resolved_exp_name,
    )

    base_output_dir = resolved_output_root
    base_output_dir.mkdir(parents=True, exist_ok=True)

    base_exp_name = resolved_exp_name
    epochs_value = _get_arg_value(base_args, "epochs")
    epochs = int(epochs_value) if epochs_value is not None else 1

    run_cap = int(args.run_cap if args.run_cap is not None else sweep_payload.get("run_cap", 1))
    startup_trials = int(args.startup_trials if args.startup_trials is not None else max(24, 2 * len(parameter_specs)))
    max_parallel = max(1, int(args.max_parallel))
    jax_multi_agent_safe = not bool(args.disable_jax_multi_agent_safe)
    cuda_visible_devices = _parse_cuda_visible_devices_arg(args.cuda_visible_devices)
    search_name = args.search_name or f"local_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    search_root = base_output_dir / _slugify(search_name, max_len=80)
    resume_mode = bool(args.resume_search)
    if search_root.exists() and any(search_root.iterdir()) and not resume_mode:
        raise FileExistsError(
            f"Search directory already exists and is not empty: {search_root}\n"
            "Use --search_name with a fresh folder name, or pass --resume_search to continue it."
        )
    search_root.mkdir(parents=True, exist_ok=True)
    (search_root / "runs").mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _dry_run_preview(
            program_path=program_path,
            base_args=base_args,
            base_exp_name=base_exp_name,
            specs=parameter_specs,
            epochs=epochs,
            sweep_payload=sweep_payload,
            search_root=search_root,
            seed=args.seed,
        )
        return

    rng = np.random.default_rng(args.seed)
    if resume_mode:
        state_path = search_root / "search_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing search state: {state_path}")
        manifest = _load_json(state_path)
        manifest.setdefault("results", [])
        stale_active = manifest.get("active_trials", [])
        if stale_active:
            print(
                f"⚠️  Ignoring {len(stale_active)} stale active trial record(s) from the previous run.",
                flush=True,
            )
        manifest["active_trials"] = []
        manifest["resumed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["run_cap"] = run_cap
        manifest["startup_trials"] = startup_trials
        manifest["candidate_pool_size"] = int(args.candidate_pool_size)
        manifest["max_parallel"] = max_parallel
        manifest["poll_interval_s"] = float(args.poll_interval_s)
        manifest["jax_multi_agent_safe"] = bool(jax_multi_agent_safe)
        manifest["jax_gpu_mem_fraction"] = None if not jax_multi_agent_safe else float(args.jax_gpu_mem_fraction)
        manifest["jax_cpu_threads"] = int(args.jax_cpu_threads)
        manifest["cuda_visible_devices"] = cuda_visible_devices
        manifest["base_args"] = base_args
    else:
        manifest: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "search_root": str(search_root),
            "sweep_file": str(sweep_path),
            "program": str(program_path),
            "resolved_data_dir": _get_arg_value(base_args, "data_dir"),
            "original_sweep_data_dir": original_data_dir,
            "original_sweep_output_dir": original_output_dir,
            "original_sweep_exp_name": original_exp_name,
            "resolved_output_root": str(base_output_dir),
            "resolved_exp_name": base_exp_name,
            "metric_name": metric_name,
            "metric_goal": metric_goal,
            "run_cap": run_cap,
            "startup_trials": startup_trials,
            "candidate_pool_size": int(args.candidate_pool_size),
            "max_parallel": max_parallel,
            "poll_interval_s": float(args.poll_interval_s),
            "jax_multi_agent_safe": bool(jax_multi_agent_safe),
            "jax_gpu_mem_fraction": None if not jax_multi_agent_safe else float(args.jax_gpu_mem_fraction),
            "jax_cpu_threads": int(args.jax_cpu_threads),
            "cuda_visible_devices": cuda_visible_devices,
            "base_args": base_args,
            "active_trials": [],
            "results": [],
        }
    _write_json(search_root / "manifest.json", manifest)

    print(f"🔎 Local HPO search root: {search_root}", flush=True)
    print(f"📂 Dataset: {_get_arg_value(base_args, 'data_dir')}", flush=True)
    print(f"📈 Optimizing {metric_name} ({metric_goal}) for {run_cap} trials", flush=True)
    print(
        f"🧪 Warmup trials: {startup_trials} | Candidate pool: {args.candidate_pool_size} | "
        f"Workers: {max_parallel}",
        flush=True,
    )
    if jax_multi_agent_safe:
        print(
            f"🛡️  Conservative JAX mode: gpu_mem_fraction={args.jax_gpu_mem_fraction}, "
            f"cpu_threads={args.jax_cpu_threads}",
            flush=True,
        )
    if cuda_visible_devices:
        print(f"🎛️  Worker GPU assignment pool: {', '.join(cuda_visible_devices)}", flush=True)

    active_trials: List[ActiveTrial] = []
    finalized_indices = [
        int(result.get("trial_index", 0))
        for result in manifest.get("results", [])
        if result.get("trial_index") is not None
    ]
    next_trial_index = (max(finalized_indices) + 1) if finalized_indices else 1
    if resume_mode:
        print(
            f"↩️  Resuming from trial {next_trial_index}; finalized trials: {len(manifest.get('results', []))}",
            flush=True,
        )

    try:
        while next_trial_index <= run_cap or active_trials:
            while next_trial_index <= run_cap and len(active_trials) < max_parallel:
                pending_signatures = {trial.base_payload["parameter_signature"] for trial in active_trials}
                params, proposal_source = _propose_next_config(
                    specs=parameter_specs,
                    completed_results=manifest["results"],
                    rng=rng,
                    startup_trials=startup_trials,
                    candidate_pool_size=int(args.candidate_pool_size),
                    extra_excluded_signatures=pending_signatures,
                )
                worker_slot = _select_worker_slot(active_trials, max_parallel=max_parallel)
                active_trial = _launch_trial(
                    trial_index=next_trial_index,
                    program_path=program_path,
                    base_args=base_args,
                    base_exp_name=base_exp_name,
                    params=params,
                    epochs=epochs,
                    search_root=search_root,
                    proposal_source=proposal_source,
                    worker_slot=worker_slot,
                    jax_multi_agent_safe=jax_multi_agent_safe,
                    jax_gpu_mem_fraction=float(args.jax_gpu_mem_fraction),
                    jax_cpu_threads=int(args.jax_cpu_threads),
                    cuda_visible_devices=cuda_visible_devices,
                )
                active_trials.append(active_trial)
                next_trial_index += 1
                manifest["active_trials"] = [_json_compatible(trial.base_payload) for trial in active_trials]
                manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
                _write_json(search_root / "search_state.json", manifest)

            finished_this_round = False
            for active_trial in list(active_trials):
                if active_trial.process.poll() is None:
                    continue
                result = _finalize_trial(
                    active_trial=active_trial,
                    metric_name=metric_name,
                    metric_goal=metric_goal,
                )
                active_trials.remove(active_trial)
                manifest["results"].append(result)
                manifest["active_trials"] = [_json_compatible(trial.base_payload) for trial in active_trials]
                manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
                _write_search_outputs(
                    search_root=search_root,
                    manifest=manifest,
                    parameter_specs=parameter_specs,
                    metric_name=metric_name,
                    metric_goal=metric_goal,
                    sweep_path=sweep_path,
                )
                finished_this_round = True

            if not finished_this_round and active_trials:
                time.sleep(max(0.5, float(args.poll_interval_s)))
    except KeyboardInterrupt:
        print("\n🛑 Interrupt received. Stopping active workers...", flush=True)
        _terminate_active_trials(active_trials)
        manifest["active_trials"] = []
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_search_outputs(
            search_root=search_root,
            manifest=manifest,
            parameter_specs=parameter_specs,
            metric_name=metric_name,
            metric_goal=metric_goal,
            sweep_path=sweep_path,
        )
        raise

    successful, failed = _sorted_results(manifest["results"])
    print("\n✅ Local HPO search complete.", flush=True)
    print(f"   Successful trials: {len(successful)}", flush=True)
    print(f"   Failed trials: {len(failed)}", flush=True)
    if successful:
        best = successful[0]
        print(
            f"   Best {metric_name}: {_format_metric(best.get('metric_value'), precision=5)} "
            f"from {best.get('trial_name')}",
            flush=True,
        )
    print(f"   Summary folder: {search_root / SUMMARY_DIR_NAME}", flush=True)


if __name__ == "__main__":
    main()
