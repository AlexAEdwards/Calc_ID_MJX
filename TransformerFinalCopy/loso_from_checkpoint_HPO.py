"""Nested LOSO hyperparameter search for checkpoint fine-tuning.

This script is intentionally metric-only: it avoids the slower infer.py plotting
and comparison pipeline while tuning learning rate and epoch count.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Dict, IO, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
import numpy as np

try:
    from wandb_utils import configure_runtime_env
except ModuleNotFoundError:
    def configure_runtime_env():
        return {}

RUNTIME_ENV_APPLIED = configure_runtime_env()

import train as train_module
import loso_adapters
import loso_from_checkpoint as base


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


LOSO_CONFIG: Dict[str, Any] = dict(base.LOSO_CONFIG)
LOSO_CONFIG["output_dir"] = str(PROJECT_ROOT / "inference_results" / "Nested_LOSO_HPO_LR_Epoch")
# Ground truth for the nested HPO is the recalculated OpenSim ID. This flag flows
# through base._apply_cli_overrides -> base._safe_trial_loader for BOTH the training
# and validation loaders, so the data loader builds the torque regression target as
#     qfrc_grf_contribution = qfrc_inverse(ProcessedData) - OpenSim_ID_recalculated
# and sets the full-ID GT to OpenSim_ID_recalculated. The full-ID reconstruction then
# recovers OpenSim_ID_recalculated = qfrc_inverse - tau_grf (full_id_tau_sign = -1),
# matching the sign convention in data_loader.py and infer.compute_full_id_curves.
LOSO_CONFIG["use_recalculated_opensim_id_gt"] = True
LOSO_CONFIG["use_OpenSimID_GT"] = True


HPO_CONFIG: Dict[str, Any] = {
    "trials_per_fold": 20,
    "hpo_seed": 123,
    "max_parallel_splits": 3,
    "objective": "selected_left_stance_moment_mae_percent_bwh_mean",
    "selection_std_weight": 0.25,
    "grid": {
        "epochs": [2, 4, 7, 10, 15],
        "learning_rate": [3e-5, 7e-5, 1.5e-4, 3e-4],
    },
}


def _timer(
    label: str,
    start_time: float,
    *,
    emit: bool = True,
    log_handle: Optional[IO[str]] = None,
) -> float:
    elapsed = time.time() - start_time
    line = f"[TIMER] {label}: {base._format_duration(elapsed)}"
    if emit:
        print(line, flush=True)
    if log_handle is not None:
        print(line, file=log_handle, flush=True)
    return elapsed


def _log_line(message: str, log_handle: Optional[IO[str]] = None, *, emit: bool = True) -> None:
    if emit:
        print(message, flush=True)
    if log_handle is not None:
        print(message, file=log_handle, flush=True)


def _print_split_summary(split_name: str, payload: Mapping[str, Any]) -> None:
    timings = payload.get("timings_s", {}) if isinstance(payload.get("timings_s"), Mapping) else {}
    timing_parts = []
    for key in ("resolve_input_config", "build_loaders_and_first_batch", "build_model_and_train_state", "training", "metric_only_eval"):
        value = timings.get(key)
        if value is not None:
            timing_parts.append(f"{key}={base._format_duration(float(value))}")
    timing_text = ", ".join(timing_parts)
    if timing_text:
        timing_text = f" | {timing_text}"
    print(
        f"[SUMMARY] {split_name}: objective={float(payload['objective']):.6f} "
        f"total={base._format_duration(float(payload['duration_s']))}{timing_text}",
        flush=True,
    )


def _objective_from_metrics(metrics: Mapping[str, Any]) -> float:
    return base._finite_float(metrics.get(HPO_CONFIG["objective"]))


def _flatten_for_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, (list, tuple)):
            flat[key] = json.dumps(base._coerce_jsonable(value), sort_keys=True)
        else:
            flat[key] = value
    return flat


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [_flatten_for_csv(row) for row in rows]
    if not flat_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in flat_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def _save_json_any(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base._coerce_jsonable(payload), indent=2), encoding="utf-8")


def _write_hpo_progress_snapshot(
    output_root: Path,
    fold_records: Sequence[Mapping[str, Any]],
    *,
    completed_candidates: int,
    total_candidates: int,
) -> None:
    progress_rows: List[Dict[str, Any]] = []
    progress_payload: Dict[str, Any] = {
        "completed_candidates": int(completed_candidates),
        "total_candidates": int(total_candidates),
        "folds": [],
    }
    for record in fold_records:
        fold_dir = Path(record["fold_dir"])
        candidate_rows = list(record.get("candidate_rows", []))
        candidate_payloads = list(record.get("candidate_payloads", []))
        _save_json_any(fold_dir / "inner_hpo_results_partial.json", candidate_payloads)
        _write_csv(fold_dir / "inner_hpo_results_partial.csv", candidate_rows)
        progress_payload["folds"].append(
            {
                "held_out_subject": str(record["test_subject"]),
                "fold_dir": str(fold_dir),
                "candidate_count": len(candidate_rows),
                "candidate_rows": candidate_rows,
            }
        )
        for row in candidate_rows:
            progress_rows.append(dict(row))
    _save_json_any(output_root / "loso_nested_hpo_progress.json", progress_payload)
    _write_csv(output_root / "loso_nested_hpo_progress.csv", progress_rows)


def _build_hpo_grid(grid_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(grid_payload, list):
        candidates = [dict(item) for item in grid_payload]
    elif isinstance(grid_payload, Mapping):
        epochs = list(grid_payload.get("epochs", []))
        learning_rates = list(grid_payload.get("learning_rate", []))
        if not epochs or not learning_rates:
            raise ValueError("HPO grid must include non-empty 'epochs' and 'learning_rate' lists.")
        candidates = [
            {"epochs": int(epoch), "learning_rate": float(learning_rate)}
            for epoch, learning_rate in product(epochs, learning_rates)
        ]
    else:
        raise ValueError("--hpo_grid_json must decode to an object or list of candidate objects.")
    if not candidates:
        raise ValueError("HPO grid produced zero candidates.")
    for candidate in candidates:
        if "epochs" not in candidate or "learning_rate" not in candidate:
            raise ValueError(f"Each HPO candidate must include epochs and learning_rate: {candidate}")
        candidate["epochs"] = int(candidate["epochs"])
        candidate["learning_rate"] = float(candidate["learning_rate"])
    return candidates


def _parse_epoch_targets(value: Optional[str]) -> List[int]:
    if value is None or not str(value).strip():
        return []
    cleaned = str(value).strip()
    if cleaned.startswith("["):
        payload = json.loads(cleaned)
        return [int(item) for item in payload]
    return [int(part.strip()) for part in cleaned.split(",") if part.strip()]


def _discover_epoch_checkpoints(checkpoint_dir: Path) -> Dict[int, Path]:
    discovered: Dict[int, Path] = {}
    for path in checkpoint_dir.glob("model_epoch_*.pkl"):
        stem = path.stem
        try:
            epoch = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        discovered[epoch] = path.resolve()
    return dict(sorted(discovered.items()))


def _resolve_checkpoint_candidates(
    checkpoint_path: Path,
    epoch_targets: Sequence[int],
) -> List[Dict[str, Any]]:
    if not epoch_targets:
        return [
            {
                "label": "source_checkpoint",
                "path": str(checkpoint_path.resolve()),
                "checkpoint_epoch": None,
                "requested_epoch_targets": [],
                "distance_from_requested_epoch": None,
            }
        ]

    epoch_checkpoints = _discover_epoch_checkpoints(checkpoint_path.parent)
    if not epoch_checkpoints:
        raise FileNotFoundError(
            f"No model_epoch_*.pkl files were found next to source checkpoint: {checkpoint_path.parent}"
        )

    by_path: Dict[str, Dict[str, Any]] = {}
    for requested_epoch in epoch_targets:
        closest_epoch = min(epoch_checkpoints, key=lambda epoch: (abs(epoch - int(requested_epoch)), epoch))
        path = epoch_checkpoints[closest_epoch]
        path_key = str(path)
        distance = abs(int(closest_epoch) - int(requested_epoch))
        if path_key not in by_path:
            by_path[path_key] = {
                "label": f"model_epoch_{closest_epoch:04d}",
                "path": path_key,
                "checkpoint_epoch": int(closest_epoch),
                "requested_epoch_targets": [],
                "distance_from_requested_epoch": int(distance),
            }
        by_path[path_key]["requested_epoch_targets"].append(int(requested_epoch))
        by_path[path_key]["distance_from_requested_epoch"] = min(
            int(by_path[path_key]["distance_from_requested_epoch"]),
            int(distance),
        )

    return sorted(
        by_path.values(),
        key=lambda item: (
            int(item["checkpoint_epoch"]) if item["checkpoint_epoch"] is not None else -1,
            str(item["path"]),
        ),
    )


def _resolve_checkpoint_candidates_from_json(value: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if value is None or not str(value).strip():
        return None
    raw_value = str(value).strip()
    if raw_value.startswith("["):
        payload = json.loads(raw_value)
    else:
        json_path = Path(raw_value).expanduser().resolve()
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    if isinstance(payload, Mapping):
        if isinstance(payload.get("checkpoint_candidates"), list):
            payload = payload["checkpoint_candidates"]
        elif isinstance(payload.get("candidates"), list):
            payload = payload["candidates"]
    if not isinstance(payload, list):
        raise ValueError("--checkpoint_candidates_json must be a JSON list or an object with checkpoint_candidates/candidates.")

    candidates: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(payload):
        if isinstance(item, str):
            record: Dict[str, Any] = {"path": item}
        elif isinstance(item, Mapping):
            record = dict(item)
        else:
            raise ValueError(f"Checkpoint candidate {index} must be a path string or object.")
        if not record.get("path"):
            raise ValueError(f"Checkpoint candidate {index} is missing required field 'path'.")
        path = Path(str(record["path"])).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint candidate does not exist: {path}")
        path_key = str(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        label = str(record.get("label") or f"checkpoint_{len(candidates):02d}_{path.parent.name}")
        candidates.append(
            {
                **record,
                "label": label,
                "path": path_key,
                "checkpoint_epoch": record.get("checkpoint_epoch"),
                "requested_epoch_targets": list(record.get("requested_epoch_targets", [])),
                "distance_from_requested_epoch": record.get("distance_from_requested_epoch"),
            }
        )
    if not candidates:
        raise ValueError("--checkpoint_candidates_json did not contain any usable checkpoint candidates.")
    return candidates


def _attach_checkpoint_candidates_to_grid(
    candidates: Sequence[Mapping[str, Any]],
    checkpoint_candidates: Sequence[Mapping[str, Any]],
    trials_per_fold: int,
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for candidate, checkpoint_candidate in product(candidates, checkpoint_candidates):
        merged = dict(candidate)
        merged["source_checkpoint_label"] = str(checkpoint_candidate["label"])
        merged["source_checkpoint_path"] = str(checkpoint_candidate["path"])
        merged["source_checkpoint_epoch"] = checkpoint_candidate.get("checkpoint_epoch")
        merged["source_checkpoint_requested_epoch_targets"] = list(
            checkpoint_candidate.get("requested_epoch_targets", [])
        )
        merged["source_checkpoint_epoch_target_distance"] = checkpoint_candidate.get(
            "distance_from_requested_epoch"
        )
        expanded.append(merged)
    return expanded[: int(trials_per_fold)]


def _build_nested_loso_folds(
    valid_subjects: Sequence[str],
    subject_to_trials: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    subjects = list(sorted(valid_subjects, key=base._subject_sort_key))
    if len(subjects) < 3:
        raise ValueError("Nested LOSO HPO requires at least 3 valid subjects.")
    folds: List[Dict[str, Any]] = []
    for test_subject in subjects:
        inner_subjects = [subject for subject in subjects if subject != test_subject]
        inner_folds = []
        for inner_val_subject in inner_subjects:
            train_subjects = [
                subject for subject in subjects if subject not in {test_subject, inner_val_subject}
            ]
            inner_folds.append(
                {
                    "inner_val_subject": inner_val_subject,
                    "train_subjects": train_subjects,
                    "train_trials": [trial for subject in train_subjects for trial in subject_to_trials[subject]],
                    "eval_trials": list(subject_to_trials[inner_val_subject]),
                }
            )
        folds.append(
            {
                "held_out_subject": test_subject,
                "inner_folds": inner_folds,
                "final_train_subjects": inner_subjects,
                "final_train_trials": [trial for subject in inner_subjects for trial in subject_to_trials[subject]],
                "held_out_trials": list(subject_to_trials[test_subject]),
            }
        )
    return folds


def _trial_subjects(trials: Sequence[Mapping[str, Any]]) -> List[str]:
    subjects = set()
    for trial in trials:
        subject = trial.get("subject_group") or trial.get("subject")
        if subject:
            subjects.add(str(subject))
    return sorted(subjects, key=base._subject_sort_key)


def _assert_split_is_leakage_safe(
    *,
    train_trials: Sequence[Mapping[str, Any]],
    eval_trials: Sequence[Mapping[str, Any]],
    forbidden_eval_subject: str,
    split_name: str,
) -> None:
    train_subjects = set(_trial_subjects(train_trials))
    eval_subjects = set(_trial_subjects(eval_trials))
    if forbidden_eval_subject in train_subjects:
        raise ValueError(f"Leakage in {split_name}: {forbidden_eval_subject} appears in training trials.")
    overlap = train_subjects & eval_subjects
    if overlap:
        raise ValueError(f"Leakage in {split_name}: train/eval subjects overlap: {sorted(overlap)}")


def _extract_subject_from_pathish(value: Any) -> Optional[str]:
    if value is None:
        return None
    parts = Path(str(value)).parts
    for part in reversed(parts):
        if part in base.OPEN_CAP_LOSO_SUBJECTS:
            return part
    for part in reversed(parts):
        if part.lower().startswith("subject"):
            return part
    return None


def _subjects_from_checkpoint_trials(checkpoint: Mapping[str, Any]) -> List[str]:
    subjects = set()
    containers = [
        checkpoint.get("train_trials"),
        checkpoint.get("val_trials"),
        checkpoint.get("metadata", {}).get("train_trials") if isinstance(checkpoint.get("metadata"), Mapping) else None,
        checkpoint.get("metadata", {}).get("val_trials") if isinstance(checkpoint.get("metadata"), Mapping) else None,
    ]
    for container in containers:
        if not isinstance(container, Sequence) or isinstance(container, (str, bytes)):
            continue
        for item in container:
            if isinstance(item, Mapping):
                subject = item.get("subject") or _extract_subject_from_pathish(
                    item.get("trial_dir") or item.get("path") or item.get("trial_path")
                )
            else:
                subject = _extract_subject_from_pathish(item)
            if subject:
                subjects.add(str(base.subject_group_id(subject)) if hasattr(base, "subject_group_id") else str(subject))
    return sorted(subjects, key=base._subject_sort_key)


def _checkpoint_overlap_report(
    checkpoint: Mapping[str, Any],
    valid_subjects: Sequence[str],
    *,
    allow_overlap: bool,
) -> Dict[str, Any]:
    checkpoint_subjects = _subjects_from_checkpoint_trials(checkpoint)
    overlap = sorted(set(checkpoint_subjects) & set(valid_subjects), key=base._subject_sort_key)
    report = {
        "checkpoint_subjects_found": checkpoint_subjects,
        "loso_subjects": list(valid_subjects),
        "overlap_subjects": overlap,
        "metadata_available": bool(checkpoint_subjects),
        "allow_overlap": bool(allow_overlap),
    }
    if overlap and not allow_overlap:
        raise ValueError(
            "Source checkpoint metadata indicates overlap with LOSO evaluation subjects: "
            f"{overlap}. Pass --allow_checkpoint_loso_overlap only if this is intentional."
        )
    return report


def _resolved_input_summary(fold_config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "deviation_learning": bool(fold_config["deviation_learning"]),
        "include_pelvis_euler": bool(fold_config["include_pelvis_euler"]),
        "include_ankle_heights": bool(fold_config.get("include_ankle_heights", True)),
        "include_jacobian_input": bool(fold_config.get("include_jacobian_input", True)),
        "input_dim": int(fold_config["input_dim"]),
        "static_dim": int(fold_config["static_dim"]),
        "layout_name": fold_config.get("resolved_input_layout"),
        "sample_trial": fold_config.get("resolved_input_sample_trial"),
    }


def _train_metric_only_split(
    *,
    split_name: str,
    split_dir: Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    train_trials: Sequence[Mapping[str, Any]],
    eval_trials: Sequence[Mapping[str, Any]],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    adapter_hidden_dim: int,
    adapter_dropout_rate: float,
    seed: int,
    save_model: bool,
    model_payload_extra: Optional[Mapping[str, Any]] = None,
    log_handle: Optional[IO[str]] = None,
    emit_progress: bool = False,
    eval_epochs: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    timings_s: Dict[str, float] = {}
    split_total_start = time.time()
    split_dir.mkdir(parents=True, exist_ok=True)
    eval_epoch_list = sorted({int(epoch) for epoch in (eval_epochs or [epochs])})
    if not eval_epoch_list:
        raise ValueError(f"{split_name}: eval_epochs cannot be empty.")
    if min(eval_epoch_list) <= 0:
        raise ValueError(f"{split_name}: eval_epochs must be positive integers, got {eval_epoch_list}.")
    if max(eval_epoch_list) > int(epochs):
        raise ValueError(f"{split_name}: eval_epochs {eval_epoch_list} exceed training epochs={epochs}.")
    eval_epoch_set = set(eval_epoch_list)
    setup_start = time.time()
    checkpoint_input_dim = int(np.asarray(checkpoint["params"]["Dense_0"]["kernel"]).shape[0])
    checkpoint_static_dim = int(np.asarray(checkpoint["params"]["Dense_1"]["kernel"]).shape[0])
    fold_config = base._resolve_fold_input_config(
        list(train_trials) + list(eval_trials),
        config,
        expected_input_dim=checkpoint_input_dim,
        expected_static_dim=checkpoint_static_dim,
    )
    _log_line(
        f"[{split_name}] layout={fold_config['resolved_input_layout']} "
        f"input_dim={fold_config['input_dim']} static_dim={fold_config['static_dim']} "
        f"epochs={epochs} lr={learning_rate:g}",
        log_handle,
        emit=emit_progress,
    )
    timings_s["resolve_input_config"] = _timer(
        f"{split_name} resolve_input_config",
        setup_start,
        emit=emit_progress,
        log_handle=log_handle,
    )

    loader_start = time.time()
    train_loader = base._safe_trial_loader(train_trials, fold_config, batch_size=batch_size, shuffle=True)
    eval_loader = base._safe_trial_loader(eval_trials, fold_config, batch_size=batch_size, shuffle=False)
    sample_batch = next(iter(train_loader))
    input_dim = int(sample_batch["input"].shape[-1])
    static_dim = int(sample_batch["static_context"].shape[-1])
    if checkpoint_input_dim != input_dim:
        raise ValueError(f"Checkpoint input_dim={checkpoint_input_dim}, loader input_dim={input_dim}.")
    if checkpoint_static_dim != static_dim:
        raise ValueError(f"Checkpoint static_dim={checkpoint_static_dim}, loader static_dim={static_dim}.")
    timings_s["build_loaders_and_first_batch"] = _timer(
        f"{split_name} build_loaders_and_first_batch",
        loader_start,
        emit=emit_progress,
        log_handle=log_handle,
    )

    model_start = time.time()
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
    dof_weights = base._build_dof_weights(fold_config)
    loss_weights = base._build_loss_weights(fold_config)
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
    timings_s["build_model_and_train_state"] = _timer(
        f"{split_name} build_model_and_train_state",
        model_start,
        emit=emit_progress,
        log_handle=log_handle,
    )

    history: List[Dict[str, Any]] = []
    epoch_results: List[Dict[str, Any]] = []
    start_time = time.time()
    training_start = time.time()
    metric_eval_total_s = 0.0
    for epoch in range(1, int(epochs) + 1):
        epoch_start = time.time()
        state, train_losses, rng = base._run_train_epoch(
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
                "epoch": int(epoch),
                "epoch_time_s": float(time.time() - epoch_start),
                "train_total_loss": float(train_losses["total_loss"]),
            }
        )
        epoch_elapsed = time.time() - epoch_start
        _log_line(
            f"[{split_name}] epoch {epoch}/{epochs} "
            f"train_loss={float(train_losses['total_loss']):.4f} "
            f"time={base._format_duration(epoch_elapsed)}",
            log_handle,
            emit=emit_progress,
        )
        if epoch in eval_epoch_set:
            eval_start = time.time()
            try:
                epoch_eval_payload = base._evaluate_loader(
                    state,
                    eval_loader,
                    eval_step=eval_step,
                    normalizers=normalizers,
                    loss_weights=loss_weights,
                    config=fold_config,
                    epoch=int(epoch),
                    require_left_kam=True,
                )
            except Exception as exc:
                raise RuntimeError(f"{split_name} metric-only evaluation failed at epoch {epoch}: {exc}") from exc
            eval_elapsed = time.time() - eval_start
            metric_eval_total_s += float(eval_elapsed)
            epoch_objective = _objective_from_metrics(epoch_eval_payload["metrics"])
            epoch_results.append(
                {
                    "epoch": int(epoch),
                    "objective": float(epoch_objective),
                    "eval_metrics": epoch_eval_payload,
                    "metric_eval_time_s": float(eval_elapsed),
                }
            )
            _log_line(
                f"[{split_name}] eval epoch {epoch}/{epochs} "
                f"objective={float(epoch_objective):.6f} "
                f"time={base._format_duration(eval_elapsed)}",
                log_handle,
                emit=emit_progress,
            )
    timings_s["training"] = _timer(
        f"{split_name} all_training_epochs",
        training_start,
        emit=emit_progress,
        log_handle=log_handle,
    )

    if len(epoch_results) != len(eval_epoch_list):
        found_epochs = [int(item["epoch"]) for item in epoch_results]
        raise RuntimeError(
            f"{split_name}: missing requested epoch evaluations. "
            f"requested={eval_epoch_list}, found={found_epochs}"
        )
    timings_s["metric_only_eval"] = float(metric_eval_total_s)
    eval_payload = dict(epoch_results[-1]["eval_metrics"])
    objective = _objective_from_metrics(eval_payload["metrics"])
    train_eval_elapsed = time.time() - start_time
    timings_s["train_eval_only"] = float(train_eval_elapsed)
    total_split_elapsed = _timer(
        f"{split_name} total_split",
        split_total_start,
        emit=emit_progress,
        log_handle=log_handle,
    )
    timings_s["total_split"] = float(total_split_elapsed)
    payload = {
        "split_name": split_name,
        "status": "completed",
        "epochs": int(epochs),
        "eval_epochs": eval_epoch_list,
        "selected_eval_epoch": int(epoch_results[-1]["epoch"]),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "objective_name": HPO_CONFIG["objective"],
        "objective": float(objective),
        "duration_s": float(total_split_elapsed),
        "timings_s": timings_s,
        "train_subjects": _trial_subjects(train_trials),
        "eval_subjects": _trial_subjects(eval_trials),
        "eval_metrics": eval_payload,
        "epoch_results": epoch_results,
        "history": history,
        "resolved_input_config": _resolved_input_summary(fold_config),
    }
    base._save_json(split_dir / "metrics.json", payload)

    if save_model:
        with (split_dir / "best_model.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "params": state.params,
                    "normalizers": normalizers,
                    "epochs": int(epochs),
                    "selected_eval_epoch": int(epoch_results[-1]["epoch"]),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "objective_name": HPO_CONFIG["objective"],
                    "objective": float(objective),
                    "eval_metrics": eval_payload,
                    "resolved_input_config": _resolved_input_summary(fold_config),
                    **dict(model_payload_extra or {}),
                },
                handle,
            )
    return payload


def _run_inner_fold_job(
    *,
    outer_index: int,
    candidate_index: int,
    inner_index: int,
    test_subject: str,
    candidate_dir: Path,
    inner_fold: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    batch_size: int,
    weight_decay: float,
    adapter_hidden_dim: int,
    adapter_dropout_rate: float,
    hpo_seed: int,
) -> Dict[str, Any]:
    inner_start = time.time()
    inner_val_subject = str(inner_fold["inner_val_subject"])
    split_name = f"{test_subject}/candidate_{candidate_index:03d}/inner_{inner_val_subject}"
    _assert_split_is_leakage_safe(
        train_trials=inner_fold["train_trials"],
        eval_trials=inner_fold["eval_trials"],
        forbidden_eval_subject=test_subject,
        split_name=split_name,
    )
    split_dir = candidate_dir / f"inner_{inner_val_subject}"
    split_dir.mkdir(parents=True, exist_ok=True)
    base._save_json(
        split_dir / "split.json",
        {
            "outer_test_subject": test_subject,
            "inner_val_subject": inner_val_subject,
            "train_subjects": list(inner_fold["train_subjects"]),
            "train_trials": list(inner_fold["train_trials"]),
            "eval_trials": list(inner_fold["eval_trials"]),
        },
    )
    log_path = split_dir / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as log_handle:
        inner_payload = _train_metric_only_split(
            split_name=split_name,
            split_dir=split_dir,
            checkpoint=checkpoint,
            config=config,
            train_trials=inner_fold["train_trials"],
            eval_trials=inner_fold["eval_trials"],
            epochs=int(candidate["epochs"]),
            learning_rate=float(candidate["learning_rate"]),
            batch_size=batch_size,
            weight_decay=weight_decay,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout_rate=adapter_dropout_rate,
            seed=int(hpo_seed + outer_index * 10000 + candidate_index * 100 + inner_index),
            save_model=False,
            log_handle=log_handle,
            emit_progress=False,
        )
    inner_elapsed = time.time() - inner_start
    return {
        "inner_index": int(inner_index),
        "inner_val_subject": inner_val_subject,
        "train_subjects": list(inner_fold["train_subjects"]),
        "split_name": split_name,
        "split_dir": split_dir,
        "metrics_path": split_dir / "metrics.json",
        "inner_payload": inner_payload,
        "objective": float(inner_payload["objective"]),
        "duration_s": float(inner_elapsed),
    }


def _candidate_epoch_group_key(candidate: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(candidate.get("source_checkpoint_path", "")),
        str(candidate.get("source_checkpoint_label", "")),
        candidate.get("source_checkpoint_epoch"),
        float(candidate["learning_rate"]),
    )


def _build_epoch_candidate_groups(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for candidate_index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["candidate_index"] = int(candidate_index)
        grouped.setdefault(_candidate_epoch_group_key(candidate), []).append(item)

    groups: List[Dict[str, Any]] = []
    for group_index, (_key, group_candidates) in enumerate(grouped.items()):
        group_candidates = sorted(group_candidates, key=lambda item: (int(item["epochs"]), int(item["candidate_index"])))
        epochs = sorted({int(item["epochs"]) for item in group_candidates})
        representative = group_candidates[0]
        groups.append(
            {
                "group_index": int(group_index),
                "candidates": group_candidates,
                "candidate_indices": [int(item["candidate_index"]) for item in group_candidates],
                "eval_epochs": epochs,
                "max_epochs": int(max(epochs)),
                "learning_rate": float(representative["learning_rate"]),
                "source_checkpoint_path": str(representative["source_checkpoint_path"]),
                "source_checkpoint_label": str(representative["source_checkpoint_label"]),
                "source_checkpoint_epoch": representative.get("source_checkpoint_epoch"),
            }
        )
    return groups


def _run_inner_fold_epoch_group_job(
    *,
    outer_index: int,
    group_index: int,
    inner_index: int,
    test_subject: str,
    group_dir: Path,
    inner_fold: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    group: Mapping[str, Any],
    batch_size: int,
    weight_decay: float,
    adapter_hidden_dim: int,
    adapter_dropout_rate: float,
    hpo_seed: int,
) -> Dict[str, Any]:
    inner_start = time.time()
    inner_val_subject = str(inner_fold["inner_val_subject"])
    split_name = f"{test_subject}/epoch_group_{group_index:03d}/inner_{inner_val_subject}"
    _assert_split_is_leakage_safe(
        train_trials=inner_fold["train_trials"],
        eval_trials=inner_fold["eval_trials"],
        forbidden_eval_subject=test_subject,
        split_name=split_name,
    )
    split_dir = group_dir / f"inner_{inner_val_subject}"
    split_dir.mkdir(parents=True, exist_ok=True)
    base._save_json(
        split_dir / "split.json",
        {
            "outer_test_subject": test_subject,
            "inner_val_subject": inner_val_subject,
            "train_subjects": list(inner_fold["train_subjects"]),
            "train_trials": list(inner_fold["train_trials"]),
            "eval_trials": list(inner_fold["eval_trials"]),
            "epoch_group": {
                "group_index": int(group_index),
                "candidate_indices": list(group["candidate_indices"]),
                "eval_epochs": list(group["eval_epochs"]),
                "max_epochs": int(group["max_epochs"]),
                "learning_rate": float(group["learning_rate"]),
                "source_checkpoint_label": str(group["source_checkpoint_label"]),
                "source_checkpoint_path": str(group["source_checkpoint_path"]),
            },
        },
    )
    log_path = split_dir / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as log_handle:
        inner_payload = _train_metric_only_split(
            split_name=split_name,
            split_dir=split_dir,
            checkpoint=checkpoint,
            config=config,
            train_trials=inner_fold["train_trials"],
            eval_trials=inner_fold["eval_trials"],
            epochs=int(group["max_epochs"]),
            learning_rate=float(group["learning_rate"]),
            batch_size=batch_size,
            weight_decay=weight_decay,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout_rate=adapter_dropout_rate,
            seed=int(hpo_seed + outer_index * 10000 + group_index * 100 + inner_index),
            save_model=False,
            log_handle=log_handle,
            emit_progress=False,
            eval_epochs=list(group["eval_epochs"]),
        )
    by_epoch = {int(item["epoch"]): item for item in inner_payload.get("epoch_results", [])}
    inner_elapsed = time.time() - inner_start
    return {
        "inner_index": int(inner_index),
        "inner_val_subject": inner_val_subject,
        "train_subjects": list(inner_fold["train_subjects"]),
        "split_name": split_name,
        "split_dir": split_dir,
        "metrics_path": split_dir / "metrics.json",
        "inner_payload": inner_payload,
        "objectives_by_epoch": {
            int(epoch): float(by_epoch[int(epoch)]["objective"]) for epoch in group["eval_epochs"]
        },
        "epoch_results_by_epoch": by_epoch,
        "duration_s": float(inner_elapsed),
    }


def _run_final_fold_job(
    *,
    outer_index: int,
    test_subject: str,
    fold_dir: Path,
    fold: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    selected_payload: Mapping[str, Any],
    batch_size: int,
    weight_decay: float,
    adapter_hidden_dim: int,
    adapter_dropout_rate: float,
    seed: int,
) -> Dict[str, Any]:
    _assert_split_is_leakage_safe(
        train_trials=fold["final_train_trials"],
        eval_trials=fold["held_out_trials"],
        forbidden_eval_subject=test_subject,
        split_name=f"{test_subject}/final",
    )
    log_path = fold_dir / "final_training_log.txt"
    final_start = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        final_payload = _train_metric_only_split(
            split_name=f"{test_subject}/final",
            split_dir=fold_dir,
            checkpoint=checkpoint,
            config=config,
            train_trials=fold["final_train_trials"],
            eval_trials=fold["held_out_trials"],
            epochs=int(selected_candidate["epochs"]),
            learning_rate=float(selected_candidate["learning_rate"]),
            batch_size=batch_size,
            weight_decay=weight_decay,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_dropout_rate=adapter_dropout_rate,
            seed=int(seed + outer_index),
            save_model=True,
            model_payload_extra={
                "held_out_subject": test_subject,
                "train_subjects": list(fold["final_train_subjects"]),
                "selected_hyperparameters": dict(selected_candidate),
                "inner_hpo_selection": dict(selected_payload),
            },
            log_handle=log_handle,
            emit_progress=False,
        )
    final_elapsed = time.time() - final_start
    base._save_json(fold_dir / "final_test_metrics.json", final_payload)
    return {
        "outer_index": int(outer_index),
        "test_subject": test_subject,
        "final_payload": final_payload,
        "final_elapsed": float(final_elapsed),
    }


def _run_final_infer_style_plots(
    *,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    adapter_hidden_dim: int,
    adapter_dropout_rate: float,
) -> Dict[str, Any]:
    test_subject = str(record["test_subject"])
    fold_dir = Path(record["fold_dir"])
    fold = record["fold"]
    model_path = fold_dir / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Final model not found for {test_subject}: {model_path}")
    with model_path.open("rb") as handle:
        final_bundle = pickle.load(handle)
    params = final_bundle["params"]
    normalizers = final_bundle["normalizers"]
    checkpoint_input_dim = int(np.asarray(params["Dense_0"]["kernel"]).shape[0])
    checkpoint_static_dim = int(np.asarray(params["Dense_1"]["kernel"]).shape[0])
    fold_config = base._resolve_fold_input_config(
        list(fold["final_train_trials"]) + list(fold["held_out_trials"]),
        config,
        expected_input_dim=checkpoint_input_dim,
        expected_static_dim=checkpoint_static_dim,
    )
    model = loso_adapters.build_loso_model(
        fold_config,
        params,
        adapter_hidden_dim=adapter_hidden_dim,
        adapter_dropout_rate=adapter_dropout_rate,
    )
    infer_fold = {
        "held_out_subject": test_subject,
        "inner_val_subject": "nested_hpo_selected",
        "held_out_trials": list(fold["held_out_trials"]),
    }
    plot_start = time.time()
    log_path = fold_dir / "final_infer_style_log.txt"
    with log_path.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            infer_payload = base._run_infer_style_evaluation(
                infer_fold,
                fold_dir=fold_dir,
                model=model,
                params=params,
                normalizers=normalizers,
                config=fold_config,
            )
    elapsed = time.time() - plot_start
    payload = {
        "held_out_subject": test_subject,
        "status": "completed",
        "duration_s": float(elapsed),
        "infer_style_eval": infer_payload,
        "log_path": str(log_path),
        "output_dir": str(fold_dir / "infer_style_eval"),
    }
    base._save_json(fold_dir / "final_infer_style_summary.json", payload)
    return payload


def _aggregate_final_infer_style_metrics(
    final_plot_results_by_subject: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    per_subject: Dict[str, float] = {}
    per_source_values: Dict[str, Dict[str, float]] = {}

    for subject, plot_payload in sorted(final_plot_results_by_subject.items()):
        if plot_payload.get("status") != "completed":
            continue
        infer_payload = plot_payload.get("infer_style_eval", {})
        subject_metric = infer_payload.get("subject_average_torque_mae_bwh_percent", {})
        subject_value = base._finite_float(subject_metric.get("mean"), default=float("nan"))
        if np.isfinite(subject_value):
            per_subject[str(subject)] = float(subject_value)

        by_source = infer_payload.get("subject_average_torque_mae_bwh_percent_by_source", {})
        for source_label, source_metric in by_source.items():
            source_value = base._finite_float(source_metric.get("mean"), default=float("nan"))
            if np.isfinite(source_value):
                per_source_values.setdefault(str(source_label), {})[str(subject)] = float(source_value)

    values = list(per_subject.values())
    aggregate = {
        "metric_name": "subject_average_torque_mae_bwh_percent",
        "description": "Cross-fold mean of per-subject average joint moment MAE normalized by BW*H percent.",
        "subject_count": len(values),
        "per_subject": per_subject,
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
    }

    by_source_aggregate: Dict[str, Any] = {}
    for source_label, source_subject_values in sorted(per_source_values.items()):
        source_values = list(source_subject_values.values())
        by_source_aggregate[source_label] = {
            "subject_count": len(source_values),
            "per_subject": source_subject_values,
            "mean": float(np.mean(source_values)) if source_values else None,
            "std": float(np.std(source_values)) if source_values else None,
        }

    return {
        "subject_average_torque_mae_bwh_percent": aggregate,
        "subject_average_torque_mae_bwh_percent_by_source": by_source_aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nested LOSO checkpoint fine-tuning HPO for learning rate and epochs.",
        add_help=False,
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to source best_model.pkl")
    parser.add_argument("--data_dir", type=str, default=None, help="OpenCap LOSO data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Nested LOSO HPO output directory")
    parser.add_argument("--trials_per_fold", type=int, default=None, help="Number of grid candidates per outer fold")
    parser.add_argument("--hpo_seed", type=int, default=None, help="Seed offset for candidate/fold training")
    parser.add_argument(
        "--max_parallel_splits",
        type=int,
        default=None,
        help="Number of inner LOSO validation splits to train concurrently per candidate.",
    )
    parser.add_argument(
        "--hpo_grid_json",
        type=str,
        default=None,
        help="JSON grid object or list of candidate objects for epochs/learning_rate.",
    )
    parser.add_argument(
        "--hpo_search_space_json",
        type=str,
        default=None,
        help="Deprecated alias for --hpo_grid_json.",
    )
    parser.add_argument(
        "--checkpoint_epoch_targets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated or JSON list of source model epochs to include as an HPO dimension. "
            "Each target resolves to the closest model_epoch_*.pkl next to --checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint_candidates_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file path or inline JSON list of checkpoint candidates. "
            "Each item can be a checkpoint path string or an object with path/label metadata. "
            "When provided, these candidates replace --checkpoint_epoch_targets."
        ),
    )
    parser.add_argument(
        "--selection_std_weight",
        type=float,
        default=None,
        help="Penalty weight in mean + weight * std candidate selection.",
    )
    parser.add_argument(
        "--verbose_inner_summaries",
        action="store_true",
        help="Print one terminal summary for every inner validation split.",
    )
    parser.add_argument(
        "--jax_cache_clear_every_candidates",
        type=int,
        default=7,
        help="Clear JAX compile/runtime caches after every N candidates. Use 0 to disable interval clearing.",
    )
    parser.add_argument(
        "--jax_cache_clear_memory_fraction",
        type=float,
        default=0.85,
        help="Clear JAX caches when system memory use reaches this fraction.",
    )
    parser.add_argument(
        "--skip_final_infer_style_plots",
        action="store_true",
        help="Skip one-off LOSO-style final held-out visualizations for selected final models.",
    )
    parser.add_argument(
        "--allow_checkpoint_loso_overlap",
        action="store_true",
        help="Allow source checkpoint metadata to overlap LOSO subjects.",
    )
    hpo_args, remaining = parser.parse_known_args()
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = original_argv
    for key, value in vars(hpo_args).items():
        setattr(args, key, value)
    return args


def main() -> None:
    hpo_total_start = time.time()
    args = base._merge_config_into_args(parse_args(), LOSO_CONFIG)
    args.trials_per_fold = int(
        args.trials_per_fold if args.trials_per_fold is not None else HPO_CONFIG["trials_per_fold"]
    )
    args.hpo_seed = int(args.hpo_seed if args.hpo_seed is not None else HPO_CONFIG["hpo_seed"])
    args.max_parallel_splits = max(
        1,
        int(
            args.max_parallel_splits
            if args.max_parallel_splits is not None
            else HPO_CONFIG["max_parallel_splits"]
        ),
    )
    selection_std_weight = float(
        args.selection_std_weight
        if args.selection_std_weight is not None
        else HPO_CONFIG["selection_std_weight"]
    )
    grid_json = args.hpo_grid_json or args.hpo_search_space_json
    grid_payload = json.loads(grid_json) if grid_json else HPO_CONFIG["grid"]

    checkpoint_candidates = _resolve_checkpoint_candidates_from_json(args.checkpoint_candidates_json)
    if args.checkpoint is None and checkpoint_candidates:
        args.checkpoint = str(checkpoint_candidates[0]["path"])
    if args.checkpoint is None:
        raise ValueError("No checkpoint specified. Pass --checkpoint, --checkpoint_candidates_json, or set LOSO_CONFIG['checkpoint'].")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint, config = base._load_checkpoint_bundle(checkpoint_path)
    config = base._apply_cli_overrides(config, args)
    base_candidates = _build_hpo_grid(grid_payload)
    checkpoint_epoch_targets = _parse_epoch_targets(args.checkpoint_epoch_targets)
    if checkpoint_candidates is None:
        checkpoint_candidates = _resolve_checkpoint_candidates(checkpoint_path, checkpoint_epoch_targets)
    elif checkpoint_epoch_targets:
        raise ValueError("--checkpoint_epoch_targets cannot be combined with --checkpoint_candidates_json.")
    hpo_candidates = _attach_checkpoint_candidates_to_grid(
        base_candidates,
        checkpoint_candidates,
        args.trials_per_fold,
    )
    hpo_epoch_groups = _build_epoch_candidate_groups(hpo_candidates)
    checkpoint_bundle_cache: Dict[str, Tuple[Mapping[str, Any], Dict[str, Any]]] = {}
    for checkpoint_candidate in checkpoint_candidates:
        candidate_checkpoint_path = Path(str(checkpoint_candidate["path"])).resolve()
        candidate_checkpoint, candidate_config = base._load_checkpoint_bundle(candidate_checkpoint_path)
        candidate_config = base._apply_cli_overrides(candidate_config, args)
        checkpoint_bundle_cache[str(candidate_checkpoint_path)] = (candidate_checkpoint, candidate_config)
    data_dir = base._resolve_loso_data_dir(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"OpenCapSubjects data directory not found: {data_dir}")

    batch_size = int(args.batch_size if args.batch_size is not None else config["batch_size"])
    adapter_hidden_dim = int(args.adapter_hidden_dim if args.adapter_hidden_dim is not None else config["ff_dim"])
    adapter_dropout_rate = float(
        args.adapter_dropout_rate if args.adapter_dropout_rate is not None else config["dropout_rate"]
    )
    weight_decay = float(args.weight_decay if args.weight_decay is not None else config["weight_decay"])
    output_root = Path(args.output_dir).resolve() if args.output_dir else checkpoint_path.parent / "loso_nested_hpo"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "folds").mkdir(exist_ok=True)

    if RUNTIME_ENV_APPLIED:
        print("Applied runtime safety env defaults:", flush=True)
        for key in sorted(RUNTIME_ENV_APPLIED.keys()):
            print(f"  {key}={os.environ.get(key)}", flush=True)

    print(f"Source checkpoint: {checkpoint_path}", flush=True)
    print(f"Source hyperparameters: {config['source_hyperparameters_path']}", flush=True)
    if checkpoint_epoch_targets:
        source_summary = ", ".join(
            f"{item['label']}<-targets{item['requested_epoch_targets']}" for item in checkpoint_candidates
        )
        print(f"Source checkpoint versions: {source_summary}", flush=True)
    print(f"OpenCap LOSO root: {data_dir}", flush=True)
    print(
        f"Nested LOSO HPO -> candidates={len(hpo_candidates)}, seed={args.hpo_seed}, "
        f"parallel_inner_splits={args.max_parallel_splits}, "
        f"objective={HPO_CONFIG['objective']}, score=mean+{selection_std_weight:g}*std",
        flush=True,
    )
    print(
        "Runtime caps -> "
        f"JAX_CPU_THREADS={os.environ.get('JAX_CPU_THREADS')}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
        f"cache_clear_every={args.jax_cache_clear_every_candidates}, "
        f"cache_clear_mem={float(args.jax_cache_clear_memory_fraction) * 100:.0f}%",
        flush=True,
    )

    _trials, all_subjects, valid_subjects, subject_to_trials = base._discover_subject_trials(
        data_dir,
        include_trunk_sway=False,
    )
    all_subject_groups = sorted(
        {base.subject_group_id(subject) if hasattr(base, "subject_group_id") else subject for subject in all_subjects},
        key=base._subject_sort_key,
    )
    skipped_subjects = sorted(
        set(subject for subject in all_subject_groups if subject not in valid_subjects),
        key=base._subject_sort_key,
    )
    missing_loso_subjects = sorted(
        set(base.OPEN_CAP_LOSO_SUBJECTS) - set(valid_subjects),
        key=base._subject_sort_key,
    )
    if missing_loso_subjects:
        local_filt_dir = PROJECT_ROOT / "OpenCapSubjects_Filt"
        hint = ""
        if local_filt_dir.exists() and data_dir.resolve() != local_filt_dir.resolve():
            hint = f" Local filtered data exists at {local_filt_dir}; omit --data_dir or pass --data_dir {local_filt_dir}."
        raise ValueError(
            "Nested LOSO HPO requires all expected LOSO subjects. "
            f"Missing valid trials for: {missing_loso_subjects}. "
            f"Discovered valid subjects: {valid_subjects}."
            f"{hint}"
        )
    overlap_report = _checkpoint_overlap_report(
        checkpoint,
        valid_subjects,
        allow_overlap=bool(args.allow_checkpoint_loso_overlap),
    )
    checkpoint_overlap_reports = {}
    for checkpoint_candidate in checkpoint_candidates:
        candidate_checkpoint_path = str(Path(str(checkpoint_candidate["path"])).resolve())
        candidate_checkpoint, _candidate_config = checkpoint_bundle_cache[candidate_checkpoint_path]
        checkpoint_overlap_reports[str(checkpoint_candidate["label"])] = _checkpoint_overlap_report(
            candidate_checkpoint,
            valid_subjects,
            allow_overlap=bool(args.allow_checkpoint_loso_overlap),
        )
    folds = _build_nested_loso_folds(valid_subjects, subject_to_trials)

    base._save_json(
        output_root / "hpo_config.json",
        {
            "source_checkpoint": str(checkpoint_path),
            "source_hyperparameters_path": config["source_hyperparameters_path"],
            "source_checkpoint_epoch_targets": checkpoint_epoch_targets,
            "source_checkpoint_candidates": checkpoint_candidates,
            "data_dir": str(data_dir),
            "requested_loso_subjects": list(base.OPEN_CAP_LOSO_SUBJECTS),
            "valid_subjects": list(valid_subjects),
            "skipped_subjects": skipped_subjects,
            "trials_per_fold": int(args.trials_per_fold),
            "hpo_seed": int(args.hpo_seed),
            "max_parallel_splits": int(args.max_parallel_splits),
            "verbose_inner_summaries": bool(args.verbose_inner_summaries),
            "jax_cache_clear_every_candidates": int(args.jax_cache_clear_every_candidates),
            "jax_cache_clear_memory_fraction": float(args.jax_cache_clear_memory_fraction),
            "final_infer_style_plots": not bool(args.skip_final_infer_style_plots),
            "runtime_thread_caps": {
                "JAX_CPU_THREADS": os.environ.get("JAX_CPU_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            },
            "objective": HPO_CONFIG["objective"],
            "selection_score": f"mean + {selection_std_weight} * std",
            "selected_left_stance_dof_names": list(base.infer_module.SELECTED_LEFT_STANCE_DOF_NAMES)
            + [base.infer_module.LEFT_STANCE_KAM_DOF_NAME],
            "hpo_candidates": hpo_candidates,
            "epoch_grouped_training": True,
            "hpo_epoch_groups": hpo_epoch_groups,
            "checkpoint_overlap_report": overlap_report,
            "checkpoint_overlap_reports_by_source": checkpoint_overlap_reports,
            "folds": [
                {
                    "held_out_subject": fold["held_out_subject"],
                    "final_train_subjects": fold["final_train_subjects"],
                    "inner_validation_subjects": [
                        inner_fold["inner_val_subject"] for inner_fold in fold["inner_folds"]
                    ],
                }
                for fold in folds
            ],
        },
    )

    fold_records: List[Dict[str, Any]] = []
    for outer_index, fold in enumerate(folds):
        test_subject = str(fold["held_out_subject"])
        fold_dir = output_root / "folds" / test_subject
        inner_root = fold_dir / "inner_hpo"
        fold_dir.mkdir(parents=True, exist_ok=True)
        base._save_json(
            fold_dir / "split.json",
            {
                "held_out_subject": test_subject,
                "final_train_subjects": list(fold["final_train_subjects"]),
                "held_out_trials": list(fold["held_out_trials"]),
                "final_train_trials": list(fold["final_train_trials"]),
                "inner_folds": [
                    {
                        "inner_val_subject": inner_fold["inner_val_subject"],
                        "train_subjects": inner_fold["train_subjects"],
                        "train_trials": list(inner_fold["train_trials"]),
                        "eval_trials": list(inner_fold["eval_trials"]),
                    }
                    for inner_fold in fold["inner_folds"]
                ],
            },
        )
        fold_records.append(
            {
                "outer_index": int(outer_index),
                "fold": fold,
                "test_subject": test_subject,
                "fold_dir": fold_dir,
                "inner_root": inner_root,
                "outer_start": time.time(),
                "candidate_rows": [],
                "candidate_payloads": [],
            }
        )

    print(
        f"\nEpoch-grouped scheduling: {len(hpo_candidates)} candidates collapsed into "
        f"{len(hpo_epoch_groups)} grouped trainings x {len(fold_records)} outer folds x "
        f"{len(fold_records[0]['fold']['inner_folds']) if fold_records else 0} inner folds, "
        f"max_workers={args.max_parallel_splits}",
        flush=True,
    )

    completed_candidate_count = 0
    for group in hpo_epoch_groups:
        group_index = int(group["group_index"])
        group_start = time.time()
        candidate_checkpoint_path = str(Path(str(group["source_checkpoint_path"])).resolve())
        candidate_checkpoint, candidate_config = checkpoint_bundle_cache[candidate_checkpoint_path]
        print(
            f"\n=== Epoch group {group_index + 1}/{len(hpo_epoch_groups)}: "
            f"source={group['source_checkpoint_label']} lr={float(group['learning_rate']):g} "
            f"eval_epochs={list(group['eval_epochs'])} max_epoch={int(group['max_epochs'])} "
            f"candidates={list(group['candidate_indices'])} ===",
            flush=True,
        )

        completed_by_subject: Dict[str, List[Dict[str, Any]]] = {
            str(record["test_subject"]): [] for record in fold_records
        }
        errors_by_subject: Dict[str, List[str]] = {
            str(record["test_subject"]): [] for record in fold_records
        }

        with ThreadPoolExecutor(max_workers=int(args.max_parallel_splits)) as executor:
            future_meta: Dict[Any, Dict[str, Any]] = {}
            for record in fold_records:
                outer_index = int(record["outer_index"])
                fold = record["fold"]
                test_subject = str(record["test_subject"])
                group_dir = Path(record["inner_root"]) / f"epoch_group_{group_index:03d}"
                group_dir.mkdir(parents=True, exist_ok=True)
                for inner_index, inner_fold in enumerate(fold["inner_folds"]):
                    future = executor.submit(
                        _run_inner_fold_epoch_group_job,
                        outer_index=outer_index,
                        group_index=group_index,
                        inner_index=inner_index,
                        test_subject=test_subject,
                        group_dir=group_dir,
                        inner_fold=inner_fold,
                        checkpoint=candidate_checkpoint,
                        config=candidate_config,
                        group=group,
                        batch_size=batch_size,
                        weight_decay=weight_decay,
                        adapter_hidden_dim=adapter_hidden_dim,
                        adapter_dropout_rate=adapter_dropout_rate,
                        hpo_seed=int(args.hpo_seed),
                    )
                    future_meta[future] = {
                        "test_subject": test_subject,
                        "inner_val_subject": str(inner_fold["inner_val_subject"]),
                    }

            for future in as_completed(future_meta):
                meta = future_meta[future]
                test_subject = str(meta["test_subject"])
                try:
                    result = future.result()
                except Exception as exc:
                    error_message = (
                        f"{test_subject}/epoch_group_{group_index:03d}/"
                        f"inner_{meta['inner_val_subject']}: {exc}"
                    )
                    print(f"[{test_subject}] epoch group {group_index + 1} inner failed: {exc}", flush=True)
                    errors_by_subject[test_subject].append(error_message)
                    continue
                if bool(args.verbose_inner_summaries):
                    _print_split_summary(str(result["split_name"]), result["inner_payload"])
                completed_by_subject[test_subject].append(result)

        group_elapsed = time.time() - group_start
        for record in fold_records:
            test_subject = str(record["test_subject"])
            completed_inner_results = completed_by_subject[test_subject]
            expected_inner_count = len(record["fold"]["inner_folds"])
            completed_by_epoch: Dict[int, List[Dict[str, Any]]] = {
                int(epoch): [] for epoch in group["eval_epochs"]
            }
            if not errors_by_subject[test_subject] and len(completed_inner_results) == expected_inner_count:
                for result in sorted(completed_inner_results, key=lambda item: int(item["inner_index"])):
                    for epoch in group["eval_epochs"]:
                        epoch_i = int(epoch)
                        epoch_result = result["epoch_results_by_epoch"][epoch_i]
                        completed_by_epoch[epoch_i].append(
                            {
                                "inner_val_subject": str(result["inner_val_subject"]),
                                "train_subjects": list(result["train_subjects"]),
                                "objective": float(epoch_result["objective"]),
                                "duration_s": float(result["duration_s"]),
                                "metrics_path": str(result["metrics_path"]),
                                "epoch": int(epoch_i),
                            }
                        )

            for candidate in group["candidates"]:
                candidate_index = int(candidate["candidate_index"])
                candidate_dir = Path(record["inner_root"]) / f"candidate_{candidate_index:03d}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                candidate_payload: Dict[str, Any] = {
                    "outer_test_subject": test_subject,
                    "candidate_index": int(candidate_index),
                    "parameters": dict(candidate),
                    "source_checkpoint_label": str(candidate["source_checkpoint_label"]),
                    "source_checkpoint_path": str(candidate["source_checkpoint_path"]),
                    "source_checkpoint_epoch": candidate.get("source_checkpoint_epoch"),
                    "epoch_group_index": int(group_index),
                    "epoch_group_max_epochs": int(group["max_epochs"]),
                    "status": "running",
                    "inner_results": [],
                }
                candidate_epoch = int(candidate["epochs"])
                inner_scores: List[float] = []
                if errors_by_subject[test_subject]:
                    candidate_payload.update(
                        {
                            "status": "failed",
                            "error": "; ".join(errors_by_subject[test_subject]),
                            "inner_mean": float("inf"),
                            "inner_std": float("inf"),
                            "selection_score": float("inf"),
                        }
                    )
                elif len(completed_inner_results) != expected_inner_count:
                    candidate_payload.update(
                        {
                            "status": "failed",
                            "error": (
                                f"Expected {expected_inner_count} inner results, "
                                f"received {len(completed_inner_results)}."
                            ),
                            "inner_mean": float("inf"),
                            "inner_std": float("inf"),
                            "selection_score": float("inf"),
                        }
                    )
                else:
                    for item in completed_by_epoch[candidate_epoch]:
                        inner_scores.append(float(item["objective"]))
                        candidate_payload["inner_results"].append(item)
                    mean_score, std_score, selection_score, _score_count = base._mean_std_score(inner_scores, selection_std_weight)
                    print(
                        f"[{test_subject}] candidate {candidate_index + 1}/{len(hpo_candidates)} complete: "
                        f"epoch={candidate_epoch} mean={mean_score:.4f} std={std_score:.4f} "
                        f"score={selection_score:.4f}",
                        flush=True,
                    )
                    candidate_payload.update(
                        {
                            "status": "completed",
                            "inner_mean": mean_score,
                            "inner_std": std_score,
                            "selection_score": selection_score,
                        }
                    )
                candidate_payload["duration_s"] = float(group_elapsed)
                _save_json_any(candidate_dir / "candidate_result.json", candidate_payload)
                record["candidate_payloads"].append(candidate_payload)
                record["candidate_rows"].append(
                    {
                        "outer_test_subject": test_subject,
                        "candidate_index": int(candidate_index),
                        "status": candidate_payload["status"],
                        "epochs": int(candidate["epochs"]),
                        "learning_rate": float(candidate["learning_rate"]),
                        "source_checkpoint_label": str(candidate["source_checkpoint_label"]),
                        "source_checkpoint_path": str(candidate["source_checkpoint_path"]),
                        "source_checkpoint_epoch": (
                            int(candidate["source_checkpoint_epoch"])
                            if candidate.get("source_checkpoint_epoch") is not None
                            else ""
                        ),
                        "epoch_group_index": int(group_index),
                        "epoch_group_max_epochs": int(group["max_epochs"]),
                        "inner_mean": float(candidate_payload["inner_mean"]),
                        "inner_std": float(candidate_payload["inner_std"]),
                        "selection_score": float(candidate_payload["selection_score"]),
                        "duration_s": float(group_elapsed),
                        "error": candidate_payload.get("error", ""),
                    }
                )
        completed_candidate_count += len(group["candidates"])
        gc.collect()
        base._maybe_clear_jax_caches(
            completed_count=group_index + 1,
            clear_every=int(args.jax_cache_clear_every_candidates),
            memory_fraction_threshold=float(args.jax_cache_clear_memory_fraction),
        )
        _write_hpo_progress_snapshot(
            output_root,
            fold_records,
            completed_candidates=completed_candidate_count,
            total_candidates=len(hpo_candidates),
        )
        print(
            f"[progress] saved epoch group {group_index + 1}/{len(hpo_epoch_groups)} snapshot "
            f"({completed_candidate_count}/{len(hpo_candidates)} candidates emitted)",
            flush=True,
        )
        _timer(f"epoch_group_{group_index:03d} all_outer_inner_folds", group_start)

    overall_rows: List[Dict[str, Any]] = []
    overall_payload: List[Dict[str, Any]] = []
    final_jobs: List[Dict[str, Any]] = []
    for record in fold_records:
        outer_start = float(record["outer_start"])
        outer_index = int(record["outer_index"])
        fold = record["fold"]
        test_subject = str(record["test_subject"])
        fold_dir = Path(record["fold_dir"])
        candidate_rows = list(record["candidate_rows"])
        candidate_payloads = list(record["candidate_payloads"])
        print(f"\n=== Finalizing outer fold {outer_index + 1}/{len(fold_records)}: test={test_subject} ===", flush=True)
        _save_json_any(fold_dir / "inner_hpo_results.json", candidate_payloads)
        _write_csv(fold_dir / "inner_hpo_results.csv", candidate_rows)
        completed = [
            row for row in candidate_rows if row["status"] == "completed" and np.isfinite(row["selection_score"])
        ]
        if not completed:
            raise RuntimeError(f"All nested HPO candidates failed for outer test subject {test_subject}.")
        selected_row = min(completed, key=lambda row: float(row["selection_score"]))
        selected_candidate = dict(hpo_candidates[int(selected_row["candidate_index"])])
        selected_payload = {
            "held_out_subject": test_subject,
            "selected_candidate_index": int(selected_row["candidate_index"]),
            "selected_hyperparameters": selected_candidate,
            "inner_mean": float(selected_row["inner_mean"]),
            "inner_std": float(selected_row["inner_std"]),
            "selection_score": float(selected_row["selection_score"]),
            "selection_score_formula": f"mean + {selection_std_weight} * std",
        }
        base._save_json(fold_dir / "selected_hyperparameters.json", selected_payload)

        print(
            f"[{test_subject}] selected candidate {selected_payload['selected_candidate_index']} "
            f"score={selected_payload['selection_score']:.4f} "
            f"source={selected_candidate['source_checkpoint_label']} "
            f"epochs={selected_candidate['epochs']} lr={selected_candidate['learning_rate']:g}",
            flush=True,
        )
        record["selected_row"] = selected_row
        record["selected_candidate"] = selected_candidate
        record["selected_payload"] = selected_payload
        final_jobs.append(record)

    print(
        f"\nRunning {len(final_jobs)} selected final outer-fold trainings "
        f"with max_workers={args.max_parallel_splits}",
        flush=True,
    )
    final_start_all = time.time()
    final_results_by_subject: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(int(args.max_parallel_splits), max(1, len(final_jobs)))) as executor:
        future_to_record = {}
        for record in final_jobs:
            selected_candidate = record["selected_candidate"]
            final_checkpoint_path = str(Path(str(selected_candidate["source_checkpoint_path"])).resolve())
            final_checkpoint, final_config = checkpoint_bundle_cache[final_checkpoint_path]
            future = executor.submit(
                _run_final_fold_job,
                outer_index=int(record["outer_index"]),
                test_subject=str(record["test_subject"]),
                fold_dir=Path(record["fold_dir"]),
                fold=record["fold"],
                checkpoint=final_checkpoint,
                config=final_config,
                selected_candidate=selected_candidate,
                selected_payload=record["selected_payload"],
                batch_size=batch_size,
                weight_decay=weight_decay,
                adapter_hidden_dim=adapter_hidden_dim,
                adapter_dropout_rate=adapter_dropout_rate,
                seed=int(args.seed),
            )
            future_to_record[future] = record
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            test_subject = str(record["test_subject"])
            result = future.result()
            final_results_by_subject[test_subject] = result
            _print_split_summary(f"{test_subject}/final", result["final_payload"])
            print(
                f"[{test_subject}] final training/eval complete in {base._format_duration(result['final_elapsed'])}",
                flush=True,
            )
    _timer("all_final_outer_fold_trainings", final_start_all)

    final_plot_results_by_subject: Dict[str, Dict[str, Any]] = {}
    if bool(args.skip_final_infer_style_plots):
        print("Skipping final infer-style visualizations (--skip_final_infer_style_plots).", flush=True)
    else:
        print("\nGenerating final one-off LOSO-style visualizations for selected models...", flush=True)
        plot_start_all = time.time()
        for record in final_jobs:
            test_subject = str(record["test_subject"])
            try:
                plot_payload = _run_final_infer_style_plots(
                    record=record,
                    config=config,
                    adapter_hidden_dim=adapter_hidden_dim,
                    adapter_dropout_rate=adapter_dropout_rate,
                )
                final_plot_results_by_subject[test_subject] = plot_payload
                print(
                    f"[{test_subject}] infer-style plots complete in "
                    f"{base._format_duration(float(plot_payload['duration_s']))}",
                    flush=True,
                )
            except Exception as exc:
                error_payload = {
                    "held_out_subject": test_subject,
                    "status": "failed",
                    "error": str(exc),
                    "output_dir": str(Path(record["fold_dir"]) / "infer_style_eval"),
                }
                final_plot_results_by_subject[test_subject] = error_payload
                base._save_json(Path(record["fold_dir"]) / "final_infer_style_summary.json", error_payload)
                print(f"[{test_subject}] infer-style plots failed: {exc}", flush=True)
        _timer("all_final_infer_style_visualizations", plot_start_all)

    for record in final_jobs:
        outer_start = float(record["outer_start"])
        outer_index = int(record["outer_index"])
        test_subject = str(record["test_subject"])
        fold_dir = Path(record["fold_dir"])
        selected_row = record["selected_row"]
        selected_candidate = record["selected_candidate"]
        selected_payload = record["selected_payload"]
        final_result = final_results_by_subject[test_subject]
        final_payload = final_result["final_payload"]
        final_elapsed = float(final_result["final_elapsed"])
        final_plot_payload = final_plot_results_by_subject.get(test_subject, {})
        final_metric = float(final_payload["objective"])
        outer_elapsed = _timer(f"{test_subject} outer_fold_total", outer_start)
        summary_row = {
            "held_out_subject": test_subject,
            "selected_candidate_index": int(selected_row["candidate_index"]),
            "source_checkpoint_label": str(selected_candidate["source_checkpoint_label"]),
            "source_checkpoint_path": str(selected_candidate["source_checkpoint_path"]),
            "source_checkpoint_epoch": (
                int(selected_candidate["source_checkpoint_epoch"])
                if selected_candidate.get("source_checkpoint_epoch") is not None
                else ""
            ),
            "epochs": int(selected_candidate["epochs"]),
            "learning_rate": float(selected_candidate["learning_rate"]),
            "inner_mean": float(selected_row["inner_mean"]),
            "inner_std": float(selected_row["inner_std"]),
            "selection_score": float(selected_row["selection_score"]),
            "final_test_objective": final_metric,
            "final_duration_s": float(final_elapsed),
            "final_infer_style_status": final_plot_payload.get("status", "skipped"),
            "final_infer_style_output_dir": final_plot_payload.get("output_dir", ""),
            "outer_duration_s": float(outer_elapsed),
        }
        overall_rows.append(summary_row)
        overall_payload.append(
            {
                **summary_row,
                "selected_hyperparameters": selected_candidate,
                "selected_payload": selected_payload,
                "final_test_metrics_path": str(fold_dir / "final_test_metrics.json"),
                "best_model_path": str(fold_dir / "best_model.pkl"),
                "final_infer_style": final_plot_payload,
            }
        )

    metric_means, metric_stds = base._aggregate_metric_dicts(
        [{"final_test_objective": row["final_test_objective"]} for row in overall_rows]
    )
    final_infer_style_metric_summary = _aggregate_final_infer_style_metrics(final_plot_results_by_subject)
    summary = {
        "source_checkpoint": str(checkpoint_path),
        "source_hyperparameters_path": config["source_hyperparameters_path"],
        "data_dir": str(data_dir),
        "valid_subjects": list(valid_subjects),
        "skipped_subjects": skipped_subjects,
        "objective": HPO_CONFIG["objective"],
        "selection_std_weight": selection_std_weight,
        "completed_folds": len(overall_rows),
        "metric_means": metric_means,
        "metric_stds": metric_stds,
        "final_infer_style_metric_summary": final_infer_style_metric_summary,
        "per_fold": overall_payload,
    }
    base._save_json(output_root / "loso_nested_hpo_summary.json", summary)
    _write_csv(output_root / "loso_nested_hpo_summary.csv", overall_rows)
    _timer("nested_loso_hpo_total", hpo_total_start)
    print(f"\nSaved nested LOSO HPO summary to {output_root / 'loso_nested_hpo_summary.json'}", flush=True)
    print(f"Saved compact CSV to {output_root / 'loso_nested_hpo_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
