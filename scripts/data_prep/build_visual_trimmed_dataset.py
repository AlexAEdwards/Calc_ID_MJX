#!/usr/bin/env python3
"""Materialize visual-cleaner decisions into a traced dataset and optionally merge it.

The source dataset is never modified. Trials with a saved trim window are
physically sliced; trials in ``keep_trials`` without a window are copied
unchanged; trials in ``remove_trials`` or without a completed decision are
excluded.

ProcessedData frame bounds come directly from ``visual_cleaning_review.json``.
Motion bounds are mapped by time through ProcessData's recorded
``core_trim_bounds_motion_aligned`` and the saved processed sample rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REVIEW_NAME = "visual_cleaning_review.json"
TRACE_NAME = "visual_cleaning_traceability.json"
TRIAL_TRACE_NAME = "Visual_Trim_Application.json"
BUILD_MANIFEST_NAME = "visual_trimmed_dataset_manifest.json"
MERGE_MANIFEST_NAME = "pd_visual_trim_merge_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_value(value: Any) -> str:
    """Hash a JSON value using a stable canonical serialization."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reset_time_array(arr: np.ndarray, filename: str) -> np.ndarray:
    if filename not in {"Time.npy", "Time_for_pos.npy"}:
        return arr
    if arr.ndim == 1 and arr.size:
        return arr - arr[0]
    return arr


def slice_axis(arr: np.ndarray, axis: int, lo: int, hi: int) -> np.ndarray:
    index = [slice(None)] * arr.ndim
    index[axis] = slice(lo, hi)
    return arr[tuple(index)]


def slice_time_aligned_array(
    arr: np.ndarray,
    *,
    source_frames: int,
    lo: int,
    hi: int,
    context: str,
) -> tuple[np.ndarray, list[int]]:
    """Slice every uniquely identifiable axis matching source_frames."""
    if arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if not isinstance(obj, dict):
            return arr, []
        changed_axes: list[int] = []
        result: dict[Any, Any] = {}
        for key, value in obj.items():
            value_arr = np.asarray(value)
            axes = [axis for axis, size in enumerate(value_arr.shape) if int(size) == source_frames]
            if len(axes) > 1:
                raise ValueError(
                    f"Ambiguous time axes for {context}.{key}: shape={value_arr.shape}, "
                    f"source_frames={source_frames}"
                )
            if len(axes) == 1:
                result[key] = slice_axis(value_arr, axes[0], lo, hi)
                changed_axes.append(int(axes[0]))
            else:
                result[key] = value
        return np.array(result, dtype=object), changed_axes

    axes = [axis for axis, size in enumerate(arr.shape) if int(size) == source_frames]
    if len(axes) > 1:
        raise ValueError(
            f"Ambiguous time axes for {context}: shape={arr.shape}, "
            f"source_frames={source_frames}"
        )
    if len(axes) == 1:
        return slice_axis(arr, axes[0], lo, hi), [int(axes[0])]
    return arr, []


def copy_processed_data(
    source_dir: Path,
    output_dir: Path,
    *,
    source_frames: int,
    lo: int,
    hi: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_dir.iterdir()):
        if src.is_dir():
            # Historical backup directories are deliberately excluded. The
            # source dataset and hashes below are the authoritative pre-trim copy.
            continue
        dst = output_dir / src.name
        record: dict[str, Any] = {
            "path": src.name,
            "source_sha256": sha256_file(src),
            "operation": "copied",
        }
        if src.suffix == ".npy":
            arr = np.load(src, allow_pickle=True)
            out, axes = slice_time_aligned_array(
                arr,
                source_frames=source_frames,
                lo=lo,
                hi=hi,
                context=str(src),
            )
            np.save(dst, out, allow_pickle=bool(arr.dtype == object))
            if axes:
                record["operation"] = "sliced"
                record["sliced_axes"] = axes
                record["source_shape"] = [int(v) for v in arr.shape]
                record["output_shape"] = [int(v) for v in out.shape]
        else:
            shutil.copy2(src, dst)
        record["output_sha256"] = sha256_file(dst)
        records.append(record)
    return records


def motion_bounds(
    trial_dir: Path,
    info: dict[str, Any],
    *,
    proc_lo: int,
    proc_hi: int,
    sample_rate_hz: float,
) -> tuple[int, int, float, float]:
    time_path = trial_dir / "Motion" / "Time.npy"
    if not time_path.exists():
        time_path = trial_dir / "Motion" / "Time_for_pos.npy"
    raw_time = np.asarray(np.load(time_path), dtype=np.float64).reshape(-1)
    if raw_time.size < 1 or not np.all(np.diff(raw_time) >= 0):
        raise ValueError(f"Invalid Motion time vector: {time_path}")

    core_bounds = info.get("core_trim_bounds_motion_aligned")
    if not isinstance(core_bounds, list) or len(core_bounds) != 2:
        raise ValueError(
            f"Missing core_trim_bounds_motion_aligned in "
            f"{trial_dir / 'ProcessedData' / 'Trial_Processing_Information.json'}"
        )
    core_start = int(core_bounds[0])
    start_time = float(raw_time[0]) + (core_start + proc_lo) / sample_rate_hz
    end_time_exclusive = float(raw_time[0]) + (core_start + proc_hi) / sample_rate_hz
    tolerance = 1e-9
    raw_lo = int(np.searchsorted(raw_time, start_time - tolerance, side="left"))
    raw_hi = int(np.searchsorted(raw_time, end_time_exclusive - tolerance, side="left"))
    raw_lo = max(0, min(raw_lo, raw_time.size))
    raw_hi = max(raw_lo + 1, min(raw_hi, raw_time.size))
    return raw_lo, raw_hi, start_time, end_time_exclusive


def copy_motion_data(
    source_dir: Path,
    output_dir: Path,
    *,
    raw_frames: int,
    lo: int,
    hi: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_dir.iterdir()):
        if src.is_dir():
            # Raw/ is intentionally excluded from the cleaned training dataset.
            continue
        dst = output_dir / src.name
        record: dict[str, Any] = {
            "path": src.name,
            "source_sha256": sha256_file(src),
            "operation": "copied",
        }
        if src.suffix == ".npy":
            arr = np.load(src, allow_pickle=True)
            out, axes = slice_time_aligned_array(
                arr,
                source_frames=raw_frames,
                lo=lo,
                hi=hi,
                context=str(src),
            )
            out = reset_time_array(out, src.name)
            np.save(dst, out, allow_pickle=bool(arr.dtype == object))
            if axes:
                record["operation"] = "sliced"
                record["sliced_axes"] = axes
                record["source_shape"] = [int(v) for v in arr.shape]
                record["output_shape"] = [int(v) for v in out.shape]
        elif src.suffix.lower() == ".mot":
            slice_mot_file(src, dst, lo, hi)
            record["operation"] = "sliced_by_row"
        else:
            shutil.copy2(src, dst)
        record["output_sha256"] = sha256_file(dst)
        records.append(record)
    return records


def slice_mot_file(src: Path, dst: Path, lo: int, hi: int) -> None:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    end_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip().lower() == "endheader"),
        None,
    )
    if end_idx is None:
        shutil.copy2(src, dst)
        return
    header = lines[: end_idx + 1]
    body = [line for line in lines[end_idx + 1 :] if line.strip()]
    if not body:
        shutil.copy2(src, dst)
        return
    columns = body[0]
    rows = body[1:]
    if hi > len(rows):
        raise ValueError(f"MOT bounds [{lo}:{hi}] exceed {len(rows)} rows: {src}")
    kept = rows[lo:hi]
    parsed: list[list[str]] = [re.split(r"\s+", row.strip()) for row in kept]
    if parsed and parsed[0]:
        first_time = float(parsed[0][0])
        for fields in parsed:
            fields[0] = f"{float(fields[0]) - first_time:.10g}"
        kept = ["\t".join(fields) for fields in parsed]

    nrows = len(kept)
    duration = float(parsed[-1][0]) if parsed else 0.0
    updated_header = []
    for line in header:
        if re.match(r"^\s*nRows\s*=", line, flags=re.IGNORECASE):
            line = re.sub(r"=.*$", f"={nrows}", line)
        elif re.match(r"^\s*datarows\s+", line, flags=re.IGNORECASE):
            line = re.sub(r"\d+\s*$", str(nrows), line)
        elif re.match(r"^\s*range\s+", line, flags=re.IGNORECASE):
            line = f"range 0 {duration:.10g}"
        updated_header.append(line)
    dst.write_text("\n".join(updated_header + [columns] + kept) + "\n", encoding="utf-8")


def update_processing_info(
    path: Path,
    *,
    label: str,
    decision: str,
    window: dict[str, Any] | None,
    source_frames: int,
    lo: int,
    hi: int,
    source_dataset: Path,
) -> None:
    info = load_json(path) if path.exists() else {}
    info.update(
        {
            "n_frames": int(hi - lo),
            "manual_visual_trim_applied": bool(window is not None),
            "manual_visual_trim_decision": decision,
            "manual_visual_trim_source_trial": label,
            "manual_visual_trim_source_dataset": str(source_dataset.resolve()),
            "manual_visual_trim_window": window,
            "manual_visual_trim_bounds_processed_half_open": [int(lo), int(hi)],
            "manual_visual_trim_n_frames_before": int(source_frames),
            "manual_visual_trim_n_frames_after": int(hi - lo),
            "manual_visual_trim_timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(path, info)


def update_trimming_traceability(
    path: Path,
    *,
    label: str,
    decision: str,
    window: dict[str, Any] | None,
    source_frames: int,
    lo: int,
    hi: int,
    source_dataset: Path,
    source_review_json: Path,
    source_review_sha256: str,
    source_visual_traceability_json: Path | None,
    source_visual_traceability_sha256: str | None,
    source_visual_trial_record_sha256: str | None,
    sample_rate_hz: float,
) -> None:
    """Compose the manual selection with ProcessData's original timeline map."""
    trace = load_json(path)
    final_mapping = trace.setdefault("final_mapping", {})
    prior_uniform_bounds = final_mapping.get("uniform_resampled_frame_bounds")
    if (
        not isinstance(prior_uniform_bounds, list)
        or len(prior_uniform_bounds) != 2
        or int(prior_uniform_bounds[1]) - int(prior_uniform_bounds[0]) != source_frames
    ):
        raise ValueError(
            f"{label}: cannot compose manual trim with {path.name}; "
            f"invalid prior uniform bounds {prior_uniform_bounds} for {source_frames} frames"
        )

    prior_uniform_lo = int(prior_uniform_bounds[0])
    uniform_lo = prior_uniform_lo + int(lo)
    uniform_hi = prior_uniform_lo + int(hi)
    dt = 1.0 / float(sample_rate_hz)
    uniform_start_s = float(
        trace.get("uniform_resampling", {}).get("start_time_s", 0.0)
    )
    timestamp = datetime.now().isoformat(timespec="seconds")
    stage = {
        "name": "manual_visual_keep_window",
        "enabled": bool(window is not None),
        "input_frame_count": int(source_frames),
        "keep_bounds_in_input": [int(lo), int(hi)],
        "keep_bounds_in_uniform_resampled_timeline": [uniform_lo, uniform_hi],
        "removed_leading_frames": int(lo),
        "removed_trailing_frames": int(source_frames - hi),
        "output_frame_count": int(hi - lo),
        "parameters": {
            "decision": decision,
            "interval_convention": "zero-based half-open [start, end)",
            "source_review_json": str(source_review_json.resolve()),
            "source_review_json_sha256": source_review_sha256,
            "source_visual_traceability_json": (
                str(source_visual_traceability_json.resolve())
                if source_visual_traceability_json is not None
                else None
            ),
            "source_visual_traceability_json_sha256": source_visual_traceability_sha256,
            "source_visual_trial_record_sha256": source_visual_trial_record_sha256,
            "visual_window": window,
        },
    }
    stages = [
        item
        for item in trace.setdefault("timeline_stages", [])
        if item.get("name") != "manual_visual_keep_window"
    ]
    stages.append(stage)
    trace["timeline_stages"] = stages
    final_mapping.update(
        {
            "uniform_resampled_frame_bounds": [uniform_lo, uniform_hi],
            "final_frame_count": int(hi - lo),
            "final_first_time_s": uniform_start_s + uniform_lo * dt,
            "final_last_time_s": uniform_start_s + (uniform_hi - 1) * dt,
            "mapping_formula": (
                f"cleaned_frame[j] corresponds to uniform_resampled_frame"
                f"[{uniform_lo} + j]"
            ),
            "source_mapping_note": (
                "The ProcessData timeline stages followed by manual_visual_keep_window "
                "compose the complete mapping to raw kinematic and force rows."
            ),
        }
    )
    history = [
        item
        for item in trace.setdefault("postprocessing_history", [])
        if item.get("operation") != "manual_visual_keep_window"
    ]
    history.append(
        {
            "timestamp": timestamp,
            "operation": "manual_visual_keep_window",
            "decision": decision,
            "source_dataset": str(source_dataset.resolve()),
            "source_trial": label,
            "input_frame_count": int(source_frames),
            "output_frame_count": int(hi - lo),
            "keep_bounds_in_input": [int(lo), int(hi)],
        }
    )
    trace["postprocessing_history"] = history
    trace["last_updated_at"] = timestamp

    # Refresh shapes in the original ProcessData output manifest. No hash is
    # stored here to avoid a self-referential trace file; file-level source and
    # output hashes live in Visual_Trim_Application.json.
    output_dir = path.parent
    noised_trace = path.stem.endswith("_noised")
    refreshed: dict[str, Any] = {}
    for recorded_name in trace.get("output_files", {}):
        candidate = output_dir / recorded_name
        if not candidate.exists() and noised_trace:
            recorded_path = Path(recorded_name)
            candidate = output_dir / (
                f"{recorded_path.stem}_noised{recorded_path.suffix}"
            )
        if not candidate.exists() or candidate.suffix != ".npy":
            continue
        arr = np.load(candidate, allow_pickle=True)
        if arr.dtype == object and arr.shape == () and isinstance(arr.item(), dict):
            refreshed[recorded_name] = {
                "shape": [],
                "dtype": "object",
                "dictionary_shapes": {
                    str(key): [int(v) for v in np.asarray(value).shape]
                    for key, value in arr.item().items()
                },
            }
        else:
            refreshed[recorded_name] = {
                "shape": [int(v) for v in arr.shape],
                "dtype": str(arr.dtype),
                "axes_matching_final_frame_count": [
                    int(axis)
                    for axis, size in enumerate(arr.shape)
                    if int(size) == int(hi - lo)
                ],
            }
    trace["output_files"] = refreshed
    write_json(path, trace)


def copy_subject_assets(source_subject: Path, output_subject: Path) -> None:
    output_subject.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_subject.iterdir()):
        if child.is_dir() and child.name.startswith("Trial_"):
            continue
        if child.name == "TempRemove":
            continue
        dst = output_subject / child.name
        if child.is_dir():
            shutil.copytree(child, dst)
        else:
            shutil.copy2(child, dst)


def build_trial(
    source_root: Path,
    output_root: Path,
    *,
    label: str,
    decision: str,
    window: dict[str, Any] | None,
    sample_rate_hz: float,
    source_review_json: Path,
    source_review_sha256: str,
    source_review_decision: dict[str, Any],
    source_visual_traceability_json: Path | None,
    source_visual_traceability_sha256: str | None,
    source_visual_trial_record: dict[str, Any] | None,
) -> dict[str, Any]:
    subject, trial = label.split("/", 1)
    source_trial = source_root / subject / trial
    output_trial = output_root / subject / trial
    source_proc = source_trial / "ProcessedData"
    source_motion = source_trial / "Motion"
    if not source_proc.is_dir() or not source_motion.is_dir():
        raise FileNotFoundError(f"Missing Motion or ProcessedData for {label}")

    ref_path = source_proc / "GRF_Cleaned.npy"
    source_frames = int(np.load(ref_path, mmap_mode="r").shape[0])
    if window is None:
        proc_lo, proc_hi = 0, source_frames
    else:
        declared_frames = int(window["source_frame_count"])
        proc_lo = int(window["keep_start_frame"])
        proc_hi = int(window["keep_end_frame_exclusive"])
        if declared_frames != source_frames:
            raise ValueError(
                f"{label}: visual window source_frame_count={declared_frames}, "
                f"current ProcessedData frames={source_frames}"
            )
        if not (0 <= proc_lo < proc_hi <= source_frames):
            raise ValueError(
                f"{label}: invalid processed bounds [{proc_lo}:{proc_hi}] for {source_frames}"
            )

    info_path = source_proc / "Trial_Processing_Information.json"
    info = load_json(info_path)
    raw_time = np.asarray(np.load(source_motion / "Time.npy")).reshape(-1)
    raw_frames = int(raw_time.size)
    if window is None:
        motion_lo, motion_hi = 0, raw_frames
        motion_start = float(raw_time[0])
        motion_end = float(raw_time[-1]) if raw_frames else 0.0
    else:
        motion_lo, motion_hi, motion_start, motion_end = motion_bounds(
            source_trial,
            info,
            proc_lo=proc_lo,
            proc_hi=proc_hi,
            sample_rate_hz=sample_rate_hz,
        )

    output_trial.mkdir(parents=True, exist_ok=True)
    motion_records = copy_motion_data(
        source_motion,
        output_trial / "Motion",
        raw_frames=raw_frames,
        lo=motion_lo,
        hi=motion_hi,
    )
    processed_records = copy_processed_data(
        source_proc,
        output_trial / "ProcessedData",
        source_frames=source_frames,
        lo=proc_lo,
        hi=proc_hi,
    )
    for name in ("Trial_Processing_Information.json", "Trial_Processing_Information_noised.json"):
        path = output_trial / "ProcessedData" / name
        if path.exists():
            update_processing_info(
                path,
                label=label,
                decision=decision,
                window=window,
                source_frames=source_frames,
                lo=proc_lo,
                hi=proc_hi,
                source_dataset=source_root,
            )
            for file_record in processed_records:
                if file_record["path"] == name:
                    file_record["output_sha256"] = sha256_file(path)
                    file_record["operation"] = "metadata_updated_after_slice"
                    break

    source_visual_trial_record_sha256 = (
        sha256_json_value(source_visual_trial_record)
        if source_visual_trial_record is not None
        else None
    )
    for name in ("Trimming_Traceability.json", "Trimming_Traceability_noised.json"):
        path = output_trial / "ProcessedData" / name
        if not path.exists():
            continue
        update_trimming_traceability(
            path,
            label=label,
            decision=decision,
            window=window,
            source_frames=source_frames,
            lo=proc_lo,
            hi=proc_hi,
            source_dataset=source_root,
            source_review_json=source_review_json,
            source_review_sha256=source_review_sha256,
            source_visual_traceability_json=source_visual_traceability_json,
            source_visual_traceability_sha256=source_visual_traceability_sha256,
            source_visual_trial_record_sha256=source_visual_trial_record_sha256,
            sample_rate_hz=sample_rate_hz,
        )
        for file_record in processed_records:
            if file_record["path"] == name:
                file_record["output_sha256"] = sha256_file(path)
                file_record["operation"] = "timeline_trace_composed_after_slice"
                break

    record = {
        "schema_version": "2.0",
        "trial": label,
        "decision": decision,
        "source_dataset": str(source_root.resolve()),
        "source_trial": str(source_trial.resolve()),
        "source_review_json": str(source_review_json.resolve()),
        "source_review_json_sha256": source_review_sha256,
        "source_review_decision": source_review_decision,
        "source_visual_traceability_json": (
            str(source_visual_traceability_json.resolve())
            if source_visual_traceability_json is not None
            else None
        ),
        "source_visual_traceability_json_sha256": source_visual_traceability_sha256,
        "source_visual_trial_record_sha256": source_visual_trial_record_sha256,
        "source_visual_selection_mapping": (
            source_visual_trial_record.get("selection_mapping")
            if source_visual_trial_record is not None
            else None
        ),
        "processed_bounds_half_open": [proc_lo, proc_hi],
        "processed_frames_before": source_frames,
        "processed_frames_after": proc_hi - proc_lo,
        "motion_bounds_half_open": [motion_lo, motion_hi],
        "motion_frames_before": raw_frames,
        "motion_frames_after": motion_hi - motion_lo,
        "motion_source_time_interval_s": [motion_start, motion_end],
        "visual_window": window,
        "processed_files": processed_records,
        "motion_files": motion_records,
    }
    write_json(output_trial / TRIAL_TRACE_NAME, record)
    return record


def validate_trial(output_root: Path, record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = record["trial"]
    trial = output_root / label
    proc = trial / "ProcessedData"
    expected = int(record["processed_frames_after"])
    required = (
        "GRF_Cleaned.npy",
        "ID_GT_MJX.npy",
        "qfrc_inverse.npy",
        "qfrc_grf_contribution.npy",
        "contactBoolean.npy",
        "pos_mjx.npy",
        "qvel_mjx.npy",
        "qacc_mjx.npy",
        "pos_inputs.npy",
        "vel_inputs.npy",
        "acc_inputs.npy",
        "Jacobian.npy",
    )
    for name in required:
        path = proc / name
        if not path.exists():
            errors.append(f"{label}: missing {name}")
            continue
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object and arr.shape == () and isinstance(arr.item(), dict):
            for key in ("jacp", "jacr"):
                value = np.asarray(arr.item().get(key))
                if value.shape[0] != expected:
                    errors.append(
                        f"{label}: {name}.{key} frames={value.shape[0]} expected={expected}"
                    )
                if not np.isfinite(value).all():
                    errors.append(f"{label}: {name}.{key} contains non-finite values")
            continue
        if arr.ndim < 1 or arr.shape[0] != expected:
            errors.append(f"{label}: {name} shape={arr.shape}, expected first axis {expected}")
        elif not np.isfinite(arr).all():
            errors.append(f"{label}: {name} contains non-finite values")
    id_path = proc / "ID_GT_MJX.npy"
    if id_path.exists() and np.load(id_path, mmap_mode="r").shape[1] != 23:
        errors.append(f"{label}: ID_GT_MJX.npy is not 23-DOF")
    for name in ("Trimming_Traceability.json", "Trimming_Traceability_noised.json"):
        path = proc / name
        if not path.exists():
            errors.append(f"{label}: missing {name}")
            continue
        trace = load_json(path)
        manual_stages = [
            item
            for item in trace.get("timeline_stages", [])
            if item.get("name") == "manual_visual_keep_window"
        ]
        if len(manual_stages) != 1:
            errors.append(f"{label}: {name} has {len(manual_stages)} manual visual stages")
        elif int(manual_stages[0].get("output_frame_count", -1)) != expected:
            errors.append(f"{label}: {name} manual visual stage has wrong output length")
        if int(trace.get("final_mapping", {}).get("final_frame_count", -1)) != expected:
            errors.append(f"{label}: {name} final_mapping has wrong output length")
    return errors


def build_dataset(
    source: Path,
    output: Path,
    *,
    review_path: Path,
    subject_prefix: str,
    sample_rate_hz: float,
    overwrite: bool,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    review = load_json(review_path)
    review_path = review_path.resolve()
    review_sha256 = sha256_file(review_path)
    visual_trace_path = source / TRACE_NAME
    visual_trace = load_json(visual_trace_path) if visual_trace_path.exists() else {}
    visual_trace_sha256 = (
        sha256_file(visual_trace_path) if visual_trace_path.exists() else None
    )
    windows = {
        label: window
        for label, window in review.get("trim_windows", {}).items()
        if label.split("/", 1)[0].startswith(subject_prefix)
    }
    keep = {
        label
        for label in review.get("keep_trials", [])
        if label.split("/", 1)[0].startswith(subject_prefix)
    }
    remove = {
        label
        for label in review.get("remove_trials", [])
        if label.split("/", 1)[0].startswith(subject_prefix)
    }
    selected = sorted(set(windows) | keep)
    if set(selected) & remove:
        raise ValueError("Review JSON places selected trials in remove_trials")
    if not selected:
        raise ValueError(f"No selected trials found for prefix {subject_prefix!r}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")

    source_trial_labels = {
        f"{subject.name}/{trial.name}"
        for subject in source.iterdir()
        if subject.is_dir()
        for trial in subject.iterdir()
        if trial.is_dir() and trial.name.startswith("Trial_")
    }
    missing_selected = sorted(set(selected) - source_trial_labels)
    if missing_selected:
        raise FileNotFoundError(
            f"{len(missing_selected)} selected trials are absent from the source; "
            f"first entries: {missing_selected[:10]}"
        )
    labeled = (
        set(review.get("keep_trials", []))
        | set(review.get("remove_trials", []))
        | set(review.get("needs_more_trimming_trials", []))
        | set(review.get("decisions", {}))
    )
    unclassified_source_trials = sorted(source_trial_labels - labeled)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building.", dir=output.parent)
    )
    try:
        subjects = sorted({label.split("/", 1)[0] for label in selected})
        for subject in subjects:
            copy_subject_assets(source / subject, temp_root / subject)

        records: list[dict[str, Any]] = []
        for index, label in enumerate(selected, start=1):
            window = windows.get(label)
            decision = "visual_trim" if window is not None else "keep_unchanged"
            records.append(
                build_trial(
                    source,
                    temp_root,
                    label=label,
                    decision=decision,
                    window=window,
                    sample_rate_hz=sample_rate_hz,
                    source_review_json=review_path,
                    source_review_sha256=review_sha256,
                    source_review_decision=review.get("decisions", {}).get(label, {}),
                    source_visual_traceability_json=(
                        visual_trace_path if visual_trace_path.exists() else None
                    ),
                    source_visual_traceability_sha256=visual_trace_sha256,
                    source_visual_trial_record=visual_trace.get("trials", {}).get(label),
                )
            )
            if index % 25 == 0 or index == len(selected):
                print(f"[build {index}/{len(selected)}] {label}", flush=True)

        validation_errors: list[str] = []
        for record in records:
            validation_errors.extend(validate_trial(temp_root, record))
        if validation_errors:
            preview = "\n".join(validation_errors[:30])
            raise ValueError(
                f"Validation failed with {len(validation_errors)} error(s):\n{preview}"
            )

        filtered_review = {
            "dataset_root": str(output),
            "source_dataset_root": str(source),
            "created_at": review.get("created_at"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "keep_trials": sorted(keep),
            "remove_trials": [],
            "needs_more_trimming_trials": sorted(windows),
            "trim_windows": {label: windows[label] for label in sorted(windows)},
            "decisions": {
                label: review.get("decisions", {}).get(label, {})
                for label in selected
            },
        }
        write_json(temp_root / REVIEW_NAME, filtered_review)
        source_trace = source / TRACE_NAME
        if source_trace.exists():
            shutil.copy2(source_trace, temp_root / TRACE_NAME)

        lengths = [int(record["processed_frames_after"]) for record in records]
        warnings = [
            {
                "trial": record["trial"],
                "processed_frames_after": int(record["processed_frames_after"]),
                "warning": "very_short_visual_selection",
            }
            for record in records
            if int(record["processed_frames_after"]) < 30
        ]
        manifest = {
            "schema_version": "2.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_dataset": str(source),
            "source_review_json": str(review_path),
            "source_review_json_sha256": review_sha256,
            "source_visual_traceability_json": (
                str(visual_trace_path.resolve()) if visual_trace_path.exists() else None
            ),
            "source_visual_traceability_json_sha256": visual_trace_sha256,
            "builder_script": str(Path(__file__).resolve()),
            "builder_script_sha256": sha256_file(Path(__file__).resolve()),
            "output_dataset": str(output),
            "subject_prefix": subject_prefix,
            "selection_rule": "trim_windows union keep_trials; remove_trials excluded",
            "sample_rate_hz": sample_rate_hz,
            "subject_count": len(subjects),
            "trial_count": len(records),
            "trimmed_trial_count": len(windows),
            "unchanged_keep_trial_count": len(keep - set(windows)),
            "excluded_remove_trial_count": len(remove),
            "excluded_remove_trials_present_in_source": len(remove & source_trial_labels),
            "excluded_remove_trials_absent_from_source": len(remove - source_trial_labels),
            "excluded_unclassified_source_trial_count": len(unclassified_source_trials),
            "excluded_unclassified_source_trials": unclassified_source_trials,
            "processed_frame_count_total": int(sum(lengths)),
            "processed_frame_count_min": int(min(lengths)),
            "processed_frame_count_max": int(max(lengths)),
            "warnings": warnings,
            "trials": [
                {
                    key: record[key]
                    for key in (
                        "trial",
                        "decision",
                        "processed_bounds_half_open",
                        "processed_frames_before",
                        "processed_frames_after",
                        "motion_bounds_half_open",
                        "motion_frames_before",
                        "motion_frames_after",
                    )
                }
                for record in records
            ],
        }
        write_json(temp_root / BUILD_MANIFEST_NAME, manifest)

        if output.exists():
            shutil.rmtree(output)
        temp_root.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def merge_into_destination(
    cleaned: Path,
    destination: Path,
    *,
    subject_prefix: str,
    replace_existing: bool,
) -> dict[str, Any]:
    cleaned = cleaned.resolve()
    destination = destination.resolve()
    manifest = load_json(cleaned / BUILD_MANIFEST_NAME)
    subjects = sorted(
        path.name
        for path in cleaned.iterdir()
        if path.is_dir() and path.name.startswith(subject_prefix)
    )
    collisions = [subject for subject in subjects if (destination / subject).exists()]
    if collisions and not replace_existing:
        raise FileExistsError(
            f"Destination already has {len(collisions)} subject(s): {collisions[:10]}"
        )

    merged: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects, start=1):
        src = cleaned / subject
        dst = destination / subject
        temp_dst = destination / f".{subject}.merging.{os.getpid()}"
        if temp_dst.exists():
            shutil.rmtree(temp_dst)
        shutil.copytree(src, temp_dst)
        if dst.exists():
            shutil.rmtree(dst)
        temp_dst.replace(dst)
        trial_count = len([p for p in dst.glob("Trial_*") if p.is_dir()])
        merged.append({"subject": subject, "trial_count": trial_count})
        print(f"[merge {index}/{len(subjects)}] {subject}: {trial_count} trials", flush=True)

    merge_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_cleaned_dataset": str(cleaned),
        "source_build_manifest": str((cleaned / BUILD_MANIFEST_NAME).resolve()),
        "destination_dataset": str(destination),
        "subject_prefix": subject_prefix,
        "subject_count": len(subjects),
        "trial_count": int(sum(item["trial_count"] for item in merged)),
        "cleaned_manifest_sha256": sha256_file(cleaned / BUILD_MANIFEST_NAME),
        "subjects": merged,
        "source_summary": {
            key: manifest.get(key)
            for key in (
                "trial_count",
                "trimmed_trial_count",
                "unchanged_keep_trial_count",
                "excluded_remove_trial_count",
                "processed_frame_count_total",
                "warnings",
            )
        },
    }
    write_json(destination / MERGE_MANIFEST_NAME, merge_manifest)
    return merge_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, default=None)
    parser.add_argument("--subject-prefix", default="PD_SUB")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge-destination", type=Path, default=None)
    parser.add_argument("--replace-existing-subjects", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    review_path = (
        args.review_json.resolve()
        if args.review_json is not None
        else source / REVIEW_NAME
    )
    manifest = build_dataset(
        source,
        args.output,
        review_path=review_path,
        subject_prefix=args.subject_prefix,
        sample_rate_hz=float(args.sample_rate_hz),
        overwrite=bool(args.overwrite),
    )
    print(
        f"Built {manifest['trial_count']} trials across "
        f"{manifest['subject_count']} subjects at {manifest['output_dataset']}"
    )
    if manifest["warnings"]:
        print(f"Warnings: {len(manifest['warnings'])} very short visual selections")

    if args.merge_destination is not None:
        merged = merge_into_destination(
            Path(manifest["output_dataset"]),
            args.merge_destination,
            subject_prefix=args.subject_prefix,
            replace_existing=bool(args.replace_existing_subjects),
        )
        print(
            f"Merged {merged['trial_count']} trials across "
            f"{merged['subject_count']} subjects into {merged['destination_dataset']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
