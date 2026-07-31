"""Aggregation and machine-readable reports for paired LOSO inference."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


def _finite(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cohort(subject: str, mapping: Optional[Mapping[str, str]]) -> str:
    if mapping and subject in mapping:
        return str(mapping[subject])
    upper = subject.upper()
    return "older" if upper.startswith("OA") else ("younger" if upper.startswith("Y") else "unknown")


def _trial_row(metric: Mapping[str, Any], cohort_mapping: Optional[Mapping[str, str]]) -> Dict[str, Any]:
    subject = str(metric.get("subject") or str(metric.get("trial", "")).split("/")[0])
    original = metric.get("original", {}).get("torque", {}).get("mae")
    fine = metric.get("fine_tuned", {}).get("torque", {}).get("mae")
    return {
        "subject": subject,
        "cohort": _cohort(subject, cohort_mapping),
        "trial": metric.get("trial"),
        "model_structure": metric.get("model_structure"),
        "n_eval_frames": metric.get("n_eval_frames", 0),
        "original_torque_mae": original,
        "fine_tuned_torque_mae": fine,
        "torque_mae_change": None if original is None or fine is None else fine - original,
        "torque_mae_improvement_percent": None if not original or fine is None else 100.0 * (original - fine) / original,
    }


def _mean(values: Iterable[Any]) -> Optional[float]:
    kept = [float(v) for v in values if v is not None and np.isfinite(v)]
    return None if not kept else float(np.mean(kept))


def build_loso_summary(
    trial_metrics: Iterable[Mapping[str, Any]],
    *,
    dataset_path: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
    fold_checkpoint: Optional[str] = None,
    cohort_mapping: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Aggregate trials with equal subject weighting for cohort/overall results."""
    trial_metrics = list(trial_metrics)
    rows = [_trial_row(m, cohort_mapping) for m in trial_metrics]
    subjects: List[Dict[str, Any]] = []
    for subject in sorted({str(row["subject"]) for row in rows}):
        selected = [row for row in rows if row["subject"] == subject]
        subjects.append({
            "subject": subject,
            "cohort": selected[0]["cohort"],
            "n_trials": len(selected),
            "original_torque_mae": _mean(r["original_torque_mae"] for r in selected),
            "fine_tuned_torque_mae": _mean(r["fine_tuned_torque_mae"] for r in selected),
            "torque_mae_change": _mean(r["torque_mae_change"] for r in selected),
            "torque_mae_improvement_percent": _mean(r["torque_mae_improvement_percent"] for r in selected),
        })
    cohorts: Dict[str, Any] = {}
    for name in sorted({str(row["cohort"]) for row in subjects}):
        selected = [row for row in subjects if row["cohort"] == name]
        cohorts[name] = {
            "n_subjects": len(selected),
            "original_torque_mae": _mean(r["original_torque_mae"] for r in selected),
            "fine_tuned_torque_mae": _mean(r["fine_tuned_torque_mae"] for r in selected),
            "torque_mae_change": _mean(r["torque_mae_change"] for r in selected),
            "torque_mae_improvement_percent": _mean(r["torque_mae_improvement_percent"] for r in selected),
        }
    overall = {
        "n_subjects": len(subjects),
        "n_trials": len(rows),
        "original_torque_mae": _mean(r["original_torque_mae"] for r in subjects),
        "fine_tuned_torque_mae": _mean(r["fine_tuned_torque_mae"] for r in subjects),
        "torque_mae_change": _mean(r["torque_mae_change"] for r in subjects),
        "torque_mae_improvement_percent": _mean(r["torque_mae_improvement_percent"] for r in subjects),
    }
    structures = sorted({str(m.get("model_structure")) for m in trial_metrics if m.get("model_structure")})
    return _finite({
        "dataset_path": dataset_path,
        "source_checkpoint": source_checkpoint,
        "fold_checkpoint": fold_checkpoint,
        "model_structures": structures,
        "overall": overall,
        "cohorts": cohorts,
        "subjects": subjects,
        "trials": rows,
    })


def write_loso_reports(
    output_dir: Path | str,
    trial_metrics: Iterable[Mapping[str, Any]],
    **metadata: Any,
) -> Dict[str, Any]:
    """Write JSON plus trial/subject CSV reports and return the summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = list(trial_metrics)
    summary = build_loso_summary(metrics, **metadata)
    with (output / "loso_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    for filename, rows in (("trial_metrics.csv", summary["trials"]), ("subject_metrics.csv", summary["subjects"])):
        if not rows:
            continue
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    flat = [{"scope": "overall", "group": "all", **summary["overall"]}]
    flat.extend({"scope": "cohort", "group": name, **values} for name, values in summary["cohorts"].items())
    with (output / "loso_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader()
        writer.writerows(flat)
    return summary


__all__ = ["build_loso_summary", "write_loso_reports"]
