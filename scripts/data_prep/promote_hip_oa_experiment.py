#!/usr/bin/env python3
"""Filter and promote the visually cleaned Hip OA dataset as an experiment.

The requested outlier rule is deterministic:

* a foot is missing when its cleaned vertical GRF never exceeds 15 N;
* candidates are trials missing either foot;
* remove ``ceil(fraction * candidate_count)`` longest candidates, with trial
  label as the deterministic tie-breaker.

Removed trials are moved to a recoverable quarantine. The retained dataset is
then atomically moved beneath ``TrustedDataSet_ByExperiment/Hip_OA``. JSON and
CSV reports preserve the metrics, decisions, source manifests, and paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from paths import artifact, dataset  # noqa: E402


REPORT_NAME = "hip_oa_readiness_outlier_report.json"
REPORT_CSV_NAME = "hip_oa_trial_outlier_metrics.csv"
QUARANTINE_MANIFEST_NAME = "hip_oa_one_foot_longest5pct_quarantine_manifest.json"
EXPERIMENT_MANIFEST_NAME = "hip_oa_experiment_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def trial_dirs(root: Path) -> list[Path]:
    return sorted(
        trial
        for subject in root.iterdir()
        if subject.is_dir() and not subject.name.startswith(".")
        for trial in subject.glob("Trial_*")
        if trial.is_dir()
    )


def trial_label(root: Path, trial: Path) -> str:
    return trial.relative_to(root).as_posix()


def percentile_summary(values: list[float]) -> dict[str, float]:
    points = (0, 1, 5, 50, 95, 99, 100)
    result = np.percentile(np.asarray(values, dtype=np.float64), points)
    return {
        name: float(value)
        for name, value in zip(("min", "p01", "p05", "p50", "p95", "p99", "max"), result)
    }


def analyze_trial(root: Path, trial: Path, threshold_n: float) -> dict[str, Any]:
    processed = trial / "ProcessedData"
    grf = np.asarray(np.load(processed / "GRF_Cleaned.npy"), dtype=np.float64)
    ident = np.asarray(np.load(processed / "ID_GT_MJX.npy"), dtype=np.float64)
    cop = np.asarray(
        np.load(processed / "COP_Cleaned_Relative.npy"), dtype=np.float64
    )
    mass = float(np.nanmedian(np.load(processed / "Mass_kg.npy")))
    frames = int(grf.shape[0])
    right_vertical = grf[:, 2]
    left_vertical = grf[:, 5]
    right_contact = int(np.count_nonzero(right_vertical > threshold_n))
    left_contact = int(np.count_nonzero(left_vertical > threshold_n))
    body_weight = max(mass * 9.80665, 1e-9)
    return {
        "trial": trial_label(root, trial),
        "frames": frames,
        "duration_s": frames / 100.0,
        "mass_kg": mass,
        "right_contact_frames": right_contact,
        "left_contact_frames": left_contact,
        "missing_right_foot_grf": right_contact == 0,
        "missing_left_foot_grf": left_contact == 0,
        "right_peak_vertical_grf_bw": float(np.max(right_vertical) / body_weight),
        "left_peak_vertical_grf_bw": float(np.max(left_vertical) / body_weight),
        "id_abs_max_nm": float(np.max(np.abs(ident))),
        "knee_ankle_abs_max_nm": float(
            np.max(np.abs(ident[:, [9, 10, 16, 17]]))
        ),
        "cop_relative_abs_max_m": float(np.max(np.abs(cop))),
    }


def build_report(
    source: Path,
    *,
    threshold_n: float,
    remove_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = [analyze_trial(source, trial, threshold_n) for trial in trial_dirs(source)]
    one_foot = [
        row
        for row in metrics
        if row["missing_right_foot_grf"] or row["missing_left_foot_grf"]
    ]
    ranked = sorted(one_foot, key=lambda row: (-int(row["frames"]), row["trial"]))
    removal_count = math.ceil(remove_fraction * len(ranked)) if ranked else 0
    remove_labels = {row["trial"] for row in ranked[:removal_count]}
    for row in metrics:
        row["one_foot_candidate"] = bool(
            row["missing_right_foot_grf"] or row["missing_left_foot_grf"]
        )
        row["selected_for_removal"] = row["trial"] in remove_labels

    distributions = {
        key: percentile_summary([float(row[key]) for row in metrics])
        for key in (
            "frames",
            "right_peak_vertical_grf_bw",
            "left_peak_vertical_grf_bw",
            "id_abs_max_nm",
            "knee_ankle_abs_max_nm",
            "cop_relative_abs_max_m",
        )
    }
    # Pattern outliers are reported, not automatically removed.
    p99_knee_ankle = distributions["knee_ankle_abs_max_nm"]["p99"]
    p99_cop = distributions["cop_relative_abs_max_m"]["p99"]
    p99_id = distributions["id_abs_max_nm"]["p99"]
    p99_frames = distributions["frames"]["p99"]
    pattern_outliers = {
        "length_above_p99": [
            row["trial"] for row in metrics if row["frames"] > p99_frames
        ],
        "knee_ankle_abs_max_above_p99": [
            row["trial"]
            for row in metrics
            if row["knee_ankle_abs_max_nm"] > p99_knee_ankle
        ],
        "cop_relative_abs_max_above_p99": [
            row["trial"]
            for row in metrics
            if row["cop_relative_abs_max_m"] > p99_cop
        ],
        "full_id_abs_max_above_p99": [
            row["trial"] for row in metrics if row["id_abs_max_nm"] > p99_id
        ],
    }
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(source.resolve()),
        "trial_count_before_filter": len(metrics),
        "subject_count_before_filter": len(
            {row["trial"].split("/", 1)[0] for row in metrics}
        ),
        "one_foot_definition": (
            "A foot is missing when GRF_Cleaned vertical GRF has zero frames "
            f"strictly above {threshold_n:g} N; right vertical column=2, left=5."
        ),
        "one_foot_contact_threshold_n": threshold_n,
        "one_foot_candidate_count": len(one_foot),
        "missing_right_candidate_count": sum(
            bool(row["missing_right_foot_grf"]) for row in one_foot
        ),
        "missing_left_candidate_count": sum(
            bool(row["missing_left_foot_grf"]) for row in one_foot
        ),
        "both_feet_missing_candidate_count": sum(
            bool(
                row["missing_right_foot_grf"]
                and row["missing_left_foot_grf"]
            )
            for row in one_foot
        ),
        "removal_rule": (
            "Sort one-foot candidates by frames descending, then label ascending; "
            "remove ceil(remove_fraction * candidate_count)."
        ),
        "remove_fraction": remove_fraction,
        "selected_removal_count": removal_count,
        "selected_removal_trials": ranked[:removal_count],
        "retained_trial_count": len(metrics) - removal_count,
        "distributions": distributions,
        "reported_pattern_outliers": pattern_outliers,
        "training_length_counts": {
            "at_least_30_frames": sum(row["frames"] >= 30 for row in metrics),
            "greater_than_64_frames": sum(row["frames"] > 64 for row in metrics),
            "less_than_30_frames": sum(row["frames"] < 30 for row in metrics),
        },
    }
    return report, metrics


def write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def filter_review_json(
    dataset: Path,
    *,
    removed_labels: set[str],
    event: dict[str, Any],
) -> None:
    path = dataset / "visual_cleaning_review.json"
    if not path.exists():
        return
    backup = dataset / "visual_cleaning_review_pre_one_foot_filter.json"
    if backup.exists():
        raise FileExistsError(f"Review backup already exists: {backup}")
    shutil.copy2(path, backup)
    review = json.loads(path.read_text(encoding="utf-8"))
    for key in ("keep_trials", "remove_trials", "needs_more_trimming_trials"):
        review[key] = [
            label for label in review.get(key, []) if label not in removed_labels
        ]
    review["trim_windows"] = {
        label: value
        for label, value in review.get("trim_windows", {}).items()
        if label not in removed_labels
    }
    review["decisions"] = {
        label: value
        for label, value in review.get("decisions", {}).items()
        if label not in removed_labels
    }
    review["dataset_root"] = str(dataset.resolve())
    review.setdefault("post_review_exclusions", []).append(event)
    review["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(path, review)


def quarantine_subject_assets_without_trials(
    dataset: Path,
    quarantine: Path,
) -> list[dict[str, Any]]:
    """Move asset-only subject folders out of the training experiment."""
    moved: list[dict[str, Any]] = []
    for subject in sorted(path for path in dataset.iterdir() if path.is_dir()):
        if any(trial.is_dir() for trial in subject.glob("Trial_*")):
            continue
        quarantine_subject = quarantine / subject.name
        quarantine_subject.mkdir(parents=True, exist_ok=True)
        assets = []
        for child in sorted(subject.iterdir()):
            destination = quarantine_subject / child.name
            if destination.exists():
                raise FileExistsError(
                    f"Cannot quarantine subject asset; destination exists: {destination}"
                )
            shutil.move(str(child), str(destination))
            assets.append(child.name)
        subject.rmdir()
        moved.append(
            {
                "subject": subject.name,
                "source_path": str(subject),
                "quarantine_path": str(quarantine_subject),
                "assets": assets,
                "reason": "no retained Trial_* directories after one-foot filter",
            }
        )
    return moved


def update_build_manifest(
    dataset: Path,
    *,
    removed_labels: set[str],
    report_sha256: str,
    event: dict[str, Any],
) -> None:
    path = dataset / "visual_trimmed_dataset_manifest.json"
    if not path.exists():
        return
    backup = dataset / "visual_trimmed_dataset_manifest_pre_one_foot_filter.json"
    if backup.exists():
        raise FileExistsError(f"Manifest backup already exists: {backup}")
    shutil.copy2(path, backup)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    old_trials = manifest.get("trials", [])
    removed_records = [
        row for row in old_trials if row.get("trial") in removed_labels
    ]
    retained_records = [
        row for row in old_trials if row.get("trial") not in removed_labels
    ]
    manifest["trials"] = retained_records
    manifest["trial_count"] = len(retained_records)
    manifest["subject_count"] = len(
        {row["trial"].split("/", 1)[0] for row in retained_records}
    )
    lengths = [int(row["processed_frames_after"]) for row in retained_records]
    manifest["processed_frame_count_total"] = int(sum(lengths))
    manifest["processed_frame_count_min"] = int(min(lengths))
    manifest["processed_frame_count_max"] = int(max(lengths))
    event = dict(event)
    event["analysis_report_sha256"] = report_sha256
    event["removed_trial_manifest_records"] = removed_records
    manifest.setdefault("post_build_filter_history", []).append(event)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Hip_OA_Cleaned"))
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(str(dataset("TrustedDataSet_ByExperiment", "Hip_OA"))),
    )
    parser.add_argument(
        "--quarantine",
        type=Path,
        default=Path("Hip_OA_Excluded_Longest5Pct_OneFoot"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(str(artifact("output", "hip_oa_promotion"))),
    )
    parser.add_argument("--contact-threshold-n", type=float, default=15.0)
    parser.add_argument("--remove-fraction", type=float, default=0.05)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    quarantine = args.quarantine.resolve()
    report_dir = args.report_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not (0.0 <= args.remove_fraction <= 1.0):
        raise ValueError("--remove-fraction must be between 0 and 1")

    report, metrics = build_report(
        source,
        threshold_n=float(args.contact_threshold_n),
        remove_fraction=float(args.remove_fraction),
    )
    report_path = report_dir / REPORT_NAME
    csv_path = report_dir / REPORT_CSV_NAME
    write_json(report_path, report)
    write_metrics_csv(csv_path, metrics)
    print(
        f"Analyzed {report['trial_count_before_filter']} trials; "
        f"one-foot candidates={report['one_foot_candidate_count']}; "
        f"selected={report['selected_removal_count']}"
    )
    print(f"Report: {report_path}")
    if not args.apply:
        print("Dry run: no trials moved and dataset not promoted.")
        return 0

    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if quarantine.exists():
        raise FileExistsError(f"Quarantine already exists: {quarantine}")
    selected = report["selected_removal_trials"]
    removed_labels = {row["trial"] for row in selected}
    quarantine.mkdir(parents=True)
    moved: list[dict[str, Any]] = []
    for row in selected:
        label = row["trial"]
        src_trial = source / label
        dst_trial = quarantine / label
        if not src_trial.is_dir():
            raise FileNotFoundError(src_trial)
        dst_trial.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_trial), str(dst_trial))
        trace_path = dst_trial / "Visual_Trim_Application.json"
        moved.append(
            {
                **row,
                "source_path_before_filter": str(src_trial),
                "quarantine_path": str(dst_trial),
                "visual_trim_application_sha256": (
                    sha256_file(trace_path) if trace_path.exists() else None
                ),
            }
        )
    quarantined_empty_subjects = quarantine_subject_assets_without_trials(
        source, quarantine
    )

    quarantine_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "recoverable": True,
        "source_dataset_before_promotion": str(source),
        "intended_destination_dataset": str(destination),
        "criterion": report["one_foot_definition"],
        "removal_rule": report["removal_rule"],
        "remove_fraction": report["remove_fraction"],
        "candidate_count": report["one_foot_candidate_count"],
        "removed_trial_count": len(moved),
        "subjects_with_no_retained_trials": quarantined_empty_subjects,
        "analysis_report": str(report_path),
        "analysis_report_sha256": sha256_file(report_path),
        "trials": moved,
    }
    write_json(quarantine / QUARANTINE_MANIFEST_NAME, quarantine_manifest)
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operation": "exclude_longest_five_percent_of_one_foot_grf_trials",
        "criterion": report["one_foot_definition"],
        "remove_fraction": report["remove_fraction"],
        "candidate_count": report["one_foot_candidate_count"],
        "removed_trial_count": len(removed_labels),
        "removed_trials": sorted(removed_labels),
        "quarantine": str(quarantine),
        "quarantine_manifest": str(
            quarantine / QUARANTINE_MANIFEST_NAME
        ),
    }
    filter_review_json(source, removed_labels=removed_labels, event=event)
    update_build_manifest(
        source,
        removed_labels=removed_labels,
        report_sha256=sha256_file(report_path),
        event=event,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    final_report = dict(report)
    final_report.update(
        {
            "applied": True,
            "destination_dataset": str(destination),
            "quarantine_dataset": str(quarantine),
            "trial_count_after_filter": len(trial_dirs(destination)),
            "subject_count_after_filter": len(
                {
                    trial.parent.name
                    for trial in trial_dirs(destination)
                }
            ),
        }
    )
    write_json(destination / REPORT_NAME, final_report)
    write_metrics_csv(destination / REPORT_CSV_NAME, metrics)

    # Correct current-location fields while preserving source-location history.
    review_path = destination / "visual_cleaning_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["dataset_root"] = str(destination)
        review.setdefault("relocation_history", []).append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "from": str(source),
                "to": str(destination),
            }
        )
        write_json(review_path, review)
    build_manifest_path = destination / "visual_trimmed_dataset_manifest.json"
    if build_manifest_path.exists():
        manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        manifest["output_dataset"] = str(destination)
        manifest.setdefault("relocation_history", []).append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "from": str(source),
                "to": str(destination),
            }
        )
        write_json(build_manifest_path, manifest)

    experiment_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "Hip_OA",
        "dataset": str(destination),
        "layout": str(dataset("TrustedDataSet_ByExperiment", "<Experiment>", "<Subject>", "Trial_*")),
        "subject_count": final_report["subject_count_after_filter"],
        "trial_count": final_report["trial_count_after_filter"],
        "source_dataset_before_move": str(source),
        "visual_trim_manifest": str(build_manifest_path),
        "visual_trim_manifest_sha256": sha256_file(build_manifest_path),
        "outlier_report": str(destination / REPORT_NAME),
        "outlier_report_sha256": sha256_file(destination / REPORT_NAME),
        "quarantine": str(quarantine),
        "quarantine_manifest": str(
            quarantine / QUARANTINE_MANIFEST_NAME
        ),
    }
    write_json(destination.parent / EXPERIMENT_MANIFEST_NAME, experiment_manifest)
    print(
        f"Promoted {final_report['trial_count_after_filter']} trials across "
        f"{final_report['subject_count_after_filter']} subjects to {destination}"
    )
    print(f"Quarantined {len(moved)} trials at {quarantine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
