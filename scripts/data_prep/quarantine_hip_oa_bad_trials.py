#!/usr/bin/env python3
"""Quarantine clearly bad Hip OA trials with an auditable, reversible move.

The selection hierarchy is deliberately conservative about existing human work:

1. Trials explicitly listed in ``remove_trials`` are quarantined.
2. Trials explicitly retained or given a trim window are always protected.
3. Unlabelled trials are quarantined when sustained stance-phase COP distance
   exceeds the configured threshold.

Nothing is deleted. Trial directories are moved to the quarantine root while
preserving ``subject/trial`` paths, and a JSON manifest records every decision.
The script defaults to a dry run and refuses to apply moves while ProcessData is
running on the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from paths import artifact, dataset  # noqa: E402


SCHEMA_VERSION = "1.0"
RULE_VERSION = "hip_oa_sustained_cop_distance_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Hip_OA"))
    parser.add_argument(
        "--quarantine-root",
        type=Path,
        default=Path("Hip_OA_Quarantine"),
        help="Destination root; subject/trial paths are preserved.",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        default=None,
        help="Defaults to <data-root>/visual_cleaning_review.json.",
    )
    parser.add_argument(
        "--cop-distance-m",
        type=float,
        default=0.35,
        help="A loaded foot sample is bad above this COP-ankle distance.",
    )
    parser.add_argument(
        "--bad-loaded-fraction",
        type=float,
        default=0.10,
        help="Quarantine an unlabeled trial above this fraction of loaded foot samples.",
    )
    parser.add_argument(
        "--contact-force-n",
        type=float,
        default=20.0,
        help="A foot is loaded when its 3-D GRF magnitude exceeds this threshold.",
    )
    parser.add_argument(
        "--include-human-rejected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include trials explicitly listed in remove_trials.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move candidates. Without this flag, only report the dry run.",
    )
    parser.add_argument(
        "--allow-active-processdata",
        action="store_true",
        help="Unsafe override of the active ProcessData guard.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(str(artifact("output", "hip_oa_quarantine"))),
        help="Dry-run and applied manifests are written here.",
    )
    return parser.parse_args()


def resolve_from_cwd(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_review(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def active_dataset_processes(data_root: Path) -> list[dict[str, Any]]:
    """Return processors/cleaners that may be reading or writing this dataset."""
    matches: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    root_text = str(data_root)
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = " ".join(argv)
        is_processor_or_cleaner = any(
            token.endswith("ProcessData.py")
            or token.endswith("run_processdata_gpu_shards.py")
            or token.endswith("InteractiveDatasetCleaner.py")
            for token in argv
        )
        references_dataset = root_text in command or (
            "--data-root" in argv
            and any(
                value == data_root.name or Path(value).name == data_root.name
                for value in argv
            )
        )
        if is_processor_or_cleaner and references_dataset:
            matches.append({"pid": int(entry.name), "argv": argv})
    return sorted(matches, key=lambda item: item["pid"])


def trial_metrics(
    processed_dir: Path,
    contact_force_n: float,
    cop_distance_m: float,
) -> dict[str, Any]:
    grf_path = processed_dir / "GRF_Cleaned.npy"
    cop_path = processed_dir / "COP_Cleaned_Relative.npy"
    grf = np.asarray(np.load(grf_path), dtype=np.float64)
    cop = np.asarray(np.load(cop_path), dtype=np.float64)
    if grf.ndim != 2 or grf.shape[1] < 6:
        raise ValueError(f"Expected GRF shape (T, >=6), got {grf.shape}")
    if cop.ndim != 2 or cop.shape[1] < 4:
        raise ValueError(f"Expected relative COP shape (T, >=4), got {cop.shape}")
    n_frames = min(grf.shape[0], cop.shape[0])
    grf = grf[:n_frames, :6]
    cop = cop[:n_frames, :4]
    if not np.isfinite(grf).all() or not np.isfinite(cop).all():
        raise ValueError("GRF or COP contains non-finite values")

    bad_count = 0
    loaded_count = 0
    sides: dict[str, Any] = {}
    for side, grf_slice, cop_slice in (
        ("right", slice(0, 3), slice(0, 2)),
        ("left", slice(3, 6), slice(2, 4)),
    ):
        force_magnitude = np.linalg.norm(grf[:, grf_slice], axis=1)
        loaded = force_magnitude > contact_force_n
        cop_distance = np.linalg.norm(cop[:, cop_slice], axis=1)
        loaded_distances = cop_distance[loaded]
        side_bad = int(np.count_nonzero(loaded_distances > cop_distance_m))
        side_loaded = int(loaded_distances.size)
        bad_count += side_bad
        loaded_count += side_loaded
        sides[side] = {
            "loaded_samples": side_loaded,
            "bad_loaded_samples": side_bad,
            "bad_loaded_fraction": (
                float(side_bad / side_loaded) if side_loaded else None
            ),
            "cop_distance_p95_m": (
                float(np.quantile(loaded_distances, 0.95))
                if side_loaded else None
            ),
            "cop_distance_p99_m": (
                float(np.quantile(loaded_distances, 0.99))
                if side_loaded else None
            ),
            "cop_distance_max_m": (
                float(np.max(loaded_distances)) if side_loaded else None
            ),
        }

    return {
        "frame_count": int(n_frames),
        "loaded_foot_samples": int(loaded_count),
        "bad_loaded_foot_samples": int(bad_count),
        "bad_loaded_fraction": (
            float(bad_count / loaded_count) if loaded_count else 1.0
        ),
        "sides": sides,
        "source_files": {
            "grf": {
                "path": str(grf_path),
                "shape": list(grf.shape),
                "sha256": sha256_file(grf_path),
            },
            "cop_relative": {
                "path": str(cop_path),
                "shape": list(cop.shape),
                "sha256": sha256_file(cop_path),
            },
        },
    }


def directory_summary(path: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def main() -> int:
    args = parse_args()
    data_root = resolve_from_cwd(args.data_root)
    quarantine_root = resolve_from_cwd(args.quarantine_root)
    report_dir = resolve_from_cwd(args.report_dir)
    review_path = (
        resolve_from_cwd(args.review_json)
        if args.review_json is not None
        else data_root / "visual_cleaning_review.json"
    )
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not review_path.is_file():
        raise FileNotFoundError(review_path)
    if data_root == quarantine_root:
        raise ValueError("The source and quarantine roots must be different")
    if not (0.0 <= args.bad_loaded_fraction <= 1.0):
        raise ValueError("--bad-loaded-fraction must be between 0 and 1")
    if args.cop_distance_m <= 0 or args.contact_force_n < 0:
        raise ValueError("COP distance must be positive and contact force nonnegative")

    active = active_dataset_processes(data_root)
    if args.apply and active and not args.allow_active_processdata:
        pids = ", ".join(str(item["pid"]) for item in active)
        raise RuntimeError(
            "Refusing to move trial folders while ProcessData or the interactive "
            "cleaner is active on "
            f"{data_root} (PIDs: {pids}). Stop it cleanly and rerun."
        )

    review = read_review(review_path)
    human_remove = set(map(str, review.get("remove_trials", [])))
    human_protected = set(map(str, review.get("keep_trials", [])))
    human_protected.update(map(str, review.get("needs_more_trimming_trials", [])))
    human_protected.update(map(str, review.get("trim_windows", {}).keys()))
    overlap = human_remove & human_protected
    if overlap:
        raise ValueError(
            "Review JSON contains trials in both remove and retained buckets: "
            + ", ".join(sorted(overlap)[:10])
        )

    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    protected_count = 0
    analyzable_unlabelled = 0
    for subject_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if subject_dir.name.startswith("."):
            continue
        for trial_dir in sorted(
            path for path in subject_dir.iterdir()
            if path.is_dir() and path.name.startswith("Trial_")
        ):
            label = f"{subject_dir.name}/{trial_dir.name}"
            if label in human_protected:
                protected_count += 1
                continue
            processed_dir = trial_dir / "ProcessedData"
            required = (
                processed_dir / "GRF_Cleaned.npy",
                processed_dir / "COP_Cleaned_Relative.npy",
            )
            is_human_reject = label in human_remove and args.include_human_rejected
            if not all(path.is_file() for path in required):
                if is_human_reject:
                    metrics = None
                else:
                    continue
            else:
                try:
                    metrics = trial_metrics(
                        processed_dir,
                        contact_force_n=float(args.contact_force_n),
                        cop_distance_m=float(args.cop_distance_m),
                    )
                except Exception as exc:
                    unreadable.append({"trial": label, "error": str(exc)})
                    continue

            if not is_human_reject:
                analyzable_unlabelled += 1
                assert metrics is not None
                if metrics["bad_loaded_fraction"] <= args.bad_loaded_fraction:
                    continue
                reason = "automated_sustained_cop_distance"
            else:
                reason = "human_remove_decision"

            destination = quarantine_root / subject_dir.name / trial_dir.name
            records.append({
                "trial": label,
                "reason": reason,
                "source": str(trial_dir),
                "destination": str(destination),
                "metrics": metrics,
                "source_directory": directory_summary(trial_dir),
                "move_status": "planned",
            })

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / f"quarantine_{stamp}.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry_run",
        "source_root": str(data_root),
        "quarantine_root": str(quarantine_root),
        "review_json": {
            "path": str(review_path),
            "sha256": sha256_file(review_path),
            "updated_at": review.get("updated_at"),
        },
        "rule": {
            "loaded_foot_definition": "3-D GRF magnitude > contact_force_n",
            "contact_force_n": float(args.contact_force_n),
            "bad_cop_definition": "COP-ankle relative distance > cop_distance_m",
            "cop_distance_m": float(args.cop_distance_m),
            "quarantine_when_bad_loaded_fraction_gt": float(args.bad_loaded_fraction),
            "human_rejected_included": bool(args.include_human_rejected),
            "human_retained_always_protected": True,
        },
        "active_processdata": active,
        "counts": {
            "human_protected": protected_count,
            "human_rejected_in_review": len(human_remove),
            "analyzable_unlabelled": analyzable_unlabelled,
            "candidates": len(records),
            "human_rejected_candidates": sum(
                item["reason"] == "human_remove_decision" for item in records
            ),
            "automated_candidates": sum(
                item["reason"] == "automated_sustained_cop_distance"
                for item in records
            ),
            "unreadable": len(unreadable),
            "moved": 0,
            "failed": 0,
        },
        "unreadable": unreadable,
        "trials": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.apply:
        print(f"Dry run: {len(records)} quarantine candidates")
        print(
            "  human rejected: "
            f"{manifest['counts']['human_rejected_candidates']}"
        )
        print(
            "  automated sustained-COP: "
            f"{manifest['counts']['automated_candidates']}"
        )
        print(f"  human retained protected: {protected_count}")
        print(f"  active ProcessData processes: {len(active)}")
        print(f"Manifest: {manifest_path}")
        return 0

    # Validate every destination before the first move so a collision cannot
    # leave the requested batch half-applied.
    collisions = [
        item["destination"]
        for item in records
        if Path(item["destination"]).exists()
    ]
    if collisions:
        raise FileExistsError(
            "Quarantine destination collision(s); no moves performed: "
            + ", ".join(collisions[:10])
        )

    quarantine_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    failed = 0
    for item in records:
        source = Path(item["source"])
        destination = Path(item["destination"])
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            if source.exists() or not destination.is_dir():
                raise RuntimeError("post-move source/destination validation failed")
            item["move_status"] = "moved"
            item["moved_at"] = datetime.now().isoformat(timespec="seconds")
            moved += 1
        except Exception as exc:
            item["move_status"] = "failed"
            item["move_error"] = str(exc)
            failed += 1
        manifest["counts"]["moved"] = moved
        manifest["counts"]["failed"] = failed
        # Persist after each move so interruption cannot erase traceability.
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Moved {moved}/{len(records)} trials to {quarantine_root}")
    print(f"Failed: {failed}")
    print(f"Manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
