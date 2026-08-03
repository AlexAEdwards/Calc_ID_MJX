#!/usr/bin/env python3
"""Stage robust PD trials with selected QC criteria for ProcessData.py.

The source PD dataset is never modified. RobustExtracted_v2 files are copied
into the staged trial's Motion directory, while subject models and Geometry
are copied once per subject.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from paths import artifact, dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(str(dataset("Datasets_Local", "PD_Dataset"))),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(str(dataset("Datasets_Local", "PD_Dataset", "RobustExtracted_v2_manifest.json"))),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(str(dataset("Datasets_Local", "PD_Robust_PASS_All"))),
    )
    parser.add_argument("--status", default="PASS", choices=("PASS", "REVIEW", "REJECT"))
    parser.add_argument("--robust-dir-name", default="RobustExtracted_v2")
    parser.add_argument(
        "--require-bilateral-contacts",
        action="store_true",
        help="Keep only trials with at least one accepted right and left contact event.",
    )
    parser.add_argument(
        "--min-excluded-peak-n",
        type=float,
        default=None,
        help="Keep trials whose largest excluded contact peak is at least this value.",
    )
    parser.add_argument(
        "--max-excluded-peak-n",
        type=float,
        default=None,
        help="Keep trials whose largest excluded contact peak is strictly below this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the selected trial count without copying files.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    status_selected = sorted(
        item["trial"]
        for item in manifest["trials"]
        if item["qc_status"] == args.status
    )
    selected = []
    for trial_id in status_selected:
        subject, trial = trial_id.split("/", 1)
        metadata_path = (
            args.source_root / subject / trial / "Motion" /
            args.robust_dir_name / "extraction_metadata.json"
        )
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("qc_status") != args.status:
            raise ValueError(
                f"{trial_id}: manifest says {args.status}, trial metadata says "
                f"{metadata.get('qc_status')}"
            )

        events = metadata.get("contact_events", [])
        accepted_sides = {
            event.get("assignment")
            for event in events
            if event.get("assignment") in {"right", "left"}
        }
        excluded_peaks = [
            float(event.get("peak_vertical_n", 0.0))
            for event in events
            if event.get("assignment") not in {"right", "left"}
        ]
        max_excluded_peak_n = max(excluded_peaks, default=0.0)

        if args.require_bilateral_contacts and accepted_sides != {"right", "left"}:
            continue
        if (
            args.min_excluded_peak_n is not None and
            max_excluded_peak_n < args.min_excluded_peak_n
        ):
            continue
        if (
            args.max_excluded_peak_n is not None and
            max_excluded_peak_n >= args.max_excluded_peak_n
        ):
            continue
        selected.append(trial_id)

    if not selected:
        raise SystemExit(f"No {args.status} trials found in {args.manifest}")

    criteria = {
        "qc_status": args.status,
        "require_bilateral_contacts": args.require_bilateral_contacts,
        "min_excluded_peak_n_inclusive": args.min_excluded_peak_n,
        "max_excluded_peak_n_exclusive": args.max_excluded_peak_n,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source_manifest": str(args.manifest),
                    "criteria": criteria,
                    "trials": len(selected),
                    "all_ids": selected,
                },
                indent=2,
            )
        )
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    staged_subjects: set[str] = set()

    for index, trial_id in enumerate(selected, start=1):
        subject, trial = trial_id.split("/", 1)
        source_subject = args.source_root / subject
        staged_subject = args.output_root / subject

        if subject not in staged_subjects:
            staged_subject.mkdir(parents=True, exist_ok=True)
            for source_file in source_subject.iterdir():
                if source_file.is_file():
                    shutil.copy2(source_file, staged_subject / source_file.name)
            source_geometry = source_subject / "Geometry"
            if source_geometry.is_dir():
                shutil.copytree(
                    source_geometry,
                    staged_subject / "Geometry",
                    dirs_exist_ok=True,
                )
            staged_subjects.add(subject)

        robust_source = source_subject / trial / "Motion" / args.robust_dir_name
        staged_motion = staged_subject / trial / "Motion"
        staged_motion.mkdir(parents=True, exist_ok=True)
        shutil.copytree(robust_source, staged_motion, dirs_exist_ok=True)

        if index % 25 == 0 or index == len(selected):
            print(f"staged {index}/{len(selected)}", flush=True)

    selection = {
        "source_manifest": str(args.manifest),
        "criteria": criteria,
        "trial_count": len(selected),
        "all_ids": selected,
    }
    selection_path = args.output_root / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "selection": str(selection_path),
                "subjects": len(staged_subjects),
                "trials": len(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
