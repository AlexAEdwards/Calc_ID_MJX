#!/usr/bin/env python3
"""Combine disjoint processed trial datasets under shared subject folders.

The source datasets are never modified. Subject-level assets are copied from
the first source containing that subject, and every Trial_* directory is
copied into the corresponding subject directory in a new output dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


CLEAN_REQUIRED = (
    "pos_mjx.npy",
    "ID_GT_MJX.npy",
    "GRF_Cleaned.npy",
    "Trial_Processing_Information.json",
)
NOISED_REQUIRED = (
    "pos_mjx_noised.npy",
    "qfrc_inverse_noised.npy",
    "Trial_Processing_Information_noised.json",
)


def trial_id(root: Path, trial: Path) -> str:
    return trial.relative_to(root).as_posix()


def discover_trials(root: Path) -> list[Path]:
    return sorted(root.glob("PD_SUB*/Trial_*"))


def validate_trial(root: Path, trial: Path) -> None:
    processed = trial / "ProcessedData"
    missing = [
        f"ProcessedData/{name}"
        for name in (*CLEAN_REQUIRED, *NOISED_REQUIRED)
        if not (processed / name).is_file()
    ]
    if missing:
        raise ValueError(f"{trial_id(root, trial)} is incomplete: {', '.join(missing)}")


def copy_subject_assets(source_subject: Path, output_subject: Path) -> None:
    output_subject.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_subject.iterdir()):
        if child.name.startswith("Trial_"):
            continue
        target = output_subject / child.name
        if target.exists():
            continue
        if child.is_dir():
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copy2(child, target)


def combine(sources: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new path so source provenance stays unambiguous."
        )

    by_id: dict[str, tuple[Path, Path]] = {}
    source_counts: dict[str, int] = {}
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"Source dataset not found: {source}")
        trials = discover_trials(source)
        source_counts[str(source.resolve())] = len(trials)
        for trial in trials:
            validate_trial(source, trial)
            label = trial_id(source, trial)
            if label in by_id:
                previous_source = by_id[label][0]
                raise ValueError(
                    f"Overlapping trial {label} in {previous_source} and {source}; refusing to overwrite."
                )
            by_id[label] = (source, trial)

    output.mkdir(parents=True)
    copied_subjects: set[str] = set()
    provenance: dict[str, str] = {}
    for index, label in enumerate(sorted(by_id), start=1):
        source, source_trial = by_id[label]
        subject_name, trial_name = label.split("/", 1)
        source_subject = source / subject_name
        output_subject = output / subject_name
        copy_subject_assets(source_subject, output_subject)
        copied_subjects.add(subject_name)
        shutil.copytree(source_trial, output_subject / trial_name)
        provenance[label] = str(source.resolve())
        if index % 50 == 0 or index == len(by_id):
            print(f"copied {index}/{len(by_id)} trials", flush=True)

    labels = sorted(by_id)
    selection = {
        "source_datasets": [str(source.resolve()) for source in sources],
        "trial_count": len(labels),
        "all_ids": labels,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dataset": str(output.resolve()),
        "source_trial_counts": source_counts,
        "subject_count": len(copied_subjects),
        "trial_count": len(labels),
        "clean_and_noised_outputs_required": True,
        "trial_source": provenance,
    }
    (output / "combined_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = combine(args.sources, args.output)
    print(json.dumps({key: value for key, value in manifest.items() if key != "trial_source"}, indent=2))


if __name__ == "__main__":
    main()
