"""Stage the OpenCap normal-walking trials as an ``OpenCapVal/`` experiment.

``OpenCapWalkingTrunkSwaySubjects/`` stores each subject twice - ``subjectN`` for
normal walking and ``subjectN_TS`` for the trunk-sway condition - and each trial
carries both a ``Video/`` (OpenCap markerless) and a ``MoCap/`` (marker-based)
``ProcessedData`` bundle::

    OpenCapWalkingTrunkSwaySubjects/subject10/trial_1/{Video,MoCap}/ProcessedData/

The trusted/experiment layout wants a single ``ProcessedData`` per trial::

    <dataset>/OpenCapVal/subject10/Trial_1/ProcessedData/

This script copies the **non-trunk-sway** trials only, taking **MoCap** as the
source, and registers ``OpenCapVal`` in the dataset's
``experiment_layout_manifest.json`` so trial discovery picks it up.

    # inspect the plan
    python scripts/stage_opencap_val_experiment.py --dest TrustedDataSet_ByExperiment

    # copy
    python scripts/stage_opencap_val_experiment.py --dest TrustedDataSet_ByExperiment --apply

    # undo
    python scripts/stage_opencap_val_experiment.py --dest TrustedDataSet_ByExperiment --revert

These trials have no ``_noised.npy`` bundle. Training must therefore run with
``--allow_missing_noised True`` (the LOEO wrapper's default) or every one of them
is silently skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

from TransformerFinal.experiment_groups import LAYOUT_MANIFEST_NAME  # noqa: E402
from paths import artifact, dataset  # noqa: E402

DEFAULT_SOURCE = dataset("OpenCapWalkingTrunkSwaySubjects")
EXPERIMENT_NAME = "OpenCapVal"
SOURCE_MANIFEST = "dataset_manifest.json"
TRUNK_SWAY_SUFFIX = "_TS"


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path)


def normal_trials(source: Path) -> List[Dict[str, str]]:
    """Return the non-trunk-sway trials as ``{subject, trial, src}`` records.

    The source dataset manifest splits each subject's trials into ``normal`` and
    ``trunk_sway`` groups; that is authoritative. The ``_TS`` folder-suffix rule is
    only a fallback for a source tree without a manifest.
    """
    manifest_path = source / SOURCE_MANIFEST
    records: List[Dict[str, str]] = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        for subject, groups in sorted(manifest.get("subjects", {}).items()):
            for entry in groups.get("normal", []):
                out = str(entry.get("output_trial", "")).strip()
                if not out:
                    continue
                subject_dir, _, trial_dir = out.partition("/")
                records.append({"subject": subject_dir, "trial": trial_dir})
    else:
        for subject_path in sorted(source.iterdir()):
            if not subject_path.is_dir() or subject_path.name.endswith(TRUNK_SWAY_SUFFIX):
                continue
            for trial_path in sorted(subject_path.glob("trial_*")):
                records.append({"subject": subject_path.name, "trial": trial_path.name})

    # Keep only trials that actually carry a MoCap bundle.
    kept = []
    for record in records:
        src = source / record["subject"] / record["trial"] / "MoCap" / "ProcessedData"
        if not (src / "pos_inputs.npy").exists():
            print(f"   ⚠️  skipping (no MoCap/ProcessedData): {record['subject']}/{record['trial']}")
            continue
        number = record["trial"].split("_")[-1]
        record["dest_trial"] = f"Trial_{number}"
        record["src"] = str(src)
        kept.append(record)
    return kept


def stage(source: Path, dest_root: Path, records: List[Dict[str, str]], *, apply: bool) -> Dict[str, Any]:
    experiment_root = dest_root / EXPERIMENT_NAME
    subjects = sorted({r["subject"] for r in records})
    staged: List[Dict[str, str]] = []

    print(f"Source: {source}")
    print(f"Dest:   {experiment_root}")
    print(f"Non-trunk-sway trials: {len(records)} across {len(subjects)} subjects")
    for subject in subjects:
        trials = [r["dest_trial"] for r in records if r["subject"] == subject]
        print(f"   {subject:<14} {len(trials)} trials -> {', '.join(trials)}")

    if not apply:
        return {"staged": []}

    for subject in subjects:
        src_subject = source / subject
        dst_subject = experiment_root / subject
        dst_subject.mkdir(parents=True, exist_ok=True)
        # Subject-level metadata and models (Patient_MD.json, PatientSize.npy, ...).
        for item in sorted(src_subject.iterdir()):
            if item.is_file():
                shutil.copy2(item, dst_subject / item.name)
            elif item.name == "Geometry" and not (dst_subject / "Geometry").exists():
                shutil.copytree(item, dst_subject / "Geometry")

    for record in records:
        dst_trial = experiment_root / record["subject"] / record["dest_trial"]
        dst_processed = dst_trial / "ProcessedData"
        if dst_processed.exists():
            raise SystemExit(f"Destination already exists, aborting: {dst_processed}")
        dst_trial.mkdir(parents=True, exist_ok=True)
        shutil.copytree(record["src"], dst_processed)

        # Carry the matching raw Motion bundle across when the source has one.
        src_motion = Path(record["src"]).parent / "Motion"
        if src_motion.is_dir() and not (dst_trial / "Motion").exists():
            shutil.copytree(src_motion, dst_trial / "Motion")

        src_trial_manifest = Path(record["src"]).parent.parent / "trial_manifest.json"
        if src_trial_manifest.exists():
            shutil.copy2(src_trial_manifest, dst_trial / "trial_manifest.json")

        staged.append(
            {
                "subject": record["subject"],
                "source_trial": f"{record['subject']}/{record['trial']}",
                "dest": str(dst_trial),
            }
        )
    return {"staged": staged}


def register_experiment(dest_root: Path, staged: List[Dict[str, str]], source: Path) -> None:
    """Add OpenCapVal to the layout manifest so discovery treats it as an experiment."""
    manifest_path = dest_root / LAYOUT_MANIFEST_NAME
    manifest: Dict[str, Any]
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"data_dir": str(dest_root), "layout": "experiment", "moves": []}

    experiments = set(manifest.get("experiments", []))
    experiments.add(EXPERIMENT_NAME)
    manifest["experiments"] = sorted(experiments)
    manifest.setdefault("staged_experiments", {})[EXPERIMENT_NAME] = {
        "staged_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "source_selection": "dataset_manifest.json -> subjects[*].normal (non-trunk-sway)",
        "processed_source": "MoCap/ProcessedData",
        "has_noised_bundle": False,
        "n_trials": len(staged),
        "trials": staged,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Registered '{EXPERIMENT_NAME}' in {manifest_path}")


def revert(dest_root: Path) -> None:
    experiment_root = dest_root / EXPERIMENT_NAME
    if experiment_root.exists():
        shutil.rmtree(experiment_root)
        print(f"✅ Removed {experiment_root}")
    else:
        print(f"   (nothing at {experiment_root})")

    manifest_path = dest_root / LAYOUT_MANIFEST_NAME
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["experiments"] = [e for e in manifest.get("experiments", []) if e != EXPERIMENT_NAME]
        manifest.get("staged_experiments", {}).pop(EXPERIMENT_NAME, None)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"   Deregistered '{EXPERIMENT_NAME}' from {manifest_path}")

    for cache in ("trial_discovery_cache.json", "trial_discovery_cache_modq.json"):
        path = dest_root / cache
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="OpenCapWalkingTrunkSwaySubjects root.")
    p.add_argument("--dest", required=True, help="Nested dataset root to stage OpenCapVal/ into.")
    p.add_argument("--apply", action="store_true", help="Perform the copy (default is a dry run).")
    p.add_argument("--revert", action="store_true", help="Delete OpenCapVal/ and deregister it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = _resolve(args.source)
    dest_root = _resolve(args.dest)
    if not dest_root.is_dir():
        raise SystemExit(f"Destination dataset does not exist: {dest_root}")

    if args.revert:
        revert(dest_root)
        return

    if not source.is_dir():
        raise SystemExit(f"Source dataset does not exist: {source}")

    records = normal_trials(source)
    if not records:
        raise SystemExit(f"No non-trunk-sway trials with a MoCap bundle found under {source}.")

    result = stage(source, dest_root, records, apply=args.apply)
    if not args.apply:
        print("\nDry run - nothing copied. Re-run with --apply.")
        return

    register_experiment(dest_root, result["staged"], source)
    for cache in ("trial_discovery_cache.json", "trial_discovery_cache_modq.json"):
        path = dest_root / cache
        if path.exists():
            path.unlink()
    print(f"\n✅ Staged {len(result['staged'])} trials into {dest_root / EXPERIMENT_NAME}")
    print("   These trials have no _noised bundle: train with --allow_missing_noised True.")


if __name__ == "__main__":
    main()
