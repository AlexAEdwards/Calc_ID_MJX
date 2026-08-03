"""Reorganize a flat trial dataset into Dataset/<Experiment>/<Subject>/Trial_#/.

Dry-run by default: nothing moves until ``--apply`` is passed.  Every applied run
writes ``experiment_layout_manifest.json`` at the dataset root so the move can be
undone with ``--revert``.

    # inspect the plan
    python scripts/reorganize_dataset_by_experiment.py \
        --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients

    # perform the moves
    python scripts/reorganize_dataset_by_experiment.py \
        --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients --apply

    # undo
    python scripts/reorganize_dataset_by_experiment.py \
        --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients --revert

Only directories carrying ``Patient_MD.json`` (or a ``Trial_*`` subfolder) are
treated as subjects; loose files and helper folders such as ``UnwantedSubjects/``
or ``OpenSimToMJX_Accuracy/`` are left untouched at the dataset root.
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

from paths import resolve as paths_resolve  # noqa: E402
from TransformerFinal.experiment_groups import (  # noqa: E402
    experiment_of_subject,
    is_subject_dir,
    list_experiment_dirs,
)

MANIFEST_NAME = "experiment_layout_manifest.json"
# Trial discovery caches key off the flat layout and go stale after the move.
STALE_CACHE_NAMES = ("trial_discovery_cache.json", "trial_discovery_cache_modq.json")


def _resolve_data_dir(raw: str) -> Path:
    data_dir = paths_resolve(raw)
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {data_dir}")
    return data_dir


def plan_moves(data_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """Return ``{"moves": [...], "unassigned": [...], "ignored": [...]}``."""
    moves: List[Dict[str, str]] = []
    unassigned: List[Dict[str, str]] = []
    ignored: List[str] = []

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not is_subject_dir(entry):
            ignored.append(entry.name)
            continue
        experiment = experiment_of_subject(entry.name)
        if experiment is None:
            unassigned.append({"subject": entry.name, "path": str(entry)})
            continue
        moves.append(
            {
                "subject": entry.name,
                "experiment": experiment,
                "from": str(entry),
                "to": str(data_dir / experiment / entry.name),
            }
        )
    return {"moves": moves, "unassigned": unassigned, "ignored": ignored}


def _print_plan(data_dir: Path, plan: Dict[str, List[Dict[str, str]]]) -> None:
    moves = plan["moves"]
    by_experiment: Dict[str, int] = {}
    for move in moves:
        by_experiment[move["experiment"]] = by_experiment.get(move["experiment"], 0) + 1

    print(f"Dataset: {data_dir}")
    print(f"Subject folders to move: {len(moves)}")
    for experiment in sorted(by_experiment):
        print(f"   {experiment:<16} {by_experiment[experiment]:>4} subjects")
    if plan["unassigned"]:
        print(f"\n⚠️  {len(plan['unassigned'])} subject folder(s) matched no experiment rule:")
        for item in plan["unassigned"]:
            print(f"   {item['subject']}")
        print("   Add a prefix rule in TransformerFinal/experiment_groups.py before applying.")
    if plan["ignored"]:
        print(f"\nLeft at dataset root (not subject folders): {', '.join(plan['ignored'])}")


def apply_moves(data_dir: Path, plan: Dict[str, Any], *, strict: bool) -> Path:
    if plan["unassigned"] and strict:
        raise SystemExit(
            f"Refusing to move: {len(plan['unassigned'])} subject(s) matched no experiment rule. "
            "Pass --allow_unassigned to move the rest and leave them at the root."
        )

    applied: List[Dict[str, str]] = []
    for move in plan["moves"]:
        src = Path(move["from"])
        dst = Path(move["to"])
        if dst.exists():
            raise SystemExit(f"Destination already exists, aborting: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        applied.append(move)

    stale = []
    for name in STALE_CACHE_NAMES:
        cache = data_dir / name
        if cache.exists():
            cache.unlink()
            stale.append(name)

    manifest_path = data_dir / MANIFEST_NAME
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "layout": "experiment",
        "moves": applied,
        "unassigned": plan["unassigned"],
        "ignored": plan["ignored"],
        "removed_caches": stale,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Moved {len(applied)} subject folder(s).")
    if stale:
        print(f"   Removed stale trial discovery cache(s): {', '.join(stale)}")
    print(f"   Manifest: {manifest_path}")
    return manifest_path


def revert(data_dir: Path, manifest_path: Path) -> None:
    if not manifest_path.exists():
        raise SystemExit(f"No manifest to revert from: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    restored = 0
    for move in reversed(manifest.get("moves", [])):
        src = Path(move["to"])
        dst = Path(move["from"])
        if not src.exists():
            print(f"   ⚠️  Missing (already reverted?): {src}")
            continue
        if dst.exists():
            raise SystemExit(f"Cannot restore, path already occupied: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        restored += 1

    # Remove the now-empty experiment directories.
    for experiment_dir in list_experiment_dirs(data_dir):
        try:
            experiment_dir.rmdir()
        except OSError:
            pass
    for name in {move["experiment"] for move in manifest.get("moves", [])}:
        candidate = data_dir / name
        if candidate.is_dir() and not any(candidate.iterdir()):
            candidate.rmdir()

    for cache_name in STALE_CACHE_NAMES:
        cache = data_dir / cache_name
        if cache.exists():
            cache.unlink()

    manifest_path.rename(manifest_path.with_suffix(".json.reverted"))
    print(f"✅ Restored {restored} subject folder(s) to the flat layout.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Dataset root (absolute, or relative to the repo).")
    p.add_argument("--apply", action="store_true", help="Perform the moves (default is a dry run).")
    p.add_argument("--revert", action="store_true", help="Undo a previous --apply using the manifest.")
    p.add_argument("--manifest", default=None, help=f"Manifest path (default: <data_dir>/{MANIFEST_NAME}).")
    p.add_argument(
        "--allow_unassigned",
        action="store_true",
        help="Move the matched subjects even if some folders match no experiment rule.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _resolve_data_dir(args.data_dir)
    manifest_path = Path(args.manifest) if args.manifest else data_dir / MANIFEST_NAME

    if args.revert:
        revert(data_dir, manifest_path)
        return

    plan = plan_moves(data_dir)
    if not plan["moves"]:
        existing = list_experiment_dirs(data_dir)
        if existing:
            print(f"Dataset already nested ({len(existing)} experiment folders). Nothing to do.")
            return
        raise SystemExit(f"No subject folders found under {data_dir}.")

    _print_plan(data_dir, plan)
    if not args.apply:
        print("\nDry run - nothing moved. Re-run with --apply to perform the moves.")
        return
    apply_moves(data_dir, plan, strict=not args.allow_unassigned)


if __name__ == "__main__":
    main()
