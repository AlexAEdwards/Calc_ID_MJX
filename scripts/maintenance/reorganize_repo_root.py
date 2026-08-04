"""Stage 1 of REFACTOR_PLAN.md: separate data and run artifacts from code.

Moves the unambiguous dataset and artifact directories under ``datasets/`` and
``artifacts/``, then leaves a **symlink at every old path** so nothing that
references the old location breaks. Both roots are on one filesystem, so each
move is a rename: instant, and it needs no free space.

    python scripts/maintenance/reorganize_repo_root.py            # dry run
    python scripts/maintenance/reorganize_repo_root.py --apply
    python scripts/maintenance/reorganize_repo_root.py --revert

Deliberately conservative about scope:

* Directories that mix analysis *code* with their outputs (``figures/``,
  ``AnklePowerAnalysis/``, ``VisAndAnalDataset/`` ...) are left alone. They
  belong under ``analysis/`` in Stage 7, not in ``artifacts/``.
* Anything tracked by git is left alone, so this stage produces no git churn.
* The two NAS mount symlinks stay at the root: relocating a symlink to a mount
  point only adds a level of indirection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from paths import REPO_ROOT
MANIFEST_NAME = "repo_root_layout_manifest.json"

# Pure data. Every one of these is untracked and contains no source.
DATASETS = [
    "TrustedDataSet_ByExperiment",
    "TrustedDataSetNoised12Distributed_EdgeHold_AllPatients",
    "TrustedDataSetNoised12Distributed_AllPatients_EstimatedWeights",
    "TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded",
    "KineticVAEDataset",
    "Datasets_Local",
    "Hip_OA",
    "Hip_OA_Quarantine",
    "Hip_OA_Excluded_Longest5Pct_OneFoot",
    "OlderYoungerAdultDataset_PostVisuallyTrimmed",
    "OlderYoungerAdultDataset_PostVisuallyTrimmed (copy)",
    "OlderYoungerAdultDataset_PreVisuallyTrimmed",
    "OldYoungAdultWalking_MJX_Processed",
    "OldYoungAdultWalking_MJX_Processed_NoTrim_NoFilt",
    "OpenCapSubjects_Filt",
    "OpenCapSubjects_NoTrim_NoFilt",
    "OpenCapWalkingTrunkSwaySubjects",
    "OpenCapValSubjectsForScott",
    "OpenCapFootOptStaging",
    "BadTrialsFromTrustedDataset",
]

# Pure run output / cache. No source in any of them.
ARTIFACTS = [
    "outputs",
    "output",
    "inference_results",
    "logs",
    "tmp",
    "RMASBFigures",
    "CHPC_HPO_results",
    "OpenCapAveMAEPerformanceVals",
    ".jax_compilation_cache",
    "__pycache__",
]

GROUPS = {"datasets": DATASETS, "artifacts": ARTIFACTS}


def _tracked(path: Path) -> bool:
    """True if git tracks anything under this path (then we leave it alone)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.name)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if out.returncode == 0:
            return True
        out = subprocess.run(
            ["git", "ls-files", str(path.name)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def plan() -> Dict[str, List[Dict[str, str]]]:
    moves: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    for group, names in GROUPS.items():
        for name in names:
            src = REPO_ROOT / name
            if not src.exists():
                skipped.append({"name": name, "reason": "not present"})
                continue
            if src.is_symlink():
                skipped.append({"name": name, "reason": "is a symlink (already moved?)"})
                continue
            if _tracked(src):
                skipped.append({"name": name, "reason": "tracked by git - left in place"})
                continue
            moves.append({"name": name, "group": group,
                          "src": str(src), "dst": str(REPO_ROOT / group / name)})
    return {"moves": moves, "skipped": skipped}


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}P"


def dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p, onerror=lambda e: None):
        for f in files:
            try:
                total += (Path(root) / f).lstat().st_size
            except OSError:
                pass
    return total


def do_apply(p: Dict[str, List[Dict[str, str]]], *, measure: bool) -> None:
    applied: List[Dict[str, str]] = []
    for group in GROUPS:
        (REPO_ROOT / group).mkdir(exist_ok=True)

    for m in p["moves"]:
        src, dst = Path(m["src"]), Path(m["dst"])
        if dst.exists():
            raise SystemExit(f"Destination already exists, aborting: {dst}")
        size = dir_size(src) if measure else 0
        os.rename(src, dst)                     # same filesystem -> instant
        # Relative symlink so the tree stays movable as a whole.
        os.symlink(Path(m["group"]) / src.name, src)
        applied.append({**m, "bytes": size})
        print(f"   {src.name:<56} -> {m['group']}/  {human(size) if measure else ''}")

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "stage": "REFACTOR_PLAN.md Stage 1",
        "note": ("Each moved directory has a compatibility symlink at its original "
                 "path. Remove the symlinks only once all callers use paths.py."),
        "moves": applied,
        "skipped": p["skipped"],
    }
    (REPO_ROOT / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"\n   manifest: {REPO_ROOT / MANIFEST_NAME}")


def retire_symlinks(*, apply: bool) -> None:
    """Delete the Stage 1 compatibility symlinks once no code needs the old names.

    Only safe after scripts/maintenance/migrate_paths_to_paths_module.py has routed every
    root-relative reference through paths.py. Reversible: --restore-symlinks
    recreates them from the manifest.
    """
    mf = REPO_ROOT / MANIFEST_NAME
    if not mf.exists():
        raise SystemExit(f"No manifest at {mf}")
    manifest = json.loads(mf.read_text())
    n = 0
    for m in manifest.get("moves", []):
        link = Path(m["src"])
        if link.is_symlink():
            n += 1
            if apply:
                link.unlink()
    print(f"   {'removed' if apply else 'would remove'} {n} compatibility symlinks")
    if apply:
        manifest["symlinks_retired"] = datetime.now().isoformat(timespec="seconds")
        mf.write_text(json.dumps(manifest, indent=2))


def restore_symlinks() -> None:
    """Recreate the compatibility symlinks from the manifest."""
    mf = REPO_ROOT / MANIFEST_NAME
    manifest = json.loads(mf.read_text())
    n = 0
    for m in manifest.get("moves", []):
        link, target = Path(m["src"]), Path(m["group"]) / Path(m["src"]).name
        if not link.exists() and not link.is_symlink():
            os.symlink(target, link)
            n += 1
    manifest.pop("symlinks_retired", None)
    mf.write_text(json.dumps(manifest, indent=2))
    print(f"   restored {n} symlinks")


def do_revert() -> None:
    mf = REPO_ROOT / MANIFEST_NAME
    if not mf.exists():
        raise SystemExit(f"No manifest at {mf}")
    manifest = json.loads(mf.read_text())
    for m in reversed(manifest.get("moves", [])):
        src, dst = Path(m["src"]), Path(m["dst"])
        if src.is_symlink():
            src.unlink()
        if dst.exists():
            os.rename(dst, src)
            print(f"   restored {src.name}")
    for group in GROUPS:
        d = REPO_ROOT / group
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    mf.rename(mf.with_suffix(".json.reverted"))
    print("Reverted.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--retire-symlinks", action="store_true",
                    help="Delete the compatibility symlinks (after paths.py migration).")
    ap.add_argument("--restore-symlinks", action="store_true",
                    help="Recreate them from the manifest.")
    ap.add_argument("--no_measure", action="store_true",
                    help="Skip du-style size measurement (much faster on 600 GB).")
    args = ap.parse_args()

    if args.revert:
        do_revert()
        return
    if args.restore_symlinks:
        restore_symlinks()
        return
    if args.retire_symlinks:
        retire_symlinks(apply=args.apply)
        return

    p = plan()
    print(f"Repo root: {REPO_ROOT}\n")
    for group in GROUPS:
        rows = [m for m in p["moves"] if m["group"] == group]
        print(f"  {group}/  <- {len(rows)} directories")
        for m in rows:
            print(f"      {m['name']}")
    if p["skipped"]:
        print("\n  skipped:")
        for s in p["skipped"]:
            print(f"      {s['name']:<56} {s['reason']}")

    if not args.apply:
        print("\nDry run - nothing moved. Re-run with --apply.")
        return
    print()
    do_apply(p, measure=not args.no_measure)


if __name__ == "__main__":
    main()
