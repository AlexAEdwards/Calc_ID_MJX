#!/usr/bin/env python3
"""Restore ProcessedData files from Untrimmed/ backups for trials where
miss_step_cleanup_applied=True.

This undoes the miss-step trim that ran due to a CLI-override bug in
ProcessData.py (RUN_MISSSTEP_POSTPROCESS in CONFIG was False, but the
argparse block unconditionally overwrote it to True).

For each affected trial:
  1. Copies every file from ProcessedData/Untrimmed/ back to ProcessedData/
  2. Clears the miss_step_* keys from Trial_Processing_Information.json
  3. Restores n_frames to the pre-miss-step value
"""

import argparse
import json
import shutil
from pathlib import Path


def restore_trial(proc_dir: Path, dry_run: bool) -> bool:
    info_path = proc_dir / "Trial_Processing_Information.json"
    untrimmed_dir = proc_dir / "Untrimmed"

    if not info_path.exists():
        print(f"  SKIP {proc_dir}: no Trial_Processing_Information.json")
        return False
    if not untrimmed_dir.is_dir():
        print(f"  SKIP {proc_dir}: no Untrimmed/ backup")
        return False

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  SKIP {proc_dir}: could not read info JSON: {e}")
        return False

    if not info.get("miss_step_cleanup_applied"):
        return False  # not affected, silent skip

    pre_miss_n_frames = info.get("miss_step_total_files_seen")  # not reliable
    # Get the true pre-miss-step length from the Untrimmed backup
    sample_files = list(untrimmed_dir.glob("*.npy"))
    if not sample_files:
        print(f"  SKIP {proc_dir}: Untrimmed/ has no .npy files")
        return False

    subj_trial = f"{proc_dir.parent.parent.name}/{proc_dir.parent.name}"

    # Count files to copy
    files = list(untrimmed_dir.iterdir())
    npy_files = [f for f in files if f.suffix == ".npy"]

    if not dry_run:
        copied = 0
        for src in npy_files:
            dst = proc_dir / src.name
            shutil.copy2(src, dst)
            copied += 1

        # Update Trial_Processing_Information.json: remove miss_step keys,
        # restore n_frames from pretrim value
        miss_keys = [k for k in info if k.startswith("miss_step")]
        for k in miss_keys:
            del info[k]

        # n_frames should now match the Untrimmed length
        import numpy as np
        ref = proc_dir / "GRF_Cleaned.npy"
        if ref.exists():
            info["n_frames"] = int(len(np.load(ref)))

        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"  RESTORED {subj_trial}: {copied} files copied from Untrimmed/")
    else:
        print(f"  DRY-RUN  {subj_trial}: would copy {len(npy_files)} files from Untrimmed/")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Restore ProcessedData from Untrimmed/ backups for miss-step-trimmed trials."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("Datasets_NAS/OldYoungAdultWalking_MJX_Processed_NoTrim"),
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without writing anything.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"Scanning: {data_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    restored = 0
    for info_path in sorted(data_root.rglob("ProcessedData/Trial_Processing_Information.json")):
        proc_dir = info_path.parent
        if restore_trial(proc_dir, dry_run=args.dry_run):
            restored += 1

    print(f"\n{'Would restore' if args.dry_run else 'Restored'}: {restored} trials")


if __name__ == "__main__":
    main()
