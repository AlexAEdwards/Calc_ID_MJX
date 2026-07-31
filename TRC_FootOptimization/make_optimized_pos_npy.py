#!/usr/bin/env python3
"""
make_optimized_pos_npy.py

Build Pos.npy from the optimized IK .mot files.

- mot_to_pos_npy(...)                : generic. Reads a .mot, selects the named
                                       coordinate columns (in order), saves Pos.npy
                                       (+ optional Time.npy). Reusable anywhere.
- make_optimized_pos_npy_dataset(...): wrapper for OpenCapWalkingTrunkSwaySubjects.
                                       For every <stem>_opt.mot in a trial's
                                       Video/Motion/Raw/, writes the optimized
                                       trial_<n>/Video/Motion/Pos.npy (alongside the
                                       existing Pos_NotOptimized.npy). Non-destructive
                                       for anything but Pos.npy itself.

Column layout matches the dataset's video_motion_manifest.json `pos_columns`
(23 coords: pelvis 6 + each leg's hip/knee/ankle/subtalar/mtp WITHOUT knee _beta,
+ lumbar ext/bend/rot). Values are kept as-is (rotations in degrees, translations
in m, matching inDegrees=yes), dtype float32.

CLI:
    python3 TRC_FootOptimization/make_optimized_pos_npy.py                     # all subjects
    python3 TRC_FootOptimization/make_optimized_pos_npy.py --subjects subject2
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = REPO_ROOT / "OpenCapWalkingTrunkSwaySubjects"
OPT_SUFFIX = "_opt"

# Fallback if a subject has no video_motion_manifest.json (kept in sync with it).
DEFAULT_POS_COLUMNS = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
    "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
    "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]


def _read_mot(mot_path):
    """Return (column_names, data ndarray[frames, ncols]) for an OpenSim .mot/.sto."""
    lines = Path(mot_path).read_text().splitlines()
    hdr_end = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    columns = lines[hdr_end + 1].split("\t")
    rows = [l.split("\t") for l in lines[hdr_end + 2:] if l.strip()]
    data = np.array(rows, dtype=float)
    return columns, data


# =============================================================================
# Generic
# =============================================================================
def mot_to_pos_npy(mot_path, pos_columns, out_pos_npy, out_time_npy=None, dtype="float32"):
    """
    Select `pos_columns` (by name, in order) from a .mot and save as Pos.npy.

    Optionally saves the time column to `out_time_npy`. Returns a summary dict.
    Raises KeyError if any requested column is absent from the .mot.
    """
    columns, data = _read_mot(mot_path)
    name_to_idx = {c: i for i, c in enumerate(columns)}
    missing = [c for c in pos_columns if c not in name_to_idx]
    if missing:
        raise KeyError(f"{Path(mot_path).name}: missing columns {missing}")

    idx = [name_to_idx[c] for c in pos_columns]
    pos = data[:, idx].astype(dtype)

    out_pos_npy = Path(out_pos_npy)
    out_pos_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_pos_npy, pos)

    time_saved = None
    if out_time_npy is not None:
        time = data[:, name_to_idx["time"]].astype("float64")
        np.save(out_time_npy, time)
        time_saved = str(out_time_npy)

    return {"pos_npy": str(out_pos_npy), "time_npy": time_saved,
            "shape": list(pos.shape), "frames": int(pos.shape[0])}


# =============================================================================
# OpenCapWalkingTrunkSwaySubjects wrapper
# =============================================================================
def _pos_columns_for_subject(subject_dir: Path):
    mf = subject_dir / "Video" / "Motion" / "video_motion_manifest.json"
    if mf.exists():
        cols = json.loads(mf.read_text()).get("pos_columns")
        if cols:
            return cols
    return DEFAULT_POS_COLUMNS


def make_optimized_pos_npy_dataset(root=DEFAULT_ROOT, subjects=None,
                                   opt_suffix=OPT_SUFFIX, write_time=True):
    root = Path(root).resolve()
    subject_dirs = sorted(p for p in root.iterdir()
                          if p.is_dir() and (p / "Video").is_dir()
                          and (subjects is None or p.name in subjects))
    if not subject_dirs:
        raise SystemExit(f"No subjects under {root}" + (f" matching {subjects}" if subjects else ""))

    results = []
    for sd in subject_dirs:
        cols = _pos_columns_for_subject(sd)
        opt_mots = sorted(sd.glob(f"trial_*/Video/Motion/Raw/*{opt_suffix}.mot"))
        print(f"[{sd.name}] {len(opt_mots)} optimized .mot")
        for mot in opt_mots:
            motion_dir = mot.parent.parent          # trial_<n>/Video/Motion
            stem = mot.stem[:-len(opt_suffix)] if mot.stem.endswith(opt_suffix) else mot.stem
            pos_out = motion_dir / "Pos.npy"
            time_out = motion_dir / "Time.npy" if (write_time and not (motion_dir / "Time.npy").exists()) else None
            rec = {"subject": sd.name, "trial": motion_dir.parent.parent.name, "stem": stem,
                   "opt_mot": str(mot), "pos_npy": str(pos_out)}
            try:
                info = mot_to_pos_npy(mot, cols, pos_out, out_time_npy=time_out)
                rec.update(info)
                rec["status"] = "ok"
                print(f"    {rec['trial']:<10} {stem:<12} -> Pos.npy {info['shape']}"
                      f"{'  (+Time.npy)' if time_out else ''}")
            except Exception as e:  # noqa: BLE001
                rec["status"] = "error"; rec["error"] = f"{type(e).__name__}: {e}"
                print(f"    {rec['trial']:<10} {stem:<12} ERROR: {rec['error']}")
            results.append(rec)

    report = root / "optimized_pos_npy_report.json"
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    report.write_text(json.dumps({"root": str(root), "pos_columns": DEFAULT_POS_COLUMNS,
                                  "counts": counts, "trials": results}, indent=2))
    print(f"\n=== {sum(counts.values())} trials: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"Report: {report}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--opt-suffix", default=OPT_SUFFIX)
    ap.add_argument("--no-time", action="store_true", help="Do not create Time.npy when missing.")
    args = ap.parse_args()
    make_optimized_pos_npy_dataset(root=args.root, subjects=args.subjects,
                                   opt_suffix=args.opt_suffix, write_time=not args.no_time)


if __name__ == "__main__":
    main()
