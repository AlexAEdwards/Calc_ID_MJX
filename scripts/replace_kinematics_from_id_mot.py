#!/usr/bin/env python3
"""Replace Pos.npy, Vel.npy, Accel.npy in a dataset with GCV-smoothed kinematics
extracted from OpenSim's *_id.mot files.

OpenSim's InverseDynamicsTool applies GCV spline smoothing to the raw IK coordinates
internally before computing torques. The smoothed positions and their spline derivatives
are written to the _id.mot output file. This script uses those as the kinematic source
so that MJX inverse dynamics uses the same kinematics OpenSim ID used.

Usage:
    python3 scripts/replace_kinematics_from_id_mot.py [--dry_run] [--subject OA1] [--workers N]

Output files (float32, matching original shape):
    Motion/Pos.npy   <- GCV-smoothed positions (rad for rotational, m for translational)
    Motion/Vel.npy   <- GCV spline velocities   (rad/s or m/s)
    Motion/Accel.npy <- np.gradient(Vel.npy, dt) (rad/s² or m/s²)

Original files are backed up to Pos_raw.npy / Vel_raw.npy / Accel_raw.npy on first run.
"""

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Resolved through paths.py rather than hardcoded, so this runs on any machine.
# Override with CALCID_DATASETS, or point --dataset at another location.
import os
import sys

from paths import dataset as _dataset

DATASET = Path(
    os.environ.get("CALCID_OSIDFILT_DATASET")
    or _dataset("OldYoungAdultWalking_MJX_Processed_NoTrim_NoFilt_OSIDFilt")
)

# Pos.npy column index → OpenSim coordinate name in the _id.mot
POS_COL_TO_OSIM = {
    0:  "pelvis_tilt",
    1:  "pelvis_list",
    2:  "pelvis_rot",
    3:  "pelvis_tx",
    4:  "pelvis_ty",
    5:  "pelvis_tz",
    6:  "hip_flex_r",
    7:  "hip_add_r",
    8:  "hip_rot_r",
    9:  "knee_flex_r",
    10: "ankle_flex_r",
    11: "subt_angle_r",
    12: "toe_angle_r",
    13: "hip_flex_l",
    14: "hip_add_l",
    15: "hip_rot_l",
    16: "knee_flex_l",
    17: "ankle_flex_l",
    18: "subt_angle_l",
    19: "toe_angle_l",
    20: "lumbar_ext",
    21: "lumbar_latbend",
    22: "lumbar_rot",
}
# Columns that are translational (metres, no degree→radian conversion needed)
TRANSLATIONAL_COLS = {3, 4, 5}
N_COLS = 23
# ---------------------------------------------------------------------------


def _find_id_mot(trial_path: Path):
    raw_dir = trial_path / "Motion" / "Raw"
    if not raw_dir.exists():
        return None
    candidates = [
        p for p in sorted(raw_dir.glob("*.mot"))
        if "id" in p.name.lower() and "_ik" not in p.name.lower()
    ]
    if not candidates:
        return None
    return sorted(candidates,
                  key=lambda p: (0 if p.name.lower().endswith("id.mot") else 1, p.name))[0]


def _load_mot(mot_path: Path):
    with open(mot_path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if "endheader" in l.lower()) + 1
    if "coordinates" in lines[start].lower():
        start += 1
    df = pd.read_csv(mot_path, sep=r"\s+", skiprows=start)
    if df.shape[1] < 2:
        df = pd.read_csv(mot_path, sep="\t", skiprows=start)
    return df


def process_trial(trial_path: Path, dry_run: bool = False) -> dict:
    """Extract GCV kinematics from *_id.mot and overwrite Pos/Vel/Accel.npy."""
    motion_dir = trial_path / "Motion"
    result = {"trial": str(trial_path), "status": "ok", "note": ""}

    # ── locate files ─────────────────────────────────────────────────────────
    mot_path = _find_id_mot(trial_path)
    if mot_path is None:
        return {**result, "status": "skip", "note": "no _id.mot found"}

    pos_path   = motion_dir / "Pos.npy"
    vel_path   = motion_dir / "Vel.npy"
    accel_path = motion_dir / "Accel.npy"
    time_path  = motion_dir / "Time_for_pos.npy"

    for p in (pos_path, vel_path, accel_path, time_path):
        if not p.exists():
            return {**result, "status": "skip", "note": f"missing {p.name}"}

    # ── load data ────────────────────────────────────────────────────────────
    try:
        df = _load_mot(mot_path)
    except Exception as e:
        return {**result, "status": "error", "note": f"mot load failed: {e}"}

    if "time" not in df.columns:
        return {**result, "status": "error", "note": "no time column in mot"}

    mot_time  = df["time"].values.astype(np.float64)
    pos_time  = np.load(time_path).astype(np.float64).ravel()
    orig_pos  = np.load(pos_path).astype(np.float32)

    T_target = len(pos_time)   # match original Pos.npy length
    dt = float(np.median(np.diff(pos_time))) if len(pos_time) > 1 else 0.01

    # ── build pos and vel arrays ─────────────────────────────────────────────
    new_pos = np.zeros((T_target, N_COLS), dtype=np.float64)
    new_vel = np.zeros((T_target, N_COLS), dtype=np.float64)

    missing_cols = []
    for col_idx, osim_name in POS_COL_TO_OSIM.items():
        vel_name = osim_name + "_vel"
        if osim_name not in df.columns or vel_name not in df.columns:
            missing_cols.append(osim_name)
            # Fall back to original Pos.npy for this column
            new_pos[:, col_idx] = orig_pos[:T_target, col_idx].astype(np.float64)
            continue

        raw_pos = df[osim_name].values.astype(np.float64)
        raw_vel = df[vel_name].values.astype(np.float64)

        if col_idx not in TRANSLATIONAL_COLS:
            raw_pos = raw_pos * (np.pi / 180.0)   # deg  → rad
            raw_vel = raw_vel * (np.pi / 180.0)   # deg/s → rad/s

        # Interpolate to Time_for_pos.npy grid (handles any length mismatch)
        if len(mot_time) == T_target and np.allclose(mot_time, pos_time, atol=1e-6):
            new_pos[:, col_idx] = raw_pos
            new_vel[:, col_idx] = raw_vel
        else:
            interp_pos = interp1d(mot_time, raw_pos,
                                  kind="linear", bounds_error=False,
                                  fill_value=(raw_pos[0], raw_pos[-1]))
            interp_vel = interp1d(mot_time, raw_vel,
                                  kind="linear", bounds_error=False,
                                  fill_value=(raw_vel[0], raw_vel[-1]))
            new_pos[:, col_idx] = interp_pos(pos_time)
            new_vel[:, col_idx] = interp_vel(pos_time)

    if missing_cols:
        result["note"] = f"fell back to orig for: {missing_cols}"

    # ── compute acceleration from the new velocity ───────────────────────────
    new_accel = np.gradient(new_vel, dt, axis=0)

    # ── backup originals on first run (idempotent) ───────────────────────────
    if not dry_run:
        for src, bak_name in [(pos_path, "Pos_raw.npy"),
                               (vel_path, "Vel_raw.npy"),
                               (accel_path, "Accel_raw.npy")]:
            bak = motion_dir / bak_name
            if not bak.exists():
                shutil.copy2(src, bak)

        np.save(pos_path,   new_pos.astype(np.float32))
        np.save(vel_path,   new_vel.astype(np.float32))
        np.save(accel_path, new_accel.astype(np.float32))

    result["T_mot"] = len(mot_time)
    result["T_pos"] = T_target
    result["missing_cols"] = missing_cols
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate only — do not write files")
    parser.add_argument("--subject", default=None,
                        help="Process only this subject (e.g. OA1)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default 4)")
    args = parser.parse_args()

    if not DATASET.exists():
        print(f"ERROR: dataset not found: {DATASET}")
        sys.exit(1)

    # Collect all trial paths
    trial_paths = []
    for subj_dir in sorted(DATASET.iterdir()):
        if not subj_dir.is_dir():
            continue
        if not (subj_dir.name.startswith("OA") or subj_dir.name.startswith("Y")):
            continue
        if args.subject and subj_dir.name != args.subject:
            continue
        for trial_dir in sorted(subj_dir.iterdir()):
            if trial_dir.is_dir() and trial_dir.name.startswith("Trial_"):
                trial_paths.append(trial_dir)

    total = len(trial_paths)
    print(f"{'DRY RUN — ' if args.dry_run else ''}Processing {total} trials "
          f"with {args.workers} workers\n")

    ok = skip = error = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_trial, tp, args.dry_run): tp
                   for tp in trial_paths}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            tag = r["status"].upper()
            note = f"  [{r['note']}]" if r["note"] else ""
            trial_name = "/".join(Path(r["trial"]).parts[-2:])
            print(f"  [{i:3d}/{total}] {tag:<5} {trial_name}{note}")
            if r["status"] == "ok":    ok    += 1
            elif r["status"] == "skip": skip += 1
            else:                       error += 1

    print(f"\nDone — ok={ok}  skip={skip}  error={error}")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
