#!/usr/bin/env python3
"""Derive GRF.npy / COP.npy / GRM.npy / Time.npy from the per-trial
trial<N>_forces_COP.mot files produced by extract_stroke_c3d_forces.py.

The .mot is treated as the authoritative force record. For each trial,
this script:
  - locates Motion/Raw/trial<N>_forces_COP.mot
  - parses the OpenSim mot header and 19-column body
  - writes Motion/GRF.npy, COP.npy, GRM.npy (each shape (N, 6) ordered
    [R_x, R_y, R_z, L_x, L_y, L_z]) and Motion/Time.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MOT_COLUMNS = [
    "time",
    "R_ground_force_vx", "R_ground_force_vy", "R_ground_force_vz",
    "R_ground_force_px", "R_ground_force_py", "R_ground_force_pz",
    "R_ground_torque_x", "R_ground_torque_y", "R_ground_torque_z",
    "L_ground_force_vx", "L_ground_force_vy", "L_ground_force_vz",
    "L_ground_force_px", "L_ground_force_py", "L_ground_force_pz",
    "L_ground_torque_x", "L_ground_torque_y", "L_ground_torque_z",
]


def read_force_mot(path: Path) -> tuple[np.ndarray, list[str]]:
    lines = path.read_text(errors="replace").splitlines()
    header_end = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break
    if header_end is None:
        raise ValueError(f"endheader marker missing in {path}")
    column_line = lines[header_end + 1]
    columns = column_line.split("\t")
    rows = []
    for raw in lines[header_end + 2:]:
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) != len(columns):
            raise ValueError(
                f"row has {len(parts)} fields, header has {len(columns)} in {path}"
            )
        rows.append([float(token) for token in parts])
    return np.asarray(rows, dtype=np.float64), columns


def to_arrays(data: np.ndarray, columns: list[str]) -> dict[str, np.ndarray]:
    if columns != MOT_COLUMNS:
        idx = {name: columns.index(name) for name in MOT_COLUMNS}
    else:
        idx = {name: i for i, name in enumerate(MOT_COLUMNS)}

    def col(name: str) -> np.ndarray:
        return data[:, idx[name]]

    time = col("time").astype(np.float64)
    grf = np.column_stack([
        col("R_ground_force_vx"), col("R_ground_force_vy"), col("R_ground_force_vz"),
        col("L_ground_force_vx"), col("L_ground_force_vy"), col("L_ground_force_vz"),
    ]).astype(np.float32)
    cop = np.column_stack([
        col("R_ground_force_px"), col("R_ground_force_py"), col("R_ground_force_pz"),
        col("L_ground_force_px"), col("L_ground_force_py"), col("L_ground_force_pz"),
    ]).astype(np.float32)
    grm = np.column_stack([
        col("R_ground_torque_x"), col("R_ground_torque_y"), col("R_ground_torque_z"),
        col("L_ground_torque_x"), col("L_ground_torque_y"), col("L_ground_torque_z"),
    ]).astype(np.float32)
    return {"time": time, "grf": grf, "cop": cop, "grm": grm}


def process_trial(motion_dir: Path) -> dict:
    raw_dir = motion_dir / "Raw"
    candidates = sorted(raw_dir.glob("trial*_forces_COP.mot"))
    if not candidates:
        raise FileNotFoundError(f"no trial*_forces_COP.mot in {raw_dir}")
    if len(candidates) > 1:
        raise ValueError(f"multiple force mot files in {raw_dir}: {candidates}")
    mot_path = candidates[0]
    data, columns = read_force_mot(mot_path)
    arrays = to_arrays(data, columns)

    np.save(motion_dir / "GRF.npy", arrays["grf"])
    np.save(motion_dir / "COP.npy", arrays["cop"])
    np.save(motion_dir / "GRM.npy", arrays["grm"])
    np.save(motion_dir / "Time.npy", arrays["time"])

    grf = arrays["grf"]
    return {
        "motion_dir": str(motion_dir),
        "mot": str(mot_path),
        "frames": int(grf.shape[0]),
        "nonzero_frames": int(np.count_nonzero(np.any(grf != 0, axis=1))),
        "grf_max_abs_per_col": [float(v) for v in np.max(np.abs(grf), axis=0)],
    }


def find_motion_dirs(dataset_root: Path) -> list[Path]:
    return sorted(p for p in dataset_root.glob("SUBJ*/Trial_*/Motion") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("StrokeDataset"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    motion_dirs = find_motion_dirs(args.dataset_root)
    if args.limit is not None:
        motion_dirs = motion_dirs[: args.limit]

    manifest = {
        "dataset_root": str(args.dataset_root.resolve()),
        "method": (
            "Parse Motion/Raw/trial<N>_forces_COP.mot and split its 19 columns "
            "into Motion/GRF.npy, COP.npy, GRM.npy (each (N,6) "
            "[R_x,R_y,R_z,L_x,L_y,L_z]) plus Motion/Time.npy."
        ),
        "trials_seen": len(motion_dirs),
        "trials_written": 0,
        "failures": [],
        "sample_trials": [],
        "totals": {
            "files_written": 0,
            "grf_nonzero_trials": 0,
            "zero_grf_trials": [],
            "max_abs_grf": 0.0,
            "max_abs_cop": 0.0,
            "max_abs_grm": 0.0,
        },
    }

    for i, motion_dir in enumerate(motion_dirs, start=1):
        try:
            result = process_trial(motion_dir)
            manifest["trials_written"] += 1
            manifest["totals"]["files_written"] += 4
            grf_max = max(result["grf_max_abs_per_col"])
            if grf_max > 0:
                manifest["totals"]["grf_nonzero_trials"] += 1
            else:
                manifest["totals"]["zero_grf_trials"].append(result["motion_dir"])
            manifest["totals"]["max_abs_grf"] = max(manifest["totals"]["max_abs_grf"], grf_max)
            grf = np.load(motion_dir / "GRF.npy")
            cop = np.load(motion_dir / "COP.npy")
            grm = np.load(motion_dir / "GRM.npy")
            manifest["totals"]["max_abs_cop"] = max(
                manifest["totals"]["max_abs_cop"], float(np.max(np.abs(cop))) if cop.size else 0.0
            )
            manifest["totals"]["max_abs_grm"] = max(
                manifest["totals"]["max_abs_grm"], float(np.max(np.abs(grm))) if grm.size else 0.0
            )
            del grf, cop, grm
            if len(manifest["sample_trials"]) < 25:
                manifest["sample_trials"].append(result)
        except Exception as exc:
            manifest["failures"].append({"motion_dir": str(motion_dir), "error": str(exc)})
        if i % 50 == 0:
            print(f"processed {i}/{len(motion_dirs)}", flush=True)

    manifest_path = args.dataset_root / "npy_from_mot_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "trials_seen": manifest["trials_seen"],
        "trials_written": manifest["trials_written"],
        "failures": len(manifest["failures"]),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
