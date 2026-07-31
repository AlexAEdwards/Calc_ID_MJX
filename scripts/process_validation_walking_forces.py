#!/usr/bin/env python3
"""Copy and convert OpenCapValidationWithVideos walking force files.

The saved arrays intentionally stay in OpenSim coordinates, matching the raw
Motion/GRF.npy, Motion/COP.npy, and Motion/GRM.npy convention consumed by
ProcessData.py before it converts forces into MuJoCo coordinates.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "Datasets_NAS/OpenCapValidationWithVideos"
DATASET_ROOT = REPO_ROOT / "OpenCapWalkingTrunkSwaySubjects"
REPORT_PATH = DATASET_ROOT / "force_processing_report.json"

FORCE_COLUMNS = {
    "GRF": [
        "R_ground_force_vx",
        "R_ground_force_vy",
        "R_ground_force_vz",
        "L_ground_force_vx",
        "L_ground_force_vy",
        "L_ground_force_vz",
    ],
    "COP": [
        "R_ground_force_px",
        "R_ground_force_py",
        "R_ground_force_pz",
        "L_ground_force_px",
        "L_ground_force_py",
        "L_ground_force_pz",
    ],
    "GRM": [
        "R_ground_torque_x",
        "R_ground_torque_y",
        "R_ground_torque_z",
        "L_ground_torque_x",
        "L_ground_torque_y",
        "L_ground_torque_z",
    ],
}

def parse_opensim_mot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    columns = lines[header_idx + 1].strip().split()
    data = np.loadtxt(path, skiprows=header_idx + 2, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(columns):
        raise ValueError(f"{path} has {data.shape[1]} data columns but {len(columns)} labels")
    return columns, data


def extract_columns(columns: list[str], data: np.ndarray, names: list[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(columns)}
    missing = [name for name in names if name not in index]
    if missing:
        raise KeyError(f"Missing columns {missing}; available columns: {columns}")
    return data[:, [index[name] for name in names]]


def resample_to_time(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    if source_time.ndim != 1 or target_time.ndim != 1:
        raise ValueError("source_time and target_time must be 1-D")
    if len(source_time) < 2:
        raise ValueError("source_time must contain at least two samples")
    if np.any(np.diff(source_time) <= 0):
        raise ValueError("source_time must be strictly increasing")
    out = np.empty((len(target_time), values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(target_time, source_time, values[:, col])
    return out.astype(np.float32)


def discover_trials() -> list[dict[str, str]]:
    specs = []
    for manifest_path in sorted(DATASET_ROOT.glob("subject*/trial_*/trial_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_subject = manifest["subject"]
        output_subject = manifest["output_subject"]
        trial_name = Path(manifest["source_ik"]).stem
        if not (trial_name.startswith("walking") or trial_name.startswith("walkingTS")):
            continue
        specs.append(
            {
                "source_subject": source_subject,
                "output_subject": output_subject,
                "trial": manifest_path.parent.name,
                "trial_name": trial_name,
            }
        )
    return specs


def process_trial(spec: dict[str, str]) -> dict:
    source_subject = spec["source_subject"]
    output_subject = spec["output_subject"]
    trial = spec["trial"]
    trial_name = spec["trial_name"]
    src = SOURCE_ROOT / source_subject / "ForceData" / f"{trial_name}_forces.mot"
    if not src.exists():
        raise FileNotFoundError(src)

    columns, data = parse_opensim_mot(src)
    source_time = data[:, columns.index("time")]
    extracted = {
        key: extract_columns(columns, data, col_names)
        for key, col_names in FORCE_COLUMNS.items()
    }

    subject_trial = DATASET_ROOT / output_subject / trial
    outputs = []
    for modality in ("MoCap", "Video"):
        motion_dir = subject_trial / modality / "Motion"
        raw_dir = motion_dir / "Raw"
        time_path = motion_dir / "Time.npy"
        pos_path = motion_dir / "Pos.npy"
        if not time_path.exists():
            raise FileNotFoundError(time_path)
        if not pos_path.exists():
            raise FileNotFoundError(pos_path)

        raw_dir.mkdir(parents=True, exist_ok=True)
        copied_mot = raw_dir / src.name
        shutil.copyfile(src, copied_mot)

        target_time = np.asarray(np.load(time_path), dtype=np.float64).reshape(-1)
        pos = np.load(pos_path)
        if len(target_time) != pos.shape[0]:
            raise ValueError(f"{time_path} length {len(target_time)} does not match Pos frames {pos.shape[0]}")

        saved = {}
        for key, values in extracted.items():
            arr = resample_to_time(source_time, values, target_time)
            out_path = motion_dir / f"{key}.npy"
            np.save(out_path, arr)
            saved[key] = {
                "path": str(out_path),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        outputs.append(
            {
                "modality": modality,
                "raw_copy": str(copied_mot),
                "target_time": str(time_path),
                "target_frames": int(len(target_time)),
                "target_dt_median": float(np.median(np.diff(target_time))) if len(target_time) > 1 else None,
                "saved": saved,
            }
        )

    return {
        "source_subject": source_subject,
        "output_subject": output_subject,
        "trial": trial,
        "trial_name": trial_name,
        "source_force_file": str(src),
        "source_rows": int(data.shape[0]),
        "source_time_start": float(source_time[0]),
        "source_time_end": float(source_time[-1]),
        "outputs": outputs,
    }


def main() -> int:
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Source root not found: {SOURCE_ROOT}")

    specs = discover_trials()
    if not specs:
        raise SystemExit(f"No trial manifests found under: {DATASET_ROOT}")

    results = [process_trial(spec) for spec in specs]
    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "coordinate_system": "OpenSim Y-up; columns are [R_x, R_y, R_z, L_x, L_y, L_z]",
        "vertical_grf_columns": {"right": 1, "left": 4},
        "source_root": str(SOURCE_ROOT),
        "dataset_root": str(DATASET_ROOT),
        "trials_processed": len(results),
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trials_processed": len(results), "report": str(REPORT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
