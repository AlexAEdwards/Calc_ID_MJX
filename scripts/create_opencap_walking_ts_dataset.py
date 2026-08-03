#!/usr/bin/env python3
"""Create a normal-walking/trunk-sway OpenCap validation dataset.

Source layout:
  Datasets_NAS/OpenCapValidationWithVideos/subject#/OpenSimData/Mocap/IK/*.mot
  Datasets_NAS/OpenCapValidationWithVideos/subject#/OpenSimData/Mocap/ID/*.sto

Output layout:
  OpenCapWalkingTrunkSwaySubjects/subject#/trial_#/Mocap/Raw/
  OpenCapWalkingTrunkSwaySubjects/subject#_TS/trial_#/Mocap/Raw/

Each output trial stores copied raw OpenSim files plus parsed NumPy arrays. Compatibility
copies are also written into Trial_<n>/MoCap and Trial_<n>/ProcessedData so repo tools that
expect OpenCapSubjects_Filt-like capitalization can inspect the parsed kinematics/ID files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from paths import artifact, dataset  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "Datasets_NAS" / "OpenCapValidationWithVideos"
DEFAULT_OUTPUT = dataset("OpenCapWalkingTrunkSwaySubjects")

OPENSIM_TO_MJX_IDX: dict[str, int] = {
    "hip_flexion_r": 6,
    "hip_adduction_r": 7,
    "hip_rotation_r": 8,
    "knee_angle_r": 11,
    "ankle_angle_r": 14,
    "subtalar_angle_r": 15,
    "mtp_angle_r": 16,
    "hip_flexion_l": 17,
    "hip_adduction_l": 18,
    "hip_rotation_l": 19,
    "knee_angle_l": 22,
    "ankle_angle_l": 25,
    "subtalar_angle_l": 26,
    "mtp_angle_l": 27,
    "lumbar_extension": 28,
    "lumbar_bending": 29,
    "lumbar_rotation": 30,
}

PELVIS_OPENSIM_TO_MJX: dict[str, tuple[int, str]] = {
    "pelvis_tx": (0, "force"),
    "pelvis_ty": (1, "force"),
    "pelvis_tz": (2, "force"),
    "pelvis_tilt": (3, "moment"),
    "pelvis_list": (4, "moment"),
    "pelvis_rotation": (5, "moment"),
}


@dataclass(frozen=True)
class StorageTable:
    header: list[str]
    columns: list[str]
    data: np.ndarray


def read_storage(path: Path) -> StorageTable:
    lines = path.read_text(errors="replace").splitlines()
    end_idx = next(
        (i for i, line in enumerate(lines) if line.strip().lower() == "endheader"),
        None,
    )
    if end_idx is None:
        raise ValueError(f"endheader marker missing in {path}")
    columns = lines[end_idx + 1].split()
    rows = []
    for line in lines[end_idx + 2 :]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != len(columns):
            raise ValueError(
                f"{path} row has {len(parts)} values, expected {len(columns)}"
            )
        rows.append([float(token) for token in parts])
    return StorageTable(lines[: end_idx + 1], columns, np.asarray(rows, dtype=np.float64))


def parse_trial_number(stem: str, *, trunk_sway: bool) -> int:
    prefix = "walkingTS" if trunk_sway else "walking"
    if not stem.startswith(prefix):
        raise ValueError(f"{stem} does not start with {prefix}")
    suffix = stem[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError(f"{stem} has non-numeric trial suffix")
    return int(suffix)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def map_id_to_mjx(columns: list[str], data: np.ndarray) -> tuple[np.ndarray, list[int]]:
    col_idx = {name: idx for idx, name in enumerate(columns)}
    out = np.full((data.shape[0], 31), np.nan, dtype=np.float32)
    for coord, mjx_idx in OPENSIM_TO_MJX_IDX.items():
        col = f"{coord}_moment"
        if col in col_idx:
            out[:, mjx_idx] = data[:, col_idx[col]].astype(np.float32)
    for coord, (mjx_idx, kind) in PELVIS_OPENSIM_TO_MJX.items():
        col = f"{coord}_{kind}"
        if col in col_idx:
            out[:, mjx_idx] = data[:, col_idx[col]].astype(np.float32)
    nan_channels = [idx for idx in range(out.shape[1]) if np.isnan(out[:, idx]).any()]
    out[:, nan_channels] = 0.0
    return out, nan_channels


def copy_if_needed(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def write_compatibility_arrays(
    trial_dir: Path,
    time: np.ndarray,
    pos: np.ndarray,
    id_full: np.ndarray,
    id_mjx: np.ndarray,
    overwrite: bool,
) -> None:
    for compat_dir in (trial_dir / "MoCap", trial_dir / "ProcessedData"):
        compat_dir.mkdir(parents=True, exist_ok=True)
        targets = {
            "Time.npy": time.astype(np.float64),
            "Pos.npy": pos.astype(np.float32),
            "pos_inputs.npy": pos.astype(np.float32),
            "OpenSim_ID.npy": id_full.astype(np.float32),
            "OpenSim_ID_recalculated.npy": id_mjx.astype(np.float32),
        }
        for name, arr in targets.items():
            path = compat_dir / name
            if overwrite or not path.exists():
                np.save(path, arr)


def process_pair(
    subject: str,
    ik_path: Path,
    id_path: Path,
    out_subject_dir: Path,
    trial_number: int,
    *,
    source_group: str,
    overwrite: bool,
) -> dict:
    trial_dir = out_subject_dir / f"trial_{trial_number}"
    raw_dir = trial_dir / "Mocap" / "Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ik_dst = raw_dir / ik_path.name
    id_dst = raw_dir / id_path.name
    copy_if_needed(ik_path, ik_dst, overwrite)
    copy_if_needed(id_path, id_dst, overwrite)

    ik = read_storage(ik_path)
    id_table = read_storage(id_path)
    if "time" not in ik.columns or "time" not in id_table.columns:
        raise ValueError(f"{ik_path.name}/{id_path.name} missing time column")

    ik_time = ik.data[:, ik.columns.index("time")]
    id_time = id_table.data[:, id_table.columns.index("time")]
    ik_values = ik.data[:, 1:].astype(np.float32)
    id_values = id_table.data[:, 1:].astype(np.float32)
    id_mjx, nan_channels = map_id_to_mjx(id_table.columns, id_table.data)

    raw_arrays = {
        "Time.npy": ik_time.astype(np.float64),
        "IK_Time.npy": ik_time.astype(np.float64),
        "ID_Time.npy": id_time.astype(np.float64),
        "Pos.npy": ik_values,
        "Kinematics.npy": ik_values,
        "OpenSim_ID.npy": id_values,
        "OpenSim_ID_MJX31.npy": id_mjx,
    }
    for name, arr in raw_arrays.items():
        path = raw_dir / name
        if overwrite or not path.exists():
            np.save(path, arr)

    write_compatibility_arrays(trial_dir, ik_time, ik_values, id_values, id_mjx, overwrite)

    metadata = {
        "subject": subject,
        "output_subject": out_subject_dir.name,
        "source_group": source_group,
        "source_ik": str(ik_path),
        "source_id": str(id_path),
        "raw_ik": str(ik_dst),
        "raw_id": str(id_dst),
        "ik_columns": ik.columns,
        "id_columns": id_table.columns,
        "kinematic_columns": ik.columns[1:],
        "id_generalized_force_columns": id_table.columns[1:],
        "id_mjx31_nan_channels_filled_with_zero": nan_channels,
        "frames": {
            "ik": int(ik_values.shape[0]),
            "id": int(id_values.shape[0]),
        },
        "shapes": {
            "Pos.npy": list(ik_values.shape),
            "OpenSim_ID.npy": list(id_values.shape),
            "OpenSim_ID_MJX31.npy": list(id_mjx.shape),
        },
        "time_match": bool(
            ik_time.shape == id_time.shape and np.allclose(ik_time, id_time, atol=1e-8)
        ),
    }
    save_json(raw_dir / "storage_metadata.json", metadata)
    save_json(trial_dir / "trial_manifest.json", metadata)
    return metadata


def collect_trials(subject_dir: Path, *, trunk_sway: bool) -> list[tuple[int, Path, Path]]:
    ik_dir = subject_dir / "OpenSimData" / "Mocap" / "IK"
    id_dir = subject_dir / "OpenSimData" / "Mocap" / "ID"
    prefix = "walkingTS" if trunk_sway else "walking"
    pairs = []
    for ik_path in sorted(ik_dir.glob(f"{prefix}*.mot")):
        if "_marker_errors" in ik_path.name or "_setup" in ik_path.name:
            continue
        if not trunk_sway and ik_path.stem.startswith("walkingTS"):
            continue
        trial_number = parse_trial_number(ik_path.stem, trunk_sway=trunk_sway)
        id_path = id_dir / f"{ik_path.stem}.sto"
        if not id_path.exists():
            raise FileNotFoundError(f"Missing ID file for {ik_path}: {id_path}")
        pairs.append((trial_number, ik_path, id_path))
    return sorted(pairs, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "dataset": output_root.name,
        "description": (
            "Normal walking subjects plus trunk-sway walking subjects from "
            "OpenCapValidationWithVideos OpenSim Mocap IK/ID storage files."
        ),
        "subjects": {},
        "totals": {
            "source_subjects": 0,
            "output_subjects": 0,
            "normal_trials": 0,
            "trunk_sway_trials": 0,
            "trials": 0,
        },
        "failures": [],
    }

    for subject_dir in sorted(source_root.glob("subject*")):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        manifest["totals"]["source_subjects"] += 1
        subject_entry = {"normal": [], "trunk_sway": []}
        for trunk_sway, suffix, group_key in (
            (False, "", "normal"),
            (True, "_TS", "trunk_sway"),
        ):
            out_subject_dir = output_root / f"{subject}{suffix}"
            out_subject_dir.mkdir(parents=True, exist_ok=True)
            try:
                trials = collect_trials(subject_dir, trunk_sway=trunk_sway)
                for trial_number, ik_path, id_path in trials:
                    meta = process_pair(
                        subject,
                        ik_path,
                        id_path,
                        out_subject_dir,
                        trial_number,
                        source_group=group_key,
                        overwrite=args.overwrite,
                    )
                    subject_entry[group_key].append(
                        {
                            "trial": trial_number,
                            "output_trial": str(
                                (out_subject_dir / f"trial_{trial_number}").relative_to(output_root)
                            ),
                            "frames": meta["frames"],
                            "time_match": meta["time_match"],
                        }
                    )
                manifest["totals"]["output_subjects"] += 1
                manifest["totals"][f"{group_key}_trials"] += len(trials)
                manifest["totals"]["trials"] += len(trials)
            except Exception as exc:
                manifest["failures"].append(
                    {"subject": subject, "group": group_key, "error": str(exc)}
                )
        manifest["subjects"][subject] = subject_entry

    save_json(output_root / "dataset_manifest.json", manifest)
    print(json.dumps(manifest["totals"], indent=2))
    if manifest["failures"]:
        print(json.dumps({"failures": manifest["failures"]}, indent=2))


if __name__ == "__main__":
    main()
