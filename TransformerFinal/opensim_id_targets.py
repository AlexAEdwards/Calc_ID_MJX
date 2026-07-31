"""Strictly align OpenSim inverse-dynamics results to ProcessedData frames.

This module is intentionally independent of ``infer.py`` and ``data_loader.py``
so the training loader can use it without an import cycle.  Training must never
fall back to length-only interpolation: ID samples are first put on the Motion
kinematic timebase, then the exact ProcessData trim bounds are applied.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


INDEPENDENT_DOF_COUNT = 23

# ProcessedData's independent 23-DOF layout.  The first alias is the standard
# OpenSim InverseDynamicsTool output; the remaining aliases cover the raw
# OpenSim-compatible ID .mot files retained by the Older/Younger dataset.
OPENSIM_ID_COLUMN_ALIASES: Mapping[int, Tuple[str, ...]] = {
    3: ("pelvis_tilt_moment", "pelvis_tilt_torque"),
    4: ("pelvis_list_moment", "pelvis_list_torque"),
    5: ("pelvis_rotation_moment", "pelvis_rot_torque"),
    6: ("hip_flexion_r_moment", "hip_flex_r_torque"),
    7: ("hip_adduction_r_moment", "hip_add_r_torque"),
    8: ("hip_rotation_r_moment", "hip_rot_r_torque"),
    9: ("knee_angle_r_moment", "knee_flex_r_torque"),
    10: ("ankle_angle_r_moment", "ankle_flex_r_torque"),
    11: ("subtalar_angle_r_moment", "subt_angle_r_torque"),
    12: ("mtp_angle_r_moment", "toe_angle_r_torque"),
    13: ("hip_flexion_l_moment", "hip_flex_l_torque"),
    14: ("hip_adduction_l_moment", "hip_add_l_torque"),
    15: ("hip_rotation_l_moment", "hip_rot_l_torque"),
    16: ("knee_angle_l_moment", "knee_flex_l_torque"),
    17: ("ankle_angle_l_moment", "ankle_flex_l_torque"),
    18: ("subtalar_angle_l_moment", "subt_angle_l_torque"),
    19: ("mtp_angle_l_moment", "toe_angle_l_torque"),
    20: ("lumbar_extension_moment", "lumbar_ext_torque"),
    21: ("lumbar_bending_moment", "lumbar_latbend_torque"),
    22: ("lumbar_rotation_moment", "lumbar_rot_torque"),
}

# Direct-torque channels sourced from ID (KAM is separately derived from the
# measured GRF and MoCap knee-to-COP vector).
DIRECT_TORQUE_ID_INDICES = (6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18)


class OpenSimIDAlignmentError(ValueError):
    """Raised when OpenSim ID cannot be defensibly aligned for supervision."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise OpenSimIDAlignmentError(f"Could not read trim metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OpenSimIDAlignmentError(f"Trim metadata is not a JSON object: {path}")
    return payload


def _candidate_id_files(trial_root: Path) -> Sequence[Path]:
    candidates = set(trial_root.glob("*.sto"))
    candidates.update((trial_root / "Motion").glob("*.sto"))
    candidates.update((trial_root / "Motion" / "Raw").glob("*.sto"))
    candidates.update((trial_root / "Motion" / "Raw").glob("*.mot"))
    for directory_name in ("OpenSimID", "OpenSimResults", "OpenSim_ID"):
        directory = trial_root / directory_name
        if directory.is_dir():
            candidates.update(directory.glob("*.sto"))
            candidates.update(directory.glob("*.mot"))

    def score(path: Path) -> Tuple[int, int, int, str]:
        name = path.name.lower()
        is_id = "inverse" in name or "invdyn" in name or name.endswith("id.mot") or "_id" in name
        return (
            0 if is_id else 1,
            0 if path.name == "inverse_dynamics.sto" else 1,
            0 if path.suffix.lower() == ".sto" else 1,
            str(path),
        )

    return [p for p in sorted(candidates, key=score) if "ik" not in p.name.lower()]


def find_opensim_id_file(trial_root: str | Path) -> Optional[Path]:
    """Return the highest-priority inverse-dynamics storage file for a trial."""
    candidates = _candidate_id_files(Path(trial_root))
    return candidates[0] if candidates else None


def _storage_header(path: Path) -> Tuple[int, Sequence[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    endheader = next((idx for idx, line in enumerate(lines) if "endheader" in line.lower()), None)
    if endheader is None:
        raise OpenSimIDAlignmentError(f"OpenSim ID file has no endheader marker: {path}")
    header_idx = endheader + 1
    while header_idx < len(lines) and not lines[header_idx].strip():
        header_idx += 1
    if header_idx < len(lines) and "coordinates" in lines[header_idx].lower():
        header_idx += 1
        while header_idx < len(lines) and not lines[header_idx].strip():
            header_idx += 1
    if header_idx >= len(lines):
        raise OpenSimIDAlignmentError(f"OpenSim ID file has no column header: {path}")
    columns = lines[header_idx].strip().split()
    if "time" not in columns:
        raise OpenSimIDAlignmentError(f"OpenSim ID file has no time column: {path}")
    if len(columns) != len(set(columns)):
        # Duplicate force-plate columns are common in legacy files, but no
        # duplicated column may be one that is used as an ID target.
        target_names = {name for aliases in OPENSIM_ID_COLUMN_ALIASES.values() for name in aliases}
        duplicated_targets = sorted(name for name in target_names if columns.count(name) > 1)
        if duplicated_targets:
            raise OpenSimIDAlignmentError(
                f"OpenSim ID target columns are duplicated in {path}: {duplicated_targets}"
            )
    return header_idx, columns


def _load_storage_targets(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[str]]:
    header_idx, columns = _storage_header(path)
    selected: Dict[int, Tuple[int, str]] = {}
    for dof_idx, aliases in OPENSIM_ID_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns:
                selected[dof_idx] = (columns.index(alias), alias)
                break
    if not selected:
        raise OpenSimIDAlignmentError(f"No supported OpenSim ID moment columns were found in {path}")

    time_col = columns.index("time")
    ordered = [(time_col, -1, "time")] + [
        (column_idx, dof_idx, name)
        for dof_idx, (column_idx, name) in sorted(selected.items())
    ]
    try:
        raw = np.loadtxt(path, skiprows=header_idx + 1, usecols=[item[0] for item in ordered], ndmin=2)
    except Exception as exc:  # noqa: BLE001
        raise OpenSimIDAlignmentError(f"Could not parse OpenSim ID values from {path}: {exc}") from exc
    source_time = np.asarray(raw[:, 0], dtype=np.float64)
    values = np.full((len(raw), INDEPENDENT_DOF_COUNT), np.nan, dtype=np.float32)
    available = np.zeros((INDEPENDENT_DOF_COUNT,), dtype=bool)
    used_names = []
    for value_col, (_storage_col, dof_idx, name) in enumerate(ordered[1:], start=1):
        values[:, dof_idx] = np.asarray(raw[:, value_col], dtype=np.float32)
        available[dof_idx] = True
        used_names.append(name)
    return source_time, values, available, used_names


def _validate_time(time: np.ndarray, *, label: str, path: Path) -> np.ndarray:
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if len(time) < 2 or not np.all(np.isfinite(time)):
        raise OpenSimIDAlignmentError(f"{label} must contain at least two finite samples: {path}")
    if not np.all(np.diff(time) > 0.0):
        raise OpenSimIDAlignmentError(f"{label} is not strictly increasing: {path}")
    return time


def _resample_without_extrapolation(
    values: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
    *,
    path: Path,
) -> np.ndarray:
    source_time = _validate_time(source_time, label="OpenSim ID time", path=path)
    target_time = _validate_time(target_time, label="kinematic time", path=path)
    tolerance = 0.51 * max(float(np.median(np.diff(source_time))), float(np.median(np.diff(target_time))))
    if target_time[0] < source_time[0] - tolerance or target_time[-1] > source_time[-1] + tolerance:
        raise OpenSimIDAlignmentError(
            "OpenSim ID does not cover the kinematic time interval without extrapolation: "
            f"ID=[{source_time[0]:.9g}, {source_time[-1]:.9g}], "
            f"kinematics=[{target_time[0]:.9g}, {target_time[-1]:.9g}], file={path}"
        )
    clipped_time = np.clip(target_time, source_time[0], source_time[-1])
    result = np.full((len(target_time), values.shape[1]), np.nan, dtype=np.float32)
    for idx in range(values.shape[1]):
        if np.all(np.isnan(values[:, idx])):
            continue
        result[:, idx] = np.interp(clipped_time, source_time, values[:, idx]).astype(np.float32)
    return result


def _bounds(info: Mapping[str, Any], key: str, upper: int, *, path: Path) -> Tuple[int, int]:
    raw = info.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise OpenSimIDAlignmentError(f"Missing required {key} in {path}")
    start, end = int(raw[0]), int(raw[1])
    if not 0 <= start < end <= upper:
        raise OpenSimIDAlignmentError(f"Invalid {key}={raw} for length {upper} in {path}")
    return start, end


@lru_cache(maxsize=2048)
def _load_aligned_cached(trial_root_text: str, target_len: int) -> Dict[str, Any]:
    trial_root = Path(trial_root_text)
    source_path = find_opensim_id_file(trial_root)
    if source_path is None:
        raise FileNotFoundError(f"No OpenSim inverse-dynamics .sto/.mot file found under {trial_root}")
    source_time, source_values, available, columns = _load_storage_targets(source_path)

    processed = trial_root / "ProcessedData"
    info_path = processed / "Trial_Processing_Information.json"
    motion_time_path = trial_root / "Motion" / "Time.npy"
    kinematic_time_path = trial_root / "Motion" / "Time_for_pos.npy"
    if not info_path.is_file():
        raise OpenSimIDAlignmentError(f"Missing ProcessedData trim metadata: {info_path}")
    if not motion_time_path.is_file():
        raise OpenSimIDAlignmentError(f"Missing kinematic Motion/Time.npy: {motion_time_path}")
    info = _read_json(info_path)
    motion_time = _validate_time(np.load(motion_time_path), label="Motion/Time.npy", path=motion_time_path)
    kinematic_time = (
        _validate_time(np.load(kinematic_time_path), label="Motion/Time_for_pos.npy", path=kinematic_time_path)
        if kinematic_time_path.is_file()
        else motion_time
    )

    declared_pretrim = info.get("core_trim_pretrim_n_frames")
    if not isinstance(declared_pretrim, (int, np.integer)):
        raise OpenSimIDAlignmentError(f"Missing core_trim_pretrim_n_frames in {info_path}")
    pretrim_len = int(declared_pretrim)
    if pretrim_len < 2:
        raise OpenSimIDAlignmentError(
            f"core_trim_pretrim_n_frames={pretrim_len} is invalid in {trial_root}"
        )
    # Mirror ProcessData.resample_dataframes_to_uniform_timestep exactly: inputs
    # are placed on a half-open, uniform 100-Hz grid spanning the overlap of the
    # kinematic and force time vectors.  This matters for datasets whose raw
    # acquisition rate is 60 Hz but whose model inputs were generated at 100 Hz.
    overlap_start = max(float(kinematic_time[0]), float(motion_time[0]))
    overlap_end = min(float(kinematic_time[-1]), float(motion_time[-1]))
    resampled_time = np.arange(overlap_start, overlap_end, 0.01, dtype=np.float64)
    if len(resampled_time) == pretrim_len:
        kinematic_pretrim_time = resampled_time
    elif len(motion_time) >= pretrim_len:
        # Compatibility for already-uniform exports.  Only accept a prefix whose
        # cadence is itself 100 Hz; arbitrary normalized stretching is forbidden.
        prefix = motion_time[:pretrim_len]
        if not np.allclose(np.diff(prefix), 0.01, rtol=1e-4, atol=1e-7):
            raise OpenSimIDAlignmentError(
                f"Could not reconstruct the ProcessData 100-Hz pretrim timebase of length "
                f"{pretrim_len} from Motion time files in {trial_root}"
            )
        kinematic_pretrim_time = prefix
    else:
        raise OpenSimIDAlignmentError(
            f"Could not reconstruct the ProcessData 100-Hz pretrim timebase of length "
            f"{pretrim_len} from Motion time files in {trial_root}"
        )
    core_start, core_end = _bounds(
        info, "core_trim_bounds_motion_aligned", pretrim_len, path=info_path
    )
    target_time = kinematic_pretrim_time[core_start:core_end]
    ds_bounds = info.get("ds_edge_trim_bounds")
    if ds_bounds is not None:
        declared_ds_pretrim = info.get("ds_edge_trim_n_frames_before")
        if int(declared_ds_pretrim) != len(target_time):
            raise OpenSimIDAlignmentError(
                f"ds_edge_trim_n_frames_before={declared_ds_pretrim} does not match the "
                f"core-trimmed length {len(target_time)} in {info_path}"
            )
        ds_start, ds_end = _bounds(info, "ds_edge_trim_bounds", len(target_time), path=info_path)
        target_time = target_time[ds_start:ds_end]
    visual_bounds = info.get("visual_keep_trim_bounds")
    if visual_bounds is not None:
        declared_visual_pretrim = info.get("visual_keep_trim_n_frames_before")
        if int(declared_visual_pretrim) != len(target_time):
            raise OpenSimIDAlignmentError(
                f"visual_keep_trim_n_frames_before={declared_visual_pretrim} does not match the "
                f"preceding trimmed length {len(target_time)} in {info_path}"
            )
        visual_start, visual_end = _bounds(
            info, "visual_keep_trim_bounds", len(target_time), path=info_path
        )
        target_time = target_time[visual_start:visual_end]

    if len(target_time) != int(target_len):
        raise OpenSimIDAlignmentError(
            "Timestamp alignment plus recorded trims did not reproduce the ProcessedData "
            f"kinematic length: aligned ID={len(target_time)}, pos_inputs={target_len}, trial={trial_root}"
        )
    # Interpolate only onto frames retained by ProcessData.  Raw files sometimes
    # omit an unused terminal sample; requiring coverage of already-trimmed-away
    # frames would reject a target whose actual supervision interval is complete.
    aligned = _resample_without_extrapolation(
        source_values, source_time, target_time, path=source_path
    )
    if not np.all(np.isfinite(aligned[:, available])):
        raise OpenSimIDAlignmentError(f"Aligned OpenSim ID contains non-finite target values: {source_path}")
    return {
        "id": aligned,
        "available_mask": available,
        "source_path": str(source_path),
        "available_columns": tuple(columns),
        "alignment": (
            "Motion time overlap -> ProcessData 100-Hz grid -> core trim -> double-support edge trim"
            + (" -> visual keep trim" if visual_bounds is not None else "")
        ),
        "core_trim_bounds": (core_start, core_end),
        "target_len": int(target_len),
    }


def load_aligned_opensim_id_target(
    trial_root: str | Path,
    *,
    target_len: int,
    required_indices: Iterable[int] = DIRECT_TORQUE_ID_INDICES,
) -> Dict[str, Any]:
    """Load a strictly time-aligned 23-DOF OpenSim ID training target.

    ``required_indices`` are checked before returning, preventing training from
    silently filling an unavailable OpenSim coordinate with an MJX target.
    """
    bundle = _load_aligned_cached(str(Path(trial_root).resolve()), int(target_len))
    required = tuple(int(idx) for idx in required_indices)
    missing = [idx for idx in required if idx >= len(bundle["available_mask"]) or not bundle["available_mask"][idx]]
    if missing:
        raise OpenSimIDAlignmentError(
            f"OpenSim ID file {bundle['source_path']} is missing required independent-DOF indices {missing}"
        )
    return {
        **bundle,
        "id": np.array(bundle["id"], copy=True),
        "available_mask": np.array(bundle["available_mask"], copy=True),
    }
