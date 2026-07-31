#!/usr/bin/env python3
"""Generate OpenSim inverse dynamics inputs from per-trial Motion/ProcessedData arrays."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded"
DEFAULT_MODEL_NAME = "OpenSimModel.osim"
DEFAULT_OUTPUT_DIR_NAME = "OpenSimResults"
DEFAULT_RIGHT_BODY = "calcn_r"
DEFAULT_LEFT_BODY = "calcn_l"
SOURCE_MOTION = "motion"
SOURCE_PROCESSED = "processed"

NPY_COORDINATES = [
    ("pelvis_tilt", False),
    ("pelvis_list", False),
    ("pelvis_rot", False),
    ("pelvis_tx", True),
    ("pelvis_ty", True),
    ("pelvis_tz", True),
    ("hip_flex_r", False),
    ("hip_add_r", False),
    ("hip_rot_r", False),
    ("knee_flex_r", False),
    ("ankle_flex_r", False),
    ("subt_angle_r", False),
    ("toe_angle_r", False),
    ("hip_flex_l", False),
    ("hip_add_l", False),
    ("hip_rot_l", False),
    ("knee_flex_l", False),
    ("ankle_flex_l", False),
    ("subt_angle_l", False),
    ("toe_angle_l", False),
    ("lumbar_ext", False),
    ("lumbar_latbend", False),
    ("lumbar_rot", False),
]

NP_TO_QPOS = {
    0: 3,
    1: 4,
    2: 5,
    3: 0,
    4: 1,
    5: 2,
    6: 6,
    7: 7,
    8: 8,
    9: 11,
    10: 14,
    11: 15,
    12: 16,
    13: 17,
    14: 18,
    15: 19,
    16: 22,
    17: 25,
    18: 26,
    19: 27,
    20: 28,
    21: 29,
    22: 30,
}

COORDINATE_ALIASES = {
    "pelvis_rot": ("pelvis_rotation",),
    "hip_flex_r": ("hip_flexion_r",),
    "hip_add_r": ("hip_adduction_r",),
    "hip_rot_r": ("hip_rotation_r",),
    "knee_flex_r": ("knee_angle_r",),
    "ankle_flex_r": ("ankle_angle_r",),
    "subt_angle_r": ("subtalar_angle_r",),
    "toe_angle_r": ("mtp_angle_r",),
    "hip_flex_l": ("hip_flexion_l",),
    "hip_add_l": ("hip_adduction_l",),
    "hip_rot_l": ("hip_rotation_l",),
    "knee_flex_l": ("knee_angle_l",),
    "ankle_flex_l": ("ankle_angle_l",),
    "subt_angle_l": ("subtalar_angle_l",),
    "toe_angle_l": ("mtp_angle_l",),
    "lumbar_ext": ("lumbar_extension",),
    "lumbar_latbend": ("lumbar_bending",),
    "lumbar_rot": ("lumbar_rotation",),
}

FORCE_COLUMNS = [
    "time",
    "R_ground_force_vx",
    "R_ground_force_vy",
    "R_ground_force_vz",
    "R_ground_force_px",
    "R_ground_force_py",
    "R_ground_force_pz",
    "R_ground_torque_x",
    "R_ground_torque_y",
    "R_ground_torque_z",
    "L_ground_force_vx",
    "L_ground_force_vy",
    "L_ground_force_vz",
    "L_ground_force_px",
    "L_ground_force_py",
    "L_ground_force_pz",
    "L_ground_torque_x",
    "L_ground_torque_y",
    "L_ground_torque_z",
]


@dataclass(frozen=True)
class TrialPaths:
    subject_dir: Path
    trial_dir: Path
    motion_dir: Path
    model_path: Path
    output_dir: Path

    @property
    def processed_dir(self) -> Path:
        return self.trial_dir / "ProcessedData"


# Subjects whose model is not named OpenSimModel.osim (e.g. SUBJ* use the OpenCap
# LaiUhlrich2022 model). Resolution falls back to these names, then to any single .osim.
FALLBACK_MODEL_NAMES = ("LaiUhlrich2022_scaled.osim", "LaiArnold2017_scaled.osim")


def resolve_model_path(subject_dir: Path, model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """Return the subject's OpenSim model, tolerating non-default filenames.

    Prefers ``model_name``; otherwise tries known alternates; otherwise, if exactly one
    ``.osim`` lives in the subject folder, uses it. Returns the default path (which may not
    exist) when nothing is found, so callers raise a clear 'missing model' error.
    """
    primary = subject_dir / model_name
    if primary.exists():
        return primary
    for alt in FALLBACK_MODEL_NAMES:
        cand = subject_dir / alt
        if cand.exists():
            return cand
    osims = [p for p in sorted(subject_dir.glob("*.osim")) if "_NoPatel" not in p.name]
    if len(osims) == 1:
        return osims[0]
    return primary


def discover_trials(
    dataset_root: Path,
    subject: str | None = None,
    trial: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
) -> list[TrialPaths]:
    trials: list[TrialPaths] = []
    for subject_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if subject and subject_dir.name != subject:
            continue
        model_path = resolve_model_path(subject_dir, model_name)
        for trial_dir in sorted(p for p in subject_dir.glob("Trial_*") if p.is_dir()):
            if trial and trial_dir.name != trial:
                continue
            motion_dir = trial_dir / "Motion"
            if motion_dir.is_dir():
                trials.append(
                    TrialPaths(
                        subject_dir=subject_dir,
                        trial_dir=trial_dir,
                        motion_dir=motion_dir,
                        model_path=model_path,
                        output_dir=trial_dir / output_dir_name,
                    )
                )
    return trials


def load_array(path: Path, expected_ndim: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.ndim != expected_ndim:
        raise ValueError(f"{path} has ndim={arr.ndim}, expected {expected_ndim}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path} contains non-finite values")
    return arr


def fit_time_to_length(time: np.ndarray, length: int, label: str) -> np.ndarray:
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if len(time) == length:
        return time
    if len(time) < 2:
        raise ValueError(f"{label} length {len(time)} cannot be fit to {length} frames")
    return np.linspace(float(time[0]), float(time[-1]), length, dtype=np.float64)


def mjx_to_opensim_coords(vec: np.ndarray) -> np.ndarray:
    """Convert MuJoCo/MJX [X, Y, Z] Z-up vectors to OpenSim [X, Y, Z] Y-up."""
    arr = np.asarray(vec, dtype=np.float64)
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = arr[..., 2]
    out[..., 2] = -arr[..., 1]
    return out


def qpos_to_npy_coordinates(qpos: np.ndarray) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[1] <= max(NP_TO_QPOS.values()):
        raise ValueError(f"pos_mjx has shape {qpos.shape}, expected (T, >=31)")
    pos = np.zeros((qpos.shape[0], len(NPY_COORDINATES)), dtype=np.float64)
    for npy_idx, qpos_idx in NP_TO_QPOS.items():
        pos[:, npy_idx] = qpos[:, qpos_idx]
    return pos


# Processed MJX arrays (pos_mjx/qvel_mjx/qacc_mjx) are produced at a fixed 100 Hz.
# OpenSim ID differentiates the coordinates file, so the time step MUST match the dt
# MuJoCo used to produce qvel/qacc -- otherwise accelerations (and thus inertial
# torques) are scaled by (dt_true/dt_used)^2. Some trials (e.g. GaitRetraining) carry
# a raw Motion/Time.npy at the mocap rate (e.g. 2000 Hz) that is NOT aligned to the
# decimated processed frames; slicing it yields a 20x-too-small dt and ~400x torque
# blow-up. We therefore only trust a motion-aligned slice when its sampling rate is
# physically consistent with the processed rate, and otherwise use uniform 100 Hz.
PROCESSED_DT_S = 0.01
_PROCESSED_DT_TOL = 0.2  # accept slices whose median dt is within +/-20% of 100 Hz


def processed_time_vector(paths: TrialPaths, n_frames: int) -> np.ndarray:
    # The processed pos_mjx is a uniformly-resampled MJX trajectory; MuJoCo integrates at
    # a fixed dt, so the time vector that matches the MJX qvel/qacc (and thus ID_GT_MJX)
    # is a UNIFORM ramp at that dt -- not the raw mocap timestamps, which don't frame-align
    # to the decimated processed data. We only read the time files to recover the sampling
    # RATE. Time_for_pos.npy is the authoritative position time; Time.npy is the fallback
    # (and may instead hold the raw mocap rate, e.g. 2000 Hz, which we reject).
    for time_name in ("Time_for_pos.npy", "Time.npy"):
        tpath = paths.motion_dir / time_name
        if not tpath.exists():
            continue
        try:
            raw_time = load_array(tpath, 1)
        except Exception:
            continue
        if len(raw_time) < 2:
            continue
        dt = float(np.median(np.diff(raw_time)))
        if abs(dt - PROCESSED_DT_S) <= _PROCESSED_DT_TOL * PROCESSED_DT_S:
            return np.arange(n_frames, dtype=np.float64) * dt
    return np.arange(n_frames, dtype=np.float64) * PROCESSED_DT_S


def parse_model_coordinate_names(model_path: Path) -> list[str]:
    text = model_path.read_text(errors="replace")
    return re.findall(r'<Coordinate\s+name="([^"]+)"', text)


def resolve_coordinate_names(model_path: Path, warnings: list[str]) -> list[str]:
    model_coordinates = parse_model_coordinate_names(model_path)
    model_set = set(model_coordinates)
    resolved: list[str] = []
    missing: list[str] = []
    for source_name, _is_translation in NPY_COORDINATES:
        candidates = (source_name, *COORDINATE_ALIASES.get(source_name, ()))
        match = next((name for name in candidates if name in model_set), None)
        if match is None:
            missing.append(f"{source_name} ({', '.join(candidates)})")
            resolved.append(candidates[-1])
        else:
            resolved.append(match)
    if missing:
        warnings.append(
            "Could not confirm these coordinates in the model; using alias names anyway: "
            + "; ".join(missing)
        )
    return resolved


def make_coordinates_matrix(pos: np.ndarray, model_path: Path, warnings: list[str]) -> tuple[list[str], np.ndarray]:
    if pos.ndim != 2 or pos.shape[1] != len(NPY_COORDINATES):
        raise ValueError(f"Pos array has shape {pos.shape}, expected (T, {len(NPY_COORDINATES)})")
    names = resolve_coordinate_names(model_path, warnings)
    coords = pos.copy()
    for idx, (_name, is_translation) in enumerate(NPY_COORDINATES):
        if not is_translation:
            coords[:, idx] = np.rad2deg(coords[:, idx])
    return names, coords


def make_force_matrix(time: np.ndarray, grf: np.ndarray, cop: np.ndarray, grm: np.ndarray) -> np.ndarray:
    for name, arr in (("GRF", grf), ("COP", cop), ("GRM", grm)):
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"{name} array has shape {arr.shape}, expected (T, 6)")
        if arr.shape[0] != len(time):
            raise ValueError(f"{name} has {arr.shape[0]} frames, expected {len(time)}")
    return np.column_stack(
        [
            time,
            grf[:, 0:3],
            cop[:, 0:3],
            grm[:, 0:3],
            grf[:, 3:6],
            cop[:, 3:6],
            grm[:, 3:6],
        ]
    )


def load_motion_source(paths: TrialPaths, use_noised: bool, warnings: list[str]) -> dict:
    pos_name = "Pos_noised.npy" if use_noised else "Pos.npy"
    vel_name = "Vel_noised.npy" if use_noised else "Vel.npy"
    accel_name = "Accel_noised.npy" if use_noised else "Accel.npy"
    pos = load_array(paths.motion_dir / pos_name, 2)
    vel = load_array(paths.motion_dir / vel_name, 2)
    accel = load_array(paths.motion_dir / accel_name, 2)
    grf = load_array(paths.motion_dir / "GRF.npy", 2)
    cop = load_array(paths.motion_dir / "COP.npy", 2)
    grm = load_array(paths.motion_dir / "GRM.npy", 2)
    force_time = load_array(paths.motion_dir / "Time.npy", 1)
    kin_time_path = paths.motion_dir / "Time_for_pos.npy"
    kin_time = load_array(kin_time_path, 1) if kin_time_path.exists() else force_time
    if not kin_time_path.exists():
        warnings.append("Time_for_pos.npy not found; using Time.npy for kinematics")
    kin_time = fit_time_to_length(kin_time, pos.shape[0], "kinematics time")
    force_time = fit_time_to_length(force_time, grf.shape[0], "force time")
    if vel.shape != pos.shape:
        raise ValueError(f"{vel_name} has shape {vel.shape}, expected {pos.shape}")
    if accel.shape != pos.shape:
        raise ValueError(f"{accel_name} has shape {accel.shape}, expected {pos.shape}")
    return {
        "source_files": {
            "pos": str(paths.motion_dir / pos_name),
            "vel": str(paths.motion_dir / vel_name),
            "accel": str(paths.motion_dir / accel_name),
            "grf": str(paths.motion_dir / "GRF.npy"),
            "cop": str(paths.motion_dir / "COP.npy"),
            "grm": str(paths.motion_dir / "GRM.npy"),
            "force_time": str(paths.motion_dir / "Time.npy"),
            "kinematics_time": str(kin_time_path if kin_time_path.exists() else paths.motion_dir / "Time.npy"),
        },
        "pos": pos,
        "grf": grf,
        "cop": cop,
        "grm": grm,
        "kin_time": kin_time,
        "force_time": force_time,
        "frame_notes": "raw Motion arrays after only generator-side unit conversion",
    }


def load_processed_source(paths: TrialPaths, use_noised: bool, warnings: list[str]) -> dict:
    suffix = "_noised" if use_noised else ""
    proc = paths.processed_dir
    qpos_path = proc / f"pos_mjx{suffix}.npy"
    grf_name = "GRF_NoFilt_Trimmed.npy"
    info_path = proc / "Trial_Processing_Information.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            grf_name = str(info.get("grf_torque_source") or grf_name)
        except Exception as exc:
            warnings.append(f"Could not read Trial_Processing_Information.json: {exc}")
    grf_mjx_path = proc / grf_name
    if not grf_mjx_path.exists():
        fallback = proc / "GRF_Cleaned.npy"
        if fallback.exists():
            warnings.append(f"{grf_name} not found; falling back to GRF_Cleaned.npy")
            grf_mjx_path = fallback
    moment_mjx_path = proc / "Moment_Cleaned.npy"
    cop_rel_path = proc / f"COP_Cleaned_Relative{suffix}.npy"
    cop_back_to_world_path = proc / f"COP_CalcFrame_GroundAligned_BackToWorld{suffix}.npy"
    ankle_r_path = proc / f"ankle_pos_r{suffix}.npy"
    ankle_l_path = proc / f"ankle_pos_l{suffix}.npy"
    qpos = load_array(qpos_path, 2)
    grf_mjx = load_array(grf_mjx_path, 2)
    moment_mjx = load_array(moment_mjx_path, 2)
    cop_rel = load_array(cop_rel_path, 2)
    ankle_r = load_array(ankle_r_path, 2)
    ankle_l = load_array(ankle_l_path, 2)
    n_frames = qpos.shape[0]
    for name, arr, width in (
        (grf_mjx_path.name, grf_mjx, 6),
        (moment_mjx_path.name, moment_mjx, 6),
        (cop_rel_path.name, cop_rel, 4),
        (ankle_r_path.name, ankle_r, 3),
        (ankle_l_path.name, ankle_l, 3),
    ):
        if arr.shape != (n_frames, width):
            raise ValueError(f"{name} has shape {arr.shape}, expected {(n_frames, width)}")
    pos = qpos_to_npy_coordinates(qpos)
    grf = np.hstack([mjx_to_opensim_coords(grf_mjx[:, 0:3]), mjx_to_opensim_coords(grf_mjx[:, 3:6])])
    grm = np.hstack([mjx_to_opensim_coords(moment_mjx[:, 0:3]), mjx_to_opensim_coords(moment_mjx[:, 3:6])])
    cop_mjx = np.zeros((n_frames, 6), dtype=np.float64)
    if cop_back_to_world_path.exists():
        cop_back_to_world = load_array(cop_back_to_world_path, 2)
        if cop_back_to_world.shape != (n_frames, 6):
            raise ValueError(
                f"{cop_back_to_world_path.name} has shape {cop_back_to_world.shape}, "
                f"expected {(n_frames, 6)}"
            )
        cop_mjx[:, 0:3] = ankle_r + cop_back_to_world[:, 0:3]
        cop_mjx[:, 3:6] = ankle_l + cop_back_to_world[:, 3:6]
        cop_source_note = "COP_CalcFrame_GroundAligned_BackToWorld + ankle positions"
    else:
        cop_mjx[:, 0:3] = ankle_r
        cop_mjx[:, 3:6] = ankle_l
        cop_mjx[:, 0:2] += cop_rel[:, 0:2]
        cop_mjx[:, 3:5] += cop_rel[:, 2:4]
        cop_mjx[:, 2] = 0.0
        cop_mjx[:, 5] = 0.0
        cop_source_note = "COP_Cleaned_Relative XY + floor Z fallback"
    cop = np.hstack([mjx_to_opensim_coords(cop_mjx[:, 0:3]), mjx_to_opensim_coords(cop_mjx[:, 3:6])])
    time = processed_time_vector(paths, n_frames)
    warnings.append(
        "Using processed MJX-aligned arrays; GRF/COP/moments are converted from MJX Z-up "
        "back to OpenSim Y-up for the .mot files."
    )
    return {
        "source_files": {
            "pos_mjx": str(qpos_path),
            "grf": str(grf_mjx_path),
            "cop_relative": str(cop_rel_path),
            "cop_full_frame": str(cop_back_to_world_path) if cop_back_to_world_path.exists() else cop_source_note,
            "ankle_pos_r": str(ankle_r_path),
            "ankle_pos_l": str(ankle_l_path),
            "grm": str(moment_mjx_path),
            "kinematics_time": "processed motion-aligned time vector",
            "force_time": "processed motion-aligned time vector",
        },
        "pos": pos,
        "grf": grf,
        "cop": cop,
        "grm": grm,
        "kin_time": time,
        "force_time": time,
        "frame_notes": (
            "processed MJX/MuJoCo Z-up arrays converted back to OpenSim Y-up; "
            f"COP reconstructed from {cop_source_note}"
        ),
    }


def write_storage(
    path: Path,
    name: str,
    columns: Iterable[str],
    data: np.ndarray,
    *,
    in_degrees: bool | None = None,
) -> None:
    columns = list(columns)
    with path.open("w") as f:
        f.write(f"{name}\n")
        f.write("version=1\n")
        f.write(f"nRows={data.shape[0]}\n")
        f.write(f"nColumns={len(columns)}\n")
        if in_degrees is not None:
            f.write(f"inDegrees={'yes' if in_degrees else 'no'}\n")
        f.write("endheader\n")
        f.write("\t".join(columns) + "\n")
        for row in data:
            f.write("\t".join(f"{float(value):.10g}" for value in row) + "\n")


def indent_xml(element: ET.Element) -> None:
    ET.indent(element, space="\t")


def write_external_loads_xml(
    path: Path,
    force_file: Path,
    coordinates_file: Path,
    right_body: str,
    left_body: str,
) -> None:
    root = ET.Element("OpenSimDocument", {"Version": "40000"})
    loads = ET.SubElement(root, "ExternalLoads", {"name": "generated_external_loads"})
    objects = ET.SubElement(loads, "objects")
    for side_name, prefix, body in (
        ("Right_GRF", "R_ground", right_body),
        ("Left_GRF", "L_ground", left_body),
    ):
        force = ET.SubElement(objects, "ExternalForce", {"name": side_name})
        ET.SubElement(force, "applied_to_body").text = body
        ET.SubElement(force, "force_expressed_in_body").text = "ground"
        ET.SubElement(force, "point_expressed_in_body").text = "ground"
        ET.SubElement(force, "force_identifier").text = f"{prefix}_force_v"
        ET.SubElement(force, "point_identifier").text = f"{prefix}_force_p"
        ET.SubElement(force, "torque_identifier").text = f"{prefix}_torque_"
        ET.SubElement(force, "data_source_name").text = "Unassigned"
    ET.SubElement(loads, "groups")
    ET.SubElement(loads, "datafile").text = str(force_file)
    ET.SubElement(loads, "external_loads_model_kinematics_file").text = str(coordinates_file)
    ET.SubElement(loads, "lowpass_cutoff_frequency_for_load_kinematics").text = "-1"
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def create_patella_free_model(model_path: Path, output_dir: Path) -> Path:
    """Return a patella-free copy of *model_path* cached in *output_dir*.

    Removes patella bodies, patellofemoral joints, CoordinateCouplerConstraints
    that couple knee_angle_r/l_beta, and the ForceSet (muscles are unused in ID
    and their path-points reference the patella body).  Without this stripping,
    the patellofemoral CoordinateCouplerConstraint introduces Lagrange multipliers
    of ~10,000 N·m that corrupt the hip/knee ID torques.
    """
    out_path = output_dir / "OpenSimModel_NoPatel.osim"
    if out_path.exists() and out_path.stat().st_mtime >= model_path.stat().st_mtime:
        return out_path

    tree = ET.parse(model_path)
    root = tree.getroot()

    for set_tag, name_pred in (
        (".//BodySet/objects", lambda n: "patella" in n),
        (".//JointSet/objects", lambda n: "patellofemoral" in n),
        (".//ConstraintSet/objects", lambda n: "patellofemoral" in n),
    ):
        container = root.find(set_tag)
        if container is not None:
            for elem in list(container):
                if name_pred(elem.get("name", "")):
                    container.remove(elem)

    force_objects = root.find(".//ForceSet/objects")
    if force_objects is not None:
        for elem in list(force_objects):
            force_objects.remove(elem)

    output_dir.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def write_id_setup_xml(
    path: Path,
    model_path: Path,
    coordinates_file: Path,
    external_loads_file: Path,
    results_dir: Path,
    output_file: str,
    time_range: tuple[float, float],
) -> None:
    root = ET.Element("OpenSimDocument", {"Version": "40000"})
    tool = ET.SubElement(root, "InverseDynamicsTool", {"name": "generated_inverse_dynamics"})
    ET.SubElement(tool, "model_file").text = str(model_path)
    ET.SubElement(tool, "coordinates_file").text = str(coordinates_file)
    ET.SubElement(tool, "time_range").text = f"{time_range[0]:.10g} {time_range[1]:.10g}"
    ET.SubElement(tool, "external_loads_file").text = str(external_loads_file)
    ET.SubElement(tool, "results_directory").text = str(results_dir)
    ET.SubElement(tool, "output_gen_force_file").text = output_file
    ET.SubElement(tool, "lowpass_cutoff_frequency_for_coordinates").text = "-1"
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def generate_trial_inputs(
    paths: TrialPaths,
    *,
    use_noised: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    right_body: str = DEFAULT_RIGHT_BODY,
    left_body: str = DEFAULT_LEFT_BODY,
    source: str = SOURCE_MOTION,
) -> dict:
    warnings: list[str] = []
    if not paths.model_path.exists():
        raise FileNotFoundError(f"Missing subject model: {paths.model_path}")

    output_dir = paths.output_dir
    coordinates_file = output_dir / "coordinates.mot"
    forces_file = output_dir / "ground_reaction.mot"
    external_loads_file = output_dir / "external_loads.xml"
    id_setup_file = output_dir / "id_setup.xml"
    manifest_file = output_dir / "input_generation_manifest.json"
    output_id_file = "inverse_dynamics.sto"
    id_model_path = create_patella_free_model(paths.model_path, output_dir)

    generated_files = [coordinates_file, forces_file, external_loads_file, id_setup_file]
    if source == SOURCE_MOTION:
        loaded = load_motion_source(paths, use_noised, warnings)
    elif source == SOURCE_PROCESSED:
        loaded = load_processed_source(paths, use_noised, warnings)
    else:
        raise ValueError(f"Unknown source '{source}'. Expected '{SOURCE_MOTION}' or '{SOURCE_PROCESSED}'.")

    if not overwrite and not dry_run and all(path.exists() for path in generated_files):
        try:
            existing_manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
        except Exception:
            existing_manifest = {}
        current_shapes = {
            "pos": list(loaded["pos"].shape),
            "grf": list(loaded["grf"].shape),
            "cop": list(loaded["cop"].shape),
            "grm": list(loaded["grm"].shape),
        }
        if (
            existing_manifest.get("source") == source
            and bool(existing_manifest.get("use_noised")) == bool(use_noised)
            and existing_manifest.get("shapes") == current_shapes
        ):
            return {
                "trial": str(paths.trial_dir),
                "status": "skipped",
                "reason": "generated files already exist for requested source",
                "source": source,
                "output_dir": str(output_dir),
                "generated_files": [str(path) for path in generated_files],
            }

    pos = loaded["pos"]
    grf = loaded["grf"]
    cop = loaded["cop"]
    grm = loaded["grm"]
    kin_time = loaded["kin_time"]
    force_time = loaded["force_time"]
    coordinate_names, coordinate_values = make_coordinates_matrix(pos, paths.model_path, warnings)
    coordinate_data = np.column_stack([kin_time, coordinate_values])
    force_data = make_force_matrix(force_time, grf, cop, grm)

    if len(kin_time) and len(force_time):
        if not math.isclose(float(kin_time[0]), float(force_time[0]), abs_tol=1e-6):
            warnings.append("kinematic and force time vectors have different start times")
        if not math.isclose(float(kin_time[-1]), float(force_time[-1]), abs_tol=1e-6):
            warnings.append("kinematic and force time vectors have different end times")

    manifest = {
        "trial": str(paths.trial_dir),
        "subject": paths.subject_dir.name,
        "model": str(paths.model_path),
        "id_model": str(id_model_path),
        "motion_dir": str(paths.motion_dir),
        "output_dir": str(output_dir),
        "use_noised": use_noised,
        "source": source,
        "source_files": loaded["source_files"],
        "frame_notes": loaded["frame_notes"],
        "shapes": {
            "pos": list(pos.shape),
            "grf": list(grf.shape),
            "cop": list(cop.shape),
            "grm": list(grm.shape),
        },
        "time_range": [float(max(kin_time[0], force_time[0])), float(min(kin_time[-1], force_time[-1]))],
        "coordinate_columns": coordinate_names,
        "force_columns": FORCE_COLUMNS,
        "right_body": right_body,
        "left_body": left_body,
        "generated_files": [str(path) for path in generated_files],
        "id_output_file": str(output_dir / output_id_file),
        "warnings": warnings,
        "status": "dry_run" if dry_run else "ok",
    }

    if dry_run:
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    write_storage(coordinates_file, "Coordinates", ["time", *coordinate_names], coordinate_data, in_degrees=True)
    write_storage(forces_file, "GroundReaction", FORCE_COLUMNS, force_data)
    write_external_loads_xml(external_loads_file, forces_file, coordinates_file, right_body, left_body)
    write_id_setup_xml(
        id_setup_file,
        id_model_path,
        coordinates_file,
        external_loads_file,
        output_dir,
        output_id_file,
        (manifest["time_range"][0], manifest["time_range"][1]),
    )
    with manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--trial", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument("--right-body", default=DEFAULT_RIGHT_BODY)
    parser.add_argument("--left-body", default=DEFAULT_LEFT_BODY)
    parser.add_argument("--source", choices=(SOURCE_MOTION, SOURCE_PROCESSED), default=SOURCE_PROCESSED)
    parser.add_argument("--use-noised", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        return 2

    trials = discover_trials(
        dataset_root,
        subject=args.subject,
        trial=args.trial,
        model_name=args.model_name,
        output_dir_name=args.output_dir_name,
    )
    if args.limit is not None:
        trials = trials[: args.limit]

    manifest = {
        "dataset_root": str(dataset_root),
        "trials_seen": len(trials),
        "trials_ok": 0,
        "trials_skipped": 0,
        "failures": [],
        "results": [],
    }
    for paths in trials:
        try:
            result = generate_trial_inputs(
                paths,
                use_noised=args.use_noised,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                right_body=args.right_body,
                left_body=args.left_body,
                source=args.source,
            )
            manifest["results"].append(result)
            if result.get("status") == "skipped":
                manifest["trials_skipped"] += 1
            else:
                manifest["trials_ok"] += 1
        except Exception as exc:
            manifest["failures"].append({"trial": str(paths.trial_dir), "error": str(exc)})

    print(json.dumps(manifest, indent=2))
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
