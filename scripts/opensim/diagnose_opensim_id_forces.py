#!/usr/bin/env python3
"""Diagnose OpenSim ID generalized-force contributors for one trial.

Run this with an environment that can import OpenSim, e.g.

    conda run -n opensim-nmd python scripts/opensim/diagnose_opensim_id_forces.py \
        TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded/02/Trial_11
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


from paths import REPO_ROOT
DEFAULT_OUTPUT_DIR_NAME = "OpenSimResults"
DEFAULT_PROCESSED_DIR_NAME = "ProcessedData"
TRANSLATIONAL_COORDS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}

QFRC_NAMES = [
    "pelvis_tx_force",
    "pelvis_ty_force",
    "pelvis_tz_force",
    "pelvis_tilt_moment",
    "pelvis_list_moment",
    "pelvis_rotation_moment",
    "hip_flexion_r_moment",
    "hip_adduction_r_moment",
    "hip_rotation_r_moment",
    "walker_knee_r_translation1_force",
    "walker_knee_r_translation2_force",
    "knee_angle_r_moment",
    "walker_knee_r_rotation2_moment",
    "walker_knee_r_rotation3_moment",
    "ankle_angle_r_moment",
    "subtalar_angle_r_moment",
    "mtp_angle_r_moment",
    "hip_flexion_l_moment",
    "hip_adduction_l_moment",
    "hip_rotation_l_moment",
    "walker_knee_l_translation1_force",
    "walker_knee_l_translation2_force",
    "knee_angle_l_moment",
    "walker_knee_l_rotation2_moment",
    "walker_knee_l_rotation3_moment",
    "ankle_angle_l_moment",
    "subtalar_angle_l_moment",
    "mtp_angle_l_moment",
    "lumbar_extension_moment",
    "lumbar_bending_moment",
    "lumbar_rotation_moment",
]

WATCH_COORDS = [
    "pelvis_tilt_moment",
    "pelvis_list_moment",
    "pelvis_rotation_moment",
    "hip_flexion_r_moment",
    "hip_adduction_r_moment",
    "hip_rotation_r_moment",
    "knee_angle_r_moment",
    "ankle_angle_r_moment",
    "subtalar_angle_r_moment",
    "hip_flexion_l_moment",
    "hip_adduction_l_moment",
    "hip_rotation_l_moment",
    "knee_angle_l_moment",
    "ankle_angle_l_moment",
    "subtalar_angle_l_moment",
    "lumbar_extension_moment",
    "lumbar_bending_moment",
    "lumbar_rotation_moment",
]


@dataclass
class StorageData:
    name: str
    columns: list[str]
    data: np.ndarray
    in_degrees: bool

    @property
    def time(self) -> np.ndarray:
        return self.data[:, 0]


def import_opensim():
    try:
        import opensim as osim  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Could not import opensim. Run this with an OpenSim-enabled environment "
            "such as conda env 'opensim-nmd'."
        ) from exc
    return osim


def read_storage(path: Path) -> StorageData:
    lines = path.read_text(errors="replace").splitlines()
    header_end = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    in_degrees = any(line.strip().lower() == "indegrees=yes" for line in lines[:header_end])
    columns = lines[header_end + 1].split()
    rows = [[float(token) for token in line.split()] for line in lines[header_end + 2 :] if line.strip()]
    return StorageData(path.stem, columns, np.asarray(rows, dtype=np.float64), in_degrees)


def finite_stats(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    mask = np.isfinite(a)
    if not np.any(mask):
        return {"mean": math.nan, "median": math.nan, "max_abs": math.nan, "rms": math.nan}
    x = a[mask]
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max_abs": float(np.max(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x * x))),
    }


def compare_series(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return {"rmse": math.nan, "mae": math.nan, "corr": math.nan, "bias": math.nan}
    x = a[mask]
    y = b[mask]
    diff = x - y
    corr = math.nan
    if float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
        corr = float(np.corrcoef(x, y)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "corr": corr,
        "bias": float(np.mean(diff)),
    }


def vector_to_numpy(vector) -> np.ndarray:
    if hasattr(vector, "to_numpy"):
        return np.asarray(vector.to_numpy(), dtype=np.float64)
    return np.asarray([float(vector.get(i)) for i in range(vector.size())], dtype=np.float64)


def coordinate_values(storage: StorageData, coordinate_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = storage.time
    raw = np.zeros((storage.data.shape[0], len(coordinate_names)), dtype=np.float64)
    column_index = {name: idx for idx, name in enumerate(storage.columns)}
    missing: list[str] = []
    for idx, name in enumerate(coordinate_names):
        if name not in column_index:
            missing.append(name)
            continue
        values = storage.data[:, column_index[name]]
        if storage.in_degrees and name not in TRANSLATIONAL_COORDS:
            values = np.deg2rad(values)
        raw[:, idx] = values
    if missing:
        print(
            "[warn] coordinate file missing coordinates; using default zeros for: "
            + ", ".join(missing),
            file=sys.stderr,
        )
    edge_order = 2 if len(time) >= 3 else 1
    qdot = np.gradient(raw, time, axis=0, edge_order=edge_order)
    qddot = np.gradient(qdot, time, axis=0, edge_order=edge_order)
    return raw, qdot, qddot


def build_model(model_path: Path, external_loads_path: Path | None, constraints_enabled: bool):
    osim = import_opensim()
    model = osim.Model(str(model_path))
    if not constraints_enabled:
        constraints = model.updConstraintSet()
        for i in range(constraints.getSize()):
            constraints.get(i).set_isEnforced(False)
    if external_loads_path is not None:
        loads = osim.ExternalLoads(str(external_loads_path), True)
        loads.connectToModel(model)
    state = model.initSystem()
    coordinate_set = model.getCoordinateSet()
    coordinate_names = [coordinate_set.get(i).getName() for i in range(coordinate_set.getSize())]
    solver = osim.InverseDynamicsSolver(model)
    return osim, model, state, coordinate_set, coordinate_names, solver


def solve_model_series(
    model_path: Path,
    external_loads_path: Path | None,
    coordinates: StorageData,
    frame_indices: Iterable[int],
    *,
    constraints_enabled: bool,
) -> tuple[list[str], np.ndarray]:
    osim, model, state, coordinate_set, coordinate_names, solver = build_model(
        model_path,
        external_loads_path,
        constraints_enabled=constraints_enabled,
    )
    q, qdot, qddot = coordinate_values(coordinates, coordinate_names)
    out = np.zeros((len(list(frame_indices)), len(coordinate_names)), dtype=np.float64)
    frame_indices = list(frame_indices)
    for out_idx, frame_idx in enumerate(frame_indices):
        state.setTime(float(coordinates.time[frame_idx]))
        for coord_idx, name in enumerate(coordinate_names):
            coord = coordinate_set.get(coord_idx)
            coord.setValue(state, float(q[frame_idx, coord_idx]), False)
            coord.setSpeedValue(state, float(qdot[frame_idx, coord_idx]))
        model.realizeVelocity(state)
        udot = osim.Vector(state.getNU(), 0.0)
        for i in range(min(state.getNU(), qddot.shape[1])):
            udot.set(i, float(qddot[frame_idx, i]))
        tau = solver.solve(state, udot)
        out[out_idx, :] = vector_to_numpy(tau)
    return coordinate_names, out


def storage_subset_by_frames(storage: StorageData, frame_indices: list[int]) -> np.ndarray:
    return storage.data[np.asarray(frame_indices, dtype=int), :]


def qfrc_mjx_subset(trial_dir: Path, frame_indices: list[int], target: str) -> tuple[list[str], np.ndarray] | None:
    path = trial_dir / DEFAULT_PROCESSED_DIR_NAME / target
    if not path.exists():
        return None
    data = np.asarray(np.load(path), dtype=np.float64)
    if max(frame_indices, default=-1) >= data.shape[0]:
        return None
    return QFRC_NAMES, data[np.asarray(frame_indices, dtype=int)]


def summarize_components(
    coordinate_names: list[str],
    tool_storage: StorageData,
    frame_indices: list[int],
    constrained_with_loads: np.ndarray,
    unconstrained_with_loads: np.ndarray,
    constrained_without_loads: np.ndarray,
    trial_dir: Path,
    tool_constraints_disabled: StorageData | None = None,
    tool_no_external_loads: StorageData | None = None,
) -> tuple[list[dict], dict]:
    tool_subset = storage_subset_by_frames(tool_storage, frame_indices)
    tool_cols = {name: i for i, name in enumerate(tool_storage.columns)}
    coord_to_solver_idx = {name: i for i, name in enumerate(coordinate_names)}

    rows: list[dict] = []
    for output_name in WATCH_COORDS:
        coord_name = output_name.removesuffix("_moment").removesuffix("_force")
        if coord_name not in coord_to_solver_idx or output_name not in tool_cols:
            continue
        j = coord_to_solver_idx[coord_name]
        tool_values = tool_subset[:, tool_cols[output_name]]
        solver_values = constrained_with_loads[:, j]
        no_constraint_values = unconstrained_with_loads[:, j]
        no_external_values = constrained_without_loads[:, j]
        constraint_effect = solver_values - no_constraint_values
        external_effect = no_external_values - solver_values
        solver_vs_tool = compare_series(solver_values, tool_values)
        constraint_effect_tool = np.full_like(tool_values, np.nan)
        external_effect_tool = np.full_like(tool_values, np.nan)
        if tool_constraints_disabled is not None and output_name in tool_constraints_disabled.columns:
            disabled_subset = storage_subset_by_frames(tool_constraints_disabled, frame_indices)
            disabled_cols = {name: i for i, name in enumerate(tool_constraints_disabled.columns)}
            constraint_effect_tool = tool_values - disabled_subset[:, disabled_cols[output_name]]
        if tool_no_external_loads is not None and output_name in tool_no_external_loads.columns:
            no_ext_subset = storage_subset_by_frames(tool_no_external_loads, frame_indices)
            no_ext_cols = {name: i for i, name in enumerate(tool_no_external_loads.columns)}
            external_effect_tool = no_ext_subset[:, no_ext_cols[output_name]] - tool_values
        row = {
            "coordinate": coord_name,
            "output_column": output_name,
            "tool_rms": finite_stats(tool_values)["rms"],
            "solver_rms": finite_stats(solver_values)["rms"],
            "solver_minus_tool_rmse": solver_vs_tool["rmse"],
            "solver_minus_tool_corr": solver_vs_tool["corr"],
            "constraint_effect_solver_rms": finite_stats(constraint_effect)["rms"],
            "constraint_effect_solver_max_abs": finite_stats(constraint_effect)["max_abs"],
            "external_effect_solver_rms": finite_stats(external_effect)["rms"],
            "external_effect_solver_max_abs": finite_stats(external_effect)["max_abs"],
            "constraint_effect_tool_rms": finite_stats(constraint_effect_tool)["rms"],
            "constraint_effect_tool_max_abs": finite_stats(constraint_effect_tool)["max_abs"],
            "external_effect_tool_rms": finite_stats(external_effect_tool)["rms"],
            "external_effect_tool_max_abs": finite_stats(external_effect_tool)["max_abs"],
            "mean_tool": finite_stats(tool_values)["mean"],
            "mean_solver": finite_stats(solver_values)["mean"],
            "mean_constraint_effect_solver": finite_stats(constraint_effect)["mean"],
            "mean_external_effect_solver": finite_stats(external_effect)["mean"],
            "mean_constraint_effect_tool": finite_stats(constraint_effect_tool)["mean"],
            "mean_external_effect_tool": finite_stats(external_effect_tool)["mean"],
        }
        rows.append(row)

    mjx_summary: dict[str, dict] = {}
    for mjx_name in ("ID_GT_MJX.npy", "qfrc_inverse.npy"):
        loaded = qfrc_mjx_subset(trial_dir, frame_indices, mjx_name)
        if loaded is None:
            continue
        mjx_names, mjx_data = loaded
        mjx_by_name = {name: i for i, name in enumerate(mjx_names)}
        comparisons = []
        for row in rows:
            out_col = str(row["output_column"])
            if out_col not in tool_cols or out_col not in mjx_by_name:
                continue
            tool_values = tool_subset[:, tool_cols[out_col]]
            mjx_values = mjx_data[:, mjx_by_name[out_col]]
            stat = compare_series(tool_values, mjx_values)
            comparisons.append({"output_column": out_col, **stat})
        if comparisons:
            mjx_summary[mjx_name] = {
                "median_rmse": float(np.median([c["rmse"] for c in comparisons])),
                "median_corr": float(np.nanmedian([c["corr"] for c in comparisons])),
                "by_coordinate": comparisons,
            }
    return rows, mjx_summary


def parse_polycoef(text: str) -> list[float]:
    vals = [float(token) for token in text.split()]
    return vals + [0.0] * (5 - len(vals))


def eval_mjx_poly(coeffs: list[float], theta: np.ndarray) -> np.ndarray:
    c = coeffs + [0.0] * (5 - len(coeffs))
    return c[0] + c[1] * theta + c[2] * theta**2 + c[3] * theta**3 + c[4] * theta**4


def eval_spline(x: list[float], y: list[float], theta: np.ndarray) -> np.ndarray:
    return np.interp(theta, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))


def extract_opensim_knee_splines(model_path: Path) -> dict[str, dict[str, dict]]:
    root = ET.parse(model_path).getroot()
    out: dict[str, dict[str, dict]] = {}
    for joint in root.iter("CustomJoint"):
        name = joint.attrib.get("name", "")
        if name not in {"walker_knee_r", "walker_knee_l"}:
            continue
        axes: dict[str, dict] = {}
        for axis in joint.iter("TransformAxis"):
            axis_name = axis.attrib.get("name", "")
            coords_el = axis.find("coordinates")
            if coords_el is None or not (coords_el.text or "").strip():
                continue
            spline = axis.find(".//SimmSpline")
            if spline is None:
                continue
            x_el = spline.find("x")
            y_el = spline.find("y")
            if x_el is None or y_el is None:
                continue
            axes[axis_name] = {
                "coordinate": (coords_el.text or "").strip(),
                "x": [float(v) for v in (x_el.text or "").split()],
                "y": [float(v) for v in (y_el.text or "").split()],
            }
        out[name] = axes
    return out


def coupled_coordinate_report(trial_dir: Path, model_path: Path) -> dict:
    processed_dir = trial_dir / DEFAULT_PROCESSED_DIR_NAME
    xml_path = trial_dir.parent / "MyosuiteModel_FIXED.xml"
    if not xml_path.exists():
        xml_path = trial_dir.parent / "MyosuiteModel.xml"
    qpos_path = processed_dir / "pos_mjx.npy"
    if not xml_path.exists() or not qpos_path.exists():
        return {"available": False, "reason": "missing MyosuiteModel XML or pos_mjx.npy"}
    qpos = np.asarray(np.load(qpos_path), dtype=np.float64)
    joint_index = {name: idx for idx, name in enumerate(QFRC_NAMES)}
    text = xml_path.read_text(errors="replace")
    mjx_rows = []
    for match in re.finditer(r'<joint[^>]+joint1="([^"]+)"[^>]+joint2="([^"]+)"[^>]+polycoef="([^"]+)"', text):
        slave, master, poly = match.groups()
        slave_key = f"{slave}_force" if "translation" in slave else f"{slave}_moment"
        master_key = f"{master}_moment"
        if slave_key not in joint_index or master_key not in joint_index:
            continue
        theta = qpos[:, joint_index[master_key]]
        stored = qpos[:, joint_index[slave_key]]
        expected = eval_mjx_poly(parse_polycoef(poly), theta)
        diff = stored - expected
        mjx_rows.append(
            {
                "slave": slave,
                "master": master,
                "polycoef_mjx_order_c0_to_c4": poly,
                "stored_minus_poly_rms": finite_stats(diff)["rms"],
                "stored_minus_poly_max_abs": finite_stats(diff)["max_abs"],
                "stored_rms": finite_stats(stored)["rms"],
                "expected_rms": finite_stats(expected)["rms"],
            }
        )
    opensim_splines = extract_opensim_knee_splines(model_path)
    spline_rows = []
    for side, knee_name in (("r", "knee_angle_r"), ("l", "knee_angle_l")):
        knee_key = f"{knee_name}_moment"
        if knee_key not in joint_index:
            continue
        theta = qpos[:, joint_index[knee_key]]
        for axis_name, spec in opensim_splines.get(f"walker_knee_{side}", {}).items():
            values = eval_spline(spec["x"], spec["y"], theta)
            spline_rows.append(
                {
                    "joint": f"walker_knee_{side}",
                    "axis": axis_name,
                    "coordinate": spec["coordinate"],
                    "opensim_spline_rms": finite_stats(values)["rms"],
                    "opensim_spline_min": float(np.min(values)),
                    "opensim_spline_max": float(np.max(values)),
                    "note": "OpenSim SimmSpline uses tabulated x/y values; MJX equality uses c0..c4 polynomial coefficients.",
                }
            )
    return {
        "available": True,
        "mjx_polynomial_checks": mjx_rows,
        "opensim_spline_ranges": spline_rows,
        "interpretation": (
            "MJX walker-knee coupled coordinates are polynomial equality constraints, "
            "while OpenSim walker-knee transforms are SimmSpline functions plus "
            "patellofemoral CoordinateCouplerConstraints. Do not copy MJX polycoef "
            "values into OpenSim coordinate columns."
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_constraints_disabled_model(source_model: Path, destination: Path) -> None:
    tree = ET.parse(source_model)
    root = tree.getroot()
    for constraint in root.iter("CoordinateCouplerConstraint"):
        enforced = constraint.find("isEnforced")
        if enforced is None:
            enforced = ET.SubElement(constraint, "isEnforced")
        enforced.text = "false"
    ET.indent(root, space="\t")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def write_id_setup_variant(
    source_setup: Path,
    destination: Path,
    *,
    model_file: Path | None = None,
    output_file: str,
    include_external_loads: bool = True,
) -> None:
    tree = ET.parse(source_setup)
    root = tree.getroot()
    for el in root.iter("output_gen_force_file"):
        el.text = output_file
    if model_file is not None:
        for el in root.iter("model_file"):
            el.text = str(model_file)
    if not include_external_loads:
        for el in root.iter("external_loads_file"):
            el.text = ""
    ET.indent(root, space="\t")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def run_inverse_dynamics_tool(setup_file: Path) -> None:
    osim = import_opensim()
    tool = osim.InverseDynamicsTool(str(setup_file))
    tool.run()


def run_tool_variants(output_dir: Path, model_path: Path, setup_path: Path, *, overwrite: bool) -> dict[str, Path]:
    outputs = {
        "constraints_disabled": output_dir / "inverse_dynamics_constraints_disabled.sto",
        "no_external_loads": output_dir / "inverse_dynamics_no_external_loads.sto",
    }
    disabled_model = output_dir / "diagnostic_constraints_disabled.osim"
    disabled_setup = output_dir / "id_setup_constraints_disabled.xml"
    no_external_setup = output_dir / "id_setup_no_external_loads.xml"

    if overwrite or not outputs["constraints_disabled"].exists():
        write_constraints_disabled_model(model_path, disabled_model)
        write_id_setup_variant(
            setup_path,
            disabled_setup,
            model_file=disabled_model,
            output_file=outputs["constraints_disabled"].name,
            include_external_loads=True,
        )
        run_inverse_dynamics_tool(disabled_setup)

    if overwrite or not outputs["no_external_loads"].exists():
        write_id_setup_variant(
            setup_path,
            no_external_setup,
            output_file=outputs["no_external_loads"].name,
            include_external_loads=False,
        )
        run_inverse_dynamics_tool(no_external_setup)

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir", type=Path, help="Path to <Subject>/Trial_<N>")
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument("--model-name", default="OpenSimModel.osim")
    parser.add_argument("--stride", type=int, default=10, help="Analyze every Nth frame (default 10).")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap after stride selection.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--skip-tool-variants", action="store_true")
    parser.add_argument("--overwrite-variants", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    trial_dir = args.trial_dir.resolve()
    subject_dir = trial_dir.parent
    output_dir = trial_dir / args.output_dir_name
    model_path = subject_dir / args.model_name
    coordinates_path = output_dir / "coordinates.mot"
    external_loads_path = output_dir / "external_loads.xml"
    setup_path = output_dir / "id_setup.xml"
    id_output_path = output_dir / "inverse_dynamics.sto"
    for path in (model_path, coordinates_path, external_loads_path, setup_path, id_output_path):
        if not path.exists():
            raise FileNotFoundError(path)

    coordinates = read_storage(coordinates_path)
    id_output = read_storage(id_output_path)
    variant_paths: dict[str, Path] = {}
    tool_constraints_disabled = None
    tool_no_external_loads = None
    if not args.skip_tool_variants:
        variant_paths = run_tool_variants(
            output_dir,
            model_path,
            setup_path,
            overwrite=args.overwrite_variants,
        )
        tool_constraints_disabled = read_storage(variant_paths["constraints_disabled"])
        tool_no_external_loads = read_storage(variant_paths["no_external_loads"])
    frame_indices = list(range(0, coordinates.data.shape[0], max(1, args.stride)))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("no frames selected")

    coord_names, constrained_with_loads = solve_model_series(
        model_path,
        external_loads_path,
        coordinates,
        frame_indices,
        constraints_enabled=True,
    )
    _, unconstrained_with_loads = solve_model_series(
        model_path,
        external_loads_path,
        coordinates,
        frame_indices,
        constraints_enabled=False,
    )
    _, constrained_without_loads = solve_model_series(
        model_path,
        None,
        coordinates,
        frame_indices,
        constraints_enabled=True,
    )

    component_rows, mjx_summary = summarize_components(
        coord_names,
        id_output,
        frame_indices,
        constrained_with_loads,
        unconstrained_with_loads,
        constrained_without_loads,
        trial_dir,
        tool_constraints_disabled=tool_constraints_disabled,
        tool_no_external_loads=tool_no_external_loads,
    )
    component_rows_sorted = sorted(
        component_rows,
        key=lambda row: float(row.get("constraint_effect_tool_rms", 0.0)),
        reverse=True,
    )
    payload = {
        "trial_dir": str(trial_dir),
        "model": str(model_path),
        "coordinates": str(coordinates_path),
        "external_loads": str(external_loads_path),
        "id_output": str(id_output_path),
        "frames_analyzed": len(frame_indices),
        "frame_stride": args.stride,
        "component_summary": component_rows_sorted,
        "tool_variant_outputs": {key: str(path) for key, path in variant_paths.items()},
        "mjx_comparison_summary": mjx_summary,
        "coupled_coordinate_report": coupled_coordinate_report(trial_dir, model_path),
        "notes": [
            "constraint_effect_tool = InverseDynamicsTool(normal) - InverseDynamicsTool(constraints disabled model)",
            "external_effect_tool = InverseDynamicsTool(no external loads) - InverseDynamicsTool(normal)",
            "solver_* fields are lower-level InverseDynamicsSolver checks; ExternalLoads are not applied there exactly like InverseDynamicsTool, so prefer *_tool fields for force-source conclusions.",
            "Large external_effect at hip/knee with small distal mismatch suggests an external-load point/frame or treadmill/global COP issue.",
            "Large constraint_effect at hip/knee suggests OpenSim constraint reactions are materially changing generalized forces.",
        ],
    }

    output_json = args.output_json or output_dir / "opensim_id_force_diagnostics.json"
    output_csv = args.output_csv or output_dir / "opensim_id_force_diagnostics.csv"
    with output_json.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    write_csv(output_csv, component_rows_sorted)

    print(json.dumps({
        "frames_analyzed": len(frame_indices),
        "json": str(output_json),
        "csv": str(output_csv),
        "tool_variant_outputs": {key: str(path) for key, path in variant_paths.items()},
        "top_constraint_effects": component_rows_sorted[:8],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
