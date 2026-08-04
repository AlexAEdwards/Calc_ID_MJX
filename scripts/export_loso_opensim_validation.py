#!/usr/bin/env python3
"""Export synchronized OpenSim validation inputs from LOSO predictions.

The force file contains the fine-tuned model's predictions without contact
masking.  COP is reconstructed with the *Video* ground-aligned rotations.
Kinematics come from Video/ProcessedData, except upper-body coordinates that
are copied from robustly aligned rows of the source walking1_opt.mot.  Coupled
coordinates are evaluated from the copied OpenSim model.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from paths import artifact, dataset  # noqa: E402


from paths import REPO_ROOT as PROJECT_ROOT
TRANSFORMER_DIR = PROJECT_ROOT / "TransformerFinal"
DEFAULT_DATASET = dataset("OpenCapWalkingTrunkSwaySubjects")
DEFAULT_LOSO = artifact(
    "outputs",
    "ReprocessedDataSet_July9_TorqueInformed_KAM_Weight_Corrected",
    "LOSO_video_evalTS_includeTS_KAMFirstStepRatio0p1_PredCOPKAM",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "OpenCapValSubjectsForScott"
SUBJECTS = ("subject2", "subject3", "subject4")
TRIAL = "trial_1"
MODEL_NAME = "OpenSimScaled_Video.osim"

PROCESSED_COORDINATES = (
    "pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "pelvis_list",
    "pelvis_rotation", "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l", "lumbar_extension",
    "lumbar_bending", "lumbar_rotation",
)
MJX_GENERALIZED_FORCE_COLUMNS = PROCESSED_COORDINATES
TRANSLATIONS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
UPPER_BODY = {
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
}
FORCE_COLUMNS = (
    "time",
    "R_ground_force_vx", "R_ground_force_vy", "R_ground_force_vz",
    "R_ground_force_px", "R_ground_force_py", "R_ground_force_pz",
    "R_ground_torque_x", "R_ground_torque_y", "R_ground_torque_z",
    "L_ground_force_vx", "L_ground_force_vy", "L_ground_force_vz",
    "L_ground_force_px", "L_ground_force_py", "L_ground_force_pz",
    "L_ground_torque_x", "L_ground_torque_y", "L_ground_torque_z",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def read_mot(path: Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None)
    if end is None or end + 1 >= len(lines):
        raise ValueError(f"Invalid OpenSim storage header: {path}")
    metadata: dict[str, str] = {}
    for line in lines[:end]:
        pieces = line.replace("=", " ").split(None, 1)
        if len(pieces) == 2:
            metadata[pieces[0].lower()] = pieces[1].strip()
    columns = lines[end + 1].split()
    rows = [[float(v) for v in line.split()] for line in lines[end + 2 :] if line.strip()]
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"Column count mismatch in {path}: {data.shape} vs {len(columns)}")
    return columns, data, metadata


def write_mot(path: Path, name: str, columns: list[str] | tuple[str, ...], data: np.ndarray,
              *, in_degrees: bool | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    degree_line = "" if in_degrees is None else f"inDegrees={'yes' if in_degrees else 'no'}\n"
    header = (
        f"{name}\nversion=1\nnRows={len(data)}\nnColumns={len(columns)}\n"
        f"range={data[0, 0]:.8f} {data[-1, 0]:.8f}\n{degree_line}endheader\n"
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(header)
        handle.write("\t".join(columns) + "\n")
        np.savetxt(handle, data, delimiter="\t", fmt="%.10f")


def model_coordinates_and_couplers(model_path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    root = ET.parse(model_path).getroot()
    coordinates: list[str] = []
    for elem in root.findall(".//Coordinate"):
        name = elem.attrib.get("name")
        if name and name not in coordinates:
            coordinates.append(name)
    couplers: dict[str, dict[str, Any]] = {}
    for constraint in root.findall(".//CoordinateCouplerConstraint"):
        enforced = (constraint.findtext("isEnforced") or "true").strip().lower() != "false"
        if not enforced:
            continue
        dependent = (constraint.findtext("dependent_coordinate_name") or "").strip()
        independent = (constraint.findtext("independent_coordinate_names") or "").split()
        function_root = constraint.find("coupled_coordinates_function")
        function = next(iter(function_root), None) if function_root is not None else None
        if not dependent or len(independent) != 1 or function is None:
            raise ValueError(f"Unsupported coordinate coupler in {model_path}: {constraint.attrib.get('name')}")
        coeff_text = function.findtext("coefficients") or ""
        couplers[dependent] = {
            "constraint": constraint.attrib.get("name", ""),
            "independent": independent[0],
            "function": function.tag.split("}")[-1],
            "coefficients": [float(v) for v in coeff_text.split()],
            "scale_factor": float(constraint.findtext("scale_factor") or 1.0),
        }
    if not coordinates:
        raise ValueError(f"No Coordinate objects found in {model_path}")
    return coordinates, couplers


def evaluate_coupler(spec: Mapping[str, Any], independent_degrees: np.ndarray) -> np.ndarray:
    # OpenSim coupling functions operate on internal radians. MOT angles are degrees.
    x = np.deg2rad(independent_degrees)
    coeff = np.asarray(spec["coefficients"], dtype=np.float64)
    kind = str(spec["function"])
    if kind == "LinearFunction" and coeff.size == 2:
        y = coeff[0] * x + coeff[1]
    elif kind == "PolynomialFunction" and coeff.size:
        y = np.polyval(coeff, x)
    else:
        raise ValueError(f"Unsupported OpenSim coupling function: {kind}")
    return np.rad2deg(float(spec["scale_factor"]) * y)


def processed_coordinates(processed_dir: Path) -> dict[str, np.ndarray]:
    q = np.asarray(np.load(processed_dir / "pos_mjx.npy"), dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(PROCESSED_COORDINATES):
        raise ValueError(f"Unexpected pos_mjx shape {q.shape} in {processed_dir}")
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(PROCESSED_COORDINATES):
        result[name] = q[:, index] if name in TRANSLATIONS else np.rad2deg(q[:, index])
    return result


def resolve_raw_alignment(
    processed: Mapping[str, np.ndarray], raw_columns: list[str], raw_data: np.ndarray,
    processing_info_path: Path,
) -> tuple[int, dict[str, Any]]:
    """Resolve processed index zero in raw MOT using kinematics, then verify metadata."""
    time = raw_data[:, raw_columns.index("time")]
    n_processed = len(next(iter(processed.values())))
    max_offset = len(raw_data) - n_processed
    if max_offset < 0:
        raise ValueError("Raw MOT is shorter than processed kinematics")
    shared = [name for name in PROCESSED_COORDINATES if name in raw_columns]
    scores: list[float] = []
    per_offset_channels: list[dict[str, float]] = []
    for offset in range(max_offset + 1):
        channel_scores: dict[str, float] = {}
        for name in shared:
            raw = raw_data[offset : offset + n_processed, raw_columns.index(name)]
            scale_floor = 0.01 if name in TRANSLATIONS else 0.5
            scale = max(float(np.nanstd(raw)), scale_floor)
            channel_scores[name] = float(np.nanmedian(np.abs(processed[name] - raw)) / scale)
        scores.append(float(np.nanmedian(list(channel_scores.values()))))
        per_offset_channels.append(channel_scores)
    order = np.argsort(scores)
    best = int(order[0])
    runner_up = float(scores[int(order[1])]) if len(order) > 1 else None
    separation_ratio = None if runner_up is None else runner_up / max(float(scores[best]), 1e-12)

    info = json.loads(processing_info_path.read_text(encoding="utf-8"))
    bounds = info.get("core_trim_bounds_motion_aligned")
    metadata_offset = int(bounds[0]) if isinstance(bounds, list) and len(bounds) == 2 else None
    metadata_reference = str(info.get("core_trim_reference_space", "unknown"))
    # Most current metadata is relative to an already motion-aligned stream, not
    # to walking1_opt.mot, so it cannot validate the raw-MOT offset. Only enforce
    # agreement when the metadata explicitly declares a raw reference space.
    metadata_comparable = "raw" in metadata_reference.lower()
    if metadata_comparable and metadata_offset is not None and abs(best - metadata_offset) > 1:
        raise ValueError(
            f"Data-driven raw offset {best} disagrees with processing metadata {metadata_offset}"
        )
    if separation_ratio is not None and separation_ratio < 1.20:
        raise ValueError(
            f"Raw alignment is ambiguous: best={best} score={scores[best]:.5g}, "
            f"runner-up ratio={separation_ratio:.3f}"
        )
    selected_time = time[best : best + n_processed]
    dt = np.diff(selected_time)
    if not np.all(np.isfinite(selected_time)) or not np.all(dt > 0):
        raise ValueError("Raw timestamps are non-finite or non-monotonic")
    median_dt = float(np.median(dt))
    if np.max(np.abs(dt - median_dt)) > max(1e-5, median_dt * 0.02):
        raise ValueError("Raw timestamps are not uniformly sampled within 2%")
    diagnostics = {
        "raw_frame_offset": best,
        "metadata_frame_offset": metadata_offset,
        "metadata_reference_space": metadata_reference,
        "metadata_offset_comparable_to_raw_mot": metadata_comparable,
        "processed_frame_count": n_processed,
        "raw_frame_count": len(raw_data),
        "best_robust_normalized_error": scores[best],
        "runner_up_error": runner_up,
        "runner_up_to_best_ratio": separation_ratio,
        "median_sample_period_seconds": median_dt,
        "shared_alignment_coordinates": shared,
        "per_coordinate_best_errors": per_offset_channels[best],
    }
    return best, diagnostics


def _source_hyperparameters(checkpoint: Mapping[str, Any], checkpoint_path: Path) -> Path:
    candidates = []
    if checkpoint.get("source_hyperparameters_path"):
        candidates.append(Path(str(checkpoint["source_hyperparameters_path"])))
    if checkpoint.get("source_checkpoint"):
        candidates.append(Path(str(checkpoint["source_checkpoint"])).parent / "hyperparameters.json")
    candidates.append(checkpoint_path.parent / "hyperparameters.json")
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate source hyperparameters for {checkpoint_path}")


def predict_unmasked(checkpoint_path: Path, trial_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Run the exact train-style window inference with contact masking forced off."""
    sys.path.insert(0, str(TRANSFORMER_DIR))
    import jax  # type: ignore
    import infer as infer_module  # type: ignore
    import loso_adapters  # type: ignore
    from data_loader import load_single_trial  # type: ignore

    # Checkpoints may contain Normalizer pickled from train.py's __main__ context.
    import train as train_module  # type: ignore
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "Normalizer"):
        setattr(main_module, "Normalizer", train_module.Normalizer)
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    params = checkpoint["params"]
    normalizers = checkpoint["normalizers"]
    hyper_path = _source_hyperparameters(checkpoint, checkpoint_path)
    raw_config = json.loads(hyper_path.read_text(encoding="utf-8"))
    config = loso_adapters.normalize_hyperparameters(raw_config, checkpoint_metadata=checkpoint)
    config.update(checkpoint.get("resolved_input_config") or {})
    config["input_dim"] = int(np.asarray(normalizers["input"].mean).shape[-1])
    config["static_dim"] = int(np.asarray(normalizers["static"].mean).shape[-1])
    try:
        config["output_dim"] = int(np.asarray(params["Dense_2"]["kernel"]).shape[-1])
    except Exception:
        config["output_dim"] = 14
    config["cop_mask"] = False

    data = load_single_trial(
        trial_root,
        trim_cop=bool(config.get("trim_cop", False)), deviation_learning=False,
        opencap_val=True, input_source="processed", use_noised=False,
        noised_gt=bool(config.get("noised_gt", False)),
        use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
        use_recalculated_opensim_id_gt=False, grf_grm_from_processed=False,
    )
    if data is None:
        raise RuntimeError(f"Failed to load trial: {trial_root}")
    features, include_pelvis, layout, blocks, _ = infer_module._resolve_train_style_inputs(
        data,
        requested_include_pelvis_euler=bool(config.get("include_pelvis_euler", False)),
        expected_input_dim=config["input_dim"],
    )
    features_z = np.asarray(normalizers["input"].normalize(features), dtype=np.float32)
    patient = np.asarray(data["patient_size"], dtype=np.float32).reshape(-1)
    patient = np.pad(patient[:4], (0, max(0, 4 - patient.size)))
    static = np.asarray([
        float(np.asarray(data["height"])[0, 0]), float(np.asarray(data["mass"])[0, 0]),
        float(data["gender"]), *patient[:4], float(data["forward_vel"]),
    ], dtype=np.float32)
    static_z = np.asarray(normalizers["static"].normalize(static), dtype=np.float32).squeeze()
    model = loso_adapters.build_loso_model(config, params)

    @jax.jit
    def predict_fn(model_params, x_batch, static_batch):
        return model.apply({"params": model_params}, x_batch, static_batch, train=False)

    _, output_metric, evaluation_mask, window_meta = infer_module._predict_with_train_style_windows(
        predict_fn=predict_fn, params=params, input_features_z=features_z,
        static_context_z=static_z, window_size=int(config.get("window_size", 110)),
        stride=int(config.get("stride", 16)), output_dim=int(config["output_dim"]),
        prediction_margin_frames=int(config.get("prediction_margin_frames", 20)),
    )
    predictions = infer_module._convert_output_to_physical_predictions(
        output_np=output_metric, data=data, normalizers=normalizers,
        detected_output_dim=int(config["output_dim"]), cop_mask=False,
        use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
        qfrc_inverse_output_dim=0, rotation_output_dim=0, jacobian_output_dim=0,
        use_gt_jacob_and_rot=False,
    )
    diagnostics = {
        "checkpoint": checkpoint_path.resolve(), "source_hyperparameters": hyper_path,
        "input_layout": layout, "input_blocks": blocks,
        "include_pelvis_euler": include_pelvis, "input_dim": features.shape[-1],
        "window": window_meta, "cop_mask": False,
        "use_grf_norm_cop": bool(config.get("use_grf_norm_cop", False)),
    }
    return {k: np.asarray(v) for k, v in predictions.items() if isinstance(v, (np.ndarray, jax.Array))}, np.asarray(evaluation_mask, dtype=bool), diagnostics


def mjx_to_opensim(array: np.ndarray) -> np.ndarray:
    """MJX Z-up (x,y,z) to OpenSim Y-up (x,z,-y)."""
    return np.stack((array[..., 0], array[..., 2], -array[..., 1]), axis=-1)


def reconstruct_absolute_cop(predicted_cop: np.ndarray, processed_dir: Path) -> np.ndarray:
    """Return T x 2 x 3 COP points in OpenSim world coordinates (meters)."""
    rotation = np.asarray(np.load(processed_dir / "WorldToGroundAlignedCalcnRotation.npy"))
    ankle_r = np.asarray(np.load(processed_dir / "ankle_pos_r.npy"))
    ankle_l = np.asarray(np.load(processed_dir / "ankle_pos_l.npy"))
    heights = np.asarray(np.load(processed_dir / "ankle_heights.npy"))
    n = len(predicted_cop)
    if rotation.shape[0] != n or ankle_r.shape[0] != n or ankle_l.shape[0] != n:
        raise ValueError("COP reconstruction inputs have inconsistent frame counts")
    if rotation.ndim == 3:
        rotation = np.repeat(rotation[:, None, :, :], 2, axis=1)
    if rotation.shape[1:] != (2, 3, 3):
        raise ValueError(f"Unexpected Video rotation shape: {rotation.shape}")
    heights = heights.reshape(n, -1)
    if heights.shape[1] == 1:
        heights = np.repeat(heights, 2, axis=1)
    ankles = np.stack((ankle_r, ankle_l), axis=1)
    local = np.empty((n, 2, 3), dtype=np.float64)
    local[:, 0, :] = np.stack((predicted_cop[:, 0], -heights[:, 0], predicted_cop[:, 1]), axis=-1)
    local[:, 1, :] = np.stack((predicted_cop[:, 2], -heights[:, 1], predicted_cop[:, 3]), axis=-1)
    relative_world = np.einsum("tfji,tfj->tfi", rotation, local)  # transpose(R) @ local
    return mjx_to_opensim(ankles + relative_world)


def force_table(times: np.ndarray, predictions: Mapping[str, np.ndarray], processed_dir: Path,
                selected: np.ndarray) -> np.ndarray:
    raw_grf = np.asarray(predictions["grf"])
    raw_moment = np.asarray(predictions["moments"])
    if raw_grf.ndim == 2 and raw_grf.shape[1] == 6:
        raw_grf = raw_grf.reshape(len(raw_grf), 2, 3)
    if raw_moment.ndim == 2 and raw_moment.shape[1] == 6:
        raw_moment = raw_moment.reshape(len(raw_moment), 2, 3)
    grf = mjx_to_opensim(raw_grf)
    moment = mjx_to_opensim(raw_moment)
    cop = reconstruct_absolute_cop(np.asarray(predictions["cop"]), processed_dir)
    if grf.shape[1:] != (2, 3) or moment.shape[1:] != (2, 3):
        raise ValueError(f"Unexpected prediction shapes: GRF {grf.shape}, moment {moment.shape}")
    parts = [times[:, None]]
    for foot in range(2):
        parts.extend((grf[selected, foot], cop[selected, foot], moment[selected, foot]))
    result = np.concatenate(parts, axis=1)
    if result.shape[1] != len(FORCE_COLUMNS) or not np.all(np.isfinite(result)):
        raise ValueError("Generated force table has an invalid shape or non-finite values")
    return result


def inverse_dynamics_tables(
    times: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    processed_dir: Path,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build predicted and ground-truth MJX net generalized-force tables.

    MJX inverse dynamics obeys qfrc_inverse = qfrc_net + tau_external, so the
    net generalized forces are qfrc_inverse - tau_external.  For the predicted
    table, tau_external is calculated from the fine-tuned LOSO GRF/COP/GRM.
    The ground-truth table is the previously computed MJX ID_GT_MJX array.
    """
    qfrc_inverse = np.asarray(predictions["qfrc_inverse"], dtype=np.float64)
    predicted_external = np.asarray(predictions["tau_grf"], dtype=np.float64)
    ground_truth = np.asarray(np.load(processed_dir / "ID_GT_MJX.npy"), dtype=np.float64)
    expected_shape = (len(qfrc_inverse), len(MJX_GENERALIZED_FORCE_COLUMNS))
    for label, array in (
        ("predicted qfrc_inverse", qfrc_inverse),
        ("predicted external generalized load", predicted_external),
        ("ground-truth ID_GT_MJX", ground_truth),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"Unexpected {label} shape {array.shape}; expected {expected_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} contains non-finite values")

    predicted_net = qfrc_inverse - predicted_external
    predicted_table = np.column_stack((times, predicted_net[selected]))
    ground_truth_table = np.column_stack((times, ground_truth[selected]))

    stored_qfrc_inverse = np.asarray(np.load(processed_dir / "qfrc_inverse.npy"), dtype=np.float64)
    stored_external = np.asarray(
        np.load(processed_dir / "qfrc_grf_contribution.npy"), dtype=np.float64
    )
    diagnostics = {
        "max_abs_inference_vs_stored_qfrc_inverse": float(
            np.max(np.abs(qfrc_inverse - stored_qfrc_inverse))
        ),
        "max_abs_ground_truth_identity_residual": float(
            np.max(np.abs(ground_truth - (stored_qfrc_inverse - stored_external)))
        ),
    }
    return predicted_table, ground_truth_table, diagnostics


def build_kinematic_table(
    times: np.ndarray, selected: np.ndarray, processed: Mapping[str, np.ndarray],
    raw_columns: list[str], raw_data: np.ndarray, raw_offset: int,
    model_coordinates: list[str], couplers: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    raw_rows = raw_offset + selected
    values: dict[str, np.ndarray] = {}
    for name in model_coordinates:
        if name in processed:
            values[name] = np.asarray(processed[name])[selected]
        elif name in UPPER_BODY:
            if name not in raw_columns:
                raise ValueError(f"Upper-body coordinate missing from raw MOT: {name}")
            values[name] = raw_data[raw_rows, raw_columns.index(name)]
    unresolved = set(model_coordinates) - set(values) - set(couplers)
    if unresolved:
        raise ValueError(f"No source for model coordinates: {sorted(unresolved)}")
    pending = dict(couplers)
    while pending:
        progressed = False
        for dependent, spec in list(pending.items()):
            independent = str(spec["independent"])
            if independent in values:
                values[dependent] = evaluate_coupler(spec, values[independent])
                del pending[dependent]
                progressed = True
        if not progressed:
            raise ValueError(f"Unresolvable coordinate couplers: {sorted(pending)}")
    result = np.column_stack([times, *[values[name] for name in model_coordinates]])
    if not np.all(np.isfinite(result)):
        raise ValueError("Generated kinematics contain non-finite values")
    return result


def export_subject(subject: str, args: argparse.Namespace) -> dict[str, Any]:
    source_subject = args.dataset_root / subject
    trial_root = source_subject / TRIAL
    processed_dir = trial_root / "Video/ProcessedData"
    raw_path = trial_root / "Video/Motion/Raw/walking1_opt.mot"
    model_path = source_subject / MODEL_NAME
    checkpoint = args.loso_root / "folds" / subject / "best_model.pkl"
    for required in (processed_dir, raw_path, model_path, checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    raw_columns, raw_data, raw_meta = read_mot(raw_path)
    processed = processed_coordinates(processed_dir)
    raw_offset, alignment = resolve_raw_alignment(
        processed, raw_columns, raw_data, processed_dir / "Trial_Processing_Information.json"
    )
    predictions, evaluation_mask, inference = predict_unmasked(checkpoint, trial_root)
    n = len(next(iter(processed.values())))
    if len(evaluation_mask) != n:
        raise ValueError(f"Evaluation mask length {len(evaluation_mask)} != kinematics length {n}")
    selected = np.flatnonzero(evaluation_mask)
    if not len(selected) or not np.all(np.diff(selected) == 1):
        raise ValueError("Expected a nonempty contiguous LOSO evaluation mask")
    raw_rows = raw_offset + selected
    times = raw_data[raw_rows, raw_columns.index("time")]

    model_coordinates, couplers = model_coordinates_and_couplers(model_path)
    kinematics = build_kinematic_table(
        times, selected, processed, raw_columns, raw_data, raw_offset,
        model_coordinates, couplers,
    )
    trimmed_raw = raw_data[raw_rows].copy()
    trimmed_raw[:, raw_columns.index("time")] = times
    forces = force_table(times, predictions, processed_dir, selected)
    predicted_id, ground_truth_id, id_diagnostics = inverse_dynamics_tables(
        times, predictions, processed_dir, selected
    )

    output_subject = args.output_dir / subject
    output_raw = output_subject / TRIAL / "Video/Motion/Raw"
    output_raw.mkdir(parents=True, exist_ok=True)
    copied_model = output_subject / MODEL_NAME
    shutil.copy2(model_path, copied_model)
    write_mot(output_raw / "walking1_forces.mot", "GroundReaction", FORCE_COLUMNS, forces)
    write_mot(output_raw / "walking1_opt_Inputs_To_MJXModel.mot", "Coordinates", ["time", *model_coordinates], kinematics, in_degrees=True)
    write_mot(output_raw / "walking1_opt_original.mot", "Coordinates", raw_columns, trimmed_raw, in_degrees=True)
    write_mot(
        output_raw / "walking1_torques_predicted_MJX.mot",
        "MJXPredictedNetInverseDynamics",
        ["time", *MJX_GENERALIZED_FORCE_COLUMNS],
        predicted_id,
    )
    write_mot(
        output_raw / "walking1_torques_ground_truth_MJX.mot",
        "MJXGroundTruthNetInverseDynamics",
        ["time", *MJX_GENERALIZED_FORCE_COLUMNS],
        ground_truth_id,
    )

    coordinate_delta = np.max(np.abs(kinematics[:, 1:] - trimmed_raw[:, [raw_columns.index(nm) for nm in model_coordinates]]), axis=0)
    manifest = {
        "subject": subject, "trial": TRIAL, "status": "ok",
        "source_model": model_path.resolve(), "cop_rotation_source": (processed_dir / "WorldToGroundAlignedCalcnRotation.npy").resolve(),
        "source_raw_kinematics": raw_path.resolve(), "checkpoint": checkpoint.resolve(),
        "output_model": copied_model.resolve(), "output_raw_directory": output_raw.resolve(),
        "frame_count": len(selected), "processed_indices_inclusive": [int(selected[0]), int(selected[-1])],
        "raw_indices_inclusive": [int(raw_rows[0]), int(raw_rows[-1])],
        "time_range_seconds": [float(times[0]), float(times[-1])],
        "alignment": alignment, "inference": inference,
        "coordinate_count": len(model_coordinates), "coordinate_order": model_coordinates,
        "upper_body_source": "robustly aligned source walking1_opt.mot rows",
        "coupled_coordinates": couplers,
        "inverse_dynamics": {
            "predicted_output": (output_raw / "walking1_torques_predicted_MJX.mot").resolve(),
            "ground_truth_output": (output_raw / "walking1_torques_ground_truth_MJX.mot").resolve(),
            "predicted_formula": "qfrc_inverse - tau_grf(predicted GRF/COP/free GRM)",
            "ground_truth_source": (processed_dir / "ID_GT_MJX.npy").resolve(),
            "coordinate_order": list(MJX_GENERALIZED_FORCE_COLUMNS),
            "units": {
                "pelvis_tx,pelvis_ty,pelvis_tz": "N",
                "all rotational coordinates": "N*m",
            },
            "diagnostics": id_diagnostics,
        },
        "max_abs_main_vs_raw_by_coordinate": dict(zip(model_coordinates, coordinate_delta)),
        "units": {"GRF": "N", "COP": "m, absolute OpenSim world frame", "GRM": "N*m", "rotations": "degrees", "translations": "m"},
        "contact_mask_applied": False,
    }
    (output_subject / "export_manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def write_readme(output_dir: Path, manifests: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"| {m['subject']} | {m['frame_count']} | {m['time_range_seconds'][0]:.2f}–{m['time_range_seconds'][1]:.2f} | "
        f"{m['alignment']['raw_frame_offset']} | {m['alignment']['best_robust_normalized_error']:.4f} |"
        for m in manifests
    )
    text = f"""# OpenSim validation inputs from LOSO predictions

This folder contains synchronized OpenSim inputs for subjects 2–4, trial 1. Subject 5 is intentionally excluded.

| Subject | Frames | Time (s) | raw offset | alignment error |
|---|---:|---:|---:|---:|
{rows}

## Files

Each `<subject>/trial_1/Video/Motion/Raw/` directory contains:

- `walking1_forces.mot`: fine-tuned LOSO GRF (N), absolute COP (m), and free GRM (N·m).
- `walking1_opt_Inputs_To_MJXModel.mot`: processed Video kinematics in the copied model's 35-coordinate order. Upper-body coordinates come from synchronized raw rows; coupled knee-beta coordinates are evaluated from the `.osim` constraints.
- `walking1_opt_original.mot`: the original raw `walking1_opt.mot` restricted to exactly the same held-out frames and timestamps.
- `walking1_torques_predicted_MJX.mot`: net MJX inverse-dynamics generalized forces calculated as `qfrc_inverse - tau_grf`, where `tau_grf` is computed from the fine-tuned LOSO GRF/COP/free-GRM predictions.
- `walking1_torques_ground_truth_MJX.mot`: ground-truth net MJX inverse dynamics from `Video/ProcessedData/ID_GT_MJX.npy`.

The torque files use the MJX model's 23 independent generalized coordinates in the same order as the processed inputs. `pelvis_tx`, `pelvis_ty`, and `pelvis_tz` are translational generalized forces in N; all other columns are rotational generalized moments in N·m. These are MJX/MuJoCo results, not OpenSim inverse-dynamics results.

`OpenSimScaled_Video.osim` is copied to each subject directory. `export_manifest.json` records sources, checks, indices, units, and coordinate-level comparisons.

## Reproduce

From the repository root:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib-cache /home/mobl/miniconda3/envs/myoconverter/bin/python scripts/export_loso_opensim_validation.py
```

## Synchronization and transforms

The exporter searches every possible integer offset between processed and raw kinematics using a robust, scale-normalized median error across 23 shared pelvis/lower-body/lumbar coordinates. It checks that the best offset is unambiguous and that timestamps are monotonic and uniform. Processing trim metadata is recorded and is also enforced when it explicitly uses the raw MOT as its reference space; the current metadata uses an already motion-aligned reference stream, so treating its start index as a raw-MOT offset would be unsafe. The official LOSO train-style evaluation mask (20-frame prediction margins here) is then applied once to force predictions, processed kinematics, and aligned raw rows. No interpolation or independent time trimming is used.

The model predicts GRF normalized by body weight and free GRM normalized by body weight × height; checkpoint normalizers restore N and N·m. Because `UseGRFNormCOP=false`, COP is restored as ground-aligned COP/height and then converted to meters by the inference code. For each foot, `[COP_x, -ankle_height, COP_z]` is rotated with the transpose of **Video** `WorldToGroundAlignedCalcnRotation.npy`, translated by the Video ankle world position, and converted from MJX Z-up to OpenSim Y-up coordinates `[x, z, -y]`. GRF and free moments receive the same axis conversion.

Contact probabilities and ground-truth contact are never used to zero, mask, or otherwise change force, COP, or moment predictions. Values in the force MOT are the unmasked fine-tuned predictions, including during swing.

## Compatibility and validation

The main kinematics file contains every coordinate found in the copied model, in model order, with translations in meters and rotations in degrees (`inDegrees=yes`). Coupled coordinates are computed from the model XML rather than copied from the source MOT. The exporter validates shapes, finite values, timestamp identity, coordinate coverage, alignment uniqueness, and required Video transform sources. The installed OpenSim Python package cannot load its shared library in this environment and `opensim-cmd` is absent, so model loading was validated structurally against the `.osim` XML but not by launching OpenSim itself.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--loso-root", type=Path, default=DEFAULT_LOSO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS), choices=SUBJECTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.loso_root = args.loso_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for subject in args.subjects:
        print(f"Exporting {subject}/{TRIAL}...", flush=True)
        manifests.append(export_subject(subject, args))
    write_readme(args.output_dir, manifests)
    summary = {"status": "ok", "subjects": [m["subject"] for m in manifests], "exports": manifests}
    (args.output_dir / "export_summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(manifests)} subjects to {args.output_dir}")


if __name__ == "__main__":
    main()
