#!/usr/bin/env python3
"""Recalculate OpenSim ID from MoCap kinematics and ProcessedData forces.

For each OpenCap trial this script builds an OpenSim InverseDynamicsTool setup
with:

* kinematics from ``Trial/MoCap/pos_mjx.npy``
* GRF/COP/free moments from ``Trial/ProcessedData``

It writes:

* ``Trial/OpenSimResults_recalculated/inverse_dynamics_recalculated.sto``
* ``Trial/MoCap/OpenSim_ID_recalculated.npy`` in MJX qpos/torque order

Run in an environment with the OpenSim Python API available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from batch_opensim_inverse_dynamics import (  # noqa: E402
    _OPENSIM_TO_MJX_IDX,
    read_storage_file,
    run_inverse_dynamics,
    write_id_html_report,
)
from generate_opensim_id_inputs import (  # noqa: E402
    DEFAULT_LEFT_BODY,
    DEFAULT_MODEL_NAME,
    DEFAULT_RIGHT_BODY,
    FORCE_COLUMNS,
    TrialPaths,
    create_patella_free_model,
    discover_trials,
    load_array,
    make_coordinates_matrix,
    make_force_matrix,
    mjx_to_opensim_coords,
    processed_time_vector,
    qpos_to_npy_coordinates,
    write_external_loads_xml,
    write_id_setup_xml,
    write_storage,
)

OUTPUT_DIR_NAME = "OpenSimResults_recalculated"
OUTPUT_STO_NAME = "inverse_dynamics_recalculated.sto"
OUTPUT_NPY_NAME = "OpenSim_ID_recalculated.npy"

# Canonical raw timebase: ProcessData decimates everything to 100 Hz, and Motion/Time.npy
# carries that raw axis. We anchor all streams to it and only trust it when its median dt
# is physically ~100 Hz (otherwise fall back to a uniform 100 Hz ramp).
RAW_TIMEBASE_HZ = 100.0
_RAW_DT_S = 1.0 / RAW_TIMEBASE_HZ
_RAW_DT_TOL = 0.2  # accept a raw Time vector whose median dt is within +/-20% of 100 Hz
_ALIGN_LEN_TOL_FRAMES = 2  # warn if a stream's length differs from the raw axis by more

# OpenSim pelvis (floating-base) residual columns -> MJX channels. The coordinates.mot fed
# to OpenSim assigns MJX qpos channels to these pelvis coordinates (via NP_TO_QPOS), so each
# coordinate's generalized force maps back to that same MJX channel. Translations are written
# as *_force columns in the .sto, rotations as *_moment. These residuals are filled only for
# structural parity with ID_GT_MJX; they are not used in training/accuracy.
PELVIS_OPENSIM_TO_MJX: dict[str, tuple[int, str]] = {
    "pelvis_tx": (0, "force"),
    "pelvis_ty": (1, "force"),
    "pelvis_tz": (2, "force"),
    "pelvis_tilt": (3, "moment"),
    "pelvis_list": (4, "moment"),
    "pelvis_rotation": (5, "moment"),
}


def _build_mjx_channel_names() -> list[str]:
    """Human-readable name for each of the 31 MJX qpos/torque channels (for warnings)."""
    names = [f"mjx_ch{i}" for i in range(31)]
    for coord, mjx_idx in _OPENSIM_TO_MJX_IDX.items():
        names[mjx_idx] = coord
    for coord, (mjx_idx, _kind) in PELVIS_OPENSIM_TO_MJX.items():
        names[mjx_idx] = coord
    # Knee coupling/secondary DOFs that exist in the MJX model but have no OpenSim ID
    # moment column (dependent coordinates) -> necessarily zero in the recalculated GT.
    for ch in (9, 10, 12, 13):
        names[ch] = f"knee_coupling_r[{ch}]"
    for ch in (20, 21, 23, 24):
        names[ch] = f"knee_coupling_l[{ch}]"
    return names


MJX_CHANNEL_NAMES = _build_mjx_channel_names()


def _raw_timebase(trial_dir: Path, fallback_len: int, warnings: list[str]) -> np.ndarray:
    """Return the raw 100 Hz time axis from ``Motion/Time.npy`` (uniform-ramp fallback)."""
    motion_time = trial_dir / "Motion" / "Time.npy"
    if motion_time.exists():
        try:
            t = np.asarray(np.load(motion_time), dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not read Motion/Time.npy ({exc}); using uniform 100 Hz ramp.")
            t = None
        if t is not None and len(t) >= 2:
            dt = float(np.median(np.diff(t)))
            if abs(dt - _RAW_DT_S) <= _RAW_DT_TOL * _RAW_DT_S:
                return t
            warnings.append(
                f"Motion/Time.npy median dt={dt:.5f}s is not ~{RAW_TIMEBASE_HZ:.0f} Hz; "
                "using uniform 100 Hz ramp instead."
            )
    return np.arange(int(fallback_len), dtype=np.float64) * _RAW_DT_S


def _load_hybrid_source(paths: TrialPaths, warnings: list[str]) -> dict[str, Any]:
    """Load filtered kinematics + cleaned forces, all co-registered on the raw 100 Hz axis.

    Both the filtered kinematics and the cleaned COP/GRF/moment are read from ``MoCap/``,
    which the pipeline writes on the raw ``Motion/Time`` 100 Hz timebase. (Previously the
    kinematics came from ``MoCap/`` while the forces came from ``ProcessedData/``, whose
    per-stream trims differ -- e.g. kinematics offset by +1 frame vs forces -- which
    misaligned them. Sourcing both from ``MoCap/`` guarantees one temporal frame.) All
    streams are then trimmed to their common overlap on that axis.
    """
    mocap = paths.trial_dir / "MoCap"
    qpos_path = mocap / "pos_mjx.npy"
    grf_path = mocap / "GRF_Cleaned.npy"
    moment_path = mocap / "Moment_Cleaned.npy"
    ankle_r_path = mocap / "ankle_pos_r.npy"
    ankle_l_path = mocap / "ankle_pos_l.npy"
    cop_back_to_world_path = mocap / "COP_CalcFrame_GroundAligned_BackToWorld.npy"
    cop_rel_path = mocap / "COP_Cleaned_Relative.npy"

    qpos = load_array(qpos_path, 2)
    grf_mjx = load_array(grf_path, 2)
    moment_mjx = load_array(moment_path, 2)
    ankle_r = load_array(ankle_r_path, 2)
    ankle_l = load_array(ankle_l_path, 2)

    raw_time = _raw_timebase(paths.trial_dir, len(qpos), warnings)

    # Align every stream to the raw 100 Hz axis and trim to the common overlap. MoCap/
    # streams share t0 with Motion/Time, so the only correction needed is trimming trailing
    # frames; we warn if any stream is more than a couple frames off the raw axis.
    stream_lengths = {
        "raw_time": len(raw_time),
        "pos_mjx": len(qpos),
        "GRF_Cleaned": len(grf_mjx),
        "Moment_Cleaned": len(moment_mjx),
        "ankle_pos_r": len(ankle_r),
        "ankle_pos_l": len(ankle_l),
    }
    raw_len = len(raw_time)
    for name, length in stream_lengths.items():
        if name != "raw_time" and abs(length - raw_len) > _ALIGN_LEN_TOL_FRAMES:
            warnings.append(
                f"stream '{name}' length {length} differs from raw 100 Hz axis "
                f"({raw_len}) by >{_ALIGN_LEN_TOL_FRAMES} frames; trimmed to overlap."
            )

    n_frames = min(stream_lengths.values())
    qpos = qpos[:n_frames]
    grf_mjx = grf_mjx[:n_frames]
    moment_mjx = moment_mjx[:n_frames]
    ankle_r = ankle_r[:n_frames]
    ankle_l = ankle_l[:n_frames]
    time_vec = raw_time[:n_frames]

    if cop_back_to_world_path.exists():
        cop_back_to_world = load_array(cop_back_to_world_path, 2)[:n_frames]
        if cop_back_to_world.shape[1] != 6:
            raise ValueError(f"{cop_back_to_world_path} has shape {cop_back_to_world.shape}, expected (T, 6)")
        cop_mjx = np.zeros((n_frames, 6), dtype=np.float64)
        cop_mjx[:, 0:3] = ankle_r + cop_back_to_world[:, 0:3]
        cop_mjx[:, 3:6] = ankle_l + cop_back_to_world[:, 3:6]
        cop_source = str(cop_back_to_world_path)
    else:
        cop_rel = load_array(cop_rel_path, 2)[:n_frames]
        if cop_rel.shape[1] != 4:
            raise ValueError(f"{cop_rel_path} has shape {cop_rel.shape}, expected (T, 4)")
        cop_mjx = np.zeros((n_frames, 6), dtype=np.float64)
        cop_mjx[:, 0:3] = ankle_r
        cop_mjx[:, 3:6] = ankle_l
        cop_mjx[:, 0:2] += cop_rel[:, 0:2]
        cop_mjx[:, 3:5] += cop_rel[:, 2:4]
        cop_source = str(cop_rel_path)

    pos = qpos_to_npy_coordinates(qpos)
    grf = np.hstack([mjx_to_opensim_coords(grf_mjx[:, 0:3]), mjx_to_opensim_coords(grf_mjx[:, 3:6])])
    grm = np.hstack([mjx_to_opensim_coords(moment_mjx[:, 0:3]), mjx_to_opensim_coords(moment_mjx[:, 3:6])])
    cop = np.hstack([mjx_to_opensim_coords(cop_mjx[:, 0:3]), mjx_to_opensim_coords(cop_mjx[:, 3:6])])

    return {
        "pos": pos,
        "grf": grf,
        "cop": cop,
        "grm": grm,
        "kin_time": time_vec,
        "force_time": time_vec,
        "n_frames": int(n_frames),
        "stream_lengths": stream_lengths,
        "source_files": {
            "mocap_pos_mjx": str(qpos_path),
            "mocap_grf_cleaned": str(grf_path),
            "mocap_cop": cop_source,
            "mocap_moment_cleaned": str(moment_path),
            "raw_timebase": str(paths.trial_dir / "Motion" / "Time.npy"),
        },
    }


def _map_sto_to_mjx_npy(sto_path: Path, n_frames: int) -> tuple[np.ndarray, list[int]]:
    """Map the recalculated ID ``.sto`` onto the 31-channel MJX layout (== ID_GT_MJX).

    Returns the ``(n_frames, 31)`` float32 array (with NaN still present in any channel the
    .sto cannot supply) and the sorted list of MJX channel indices that contain NaN.
    """
    columns, rows = read_storage_file(sto_path)
    data = np.asarray(rows, dtype=np.float64)
    col_idx = {name: idx for idx, name in enumerate(columns)}
    out = np.full((data.shape[0], 31), np.nan, dtype=np.float32)

    # Leg + lumbar joint moments.
    for coord, mjx_idx in _OPENSIM_TO_MJX_IDX.items():
        col = f"{coord}_moment"
        if col in col_idx:
            out[:, mjx_idx] = data[:, col_idx[col]].astype(np.float32)

    # Pelvis/floating-base residuals -> channels 0-5 (structural parity with ID_GT_MJX).
    for coord, (mjx_idx, kind) in PELVIS_OPENSIM_TO_MJX.items():
        col = f"{coord}_{kind}"
        if col in col_idx:
            out[:, mjx_idx] = data[:, col_idx[col]].astype(np.float32)

    if out.shape[0] != int(n_frames):
        source_t = np.linspace(0.0, 1.0, out.shape[0], dtype=np.float64)
        target_t = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)
        resized = np.empty((int(n_frames), out.shape[1]), dtype=np.float32)
        for col in range(out.shape[1]):
            resized[:, col] = np.interp(target_t, source_t, out[:, col]).astype(np.float32)
        out = resized

    nan_channels = [ch for ch in range(out.shape[1]) if bool(np.isnan(out[:, ch]).any())]
    return out.astype(np.float32), nan_channels


def process_trial(
    paths: TrialPaths,
    *,
    overwrite: bool,
    dry_run: bool,
    right_body: str,
    left_body: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    out_dir = paths.output_dir
    mocap_dir = paths.trial_dir / "MoCap"
    sto_path = out_dir / OUTPUT_STO_NAME
    npy_path = mocap_dir / OUTPUT_NPY_NAME
    if sto_path.exists() and npy_path.exists() and not overwrite:
        return {
            "subject": paths.subject_dir.name,
            "trial": paths.trial_dir.name,
            "status": "skipped",
            "reason": "recalculated outputs already exist",
            "sto_path": str(sto_path),
            "npy_path": str(npy_path),
        }

    if not paths.model_path.exists():
        raise FileNotFoundError(f"Missing subject model: {paths.model_path}")
    warnings: list[str] = []
    loaded = _load_hybrid_source(paths, warnings)
    coord_names, coord_values = make_coordinates_matrix(loaded["pos"], paths.model_path, warnings)
    coord_data = np.column_stack([loaded["kin_time"], coord_values])
    force_data = make_force_matrix(loaded["force_time"], loaded["grf"], loaded["cop"], loaded["grm"])

    manifest = {
        "subject": paths.subject_dir.name,
        "trial": paths.trial_dir.name,
        "trial_dir": str(paths.trial_dir),
        "model": str(paths.model_path),
        "source": "mocap_kinematics_processed_forces",
        "source_files": loaded["source_files"],
        "shapes": {
            "pos": list(loaded["pos"].shape),
            "grf": list(loaded["grf"].shape),
            "cop": list(loaded["cop"].shape),
            "grm": list(loaded["grm"].shape),
        },
        "warnings": warnings,
        "sto_path": str(sto_path),
        "npy_path": str(npy_path),
    }
    if dry_run:
        manifest["status"] = "dry_run"
        return manifest

    out_dir.mkdir(parents=True, exist_ok=True)
    mocap_dir.mkdir(parents=True, exist_ok=True)
    id_model_path = create_patella_free_model(paths.model_path, out_dir)
    coordinates_file = out_dir / "coordinates.mot"
    forces_file = out_dir / "ground_reaction.mot"
    external_loads_file = out_dir / "external_loads.xml"
    setup_file = out_dir / "id_setup.xml"

    write_storage(coordinates_file, "Coordinates", ["time", *coord_names], coord_data, in_degrees=True)
    write_storage(forces_file, "GroundReaction", FORCE_COLUMNS, force_data)
    write_external_loads_xml(external_loads_file, forces_file, coordinates_file, right_body, left_body)
    write_id_setup_xml(
        setup_file,
        id_model_path,
        coordinates_file,
        external_loads_file,
        out_dir,
        OUTPUT_STO_NAME,
        (float(loaded["kin_time"][0]), float(loaded["kin_time"][-1])),
    )
    (out_dir / "input_generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run_inverse_dynamics(setup_file)
    opensim_id, nan_channels = _map_sto_to_mjx_npy(sto_path, n_frames=int(loaded["pos"].shape[0]))
    # Assign every NaN to zero so the GT is fully finite (the knee-coupling channels have no
    # OpenSim ID column; any other NaN would otherwise poison downstream torque losses).
    if nan_channels:
        opensim_id = np.nan_to_num(opensim_id, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.save(npy_path, opensim_id)
    html_path = write_id_html_report(sto_path)
    manifest.update(
        {
            "status": "ok",
            "elapsed_sec": float(time.perf_counter() - start),
            "html_report": str(html_path),
            "n_frames": int(loaded.get("n_frames", loaded["pos"].shape[0])),
            "stream_lengths": loaded.get("stream_lengths", {}),
            "nan_channels": nan_channels,
            "nan_channel_names": [MJX_CHANNEL_NAMES[ch] for ch in nan_channels],
        }
    )
    (out_dir / "recalculated_opensim_id_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "OpenCapSubjects_Filt")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--trial", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--right-body", default=DEFAULT_RIGHT_BODY)
    parser.add_argument("--left-body", default=DEFAULT_LEFT_BODY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-name", default="opensim_id_recalculated_manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    trials = discover_trials(
        dataset_root,
        subject=args.subject,
        trial=args.trial,
        model_name=args.model_name,
        output_dir_name=OUTPUT_DIR_NAME,
    )
    if args.limit is not None:
        trials = trials[: int(args.limit)]
    manifest: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "source": "mocap_kinematics_processed_forces",
        "output_sto_name": OUTPUT_STO_NAME,
        "output_npy_name": OUTPUT_NPY_NAME,
        "trials_seen": len(trials),
        "trials_ok": 0,
        "trials_skipped": 0,
        "trials_failed": 0,
        "results": [],
        "failures": [],
    }
    for paths in trials:
        try:
            result = process_trial(
                paths,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
                right_body=str(args.right_body),
                left_body=str(args.left_body),
            )
            manifest["results"].append(result)
            if result.get("status") == "skipped":
                manifest["trials_skipped"] += 1
            elif result.get("status") in {"ok", "dry_run"}:
                manifest["trials_ok"] += 1
        except Exception as exc:
            manifest["trials_failed"] += 1
            manifest["failures"].append(
                {
                    "subject": paths.subject_dir.name,
                    "trial": paths.trial_dir.name,
                    "trial_dir": str(paths.trial_dir),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
        done = manifest["trials_ok"] + manifest["trials_skipped"] + manifest["trials_failed"]
        print(f"processed {done}/{manifest['trials_seen']}", flush=True)
    if not args.dry_run:
        manifest_path = dataset_root / str(args.manifest_name)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["manifest_path"] = str(manifest_path)
    print(json.dumps(manifest, indent=2))

    # End-of-run NaN warning: which MJX channels contained NaN (now zeroed) and in how many
    # trials. Channels with no OpenSim ID column (knee-coupling DOFs) are expected here.
    nan_trial_counts: dict[int, int] = {}
    for result in manifest["results"]:
        for ch in result.get("nan_channels", []):
            nan_trial_counts[int(ch)] = nan_trial_counts.get(int(ch), 0) + 1
    n_processed = manifest["trials_ok"]
    if nan_trial_counts:
        print(
            "\n[WARNING] NaNs were found and set to 0 in the recalculated OpenSim ID. "
            f"Affected MJX channels ({n_processed} trial(s) processed):",
            flush=True,
        )
        for ch in sorted(nan_trial_counts):
            print(
                f"  ch{ch:2d}  {MJX_CHANNEL_NAMES[ch]:<22s} NaN in {nan_trial_counts[ch]} trial(s)",
                flush=True,
            )
    elif n_processed:
        print("\n[OK] No NaNs found in any recalculated OpenSim ID column.", flush=True)
    return 1 if manifest["trials_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
