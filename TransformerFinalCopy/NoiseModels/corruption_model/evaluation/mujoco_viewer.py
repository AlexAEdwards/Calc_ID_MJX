from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import numpy as np

from corruption_model.io.load_paired import CANONICAL_DOF_NAMES


MOTION_TO_QPOS_INDEX = {
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
VIEWER_MODEL_FILENAMES = ("MyosuiteModel_FIXED.xml", "MyosuiteModel.xml")


def resolve_subject_mujoco_model_path(trial_dir: Path) -> Path:
    subject_dir = trial_dir.parent
    for filename in VIEWER_MODEL_FILENAMES:
        candidate = subject_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {' or '.join(VIEWER_MODEL_FILENAMES)} under {subject_dir}")


def map_motion_pos_to_qpos(motion_pos: np.ndarray, *, qpos_size: int) -> np.ndarray:
    motion_np = np.asarray(motion_pos, dtype=np.float64)
    expected_dof = len(CANONICAL_DOF_NAMES)
    if motion_np.ndim != 2 or motion_np.shape[1] != expected_dof:
        raise ValueError(f"Expected motion_pos shape (T, {expected_dof}), got {motion_np.shape}")
    qpos = np.zeros((motion_np.shape[0], int(qpos_size)), dtype=np.float64)
    for motion_idx, qpos_idx in MOTION_TO_QPOS_INDEX.items():
        if qpos_idx < qpos.shape[1]:
            qpos[:, qpos_idx] = motion_np[:, motion_idx]
    return qpos


def apply_xml_joint_couplings(qpos_matrix: np.ndarray, *, model_path: Path, mj_model: object) -> np.ndarray:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MuJoCo is required to display the dry-run viewer.") from exc

    qpos_out = np.asarray(qpos_matrix, dtype=np.float64).copy()
    try:
        root = ET.parse(str(model_path)).getroot()
    except Exception:
        return qpos_out

    for equality in root.iter("equality"):
        for joint_eq in equality.iter("joint"):
            slave_name = joint_eq.get("joint1")
            master_name = joint_eq.get("joint2")
            if slave_name is None or master_name is None:
                continue
            coeffs = [float(value) for value in joint_eq.get("polycoef", "0 1 0 0 0").split()]
            coeffs.extend([0.0] * max(0, 5 - len(coeffs)))

            slave_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, slave_name)
            master_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, master_name)
            if slave_id < 0 or master_id < 0:
                continue

            slave_qpos_idx = int(mj_model.jnt_qposadr[slave_id])
            master_qpos_idx = int(mj_model.jnt_qposadr[master_id])
            if slave_qpos_idx >= qpos_out.shape[1] or master_qpos_idx >= qpos_out.shape[1]:
                continue

            theta = qpos_out[:, master_qpos_idx]
            qpos_out[:, slave_qpos_idx] = (
                coeffs[0]
                + coeffs[1] * theta
                + coeffs[2] * theta**2
                + coeffs[3] * theta**3
                + coeffs[4] * theta**4
            )
    return qpos_out


def build_viewer_qpos_from_motion_pos(motion_pos: np.ndarray, *, model_path: Path) -> np.ndarray:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MuJoCo is required to display the dry-run viewer.") from exc

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos_matrix = map_motion_pos_to_qpos(motion_pos, qpos_size=int(mj_model.nq))
    return apply_xml_joint_couplings(qpos_matrix, model_path=model_path, mj_model=mj_model)


def estimate_playback_dt(time_vec: np.ndarray) -> float:
    arr = np.asarray(time_vec, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return 0.01
    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.01
    return max(float(np.median(diffs)), 1e-4)


def show_dry_run_mujoco_viewer(
    *,
    trial_dir: Path,
    time_vec: np.ndarray,
    motion_pos: np.ndarray,
    source_name: str,
) -> Dict[str, str | None]:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:  # pragma: no cover
        message = f"MuJoCo viewer unavailable: {exc}"
        print(f"  Skipping MuJoCo viewer for {trial_dir} [{source_name}]: {message}", flush=True)
        return {"status": "skipped_unavailable", "model_path": None, "error": message}

    model_path = resolve_subject_mujoco_model_path(trial_dir)
    qpos_matrix = build_viewer_qpos_from_motion_pos(motion_pos, model_path=model_path)

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_data = mujoco.MjData(mj_model)
    if qpos_matrix.shape[1] != mj_model.nq:
        raise ValueError(f"Viewer qpos width {qpos_matrix.shape[1]} does not match model.nq {mj_model.nq}")
    if qpos_matrix.shape[0] == 0:
        raise ValueError(f"Viewer qpos is empty for {trial_dir}")
    if not np.isfinite(qpos_matrix).all():
        raise ValueError(f"Viewer qpos contains non-finite values for {trial_dir}")

    playback_dt = estimate_playback_dt(time_vec)
    print(
        f"  Launching MuJoCo viewer for {trial_dir.parent.name}/{trial_dir.name} "
        f"[{source_name}] with model {model_path.name}",
        flush=True,
    )
    print("  (Close the viewer window to continue.)", flush=True)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if len(viewer.opt.geomgroup) > 2:
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[2] = 0

        frame_idx = 0
        num_frames = int(qpos_matrix.shape[0])
        while viewer.is_running():
            mj_data.qpos[:] = qpos_matrix[frame_idx]
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(playback_dt)
            frame_idx = (frame_idx + 1) % num_frames

    return {
        "status": "displayed",
        "model_path": str(model_path),
        "error": None,
        "source_name": source_name,
    }
