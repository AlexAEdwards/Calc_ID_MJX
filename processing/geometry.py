"""Floor estimation, pelvis alignment and ground-aligned reference frames.

Extracted verbatim from ProcessData.py in REFACTOR_PLAN.md Stage 6. An AST closure
pass confirmed the cluster is dependency-closed once ``_normalize`` travels with
it - nothing outside geometry used that helper.

Everything here exists to answer one question: where is the ground, and how do
you express a quantity relative to it rather than to the lab?

* **Where the floor is.** ``_toe_trough_indices`` finds the local minima of toe
  height and ``estimate_floor_height_from_toe_troughs`` takes a robust statistic
  over them, on the assumption that the toe is momentarily on the floor at each
  trough. Both are sensitive to trial length, which is why regenerating a
  post-processed (already trimmed) trial does not reproduce its stored outputs:
  a different number of troughs shifts the floor estimate and every array derived
  from it. See the determinism section of REFACTOR_PLAN.md.

* **Ground-aligned frames.** ``_build_ground_aligned_rotation`` and
  ``_compose_world_to_ground_aligned`` construct a frame whose vertical axis is
  gravity and whose horizontal axes follow the segment's heading, so a quantity
  can be read as fore-aft / medio-lateral / vertical regardless of how the
  subject was oriented in the capture volume.
  ``_add_ankle_height_to_ground_aligned_y`` restores the ankle offset that the
  rotation alone discards, and ``_extract_ground_to_calc_rotations_from_qpos``
  recovers the same frames from a MuJoCo qpos trajectory via forward kinematics.

* **Derived measures.** ``_compute_foot_progression_angle_deg`` is the foot's
  heading relative to travel; ``_compute_knee_to_cop_vectors`` produces the lever
  arm that, with GRF, yields the knee adduction moment.

``align_myosuite_pelvis`` is the odd one out: it rewrites pelvis translation and
orientation into the MuJoCo model frame, which is why the stored pelvis channels
are not lab coordinates and will not match raw motion.

ProcessData.py re-exports every name, so its callers are unchanged.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R


def _toe_trough_indices(z: np.ndarray, min_distance_frames: int = 5) -> np.ndarray:
    """
    Detect trough indices in a 1-D toe-height trace.
    Uses find_peaks on the inverted signal, with a local-minima fallback.
    """
    z = np.asarray(z).reshape(-1)
    if z.size < 3:
        return np.array([], dtype=int)

    idx, _ = find_peaks(-z, distance=max(1, int(min_distance_frames)))
    if idx.size > 0:
        return idx.astype(int)

    # Fallback local-minima detector for very smooth / short traces.
    mins = np.where((z[1:-1] <= z[:-2]) & (z[1:-1] <= z[2:]))[0] + 1
    if mins.size > 0:
        return mins.astype(int)

    return np.array([], dtype=int)


def estimate_floor_height_from_toe_troughs(
    toes_z_r: np.ndarray,
    toes_z_l: np.ndarray,
    percentile: float = 10.0,
    offset_m: float = 0.015,
    min_troughs_for_direct_percentile: int = 5,
    interp_samples: int = 200,
) -> tuple[float, int]:
    """
    Estimate floor height from toe-Z troughs:
      1) detect troughs per foot separately,
      2) combine trough heights,
      3) use requested percentile of trough heights,
      4) add a small offset.

    If trough count is sparse, interpolate between detected trough heights
    and compute percentile on the interpolated series.
    Returns: (floor_height, trough_count)
    """
    z_r = np.asarray(toes_z_r, dtype=float).reshape(-1)
    z_l = np.asarray(toes_z_l, dtype=float).reshape(-1)

    idx_r = _toe_trough_indices(z_r)
    idx_l = _toe_trough_indices(z_l)
    vals_r = z_r[idx_r] if idx_r.size else np.array([], dtype=float)
    vals_l = z_l[idx_l] if idx_l.size else np.array([], dtype=float)

    trough_vals = np.concatenate([vals_r, vals_l])
    trough_count = int(trough_vals.size)

    if trough_count == 0:
        # Last-resort fallback: use the minimum toe height from each side.
        trough_vals = np.array([float(np.min(z_r)), float(np.min(z_l))], dtype=float)
        trough_count = int(trough_vals.size)

    if trough_count >= int(min_troughs_for_direct_percentile):
        base_height = float(np.percentile(trough_vals, percentile))
    else:
        # Sparse troughs: interpolate between combined trough heights first.
        comb_idx = np.concatenate([idx_r, idx_l]).astype(float)
        comb_val = np.concatenate([vals_r, vals_l]).astype(float)
        if comb_idx.size == 0:
            comb_idx = np.array([0.0, float(max(len(z_r), 1) - 1)], dtype=float)
            comb_val = np.array([trough_vals[0], trough_vals[-1]], dtype=float)
        elif comb_idx.size == 1:
            comb_idx = np.array([0.0, float(max(len(z_r), 1) - 1)], dtype=float)
            comb_val = np.array([comb_val[0], comb_val[0]], dtype=float)
        else:
            order = np.argsort(comb_idx)
            comb_idx = comb_idx[order]
            comb_val = comb_val[order]
            uniq_idx, uniq_pos = np.unique(comb_idx, return_index=True)
            comb_idx = uniq_idx
            comb_val = comb_val[uniq_pos]
            if comb_idx.size == 1:
                comb_idx = np.array([0.0, float(max(len(z_r), 1) - 1)], dtype=float)
                comb_val = np.array([comb_val[0], comb_val[0]], dtype=float)

        dense_x = np.linspace(comb_idx[0], comb_idx[-1], int(max(2, interp_samples)))
        dense_y = np.interp(dense_x, comb_idx, comb_val)
        base_height = float(np.percentile(dense_y, percentile))

    return base_height + float(offset_m), trough_count


def align_myosuite_pelvis(data: np.ndarray,
                           vel:   np.ndarray | None = None,
                           accel: np.ndarray | None = None,
                           GRF:   np.ndarray | None = None,
                           GRM:   np.ndarray | None = None,
                           COP:   np.ndarray | None = None):
    """
    Rotate all data so the median pelvis yaw is zero (forward = +X).

    data : (T, 6)  [tilt, list, rotation, tx, ty, tz]
    Returns rotated versions of all inputs (same shapes as inputs).
    """
    ROT_IDX = slice(0, 3)
    LIN_IDX = slice(3, 6)

    median_yaw  = np.median(data[:, 2])           # pelvis_rotation = yaw
    y_corr      = R.from_euler("Y", -median_yaw)

    # ── Linear positions ────────────────────────────────────────
    aligned_data = data.copy()
    aligned_data[:, LIN_IDX] = y_corr.apply(data[:, LIN_IDX])

    # ── Euler orientation ───────────────────────────────────────
    r_orig = R.from_euler("ZXY", data[:, ROT_IDX])
    r_new  = y_corr * r_orig
    aligned_data[:, ROT_IDX] = r_new.as_euler("ZXY")

    def _rot_6col(arr):
        """Rotate two concatenated 3-vectors [R(0:3), L(3:6)] per row."""
        out = arr.copy()
        out[:, 0:3] = y_corr.apply(arr[:, 0:3])
        out[:, 3:6] = y_corr.apply(arr[:, 3:6])
        return out

    # ── Velocity rotation ───────────────────────────────────────
    # For (T,6) force-style arrays: rotate both R and L 3-vectors via _rot_6col.
    # For (T,23) kinematics: only cols 3:6 (pelvis linear velocity) are 3D vectors;
    #   cols 0:3 (Euler angles) and cols 6+ (scalar joint velocities) must NOT be
    #   rotated through y_corr.apply() — they are scalars, not spatial vectors.
    # For (T,3): rotate directly.
    if vel is not None:
        al_vel = vel.copy()
        if vel.shape[1] == 6:
            al_vel = _rot_6col(vel)
        elif vel.shape[1] > 6:
            al_vel[:, 3:6] = y_corr.apply(vel[:, 3:6])
        else:  # (T, 3)
            al_vel = y_corr.apply(vel)
    else:
        al_vel = None

    # ── Acceleration rotation (same logic as velocity) ──────────
    if accel is not None:
        al_accel = accel.copy()
        if accel.shape[1] == 6:
            al_accel = _rot_6col(accel)
        elif accel.shape[1] > 6:
            al_accel[:, 3:6] = y_corr.apply(accel[:, 3:6])
        else:  # (T, 3)
            al_accel = y_corr.apply(accel)
    else:
        al_accel = None

    al_GRF = _rot_6col(GRF) if GRF is not None else None
    al_GRM = _rot_6col(GRM) if GRM is not None else None
    al_COP = _rot_6col(COP) if COP is not None else None

    return aligned_data, al_vel, al_accel, al_GRF, al_GRM, al_COP


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _apply_rotation_batch(rot_tij: np.ndarray, vec_tj: np.ndarray) -> np.ndarray:
    """Batch matrix-vector multiply: out[t] = rot[t] @ vec[t]."""
    return np.einsum("tij,tj->ti", rot_tij, vec_tj)


def _build_ground_aligned_rotation(
    R_wb: np.ndarray,
    n_w: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """
    Build the body->ground-aligned-body rotation for one frame.
    """
    n_w = _normalize(n_w)

    x_w = R_wb[:, 0]
    xg_w = x_w - np.dot(x_w, n_w) * n_w
    if np.linalg.norm(xg_w) < 1e-10:
        z_w = R_wb[:, 2]
        xg_w = np.cross(n_w, z_w)
    xg_w = _normalize(xg_w)

    yg_w = n_w.copy()
    zg_w = _normalize(np.cross(xg_w, yg_w))
    xg_w = _normalize(np.cross(yg_w, zg_w))

    R_wg = np.column_stack([xg_w, yg_w, zg_w])
    return R_wg.T @ R_wb


def _compose_world_to_ground_aligned(rot_w_to_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compose per-frame world->ground-aligned-calc rotation matrices.
    """
    T = rot_w_to_b.shape[0]
    rot_w_to_ga = np.zeros_like(rot_w_to_b)
    angle_deg = np.zeros(T, dtype=np.float64)

    for t in range(T):
        R_wb = rot_w_to_b[t].T
        R_ga_b = _build_ground_aligned_rotation(R_wb)
        rot_w_to_ga[t] = R_ga_b @ rot_w_to_b[t]
        cos_theta = (np.trace(R_ga_b) - 1.0) / 2.0
        angle_deg[t] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    return rot_w_to_ga, angle_deg


def _add_ankle_height_to_ground_aligned_y(
    cop_ground_aligned: np.ndarray,
    ankle_h: np.ndarray,
) -> np.ndarray:
    out = cop_ground_aligned.copy()
    out[:, 1] += ankle_h[:, 0]
    out[:, 4] += ankle_h[:, 1]
    return out


def _extract_ground_to_calc_rotations_from_qpos(
    model: mujoco.MjModel,
    qpos_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-simulate qpos frames and extract world->calc rotations and body positions.
    """
    calcn_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
    calcn_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_l")
    toes_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toes_r")
    toes_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toes_l")

    if min(calcn_r_id, calcn_l_id, toes_r_id, toes_l_id) < 0:
        raise ValueError("Could not find calcn_r, calcn_l, toes_r, and toes_l bodies in the model")

    qpos = np.asarray(qpos_matrix, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"pos_mjx must be 2D, got shape {qpos.shape}")
    if qpos.shape[1] != model.nq:
        raise ValueError(f"pos_mjx width {qpos.shape[1]} does not match model.nq {model.nq}")

    data = mujoco.MjData(model)
    T = qpos.shape[0]
    rot_g_to_r = np.zeros((T, 3, 3), dtype=np.float64)
    rot_g_to_l = np.zeros((T, 3, 3), dtype=np.float64)
    calcn_pos_r = np.zeros((T, 3), dtype=np.float64)
    calcn_pos_l = np.zeros((T, 3), dtype=np.float64)
    toes_pos_r = np.zeros((T, 3), dtype=np.float64)
    toes_pos_l = np.zeros((T, 3), dtype=np.float64)

    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        rot_g_to_r[t] = data.xmat[calcn_r_id].reshape(3, 3).T
        rot_g_to_l[t] = data.xmat[calcn_l_id].reshape(3, 3).T
        calcn_pos_r[t] = data.xpos[calcn_r_id]
        calcn_pos_l[t] = data.xpos[calcn_l_id]
        toes_pos_r[t] = data.xpos[toes_r_id]
        toes_pos_l[t] = data.xpos[toes_l_id]

    return rot_g_to_r, rot_g_to_l, calcn_pos_r, calcn_pos_l, toes_pos_r, toes_pos_l


def _compute_foot_progression_angle_deg(
    calcn_pos_r: np.ndarray,
    calcn_pos_l: np.ndarray,
    toes_pos_r: np.ndarray,
    toes_pos_l: np.ndarray,
) -> np.ndarray:
    """
    Compute per-frame FPA from the ground-plane toes-calcaneus vector.
    """
    v_r = toes_pos_r - calcn_pos_r
    v_l = toes_pos_l - calcn_pos_l
    fpa_r = np.degrees(np.arctan2(v_r[:, 1], v_r[:, 0]))
    fpa_l = np.degrees(np.arctan2(v_l[:, 1], v_l[:, 0]))
    return np.column_stack([fpa_r, fpa_l])


def _compute_knee_to_cop_vectors(
    cop_rel: np.ndarray,
    ankle_pos_r: np.ndarray,
    ankle_pos_l: np.ndarray,
    knee_pos_r: np.ndarray,
    knee_pos_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct world-frame COP and knee-to-COP vectors using the same
    lab-frame convention as the standalone forward-simulation script.
    """
    cop_rel = np.asarray(cop_rel, dtype=np.float64)
    ankle_pos_r = np.asarray(ankle_pos_r, dtype=np.float64)
    ankle_pos_l = np.asarray(ankle_pos_l, dtype=np.float64)
    knee_pos_r = np.asarray(knee_pos_r, dtype=np.float64)
    knee_pos_l = np.asarray(knee_pos_l, dtype=np.float64)

    if cop_rel.ndim != 2 or cop_rel.shape[1] < 4:
        raise ValueError(f"COP_Cleaned_Relative has invalid shape {cop_rel.shape}")
    if ankle_pos_r.shape != ankle_pos_l.shape or knee_pos_r.shape != knee_pos_l.shape:
        raise ValueError("Left and right knee/ankle arrays must have matching shapes")
    if ankle_pos_r.shape[0] != cop_rel.shape[0] or knee_pos_r.shape[0] != cop_rel.shape[0]:
        raise ValueError("Position and COP arrays must have the same length")

    T = int(cop_rel.shape[0])
    cop_world_r = np.zeros((T, 3), dtype=np.float64)
    cop_world_l = np.zeros((T, 3), dtype=np.float64)

    cop_world_r[:, :2] = ankle_pos_r[:, :2] + cop_rel[:, :2]
    # COP lies on the MuJoCo ground plane (world Z=0).  The legacy implementation
    # used ankle_pos[:, 2], which placed COP at ankle height and removed the vertical
    # component of the knee->COP moment arm.
    cop_world_r[:, 2] = 0.0
    cop_world_l[:, :2] = ankle_pos_l[:, :2] + cop_rel[:, 2:4]
    cop_world_l[:, 2] = 0.0

    vec_knee_to_cop_r = cop_world_r - knee_pos_r
    vec_knee_to_cop_l = cop_world_l - knee_pos_l
    combined = np.column_stack([vec_knee_to_cop_r, vec_knee_to_cop_l])
    return combined, cop_world_r, cop_world_l
