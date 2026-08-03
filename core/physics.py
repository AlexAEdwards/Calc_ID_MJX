"""External-moment and tau = J^T . F reconstruction.

Extracted verbatim from train.py in REFACTOR_PLAN.md Stage 5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_full_external_moments(
    cop_pred_unnorm: jnp.ndarray,  # (batch, seq, 4) [rx, rz, lx, lz] in ground-aligned calc frame
    grf_pred_unnorm: jnp.ndarray,  # (batch, seq, 6)
    free_moments_pred_unnorm: jnp.ndarray,  # (batch, seq, 2) [rz, lz]
    ankle_heights: jnp.ndarray,  # (batch, seq, 2) [right, left]
    rot_w_to_ga: jnp.ndarray,  # (batch, seq, 2, 3, 3) world->ground-aligned calc rotation
) -> jnp.ndarray:
    """
    Compute full external moment about each foot origin using COP, GRF, and free moments.
    """
    # Predicted COP channels are [X, Z] in the ground-aligned calc frame.
    # Build 3D ground-aligned vectors by inserting ankle height as Y.
    cop_r_ga = jnp.concatenate(
        [cop_pred_unnorm[..., 0:1], -ankle_heights[..., 0:1], cop_pred_unnorm[..., 1:2]],
        axis=-1
    )
    cop_l_ga = jnp.concatenate(
        [cop_pred_unnorm[..., 2:3], -ankle_heights[..., 1:2], cop_pred_unnorm[..., 3:4]],
        axis=-1
    )

    # Rotate ground-aligned COP vectors back to world:
    # R_ga->w = (R_w->ga)^T
    rot_w_to_ga_r = rot_w_to_ga[:, :, 0]  # (batch, seq, 3, 3)
    rot_w_to_ga_l = rot_w_to_ga[:, :, 1]
    rot_ga_to_w_r = jnp.swapaxes(rot_w_to_ga_r, -1, -2)
    rot_ga_to_w_l = jnp.swapaxes(rot_w_to_ga_l, -1, -2)
    cop_r = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_r, cop_r_ga)
    cop_l = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_l, cop_l_ga)

    grf_r = grf_pred_unnorm[..., :3]
    grf_l = grf_pred_unnorm[..., 3:6]
    
    # Reconstruct 3D moments from 1D predictions (assume Mx=My=0)
    mz_r = free_moments_pred_unnorm[..., 0:1]
    mz_l = free_moments_pred_unnorm[..., 1:2]
    
    mom_r = jnp.concatenate([jnp.zeros_like(mz_r), jnp.zeros_like(mz_r), mz_r], axis=-1)
    mom_l = jnp.concatenate([jnp.zeros_like(mz_l), jnp.zeros_like(mz_l), mz_l], axis=-1)

    # Cross product: r x F
    # r is the vector from the point of force application (COP) to the moment reference point.
    # Usually M_total = M_free + (r x F)
    # Here r is COP relative to ankle, expressed in world coordinates.
    
    m_r_induced = jnp.cross(cop_r, grf_r)
    m_l_induced = jnp.cross(cop_l, grf_l)
    
    m_r_total = m_r_induced + mom_r
    m_l_total = m_l_induced + mom_l
    
    return jnp.concatenate([m_r_total, m_l_total], axis=-1)


def compute_tau_grf_from_predictions(
    grf_pred: jnp.ndarray,  # (batch, seq, 6) [right_xyz, left_xyz]
    moments_pred: jnp.ndarray,  # (batch, seq, 6) - full moments computed from COP/GRF/free moment
    jacp: jnp.ndarray,  # (batch, seq, 2, 3, 39)
    jacr: jnp.ndarray,  # (batch, seq, 2, 3, 39)
) -> jnp.ndarray:
    """
    Compute τ_grf = Jp^T @ GRF + Jr^T @ M for each timestep.
    
    Args:
        grf_pred: Predicted GRF [right_xyz, left_xyz]
        moments_pred: Full external moments [right_xyz, left_xyz]
        jacp: Position Jacobian (batch, seq, 2 bodies, 3 spatial, 39 dofs)
        jacr: Rotation Jacobian
    
    Returns:
        tau_grf: Joint torques from GRF (batch, seq, 39)
    """
    # Split into right/left
    grf_r = grf_pred[..., :3]  # (batch, seq, 3)
    grf_l = grf_pred[..., 3:]  # (batch, seq, 3)
    moment_r = moments_pred[..., :3]  # (batch, seq, 3)
    moment_l = moments_pred[..., 3:]  # (batch, seq, 3)
    
    # Jp^T @ F: need to einsum over spatial dimension
    # jacp shape: (batch, seq, 2, 3, 39)
    # force shape: (batch, seq, 3)
    
    # For right foot (body 0? Check BatchDataProcessing): 
    # In BatchDataProcessing: External_Force = External_Force.at[calcn_l_id, ...].set(...)
    # jacobian_data['jacp'] has shape (T, 2, 3, nv). 
    # Usually index 0 is right, index 1 is left based on how it was saved?
    # Let's check BatchDataProcessing.py saving order.
    # "jacobian_data['jacp'] = np.stack([jacp_r, jacp_l], axis=1)" -> So 0 is Right, 1 is Left.
    
    tau_p_r = jnp.einsum('bsij,bsi->bsj', jacp[:, :, 0], grf_r)  # (batch, seq, 39)
    tau_p_l = jnp.einsum('bsij,bsi->bsj', jacp[:, :, 1], grf_l)  # (batch, seq, 39)
    
    # Jr^T @ M
    tau_r_r = jnp.einsum('bsij,bsi->bsj', jacr[:, :, 0], moment_r)  # (batch, seq, 39)
    tau_r_l = jnp.einsum('bsij,bsi->bsj', jacr[:, :, 1], moment_l)  # (batch, seq, 39)
    
    tau_grf = tau_p_r + tau_p_l + tau_r_r + tau_r_l
    
    return tau_grf


def decode_cop_signal_to_length(
    cop_signal: Any,
    grf_ratio: Any,
    height_m: Any,
    *,
    use_grf_norm_cop: bool = False,
    contact_probability: Any = None,
    contact_threshold: float = 0.5,
    xp=jnp,
    eps: float = 1e-6,
) -> Any:
    """
    Convert the model COP signal to length units.

    Default COP signal is COP/height. With UseGRFNormCOP it is
    (COP/height) * (|GRF|/BW), so decode by dividing by each foot's
    bodyweight-normalized GRF magnitude before multiplying by height.
    """
    cop_arr = xp.asarray(cop_signal)
    h = xp.asarray(height_m, dtype=cop_arr.dtype)
    if not use_grf_norm_cop:
        return cop_arr * h

    grf_arr = xp.asarray(grf_ratio, dtype=cop_arr.dtype)
    eps_arr = xp.asarray(eps, dtype=cop_arr.dtype)
    mag_r_sq = xp.sum(xp.square(grf_arr[..., 0:3]), axis=-1, keepdims=True)
    mag_l_sq = xp.sum(xp.square(grf_arr[..., 3:6]), axis=-1, keepdims=True)
    mag_r = xp.sqrt(xp.maximum(mag_r_sq, eps_arr * eps_arr))
    mag_l = xp.sqrt(xp.maximum(mag_l_sq, eps_arr * eps_arr))
    decoded = xp.concatenate([
        cop_arr[..., 0:2] * h / mag_r,
        cop_arr[..., 2:4] * h / mag_l,
    ], axis=-1)
    if contact_probability is None:
        return decoded

    contact = xp.asarray(contact_probability, dtype=cop_arr.dtype)
    threshold = xp.asarray(contact_threshold, dtype=cop_arr.dtype)
    mask_r = (contact[..., 0:1] >= threshold).astype(cop_arr.dtype)
    mask_l = (contact[..., 1:2] >= threshold).astype(cop_arr.dtype)
    return xp.concatenate([
        decoded[..., 0:2] * mask_r,
        decoded[..., 2:4] * mask_l,
    ], axis=-1)
