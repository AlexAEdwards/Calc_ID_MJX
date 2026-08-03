"""Training losses and the prediction-splitting helpers they need.

Extracted verbatim from train.py in REFACTOR_PLAN.md Stage 5. compute_total_loss
is the training objective; the rest of this module is the dependency cluster it
pulls in, moved together so the loss has no back-reference into train.py.

train.py re-exports every name here, so existing imports and its own internal
uses (make_train_step, make_eval_step, plot_predictions, main) are unaffected.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from core.constants import (
    CONTACT_SLICE, COP_SLICE, GRF_SLICE, MOMENTS_SLICE,
    ROTATION_OUTPUT_DIM, STANDARD_OUTPUT_DIM,
)
from core.physics import (
    compute_full_external_moments,
    compute_tau_grf_from_predictions,
    decode_cop_signal_to_length,
)


def _safe_vector_norm(
    vec: Any,
    *,
    xp=jnp,
    axis: int = -1,
    keepdims: bool = False,
    eps: float = 1e-6,
) -> Any:
    """Stable vector norm that never returns zero."""
    squared = xp.sum(xp.square(vec), axis=axis, keepdims=keepdims)
    return xp.sqrt(xp.maximum(squared, eps))


def _normalize_vector(
    vec: Any,
    *,
    xp=jnp,
    axis: int = -1,
    eps: float = 1e-6,
) -> Any:
    """Normalize a vector with epsilon protection."""
    return vec / _safe_vector_norm(vec, xp=xp, axis=axis, keepdims=True, eps=eps)


def project_rotation_matrices(
    rot: Any,
    xp=jnp,
) -> Any:
    """Project arbitrary 3x3 matrices to proper rotations with stable Gram-Schmidt."""
    leading_shape = tuple(rot.shape[:-2])
    rot_flat = xp.reshape(rot, (-1, 3, 3)).astype(rot.dtype)
    rot_flat = xp.where(xp.isfinite(rot_flat), rot_flat, xp.zeros_like(rot_flat))

    col1 = rot_flat[..., :, 0]
    col2 = rot_flat[..., :, 1]
    col3 = rot_flat[..., :, 2]

    basis1 = _normalize_vector(col1, xp=xp, eps=1e-6)

    cand2 = col2 - xp.sum(basis1 * col2, axis=-1, keepdims=True) * basis1
    cand2_norm = _safe_vector_norm(cand2, xp=xp, axis=-1, keepdims=True, eps=1e-12)

    fallback2 = col3 - xp.sum(basis1 * col3, axis=-1, keepdims=True) * basis1
    fallback2_norm = _safe_vector_norm(fallback2, xp=xp, axis=-1, keepdims=True, eps=1e-12)

    x_axis = xp.broadcast_to(
        xp.asarray([1.0, 0.0, 0.0], dtype=rot_flat.dtype),
        basis1.shape,
    )
    y_axis = xp.broadcast_to(
        xp.asarray([0.0, 1.0, 0.0], dtype=rot_flat.dtype),
        basis1.shape,
    )
    arbitrary_axis = xp.where(xp.abs(basis1[..., 0:1]) < 0.9, x_axis, y_axis)
    arbitrary2 = arbitrary_axis - xp.sum(basis1 * arbitrary_axis, axis=-1, keepdims=True) * basis1

    use_fallback2 = cand2_norm <= 1e-6
    cand2 = xp.where(use_fallback2, fallback2, cand2)
    cand2_norm = xp.where(use_fallback2, fallback2_norm, cand2_norm)
    cand2 = xp.where(cand2_norm <= 1e-6, arbitrary2, cand2)

    basis2 = _normalize_vector(cand2, xp=xp, eps=1e-6)
    basis3 = xp.cross(basis1, basis2)
    basis3 = _normalize_vector(basis3, xp=xp, eps=1e-6)
    basis2 = xp.cross(basis3, basis1)
    basis2 = _normalize_vector(basis2, xp=xp, eps=1e-6)

    projected = xp.stack([basis1, basis2, basis3], axis=-1)
    return xp.reshape(projected, leading_shape + (3, 3))


def split_model_predictions(
    pred: Any,
    qfrc_inverse_output_dim: int = 0,
    rotation_output_dim: int = ROTATION_OUTPUT_DIM,
) -> Tuple[Any, Any, Any, Any, Optional[Any], Optional[Any], Optional[Any]]:
    """Split model outputs into standard channels plus any legacy auxiliary heads."""
    cop_pred = pred[..., COP_SLICE]
    grf_pred = pred[..., GRF_SLICE]
    moments_pred = pred[..., MOMENTS_SLICE]
    contact_pred = pred[..., CONTACT_SLICE]
    offset = STANDARD_OUTPUT_DIM
    qfrc_inverse_pred = None
    if qfrc_inverse_output_dim > 0:
        qfrc_inverse_pred = pred[..., offset:offset + qfrc_inverse_output_dim]
        offset += qfrc_inverse_output_dim
    rotation_pred = None
    if rotation_output_dim > 0:
        rotation_pred = pred[..., offset:offset + rotation_output_dim]
        offset += rotation_output_dim
    jacobian_pred = None
    return (
        cop_pred,
        grf_pred,
        moments_pred,
        contact_pred,
        qfrc_inverse_pred,
        rotation_pred,
        jacobian_pred,
    )


def select_torque_jacobians(
    pred: Any,
    batch: Dict[str, Any],
    normalizers: Dict[str, Any],
    xp=jnp,
) -> Tuple[Any, Any]:
    """Use preprocessed Jacobians for torque computation."""
    return batch["jacp"], batch["jacr"]


def compute_predicted_knee_to_cop_vectors(
    cop_pred_unnorm: jnp.ndarray,  # (batch, seq, 4) [rx, rz, lx, lz] in ground-aligned calc frame
    ankle_pos_global: jnp.ndarray,  # (batch, seq, 2, 3) [right, left] world XYZ
    knee_pos_global: jnp.ndarray,  # (batch, seq, 2, 3) [right, left] world XYZ
    rot_w_to_ga: jnp.ndarray,  # (batch, seq, 2, 3, 3) world->ground-aligned calc rotation
) -> jnp.ndarray:
    """Build predicted knee->COP vectors from predicted COP and global ankle/knee positions.

    The model predicts COP as right/left [calc-frame X, calc-frame Z]. We rotate that
    horizontal foot-relative vector back to world coordinates, then add the global
    ankle-to-knee offset:

        knee->COP = (ankle_global - knee_global) + R_ga_to_world * COP_pred_ga

    Columns match ProcessData.py's saved schema:
    [R_x, R_y, R_z, L_x, L_y, L_z].
    """
    rot_ga_to_w_r = jnp.swapaxes(rot_w_to_ga[:, :, 0], -1, -2)
    rot_ga_to_w_l = jnp.swapaxes(rot_w_to_ga[:, :, 1], -1, -2)
    # Ground-aligned Y is world vertical. COP is on world Z=0, so its vertical
    # displacement from the ankle is -ankle_world_z, not zero.
    cop_r_vertical_ga = -ankle_pos_global[:, :, 0, 2:3]
    cop_l_vertical_ga = -ankle_pos_global[:, :, 1, 2:3]
    cop_r_ga = jnp.concatenate(
        [cop_pred_unnorm[..., 0:1], cop_r_vertical_ga, cop_pred_unnorm[..., 1:2]],
        axis=-1,
    )
    cop_l_ga = jnp.concatenate(
        [cop_pred_unnorm[..., 2:3], cop_l_vertical_ga, cop_pred_unnorm[..., 3:4]],
        axis=-1,
    )
    cop_r_world_rel = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_r, cop_r_ga)
    cop_l_world_rel = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_l, cop_l_ga)
    ankle_to_knee_r = ankle_pos_global[:, :, 0] - knee_pos_global[:, :, 0]
    ankle_to_knee_l = ankle_pos_global[:, :, 1] - knee_pos_global[:, :, 1]
    vec_r = ankle_to_knee_r + cop_r_world_rel
    vec_l = ankle_to_knee_l + cop_l_world_rel
    return jnp.concatenate([vec_r, vec_l], axis=-1)


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray = 1.0) -> jnp.ndarray:
    """Compute weighted Mean Squared Error."""
    return jnp.mean(weights * jnp.square(pred - target))


def huber_loss(pred: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray = 1.0,
               delta: float = 1.0) -> jnp.ndarray:
    """Weighted Huber (smooth-L1) loss.

    Quadratic for |error| <= delta and linear beyond, so heel-strike transients and
    occasional GRF/COP artifacts are down-weighted relative to plain MSE. With the
    same weighted-mean reduction as ``mse_loss`` so loss weights stay comparable.
    Because inputs are Z-scored, delta ~ 1.0 sits near one standard deviation.
    """
    err = pred - target
    abs_err = jnp.abs(err)
    quadratic = jnp.minimum(abs_err, delta)
    linear = abs_err - quadratic
    per_elem = 0.5 * jnp.square(quadratic) + delta * linear
    return jnp.mean(weights * per_elem)


def compute_total_loss(
    pred: jnp.ndarray,
    batch: Dict[str, jnp.ndarray],
    normalizers: Dict,
    loss_weights: Dict[str, float],
    use_contact_weighting: bool,
    magOnOff: bool,
    contactOnOff: bool,
    only_supervise_stance: bool,
    contact_weight_multiplier: float = 1.5,
    magWeight: float = 3.0,
    dof_weights_dict: Optional[Dict[int, float]] = None,  # New parameter for DOF weights
    epoch: float = 1.0,
    total_epochs: float = 1.0,
    cop_mask: bool = True,
    use_grf_norm_cop: bool = False,
    use_gt_jacob_and_rot: bool = False,
    compute_effect_diagnostics: bool = False,
    robust_loss: str = "mse",
    huber_delta: float = 1.0,
    contact_mask_source: str = "gt",
    contact_mix_max_alpha: float = 0.5,
    use_full_id_gt_for_torque: bool = False,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute direct COP/GRF/moment/contact losses plus torque supervision.

    When ``compute_effect_diagnostics`` is False (default), the Tau->COP,
    Tau->GRF, and TauKAM/TauGRF gradient-ratio diagnostics are skipped. These
    diagnostics are logging-only and require extra ``jax.grad`` passes per call,
    so leaving them off removes that per-step cost from training and evaluation.
    """
    cop_pred, grf_pred, moments_pred, contact_pred, qfrc_inverse_pred, rotation_pred, jacobian_pred = split_model_predictions(
        pred,
        qfrc_inverse_output_dim=0,
        rotation_output_dim=0,
    )

    # --- Contact Weighting ---
    contact_bool = batch["contactBoolean"]
    supervision_mask = batch.get("supervision_mask", None)
    if supervision_mask is None:
        supervision_mask = jnp.ones(contact_bool.shape[:-1] + (1,), dtype=pred.dtype)
    else:
        if supervision_mask.ndim == 2:
            supervision_mask = supervision_mask[..., None]
        supervision_mask = supervision_mask.astype(pred.dtype)

    # --- Per-window speed/gender balancing weight ---
    # Each window carries a scalar sample_weight (1.0 when balancing is disabled).
    # Folding it into the supervision mask scales that window's contribution to
    # every downstream loss component, yielding a weighted average across the batch.
    sample_weight = batch.get("sample_weight", None)
    if sample_weight is not None:
        sample_weight = sample_weight.astype(pred.dtype).reshape((-1,) + (1,) * (supervision_mask.ndim - 1))
        supervision_mask = supervision_mask * sample_weight

    contact_r = contact_bool[..., 0:1] # (batch, seq, 1)
    contact_l = contact_bool[..., 1:2] # (batch, seq, 1)
    
    # Create weight masks
    weight_r = 1.0 + (contact_weight_multiplier - 1.0) * contact_r
    weight_l = 1.0 + (contact_weight_multiplier - 1.0) * contact_l
    
    if not use_contact_weighting:
        weight_r = jnp.ones_like(weight_r)
        weight_l = jnp.ones_like(weight_l)

    cop_weights = jnp.concatenate([
        jnp.tile(weight_r, (1, 1, 2)),
        jnp.tile(weight_l, (1, 1, 2)),
    ], axis=-1) * supervision_mask
    grf_weights = jnp.concatenate([
        jnp.tile(weight_r, (1, 1, 3)),
        jnp.tile(weight_l, (1, 1, 3)),
    ], axis=-1) * supervision_mask
    moments_weights = jnp.concatenate([
        jnp.tile(weight_r, (1, 1, 1)),
        jnp.tile(weight_l, (1, 1, 1)),
    ], axis=-1) * supervision_mask

    output_mask_r = (contact_r > 0.5).astype(pred.dtype)
    output_mask_l = (contact_l > 0.5).astype(pred.dtype)
    if cop_mask:
        cop_weights = cop_weights * jnp.concatenate([
            jnp.tile(output_mask_r, (1, 1, 2)),
            jnp.tile(output_mask_l, (1, 1, 2)),
        ], axis=-1)
        grf_weights = grf_weights * jnp.concatenate([
            jnp.tile(output_mask_r, (1, 1, 3)),
            jnp.tile(output_mask_l, (1, 1, 3)),
        ], axis=-1)
        moments_weights = moments_weights * jnp.concatenate([
            output_mask_r,
            output_mask_l,
        ], axis=-1)

    cop_pred_abs = cop_pred
    grf_pred_abs = grf_pred
    moments_pred_abs = moments_pred

    if cop_mask:
        cop_abs_unnorm = normalizers["cop"].unnormalize(cop_pred_abs)
        grf_abs_unnorm = normalizers["grf"].unnormalize(grf_pred_abs)
        mom_abs_unnorm = normalizers["moments"].unnormalize(moments_pred_abs)

        cop_abs_unnorm_masked = jnp.concatenate([cop_abs_unnorm[..., 0:2] * output_mask_r, cop_abs_unnorm[..., 2:4] * output_mask_l], axis=-1)
        grf_abs_unnorm_masked = jnp.concatenate([grf_abs_unnorm[..., 0:3] * output_mask_r, grf_abs_unnorm[..., 3:6] * output_mask_l], axis=-1)
        mom_abs_unnorm_masked = jnp.concatenate([mom_abs_unnorm[..., 0:1] * output_mask_r, mom_abs_unnorm[..., 1:2] * output_mask_l], axis=-1)

        cop_pred_abs = normalizers["cop"].normalize(cop_abs_unnorm_masked)
        grf_pred_abs = normalizers["grf"].normalize(grf_abs_unnorm_masked)
        moments_pred_abs = normalizers["moments"].normalize(mom_abs_unnorm_masked)

    # Primary regression loss for COP/GRF/moments: MSE (default) or robust Huber.
    def _primary_loss(pred_local, target_local, weights_local):
        if robust_loss == "huber":
            return huber_loss(pred_local, target_local, weights_local, delta=huber_delta)
        return mse_loss(pred_local, target_local, weights_local)

    cop_loss = _primary_loss(cop_pred_abs, batch["cop"], cop_weights)
    grf_loss = _primary_loss(grf_pred_abs, batch["grf"], grf_weights)
    moments_loss = _primary_loss(moments_pred_abs, batch["moments"], moments_weights)

    # The model does not predict qfrc_inverse / rotation / jacobian residuals (those
    # output heads are disabled — see split_model_predictions called with dims 0). Their
    # loss terms are therefore fixed to zero (assigned once just before the return dict)
    # and their prediction branches are omitted here. Only rotation_pred_phys is still
    # needed: it is the projected world->ground-aligned rotation used by the torque
    # reconstruction below.
    rotation_pred_phys = project_rotation_matrices(
        jnp.asarray(batch["rot_w_to_ga"], dtype=pred.dtype),
        xp=jnp,
    )

    eps = 1e-7
    contact_pred_clipped = jnp.clip(contact_pred, eps, 1.0 - eps)
    contact_bce = -(
        contact_bool * jnp.log(contact_pred_clipped) +
        (1.0 - contact_bool) * jnp.log(1.0 - contact_pred_clipped)
    )
    contact_loss = jnp.mean(contact_bce * supervision_mask)

    output_reg_loss = jnp.zeros_like(cop_loss)

    static_unnorm = normalizers["static"].unnormalize(batch["static_context"])
    height_m = static_unnorm[:, 0:1, None]
    mass_kg = static_unnorm[:, 1:2, None]

    cop_pred_phys = normalizers["cop"].unnormalize(cop_pred_abs)
    grf_pred_phys = normalizers["grf"].unnormalize(grf_pred_abs)
    moments_pred_phys = normalizers["moments"].unnormalize(moments_pred_abs)

    # --- Plan 5: contact mask used for the TORQUE reconstruction path ---
    # 'gt' (default) reproduces prior behavior exactly: the torque wrench is gated by
    # ground-truth contact. 'pred'/'mixed' gate (partly) by the model's own predicted
    # contact so training reflects inference-time behavior, where predicted-contact
    # errors corrupt COP/GRF/torque. In non-'gt' modes we recompute the physical wrench
    # from the RAW (un-GT-masked) predictions so predicted contact can both add a foot
    # (false stance in swing) and remove one (false swing in stance). Direct COP/GRF/
    # moment supervision above stays GT-masked so its targets remain well-defined.
    pred_contact_hard_r = (contact_pred[..., 0:1] >= 0.5).astype(pred.dtype)
    pred_contact_hard_l = (contact_pred[..., 1:2] >= 0.5).astype(pred.dtype)
    if contact_mask_source == "pred":
        torque_mask_r = pred_contact_hard_r
        torque_mask_l = pred_contact_hard_l
    elif contact_mask_source == "mixed":
        alpha = jnp.clip(
            jnp.asarray(epoch, dtype=pred.dtype) / jnp.maximum(jnp.asarray(total_epochs, dtype=pred.dtype), 1.0),
            0.0, 1.0,
        ) * jnp.asarray(contact_mix_max_alpha, dtype=pred.dtype)
        torque_mask_r = (1.0 - alpha) * output_mask_r + alpha * pred_contact_hard_r
        torque_mask_l = (1.0 - alpha) * output_mask_l + alpha * pred_contact_hard_l
    else:  # 'gt'
        torque_mask_r = output_mask_r
        torque_mask_l = output_mask_l

    if cop_mask:
        mask_r_t = torque_mask_r
        mask_l_t = torque_mask_l
        if contact_mask_source == "gt":
            base_cop_phys, base_grf_phys, base_mom_phys = cop_pred_phys, grf_pred_phys, moments_pred_phys
        else:
            base_cop_phys = normalizers["cop"].unnormalize(cop_pred)
            base_grf_phys = normalizers["grf"].unnormalize(grf_pred)
            base_mom_phys = normalizers["moments"].unnormalize(moments_pred)
        cop_pred_phys = jnp.concatenate([
            base_cop_phys[..., 0:2] * mask_r_t,
            base_cop_phys[..., 2:4] * mask_l_t,
        ], axis=-1)
        grf_pred_phys = jnp.concatenate([
            base_grf_phys[..., 0:3] * mask_r_t,
            base_grf_phys[..., 3:6] * mask_l_t,
        ], axis=-1)
        moments_pred_phys = jnp.concatenate([
            base_mom_phys[..., 0:1] * mask_r_t,
            base_mom_phys[..., 1:2] * mask_l_t,
        ], axis=-1)

    grf_unnorm = grf_pred_phys * (mass_kg * 9.8067)
    cop_unnorm = decode_cop_signal_to_length(
        cop_pred_phys,
        grf_pred_phys,
        height_m,
        use_grf_norm_cop=use_grf_norm_cop,
        contact_probability=contact_pred if use_grf_norm_cop else None,
        xp=jnp,
    )
    moments_unnorm = moments_pred_phys * (mass_kg * 9.8067 * height_m)
    grf_pred_abs = grf_pred_phys

    torque_cop_unnorm = cop_unnorm
    torque_grf_unnorm = grf_unnorm
    torque_moments_unnorm = moments_unnorm
    # When use_gt_jacob_and_rot is set, the predicted COP is taken to world with the
    # ground-truth (MoCap) rotation and the wrench is mapped to joint torques with the
    # ground-truth (MoCap) Jacobian, instead of the video (ProcessedData) terms. The
    # model still consumes video inputs; only the torque reconstruction uses GT kinematics.
    if use_gt_jacob_and_rot:
        torque_rotation_pred_phys = project_rotation_matrices(
            jnp.asarray(batch["gt_rot_w_to_ga"], dtype=pred.dtype), xp=jnp
        )
        kam_ankle_pos = jnp.asarray(batch["gt_ankle_pos"], dtype=pred.dtype)
        kam_knee_pos = jnp.asarray(batch["gt_knee_pos"], dtype=pred.dtype)
    else:
        torque_rotation_pred_phys = rotation_pred_phys
        kam_ankle_pos = jnp.asarray(batch["ankle_pos"], dtype=pred.dtype)
        kam_knee_pos = jnp.asarray(batch["knee_pos"], dtype=pred.dtype)
    full_moments = compute_full_external_moments(
        torque_cop_unnorm,
        torque_grf_unnorm,
        torque_moments_unnorm,
        batch["ankle_heights"],
        torque_rotation_pred_phys,
    )

    if use_gt_jacob_and_rot:
        jacp_for_tau = jnp.asarray(batch["gt_jacp"], dtype=pred.dtype)
        jacr_for_tau = jnp.asarray(batch["gt_jacr"], dtype=pred.dtype)
    else:
        jacp_for_tau, jacr_for_tau = select_torque_jacobians(
            pred,
            batch,
            normalizers,
            xp=jnp,
        )
    tau_grf_pred = compute_tau_grf_from_predictions(
        torque_grf_unnorm, full_moments, jacp_for_tau, jacr_for_tau
    )
    if use_full_id_gt_for_torque:
        # A torque-informed COP/GRF model predicts the external generalized-force
        # contribution (tau_grf). Its corresponding full inverse-dynamics prediction is
        # qfrc_inverse - tau_grf. Therefore, when OpenSim full ID is ground truth, the
        # equivalent external-torque target is qfrc_inverse - OpenSim_ID. This keeps the
        # kinematics/qfrc_inverse term on the same aligned input frames as the ID target.
        if batch.get("id_gt_mjx") is None:
            raise ValueError(
                "use_full_id_gt_for_torque=True requires batch['id_gt_mjx']."
            )
        qfrc_inverse_for_target = batch.get("qfrc_inverse_input_raw")
        if qfrc_inverse_for_target is None:
            raise ValueError(
                "use_full_id_gt_for_torque=True requires batch['qfrc_inverse_input_raw']."
            )
        full_id_target = jnp.asarray(batch["id_gt_mjx"], dtype=pred.dtype)
        opensim_tau_grf_target = (
            jnp.asarray(qfrc_inverse_for_target, dtype=pred.dtype) - full_id_target
        )
        # OpenSim has no generalized forces for the three pelvis translations in
        # these files. Keep the legacy target in unavailable columns so NaN*zero
        # cannot contaminate the masked loss; all supervised rotational DOFs use ID.
        target_tau_grf = jnp.where(
            jnp.isfinite(full_id_target),
            opensim_tau_grf_target,
            jnp.asarray(batch["qfrc_grf_contribution"], dtype=pred.dtype),
        )
    else:
        target_tau_grf = jnp.asarray(batch["qfrc_grf_contribution"], dtype=pred.dtype)
    nv = int(target_tau_grf.shape[-1])
    cop_target_phys = normalizers["cop"].unnormalize(batch["cop"])
    grf_target_phys = normalizers["grf"].unnormalize(batch["grf"])
    moments_target_phys = normalizers["moments"].unnormalize(batch["moments"])
    target_grf_unnorm = grf_target_phys * (mass_kg * 9.8067)
    target_cop_unnorm = decode_cop_signal_to_length(
        cop_target_phys,
        grf_target_phys,
        height_m,
        use_grf_norm_cop=use_grf_norm_cop,
        xp=jnp,
    )
    target_moments_unnorm = moments_target_phys * (mass_kg * 9.8067 * height_m)

    active_indices = np.array([
        6, 7, 9, 10, 11,
        13, 14, 16, 17, 18,
        20, 21, 22,
    ], dtype=np.int32)
    active_indices = active_indices[active_indices < nv]
    active_indices = jnp.asarray(active_indices, dtype=jnp.int32)
    torque_mask = jnp.zeros((nv,))
    torque_mask = torque_mask.at[active_indices].set(1.0)
    torque_mask = torque_mask[None, None, :]

    norm_factor = mass_kg * 9.8067 * height_m
    target_norm = target_tau_grf / norm_factor
    pred_norm = tau_grf_pred / norm_factor
    torque_diff = pred_norm - target_norm
    raw_error = jnp.square(torque_diff)

    torque_knee_adduction_loss = jnp.zeros_like(cop_loss)
    knee_to_cop_vectors = batch.get("knee_to_cop_vectors", None)
    if knee_to_cop_vectors is not None and int(knee_to_cop_vectors.shape[-1]) >= 6:
        knee_to_cop_vectors = jnp.asarray(knee_to_cop_vectors, dtype=pred.dtype)
        pred_knee_to_cop_vectors = compute_predicted_knee_to_cop_vectors(
            cop_unnorm,
            kam_ankle_pos,
            kam_knee_pos,
            torque_rotation_pred_phys,
        )
        pred_z_vec_l = pred_knee_to_cop_vectors[..., 5]
        pred_y_vec_l = pred_knee_to_cop_vectors[..., 4]
        z_vec_l = knee_to_cop_vectors[..., 5]
        y_vec_l = knee_to_cop_vectors[..., 4]
        kam_l_pred = pred_z_vec_l * grf_unnorm[..., 4] - pred_y_vec_l * grf_unnorm[..., 5]
        kam_l_gt = z_vec_l * target_grf_unnorm[..., 4] - y_vec_l * target_grf_unnorm[..., 5]
        kam_l_diff_norm = (kam_l_pred - kam_l_gt) / jnp.maximum(norm_factor[..., 0], 1e-8)
        kam_l_weights = (output_mask_l[..., 0] * supervision_mask[..., 0]).astype(pred.dtype)
        torque_knee_adduction_loss = (
            jnp.sum(jnp.square(kam_l_diff_norm) * kam_l_weights) /
            jnp.maximum(jnp.sum(kam_l_weights), 1.0)
        )

    contact_multiplier = jnp.ones((1, 1, nv))
    if use_contact_weighting:
        right_indices = np.array([6, 7, 8, 9, 10, 11, 12], dtype=np.int32)
        left_indices = np.array([13, 14, 15, 16, 17, 18, 19], dtype=np.int32)
        right_indices = jnp.asarray(right_indices[right_indices < nv], dtype=jnp.int32)
        left_indices = jnp.asarray(left_indices[left_indices < nv], dtype=jnp.int32)
        right_mask = jnp.zeros((nv,))
        right_mask = right_mask.at[right_indices].set(1.0)
        right_mask = right_mask[None, None, :]
        left_mask = jnp.zeros((nv,))
        left_mask = left_mask.at[left_indices].set(1.0)
        left_mask = left_mask[None, None, :]
        contact_multiplier = (
            1.0 * (1.0 - right_mask - left_mask) +
            weight_r * right_mask +
            weight_l * left_mask
        )

    magnitude_multiplier = jnp.abs(target_norm) / magWeight + 0.5

    if dof_weights_dict is None:
        dof_weights_dict = {
            6: 1.0,
            7: 1.0,
            9: 1.0,
            10: 1.0,
            11: 1.0,
            13: 1.0,
            14: 1.0,
            16: 1.0,
            17: 1.0,
            18: 1.0,
            20: 1.0,
            21: 1.0,
            22: 1.0,
        }

    individual_multiplier = jnp.ones((nv,))
    valid_weight_items = [
        (idx, val) for idx, val in dof_weights_dict.items()
        if int(idx) < nv
    ]
    if valid_weight_items:
        ind_indices = jnp.asarray([idx for idx, _val in valid_weight_items], dtype=jnp.int32)
        ind_values = jnp.asarray([val for _idx, val in valid_weight_items], dtype=pred.dtype)
        individual_multiplier = individual_multiplier.at[ind_indices].set(ind_values)
    individual_multiplier = individual_multiplier[None, None, :]

    if not magOnOff:
        magnitude_multiplier = jnp.ones_like(magnitude_multiplier)
    if not contactOnOff:
        contact_multiplier = jnp.ones_like(contact_multiplier)

    raw_weights = contact_multiplier * magnitude_multiplier * individual_multiplier
    raw_weights = raw_weights * torque_mask
    final_weights = raw_weights * supervision_mask
    num_active_dof = jnp.sum(torque_mask)

    def _weighted_torque_loss_from_prediction(tau_grf_pred_local: jnp.ndarray) -> jnp.ndarray:
        pred_norm_local = tau_grf_pred_local / norm_factor
        raw_error_local = jnp.square(pred_norm_local - target_norm)
        weighted_error_local = raw_error_local * final_weights
        return jnp.sum(weighted_error_local) / jnp.maximum(num_active_dof, 1.0)

    torque_loss = _weighted_torque_loss_from_prediction(tau_grf_pred)

    def _branch_outputs_to_unnorm(
        cop_pred_abs_local: jnp.ndarray,
        grf_pred_abs_local: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cop_pred_phys_local = normalizers["cop"].unnormalize(cop_pred_abs_local)
        grf_pred_phys_local = normalizers["grf"].unnormalize(grf_pred_abs_local)
        if cop_mask:
            cop_pred_phys_local = jnp.concatenate([
                cop_pred_phys_local[..., 0:2] * mask_r_t,
                cop_pred_phys_local[..., 2:4] * mask_l_t,
            ], axis=-1)
            grf_pred_phys_local = jnp.concatenate([
                grf_pred_phys_local[..., 0:3] * mask_r_t,
                grf_pred_phys_local[..., 3:6] * mask_l_t,
            ], axis=-1)
        return (
            decode_cop_signal_to_length(
                cop_pred_phys_local,
                grf_pred_phys_local,
                height_m,
                use_grf_norm_cop=use_grf_norm_cop,
                contact_probability=contact_pred if use_grf_norm_cop else None,
                xp=jnp,
            ),
            grf_pred_phys_local * (mass_kg * 9.8067),
        )

    def _torque_loss_from_branch_outputs(
        cop_pred_abs_local: jnp.ndarray,
        grf_pred_abs_local: jnp.ndarray,
    ) -> jnp.ndarray:
        cop_unnorm_local, grf_unnorm_local = _branch_outputs_to_unnorm(
            cop_pred_abs_local,
            grf_pred_abs_local,
        )
        full_moments_local = compute_full_external_moments(
            cop_unnorm_local,
            grf_unnorm_local,
            torque_moments_unnorm,
            batch["ankle_heights"],
            torque_rotation_pred_phys,
        )
        tau_grf_local = compute_tau_grf_from_predictions(
            grf_unnorm_local,
            full_moments_local,
            jacp_for_tau,
            jacr_for_tau,
        )
        return _weighted_torque_loss_from_prediction(tau_grf_local)

    def _knee_adduction_loss_from_branch_outputs(
        cop_pred_abs_local: jnp.ndarray,
        grf_pred_abs_local: jnp.ndarray,
    ) -> jnp.ndarray:
        knee_to_cop_local = batch.get("knee_to_cop_vectors", None)
        if knee_to_cop_local is None or int(knee_to_cop_local.shape[-1]) < 6:
            return jnp.zeros_like(cop_loss)
        cop_unnorm_local, grf_unnorm_local = _branch_outputs_to_unnorm(
            cop_pred_abs_local,
            grf_pred_abs_local,
        )
        knee_to_cop_local = jnp.asarray(knee_to_cop_local, dtype=pred.dtype)
        pred_knee_to_cop_local = compute_predicted_knee_to_cop_vectors(
            cop_unnorm_local,
            kam_ankle_pos,
            kam_knee_pos,
            torque_rotation_pred_phys,
        )
        pred_z_vec_l = pred_knee_to_cop_local[..., 5]
        pred_y_vec_l = pred_knee_to_cop_local[..., 4]
        z_vec_l = knee_to_cop_local[..., 5]
        y_vec_l = knee_to_cop_local[..., 4]
        kam_l_pred = pred_z_vec_l * grf_unnorm_local[..., 4] - pred_y_vec_l * grf_unnorm_local[..., 5]
        kam_l_gt = z_vec_l * target_grf_unnorm[..., 4] - y_vec_l * target_grf_unnorm[..., 5]
        kam_l_diff_norm = (kam_l_pred - kam_l_gt) / jnp.maximum(norm_factor[..., 0], 1e-8)
        kam_l_weights = (output_mask_l[..., 0] * supervision_mask[..., 0]).astype(pred.dtype)
        return (
            jnp.sum(jnp.square(kam_l_diff_norm) * kam_l_weights) /
            jnp.maximum(jnp.sum(kam_l_weights), 1.0)
        )

    def _cop_direct_loss_from_pred(cop_pred_abs_local: jnp.ndarray) -> jnp.ndarray:
        return _primary_loss(cop_pred_abs_local, batch["cop"], cop_weights) / 4

    def _grf_direct_loss_from_pred(grf_pred_abs_local: jnp.ndarray) -> jnp.ndarray:
        return _primary_loss(grf_pred_abs_local, batch["grf"], grf_weights) / 6

    def _grad_rms(grad_tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.mean(jnp.square(grad_tensor)))

    torque_cop_effect_loss = jnp.zeros_like(torque_loss)
    torque_grf_effect_loss = jnp.zeros_like(torque_loss)
    torque_knee_adduction_cop_grad_rms = jnp.zeros_like(torque_loss)
    torque_knee_adduction_grf_grad_rms = jnp.zeros_like(torque_loss)
    torque_cop_grad_rms = jnp.zeros_like(torque_loss)
    torque_grf_grad_rms = jnp.zeros_like(torque_loss)
    torque_knee_adduction_to_torque_cop_grad_ratio = jnp.zeros_like(torque_loss)
    torque_knee_adduction_to_torque_grf_grad_ratio = jnp.zeros_like(torque_loss)
    grad_eps = jnp.asarray(1e-8, dtype=pred.dtype)

    # These Tau->COP / Tau->GRF gradient-ratio diagnostics are logging-only: they
    # are NOT summed into total_loss below. Each requires two jax.grad passes, so
    # computing them unconditionally adds four backward passes to every train and
    # eval step. Gate them behind compute_effect_diagnostics; when off they stay at
    # the zero values initialized above.
    if compute_effect_diagnostics:
        torque_weight = jnp.asarray(loss_weights.get("torque", 1.0), dtype=pred.dtype)
        kam_weight = jnp.asarray(loss_weights.get("torque_knee_adduction", 0.0), dtype=pred.dtype)
        torque_cop_grad = jax.grad(
            lambda cop_pred_abs_local: _torque_loss_from_branch_outputs(
                cop_pred_abs_local,
                jax.lax.stop_gradient(grf_pred_abs),
            )
        )(cop_pred_abs)
        direct_cop_grad = jax.grad(_cop_direct_loss_from_pred)(cop_pred_abs)
        torque_cop_effect_loss = cop_loss * (
            _grad_rms(torque_cop_grad) / jnp.maximum(_grad_rms(direct_cop_grad), grad_eps)
        )
        torque_cop_grad_rms = _grad_rms(torque_weight * torque_cop_grad)

        kam_cop_grad = jax.grad(
            lambda cop_pred_abs_local: _knee_adduction_loss_from_branch_outputs(
                cop_pred_abs_local,
                jax.lax.stop_gradient(grf_pred_abs),
            )
        )(cop_pred_abs)
        torque_knee_adduction_cop_grad_rms = _grad_rms(kam_weight * kam_cop_grad)
        torque_knee_adduction_to_torque_cop_grad_ratio = (
            torque_knee_adduction_cop_grad_rms / jnp.maximum(torque_cop_grad_rms, grad_eps)
        )

        torque_grf_grad = jax.grad(
            lambda grf_pred_abs_local: _torque_loss_from_branch_outputs(
                jax.lax.stop_gradient(cop_pred_abs),
                grf_pred_abs_local,
            )
        )(grf_pred_abs)
        direct_grf_grad = jax.grad(_grf_direct_loss_from_pred)(grf_pred_abs)
        torque_grf_effect_loss = grf_loss * (
            _grad_rms(torque_grf_grad) / jnp.maximum(_grad_rms(direct_grf_grad), grad_eps)
        )
        torque_grf_grad_rms = _grad_rms(torque_weight * torque_grf_grad)

        kam_grf_grad = jax.grad(
            lambda grf_pred_abs_local: _knee_adduction_loss_from_branch_outputs(
                jax.lax.stop_gradient(cop_pred_abs),
                grf_pred_abs_local,
            )
        )(grf_pred_abs)
        torque_knee_adduction_grf_grad_rms = _grad_rms(kam_weight * kam_grf_grad)
        torque_knee_adduction_to_torque_grf_grad_ratio = (
            torque_knee_adduction_grf_grad_rms / jnp.maximum(torque_grf_grad_rms, grad_eps)
        )

    com_accel = batch["com_accel"]
    pred_fx = (grf_pred_abs[..., 0] + grf_pred_abs[..., 3]) * mass_kg * 9.8067
    res_x = mass_kg * com_accel[..., 0] - pred_fx
    pred_fy = (grf_pred_abs[..., 1] + grf_pred_abs[..., 4]) * mass_kg * 9.8067
    res_y = mass_kg * com_accel[..., 1] - pred_fy
    pred_fz = (grf_pred_abs[..., 2] + grf_pred_abs[..., 5]) * mass_kg * 9.8067
    gravity = 9.8067
    res_z = mass_kg * (com_accel[..., 2] + gravity) - pred_fz
    grf_res = jnp.stack([res_x, res_y, res_z], axis=-1)
    if "grf_res" in normalizers:
        grf_res_norm = normalizers["grf_res"].normalize(grf_res)
        grf_correction_loss = jnp.mean(jnp.square(grf_res_norm) * supervision_mask)
    else:
        grf_correction_loss = jnp.mean(jnp.square(grf_res) * supervision_mask)

    cop_loss = cop_loss / 4
    grf_loss = grf_loss / 6
    moments_loss = moments_loss / 2
    qfrc_inverse_loss = jnp.zeros_like(cop_loss)
    qfrc_inverse_input_reg_loss = jnp.zeros_like(cop_loss)
    rotation_loss = jnp.zeros_like(cop_loss)
    rotation_input_reg_loss = jnp.zeros_like(cop_loss)
    jacobian_loss = jnp.zeros_like(cop_loss)
    jacobian_input_reg_loss = jnp.zeros_like(cop_loss)
    contact_loss = contact_loss / 2
    grf_correction_loss = grf_correction_loss / 3

    epoch_f = jnp.asarray(epoch, dtype=jnp.float32)
    total_epochs_f = jnp.maximum(jnp.asarray(total_epochs, dtype=jnp.float32), 1.0)
    output_reg_multiplier = jnp.clip(1.0 - (epoch_f / total_epochs_f), 0.0, 1.0)
    total_loss = (
        loss_weights.get("cop", 1.0) * cop_loss +
        loss_weights.get("grf", 1.0) * grf_loss +
        loss_weights.get("moments", 1.0) * moments_loss +
        loss_weights.get("contact", 1.0) * contact_loss +
        loss_weights.get("torque", 1.0) * torque_loss +
        loss_weights.get("torque_knee_adduction", 0.0) * torque_knee_adduction_loss +
        loss_weights.get("grf_correction", 1.0) * grf_correction_loss
    )
    
    return total_loss, {
        "cop_loss": cop_loss,
        "grf_loss": grf_loss,
        "moments_loss": moments_loss,
        "qfrc_inverse_loss": qfrc_inverse_loss,
        "qfrc_inverse_input_reg_loss": qfrc_inverse_input_reg_loss,
        "rotation_loss": rotation_loss,
        "rotation_input_reg_loss": rotation_input_reg_loss,
        "jacobian_loss": jacobian_loss,
        "jacobian_input_reg_loss": jacobian_input_reg_loss,
        "contact_loss": contact_loss,
        "torque_loss": torque_loss,
        "torque_knee_adduction_loss": torque_knee_adduction_loss,
        "torque_cop_effect_loss": torque_cop_effect_loss,
        "torque_grf_effect_loss": torque_grf_effect_loss,
        "torque_cop_grad_rms": torque_cop_grad_rms,
        "torque_grf_grad_rms": torque_grf_grad_rms,
        "torque_knee_adduction_cop_grad_rms": torque_knee_adduction_cop_grad_rms,
        "torque_knee_adduction_grf_grad_rms": torque_knee_adduction_grf_grad_rms,
        "torque_knee_adduction_to_torque_cop_grad_ratio": torque_knee_adduction_to_torque_cop_grad_ratio,
        "torque_knee_adduction_to_torque_grf_grad_ratio": torque_knee_adduction_to_torque_grf_grad_ratio,
        "grf_correction_loss": grf_correction_loss,
        "output_reg_loss": output_reg_loss,
        "total_loss": total_loss,
    }
