"""Utilities for direct joint-torque prediction models.

The direct-torque model predicts selected joint moments directly in
percent body-weight-height units.  Most channels come from the independent
23-DOF ``ID_GT_MJX`` bundle; knee adduction moment is computed from the same
KneeToCOP/GRF formula used by ``infer.py``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - lets metadata tools import without JAX.
    jnp = None


MODEL_STRUCTURE = "direct_torque"

DIRECT_TORQUE_NAMES = (
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_flexion_r",
    "knee_adduction_r",
    "ankle_flexion_r",
    "subtalar_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_flexion_l",
    "knee_adduction_l",
    "ankle_flexion_l",
    "subtalar_l",
)

DIRECT_TORQUE_OUTPUT_DIM = len(DIRECT_TORQUE_NAMES)

# Independent 23-DOF ProcessData schema:
# pelvis(6), R hip(3), R knee, R ankle, R subtalar, R mtp,
# L hip(3), L knee, L ankle, L subtalar, L mtp, lumbar(3)
ID_GT_COLUMN_BY_TARGET = {
    "hip_flexion_r": 6,
    "hip_adduction_r": 7,
    "hip_rotation_r": 8,
    "knee_flexion_r": 9,
    "ankle_flexion_r": 10,
    "subtalar_r": 11,
    "hip_flexion_l": 13,
    "hip_adduction_l": 14,
    "hip_rotation_l": 15,
    "knee_flexion_l": 16,
    "ankle_flexion_l": 17,
    "subtalar_l": 18,
}

GRAVITY = 9.8067


def _xp_from_array(arr: Any):
    if jnp is not None and hasattr(arr, "__module__") and "jax" in str(type(arr)).lower():
        return jnp
    return np


def _as_xp(xp_name: str):
    if xp_name == "jax":
        if jnp is None:
            raise RuntimeError("JAX is required for xp='jax'")
        return jnp
    return np


def is_direct_torque_hparams(hparams: Mapping[str, Any]) -> bool:
    """Return True when a hyperparameter/checkpoint dict declares direct torque."""
    if not hparams:
        return False
    structure = str(
        hparams.get("model_structure")
        or hparams.get("model_type")
        or hparams.get("architecture")
        or ""
    ).strip().lower()
    if structure in {"direct_torque", "directtorque", "torque_direct"}:
        return True
    return bool(hparams.get("direct_torque_model", False))


def bodyweight_height_norm_factor_from_static(static_context: Any, *, xp=np) -> Any:
    """Return BW*height from raw static context [height, mass, ...]."""
    static_arr = xp.asarray(static_context)
    height = static_arr[..., 0:1]
    mass = static_arr[..., 1:2]
    return mass * height * GRAVITY


def _normalization_factor_for_series(
    id_gt_mjx: Any,
    static_context: Any,
    qfrc_inverse_norm_factor: Optional[Any],
    *,
    xp=np,
) -> Any:
    """Return a factor broadcastable to (..., 1) torque channels."""
    if qfrc_inverse_norm_factor is not None:
        factor = xp.asarray(qfrc_inverse_norm_factor)
        if factor.ndim >= 2:
            return factor[..., 0:1]
        return factor[..., None]
    static_arr = xp.asarray(static_context)
    factor = bodyweight_height_norm_factor_from_static(static_arr, xp=xp)
    while factor.ndim < xp.asarray(id_gt_mjx).ndim:
        factor = factor[..., None, :]
    return factor


def compute_kam_percent_bwh_from_grf(
    knee_to_cop_vectors: Any,
    grf_normalized_by_bw: Any,
    static_context: Any,
    *,
    side: str,
    xp=np,
) -> Any:
    """Compute knee adduction moment in %BW*height.

    ``grf_normalized_by_bw`` is the loader's GRF target: physical GRF divided by
    body weight.  Therefore KAM/(BW*height)*100 is:
    ``(z_vec * grf_y_bw - y_vec * grf_z_bw) / height * 100``.
    """
    vectors = xp.asarray(knee_to_cop_vectors)
    grf = xp.asarray(grf_normalized_by_bw)
    static_arr = xp.asarray(static_context)
    height = static_arr[..., 0:1]
    while height.ndim < grf.ndim:
        height = height[..., None, :]

    if side.lower().startswith("r"):
        y_vec = vectors[..., 1]
        z_vec = vectors[..., 2]
        grf_y = grf[..., 1]
        grf_z = grf[..., 2]
    else:
        y_vec = vectors[..., 4]
        z_vec = vectors[..., 5]
        grf_y = grf[..., 4]
        grf_z = grf[..., 5]

    kam_ratio = (z_vec * grf_y - y_vec * grf_z) / xp.squeeze(height, axis=-1)
    return kam_ratio * 100.0


def build_direct_torque_targets(
    batch: Mapping[str, Any],
    *,
    xp_name: str = "numpy",
) -> Any:
    """Build the 14-channel direct torque target in %BW*height units."""
    xp = _as_xp("jax" if xp_name == "jax" else "numpy")
    id_gt = batch.get("id_gt_mjx")
    if id_gt is None:
        qfrc = batch.get("qfrc_inverse_gt_raw")
        if qfrc is None:
            qfrc = batch.get("qfrc_inverse_input_raw")
        tau = batch.get("qfrc_grf_contribution")
        if qfrc is None or tau is None:
            raise ValueError("Direct torque target requires id_gt_mjx or qfrc_inverse_raw + qfrc_grf_contribution")
        id_gt = xp.asarray(qfrc) - xp.asarray(tau)
    else:
        id_gt = xp.asarray(id_gt)

    static_context = xp.asarray(batch["static_context"])
    norm_factor = _normalization_factor_for_series(
        id_gt,
        static_context,
        batch.get("qfrc_inverse_norm_factor"),
        xp=xp,
    )

    def _id_percent(name: str) -> Any:
        return id_gt[..., ID_GT_COLUMN_BY_TARGET[name]] / xp.squeeze(norm_factor, axis=-1) * 100.0

    knee_to_cop = batch.get("knee_to_cop_vectors")
    if knee_to_cop is None:
        raise ValueError("Direct torque target requires knee_to_cop_vectors for KAM channels")
    kam_r = compute_kam_percent_bwh_from_grf(
        knee_to_cop,
        batch["grf"],
        static_context,
        side="r",
        xp=xp,
    )
    kam_l = compute_kam_percent_bwh_from_grf(
        knee_to_cop,
        batch["grf"],
        static_context,
        side="l",
        xp=xp,
    )

    channels = [
        _id_percent("hip_flexion_r"),
        _id_percent("hip_adduction_r"),
        _id_percent("hip_rotation_r"),
        _id_percent("knee_flexion_r"),
        kam_r,
        _id_percent("ankle_flexion_r"),
        _id_percent("subtalar_r"),
        _id_percent("hip_flexion_l"),
        _id_percent("hip_adduction_l"),
        _id_percent("hip_rotation_l"),
        _id_percent("knee_flexion_l"),
        kam_l,
        _id_percent("ankle_flexion_l"),
        _id_percent("subtalar_l"),
    ]
    return xp.stack(channels, axis=-1)


def direct_torque_percent_to_nm(torque_percent_bwh: Any, static_context: Any, *, xp=np) -> Any:
    """Convert %BW*height direct torque channels back to Nm."""
    torque = xp.asarray(torque_percent_bwh)
    factor = bodyweight_height_norm_factor_from_static(static_context, xp=xp)
    while factor.ndim < torque.ndim:
        factor = factor[..., None, :]
    return torque / 100.0 * factor


def finite_direct_torque_mask(targets: Any, *, xp=np) -> Any:
    """Return a frame-level mask requiring all direct torque channels to be finite."""
    arr = xp.asarray(targets)
    return xp.all(xp.isfinite(arr), axis=-1, keepdims=True)
