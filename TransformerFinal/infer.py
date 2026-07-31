"""Inference script for COP/GRF/Moments prediction model.

Workflow:
1. Load trial via data_loader (includes physiological normalization: COP/h, GRF/m, Moments/m).
2. Augmented input features are Z-score normalized.
3. Model predicts the standard 14 outputs (COP:4 as [Rx,Rz,Lx,Lz], GRF:6, Moments:2, Contact:2).
4. Predictions are unnormalized back to physical units (Newtons, Nm) for physics comparison.
5. Physics consistency is checked via Jacobian-based torque calculation using
   the preprocessed Jacobian, rotation, and qfrc_inverse files.
"""

import os
try:
    from wandb_utils import configure_runtime_env
except ModuleNotFoundError:
    def configure_runtime_env():
        return {}

RUNTIME_ENV_APPLIED = configure_runtime_env()

import sys
import json
import argparse
import pickle
import time
import gc
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Mapping
from glob import glob
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks as scipy_find_peaks
from flax import linen as nn
from flax.training import train_state
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import memory-efficient data loader
from data_loader import (
    TrialDataLoader,
    build_window_supervision_mask,
    flatten_jacobian_components,
    load_single_trial,
    mocap_processed_dir,
    normalize_input_source_name,
    select_pos_input_columns,
    source_processed_dir,
    unnormalize_qfrc_inverse_by_bw_height,
    validate_prediction_margin,
    video_processed_dir,
)


# =============================================================================
# Configuration
# =============================================================================
# Post-inference filtering settings
FilterPostInfer = False  # Temporarily disable post-inference filtering while debugging edge effects
FILTER_CUTOFF_HZ = 6.0  # Cutoff frequency for 4th-order Butterworth filter
FILTER_SAMPLING_RATE_HZ = 100.0  # Assumed sampling rate of the data
DEFAULT_RESTRICT_BOUNDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "input_distributions_stance"
    / "trusted"
    / "cnn_temporal_input_percentile_bounds.npy"
)


# =============================================================================
# Constants & Indices
# =============================================================================
# Jacobian indices for Knee and Ankle flexion/extension
IDX_KNEE_R, IDX_ANKLE_R = 11, 14
IDX_KNEE_L, IDX_ANKLE_L = 22, 25

LEFT_STANCE_THRESHOLD_N = 10.0
COMPLETE_STANCE_THRESHOLD_N = 5.0          # low boundary threshold (N) for walking out stance edges
COMPLETE_STANCE_MIN_FRAMES = 20            # legacy; retained for metadata only
COMPLETE_STANCE_MIN_IMPULSE_BW_RATIO = 0.55  # legacy; retained for metadata only
# Dual-threshold (hysteresis) stance detection: a region only counts as a stance
# if its vertical GRF rises above this fraction of body weight (noise-immune
# core). Stance boundaries are then walked outward from the core to the 5 N
# low threshold above, capturing the low-force heel-strike / toe-off tails.
COMPLETE_STANCE_CORE_BW_RATIO = 0.20
OPENSIM_PEAK_MIN_W_PER_KG = 1.0        # minimum OpenSim ankle power peak to store (W/kg)
OPENSIM_PEAK_MIN_TRIAL_FRAMES = 64      # 0.64 s at 100 Hz — skip peaks if trial is shorter
# Fallback average stance durations (frames at 100 Hz) when no complete stances are found in trial
FALLBACK_STANCE_FRAMES_BY_SPEED: Dict[str, int] = {"80": 65, "100": 58, "120": 52}
DEFAULT_SPEED_MAPPING_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "old_young_adult_walking_speed_mapping.json"
)
_SPEED_MAPPING_CACHE: Dict[str, Any] = {}
LEFT_STANCE_KAM_DOF_NAME = "knee_adduction_moment_l"
SELECTED_LEFT_STANCE_DOF_NAMES = (
    "ankle_angle_l",
    "hip_rotation_l",
    "hip_adduction_l",
    "hip_flexion_l",
    "knee_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
    "subtalar_angle_l",
)

STANDARD_OUTPUT_DIM = 14
COP_SLICE = slice(0, 4)
GRF_SLICE = slice(4, 10)
MOMENTS_SLICE = slice(10, 12)
CONTACT_SLICE = slice(12, 14)
ROTATION_RESIDUAL_FEET = 2
ROTATION_RESIDUAL_AXIS_DIM = 3
ROTATION_OUTPUT_DIM = ROTATION_RESIDUAL_FEET * ROTATION_RESIDUAL_AXIS_DIM
DEFAULT_ROTATION_RESIDUAL_MAX_DEG = 15.0
PREDICTED_JACOBIAN_BODY_COUNT = 2
PREDICTED_JACOBIAN_COMPONENT_COUNT = 2
PREDICTED_JACOBIAN_SPATIAL_DIMS = 3
PREDICTED_JACOBIAN_DOF_COUNT = 23
PREDICTED_JACOBIAN_FLAT_DIM = (
    PREDICTED_JACOBIAN_BODY_COUNT
    * PREDICTED_JACOBIAN_COMPONENT_COUNT
    * PREDICTED_JACOBIAN_SPATIAL_DIMS
    * PREDICTED_JACOBIAN_DOF_COUNT
)

MODEL_DOF_NAMES = (
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)

MODEL_31_TO_INDEPENDENT_INDICES = np.asarray(
    (
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 11, 14, 15, 16,
        17, 18, 19, 22, 25, 26, 27,
        28, 29, 30,
    ),
    dtype=np.int64,
)

MODEL_43_TO_INDEPENDENT_INDICES = np.asarray(
    (
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 12, 15, 16, 17,
        18, 19, 20, 24, 27, 28, 29,
        30, 31, 32,
    ),
    dtype=np.int64,
)


def flatten_rotation_matrices(rot: Any, xp=np) -> Any:
    """Flatten (..., 2, 3, 3) rotation bundles to (..., 18)."""
    leading_shape = tuple(rot.shape[:-3])
    return xp.reshape(rot, leading_shape + (-1,))


def unflatten_rotation_matrices(rot_flat: Any, xp=np) -> Any:
    """Restore flattened (..., 18) rotation bundles to (..., 2, 3, 3)."""
    leading_shape = tuple(rot_flat.shape[:-1])
    return xp.reshape(rot_flat, leading_shape + (2, 3, 3))


def unflatten_rotation_residuals(rot_residual_flat: Any, xp=np) -> Any:
    """Restore flattened (..., 6) residual bundles to (..., 2, 3)."""
    leading_shape = tuple(rot_residual_flat.shape[:-1])
    return xp.reshape(
        rot_residual_flat,
        leading_shape + (ROTATION_RESIDUAL_FEET, ROTATION_RESIDUAL_AXIS_DIM),
    )


def _vector_norm(vec: Any, *, xp=np, axis: int = -1, keepdims: bool = False) -> Any:
    squared = xp.sum(xp.square(vec), axis=axis, keepdims=keepdims)
    return xp.sqrt(squared)


def bound_rotation_residual_axis_angles(
    raw_residual: Any,
    *,
    max_residual_deg: float,
    xp=np,
    eps: float = 1e-6,
) -> Any:
    """Bound residual axis-angle vectors so their magnitude saturates smoothly."""
    residual = xp.asarray(raw_residual)
    max_residual_deg = float(max(max_residual_deg, 0.0))
    if max_residual_deg == 0.0:
        return xp.zeros_like(residual)

    max_residual_rad = xp.asarray(np.deg2rad(max_residual_deg), dtype=residual.dtype)
    raw_norm = _vector_norm(residual, xp=xp, axis=-1, keepdims=True)
    bounded_norm = max_residual_rad * xp.tanh(raw_norm / max_residual_rad)
    scale = bounded_norm / xp.maximum(raw_norm, xp.asarray(eps, dtype=residual.dtype))
    return residual * scale


def _skew_symmetric(vec: Any, *, xp=np) -> Any:
    zeros = xp.zeros_like(vec[..., 0])
    x = vec[..., 0]
    y = vec[..., 1]
    z = vec[..., 2]
    return xp.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        axis=-1,
    ).reshape(vec.shape[:-1] + (3, 3))


def axis_angle_to_rotation_matrices(
    axis_angle: Any,
    *,
    xp=np,
    eps: float = 1e-6,
) -> Any:
    """Convert axis-angle vectors (..., 3) to rotation matrices (..., 3, 3)."""
    axis_angle = xp.asarray(axis_angle)
    theta_sq = xp.sum(xp.square(axis_angle), axis=-1, keepdims=True)
    theta = xp.sqrt(theta_sq)
    eps_sq = xp.asarray(eps * eps, dtype=axis_angle.dtype)

    sin_over_theta = xp.where(
        theta_sq > eps_sq,
        xp.sin(theta) / xp.maximum(theta, xp.asarray(eps, dtype=axis_angle.dtype)),
        1.0 - (theta_sq / 6.0) + (theta_sq * theta_sq / 120.0),
    )
    one_minus_cos_over_theta_sq = xp.where(
        theta_sq > eps_sq,
        (1.0 - xp.cos(theta)) / xp.maximum(theta_sq, eps_sq),
        0.5 - (theta_sq / 24.0) + (theta_sq * theta_sq / 720.0),
    )

    skew = _skew_symmetric(axis_angle, xp=xp)
    skew_sq = xp.matmul(skew, skew)
    identity = xp.broadcast_to(xp.eye(3, dtype=axis_angle.dtype), axis_angle.shape[:-1] + (3, 3))
    return identity + sin_over_theta[..., None] * skew + one_minus_cos_over_theta_sq[..., None] * skew_sq


def project_rotation_matrices(rot: Any, xp=np) -> Any:
    """Project arbitrary 3x3 matrices to the nearest proper rotation matrices."""
    leading_shape = tuple(rot.shape[:-2])
    rot_flat = xp.reshape(rot, (-1, 3, 3))
    u, _s, vh = xp.linalg.svd(rot_flat, full_matrices=False)
    det_sign = xp.where(xp.linalg.det(u @ vh) < 0.0, -1.0, 1.0).astype(rot.dtype)
    ones = xp.ones_like(det_sign)
    zeros = xp.zeros_like(det_sign)
    correction = xp.stack(
        [
            xp.stack([ones, zeros, zeros], axis=-1),
            xp.stack([zeros, ones, zeros], axis=-1),
            xp.stack([zeros, zeros, det_sign], axis=-1),
        ],
        axis=-2,
    )
    projected = u @ correction @ vh
    return xp.reshape(projected, leading_shape + (3, 3))


def compose_residual_rotation_predictions(
    rotation_residual_flat: Any,
    rotation_input: Any,
    *,
    xp=np,
) -> Tuple[Any, Any, Any]:
    """Decode a residual rotation head and compose it with the input rotation."""
    rotation_input_phys = project_rotation_matrices(xp.asarray(rotation_input), xp=xp)
    rotation_residual_axis_angle = unflatten_rotation_residuals(rotation_residual_flat, xp=xp)
    rotation_delta = axis_angle_to_rotation_matrices(rotation_residual_axis_angle, xp=xp)
    rotation_pred_phys = xp.matmul(rotation_delta, rotation_input_phys)
    return rotation_residual_axis_angle, rotation_delta, rotation_pred_phys


def split_model_predictions(
    pred: Any,
    qfrc_inverse_output_dim: int = 0,
    rotation_output_dim: int = 0,
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


def _decode_predicted_jacobian_tail(
    jacobian_tail: np.ndarray,
    expected_flat_dim: int = PREDICTED_JACOBIAN_FLAT_DIM,
) -> Optional[Dict[str, np.ndarray]]:
    """Legacy helper for old checkpoints that had a Predicted_Jacobian tail."""
    tail = np.asarray(jacobian_tail, dtype=np.float32)
    if tail.ndim < 2 or tail.shape[-1] < expected_flat_dim:
        return None

    flat_dim = int(expected_flat_dim)
    divisor = (
        PREDICTED_JACOBIAN_BODY_COUNT
        * PREDICTED_JACOBIAN_COMPONENT_COUNT
        * PREDICTED_JACOBIAN_SPATIAL_DIMS
    )
    if flat_dim <= 0 or flat_dim % divisor != 0:
        return None
    dof_count = flat_dim // divisor

    flat = tail[..., :expected_flat_dim]
    per_component = PREDICTED_JACOBIAN_BODY_COUNT * PREDICTED_JACOBIAN_SPATIAL_DIMS * dof_count
    jacp_flat = flat[..., :per_component]
    jacr_flat = flat[..., per_component:per_component * 2]

    prefix_shape = flat.shape[:-1]
    jacp = jacp_flat.reshape(prefix_shape + (PREDICTED_JACOBIAN_BODY_COUNT, PREDICTED_JACOBIAN_SPATIAL_DIMS, dof_count))
    jacr = jacr_flat.reshape(prefix_shape + (PREDICTED_JACOBIAN_BODY_COUNT, PREDICTED_JACOBIAN_SPATIAL_DIMS, dof_count))
    return {
        "flat": flat,
        "jacp": jacp,
        "jacr": jacr,
    }


def _qfrc_inverse_phys_from_scaled(
    qfrc_inverse_scaled: np.ndarray,
    data: Dict[str, np.ndarray],
) -> np.ndarray:
    """Convert qfrc_inverse from Nm / (BW * H) back to physical Nm."""
    qfrc_scaled = np.asarray(qfrc_inverse_scaled, dtype=np.float32)
    norm_factor = data.get("qfrc_inverse_norm_factor")
    if norm_factor is not None:
        return qfrc_scaled * np.asarray(norm_factor, dtype=np.float32)
    if data.get("mass") is not None and data.get("height") is not None:
        return unnormalize_qfrc_inverse_by_bw_height(
            qfrc_scaled,
            data["mass"],
            data["height"],
            xp=np,
        ).astype(np.float32, copy=False)
    return qfrc_scaled


def _decode_residual_prediction(
    residual_pred_z: np.ndarray,
    input_value: np.ndarray,
    normalizer: "Normalizer",
) -> np.ndarray:
    """Decode a residual prediction back into a full signal in input units."""
    residual_pred_z = np.asarray(residual_pred_z, dtype=np.float32)
    input_value = np.asarray(input_value, dtype=np.float32)
    std = np.asarray(normalizer.std, dtype=np.float32)
    return input_value + residual_pred_z * std


# =============================================================================
# Model Architecture (must match training)
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        position = jnp.arange(seq_len)
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = position[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return x + emb[None, :, :]


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True,
                 film_gamma: jnp.ndarray = None, film_beta: jnp.ndarray = None) -> jnp.ndarray:
        # Optional FiLM subject conditioning (mirrors train.py): modulate the normalized
        # features with per-layer (gamma, beta) from the static token. (1 + gamma) so it
        # is near-identity at init. When film_gamma is None the block is unchanged.
        def _film(h):
            if film_gamma is None:
                return h
            return h * (1.0 + film_gamma[:, None, :]) + film_beta[:, None, :]

        residual = x
        x = nn.LayerNorm()(x)
        x = _film(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x, x, deterministic=not train)
        x = residual + attn_out

        residual = x
        x = nn.LayerNorm()(x)
        x = _film(x)
        ff_out = nn.Dense(self.ff_dim)(x)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
        x = residual + ff_out

        return x


class KinematicsToCOPGRFMoments(nn.Module):
    input_dim: int = 54  # Overridden at runtime from loaded feature tensor.
    static_dim: int = 8  # height, mass, gender, patient_size(4), forward_vel
    output_dim: int = 14  # COP:4 + GRF:6 + Moments:2 + Contact:2
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout_rate: float = 0.1
    use_film: bool = False  # Plan 7: per-layer FiLM subject conditioning (default off)

    @nn.compact
    def __call__(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # 1. Input projection
        x = nn.Dense(self.d_model)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        # Positional Encoding
        x = SinusoidalPosEmb(dim=self.d_model)(x)

        # 2. Static Branch
        s = nn.Dense(self.d_model)(static_context)
        s = nn.gelu(s)
        s = nn.LayerNorm()(s)

        # 2b. Optional FiLM conditioning params (per layer, gamma+beta of width d_model).
        film_params = None
        if self.use_film:
            film_params = nn.Dense(self.num_layers * 2 * self.d_model, name="film_mlp")(s)
            film_params = film_params.reshape(s.shape[0], self.num_layers, 2, self.d_model)

        # 3. Prepend Static Token
        s = jnp.expand_dims(s, axis=1)
        x = jnp.concatenate([s, x], axis=1)

        for _layer_idx in range(self.num_layers):
            film_gamma = film_params[:, _layer_idx, 0, :] if film_params is not None else None
            film_beta = film_params[:, _layer_idx, 1, :] if film_params is not None else None
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train, film_gamma=film_gamma, film_beta=film_beta)

        # 4. Remove Static Token
        x = x[:, 1:, :]
        x = nn.LayerNorm()(x)

        # 5. Project to the standard 14 outputs.
        raw_out = nn.Dense(self.output_dim)(x)          # (batch, seq, output_dim)

        # 6. Contact prediction: sigmoid for soft prob
        cop_raw = raw_out[..., COP_SLICE]
        grf_raw = raw_out[..., GRF_SLICE]
        mom_raw = raw_out[..., MOMENTS_SLICE]
        contact_logits = raw_out[..., CONTACT_SLICE]
        contact_prob   = nn.sigmoid(contact_logits)     # soft [0,1] — dims 12-13 of output
        # Note: Hard masking is handled in physical space outside the model.
        out = jnp.concatenate([cop_raw, grf_raw, mom_raw, contact_prob], axis=-1)
        return out


# =============================================================================
# Normalizer
# =============================================================================
class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def unnormalize(self, x):
        return x * self.std + self.mean


def decode_cop_signal_to_length_np(
    cop_signal: np.ndarray,
    grf_ratio: np.ndarray,
    height_m: np.ndarray,
    *,
    use_grf_norm_cop: bool = False,
    contact_probability: Optional[np.ndarray] = None,
    contact_threshold: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Decode COP/height or GRFNorm COP targets back to length units."""
    cop_arr = np.asarray(cop_signal, dtype=np.float32)
    h = np.asarray(height_m, dtype=np.float32)
    if not use_grf_norm_cop:
        return cop_arr * h

    grf_arr = np.asarray(grf_ratio, dtype=np.float32)
    mag_r = np.linalg.norm(grf_arr[..., 0:3], axis=-1, keepdims=True)
    mag_l = np.linalg.norm(grf_arr[..., 3:6], axis=-1, keepdims=True)
    mag_r = np.maximum(mag_r, eps).astype(np.float32, copy=False)
    mag_l = np.maximum(mag_l, eps).astype(np.float32, copy=False)
    decoded = np.concatenate([
        cop_arr[..., 0:2] * h / mag_r,
        cop_arr[..., 2:4] * h / mag_l,
    ], axis=-1).astype(np.float32, copy=False)
    if contact_probability is None:
        return decoded

    contact = np.asarray(contact_probability, dtype=np.float32)
    mask_r = (contact[..., 0:1] >= np.float32(contact_threshold)).astype(np.float32)
    mask_l = (contact[..., 1:2] >= np.float32(contact_threshold)).astype(np.float32)
    return np.concatenate([
        decoded[..., 0:2] * mask_r,
        decoded[..., 2:4] * mask_l,
    ], axis=-1).astype(np.float32, copy=False)


def zero_cop_where_contact_below_threshold_np(
    cop_m: np.ndarray,
    contact_probability: Optional[np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray:
    """Zero each foot's COP where that foot's predicted contact is low."""
    cop = np.asarray(cop_m, dtype=np.float32).copy()
    if contact_probability is None:
        return cop
    contact = np.asarray(contact_probability, dtype=np.float32)
    if cop.ndim != 2 or cop.shape[-1] < 4 or contact.ndim != 2 or contact.shape[-1] < 2:
        return cop
    cop[:, 0:2] *= (contact[:, 0:1] >= threshold).astype(np.float32)
    cop[:, 2:4] *= (contact[:, 1:2] >= threshold).astype(np.float32)
    return cop


def load_input_restriction_bounds(bounds_path: Optional[str], expected_dim: int) -> np.ndarray:
    resolved_path = Path(bounds_path) if bounds_path is not None else DEFAULT_RESTRICT_BOUNDS_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Restriction bounds file not found: {resolved_path}. "
            "Pass --restrict_max_vals_path with a compatible bounds file."
        )
    bounds = np.load(resolved_path)
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(f"Restriction bounds must have shape (2, input_dim), got {bounds.shape}")
    if bounds.shape[1] != expected_dim:
        raise ValueError(
            f"Restriction bounds input_dim mismatch: bounds have {bounds.shape[1]} dims, "
            f"but this model expects {expected_dim} temporal input dims"
        )
    return bounds.astype(np.float32)


def apply_input_restriction(
    input_features: np.ndarray,
    input_normalizer: Normalizer,
    raw_bounds: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    mean = np.asarray(input_normalizer.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(input_normalizer.std, dtype=np.float32).reshape(-1)
    lower_raw = np.asarray(raw_bounds[0], dtype=np.float32).reshape(-1)
    upper_raw = np.asarray(raw_bounds[1], dtype=np.float32).reshape(-1)
    lower_z = (lower_raw - mean) / std
    upper_z = (upper_raw - mean) / std

    input_z = input_normalizer.normalize(input_features).astype(np.float32)
    below_mask = input_z < lower_z
    above_mask = input_z > upper_z
    outside_mask = below_mask | above_mask
    clipped_z = np.clip(input_z, lower_z, upper_z).astype(np.float32)

    frames_outside_mask = np.any(outside_mask, axis=1)
    num_frames = int(input_z.shape[0])
    num_values = int(outside_mask.size)
    frames_outside = int(np.sum(frames_outside_mask))
    values_outside = int(np.sum(outside_mask))

    summary = {
        "enabled": True,
        "num_frames": num_frames,
        "frames_outside": frames_outside,
        "frames_outside_percent": (100.0 * frames_outside / num_frames) if num_frames else 0.0,
        "values_outside": values_outside,
        "values_outside_percent": (100.0 * values_outside / num_values) if num_values else 0.0,
        "per_feature_values_outside": np.sum(outside_mask, axis=0).astype(int).tolist(),
        "lower_z": lower_z.astype(np.float32).tolist(),
        "upper_z": upper_z.astype(np.float32).tolist(),
    }
    return clipped_z, summary


# =============================================================================
# Post-Inference Filtering
# =============================================================================

def apply_butterworth_filter(signal: np.ndarray, cutoff_hz: float = FILTER_CUTOFF_HZ, 
                             fs_hz: float = FILTER_SAMPLING_RATE_HZ, order: int = 4) -> np.ndarray:
    """Apply 4th-order Butterworth lowpass filter using filtfilt.
    
    Args:
        signal: Input signal, shape (time_steps, features) or (time_steps,)
        cutoff_hz: Cutoff frequency in Hz
        fs_hz: Sampling frequency in Hz
        order: Filter order (default: 4)
    
    Returns:
        Filtered signal with same shape as input
    """
    # Compute normalized frequency (Nyquist frequency = fs/2)
    nyquist = fs_hz / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    
    # Design Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Apply filter
    if signal.ndim == 1:
        # 1D signal
        filtered = filtfilt(b, a, signal, padtype='constant')
    else:
        # 2D signal: filter each feature independently
        filtered = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            filtered[:, i] = filtfilt(b, a, signal[:, i], padtype='constant')
    
    return filtered


def apply_butterworth_filter_masked(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    cutoff_hz: float = FILTER_CUTOFF_HZ,
    fs_hz: float = FILTER_SAMPLING_RATE_HZ,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth filtering only on contiguous valid segments."""
    signal_np = np.asarray(signal)
    mask = np.asarray(valid_mask).reshape(-1).astype(bool)
    if signal_np.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Signal/mask length mismatch: {signal_np.shape[0]} vs {mask.shape[0]}"
        )

    filtered = signal_np.copy()
    if not np.any(mask):
        return filtered

    nyquist = fs_hz / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    min_len = 3 * max(len(a), len(b)) + 1

    valid_idx = np.flatnonzero(mask)
    split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
    valid_segments = np.split(valid_idx, split_points)

    for seg_idx in valid_segments:
        if seg_idx.size < min_len:
            continue
        start = int(seg_idx[0])
        end = int(seg_idx[-1]) + 1
        filtered[start:end] = apply_butterworth_filter(
            signal_np[start:end],
            cutoff_hz=cutoff_hz,
            fs_hz=fs_hz,
            order=order,
        )

    return filtered


# =============================================================================
# Data Loading
# =============================================================================

def load_trial_data(
    trial_path: str,
    opencap_val: bool = False,
    input_source: str = "processed",
    use_noised: bool = False,
    use_grf_norm_cop: bool = False,
    use_grf_nofilt: Optional[bool] = None,
    use_os_filtering: bool = False,
    use_recalculated_opensim_id_gt: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """Load all data from a single trial using consolidated data_loader logic."""
    return load_single_trial(
        Path(trial_path),
        opencap_val=opencap_val,
        input_source=input_source,
        use_noised=use_noised,
        use_grf_norm_cop=use_grf_norm_cop,
        use_grf_nofilt=use_grf_nofilt,
        use_os_filtering=use_os_filtering,
        use_recalculated_opensim_id_gt=use_recalculated_opensim_id_gt,
    )

def find_trial(data_dir: str, trial_name: str) -> Optional[Tuple[str, str]]:
    """Find a trial path. Explicit subject/trial requests must match exactly."""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}. Check if NAS is mounted.")
    if not data_dir.is_dir():
        raise ValueError(f"Provided data_dir is not a directory: {data_dir}")

    # 1. Try direct match for explicit subject/trial requests.
    parts = trial_name.split("/")
    if len(parts) == 2:
        subject, trial = parts
        trial_path = data_dir / subject / trial
        patient_path = data_dir / subject / "Patient_MD.json"
        if (
            trial_path.exists()
            and patient_path.exists()
            and (video_processed_dir(trial_path) / "pos_inputs.npy").exists()
        ):
            return str(trial_path), str(patient_path)

        print(
            f"   ⚠️ Requested trial '{trial_name}' was not found exactly under {data_dir}. "
            "Skipping this trial."
        )
        return None

    # 2. Try searching one level deep for partial trial names only.
    # data_dir/subject/trial
    for subject_dir in data_dir.iterdir():
        if not subject_dir.is_dir():
            continue

        trial_dir = subject_dir / parts[-1] if parts else None
        if trial_dir and trial_dir.exists() and (video_processed_dir(trial_dir) / "pos_inputs.npy").exists():
            patient_path = subject_dir / "Patient_MD.json"
            if patient_path.exists():
                return str(trial_dir), str(patient_path)

    # 3. Last resort: recursive search using the Video/ProcessedData layout.
    print(f"   ⚠️ Trial '{trial_name}' not found by direct match in {data_dir}. Searching...")
    for pd_path in data_dir.glob("**/Video/ProcessedData"):
        trial_path = pd_path.parent.parent
        if trial_name.lower() in str(trial_path).lower():
            # Find Patient_MD by going up until we find it
            curr = trial_path
            for _ in range(3): # Check up to 3 levels up
                patient_path = curr / "Patient_MD.json"
                if patient_path.exists():
                    return str(trial_path), str(patient_path)
                curr = curr.parent
    
    print(f"   ⚠️ Could not find trial '{trial_name}' in {data_dir}. Skipping this trial.")
    return None


OPENSIM_ID_COLUMN_MAP: Dict[int, str] = {
    3: "pelvis_tilt_moment",
    4: "pelvis_list_moment",
    5: "pelvis_rotation_moment",
    6: "hip_flexion_r_moment",
    7: "hip_adduction_r_moment",
    8: "hip_rotation_r_moment",
    11: "knee_angle_r_moment",
    14: "ankle_angle_r_moment",
    15: "subtalar_angle_r_moment",
    17: "hip_flexion_l_moment",
    18: "hip_adduction_l_moment",
    19: "hip_rotation_l_moment",
    22: "knee_angle_l_moment",
    25: "ankle_angle_l_moment",
    26: "subtalar_angle_l_moment",
    28: "lumbar_extension_moment",
    29: "lumbar_bending_moment",
    30: "lumbar_rotation_moment",
}


def _load_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_motion_aligned_trim_reference(processed_dir: Path) -> Optional[Dict[str, Any]]:
    """Mirror the ProcessData trim reference used to align motion-space signals."""
    meta_path = processed_dir / "Trial_Processing_Information.json"
    payload = _load_json_dict(meta_path)
    if not payload:
        return None

    bounds = payload.get("core_trim_bounds_motion_aligned")
    pretrim_n_frames = payload.get("core_trim_pretrim_n_frames")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None
    if not isinstance(pretrim_n_frames, (int, np.integer)):
        return None

    start_idx = int(bounds[0])
    end_idx = int(bounds[1])
    pretrim_n_frames = int(pretrim_n_frames)
    if not (0 <= start_idx <= end_idx <= pretrim_n_frames):
        return None

    trim_ref: Dict[str, Any] = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "pretrim_n_frames": pretrim_n_frames,
    }

    ds_bounds = payload.get("ds_edge_trim_bounds")
    ds_pretrim_n_frames = payload.get("ds_edge_trim_n_frames_before")
    if (
        isinstance(ds_bounds, (list, tuple))
        and len(ds_bounds) == 2
        and isinstance(ds_pretrim_n_frames, (int, np.integer))
    ):
        ds_start = int(ds_bounds[0])
        ds_end = int(ds_bounds[1])
        ds_pretrim_n_frames = int(ds_pretrim_n_frames)
        if 0 <= ds_start <= ds_end <= ds_pretrim_n_frames:
            trim_ref.update(
                {
                    "ds_start_idx": ds_start,
                    "ds_end_idx": ds_end,
                    "ds_pretrim_n_frames": ds_pretrim_n_frames,
                    "ds_edge_trim_bounds": [ds_start, ds_end],
                    "ds_edge_trim_n_frames_before": ds_pretrim_n_frames,
                }
            )

    return trim_ref


def _apply_ds_edge_trim_if_needed(
    values: np.ndarray,
    info: Mapping[str, Any],
    target_len: Optional[int] = None,
) -> np.ndarray:
    """Apply the final double-support edge trim recorded after core GRF trimming."""
    bounds = info.get("ds_edge_trim_bounds")
    pretrim_n_frames = info.get("ds_edge_trim_n_frames_before")
    if not (
        isinstance(bounds, (list, tuple))
        and len(bounds) == 2
        and isinstance(pretrim_n_frames, (int, np.integer))
    ):
        return values

    start = int(bounds[0])
    end = int(bounds[1])
    pretrim_n_frames = int(pretrim_n_frames)
    if not (0 <= start <= end <= pretrim_n_frames):
        return values
    if len(values) != pretrim_n_frames:
        return values

    trimmed = values[start:end]
    if target_len is not None and len(trimmed) != int(target_len):
        return values
    return trimmed


def _load_opensim_sto(sto_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load an OpenSim .sto table while preserving column names."""
    with sto_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    start_row = 0
    header_info: Dict[str, str] = {}
    for idx, line in enumerate(lines):
        if "=" in line:
            parts = line.strip().split("=")
            if len(parts) == 2:
                header_info[parts[0]] = parts[1]
        if "endheader" in line.lower():
            start_row = idx + 1
            break

    if start_row < len(lines) and "coordinates" in lines[start_row].lower():
        start_row += 1

    df = pd.read_csv(sto_path, sep=r"\s+", skiprows=start_row)
    if len(df.columns) < 2:
        df = pd.read_csv(sto_path, sep="\t", skiprows=start_row)
    return df, header_info


def _resample_series_matrix(
    values: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """Resample a `(T, D)` matrix onto a target timebase with linear interpolation."""
    values = np.asarray(values, dtype=np.float32)
    source_time = np.asarray(source_time, dtype=np.float64).reshape(-1)
    target_time = np.asarray(target_time, dtype=np.float64).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D values for resampling, got shape {values.shape}")
    if len(source_time) != len(values):
        raise ValueError(
            f"Source time length mismatch for resampling: time={len(source_time)} values={len(values)}"
        )
    if len(values) == 0:
        return values.copy()
    if np.array_equal(source_time, target_time):
        return values.copy()
    return np.stack(
        [
            np.interp(
                target_time,
                source_time,
                values[:, col],
                left=float(values[0, col]),
                right=float(values[-1, col]),
            )
            for col in range(values.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def _find_opensim_id_sto_file(trial_path: Path) -> Optional[Path]:
    """Find the most likely OpenSim inverse-dynamics STO file for a trial."""
    candidates = list(trial_path.glob("*.sto"))
    motion_dir = trial_path / "Motion"
    if motion_dir.exists():
        candidates.extend(motion_dir.glob("*.sto"))

    filtered = [path for path in candidates if "_ik" not in path.name.lower()]
    if not filtered:
        return None

    def _score(path: Path) -> Tuple[int, int, str]:
        name = path.name.lower()
        return (
            0 if "inverse" in name or "id" in name else 1,
            0 if path.parent.name == "Motion" else 1,
            name,
        )

    return sorted(filtered, key=_score)[0]


def load_aligned_opensim_id_ground_truth(
    trial_path: str | Path,
    *,
    target_len: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Load an OpenSim ID `.sto`, align it to ProcessData trim bounds, and return MJX-indexed torques."""
    trial_root = Path(trial_path)
    sto_path = _find_opensim_id_sto_file(trial_root)
    if sto_path is None:
        return None

    try:
        sto_df, _header = _load_opensim_sto(sto_path)
    except Exception as exc:
        print(f"   ⚠️ Failed to load OpenSim ID STO ({sto_path.name}): {exc}")
        return None

    if "time" not in sto_df.columns:
        print(f"   ⚠️ OpenSim ID STO is missing a time column: {sto_path}")
        return None

    source_time = np.asarray(sto_df["time"], dtype=np.float64).reshape(-1)
    n_dofs = len(get_dof_names())
    opensim_id = np.full((len(sto_df), n_dofs), np.nan, dtype=np.float32)
    available_mask = np.zeros((n_dofs,), dtype=bool)
    available_columns: List[str] = []
    for dof_idx, sto_col in OPENSIM_ID_COLUMN_MAP.items():
        if sto_col not in sto_df.columns:
            continue
        opensim_id[:, dof_idx] = np.asarray(sto_df[sto_col], dtype=np.float32)
        available_mask[dof_idx] = True
        available_columns.append(sto_col)

    if not np.any(available_mask):
        print(f"   ⚠️ OpenSim ID STO did not contain any expected torque columns: {sto_path.name}")
        return None

    motion_time_path = trial_root / "Motion" / "Time.npy"
    if motion_time_path.exists():
        try:
            motion_time = np.asarray(np.load(motion_time_path), dtype=np.float64).reshape(-1)
            if len(motion_time) > 1 and len(motion_time) != len(opensim_id):
                opensim_id = _resample_series_matrix(opensim_id, source_time, motion_time)
                source_time = motion_time
        except Exception as exc:
            print(f"   ⚠️ Could not align OpenSim ID STO to Motion/Time.npy: {exc}")

    trim_ref = _load_motion_aligned_trim_reference(video_processed_dir(trial_root))
    if trim_ref is not None and len(opensim_id) == int(trim_ref["pretrim_n_frames"]):
        opensim_id = opensim_id[int(trim_ref["start_idx"]):int(trim_ref["end_idx"])]
        opensim_id = _apply_ds_edge_trim_if_needed(
            opensim_id,
            trim_ref,
            target_len=target_len,
        )
    elif trim_ref is not None and len(opensim_id) != int(trim_ref["pretrim_n_frames"]):
        print(
            "   ⚠️ OpenSim ID STO length does not match ProcessedData trim reference "
            f"({len(opensim_id)} vs {trim_ref['pretrim_n_frames']}); keeping aligned series without trim sync."
        )

    if target_len is not None and len(opensim_id) != int(target_len):
        target_len = int(target_len)
        if len(opensim_id) <= 1:
            opensim_id = np.repeat(opensim_id[:1], target_len, axis=0)
        else:
            opensim_id = _resample_series_matrix(
                opensim_id,
                np.linspace(0.0, 1.0, len(opensim_id), dtype=np.float64),
                np.linspace(0.0, 1.0, target_len, dtype=np.float64),
            )

    return {
        "id": opensim_id.astype(np.float32),
        "available_mask": available_mask,
        "source_path": str(sto_path),
        "available_columns": available_columns,
    }


# Knee-coupling/secondary DOFs have no OpenSim ID column; the recalculated GT zeros them,
# so they are marked unavailable (the MJX reference fills them in blended metrics/plots).
_RECALC_KNEE_COUPLING_CHANNELS = (9, 10, 12, 13, 20, 21, 23, 24)


def load_recalculated_opensim_id_ground_truth(
    trial_path: str | Path,
    *,
    target_len: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Load ``MoCap/OpenSim_ID_recalculated.npy`` (already MJX-indexed, 31 channels) as a GT
    bundle mirroring ``load_aligned_opensim_id_ground_truth``. This is the recalculated
    OpenSim ID computed from MoCap kinematics + cleaned forces (the primary GT)."""
    npy_path = mocap_processed_dir(Path(trial_path)) / "OpenSim_ID_recalculated.npy"
    if not npy_path.exists():
        return None
    try:
        opensim_id = np.asarray(np.load(npy_path), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️ Failed to load recalculated OpenSim ID ({npy_path.name}): {exc}")
        return None
    if opensim_id.ndim != 2:
        print(f"   ⚠️ Recalculated OpenSim ID has unexpected shape {opensim_id.shape}: {npy_path}")
        return None

    if target_len is not None and len(opensim_id) != int(target_len):
        target_len = int(target_len)
        if len(opensim_id) <= 1:
            opensim_id = np.repeat(opensim_id[:1], target_len, axis=0)
        else:
            opensim_id = _resample_series_matrix(
                opensim_id,
                np.linspace(0.0, 1.0, len(opensim_id), dtype=np.float64),
                np.linspace(0.0, 1.0, target_len, dtype=np.float64),
            )

    n_dofs = opensim_id.shape[1]
    available_mask = np.ones((n_dofs,), dtype=bool)
    for ch in _RECALC_KNEE_COUPLING_CHANNELS:
        if ch < n_dofs:
            available_mask[ch] = False
    return {
        "id": opensim_id.astype(np.float32),
        "available_mask": available_mask,
        "source_path": str(npy_path),
        "available_columns": [],
    }


def load_mjx_id_reference_ground_truth(
    trial_path: str | Path,
    *,
    target_len: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load original MoCap ``ID_GT_MJX.npy`` for optional plot reference when recalc OpenSim GT is active."""
    mjx_path = mocap_processed_dir(Path(trial_path)) / "ID_GT_MJX.npy"
    if not mjx_path.exists():
        return None
    try:
        mjx_id = np.asarray(np.load(mjx_path), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️ Failed to load MJX ID reference ({mjx_path.name}): {exc}")
        return None
    if target_len is not None and len(mjx_id) != int(target_len):
        target_len = int(target_len)
        if len(mjx_id) <= 1:
            mjx_id = np.repeat(mjx_id[:1], target_len, axis=0)
        else:
            mjx_id = _resample_series_matrix(
                mjx_id,
                np.linspace(0.0, 1.0, len(mjx_id), dtype=np.float64),
                np.linspace(0.0, 1.0, target_len, dtype=np.float64),
            )
    return mjx_id.astype(np.float32, copy=False)


def load_opensim_id_ground_truth_bundle(
    trial_path: str | Path,
    *,
    target_len: Optional[int] = None,
    use_recalculated: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load OpenSim ID GT for plotting/metrics (recalculated ``.npy`` or legacy aligned ``.sto``)."""
    if use_recalculated:
        bundle = load_recalculated_opensim_id_ground_truth(trial_path, target_len=target_len)
        if bundle is not None:
            return bundle
        print(
            "   ⚠️ use_recalculated_opensim_id_gt requested but "
            "MoCap/OpenSim_ID_recalculated.npy was not found."
        )
        return None
    return load_aligned_opensim_id_ground_truth(trial_path, target_len=target_len)


def _coerce_bool(value, default: bool = False) -> bool:
    """Robust bool parsing for config values that may be bool/int/str."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return bool(value)


def _parse_optional_bool_arg(value):
    if value is None:
        return True
    return _coerce_bool(value, default=True)


def _build_train_style_temporal_input(
    data: Dict[str, np.ndarray],
    include_pelvis_euler: bool,
    include_auxiliary_denoising_inputs: bool,
    include_ankle_heights: bool = True,
    include_jacobian_input: bool = True,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Build temporal inputs exactly as train/data_loader do."""
    pos_input = select_pos_input_columns(
        data["pos"],
        include_pelvis_euler=include_pelvis_euler,
    )
    blocks: List[Tuple[str, int]] = [
        ("pelvis_rot", int(data["pelvis_rot"].shape[-1])),
        ("pos", int(pos_input.shape[-1])),
        ("vel", int(data["vel"].shape[-1])),
        ("com_r", int(data["com_r"].shape[-1])),
        ("com_l", int(data["com_l"].shape[-1])),
        ("com_accel", int(data["com_accel"].shape[-1])),
    ]
    parts = [
        data["pelvis_rot"],
        pos_input,
        data["vel"],
        data["com_r"],
        data["com_l"],
        data["com_accel"],
    ]

    if include_ankle_heights:
        parts.append(data["ankle_heights"])
        blocks.append(("ankle_heights", int(data["ankle_heights"].shape[-1])))
    if include_jacobian_input:
        jacobian_input = flatten_jacobian_components(data["jacp"], data["jacr"])
        parts.append(jacobian_input)
        blocks.append(("jacobian_input", int(jacobian_input.shape[-1])))

    # Geometry context is always included in training.
    parts.extend([
        data["foot_progression_angle"],
        data["calcn_to_floor_angle"],
    ])
    blocks.extend([
        ("foot_progression_angle", int(data["foot_progression_angle"].shape[-1])),
        ("calcn_to_floor_angle", int(data["calcn_to_floor_angle"].shape[-1])),
    ])

    if include_auxiliary_denoising_inputs:
        qfrc_input = np.asarray(data["qfrc_inverse"], dtype=np.float32)
        parts.append(qfrc_input)
        blocks.append(("qfrc_inverse_input", int(qfrc_input.shape[-1])))
        rot_input = flatten_rotation_matrices(data["rot_w_to_ga"])
        parts.append(rot_input)
        blocks.append(("rot_w_to_ga_input_flat", int(rot_input.shape[-1])))

    return np.concatenate(parts, axis=1), blocks


def _resolve_train_style_inputs(
    data: Dict[str, np.ndarray],
    requested_include_pelvis_euler: bool,
    expected_input_dim: int,
) -> Tuple[np.ndarray, bool, str, List[Tuple[str, int]], Dict[str, int]]:
    """Resolve train-style input layout deterministically from checkpoint dim."""
    req_include = _coerce_bool(requested_include_pelvis_euler, default=True)

    layout_entries = []
    diagnostics: Dict[str, int] = {
        "contactBoolean_dim_not_used": int(data["contactBoolean"].shape[-1]),
    }

    for include_pelvis_flag in (True, False):
        for include_aux_flag in (True, False):
            for include_ankle_heights_flag in (True, False):
                for include_jacobian_flag in (True, False):
                    features, blocks = _build_train_style_temporal_input(
                        data,
                        include_pelvis_euler=include_pelvis_flag,
                        include_auxiliary_denoising_inputs=include_aux_flag,
                        include_ankle_heights=include_ankle_heights_flag,
                        include_jacobian_input=include_jacobian_flag,
                    )
                    label = (
                        "train_direct"
                        + ("" if include_pelvis_flag else "_no_pelvis_euler")
                        + ("_with_ankle_heights" if include_ankle_heights_flag else "_legacy_no_ankle_heights")
                        + ("_with_jacobian" if include_jacobian_flag else "_legacy_no_jacobian")
                        + ("_with_aux_inputs" if include_aux_flag else "_legacy_inputs")
                    )
                    entry = {
                        "features": features,
                        "blocks": blocks,
                        "dim": int(features.shape[-1]),
                        "include_pelvis": include_pelvis_flag,
                        "include_aux": include_aux_flag,
                        "include_ankle_heights": include_ankle_heights_flag,
                        "include_jacobian_input": include_jacobian_flag,
                        "label": label,
                    }
                    layout_entries.append(entry)
                    diagnostics[f"{label}_dim"] = entry["dim"]

    ordered_entries = sorted(
        layout_entries,
        key=lambda entry: (
            0 if not entry["include_aux"] else 1,
            0 if entry.get("include_ankle_heights", False) else 1,
            0 if entry.get("include_jacobian_input", False) else 1,
            0 if entry["include_pelvis"] == req_include else 1,
        ),
    )

    for entry in ordered_entries:
        if entry["dim"] == expected_input_dim:
            return (
                entry["features"],
                entry["include_pelvis"],
                entry["label"],
                entry["blocks"],
                diagnostics,
            )

    for entry in ordered_entries:
        if (
            entry["include_pelvis"] == req_include
        ):
            return (
                entry["features"],
                entry["include_pelvis"],
                f"{entry['label']}(dim_mismatch)",
                entry["blocks"],
                diagnostics,
            )

    fallback_entry = ordered_entries[0]
    return (
        fallback_entry["features"],
        fallback_entry["include_pelvis"],
        f"{fallback_entry['label']}(dim_mismatch)",
        fallback_entry["blocks"],
        diagnostics,
    )


def _build_inference_window_starts(seq_len: int, window_size: int, stride: int) -> List[int]:
    """Match training-time fixed windows, but ensure the tail is covered."""
    if seq_len <= 0:
        return []
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    if seq_len <= window_size:
        return [0]

    starts = list(range(0, seq_len - window_size + 1, stride))
    tail_start = seq_len - window_size
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def _predict_with_train_style_windows(
    predict_fn,
    params,
    input_features_z: np.ndarray,
    static_context_z: np.ndarray,
    window_size: int,
    stride: int,
    output_dim: int,
    prediction_margin_frames: int,
    max_windows_per_batch: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Run inference on fixed windows to match training-time sequence length, then
    average overlapping center-valid predictions back onto the trial timeline.

    Only frames inside each window's supervision-valid center region are kept.
    Frames outside that region do not contribute to the stitched prediction.
    """
    seq_len = int(input_features_z.shape[0])
    validate_prediction_margin(window_size, prediction_margin_frames)
    starts = _build_inference_window_starts(seq_len, window_size, stride)
    if not starts:
        raise ValueError("Cannot build inference windows for an empty sequence")

    kept_pred_sum = np.zeros((seq_len, output_dim), dtype=np.float32)
    kept_pred_count = np.zeros((seq_len, 1), dtype=np.float32)

    for chunk_start in range(0, len(starts), max_windows_per_batch):
        chunk_starts = starts[chunk_start:chunk_start + max_windows_per_batch]
        window_batch = []
        for start in chunk_starts:
            end = start + window_size
            window = input_features_z[start:end]
            if window.shape[0] < window_size:
                pad_len = window_size - window.shape[0]
                window = np.pad(window, ((0, pad_len), (0, 0)), mode="edge")
            window_batch.append(window.astype(np.float32, copy=False))

        x_batch = jnp.array(np.stack(window_batch, axis=0))
        static_batch = jnp.array(
            np.repeat(static_context_z[np.newaxis, :], len(chunk_starts), axis=0)
        )
        chunk_pred = np.asarray(predict_fn(params, x_batch, static_batch), dtype=np.float32)

        for pred_window, start in zip(chunk_pred, chunk_starts):
            end = min(start + window_size, seq_len)
            valid_len = end - start
            kept_mask_window = build_window_supervision_mask(
                window_size=window_size,
                window_start_idx=start,
                trial_length=seq_len,
                prediction_margin_frames=prediction_margin_frames,
            )[:valid_len].astype(np.float32)
            kept_pred_sum[start:end] += pred_window[:valid_len] * kept_mask_window
            kept_pred_count[start:end] += kept_mask_window

    kept_valid_mask = (kept_pred_count[:, 0] > 0.0)
    kept_predictions = np.zeros((seq_len, output_dim), dtype=np.float32)
    if np.any(kept_valid_mask):
        kept_predictions[kept_valid_mask] = (
            kept_pred_sum[kept_valid_mask]
            / np.maximum(kept_pred_count[kept_valid_mask], 1.0)
        )

    return kept_predictions, kept_predictions.copy(), kept_valid_mask, {
        "num_windows": len(starts),
        "window_size": int(window_size),
        "stride": int(stride),
        "prediction_margin_frames": int(prediction_margin_frames),
        "evaluation_frame_count": int(np.sum(kept_valid_mask)),
    }


def _normalize_evaluation_mask(mask: Optional[np.ndarray], seq_len: int) -> np.ndarray:
    """Return a 1D boolean evaluation mask with the requested length."""
    if mask is None:
        return np.ones((seq_len,), dtype=bool)
    mask_np = np.asarray(mask).reshape(-1)
    if mask_np.shape[0] != seq_len:
        raise ValueError(
            f"Evaluation mask length mismatch: expected {seq_len}, got {mask_np.shape[0]}"
        )
    return mask_np.astype(bool)


def _mask_prediction_dict_for_display(
    predictions: Dict[str, np.ndarray],
    evaluation_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Mask non-evaluated frames so displayed curves only show center-valid averages."""
    mask = _normalize_evaluation_mask(evaluation_mask, len(next(iter(predictions.values()))))
    masked_predictions: Dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        if key.startswith("_"):
            masked_predictions[key] = value
            continue
        value_np = np.asarray(value)
        if value_np.ndim >= 1 and value_np.shape[0] == mask.shape[0]:
            masked_value = value_np.astype(np.float32, copy=True)
            masked_value[~mask] = np.nan
            masked_predictions[key] = masked_value
        else:
            masked_predictions[key] = value
    return masked_predictions


def _masked_rmse(pred: np.ndarray, target: np.ndarray, evaluation_mask: np.ndarray) -> float:
    """Compute RMSE over valid evaluation frames only."""
    mask = _normalize_evaluation_mask(evaluation_mask, np.asarray(pred).shape[0])
    if not np.any(mask):
        return float("nan")
    diff = np.asarray(pred)[mask] - np.asarray(target)[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def _masked_mae(pred: np.ndarray, target: np.ndarray, evaluation_mask: np.ndarray) -> float:
    """Compute MAE over valid evaluation frames only."""
    mask = _normalize_evaluation_mask(evaluation_mask, np.asarray(pred).shape[0])
    if not np.any(mask):
        return float("nan")
    diff = np.asarray(pred)[mask] - np.asarray(target)[mask]
    return float(np.mean(np.abs(diff)))


def _masked_mean_diff(pred: np.ndarray, target: np.ndarray, evaluation_mask: np.ndarray) -> np.ndarray:
    """Compute per-channel mean signed error over valid frames only."""
    mask = _normalize_evaluation_mask(evaluation_mask, np.asarray(pred).shape[0])
    if not np.any(mask):
        return np.full(np.asarray(pred).shape[1:], np.nan, dtype=np.float64)
    diff = np.asarray(pred)[mask] - np.asarray(target)[mask]
    return np.mean(diff, axis=0)


def _masked_rmse_per_channel(pred: np.ndarray, target: np.ndarray, evaluation_mask: np.ndarray) -> np.ndarray:
    """Compute per-channel RMSE over valid frames only."""
    mask = _normalize_evaluation_mask(evaluation_mask, np.asarray(pred).shape[0])
    if not np.any(mask):
        return np.full(np.asarray(pred).shape[1:], np.nan, dtype=np.float64)
    diff = np.asarray(pred)[mask] - np.asarray(target)[mask]
    return np.sqrt(np.mean(diff ** 2, axis=0))


def _masked_tensor_rmse(lhs: Any, rhs: Any, evaluation_mask: np.ndarray) -> float:
    lhs_np = np.asarray(lhs, dtype=np.float32)
    rhs_np = np.asarray(rhs, dtype=np.float32)
    mask = _normalize_evaluation_mask(evaluation_mask, lhs_np.shape[0])
    if rhs_np.shape != lhs_np.shape:
        raise ValueError(f"Tensor shape mismatch for RMSE: {lhs_np.shape} vs {rhs_np.shape}")
    if not np.any(mask):
        return float("nan")
    diff = lhs_np[mask] - rhs_np[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def _masked_tensor_mae(lhs: Any, rhs: Any, evaluation_mask: np.ndarray) -> float:
    lhs_np = np.asarray(lhs, dtype=np.float32)
    rhs_np = np.asarray(rhs, dtype=np.float32)
    mask = _normalize_evaluation_mask(evaluation_mask, lhs_np.shape[0])
    if rhs_np.shape != lhs_np.shape:
        raise ValueError(f"Tensor shape mismatch for MAE: {lhs_np.shape} vs {rhs_np.shape}")
    if not np.any(mask):
        return float("nan")
    diff = np.abs(lhs_np[mask] - rhs_np[mask])
    return float(np.mean(diff))


def _resolve_qfrc_inverse_gt_reference(ground_truth: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    if ground_truth.get("qfrc_inverse_mocap") is not None:
        return np.asarray(ground_truth["qfrc_inverse_mocap"], dtype=np.float32), "qfrc_inverse_mocap"
    if ground_truth.get("qfrc_inverse_processed") is not None:
        return np.asarray(ground_truth["qfrc_inverse_processed"], dtype=np.float32), "qfrc_inverse_processed"
    if ground_truth.get("qfrc_inverse") is not None:
        return np.asarray(ground_truth["qfrc_inverse"], dtype=np.float32), "qfrc_inverse"
    return None, "missing"


def _compute_qfrc_inverse_rmse_metrics(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    evaluation_mask: np.ndarray,
    norm_factor: float,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    gt_ref, gt_source = _resolve_qfrc_inverse_gt_reference(ground_truth)
    metrics["qfrc_inverse_gt_source"] = gt_source
    pred_qfrc_inverse = predictions.get("qfrc_inverse")
    processed_ref = ground_truth.get("qfrc_inverse_processed")
    mocap_ref = ground_truth.get("qfrc_inverse_mocap")

    def _rmse(ref_value: Optional[np.ndarray]) -> float:
        if ref_value is None or pred_qfrc_inverse is None:
            return float("nan")
        return float(
            _masked_rmse(
                np.asarray(pred_qfrc_inverse, dtype=np.float32),
                np.asarray(ref_value, dtype=np.float32),
                evaluation_mask,
            )
        )

    def _rmse_between(lhs: Optional[np.ndarray], rhs: Optional[np.ndarray]) -> float:
        if lhs is None or rhs is None:
            return float("nan")
        return float(
            _masked_rmse(
                np.asarray(lhs, dtype=np.float32),
                np.asarray(rhs, dtype=np.float32),
                evaluation_mask,
            )
        )

    pred_vs_processed_rmse = _rmse(processed_ref)
    pred_vs_mocap_rmse = _rmse(mocap_ref)
    processed_vs_mocap_rmse = _rmse_between(processed_ref, mocap_ref)

    metrics["qfrc_inverse_processed_available"] = bool(processed_ref is not None)
    metrics["qfrc_inverse_mocap_available"] = bool(mocap_ref is not None)
    metrics["qfrc_inverse_pred_vs_processed_rmse"] = pred_vs_processed_rmse
    metrics["qfrc_inverse_pred_vs_processed_rmse_bwh"] = float((pred_vs_processed_rmse / norm_factor) * 100.0)
    metrics["qfrc_inverse_pred_vs_mocap_rmse"] = pred_vs_mocap_rmse
    metrics["qfrc_inverse_pred_vs_mocap_rmse_bwh"] = float((pred_vs_mocap_rmse / norm_factor) * 100.0)
    metrics["qfrc_inverse_processed_vs_mocap_rmse"] = processed_vs_mocap_rmse
    metrics["qfrc_inverse_processed_vs_mocap_rmse_bwh"] = float((processed_vs_mocap_rmse / norm_factor) * 100.0)

    if gt_ref is not None:
        processed_rmse = _rmse_between(processed_ref, gt_ref)
        predicted_rmse = _rmse(gt_ref)
        metrics["qfrc_inverse_processed_vs_gt_rmse"] = processed_rmse
        metrics["qfrc_inverse_pred_vs_gt_rmse"] = predicted_rmse
        metrics["qfrc_inverse_processed_vs_gt_rmse_bwh"] = float((processed_rmse / norm_factor) * 100.0)
        metrics["qfrc_inverse_pred_vs_gt_rmse_bwh"] = float((predicted_rmse / norm_factor) * 100.0)
        metrics["qfrc_inverse_pred_minus_processed_rmse"] = float(predicted_rmse - processed_rmse)
        metrics["qfrc_inverse_processed_minus_pred_rmse"] = float(processed_rmse - predicted_rmse)
        metrics["qfrc_inverse_pred_minus_processed_rmse_bwh"] = float(((predicted_rmse - processed_rmse) / norm_factor) * 100.0)
        metrics["qfrc_inverse_processed_minus_pred_rmse_bwh"] = float(((processed_rmse - predicted_rmse) / norm_factor) * 100.0)
    return metrics


def _build_rotation_comparison_stats(
    predicted_rot: np.ndarray,
    processed_rot: np.ndarray,
    gt_rot: np.ndarray,
    evaluation_mask: np.ndarray,
) -> Dict[str, Any]:
    mask = _normalize_evaluation_mask(evaluation_mask, len(predicted_rot))

    def _angle_stats(rot_a: np.ndarray, rot_b: np.ndarray) -> Dict[str, float]:
        rot_a_np = np.asarray(rot_a, dtype=np.float64)
        rot_b_np = np.asarray(rot_b, dtype=np.float64)
        flat_a = rot_a_np.reshape((-1, 3, 3))
        flat_b = rot_b_np.reshape((-1, 3, 3))

        def _project(flat_rot: np.ndarray) -> np.ndarray:
            projected = np.empty_like(flat_rot)
            for idx, mat in enumerate(flat_rot):
                u, _, vh = np.linalg.svd(mat, full_matrices=False)
                proj = u @ vh
                if np.linalg.det(proj) < 0.0:
                    u[:, -1] *= -1.0
                    proj = u @ vh
                projected[idx] = proj
            return projected

        rot_a_proj = _project(flat_a).reshape(rot_a_np.shape)
        rot_b_proj = _project(flat_b).reshape(rot_b_np.shape)
        rot_err = rot_a_proj @ np.swapaxes(rot_b_proj, -1, -2)
        trace = np.sum(np.diagonal(rot_err, axis1=-2, axis2=-1), axis=-1)
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        sin_terms = np.stack(
            [
                rot_err[..., 2, 1] - rot_err[..., 1, 2],
                rot_err[..., 0, 2] - rot_err[..., 2, 0],
                rot_err[..., 1, 0] - rot_err[..., 0, 1],
            ],
            axis=-1,
        )
        angle_deg = np.degrees(np.arctan2(0.5 * np.linalg.norm(sin_terms, axis=-1), cos_theta))
        masked_angles = angle_deg[mask]
        if masked_angles.size == 0:
            return {
                "overall_mean_deg": float("nan"),
                "overall_rmse_deg": float("nan"),
                "right_mean_deg": float("nan"),
                "right_rmse_deg": float("nan"),
                "left_mean_deg": float("nan"),
                "left_rmse_deg": float("nan"),
            }
        return {
            "overall_mean_deg": float(np.mean(masked_angles)),
            "overall_rmse_deg": float(np.sqrt(np.mean(masked_angles ** 2))),
            "right_mean_deg": float(np.mean(masked_angles[:, 0])),
            "right_rmse_deg": float(np.sqrt(np.mean(masked_angles[:, 0] ** 2))),
            "left_mean_deg": float(np.mean(masked_angles[:, 1])),
            "left_rmse_deg": float(np.sqrt(np.mean(masked_angles[:, 1] ** 2))),
        }

    pred_summary = _angle_stats(predicted_rot, gt_rot)
    processed_summary = _angle_stats(processed_rot, gt_rot)
    return {
        "predicted_vs_mocap": pred_summary,
        "processed_vs_mocap": processed_summary,
        "improvement_pred_minus_processed": {key: float(pred_summary[key] - processed_summary[key]) for key in pred_summary},
    }


def _build_jacobian_comparison_stats(
    predicted_jacp: np.ndarray,
    predicted_jacr: np.ndarray,
    processed_jacp: np.ndarray,
    processed_jacr: np.ndarray,
    gt_jacp: np.ndarray,
    gt_jacr: np.ndarray,
    evaluation_mask: np.ndarray,
) -> Dict[str, Any]:
    def _component_stats(candidate: np.ndarray, gt_value: np.ndarray) -> Dict[str, float]:
        return {
            "overall_rmse": _masked_tensor_rmse(candidate, gt_value, evaluation_mask),
            "overall_mae": _masked_tensor_mae(candidate, gt_value, evaluation_mask),
            "right_rmse": _masked_tensor_rmse(candidate[:, 0], gt_value[:, 0], evaluation_mask),
            "right_mae": _masked_tensor_mae(candidate[:, 0], gt_value[:, 0], evaluation_mask),
            "left_rmse": _masked_tensor_rmse(candidate[:, 1], gt_value[:, 1], evaluation_mask),
            "left_mae": _masked_tensor_mae(candidate[:, 1], gt_value[:, 1], evaluation_mask),
        }

    predicted_combined = np.concatenate([np.asarray(predicted_jacp, dtype=np.float32).reshape(len(predicted_jacp), -1), np.asarray(predicted_jacr, dtype=np.float32).reshape(len(predicted_jacr), -1)], axis=1)
    processed_combined = np.concatenate([np.asarray(processed_jacp, dtype=np.float32).reshape(len(processed_jacp), -1), np.asarray(processed_jacr, dtype=np.float32).reshape(len(processed_jacr), -1)], axis=1)
    gt_combined = np.concatenate([np.asarray(gt_jacp, dtype=np.float32).reshape(len(gt_jacp), -1), np.asarray(gt_jacr, dtype=np.float32).reshape(len(gt_jacr), -1)], axis=1)
    stats = {
        "predicted_vs_mocap": {
            "jacp": _component_stats(np.asarray(predicted_jacp, dtype=np.float32), np.asarray(gt_jacp, dtype=np.float32)),
            "jacr": _component_stats(np.asarray(predicted_jacr, dtype=np.float32), np.asarray(gt_jacr, dtype=np.float32)),
            "combined": {
                "overall_rmse": _masked_tensor_rmse(predicted_combined, gt_combined, evaluation_mask),
                "overall_mae": _masked_tensor_mae(predicted_combined, gt_combined, evaluation_mask),
            },
        },
        "processed_vs_mocap": {
            "jacp": _component_stats(np.asarray(processed_jacp, dtype=np.float32), np.asarray(gt_jacp, dtype=np.float32)),
            "jacr": _component_stats(np.asarray(processed_jacr, dtype=np.float32), np.asarray(gt_jacr, dtype=np.float32)),
            "combined": {
                "overall_rmse": _masked_tensor_rmse(processed_combined, gt_combined, evaluation_mask),
                "overall_mae": _masked_tensor_mae(processed_combined, gt_combined, evaluation_mask),
            },
        },
    }
    stats["improvement_pred_minus_processed"] = {
        "jacp": {key: float(stats["predicted_vs_mocap"]["jacp"][key] - stats["processed_vs_mocap"]["jacp"][key]) for key in stats["predicted_vs_mocap"]["jacp"]},
        "jacr": {key: float(stats["predicted_vs_mocap"]["jacr"][key] - stats["processed_vs_mocap"]["jacr"][key]) for key in stats["predicted_vs_mocap"]["jacr"]},
        "combined": {key: float(stats["predicted_vs_mocap"]["combined"][key] - stats["processed_vs_mocap"]["combined"][key]) for key in stats["predicted_vs_mocap"]["combined"]},
    }
    return stats


def _compute_tau_from_candidate_geometry(
    *,
    cop_xz: np.ndarray,
    ankle_heights: np.ndarray,
    grf_world: np.ndarray,
    moments_world: np.ndarray,
    rot_w_to_ga: np.ndarray,
    jacp: np.ndarray,
    jacr: np.ndarray,
) -> np.ndarray:
    rot_ga_to_w = np.swapaxes(np.asarray(rot_w_to_ga, dtype=np.float32), -1, -2)
    cop_xz_np = np.asarray(cop_xz, dtype=np.float32)
    ankle_heights_np = np.asarray(ankle_heights, dtype=np.float32)
    grf_world_np = np.asarray(grf_world, dtype=np.float32)
    moments_world_np = np.asarray(moments_world, dtype=np.float32)
    jacp_np = np.asarray(jacp, dtype=np.float32)
    jacr_np = np.asarray(jacr, dtype=np.float32)

    cop_world = np.zeros((len(cop_xz_np), 2, 3), dtype=np.float32)
    cop_world[:, 0] = np.einsum(
        "tij,tj->ti",
        rot_ga_to_w[:, 0],
        np.column_stack([cop_xz_np[:, 0], ankle_heights_np[:, 0], cop_xz_np[:, 1]]),
    )
    cop_world[:, 1] = np.einsum(
        "tij,tj->ti",
        rot_ga_to_w[:, 1],
        np.column_stack([cop_xz_np[:, 2], ankle_heights_np[:, 1], cop_xz_np[:, 3]]),
    )

    f_r = grf_world_np[:, 0:3]
    f_l = grf_world_np[:, 3:6]
    m_r = moments_world_np[:, 0:3]
    m_l = moments_world_np[:, 3:6]
    m_total_r = m_r + np.cross(cop_world[:, 0], f_r)
    m_total_l = m_l + np.cross(cop_world[:, 1], f_l)
    tau_r = np.einsum("tji,tj->ti", jacp_np[:, 0], f_r) + np.einsum("tji,tj->ti", jacr_np[:, 0], m_total_r)
    tau_l = np.einsum("tji,tj->ti", jacp_np[:, 1], f_l) + np.einsum("tji,tj->ti", jacr_np[:, 1], m_total_l)
    return (tau_r + tau_l).astype(np.float32)


def _compute_predicted_knee_to_cop_vectors_np(
    *,
    cop_pred_xz: np.ndarray,
    ankle_pos_global: np.ndarray,
    knee_pos_global: np.ndarray,
    rot_w_to_ga: np.ndarray,
) -> np.ndarray:
    """Build [R_xyz, L_xyz] knee->predicted-COP vectors in world coordinates."""
    cop_pred_xz = np.asarray(cop_pred_xz, dtype=np.float32)
    ankle_pos_global = np.asarray(ankle_pos_global, dtype=np.float32)
    knee_pos_global = np.asarray(knee_pos_global, dtype=np.float32)
    rot_ga_to_w = np.swapaxes(np.asarray(rot_w_to_ga, dtype=np.float32), -1, -2)
    zeros = np.zeros((len(cop_pred_xz),), dtype=np.float32)
    cop_r_ga = np.stack([cop_pred_xz[:, 0], zeros, cop_pred_xz[:, 1]], axis=1)
    cop_l_ga = np.stack([cop_pred_xz[:, 2], zeros, cop_pred_xz[:, 3]], axis=1)
    cop_r_world_rel = np.einsum("tij,tj->ti", rot_ga_to_w[:, 0], cop_r_ga)
    cop_l_world_rel = np.einsum("tij,tj->ti", rot_ga_to_w[:, 1], cop_l_ga)
    vec_r = (ankle_pos_global[:, 0] - knee_pos_global[:, 0]) + cop_r_world_rel
    vec_l = (ankle_pos_global[:, 1] - knee_pos_global[:, 1]) + cop_l_world_rel
    return np.concatenate([vec_r, vec_l], axis=1).astype(np.float32, copy=False)


def _build_knee_flexion_torque_comparison_stats(
    *,
    predictions: Dict[str, Any],
    data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    left_stance_mask: np.ndarray,
) -> Dict[str, Any]:
    dof_names = get_dof_names()
    try:
        knee_idx = dof_names.index("knee_angle_l")
    except ValueError:
        return {"available": False, "reason": "knee_angle_l_missing"}

    qfrc_inverse_ref = ground_truth.get("qfrc_inverse_mocap", ground_truth.get("qfrc_inverse"))
    id_full_gt, _, _ = (None, None, None)
    pred_full_id, gt_full_id, _ = compute_full_id_curves(predictions, ground_truth)
    id_full_gt = gt_full_id
    if qfrc_inverse_ref is None or id_full_gt is None:
        return {"available": False, "reason": "missing_qfrc_inverse_or_id_full"}

    predicted_rot = predictions.get("rot_w_to_ga")
    predicted_jacp = predictions.get("predicted_jacobian_jacp")
    predicted_jacr = predictions.get("predicted_jacobian_jacr")
    if predicted_rot is None or predicted_jacp is None or predicted_jacr is None:
        return {"available": False, "reason": "missing_predicted_rotation_or_jacobian"}

    gt_rot = np.asarray(ground_truth["rot_w_to_ga"], dtype=np.float32)
    gt_jacp = np.asarray(data["gt_jacp"], dtype=np.float32)
    gt_jacr = np.asarray(data["gt_jacr"], dtype=np.float32)
    processed_rot = np.asarray(data["rot_w_to_ga"], dtype=np.float32)
    processed_jacp = np.asarray(data["jacp"], dtype=np.float32)
    processed_jacr = np.asarray(data["jacr"], dtype=np.float32)
    qfrc_inverse_ref_np = np.asarray(qfrc_inverse_ref, dtype=np.float32)
    id_full_gt_np = np.asarray(id_full_gt, dtype=np.float32)

    common_kwargs = {
        "cop_xz": np.asarray(data["cop_raw"], dtype=np.float32),
        "ankle_heights": np.asarray(data["ankle_heights"], dtype=np.float32),
        "grf_world": np.asarray(data["grf_raw"], dtype=np.float32),
        "moments_world": np.asarray(data["moments_raw"], dtype=np.float32),
    }
    tau_with_pred_jac = _compute_tau_from_candidate_geometry(
        rot_w_to_ga=gt_rot,
        jacp=np.asarray(predicted_jacp, dtype=np.float32),
        jacr=np.asarray(predicted_jacr, dtype=np.float32),
        **common_kwargs,
    )
    tau_with_processed_jac = _compute_tau_from_candidate_geometry(
        rot_w_to_ga=gt_rot,
        jacp=processed_jacp,
        jacr=processed_jacr,
        **common_kwargs,
    )
    tau_with_pred_rot = _compute_tau_from_candidate_geometry(
        rot_w_to_ga=np.asarray(predicted_rot, dtype=np.float32),
        jacp=gt_jacp,
        jacr=gt_jacr,
        **common_kwargs,
    )
    tau_with_processed_rot = _compute_tau_from_candidate_geometry(
        rot_w_to_ga=processed_rot,
        jacp=gt_jacp,
        jacr=gt_jacr,
        **common_kwargs,
    )

    target_width = _resolve_full_id_target_width(
        qfrc_inverse_ref_np,
        id_full_gt_np,
        tau_with_pred_jac,
        tau_with_processed_jac,
        tau_with_pred_rot,
        tau_with_processed_rot,
    )
    qfrc_inverse_ref_np = _coerce_full_id_width(
        qfrc_inverse_ref_np,
        target_width,
        label="qfrc_inverse_reference",
    )
    id_full_gt_np = _coerce_full_id_width(
        id_full_gt_np,
        target_width,
        label="full_id_gt",
        fill_value=np.nan,
    )
    tau_with_pred_jac = _coerce_full_id_width(
        tau_with_pred_jac,
        target_width,
        label="tau_with_pred_jac",
    )
    tau_with_processed_jac = _coerce_full_id_width(
        tau_with_processed_jac,
        target_width,
        label="tau_with_processed_jac",
    )
    tau_with_pred_rot = _coerce_full_id_width(
        tau_with_pred_rot,
        target_width,
        label="tau_with_pred_rot",
    )
    tau_with_processed_rot = _coerce_full_id_width(
        tau_with_processed_rot,
        target_width,
        label="tau_with_processed_rot",
    )

    id_with_pred_jac = qfrc_inverse_ref_np - tau_with_pred_jac
    id_with_processed_jac = qfrc_inverse_ref_np - tau_with_processed_jac
    id_with_pred_rot = qfrc_inverse_ref_np - tau_with_pred_rot
    id_with_processed_rot = qfrc_inverse_ref_np - tau_with_processed_rot

    mask = _normalize_evaluation_mask(left_stance_mask, len(id_full_gt_np))
    gt_knee = id_full_gt_np[:, knee_idx : knee_idx + 1]

    pred_jac_mae = _masked_mae(id_with_pred_jac[:, knee_idx : knee_idx + 1], gt_knee, mask)
    processed_jac_mae = _masked_mae(id_with_processed_jac[:, knee_idx : knee_idx + 1], gt_knee, mask)
    pred_rot_mae = _masked_mae(id_with_pred_rot[:, knee_idx : knee_idx + 1], gt_knee, mask)
    processed_rot_mae = _masked_mae(id_with_processed_rot[:, knee_idx : knee_idx + 1], gt_knee, mask)

    return {
        "available": True,
        "signal": "id_full",
        "dof_name": "knee_angle_l",
        "jacobian": {
            "predicted_mae": float(pred_jac_mae),
            "processed_mae": float(processed_jac_mae),
            "pred_minus_processed_mae": float(pred_jac_mae - processed_jac_mae),
            "processed_minus_pred_mae": float(processed_jac_mae - pred_jac_mae),
        },
        "rotation": {
            "predicted_mae": float(pred_rot_mae),
            "processed_mae": float(processed_rot_mae),
            "pred_minus_processed_mae": float(pred_rot_mae - processed_rot_mae),
            "processed_minus_pred_mae": float(processed_rot_mae - pred_rot_mae),
        },
    }


def create_rotation_jacobian_comparison_plot(
    trial_name: str,
    rotation_stats: Dict[str, Any],
    jacobian_stats: Dict[str, Any],
    knee_torque_stats: Optional[Dict[str, Any]] = None,
    *,
    save_path: Optional[str] = None,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Rotation Geodesic RMSE vs MoCap",
            "Rotation Geodesic Mean vs MoCap",
            "Jacobian RMSE vs MoCap",
            "Jacobian MAE vs MoCap",
            "Left Knee Flexion Torque MAE with Rotation Swap",
            "Left Knee Flexion Torque MAE with Jacobian Swap",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )
    labels_rot = ["Overall", "Right", "Left"]
    fig.add_trace(go.Bar(name="Predicted", x=labels_rot, y=[rotation_stats["predicted_vs_mocap"]["overall_rmse_deg"], rotation_stats["predicted_vs_mocap"]["right_rmse_deg"], rotation_stats["predicted_vs_mocap"]["left_rmse_deg"]], marker_color="#E94F37"), row=1, col=1)
    fig.add_trace(go.Bar(name="ProcessedData", x=labels_rot, y=[rotation_stats["processed_vs_mocap"]["overall_rmse_deg"], rotation_stats["processed_vs_mocap"]["right_rmse_deg"], rotation_stats["processed_vs_mocap"]["left_rmse_deg"]], marker_color="#2E86AB"), row=1, col=1)
    fig.add_trace(go.Bar(name="Predicted", x=labels_rot, y=[rotation_stats["predicted_vs_mocap"]["overall_mean_deg"], rotation_stats["predicted_vs_mocap"]["right_mean_deg"], rotation_stats["predicted_vs_mocap"]["left_mean_deg"]], marker_color="#E94F37", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name="ProcessedData", x=labels_rot, y=[rotation_stats["processed_vs_mocap"]["overall_mean_deg"], rotation_stats["processed_vs_mocap"]["right_mean_deg"], rotation_stats["processed_vs_mocap"]["left_mean_deg"]], marker_color="#2E86AB", showlegend=False), row=1, col=2)
    jac_labels = ["jacp overall", "jacp right", "jacp left", "jacr overall", "jacr right", "jacr left", "combined overall"]
    fig.add_trace(go.Bar(name="Predicted", x=jac_labels, y=[jacobian_stats["predicted_vs_mocap"]["jacp"]["overall_rmse"], jacobian_stats["predicted_vs_mocap"]["jacp"]["right_rmse"], jacobian_stats["predicted_vs_mocap"]["jacp"]["left_rmse"], jacobian_stats["predicted_vs_mocap"]["jacr"]["overall_rmse"], jacobian_stats["predicted_vs_mocap"]["jacr"]["right_rmse"], jacobian_stats["predicted_vs_mocap"]["jacr"]["left_rmse"], jacobian_stats["predicted_vs_mocap"]["combined"]["overall_rmse"]], marker_color="#E94F37", showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(name="ProcessedData", x=jac_labels, y=[jacobian_stats["processed_vs_mocap"]["jacp"]["overall_rmse"], jacobian_stats["processed_vs_mocap"]["jacp"]["right_rmse"], jacobian_stats["processed_vs_mocap"]["jacp"]["left_rmse"], jacobian_stats["processed_vs_mocap"]["jacr"]["overall_rmse"], jacobian_stats["processed_vs_mocap"]["jacr"]["right_rmse"], jacobian_stats["processed_vs_mocap"]["jacr"]["left_rmse"], jacobian_stats["processed_vs_mocap"]["combined"]["overall_rmse"]], marker_color="#2E86AB", showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(name="Predicted", x=jac_labels, y=[jacobian_stats["predicted_vs_mocap"]["jacp"]["overall_mae"], jacobian_stats["predicted_vs_mocap"]["jacp"]["right_mae"], jacobian_stats["predicted_vs_mocap"]["jacp"]["left_mae"], jacobian_stats["predicted_vs_mocap"]["jacr"]["overall_mae"], jacobian_stats["predicted_vs_mocap"]["jacr"]["right_mae"], jacobian_stats["predicted_vs_mocap"]["jacr"]["left_mae"], jacobian_stats["predicted_vs_mocap"]["combined"]["overall_mae"]], marker_color="#E94F37", showlegend=False), row=2, col=2)
    fig.add_trace(go.Bar(name="ProcessedData", x=jac_labels, y=[jacobian_stats["processed_vs_mocap"]["jacp"]["overall_mae"], jacobian_stats["processed_vs_mocap"]["jacp"]["right_mae"], jacobian_stats["processed_vs_mocap"]["jacp"]["left_mae"], jacobian_stats["processed_vs_mocap"]["jacr"]["overall_mae"], jacobian_stats["processed_vs_mocap"]["jacr"]["right_mae"], jacobian_stats["processed_vs_mocap"]["jacr"]["left_mae"], jacobian_stats["processed_vs_mocap"]["combined"]["overall_mae"]], marker_color="#2E86AB", showlegend=False), row=2, col=2)
    if knee_torque_stats is not None and knee_torque_stats.get("available"):
        fig.add_trace(go.Bar(name="Predicted", x=["Left knee MAE"], y=[knee_torque_stats["rotation"]["predicted_mae"]], marker_color="#E94F37", showlegend=False), row=3, col=1)
        fig.add_trace(go.Bar(name="ProcessedData", x=["Left knee MAE"], y=[knee_torque_stats["rotation"]["processed_mae"]], marker_color="#2E86AB", showlegend=False), row=3, col=1)
        fig.add_trace(go.Bar(name="Predicted", x=["Left knee MAE"], y=[knee_torque_stats["jacobian"]["predicted_mae"]], marker_color="#E94F37", showlegend=False), row=3, col=2)
        fig.add_trace(go.Bar(name="ProcessedData", x=["Left knee MAE"], y=[knee_torque_stats["jacobian"]["processed_mae"]], marker_color="#2E86AB", showlegend=False), row=3, col=2)
        fig.add_annotation(x=0.22, y=0.03, xref="paper", yref="paper", text=f"Pred - Processed: {knee_torque_stats['rotation']['pred_minus_processed_mae']:.4f}", showarrow=False)
        fig.add_annotation(x=0.78, y=0.03, xref="paper", yref="paper", text=f"Pred - Processed: {knee_torque_stats['jacobian']['pred_minus_processed_mae']:.4f}", showarrow=False)
    fig.update_layout(title=dict(text="<b>Rotation/Jacobian Accuracy vs MoCap Ground Truth</b><br>" f"<span style='font-size:12px'>{trial_name}</span>", x=0.5), barmode="group", height=1320, width=1800, template="plotly_white", margin=dict(t=100, b=80, l=60, r=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_yaxes(title_text="Degrees", row=1, col=1)
    fig.update_yaxes(title_text="Degrees", row=1, col=2)
    fig.update_yaxes(title_text="RMSE", row=2, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=2)
    fig.update_yaxes(title_text="MAE (Nm)", row=3, col=1)
    fig.update_yaxes(title_text="MAE (Nm)", row=3, col=2)
    fig.update_xaxes(tickangle=-20, row=2, col=1)
    fig.update_xaxes(tickangle=-20, row=2, col=2)
    if save_path:
        fig.write_html(save_path)
        print(f"💾 Saved rotation/jacobian comparison plot to: {save_path}")
    return fig


def create_rotation_jacobian_summary_dashboard(
    all_metrics: List[Dict[str, Any]],
    *,
    save_path: Optional[str] = None,
) -> Optional[go.Figure]:
    """Create a cross-trial summary of rotation/Jacobian/qfrc_inverse MoCap comparisons."""

    trial_names = [m.get("trial_name", f"Trial {idx + 1}") for idx, m in enumerate(all_metrics)]

    def _as_finite_float(value: Any) -> float:
        try:
            cast = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return cast if np.isfinite(cast) else float("nan")

    def _collect_box_series(extractor):
        series = []
        for trial_name, metrics in zip(trial_names, all_metrics):
            value = _as_finite_float(extractor(metrics))
            if np.isfinite(value):
                series.append((trial_name, value))
        return series

    plot_specs = [
        (
            "Rotation Geodesic RMSE vs MoCap",
            [
                (
                    "Predicted overall",
                    "#E94F37",
                    lambda m: m.get("rotation_vs_mocap_comparison", {}).get("predicted_vs_mocap", {}).get("overall_rmse_deg"),
                ),
                (
                    "Processed overall",
                    "#2E86AB",
                    lambda m: m.get("rotation_vs_mocap_comparison", {}).get("processed_vs_mocap", {}).get("overall_rmse_deg"),
                ),
                (
                    "Predicted minus processed",
                    "#6A4C93",
                    lambda m: m.get("rotation_vs_mocap_comparison", {}).get("improvement_pred_minus_processed", {}).get("overall_rmse_deg"),
                ),
            ],
            "Degrees",
        ),
        (
            "Rotation Geodesic Mean vs MoCap",
            [
                (
                    "Predicted overall",
                    "#E94F37",
                    lambda m: m.get("rotation_vs_mocap_comparison", {}).get("predicted_vs_mocap", {}).get("overall_mean_deg"),
                ),
                (
                    "Processed overall",
                    "#2E86AB",
                    lambda m: m.get("rotation_vs_mocap_comparison", {}).get("processed_vs_mocap", {}).get("overall_mean_deg"),
                ),
                (
                    "Predicted minus processed",
                    "#6A4C93",
                    lambda m: m.get("rotation_vs_mocap_comparison", {}).get("improvement_pred_minus_processed", {}).get("overall_mean_deg"),
                ),
            ],
            "Degrees",
        ),
        (
            "Jacobian RMSE vs MoCap",
            [
                (
                    "Predicted jacp",
                    "#E94F37",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("predicted_vs_mocap", {}).get("jacp", {}).get("overall_rmse"),
                ),
                (
                    "Processed jacp",
                    "#2E86AB",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("processed_vs_mocap", {}).get("jacp", {}).get("overall_rmse"),
                ),
                (
                    "Predicted jacr",
                    "#F4A261",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("predicted_vs_mocap", {}).get("jacr", {}).get("overall_rmse"),
                ),
                (
                    "Processed jacr",
                    "#2A9D8F",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("processed_vs_mocap", {}).get("jacr", {}).get("overall_rmse"),
                ),
                (
                    "Predicted combined",
                    "#D62828",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("predicted_vs_mocap", {}).get("combined", {}).get("overall_rmse"),
                ),
                (
                    "Processed combined",
                    "#457B9D",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("processed_vs_mocap", {}).get("combined", {}).get("overall_rmse"),
                ),
                (
                    "Combined delta",
                    "#6A4C93",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("improvement_pred_minus_processed", {}).get("combined", {}).get("overall_rmse"),
                ),
            ],
            "RMSE",
        ),
        (
            "Jacobian MAE vs MoCap",
            [
                (
                    "Predicted jacp",
                    "#E94F37",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("predicted_vs_mocap", {}).get("jacp", {}).get("overall_mae"),
                ),
                (
                    "Processed jacp",
                    "#2E86AB",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("processed_vs_mocap", {}).get("jacp", {}).get("overall_mae"),
                ),
                (
                    "Predicted jacr",
                    "#F4A261",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("predicted_vs_mocap", {}).get("jacr", {}).get("overall_mae"),
                ),
                (
                    "Processed jacr",
                    "#2A9D8F",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("processed_vs_mocap", {}).get("jacr", {}).get("overall_mae"),
                ),
                (
                    "Predicted combined",
                    "#D62828",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("predicted_vs_mocap", {}).get("combined", {}).get("overall_mae"),
                ),
                (
                    "Processed combined",
                    "#457B9D",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("processed_vs_mocap", {}).get("combined", {}).get("overall_mae"),
                ),
                (
                    "Combined delta",
                    "#6A4C93",
                    lambda m: m.get("jacobian_vs_mocap_comparison", {}).get("improvement_pred_minus_processed", {}).get("combined", {}).get("overall_mae"),
                ),
            ],
            "MAE",
        ),
        (
            "qfrc_inverse RMSE vs MoCap",
            [
                (
                    "Predicted vs MoCap",
                    "#E94F37",
                    lambda m: m.get("qfrc_inverse_pred_vs_mocap_rmse"),
                ),
                (
                    "Processed vs MoCap",
                    "#2E86AB",
                    lambda m: m.get("qfrc_inverse_processed_vs_mocap_rmse"),
                ),
                (
                    "Predicted minus processed",
                    "#6A4C93",
                    lambda m: (
                        _as_finite_float(m.get("qfrc_inverse_pred_vs_mocap_rmse"))
                        - _as_finite_float(m.get("qfrc_inverse_processed_vs_mocap_rmse"))
                    ),
                ),
            ],
            "RMSE (Nm)",
        ),
        (
            "Left Knee Flexion Torque MAE with Rotation Swap",
            [
                (
                    "Predicted rotation",
                    "#E94F37",
                    lambda m: m.get("knee_flexion_torque_vs_mocap_comparison", {}).get("rotation", {}).get("predicted_mae"),
                ),
                (
                    "Processed rotation",
                    "#2E86AB",
                    lambda m: m.get("knee_flexion_torque_vs_mocap_comparison", {}).get("rotation", {}).get("processed_mae"),
                ),
                (
                    "Rotation delta",
                    "#6A4C93",
                    lambda m: m.get("knee_flexion_torque_vs_mocap_comparison", {}).get("rotation", {}).get("pred_minus_processed_mae"),
                ),
            ],
            "MAE (Nm)",
        ),
        (
            "Left Knee Flexion Torque MAE with Jacobian Swap",
            [
                (
                    "Predicted Jacobian",
                    "#E94F37",
                    lambda m: m.get("knee_flexion_torque_vs_mocap_comparison", {}).get("jacobian", {}).get("predicted_mae"),
                ),
                (
                    "Processed Jacobian",
                    "#2E86AB",
                    lambda m: m.get("knee_flexion_torque_vs_mocap_comparison", {}).get("jacobian", {}).get("processed_mae"),
                ),
                (
                    "Jacobian delta",
                    "#6A4C93",
                    lambda m: m.get("knee_flexion_torque_vs_mocap_comparison", {}).get("jacobian", {}).get("pred_minus_processed_mae"),
                ),
            ],
            "MAE (Nm)",
        ),
    ]

    available_specs = []
    for title, series_specs, yaxis_title in plot_specs:
        series_data = []
        for label, color, extractor in series_specs:
            entries = _collect_box_series(extractor)
            if entries:
                series_data.append((label, color, entries))
        if series_data:
            available_specs.append((title, yaxis_title, series_data))

    if not available_specs:
        return None

    n_plots = len(available_specs)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = int(np.ceil(n_plots / n_cols))
    subplot_titles = [title for title, _, _ in available_specs]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    annotations = []
    for plot_idx, (title, yaxis_title, series_data) in enumerate(available_specs):
        row = plot_idx // n_cols + 1
        col = plot_idx % n_cols + 1
        annotation_lines = []
        for label, color, entries in series_data:
            y_values = [value for _, value in entries]
            customdata = [[trial_name] for trial_name, _ in entries]
            fig.add_trace(
                go.Box(
                    y=y_values,
                    name=label,
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0.0,
                    marker=dict(color=color, size=8, opacity=0.78),
                    line=dict(color=color),
                    fillcolor="rgba(0,0,0,0)",
                    customdata=customdata,
                    hovertemplate=(
                        f"{label}<br>Trial: %{{customdata[0]}}<br>Value: %{{y:.4f}}<extra>{title}</extra>"
                    ),
                    showlegend=(plot_idx == 0),
                ),
                row=row,
                col=col,
            )
            annotation_lines.append(f"{label}: mean={np.mean(y_values):.4f} (n={len(y_values)})")

        fig.update_yaxes(title_text=yaxis_title, row=row, col=col)
        annotations.append(
            dict(
                text="<br>".join(annotation_lines),
                xref=f"x{plot_idx + 1} domain" if plot_idx > 0 else "x domain",
                yref=f"y{plot_idx + 1} domain" if plot_idx > 0 else "y domain",
                x=1.0,
                y=1.0,
                xanchor="right",
                yanchor="top",
                align="right",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                font=dict(size=11),
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>OpenCapSubjects Rotation/Jacobian/qfrc Summary vs MoCap</b><br><span style='font-size:12px'>Box-and-whisker distributions with individual trial points</span>",
            x=0.5,
        ),
        template="plotly_white",
        height=max(540, 430 * n_rows),
        width=1800 if n_cols == 2 else 950,
        boxmode="group",
        margin=dict(t=110, b=80, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        annotations=list(fig.layout.annotations) + annotations,
    )
    fig.update_xaxes(tickangle=-18)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.write_html(save_path)
        print(f"💾 Saved rotation/jacobian summary dashboard to: {save_path}")
    return fig


def _get_side_stance_mask(
    grf_raw: np.ndarray,
    evaluation_mask: Optional[np.ndarray],
    side: str,
    threshold: float = LEFT_STANCE_THRESHOLD_N,
) -> np.ndarray:
    """Return valid evaluation frames where the requested foot is in stance."""
    grf_raw = np.asarray(grf_raw)
    valid_eval_mask = (
        _normalize_evaluation_mask(evaluation_mask, len(grf_raw))
        if evaluation_mask is not None
        else np.ones(len(grf_raw), dtype=bool)
    )
    if grf_raw.ndim != 2 or grf_raw.shape[1] <= 5:
        raise ValueError(
            "Expected grf_raw with shape (frames, >=6) for stance detection."
        )
    vertical_idx = 2 if str(side).strip().lower() == "right" else 5
    return (np.abs(grf_raw[:, vertical_idx]) > float(threshold)) & valid_eval_mask


def _get_stance_phases_from_mask(stance_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a boolean stance mask into contiguous stance intervals."""
    stance_mask = np.asarray(stance_mask, dtype=bool).reshape(-1)
    diff = np.diff(stance_mask.astype(int), prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(ends) < len(starts):
        ends = np.append(ends, len(stance_mask))
    if len(starts) > len(ends):
        starts = starts[:len(ends)]

    phases: List[Tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if int(end - start) > 5:
            phases.append((int(start), int(end)))
    return phases


def _dual_threshold_stance_intervals(
    grf_raw: np.ndarray,
    side: str,
    body_weight_n: float,
    evaluation_mask: Optional[np.ndarray] = None,
    low_threshold_n: float = COMPLETE_STANCE_THRESHOLD_N,
    core_bw_ratio: float = COMPLETE_STANCE_CORE_BW_RATIO,
) -> List[Tuple[int, int]]:
    """Hysteresis stance extraction from vertical GRF.

    A region only counts as a stance if its vertical GRF rises above a high,
    noise-immune core threshold (``core_bw_ratio`` * body weight) for more than
    a few frames. Each confirmed core is then expanded outward -- earlier for
    the start, later for the end -- to the first frames where vertical GRF drops
    below ``low_threshold_n`` (5 N), capturing the low-force heel-strike and
    toe-off tails. Overlapping expanded regions (two cores sharing one low-force
    span) are merged. Returns ``(start, end_exclusive)`` intervals.
    """
    grf_raw = np.asarray(grf_raw)
    n = len(grf_raw)
    if grf_raw.ndim != 2 or grf_raw.shape[1] <= 5:
        raise ValueError(
            "Expected grf_raw with shape (frames, >=6) for stance detection."
        )
    valid = (
        _normalize_evaluation_mask(evaluation_mask, n)
        if evaluation_mask is not None
        else np.ones(n, dtype=bool)
    )
    vertical_idx = 2 if str(side).strip().lower() == "right" else 5
    vertical_force = np.abs(np.asarray(grf_raw, dtype=np.float64)[:, vertical_idx])

    high_threshold_n = float(core_bw_ratio) * float(body_weight_n)
    low_mask = (vertical_force > float(low_threshold_n)) & valid
    high_mask = (vertical_force > high_threshold_n) & valid

    # Confident stance cores (>5-frame runs above the BW core threshold; brief
    # high blips are dropped by _get_stance_phases_from_mask's length guard).
    cores = _get_stance_phases_from_mask(high_mask)

    expanded: List[Tuple[int, int]] = []
    for core_start, core_end in cores:
        start = core_start
        while start > 0 and low_mask[start - 1]:
            start -= 1
        end = core_end
        while end < n and low_mask[end]:
            end += 1
        expanded.append((int(start), int(end)))

    # Merge overlapping / adjacent intervals.
    expanded.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in expanded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _find_complete_stance_phases(
    grf_raw: np.ndarray,
    side: str,
    body_weight_n: float,
    evaluation_mask: Optional[np.ndarray] = None,
    low_threshold_n: float = COMPLETE_STANCE_THRESHOLD_N,
    core_bw_ratio: float = COMPLETE_STANCE_CORE_BW_RATIO,
) -> List[Dict[str, float]]:
    """Return complete stance phases for one side via dual-threshold detection.

    A stance is gated solely by the presence of a >=``core_bw_ratio`` * BW
    vertical-GRF core; its boundaries are walked out to ``low_threshold_n``
    (5 N). No separate min-length or impulse filter is applied.
    """
    intervals = _dual_threshold_stance_intervals(
        grf_raw,
        side=side,
        body_weight_n=body_weight_n,
        evaluation_mask=evaluation_mask,
        low_threshold_n=low_threshold_n,
        core_bw_ratio=core_bw_ratio,
    )
    vertical_idx = 2 if str(side).strip().lower() == "right" else 5
    vertical_force = np.abs(np.asarray(grf_raw, dtype=np.float64)[:, vertical_idx])
    core_threshold_n = float(core_bw_ratio) * float(body_weight_n)

    phases: List[Dict[str, float]] = []
    for start, end in intervals:
        length = int(end - start)
        seg = vertical_force[start:end]
        phases.append(
            {
                "start_frame": int(start),
                "end_frame_exclusive": int(end),
                "length_frames": length,
                "integral_n_frames": float(np.sum(seg)),
                "required_integral_n_frames": 0.0,
                "mean_vgrf_n": float(np.mean(seg)) if length else 0.0,
                "required_mean_vgrf_n": 0.0,
                "core_threshold_n": core_threshold_n,
                "core_bw_ratio": float(core_bw_ratio),
            }
        )
    return phases


def _load_patient_age_gender(patient_path: str | Path) -> Dict[str, Any]:
    payload = _load_json_dict(Path(patient_path)) or {}
    age = payload.get("Age", payload.get("age"))
    sex = payload.get("BiologicalSex", payload.get("Gender", payload.get("Sex")))
    mass = payload.get("Mass_kg", payload.get("Mass", payload.get("mass")))
    height = payload.get("Height_m", payload.get("Height", payload.get("height")))
    age_years: Optional[float] = None
    mass_kg: Optional[float] = None
    height_m: Optional[float] = None
    try:
        if age is not None and str(age).strip() != "":
            age_years = float(age)
    except Exception:
        age_years = None
    try:
        if mass is not None and str(mass).strip() != "":
            mass_kg = float(mass)
    except Exception:
        mass_kg = None
    try:
        if height is not None and str(height).strip() != "":
            height_m = float(height)
    except Exception:
        height_m = None
    return {
        "age_years": age_years,
        "gender": str(sex) if sex is not None else None,
        "mass_kg": mass_kg,
        "height_m": height_m,
    }


def _serialize_float_list_with_none(values: np.ndarray) -> List[Optional[float]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out: List[Optional[float]] = []
    for v in arr:
        out.append(float(v) if np.isfinite(v) else None)
    return out


def _interpolate_sparse_power_to_101(
    percent_sparse: np.ndarray,
    power_sparse: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate sparse stance power samples to 0..100%; keep unavailable regions as NaN."""
    percent_sparse = np.asarray(percent_sparse, dtype=np.float64).reshape(-1)
    power_sparse = np.asarray(power_sparse, dtype=np.float64).reshape(-1)
    x_new = np.linspace(0.0, 100.0, 101, dtype=np.float64)
    y_new = np.full_like(x_new, np.nan, dtype=np.float64)

    if percent_sparse.size == 0:
        return x_new, y_new

    uniq_x, uniq_idx = np.unique(percent_sparse, return_index=True)
    uniq_y = power_sparse[uniq_idx]
    finite = np.isfinite(uniq_x) & np.isfinite(uniq_y)
    uniq_x = uniq_x[finite]
    uniq_y = uniq_y[finite]
    if uniq_x.size < 2:
        if uniq_x.size == 1:
            nearest_idx = int(np.argmin(np.abs(x_new - uniq_x[0])))
            y_new[nearest_idx] = float(uniq_y[0])
        return x_new, y_new

    interpolated = np.interp(x_new, uniq_x, uniq_y, left=np.nan, right=np.nan)
    valid_band = (x_new >= float(np.min(uniq_x))) & (x_new <= float(np.max(uniq_x)))
    y_new[valid_band] = interpolated[valid_band]
    return x_new, y_new


def build_complete_stance_peak_report(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    patient_path: str | Path,
    evaluation_mask: Optional[np.ndarray] = None,
    trial_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Build peak ankle/knee metrics over complete stance phases for each side."""
    tau_pred = predictions.get("tau_grf")
    selected_gt, selected_label, _mjx_gt, _opensim_gt, _opensim_mask = resolve_full_id_reference_curves(
        ground_truth
    )
    qfrc_candidates: List[Tuple[str, Optional[np.ndarray]]] = [
        ("qfrc_inverse_pred", predictions.get("qfrc_inverse")),
        ("qfrc_inverse_mocap", ground_truth.get("qfrc_inverse_mocap")),
        ("qfrc_inverse_processed", ground_truth.get("qfrc_inverse_processed")),
        ("qfrc_inverse", ground_truth.get("qfrc_inverse")),
    ]
    qfrc_source = "missing_qfrc_inverse"
    qfrc_ref = None
    for src_name, candidate in qfrc_candidates:
        if candidate is not None:
            qfrc_ref = candidate
            qfrc_source = src_name
            break

    if tau_pred is None or selected_gt is None or qfrc_ref is None:
        return {
            "available": False,
            "reason": "full_id_torque_unavailable",
            "torque_source": str(qfrc_source),
            "subject": _load_patient_age_gender(patient_path),
        }

    target_width = _resolve_full_id_target_width(tau_pred, selected_gt, qfrc_ref)
    try:
        tau_pred_arr = _coerce_full_id_width(tau_pred, target_width, label="tau_grf_pred")
        gt_full_arr = _coerce_full_id_width(selected_gt, target_width, label=selected_label, fill_value=np.nan)
        qfrc_ref_arr = _coerce_full_id_width(qfrc_ref, target_width, label=qfrc_source)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"full_id_width_resolution_failed: {exc}",
            "torque_source": str(qfrc_source),
            "subject": _load_patient_age_gender(patient_path),
        }
    full_id_pred = np.asarray(qfrc_ref_arr, dtype=np.float64) - np.asarray(tau_pred_arr, dtype=np.float64)
    full_id_gt = np.asarray(gt_full_arr, dtype=np.float64)
    id_source = f"{selected_label}_with_{qfrc_source}"

    dof_names = list(get_dof_display_names(target_width))
    name_to_idx = {name: idx for idx, name in enumerate(dof_names)}
    required = {
        "right": {"ankle": "ankle_angle_r", "knee": "knee_angle_r"},
        "left": {"ankle": "ankle_angle_l", "knee": "knee_angle_l"},
    }
    for side_name, side_dofs in required.items():
        for label, dof_name in side_dofs.items():
            if dof_name not in name_to_idx:
                return {
                    "available": False,
                    "reason": f"missing_dof_{dof_name}",
                    "torque_source": str(id_source),
                    "subject": _load_patient_age_gender(patient_path),
                    "resolved_torque_width": int(target_width),
                    "available_dofs": dof_names,
                }

    pred_full = np.asarray(full_id_pred, dtype=np.float64)
    gt_full = np.asarray(full_id_gt, dtype=np.float64)
    grf_raw = np.asarray(data["grf_raw"], dtype=np.float64)
    # Use full-DOF MuJoCo velocities for torque*omega power when available.
    # `vel` (vel_inputs.npy) may be a reduced feature-space tensor and can be narrower.
    vel_source = "qvel_mjx_gt"
    vel_raw = data.get("qvel_mjx_gt")
    if vel_raw is None:
        vel_source = "qvel_mjx_input"
        vel_raw = data.get("qvel_mjx_input")
    if vel_raw is None:
        vel_source = "vel_inputs"
        vel_raw = data.get("vel")
    vel = np.asarray(vel_raw, dtype=np.float64) if vel_raw is not None else np.asarray([])
    if vel.ndim != 2:
        return {
            "available": False,
            "reason": "velocity_array_unavailable",
            "torque_source": str(id_source),
            "subject": _load_patient_age_gender(patient_path),
                "velocity_source": vel_source,
        }
    n_frames = int(min(len(pred_full), len(gt_full), len(grf_raw), len(vel)))
    pred_full = pred_full[:n_frames]
    gt_full = gt_full[:n_frames]
    grf_raw = grf_raw[:n_frames]
    vel = vel[:n_frames]
    eval_mask = _normalize_evaluation_mask(evaluation_mask, n_frames) if evaluation_mask is not None else None

    mass = float(data["mass"][0, 0])
    body_weight_n = mass * 9.8067

    # Load OpenSim ID ankle power (aligned to processed frame count) for peak detection.
    _opensim_power_by_side: Dict[str, np.ndarray] = {}
    _opensim_id_path_str: Optional[str] = None
    _opensim_id_reason: Optional[str] = None
    if trial_path is not None and n_frames >= OPENSIM_PEAK_MIN_TRIAL_FRAMES:
        _raw_pw, _raw_path, _raw_reason = _load_raw_opensim_ankle_power_w_per_kg(
            trial_path, mass_kg=mass, target_len=n_frames
        )
        _opensim_power_by_side = _raw_pw
        _opensim_id_path_str = str(_raw_path) if _raw_path is not None else None
        _opensim_id_reason = _raw_reason

    report: Dict[str, Any] = {
        "available": True,
        "analysis": "complete_stance_peak_torque_metrics",
        "criteria": {
            "detection": "dual_threshold_hysteresis",
            "core_bw_ratio": float(COMPLETE_STANCE_CORE_BW_RATIO),
            "low_threshold_n": float(COMPLETE_STANCE_THRESHOLD_N),
            "interpolation_points": 101,
            "exclude_prediction_window_edge_frames": True,
        },
        "torque_source": str(id_source),
        "resolved_torque_width": int(target_width),
        "trial_frame_count": int(n_frames),
        "subject": _load_patient_age_gender(patient_path),
        "velocity_source": vel_source,
        "sides": {},
    }

    ankle_peak_candidates_gt: List[float] = []
    ankle_peak_candidates_pred: List[float] = []
    for side_name in ("right", "left"):
        complete_phases = _find_complete_stance_phases(
            grf_raw=grf_raw,
            side=side_name,
            body_weight_n=body_weight_n,
            evaluation_mask=None,
        )
        ankle_idx = name_to_idx[required[side_name]["ankle"]]
        knee_idx = name_to_idx[required[side_name]["knee"]]
        if ankle_idx >= pred_full.shape[1] or ankle_idx >= gt_full.shape[1] or ankle_idx >= vel.shape[1]:
            return {
                "available": False,
                "reason": f"ankle_index_out_of_bounds_{side_name}",
                "torque_source": str(id_source),
                "subject": _load_patient_age_gender(patient_path),
                "resolved_torque_width": int(target_width),
                "ankle_idx": int(ankle_idx),
                "pred_width": int(pred_full.shape[1]),
                "gt_width": int(gt_full.shape[1]),
                "vel_width": int(vel.shape[1]),
                "velocity_source": vel_source,
            }
        if knee_idx >= pred_full.shape[1] or knee_idx >= gt_full.shape[1]:
            return {
                "available": False,
                "reason": f"knee_index_out_of_bounds_{side_name}",
                "torque_source": str(id_source),
                "subject": _load_patient_age_gender(patient_path),
            }

        phase_entries: List[Dict[str, Any]] = []
        for phase in complete_phases:
            start = int(phase["start_frame"])
            end = int(phase["end_frame_exclusive"])
            pred_ankle = pred_full[start:end, ankle_idx]
            gt_ankle = gt_full[start:end, ankle_idx]
            pred_knee = pred_full[start:end, knee_idx]
            gt_knee = gt_full[start:end, knee_idx]
            if pred_ankle.size == 0 or gt_ankle.size == 0 or pred_knee.size == 0 or gt_knee.size == 0:
                continue

            stance_len = int(end - start)
            stance_percent_full = np.linspace(0.0, 100.0, stance_len, dtype=np.float64)
            valid_local_mask = (
                np.asarray(eval_mask[start:end], dtype=bool)
                if eval_mask is not None
                else np.ones((stance_len,), dtype=bool)
            )
            valid_count = int(np.sum(valid_local_mask))
            pred_power_sparse = pred_ankle[valid_local_mask] * vel[start:end, ankle_idx][valid_local_mask]
            gt_power_sparse = gt_ankle[valid_local_mask] * vel[start:end, ankle_idx][valid_local_mask]
            percent_sparse = stance_percent_full[valid_local_mask]

            interp_percent, pred_power_interp = _interpolate_sparse_power_to_101(
                percent_sparse,
                pred_power_sparse,
            )
            _, gt_power_interp = _interpolate_sparse_power_to_101(
                percent_sparse,
                gt_power_sparse,
            )

            pred_ankle_abs_peak = float(np.max(np.abs(pred_ankle)))
            gt_ankle_abs_peak = float(np.max(np.abs(gt_ankle)))
            ankle_peak_candidates_pred.append(pred_ankle_abs_peak)
            ankle_peak_candidates_gt.append(gt_ankle_abs_peak)

            phase_entries.append(
                {
                    **phase,
                    "pred_peak_ankle_torque_abs_nm": pred_ankle_abs_peak,
                    "gt_peak_ankle_torque_abs_nm": gt_ankle_abs_peak,
                    "pred_peak_knee_flexion_nm": float(np.max(pred_knee)),
                    "gt_peak_knee_flexion_nm": float(np.max(gt_knee)),
                    "pred_peak_knee_extension_nm": float(np.min(pred_knee)),
                    "gt_peak_knee_extension_nm": float(np.min(gt_knee)),
                    "ankle_power": {
                        "units": "W",
                        "angular_velocity_source": str(vel_source),
                        "angular_velocity_dof_name": required[side_name]["ankle"],
                        "valid_frame_count_after_edge_exclusion": valid_count,
                        "stance_percent_full": _serialize_float_list_with_none(stance_percent_full),
                        "stance_percent_valid": _serialize_float_list_with_none(percent_sparse),
                        "pred_power_valid_w": _serialize_float_list_with_none(pred_power_sparse),
                        "gt_power_valid_w": _serialize_float_list_with_none(gt_power_sparse),
                        "stance_percent_101": _serialize_float_list_with_none(interp_percent),
                        "pred_power_101_w": _serialize_float_list_with_none(pred_power_interp),
                        "gt_power_101_w": _serialize_float_list_with_none(gt_power_interp),
                        "summary": {
                            "pred_peak_w": (
                                float(np.nanmax(pred_power_interp))
                                if np.isfinite(pred_power_interp).any()
                                else None
                            ),
                            "gt_peak_w": (
                                float(np.nanmax(gt_power_interp))
                                if np.isfinite(gt_power_interp).any()
                                else None
                            ),
                            "pred_min_w": (
                                float(np.nanmin(pred_power_interp))
                                if np.isfinite(pred_power_interp).any()
                                else None
                            ),
                            "gt_min_w": (
                                float(np.nanmin(gt_power_interp))
                                if np.isfinite(gt_power_interp).any()
                                else None
                            ),
                            "pred_mean_w": (
                                float(np.nanmean(pred_power_interp))
                                if np.isfinite(pred_power_interp).any()
                                else None
                            ),
                            "gt_mean_w": (
                                float(np.nanmean(gt_power_interp))
                                if np.isfinite(gt_power_interp).any()
                                else None
                            ),
                        },
                    },
                }
            )

        # --- OpenSim ID ankle power peak detection for this side ---
        _os_side_power = _opensim_power_by_side.get(side_name)
        if trial_path is None:
            _opensim_peaks_entry: Dict[str, Any] = {
                "available": False, "reason": "trial_path_not_provided",
            }
        elif n_frames < OPENSIM_PEAK_MIN_TRIAL_FRAMES:
            _opensim_peaks_entry = {
                "available": False, "reason": "trial_too_short",
                "n_frames": n_frames, "min_frames": OPENSIM_PEAK_MIN_TRIAL_FRAMES,
            }
        elif _opensim_id_reason is not None:
            _opensim_peaks_entry = {
                "available": False, "reason": _opensim_id_reason,
                "opensim_id_path": _opensim_id_path_str,
            }
        elif _os_side_power is None:
            _opensim_peaks_entry = {
                "available": False, "reason": "opensim_id_missing_side",
                "opensim_id_path": _opensim_id_path_str,
            }
        else:
            _os_arr = np.asarray(_os_side_power, dtype=np.float64)
            _peak_idxs, _ = scipy_find_peaks(_os_arr, height=float(OPENSIM_PEAK_MIN_W_PER_KG))
            _peaks_list: List[Dict[str, Any]] = []
            for _pidx in _peak_idxs:
                _pidx = int(_pidx)
                _os_val = float(_os_arr[_pidx])
                _gt_pwr = float(gt_full[_pidx, ankle_idx] * vel[_pidx, ankle_idx] / mass)
                _in_mask = eval_mask is None or bool(eval_mask[_pidx])
                _pred_pwr = (
                    float(pred_full[_pidx, ankle_idx] * vel[_pidx, ankle_idx] / mass)
                    if _in_mask else None
                )
                _peaks_list.append({
                    "frame": _pidx,
                    "time_s": round(float(_pidx) / float(FILTER_SAMPLING_RATE_HZ), 4),
                    "opensim_w_per_kg": round(_os_val, 4),
                    "mjx_gt_w_per_kg": round(_gt_pwr, 4),
                    "pred_w_per_kg": round(_pred_pwr, 4) if _pred_pwr is not None else None,
                })
            _opensim_peaks_entry = {
                "available": True,
                "peak_min_w_per_kg": float(OPENSIM_PEAK_MIN_W_PER_KG),
                "opensim_id_path": _opensim_id_path_str,
                "n_peaks": len(_peaks_list),
                "peaks": _peaks_list,
            }

        report["sides"][side_name] = {
            "complete_stance_count": int(len(phase_entries)),
            "complete_stances": phase_entries,
            "opensim_id_peaks": _opensim_peaks_entry,
        }

    report["overall_peak_ankle_torque_any_complete_stance"] = {
        "pred_peak_ankle_torque_abs_nm": (
            float(np.max(ankle_peak_candidates_pred))
            if ankle_peak_candidates_pred
            else None
        ),
        "gt_peak_ankle_torque_abs_nm": (
            float(np.max(ankle_peak_candidates_gt))
            if ankle_peak_candidates_gt
            else None
        ),
    }
    return report


# =============================================================================
# Ankle Power vs. Percent Stance (including edge partial stances)
# =============================================================================

def _load_speed_mapping(mapping_json_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and cache the speed-mapping JSON."""
    path = Path(mapping_json_path) if mapping_json_path is not None else DEFAULT_SPEED_MAPPING_PATH
    key = str(path)
    if key not in _SPEED_MAPPING_CACHE:
        try:
            with open(path) as fh:
                _SPEED_MAPPING_CACHE[key] = json.load(fh)
        except Exception:
            _SPEED_MAPPING_CACHE[key] = {}
    return _SPEED_MAPPING_CACHE[key]


def _lookup_trial_speed_code(trial_name: str, mapping_json_path: Optional[Path] = None) -> Optional[str]:
    """Return speed code ('80', '100', or '120') for a trial name like 'OA1/Trial_11'."""
    mapping = _load_speed_mapping(mapping_json_path)
    parts = trial_name.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    subject_id, trial_id = parts[0], parts[1]
    for entry in mapping.get("subjects", {}).get(subject_id, {}).get("trials", []):
        if entry.get("processed_trial") == trial_id:
            return str(entry["speed_code"])
    return None


def _partition_stance_phases(
    all_phases: List[Tuple[int, int]],
    n_frames: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split stance phases into (beginning_partial, end_partial, internal).

    Beginning partial: starts at frame 0 (heel-strike occurred before trial began).
    End partial: ends at the last frame (toe-off occurs after trial ends).
    Internal: both boundaries lie within the trial.
    A phase spanning the entire trial is treated as both beginning and end partial;
    only beginning_partial is populated in that case.
    """
    beginning_partial: Optional[Tuple[int, int]] = None
    end_partial: Optional[Tuple[int, int]] = None
    internal: List[Tuple[int, int]] = []
    for start, end in all_phases:
        if start == 0:
            beginning_partial = (start, end)
        elif end == n_frames:
            end_partial = (start, end)
        else:
            internal.append((start, end))
    return beginning_partial, end_partial, internal


def _ankle_power_curve_101(
    full_id_pred: np.ndarray,
    vel: np.ndarray,
    ankle_idx: int,
    start: int,
    end: int,
    eval_mask: Optional[np.ndarray],
    start_pct: float,
    end_pct: float,
) -> np.ndarray:
    """Compute ankle power for frames [start, end) and return a 101-point NaN curve.

    The observed frames are mapped to [start_pct, end_pct] on the 0-100% stance axis.
    Portions outside that range remain NaN (unobserved).
    """
    stance_len = end - start
    if stance_len <= 0:
        return np.full(101, np.nan, dtype=np.float64)

    percent_full = np.linspace(start_pct, end_pct, stance_len, dtype=np.float64)
    if eval_mask is not None:
        valid_local = np.asarray(eval_mask[start:end], dtype=bool)
    else:
        valid_local = np.ones(stance_len, dtype=bool)

    power_sparse = (
        np.asarray(full_id_pred[start:end, ankle_idx], dtype=np.float64)[valid_local]
        * np.asarray(vel[start:end, ankle_idx], dtype=np.float64)[valid_local]
    )
    percent_sparse = percent_full[valid_local]

    _, curve = _interpolate_sparse_power_to_101(percent_sparse, power_sparse)
    return curve


def build_ankle_power_avg_report(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    trial_name: str,
    evaluation_mask: Optional[np.ndarray] = None,
    mapping_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute average ankle power vs. % stance across both sides.

    Complete stances (internal, meeting the standard criteria) are mapped to 0-100%.
    Partial stances at the trial beginning (heel-strike before trial start) are
    anchored to the right edge (100%) and extend leftward based on estimated full
    stance duration.  Partial stances at the trial end are anchored to the left
    edge (0%) and extend rightward.  NaN regions (unobserved portions) are excluded
    from the per-point average.
    """
    tau_pred = predictions.get("tau_grf")
    qfrc_ref = None
    qfrc_source = "missing"
    for src, candidate in [
        ("qfrc_inverse_pred", predictions.get("qfrc_inverse")),
        ("qfrc_inverse_mocap", ground_truth.get("qfrc_inverse_mocap")),
        ("qfrc_inverse_processed", ground_truth.get("qfrc_inverse_processed")),
        ("qfrc_inverse", ground_truth.get("qfrc_inverse")),
    ]:
        if candidate is not None:
            qfrc_ref = candidate
            qfrc_source = src
            break

    if tau_pred is None or qfrc_ref is None:
        return {"available": False, "reason": "torque_unavailable", "torque_source": qfrc_source}

    target_width = _resolve_full_id_target_width(tau_pred, qfrc_ref)
    try:
        tau_arr = _coerce_full_id_width(tau_pred, target_width, label="tau_grf_pred")
        qfrc_arr = _coerce_full_id_width(qfrc_ref, target_width, label=qfrc_source)
    except Exception as exc:
        return {"available": False, "reason": f"width_resolution_failed: {exc}"}

    full_id_pred = np.asarray(qfrc_arr, dtype=np.float64) - np.asarray(tau_arr, dtype=np.float64)

    vel_raw = data.get("qvel_mjx_gt")
    if vel_raw is None:
        vel_raw = data.get("qvel_mjx_input")
    if vel_raw is None:
        vel_raw = data.get("vel")
    if vel_raw is None:
        return {"available": False, "reason": "velocity_unavailable"}
    vel = np.asarray(vel_raw, dtype=np.float64)
    if vel.ndim != 2:
        return {"available": False, "reason": "velocity_array_not_2d"}

    grf_raw = np.asarray(data["grf_raw"], dtype=np.float64)
    n_frames = int(min(len(full_id_pred), len(grf_raw), len(vel)))
    full_id_pred = full_id_pred[:n_frames]
    grf_raw = grf_raw[:n_frames]
    vel = vel[:n_frames]

    mass = float(data["mass"][0, 0])
    body_weight_n = mass * 9.8067
    eval_mask = _normalize_evaluation_mask(evaluation_mask, n_frames) if evaluation_mask is not None else None

    speed_code = _lookup_trial_speed_code(trial_name, mapping_json_path)
    fallback_frames = float(FALLBACK_STANCE_FRAMES_BY_SPEED.get(speed_code or "", 58))

    dof_names = list(get_dof_display_names(target_width))
    name_to_idx = {name: idx for idx, name in enumerate(dof_names)}

    all_curves: List[np.ndarray] = []
    per_side_meta: Dict[str, Any] = {}

    for side in ("right", "left"):
        ankle_dof = f"ankle_angle_{side[0]}"
        if ankle_dof not in name_to_idx:
            per_side_meta[side] = {"skipped": True, "reason": f"dof_{ankle_dof}_not_found"}
            continue
        ankle_idx = name_to_idx[ankle_dof]
        if ankle_idx >= full_id_pred.shape[1] or ankle_idx >= vel.shape[1]:
            per_side_meta[side] = {"skipped": True, "reason": "ankle_index_out_of_bounds"}
            continue

        all_phases = _dual_threshold_stance_intervals(
            grf_raw, side=side, body_weight_n=body_weight_n, evaluation_mask=None,
        )
        beginning_partial, end_partial, internal_phases = _partition_stance_phases(all_phases, n_frames)

        # Internal phases are already complete stances (gated by the >=20% BW
        # core in dual-threshold detection); no extra length/impulse filter.
        complete_phases: List[Tuple[int, int]] = list(internal_phases)

        # Estimated full stance length for this side.
        if complete_phases:
            avg_stance_frames = float(np.mean([e - s for s, e in complete_phases]))
        else:
            avg_stance_frames = fallback_frames

        side_curves: List[np.ndarray] = []

        for s, e in complete_phases:
            curve = _ankle_power_curve_101(
                full_id_pred, vel, ankle_idx, s, e, eval_mask,
                start_pct=0.0, end_pct=100.0,
            )
            side_curves.append(curve)

        if beginning_partial is not None:
            s, e = beginning_partial
            partial_len = e - s
            start_pct = max(0.0, (1.0 - partial_len / avg_stance_frames) * 100.0)
            curve = _ankle_power_curve_101(
                full_id_pred, vel, ankle_idx, s, e, eval_mask,
                start_pct=start_pct, end_pct=100.0,
            )
            side_curves.append(curve)

        if end_partial is not None:
            s, e = end_partial
            partial_len = e - s
            end_pct = min(100.0, (partial_len / avg_stance_frames) * 100.0)
            curve = _ankle_power_curve_101(
                full_id_pred, vel, ankle_idx, s, e, eval_mask,
                start_pct=0.0, end_pct=end_pct,
            )
            side_curves.append(curve)

        all_curves.extend(side_curves)
        per_side_meta[side] = {
            "complete_stance_count": len(complete_phases),
            "beginning_partial": beginning_partial is not None,
            "end_partial": end_partial is not None,
            "stances_contributing": len(side_curves),
            "avg_stance_frames_used": float(avg_stance_frames),
        }

    if not all_curves:
        return {
            "available": False,
            "reason": "no_stance_phases_found",
            "speed_code": speed_code,
            "per_side": per_side_meta,
        }

    stacked = np.stack(all_curves, axis=0)  # (N, 101)
    n_contributing = np.sum(~np.isnan(stacked), axis=0).tolist()
    avg_curve = np.nanmean(stacked, axis=0)
    avg_curve_serialized = [float(v) if np.isfinite(v) else None for v in avg_curve]

    return {
        "available": True,
        "analysis": "ankle_power_vs_stance_percent",
        "sides": "both_averaged",
        "speed_code": speed_code,
        "total_stances_contributing": len(all_curves),
        "n_contributing_per_percent": n_contributing,
        "avg_ankle_power_watts_101": avg_curve_serialized,
        "stance_percent_axis": list(range(101)),
        "per_side": per_side_meta,
        "criteria": {
            "detection": "dual_threshold_hysteresis",
            "core_bw_ratio": float(COMPLETE_STANCE_CORE_BW_RATIO),
            "low_threshold_n": float(COMPLETE_STANCE_THRESHOLD_N),
            "fallback_stance_frames": float(fallback_frames),
        },
    }


def build_ankle_power_per_stance_report(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    patient_path: str | Path,
    evaluation_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Report GT and predicted ankle power vs time for each complete stance individually.

    Each complete stance is returned at its original frame resolution (not resampled to
    101 points, not averaged).  Power is mass-normalised (W/kg).  The time axis is
    relative to the start of each stance (seconds at FILTER_SAMPLING_RATE_HZ).

    Only internal complete stances (not edge-clipped) that pass the standard
    3-criterion check are included.
    """
    # --- resolve torques (mirrors build_complete_stance_peak_report) ---
    tau_pred = predictions.get("tau_grf")
    selected_gt, selected_label, _mjx_gt, _opensim_gt, _opensim_mask = resolve_full_id_reference_curves(
        ground_truth
    )
    qfrc_ref = None
    qfrc_source = "missing_qfrc_inverse"
    for src_name, candidate in [
        ("qfrc_inverse_pred", predictions.get("qfrc_inverse")),
        ("qfrc_inverse_mocap", ground_truth.get("qfrc_inverse_mocap")),
        ("qfrc_inverse_processed", ground_truth.get("qfrc_inverse_processed")),
        ("qfrc_inverse", ground_truth.get("qfrc_inverse")),
    ]:
        if candidate is not None:
            qfrc_ref = candidate
            qfrc_source = src_name
            break

    if tau_pred is None or selected_gt is None or qfrc_ref is None:
        return {"available": False, "reason": "torque_unavailable"}

    target_width = _resolve_full_id_target_width(tau_pred, selected_gt, qfrc_ref)
    try:
        tau_arr  = _coerce_full_id_width(tau_pred,    target_width, label="tau_grf_pred")
        qfrc_arr = _coerce_full_id_width(qfrc_ref,    target_width, label=qfrc_source)
        gt_arr   = _coerce_full_id_width(selected_gt, target_width, label=selected_label, fill_value=np.nan)
    except Exception as exc:
        return {"available": False, "reason": f"width_resolution_failed: {exc}"}

    full_id_pred = np.asarray(qfrc_arr, dtype=np.float64) - np.asarray(tau_arr, dtype=np.float64)
    full_id_gt   = np.asarray(gt_arr,   dtype=np.float64)

    # --- velocity ---
    vel_raw = data.get("qvel_mjx_gt")
    if vel_raw is None:
        vel_raw = data.get("qvel_mjx_input")
    if vel_raw is None:
        vel_raw = data.get("vel")
    if vel_raw is None:
        return {"available": False, "reason": "velocity_unavailable"}
    vel = np.asarray(vel_raw, dtype=np.float64)
    if vel.ndim != 2:
        return {"available": False, "reason": "velocity_array_not_2d"}

    grf_raw = np.asarray(data["grf_raw"], dtype=np.float64)
    n_frames = int(min(len(full_id_pred), len(full_id_gt), len(grf_raw), len(vel)))
    full_id_pred = full_id_pred[:n_frames]
    full_id_gt   = full_id_gt[:n_frames]
    grf_raw      = grf_raw[:n_frames]
    vel          = vel[:n_frames]

    mass = float(data["mass"][0, 0])
    if mass <= 0:
        return {"available": False, "reason": "invalid_mass"}
    body_weight_n = mass * 9.8067

    eval_mask = _normalize_evaluation_mask(evaluation_mask, n_frames) if evaluation_mask is not None else None

    dof_names  = list(get_dof_display_names(target_width))
    name_to_idx = {name: idx for idx, name in enumerate(dof_names)}

    all_stances: List[Dict[str, Any]] = []

    for side in ("right", "left"):
        ankle_dof = f"ankle_angle_{side[0]}"
        if ankle_dof not in name_to_idx:
            continue
        ankle_idx = name_to_idx[ankle_dof]
        if ankle_idx >= full_id_pred.shape[1] or ankle_idx >= vel.shape[1]:
            continue

        all_phases = _dual_threshold_stance_intervals(
            grf_raw, side=side, body_weight_n=body_weight_n, evaluation_mask=None,
        )

        for start, end in all_phases:
            # Internal only — skip edge-clipped stances
            if start == 0 or end == n_frames:
                continue
            length = end - start

            # Frame-level validity from evaluation mask
            if eval_mask is not None:
                valid = np.asarray(eval_mask[start:end], dtype=bool)
            else:
                valid = np.ones(length, dtype=bool)

            time_s = (np.arange(length) / FILTER_SAMPLING_RATE_HZ).tolist()

            # GT power — always reported (no eval mask applied)
            gt_torque = full_id_gt[start:end, ankle_idx]
            gt_vel    = vel[start:end, ankle_idx]
            gt_power  = (gt_torque * gt_vel / mass).tolist()

            # Pred power — None where outside eval mask
            pred_torque = full_id_pred[start:end, ankle_idx]
            pred_vel    = vel[start:end, ankle_idx]
            pred_power_arr = pred_torque * pred_vel / mass
            pred_power: List[Optional[float]] = [
                float(v) if valid[i] and np.isfinite(v) else None
                for i, v in enumerate(pred_power_arr)
            ]

            all_stances.append({
                "side":          side,
                "start_frame":   int(start),
                "end_frame":     int(end),
                "length_frames": int(length),
                "duration_s":    round(length / FILTER_SAMPLING_RATE_HZ, 4),
                "time_s":        [round(t, 4) for t in time_s],
                "gt_power_w_per_kg":   [round(v, 6) if np.isfinite(v) else None for v in gt_power],
                "pred_power_w_per_kg": pred_power,
            })

    if not all_stances:
        return {"available": False, "reason": "no_complete_stances_found"}

    return {
        "available":    True,
        "analysis":     "ankle_power_vs_time_per_stance",
        "mass_kg":      round(mass, 4),
        "sample_rate_hz": float(FILTER_SAMPLING_RATE_HZ),
        "total_stances": len(all_stances),
        "criteria": {
            "detection":       "dual_threshold_hysteresis",
            "core_bw_ratio":   float(COMPLETE_STANCE_CORE_BW_RATIO),
            "low_threshold_n": float(COMPLETE_STANCE_THRESHOLD_N),
        },
        "stances": all_stances,
    }


# =============================================================================
# Full-Trial Ankle Power Plotting
# =============================================================================

def _find_raw_opensim_id_mot_file(trial_path: str | Path) -> Optional[Path]:
    """Find the Raw/ OpenSim ID .mot file used for ankle power."""
    raw_dir = Path(trial_path) / "Motion" / "Raw"
    if not raw_dir.exists():
        return None

    candidates = sorted(raw_dir.glob("*.mot"))
    candidates = [
        path for path in candidates
        if "id" in path.name.lower() and "_ik" not in path.name.lower()
    ]
    if not candidates:
        return None

    def _score(path: Path) -> Tuple[int, str]:
        name = path.name.lower()
        return (0 if name.endswith("id.mot") else 1, name)

    return sorted(candidates, key=_score)[0]


def _align_motion_raw_series_to_processed_frames(
    values: np.ndarray,
    trial_path: str | Path,
    target_len: int,
    source_time: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply known motion-space trims, then resample to processed frame count if needed."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    trial_root = Path(trial_path)
    info = _load_json_dict(video_processed_dir(trial_root) / "Trial_Processing_Information.json") or {}

    motion_time_path = trial_root / "Motion" / "Time.npy"
    if source_time is not None and motion_time_path.exists():
        try:
            source_time_arr = np.asarray(source_time, dtype=np.float64).reshape(-1)
            motion_time = np.asarray(np.load(motion_time_path), dtype=np.float64).reshape(-1)
            if (
                len(source_time_arr) == len(values)
                and len(source_time_arr) > 1
                and len(motion_time) > 1
            ):
                order = np.argsort(source_time_arr)
                sorted_time = source_time_arr[order]
                sorted_values = values[order]
                unique_time, unique_idx = np.unique(sorted_time, return_index=True)
                unique_values = sorted_values[unique_idx]
                if len(unique_time) > 1:
                    source_span = float(unique_time[-1] - unique_time[0])
                    motion_span = float(np.nanmax(motion_time) - np.nanmin(motion_time))
                    overlap = min(float(unique_time[-1]), float(np.nanmax(motion_time))) - max(
                        float(unique_time[0]),
                        float(np.nanmin(motion_time)),
                    )
                    min_span = max(min(source_span, motion_span), 1e-9)
                    if overlap / min_span < 0.5:
                        unique_time = unique_time - unique_time[0]
                        motion_time = motion_time - motion_time[0]
                    values = np.interp(
                        motion_time,
                        unique_time,
                        unique_values,
                        left=float(unique_values[0]),
                        right=float(unique_values[-1]),
                    )
        except Exception:
            pass

    bounds = info.get("core_trim_bounds_motion_aligned")
    if (
        isinstance(bounds, (list, tuple))
        and len(bounds) == 2
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in bounds)
    ):
        start, end = int(bounds[0]), int(bounds[1])
        if 0 <= start <= end <= len(values):
            trimmed = values[start:end]
        else:
            trimmed = values
    else:
        trimmed = values

    trimmed = _apply_ds_edge_trim_if_needed(trimmed, info, target_len=target_len)
    if len(trimmed) == int(target_len):
        return trimmed

    grf_bounds = info.get("grf_trim_bounds_motion_aligned")
    weak_bounds = info.get("weak_edge_trim_bounds_after_grf")
    outlier_bounds = info.get("outlier_trim_bounds_after_weak_edge")
    if isinstance(grf_bounds, (list, tuple)) and len(grf_bounds) == 2:
        start, end = int(grf_bounds[0]), int(grf_bounds[1])
        if 0 <= start <= end <= len(values):
            candidate = values[start:end]
            for next_bounds in (weak_bounds, outlier_bounds):
                if isinstance(next_bounds, (list, tuple)) and len(next_bounds) == 2:
                    ns, ne = int(next_bounds[0]), int(next_bounds[1])
                    if 0 <= ns <= ne <= len(candidate):
                        candidate = candidate[ns:ne]
            candidate = _apply_ds_edge_trim_if_needed(
                candidate,
                info,
                target_len=target_len,
            )
            if len(candidate) == int(target_len):
                return candidate
            trimmed = candidate

    if len(trimmed) == 0:
        return np.full((int(target_len),), np.nan, dtype=np.float64)
    if len(trimmed) == 1:
        return np.repeat(trimmed, int(target_len))

    source = np.linspace(0.0, 1.0, len(trimmed), dtype=np.float64)
    target = np.linspace(0.0, 1.0, int(target_len), dtype=np.float64)
    return np.interp(target, source, trimmed)


def _load_raw_opensim_ankle_power_w_per_kg(
    trial_path: str | Path,
    mass_kg: float,
    target_len: int,
) -> Tuple[Dict[str, np.ndarray], Optional[Path], Optional[str]]:
    """Load full-trial right/left ankle power from Motion/Raw OpenSim ID .mot."""
    id_path = _find_raw_opensim_id_mot_file(trial_path)
    if id_path is None:
        return {}, None, "missing_raw_id_mot"

    try:
        df, _header = _load_opensim_sto(id_path)
    except Exception as exc:
        return {}, id_path, f"load_failed: {exc}"

    out: Dict[str, np.ndarray] = {}
    for side_name, side_code in (("right", "r"), ("left", "l")):
        torque_col = f"ankle_flex_{side_code}_torque"
        velocity_col = f"ankle_flex_{side_code}_vel"
        if torque_col not in df.columns or velocity_col not in df.columns:
            continue
        source_time = np.asarray(df["time"], dtype=np.float64) if "time" in df.columns else None
        torque_nm = np.asarray(df[torque_col], dtype=np.float64)
        velocity_rad_s = np.asarray(df[velocity_col], dtype=np.float64) * np.pi / 180.0
        power_w_per_kg = (torque_nm * velocity_rad_s) / float(mass_kg)
        out[side_name] = _align_motion_raw_series_to_processed_frames(
            power_w_per_kg,
            trial_path,
            target_len=target_len,
            source_time=source_time,
        )

    if not out:
        return out, id_path, "missing_ankle_power_columns"
    return out, id_path, None


def create_full_trial_ankle_power_plot(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    trial_path: str | Path,
    trial_name: str,
    save_path: str | Path,
    evaluation_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Plot MJX GT, Raw OpenSim, and predicted ankle power for the full trial."""
    mass = float(data["mass"][0, 0])
    if not np.isfinite(mass) or mass <= 0.0:
        return {"available": False, "reason": "invalid_mass"}

    full_id_pred, _selected_full_id_gt, id_source = compute_full_id_curves(predictions, ground_truth)
    mjx_gt = ground_truth.get("id_gt_mjx")
    if full_id_pred is None or mjx_gt is None:
        return {"available": False, "reason": f"full_id_unavailable: {id_source}"}

    vel_raw = data.get("qvel_mjx_gt")
    vel_source = "qvel_mjx_gt"
    if vel_raw is None:
        vel_raw = data.get("qvel_mjx_input")
        vel_source = "qvel_mjx_input"
    if vel_raw is None:
        vel_raw = data.get("vel")
        vel_source = "vel_inputs"
    if vel_raw is None:
        return {"available": False, "reason": "velocity_unavailable"}

    vel = np.asarray(vel_raw, dtype=np.float64)
    full_id_pred = np.asarray(full_id_pred, dtype=np.float64)
    target_width = _resolve_full_id_target_width(full_id_pred, mjx_gt)
    try:
        full_id_pred = _coerce_full_id_width(full_id_pred, target_width, label="full_id_pred")
        full_id_gt = _coerce_full_id_width(mjx_gt, target_width, label="id_gt_mjx", fill_value=np.nan)
    except Exception as exc:
        return {"available": False, "reason": f"full_id_width_resolution_failed: {exc}"}

    full_id_gt = np.asarray(full_id_gt, dtype=np.float64)
    if vel.ndim != 2 or full_id_pred.ndim != 2 or full_id_gt.ndim != 2:
        return {"available": False, "reason": "unexpected_array_shape"}

    n_frames = int(min(len(full_id_pred), len(full_id_gt), len(vel)))
    if n_frames <= 0:
        return {"available": False, "reason": "empty_trial"}
    full_id_pred = full_id_pred[:n_frames]
    full_id_gt = full_id_gt[:n_frames]
    vel = vel[:n_frames]

    raw_power, raw_source_path, raw_reason = _load_raw_opensim_ankle_power_w_per_kg(
        trial_path,
        mass_kg=mass,
        target_len=n_frames,
    )

    eval_mask = (
        _normalize_evaluation_mask(evaluation_mask, n_frames)
        if evaluation_mask is not None
        else np.ones((n_frames,), dtype=bool)
    )
    time_s = np.arange(n_frames, dtype=np.float64) / float(FILTER_SAMPLING_RATE_HZ)

    dof_names = list(get_dof_display_names(full_id_pred.shape[1]))
    name_to_idx = {name: idx for idx, name in enumerate(dof_names)}
    side_specs = (
        ("right", "Right", "ankle_angle_r"),
        ("left", "Left", "ankle_angle_l"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    plotted_any = False
    for ax, (side_key, side_label, ankle_name) in zip(axes, side_specs):
        ankle_idx = name_to_idx.get(ankle_name)
        if ankle_idx is None or ankle_idx >= full_id_pred.shape[1] or ankle_idx >= vel.shape[1]:
            ax.set_title(f"{side_label} ankle power unavailable ({ankle_name})")
            ax.grid(True, alpha=0.3)
            continue

        mjx_gt_power = full_id_gt[:, ankle_idx] * vel[:, ankle_idx] / mass
        pred_power = full_id_pred[:, ankle_idx] * vel[:, ankle_idx] / mass
        pred_power = np.where(eval_mask, pred_power, np.nan)

        ax.plot(time_s, mjx_gt_power, label="MJX calculated GT", color="#2E86AB", linewidth=1.5)
        if side_key in raw_power:
            ax.plot(
                time_s,
                raw_power[side_key],
                label="OpenSim Raw",
                color="#444444",
                linewidth=1.2,
                alpha=0.85,
            )
        ax.plot(time_s, pred_power, label="Predicted", color="#E94F37", linewidth=1.3, alpha=0.9)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.set_ylabel(f"{side_label}\nPower (W/kg)")
        ax.grid(True, alpha=0.3)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return {"available": False, "reason": "ankle_dofs_unavailable"}

    title = f"Full-Trial Ankle Power: {trial_name}"
    if raw_reason is not None:
        title += f"\nRaw OpenSim unavailable: {raw_reason}"
    fig.suptitle(title, fontsize=13)
    axes[-1].set_xlabel("Time (s)")
    handles_by_label: Dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update({label: handle for handle, label in zip(handles, labels)})
    labels = list(handles_by_label.keys())
    handles = [handles_by_label[label] for label in labels]
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "available": True,
        "path": str(save_path),
        "id_source": id_source,
        "velocity_source": vel_source,
        "raw_opensim_source": str(raw_source_path) if raw_source_path is not None else None,
        "raw_opensim_available_sides": sorted(raw_power.keys()),
        "n_frames": int(n_frames),
        "units": "W/kg",
    }


def _select_stance_phases_for_analysis(all_phases: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Match the legacy left-stance phase selection logic used in the existing analysis."""
    if len(all_phases) > 2:
        return all_phases[1:-1]
    if len(all_phases) == 2:
        first = all_phases[0][1] - all_phases[0][0]
        second = all_phases[1][1] - all_phases[1][0]
        return [all_phases[0] if first >= second else all_phases[1]]
    if len(all_phases) == 1:
        return list(all_phases)
    return []


def _compute_phase_normalized_mae_report(
    pred: np.ndarray,
    target: np.ndarray,
    channel_names: List[str],
    phases: List[Tuple[int, int]],
) -> Dict[str, float]:
    """Compute 0-100% stance MAE per channel for the selected stance phases."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or target.ndim != 2 or pred.shape != target.shape:
        raise ValueError("Expected pred and target to have identical 2D shapes.")
    if not phases:
        return {}

    x_new = np.linspace(0.0, 100.0, 101)
    mae_report: Dict[str, float] = {}
    for ch_idx, channel_name in enumerate(channel_names):
        stacked_pred = []
        stacked_gt = []
        for start, end in phases:
            seg_pred = pred[start:end, ch_idx]
            seg_gt = target[start:end, ch_idx]
            if len(seg_pred) < 2 or len(seg_gt) < 2:
                continue
            x_old = np.linspace(0.0, 100.0, len(seg_pred))
            stacked_pred.append(np.interp(x_new, x_old, seg_pred))
            stacked_gt.append(np.interp(x_new, x_old, seg_gt))

        if stacked_pred:
            stacked_pred_np = np.asarray(stacked_pred, dtype=np.float64)
            stacked_gt_np = np.asarray(stacked_gt, dtype=np.float64)
            mae_report[channel_name] = float(
                np.mean(np.abs(stacked_pred_np - stacked_gt_np))
            )
    return mae_report


def build_bilateral_stance_mae_report(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    data: Dict[str, np.ndarray],
    evaluation_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Build a separate stance-phase MAE report for both right and left analyses."""
    mass = float(data["mass"][0, 0])
    height = float(data["height"][0, 0])
    norm_factor = mass * height * 9.8067
    body_weight = mass * 9.8067

    full_id_pred, full_id_gt, id_source = compute_full_id_curves(predictions, ground_truth)
    if full_id_pred is not None and full_id_gt is not None:
        pred_torque_norm = np.asarray(full_id_pred, dtype=np.float64) / norm_factor * 100.0
        gt_torque_norm = np.asarray(full_id_gt, dtype=np.float64) / norm_factor * 100.0
    else:
        pred_torque_norm = -np.asarray(
            predictions["qfrc_grf_contribution"], dtype=np.float64
        ) / norm_factor * 100.0
        gt_torque_norm = -np.asarray(ground_truth["tau_grf"], dtype=np.float64) / norm_factor * 100.0
        id_source = "raw_grf_contribution_fallback"

    pred_grf_norm = np.asarray(predictions["grf"], dtype=np.float64) / body_weight * 100.0
    gt_grf_norm = np.asarray(ground_truth["grf"], dtype=np.float64) / body_weight * 100.0
    pred_cop_norm = np.asarray(predictions["cop"], dtype=np.float64) / height * 100.0
    gt_cop_norm = np.asarray(ground_truth["cop"], dtype=np.float64) / height * 100.0

    dof_names = list(get_dof_names())
    torque_width = int(min(pred_torque_norm.shape[1], gt_torque_norm.shape[1]))
    if len(dof_names) > torque_width:
        dof_names = dof_names[:torque_width]
    elif len(dof_names) < torque_width:
        dof_names.extend([f"DOF_{i}" for i in range(len(dof_names), torque_width)])
    side_specs = {
        "right": {
            "cop_indices": [0, 1],
            "cop_names": ["COP_X_Right", "COP_Z_Right"],
            "grf_indices": [0, 1, 2],
            "grf_names": ["GRF_X_Right", "GRF_Y_Right", "GRF_Z_Right"],
        },
        "left": {
            "cop_indices": [2, 3],
            "cop_names": ["COP_X_Left", "COP_Z_Left"],
            "grf_indices": [3, 4, 5],
            "grf_names": ["GRF_X_Left", "GRF_Y_Left", "GRF_Z_Left"],
        },
    }

    report: Dict[str, object] = {
        "analysis": "phase_normalized_stance_mae",
        "phase": "stance",
        "units": {
            "torque_mae": "%BW*H",
            "cop_mae": "% height",
            "grf_mae": "%BW",
        },
        "torque_source": str(id_source),
        "stance_threshold_N": float(LEFT_STANCE_THRESHOLD_N),
        "sides": {},
    }

    for side_name, side_spec in side_specs.items():
        stance_mask = _get_side_stance_mask(
            data["grf_raw"],
            evaluation_mask,
            side=side_name,
            threshold=LEFT_STANCE_THRESHOLD_N,
        )
        all_phases = _get_stance_phases_from_mask(stance_mask)
        phases = _select_stance_phases_for_analysis(all_phases)
        selected_phase_durations = [end - start for start, end in phases]

        report["sides"][side_name] = {
            "stance_frame_count": int(np.sum(stance_mask)),
            "detected_stance_phase_count": int(len(all_phases)),
            "selected_stance_phase_count": int(len(phases)),
            "selected_average_phase_duration_frames": (
                float(np.mean(selected_phase_durations))
                if selected_phase_durations
                else None
            ),
            "torque_mae_percent_bwh": _compute_phase_normalized_mae_report(
                pred_torque_norm[:, :torque_width],
                gt_torque_norm[:, :torque_width],
                dof_names,
                phases,
            ),
            "cop_mae_percent_height": _compute_phase_normalized_mae_report(
                pred_cop_norm[:, side_spec["cop_indices"]],
                gt_cop_norm[:, side_spec["cop_indices"]],
                side_spec["cop_names"],
                phases,
            ),
            "grf_mae_percent_bw": _compute_phase_normalized_mae_report(
                pred_grf_norm[:, side_spec["grf_indices"]],
                gt_grf_norm[:, side_spec["grf_indices"]],
                side_spec["grf_names"],
                phases,
            ),
        }

    return report


def _convert_output_to_physical_predictions(
    output_np: np.ndarray,
    data: Dict[str, np.ndarray],
    normalizers: Dict[str, "Normalizer"],
    detected_output_dim: int,
    cop_mask: bool,
    use_grf_norm_cop: bool = False,
    qfrc_inverse_output_dim: int = 0,
    rotation_output_dim: int = 0,
    jacobian_output_dim: int = PREDICTED_JACOBIAN_FLAT_DIM,
    use_gt_jacob_and_rot: bool = False,
) -> Dict[str, np.ndarray]:
    """Convert model outputs to physical COP/GRF/Moments/tau arrays.

    When use_gt_jacob_and_rot is set, the predicted forces are mapped to torque
    using the ground-truth (MoCap) Jacobian, rotation, and qfrc_inverse instead of
    the video (ProcessedData) ones -- isolating how much of the COP/GRF signal the
    model recovers from OpenCap kinematics, independent of the kinematic transforms.
    """
    output_np = np.asarray(output_np, dtype=np.float32)
    qfrc_inverse_output_dim = 0
    rotation_output_dim = 0
    legacy_six_moment_layout = (
        detected_output_dim == 16
        and qfrc_inverse_output_dim == 0
        and rotation_output_dim == 0
    )

    if legacy_six_moment_layout:
        cop_pred_raw = output_np[:, 0:4]
        grf_pred_raw = output_np[:, 4:10]
        moments_pred_raw = output_np[:, 10:16]
        contact_pred = None
        jacobian_pred = None
        qfrc_inverse_pred = None
        rotation_pred = None
    else:
        (
            cop_pred_raw,
            grf_pred_raw,
            moments_pred_raw,
            contact_pred,
            qfrc_inverse_pred,
            rotation_pred,
            jacobian_pred,
        ) = split_model_predictions(
            output_np,
            qfrc_inverse_output_dim=qfrc_inverse_output_dim,
            rotation_output_dim=rotation_output_dim,
        )

    cop_pred_norm = normalizers["cop"].unnormalize(cop_pred_raw)
    grf_pred_norm = normalizers["grf"].unnormalize(grf_pred_raw)
    moments_pred_ratio = normalizers["moments"].unnormalize(moments_pred_raw)

    predicted_jacobian = None
    if use_gt_jacob_and_rot and data.get("qfrc_inverse_mocap") is not None:
        qfrc_inverse_phys = np.asarray(data["qfrc_inverse_mocap"], dtype=np.float32)
    elif data.get("qfrc_inverse_raw") is not None:
        qfrc_inverse_phys = np.asarray(data["qfrc_inverse_raw"], dtype=np.float32)
    elif data.get("qfrc_inverse") is not None:
        qfrc_inverse_phys = _qfrc_inverse_phys_from_scaled(
            data["qfrc_inverse"],
            data,
        )
    else:
        qfrc_inverse_phys = None

    _rot_src = (
        data["gt_rot_w_to_ga"]
        if (use_gt_jacob_and_rot and data.get("gt_rot_w_to_ga") is not None)
        else data["rot_w_to_ga"]
    )
    rotation_pred_phys = project_rotation_matrices(
        np.asarray(_rot_src, dtype=np.float32),
        xp=np,
    ).astype(np.float32, copy=False)

    if cop_mask and contact_pred is not None:
        contact_prob = contact_pred
        contact_hard = (contact_prob > 0.5).astype(np.float32)
        mask_r = contact_hard[:, 0:1]
        mask_l = contact_hard[:, 1:2]

        cop_pred_norm = np.concatenate([
            cop_pred_norm[:, 0:2] * mask_r,
            cop_pred_norm[:, 2:4] * mask_l,
        ], axis=-1)
        grf_pred_norm = np.concatenate([
            grf_pred_norm[:, 0:3] * mask_r,
            grf_pred_norm[:, 3:6] * mask_l,
        ], axis=-1)
        moments_pred_ratio = np.concatenate([
            moments_pred_ratio[:, 0:1] * mask_r,
            moments_pred_ratio[:, 1:2] * mask_l,
        ], axis=-1)

    h = data["height"]
    m = data["mass"]

    grf_pred = grf_pred_norm * m * 9.8067
    cop_pred = decode_cop_signal_to_length_np(
        cop_pred_norm,
        grf_pred_norm,
        h,
        use_grf_norm_cop=use_grf_norm_cop,
        contact_probability=contact_pred if use_grf_norm_cop else None,
    )

    if moments_pred_ratio.shape[-1] == 6:
        moments_pred = moments_pred_ratio * m * h * 9.8067
    else:
        moments_pred_z = moments_pred_ratio * m * h * 9.8067
        mz_r = moments_pred_z[:, 0:1]
        mz_l = moments_pred_z[:, 1:2]
        mom_r = np.concatenate([np.zeros_like(mz_r), np.zeros_like(mz_r), mz_r], axis=-1)
        mom_l = np.concatenate([np.zeros_like(mz_l), np.zeros_like(mz_l), mz_l], axis=-1)
        moments_pred = np.concatenate([mom_r, mom_l], axis=-1)

    rot_w_to_ga = rotation_pred_phys
    rot_ga_to_w_r = np.transpose(rot_w_to_ga[:, 0], (0, 2, 1))
    rot_ga_to_w_l = np.transpose(rot_w_to_ga[:, 1], (0, 2, 1))

    cop_r_ga_pred = np.stack([cop_pred[:, 0], data["ankle_heights"][:, 0], cop_pred[:, 1]], axis=1)
    cop_l_ga_pred = np.stack([cop_pred[:, 2], data["ankle_heights"][:, 1], cop_pred[:, 3]], axis=1)
    r_vec_R = np.einsum("tij,tj->ti", rot_ga_to_w_r, cop_r_ga_pred)
    r_vec_L = np.einsum("tij,tj->ti", rot_ga_to_w_l, cop_l_ga_pred)

    F_R = grf_pred[:, 0:3]
    M_free_R = moments_pred[:, 0:3]
    M_total_R = M_free_R + np.cross(r_vec_R, F_R)

    F_L = grf_pred[:, 3:6]
    M_free_L = moments_pred[:, 3:6]
    M_total_L = M_free_L + np.cross(r_vec_L, F_L)

    if use_gt_jacob_and_rot and data.get("gt_jacp") is not None and data.get("gt_jacr") is not None:
        jacp = data["gt_jacp"]
        jacr = data["gt_jacr"]
    else:
        jacp = data["jacp"]
        jacr = data["jacr"]
    tau_R = np.einsum("tji,tj->ti", jacp[:, 0], F_R) + np.einsum("tji,tj->ti", jacr[:, 0], M_total_R)
    tau_L = np.einsum("tji,tj->ti", jacp[:, 1], F_L) + np.einsum("tji,tj->ti", jacr[:, 1], M_total_L)

    if use_gt_jacob_and_rot and data.get("gt_ankle_pos") is not None and data.get("gt_knee_pos") is not None:
        kam_ankle_pos = data["gt_ankle_pos"]
        kam_knee_pos = data["gt_knee_pos"]
    else:
        kam_ankle_pos = data.get("ankle_pos")
        kam_knee_pos = data.get("knee_pos")
    predicted_knee_to_cop_vectors = None
    if kam_ankle_pos is not None and kam_knee_pos is not None:
        predicted_knee_to_cop_vectors = _compute_predicted_knee_to_cop_vectors_np(
            cop_pred_xz=cop_pred,
            ankle_pos_global=kam_ankle_pos,
            knee_pos_global=kam_knee_pos,
            rot_w_to_ga=rotation_pred_phys,
        )

    result = {
        "cop": cop_pred,
        "grf": grf_pred,
        "moments": moments_pred,
        "tau_grf": tau_R + tau_L,
        "qfrc_grf_contribution": tau_R + tau_L,
        "qfrc_inverse": qfrc_inverse_phys,
        "rot_w_to_ga": rotation_pred_phys,
        "predicted_knee_to_cop_vectors": predicted_knee_to_cop_vectors,
        "contact": None if contact_pred is None else np.asarray(contact_pred, dtype=np.float32),
    }
    return result


# =============================================================================
# Physics Computation
# =============================================================================

def compute_tau_grf(grf: np.ndarray, torques: np.ndarray, jacp: np.ndarray, jacr: np.ndarray) -> np.ndarray:
    """Compute τ_grf = Jp^T @ GRF + Jr^T @ M for each timestep."""
    seq_len = grf.shape[0]
    tau_grf = np.zeros((seq_len, int(jacp.shape[-1])), dtype=np.float32)
    
    for t in range(seq_len):
        # Right foot (body 0)
        tau_grf[t] += jacp[t, 0].T @ grf[t, :3]
        tau_grf[t] += jacr[t, 0].T @ torques[t, :3]
        # Left foot (body 1)
        tau_grf[t] += jacp[t, 1].T @ grf[t, 3:]
        tau_grf[t] += jacr[t, 1].T @ torques[t, 3:]
    
    return tau_grf


# =============================================================================
# Visualization with Plotly
# =============================================================================

def _add_prediction_margin_shading(
    fig: go.Figure,
    time_axis: np.ndarray,
    prediction_margin_frames: int,
) -> None:
    """Shade the first and last prediction-margin frames on time-series plots."""
    if prediction_margin_frames <= 0:
        return

    time_axis = np.asarray(time_axis, dtype=np.float64).reshape(-1)
    if time_axis.size == 0:
        return

    shaded_frames = min(int(prediction_margin_frames), int(time_axis.size))
    if shaded_frames <= 0:
        return

    dt = float(np.median(np.diff(time_axis))) if time_axis.size > 1 else 1.0 / 100.0
    left_x0 = float(time_axis[0] - 0.5 * dt)
    left_x1 = float(time_axis[shaded_frames - 1] + 0.5 * dt)
    right_x0 = float(time_axis[time_axis.size - shaded_frames] - 0.5 * dt)
    right_x1 = float(time_axis[-1] + 0.5 * dt)

    shade_style = dict(
        fillcolor="rgba(120, 120, 120, 0.16)",
        line_width=0,
        layer="below",
    )
    fig.add_vrect(x0=left_x0, x1=left_x1, row="all", col="all", **shade_style)
    fig.add_vrect(x0=right_x0, x1=right_x1, row="all", col="all", **shade_style)


def create_timeseries_plot(
    time_axis: np.ndarray,
    predictions: Dict[str, np.ndarray],
    predictions_alt: Optional[Dict[str, np.ndarray]],
    ground_truth: Dict[str, np.ndarray],
    trial_name: str,
    side: str = 'Right',
    save_path: str = None,
    pred_label: str = "Prediction (OpenCap input)",
    alt_pred_label: str = "Prediction (MotionCapture input)",
    evaluation_mask: Optional[np.ndarray] = None,
    metric_predictions: Optional[Dict[str, np.ndarray]] = None,
    metric_predictions_alt: Optional[Dict[str, np.ndarray]] = None,
    prediction_margin_frames: int = 0,
):
    """Create interactive Plotly visualization for a specific side (Right or Left)."""
    
    # Create subplots: 2 rows (COP, GRF) x 3 cols (X, Y, Z)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f'COP {side} X', f'COP {side} Z', f'COP {side} Y (derived)',
            f'GRF {side} X', f'GRF {side} Y', f'GRF {side} Z',
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    
    colors = {'gt': '#2E86AB', 'mjx': '#6C757D', 'pred': '#E94F37', 'pred_alt': '#1B9E77'}
    metric_predictions = metric_predictions if metric_predictions is not None else predictions
    metric_predictions_alt = (
        metric_predictions_alt if metric_predictions_alt is not None else predictions_alt
    )
    
    # Determine indices based on side
    if side == 'Right':
        cop_indices = [0, 1] # Rx, Rz
        grf_indices = [0, 1, 2] # Rx, Ry, Rz
    else: # Left
        cop_indices = [2, 3] # Lx, Lz
        grf_indices = [3, 4, 5] # Lx, Ly, Lz
    
    # Row 1: COP
    for i, name in enumerate(['X', 'Z', 'Y']):
        gt_cop = ground_truth['cop']
        pred_cop = predictions['cop']
        
        if i < 2:
            # X or Z (ground-aligned calc frame channels)
            idx = cop_indices[i]
            gt_val = gt_cop[:, idx]
            pred_val = pred_cop[:, idx]
        else:
            # Y is not supervised directly in this representation.
            gt_val = np.zeros_like(gt_cop[:, 0])
            pred_val = np.zeros_like(pred_cop[:, 0])
            
        fig.add_trace(go.Scatter(
            x=time_axis, y=gt_val,
            name='Ground Truth', line=dict(color=colors['gt'], width=2),
            legendgroup='gt', showlegend=(i == 0),
        ), row=1, col=i+1)
        fig.add_trace(go.Scatter(
            x=time_axis, y=pred_val,
            name=pred_label, line=dict(color=colors['pred'], width=2, dash='dash'),
            legendgroup='pred', showlegend=(i == 0),
        ), row=1, col=i+1)
        if predictions_alt is not None:
            pred_alt_cop = predictions_alt['cop']
            if i < 2:
                pred_alt_val = pred_alt_cop[:, idx]
            else:
                pred_alt_val = np.zeros_like(pred_alt_cop[:, 0])
            fig.add_trace(go.Scatter(
                x=time_axis, y=pred_alt_val,
                name=alt_pred_label, line=dict(color=colors['pred_alt'], width=2, dash='dot'),
                legendgroup='pred_alt', showlegend=(i == 0),
            ), row=1, col=i+1)
        fig.update_yaxes(title_text='Position (m)', row=1, col=i+1)
        fig.update_xaxes(title_text='Time (s)', row=1, col=i+1)
    
    # Row 2: GRF
    for i, name in enumerate(['X', 'Y', 'Z']):
        idx = grf_indices[i]
        fig.add_trace(go.Scatter(
            x=time_axis, y=ground_truth['grf'][:, idx],
            name='Ground Truth', line=dict(color=colors['gt'], width=2),
            legendgroup='gt', showlegend=False,
        ), row=2, col=i+1)
        fig.add_trace(go.Scatter(
            x=time_axis, y=predictions['grf'][:, idx],
            name=pred_label, line=dict(color=colors['pred'], width=2, dash='dash'),
            legendgroup='pred', showlegend=False,
        ), row=2, col=i+1)
        if predictions_alt is not None:
            fig.add_trace(go.Scatter(
                x=time_axis, y=predictions_alt['grf'][:, idx],
                name=alt_pred_label, line=dict(color=colors['pred_alt'], width=2, dash='dot'),
                legendgroup='pred_alt', showlegend=False,
            ), row=2, col=i+1)
        fig.update_yaxes(title_text='Force (N)', row=2, col=i+1)
        fig.update_xaxes(title_text='Time (s)', row=2, col=i+1)
    
    # Compute RMSE metrics for this side
    # COP RMSE (only X, Z channels)
    cop_rmse = _masked_rmse(
        metric_predictions['cop'][:, cop_indices],
        ground_truth['cop'][:, cop_indices],
        evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
    )
    
    # GRF RMSE (X, Y, Z)
    grf_rmse = _masked_rmse(
        metric_predictions['grf'][:, grf_indices],
        ground_truth['grf'][:, grf_indices],
        evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
    )

    rmse_suffix = ""
    if metric_predictions_alt is not None:
        cop_rmse_alt = _masked_rmse(
            metric_predictions_alt['cop'][:, cop_indices],
            ground_truth['cop'][:, cop_indices],
            evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
        )
        grf_rmse_alt = _masked_rmse(
            metric_predictions_alt['grf'][:, grf_indices],
            ground_truth['grf'][:, grf_indices],
            evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
        )
        rmse_suffix = (
            f' | {alt_pred_label}: COP RMSE {cop_rmse_alt:.4f} m, GRF RMSE {grf_rmse_alt:.1f} N'
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Inference Results ({side}): {trial_name}</b><br>'
                 f'<span style="font-size:12px">{pred_label}: COP RMSE {cop_rmse:.4f} m, GRF RMSE {grf_rmse:.1f} N'
                 f'{rmse_suffix}</span>',
            x=0.5,
            y=0.98,
            font=dict(size=16),
        ),
        height=800,
        width=1400,
        margin=dict(t=100, b=60, l=60, r=60),
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.1,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        hovermode='x unified',
    )
    _add_prediction_margin_shading(fig, time_axis, prediction_margin_frames)
    
    if save_path:
        fig.write_html(save_path)
        print(f"💾 Saved {side} interactive plot to: {save_path}")
    
    return fig


def get_dof_names() -> List[str]:
    """Get DOF names in the independent model-space order."""
    return list(MODEL_DOF_NAMES)


def get_dof_display_names(target_width: Optional[int] = None) -> List[str]:
    """Return DOF display names aligned to a specific torque width.

    The base model uses a 23-channel independent full-ID layout; stance analysis
    can append left/right KAM channels on top of that.
    """
    dof_names = list(get_dof_names())
    if target_width is None:
        return dof_names

    target_width = int(target_width)
    if target_width <= len(dof_names):
        return dof_names[:target_width]

    display_names = list(dof_names)
    extra_names = [
        LEFT_STANCE_KAM_DOF_NAME,
        "knee_adduction_moment_r",
    ]
    for extra_name in extra_names:
        if len(display_names) >= target_width:
            break
        display_names.append(extra_name)
    while len(display_names) < target_width:
        display_names.append(f"DOF_{len(display_names)}")
    return display_names


def get_selected_left_stance_dof_indices() -> List[int]:
    """Return the torque DOF indices included in the left-stance accuracy metrics."""
    dof_names = get_dof_names()
    name_to_idx = {name: idx for idx, name in enumerate(dof_names)}
    missing_names = [
        dof_name for dof_name in SELECTED_LEFT_STANCE_DOF_NAMES
        if dof_name not in name_to_idx
    ]
    if missing_names:
        raise KeyError(
            "Selected left-stance DOFs are missing from get_dof_names(): "
            + ", ".join(missing_names)
        )
    return [name_to_idx[dof_name] for dof_name in SELECTED_LEFT_STANCE_DOF_NAMES]


def get_left_stance_mask(
    grf_raw: np.ndarray,
    evaluation_mask: Optional[np.ndarray] = None,
    threshold: float = LEFT_STANCE_THRESHOLD_N,
) -> np.ndarray:
    """Return valid evaluation frames where the left foot is in stance."""
    grf_raw = np.asarray(grf_raw)
    valid_eval_mask = (
        _normalize_evaluation_mask(evaluation_mask, len(grf_raw))
        if evaluation_mask is not None
        else np.ones(len(grf_raw), dtype=bool)
    )
    if grf_raw.ndim != 2 or grf_raw.shape[1] <= 5:
        raise ValueError(
            "Expected grf_raw with shape (frames, >=6) for left-stance detection."
    )
    return (np.abs(grf_raw[:, 5]) > float(threshold)) & valid_eval_mask


def _coerce_full_id_width(
    values: Optional[np.ndarray],
    target_width: int,
    *,
    label: str,
    fill_value: float = 0.0,
) -> Optional[np.ndarray]:
    """Project/pad full-ID tensors onto a shared comparison width."""
    if values is None:
        return None

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be 2D for full-ID comparisons, got shape {arr.shape}")

    width = int(arr.shape[1])
    if width == int(target_width):
        return arr
    if width == 43 and int(target_width) == 23:
        return arr[:, MODEL_43_TO_INDEPENDENT_INDICES]
    if width == 31 and int(target_width) == 23:
        return arr[:, MODEL_31_TO_INDEPENDENT_INDICES]
    if width == 39 and int(target_width) == 23:
        return arr[:, MODEL_31_TO_INDEPENDENT_INDICES]
    if width == 31 and int(target_width) == 39:
        padded = np.full((arr.shape[0], 39), fill_value, dtype=np.float32)
        padded[:, :31] = arr
        return padded
    raise ValueError(f"Unsupported full-ID width conversion for {label}: {width} -> {target_width}")


def _resolve_full_id_target_width(*arrays: Optional[np.ndarray]) -> int:
    """Prefer the independent 23-DOF width when mixed full-ID layouts appear."""
    widths = {
        int(np.asarray(arr).shape[1])
        for arr in arrays
        if arr is not None and np.asarray(arr).ndim == 2
    }
    if not widths:
        return len(MODEL_DOF_NAMES)
    if widths.issubset({23, 31, 39, 43}):
        return 23
    if len(widths) == 1:
        return next(iter(widths))
    raise ValueError(f"Unable to resolve a common full-ID width from: {sorted(widths)}")


def resolve_full_id_reference_curves(
    ground_truth: Mapping[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Resolve the torque reference used for full-ID comparisons and plotting.

    Returns `(selected_gt, selected_label, mjx_gt, opensim_gt, opensim_mask)`.
    When `use_OpenSimID_GT` is enabled, OpenSim is used where available and MJX
    fills any missing DOFs.
    """
    mjx_gt = ground_truth.get("id_gt_mjx")
    opensim_gt = ground_truth.get("opensim_id_gt")
    opensim_mask = ground_truth.get("opensim_id_available_mask")
    use_opensim_id_gt = bool(ground_truth.get("use_OpenSimID_GT", False))
    use_recalculated_opensim_id_gt = bool(ground_truth.get("use_recalculated_opensim_id_gt", False))
    mjx_reference = ground_truth.get("mjx_id_reference")

    if opensim_mask is not None:
        opensim_mask = np.asarray(opensim_mask, dtype=bool).reshape(-1)

    target_width = _resolve_full_id_target_width(mjx_gt, opensim_gt)
    mjx_gt = _coerce_full_id_width(mjx_gt, target_width, label="MJX_ID", fill_value=np.nan)
    opensim_gt = _coerce_full_id_width(opensim_gt, target_width, label="OpenSim_ID_STO", fill_value=np.nan)
    if mjx_reference is not None:
        mjx_reference = _coerce_full_id_width(
            mjx_reference,
            target_width,
            label="MJX_ID_reference",
            fill_value=np.nan,
        )
    if opensim_mask is not None and len(opensim_mask) != target_width:
        if len(opensim_mask) == 43 and target_width == 23:
            opensim_mask = opensim_mask[MODEL_43_TO_INDEPENDENT_INDICES]
        elif len(opensim_mask) in (31, 39) and target_width == 23:
            opensim_mask = opensim_mask[MODEL_31_TO_INDEPENDENT_INDICES]
        elif len(opensim_mask) == 31 and target_width == 39:
            padded_mask = np.zeros((39,), dtype=bool)
            padded_mask[:31] = opensim_mask
            opensim_mask = padded_mask
        else:
            raise ValueError(
                "Unsupported OpenSim availability-mask width conversion: "
                f"{len(opensim_mask)} -> {target_width}"
            )

    if use_recalculated_opensim_id_gt:
        if opensim_gt is not None:
            return (
                np.asarray(opensim_gt),
                "Recalculated_OpenSim_ID",
                mjx_reference if mjx_reference is not None else mjx_gt,
                opensim_gt,
                opensim_mask,
            )
        if mjx_gt is not None:
            return (
                np.asarray(mjx_gt),
                "Recalculated_OpenSim_ID",
                mjx_reference,
                opensim_gt,
                opensim_mask,
            )

    if use_opensim_id_gt and opensim_gt is not None:
        if mjx_gt is not None and opensim_mask is not None and opensim_gt.shape[1] == mjx_gt.shape[1]:
            selected_gt = np.array(mjx_gt, copy=True)
            selected_gt[:, opensim_mask] = np.asarray(opensim_gt)[:, opensim_mask]
        else:
            selected_gt = np.asarray(opensim_gt)
        return selected_gt, "GT", mjx_gt, opensim_gt, opensim_mask

    if mjx_gt is not None:
        return np.asarray(mjx_gt), "MJX_ID", mjx_gt, opensim_gt, opensim_mask

    if opensim_gt is not None:
        return np.asarray(opensim_gt), "GT", mjx_gt, opensim_gt, opensim_mask

    return None, "GT", mjx_gt, opensim_gt, opensim_mask


def compute_full_id_curves(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    qfrc_inverse_override: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Build full-ID curves.
    
    Preferred OpenCap behavior:
    - full_id_gt   = ID_GT_MJX
    - full_id_pred = qfrc_inverse(selected input source) - tau_pred
    """
    tau_pred = predictions.get("tau_grf")
    tau_gt = ground_truth.get("tau_grf")
    if tau_pred is None or tau_gt is None:
        return None, None, "missing_tau"

    qfrc_inverse = qfrc_inverse_override if qfrc_inverse_override is not None else predictions.get("qfrc_inverse")
    if qfrc_inverse is None:
        qfrc_inverse = ground_truth.get("qfrc_inverse")
    selected_gt, selected_label, _mjx_gt, _opensim_gt, opensim_mask = resolve_full_id_reference_curves(
        ground_truth
    )

    target_width = _resolve_full_id_target_width(tau_pred, qfrc_inverse, selected_gt)
    tau_pred = _coerce_full_id_width(tau_pred, target_width, label="tau_grf_pred")
    qfrc_inverse = _coerce_full_id_width(qfrc_inverse, target_width, label="qfrc_inverse")
    selected_gt = _coerce_full_id_width(selected_gt, target_width, label=selected_label, fill_value=np.nan)
    if opensim_mask is not None and len(opensim_mask) != target_width:
        opensim_mask = opensim_mask[:target_width]

    if selected_gt is not None and qfrc_inverse is not None:
        tau_sign = float(ground_truth.get("full_id_tau_sign", -1.0))
        full_id_pred = qfrc_inverse + tau_sign * tau_pred
        full_id_gt = selected_gt
        if selected_label == "Recalculated_OpenSim_ID":
            return full_id_pred, full_id_gt, "Recalculated_OpenSim_ID"
        if selected_label == "GT":
            if opensim_mask is not None and np.any(~opensim_mask):
                return full_id_pred, full_id_gt, "OpenSim_ID_STO_with_MJX_fallback"
            return full_id_pred, full_id_gt, "OpenSim_ID_STO"
        sign_label = "plus_tau" if tau_sign >= 0 else "minus_tau"
        return full_id_pred, full_id_gt, f"ID_GT_MJX_with_qfrc_inverse_{sign_label}"

    if selected_gt is not None:
        return None, None, "missing_qfrc_inverse"




def create_joint_group_plot(
    time_axis: np.ndarray,
    predictions: Dict[str, np.ndarray],
    predictions_alt: Optional[Dict[str, np.ndarray]],
    ground_truth: Dict[str, np.ndarray],
    dof_names: List[str],
    joint_indices: List[int],
    group_name: str,
    trial_name: str,
    qfrc_inverse_pred: Optional[np.ndarray] = None,
    qfrc_inverse_alt: Optional[np.ndarray] = None,
    save_path: str = None,
    pred_label: str = "Prediction (OpenCap input)",
    alt_pred_label: str = "Prediction (MotionCapture input)",
    evaluation_mask: Optional[np.ndarray] = None,
    metric_predictions: Optional[Dict[str, np.ndarray]] = None,
    metric_predictions_alt: Optional[Dict[str, np.ndarray]] = None,
    prediction_margin_frames: int = 0,
) -> go.Figure:
    """Create a plot for a specific joint group with each joint as a separate subplot.
    
    Plots full-ID results using ID_GT_MJX when available.
    """
    
    n_joints = len(joint_indices)
    if n_joints == 0:
        return None
    
    # Compute full ID using the best available source.
    metric_predictions = metric_predictions if metric_predictions is not None else predictions
    metric_predictions_alt = (
        metric_predictions_alt if metric_predictions_alt is not None else predictions_alt
    )

    tau_grf_pred_full, tau_grf_gt_full, id_source = compute_full_id_curves(
        predictions, ground_truth, qfrc_inverse_override=qfrc_inverse_pred
    )
    if tau_grf_pred_full is None or tau_grf_gt_full is None:
        print(f"   ⚠️  Skipping {group_name} full-ID plot: missing torque arrays")
        return None
    tau_grf_pred_metric_full, _, _ = compute_full_id_curves(
        metric_predictions,
        ground_truth,
        qfrc_inverse_override=qfrc_inverse_pred,
    )
    tau_grf_pred_alt_full = None
    tau_grf_pred_alt_metric_full = None
    if predictions_alt is not None:
        tau_grf_pred_alt_full, _, _ = compute_full_id_curves(
            predictions_alt,
            ground_truth,
            qfrc_inverse_override=(
                qfrc_inverse_alt if qfrc_inverse_alt is not None else qfrc_inverse_pred
            ),
        )
    if metric_predictions_alt is not None:
        tau_grf_pred_alt_metric_full, _, _ = compute_full_id_curves(
            metric_predictions_alt,
            ground_truth,
            qfrc_inverse_override=(
                qfrc_inverse_alt if qfrc_inverse_alt is not None else qfrc_inverse_pred
            ),
        )

    # Determine grid layout (prefer 2 columns for better readability)
    n_cols = 2
    n_rows = (n_joints + n_cols - 1) // n_cols
    
    # Get joint names and compute RMSE for each joint (using full ID)
    joint_names = []
    joint_rmses = []
    for idx in joint_indices:
        joint_name = dof_names[idx]
        joint_rmse = _masked_rmse(
            tau_grf_pred_metric_full[:, idx],
            tau_grf_gt_full[:, idx],
            evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
        )
        joint_names.append(f"{joint_name}<br><span style='font-size:10px;color:gray'>RMSE: {joint_rmse:.2f} Nm</span>")
        joint_rmses.append(joint_rmse)
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=joint_names,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )
    
    colors = {'gt': '#2E86AB', 'pred': '#E94F37', 'pred_alt': '#1B9E77', 'mjx': '#6C757D'}
    
    _selected_gt, metric_reference_label, mjx_gt_full, opensim_gt_full, opensim_mask = (
        resolve_full_id_reference_curves(ground_truth)
    )

    # Plot each joint
    for plot_idx, dof_idx in enumerate(joint_indices):
        row = plot_idx // n_cols + 1
        col = plot_idx % n_cols + 1
        joint_name = dof_names[dof_idx]

        if opensim_gt_full is not None and (
            opensim_mask is None or (dof_idx < len(opensim_mask) and opensim_mask[dof_idx])
        ):
            fig.add_trace(go.Scatter(
                x=time_axis, y=np.asarray(opensim_gt_full)[:, dof_idx],
                name='GT', line=dict(color=colors['gt'], width=2),
                legendgroup='gt', showlegend=(plot_idx == 0),
                hovertemplate=f'<b>{joint_name}</b><br>Time: %{{x:.2f}}s<br>GT: %{{y:.2f}} Nm<extra></extra>',
            ), row=row, col=col)

        if mjx_gt_full is not None:
            fig.add_trace(go.Scatter(
                x=time_axis, y=np.asarray(mjx_gt_full)[:, dof_idx],
                name='MJX_ID', line=dict(color=colors['mjx'], width=2, dash='dot'),
                legendgroup='mjx', showlegend=(plot_idx == 0),
                hovertemplate=f'<b>{joint_name}</b><br>Time: %{{x:.2f}}s<br>MJX_ID: %{{y:.2f}} Nm<extra></extra>',
            ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=time_axis, y=tau_grf_pred_full[:, dof_idx],
            name=pred_label, line=dict(color=colors['pred'], width=2, dash='dash'),
            legendgroup='pred', showlegend=(plot_idx == 0),
            hovertemplate=f'<b>{joint_name}</b><br>Time: %{{x:.2f}}s<br>Pred Full ID: %{{y:.2f}} Nm<extra></extra>',
        ), row=row, col=col)
        if tau_grf_pred_alt_full is not None:
            fig.add_trace(go.Scatter(
                x=time_axis, y=tau_grf_pred_alt_full[:, dof_idx],
                name=alt_pred_label, line=dict(color=colors['pred_alt'], width=2, dash='dot'),
                legendgroup='pred_alt', showlegend=(plot_idx == 0),
                hovertemplate=f'<b>{joint_name}</b><br>Time: %{{x:.2f}}s<br>Alt Pred Full ID: %{{y:.2f}} Nm<extra></extra>',
            ), row=row, col=col)
        
        # Update axes - add labels to all subplots
        fig.update_xaxes(
            title_text='Time (s)',
            row=row, col=col,
            showticklabels=True
        )
        fig.update_yaxes(
            title_text='Torque (Nm)',
            row=row, col=col,
            showticklabels=True
        )
    
    # Compute overall RMSE for this group (using full ID)
    group_rmse = _masked_rmse(
        tau_grf_pred_metric_full[:, joint_indices],
        tau_grf_gt_full[:, joint_indices],
        evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
    )
    rmse_suffix = ""
    if tau_grf_pred_alt_metric_full is not None:
        group_rmse_alt = _masked_rmse(
            tau_grf_pred_alt_metric_full[:, joint_indices],
            tau_grf_gt_full[:, joint_indices],
            evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
        )
        rmse_suffix = f" | {alt_pred_label} RMSE: {group_rmse_alt:.2f} Nm"
    if str(id_source).startswith("OpenSim_ID_STO"):
        source_label = f"Full ID metric reference: {metric_reference_label} (OpenSim STO)"
    elif id_source == "ID_GT_MJX_with_qfrc_inverse":
        source_label = f"Full ID metric reference: {metric_reference_label}"
    else:
        source_label = f"Full ID source: {id_source}"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{group_name} - Full ID: {trial_name}</b><br>'
                 f'<span style="font-size:12px">{pred_label} RMSE: {group_rmse:.2f} Nm{rmse_suffix} | {source_label}</span>',
            x=0.5, y=0.98,
        ),
        height=200 + 200 * n_rows,
        width=1200,
        margin=dict(t=100, b=40, l=60, r=30),
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        hovermode='x unified',
    )
    _add_prediction_margin_shading(fig, time_axis, prediction_margin_frames)
    
    if save_path:
        fig.write_html(save_path)
        print(f"💾 Saved {group_name} plot to: {save_path}")
    
    return fig


def create_all_dofs_plot(
    time_axis: np.ndarray,
    predictions: Dict[str, np.ndarray],
    predictions_alt: Optional[Dict[str, np.ndarray]],
    ground_truth: Dict[str, np.ndarray],
    trial_name: str,
    qfrc_inverse_pred: Optional[np.ndarray] = None,
    qfrc_inverse_alt: Optional[np.ndarray] = None,
    save_path: str = None,
    pred_label: str = "Prediction (OpenCap input)",
    alt_pred_label: str = "Prediction (MotionCapture input)",
    evaluation_mask: Optional[np.ndarray] = None,
    metric_predictions: Optional[Dict[str, np.ndarray]] = None,
    metric_predictions_alt: Optional[Dict[str, np.ndarray]] = None,
    prediction_margin_frames: int = 0,
) -> go.Figure:
    """Create interactive Plotly visualization for all full-ID DOFs."""

    metric_predictions = metric_predictions if metric_predictions is not None else predictions
    metric_predictions_alt = (
        metric_predictions_alt if metric_predictions_alt is not None else predictions_alt
    )
    full_id_pred, full_id_gt, id_source = compute_full_id_curves(
        predictions, ground_truth, qfrc_inverse_override=qfrc_inverse_pred
    )
    if full_id_pred is None or full_id_gt is None:
        print("   ⚠️  Skipping all DOFs full-ID plot: missing torque arrays")
        return None
    full_id_metric_pred, _, _ = compute_full_id_curves(
        metric_predictions,
        ground_truth,
        qfrc_inverse_override=qfrc_inverse_pred,
    )
    full_id_pred_alt = None
    full_id_metric_pred_alt = None
    if predictions_alt is not None:
        full_id_pred_alt, _, _ = compute_full_id_curves(
            predictions_alt,
            ground_truth,
            qfrc_inverse_override=(qfrc_inverse_alt if qfrc_inverse_alt is not None else qfrc_inverse_pred),
        )
    if metric_predictions_alt is not None:
        full_id_metric_pred_alt, _, _ = compute_full_id_curves(
            metric_predictions_alt,
            ground_truth,
            qfrc_inverse_override=(qfrc_inverse_alt if qfrc_inverse_alt is not None else qfrc_inverse_pred),
        )

    n_dofs = full_id_pred.shape[1]
    n_cols = 8
    n_rows = (n_dofs + n_cols - 1) // n_cols
    
    # Get DOF names
    dof_names = get_dof_names()
    # Truncate long names for subplot titles (max 20 chars)
    subplot_titles = [name[:20] + '...' if len(name) > 20 else name for name in dof_names[:n_dofs]]
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )
    
    colors = {'gt': '#2E86AB', 'mjx': '#6C757D', 'pred': '#E94F37', 'pred_alt': '#1B9E77'}
    _selected_gt, metric_reference_label, mjx_gt_full, opensim_gt_full, opensim_mask = (
        resolve_full_id_reference_curves(ground_truth)
    )
    
    # Plot all DOFs
    for dof_idx in range(n_dofs):
        row = dof_idx // n_cols + 1
        col = dof_idx % n_cols + 1
        dof_name = dof_names[dof_idx] if dof_idx < len(dof_names) else f'DOF_{dof_idx}'
        
        if opensim_gt_full is not None and (
            opensim_mask is None or (dof_idx < len(opensim_mask) and opensim_mask[dof_idx])
        ):
            fig.add_trace(go.Scatter(
                x=time_axis, y=np.asarray(opensim_gt_full)[:, dof_idx],
                name='GT', line=dict(color=colors['gt'], width=1.5),
                legendgroup='gt', showlegend=(dof_idx == 0),
                hovertemplate=f'<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>GT: %{{y:.2f}} Nm<extra></extra>',
            ), row=row, col=col)

        if mjx_gt_full is not None:
            fig.add_trace(go.Scatter(
                x=time_axis, y=np.asarray(mjx_gt_full)[:, dof_idx],
                name='MJX_ID', line=dict(color=colors['mjx'], width=1.5, dash='dot'),
                legendgroup='mjx', showlegend=(dof_idx == 0),
                hovertemplate=f'<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>MJX_ID: %{{y:.2f}} Nm<extra></extra>',
            ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=time_axis, y=full_id_pred[:, dof_idx],
            name=pred_label, line=dict(color=colors['pred'], width=1.5, dash='dash'),
            legendgroup='pred', showlegend=(dof_idx == 0),
            hovertemplate=f'<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>Pred Full ID: %{{y:.2f}} Nm<extra></extra>',
        ), row=row, col=col)
        if full_id_pred_alt is not None:
            fig.add_trace(go.Scatter(
                x=time_axis, y=full_id_pred_alt[:, dof_idx],
                name=alt_pred_label, line=dict(color=colors['pred_alt'], width=1.5, dash='dot'),
                legendgroup='pred_alt', showlegend=(dof_idx == 0),
                hovertemplate=f'<b>{dof_name}</b><br>Time: %{{x:.2f}}s<br>Alt Pred Full ID: %{{y:.2f}} Nm<extra></extra>',
            ), row=row, col=col)
        
        # Update axes - add labels to all subplots
        fig.update_xaxes(
            title_text='Time (s)',
            row=row, col=col,
            showticklabels=True
        )
        fig.update_yaxes(
            title_text='Torque (Nm)',
            row=row, col=col,
            showticklabels=True
        )
    
    # Compute RMSE for all DOFs
    tau_rmse = _masked_rmse(
        full_id_metric_pred,
        full_id_gt,
        evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
    )
    tau_rmse_per_dof = _masked_rmse_per_channel(
        full_id_metric_pred,
        full_id_gt,
        evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
    )
    tau_rmse_alt = None
    if full_id_metric_pred_alt is not None:
        tau_rmse_alt = _masked_rmse(
            full_id_metric_pred_alt,
            full_id_gt,
            evaluation_mask if evaluation_mask is not None else np.ones(len(time_axis), dtype=bool),
        )
    
    # Get DOF names for min/max RMSE
    min_dof_idx = tau_rmse_per_dof.argmin()
    max_dof_idx = tau_rmse_per_dof.argmax()
    min_dof_name = dof_names[min_dof_idx] if min_dof_idx < len(dof_names) else f'DOF_{min_dof_idx}'
    max_dof_name = dof_names[max_dof_idx] if max_dof_idx < len(dof_names) else f'DOF_{max_dof_idx}'
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>All Full-ID DOFs: {trial_name}</b><br>'
                 f'<span style="font-size:12px">{pred_label} RMSE: {tau_rmse:.1f} Nm'
                 f'{"" if tau_rmse_alt is None else f" | {alt_pred_label} RMSE: {tau_rmse_alt:.1f} Nm"} | '
                 f'Min RMSE: {tau_rmse_per_dof.min():.1f} Nm ({min_dof_name}) | '
                 f'Max RMSE: {tau_rmse_per_dof.max():.1f} Nm ({max_dof_name}) | Ref: {metric_reference_label}</span>',
            x=0.5, y=0.98,
        ),
        height=150 + 120 * n_rows,
        width=1600,
        margin=dict(t=120, b=40, l=60, r=30),
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        hovermode='x unified',
    )
    _add_prediction_margin_shading(fig, time_axis, prediction_margin_frames)
    
    if save_path:
        fig.write_html(save_path)
        print(f"💾 Saved all DOFs plot to: {save_path}")
        
        # Generate individual joint group plots
        # Find indices for each joint group by exact name matching
        knee_indices = []
        ankle_indices = []
        hip_indices = []
        lumbar_indices = []
        
        # Create name to index mapping
        name_to_idx = {name: i for i, name in enumerate(dof_names)}
        
        # Knee joints
        for name in ['knee_angle_r', 'knee_angle_l']:
            if name in name_to_idx:
                idx = name_to_idx[name]
                if idx < n_dofs:
                    knee_indices.append(idx)
        
        # Ankle and foot joints
        for name in ['ankle_angle_r', 'ankle_angle_l', 
                     'subtalar_angle_r', 'subtalar_angle_l',
                     'mtp_angle_r', 'mtp_angle_l']:
            if name in name_to_idx:
                idx = name_to_idx[name]
                if idx < n_dofs:
                    ankle_indices.append(idx)
        
        # Hip joints
        for name in ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                     'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l']:
            if name in name_to_idx:
                idx = name_to_idx[name]
                if idx < n_dofs:
                    hip_indices.append(idx)

        # Lumbar joints
        for name in ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']:
            if name in name_to_idx:
                idx = name_to_idx[name]
                if idx < n_dofs:
                    lumbar_indices.append(idx)
        
        # Generate file paths based on save_path
        base_path = Path(save_path)
        base_dir = base_path.parent
        base_stem = base_path.stem  # filename without extension
        
        # Create knee joints plot
        if knee_indices:
            knee_path = base_dir / f"{base_stem}_knee_joints.html"
            create_joint_group_plot(
                time_axis, predictions, predictions_alt, ground_truth, dof_names,
                knee_indices, "Knee Joints", trial_name,
                qfrc_inverse_pred=qfrc_inverse_pred,
                qfrc_inverse_alt=qfrc_inverse_alt,
                save_path=str(knee_path),
                pred_label=pred_label,
                alt_pred_label=alt_pred_label,
                evaluation_mask=evaluation_mask,
                metric_predictions=metric_predictions,
                metric_predictions_alt=metric_predictions_alt,
                prediction_margin_frames=prediction_margin_frames,
            )
        
        # Create ankle joints plot
        if ankle_indices:
            ankle_path = base_dir / f"{base_stem}_ankle_joints.html"
            create_joint_group_plot(
                time_axis, predictions, predictions_alt, ground_truth, dof_names,
                ankle_indices, "Ankle & Foot Joints", trial_name,
                qfrc_inverse_pred=qfrc_inverse_pred,
                qfrc_inverse_alt=qfrc_inverse_alt,
                save_path=str(ankle_path),
                pred_label=pred_label,
                alt_pred_label=alt_pred_label,
                evaluation_mask=evaluation_mask,
                metric_predictions=metric_predictions,
                metric_predictions_alt=metric_predictions_alt,
                prediction_margin_frames=prediction_margin_frames,
            )
        
        # Create hip joints plot
        if hip_indices:
            hip_path = base_dir / f"{base_stem}_hip_joints.html"
            create_joint_group_plot(
                time_axis, predictions, predictions_alt, ground_truth, dof_names,
                hip_indices, "Hip Joints", trial_name,
                qfrc_inverse_pred=qfrc_inverse_pred,
                qfrc_inverse_alt=qfrc_inverse_alt,
                save_path=str(hip_path),
                pred_label=pred_label,
                alt_pred_label=alt_pred_label,
                evaluation_mask=evaluation_mask,
                metric_predictions=metric_predictions,
                metric_predictions_alt=metric_predictions_alt,
                prediction_margin_frames=prediction_margin_frames,
            )

        # Create lumbar joints plot
        if lumbar_indices:
            lumbar_path = base_dir / f"{base_stem}_lumbar_joints.html"
            create_joint_group_plot(
                time_axis, predictions, predictions_alt, ground_truth, dof_names,
                lumbar_indices, "Lumbar Joints", trial_name,
                qfrc_inverse_pred=qfrc_inverse_pred,
                qfrc_inverse_alt=qfrc_inverse_alt,
                save_path=str(lumbar_path),
                pred_label=pred_label,
                alt_pred_label=alt_pred_label,
                evaluation_mask=evaluation_mask,
                metric_predictions=metric_predictions,
                metric_predictions_alt=metric_predictions_alt,
                prediction_margin_frames=prediction_margin_frames,
            )
    
    return fig


def save_model_info(
    model: KinematicsToCOPGRFMoments,
    params: Dict,
    normalizers: Dict,
    checkpoint_path: str,
    train_trials: Optional[List] = None,
    val_trials: Optional[List] = None,
    save_path: str = None,
    output_reg_weight: Optional[float] = None,
) -> None:
    """Save comprehensive model information to a text file."""
    
    info_lines = []
    info_lines.append("=" * 80)
    info_lines.append("MODEL INFORMATION")
    info_lines.append("=" * 80)
    info_lines.append("")
    
    # Model Architecture
    info_lines.append("MODEL ARCHITECTURE")
    info_lines.append("-" * 80)
    info_lines.append(f"Model Type: KinematicsToCOPGRFMoments")
    info_lines.append(f"Input Dimension: {model.input_dim}")
    info_lines.append(f"Output Dimension: {model.output_dim}")
    info_lines.append(f"  - COP: 4 (right_xy, left_xy) [Normalized by height]")
    info_lines.append(f"  - GRF: 6 (right_xyz, left_xyz) [Normalized by mass]")
    info_lines.append(f"  - Free Moments: 2 (right_z, left_z) [Normalized by mass]")
    info_lines.append(f"d_model: {model.d_model}")
    info_lines.append(f"num_heads: {model.num_heads}")
    info_lines.append(f"num_layers: {model.num_layers}")
    info_lines.append(f"ff_dim: {model.ff_dim}")
    info_lines.append(f"dropout_rate: {model.dropout_rate}")
    if output_reg_weight is not None:
        info_lines.append(f"output_reg_weight (train): {output_reg_weight}")
    info_lines.append("")
    
    # Detailed Architecture Flow
    info_lines.append("DETAILED ARCHITECTURE FLOW")
    info_lines.append("-" * 80)
    info_lines.append("")
    info_lines.append("1. INPUT PROJECTION")
    info_lines.append("   - Dense(input_dim={} → d_model={})".format(model.input_dim, model.d_model))
    info_lines.append("   - Projects temporal feature vector to {} dimensions".format(model.d_model))
    info_lines.append("")
    info_lines.append("2. POSITIONAL ENCODING")
    info_lines.append("   - SinusoidalPosEmb(dim={})".format(model.d_model))
    info_lines.append("   - Adds sinusoidal positional embeddings to input")
    info_lines.append("   - Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))")
    info_lines.append("             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
    info_lines.append("   - Element-wise addition with projected input")
    info_lines.append("")
    info_lines.append("3. TRANSFORMER BLOCKS ({} layers)".format(model.num_layers))
    info_lines.append("   " + "-" * 76)
    for i in range(model.num_layers):
        info_lines.append("   Block {}:".format(i+1))
        info_lines.append("   ┌─────────────────────────────────────────────────────────────┐")
        info_lines.append("   │ 3.1. MULTI-HEAD SELF-ATTENTION                              │")
        info_lines.append("   │   ├─ LayerNorm (normalize input)                            │")
        info_lines.append("   │   ├─ MultiHeadDotProductAttention:                         │")
        info_lines.append("   │   │   ├─ num_heads: {}                                      │".format(model.num_heads))
        info_lines.append("   │   │   ├─ qkv_features: {} (Q, K, V dim per head: {})       │".format(
            model.d_model, model.d_model // model.num_heads))
        info_lines.append("   │   │   ├─ dropout_rate: {}                                   │".format(model.dropout_rate))
        info_lines.append("   │   │   └─ Attention: Q @ K^T / sqrt(d_k) @ V                │")
        info_lines.append("   │   └─ Residual connection: x = x + attention(x)              │")
        info_lines.append("   │                                                             │")
        info_lines.append("   │ 3.2. FEED-FORWARD NETWORK                                   │")
        info_lines.append("   │   ├─ LayerNorm (normalize input)                            │")
        info_lines.append("   │   ├─ Dense({} → {})                                        │".format(model.d_model, model.ff_dim))
        info_lines.append("   │   ├─ GELU activation                                        │")
        info_lines.append("   │   ├─ Dropout(rate={})                                       │".format(model.dropout_rate))
        info_lines.append("   │   ├─ Dense({} → {})                                        │".format(model.ff_dim, model.d_model))
        info_lines.append("   │   ├─ Dropout(rate={})                                       │".format(model.dropout_rate))
        info_lines.append("   │   └─ Residual connection: x = x + ffn(x)                    │")
        info_lines.append("   └─────────────────────────────────────────────────────────────┘")
        if i < model.num_layers - 1:
            info_lines.append("")
    info_lines.append("")
    info_lines.append("4. OUTPUT LAYER")
    info_lines.append("   - LayerNorm (final normalization)")
    info_lines.append("   - Dense(d_model={} → output_dim={})".format(model.d_model, model.output_dim))
    info_lines.append("")
    info_lines.append("5. OUTPUT MASKING")
    info_lines.append("   - COP Z components (indices 2, 5) forced to 0")
    info_lines.append("   - Mask: [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]")
    info_lines.append("")
    
    # Transformer Block Details
    info_lines.append("TRANSFORMER BLOCK DETAILS")
    info_lines.append("-" * 80)
    info_lines.append("Each TransformerBlock contains:")
    info_lines.append("")
    info_lines.append("  A. Multi-Head Self-Attention:")
    info_lines.append("     - Input shape: (batch, seq_len, d_model={})".format(model.d_model))
    info_lines.append("     - Number of heads: {}".format(model.num_heads))
    info_lines.append("     - Dimension per head: {} / {} = {}".format(
        model.d_model, model.num_heads, model.d_model // model.num_heads))
    info_lines.append("     - Attention mechanism:")
    info_lines.append("       1. Q = X @ W_q, K = X @ W_k, V = X @ W_v")
    info_lines.append("       2. Split into {} heads".format(model.num_heads))
    info_lines.append("       3. Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V")
    info_lines.append("       4. Concatenate heads and project")
    info_lines.append("     - Dropout applied during training")
    info_lines.append("     - Residual connection: output = input + attention(input)")
    info_lines.append("")
    info_lines.append("  B. Feed-Forward Network:")
    info_lines.append("     - Two linear transformations with expansion:")
    info_lines.append("       Linear1: {} → {} (expansion factor: {:.1f}x)".format(
        model.d_model, model.ff_dim, model.ff_dim / model.d_model))
    info_lines.append("       Activation: GELU (Gaussian Error Linear Unit)")
    info_lines.append("       Linear2: {} → {} (projection back)".format(model.ff_dim, model.d_model))
    info_lines.append("     - Dropout applied after each linear layer")
    info_lines.append("     - Residual connection: output = input + ffn(input)")
    info_lines.append("")
    info_lines.append("  C. Layer Normalization:")
    info_lines.append("     - Applied before attention and before FFN")
    info_lines.append("     - Normalizes across feature dimension (d_model)")
    info_lines.append("     - Formula: (x - mean(x)) / sqrt(var(x) + eps)")
    info_lines.append("")
    info_lines.append("  D. Residual Connections:")
    info_lines.append("     - Two residual connections per block")
    info_lines.append("     - Helps with gradient flow and training stability")
    info_lines.append("")
    
    # Parameter Counts
    info_lines.append("PARAMETER COUNTS")
    info_lines.append("-" * 80)
    total_params = 0
    for layer_name, layer_params in params.items():
        layer_count = sum(p.size for p in jax.tree_util.tree_leaves(layer_params))
        total_params += layer_count
        info_lines.append(f"  {layer_name}: {layer_count:,} parameters")
    info_lines.append(f"Total Parameters: {total_params:,}")
    info_lines.append(f"Total Parameters (MB): {total_params * 4 / (1024**2):.2f}")
    info_lines.append("")
    
    # Normalizer Statistics
    info_lines.append("NORMALIZER STATISTICS")
    info_lines.append("-" * 80)
    for key, norm in normalizers.items():
        mean = norm.mean
        std = norm.std
        if mean.ndim == 0:
            info_lines.append(f"{key}:")
            info_lines.append(f"  Mean: {mean:.6f}")
            info_lines.append(f"  Std:  {std:.6f}")
        else:
            info_lines.append(f"{key}:")
            info_lines.append(f"  Shape: {mean.shape}")
            info_lines.append(f"  Mean range: [{mean.min():.6f}, {mean.max():.6f}]")
            info_lines.append(f"  Std range:  [{std.min():.6f}, {std.max():.6f}]")
            info_lines.append(f"  Mean (first 5): {mean.flatten()[:5]}")
            info_lines.append(f"  Std (first 5):  {std.flatten()[:5]}")
        info_lines.append("")
    
    # Training Split Info
    info_lines.append("TRAINING DATA SPLIT")
    info_lines.append("-" * 80)
    if train_trials is not None and val_trials is not None:
        info_lines.append(f"Train trials: {len(train_trials)}")
        info_lines.append(f"Validation trials: {len(val_trials)}")
        info_lines.append(f"Total trials: {len(train_trials) + len(val_trials)}")
        info_lines.append(f"Train ratio: {len(train_trials) / (len(train_trials) + len(val_trials)):.2%}")
    else:
        info_lines.append("Split information not available")
    info_lines.append("")
    
    # Checkpoint Info
    info_lines.append("CHECKPOINT INFORMATION")
    info_lines.append("-" * 80)
    info_lines.append(f"Checkpoint path: {checkpoint_path}")
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        size_mb = checkpoint_file.stat().st_size / (1024**2)
        info_lines.append(f"Checkpoint size: {size_mb:.2f} MB")
        mtime = checkpoint_file.stat().st_mtime
        info_lines.append(f"Last modified: {datetime.fromtimestamp(mtime)}")
    info_lines.append("")
    
    # Model Output Structure
    info_lines.append("OUTPUT STRUCTURE")
    info_lines.append("-" * 80)
    info_lines.append("Indices 0-5:   COP [right_x, right_y, right_z, left_x, left_y, left_z]")
    info_lines.append("  Note: COP Z (indices 2, 5) are forced to 0 (ground plane)")
    info_lines.append("Indices 6-11:  GRF [right_x, right_y, right_z, left_x, left_y, left_z]")
    info_lines.append("Indices 12-17: Free Moments [right_x, right_y, right_z, left_x, left_y, left_z]")
    info_lines.append("")
    info_lines.append("Derived Output:")
    info_lines.append("  τ_grf: Joint torques from GRF (23 independent DOFs)")
    info_lines.append("    Computed via: τ_grf = Jp^T @ GRF + Jr^T @ M_full")
    info_lines.append("")
    
    # Physics
    info_lines.append("PHYSICS COMPUTATION")
    info_lines.append("-" * 80)
    info_lines.append("The model predicts COP, GRF, and free moments from kinematics.")
    info_lines.append("Joint torques (τ_grf) are computed using contact Jacobians:")
    info_lines.append("  - Jp: Position Jacobian at contact points")
    info_lines.append("  - Jr: Rotation Jacobian at contact points")
    info_lines.append("  - M_full: Full external torques from External_Force.npy")
    info_lines.append("")
    
    info_lines.append("=" * 80)
    
    # Write to file
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(info_lines))
        print(f"💾 Saved model info to: {save_path}")


def create_error_distribution_plot(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    trial_name: str,
    save_path: str = None,
    evaluation_mask: Optional[np.ndarray] = None,
    metric_predictions: Optional[Dict[str, np.ndarray]] = None,
) -> go.Figure:
    """Create error distribution plots."""
    metric_predictions = metric_predictions if metric_predictions is not None else predictions
    mask = (
        _normalize_evaluation_mask(evaluation_mask, len(metric_predictions['cop']))
        if evaluation_mask is not None
        else np.ones(len(metric_predictions['cop']), dtype=bool)
    )
    torque_err_name = 'τ_grf (Nm)'
    torque_err_values = (metric_predictions['tau_grf'][mask] - ground_truth['tau_grf'][mask]).flatten()
    full_id_metric_pred, full_id_metric_gt, full_id_source = compute_full_id_curves(
        metric_predictions,
        ground_truth,
    )
    if full_id_metric_pred is not None and full_id_metric_gt is not None:
        torque_err_name = 'Full ID Torque (Nm)'
        torque_err_values = (full_id_metric_pred[mask] - full_id_metric_gt[mask]).flatten()
    
    errors = {
        'COP (m)': (metric_predictions['cop'][mask] - ground_truth['cop'][mask]).flatten(),
        'GRF (N)': (metric_predictions['grf'][mask] - ground_truth['grf'][mask]).flatten(),
        'Moment (Nm)': (metric_predictions['moments'][mask] - ground_truth['moments'][mask]).flatten(),
        torque_err_name: torque_err_values,
    }
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(errors.keys()))
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = ['#2E86AB', '#E94F37']
    
    subplot_idx = 1
    for (name, err), (row, col), color in zip(errors.items(), positions, colors):
        rmse = np.sqrt(np.mean(err ** 2))
        fig.add_trace(go.Histogram(
            x=err, name=name, marker_color=color, opacity=0.7,
            nbinsx=50,
        ), row=row, col=col)
        fig.update_yaxes(title_text='Count', row=row, col=col)
        # Add RMSE annotation using paper coordinates
        x_pos = 0.25 if col == 1 else 0.75
        y_pos = 0.75 if row == 1 else 0.25
        fig.add_annotation(
            x=x_pos, y=y_pos, xref='paper', yref='paper',
            text=f'RMSE: {rmse:.4f}', showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='white', bordercolor='gray', borderwidth=1,
        )
        subplot_idx += 1
    
    fig.update_layout(
        title=dict(text=f'<b>Error Distributions: {trial_name}</b>', x=0.5, y=0.95),
        height=400, width=1000,
        margin=dict(t=100, b=60, l=60, r=60),
        template='plotly_white',
        showlegend=False,
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"💾 Saved error distribution to: {save_path}")
    
    return fig


def create_summary_dashboard(all_metrics: List[Dict], output_dir: str):
    """Create a comprehensive summary dashboard of all inference results."""
    
    print("\n📊 Generating summary dashboard...")
    summary_dir = os.path.join(output_dir, "summary_dashboard")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Convert to DataFrame for easier plotting if pandas is available, else use lists
    # We'll stick to Plotly for consistency
    
    trial_names = [m.get('trial_name', f'Trial {i}') for i, m in enumerate(all_metrics)]
    cop_rmses = [m['cop_rmse'] for m in all_metrics]
    grf_rmses = [m['grf_rmse'] for m in all_metrics]
    moments_rmses = [m['moments_rmse'] for m in all_metrics]
    torque_rmses = [m['torque_rmse'] for m in all_metrics]
    torque_nrmses = [m.get('torque_nrmse', 0.0) for m in all_metrics]
    torque_mae_bwhs = [m.get('torque_mae_bwh', np.nan) for m in all_metrics]
    dual_input_available = any('motioncapture_input_torque_mae_bwh' in m for m in all_metrics)
    if dual_input_available:
        opencap_mae_bwhs = [
            m.get('opencap_input_torque_mae_bwh', m.get('torque_mae_bwh', np.nan))
            for m in all_metrics
        ]
        motioncapture_mae_bwhs = [
            m.get('motioncapture_input_torque_mae_bwh', np.nan)
            for m in all_metrics
        ]
    
    # 1. Summary Table
    header_values = ['Trial', 'COP RMSE (m)', 'GRF RMSE (N)', 'Moments RMSE (Nm)', 'Torque RMSE (Nm)', 'Torque nRMSE', 'Torque MAE (%BW*H)']
    cell_values = [
        trial_names,
        [f"{x:.4f}" for x in cop_rmses],
        [f"{x:.2f}" for x in grf_rmses],
        [f"{x:.2f}" for x in moments_rmses],
        [f"{x:.2f}" for x in torque_rmses],
        [f"{x:.4f}" for x in torque_nrmses],
        [f"{x:.3f}" for x in torque_mae_bwhs],
    ]
    if dual_input_available:
        header_values.extend(['OpenCap MAE (%BW*H)', 'MotionCapture MAE (%BW*H)'])
        cell_values.extend([
            [f"{x:.3f}" for x in opencap_mae_bwhs],
            [f"{x:.3f}" for x in motioncapture_mae_bwhs],
        ])

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=header_values,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=cell_values,
                   fill_color='lavender',
                   align='left'))
    ])
    fig_table.update_layout(title='Summary Metrics per Trial')
    fig_table.write_html(os.path.join(summary_dir, "summary_table.html"))
    
    # 2. Distribution Plots (Box Plots)
    fig_box = make_subplots(rows=1, cols=4, subplot_titles=('COP RMSE', 'GRF RMSE', 'Moments RMSE', 'Torque RMSE'))
    
    fig_box.add_trace(go.Box(y=cop_rmses, name='COP', boxpoints='all', jitter=0.3, pointpos=-1.8), row=1, col=1)
    fig_box.add_trace(go.Box(y=grf_rmses, name='GRF', boxpoints='all', jitter=0.3, pointpos=-1.8), row=1, col=2)
    fig_box.add_trace(go.Box(y=moments_rmses, name='Moments', boxpoints='all', jitter=0.3, pointpos=-1.8), row=1, col=3)
    fig_box.add_trace(go.Box(y=torque_rmses, name='Torque', boxpoints='all', jitter=0.3, pointpos=-1.8), row=1, col=4)
    
    fig_box.update_yaxes(title_text="RMSE (m)", row=1, col=1)
    fig_box.update_yaxes(title_text="RMSE (N)", row=1, col=2)
    fig_box.update_yaxes(title_text="RMSE (Nm)", row=1, col=3)
    fig_box.update_yaxes(title_text="RMSE (Nm)", row=1, col=4)
    
    fig_box.update_layout(title_text="Distribution of RMSE Metrics across Validation Set", showlegend=False)
    fig_box.write_html(os.path.join(summary_dir, "metrics_distribution.html"))
    
    # 3. Bar Charts for each metric
    fig_bar = make_subplots(rows=4, cols=1, subplot_titles=('COP RMSE per Trial', 'GRF RMSE per Trial', 'Moments RMSE per Trial', 'Torque RMSE per Trial'),
                            vertical_spacing=0.05)
    
    fig_bar.add_trace(go.Bar(x=trial_names, y=cop_rmses, name='COP RMSE'), row=1, col=1)
    fig_bar.add_trace(go.Bar(x=trial_names, y=grf_rmses, name='GRF RMSE'), row=2, col=1)
    fig_bar.add_trace(go.Bar(x=trial_names, y=moments_rmses, name='Moments RMSE'), row=3, col=1)
    fig_bar.add_trace(go.Bar(x=trial_names, y=torque_rmses, name='Torque RMSE'), row=4, col=1)
    
    fig_bar.update_yaxes(title_text="RMSE (m)", row=1, col=1)
    fig_bar.update_yaxes(title_text="RMSE (N)", row=2, col=1)
    fig_bar.update_yaxes(title_text="RMSE (Nm)", row=3, col=1)
    fig_bar.update_yaxes(title_text="RMSE (Nm)", row=4, col=1)
    
    fig_bar.update_layout(height=1200, title_text="Per-Trial Performance Metrics", showlegend=False)
    fig_bar.write_html(os.path.join(summary_dir, "metrics_per_trial.html"))

    if dual_input_available:
        fig_compare = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                'Left-Stance Selected-DOF Torque MAE (%BW*H) per Trial',
                'Left-Stance Selected-DOF Torque RMSE (%BW*H) per Trial',
            ),
            vertical_spacing=0.14,
        )
        opencap_rmse_bwhs = [
            m.get('opencap_input_torque_rmse_bwh', m.get('torque_rmse_bwh', np.nan))
            for m in all_metrics
        ]
        motioncapture_rmse_bwhs = [
            m.get('motioncapture_input_torque_rmse_bwh', np.nan)
            for m in all_metrics
        ]

        fig_compare.add_trace(
            go.Bar(x=trial_names, y=opencap_mae_bwhs, name='OpenCap MAE', marker_color='#E94F37'),
            row=1, col=1,
        )
        fig_compare.add_trace(
            go.Bar(x=trial_names, y=motioncapture_mae_bwhs, name='MotionCapture MAE', marker_color='#1B9E77'),
            row=1, col=1,
        )
        fig_compare.add_trace(
            go.Bar(x=trial_names, y=opencap_rmse_bwhs, name='OpenCap RMSE', marker_color='#F4A261'),
            row=2, col=1,
        )
        fig_compare.add_trace(
            go.Bar(x=trial_names, y=motioncapture_rmse_bwhs, name='MotionCapture RMSE', marker_color='#2A9D8F'),
            row=2, col=1,
        )
        fig_compare.update_yaxes(title_text='%BW*H', row=1, col=1)
        fig_compare.update_yaxes(title_text='%BW*H', row=2, col=1)
        fig_compare.update_layout(
            height=900,
            title_text='OpenCap vs MotionCapture Left-Stance Selected-DOF Torque Error Summary',
            barmode='group',
        )
        fig_compare.write_html(os.path.join(summary_dir, "input_comparison_metrics.html"))
    
    # 4. Per-DOF Torque RMSE Distributions
    if 'torque_rmse_per_dof' in all_metrics[0]:
        dof_names = list(all_metrics[0].get('torque_metric_dof_names', get_dof_names()))
        num_dofs = len(all_metrics[0]['torque_rmse_per_dof'])
        # Ensure dof_names matches num_dofs
        if len(dof_names) > num_dofs:
            dof_names = dof_names[:num_dofs]
        elif len(dof_names) < num_dofs:
            dof_names.extend([f"DOF_{i}" for i in range(len(dof_names), num_dofs)])
            
        # Collect RMSEs for each DOF across all trials
        # Shape: (num_trials, num_dofs)
        all_dof_rmses = np.array([m['torque_rmse_per_dof'] for m in all_metrics])
        
        # Create subplots for histograms
        n_cols = 4
        n_rows = (num_dofs + n_cols - 1) // n_cols
        
        fig_dof_hist = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=dof_names,
                                     vertical_spacing=0.03, horizontal_spacing=0.03)
        
        for i in range(num_dofs):
            row = i // n_cols + 1
            col = i % n_cols + 1
            dof_rmse_data = all_dof_rmses[:, i]
            
            fig_dof_hist.add_trace(go.Histogram(
                x=dof_rmse_data, 
                name=dof_names[i],
                nbinsx=20,
                marker_color='#636EFA'
            ), row=row, col=col)
            
            # Add mean line
            mean_val = np.mean(dof_rmse_data)
            fig_dof_hist.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="red", 
                                   row=row, col=col, annotation_text=f"Avg: {mean_val:.2f}")
            
            fig_dof_hist.update_xaxes(title_text="RMSE (Nm)", row=row, col=col)
            fig_dof_hist.update_yaxes(title_text="Count", row=row, col=col)

        fig_dof_hist.update_layout(height=300*n_rows, title_text="Torque RMSE Distribution per DOF", showlegend=False)
        fig_dof_hist.write_html(os.path.join(summary_dir, "dof_rmse_distribution.html"))
    
    print(f"✅ Summary dashboard saved to: {summary_dir}")


# =============================================================================
# Main Inference
# =============================================================================

def run_inference(
    checkpoint_path: str,
    data_dir: str,
    trial_name: str,
    output_dir: str = "inference_results",
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    prediction_margin_frames: Optional[int] = None,
    d_model: Optional[int] = None,
    num_layers: Optional[int] = None,
    ff_dim: Optional[int] = None,
    no_plots: bool = False,
    lightweight: bool = False,
    make_graph: bool = False,
    use_noised: Optional[bool] = None,
    include_pelvis_euler: Optional[bool] = None,
    min_trial_length: int = 100,
    opencap_val_dataset: bool = False,
    input_source: str = "processed",
    restrict_max_vals: bool = False,
    restrict_max_vals_path: Optional[str] = None,
    use_OpenSimID_GT: bool = False,
    use_recalculated_opensim_id_gt: bool = False,
    use_grf_norm_cop: Optional[bool] = None,
    use_grf_nofilt: Optional[bool] = None,
    use_os_filtering: Optional[bool] = None,
    use_gt_jacob_and_rot: bool = False,
):
    """Run inference on a trial and generate visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    input_source_norm = str(input_source).strip().lower()
    input_source_display = "MotionCapture" if input_source_norm == "mocap" else "OpenCap"
    
    print("=" * 70)
    print("COP/GRF/Moments Inference")
    print("=" * 70)

    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    hyperparams_file = Path(checkpoint_path).parent / "hyperparameters.json"
    train_trials = None
    val_trials = None
    checkpoint = None
    params = None
    normalizers = None

    print(f"   Trial: {trial_name}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Hyperparameters file: {hyperparams_file} (exists={hyperparams_file.exists()})")
    print(f"   Output dir: {output_dir}")
    print(
        "   Requested inference mode: "
        f"input_source={input_source_norm}, opencap_val_dataset={opencap_val_dataset}, "
        f"lightweight={lightweight}, no_plots={no_plots}, use_OpenSimID_GT={use_OpenSimID_GT}, "
        f"use_recalculated_opensim_id_gt={use_recalculated_opensim_id_gt}, "
        f"use_gt_jacob_and_rot={use_gt_jacob_and_rot}"
    )

    if not Path(checkpoint_path).exists():
        print(f"   ❌ Checkpoint file does not exist: {checkpoint_path}")
        return None, None, None, None, None, None, None

    print("\n📦 Loading checkpoint bundle...")
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
        params = checkpoint["params"]
        normalizers = checkpoint["normalizers"]
        train_trials = checkpoint.get("train_trials")
        val_trials = checkpoint.get("val_trials")
        checkpoint_keys = sorted(checkpoint.keys())
        print(
            "   ✅ Checkpoint loaded: "
            f"{len(checkpoint_keys)} top-level keys, "
            f"sample={checkpoint_keys[:12]}{'...' if len(checkpoint_keys) > 12 else ''}"
        )
        print(f"   ✅ Normalizers available: {sorted(normalizers.keys())}")
    except Exception as exc:
        print(f"   ❌ Failed to load checkpoint bundle: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None, None, None, None, None, None, None

    if train_trials is None or val_trials is None:
        split_file = Path(checkpoint_path).parent / "train_val_split.json"
        if split_file.exists():
            try:
                with open(split_file, "r") as f:
                    split_info = json.load(f)
                train_trials = split_info.get("train_trials", train_trials)
                val_trials = split_info.get("val_trials", val_trials)
                print(
                    "   ✅ Loaded train/val split fallback from "
                    f"{split_file} (train={0 if train_trials is None else len(train_trials)}, "
                    f"val={0 if val_trials is None else len(val_trials)})"
                )
            except Exception as exc:
                print(f"   ⚠️ Could not load split fallback {split_file}: {type(exc).__name__}: {exc}")
        else:
            print(f"   ⚠️ No train_val_split.json found next to checkpoint: {split_file}")
    
    # Inject input source into data if it's not already there (used by KAM logic)
    # data is loaded later in the function, but we can prepare the info
    resolved_use_noised = _coerce_bool(use_noised, default=False)
    resolved_include_pelvis_euler = _coerce_bool(include_pelvis_euler, default=True)
    resolved_use_grf_norm_cop = _coerce_bool(use_grf_norm_cop, default=False)
    resolved_use_os_filtering = _coerce_bool(use_os_filtering, default=False)
    if hyperparams_file.exists():
        try:
            with open(hyperparams_file, 'r') as f:
                _hp_config = json.load(f)
            if use_noised is None and "UseNoised" in _hp_config:
                resolved_use_noised = _coerce_bool(_hp_config.get("UseNoised"), default=resolved_use_noised)
                print(f"   📐 UseNoised from checkpoint config: {resolved_use_noised}")
            if include_pelvis_euler is None and "includePelvisEuler" in _hp_config:
                resolved_include_pelvis_euler = _coerce_bool(
                    _hp_config.get("includePelvisEuler"),
                    default=resolved_include_pelvis_euler,
                )
                print(f"   📐 includePelvisEuler from checkpoint config: {resolved_include_pelvis_euler}")
            if use_grf_norm_cop is None and "UseGRFNormCOP" in _hp_config:
                resolved_use_grf_norm_cop = _coerce_bool(
                    _hp_config.get("UseGRFNormCOP"),
                    default=resolved_use_grf_norm_cop,
                )
                print(f"   📐 UseGRFNormCOP from checkpoint config: {resolved_use_grf_norm_cop}")
            if use_os_filtering is None and "UseOSFiltering" in _hp_config:
                resolved_use_os_filtering = _coerce_bool(
                    _hp_config.get("UseOSFiltering"),
                    default=resolved_use_os_filtering,
                )
                print(f"   📐 UseOSFiltering from checkpoint config: {resolved_use_os_filtering}")
        except Exception as e:
            print(f"   ⚠️ Could not read input flags from hyperparameters: {e}")

    if opencap_val_dataset and input_source_norm != "mocap" and resolved_use_noised:
        print(
            "   📐 OpenCapValDataset detected: forcing UseNoised=False for OpenCapSubjects inputs "
            "because that dataset should use clean ProcessedData kinematics."
        )
        resolved_use_noised = False

    if input_source_norm == "mocap":
        input_source_display = "MotionCapture"
    elif resolved_use_noised:
        input_source_display = "OpenCap (noised kinematics)"
    else:
        input_source_display = "OpenCap"
    
    # Load trial data
    print(f"\n📂 Loading trial: {trial_name}")
    
    # Determine if trial was in train or val set
    split_status = "unknown"
    if train_trials is not None and val_trials is not None:
        # Helper to extract trial name from entry (could be string or dict)
        def get_trial_name(entry):
            if isinstance(entry, dict):
                # Format: "subject/trial_name"
                subject = entry.get('subject', '')
                trial = entry.get('trial_name', entry.get('trial', ''))
                return f"{subject}/{trial}"
            return str(entry)
        
        # Normalize trial name for comparison
        trial_normalized = trial_name.lower().replace('/', '_').replace('\\', '_')
        
        # Exact matching with normalized strings
        train_names_normalized = [get_trial_name(t).lower().replace('/', '_').replace('\\', '_') for t in train_trials]
        val_names_normalized = [get_trial_name(t).lower().replace('/', '_').replace('\\', '_') for t in val_trials]
        
        if trial_normalized in train_names_normalized:
            split_status = "TRAIN"
        elif trial_normalized in val_names_normalized:
            split_status = "VALIDATION"
        else:
            # Fallback to substring matching if exact normalized match fails
            for t in train_trials:
                t_name = get_trial_name(t).lower()
                if trial_name.lower() in t_name or t_name in trial_name.lower():
                    split_status = "TRAIN"
                    break
            if split_status == "unknown":
                for t in val_trials:
                    t_name = get_trial_name(t).lower()
                    if trial_name.lower() in t_name or t_name in trial_name.lower():
                        split_status = "VALIDATION"
                        break
        
        print(f"   📊 Split status: {split_status} (train: {len(train_trials)}, val: {len(val_trials)})")
    else:
        print(f"   ⚠️  Train/val split info not saved in checkpoint")
    trial_match = find_trial(data_dir, trial_name)
    if trial_match is None:
        print(f"   ❌ Trial not found in data_dir={data_dir}: {trial_name}")
        return None, None, None, None, None, None, None

    trial_path, patient_path = trial_match
    data = load_trial_data(
        trial_path,
        opencap_val=opencap_val_dataset,
        input_source=input_source,
        use_noised=resolved_use_noised,
        use_grf_norm_cop=resolved_use_grf_norm_cop,
        use_grf_nofilt=use_grf_nofilt,
        use_os_filtering=resolved_use_os_filtering,
        use_recalculated_opensim_id_gt=use_recalculated_opensim_id_gt,
    )
    
    if data is not None:
        # Pass resolution metadata for KAM calculation
        data["input_source"] = input_source
        data["use_noised"] = resolved_use_noised
    
    if data is None:
        print("❌ Failed to load trial data")
        return None, None, None, None, None, None, None
    
    # Check trial length
    actual_length = len(data['pos'])
    if actual_length < min_trial_length:
        print(f"⚠️  Skipping trial {trial_name}: length {actual_length} < min_trial_length {min_trial_length}")
        return None, None, None, None, None, None, None
    
    print(f"   Subject mass: {data['mass'][0, 0]:.1f} kg")
    print(f"   Subject height: {data['height'][0, 0]:.2f} m")
    print(f"   Sequence length: {len(data['pos'])} frames")
    print(f"   Input source: {input_source} ({input_source_display})")
    print(f"   Prediction kinematics: {data.get('input_kinematics_source', 'Pos')}")
    print(f"   Ground-truth source: {data.get('ground_truth_source', 'selected input source')}")
    print(f"   GRF source: {data.get('grf_source_filename', 'GRF_Cleaned.npy')}")
    print(f"   COP target mode: {'GRFNorm COP' if resolved_use_grf_norm_cop else 'standard COP/height'}")
    opensim_id_bundle = load_opensim_id_ground_truth_bundle(
        trial_path,
        target_len=len(data["pos"]),
        use_recalculated=use_recalculated_opensim_id_gt,
    )
    if opensim_id_bundle is not None:
        if use_recalculated_opensim_id_gt:
            print(
                "   OpenSim ID GT: "
                f"{Path(opensim_id_bundle['source_path']).name} "
                "(recalculated MoCap kinematics + cleaned forces)"
            )
        else:
            print(
                "   OpenSim ID GT: "
                f"{Path(opensim_id_bundle['source_path']).name} "
                f"({len(opensim_id_bundle['available_columns'])} mapped torque columns)"
            )
    elif use_recalculated_opensim_id_gt:
        print(
            "   ⚠️ use_recalculated_opensim_id_gt requested, but "
            "MoCap/OpenSim_ID_recalculated.npy could not be loaded."
        )
    elif use_OpenSimID_GT:
        print("   ⚠️ use_OpenSimID_GT requested, but no aligned OpenSim ID STO could be loaded.")
    current_trial_jacobian_flat_dim = int(flatten_jacobian_components(data["jacp"], data["jacr"]).shape[-1])
    
    # Default model parameters
    model_params = {
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 4,
        "ff_dim": 1024,
        "dropout_rate": 0.1,
        "qfrc_inverse_output_dim": 0,
        "rotation_output_dim": 0,
        "jacobian_output_dim": 0,
        "use_film": False,
    }
    output_reg_weight = None
    detected_output_dim = STANDARD_OUTPUT_DIM
    resolved_window_size = int(window_size) if window_size is not None else 64
    resolved_stride = int(stride) if stride is not None else 16
    resolved_prediction_margin_frames = (
        int(prediction_margin_frames) if prediction_margin_frames is not None else 20
    )
    
    if hyperparams_file.exists():
        print(f"\n📄 Found hyperparameters file: {hyperparams_file}")
        try:
            with open(hyperparams_file, 'r') as f:
                config = json.load(f)
            if "d_model" in config:
                model_params["d_model"] = int(config["d_model"])
            if "num_layers" in config:
                model_params["num_layers"] = int(config["num_layers"])
            if "ff_dim" in config:
                model_params["ff_dim"] = int(config["ff_dim"])
            if "dropout_rate" in config:
                model_params["dropout_rate"] = float(config["dropout_rate"])
            if "output_dim" in config:
                detected_output_dim = int(config["output_dim"])
            if "use_film" in config:
                model_params["use_film"] = bool(config["use_film"])
            if window_size is None and "window_size" in config:
                resolved_window_size = int(config["window_size"])
                print(f"   Updated window_size from config: {resolved_window_size}")
            if stride is None and "stride" in config:
                resolved_stride = int(config["stride"])
                print(f"   Updated stride from config: {resolved_stride}")
            if prediction_margin_frames is None and "prediction_margin_frames" in config:
                resolved_prediction_margin_frames = int(config["prediction_margin_frames"])
                print(
                    "   Updated prediction_margin_frames from config: "
                    f"{resolved_prediction_margin_frames}"
                )
            if include_pelvis_euler is None and "includePelvisEuler" in config:
                resolved_include_pelvis_euler = _coerce_bool(
                    config["includePelvisEuler"],
                    default=resolved_include_pelvis_euler,
                )
                print(f"   Updated includePelvisEuler from config: {resolved_include_pelvis_euler}")
            if "output_reg_weight" in config:
                output_reg_weight = float(config["output_reg_weight"])
                print(f"   Loaded output_reg_weight from config: {output_reg_weight}")
                if output_reg_weight != 0.0:
                    print("   ℹ️ output_reg_weight is ignored because inference uses direct targets.")
                output_reg_weight = None

            print(
                f"   Loaded config: d_model={model_params['d_model']}, num_layers={model_params['num_layers']}, "
                f"ff_dim={model_params['ff_dim']}, dropout_rate={model_params['dropout_rate']}, "
                f"output_dim={detected_output_dim}"
            )
        except Exception as e:
            print(f"   ⚠️ Error loading hyperparameters: {e}")
            print("   Using default model parameters.")
    else:
        print("\n⚠️ No hyperparameters.json found. Attempting auto-detection from checkpoint...")
        # Auto-detect d_model from input projection kernel if possible.
        try:
            if "Dense_0" in params and "kernel" in params["Dense_0"]:
                dense0_kernel = params["Dense_0"]["kernel"]
                if dense0_kernel.ndim == 2:
                    model_params["d_model"] = int(dense0_kernel.shape[1])
                    print(
                        f"   🔍 Auto-detected d_model: {model_params['d_model']} "
                        f"(from Dense_0 kernel shape {tuple(dense0_kernel.shape)})"
                    )
            else:
                flat_params = jax.tree_util.tree_leaves(params)
                for p in flat_params:
                    if p.ndim == 2 and p.shape[0] > 16 and p.shape[1] >= 64:
                        model_params["d_model"] = int(p.shape[1])
                        print(f"   🔍 Auto-detected d_model: {model_params['d_model']} (from kernel shape {tuple(p.shape)})")
                        break
            
            # Auto-detect num_layers by counting TransformerBlock keys
            # params is a nested dict: {'Dense_0': {...}, 'TransformerBlock_0': {...}, ...}
            tb_count = 0
            for key in params.keys():
                if "TransformerBlock" in key:
                    tb_count += 1
            if tb_count > 0:
                model_params["num_layers"] = tb_count
                print(f"   🔍 Auto-detected num_layers: {model_params['num_layers']}")
                
            # Auto-detect ff_dim from TransformerBlock_0/Dense_0 kernel shape (d_model, ff_dim)
            if "TransformerBlock_0" in params:
                tb0 = params["TransformerBlock_0"]
                # In TransformerBlock, ff_out = nn.Dense(self.ff_dim)(x) is the first Dense layer in the FF block
                # But wait, MultiHeadDotProductAttention also has Dense layers.
                # Let's look for a kernel with shape (d_model, X) where X != d_model and X != output_dim
                for k, v in tb0.items():
                    if "Dense_0" in k: # This is usually the ff_dim projection
                        kernel = v.get("kernel")
                        if kernel is not None and kernel.shape[0] == model_params["d_model"]:
                            model_params["ff_dim"] = kernel.shape[1]
                            print(f"   🔍 Auto-detected ff_dim: {model_params['ff_dim']}")
                            break
        except Exception as e:
            print(f"   ⚠️ Auto-detection failed: {e}")

    # FiLM auto-detection from the checkpoint param tree (authoritative): a FiLM-trained
    # model has a top-level "film_mlp" Dense. This works even when the config JSON omits
    # use_film, and guarantees the reconstructed architecture matches the saved params.
    try:
        if isinstance(params, dict) and "film_mlp" in params:
            if not model_params["use_film"]:
                print("   🔍 Auto-detected FiLM conditioning (film_mlp in checkpoint)")
            model_params["use_film"] = True
        elif model_params["use_film"] and isinstance(params, dict):
            print("   ⚠️ Config requested FiLM but no film_mlp params found; disabling to match checkpoint.")
            model_params["use_film"] = False
    except Exception as e:
        print(f"   ⚠️ FiLM auto-detection failed: {e}")

    # Override with CLI arguments if provided
    if d_model is not None:
        model_params["d_model"] = d_model
        print(f"   🔧 Overriding d_model from CLI: {d_model}")
    if num_layers is not None:
        model_params["num_layers"] = num_layers
        print(f"   🔧 Overriding num_layers from CLI: {num_layers}")
    if ff_dim is not None:
        model_params["ff_dim"] = ff_dim
        print(f"   🔧 Overriding ff_dim from CLI: {ff_dim}")
    if window_size is not None:
        print(f"   🔧 Overriding window_size from CLI: {resolved_window_size}")
    if stride is not None:
        print(f"   🔧 Overriding stride from CLI: {resolved_stride}")
    if prediction_margin_frames is not None:
        print(
            "   🔧 Overriding prediction_margin_frames from CLI: "
            f"{resolved_prediction_margin_frames}"
        )

    validate_prediction_margin(resolved_window_size, resolved_prediction_margin_frames)

    # Detect output dimension from checkpoint params when it was not provided in hyperparameters.
    try:
        output_kernel = None
        if isinstance(params, dict) and "Dense_2" in params and isinstance(params["Dense_2"], dict):
            output_kernel = params["Dense_2"].get("kernel")

        if output_kernel is None:
            flat_params = jax.tree_util.tree_leaves(params)
            target_dims = {
                STANDARD_OUTPUT_DIM,
                16,
            }
            for p in flat_params:
                if (
                    getattr(p, "ndim", 0) == 2
                    and int(p.shape[0]) == int(model_params["d_model"])
                    and int(p.shape[1]) in target_dims
                ):
                    output_kernel = p
                    break

        if output_kernel is not None and getattr(output_kernel, "ndim", 0) == 2:
            detected_output_dim = int(output_kernel.shape[1])
            print(f"   🔍 Detected output dimension from checkpoint: {detected_output_dim}")
    except Exception as e:
        print(f"   ⚠️ Could not detect output dimension: {e}")
        print(f"   Using default output_dim={detected_output_dim}")

    model_params["qfrc_inverse_output_dim"] = 0
    model_params["rotation_output_dim"] = 0
    model_params["jacobian_output_dim"] = 0
    detected_output_dim = min(int(detected_output_dim), STANDARD_OUTPUT_DIM)

    # Detect cop_mask flag from hyperparameters (default True for backward compat).
    cop_mask = True
    if hyperparams_file.exists():
        try:
            with open(hyperparams_file, 'r') as f:
                config = json.load(f)
            cop_mask = _coerce_bool(config.get("cop_mask", True), default=True)
        except:
            pass
    print(f"   📐 COP Masking: {'Enabled' if cop_mask else 'Disabled'}")
    print("   📐 Architecture: Transformer")
    print(
        "   📐 Inference windows: "
        f"window_size={resolved_window_size}, stride={resolved_stride}, "
        f"prediction_margin_frames={resolved_prediction_margin_frames}"
    )
    print(f"   📐 includePelvisEuler requested: {resolved_include_pelvis_euler}")

    # Create model
    print("\n🔧 Creating model...")
    
    expected_input_dim = int(normalizers['input'].mean.shape[-1])
    input_features, resolved_include_pelvis_euler, input_layout, input_block_dims, input_dim_diag = _resolve_train_style_inputs(
        data=data,
        requested_include_pelvis_euler=resolved_include_pelvis_euler,
        expected_input_dim=expected_input_dim,
    )
    if resolved_include_pelvis_euler != _coerce_bool(include_pelvis_euler, default=resolved_include_pelvis_euler):
        print(
            "   🔧 Adjusted includePelvisEuler for checkpoint compatibility: "
            f"{_coerce_bool(include_pelvis_euler, default=resolved_include_pelvis_euler)} -> {resolved_include_pelvis_euler}"
        )
    print(
        f"   📐 Input layout: {input_layout}, "
        f"input_dim={input_features.shape[-1]} (expected {expected_input_dim})"
    )

    patient_size = np.asarray(data.get("patient_size", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if patient_size.size < 4:
        patient_size_padded = np.zeros(4, dtype=np.float32)
        patient_size_padded[:patient_size.size] = patient_size
        patient_size = patient_size_padded
    forward_vel = float(data.get("forward_vel", 0.0))

    static_context = np.array([
        data['height'][0, 0],
        data['mass'][0, 0],
        data['gender'],
        patient_size[0],
        patient_size[1],
        patient_size[2],
        patient_size[3],
        forward_vel,
    ], dtype=np.float32)

    input_dim = input_features.shape[-1]
    static_dim = static_context.shape[-1]

    model = KinematicsToCOPGRFMoments(
        input_dim=input_dim,
        static_dim=static_dim,
        output_dim=detected_output_dim,
        d_model=model_params["d_model"],
        num_layers=model_params["num_layers"],
        num_heads=model_params["num_heads"],
        ff_dim=model_params["ff_dim"],
        dropout_rate=model_params.get("dropout_rate", 0.1),
        use_film=model_params.get("use_film", False),
    )
    if model_params.get("use_film", False):
        print("   FiLM subject conditioning: Enabled")
    print("   Auxiliary model heads: Disabled (GRF/COP/GRM-only output)")
    
    # Check if dimensions match normalizer (sanity check)
    actual_input_dim = normalizers['input'].mean.shape[-1]
    if input_features.shape[-1] != actual_input_dim:
        print(f"   ❌ CRITICAL ERROR: Constructed input has {input_features.shape[-1]} dims, but model/normalizer expects {actual_input_dim}")
        print("      Train-style feature block dimensions:")
        for name, dim in input_block_dims:
            print(f"      - {name}: {dim}")
        for diag_name in sorted(key for key in input_dim_diag.keys() if key.endswith("_dim")):
            if diag_name == "contactBoolean_dim_not_used":
                continue
            print(f"      - {diag_name}: {input_dim_diag[diag_name]}")
        print(
            "      - contactBoolean (not an input feature): "
            f"{input_dim_diag['contactBoolean_dim_not_used']}"
        )
        return None, None, None, None, None, None, None

    actual_static_dim = normalizers['static'].mean.shape[-1]
    if static_context.shape[-1] != actual_static_dim:
        print(f"   ❌ CRITICAL ERROR: Constructed static context has {static_context.shape[-1]} dims, but model/normalizer expects {actual_static_dim}")
        print(f"      - static context: [height, mass, gender, patient_size(4), forward_vel]")
        return None, None, None, None, None, None, None

    restriction_summary = {
        "enabled": False,
        "num_frames": int(len(input_features)),
        "frames_outside": 0,
        "frames_outside_percent": 0.0,
        "values_outside": 0,
        "values_outside_percent": 0.0,
        "per_feature_values_outside": [0] * int(actual_input_dim),
    }

    # Normalize features, optionally clipping temporal inputs in Z-score space.
    if restrict_max_vals:
        raw_bounds = load_input_restriction_bounds(restrict_max_vals_path, expected_dim=actual_input_dim)
        input_combined, restriction_summary = apply_input_restriction(
            input_features=input_features,
            input_normalizer=normalizers['input'],
            raw_bounds=raw_bounds,
        )
        resolved_bounds_path = Path(restrict_max_vals_path) if restrict_max_vals_path is not None else DEFAULT_RESTRICT_BOUNDS_PATH
        restriction_summary["bounds_path"] = str(resolved_bounds_path)
        print(
            "   RestrictMaxVals: "
            f"{restriction_summary['frames_outside']}/{restriction_summary['num_frames']} frames "
            f"({restriction_summary['frames_outside_percent']:.2f}%) had at least one temporal input outside the trusted range"
        )
    else:
        input_combined = normalizers['input'].normalize(input_features)

    static_combined = normalizers['static'].normalize(static_context)
    
    # Ensure static_combined is 1D
    if static_combined.ndim == 2:
        static_combined = static_combined.squeeze()

    static_batch = jnp.array(static_combined[np.newaxis, ...]) # Add batch dim (1, S)
    
    # JIT compile inference
    @jax.jit
    def predict(params, x, static):
        return model.apply({'params': params}, x, static, train=False)
    
    # Warmup
    print("\n🔥 Warming up JIT...")
    warmup_window = np.asarray(input_combined[:resolved_window_size], dtype=np.float32)
    if warmup_window.shape[0] < resolved_window_size:
        pad_len = resolved_window_size - warmup_window.shape[0]
        warmup_window = np.pad(warmup_window, ((0, pad_len), (0, 0)), mode="edge")
    _ = predict(
        params,
        jnp.array(warmup_window[np.newaxis, ...]),
        static_batch,
    )
    
    # Run inference with timing
    print("\n⚡ Running inference...")
    
    # Process the trial using the same fixed window size used during training.
    start_time = time.perf_counter()
    output_np_kept, output_np_metric, evaluation_mask, window_meta = _predict_with_train_style_windows(
        predict_fn=predict,
        params=params,
        input_features_z=np.asarray(input_combined, dtype=np.float32),
        static_context_z=np.asarray(static_combined, dtype=np.float32),
        window_size=int(resolved_window_size),
        stride=int(resolved_stride),
        output_dim=int(detected_output_dim),
        prediction_margin_frames=int(resolved_prediction_margin_frames),
    )
    inference_time = (time.perf_counter() - start_time) * 1000
    
    print(f"   Inference time: {inference_time:.2f} ms")
    print(f"   Throughput: {len(data['pos']) / (inference_time / 1000):.0f} frames/sec")
    print(
        f"   Window aggregation: {window_meta['num_windows']} windows "
        f"(window_size={window_meta['window_size']}, stride={window_meta['stride']}, "
        f"prediction_margin_frames={window_meta['prediction_margin_frames']})"
    )
    
    time_axis = np.arange(len(data['pos'])) / 100.0  # Assume 100 Hz
    predictions = _convert_output_to_physical_predictions(
        output_np=output_np_kept,
        data=data,
        normalizers=normalizers,
        detected_output_dim=detected_output_dim,
        cop_mask=cop_mask,
        use_grf_norm_cop=resolved_use_grf_norm_cop,
        qfrc_inverse_output_dim=int(model_params.get("qfrc_inverse_output_dim", 0)),
        rotation_output_dim=int(model_params.get("rotation_output_dim", 0)),
        jacobian_output_dim=int(model_params.get("jacobian_output_dim", PREDICTED_JACOBIAN_FLAT_DIM)),
        use_gt_jacob_and_rot=bool(use_gt_jacob_and_rot),
    )
    evaluation_predictions = _convert_output_to_physical_predictions(
        output_np=output_np_metric,
        data=data,
        normalizers=normalizers,
        detected_output_dim=detected_output_dim,
        cop_mask=cop_mask,
        use_grf_norm_cop=resolved_use_grf_norm_cop,
        qfrc_inverse_output_dim=int(model_params.get("qfrc_inverse_output_dim", 0)),
        rotation_output_dim=int(model_params.get("rotation_output_dim", 0)),
        jacobian_output_dim=int(model_params.get("jacobian_output_dim", PREDICTED_JACOBIAN_FLAT_DIM)),
        use_gt_jacob_and_rot=bool(use_gt_jacob_and_rot),
    )

    mjx_id_reference = (
        load_mjx_id_reference_ground_truth(trial_path, target_len=len(data["pos"]))
        if use_recalculated_opensim_id_gt
        else None
    )

    ground_truth = {
        'cop': data.get('cop_gt_raw', data['cop_raw']).copy(),      # RAW PHYSICAL UNITS
        'grf': data.get('grf_gt_raw', data['grf_raw']).copy(),      # RAW PHYSICAL UNITS
        'moments': data.get('moments_gt_raw', data['moments_raw']).copy(), # RAW PHYSICAL UNITS
        'tau_grf': data.get('tau_grf_gt', data['qfrc_grf_contribution']).copy(),
        'id_gt_mjx': data['id_gt_mjx'],
        'opensim_id_gt': None if opensim_id_bundle is None else np.array(opensim_id_bundle['id'], copy=True),
        'opensim_id_available_mask': None if opensim_id_bundle is None else np.array(opensim_id_bundle['available_mask'], copy=True),
        'opensim_id_source_path': None if opensim_id_bundle is None else str(opensim_id_bundle['source_path']),
        'use_OpenSimID_GT': bool(use_OpenSimID_GT) or bool(use_recalculated_opensim_id_gt),
        'use_recalculated_opensim_id_gt': bool(use_recalculated_opensim_id_gt),
        'mjx_id_reference': mjx_id_reference,
        'qfrc_inverse': data.get('qfrc_inverse_raw', data['qfrc_inverse']),
        'qfrc_inverse_processed': data.get('qfrc_inverse_processed', data.get('qfrc_inverse_raw', data.get('qfrc_inverse'))),
        'qfrc_inverse_mocap': data.get('qfrc_inverse_mocap'),
        'rot_w_to_ga': data.get('gt_rot_w_to_ga'),
        'jacp_gt': data.get('gt_jacp'),
        'jacr_gt': data.get('gt_jacr'),
        'source': data.get('ground_truth_source', 'selected input source'),
    }

    # Mask Ground Truth as well (should already be done by data_loader, but good for safety)
    cop_gt, grf_gt, moments_gt = ground_truth['cop'], ground_truth['grf'], ground_truth['moments']

    swing_r_gt = np.abs(grf_gt[:, 2]) < 1.0
    swing_l_gt = np.abs(grf_gt[:, 5]) < 1.0
    cop_gt[swing_r_gt, 0:2] = 0.0
    grf_gt[swing_r_gt, 0:3] = 0.0
    moments_gt[swing_r_gt, 0:1] = 0.0
    cop_gt[swing_l_gt, 2:4] = 0.0
    grf_gt[swing_l_gt, 3:6] = 0.0
    moments_gt[swing_l_gt, 1:2] = 0.0

    evaluation_mask = _normalize_evaluation_mask(evaluation_mask, len(data['pos']))

    # Apply post-inference filtering if enabled
    if FilterPostInfer:
        print(f"\n🔧 Applying post-inference Butterworth filter (Order=4, Cutoff={FILTER_CUTOFF_HZ}Hz, Fs={FILTER_SAMPLING_RATE_HZ}Hz)...")

        # Filter only the kept center-valid region.
        predictions['cop'] = apply_butterworth_filter_masked(predictions['cop'], evaluation_mask)
        predictions['grf'] = apply_butterworth_filter_masked(predictions['grf'], evaluation_mask)
        predictions['moments'] = apply_butterworth_filter_masked(predictions['moments'], evaluation_mask)
        predictions['tau_grf'] = apply_butterworth_filter_masked(predictions['tau_grf'], evaluation_mask)
        predictions['qfrc_grf_contribution'] = predictions['tau_grf']
        if predictions.get('qfrc_inverse') is not None:
            predictions['qfrc_inverse'] = apply_butterworth_filter_masked(predictions['qfrc_inverse'], evaluation_mask)
        evaluation_predictions['cop'] = apply_butterworth_filter_masked(evaluation_predictions['cop'], evaluation_mask)
        evaluation_predictions['grf'] = apply_butterworth_filter_masked(evaluation_predictions['grf'], evaluation_mask)
        evaluation_predictions['moments'] = apply_butterworth_filter_masked(evaluation_predictions['moments'], evaluation_mask)
        evaluation_predictions['tau_grf'] = apply_butterworth_filter_masked(evaluation_predictions['tau_grf'], evaluation_mask)
        evaluation_predictions['qfrc_grf_contribution'] = evaluation_predictions['tau_grf']
        if evaluation_predictions.get('qfrc_inverse') is not None:
            evaluation_predictions['qfrc_inverse'] = apply_butterworth_filter_masked(
                evaluation_predictions['qfrc_inverse'],
                evaluation_mask,
            )

        # Filter ground truth only on the kept region as well.
        ground_truth['cop'] = apply_butterworth_filter_masked(ground_truth['cop'], evaluation_mask)
        ground_truth['grf'] = apply_butterworth_filter_masked(ground_truth['grf'], evaluation_mask)
        ground_truth['moments'] = apply_butterworth_filter_masked(ground_truth['moments'], evaluation_mask)
        ground_truth['tau_grf'] = apply_butterworth_filter_masked(ground_truth['tau_grf'], evaluation_mask)
        if ground_truth['id_gt_mjx'] is not None:
            ground_truth['id_gt_mjx'] = apply_butterworth_filter_masked(ground_truth['id_gt_mjx'], evaluation_mask)
        if ground_truth.get('opensim_id_gt') is not None:
            ground_truth['opensim_id_gt'] = apply_butterworth_filter_masked(
                ground_truth['opensim_id_gt'],
                evaluation_mask,
            )
        if ground_truth['qfrc_inverse'] is not None:
            ground_truth['qfrc_inverse'] = apply_butterworth_filter_masked(ground_truth['qfrc_inverse'], evaluation_mask)
        if ground_truth.get('qfrc_inverse_processed') is not None:
            ground_truth['qfrc_inverse_processed'] = apply_butterworth_filter_masked(
                ground_truth['qfrc_inverse_processed'],
                evaluation_mask,
            )
        if ground_truth.get('qfrc_inverse_mocap') is not None:
            ground_truth['qfrc_inverse_mocap'] = apply_butterworth_filter_masked(
                ground_truth['qfrc_inverse_mocap'],
                evaluation_mask,
            )
        if resolved_use_grf_norm_cop:
            predictions['cop'] = zero_cop_where_contact_below_threshold_np(
                predictions['cop'],
                predictions.get('contact'),
                threshold=0.5,
            )
            evaluation_predictions['cop'] = zero_cop_where_contact_below_threshold_np(
                evaluation_predictions['cop'],
                evaluation_predictions.get('contact'),
                threshold=0.5,
            )
        print("   ✅ Filtering complete")

    evaluation_frame_count = int(np.sum(evaluation_mask))
    print(
        "   Evaluation region: "
        f"{evaluation_frame_count}/{len(data['pos'])} frames "
        f"(prediction_margin_frames={resolved_prediction_margin_frames})"
    )
    
    # Compute metrics
    print("\n📊 Computing metrics...")

    # Body Weight * Height Normalization
    mass = data['mass'][0, 0]
    height = data['height'][0, 0]
    norm_factor = mass * height * 9.8067
    metrics = None
    rotation_comparison_stats = None
    jacobian_comparison_stats = None
    knee_torque_comparison_stats = None
    if evaluation_frame_count <= 0:
        print(
            "   ⚠️ No evaluation frames remain after center-window aggregation "
            f"(prediction_margin_frames={resolved_prediction_margin_frames})."
        )
    else:
        selected_torque_indices = get_selected_left_stance_dof_indices()
        selected_torque_names = [get_dof_names()[idx] for idx in selected_torque_indices]
        left_stance_mask = get_left_stance_mask(
            data['grf_raw'],
            evaluation_mask,
            threshold=LEFT_STANCE_THRESHOLD_N,
        )
        left_stance_frame_count = int(np.sum(left_stance_mask))
        torque_metric_source = "tau_grf_contribution"
        torque_reference_label = "MJX_ID"
        if bool(use_OpenSimID_GT) or bool(use_recalculated_opensim_id_gt):
            full_id_metric_pred, full_id_metric_gt, full_id_metric_source = compute_full_id_curves(
                evaluation_predictions,
                ground_truth,
            )
            if full_id_metric_pred is not None and full_id_metric_gt is not None:
                torque_pred_selected = full_id_metric_pred[:, selected_torque_indices]
                torque_gt_selected = full_id_metric_gt[:, selected_torque_indices]
                torque_metric_source = full_id_metric_source
                torque_reference_label = resolve_full_id_reference_curves(ground_truth)[1]
            else:
                torque_pred_selected = evaluation_predictions['tau_grf'][:, selected_torque_indices]
                torque_gt_selected = ground_truth['tau_grf'][:, selected_torque_indices]
        else:
            torque_pred_selected = evaluation_predictions['tau_grf'][:, selected_torque_indices]
            torque_gt_selected = ground_truth['tau_grf'][:, selected_torque_indices]
        torque_rmse_per_dof = _masked_rmse_per_channel(
            torque_pred_selected,
            torque_gt_selected,
            left_stance_mask,
        )
        cop_bias_per_channel = _masked_mean_diff(
            evaluation_predictions['cop'],
            ground_truth['cop'],
            evaluation_mask,
        )
        grf_bias_per_channel = _masked_mean_diff(
            evaluation_predictions['grf'],
            ground_truth['grf'],
            evaluation_mask,
        )
        torque_rmse_bwh_per_dof = (torque_rmse_per_dof / norm_factor) * 100
        torque_rmse = _masked_rmse(
            torque_pred_selected,
            torque_gt_selected,
            left_stance_mask,
        )
        torque_rmse_bwh = (torque_rmse / norm_factor) * 100

        gt_torque = torque_gt_selected[left_stance_mask]
        if left_stance_frame_count > 0:
            gt_std = np.std(gt_torque, axis=0)
            gt_std_safe = np.where(gt_std < 1e-6, 1.0, gt_std)
            torque_nrmse_per_dof = torque_rmse_per_dof / gt_std_safe
            torque_mae = _masked_mae(
                torque_pred_selected,
                torque_gt_selected,
                left_stance_mask,
            )
        else:
            gt_std_safe = np.ones(len(selected_torque_indices), dtype=np.float64)
            torque_nrmse_per_dof = np.full(len(selected_torque_indices), np.nan, dtype=np.float64)
            torque_mae = float("nan")

        metrics = {
            'cop_rmse': _masked_rmse(evaluation_predictions['cop'], ground_truth['cop'], evaluation_mask),
            'grf_rmse': _masked_rmse(evaluation_predictions['grf'], ground_truth['grf'], evaluation_mask),
            'moments_rmse': _masked_rmse(evaluation_predictions['moments'], ground_truth['moments'], evaluation_mask),
            'cop_bias_per_channel': cop_bias_per_channel.tolist(),
            'grf_bias_per_channel': grf_bias_per_channel.tolist(),
            'torque_rmse': float(torque_rmse),
            'torque_rmse_bwh': float(torque_rmse_bwh),
            'torque_nrmse': float(np.mean(torque_nrmse_per_dof)),
            'torque_rmse_per_dof': torque_rmse_per_dof.tolist(),
            'torque_rmse_bwh_per_dof': torque_rmse_bwh_per_dof.tolist(),
            'torque_nrmse_per_dof': torque_nrmse_per_dof.tolist(),
            'torque_mae': float(torque_mae),
            'torque_mae_bwh': float((torque_mae / norm_factor) * 100.0),
            'torque_metric_dof_names': selected_torque_names,
            'torque_metric_scope': 'left_stance_selected_dofs',
            'torque_metric_side': 'left',
            'torque_metric_phase': 'stance',
            'torque_metric_left_stance_frame_count': left_stance_frame_count,
            'torque_metric_stance_threshold_N': float(LEFT_STANCE_THRESHOLD_N),
            'torque_metric_source': torque_metric_source,
            'torque_reference_label': torque_reference_label,
            'use_OpenSimID_GT': bool(use_OpenSimID_GT),
            'use_recalculated_opensim_id_gt': bool(use_recalculated_opensim_id_gt),
            'inference_time_ms': inference_time,
            'num_frames': len(data['pos']),
            'evaluation_frame_count': evaluation_frame_count,
            'window_size': int(resolved_window_size),
            'stride': int(resolved_stride),
            'prediction_margin_frames': int(resolved_prediction_margin_frames),
            'input_source': input_source_norm,
            'input_source_label': input_source_display,
            'input_kinematics_source': data.get('input_kinematics_source', 'Pos'),
            'use_noised_inputs': bool(data.get('use_noised_inputs', False)),
            'ground_truth_source': data.get('ground_truth_source', 'selected input source'),
            'restrict_max_vals': restriction_summary,
        }
        metrics.update(
            _compute_qfrc_inverse_rmse_metrics(
                evaluation_predictions,
                ground_truth,
                evaluation_mask,
                norm_factor,
            )
        )
        if (
            opencap_val_dataset
            and input_source_norm != "mocap"
            and evaluation_predictions.get('rot_w_to_ga') is not None
            and ground_truth.get('rot_w_to_ga') is not None
            and evaluation_predictions.get('predicted_jacobian_jacp') is not None
            and evaluation_predictions.get('predicted_jacobian_jacr') is not None
            and data.get('jacp') is not None
            and data.get('jacr') is not None
            and ground_truth.get('jacp_gt') is not None
            and ground_truth.get('jacr_gt') is not None
        ):
            rotation_comparison_stats = _build_rotation_comparison_stats(
                evaluation_predictions['rot_w_to_ga'],
                np.asarray(data['rot_w_to_ga'], dtype=np.float32),
                np.asarray(ground_truth['rot_w_to_ga'], dtype=np.float32),
                evaluation_mask,
            )
            jacobian_comparison_stats = _build_jacobian_comparison_stats(
                evaluation_predictions['predicted_jacobian_jacp'],
                evaluation_predictions['predicted_jacobian_jacr'],
                np.asarray(data['jacp'], dtype=np.float32),
                np.asarray(data['jacr'], dtype=np.float32),
                np.asarray(ground_truth['jacp_gt'], dtype=np.float32),
                np.asarray(ground_truth['jacr_gt'], dtype=np.float32),
                evaluation_mask,
            )
            metrics['rotation_vs_mocap_comparison'] = rotation_comparison_stats
            metrics['jacobian_vs_mocap_comparison'] = jacobian_comparison_stats
            knee_torque_comparison_stats = _build_knee_flexion_torque_comparison_stats(
                predictions=evaluation_predictions,
                data=data,
                ground_truth=ground_truth,
                left_stance_mask=left_stance_mask,
            )
            metrics['knee_flexion_torque_vs_mocap_comparison'] = knee_torque_comparison_stats

        print(f"   COP RMSE:         {metrics['cop_rmse']:.4f} m")
        print(f"   GRF RMSE:         {metrics['grf_rmse']:.1f} N")
        print(f"   Moments RMSE:     {metrics['moments_rmse']:.2f} Nm")
        print(f"   COP bias/ch:      {np.array2string(cop_bias_per_channel, precision=4)} m")
        print(f"   GRF bias/ch:      {np.array2string(grf_bias_per_channel, precision=2)} N")
        print(f"   Left-stance frames: {left_stance_frame_count}")
        print(f"   Torque reference: {metrics['torque_reference_label']} ({metrics['torque_metric_source']})")
        print(f"   Torque RMSE:      {metrics['torque_rmse']:.2f} Nm")
        print(f"   Torque RMSE BWH:  {metrics['torque_rmse_bwh']:.3f} %BW*H")
        print(f"   Torque MAE BWH:   {metrics['torque_mae_bwh']:.3f} %BW*H")
        print(f"   Torque DOFs:      {', '.join(selected_torque_names)}")
        print(f"   Torque nRMSE:     {metrics['torque_nrmse']:.4f} (normalized by GT std)")

    # Optional OpenCap comparison: same MoCap ground truth, alternate temporal inputs from MoCap/.
    secondary_predictions = None
    secondary_metrics = None
    secondary_mae_report = None
    motioncapture_stance_cop_mae = {}
    if opencap_val_dataset and input_source_norm != "mocap":
        print("\n🔁 Running secondary inference with MotionCapture inputs for OpenCap comparison...")
        try:
            sec_mae, sec_metrics, sec_preds, _, _, _, _ = run_inference(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                trial_name=trial_name,
                output_dir=output_dir,
                window_size=resolved_window_size,
                stride=resolved_stride,
                prediction_margin_frames=resolved_prediction_margin_frames,
                d_model=d_model,
                num_layers=num_layers,
                ff_dim=ff_dim,
                no_plots=True,
                lightweight=True,
                make_graph=False,
                use_noised=False,
                min_trial_length=min_trial_length,
                opencap_val_dataset=opencap_val_dataset,
                input_source="mocap",
                restrict_max_vals=restrict_max_vals,
                restrict_max_vals_path=restrict_max_vals_path,
                use_OpenSimID_GT=use_OpenSimID_GT,
                use_recalculated_opensim_id_gt=use_recalculated_opensim_id_gt,
                use_grf_norm_cop=resolved_use_grf_norm_cop,
                use_grf_nofilt=use_grf_nofilt,
            )
            if sec_preds is not None:
                secondary_predictions = sec_preds
                secondary_metrics = sec_metrics
                secondary_mae_report = sec_mae
            if metrics is not None and sec_metrics is not None and sec_preds is not None:
                motioncapture_stance_cop_mae = _extract_stance_cop_mae_percent_height(sec_mae)
                metrics['opencap_input_cop_rmse'] = float(metrics.get('cop_rmse', np.nan))
                metrics['opencap_input_grf_rmse'] = float(metrics.get('grf_rmse', np.nan))
                metrics['opencap_input_moments_rmse'] = float(metrics.get('moments_rmse', np.nan))
                metrics['opencap_input_torque_rmse'] = float(metrics['torque_rmse'])
                metrics['opencap_input_torque_rmse_bwh'] = float(metrics['torque_rmse_bwh'])
                metrics['opencap_input_torque_mae_bwh'] = float(metrics['torque_mae_bwh'])
                metrics['opencap_input_qfrc_inverse_processed_vs_gt_rmse'] = float(metrics.get('qfrc_inverse_processed_vs_gt_rmse', np.nan))
                metrics['opencap_input_qfrc_inverse_pred_vs_gt_rmse'] = float(metrics.get('qfrc_inverse_pred_vs_gt_rmse', np.nan))
                metrics['opencap_input_qfrc_inverse_processed_minus_pred_rmse'] = float(metrics.get('qfrc_inverse_processed_minus_pred_rmse', np.nan))
                metrics['opencap_input_qfrc_inverse_pred_vs_processed_rmse'] = float(metrics.get('qfrc_inverse_pred_vs_processed_rmse', np.nan))
                metrics['opencap_input_qfrc_inverse_pred_vs_mocap_rmse'] = float(metrics.get('qfrc_inverse_pred_vs_mocap_rmse', np.nan))
                metrics['opencap_input_qfrc_inverse_processed_vs_mocap_rmse'] = float(metrics.get('qfrc_inverse_processed_vs_mocap_rmse', np.nan))
                metrics['motioncapture_input_cop_rmse'] = float(sec_metrics.get('cop_rmse', np.nan))
                metrics['motioncapture_input_grf_rmse'] = float(sec_metrics.get('grf_rmse', np.nan))
                metrics['motioncapture_input_moments_rmse'] = float(sec_metrics.get('moments_rmse', np.nan))
                metrics['motioncapture_input_torque_rmse'] = float(sec_metrics.get('torque_rmse', np.nan))
                metrics['motioncapture_input_torque_rmse_bwh'] = float(sec_metrics.get('torque_rmse_bwh', np.nan))
                metrics['motioncapture_input_torque_mae_bwh'] = float(sec_metrics.get('torque_mae_bwh', np.nan))
                metrics['motioncapture_input_qfrc_inverse_processed_vs_gt_rmse'] = float(sec_metrics.get('qfrc_inverse_processed_vs_gt_rmse', np.nan))
                metrics['motioncapture_input_qfrc_inverse_pred_vs_gt_rmse'] = float(sec_metrics.get('qfrc_inverse_pred_vs_gt_rmse', np.nan))
                metrics['motioncapture_input_qfrc_inverse_processed_minus_pred_rmse'] = float(sec_metrics.get('qfrc_inverse_processed_minus_pred_rmse', np.nan))
                metrics['motioncapture_input_qfrc_inverse_pred_vs_processed_rmse'] = float(sec_metrics.get('qfrc_inverse_pred_vs_processed_rmse', np.nan))
                metrics['motioncapture_input_qfrc_inverse_pred_vs_mocap_rmse'] = float(sec_metrics.get('qfrc_inverse_pred_vs_mocap_rmse', np.nan))
                metrics['motioncapture_input_qfrc_inverse_processed_vs_mocap_rmse'] = float(sec_metrics.get('qfrc_inverse_processed_vs_mocap_rmse', np.nan))
                metrics['motioncapture_input_stance_cop_mae_percent_height'] = motioncapture_stance_cop_mae

                # Backward-compatible aliases used by older result readers.
                metrics['video_input_torque_rmse'] = metrics['opencap_input_torque_rmse']
                metrics['video_input_torque_rmse_bwh'] = metrics['opencap_input_torque_rmse_bwh']
                metrics['video_input_torque_mae_bwh'] = metrics['opencap_input_torque_mae_bwh']
                metrics['video_input_stance_cop_mae_percent_height'] = opencap_stance_cop_mae
                metrics['mocap_input_torque_rmse'] = metrics['motioncapture_input_torque_rmse']
                metrics['mocap_input_torque_rmse_bwh'] = metrics['motioncapture_input_torque_rmse_bwh']
                metrics['mocap_input_torque_mae_bwh'] = metrics['motioncapture_input_torque_mae_bwh']
                metrics['mocap_input_cop_rmse'] = metrics['motioncapture_input_cop_rmse']
                metrics['mocap_input_grf_rmse'] = metrics['motioncapture_input_grf_rmse']
                metrics['mocap_input_moments_rmse'] = metrics['motioncapture_input_moments_rmse']
                metrics['mocap_input_stance_cop_mae_percent_height'] = motioncapture_stance_cop_mae

                print("   ✅ OpenCap dual-input comparison:")
                print(
                    f"      OpenCap input        | Torque RMSE: {metrics['opencap_input_torque_rmse']:.2f} Nm"
                    f" | RMSE BWH: {metrics['opencap_input_torque_rmse_bwh']:.3f}% | MAE BWH: {metrics['opencap_input_torque_mae_bwh']:.3f}%"
                )
                print(
                    f"      MotionCapture input | Torque RMSE: {metrics['motioncapture_input_torque_rmse']:.2f} Nm"
                    f" | RMSE BWH: {metrics['motioncapture_input_torque_rmse_bwh']:.3f}% | MAE BWH: {metrics['motioncapture_input_torque_mae_bwh']:.3f}%"
                )
        except Exception as _sec_exc:
            print(f"   ⚠️ Secondary MotionCapture-input inference failed: {_sec_exc}")

    # Keep the stitched center-only aggregation as the canonical prediction view.
    predictions["_metric_view"] = evaluation_predictions
    predictions["_evaluation_mask"] = evaluation_mask
    if secondary_predictions is not None:
        secondary_mask = secondary_predictions.get("_evaluation_mask")
        if secondary_mask is None:
            secondary_mask = np.ones((len(secondary_predictions["cop"]),), dtype=bool)
        secondary_predictions["_evaluation_mask"] = _normalize_evaluation_mask(
            secondary_mask,
            len(secondary_predictions["cop"]),
        )
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    
    safe_trial_name = trial_name.replace('/', '_')
    
    # Create trial-specific output directory
    trial_output_dir = os.path.join(output_dir, safe_trial_name)
    if not lightweight:
        os.makedirs(trial_output_dir, exist_ok=True)
    
    # Save model information
    if not lightweight:
        print("\n📝 Saving model information...")
        save_model_info(
            model, params, normalizers, checkpoint_path,
            train_trials, val_trials,
            save_path=os.path.join(trial_output_dir, "model_info.txt"),
            output_reg_weight=None,
        )
    
    # Add split status to trial name for display
    trial_display = f"{trial_name} [{split_status}]"

    # For side-by-side plotting only, align sequence lengths if dual-input predictions differ.
    plot_time_axis = time_axis
    plot_predictions = _mask_prediction_dict_for_display(
        {k: v for k, v in predictions.items() if not k.startswith("_")},
        evaluation_mask,
    )
    plot_ground_truth = ground_truth
    plot_metric_predictions = _mask_prediction_dict_for_display(
        predictions.get("_metric_view", plot_predictions),
        evaluation_mask,
    )
    plot_secondary_predictions = (
        _mask_prediction_dict_for_display(
            {k: v for k, v in secondary_predictions.items() if not k.startswith("_")},
            secondary_predictions["_evaluation_mask"],
        )
        if secondary_predictions is not None
        else None
    )
    plot_metric_secondary_predictions = (
        _mask_prediction_dict_for_display(
            secondary_predictions.get("_metric_view", plot_secondary_predictions),
            secondary_predictions["_evaluation_mask"],
        )
        if secondary_predictions is not None
        else None
    )
    if secondary_predictions is not None:
        min_plot_len = min(
            len(time_axis),
            len(plot_predictions['cop']),
            len(ground_truth['cop']),
            len(plot_secondary_predictions['cop']),
        )
        if min_plot_len < len(time_axis):
            print(
                f"   ⚠️ Plot alignment: trimming to {min_plot_len} frames "
                f"(video={len(plot_predictions['cop'])}, mocap={len(plot_secondary_predictions['cop'])})"
            )
        plot_time_axis = time_axis[:min_plot_len]
        plot_predictions = {k: v[:min_plot_len] for k, v in plot_predictions.items()}
        plot_metric_predictions = {k: v[:min_plot_len] for k, v in plot_metric_predictions.items()}
        plot_ground_truth = {}
        for k, v in ground_truth.items():
            if v is None:
                plot_ground_truth[k] = None
            elif isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(time_axis):
                plot_ground_truth[k] = v[:min_plot_len]
            else:
                plot_ground_truth[k] = v
        plot_secondary_predictions = {k: v[:min_plot_len] for k, v in plot_secondary_predictions.items()}
        plot_metric_secondary_predictions = {
            k: v[:min_plot_len] for k, v in plot_metric_secondary_predictions.items()
        }
    
    # 4. Create visualizations
    if not no_plots and not lightweight:
        if rotation_comparison_stats is not None and jacobian_comparison_stats is not None:
            create_rotation_jacobian_comparison_plot(
                trial_display,
                rotation_comparison_stats,
                jacobian_comparison_stats,
                knee_torque_stats=knee_torque_comparison_stats,
                save_path=os.path.join(trial_output_dir, "rotation_jacobian_comparison.html"),
            )
        # Right side plot
        fig_right = create_timeseries_plot(
            plot_time_axis, plot_predictions, plot_secondary_predictions, plot_ground_truth, trial_display,
            side='Right',
            save_path=os.path.join(trial_output_dir, "timeseries_right.html"),
            pred_label="Prediction (OpenCap input)",
            alt_pred_label="Prediction (MotionCapture input)",
            evaluation_mask=evaluation_mask[:len(plot_time_axis)],
            metric_predictions=plot_metric_predictions,
            metric_predictions_alt=plot_metric_secondary_predictions,
            prediction_margin_frames=resolved_prediction_margin_frames,
        )
        
        # Left side plot
        fig_left = create_timeseries_plot(
            plot_time_axis, plot_predictions, plot_secondary_predictions, plot_ground_truth, trial_display,
            side='Left',
            save_path=os.path.join(trial_output_dir, "timeseries_left.html"),
            pred_label="Prediction (OpenCap input)",
            alt_pred_label="Prediction (MotionCapture input)",
            evaluation_mask=evaluation_mask[:len(plot_time_axis)],
            metric_predictions=plot_metric_predictions,
            metric_predictions_alt=plot_metric_secondary_predictions,
            prediction_margin_frames=resolved_prediction_margin_frames,
        )
        
        # Error distribution plot
        fig_errors = create_error_distribution_plot(
            predictions, ground_truth, trial_display,
            save_path=os.path.join(trial_output_dir, "errors.html"),
            evaluation_mask=evaluation_mask,
            metric_predictions=evaluation_predictions,
        )
    
    # Analyze stance phase torques
    mae_report, stance_results = analyze_stance_phase_torques(
        evaluation_predictions,
        ground_truth,
        data,
        trial_output_dir,
        safe_trial_name,
        no_plots=no_plots,
        lightweight=lightweight,
        evaluation_mask=evaluation_mask,
    )

    bilateral_stance_mae_report = build_bilateral_stance_mae_report(
        evaluation_predictions,
        ground_truth,
        data,
        evaluation_mask=evaluation_mask,
    )
    if not lightweight:
        bilateral_report_path = os.path.join(
            trial_output_dir,
            f"{safe_trial_name}_stance_mae_both_legs.json",
        )
        with open(bilateral_report_path, "w") as f:
            json.dump(bilateral_stance_mae_report, f, indent=2)
        print(f"💾 Saved bilateral stance MAE report to: {bilateral_report_path}")

    complete_stance_peak_report = build_complete_stance_peak_report(
        evaluation_predictions,
        ground_truth,
        data,
        patient_path=patient_path,
        evaluation_mask=evaluation_mask,
        trial_path=trial_path,
    )

    if "OldYoungAdultWalking_MJX_Processed" in str(data_dir):
        ankle_power_avg_report = build_ankle_power_avg_report(
            evaluation_predictions,
            ground_truth,
            data,
            trial_name=trial_name,
            evaluation_mask=evaluation_mask,
        )
        complete_stance_peak_report["ankle_power_vs_stance_percent"] = ankle_power_avg_report

        ankle_power_per_stance_report = build_ankle_power_per_stance_report(
            evaluation_predictions,
            ground_truth,
            data,
            patient_path=patient_path,
            evaluation_mask=evaluation_mask,
        )
        complete_stance_peak_report["ankle_power_per_stance"] = ankle_power_per_stance_report

    if not no_plots and not lightweight:
        ankle_power_plot_path = os.path.join(
            trial_output_dir,
            f"{safe_trial_name}_full_trial_ankle_power.png",
        )
        ankle_power_plot_report = create_full_trial_ankle_power_plot(
            evaluation_predictions,
            ground_truth,
            data,
            trial_path=trial_path,
            trial_name=trial_display,
            save_path=ankle_power_plot_path,
            evaluation_mask=evaluation_mask,
        )
        if ankle_power_plot_report.get("available"):
            print(f"💾 Saved full-trial ankle power plot to: {ankle_power_plot_path}")
        else:
            print(
                "   ⚠️ Full-trial ankle power plot skipped: "
                f"{ankle_power_plot_report.get('reason', 'unknown_reason')}"
            )
        complete_stance_peak_report["full_trial_ankle_power_plot"] = ankle_power_plot_report

    if not lightweight:
        complete_stance_report_path = os.path.join(
            trial_output_dir,
            f"{safe_trial_name}_complete_stance_peak_metrics.json",
        )
        with open(complete_stance_report_path, "w") as f:
            json.dump(complete_stance_peak_report, f, indent=2)
        print(f"💾 Saved complete-stance peak metrics report to: {complete_stance_report_path}")

    if metrics is not None:
        metrics["bilateral_stance_mae_report"] = bilateral_stance_mae_report
        metrics["complete_stance_peak_metrics"] = complete_stance_peak_report
        if secondary_metrics is not None:
            metrics["opencap_input_bilateral_stance_mae_report"] = bilateral_stance_mae_report
            motioncapture_bilateral_report = secondary_metrics.get("bilateral_stance_mae_report")
            if isinstance(motioncapture_bilateral_report, dict):
                metrics["motioncapture_input_bilateral_stance_mae_report"] = motioncapture_bilateral_report

    if metrics is not None and mae_report:
        metrics["stance_cop_mae_percent_height"] = _extract_stance_cop_mae_percent_height(mae_report)
        metrics.update(_extract_left_stance_kam_metrics(mae_report, stance_results))
        if secondary_metrics is not None:
            metrics["opencap_input_stance_cop_mae_percent_height"] = dict(metrics["stance_cop_mae_percent_height"])
            metrics["motioncapture_input_stance_cop_mae_percent_height"] = motioncapture_stance_cop_mae
            if "left_stance_kam_mae_bwh" in metrics:
                metrics["opencap_input_left_stance_kam_mae_bwh"] = float(metrics["left_stance_kam_mae_bwh"])
            if "left_stance_kam_rmse_bwh" in metrics:
                metrics["opencap_input_left_stance_kam_rmse_bwh"] = float(metrics["left_stance_kam_rmse_bwh"])
            if "left_stance_kam_mae_bwh" in secondary_metrics:
                metrics["motioncapture_input_left_stance_kam_mae_bwh"] = float(
                    secondary_metrics["left_stance_kam_mae_bwh"]
                )
            if "left_stance_kam_rmse_bwh" in secondary_metrics:
                metrics["motioncapture_input_left_stance_kam_rmse_bwh"] = float(
                    secondary_metrics["left_stance_kam_rmse_bwh"]
                )
    
    # Publication-ready plots
    if make_graph:
        model_name = Path(checkpoint_path).parent.name
        make_publication_plots(predictions, ground_truth, trial_name, output_dir, model_name)
    
    # All DOFs plot
    if not no_plots and not lightweight:
        if data.get("input_source_folder") == "MoCap":
            primary_qfrc_for_plots = plot_predictions.get(
                "qfrc_inverse",
                plot_ground_truth.get("qfrc_inverse_mocap", plot_ground_truth.get("qfrc_inverse")),
            )
        else:
            primary_qfrc_for_plots = plot_predictions.get(
                "qfrc_inverse",
                plot_ground_truth.get("qfrc_inverse_processed", plot_ground_truth.get("qfrc_inverse")),
            )
        secondary_qfrc_for_plots = (
            plot_secondary_predictions.get("qfrc_inverse")
            if plot_secondary_predictions is not None and plot_secondary_predictions.get("qfrc_inverse") is not None
            else plot_ground_truth.get("qfrc_inverse_mocap", primary_qfrc_for_plots)
        )
        fig_all_dofs = create_all_dofs_plot(
            plot_time_axis, plot_predictions, plot_secondary_predictions, plot_ground_truth, trial_display,
            qfrc_inverse_pred=primary_qfrc_for_plots,
            qfrc_inverse_alt=secondary_qfrc_for_plots,
            save_path=os.path.join(trial_output_dir, "all_dofs.html"),
            pred_label="Prediction (OpenCap input)",
            alt_pred_label="Prediction (MotionCapture input)",
            evaluation_mask=evaluation_mask[:len(plot_time_axis)],
            metric_predictions=plot_metric_predictions,
            metric_predictions_alt=plot_metric_secondary_predictions,
            prediction_margin_frames=resolved_prediction_margin_frames,
        )
    
    if not lightweight:
        print(f"📁 All outputs saved to: {trial_output_dir}")
    
    print("\n" + "=" * 70)
    print("✅ Inference complete!")
    print("=" * 70)
    print(
        "   Summary: "
        f"metrics={'yes' if metrics is not None else 'no'}, "
        f"mae_report={'yes' if mae_report is not None else 'no'}, "
        f"stance_results={0 if not stance_results else len(stance_results)} DOFs, "
        f"secondary_mae_report={'yes' if secondary_mae_report is not None else 'no'}"
    )
    
    # Add trial name to metrics for summary dashboard
    if metrics is not None:
        metrics['trial_name'] = trial_name
    
    return mae_report, metrics, predictions, ground_truth, time_axis, stance_results, secondary_mae_report


def make_publication_plots(predictions, ground_truth, trial_name, output_dir, model_name):
    """
    Generate professional/publication-ready plots for specified DOFs.
    DOFs: Knee Moment R, Ankle Dorsiflexion R, Hip Flexion R, Hip Adduction R, Hip Rotation R.
    Units: Nm.
    X-axis: Time (seconds) - 100Hz.
    If trial > 2s, plot a random 2s segment.
    """
    print(f"\n🎨 Generating publication-ready plots for {trial_name}...")
    
    full_id_pred, full_id_gt, id_source = compute_full_id_curves(predictions, ground_truth)
    if full_id_pred is None or full_id_gt is None:
        print("   ⚠️  Skipping publication plots: missing torque signals")
        return
    _selected_gt, metric_reference_label, mjx_gt_full, opensim_gt_full, opensim_mask = (
        resolve_full_id_reference_curves(ground_truth)
    )
    if not str(id_source).startswith("ID_GT_MJX"):
        print(f"   ⚠️  Publication plot fallback source: {id_source}")
    
    # Mapping DOFs (Right Side Only)
    dofs = {
        'Hip Flexion R': 6,
        'Hip Adduction R': 7,
        'Hip Rotation R': 8,
        'Knee Moment R': 11,
        'Ankle Dorsiflexion R': 14
    }
    
    # Time axis (100Hz)
    n_frames = full_id_pred.shape[0]
    time_axis = np.arange(n_frames) / 100.0
    
    # Determine segment to plot
    # If trial > 2s (200 frames), pick a random 2s segment
    if n_frames > 200:
        start_idx = np.random.randint(0, n_frames - 200)
        end_idx = start_idx + 200
    else:
        start_idx = 0
        end_idx = n_frames
        
    plot_time = time_axis[start_idx:end_idx]
    # Normalize time to start at 0
    plot_time = plot_time - plot_time[0]
    
    # Figures directory (Use project root figures folder)
    fig_root = os.path.join(os.getcwd(), "figures", model_name)
    os.makedirs(fig_root, exist_ok=True)
    
    # Set publication-ready font
    plt.rcParams['font.family'] = 'serif'
    
    for dof_label, dof_idx in dofs.items():
        plt.figure(figsize=(12, 8))
        
        # Plot GT/OpenSim, MJX_ID, and prediction on the same full-ID axis.
        if opensim_gt_full is not None and (
            opensim_mask is None or (dof_idx < len(opensim_mask) and opensim_mask[dof_idx])
        ):
            plt.plot(
                plot_time,
                np.asarray(opensim_gt_full)[start_idx:end_idx, dof_idx],
                color='black',
                label='GT',
                linewidth=6,
                alpha=0.85,
            )
        if mjx_gt_full is not None:
            plt.plot(
                plot_time,
                np.asarray(mjx_gt_full)[start_idx:end_idx, dof_idx],
                color='#6C757D',
                label='MJX_ID',
                linewidth=5,
                linestyle=':',
                alpha=0.9,
            )
        plt.plot(plot_time, full_id_pred[start_idx:end_idx, dof_idx], 
                 color='#E94F37', label='Model Prediction', linewidth=6, linestyle='--')
        
        # Publication-ready styling
        plt.xlabel("Time (s)", fontsize=24, fontweight='bold', labelpad=15)
        plt.ylabel("Joint Torque (Nm)", fontsize=24, fontweight='bold', labelpad=15)
        
        # Legend with no frame
        plt.legend(frameon=False, fontsize=20, loc='lower right', prop={'weight': 'bold', 'size': 20})
        
        # Clean axes
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Tick styling
        plt.xticks(fontsize=18, fontweight='bold')
        plt.yticks(fontsize=18, fontweight='bold')
        
        # Add grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot (Title is present here)
        safe_trial = trial_name.replace('/', '_').replace(' ', '_')
        save_name = f"{dof_label.replace(' ', '_')}_{safe_trial}.png"
        save_path = os.path.join(fig_root, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"   💾 Saved: {save_path}")
        
        # Explicitly show for interactive sessions
        plt.show(block=False)
        plt.pause(0.1) 
        plt.close()
    
    print(f"✅ All publication plots saved to: {fig_root}")


def analyze_stance_phase_torques(
    predictions,
    ground_truth,
    data,
    output_dir,
    trial_name,
    no_plots=False,
    lightweight=False,
    evaluation_mask: Optional[np.ndarray] = None,
):
    """
    Analyze stance-phase torques, GRFs, and COPs.
    
    1. Normalize torques: T_norm = T / (mass * height * 9.8) * 100
    1b. Normalize COP: COP_norm = COP / height * 100
    2. Identify stance phases (GRF_Z > 20N)
    3. Interpolate to 0-100%
    4. Stack and plot
    5. Calculate MAE
    """
    print("\n📊 Analyzing Stance Phase Torques...")
    
    mass = data['mass'][0, 0]
    height = data['height'][0, 0]
    
    # Normalize by Body Weight * Height
    # BW = mass * 9.8067
    norm_factor = mass * height * 9.8067
    
    full_id_pred, full_id_gt, id_source = compute_full_id_curves(predictions, ground_truth)
    _selected_gt, metric_reference_label, mjx_gt_full, opensim_gt_full, opensim_mask = (
        resolve_full_id_reference_curves(ground_truth)
    )
    if full_id_pred is not None and full_id_gt is not None:
        pred_norm = full_id_pred / norm_factor * 100
        gt_norm = full_id_gt / norm_factor * 100
        mjx_norm = None if mjx_gt_full is None else (np.asarray(mjx_gt_full) / norm_factor * 100)
        opensim_norm = None if opensim_gt_full is None else (np.asarray(opensim_gt_full) / norm_factor * 100)
        if not str(id_source).startswith("ID_GT_MJX") and not str(id_source).startswith("OpenSim_ID_STO"):
            print(f"   ⚠️  Warning: Full ID fallback source: {id_source}")
    else:
        # Fallback to just GRF contribution if full ID isn't available
        pred_norm = -predictions['qfrc_grf_contribution'] / norm_factor * 100
        gt_norm = -ground_truth['tau_grf'] / norm_factor * 100
        mjx_norm = None
        opensim_norm = None
        print("   ⚠️  Warning: Full ID unavailable, showing raw GRF contribution torques.")
    
    # Normalize GRF by Body Weight (%)
    bw = mass * 9.8067
    pred_grf_norm = predictions['grf'] / bw * 100
    gt_grf_norm = ground_truth['grf'] / bw * 100

    # Normalize COP by subject height (%)
    pred_cop_norm = predictions['cop'] / height * 100
    gt_cop_norm = ground_truth['cop'] / height * 100
    
    # Use raw GRF (in Newtons) for stance phase detection threshold
    grf = data['grf_raw']
    valid_eval_mask = (
        _normalize_evaluation_mask(evaluation_mask, len(grf))
        if evaluation_mask is not None
        else np.ones(len(grf), dtype=bool)
    )
    
    dof_names = list(get_dof_display_names(pred_norm.shape[1]))
    selected_left_indices = list(get_selected_left_stance_dof_indices())
    
    # --- Inject KAM Calculation ---
    input_source = data.get("input_source", "processed").lower()
    use_noised = data.get("use_noised", False)
    
    # KAM moment arm sourcing:
    #  - Ground-truth KAM uses measured GT GRF plus the saved GT knee->COP vectors.
    #  - Predicted KAM uses predicted GRF plus a predicted knee->COP vector constructed
    #    from predicted COP, selected ankle/knee global positions, and the selected
    #    world->ground-aligned rotation.
    _trial_dir = Path(data.get("trial_dir", ""))
    _mocap_base = source_processed_dir(_trial_dir, "mocap")
    _input_base = Path(str(data.get("input_processed_dir") or source_processed_dir(_trial_dir, input_source)))
    gt_kam_base = _mocap_base if _mocap_base.exists() else _input_base
    pred_kam_base = _input_base if _input_base.exists() else _mocap_base

    def _load_kam_vectors(base):
        for _name in ("KneeToCOP_Vectors_Mocap.npy", "KneeToCOP_Vectors.npy"):
            _p = base / _name
            if _p.exists():
                return np.load(_p), _name
        return None, None

    knee_cop_gt, _gt_vec_name = _load_kam_vectors(gt_kam_base)
    knee_cop_pred, _pred_vec_name = _load_kam_vectors(pred_kam_base)
    kam_path = (gt_kam_base / _gt_vec_name) if knee_cop_gt is not None else None
    if knee_cop_gt is None:
        print(f"   ⚠️ KAM skipped: no KneeToCOP vectors in {gt_kam_base}", flush=True)
    else:
        print(f"   🔍 KAM moment arm — GT(MoCap): {_gt_vec_name} from {gt_kam_base}; "
              f"pred({input_source}): {_pred_vec_name} from {pred_kam_base}")

    if knee_cop_gt is not None:
        pred_grf = predictions['grf']
        gt_grf = ground_truth['grf']
        knee_cop_pred_from_prediction = predictions.get("predicted_knee_to_cop_vectors")

        # Match lengths for both moment arms.
        n_frames = len(pred_norm)

        def _fit_len(kc):
            if kc is None:
                return None
            if len(kc) > n_frames:
                return kc[:n_frames]
            if len(kc) < n_frames:
                return np.pad(kc, ((0, n_frames - len(kc)), (0, 0)), mode='edge')
            return kc

        knee_cop_gt = _fit_len(knee_cop_gt)
        knee_cop_pred_measured = _fit_len(knee_cop_pred)  # file-loaded input-source arm (pre-overwrite)
        knee_cop_pred = _fit_len(knee_cop_pred_from_prediction)
        if knee_cop_pred is None:
            knee_cop_pred = knee_cop_pred_measured
        if knee_cop_pred is None:
            knee_cop_pred = knee_cop_gt  # last-resort fallback if predicted geometry is unavailable

        # KAM = z_vec * GRF_y - y_vec * GRF_z. GT uses the measured GT arm; the
        # prediction uses the arm induced by predicted COP.
        kam_l_pred = (knee_cop_pred[:, 5] * pred_grf[:, 4]) - (knee_cop_pred[:, 4] * pred_grf[:, 5])
        kam_l_pred_norm = kam_l_pred / norm_factor * 100
        kam_l_gt = (knee_cop_gt[:, 5] * gt_grf[:, 4]) - (knee_cop_gt[:, 4] * gt_grf[:, 5])
        kam_l_gt_norm = kam_l_gt / norm_factor * 100

        kam_l_mjx = np.full_like(kam_l_gt_norm, np.nan)
        kam_l_opensim = np.full_like(kam_l_gt_norm, np.nan)

        # --- Right Leg KAM ---
        try:
            kam_r_pred = (knee_cop_pred[:, 2] * pred_grf[:, 1]) - (knee_cop_pred[:, 1] * pred_grf[:, 2])
            kam_r_pred_norm = kam_r_pred / norm_factor * 100
            kam_r_gt = (knee_cop_gt[:, 2] * gt_grf[:, 1]) - (knee_cop_gt[:, 1] * gt_grf[:, 2])
            kam_r_gt_norm = kam_r_gt / norm_factor * 100
            kam_r_mjx = np.full_like(kam_r_gt_norm, np.nan)
            kam_r_opensim = np.full_like(kam_r_gt_norm, np.nan)
        except Exception as e:
            # Fallback if something goes wrong with indexing
            kam_r_pred_norm = np.zeros_like(kam_l_pred_norm)
            kam_r_gt_norm = np.zeros_like(kam_l_gt_norm)
            kam_r_mjx = np.full_like(kam_l_gt_norm, np.nan)
            kam_r_opensim = np.full_like(kam_l_gt_norm, np.nan)
            print(f"Warning: R KAM err {e}")

        pred_norm = np.column_stack([pred_norm, kam_l_pred_norm, kam_r_pred_norm])
        gt_norm = np.column_stack([gt_norm, kam_l_gt_norm, kam_r_gt_norm])
        
        if mjx_norm is not None:
            mjx_norm = np.column_stack([mjx_norm, kam_l_mjx, kam_r_mjx])
        if opensim_norm is not None:
            opensim_norm = np.column_stack([opensim_norm, kam_l_opensim, kam_r_opensim])
            
        new_idx_l = pred_norm.shape[1] - 2
        selected_left_indices.append(new_idx_l)
        dof_names = list(get_dof_display_names(pred_norm.shape[1]))

    # ------------------------------

    # Helper to find stance phases
    def get_stance_phases(grf_z, threshold=LEFT_STANCE_THRESHOLD_N):
        is_stance = (np.abs(grf_z) > threshold) & valid_eval_mask
        diff = np.diff(is_stance.astype(int), prepend=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Handle edge cases
        if len(ends) < len(starts):
            ends = np.append(ends, len(grf_z))
        if len(starts) > len(ends): # Should not happen with prepend=0 logic but safety first
            starts = starts[:len(ends)]
            
        phases = []
        for s, e in zip(starts, ends):
            if e - s > 5: # Minimum duration filter
                phases.append((s, e))
        return phases

    all_phases = get_stance_phases(grf[:, 5])
    durations = [e - s for s, e in all_phases]
    avg_dur = np.mean(durations) if durations else 0
    print(
        "   🦶 Left Stance: "
        f"Found {len(all_phases)} phases, average duration: {avg_dur:.2f} frames"
    )

    if len(all_phases) > 2:
        phases = all_phases[1:-1]
    elif len(all_phases) == 2:
        if (all_phases[0][1] - all_phases[0][0]) >= (all_phases[1][1] - all_phases[1][0]):
            phases = [all_phases[0]]
        else:
            phases = [all_phases[1]]
    elif len(all_phases) == 1:
        phases = all_phases
    else:
        phases = []

    all_results = {}
    for dof_idx in selected_left_indices:
        dof_name = dof_names[dof_idx]
        stacked_pred = []
        stacked_gt = []
        stacked_mjx = []
        stacked_opensim = []

        for s, e in phases:
            seg_pred = pred_norm[s:e, dof_idx]
            seg_gt = gt_norm[s:e, dof_idx]

            x_old = np.linspace(0, 100, len(seg_pred))
            x_new = np.linspace(0, 100, 101)

            f_pred = interp1d(x_old, seg_pred, kind='linear')
            f_gt = interp1d(x_old, seg_gt, kind='linear')

            stacked_pred.append(f_pred(x_new))
            stacked_gt.append(f_gt(x_new))
            if mjx_norm is not None:
                stacked_mjx.append(interp1d(x_old, mjx_norm[s:e, dof_idx], kind='linear')(x_new))
            if opensim_norm is not None and (
                opensim_mask is None or (dof_idx < len(opensim_mask) and opensim_mask[dof_idx])
            ):
                stacked_opensim.append(interp1d(x_old, opensim_norm[s:e, dof_idx], kind='linear')(x_new))

        if stacked_pred:
            all_results[dof_name] = {
                'pred': np.array(stacked_pred),
                'gt': np.array(stacked_gt),
                'mjx': None if not stacked_mjx else np.array(stacked_mjx),
                'opensim': None if not stacked_opensim else np.array(stacked_opensim),
                'avg_duration': avg_dur
            }
        else:
            all_results[dof_name] = None
    
    # Calculate MAE first
    mae_report = {}
    for dof_name, res in all_results.items():
        if res is not None:
            mae = np.mean(np.abs(res['pred'] - res['gt']))
            mae_report[dof_name] = float(mae)

    if no_plots:
        return mae_report, all_results

    # Plotting
    # Create a large figure
    n_dofs = len(all_results)
    cols = 4
    rows = (n_dofs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    
    for i, (dof_name, res) in enumerate(all_results.items()):
        ax = axes[i]
        if res is None:
            ax.text(0.5, 0.5, "No Stance Phases", ha='center', va='center')
            continue
            
        preds = res['pred'] # (N_steps, 101)
        gts = res['gt']     # (N_steps, 101)
        
        # Plot individual steps with low alpha
        for j in range(len(preds)):
            ax.plot(preds[j], color='red', alpha=0.1)
            ax.plot(gts[j], color='blue', alpha=0.1)
            
        # Plot mean
        ax.plot(np.mean(preds, axis=0), color='red', label='Pred', linewidth=2)
        if res.get('opensim') is not None:
            ax.plot(np.mean(res['opensim'], axis=0), color='blue', label='GT', linewidth=2)
        if res.get('mjx') is not None:
            ax.plot(np.mean(res['mjx'], axis=0), color='#6C757D', label='MJX_ID', linewidth=2, linestyle=':')
        elif res.get('opensim') is None:
            ax.plot(np.mean(gts, axis=0), color='blue', label='MJX_ID', linewidth=2)

        mae = mae_report[dof_name]
        ax.set_title(f"{dof_name}\nMAE vs {metric_reference_label}: {mae:.3f}")
        ax.set_xlabel("% Stance")
        if "COP" in dof_name:
            ax.set_ylabel("COP (% height)")
        elif "GRF" in dof_name:
            ax.set_ylabel("GRF (%BW)")
        else:
            ax.set_ylabel("Torque (%BW*H)")
        
        # Add MAE value on the plot
        ax.annotate(f"MAE: {mae:.2f}", xy=(0.5, 0.9), xycoords='axes fraction',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    if not no_plots and not lightweight:
        save_path = Path(output_dir) / f"{trial_name}_stance_analysis.png"
        plt.savefig(save_path, dpi=150)
        print(f"   📈 Saved stance analysis plot to {save_path}")
    plt.close()
    
    # Save MAE report
    if not lightweight:
        report_path = Path(output_dir) / f"{trial_name}_mae_report.json"
        # Convert numpy floats to python floats
        mae_report_serializable = {k: float(v) for k, v in mae_report.items()}
        with open(report_path, 'w') as f:
            json.dump(mae_report_serializable, f, indent=2)
        print(f"   📄 Saved MAE report to {report_path}")
    
    return mae_report, all_results


def _compute_average_mae_per_dof(overall_mae: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Average MAE per DOF across trials for a single input source."""
    all_dofs = set()
    for trial_mae in overall_mae.values():
        all_dofs.update(trial_mae.keys())

    dof_averages = {}
    for dof in all_dofs:
        dof_vals = [trial_mae[dof] for trial_mae in overall_mae.values() if dof in trial_mae]
        if dof_vals:
            dof_averages[dof] = float(np.mean(dof_vals))
    return dof_averages


def _extract_stance_cop_mae_percent_height(mae_report: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Return the stance-phase COP MAE entries normalized by subject height (%)."""
    if not mae_report:
        return {}

    cop_keys = (
        "COP_X_Right",
        "COP_Z_Right",
        "COP_X_Left",
        "COP_Z_Left",
    )
    return {
        key: float(mae_report[key])
        for key in cop_keys
        if key in mae_report and np.isfinite(mae_report[key])
    }


def _extract_left_stance_kam_metrics(
    mae_report: Optional[Dict[str, float]],
    stance_results: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """Return left-stance Knee Adduction Moment summary metrics when available."""
    metrics: Dict[str, float] = {}
    if mae_report and LEFT_STANCE_KAM_DOF_NAME in mae_report:
        kam_mae = float(mae_report[LEFT_STANCE_KAM_DOF_NAME])
        if np.isfinite(kam_mae):
            metrics["left_stance_kam_mae_bwh"] = kam_mae

    if stance_results and LEFT_STANCE_KAM_DOF_NAME in stance_results:
        kam_entry = stance_results.get(LEFT_STANCE_KAM_DOF_NAME)
        if isinstance(kam_entry, dict):
            pred = np.asarray(kam_entry.get("pred"))
            gt = np.asarray(kam_entry.get("gt"))
            if pred.size > 0 and gt.size > 0 and pred.shape == gt.shape:
                diff = pred - gt
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                if np.isfinite(rmse):
                    metrics["left_stance_kam_rmse_bwh"] = rmse
    return metrics


def _normalize_bilateral_summary_metric_name(
    metric_group: str,
    metric_name: str,
) -> str:
    """Collapse COP/GRF side suffixes while keeping torque DOF names intact."""
    if metric_group in {"cop_mae_percent_height", "grf_mae_percent_bw"}:
        for suffix in ("_Right", "_Left"):
            if metric_name.endswith(suffix):
                return metric_name[: -len(suffix)]
    return metric_name


def _build_flat_bilateral_stance_mae_rows(
    trial_reports: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build flat average rows with right/left stance MAEs side-by-side."""
    metric_specs = (
        ("torque_mae_percent_bwh", "torque", "%BW*H"),
        ("cop_mae_percent_height", "cop", "% height"),
        ("grf_mae_percent_bw", "grf", "%BW"),
    )

    flat_rows: List[Dict[str, object]] = []
    for metric_group, category, unit in metric_specs:
        grouped_values: Dict[str, Dict[str, List[float]]] = {}

        for report in trial_reports.values():
            if not isinstance(report, dict):
                continue

            sides_payload = report.get("sides", {})
            if not isinstance(sides_payload, dict):
                continue

            for side_name in ("right", "left"):
                side_payload = sides_payload.get(side_name, {})
                if not isinstance(side_payload, dict):
                    continue

                metric_payload = side_payload.get(metric_group, {})
                if not isinstance(metric_payload, dict):
                    continue

                for metric_name, value in metric_payload.items():
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        continue

                    normalized_name = _normalize_bilateral_summary_metric_name(
                        metric_group,
                        str(metric_name),
                    )
                    grouped_values.setdefault(
                        normalized_name,
                        {"right": [], "left": []},
                    )[side_name].append(float(value))

        for metric_name in sorted(grouped_values):
            right_vals = grouped_values[metric_name]["right"]
            left_vals = grouped_values[metric_name]["left"]
            flat_rows.append(
                {
                    "category": category,
                    "metric": metric_name,
                    "right_mae": float(np.mean(right_vals)) if right_vals else None,
                    "left_mae": float(np.mean(left_vals)) if left_vals else None,
                    "unit": unit,
                }
            )

    return flat_rows


def _compute_average_bilateral_stance_mae(
    source_trial_reports: Dict[str, Dict[str, Dict[str, object]]],
) -> Dict[str, object]:
    """Average bilateral stance MAE metrics across all processed trials."""
    valid_trial_counts = [
        len(trial_reports)
        for trial_reports in source_trial_reports.values()
        if trial_reports
    ]
    averages: Dict[str, object] = {
        "analysis": "phase_normalized_stance_mae_average_over_trials",
        "phase": "stance",
        "trial_count": int(max(valid_trial_counts)) if valid_trial_counts else 0,
    }

    for source_name, trial_reports in source_trial_reports.items():
        if not trial_reports:
            continue
        averages[source_name] = _build_flat_bilateral_stance_mae_rows(trial_reports)

    return averages


def create_mae_boxplots(
    overall_mae: Dict[str, Dict[str, float]],
    output_dir: str,
    overall_mae_motioncapture: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Generate box plots for stance-phase COP, torque, and GRF MAEs across trials."""
    print("\n📦 Generating MAE box plots...")
    
    # Extract all DOFs present in the reports
    all_dofs = set()
    for trial_mae in overall_mae.values():
        all_dofs.update(trial_mae.keys())
    if overall_mae_motioncapture:
        for trial_mae in overall_mae_motioncapture.values():
            all_dofs.update(trial_mae.keys())
    
    # Group DOFs
    cop_dofs = [d for d in all_dofs if d.upper().startswith('COP_')]
    hip_dofs = [d for d in all_dofs if d.lower().startswith('hip_')]
    lumbar_dofs = [d for d in all_dofs if 'lumbar' in d.lower()]
    ankle_dofs = [d for d in all_dofs if 'ankle' in d.lower() or 'subtalar' in d.lower() or 'mtp' in d.lower()]
    knee_dofs = [d for d in all_dofs if 'knee' in d.lower()]
    grf_dofs = [d for d in all_dofs if 'grf' in d.upper()]
    
    # Sort for consistent plotting
    cop_dofs.sort()
    hip_dofs.sort()
    lumbar_dofs.sort()
    ankle_dofs.sort()
    knee_dofs.sort()
    grf_dofs.sort()
    
    groups = {
        'COP DOFs': cop_dofs,
        'Hip DOFs': hip_dofs,
        'Lumbar DOFs': lumbar_dofs,
        'Ankle DOFs': ankle_dofs,
        'Knee DOFs': knee_dofs,
        'GRF DOFs': grf_dofs
    }
    
    # Prepare data for plotting
    fig, axes = plt.subplots(1, len(groups), figsize=(8 * len(groups), 8))
    source_colors = {
        "OpenCap": "#E94F37",
        "MotionCapture": "#1B9E77",
    }
    legend_handles = []
    legend_labels = []
    
    for i, (group_name, dofs) in enumerate(groups.items()):
        if not dofs:
            axes[i].text(0.5, 0.5, f"No data for {group_name}", ha='center', va='center')
            continue
            
        opencap_data = []
        motioncapture_data = []
        labels = []
        
        for dof in dofs:
            opencap_vals = [trial_mae[dof] for trial_mae in overall_mae.values() if dof in trial_mae]
            motioncapture_vals = (
                [trial_mae[dof] for trial_mae in overall_mae_motioncapture.values() if dof in trial_mae]
                if overall_mae_motioncapture else []
            )
            if opencap_vals or motioncapture_vals:
                opencap_data.append(opencap_vals if opencap_vals else [np.nan])
                motioncapture_data.append(motioncapture_vals if motioncapture_vals else [np.nan])
                labels.append(dof.replace('_', '\n'))
        
        if opencap_data:
            positions = np.arange(1, len(labels) + 1, dtype=np.float32)
            width = 0.32 if overall_mae_motioncapture else 0.5
            opencap_positions = positions - (width / 2.0 if overall_mae_motioncapture else 0.0)
            bp_opencap = axes[i].boxplot(
                opencap_data,
                positions=opencap_positions,
                widths=width,
                patch_artist=True,
                manage_ticks=False,
            )
            for patch in bp_opencap['boxes']:
                patch.set(facecolor=source_colors["OpenCap"], alpha=0.55)
            for key in ("medians", "caps", "whiskers"):
                for artist in bp_opencap[key]:
                    artist.set(color=source_colors["OpenCap"])
            if "OpenCap" not in legend_labels and bp_opencap['boxes']:
                legend_handles.append(bp_opencap['boxes'][0])
                legend_labels.append("OpenCap")

            if overall_mae_motioncapture:
                motioncapture_positions = positions + (width / 2.0)
                bp_motioncapture = axes[i].boxplot(
                    motioncapture_data,
                    positions=motioncapture_positions,
                    widths=width,
                    patch_artist=True,
                    manage_ticks=False,
                )
                for patch in bp_motioncapture['boxes']:
                    patch.set(facecolor=source_colors["MotionCapture"], alpha=0.55)
                for key in ("medians", "caps", "whiskers"):
                    for artist in bp_motioncapture[key]:
                        artist.set(color=source_colors["MotionCapture"])
                if "MotionCapture" not in legend_labels and bp_motioncapture['boxes']:
                    legend_handles.append(bp_motioncapture['boxes'][0])
                    legend_labels.append("MotionCapture")

            axes[i].set_xticks(positions)
            axes[i].set_xticklabels(labels)
            axes[i].set_title(group_name, fontsize=14, fontweight='bold')
            if group_name == 'COP DOFs':
                axes[i].set_ylabel('MAE (% height)', fontsize=12)
            elif group_name == 'GRF DOFs':
                axes[i].set_ylabel('MAE (%BW)', fontsize=12)
            else:
                axes[i].set_ylabel('MAE (%BW*H)', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(axis='y', linestyle='--', alpha=0.25)

    fig.suptitle("Stance-Phase MAE Summary: OpenCap vs MotionCapture", fontsize=16, fontweight='bold')
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(legend_labels), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = Path(output_dir) / "mae_boxplots.png"
    plt.savefig(save_path, dpi=300)
    print(f"   📊 Saved MAE box plots to {save_path}")
    plt.close()


def _wandb_artifact_version_number(artifact_name: str) -> int:
    """Extract numeric artifact version from '<entity>/<project>/<name>:v123'."""
    match = re.search(r":v(\d+)$", str(artifact_name).strip())
    if not match:
        return -1
    return int(match.group(1))


def _normalize_wandb_run_path(run_path: str) -> str:
    """Accept either 'entity/project/run_id' or wandb.ai URL and normalize."""
    clean = str(run_path).strip().rstrip("/")
    if "wandb.ai/" not in clean:
        return clean

    # Typical URL: https://wandb.ai/<entity>/<project>/runs/<run_id>
    parts = clean.split("wandb.ai/", 1)[-1].split("/")
    if len(parts) >= 4 and parts[2] == "runs":
        return f"{parts[0]}/{parts[1]}/{parts[3]}"
    return clean


def resolve_checkpoint_path(args: argparse.Namespace) -> str:
    """Resolve local checkpoint path, optionally by downloading from W&B."""
    local_checkpoint = str(args.checkpoint).strip()
    if local_checkpoint:
        checkpoint_path = Path(local_checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    if args.wandb_artifact and args.wandb_run_path:
        raise ValueError("Use only one of --wandb_artifact or --wandb_run_path.")
    if not args.wandb_artifact and not args.wandb_run_path:
        raise ValueError(
            "Provide either --checkpoint (local file) or a W&B source (--wandb_artifact / --wandb_run_path)."
        )

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise ImportError("wandb is required for --wandb_artifact/--wandb_run_path. Install with `pip install wandb`.") from exc

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = str(args.wandb_api_key).strip()

    api = wandb.Api()
    download_root = Path(args.wandb_download_dir).expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    artifact = None
    if args.wandb_artifact:
        artifact_ref = str(args.wandb_artifact).strip()
        if ":" not in artifact_ref:
            artifact_ref = f"{artifact_ref}:{args.wandb_artifact_alias}"
        print(f"☁️  Resolving W&B artifact: {artifact_ref}")
        artifact = api.artifact(artifact_ref, type="model")
    else:
        run_path = _normalize_wandb_run_path(args.wandb_run_path)
        print(f"☁️  Resolving W&B run: {run_path}")
        run = api.run(run_path)
        artifact_name = str(args.wandb_artifact_name).strip()
        alias = str(args.wandb_artifact_alias).strip()

        logged_artifacts = list(run.logged_artifacts())
        candidates = []
        for item in logged_artifacts:
            full_name = str(getattr(item, "name", ""))
            collection_name = full_name.split(":")[0].split("/")[-1]
            item_aliases = list(getattr(item, "aliases", []))
            item_type = str(getattr(item, "type", ""))
            if collection_name != artifact_name:
                continue
            if item_type and item_type != "model":
                continue
            if alias and alias not in item_aliases:
                continue
            candidates.append(item)

        if not candidates and alias:
            print(
                f"   ⚠️ No model artifact named '{artifact_name}' with alias '{alias}' on this run. "
                "Falling back to latest version of that artifact name."
            )
            for item in logged_artifacts:
                full_name = str(getattr(item, "name", ""))
                collection_name = full_name.split(":")[0].split("/")[-1]
                item_type = str(getattr(item, "type", ""))
                if collection_name == artifact_name and (not item_type or item_type == "model"):
                    candidates.append(item)

        if not candidates:
            raise ValueError(
                f"No model artifact named '{artifact_name}' found for run '{run_path}'. "
                "Try --wandb_artifact with a fully qualified artifact reference."
            )

        candidates.sort(key=lambda x: _wandb_artifact_version_number(getattr(x, "name", "")), reverse=True)
        artifact = candidates[0]
        print(f"   ✅ Selected artifact: {artifact.name}")

    artifact_dir = Path(artifact.download(root=str(download_root))).resolve()
    preferred = [
        artifact_dir / "best_model.pkl",
        artifact_dir / "model.pkl",
        artifact_dir / "checkpoint.pkl",
    ]
    checkpoint_path = next((p for p in preferred if p.exists()), None)
    if checkpoint_path is None:
        pkl_files = sorted(artifact_dir.rglob("*.pkl"))
        checkpoint_path = pkl_files[0] if pkl_files else None

    if checkpoint_path is None:
        raise FileNotFoundError(f"No .pkl checkpoint found in downloaded artifact directory: {artifact_dir}")

    print(f"   ✅ Downloaded checkpoint: {checkpoint_path}")
    return str(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Run inference on COP/GRF model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to local best_model.pkl")
    parser.add_argument(
        "--wandb_artifact",
        type=str,
        default="",
        help="W&B model artifact ref, e.g. entity/project/best_model_checkpoint:v12 or :best",
    )
    parser.add_argument(
        "--wandb_run_path",
        type=str,
        default="",
        help="W&B run path (entity/project/run_id or wandb.ai URL).",
    )
    parser.add_argument(
        "--wandb_artifact_name",
        type=str,
        default="best_model_checkpoint",
        help="Artifact collection name when selecting model from --wandb_run_path.",
    )
    parser.add_argument(
        "--wandb_artifact_alias",
        type=str,
        default="best",
        help="Artifact alias when selecting model from --wandb_run_path or shorthand --wandb_artifact.",
    )
    parser.add_argument(
        "--wandb_download_dir",
        type=str,
        default="wandb_artifacts",
        help="Directory for downloaded W&B artifacts.",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default="",
        help="Optional W&B API key override for artifact download.",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--trial_name", type=str, help="Specific trial name to run (e.g. subject1/Trial_1)")
    parser.add_argument("--test_json", type=str, help="Path to JSON file with list of trials")
    parser.add_argument("--all_val", action="store_true", help="Run on all validation trials in checkpoint")
    parser.add_argument("--output", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--window_size", type=int, help="Override window size")
    parser.add_argument("--stride", type=int, help="Override inference stride")
    parser.add_argument(
        "--prediction_margin_frames",
        type=int,
        help="Override the center-only evaluation margin in frames",
    )
    parser.add_argument("--d_model", type=int, help="Override d_model")
    parser.add_argument("--num_layers", type=int, help="Override num_layers")
    parser.add_argument("--ff_dim", type=int, help="Override ff_dim")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    parser.add_argument("--lightweight", action="store_true", help="Minimal metrics only (faster)")
    parser.add_argument("--make_graph", action="store_true", help="Generate publication plots")
    parser.add_argument(
        "--UseNoised",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help="Use the *_noised.npy prediction bundle for model inputs and prediction-side physics. If omitted, inference falls back to the checkpoint hyperparameters when available.",
    )
    parser.add_argument(
        "--includePelvisEuler",
        "--inlcudePelvisEuler",
        dest="includePelvisEuler",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help="If false, drop pelvis_tilt/list/rotation from pos_inputs before building the inference model input. If omitted, inference falls back to the checkpoint hyperparameters when available.",
    )
    parser.add_argument(
        "--UseGRFNormCOP",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help="Decode a model trained on COP_CalcFrame_GroundAligned_GRFNorm.npy. If omitted, inference falls back to checkpoint hyperparameters when available.",
    )
    parser.add_argument(
        "--UseOSFiltering",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help="Use the OpenSim-filtered (_OSfilt) inputs/targets at inference. If omitted, falls back to the checkpoint's UseOSFiltering hyperparameter when available.",
    )
    parser.add_argument(
        "--use_GRF_NoFilt",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help=(
            "Use ProcessedData/GRF_NoFilt_Trimmed.npy as the raw GRF source. "
            "If omitted, inference uses GRF_NoFilt_Trimmed.npy when present and otherwise falls back to GRF_Cleaned.npy. "
            "Pass false to force GRF_Cleaned.npy."
        ),
    )
    parser.add_argument("--RestrictMaxVals", action="store_true", help="Clip normalized temporal inputs to trusted 0.5/99.5 percentile-derived z-score limits")
    parser.add_argument(
        "--restrict_max_vals_path",
        type=str,
        default=None,
        help="Optional path to a (2, input_dim) numpy file containing trusted raw temporal input percentile bounds",
    )
    parser.add_argument("--min_trial_length", type=int, default=1, help="Minimum trial length to process")
    parser.add_argument("--OpenCapValDataset", action="store_true", help="Load MoCap ground truth from subject folders")
    parser.add_argument("--OpenCapDataset", action="store_true", help="Alias for --OpenCapValDataset")
    parser.add_argument(
        "--use_OpenSimID_GT",
        nargs="?",
        const=True,
        default=False,
        type=_parse_optional_bool_arg,
        help="If true, use aligned OpenSim ID STO torques as the torque/full-ID ground truth where available, while still plotting MJX_ID for reference.",
    )
    parser.add_argument(
        "--use_recalculated_opensim_id_gt",
        nargs="?",
        const=True,
        default=False,
        type=_parse_optional_bool_arg,
        help=(
            "If true, use MoCap/OpenSim_ID_recalculated.npy as the torque/full-ID ground truth "
            "for evaluation and plots (with original MoCap/ID_GT_MJX.npy shown as reference)."
        ),
    )
    parser.add_argument(
        "--useGTJacobAndRot",
        "--use_gt_jacob_and_rot",
        dest="use_gt_jacob_and_rot",
        nargs="?",
        const=True,
        default=False,
        type=_parse_optional_bool_arg,
        help=(
            "Use GT/MoCap Jacobian, rotation, qfrc_inverse, ankle positions, and knee positions "
            "for torque/KAM reconstruction at inference."
        ),
    )
    parser.add_argument(
        "--clear_jax_cache_every",
        type=int,
        default=0,
        help="Clear JAX compile caches every N trials (0 disables). Helps memory stability in long runs.",
    )
    
    args = parser.parse_args()
    if args.OpenCapDataset:
        args.OpenCapValDataset = True

    if RUNTIME_ENV_APPLIED:
        print("🔒 Applied runtime safety env defaults:")
        for _k in sorted(RUNTIME_ENV_APPLIED.keys()):
            print(f"   {_k}={os.environ.get(_k)}")

    try:
        checkpoint_path = resolve_checkpoint_path(args)
    except Exception as e:
        print(f"❌ Failed to resolve checkpoint: {e}")
        return
    print(f"📌 Using checkpoint: {checkpoint_path}")
    
    # 1. Determine list of trials
    trials_to_run = []
    
    if args.trial_name:
        trials_to_run.append(args.trial_name)
    
    elif args.test_json:
        with open(args.test_json, 'r') as f:
            test_data = json.load(f)
        
        # Format could be list of strings or list of trial objects
        for t in test_data:
            if isinstance(t, str):
                trials_to_run.append(t)
            elif isinstance(t, dict) and "trial_name" in t:
                trials_to_run.append(t["trial_name"])
            elif isinstance(t, dict) and "trial" in t:
                 trials_to_run.append(t["trial"])
        print(f"📂 Loaded {len(trials_to_run)} trials from {args.test_json}")
        
    elif args.all_val:
        # Load validation trials from checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        val_trials = checkpoint.get('val_trials', [])
        if not val_trials:
            # Try to load from split file
            split_file = Path(checkpoint_path).parent / "train_val_split.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    split_info = json.load(f)
                val_trials = split_info.get('val_trials', [])
        
        if not val_trials:
            print("❌ No validation trials found in checkpoint or split file")
            return
            
        for t in val_trials:
            if isinstance(t, dict):
                # Use trial_name directly; it often contains the subject name anyway
                # E.g., "GaitRetraining_Subject112/Trial_17"
                trials_to_run.append(t['trial_name'])
            else:
                trials_to_run.append(str(t))
        print(f"📂 Loaded {len(trials_to_run)} validation trials")
        
        # Ensure we only run these validation trials and label them correctly
        # We don't need to do anything extra here as trials_to_run is now strictly val_trials
    
    if not trials_to_run:
        print("❌ No trials specified. Use --trial_name, --test_json, or --all_val")
        return

    # 2. Run inference on each trial
    all_metrics = []
    overall_mae = {}
    overall_mae_motioncapture = {}
    aggregated_stance_data = {}
    
    # Create main output directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    for trial_idx, trial in enumerate(tqdm(trials_to_run, desc="Running inference"), start=1):
        try:
            mae_report, metrics, predictions, ground_truth, time_axis, stance_results, secondary_mae_report = run_inference(
                checkpoint_path=checkpoint_path,
                data_dir=args.data_dir,
                trial_name=trial,
                output_dir=str(output_base),
                window_size=args.window_size,
                stride=args.stride,
                prediction_margin_frames=args.prediction_margin_frames,
                d_model=args.d_model,
                num_layers=args.num_layers,
                ff_dim=args.ff_dim,
                no_plots=args.no_plots,
                lightweight=args.lightweight,
                make_graph=args.make_graph,
                use_noised=args.UseNoised,
                include_pelvis_euler=args.includePelvisEuler,
                use_grf_norm_cop=args.UseGRFNormCOP,
                use_grf_nofilt=args.use_GRF_NoFilt,
                use_os_filtering=args.UseOSFiltering,
                use_gt_jacob_and_rot=args.use_gt_jacob_and_rot,
                min_trial_length=args.min_trial_length,
                opencap_val_dataset=args.OpenCapValDataset,
                restrict_max_vals=args.RestrictMaxVals,
                restrict_max_vals_path=args.restrict_max_vals_path,
                use_OpenSimID_GT=args.use_OpenSimID_GT,
                use_recalculated_opensim_id_gt=args.use_recalculated_opensim_id_gt,
            )
            
            if mae_report:
                overall_mae[trial] = mae_report
            if secondary_mae_report:
                overall_mae_motioncapture[trial] = secondary_mae_report
            if metrics:
                all_metrics.append(metrics)
            
            # Aggregate stance phase data
            if stance_results:
                for dof_name, res in stance_results.items():
                    if res is not None:
                        if dof_name not in aggregated_stance_data:
                            aggregated_stance_data[dof_name] = {'pred': [], 'gt': []}
                        aggregated_stance_data[dof_name]['pred'].append(res['pred'])
                        aggregated_stance_data[dof_name]['gt'].append(res['gt'])
                        
        except Exception as e:
            print(f"\n❌ Error running inference on {trial}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            gc.collect()
            if args.clear_jax_cache_every > 0 and (trial_idx % args.clear_jax_cache_every == 0):
                try:
                    jax.clear_caches()
                except Exception:
                    pass

    if all_metrics and any('motioncapture_input_torque_mae_bwh' in m for m in all_metrics):
        trial_details = {}
        opencap_mae_vals = []
        motioncapture_mae_vals = []
        opencap_rmse_vals = []
        motioncapture_rmse_vals = []
        stance_cop_keys = (
            "COP_X_Right",
            "COP_Z_Right",
            "COP_X_Left",
            "COP_Z_Left",
        )
        opencap_stance_cop_vals = {key: [] for key in stance_cop_keys}
        motioncapture_stance_cop_vals = {key: [] for key in stance_cop_keys}

        for metrics in all_metrics:
            trial_name = metrics.get('trial_name', 'unknown_trial')
            opencap_mae = float(metrics.get('opencap_input_torque_mae_bwh', metrics.get('torque_mae_bwh', np.nan)))
            motioncapture_mae = float(metrics.get('motioncapture_input_torque_mae_bwh', np.nan))
            opencap_rmse = float(metrics.get('opencap_input_torque_rmse_bwh', metrics.get('torque_rmse_bwh', np.nan)))
            motioncapture_rmse = float(metrics.get('motioncapture_input_torque_rmse_bwh', np.nan))
            opencap_stance_cop = metrics.get(
                'opencap_input_stance_cop_mae_percent_height',
                metrics.get('stance_cop_mae_percent_height', {}),
            )
            motioncapture_stance_cop = metrics.get(
                'motioncapture_input_stance_cop_mae_percent_height',
                {},
            )

            trial_details[trial_name] = {
                "opencap_input": {
                    "torque_mae_bwh_percent": opencap_mae,
                    "torque_rmse_bwh_percent": opencap_rmse,
                    "stance_cop_mae_percent_height": {
                        key: float(opencap_stance_cop[key])
                        for key in stance_cop_keys
                        if key in opencap_stance_cop and np.isfinite(opencap_stance_cop[key])
                    },
                },
                "motioncapture_input": {
                    "torque_mae_bwh_percent": motioncapture_mae,
                    "torque_rmse_bwh_percent": motioncapture_rmse,
                    "stance_cop_mae_percent_height": {
                        key: float(motioncapture_stance_cop[key])
                        for key in stance_cop_keys
                        if key in motioncapture_stance_cop and np.isfinite(motioncapture_stance_cop[key])
                    },
                },
            }

            if np.isfinite(opencap_mae):
                opencap_mae_vals.append(opencap_mae)
            if np.isfinite(motioncapture_mae):
                motioncapture_mae_vals.append(motioncapture_mae)
            if np.isfinite(opencap_rmse):
                opencap_rmse_vals.append(opencap_rmse)
            if np.isfinite(motioncapture_rmse):
                motioncapture_rmse_vals.append(motioncapture_rmse)
            for key in stance_cop_keys:
                if key in opencap_stance_cop and np.isfinite(opencap_stance_cop[key]):
                    opencap_stance_cop_vals[key].append(float(opencap_stance_cop[key]))
                if key in motioncapture_stance_cop and np.isfinite(motioncapture_stance_cop[key]):
                    motioncapture_stance_cop_vals[key].append(float(motioncapture_stance_cop[key]))

        comparison_summary = {
            "ground_truth_source": "MoCap",
            "torque_metric_scope": all_metrics[0].get("torque_metric_scope"),
            "torque_metric_side": all_metrics[0].get("torque_metric_side"),
            "torque_metric_phase": all_metrics[0].get("torque_metric_phase"),
            "torque_metric_dof_names": all_metrics[0].get("torque_metric_dof_names", []),
            "averages": {
                "opencap_input": {
                    "torque_mae_bwh_percent": float(np.mean(opencap_mae_vals)) if opencap_mae_vals else None,
                    "torque_rmse_bwh_percent": float(np.mean(opencap_rmse_vals)) if opencap_rmse_vals else None,
                    "stance_cop_mae_percent_height": {
                        key: float(np.mean(values)) if values else None
                        for key, values in opencap_stance_cop_vals.items()
                    },
                },
                "motioncapture_input": {
                    "torque_mae_bwh_percent": float(np.mean(motioncapture_mae_vals)) if motioncapture_mae_vals else None,
                    "torque_rmse_bwh_percent": float(np.mean(motioncapture_rmse_vals)) if motioncapture_rmse_vals else None,
                    "stance_cop_mae_percent_height": {
                        key: float(np.mean(values)) if values else None
                        for key, values in motioncapture_stance_cop_vals.items()
                    },
                },
            },
            "trial_details": trial_details,
        }

        comparison_summary_path = output_base / "overall_input_comparison_summary.json"
        with open(comparison_summary_path, "w") as f:
            json.dump(comparison_summary, f, indent=2)
        print(f"✅ Saved OpenCap vs MotionCapture summary to: {comparison_summary_path}")
        if comparison_summary["averages"]["opencap_input"]["torque_mae_bwh_percent"] is not None:
            print(
                "   OpenCap mean Torque MAE BWH: "
                f"{comparison_summary['averages']['opencap_input']['torque_mae_bwh_percent']:.3f} %BW*H"
            )
        if comparison_summary["averages"]["motioncapture_input"]["torque_mae_bwh_percent"] is not None:
            print(
                "   MotionCapture mean Torque MAE BWH: "
                f"{comparison_summary['averages']['motioncapture_input']['torque_mae_bwh_percent']:.3f} %BW*H"
            )

    bilateral_source_reports: Dict[str, Dict[str, Dict[str, object]]] = {}
    opencap_bilateral_trial_reports = {
        metrics.get("trial_name", f"trial_{idx}"): metrics.get(
            "opencap_input_bilateral_stance_mae_report",
            metrics.get("bilateral_stance_mae_report"),
        )
        for idx, metrics in enumerate(all_metrics)
        if isinstance(
            metrics.get(
                "opencap_input_bilateral_stance_mae_report",
                metrics.get("bilateral_stance_mae_report"),
            ),
            dict,
        )
    }
    motioncapture_bilateral_trial_reports = {
        metrics.get("trial_name", f"trial_{idx}"): metrics.get(
            "motioncapture_input_bilateral_stance_mae_report"
        )
        for idx, metrics in enumerate(all_metrics)
        if isinstance(metrics.get("motioncapture_input_bilateral_stance_mae_report"), dict)
    }

    if motioncapture_bilateral_trial_reports:
        bilateral_source_reports["opencap_input_mae"] = opencap_bilateral_trial_reports
        bilateral_source_reports["motioncapture_input_mae"] = motioncapture_bilateral_trial_reports
    elif opencap_bilateral_trial_reports:
        bilateral_source_reports["primary_input_mae"] = opencap_bilateral_trial_reports

    if bilateral_source_reports:
        bilateral_average_report = _compute_average_bilateral_stance_mae(
            bilateral_source_reports
        )
        bilateral_average_path = output_base / "overall_stance_mae_both_legs_average.json"
        with open(bilateral_average_path, "w") as f:
            json.dump(bilateral_average_report, f, indent=2)
        print(f"✅ Saved averaged bilateral stance MAE report to: {bilateral_average_path}")
            
    # 3. Generate summary results if multiple trials were processed
    if len(all_metrics) > 1:
        # Save overall MAE report
        if overall_mae:
            dof_averages = _compute_average_mae_per_dof(overall_mae)
            mocap_dof_averages = _compute_average_mae_per_dof(overall_mae_motioncapture) if overall_mae_motioncapture else {}
            report_data = {
                "torque_metric_scope": "left_stance_selected_dofs",
                "torque_metric_side": "left",
                "torque_metric_phase": "stance",
                "torque_metric_dof_names": list(SELECTED_LEFT_STANCE_DOF_NAMES),
                "average_mae_per_dof": dof_averages,
                "average_mae_per_dof_opencap_input": dof_averages,
                "average_mae_per_dof_motioncapture_input": mocap_dof_averages,
                "trial_details": overall_mae,
                "trial_details_opencap_input": overall_mae,
                "trial_details_motioncapture_input": overall_mae_motioncapture,
            }
            
            mae_report_path = output_base / "overall_mae_report.json"
            with open(mae_report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"✅ Saved overall MAE report with averages to: {mae_report_path}")
            
            # Generate box plots
            create_mae_boxplots(overall_mae, str(output_base), overall_mae_motioncapture=overall_mae_motioncapture)
            
        # Save and process aggregated stance data
        if aggregated_stance_data:
            stance_data_path = output_base / "aggregated_stance_data.pkl"
            with open(stance_data_path, 'wb') as f:
                pickle.dump(aggregated_stance_data, f)
            
            # Compute summary statistics (MAE per DOF across all trials)
            summary_stats = {}
            for dof, data in aggregated_stance_data.items():
                all_preds = np.vstack(data['pred'])
                all_gts = np.vstack(data['gt'])
                diff = all_preds - all_gts
                summary_stats[dof] = {
                    'MAE': float(np.mean(np.abs(diff))),
                    'RMSE': float(np.sqrt(np.mean(diff**2))),
                    'Count': int(all_preds.shape[0])
                }
            
            stats_path = output_base / "aggregated_stance_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            print(f"✅ Saved aggregated stance statistics to: {stats_path}")

        rotation_jacobian_summary_path = output_base / "summary_dashboard" / "rotation_jacobian_comparison_summary.html"
        create_rotation_jacobian_summary_dashboard(
            all_metrics,
            save_path=str(rotation_jacobian_summary_path),
        )
        create_summary_dashboard(all_metrics, str(output_base))


if __name__ == "__main__":
    main()
