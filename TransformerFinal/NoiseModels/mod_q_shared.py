"""Shared helpers for the mod_q training/inference pipeline.

This module keeps the clean-kinematics mod_q path self-contained so the train
and inference entrypoints can share one set of schema, input, model, and
physics helpers without duplicating a third copy of the old pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
import xml.etree.ElementTree as ET
from runtime_model_utils import ensure_modq_runtime_xml, resolve_modq_runtime_xml

try:  # Optional at import time; physics helpers only need it when called.
    import mujoco
    from mujoco import mjx
except Exception:  # pragma: no cover - defensive fallback for import-time safety.
    mujoco = None
    mjx = None


# =============================================================================
# Fixed mod_q schema
# =============================================================================

MOD_Q_COP_DIM = 4
MOD_Q_GRF_DIM = 6
MOD_Q_GRM_DIM = 2
MOD_Q_CONTACT_DIM = 2
MOD_Q_POS_DIM = 16
MOD_Q_VEL_DIM = 19
MOD_Q_ACC_DIM = 19

MOD_Q_OUTPUT_DIMS = {
    "cop": MOD_Q_COP_DIM,
    "grf": MOD_Q_GRF_DIM,
    "grm": MOD_Q_GRM_DIM,
    "contact": MOD_Q_CONTACT_DIM,
    "pos": MOD_Q_POS_DIM,
    "vel": MOD_Q_VEL_DIM,
    "acc": MOD_Q_ACC_DIM,
}

MOD_Q_OUTPUT_SCHEMA = (
    ("cop", 0, 4),
    ("grf", 4, 10),
    ("grm", 10, 12),
    ("contact", 12, 14),
    ("pos", 14, 30),
    ("vel", 30, 49),
    ("acc", 49, 68),
)

MOD_Q_OUTPUT_DIM = 68
MOD_Q_KINEMATIC_LAYOUT = {
    "pos": {"start": 14, "end": 30, "dim": MOD_Q_POS_DIM},
    "vel": {"start": 30, "end": 49, "dim": MOD_Q_VEL_DIM},
    "acc": {"start": 49, "end": 68, "dim": MOD_Q_ACC_DIM},
}
MOD_Q_QPRIME_LAYOUT = MOD_Q_KINEMATIC_LAYOUT

MOD_Q_INPUT_BLOCKS = (
    {"name": "pelvis_rot", "dim": 6},
    {"name": "pos", "dim": MOD_Q_POS_DIM},
    {"name": "vel", "dim": 19},
    {"name": "acc", "dim": 19},
    {"name": "com_r", "dim": 3},
    {"name": "com_l", "dim": 3},
    {"name": "com_accel", "dim": 3},
    {"name": "foot_progression_angle", "dim": 2},
    {"name": "calcn_to_floor_angle", "dim": 2},
)

MOD_Q_FORCED_FLAGS = {
    "UseNoised": True,
    "includePelvisEuler": True,
    "PredictJacobian": False,
    "DeviationLearning": True,
}

# Backward-compatible aliases expected by the new entrypoints.
MODQ_FORCED_FLAGS = MOD_Q_FORCED_FLAGS
MODQ_QPRIME_LAYOUT = MOD_Q_QPRIME_LAYOUT
MODQ_OUTPUT_DIM = MOD_Q_OUTPUT_DIM

MOD_Q_DEFAULT_STATIC_DIM = 8
MOD_Q_DEFAULT_NUM_HEADS = 4


def mod_q_output_schema_dict() -> Dict[str, Tuple[int, int]]:
    """Return the fixed output schema as a compact dict."""
    return {name: (start, end) for name, start, end in MOD_Q_OUTPUT_SCHEMA}


def mod_q_output_dims() -> Dict[str, int]:
    return dict(MOD_Q_OUTPUT_DIMS)


def mod_q_input_feature_blocks() -> List[Dict[str, Any]]:
    return [dict(block) for block in MOD_Q_INPUT_BLOCKS]


def mod_q_checkpoint_metadata() -> Dict[str, Any]:
    """Metadata that should be written into checkpoints/hyperparameters."""
    return {
        "model_type": "mod_q",
        "output_schema": mod_q_output_schema_dict(),
        "qprime_layout": dict(MOD_Q_QPRIME_LAYOUT),
        "kinematic_layout": dict(MOD_Q_KINEMATIC_LAYOUT),
        "input_feature_blocks": mod_q_input_feature_blocks(),
        "subject_grouped_batches": True,
        "physics_backend": "mjx_jit_differentiable",
        "rotation_loss_type": "geodesic_mse",
        "forced_flags": dict(MOD_Q_FORCED_FLAGS),
        "UseNoised": True,
        "includePelvisEuler": True,
        "PredictJacobian": False,
        "DeviationLearning": False,
        "contact_boolean_is_model_input": False,
        "derived_qprime_from_templates": False,
        "kinematics_prediction_mode": "direct_pos_vel_acc",
    }


def mod_q_forced_args() -> Dict[str, Any]:
    """Convenience helper for scripts that want to enforce the mod_q defaults."""
    return dict(MOD_Q_FORCED_FLAGS)


# Populate alias values after helper definitions exist.
MODQ_INPUT_FEATURE_BLOCKS = mod_q_input_feature_blocks()
MODQ_OUTPUT_SCHEMA = mod_q_output_schema_dict()


# =============================================================================
# Minimal feature helpers
# =============================================================================

def _as_float32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def select_pos_input_columns(pos: np.ndarray, include_pelvis_euler: bool = True) -> np.ndarray:
    """Mirror the current train/data-loader behavior for position features."""
    pos = _as_float32(pos)
    if include_pelvis_euler or pos.ndim != 2 or pos.shape[1] <= 3:
        return pos
    return pos[:, 3:]


def flatten_jacobian_components(jacp: np.ndarray, jacr: np.ndarray) -> np.ndarray:
    """Flatten jacp/jacr into a single per-frame feature vector."""
    jacp = _as_float32(jacp)
    jacr = _as_float32(jacr)
    leading_shape = tuple(jacp.shape[:-3])
    return np.concatenate(
        [jacp.reshape(leading_shape + (-1,)), jacr.reshape(leading_shape + (-1,))],
        axis=-1,
    )


def build_mod_q_temporal_input(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Build the fixed mod_q temporal input layout."""
    parts = [
        _as_float32(data["pelvis_rot"]),
        _as_float32(data["pos"]),
        _as_float32(data["vel"]),
        _as_float32(data["acc"]),
        _as_float32(data["com_r"]),
        _as_float32(data["com_l"]),
        _as_float32(data["com_accel"]),
        _as_float32(data["foot_progression_angle"]),
        _as_float32(data["calcn_to_floor_angle"]),
    ]
    blocks = [
        {"name": "pelvis_rot", "dim": int(parts[0].shape[-1])},
        {"name": "pos", "dim": int(parts[1].shape[-1])},
        {"name": "vel", "dim": int(parts[2].shape[-1])},
        {"name": "acc", "dim": int(parts[3].shape[-1])},
        {"name": "com_r", "dim": int(parts[4].shape[-1])},
        {"name": "com_l", "dim": int(parts[5].shape[-1])},
        {"name": "com_accel", "dim": int(parts[6].shape[-1])},
        {"name": "foot_progression_angle", "dim": int(parts[7].shape[-1])},
        {"name": "calcn_to_floor_angle", "dim": int(parts[8].shape[-1])},
    ]
    return np.concatenate(parts, axis=1), blocks


def build_mod_q_static_context(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Build the fixed 8-D static context used by the project."""
    height = float(np.asarray(data["height"]).reshape(-1)[0])
    mass = float(np.asarray(data["mass"]).reshape(-1)[0])
    gender_arr = np.asarray(data.get("gender", 0.5), dtype=np.float32).reshape(-1)
    gender = float(gender_arr[0] if gender_arr.size else 0.5)
    patient_size = np.asarray(data.get("patient_size", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(-1)
    patient_size_vec = np.zeros(4, dtype=np.float32)
    patient_size_vec[: min(4, patient_size.size)] = patient_size[: min(4, patient_size.size)]
    forward_vel = float(np.asarray(data.get("forward_vel", 0.0)).reshape(-1)[0])
    return np.array(
        [height, mass, gender, *patient_size_vec.tolist(), forward_vel],
        dtype=np.float32,
    )


# =============================================================================
# Transformer backbone
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        position = jnp.arange(seq_len)
        half_dim = self.dim // 2
        emb_scale = jnp.log(10000.0) / jnp.maximum(half_dim - 1, 1)
        emb_scale = jnp.exp(jnp.arange(half_dim) * -emb_scale)
        emb = position[:, None] * emb_scale[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return x + emb[None, :, :]


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x, x, deterministic=not train)
        x = residual + x

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.ff_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        return residual + x


class ModQTaskHead(nn.Module):
    head_d_model: int = 128
    head_num_layers: int = 2
    head_num_heads: int = 4
    head_ff_dim: int = 256
    output_dim: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.head_d_model)(x)
        x = nn.gelu(x)
        for _ in range(self.head_num_layers):
            x = TransformerBlock(
                d_model=self.head_d_model,
                num_heads=self.head_num_heads,
                ff_dim=self.head_ff_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.output_dim)(x)


class ModQKinematicsToOutputs(nn.Module):
    """Shared mod_q transformer backbone with either a single head or task heads."""

    input_dim: int = 40
    static_dim: int = MOD_Q_DEFAULT_STATIC_DIM
    output_dim: int = MOD_Q_OUTPUT_DIM
    d_model: int = 256
    num_heads: int = MOD_Q_DEFAULT_NUM_HEADS
    num_layers: int = 4
    ff_dim: int = 1024
    dropout_rate: float = 0.1
    use_cnn: bool = True
    cnn_num_layers: int = 2
    cnn_kernel_sizes: tuple = (3, 5)
    use_multitask: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.d_model)(x)
        if self.use_cnn:
            residual = nn.gelu(x)
            kernels = list(self.cnn_kernel_sizes) if len(self.cnn_kernel_sizes) > 0 else [3]
            for i in range(self.cnn_num_layers):
                kernel = kernels[i] if i < len(kernels) else kernels[-1]
                x = nn.Conv(
                    features=self.d_model,
                    kernel_size=(kernel,),
                    strides=(1,),
                    padding="same",
                    name=f"cnn_conv_{i}",
                )(x)
                x = nn.gelu(x)
            alpha = self.param("cnn_gate", nn.initializers.ones, (1, 1, self.d_model))
            beta = self.param("res_gate", nn.initializers.ones, (1, 1, self.d_model))
            x = residual * beta + alpha * x
            x = nn.gelu(x)
        else:
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        x = SinusoidalPosEmb(dim=self.d_model)(x)

        static_token = nn.Dense(self.d_model)(static_context)
        static_token = nn.gelu(static_token)
        static_token = nn.LayerNorm()(static_token)
        static_token = jnp.expand_dims(static_token, axis=1)
        x = jnp.concatenate([static_token, x], axis=1)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = x[:, 1:, :]
        x = nn.LayerNorm()(x)

        if not self.use_multitask:
            return nn.Dense(self.output_dim)(x)

        cop = ModQTaskHead(output_dim=MOD_Q_COP_DIM, dropout_rate=self.dropout_rate, name="cop_head")(x, train=train)
        grf = ModQTaskHead(output_dim=MOD_Q_GRF_DIM, dropout_rate=self.dropout_rate, name="grf_head")(x, train=train)
        grm = ModQTaskHead(output_dim=MOD_Q_GRM_DIM, dropout_rate=self.dropout_rate, name="grm_head")(x, train=train)
        contact = ModQTaskHead(output_dim=MOD_Q_CONTACT_DIM, dropout_rate=self.dropout_rate, name="contact_head")(x, train=train)
        pos = ModQTaskHead(output_dim=MOD_Q_POS_DIM, dropout_rate=self.dropout_rate, name="pos_head")(x, train=train)
        vel = ModQTaskHead(output_dim=MOD_Q_VEL_DIM, dropout_rate=self.dropout_rate, name="vel_head")(x, train=train)
        acc = ModQTaskHead(output_dim=MOD_Q_ACC_DIM, dropout_rate=self.dropout_rate, name="acc_head")(x, train=train)
        return jnp.concatenate([cop, grf, grm, nn.sigmoid(contact), pos, vel, acc], axis=-1)


ModQTransformer = ModQKinematicsToOutputs


# =============================================================================
# Prediction parsing and physics helpers
# =============================================================================

def split_mod_q_predictions(pred: Any) -> Dict[str, Any]:
    """Split mod_q predictions into their fixed semantic blocks."""
    pred = jnp.asarray(pred)
    grm = pred[..., 10:12]
    return {
        "cop": pred[..., 0:4],
        "grf": pred[..., 4:10],
        "grm": grm,
        "moments": grm,
        "contact": jax.nn.sigmoid(pred[..., 12:14]),
        "pos": pred[..., 14:30],
        "vel": pred[..., 30:49],
        "acc": pred[..., 49:68],
    }


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

POS_INPUT_NPY_IDXS = jnp.asarray([0, 1, 2, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22], dtype=jnp.int32)
VEL_INPUT_NPY_IDXS = jnp.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22], dtype=jnp.int32)
ACC_INPUT_NPY_IDXS = jnp.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22], dtype=jnp.int32)

POS_OUTPUT_Q_IDXS = jnp.asarray([NP_TO_QPOS[int(idx)] for idx in POS_INPUT_NPY_IDXS.tolist()], dtype=jnp.int32)
VEL_OUTPUT_Q_IDXS = jnp.asarray([NP_TO_QPOS[int(idx)] for idx in VEL_INPUT_NPY_IDXS.tolist()], dtype=jnp.int32)
ACC_OUTPUT_Q_IDXS = jnp.asarray([NP_TO_QPOS[int(idx)] for idx in ACC_INPUT_NPY_IDXS.tolist()], dtype=jnp.int32)
POS_OUTPUT_Q_IDXS_NP = np.asarray(POS_OUTPUT_Q_IDXS, dtype=np.int32)
VEL_OUTPUT_Q_IDXS_NP = np.asarray(VEL_OUTPUT_Q_IDXS, dtype=np.int32)
ACC_OUTPUT_Q_IDXS_NP = np.asarray(ACC_OUTPUT_Q_IDXS, dtype=np.int32)


def _parse_coupling_spec(xml_path: str, model: Any) -> Dict[str, np.ndarray]:
    slave_indices: List[int] = []
    master_indices: List[int] = []
    coeffs: List[List[float]] = []

    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return {
            "slave_idx": np.zeros((0,), dtype=np.int32),
            "master_idx": np.zeros((0,), dtype=np.int32),
            "coeffs": np.zeros((0, 5), dtype=np.float32),
        }

    for equality in root.iter("equality"):
        for joint in equality.iter("joint"):
            slave_name = joint.get("joint1")
            master_name = joint.get("joint2")
            if slave_name is None or master_name is None:
                continue
            slave_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, slave_name)
            master_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, master_name)
            if slave_id < 0 or master_id < 0:
                continue
            poly = joint.get("polycoef", "0 1 0 0 0").split()
            poly_f = [float(c) for c in poly]
            if len(poly_f) < 5:
                poly_f.extend([0.0] * (5 - len(poly_f)))
            coeffs.append(poly_f[:5])
            slave_indices.append(int(model.jnt_dofadr[slave_id]))
            master_indices.append(int(model.jnt_dofadr[master_id]))

    return {
        "slave_idx": np.asarray(slave_indices, dtype=np.int32),
        "master_idx": np.asarray(master_indices, dtype=np.int32),
        "coeffs": np.asarray(coeffs, dtype=np.float32).reshape((-1, 5)) if coeffs else np.zeros((0, 5), dtype=np.float32),
    }


def _apply_couplings_jax(
    qpos: jax.Array,
    qvel: jax.Array,
    qacc: jax.Array,
    slave_idx: jax.Array,
    master_idx: jax.Array,
    coeffs: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    if slave_idx.shape[0] == 0:
        return qpos, qvel, qacc

    theta = qpos[:, master_idx]
    theta_vel = qvel[:, master_idx]
    theta_acc = qacc[:, master_idx]

    c0 = coeffs[:, 0][None, :]
    c1 = coeffs[:, 1][None, :]
    c2 = coeffs[:, 2][None, :]
    c3 = coeffs[:, 3][None, :]
    c4 = coeffs[:, 4][None, :]

    q_slave = c0 + c1 * theta + c2 * theta**2 + c3 * theta**3 + c4 * theta**4
    dq_dtheta = c1 + 2.0 * c2 * theta + 3.0 * c3 * theta**2 + 4.0 * c4 * theta**3
    d2q_dtheta2 = 2.0 * c2 + 6.0 * c3 * theta + 12.0 * c4 * theta**2
    v_slave = dq_dtheta * theta_vel
    a_slave = dq_dtheta * theta_acc + d2q_dtheta2 * theta_vel**2

    qpos = qpos.at[:, slave_idx].set(q_slave)
    qvel = qvel.at[:, slave_idx].set(v_slave)
    qacc = qacc.at[:, slave_idx].set(a_slave)
    return qpos, qvel, qacc


def _build_subject_kinematics_reconstructor(
    *,
    slave_idx: np.ndarray,
    master_idx: np.ndarray,
    coeffs: np.ndarray,
):
    slave_idx_j = jnp.asarray(slave_idx, dtype=jnp.int32)
    master_idx_j = jnp.asarray(master_idx, dtype=jnp.int32)
    coeffs_j = jnp.asarray(coeffs, dtype=jnp.float32)

    def _reconstruct(
        pos_pred: jax.Array,
        vel_pred: jax.Array,
        acc_pred: jax.Array,
        qpos_template: jax.Array,
        qvel_template: jax.Array,
        qacc_template: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        qpos = qpos_template.at[:, POS_OUTPUT_Q_IDXS].set(pos_pred)
        qvel = qvel_template.at[:, VEL_OUTPUT_Q_IDXS].set(vel_pred)
        qacc = qacc_template.at[:, ACC_OUTPUT_Q_IDXS].set(acc_pred)
        return _apply_couplings_jax(qpos, qvel, qacc, slave_idx_j, master_idx_j, coeffs_j)

    return jax.jit(_reconstruct)


def _generic_kinematics_reconstructor(
    pos_pred: jax.Array,
    vel_pred: jax.Array,
    acc_pred: jax.Array,
    qpos_template: jax.Array,
    qvel_template: jax.Array,
    qacc_template: jax.Array,
    slave_idx: jax.Array,
    master_idx: jax.Array,
    coeffs: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    qpos = qpos_template.at[:, POS_OUTPUT_Q_IDXS].set(pos_pred)
    qvel = qvel_template.at[:, VEL_OUTPUT_Q_IDXS].set(vel_pred)
    qacc = qacc_template.at[:, ACC_OUTPUT_Q_IDXS].set(acc_pred)
    return _apply_couplings_jax(qpos, qvel, qacc, slave_idx, master_idx, coeffs)


_GENERIC_KINEMATICS_RECONSTRUCTOR = jax.jit(_generic_kinematics_reconstructor)


def _mujoco_rotation_to_world(rot_w_to_ga: np.ndarray) -> np.ndarray:
    """Convert world->ground-aligned rotations to ground-aligned->world rotations."""
    rot_w_to_ga = np.asarray(rot_w_to_ga, dtype=np.float32)
    return np.swapaxes(rot_w_to_ga, -1, -2)


def cop_ground_aligned_to_world(
    cop_ground_aligned: np.ndarray,
    rot_w_to_ga: np.ndarray,
    ankle_heights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rotate ground-aligned COP vectors back to world coordinates."""
    cop = np.asarray(cop_ground_aligned, dtype=np.float32)
    rot = _mujoco_rotation_to_world(rot_w_to_ga)
    if ankle_heights is None:
        heights = np.zeros((len(cop), 2), dtype=np.float32)
    else:
        heights = np.asarray(ankle_heights, dtype=np.float32)
    if cop.ndim == 2 and cop.shape[1] >= 4:
        cop_r = np.einsum(
            "tij,tj->ti",
            rot[:, 0],
            np.column_stack([cop[:, 0], heights[:, 0], cop[:, 1]]),
        )
        cop_l = np.einsum(
            "tij,tj->ti",
            rot[:, 1],
            np.column_stack([cop[:, 2], heights[:, 1], cop[:, 3]]),
        )
        return np.column_stack([cop_r[:, 0], cop_r[:, 1], cop_r[:, 2], cop_l[:, 0], cop_l[:, 1], cop_l[:, 2]])
    return cop


def compute_full_external_moments(
    cop_pred_unnorm: np.ndarray,
    grf_pred_unnorm: np.ndarray,
    free_moments_pred_unnorm: np.ndarray,
    ankle_heights: np.ndarray,
    rot_w_to_ga: np.ndarray,
) -> np.ndarray:
    """Reconstruct full external moments about each foot origin in world space."""
    cop_pred_unnorm = np.asarray(cop_pred_unnorm, dtype=np.float32)
    grf_pred_unnorm = np.asarray(grf_pred_unnorm, dtype=np.float32)
    free_moments_pred_unnorm = np.asarray(free_moments_pred_unnorm, dtype=np.float32)
    ankle_heights = np.asarray(ankle_heights, dtype=np.float32)
    rot_ga_to_w = _mujoco_rotation_to_world(rot_w_to_ga)

    cop_r_ga = np.column_stack([cop_pred_unnorm[:, 0], ankle_heights[:, 0], cop_pred_unnorm[:, 1]])
    cop_l_ga = np.column_stack([cop_pred_unnorm[:, 2], ankle_heights[:, 1], cop_pred_unnorm[:, 3]])

    cop_r = np.einsum("tij,tj->ti", rot_ga_to_w[:, 0], cop_r_ga)
    cop_l = np.einsum("tij,tj->ti", rot_ga_to_w[:, 1], cop_l_ga)

    grf_r = grf_pred_unnorm[:, 0:3]
    grf_l = grf_pred_unnorm[:, 3:6]
    mom_r = np.column_stack([np.zeros(len(free_moments_pred_unnorm), dtype=np.float32), np.zeros(len(free_moments_pred_unnorm), dtype=np.float32), free_moments_pred_unnorm[:, 0]])
    mom_l = np.column_stack([np.zeros(len(free_moments_pred_unnorm), dtype=np.float32), np.zeros(len(free_moments_pred_unnorm), dtype=np.float32), free_moments_pred_unnorm[:, 1]])

    m_r = np.cross(cop_r, grf_r) + mom_r
    m_l = np.cross(cop_l, grf_l) + mom_l
    return np.column_stack([m_r, m_l])


def compute_tau_grf_from_predictions(
    grf_pred: np.ndarray,
    moments_pred: np.ndarray,
    jacp: np.ndarray,
    jacr: np.ndarray,
) -> np.ndarray:
    """Compute tau_grf = Jp^T F + Jr^T M for both feet."""
    grf_pred = np.asarray(grf_pred, dtype=np.float32)
    moments_pred = np.asarray(moments_pred, dtype=np.float32)
    jacp = np.asarray(jacp, dtype=np.float32)
    jacr = np.asarray(jacr, dtype=np.float32)

    tau = np.zeros((grf_pred.shape[0], jacp.shape[-1]), dtype=np.float32)
    tau += np.einsum("tji,tj->ti", jacp[:, 0], grf_pred[:, 0:3])
    tau += np.einsum("tji,tj->ti", jacr[:, 0], moments_pred[:, 0:3])
    tau += np.einsum("tji,tj->ti", jacp[:, 1], grf_pred[:, 3:6])
    tau += np.einsum("tji,tj->ti", jacr[:, 1], moments_pred[:, 3:6])
    return tau


def project_rotation_matrices(rot: Any, *, xp=np):
    """Project arbitrary 3x3 matrices onto SO(3) with a thin SVD-based helper."""
    rot = xp.asarray(rot)
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {rot.shape}")

    flat = rot.reshape((-1, 3, 3))
    u, _, vh = xp.linalg.svd(flat, full_matrices=False)
    det_uv = xp.linalg.det(u @ vh)
    sign = xp.where(det_uv < 0, -xp.ones_like(det_uv), xp.ones_like(det_uv))
    column_sign = xp.concatenate(
        [xp.ones((sign.shape[0], 2), dtype=rot.dtype), sign[:, None]],
        axis=-1,
    )
    u = u * column_sign[:, None, :]
    proj = u @ vh
    return proj.reshape(rot.shape)


def masked_mean(values: Any, mask: Optional[Any] = None, *, xp=np, eps: float = 1e-8):
    """Compute a broadcast-friendly masked mean for numpy or JAX arrays."""
    values = xp.asarray(values)
    if mask is None:
        return xp.mean(values)
    mask = xp.asarray(mask, dtype=values.dtype)
    while mask.ndim > values.ndim and mask.shape[-1] == 1:
        mask = mask[..., 0]
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    mask = xp.broadcast_to(mask, values.shape)
    weight = xp.sum(mask)
    return xp.sum(values * mask) / xp.maximum(weight, xp.asarray(eps, dtype=values.dtype))


def rotation_geodesic_angle(
    rot_a: Any,
    rot_b: Any,
    *,
    xp=np,
    project: bool = True,
):
    """Return the SO(3) geodesic angle in radians for matching rotation pairs."""
    rot_a = xp.asarray(rot_a)
    rot_b = xp.asarray(rot_b)
    if project:
        rot_a = project_rotation_matrices(rot_a, xp=xp)
        rot_b = project_rotation_matrices(rot_b, xp=xp)
    if rot_a.shape[-2:] != (3, 3) or rot_b.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrices with shape (..., 3, 3), got {rot_a.shape} and {rot_b.shape}"
        )

    rot_err = rot_a @ xp.swapaxes(rot_b, -1, -2)
    trace = xp.sum(xp.diagonal(rot_err, axis1=-2, axis2=-1), axis=-1)
    cos_theta = xp.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    sin_terms = xp.stack(
        [
            rot_err[..., 2, 1] - rot_err[..., 1, 2],
            rot_err[..., 0, 2] - rot_err[..., 2, 0],
            rot_err[..., 1, 0] - rot_err[..., 0, 1],
        ],
        axis=-1,
    )
    sin_theta = 0.5 * xp.linalg.norm(sin_terms, axis=-1)
    return xp.arctan2(sin_theta, cos_theta)


def geodesic_rotation_mse(
    rot_a: Any,
    rot_b: Any,
    mask: Optional[Any] = None,
    *,
    xp=np,
    project: bool = True,
):
    """Return the masked mean-squared geodesic error in radians^2."""
    angle = rotation_geodesic_angle(rot_a, rot_b, xp=xp, project=project)
    return masked_mean(angle**2, mask=mask, xp=xp)


def rotation_geodesic_summary_deg(
    rot_a: Any,
    rot_b: Any,
    mask: Optional[Any] = None,
    *,
    xp=np,
    project: bool = True,
) -> Dict[str, Any]:
    """Return compact geodesic summaries in degrees for the two-foot rotation bundle."""
    angle_deg = xp.degrees(rotation_geodesic_angle(rot_a, rot_b, xp=xp, project=project))
    overall_mean_deg = masked_mean(angle_deg, mask=mask, xp=xp)
    overall_rmse_deg = xp.sqrt(masked_mean(angle_deg**2, mask=mask, xp=xp))

    if mask is None:
        right_mask = None
        left_mask = None
    else:
        mask_arr = xp.asarray(mask)
        if mask_arr.ndim == angle_deg.ndim and mask_arr.shape[-1] == angle_deg.shape[-1]:
            right_mask = mask_arr[..., 0]
            left_mask = mask_arr[..., 1]
        else:
            right_mask = mask_arr
            left_mask = mask_arr

    right_mean_deg = masked_mean(angle_deg[..., 0], mask=right_mask, xp=xp)
    left_mean_deg = masked_mean(angle_deg[..., 1], mask=left_mask, xp=xp)
    return {
        "overall_mean_deg": overall_mean_deg,
        "overall_rmse_deg": overall_rmse_deg,
        "right_mean_deg": right_mean_deg,
        "left_mean_deg": left_mean_deg,
        "mean_deg": overall_mean_deg,
        "rmse_deg": overall_rmse_deg,
    }


def _compose_world_to_ground_aligned_single(R_wb: np.ndarray) -> np.ndarray:
    """Match ProcessData.py ground-aligned calcaneus rotation construction."""
    R_wb = np.asarray(R_wb, dtype=np.float64)
    n_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    x_w = R_wb[:, 0]
    xg_w = x_w - np.dot(x_w, n_w) * n_w
    if np.linalg.norm(xg_w) < 1e-10:
        z_w = R_wb[:, 2]
        xg_w = np.cross(n_w, z_w)
    xg_w = xg_w / max(np.linalg.norm(xg_w), 1e-12)

    yg_w = n_w.copy()
    zg_w = np.cross(xg_w, yg_w)
    zg_w = zg_w / max(np.linalg.norm(zg_w), 1e-12)
    xg_w = np.cross(yg_w, zg_w)
    xg_w = xg_w / max(np.linalg.norm(xg_w), 1e-12)

    R_wg = np.column_stack([xg_w, yg_w, zg_w])
    return (R_wg.T @ R_wb).astype(np.float32)


def _safe_normalize_jax(vec: jax.Array, eps: float = 1e-8) -> jax.Array:
    norm = jnp.linalg.norm(vec)
    return vec / jnp.maximum(norm, jnp.asarray(eps, dtype=vec.dtype))


def _compose_world_to_ground_aligned_single_jax(rot_w_to_body: jax.Array) -> jax.Array:
    """Match ProcessData.py world->ground-aligned calcaneus rotation construction."""
    rot_w_to_body = jnp.asarray(rot_w_to_body)
    n_w = jnp.array([0.0, 0.0, 1.0], dtype=rot_w_to_body.dtype)

    # ProcessData builds the ground-aligned frame from the body axes expressed in
    # world coordinates, so transpose the incoming world->body matrix first.
    R_wb = rot_w_to_body.T
    x_w = R_wb[:, 0]
    xg_proj = x_w - jnp.dot(x_w, n_w) * n_w
    xg_alt = jnp.cross(n_w, R_wb[:, 2])
    xg_w = jnp.where(jnp.linalg.norm(xg_proj) < 1e-10, xg_alt, xg_proj)
    xg_w = _safe_normalize_jax(xg_w)
    yg_w = n_w
    zg_w = _safe_normalize_jax(jnp.cross(xg_w, yg_w))
    xg_w = _safe_normalize_jax(jnp.cross(yg_w, zg_w))
    R_wg = jnp.stack([xg_w, yg_w, zg_w], axis=1)
    R_ga_b = R_wg.T @ R_wb
    return R_ga_b @ rot_w_to_body


def _build_mjx_subject_physics_runner(
    mjx_model: Any,
    *,
    calcn_r_id: int,
    calcn_l_id: int,
):
    """Build a fully differentiable q_prime -> physics runner for one subject model."""

    calcn_r_id = int(calcn_r_id)
    calcn_l_id = int(calcn_l_id)

    def _single_frame(
        qpos_t: jax.Array,
        qvel_t: jax.Array,
        qacc_t: jax.Array,
        cop_t: jax.Array,
        grf_t: jax.Array,
        grm_t: jax.Array,
        ankle_t: jax.Array,
    ) -> Dict[str, jax.Array]:
        data = mjx.make_data(mjx_model).replace(qpos=qpos_t, qvel=qvel_t, qacc=qacc_t)
        data = mjx.inverse(mjx_model, data)

        body_rot_r = data.xmat[calcn_r_id].T
        body_rot_l = data.xmat[calcn_l_id].T
        rot_w_to_ga_r = _compose_world_to_ground_aligned_single_jax(body_rot_r)
        rot_w_to_ga_l = _compose_world_to_ground_aligned_single_jax(body_rot_l)
        rot_w_to_ga = jnp.stack([rot_w_to_ga_r, rot_w_to_ga_l], axis=0)

        jacp_r_nv3, jacr_r_nv3 = mjx.jac(mjx_model, data, data.xpos[calcn_r_id], calcn_r_id)
        jacp_l_nv3, jacr_l_nv3 = mjx.jac(mjx_model, data, data.xpos[calcn_l_id], calcn_l_id)
        jacp_r = jacp_r_nv3.T
        jacr_r = jacr_r_nv3.T
        jacp_l = jacp_l_nv3.T
        jacr_l = jacr_l_nv3.T
        jacp = jnp.stack([jacp_r, jacp_l], axis=0)
        jacr = jnp.stack([jacr_r, jacr_l], axis=0)

        ankle_heights = jnp.asarray(ankle_t, dtype=qpos_t.dtype)
        cop_r_ga = jnp.array([cop_t[0], ankle_heights[0], cop_t[1]], dtype=qpos_t.dtype)
        cop_l_ga = jnp.array([cop_t[2], ankle_heights[1], cop_t[3]], dtype=qpos_t.dtype)
        cop_r_world = rot_w_to_ga_r.T @ cop_r_ga
        cop_l_world = rot_w_to_ga_l.T @ cop_l_ga
        cop_world = jnp.concatenate([cop_r_world, cop_l_world], axis=0)

        grf_r = grf_t[:3]
        grf_l = grf_t[3:6]
        grm_r = jnp.array([0.0, 0.0, grm_t[0]], dtype=qpos_t.dtype)
        grm_l = jnp.array([0.0, 0.0, grm_t[1]], dtype=qpos_t.dtype)
        moments_r = jnp.cross(cop_r_world, grf_r) + grm_r
        moments_l = jnp.cross(cop_l_world, grf_l) + grm_l
        full_moments = jnp.concatenate([moments_r, moments_l], axis=0)

        tau_grf = (
            jacp_r_nv3 @ grf_r
            + jacr_r_nv3 @ moments_r
            + jacp_l_nv3 @ grf_l
            + jacr_l_nv3 @ moments_l
        )
        # MuJoCo's raw qfrc_inverse still carries the inverse-solver treatment of
        # constraints. Add qfrc_constraint back immediately and use the corrected
        # qfrc_inverse for all downstream dynamics.
        qfrc_inverse_raw = data.qfrc_inverse
        qfrc_constraint = data.qfrc_constraint
        qfrc_inverse = qfrc_inverse_raw + qfrc_constraint
        full_id = qfrc_inverse - tau_grf

        return {
            "qfrc_inverse": qfrc_inverse,
            "qfrc_constraint": qfrc_constraint,
            "jacp": jacp,
            "jacr": jacr,
            "rot_w_to_ga": rot_w_to_ga,
            "tau_grf": tau_grf,
            "full_id": full_id,
            "cop_world": cop_world,
            "full_moments": full_moments,
            "ankle_heights": ankle_heights,
        }

    return jax.jit(jax.vmap(_single_frame))


def _single_frame_generic_mjx_physics(
    mjx_model: Any,
    calcn_r_id: jax.Array,
    calcn_l_id: jax.Array,
    qpos_t: jax.Array,
    qvel_t: jax.Array,
    qacc_t: jax.Array,
    cop_t: jax.Array,
    grf_t: jax.Array,
    grm_t: jax.Array,
    ankle_t: jax.Array,
) -> Dict[str, jax.Array]:
    data = mjx.make_data(mjx_model).replace(qpos=qpos_t, qvel=qvel_t, qacc=qacc_t)
    data = mjx.inverse(mjx_model, data)

    body_rot_r = data.xmat[calcn_r_id].T
    body_rot_l = data.xmat[calcn_l_id].T
    rot_w_to_ga_r = _compose_world_to_ground_aligned_single_jax(body_rot_r)
    rot_w_to_ga_l = _compose_world_to_ground_aligned_single_jax(body_rot_l)
    rot_w_to_ga = jnp.stack([rot_w_to_ga_r, rot_w_to_ga_l], axis=0)

    jacp_r_nv3, jacr_r_nv3 = mjx.jac(mjx_model, data, data.xpos[calcn_r_id], calcn_r_id)
    jacp_l_nv3, jacr_l_nv3 = mjx.jac(mjx_model, data, data.xpos[calcn_l_id], calcn_l_id)
    jacp_r = jacp_r_nv3.T
    jacr_r = jacr_r_nv3.T
    jacp_l = jacp_l_nv3.T
    jacr_l = jacr_l_nv3.T
    jacp = jnp.stack([jacp_r, jacp_l], axis=0)
    jacr = jnp.stack([jacr_r, jacr_l], axis=0)

    ankle_heights = jnp.asarray(ankle_t, dtype=qpos_t.dtype)
    cop_r_ga = jnp.array([cop_t[0], ankle_heights[0], cop_t[1]], dtype=qpos_t.dtype)
    cop_l_ga = jnp.array([cop_t[2], ankle_heights[1], cop_t[3]], dtype=qpos_t.dtype)
    cop_r_world = rot_w_to_ga_r.T @ cop_r_ga
    cop_l_world = rot_w_to_ga_l.T @ cop_l_ga
    cop_world = jnp.concatenate([cop_r_world, cop_l_world], axis=0)

    grf_r = grf_t[:3]
    grf_l = grf_t[3:6]
    grm_r = jnp.array([0.0, 0.0, grm_t[0]], dtype=qpos_t.dtype)
    grm_l = jnp.array([0.0, 0.0, grm_t[1]], dtype=qpos_t.dtype)
    moments_r = jnp.cross(cop_r_world, grf_r) + grm_r
    moments_l = jnp.cross(cop_l_world, grf_l) + grm_l
    full_moments = jnp.concatenate([moments_r, moments_l], axis=0)

    tau_grf = (
        jacp_r_nv3 @ grf_r
        + jacr_r_nv3 @ moments_r
        + jacp_l_nv3 @ grf_l
        + jacr_l_nv3 @ moments_l
    )
    qfrc_inverse_raw = data.qfrc_inverse
    qfrc_constraint = data.qfrc_constraint
    qfrc_inverse = qfrc_inverse_raw + qfrc_constraint
    full_id = qfrc_inverse - tau_grf

    return {
        "qfrc_inverse": qfrc_inverse,
        "qfrc_constraint": qfrc_constraint,
        "jacp": jacp,
        "jacr": jacr,
        "rot_w_to_ga": rot_w_to_ga,
        "tau_grf": tau_grf,
        "full_id": full_id,
        "cop_world": cop_world,
        "full_moments": full_moments,
        "ankle_heights": ankle_heights,
    }


_GENERIC_MJX_PHYSICS_RUNNER = jax.jit(
    jax.vmap(
        _single_frame_generic_mjx_physics,
        in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0),
    )
)


@dataclass
class DerivedPhysicsResult:
    qfrc_inverse: np.ndarray
    qfrc_constraint: np.ndarray
    jacp: np.ndarray
    jacr: np.ndarray
    rot_w_to_ga: np.ndarray
    tau_grf: np.ndarray
    full_id: np.ndarray
    cop_world: np.ndarray
    detached: bool = False


class Normalizer:
    """Simple mean/std normalizer shared by train_mod_q and infer_mod_q."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < eps, eps, self.std)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean


def compute_derived_physics_from_qprime(
    xml_path: Path | str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    cop_pred: np.ndarray,
    grf_pred: np.ndarray,
    grm_pred: np.ndarray,
    ankle_heights: np.ndarray,
    *,
    detached_fallback: bool = True,
) -> DerivedPhysicsResult:
    """Run the shared MJX/JAX physics kernel and return host arrays."""
    if mujoco is None or mjx is None:
        if detached_fallback:
            raise ImportError(
                "mujoco/mjx is unavailable; derived physics requires MJX."
            )
        raise ImportError("mujoco/mjx is unavailable.")

    adapter = _get_global_modq_physics_adapter()
    physics = adapter.evaluate(
        qpos=qpos,
        qvel=qvel,
        qacc=qacc,
        cop_phys=cop_pred,
        grf_phys=grf_pred,
        moments_phys=grm_pred,
        ankle_heights=ankle_heights,
        xml_path=str(xml_path),
    )
    if physics is None:
        raise RuntimeError(f"Failed to evaluate mod_q physics for model: {xml_path}")
    return DerivedPhysicsResult(
        qfrc_inverse=np.asarray(physics["qfrc_inverse"], dtype=np.float32),
        qfrc_constraint=np.asarray(physics["qfrc_constraint"], dtype=np.float32),
        jacp=np.asarray(physics["jacp"], dtype=np.float32),
        jacr=np.asarray(physics["jacr"], dtype=np.float32),
        rot_w_to_ga=np.asarray(physics["rot_w_to_ga"], dtype=np.float32),
        tau_grf=np.asarray(physics["tau_grf"], dtype=np.float32),
        full_id=np.asarray(physics["full_id"], dtype=np.float32),
        cop_world=np.asarray(physics["cop_world"], dtype=np.float32),
        detached=False,
    )


class ModQPhysicsAdapter:
    """Model-cached differentiable MJX physics adapter."""

    def __init__(self):
        self._subject_context_cache: Dict[str, Dict[str, Any]] = {}
        self._structure_runner_cache: Dict[str, Any] = {}
        self._structure_reconstructor_cache: Dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(mujoco is not None and mjx is not None)

    def _cache_subject_context(self, cache_key: str, context: Dict[str, Any]) -> None:
        self._subject_context_cache[cache_key] = context
        runtime_xml = context.get("runtime_xml_path")
        if runtime_xml:
            self._subject_context_cache[str(runtime_xml)] = context

    def get_subject_context(self, xml_path: str) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("mujoco/mjx is unavailable; cannot build mod_q physics contexts.")

        xml_path = str(xml_path)
        cached = self._subject_context_cache.get(xml_path)
        if cached is not None:
            return cached

        runtime_info = ensure_modq_runtime_xml(xml_path)
        runtime_xml = str(runtime_info.runtime_xml)
        cached = self._subject_context_cache.get(runtime_xml)
        if cached is not None:
            self._subject_context_cache[xml_path] = cached
            return cached

        model = mujoco.MjModel.from_xml_path(runtime_xml)
        calcn_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
        calcn_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_l")
        if min(calcn_r_id, calcn_l_id) < 0:
            raise ValueError(f"Could not resolve calcaneus bodies in XML: {runtime_xml}")

        coupling_spec = _parse_coupling_spec(runtime_xml, model)
        context = {
            "source_xml_path": str(runtime_info.source_xml),
            "runtime_xml_path": runtime_xml,
            "structure_key": str(runtime_info.structure_key),
            "core_shape": tuple(runtime_info.core_shape),
            "runtime_shape": tuple(runtime_info.runtime_shape),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "mjx_model": mjx.put_model(model),
            "calcn_r_id": jnp.asarray(calcn_r_id, dtype=jnp.int32),
            "calcn_l_id": jnp.asarray(calcn_l_id, dtype=jnp.int32),
            "slave_idx": jnp.asarray(coupling_spec["slave_idx"], dtype=jnp.int32),
            "master_idx": jnp.asarray(coupling_spec["master_idx"], dtype=jnp.int32),
            "coeffs": jnp.asarray(coupling_spec["coeffs"], dtype=jnp.float32),
        }
        self._cache_subject_context(xml_path, context)
        return context

    def get_jit_context(self, xml_path: str) -> Dict[str, Any]:
        context = self.get_subject_context(xml_path)
        return {
            "mjx_model": context["mjx_model"],
            "calcn_r_id": context["calcn_r_id"],
            "calcn_l_id": context["calcn_l_id"],
            "slave_idx": context["slave_idx"],
            "master_idx": context["master_idx"],
            "coeffs": context["coeffs"],
        }

    def get_structure_key(self, xml_path: str) -> str:
        return str(self.get_subject_context(xml_path)["structure_key"])

    def get_runtime_xml_path(self, xml_path: str) -> str:
        return str(self.get_subject_context(xml_path)["runtime_xml_path"])

    def get_runner(self, xml_path: str):
        if not self.available:
            return None
        structure_key = self.get_structure_key(xml_path)
        runner = self._structure_runner_cache.get(structure_key)
        if runner is None:
            runner = _GENERIC_MJX_PHYSICS_RUNNER
            self._structure_runner_cache[structure_key] = runner
        return runner

    def get_reconstructor(self, xml_path: str):
        if not self.available:
            return None
        structure_key = self.get_structure_key(xml_path)
        reconstructor = self._structure_reconstructor_cache.get(structure_key)
        if reconstructor is None:
            reconstructor = _GENERIC_KINEMATICS_RECONSTRUCTOR
            self._structure_reconstructor_cache[structure_key] = reconstructor
        return reconstructor

    def get_state_dims(self, xml_path: str) -> Tuple[int, int]:
        if not self.available:
            raise RuntimeError("mujoco/mjx is unavailable; cannot resolve subject state dims.")
        context = self.get_subject_context(xml_path)
        return int(context["nq"]), int(context["nv"])

    def reconstruct_state_jax(
        self,
        *,
        pos_pred: np.ndarray | jax.Array,
        vel_pred: np.ndarray | jax.Array,
        acc_pred: np.ndarray | jax.Array,
        qpos_template: np.ndarray | jax.Array,
        qvel_template: np.ndarray | jax.Array,
        qacc_template: np.ndarray | jax.Array,
        xml_path: Optional[str] = None,
        physics_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        context = physics_context if physics_context is not None else self.get_jit_context(str(xml_path))
        return _GENERIC_KINEMATICS_RECONSTRUCTOR(
            jnp.asarray(pos_pred, dtype=jnp.float32),
            jnp.asarray(vel_pred, dtype=jnp.float32),
            jnp.asarray(acc_pred, dtype=jnp.float32),
            jnp.asarray(qpos_template, dtype=jnp.float32),
            jnp.asarray(qvel_template, dtype=jnp.float32),
            jnp.asarray(qacc_template, dtype=jnp.float32),
            jnp.asarray(context["slave_idx"], dtype=jnp.int32),
            jnp.asarray(context["master_idx"], dtype=jnp.int32),
            jnp.asarray(context["coeffs"], dtype=jnp.float32),
        )

    def evaluate_jax(
        self,
        qpos: np.ndarray | jax.Array,
        qvel: np.ndarray | jax.Array,
        qacc: np.ndarray | jax.Array,
        cop_phys: np.ndarray | jax.Array,
        grf_phys: np.ndarray | jax.Array,
        moments_phys: np.ndarray | jax.Array,
        ankle_heights: np.ndarray | jax.Array,
        xml_path: Optional[str] = None,
        physics_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, jax.Array]]:
        context = physics_context if physics_context is not None else self.get_jit_context(str(xml_path))
        runner = self.get_runner(str(xml_path)) if xml_path is not None else _GENERIC_MJX_PHYSICS_RUNNER
        if runner is None:
            return None
        return runner(
            context["mjx_model"],
            context["calcn_r_id"],
            context["calcn_l_id"],
            jnp.asarray(qpos, dtype=jnp.float32),
            jnp.asarray(qvel, dtype=jnp.float32),
            jnp.asarray(qacc, dtype=jnp.float32),
            jnp.asarray(cop_phys, dtype=jnp.float32),
            jnp.asarray(grf_phys, dtype=jnp.float32),
            jnp.asarray(moments_phys, dtype=jnp.float32),
            jnp.asarray(ankle_heights, dtype=jnp.float32),
        )

    def evaluate(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        cop_phys: np.ndarray,
        grf_phys: np.ndarray,
        moments_phys: np.ndarray,
        ankle_heights: np.ndarray,
        xml_path: Optional[str] = None,
        physics_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        try:
            result = self.evaluate_jax(
                qpos=qpos,
                qvel=qvel,
                qacc=qacc,
                cop_phys=cop_phys,
                grf_phys=grf_phys,
                moments_phys=moments_phys,
                ankle_heights=ankle_heights,
                xml_path=xml_path,
                physics_context=physics_context,
            )
        except Exception:
            return None
        if result is None:
            return None
        return jax.device_get(result)


_GLOBAL_MODQ_PHYSICS_ADAPTER: Optional[ModQPhysicsAdapter] = None


def _get_global_modq_physics_adapter() -> ModQPhysicsAdapter:
    global _GLOBAL_MODQ_PHYSICS_ADAPTER
    if _GLOBAL_MODQ_PHYSICS_ADAPTER is None:
        _GLOBAL_MODQ_PHYSICS_ADAPTER = ModQPhysicsAdapter()
    return _GLOBAL_MODQ_PHYSICS_ADAPTER


# =============================================================================
# Optional training convenience helpers
# =============================================================================

def build_mod_q_loss_weights(
    *,
    cop_weight: float = 1.0,
    grf_weight: float = 1.0,
    moments_weight: float = 0.25,
    contact_weight: float = 1.0,
    torque_weight: float = 2.0,
    grf_correction_weight: float = 0.0,
    output_reg_weight: float = 0.0,
    jacobian_weight: float = 1.0,
    qpos_weight: float = 1.0,
    qvel_weight: float = 1.0,
    qacc_weight: float = 1.0,
    rotation_weight: float = 1.0,
    qfrc_inverse_weight: float = 1.0,
) -> Dict[str, float]:
    return {
        "cop": float(cop_weight),
        "grf": float(grf_weight),
        "moments": float(moments_weight),
        "contact": float(contact_weight),
        "torque": float(torque_weight),
        "grf_correction": float(grf_correction_weight),
        "output_reg": float(output_reg_weight),
        "jacobian": float(jacobian_weight),
        "pos": float(qpos_weight),
        "vel": float(qvel_weight),
        "acc": float(qacc_weight),
        "rotation": float(rotation_weight),
        "qfrc_inverse": float(qfrc_inverse_weight),
    }


def build_modq_input_features(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    return build_mod_q_temporal_input(data)


def build_modq_static_context(data: Dict[str, np.ndarray]) -> np.ndarray:
    return build_mod_q_static_context(data)


def build_mod_q_input_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    temporal, _ = build_mod_q_temporal_input(data)
    return temporal


def reconstruct_mod_q_state(
    *,
    pos_pred: np.ndarray,
    vel_pred: np.ndarray,
    acc_pred: np.ndarray,
    qpos_template: np.ndarray,
    qvel_template: np.ndarray,
    qacc_template: np.ndarray,
    xml_path: Path | str,
) -> Dict[str, np.ndarray]:
    adapter = _get_global_modq_physics_adapter()
    qpos, qvel, qacc = adapter.reconstruct_state_jax(
        pos_pred=pos_pred,
        vel_pred=vel_pred,
        acc_pred=acc_pred,
        qpos_template=qpos_template,
        qvel_template=qvel_template,
        qacc_template=qacc_template,
        xml_path=str(xml_path),
    )
    return {
        "qpos": np.asarray(jax.device_get(qpos), dtype=np.float32),
        "qvel": np.asarray(jax.device_get(qvel), dtype=np.float32),
        "qacc": np.asarray(jax.device_get(qacc), dtype=np.float32),
    }


def build_mod_q_model(input_dim: int, static_dim: int, model_cfg: Dict[str, Any]) -> ModQTransformer:
    kernels = model_cfg.get("cnn_kernel_sizes", (3, 5))
    if isinstance(kernels, str):
        kernels = tuple(int(k.strip()) for k in kernels.split(",") if k.strip())
    else:
        kernels = tuple(int(k) for k in kernels)
    return ModQTransformer(
        input_dim=int(input_dim),
        static_dim=int(static_dim),
        output_dim=int(model_cfg.get("output_dim", MOD_Q_OUTPUT_DIM)),
        d_model=int(model_cfg.get("d_model", 256)),
        num_heads=int(model_cfg.get("num_heads", MOD_Q_DEFAULT_NUM_HEADS)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        ff_dim=int(model_cfg.get("ff_dim", 1024)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.1)),
        use_cnn=bool(model_cfg.get("use_cnn", True)),
        cnn_num_layers=int(model_cfg.get("cnn_num_layers", 2)),
        cnn_kernel_sizes=kernels,
        use_multitask=bool(model_cfg.get("use_multitask", False)),
    )


def decode_mod_q_predictions(output_np: np.ndarray) -> Dict[str, np.ndarray]:
    pred = np.asarray(output_np, dtype=np.float32)
    grm = pred[..., 10:12]
    return {
        "cop": pred[..., 0:4],
        "grf": pred[..., 4:10],
        "grm": grm,
        "moments": grm,
        "contact": 1.0 / (1.0 + np.exp(-pred[..., 12:14])),
        "pos": pred[..., 14:30],
        "vel": pred[..., 30:49],
        "acc": pred[..., 49:68],
    }


def split_modq_predictions(pred: Any) -> Dict[str, Any]:
    return split_mod_q_predictions(pred)


def unnormalize_modq_predictions(
    pred: Any,
    normalizers: Dict[str, Normalizer],
    *,
    pred_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = dict(split_mod_q_predictions(pred) if pred_dict is None else pred_dict)
    if "cop" in normalizers:
        out["cop"] = normalizers["cop"].unnormalize(out["cop"])
    if "grf" in normalizers:
        out["grf"] = normalizers["grf"].unnormalize(out["grf"])
    if "moments" in normalizers:
        out["moments"] = normalizers["moments"].unnormalize(out["moments"])
    if "pos" in normalizers:
        out["pos"] = normalizers["pos"].unnormalize(out["pos"])
    if "vel" in normalizers:
        out["vel"] = normalizers["vel"].unnormalize(out["vel"])
    if "acc" in normalizers:
        out["acc"] = normalizers["acc"].unnormalize(out["acc"])
    return out


def mod_q_physics_adapter(
    qprime: Dict[str, np.ndarray],
    cop_phys: np.ndarray,
    grf_phys: np.ndarray,
    moments_phys: np.ndarray,
    ankle_heights: np.ndarray,
    rot_w_to_ga_gt: Optional[np.ndarray] = None,
    xml_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    if xml_path is None:
        raise ValueError("xml_path is required for mod_q physics inference")
    result = compute_derived_physics_from_qprime(
        xml_path=xml_path,
        qpos=qprime["qpos"],
        qvel=qprime["qvel"],
        qacc=qprime["qacc"],
        cop_pred=cop_phys,
        grf_pred=grf_phys,
        grm_pred=moments_phys,
        ankle_heights=ankle_heights,
        detached_fallback=True,
    )
    return {
        "rot_w_to_ga": result.rot_w_to_ga,
        "jacp": result.jacp,
        "jacr": result.jacr,
        "qfrc_constraint": result.qfrc_constraint,
        "qfrc_inverse": result.qfrc_inverse,
        "tau_grf": result.tau_grf,
        "full_id": result.full_id,
        "cop_world": result.cop_world,
    }
