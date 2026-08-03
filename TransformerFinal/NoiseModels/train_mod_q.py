"""Training entrypoint for the mod_q pipeline.

This version keeps the transformer backbone style but changes the output schema
to predict COP, GRF, GRM, contact, and the full MJX q_prime state
([qpos, qvel, qacc]).

The script is intentionally self-contained so it can run before the shared
`mod_q_shared.py` helper lands, while still importing it if it exists later.
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
import pickle
import queue
import random
import re
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wandb_utils import WandbLogger, configure_runtime_env

RUNTIME_ENV_APPLIED = configure_runtime_env()

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

from data_loader import (
    build_window_start_indices,
    build_window_supervision_mask,
    select_pos_input_columns,
    validate_prediction_margin,
)
from runtime_model_utils import RUNTIME_XML_NAME, modq_runtime_structure_key, resolve_modq_runtime_xml
from paths import artifact, dataset  # noqa: E402

try:  # Optional shared helper, if it lands later.
    from mod_q_shared import (  # type: ignore
        MODQ_FORCED_FLAGS,
        MODQ_OUTPUT_DIM,
        MODQ_INPUT_FEATURE_BLOCKS,
        MODQ_OUTPUT_SCHEMA,
        MODQ_QPRIME_LAYOUT,
        ModQPhysicsAdapter,
        ModQTransformer,
        geodesic_rotation_mse,
        build_modq_input_features,
        build_modq_static_context,
        project_rotation_matrices,
        split_modq_predictions,
        rotation_geodesic_summary_deg,
        unnormalize_modq_predictions,
    )
    SHARED_MODQ_AVAILABLE = True
    SHARED_MODQ_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - local fallback until shared helper exists.
    SHARED_MODQ_AVAILABLE = False
    SHARED_MODQ_IMPORT_ERROR = exc
    MODQ_OUTPUT_DIM = 68
    MODQ_OUTPUT_SCHEMA = {
        "cop": (0, 4),
        "grf": (4, 10),
        "moments": (10, 12),
        "contact": (12, 14),
        "pos": (14, 30),
        "vel": (30, 49),
        "acc": (49, 68),
    }
    MODQ_QPRIME_LAYOUT = {
        "pos_dim": 16,
        "vel_dim": 19,
        "acc_dim": 19,
        "total_dim": 54,
    }
    MODQ_INPUT_FEATURE_BLOCKS = [
        {"name": "pelvis_rot", "dim": 6},
        {"name": "pos", "dim": 16},
        {"name": "vel", "dim": 19},
        {"name": "acc", "dim": 19},
        {"name": "com_r", "dim": 3},
        {"name": "com_l", "dim": 3},
        {"name": "com_accel", "dim": 3},
        {"name": "foot_progression_angle", "dim": 2},
        {"name": "calcn_to_floor_angle", "dim": 2},
    ]
    MODQ_FORCED_FLAGS = {
        "UseNoised": True,
        "includePelvisEuler": True,
        "PredictJacobian": False,
        "DeviationLearning": True,
        "subject_grouped_batches": True,
    }

    def build_modq_input_features(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        parts = [
            np.asarray(data["pelvis_rot"], dtype=np.float32),
            np.asarray(data["pos"], dtype=np.float32),
            np.asarray(data["vel"], dtype=np.float32),
            np.asarray(data["acc"], dtype=np.float32),
            np.asarray(data["com_r"], dtype=np.float32),
            np.asarray(data["com_l"], dtype=np.float32),
            np.asarray(data["com_accel"], dtype=np.float32),
            np.asarray(data["foot_progression_angle"], dtype=np.float32),
            np.asarray(data["calcn_to_floor_angle"], dtype=np.float32),
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

    def build_modq_static_context(data: Dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(
            [
                data["height"][0, 0],
                data["mass"][0, 0],
                data["gender"],
                data["patient_size"][0],
                data["patient_size"][1],
                data["patient_size"][2],
                data["patient_size"][3],
                data["forward_vel"],
            ],
            dtype=np.float32,
        )

    def split_modq_predictions(pred: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        out = {}
        for key, (start, end) in MODQ_OUTPUT_SCHEMA.items():
            out[key] = pred[..., start:end]
        return out

    def unnormalize_modq_predictions(
        pred: jnp.ndarray,
        normalizers: Dict[str, "Normalizer"],
        *,
        pred_dict: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, jnp.ndarray]:
        if pred_dict is None:
            pred_dict = split_modq_predictions(pred)
        out = dict(pred_dict)
        out["cop"] = normalizers["cop"].unnormalize(out["cop"])
        out["grf"] = normalizers["grf"].unnormalize(out["grf"])
        out["moments"] = normalizers["moments"].unnormalize(out["moments"])
        out["pos"] = normalizers["pos"].unnormalize(out["pos"])
        return out

    def _normalize_world_vec(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < eps:
            return np.zeros_like(vec)
        return vec / norm

    def _compose_world_to_ground_aligned_single_np(rot_w_to_body: np.ndarray) -> np.ndarray:
        """Match ProcessData.py ground-aligned calcaneus rotation construction."""
        rot_w_to_body = np.asarray(rot_w_to_body, dtype=np.float64)
        n_w = _normalize_world_vec(np.array([0.0, 0.0, 1.0], dtype=np.float64))

        rot_body_to_world = rot_w_to_body.T
        x_w = rot_body_to_world[:, 0]
        xg_w = x_w - np.dot(x_w, n_w) * n_w
        if np.linalg.norm(xg_w) < 1e-10:
            z_w = rot_body_to_world[:, 2]
            xg_w = np.cross(n_w, z_w)
        xg_w = _normalize_world_vec(xg_w)

        yg_w = n_w.copy()
        zg_w = _normalize_world_vec(np.cross(xg_w, yg_w))
        xg_w = _normalize_world_vec(np.cross(yg_w, zg_w))

        rot_ground_aligned_to_world = np.column_stack([xg_w, yg_w, zg_w])
        rot_ground_aligned_to_body = rot_ground_aligned_to_world.T @ rot_body_to_world
        return rot_ground_aligned_to_body @ rot_w_to_body

    def _vector_norm(x: Any, *, xp=jnp, axis: int = -1, keepdims: bool = False) -> Any:
        return xp.sqrt(xp.sum(xp.square(x), axis=axis, keepdims=keepdims))

    def _safe_vector_norm(x: Any, *, xp=jnp, axis: int = -1, keepdims: bool = False, eps: float = 1e-12) -> Any:
        return xp.maximum(_vector_norm(x, xp=xp, axis=axis, keepdims=keepdims), xp.asarray(eps, dtype=x.dtype))

    def _normalize_vector(vec: Any, *, xp=jnp, eps: float = 1e-6) -> Any:
        norm = _safe_vector_norm(vec, xp=xp, axis=-1, keepdims=True, eps=eps)
        return vec / norm

    def _skew_symmetric(vec: Any, *, xp=jnp) -> Any:
        vec = xp.asarray(vec)
        zero = xp.zeros_like(vec[..., 0])
        x = vec[..., 0]
        y = vec[..., 1]
        z = vec[..., 2]
        row1 = xp.stack([zero, -z, y], axis=-1)
        row2 = xp.stack([z, zero, -x], axis=-1)
        row3 = xp.stack([-y, x, zero], axis=-1)
        return xp.stack([row1, row2, row3], axis=-2)

    def _broadcast_rotation_identity(leading_shape: Sequence[int], *, dtype: Any, xp=jnp) -> Any:
        identity = xp.asarray(np.eye(3, dtype=dtype))
        return xp.broadcast_to(identity, tuple(leading_shape) + (3, 3))

    def project_rotation_matrices(rot: Any, xp=jnp) -> Any:
        rot = xp.asarray(rot)
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

        x_axis = xp.broadcast_to(xp.asarray([1.0, 0.0, 0.0], dtype=rot_flat.dtype), basis1.shape)
        y_axis = xp.broadcast_to(xp.asarray([0.0, 1.0, 0.0], dtype=rot_flat.dtype), basis1.shape)
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

    def rotation_geodesic_angle(rotation_a: Any, rotation_b: Any, *, xp=jnp) -> Any:
        rotation_a = xp.asarray(rotation_a)
        rotation_b = xp.asarray(rotation_b)
        rotation_err = xp.matmul(rotation_a, xp.swapaxes(rotation_b, -1, -2))
        trace = rotation_err[..., 0, 0] + rotation_err[..., 1, 1] + rotation_err[..., 2, 2]
        cos_theta = xp.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        skew_vec = xp.stack(
            [
                rotation_err[..., 2, 1] - rotation_err[..., 1, 2],
                rotation_err[..., 0, 2] - rotation_err[..., 2, 0],
                rotation_err[..., 1, 0] - rotation_err[..., 0, 1],
            ],
            axis=-1,
        )
        sin_theta = 0.5 * _vector_norm(skew_vec, xp=xp, axis=-1)
        return xp.arctan2(sin_theta, cos_theta)

    def masked_mean(values: Any, mask: Optional[Any], *, xp=jnp) -> Any:
        values = xp.asarray(values)
        if mask is None:
            return xp.mean(values)
        weights = xp.asarray(mask, dtype=values.dtype)
        while weights.ndim > values.ndim and weights.shape[-1] == 1:
            weights = weights[..., 0]
        while weights.ndim < values.ndim:
            weights = weights[..., None]
        weights = xp.broadcast_to(weights, values.shape)
        denom = xp.maximum(xp.sum(weights), xp.asarray(1.0, dtype=values.dtype))
        return xp.sum(values * weights) / denom

    def geodesic_rotation_mse(rotation_a: Any, rotation_b: Any, supervision_mask: Optional[Any], *, xp=jnp) -> Any:
        angle = rotation_geodesic_angle(rotation_a, rotation_b, xp=xp)
        return masked_mean(xp.square(angle), supervision_mask, xp=xp)

    def rotation_geodesic_summary_deg(
        rotation_a: Any,
        rotation_b: Any,
        supervision_mask: Optional[Any],
        *,
        xp=jnp,
    ) -> Dict[str, Any]:
        angle_rad = rotation_geodesic_angle(rotation_a, rotation_b, xp=xp)
        angle_deg = angle_rad * xp.asarray(180.0 / np.pi, dtype=angle_rad.dtype)
        overall_mean_deg = masked_mean(angle_deg, supervision_mask, xp=xp)
        overall_rmse_deg = xp.sqrt(masked_mean(xp.square(angle_deg), supervision_mask, xp=xp))
        right_mean_deg = masked_mean(angle_deg[..., 0], supervision_mask, xp=xp)
        left_mean_deg = masked_mean(angle_deg[..., 1], supervision_mask, xp=xp)
        return {
            "overall_mean_deg": overall_mean_deg,
            "overall_rmse_deg": overall_rmse_deg,
            "right_mean_deg": right_mean_deg,
            "left_mean_deg": left_mean_deg,
            "mean_deg": overall_mean_deg,
            "rmse_deg": overall_rmse_deg,
        }

    class ModQPhysicsAdapter:
        """Host-side physics adapter with an explicit detached fallback."""

        def __init__(self):
            self._mujoco = None
            self._model_cache: Dict[str, Any] = {}
            self._body_cache: Dict[str, Tuple[int, int, int, int]] = {}
            try:
                import mujoco  # type: ignore

                self._mujoco = mujoco
            except Exception:
                self._mujoco = None

        @property
        def available(self) -> bool:
            return self._mujoco is not None

        def get_runner(self, xml_path: str):
            return None

        def _get_model(self, xml_path: str):
            if xml_path in self._model_cache:
                return self._model_cache[xml_path]
            if self._mujoco is None:
                return None
            model = self._mujoco.MjModel.from_xml_path(str(xml_path))
            self._model_cache[xml_path] = model
            self._body_cache[xml_path] = (
                self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_BODY, "calcn_r"),
                self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_BODY, "calcn_l"),
                self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_BODY, "toes_r"),
                self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_BODY, "toes_l"),
            )
            return model

    def evaluate(
            self,
            qpos: np.ndarray,
            qvel: np.ndarray,
            qacc: np.ndarray,
            cop_phys: np.ndarray,
            grf_phys: np.ndarray,
            moments_phys: np.ndarray,
            ankle_heights: np.ndarray,
            xml_path: str,
        ) -> Optional[Dict[str, np.ndarray]]:
            if not self.available:
                return None

            mujoco = self._mujoco
            assert mujoco is not None
            model = self._get_model(xml_path)
            if model is None:
                return None

            calcn_r_id, calcn_l_id, toes_r_id, toes_l_id = self._body_cache[xml_path]
            if min(calcn_r_id, calcn_l_id, toes_r_id, toes_l_id) < 0:
                return None

            qpos = np.asarray(qpos, dtype=np.float64)
            qvel = np.asarray(qvel, dtype=np.float64)
            qacc = np.asarray(qacc, dtype=np.float64)
            cop_phys = np.asarray(cop_phys, dtype=np.float64)
            grf_phys = np.asarray(grf_phys, dtype=np.float64)
            moments_phys = np.asarray(moments_phys, dtype=np.float64)
            ankle_heights = np.asarray(ankle_heights, dtype=np.float64)

            t_len = int(qpos.shape[0])
            nv = int(model.nv)
            qfrc_constraint = np.zeros((t_len, nv), dtype=np.float32)
            qfrc_inverse = np.zeros((t_len, nv), dtype=np.float32)
            jacp = np.zeros((t_len, 2, 3, nv), dtype=np.float32)
            jacr = np.zeros((t_len, 2, 3, nv), dtype=np.float32)
            rot_w_to_ga = np.zeros((t_len, 2, 3, 3), dtype=np.float32)
            tau_grf = np.zeros((t_len, nv), dtype=np.float32)

            data = mujoco.MjData(model)
            for t in range(t_len):
                data.qpos[:] = qpos[t]
                data.qvel[:] = qvel[t]
                data.qacc[:] = qacc[t]
                mujoco.mj_forward(model, data)
                mujoco.mj_inverse(model, data)
                qfrc_inverse_raw = np.asarray(data.qfrc_inverse, dtype=np.float32)
                qfrc_constraint[t] = np.asarray(data.qfrc_constraint, dtype=np.float32)
                qfrc_inverse[t] = qfrc_inverse_raw + qfrc_constraint[t]

                for foot_idx, body_id in enumerate((calcn_r_id, calcn_l_id)):
                    jp = np.zeros((3, nv), dtype=np.float64)
                    jr = np.zeros((3, nv), dtype=np.float64)
                    mujoco.mj_jacBody(model, data, jp, jr, body_id)
                    jacp[t, foot_idx] = jp
                    jacr[t, foot_idx] = jr
                    body_rot_w_to_b = data.xmat[body_id].reshape(3, 3).T
                    rot_w_to_ga[t, foot_idx] = _compose_world_to_ground_aligned_single_np(body_rot_w_to_b)

                cop_r_ga = np.array([cop_phys[t, 0], ankle_heights[t, 0], cop_phys[t, 1]], dtype=np.float64)
                cop_l_ga = np.array([cop_phys[t, 2], ankle_heights[t, 1], cop_phys[t, 3]], dtype=np.float64)
                rot_ga_to_w_r = rot_w_to_ga[t, 0].T
                rot_ga_to_w_l = rot_w_to_ga[t, 1].T
                cop_r_w = rot_ga_to_w_r @ cop_r_ga
                cop_l_w = rot_ga_to_w_l @ cop_l_ga
                grf_r = grf_phys[t, :3]
                grf_l = grf_phys[t, 3:6]
                mom_r = np.array([0.0, 0.0, moments_phys[t, 0]], dtype=np.float64)
                mom_l = np.array([0.0, 0.0, moments_phys[t, 1]], dtype=np.float64)
                m_r_total = mom_r + np.cross(cop_r_w, grf_r)
                m_l_total = mom_l + np.cross(cop_l_w, grf_l)
                tau_grf[t] = (
                    jacp[t, 0].T @ grf_r
                    + jacr[t, 0].T @ m_r_total
                    + jacp[t, 1].T @ grf_l
                    + jacr[t, 1].T @ m_l_total
                ).astype(np.float32)

            full_id = qfrc_inverse - tau_grf
            return {
                "qfrc_inverse": qfrc_inverse,
                "qfrc_constraint": qfrc_constraint,
                "jacp": jacp,
                "jacr": jacr,
                "rot_w_to_ga": rot_w_to_ga,
                "tau_grf": tau_grf,
                "full_id": full_id,
            }


if not SHARED_MODQ_AVAILABLE:
    class SinusoidalPosEmb(nn.Module):
        dim: int

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            seq_len = x.shape[1]
            position = jnp.arange(seq_len)
            half_dim = self.dim // 2
            emb_scale = jnp.log(10000.0) / jnp.maximum(half_dim - 1, 1)
            emb = jnp.exp(jnp.arange(half_dim) * -emb_scale)
            emb = position[:, None] * emb[None, :]
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
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                dropout_rate=self.dropout_rate,
            )(x, x, deterministic=not train)
            x = residual + attn_out

            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.ff_dim)(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = nn.Dense(self.d_model)(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            return residual + x


    class ModQTransformer(nn.Module):
        input_dim: int
        static_dim: int
        output_dim: int
        d_model: int = 256
        num_heads: int = 4
        num_layers: int = 4
        ff_dim: int = 1024
        dropout_rate: float = 0.1
        use_cnn: bool = True
        cnn_num_layers: int = 2
        cnn_kernel_sizes: Tuple[int, ...] = (3, 5)

        @nn.compact
        def __call__(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
            x = nn.Dense(self.d_model)(x)
            if self.use_cnn:
                residual = nn.gelu(x)
                kernels = list(self.cnn_kernel_sizes) if self.cnn_kernel_sizes else [3]
                for i in range(self.cnn_num_layers):
                    k = kernels[i] if i < len(kernels) else kernels[-1]
                    x = nn.Conv(
                        features=self.d_model,
                        kernel_size=(k,),
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
            static = nn.Dense(self.d_model)(static_context)
            static = nn.gelu(static)
            static = nn.LayerNorm()(static)
            x = jnp.concatenate([static[:, None, :], x], axis=1)

            for _ in range(self.num_layers):
                x = TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    ff_dim=self.ff_dim,
                    dropout_rate=self.dropout_rate,
                )(x, train=train)

            x = nn.LayerNorm()(x[:, 1:, :])
            out = nn.Dense(self.output_dim)(x)
            out = out.at[..., MODQ_OUTPUT_SCHEMA["contact"][0]:MODQ_OUTPUT_SCHEMA["contact"][1]].set(
                nn.sigmoid(out[..., MODQ_OUTPUT_SCHEMA["contact"][0]:MODQ_OUTPUT_SCHEMA["contact"][1]])
            )
            return out


@dataclass(frozen=True)
class TrialRecord:
    subject: str
    trial_name: str
    training_data_path: str
    length: int
    subject_model_xml: str
    subject_structure_key: str = ""


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < eps, eps, self.std)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".16g")
    return json.dumps(str(value))


def save_model_parameters_yaml(params: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {_yaml_scalar(item)}\n")
            else:
                f.write(f"{key}: {_yaml_scalar(value)}\n")


def _parse_optional_bool_arg(value):
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _with_suffix(filename: str, suffix: str = "_noised") -> str:
    path = Path(filename)
    if path.suffix:
        return f"{path.stem}{suffix}{path.suffix}"
    return f"{filename}{suffix}"


def _load_npy_optional(base_dir: Path, names: Sequence[str], *, allow_pickle: bool = False) -> Optional[np.ndarray]:
    for name in names:
        path = base_dir / name
        if path.exists():
            return np.load(path, allow_pickle=allow_pickle)
    return None


def _load_first_existing(base_dir: Path, name: str, *, noised: bool = False, allow_pickle: bool = False) -> Optional[np.ndarray]:
    names = [_with_suffix(name)] if noised else []
    names.append(name)
    return _load_npy_optional(base_dir, names, allow_pickle=allow_pickle)


def _resolve_subject_model_xml(subject_dir: Path) -> Optional[str]:
    try:
        return str(resolve_modq_runtime_xml(subject_dir))
    except Exception:
        fixed = subject_dir / "MyosuiteModel_FIXED.xml"
        raw = subject_dir / "MyosuiteModel.xml"
        if fixed.exists():
            return str(fixed)
        if raw.exists():
            return str(raw)
        return None


def _modq_required_template_files(proc_dir: Path, use_noised: bool = True) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    required = [
        "pos_inputs.npy",
        "vel_inputs.npy",
        "acc_inputs.npy",
        "pos_mjx.npy",
        "qvel_mjx.npy",
        "qacc_mjx.npy",
    ]
    if use_noised:
        required.extend(
            [
                "pos_inputs_noised.npy",
                "vel_inputs_noised.npy",
                "acc_inputs_noised.npy",
                "pos_mjx_noised.npy",
                "qvel_mjx_noised.npy",
                "qacc_mjx_noised.npy",
            ]
        )
    for name in required:
        if not (proc_dir / name).exists():
            missing.append(name)
    return len(missing) == 0, missing


def discover_all_trials_modq(data_dir: str, refresh_cache: bool = False) -> List[Dict[str, Any]]:
    data_root = Path(data_dir)
    if not data_root.exists():
        return []

    cache_file = data_root / "trial_discovery_cache_modq.json"
    if cache_file.exists() and not refresh_cache:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                cache_usable = True
                valid = []
                for entry in cached:
                    td = Path(entry.get("training_data_path", ""))
                    xml_path = Path(str(entry.get("subject_model_xml", "")))
                    if xml_path.name != RUNTIME_XML_NAME or not xml_path.exists():
                        cache_usable = False
                        break
                    has_templates, _ = _modq_required_template_files(td, use_noised=True) if td.exists() and td.name == "ProcessedData" else (False, [])
                    if td.exists() and td.name == "ProcessedData" and has_templates:
                        valid.append(entry)
                if cache_usable and valid:
                    return valid
        except Exception:
            pass

    trials: List[Dict[str, Any]] = []
    skipped_missing_templates = 0
    for subject_dir in sorted([p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        subject_model_xml = _resolve_subject_model_xml(subject_dir)
        if subject_model_xml is None:
            continue
        for trial_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]):
            proc_dir = trial_dir / "ProcessedData"
            if not proc_dir.exists():
                continue
            valid_templates, _ = _modq_required_template_files(proc_dir, use_noised=True)
            if not valid_templates:
                skipped_missing_templates += 1
                continue
            pos_path = proc_dir / "pos_inputs.npy"
            if not pos_path.exists():
                pos_path = proc_dir / "pos_inputs_noised.npy"
            if not pos_path.exists():
                continue
            try:
                trial_len = int(np.load(pos_path, mmap_mode="r").shape[0])
            except Exception:
                continue
            trials.append(
                {
                    "subject": subject_dir.name,
                    "trial_name": f"{subject_dir.name}/{trial_dir.name}",
                    "training_data_path": str(proc_dir),
                    "length": trial_len,
                    "subject_model_xml": subject_model_xml,
                    "subject_structure_key": (
                        modq_runtime_structure_key(subject_model_xml)
                        if Path(subject_model_xml).name == RUNTIME_XML_NAME
                        else str(subject_model_xml)
                    ),
                }
            )

    try:
        cache_file.write_text(json.dumps(trials, indent=2), encoding="utf-8")
    except Exception:
        pass
    if skipped_missing_templates > 0:
        _ts_print(
            f"Excluded {skipped_missing_templates} mod_q trial(s) from discovery because they are still missing "
            "the required clean and/or noised MJX template files.",
        )
    return trials


def _load_trial_bundle(trial: TrialRecord, use_noised: bool) -> Optional[Dict[str, Any]]:
    proc_dir = Path(trial.training_data_path)
    trial_root = proc_dir.parent

    pos_name = _with_suffix("pos_inputs.npy") if use_noised else "pos_inputs.npy"
    vel_name = _with_suffix("vel_inputs.npy") if use_noised else "vel_inputs.npy"
    acc_name = _with_suffix("acc_inputs.npy") if use_noised else "acc_inputs.npy"
    pelvis_name = _with_suffix("pelvis_rot_matrix.npy") if use_noised else "pelvis_rot_matrix.npy"
    com_r_name = _with_suffix("COM_r.npy") if use_noised else "COM_r.npy"
    com_l_name = _with_suffix("COM_l.npy") if use_noised else "COM_l.npy"
    com_acc_name = _with_suffix("COM_Acc_Global.npy") if use_noised else "COM_Acc_Global.npy"
    fwd_name = _with_suffix("forwardVel.npy") if use_noised else "forwardVel.npy"
    fpa_name = _with_suffix("Foot_ProgressionAngle.npy") if use_noised else "Foot_ProgressionAngle.npy"
    calcn_name = _with_suffix("CalcnToFloor_AngleDeg.npy") if use_noised else "CalcnToFloor_AngleDeg.npy"

    pos = _load_first_existing(proc_dir, pos_name, noised=False)
    vel = _load_first_existing(proc_dir, vel_name, noised=False)
    acc = _load_first_existing(proc_dir, acc_name, noised=False)
    pos_gt = _load_first_existing(proc_dir, "pos_inputs.npy", noised=False)
    vel_gt = _load_first_existing(proc_dir, "vel_inputs.npy", noised=False)
    acc_gt = _load_first_existing(proc_dir, "acc_inputs.npy", noised=False)
    pelvis_rot = _load_first_existing(proc_dir, pelvis_name, noised=False)
    com_r = _load_first_existing(proc_dir, com_r_name, noised=False)
    com_l = _load_first_existing(proc_dir, com_l_name, noised=False)
    com_accel = _load_first_existing(proc_dir, com_acc_name, noised=False)
    forward_vel = _load_first_existing(proc_dir, fwd_name, noised=False)
    foot_progression_angle = _load_first_existing(proc_dir, fpa_name, noised=False)
    calcn_to_floor_angle = _load_first_existing(proc_dir, calcn_name, noised=False)

    required = [
        pos,
        vel,
        acc,
        pos_gt,
        vel_gt,
        acc_gt,
        pelvis_rot,
        com_r,
        com_l,
        com_accel,
        foot_progression_angle,
        calcn_to_floor_angle,
    ]
    if any(x is None for x in required):
        return None

    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    acc = np.asarray(acc, dtype=np.float32)
    pos_gt = np.asarray(pos_gt, dtype=np.float32)
    vel_gt = np.asarray(vel_gt, dtype=np.float32)
    acc_gt = np.asarray(acc_gt, dtype=np.float32)
    pelvis_rot = np.asarray(pelvis_rot, dtype=np.float32)
    if pelvis_rot.ndim == 3:
        pelvis_rot = pelvis_rot[:, :, :2].reshape(len(pelvis_rot), 6)
    com_r = np.asarray(com_r, dtype=np.float32)
    com_l = np.asarray(com_l, dtype=np.float32)
    com_accel = np.asarray(com_accel, dtype=np.float32)
    foot_progression_angle = np.asarray(foot_progression_angle, dtype=np.float32)
    calcn_to_floor_angle = np.asarray(calcn_to_floor_angle, dtype=np.float32)

    if foot_progression_angle.ndim == 1:
        foot_progression_angle = foot_progression_angle[:, None]
    if foot_progression_angle.shape[1] == 1:
        foot_progression_angle = np.repeat(foot_progression_angle, 2, axis=1)
    if calcn_to_floor_angle.ndim == 1:
        calcn_to_floor_angle = calcn_to_floor_angle[:, None]
    if calcn_to_floor_angle.shape[1] == 1:
        calcn_to_floor_angle = np.repeat(calcn_to_floor_angle, 2, axis=1)

    height = _load_first_existing(proc_dir, "Height_m.npy")
    mass = _load_first_existing(proc_dir, "Mass_kg.npy")
    patient_size = _load_first_existing(trial_root.parent, "PatientSize.npy")
    if patient_size is None:
        patient_size = _load_first_existing(trial_root, "PatientSize.npy")
    if patient_size is None:
        patient_size = np.zeros(4, dtype=np.float32)
    patient_size = np.asarray(patient_size, dtype=np.float32).reshape(-1)
    if patient_size.size < 4:
        padded = np.zeros(4, dtype=np.float32)
        padded[: patient_size.size] = patient_size
        patient_size = padded

    gender = 0.5
    md_path = trial_root / "Patient_MD.json"
    if md_path.exists():
        try:
            md = json.loads(md_path.read_text(encoding="utf-8"))
            sex = str(md.get("BiologicalSex", "")).lower()
            if sex == "male":
                gender = 1.0
            elif sex == "female":
                gender = 0.0
        except Exception:
            pass

    if height is None or mass is None:
        return None
    height = np.asarray(height, dtype=np.float32).reshape(-1, 1)
    mass = np.asarray(mass, dtype=np.float32).reshape(-1, 1)
    forward_vel_scalar = float(np.mean(np.asarray(forward_vel, dtype=np.float32).reshape(-1))) if forward_vel is not None else 0.0

    cop_calc = _load_first_existing(proc_dir, "COP_CalcFrame_GroundAligned.npy")
    grf = _load_first_existing(proc_dir, "GRF_Cleaned.npy")
    if grf is None:
        grf = _load_first_existing(proc_dir, "GRF_Filtered.npy")
    moments = _load_first_existing(proc_dir, "Moment_Cleaned.npy")
    if moments is None:
        moments = _load_first_existing(proc_dir, "GRM_Filtered.npy")
    contact_boolean = _load_first_existing(proc_dir, "contactBoolean.npy")
    ankle_heights = _load_first_existing(proc_dir, "ankle_heights.npy")
    gt_rot_w_to_ga = _load_first_existing(proc_dir, "WorldToGroundAlignedCalcnRotation.npy")
    jacobian_data = _load_first_existing(proc_dir, "Jacobian.npy", allow_pickle=True)
    qfrc_grf_contribution = _load_first_existing(proc_dir, "qfrc_grf_contribution.npy")
    qfrc_inverse_gt = _load_first_existing(proc_dir, "qfrc_inverse.npy")
    id_gt_mjx = _load_first_existing(proc_dir, "ID_GT_MJX.npy")
    qpos_input = _load_first_existing(proc_dir, _with_suffix("pos_mjx.npy") if use_noised else "pos_mjx.npy")
    qvel_input = _load_first_existing(proc_dir, _with_suffix("qvel_mjx.npy") if use_noised else "qvel_mjx.npy")
    if qvel_input is None:
        qvel_input = _load_first_existing(proc_dir, _with_suffix("vel_mjx.npy") if use_noised else "vel_mjx.npy")
    qacc_input = _load_first_existing(proc_dir, _with_suffix("qacc_mjx.npy") if use_noised else "qacc_mjx.npy")
    if qacc_input is None:
        qacc_input = _load_first_existing(proc_dir, _with_suffix("acc_mjx.npy") if use_noised else "acc_mjx.npy")

    if any(
        x is None
        for x in [
            cop_calc,
            grf,
            moments,
            contact_boolean,
            ankle_heights,
            gt_rot_w_to_ga,
            jacobian_data,
            qfrc_inverse_gt,
            qpos_input,
        ]
    ):
        return None

    cop_calc = np.asarray(cop_calc, dtype=np.float32)
    if cop_calc.ndim != 2 or cop_calc.shape[1] < 6:
        return None
    cop = np.column_stack([cop_calc[:, 0], cop_calc[:, 2], cop_calc[:, 3], cop_calc[:, 5]]).astype(np.float32)
    grf = np.asarray(grf, dtype=np.float32)
    moments = np.asarray(moments, dtype=np.float32)
    if moments.ndim == 1:
        moments = moments[:, None]
    if moments.shape[1] >= 6:
        moments = moments[:, [2, 5]]
    contact_boolean = np.asarray(contact_boolean, dtype=np.float32)
    if contact_boolean.ndim == 1:
        contact_boolean = contact_boolean[:, None]
    if contact_boolean.shape[1] == 1:
        contact_boolean = np.repeat(contact_boolean, 2, axis=1)
    elif contact_boolean.shape[1] > 2:
        contact_boolean = contact_boolean[:, :2]
    ankle_heights = np.asarray(ankle_heights, dtype=np.float32)
    if ankle_heights.ndim == 1:
        ankle_heights = ankle_heights[:, None]
    gt_rot_w_to_ga = np.asarray(gt_rot_w_to_ga, dtype=np.float32)
    if gt_rot_w_to_ga.ndim != 4 or gt_rot_w_to_ga.shape[1:] != (2, 3, 3):
        return None
    jacobian_dict = jacobian_data.item() if hasattr(jacobian_data, "item") else jacobian_data
    jacp = np.asarray(jacobian_dict["jacp"], dtype=np.float32)
    jacr = np.asarray(jacobian_dict["jacr"], dtype=np.float32)
    body_ids = np.asarray(jacobian_dict.get("body_ids", np.array([0, 1], dtype=np.int32)))
    qfrc_inverse_gt = np.asarray(qfrc_inverse_gt, dtype=np.float32)
    if id_gt_mjx is None:
        id_gt_mjx = qfrc_inverse_gt.copy()
    else:
        id_gt_mjx = np.asarray(id_gt_mjx, dtype=np.float32)
    if qfrc_grf_contribution is None:
        qfrc_grf_contribution = qfrc_inverse_gt - id_gt_mjx
    else:
        qfrc_grf_contribution = np.asarray(qfrc_grf_contribution, dtype=np.float32)
    qpos_input = np.asarray(qpos_input, dtype=np.float32)
    if qvel_input is None:
        qvel_input = np.zeros_like(qpos_input, dtype=np.float32)
    else:
        qvel_input = np.asarray(qvel_input, dtype=np.float32)
    if qacc_input is None:
        qacc_input = np.zeros_like(qpos_input, dtype=np.float32)
    else:
        qacc_input = np.asarray(qacc_input, dtype=np.float32)

    lengths = [
        len(pos),
        len(vel),
        len(acc),
        len(pos_gt),
        len(vel_gt),
        len(acc_gt),
        len(pelvis_rot),
        len(com_r),
        len(com_l),
        len(com_accel),
        len(foot_progression_angle),
        len(calcn_to_floor_angle),
        len(cop),
        len(grf),
        len(moments),
        len(contact_boolean),
        len(ankle_heights),
        len(gt_rot_w_to_ga),
        len(jacp),
        len(jacr),
        len(qfrc_grf_contribution),
        len(qfrc_inverse_gt),
        len(qpos_input),
        len(qvel_input),
        len(qacc_input),
    ]
    if id_gt_mjx is not None:
        lengths.append(len(id_gt_mjx))
    min_len = min(lengths)

    def _trim(x):
        return x[:min_len]

    bundle = {
        "subject": trial.subject,
        "trial_name": trial.trial_name,
        "training_data_path": trial.training_data_path,
        "subject_model_xml": trial.subject_model_xml,
        "length": int(min_len),
        "pos": _trim(pos),
        "vel": _trim(vel),
        "acc": _trim(acc),
        "pos_noised": _trim(pos),
        "vel_noised": _trim(vel),
        "acc_noised": _trim(acc),
        "pos_gt": _trim(pos_gt),
        "vel_gt": _trim(vel_gt),
        "acc_gt": _trim(acc_gt),
        "pelvis_rot": _trim(pelvis_rot),
        "com_r": _trim(com_r),
        "com_l": _trim(com_l),
        "com_accel": _trim(com_accel),
        "foot_progression_angle": _trim(foot_progression_angle),
        "calcn_to_floor_angle": _trim(calcn_to_floor_angle),
        "height": _trim(height),
        "mass": _trim(mass),
        "gender": float(gender),
        "patient_size": np.asarray(patient_size, dtype=np.float32),
        "forward_vel": np.float32(forward_vel_scalar),
        # Store GRF/COP/GRM in body-scaled units here; `_prepare_batch` z-scores
        # them before they reach the model or direct losses.
        "cop": _trim(cop) / _trim(height),
        "grf": _trim(grf) / (_trim(mass) * 9.8067),
        "moments": _trim(moments) / (_trim(mass) * 9.8067 * _trim(height)),
        "contactBoolean": _trim(contact_boolean),
        "ankle_heights": _trim(ankle_heights),
        "rot_w_to_ga": _trim(gt_rot_w_to_ga),
        "gt_rot_w_to_ga": _trim(gt_rot_w_to_ga),
        "jacp": _trim(jacp),
        "jacr": _trim(jacr),
        "body_ids": body_ids,
        "qfrc_grf_contribution": _trim(qfrc_grf_contribution),
        "qfrc_inverse_gt": _trim(qfrc_inverse_gt),
        "id_gt_mjx": _trim(id_gt_mjx) if id_gt_mjx is not None else None,
        "qpos_mjx_input": _trim(qpos_input),
        "qvel_mjx_input": _trim(qvel_input),
        "qacc_mjx_input": _trim(qacc_input),
    }
    return bundle


def _build_windows_for_trial(
    bundle: Dict[str, Any],
    window_size: int,
    stride: int,
    prediction_margin_frames: int,
) -> List[Dict[str, Any]]:
    starts = build_window_start_indices(len(bundle["pos"]), window_size, stride)
    windows: List[Dict[str, Any]] = []
    for start in starts:
        end = start + window_size
        if end > len(bundle["pos"]):
            continue
        supervision_mask = build_window_supervision_mask(
            window_size=window_size,
            window_start_idx=start,
            trial_length=len(bundle["pos"]),
            prediction_margin_frames=prediction_margin_frames,
        )
        input_parts = [
            bundle["pelvis_rot"][start:end],
            bundle["pos"][start:end],
            bundle["vel"][start:end],
            bundle["acc"][start:end],
            bundle["com_r"][start:end],
            bundle["com_l"][start:end],
            bundle["com_accel"][start:end],
            bundle["foot_progression_angle"][start:end],
            bundle["calcn_to_floor_angle"][start:end],
        ]
        input_window = np.concatenate(input_parts, axis=1).astype(np.float32)
        static_context = build_modq_static_context(bundle).astype(np.float32)
        windows.append(
            {
                "input": input_window,
                "static_context": static_context,
                "supervision_mask": supervision_mask.astype(np.float32),
                "trial_name": bundle["trial_name"],
                "subject": bundle["subject"],
                "subject_model_xml": bundle["subject_model_xml"],
                "window_start_idx": np.int32(start),
                "trial_length": np.int32(len(bundle["pos"])),
                "cop": bundle["cop"][start:end],
                "grf": bundle["grf"][start:end],
                "moments": bundle["moments"][start:end],
                "contactBoolean": bundle["contactBoolean"][start:end],
                "ankle_heights": bundle["ankle_heights"][start:end],
                "rot_w_to_ga": bundle["rot_w_to_ga"][start:end],
                "gt_rot_w_to_ga": bundle["gt_rot_w_to_ga"][start:end],
                "jacp": bundle["jacp"][start:end],
                "jacr": bundle["jacr"][start:end],
                "qfrc_grf_contribution": bundle["qfrc_grf_contribution"][start:end],
                "qfrc_inverse_gt": bundle["qfrc_inverse_gt"][start:end],
                "id_gt_mjx": bundle["id_gt_mjx"][start:end] if bundle.get("id_gt_mjx") is not None else None,
                "pos_noised": bundle["pos_noised"][start:end],
                "vel_noised": bundle["vel_noised"][start:end],
                "acc_noised": bundle["acc_noised"][start:end],
                "pos_gt": bundle["pos_gt"][start:end],
                "vel_gt": bundle["vel_gt"][start:end],
                "acc_gt": bundle["acc_gt"][start:end],
                "qpos_mjx_input": bundle["qpos_mjx_input"][start:end],
                "qvel_mjx_input": bundle["qvel_mjx_input"][start:end],
                "qacc_mjx_input": bundle["qacc_mjx_input"][start:end],
                "com_accel": bundle["com_accel"][start:end],
                "height": bundle["height"][start:end],
                "mass": bundle["mass"][start:end],
                "gender": bundle["gender"],
                "patient_size": bundle["patient_size"],
                "forward_vel": bundle["forward_vel"],
            }
        )
    return windows


def _collate_windows(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    for key in windows[0].keys():
        values = [w[key] for w in windows]
        if isinstance(values[0], np.ndarray):
            batch[key] = np.stack(values, axis=0)
        elif isinstance(values[0], (float, np.floating, int, np.integer)):
            batch[key] = np.asarray(values)
        else:
            batch[key] = values
    return batch


class ModQSubjectBatcher:
    def __init__(
        self,
        trials: List[Dict[str, Any]],
        *,
        window_size: int,
        stride: int,
        prediction_margin_frames: int,
        batch_size: int,
        shuffle: bool,
        use_noised: bool,
        cache_trials: bool = True,
    ):
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.prediction_margin_frames = int(prediction_margin_frames)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.use_noised = bool(use_noised)
        self._cache_trials = bool(cache_trials)
        self._trial_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._subject_window_cache: Dict[str, List[Dict[str, Any]]] = {}

        self.subject_to_trials: Dict[str, List[TrialRecord]] = defaultdict(list)
        self.subject_to_structure_key: Dict[str, str] = {}
        for trial in trials:
            rec = TrialRecord(
                subject=str(trial["subject"]),
                trial_name=str(trial["trial_name"]),
                training_data_path=str(trial["training_data_path"]),
                length=int(trial["length"]),
                subject_model_xml=str(trial.get("subject_model_xml", "")),
                subject_structure_key=str(trial.get("subject_structure_key", "")),
            )
            self.subject_to_trials[rec.subject].append(rec)
            if rec.subject not in self.subject_to_structure_key:
                self.subject_to_structure_key[rec.subject] = rec.subject_structure_key

        self.subjects = sorted(
            self.subject_to_trials.keys(),
            key=lambda subject: (self.subject_to_structure_key.get(subject, ""), subject),
        )

    def _window_count_for_trial_length(self, trial_length: int) -> int:
        trial_length = int(trial_length)
        if trial_length <= 0:
            return 0
        if trial_length <= self.window_size:
            return 1
        starts = range(0, trial_length - self.window_size + 1, self.stride)
        count = len(starts)
        tail_start = trial_length - self.window_size
        if count == 0:
            return 1
        last_start = (count - 1) * self.stride
        if last_start != tail_start:
            count += 1
        return count

    def _window_count_for_subject(self, subject: str) -> int:
        return sum(self._window_count_for_trial_length(trial.length) for trial in self.subject_to_trials.get(subject, []))

    def _ordered_subjects(self) -> List[str]:
        subjects = list(self.subjects)
        if not self.shuffle:
            return subjects

        grouped: Dict[str, List[str]] = defaultdict(list)
        for subject in subjects:
            grouped[self.subject_to_structure_key.get(subject, "")].append(subject)

        structure_keys = list(grouped.keys())
        random.shuffle(structure_keys)

        ordered: List[str] = []
        for structure_key in structure_keys:
            subject_group = list(grouped[structure_key])
            random.shuffle(subject_group)
            ordered.extend(subject_group)
        return ordered

    def _load_bundle(self, trial: TrialRecord) -> Optional[Dict[str, Any]]:
        if self._cache_trials and trial.training_data_path in self._trial_cache:
            return self._trial_cache[trial.training_data_path]
        bundle = _load_trial_bundle(trial, self.use_noised)
        if self._cache_trials:
            self._trial_cache[trial.training_data_path] = bundle
        return bundle

    def _get_subject_windows_cached(self, subject: str) -> List[Dict[str, Any]]:
        if subject in self._subject_window_cache:
            return list(self._subject_window_cache[subject])
        windows = []
        trials = list(self.subject_to_trials.get(subject, []))
        for trial in trials:
            bundle = self._load_bundle(trial)
            if bundle is None:
                continue
            windows.extend(
                _build_windows_for_trial(
                    bundle,
                    self.window_size,
                    self.stride,
                    self.prediction_margin_frames,
                )
            )
        self._subject_window_cache[subject] = list(windows)
        return list(windows)

    def _iter_subject_windows(self, subject: str) -> List[Dict[str, Any]]:
        windows = self._get_subject_windows_cached(subject)
        if self.shuffle:
            random.shuffle(windows)
        return windows

    def _release_subject_cache(self, subject: str) -> None:
        self._subject_window_cache.pop(subject, None)
        trial_records = self.subject_to_trials.get(subject, [])
        for trial in trial_records:
            self._trial_cache.pop(trial.training_data_path, None)

    def num_batches(self) -> int:
        total = 0
        for subject in self.subjects:
            total += self._window_count_for_subject(subject) // self.batch_size
        return total

    def iter_batches(self) -> Iterable[Dict[str, Any]]:
        subjects = self._ordered_subjects()
        for subject in subjects:
            windows = self._iter_subject_windows(subject)
            try:
                for start in range(0, len(windows), self.batch_size):
                    chunk = windows[start : start + self.batch_size]
                    if not chunk or len(chunk) < self.batch_size:
                        continue
                    batch = _collate_windows(chunk)
                    batch["subject"] = subject
                    batch["subject_grouped_batches"] = True
                    yield batch
            finally:
                self._release_subject_cache(subject)


def _stack_normalizer_values(values: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(v).reshape(-1, v.shape[-1]) for v in values], axis=0)


def _fit_normalizer(values: List[np.ndarray], *, eps: float) -> Normalizer:
    stacked = _stack_normalizer_values(values)
    return Normalizer(stacked.mean(axis=0), stacked.std(axis=0), eps=eps)


def compute_normalizers_from_batches(batch_iter: Iterable[Dict[str, Any]], max_batches: int = 100) -> Dict[str, Normalizer]:
    input_samples: List[np.ndarray] = []
    static_samples: List[np.ndarray] = []
    cop_samples: List[np.ndarray] = []
    grf_samples: List[np.ndarray] = []
    moments_samples: List[np.ndarray] = []
    pos_samples: List[np.ndarray] = []
    vel_target_samples: List[np.ndarray] = []
    acc_samples: List[np.ndarray] = []
    tau_samples: List[np.ndarray] = []
    grf_res_samples: List[np.ndarray] = []

    for batch_idx, batch in enumerate(batch_iter):
        input_samples.append(np.asarray(batch["input"], dtype=np.float32))
        static_samples.append(np.asarray(batch["static_context"], dtype=np.float32))
        cop_samples.append(np.asarray(batch["cop"], dtype=np.float32))
        grf_samples.append(np.asarray(batch["grf"], dtype=np.float32))
        moments_samples.append(np.asarray(batch["moments"], dtype=np.float32))
        pos_samples.append(np.asarray(batch["pos_gt"], dtype=np.float32))
        vel_target_samples.append(np.asarray(batch["vel_gt"], dtype=np.float32))
        acc_samples.append(np.asarray(batch["acc_gt"], dtype=np.float32))
        tau_samples.append(np.asarray(batch["qfrc_grf_contribution"], dtype=np.float32))

        com_accel = np.asarray(batch["com_accel"], dtype=np.float32)
        grf = np.asarray(batch["grf"], dtype=np.float32)
        mass = np.asarray(batch["mass"], dtype=np.float32)
        if mass.ndim == 3 and mass.shape[-1] == 1:
            mass = mass[..., 0]
        body_weight = mass * 9.8067
        fx = (grf[..., 0] + grf[..., 3]) * body_weight
        fy = (grf[..., 1] + grf[..., 4]) * body_weight
        fz = (grf[..., 2] + grf[..., 5]) * body_weight
        res_x = mass * com_accel[..., 0] - fx
        res_y = mass * com_accel[..., 1] - fy
        res_z = mass * (com_accel[..., 2] + 9.8067) - fz
        grf_res_samples.append(np.stack([res_x, res_y, res_z], axis=-1))

        if batch_idx + 1 >= max_batches:
            break

    if not input_samples:
        raise RuntimeError("No training batches were available for normalizer computation.")

    normalizers = {
        "input": _fit_normalizer(input_samples, eps=1e-8),
        "static": _fit_normalizer(static_samples, eps=1e-8),
        "cop": _fit_normalizer(cop_samples, eps=1e-3),
        "grf": _fit_normalizer(grf_samples, eps=1e-3),
        "moments": _fit_normalizer(moments_samples, eps=1e-3),
        "pos": _fit_normalizer(pos_samples, eps=1e-3),
        "vel": _fit_normalizer(vel_target_samples, eps=1e-3),
        "acc": _fit_normalizer(acc_samples, eps=1e-3),
        "tau": _fit_normalizer(tau_samples, eps=1e-3),
        "grf_res": _fit_normalizer(grf_res_samples, eps=1e-3),
    }
    return normalizers


def normalize_batch(batch: Dict[str, Any], normalizers: Dict[str, Normalizer]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if key == "input":
            out[key] = normalizers["input"].normalize(value)
        elif key == "static_context":
            out[key] = normalizers["static"].normalize(value)
        elif key == "cop":
            out[key] = normalizers["cop"].normalize(value)
        elif key == "grf":
            out[key] = normalizers["grf"].normalize(value)
        elif key == "moments":
            out[key] = normalizers["moments"].normalize(value)
        elif key == "pos_gt":
            out[key] = normalizers["pos"].normalize(value)
        elif key == "pos_noised":
            out[key] = normalizers["pos"].normalize(value)
        elif key == "vel_gt":
            out[key] = normalizers["vel"].normalize(value)
        elif key == "vel_noised":
            out[key] = normalizers["vel"].normalize(value)
        elif key == "acc_gt":
            out[key] = normalizers["acc"].normalize(value)
        elif key == "acc_noised":
            out[key] = normalizers["acc"].normalize(value)
        elif key == "qfrc_grf_contribution":
            out[key] = value
        else:
            out[key] = value
    return out


def _raw_scale_factors_from_batch(
    batch: Dict[str, Any],
    *,
    dtype: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return raw per-frame height, mass, BW, and BW*height scales.

    These values must come from the untouched batch tensors, not the z-scored
    `static_context`, because they are used to restore physical units before the
    MJX/torque path runs.
    """
    height_raw = jnp.asarray(batch["height"], dtype=dtype)
    mass_raw = jnp.asarray(batch["mass"], dtype=dtype)
    if height_raw.ndim == 3 and height_raw.shape[-1] == 1:
        height_raw = height_raw[..., 0]
    if mass_raw.ndim == 3 and mass_raw.shape[-1] == 1:
        mass_raw = mass_raw[..., 0]
    body_weight_raw = mass_raw * jnp.asarray(9.8067, dtype=dtype)
    bw_height_raw = body_weight_raw * height_raw
    return height_raw, mass_raw, body_weight_raw, bw_height_raw


def _restore_output_units(
    cop_z: jnp.ndarray,
    grf_z: jnp.ndarray,
    moments_z: jnp.ndarray,
    normalizers: Dict[str, Normalizer],
    *,
    height_raw: jnp.ndarray,
    body_weight_raw: jnp.ndarray,
    bw_height_raw: jnp.ndarray,
    contact_scale: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert z-score predictions back to physical units.

    `cop_z`, `grf_z`, and `moments_z` are z-scores of already body-scaled
    signals:
    - COP: height-normalized
    - GRF: body-weight-normalized
    - GRM/free moment: (body-weight * height)-normalized
    """
    cop_height_norm = normalizers["cop"].unnormalize(cop_z)
    grf_bw_norm = normalizers["grf"].unnormalize(grf_z)
    moments_bw_height_norm = normalizers["moments"].unnormalize(moments_z)

    if contact_scale is not None:
        cop_height_norm = jnp.concatenate(
            [
                cop_height_norm[..., 0:2] * contact_scale[..., 0:1],
                cop_height_norm[..., 2:4] * contact_scale[..., 1:2],
            ],
            axis=-1,
        )
        grf_bw_norm = jnp.concatenate(
            [
                grf_bw_norm[..., 0:3] * contact_scale[..., 0:1],
                grf_bw_norm[..., 3:6] * contact_scale[..., 1:2],
            ],
            axis=-1,
        )
        moments_bw_height_norm = jnp.concatenate(
            [
                moments_bw_height_norm[..., 0:1] * contact_scale[..., 0:1],
                moments_bw_height_norm[..., 1:2] * contact_scale[..., 1:2],
            ],
            axis=-1,
        )

    cop_phys = cop_height_norm * height_raw[..., None]
    grf_phys = grf_bw_norm * body_weight_raw[..., None]
    moments_phys = moments_bw_height_norm * bw_height_raw[..., None]
    return cop_phys, grf_phys, moments_phys


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray, weights: jnp.ndarray = 1.0) -> jnp.ndarray:
    return jnp.mean(weights * jnp.square(pred - target))


def _rmse(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean(jnp.square(x)))


def _indexed_rmse(x: jnp.ndarray, indices: Sequence[int]) -> jnp.ndarray:
    valid = [int(idx) for idx in indices if int(idx) < int(x.shape[-1])]
    if not valid:
        return jnp.asarray(0.0, dtype=x.dtype)
    return _rmse(x[..., valid])


def _weighted_loss_term(loss: jnp.ndarray, weight: float) -> jnp.ndarray:
    if float(weight) == 0.0:
        return jnp.asarray(0.0, dtype=loss.dtype)
    return jnp.asarray(float(weight), dtype=loss.dtype) * loss


def _losses_require_differentiable_mjx(loss_weights: Dict[str, float]) -> bool:
    return any(
        float(loss_weights.get(name, 0.0)) > 0.0
        for name in ("qfrc_inverse", "jacobian", "rotation", "full_id")
    )


def _required_mjx_gradient_channels(loss_weights: Dict[str, float]) -> Tuple[str, ...]:
    required: List[str] = []
    if float(loss_weights.get("qfrc_inverse", 0.0)) > 0.0 or float(loss_weights.get("full_id", 0.0)) > 0.0:
        required.append("acc")
    if float(loss_weights.get("jacobian", 0.0)) > 0.0 or float(loss_weights.get("rotation", 0.0)) > 0.0:
        required.append("pos")
    if not required:
        required.extend(["pos", "vel", "acc"])
    return tuple(dict.fromkeys(required))


WARMUP_DIRECT_LOSS_NAMES: Tuple[str, ...] = (
    "cop",
    "grf",
    "moments",
    "contact",
    "pos",
    "vel",
    "acc",
)

PHYSICS_DERIVED_LOSS_NAMES: Tuple[str, ...] = (
    "torque",
    "grf_correction",
    "qfrc_inverse",
    "jacobian",
    "rotation",
    "full_id",
)


def _make_warmup_loss_weights(full_loss_weights: Dict[str, float]) -> Dict[str, float]:
    warmup_loss_weights = dict(full_loss_weights)
    for loss_name in PHYSICS_DERIVED_LOSS_NAMES:
        warmup_loss_weights[loss_name] = 0.0
    return warmup_loss_weights


KINEMATIC_EQUIV_TERMS: Tuple[str, ...] = ("qfrc_inverse", "jacobian", "rotation")
KINEMATIC_EQUIV_COMPONENTS: Tuple[str, ...] = ("pos", "vel", "acc")


def _zero_kinematic_equiv_metrics(dtype: Any) -> Dict[str, jnp.ndarray]:
    zero = jnp.asarray(0.0, dtype=dtype)
    metrics: Dict[str, jnp.ndarray] = {}
    for term in KINEMATIC_EQUIV_TERMS:
        for component in KINEMATIC_EQUIV_COMPONENTS:
            for prefix in ("raw", "scaled"):
                metrics[f"{term}_{prefix}_equiv_{component}_loss"] = zero
                metrics[f"{term}_{prefix}_equiv_{component}_rmse"] = zero
                metrics[f"{term}_{prefix}_equiv_{component}_rmse_phys"] = zero
                metrics[f"{term}_{prefix}_grad_{component}_l2"] = zero
            metrics[f"{term}_equiv_{component}_loss"] = zero
            metrics[f"{term}_equiv_{component}_rmse"] = zero
            metrics[f"{term}_equiv_{component}_rmse_phys"] = zero
            metrics[f"{term}_grad_{component}_l2"] = zero
    return metrics


def _equivalent_mse_metrics_from_grad(
    grad_slice: jnp.ndarray,
    scale_std: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    grad_arr = jnp.asarray(grad_slice)
    std_arr = jnp.asarray(scale_std, dtype=grad_arr.dtype)
    element_count = max(1, int(grad_arr.size))
    equiv_error = 0.5 * jnp.asarray(float(element_count), dtype=grad_arr.dtype) * grad_arr
    equiv_loss = jnp.mean(jnp.square(equiv_error))
    equiv_rmse = jnp.sqrt(equiv_loss)
    equiv_rmse_phys = jnp.sqrt(jnp.mean(jnp.square(equiv_error * std_arr)))
    grad_l2 = jnp.linalg.norm(grad_arr)
    return {
        "loss": equiv_loss,
        "rmse": equiv_rmse,
        "rmse_phys": equiv_rmse_phys,
        "grad_l2": grad_l2,
    }


def _qfrc_inverse_loss_from_physics(
    physics: Dict[str, jnp.ndarray],
    batch: Dict[str, Any],
    norm_factor: jnp.ndarray,
    *,
    dtype: Any,
) -> jnp.ndarray:
    return jnp.mean(
        jnp.square((physics["qfrc_inverse"] - jnp.asarray(batch["qfrc_inverse_gt"], dtype=dtype)) / norm_factor[..., None])
    )


def _jacobian_loss_from_physics(
    physics: Dict[str, jnp.ndarray],
    batch: Dict[str, Any],
    *,
    dtype: Any,
) -> jnp.ndarray:
    return (
        jnp.mean(jnp.square(physics["jacp"] - jnp.asarray(batch["jacp"], dtype=dtype)))
        + jnp.mean(jnp.square(physics["jacr"] - jnp.asarray(batch["jacr"], dtype=dtype)))
    )


def _rotation_loss_from_physics(
    physics: Dict[str, jnp.ndarray],
    rot_w_to_ga_gt: jnp.ndarray,
    supervision_mask: jnp.ndarray,
) -> jnp.ndarray:
    return geodesic_rotation_mse(
        physics["rot_w_to_ga"],
        rot_w_to_ga_gt,
        supervision_mask,
        xp=jnp,
        project=False,
    )


def _full_id_loss_from_physics(
    physics: Dict[str, jnp.ndarray],
    batch: Dict[str, Any],
    norm_factor: jnp.ndarray,
    *,
    dtype: Any,
) -> jnp.ndarray:
    return jnp.mean(
        jnp.square((physics["full_id"] - jnp.asarray(batch["id_gt_mjx"], dtype=dtype)) / norm_factor[..., None])
    )


def _compute_physics_term_losses(
    physics: Dict[str, jnp.ndarray],
    batch: Dict[str, Any],
    *,
    norm_factor: jnp.ndarray,
    rot_w_to_ga_gt: jnp.ndarray,
    supervision_mask: jnp.ndarray,
    dtype: Any,
) -> Dict[str, jnp.ndarray]:
    return {
        "qfrc_inverse": _qfrc_inverse_loss_from_physics(physics, batch, norm_factor, dtype=dtype),
        "jacobian": _jacobian_loss_from_physics(physics, batch, dtype=dtype),
        "rotation": _rotation_loss_from_physics(physics, rot_w_to_ga_gt, supervision_mask),
        "full_id": _full_id_loss_from_physics(physics, batch, norm_factor, dtype=dtype),
    }


def _compute_kinematic_equivalent_metrics(
    *,
    batch: Dict[str, Any],
    normalizers: Dict[str, Normalizer],
    loss_weights: Dict[str, float],
    cop_pred: jnp.ndarray,
    grf_pred: jnp.ndarray,
    moments_pred: jnp.ndarray,
    contact_pred: jnp.ndarray,
    pos_pred: jnp.ndarray,
    vel_pred: jnp.ndarray,
    acc_pred: jnp.ndarray,
    cop_mask: bool,
    physics_runner: Optional[Any],
    kinematics_reconstructor: Optional[Any],
    physics_context: Optional[Dict[str, Any]],
    run_physics: bool,
) -> Dict[str, jnp.ndarray]:
    dtype = pos_pred.dtype
    zero_metrics = _zero_kinematic_equiv_metrics(dtype)
    if (
        not run_physics
        or physics_runner is None
        or kinematics_reconstructor is None
        or physics_context is None
        or not physics_context
    ):
        return zero_metrics

    height_raw, _mass_raw, body_weight_raw, norm_factor = _raw_scale_factors_from_batch(batch, dtype=dtype)
    supervision_mask = jnp.asarray(
        batch.get("supervision_mask", np.ones(pos_pred.shape[:2] + (1,), dtype=np.float32)),
        dtype=dtype,
    )
    if supervision_mask.ndim == 2:
        supervision_mask = supervision_mask[..., None]
    rot_w_to_ga_gt = jnp.asarray(batch["gt_rot_w_to_ga"], dtype=dtype)

    def _term_loss_for_outputs(
        pos_inner: jnp.ndarray,
        vel_inner: jnp.ndarray,
        acc_inner: jnp.ndarray,
        term_name: str,
    ) -> jnp.ndarray:
        contact_soft = jnp.clip(contact_pred, 0.0, 1.0) if cop_mask else None
        cop_phys, grf_phys, moments_phys = _restore_output_units(
            cop_pred,
            grf_pred,
            moments_pred,
            normalizers,
            height_raw=height_raw,
            body_weight_raw=body_weight_raw,
            bw_height_raw=norm_factor,
            contact_scale=contact_soft,
        )
        pos_phys = normalizers["pos"].unnormalize(pos_inner)
        vel_phys = normalizers["vel"].unnormalize(vel_inner)
        acc_phys = normalizers["acc"].unnormalize(acc_inner)
        flat_frames = pos_phys.shape[0] * pos_phys.shape[1]
        qpos_flat, qvel_flat, qacc_flat = kinematics_reconstructor(
            pos_phys.reshape((flat_frames, pos_phys.shape[-1])),
            vel_phys.reshape((flat_frames, vel_phys.shape[-1])),
            acc_phys.reshape((flat_frames, acc_phys.shape[-1])),
            jnp.asarray(batch["qpos_mjx_input"], dtype=dtype).reshape((flat_frames, batch["qpos_mjx_input"].shape[-1])),
            jnp.asarray(batch["qvel_mjx_input"], dtype=dtype).reshape((flat_frames, batch["qvel_mjx_input"].shape[-1])),
            jnp.asarray(batch["qacc_mjx_input"], dtype=dtype).reshape((flat_frames, batch["qacc_mjx_input"].shape[-1])),
            jnp.asarray(physics_context["slave_idx"], dtype=jnp.int32),
            jnp.asarray(physics_context["master_idx"], dtype=jnp.int32),
            jnp.asarray(physics_context["coeffs"], dtype=jnp.float32),
        )
        qpos_phys = qpos_flat.reshape(pos_phys.shape[:2] + (qpos_flat.shape[-1],))
        qvel_phys = qvel_flat.reshape(vel_phys.shape[:2] + (qvel_flat.shape[-1],))
        qacc_phys = qacc_flat.reshape(acc_phys.shape[:2] + (qacc_flat.shape[-1],))
        physics_flat = physics_runner(
            physics_context["mjx_model"],
            jnp.asarray(physics_context["calcn_r_id"], dtype=jnp.int32),
            jnp.asarray(physics_context["calcn_l_id"], dtype=jnp.int32),
            qpos_phys.reshape((flat_frames, qpos_phys.shape[-1])),
            qvel_phys.reshape((flat_frames, qvel_phys.shape[-1])),
            qacc_phys.reshape((flat_frames, qacc_phys.shape[-1])),
            cop_phys.reshape((flat_frames, cop_phys.shape[-1])),
            grf_phys.reshape((flat_frames, grf_phys.shape[-1])),
            moments_phys.reshape((flat_frames, moments_phys.shape[-1])),
            jnp.asarray(batch["ankle_heights"], dtype=dtype).reshape((flat_frames, batch["ankle_heights"].shape[-1])),
        )
        physics = {
            key: value.reshape(qpos_phys.shape[:2] + value.shape[1:])
            for key, value in physics_flat.items()
        }
        term_losses = _compute_physics_term_losses(
            physics,
            batch,
            norm_factor=norm_factor,
            rot_w_to_ga_gt=rot_w_to_ga_gt,
            supervision_mask=supervision_mask,
            dtype=dtype,
        )
        return term_losses[term_name]

    std_map = {
        "pos": jnp.asarray(normalizers["pos"].std, dtype=dtype),
        "vel": jnp.asarray(normalizers["vel"].std, dtype=dtype),
        "acc": jnp.asarray(normalizers["acc"].std, dtype=dtype),
    }
    metrics = dict(zero_metrics)
    for term in KINEMATIC_EQUIV_TERMS:
        weight = float(loss_weights.get(term, 0.0))

        def _term_fn(pos_inner: jnp.ndarray, vel_inner: jnp.ndarray, acc_inner: jnp.ndarray) -> jnp.ndarray:
            return _term_loss_for_outputs(pos_inner, vel_inner, acc_inner, term)

        grad_pos, grad_vel, grad_acc = jax.grad(_term_fn, argnums=(0, 1, 2))(pos_pred, vel_pred, acc_pred)
        for component, grad_value in (("pos", grad_pos), ("vel", grad_vel), ("acc", grad_acc)):
            raw_equiv = _equivalent_mse_metrics_from_grad(grad_value, std_map[component])
            scaled_grad = jnp.asarray(weight, dtype=dtype) * grad_value
            scaled_equiv = _equivalent_mse_metrics_from_grad(scaled_grad, std_map[component])
            metrics[f"{term}_raw_equiv_{component}_loss"] = raw_equiv["loss"]
            metrics[f"{term}_raw_equiv_{component}_rmse"] = raw_equiv["rmse"]
            metrics[f"{term}_raw_equiv_{component}_rmse_phys"] = raw_equiv["rmse_phys"]
            metrics[f"{term}_raw_grad_{component}_l2"] = raw_equiv["grad_l2"]
            metrics[f"{term}_scaled_equiv_{component}_loss"] = scaled_equiv["loss"]
            metrics[f"{term}_scaled_equiv_{component}_rmse"] = scaled_equiv["rmse"]
            metrics[f"{term}_scaled_equiv_{component}_rmse_phys"] = scaled_equiv["rmse_phys"]
            metrics[f"{term}_scaled_grad_{component}_l2"] = scaled_equiv["grad_l2"]
            metrics[f"{term}_equiv_{component}_loss"] = scaled_equiv["loss"]
            metrics[f"{term}_equiv_{component}_rmse"] = scaled_equiv["rmse"]
            metrics[f"{term}_equiv_{component}_rmse_phys"] = scaled_equiv["rmse_phys"]
            metrics[f"{term}_grad_{component}_l2"] = scaled_equiv["grad_l2"]
    return metrics


def _stage_for_epoch(epoch: int, warmup_epochs: int) -> str:
    return "warmup" if int(epoch) <= max(0, int(warmup_epochs)) else "full"


def _stage_runs_physics(stage: str) -> bool:
    return stage == "full"


def _stage_description(stage: str) -> str:
    if stage == "warmup":
        return "direct losses only, no MJX, no predicted torque"
    return "full physics stage"


def _slice_batch_for_gradient_smoke_test(batch: Dict[str, Any], frames: int = 4) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    seq_len = int(batch["cop"].shape[1])
    frame_count = max(1, min(int(frames), seq_len))

    for key, value in batch.items():
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            arr = jnp.asarray(value)
            if arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[1] == seq_len:
                out[key] = arr[:1, :frame_count]
            elif arr.ndim >= 1 and arr.shape[0] > 0:
                out[key] = arr[:1]
            else:
                out[key] = arr
        elif isinstance(value, list):
            out[key] = value[:1]
        else:
            out[key] = value
    return out


def _gradient_slice_stats(arr: jnp.ndarray) -> Dict[str, float]:
    grad_arr = jnp.asarray(arr)
    abs_grad = jnp.abs(grad_arr)
    return {
        "finite": 1.0 if bool(jnp.all(jnp.isfinite(grad_arr))) else 0.0,
        "absmax": float(jnp.max(abs_grad)) if grad_arr.size else 0.0,
        "l2": float(jnp.linalg.norm(grad_arr)) if grad_arr.size else 0.0,
    }


def _run_mjx_gradient_smoke_test(
    *,
    batch: Dict[str, Any],
    normalizers: Dict[str, Normalizer],
    physics_context: Dict[str, Any],
    physics_runner: Any,
    kinematics_reconstructor: Any,
    deviation_learning: bool,
    cop_mask: bool,
    use_contact_weighting: bool,
    contact_weight_multiplier: float,
    required_channels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    if physics_runner is None or kinematics_reconstructor is None or not physics_context:
        raise RuntimeError("MJX gradient smoke test requires an initialized runner, reconstructor, and physics context.")

    probe_batch = _slice_batch_for_gradient_smoke_test(batch, frames=4)
    contact_probe = jnp.clip(0.25 + 0.5 * jnp.asarray(probe_batch["contactBoolean"], dtype=jnp.float32), 0.05, 0.95)
    pred_probe = jnp.concatenate(
        [
            jnp.asarray(probe_batch["cop"], dtype=jnp.float32),
            jnp.asarray(probe_batch["grf"], dtype=jnp.float32),
            jnp.asarray(probe_batch["moments"], dtype=jnp.float32),
            contact_probe,
            jnp.asarray(probe_batch["pos_gt"], dtype=jnp.float32),
            jnp.asarray(probe_batch["vel_gt"], dtype=jnp.float32),
            jnp.asarray(probe_batch["acc_gt"], dtype=jnp.float32),
        ],
        axis=-1,
    )
    zero_loss_weights = {
        "cop": 0.0,
        "grf": 0.0,
        "moments": 0.0,
        "contact": 0.0,
        "torque": 0.0,
        "grf_correction": 0.0,
        "output_reg": 0.0,
        "pos": 0.0,
        "vel": 0.0,
        "acc": 0.0,
        "qfrc_inverse": 0.0,
        "jacobian": 0.0,
        "rotation": 0.0,
        "full_id": 0.0,
    }

    def _physics_scalar(pred_probe_inner: jnp.ndarray) -> jnp.ndarray:
        _, _, aux = _compute_direct_loss(
            pred_probe_inner,
            probe_batch,
            normalizers,
            loss_weights=zero_loss_weights,
            deviation_learning=deviation_learning,
            cop_mask=cop_mask,
            use_contact_weighting=use_contact_weighting,
            contact_weight_multiplier=contact_weight_multiplier,
            physics_runner=physics_runner,
            kinematics_reconstructor=kinematics_reconstructor,
            physics_context=physics_context,
        )
        physics = aux["physics"]
        if not physics:
            raise RuntimeError("Differentiable MJX physics path was unavailable during the smoke test.")
        return (
            jnp.mean(jnp.square(physics["qfrc_inverse"]))
            + jnp.mean(jnp.square(physics["full_id"]))
            + 0.1 * jnp.mean(jnp.square(physics["jacp"]))
            + 0.1 * jnp.mean(jnp.square(physics["jacr"]))
            + 0.1 * jnp.mean(jnp.square(physics["rot_w_to_ga"]))
            + 0.1 * jnp.mean(jnp.square(physics["tau_grf"]))
            + 0.05 * jnp.mean(jnp.square(physics.get("ankle_heights", 0.0)))
            + 0.02 * jnp.mean(jnp.square(physics.get("cop_world", 0.0)))
            + 0.02 * jnp.mean(jnp.square(physics.get("full_moments", 0.0)))
        )

    grad_probe = jax.grad(_physics_scalar)(pred_probe)
    grad_parts = split_modq_predictions(grad_probe)
    stats = {name: _gradient_slice_stats(value) for name, value in grad_parts.items()}
    weak_channels = [
        name
        for name in ("pos", "vel", "acc")
        if stats[name]["finite"] < 0.5 or stats[name]["absmax"] <= 1e-12
    ]
    missing_required = [name for name in required_channels if name in weak_channels]
    if len(weak_channels) == 3 or missing_required:
        raise RuntimeError(
            "MJX gradient smoke test failed; no usable backprop signal reached "
            f"for required channel(s): {', '.join(missing_required or weak_channels)}. "
            f"Required={', '.join(required_channels)}; "
            "all stats="
            + ", ".join(
                f"{name}(finite={int(stats[name]['finite'])}, absmax={stats[name]['absmax']:.3e}, l2={stats[name]['l2']:.3e})"
                for name in ("pos", "vel", "acc")
            )
        )
    return stats


def _tree_all_finite(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    is_finite = jnp.asarray(True)
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        is_finite = jnp.logical_and(is_finite, jnp.all(jnp.isfinite(arr)))
    return is_finite


def _tree_nan_to_num(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(jnp.asarray(x), nan=0.0, posinf=0.0, neginf=0.0),
        tree,
    )


def split_qprime_from_prediction(pred_dict: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return pred_dict["qpos"], pred_dict["qvel"], pred_dict["qacc"]


def _build_modq_metrics(
    pred: jnp.ndarray,
    batch: Dict[str, Any],
    normalizers: Dict[str, Normalizer],
    *,
    loss_weights: Dict[str, float],
    adapter: ModQPhysicsAdapter,
    epoch: float,
    total_epochs: float,
    cop_mask: bool,
    use_contact_weighting: bool,
    contact_weight_multiplier: float,
    mag_on_off: bool,
    contact_on_off: bool,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    del epoch, total_epochs, mag_on_off, contact_on_off

    xml_path = _resolve_batch_xml_path(batch)
    physics_context: Dict[str, Any] = {}
    physics_runner: Optional[Any] = None
    kinematics_reconstructor: Optional[Any] = None
    if xml_path and adapter.available:
        physics_context = adapter.get_jit_context(xml_path)
        physics_runner = adapter.get_runner(xml_path)
        kinematics_reconstructor = adapter.get_reconstructor(xml_path)

    total_loss, metrics, aux = _compute_direct_loss(
        pred,
        batch,
        normalizers,
        loss_weights=loss_weights,
        deviation_learning=True,
        cop_mask=cop_mask,
        use_contact_weighting=use_contact_weighting,
        contact_weight_multiplier=contact_weight_multiplier,
        physics_runner=physics_runner,
        kinematics_reconstructor=kinematics_reconstructor,
        physics_context=physics_context,
        run_physics=True,
    )
    physics_metrics = {
        "physics_adapter_available": 1.0 if adapter.available else 0.0,
        "physics_available": float(np.asarray(metrics.get("physics_available", 0.0))),
    }
    return total_loss, metrics, {"qpos_phys": np.asarray(aux["qpos_phys"])}, physics_metrics


def create_train_state(rng, model, input_shape, static_shape, learning_rate=1e-4, weight_decay=1e-2):
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    dummy_static = jnp.ones(static_shape, dtype=jnp.float32)
    params = model.init(rng, dummy_input, dummy_static, train=False)["params"]
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate, weight_decay=weight_decay))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(
    *,
    normalizers: Dict[str, Normalizer],
    loss_weights: Dict[str, float],
    deviation_learning: bool,
    cop_mask: bool,
    use_contact_weighting: bool,
    contact_weight_multiplier: float,
    physics_runner: Optional[Any],
    kinematics_reconstructor: Optional[Any],
    run_physics: bool,
):
    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch, physics_context, dropout_rng):
        params_finite_pre = _tree_all_finite(state.params)

        def loss_fn(params):
            pred = state.apply_fn(
                {"params": params},
                batch["input"],
                batch["static_context"],
                train=True,
                rngs={"dropout": dropout_rng},
            )
            loss, metrics, aux = _compute_direct_loss(
                pred,
                batch,
                normalizers,
                loss_weights=loss_weights,
                deviation_learning=deviation_learning,
                cop_mask=cop_mask,
                use_contact_weighting=use_contact_weighting,
                contact_weight_multiplier=contact_weight_multiplier,
                physics_runner=physics_runner,
                kinematics_reconstructor=kinematics_reconstructor,
                physics_context=physics_context,
                run_physics=run_physics,
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        loss_finite = jnp.all(jnp.isfinite(loss))
        metrics_finite = _tree_all_finite(metrics)
        grads_finite = _tree_all_finite(grads)
        grad_global_norm = optax.global_norm(grads)
        safe_grads = _tree_nan_to_num(grads)
        proposed_state = state.apply_gradients(grads=safe_grads)
        state_finite_post = _tree_all_finite(proposed_state)
        should_apply = jnp.logical_and(
            jnp.logical_and(jnp.logical_and(loss_finite, metrics_finite), params_finite_pre),
            jnp.logical_and(grads_finite, state_finite_post),
        )
        state = jax.lax.cond(should_apply, lambda _: proposed_state, lambda _: state, operand=None)

        safe_loss = jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
        safe_metrics = _tree_nan_to_num(metrics)
        safe_metrics = dict(safe_metrics)
        safe_metrics["forward_finite"] = jnp.logical_and(loss_finite, metrics_finite).astype(jnp.float32)
        safe_metrics["grads_finite"] = grads_finite.astype(jnp.float32)
        safe_metrics["params_finite_pre"] = params_finite_pre.astype(jnp.float32)
        safe_metrics["state_finite_post"] = state_finite_post.astype(jnp.float32)
        safe_metrics["grad_global_norm"] = jnp.nan_to_num(grad_global_norm, nan=0.0, posinf=1e6, neginf=1e6)
        safe_metrics["update_skipped"] = (1.0 - should_apply.astype(jnp.float32))
        return state, safe_loss, safe_metrics

    return train_step


def make_eval_step(
    *,
    normalizers: Dict[str, Normalizer],
    loss_weights: Dict[str, float],
    deviation_learning: bool,
    cop_mask: bool,
    use_contact_weighting: bool,
    contact_weight_multiplier: float,
    physics_runner: Optional[Any],
    kinematics_reconstructor: Optional[Any],
    run_physics: bool,
):
    @jax.jit
    def eval_step(state, batch, physics_context):
        pred = state.apply_fn({"params": state.params}, batch["input"], batch["static_context"], train=False)
        loss, metrics, _ = _compute_direct_loss(
            pred,
            batch,
            normalizers,
            loss_weights=loss_weights,
            deviation_learning=deviation_learning,
            cop_mask=cop_mask,
            use_contact_weighting=use_contact_weighting,
            contact_weight_multiplier=contact_weight_multiplier,
            physics_runner=physics_runner,
            kinematics_reconstructor=kinematics_reconstructor,
            physics_context=physics_context,
            run_physics=run_physics,
        )
        loss_finite = jnp.all(jnp.isfinite(loss))
        metrics_finite = _tree_all_finite(metrics)
        safe_loss = jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
        safe_metrics = _tree_nan_to_num(metrics)
        safe_metrics = dict(safe_metrics)
        safe_metrics["forward_finite"] = jnp.logical_and(loss_finite, metrics_finite).astype(jnp.float32)
        safe_metrics["grads_finite"] = jnp.asarray(1.0, dtype=jnp.float32)
        safe_metrics["params_finite_pre"] = _tree_all_finite(state.params).astype(jnp.float32)
        safe_metrics["state_finite_post"] = jnp.asarray(1.0, dtype=jnp.float32)
        safe_metrics["grad_global_norm"] = jnp.asarray(0.0, dtype=jnp.float32)
        safe_metrics["update_skipped"] = jnp.asarray(0.0, dtype=jnp.float32)
        return safe_loss, safe_metrics

    return eval_step


def _prepare_batch(batch: Dict[str, Any], normalizers: Dict[str, Normalizer]) -> Dict[str, Any]:
    batch = normalize_batch(batch, normalizers)
    batch["input"] = jnp.asarray(batch["input"], dtype=jnp.float32)
    batch["static_context"] = jnp.asarray(batch["static_context"], dtype=jnp.float32)
    batch["cop"] = jnp.asarray(batch["cop"], dtype=jnp.float32)
    batch["grf"] = jnp.asarray(batch["grf"], dtype=jnp.float32)
    batch["moments"] = jnp.asarray(batch["moments"], dtype=jnp.float32)
    batch["contactBoolean"] = jnp.asarray(batch["contactBoolean"], dtype=jnp.float32)
    batch["pos_noised"] = jnp.asarray(batch["pos_noised"], dtype=jnp.float32)
    batch["vel_noised"] = jnp.asarray(batch["vel_noised"], dtype=jnp.float32)
    batch["acc_noised"] = jnp.asarray(batch["acc_noised"], dtype=jnp.float32)
    batch["pos_gt"] = jnp.asarray(batch["pos_gt"], dtype=jnp.float32)
    batch["vel_gt"] = jnp.asarray(batch["vel_gt"], dtype=jnp.float32)
    batch["acc_gt"] = jnp.asarray(batch["acc_gt"], dtype=jnp.float32)
    batch["qpos_mjx_input"] = jnp.asarray(batch["qpos_mjx_input"], dtype=jnp.float32)
    batch["qvel_mjx_input"] = jnp.asarray(batch["qvel_mjx_input"], dtype=jnp.float32)
    batch["qacc_mjx_input"] = jnp.asarray(batch["qacc_mjx_input"], dtype=jnp.float32)
    batch["qfrc_grf_contribution"] = jnp.asarray(batch["qfrc_grf_contribution"], dtype=jnp.float32)
    batch["qfrc_inverse_gt"] = jnp.asarray(batch["qfrc_inverse_gt"], dtype=jnp.float32)
    batch["id_gt_mjx"] = jnp.asarray(batch["id_gt_mjx"], dtype=jnp.float32)
    batch["jacp"] = jnp.asarray(batch["jacp"], dtype=jnp.float32)
    batch["jacr"] = jnp.asarray(batch["jacr"], dtype=jnp.float32)
    batch["gt_rot_w_to_ga"] = jnp.asarray(batch["gt_rot_w_to_ga"], dtype=jnp.float32)
    batch["ankle_heights"] = jnp.asarray(batch["ankle_heights"], dtype=jnp.float32)
    batch["com_accel"] = jnp.asarray(batch["com_accel"], dtype=jnp.float32)
    batch["height"] = jnp.asarray(batch["height"], dtype=jnp.float32)
    batch["mass"] = jnp.asarray(batch["mass"], dtype=jnp.float32)
    return batch


def _batch_for_jit(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in batch.items()
        if isinstance(value, (jnp.ndarray, np.ndarray))
    }


def _device_put_jit_batch(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {key: jax.device_put(value, device=device) for key, value in batch.items()}


def _make_prefetched_batch_item(
    raw_batch: Dict[str, Any],
    *,
    normalizers: Dict[str, Normalizer],
    device: Any,
) -> Dict[str, Any]:
    batch = _prepare_batch(raw_batch, normalizers)
    jit_batch = _device_put_jit_batch(_batch_for_jit(batch), device)
    batch_shape = tuple(int(x) for x in np.asarray(raw_batch["input"]).shape)
    return {
        "raw_batch": raw_batch,
        "jit_batch": jit_batch,
        "batch_shape": batch_shape,
    }


def _iter_prefetched_batches(
    raw_loader: Iterable[Dict[str, Any]],
    *,
    normalizers: Dict[str, Normalizer],
    device: Any,
    prefetch_batches: int,
) -> Iterable[Dict[str, Any]]:
    prefetch_count = max(0, int(prefetch_batches))
    if prefetch_count <= 0:
        for raw_batch in raw_loader:
            yield _make_prefetched_batch_item(raw_batch, normalizers=normalizers, device=device)
        return

    item_queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, prefetch_count))
    sentinel = object()
    worker_errors: deque[BaseException] = deque()

    def _worker() -> None:
        try:
            for raw_batch in raw_loader:
                item_queue.put(_make_prefetched_batch_item(raw_batch, normalizers=normalizers, device=device))
        except BaseException as exc:  # pragma: no cover - background error propagation
            worker_errors.append(exc)
        finally:
            item_queue.put(sentinel)

    worker = threading.Thread(target=_worker, name="modq-batch-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = item_queue.get()
            if item is sentinel:
                break
            yield item
        if worker_errors:
            raise worker_errors[0]
    finally:
        worker.join(timeout=1.0)


def _compute_full_external_moments(
    cop_phys: jnp.ndarray,
    grf_phys: jnp.ndarray,
    moments_phys: jnp.ndarray,
    ankle_heights: jnp.ndarray,
    rot_w_to_ga: jnp.ndarray,
) -> jnp.ndarray:
    cop_r_ga = jnp.concatenate([cop_phys[..., 0:1], ankle_heights[..., 0:1], cop_phys[..., 1:2]], axis=-1)
    cop_l_ga = jnp.concatenate([cop_phys[..., 2:3], ankle_heights[..., 1:2], cop_phys[..., 3:4]], axis=-1)
    rot_ga_to_w_r = jnp.swapaxes(rot_w_to_ga[:, :, 0], -1, -2)
    rot_ga_to_w_l = jnp.swapaxes(rot_w_to_ga[:, :, 1], -1, -2)
    cop_r = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_r, cop_r_ga)
    cop_l = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_l, cop_l_ga)
    grf_r = grf_phys[..., :3]
    grf_l = grf_phys[..., 3:6]
    mom_r = jnp.concatenate([jnp.zeros_like(moments_phys[..., 0:1]), jnp.zeros_like(moments_phys[..., 0:1]), moments_phys[..., 0:1]], axis=-1)
    mom_l = jnp.concatenate([jnp.zeros_like(moments_phys[..., 1:2]), jnp.zeros_like(moments_phys[..., 1:2]), moments_phys[..., 1:2]], axis=-1)
    return jnp.concatenate([jnp.cross(cop_r, grf_r) + mom_r, jnp.cross(cop_l, grf_l) + mom_l], axis=-1)


def _cop_ground_aligned_to_world(
    cop_phys: jnp.ndarray,
    ankle_heights: jnp.ndarray,
    rot_w_to_ga: jnp.ndarray,
    height_raw: jnp.ndarray,
) -> jnp.ndarray:
    cop_m = cop_phys
    cop_r_ga = jnp.concatenate([cop_m[..., 0:1], ankle_heights[..., 0:1], cop_m[..., 1:2]], axis=-1)
    cop_l_ga = jnp.concatenate([cop_m[..., 2:3], ankle_heights[..., 1:2], cop_m[..., 3:4]], axis=-1)
    rot_ga_to_w_r = jnp.swapaxes(rot_w_to_ga[:, :, 0], -1, -2)
    rot_ga_to_w_l = jnp.swapaxes(rot_w_to_ga[:, :, 1], -1, -2)
    cop_r = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_r, cop_r_ga)
    cop_l = jnp.einsum("bsij,bsj->bsi", rot_ga_to_w_l, cop_l_ga)
    return jnp.concatenate([cop_r, cop_l], axis=-1)


def _compute_tau_grf_from_predictions(
    grf_phys: jnp.ndarray,
    full_moments_phys: jnp.ndarray,
    jacp: jnp.ndarray,
    jacr: jnp.ndarray,
) -> jnp.ndarray:
    tau_p_r = jnp.einsum("bsij,bsi->bsj", jacp[:, :, 0], grf_phys[..., :3])
    tau_p_l = jnp.einsum("bsij,bsi->bsj", jacp[:, :, 1], grf_phys[..., 3:6])
    tau_r_r = jnp.einsum("bsij,bsi->bsj", jacr[:, :, 0], full_moments_phys[..., :3])
    tau_r_l = jnp.einsum("bsij,bsi->bsj", jacr[:, :, 1], full_moments_phys[..., 3:6])
    return tau_p_r + tau_p_l + tau_r_r + tau_r_l


def _compute_direct_loss(
    pred: jnp.ndarray,
    batch: Dict[str, Any],
    normalizers: Dict[str, Normalizer],
    *,
    loss_weights: Dict[str, float],
    deviation_learning: bool,
    cop_mask: bool,
    use_contact_weighting: bool,
    contact_weight_multiplier: float,
    physics_runner: Optional[Any] = None,
    kinematics_reconstructor: Optional[Any] = None,
    physics_context: Optional[Dict[str, Any]] = None,
    run_physics: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    pred_parts = split_modq_predictions(pred)
    cop_pred = pred_parts["cop"]
    grf_pred = pred_parts["grf"]
    moments_pred = pred_parts["moments"]
    contact_pred = pred_parts["contact"]
    pos_pred_raw = pred_parts["pos"]
    vel_pred_raw = pred_parts["vel"]
    acc_pred_raw = pred_parts["acc"]

    contact_bool = jnp.asarray(batch["contactBoolean"], dtype=pred.dtype)
    supervision_mask = jnp.asarray(batch.get("supervision_mask", np.ones(contact_bool.shape[:-1] + (1,), dtype=np.float32)), dtype=pred.dtype)
    if supervision_mask.ndim == 2:
        supervision_mask = supervision_mask[..., None]

    contact_r = contact_bool[..., 0:1]
    contact_l = contact_bool[..., 1:2]
    weight_r = 1.0 + (contact_weight_multiplier - 1.0) * contact_r if use_contact_weighting else jnp.ones_like(contact_r)
    weight_l = 1.0 + (contact_weight_multiplier - 1.0) * contact_l if use_contact_weighting else jnp.ones_like(contact_l)

    cop_weights = jnp.concatenate([jnp.tile(weight_r, (1, 1, 2)), jnp.tile(weight_l, (1, 1, 2))], axis=-1) * supervision_mask
    grf_weights = jnp.concatenate([jnp.tile(weight_r, (1, 1, 3)), jnp.tile(weight_l, (1, 1, 3))], axis=-1) * supervision_mask
    moments_weights = jnp.concatenate([weight_r, weight_l], axis=-1) * supervision_mask

    pos_noised_z = jnp.asarray(batch["pos_noised"], dtype=pred.dtype)
    vel_noised_z = jnp.asarray(batch["vel_noised"], dtype=pred.dtype)
    acc_noised_z = jnp.asarray(batch["acc_noised"], dtype=pred.dtype)
    if deviation_learning:
        pos_pred = pos_noised_z + pos_pred_raw
        vel_pred = vel_noised_z + vel_pred_raw
        acc_pred = acc_noised_z + acc_pred_raw
    else:
        pos_pred = pos_pred_raw
        vel_pred = vel_pred_raw
        acc_pred = acc_pred_raw

    cop_loss = mse_loss(cop_pred, batch["cop"], cop_weights) / 4.0
    grf_loss = mse_loss(grf_pred, batch["grf"], grf_weights) / 6.0
    moments_loss = mse_loss(moments_pred, batch["moments"], moments_weights) / 2.0
    contact_clipped = jnp.clip(contact_pred, 1e-7, 1.0 - 1e-7)
    contact_bce = -(contact_bool * jnp.log(contact_clipped) + (1.0 - contact_bool) * jnp.log(1.0 - contact_clipped))
    contact_loss = jnp.mean(contact_bce * supervision_mask) / 2.0
    pos_loss = mse_loss(pos_pred, batch["pos_gt"], supervision_mask)
    vel_loss = mse_loss(vel_pred, batch["vel_gt"], supervision_mask)
    acc_loss = mse_loss(acc_pred, batch["acc_gt"], supervision_mask)

    height_raw, mass_raw, body_weight_raw, norm_factor = _raw_scale_factors_from_batch(
        batch,
        dtype=pred.dtype,
    )
    gt_cop_phys, gt_grf_phys, _ = _restore_output_units(
        jnp.asarray(batch["cop"], dtype=pred.dtype),
        jnp.asarray(batch["grf"], dtype=pred.dtype),
        jnp.asarray(batch["moments"], dtype=pred.dtype),
        normalizers,
        height_raw=height_raw,
        body_weight_raw=body_weight_raw,
        bw_height_raw=norm_factor,
    )

    gt_pos_phys = normalizers["pos"].unnormalize(jnp.asarray(batch["pos_gt"], dtype=pred.dtype))
    gt_vel_phys = normalizers["vel"].unnormalize(jnp.asarray(batch["vel_gt"], dtype=pred.dtype))
    gt_acc_phys = normalizers["acc"].unnormalize(jnp.asarray(batch["acc_gt"], dtype=pred.dtype))
    pos_noised_phys = normalizers["pos"].unnormalize(pos_noised_z)
    vel_noised_phys = normalizers["vel"].unnormalize(vel_noised_z)
    acc_noised_phys = normalizers["acc"].unnormalize(acc_noised_z)
    ankle_heights_gt = jnp.asarray(batch["ankle_heights"], dtype=pred.dtype)
    rot_w_to_ga_gt = jnp.asarray(batch["gt_rot_w_to_ga"], dtype=pred.dtype)

    pos_phys = normalizers["pos"].unnormalize(pos_pred)
    vel_phys = normalizers["vel"].unnormalize(vel_pred)
    acc_phys = normalizers["acc"].unnormalize(acc_pred)
    pos_pred_rmse = _rmse(pos_phys - gt_pos_phys)
    pos_noised_rmse = _rmse(pos_noised_phys - gt_pos_phys)
    vel_pred_rmse = _rmse(vel_phys - gt_vel_phys)
    vel_noised_rmse = _rmse(vel_noised_phys - gt_vel_phys)
    acc_pred_rmse = _rmse(acc_phys - gt_acc_phys)
    acc_noised_rmse = _rmse(acc_noised_phys - gt_acc_phys)

    if not run_physics:
        zero = jnp.asarray(0.0, dtype=pred.dtype)
        pred_cop_phys, pred_grf_phys, _ = _restore_output_units(
            cop_pred,
            grf_pred,
            moments_pred,
            normalizers,
            height_raw=height_raw,
            body_weight_raw=body_weight_raw,
            bw_height_raw=norm_factor,
        )
        cop_rmse_m = _rmse(pred_cop_phys - gt_cop_phys)
        grf_rmse_n = _rmse(pred_grf_phys - gt_grf_phys)
        if deviation_learning:
            output_reg_loss = (
                jnp.mean(jnp.abs(pos_pred_raw) * supervision_mask)
                + jnp.mean(jnp.abs(vel_pred_raw) * supervision_mask)
                + jnp.mean(jnp.abs(acc_pred_raw) * supervision_mask)
            ) / 3.0
        else:
            output_reg_loss = zero
        total_loss = (
            _weighted_loss_term(cop_loss, loss_weights["cop"])
            + _weighted_loss_term(grf_loss, loss_weights["grf"])
            + _weighted_loss_term(moments_loss, loss_weights["moments"])
            + _weighted_loss_term(contact_loss, loss_weights["contact"])
            + _weighted_loss_term(zero, loss_weights["torque"])
            + _weighted_loss_term(zero, loss_weights["grf_correction"])
            + _weighted_loss_term(output_reg_loss, loss_weights["output_reg"])
            + _weighted_loss_term(pos_loss, loss_weights["pos"])
            + _weighted_loss_term(vel_loss, loss_weights["vel"])
            + _weighted_loss_term(acc_loss, loss_weights["acc"])
            + _weighted_loss_term(zero, loss_weights["qfrc_inverse"])
            + _weighted_loss_term(zero, loss_weights["jacobian"])
            + _weighted_loss_term(zero, loss_weights["rotation"])
            + _weighted_loss_term(zero, loss_weights["full_id"])
        )
        metrics = {
            "cop_loss": cop_loss,
            "grf_loss": grf_loss,
            "moments_loss": moments_loss,
            "contact_loss": contact_loss,
            "torque_loss": zero,
            "torque_rmse": zero,
            "torque_rmse_norm": zero,
            "tau_grf_rmse_nm": zero,
            "tau_grf_rmse_norm": zero,
            "torque_rel_rmse": zero,
            "torque_gt_rms": zero,
            "cop_rmse_m": cop_rmse_m,
            "grf_rmse_n": grf_rmse_n,
            "qfrc_inverse_rmse_nm": zero,
            "qfrc_inverse_noised_rmse_nm": zero,
            "full_id_rmse_nm": zero,
            "jacobian_torque_rmse_nm": zero,
            "jacobian_knee_r_rmse_nm": zero,
            "jacobian_knee_l_rmse_nm": zero,
            "jacobian_knee_rmse_nm": zero,
            "jacobian_ankle_r_rmse_nm": zero,
            "jacobian_ankle_l_rmse_nm": zero,
            "jacobian_ankle_rmse_nm": zero,
            "cop_world_gtrot_rmse_m": zero,
            "cop_world_gtrot_x_rmse_m": zero,
            "cop_world_gtrot_y_rmse_m": zero,
            "cop_world_gtrot_z_rmse_m": zero,
            "cop_world_predrot_rmse_m": zero,
            "cop_world_predrot_x_rmse_m": zero,
            "cop_world_predrot_y_rmse_m": zero,
            "cop_world_predrot_z_rmse_m": zero,
            "rotation_matrix_rmse": zero,
            "rotation_matrix_noised_rmse": zero,
            "rotation_geodesic_rmse_deg": zero,
            "rotation_geodesic_noised_rmse_deg": zero,
            "rot_wrench_rmse_nm": zero,
            "rot_wrench_r_rmse_nm": zero,
            "rot_wrench_l_rmse_nm": zero,
            "grf_correction_loss": zero,
            "output_reg_loss": output_reg_loss,
            "pos_loss": pos_loss,
            "vel_loss": vel_loss,
            "acc_loss": acc_loss,
            "pos_pred_rmse": pos_pred_rmse,
            "pos_noised_rmse": pos_noised_rmse,
            "vel_pred_rmse": vel_pred_rmse,
            "vel_noised_rmse": vel_noised_rmse,
            "acc_pred_rmse": acc_pred_rmse,
            "acc_noised_rmse": acc_noised_rmse,
            "qfrc_inverse_loss": zero,
            "jacobian_loss": zero,
            "rotation_loss": zero,
            "full_id_loss": zero,
            "physics_available": zero,
            "total_loss": total_loss,
        }
        return total_loss, metrics, {"physics": {}}

    contact_soft = jnp.clip(contact_pred, 0.0, 1.0) if cop_mask else None
    cop_phys, grf_phys, moments_phys = _restore_output_units(
        cop_pred,
        grf_pred,
        moments_pred,
        normalizers,
        height_raw=height_raw,
        body_weight_raw=body_weight_raw,
        bw_height_raw=norm_factor,
        contact_scale=contact_soft,
    )

    if kinematics_reconstructor is not None and physics_context is not None:
        flat_frames = pos_phys.shape[0] * pos_phys.shape[1]
        qpos_flat, qvel_flat, qacc_flat = kinematics_reconstructor(
            pos_phys.reshape((flat_frames, pos_phys.shape[-1])),
            vel_phys.reshape((flat_frames, vel_phys.shape[-1])),
            acc_phys.reshape((flat_frames, acc_phys.shape[-1])),
            jnp.asarray(batch["qpos_mjx_input"], dtype=pred.dtype).reshape((flat_frames, batch["qpos_mjx_input"].shape[-1])),
            jnp.asarray(batch["qvel_mjx_input"], dtype=pred.dtype).reshape((flat_frames, batch["qvel_mjx_input"].shape[-1])),
            jnp.asarray(batch["qacc_mjx_input"], dtype=pred.dtype).reshape((flat_frames, batch["qacc_mjx_input"].shape[-1])),
            jnp.asarray(physics_context["slave_idx"], dtype=jnp.int32),
            jnp.asarray(physics_context["master_idx"], dtype=jnp.int32),
            jnp.asarray(physics_context["coeffs"], dtype=jnp.float32),
        )
        qpos_phys = qpos_flat.reshape(pos_phys.shape[:2] + (qpos_flat.shape[-1],))
        qvel_phys = qvel_flat.reshape(vel_phys.shape[:2] + (qvel_flat.shape[-1],))
        qacc_phys = qacc_flat.reshape(acc_phys.shape[:2] + (qacc_flat.shape[-1],))
    else:
        qpos_phys = jnp.asarray(batch["qpos_mjx_input"], dtype=pred.dtype)
        qvel_phys = jnp.asarray(batch["qvel_mjx_input"], dtype=pred.dtype)
        qacc_phys = jnp.asarray(batch["qacc_mjx_input"], dtype=pred.dtype)

    qfrc_inverse_loss = jnp.asarray(0.0, dtype=pred.dtype)
    jacobian_loss = jnp.asarray(0.0, dtype=pred.dtype)
    rotation_loss = jnp.asarray(0.0, dtype=pred.dtype)
    full_id_loss = jnp.asarray(0.0, dtype=pred.dtype)

    if physics_runner is not None and kinematics_reconstructor is not None and physics_context is not None:
        flat_frames = qpos_phys.shape[0] * qpos_phys.shape[1]
        physics_flat = physics_runner(
            physics_context["mjx_model"],
            jnp.asarray(physics_context["calcn_r_id"], dtype=jnp.int32),
            jnp.asarray(physics_context["calcn_l_id"], dtype=jnp.int32),
            qpos_phys.reshape((flat_frames, qpos_phys.shape[-1])),
            qvel_phys.reshape((flat_frames, qvel_phys.shape[-1])),
            qacc_phys.reshape((flat_frames, qacc_phys.shape[-1])),
            cop_phys.reshape((flat_frames, cop_phys.shape[-1])),
            grf_phys.reshape((flat_frames, grf_phys.shape[-1])),
            moments_phys.reshape((flat_frames, moments_phys.shape[-1])),
            jnp.asarray(batch["ankle_heights"], dtype=pred.dtype).reshape((flat_frames, batch["ankle_heights"].shape[-1])),
        )

        def _reshape_physics(x: jnp.ndarray) -> jnp.ndarray:
            return x.reshape(qpos_phys.shape[:2] + x.shape[1:])

        physics = {key: _reshape_physics(value) for key, value in physics_flat.items()}
        full_moments_phys = physics["full_moments"]
        tau_grf_pred = physics["tau_grf"]
        rot_w_to_ga_pred_phys = physics["rot_w_to_ga"]
        rot_w_to_ga_gt_phys = rot_w_to_ga_gt
        physics_term_losses = _compute_physics_term_losses(
            physics,
            batch,
            norm_factor=norm_factor,
            rot_w_to_ga_gt=rot_w_to_ga_gt_phys,
            supervision_mask=supervision_mask,
            dtype=pred.dtype,
        )
        qfrc_inverse_loss = physics_term_losses["qfrc_inverse"]
        jacobian_loss = physics_term_losses["jacobian"]
        rotation_loss = physics_term_losses["rotation"]
        rotation_geodesic_rmse_deg = rotation_geodesic_summary_deg(
            rot_w_to_ga_pred_phys,
            rot_w_to_ga_gt_phys,
            supervision_mask,
            xp=jnp,
            project=False,
        )["overall_rmse_deg"]
        full_id_loss = physics_term_losses["full_id"]
        physics_available = jnp.asarray(1.0, dtype=pred.dtype)
    else:
        full_moments_phys = _compute_full_external_moments(
            cop_phys,
            grf_phys,
            moments_phys,
            jnp.asarray(batch["ankle_heights"], dtype=pred.dtype),
            jnp.asarray(batch["rot_w_to_ga"], dtype=pred.dtype),
        )
        tau_grf_pred = _compute_tau_grf_from_predictions(
            grf_phys,
            full_moments_phys,
            jnp.asarray(batch["jacp"], dtype=pred.dtype),
            jnp.asarray(batch["jacr"], dtype=pred.dtype),
        )
        physics = {}
        physics_available = jnp.asarray(0.0, dtype=pred.dtype)

    torque_target = jnp.asarray(batch["qfrc_grf_contribution"], dtype=pred.dtype)
    torque_error = tau_grf_pred - torque_target
    torque_loss = jnp.mean(jnp.square(torque_error / norm_factor[..., None]))
    torque_rmse = jnp.sqrt(jnp.mean(jnp.square(torque_error)))
    torque_rmse_norm = jnp.sqrt(jnp.mean(jnp.square(torque_error / norm_factor[..., None])))
    torque_gt_rms = jnp.sqrt(jnp.mean(jnp.square(torque_target)))
    torque_rel_rmse = torque_rmse / jnp.maximum(torque_gt_rms, jnp.asarray(1e-6, dtype=pred.dtype))
    cop_rmse_m = _rmse(cop_phys - gt_cop_phys)
    grf_rmse_n = _rmse(grf_phys - gt_grf_phys)
    qfrc_inverse_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    qfrc_inverse_noised_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    full_id_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    jacobian_torque_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    jacobian_knee_r_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    jacobian_knee_l_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    jacobian_knee_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    jacobian_ankle_r_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    jacobian_ankle_l_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    jacobian_ankle_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    cop_world_gtrot_rmse_m = jnp.asarray(0.0, dtype=pred.dtype)
    cop_world_gtrot_x_rmse_m = jnp.asarray(0.0, dtype=pred.dtype)
    cop_world_gtrot_y_rmse_m = jnp.asarray(0.0, dtype=pred.dtype)
    cop_world_gtrot_z_rmse_m = jnp.asarray(0.0, dtype=pred.dtype)
    cop_world_predrot_rmse_m = jnp.asarray(0.0, dtype=pred.dtype)
    cop_world_predrot_x_rmse_m = jnp.asarray(0.0, dtype=pred.dtype)
    cop_world_predrot_y_rmse_m = jnp.asarray(0.0, dtype=pred.dtype)
    cop_world_predrot_z_rmse_m = jnp.asarray(0.0, dtype=pred.dtype)
    rotation_matrix_rmse = jnp.asarray(0.0, dtype=pred.dtype)
    rotation_matrix_noised_rmse = jnp.asarray(0.0, dtype=pred.dtype)
    rotation_geodesic_rmse_deg = jnp.asarray(0.0, dtype=pred.dtype)
    rotation_geodesic_noised_rmse_deg = jnp.asarray(0.0, dtype=pred.dtype)
    rot_wrench_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    rot_wrench_r_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    rot_wrench_l_rmse_nm = jnp.asarray(0.0, dtype=pred.dtype)
    if physics:
        qfrc_inverse_error = physics["qfrc_inverse"] - jnp.asarray(batch["qfrc_inverse_gt"], dtype=pred.dtype)
        qfrc_inverse_rmse_nm = _rmse(qfrc_inverse_error)
        full_id_rmse_nm = _rmse(physics["full_id"] - jnp.asarray(batch["id_gt_mjx"], dtype=pred.dtype))
        rotation_matrix_rmse = _rmse(physics["rot_w_to_ga"] - rot_w_to_ga_gt)
        tau_grf_gt_jac = _compute_tau_grf_from_predictions(
            grf_phys,
            full_moments_phys,
            jnp.asarray(batch["jacp"], dtype=pred.dtype),
            jnp.asarray(batch["jacr"], dtype=pred.dtype),
        )
        jacobian_torque_error = tau_grf_pred - tau_grf_gt_jac
        jacobian_torque_rmse_nm = _rmse(jacobian_torque_error)
        jacobian_knee_r_rmse_nm = _indexed_rmse(jacobian_torque_error, [11])
        jacobian_knee_l_rmse_nm = _indexed_rmse(jacobian_torque_error, [22])
        jacobian_knee_rmse_nm = _indexed_rmse(jacobian_torque_error, [11, 22])
        jacobian_ankle_r_rmse_nm = _indexed_rmse(jacobian_torque_error, [14])
        jacobian_ankle_l_rmse_nm = _indexed_rmse(jacobian_torque_error, [25])
        jacobian_ankle_rmse_nm = _indexed_rmse(jacobian_torque_error, [14, 25])

        flat_frames = qpos_phys.shape[0] * qpos_phys.shape[1]
        noised_physics_flat = physics_runner(
            physics_context["mjx_model"],
            jnp.asarray(physics_context["calcn_r_id"], dtype=jnp.int32),
            jnp.asarray(physics_context["calcn_l_id"], dtype=jnp.int32),
            jnp.asarray(batch["qpos_mjx_input"], dtype=pred.dtype).reshape((flat_frames, batch["qpos_mjx_input"].shape[-1])),
            jnp.asarray(batch["qvel_mjx_input"], dtype=pred.dtype).reshape((flat_frames, batch["qvel_mjx_input"].shape[-1])),
            jnp.asarray(batch["qacc_mjx_input"], dtype=pred.dtype).reshape((flat_frames, batch["qacc_mjx_input"].shape[-1])),
            cop_phys.reshape((flat_frames, cop_phys.shape[-1])),
            grf_phys.reshape((flat_frames, grf_phys.shape[-1])),
            moments_phys.reshape((flat_frames, moments_phys.shape[-1])),
            ankle_heights_gt.reshape((flat_frames, ankle_heights_gt.shape[-1])),
        )
        noised_physics = {key: _reshape_physics(value) for key, value in noised_physics_flat.items()}
        qfrc_inverse_noised_error = noised_physics["qfrc_inverse"] - jnp.asarray(batch["qfrc_inverse_gt"], dtype=pred.dtype)
        qfrc_inverse_noised_rmse_nm = _rmse(qfrc_inverse_noised_error)
        rotation_matrix_noised_rmse = _rmse(noised_physics["rot_w_to_ga"] - rot_w_to_ga_gt)
        rot_w_to_ga_noised_phys = noised_physics["rot_w_to_ga"]
        rotation_geodesic_noised_rmse_deg = rotation_geodesic_summary_deg(
            rot_w_to_ga_noised_phys,
            rot_w_to_ga_gt_phys,
            supervision_mask,
            xp=jnp,
            project=False,
        )["overall_rmse_deg"]

        gt_cop_world = _cop_ground_aligned_to_world(
            gt_cop_phys,
            ankle_heights_gt,
            rot_w_to_ga_gt,
            height_raw,
        )
        pred_cop_world_gtrot = _cop_ground_aligned_to_world(
            cop_phys,
            ankle_heights_gt,
            rot_w_to_ga_gt,
            height_raw,
        )
        pred_cop_world_predrot = _cop_ground_aligned_to_world(
            cop_phys,
            ankle_heights_gt,
            physics["rot_w_to_ga"],
            height_raw,
        )
        cop_world_gtrot_error = pred_cop_world_gtrot - gt_cop_world
        cop_world_predrot_error = pred_cop_world_predrot - gt_cop_world
        cop_world_gtrot_rmse_m = _rmse(cop_world_gtrot_error)
        cop_world_gtrot_x_rmse_m = _indexed_rmse(cop_world_gtrot_error, [0, 3])
        cop_world_gtrot_y_rmse_m = _indexed_rmse(cop_world_gtrot_error, [1, 4])
        cop_world_gtrot_z_rmse_m = _indexed_rmse(cop_world_gtrot_error, [2, 5])
        cop_world_predrot_rmse_m = _rmse(cop_world_predrot_error)
        cop_world_predrot_x_rmse_m = _indexed_rmse(cop_world_predrot_error, [0, 3])
        cop_world_predrot_y_rmse_m = _indexed_rmse(cop_world_predrot_error, [1, 4])
        cop_world_predrot_z_rmse_m = _indexed_rmse(cop_world_predrot_error, [2, 5])

        # Isolate the COP rotation effect in physical units (m, N, Nm) by holding
        # COP/GRF/GRM and ankle height fixed while swapping only rot_w_to_ga.
        cop_phys_m = cop_phys
        grf_phys_n = grf_phys
        moments_phys_nm = moments_phys
        full_moments_gt_rot_nm = _compute_full_external_moments(
            cop_phys_m,
            grf_phys_n,
            moments_phys_nm,
            ankle_heights_gt,
            rot_w_to_ga_gt,
        )
        full_moments_pred_rot_nm = _compute_full_external_moments(
            cop_phys_m,
            grf_phys_n,
            moments_phys_nm,
            ankle_heights_gt,
            physics["rot_w_to_ga"],
        )
        rot_wrench_error_nm = full_moments_pred_rot_nm - full_moments_gt_rot_nm
        rot_wrench_rmse_nm = _rmse(rot_wrench_error_nm)
        rot_wrench_r_rmse_nm = _rmse(rot_wrench_error_nm[..., :3])
        rot_wrench_l_rmse_nm = _rmse(rot_wrench_error_nm[..., 3:6])

    com_accel = jnp.asarray(batch["com_accel"], dtype=pred.dtype)
    pred_fx = (grf_phys[..., 0] + grf_phys[..., 3])
    pred_fy = grf_phys[..., 1] + grf_phys[..., 4]
    pred_fz = grf_phys[..., 2] + grf_phys[..., 5]
    res_x = mass_raw * com_accel[..., 0] - pred_fx
    res_y = mass_raw * com_accel[..., 1] - pred_fy
    res_z = mass_raw * (com_accel[..., 2] + 9.8067) - pred_fz
    grf_res = jnp.stack([res_x, res_y, res_z], axis=-1)
    grf_correction_loss = jnp.mean(jnp.square(normalizers["grf_res"].normalize(grf_res)) * supervision_mask) / 3.0

    if deviation_learning:
        output_reg_loss = (
            jnp.mean(jnp.abs(pos_pred_raw) * supervision_mask)
            + jnp.mean(jnp.abs(vel_pred_raw) * supervision_mask)
            + jnp.mean(jnp.abs(acc_pred_raw) * supervision_mask)
        ) / 3.0
    else:
        output_reg_loss = jnp.asarray(0.0, dtype=pred.dtype)
    total_loss = (
        _weighted_loss_term(cop_loss, loss_weights["cop"])
        + _weighted_loss_term(grf_loss, loss_weights["grf"])
        + _weighted_loss_term(moments_loss, loss_weights["moments"])
        + _weighted_loss_term(contact_loss, loss_weights["contact"])
        + _weighted_loss_term(torque_loss, loss_weights["torque"])
        + _weighted_loss_term(grf_correction_loss, loss_weights["grf_correction"])
        + _weighted_loss_term(output_reg_loss, loss_weights["output_reg"])
        + _weighted_loss_term(pos_loss, loss_weights["pos"])
        + _weighted_loss_term(vel_loss, loss_weights["vel"])
        + _weighted_loss_term(acc_loss, loss_weights["acc"])
        + _weighted_loss_term(qfrc_inverse_loss, loss_weights["qfrc_inverse"])
        + _weighted_loss_term(jacobian_loss, loss_weights["jacobian"])
        + _weighted_loss_term(rotation_loss, loss_weights["rotation"])
        + _weighted_loss_term(full_id_loss, loss_weights["full_id"])
    )

    metrics = {
        "cop_loss": cop_loss,
        "grf_loss": grf_loss,
        "moments_loss": moments_loss,
        "contact_loss": contact_loss,
        "torque_loss": torque_loss,
        "torque_rmse": torque_rmse,
        "torque_rmse_norm": torque_rmse_norm,
        "tau_grf_rmse_nm": torque_rmse,
        "tau_grf_rmse_norm": torque_rmse_norm,
        "torque_rel_rmse": torque_rel_rmse,
        "torque_gt_rms": torque_gt_rms,
        "cop_rmse_m": cop_rmse_m,
        "grf_rmse_n": grf_rmse_n,
        "qfrc_inverse_rmse_nm": qfrc_inverse_rmse_nm,
        "qfrc_inverse_noised_rmse_nm": qfrc_inverse_noised_rmse_nm,
        "full_id_rmse_nm": full_id_rmse_nm,
        "jacobian_torque_rmse_nm": jacobian_torque_rmse_nm,
        "jacobian_knee_r_rmse_nm": jacobian_knee_r_rmse_nm,
        "jacobian_knee_l_rmse_nm": jacobian_knee_l_rmse_nm,
        "jacobian_knee_rmse_nm": jacobian_knee_rmse_nm,
        "jacobian_ankle_r_rmse_nm": jacobian_ankle_r_rmse_nm,
        "jacobian_ankle_l_rmse_nm": jacobian_ankle_l_rmse_nm,
        "jacobian_ankle_rmse_nm": jacobian_ankle_rmse_nm,
        "cop_world_gtrot_rmse_m": cop_world_gtrot_rmse_m,
        "cop_world_gtrot_x_rmse_m": cop_world_gtrot_x_rmse_m,
        "cop_world_gtrot_y_rmse_m": cop_world_gtrot_y_rmse_m,
        "cop_world_gtrot_z_rmse_m": cop_world_gtrot_z_rmse_m,
        "cop_world_predrot_rmse_m": cop_world_predrot_rmse_m,
        "cop_world_predrot_x_rmse_m": cop_world_predrot_x_rmse_m,
        "cop_world_predrot_y_rmse_m": cop_world_predrot_y_rmse_m,
        "cop_world_predrot_z_rmse_m": cop_world_predrot_z_rmse_m,
        "rotation_matrix_rmse": rotation_matrix_rmse,
        "rotation_matrix_noised_rmse": rotation_matrix_noised_rmse,
        "rotation_geodesic_rmse_deg": rotation_geodesic_rmse_deg,
        "rotation_geodesic_noised_rmse_deg": rotation_geodesic_noised_rmse_deg,
        "rot_wrench_rmse_nm": rot_wrench_rmse_nm,
        "rot_wrench_r_rmse_nm": rot_wrench_r_rmse_nm,
        "rot_wrench_l_rmse_nm": rot_wrench_l_rmse_nm,
        "grf_correction_loss": grf_correction_loss,
        "output_reg_loss": output_reg_loss,
        "pos_loss": pos_loss,
        "vel_loss": vel_loss,
        "acc_loss": acc_loss,
        "pos_pred_rmse": pos_pred_rmse,
        "pos_noised_rmse": pos_noised_rmse,
        "vel_pred_rmse": vel_pred_rmse,
        "vel_noised_rmse": vel_noised_rmse,
        "acc_pred_rmse": acc_pred_rmse,
        "acc_noised_rmse": acc_noised_rmse,
        "qfrc_inverse_loss": qfrc_inverse_loss,
        "jacobian_loss": jacobian_loss,
        "rotation_loss": rotation_loss,
        "full_id_loss": full_id_loss,
        "physics_available": physics_available,
        "total_loss": total_loss,
    }
    aux = {
        "cop_phys": cop_phys,
        "grf_phys": grf_phys,
        "moments_phys": moments_phys,
        "pos_phys": pos_phys,
        "vel_phys": vel_phys,
        "acc_phys": acc_phys,
        "qpos_phys": qpos_phys,
        "qvel_phys": qvel_phys,
        "qacc_phys": qacc_phys,
        "full_moments_phys": full_moments_phys,
        "tau_grf_pred": tau_grf_pred,
        "norm_factor": norm_factor,
        "physics": physics,
    }
    return total_loss, metrics, aux


def make_kinematic_equiv_probe_step(
    *,
    normalizers: Dict[str, Normalizer],
    loss_weights: Dict[str, float],
    deviation_learning: bool,
    cop_mask: bool,
    physics_runner: Optional[Any],
    kinematics_reconstructor: Optional[Any],
    run_physics: bool,
    train_mode: bool,
):
    if train_mode:
        @jax.jit
        def probe_step(state, batch, physics_context, dropout_rng):
            pred = state.apply_fn(
                {"params": state.params},
                batch["input"],
                batch["static_context"],
                train=True,
                rngs={"dropout": dropout_rng},
            )
            pred_parts = split_modq_predictions(pred)
            pos_noised_z = jnp.asarray(batch["pos_noised"], dtype=pred.dtype)
            vel_noised_z = jnp.asarray(batch["vel_noised"], dtype=pred.dtype)
            acc_noised_z = jnp.asarray(batch["acc_noised"], dtype=pred.dtype)
            if deviation_learning:
                pos_pred = pos_noised_z + pred_parts["pos"]
                vel_pred = vel_noised_z + pred_parts["vel"]
                acc_pred = acc_noised_z + pred_parts["acc"]
            else:
                pos_pred = pred_parts["pos"]
                vel_pred = pred_parts["vel"]
                acc_pred = pred_parts["acc"]
            metrics = _compute_kinematic_equivalent_metrics(
                batch=batch,
                normalizers=normalizers,
                loss_weights=loss_weights,
                cop_pred=pred_parts["cop"],
                grf_pred=pred_parts["grf"],
                moments_pred=pred_parts["moments"],
                contact_pred=pred_parts["contact"],
                pos_pred=pos_pred,
                vel_pred=vel_pred,
                acc_pred=acc_pred,
                cop_mask=cop_mask,
                physics_runner=physics_runner,
                kinematics_reconstructor=kinematics_reconstructor,
                physics_context=physics_context,
                run_physics=run_physics,
            )
            return _tree_nan_to_num(metrics)
    else:
        @jax.jit
        def probe_step(state, batch, physics_context):
            pred = state.apply_fn({"params": state.params}, batch["input"], batch["static_context"], train=False)
            pred_parts = split_modq_predictions(pred)
            pos_noised_z = jnp.asarray(batch["pos_noised"], dtype=pred.dtype)
            vel_noised_z = jnp.asarray(batch["vel_noised"], dtype=pred.dtype)
            acc_noised_z = jnp.asarray(batch["acc_noised"], dtype=pred.dtype)
            if deviation_learning:
                pos_pred = pos_noised_z + pred_parts["pos"]
                vel_pred = vel_noised_z + pred_parts["vel"]
                acc_pred = acc_noised_z + pred_parts["acc"]
            else:
                pos_pred = pred_parts["pos"]
                vel_pred = pred_parts["vel"]
                acc_pred = pred_parts["acc"]
            metrics = _compute_kinematic_equivalent_metrics(
                batch=batch,
                normalizers=normalizers,
                loss_weights=loss_weights,
                cop_pred=pred_parts["cop"],
                grf_pred=pred_parts["grf"],
                moments_pred=pred_parts["moments"],
                contact_pred=pred_parts["contact"],
                pos_pred=pos_pred,
                vel_pred=vel_pred,
                acc_pred=acc_pred,
                cop_mask=cop_mask,
                physics_runner=physics_runner,
                kinematics_reconstructor=kinematics_reconstructor,
                physics_context=physics_context,
                run_physics=run_physics,
            )
            return _tree_nan_to_num(metrics)

    return probe_step


def _compute_detached_physics_losses(
    aux: Dict[str, jnp.ndarray],
    batch: Dict[str, Any],
    adapter: ModQPhysicsAdapter,
) -> Tuple[float, Dict[str, float]]:
    if not adapter.available:
        return 0.0, {
            "physics_adapter_available": 0.0,
            "qfrc_inverse_loss_detached": 0.0,
            "jacobian_loss_detached": 0.0,
            "rotation_loss_detached": 0.0,
            "full_id_loss_detached": 0.0,
        }

    qpos_phys = np.asarray(aux["qpos_phys"], dtype=np.float32)
    qvel_phys = np.asarray(aux["qvel_phys"], dtype=np.float32)
    qacc_phys = np.asarray(aux["qacc_phys"], dtype=np.float32)
    cop_phys = np.asarray(aux["cop_phys"], dtype=np.float32)
    grf_phys = np.asarray(aux["grf_phys"], dtype=np.float32)
    moments_phys = np.asarray(aux["moments_phys"], dtype=np.float32)
    norm_factor = np.asarray(aux["norm_factor"], dtype=np.float32)
    if norm_factor.ndim == 2:
        norm_factor = norm_factor[..., None]
    total = 0.0
    metrics = {
        "physics_adapter_available": 1.0,
        "qfrc_inverse_loss_detached": 0.0,
        "jacobian_loss_detached": 0.0,
        "rotation_loss_detached": 0.0,
        "full_id_loss_detached": 0.0,
    }
    sample_count = 0
    for sample_idx in range(qpos_phys.shape[0]):
        xml_path = batch["subject_model_xml"][sample_idx] if isinstance(batch["subject_model_xml"], list) else batch["subject_model_xml"]
        physics = adapter.evaluate(
            qpos_phys[sample_idx],
            qvel_phys[sample_idx],
            qacc_phys[sample_idx],
            cop_phys[sample_idx],
            grf_phys[sample_idx],
            moments_phys[sample_idx],
            np.asarray(batch["ankle_heights"][sample_idx]),
            str(xml_path),
        )
        if physics is None:
            continue
        sample_count += 1
        qfrc_inverse_loss = float(np.mean(np.square((physics["qfrc_inverse"] - np.asarray(batch["qfrc_inverse_gt"][sample_idx])) / norm_factor[sample_idx])))
        jacobian_loss = float(
            np.mean(np.square(physics["jacp"] - np.asarray(batch["jacp"][sample_idx])))
            + np.mean(np.square(physics["jacr"] - np.asarray(batch["jacr"][sample_idx])))
        )
        rotation_loss = float(np.mean(np.square(physics["rot_w_to_ga"] - np.asarray(batch["gt_rot_w_to_ga"][sample_idx]))))
        full_id_loss = float(np.mean(np.square((physics["full_id"] - np.asarray(batch["id_gt_mjx"][sample_idx])) / norm_factor[sample_idx])))
        total += qfrc_inverse_loss + jacobian_loss + rotation_loss + full_id_loss
        metrics["qfrc_inverse_loss_detached"] += qfrc_inverse_loss
        metrics["jacobian_loss_detached"] += jacobian_loss
        metrics["rotation_loss_detached"] += rotation_loss
        metrics["full_id_loss_detached"] += full_id_loss

    if sample_count > 0:
        for key in list(metrics.keys()):
            if key != "physics_adapter_available":
                metrics[key] /= float(sample_count)
        total /= float(sample_count)
    return total, metrics


def _make_loss_weights(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "cop": float(args.cop_weight),
        "grf": float(args.grf_weight),
        "moments": float(args.moments_weight),
        "contact": float(args.contact_weight),
        "torque": float(args.torque_weight),
        "grf_correction": float(args.grf_correction_weight),
        "output_reg": 0.0,
        "pos": float(args.qpos_weight),
        "vel": float(args.qvel_weight),
        "acc": float(args.qacc_weight),
        "qfrc_inverse": float(args.qfrc_inverse_weight),
        "jacobian": float(args.jacobian_weight),
        "rotation": float(args.rotation_weight),
        "full_id": float(args.full_id_weight),
    }


LOSS_REPORT_ORDER: List[Tuple[str, Optional[str], str]] = [
    ("cop_loss", "cop", "cop"),
    ("grf_loss", "grf", "grf"),
    ("moments_loss", "moments", "mom"),
    ("contact_loss", "contact", "contact"),
    ("torque_loss", "torque", "torque"),
    ("grf_correction_loss", "grf_correction", "grf_corr"),
    ("output_reg_loss", "output_reg", "out_reg"),
    ("pos_loss", "pos", "pos"),
    ("vel_loss", "vel", "vel"),
    ("acc_loss", "acc", "acc"),
    ("qfrc_inverse_loss", "qfrc_inverse", "qfrc_inv"),
    ("jacobian_loss", "jacobian", "jac"),
    ("rotation_loss", "rotation", "rot"),
    ("full_id_loss", "full_id", "full_id"),
    ("physics_available", None, "phys_avail"),
]

TORQUE_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("tau_grf_rmse_nm", "tau_grf_nm"),
    ("tau_grf_rmse_norm", "tau_grf_norm"),
    ("full_id_rmse_nm", "full_id_nm"),
    ("torque_rel_rmse", "tau_grf_rel"),
    ("torque_gt_rms", "tau_grf_gt_rms"),
]

SIGNAL_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("cop_rmse_m", "cop_rmse_m"),
    ("grf_rmse_n", "grf_rmse_n"),
]

COP_WORLD_GTROT_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("cop_world_gtrot_rmse_m", "gtrot_m"),
    ("cop_world_gtrot_x_rmse_m", "gtrot_x_m"),
    ("cop_world_gtrot_y_rmse_m", "gtrot_y_m"),
    ("cop_world_gtrot_z_rmse_m", "gtrot_z_m"),
]

COP_WORLD_PREDROT_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("cop_world_predrot_rmse_m", "predrot_m"),
    ("cop_world_predrot_x_rmse_m", "predrot_x_m"),
    ("cop_world_predrot_y_rmse_m", "predrot_y_m"),
    ("cop_world_predrot_z_rmse_m", "predrot_z_m"),
]

BASELINE_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("qfrc_inverse_rmse_nm", "qfrc_pred_nm"),
    ("qfrc_inverse_noised_rmse_nm", "qfrc_noised_nm"),
    ("rotation_matrix_rmse", "rot_pred"),
    ("rotation_matrix_noised_rmse", "rot_noised"),
]

KINEMATICS_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("pos_pred_rmse", "pos_pred"),
    ("pos_noised_rmse", "pos_noised"),
    ("vel_pred_rmse", "vel_pred"),
    ("vel_noised_rmse", "vel_noised"),
    ("acc_pred_rmse", "acc_pred"),
    ("acc_noised_rmse", "acc_noised"),
]

JACOBIAN_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("jacobian_torque_rmse_nm", "jac_tau_rmse_nm"),
    ("jacobian_knee_r_rmse_nm", "jac_knee_r_nm"),
    ("jacobian_knee_l_rmse_nm", "jac_knee_l_nm"),
    ("jacobian_knee_rmse_nm", "jac_knees_nm"),
    ("jacobian_ankle_r_rmse_nm", "jac_ankle_r_nm"),
    ("jacobian_ankle_l_rmse_nm", "jac_ankle_l_nm"),
    ("jacobian_ankle_rmse_nm", "jac_ankles_nm"),
]

ROTATION_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("rot_wrench_rmse_nm", "rot_wrench_nm"),
    ("rot_wrench_r_rmse_nm", "rot_wrench_r_nm"),
    ("rot_wrench_l_rmse_nm", "rot_wrench_l_nm"),
]

ROTATION_GEODESIC_DEBUG_REPORT_ORDER: List[Tuple[str, str]] = [
    ("rotation_geodesic_rmse_deg", "geo_deg"),
    ("rotation_geodesic_noised_rmse_deg", "geo_noised_deg"),
]

KINEMATIC_EQUIV_LOG_LABELS: Dict[str, str] = {
    "qfrc_inverse": "qfrc_inv",
    "jacobian": "jac",
    "rotation": "rot",
}


def _mean_metric(metrics: Dict[str, List[float]], key: str) -> float:
    values = metrics.get(key, [])
    if not values:
        return float("nan")
    return float(np.mean(values))


def _format_metric_pairs(
    metrics: Dict[str, List[float]],
    loss_weights: Dict[str, float],
    *,
    scaled: bool,
    items: Sequence[Tuple[str, Optional[str], str]],
) -> str:
    parts: List[str] = []
    for metric_key, weight_key, label in items:
        value = _mean_metric(metrics, metric_key)
        if scaled and weight_key is not None:
            value *= float(loss_weights.get(weight_key, 1.0))
        parts.append(f"{label}={value:.4f}")
    return " ".join(parts)


def _format_debug_metric_pairs(
    metrics: Dict[str, List[float]],
    items: Sequence[Tuple[str, str]],
) -> str:
    parts: List[str] = []
    for metric_key, label in items:
        value = _mean_metric(metrics, metric_key)
        parts.append(f"{label}={value:.4f}")
    return " ".join(parts)


def _ts_print(*values: Any, sep: str = " ", end: str = "\n", flush: bool = True) -> None:
    text = sep.join(str(value) for value in values)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = text.splitlines() or [""]
    prefixed = "\n".join(f"[{timestamp}] {line}" if line else f"[{timestamp}]" for line in lines)
    print(prefixed, end=end, flush=flush)


def _print_loss_report(
    header: str,
    metrics: Dict[str, List[float]],
    loss_weights: Dict[str, float],
) -> None:
    group_a = LOSS_REPORT_ORDER[:7]
    group_b = LOSS_REPORT_ORDER[7:]
    total_value = _mean_metric(metrics, "total_loss")
    _ts_print(f"{header}: total={total_value:.4f}")
    _ts_print(f"  Raw A    {_format_metric_pairs(metrics, loss_weights, scaled=False, items=group_a)}")
    _ts_print(f"  Raw B    {_format_metric_pairs(metrics, loss_weights, scaled=False, items=group_b)}")
    _ts_print(f"  Scaled A {_format_metric_pairs(metrics, loss_weights, scaled=True, items=group_a)}")
    _ts_print(f"  Scaled B {_format_metric_pairs(metrics, loss_weights, scaled=True, items=group_b)}")
    if any(metrics.get(metric_key) for metric_key, _label in TORQUE_DEBUG_REPORT_ORDER):
        _ts_print(f"  Torque   {_format_debug_metric_pairs(metrics, TORQUE_DEBUG_REPORT_ORDER)}")
    if any(metrics.get(metric_key) for metric_key, _label in SIGNAL_DEBUG_REPORT_ORDER):
        _ts_print(f"  Signals  {_format_debug_metric_pairs(metrics, SIGNAL_DEBUG_REPORT_ORDER)}")
    if any(metrics.get(metric_key) for metric_key, _label in COP_WORLD_GTROT_DEBUG_REPORT_ORDER):
        _ts_print(f"  COPWorld {_format_debug_metric_pairs(metrics, COP_WORLD_GTROT_DEBUG_REPORT_ORDER)}")
    if any(metrics.get(metric_key) for metric_key, _label in COP_WORLD_PREDROT_DEBUG_REPORT_ORDER):
        _ts_print(f"  COPWorld {_format_debug_metric_pairs(metrics, COP_WORLD_PREDROT_DEBUG_REPORT_ORDER)}")
    if any(metrics.get(metric_key) for metric_key, _label in BASELINE_DEBUG_REPORT_ORDER):
        _ts_print(f"  Compare  {_format_debug_metric_pairs(metrics, BASELINE_DEBUG_REPORT_ORDER)}")
    if any(metrics.get(metric_key) for metric_key, _label in KINEMATICS_DEBUG_REPORT_ORDER):
        _ts_print(f"  Kine     {_format_debug_metric_pairs(metrics, KINEMATICS_DEBUG_REPORT_ORDER)}")
    if any(metrics.get(metric_key) for metric_key, _label in JACOBIAN_DEBUG_REPORT_ORDER):
        _ts_print(f"  Jacobian {_format_debug_metric_pairs(metrics, JACOBIAN_DEBUG_REPORT_ORDER)}")
    if any(metrics.get(metric_key) for metric_key, _label in ROTATION_DEBUG_REPORT_ORDER):
        _ts_print(f"  Rotation {_format_debug_metric_pairs(metrics, ROTATION_DEBUG_REPORT_ORDER)}")
    if any(metrics.get(metric_key) for metric_key, _label in ROTATION_GEODESIC_DEBUG_REPORT_ORDER):
        _ts_print(f"  RotGeo   {_format_debug_metric_pairs(metrics, ROTATION_GEODESIC_DEBUG_REPORT_ORDER)}")
    for term in KINEMATIC_EQUIV_TERMS:
        if not any(metrics.get(f"{term}_raw_equiv_{component}_rmse") or metrics.get(f"{term}_scaled_equiv_{component}_rmse") for component in KINEMATIC_EQUIV_COMPONENTS):
            continue
        label = KINEMATIC_EQUIV_LOG_LABELS[term]
        _ts_print(
            "  PhysEqZR",
            f"{label} "
            + " ".join(
                f"{component}={_mean_metric(metrics, f'{term}_raw_equiv_{component}_rmse'):.4f}"
                for component in KINEMATIC_EQUIV_COMPONENTS
            ),
        )
        _ts_print(
            "  PhysEqPR",
            f"{label} "
            + " ".join(
                f"{component}={_mean_metric(metrics, f'{term}_raw_equiv_{component}_rmse_phys'):.4f}"
                for component in KINEMATIC_EQUIV_COMPONENTS
            ),
        )
        _ts_print(
            "  PhysEqGR",
            f"{label} "
            + " ".join(
                f"{component}={_mean_metric(metrics, f'{term}_raw_grad_{component}_l2'):.4f}"
                for component in KINEMATIC_EQUIV_COMPONENTS
            ),
        )
        _ts_print(
            "  PhysEqZS",
            f"{label} "
            + " ".join(
                f"{component}={_mean_metric(metrics, f'{term}_scaled_equiv_{component}_rmse'):.4f}"
                for component in KINEMATIC_EQUIV_COMPONENTS
            ),
        )
        _ts_print(
            "  PhysEqPS",
            f"{label} "
            + " ".join(
                f"{component}={_mean_metric(metrics, f'{term}_scaled_equiv_{component}_rmse_phys'):.4f}"
                for component in KINEMATIC_EQUIV_COMPONENTS
            ),
        )
        _ts_print(
            "  PhysEqGS",
            f"{label} "
            + " ".join(
                f"{component}={_mean_metric(metrics, f'{term}_scaled_grad_{component}_l2'):.4f}"
                for component in KINEMATIC_EQUIV_COMPONENTS
            ),
        )


def _wandb_epoch_metrics(prefix: str, metrics: Dict[str, List[float]], loss_weights: Dict[str, float]) -> Dict[str, float]:
    logged: Dict[str, float] = {f"{prefix}/total_loss": _mean_metric(metrics, "total_loss")}
    for metric_key, weight_key, _label in LOSS_REPORT_ORDER:
        mean_value = _mean_metric(metrics, metric_key)
        logged[f"{prefix}/{metric_key}"] = mean_value
        if weight_key is not None:
            logged[f"{prefix}/{metric_key}_scaled"] = mean_value * float(loss_weights.get(weight_key, 1.0))
    for metric_key, _label in TORQUE_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for metric_key, _label in SIGNAL_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for metric_key, _label in COP_WORLD_GTROT_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for metric_key, _label in COP_WORLD_PREDROT_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for metric_key, _label in BASELINE_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for metric_key, _label in KINEMATICS_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for metric_key, _label in JACOBIAN_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for metric_key, _label in ROTATION_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for metric_key, _label in ROTATION_GEODESIC_DEBUG_REPORT_ORDER:
        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    for term in KINEMATIC_EQUIV_TERMS:
        for component in KINEMATIC_EQUIV_COMPONENTS:
            for variant in ("raw", "scaled"):
                for suffix in ("loss", "rmse", "rmse_phys"):
                    metric_key = f"{term}_{variant}_equiv_{component}_{suffix}"
                    if metrics.get(metric_key):
                        logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
                metric_key = f"{term}_{variant}_grad_{component}_l2"
                if metrics.get(metric_key):
                    logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
            for suffix in ("loss", "rmse", "rmse_phys"):
                metric_key = f"{term}_equiv_{component}_{suffix}"
                if metrics.get(metric_key):
                    logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
            metric_key = f"{term}_grad_{component}_l2"
            if metrics.get(metric_key):
                logged[f"{prefix}/{metric_key}"] = _mean_metric(metrics, metric_key)
    return logged


def _sample_label(batch: Dict[str, Any], sample_idx: int) -> str:
    trial_name = batch.get("trial_name", "unknown")
    if isinstance(trial_name, (list, tuple)) and sample_idx < len(trial_name):
        return str(trial_name[sample_idx])
    return str(trial_name)


def _sample_frame_mask(batch: Dict[str, Any], sample_idx: int, seq_len: int) -> np.ndarray:
    mask = batch.get("supervision_mask")
    if mask is None:
        return np.ones(seq_len, dtype=bool)
    mask_np = np.asarray(mask[sample_idx], dtype=np.float32)
    if mask_np.ndim > 1:
        mask_np = mask_np[..., 0]
    if mask_np.shape[0] != seq_len:
        return np.ones(seq_len, dtype=bool)
    return mask_np > 0.5


def _masked_channel_errors(pred: np.ndarray, target: np.ndarray, frame_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_np = np.asarray(pred, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    if pred_np.ndim == 1:
        pred_np = pred_np[:, None]
        target_np = target_np[:, None]
    valid = np.asarray(frame_mask, dtype=bool)
    if valid.ndim != 1 or valid.shape[0] != pred_np.shape[0] or not np.any(valid):
        valid = np.ones(pred_np.shape[0], dtype=bool)
    diff = pred_np[valid] - target_np[valid]
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(np.square(diff), axis=0))
    return mae.astype(np.float32), rmse.astype(np.float32)


def _render_modq_split_panel(
    axes: np.ndarray,
    *,
    split_name: str,
    batch: Dict[str, Any],
    pred: np.ndarray,
    normalizers: Dict[str, Normalizer],
    sample_idx: int,
    deviation_learning: bool,
    loss_weights: Dict[str, float],
    epoch_metrics: Optional[Dict[str, List[float]]],
) -> Dict[str, float]:
    pred_jnp = jnp.asarray(pred, dtype=jnp.float32)
    pred_parts = split_modq_predictions(pred_jnp)
    pos_pred_z = pred_parts["pos"]
    vel_pred_z = pred_parts["vel"]
    acc_pred_z = pred_parts["acc"]
    if deviation_learning:
        pos_pred_z = pos_pred_z + jnp.asarray(batch["pos_noised"], dtype=jnp.float32)
        vel_pred_z = vel_pred_z + jnp.asarray(batch["vel_noised"], dtype=jnp.float32)
        acc_pred_z = acc_pred_z + jnp.asarray(batch["acc_noised"], dtype=jnp.float32)

    height_raw, _, body_weight_raw, bw_height_raw = _raw_scale_factors_from_batch(batch, dtype=jnp.float32)
    contact_scale = jnp.clip(pred_parts["contact"], 0.0, 1.0)
    cop_pred_phys, grf_pred_phys, moments_pred_phys = _restore_output_units(
        pred_parts["cop"],
        pred_parts["grf"],
        pred_parts["moments"],
        normalizers,
        height_raw=height_raw,
        body_weight_raw=body_weight_raw,
        bw_height_raw=bw_height_raw,
        contact_scale=contact_scale,
    )
    cop_gt_phys, grf_gt_phys, moments_gt_phys = _restore_output_units(
        jnp.asarray(batch["cop"], dtype=jnp.float32),
        jnp.asarray(batch["grf"], dtype=jnp.float32),
        jnp.asarray(batch["moments"], dtype=jnp.float32),
        normalizers,
        height_raw=height_raw,
        body_weight_raw=body_weight_raw,
        bw_height_raw=bw_height_raw,
    )
    pos_pred_phys = normalizers["pos"].unnormalize(pos_pred_z)
    vel_pred_phys = normalizers["vel"].unnormalize(vel_pred_z)
    acc_pred_phys = normalizers["acc"].unnormalize(acc_pred_z)
    pos_gt_phys = normalizers["pos"].unnormalize(jnp.asarray(batch["pos_gt"], dtype=jnp.float32))
    vel_gt_phys = normalizers["vel"].unnormalize(jnp.asarray(batch["vel_gt"], dtype=jnp.float32))
    acc_gt_phys = normalizers["acc"].unnormalize(jnp.asarray(batch["acc_gt"], dtype=jnp.float32))

    seq_len = int(np.asarray(batch["cop"][sample_idx]).shape[0])
    frames = np.arange(seq_len)
    frame_mask = _sample_frame_mask(batch, sample_idx, seq_len)
    label = _sample_label(batch, sample_idx)

    cop_pred_np = np.asarray(cop_pred_phys[sample_idx])
    cop_gt_np = np.asarray(cop_gt_phys[sample_idx])
    grf_pred_np = np.asarray(grf_pred_phys[sample_idx])
    grf_gt_np = np.asarray(grf_gt_phys[sample_idx])
    moments_pred_np = np.asarray(moments_pred_phys[sample_idx])
    moments_gt_np = np.asarray(moments_gt_phys[sample_idx])
    pos_pred_np = np.asarray(pos_pred_phys[sample_idx])
    pos_gt_np = np.asarray(pos_gt_phys[sample_idx])
    vel_pred_np = np.asarray(vel_pred_phys[sample_idx])
    vel_gt_np = np.asarray(vel_gt_phys[sample_idx])
    acc_pred_np = np.asarray(acc_pred_phys[sample_idx])
    acc_gt_np = np.asarray(acc_gt_phys[sample_idx])
    contact_pred_np = np.asarray(contact_scale[sample_idx])
    contact_gt_np = np.asarray(batch["contactBoolean"][sample_idx])

    def _plot_group(ax, target, pred_vals, title, unit, scale=1.0, labels: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        target_np = np.asarray(target)
        pred_vals_np = np.asarray(pred_vals)
        mae, rmse = _masked_channel_errors(pred_vals_np, target_np, frame_mask)
        n_channels = int(target_np.shape[-1]) if target_np.ndim > 1 else 1
        chan_labels = labels if labels is not None else [f"C{i+1}" for i in range(n_channels)]
        for channel_idx in range(n_channels):
            suffix = chan_labels[channel_idx] if channel_idx < len(chan_labels) else f"C{channel_idx + 1}"
            ax.plot(frames, target_np[:, channel_idx] * scale, linewidth=1.6, label=f"GT {suffix}")
            ax.plot(frames, pred_vals_np[:, channel_idx] * scale, linestyle="--", linewidth=1.3, label=f"Pred {suffix}")
        unit_fmt = f" {unit}" if unit else ""
        ax.set_title(
            f"{split_name} {title}\nMAE {float(np.mean(mae)) * scale:.3g}{unit_fmt} | RMSE {float(np.mean(rmse)) * scale:.3g}{unit_fmt}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        return mae, rmse

    cop_labels = ["Rx", "Rz", "Lx", "Lz"]
    grf_labels = ["Rx", "Ry", "Rz", "Lx", "Ly", "Lz"]
    q_labels = [f"q{i+1}" for i in range(min(4, pos_gt_np.shape[-1]))]

    cop_mae, cop_rmse = _plot_group(axes[0], cop_gt_np, cop_pred_np, f"COP [{label}]", "cm", scale=100.0, labels=cop_labels)
    axes[0].legend(fontsize=6, ncol=2, loc="upper right")
    grf_mae, grf_rmse = _plot_group(axes[1], grf_gt_np, grf_pred_np, "GRF", "N", labels=grf_labels)
    pos_mae, pos_rmse = _plot_group(axes[2], pos_gt_np[:, : len(q_labels)], pos_pred_np[:, : len(q_labels)], "qpos (first 4)", "", labels=q_labels)
    vel_mae, vel_rmse = _plot_group(axes[3], vel_gt_np[:, : len(q_labels)], vel_pred_np[:, : len(q_labels)], "qvel (first 4)", "", labels=q_labels)
    acc_mae, acc_rmse = _plot_group(axes[4], acc_gt_np[:, : len(q_labels)], acc_pred_np[:, : len(q_labels)], "qacc (first 4)", "", labels=q_labels)

    contact_mae, _ = _masked_channel_errors(contact_pred_np, contact_gt_np, frame_mask)
    moment_mae, moment_rmse = _masked_channel_errors(moments_pred_np, moments_gt_np, frame_mask)
    axes[5].set_axis_off()
    lines = [
        f"{split_name} summary",
        f"Trial: {label}",
        f"COP mean MAE/RMSE: {float(np.mean(cop_mae)) * 100:.2f} / {float(np.mean(cop_rmse)) * 100:.2f} cm",
        f"GRF mean MAE/RMSE: {float(np.mean(grf_mae)):.2f} / {float(np.mean(grf_rmse)):.2f} N",
        f"Moment mean MAE/RMSE: {float(np.mean(moment_mae)):.3f} / {float(np.mean(moment_rmse)):.3f} Nm",
        f"qpos mean MAE/RMSE: {float(np.mean(pos_mae)):.4f} / {float(np.mean(pos_rmse)):.4f}",
        f"qvel mean MAE/RMSE: {float(np.mean(vel_mae)):.4f} / {float(np.mean(vel_rmse)):.4f}",
        f"qacc mean MAE/RMSE: {float(np.mean(acc_mae)):.4f} / {float(np.mean(acc_rmse)):.4f}",
        f"Contact mean abs err: {float(np.mean(contact_mae)):.4f}",
        f"Valid frames: {int(np.sum(frame_mask))}/{int(len(frame_mask))}",
    ]
    if epoch_metrics:
        total_loss = float(np.mean(epoch_metrics["total_loss"])) if epoch_metrics.get("total_loss") else float("nan")
        lines.append(f"Epoch avg total loss: {total_loss:.5f}")
        for loss_name in ("cop", "grf", "moments", "pos", "vel", "acc", "contact", "torque"):
            metric_key = f"{loss_name}_loss"
            if epoch_metrics.get(metric_key):
                raw = float(np.mean(epoch_metrics[metric_key]))
                scaled = raw * float(loss_weights.get(loss_name, 1.0))
                lines.append(f"{loss_name}: {raw:.4f} raw | {scaled:.4f} scaled")
    y = 0.98
    for line in lines:
        axes[5].text(0.02, y, line, transform=axes[5].transAxes, va="top", fontsize=9)
        y -= 0.09
        if y < 0.04:
            break

    return {
        "cop_mae_cm": float(np.mean(cop_mae)) * 100.0,
        "grf_mae_n": float(np.mean(grf_mae)),
        "moment_mae_nm": float(np.mean(moment_mae)),
        "pos_mae": float(np.mean(pos_mae)),
        "vel_mae": float(np.mean(vel_mae)),
        "acc_mae": float(np.mean(acc_mae)),
        "contact_mae": float(np.mean(contact_mae)),
    }


def plot_modq_predictions(
    train_batch: Dict[str, Any],
    train_pred: np.ndarray,
    val_batch: Dict[str, Any],
    val_pred: np.ndarray,
    normalizers: Dict[str, Normalizer],
    epoch: int,
    output_dir: Path,
    *,
    deviation_learning: bool,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Path:
    loss_weights = loss_weights or {}
    fig, axes = plt.subplots(6, 2, figsize=(18, 24))
    fig.subplots_adjust(hspace=0.45, wspace=0.20, top=0.95, bottom=0.03, left=0.05, right=0.98)

    train_stats = _render_modq_split_panel(
        axes[:, 0],
        split_name="TRAIN",
        batch=train_batch,
        pred=train_pred,
        normalizers=normalizers,
        sample_idx=0,
        deviation_learning=deviation_learning,
        loss_weights=loss_weights,
        epoch_metrics=train_metrics,
    )
    val_stats = _render_modq_split_panel(
        axes[:, 1],
        split_name="VAL",
        batch=val_batch,
        pred=val_pred,
        normalizers=normalizers,
        sample_idx=0,
        deviation_learning=deviation_learning,
        loss_weights=loss_weights,
        epoch_metrics=val_metrics,
    )

    fig.suptitle(
        "mod_q Prediction Summary "
        f"| Epoch {int(epoch)} "
        f"| Train COP/GRF/qpos MAE = {train_stats['cop_mae_cm']:.2f} cm / {train_stats['grf_mae_n']:.2f} N / {train_stats['pos_mae']:.4f} "
        f"| Val COP/GRF/qpos MAE = {val_stats['cop_mae_cm']:.2f} cm / {val_stats['grf_mae_n']:.2f} N / {val_stats['pos_mae']:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plot_path = output_dir / f"prediction_summary_epoch_{int(epoch):04d}.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _nonfinite_metric_names(metric_values: Dict[str, float]) -> List[str]:
    bad: List[str] = []
    for key, value in metric_values.items():
        try:
            if not np.isfinite(float(value)):
                bad.append(key)
        except Exception:
            bad.append(key)
    return bad


def _summarize_array(name: str, value: Any) -> str:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return f"{name}:empty"
        finite = np.isfinite(arr)
        return (
            f"{name}:shape={arr.shape} "
            f"finite={int(finite.all())} "
            f"min={float(np.nanmin(arr)):.4g} "
            f"max={float(np.nanmax(arr)):.4g} "
            f"mean={float(np.nanmean(arr)):.4g} "
            f"absmax={float(np.nanmax(np.abs(arr))):.4g}"
        )
    except Exception as exc:
        return f"{name}:summary_error={exc}"


def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return bool(value)


def _make_run_name(args: argparse.Namespace) -> str:
    base = str(args.exp_name).strip() or "mod_q"
    run_name = f"{base}_D{args.d_model}_L{args.num_layers}_W{args.window_size}_FF{args.ff_dim}"
    if bool(getattr(args, "DeviationLearning", False)):
        run_name += "_DEV"
    return run_name


def _resolve_batch_xml_path(batch: Dict[str, Any]) -> Optional[str]:
    xml_value = batch.get("subject_model_xml")
    if isinstance(xml_value, (list, tuple)):
        return str(xml_value[0]) if xml_value else None
    if isinstance(xml_value, np.ndarray):
        return str(xml_value.reshape(-1)[0]) if xml_value.size else None
    if xml_value is None:
        return None
    return str(xml_value)


def _structure_key_for_batch(batch: Dict[str, Any], adapter: Any) -> str:
    xml_path = _resolve_batch_xml_path(batch)
    if xml_path and getattr(adapter, "available", False):
        try:
            return str(adapter.get_structure_key(xml_path))
        except Exception:
            return f"path::{xml_path}"
    return "full_detached_fallback"


def _collect_structure_probe_batches(
    batcher: "ModQSubjectBatcher",
    *,
    normalizers: Dict[str, Normalizer],
    device: Any,
    adapter: Any,
    expected_structure_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    remaining = {str(key) for key in (expected_structure_keys or []) if str(key).strip()}
    probes: Dict[str, Dict[str, Any]] = {}
    for raw_batch in batcher.iter_batches():
        structure_key = _structure_key_for_batch(raw_batch, adapter)
        if structure_key in probes:
            continue
        probes[structure_key] = _make_prefetched_batch_item(raw_batch, normalizers=normalizers, device=device)
        if remaining:
            remaining.discard(structure_key)
            if not remaining:
                break
    return probes


def _touch_step_cache(cache: Dict[str, Any], cache_key: str) -> None:
    move_to_end = getattr(cache, "move_to_end", None)
    if callable(move_to_end) and cache_key in cache:
        move_to_end(cache_key)


def _drop_companion_cache_entries(
    primary_cache: Dict[str, Any],
    companion_cache: Dict[str, Any],
) -> None:
    for key in list(companion_cache.keys()):
        if key not in primary_cache:
            del companion_cache[key]


def _evict_full_stage_step_caches(
    *,
    train_step_cache: Dict[str, Any],
    eval_step_cache: Dict[str, Any],
    keep_cache_key: str,
    full_stage_cache_limit: int,
) -> None:
    if int(full_stage_cache_limit) <= 0:
        return
    full_keys = [key for key in train_step_cache.keys() if str(key).startswith("full::")]
    if len(full_keys) <= int(full_stage_cache_limit):
        return

    evicted: List[str] = []
    for key in list(full_keys):
        if len([k for k in train_step_cache.keys() if str(k).startswith("full::")]) <= int(full_stage_cache_limit):
            break
        if key == keep_cache_key:
            continue
        if key in train_step_cache:
            del train_step_cache[key]
        if key in eval_step_cache:
            del eval_step_cache[key]
        evicted.append(key)

    if evicted:
        gc.collect()
        if hasattr(jax, "clear_caches"):
            try:
                jax.clear_caches()
            except Exception:
                pass
        _ts_print(
            "[MJX] Evicted full-stage compiled cache entries to reduce RAM pressure: "
            + ", ".join(evicted)
        )


def _read_host_memory_snapshot() -> Optional[Dict[str, float]]:
    meminfo_path = "/proc/meminfo"
    if not os.path.exists(meminfo_path):
        return None
    values_kb: Dict[str, float] = {}
    try:
        with open(meminfo_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if not parts:
                    continue
                try:
                    values_kb[key.strip()] = float(parts[0])
                except ValueError:
                    continue
    except OSError:
        return None

    total_kb = values_kb.get("MemTotal")
    available_kb = values_kb.get("MemAvailable")
    if not total_kb or not available_kb or total_kb <= 0.0:
        return None
    return {
        "total_gb": total_kb / (1024.0 * 1024.0),
        "available_gb": available_kb / (1024.0 * 1024.0),
        "available_frac": available_kb / total_kb,
    }


def _periodic_runtime_cleanup(
    *,
    step_index: int,
    every_n_steps: int,
    stage_name: str,
    train_step_cache: Dict[str, Any],
    eval_step_cache: Dict[str, Any],
    active_cache_key: str,
    full_stage_cache_limit: int,
    low_ram_available_gb: float,
    low_ram_available_frac: float,
) -> None:
    if int(every_n_steps) <= 0 or int(step_index) <= 0 or (int(step_index) % int(every_n_steps)) != 0:
        return

    train_full = sum(1 for key in train_step_cache.keys() if str(key).startswith("full::"))
    eval_full = sum(1 for key in eval_step_cache.keys() if str(key).startswith("full::"))
    memory_snapshot = _read_host_memory_snapshot()
    memory_summary = ""
    aggressive_cleanup = False
    if memory_snapshot is not None:
        available_gb = float(memory_snapshot["available_gb"])
        available_frac = float(memory_snapshot["available_frac"])
        memory_summary = (
            f", host_ram_available_gb={available_gb:.2f}, "
            f"host_ram_available_pct={available_frac * 100.0:.1f}"
        )
        aggressive_cleanup = (
            available_gb <= float(low_ram_available_gb)
            or available_frac <= float(low_ram_available_frac)
        )
    _ts_print(
        f"[MJX] Periodic runtime cleanup at {stage_name} step {int(step_index)} "
        f"(train_full_cache={train_full}, eval_full_cache={eval_full}{memory_summary})."
    )

    if aggressive_cleanup and active_cache_key.startswith("full::"):
        _ts_print(
            f"[MJX] Host RAM is low; escalating cleanup at {stage_name} step {int(step_index)} "
            f"by shrinking full-stage caches toward the active structure '{active_cache_key}'."
        )
        _evict_full_stage_step_caches(
            train_step_cache=train_step_cache,
            eval_step_cache=eval_step_cache,
            keep_cache_key=active_cache_key,
            full_stage_cache_limit=1 if int(full_stage_cache_limit) > 0 else 0,
        )

    gc.collect()
    if hasattr(jax, "clear_caches"):
        try:
            jax.clear_caches()
        except Exception:
            pass


def _trim_compiled_probe_batches(
    *,
    train_probe_batches: Dict[str, Dict[str, Any]],
    val_probe_batches: Dict[str, Dict[str, Any]],
    train_step_cache: Dict[str, Any],
    eval_step_cache: Dict[str, Any],
) -> None:
    compiled_keys = {
        str(key)[len("full::") :]
        for key in train_step_cache.keys()
        if str(key).startswith("full::") and key in eval_step_cache
    }
    if not compiled_keys:
        return
    for structure_key in list(train_probe_batches.keys()):
        if structure_key in compiled_keys:
            train_probe_batches.pop(structure_key, None)
    for structure_key in list(val_probe_batches.keys()):
        if structure_key in compiled_keys:
            val_probe_batches.pop(structure_key, None)


def _clone_state_for_warmup(state: Any, device: Any) -> Any:
    host_state = jax.device_get(state)
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(np.asarray(x), device=device) if isinstance(x, (jax.Array, np.ndarray)) else x,
        host_state,
    )


def _compile_ahead_structure_keys(
    *,
    anchor_structure_key: str,
    structure_order: Sequence[str],
    compile_ahead_groups: int,
) -> List[str]:
    anchor_structure_key = str(anchor_structure_key)
    if not anchor_structure_key:
        return []
    window_size = max(1, int(compile_ahead_groups))
    ordered_unique: List[str] = []
    seen: set[str] = set()
    for key in structure_order:
        key_str = str(key).strip()
        if not key_str or key_str in seen:
            continue
        seen.add(key_str)
        ordered_unique.append(key_str)
    if anchor_structure_key not in ordered_unique:
        return [anchor_structure_key]
    anchor_idx = ordered_unique.index(anchor_structure_key)
    return ordered_unique[anchor_idx : anchor_idx + window_size]


def _precompile_full_stage_steps(
    *,
    state: Any,
    train_probe_batches: Dict[str, Dict[str, Any]],
    val_probe_batches: Dict[str, Dict[str, Any]],
    train_step_cache: Dict[str, Any],
    eval_step_cache: Dict[str, Any],
    adapter: Any,
    normalizers: Dict[str, Normalizer],
    loss_weights: Dict[str, float],
    deviation_learning: bool,
    cop_mask: bool,
    use_contact_weighting: bool,
    contact_weight_multiplier: float,
    prefetch_device: Any,
    full_stage_cache_limit: int,
    precompile_max_groups: int,
    ordered_probe_keys: Optional[Sequence[str]] = None,
    reason: str = "full-stage precompile warmup",
) -> None:
    all_probe_batches: Dict[str, Dict[str, Any]] = dict(train_probe_batches)
    for structure_key, probe in val_probe_batches.items():
        all_probe_batches.setdefault(structure_key, probe)
    if not all_probe_batches:
        _ts_print(f"[MJX] No structure-group probe batches were available for {reason}.")
        return

    probe_key_order = [str(key).strip() for key in (ordered_probe_keys or sorted(all_probe_batches.keys())) if str(key).strip()]
    if not ordered_probe_keys:
        max_groups = int(precompile_max_groups)
        if max_groups > 0:
            probe_key_order = probe_key_order[:max_groups]
    keys_to_prepare = [
        key
        for key in probe_key_order
        if key in all_probe_batches
        and (f"full::{key}" not in train_step_cache or f"full::{key}" not in eval_step_cache)
    ]
    if not keys_to_prepare:
        return

    _ts_print(
        f"[MJX] Starting {reason} for {len(keys_to_prepare)} structure group(s)."
    )
    for probe_idx, structure_key in enumerate(keys_to_prepare):
        prefetched = all_probe_batches[structure_key]
        raw_batch = prefetched["raw_batch"]
        jit_batch = prefetched["jit_batch"]
        xml_path = _resolve_batch_xml_path(raw_batch)
        cache_key = f"full::{structure_key}"
        if xml_path and adapter.available:
            physics_context = adapter.get_jit_context(xml_path)
            physics_runner = adapter.get_runner(xml_path)
            kinematics_reconstructor = adapter.get_reconstructor(xml_path)
            runtime_xml = adapter.get_runtime_xml_path(xml_path)
        else:
            physics_context = {}
            physics_runner = None
            kinematics_reconstructor = None
            runtime_xml = None

        if cache_key not in train_step_cache:
            train_step_cache[cache_key] = make_train_step(
                normalizers=normalizers,
                loss_weights=loss_weights,
                deviation_learning=deviation_learning,
                cop_mask=cop_mask,
                use_contact_weighting=use_contact_weighting,
                contact_weight_multiplier=contact_weight_multiplier,
                physics_runner=physics_runner,
                kinematics_reconstructor=kinematics_reconstructor,
                run_physics=True,
            )
        _touch_step_cache(train_step_cache, cache_key)
        if cache_key not in eval_step_cache:
            eval_step_cache[cache_key] = make_eval_step(
                normalizers=normalizers,
                loss_weights=loss_weights,
                deviation_learning=deviation_learning,
                cop_mask=cop_mask,
                use_contact_weighting=use_contact_weighting,
                contact_weight_multiplier=contact_weight_multiplier,
                physics_runner=physics_runner,
                kinematics_reconstructor=kinematics_reconstructor,
                run_physics=True,
            )
        _touch_step_cache(eval_step_cache, cache_key)
        _evict_full_stage_step_caches(
            train_step_cache=train_step_cache,
            eval_step_cache=eval_step_cache,
            keep_cache_key=cache_key,
            full_stage_cache_limit=full_stage_cache_limit,
        )

        _ts_print(
            f"[MJX] Precompiling structure group '{structure_key}' "
            f"subject={raw_batch.get('subject', 'unknown')} xml={xml_path} "
            f"runtime_xml={runtime_xml if runtime_xml is not None else 'detached_fallback'}"
        )
        probe_state = _clone_state_for_warmup(state, prefetch_device)
        probe_rng = jax.random.fold_in(jax.random.PRNGKey(0), probe_idx)
        _probe_state, train_loss, _train_metrics = train_step_cache[cache_key](
            probe_state,
            jit_batch,
            physics_context,
            probe_rng,
        )
        eval_loss, _eval_metrics = eval_step_cache[cache_key](
            state,
            jit_batch,
            physics_context,
        )
        _ = jax.device_get({"train_loss": train_loss, "eval_loss": eval_loss})
    _ts_print(f"[MJX] {reason} complete.")


def _split_train_val_by_subject(trials: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    subject_to_trials: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        subject_to_trials[str(trial["subject"])].append(trial)
    subjects = sorted(subject_to_trials.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)
    if len(subjects) <= 1:
        flat = list(trials)
        rng.shuffle(flat)
        pivot = max(1, int(0.8 * len(flat)))
        return flat[:pivot], flat[pivot:] or flat[:1]
    pivot = max(1, int(0.8 * len(subjects)))
    train_subjects = set(subjects[:pivot])
    train_trials = [t for t in trials if t["subject"] in train_subjects]
    val_trials = [t for t in trials if t["subject"] not in train_subjects]
    return train_trials, val_trials


def _build_model_from_sample(sample_batch: Dict[str, Any], args: argparse.Namespace):
    input_dim = int(sample_batch["input"].shape[-1])
    static_dim = int(sample_batch["static_context"].shape[-1])
    model = ModQTransformer(
        input_dim=input_dim,
        static_dim=static_dim,
        output_dim=int(MODQ_OUTPUT_DIM),
        d_model=int(args.d_model),
        num_heads=4,
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout_rate=float(args.dropout_rate),
        use_cnn=bool(args.use_cnn),
        cnn_num_layers=int(args.cnn_num_layers),
        cnn_kernel_sizes=tuple(int(k) for k in str(args.cnn_kernel_sizes).split(",") if str(k).strip()),
    )
    return model, input_dim, static_dim


def _batch_generator_for_trials(
    trials: List[Dict[str, Any]],
    *,
    window_size: int,
    stride: int,
    prediction_margin_frames: int,
    batch_size: int,
    shuffle: bool,
    use_noised: bool,
) -> Iterable[Dict[str, Any]]:
    batcher = ModQSubjectBatcher(
        trials,
        window_size=window_size,
        stride=stride,
        prediction_margin_frames=prediction_margin_frames,
        batch_size=batch_size,
        shuffle=shuffle,
        use_noised=use_noised,
    )
    yield from batcher.iter_batches()


def _sample_batches_from_trials(
    trials: List[Dict[str, Any]],
    *,
    window_size: int,
    stride: int,
    prediction_margin_frames: int,
    batch_size: int,
    use_noised: bool,
    max_batches: int = 8,
) -> List[Dict[str, Any]]:
    batches = []
    for batch in _batch_generator_for_trials(
        trials,
        window_size=window_size,
        stride=stride,
        prediction_margin_frames=prediction_margin_frames,
        batch_size=batch_size,
        shuffle=True,
        use_noised=use_noised,
    ):
        batches.append(batch)
        if len(batches) >= max_batches:
            break
    return batches


def _count_processed_trials_missing_q_mjx(data_dir: str) -> Tuple[int, int]:
    root = Path(data_dir)
    processed_count = 0
    missing_count = 0
    if not root.exists():
        return 0, 0
    for processed_dir in root.rglob("ProcessedData"):
        if not processed_dir.is_dir():
            continue
        processed_count += 1
        missing_clean = not (processed_dir / "qvel_mjx.npy").exists() or not (processed_dir / "qacc_mjx.npy").exists()
        missing_noised = (
            not (processed_dir / "qvel_mjx_noised.npy").exists()
            or not (processed_dir / "qacc_mjx_noised.npy").exists()
        )
        if missing_clean or missing_noised:
            missing_count += 1
    return processed_count, missing_count


def _save_checkpoint(
    output_dir: Path,
    state: train_state.TrainState,
    normalizers: Dict[str, Normalizer],
    *,
    train_trials: List[Dict[str, Any]],
    val_trials: List[Dict[str, Any]],
    best_val_loss: float,
    args: argparse.Namespace,
    input_dim: int,
    static_dim: int,
    output_dim: int,
    active_loss_weights: Dict[str, float],
    full_loss_weights: Dict[str, float],
    warmup_loss_weights: Dict[str, float],
    active_stage: str,
) -> None:
    ckpt = {
        "mod_q_metadata": {
            "DeviationLearning": bool(args.DeviationLearning),
            "kinematics_prediction_mode": "residual_over_noised" if bool(args.DeviationLearning) else "direct_absolute",
        },
        "params": state.params,
        "normalizers": normalizers,
        "train_trials": train_trials,
        "val_trials": val_trials,
        "best_val_loss": best_val_loss,
        "model_type": "mod_q",
        "output_schema": MODQ_OUTPUT_SCHEMA,
        "qprime_layout": MODQ_QPRIME_LAYOUT,
        "input_feature_blocks": MODQ_INPUT_FEATURE_BLOCKS,
        "subject_grouped_batches": True,
        "forced_flags": {**MODQ_FORCED_FLAGS, "DeviationLearning": bool(args.DeviationLearning)},
        "input_dim": int(input_dim),
        "static_dim": int(static_dim),
        "output_dim": int(output_dim),
        "physics_backend": "mjx_jit_differentiable",
        "rotation_loss_type": "geodesic_mse",
        "runtime_model_xml_name": RUNTIME_XML_NAME,
        "runtime_structure_grouping": "core_shape_with_family_max_nsite",
        "use_cnn": bool(args.use_cnn),
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "ff_dim": int(args.ff_dim),
        "dropout_rate": float(args.dropout_rate),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "loss_weights": active_loss_weights,
        "full_loss_weights": full_loss_weights,
        "warmup_loss_weights": warmup_loss_weights,
        "warmup_epochs": int(args.warmup_epochs),
        "training_stage_at_save": active_stage,
        "DeviationLearning": bool(args.DeviationLearning),
        "kinematics_prediction_mode": "residual_over_noised" if bool(args.DeviationLearning) else "direct_absolute",
    }
    with open(output_dir / "best_model.pkl", "wb") as f:
        pickle.dump(ckpt, f)

    hyperparams = {
        "model_type": "mod_q",
        "output_schema": MODQ_OUTPUT_SCHEMA,
        "qprime_layout": MODQ_QPRIME_LAYOUT,
        "input_feature_blocks": MODQ_INPUT_FEATURE_BLOCKS,
        "subject_grouped_batches": True,
        "forced_flags": {**MODQ_FORCED_FLAGS, "DeviationLearning": bool(args.DeviationLearning)},
        "input_dim": int(input_dim),
        "static_dim": int(static_dim),
        "output_dim": int(output_dim),
        "physics_backend": "mjx_jit_differentiable",
        "rotation_loss_type": "geodesic_mse",
        "runtime_model_xml_name": RUNTIME_XML_NAME,
        "runtime_structure_grouping": "core_shape_with_family_max_nsite",
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "ff_dim": int(args.ff_dim),
        "dropout_rate": float(args.dropout_rate),
        "use_cnn": bool(args.use_cnn),
        "cnn_num_layers": int(args.cnn_num_layers),
        "cnn_kernel_sizes": [int(k) for k in str(args.cnn_kernel_sizes).split(",") if str(k).strip()],
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "prefetch_batches": int(args.prefetch_batches),
        "prediction_margin_frames": int(args.prediction_margin_frames),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "loss_weights": active_loss_weights,
        "full_loss_weights": full_loss_weights,
        "warmup_loss_weights": warmup_loss_weights,
        "warmup_epochs": int(args.warmup_epochs),
        "training_stage_at_save": active_stage,
        "UseNoised": True,
        "includePelvisEuler": True,
        "PredictJacobian": False,
        "DeviationLearning": bool(args.DeviationLearning),
        "kinematics_prediction_mode": "residual_over_noised" if bool(args.DeviationLearning) else "direct_absolute",
        "derived_qprime_from_templates": False,
    }
    with open(output_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2)
    save_model_parameters_yaml(hyperparams, str(output_dir / "model_parameters.yaml"))


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the mod_q transformer")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(artifact("outputs", "mod_q")))
    parser.add_argument("--exp_name", type=str, default="mod_q")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--prefetch_batches", type=int, default=2)
    parser.add_argument("--prediction_margin_frames", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--use_cnn", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--cnn_num_layers", type=int, default=2)
    parser.add_argument("--cnn_kernel_sizes", type=str, default="3,5")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument("--save_final_predictions_only", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="gait-dynamics-jax")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--cop_weight", type=float, default=1.0)
    parser.add_argument("--grf_weight", type=float, default=1.0)
    parser.add_argument("--moments_weight", type=float, default=0.25)
    parser.add_argument("--contact_weight", type=float, default=1.0)
    parser.add_argument("--torque_weight", type=float, default=2.0)
    parser.add_argument("--grf_correction_weight", type=float, default=0.0)
    parser.add_argument("--output_reg_weight", type=float, default=0.0)
    parser.add_argument(
        "--DeviationLearning",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Interpret pos/vel/acc outputs as residuals added to the noised kinematic inputs before loss/physics. Defaults to True for mod_q.",
    )
    parser.add_argument("--qpos_weight", type=float, default=1.0)
    parser.add_argument("--qvel_weight", type=float, default=1.0)
    parser.add_argument("--qacc_weight", type=float, default=1.0)
    parser.add_argument("--qfrc_inverse_weight", type=float, default=1.0)
    parser.add_argument("--jacobian_weight", type=float, default=1.0)
    parser.add_argument("--rotation_weight", type=float, default=1.0)
    parser.add_argument("--full_id_weight", type=float, default=0.0)
    parser.add_argument("--full_stage_cache_limit", type=int, default=2)
    parser.add_argument("--full_stage_precompile_max_groups", type=int, default=0)
    parser.add_argument(
        "--full_stage_compile_ahead_groups",
        type=int,
        default=1,
        help="When a new full-stage structure group is first encountered, sequentially precompile this many upcoming structure groups.",
    )
    parser.add_argument(
        "--clear_runtime_cache_every",
        type=int,
        default=0,
        help="Run a periodic gc/JAX cache cleanup every N full-stage train/val steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--low_ram_available_gb",
        type=float,
        default=8.0,
        help="Escalate periodic cleanup when host RAM available falls to this many GB or lower.",
    )
    parser.add_argument(
        "--low_ram_available_frac",
        type=float,
        default=0.10,
        help="Escalate periodic cleanup when host RAM available falls to this fraction or lower.",
    )
    parser.add_argument("--use_contact_weighting", type=lambda x: str(x).lower() != "false", default=False)
    parser.add_argument("--contact_weight_multiplier", type=float, default=1.5)
    parser.add_argument("--magOnOff", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--contactOnOff", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--magWeight", type=float, default=3.0)
    parser.add_argument("--cop_mask", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument(
        "--log_kinematic_equiv",
        type=lambda x: str(x).lower() != "false",
        default=True,
        help="Log sparse kinematic-equivalent diagnostics for qfrc_inverse, jacobian, and rotation.",
    )
    parser.add_argument(
        "--kinematic_equiv_interval",
        type=int,
        default=100,
        help="Run the train-time kinematic-equivalent probe every N steps; 0 disables train-time probes.",
    )
    parser.add_argument(
        "--check_mjx_gradients",
        action="store_true",
        help="Run a one-off JAX grad smoke test through the differentiable MJX path before training.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _make_parser()
    args = parser.parse_args(argv)

    # Force the requested mod_q defaults.
    args.UseNoised = True
    args.includePelvisEuler = True
    args.PredictJacobian = False

    if not args.window_size or args.window_size <= 0:
        parser.error("--window_size must be > 0")
    validate_prediction_margin(args.window_size, args.prediction_margin_frames)

    run_name = _make_run_name(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if RUNTIME_ENV_APPLIED:
        _ts_print("Applied runtime env:", RUNTIME_ENV_APPLIED)

    wandb_tags = [tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()]
    wandb_logger = WandbLogger(
        enabled=bool(args.use_wandb),
        project=args.wandb_project,
        run_name=run_name,
        config=dict(vars(args)),
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        tags=wandb_tags,
        dir=str(output_dir),
    )

    trials = discover_all_trials_modq(args.data_dir, refresh_cache=args.refresh_cache)
    if not trials:
        raise RuntimeError(f"No mod_q-compatible trials found under {args.data_dir}")

    train_trials, val_trials = _split_train_val_by_subject(trials)
    with open(output_dir / "train_val_split.json", "w", encoding="utf-8") as f:
        json.dump({"train_trials": train_trials, "val_trials": val_trials}, f, indent=2)

    train_batcher = ModQSubjectBatcher(
        train_trials,
        window_size=args.window_size,
        stride=args.stride,
        prediction_margin_frames=args.prediction_margin_frames,
        batch_size=args.batch_size,
        shuffle=True,
        use_noised=True,
    )
    val_batcher = ModQSubjectBatcher(
        val_trials,
        window_size=args.window_size,
        stride=args.stride,
        prediction_margin_frames=args.prediction_margin_frames,
        batch_size=args.batch_size,
        shuffle=False,
        use_noised=True,
    )
    train_batches: List[Dict[str, Any]] = []
    for batch in train_batcher.iter_batches():
        train_batches.append(batch)
        if len(train_batches) >= 8:
            break
    if not train_batches:
        processed_count, missing_q_mjx = _count_processed_trials_missing_q_mjx(args.data_dir)
        if processed_count > 0 and missing_q_mjx > 0:
            raise RuntimeError(
                "No training batches could be assembled. "
                f"{missing_q_mjx} of {processed_count} ProcessedData folders are missing clean and/or noised "
                "pos_mjx templates required by mod_q."
            )
        raise RuntimeError("No training batches could be assembled.")

    normalizers = compute_normalizers_from_batches(train_batches, max_batches=len(train_batches))
    sample_batch = _prepare_batch(train_batches[0], normalizers)
    model, input_dim, static_dim = _build_model_from_sample(sample_batch, args)
    output_dim = int(MODQ_OUTPUT_DIM)

    rng = jax.random.PRNGKey(42)
    state = create_train_state(
        rng,
        model,
        sample_batch["input"].shape,
        sample_batch["static_context"].shape,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    prefetch_device = jax.devices()[0]

    if args.resume_checkpoint:
        with open(args.resume_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        if "params" in ckpt:
            state = state.replace(params=ckpt["params"])
        if "normalizers" in ckpt:
            normalizers = ckpt["normalizers"]

    full_loss_weights = _make_loss_weights(args)
    warmup_loss_weights = _make_warmup_loss_weights(full_loss_weights)
    _ts_print("Loss weights (full stage):", json.dumps(full_loss_weights, indent=2, sort_keys=True))
    if int(args.warmup_epochs) > 0:
        _ts_print(f"Warmup epochs: {int(args.warmup_epochs)}")
        _ts_print("Loss weights (warmup stage):", json.dumps(warmup_loss_weights, indent=2, sort_keys=True))
    _ts_print(
        f"[PREFETCH] Using prefetch_batches={int(args.prefetch_batches)} on device={prefetch_device.platform}:{prefetch_device.id}"
    )
    adapter = ModQPhysicsAdapter()
    train_structure_keys = sorted(
        {
            str(trial.get("subject_structure_key", "")).strip()
            for trial in train_trials
            if str(trial.get("subject_structure_key", "")).strip()
        }
    )
    val_structure_keys = sorted(
        {
            str(trial.get("subject_structure_key", "")).strip()
            for trial in val_trials
            if str(trial.get("subject_structure_key", "")).strip()
        }
    )
    full_stage_structure_order = list(dict.fromkeys(train_structure_keys + val_structure_keys))
    train_probe_batches = _collect_structure_probe_batches(
        train_batcher,
        normalizers=normalizers,
        device=prefetch_device,
        adapter=adapter,
        expected_structure_keys=train_structure_keys,
    )
    val_probe_batches = _collect_structure_probe_batches(
        val_batcher,
        normalizers=normalizers,
        device=prefetch_device,
        adapter=adapter,
        expected_structure_keys=val_structure_keys,
    )
    _ts_print(
        f"[MJX] Prepared {len(train_probe_batches)} train and {len(val_probe_batches)} validation "
        "structure-group probe batch(es) for full-stage precompile warmup."
    )
    mjx_gradients_required = _losses_require_differentiable_mjx(full_loss_weights) or bool(args.check_mjx_gradients)
    if not SHARED_MODQ_AVAILABLE:
        message = "Shared mod_q helper import failed; the differentiable MJX path is unavailable."
        if SHARED_MODQ_IMPORT_ERROR is not None:
            message += f" Import error: {type(SHARED_MODQ_IMPORT_ERROR).__name__}: {SHARED_MODQ_IMPORT_ERROR}"
        if mjx_gradients_required:
            raise RuntimeError(message)
        _ts_print(f"[WARN] {message}")
    elif not adapter.available:
        message = "mujoco/mjx is unavailable; the differentiable MJX path is disabled."
        if mjx_gradients_required:
            raise RuntimeError(message)
        _ts_print(f"[WARN] {message}")

    if float(full_loss_weights.get("torque", 0.0)) > 0.0 and not adapter.available:
        _ts_print(
            "[WARN] Torque loss will use the detached fallback path, so qpos/qvel/qacc will not backpropagate "
            "through live MJX Jacobians for that term.",
        )

    if args.check_mjx_gradients and int(args.warmup_epochs) > 0:
        _ts_print(
            f"[MJX] Gradient smoke test deferred until epoch {int(args.warmup_epochs) + 1}, "
            "when the full physics stage begins.",
        )

    train_step_cache: Dict[str, Any] = OrderedDict()
    eval_step_cache: Dict[str, Any] = OrderedDict()
    train_kinematic_equiv_cache: Dict[str, Any] = OrderedDict()
    eval_kinematic_equiv_cache: Dict[str, Any] = OrderedDict()
    best_val_loss = float("inf")
    best_val_stage: Optional[str] = None
    warned_train_batch_shapes: set[Tuple[int, ...]] = set()
    warned_val_batch_shapes: set[Tuple[int, ...]] = set()
    warned_train_structure_keys: set[str] = set()
    warned_val_structure_keys: set[str] = set()
    train_total_batches = train_batcher.num_batches()
    val_total_batches = val_batcher.num_batches()
    mjx_smoke_test_ran = False
    full_stage_precompile_done = False
    current_stage: Optional[str] = None
    compile_ahead_groups = max(1, int(args.full_stage_compile_ahead_groups))
    if compile_ahead_groups > int(args.full_stage_cache_limit):
        _ts_print(
            "[WARN] full_stage_compile_ahead_groups exceeds full_stage_cache_limit. "
            "Compile-ahead groups may be evicted before they are reused."
        )

    # Training loop
    for epoch in range(1, int(args.epochs) + 1):
        active_stage = _stage_for_epoch(epoch, int(args.warmup_epochs))
        run_physics = _stage_runs_physics(active_stage)
        active_loss_weights = full_loss_weights if run_physics else warmup_loss_weights
        if active_stage != current_stage:
            if current_stage is not None and active_stage != current_stage:
                best_val_loss = float("inf")
                best_val_stage = None
                _ts_print(
                    f"[STAGE] Switching from {current_stage} to {active_stage}; resetting best validation tracker "
                    "because the optimization objective has changed.",
                )
            _ts_print(
                f"[STAGE] Entering {active_stage} stage: {_stage_description(active_stage)}",
            )
            _ts_print(
                f"[STAGE] Active loss weights ({active_stage}): "
                + json.dumps(active_loss_weights, indent=2, sort_keys=True),
            )
            current_stage = active_stage
            if run_physics and not full_stage_precompile_done:
                _precompile_full_stage_steps(
                    state=state,
                    train_probe_batches=train_probe_batches,
                    val_probe_batches=val_probe_batches,
                    train_step_cache=train_step_cache,
                    eval_step_cache=eval_step_cache,
                    adapter=adapter,
                    normalizers=normalizers,
                    loss_weights=active_loss_weights,
                    deviation_learning=bool(args.DeviationLearning),
                    cop_mask=bool(args.cop_mask),
                    use_contact_weighting=bool(args.use_contact_weighting),
                    contact_weight_multiplier=float(args.contact_weight_multiplier),
                    prefetch_device=prefetch_device,
                    full_stage_cache_limit=int(args.full_stage_cache_limit),
                    precompile_max_groups=int(args.full_stage_precompile_max_groups),
                    ordered_probe_keys=full_stage_structure_order,
                )
                _trim_compiled_probe_batches(
                    train_probe_batches=train_probe_batches,
                    val_probe_batches=val_probe_batches,
                    train_step_cache=train_step_cache,
                    eval_step_cache=eval_step_cache,
                )
                gc.collect()
                full_stage_precompile_done = True

        if run_physics and args.check_mjx_gradients and not mjx_smoke_test_ran:
            debug_xml_path = _resolve_batch_xml_path(train_batches[0])
            if not debug_xml_path or not adapter.available:
                raise RuntimeError("Cannot run the MJX gradient smoke test without an available runtime XML and MJX adapter.")
            debug_context = adapter.get_jit_context(debug_xml_path)
            debug_runner = adapter.get_runner(debug_xml_path)
            debug_reconstructor = adapter.get_reconstructor(debug_xml_path)
            grad_stats = _run_mjx_gradient_smoke_test(
                batch=sample_batch,
                normalizers=normalizers,
                physics_context=debug_context,
                physics_runner=debug_runner,
                kinematics_reconstructor=debug_reconstructor,
                deviation_learning=bool(args.DeviationLearning),
                cop_mask=bool(args.cop_mask),
                use_contact_weighting=bool(args.use_contact_weighting),
                contact_weight_multiplier=float(args.contact_weight_multiplier),
                required_channels=_required_mjx_gradient_channels(full_loss_weights),
            )
            _ts_print("[MJX] Gradient smoke test passed.")
            _ts_print(
                "[MJX] Gradient stats: "
                + ", ".join(
                    f"{name}(finite={int(stats['finite'])}, absmax={stats['absmax']:.3e}, l2={stats['l2']:.3e})"
                    for name, stats in grad_stats.items()
                ),
            )
            weak_optional = [
                name
                for name in ("pos", "vel", "acc")
                if name not in _required_mjx_gradient_channels(full_loss_weights)
                and (grad_stats[name]["finite"] < 0.5 or grad_stats[name]["absmax"] <= 1e-12)
            ]
            if weak_optional:
                _ts_print(
                    "[MJX] Optional gradient channels were weak on the probe batch: "
                    + ", ".join(weak_optional)
                )
            mjx_smoke_test_ran = True

        _ts_print(
            f"[PHASE] Epoch {epoch} starting training stage={active_stage} "
            f"({_stage_description(active_stage)}) ({train_total_batches} batch(es))."
        )
        train_metrics: Dict[str, List[float]] = defaultdict(list)
        last_train_raw_batch: Optional[Dict[str, Any]] = None
        train_loader = _iter_prefetched_batches(
            train_batcher.iter_batches(),
            normalizers=normalizers,
            device=prefetch_device,
            prefetch_batches=int(args.prefetch_batches),
        )
        train_step_count = 0
        for step, prefetched in enumerate(train_loader):
            train_step_count = step + 1
            raw_batch = prefetched["raw_batch"]
            jit_batch = prefetched["jit_batch"]
            last_train_raw_batch = raw_batch
            batch_shape = prefetched["batch_shape"]
            if batch_shape not in warned_train_batch_shapes:
                warned_train_batch_shapes.add(batch_shape)
                _ts_print(f"[JIT] Train batch shape observed: {batch_shape}")
                if batch_shape[0] != int(args.batch_size):
                    _ts_print(
                        f"[JIT] Train batch size {batch_shape[0]} differs from requested batch_size={int(args.batch_size)}. "
                        "This can trigger an extra XLA compile.",
                    )
            xml_path = _resolve_batch_xml_path(raw_batch)
            if run_physics and xml_path and adapter.available:
                structure_key = adapter.get_structure_key(xml_path)
                cache_key = f"full::{structure_key}"
                physics_context = adapter.get_jit_context(xml_path)
            elif run_physics:
                structure_key = "full_detached_fallback"
                cache_key = "full::detached_fallback"
                physics_context = {}
            else:
                structure_key = "warmup_direct_only"
                cache_key = "warmup::direct_only"
                physics_context = {}
            if run_physics and structure_key not in warned_train_structure_keys:
                warned_train_structure_keys.add(structure_key)
                _ts_print(
                    f"[WARN] Full-stage batch entering structure group '{structure_key}' "
                    f"at epoch={epoch} step={step + 1} subject={raw_batch.get('subject', 'unknown')} "
                    f"xml={xml_path}. A new structure group can trigger a fresh MJX/JIT compile and a memory spike."
                )
                compile_ahead_keys = _compile_ahead_structure_keys(
                    anchor_structure_key=structure_key,
                    structure_order=full_stage_structure_order,
                    compile_ahead_groups=compile_ahead_groups,
                )
                if compile_ahead_keys:
                    _precompile_full_stage_steps(
                        state=state,
                        train_probe_batches=train_probe_batches,
                        val_probe_batches=val_probe_batches,
                        train_step_cache=train_step_cache,
                        eval_step_cache=eval_step_cache,
                        adapter=adapter,
                        normalizers=normalizers,
                        loss_weights=active_loss_weights,
                        deviation_learning=bool(args.DeviationLearning),
                        cop_mask=bool(args.cop_mask),
                        use_contact_weighting=bool(args.use_contact_weighting),
                        contact_weight_multiplier=float(args.contact_weight_multiplier),
                        prefetch_device=prefetch_device,
                        full_stage_cache_limit=int(args.full_stage_cache_limit),
                        precompile_max_groups=0,
                        ordered_probe_keys=compile_ahead_keys,
                        reason=f"compile-ahead from structure '{structure_key}'",
                    )
                    _trim_compiled_probe_batches(
                        train_probe_batches=train_probe_batches,
                        val_probe_batches=val_probe_batches,
                        train_step_cache=train_step_cache,
                        eval_step_cache=eval_step_cache,
                    )
            if cache_key not in train_step_cache:
                physics_runner = adapter.get_runner(xml_path) if run_physics and xml_path and adapter.available else None
                kinematics_reconstructor = (
                    adapter.get_reconstructor(xml_path) if run_physics and xml_path and adapter.available else None
                )
                if run_physics:
                    if xml_path and adapter.available:
                        runtime_xml = adapter.get_runtime_xml_path(xml_path)
                        _ts_print(
                            f"[JIT] Compiling new train_step for structure group: {structure_key} "
                            f"(sample runtime xml: {runtime_xml})",
                        )
                    else:
                        _ts_print(
                            "[JIT] Compiling new train_step for full stage without MJX adapter context; "
                            "using detached fallback physics terms.",
                        )
                else:
                    _ts_print("[JIT] Compiling new train_step for warmup stage: direct losses only (no MJX).")
                train_step_cache[cache_key] = make_train_step(
                    normalizers=normalizers,
                    loss_weights=active_loss_weights,
                    deviation_learning=bool(args.DeviationLearning),
                    cop_mask=bool(args.cop_mask),
                    use_contact_weighting=bool(args.use_contact_weighting),
                    contact_weight_multiplier=float(args.contact_weight_multiplier),
                    physics_runner=physics_runner,
                    kinematics_reconstructor=kinematics_reconstructor,
                    run_physics=run_physics,
                )
            _touch_step_cache(train_step_cache, cache_key)
            if run_physics and cache_key.startswith("full::"):
                _evict_full_stage_step_caches(
                    train_step_cache=train_step_cache,
                    eval_step_cache=eval_step_cache,
                    keep_cache_key=cache_key,
                    full_stage_cache_limit=int(args.full_stage_cache_limit),
                )
                _drop_companion_cache_entries(train_step_cache, train_kinematic_equiv_cache)
                _drop_companion_cache_entries(eval_step_cache, eval_kinematic_equiv_cache)
            dropout_rng = jax.random.fold_in(jax.random.PRNGKey(epoch), step)
            should_probe_kinematic_equiv = (
                bool(args.log_kinematic_equiv)
                and run_physics
                and int(args.kinematic_equiv_interval) > 0
                and ((step + 1) % int(args.kinematic_equiv_interval) == 0)
            )
            if should_probe_kinematic_equiv:
                if cache_key not in train_kinematic_equiv_cache:
                    physics_runner = adapter.get_runner(xml_path) if run_physics and xml_path and adapter.available else None
                    kinematics_reconstructor = (
                        adapter.get_reconstructor(xml_path) if run_physics and xml_path and adapter.available else None
                    )
                    train_kinematic_equiv_cache[cache_key] = make_kinematic_equiv_probe_step(
                        normalizers=normalizers,
                        loss_weights=active_loss_weights,
                        deviation_learning=bool(args.DeviationLearning),
                        cop_mask=bool(args.cop_mask),
                        physics_runner=physics_runner,
                        kinematics_reconstructor=kinematics_reconstructor,
                        run_physics=run_physics,
                        train_mode=True,
                    )
                _touch_step_cache(train_kinematic_equiv_cache, cache_key)
                probe_metrics = train_kinematic_equiv_cache[cache_key](state, jit_batch, physics_context, dropout_rng)
                probe_metric_values = {key: float(value) for key, value in jax.device_get(probe_metrics).items()}
                for key, value in probe_metric_values.items():
                    train_metrics[key].append(value)
            state, direct_loss, metrics = train_step_cache[cache_key](state, jit_batch, physics_context, dropout_rng)
            host_metric_values = jax.device_get({"total_loss": direct_loss, **metrics})
            batch_metric_values = {key: float(value) for key, value in host_metric_values.items()}
            total_loss = batch_metric_values["total_loss"]
            if batch_metric_values.get("update_skipped", 0.0) > 0.5:
                _ts_print(
                    f"[NUMERIC] Skipped non-finite train update at epoch={epoch} step={step + 1} "
                    f"subject={raw_batch.get('subject', 'unknown')} xml={xml_path} "
                    f"params_finite_pre={batch_metric_values.get('params_finite_pre', float('nan')):.0f} "
                    f"forward_finite={batch_metric_values.get('forward_finite', float('nan')):.0f} "
                    f"grads_finite={batch_metric_values.get('grads_finite', float('nan')):.0f} "
                    f"state_finite_post={batch_metric_values.get('state_finite_post', float('nan')):.0f} "
                    f"grad_global_norm={batch_metric_values.get('grad_global_norm', float('nan')):.4g}",
                )
                for summary in (
                    _summarize_array("input", raw_batch.get("input")),
                    _summarize_array("static_context", raw_batch.get("static_context")),
                    _summarize_array("cop", raw_batch.get("cop")),
                    _summarize_array("grf", raw_batch.get("grf")),
                    _summarize_array("moments", raw_batch.get("moments")),
                    _summarize_array("contactBoolean", raw_batch.get("contactBoolean")),
                    _summarize_array("pos_gt", raw_batch.get("pos_gt")),
                    _summarize_array("vel_gt", raw_batch.get("vel_gt")),
                    _summarize_array("acc_gt", raw_batch.get("acc_gt")),
                    _summarize_array("qpos_mjx_input", raw_batch.get("qpos_mjx_input")),
                    _summarize_array("qvel_mjx_input", raw_batch.get("qvel_mjx_input")),
                    _summarize_array("qacc_mjx_input", raw_batch.get("qacc_mjx_input")),
                    _summarize_array("qfrc_inverse_gt", raw_batch.get("qfrc_inverse_gt")),
                    _summarize_array("jacp", raw_batch.get("jacp")),
                    _summarize_array("jacr", raw_batch.get("jacr")),
                    _summarize_array("gt_rot_w_to_ga", raw_batch.get("gt_rot_w_to_ga")),
                ):
                    _ts_print(f"[NUMERIC] {summary}")
            bad_metrics = _nonfinite_metric_names(batch_metric_values)
            if bad_metrics:
                _ts_print(
                    f"[NUMERIC] Non-finite train metrics at epoch={epoch} step={step + 1} "
                    f"subject={raw_batch.get('subject', 'unknown')} xml={xml_path}",
                )
                _ts_print(
                    "[NUMERIC] Offending terms: "
                    + ", ".join(f"{name}={batch_metric_values[name]}" for name in bad_metrics),
                )
                raise FloatingPointError(
                    f"Non-finite train metrics detected at epoch {epoch} step {step + 1}: {', '.join(bad_metrics)}"
                )

            train_metrics["total_loss"].append(total_loss)
            for key, value in batch_metric_values.items():
                if key == "total_loss":
                    continue
                train_metrics[key].append(value)

            if (step + 1) % max(1, int(args.log_interval)) == 0:
                _print_loss_report(
                    f"Epoch {epoch} Step {step + 1}/{train_total_batches}",
                    train_metrics,
                    active_loss_weights,
                )
            if run_physics:
                _periodic_runtime_cleanup(
                    step_index=step + 1,
                    every_n_steps=int(args.clear_runtime_cache_every),
                    stage_name="train",
                    train_step_cache=train_step_cache,
                    eval_step_cache=eval_step_cache,
                    active_cache_key=cache_key,
                    full_stage_cache_limit=int(args.full_stage_cache_limit),
                    low_ram_available_gb=float(args.low_ram_available_gb),
                    low_ram_available_frac=float(args.low_ram_available_frac),
                )

        _ts_print(
            f"[PHASE] Epoch {epoch} training complete after {train_step_count}/{train_total_batches} batch(es). "
            f"Starting validation stage={active_stage} ({val_total_batches} batch(es)).",
        )
        val_metrics: Dict[str, List[float]] = defaultdict(list)
        last_val_raw_batch: Optional[Dict[str, Any]] = None
        val_loader = _iter_prefetched_batches(
            val_batcher.iter_batches(),
            normalizers=normalizers,
            device=prefetch_device,
            prefetch_batches=int(args.prefetch_batches),
        )
        val_step_count = 0
        for val_step, prefetched in enumerate(val_loader, start=1):
            val_step_count = val_step
            raw_batch = prefetched["raw_batch"]
            jit_batch = prefetched["jit_batch"]
            last_val_raw_batch = raw_batch
            batch_shape = prefetched["batch_shape"]
            if batch_shape not in warned_val_batch_shapes:
                warned_val_batch_shapes.add(batch_shape)
                _ts_print(f"[JIT] Val batch shape observed: {batch_shape}")
                if batch_shape[0] != int(args.batch_size):
                    _ts_print(
                        f"[JIT] Val batch size {batch_shape[0]} differs from requested batch_size={int(args.batch_size)}. "
                        "This can trigger an extra XLA compile.",
                    )
            xml_path = _resolve_batch_xml_path(raw_batch)
            if run_physics and xml_path and adapter.available:
                structure_key = adapter.get_structure_key(xml_path)
                cache_key = f"full::{structure_key}"
                physics_context = adapter.get_jit_context(xml_path)
            elif run_physics:
                structure_key = "full_detached_fallback"
                cache_key = "full::detached_fallback"
                physics_context = {}
            else:
                structure_key = "warmup_direct_only"
                cache_key = "warmup::direct_only"
                physics_context = {}
            if run_physics and structure_key not in warned_val_structure_keys:
                warned_val_structure_keys.add(structure_key)
                _ts_print(
                    f"[WARN] Full-stage validation entering structure group '{structure_key}' "
                    f"at epoch={epoch} subject={raw_batch.get('subject', 'unknown')} xml={xml_path}. "
                    "A new structure group can trigger a fresh MJX/JIT compile and a memory spike."
                )
                compile_ahead_keys = _compile_ahead_structure_keys(
                    anchor_structure_key=structure_key,
                    structure_order=full_stage_structure_order,
                    compile_ahead_groups=compile_ahead_groups,
                )
                if compile_ahead_keys:
                    _precompile_full_stage_steps(
                        state=state,
                        train_probe_batches=train_probe_batches,
                        val_probe_batches=val_probe_batches,
                        train_step_cache=train_step_cache,
                        eval_step_cache=eval_step_cache,
                        adapter=adapter,
                        normalizers=normalizers,
                        loss_weights=active_loss_weights,
                        deviation_learning=bool(args.DeviationLearning),
                        cop_mask=bool(args.cop_mask),
                        use_contact_weighting=bool(args.use_contact_weighting),
                        contact_weight_multiplier=float(args.contact_weight_multiplier),
                        prefetch_device=prefetch_device,
                        full_stage_cache_limit=int(args.full_stage_cache_limit),
                        precompile_max_groups=0,
                        ordered_probe_keys=compile_ahead_keys,
                        reason=f"compile-ahead from validation structure '{structure_key}'",
                    )
                    _trim_compiled_probe_batches(
                        train_probe_batches=train_probe_batches,
                        val_probe_batches=val_probe_batches,
                        train_step_cache=train_step_cache,
                        eval_step_cache=eval_step_cache,
                    )
            if cache_key not in eval_step_cache:
                physics_runner = adapter.get_runner(xml_path) if run_physics and xml_path and adapter.available else None
                kinematics_reconstructor = (
                    adapter.get_reconstructor(xml_path) if run_physics and xml_path and adapter.available else None
                )
                if run_physics:
                    if xml_path and adapter.available:
                        runtime_xml = adapter.get_runtime_xml_path(xml_path)
                        _ts_print(
                            f"[JIT] Compiling new eval_step for structure group: {structure_key} "
                            f"(sample runtime xml: {runtime_xml})",
                        )
                    else:
                        _ts_print(
                            "[JIT] Compiling new eval_step for full stage without MJX adapter context; "
                            "using detached fallback physics terms.",
                        )
                else:
                    _ts_print("[JIT] Compiling new eval_step for warmup stage: direct losses only (no MJX).")
                eval_step_cache[cache_key] = make_eval_step(
                    normalizers=normalizers,
                    loss_weights=active_loss_weights,
                    deviation_learning=bool(args.DeviationLearning),
                    cop_mask=bool(args.cop_mask),
                    use_contact_weighting=bool(args.use_contact_weighting),
                    contact_weight_multiplier=float(args.contact_weight_multiplier),
                    physics_runner=physics_runner,
                    kinematics_reconstructor=kinematics_reconstructor,
                    run_physics=run_physics,
                )
            _touch_step_cache(eval_step_cache, cache_key)
            if run_physics and cache_key.startswith("full::"):
                _evict_full_stage_step_caches(
                    train_step_cache=train_step_cache,
                    eval_step_cache=eval_step_cache,
                    keep_cache_key=cache_key,
                    full_stage_cache_limit=int(args.full_stage_cache_limit),
                )
                _drop_companion_cache_entries(train_step_cache, train_kinematic_equiv_cache)
                _drop_companion_cache_entries(eval_step_cache, eval_kinematic_equiv_cache)
            total_loss, metrics = eval_step_cache[cache_key](state, jit_batch, physics_context)
            host_metric_values = jax.device_get({"total_loss": total_loss, **metrics})
            batch_metric_values = {key: float(value) for key, value in host_metric_values.items()}
            total_value = batch_metric_values["total_loss"]
            if batch_metric_values.get("forward_finite", 1.0) < 0.5:
                _ts_print(
                    f"[NUMERIC] Non-finite val forward sanitized at epoch={epoch} "
                    f"subject={raw_batch.get('subject', 'unknown')} xml={xml_path}",
                )
            bad_metrics = _nonfinite_metric_names(batch_metric_values)
            if bad_metrics:
                _ts_print(
                    f"[NUMERIC] Non-finite val metrics at epoch={epoch} "
                    f"subject={raw_batch.get('subject', 'unknown')} xml={xml_path}",
                )
                _ts_print(
                    "[NUMERIC] Offending terms: "
                    + ", ".join(f"{name}={batch_metric_values[name]}" for name in bad_metrics),
                )
                raise FloatingPointError(
                    f"Non-finite val metrics detected at epoch {epoch}: {', '.join(bad_metrics)}"
                )
            val_metrics["total_loss"].append(total_value)
            for key, value in batch_metric_values.items():
                if key == "total_loss":
                    continue
                val_metrics[key].append(value)
            should_probe_kinematic_equiv = bool(args.log_kinematic_equiv) and run_physics and val_step == 1
            if should_probe_kinematic_equiv:
                if cache_key not in eval_kinematic_equiv_cache:
                    physics_runner = adapter.get_runner(xml_path) if run_physics and xml_path and adapter.available else None
                    kinematics_reconstructor = (
                        adapter.get_reconstructor(xml_path) if run_physics and xml_path and adapter.available else None
                    )
                    eval_kinematic_equiv_cache[cache_key] = make_kinematic_equiv_probe_step(
                        normalizers=normalizers,
                        loss_weights=active_loss_weights,
                        deviation_learning=bool(args.DeviationLearning),
                        cop_mask=bool(args.cop_mask),
                        physics_runner=physics_runner,
                        kinematics_reconstructor=kinematics_reconstructor,
                        run_physics=run_physics,
                        train_mode=False,
                    )
                _touch_step_cache(eval_kinematic_equiv_cache, cache_key)
                probe_metrics = eval_kinematic_equiv_cache[cache_key](state, jit_batch, physics_context)
                probe_metric_values = {key: float(value) for key, value in jax.device_get(probe_metrics).items()}
                for key, value in probe_metric_values.items():
                    val_metrics[key].append(value)
            if run_physics:
                _periodic_runtime_cleanup(
                    step_index=val_step,
                    every_n_steps=int(args.clear_runtime_cache_every),
                    stage_name="val",
                    train_step_cache=train_step_cache,
                    eval_step_cache=eval_step_cache,
                    active_cache_key=cache_key,
                    full_stage_cache_limit=int(args.full_stage_cache_limit),
                    low_ram_available_gb=float(args.low_ram_available_gb),
                    low_ram_available_frac=float(args.low_ram_available_frac),
                )

        _ts_print(
            f"[PHASE] Epoch {epoch} validation complete after {val_step_count}/{val_total_batches} batch(es).",
        )
        train_total = float(np.mean(train_metrics["total_loss"])) if train_metrics["total_loss"] else float("nan")
        val_total = float(np.mean(val_metrics["total_loss"])) if val_metrics["total_loss"] else float("nan")
        _ts_print(f"Epoch {epoch} Summary (stage={active_stage})")
        _print_loss_report("  Train", train_metrics, active_loss_weights)
        _print_loss_report("  Val", val_metrics, active_loss_weights)

        should_plot_predictions = False
        if args.save_final_predictions_only:
            should_plot_predictions = epoch == int(args.epochs)
        elif int(args.vis_interval) > 0:
            should_plot_predictions = (epoch % int(args.vis_interval)) == 0

        if should_plot_predictions:
            try:
                train_vis_batch = last_train_raw_batch if last_train_raw_batch is not None else next(iter(train_batcher.iter_batches()))
                val_vis_batch = last_val_raw_batch if last_val_raw_batch is not None else next(iter(val_batcher.iter_batches()))
                train_vis_prepared = _prepare_batch(train_vis_batch, normalizers)
                val_vis_prepared = _prepare_batch(val_vis_batch, normalizers)
                train_vis_pred = jax.device_get(
                    state.apply_fn(
                        {"params": state.params},
                        train_vis_prepared["input"],
                        train_vis_prepared["static_context"],
                        train=False,
                    )
                )
                val_vis_pred = jax.device_get(
                    state.apply_fn(
                        {"params": state.params},
                        val_vis_prepared["input"],
                        val_vis_prepared["static_context"],
                        train=False,
                    )
                )
                plot_path = plot_modq_predictions(
                    train_vis_prepared,
                    train_vis_pred,
                    val_vis_prepared,
                    val_vis_pred,
                    normalizers,
                    epoch,
                    output_dir,
                    deviation_learning=bool(args.DeviationLearning),
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    loss_weights=active_loss_weights,
                )
                _ts_print(f"[VIS] Saved prediction summary plot to {plot_path}")
            except Exception as exc:
                _ts_print(f"[VIS] Failed to generate prediction summary plot for epoch {epoch}: {exc}")

        wandb_payload = {"epoch": epoch, "training_stage": active_stage}
        wandb_payload.update(_wandb_epoch_metrics("train", train_metrics, active_loss_weights))
        wandb_payload.update(_wandb_epoch_metrics("val", val_metrics, active_loss_weights))
        wandb_logger.log(wandb_payload, step=epoch)

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_val_stage = active_stage
            _save_checkpoint(
                output_dir,
                state,
                normalizers,
                train_trials=train_trials,
                val_trials=val_trials,
                best_val_loss=best_val_loss,
                args=args,
                input_dim=input_dim,
                static_dim=static_dim,
                output_dim=output_dim,
                active_loss_weights=active_loss_weights,
                full_loss_weights=full_loss_weights,
                warmup_loss_weights=warmup_loss_weights,
                active_stage=active_stage,
            )
            _ts_print(f"Saved new best checkpoint with val loss {best_val_loss:.4f} (stage={best_val_stage})")

        gc.collect()

    _ts_print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
