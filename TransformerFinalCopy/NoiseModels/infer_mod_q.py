"""Inference script for the mod_q checkpoint family.

This is a lightweight companion to `train_mod_q.py`.
It expects explicit checkpoint metadata and keeps the fixed mod_q schema:
COP(4) + GRF(6) + GRM(2) + Contact(2) + clean pos(16) + vel(19) + acc(19),
then derives the coupled MJX state from those predictions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from data_loader import (
    TrialDataLoader,
    build_window_supervision_mask,
    load_single_trial,
    select_pos_input_columns,
    build_window_start_indices,
)
from infer import (
    LEFT_STANCE_THRESHOLD_N,
    FilterPostInfer,
    _compute_average_bilateral_stance_mae,
    _compute_average_mae_per_dof,
    _extract_stance_cop_mae_percent_height,
    _mask_prediction_dict_for_display,
    _masked_mae,
    _masked_mean_diff,
    _masked_rmse,
    _masked_rmse_per_channel,
    _normalize_evaluation_mask,
    analyze_stance_phase_torques,
    apply_butterworth_filter_masked,
    build_bilateral_stance_mae_report,
    create_all_dofs_plot,
    create_error_distribution_plot,
    create_mae_boxplots,
    create_summary_dashboard,
    create_timeseries_plot,
    find_trial as infer_find_trial,
    get_dof_names,
    get_left_stance_mask,
    get_selected_left_stance_dof_indices,
    make_publication_plots,
)
from runtime_model_utils import resolve_modq_runtime_xml


MOD_Q_MODEL_TYPE = "mod_q"
MOD_Q_OUTPUT_SCHEMA = (
    ("cop", 4),
    ("grf", 6),
    ("moments", 2),
    ("contact", 2),
    ("pos", 16),
    ("vel", 19),
    ("acc", 19),
)
MOD_Q_OUTPUT_DIM = int(sum(width for _, width in MOD_Q_OUTPUT_SCHEMA))
MOD_Q_QPRIME_LAYOUT = ("pos", "vel", "acc")
MOD_Q_INPUT_BLOCKS = (
    ("pelvis_rot", 6),
    ("pos", 16),
    ("vel", None),
    ("acc", 19),
    ("com_r", 3),
    ("com_l", 3),
    ("com_accel", 3),
    ("foot_progression_angle", 2),
    ("calcn_to_floor_angle", 2),
)
MOD_Q_POS_MTP_REMOVE_IDXS = (8, 14)
MOD_Q_VEL_ACC_MTP_REMOVE_IDXS = (11, 17)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return bool(value)


def _resolve_shared_imports():
    try:
        from mod_q_shared import (  # type: ignore
            MOD_Q_OUTPUT_SCHEMA as SHARED_OUTPUT_SCHEMA,
            MOD_Q_QPRIME_LAYOUT as SHARED_QPRIME_LAYOUT,
            build_mod_q_input_features,
            build_mod_q_static_context,
            build_mod_q_model,
            decode_mod_q_predictions,
            mod_q_physics_adapter,
            mod_q_checkpoint_metadata,
            reconstruct_mod_q_state,
            Normalizer,
            rotation_geodesic_summary_deg,
        )
        return {
            "shared": True,
            "output_schema": SHARED_OUTPUT_SCHEMA,
            "qprime_layout": SHARED_QPRIME_LAYOUT,
            "build_input": build_mod_q_input_features,
            "build_static": build_mod_q_static_context,
            "build_model": build_mod_q_model,
            "decode": decode_mod_q_predictions,
            "physics": mod_q_physics_adapter,
            "checkpoint_metadata": mod_q_checkpoint_metadata,
            "reconstruct_state": reconstruct_mod_q_state,
            "Normalizer": Normalizer,
            "rotation_summary": rotation_geodesic_summary_deg,
        }
    except Exception:
        pass

    class Normalizer:
        def __init__(self, mean: np.ndarray, std: np.ndarray):
            self.mean = np.asarray(mean)
            self.std = np.asarray(std)

        def normalize(self, x):
            return (x - self.mean) / self.std

        def unnormalize(self, x):
            return x * self.std + self.mean

    def build_mod_q_input_features(data: Dict[str, np.ndarray], include_pelvis_euler: bool = True):
        parts = [
            data["pelvis_rot"],
            data["pos"],
            data["vel"],
            data["acc"],
            data["com_r"],
            data["com_l"],
            data["com_accel"],
            data["foot_progression_angle"],
            data["calcn_to_floor_angle"],
        ]
        return np.concatenate(parts, axis=1)

    def build_mod_q_static_context(data: Dict[str, np.ndarray]) -> np.ndarray:
        gender_val = data.get("gender", 0.5)
        gender_arr = np.asarray(gender_val, dtype=np.float32).reshape(-1)
        return np.array(
            [
                float(np.asarray(data["height"]).reshape(-1)[0]),
                float(np.asarray(data["mass"]).reshape(-1)[0]),
                float(gender_arr[0] if gender_arr.size else 0.5),
                float(np.asarray(data["patient_size"]).reshape(-1)[0]),
                float(np.asarray(data["patient_size"]).reshape(-1)[1]),
                float(np.asarray(data["patient_size"]).reshape(-1)[2]),
                float(np.asarray(data["patient_size"]).reshape(-1)[3]),
                float(np.asarray(data["forward_vel"]).reshape(-1)[0]),
            ],
            dtype=np.float32,
        )

    def _normalize_world_vec(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < eps:
            return np.zeros_like(vec)
        return vec / norm

    def _compose_world_to_ground_aligned_single_np(rot_w_to_body: np.ndarray) -> np.ndarray:
        """Match ProcessData.py / train_mod_q.py ground-aligned calcaneus rotation construction."""
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
            ff_out = nn.Dense(self.ff_dim)(x)
            ff_out = nn.gelu(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            ff_out = nn.Dense(self.d_model)(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            return residual + ff_out

    class ModQTransformer(nn.Module):
        input_dim: int
        static_dim: int = 8
        output_dim: int = MOD_Q_OUTPUT_DIM
        d_model: int = 256
        num_heads: int = 4
        num_layers: int = 4
        ff_dim: int = 1024
        dropout_rate: float = 0.1
        use_cnn: bool = True
        cnn_num_layers: int = 2
        cnn_kernel_sizes: tuple = (3, 5)

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
                x = residual + x
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
            return nn.Dense(self.output_dim)(x)

    def build_mod_q_model(input_dim: int, static_dim: int, model_cfg: Dict[str, Any]):
        return ModQTransformer(
            input_dim=input_dim,
            static_dim=static_dim,
            output_dim=int(model_cfg.get("output_dim", MOD_Q_OUTPUT_DIM)),
            d_model=int(model_cfg.get("d_model", 256)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            ff_dim=int(model_cfg.get("ff_dim", 1024)),
            dropout_rate=float(model_cfg.get("dropout_rate", 0.1)),
            use_cnn=_coerce_bool(model_cfg.get("use_cnn", True), default=True),
            cnn_num_layers=int(model_cfg.get("cnn_num_layers", 2)),
            cnn_kernel_sizes=tuple(int(v) for v in model_cfg.get("cnn_kernel_sizes", (3, 5))),
        )

    def decode_mod_q_predictions(output_np: np.ndarray) -> Dict[str, np.ndarray]:
        out = np.asarray(output_np, dtype=np.float32)
        idx = 0
        decoded = {}
        for name, width in MOD_Q_OUTPUT_SCHEMA:
            decoded[name] = out[..., idx : idx + width]
            idx += width
        decoded["contact"] = 1.0 / (1.0 + np.exp(-decoded["contact"]))
        return decoded

    def mod_q_checkpoint_metadata() -> Dict[str, Any]:
        return {
            "model_type": MOD_Q_MODEL_TYPE,
            "output_schema": [list(item) for item in MOD_Q_OUTPUT_SCHEMA],
            "qprime_layout": list(MOD_Q_QPRIME_LAYOUT),
            "input_feature_blocks": [
                ["pelvis_rot", 6],
                ["pos", 16],
                ["vel", "data_loader_vel_dim"],
                ["acc", 21],
                ["com_r", 3],
                ["com_l", 3],
                ["com_accel", 3],
                ["foot_progression_angle", 2],
                ["calcn_to_floor_angle", 2],
            ],
            "subject_grouped_batches": True,
            "UseNoised": True,
            "includePelvisEuler": True,
            "PredictJacobian": False,
            "DeviationLearning": False,
        }

    def _load_mujoco_module():
        try:
            import mujoco  # type: ignore

            return mujoco
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(
                "mujoco is required for mod_q physics inference but could not be imported."
            ) from exc

    def _resolve_subject_xml(trial_path: Path, data: Dict[str, Any]) -> Path:
        xml_hint = data.get("subject_model_xml")
        if xml_hint:
            xml_path = Path(xml_hint)
            if xml_path.exists():
                return xml_path
        trial_root = trial_path.parent
        fixed_xml = trial_root / "MyosuiteModel_FIXED.xml"
        if fixed_xml.exists():
            return fixed_xml
        raw_xml = trial_root / "MyosuiteModel.xml"
        if raw_xml.exists():
            return raw_xml
        raise FileNotFoundError(f"Could not resolve subject XML for {trial_path}")

    def mod_q_physics_adapter(
        qprime: Dict[str, np.ndarray],
        cop_phys: np.ndarray,
        grf_phys: np.ndarray,
        moments_phys: np.ndarray,
        ankle_heights: np.ndarray,
        rot_w_to_ga_gt: Optional[np.ndarray] = None,
        xml_path: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        mujoco = _load_mujoco_module()
        if xml_path is None:
            raise ValueError("xml_path is required for mod_q physics inference")

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        qpos = np.asarray(qprime["qpos"], dtype=np.float64)
        qvel = np.asarray(qprime["qvel"], dtype=np.float64)
        qacc = np.asarray(qprime["qacc"], dtype=np.float64)
        T = int(qpos.shape[0])
        nv = int(model.nv)

        rot_w_to_ga = np.zeros((T, 2, 3, 3), dtype=np.float64)
        jacp = np.zeros((T, 2, 3, nv), dtype=np.float64)
        jacr = np.zeros((T, 2, 3, nv), dtype=np.float64)
        qfrc_constraint = np.zeros((T, nv), dtype=np.float64)
        qfrc_inverse = np.zeros((T, nv), dtype=np.float64)
        qfrc_grf = np.zeros((T, nv), dtype=np.float64)

        calcn_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
        calcn_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_l")
        if min(calcn_r_id, calcn_l_id) < 0:
            raise ValueError("Could not resolve calcn_r/calcn_l in the MuJoCo model")

        data = mujoco.MjData(model)
        for t in range(T):
            data.qpos[:] = qpos[t]
            data.qvel[:] = qvel[t]
            data.qacc[:] = qacc[t]
            mujoco.mj_forward(model, data)
            mujoco.mj_inverse(model, data)

            qfrc_inverse_raw = np.asarray(data.qfrc_inverse, dtype=np.float64)
            qfrc_constraint[t] = np.asarray(data.qfrc_constraint, dtype=np.float64)
            qfrc_inverse[t] = qfrc_inverse_raw + qfrc_constraint[t]
            for foot_idx, body_id in enumerate((calcn_r_id, calcn_l_id)):
                jp = np.zeros((3, nv), dtype=np.float64)
                jr = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacBody(model, data, jp, jr, body_id)
                jacp[t, foot_idx] = jp
                jacr[t, foot_idx] = jr

                body_rot_w_to_b = data.xmat[body_id].reshape(3, 3).T
                rot_w_to_ga[t, foot_idx] = _compose_world_to_ground_aligned_single_np(body_rot_w_to_b)

        rot_ga_to_w = np.swapaxes(rot_w_to_ga, -1, -2)
        cop_world = np.zeros((T, 2, 3), dtype=np.float64)
        for side in range(2):
            if side == 0:
                cop_ga = np.column_stack([cop_phys[:, 0], ankle_heights[:, 0], cop_phys[:, 1]])
            else:
                cop_ga = np.column_stack([cop_phys[:, 2], ankle_heights[:, 1], cop_phys[:, 3]])
            cop_world[:, side] = np.einsum("tij,tj->ti", rot_ga_to_w[:, side], cop_ga)

        f_r = grf_phys[:, 0:3]
        f_l = grf_phys[:, 3:6]
        m_r = np.column_stack([
            np.zeros(T, dtype=np.float64),
            np.zeros(T, dtype=np.float64),
            moments_phys[:, 0],
        ])
        m_l = np.column_stack([
            np.zeros(T, dtype=np.float64),
            np.zeros(T, dtype=np.float64),
            moments_phys[:, 1],
        ])

        m_total_r = m_r + np.cross(cop_world[:, 0], f_r)
        m_total_l = m_l + np.cross(cop_world[:, 1], f_l)

        tau_r = np.einsum("tji,tj->ti", jacp[:, 0], f_r) + np.einsum("tji,tj->ti", jacr[:, 0], m_total_r)
        tau_l = np.einsum("tji,tj->ti", jacp[:, 1], f_l) + np.einsum("tji,tj->ti", jacr[:, 1], m_total_l)
        tau_grf = tau_r + tau_l

        return {
            "rot_w_to_ga": rot_w_to_ga.astype(np.float32),
            "jacp": jacp.astype(np.float32),
            "jacr": jacr.astype(np.float32),
            "qfrc_constraint": qfrc_constraint.astype(np.float32),
            "qfrc_inverse": qfrc_inverse.astype(np.float32),
            "tau_grf": tau_grf.astype(np.float32),
            "full_id": (qfrc_inverse - tau_grf).astype(np.float32),
            "cop_world": cop_world.astype(np.float32),
        }

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
        raise RuntimeError("Shared mod_q reconstruction helpers could not be imported.")

    def rotation_geodesic_summary_deg(
        rot_a: Any,
        rot_b: Any,
        mask: Optional[Any] = None,
        *,
        xp=np,
        project: bool = True,
    ) -> Dict[str, Any]:
        del mask
        rot_a = xp.asarray(rot_a)
        rot_b = xp.asarray(rot_b)
        if project:
            flat_a = np.asarray(rot_a, dtype=np.float64).reshape((-1, 3, 3))
            flat_b = np.asarray(rot_b, dtype=np.float64).reshape((-1, 3, 3))

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

            rot_a = _project(flat_a).reshape(rot_a.shape)
            rot_b = _project(flat_b).reshape(rot_b.shape)

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
        angle_deg = xp.degrees(xp.arctan2(0.5 * xp.linalg.norm(sin_terms, axis=-1), cos_theta))
        overall_mean_deg = xp.mean(angle_deg)
        overall_rmse_deg = xp.sqrt(xp.mean(angle_deg**2))
        return {
            "overall_mean_deg": overall_mean_deg,
            "overall_rmse_deg": overall_rmse_deg,
            "right_mean_deg": xp.mean(angle_deg[..., 0]),
            "left_mean_deg": xp.mean(angle_deg[..., 1]),
            "mean_deg": overall_mean_deg,
            "rmse_deg": overall_rmse_deg,
        }

    return {
        "shared": False,
        "output_schema": MOD_Q_OUTPUT_SCHEMA,
        "qprime_layout": MOD_Q_QPRIME_LAYOUT,
        "build_input": build_mod_q_input_features,
            "build_static": build_mod_q_static_context,
            "build_model": build_mod_q_model,
            "decode": decode_mod_q_predictions,
            "physics": mod_q_physics_adapter,
            "checkpoint_metadata": mod_q_checkpoint_metadata,
            "reconstruct_state": reconstruct_mod_q_state,
            "Normalizer": Normalizer,
            "rotation_summary": rotation_geodesic_summary_deg,
    }


_IMPL = _resolve_shared_imports()
Normalizer = _IMPL["Normalizer"]
build_mod_q_input_features = _IMPL["build_input"]
build_mod_q_static_context = _IMPL["build_static"]
build_mod_q_model = _IMPL["build_model"]
decode_mod_q_predictions = _IMPL["decode"]
mod_q_physics_adapter = _IMPL["physics"]
mod_q_checkpoint_metadata = _IMPL["checkpoint_metadata"]
reconstruct_mod_q_state = _IMPL["reconstruct_state"]
rotation_geodesic_summary_deg = _IMPL["rotation_summary"]


def _resolve_subject_xml(trial_path: Path, data: Dict[str, Any]) -> Path:
    xml_hint = data.get("subject_model_xml")
    if xml_hint:
        hinted = Path(str(xml_hint))
        try:
            return resolve_modq_runtime_xml(hinted)
        except Exception:
            if hinted.exists():
                return hinted

    trial_root = trial_path.parent
    try:
        return resolve_modq_runtime_xml(trial_root)
    except Exception:
        fixed_xml = trial_root / "MyosuiteModel_FIXED.xml"
        if fixed_xml.exists():
            return fixed_xml
        raw_xml = trial_root / "MyosuiteModel.xml"
        if raw_xml.exists():
            return raw_xml
        raise FileNotFoundError(f"Could not resolve subject XML for {trial_path}")


def _window_starts(seq_len: int, window_size: int, stride: int) -> List[int]:
    return build_window_start_indices(seq_len=seq_len, window_size=window_size, stride=stride)


def _windowed_predict(
    model,
    params,
    input_z: np.ndarray,
    static_z: np.ndarray,
    window_size: int,
    stride: int,
    prediction_margin_frames: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    seq_len = int(len(input_z))
    starts = _window_starts(seq_len, window_size, stride)
    if not starts:
        raise ValueError("No inference windows could be constructed")

    window_outputs = []
    for start in starts:
        end = min(start + window_size, seq_len)
        if end - start < window_size:
            start = max(0, end - window_size)
            end = start + window_size
        x = jnp.asarray(input_z[start:end][None, ...])
        s = jnp.asarray(static_z[None, ...])
        out = model.apply({"params": params}, x, s, train=False)
        window_outputs.append(np.asarray(out[0], dtype=np.float32))

    output_dim = int(window_outputs[0].shape[-1])
    agg = np.zeros((seq_len, output_dim), dtype=np.float32)
    counts = np.zeros((seq_len, 1), dtype=np.float32)
    for start, out in zip(starts, window_outputs):
        end = min(start + window_size, seq_len)
        span = end - start
        kept_mask_window = build_window_supervision_mask(
            window_size=window_size,
            window_start_idx=start,
            trial_length=seq_len,
            prediction_margin_frames=prediction_margin_frames,
        )[:span].astype(np.float32)
        agg[start:end] += out[:span] * kept_mask_window
        counts[start:end] += kept_mask_window
    evaluation_mask = counts[:, 0] > 0.0
    stitched = np.zeros((seq_len, output_dim), dtype=np.float32)
    if np.any(evaluation_mask):
        stitched[evaluation_mask] = agg[evaluation_mask] / np.maximum(counts[evaluation_mask], 1.0)
    return stitched, counts.squeeze(-1), evaluation_mask, {
        "num_windows": int(len(starts)),
        "window_size": int(window_size),
        "stride": int(stride),
        "prediction_margin_frames": int(prediction_margin_frames),
        "evaluation_frame_count": int(np.sum(evaluation_mask)),
    }


def _truncate_timeseries_mapping(values: Dict[str, Any], length: int) -> Dict[str, Any]:
    truncated: Dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            truncated[key] = None
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] >= length:
            truncated[key] = value[:length]
            continue
        truncated[key] = value
    return truncated


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
    stats = {
        "predicted_vs_mocap": {
            "overall_mean_deg": float(pred_summary["overall_mean_deg"]),
            "overall_rmse_deg": float(pred_summary["overall_rmse_deg"]),
            "right_mean_deg": float(pred_summary["right_mean_deg"]),
            "right_rmse_deg": float(pred_summary["right_rmse_deg"]),
            "left_mean_deg": float(pred_summary["left_mean_deg"]),
            "left_rmse_deg": float(pred_summary["left_rmse_deg"]),
        },
        "processed_vs_mocap": {
            "overall_mean_deg": float(processed_summary["overall_mean_deg"]),
            "overall_rmse_deg": float(processed_summary["overall_rmse_deg"]),
            "right_mean_deg": float(processed_summary["right_mean_deg"]),
            "right_rmse_deg": float(processed_summary["right_rmse_deg"]),
            "left_mean_deg": float(processed_summary["left_mean_deg"]),
            "left_rmse_deg": float(processed_summary["left_rmse_deg"]),
        },
    }
    stats["improvement_pred_minus_processed"] = {
        "overall_mean_deg": float(stats["predicted_vs_mocap"]["overall_mean_deg"] - stats["processed_vs_mocap"]["overall_mean_deg"]),
        "overall_rmse_deg": float(stats["predicted_vs_mocap"]["overall_rmse_deg"] - stats["processed_vs_mocap"]["overall_rmse_deg"]),
        "right_mean_deg": float(stats["predicted_vs_mocap"]["right_mean_deg"] - stats["processed_vs_mocap"]["right_mean_deg"]),
        "right_rmse_deg": float(stats["predicted_vs_mocap"]["right_rmse_deg"] - stats["processed_vs_mocap"]["right_rmse_deg"]),
        "left_mean_deg": float(stats["predicted_vs_mocap"]["left_mean_deg"] - stats["processed_vs_mocap"]["left_mean_deg"]),
        "left_rmse_deg": float(stats["predicted_vs_mocap"]["left_rmse_deg"] - stats["processed_vs_mocap"]["left_rmse_deg"]),
    }
    return stats


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

    predicted_combined = np.concatenate(
        [
            np.asarray(predicted_jacp, dtype=np.float32).reshape(len(predicted_jacp), -1),
            np.asarray(predicted_jacr, dtype=np.float32).reshape(len(predicted_jacr), -1),
        ],
        axis=1,
    )
    processed_combined = np.concatenate(
        [
            np.asarray(processed_jacp, dtype=np.float32).reshape(len(processed_jacp), -1),
            np.asarray(processed_jacr, dtype=np.float32).reshape(len(processed_jacr), -1),
        ],
        axis=1,
    )
    gt_combined = np.concatenate(
        [
            np.asarray(gt_jacp, dtype=np.float32).reshape(len(gt_jacp), -1),
            np.asarray(gt_jacr, dtype=np.float32).reshape(len(gt_jacr), -1),
        ],
        axis=1,
    )

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
        "jacp": {
            key: float(stats["predicted_vs_mocap"]["jacp"][key] - stats["processed_vs_mocap"]["jacp"][key])
            for key in stats["predicted_vs_mocap"]["jacp"]
        },
        "jacr": {
            key: float(stats["predicted_vs_mocap"]["jacr"][key] - stats["processed_vs_mocap"]["jacr"][key])
            for key in stats["predicted_vs_mocap"]["jacr"]
        },
        "combined": {
            key: float(stats["predicted_vs_mocap"]["combined"][key] - stats["processed_vs_mocap"]["combined"][key])
            for key in stats["predicted_vs_mocap"]["combined"]
        },
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

    f_r_ga = grf_world_np[:, 0:3]
    f_l_ga = grf_world_np[:, 3:6]
    m_r_ga = np.column_stack([
        np.zeros(len(moments_world_np), dtype=np.float32),
        np.zeros(len(moments_world_np), dtype=np.float32),
        moments_world_np[:, 0],
    ]) if moments_world_np.shape[1] == 2 else moments_world_np[:, 0:3]
    m_l_ga = np.column_stack([
        np.zeros(len(moments_world_np), dtype=np.float32),
        np.zeros(len(moments_world_np), dtype=np.float32),
        moments_world_np[:, 1],
    ]) if moments_world_np.shape[1] == 2 else moments_world_np[:, 3:6]

    f_r = np.einsum("tij,tj->ti", rot_ga_to_w[:, 0], f_r_ga)
    f_l = np.einsum("tij,tj->ti", rot_ga_to_w[:, 1], f_l_ga)
    m_r = np.einsum("tij,tj->ti", rot_ga_to_w[:, 0], m_r_ga)
    m_l = np.einsum("tij,tj->ti", rot_ga_to_w[:, 1], m_l_ga)

    m_total_r = m_r + np.cross(cop_world[:, 0], f_r)
    m_total_l = m_l + np.cross(cop_world[:, 1], f_l)
    tau_r = np.einsum("tji,tj->ti", jacp_np[:, 0], f_r) + np.einsum("tji,tj->ti", jacr_np[:, 0], m_total_r)
    tau_l = np.einsum("tji,tj->ti", jacp_np[:, 1], f_l) + np.einsum("tji,tj->ti", jacr_np[:, 1], m_total_l)
    return (tau_r + tau_l).astype(np.float32)


def _build_calcaneus_applied_wrench(
    *,
    cop_xz: np.ndarray,
    ankle_heights: np.ndarray,
    grf_world: np.ndarray,
    moments_world: np.ndarray,
    rot_w_to_ga: np.ndarray,
) -> Dict[str, np.ndarray]:
    rot_ga_to_w = np.swapaxes(np.asarray(rot_w_to_ga, dtype=np.float32), -1, -2)
    cop_xz_np = np.asarray(cop_xz, dtype=np.float32)
    ankle_heights_np = np.asarray(ankle_heights, dtype=np.float32)
    grf_world_np = np.asarray(grf_world, dtype=np.float32)
    moments_world_np = np.asarray(moments_world, dtype=np.float32)

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

    force_r = grf_world_np[:, 0:3]
    force_l = grf_world_np[:, 3:6]
    if moments_world_np.shape[1] == 2:
        moment_r = np.column_stack([
            np.zeros(len(moments_world_np), dtype=np.float32),
            np.zeros(len(moments_world_np), dtype=np.float32),
            moments_world_np[:, 0],
        ])
        moment_l = np.column_stack([
            np.zeros(len(moments_world_np), dtype=np.float32),
            np.zeros(len(moments_world_np), dtype=np.float32),
            moments_world_np[:, 1],
        ])
    else:
        moment_r = moments_world_np[:, 0:3]
        moment_l = moments_world_np[:, 3:6]

    total_moment_r = moment_r + np.cross(cop_world[:, 0], force_r)
    total_moment_l = moment_l + np.cross(cop_world[:, 1], force_l)
    return {
        "force_r": force_r.astype(np.float32),
        "force_l": force_l.astype(np.float32),
        "total_moment_r": total_moment_r.astype(np.float32),
        "total_moment_l": total_moment_l.astype(np.float32),
        "cop_world": cop_world.astype(np.float32),
    }


def _compute_tau_from_applied_wrench(
    *,
    jacp: np.ndarray,
    jacr: np.ndarray,
    force_r: np.ndarray,
    force_l: np.ndarray,
    total_moment_r: np.ndarray,
    total_moment_l: np.ndarray,
) -> np.ndarray:
    jacp_np = np.asarray(jacp, dtype=np.float32)
    jacr_np = np.asarray(jacr, dtype=np.float32)
    force_r_np = np.asarray(force_r, dtype=np.float32)
    force_l_np = np.asarray(force_l, dtype=np.float32)
    total_moment_r_np = np.asarray(total_moment_r, dtype=np.float32)
    total_moment_l_np = np.asarray(total_moment_l, dtype=np.float32)
    tau_r = np.einsum("tji,tj->ti", jacp_np[:, 0], force_r_np) + np.einsum("tji,tj->ti", jacr_np[:, 0], total_moment_r_np)
    tau_l = np.einsum("tji,tj->ti", jacp_np[:, 1], force_l_np) + np.einsum("tji,tj->ti", jacr_np[:, 1], total_moment_l_np)
    return (tau_r + tau_l).astype(np.float32)


def _rotation_geodesic_timeseries_deg(
    rotation_a: np.ndarray,
    rotation_b: np.ndarray,
) -> np.ndarray:
    rot_a = np.asarray(rotation_a, dtype=np.float64)
    rot_b = np.asarray(rotation_b, dtype=np.float64)
    rot_err = rot_a @ np.swapaxes(rot_b, -1, -2)
    trace = np.trace(rot_err, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    skew_vec = np.stack(
        [
            rot_err[..., 2, 1] - rot_err[..., 1, 2],
            rot_err[..., 0, 2] - rot_err[..., 2, 0],
            rot_err[..., 1, 0] - rot_err[..., 0, 1],
        ],
        axis=-1,
    )
    sin_theta = 0.5 * np.linalg.norm(skew_vec, axis=-1)
    return np.degrees(np.arctan2(sin_theta, cos_theta)).astype(np.float32)


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
    id_full_gt = ground_truth.get("id_full")
    if qfrc_inverse_ref is None or id_full_gt is None:
        return {"available": False, "reason": "missing_qfrc_inverse_or_id_full"}

    gt_rot = np.asarray(ground_truth["rot_w_to_ga"], dtype=np.float32)
    gt_jacp = np.asarray(data["gt_jacp"], dtype=np.float32)
    gt_jacr = np.asarray(data["gt_jacr"], dtype=np.float32)
    processed_rot = np.asarray(data["rot_w_to_ga"], dtype=np.float32)
    processed_jacp = np.asarray(data["jacp"], dtype=np.float32)
    processed_jacr = np.asarray(data["jacr"], dtype=np.float32)
    predicted_rot = np.asarray(predictions["rot_w_to_ga"], dtype=np.float32)
    predicted_jacp = np.asarray(predictions["jacp"], dtype=np.float32)
    predicted_jacr = np.asarray(predictions["jacr"], dtype=np.float32)
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
        jacp=predicted_jacp,
        jacr=predicted_jacr,
        **common_kwargs,
    )
    tau_with_processed_jac = _compute_tau_from_candidate_geometry(
        rot_w_to_ga=gt_rot,
        jacp=processed_jacp,
        jacr=processed_jacr,
        **common_kwargs,
    )
    tau_with_pred_rot = _compute_tau_from_candidate_geometry(
        rot_w_to_ga=predicted_rot,
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
    pred_rot_rmse = [
        rotation_stats["predicted_vs_mocap"]["overall_rmse_deg"],
        rotation_stats["predicted_vs_mocap"]["right_rmse_deg"],
        rotation_stats["predicted_vs_mocap"]["left_rmse_deg"],
    ]
    proc_rot_rmse = [
        rotation_stats["processed_vs_mocap"]["overall_rmse_deg"],
        rotation_stats["processed_vs_mocap"]["right_rmse_deg"],
        rotation_stats["processed_vs_mocap"]["left_rmse_deg"],
    ]
    pred_rot_mean = [
        rotation_stats["predicted_vs_mocap"]["overall_mean_deg"],
        rotation_stats["predicted_vs_mocap"]["right_mean_deg"],
        rotation_stats["predicted_vs_mocap"]["left_mean_deg"],
    ]
    proc_rot_mean = [
        rotation_stats["processed_vs_mocap"]["overall_mean_deg"],
        rotation_stats["processed_vs_mocap"]["right_mean_deg"],
        rotation_stats["processed_vs_mocap"]["left_mean_deg"],
    ]

    fig.add_trace(go.Bar(name="Predicted MJX", x=labels_rot, y=pred_rot_rmse, marker_color="#E94F37"), row=1, col=1)
    fig.add_trace(go.Bar(name="ProcessedData", x=labels_rot, y=proc_rot_rmse, marker_color="#2E86AB"), row=1, col=1)
    fig.add_trace(go.Bar(name="Predicted MJX", x=labels_rot, y=pred_rot_mean, marker_color="#E94F37", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name="ProcessedData", x=labels_rot, y=proc_rot_mean, marker_color="#2E86AB", showlegend=False), row=1, col=2)

    jac_labels = [
        "jacp overall",
        "jacp right",
        "jacp left",
        "jacr overall",
        "jacr right",
        "jacr left",
        "combined overall",
    ]
    pred_jac_rmse = [
        jacobian_stats["predicted_vs_mocap"]["jacp"]["overall_rmse"],
        jacobian_stats["predicted_vs_mocap"]["jacp"]["right_rmse"],
        jacobian_stats["predicted_vs_mocap"]["jacp"]["left_rmse"],
        jacobian_stats["predicted_vs_mocap"]["jacr"]["overall_rmse"],
        jacobian_stats["predicted_vs_mocap"]["jacr"]["right_rmse"],
        jacobian_stats["predicted_vs_mocap"]["jacr"]["left_rmse"],
        jacobian_stats["predicted_vs_mocap"]["combined"]["overall_rmse"],
    ]
    proc_jac_rmse = [
        jacobian_stats["processed_vs_mocap"]["jacp"]["overall_rmse"],
        jacobian_stats["processed_vs_mocap"]["jacp"]["right_rmse"],
        jacobian_stats["processed_vs_mocap"]["jacp"]["left_rmse"],
        jacobian_stats["processed_vs_mocap"]["jacr"]["overall_rmse"],
        jacobian_stats["processed_vs_mocap"]["jacr"]["right_rmse"],
        jacobian_stats["processed_vs_mocap"]["jacr"]["left_rmse"],
        jacobian_stats["processed_vs_mocap"]["combined"]["overall_rmse"],
    ]
    pred_jac_mae = [
        jacobian_stats["predicted_vs_mocap"]["jacp"]["overall_mae"],
        jacobian_stats["predicted_vs_mocap"]["jacp"]["right_mae"],
        jacobian_stats["predicted_vs_mocap"]["jacp"]["left_mae"],
        jacobian_stats["predicted_vs_mocap"]["jacr"]["overall_mae"],
        jacobian_stats["predicted_vs_mocap"]["jacr"]["right_mae"],
        jacobian_stats["predicted_vs_mocap"]["jacr"]["left_mae"],
        jacobian_stats["predicted_vs_mocap"]["combined"]["overall_mae"],
    ]
    proc_jac_mae = [
        jacobian_stats["processed_vs_mocap"]["jacp"]["overall_mae"],
        jacobian_stats["processed_vs_mocap"]["jacp"]["right_mae"],
        jacobian_stats["processed_vs_mocap"]["jacp"]["left_mae"],
        jacobian_stats["processed_vs_mocap"]["jacr"]["overall_mae"],
        jacobian_stats["processed_vs_mocap"]["jacr"]["right_mae"],
        jacobian_stats["processed_vs_mocap"]["jacr"]["left_mae"],
        jacobian_stats["processed_vs_mocap"]["combined"]["overall_mae"],
    ]

    fig.add_trace(go.Bar(name="Predicted MJX", x=jac_labels, y=pred_jac_rmse, marker_color="#E94F37", showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(name="ProcessedData", x=jac_labels, y=proc_jac_rmse, marker_color="#2E86AB", showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(name="Predicted MJX", x=jac_labels, y=pred_jac_mae, marker_color="#E94F37", showlegend=False), row=2, col=2)
    fig.add_trace(go.Bar(name="ProcessedData", x=jac_labels, y=proc_jac_mae, marker_color="#2E86AB", showlegend=False), row=2, col=2)

    if knee_torque_stats is not None and knee_torque_stats.get("available"):
        fig.add_trace(
            go.Bar(
                name="Predicted MJX",
                x=["Left knee MAE"],
                y=[knee_torque_stats["rotation"]["predicted_mae"]],
                marker_color="#E94F37",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name="ProcessedData",
                x=["Left knee MAE"],
                y=[knee_torque_stats["rotation"]["processed_mae"]],
                marker_color="#2E86AB",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name="Predicted MJX",
                x=["Left knee MAE"],
                y=[knee_torque_stats["jacobian"]["predicted_mae"]],
                marker_color="#E94F37",
                showlegend=False,
            ),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                name="ProcessedData",
                x=["Left knee MAE"],
                y=[knee_torque_stats["jacobian"]["processed_mae"]],
                marker_color="#2E86AB",
                showlegend=False,
            ),
            row=3,
            col=2,
        )
        fig.add_annotation(
            x=0.22,
            y=0.03,
            xref="paper",
            yref="paper",
            text=(
                f"Pred - Processed: {knee_torque_stats['rotation']['pred_minus_processed_mae']:.4f}"
            ),
            showarrow=False,
        )
        fig.add_annotation(
            x=0.78,
            y=0.03,
            xref="paper",
            yref="paper",
            text=(
                f"Pred - Processed: {knee_torque_stats['jacobian']['pred_minus_processed_mae']:.4f}"
            ),
            showarrow=False,
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Rotation/Jacobian Accuracy vs MoCap Ground Truth</b><br>"
                f"<span style='font-size:12px'>{trial_name}</span>"
            ),
            x=0.5,
        ),
        barmode="group",
        height=1380,
        width=1800,
        template="plotly_white",
        margin=dict(t=100, b=80, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="Degrees", row=1, col=1)
    fig.update_yaxes(title_text="Degrees", row=1, col=2)
    fig.update_yaxes(title_text="RMSE", row=2, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=2)
    fig.update_yaxes(title_text="Torque MAE", row=3, col=1)
    fig.update_yaxes(title_text="Torque MAE", row=3, col=2)
    fig.update_xaxes(tickangle=-20, row=2, col=1)
    fig.update_xaxes(tickangle=-20, row=2, col=2)

    if save_path:
        fig.write_html(save_path)
        print(f"💾 Saved rotation/jacobian comparison plot to: {save_path}")
    return fig


def create_bilateral_cop_grf_plot(
    time_axis: np.ndarray,
    predictions: Dict[str, np.ndarray],
    predictions_alt: Optional[Dict[str, np.ndarray]],
    ground_truth: Dict[str, np.ndarray],
    trial_name: str,
    *,
    save_path: Optional[str] = None,
    pred_label: str = "Prediction (OpenCap input)",
    alt_pred_label: str = "Prediction (MotionCapture input)",
    evaluation_mask: Optional[np.ndarray] = None,
    metric_predictions: Optional[Dict[str, np.ndarray]] = None,
    metric_predictions_alt: Optional[Dict[str, np.ndarray]] = None,
) -> go.Figure:
    metric_predictions = metric_predictions if metric_predictions is not None else predictions
    metric_predictions_alt = metric_predictions_alt if metric_predictions_alt is not None else predictions_alt
    mask = (
        _normalize_evaluation_mask(evaluation_mask, len(time_axis))
        if evaluation_mask is not None
        else np.ones(len(time_axis), dtype=bool)
    )
    colors = {"gt": "#2E86AB", "pred": "#E94F37", "pred_alt": "#1B9E77"}
    subplot_titles = [
        "Right COP X", "Right COP Z", "Right GRF X", "Right GRF Y", "Right GRF Z",
        "Left COP X", "Left COP Z", "Left GRF X", "Left GRF Y", "Left GRF Z",
    ]
    fig = make_subplots(
        rows=2,
        cols=5,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.05,
    )

    row_specs = [
        ("Right", [0, 1], [0, 1, 2], 1),
        ("Left", [2, 3], [3, 4, 5], 2),
    ]
    for side_name, cop_indices, grf_indices, row in row_specs:
        for col, idx in enumerate(cop_indices, start=1):
            component = "X" if col == 1 else "Z"
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=ground_truth["cop"][:, idx],
                    name="Ground Truth",
                    line=dict(color=colors["gt"], width=2),
                    legendgroup="gt",
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=predictions["cop"][:, idx],
                    name=pred_label,
                    line=dict(color=colors["pred"], width=2, dash="dash"),
                    legendgroup="pred",
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )
            if predictions_alt is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=predictions_alt["cop"][:, idx],
                        name=alt_pred_label,
                        line=dict(color=colors["pred_alt"], width=2, dash="dot"),
                        legendgroup="pred_alt",
                        showlegend=(row == 1 and col == 1),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_yaxes(title_text=f"{side_name} COP {component} (m)", row=row, col=col)
            fig.update_xaxes(title_text="Time (s)", row=row, col=col)

        for offset, idx in enumerate(grf_indices, start=3):
            component = ("X", "Y", "Z")[offset - 3]
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=ground_truth["grf"][:, idx],
                    name="Ground Truth",
                    line=dict(color=colors["gt"], width=2),
                    legendgroup="gt",
                    showlegend=False,
                ),
                row=row,
                col=offset,
            )
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=predictions["grf"][:, idx],
                    name=pred_label,
                    line=dict(color=colors["pred"], width=2, dash="dash"),
                    legendgroup="pred",
                    showlegend=False,
                ),
                row=row,
                col=offset,
            )
            if predictions_alt is not None:
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=predictions_alt["grf"][:, idx],
                        name=alt_pred_label,
                        line=dict(color=colors["pred_alt"], width=2, dash="dot"),
                        legendgroup="pred_alt",
                        showlegend=False,
                    ),
                    row=row,
                    col=offset,
                )
            fig.update_yaxes(title_text=f"{side_name} GRF {component} (N)", row=row, col=offset)
            fig.update_xaxes(title_text="Time (s)", row=row, col=offset)

    rmse_summary: List[str] = []
    for side_name, cop_indices, grf_indices, _row in row_specs:
        cop_rmse = _masked_rmse(metric_predictions["cop"][:, cop_indices], ground_truth["cop"][:, cop_indices], mask)
        grf_rmse = _masked_rmse(metric_predictions["grf"][:, grf_indices], ground_truth["grf"][:, grf_indices], mask)
        side_summary = f"{pred_label} {side_name}: COP RMSE {cop_rmse:.4f} m, GRF RMSE {grf_rmse:.1f} N"
        if metric_predictions_alt is not None:
            cop_rmse_alt = _masked_rmse(metric_predictions_alt["cop"][:, cop_indices], ground_truth["cop"][:, cop_indices], mask)
            grf_rmse_alt = _masked_rmse(metric_predictions_alt["grf"][:, grf_indices], ground_truth["grf"][:, grf_indices], mask)
            side_summary += f" | {alt_pred_label}: COP RMSE {cop_rmse_alt:.4f} m, GRF RMSE {grf_rmse_alt:.1f} N"
        rmse_summary.append(side_summary)

    fig.update_layout(
        title=dict(
            text="<b>Bilateral COP/GRF Comparison: "
            f"{trial_name}</b><br><span style='font-size:12px'>{' | '.join(rmse_summary)}</span>",
            x=0.5,
            y=0.98,
            font=dict(size=16),
        ),
        height=900,
        width=1800,
        margin=dict(t=110, b=60, l=60, r=60),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.10,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        hovermode="x unified",
    )
    if save_path:
        fig.write_html(save_path)
        print(f"💾 Saved bilateral COP/GRF plot to: {save_path}")
    return fig


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    if "params" not in ckpt:
        raise KeyError("Checkpoint is missing `params`")
    return ckpt


def _rebuild_normalizers(raw_normalizers: Dict[str, Any]) -> Dict[str, Normalizer]:
    rebuilt: Dict[str, Normalizer] = {}
    for key, norm in raw_normalizers.items():
        if hasattr(norm, "mean") and hasattr(norm, "std"):
            rebuilt[key] = Normalizer(np.asarray(norm.mean), np.asarray(norm.std))
        elif isinstance(norm, dict) and "mean" in norm and "std" in norm:
            rebuilt[key] = Normalizer(np.asarray(norm["mean"]), np.asarray(norm["std"]))
    return rebuilt


def _parse_mod_q_input_feature_blocks(metadata: Dict[str, Any]) -> List[Tuple[str, int]]:
    default_dims = {
        str(name): (None if dim is None else int(dim))
        for name, dim in MOD_Q_INPUT_BLOCKS
    }
    ordered_names = [str(name) for name, _ in MOD_Q_INPUT_BLOCKS]
    raw_blocks = metadata.get("input_feature_blocks")
    parsed_dims: Dict[str, Optional[int]] = {}

    if isinstance(raw_blocks, (list, tuple)):
        for entry in raw_blocks:
            name = None
            raw_dim = None
            if isinstance(entry, dict):
                name = entry.get("name")
                raw_dim = entry.get("dim")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                name = entry[0]
                raw_dim = entry[1]
            if name is None:
                continue
            name = str(name)
            if name not in default_dims:
                continue
            dim_value: Optional[int]
            try:
                dim_value = int(raw_dim) if raw_dim is not None else None
            except Exception:
                dim_value = None
            parsed_dims[name] = dim_value

    blocks: List[Tuple[str, int]] = []
    for name in ordered_names:
        dim = parsed_dims.get(name)
        if dim is None:
            dim = default_dims[name]
        if dim is None:
            raise ValueError(f"Could not resolve expected dimension for mod_q input block '{name}'")
        blocks.append((name, int(dim)))
    return blocks


def _coerce_mod_q_block_dim(
    name: str,
    arr: np.ndarray,
    expected_dim: int,
) -> Tuple[np.ndarray, Optional[str]]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D temporal block '{name}', got shape {arr.shape}")
    current_dim = int(arr.shape[1])
    if current_dim == expected_dim:
        return arr, None

    # Match the feature pruning used by the training-side data processing
    # when OpenCap inputs still carry MTP channels.
    if name == "pos" and current_dim == 18 and expected_dim == 16:
        coerced = np.delete(arr, MOD_Q_POS_MTP_REMOVE_IDXS, axis=1)
        return coerced, f"removed MTP columns for block '{name}' ({current_dim} -> {expected_dim})"
    if name in {"vel", "acc"} and current_dim == 21 and expected_dim == 19:
        coerced = np.delete(arr, MOD_Q_VEL_ACC_MTP_REMOVE_IDXS, axis=1)
        return coerced, f"removed MTP columns for block '{name}' ({current_dim} -> {expected_dim})"

    if current_dim > expected_dim:
        coerced = arr[:, :expected_dim]
        return coerced, f"truncated block '{name}' ({current_dim} -> {expected_dim})"

    pad = np.zeros((arr.shape[0], expected_dim - current_dim), dtype=np.float32)
    coerced = np.concatenate([arr, pad], axis=1)
    return coerced, f"zero-padded block '{name}' ({current_dim} -> {expected_dim})"


def _coerce_optional_mod_q_block(
    name: str,
    value: Optional[np.ndarray],
    expected_dim: int,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    coerced, _note = _coerce_mod_q_block_dim(name, np.asarray(value, dtype=np.float32), expected_dim)
    return coerced


def _coerce_optional_mod_q_moments(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        return arr
    if arr.shape[1] == 2:
        return arr
    if arr.shape[1] >= 6:
        return arr[:, [2, 5]]
    if arr.shape[1] == 1:
        return np.repeat(arr, 2, axis=1)
    return arr[:, :2]


def _build_mod_q_temporal_features_for_checkpoint(
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    normalizers: Dict[str, Normalizer],
) -> Tuple[np.ndarray, List[Tuple[str, int]], List[str]]:
    block_sources = {
        "pelvis_rot": np.asarray(data["pelvis_rot"], dtype=np.float32),
        "pos": np.asarray(data["pos"], dtype=np.float32),
        "vel": np.asarray(data["vel"], dtype=np.float32),
        "acc": np.asarray(data["acc"], dtype=np.float32),
        "com_r": np.asarray(data["com_r"], dtype=np.float32),
        "com_l": np.asarray(data["com_l"], dtype=np.float32),
        "com_accel": np.asarray(data["com_accel"], dtype=np.float32),
        "foot_progression_angle": np.asarray(data["foot_progression_angle"], dtype=np.float32),
        "calcn_to_floor_angle": np.asarray(data["calcn_to_floor_angle"], dtype=np.float32),
    }
    expected_blocks = _parse_mod_q_input_feature_blocks(metadata)
    parts: List[np.ndarray] = []
    notes: List[str] = []
    actual_blocks: List[Tuple[str, int]] = []
    for name, expected_dim in expected_blocks:
        if name not in block_sources:
            raise KeyError(f"Missing mod_q temporal block '{name}' in trial data")
        coerced, note = _coerce_mod_q_block_dim(name, block_sources[name], expected_dim)
        parts.append(coerced)
        actual_blocks.append((name, int(coerced.shape[1])))
        if note:
            notes.append(note)
    temporal_features = np.concatenate(parts, axis=1).astype(np.float32)

    input_norm = normalizers.get("input")
    if input_norm is not None:
        expected_total = int(np.asarray(input_norm.mean).shape[-1])
        actual_total = int(temporal_features.shape[1])
        if actual_total != expected_total:
            block_desc = ", ".join(f"{name}={dim}" for name, dim in actual_blocks)
            raise ValueError(
                "mod_q temporal feature layout still mismatches checkpoint after reconciliation: "
                f"{actual_total} vs expected {expected_total}. Blocks: {block_desc}"
            )
    return temporal_features, actual_blocks, notes


def _resolve_mod_q_state_templates(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    notes: List[str] = []

    qpos_template = data.get("qpos_mjx_input")
    if qpos_template is None:
        qpos_template = data.get("qpos_mjx_gt", data.get("pos_mjx"))
        if qpos_template is not None:
            notes.append("qpos_mjx_input missing; using qpos ground-truth template")
    if qpos_template is None:
        raise RuntimeError(
            "mod_q inference requires at least one qpos template (qpos_mjx_input or qpos_mjx_gt/pos_mjx)."
        )
    qpos_template = np.asarray(qpos_template, dtype=np.float32)

    qvel_template = data.get("qvel_mjx_input")
    if qvel_template is None:
        qvel_template = data.get("qvel_mjx_gt")
        if qvel_template is not None:
            notes.append("qvel_mjx_input missing; using qvel ground-truth template")
        else:
            qvel_template = np.zeros_like(qpos_template, dtype=np.float32)
            notes.append("qvel_mjx_input missing; using zeros_like(qpos_template)")
    qvel_template = np.asarray(qvel_template, dtype=np.float32)

    qacc_template = data.get("qacc_mjx_input")
    if qacc_template is None:
        qacc_template = data.get("qacc_mjx_gt")
        if qacc_template is not None:
            notes.append("qacc_mjx_input missing; using qacc ground-truth template")
        else:
            qacc_template = np.zeros_like(qpos_template, dtype=np.float32)
            notes.append("qacc_mjx_input missing; using zeros_like(qpos_template)")
    qacc_template = np.asarray(qacc_template, dtype=np.float32)

    return qpos_template, qvel_template, qacc_template, notes


def _build_gt_dict(data: Dict[str, Any]) -> Dict[str, Optional[np.ndarray]]:
    return {
        "cop": np.asarray(data.get("cop_gt_raw", data.get("cop_raw", data.get("cop"))), dtype=np.float32) if data.get("cop_gt_raw", data.get("cop_raw", data.get("cop"))) is not None else None,
        "grf": np.asarray(data.get("grf_gt_raw", data.get("grf_raw", data.get("grf"))), dtype=np.float32) if data.get("grf_gt_raw", data.get("grf_raw", data.get("grf"))) is not None else None,
        "moments": _coerce_optional_mod_q_moments(data.get("moments_gt_raw", data.get("moments_raw", data.get("moments")))),
        "pos_gt": _coerce_optional_mod_q_block("pos", data.get("pos_gt"), 16),
        "vel_gt": _coerce_optional_mod_q_block("vel", data.get("vel_gt"), 19),
        "acc_gt": _coerce_optional_mod_q_block("acc", data.get("acc_gt"), 19),
        "tau_grf": np.asarray(data.get("tau_grf_gt", data.get("qfrc_grf_contribution")), dtype=np.float32) if data.get("tau_grf_gt", data.get("qfrc_grf_contribution")) is not None else None,
        "qfrc_inverse": np.asarray(data.get("qfrc_inverse_raw", data.get("qfrc_inverse")), dtype=np.float32) if data.get("qfrc_inverse_raw", data.get("qfrc_inverse")) is not None else None,
        "qfrc_inverse_processed": np.asarray(data.get("qfrc_inverse_processed"), dtype=np.float32) if data.get("qfrc_inverse_processed") is not None else None,
        "qfrc_inverse_mocap": np.asarray(data.get("qfrc_inverse_mocap"), dtype=np.float32) if data.get("qfrc_inverse_mocap") is not None else None,
        "id_gt_mjx": np.asarray(data.get("id_gt_mjx"), dtype=np.float32) if data.get("id_gt_mjx") is not None else None,
        "rot_w_to_ga": np.asarray(data.get("gt_rot_w_to_ga", data.get("rot_w_to_ga")), dtype=np.float32) if data.get("gt_rot_w_to_ga", data.get("rot_w_to_ga")) is not None else None,
        "qpos_mjx_gt": np.asarray(data.get("qpos_mjx_gt", data.get("pos_mjx")), dtype=np.float32) if data.get("qpos_mjx_gt", data.get("pos_mjx")) is not None else None,
        "qvel_mjx_gt": np.asarray(data.get("qvel_mjx_gt"), dtype=np.float32) if data.get("qvel_mjx_gt") is not None else None,
        "qacc_mjx_gt": np.asarray(data.get("qacc_mjx_gt"), dtype=np.float32) if data.get("qacc_mjx_gt") is not None else None,
        "source": data.get("ground_truth_source", "selected input source"),
    }


def _compute_selected_torque_metrics(
    pred_signal: Optional[np.ndarray],
    gt_signal: Optional[np.ndarray],
    selected_indices: Sequence[int],
    stance_mask: np.ndarray,
    norm_factor: float,
) -> Dict[str, Any]:
    base = {
        "rmse": float("nan"),
        "rmse_bwh": float("nan"),
        "nrmse": float("nan"),
        "mae": float("nan"),
        "mae_bwh": float("nan"),
        "rmse_per_dof": [],
        "rmse_bwh_per_dof": [],
        "nrmse_per_dof": [],
        "available": False,
    }
    if pred_signal is None or gt_signal is None:
        return base

    pred_arr = np.asarray(pred_signal, dtype=np.float32)
    gt_arr = np.asarray(gt_signal, dtype=np.float32)
    pred_sel = pred_arr[:, selected_indices]
    gt_sel = gt_arr[:, selected_indices]
    rmse_per_dof = _masked_rmse_per_channel(pred_sel, gt_sel, stance_mask)
    rmse = _masked_rmse(pred_sel, gt_sel, stance_mask)
    mae = _masked_mae(pred_sel, gt_sel, stance_mask) if np.any(stance_mask) else float("nan")

    if np.any(stance_mask):
        gt_std = np.std(gt_sel[stance_mask], axis=0)
        gt_std_safe = np.where(gt_std < 1e-6, 1.0, gt_std)
        nrmse_per_dof = rmse_per_dof / gt_std_safe
        nrmse = float(np.mean(nrmse_per_dof))
    else:
        nrmse_per_dof = np.full(len(selected_indices), np.nan, dtype=np.float64)
        nrmse = float("nan")

    base.update(
        {
            "rmse": float(rmse),
            "rmse_bwh": float((rmse / norm_factor) * 100.0),
            "nrmse": nrmse,
            "mae": float(mae),
            "mae_bwh": float((mae / norm_factor) * 100.0),
            "rmse_per_dof": rmse_per_dof.tolist(),
            "rmse_bwh_per_dof": ((rmse_per_dof / norm_factor) * 100.0).tolist(),
            "nrmse_per_dof": nrmse_per_dof.tolist(),
            "available": True,
        }
    )
    return base


def _convert_predictions(
    output_np: np.ndarray,
    data: Dict[str, Any],
    normalizers: Dict[str, Normalizer],
    *,
    deviation_learning: bool = False,
) -> Dict[str, np.ndarray]:
    decoded = decode_mod_q_predictions(output_np)
    cop_ratio = decoded["cop"]
    grf_ratio = decoded["grf"]
    mom_ratio = decoded["moments"]
    contact = decoded["contact"]
    pos = decoded["pos"]
    vel = decoded["vel"]
    acc = decoded["acc"]

    if "cop" in normalizers:
        cop_ratio = normalizers["cop"].unnormalize(cop_ratio)
    if "grf" in normalizers:
        grf_ratio = normalizers["grf"].unnormalize(grf_ratio)
    if "moments" in normalizers:
        mom_ratio = normalizers["moments"].unnormalize(mom_ratio)
    if deviation_learning:
        pos_expected_dim = int(np.asarray(decoded["pos"], dtype=np.float32).shape[-1])
        vel_expected_dim = int(np.asarray(decoded["vel"], dtype=np.float32).shape[-1])
        acc_expected_dim = int(np.asarray(decoded["acc"], dtype=np.float32).shape[-1])
        pos_noised = _coerce_optional_mod_q_block("pos", data.get("pos"), pos_expected_dim)
        vel_noised = _coerce_optional_mod_q_block("vel", data.get("vel"), vel_expected_dim)
        acc_noised = _coerce_optional_mod_q_block("acc", data.get("acc"), acc_expected_dim)
        if pos_noised is None or vel_noised is None or acc_noised is None:
            raise RuntimeError("Deviation-learning mod_q inference requires pos/vel/acc baseline inputs.")
        if "pos" in normalizers:
            pos = normalizers["pos"].unnormalize(decoded["pos"] + normalizers["pos"].normalize(pos_noised))
        else:
            pos = decoded["pos"] + pos_noised
        if "vel" in normalizers:
            vel = normalizers["vel"].unnormalize(decoded["vel"] + normalizers["vel"].normalize(vel_noised))
        else:
            vel = decoded["vel"] + vel_noised
        if "acc" in normalizers:
            acc = normalizers["acc"].unnormalize(decoded["acc"] + normalizers["acc"].normalize(acc_noised))
        else:
            acc = decoded["acc"] + acc_noised
    else:
        if "pos" in normalizers:
            pos = normalizers["pos"].unnormalize(pos)
        if "vel" in normalizers:
            vel = normalizers["vel"].unnormalize(vel)
        if "acc" in normalizers:
            acc = normalizers["acc"].unnormalize(acc)

    height = np.asarray(data["height"], dtype=np.float32).reshape(-1, 1)
    mass = np.asarray(data["mass"], dtype=np.float32).reshape(-1, 1)
    cop_phys = np.asarray(cop_ratio, dtype=np.float32) * height
    grf_phys = np.asarray(grf_ratio, dtype=np.float32) * (mass * 9.8067)
    mom_phys = np.asarray(mom_ratio, dtype=np.float32) * (mass * 9.8067 * height)
    return {
        "cop": cop_phys,
        "grf": grf_phys,
        "moments": mom_phys,
        "contact": np.asarray(contact, dtype=np.float32),
        "pos": np.asarray(pos, dtype=np.float32),
        "vel": np.asarray(vel, dtype=np.float32),
        "acc": np.asarray(acc, dtype=np.float32),
        "decoded_raw": decoded,
    }


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_trial_name(entry: Any) -> str:
    if isinstance(entry, dict):
        subject = str(entry.get("subject", "")).strip()
        trial = str(entry.get("trial_name", entry.get("trial", ""))).strip()
        if subject and trial and "/" not in trial:
            return f"{subject}/{trial}"
        return trial or subject
    return str(entry)


def _normalize_trial_name(trial_name: str) -> str:
    return str(trial_name).strip().lower().replace("\\", "/")


def _load_mod_q_checkpoint_bundle(checkpoint_path: Path) -> Dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint_path)
    metadata = dict(ckpt.get("mod_q_metadata", {}))
    metadata.update(
        {
            k: ckpt.get(k)
            for k in ("model_type", "output_schema", "qprime_layout", "input_feature_blocks")
            if k in ckpt
        }
    )
    if metadata.get("model_type", MOD_Q_MODEL_TYPE) != MOD_Q_MODEL_TYPE:
        raise ValueError(f"Checkpoint model_type must be '{MOD_Q_MODEL_TYPE}', got {metadata.get('model_type')}")

    hyperparams = _load_optional_json(checkpoint_path.parent / "hyperparameters.json")
    split_info = _load_optional_json(checkpoint_path.parent / "train_val_split.json")
    train_trials = ckpt.get("train_trials") or split_info.get("train_trials", [])
    val_trials = ckpt.get("val_trials") or split_info.get("val_trials", [])
    normalizers = _rebuild_normalizers(ckpt.get("normalizers", {}))

    default_window_size = int(
        metadata.get(
            "window_size",
            ckpt.get("window_size", hyperparams.get("window_size", 125)),
        )
    )
    default_stride = int(
        metadata.get(
            "stride",
            ckpt.get("stride", hyperparams.get("stride", max(1, default_window_size // 4))),
        )
    )
    default_prediction_margin_frames = int(
        metadata.get(
            "prediction_margin_frames",
            ckpt.get("prediction_margin_frames", hyperparams.get("prediction_margin_frames", 15)),
        )
    )
    default_use_noised = _coerce_bool(
        metadata.get("UseNoised", hyperparams.get("UseNoised", True)),
        default=True,
    )
    default_deviation_learning = _coerce_bool(
        metadata.get("DeviationLearning", hyperparams.get("DeviationLearning", False)),
        default=False,
    )

    model_cfg = {
        "d_model": metadata.get("d_model", ckpt.get("d_model", hyperparams.get("d_model", 256))),
        "num_layers": metadata.get("num_layers", ckpt.get("num_layers", hyperparams.get("num_layers", 4))),
        "ff_dim": metadata.get("ff_dim", ckpt.get("ff_dim", hyperparams.get("ff_dim", 1024))),
        "dropout_rate": metadata.get("dropout_rate", ckpt.get("dropout_rate", hyperparams.get("dropout_rate", 0.1))),
        "cnn_num_layers": metadata.get("cnn_num_layers", ckpt.get("cnn_num_layers", hyperparams.get("cnn_num_layers", 2))),
        "cnn_kernel_sizes": metadata.get("cnn_kernel_sizes", ckpt.get("cnn_kernel_sizes", hyperparams.get("cnn_kernel_sizes", (3, 5)))),
        "use_cnn": metadata.get("use_cnn", ckpt.get("use_cnn", hyperparams.get("use_cnn", True))),
        "output_dim": int(metadata.get("output_dim", ckpt.get("output_dim", MOD_Q_OUTPUT_DIM))),
    }
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": ckpt,
        "metadata": metadata,
        "hyperparams": hyperparams,
        "train_trials": train_trials,
        "val_trials": val_trials,
        "normalizers": normalizers,
        "model_cfg": model_cfg,
        "default_window_size": default_window_size,
        "default_stride": default_stride,
        "default_prediction_margin_frames": default_prediction_margin_frames,
        "default_use_noised": default_use_noised,
        "default_deviation_learning": default_deviation_learning,
    }


def _resolve_split_status(
    trial_name: str,
    train_trials: Sequence[Any],
    val_trials: Sequence[Any],
) -> str:
    trial_norm = _normalize_trial_name(trial_name)
    train_names = {_normalize_trial_name(_extract_trial_name(entry)) for entry in train_trials}
    val_names = {_normalize_trial_name(_extract_trial_name(entry)) for entry in val_trials}
    if trial_norm in train_names:
        return "TRAIN"
    if trial_norm in val_names:
        return "VALIDATION"
    return "unknown"


def _load_trials_from_json(json_path: Path) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    trials: List[str] = []
    for entry in test_data:
        if isinstance(entry, str):
            trials.append(entry)
        elif isinstance(entry, dict):
            trial_name = str(entry.get("trial_name", entry.get("trial", ""))).strip()
            if trial_name:
                trials.append(trial_name)
    return trials


def run_mod_q_inference(
    checkpoint_path: str,
    data_dir: str,
    trial_name: str,
    output_dir: str,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    prediction_margin_frames: Optional[int] = None,
    *,
    no_plots: bool = False,
    lightweight: bool = False,
    make_graph: bool = False,
    opencap_val_dataset: bool = False,
    input_source: str = "processed",
    use_noised: Optional[bool] = None,
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checkpoint_path = str(checkpoint_path)
    bundle = bundle or _load_mod_q_checkpoint_bundle(Path(checkpoint_path))
    trial_match = infer_find_trial(data_dir, trial_name)
    if trial_match is None:
        raise FileNotFoundError(f"Could not resolve trial '{trial_name}' under {data_dir}")
    trial_path_str, _patient_path = trial_match
    trial_path = Path(trial_path_str)

    metadata = dict(bundle["metadata"])
    normalizers = bundle["normalizers"]
    model_cfg = dict(bundle["model_cfg"])
    split_status = _resolve_split_status(
        trial_name,
        bundle.get("train_trials", []),
        bundle.get("val_trials", []),
    )

    input_source_norm = str(input_source).strip().lower()
    resolved_use_noised = (
        bundle["default_use_noised"] if use_noised is None else _coerce_bool(use_noised, default=bundle["default_use_noised"])
    )
    if opencap_val_dataset and input_source_norm != "mocap" and resolved_use_noised:
        print(
            "   📐 OpenCapValDataset detected: forcing UseNoised=False for OpenCapSubjects inputs "
            "because that dataset should use clean ProcessedData kinematics."
        )
        resolved_use_noised = False
    if input_source_norm == "mocap":
        resolved_use_noised = False
    deviation_learning = bool(metadata.get("DeviationLearning", bundle["default_deviation_learning"]))

    data = load_single_trial(
        trial_path,
        use_noised=resolved_use_noised,
        noised_gt=False,
        opencap_val=opencap_val_dataset,
        input_source=input_source_norm,
        deviation_learning=deviation_learning,
    )
    if data is None:
        raise RuntimeError(f"Failed to load trial data from {trial_path}")

    input_source_display = "MotionCapture" if input_source_norm == "mocap" else (
        "OpenCap (noised kinematics)" if resolved_use_noised else "OpenCap"
    )

    temporal_features, temporal_blocks, temporal_notes = _build_mod_q_temporal_features_for_checkpoint(
        data,
        metadata,
        normalizers,
    )
    for note in temporal_notes:
        print(f"   🔧 mod_q input reconciliation: {note}")
    if temporal_blocks:
        block_desc = ", ".join(f"{name}={dim}" for name, dim in temporal_blocks)
        print(f"   📐 mod_q temporal blocks: {block_desc}")
    static_context = build_mod_q_static_context(data)

    if "input" in normalizers:
        temporal_features_z = normalizers["input"].normalize(temporal_features).astype(np.float32)
    else:
        temporal_features_z = temporal_features.astype(np.float32)
    if "static" in normalizers:
        static_context_z = normalizers["static"].normalize(static_context[None, :]).astype(np.float32)[0]
    else:
        static_context_z = static_context.astype(np.float32)

    model = build_mod_q_model(temporal_features_z.shape[-1], static_context_z.shape[-1], model_cfg)
    params = bundle["checkpoint"]["params"]
    resolved_window_size = int(window_size) if window_size is not None else int(bundle["default_window_size"])
    resolved_stride = int(stride) if stride is not None else int(bundle["default_stride"])
    resolved_prediction_margin_frames = (
        int(prediction_margin_frames)
        if prediction_margin_frames is not None
        else int(bundle.get("default_prediction_margin_frames", 15))
    )

    inference_start = time.perf_counter()
    output_np, counts, evaluation_mask, window_meta = _windowed_predict(
        model=model,
        params=params,
        input_z=temporal_features_z,
        static_z=static_context_z,
        window_size=resolved_window_size,
        stride=resolved_stride,
        prediction_margin_frames=resolved_prediction_margin_frames,
    )
    if int(output_np.shape[-1]) < MOD_Q_OUTPUT_DIM:
        raise ValueError(
            f"Checkpoint output dim {output_np.shape[-1]} is smaller than mod_q schema dim {MOD_Q_OUTPUT_DIM}"
        )

    pred = _convert_predictions(
        output_np,
        data,
        normalizers,
        deviation_learning=deviation_learning,
    )
    xml_path = _resolve_subject_xml(trial_path, data)
    qpos_template, qvel_template, qacc_template, template_notes = _resolve_mod_q_state_templates(data)
    for note in template_notes:
        print(f"   🔧 mod_q q-template fallback: {note}")

    qprime = reconstruct_mod_q_state(
        pos_pred=pred["pos"],
        vel_pred=pred["vel"],
        acc_pred=pred["acc"],
        qpos_template=qpos_template,
        qvel_template=qvel_template,
        qacc_template=qacc_template,
        xml_path=xml_path,
    )
    ankle_heights = np.asarray(data["ankle_heights"], dtype=np.float32)
    physics = mod_q_physics_adapter(
        qprime=qprime,
        cop_phys=pred["cop"],
        grf_phys=pred["grf"],
        moments_phys=pred["moments"],
        ankle_heights=ankle_heights,
        rot_w_to_ga_gt=np.asarray(data.get("gt_rot_w_to_ga", data.get("rot_w_to_ga")), dtype=np.float32)
        if data.get("gt_rot_w_to_ga", data.get("rot_w_to_ga")) is not None
        else None,
        xml_path=xml_path,
    )
    inference_time_ms = float((time.perf_counter() - inference_start) * 1000.0)

    predictions = {
        **pred,
        "qpos": qprime["qpos"],
        "qvel": qprime["qvel"],
        "qacc": qprime["qacc"],
        "rot_w_to_ga": physics["rot_w_to_ga"],
        "jacp": physics["jacp"],
        "jacr": physics["jacr"],
        "qfrc_constraint": physics.get("qfrc_constraint"),
        "qfrc_inverse": physics["qfrc_inverse"],
        "tau_grf": physics["tau_grf"],
        "qfrc_grf_contribution": physics["tau_grf"],
        "id_full": physics["full_id"],
        "cop_world": physics["cop_world"],
    }
    ground_truth = _build_gt_dict(data)
    ground_truth["id_full"] = (
        np.asarray(ground_truth["id_gt_mjx"], dtype=np.float32)
        if ground_truth.get("id_gt_mjx") is not None
        else None
    )

    rotation_metrics: Dict[str, float] = {}
    if ground_truth.get("rot_w_to_ga") is not None:
        summary_deg = rotation_geodesic_summary_deg(
            predictions["rot_w_to_ga"],
            ground_truth["rot_w_to_ga"],
            xp=np,
        )
        rotation_metrics = {
            "rotation_geodesic_mean_deg": float(summary_deg["overall_mean_deg"]),
            "rotation_geodesic_rmse_deg": float(summary_deg["overall_rmse_deg"]),
            "rotation_geodesic_right_mean_deg": float(summary_deg["right_mean_deg"]),
            "rotation_geodesic_left_mean_deg": float(summary_deg["left_mean_deg"]),
        }

    rotation_comparison_stats: Optional[Dict[str, Any]] = None
    jacobian_comparison_stats: Optional[Dict[str, Any]] = None
    knee_torque_comparison_stats: Optional[Dict[str, Any]] = None

    evaluation_mask = _normalize_evaluation_mask(evaluation_mask, len(data["pos"]))
    time_axis = np.arange(len(data["pos"]), dtype=np.float32) / 100.0
    evaluation_predictions = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in predictions.items()
        if not str(key).startswith("_")
    }

    if FilterPostInfer:
        for target_dict in (predictions, evaluation_predictions):
            for key in ("cop", "grf", "moments", "tau_grf", "qfrc_grf_contribution", "qfrc_inverse"):
                if target_dict.get(key) is not None:
                    target_dict[key] = apply_butterworth_filter_masked(target_dict[key], evaluation_mask)
        for key in ("cop", "grf", "moments", "tau_grf", "qfrc_inverse", "qfrc_inverse_processed", "qfrc_inverse_mocap", "id_gt_mjx", "id_full"):
            if ground_truth.get(key) is not None:
                ground_truth[key] = apply_butterworth_filter_masked(ground_truth[key], evaluation_mask)

    evaluation_frame_count = int(np.sum(evaluation_mask))
    mass = float(data["mass"][0, 0])
    height = float(data["height"][0, 0])
    norm_factor = mass * height * 9.8067
    selected_torque_indices = get_selected_left_stance_dof_indices()
    selected_torque_names = [get_dof_names()[idx] for idx in selected_torque_indices]
    left_stance_mask = get_left_stance_mask(
        data["grf_raw"],
        evaluation_mask,
        threshold=LEFT_STANCE_THRESHOLD_N,
    )
    left_stance_frame_count = int(np.sum(left_stance_mask))
    grf_torque_metrics = _compute_selected_torque_metrics(
        evaluation_predictions.get("tau_grf"),
        ground_truth.get("tau_grf"),
        selected_torque_indices,
        left_stance_mask,
        norm_factor,
    )
    joint_torque_metrics = _compute_selected_torque_metrics(
        evaluation_predictions.get("id_full"),
        ground_truth.get("id_full"),
        selected_torque_indices,
        left_stance_mask,
        norm_factor,
    )
    primary_torque_metrics = joint_torque_metrics if joint_torque_metrics["available"] else grf_torque_metrics
    primary_torque_signal = "id_full" if joint_torque_metrics["available"] else "tau_grf"

    cop_bias_per_channel = _masked_mean_diff(
        evaluation_predictions["cop"],
        ground_truth["cop"],
        evaluation_mask,
    )
    grf_bias_per_channel = _masked_mean_diff(
        evaluation_predictions["grf"],
        ground_truth["grf"],
        evaluation_mask,
    )

    metrics: Dict[str, Any] = {
        "cop_rmse": _masked_rmse(evaluation_predictions["cop"], ground_truth["cop"], evaluation_mask),
        "grf_rmse": _masked_rmse(evaluation_predictions["grf"], ground_truth["grf"], evaluation_mask),
        "moments_rmse": _masked_rmse(evaluation_predictions["moments"], ground_truth["moments"], evaluation_mask),
        "cop_bias_per_channel": cop_bias_per_channel.tolist(),
        "grf_bias_per_channel": grf_bias_per_channel.tolist(),
        "torque_rmse": float(primary_torque_metrics["rmse"]),
        "torque_rmse_bwh": float(primary_torque_metrics["rmse_bwh"]),
        "torque_nrmse": float(primary_torque_metrics["nrmse"]),
        "torque_rmse_per_dof": list(primary_torque_metrics["rmse_per_dof"]),
        "torque_rmse_bwh_per_dof": list(primary_torque_metrics["rmse_bwh_per_dof"]),
        "torque_nrmse_per_dof": list(primary_torque_metrics["nrmse_per_dof"]),
        "torque_mae": float(primary_torque_metrics["mae"]),
        "torque_mae_bwh": float(primary_torque_metrics["mae_bwh"]),
        "torque_metric_signal": primary_torque_signal,
        "torque_metric_dof_names": selected_torque_names,
        "torque_metric_scope": "left_stance_selected_dofs",
        "torque_metric_side": "left",
        "torque_metric_phase": "stance",
        "torque_metric_left_stance_frame_count": left_stance_frame_count,
        "torque_metric_stance_threshold_N": float(LEFT_STANCE_THRESHOLD_N),
        "joint_torque_metrics_available": bool(joint_torque_metrics["available"]),
        "joint_torque_rmse": float(joint_torque_metrics["rmse"]),
        "joint_torque_rmse_bwh": float(joint_torque_metrics["rmse_bwh"]),
        "joint_torque_nrmse": float(joint_torque_metrics["nrmse"]),
        "joint_torque_rmse_per_dof": list(joint_torque_metrics["rmse_per_dof"]),
        "joint_torque_rmse_bwh_per_dof": list(joint_torque_metrics["rmse_bwh_per_dof"]),
        "joint_torque_nrmse_per_dof": list(joint_torque_metrics["nrmse_per_dof"]),
        "joint_torque_mae": float(joint_torque_metrics["mae"]),
        "joint_torque_mae_bwh": float(joint_torque_metrics["mae_bwh"]),
        "grf_torque_metrics_available": bool(grf_torque_metrics["available"]),
        "grf_torque_rmse": float(grf_torque_metrics["rmse"]),
        "grf_torque_rmse_bwh": float(grf_torque_metrics["rmse_bwh"]),
        "grf_torque_nrmse": float(grf_torque_metrics["nrmse"]),
        "grf_torque_rmse_per_dof": list(grf_torque_metrics["rmse_per_dof"]),
        "grf_torque_rmse_bwh_per_dof": list(grf_torque_metrics["rmse_bwh_per_dof"]),
        "grf_torque_nrmse_per_dof": list(grf_torque_metrics["nrmse_per_dof"]),
        "grf_torque_mae": float(grf_torque_metrics["mae"]),
        "grf_torque_mae_bwh": float(grf_torque_metrics["mae_bwh"]),
        "inference_time_ms": inference_time_ms,
        "num_frames": len(data["pos"]),
        "evaluation_frame_count": evaluation_frame_count,
        "window_size": int(resolved_window_size),
        "stride": int(resolved_stride),
        "prediction_margin_frames": int(resolved_prediction_margin_frames),
        "input_source": input_source_norm,
        "input_source_label": input_source_display,
        "input_kinematics_source": data.get("input_kinematics_source", "Pos"),
        "use_noised_inputs": bool(data.get("use_noised_inputs", False)),
        "ground_truth_source": data.get("ground_truth_source", "selected input source"),
        "restrict_max_vals": {"enabled": False},
    }
    metrics.update(
        _compute_qfrc_inverse_rmse_metrics(
            evaluation_predictions,
            ground_truth,
            evaluation_mask,
            norm_factor,
        )
    )
    metrics.update(rotation_metrics)

    if (
        opencap_val_dataset
        and input_source_norm != "mocap"
        and data.get("rot_w_to_ga") is not None
        and ground_truth.get("rot_w_to_ga") is not None
        and data.get("jacp") is not None
        and data.get("jacr") is not None
        and data.get("gt_jacp") is not None
        and data.get("gt_jacr") is not None
    ):
        rotation_comparison_stats = _build_rotation_comparison_stats(
            predictions["rot_w_to_ga"],
            np.asarray(data["rot_w_to_ga"], dtype=np.float32),
            np.asarray(ground_truth["rot_w_to_ga"], dtype=np.float32),
            evaluation_mask,
        )
        jacobian_comparison_stats = _build_jacobian_comparison_stats(
            predictions["jacp"],
            predictions["jacr"],
            np.asarray(data["jacp"], dtype=np.float32),
            np.asarray(data["jacr"], dtype=np.float32),
            np.asarray(data["gt_jacp"], dtype=np.float32),
            np.asarray(data["gt_jacr"], dtype=np.float32),
            evaluation_mask,
        )
        metrics["rotation_vs_mocap_comparison"] = rotation_comparison_stats
        metrics["jacobian_vs_mocap_comparison"] = jacobian_comparison_stats
        knee_torque_comparison_stats = _build_knee_flexion_torque_comparison_stats(
            predictions=predictions,
            data=data,
            ground_truth=ground_truth,
            left_stance_mask=left_stance_mask,
        )
        metrics["knee_flexion_torque_vs_mocap_comparison"] = knee_torque_comparison_stats

    primary_label = "Prediction (MotionCapture input)" if input_source_norm == "mocap" else "Prediction (OpenCap input)"
    secondary_label = "Prediction (OpenCap input)" if input_source_norm == "mocap" else "Prediction (MotionCapture input)"

    secondary_predictions = None
    secondary_metrics = None
    secondary_mae_report = None
    if opencap_val_dataset and input_source_norm != "mocap":
        try:
            secondary_result = run_mod_q_inference(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                trial_name=trial_name,
                output_dir=output_dir,
                window_size=resolved_window_size,
                stride=resolved_stride,
                prediction_margin_frames=resolved_prediction_margin_frames,
                no_plots=True,
                lightweight=True,
                make_graph=False,
                opencap_val_dataset=opencap_val_dataset,
                input_source="mocap",
                use_noised=False,
                bundle=bundle,
            )
            secondary_predictions = secondary_result.get("predictions")
            secondary_metrics = secondary_result.get("metrics")
            secondary_mae_report = secondary_result.get("mae_report")
        except Exception as exc:
            print(f"   ⚠️ Secondary MotionCapture-input inference failed for {trial_name}: {exc}")

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

    safe_trial_name = trial_name.replace("/", "_")
    trial_output_dir = Path(output_dir) / safe_trial_name
    if not lightweight:
        trial_output_dir.mkdir(parents=True, exist_ok=True)
        plot_artifacts = {
            "timeseries_right": str(trial_output_dir / "timeseries_right.html"),
            "timeseries_left": str(trial_output_dir / "timeseries_left.html"),
            "bilateral_cop_grf": str(trial_output_dir / "cop_grf_bilateral.html"),
            "all_dofs": str(trial_output_dir / "all_dofs.html"),
            "knee_torque": str(trial_output_dir / "all_dofs_knee_joints.html"),
            "ankle_torque": str(trial_output_dir / "all_dofs_ankle_joints.html"),
            "rotation_jacobian_comparison": str(trial_output_dir / "rotation_jacobian_comparison.html"),
        }
    else:
        plot_artifacts = {}

    mae_report, stance_results = analyze_stance_phase_torques(
        evaluation_predictions,
        ground_truth,
        data,
        str(trial_output_dir),
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
    if metrics is not None:
        metrics["bilateral_stance_mae_report"] = bilateral_stance_mae_report
        metrics["stance_cop_mae_percent_height"] = _extract_stance_cop_mae_percent_height(mae_report)

    if opencap_val_dataset and input_source_norm != "mocap" and secondary_predictions is not None and secondary_metrics is not None:
        opencap_stance_cop_mae = _extract_stance_cop_mae_percent_height(mae_report)
        motioncapture_stance_cop_mae = _extract_stance_cop_mae_percent_height(secondary_mae_report)
        metrics["opencap_input_cop_rmse"] = float(metrics.get("cop_rmse", np.nan))
        metrics["opencap_input_grf_rmse"] = float(metrics.get("grf_rmse", np.nan))
        metrics["opencap_input_moments_rmse"] = float(metrics.get("moments_rmse", np.nan))
        metrics["opencap_input_torque_rmse"] = float(metrics["torque_rmse"])
        metrics["opencap_input_torque_rmse_bwh"] = float(metrics["torque_rmse_bwh"])
        metrics["opencap_input_torque_mae_bwh"] = float(metrics["torque_mae_bwh"])
        metrics["opencap_input_torque_metric_signal"] = str(metrics.get("torque_metric_signal", "unknown"))
        metrics["opencap_input_qfrc_inverse_processed_vs_gt_rmse"] = float(metrics.get("qfrc_inverse_processed_vs_gt_rmse", np.nan))
        metrics["opencap_input_qfrc_inverse_pred_vs_gt_rmse"] = float(metrics.get("qfrc_inverse_pred_vs_gt_rmse", np.nan))
        metrics["opencap_input_qfrc_inverse_processed_minus_pred_rmse"] = float(metrics.get("qfrc_inverse_processed_minus_pred_rmse", np.nan))
        metrics["opencap_input_qfrc_inverse_pred_vs_processed_rmse"] = float(metrics.get("qfrc_inverse_pred_vs_processed_rmse", np.nan))
        metrics["opencap_input_qfrc_inverse_pred_vs_mocap_rmse"] = float(metrics.get("qfrc_inverse_pred_vs_mocap_rmse", np.nan))
        metrics["opencap_input_qfrc_inverse_processed_vs_mocap_rmse"] = float(metrics.get("qfrc_inverse_processed_vs_mocap_rmse", np.nan))
        metrics["opencap_input_stance_cop_mae_percent_height"] = opencap_stance_cop_mae
        metrics["motioncapture_input_cop_rmse"] = float(secondary_metrics.get("cop_rmse", np.nan))
        metrics["motioncapture_input_grf_rmse"] = float(secondary_metrics.get("grf_rmse", np.nan))
        metrics["motioncapture_input_moments_rmse"] = float(secondary_metrics.get("moments_rmse", np.nan))
        metrics["motioncapture_input_torque_rmse"] = float(secondary_metrics.get("torque_rmse", np.nan))
        metrics["motioncapture_input_torque_rmse_bwh"] = float(secondary_metrics.get("torque_rmse_bwh", np.nan))
        metrics["motioncapture_input_torque_mae_bwh"] = float(secondary_metrics.get("torque_mae_bwh", np.nan))
        metrics["motioncapture_input_torque_metric_signal"] = str(secondary_metrics.get("torque_metric_signal", "unknown"))
        metrics["motioncapture_input_qfrc_inverse_processed_vs_gt_rmse"] = float(secondary_metrics.get("qfrc_inverse_processed_vs_gt_rmse", np.nan))
        metrics["motioncapture_input_qfrc_inverse_pred_vs_gt_rmse"] = float(secondary_metrics.get("qfrc_inverse_pred_vs_gt_rmse", np.nan))
        metrics["motioncapture_input_qfrc_inverse_processed_minus_pred_rmse"] = float(secondary_metrics.get("qfrc_inverse_processed_minus_pred_rmse", np.nan))
        metrics["motioncapture_input_qfrc_inverse_pred_vs_processed_rmse"] = float(secondary_metrics.get("qfrc_inverse_pred_vs_processed_rmse", np.nan))
        metrics["motioncapture_input_qfrc_inverse_pred_vs_mocap_rmse"] = float(secondary_metrics.get("qfrc_inverse_pred_vs_mocap_rmse", np.nan))
        metrics["motioncapture_input_qfrc_inverse_processed_vs_mocap_rmse"] = float(secondary_metrics.get("qfrc_inverse_processed_vs_mocap_rmse", np.nan))
        metrics["motioncapture_input_stance_cop_mae_percent_height"] = motioncapture_stance_cop_mae
        metrics["video_input_torque_rmse"] = metrics["opencap_input_torque_rmse"]
        metrics["video_input_torque_rmse_bwh"] = metrics["opencap_input_torque_rmse_bwh"]
        metrics["video_input_torque_mae_bwh"] = metrics["opencap_input_torque_mae_bwh"]
        metrics["video_input_stance_cop_mae_percent_height"] = opencap_stance_cop_mae
        metrics["mocap_input_torque_rmse"] = metrics["motioncapture_input_torque_rmse"]
        metrics["mocap_input_torque_rmse_bwh"] = metrics["motioncapture_input_torque_rmse_bwh"]
        metrics["mocap_input_torque_mae_bwh"] = metrics["motioncapture_input_torque_mae_bwh"]
        metrics["mocap_input_cop_rmse"] = metrics["motioncapture_input_cop_rmse"]
        metrics["mocap_input_grf_rmse"] = metrics["motioncapture_input_grf_rmse"]
        metrics["mocap_input_moments_rmse"] = metrics["motioncapture_input_moments_rmse"]
        metrics["mocap_input_stance_cop_mae_percent_height"] = motioncapture_stance_cop_mae
        metrics["opencap_input_bilateral_stance_mae_report"] = bilateral_stance_mae_report
        motioncapture_bilateral_report = secondary_metrics.get("bilateral_stance_mae_report")
        if isinstance(motioncapture_bilateral_report, dict):
            metrics["motioncapture_input_bilateral_stance_mae_report"] = motioncapture_bilateral_report

    if not lightweight:
        with open(trial_output_dir / f"{safe_trial_name}_stance_mae_both_legs.json", "w", encoding="utf-8") as f:
            json.dump(bilateral_stance_mae_report, f, indent=2)

        np.savez_compressed(
            trial_output_dir / "mod_q_predictions.npz",
            **{
                key: value
                for key, value in predictions.items()
                if isinstance(value, np.ndarray) and not str(key).startswith("_")
            },
        )
        summary_payload = {
            "checkpoint": checkpoint_path,
            "trial": trial_name,
            "trial_path": str(trial_path),
            "xml_path": str(xml_path),
            "output_dim": int(output_np.shape[-1]),
            "window_size": int(resolved_window_size),
            "stride": int(resolved_stride),
            "prediction_margin_frames": int(resolved_prediction_margin_frames),
            "model_type": MOD_Q_MODEL_TYPE,
            "metadata": metadata,
            "window_meta": window_meta,
            "window_counts_mean": float(np.mean(counts[evaluation_mask])) if np.any(evaluation_mask) else 0.0,
            "split_status": split_status,
            "input_source": input_source_norm,
            "input_source_label": input_source_display,
            "ground_truth_source": data.get("ground_truth_source", "selected input source"),
            "metrics": metrics,
            "plot_artifacts": plot_artifacts,
        }
        summary_payload.update(rotation_metrics)
        with open(trial_output_dir / "mod_q_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        trial_display = f"{trial_name} [{split_status}]"
        if rotation_comparison_stats is not None and jacobian_comparison_stats is not None:
            create_rotation_jacobian_comparison_plot(
                trial_display,
                rotation_comparison_stats,
                jacobian_comparison_stats,
                knee_torque_comparison_stats,
                save_path=str(trial_output_dir / "rotation_jacobian_comparison.html"),
            )

        plot_time_axis = time_axis
        plot_predictions = _mask_prediction_dict_for_display(
            {k: v for k, v in predictions.items() if not k.startswith("_")},
            evaluation_mask,
        )
        plot_metric_predictions = _mask_prediction_dict_for_display(
            predictions.get("_metric_view", plot_predictions),
            evaluation_mask,
        )
        plot_ground_truth = ground_truth
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
                len(plot_predictions["cop"]),
                len(plot_ground_truth["cop"]),
                len(plot_secondary_predictions["cop"]),
            )
            plot_time_axis = time_axis[:min_plot_len]
            plot_predictions = _truncate_timeseries_mapping(plot_predictions, min_plot_len)
            plot_metric_predictions = _truncate_timeseries_mapping(plot_metric_predictions, min_plot_len)
            plot_ground_truth = _truncate_timeseries_mapping(plot_ground_truth, min_plot_len)
            plot_secondary_predictions = _truncate_timeseries_mapping(plot_secondary_predictions, min_plot_len)
            plot_metric_secondary_predictions = _truncate_timeseries_mapping(plot_metric_secondary_predictions, min_plot_len)

        if not no_plots:
            create_timeseries_plot(
                plot_time_axis,
                plot_predictions,
                plot_secondary_predictions,
                plot_ground_truth,
                trial_display,
                side="Right",
                save_path=str(trial_output_dir / "timeseries_right.html"),
                pred_label=primary_label,
                alt_pred_label=secondary_label,
                evaluation_mask=evaluation_mask[: len(plot_time_axis)],
                metric_predictions=plot_metric_predictions,
                metric_predictions_alt=plot_metric_secondary_predictions,
                prediction_margin_frames=resolved_prediction_margin_frames,
            )
            create_timeseries_plot(
                plot_time_axis,
                plot_predictions,
                plot_secondary_predictions,
                plot_ground_truth,
                trial_display,
                side="Left",
                save_path=str(trial_output_dir / "timeseries_left.html"),
                pred_label=primary_label,
                alt_pred_label=secondary_label,
                evaluation_mask=evaluation_mask[: len(plot_time_axis)],
                metric_predictions=plot_metric_predictions,
                metric_predictions_alt=plot_metric_secondary_predictions,
                prediction_margin_frames=resolved_prediction_margin_frames,
            )
            create_bilateral_cop_grf_plot(
                plot_time_axis,
                plot_predictions,
                plot_secondary_predictions,
                plot_ground_truth,
                trial_display,
                save_path=str(trial_output_dir / "cop_grf_bilateral.html"),
                pred_label=primary_label,
                alt_pred_label=secondary_label,
                evaluation_mask=evaluation_mask[: len(plot_time_axis)],
                metric_predictions=plot_metric_predictions,
                metric_predictions_alt=plot_metric_secondary_predictions,
            )
            create_error_distribution_plot(
                plot_predictions,
                plot_ground_truth,
                trial_display,
                save_path=str(trial_output_dir / "errors.html"),
                evaluation_mask=evaluation_mask[: len(plot_time_axis)],
                metric_predictions=plot_metric_predictions,
            )
            primary_qfrc_for_plots = (
                plot_predictions.get("qfrc_inverse")
                if input_source_norm == "mocap"
                else plot_predictions.get("qfrc_inverse")
            )
            secondary_qfrc_for_plots = (
                plot_secondary_predictions.get("qfrc_inverse")
                if plot_secondary_predictions is not None and plot_secondary_predictions.get("qfrc_inverse") is not None
                else plot_ground_truth.get("qfrc_inverse_mocap", primary_qfrc_for_plots)
            )

            create_all_dofs_plot(
                plot_time_axis,
                plot_predictions,
                plot_secondary_predictions,
                plot_ground_truth,
                trial_display,
                qfrc_inverse_pred=primary_qfrc_for_plots,
                qfrc_inverse_alt=secondary_qfrc_for_plots,
                save_path=str(trial_output_dir / "all_dofs.html"),
                pred_label=primary_label,
                alt_pred_label=secondary_label,
                evaluation_mask=evaluation_mask[: len(plot_time_axis)],
                metric_predictions=plot_metric_predictions,
                metric_predictions_alt=plot_metric_secondary_predictions,
                prediction_margin_frames=resolved_prediction_margin_frames,
            )
            
            # Focused knee / rotation / jacobian comparison plots.
            try:
                dof_names = get_dof_names()
                knee_r_idx = dof_names.index("knee_angle_r")
                knee_l_idx = dof_names.index("knee_angle_l")

                comparison_refs = _mask_prediction_dict_for_display(
                    {
                        "qfrc_inverse_processed": ground_truth.get("qfrc_inverse_processed"),
                        "qfrc_inverse_mocap": ground_truth.get("qfrc_inverse_mocap"),
                        "rot_w_to_ga_processed": np.asarray(data["rot_w_to_ga"], dtype=np.float32),
                        "rot_w_to_ga_mocap": np.asarray(ground_truth["rot_w_to_ga"], dtype=np.float32),
                        "jacp_processed": np.asarray(data["jacp"], dtype=np.float32),
                        "jacr_processed": np.asarray(data["jacr"], dtype=np.float32),
                        "jacp_mocap": np.asarray(data["gt_jacp"], dtype=np.float32),
                        "jacr_mocap": np.asarray(data["gt_jacr"], dtype=np.float32),
                        "cop_mocap": np.asarray(ground_truth["cop"], dtype=np.float32),
                        "grf_mocap": np.asarray(ground_truth["grf"], dtype=np.float32),
                        "moments_mocap": np.asarray(ground_truth["moments"], dtype=np.float32),
                        "ankle_heights": np.asarray(data["ankle_heights"], dtype=np.float32),
                    },
                    evaluation_mask,
                )
                comparison_refs = _truncate_timeseries_mapping(comparison_refs, len(plot_time_axis))

                qfrc_pred = plot_predictions.get("qfrc_inverse")
                qfrc_proc = comparison_refs.get("qfrc_inverse_processed")
                qfrc_mocap = comparison_refs.get("qfrc_inverse_mocap")
                if qfrc_pred is not None and qfrc_proc is not None and qfrc_mocap is not None:
                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Right Knee qfrc_inverse", "Left Knee qfrc_inverse"),
                    )
                    for row, knee_idx, knee_name in (
                        (1, knee_r_idx, "Right"),
                        (2, knee_l_idx, "Left"),
                    ):
                        fig.add_trace(go.Scatter(x=plot_time_axis, y=qfrc_pred[:, knee_idx], mode="lines", name="Predicted"), row=row, col=1)
                        fig.add_trace(go.Scatter(x=plot_time_axis, y=qfrc_proc[:, knee_idx], mode="lines", name="ProcessedData"), row=row, col=1)
                        fig.add_trace(go.Scatter(x=plot_time_axis, y=qfrc_mocap[:, knee_idx], mode="lines", name="MoCap GT"), row=row, col=1)
                        fig.update_yaxes(title_text=f"{knee_name} Knee Torque (Nm)", row=row, col=1)
                    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                    fig.update_layout(title=f"{trial_display}: Knee qfrc_inverse Comparison", height=800)
                    fig.write_html(str(trial_output_dir / "qfrc_knee_comparison.html"))

                rot_pred = plot_predictions.get("rot_w_to_ga")
                rot_proc = comparison_refs.get("rot_w_to_ga_processed")
                rot_mocap = comparison_refs.get("rot_w_to_ga_mocap")
                if rot_pred is not None and rot_proc is not None and rot_mocap is not None:
                    pred_vs_gt_deg = _rotation_geodesic_timeseries_deg(rot_pred, rot_mocap)
                    proc_vs_gt_deg = _rotation_geodesic_timeseries_deg(rot_proc, rot_mocap)
                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Right Calcaneus Rotation Geodesic Error", "Left Calcaneus Rotation Geodesic Error"),
                    )
                    for row, side_idx, side_name in ((1, 0, "Right"), (2, 1, "Left")):
                        pred_rmse = float(np.sqrt(np.mean(np.square(pred_vs_gt_deg[:, side_idx]))))
                        proc_rmse = float(np.sqrt(np.mean(np.square(proc_vs_gt_deg[:, side_idx]))))
                        fig.add_trace(go.Scatter(x=plot_time_axis, y=pred_vs_gt_deg[:, side_idx], mode="lines", name="Predicted vs GT"), row=row, col=1)
                        fig.add_trace(go.Scatter(x=plot_time_axis, y=proc_vs_gt_deg[:, side_idx], mode="lines", name="ProcessedData vs GT"), row=row, col=1)
                        fig.update_yaxes(title_text=f"{side_name} Geodesic Error (deg)", row=row, col=1)
                        fig.add_annotation(
                            x=0.99,
                            y=0.95 - 0.48 * (row - 1),
                            xref="paper",
                            yref="paper",
                            text=f"{side_name} RMSE: Pred {pred_rmse:.3f} deg | OC {proc_rmse:.3f} deg",
                            showarrow=False,
                            xanchor="right",
                        )
                    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                    fig.update_layout(title=f"{trial_display}: Rotation Geodesic Comparison to MoCap GT", height=800)
                    fig.write_html(str(trial_output_dir / "rotation_geodesic_comparison.html"))

                jacp_pred = plot_predictions.get("jacp")
                jacr_pred = plot_predictions.get("jacr")
                jacp_proc = comparison_refs.get("jacp_processed")
                jacr_proc = comparison_refs.get("jacr_processed")
                jacp_mocap = comparison_refs.get("jacp_mocap")
                jacr_mocap = comparison_refs.get("jacr_mocap")
                cop_mocap = comparison_refs.get("cop_mocap")
                grf_mocap = comparison_refs.get("grf_mocap")
                moments_mocap = comparison_refs.get("moments_mocap")
                ankle_heights_plot = comparison_refs.get("ankle_heights")
                if (
                    jacp_pred is not None and jacr_pred is not None
                    and jacp_proc is not None and jacr_proc is not None
                    and jacp_mocap is not None and jacr_mocap is not None
                    and cop_mocap is not None and grf_mocap is not None
                    and moments_mocap is not None and ankle_heights_plot is not None
                    and rot_mocap is not None
                ):
                    gt_wrench = _build_calcaneus_applied_wrench(
                        cop_xz=cop_mocap,
                        ankle_heights=ankle_heights_plot,
                        grf_world=grf_mocap,
                        moments_world=moments_mocap,
                        rot_w_to_ga=rot_mocap,
                    )
                    tau_pred = _compute_tau_from_applied_wrench(
                        jacp=jacp_pred,
                        jacr=jacr_pred,
                        force_r=gt_wrench["force_r"],
                        force_l=gt_wrench["force_l"],
                        total_moment_r=gt_wrench["total_moment_r"],
                        total_moment_l=gt_wrench["total_moment_l"],
                    )
                    tau_proc = _compute_tau_from_applied_wrench(
                        jacp=jacp_proc,
                        jacr=jacr_proc,
                        force_r=gt_wrench["force_r"],
                        force_l=gt_wrench["force_l"],
                        total_moment_r=gt_wrench["total_moment_r"],
                        total_moment_l=gt_wrench["total_moment_l"],
                    )
                    tau_gt = _compute_tau_from_applied_wrench(
                        jacp=jacp_mocap,
                        jacr=jacr_mocap,
                        force_r=gt_wrench["force_r"],
                        force_l=gt_wrench["force_l"],
                        total_moment_r=gt_wrench["total_moment_r"],
                        total_moment_l=gt_wrench["total_moment_l"],
                    )

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Right Knee Torque from Calcaneus Wrench", "Left Knee Torque from Calcaneus Wrench"),
                    )
                    for row, knee_idx, side_name in ((1, knee_r_idx, "Right"), (2, knee_l_idx, "Left")):
                        pred_rmse = float(np.sqrt(np.mean(np.square(tau_pred[:, knee_idx] - tau_gt[:, knee_idx]))))
                        proc_rmse = float(np.sqrt(np.mean(np.square(tau_proc[:, knee_idx] - tau_gt[:, knee_idx]))))
                        fig.add_trace(go.Scatter(x=plot_time_axis, y=tau_pred[:, knee_idx], mode="lines", name="Predicted Jacobian"), row=row, col=1)
                        fig.add_trace(go.Scatter(x=plot_time_axis, y=tau_proc[:, knee_idx], mode="lines", name="ProcessedData Jacobian"), row=row, col=1)
                        fig.add_trace(go.Scatter(x=plot_time_axis, y=tau_gt[:, knee_idx], mode="lines", name="MoCap GT Jacobian"), row=row, col=1)
                        fig.update_yaxes(title_text=f"{side_name} Knee Torque (Nm)", row=row, col=1)
                        fig.add_annotation(
                            x=0.99,
                            y=0.95 - 0.48 * (row - 1),
                            xref="paper",
                            yref="paper",
                            text=f"{side_name} RMSE: Pred {pred_rmse:.3f} Nm | OC {proc_rmse:.3f} Nm",
                            showarrow=False,
                            xanchor="right",
                        )
                    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                    fig.update_layout(title=f"{trial_display}: Knee Torque from Common MoCap Calcaneus Wrench", height=800)
                    fig.write_html(str(trial_output_dir / "jacobian_knee_torque_comparison.html"))
            except Exception as e:
                print(f"Failed to generate focused knee/rotation/jacobian plots: {e}")

        if make_graph:
            make_publication_plots(
                predictions,
                ground_truth,
                trial_name,
                str(trial_output_dir),
                bundle["checkpoint_path"].parent.name,
            )

    metrics["trial_name"] = trial_name
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "metadata": metadata,
        "trial_path": str(trial_path),
        "xml_path": str(xml_path),
        "rotation_metrics": rotation_metrics,
        "metrics": metrics,
        "mae_report": mae_report,
        "stance_results": stance_results,
        "secondary_mae_report": secondary_mae_report,
        "secondary_predictions": secondary_predictions,
        "secondary_metrics": secondary_metrics,
        "trial_output_dir": str(trial_output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for a train_mod_q checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--trial_name", type=str, default=None)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--all_val", action="store_true")
    parser.add_argument("--output", type=str, default="infer_mod_q_output")
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--prediction_margin_frames", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--make_graph", action="store_true")
    parser.add_argument("--OpenCapValDataset", action="store_true", help="Load MoCap ground truth from subject folders")
    parser.add_argument("--OpenCapDataset", action="store_true", help="Alias for --OpenCapValDataset")
    parser.add_argument(
        "--clear_jax_cache_every",
        type=int,
        default=0,
        help="Clear JAX compile caches every N trials (0 disables).",
    )
    args = parser.parse_args()
    if args.OpenCapDataset:
        args.OpenCapValDataset = True

    checkpoint_path = Path(args.checkpoint)
    bundle = _load_mod_q_checkpoint_bundle(checkpoint_path)

    trials_to_run: List[str] = []
    if args.trial_name:
        trials_to_run.append(args.trial_name)
    elif args.test_json:
        trials_to_run = _load_trials_from_json(Path(args.test_json))
        print(f"📂 Loaded {len(trials_to_run)} trials from {args.test_json}")
    elif args.all_val:
        trials_to_run = [
            _extract_trial_name(entry)
            for entry in bundle.get("val_trials", [])
            if str(_extract_trial_name(entry)).strip()
        ]
        print(f"📂 Loaded {len(trials_to_run)} validation trials")
    elif args.OpenCapValDataset and Path("OpenCap_trials.json").exists():
        default_json = Path("OpenCap_trials.json")
        trials_to_run = _load_trials_from_json(default_json)
        print(f"📂 Loaded {len(trials_to_run)} OpenCap trials from {default_json}")

    if not trials_to_run:
        raise SystemExit("No trials specified. Use --trial_name, --test_json, --all_val, or --OpenCapValDataset with OpenCap_trials.json present.")

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    all_metrics: List[Dict[str, Any]] = []
    overall_mae: Dict[str, Dict[str, float]] = {}
    overall_mae_motioncapture: Dict[str, Dict[str, float]] = {}
    aggregated_stance_data: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for trial_idx, trial in enumerate(tqdm(trials_to_run, desc="Running mod_q inference"), start=1):
        try:
            result = run_mod_q_inference(
                checkpoint_path=str(checkpoint_path),
                data_dir=args.data_dir,
                trial_name=trial,
                output_dir=str(output_base),
                window_size=args.window_size,
                stride=args.stride,
                prediction_margin_frames=args.prediction_margin_frames,
                no_plots=args.no_plots,
                lightweight=args.lightweight,
                make_graph=args.make_graph,
                opencap_val_dataset=args.OpenCapValDataset,
                bundle=bundle,
            )
            mae_report = result.get("mae_report")
            metrics = result.get("metrics")
            stance_results = result.get("stance_results")
            secondary_mae_report = result.get("secondary_mae_report")
            if mae_report:
                overall_mae[trial] = mae_report
            if secondary_mae_report:
                overall_mae_motioncapture[trial] = secondary_mae_report
            if metrics:
                all_metrics.append(metrics)
            if stance_results:
                for dof_name, stance_payload in stance_results.items():
                    if stance_payload is None:
                        continue
                    aggregated_stance_data.setdefault(dof_name, {"pred": [], "gt": []})
                    aggregated_stance_data[dof_name]["pred"].append(stance_payload["pred"])
                    aggregated_stance_data[dof_name]["gt"].append(stance_payload["gt"])
        except Exception as exc:
            print(f"\n❌ Error running mod_q inference on {trial}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            if args.clear_jax_cache_every > 0 and (trial_idx % args.clear_jax_cache_every == 0):
                try:
                    jax.clear_caches()
                except Exception:
                    pass

    if all_metrics and any("motioncapture_input_torque_mae_bwh" in m for m in all_metrics):
        trial_details = {}
        opencap_mae_vals: List[float] = []
        motioncapture_mae_vals: List[float] = []
        opencap_rmse_vals: List[float] = []
        motioncapture_rmse_vals: List[float] = []
        opencap_qfrc_processed_rmse_vals: List[float] = []
        motioncapture_qfrc_processed_rmse_vals: List[float] = []
        opencap_qfrc_pred_rmse_vals: List[float] = []
        motioncapture_qfrc_pred_rmse_vals: List[float] = []
        opencap_qfrc_pred_vs_processed_vals: List[float] = []
        motioncapture_qfrc_pred_vs_processed_vals: List[float] = []
        opencap_qfrc_pred_vs_mocap_vals: List[float] = []
        motioncapture_qfrc_pred_vs_mocap_vals: List[float] = []
        opencap_qfrc_processed_vs_mocap_vals: List[float] = []
        motioncapture_qfrc_processed_vs_mocap_vals: List[float] = []
        opencap_qfrc_improvement_vals: List[float] = []
        motioncapture_qfrc_improvement_vals: List[float] = []
        stance_cop_keys = ("COP_X_Right", "COP_Z_Right", "COP_X_Left", "COP_Z_Left")
        opencap_stance_cop_vals = {key: [] for key in stance_cop_keys}
        motioncapture_stance_cop_vals = {key: [] for key in stance_cop_keys}
        for metrics in all_metrics:
            current_trial_name = metrics.get("trial_name", "unknown_trial")
            opencap_mae = float(metrics.get("opencap_input_torque_mae_bwh", metrics.get("torque_mae_bwh", np.nan)))
            motioncapture_mae = float(metrics.get("motioncapture_input_torque_mae_bwh", np.nan))
            opencap_rmse = float(metrics.get("opencap_input_torque_rmse_bwh", metrics.get("torque_rmse_bwh", np.nan)))
            motioncapture_rmse = float(metrics.get("motioncapture_input_torque_rmse_bwh", np.nan))
            opencap_qfrc_processed_rmse = float(metrics.get("opencap_input_qfrc_inverse_processed_vs_gt_rmse", metrics.get("qfrc_inverse_processed_vs_gt_rmse", np.nan)))
            motioncapture_qfrc_processed_rmse = float(metrics.get("motioncapture_input_qfrc_inverse_processed_vs_gt_rmse", np.nan))
            opencap_qfrc_pred_rmse = float(metrics.get("opencap_input_qfrc_inverse_pred_vs_gt_rmse", metrics.get("qfrc_inverse_pred_vs_gt_rmse", np.nan)))
            motioncapture_qfrc_pred_rmse = float(metrics.get("motioncapture_input_qfrc_inverse_pred_vs_gt_rmse", np.nan))
            opencap_qfrc_pred_vs_processed = float(metrics.get("opencap_input_qfrc_inverse_pred_vs_processed_rmse", metrics.get("qfrc_inverse_pred_vs_processed_rmse", np.nan)))
            motioncapture_qfrc_pred_vs_processed = float(metrics.get("motioncapture_input_qfrc_inverse_pred_vs_processed_rmse", np.nan))
            opencap_qfrc_pred_vs_mocap = float(metrics.get("opencap_input_qfrc_inverse_pred_vs_mocap_rmse", metrics.get("qfrc_inverse_pred_vs_mocap_rmse", np.nan)))
            motioncapture_qfrc_pred_vs_mocap = float(metrics.get("motioncapture_input_qfrc_inverse_pred_vs_mocap_rmse", np.nan))
            opencap_qfrc_processed_vs_mocap = float(metrics.get("opencap_input_qfrc_inverse_processed_vs_mocap_rmse", metrics.get("qfrc_inverse_processed_vs_mocap_rmse", np.nan)))
            motioncapture_qfrc_processed_vs_mocap = float(metrics.get("motioncapture_input_qfrc_inverse_processed_vs_mocap_rmse", np.nan))
            opencap_qfrc_improvement = float(metrics.get("opencap_input_qfrc_inverse_processed_minus_pred_rmse", metrics.get("qfrc_inverse_processed_minus_pred_rmse", np.nan)))
            motioncapture_qfrc_improvement = float(metrics.get("motioncapture_input_qfrc_inverse_processed_minus_pred_rmse", np.nan))
            opencap_stance_cop = metrics.get(
                "opencap_input_stance_cop_mae_percent_height",
                metrics.get("stance_cop_mae_percent_height", {}),
            )
            motioncapture_stance_cop = metrics.get("motioncapture_input_stance_cop_mae_percent_height", {})
            trial_details[current_trial_name] = {
                "opencap_input": {
                    "torque_mae_bwh_percent": opencap_mae,
                    "torque_rmse_bwh_percent": opencap_rmse,
                    "qfrc_inverse_processed_vs_gt_rmse": opencap_qfrc_processed_rmse,
                    "qfrc_inverse_pred_vs_gt_rmse": opencap_qfrc_pred_rmse,
                    "qfrc_inverse_pred_vs_processed_rmse": opencap_qfrc_pred_vs_processed,
                    "qfrc_inverse_pred_vs_mocap_rmse": opencap_qfrc_pred_vs_mocap,
                    "qfrc_inverse_processed_vs_mocap_rmse": opencap_qfrc_processed_vs_mocap,
                    "qfrc_inverse_processed_minus_pred_rmse": opencap_qfrc_improvement,
                    "stance_cop_mae_percent_height": {
                        key: float(opencap_stance_cop[key])
                        for key in stance_cop_keys
                        if key in opencap_stance_cop and np.isfinite(opencap_stance_cop[key])
                    },
                },
                "motioncapture_input": {
                    "torque_mae_bwh_percent": motioncapture_mae,
                    "torque_rmse_bwh_percent": motioncapture_rmse,
                    "qfrc_inverse_processed_vs_gt_rmse": motioncapture_qfrc_processed_rmse,
                    "qfrc_inverse_pred_vs_gt_rmse": motioncapture_qfrc_pred_rmse,
                    "qfrc_inverse_pred_vs_processed_rmse": motioncapture_qfrc_pred_vs_processed,
                    "qfrc_inverse_pred_vs_mocap_rmse": motioncapture_qfrc_pred_vs_mocap,
                    "qfrc_inverse_processed_vs_mocap_rmse": motioncapture_qfrc_processed_vs_mocap,
                    "qfrc_inverse_processed_minus_pred_rmse": motioncapture_qfrc_improvement,
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
            if np.isfinite(opencap_qfrc_processed_rmse):
                opencap_qfrc_processed_rmse_vals.append(opencap_qfrc_processed_rmse)
            if np.isfinite(motioncapture_qfrc_processed_rmse):
                motioncapture_qfrc_processed_rmse_vals.append(motioncapture_qfrc_processed_rmse)
            if np.isfinite(opencap_qfrc_pred_rmse):
                opencap_qfrc_pred_rmse_vals.append(opencap_qfrc_pred_rmse)
            if np.isfinite(motioncapture_qfrc_pred_rmse):
                motioncapture_qfrc_pred_rmse_vals.append(motioncapture_qfrc_pred_rmse)
            if np.isfinite(opencap_qfrc_pred_vs_processed):
                opencap_qfrc_pred_vs_processed_vals.append(opencap_qfrc_pred_vs_processed)
            if np.isfinite(motioncapture_qfrc_pred_vs_processed):
                motioncapture_qfrc_pred_vs_processed_vals.append(motioncapture_qfrc_pred_vs_processed)
            if np.isfinite(opencap_qfrc_pred_vs_mocap):
                opencap_qfrc_pred_vs_mocap_vals.append(opencap_qfrc_pred_vs_mocap)
            if np.isfinite(motioncapture_qfrc_pred_vs_mocap):
                motioncapture_qfrc_pred_vs_mocap_vals.append(motioncapture_qfrc_pred_vs_mocap)
            if np.isfinite(opencap_qfrc_processed_vs_mocap):
                opencap_qfrc_processed_vs_mocap_vals.append(opencap_qfrc_processed_vs_mocap)
            if np.isfinite(motioncapture_qfrc_processed_vs_mocap):
                motioncapture_qfrc_processed_vs_mocap_vals.append(motioncapture_qfrc_processed_vs_mocap)
            if np.isfinite(opencap_qfrc_improvement):
                opencap_qfrc_improvement_vals.append(opencap_qfrc_improvement)
            if np.isfinite(motioncapture_qfrc_improvement):
                motioncapture_qfrc_improvement_vals.append(motioncapture_qfrc_improvement)
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
                    "qfrc_inverse_processed_vs_gt_rmse": float(np.mean(opencap_qfrc_processed_rmse_vals)) if opencap_qfrc_processed_rmse_vals else None,
                    "qfrc_inverse_pred_vs_gt_rmse": float(np.mean(opencap_qfrc_pred_rmse_vals)) if opencap_qfrc_pred_rmse_vals else None,
                    "qfrc_inverse_pred_vs_processed_rmse": float(np.mean(opencap_qfrc_pred_vs_processed_vals)) if opencap_qfrc_pred_vs_processed_vals else None,
                    "qfrc_inverse_pred_vs_mocap_rmse": float(np.mean(opencap_qfrc_pred_vs_mocap_vals)) if opencap_qfrc_pred_vs_mocap_vals else None,
                    "qfrc_inverse_processed_vs_mocap_rmse": float(np.mean(opencap_qfrc_processed_vs_mocap_vals)) if opencap_qfrc_processed_vs_mocap_vals else None,
                    "qfrc_inverse_processed_minus_pred_rmse": float(np.mean(opencap_qfrc_improvement_vals)) if opencap_qfrc_improvement_vals else None,
                    "stance_cop_mae_percent_height": {
                        key: float(np.mean(values)) if values else None
                        for key, values in opencap_stance_cop_vals.items()
                    },
                },
                "motioncapture_input": {
                    "torque_mae_bwh_percent": float(np.mean(motioncapture_mae_vals)) if motioncapture_mae_vals else None,
                    "torque_rmse_bwh_percent": float(np.mean(motioncapture_rmse_vals)) if motioncapture_rmse_vals else None,
                    "qfrc_inverse_processed_vs_gt_rmse": float(np.mean(motioncapture_qfrc_processed_rmse_vals)) if motioncapture_qfrc_processed_rmse_vals else None,
                    "qfrc_inverse_pred_vs_gt_rmse": float(np.mean(motioncapture_qfrc_pred_rmse_vals)) if motioncapture_qfrc_pred_rmse_vals else None,
                    "qfrc_inverse_pred_vs_processed_rmse": float(np.mean(motioncapture_qfrc_pred_vs_processed_vals)) if motioncapture_qfrc_pred_vs_processed_vals else None,
                    "qfrc_inverse_pred_vs_mocap_rmse": float(np.mean(motioncapture_qfrc_pred_vs_mocap_vals)) if motioncapture_qfrc_pred_vs_mocap_vals else None,
                    "qfrc_inverse_processed_vs_mocap_rmse": float(np.mean(motioncapture_qfrc_processed_vs_mocap_vals)) if motioncapture_qfrc_processed_vs_mocap_vals else None,
                    "qfrc_inverse_processed_minus_pred_rmse": float(np.mean(motioncapture_qfrc_improvement_vals)) if motioncapture_qfrc_improvement_vals else None,
                    "stance_cop_mae_percent_height": {
                        key: float(np.mean(values)) if values else None
                        for key, values in motioncapture_stance_cop_vals.items()
                    },
                },
            },
            "trial_details": trial_details,
        }
        comparison_summary_path = output_base / "overall_input_comparison_summary.json"
        with open(comparison_summary_path, "w", encoding="utf-8") as f:
            json.dump(comparison_summary, f, indent=2)
        print(f"✅ Saved OpenCap vs MotionCapture summary to: {comparison_summary_path}")

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
        metrics.get("trial_name", f"trial_{idx}"): metrics.get("motioncapture_input_bilateral_stance_mae_report")
        for idx, metrics in enumerate(all_metrics)
        if isinstance(metrics.get("motioncapture_input_bilateral_stance_mae_report"), dict)
    }
    if motioncapture_bilateral_trial_reports:
        bilateral_source_reports["opencap_input_mae"] = opencap_bilateral_trial_reports
        bilateral_source_reports["motioncapture_input_mae"] = motioncapture_bilateral_trial_reports
    elif opencap_bilateral_trial_reports:
        bilateral_source_reports["primary_input_mae"] = opencap_bilateral_trial_reports
    if bilateral_source_reports:
        bilateral_average_report = _compute_average_bilateral_stance_mae(bilateral_source_reports)
        bilateral_average_path = output_base / "overall_stance_mae_both_legs_average.json"
        with open(bilateral_average_path, "w", encoding="utf-8") as f:
            json.dump(bilateral_average_report, f, indent=2)
        print(f"✅ Saved averaged bilateral stance MAE report to: {bilateral_average_path}")

    if len(all_metrics) > 1:
        if overall_mae:
            dof_averages = _compute_average_mae_per_dof(overall_mae)
            mocap_dof_averages = _compute_average_mae_per_dof(overall_mae_motioncapture) if overall_mae_motioncapture else {}
            report_data = {
                "torque_metric_scope": "left_stance_selected_dofs",
                "torque_metric_side": "left",
                "torque_metric_phase": "stance",
                "torque_metric_dof_names": list(all_metrics[0].get("torque_metric_dof_names", [])) if all_metrics else [],
                "average_mae_per_dof": dof_averages,
                "average_mae_per_dof_opencap_input": dof_averages,
                "average_mae_per_dof_motioncapture_input": mocap_dof_averages,
                "trial_details": overall_mae,
                "trial_details_opencap_input": overall_mae,
                "trial_details_motioncapture_input": overall_mae_motioncapture,
            }
            mae_report_path = output_base / "overall_mae_report.json"
            with open(mae_report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"✅ Saved overall MAE report with averages to: {mae_report_path}")
            create_mae_boxplots(overall_mae, str(output_base), overall_mae_motioncapture=overall_mae_motioncapture)

        if aggregated_stance_data:
            stance_data_path = output_base / "aggregated_stance_data.pkl"
            with open(stance_data_path, "wb") as f:
                pickle.dump(aggregated_stance_data, f)
            summary_stats = {}
            for dof_name, stance_payload in aggregated_stance_data.items():
                all_preds = np.vstack(stance_payload["pred"])
                all_gts = np.vstack(stance_payload["gt"])
                diff = all_preds - all_gts
                summary_stats[dof_name] = {
                    "MAE": float(np.mean(np.abs(diff))),
                    "RMSE": float(np.sqrt(np.mean(diff ** 2))),
                    "Count": int(all_preds.shape[0]),
                }
            stats_path = output_base / "aggregated_stance_statistics.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(summary_stats, f, indent=2)
            print(f"✅ Saved aggregated stance statistics to: {stats_path}")

        create_summary_dashboard(all_metrics, str(output_base))


if __name__ == "__main__":
    main()
