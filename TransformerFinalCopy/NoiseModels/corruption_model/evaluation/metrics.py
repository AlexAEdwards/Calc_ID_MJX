from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.signal import welch

from corruption_model.preprocess.symmetry import build_left_right_index_map


def compute_basic_metrics(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(estimate, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
        "std": float(np.std(diff)),
    }


def covariance_similarity(a: np.ndarray, b: np.ndarray) -> float:
    cov_a = np.cov(np.asarray(a, dtype=np.float32).T)
    cov_b = np.cov(np.asarray(b, dtype=np.float32).T)
    return float(np.mean(np.abs(cov_a - cov_b)))


def compute_per_joint_residual_stats(residual: np.ndarray, dof_names: List[str]) -> Dict[str, object]:
    residual_np = np.asarray(residual, dtype=np.float32)
    if residual_np.ndim != 2:
        raise ValueError(f"Expected residual shape [T, D], got {residual_np.shape}")
    joint_rows = []
    for joint_idx in range(residual_np.shape[1]):
        joint_name = dof_names[joint_idx] if joint_idx < len(dof_names) else f"dof_{joint_idx}"
        joint_rows.append(
            {
                "joint_idx": int(joint_idx),
                "joint_name": joint_name,
                "mean": float(np.mean(residual_np[:, joint_idx])),
                "std": float(np.std(residual_np[:, joint_idx])),
            }
        )
    return {
        "num_frames": int(residual_np.shape[0]),
        "num_joints": int(residual_np.shape[1]),
        "joint_stats": joint_rows,
    }


def compute_per_joint_residual_stats_with_points(
    residual: np.ndarray,
    dof_names: List[str],
) -> Dict[str, object]:
    stats = compute_per_joint_residual_stats(residual, dof_names)
    residual_np = np.asarray(residual, dtype=np.float32)
    joint_rows = []
    for joint_idx, row in enumerate(stats["joint_stats"]):
        joint_rows.append(
            {
                **row,
                "points": residual_np[:, joint_idx].astype(np.float32).tolist(),
            }
        )
    stats["joint_stats"] = joint_rows
    return stats


def compute_average_curve_residual_stats(
    clean: np.ndarray,
    synthetic_curves: List[np.ndarray],
    dof_names: List[str],
) -> Dict[str, object]:
    clean_np = np.asarray(clean, dtype=np.float32)
    if not synthetic_curves:
        raise ValueError("synthetic_curves must not be empty")
    curve_stats = []
    for curve in synthetic_curves:
        residual = np.asarray(curve, dtype=np.float32) - clean_np
        curve_stats.append(compute_per_joint_residual_stats(residual, dof_names)["joint_stats"])

    joint_rows = []
    num_joints = clean_np.shape[1]
    for joint_idx in range(num_joints):
        joint_name = dof_names[joint_idx] if joint_idx < len(dof_names) else f"dof_{joint_idx}"
        means = [float(curve_stats[curve_idx][joint_idx]["mean"]) for curve_idx in range(len(curve_stats))]
        stds = [float(curve_stats[curve_idx][joint_idx]["std"]) for curve_idx in range(len(curve_stats))]
        joint_rows.append(
            {
                "joint_idx": int(joint_idx),
                "joint_name": joint_name,
                "mean": float(np.mean(means)),
                "std": float(np.mean(stds)),
                "curve_means": means,
                "curve_stds": stds,
            }
        )
    return {
        "num_curves": int(len(synthetic_curves)),
        "num_frames": int(clean_np.shape[0]),
        "num_joints": int(num_joints),
        "joint_stats": joint_rows,
    }


def compute_trialwise_residual_summary(
    clean: np.ndarray,
    synthetic_curves: List[np.ndarray],
    dof_names: List[str],
) -> Dict[str, object]:
    clean_np = np.asarray(clean, dtype=np.float32)
    if not synthetic_curves:
        raise ValueError("synthetic_curves must not be empty")
    num_joints = clean_np.shape[1]
    curve_rows = []
    for curve_idx, curve in enumerate(synthetic_curves):
        residual = np.asarray(curve, dtype=np.float32) - clean_np
        joint_stats = []
        for joint_idx in range(num_joints):
            joint_name = dof_names[joint_idx] if joint_idx < len(dof_names) else f"dof_{joint_idx}"
            joint_stats.append(
                {
                    "joint_idx": int(joint_idx),
                    "joint_name": joint_name,
                    "mean": float(np.mean(residual[:, joint_idx])),
                    "std": float(np.std(residual[:, joint_idx])),
                }
            )
        curve_rows.append({"curve_idx": int(curve_idx), "joint_stats": joint_stats})
    return {
        "num_curves": int(len(synthetic_curves)),
        "num_joints": int(num_joints),
        "curve_stats": curve_rows,
    }


def compute_real_trialwise_residual_summary(
    residual_trials: List[np.ndarray],
    dof_names: List[str],
) -> Dict[str, object]:
    if not residual_trials:
        raise ValueError("residual_trials must not be empty")
    num_joints = int(np.asarray(residual_trials[0], dtype=np.float32).shape[1])
    trial_rows = []
    for trial_idx, residual in enumerate(residual_trials):
        residual_np = np.asarray(residual, dtype=np.float32)
        joint_stats = []
        for joint_idx in range(num_joints):
            joint_name = dof_names[joint_idx] if joint_idx < len(dof_names) else f"dof_{joint_idx}"
            joint_stats.append(
                {
                    "joint_idx": int(joint_idx),
                    "joint_name": joint_name,
                    "mean": float(np.mean(residual_np[:, joint_idx])),
                    "std": float(np.std(residual_np[:, joint_idx])),
                }
            )
        trial_rows.append({"trial_idx": int(trial_idx), "joint_stats": joint_stats})
    return {
        "num_trials": int(len(residual_trials)),
        "num_joints": int(num_joints),
        "trial_stats": trial_rows,
    }


def compute_per_joint_psd_stats(
    residual: np.ndarray,
    dof_names: List[str],
    fs_hz: float,
) -> Dict[str, object]:
    residual_np = np.asarray(residual, dtype=np.float32)
    if residual_np.ndim != 2:
        raise ValueError(f"Expected residual shape [T, D], got {residual_np.shape}")
    joint_rows = []
    freqs_out = None
    for joint_idx in range(residual_np.shape[1]):
        joint_name = dof_names[joint_idx] if joint_idx < len(dof_names) else f"dof_{joint_idx}"
        freqs, psd = welch(residual_np[:, joint_idx], fs=float(fs_hz), nperseg=min(256, residual_np.shape[0]))
        if freqs_out is None:
            freqs_out = freqs.astype(np.float32)
        joint_rows.append(
            {
                "joint_idx": int(joint_idx),
                "joint_name": joint_name,
                "psd": psd.astype(np.float32).tolist(),
            }
        )
    return {
        "fs_hz": float(fs_hz),
        "num_frames": int(residual_np.shape[0]),
        "num_joints": int(residual_np.shape[1]),
        "freqs_hz": freqs_out.tolist() if freqs_out is not None else [],
        "joint_psd": joint_rows,
    }


def compute_average_curve_psd_stats(
    clean: np.ndarray,
    synthetic_curves: List[np.ndarray],
    dof_names: List[str],
    fs_hz: float,
) -> Dict[str, object]:
    clean_np = np.asarray(clean, dtype=np.float32)
    if not synthetic_curves:
        raise ValueError("synthetic_curves must not be empty")
    curve_psd_stats = []
    for curve in synthetic_curves:
        residual = np.asarray(curve, dtype=np.float32) - clean_np
        curve_psd_stats.append(compute_per_joint_psd_stats(residual, dof_names, fs_hz))
    freqs_hz = curve_psd_stats[0]["freqs_hz"]
    num_joints = clean_np.shape[1]
    joint_rows = []
    for joint_idx in range(num_joints):
        joint_name = dof_names[joint_idx] if joint_idx < len(dof_names) else f"dof_{joint_idx}"
        psd_stack = np.stack(
            [np.asarray(curve_psd_stats[curve_idx]["joint_psd"][joint_idx]["psd"], dtype=np.float32) for curve_idx in range(len(curve_psd_stats))],
            axis=0,
        )
        joint_rows.append(
            {
                "joint_idx": int(joint_idx),
                "joint_name": joint_name,
                "psd": np.mean(psd_stack, axis=0).astype(np.float32).tolist(),
            }
        )
    return {
        "fs_hz": float(fs_hz),
        "num_curves": int(len(synthetic_curves)),
        "num_frames": int(clean_np.shape[0]),
        "num_joints": int(num_joints),
        "freqs_hz": freqs_hz,
        "joint_psd": joint_rows,
    }


def compute_residual_scale_vector_from_stats(
    real_stats: Dict[str, object],
    synthetic_stats: Dict[str, object],
    dof_names: List[str],
    clip_min: float = 0.25,
    clip_max: float = 4.0,
    eps: float = 1e-6,
) -> np.ndarray:
    num_joints = len(dof_names)
    real_rows = real_stats.get("joint_stats", [])
    synthetic_rows = synthetic_stats.get("joint_stats", [])
    if len(real_rows) < num_joints or len(synthetic_rows) < num_joints:
        raise ValueError("real_stats and synthetic_stats must both contain joint_stats for every DOF.")

    def _rms_from_row(row: Dict[str, object]) -> float:
        mean = float(row.get("mean", 0.0))
        std = float(row.get("std", 0.0))
        return float(np.sqrt((mean ** 2) + (std ** 2)))

    real_rms = np.asarray([_rms_from_row(real_rows[idx]) for idx in range(num_joints)], dtype=np.float32)
    synthetic_rms = np.asarray([_rms_from_row(synthetic_rows[idx]) for idx in range(num_joints)], dtype=np.float32)

    scale = np.ones((num_joints,), dtype=np.float32)
    visited: set[int] = set()
    index_map = build_left_right_index_map(dof_names)
    for idx in range(num_joints):
        if idx in visited:
            continue
        partner_idx = index_map.get(idx)
        if partner_idx is not None:
            pooled_real = float(np.mean([real_rms[idx], real_rms[partner_idx]]))
            pooled_synth = float(np.mean([synthetic_rms[idx], synthetic_rms[partner_idx]]))
            factor = 1.0 if pooled_synth <= eps else float(np.clip(pooled_real / pooled_synth, clip_min, clip_max))
            scale[idx] = factor
            scale[partner_idx] = factor
            visited.add(idx)
            visited.add(partner_idx)
        else:
            factor = 1.0 if synthetic_rms[idx] <= eps else float(np.clip(real_rms[idx] / synthetic_rms[idx], clip_min, clip_max))
            scale[idx] = factor
            visited.add(idx)
    return scale.astype(np.float32)
