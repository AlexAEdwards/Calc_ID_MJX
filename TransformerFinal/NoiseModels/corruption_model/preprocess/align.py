from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from corruption_model.types import TrialPair


ROBUST_CHANNEL_NAMES = ("pelvis_ty", "hip_flexion_r", "knee_angle_r", "lumbar_extension")


@dataclass
class AlignmentResult:
    q_mocap_aligned: np.ndarray
    q_opencap_aligned: np.ndarray
    lag_frames: int
    lag_seconds: float
    alignment_score: float


def _channel_indices(dof_names: Sequence[str]) -> list[int]:
    indices = []
    for name in ROBUST_CHANNEL_NAMES:
        if name in dof_names:
            indices.append(dof_names.index(name))
    return indices if indices else list(range(min(3, len(dof_names))))


def _shift_signal(signal: np.ndarray, lag_frames: int) -> np.ndarray:
    if lag_frames == 0:
        return signal.copy()
    x = np.arange(signal.shape[0], dtype=np.float32)
    shifted = np.empty_like(signal, dtype=np.float32)
    for col_idx in range(signal.shape[1]):
        shifted[:, col_idx] = np.interp(x, x - lag_frames, signal[:, col_idx], left=signal[0, col_idx], right=signal[-1, col_idx])
    return shifted


def estimate_global_lag(trial: TrialPair, sample_rate_hz: float, max_lag_frames: int) -> AlignmentResult:
    indices = _channel_indices(trial.subject_metadata.dof_names)
    ref = trial.q_mocap[:, indices]
    src = trial.q_opencap[:, indices]
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        shifted = _shift_signal(src, lag)
        score = float(np.mean([
            np.corrcoef(ref[:, i], shifted[:, i])[0, 1] if np.std(ref[:, i]) > 1e-8 and np.std(shifted[:, i]) > 1e-8 else 0.0
            for i in range(ref.shape[1])
        ]))
        if np.isnan(score):
            score = -np.inf
        if score > best_score:
            best_score = score
            best_lag = lag
    q_opencap_aligned = _shift_signal(trial.q_opencap, best_lag)
    return AlignmentResult(
        q_mocap_aligned=trial.q_mocap.copy(),
        q_opencap_aligned=q_opencap_aligned,
        lag_frames=int(best_lag),
        lag_seconds=float(best_lag / sample_rate_hz),
        alignment_score=float(best_score if np.isfinite(best_score) else 0.0),
    )


def shift_with_interpolation(signal: np.ndarray, lag_frames: float) -> np.ndarray:
    x = np.arange(signal.shape[0], dtype=np.float32)
    shifted = np.empty_like(signal, dtype=np.float32)
    for col_idx in range(signal.shape[1]):
        shifted[:, col_idx] = np.interp(
            x,
            x - lag_frames,
            signal[:, col_idx],
            left=signal[0, col_idx],
            right=signal[-1, col_idx],
        )
    return shifted
