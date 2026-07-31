from __future__ import annotations

from dataclasses import replace

import numpy as np

from corruption_model.types import MocapTrial, TrialPair


def validate_time_series(time: np.ndarray, frame_count: int, sample_rate_hz: float, tolerance_hz: float = 1.0) -> None:
    if time.ndim != 1:
        raise ValueError(f"time must be 1D, got shape {time.shape}")
    if time.shape[0] != frame_count:
        raise ValueError(f"time length {time.shape[0]} does not match frame count {frame_count}")
    if frame_count > 1:
        diffs = np.diff(time)
        mean_dt = float(np.mean(diffs))
        observed_fs = 1.0 / mean_dt
        if abs(observed_fs - sample_rate_hz) > tolerance_hz:
            raise ValueError(f"sample rate {observed_fs:.3f} Hz differs from expected {sample_rate_hz:.3f} Hz")


def _resample_matrix(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32)
    source_time_np = np.asarray(source_time, dtype=np.float32)
    target_time_np = np.asarray(target_time, dtype=np.float32)
    if values_np.ndim == 1:
        return np.interp(target_time_np, source_time_np, values_np).astype(np.float32)
    out = np.empty((target_time_np.shape[0], values_np.shape[1]), dtype=np.float32)
    for col_idx in range(values_np.shape[1]):
        out[:, col_idx] = np.interp(
            target_time_np,
            source_time_np,
            values_np[:, col_idx],
            left=values_np[0, col_idx],
            right=values_np[-1, col_idx],
        ).astype(np.float32)
    return out


def _build_uniform_time_vector(source_time: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    source_time_np = np.asarray(source_time, dtype=np.float32)
    if source_time_np.shape[0] <= 1:
        return source_time_np.copy()
    start_time = float(source_time_np[0])
    end_time = float(source_time_np[-1])
    duration = max(end_time - start_time, 0.0)
    frame_count = max(int(round(duration * float(sample_rate_hz))) + 1, 2)
    return np.linspace(start_time, end_time, frame_count, dtype=np.float32)


def harmonize_trial_pair(trial: TrialPair, sample_rate_hz: float) -> TrialPair:
    if trial.q_mocap.shape != trial.q_opencap.shape:
        raise ValueError(f"{trial.subject_id}/{trial.trial_id}: q_mocap and q_opencap shapes do not match")
    if trial.q_mocap.shape[1] != trial.subject_metadata.num_dofs:
        raise ValueError(f"{trial.subject_id}/{trial.trial_id}: DOF mismatch against Patient_MD metadata")
    validate_time_series(trial.time, trial.q_mocap.shape[0], sample_rate_hz=sample_rate_hz)
    if not np.all(np.isfinite(trial.q_mocap)) or not np.all(np.isfinite(trial.q_opencap)):
        raise ValueError(f"{trial.subject_id}/{trial.trial_id}: found non-finite kinematics")
    return trial


def harmonize_mocap_trial(trial: MocapTrial, sample_rate_hz: float) -> MocapTrial:
    if trial.pos.shape[1] != trial.subject_metadata.num_dofs:
        raise ValueError(f"{trial.subject_id}/{trial.trial_id}: Pos.npy width mismatch against Patient_MD")
    if trial.time_for_pos.shape[0] != trial.pos.shape[0]:
        raise ValueError(f"{trial.subject_id}/{trial.trial_id}: position time length mismatch")
    if trial.time_for_pos.shape[0] > 1:
        diffs = np.diff(trial.time_for_pos)
        observed_fs = float(1.0 / np.mean(diffs))
        if abs(observed_fs - sample_rate_hz) > 1.0:
            target_time_for_pos = _build_uniform_time_vector(trial.time_for_pos, sample_rate_hz=sample_rate_hz)
            return replace(
                trial,
                time_for_pos=target_time_for_pos,
                pos=_resample_matrix(trial.pos, trial.time_for_pos, target_time_for_pos),
                vel=_resample_matrix(trial.vel, trial.time_for_pos, target_time_for_pos),
                accel=_resample_matrix(trial.accel, trial.time_for_pos, target_time_for_pos),
                meta={
                    **trial.meta,
                    "resampled_position_signals_to_hz": float(sample_rate_hz),
                    "original_position_sample_rate_hz": observed_fs,
                },
            )
    validate_time_series(trial.time_for_pos, trial.pos.shape[0], sample_rate_hz=sample_rate_hz)
    return trial
