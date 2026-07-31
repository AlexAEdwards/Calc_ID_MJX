from __future__ import annotations

import numpy as np

from corruption_model.preprocess.phase import (
    compute_gait_phase,
    compute_stance_swing_phase_positions,
    detect_contact_from_grf,
    resample_contact_mask,
)
from corruption_model.types import ResidualTrial, TrialPair


def _differentiate(q: np.ndarray, time: np.ndarray) -> np.ndarray:
    if q.shape[0] < 2:
        return np.zeros_like(q, dtype=np.float32)
    return np.gradient(q.astype(np.float64), time.astype(np.float64), axis=0).astype(np.float32)


def compute_residual_trial(
    trial: TrialPair,
    q_mocap_aligned: np.ndarray,
    q_opencap_aligned: np.ndarray,
    lag_frames: int,
    lag_seconds: float,
    alignment_score: float,
) -> ResidualTrial:
    residual = (q_opencap_aligned - q_mocap_aligned).astype(np.float32)
    target_length = residual.shape[0]
    contact_mask = trial.contact_mask
    if contact_mask is None and trial.grf is not None:
        contact_mask = detect_contact_from_grf(trial.grf)
    if contact_mask is not None:
        contact_mask = resample_contact_mask(contact_mask, target_length)
    phase = compute_gait_phase(contact_mask) if contact_mask is not None else np.linspace(0.0, 1.0, target_length, endpoint=False, dtype=np.float32)
    if contact_mask is not None:
        phase_positions = compute_stance_swing_phase_positions(contact_mask)
    else:
        phase_positions = np.floor(np.linspace(0.0, 200.0, target_length, endpoint=False, dtype=np.float32)).astype(np.int32)
    qvel = _differentiate(q_mocap_aligned, trial.time)
    speed = np.linalg.norm(qvel, axis=1).astype(np.float32)
    return ResidualTrial(
        subject_metadata=trial.subject_metadata,
        trial_id=trial.trial_id,
        activity=trial.activity,
        time=trial.time,
        q_clean=q_mocap_aligned.astype(np.float32),
        q_target=q_opencap_aligned.astype(np.float32),
        residual=residual,
        phase=phase.astype(np.float32),
        phase_positions=phase_positions.astype(np.int32),
        phase_bins=phase_positions.astype(np.int32),
        speed=speed,
        lag_frames=int(lag_frames),
        lag_seconds=float(lag_seconds),
        alignment_score=float(alignment_score),
        mask_valid=trial.mask_valid,
        meta=dict(trial.meta),
    )
