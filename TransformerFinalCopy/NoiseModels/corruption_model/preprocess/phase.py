from __future__ import annotations

import numpy as np


PHASE_POINTS_PER_SEGMENT = 100
PHASE_CYCLE_LENGTH = 2 * PHASE_POINTS_PER_SEGMENT


def detect_contact_from_grf(grf: np.ndarray, threshold_n: float = 5.0) -> np.ndarray:
    grf = np.asarray(grf, dtype=np.float32)
    if grf.ndim != 2 or grf.shape[1] < 6:
        raise ValueError(f"Expected GRF with shape [T, >=6], got {grf.shape}")
    right_vertical = grf[:, 2]
    left_vertical = grf[:, 5]
    return np.stack([right_vertical > threshold_n, left_vertical > threshold_n], axis=1)


def resample_contact_mask(contact_mask: np.ndarray, target_length: int) -> np.ndarray:
    contact_mask = np.asarray(contact_mask).astype(np.float32)
    if contact_mask.ndim != 2:
        raise ValueError(f"Expected contact mask with shape [T, C], got {contact_mask.shape}")
    source_length = contact_mask.shape[0]
    if source_length == target_length:
        return contact_mask.astype(bool)
    if source_length <= 1:
        return np.repeat(contact_mask.astype(bool), target_length, axis=0)
    source_x = np.linspace(0.0, 1.0, source_length, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    out = np.empty((target_length, contact_mask.shape[1]), dtype=bool)
    for col_idx in range(contact_mask.shape[1]):
        interp = np.interp(target_x, source_x, contact_mask[:, col_idx], left=contact_mask[0, col_idx], right=contact_mask[-1, col_idx])
        out[:, col_idx] = interp >= 0.5
    return out


def compute_gait_phase(contact_mask: np.ndarray) -> np.ndarray:
    contact_mask = np.asarray(contact_mask).astype(bool)
    t = contact_mask.shape[0]
    phase = np.zeros((t,), dtype=np.float32)
    cycle_starts = np.where(np.diff(contact_mask[:, 0].astype(np.int32), prepend=0) == 1)[0]
    if cycle_starts.size < 2:
        return np.linspace(0.0, 1.0, t, dtype=np.float32, endpoint=False)
    for start_idx, end_idx in zip(cycle_starts[:-1], cycle_starts[1:]):
        length = max(end_idx - start_idx, 1)
        phase[start_idx:end_idx] = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    phase[cycle_starts[-1] :] = np.linspace(0.0, 1.0, t - cycle_starts[-1], endpoint=False, dtype=np.float32)
    return phase


def bin_phase(phase: np.ndarray, phase_bins: int) -> np.ndarray:
    clipped = np.clip(np.asarray(phase, dtype=np.float32), 0.0, 0.999999)
    return np.minimum((clipped * phase_bins).astype(np.int32), phase_bins - 1)


def compute_stance_swing_phase_positions(contact_mask: np.ndarray) -> np.ndarray:
    contact_mask = np.asarray(contact_mask).astype(bool)
    if contact_mask.ndim != 2 or contact_mask.shape[0] == 0:
        raise ValueError(f"Expected contact mask with shape [T, C], got {contact_mask.shape}")
    primary_contact = contact_mask[:, 0].astype(bool)
    positions = np.zeros((primary_contact.shape[0],), dtype=np.int32)
    run_starts = np.concatenate([[0], np.where(np.diff(primary_contact.astype(np.int32)) != 0)[0] + 1, [primary_contact.shape[0]]])
    for start_idx, end_idx in zip(run_starts[:-1], run_starts[1:]):
        run_length = max(end_idx - start_idx, 1)
        if primary_contact[start_idx]:
            positions[start_idx:end_idx] = np.floor(
                np.linspace(0.0, PHASE_POINTS_PER_SEGMENT, run_length, endpoint=False, dtype=np.float32)
            ).astype(np.int32)
        else:
            positions[start_idx:end_idx] = (
                PHASE_POINTS_PER_SEGMENT
                + np.floor(
                    np.linspace(0.0, PHASE_POINTS_PER_SEGMENT, run_length, endpoint=False, dtype=np.float32)
                ).astype(np.int32)
            )
    return np.clip(positions, 0, PHASE_CYCLE_LENGTH - 1).astype(np.int32)


def compute_stance_swing_phase_positions_from_grf(grf: np.ndarray, target_length: int | None = None) -> np.ndarray:
    contact_mask = detect_contact_from_grf(grf)
    if target_length is not None:
        contact_mask = resample_contact_mask(contact_mask, target_length=target_length)
    return compute_stance_swing_phase_positions(contact_mask)


def circular_phase_distance(positions: np.ndarray, center: int, cycle_length: int = PHASE_CYCLE_LENGTH) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int32)
    delta = np.abs(pos - int(center))
    return np.minimum(delta, cycle_length - delta).astype(np.int32)


def nearest_phase_window(values: np.ndarray, positions: np.ndarray, center: int, window_frames: int) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32)
    positions_np = np.asarray(positions, dtype=np.int32)
    if values_np.shape[0] != positions_np.shape[0]:
        raise ValueError("values and positions must have the same number of frames")
    if values_np.shape[0] == 0:
        return np.zeros((0,) + values_np.shape[1:], dtype=np.float32)
    k = max(1, min(int(window_frames), values_np.shape[0]))
    distances = circular_phase_distance(positions_np, center)
    nearest_idx = np.argpartition(distances, kth=k - 1)[:k]
    nearest_idx = nearest_idx[np.argsort(distances[nearest_idx], kind="stable")]
    return values_np[nearest_idx].astype(np.float32)
