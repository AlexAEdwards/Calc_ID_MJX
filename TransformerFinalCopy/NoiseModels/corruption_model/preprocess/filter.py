from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch


def butter_lowpass_filter(data: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs_hz
    normal_cutoff = min(max(cutoff_hz / nyquist, 1e-5), 0.999)
    b, a = butter(order, normal_cutoff, btype="low")
    padlen = 3 * max(len(a), len(b))

    def _apply_filtfilt(signal: np.ndarray) -> np.ndarray:
        try:
            return filtfilt(b, a, signal).astype(np.float32)
        except ValueError as exc:
            if "padlen" not in str(exc):
                raise
            try:
                return filtfilt(b, a, signal, method="gust").astype(np.float32)
            except Exception:
                return np.asarray(signal, dtype=np.float32)

    if data.ndim == 1:
        if data.shape[0] <= padlen:
            return _apply_filtfilt(np.asarray(data, dtype=np.float64))
        return filtfilt(b, a, data).astype(np.float32)
    out = np.empty_like(data, dtype=np.float32)
    for col_idx in range(data.shape[1]):
        column = np.asarray(data[:, col_idx], dtype=np.float64)
        if column.shape[0] <= padlen:
            out[:, col_idx] = _apply_filtfilt(column)
        else:
            out[:, col_idx] = filtfilt(b, a, column).astype(np.float32)
    return out


def differentiate_signal(data: np.ndarray, time: np.ndarray) -> np.ndarray:
    data_np = np.asarray(data, dtype=np.float64)
    time_np = np.asarray(time, dtype=np.float64)
    if data_np.ndim == 1:
        return np.gradient(data_np, time_np).astype(np.float32)
    return np.gradient(data_np, time_np, axis=0).astype(np.float32)


def estimate_joint_cutoff_hz(reference: np.ndarray, target: np.ndarray, fs_hz: float, default_cutoff_hz: float) -> float:
    if reference.shape[0] < 8 or target.shape[0] < 8:
        return float(default_cutoff_hz)
    cutoff_candidates = []
    for col_idx in range(reference.shape[1]):
        freqs_ref, psd_ref = welch(reference[:, col_idx], fs=fs_hz, nperseg=min(256, reference.shape[0]))
        freqs_tgt, psd_tgt = welch(target[:, col_idx], fs=fs_hz, nperseg=min(256, target.shape[0]))
        if not np.allclose(freqs_ref, freqs_tgt):
            continue
        ratio = np.divide(psd_tgt, np.maximum(psd_ref, 1e-8))
        high_idx = np.where(ratio < 0.5)[0]
        if high_idx.size:
            cutoff_candidates.append(float(freqs_ref[high_idx[0]]))
    if not cutoff_candidates:
        return float(default_cutoff_hz)
    return float(np.clip(np.median(cutoff_candidates), 0.5, fs_hz * 0.45))


def estimate_effective_cutoff_hz(signal: np.ndarray, fs_hz: float, power_fraction: float = 0.95, default_cutoff_hz: float = 6.0) -> float:
    signal_np = np.asarray(signal, dtype=np.float32)
    if signal_np.ndim != 2 or signal_np.shape[0] < 8:
        return float(default_cutoff_hz)
    cutoff_candidates = []
    for col_idx in range(signal_np.shape[1]):
        freqs, psd = welch(signal_np[:, col_idx], fs=fs_hz, nperseg=min(256, signal_np.shape[0]))
        total_power = float(np.sum(psd))
        if total_power <= 1e-8:
            continue
        cumulative = np.cumsum(psd) / total_power
        cutoff_idx = int(np.searchsorted(cumulative, power_fraction))
        cutoff_idx = min(max(cutoff_idx, 0), len(freqs) - 1)
        cutoff_candidates.append(float(freqs[cutoff_idx]))
    if not cutoff_candidates:
        return float(default_cutoff_hz)
    return float(np.clip(np.median(cutoff_candidates), 0.5, fs_hz * 0.45))
