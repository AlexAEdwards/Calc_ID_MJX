"""Kinematic and force signal filtering.

Extracted verbatim from ProcessData.py in REFACTOR_PLAN.md Stage 6. These four
functions form a closed cluster: an AST pass confirmed they call nothing else
defined in ProcessData.py, so the move cannot drag hidden state along.

ProcessData.py re-exports every name, so its ~8k lines of callers are unchanged.

Note butter_lowpass_filter's ``cutoff`` default is never used - every call site
passes an explicit value, usually cfg["FILTER_CUTOFF_HZ"].
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data: np.ndarray, cutoff: float = 6.0,
                           fs: float = 100.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    if data.ndim == 1:
        return filtfilt(b, a, data)
    out = np.empty_like(data)
    for col in range(data.shape[1]):
        out[:, col] = filtfilt(b, a, data[:, col])
    return out


def gcv_derivatives(qpos: np.ndarray, dt: float):
    """OpenSim-style velocity/acceleration via a GCV smoothing spline of the positions.

    OpenSim's InverseDynamicsTool fits a generalized-cross-validation (GCV) spline to the
    coordinates and differentiates it to obtain qdot/qddot. We replicate that "OpenSim
    filtering" technique here so MJX inverse dynamics can be run on OpenSim-derived
    derivatives (scipy's make_smoothing_spline is the Reinsch/Woltring GCV spline; lam=None
    selects the smoothing by GCV, analogous to OpenSim's GCVSpline).

    Each qpos column is splined independently, so this requires qpos to have one column per
    generalized speed (no quaternion free joint -- true here, where the pelvis is six 1-DOF
    joints, so nq == nv). Returns (qvel, qacc) with the same shape as qpos.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    T, ncol = qpos.shape
    # Fit on unit (sample-index) spacing, then scale derivatives by 1/dt and 1/dt**2.
    # The GCV lambda selection is sensitive to the x-scale: on a 0.01 s axis it
    # over-smooths to a near-flat fit, whereas on the index axis GCV recovers the
    # signal exactly (fit corr ~1.0; resulting qacc matches MJX qacc to ~0.998).
    idx = np.arange(T, dtype=np.float64)
    dt = float(dt)
    qvel = np.empty_like(qpos)
    qacc = np.empty_like(qpos)
    for c in range(ncol):
        spl = make_smoothing_spline(idx, qpos[:, c], lam=None)
        qvel[:, c] = spl.derivative(1)(idx) / dt
        qacc[:, c] = spl.derivative(2)(idx) / (dt * dt)
    return qvel, qacc


def apply_kinematics_filtering(pos: np.ndarray, vel: np.ndarray, accel: np.ndarray,
                               cfg: dict, fs: float):
    """Per-channel 6 Hz Butterworth on Pos/Vel/Accel.

    Per-channel toggles (FILTER_POS/VEL/ACCEL) and cutoffs (FILTER_CUTOFF_*_HZ)
    default to None, in which case they follow the global ENABLE_KINEMATICS_FILTERING
    and FILTER_CUTOFF_HZ. With all overrides None this is byte-identical to the
    previous whole-block filter. Used by the filter-ablation study to isolate which
    kinematic channel (and cutoff) drives the MJX peak-ankle-power attenuation.
    """
    global_on = bool(cfg.get("ENABLE_KINEMATICS_FILTERING", True))
    order = int(cfg.get("FILTER_ORDER", 2))
    global_cutoff = float(cfg.get("FILTER_CUTOFF_HZ", 6.0))

    def _channel(name: str, data: np.ndarray) -> np.ndarray:
        toggle = cfg.get(f"FILTER_{name}", None)
        on = global_on if toggle is None else bool(toggle)
        if not on:
            return data
        co = cfg.get(f"FILTER_CUTOFF_{name}_HZ", None)
        cutoff = global_cutoff if co is None else float(co)
        return butter_lowpass_filter(data, cutoff, fs, order=order)

    pos = _channel("POS", pos)
    vel = _channel("VEL", vel)
    accel = _channel("ACCEL", accel)
    chans = [n for n in ("POS", "VEL", "ACCEL")
             if (global_on if cfg.get(f"FILTER_{n}") is None else bool(cfg.get(f"FILTER_{n}")))]
    print(f"    [Kinematics Filter] filtered channels={chans or 'none'} "
          f"(global={global_on}, cutoff={global_cutoff}Hz, order={order})")
    return pos, vel, accel


def filter_segment_wise(data: np.ndarray, vertical_force: np.ndarray,
                        cutoff: float = 6.0, fs: float = 100.0,
                        order: int = 2, pad_width: int = 15,
                        force_threshold: float = 1.0,
                        edge_hold: bool = False) -> np.ndarray:
    """
    Filter data only during stance segments (where vertical_force > threshold).
    Pads around each stance segment before filtfilt. By default padding is
    zeros; with edge_hold=True padding repeats the first/last stance value.
    """
    result = data.copy()
    is_stance = vertical_force > force_threshold
    is_padded = np.concatenate(([False], is_stance, [False]))
    diffs = np.diff(is_padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]

    nyq  = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")

    for s, e in zip(starts, ends):
        seg = data[s:e]
        pad_s = min(pad_width, s)
        pad_e = min(pad_width, data.shape[0] - e)

        if edge_hold and seg.shape[0] > 0:
            pre = np.repeat(seg[:1], pad_s, axis=0)
            post = np.repeat(seg[-1:], pad_e, axis=0)
        else:
            pre = np.zeros((pad_s,) + seg.shape[1:], dtype=seg.dtype)
            post = np.zeros((pad_e,) + seg.shape[1:], dtype=seg.dtype)
        padded = np.concatenate([pre, seg, post], axis=0)

        if padded.ndim == 1:
            filt = filtfilt(b, a, padded)[pad_s: pad_s + (e - s)]
        else:
            filt = np.empty_like(padded)
            for col in range(padded.shape[1]):
                filt[:, col] = filtfilt(b, a, padded[:, col])
            filt = filt[pad_s: pad_s + (e - s)]

        result[s:e] = filt
    return result
