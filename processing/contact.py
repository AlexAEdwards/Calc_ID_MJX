"""Foot-ground contact detection and stance-phase segmentation.

Extracted verbatim from ProcessData.py in REFACTOR_PLAN.md Stage 6. An AST pass
confirmed the cluster is dependency-closed: these functions call nothing else
defined in ProcessData.py, so the move cannot drag hidden state along.

Three of them detect stance from vertical GRF and differ only in convention, which
is worth knowing before picking one:

* ``create_contact_boolean`` - MuJoCo Z-up GRF (vertical in columns 2 and 5),
  returns a (T, 2) float array [right, left], drops stances under
  ``min_stance_frames``.
* ``get_stance_phases`` - caller passes the vertical column index explicitly, and
  the default threshold is 15 N rather than 1 N.
* ``_detect_stance_phases`` - MuJoCo Z-up as well, but returns per-side dicts and
  by default *excludes* stances touching either end of the trial.

``zero_short_grf_cop_stances`` is the odd one out: it runs before MuJoCo
conversion, on the OpenSim force-plate layout where vertical GRF is columns 1 and
4. Passing MuJoCo-ordered data to it silently reads the wrong axis.

ProcessData.py re-exports every name, so its callers are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def create_contact_boolean(grf_data: np.ndarray,
                            threshold: float = 1.0,
                            min_stance_frames: int = 4) -> np.ndarray:
    """
    Returns (T, 2) boolean array [right, left].
    GRF format: [Rx, Ry, Rz, Lx, Ly, Lz] – Z-up (MuJoCo).
    """
    def _remove_short_stances(contact_1d: np.ndarray, min_len: int) -> np.ndarray:
        """Zero out contiguous stance segments shorter than min_len frames."""
        if min_len <= 1:
            return contact_1d
        c = contact_1d.astype(bool)
        padded = np.concatenate(([False], c, [False]))
        d = np.diff(padded.astype(np.int32))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        cleaned = c.copy()
        for s, e in zip(starts, ends):
            if (e - s) < min_len:
                cleaned[s:e] = False
        return cleaned.astype(np.float32)

    right = (grf_data[:, 2] > threshold).astype(np.float32)
    left  = (grf_data[:, 5] > threshold).astype(np.float32)
    right = _remove_short_stances(right, int(min_stance_frames))
    left = _remove_short_stances(left, int(min_stance_frames))
    return np.stack([right, left], axis=1)


def zero_short_grf_cop_stances(
    grf: np.ndarray,
    grm: np.ndarray | None,
    cop: np.ndarray,
    contact_threshold_n: float = 1.0,
    max_frames: int = 25,
    min_peak_n: float = 50.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    """
    Zero short/low-peak non-edge stance phases for each foot.

    Inputs are in the OpenSim-style force-plate layout used before MuJoCo
    conversion: right vertical GRF is column 1, left vertical GRF is column 4.
    When a stance is flagged, all three GRF columns, all three GRM columns
    when present, and all three COP columns for that foot are zeroed over the
    flagged frames.
    """
    grf_out = np.asarray(grf).copy()
    grm_out = None if grm is None else np.asarray(grm).copy()
    cop_out = np.asarray(cop).copy()
    report: dict[str, Any] = {
        "n_flagged": 0,
        "n_frames_zeroed": 0,
        "stances": [],
    }

    if grf_out.ndim != 2 or grf_out.shape[1] < 6:
        return grf_out, grm_out, cop_out, report
    if cop_out.ndim != 2 or cop_out.shape[0] != grf_out.shape[0] or cop_out.shape[1] < 6:
        return grf_out, grm_out, cop_out, report
    if grm_out is not None and (
        grm_out.ndim != 2 or grm_out.shape[0] != grf_out.shape[0] or grm_out.shape[1] < 6
    ):
        grm_out = grm

    n_frames = int(grf_out.shape[0])
    for foot, v_col, cols in (
        ("R", 1, slice(0, 3)),
        ("L", 4, slice(3, 6)),
    ):
        vgrf = grf_out[:, v_col]
        contact = np.asarray(vgrf, dtype=np.float64) > float(contact_threshold_n)
        padded = np.concatenate(([False], contact, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            start = int(start)
            end = int(end)
            if start == 0 or end == n_frames:
                continue
            duration = end - start
            peak_n = float(np.nanmax(vgrf[start:end])) if end > start else 0.0
            reasons: list[str] = []
            if duration < int(max_frames):
                reasons.append("short_duration")
            if peak_n <= float(min_peak_n):
                reasons.append("low_peak")
            if not reasons:
                continue

            grf_out[start:end, cols] = 0.0
            if grm_out is not None and grm_out is not grm:
                grm_out[start:end, cols] = 0.0
            cop_out[start:end, cols] = 0.0
            report["n_flagged"] += 1
            report["n_frames_zeroed"] += duration
            report["stances"].append({
                "foot": foot,
                "start": start,
                "end": end,
                "duration": duration,
                "peak_n": peak_n,
                "reasons": reasons,
            })

    return grf_out, grm_out, cop_out, report


def get_stance_phases(grf_data: np.ndarray, leg_idx: int,
                       threshold: float = 15.0) -> list:
    """Returns list of dicts {start, end, duration_frames} for each stance."""
    vgrf = grf_data[:, leg_idx]
    mask = (vgrf > threshold).astype(int)
    padded = np.concatenate(([0], mask, [0]))
    diffs  = np.diff(padded)
    starts = np.where(diffs ==  1)[0]
    ends   = np.where(diffs == -1)[0]
    phases = []
    for s, e in zip(starts, ends):
        phases.append({"start": s, "end": e, "duration_frames": e - s})
    return phases


def _stance_segments(contact_1d: np.ndarray) -> list[tuple[int, int]]:
    c = np.asarray(contact_1d).astype(bool)
    padded = np.concatenate(([False], c, [False]))
    d = np.diff(padded.astype(np.int32))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


def _detect_stance_phases(grf_data: np.ndarray,
                           threshold: float = 1.0,
                           min_duration: int = 10,
                           include_boundary: bool = False) -> dict:
    """
    Detect stance phases for right and left feet.
    Returns dict with keys 'Right' and 'Left', each a list of dicts
    {start, end, duration_frames, partial_begin, partial_end}.
    """
    result = {}
    configs = [("Right", 2), ("Left", 5)]
    T = grf_data.shape[0]
    for side, idx in configs:
        vgrf = grf_data[:, idx]
        is_stance = vgrf > threshold
        padded = np.concatenate(([False], is_stance, [False]))
        diffs  = np.diff(padded.astype(int))
        starts = np.where(diffs ==  1)[0]
        ends   = np.where(diffs == -1)[0]
        phases = []
        for s, e in zip(starts, ends):
            partial_begin = (s == 0)
            partial_end = (e == T)
            if (partial_begin or partial_end) and not include_boundary:
                continue
            if e - s >= min_duration:
                phases.append({
                    "start": int(s),
                    "end": int(e),
                    "duration_frames": int(e - s),
                    "partial_begin": bool(partial_begin),
                    "partial_end": bool(partial_end),
                })
        result[side] = phases
    return result
