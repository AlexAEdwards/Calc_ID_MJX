"""Centre-of-pressure cleaning and normalisation.

Extracted verbatim from ProcessData.py in REFACTOR_PLAN.md Stage 6. The cluster is
dependency-closed apart from ``filter_segment_wise``, which now comes from
``processing.filtering``.

COP is only meaningful while a foot is loaded. Outside stance the force plate
reports noise around the origin, so both functions here are written to touch only
the loaded frames:

* ``clean_and_filter_cop`` drops out-of-bounds samples, trims the unreliable
  frames at each end of a stance, linearly extrapolates back over them from the
  interior slope, then filters each stance segment independently. It reads
  vertical GRF at the MuJoCo Z-up indices 2 (right) and 5 (left).
* ``_multiply_cop_by_bodyweight_normalized_grf_magnitude`` rescales an already
  height-normalised, ground-aligned COP by |GRF|/bodyweight per foot, which zeroes
  swing frames as a side effect of the GRF being zero there.

ProcessData.py re-exports both names, so its callers are unchanged.
"""

from __future__ import annotations

import numpy as np

from processing.filtering import filter_segment_wise


def clean_and_filter_cop(cop_data: np.ndarray,
                          grf_data: np.ndarray,
                          trim_start_frames: int = 3,
                          trim_end_frames:   int = 3,
                          extrapolation_frames: int = 6,
                          pad_width:         int = 15,
                          edge_hold:         bool = False,
                          cutoff:            float = 6.0,
                          fs:                float = 100.0,
                          order:             int   = 2,
                          outlier_threshold: float = 5.0) -> np.ndarray:
    """
    Outlier removal + edge extrapolation + segment-wise Butterworth on COP.
    GRF uses MuJoCo Z-up vertical indices: 2 (right) and 5 (left).

    cop_data : (T, 4) – [Rx, Ry, Lx, Ly]
    grf_data : (T, 6) – [Rx, Ry, Rz, Lx, Ly, Lz]
    Returns cleaned (T, 4).
    """
    cop_np  = cop_data.copy()
    foot_configs = [
        (0, 1, 2),   # col_x, col_y, grf_idx for right foot
        (2, 3, 5),   # col_x, col_y, grf_idx for left  foot
    ]

    for foot_idx, (col_x, col_y, grf_idx) in enumerate(foot_configs):
        # ── Outlier removal ──────────────────────────────────────
        outlier_mask = (
            (np.abs(cop_np[:, col_x]) > outlier_threshold) |
            (np.abs(cop_np[:, col_y]) > outlier_threshold)
        )
        cop_np[outlier_mask, col_x] = 0.0
        cop_np[outlier_mask, col_y] = 0.0

        # ── Find non-zero segments ───────────────────────────────
        is_nonzero = (np.abs(cop_np[:, col_x]) > 1e-9) | (np.abs(cop_np[:, col_y]) > 1e-9)
        is_nz_pad  = np.concatenate(([False], is_nonzero, [False]))
        diff       = np.diff(is_nz_pad.astype(int))
        starts     = np.where(diff ==  1)[0]
        ends       = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            seg_len = end - start
            if seg_len <= (trim_start_frames + trim_end_frames):
                cop_np[start:end, col_x] = 0.0
                cop_np[start:end, col_y] = 0.0
                continue

            # ── Edge extrapolation – START ───────────────────────
            ti = start + trim_start_frames
            slope_end = min(ti + int(extrapolation_frames), end)
            if slope_end > ti + 1:
                sx = np.mean(np.diff(cop_np[ti:slope_end, col_x]))
                sy = np.mean(np.diff(cop_np[ti:slope_end, col_y]))
                for k in range(trim_start_frames):
                    d = trim_start_frames - k
                    cop_np[start + k, col_x] = cop_np[ti, col_x] - d * sx / 6
                    cop_np[start + k, col_y] = cop_np[ti, col_y] - d * sy / 6
            else:
                cop_np[start:ti, col_x] = cop_np[ti, col_x]
                cop_np[start:ti, col_y] = cop_np[ti, col_y]

            # ── Edge extrapolation – END ─────────────────────────
            te = end - trim_end_frames
            slope_st = max(start, te - int(extrapolation_frames))
            if te > slope_st + 1:
                sx = np.mean(np.diff(cop_np[slope_st:te, col_x]))
                sy = np.mean(np.diff(cop_np[slope_st:te, col_y]))
                for k in range(trim_end_frames):
                    d = k + 1
                    cop_np[te + k, col_x] = cop_np[te - 1, col_x] + d * sx / 6
                    cop_np[te + k, col_y] = cop_np[te - 1, col_y] + d * sy / 6
            else:
                cop_np[te:end, col_x] = cop_np[te - 1, col_x]
                cop_np[te:end, col_y] = cop_np[te - 1, col_y]

        # ── Segment-wise Butterworth ─────────────────────────────
        cop_np[:, col_x] = filter_segment_wise(
            cop_np[:, col_x], grf_data[:, grf_idx],
            cutoff=cutoff, fs=fs, order=order, pad_width=pad_width,
            force_threshold=1.0, edge_hold=edge_hold,
        )
        cop_np[:, col_y] = filter_segment_wise(
            cop_np[:, col_y], grf_data[:, grf_idx],
            cutoff=cutoff, fs=fs, order=order, pad_width=pad_width,
            force_threshold=1.0, edge_hold=edge_hold,
        )

    # ── Re-apply GRF mask for consistency ───────────────────────
    cop_np[grf_data[:, 2] < 1.0, 0:2] = 0.0
    cop_np[grf_data[:, 5] < 1.0, 2:4] = 0.0
    return cop_np


def _multiply_cop_by_bodyweight_normalized_grf_magnitude(
    cop_ground_aligned: np.ndarray,
    grf_mj: np.ndarray,
    mass_kg: np.ndarray,
    height_m: np.ndarray,
) -> np.ndarray:
    """
    Scale each height-normalized, ground-aligned COP vector by that foot's |GRF|/BW.
    COP and GRF columns are [Rx,Ry,Rz,Lx,Ly,Lz].
    """
    cop_ground_aligned = np.asarray(cop_ground_aligned, dtype=np.float64)
    grf_mj = np.asarray(grf_mj, dtype=np.float64)
    mass_kg = np.asarray(mass_kg, dtype=np.float64).reshape(-1)
    height_m = np.asarray(height_m, dtype=np.float64).reshape(-1)

    if cop_ground_aligned.ndim != 2 or cop_ground_aligned.shape[1] != 6:
        raise ValueError(f"COP_CalcFrame_GroundAligned has invalid shape {cop_ground_aligned.shape}")
    if grf_mj.ndim != 2 or grf_mj.shape[1] < 6:
        raise ValueError(f"GRF_Cleaned has invalid shape {grf_mj.shape}")
    if not (cop_ground_aligned.shape[0] == grf_mj.shape[0] == mass_kg.shape[0] == height_m.shape[0]):
        raise ValueError(
            "length mismatch for COP/GRF/Mass/Height "
            f"(cop={cop_ground_aligned.shape[0]}, grf={grf_mj.shape[0]}, "
            f"mass={mass_kg.shape[0]}, height={height_m.shape[0]})"
        )

    body_weight = mass_kg * 9.8067
    if np.any(~np.isfinite(body_weight)) or np.any(body_weight <= 0.0):
        raise ValueError("invalid body-weight normalization factor")
    if np.any(~np.isfinite(height_m)) or np.any(height_m <= 0.0):
        raise ValueError("invalid height normalization factor")

    grf_mag_r = np.linalg.norm(grf_mj[:, 0:3], axis=1)
    grf_mag_l = np.linalg.norm(grf_mj[:, 3:6], axis=1)
    out = cop_ground_aligned / height_m[:, np.newaxis]
    out[:, 0:3] *= (grf_mag_r / body_weight)[:, np.newaxis]
    out[:, 3:6] *= (grf_mag_l / body_weight)[:, np.newaxis]
    return out
