#!/usr/bin/env python3
"""Interactive COP filtering experiment on random subject/trial folders.

This is read-only: it never saves, overwrites, or backs up any dataset files.

For each randomly selected trial/source directory, the script loads
GRF_Cleaned.npy and every COP_*.npy file with a compatible time axis, then:

1. Zeros COP samples where that foot is below the vGRF contact threshold.
2. Finds stance segments where vGRF > threshold for more than 10 frames.
3. Multiplies COP by vGRF inside each stance segment.
4. Applies a 2nd-order 6 Hz Butterworth filtfilt with zero padding.
5. Divides by vGRF inside stance segments to restore COP length units.
6. Plots dataset COP, this experiment's cleaned COP, and the COP cleaner from
   ProcessData.py for visual inspection.

Close the figure window to move to another random trial. Press Ctrl-C in the
terminal to quit.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

# Repo root via paths.py, not via __file__: this script no longer lives at
# the repo root, so its own directory is not the base for relative --dataset.
from paths import REPO_ROOT
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt


DEFAULT_DATASET = "TrustedDataSetNoised12DistributedUnFiltered_Trimmed"
DEFAULT_SOURCE_DIRS = ("ProcessedData", "MoCap")
GRF_FILENAME = "GRF_Cleaned.npy"
TRAIN_COP_FILENAME = "COP_CalcFrame_GroundAligned.npy"
VGRF_THRESHOLD_N = 1.0
MIN_STANCE_FRAMES = 10
FILTER_CUTOFF_HZ = 6.0
FILTER_FS_HZ = 100.0
FILTER_ORDER = 2
ZERO_PAD_FRAMES = 30
PROCESSDATA_COP_TRIM_START_FRAMES = 3
PROCESSDATA_COP_TRIM_END_FRAMES = 3
PROCESSDATA_COP_FILTER_PAD_WIDTH = 15
PROCESSDATA_COP_EXTRAPOLATION_FRAMES = 6


def _extract_train_cop_channels(cop: np.ndarray) -> np.ndarray | None:
    """Return the 4 COP channels used by training: [Rx, Rz, Lx, Lz]."""
    cop = np.asarray(cop)
    if cop.ndim != 2:
        return None
    if cop.shape[1] >= 6:
        return cop[:, [0, 2, 3, 5]]
    if cop.shape[1] == 4:
        return cop
    return None


def _is_cop_candidate(path: Path) -> bool:
    return path.name == TRAIN_COP_FILENAME


def _discover_source_dirs(dataset_root: Path, source_dir_names: Iterable[str]) -> list[Path]:
    source_dirs: list[Path] = []
    seen: set[Path] = set()
    for trial_dir in sorted(p for p in dataset_root.rglob("Trial_*") if p.is_dir()):
        for name in source_dir_names:
            source_dir = trial_dir / name
            if not source_dir.is_dir():
                continue
            if not (source_dir / GRF_FILENAME).exists():
                continue
            if not any(_is_cop_candidate(p) for p in source_dir.glob("COP_*.npy")):
                continue
            resolved = source_dir.resolve()
            if resolved not in seen:
                source_dirs.append(source_dir)
                seen.add(resolved)
    return source_dirs


def _leg_layout(n_cols: int) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if n_cols == 6:
        return (0, 1, 2), (3, 4, 5)
    if n_cols == 4:
        return (0, 1), (2, 3)
    return None


def _stance_segments(vgrf: np.ndarray,
                     threshold: float = VGRF_THRESHOLD_N,
                     min_frames: int = MIN_STANCE_FRAMES) -> list[tuple[int, int]]:
    contact = np.asarray(vgrf) > float(threshold)
    padded = np.concatenate(([False], contact, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if int(end) - int(start) > int(min_frames)
    ]


def _filtfilt_zero_padded(x: np.ndarray,
                          cutoff_hz: float = FILTER_CUTOFF_HZ,
                          fs_hz: float = FILTER_FS_HZ,
                          order: int = FILTER_ORDER,
                          pad_frames: int = ZERO_PAD_FRAMES) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < max(4, order * 3):
        return x.copy()
    b, a = butter(order, cutoff_hz / (0.5 * fs_hz), btype="low")
    pad = int(max(0, pad_frames))
    padded = np.pad(x, (pad, pad), mode="constant", constant_values=0.0)
    try:
        y = filtfilt(b, a, padded)
    except ValueError:
        return x.copy()
    return y[pad:pad + x.size]


def _processdata_filter_segment_wise(data: np.ndarray,
                                     vertical_force: np.ndarray,
                                     cutoff: float = 6.0,
                                     fs: float = 100.0,
                                     order: int = 2,
                                     pad_width: int = 15,
                                     force_threshold: float = 1.0) -> np.ndarray:
    """Local copy of ProcessData.py filter_segment_wise."""
    result = data.copy()
    is_stance = vertical_force > force_threshold
    is_padded = np.concatenate(([False], is_stance, [False]))
    diffs = np.diff(is_padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")

    for start, end in zip(starts, ends):
        seg = data[start:end]
        if seg.size == 0:
            continue
        pad_s = min(pad_width, start)
        pad_e = min(pad_width, data.shape[0] - end)

        pre = np.zeros((pad_s,) + seg.shape[1:], dtype=seg.dtype)
        post = np.zeros((pad_e,) + seg.shape[1:], dtype=seg.dtype)
        padded = np.concatenate([pre, seg, post], axis=0)

        try:
            if padded.ndim == 1:
                filt = filtfilt(b, a, padded)[pad_s: pad_s + (end - start)]
            else:
                filt_full = np.empty_like(padded)
                for col in range(padded.shape[1]):
                    filt_full[:, col] = filtfilt(b, a, padded[:, col])
                filt = filt_full[pad_s: pad_s + (end - start)]
        except ValueError:
            continue

        result[start:end] = filt
    return result


def _processdata_clean_and_filter_cop(cop_data: np.ndarray,
                                      grf_data: np.ndarray,
                                      trim_start_frames: int = 8,
                                      trim_end_frames: int = 8,
                                      extrapolation_frames: int = 3,
                                      pad_width: int = 15,
                                      cutoff: float = 6.0,
                                      fs: float = 100.0,
                                      order: int = 2,
                                      outlier_threshold: float = 5.0) -> np.ndarray:
    """Local copy of ProcessData.py clean_and_filter_cop."""
    cop_np = cop_data.copy()
    foot_configs = [
        (0, 1, 2),
        (2, 3, 5),
    ]

    for col_x, col_y, grf_idx in foot_configs:
        outlier_mask = (
            (np.abs(cop_np[:, col_x]) > outlier_threshold)
            | (np.abs(cop_np[:, col_y]) > outlier_threshold)
        )
        cop_np[outlier_mask, col_x] = 0.0
        cop_np[outlier_mask, col_y] = 0.0

        is_nonzero = (np.abs(cop_np[:, col_x]) > 1e-9) | (np.abs(cop_np[:, col_y]) > 1e-9)
        is_nz_pad = np.concatenate(([False], is_nonzero, [False]))
        diff = np.diff(is_nz_pad.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            seg_len = end - start
            if seg_len <= (trim_start_frames + trim_end_frames):
                cop_np[start:end, col_x] = 0.0
                cop_np[start:end, col_y] = 0.0
                continue

            trim_start = start + trim_start_frames
            slope_end = min(trim_start + int(extrapolation_frames), end)
            if slope_end > trim_start + 1:
                sx = np.mean(np.diff(cop_np[trim_start:slope_end, col_x]))
                sy = np.mean(np.diff(cop_np[trim_start:slope_end, col_y]))
                for k in range(trim_start_frames):
                    d = trim_start_frames - k
                    cop_np[start + k, col_x] = cop_np[trim_start, col_x] - d * sx / 6
                    cop_np[start + k, col_y] = cop_np[trim_start, col_y] - d * sy / 6
            else:
                cop_np[start:trim_start, col_x] = cop_np[trim_start, col_x]
                cop_np[start:trim_start, col_y] = cop_np[trim_start, col_y]

            trim_end = end - trim_end_frames
            slope_start = max(start, trim_end - int(extrapolation_frames))
            if trim_end > slope_start + 1:
                sx = np.mean(np.diff(cop_np[slope_start:trim_end, col_x]))
                sy = np.mean(np.diff(cop_np[slope_start:trim_end, col_y]))
                for k in range(trim_end_frames):
                    d = k + 1
                    cop_np[trim_end + k, col_x] = cop_np[trim_end - 1, col_x] + d * sx / 6
                    cop_np[trim_end + k, col_y] = cop_np[trim_end - 1, col_y] + d * sy / 6
            else:
                cop_np[trim_end:end, col_x] = cop_np[trim_end - 1, col_x]
                cop_np[trim_end:end, col_y] = cop_np[trim_end - 1, col_y]

        cop_np[:, col_x] = _processdata_filter_segment_wise(
            cop_np[:, col_x],
            grf_data[:, grf_idx],
            cutoff=cutoff,
            fs=fs,
            order=order,
            pad_width=pad_width,
            force_threshold=1.0,
        )
        cop_np[:, col_y] = _processdata_filter_segment_wise(
            cop_np[:, col_y],
            grf_data[:, grf_idx],
            cutoff=cutoff,
            fs=fs,
            order=order,
            pad_width=pad_width,
            force_threshold=1.0,
        )

    cop_np[grf_data[:, 2] < 1.0, 0:2] = 0.0
    cop_np[grf_data[:, 5] < 1.0, 2:4] = 0.0
    return cop_np


def clean_cop_experiment(cop: np.ndarray, grf: np.ndarray) -> np.ndarray | None:
    """Apply the experimental vGRF-weighted COP filtering pass."""
    cop = np.asarray(cop, dtype=np.float64)
    grf = np.asarray(grf, dtype=np.float64)
    if cop.ndim != 2 or grf.ndim != 2 or grf.shape[1] < 6:
        return None
    if cop.shape[0] != grf.shape[0]:
        return None

    layout = _leg_layout(cop.shape[1])
    if layout is None:
        return None

    out = cop.copy()
    foot_specs = [
        (layout[0], grf[:, 2]),
        (layout[1], grf[:, 5]),
    ]

    for cols, vgrf in foot_specs:
        swing_mask = vgrf <= VGRF_THRESHOLD_N
        out[np.ix_(swing_mask, cols)] = 0.0

        for start, end in _stance_segments(vgrf):
            force = np.maximum(vgrf[start:end], VGRF_THRESHOLD_N)
            for col in cols:
                weighted = out[start:end, col] * force
                filtered_weighted = _filtfilt_zero_padded(weighted)
                out[start:end, col] = filtered_weighted / force

        out[np.ix_(swing_mask, cols)] = 0.0

    return out.astype(cop.dtype, copy=False)


def clean_cop_processdata(cop_train: np.ndarray, grf: np.ndarray) -> np.ndarray | None:
    """Apply the current ProcessData.py COP cleaner to train-used COP channels."""
    cop_train = np.asarray(cop_train)
    grf = np.asarray(grf)
    if cop_train.ndim != 2 or cop_train.shape[1] != 4:
        return None
    if grf.ndim != 2 or grf.shape[1] < 6 or grf.shape[0] != cop_train.shape[0]:
        return None
    cleaned = _processdata_clean_and_filter_cop(
        cop_train,
        grf,
        trim_start_frames=PROCESSDATA_COP_TRIM_START_FRAMES,
        trim_end_frames=PROCESSDATA_COP_TRIM_END_FRAMES,
        extrapolation_frames=PROCESSDATA_COP_EXTRAPOLATION_FRAMES,
        pad_width=PROCESSDATA_COP_FILTER_PAD_WIDTH,
        cutoff=FILTER_CUTOFF_HZ,
        fs=FILTER_FS_HZ,
        order=FILTER_ORDER,
    )
    return cleaned.astype(cop_train.dtype, copy=False)


def _load_trial_cop_results(source_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    grf = np.load(source_dir / GRF_FILENAME)
    results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for cop_path in sorted(p for p in source_dir.glob("COP_*.npy") if _is_cop_candidate(p)):
        try:
            loaded = np.load(cop_path)
        except Exception:
            continue
        before = _extract_train_cop_channels(loaded)
        if before is None:
            continue
        experiment = clean_cop_experiment(before, grf)
        processdata = clean_cop_processdata(before, grf)
        if experiment is None or processdata is None:
            continue
        results.append((cop_path.name, before, experiment, processdata))
    return results


def _plot_results(source_dir: Path, results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(18, max(3.0, 2.3 * n)), squeeze=False)
    trial_label = "/".join(source_dir.parts[-3:])
    fig.suptitle(
        f"Train COP channels [Rx, Rz, Lx, Lz]: {trial_label}",
        fontsize=13,
    )

    channel_labels = ("R x", "R z", "L x", "L z")
    for row, (name, before, experiment, processdata) in enumerate(results):
        t = np.arange(before.shape[0])
        panels = (
            ("Dataset COP", before),
            ("Experiment Cleaned COP", experiment),
            ("ProcessData COP Cleaning", processdata),
        )
        for col_idx, (title, arr) in enumerate(panels):
            ax = axes[row, col_idx]
            for cop_col, label in enumerate(channel_labels[:arr.shape[1]]):
                ax.plot(t, arr[:, cop_col], linewidth=0.9, label=label)
            ax.set_title(f"{title}: {name}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("COP")
            if row == 0:
                ax.legend(loc="upper right", fontsize=7, ncol=2)
        if row == n - 1:
            for ax in axes[row]:
                ax.set_xlabel("Frame")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Dataset root containing subject/Trial_* folders")
    parser.add_argument("--source_dirs", default=",".join(DEFAULT_SOURCE_DIRS),
                        help="Comma-separated source dirs to sample, default ProcessedData,MoCap")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for repeatable trial order")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = REPO_ROOT / dataset_root
    source_dir_names = tuple(
        name.strip() for name in str(args.source_dirs).split(",") if name.strip()
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not source_dir_names:
        raise ValueError("--source_dirs must include at least one directory name")

    rng = random.Random(args.seed)
    source_dirs = _discover_source_dirs(dataset_root, source_dir_names)
    if not source_dirs:
        raise RuntimeError(
            f"No source dirs with {GRF_FILENAME} and {TRAIN_COP_FILENAME} found under {dataset_root}"
        )

    print(f"Discovered {len(source_dirs)} candidate source directories.")
    print("Close each plot window to sample another random trial. Press Ctrl-C to quit.")

    while True:
        source_dir = rng.choice(source_dirs)
        results = _load_trial_cop_results(source_dir)
        if not results:
            continue
        print(f"Plotting {'/'.join(source_dir.parts[-3:])} ({len(results)} COP files)")
        _plot_results(source_dir, results)


if __name__ == "__main__":
    main()
