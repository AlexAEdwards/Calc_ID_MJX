#!/usr/bin/env python3
"""Find trials containing short or low-peak GRF stance phases.

The script scans subject/Trial_* folders for GRF_Cleaned.npy files, identifies
right/left stance segments from vertical GRF, excludes stance segments touching
the first or last frame of the trial, and reports trials with at least one
"short" stance.

A stance is flagged when either:
  - duration is shorter than --max_frames, default 25 frames
  - peak vertical GRF never exceeds --min_peak_n, default 50 N

This script is read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# Repo root via paths.py, not via __file__: this script no longer lives at
# the repo root, so its own directory is not the base for relative --dataset.
from paths import REPO_ROOT
from typing import Iterable

import numpy as np


DEFAULT_DATASET = "Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects_NoTrim_NoFilt"
DEFAULT_SOURCE_DIRS = ("ProcessedData", "MoCap")
GRF_FILENAME = "GRF_Cleaned.npy"
RIGHT_VGRF_COL = 2
LEFT_VGRF_COL = 5


@dataclass
class ShortStance:
    foot: str
    start: int
    end: int
    duration: int
    peak_n: float
    reasons: tuple[str, ...]


def _stance_segments(vgrf: np.ndarray, threshold_n: float) -> list[tuple[int, int]]:
    contact = np.asarray(vgrf, dtype=np.float64) > float(threshold_n)
    padded = np.concatenate(([False], contact, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _iter_grf_sources(dataset_root: Path, source_dir_names: Iterable[str]) -> list[Path]:
    source_dirs: list[Path] = []
    seen: set[Path] = set()
    for trial_dir in sorted(p for p in dataset_root.rglob("Trial_*") if p.is_dir()):
        for source_name in source_dir_names:
            source_dir = trial_dir / source_name
            grf_path = source_dir / GRF_FILENAME
            if not grf_path.exists():
                continue
            resolved = source_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            source_dirs.append(source_dir)
    return source_dirs


def analyze_grf_file(
    grf_path: Path,
    *,
    contact_threshold_n: float,
    max_frames: int,
    min_peak_n: float,
) -> list[ShortStance]:
    grf = np.asarray(np.load(grf_path), dtype=np.float64)
    if grf.ndim != 2 or grf.shape[1] <= max(RIGHT_VGRF_COL, LEFT_VGRF_COL):
        raise ValueError(f"Unexpected GRF shape {grf.shape}")

    n_frames = int(grf.shape[0])
    flagged: list[ShortStance] = []
    for foot, col in (("R", RIGHT_VGRF_COL), ("L", LEFT_VGRF_COL)):
        vgrf = grf[:, col]
        for start, end in _stance_segments(vgrf, contact_threshold_n):
            if start == 0 or end == n_frames:
                continue
            duration = int(end - start)
            peak_n = float(np.nanmax(vgrf[start:end])) if end > start else 0.0
            reasons: list[str] = []
            if duration < int(max_frames):
                reasons.append(f"duration<{int(max_frames)}")
            if peak_n <= float(min_peak_n):
                reasons.append(f"peak<={float(min_peak_n):g}N")
            if reasons:
                flagged.append(
                    ShortStance(
                        foot=foot,
                        start=int(start),
                        end=int(end),
                        duration=duration,
                        peak_n=peak_n,
                        reasons=tuple(reasons),
                    )
                )
    return flagged


def _format_trial_label(source_dir: Path, dataset_root: Path) -> str:
    try:
        return str(source_dir.relative_to(dataset_root))
    except ValueError:
        return str(source_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Dataset root containing subject/Trial_* folders")
    parser.add_argument("--source_dirs", default=",".join(DEFAULT_SOURCE_DIRS),
                        help="Comma-separated per-trial folders to scan")
    parser.add_argument("--contact_threshold_n", type=float, default=1.0,
                        help="Vertical GRF threshold for stance detection, default 1 N")
    parser.add_argument("--max_frames", type=int, default=25,
                        help="Flag stances shorter than this many frames, default 25")
    parser.add_argument("--min_peak_n", type=float, default=50.0,
                        help="Flag stances whose peak vertical GRF is <= this value, default 50 N")
    parser.add_argument("--details", action="store_true",
                        help="Print each flagged stance, not just per-trial counts")
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
        raise ValueError("--source_dirs must include at least one source directory")

    source_dirs = _iter_grf_sources(dataset_root, source_dir_names)
    results: list[tuple[Path, list[ShortStance]]] = []
    errors: list[tuple[Path, str]] = []

    for source_dir in source_dirs:
        grf_path = source_dir / GRF_FILENAME
        try:
            flagged = analyze_grf_file(
                grf_path,
                contact_threshold_n=float(args.contact_threshold_n),
                max_frames=int(args.max_frames),
                min_peak_n=float(args.min_peak_n),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append((grf_path, str(exc)))
            continue
        if flagged:
            results.append((source_dir, flagged))

    print(f"Scanned {len(source_dirs)} GRF source directories under {dataset_root}")
    print(
        "Short stance criteria: "
        f"duration < {int(args.max_frames)} frames OR peak <= {float(args.min_peak_n):g} N; "
        f"contact threshold = {float(args.contact_threshold_n):g} N; edge stances excluded"
    )
    print()

    if not results:
        print("No non-edge short/low-peak stances found.")
    else:
        print(f"Trials/source folders containing short stances: {len(results)}")
        for source_dir, stances in results:
            label = _format_trial_label(source_dir, dataset_root)
            print(f"- {label}: {len(stances)} short stance(s)")
            if args.details:
                for stance in stances:
                    reasons = ", ".join(stance.reasons)
                    print(
                        f"    {stance.foot} [{stance.start}:{stance.end}] "
                        f"duration={stance.duration} peak={stance.peak_n:.2f} N "
                        f"({reasons})"
                    )

    if errors:
        print()
        print(f"Errors while reading {len(errors)} GRF file(s):")
        for path, msg in errors[:25]:
            print(f"- {path}: {msg}")
        if len(errors) > 25:
            print(f"... {len(errors) - 25} more")


if __name__ == "__main__":
    main()
