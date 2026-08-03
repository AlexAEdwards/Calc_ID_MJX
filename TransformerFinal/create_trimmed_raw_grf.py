#!/usr/bin/env python3
"""Create trimmed, unfiltered GRF arrays aligned to ProcessedData frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from paths import artifact, dataset  # noqa: E402


INFO_FILENAME = "Trial_Processing_Information.json"
OUT_FILENAME = "GRF_NoFilt_Trimmed.npy"


def convert_to_mujoco_coords(vec: np.ndarray) -> np.ndarray:
    """Convert OpenSim [X, Y, Z] Y-up vectors to MuJoCo [X, -Z, Y] Z-up."""
    arr = np.asarray(vec)
    if arr.ndim == 1:
        return np.array([arr[0], -arr[2], arr[1]], dtype=arr.dtype)
    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0]
    out[:, 1] = -arr[:, 2]
    out[:, 2] = arr[:, 1]
    return out


def _load_npy_numeric(path: Path) -> np.ndarray:
    try:
        arr = np.load(path)
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr.astype(np.float64, copy=False)
    return np.asarray(arr)


def _fit_time_to_length(t: np.ndarray, target_len: int) -> np.ndarray:
    t = np.asarray(t).reshape(-1)
    if t.size == target_len:
        return t
    if t.size >= 2:
        return np.linspace(float(t[0]), float(t[-1]), target_len)
    return np.arange(target_len, dtype=float)


def _interp_to(t_src: np.ndarray, data: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return interp1d(t_src, data, kind="linear", fill_value="extrapolate", bounds_error=False)(t_new)
    out = np.empty((len(t_new), data.shape[1]), dtype=np.float64)
    for col in range(data.shape[1]):
        out[:, col] = interp1d(
            t_src,
            data[:, col],
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )(t_new)
    return out


def _align_grf_to_pelvis_yaw(pos: np.ndarray, grf: np.ndarray) -> np.ndarray:
    y_corr = R.from_euler("Y", -float(np.median(pos[:, 2])))
    out = np.asarray(grf, dtype=np.float64).copy()
    out[:, 0:3] = y_corr.apply(out[:, 0:3])
    out[:, 3:6] = y_corr.apply(out[:, 3:6])
    return out


def _motion_aligned_raw_grf(trial_dir: Path, expected_len: Optional[int]) -> tuple[np.ndarray, str]:
    motion_dir = trial_dir / "Motion"
    raw_path = motion_dir / "GRF.npy"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw GRF: {raw_path}")

    grf_raw = np.asarray(_load_npy_numeric(raw_path), dtype=np.float64)
    if grf_raw.ndim != 2 or grf_raw.shape[1] < 6:
        raise ValueError(f"Expected raw GRF shape (T, >=6), got {grf_raw.shape}")
    grf_raw = grf_raw[:, :6]

    if expected_len is not None and grf_raw.shape[0] == expected_len:
        return grf_raw, str(raw_path)

    pos_path = motion_dir / "Pos.npy"
    force_time_path = motion_dir / "Time.npy"
    kin_time_path = motion_dir / "Time_for_pos.npy"
    if not kin_time_path.exists():
        kin_time_path = trial_dir / "Motion" / "Time_for_pos.npy"
    if not (pos_path.exists() and force_time_path.exists()):
        return grf_raw, str(raw_path)

    pos = np.asarray(_load_npy_numeric(pos_path), dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] < 6:
        return grf_raw, str(raw_path)
    force_time = _fit_time_to_length(_load_npy_numeric(force_time_path), grf_raw.shape[0])
    kin_time = (
        _fit_time_to_length(_load_npy_numeric(kin_time_path), pos.shape[0])
        if kin_time_path.exists()
        else _fit_time_to_length(force_time, pos.shape[0])
    )
    t_start = max(float(kin_time[0]), float(force_time[0]))
    t_end = min(float(kin_time[-1]), float(force_time[-1]))
    t_new = np.arange(t_start, t_end, 0.01)
    if expected_len is not None and t_new.size != expected_len:
        t_new = np.linspace(t_start, t_end, expected_len)
    if t_new.size < 2:
        return grf_raw, str(raw_path)

    pos_rs = _interp_to(kin_time, pos, t_new)
    grf_rs = _interp_to(force_time, grf_raw, t_new)
    grf_aligned = _align_grf_to_pelvis_yaw(pos_rs, grf_rs)
    return grf_aligned, f"{raw_path} (resampled/aligned)"


def _read_info(processed_dir: Path) -> Dict[str, Any]:
    info_path = processed_dir / INFO_FILENAME
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {INFO_FILENAME}")
    with info_path.open("r") as f:
        return json.load(f)


def _bounds(value: Any, name: str) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list, got {value!r}")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < start:
        raise ValueError(f"{name} has invalid bounds {value!r}")
    return start, end


def _apply_bounds(arr: np.ndarray, bounds: Tuple[int, int], label: str) -> np.ndarray:
    start, end = bounds
    if end > len(arr):
        raise ValueError(f"{label} end {end} exceeds array length {len(arr)}")
    return arr[start:end]


def _processed_dirs(data_root: Path) -> Iterable[Path]:
    for info_path in sorted(data_root.glob("*/*/ProcessedData/" + INFO_FILENAME)):
        yield info_path.parent


def _load_source_grf(
    trial_dir: Path,
    core_bounds: Tuple[int, int],
    expected_len: Optional[int],
    fallback_nofilt_root: Optional[Path],
) -> Tuple[np.ndarray, str, str]:
    raw, raw_source = _motion_aligned_raw_grf(trial_dir, expected_len)

    if core_bounds[1] <= raw.shape[0]:
        return raw[:, :6], "opensim", raw_source

    if fallback_nofilt_root is not None:
        rel_trial = Path(trial_dir.parent.name) / trial_dir.name
        fallback_path = fallback_nofilt_root / rel_trial / "ProcessedData" / "GRF_Cleaned.npy"
        if fallback_path.exists():
            fallback = np.asarray(np.load(fallback_path), dtype=np.float64)
            if fallback.ndim == 2 and fallback.shape[1] >= 6 and core_bounds[1] <= fallback.shape[0]:
                return fallback[:, :6], "mujoco", str(fallback_path)

    raise ValueError(f"core trim end {core_bounds[1]} exceeds array length {raw.shape[0]}")


def create_trial_grf(
    processed_dir: Path,
    overwrite: bool = False,
    fallback_nofilt_root: Optional[Path] = None,
) -> Dict[str, Any]:
    trial_dir = processed_dir.parent
    trial_id = f"{trial_dir.parent.name}/{trial_dir.name}"
    out_path = processed_dir / OUT_FILENAME

    if out_path.exists() and not overwrite:
        arr = np.load(out_path)
        return {
            "trial": trial_id,
            "status": "exists",
            "output": str(out_path),
            "frames": int(arr.shape[0]),
        }

    info = _read_info(processed_dir)
    cleaned_path = processed_dir / "GRF_Cleaned.npy"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing ProcessedData GRF_Cleaned.npy for length check")

    core_bounds = (
        _bounds(info.get("core_trim_bounds_motion_aligned"), "core_trim_bounds_motion_aligned")
        or _bounds(info.get("grf_trim_bounds_motion_aligned"), "grf_trim_bounds_motion_aligned")
    )
    if core_bounds is None:
        raise ValueError("No motion-aligned trim bounds found in processing info")

    expected_len = info.get("core_trim_pretrim_n_frames")
    expected_len = int(expected_len) if expected_len is not None else None
    grf_source, source_frame, source_path = _load_source_grf(
        trial_dir,
        core_bounds,
        expected_len,
        fallback_nofilt_root,
    )
    grf_source = _apply_bounds(grf_source, core_bounds, "core trim")

    ds_bounds = _bounds(info.get("ds_edge_trim_bounds"), "ds_edge_trim_bounds")
    if bool(info.get("ds_edge_trim_applied", False)) and ds_bounds is not None:
        grf_source = _apply_bounds(grf_source, ds_bounds, "post-visual/ds-edge trim")

    grf_edge_bounds = _bounds(info.get("grf_edge_trim_bounds"), "grf_edge_trim_bounds")
    if bool(info.get("grf_edge_trim_applied", False)) and grf_edge_bounds is not None:
        grf_source = _apply_bounds(grf_source, grf_edge_bounds, "GRF edge trim")

    visual_bounds = _bounds(info.get("visual_keep_trim_bounds"), "visual_keep_trim_bounds")
    if bool(info.get("visual_keep_trim_applied", False)) and visual_bounds is not None:
        grf_source = _apply_bounds(grf_source, visual_bounds, "visual keep trim")

    if source_frame == "mujoco":
        grf_mj = grf_source
    else:
        grf_mj = np.hstack(
            [
                convert_to_mujoco_coords(grf_source[:, 0:3]),
                convert_to_mujoco_coords(grf_source[:, 3:6]),
            ]
        )

    was_negated = info.get("was_negated") or {}
    if bool(was_negated.get("right", False)):
        grf_mj[:, 0:3] = grf_mj[:, 0:3] * np.array([-1.0, -1.0, 1.0])
    if bool(was_negated.get("left", False)):
        grf_mj[:, 3:6] = grf_mj[:, 3:6] * np.array([-1.0, -1.0, 1.0])

    cleaned_len = int(np.load(cleaned_path, mmap_mode="r").shape[0])
    if int(grf_mj.shape[0]) != cleaned_len:
        if int(info.get("n_frames", -1)) == cleaned_len and int(grf_mj.shape[0]) > cleaned_len:
            grf_mj = grf_mj[:cleaned_len]
        else:
            raise ValueError(
                f"Trimmed raw GRF length {grf_mj.shape[0]} does not match "
                f"GRF_Cleaned length {cleaned_len}"
            )

    np.save(out_path, grf_mj.astype(np.float32))
    return {
        "trial": trial_id,
        "status": "created",
        "output": str(out_path),
        "source": source_path,
        "source_frame": source_frame,
        "source_frames": int(grf_source.shape[0]),
        "frames": int(grf_mj.shape[0]),
        "core_trim_bounds_motion_aligned": list(core_bounds),
        "ds_edge_trim_bounds": list(ds_bounds) if ds_bounds is not None else None,
        "visual_keep_trim_bounds": list(visual_bounds) if visual_bounds is not None else None,
        "grf_edge_trim_bounds": list(grf_edge_bounds) if grf_edge_bounds is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create ProcessedData/GRF_NoFilt_Trimmed.npy from Motion/GRF.npy using "
            "the saved ProcessedData trim metadata."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("OlderYoungerAdultDataset_PostVisuallyTrimmed"),
        help="Dataset root containing Subject/Trial/ProcessedData folders.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files that already exist.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(str(artifact("outputs", "trimmed_raw_grf_generation_report.json"))),
        help="JSON report path.",
    )
    parser.add_argument(
        "--fallback_nofilt_root",
        type=Path,
        default=Path("OldYoungAdultWalking_MJX_Processed_NoTrim_NoFilt"),
        help=(
            "Optional no-trim/no-filter dataset root used only when Motion/GRF.npy is "
            "already shorter than the saved ProcessedData trim window."
        ),
    )
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    fallback_root = args.fallback_nofilt_root if args.fallback_nofilt_root.exists() else None

    for processed_dir in _processed_dirs(data_root):
        try:
            records.append(
                create_trial_grf(
                    processed_dir,
                    overwrite=args.overwrite,
                    fallback_nofilt_root=fallback_root,
                )
            )
        except Exception as exc:
            trial_dir = processed_dir.parent
            errors.append(
                {
                    "trial": f"{trial_dir.parent.name}/{trial_dir.name}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = {
        "data_root": str(data_root),
        "output_filename": OUT_FILENAME,
        "n_trials_seen": len(records) + len(errors),
        "n_created": sum(1 for r in records if r["status"] == "created"),
        "n_existing": sum(1 for r in records if r["status"] == "exists"),
        "n_errors": len(errors),
        "records": records,
        "errors": errors,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Processed {summary['n_trials_seen']} trials: "
        f"{summary['n_created']} created, {summary['n_existing']} already existed, "
        f"{summary['n_errors']} errors."
    )
    print(f"Report: {args.report}")
    if errors:
        print("First errors:")
        for err in errors[:5]:
            print(f"  {err['trial']}: {err['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
