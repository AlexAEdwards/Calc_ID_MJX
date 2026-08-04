#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import ezc3d
import numpy as np


TRIAL_RE = re.compile(r"Trial_(\d+)$")
RIGHT_MARKERS = ("R.Heel", "R.MT1", "R.MT2", "R.MT5", "R.Ankle")
LEFT_MARKERS = ("L.Heel", "L.MT1", "L.MT2", "L.MT5", "L.Ankle")
PROCESSDATA_TRIAL_SUBSET = (
    "PD_SUB01_off/Trial_1",
    "PD_SUB02_off/Trial_14",
    "PD_SUB03_on/Trial_17",
    "PD_SUB05_on/Trial_5",
    "PD_SUB06_on/Trial_8",
    "PD_SUB08_off/Trial_17",
    "PD_SUB10_off/Trial_12",
    "PD_SUB11_off/Trial_15",
    "PD_SUB12_off/Trial_4",
    "PD_SUB13_on/Trial_10",
    "PD_SUB14_on/Trial_12",
    "PD_SUB16_off/Trial_5",
    "PD_SUB17_on/Trial_11",
    "PD_SUB18_on/Trial_13",
    "PD_SUB19_on/Trial_17",
    "PD_SUB20_on/Trial_4",
    "PD_SUB22_off/Trial_10",
    "PD_SUB23_on/Trial_17",
    "PD_SUB24_on/Trial_6",
    "PD_SUB26_off/Trial_9",
)


def finite_minmax(values: np.ndarray) -> list[float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return [None, None]
    return [float(np.nanmin(finite)), float(np.nanmax(finite))]


def subject_to_preaddbio_name(subject: str) -> str:
    if not subject.startswith("PD_"):
        raise ValueError(f"Unexpected PD subject name: {subject}")
    return subject[3:]


def trial_to_walk_trc(preaddbio_subject_dir: Path, trial_num: int) -> Path:
    trcs = sorted(preaddbio_subject_dir.glob("walk*.trc"))
    if trial_num < 1 or trial_num > len(trcs):
        raise ValueError(
            f"Trial {trial_num} is outside addBio_Boari.m walk*.trc mapping "
            f"for {preaddbio_subject_dir}"
        )
    return trcs[trial_num - 1]


def walk_trc_to_c3d(preaddbio_subject_dir: Path, walk_trc: Path) -> Path:
    walk_num = walk_trc.stem.removeprefix("walk")
    c3d = preaddbio_subject_dir / f"{preaddbio_subject_dir.name}_walk_{walk_num}.c3d"
    if not c3d.exists():
        raise FileNotFoundError(c3d)
    return c3d


def trial_to_c3d_and_trc(preaddbio_subject_dir: Path, trial_num: int) -> tuple[Path, Path]:
    walk_trc = trial_to_walk_trc(preaddbio_subject_dir, trial_num)
    return walk_trc_to_c3d(preaddbio_subject_dir, walk_trc), walk_trc


def marker_centroid_m(points: np.ndarray, labels: list[str], markers: tuple[str, ...]) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    marker_arrays = []
    for marker in markers:
        idx = label_to_idx.get(marker)
        if idx is None:
            continue
        xyz = points[:3, idx, :].T.astype(np.float64) / 1000.0
        xyz[~np.isfinite(xyz)] = np.nan
        marker_arrays.append(xyz)
    if not marker_arrays:
        return np.full((points.shape[2], 3), np.nan, dtype=np.float64)
    return np.nanmean(np.stack(marker_arrays, axis=0), axis=0)


def read_trc_markers(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Frame#"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"No Frame# header found in {path}")

    header = lines[header_idx].rstrip("\n").split("\t")
    marker_names = [name for name in header[2:] if name]
    data = {name: [] for name in marker_names}
    for line in lines[header_idx + 2:]:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        for marker_idx, marker in enumerate(marker_names):
            col = 2 + marker_idx * 3
            try:
                data[marker].append([float(parts[col]), float(parts[col + 1]), float(parts[col + 2])])
            except (IndexError, ValueError):
                data[marker].append([np.nan, np.nan, np.nan])
    return {marker: np.asarray(values, dtype=np.float64) for marker, values in data.items()}


def estimate_c3d_to_trc(points: np.ndarray, labels: list[str], trc_path: Path) -> tuple[np.ndarray, np.ndarray, float, int]:
    trc = read_trc_markers(trc_path)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    source_rows = []
    target_rows = []
    for marker in sorted(set(label_to_idx).intersection(trc)):
        source = points[:3, label_to_idx[marker], :].T.astype(np.float64)
        target = trc[marker]
        n = min(source.shape[0], target.shape[0])
        if n == 0:
            continue
        source = source[:n]
        target = target[:n]
        valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        if np.any(valid):
            source_rows.append(source[valid])
            target_rows.append(target[valid])
    if not source_rows:
        raise ValueError(f"No shared finite C3D/TRC marker samples for {trc_path}")

    source_all = np.vstack(source_rows)
    target_all = np.vstack(target_rows)
    if source_all.shape[0] < 30:
        raise ValueError(f"Too few shared C3D/TRC marker samples for {trc_path}: {source_all.shape[0]}")

    source_mean = source_all.mean(axis=0)
    target_mean = target_all.mean(axis=0)
    source_centered = source_all - source_mean
    target_centered = target_all - target_mean
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation_mm = target_mean - source_mean @ rotation
    residual = source_all @ rotation + translation_mm - target_all
    rms_error_mm = float(np.sqrt(np.nanmean(np.sum(residual * residual, axis=1))))
    return rotation, translation_mm, rms_error_mm, int(source_all.shape[0])


def repair_force_platform_metadata(c3d: ezc3d.c3d) -> list[int]:
    force_platform = c3d["parameters"].get("FORCE_PLATFORM")
    if not force_platform:
        return []
    channels = np.asarray(force_platform["CHANNEL"]["value"])
    platform_types = np.asarray(force_platform["TYPE"]["value"]).astype(int)
    valid = []
    for idx, platform_type in enumerate(platform_types):
        required_channels = 8 if platform_type == 3 else 6
        if np.all(channels[:required_channels, idx] > 0):
            valid.append(idx)

    force_platform["USED"]["value"] = np.asarray([len(valid)], dtype=int)
    for key in ("TYPE", "CORNERS", "ORIGIN", "CHANNEL", "CAL_MATRIX"):
        arr = np.asarray(force_platform[key]["value"])
        if arr.ndim >= 1 and arr.shape[-1] == len(platform_types):
            force_platform[key]["value"] = arr[..., valid]
    return valid


def platform_bounds_xz_m(platform: dict[str, Any], margin_m: float) -> tuple[float, float, float, float]:
    corners = np.asarray(platform["corners"], dtype=np.float64) / 1000.0
    x = corners[0]
    z = corners[2]
    return (
        float(np.nanmin(x) - margin_m),
        float(np.nanmax(x) + margin_m),
        float(np.nanmin(z) - margin_m),
        float(np.nanmax(z) + margin_m),
    )


def quiet_baseline(values: np.ndarray, sample_rate_hz: float, quiet_seconds: float, lowest_fraction: float) -> tuple[np.ndarray, int, np.ndarray]:
    n = values.shape[0]
    edge_n = int(max(1, min(n, round(sample_rate_hz * quiet_seconds))))
    edge_idx = np.r_[0:edge_n, max(0, n - edge_n):n]
    vertical_abs = np.abs(values[:, 1])
    finite_vertical = np.isfinite(vertical_abs)
    if np.any(finite_vertical):
        rank_count = int(max(edge_n, round(n * lowest_fraction)))
        rank_idx = np.argsort(np.where(finite_vertical, vertical_abs, np.inf))[:rank_count]
        candidate_idx = np.unique(np.r_[edge_idx, rank_idx])
    else:
        candidate_idx = np.unique(edge_idx)
    candidates = values[candidate_idx]
    good = np.isfinite(candidates).all(axis=1)
    if good.sum() == 0:
        return np.zeros(values.shape[1], dtype=np.float64), 0, candidate_idx[:0]
    return np.nanmedian(candidates[good], axis=0), int(good.sum()), candidate_idx[good]


def normalize_platform_sign(force_corr: np.ndarray, force_threshold_n: float) -> int:
    vertical = force_corr[:, 1]
    active = np.abs(vertical) > force_threshold_n
    if np.sum(active) < 5:
        return 1
    return 1 if float(np.nanmedian(vertical[active])) >= 0.0 else -1


def downsample_platform(
    platform: dict[str, Any],
    ratio: int,
    analog_rate: float,
    force_threshold_n: float,
    max_force_n: float,
    cop_margin_m: float,
    baseline_seconds: float,
    baseline_lowest_fraction: float,
) -> dict[str, Any]:
    force_raw = np.asarray(platform["force"], dtype=np.float64).T
    moment_raw_nm = np.asarray(platform["moment"], dtype=np.float64).T / 1000.0
    cop_raw_m = np.asarray(platform["center_of_pressure"], dtype=np.float64).T / 1000.0
    n = min(len(force_raw), len(moment_raw_nm), len(cop_raw_m))
    n = (n // ratio) * ratio
    force_raw = force_raw[:n]
    moment_raw_nm = moment_raw_nm[:n]
    cop_raw_m = cop_raw_m[:n]

    force_offset, n_baseline, baseline_idx = quiet_baseline(force_raw, analog_rate, baseline_seconds, baseline_lowest_fraction)
    moment_candidates = moment_raw_nm[baseline_idx] if baseline_idx.size else moment_raw_nm
    moment_good = np.isfinite(moment_candidates).all(axis=1)
    moment_offset_nm = np.nanmedian(moment_candidates[moment_good], axis=0) if np.any(moment_good) else np.zeros(3)

    force_corr = force_raw - force_offset
    moment_corr_nm = moment_raw_nm - moment_offset_nm
    sign = normalize_platform_sign(force_corr, force_threshold_n)
    force_corr *= sign
    moment_corr_nm *= sign

    xmin, xmax, zmin, zmax = platform_bounds_xz_m(platform, cop_margin_m)
    active = force_corr[:, 1] > force_threshold_n
    active &= np.linalg.norm(force_corr, axis=1) <= max_force_n
    active &= np.isfinite(cop_raw_m).all(axis=1)
    active &= cop_raw_m[:, 0] >= xmin
    active &= cop_raw_m[:, 0] <= xmax
    active &= cop_raw_m[:, 2] >= zmin
    active &= cop_raw_m[:, 2] <= zmax

    force_blocks = force_corr.reshape(-1, ratio, 3)
    moment_blocks = moment_corr_nm.reshape(-1, ratio, 3)
    cop_blocks = cop_raw_m.reshape(-1, ratio, 3)
    active_blocks = active.reshape(-1, ratio)

    force_ds = np.nanmean(force_blocks, axis=1)
    moment_ds = np.nanmean(moment_blocks, axis=1)
    weights = np.where(active_blocks, np.maximum(force_blocks[:, :, 1], 0.0), 0.0)
    denom = weights.sum(axis=1)
    cop_ds = np.full((force_ds.shape[0], 3), np.nan, dtype=np.float64)
    valid = denom > 0
    if np.any(valid):
        cop_ds[valid] = (cop_blocks[valid] * weights[valid, :, None]).sum(axis=1) / denom[valid, None]
    active_ds = valid & (force_ds[:, 1] > force_threshold_n)

    return {
        "force": force_ds,
        "moment": moment_ds,
        "cop": cop_ds,
        "active": active_ds,
        "sign_multiplier": sign,
        "force_offset_n": force_offset.tolist(),
        "moment_offset_nm": moment_offset_nm.tolist(),
        "baseline_samples": n_baseline,
        "bounds_xz_m": [xmin, xmax, zmin, zmax],
        "active_frames": int(np.sum(active_ds)),
        "raw_force_minmax_n": [finite_minmax(force_raw[:, axis]) for axis in range(3)],
        "corrected_force_minmax_n": [finite_minmax(force_corr[:, axis]) for axis in range(3)],
    }


def raw_vector_to_output(vector_c3d: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return vector_c3d @ rotation


def raw_point_to_output(point_c3d_m: np.ndarray, rotation: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    return ((point_c3d_m * 1000.0) @ rotation + translation_mm) / 1000.0


def load_c3d_plate_data(
    c3d_path: Path,
    trc_path: Path,
    force_threshold_n: float,
    max_force_n: float,
    cop_margin_m: float,
    baseline_seconds: float,
    baseline_lowest_fraction: float,
    max_transform_rms_mm: float,
) -> dict[str, Any]:
    base = ezc3d.c3d(str(c3d_path), extract_forceplat_data=False)
    valid_platforms = repair_force_platform_metadata(base)
    with tempfile.NamedTemporaryFile(suffix=".c3d", delete=True) as tmp:
        base.write(tmp.name)
        c3d = ezc3d.c3d(tmp.name, extract_forceplat_data=True)

    analog_rate = float(c3d["header"]["analogs"]["frame_rate"])
    point_rate = float(c3d["header"]["points"]["frame_rate"])
    ratio = int(round(analog_rate / point_rate))
    if ratio < 1:
        raise ValueError(f"Invalid analog/point rate ratio in {c3d_path}: {analog_rate}/{point_rate}")

    labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    points = np.asarray(c3d["data"]["points"], dtype=np.float64)
    right_foot = marker_centroid_m(points, labels, RIGHT_MARKERS)
    left_foot = marker_centroid_m(points, labels, LEFT_MARKERS)

    try:
        rotation, translation_mm, transform_rms_mm, transform_samples = estimate_c3d_to_trc(points, labels, trc_path)
        transform_error = None
    except Exception as exc:
        rotation = np.eye(3)
        translation_mm = np.zeros(3)
        transform_rms_mm = float("nan")
        transform_samples = 0
        transform_error = str(exc)
    use_transform = transform_error is None and np.isfinite(transform_rms_mm) and transform_rms_mm <= max_transform_rms_mm
    if not use_transform:
        rotation = np.eye(3)
        translation_mm = np.zeros(3)

    platforms = []
    for platform_idx, platform in enumerate(c3d["data"]["platform"], start=1):
        extracted = downsample_platform(
            platform,
            ratio=ratio,
            analog_rate=analog_rate,
            force_threshold_n=force_threshold_n,
            max_force_n=max_force_n,
            cop_margin_m=cop_margin_m,
            baseline_seconds=baseline_seconds,
            baseline_lowest_fraction=baseline_lowest_fraction,
        )
        platforms.append({"platform_index": platform_idx, **extracted})

    n_frames = min(
        [points.shape[2], right_foot.shape[0], left_foot.shape[0]]
        + [platform["force"].shape[0] for platform in platforms]
    )
    return {
        "analog_rate": analog_rate,
        "point_rate": point_rate,
        "ratio": ratio,
        "frames": n_frames,
        "valid_platform_indices": [int(i) for i in valid_platforms],
        "right_foot": right_foot[:n_frames],
        "left_foot": left_foot[:n_frames],
        "rotation_c3d_to_trc": rotation,
        "translation_c3d_to_trc_mm": translation_mm,
        "transform_rms_error_mm": transform_rms_mm,
        "transform_samples": transform_samples,
        "transform_error": transform_error,
        "used_transform": bool(use_transform),
        "platforms": [
            {key: (value[:n_frames] if isinstance(value, np.ndarray) and value.shape[:1] == (platform["force"].shape[0],) else value)
             for key, value in platform.items()}
            for platform in platforms
        ],
    }


def combine_platforms_by_foot(extracted: dict[str, Any], ambiguity_margin_m: float, max_ambiguous_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_frames = int(extracted["frames"])
    grf = np.zeros((n_frames, 6), dtype=np.float64)
    cop = np.zeros((n_frames, 6), dtype=np.float64)
    grm = np.zeros((n_frames, 6), dtype=np.float64)
    stats: dict[str, Any] = {
        "right_assignments": 0,
        "left_assignments": 0,
        "unassigned_active_platform_frames": 0,
        "active_platform_frames": 0,
        "inactive_or_rejected_platform_frames": 0,
        "ambiguous_platform_frames": 0,
        "ambiguous_examples": [],
    }
    right_xz = extracted["right_foot"][:, [0, 2]]
    left_xz = extracted["left_foot"][:, [0, 2]]
    rotation = extracted["rotation_c3d_to_trc"]
    translation_mm = extracted["translation_c3d_to_trc_mm"]

    for frame in range(n_frames):
        side_payload: dict[str, list[dict[str, Any]]] = {"right": [], "left": []}
        for platform in extracted["platforms"]:
            if not platform["active"][frame]:
                stats["inactive_or_rejected_platform_frames"] += 1
                continue
            stats["active_platform_frames"] += 1
            cop_frame = platform["cop"][frame]
            if not np.isfinite(cop_frame).all():
                stats["unassigned_active_platform_frames"] += 1
                continue
            cop_xz = cop_frame[[0, 2]]
            d_right = np.linalg.norm(cop_xz - right_xz[frame]) if np.isfinite(right_xz[frame]).all() else np.inf
            d_left = np.linalg.norm(cop_xz - left_xz[frame]) if np.isfinite(left_xz[frame]).all() else np.inf
            if not np.isfinite(d_right) and not np.isfinite(d_left):
                stats["unassigned_active_platform_frames"] += 1
                continue
            side = "right" if d_right <= d_left else "left"
            diff = abs(d_right - d_left)
            if diff <= ambiguity_margin_m:
                stats["ambiguous_platform_frames"] += 1
                if len(stats["ambiguous_examples"]) < max_ambiguous_samples:
                    stats["ambiguous_examples"].append({
                        "frame": int(frame),
                        "time_s": float(frame / extracted["point_rate"]),
                        "platform_index": int(platform["platform_index"]),
                        "assigned": side,
                        "right_distance_m": float(d_right),
                        "left_distance_m": float(d_left),
                        "distance_difference_m": float(diff),
                        "cop_xz_m": [float(cop_xz[0]), float(cop_xz[1])],
                    })
            side_payload[side].append(platform)
            stats[f"{side}_assignments"] += 1

        for side, sl in (("right", slice(0, 3)), ("left", slice(3, 6))):
            payload = side_payload[side]
            if not payload:
                continue
            forces_raw = np.stack([p["force"][frame] for p in payload], axis=0)
            moments_raw = np.stack([p["moment"][frame] for p in payload], axis=0)
            cops_raw = np.stack([p["cop"][frame] for p in payload], axis=0)
            weights = np.asarray([max(float(p["force"][frame, 1]), 0.0) for p in payload], dtype=np.float64)
            grf[frame, sl] = raw_vector_to_output(forces_raw, rotation).sum(axis=0)
            grm[frame, sl] = raw_vector_to_output(moments_raw, rotation).sum(axis=0)
            if weights.sum() > 0:
                weighted_cop_raw = np.average(cops_raw, axis=0, weights=weights)
                cop[frame, sl] = raw_point_to_output(weighted_cop_raw[None, :], rotation, translation_mm)[0]
    return grf, cop, grm, stats


def write_force_mot(path: Path, time: np.ndarray, grf: np.ndarray, cop: np.ndarray, grm: np.ndarray) -> None:
    cols = [
        "time",
        "R_ground_force_vx", "R_ground_force_vy", "R_ground_force_vz",
        "R_ground_force_px", "R_ground_force_py", "R_ground_force_pz",
        "R_ground_torque_x", "R_ground_torque_y", "R_ground_torque_z",
        "L_ground_force_vx", "L_ground_force_vy", "L_ground_force_vz",
        "L_ground_force_px", "L_ground_force_py", "L_ground_force_pz",
        "L_ground_torque_x", "L_ground_torque_y", "L_ground_torque_z",
    ]
    data = np.column_stack([time, grf[:, :3], cop[:, :3], grm[:, :3], grf[:, 3:], cop[:, 3:], grm[:, 3:]])
    with path.open("w") as f:
        f.write("coordinates\n")
        f.write("version=1\n")
        f.write(f"nRows={data.shape[0]}\n")
        f.write(f"nColumns={data.shape[1]}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write("\t".join(cols) + "\n")
        for row in data:
            f.write("\t".join(f"{value:.10g}" for value in row) + "\n")


def process_trial(
    motion_dir: Path,
    preaddbio_root: Path,
    force_threshold_n: float,
    max_force_n: float,
    cop_margin_m: float,
    baseline_seconds: float,
    baseline_lowest_fraction: float,
    ambiguity_margin_m: float,
    max_ambiguous_samples: int,
    max_transform_rms_mm: float,
) -> dict[str, Any]:
    subject = motion_dir.parent.parent.name
    trial_match = TRIAL_RE.match(motion_dir.parent.name)
    if not trial_match:
        raise ValueError(f"Unexpected trial folder: {motion_dir.parent}")
    trial_num = int(trial_match.group(1))
    preaddbio_subject_dir = preaddbio_root / subject_to_preaddbio_name(subject)
    c3d_path, trc_path = trial_to_c3d_and_trc(preaddbio_subject_dir, trial_num)

    extracted = load_c3d_plate_data(
        c3d_path=c3d_path,
        trc_path=trc_path,
        force_threshold_n=force_threshold_n,
        max_force_n=max_force_n,
        cop_margin_m=cop_margin_m,
        baseline_seconds=baseline_seconds,
        baseline_lowest_fraction=baseline_lowest_fraction,
        max_transform_rms_mm=max_transform_rms_mm,
    )
    grf, cop, grm, stats = combine_platforms_by_foot(extracted, ambiguity_margin_m, max_ambiguous_samples)
    time = np.arange(grf.shape[0], dtype=np.float64) / float(extracted["point_rate"])

    np.save(motion_dir / "GRF.npy", grf.astype(np.float32))
    np.save(motion_dir / "COP.npy", cop.astype(np.float32))
    np.save(motion_dir / "GRM.npy", grm.astype(np.float32))
    np.save(motion_dir / "Time.npy", time)

    raw_dir = motion_dir / "Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mot_path = raw_dir / f"trial{trial_num}_forces_COP.mot"
    write_force_mot(mot_path, time, grf, cop, grm)

    return {
        "trial": f"{subject}/{motion_dir.parent.name}",
        "c3d": str(c3d_path),
        "trc": str(trc_path),
        "frames": int(grf.shape[0]),
        "analog_rate": extracted["analog_rate"],
        "point_rate": extracted["point_rate"],
        "rate_ratio": extracted["ratio"],
        "valid_platform_indices": extracted["valid_platform_indices"],
        "used_transform": extracted["used_transform"],
        "rotation_c3d_to_trc": extracted["rotation_c3d_to_trc"].tolist(),
        "translation_c3d_to_trc_mm": extracted["translation_c3d_to_trc_mm"].tolist(),
        "transform_rms_error_mm": extracted["transform_rms_error_mm"],
        "transform_samples": extracted["transform_samples"],
        "transform_error": extracted["transform_error"],
        "platforms": [
            {
                "platform_index": int(platform["platform_index"]),
                "active_frames": int(platform["active_frames"]),
                "sign_multiplier": int(platform["sign_multiplier"]),
                "force_offset_n": platform["force_offset_n"],
                "moment_offset_nm": platform["moment_offset_nm"],
                "baseline_samples": int(platform["baseline_samples"]),
                "bounds_xz_m": platform["bounds_xz_m"],
                "raw_force_minmax_n": platform["raw_force_minmax_n"],
                "corrected_force_minmax_n": platform["corrected_force_minmax_n"],
            }
            for platform in extracted["platforms"]
        ],
        "max_abs_grf_n": float(np.nanmax(np.abs(grf))) if grf.size else 0.0,
        "max_abs_cop_m": float(np.nanmax(np.abs(cop))) if cop.size else 0.0,
        "max_abs_grm_nm": float(np.nanmax(np.abs(grm))) if grm.size else 0.0,
        "right_vertical_max_n": float(np.nanmax(grf[:, 1])) if grf.size else 0.0,
        "left_vertical_max_n": float(np.nanmax(grf[:, 4])) if grf.size else 0.0,
        **stats,
    }


def discover_motion_dirs(dataset_root: Path, only: list[str] | None, limit: int | None) -> list[Path]:
    wanted = set(only or [])
    motion_dirs = []
    for motion_dir in sorted(dataset_root.glob("*/Trial_*/Motion")):
        label = f"{motion_dir.parent.parent.name}/{motion_dir.parent.name}"
        if wanted and label not in wanted:
            continue
        motion_dirs.append(motion_dir)
    if limit is not None:
        motion_dirs = motion_dirs[:limit]
    return motion_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("PD_Dataset"))
    parser.add_argument("--preaddbio-root", type=Path, default=Path("NAS_Data/Users/DMagruder/DATA/Boari_preAddBio"))
    parser.add_argument("--force-threshold-n", type=float, default=20.0)
    parser.add_argument("--max-force-n", type=float, default=5000.0)
    parser.add_argument("--cop-margin-m", type=float, default=0.20)
    parser.add_argument("--baseline-seconds", type=float, default=0.50)
    parser.add_argument("--baseline-lowest-fraction", type=float, default=0.10)
    parser.add_argument("--ambiguity-margin-m", type=float, default=0.10)
    parser.add_argument("--max-ambiguous-samples", type=int, default=20)
    parser.add_argument("--max-transform-rms-mm", type=float, default=75.0)
    parser.add_argument("--processdata-subset", action="store_true", help="Process 20 evenly distributed PD subject/trial combinations.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional labels like PD_SUB01_off/Trial_1.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    only = list(args.only or [])
    if args.processdata_subset:
        only = list(PROCESSDATA_TRIAL_SUBSET) + only
    motion_dirs = discover_motion_dirs(args.dataset_root, only, args.limit)

    manifest: dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "preaddbio_root": str(args.preaddbio_root.resolve()),
        "force_threshold_n": args.force_threshold_n,
        "max_force_n": args.max_force_n,
        "cop_margin_m": args.cop_margin_m,
        "baseline_seconds": args.baseline_seconds,
        "baseline_lowest_fraction": args.baseline_lowest_fraction,
        "ambiguity_margin_m": args.ambiguity_margin_m,
        "max_transform_rms_mm": args.max_transform_rms_mm,
        "processdata_subset": bool(args.processdata_subset),
        "only": only,
        "method": (
            "Plate-first Boari extraction using repaired FORCE_PLATFORM metadata. Forces are baseline-corrected, "
            "per-platform vertical signs are normalized so +Y is upward, active COP samples are filtered against "
            "plate bounds, active platforms are assigned to nearest raw C3D foot marker centroids, and outputs are "
            "rotated into the source walk*.trc frame when the fitted transform is reliable."
        ),
        "major_assumptions": [
            "Boari C3D force data are stored by force plate, not by foot.",
            "Boari vertical force is the C3D/platform Y component after ezc3d extraction.",
            "The declared seventh force platform has invalid zero-channel metadata and should be removed.",
            "Raw C3D marker/COP coordinates are used for foot assignment.",
            "Final force outputs should share the copied walk*.trc/AddBiomechanics kinematic frame whenever transform RMS is acceptable.",
            "Cross-plate contacts are summed per foot.",
        ],
        "output_columns": {
            "GRF.npy": ["R_x", "R_y_vertical", "R_z", "L_x", "L_y_vertical", "L_z"],
            "COP.npy": ["R_x", "R_y", "R_z", "L_x", "L_y", "L_z"],
            "GRM.npy": ["R_x", "R_y", "R_z", "L_x", "L_y", "L_z"],
        },
        "trials_seen": len(motion_dirs),
        "trials_written": 0,
        "failures": [],
        "trials": [],
        "ambiguous_trials": [],
        "transform_fallback_trials": [],
    }

    for idx, motion_dir in enumerate(motion_dirs, start=1):
        try:
            result = process_trial(
                motion_dir=motion_dir,
                preaddbio_root=args.preaddbio_root,
                force_threshold_n=args.force_threshold_n,
                max_force_n=args.max_force_n,
                cop_margin_m=args.cop_margin_m,
                baseline_seconds=args.baseline_seconds,
                baseline_lowest_fraction=args.baseline_lowest_fraction,
                ambiguity_margin_m=args.ambiguity_margin_m,
                max_ambiguous_samples=args.max_ambiguous_samples,
                max_transform_rms_mm=args.max_transform_rms_mm,
            )
            manifest["trials_written"] += 1
            manifest["trials"].append(result)
            if result["ambiguous_platform_frames"]:
                manifest["ambiguous_trials"].append({
                    "trial": result["trial"],
                    "ambiguous_platform_frames": result["ambiguous_platform_frames"],
                    "examples": result["ambiguous_examples"],
                })
            if not result["used_transform"]:
                manifest["transform_fallback_trials"].append({
                    "trial": result["trial"],
                    "transform_rms_error_mm": result["transform_rms_error_mm"],
                    "transform_error": result["transform_error"],
                })
        except Exception as exc:
            manifest["failures"].append({"trial": str(motion_dir), "error": str(exc)})
        if idx % 50 == 0:
            print(f"processed {idx}/{len(motion_dirs)}", flush=True)

    manifest_path = args.dataset_root / "boari_c3d_force_extraction_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "trials_seen": manifest["trials_seen"],
        "trials_written": manifest["trials_written"],
        "failures": len(manifest["failures"]),
        "ambiguous_trials": len(manifest["ambiguous_trials"]),
        "transform_fallback_trials": len(manifest["transform_fallback_trials"]),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
