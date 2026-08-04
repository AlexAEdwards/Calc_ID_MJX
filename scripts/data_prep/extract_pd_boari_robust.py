#!/usr/bin/env python3
"""Versioned, QC-first extraction of Boari C3D forces and PD kinematics.

Outputs are written below Motion/RobustExtracted_v2 and never replace the
existing Motion arrays.  A trial is training-ready only when qc_status=PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ezc3d
import numpy as np
from scipy.signal import butter, sosfiltfilt
from paths import artifact, dataset  # noqa: E402


VERSION = "2.0.2"
TRIAL_RE = re.compile(r"Trial_(\d+)$")
SEGMENT_RE = re.compile(r"_segment_(\d+)_ik\.mot$")
RIGHT_MARKERS = ("R.Heel", "R.MT1", "R.MT2", "R.MT5", "R.Ankle")
LEFT_MARKERS = ("L.Heel", "L.MT1", "L.MT2", "L.MT5", "L.Ankle")
KINEMATIC_23 = (
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)
TRANSLATIONS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
FORCE_COLUMNS = (
    "time",
    "R_ground_force_vx", "R_ground_force_vy", "R_ground_force_vz",
    "R_ground_force_px", "R_ground_force_py", "R_ground_force_pz",
    "R_ground_torque_x", "R_ground_torque_y", "R_ground_torque_z",
    "L_ground_force_vx", "L_ground_force_vy", "L_ground_force_vz",
    "L_ground_force_px", "L_ground_force_py", "L_ground_force_pz",
    "L_ground_torque_x", "L_ground_torque_y", "L_ground_torque_z",
)


@dataclass
class Config:
    force_cutoff_hz: float = 20.0
    kinematics_cutoff_hz: float = 6.0
    force_on_n: float = 30.0
    force_off_n: float = 15.0
    noise_on_sigma: float = 8.0
    noise_off_sigma: float = 4.0
    min_contact_s: float = 0.040
    bridge_gap_s: float = 0.020
    plate_margin_m: float = 0.05
    max_cop_foot_distance_m: float = 0.35
    side_margin_m: float = 0.05
    min_side_consistency: float = 0.55
    review_excluded_force_n: float = 50.0
    reject_excluded_force_n: float = 100.0
    max_transform_rms_mm: float = 10.0
    max_force_n: float = 5000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mot(path: Path) -> tuple[np.ndarray, list[str], bool]:
    lines = path.read_text(errors="replace").splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None)
    if end is None or end + 1 >= len(lines):
        raise ValueError(f"Invalid MOT header: {path}")
    in_degrees = False
    for line in lines[:end]:
        if line.lower().startswith("indegrees="):
            in_degrees = line.split("=", 1)[1].strip().lower() == "yes"
    columns = lines[end + 1].split()
    rows = [[float(value) for value in line.split()] for line in lines[end + 2:] if line.strip()]
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"MOT data/header mismatch: {path}")
    return data, columns, in_degrees


def read_trc(path: Path) -> dict[str, Any]:
    lines = path.read_text(errors="replace").splitlines()
    header_i = next((i for i, line in enumerate(lines) if line.startswith("Frame#")), None)
    if header_i is None:
        raise ValueError(f"No Frame# header in {path}")
    header = lines[header_i].split("\t")
    marker_entries = []
    seen = set()
    for col in range(2, len(header), 3):
        name = header[col].strip()
        if name and name not in seen:
            marker_entries.append((name, col))
            seen.add(name)
    frames, times, marker_rows = [], [], {name: [] for name, _ in marker_entries}
    for line in lines[header_i + 2:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        frames.append(int(float(fields[0])))
        times.append(float(fields[1]))
        for name, col in marker_entries:
            try:
                marker_rows[name].append([float(fields[col]), float(fields[col + 1]), float(fields[col + 2])])
            except (IndexError, ValueError):
                marker_rows[name].append([np.nan, np.nan, np.nan])
    return {
        "frames": np.asarray(frames, dtype=np.int64),
        "time_original": np.asarray(times, dtype=np.float64),
        "markers_mm": {name: np.asarray(values, dtype=np.float64) for name, values in marker_rows.items()},
    }


def load_kinematics(raw_dir: Path, expected_length: int) -> tuple[np.ndarray, list[str], list[str]]:
    files = sorted(
        raw_dir.glob("*_segment_*_ik.mot"),
        key=lambda path: int(SEGMENT_RE.search(path.name).group(1)) if SEGMENT_RE.search(path.name) else -1,
    )
    if not files:
        raise FileNotFoundError(f"No IK MOT files in {raw_dir}")
    arrays, reference_columns, in_degrees = [], None, None
    for path in files:
        data, columns, degrees = read_mot(path)
        if reference_columns is None:
            reference_columns, in_degrees = columns, degrees
        if columns != reference_columns or degrees != in_degrees:
            raise ValueError(f"Inconsistent IK schema in {raw_dir}")
        arrays.append(data[:, 1:])
    full = np.concatenate(arrays, axis=0)
    if len(full) != expected_length:
        raise ValueError(f"IK/TRC length mismatch: {len(full)} vs {expected_length}")
    if in_degrees:
        for index, name in enumerate(reference_columns[1:]):
            if name not in TRANSLATIONS:
                full[:, index] = np.deg2rad(full[:, index])
    return full, reference_columns[1:], [path.name for path in files]


def marker_centroid(points_mm: np.ndarray, labels: list[str], wanted: tuple[str, ...]) -> np.ndarray:
    label_index = {label.split(":")[-1]: i for i, label in enumerate(labels)}
    values = []
    for name in wanted:
        if name in label_index:
            xyz = points_mm[:3, label_index[name], :].T.astype(np.float64)
            xyz[~np.isfinite(xyz)] = np.nan
            values.append(xyz)
    if not values:
        return np.full((points_mm.shape[2], 3), np.nan)
    return np.nanmean(np.stack(values), axis=0)


def marker_alignment_error(
    indices: np.ndarray, points_mm: np.ndarray, labels: list[str], trc_markers: dict[str, np.ndarray]
) -> float:
    label_index = {label.split(":")[-1]: i for i, label in enumerate(labels)}
    errors = []
    stride = max(1, len(indices) // 100)
    for name in sorted(set(label_index).intersection(trc_markers)):
        source = points_mm[:3, label_index[name], :].T[indices[::stride]]
        target = trc_markers[name][::stride]
        valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        if np.any(valid):
            errors.append(np.linalg.norm(source[valid] - target[valid], axis=1))
    return float(np.median(np.concatenate(errors))) if errors else math.inf


def align_trc_to_c3d(
    trc: dict[str, Any], points_mm: np.ndarray, labels: list[str], point_rate: float, first_frame: int
) -> tuple[np.ndarray, dict[str, Any]]:
    n_c3d = points_mm.shape[2]
    time_indices = np.rint(trc["time_original"] * point_rate).astype(int) - first_frame
    frame_indices = trc["frames"].astype(int) - 1 - first_frame
    candidates = []
    for base_name, base in (("time", time_indices), ("frame", frame_indices)):
        for shift in range(-2, 3):
            indices = base + shift
            if len(indices) and indices[0] >= 0 and indices[-1] < n_c3d and np.all(np.diff(indices) > 0):
                error = marker_alignment_error(indices, points_mm, labels, trc["markers_mm"])
                candidates.append((error, base_name, shift, indices))
    if not candidates:
        raise ValueError(
            f"TRC interval cannot map into C3D: time indices {time_indices[0]}..{time_indices[-1]}, "
            f"C3D frames={n_c3d}"
        )
    error, method, shift, indices = min(candidates, key=lambda item: item[0])
    return indices, {"method": method, "shift_frames": shift, "median_marker_error_mm": error}


def rigid_transform_aligned(
    indices: np.ndarray, points_mm: np.ndarray, labels: list[str], trc_markers: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, float, int]:
    label_index = {label.split(":")[-1]: i for i, label in enumerate(labels)}
    source_rows, target_rows = [], []
    for name in sorted(set(label_index).intersection(trc_markers)):
        source = points_mm[:3, label_index[name], :].T[indices]
        target = trc_markers[name]
        valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        if np.any(valid):
            source_rows.append(source[valid])
            target_rows.append(target[valid])
    if not source_rows:
        raise ValueError("No shared finite C3D/TRC markers")
    source, target = np.vstack(source_rows), np.vstack(target_rows)
    sm, tm = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - sm).T @ (target - tm))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = tm - sm @ rotation
    residual = source @ rotation + translation - target
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return rotation, translation, rms, len(source)


def repair_platform_metadata(c3d: ezc3d.c3d) -> tuple[list[int], dict[str, Any]]:
    fp = c3d["parameters"].get("FORCE_PLATFORM")
    if not fp:
        raise ValueError("No FORCE_PLATFORM metadata")
    channels = np.asarray(fp["CHANNEL"]["value"])
    types = np.asarray(fp["TYPE"]["value"]).astype(int)
    valid = []
    for index, platform_type in enumerate(types):
        required = 8 if platform_type == 3 else 6
        if np.all(channels[:required, index] > 0):
            valid.append(index)
    original_used = int(np.asarray(fp["USED"]["value"]).ravel()[0])
    fp["USED"]["value"] = np.asarray([len(valid)], dtype=int)
    for key in ("TYPE", "CORNERS", "ORIGIN", "CHANNEL", "CAL_MATRIX"):
        values = np.asarray(fp[key]["value"])
        if values.ndim and values.shape[-1] == len(types):
            fp[key]["value"] = values[..., valid]
    return valid, {"declared_used": original_used, "types": types.tolist(), "valid_indices": valid}


def robust_baseline(force: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    vertical = force[:, 1]
    finite = np.isfinite(vertical)
    if finite.sum() < 20:
        raise ValueError("Too few finite force samples")
    values = vertical[finite]
    q25, q75 = np.percentile(values, [25, 75])
    width = max((q75 - q25) / 8.0, 2.0)
    bins = int(np.clip((values.max() - values.min()) / width, 32, 256))
    hist, edges = np.histogram(values, bins=bins)
    center = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2.0
    derivative = np.abs(np.gradient(vertical))
    band = max(5.0, (edges[1] - edges[0]) * 1.5)
    quiet = finite & (np.abs(vertical - center) <= band)
    quiet &= derivative <= np.nanpercentile(derivative[finite], 70)
    if quiet.sum() < 20:
        quiet = finite & (np.abs(vertical - center) <= band * 2)
    offset = np.nanmedian(force[quiet], axis=0)
    corrected_y = vertical - offset[1]
    pos = np.nanpercentile(corrected_y, 99.5)
    neg = -np.nanpercentile(corrected_y, 0.5)
    sign = 1.0 if pos >= neg else -1.0
    noise = 1.4826 * np.nanmedian(np.abs(corrected_y[quiet] - np.nanmedian(corrected_y[quiet])))
    return offset, quiet, float(sign), float(max(noise, 0.5))


def lowpass(values: np.ndarray, rate: float, cutoff: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 15 or cutoff <= 0 or cutoff >= rate / 2:
        return values.copy()
    sos = butter(4, cutoff, btype="lowpass", fs=rate, output="sos")
    return sosfiltfilt(sos, values, axis=0)


def interpolate_nonfinite(values: np.ndarray, default: float = 0.0) -> tuple[np.ndarray, list[int]]:
    """Linearly fill isolated nonfinite samples; report wholly missing columns."""
    result = np.asarray(values, dtype=np.float64).copy()
    missing_columns = []
    x = np.arange(len(result))
    for column in range(result.shape[1]):
        finite = np.isfinite(result[:, column])
        if not np.any(finite):
            result[:, column] = default
            missing_columns.append(column)
        elif not np.all(finite):
            result[:, column] = np.interp(x, x[finite], result[finite, column])
    return result, missing_columns


def downsample(values: np.ndarray, ratio: int, mode: str = "mean", weights: np.ndarray | None = None) -> np.ndarray:
    n = (len(values) // ratio) * ratio
    blocks = values[:n].reshape(-1, ratio, *values.shape[1:])
    if mode == "weighted":
        if weights is None:
            raise ValueError("weights required")
        wb = weights[:n].reshape(-1, ratio)
        denom = wb.sum(axis=1)
        result = np.full((len(blocks), *values.shape[1:]), np.nan)
        valid = denom > 0
        result[valid] = (blocks[valid] * wb[valid][..., None]).sum(axis=1) / denom[valid, None]
        return result
    return np.nanmean(blocks, axis=1)


def hysteresis_contacts(vertical: np.ndarray, on: float, off: float, rate: float, min_s: float, gap_s: float) -> np.ndarray:
    active = np.zeros(len(vertical), dtype=bool)
    state = False
    for i, value in enumerate(vertical):
        if not state and value >= on:
            state = True
        elif state and value < off:
            state = False
        active[i] = state
    max_gap = int(round(gap_s * rate))
    if max_gap:
        false_runs = runs(~active)
        for start, end in false_runs:
            if start > 0 and end < len(active) and end - start <= max_gap:
                active[start:end] = True
    min_len = max(1, int(round(min_s * rate)))
    for start, end in runs(active):
        if end - start < min_len:
            active[start:end] = False
    return active


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def load_platforms(c3d_path: Path, cfg: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = ezc3d.c3d(str(c3d_path), extract_forceplat_data=False)
    valid, metadata = repair_platform_metadata(base)
    with tempfile.NamedTemporaryFile(suffix=".c3d") as temp:
        base.write(temp.name)
        c3d = ezc3d.c3d(temp.name, extract_forceplat_data=True)
    analog_rate = float(c3d["header"]["analogs"]["frame_rate"])
    point_rate = float(c3d["header"]["points"]["frame_rate"])
    ratio = int(round(analog_rate / point_rate))
    if not np.isclose(analog_rate, point_rate * ratio):
        raise ValueError(f"Nonintegral analog/point rate: {analog_rate}/{point_rate}")
    platforms = []
    for number, raw in enumerate(c3d["data"]["platform"], start=1):
        force = np.asarray(raw["force"], dtype=np.float64).T
        moment = np.asarray(raw["moment"], dtype=np.float64).T / 1000.0
        cop = np.asarray(raw["center_of_pressure"], dtype=np.float64).T / 1000.0
        free = np.asarray(raw.get("Tz", np.zeros_like(raw["force"])), dtype=np.float64).T / 1000.0
        n = min(len(force), len(moment), len(cop), len(free))
        force, moment, cop, free = force[:n], moment[:n], cop[:n], free[:n]
        force, missing_force_columns = interpolate_nonfinite(force)
        moment, missing_moment_columns = interpolate_nonfinite(moment)
        free, missing_free_columns = interpolate_nonfinite(free)
        if missing_force_columns:
            raise ValueError(f"Platform {number} has missing force columns {missing_force_columns}")
        offset, quiet, sign, noise = robust_baseline(force)
        moment_offset = np.nanmedian(moment[quiet], axis=0)
        free_offset = np.nanmedian(free[quiet], axis=0)
        force = lowpass((force - offset) * sign, analog_rate, cfg.force_cutoff_hz)
        moment = lowpass((moment - moment_offset) * sign, analog_rate, cfg.force_cutoff_hz)
        free = lowpass((free - free_offset) * sign, analog_rate, cfg.force_cutoff_hz)
        force_ds = downsample(force, ratio)
        moment_ds = downsample(moment, ratio)
        free_ds = downsample(free, ratio)
        weights = np.maximum(force[:, 1], 0.0)
        cop_ds = downsample(cop, ratio, mode="weighted", weights=weights)
        corners = np.asarray(raw["corners"], dtype=np.float64) / 1000.0
        xmin, xmax = np.nanmin(corners[0]) - cfg.plate_margin_m, np.nanmax(corners[0]) + cfg.plate_margin_m
        zmin, zmax = np.nanmin(corners[2]) - cfg.plate_margin_m, np.nanmax(corners[2]) + cfg.plate_margin_m
        on = max(cfg.force_on_n, cfg.noise_on_sigma * noise)
        off = max(cfg.force_off_n, cfg.noise_off_sigma * noise)
        contact = hysteresis_contacts(force_ds[:, 1], on, off, point_rate, cfg.min_contact_s, cfg.bridge_gap_s)
        contact &= np.linalg.norm(force_ds, axis=1) <= cfg.max_force_n
        contact &= np.isfinite(cop_ds).all(axis=1)
        contact &= (cop_ds[:, 0] >= xmin) & (cop_ds[:, 0] <= xmax)
        contact &= (cop_ds[:, 2] >= zmin) & (cop_ds[:, 2] <= zmax)
        platforms.append({
            "number": number, "force": force_ds, "moment_origin": moment_ds, "free_moment": free_ds,
            "cop": cop_ds, "contact": contact, "noise_n": noise, "threshold_on_n": on,
            "threshold_off_n": off, "force_offset_n": offset.tolist(), "sign": int(sign),
            "bounds_xz_m": [xmin, xmax, zmin, zmax],
            "missing_moment_columns_filled": missing_moment_columns,
            "missing_free_moment_columns_filled": missing_free_columns,
        })
    context = {
        "base": c3d, "valid_indices": valid, "platform_metadata": metadata,
        "analog_rate": analog_rate, "point_rate": point_rate, "ratio": ratio,
    }
    return context, platforms


def foot_centroid_from_trc(trc: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    arrays = [trc["markers_mm"][name] / 1000.0 for name in names if name in trc["markers_mm"]]
    if not arrays:
        return np.full((len(trc["frames"]), 3), np.nan)
    return np.nanmean(np.stack(arrays), axis=0)


def assign_events(
    platforms: list[dict[str, Any]], indices: np.ndarray, right: np.ndarray, left: np.ndarray,
    time: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(indices)
    grf, cop, grm = np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 6))
    mask = np.zeros((n, 2), dtype=bool)
    confidence = np.zeros((n, 2), dtype=np.float32)
    payload: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]] = {}
    events = []
    for platform in platforms:
        valid_indices = indices[indices < len(platform["contact"])]
        if len(valid_indices) != n:
            raise ValueError("Aligned point indices exceed platform data")
        contact = platform["contact"][indices]
        for start, end in runs(contact):
            local = np.arange(start, end)
            source = indices[local]
            plate_cop = platform["cop"][source]
            plate_force = platform["force"][source]
            plate_free = platform["free_moment"][source]
            dr = np.linalg.norm(plate_cop[:, [0, 2]] - right[local][:, [0, 2]], axis=1)
            dl = np.linalg.norm(plate_cop[:, [0, 2]] - left[local][:, [0, 2]], axis=1)
            finite = np.isfinite(dr) & np.isfinite(dl)
            peak = float(np.nanmax(plate_force[:, 1]))
            event = {
                "platform": platform["number"], "start": int(start), "end": int(end),
                "start_time_s": float(time[start]), "end_time_s": float(time[end - 1]),
                "peak_vertical_n": peak,
            }
            if finite.sum() < max(3, len(local) // 2):
                event.update({"assignment": "unassigned", "reason": "missing_foot_markers"})
                events.append(event)
                continue
            med_r, med_l = float(np.median(dr[finite])), float(np.median(dl[finite]))
            side = 0 if med_r <= med_l else 1
            best, other = (med_r, med_l) if side == 0 else (med_l, med_r)
            nearest_side = np.where(dr <= dl, 0, 1)
            consistency = float(np.mean(nearest_side[finite] == side))
            margin = other - best
            event.update({
                "median_right_distance_m": med_r, "median_left_distance_m": med_l,
                "distance_margin_m": margin, "side_consistency": consistency,
            })
            if best > cfg.max_cop_foot_distance_m:
                event.update({"assignment": "excluded", "reason": "cop_far_from_both_feet"})
                events.append(event)
                continue
            if margin < cfg.side_margin_m or consistency < cfg.min_side_consistency:
                event.update({"assignment": "ambiguous", "reason": "insufficient_side_separation"})
                events.append(event)
                continue
            side_name = "right" if side == 0 else "left"
            event.update({"assignment": side_name, "reason": "accepted"})
            events.append(event)
            for j, frame in enumerate(local):
                payload.setdefault((frame, side), []).append(
                    (plate_force[j], plate_cop[j], plate_free[j], margin)
                )
    for (frame, side), values in payload.items():
        sl = slice(0, 3) if side == 0 else slice(3, 6)
        forces = np.stack([item[0] for item in values])
        cops = np.stack([item[1] for item in values])
        free = np.stack([item[2] for item in values])
        weights = np.maximum(forces[:, 1], 0.0)
        grf[frame, sl] = forces.sum(axis=0)
        if weights.sum() > 0:
            combined_cop = np.average(cops, axis=0, weights=weights)
            cop[frame, sl] = combined_cop
        # ezc3d Tz is the free moment at COP; summing free moments is valid.
        grm[frame, sl] = free.sum(axis=0)
        mask[frame, side] = True
        confidence[frame, side] = min(item[3] for item in values)
    return grf, cop, grm, mask, confidence, events


def transform_outputs(
    grf: np.ndarray, cop: np.ndarray, grm: np.ndarray, mask: np.ndarray,
    rotation: np.ndarray, translation_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for side, sl in ((0, slice(0, 3)), (1, slice(3, 6))):
        grf[:, sl] = grf[:, sl] @ rotation
        grm[:, sl] = grm[:, sl] @ rotation
        active = mask[:, side]
        cop[active, sl] = ((cop[active, sl] * 1000.0) @ rotation + translation_mm) / 1000.0
    return grf, cop, grm


def filter_kinematics(full: np.ndarray, columns: list[str], time: np.ndarray, cutoff: float) -> np.ndarray:
    result = full.copy()
    rate = 1.0 / float(np.median(np.diff(time)))
    for index, name in enumerate(columns):
        values = result[:, index]
        if name not in TRANSLATIONS:
            values = np.unwrap(values)
        result[:, index] = lowpass(values, rate, cutoff)
    return result


def write_mot(path: Path, time: np.ndarray, values: np.ndarray, columns: list[str], in_degrees: bool = False) -> None:
    data = np.column_stack([time, values])
    with path.open("w") as handle:
        handle.write(f"{path.stem}\nversion=1\nnRows={len(data)}\nnColumns={data.shape[1]}\n")
        handle.write(f"inDegrees={'yes' if in_degrees else 'no'}\nendheader\n")
        handle.write("\t".join(["time", *columns]) + "\n")
        for row in data:
            handle.write("\t".join(f"{value:.10g}" for value in row) + "\n")


def write_force_mot(path: Path, time: np.ndarray, grf: np.ndarray, cop: np.ndarray, grm: np.ndarray) -> None:
    values = np.column_stack([grf[:, :3], cop[:, :3], grm[:, :3], grf[:, 3:], cop[:, 3:], grm[:, 3:]])
    write_mot(path, time, values, list(FORCE_COLUMNS[1:]), in_degrees=False)


def qc_status(
    events: list[dict[str, Any]], grf: np.ndarray, transform_rms: float, cfg: Config
) -> tuple[str, list[str]]:
    reasons = []
    if transform_rms > cfg.max_transform_rms_mm:
        reasons.append(f"transform_rms_mm={transform_rms:.3f}")
    accepted = [event for event in events if event["assignment"] in {"right", "left"}]
    excluded = [event for event in events if event["assignment"] not in {"right", "left"}]
    max_excluded = max((event["peak_vertical_n"] for event in excluded), default=0.0)
    if not accepted:
        reasons.append("no_accepted_contacts")
    accepted_sides = {event["assignment"] for event in accepted}
    if accepted and accepted_sides != {"right", "left"}:
        reasons.append("contacts_for_only_one_foot")
    if max_excluded >= cfg.review_excluded_force_n:
        reasons.append(f"excluded_contact_peak_n={max_excluded:.3f}")
    if np.min(grf[:, [1, 4]]) < -1e-3:
        reasons.append("negative_vertical_force_after_transform")
    if transform_rms > cfg.max_transform_rms_mm or not accepted:
        return "REJECT", reasons
    if max_excluded >= cfg.review_excluded_force_n or accepted_sides != {"right", "left"}:
        return "REVIEW", reasons
    return "PASS", reasons


def process_trial(
    motion_dir: Path, source_root: Path, output_name: str, cfg: Config,
    overwrite: bool, hash_sources: bool
) -> dict[str, Any]:
    trial_dir, subject_dir = motion_dir.parent, motion_dir.parent.parent
    match = TRIAL_RE.match(trial_dir.name)
    if not match:
        raise ValueError(f"Unexpected trial name: {trial_dir}")
    trial_number = int(match.group(1))
    source_subject = source_root / subject_dir.name.removeprefix("PD_")
    trcs = sorted(source_subject.glob("walk*.trc"))
    if not 1 <= trial_number <= len(trcs):
        raise ValueError(f"No source TRC for {subject_dir.name}/{trial_dir.name}")
    trc_path = trcs[trial_number - 1]
    walk_number = trc_path.stem.removeprefix("walk")
    c3d_path = source_subject / f"{source_subject.name}_walk_{walk_number}.c3d"
    output = motion_dir / output_name
    if output.exists() and not overwrite:
        metadata_path = output / "extraction_metadata.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "version": VERSION, "trial": f"{subject_dir.name}/{trial_dir.name}",
        "walk_id": f"walk{walk_number}", "source_c3d": str(c3d_path),
        "source_trc": str(trc_path), "output_dir": str(output), "assumptions": [
            "TRC times or frame numbers identify original C3D point frames.",
            "The six force platforms with valid nonzero channels are calibrated.",
            "C3D Y is the vertical force component before the fitted rigid transform.",
            "High-confidence COP should remain within 0.35 m of one foot centroid.",
            "A single plate contact belongs to at most one foot.",
            "ezc3d Tz is the free moment at the reported COP.",
            "Existing IK coordinates are authoritative when their row count matches the TRC.",
        ],
    }
    if hash_sources:
        metadata["source_sha256"] = {"c3d": sha256(c3d_path), "trc": sha256(trc_path)}
    try:
        trc = read_trc(trc_path)
        full_raw, full_columns, ik_files = load_kinematics(motion_dir / "Raw", len(trc["frames"]))
        if hash_sources:
            metadata["source_sha256"]["ik_mot"] = {
                name: sha256(motion_dir / "Raw" / name) for name in ik_files
            }
        context, platforms = load_platforms(c3d_path, cfg)
        c3d = context["base"]
        points = np.asarray(c3d["data"]["points"], dtype=np.float64)
        labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
        indices, alignment = align_trc_to_c3d(
            trc, points, labels, context["point_rate"], int(c3d["header"]["points"]["first_frame"])
        )
        rotation, translation, rms, samples = rigid_transform_aligned(indices, points, labels, trc["markers_mm"])
        time = trc["time_original"] - trc["time_original"][0]
        if np.any(np.diff(time) <= 0):
            raise ValueError("Nonmonotonic TRC time")
        right = foot_centroid_from_trc(trc, RIGHT_MARKERS)
        left = foot_centroid_from_trc(trc, LEFT_MARKERS)
        grf, cop, grm, mask, confidence, events = assign_events(platforms, indices, right, left, time, cfg)
        grf, cop, grm = transform_outputs(grf, cop, grm, mask, rotation, translation)
        full_filtered = filter_kinematics(full_raw, full_columns, time, cfg.kinematics_cutoff_hz)
        name_index = {name: i for i, name in enumerate(full_columns)}
        missing = [name for name in KINEMATIC_23 if name not in name_index]
        if missing:
            raise ValueError(f"Missing 23-DOF coordinates: {missing}")
        pos_raw = full_raw[:, [name_index[name] for name in KINEMATIC_23]]
        pos = full_filtered[:, [name_index[name] for name in KINEMATIC_23]]
        vel = np.gradient(pos, time, axis=0)
        accel = np.gradient(vel, time, axis=0)
        status, reasons = qc_status(events, grf, rms, cfg)
        pos_raw = pos_raw.astype(np.float32)
        pos = pos.astype(np.float32)
        vel = vel.astype(np.float32)
        accel = accel.astype(np.float32)
        grf = grf.astype(np.float32)
        cop = cop.astype(np.float32)
        grm = grm.astype(np.float32)
        confidence = confidence.astype(np.float32)
        arrays = {
            "Pos_raw": pos_raw, "Pos": pos, "Vel": vel, "Accel": accel, "GRF": grf,
            "COP": cop, "GRM": grm, "Time": time, "Time_for_pos": time,
            "ContactMask": mask, "ForceAssignmentConfidence": confidence,
        }
        for name, values in arrays.items():
            dtype = np.bool_ if values.dtype == np.bool_ else (np.float64 if name.startswith("Time") else np.float32)
            np.save(output / f"{name}.npy", values.astype(dtype))
        write_mot(output / "Kinematics_full.mot", time, full_filtered, full_columns)
        write_mot(output / "Kinematics_23dof.mot", time, pos, list(KINEMATIC_23))
        write_force_mot(output / "Forces_cleaned.mot", time, grf, cop, grm)
        platform_qc = [{
            key: value for key, value in platform.items()
            if key not in {"force", "moment_origin", "free_moment", "cop", "contact"}
        } | {"contact_frames_full_c3d": int(platform["contact"].sum())} for platform in platforms]
        metadata.update({
            "qc_status": status, "qc_reasons": reasons, "ik_files": ik_files,
            "kinematic_columns_full": full_columns, "kinematic_columns_23": list(KINEMATIC_23),
            "frames": len(time), "time_start_original_s": float(trc["time_original"][0]),
            "time_end_original_s": float(trc["time_original"][-1]),
            "point_rate_hz": context["point_rate"], "analog_rate_hz": context["analog_rate"],
            "rate_ratio": context["ratio"], "alignment": alignment,
            "transform": {"rotation": rotation.tolist(), "translation_mm": translation.tolist(),
                          "rms_error_mm": rms, "samples": samples},
            "force_platform_metadata": context["platform_metadata"], "platform_qc": platform_qc,
            "contact_events": events,
            "contact_summary": {
                "events_total": len(events),
                "events_right": sum(event["assignment"] == "right" for event in events),
                "events_left": sum(event["assignment"] == "left" for event in events),
                "events_excluded": sum(event["assignment"] not in {"right", "left"} for event in events),
                "max_vertical_grf_n": float(np.max(grf[:, [1, 4]])),
            },
            "filters": {"force_cutoff_hz": cfg.force_cutoff_hz,
                        "kinematics_cutoff_hz": cfg.kinematics_cutoff_hz},
        })
    except Exception as exc:
        metadata.update({"qc_status": "REJECT", "qc_reasons": [f"{type(exc).__name__}: {exc}"]})
    (output / "extraction_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def discover(dataset_root: Path, only: set[str], limit: int | None) -> list[Path]:
    dirs = []
    for motion in sorted(dataset_root.glob("PD_SUB*/Trial_*/Motion")):
        label = f"{motion.parent.parent.name}/{motion.parent.name}"
        if not only or label in only:
            dirs.append(motion)
    return dirs[:limit] if limit is not None else dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(str(dataset("Datasets_Local", "PD_Dataset"))))
    parser.add_argument("--source-root", type=Path, default=Path(str(dataset("Datasets_Local", "Boari_preAddBio"))))
    parser.add_argument("--output-name", default="RobustExtracted_v2")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-sources", action="store_true")
    args = parser.parse_args()
    cfg = Config()
    motion_dirs = discover(args.dataset_root, set(args.only), args.limit)
    manifest = {
        "version": VERSION, "dataset_root": str(args.dataset_root.resolve()),
        "source_root": str(args.source_root.resolve()), "output_name": args.output_name,
        "config": cfg.__dict__, "trials_seen": len(motion_dirs), "trials": [],
    }
    for index, motion in enumerate(motion_dirs, start=1):
        result = process_trial(motion, args.source_root, args.output_name, cfg, args.overwrite, args.hash_sources)
        manifest["trials"].append({
            "trial": result["trial"], "qc_status": result["qc_status"],
            "qc_reasons": result.get("qc_reasons", []),
            "output_dir": result["output_dir"],
        })
        if index % 25 == 0 or index == len(motion_dirs):
            print(f"processed {index}/{len(motion_dirs)}", flush=True)
    counts = {status: sum(item["qc_status"] == status for item in manifest["trials"])
              for status in ("PASS", "REVIEW", "REJECT")}
    manifest["qc_counts"] = counts
    manifest_path = args.dataset_root / f"{args.output_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "qc_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
