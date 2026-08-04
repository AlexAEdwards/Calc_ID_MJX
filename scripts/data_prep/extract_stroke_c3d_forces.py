#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ezc3d
import numpy as np


RIGHT_MARKERS = ("RHEE", "RTOE", "RANK", "RKNE")
LEFT_MARKERS = ("LHEE", "LTOE", "LANK", "LKNE")
ROTATION_MARKERS = (
    "C7", "T10", "CLAV", "STRN",
    "LASI", "RASI", "SACR",
    "LHEE", "LTOE", "LANK", "LKNE",
    "RHEE", "RTOE", "RANK", "RKNE",
)
PROCESSDATA_TRIAL_SUBSET = (
    "SUBJ09/Trial_2", "SUBJ10/Trial_1", "SUBJ100/Trial_1",
    "SUBJ100/Trial_3", "SUBJ119/Trial_2", "SUBJ120/Trial_2",
    "SUBJ123/Trial_1", "SUBJ125/Trial_3", "SUBJ131/Trial_3",
    "SUBJ25/Trial_3", "SUBJ32/Trial_2", "SUBJ33/Trial_3",
    "SUBJ36/Trial_3", "SUBJ40/Trial_4", "SUBJ50/Trial_3",
    "TVC03/Trial_3", "TVC04/Trial_3", "TVC36/Trial_2",
    "TVC53/Trial_1", "TVC60/Trial_2",
)


def clean_marker_label(label: str) -> str:
    return label.split(":")[-1]


def finite_minmax(values: np.ndarray) -> list[float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return [None, None]
    return [float(np.nanmin(finite)), float(np.nanmax(finite))]


def marker_centroid(points: np.ndarray, labels: list[str], markers: tuple[str, ...]) -> np.ndarray:
    label_to_idx = {clean_marker_label(label): idx for idx, label in enumerate(labels)}
    arrays = []
    for marker in markers:
        idx = label_to_idx.get(marker)
        if idx is None:
            continue
        xyz_mm = points[:3, idx, :].T.astype(np.float64)
        xyz_mm[~np.isfinite(xyz_mm)] = np.nan
        arrays.append(xyz_mm)
    if not arrays:
        return np.full((points.shape[2], 3), np.nan, dtype=np.float64)
    return np.nanmean(np.stack(arrays, axis=0), axis=0)


def interpolate_markers_to_analog(marker_xyz_mm: np.ndarray, n_analog: int) -> np.ndarray:
    if marker_xyz_mm.shape[0] == n_analog:
        return marker_xyz_mm
    src = np.linspace(0.0, 1.0, marker_xyz_mm.shape[0])
    dst = np.linspace(0.0, 1.0, n_analog)
    out = np.empty((n_analog, 3), dtype=np.float64)
    for axis in range(3):
        values = marker_xyz_mm[:, axis]
        valid = np.isfinite(values)
        if valid.sum() < 2:
            out[:, axis] = np.nan
        else:
            out[:, axis] = np.interp(dst, src[valid], values[valid])
    return out


def read_trc_markers(path: Path, markers: tuple[str, ...]) -> dict[str, np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Frame#"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No Frame# header found in {path}")

    header = lines[header_idx].rstrip("\n").split("\t")
    data_start = header_idx + 2
    marker_names = header[2:]
    marker_cols = {
        marker: 2 + marker_names.index(marker) * 3
        for marker in markers
        if marker in marker_names
    }
    data = {marker: [] for marker in marker_cols}
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        for marker, col in marker_cols.items():
            try:
                data[marker].append([float(parts[col]), float(parts[col + 1]), float(parts[col + 2])])
            except (IndexError, ValueError):
                data[marker].append([np.nan, np.nan, np.nan])
    return {marker: np.asarray(values, dtype=np.float64) for marker, values in data.items()}


def estimate_rotation_from_trc(points: np.ndarray, labels: list[str], trc_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    trc_markers = read_trc_markers(trc_path, ROTATION_MARKERS)
    label_to_idx = {clean_marker_label(label): idx for idx, label in enumerate(labels)}
    source_rows = []
    target_rows = []
    for marker in ROTATION_MARKERS:
        if marker not in label_to_idx or marker not in trc_markers:
            continue
        source = points[:3, label_to_idx[marker], :].T.astype(np.float64)
        target = trc_markers[marker]
        n = min(source.shape[0], target.shape[0])
        source = source[:n]
        target = target[:n]
        valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        if np.any(valid):
            source_rows.append(source[valid])
            target_rows.append(target[valid])
    if not source_rows:
        raise ValueError(f"No shared C3D/TRC marker samples for {trc_path}")

    source_all = np.vstack(source_rows)
    target_all = np.vstack(target_rows)
    if source_all.shape[0] < 10:
        raise ValueError(f"Too few C3D/TRC marker samples for {trc_path}")

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
    return rotation, translation_mm, rms_error_mm


def platform_bounds_xy_mm(platform: dict[str, Any], margin_mm: float) -> tuple[float, float, float, float]:
    corners = np.asarray(platform["corners"], dtype=np.float64)
    x = corners[0]
    y = corners[1]
    return (
        float(np.nanmin(x) - margin_mm),
        float(np.nanmax(x) + margin_mm),
        float(np.nanmin(y) - margin_mm),
        float(np.nanmax(y) + margin_mm),
    )


def quiet_baseline(values: np.ndarray, sample_rate_hz: float, quiet_seconds: float, lowest_fraction: float) -> tuple[np.ndarray, int, np.ndarray]:
    n = values.shape[0]
    edge_n = int(max(1, min(n, round(sample_rate_hz * quiet_seconds))))
    edge_idx = np.r_[0:edge_n, max(0, n - edge_n):n]
    vertical = np.abs(values[:, 2])
    finite_vertical = np.isfinite(vertical)
    if np.any(finite_vertical):
        rank_count = int(max(edge_n, round(n * lowest_fraction)))
        rank_idx = np.argsort(np.where(finite_vertical, vertical, np.inf))[:rank_count]
        candidate_idx = np.unique(np.r_[edge_idx, rank_idx])
    else:
        candidate_idx = np.unique(edge_idx)
    candidates = values[candidate_idx]
    good = np.isfinite(candidates).all(axis=1)
    if good.sum() == 0:
        return np.zeros(values.shape[1], dtype=np.float64), 0, candidate_idx[:0]
    return np.nanmedian(candidates[good], axis=0), int(good.sum()), candidate_idx[good]


def c3d_vector_to_trc(vector_c3d: np.ndarray, rotation_c3d_to_trc: np.ndarray) -> np.ndarray:
    return vector_c3d @ rotation_c3d_to_trc


def c3d_point_to_trc_m(point_c3d_mm: np.ndarray, rotation_c3d_to_trc: np.ndarray, translation_c3d_to_trc_mm: np.ndarray) -> np.ndarray:
    return (point_c3d_mm @ rotation_c3d_to_trc + translation_c3d_to_trc_mm) / 1000.0


def c3d_tz_to_vector(tz_nm: np.ndarray) -> np.ndarray:
    out = np.zeros((tz_nm.shape[0], 3), dtype=np.float64)
    out[:, 2] = tz_nm
    return out


def extract_tz_nmm(platform: dict[str, Any], n_analog: int) -> np.ndarray:
    if "Tz" in platform:
        tz = np.asarray(platform["Tz"], dtype=np.float64)
        if tz.ndim == 2:
            if tz.shape[0] == 3:
                return tz.T[:n_analog, 2]
            if tz.shape[1] == 3:
                return tz[:n_analog, 2]
        if tz.ndim == 1:
            return tz[:n_analog]
    moment = np.asarray(platform.get("moment", np.zeros((3, n_analog))), dtype=np.float64)
    if moment.ndim == 2 and moment.shape[0] == 3:
        return moment.T[:n_analog, 2]
    if moment.ndim == 2 and moment.shape[1] == 3:
        return moment[:n_analog, 2]
    return np.zeros(n_analog, dtype=np.float64)


def load_c3d_plate_data(
    c3d_path: Path,
    trc_path: Path,
    force_threshold_n: float,
    cop_margin_m: float,
    max_force_n: float,
    baseline_seconds: float,
    baseline_lowest_fraction: float,
) -> dict[str, Any]:
    c3d = ezc3d.c3d(str(c3d_path), extract_forceplat_data=True)
    platforms_raw = c3d["data"].get("platform", [])
    if not platforms_raw:
        raise ValueError(f"No FORCE_PLATFORM data extracted from {c3d_path}")

    analog_rate = float(c3d["header"]["analogs"]["frame_rate"])
    point_rate = float(c3d["header"]["points"]["frame_rate"])
    labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    points = np.asarray(c3d["data"]["points"], dtype=np.float64)
    n_analog = min(np.asarray(p["force"]).shape[1] for p in platforms_raw)

    right_marker_mm = interpolate_markers_to_analog(marker_centroid(points, labels, RIGHT_MARKERS), n_analog)
    left_marker_mm = interpolate_markers_to_analog(marker_centroid(points, labels, LEFT_MARKERS), n_analog)
    try:
        rotation_c3d_to_trc, translation_c3d_to_trc_mm, rotation_rms_error_mm = estimate_rotation_from_trc(points, labels, trc_path)
        rotation_error = None
    except Exception as exc:
        rotation_c3d_to_trc = np.eye(3, dtype=np.float64)
        translation_c3d_to_trc_mm = np.zeros(3, dtype=np.float64)
        rotation_rms_error_mm = float("nan")
        rotation_error = str(exc)

    platforms = []
    for plate_idx, platform in enumerate(platforms_raw, start=1):
        force_raw = np.asarray(platform["force"], dtype=np.float64).T[:n_analog]
        cop_raw = np.asarray(platform["center_of_pressure"], dtype=np.float64).T[:n_analog]
        tz_raw_nmm = extract_tz_nmm(platform, n_analog)
        force_offset, n_baseline, baseline_idx = quiet_baseline(force_raw, analog_rate, baseline_seconds, baseline_lowest_fraction)
        tz_candidates = tz_raw_nmm[baseline_idx] if baseline_idx.size else tz_raw_nmm
        tz_finite = tz_candidates[np.isfinite(tz_candidates)]
        tz_offset = float(np.nanmedian(tz_finite)) if tz_finite.size else 0.0

        force_corr = force_raw - force_offset
        tz_corr_nm = (tz_raw_nmm - tz_offset) / 1000.0
        vertical = force_corr[:, 2]
        xmin, xmax, ymin, ymax = platform_bounds_xy_mm(platform, cop_margin_m * 1000.0)

        active = vertical > force_threshold_n
        active &= np.linalg.norm(force_corr, axis=1) <= max_force_n
        active &= np.isfinite(cop_raw).all(axis=1)
        active &= cop_raw[:, 0] >= xmin
        active &= cop_raw[:, 0] <= xmax
        active &= cop_raw[:, 1] >= ymin
        active &= cop_raw[:, 1] <= ymax

        platforms.append({
            "plate_index": plate_idx,
            "force_pipeline": c3d_vector_to_trc(force_corr, rotation_c3d_to_trc),
            "cop_pipeline": c3d_point_to_trc_m(cop_raw, rotation_c3d_to_trc, translation_c3d_to_trc_mm),
            "grm_pipeline": c3d_vector_to_trc(c3d_tz_to_vector(tz_corr_nm), rotation_c3d_to_trc),
            "assignment_cop_xy_mm": cop_raw[:, [0, 1]],
            "vertical_load_n": vertical,
            "active": active,
            "force_offset_n": force_offset.tolist(),
            "tz_offset_nmm": tz_offset,
            "baseline_samples": n_baseline,
            "bounds_xy_mm": [xmin, xmax, ymin, ymax],
            "active_frames": int(np.sum(active)),
            "raw_force_minmax_n": [finite_minmax(force_raw[:, axis]) for axis in range(3)],
            "corrected_force_minmax_n": [finite_minmax(force_corr[:, axis]) for axis in range(3)],
        })

    return {
        "analog_rate": analog_rate,
        "point_rate": point_rate,
        "frames": n_analog,
        "right_marker_mm": right_marker_mm,
        "left_marker_mm": left_marker_mm,
        "rotation_c3d_to_trc": rotation_c3d_to_trc,
        "rotation_trc_to_c3d": rotation_c3d_to_trc.T,
        "translation_c3d_to_trc_mm": translation_c3d_to_trc_mm,
        "rotation_rms_error_mm": rotation_rms_error_mm,
        "rotation_error": rotation_error,
        "platforms": platforms,
    }


def combine_plates_by_foot(extracted: dict[str, Any], ambiguity_margin_m: float, max_ambiguous_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n = int(extracted["frames"])
    grf = np.zeros((n, 6), dtype=np.float64)
    cop = np.zeros((n, 6), dtype=np.float64)
    grm = np.zeros((n, 6), dtype=np.float64)
    stats: dict[str, Any] = {
        "right_platform_frames": 0,
        "left_platform_frames": 0,
        "active_platform_frames": 0,
        "inactive_or_rejected_platform_frames": 0,
        "ambiguous_platform_frames": 0,
        "ambiguous_examples": [],
    }
    ambiguity_margin_mm = ambiguity_margin_m * 1000.0

    right_xy = extracted["right_marker_mm"][:, [0, 1]]
    left_xy = extracted["left_marker_mm"][:, [0, 1]]

    for frame in range(n):
        side_payload: dict[str, list[dict[str, Any]]] = {"right": [], "left": []}
        for platform in extracted["platforms"]:
            if not platform["active"][frame]:
                stats["inactive_or_rejected_platform_frames"] += 1
                continue

            stats["active_platform_frames"] += 1
            cop_xy = platform["assignment_cop_xy_mm"][frame]
            d_right = np.linalg.norm(cop_xy - right_xy[frame]) if np.isfinite(right_xy[frame]).all() else np.inf
            d_left = np.linalg.norm(cop_xy - left_xy[frame]) if np.isfinite(left_xy[frame]).all() else np.inf
            side = "right" if d_right <= d_left else "left"
            diff = abs(d_right - d_left)

            if diff <= ambiguity_margin_mm:
                stats["ambiguous_platform_frames"] += 1
                if len(stats["ambiguous_examples"]) < max_ambiguous_samples:
                    stats["ambiguous_examples"].append({
                        "frame": int(frame),
                        "time_s": float(frame / extracted["analog_rate"]),
                        "plate_index": int(platform["plate_index"]),
                        "assigned": side,
                        "right_distance_m": float(d_right / 1000.0),
                        "left_distance_m": float(d_left / 1000.0),
                        "distance_difference_m": float(diff / 1000.0),
                        "cop_xy_m": [float(cop_xy[0] / 1000.0), float(cop_xy[1] / 1000.0)],
                    })

            side_payload[side].append(platform)
            stats[f"{side}_platform_frames"] += 1

        for side, sl in (("right", slice(0, 3)), ("left", slice(3, 6))):
            payload = side_payload[side]
            if not payload:
                continue
            forces = np.stack([p["force_pipeline"][frame] for p in payload], axis=0)
            cops = np.stack([p["cop_pipeline"][frame] for p in payload], axis=0)
            moments = np.stack([p["grm_pipeline"][frame] for p in payload], axis=0)
            weights = np.asarray([max(float(p["vertical_load_n"][frame]), 0.0) for p in payload], dtype=np.float64)

            grf[frame, sl] = forces.sum(axis=0)
            grm[frame, sl] = moments.sum(axis=0)
            if weights.sum() > 0:
                cop[frame, sl] = np.average(cops, axis=0, weights=weights)

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


def resolve_target_paths(mapping_row: dict[str, Any], dataset_root: Path) -> tuple[Path, Path, Path]:
    target = Path(mapping_row["target"]).as_posix().replace("StrokeDataSet/", "StrokeDataset/")
    raw_mot_path = Path(target)
    if not raw_mot_path.is_absolute():
        raw_mot_path = dataset_root.parent / raw_mot_path
    motion_dir = raw_mot_path.parent.parent
    source_c3d = dataset_root.parent / mapping_row["source_c3d"]
    subject = motion_dir.parent.parent.name
    trial = motion_dir.parent.name.replace("Trial_", "trial")
    trc_path = dataset_root.parent / "NAS_Data/Users/DMagruder/DATA/Criekinge" / subject / "trials" / trial / "markers.trc"
    return raw_mot_path, source_c3d, trc_path


def process_mapping(
    mapping_row: dict[str, Any],
    dataset_root: Path,
    force_threshold_n: float,
    cop_margin_m: float,
    max_force_n: float,
    baseline_seconds: float,
    baseline_lowest_fraction: float,
    ambiguity_margin_m: float,
    max_ambiguous_samples: int,
) -> dict[str, Any]:
    raw_mot_path, source_c3d, trc_path = resolve_target_paths(mapping_row, dataset_root)
    motion_dir = raw_mot_path.parent.parent

    extracted = load_c3d_plate_data(
        source_c3d,
        trc_path,
        force_threshold_n=force_threshold_n,
        cop_margin_m=cop_margin_m,
        max_force_n=max_force_n,
        baseline_seconds=baseline_seconds,
        baseline_lowest_fraction=baseline_lowest_fraction,
    )
    grf, cop, grm, stats = combine_plates_by_foot(extracted, ambiguity_margin_m, max_ambiguous_samples)
    time = np.arange(grf.shape[0], dtype=np.float64) / extracted["analog_rate"]

    raw_mot_path.parent.mkdir(parents=True, exist_ok=True)
    write_force_mot(raw_mot_path, time, grf, cop, grm)
    np.save(motion_dir / "GRF.npy", grf.astype(np.float32))
    np.save(motion_dir / "COP.npy", cop.astype(np.float32))
    np.save(motion_dir / "GRM.npy", grm.astype(np.float32))
    np.save(motion_dir / "Time.npy", time.astype(np.float64))

    vertical_r = grf[:, 1]
    vertical_l = grf[:, 4]
    return {
        "trial": f"{motion_dir.parent.parent.name}/{motion_dir.parent.name}",
        "source_c3d": str(source_c3d),
        "source_trc": str(trc_path),
        "raw_mot": str(raw_mot_path),
        "frames": int(grf.shape[0]),
        "analog_rate": extracted["analog_rate"],
        "point_rate": extracted["point_rate"],
        "force_threshold_n": force_threshold_n,
        "output_convention": "OpenSim-style Y-up per foot: GRF [x, vertical_y, z], COP [x, y, z], GRM [x, vertical_y, z]",
        "rotation_c3d_to_trc": extracted["rotation_c3d_to_trc"].tolist(),
        "rotation_trc_to_c3d": extracted["rotation_trc_to_c3d"].tolist(),
        "translation_c3d_to_trc_mm": extracted["translation_c3d_to_trc_mm"].tolist(),
        "rotation_rms_error_mm": extracted["rotation_rms_error_mm"],
        "rotation_error": extracted["rotation_error"],
        "platforms": [
            {
                "plate_index": p["plate_index"],
                "active_frames": p["active_frames"],
                "force_offset_n": p["force_offset_n"],
                "tz_offset_nmm": p["tz_offset_nmm"],
                "baseline_samples": p["baseline_samples"],
                "bounds_xy_mm": p["bounds_xy_mm"],
                "raw_force_minmax_n": p["raw_force_minmax_n"],
                "corrected_force_minmax_n": p["corrected_force_minmax_n"],
            }
            for p in extracted["platforms"]
        ],
        "max_abs_grf_n": float(np.nanmax(np.abs(grf))) if grf.size else 0.0,
        "max_abs_cop_m": float(np.nanmax(np.abs(cop))) if cop.size else 0.0,
        "max_abs_grm_nm": float(np.nanmax(np.abs(grm))) if grm.size else 0.0,
        "right_vertical_max_n": float(np.nanmax(vertical_r)) if vertical_r.size else 0.0,
        "left_vertical_max_n": float(np.nanmax(vertical_l)) if vertical_l.size else 0.0,
        **stats,
    }


def selected_rows(mapping: dict[str, Any], only: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    wanted = set(only or [])
    for subject, subject_rows in sorted(mapping["mappings"].items()):
        for row in subject_rows:
            label = f"{subject}/Trial_{row['trial']}"
            if wanted and label not in wanted:
                continue
            rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("StrokeDataset"))
    parser.add_argument("--mapping", type=Path, default=Path("StrokeDataset/force_trial_mapping.json"))
    parser.add_argument("--force-threshold-n", type=float, default=10.0)
    parser.add_argument("--cop-margin-m", type=float, default=0.20)
    parser.add_argument("--max-force-n", type=float, default=5000.0)
    parser.add_argument("--baseline-seconds", type=float, default=0.50)
    parser.add_argument("--baseline-lowest-fraction", type=float, default=0.10)
    parser.add_argument("--ambiguity-margin-m", type=float, default=0.10)
    parser.add_argument("--max-ambiguous-samples", type=int, default=20)
    parser.add_argument("--processdata-subset", action="store_true", help="Run the 20 StrokeDataset trials listed in ProcessData.py CONFIG.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional labels like SUBJ09/Trial_2.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    mapping = json.load(args.mapping.open())
    if args.processdata_subset:
        args.only = list(PROCESSDATA_TRIAL_SUBSET) + list(args.only or [])
    rows = selected_rows(mapping, args.only, args.limit)

    manifest: dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "mapping": str(args.mapping.resolve()),
        "force_threshold_n": args.force_threshold_n,
        "max_force_n": args.max_force_n,
        "cop_margin_m": args.cop_margin_m,
        "baseline_seconds": args.baseline_seconds,
        "baseline_lowest_fraction": args.baseline_lowest_fraction,
        "ambiguity_margin_m": args.ambiguity_margin_m,
        "method": (
            "Plate-first ezc3d extraction. Force platforms are read independently in the raw C3D lab frame, "
            "baseline-corrected, thresholded at corrected C3D Fz, assigned to the nearest raw C3D right/left "
            "foot marker centroid, then summed per foot. Outputs are rotated into the per-trial exported "
            "TRC/OpenSim frame so GRF, COP, GRM, and Pos.npy share the same coordinate frame."
        ),
        "major_assumptions": [
            "C3D force data are stored by force plate, not by foot.",
            "C3D markers, COP, force vectors, and force-plate corners are in the original lab/world frame.",
            "AddBiomechanics rotated marker/IK data; final force outputs are rotated into that same TRC/OpenSim frame.",
            "Pos.npy is unchanged; force/COP/GRM outputs are transformed instead of rotating the pelvis/root body back.",
            "Final Motion arrays remain compatible with ProcessData.py.",
            "C3D Fz is the raw vertical force and maps through the fitted C3D-to-TRC transform to OpenSim-style output Y.",
            "COP from ezc3d is in millimeters and is saved in meters.",
            "Low-force/out-of-bounds COP is rejected.",
            "Nearest-foot assignment is used, with ambiguous frames reported.",
            "Cross-plate foot strikes are summed per foot.",
        ],
        "coordinate_transform": {
            "GRF.npy": "raw C3D force vectors rotated by the fitted per-trial C3D-to-TRC rotation",
            "COP.npy": "raw C3D COP points rotated and translated by the fitted per-trial C3D-to-TRC rigid transform, then converted mm to m",
            "GRM.npy": "corrected raw C3D free-moment vector [0, 0, Tz_Nm] rotated by the fitted per-trial C3D-to-TRC rotation",
        },
        "output_columns": {
            "GRF.npy": ["R_x", "R_y_vertical", "R_z", "L_x", "L_y_vertical", "L_z"],
            "COP.npy": ["R_x", "R_y", "R_z", "L_x", "L_y", "L_z"],
            "GRM.npy": ["R_x", "R_y_vertical", "R_z", "L_x", "L_y_vertical", "L_z"],
        },
        "trials_seen": len(rows),
        "processdata_subset": bool(args.processdata_subset),
        "only": list(args.only or []),
        "trials_written": 0,
        "failures": [],
        "trials": [],
        "ambiguous_trials": [],
    }

    for i, row in enumerate(rows, start=1):
        try:
            result = process_mapping(
                row,
                args.dataset_root,
                force_threshold_n=args.force_threshold_n,
                cop_margin_m=args.cop_margin_m,
                max_force_n=args.max_force_n,
                baseline_seconds=args.baseline_seconds,
                baseline_lowest_fraction=args.baseline_lowest_fraction,
                ambiguity_margin_m=args.ambiguity_margin_m,
                max_ambiguous_samples=args.max_ambiguous_samples,
            )
            manifest["trials_written"] += 1
            manifest["trials"].append(result)
            if result["ambiguous_platform_frames"]:
                manifest["ambiguous_trials"].append({
                    "trial": result["trial"],
                    "ambiguous_platform_frames": result["ambiguous_platform_frames"],
                    "examples": result["ambiguous_examples"],
                })
        except Exception as exc:
            manifest["failures"].append({"row": row, "error": str(exc)})
        if i % 50 == 0:
            print(f"processed {i}/{len(rows)}", flush=True)

    manifest_path = args.dataset_root / "stroke_c3d_force_reextraction_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(json.dumps({
        "trials_seen": manifest["trials_seen"],
        "trials_written": manifest["trials_written"],
        "failures": len(manifest["failures"]),
        "ambiguous_trials": len(manifest["ambiguous_trials"]),
        "manifest": str(manifest_path),
    }, indent=2))

    if manifest["ambiguous_trials"]:
        print("Ambiguous assignment trials:")
        for item in manifest["ambiguous_trials"][:50]:
            print(f"  {item['trial']}: {item['ambiguous_platform_frames']} platform-frame assignments")
        if len(manifest["ambiguous_trials"]) > 50:
            print(f"  ... {len(manifest['ambiguous_trials']) - 50} more")


if __name__ == "__main__":
    main()
