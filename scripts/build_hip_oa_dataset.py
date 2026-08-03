#!/usr/bin/env python3
"""QC-first extraction of the Hip_OA dataset from Datasets_Local/Bertaux_withForces.

Follows the extraction conventions established by scripts/extract_pd_boari_robust.py
(versioned output, Config dataclass, hysteresis contact detection, per-event foot
assignment with distance/margin/consistency metrics, PASS/REVIEW/REJECT QC) and the
force interchange format of scripts/npy_from_force_mot.py.

Source (AddBiomechanics export, one directory per *session*, e.g. HOA001_M0):
    _subject.json                                  subject metadata
    OpenSimData/Model/LaiUhlrich2022_scaled.osim   scaled OpenSim model
    osim_results/IK/trial<N>_segment_0_ik.mot      IK kinematics, 34 coords, radians
    trials/trial<N>/markers.trc                    marker trajectories, 100 Hz, metres
    trials/trial<N>/grf.mot                        2-plate ground reaction, 19 cols, 1000 Hz

Output:
    Hip_OA/<session>/Patient_MD.json
    Hip_OA/<session>/OpenSimModel.osim
    Hip_OA/<session>/Trial_<N>/Motion/{Pos,Vel,Accel}.npy       (T_kin, 23) float32
    Hip_OA/<session>/Trial_<N>/Motion/Time_for_pos.npy          (T_kin,)   float64
    Hip_OA/<session>/Trial_<N>/Motion/{GRF,COP,GRM}.npy         (T_frc, 6) float32
    Hip_OA/<session>/Trial_<N>/Motion/Time.npy                  (T_frc,)   float64
    Hip_OA/<session>/Trial_<N>/Motion/ContactMask.npy           (T_frc, 2) bool
    Hip_OA/<session>/Trial_<N>/Motion/ContaminatedMask.npy      (T_frc,)   bool
    Hip_OA/<session>/Trial_<N>/Motion/ForceAssignmentConfidence.npy (T_frc, 2) float32
    Hip_OA/<session>/Trial_<N>/Motion/Raw/                      source + canonical .mot
    Hip_OA/<session>/Trial_<N>/Motion/force_plate_assignment.json

Column conventions (ProcessData.py POS_COLUMNS / npy_from_force_mot.py MOT_COLUMNS):
    Pos/Vel/Accel   -> 23 DOF, no arms, no knee_angle_beta; radians, metres for pelvis_t*
    GRF/COP/GRM     -> [R_x, R_y, R_z, L_x, L_y, L_z]; N, m, N*m; OpenSim Y-up world frame

Kinematics (100 Hz) and forces (1000 Hz) are kept on their native rates. ProcessData.py
resamples both onto a uniform grid using Time_for_pos.npy and Time.npy.

Timebase
--------
The AddBiomechanics IK .mot and PointKinematics .sto time columns drift (dt ~0.0099842 s
against a true 100 Hz capture, ~19 ms of skew by the end of a 6 s trial). The TRC header is
authoritative: kinematics time is rebuilt as arange(n) / DataRate, and foot landmarks are
taken from the TRC markers, which share that grid and the force-plate world frame.

Force-plate -> foot assignment
------------------------------
Bertaux is a long instrumented walkway: a plate is NOT a foot. Plate/foot pairing varies per
trial and a single plate frequently records two consecutive footfalls, sometimes at once.
Each plate contact is assigned to the foot whose heel->toe segment the COP tracks most
closely (majority vote over the contact). Nothing is dropped or zeroed: contacts whose vote
is not unanimous, or whose COP sits far from both feet, are flagged and their offending
sub-intervals recorded in force_plate_assignment.json for visual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from paths import artifact, dataset  # noqa: E402

VERSION = "1.0.0"
TRIAL_RE = re.compile(r"^trial(\d+)$")

# 23-DOF save order, identical to ProcessData.POS_COLUMNS.
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

# Canonical 19-column force interchange format (npy_from_force_mot.MOT_COLUMNS).
FORCE_COLUMNS = (
    "time",
    "R_ground_force_vx", "R_ground_force_vy", "R_ground_force_vz",
    "R_ground_force_px", "R_ground_force_py", "R_ground_force_pz",
    "R_ground_torque_x", "R_ground_torque_y", "R_ground_torque_z",
    "L_ground_force_vx", "L_ground_force_vy", "L_ground_force_vz",
    "L_ground_force_px", "L_ground_force_py", "L_ground_force_pz",
    "L_ground_torque_x", "L_ground_torque_y", "L_ground_torque_z",
)

# Plug-in-Gait foot markers; heel->toe is the segment the COP should track during stance.
RIGHT_MARKERS = {"heel": "RHEE", "toe": "RTOE", "ankle": "RANK"}
LEFT_MARKERS = {"heel": "LHEE", "toe": "LTOE", "ankle": "LANK"}
SIDE_SLICE = {0: slice(0, 3), 1: slice(3, 6)}
SIDE_NAME = {0: "right", 1: "left"}


@dataclass
class Config:
    force_on_n: float = 30.0
    force_off_n: float = 15.0
    min_contact_s: float = 0.040
    bridge_gap_s: float = 0.020
    # A contact is assessed on frames carrying real load, so ramp-on/ramp-off samples
    # (where the COP is noisy) do not dominate the foot vote.
    assess_rel_peak: float = 0.05
    max_cop_foot_distance_m: float = 0.35   # gross failure: COP nowhere near either foot
    contamination_distance_m: float = 0.10  # tighter flag for "COP left the assigned foot"
    side_margin_m: float = 0.05
    min_side_consistency: float = 0.97      # <1.0 => two feet shared this plate contact
    min_flag_run_frames: int = 3
    max_force_n: float = 5000.0
    review_excluded_force_n: float = 50.0
    max_timebase_mismatch_s: float = 0.05
    peak_grf_bw_max: float = 3.0            # sanity bound on peak vertical force / body weight


# ── generic helpers (mirrors extract_pd_boari_robust.py) ─────────────────────
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of `mask` as [start, end) index pairs."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def hysteresis_contacts(vertical, on, off, rate, min_s, gap_s) -> np.ndarray:
    """Schmitt-trigger contact detection; vectorised form of the reference loop."""
    vertical = np.asarray(vertical, dtype=np.float64)
    n = vertical.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    # State flips to True at the first sample >= on and to False at the first sample < off;
    # samples in the dead band inherit the previous state.
    change = np.where(vertical >= on, 1, np.where(vertical < off, -1, 0))
    nonzero = change != 0
    last = np.where(nonzero, np.arange(n), -1)
    np.maximum.accumulate(last, out=last)
    active = np.zeros(n, dtype=bool)
    seen = last >= 0
    active[seen] = change[last[seen]] == 1

    max_gap = int(round(gap_s * rate))
    if max_gap:
        for start, end in runs(~active):
            if start > 0 and end < n and end - start <= max_gap:
                active[start:end] = True
    min_len = max(1, int(round(min_s * rate)))
    for start, end in runs(active):
        if end - start < min_len:
            active[start:end] = False
    return active


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


# ── readers ──────────────────────────────────────────────────────────────────
def read_mot(path: Path) -> tuple[np.ndarray, list[str], bool]:
    """Read an OpenSim .mot/.sto. Tolerates the '-nan(ind)' MSVC spelling."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None)
    if end is None or end + 1 >= len(lines):
        raise ValueError(f"Invalid MOT header: {path}")
    in_degrees = False
    for line in lines[:end]:
        if line.lower().replace(" ", "").startswith("indegrees="):
            in_degrees = line.split("=", 1)[1].strip().lower() == "yes"
    columns = lines[end + 1].split("\t")
    if len(columns) == 1:
        columns = lines[end + 1].split()
    body = re.sub(r"-?nan\(ind\)", "nan", "\n".join(lines[end + 2:]), flags=re.IGNORECASE)
    data = np.loadtxt(StringIO(body), dtype=np.float64, ndmin=2)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"MOT data/header mismatch: {path}")
    return data, columns, in_degrees


def read_trc(path: Path) -> dict[str, Any]:
    """Marker trajectories in metres on the header-declared (authoritative) time grid."""
    lines = path.read_text(errors="replace").splitlines()
    header_i = next((i for i, line in enumerate(lines) if line.startswith("Frame#")), None)
    if header_i is None or header_i < 2:
        raise ValueError(f"No Frame# header in {path}")
    meta = dict(zip((k.strip() for k in lines[header_i - 2].split("\t")),
                    (v.strip() for v in lines[header_i - 1].split("\t"))))
    rate = float(meta["DataRate"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"Bad DataRate in {path}: {meta.get('DataRate')}")
    units = meta.get("Units", "m").strip().lower()
    if units in {"m", "meters", "metres"}:
        scale = 1.0
    elif units in {"mm", "millimeters", "millimetres"}:
        scale = 1e-3
    else:
        raise ValueError(f"Unsupported TRC units {meta.get('Units')!r} in {path}")

    header = lines[header_i].split("\t")
    entries, seen = [], set()
    for col in range(2, len(header), 3):
        name = header[col].strip()
        if name and name not in seen:
            entries.append((name, col))
            seen.add(name)
    rows = [line.split("\t") for line in lines[header_i + 2:] if line.strip()]
    markers = {}
    for name, col in entries:
        xyz = np.full((len(rows), 3), np.nan)
        for i, fields in enumerate(rows):
            try:
                xyz[i] = (float(fields[col]), float(fields[col + 1]), float(fields[col + 2]))
            except (IndexError, ValueError):
                pass
        markers[name] = xyz * scale
    n = len(rows)
    declared = int(float(meta.get("NumFrames", n)))
    return {
        "n_frames": n, "declared_frames": declared, "rate_hz": rate, "units": meta.get("Units"),
        "time": np.arange(n, dtype=np.float64) / rate,
        "markers": markers,
    }


def _header_int(path: Path, key: str) -> int | None:
    for line in path.read_text(errors="replace").splitlines():
        if line.strip().lower() == "endheader":
            return None
        if line.strip().lower().startswith(f"{key.lower()}="):
            try:
                return int(float(line.split("=", 1)[1]))
            except ValueError:
                return None
    return None


# ── kinematics ───────────────────────────────────────────────────────────────
def load_kinematics(ik_path: Path, trc: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict]:
    """23-DOF Pos (radians / metres) on the TRC time grid, plus provenance notes."""
    data, columns, in_degrees = read_mot(ik_path)
    index = {name: i for i, name in enumerate(columns)}
    missing = [name for name in KINEMATIC_23 if name not in index]
    if missing:
        raise ValueError(f"{ik_path}: IK MOT missing 23-DOF coordinates {missing}")

    pos = np.column_stack([data[:, index[name]] for name in KINEMATIC_23])
    if in_degrees:
        rotational = [i for i, name in enumerate(KINEMATIC_23) if name not in TRANSLATIONS]
        pos[:, rotational] = np.deg2rad(pos[:, rotational])

    mot_time = data[:, index["time"]]
    n = pos.shape[0]
    notes = {
        "ik_rows": int(n),
        "trc_frames": int(trc["n_frames"]),
        "trc_rate_hz": trc["rate_hz"],
        "ik_mot_time_end_s": float(mot_time[-1]) if n else None,
        "ik_mot_median_dt_s": float(np.median(np.diff(mot_time))) if n > 1 else None,
        "timebase_source": "trc_header",
        "row_count_matches_trc": bool(n == trc["n_frames"]),
    }
    if n == trc["n_frames"]:
        time = trc["time"]
    else:
        # Row counts disagree: keep every IK row and extend the TRC grid at its own rate,
        # rather than silently truncating either record.
        time = np.arange(n, dtype=np.float64) / trc["rate_hz"]
        notes["timebase_source"] = "trc_rate_extended"
    return pos, time, notes


# ── foot geometry ────────────────────────────────────────────────────────────
def foot_tracks(trc: dict[str, Any]) -> tuple[dict[int, dict[str, np.ndarray]], list[str]]:
    tracks, filled = {}, []
    for side, names in ((0, RIGHT_MARKERS), (1, LEFT_MARKERS)):
        entry = {}
        for role, marker in names.items():
            if marker not in trc["markers"]:
                raise ValueError(f"TRC missing foot marker {marker}")
            xyz, missing_cols = interpolate_nonfinite(trc["markers"][marker], default=np.nan)
            if missing_cols:
                filled.append(f"{marker}:{missing_cols}")
            entry[role] = xyz
        tracks[side] = entry
    return tracks, filled


def _dist_to_segment(px, pz, ax, az, bx, bz):
    """Horizontal distance from point(s) to the heel->toe segment(s)."""
    vx, vz = bx - ax, bz - az
    wx, wz = px - ax, pz - az
    denom = vx * vx + vz * vz
    t = np.clip((wx * vx + wz * vz) / np.where(denom == 0.0, 1e-9, denom), 0.0, 1.0)
    return np.hypot(wx - t * vx, wz - t * vz)


def foot_distances(tracks, side, frames, cop_x, cop_z) -> tuple[np.ndarray, np.ndarray]:
    """(segment distance, centroid distance) from the COP to one foot, per frame."""
    heel = tracks[side]["heel"][frames]
    toe = tracks[side]["toe"][frames]
    ankle = tracks[side]["ankle"][frames]
    seg = _dist_to_segment(cop_x, cop_z, heel[:, 0], heel[:, 2], toe[:, 0], toe[:, 2])
    centroid = np.nanmean(np.stack([heel, toe, ankle]), axis=0)
    cen = np.hypot(cop_x - centroid[:, 0], cop_z - centroid[:, 2])
    return seg, cen


# ── force plates ─────────────────────────────────────────────────────────────
def split_forces(grf_path: Path, tracks, kin_rate: float, cfg: Config) -> tuple[dict, dict]:
    """Split the 2-plate grf.mot into per-foot GRF/COP/GRM plus an assignment report."""
    data, columns, _ = read_mot(grf_path)
    index = {name: i for i, name in enumerate(columns)}
    if "time" not in index:
        raise ValueError(f"{grf_path}: no time column")
    time = data[:, index["time"]]
    n = time.size
    if n < 2:
        raise ValueError(f"{grf_path}: {n} rows")
    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError(f"{grf_path}: nonmonotonic time")
    rate = 1.0 / dt

    grf = np.zeros((n, 6))
    grm = np.zeros((n, 6))
    cop = np.zeros((n, 6))
    cop_num = np.zeros((n, 2, 3))
    cop_den = np.zeros((n, 2))
    contact_mask = np.zeros((n, 2), dtype=bool)
    confidence = np.zeros((n, 2), dtype=np.float64)
    loaded_count = np.zeros((n, 2), dtype=np.int16)
    contaminated_mask = np.zeros(n, dtype=bool)   # union across plates and flag kinds

    plates = sorted({int(m.group(1)) for c in columns
                     if (m := re.match(r"ground_force_(\d+)_vx$", c))})
    if not plates:
        raise ValueError(f"{grf_path}: no ground_force_<n>_vx columns")

    events: list[dict[str, Any]] = []
    for plate in plates:
        need = ([f"ground_force_{plate}_v{a}" for a in "xyz"]
                + [f"ground_force_{plate}_p{a}" for a in "xyz"]
                + [f"ground_moment_{plate}_m{a}" for a in "xyz"])
        missing = [c for c in need if c not in index]
        if missing:
            raise ValueError(f"{grf_path}: plate {plate} missing columns {missing}")
        force = np.nan_to_num(data[:, [index[c] for c in need[0:3]]])
        point = data[:, [index[c] for c in need[3:6]]]
        free = np.nan_to_num(data[:, [index[c] for c in need[6:9]]])

        active = hysteresis_contacts(force[:, 1], cfg.force_on_n, cfg.force_off_n,
                                     rate, cfg.min_contact_s, cfg.bridge_gap_s)
        active &= np.linalg.norm(force, axis=1) <= cfg.max_force_n
        active &= np.isfinite(point).all(axis=1)

        for start, end in runs(active):
            rows = np.arange(start, end)
            fy = force[rows, 1]
            peak = float(fy.max())
            event: dict[str, Any] = {
                "plate": plate,
                "start_frame": int(start), "end_frame": int(end - 1),
                "start_time_s": float(time[start]), "end_time_s": float(time[end - 1]),
                "frames": int(rows.size), "peak_vertical_n": peak,
            }
            assess = fy >= max(cfg.force_on_n, cfg.assess_rel_peak * peak)
            if assess.sum() < 3:
                event.update(assignment="unassigned", reason="too_few_loaded_frames",
                             contaminated=False, contaminated_regions=[])
                events.append(event)
                continue

            arows = rows[assess]
            # Foot markers live on the kinematics grid; map force samples onto it.
            kframes = np.clip(np.rint(time[arows] * kin_rate).astype(int),
                              0, len(tracks[0]["heel"]) - 1)
            cx, cz = point[arows, 0], point[arows, 2]
            seg_r, cen_r = foot_distances(tracks, 0, kframes, cx, cz)
            seg_l, cen_l = foot_distances(tracks, 1, kframes, cx, cz)
            finite = np.isfinite(seg_r) & np.isfinite(seg_l)
            if finite.sum() < max(3, assess.sum() // 2):
                event.update(assignment="unassigned", reason="missing_foot_markers",
                             contaminated=False, contaminated_regions=[])
                events.append(event)
                continue

            med_r, med_l = float(np.median(seg_r[finite])), float(np.median(seg_l[finite]))
            side = 0 if med_r <= med_l else 1
            best, other = (med_r, med_l) if side == 0 else (med_l, med_r)
            nearest = np.where(seg_r <= seg_l, 0, 1)
            consistency = float(np.mean(nearest[finite] == side))
            min_seg = np.minimum(seg_r, seg_l)
            min_cen = np.minimum(cen_r, cen_l)

            # Flagged sub-intervals, indexed back onto the force timebase. Kinds can
            # overlap, so frames are also unioned into a mask for an honest frame count.
            regions = []
            flagged = np.zeros(arows.size, dtype=bool)
            flags = (
                ("other_foot_on_same_plate", finite & (nearest != side)),
                ("cop_far_from_assigned_foot", finite & (min_seg > cfg.contamination_distance_m)),
                ("cop_far_from_both_feet", finite & (min_cen > cfg.max_cop_foot_distance_m)),
            )
            for kind, mask in flags:
                for a, b in runs(mask):
                    if b - a < cfg.min_flag_run_frames:
                        continue
                    fa, fb = int(arows[a]), int(arows[b - 1])
                    flagged[a:b] = True
                    regions.append({
                        "kind": kind,
                        "start_frame": fa, "end_frame": fb,
                        "start_time_s": float(time[fa]), "end_time_s": float(time[fb]),
                        "frames": fb - fa + 1,
                        "likely_other_foot": SIDE_NAME[1 - side] if kind == "other_foot_on_same_plate" else None,
                        "max_distance_m": float(min_seg[a:b].max()),
                    })

            contaminated = bool(regions) and (
                consistency < cfg.min_side_consistency
                or float(np.percentile(min_seg[finite], 95)) > cfg.contamination_distance_m
            )
            quality = "accepted"
            if best > cfg.max_cop_foot_distance_m:
                quality = "cop_far_from_both_feet"
            elif (other - best) < cfg.side_margin_m or consistency < cfg.min_side_consistency:
                quality = "ambiguous"
            event.update({
                "assignment": SIDE_NAME[side],
                "assignment_quality": quality,
                "reason": "majority_vote",
                "median_right_distance_m": med_r,
                "median_left_distance_m": med_l,
                "distance_margin_m": float(other - best),
                "side_consistency": consistency,
                "p95_min_distance_m": float(np.percentile(min_seg[finite], 95)),
                "assessed_frames": int(finite.sum()),
                "contaminated": contaminated,
                "contaminated_frames": int(flagged.sum()) if contaminated else 0,
                "contaminated_regions": regions if contaminated else [],
            })
            events.append(event)
            if contaminated:
                contaminated_mask[arows[flagged]] = True

            sl = SIDE_SLICE[side]
            grf[rows, sl] += force[rows]
            grm[rows, sl] += free[rows]                       # free moments add
            weight = np.clip(force[rows, 1], 0.0, None)
            cop_num[rows, side] += weight[:, None] * np.nan_to_num(point[rows])
            cop_den[rows, side] += weight
            contact_mask[rows, side] = True
            loaded_count[arows, side] += 1
            conf = 0.0 if quality != "accepted" else float(min(other - best, 1.0))
            prev = confidence[rows, side]
            confidence[rows, side] = np.where(prev == 0.0, conf, np.minimum(prev, conf))

    for side in (0, 1):
        ok = cop_den[:, side] > 0
        cop[ok, SIDE_SLICE[side]] = cop_num[ok, side] / cop_den[ok, side, None]

    merged = [
        {"start_frame": a, "end_frame": b - 1, "start_time_s": float(time[a]),
         "end_time_s": float(time[b - 1]), "frames": b - a}
        for a, b in runs(loaded_count.max(axis=1) > 1)
    ]
    contaminated_frames = int(contaminated_mask.sum())
    report = {
        "source_grf_mot": str(grf_path),
        "force_frames": n,
        "force_rate_hz": rate,
        "force_duration_s": float(time[-1] - time[0]),
        "plates": plates,
        "contact_events": events,
        "both_plates_same_foot_regions": merged,
        "contact_summary": {
            "events_total": len(events),
            "events_right": sum(e["assignment"] == "right" for e in events),
            "events_left": sum(e["assignment"] == "left" for e in events),
            "events_unassigned": sum(e["assignment"] == "unassigned" for e in events),
            "events_ambiguous": sum(e.get("assignment_quality") == "ambiguous" for e in events),
            "events_cop_far": sum(e.get("assignment_quality") == "cop_far_from_both_feet" for e in events),
            "events_contaminated": sum(bool(e.get("contaminated")) for e in events),
            "max_vertical_grf_n": float(np.max(grf[:, [1, 4]])) if n else 0.0,
        },
        "contaminated": any(e.get("contaminated") for e in events),
        "contaminated_frames": contaminated_frames,
        "contaminated_frame_fraction": contaminated_frames / n,
    }
    arrays = {"grf": grf, "cop": cop, "grm": grm, "time": time,
              "contact_mask": contact_mask, "confidence": confidence,
              "contaminated_mask": contaminated_mask}
    return arrays, report


# ── writers ──────────────────────────────────────────────────────────────────
def write_mot(path: Path, time, values, columns, in_degrees=False) -> None:
    table = np.column_stack([time, values])
    with path.open("w") as handle:
        handle.write(f"{path.stem}\nversion=1\nnRows={len(table)}\nnColumns={table.shape[1]}\n")
        handle.write(f"inDegrees={'yes' if in_degrees else 'no'}\nendheader\n")
        handle.write("\t".join(["time", *columns]) + "\n")
        for row in table:
            handle.write("\t".join(f"{value:.10g}" for value in row) + "\n")


def write_force_mot(path: Path, time, grf, cop, grm) -> None:
    """Canonical 19-column R/L force file consumed by scripts/npy_from_force_mot.py."""
    values = np.column_stack([grf[:, :3], cop[:, :3], grm[:, :3],
                              grf[:, 3:], cop[:, 3:], grm[:, 3:]])
    write_mot(path, time, values, list(FORCE_COLUMNS[1:]))


# ── QC ───────────────────────────────────────────────────────────────────────
def qc_status(report: dict, notes: dict, grf: np.ndarray, body_weight_n: float,
              cfg: Config) -> tuple[str, list[str]]:
    """Advisory QC. Nothing is dropped: every extracted trial is written regardless."""
    reasons: list[str] = []
    events = report["contact_events"]
    accepted = [e for e in events if e.get("assignment_quality") == "accepted"]
    assigned = [e for e in events if e["assignment"] in {"right", "left"}]
    unassigned = [e for e in events if e["assignment"] == "unassigned"]

    if not notes["row_count_matches_trc"]:
        reasons.append(f"ik_rows={notes['ik_rows']} != trc_frames={notes['trc_frames']}")
    mismatch = abs(report["force_duration_s"] - notes["kinematics_duration_s"])
    if mismatch > cfg.max_timebase_mismatch_s:
        reasons.append(f"timebase_mismatch_s={mismatch:.4f}")
    if not assigned:
        reasons.append("no_assigned_contacts")
    elif {e["assignment"] for e in assigned} != {"right", "left"}:
        reasons.append("contacts_for_only_one_foot")
    if report["contact_summary"]["events_contaminated"]:
        reasons.append(f"contaminated_contacts={report['contact_summary']['events_contaminated']}")
    if not accepted and assigned:
        reasons.append("no_unambiguous_contacts")
    max_unassigned = max((e["peak_vertical_n"] for e in unassigned), default=0.0)
    if max_unassigned >= cfg.review_excluded_force_n:
        reasons.append(f"unassigned_contact_peak_n={max_unassigned:.1f}")
    if grf.size and float(np.min(grf[:, [1, 4]])) < -1e-3:
        reasons.append("negative_vertical_force")
    peak_bw = (float(np.max(grf[:, [1, 4]])) / body_weight_n) if body_weight_n > 0 else 0.0
    if peak_bw > cfg.peak_grf_bw_max:
        reasons.append(f"peak_vertical_grf_bw={peak_bw:.2f}")

    if not assigned or not notes["row_count_matches_trc"] or mismatch > cfg.max_timebase_mismatch_s:
        return "REJECT", reasons
    if reasons:
        return "REVIEW", reasons
    return "PASS", reasons


# ── per-trial / per-session ──────────────────────────────────────────────────
ASSUMPTIONS = [
    "The TRC header DataRate is the authoritative capture rate; the IK .mot and "
    "PointKinematics .sto time columns drift and are not trusted.",
    "IK .mot row i corresponds to TRC frame i.",
    "markers.trc, grf.mot and the IK coordinates share one Y-up world frame in metres.",
    "grf.mot COP is reported on the ground plane and its moment columns are free moments.",
    "A plate contact may be shared by both feet; it is assigned to the majority foot and "
    "the disagreeing sub-intervals are flagged rather than dropped.",
    "Where two plates carry the same foot at once, forces and free moments sum and the COP "
    "is the vertical-force-weighted average.",
]


def patient_md(session: str, subject_json: Path) -> dict:
    meta = json.loads(subject_json.read_text())
    cohort = ("HipOA" if session.startswith("HOA")
              else "Healthy" if session.startswith("HEA") else "Unknown")
    return {
        "Patient_ID": session,
        "Height_m": float(meta["heightM"]),
        "Mass_kg": float(meta["massKg"]),
        "BiologicalSex": str(meta.get("sex", "")).lower(),
        "Note": f"Extracted from Datasets_Local/Bertaux_withForces/{session}/_subject.json",
        "Cohort": cohort,
        "SubjectIdentifier": meta.get("subjectIdentifier", session.split("_")[0]),
        "Session": session.split("_", 1)[1] if "_" in session else "",
        "PrePost": meta.get("prepost"),
        "AgeYears": meta.get("ageYears"),
        "ImpairedLeg": meta.get("impairedLeg"),
        "SubjectTags": [str(t) for t in meta.get("subjectTags", [])],
    }


def process_trial(src: Path, session: str, trial: str, out_trial: Path, body_weight_n: float,
                  cfg: Config, copy_raw: bool, hash_sources: bool) -> dict:
    ik_path = src / "osim_results" / "IK" / f"{trial}_segment_0_ik.mot"
    trc_path = src / "trials" / trial / "markers.trc"
    grf_path = src / "trials" / trial / "grf.mot"

    metadata: dict[str, Any] = {
        "version": VERSION,
        "session": session,
        "source_trial": trial,
        "trial": out_trial.name,
        "source_ik_mot": str(ik_path),
        "source_trc": str(trc_path),
        "source_grf_mot": str(grf_path),
        "assumptions": ASSUMPTIONS,
        "config": asdict(cfg),
    }
    for label, path in (("ik", ik_path), ("trc", trc_path), ("grf", grf_path)):
        if not path.exists():
            metadata.update(qc_status="REJECT", qc_reasons=[f"missing_{label}"])
            return metadata
    if hash_sources:
        metadata["source_sha256"] = {"ik_mot": sha256(ik_path), "trc": sha256(trc_path),
                                     "grf_mot": sha256(grf_path)}

    trc = read_trc(trc_path)
    pos, kin_time, notes = load_kinematics(ik_path, trc)
    notes["kinematics_duration_s"] = float(kin_time[-1] - kin_time[0]) if kin_time.size > 1 else 0.0
    tracks, filled = foot_tracks(trc)
    arrays, report = split_forces(grf_path, tracks, trc["rate_hz"], cfg)

    vel = np.gradient(pos, kin_time, axis=0)
    accel = np.gradient(vel, kin_time, axis=0)
    status, reasons = qc_status(report, notes, arrays["grf"], body_weight_n, cfg)

    motion = out_trial / "Motion"
    raw = motion / "Raw"
    raw.mkdir(parents=True, exist_ok=True)
    to_save = {
        "Pos": pos.astype(np.float32), "Vel": vel.astype(np.float32),
        "Accel": accel.astype(np.float32), "Time_for_pos": kin_time.astype(np.float64),
        "GRF": arrays["grf"].astype(np.float32), "COP": arrays["cop"].astype(np.float32),
        "GRM": arrays["grm"].astype(np.float32), "Time": arrays["time"].astype(np.float64),
        "ContactMask": arrays["contact_mask"],
        "ForceAssignmentConfidence": arrays["confidence"].astype(np.float32),
        "ContaminatedMask": arrays["contaminated_mask"],
    }
    for name, values in to_save.items():
        np.save(motion / f"{name}.npy", values)

    trial_n = out_trial.name.split("_")[-1]
    write_force_mot(raw / f"trial{trial_n}_forces_COP.mot", arrays["time"],
                    arrays["grf"], arrays["cop"], arrays["grm"])
    write_mot(raw / f"trial{trial_n}_kinematics_23dof.mot", kin_time, pos, list(KINEMATIC_23))
    if copy_raw:
        shutil.copy2(ik_path, raw / ik_path.name)
        shutil.copy2(grf_path, raw / f"{trial}_grf.mot")

    metadata.update(report)
    metadata.update({
        "qc_status": status,
        "qc_reasons": reasons,
        "kinematics": notes | {
            "frames": int(pos.shape[0]),
            "columns": list(KINEMATIC_23),
            "units": "radians; metres for pelvis_tx/ty/tz",
        },
        "trc": {"rate_hz": trc["rate_hz"], "frames": trc["n_frames"],
                "declared_frames": trc["declared_frames"], "units": trc["units"],
                "foot_markers_interpolated": filled},
        "body_weight_n": body_weight_n,
        "outputs": {name: list(np.shape(v)) for name, v in to_save.items()},
        "force_columns": ["R_x", "R_y", "R_z", "L_x", "L_y", "L_z"],
    })
    (motion / "force_plate_assignment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def build_session(src_root: Path, out_root: Path, session: str, cfg: Config,
                  copy_raw: bool, hash_sources: bool, overwrite: bool) -> dict:
    src = src_root / session
    dst = out_root / session
    summary: dict[str, Any] = {"session": session, "trials": [], "errors": []}

    dst.mkdir(parents=True, exist_ok=True)
    md = patient_md(session, src / "_subject.json")
    (dst / "Patient_MD.json").write_text(json.dumps(md, indent=4) + "\n")
    model_src = src / "OpenSimData" / "Model" / "LaiUhlrich2022_scaled.osim"
    model_dst = dst / "OpenSimModel.osim"
    if overwrite or not model_dst.exists():
        shutil.copy2(model_src, model_dst)
    summary["patient_md"] = md
    body_weight_n = md["Mass_kg"] * 9.8067

    trials = sorted((p.name for p in (src / "trials").iterdir()
                     if p.is_dir() and TRIAL_RE.match(p.name)),
                    key=lambda t: int(TRIAL_RE.match(t).group(1)))
    for trial in trials:
        number = int(TRIAL_RE.match(trial).group(1))
        out_trial = dst / f"Trial_{number}"
        marker = out_trial / "Motion" / "force_plate_assignment.json"
        if marker.exists() and not overwrite:
            summary["trials"].append(json.loads(marker.read_text()) | {"skipped_existing": True})
            continue
        try:
            meta = process_trial(src, session, trial, out_trial, body_weight_n,
                                 cfg, copy_raw, hash_sources)
        except Exception as exc:                                       # noqa: BLE001
            summary["errors"].append({"trial": trial, "error": f"{type(exc).__name__}: {exc}"})
            continue
        summary["trials"].append(meta)
    return summary


def _compact(meta: dict) -> dict:
    contaminated = [
        {"plate": e["plate"], "assigned_foot": e["assignment"],
         "start_time_s": e["start_time_s"], "end_time_s": e["end_time_s"],
         "side_consistency": e.get("side_consistency"),
         "p95_min_distance_m": e.get("p95_min_distance_m"),
         "contaminated_regions": e["contaminated_regions"]}
        for e in meta.get("contact_events", []) if e.get("contaminated")
    ]
    return {
        "session": meta["session"], "trial": meta.get("trial"),
        "source_trial": meta.get("source_trial"),
        "qc_status": meta.get("qc_status"), "qc_reasons": meta.get("qc_reasons", []),
        "contaminated": meta.get("contaminated", False),
        "contaminated_frame_fraction": meta.get("contaminated_frame_fraction", 0.0),
        "contact_summary": meta.get("contact_summary", {}),
        "both_plates_same_foot_regions": meta.get("both_plates_same_foot_regions", []),
        "contaminated_contacts": contaminated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=Path(str(dataset("Datasets_Local", "Bertaux_withForces"))))
    parser.add_argument("--output-root", type=Path, default=Path("Hip_OA"))
    parser.add_argument("--prefixes", nargs="+", default=["HOA", "HEA"])
    parser.add_argument("--only", nargs="*", default=[], help="explicit session names")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-sources", action="store_true")
    parser.add_argument("--no-copy-raw", action="store_true",
                        help="skip copying the source IK/GRF .mot files into Motion/Raw")
    args = parser.parse_args()

    cfg = Config()
    sessions = sorted(args.only) or sorted(
        d.name for d in args.source_root.iterdir()
        if d.is_dir() and d.name.startswith(tuple(args.prefixes)))
    if args.limit is not None:
        sessions = sessions[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"building {len(sessions)} sessions -> {args.output_root}", flush=True)

    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(build_session, args.source_root, args.output_root, session, cfg,
                        not args.no_copy_raw, args.hash_sources, args.overwrite): session
            for session in sessions
        }
        for i, future in enumerate(as_completed(futures), start=1):
            session = futures[future]
            try:
                summaries.append(future.result())
            except Exception as exc:                                   # noqa: BLE001
                summaries.append({"session": session, "trials": [],
                                  "errors": [{"trial": None, "error": f"{type(exc).__name__}: {exc}"}]})
            if i % 10 == 0 or i == len(sessions):
                print(f"  {i}/{len(sessions)} sessions", flush=True)

    summaries.sort(key=lambda s: s["session"])
    trials = [t for s in summaries for t in s["trials"]]
    counts = {status: sum(t.get("qc_status") == status for t in trials)
              for status in ("PASS", "REVIEW", "REJECT")}
    contaminated = [_compact(t) for t in trials if t.get("contaminated")]
    manifest = {
        "version": VERSION,
        "source_root": str(args.source_root.resolve()),
        "dataset_root": str(args.output_root.resolve()),
        "config": asdict(cfg),
        "assumptions": ASSUMPTIONS,
        "column_conventions": {
            "Pos/Vel/Accel.npy": list(KINEMATIC_23),
            "kinematics_units": "radians; metres for pelvis_tx/ty/tz",
            "GRF/COP/GRM.npy": ["R_x", "R_y", "R_z", "L_x", "L_y", "L_z"],
            "force_units": "GRF N, COP m, GRM N*m (free moment)",
            "frame": "OpenSim Y-up world frame, shared by markers.trc and grf.mot",
            "timebases": ("Time_for_pos.npy = kinematics grid from the TRC DataRate (100 Hz); "
                          "Time.npy = force grid from grf.mot (1000 Hz)"),
        },
        "sessions": len(summaries),
        "trials_written": len(trials),
        "trials_failed": sum(len(s["errors"]) for s in summaries),
        "qc_counts": counts,
        "contaminated_trials": len(contaminated),
        "contaminated_trial_fraction": (len(contaminated) / len(trials)) if trials else 0.0,
        "per_session": [
            {"session": s["session"],
             "trials": len(s["trials"]),
             "qc_counts": {k: sum(t.get("qc_status") == k for t in s["trials"])
                           for k in ("PASS", "REVIEW", "REJECT")},
             "contaminated_trials": [t.get("trial") for t in s["trials"] if t.get("contaminated")],
             "errors": s["errors"]}
            for s in summaries
        ],
        "contaminated_detail": contaminated,
    }
    path = args.output_root / "extraction_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in
                      ("sessions", "trials_written", "trials_failed", "qc_counts",
                       "contaminated_trials", "contaminated_trial_fraction")}, indent=2))
    print(f"manifest -> {path}")
    failures = [e for s in summaries for e in s["errors"]]
    if failures:
        print(json.dumps({"failures": failures[:20], "n_failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
