#!/usr/bin/env python3
"""Compare OpenSim inverse dynamics vs MJX ID for OpenCap validation subjects.

Time-aligns external OpenSim ID ``walking#.sto`` files onto the MoCap ground-truth
timeline (``Trial/MoCap/Time.npy``) from ``ProcessData.py --OC_Mocap``. All MJX
arrays (``pos_mjx.npy``, ``ID_GT_MJX.npy``) are read exclusively from ``MoCap/`` —
never from ``ProcessedData/`` (OpenCap motion inputs).

Examples
--------
    # All subjects + dataset summary folder
    python scripts/opensim/compare_opensim_mjx_id_opencap.py \\
        --dataset-root OpenCapSubjects_Filt --all-subjects

    # Single subject
    python scripts/opensim/compare_opensim_mjx_id_opencap.py \\
        --dataset-root OpenCapSubjects_Filt --subject subject5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from batch_opensim_inverse_dynamics import (  # noqa: E402
    _OPENSIM_TO_MJX_IDX,
    read_storage_file,
)

# Hip / knee / ankle rotational DOFs in display order (label, OpenSim coord, MJX qpos idx).
HIP_KNEE_ANKLE_DOFS: list[tuple[str, str, int]] = [
    ("R Hip Flexion", "hip_flexion_r", 6),
    ("R Hip Adduction", "hip_adduction_r", 7),
    ("R Hip Rotation", "hip_rotation_r", 8),
    ("R Knee", "knee_angle_r", 11),
    ("R Ankle", "ankle_angle_r", 14),
    ("L Hip Flexion", "hip_flexion_l", 17),
    ("L Hip Adduction", "hip_adduction_l", 18),
    ("L Hip Rotation", "hip_rotation_l", 19),
    ("L Knee", "knee_angle_l", 22),
    ("L Ankle", "ankle_angle_l", 25),
]

# OpenSim IK ``.mot`` column names → MJX qpos index (degrees in file → radians in plots).
IK_COORD_TO_MJX: dict[str, int] = {
    "hip_flexion_r": 6,
    "hip_adduction_r": 7,
    "hip_rotation_r": 8,
    "knee_angle_r": 11,
    "ankle_angle_r": 14,
    "hip_flexion_l": 17,
    "hip_adduction_l": 18,
    "hip_rotation_l": 19,
    "knee_angle_l": 22,
    "ankle_angle_l": 25,
}


@dataclass
class TrialBundle:
    trial_name: str
    trial_dir: Path
    mocap_dir: Path
    time_s: np.ndarray
    mjx_angles_rad: np.ndarray
    ik_angles_rad: np.ndarray | None
    mjx_id_nm: np.ndarray
    os_id_nm: np.ndarray
    os_sto_path: Path
    ik_mot_path: Path | None
    mjx_id_path: Path
    pos_mjx_path: Path
    time_path: Path
    alignment_notes: list[str]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resample_matrix(
    values: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source_time = np.asarray(source_time, dtype=np.float64).reshape(-1)
    target_time = np.asarray(target_time, dtype=np.float64).reshape(-1)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(source_time) != values.shape[0]:
        raise ValueError(
            f"time/values length mismatch: time={len(source_time)} values={values.shape[0]}"
        )
    if len(source_time) == len(target_time) and np.allclose(source_time, target_time, atol=1e-4):
        return values.copy()
    out = np.empty((len(target_time), values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(
            target_time,
            source_time,
            values[:, col],
            left=float(values[0, col]),
            right=float(values[-1, col]),
        )
    return out


def _find_walking_sto(trial_dir: Path) -> Path | None:
    for trial_num in ("1", "2", "3", "4"):
        if trial_dir.name == f"Trial_{trial_num}":
            candidate = trial_dir / f"walking{trial_num}.sto"
            if candidate.exists():
                return candidate
    candidates = sorted(p for p in trial_dir.glob("walking*.sto") if "_ik" not in p.name.lower())
    return candidates[0] if candidates else None


def _find_walking_ik_mot(trial_dir: Path) -> Path | None:
    for trial_num in ("1", "2", "3", "4"):
        if trial_dir.name == f"Trial_{trial_num}":
            for candidate in (
                trial_dir / f"walking{trial_num}.mot",
                trial_dir / "MoCap" / "Raw" / f"walking{trial_num}.mot",
            ):
                if candidate.exists():
                    return candidate
    search_dirs = [trial_dir, trial_dir / "MoCap" / "Raw"]
    for folder in search_dirs:
        if not folder.exists():
            continue
        for candidate in sorted(folder.glob("walking*.mot")):
            name = candidate.name.lower()
            if "_ik" in name or "_id" in name:
                continue
            return candidate
    return None


# Maximum residual lag (seconds) searched after zeroing both time origins. Origin-zeroing
# removes the gross clock offset; only a small trim-induced lag should remain, so this is
# kept under a stride to avoid locking onto a neighbouring gait cycle.
ALIGN_MAX_LAG_S = 1.0


def _alignment_axes(
    source_time: np.ndarray, target_time: np.ndarray, align_mode: str, lag_dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map (source_time, target_time) into the frame implied by an alignment transform."""
    if align_mode == "absolute":
        return source_time, target_time
    return source_time - source_time[0], (target_time - target_time[0]) + lag_dt


def _corr_at_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Pearson r between ``a`` shifted by ``lag`` samples and ``b`` (lag>0: a delayed)."""
    if lag > 0:
        aa, bb = a[lag:], b[: len(b) - lag]
    elif lag < 0:
        aa, bb = a[: len(a) + lag], b[-lag:]
    else:
        aa, bb = a, b
    m = np.isfinite(aa) & np.isfinite(bb)
    if np.count_nonzero(m) < 8:
        return float("nan")
    av, bv = aa[m], bb[m]
    if np.std(av) < 1e-9 or np.std(bv) < 1e-9:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _dof_weights(mjx_id: np.ndarray) -> dict[int, float]:
    """Variance of each hip/knee/ankle MJX torque channel (alignment weight)."""
    weights: dict[int, float] = {}
    for _label, _coord, ch in HIP_KNEE_ANKLE_DOFS:
        b = mjx_id[:, ch]
        m = np.isfinite(b)
        weights[ch] = float(np.var(b[m])) if np.count_nonzero(m) > 8 else 0.0
    return weights


def _weighted_corr(a_mat: np.ndarray, b_mat: np.ndarray, weights: dict[int, float], lag: int) -> float:
    """Variance-weighted mean correlation across DOFs at an integer sample ``lag``."""
    s, wsum = 0.0, 0.0
    for _label, _coord, ch in HIP_KNEE_ANKLE_DOFS:
        w = weights.get(ch, 0.0)
        if w <= 0:
            continue
        r = _corr_at_lag(a_mat[:, ch], b_mat[:, ch], lag)
        if not np.isfinite(r):
            continue
        s += w * r
        wsum += w
    return s / wsum if wsum > 0 else float("nan")


def _alignment_score(a_mat: np.ndarray, b_mat: np.ndarray, weights: dict[int, float]) -> float:
    """Variance-weighted mean correlation across DOFs at zero lag (agreement metric)."""
    return _weighted_corr(a_mat, b_mat, weights, 0)


def _best_lag(a_coarse: np.ndarray, b_ref: np.ndarray, weights: dict[int, float], max_lag: int) -> int:
    """Integer sample lag maximizing weighted DOF correlation; ties favour smaller |lag|."""
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        score = _weighted_corr(a_coarse, b_ref, weights, lag)
        if not np.isfinite(score):
            continue
        if score > best_score + 1e-4 or (score > best_score - 1e-4 and abs(lag) < abs(best_lag)):
            best_score, best_lag = score, lag
    return best_lag


def _refine_lag(a_coarse: np.ndarray, b_ref: np.ndarray, weights: dict[int, float],
                lag: int, max_lag: int) -> float:
    """Parabolic sub-sample refinement of the integer ``lag`` from its two neighbours."""
    if abs(lag) >= max_lag:
        return float(lag)
    s0 = _weighted_corr(a_coarse, b_ref, weights, lag)
    sm = _weighted_corr(a_coarse, b_ref, weights, lag - 1)
    sp = _weighted_corr(a_coarse, b_ref, weights, lag + 1)
    if not (np.isfinite(s0) and np.isfinite(sm) and np.isfinite(sp)):
        return float(lag)
    denom = sm - 2.0 * s0 + sp
    if abs(denom) < 1e-9:
        return float(lag)
    delta = 0.5 * (sm - sp) / denom
    if not np.isfinite(delta) or abs(delta) > 1.0:
        return float(lag)
    return lag + delta


def _apply_transform(
    source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray,
    align_mode: str, lag_dt: float,
) -> np.ndarray:
    """Resample ``values`` onto ``target_time`` under an alignment transform."""
    src_t, tgt_t = _alignment_axes(source_time, target_time, align_mode, lag_dt)
    return _resample_matrix(values, src_t, tgt_t)


def _read_ik_raw(mot_path: Path) -> tuple[np.ndarray | None, np.ndarray | None, list[str], list[str]]:
    """Return (source_time, raw (T,31) angles in radians, loaded coords, notes) from an IK .mot."""
    try:
        columns, rows = read_storage_file(mot_path)
    except Exception as exc:
        return None, None, [], [f"failed to read IK mot: {exc}"]
    if "time" not in columns:
        return None, None, [], ["IK mot missing time column"]

    data = np.asarray(rows, dtype=np.float64)
    col_idx = {name: idx for idx, name in enumerate(columns)}
    source_time = data[:, col_idx["time"]]
    out = np.full((len(source_time), 31), np.nan, dtype=np.float64)
    loaded: list[str] = []
    for coord, mjx_idx in IK_COORD_TO_MJX.items():
        if coord in col_idx:
            out[:, mjx_idx] = np.deg2rad(data[:, col_idx[coord]])
            loaded.append(coord)
    if not loaded:
        return source_time, None, [], ["no hip/knee/ankle coordinates found in IK mot"]
    return source_time, out, loaded, []


def _estimate_kinematic_alignment(
    ik_source_time: np.ndarray, ik_raw31: np.ndarray,
    target_time: np.ndarray, mjx_angles: np.ndarray,
) -> tuple[str, float, list[str]]:
    """Find the time transform aligning raw-mocap IK angles to the filtered MJX kinematics.

    Filtered MJX (``pos_mjx``) and raw-mocap IK are the *same joint angles* (one filtered),
    so they cross-correlate far more sharply than the engine-dependent ID torques. We zero
    both time origins, find the integer lag maximizing the variance-weighted hip/knee/ankle
    angle correlation, refine it to sub-sample precision with a parabolic fit, and keep that
    over the legacy absolute-clock resample only if it agrees better."""
    n_target = len(target_time)
    weights = _dof_weights(mjx_angles)

    ik_abs = _resample_matrix(ik_raw31, ik_source_time, target_time)
    score_abs = _alignment_score(ik_abs, mjx_angles, weights)

    src_z = ik_source_time - ik_source_time[0]
    tgt_z = target_time - target_time[0]
    dt = float(np.median(np.diff(target_time))) if n_target > 1 else 0.01
    max_lag = min(max(1, int(round(ALIGN_MAX_LAG_S / dt))) if dt > 0 else 1, n_target // 2)
    ik_coarse = _resample_matrix(ik_raw31, src_z, tgt_z)
    lag_int = _best_lag(ik_coarse, mjx_angles, weights, max_lag) if max_lag >= 1 else 0
    lag_frac = _refine_lag(ik_coarse, mjx_angles, weights, lag_int, max_lag)
    ik_lag = _resample_matrix(ik_raw31, src_z, tgt_z + lag_frac * dt)
    score_lag = _alignment_score(ik_lag, mjx_angles, weights)

    use_xcorr = np.isfinite(score_lag) and (not np.isfinite(score_abs) or score_lag > score_abs + 1e-4)
    if use_xcorr:
        mode, lag_dt = "xcorr", lag_frac * dt
    else:
        mode, lag_dt = "absolute", 0.0
    notes = [
        (f"kinematic alignment={mode} (angle score_abs={score_abs:.3f}, "
         f"score_xcorr={score_lag:.3f}, lag={lag_frac:.2f} samples / {lag_dt * 1e3:.1f} ms)"),
    ]
    return mode, lag_dt, notes


def _read_opensim_id_raw(sto_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (source_time, raw (T,31) moments, loaded column names) from a .sto."""
    columns, rows = read_storage_file(sto_path)
    data = np.asarray(rows, dtype=np.float64)
    if "time" not in columns:
        raise ValueError(f"{sto_path} missing time column")

    col_idx = {name: idx for idx, name in enumerate(columns)}
    source_time = data[:, col_idx["time"]]
    out = np.full((len(source_time), 31), np.nan, dtype=np.float64)
    loaded_cols: list[str] = []
    for coord, mjx_idx in _OPENSIM_TO_MJX_IDX.items():
        moment_col = f"{coord}_moment"
        if moment_col in col_idx:
            out[:, mjx_idx] = data[:, col_idx[moment_col]]
            loaded_cols.append(moment_col)

    if not loaded_cols:
        raise ValueError(f"{sto_path} has no expected *_moment columns")
    return source_time, out, loaded_cols


def _load_opensim_id_nm(
    sto_path: Path,
    target_time: np.ndarray,
    align_mode: str,
    lag_dt: float,
) -> tuple[np.ndarray, list[str]]:
    """Resample OpenSim ID onto the MoCap timebase using the kinematics-derived transform.

    The OpenSim ID ``.sto`` shares the raw-mocap clock with the IK ``.mot`` that the
    kinematic alignment was estimated from, so we simply reuse that transform here."""
    source_time, raw_os, loaded_cols = _read_opensim_id_raw(sto_path)
    os_out = _apply_transform(source_time, raw_os, target_time, align_mode, lag_dt)
    notes = [
        f"OpenSim ID resampled from {len(source_time)} -> {len(target_time)} frames "
        f"({len(loaded_cols)} moment columns), align={align_mode}, lag={lag_dt * 1e3:.1f} ms",
    ]
    return os_out, notes


def _load_trial_bundle(trial_dir: Path) -> TrialBundle:
    """Load MoCap ground truth and time-align external OpenSim ID onto MoCap/Time.npy."""
    mocap_dir = trial_dir / "MoCap"
    if not mocap_dir.exists():
        raise FileNotFoundError(f"Missing MoCap folder: {mocap_dir}")

    time_path = mocap_dir / "Time.npy"
    pos_mjx_path = mocap_dir / "pos_mjx.npy"
    id_path = mocap_dir / "ID_GT_MJX.npy"

    for required in (time_path, pos_mjx_path, id_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing required MoCap ground-truth file: {required}. "
                "Re-run ProcessData.py with --OC_Mocap for this trial."
            )

    sto_path = _find_walking_sto(trial_dir)
    if sto_path is None:
        raise FileNotFoundError(f"No walking*.sto OpenSim ID file in {trial_dir}")

    time_s = np.load(time_path).astype(np.float64).reshape(-1)
    mjx_angles = np.load(pos_mjx_path).astype(np.float64)
    mjx_id = np.load(id_path).astype(np.float64)

    notes: list[str] = [
        f"MoCap GT timebase: {time_path}",
        f"MoCap GT kinematics: {pos_mjx_path}",
        f"MoCap GT MJX ID: {id_path}",
    ]
    n_time = len(time_s)
    n_pos = mjx_angles.shape[0]
    n_id = mjx_id.shape[0]
    n = min(n_time, n_pos, n_id)
    if n == 0:
        raise ValueError(
            f"Empty MoCap arrays for {trial_dir.name} "
            f"(time={n_time}, pos={n_pos}, id={n_id})"
        )
    if n != n_time or n != n_pos or n != n_id:
        notes.append(
            f"aligned MoCap arrays to common length {n} "
            f"(time={n_time}, pos={n_pos}, id={n_id})"
        )
        time_s = time_s[:n]
        mjx_angles = mjx_angles[:n]
        mjx_id = mjx_id[:n]

    # Determine the time-alignment transform from KINEMATICS only: raw-mocap IK angles vs
    # the filtered MJX kinematics (pos_mjx). The OpenSim ID .sto shares the IK .mot clock,
    # so the same transform aligns the torques.
    ik_mot_path = _find_walking_ik_mot(trial_dir)
    align_mode, lag_dt = "absolute", 0.0
    ik_angles = None
    if ik_mot_path is not None:
        ik_source_time, ik_raw31, _ik_loaded, ik_read_notes = _read_ik_raw(ik_mot_path)
        notes.extend(ik_read_notes)
        if ik_raw31 is not None:
            align_mode, lag_dt, est_notes = _estimate_kinematic_alignment(
                ik_source_time, ik_raw31, time_s, mjx_angles
            )
            notes.extend(est_notes)
            ik_angles = _apply_transform(ik_source_time, ik_raw31, time_s, align_mode, lag_dt)
            notes.append(
                f"IK mot resampled from {len(ik_source_time)} -> {len(time_s)} frames (align={align_mode})"
            )
    else:
        notes.append("no raw mocap IK .mot found; OpenSim ID uses absolute clock (no kinematic alignment)")

    os_id, os_notes = _load_opensim_id_nm(sto_path, time_s, align_mode, lag_dt)
    notes.extend(os_notes)

    meta = _load_json(mocap_dir / "Trial_Processing_Information.json") or {}
    notes.append(f"MoCap frames={meta.get('n_frames', n)} pipeline={meta.get('pipeline', 'unknown')}")

    return TrialBundle(
        trial_name=trial_dir.name,
        trial_dir=trial_dir,
        mocap_dir=mocap_dir,
        time_s=time_s,
        mjx_angles_rad=mjx_angles,
        ik_angles_rad=ik_angles,
        mjx_id_nm=mjx_id,
        os_id_nm=os_id,
        os_sto_path=sto_path,
        ik_mot_path=ik_mot_path,
        mjx_id_path=id_path,
        pos_mjx_path=pos_mjx_path,
        time_path=time_path,
        alignment_notes=notes,
    )


def _masked_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"mae": math.nan, "rmse": math.nan, "r": math.nan, "bias_os_minus_mjx": math.nan, "n": 0}
    diff = a[mask] - b[mask]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    if np.std(a[mask]) < 1e-9 or np.std(b[mask]) < 1e-9:
        r = math.nan
    else:
        r = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return {"mae": mae, "rmse": rmse, "r": r, "bias_os_minus_mjx": bias, "n": int(np.count_nonzero(mask))}


def _plot_trial(bundle: TrialBundle, output_dir: Path) -> dict[str, Any]:
    n_dof = len(HIP_KNEE_ANKLE_DOFS)
    fig, axes = plt.subplots(n_dof, 2, figsize=(16, 2.2 * n_dof), sharex="col")
    if n_dof == 1:
        axes = np.array([axes])

    metrics: dict[str, Any] = {
        "subject": bundle.trial_dir.parent.name,
        "trial": bundle.trial_name,
        "ground_truth_source": "MoCap",
        "mocap_dir": str(bundle.mocap_dir),
        "time_path": str(bundle.time_path),
        "pos_mjx_path": str(bundle.pos_mjx_path),
        "mjx_id_path": str(bundle.mjx_id_path),
        "os_sto_path": str(bundle.os_sto_path),
        "ik_mot_path": str(bundle.ik_mot_path) if bundle.ik_mot_path else None,
        "n_frames": int(len(bundle.time_s)),
        "time_range_s": [float(bundle.time_s[0]), float(bundle.time_s[-1])],
        "alignment_notes": bundle.alignment_notes,
        "kinematics": {},
        "id_torques": {},
    }

    for row, (label, _coord, mjx_idx) in enumerate(HIP_KNEE_ANKLE_DOFS):
        ax_kin = axes[row, 0]
        ax_id = axes[row, 1]

        mjx_angle = np.rad2deg(bundle.mjx_angles_rad[:, mjx_idx])
        ax_kin.plot(bundle.time_s, mjx_angle, color="#1f77b4", linewidth=1.5, label="MoCap/ pos_mjx (GT)")
        if bundle.ik_angles_rad is not None:
            ik_angle = np.rad2deg(bundle.ik_angles_rad[:, mjx_idx])
            ax_kin.plot(
                bundle.time_s,
                ik_angle,
                color="#ff7f0e",
                linewidth=1.0,
                alpha=0.85,
                linestyle="--",
                label="Raw mocap IK .mot",
            )
            metrics["kinematics"][label] = _masked_metrics(ik_angle, mjx_angle)

        os_torque = bundle.os_id_nm[:, mjx_idx]
        mjx_torque = bundle.mjx_id_nm[:, mjx_idx]
        ax_id.plot(bundle.time_s, os_torque, color="#d62728", linewidth=1.5, label="OpenSim ID (.sto)")
        ax_id.plot(
            bundle.time_s,
            mjx_torque,
            color="#2ca02c",
            linewidth=1.2,
            alpha=0.9,
            label="MoCap/ ID_GT_MJX",
        )
        metrics["id_torques"][label] = _masked_metrics(os_torque, mjx_torque)

        ax_kin.set_ylabel(f"{label}\n(deg)")
        ax_id.set_ylabel(f"{label}\n(Nm)")
        if row == 0:
            ax_kin.set_title("Kinematics")
            ax_id.set_title("Inverse Dynamics Torques")
        if row == n_dof - 1:
            ax_kin.set_xlabel("Time (s)")
            ax_id.set_xlabel("Time (s)")
        ax_kin.grid(True, alpha=0.25)
        ax_id.grid(True, alpha=0.25)
        ax_kin.legend(loc="upper right", fontsize=7)
        ax_id.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"{bundle.trial_dir.parent.name}/{bundle.trial_name} — OpenSim vs MoCap GT (MoCap/ timebase)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    png_path = output_dir / f"{bundle.trial_dir.parent.name}_{bundle.trial_name}_os_vs_mjx_hip_knee_ankle.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    # Compact summary panel: torque MAE by DOF
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    labels = [item[0] for item in HIP_KNEE_ANKLE_DOFS]
    mae_vals = [metrics["id_torques"][lab]["mae"] for lab in labels]
    r_vals = [metrics["id_torques"][lab]["r"] for lab in labels]
    x = np.arange(len(labels))
    ax2.bar(x, mae_vals, color="#6baed6")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("Torque MAE (Nm)")
    ax2.set_title(f"{bundle.trial_dir.parent.name}/{bundle.trial_name} — |OpenSim − MJX| torque MAE")
    ax2.grid(True, axis="y", alpha=0.25)
    for i, r_val in enumerate(r_vals):
        if math.isfinite(r_val):
            ax2.text(i, mae_vals[i], f"r={r_val:.2f}", ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    mae_png = output_dir / f"{bundle.trial_dir.parent.name}_{bundle.trial_name}_torque_mae_summary.png"
    fig2.savefig(mae_png, dpi=160)
    plt.close(fig2)

    metrics["figure_paths"] = [str(png_path), str(mae_png)]
    return metrics


def _plot_subject_overlay(all_metrics: list[dict[str, Any]], subject: str, output_dir: Path) -> None:
    if not all_metrics:
        return
    fig, axes = plt.subplots(len(HIP_KNEE_ANKLE_DOFS), 1, figsize=(12, 2.0 * len(HIP_KNEE_ANKLE_DOFS)), sharex=True)
    if len(HIP_KNEE_ANKLE_DOFS) == 1:
        axes = [axes]
    trial_names = [m["trial"] for m in all_metrics]
    x = np.arange(len(trial_names))
    for row, (label, _coord, _idx) in enumerate(HIP_KNEE_ANKLE_DOFS):
        ax = axes[row]
        mae_vals = [metrics["id_torques"][label]["mae"] for metrics in all_metrics]
        r_vals = [metrics["id_torques"][label]["r"] for metrics in all_metrics]
        ax.bar(x, mae_vals, color="#6baed6", alpha=0.9)
        ax.set_ylabel("MAE (Nm)")
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)
        for i, (mae, r_val) in enumerate(zip(mae_vals, r_vals)):
            if math.isfinite(r_val):
                ax.text(i, mae, f"r={r_val:.2f}", ha="center", va="bottom", fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(trial_names)
    fig.suptitle(f"{subject} — OpenSim vs MJX torque MAE across trials", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = output_dir / f"{subject}_all_trials_torque_mae.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)


DEFAULT_SUMMARY_DIRNAME = "OpenSimVsMJX_ID_Summary"


def _discover_subject_dirs(
    dataset_root: Path,
    *,
    subject: str | None,
    subjects: str | None,
    all_subjects: bool,
) -> list[Path]:
    if subject and (subjects or all_subjects):
        raise SystemExit("Use only one of --subject, --subjects, or --all-subjects.")

    if subject:
        subject_dir = dataset_root / subject
        if not subject_dir.is_dir():
            raise SystemExit(f"Subject folder not found: {subject_dir}")
        return [subject_dir]

    if subjects:
        names = [s.strip() for s in subjects.split(",") if s.strip()]
        if not names:
            raise SystemExit("--subjects was provided but empty.")
        subject_dirs: list[Path] = []
        for name in names:
            subject_dir = dataset_root / name
            if not subject_dir.is_dir():
                raise SystemExit(f"Subject folder not found: {subject_dir}")
            subject_dirs.append(subject_dir)
        return subject_dirs

    if all_subjects or (subject is None and subjects is None):
        subject_dirs = sorted(
            p for p in dataset_root.iterdir()
            if p.is_dir() and p.name.lower().startswith("subject")
        )
        if not subject_dirs:
            raise SystemExit(f"No subject folders found under {dataset_root}")
        return subject_dirs

    raise SystemExit("Specify --subject, --subjects, or --all-subjects.")


def _trial_dirs_for_subject(subject_dir: Path, trial_filter: str | None) -> list[Path]:
    trial_dirs = sorted(
        p for p in subject_dir.iterdir() if p.is_dir() and p.name.startswith("Trial_")
    )
    if trial_filter:
        trial_dirs = [p for p in trial_dirs if p.name == trial_filter]
    return trial_dirs


def _agg_scalar(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "n": int(arr.size),
    }


def _flatten_trial_rows(trial_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trial_metrics:
        subject = trial["subject"]
        trial_name = trial["trial"]
        for dof_label, stats in trial["id_torques"].items():
            rows.append(
                {
                    "subject": subject,
                    "trial": trial_name,
                    "trial_id": f"{subject}/{trial_name}",
                    "dof": dof_label,
                    "mae_nm": stats["mae"],
                    "rmse_nm": stats["rmse"],
                    "r": stats["r"],
                    "bias_os_minus_mjx_nm": stats["bias_os_minus_mjx"],
                    "n_frames": stats["n"],
                }
            )
    return rows


def _aggregate_by_dof(trial_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, _coord, _idx in HIP_KNEE_ANKLE_DOFS:
        mae_vals = [t["id_torques"][label]["mae"] for t in trial_metrics if label in t["id_torques"]]
        rmse_vals = [t["id_torques"][label]["rmse"] for t in trial_metrics if label in t["id_torques"]]
        r_vals = [t["id_torques"][label]["r"] for t in trial_metrics if label in t["id_torques"]]
        bias_vals = [
            t["id_torques"][label]["bias_os_minus_mjx"]
            for t in trial_metrics
            if label in t["id_torques"]
        ]
        out[label] = {
            "mae_nm": _agg_scalar(mae_vals),
            "rmse_nm": _agg_scalar(rmse_vals),
            "r": _agg_scalar(r_vals),
            "bias_os_minus_mjx_nm": _agg_scalar(bias_vals),
        }
    return out


def _aggregate_by_subject(subject_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for subject_result in subject_results:
        subject = subject_result["subject"]
        trials = subject_result.get("trials", [])
        out[subject] = {
            "n_trials": len(trials),
            "n_failures": len(subject_result.get("failures", [])),
            "per_dof": _aggregate_by_dof(trials),
        }
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_dataset_per_dof_boxplot(trial_metrics: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [item[0] for item in HIP_KNEE_ANKLE_DOFS]
    data = [
        [t["id_torques"][label]["mae"] for t in trial_metrics if label in t["id_torques"]]
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Torque MAE (Nm)")
    ax.set_title(f"OpenSim vs MoCap/ ID_GT_MJX — torque MAE by DOF (n={len(trial_metrics)} trials)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_per_dof_mae_boxplot.png", dpi=160)
    plt.close(fig)


def _plot_dataset_per_subject_heatmap(subject_aggs: dict[str, dict[str, Any]], output_dir: Path) -> None:
    subjects = sorted(subject_aggs)
    labels = [item[0] for item in HIP_KNEE_ANKLE_DOFS]
    mat = np.full((len(subjects), len(labels)), np.nan, dtype=np.float64)
    for i, subject in enumerate(subjects):
        per_dof = subject_aggs[subject]["per_dof"]
        for j, label in enumerate(labels):
            mat[i, j] = per_dof[label]["mae_nm"]["mean"]

    fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * len(subjects))))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_title("Mean torque MAE (Nm) by subject and DOF\nOpenSim ID (.sto) vs MoCap/ ID_GT_MJX")
    fig.colorbar(im, ax=ax, label="MAE (Nm)")
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_per_subject_dof_mae_heatmap.png", dpi=160)
    plt.close(fig)


def _plot_dataset_per_subject_mean_mae(subject_aggs: dict[str, dict[str, Any]], output_dir: Path) -> None:
    subjects = sorted(subject_aggs)
    mean_mae: list[float] = []
    for subject in subjects:
        vals = [
            subject_aggs[subject]["per_dof"][label]["mae_nm"]["mean"]
            for label, _coord, _idx in HIP_KNEE_ANKLE_DOFS
        ]
        mean_mae.append(float(np.nanmean(vals)))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(subjects))
    ax.bar(x, mean_mae, color="#6baed6")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylabel("Mean torque MAE (Nm)")
    ax.set_title("Mean OpenSim vs MoCap/ ID torque MAE across hip/knee/ankle DOFs")
    ax.grid(True, axis="y", alpha=0.25)
    for i, val in enumerate(mean_mae):
        if math.isfinite(val):
            ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_per_subject_mean_mae.png", dpi=160)
    plt.close(fig)


def _plot_dataset_per_dof_mean_with_std(dataset_agg: dict[str, Any], output_dir: Path) -> None:
    labels = [item[0] for item in HIP_KNEE_ANKLE_DOFS]
    means = [dataset_agg["per_dof"][label]["mae_nm"]["mean"] for label in labels]
    stds = [dataset_agg["per_dof"][label]["mae_nm"]["std"] for label in labels]
    rs = [dataset_agg["per_dof"][label]["r"]["mean"] for label in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#6baed6", alpha=0.9, ecolor="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Torque MAE (Nm)")
    ax.set_title(
        f"Dataset mean ± SD torque MAE by DOF "
        f"(n={dataset_agg['n_trials']} trials, {dataset_agg['n_subjects']} subjects)"
    )
    ax.grid(True, axis="y", alpha=0.25)
    for i, (mae, r_val) in enumerate(zip(means, rs)):
        if math.isfinite(r_val):
            ax.text(i, mae, f"r={r_val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_per_dof_mean_mae_std.png", dpi=160)
    plt.close(fig)


def _write_dataset_summary(
    summary_dir: Path,
    dataset_root: Path,
    subject_results: list[dict[str, Any]],
    all_trial_metrics: list[dict[str, Any]],
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    subject_aggs = _aggregate_by_subject(subject_results)
    dataset_agg = {
        "n_subjects": len(subject_results),
        "n_trials": len(all_trial_metrics),
        "per_dof": _aggregate_by_dof(all_trial_metrics),
    }

    flat_rows = _flatten_trial_rows(all_trial_metrics)
    _write_csv(
        summary_dir / "trial_metrics.csv",
        flat_rows,
        fieldnames=[
            "subject",
            "trial",
            "trial_id",
            "dof",
            "mae_nm",
            "rmse_nm",
            "r",
            "bias_os_minus_mjx_nm",
            "n_frames",
        ],
    )

    subject_rows: list[dict[str, Any]] = []
    for subject, agg in sorted(subject_aggs.items()):
        for label, _coord, _idx in HIP_KNEE_ANKLE_DOFS:
            stats = agg["per_dof"][label]["mae_nm"]
            subject_rows.append(
                {
                    "subject": subject,
                    "dof": label,
                    "mean_mae_nm": stats["mean"],
                    "median_mae_nm": stats["median"],
                    "std_mae_nm": stats["std"],
                    "n_trials": stats["n"],
                }
            )
    _write_csv(
        summary_dir / "subject_dof_mean_mae.csv",
        subject_rows,
        fieldnames=["subject", "dof", "mean_mae_nm", "median_mae_nm", "std_mae_nm", "n_trials"],
    )

    payload = {
        "dataset_root": str(dataset_root),
        "ground_truth_source": "MoCap",
        "ground_truth_files": [
            "MoCap/Time.npy",
            "MoCap/pos_mjx.npy",
            "MoCap/ID_GT_MJX.npy",
        ],
        "summary_dir": str(summary_dir),
        "n_subjects": dataset_agg["n_subjects"],
        "n_trials": dataset_agg["n_trials"],
        "dataset_aggregate": dataset_agg,
        "subjects": subject_results,
        "subject_aggregates": subject_aggs,
    }
    (summary_dir / "dataset_opensim_vs_mjx_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    if all_trial_metrics:
        _plot_dataset_per_dof_boxplot(all_trial_metrics, summary_dir)
        _plot_dataset_per_subject_heatmap(subject_aggs, summary_dir)
        _plot_dataset_per_subject_mean_mae(subject_aggs, summary_dir)
        _plot_dataset_per_dof_mean_with_std(dataset_agg, summary_dir)


def process_subject(
    subject_dir: Path,
    *,
    trial_filter: str | None,
    output_dir: Path | None,
) -> dict[str, Any]:
    subject = subject_dir.name
    subject_output = (
        output_dir.resolve()
        if output_dir is not None
        else subject_dir / "OpenSimVsMJX_ID"
    )
    subject_output.mkdir(parents=True, exist_ok=True)

    trial_dirs = _trial_dirs_for_subject(subject_dir, trial_filter)
    if trial_filter and not trial_dirs:
        raise SystemExit(f"Trial not found for {subject}: {trial_filter}")

    all_metrics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    print(f"\n=== {subject} ===")
    print(f"  output_dir : {subject_output}")
    print(f"  trials     : {[p.name for p in trial_dirs]}")

    for trial_dir in trial_dirs:
        print(f"\n--- {subject}/{trial_dir.name} ---")
        try:
            bundle = _load_trial_bundle(trial_dir)
            for note in bundle.alignment_notes:
                print(f"  {note}")
            metrics = _plot_trial(bundle, subject_output)
            all_metrics.append(metrics)
            print(f"  saved: {metrics['figure_paths'][0]}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append({"subject": subject, "trial": trial_dir.name, "error": str(exc)})

    if all_metrics:
        _plot_subject_overlay(all_metrics, subject, subject_output)

    summary = {
        "subject": subject,
        "dataset_root": str(subject_dir.parent.resolve()),
        "ground_truth_source": "MoCap",
        "ground_truth_files": [
            "MoCap/Time.npy",
            "MoCap/pos_mjx.npy",
            "MoCap/ID_GT_MJX.npy",
        ],
        "output_dir": str(subject_output),
        "trials": all_metrics,
        "failures": failures,
    }
    summary_path = subject_output / f"{subject}_opensim_vs_mjx_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote subject summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare external OpenSim ID (.sto) with MoCap/ ground truth "
            "(Time.npy, pos_mjx.npy, ID_GT_MJX.npy). ProcessedData/ is never used."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "OpenCapSubjects_Filt",
        help="Dataset root containing subject folders.",
    )
    subject_group = parser.add_mutually_exclusive_group()
    subject_group.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Process a single subject folder, e.g. subject5.",
    )
    subject_group.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject folders, e.g. subject2,subject5.",
    )
    subject_group.add_argument(
        "--all-subjects",
        action="store_true",
        help="Process every folder under dataset-root whose name starts with 'subject'.",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        help="Optional trial folder name, e.g. Trial_1. Default: all trials.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override per-subject output directory (single-subject runs only).",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help=(
            "Dataset-level summary folder "
            f"(default: <dataset-root>/{DEFAULT_SUMMARY_DIRNAME}; written when >1 subject)."
        ),
    )
    parser.add_argument(
        "--no-dataset-summary",
        action="store_true",
        help="Skip writing the dataset-level summary folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if args.output_dir is not None and (args.all_subjects or args.subjects):
        raise SystemExit("--output-dir can only be used with a single --subject run.")

    subject_dirs = _discover_subject_dirs(
        dataset_root,
        subject=args.subject,
        subjects=args.subjects,
        all_subjects=args.all_subjects,
    )

    print("Comparing OpenSim ID vs MoCap GT")
    print(f"  dataset_root : {dataset_root}")
    print(f"  gt_source    : MoCap/ only (not ProcessedData/)")
    print(f"  subjects     : {[p.name for p in subject_dirs]}")

    subject_results: list[dict[str, Any]] = []
    all_failures: list[dict[str, str]] = []

    for subject_dir in subject_dirs:
        try:
            result = process_subject(
                subject_dir,
                trial_filter=args.trial,
                output_dir=args.output_dir,
            )
            subject_results.append(result)
            all_failures.extend(result.get("failures", []))
        except SystemExit:
            raise
        except Exception as exc:
            print(f"\nFAILED subject {subject_dir.name}: {exc}")
            all_failures.append({"subject": subject_dir.name, "trial": "", "error": str(exc)})

    all_trial_metrics = [trial for result in subject_results for trial in result.get("trials", [])]
    write_dataset_summary = (
        not args.no_dataset_summary
        and len(subject_dirs) > 1
        and bool(all_trial_metrics)
    )
    if write_dataset_summary:
        summary_dir = (
            args.summary_dir.resolve()
            if args.summary_dir is not None
            else dataset_root / DEFAULT_SUMMARY_DIRNAME
        )
        _write_dataset_summary(summary_dir, dataset_root, subject_results, all_trial_metrics)
        print(f"\nWrote dataset summary: {summary_dir}")
    elif len(subject_dirs) > 1 and not args.no_dataset_summary and not all_trial_metrics:
        print("\nNo successful trials; skipped dataset summary.")

    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
