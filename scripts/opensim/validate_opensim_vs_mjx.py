#!/usr/bin/env python3
"""Validate MJX inverse dynamics against OpenSim ID (the field standard).

Both pipelines consume the SAME clean processed kinematics (``pos_mjx``) and GRF, so any
difference in joint moments is attributable to the ID method, not the inputs. This script:

  1. Ensures each trial has an OpenSim ``inverse_dynamics.sto`` (runs ID if missing).
  2. Compares OpenSim moments vs ``ID_GT_MJX.npy`` per DOF, overall, and during stance.
  3. Computes ankle power (W/kg) using a SHARED ``qvel_mjx`` so the difference isolates torque.
  4. Writes per-trial + per-subject ``AccuracyMetrics.json`` and a dataset-level
     ``OpenSimToMJX_Accuracy/`` folder with ``Accuracy_Summary.json`` and summary plots.

Metric conventions mirror the existing codebase:
  - norm_factor = mass_kg * height_m * 9.8067   (TransformerFinal/.../infer_mod_q.py:2184)
  - mae_bwh = (mae / norm_factor) * 100         (infer_mod_q.py:1773)
  - stance via dual_threshold_stance_intervals  (scripts/opensim/ankle_stance_common.py)

Run under the ``opencap-processing`` conda env (needs the OpenSim Python API for ID).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from batch_opensim_inverse_dynamics import (  # noqa: E402
    _OPENSIM_TO_MJX_IDX,
    existing_generation_matches,
    process_trial,
    read_storage_file,
)
from generate_opensim_id_inputs import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_LEFT_BODY,
    DEFAULT_RIGHT_BODY,
    TrialPaths,
    discover_trials,
)
from ankle_stance_common import (  # noqa: E402
    ANKLE_DOF_IDX,
    GRAVITY,
    dual_threshold_stance_intervals,
    stance_peak,
)
from prescribed_accel_id import compute_prescribed_id_31ch  # noqa: E402

# DOF groups (rotational, actuated). Right/left leg DOFs use that leg's stance;
# axial DOFs use the union of both feet in stance.
RIGHT_LEG_DOFS = ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
                  "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r"]
LEFT_LEG_DOFS = ["hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
                 "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l"]
AXIAL_DOFS = ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]
DOF_ORDER = RIGHT_LEG_DOFS + LEFT_LEG_DOFS + AXIAL_DOFS
SCATTER_DOFS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
PCT_GRID = np.linspace(0.0, 100.0, 101)


# --- masked reductions (mirror TransformerFinal/infer.py:1292-1340) -----------
def _valid(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b)


def _mae(a, b, mask):
    m = _valid(a, b, mask)
    return float(np.mean(np.abs(a[m] - b[m]))) if np.any(m) else float("nan")


def _rmse(a, b, mask):
    m = _valid(a, b, mask)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if np.any(m) else float("nan")


def _bias(a, b, mask):  # mean signed error (os - mjx)
    m = _valid(a, b, mask)
    return float(np.mean(a[m] - b[m])) if np.any(m) else float("nan")


def _pearson(a, b, mask):
    m = _valid(a, b, mask)
    if np.count_nonzero(m) < 3:
        return float("nan")
    av, bv = a[m], b[m]
    if np.std(av) < 1e-9 or np.std(bv) < 1e-9:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _nrmse(a, b, mask):
    m = _valid(a, b, mask)
    if not np.any(m):
        return float("nan")
    gt_std = np.std(b[m])
    gt_std = gt_std if gt_std >= 1e-6 else 1.0
    return float(_rmse(a, b, mask) / gt_std)


def _stance_range_bwh(sig, intervals, norm_factor):
    """Torque range at a joint, in %BWxH units.

    For each stance interval take the min and max of the signal; the range is the
    mean-max minus the mean-min across stances, divided by norm_factor and x100."""
    mins, maxs = [], []
    for s, e in intervals:
        seg = np.asarray(sig[max(0, s):e], dtype=np.float64)
        seg = seg[np.isfinite(seg)]
        if seg.size:
            mins.append(float(np.min(seg)))
            maxs.append(float(np.max(seg)))
    if not mins:
        return float("nan")
    rng = float(np.mean(maxs) - np.mean(mins))
    return (rng / norm_factor) * 100.0


def _dof_metrics(os_sig, mjx_sig, mask, norm_factor):
    mae = _mae(os_sig, mjx_sig, mask)
    rmse = _rmse(os_sig, mjx_sig, mask)
    return {
        "mae": mae,
        "rmse": rmse,
        "mae_bwh": (mae / norm_factor) * 100.0,
        "rmse_bwh": (rmse / norm_factor) * 100.0,
        "nrmse": _nrmse(os_sig, mjx_sig, mask),
        "r": _pearson(os_sig, mjx_sig, mask),
        "bias": _bias(os_sig, mjx_sig, mask),
    }


# --- loaders ------------------------------------------------------------------
def load_subject_md(subject_dir: Path) -> tuple[float, float]:
    md = json.loads((subject_dir / "Patient_MD.json").read_text())
    return float(md["Mass_kg"]), float(md["Height_m"])


def load_opensim_31ch(sto_path: Path) -> tuple[np.ndarray, int]:
    """Return (T, 31) OpenSim moments mapped to MJX qpos channels, and frame count."""
    columns, rows = read_storage_file(sto_path)
    data = np.asarray(rows, dtype=np.float64)
    col_idx = {c: i for i, c in enumerate(columns)}
    n = data.shape[0]
    out = np.full((n, 31), np.nan, dtype=np.float64)
    for coord, mjx_i in _OPENSIM_TO_MJX_IDX.items():
        col = f"{coord}_moment"
        if col in col_idx:
            out[:, mjx_i] = data[:, col_idx[col]]
    return out, n


def stance_mask_from_intervals(intervals, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in intervals:
        m[max(0, s):min(n, e)] = True
    return m


def resample_pct(seg: np.ndarray) -> np.ndarray | None:
    seg = np.asarray(seg, dtype=np.float64)
    finite = np.isfinite(seg)
    if seg.size < 3 or np.count_nonzero(finite) < 3:
        return None
    x = np.linspace(0.0, 100.0, seg.size)
    return np.interp(PCT_GRID, x[finite], seg[finite])


# --- per-trial ----------------------------------------------------------------
def _ankle_power_block(os_31: np.ndarray, mjx_side: np.ndarray, qvel_use: np.ndarray,
                       mass: float, r_int, l_int, r_stance, l_stance) -> dict[str, Any]:
    """Ankle power (W/kg) metrics for one MJX variant against the OpenSim reference.

    The same ``qvel_use`` multiplies both the OpenSim and MJX moments so each panel is
    internally self-consistent with its own filtering (MJX-filter velocity for the normal
    variant, OS-filter velocity for the GCVSpline variant)."""
    ankle_power: dict[str, Any] = {}
    for side, ch in ANKLE_DOF_IDX.items():
        intervals = r_int if side == "right" else l_int
        smask = r_stance if side == "right" else l_stance
        p_os = os_31[:, ch] * qvel_use[:, ch] / mass
        p_mjx = mjx_side[:, ch] * qvel_use[:, ch] / mass
        peaks = []  # (os_peak, mjx_peak) push-off peaks per stance
        curves_os, curves_mjx = [], []
        for s, e in intervals:
            po, _ = stance_peak(p_os, s, e, last_half=True)
            pm, _ = stance_peak(p_mjx, s, e, last_half=True)
            if np.isfinite(po) and np.isfinite(pm):
                peaks.append((float(po), float(pm)))
            co, cm = resample_pct(p_os[s:e]), resample_pct(p_mjx[s:e])
            if co is not None and cm is not None:
                curves_os.append(co)
                curves_mjx.append(cm)
        peaks_arr = np.array(peaks) if peaks else np.zeros((0, 2))
        ankle_power[side] = {
            "curve_mae_Wkg": _mae(p_os, p_mjx, smask),
            "curve_rmse_Wkg": _rmse(p_os, p_mjx, smask),
            "peak_os_Wkg": float(np.mean(peaks_arr[:, 0])) if len(peaks_arr) else float("nan"),
            "peak_mjx_Wkg": float(np.mean(peaks_arr[:, 1])) if len(peaks_arr) else float("nan"),
            "peak_abs_err_Wkg": (float(np.mean(np.abs(peaks_arr[:, 0] - peaks_arr[:, 1])))
                                 if len(peaks_arr) else float("nan")),
            "peak_r": _pearson(peaks_arr[:, 0], peaks_arr[:, 1], np.ones(len(peaks_arr), bool))
                      if len(peaks_arr) >= 3 else float("nan"),
            "n_stances": int(len(peaks_arr)),
            "per_stance_peaks": peaks_arr.tolist(),
            "_curves_os": curves_os,    # consumed by plotting, stripped from JSON
            "_curves_mjx": curves_mjx,
        }
    return ankle_power


def compute_trial_metrics(paths: TrialPaths) -> dict[str, Any]:
    proc = paths.processed_dir
    sto = paths.output_dir / "inverse_dynamics.sto"
    mass, height = load_subject_md(paths.subject_dir)
    norm_factor = mass * height * GRAVITY

    os_31, n_os = load_opensim_31ch(sto)
    mjx = np.load(proc / "ID_GT_MJX.npy").astype(np.float64)
    qvel = np.load(proc / "qvel_mjx.npy").astype(np.float64)
    grf = np.load(proc / "GRF_Cleaned.npy").astype(np.float64)

    # Prescribed-acceleration ID: hand OpenSim the exact MJX qvel/qacc (vs GCVSpline derivation).
    try:
        presc_31, n_presc = compute_prescribed_id_31ch(paths)
    except Exception as exc:  # noqa: BLE001
        presc_31, n_presc = None, n_os
        print(f"  [warn] prescribed-accel ID failed for {paths.trial_dir.name}: {exc}", flush=True)

    # MJX ID computed with OpenSim-style GCVSpline filtering (ProcessData --os-filtering output).
    osfilt_path = proc / "ID_GT_MJX_OSfilt.npy"
    mjx_osfilt = np.load(osfilt_path).astype(np.float64) if osfilt_path.exists() else None
    qvel_osfilt_path = proc / "qvel_mjx_OSfilt.npy"
    qvel_osfilt = (np.load(qvel_osfilt_path).astype(np.float64)
                   if (mjx_osfilt is not None and qvel_osfilt_path.exists()) else None)

    lengths = [n_os, mjx.shape[0], qvel.shape[0], grf.shape[0], n_presc]
    if mjx_osfilt is not None:
        lengths.append(mjx_osfilt.shape[0])
    n = min(lengths)
    gap = max(n_os, mjx.shape[0]) - min(n_os, mjx.shape[0])
    os_31, mjx, qvel, grf = os_31[:n], mjx[:n], qvel[:n], grf[:n]
    if presc_31 is not None:
        presc_31 = presc_31[:n]
    if mjx_osfilt is not None:
        mjx_osfilt = mjx_osfilt[:n]
    if qvel_osfilt is not None:
        qvel_osfilt = qvel_osfilt[:n]

    bw_n = mass * GRAVITY
    r_int = dual_threshold_stance_intervals(grf, "right", bw_n)
    l_int = dual_threshold_stance_intervals(grf, "left", bw_n)
    r_stance = stance_mask_from_intervals(r_int, n)
    l_stance = stance_mask_from_intervals(l_int, n)
    union_stance = r_stance | l_stance
    full = np.ones(n, dtype=bool)

    def stance_for(dof: str) -> np.ndarray:
        if dof in RIGHT_LEG_DOFS:
            return r_stance
        if dof in LEFT_LEG_DOFS:
            return l_stance
        return union_stance

    def intervals_for(dof: str):
        if dof in RIGHT_LEG_DOFS:
            return r_int
        if dof in LEFT_LEG_DOFS:
            return l_int
        return list(r_int) + list(l_int)

    per_dof: dict[str, Any] = {}
    for dof in DOF_ORDER:
        ch = _OPENSIM_TO_MJX_IDX[dof]
        os_sig, mjx_sig = os_31[:, ch], mjx[:, ch]
        d = _dof_metrics(os_sig, mjx_sig, full, norm_factor)
        d["stance"] = _dof_metrics(os_sig, mjx_sig, stance_for(dof), norm_factor)
        # Torque range at this joint (mean stance max - mean stance min, %BWxH),
        # from the OpenSim reference signal; denominator for the %-of-range plot.
        d["range_bwh"] = _stance_range_bwh(os_sig, intervals_for(dof), norm_factor)
        # Prescribed-acceleration variant (exact MJX qvel/qacc) vs MJX GT.
        if presc_31 is not None:
            pm = _dof_metrics(presc_31[:, ch], mjx_sig, full, norm_factor)
            pm["stance"] = _dof_metrics(presc_31[:, ch], mjx_sig, stance_for(dof), norm_factor)
            d["mjxacc"] = pm
        # Process 3: OpenSim (OS filtering) vs MJX (OS filtering) -- engine difference under
        # matched OpenSim-style GCVSpline kinematics.
        if mjx_osfilt is not None:
            om = _dof_metrics(os_sig, mjx_osfilt[:, ch], full, norm_factor)
            om["stance"] = _dof_metrics(os_sig, mjx_osfilt[:, ch], stance_for(dof), norm_factor)
            d["osfilt"] = om
        per_dof[dof] = d

    # overall = mean across DOFs; global_rmse_bwh = pooled RMSE across all DOF channels
    def overall_block(use_stance: bool) -> dict[str, float]:
        maes, rmses, nrmses, rs = [], [], [], []
        pooled_sq, pooled_n = 0.0, 0
        for dof in DOF_ORDER:
            ch = _OPENSIM_TO_MJX_IDX[dof]
            os_sig, mjx_sig = os_31[:, ch], mjx[:, ch]
            mask = stance_for(dof) if use_stance else full
            m = _valid(os_sig, mjx_sig, mask)
            if np.any(m):
                diff = os_sig[m] - mjx_sig[m]
                pooled_sq += float(np.sum(diff ** 2))
                pooled_n += int(np.count_nonzero(m))
            d = per_dof[dof]["stance"] if use_stance else per_dof[dof]
            maes.append(d["mae_bwh"])
            rmses.append(d["rmse_bwh"])
            nrmses.append(d["nrmse"])
            rs.append(d["r"])
        global_rmse = float(np.sqrt(pooled_sq / pooled_n)) if pooled_n else float("nan")
        return {
            "mae_bwh": float(np.nanmean(maes)),
            "rmse_bwh": float(np.nanmean(rmses)),
            "global_rmse_bwh": (global_rmse / norm_factor) * 100.0,
            "mean_nrmse": float(np.nanmean(nrmses)),
            "mean_r": float(np.nanmean(rs)),
        }

    overall = overall_block(use_stance=False)
    overall["stance"] = overall_block(use_stance=True)
    if presc_31 is not None:
        overall["mjxacc"] = {
            "mae_bwh": float(np.nanmean([per_dof[d]["mjxacc"]["mae_bwh"] for d in DOF_ORDER])),
            "mean_r": float(np.nanmean([per_dof[d]["mjxacc"]["r"] for d in DOF_ORDER])),
            "stance": {"mae_bwh": float(np.nanmean(
                [per_dof[d]["mjxacc"]["stance"]["mae_bwh"] for d in DOF_ORDER]))},
        }
    if mjx_osfilt is not None:
        overall["osfilt"] = {
            "mae_bwh": float(np.nanmean([per_dof[d]["osfilt"]["mae_bwh"] for d in DOF_ORDER])),
            "mean_r": float(np.nanmean([per_dof[d]["osfilt"]["r"] for d in DOF_ORDER])),
            "stance": {"mae_bwh": float(np.nanmean(
                [per_dof[d]["osfilt"]["stance"]["mae_bwh"] for d in DOF_ORDER]))},
        }

    # --- ankle power (W/kg) ---
    # Normal variant: MJX moments with MJX-filtered velocity (field-standard comparison).
    ankle_power = _ankle_power_block(os_31, mjx, qvel, mass,
                                     r_int, l_int, r_stance, l_stance)
    # GCVSpline variant: MJX OS-filtered moments with OS-filtered velocity.
    ankle_power_osfilt = (
        _ankle_power_block(os_31, mjx_osfilt, qvel_osfilt, mass,
                           r_int, l_int, r_stance, l_stance)
        if (mjx_osfilt is not None and qvel_osfilt is not None) else None)

    knee_flexion = {
        "right": {k: per_dof["knee_angle_r"][k] for k in ("mae_bwh", "rmse_bwh", "nrmse", "r", "bias")},
        "left": {k: per_dof["knee_angle_l"][k] for k in ("mae_bwh", "rmse_bwh", "nrmse", "r", "bias")},
    }
    knee_flexion["right"]["stance"] = per_dof["knee_angle_r"]["stance"]
    knee_flexion["left"]["stance"] = per_dof["knee_angle_l"]["stance"]

    # Raw per-frame pairs for the scatter plot (underscore key -> stripped from JSON).
    scatter = {d: (os_31[:, _OPENSIM_TO_MJX_IDX[d]], mjx[:, _OPENSIM_TO_MJX_IDX[d]])
               for d in SCATTER_DOFS}
    scatter_osfilt = ({d: (os_31[:, _OPENSIM_TO_MJX_IDX[d]], mjx_osfilt[:, _OPENSIM_TO_MJX_IDX[d]])
                       for d in SCATTER_DOFS} if mjx_osfilt is not None else None)

    return {
        "trial": paths.trial_dir.name,
        "subject": paths.subject_dir.name,
        "n_frames": int(n),
        "frame_align_gap": int(gap),
        "mass_kg": mass,
        "height_m": height,
        "norm_factor_N_m": norm_factor,
        "right_stance_frames": int(np.count_nonzero(r_stance)),
        "left_stance_frames": int(np.count_nonzero(l_stance)),
        "per_dof": per_dof,
        "overall": overall,
        "ankle_power": ankle_power,
        "ankle_power_osfilt": ankle_power_osfilt,
        "knee_flexion": knee_flexion,
        "warnings": (["frame_align_gap>2"] if gap > 2 else []),
        "_scatter": scatter,
        "_scatter_osfilt": scatter_osfilt,
    }


def _clean_for_json(obj: Any) -> Any:
    """Recursively drop private (underscore-prefixed) keys and coerce numpy scalars."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_clean_for_json(payload), f, indent=2)
        f.write("\n")


# --- aggregation --------------------------------------------------------------
def agg_scalar(values: list[Any]) -> dict[str, float]:
    a = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(a)), "median": float(np.median(a)),
            "std": float(np.std(a)), "n": int(a.size)}


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    per_dof: dict[str, Any] = {}
    for dof in DOF_ORDER:
        per_dof[dof] = {}
        for scope in ("overall", "stance"):
            for metric in ("mae_bwh", "rmse_bwh", "nrmse", "r", "bias"):
                vals = []
                for t in trials:
                    d = t["per_dof"][dof]
                    dd = d if scope == "overall" else d["stance"]
                    vals.append(dd.get(metric))
                per_dof[dof][f"{scope}_{metric}"] = agg_scalar(vals)
        # Prescribed-acceleration (MJX qvel/qacc) and OS-filtering variants: full and stance.
        for scope, pick in (("mjxacc", lambda d: d.get("mjxacc")),
                            ("mjxacc_stance", lambda d: (d.get("mjxacc") or {}).get("stance")),
                            ("osfilt", lambda d: d.get("osfilt")),
                            ("osfilt_stance", lambda d: (d.get("osfilt") or {}).get("stance"))):
            for metric in ("mae_bwh", "rmse_bwh", "r"):
                vals = []
                for t in trials:
                    sub = pick(t["per_dof"][dof])
                    vals.append(sub.get(metric) if sub else None)
                per_dof[dof][f"{scope}_{metric}"] = agg_scalar(vals)
        # Joint torque range (%BWxH) -- denominator for the %-of-range error plot.
        per_dof[dof]["range_bwh"] = agg_scalar(
            [t["per_dof"][dof].get("range_bwh") for t in trials])
    overall: dict[str, Any] = {}
    for scope in ("overall", "stance"):
        for metric in ("mae_bwh", "rmse_bwh", "global_rmse_bwh", "mean_nrmse", "mean_r"):
            vals = []
            for t in trials:
                o = t["overall"] if scope == "overall" else t["overall"]["stance"]
                vals.append(o.get(metric))
            overall[f"{scope}_{metric}"] = agg_scalar(vals)
    ankle_metrics = ("curve_mae_Wkg", "curve_rmse_Wkg", "peak_os_Wkg",
                     "peak_mjx_Wkg", "peak_abs_err_Wkg", "peak_r")
    ankle: dict[str, Any] = {}
    for side in ("right", "left"):
        ankle[side] = {
            m: agg_scalar([t["ankle_power"][side].get(m) for t in trials])
            for m in ankle_metrics
        }
    ankle_osfilt: dict[str, Any] = {}
    if any(t.get("ankle_power_osfilt") for t in trials):
        for side in ("right", "left"):
            ankle_osfilt[side] = {
                m: agg_scalar([(t.get("ankle_power_osfilt") or {}).get(side, {}).get(m)
                               for t in trials])
                for m in ankle_metrics
            }
    knee: dict[str, Any] = {}
    for side in ("right", "left"):
        knee[side] = {
            m: agg_scalar([t["knee_flexion"][side].get(m) for t in trials])
            for m in ("mae_bwh", "rmse_bwh", "nrmse", "r", "bias")
        }
    return {"n_trials": len(trials), "per_dof": per_dof, "overall": overall,
            "ankle_power": ankle, "ankle_power_osfilt": ankle_osfilt or None,
            "knee_flexion": knee}


# --- plotting -----------------------------------------------------------------
def _fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) of least-squares y ~ x."""
    if len(x) < 2 or np.std(x) < 1e-9:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return float(slope), float(intercept), float(r * r)


def plot_perdof_mae(out_dir: Path, dataset_agg: dict) -> None:
    labels = DOF_ORDER
    pd = dataset_agg["per_dof"]

    def series(key):
        return ([pd[d][key]["mean"] for d in labels], [pd[d][key]["std"] for d in labels])

    def available(key):
        return all(key in pd[d] and np.isfinite(pd[d][key]["mean"]) for d in labels)

    # Each process: (full_key, stance_key, full_color, stance_color, label).
    # P1 always present; P2 (prescribed MJX accel/vel) and P3 (both OS-filtered) optional.
    processes = [("overall_mae_bwh", "stance_mae_bwh", "#1f6feb", "#79b0f5",
                  "OS(OSfilt) vs MJX(MJXfilt)")]
    if available("osfilt_mae_bwh"):
        processes.append(("osfilt_mae_bwh", "osfilt_stance_mae_bwh", "#6f42c1", "#b08fe0",
                          "OS(OSfilt) vs MJX(OSfilt)"))

    nbar = 2 * len(processes)          # Full + Stance per process
    w = min(0.4, 0.82 / nbar)
    x = np.arange(len(labels))
    # Centered bar offsets for the nbar bars at each DOF.
    offsets = (np.arange(nbar) - (nbar - 1) / 2.0) * w

    fig, ax = plt.subplots(figsize=(16 if len(processes) > 1 else 14, 6.5))
    for pi, (fk, sk, fc, sc, lab) in enumerate(processes):
        f_mean, f_sd = series(fk)
        s_mean, s_sd = series(sk)
        ax.bar(x + offsets[2 * pi], f_mean, w, yerr=f_sd, capsize=1.2,
               label=f"{lab} - Full", color=fc)
        ax.bar(x + offsets[2 * pi + 1], s_mean, w, yerr=s_sd, capsize=1.2,
               label=f"{lab} - Stance", color=sc)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MAE  (% body-weight x height)")
    title = "OpenSim vs MJX moment error per DOF (dataset mean +/- SD)"
    if len(processes) == 2:
        title += "\nOpenSim-filter cross-comparison vs matched OS-filter (engine difference)"
    ax.set_title(title)
    ax.legend(ncol=len(processes), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "perdof_mae_bwh.png", dpi=130)
    plt.close(fig)


def plot_perdof_mae_pct_range(out_dir: Path, dataset_agg: dict) -> None:
    """Same 4-bar-per-DOF layout as plot_perdof_mae, but the error is expressed as a
    percentage of that joint's torque range (mean stance max - mean stance min of the
    BWxH-normalized OpenSim torque)."""
    labels = DOF_ORDER
    pd = dataset_agg["per_dof"]

    def rng(d):
        return pd[d].get("range_bwh", {}).get("mean", float("nan"))

    def series(key):
        # value = MAE / range * 100; SD scaled by the same range so bars stay comparable.
        means, sds = [], []
        for d in labels:
            r = rng(d)
            if np.isfinite(r) and r > 1e-9:
                means.append(pd[d][key]["mean"] / r * 100.0)
                sds.append(pd[d][key]["std"] / r * 100.0)
            else:
                means.append(float("nan"))
                sds.append(float("nan"))
        return means, sds

    def available(key):
        return all(key in pd[d] and np.isfinite(pd[d][key]["mean"]) for d in labels)

    processes = [("overall_mae_bwh", "stance_mae_bwh", "#1f6feb", "#79b0f5",
                  "OS(OSfilt) vs MJX(MJXfilt)")]
    if available("osfilt_mae_bwh"):
        processes.append(("osfilt_mae_bwh", "osfilt_stance_mae_bwh", "#6f42c1", "#b08fe0",
                          "OS(OSfilt) vs MJX(OSfilt)"))

    nbar = 2 * len(processes)
    w = min(0.4, 0.82 / nbar)
    x = np.arange(len(labels))
    offsets = (np.arange(nbar) - (nbar - 1) / 2.0) * w

    fig, ax = plt.subplots(figsize=(16 if len(processes) > 1 else 14, 6.5))
    for pi, (fk, sk, fc, sc, lab) in enumerate(processes):
        f_mean, f_sd = series(fk)
        s_mean, s_sd = series(sk)
        ax.bar(x + offsets[2 * pi], f_mean, w, yerr=f_sd, capsize=1.2,
               label=f"{lab} - Full", color=fc)
        ax.bar(x + offsets[2 * pi + 1], s_mean, w, yerr=s_sd, capsize=1.2,
               label=f"{lab} - Stance", color=sc)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MAE  (% of joint torque range)")
    title = ("OpenSim vs MJX moment error per DOF, normalized by joint torque range\n"
             "range = mean stance-max - mean stance-min of BWxH-normalized torque")
    ax.set_title(title)
    ax.legend(ncol=len(processes), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "perdof_mae_pct_range.png", dpi=130)
    plt.close(fig)


def plot_perdof_box(out_dir: Path, all_trials: list[dict]) -> None:
    def col(d, pick):
        return [v for t in all_trials
                if np.isfinite(v := pick(t["per_dof"][d]))]

    norm = [col(d, lambda pd: pd["mae_bwh"]) for d in DOF_ORDER]
    gcv = [col(d, lambda pd: (pd.get("osfilt") or {}).get("mae_bwh", float("nan")))
           for d in DOF_ORDER]
    has_gcv = any(len(c) for c in gcv)

    x = np.arange(len(DOF_ORDER))
    fig, ax = plt.subplots(figsize=(16 if has_gcv else 14, 6))
    if has_gcv:
        bn = ax.boxplot(norm, positions=x - 0.2, widths=0.35, showfliers=False,
                        patch_artist=True)
        bg = ax.boxplot(gcv, positions=x + 0.2, widths=0.35, showfliers=False,
                        patch_artist=True)
        for b in bn["boxes"]:
            b.set(facecolor="#79b0f5", alpha=0.8)
        for b in bg["boxes"]:
            b.set(facecolor="#b08fe0", alpha=0.8)
        ax.legend([bn["boxes"][0], bg["boxes"][0]],
                  ["MJX normal filter", "MJX GCVSpline filter"], loc="upper left")
    else:
        ax.boxplot(norm, positions=x, showfliers=False)
    ax.set_xticks(x)
    ax.set_xticklabels(DOF_ORDER, rotation=45, ha="right")
    ax.set_ylabel("MAE  (% body-weight x height)")
    ax.set_title("Per-trial MAE distribution per DOF (full trial)\n"
                 "MJX normal filter vs MJX GCVSpline filter (both vs OpenSim)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "perdof_box.png", dpi=130)
    plt.close(fig)


def _joint_scatter_row(axes, all_trials, scatter_key, color, row_label, cap):
    for ax, dof in zip(axes, SCATTER_DOFS):
        os_all, mjx_all = [], []
        for t in all_trials:
            sc = (t.get(scatter_key) or {}).get(dof)
            if sc is not None:
                os_all.append(np.asarray(sc[0], float))
                mjx_all.append(np.asarray(sc[1], float))
        if not os_all:
            ax.set_title(f"{dof}\n(no data)")
            continue
        os_v = np.concatenate(os_all)
        mjx_v = np.concatenate(mjx_all)
        good = np.isfinite(os_v) & np.isfinite(mjx_v)
        os_v, mjx_v = os_v[good], mjx_v[good]
        if len(os_v) > cap:
            sel = np.random.RandomState(0).choice(len(os_v), cap, replace=False)
            os_v, mjx_v = os_v[sel], mjx_v[sel]
        ax.scatter(mjx_v, os_v, s=3, alpha=0.25, color=color, edgecolors="none")
        lo = float(min(os_v.min(), mjx_v.min()))
        hi = float(max(os_v.max(), mjx_v.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
        _, _, r2 = _fit_r2(mjx_v, os_v)
        rmse = float(np.sqrt(np.mean((os_v - mjx_v) ** 2)))
        ax.set_title(f"{row_label} | {dof}\nR2={r2:.3f}  RMSE={rmse:.1f} N.m")
        ax.set_xlabel("MJX moment (N.m)")
        ax.set_ylabel("OpenSim moment (N.m)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)


def plot_joint_scatter(out_dir: Path, all_trials: list[dict], cap: int = 6000) -> None:
    has_gcv = any(t.get("_scatter_osfilt") for t in all_trials)
    nrows = 2 if has_gcv else 1
    nc = len(SCATTER_DOFS)
    fig, axes = plt.subplots(nrows, nc, figsize=(5 * nc, 5 * nrows), squeeze=False)
    _joint_scatter_row(axes[0], all_trials, "_scatter", "#1f6feb", "MJX normal", cap)
    if has_gcv:
        _joint_scatter_row(axes[1], all_trials, "_scatter_osfilt", "#6f42c1",
                           "MJX GCVSpline", cap)
    fig.suptitle("OpenSim vs MJX joint moments (all frames)\n"
                 "top: MJX normal filter   bottom: MJX GCVSpline filter")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_scatter.png", dpi=130)
    plt.close(fig)


def _ankle_peak_row(axes, all_trials, ap_key, dot_color, row_label):
    for ax, side in zip(axes, ("right", "left")):
        pairs = []
        for t in all_trials:
            ap = t.get(ap_key)
            if ap is not None:
                pairs.extend(ap[side].get("per_stance_peaks", []))
        pairs = np.array(pairs, dtype=np.float64) if pairs else np.zeros((0, 2))
        if len(pairs) == 0:
            ax.set_title(f"{row_label} | {side} ankle (no stances)")
            continue
        os_p, mjx_p = pairs[:, 0], pairs[:, 1]
        ax.scatter(mjx_p, os_p, s=18, alpha=0.6, color=dot_color, edgecolors="none")
        lo = float(min(os_p.min(), mjx_p.min()))
        hi = float(max(os_p.max(), mjx_p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
        slope, intercept, r2 = _fit_r2(mjx_p, os_p)
        if np.isfinite(slope):
            xs = np.array([lo, hi])
            ax.plot(xs, slope * xs + intercept, color="#1f6feb", lw=1.5,
                    label=f"fit (R2={r2:.3f})")
        mae = float(np.mean(np.abs(os_p - mjx_p)))
        ax.set_title(f"{row_label} | {side} push-off peak\nMAE={mae:.3f} W/kg  n={len(pairs)}")
        ax.set_xlabel("MJX peak power (W/kg)")
        ax.set_ylabel("OpenSim peak power (W/kg)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)


def plot_ankle_peak_scatter(out_dir: Path, all_trials: list[dict]) -> None:
    has_gcv = any(t.get("ankle_power_osfilt") for t in all_trials)
    nrows = 2 if has_gcv else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5.5 * nrows), squeeze=False)
    _ankle_peak_row(axes[0], all_trials, "ankle_power", "#e07b20", "MJX normal")
    if has_gcv:
        _ankle_peak_row(axes[1], all_trials, "ankle_power_osfilt", "#9467bd",
                        "MJX GCVSpline")
    fig.suptitle("Per-stance peak ankle power: OpenSim vs MJX\n"
                 "top: MJX normal filter   bottom: MJX GCVSpline filter")
    fig.tight_layout()
    fig.savefig(out_dir / "ankle_power_peak_scatter.png", dpi=130)
    plt.close(fig)


def _ankle_curve_row(axes, all_trials, ap_key, mjx_color, row_label):
    for ax, side in zip(axes, ("right", "left")):
        os_curves, mjx_curves = [], []
        for t in all_trials:
            ap = t.get(ap_key)
            if ap is None:
                continue
            os_curves.extend(ap[side].get("_curves_os", []))
            mjx_curves.extend(ap[side].get("_curves_mjx", []))
        if not os_curves:
            ax.set_title(f"{row_label} | {side} ankle (no stances)")
            continue
        os_arr = np.array(os_curves)
        mjx_arr = np.array(mjx_curves)
        for arr, color, lab in ((os_arr, "#1f6feb", "OpenSim"),
                                (mjx_arr, mjx_color, "MJX")):
            mean = np.nanmean(arr, axis=0)
            sd = np.nanstd(arr, axis=0)
            ax.plot(PCT_GRID, mean, color=color, lw=2, label=lab)
            ax.fill_between(PCT_GRID, mean - sd, mean + sd, color=color, alpha=0.2)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_title(f"{row_label} | {side} ankle power (n={len(os_curves)} stances)")
        ax.set_xlabel("% stance")
        ax.set_ylabel("Ankle power (W/kg)")
        ax.legend()
        ax.grid(alpha=0.3)


def plot_ankle_power_curve(out_dir: Path, all_trials: list[dict]) -> None:
    has_gcv = any(t.get("ankle_power_osfilt") for t in all_trials)
    nrows = 2 if has_gcv else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5.5 * nrows), squeeze=False)
    _ankle_curve_row(axes[0], all_trials, "ankle_power", "#e07b20", "MJX normal")
    if has_gcv:
        _ankle_curve_row(axes[1], all_trials, "ankle_power_osfilt", "#9467bd",
                         "MJX GCVSpline")
    fig.suptitle("Stance-normalized ankle power: OpenSim vs MJX (mean +/- SD)\n"
                 "top: MJX normal filter   bottom: MJX GCVSpline filter")
    fig.tight_layout()
    fig.savefig(out_dir / "ankle_power_curve.png", dpi=130)
    plt.close(fig)


def plot_subject_heatmap(out_dir: Path, subject_aggs: dict[str, dict]) -> None:
    subjects = sorted(subject_aggs.keys())
    if not subjects:
        return

    def matrix(key):
        return np.array([[subject_aggs[s]["per_dof"][d].get(key, {}).get("mean", float("nan"))
                          for d in DOF_ORDER] for s in subjects], dtype=np.float64)

    mat_norm = matrix("overall_mae_bwh")
    mat_gcv = matrix("osfilt_mae_bwh")
    has_gcv = np.isfinite(mat_gcv).any()
    ncols = 2 if has_gcv else 1
    # Shared color scale so the two panels are directly comparable.
    vmax = np.nanpercentile(np.concatenate([mat_norm.ravel(),
                            mat_gcv.ravel() if has_gcv else mat_norm.ravel()]), 98)
    fig, axes = plt.subplots(1, ncols, figsize=(15 * ncols, max(4, 0.4 * len(subjects))),
                             squeeze=False)
    panels = [(axes[0][0], mat_norm, "MJX normal filter")]
    if has_gcv:
        panels.append((axes[0][1], mat_gcv, "MJX GCVSpline filter"))
    im = None
    for ax, mat, label in panels:
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(DOF_ORDER)))
        ax.set_xticklabels(DOF_ORDER, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(subjects)))
        ax.set_yticklabels(subjects)
        ax.set_title(f"Per-subject MAE (% BWxH), full trial\n{label}")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="MAE (% BWxH)")
    fig.savefig(out_dir / "per_subject_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_overall_summary(out_dir: Path, dataset_agg: dict) -> None:
    ov = dataset_agg["overall"]
    ap = dataset_agg["ankle_power"]
    pd = dataset_agg["per_dof"]
    ap_g = dataset_agg.get("ankle_power_osfilt")

    def dof_mean(key):
        vals = [pd[d].get(key, {}).get("mean", float("nan")) for d in DOF_ORDER]
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

    def ank(agg, side):
        return (agg or {}).get(side, {}).get("peak_abs_err_Wkg", {}).get("mean", float("nan"))

    # (label, normal_value, gcvspline_value)
    rows = [
        ("Mean MAE (% BWxH), full", ov["overall_mae_bwh"]["mean"], dof_mean("osfilt_mae_bwh")),
        ("Mean MAE (% BWxH), stance", ov["stance_mae_bwh"]["mean"], dof_mean("osfilt_stance_mae_bwh")),
        ("Mean RMSE (% BWxH), full", ov["overall_rmse_bwh"]["mean"], dof_mean("osfilt_rmse_bwh")),
        ("Global RMSE (% BWxH), full", ov["overall_global_rmse_bwh"]["mean"], float("nan")),
        ("Mean Pearson r, full", ov["overall_mean_r"]["mean"], dof_mean("osfilt_r")),
        ("Ankle peak MAE R (W/kg)", ap["right"]["peak_abs_err_Wkg"]["mean"], ank(ap_g, "right")),
        ("Ankle peak MAE L (W/kg)", ap["left"]["peak_abs_err_Wkg"]["mean"], ank(ap_g, "left")),
    ]
    has_gcv = any(np.isfinite(r[2]) for r in rows)
    labels = [r[0] for r in rows]
    norm_vals = [r[1] for r in rows]
    gcv_vals = [r[2] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if has_gcv:
        h = 0.38
        ax.barh(y - h / 2, norm_vals, h, color="#1f6feb", label="MJX normal filter")
        ax.barh(y + h / 2, gcv_vals, h, color="#6f42c1", label="MJX GCVSpline filter")
        for yi, v in zip(y - h / 2, norm_vals):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
        for yi, v in zip(y + h / 2, gcv_vals):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
        ax.legend(loc="lower right")
    else:
        ax.barh(y, norm_vals, color="#1f6feb")
        for yi, v in zip(y, norm_vals):
            ax.text(v, yi, f"  {v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"OpenSim vs MJX ID -- dataset summary (n={dataset_agg['n_trials']} trials)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_summary.png", dpi=130)
    plt.close(fig)


def make_all_plots(out_dir: Path, all_trials: list[dict],
                   subject_aggs: dict[str, dict], dataset_agg: dict) -> None:
    plot_perdof_mae(out_dir, dataset_agg)
    plot_perdof_mae_pct_range(out_dir, dataset_agg)
    plot_perdof_box(out_dir, all_trials)
    plot_joint_scatter(out_dir, all_trials)
    plot_ankle_peak_scatter(out_dir, all_trials)
    plot_ankle_power_curve(out_dir, all_trials)
    plot_subject_heatmap(out_dir, subject_aggs)
    plot_overall_summary(out_dir, dataset_agg)


# --- driver -------------------------------------------------------------------
def ensure_id(paths: TrialPaths, *, run_id: bool, overwrite: bool) -> str:
    sto = paths.output_dir / "inverse_dynamics.sto"
    if (
        sto.exists()
        and not overwrite
        and existing_generation_matches(paths, source="processed", use_noised=False)
    ):
        return "reused"
    if not run_id:
        return "missing"
    process_trial(paths, use_noised=False, overwrite=overwrite, dry_run=False,
                  right_body=DEFAULT_RIGHT_BODY, left_body=DEFAULT_LEFT_BODY, source="processed")
    return "ran"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--subject", default=None)
    p.add_argument("--trial", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Take the first N discovered trials.")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly select N trials from across the dataset (uses --seed).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --sample.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run OpenSim ID even if inverse_dynamics.sto exists.")
    p.add_argument("--no-run-id", dest="run_id", action="store_false",
                   help="Do not run ID; only read existing inverse_dynamics.sto.")
    p.set_defaults(run_id=True)
    p.add_argument("--output-folder", default="OpenSimToMJX_Accuracy")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        return 2

    trials = discover_trials(dataset_root, subject=args.subject, trial=args.trial)
    if args.sample is not None and args.sample < len(trials):
        rng = np.random.RandomState(args.seed)
        idx = sorted(rng.choice(len(trials), args.sample, replace=False))
        trials = [trials[i] for i in idx]
        print(f"Randomly sampled {len(trials)} trials (seed={args.seed}).")
    if args.limit is not None:
        trials = trials[: args.limit]

    all_trials: list[dict[str, Any]] = []
    by_subject: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, str]] = []

    for paths in trials:
        tag = f"{paths.subject_dir.name}/{paths.trial_dir.name}"
        try:
            status = ensure_id(paths, run_id=args.run_id, overwrite=args.overwrite)
            if status == "missing":
                failures.append({"trial": tag, "error": "inverse_dynamics.sto missing (--no-run-id)"})
                continue
            metrics = compute_trial_metrics(paths)
            metrics["id_status"] = status
            write_json(paths.output_dir / "AccuracyMetrics.json", metrics)
            all_trials.append(metrics)
            by_subject.setdefault(paths.subject_dir.name, []).append(metrics)
            print(f"[ok] {tag}  MAE%={metrics['overall']['mae_bwh']:.3f}  "
                  f"r={metrics['overall']['mean_r']:.3f}  ({status})", flush=True)
        except FileNotFoundError as exc:
            # Expected for incomplete trials (e.g. missing pos_mjx.npy); skip quietly.
            missing = Path(str(exc)).name
            failures.append({"trial": tag, "error": f"missing input: {missing}"})
            print(f"[skip] {tag}: missing input file ({missing})", flush=True)
        except Exception as exc:  # noqa: BLE001
            # Unexpected error: keep the full traceback for debugging.
            failures.append({"trial": tag, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {tag}: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()

    if not all_trials:
        print("No trials processed successfully.", file=sys.stderr)
        print(json.dumps({"failures": failures}, indent=2))
        return 1

    subject_aggs: dict[str, dict] = {}
    for subj, subj_trials in by_subject.items():
        agg = aggregate(subj_trials)
        agg["subject"] = subj
        subject_aggs[subj] = agg
        write_json(trials[0].subject_dir.parent / subj / "AccuracyMetrics.json", agg)

    dataset_agg = aggregate(all_trials)
    dataset_agg["n_subjects"] = len(subject_aggs)

    out_dir = dataset_root / args.output_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "dataset_root": str(dataset_root),
            "source": "processed", "use_noised": False,
            "norm": "mass_kg * height_m * 9.8067", "gravity": GRAVITY,
            "n_trials": len(all_trials), "n_subjects": len(subject_aggs),
        },
        "dataset": dataset_agg,
        "subjects": subject_aggs,
        "trials": [_clean_for_json(t) for t in all_trials],
        "failures": failures,
    }
    write_json(out_dir / "Accuracy_Summary.json", summary)
    make_all_plots(out_dir, all_trials, subject_aggs, dataset_agg)

    print(f"\n=== Dataset summary (n={len(all_trials)} trials, {len(subject_aggs)} subjects) ===")
    ov = dataset_agg["overall"]
    print(f"  Full   MAE%={ov['overall_mae_bwh']['mean']:.3f}  "
          f"RMSE%={ov['overall_rmse_bwh']['mean']:.3f}  r={ov['overall_mean_r']['mean']:.3f}")
    print(f"  Stance MAE%={ov['stance_mae_bwh']['mean']:.3f}  "
          f"RMSE%={ov['stance_rmse_bwh']['mean']:.3f}")
    pd0 = dataset_agg["per_dof"][DOF_ORDER[0]]
    if "mjxacc_mae_bwh" in pd0:
        mj_full = np.nanmean([dataset_agg["per_dof"][d]["mjxacc_mae_bwh"]["mean"] for d in DOF_ORDER])
        mj_st = np.nanmean([dataset_agg["per_dof"][d]["mjxacc_stance_mae_bwh"]["mean"] for d in DOF_ORDER])
        print(f"  P2 OS(MJXfilt)vsMJX(MJXfilt)  Full MAE%={mj_full:.3f}  Stance MAE%={mj_st:.3f}")
    if "osfilt_mae_bwh" in pd0 and np.isfinite(pd0["osfilt_mae_bwh"]["mean"]):
        of_full = np.nanmean([dataset_agg["per_dof"][d]["osfilt_mae_bwh"]["mean"] for d in DOF_ORDER])
        of_st = np.nanmean([dataset_agg["per_dof"][d]["osfilt_stance_mae_bwh"]["mean"] for d in DOF_ORDER])
        print(f"  P3 OS(OSfilt)vsMJX(OSfilt)    Full MAE%={of_full:.3f}  Stance MAE%={of_st:.3f}")
    print(f"  (P1 OS(OSfilt)vsMJX(MJXfilt)  Full MAE%={ov['overall_mae_bwh']['mean']:.3f})")
    ap = dataset_agg["ankle_power"]
    print(f"  Ankle peak MAE: R={ap['right']['peak_abs_err_Wkg']['mean']:.3f}  "
          f"L={ap['left']['peak_abs_err_Wkg']['mean']:.3f} W/kg")
    print(f"  Outputs -> {out_dir}")
    if failures:
        print(f"  {len(failures)} trial(s) failed.")
    return 1 if failures and not all_trials else 0


if __name__ == "__main__":
    raise SystemExit(main())
