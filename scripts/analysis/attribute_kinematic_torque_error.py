#!/usr/bin/env python3
"""Attribute per-DOF stance-phase torque error to MoCap-vs-video kinematic differences.

This is a *model-free* analysis. It holds the OpenCap/ProcessedData force
signals fixed (GRF + free moments), holds the MoCap COP fixed, and substitutes
only the kinematic transform terms (Jacobian, ground-aligned-calc rotation, and
qfrc_inverse) between their MoCap (GT) and video (ProcessedData) versions. This
isolates how much torque error each term injects, independent of any model
prediction error.

The MoCap COP is used for both the MoCap and ProcessedData reconstructions by
default. Use --include_processed_cop_attribution only for an explicit COP swap
counterfactual. Metrics are aggregated over left-foot stance frames only.

Reconstruction (matches TransformerFinal/data_loader.py and infer.py, sign-checked):

    tau_grf = sum_foot [ Jp^T F + Jr^T (M_free + (R_ga->w . cop_calc) x F) ]
    full_ID = qfrc_inverse - tau_grf

Per-DOF attribution via one-at-a-time (OAT) substitution from the all-MoCap
reference, with the residual interaction reported so the first-order split can be
trusted (or escalated to Shapley if the residual is large):

    ID_M       = f(J_M, R_M, cop_M, qfrc_M, F_v)       # GT kinematics/COP/qfrc, video forces
    ID_v       = f(J_v, R_v, cop_M, qfrc_v, F_v)       # video J/R/qfrc, GT COP, video forces
    C_qfrc     = ID_M - f(.,.,.,qfrc_v) = qfrc_M - qfrc_v   (additive, exact)
    C_J        = ID_M - f(J_v, R_M, cop_M, qfrc_M)
    C_R        = ID_M - f(J_M, R_v, cop_M, qfrc_M)
    total      = ID_M - ID_v
    interaction= total - (C_qfrc + C_J + C_R)

Each contribution is normalised by body-weight x height (% BW*H) and aggregated
over left-foot stance frames per DOF, per subject, then across the cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from paths import artifact, dataset  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GRAVITY = 9.8067

# MJX qfrc index -> (name, side).  Side selects the stance mask: R/L foot contact,
# or "C" (central, e.g. lumbar) supervised whenever either foot is in stance.
DOF_INFO: List[Tuple[int, str, str]] = [
    (6, "hip_flexion_r", "R"), (7, "hip_adduction_r", "R"), (8, "hip_rotation_r", "R"),
    (11, "knee_r", "R"), (14, "ankle_r", "R"), (15, "subtalar_r", "R"),
    (17, "hip_flexion_l", "L"), (18, "hip_adduction_l", "L"), (19, "hip_rotation_l", "L"),
    (22, "knee_l", "L"), (25, "ankle_l", "L"), (26, "subtalar_l", "L"),
    (28, "lumbar_extension", "C"), (29, "lumbar_bending", "C"), (30, "lumbar_rotation", "C"),
]
def _term_list(include_processed_cop_attribution: bool) -> List[str]:
    """Attribution columns. By default the MoCap COP is fixed everywhere, so the
    COP term is dropped and only Jacobian/rotation/qfrc_inverse can differ."""
    factors = ["qfrc_inverse", "jacobian", "rotation"]
    if include_processed_cop_attribution:
        factors.append("cop")
    return factors + ["total", "interaction"]


def _load(folder: Path, name: str, allow_pickle: bool = False) -> Optional[np.ndarray]:
    path = folder / name
    if not path.exists():
        return None
    return np.load(path, allow_pickle=allow_pickle)


def _load_jacobian(folder: Path) -> Optional[Dict[str, np.ndarray]]:
    arr = _load(folder, "Jacobian.npy", allow_pickle=True)
    if arr is None:
        return None
    data = arr.item()
    return {"jacp": np.asarray(data["jacp"], dtype=np.float64),
            "jacr": np.asarray(data["jacr"], dtype=np.float64)}


def _tau_grf(jac: Dict[str, np.ndarray], rot: np.ndarray, cop6: np.ndarray,
             grf: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """GRF-contribution joint torque (T, n_dof) from external forces + kinematics."""
    jacp, jacr = jac["jacp"], jac["jacr"]
    tau = np.zeros((len(grf), jacp.shape[-1]), dtype=np.float64)
    for foot, sl in ((0, slice(0, 3)), (1, slice(3, 6))):
        F = grf[:, sl]
        M_free = moments[:, sl]
        rot_ga_to_w = np.transpose(rot[:, foot], (0, 2, 1))          # R_ga->w = R_w->ga^T
        cop_world = np.einsum("tij,tj->ti", rot_ga_to_w, cop6[:, sl])
        M_total = M_free + np.cross(cop_world, F)
        tau = tau + np.einsum("tji,tj->ti", jacp[:, foot], F)
        tau = tau + np.einsum("tji,tj->ti", jacr[:, foot], M_total)
    return tau


def _full_id(jac, rot, cop6, qfrc, grf, moments) -> np.ndarray:
    return qfrc - _tau_grf(jac, rot, cop6, grf, moments)


def _discover_trials(data_dir: Path, exclude_subjects: set) -> List[Tuple[str, Path]]:
    trials: List[Tuple[str, Path]] = []
    for subject_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("subject")):
        if subject_dir.name in exclude_subjects:
            continue
        for trial_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            if (trial_dir / "MoCap").is_dir() and (trial_dir / "ProcessedData").is_dir():
                trials.append((subject_dir.name, trial_dir))
    return trials


REQUIRED = ["GRF_Cleaned.npy", "Moment_Cleaned.npy", "COP_CalcFrame_GroundAligned.npy",
            "WorldToGroundAlignedCalcnRotation.npy", "qfrc_inverse.npy", "contactBoolean.npy",
            "Mass_kg.npy", "Height_m.npy"]


def _format_lengths(lengths: Dict[str, int]) -> str:
    grouped: Dict[int, List[str]] = {}
    for name, value in lengths.items():
        grouped.setdefault(int(value), []).append(name)
    return "; ".join(
        f"{length}: {', '.join(sorted(names))}"
        for length, names in sorted(grouped.items())
    )


def _alignment_signal(grf: np.ndarray) -> np.ndarray:
    """Vertical GRF trace used only for MoCap/ProcessedData temporal alignment."""
    grf = np.asarray(grf, dtype=np.float64)
    if grf.ndim != 2 or grf.shape[1] < 6:
        return np.asarray([], dtype=np.float64)
    return grf[:, 2] + grf[:, 5]


def _score_alignment(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or len(b) < 5:
        return -np.inf
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std > 1e-8 and b_std > 1e-8:
        return float(np.corrcoef(a, b)[0, 1])
    return -float(np.mean(np.abs(a - b)))


def _resolve_alignment(
    mocap_signal: np.ndarray,
    processed_signal: np.ndarray,
    max_lag: int,
) -> Optional[Dict[str, int | float]]:
    """Find paired MoCap/ProcessedData slices.

    `offset` means MoCap index = ProcessedData index + offset.
    """
    best: Optional[Dict[str, int | float]] = None
    len_m = int(len(mocap_signal))
    len_p = int(len(processed_signal))
    for offset in range(-int(max_lag), int(max_lag) + 1):
        p0 = max(0, -offset)
        m0 = max(0, offset)
        overlap = min(len_p - p0, len_m - m0)
        if overlap < 5:
            continue
        score = _score_alignment(
            mocap_signal[m0:m0 + overlap],
            processed_signal[p0:p0 + overlap],
        )
        # Prefer higher score, then more overlap, then smaller absolute offset.
        rank = (score, overlap, -abs(offset))
        if best is None or rank > best["_rank"]:
            best = {
                "offset": int(offset),
                "mocap_start": int(m0),
                "processed_start": int(p0),
                "overlap": int(overlap),
                "score": float(score),
                "_rank": rank,
            }
    if best is not None:
        best.pop("_rank", None)
    return best


def _process_trial(
    trial_dir: Path,
    trim_edges: int,
    include_processed_cop_attribution: bool,
    allow_truncate_misaligned: bool,
    temporal_align: bool,
    align_max_lag: int,
) -> Optional[Dict[str, np.ndarray]]:
    mo, vi = trial_dir / "MoCap", trial_dir / "ProcessedData"
    required_mocap = [
        "Jacobian.npy",
        "COP_CalcFrame_GroundAligned.npy",
        "WorldToGroundAlignedCalcnRotation.npy",
        "qfrc_inverse.npy",
        "Mass_kg.npy",
        "Height_m.npy",
        "GRF_Cleaned.npy",
    ]
    required_processed = [
        "Jacobian.npy",
        "GRF_Cleaned.npy",
        "Moment_Cleaned.npy",
        "WorldToGroundAlignedCalcnRotation.npy",
        "qfrc_inverse.npy",
        "contactBoolean.npy",
    ]
    if include_processed_cop_attribution:
        required_processed.append("COP_CalcFrame_GroundAligned.npy")
    for f in required_mocap:
        if not (mo / f).exists():
            return None
    for f in required_processed:
        if not (vi / f).exists():
            return None

    jac_m, jac_v = _load_jacobian(mo), _load_jacobian(vi)
    rot_m = _load(mo, "WorldToGroundAlignedCalcnRotation.npy")
    rot_v = _load(vi, "WorldToGroundAlignedCalcnRotation.npy")
    cop_m = _load(mo, "COP_CalcFrame_GroundAligned.npy")
    cop_v = _load(vi, "COP_CalcFrame_GroundAligned.npy") if include_processed_cop_attribution else None
    qfrc_m = _load(mo, "qfrc_inverse.npy")
    qfrc_v = _load(vi, "qfrc_inverse.npy")
    grf_m_for_alignment = _load(mo, "GRF_Cleaned.npy")
    # Ground-truth forces are the OpenCap/ProcessedData force pipeline for both reconstructions.
    grf = _load(vi, "GRF_Cleaned.npy")
    mom = _load(vi, "Moment_Cleaned.npy")
    contact = _load(vi, "contactBoolean.npy")
    mass = _load(mo, "Mass_kg.npy")
    height = _load(mo, "Height_m.npy")

    arrays = [rot_m, rot_v, cop_m, qfrc_m, qfrc_v, grf_m_for_alignment, grf, mom, contact,
              jac_m["jacp"], jac_m["jacr"], jac_v["jacp"], jac_v["jacr"]]
    if include_processed_cop_attribution:
        arrays.append(cop_v)
    if any(a is None for a in arrays) or jac_m is None or jac_v is None:
        return None

    length_check = {
        "MoCap/rot": len(rot_m),
        "MoCap/cop": len(cop_m),
        "MoCap/qfrc_inverse": len(qfrc_m),
        "MoCap/grf_alignment": len(grf_m_for_alignment),
        "MoCap/jacp": len(jac_m["jacp"]),
        "MoCap/jacr": len(jac_m["jacr"]),
        "ProcessedData/rot": len(rot_v),
        "ProcessedData/qfrc_inverse": len(qfrc_v),
        "ProcessedData/grf": len(grf),
        "ProcessedData/moment": len(mom),
        "ProcessedData/contact": len(contact),
        "ProcessedData/jacp": len(jac_v["jacp"]),
        "ProcessedData/jacr": len(jac_v["jacr"]),
    }
    if include_processed_cop_attribution:
        length_check["ProcessedData/cop"] = len(cop_v)
    if mass is not None:
        length_check["MoCap/mass"] = len(mass)
    if height is not None:
        length_check["MoCap/height"] = len(height)
    if len(set(length_check.values())) != 1 and not (temporal_align or allow_truncate_misaligned):
        print(
            f"  ! skipped {trial_dir.parent.name}/{trial_dir.name}: "
            f"MoCap/ProcessedData length mismatch ({_format_lengths(length_check)})"
        )
        return None

    if temporal_align:
        alignment = _resolve_alignment(
            _alignment_signal(grf_m_for_alignment),
            _alignment_signal(grf),
            max_lag=align_max_lag,
        )
        if alignment is None:
            print(f"  ! skipped {trial_dir.parent.name}/{trial_dir.name}: no valid temporal alignment")
            return None
        m0 = int(alignment["mocap_start"])
        p0 = int(alignment["processed_start"])
        n = int(alignment["overlap"])
    else:
        alignment = {
            "offset": 0,
            "mocap_start": 0,
            "processed_start": 0,
            "overlap": int(min(len(a) for a in arrays)),
            "score": None,
        }
        m0 = 0
        p0 = 0
        n = int(alignment["overlap"])
    if mass is not None:
        n = min(n, len(mass) - m0)
    if height is not None:
        n = min(n, len(height) - m0)
    if trim_edges > 0:
        lo, hi = trim_edges, n - trim_edges
    else:
        lo, hi = 0, n
    if hi - lo < 5:
        return None

    def cut_mocap(a):
        return None if a is None else np.asarray(a, dtype=np.float64)[m0 + lo:m0 + hi]

    def cut_processed(a):
        return None if a is None else np.asarray(a, dtype=np.float64)[p0 + lo:p0 + hi]

    jac_m = {k: cut_mocap(v) for k, v in jac_m.items()}
    jac_v = {k: cut_processed(v) for k, v in jac_v.items()}
    rot_m, rot_v = cut_mocap(rot_m), cut_processed(rot_v)
    cop_m = cut_mocap(cop_m)
    cop_v = cut_processed(cop_v) if include_processed_cop_attribution else None
    qfrc_m, qfrc_v = cut_mocap(qfrc_m), cut_processed(qfrc_v)
    grf, mom, contact = cut_processed(grf), cut_processed(mom), cut_processed(contact)
    mass_v = cut_mocap(mass) if mass is not None else None
    height_v = cut_mocap(height) if height is not None else None

    bwh = (np.median(mass_v) if mass_v is not None else 1.0) * GRAVITY * \
          (np.median(height_v) if height_v is not None else 1.0)
    if not np.isfinite(bwh) or bwh <= 0:
        return None

    id_m = _full_id(jac_m, rot_m, cop_m, qfrc_m, grf, mom)
    # By default the ground-truth (MoCap) calc-aligned COP is used everywhere;
    # only the Jacobian, rotation, and qfrc_inverse are allowed to differ.
    cop_v_eff = cop_v if include_processed_cop_attribution else cop_m
    id_v = _full_id(jac_v, rot_v, cop_v_eff, qfrc_v, grf, mom)

    contrib = {
        "qfrc_inverse": qfrc_m - qfrc_v,
        "jacobian": id_m - _full_id(jac_v, rot_m, cop_m, qfrc_m, grf, mom),
        "rotation": id_m - _full_id(jac_m, rot_v, cop_m, qfrc_m, grf, mom),
    }
    if include_processed_cop_attribution:
        contrib["cop"] = id_m - _full_id(jac_m, rot_m, cop_v, qfrc_m, grf, mom)
    factor_sum = sum(contrib.values())
    contrib["total"] = id_m - id_v
    contrib["interaction"] = contrib["total"] - factor_sum
    # Normalise to % BW*H.
    contrib = {k: v / bwh * 100.0 for k, v in contrib.items()}

    stance_l = contact[:, 1] > 0.5

    out: Dict[str, np.ndarray] = {}
    out["_alignment_offset"] = np.array([float(alignment["offset"])])
    out["_alignment_score"] = np.array([
        float(alignment["score"]) if alignment["score"] is not None else np.nan
    ])
    out["_alignment_overlap_before_edge_trim"] = np.array([float(alignment["overlap"])])
    out["_frames_after_edge_trim"] = np.array([float(hi - lo)])
    for dof_idx, name, _side in DOF_INFO:
        mask = stance_l
        if not np.any(mask):
            continue
        for term in contrib:
            vals = np.abs(contrib[term][mask, dof_idx])
            out[f"{name}|{term}|sum"] = np.array([float(np.sum(vals))])
            out[f"{name}|{term}|sumsq"] = np.array([float(np.sum(vals ** 2))])
            out[f"{name}|{term}|n"] = np.array([float(np.sum(mask))])

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=str, default=None,
                    help="OpenCap dataset root (default: <repo>/OpenCapSubjects_Filt).")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: <repo>/inference_results/KinematicTorqueAttribution).")
    ap.add_argument("--exclude_subjects", type=str, default="",
                    help="Comma-separated subjects to skip, e.g. 'subject5'.")
    ap.add_argument("--trim_edges", type=int, default=20,
                    help="Drop this many frames from each end of the temporally aligned overlap.")
    ap.add_argument("--hold_gt_cop", action="store_true",
                    help="Deprecated/no-op: MoCap COP is now always held fixed unless "
                         "--include_processed_cop_attribution is passed.")
    ap.add_argument("--include_processed_cop_attribution", action="store_true",
                    help="Also swap ProcessedData COP as a separate counterfactual term. "
                         "By default, MoCap COP is used for both MoCap and ProcessedData "
                         "reconstructions.")
    ap.add_argument("--allow_truncate_misaligned", action="store_true",
                    help="Allow the previous behavior of truncating MoCap/ProcessedData arrays "
                         "to the shortest length. By default, trials with length mismatches are "
                         "skipped so the analysis only uses frame-aligned trials.")
    ap.add_argument("--no_temporal_align", action="store_true",
                    help="Disable MoCap-to-ProcessedData alignment by vertical GRF. If disabled, "
                         "length-mismatched trials are skipped unless --allow_truncate_misaligned "
                         "is also passed.")
    ap.add_argument("--align_max_lag", type=int, default=30,
                    help="Maximum absolute frame lag searched when aligning MoCap to ProcessedData.")
    args = ap.parse_args()
    terms = _term_list(args.include_processed_cop_attribution)

    data_dir = Path(args.data_dir).resolve() if args.data_dir else (PROJECT_ROOT / "OpenCapSubjects_Filt")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (artifact("inference_results") / "KinematicTorqueAttribution")
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {s.strip() for s in args.exclude_subjects.replace(",", " ").split() if s.strip()}

    trials = _discover_trials(data_dir, exclude)
    if not trials:
        raise SystemExit(f"No trials with both MoCap/ and ProcessedData/ under {data_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Found {len(trials)} trials across "
          f"{len({s for s, _ in trials})} subjects "
          f"(excluded: {sorted(exclude) or 'none'})")

    # subject -> dof -> term -> [sum, sumsq, n]
    per_subject: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    alignment_rows: List[Dict[str, object]] = []
    n_ok = 0
    for subject, trial_dir in trials:
        res = _process_trial(
            trial_dir,
            args.trim_edges,
            args.include_processed_cop_attribution,
            args.allow_truncate_misaligned,
            not args.no_temporal_align,
            args.align_max_lag,
        )
        if res is None:
            continue
        n_ok += 1
        alignment_rows.append({
            "subject": subject,
            "trial": trial_dir.name,
            "offset_frames_mocap_equals_processed_plus": float(res["_alignment_offset"][0]),
            "alignment_score": float(res["_alignment_score"][0]),
            "overlap_before_edge_trim": int(res["_alignment_overlap_before_edge_trim"][0]),
            "frames_after_edge_trim": int(res["_frames_after_edge_trim"][0]),
            "trim_edges": int(args.trim_edges),
        })
        bucket = per_subject.setdefault(subject, {})
        for _, name, _side in DOF_INFO:
            dof_b = bucket.setdefault(name, {t: np.zeros(3) for t in terms})
            for term in terms:
                key = f"{name}|{term}|sum"
                if key in res:
                    dof_b[term][0] += float(res[f"{name}|{term}|sum"][0])
                    dof_b[term][1] += float(res[f"{name}|{term}|sumsq"][0])
                    dof_b[term][2] += float(res[f"{name}|{term}|n"][0])

    if not per_subject:
        raise SystemExit("No trials processed successfully.")
    print(f"Processed {n_ok} trials.")

    # Per-subject MAE (frame-weighted), then cohort mean +/- std (subject-weighted).
    subjects = sorted(per_subject.keys())
    cohort: Dict[str, Dict[str, Dict[str, float]]] = {}
    factor_terms = [t for t in terms if t not in ("total", "interaction")]
    for _, name, _side in DOF_INFO:
        cohort[name] = {}
        for term in terms:
            subj_mae = []
            for s in subjects:
                acc = per_subject[s].get(name, {}).get(term)
                if acc is not None and acc[2] > 0:
                    subj_mae.append(acc[0] / acc[2])
            if subj_mae:
                cohort[name][term] = {
                    "mean": float(np.mean(subj_mae)),
                    "std": float(np.std(subj_mae, ddof=0)),
                    "n_subjects": len(subj_mae),
                }
            else:
                cohort[name][term] = {"mean": float("nan"), "std": float("nan"), "n_subjects": 0}
        pooled_subj = []
        summed_subj = []
        for s in subjects:
            maes = []
            for term in factor_terms:
                acc = per_subject[s].get(name, {}).get(term)
                if acc is not None and acc[2] > 0:
                    maes.append(acc[0] / acc[2])
            if len(maes) == len(factor_terms):
                arr = np.asarray(maes, dtype=np.float64)
                pooled_subj.append(float(np.sqrt(np.sum(arr ** 2))))
                summed_subj.append(float(np.sum(arr)))
        cohort[name]["pooled_factors"] = {
            "mean": float(np.mean(pooled_subj)) if pooled_subj else float("nan"),
            "std": float(np.std(pooled_subj, ddof=0)) if pooled_subj else float("nan"),
            "n_subjects": len(pooled_subj),
        }
        cohort[name]["summed_factors"] = {
            "mean": float(np.mean(summed_subj)) if summed_subj else float("nan"),
            "std": float(np.std(summed_subj, ddof=0)) if summed_subj else float("nan"),
            "n_subjects": len(summed_subj),
        }

    # ---- write CSV ----
    csv_path = out_dir / "kinematic_attribution_per_dof.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["dof"]
        for term in terms:
            header += [f"{term}_mean_pctBWH", f"{term}_std_pctBWH"]
        header += [
            "summed_factors_mean_pctBWH",
            "summed_factors_std_pctBWH",
            "pooled_factors_mean_pctBWH",
            "pooled_factors_std_pctBWH",
        ]
        w.writerow(header)
        for _, name, _side in DOF_INFO:
            row = [name]
            for term in terms:
                row += [f"{cohort[name][term]['mean']:.4f}", f"{cohort[name][term]['std']:.4f}"]
            row += [
                f"{cohort[name]['summed_factors']['mean']:.4f}",
                f"{cohort[name]['summed_factors']['std']:.4f}",
                f"{cohort[name]['pooled_factors']['mean']:.4f}",
                f"{cohort[name]['pooled_factors']['std']:.4f}",
            ]
            w.writerow(row)

    alignment_csv_path = out_dir / "temporal_alignment_report.csv"
    with alignment_csv_path.open("w", newline="") as fh:
        fieldnames = [
            "subject",
            "trial",
            "offset_frames_mocap_equals_processed_plus",
            "alignment_score",
            "overlap_before_edge_trim",
            "frames_after_edge_trim",
            "trim_edges",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in alignment_rows:
            w.writerow(row)

    summary = {
        "data_dir": str(data_dir),
        "n_trials_processed": n_ok,
        "subjects": subjects,
        "hold_gt_cop": not bool(args.include_processed_cop_attribution),
        "include_processed_cop_attribution": bool(args.include_processed_cop_attribution),
        "allow_truncate_misaligned": bool(args.allow_truncate_misaligned),
        "temporal_align": not bool(args.no_temporal_align),
        "align_max_lag": int(args.align_max_lag),
        "trim_edges": int(args.trim_edges),
        "temporal_alignment": alignment_rows,
        "attributed_factors": [t for t in terms if t not in ("total", "interaction")],
        "units": "percent of body-weight x height (BW*H), left-foot stance-phase MAE",
        "force_source": "ProcessedData/GRF_Cleaned.npy and ProcessedData/Moment_Cleaned.npy",
        "cop_source": "MoCap/COP_CalcFrame_GroundAligned.npy unless include_processed_cop_attribution=true",
        "stance_mask_source": "ProcessedData/contactBoolean.npy left foot",
        "summed_factors": "sum of per-factor stance MAE %BW*H values; useful as a stacked visual but pessimistic when factors cancel",
        "pooled_factors": "sqrt(sum(per-factor stance MAE %BW*H squared)) computed per subject, then averaged across subjects",
        "per_dof": cohort,
    }
    (out_dir / "kinematic_attribution_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- console table ----
    print("\nPer-DOF left-stance torque error attribution (% BW*H, cohort mean):")
    cols = [t for t in terms if t not in ("total", "interaction")] + ["summed_factors", "pooled_factors", "interaction", "total"]
    head = f"{'DOF':20s}" + "".join(f"{c[:9]:>11s}" for c in cols)
    print(head)
    print("-" * len(head))
    for _, name, _side in DOF_INFO:
        line = f"{name:20s}"
        for c in cols:
            line += f"{cohort[name][c]['mean']:11.3f}"
        print(line)

    # ---- stacked bar plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for _, n, side in DOF_INFO if side != "R"]
        parts = [t for t in terms if t not in ("total", "interaction")]
        colors = {"qfrc_inverse": "#2E86AB", "jacobian": "#E94F37",
                  "rotation": "#F6AE2D", "cop": "#5B8C5A"}
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 6))
        bottom = np.zeros(len(names))
        for term in parts:
            vals = np.array([cohort[n][term]["mean"] for n in names])
            ax.bar(x, vals, bottom=bottom, label=term, color=colors[term])
            bottom += vals
        totals = np.array([cohort[n]["total"]["mean"] for n in names])
        pooled = np.array([cohort[n]["pooled_factors"]["mean"] for n in names])
        ax.plot(x, totals, "k_", markersize=18, markeredgewidth=2, label="total (all-video)")
        ax.plot(x, pooled, "ko", markersize=4, label="pooled factors")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Stance torque error (% BW*H)")
        ax.set_title("Per-DOF torque error attributed to MoCap-vs-video kinematic terms")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "kinematic_attribution_stacked_bar.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")

    print(f"\nWrote:\n  {csv_path}\n  {out_dir / 'kinematic_attribution_summary.json'}"
          f"\n  {alignment_csv_path}"
          f"\n  {out_dir / 'kinematic_attribution_stacked_bar.png'}")


if __name__ == "__main__":
    main()
