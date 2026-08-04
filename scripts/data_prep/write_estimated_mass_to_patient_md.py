#!/usr/bin/env python3
"""Estimate each subject's mass from vertical GRF + CoM kinematics and write
the result back into Patient_MD.json in-place.

Identical estimation logic to estimate_mass_from_grf.py:
  - Excludes trials with < MIN_FRAMES frames.
  - Trims EDGE_TRIM frames off each end (analysis-only; data files untouched).
  - Per-trial: m = mean(F_vert) / (g + mean(a_cm,vert))
  - Per-subject: median of per-trial estimates after MAD-based outlier exclusion.

Patient_MD.json is updated with:
  - Mass_kg            : estimated mass (overwrites old value)
  - Mass_kg_reported   : original reported value (written once; never overwritten)
  - Mass_kg_est_std    : between-trial std of the kept estimates (kg)
  - Mass_kg_est_n      : number of trials used
  - Mass_kg_est_source : 'GRF_estimated'
"""
from __future__ import annotations

import json
import os
from glob import glob

import numpy as np

G = 9.81
MIN_FRAMES = 80
EDGE_TRIM = 15

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "TrustedDataSetNoised12Distributed_AllPatients_EstimatedWeights",
)


def cohort_of(subject: str) -> str:
    if subject[0].isdigit():       return "numeric"
    if subject.startswith("SUBJ"): return "Stroke"
    if subject.startswith("GaitRetraining"): return "GaitRetraining"
    if subject.startswith("OA"):   return "OA"
    if subject.startswith("S_GAH_"): return "S_GAH"
    if subject.startswith("Y"):    return "Y"
    if subject.startswith("S"):    return "S"
    return "Other"


def vertical_grf_mean(grf: np.ndarray) -> float:
    v = int(np.argmax(grf[:, 0:3].mean(0)))
    return float((grf[:, v] + grf[:, v + 3]).mean())


def com_acc_mean(pd_dir: str, n_frames_after_trim: int) -> float:
    path = os.path.join(pd_dir, "COM_Acc_Global.npy")
    if not os.path.isfile(path):
        return 0.0
    acc = np.load(path)
    if acc.ndim != 2 or acc.shape[1] < 3 or np.isnan(acc).any():
        return 0.0
    acc = acc[EDGE_TRIM:-EDGE_TRIM]
    if len(acc) != n_frames_after_trim:
        return 0.0
    return float(acc[:, int(np.argmax(acc.std(0)))].mean())


def estimate_trial(trial_dir: str) -> float | None:
    pd_dir = os.path.join(trial_dir, "ProcessedData")
    nofilt = os.path.join(pd_dir, "GRF_NoFilt_Trimmed.npy")
    cleaned = os.path.join(pd_dir, "GRF_Cleaned.npy")
    grf_path = nofilt if os.path.isfile(nofilt) else (cleaned if os.path.isfile(cleaned) else None)
    if grf_path is None:
        return None
    grf = np.load(grf_path)
    if grf.ndim != 2 or grf.shape[1] < 6 or np.isnan(grf).any() or grf.shape[0] < MIN_FRAMES:
        return None
    grf = grf[EDGE_TRIM:-EDGE_TRIM]
    if grf.shape[0] < 1:
        return None
    mean_f = vertical_grf_mean(grf)
    mean_a = com_acc_mean(pd_dir, len(grf))
    if mean_f < 50:  # implausibly low; skip
        return None
    return mean_f / (G + mean_a)


def mad_filter(vals: np.ndarray, thresh: float = 3.0) -> np.ndarray:
    """Remove per-trial outliers within a subject using modified z-score."""
    vals = vals[~np.isnan(vals)]
    if vals.size < 4:
        return vals
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        return vals
    return vals[np.abs(vals - med) / (1.4826 * mad) <= thresh]


def estimate_subject(subject_dir: str) -> dict | None:
    trials = sorted(glob(os.path.join(subject_dir, "Trial_*")))
    estimates = []
    for T in trials:
        e = estimate_trial(T)
        if e is not None:
            estimates.append(e)
    if not estimates:
        return None
    est = np.array(estimates)
    kept = mad_filter(est)
    if kept.size == 0:
        return None
    return dict(
        mass_kg=float(np.median(kept)),
        std_kg=float(kept.std(ddof=1)) if kept.size >= 2 else 0.0,
        n_trials=int(kept.size),
    )


def main() -> None:
    subjects = sorted(
        d for d in os.listdir(DATASET)
        if os.path.isdir(os.path.join(DATASET, d))
        and os.path.isfile(os.path.join(DATASET, d, "Patient_MD.json"))
    )

    updated = skipped = already_done = 0
    results = []

    for subj in subjects:
        sdir = os.path.join(DATASET, subj)
        md_path = os.path.join(sdir, "Patient_MD.json")
        md = json.load(open(md_path))

        est = estimate_subject(sdir)
        if est is None:
            print(f"  SKIP (no usable trials)  {subj}")
            skipped += 1
            continue

        # Preserve original reported mass the first time (never overwrite it)
        if "Mass_kg_reported" not in md:
            md["Mass_kg_reported"] = md.get("Mass_kg")

        old = md.get("Mass_kg")
        md["Mass_kg"] = round(est["mass_kg"], 4)
        md["Mass_kg_est_std"] = round(est["std_kg"], 4)
        md["Mass_kg_est_n"] = est["n_trials"]
        md["Mass_kg_est_source"] = "GRF_estimated"

        with open(md_path, "w") as f:
            json.dump(md, f, indent=2)

        cohort = cohort_of(subj)
        pct = 100 * (est["mass_kg"] - (md["Mass_kg_reported"] or 0)) / (md["Mass_kg_reported"] or 1)
        results.append(dict(subject=subj, cohort=cohort,
                            reported=md["Mass_kg_reported"], estimated=est["mass_kg"],
                            std=est["std_kg"], n=est["n_trials"], pct_diff=pct))
        updated += 1

    print(f"\nDone: {updated} subjects updated, {skipped} skipped (no usable trials).")

    # ---- per-cohort summary ----
    import collections
    by_cohort: dict[str, list] = collections.defaultdict(list)
    for r in results:
        by_cohort[r["cohort"]].append(r)

    print(f"\n{'Cohort':<18} {'n':>4}  {'median Δ%':>10}  {'MAE%':>7}  {'median std(kg)':>14}")
    print("-" * 60)
    for cohort in ["numeric","OA","Y","S","S_GAH","GaitRetraining","Stroke","Other"]:
        rows = by_cohort.get(cohort, [])
        if not rows:
            continue
        errs = np.array([r["pct_diff"] for r in rows])
        stds = np.array([r["std"] for r in rows])
        print(f"{cohort:<18} {len(rows):>4}  {np.median(errs):>+10.2f}%  "
              f"{np.mean(np.abs(errs)):>6.2f}%  {np.median(stds):>14.3f} kg")

    print(f"\nMass_kg updated in Patient_MD.json for all {updated} subjects.")
    print("Original values preserved as Mass_kg_reported.")


if __name__ == "__main__":
    main()
