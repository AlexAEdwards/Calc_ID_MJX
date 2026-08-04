#!/usr/bin/env python3
"""Estimate each subject's mass from vertical GRF + CoM kinematics, and compare
to the reported mass in Patient_MD.json.

Physics
-------
Vertical Newton's 2nd law for the whole body, integrated over a trial:

    m * (a_cm + g) = F_grf(t)
    integral:  m * (dv_cm + g*dt) = INT F_grf dt

Dividing the integral form by the trial duration turns every term into a trial
mean, which is robust to sample rate and exact trim window:

    m = mean(F_vert) / (g + mean(a_cm,vert))

Two estimators are reported per trial:
  * naive     : m = mean(F_vert) / g                (assumes dv_cm = 0)
  * corrected : m = mean(F_vert) / (g + mean(a_cm)) (uses kinematic CoM accel)

GRF source (decided with the user)
----------------------------------
  * Subjects with ProcessedData/GRF_NoFilt_Trimmed.npy  -> use it (non-filtered, trimmed).
  * SUBJ* (older/younger cohort) lack that file          -> fall back to GRF_Cleaned.npy
    (filtered). Filtering preserves the trial mean to <0.25%, so the mass estimate
    is effectively identical; such rows are flagged grf_source='cleaned_fallback'.

All processed-space arrays (GRF, COM_Acc_Global) are mutually aligned with
vertical on column index 2; the script auto-detects the vertical axis defensively.
"""
from __future__ import annotations

import json
import os
from glob import glob

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81  # m/s^2, matches the dataset's mass*g convention
MIN_FRAMES = 80  # exclude trials shorter than this (too few frames to trust the trial mean)
EDGE_TRIM = 15   # drop this many frames off each end before estimating (analysis-only; files untouched).
                 # Guards against un-captured double-support / loading transients at trial edges.

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "mass_estimation",
)
FIG_DIR = os.path.join(OUT_DIR, "figures")


def cohort_of(subject: str) -> str:
    """One label per source folder-prefix. Order matters: SUBJ / S_GAH_ before bare S.
    SUBJ* is a public stroke dataset added after GRF_NoFilt_Trimmed was generated, so it
    is the only group on the GRF_Cleaned fallback."""
    if subject[0].isdigit():
        return "numeric"
    if subject.startswith("SUBJ"):
        return "Stroke"
    if subject.startswith("GaitRetraining"):  # includes GaitRetraining_SubjectR*
        return "GaitRetraining"
    if subject.startswith("OA"):
        return "OA"
    if subject.startswith("S_GAH_"):
        return "S_GAH"
    if subject.startswith("Y"):
        return "Y"
    if subject.startswith("S"):
        return "S"
    return "Other"


def vertical_grf_series(grf: np.ndarray) -> tuple[np.ndarray, int]:
    """Total vertical GRF (sum of both feet). GRF is (n, 6): foot1 xyz, foot2 xyz.
    Vertical axis = per-foot column (0..2) with the largest positive mean."""
    v = int(np.argmax(grf[:, 0:3].mean(0)))
    return grf[:, v] + grf[:, v + 3], v


def reported_mass(subject_dir: str) -> float | None:
    md = os.path.join(subject_dir, "Patient_MD.json")
    if not os.path.isfile(md):
        return None
    try:
        return float(json.load(open(md)).get("Mass_kg"))
    except (ValueError, TypeError):
        return None


def process_trial(trial_dir: str) -> dict | None:
    pd_dir = os.path.join(trial_dir, "ProcessedData")
    nofilt = os.path.join(pd_dir, "GRF_NoFilt_Trimmed.npy")
    cleaned = os.path.join(pd_dir, "GRF_Cleaned.npy")
    if os.path.isfile(nofilt):
        grf_path, source = nofilt, "nofilt_trimmed"
    elif os.path.isfile(cleaned):
        grf_path, source = cleaned, "cleaned_fallback"
    else:
        return None

    grf = np.load(grf_path)
    if grf.ndim != 2 or grf.shape[1] < 6 or np.isnan(grf).any():
        return None
    if grf.shape[0] < MIN_FRAMES:
        return None
    # Analysis-only edge trim (does not modify files): drop EDGE_TRIM frames off each end.
    # MIN_FRAMES (>2*EDGE_TRIM) guarantees a non-empty interior remains.
    grf = grf[EDGE_TRIM:-EDGE_TRIM]

    v_grf, v_axis = vertical_grf_series(grf)
    mean_vgrf = float(v_grf.mean())
    peak_vgrf = float(v_grf.max())

    # Per-foot GRF-quality diagnostics (surface data artifacts seen in OY cohort):
    #   * a foot that never unloads (% frames < 10 N ~ 0) -> baseline offset / overground.
    #   * vertical peak below reported body weight -> physically impossible in walking,
    #     i.e. GRF and reported mass are mutually inconsistent.
    foot1, foot2 = grf[:, v_axis], grf[:, v_axis + 3]
    swing_frac_min = float(min((foot1 < 10).mean(), (foot2 < 10).mean()))

    # CoM vertical acceleration (kinematic, independent of GRF). Vertical = axis
    # with the largest std (vertical accel dominates during gait).
    mean_acc = 0.0
    acc_ok = False
    acc_path = os.path.join(pd_dir, "COM_Acc_Global.npy")
    if os.path.isfile(acc_path):
        acc = np.load(acc_path)
        if acc.ndim == 2 and acc.shape[1] >= 3 and not np.isnan(acc).any():
            acc = acc[EDGE_TRIM:-EDGE_TRIM]  # keep frame-aligned with the edge-trimmed GRF
            a_axis = int(np.argmax(acc.std(0)))
            mean_acc = float(acc[:, a_axis].mean())
            acc_ok = True

    # Trial duration / sample count (for reporting + aggregation weighting)
    n = grf.shape[0]
    duration = np.nan
    tpath = os.path.join(trial_dir, "Motion", "Time.npy")
    if os.path.isfile(tpath):
        t = np.load(tpath)
        duration = float(t[-1] - t[0])

    treadmill = np.nan
    tm = os.path.join(trial_dir, "Motion", "treadmill_speed.npy")
    if os.path.isfile(tm):
        try:
            treadmill = float(np.load(tm).reshape(-1)[0])
        except Exception:
            pass

    est_naive = mean_vgrf / G
    est_corr = mean_vgrf / (G + mean_acc)

    return dict(
        grf_source=source,
        v_axis=v_axis,
        n_frames=n,
        duration_s=duration,
        treadmill_speed=treadmill,
        mean_vGRF_N=mean_vgrf,
        peak_vGRF_N=peak_vgrf,
        swing_frac_min=swing_frac_min,
        mean_vCOMacc=mean_acc if acc_ok else np.nan,
        est_mass_naive_kg=est_naive,
        est_mass_corrected_kg=est_corr,
    )


def exclude_trial_outliers(vals: np.ndarray, thresh: float = 3.0) -> np.ndarray:
    """Robust within-subject outlier removal on per-trial estimates.
    Drops trials whose modified z-score |x-median|/(1.4826*MAD) > thresh.
    No-ops when <4 trials (too few to define an outlier) or MAD==0."""
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 4:
        return vals
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        return vals
    keep = np.abs(vals - med) / (1.4826 * mad) <= thresh
    return vals[keep]


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    rows = []
    subjects = sorted(
        d for d in os.listdir(DATASET)
        if os.path.isdir(os.path.join(DATASET, d))
        and os.path.isfile(os.path.join(DATASET, d, "Patient_MD.json"))
    )
    for subj in subjects:
        sdir = os.path.join(DATASET, subj)
        rmass = reported_mass(sdir)
        if rmass is None:
            continue
        for trial_dir in sorted(glob(os.path.join(sdir, "Trial_*"))):
            res = process_trial(trial_dir)
            if res is None:
                continue
            res.update(
                subject=subj,
                cohort=cohort_of(subj),
                trial=os.path.basename(trial_dir),
                reported_mass_kg=rmass,
            )
            rows.append(res)

    if not rows:
        raise SystemExit("No trials processed -- check dataset path.")

    df = pd.DataFrame(rows)
    df["err_naive_pct"] = 100 * (df.est_mass_naive_kg - df.reported_mass_kg) / df.reported_mass_kg
    df["err_corrected_pct"] = 100 * (df.est_mass_corrected_kg - df.reported_mass_kg) / df.reported_mass_kg
    # GRF-quality flags (do not exclude; mark for inspection / filtering)
    #   peak_below_BW : vertical peak < reported body weight -> impossible in gait,
    #                   so GRF & reported mass are mutually inconsistent.
    #   foot_no_unload: some foot never drops below 10 N -> plate offset / overground.
    df["flag_peak_below_BW"] = df.peak_vGRF_N < (df.reported_mass_kg * G)
    df["flag_foot_no_unload"] = df.swing_frac_min < 0.05
    df["grf_quality_ok"] = ~(df.flag_peak_below_BW | df.flag_foot_no_unload)
    # plausibility flag (does not exclude; just marks for inspection)
    df["plausible"] = (df.mean_vGRF_N > 200) & df.est_mass_corrected_kg.between(25, 200)

    cols = [
        "subject", "cohort", "trial", "grf_source", "n_frames", "duration_s",
        "treadmill_speed", "v_axis", "reported_mass_kg", "mean_vGRF_N",
        "peak_vGRF_N", "swing_frac_min", "mean_vCOMacc", "est_mass_naive_kg",
        "est_mass_corrected_kg", "err_naive_pct", "err_corrected_pct",
        "flag_peak_below_BW", "flag_foot_no_unload", "grf_quality_ok", "plausible",
    ]
    df = df[cols].sort_values(["subject", "trial"]).reset_index(drop=True)
    per_trial_csv = os.path.join(OUT_DIR, "per_trial_mass_estimates.csv")
    df.to_csv(per_trial_csv, index=False)

    # ---- per-subject aggregation (median over that subject's trials) ----
    # Between-trial variability (consistency) is computed AFTER excluding trial-level
    # outliers within each subject; reported as std (kg) and CV (% of subject mean).
    subj_rows = []
    for subj, g in df[df.plausible].groupby("subject"):
        est = g.est_mass_corrected_kg.values
        kept = exclude_trial_outliers(est)
        std_kept = float(kept.std(ddof=1)) if kept.size >= 2 else np.nan
        cv_kept = float(100 * std_kept / kept.mean()) if kept.size >= 2 else np.nan
        subj_rows.append(dict(
            subject=subj,
            cohort=g.cohort.iloc[0],
            grf_source=g.grf_source.iloc[0],
            n_trials=len(g),
            n_trials_kept=int(kept.size),
            n_trial_outliers=int(len(est) - kept.size),
            grf_quality_ok_frac=float(g.grf_quality_ok.mean()),
            reported_mass_kg=g.reported_mass_kg.iloc[0],
            est_mass_naive_kg=float(np.nanmedian(g.est_mass_naive_kg.values)),
            est_mass_corrected_kg=float(np.nanmedian(kept)) if kept.size else np.nan,
            trial_std_kg=std_kept,         # between-trial std after outlier exclusion
            trial_cv_pct=cv_kept,          # between-trial CV% after outlier exclusion
        ))
    sdf = pd.DataFrame(subj_rows)
    sdf["err_naive_pct"] = 100 * (sdf.est_mass_naive_kg - sdf.reported_mass_kg) / sdf.reported_mass_kg
    sdf["err_corrected_pct"] = 100 * (sdf.est_mass_corrected_kg - sdf.reported_mass_kg) / sdf.reported_mass_kg
    sdf = sdf.sort_values("subject").reset_index(drop=True)
    per_subj_csv = os.path.join(OUT_DIR, "per_subject_mass_estimates.csv")
    sdf.to_csv(per_subj_csv, index=False)

    make_figures(df, sdf)
    write_summary(df, sdf, per_trial_csv, per_subj_csv)


# ----------------------------------------------------------------------------- figures
COHORT_COLORS = {
    "numeric": "#1f77b4", "OA": "#ff7f0e", "Y": "#2ca02c", "S": "#9467bd",
    "S_GAH": "#8c564b", "GaitRetraining": "#e377c2", "Stroke": "#d62728",
    "Other": "#7f7f7f",
}
COHORT_ORDER = ["numeric", "OA", "Y", "S", "S_GAH", "GaitRetraining", "Stroke", "Other"]


def _pearson(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def make_figures(df: pd.DataFrame, sdf: pd.DataFrame) -> None:
    rep = sdf.reported_mass_kg.values
    est = sdf.est_mass_corrected_kg.values

    # Fig 1: estimated vs reported (subject level)
    fig, ax = plt.subplots(figsize=(7, 7))
    for c, sub in sdf.groupby("cohort"):
        ax.scatter(sub.reported_mass_kg, sub.est_mass_corrected_kg, s=28,
                   alpha=0.75, label=c, color=COHORT_COLORS.get(c, "gray"),
                   edgecolor="k", linewidth=0.3)
    lo, hi = np.nanmin([rep, est]) - 3, np.nanmax([rep, est]) + 3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    slope, intercept = np.polyfit(rep, est, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, slope * xs + intercept, "r-", lw=1.2,
            label=f"fit: y={slope:.3f}x+{intercept:.1f}")
    r = _pearson(rep, est)
    mae = np.nanmean(np.abs(est - rep))
    bias = np.nanmean(est - rep)
    ax.set_title(f"Estimated vs reported mass (subject level, corrected)\n"
                 f"r={r:.3f}  MAE={mae:.2f} kg  mean bias={bias:+.2f} kg  n={len(sdf)}")
    ax.set_xlabel("Reported mass (kg)")
    ax.set_ylabel("Estimated mass (kg)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "01_estimated_vs_reported.png"), dpi=150)
    plt.close(fig)

    # Fig 2: Bland-Altman (subject level)
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_ax = (rep + est) / 2
    diff = est - rep
    for c, sub in sdf.groupby("cohort"):
        m = (sub.reported_mass_kg.values + sub.est_mass_corrected_kg.values) / 2
        d = sub.est_mass_corrected_kg.values - sub.reported_mass_kg.values
        ax.scatter(m, d, s=28, alpha=0.75, label=c, color=COHORT_COLORS.get(c, "gray"),
                   edgecolor="k", linewidth=0.3)
    md = np.nanmean(diff); sd = np.nanstd(diff)
    for y, ls, lab in [(md, "-", f"mean {md:+.2f}"),
                       (md + 1.96 * sd, "--", f"+1.96SD {md+1.96*sd:+.2f}"),
                       (md - 1.96 * sd, "--", f"-1.96SD {md-1.96*sd:+.2f}")]:
        ax.axhline(y, color="r", ls=ls, lw=1)
        ax.text(ax.get_xlim()[1], y, " " + lab, va="center", fontsize=8, color="r")
    ax.set_title("Bland-Altman: estimated - reported mass (subject level)")
    ax.set_xlabel("Mean of estimated & reported (kg)")
    ax.set_ylabel("Estimated - reported (kg)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "02_bland_altman.png"), dpi=150)
    plt.close(fig)

    # Fig 3: error distribution (subject level), naive vs corrected
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(np.nanmin(sdf.err_corrected_pct) - 1,
                       np.nanmax(sdf.err_corrected_pct) + 1, 40)
    ax.hist(sdf.err_naive_pct, bins=bins, alpha=0.45, label=f"naive (med {sdf.err_naive_pct.median():+.2f}%)")
    ax.hist(sdf.err_corrected_pct, bins=bins, alpha=0.55, label=f"corrected (med {sdf.err_corrected_pct.median():+.2f}%)")
    ax.axvline(0, color="k", lw=1)
    ax.set_title("Per-subject % error vs reported mass")
    ax.set_xlabel("(estimated - reported) / reported  [%]")
    ax.set_ylabel("subjects")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "03_error_histogram.png"), dpi=150)
    plt.close(fig)

    # Fig 4: trend diagnostics (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # (a) error vs reported mass -> proportional bias?
    ax = axs[0, 0]
    for c, sub in sdf.groupby("cohort"):
        ax.scatter(sub.reported_mass_kg, sub.err_corrected_pct, s=24, alpha=0.7,
                   color=COHORT_COLORS.get(c, "gray"), label=c, edgecolor="k", linewidth=0.3)
    sl, ic = np.polyfit(rep, sdf.err_corrected_pct.values, 1)
    xs = np.array([rep.min(), rep.max()])
    ax.plot(xs, sl * xs + ic, "r-", lw=1.2, label=f"slope {sl:+.3f}%/kg")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("Reported mass (kg)")
    ax.set_ylabel("% error (corrected)"); ax.set_title("(a) Error vs reported mass")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) error by cohort (box)
    ax = axs[0, 1]
    groups, labels = [], []
    for c in COHORT_ORDER:
        v = sdf.loc[sdf.cohort == c, "err_corrected_pct"].dropna().values
        if v.size:
            groups.append(v); labels.append(f"{c}\n(n={v.size})")
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.tick_params(axis="x", labelsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("% error (corrected)"); ax.set_title("(b) Error by cohort")
    ax.grid(alpha=0.3, axis="y")

    # (c) within-subject trial spread vs reported mass
    ax = axs[1, 0]
    ax.scatter(sdf.reported_mass_kg, sdf.trial_std_kg, s=24, alpha=0.7,
               c=[COHORT_COLORS.get(c, "gray") for c in sdf.cohort])
    ax.set_xlabel("Reported mass (kg)"); ax.set_ylabel("Std of per-trial est (kg)")
    ax.set_title("(c) Within-subject trial repeatability"); ax.grid(alpha=0.3)

    # (d) naive vs corrected per-subject error (effect of CoM correction)
    ax = axs[1, 1]
    ax.scatter(sdf.err_naive_pct, sdf.err_corrected_pct, s=24, alpha=0.7,
               c=[COHORT_COLORS.get(c, "gray") for c in sdf.cohort])
    lim = [min(sdf.err_naive_pct.min(), sdf.err_corrected_pct.min()) - 1,
           max(sdf.err_naive_pct.max(), sdf.err_corrected_pct.max()) + 1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("% error naive"); ax.set_ylabel("% error corrected")
    ax.set_title("(d) Effect of CoM-accel correction"); ax.grid(alpha=0.3)
    ax.set_xlim(lim); ax.set_ylim(lim)

    fig.suptitle("Mass-estimation trend diagnostics (subject level)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "04_trend_diagnostics.png"), dpi=150)
    plt.close(fig)

    # Fig 5: agreement after dropping GRF-quality-flagged subjects (>=50% clean trials)
    clean = sdf[sdf.grf_quality_ok_frac >= 0.5]
    fig, ax = plt.subplots(figsize=(7, 7))
    for c, sub in clean.groupby("cohort"):
        ax.scatter(sub.reported_mass_kg, sub.est_mass_corrected_kg, s=28, alpha=0.75,
                   label=c, color=COHORT_COLORS.get(c, "gray"), edgecolor="k", linewidth=0.3)
    cr = clean.reported_mass_kg.values; ce = clean.est_mass_corrected_kg.values
    lo, hi = np.nanmin([cr, ce]) - 3, np.nanmax([cr, ce]) + 3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    r = _pearson(cr, ce); mae = np.nanmean(np.abs(ce - cr)); bias = np.nanmean(ce - cr)
    ax.set_title(f"Estimated vs reported -- GRF-quality-clean subjects only\n"
                 f"r={r:.3f}  MAE={mae:.2f} kg  bias={bias:+.2f} kg  n={len(clean)} "
                 f"(dropped {len(sdf)-len(clean)} flagged)")
    ax.set_xlabel("Reported mass (kg)"); ax.set_ylabel("Estimated mass (kg)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "05_clean_only_agreement.png"), dpi=150)
    plt.close(fig)

    # Fig 6: within-subject between-trial consistency by cohort (outliers excluded)
    multi = sdf[sdf.n_trials_kept >= 2]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, lab in [(axs[0], "trial_std_kg", "between-trial std (kg)"),
                         (axs[1], "trial_cv_pct", "between-trial CV (%)")]:
        groups, labels = [], []
        for c in COHORT_ORDER:
            vals = multi.loc[multi.cohort == c, col].dropna().values
            if vals.size:
                groups.append(vals); labels.append(f"{c}\n(n={vals.size})")
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.set_ylabel(lab); ax.tick_params(axis="x", labelsize=8); ax.grid(alpha=0.3, axis="y")
    axs[0].set_title("Per-subject weight-estimate spread across trials")
    axs[1].set_title("Per-subject weight-estimate spread (normalized)")
    fig.suptitle("Within-subject between-trial consistency by cohort (trial outliers excluded)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "06_within_subject_variability.png"), dpi=150)
    plt.close(fig)


def cohort_accuracy(sdf: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort accuracy of the (corrected) estimate vs reported mass, subject-level."""
    def agg(g):
        err = g.err_corrected_pct.values
        dkg = (g.est_mass_corrected_kg - g.reported_mass_kg).values
        return pd.Series(dict(
            n_subjects=len(g),
            median_err_pct=float(np.median(err)),
            mean_err_pct=float(np.mean(err)),          # signed bias
            MAE_pct=float(np.mean(np.abs(err))),
            RMSE_pct=float(np.sqrt(np.mean(err ** 2))),
            MAE_kg=float(np.mean(np.abs(dkg))),
            pearson_r=_pearson(g.reported_mass_kg.values, g.est_mass_corrected_kg.values),
        ))
    return sdf.groupby("cohort").apply(agg, include_groups=False).round(3)


def cohort_variability(sdf: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort summary of within-subject between-trial consistency (outliers excluded).
    Only subjects with >=2 kept trials contribute a variability value."""
    v = sdf[sdf.n_trials_kept >= 2]
    out = v.groupby("cohort").agg(
        n_subjects_multi_trial=("subject", "size"),
        median_trial_std_kg=("trial_std_kg", "median"),
        mean_trial_std_kg=("trial_std_kg", "mean"),
        median_trial_cv_pct=("trial_cv_pct", "median"),
        p90_trial_cv_pct=("trial_cv_pct", lambda s: float(s.quantile(0.90))),
        total_trial_outliers=("n_trial_outliers", "sum"),
    )
    return out.round(3)


def write_summary(df, sdf, per_trial_csv, per_subj_csv) -> None:
    def stats(s):
        s = s.dropna()
        return dict(median=float(s.median()), mean=float(s.mean()),
                    mae=float(s.abs().mean()), p5=float(s.quantile(.05)),
                    p95=float(s.quantile(.95)))
    clean = sdf[sdf.grf_quality_ok_frac >= 0.5]
    summary = dict(
        n_trials=int(len(df)),
        n_trials_flag_peak_below_BW=int(df.flag_peak_below_BW.sum()),
        n_trials_flag_foot_no_unload=int(df.flag_foot_no_unload.sum()),
        n_trials_grf_quality_ok=int(df.grf_quality_ok.sum()),
        n_subjects=int(len(sdf)),
        n_subjects_cleaned_fallback=int((sdf.grf_source == "cleaned_fallback").sum()),
        n_subjects_grf_quality_clean=int(len(clean)),
        pearson_r_subject=_pearson(sdf.reported_mass_kg.values, sdf.est_mass_corrected_kg.values),
        pearson_r_subject_clean=_pearson(clean.reported_mass_kg.values, clean.est_mass_corrected_kg.values),
        subject_err_corrected_pct=stats(sdf.err_corrected_pct),
        subject_err_naive_pct=stats(sdf.err_naive_pct),
        subject_err_corrected_pct_clean=stats(clean.err_corrected_pct),
        per_cohort=sdf.groupby("cohort").err_corrected_pct.agg(
            ["count", "median", "mean", lambda s: float(s.abs().mean())]
        ).rename(columns={"<lambda_0>": "mae"}).to_dict("index"),
        per_cohort_accuracy=cohort_accuracy(sdf).to_dict("index"),
        per_cohort_trial_variability=cohort_variability(sdf).to_dict("index"),
    )
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"Processed {summary['n_trials']} trials across {summary['n_subjects']} subjects "
          f"({summary['n_subjects_cleaned_fallback']} via cleaned-fallback).")
    print(f"Subject-level Pearson r (est vs reported): {summary['pearson_r_subject']:.3f}")
    sc = summary["subject_err_corrected_pct"]
    print(f"Subject %error (corrected): median {sc['median']:+.2f}%  MAE {sc['mae']:.2f}%  "
          f"[p5 {sc['p5']:+.2f}, p95 {sc['p95']:+.2f}]")
    print(f"GRF-quality flags: {summary['n_trials_flag_peak_below_BW']} trials peak<BW, "
          f"{summary['n_trials_flag_foot_no_unload']} trials foot-no-unload "
          f"({summary['n_trials_grf_quality_ok']}/{summary['n_trials']} clean)")
    scc = summary["subject_err_corrected_pct_clean"]
    print(f"Clean-GRF subjects (n={summary['n_subjects_grf_quality_clean']}): "
          f"r={summary['pearson_r_subject_clean']:.3f}  median {scc['median']:+.2f}%  MAE {scc['mae']:.2f}%")
    print("\nAccuracy of estimated vs reported weight (subject-level, corrected estimate):")
    print(cohort_accuracy(sdf).to_string())
    print("\nBetween-trial consistency of the weight estimate (per subject, outliers excluded):")
    cv = cohort_variability(sdf)
    print(cv.to_string())
    print(f"  ({int(sdf.n_trial_outliers.sum())} trial-level outliers excluded across all subjects)")
    print(f"\nWrote:\n  {per_trial_csv}\n  {per_subj_csv}\n  {os.path.join(OUT_DIR, 'summary.json')}")
    print(f"  figures -> {FIG_DIR}/01..04_*.png")


if __name__ == "__main__":
    main()
