#!/usr/bin/env python
"""Test_New_Features.py — fractional-factorial ablation of the new training features.

Sweeps five opt-in features added to train.py, each trained for a fixed number of
epochs on a fast subject subset with NO plots, and reports which features improve
validation accuracy using a proper factorial-effects analysis.

Factors (each OFF = prior behavior, ON = feature enabled):
    A  lr_schedule    warmup->cosine LR        (OFF = --no_lr_schedule true)
    B  huber          robust Huber loss        (OFF = MSE)
    C  honest_norm    stance-only Z-score stats (+ larger sample)
    D  contact_mixed  annealed predicted-contact masking in the torque path
    E  film           per-layer FiLM subject conditioning

Design: 2^(5-1) resolution-V fractional factorial, generator E = A*B*C*D (16 runs).
Main effects are aliased only with 4-way interactions and 2-way interactions only
with 3-way interactions, so every main effect and every 2-way interaction is clean.
An extra explicit "all-off" baseline run is added for reference (not used in the
effects math). The comparison metric is the best (min-over-epochs) validation
moment-MAE in %BW·h read from each run's rmse_history.json — a physical quantity on
the same fixed val split, so it is directly comparable across MSE/Huber/etc. runs and
independent of which epoch each run happened to select as "best".

Efficiency: runs are sequential (one GPU), share the JAX compilation cache, use a
window-level random subset (--window_frac of all windows pooled across every subject,
split --window_train_frac train / rest val), print only per-epoch summaries
(--quiet_steps), skip all plotting, and disable WandB. Use --resume to skip runs whose
results already exist.

Usage:
    python Test_New_Features.py                     # full 16-run sweep + baseline
    python Test_New_Features.py --epochs 20 --max_subjects 8
    python Test_New_Features.py --dry_run           # print the design, run nothing
    python Test_New_Features.py --analyze_only      # re-analyze existing results
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Factor order is fixed: A, B, C, D, E.
FACTORS = ["lr_schedule", "huber", "honest_norm", "contact_mixed", "film"]
FACTOR_LETTERS = {"lr_schedule": "A", "huber": "B", "honest_norm": "C",
                  "contact_mixed": "D", "film": "E"}

# Lower is better. Primary ranking metric = best (min-over-epochs) val joint-moment MAE
# %BW·h: the mean bilateral stance MAE across ankle, hip flexion, hip adduction, knee
# flexion, and knee adduction (KAM).
PRIMARY_METRIC = "joint_moment_mae_bwh"
SELECTION_METRIC = "joint_moment_mae"  # --best_model_metric value that matches PRIMARY_METRIC


def feature_flags(factor: str, on: bool):
    """Return the train_single_model.py CLI flags that set one factor on/off."""
    if factor == "lr_schedule":
        # Schedule is on by default; OFF means constant LR.
        return ["--no_lr_schedule", "false"] if on else ["--no_lr_schedule", "true"]
    if factor == "huber":
        return ["--robust_loss", "huber"] if on else ["--robust_loss", "mse"]
    if factor == "honest_norm":
        return (["--normalizer_stance_only", "true", "--normalizer_max_batches", "300"]
                if on else ["--normalizer_stance_only", "false", "--normalizer_max_batches", "100"])
    if factor == "contact_mixed":
        return ["--contact_mask_source", "mixed"] if on else ["--contact_mask_source", "gt"]
    if factor == "film":
        return ["--subject_film", "true"] if on else ["--subject_film", "false"]
    raise ValueError(f"unknown factor {factor}")


def build_design():
    """Return the list of runs: each is dict(name, levels{factor:bool}).

    16 resolution-V runs (E = A*B*C*D over {-1,+1}) plus one all-off baseline.
    """
    runs = []
    # Full factorial over the first four factors; E is determined by the generator.
    for combo in itertools.product([-1, 1], repeat=4):
        a, b, c, d = combo
        e = a * b * c * d  # generator E = ABCD -> resolution V
        levels = {
            "lr_schedule": a == 1,
            "huber": b == 1,
            "honest_norm": c == 1,
            "contact_mixed": d == 1,
            "film": e == 1,
        }
        code = "".join(f"{FACTOR_LETTERS[f]}{1 if levels[f] else 0}" for f in FACTORS)
        runs.append({"name": code, "levels": levels, "in_design": True})

    # Explicit all-off baseline (reference only; not part of the effects estimation).
    baseline_levels = {f: False for f in FACTORS}
    runs.append({"name": "BASELINE_ALLOFF", "levels": baseline_levels, "in_design": False})
    return runs


def run_cmd(run, args):
    """Assemble the train_single_model.py command for one run."""
    exp_name = f"{args.tag}_{run['name']}"
    cmd = [
        args.python, str(SCRIPT_DIR / "train_single_model.py"),
        "--exp_name", exp_name,
        "--epochs", str(args.epochs),
        # Fast subset: pool windows from ALL subjects, keep a fraction, split at the
        # window level (mixes subjects), so every run trains/validates on comparable data.
        "--window_split_frac", str(args.window_frac),
        "--window_train_frac", str(args.window_train_frac),
        "--best_model_metric", SELECTION_METRIC,  # align saved ckpt with the reported metric
        "--BestModelByTorque", "false",           # else it would override best_model_metric
        "--effect_diagnostics", "false",
        "--quiet_steps", "true",                 # only per-epoch summaries (faster/cleaner)
        "--no_plots", "true",
        "--no_wandb",
        "--force",
    ]
    if args.max_subjects:
        cmd += ["--max_subjects", str(args.max_subjects)]
    if args.data_dir:
        cmd += ["--data_dir", args.data_dir]
    if args.batch_size:
        cmd += ["--batch_size", str(args.batch_size)]
    for factor in FACTORS:
        cmd += feature_flags(factor, run["levels"][factor])
    return exp_name, cmd


def parse_result(exp_name):
    """Extract metrics for a completed run from its output artifacts."""
    out_dir = OUTPUTS_DIR / exp_name
    rmse_path = out_dir / "rmse_history.json"
    summary_path = out_dir / "training_summary.json"
    result = {"exp_name": exp_name, "status": "missing"}
    if rmse_path.exists():
        try:
            hist = json.loads(rmse_path.read_text())
            val = hist.get("val", {})

            def _min(key):
                xs = [v for v in val.get(key, []) if v is not None and v == v and v not in (float("inf"),)]
                return float(min(xs)) if xs else float("nan")

            result.update({
                "status": "ok",
                "joint_moment_mae_bwh": _min("joint_moment_mae_bwh"),
                "knee_adduction_mae_bwh": _min("knee_adduction_mae_bwh"),
                "moment_mae_bwh": _min("moment_mae_bwh"),
                "moments_rmse_Nm": _min("moments_overall_rmse_Nm"),
                "grf_rmse_N": _min("grf_overall_rmse_N"),
                "cop_rmse_m": _min("cop_overall_rmse_m"),
                "epochs_recorded": len(val.get("joint_moment_mae_bwh", [])),
            })
        except Exception as e:
            result["status"] = f"parse_error: {e}"
    if summary_path.exists():
        try:
            summ = json.loads(summary_path.read_text())
            result["best_val_loss"] = summ.get("best_val_loss")
            result["best_model_epoch"] = summ.get("best_model_epoch")
        except Exception:
            pass
    return result


def already_done(exp_name):
    r = parse_result(exp_name)
    return r["status"] == "ok" and r.get(PRIMARY_METRIC) == r.get(PRIMARY_METRIC)  # not NaN


def factorial_effects(design_rows, metric):
    """Main effects and 2-way interactions from the 16 resolution-V runs.

    Each factor level is coded -1/+1. For a balanced 2-level design the effect of a
    contrast column c is mean(y where c=+1) - mean(y where c=-1) = sum(y*c) / (N/2).
    Negative effect => the feature/interaction REDUCES the (lower-is-better) metric.
    """
    rows = [r for r in design_rows if r.get("in_design") and r["result"]["status"] == "ok"
            and r["result"].get(metric) == r["result"].get(metric)]
    n = len(rows)
    if n < 8:
        return None, f"only {n} usable design runs (need the 16-run design); effects not estimated"

    def sign(factor, row):
        return 1 if row["levels"][factor] else -1

    y = [r["result"][metric] for r in rows]
    grand_mean = sum(y) / n

    effects = []
    # Main effects
    for f in FACTORS:
        contrast = sum(yi * sign(f, r) for yi, r in zip(y, rows))
        effects.append((FACTOR_LETTERS[f], f, contrast / (n / 2.0), "main"))
    # 2-way interactions (clean in resolution V)
    for f1, f2 in itertools.combinations(FACTORS, 2):
        contrast = sum(yi * sign(f1, r) * sign(f2, r) for yi, r in zip(y, rows))
        label = FACTOR_LETTERS[f1] + FACTOR_LETTERS[f2]
        name = f"{f1} x {f2}"
        effects.append((label, name, contrast / (n / 2.0), "2-way"))

    return {"grand_mean": grand_mean, "n": n, "effects": effects}, None


def print_analysis(design_rows, args):
    metric = PRIMARY_METRIC
    unit = "%BW·h"
    print("\n" + "=" * 78)
    print(f"ABLATION ANALYSIS — primary metric: best val {metric} ({unit}, lower is better)")
    print("=" * 78)

    # Per-run table
    ok = [r for r in design_rows if r["result"]["status"] == "ok"]
    ok_sorted = sorted(ok, key=lambda r: (r["result"].get(metric) if r["result"].get(metric) == r["result"].get(metric) else float("inf")))
    print(f"\nRuns ranked by best val {metric} ({unit}):")
    print(f"  {'run':<20} {'A':>2}{'B':>2}{'C':>2}{'D':>2}{'E':>2}  {metric:>12}  {'momRMSE_Nm':>11}  {'grfRMSE_N':>10}")
    for r in ok_sorted:
        lv = r["levels"]
        flags = "".join(f"{1 if lv[f] else 0:>2}" for f in FACTORS)
        res = r["result"]
        print(f"  {r['name']:<20} {flags}  {res.get(metric, float('nan')):>12.4f}  "
              f"{res.get('moments_rmse_Nm', float('nan')):>11.4f}  {res.get('grf_rmse_N', float('nan')):>10.3f}")

    # Baseline reference
    base = next((r for r in design_rows if r["name"].endswith("ALLOFF")), None)
    if base and base["result"]["status"] == "ok":
        print(f"\n  Reference all-off baseline: {base['result'].get(metric, float('nan')):.4f} {unit}")

    # Factorial effects
    eff, err = factorial_effects(design_rows, metric)
    if err:
        print(f"\n⚠️  {err}")
        return
    print(f"\nGrand mean (16-run design): {eff['grand_mean']:.4f} {unit}  (n={eff['n']})")
    print("\nEFFECTS  (negative = LOWERS error = feature/interaction HELPS):")
    mains = sorted([e for e in eff["effects"] if e[3] == "main"], key=lambda e: e[2])
    print("\n  Main effects (ranked, most helpful first):")
    print(f"    {'':2}  {'feature':<16} {'effect':>10}  verdict")
    for letter, name, val, _ in mains:
        verdict = "HELPS" if val < 0 else ("hurts" if val > 0 else "neutral")
        print(f"    {letter:<2}  {name:<16} {val:>+10.4f}  {verdict}")
    inters = sorted([e for e in eff["effects"] if e[3] == "2-way"], key=lambda e: abs(e[2]), reverse=True)
    print("\n  Strongest 2-way interactions (|effect|, top 5):")
    for letter, name, val, _ in inters[:5]:
        print(f"    {letter:<3} {name:<28} {val:>+10.4f}")

    # Verdict summary
    helpers = [name for _, name, val, kind in mains if kind == "main" and val < 0]
    hurters = [name for _, name, val, kind in mains if kind == "main" and val > 0]
    print("\nSUMMARY:")
    print(f"  Features that improved accuracy: {', '.join(helpers) if helpers else 'none'}")
    print(f"  Features that hurt accuracy:     {', '.join(hurters) if hurters else 'none'}")
    if ok_sorted:
        best = ok_sorted[0]
        on = [f for f in FACTORS if best['levels'][f]]
        print(f"  Best single run: {best['name']}  ({best['result'].get(metric):.4f} {unit}); features on: {', '.join(on) if on else 'none'}")

    # Persist machine-readable results
    out = {
        "primary_metric": metric,
        "unit": unit,
        "grand_mean": eff["grand_mean"],
        "n_design_runs": eff["n"],
        "effects": [{"label": l, "name": nm, "effect": v, "kind": k} for l, nm, v, k in eff["effects"]],
        "runs": [{"name": r["name"], "levels": r["levels"], "in_design": r.get("in_design", False),
                  "result": r["result"]} for r in design_rows],
    }
    results_path = OUTPUTS_DIR / f"{args.tag}_ablation_results.json"
    results_path.write_text(json.dumps(out, indent=2))
    print(f"\n📝 Saved results + effects to: {results_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--window_frac", type=float, default=0.5,
                    help="Fraction of ALL windows (pooled across every subject) used per run. "
                         "Smaller = faster but noisier. Default 0.5.")
    ap.add_argument("--window_train_frac", type=float, default=0.7,
                    help="Of the sampled windows, fraction used for training (rest = validation). Default 0.7.")
    ap.add_argument("--max_subjects", type=int, default=0,
                    help="Optional extra cap on the subject pool before window sampling. 0 = all subjects.")
    ap.add_argument("--data_dir", type=str, default=None, help="Override dataset (else train_single_model default).")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--tag", type=str, default="AblatV", help="Prefix for run folders + results file.")
    ap.add_argument("--python", type=str, default=sys.executable, help="Interpreter for training subprocess.")
    ap.add_argument("--resume", action="store_true", help="Skip runs whose results already exist.")
    ap.add_argument("--dry_run", action="store_true", help="Print the design and commands; run nothing.")
    ap.add_argument("--analyze_only", action="store_true", help="Skip training; analyze existing run outputs.")
    args = ap.parse_args()

    design = build_design()
    print(f"Design: {sum(1 for r in design if r['in_design'])} resolution-V runs + "
          f"{sum(1 for r in design if not r['in_design'])} baseline "
          f"= {len(design)} total, {args.epochs} epochs each, "
          f"window_frac={args.window_frac} (train/val {args.window_train_frac:.0%}/"
          f"{1-args.window_train_frac:.0%})" + (f", max_subjects={args.max_subjects}" if args.max_subjects else ""))

    if args.dry_run:
        for r in design:
            exp_name, cmd = run_cmd(r, args)
            print(f"\n[{r['name']}]  in_design={r['in_design']}")
            print("  " + " ".join(cmd))
        return

    t0 = time.time()
    for i, r in enumerate(design, 1):
        exp_name, cmd = run_cmd(r, args)
        r["result"] = None
        if args.analyze_only:
            r["result"] = parse_result(exp_name)
            continue
        if args.resume and already_done(exp_name):
            print(f"\n⏭️  [{i}/{len(design)}] {exp_name}: results exist, skipping (resume).")
            r["result"] = parse_result(exp_name)
            continue
        print(f"\n{'#'*78}\n# [{i}/{len(design)}] RUN {exp_name}\n#   " + " ".join(cmd) + f"\n{'#'*78}", flush=True)
        rc = subprocess.run(cmd).returncode
        r["result"] = parse_result(exp_name)
        r["result"]["returncode"] = rc
        status = r["result"].get(PRIMARY_METRIC, float("nan"))
        print(f"   -> rc={rc}, best {PRIMARY_METRIC}={status}")

    elapsed = time.time() - t0
    if not args.analyze_only:
        print(f"\n⏱️  Sweep wall time: {elapsed/60:.1f} min for {len(design)} runs")
    print_analysis(design, args)


if __name__ == "__main__":
    main()
