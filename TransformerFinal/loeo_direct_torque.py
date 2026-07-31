"""Leave-one-experiment-out (LOEO) sweep for the direct joint-torque model.

Expects the nested dataset layout produced by
``scripts/reorganize_dataset_by_experiment.py``::

    Dataset/<Experiment>/<Subject>/Trial_#/ProcessedData/...

For every experiment folder the wrapper

1. trains a fresh direct-torque model on **all other** experiments
   (``train_directTorque.py --exclude_experiments <E>``),
2. runs inference with that model on the held-out experiment, writing results
   into ``<Subject>/Trial_#/inference_results/`` inside the dataset itself, and
3. once every experiment has been covered, aggregates accuracy per experiment
   and over the whole dataset.

Each round runs in its own subprocess so JAX/GPU memory is released between
models.  Completed rounds are skipped on re-run unless ``--force`` is passed, so
an interrupted sweep can simply be restarted.

    python TransformerFinal/loeo_direct_torque.py \
        --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients \
        --output_root outputs/DirectTorque_LOEO --epochs 40

    # metrics only, after the sweep has finished
    python TransformerFinal/loeo_direct_torque.py \
        --data_dir ... --output_root outputs/DirectTorque_LOEO --aggregate_only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from direct_torque_utils import DIRECT_TORQUE_NAMES  # noqa: E402
from experiment_groups import (  # noqa: E402
    DEFAULT_ALWAYS_EXCLUDED_EXPERIMENTS,
    detect_layout,
    list_experiment_dirs,
)

TRAIN_SCRIPT = SCRIPT_DIR / "train_directTorque.py"
INFER_SCRIPT = SCRIPT_DIR / "infer_directTorque.py"

# Training flags forwarded verbatim from the wrapper to train_directTorque.py.
FORWARDED_TRAIN_FLAGS = (
    "epochs",
    "batch_size",
    "d_model",
    "num_layers",
    "num_heads",
    "ff_dim",
    "window_size",
    "stride",
    "prediction_margin_frames",
    "learning_rate",
    "dropout_rate",
    "weight_decay",
    "normalizer_max_batches",
    "val_fraction",
    "seed",
    "robust_loss",
    "huber_delta",
    "scan_workers",
    "allow_missing_noised",
    "edge_mode",
    "edge_trim_frames",
)

POOLING_KEYS = (
    "n",
    "sum_abs_err",
    "sum_sq_err",
    "sum_err",
    "sum_pred",
    "sum_gt",
    "sum_pred_sq",
    "sum_gt_sq",
    "sum_pred_gt",
)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _run(cmd: Sequence[str], *, log_path: Path, label: str) -> None:
    """Stream a subprocess to stdout and to ``log_path`` at the same time."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {label}\n# {' '.join(str(c) for c in cmd)}\n\n")
        log.flush()
        process = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        returncode = process.wait()
    elapsed = time.time() - started
    if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode} after {elapsed / 60:.1f} min. Log: {log_path}")
    print(f"   ✓ {label} finished in {elapsed / 60:.1f} min", flush=True)


def _train_command(
    args: argparse.Namespace,
    data_dir: Path,
    held_out: str,
    run_dir: Path,
    always_excluded: Sequence[str],
) -> List[str]:
    # The round's hold-out plus any experiment barred from every training set.
    # ``dict.fromkeys`` keeps the order stable and drops the duplicate when the
    # held-out experiment is itself always-excluded.
    excluded = list(dict.fromkeys([held_out, *always_excluded]))
    cmd: List[str] = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data_dir", str(data_dir),
        "--output_dir", str(run_dir),
        "--layout", "experiment",
        "--exclude_experiments", ",".join(excluded),
        "--exp_name", f"loeo_holdout_{held_out}",
    ]
    for flag in FORWARDED_TRAIN_FLAGS:
        value = getattr(args, flag, None)
        if value is not None:
            cmd += [f"--{flag}", str(value)]
    if args.no_train_plots:
        cmd.append("--no_plots")
    if args.quiet_steps:
        cmd.append("--quiet_steps")
    cmd += list(args.train_arg or [])
    return cmd


def _infer_command(args: argparse.Namespace, data_dir: Path, held_out: str, checkpoint: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(INFER_SCRIPT),
        "--checkpoint", str(checkpoint),
        "--data_dir", str(data_dir),
        "--layout", "experiment",
        "--experiment", held_out,
        "--write_to_trial_dir",
        "--results_dir_name", str(args.results_dir_name),
        "--output", str(_resolve(args.output_root) / f"hold_out_{held_out}" / "inference"),
        "--batch_size", str(args.infer_batch_size),
        "--scan_workers", str(args.scan_workers),
    ]
    if args.results_subdir:
        cmd += ["--results_subdir", str(args.results_subdir)]
    if args.no_inference_plots:
        cmd.append("--no_plots")
    return cmd


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _checkpoint_matches_dataset_schema(checkpoint: Path, data_dir: Path) -> tuple[bool, str]:
    """Reject checkpoints whose recorded pos/vel blocks predate the dataset schema.

    The knee-channel migration changes the default direct-torque input from 370
    to 374 features (pos 13->15 after pelvis-Euler removal; vel 19->21). Without
    this guard, an interrupted/restarted LOEO run could silently reuse a
    knee-free checkpoint and then fail only after inference begins.
    """
    schema_path = data_dir / "Kinematic_Input_Schema.json"
    hparams_path = checkpoint.parent / "hyperparameters.json"
    if not schema_path.exists():
        return True, "dataset has no explicit Kinematic_Input_Schema.json"
    if not hparams_path.exists():
        return False, f"missing {hparams_path.name}"

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        hparams = json.loads(hparams_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"could not read schema metadata ({exc})"

    pos_columns = schema.get("position_input_columns")
    vel_columns = schema.get("velocity_input_columns")
    if not isinstance(pos_columns, list) or not isinstance(vel_columns, list):
        return False, "dataset schema does not declare position/velocity columns"

    include_pelvis = _as_bool(
        hparams.get("include_pelvis_euler", hparams.get("includePelvisEuler", False))
    )
    expected_pos_dim = len(pos_columns) if include_pelvis else len(pos_columns) - 3
    expected_vel_dim = len(vel_columns)

    layout = hparams.get("input_feature_layout") or {}
    blocks = layout.get("blocks") or hparams.get("input_feature_blocks") or []
    block_dims = {
        str(block.get("name")): int(block.get("dim"))
        for block in blocks
        if isinstance(block, Mapping) and block.get("name") is not None and block.get("dim") is not None
    }
    if "pos" not in block_dims or "vel" not in block_dims:
        return False, "checkpoint does not record traceable pos/vel feature blocks"

    if block_dims["pos"] != expected_pos_dim or block_dims["vel"] != expected_vel_dim:
        return (
            False,
            "feature blocks are incompatible "
            f"(checkpoint pos/vel={block_dims['pos']}/{block_dims['vel']}, "
            f"dataset expects {expected_pos_dim}/{expected_vel_dim})",
        )

    recorded_total = hparams.get("input_dim")
    block_total = sum(block_dims.values())
    if recorded_total is not None and int(recorded_total) != block_total:
        return (
            False,
            f"checkpoint input_dim={recorded_total} but recorded blocks sum to {block_total}",
        )
    return True, f"pos/vel blocks match ({expected_pos_dim}/{expected_vel_dim})"


def run_sweep(
    args: argparse.Namespace,
    data_dir: Path,
    experiments: Sequence[str],
    output_root: Path,
    always_excluded: Sequence[str],
) -> None:
    for index, held_out in enumerate(experiments, start=1):
        run_dir = output_root / f"hold_out_{held_out}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "best_model.pkl"
        infer_summary = run_dir / "inference" / "summary_metrics.json"

        trained_on = [e for e in experiments if e != held_out and e not in set(always_excluded)]
        print("\n" + "=" * 78, flush=True)
        print(f"[{index}/{len(experiments)}] Hold-out experiment: {held_out}", flush=True)
        print(f"   trains on: {trained_on}", flush=True)
        if always_excluded:
            print(f"   never trained on: {list(always_excluded)}", flush=True)
        print("=" * 78, flush=True)

        trained_now = False
        checkpoint_compatible = False
        checkpoint_reason = "checkpoint does not exist"
        if checkpoint.exists() and not args.force:
            checkpoint_compatible, checkpoint_reason = _checkpoint_matches_dataset_schema(
                checkpoint, data_dir
            )

        if checkpoint.exists() and not args.force and checkpoint_compatible:
            print(
                f"   ↩︎  Reusing existing checkpoint {checkpoint} "
                f"({checkpoint_reason}; pass --force to retrain).",
                flush=True,
            )
        else:
            if checkpoint.exists() and not args.force and not checkpoint_compatible:
                print(
                    f"   ♻️  Existing checkpoint is incompatible with the current dataset: "
                    f"{checkpoint_reason}. Retraining this fold.",
                    flush=True,
                )
            _run(
                _train_command(args, data_dir, held_out, run_dir, always_excluded),
                log_path=run_dir / "train.log",
                label=f"train (hold out {held_out})",
            )
            trained_now = True
            if not checkpoint.exists():
                raise RuntimeError(f"Training finished but {checkpoint} was not written.")

        if infer_summary.exists() and not trained_now and not args.force and not args.force_inference:
            print(f"   ↩︎  Reusing existing inference results {infer_summary}.", flush=True)
            continue
        _run(
            _infer_command(args, data_dir, held_out, checkpoint),
            log_path=run_dir / "inference.log",
            label=f"inference (hold out {held_out})",
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _collect_trial_metrics(output_root: Path, experiments: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for held_out in experiments:
        summary_path = output_root / f"hold_out_{held_out}" / "inference" / "summary_metrics.json"
        if not summary_path.exists():
            print(f"   ⚠️  Missing inference summary for {held_out}: {summary_path}", flush=True)
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        for trial in summary.get("trials", []):
            row = dict(trial)
            # ``experiment`` comes from discovery; fall back to the hold-out name
            # so a summary written before the key existed still aggregates.
            row["experiment"] = str(row.get("experiment") or held_out)
            row["held_out_experiment"] = held_out
            rows.append(row)
    return rows


def _empty_pool(n_channels: int) -> Dict[str, List[float]]:
    return {key: [0.0] * n_channels for key in POOLING_KEYS}


def _accumulate(pool: Dict[str, List[float]], trial: Mapping[str, Any]) -> None:
    trial_pool = trial.get("pooling") or {}
    for key in POOLING_KEYS:
        values = trial_pool.get(key) or []
        for idx, value in enumerate(values):
            if idx < len(pool[key]) and math.isfinite(float(value)):
                pool[key][idx] += float(value)


def _pool_to_metrics(pool: Mapping[str, Sequence[float]], channels: Sequence[str]) -> Dict[str, Any]:
    """Exact frame-weighted (micro) metrics from accumulated sufficient statistics."""
    per_channel: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(channels):
        n = float(pool["n"][idx])
        if n <= 0:
            per_channel[name] = {"n_frames": 0, "mae_bwh": float("nan"), "rmse_bwh": float("nan"),
                                 "bias_bwh": float("nan"), "r": float("nan")}
            continue
        mean_pred = pool["sum_pred"][idx] / n
        mean_gt = pool["sum_gt"][idx] / n
        cov = pool["sum_pred_gt"][idx] / n - mean_pred * mean_gt
        var_pred = max(pool["sum_pred_sq"][idx] / n - mean_pred**2, 0.0)
        var_gt = max(pool["sum_gt_sq"][idx] / n - mean_gt**2, 0.0)
        denom = math.sqrt(var_pred * var_gt)
        per_channel[name] = {
            "n_frames": int(n),
            "mae_bwh": pool["sum_abs_err"][idx] / n,
            "rmse_bwh": math.sqrt(pool["sum_sq_err"][idx] / n),
            "bias_bwh": pool["sum_err"][idx] / n,
            "r": (cov / denom) if denom > 0 else float("nan"),
        }

    total_n = sum(float(v) for v in pool["n"])
    overall_mae = sum(pool["sum_abs_err"]) / total_n if total_n > 0 else float("nan")
    overall_rmse = math.sqrt(sum(pool["sum_sq_err"]) / total_n) if total_n > 0 else float("nan")
    finite_r = [c["r"] for c in per_channel.values() if math.isfinite(c["r"])]
    return {
        "mae_bwh": overall_mae,
        "rmse_bwh": overall_rmse,
        "mean_channel_r": (sum(finite_r) / len(finite_r)) if finite_r else float("nan"),
        "n_channel_frames": int(total_n),
        "per_channel": per_channel,
    }


def _mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else float("nan")


def _macro_metrics(trials: Sequence[Mapping[str, Any]], channels: Sequence[str]) -> Dict[str, Any]:
    """Unweighted mean across trials — every trial counts the same."""
    return {
        "n_trials": len(trials),
        "mae_bwh": _mean(t.get("mae_bwh") for t in trials),
        "rmse_bwh": _mean(t.get("rmse_bwh") for t in trials),
        "mae_bwh_left": _mean(t.get("mae_bwh_left") for t in trials),
        "per_channel_mae_bwh": {
            name: _mean((t.get("per_channel") or {}).get(name, {}).get("mae_bwh") for t in trials)
            for name in channels
        },
    }


def _subject_of(trial: Mapping[str, Any]) -> str:
    subject = str(trial.get("subject") or "")
    if subject:
        return subject
    return str(trial.get("trial", "")).split("/")[0]


def aggregate(output_root: Path, experiments: Sequence[str], *, data_dir: Path) -> Dict[str, Any]:
    channels = list(DIRECT_TORQUE_NAMES)
    trials = _collect_trial_metrics(output_root, experiments)
    if not trials:
        raise SystemExit(f"No inference summaries found under {output_root}. Run the sweep first.")

    accuracy_dir = output_root / "accuracy"
    accuracy_dir.mkdir(parents=True, exist_ok=True)

    # --- per-experiment -----------------------------------------------------
    per_experiment: Dict[str, Any] = {}
    overall_pool = _empty_pool(len(channels))
    for experiment in sorted({t["experiment"] for t in trials}):
        subset = [t for t in trials if t["experiment"] == experiment]
        pool = _empty_pool(len(channels))
        for trial in subset:
            _accumulate(pool, trial)
            _accumulate(overall_pool, trial)
        per_experiment[experiment] = {
            "n_trials": len(subset),
            "n_subjects": len({_subject_of(t) for t in subset}),
            "micro": _pool_to_metrics(pool, channels),
            "macro": _macro_metrics(subset, channels),
        }

    # --- overall ------------------------------------------------------------
    experiment_macro_mae = [v["macro"]["mae_bwh"] for v in per_experiment.values()]
    experiment_macro_rmse = [v["macro"]["rmse_bwh"] for v in per_experiment.values()]
    overall = {
        "n_experiments": len(per_experiment),
        "n_trials": len(trials),
        "n_subjects": len({_subject_of(t) for t in trials}),
        # Frame-weighted across every held-out frame in the dataset.
        "micro": _pool_to_metrics(overall_pool, channels),
        # Every trial weighted equally.
        "macro_over_trials": _macro_metrics(trials, channels),
        # Every experiment weighted equally, so the 105-subject Stroke group does
        # not drown out the 16-subject Numeric group.
        "macro_over_experiments": {
            "mae_bwh": _mean(experiment_macro_mae),
            "rmse_bwh": _mean(experiment_macro_rmse),
        },
    }

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "channels": channels,
        "units": "percent bodyweight * height (%BW*H)",
        "protocol": "leave-one-experiment-out; each experiment scored by the model that never saw it",
        "per_experiment": per_experiment,
        "overall": overall,
    }
    with (accuracy_dir / "loeo_accuracy.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_trial_csv(accuracy_dir / "per_trial_metrics.csv", trials, channels)
    _write_subject_csv(accuracy_dir / "per_subject_metrics.csv", trials, channels)
    _write_experiment_csv(accuracy_dir / "per_experiment_metrics.csv", per_experiment, channels)
    _print_report(report)
    print(f"\nWrote accuracy report to {accuracy_dir}", flush=True)
    return report


def _write_trial_csv(path: Path, trials: Sequence[Mapping[str, Any]], channels: Sequence[str]) -> None:
    fields = ["experiment", "subject", "trial", "n_eval_frames", "mae_bwh", "rmse_bwh", "mae_bwh_left", "rmse_bwh_left"]
    fields += [f"mae_{name}" for name in channels] + [f"r_{name}" for name in channels]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trial in sorted(trials, key=lambda t: (t.get("experiment", ""), _subject_of(t), t.get("trial", ""))):
            per_channel = trial.get("per_channel") or {}
            row = {key: trial.get(key) for key in fields if key in trial}
            row["subject"] = _subject_of(trial)
            for name in channels:
                row[f"mae_{name}"] = per_channel.get(name, {}).get("mae_bwh")
                row[f"r_{name}"] = per_channel.get(name, {}).get("r")
            writer.writerow(row)


def _write_subject_csv(path: Path, trials: Sequence[Mapping[str, Any]], channels: Sequence[str]) -> None:
    by_subject: Dict[tuple, List[Mapping[str, Any]]] = {}
    for trial in trials:
        by_subject.setdefault((trial.get("experiment", ""), _subject_of(trial)), []).append(trial)

    fields = ["experiment", "subject", "n_trials", "n_eval_frames", "mae_bwh", "rmse_bwh", "mean_channel_r"]
    fields += [f"mae_{name}" for name in channels]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (experiment, subject), subset in sorted(by_subject.items()):
            pool = _empty_pool(len(channels))
            for trial in subset:
                _accumulate(pool, trial)
            metrics = _pool_to_metrics(pool, channels)
            row = {
                "experiment": experiment,
                "subject": subject,
                "n_trials": len(subset),
                "n_eval_frames": sum(int(t.get("n_eval_frames", 0)) for t in subset),
                "mae_bwh": metrics["mae_bwh"],
                "rmse_bwh": metrics["rmse_bwh"],
                "mean_channel_r": metrics["mean_channel_r"],
            }
            for name in channels:
                row[f"mae_{name}"] = metrics["per_channel"][name]["mae_bwh"]
            writer.writerow(row)


def _write_experiment_csv(path: Path, per_experiment: Mapping[str, Any], channels: Sequence[str]) -> None:
    fields = [
        "experiment", "n_subjects", "n_trials",
        "mae_bwh_micro", "rmse_bwh_micro", "mean_channel_r",
        "mae_bwh_macro", "rmse_bwh_macro", "mae_bwh_left_macro",
    ]
    fields += [f"mae_{name}" for name in channels] + [f"r_{name}" for name in channels]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for experiment in sorted(per_experiment):
            entry = per_experiment[experiment]
            micro, macro = entry["micro"], entry["macro"]
            row = {
                "experiment": experiment,
                "n_subjects": entry["n_subjects"],
                "n_trials": entry["n_trials"],
                "mae_bwh_micro": micro["mae_bwh"],
                "rmse_bwh_micro": micro["rmse_bwh"],
                "mean_channel_r": micro["mean_channel_r"],
                "mae_bwh_macro": macro["mae_bwh"],
                "rmse_bwh_macro": macro["rmse_bwh"],
                "mae_bwh_left_macro": macro["mae_bwh_left"],
            }
            for name in channels:
                row[f"mae_{name}"] = micro["per_channel"][name]["mae_bwh"]
                row[f"r_{name}"] = micro["per_channel"][name]["r"]
            writer.writerow(row)


def _print_report(report: Mapping[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("Leave-one-experiment-out accuracy  (%BW*H, all 14 torque channels)")
    print("=" * 78)
    header = f"{'Experiment':<16}{'Subj':>6}{'Trials':>8}{'MAE':>10}{'RMSE':>10}{'mean r':>9}{'MAE/trial':>12}"
    print(header)
    print("-" * len(header))
    for experiment in sorted(report["per_experiment"]):
        entry = report["per_experiment"][experiment]
        micro, macro = entry["micro"], entry["macro"]
        print(
            f"{experiment:<16}{entry['n_subjects']:>6}{entry['n_trials']:>8}"
            f"{micro['mae_bwh']:>10.4f}{micro['rmse_bwh']:>10.4f}"
            f"{micro['mean_channel_r']:>9.3f}{macro['mae_bwh']:>12.4f}"
        )
    overall = report["overall"]
    print("-" * len(header))
    print(
        f"{'OVERALL':<16}{overall['n_subjects']:>6}{overall['n_trials']:>8}"
        f"{overall['micro']['mae_bwh']:>10.4f}{overall['micro']['rmse_bwh']:>10.4f}"
        f"{overall['micro']['mean_channel_r']:>9.3f}"
        f"{overall['macro_over_trials']['mae_bwh']:>12.4f}"
    )
    print(
        f"   frame-weighted MAE={overall['micro']['mae_bwh']:.4f} | "
        f"trial-mean MAE={overall['macro_over_trials']['mae_bwh']:.4f} | "
        f"experiment-mean MAE={overall['macro_over_experiments']['mae_bwh']:.4f}"
    )

    print("\nPer-channel (frame-weighted over the whole dataset):")
    print(f"{'Channel':<20}{'MAE':>10}{'RMSE':>10}{'bias':>10}{'r':>8}")
    print("-" * 58)
    for name, stats in overall["micro"]["per_channel"].items():
        print(f"{name:<20}{stats['mae_bwh']:>10.4f}{stats['rmse_bwh']:>10.4f}"
              f"{stats['bias_bwh']:>10.4f}{stats['r']:>8.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Nested dataset root (Dataset/<Experiment>/<Subject>/Trial_#).")
    p.add_argument("--output_root", default="outputs/DirectTorque_LOEO", help="Where per-round models and logs go.")
    p.add_argument("--experiments", default="", help="Comma-separated subset of experiments to run (default: all).")
    p.add_argument("--skip_experiments", default="", help="Comma-separated experiments to leave out of the sweep.")
    p.add_argument(
        "--always_exclude_experiments",
        default=",".join(DEFAULT_ALWAYS_EXCLUDED_EXPERIMENTS),
        help=(
            "Comma-separated experiments barred from EVERY model's training set. They are still "
            "evaluated: each keeps its own hold-out round. Names not present in the dataset are "
            "warned about, not an error. Pass an empty string to disable. "
            f"Default: {','.join(DEFAULT_ALWAYS_EXCLUDED_EXPERIMENTS)}"
        ),
    )
    p.add_argument("--aggregate_only", action="store_true", help="Skip training/inference; recompute metrics only.")
    p.add_argument("--force", action="store_true", help="Retrain and re-infer even when outputs already exist.")
    p.add_argument("--force_inference", action="store_true", help="Reuse checkpoints but re-run inference.")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--ff_dim", type=int, default=None)
    p.add_argument("--window_size", type=int, default=None)
    p.add_argument("--edge_mode", choices=["legacy", "train", "infer"], default=None,
                   help="Edge-frame policy forwarded to training. 'train' trims edge_trim_frames "
                        "off each trial end before windowing and supervises every frame; inference "
                        "then predicts the edges but leaves them out of the metrics.")
    p.add_argument("--edge_trim_frames", type=int, default=None,
                   help="Frames trimmed per trial end (edge_mode=train) and excluded from metrics.")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--prediction_margin_frames", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--dropout_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--normalizer_max_batches", type=int, default=None)
    p.add_argument("--val_fraction", type=float, default=None,
                   help="Fraction of the training subjects held out for model selection.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--robust_loss", choices=["mse", "huber"], default=None)
    p.add_argument("--huber_delta", type=float, default=None)
    p.add_argument("--scan_workers", type=int, default=3)
    p.add_argument(
        "--allow_missing_noised",
        default="True",
        help=(
            "Trials with no _noised.npy bundle at all (e.g. the OpenCapVal cohort) use their "
            "clean files instead of being silently skipped. A partial bundle still fails strictly."
        ),
    )
    p.add_argument("--train_arg", action="append", default=[],
                   help="Extra raw flag passed through to train_directTorque.py (repeatable).")

    p.add_argument("--infer_batch_size", type=int, default=128)
    p.add_argument("--results_dir_name", default="inference_results",
                   help="Folder created under each held-out trial for its predictions.")
    p.add_argument("--results_subdir", default="", help="Optional run-name level under the results folder.")
    p.add_argument("--no_inference_plots", action="store_true")
    p.add_argument("--no_train_plots", action="store_true")
    p.add_argument("--quiet_steps", action="store_true")
    return p.parse_args()


def _select_experiments(data_dir: Path, args: argparse.Namespace) -> List[str]:
    layout = detect_layout(data_dir)
    if layout != "experiment":
        raise SystemExit(
            f"{data_dir} is not in the nested Experiment/Subject/Trial layout.\n"
            "Run: python scripts/reorganize_dataset_by_experiment.py "
            f"--data_dir {data_dir} --apply"
        )
    available = [p.name for p in list_experiment_dirs(data_dir)]
    if not available:
        raise SystemExit(f"No experiment folders found under {data_dir}.")

    requested = [e.strip() for e in str(args.experiments).split(",") if e.strip()] or list(available)
    skipped = {e.strip() for e in str(args.skip_experiments).split(",") if e.strip()}
    unknown = sorted((set(requested) | skipped) - set(available))
    if unknown:
        raise SystemExit(f"Unknown experiment(s) {unknown}. Available: {available}")
    return [e for e in requested if e not in skipped]


def _resolve_always_excluded(data_dir: Path, args: argparse.Namespace) -> List[str]:
    """Experiments barred from every training set, narrowed to those actually present.

    Unlike ``--experiments``, an unknown name here is a warning rather than an error:
    the default (``Hip_OA``) names a cohort that lives in the standalone ``Hip_OA/``
    export and is not part of the trusted datasets yet, so it must be harmless to
    carry until it is merged in.
    """
    requested = [e.strip() for e in str(args.always_exclude_experiments).split(",") if e.strip()]
    if not requested:
        return []
    available = {p.name for p in list_experiment_dirs(data_dir)}
    present = [e for e in requested if e in available]
    absent = [e for e in requested if e not in available]
    if absent:
        print(
            f"   ℹ️  always-excluded experiment(s) not present in this dataset, nothing to exclude: {absent}",
            flush=True,
        )
    return present


def main() -> None:
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiments = _select_experiments(data_dir, args)
    always_excluded = _resolve_always_excluded(data_dir, args)
    print(f"Dataset:      {data_dir}")
    print(f"Output root:  {output_root}")
    print(f"Experiments:  {experiments}")
    if always_excluded:
        print(f"Never trained on (still evaluated): {always_excluded}")

    if not args.aggregate_only:
        with (output_root / "sweep_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"started": datetime.now().isoformat(timespec="seconds"),
                 "data_dir": str(data_dir), "experiments": experiments,
                 "always_excluded_requested": [
                     e.strip() for e in str(args.always_exclude_experiments).split(",") if e.strip()
                 ],
                 "always_excluded_applied": always_excluded,
                 "args": vars(args)},
                f, indent=2, default=str,
            )
        run_sweep(args, data_dir, experiments, output_root, always_excluded)

    aggregate(output_root, experiments, data_dir=data_dir)


if __name__ == "__main__":
    main()
