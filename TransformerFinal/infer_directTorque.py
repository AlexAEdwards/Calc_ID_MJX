"""Inference for direct joint-torque checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from data_loader import TrialDataLoader  # noqa: E402
from direct_torque_utils import (  # noqa: E402
    DIRECT_TORQUE_NAMES,
    MODEL_STRUCTURE,
    build_direct_torque_targets,
    direct_torque_percent_to_nm,
    is_direct_torque_hparams,
)
from train import discover_all_trials  # noqa: E402
from train_directTorque import KinematicsToDirectTorque, normalize_direct_batch  # noqa: E402


def _left_leg_direct_torque_indices(output_dim: int) -> List[int]:
    indices = [
        idx
        for idx, name in enumerate(list(DIRECT_TORQUE_NAMES)[: int(output_dim)])
        if str(name).endswith("_l")
    ]
    if indices:
        return indices
    midpoint = int(output_dim) // 2
    return list(range(midpoint, int(output_dim)))


def _per_channel_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """Per-channel accuracy plus the sufficient statistics needed to pool trials.

    ``pooling`` carries per-channel sums so an aggregator can recover exact
    frame-weighted MAE/RMSE/bias/Pearson-r over any set of trials without
    re-reading the waveforms.
    """
    names = list(DIRECT_TORQUE_NAMES)[: pred.shape[-1]]
    per_channel: Dict[str, Dict[str, float]] = {}
    pooling: Dict[str, List[float]] = {
        key: [] for key in ("n", "sum_abs_err", "sum_sq_err", "sum_err", "sum_pred", "sum_gt", "sum_pred_sq", "sum_gt_sq", "sum_pred_gt")
    }

    for idx, name in enumerate(names):
        p = np.asarray(pred[:, idx], dtype=np.float64)
        g = np.asarray(target[:, idx], dtype=np.float64)
        ok = np.isfinite(p) & np.isfinite(g)
        p, g = p[ok], g[ok]
        n = int(p.size)
        err = p - g

        pooling["n"].append(float(n))
        pooling["sum_abs_err"].append(float(np.sum(np.abs(err))))
        pooling["sum_sq_err"].append(float(np.sum(np.square(err))))
        pooling["sum_err"].append(float(np.sum(err)))
        pooling["sum_pred"].append(float(np.sum(p)))
        pooling["sum_gt"].append(float(np.sum(g)))
        pooling["sum_pred_sq"].append(float(np.sum(np.square(p))))
        pooling["sum_gt_sq"].append(float(np.sum(np.square(g))))
        pooling["sum_pred_gt"].append(float(np.sum(p * g)))

        if n == 0:
            per_channel[name] = {"n_frames": 0, "mae_bwh": float("nan"), "rmse_bwh": float("nan"),
                                 "bias_bwh": float("nan"), "r": float("nan")}
            continue
        denom = float(np.std(p) * np.std(g))
        r = float(np.mean((p - p.mean()) * (g - g.mean())) / denom) if denom > 0 else float("nan")
        per_channel[name] = {
            "n_frames": n,
            "mae_bwh": float(np.mean(np.abs(err))),
            "rmse_bwh": float(np.sqrt(np.mean(np.square(err)))),
            "bias_bwh": float(np.mean(err)),
            "r": r,
        }
    return {"per_channel": per_channel, "pooling": pooling, "channels": names}


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _make_loader(trials: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], batch_size: int) -> TrialDataLoader:
    return TrialDataLoader(
        list(trials),
        window_size=int(cfg["window_size"]),
        stride=int(cfg.get("inference_stride", cfg.get("stride", 16))),
        batch_size=int(batch_size),
        shuffle=False,
        trim_cop=bool(cfg.get("trim_cop", False)),
        deviation_learning=False,
        use_noised=bool(cfg.get("use_noised", cfg.get("UseNoised", False))),
        noised_gt=bool(cfg.get("noised_gt", cfg.get("NoisedGT", False))),
        predict_jacobian=False,
        opencap_val=str(cfg.get("layout", "trusted")).lower() == "opencap",
        input_source=str(cfg.get("input_source", "processed")),
        include_pelvis_euler=bool(cfg.get("include_pelvis_euler", cfg.get("includePelvisEuler", False))),
        include_ankle_heights=bool(cfg.get("include_ankle_heights", True)),
        include_jacobian_input=bool(cfg.get("include_jacobian_input", cfg.get("includeJacobianInput", True))),
        include_auxiliary_denoising_inputs=bool(cfg.get("include_auxiliary_denoising_inputs", True)),
        prediction_margin_frames=int(cfg.get("prediction_margin_frames", 20)),
        use_grf_norm_cop=bool(cfg.get("use_grf_norm_cop", cfg.get("UseGRFNormCOP", False))),
        use_os_filtering=bool(cfg.get("use_os_filtering", False)),
        use_grf_nofilt=bool(cfg.get("use_grf_nofilt", True)),
        allow_missing_noised=bool(cfg.get("allow_missing_noised", False)),
        # Inference never trims: the full trial is windowed so edge frames DO get
        # predictions. They are excluded from accuracy via scoring_mask below, not
        # by withholding the prediction. A legacy checkpoint keeps legacy behaviour.
        edge_mode=("infer" if str(cfg.get("edge_mode", "legacy")) != "legacy" else "legacy"),
        edge_trim_frames=int(cfg.get("edge_trim_frames", 0)),
        drop_last=False,
    )


def _normalize_flag_keys(hparams: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(hparams)
    aliases = {
        "UseNoised": "use_noised",
        "NoisedGT": "noised_gt",
        "includePelvisEuler": "include_pelvis_euler",
        "includeJacobianInput": "include_jacobian_input",
        "UseGRFNormCOP": "use_grf_norm_cop",
    }
    for old, new in aliases.items():
        if new not in cfg and old in cfg:
            cfg[new] = cfg[old]
    return cfg


def _discover_requested_trials(data_dir: Path, cfg: Mapping[str, Any], args: argparse.Namespace) -> List[Mapping[str, Any]]:
    trials = discover_all_trials(
        str(data_dir),
        refresh_cache=True,
        scan_workers=int(args.scan_workers),
        layout=str(cfg.get("layout") or args.layout or "trusted"),
    )
    if args.experiment:
        wanted_experiments = {part.strip() for part in str(args.experiment).split(",") if part.strip()}
        available = sorted({str(t.get("experiment") or "") for t in trials} - {""})
        unknown = sorted(wanted_experiments - set(available))
        if unknown:
            raise ValueError(f"Unknown experiment(s) {unknown}. Available: {available}")
        trials = [t for t in trials if str(t.get("experiment") or "") in wanted_experiments]
    if args.subject:
        trials = [t for t in trials if str(t.get("subject")) == args.subject]
    if args.trial:
        wanted = args.trial.strip("/")
        trials = [
            t for t in trials
            if f"{t.get('subject')}/{t.get('trial_name')}" == wanted
            or str(t.get("trial_name")) == wanted
        ]
    if args.max_trials and args.max_trials > 0:
        trials = trials[: int(args.max_trials)]
    return trials


def _infer_one_trial(
    trial: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    hparams: Mapping[str, Any],
    output_dir: Path,
    *,
    batch_size: int,
    trial_output_dir: Path | None = None,
    checkpoint_label: str = "",
    make_plot: bool = True,
) -> Dict[str, Any]:
    normalizers = checkpoint["normalizers"]
    cfg = _normalize_flag_keys(hparams)
    loader = _make_loader([trial], cfg, batch_size=batch_size)
    sample = next(iter(loader))
    sample_norm = normalize_direct_batch(sample, normalizers)
    input_dim = int(sample_norm["input"].shape[-1])
    static_dim = int(sample_norm["static_context"].shape[-1])
    model = KinematicsToDirectTorque(
        input_dim=input_dim,
        static_dim=static_dim,
        output_dim=int(hparams.get("output_dim", len(DIRECT_TORQUE_NAMES))),
        d_model=int(hparams.get("d_model", 384)),
        num_heads=int(hparams.get("num_heads", 4)),
        num_layers=int(hparams.get("num_layers", 4)),
        ff_dim=int(hparams.get("ff_dim", 1536)),
        dropout_rate=float(hparams.get("dropout_rate", 0.158504)),
    )

    @jax.jit
    def _apply(params, x, static_context):
        return model.apply({"params": params}, x, static_context, train=False)

    # Recreate the loader after consuming the sample batch.
    loader = _make_loader([trial], cfg, batch_size=batch_size)
    pred_sum = None
    target_sum = None
    count = None
    static_context_raw = None
    trial_len = None
    all_metrics = []
    for raw_batch in loader:
        norm_batch = normalize_direct_batch(raw_batch, normalizers)
        pred_z = _apply(checkpoint["params"], norm_batch["input"], norm_batch["static_context"])
        pred_pct = np.asarray(normalizers["direct_torque"].unnormalize(pred_z))
        target_pct = np.asarray(norm_batch["direct_torque_target_raw"])
        mask = np.asarray(norm_batch["supervision_mask"])
        if mask.ndim == 2:
            mask = mask[..., None]
        finite = np.asarray(norm_batch["direct_torque_finite_mask"])
        mask = mask * finite

        starts = np.asarray(raw_batch["window_start_idx"], dtype=int)
        lengths = np.asarray(raw_batch["trial_length"], dtype=int)
        trial_len = int(lengths.max()) if trial_len is None else max(trial_len, int(lengths.max()))
        if pred_sum is None:
            pred_sum = np.zeros((trial_len, pred_pct.shape[-1]), dtype=np.float64)
            target_sum = np.zeros_like(pred_sum)
            count = np.zeros((trial_len, 1), dtype=np.float64)
            static_context_raw = np.asarray(norm_batch["static_context_raw"])[0]
        elif trial_len > pred_sum.shape[0]:
            grow = trial_len - pred_sum.shape[0]
            pred_sum = np.pad(pred_sum, ((0, grow), (0, 0)))
            target_sum = np.pad(target_sum, ((0, grow), (0, 0)))
            count = np.pad(count, ((0, grow), (0, 0)))

        for b, start in enumerate(starts):
            for i in range(pred_pct.shape[1]):
                idx = int(start) + i
                if idx < 0 or idx >= pred_sum.shape[0]:
                    continue
                if float(mask[b, i, 0]) <= 0.0:
                    continue
                pred_sum[idx] += pred_pct[b, i]
                target_sum[idx] += target_pct[b, i]
                count[idx, 0] += 1.0

        valid = mask > 0
        if np.any(valid):
            err = pred_pct - target_pct
            left_indices = _left_leg_direct_torque_indices(err.shape[-1])
            err_left = err[..., left_indices]
            # ``supervision_mask`` and ``direct_torque_finite_mask`` are
            # frame-level masks with a singleton channel axis.  Broadcast the
            # combined mask before selecting torque channels; indexing the
            # singleton axis directly fails for every channel after index 0.
            mask_left = np.broadcast_to(mask, err.shape)[..., left_indices]
            all_metrics.append(
                {
                    "mae_bwh": float(np.sum(np.abs(err_left) * mask_left) / max(np.sum(mask_left), 1.0)),
                    "rmse_bwh": float(np.sqrt(np.sum(np.square(err_left) * mask_left) / max(np.sum(mask_left), 1.0))),
                }
            )

    if pred_sum is None or count is None:
        raise RuntimeError(f"No windows produced for {trial}")
    valid_frames = count[:, 0] > 0
    pred_pct = np.full_like(pred_sum, np.nan, dtype=np.float64)
    target_pct = np.full_like(target_sum, np.nan, dtype=np.float64)
    pred_pct[valid_frames] = pred_sum[valid_frames] / count[valid_frames]
    target_pct[valid_frames] = target_sum[valid_frames] / count[valid_frames]
    pred_nm = np.asarray(direct_torque_percent_to_nm(pred_pct, static_context_raw, xp=np))
    target_nm = np.asarray(direct_torque_percent_to_nm(target_pct, static_context_raw, xp=np))

    # Two distinct masks. `coverage` is where a prediction exists at all; `scoring`
    # additionally drops the trial's first/last edge_trim frames. Under edge_mode
    # 'infer' coverage is the whole trial, so the edge frames keep their predictions
    # and are merely left out of the accuracy numbers.
    coverage = valid_frames
    edge_trim = int(cfg.get("edge_trim_frames", 0)) if str(cfg.get("edge_mode", "legacy")) != "legacy" else 0
    scoring = coverage.copy()
    if edge_trim > 0:
        n = scoring.shape[0]
        if 2 * edge_trim < n:
            scoring[:edge_trim] = False
            scoring[n - edge_trim:] = False
        else:
            # Trial too short to trim both ends; score what coverage allows and say so.
            scoring[:] = coverage

    pred_eval = pred_pct[scoring]
    target_eval = target_pct[scoring]
    err = pred_eval - target_eval
    channel_stats = _per_channel_metrics(pred_eval, target_eval)
    left_indices = _left_leg_direct_torque_indices(err.shape[-1])
    err_left = err[:, left_indices] if err.size and left_indices else err

    subject = str(trial.get("subject"))
    trial_label = str(trial.get("trial_name") or trial.get("trial") or "")
    metrics = {
        # ``trial`` keeps the historical "<subject>/<trial_name>" form; ``trial_name``
        # from discovery is already "<subject>/Trial_#" for the trusted layouts.
        "trial": trial_label if "/" in trial_label else f"{subject}/{trial_label}",
        "subject": subject,
        "experiment": str(trial.get("experiment") or ""),
        "n_eval_frames": int(np.sum(scoring)),
        "n_predicted_frames": int(np.sum(coverage)),
        "edge_trim_frames_excluded_from_metrics": edge_trim,
        "channels": channel_stats["channels"],
        # Headline numbers are over all 14 channels; the left-leg pair is kept
        # because earlier direct-torque reports were left-leg only.
        "mae_bwh": float(np.nanmean(np.abs(err))) if err.size else float("nan"),
        "rmse_bwh": float(np.sqrt(np.nanmean(np.square(err)))) if err.size else float("nan"),
        "mae_bwh_left": float(np.nanmean(np.abs(err_left))) if err_left.size else float("nan"),
        "rmse_bwh_left": float(np.sqrt(np.nanmean(np.square(err_left)))) if err_left.size else float("nan"),
        "per_channel": channel_stats["per_channel"],
        "per_channel_mae_bwh": {
            name: stats["mae_bwh"] for name, stats in channel_stats["per_channel"].items()
        },
        "pooling": channel_stats["pooling"],
    }
    if checkpoint_label:
        metrics["checkpoint"] = str(checkpoint_label)

    if trial_output_dir is not None:
        trial_dir = Path(trial_output_dir)
    else:
        trial_dir = output_dir / str(metrics["trial"]).replace("/", "__")
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics["results_dir"] = str(trial_dir)
    np.save(trial_dir / "direct_torque_pred_percent_bwh.npy", pred_pct.astype(np.float32))
    np.save(trial_dir / "direct_torque_gt_percent_bwh.npy", target_pct.astype(np.float32))
    np.save(trial_dir / "direct_torque_pred_nm.npy", pred_nm.astype(np.float32))
    np.save(trial_dir / "direct_torque_gt_nm.npy", target_nm.astype(np.float32))
    # prediction_coverage: a prediction exists here. scoring_mask: score this frame.
    # Under edge_mode 'infer' they differ only in the trial's first/last edge_trim
    # frames, which are predicted but deliberately not scored.
    np.save(trial_dir / "prediction_coverage.npy", coverage)
    np.save(trial_dir / "scoring_mask.npy", scoring)
    # Retained under its historical name for back-compat; equals scoring_mask, i.e.
    # the frames the metrics were computed on.
    np.save(trial_dir / "evaluation_mask.npy", scoring)
    with (trial_dir / "direct_torque_names.json").open("w", encoding="utf-8") as f:
        json.dump(list(DIRECT_TORQUE_NAMES), f, indent=2)
    with (trial_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if make_plot:
        fig, axes = plt.subplots(7, 2, figsize=(14, 18), sharex=True)
        axes = axes.reshape(-1)
        x = np.arange(pred_pct.shape[0])
        for idx, name in enumerate(DIRECT_TORQUE_NAMES):
            ax = axes[idx]
            ax.plot(x, target_pct[:, idx], label="GT", linewidth=1.2)
            ax.plot(x, pred_pct[:, idx], label="Pred", linewidth=1.0)
            ax.set_title(name)
            ax.set_ylabel("%BW*H")
            ax.grid(True, alpha=0.25)
        axes[0].legend()
        axes[-1].set_xlabel("Frame")
        fig.tight_layout()
        fig.savefig(trial_dir / "direct_torque_timeseries.png", dpi=150)
        plt.close(fig)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a direct-torque checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output", default="direct_torque_inference")
    p.add_argument("--subject", default=None)
    p.add_argument("--trial", default=None, help="Either Trial_# or Subject/Trial_#")
    p.add_argument("--experiment", default=None, help="Comma-separated experiment folder names (--layout experiment).")
    p.add_argument("--max_trials", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--layout",
        choices=["trusted", "experiment", "opencap"],
        default=None,
        help="Overrides the layout recorded in the checkpoint's hyperparameters.json.",
    )
    p.add_argument("--scan_workers", type=int, default=3)
    p.add_argument(
        "--write_to_trial_dir",
        action="store_true",
        help="Write each trial's results into <trial_root>/inference_results/ inside the dataset.",
    )
    p.add_argument(
        "--results_dir_name",
        default="inference_results",
        help="Folder name used under each trial root by --write_to_trial_dir.",
    )
    p.add_argument(
        "--results_subdir",
        default="",
        help="Optional extra level under the results folder, e.g. a run name.",
    )
    p.add_argument("--no_plots", action="store_true", help="Skip the per-trial timeseries PNG.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)
    hparams = _load_json(checkpoint_path.parent / "hyperparameters.json")
    if not is_direct_torque_hparams(hparams) and not is_direct_torque_hparams(checkpoint):
        raise ValueError(
            f"{checkpoint_path} does not declare model_structure='{MODEL_STRUCTURE}'. "
            "Use train_directTorque.py checkpoints with infer_directTorque.py."
        )
    hparams = {**hparams, **{k: v for k, v in checkpoint.items() if k in {"model_structure", "model_type", "output_dim"}}}
    if args.layout:
        hparams["layout"] = args.layout
    hparams.setdefault("layout", "trusted")
    data_dir = Path(args.data_dir or hparams.get("data_dir") or PROJECT_ROOT / "TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded")
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = _discover_requested_trials(data_dir, hparams, args)
    if not trials:
        raise ValueError("No trials matched the requested inference selection.")
    all_metrics = []
    failures = []
    for trial in tqdm(trials, desc="Direct torque inference"):
        trial_output_dir = None
        if args.write_to_trial_dir:
            trial_root = Path(str(trial.get("trial_root")))
            trial_output_dir = trial_root / str(args.results_dir_name)
            if args.results_subdir:
                trial_output_dir = trial_output_dir / str(args.results_subdir)
        try:
            all_metrics.append(
                _infer_one_trial(
                    trial,
                    checkpoint,
                    hparams,
                    output_dir,
                    batch_size=int(args.batch_size),
                    trial_output_dir=trial_output_dir,
                    checkpoint_label=str(checkpoint_path),
                    make_plot=not bool(args.no_plots),
                )
            )
        except Exception as exc:  # a single unusable trial must not sink the sweep
            label = f"{trial.get('subject')}/{trial.get('trial')}"
            failures.append({"trial": label, "error": f"{type(exc).__name__}: {exc}"})
            print(f"   ⚠️  Inference failed for {label}: {exc}", flush=True)

    if not all_metrics:
        raise RuntimeError(f"Inference failed for all {len(trials)} requested trials.")

    summary = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "experiments": sorted({m.get("experiment", "") for m in all_metrics} - {""}),
        "n_trials": len(all_metrics),
        "n_failed": len(failures),
        "failures": failures,
        "mean_mae_bwh": float(np.mean([m["mae_bwh"] for m in all_metrics])),
        "mean_rmse_bwh": float(np.mean([m["rmse_bwh"] for m in all_metrics])),
        "mean_mae_bwh_left": float(np.mean([m["mae_bwh_left"] for m in all_metrics])),
        "trials": all_metrics,
    }
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
