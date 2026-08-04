"""Train a direct joint-torque Transformer.

This is a torque-only sibling of ``train.py``.  It uses the same temporal input
layout and default architecture as ``train_single_model.py``, but its output is a
14-channel direct torque vector in normalized %BW*height units.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from data_loader import TrialDataLoader  # noqa: E402
from direct_torque_utils import (  # noqa: E402
    DIRECT_TORQUE_NAMES,
    DIRECT_TORQUE_OUTPUT_DIM,
    MODEL_STRUCTURE,
    build_direct_torque_targets,
    direct_torque_percent_to_nm,
    finite_direct_torque_mask,
)
from train import (  # noqa: E402
    Normalizer,
    SinusoidalPosEmb,
    TransformerBlock,
    create_train_state,
    discover_all_trials,
    infer_input_feature_layout_from_loader,
)
from paths import artifact, dataset  # noqa: E402


DEFAULT_CONFIG: Dict[str, Any] = {
    "data_dir": str(PROJECT_ROOT / "TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded"),
    "output_dir": str(artifact("outputs") / "DirectTorque_Default"),
    "d_model": 384,
    "num_layers": 4,
    "num_heads": 4,
    # window_size dropped 110 -> 70 alongside edge_mode='train': the trim is applied
    # before windowing, so a trial needs window_size + 2*edge_trim frames to survive.
    # At 70 that threshold is 110 frames, which keeps 74.9% of trials trainable
    # (80 would keep 66.7%, 110 only 45.6%).
    "window_size": 70,
    "stride": 16,
    # Only used by edge_mode='legacy' (the within-window centre crop).
    "prediction_margin_frames": 20,
    # 'train' trims edge_trim_frames off each trial end before windowing and
    # supervises every frame of every window; 'legacy' is the old centre-crop.
    "edge_mode": "train",
    "edge_trim_frames": 20,
    "learning_rate": 0.000191462,
    "dropout_rate": 0.158504,
    "ff_dim": 1536,
    "epochs": 40,
    "batch_size": 64,
    "weight_decay": 0.001,
    "trim_cop": False,
    "include_pelvis_euler": False,
    "include_jacobian_input": True,
    "include_ankle_heights": True,
    "include_auxiliary_denoising_inputs": True,
    "use_noised": True,
    "noised_gt": True,
    "use_grf_norm_cop": False,
    "use_os_filtering": False,
    "use_grf_nofilt": True,
    "robust_loss": "huber",
    "huber_delta": 1.0,
    "normalizer_max_batches": 100,
    "no_plots": False,
    "val_fraction": 0.2,
    "seed": 42,
    "scan_workers": 3,
    "layout": "trusted",
    "allow_missing_noised": False,
}


class KinematicsToDirectTorque(nn.Module):
    """Same backbone shape as train.py, with a plain linear torque head."""

    input_dim: int
    static_dim: int = 8
    output_dim: int = DIRECT_TORQUE_OUTPUT_DIM
    d_model: int = 384
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1536
    dropout_rate: float = 0.158504
    use_film: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.d_model)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = SinusoidalPosEmb(dim=self.d_model)(x)

        s = nn.Dense(self.d_model)(static_context)
        s = nn.gelu(s)
        s = nn.LayerNorm()(s)

        film_params = None
        if self.use_film:
            film_params = nn.Dense(self.num_layers * 2 * self.d_model, name="film_mlp")(s)
            film_params = film_params.reshape(s.shape[0], self.num_layers, 2, self.d_model)

        x = jnp.concatenate([jnp.expand_dims(s, axis=1), x], axis=1)
        for layer_idx in range(self.num_layers):
            film_gamma = film_params[:, layer_idx, 0, :] if film_params is not None else None
            film_beta = film_params[:, layer_idx, 1, :] if film_params is not None else None
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train, film_gamma=film_gamma, film_beta=film_beta)

        x = x[:, 1:, :]
        x = nn.LayerNorm()(x)
        return nn.Dense(self.output_dim)(x)


def _subject_key(trial: Mapping[str, Any]) -> str:
    return str(trial.get("subject") or Path(str(trial.get("path", ""))).parts[-2])


def _experiment_key(trial: Mapping[str, Any]) -> str:
    """Experiment folder a trial came from ("" for the flat layout)."""
    return str(trial.get("experiment") or "")


def _filter_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    exclude_prefixes: Iterable[str] = (),
    exclude_trials: Iterable[str] = (),
    exclude_experiments: Iterable[str] = (),
    include_experiments: Iterable[str] = (),
    max_subjects: int = 0,
) -> List[Mapping[str, Any]]:
    prefixes = tuple(str(p) for p in exclude_prefixes if str(p))
    exclude_tokens = tuple(str(token).strip().strip("/") for token in exclude_trials if str(token).strip())
    excluded_experiments = {str(e).strip() for e in exclude_experiments if str(e).strip()}
    included_experiments = {str(e).strip() for e in include_experiments if str(e).strip()}

    def _trial_label(trial: Mapping[str, Any]) -> str:
        subject = _subject_key(trial)
        trial_name = str(trial.get("trial_name") or Path(str(trial.get("path", ""))).name)
        return f"{subject}/{trial_name}".strip("/")

    filtered = []
    for trial in trials:
        subject = _subject_key(trial)
        label = _trial_label(trial)
        experiment = _experiment_key(trial)
        if excluded_experiments and experiment in excluded_experiments:
            continue
        if included_experiments and experiment not in included_experiments:
            continue
        if prefixes and subject.startswith(prefixes):
            continue
        if any(label == token or label.startswith(f"{token}/") or subject == token for token in exclude_tokens):
            continue
        filtered.append(trial)
    if max_subjects and max_subjects > 0:
        subjects = sorted({_subject_key(t) for t in filtered})[: int(max_subjects)]
        keep = set(subjects)
        filtered = [t for t in filtered if _subject_key(t) in keep]
    return filtered


def _split_by_subject(
    trials: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], List[str], List[str]]:
    subjects = sorted({_subject_key(t) for t in trials})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * float(val_fraction)))) if len(subjects) > 1 else 0
    val_subjects = set(subjects[:n_val])
    train_subjects = [s for s in subjects if s not in val_subjects]
    train_trials = [t for t in trials if _subject_key(t) in train_subjects]
    val_trials = [t for t in trials if _subject_key(t) in val_subjects]
    return train_trials, val_trials, train_subjects, sorted(val_subjects)


def _make_loader(trials: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], *, shuffle: bool) -> TrialDataLoader:
    return TrialDataLoader(
        list(trials),
        window_size=int(cfg["window_size"]),
        stride=int(cfg["stride"]),
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        trim_cop=bool(cfg["trim_cop"]),
        deviation_learning=False,
        use_noised=bool(cfg["use_noised"]),
        noised_gt=bool(cfg["noised_gt"]),
        predict_jacobian=False,
        opencap_val=str(cfg.get("layout", "trusted")).lower() == "opencap",
        input_source=str(cfg.get("input_source", "processed")),
        include_pelvis_euler=bool(cfg["include_pelvis_euler"]),
        include_ankle_heights=bool(cfg["include_ankle_heights"]),
        include_jacobian_input=bool(cfg["include_jacobian_input"]),
        include_auxiliary_denoising_inputs=bool(cfg["include_auxiliary_denoising_inputs"]),
        prediction_margin_frames=int(cfg["prediction_margin_frames"]),
        use_grf_norm_cop=bool(cfg["use_grf_norm_cop"]),
        use_os_filtering=bool(cfg["use_os_filtering"]),
        use_grf_nofilt=bool(cfg["use_grf_nofilt"]),
        allow_missing_noised=bool(cfg.get("allow_missing_noised", False)),
        # Validation shares the training edge policy so checkpoint selection measures
        # the same objective the model is optimising.
        edge_mode=str(cfg.get("edge_mode", "legacy")),
        edge_trim_frames=int(cfg.get("edge_trim_frames", 0)),
        drop_last=False,
    )


def _add_direct_targets_np(batch: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(batch)
    target = np.asarray(build_direct_torque_targets(batch, xp_name="numpy"), dtype=np.float32)
    out["direct_torque_target_raw"] = target
    out["direct_torque_finite_mask"] = np.asarray(finite_direct_torque_mask(target, xp=np), dtype=np.float32)
    return out


def compute_direct_normalizers(loader: TrialDataLoader, max_batches: int, *, quiet: bool = False) -> Dict[str, Normalizer]:
    if not quiet:
        print(f"   Sampling up to {max_batches} batches for direct-torque normalizers...", flush=True)
    input_samples, static_samples, torque_samples = [], [], []
    count = 0
    for batch in loader:
        batch = _add_direct_targets_np(batch)
        input_samples.append(np.asarray(batch["input"]))
        static_samples.append(np.asarray(batch["static_context"]))
        torque_samples.append(np.asarray(batch["direct_torque_target_raw"]))
        count += 1
        if count >= int(max_batches):
            break
    if not input_samples:
        raise ValueError("No batches were produced while computing direct-torque normalizers.")

    input_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in input_samples], axis=0)
    static_flat = np.concatenate(static_samples, axis=0)
    torque_flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in torque_samples], axis=0)
    finite = np.all(np.isfinite(torque_flat), axis=1)
    torque_flat = torque_flat[finite]
    if torque_flat.size == 0:
        raise ValueError("Direct-torque target normalizer saw no finite target rows.")
    def _build_normalizers() -> Dict[str, Normalizer]:
        return {
            "input": Normalizer(input_flat, eps=1e-8, name="input"),
            "static": Normalizer(static_flat, eps=1e-8, name="static"),
            "direct_torque": Normalizer(torque_flat, eps=1e-3, name="direct_torque"),
        }

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            normalizers = _build_normalizers()
    else:
        normalizers = _build_normalizers()
    print(
        f"   Direct normalizers ready from {input_flat.shape[0]} frames "
        f"({torque_flat.shape[0]} finite torque rows).",
        flush=True,
    )
    return normalizers


def normalize_direct_batch(batch: Mapping[str, Any], normalizers: Mapping[str, Normalizer]) -> Dict[str, Any]:
    raw = _add_direct_targets_np(batch)
    out = dict(raw)
    out["input"] = normalizers["input"].normalize(np.asarray(raw["input"]))
    out["static_context"] = normalizers["static"].normalize(np.asarray(raw["static_context"]))
    out["static_context_raw"] = np.asarray(raw["static_context"])
    out["direct_torque_target"] = normalizers["direct_torque"].normalize(
        np.asarray(raw["direct_torque_target_raw"])
    )
    return {k: jnp.asarray(v) if hasattr(v, "dtype") and np.asarray(v).dtype.kind in {"b", "i", "u", "f", "c"} else v for k, v in out.items()}


def _masked_loss(pred: Any, target: Any, mask: Any, robust_loss: str, huber_delta: float) -> Any:
    err = pred - target
    if robust_loss == "huber":
        abs_err = jnp.abs(err)
        quadratic = jnp.minimum(abs_err, huber_delta)
        per_elem = 0.5 * jnp.square(quadratic) + huber_delta * (abs_err - quadratic)
    else:
        per_elem = jnp.square(err)
    while mask.ndim < per_elem.ndim:
        mask = mask[..., None]
    denom = jnp.maximum(jnp.sum(mask) * pred.shape[-1], 1.0)
    return jnp.sum(per_elem * mask) / denom


def make_direct_train_step(normalizers: Mapping[str, Normalizer], robust_loss: str, huber_delta: float):
    @jax.jit
    def train_step(state: train_state.TrainState, batch: Mapping[str, Any], dropout_rng: Any):
        def loss_fn(params):
            pred = state.apply_fn(
                {"params": params},
                batch["input"],
                batch["static_context"],
                train=True,
                rngs={"dropout": dropout_rng},
            )
            mask = batch["supervision_mask"]
            if mask.ndim == 2:
                mask = mask[..., None]
            mask = mask * batch["direct_torque_finite_mask"]
            if "sample_weight" in batch:
                sample_weight = batch["sample_weight"].reshape((-1,) + (1,) * (mask.ndim - 1))
                mask = mask * sample_weight
            loss = _masked_loss(pred, batch["direct_torque_target"], mask, robust_loss, huber_delta)
            pred_pct = normalizers["direct_torque"].unnormalize(pred)
            tgt_pct = batch["direct_torque_target_raw"]
            abs_err = jnp.abs(pred_pct - tgt_pct)
            sq_err = jnp.square(pred_pct - tgt_pct)
            while mask.ndim < abs_err.ndim:
                mask = mask[..., None]
            denom = jnp.maximum(jnp.sum(mask) * pred.shape[-1], 1.0)
            metrics = {
                "loss": loss,
                "mae_bwh": jnp.sum(abs_err * mask) / denom,
                "rmse_bwh": jnp.sqrt(jnp.sum(sq_err * mask) / denom),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        metrics["grad_norm"] = optax.global_norm(grads)
        return state, metrics

    return train_step


def make_direct_eval_step(normalizers: Mapping[str, Normalizer], robust_loss: str, huber_delta: float):
    @jax.jit
    def eval_step(state: train_state.TrainState, batch: Mapping[str, Any]):
        pred = state.apply_fn({"params": state.params}, batch["input"], batch["static_context"], train=False)
        mask = batch["supervision_mask"]
        if mask.ndim == 2:
            mask = mask[..., None]
        mask = mask * batch["direct_torque_finite_mask"]
        loss = _masked_loss(pred, batch["direct_torque_target"], mask, robust_loss, huber_delta)
        pred_pct = normalizers["direct_torque"].unnormalize(pred)
        tgt_pct = batch["direct_torque_target_raw"]
        abs_err = jnp.abs(pred_pct - tgt_pct)
        sq_err = jnp.square(pred_pct - tgt_pct)
        while mask.ndim < abs_err.ndim:
            mask = mask[..., None]
        denom = jnp.maximum(jnp.sum(mask) * pred.shape[-1], 1.0)
        return {
            "loss": loss,
            "mae_bwh": jnp.sum(abs_err * mask) / denom,
            "rmse_bwh": jnp.sqrt(jnp.sum(sq_err * mask) / denom),
        }, pred

    return eval_step


def _mean_metrics(metric_list: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    keys = sorted({k for m in metric_list for k in m})
    return {k: float(np.mean([np.asarray(m[k]).item() for m in metric_list if k in m])) for k in keys}


def _save_checkpoint(path: Path, state: train_state.TrainState, normalizers: Mapping[str, Any], cfg: Mapping[str, Any], epoch: int, metrics: Mapping[str, float]) -> None:
    payload = {
        "params": state.params,
        "normalizers": dict(normalizers),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "model_structure": MODEL_STRUCTURE,
        "model_type": MODEL_STRUCTURE,
        "direct_torque_names": list(DIRECT_TORQUE_NAMES),
        "output_dim": DIRECT_TORQUE_OUTPUT_DIM,
        "normalization": "percent_bodyweight_height",
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _plot_direct_torque_summary(
    train_batch: Mapping[str, Any],
    train_pred: Any,
    val_batch: Mapping[str, Any],
    val_pred: Any,
    normalizers: Mapping[str, Normalizer],
    history: Sequence[Mapping[str, float]],
    output_dir: Path,
    epoch: int,
) -> None:
    """Write the compact summary plot updated after each epoch."""
    train_pred_pct = np.asarray(normalizers["direct_torque"].unnormalize(train_pred))[0]
    val_pred_pct = np.asarray(normalizers["direct_torque"].unnormalize(val_pred))[0]
    train_gt_pct = np.asarray(train_batch["direct_torque_target_raw"])[0]
    val_gt_pct = np.asarray(val_batch["direct_torque_target_raw"])[0]
    train_mask = np.asarray(train_batch["supervision_mask"])[0].astype(bool)
    val_mask = np.asarray(val_batch["supervision_mask"])[0].astype(bool)
    if train_mask.ndim > 1:
        train_mask = np.squeeze(train_mask, axis=-1)
    if val_mask.ndim > 1:
        val_mask = np.squeeze(val_mask, axis=-1)

    fig = plt.figure(figsize=(22, 18))
    grid = fig.add_gridspec(5, 4, height_ratios=[0.85, 1, 1, 1, 1], hspace=0.52, wspace=0.32)

    ax_loss = fig.add_subplot(grid[0, :2])
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row.get("train_loss", np.nan)) for row in history]
    val_loss = [float(row.get("val_loss", np.nan)) for row in history]
    ax_loss.plot(epochs, train_loss, label="Train loss", marker="o", linewidth=1.5)
    ax_loss.plot(epochs, val_loss, label="Val loss", marker="o", linewidth=1.5)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend()

    ax_mae = fig.add_subplot(grid[0, 2:])
    train_mae = [float(row.get("train_mae_bwh", np.nan)) for row in history]
    val_mae = [float(row.get("val_mae_bwh", np.nan)) for row in history]
    ax_mae.plot(epochs, train_mae, label="Train MAE", marker="o", linewidth=1.5)
    ax_mae.plot(epochs, val_mae, label="Val MAE", marker="o", linewidth=1.5)
    ax_mae.set_title("Direct Torque MAE (%BW*H)")
    ax_mae.set_xlabel("Epoch")
    ax_mae.grid(True, alpha=0.25)
    ax_mae.legend()

    for idx, name in enumerate(DIRECT_TORQUE_NAMES):
        row = 1 + idx // 4
        col = idx % 4
        ax = fig.add_subplot(grid[row, col])
        x_train = np.arange(train_gt_pct.shape[0])
        x_val = np.arange(val_gt_pct.shape[0])
        ax.plot(x_train[train_mask], train_gt_pct[train_mask, idx], color="#1f77b4", linewidth=1.2, label="Train GT")
        ax.plot(x_train[train_mask], train_pred_pct[train_mask, idx], color="#ff7f0e", linewidth=1.0, label="Train Pred")
        ax.plot(x_val[val_mask], val_gt_pct[val_mask, idx], color="#2ca02c", linewidth=1.2, alpha=0.8, label="Val GT")
        ax.plot(x_val[val_mask], val_pred_pct[val_mask, idx], color="#d62728", linewidth=1.0, alpha=0.85, label="Val Pred")
        ax.set_title(name)
        ax.set_ylabel("%BW*H")
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f"Direct Torque Summary - Epoch {epoch}", fontsize=16, fontweight="bold")
    fig.savefig(output_dir / "summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / f"summary_epoch_{int(epoch):04d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a direct 14-channel joint-torque model.")
    p.add_argument("--data_dir", default=DEFAULT_CONFIG["data_dir"])
    p.add_argument("--output_dir", default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--exp_name", default="direct_torque", help="Stored in hyperparameters for wrapper/run tracking.")
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--d_model", type=int, default=DEFAULT_CONFIG["d_model"])
    p.add_argument("--num_layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    p.add_argument("--num_heads", type=int, default=DEFAULT_CONFIG["num_heads"])
    p.add_argument("--ff_dim", type=int, default=DEFAULT_CONFIG["ff_dim"])
    p.add_argument("--window_size", type=int, default=DEFAULT_CONFIG["window_size"])
    p.add_argument("--stride", type=int, default=DEFAULT_CONFIG["stride"])
    p.add_argument("--prediction_margin_frames", type=int, default=DEFAULT_CONFIG["prediction_margin_frames"],
                   help="Within-window centre crop. Only used when --edge_mode legacy.")
    p.add_argument("--edge_mode", choices=list(("legacy", "train", "infer")),
                   default=DEFAULT_CONFIG["edge_mode"],
                   help="'train': trim edge_trim_frames off each trial end before windowing and "
                        "supervise every frame. 'legacy': old within-window centre crop.")
    p.add_argument("--edge_trim_frames", type=int, default=DEFAULT_CONFIG["edge_trim_frames"],
                   help="Frames trimmed from each end of the trial before windowing (edge_mode=train). "
                        "Trials with fewer than window_size frames remaining are dropped.")
    p.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--dropout_rate", type=float, default=DEFAULT_CONFIG["dropout_rate"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--normalizer_max_batches", type=int, default=DEFAULT_CONFIG["normalizer_max_batches"])
    p.add_argument("--val_fraction", type=float, default=DEFAULT_CONFIG["val_fraction"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--max_subjects", type=int, default=0)
    p.add_argument("--exclude_prefixes", default="")
    p.add_argument("--exclude_trials", default="")
    p.add_argument(
        "--exclude_experiments",
        default="",
        help="Comma-separated (or JSON list) experiment folder names to hold out. Requires --layout experiment.",
    )
    p.add_argument(
        "--include_experiments",
        default="",
        help="Comma-separated (or JSON list) experiment folder names to keep. Requires --layout experiment.",
    )
    p.add_argument("--save_model_epochs", default="")
    p.add_argument("--layout", choices=["trusted", "experiment", "opencap"], default=DEFAULT_CONFIG["layout"])
    p.add_argument("--input_source", default="processed")
    p.add_argument("--scan_workers", type=int, default=DEFAULT_CONFIG["scan_workers"])
    p.add_argument("--UseNoised", type=str, default=str(DEFAULT_CONFIG["use_noised"]))
    p.add_argument("--NoisedGT", type=str, default=str(DEFAULT_CONFIG["noised_gt"]))
    p.add_argument("--UseGRFNormCOP", type=str, default=str(DEFAULT_CONFIG["use_grf_norm_cop"]))
    p.add_argument("--use_grf_nofilt", type=str, default=str(DEFAULT_CONFIG["use_grf_nofilt"]))
    p.add_argument("--use_os_filtering", type=str, default=str(DEFAULT_CONFIG["use_os_filtering"]))
    p.add_argument("--includePelvisEuler", type=str, default=str(DEFAULT_CONFIG["include_pelvis_euler"]))
    p.add_argument("--includeJacobianInput", type=str, default=str(DEFAULT_CONFIG["include_jacobian_input"]))
    p.add_argument(
        "--allow_missing_noised",
        type=str,
        default=str(DEFAULT_CONFIG["allow_missing_noised"]),
        help=(
            "Let trials that carry no _noised.npy bundle at all fall back to their clean files "
            "instead of being skipped. A partial bundle still fails strictly."
        ),
    )
    p.add_argument("--robust_loss", choices=["mse", "huber"], default=DEFAULT_CONFIG["robust_loss"])
    p.add_argument("--huber_delta", type=float, default=DEFAULT_CONFIG["huber_delta"])
    p.add_argument("--no_lr_schedule", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--quiet_steps", action="store_true")
    return p.parse_args()


def _str_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_list_arg(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _parse_epoch_list_arg(value: Any) -> List[int]:
    epochs: List[int] = []
    for item in _parse_list_arg(value):
        try:
            epochs.append(int(item))
        except ValueError:
            continue
    return sorted(set(epoch for epoch in epochs if epoch > 0))


def main() -> None:
    args = parse_args()
    if bool(args.quiet_steps):
        os.environ["MJX_DATALOADER_QUIET"] = "1"
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(vars(args))
    cfg["use_noised"] = _str_bool(args.UseNoised)
    cfg["noised_gt"] = _str_bool(args.NoisedGT)
    cfg["use_grf_norm_cop"] = _str_bool(args.UseGRFNormCOP)
    cfg["use_grf_nofilt"] = _str_bool(args.use_grf_nofilt)
    cfg["use_os_filtering"] = _str_bool(args.use_os_filtering)
    cfg["include_pelvis_euler"] = _str_bool(args.includePelvisEuler)
    cfg["include_jacobian_input"] = _str_bool(args.includeJacobianInput)
    cfg["allow_missing_noised"] = _str_bool(args.allow_missing_noised)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    trials = discover_all_trials(
        str(data_dir),
        refresh_cache=True,
        scan_workers=int(args.scan_workers),
        layout=str(cfg.get("layout", "trusted")),
    )
    prefixes = _parse_list_arg(args.exclude_prefixes)
    excluded_trials = _parse_list_arg(args.exclude_trials)
    excluded_experiments = _parse_list_arg(args.exclude_experiments)
    included_experiments = _parse_list_arg(args.include_experiments)

    if excluded_experiments or included_experiments:
        available = sorted({_experiment_key(t) for t in trials} - {""})
        if not available:
            raise ValueError(
                "Experiment filtering was requested but no discovered trial carries an experiment. "
                "Reorganize the dataset (scripts/data_prep/reorganize_dataset_by_experiment.py) and pass --layout experiment."
            )
        unknown = sorted((set(excluded_experiments) | set(included_experiments)) - set(available))
        if unknown:
            raise ValueError(f"Unknown experiment(s) {unknown}. Available: {available}")

    before_filter = len(trials)
    trials = _filter_trials(
        trials,
        exclude_prefixes=prefixes,
        exclude_trials=excluded_trials,
        exclude_experiments=excluded_experiments,
        include_experiments=included_experiments,
        max_subjects=int(args.max_subjects),
    )
    if prefixes or excluded_trials or excluded_experiments or included_experiments:
        print(
            f"Excluded {before_filter - len(trials)} trials "
            f"(prefixes={prefixes}, explicit={len(excluded_trials)}, "
            f"held-out experiments={excluded_experiments}).",
            flush=True,
        )
    if not trials:
        raise ValueError("Every discovered trial was filtered out; nothing left to train on.")
    train_trials, val_trials, train_subjects, val_subjects = _split_by_subject(
        trials,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    if not train_trials or not val_trials:
        raise ValueError(f"Need non-empty train and validation splits; got train={len(train_trials)} val={len(val_trials)}")

    print(f"Direct torque dataset: {data_dir}")
    print(f"Trials: train={len(train_trials)} val={len(val_trials)}")
    print(f"Subjects: train={len(train_subjects)} val={len(val_subjects)}")
    if bool(args.quiet_steps):
        print(f"Val subjects: {len(val_subjects)} subject(s)")
    else:
        print(f"Val subjects: {val_subjects}")

    train_loader = _make_loader(train_trials, cfg, shuffle=True)
    val_loader = _make_loader(val_trials, cfg, shuffle=False)
    normalizers = compute_direct_normalizers(
        train_loader,
        int(args.normalizer_max_batches),
        quiet=bool(args.quiet_steps),
    )

    sample_batch = normalize_direct_batch(next(iter(train_loader)), normalizers)
    input_dim = int(sample_batch["input"].shape[-1])
    static_dim = int(sample_batch["static_context"].shape[-1])
    cfg["input_dim"] = input_dim
    cfg["static_dim"] = static_dim
    cfg["output_dim"] = DIRECT_TORQUE_OUTPUT_DIM
    cfg["model_structure"] = MODEL_STRUCTURE
    cfg["model_type"] = MODEL_STRUCTURE
    cfg["direct_torque_names"] = list(DIRECT_TORQUE_NAMES)
    cfg["exp_name"] = str(args.exp_name)
    cfg["save_model_epochs"] = _parse_epoch_list_arg(args.save_model_epochs)
    cfg["exclude_prefixes"] = prefixes
    cfg["exclude_trials"] = excluded_trials
    cfg["exclude_experiments"] = excluded_experiments
    cfg["include_experiments"] = included_experiments
    cfg["train_experiments"] = sorted({_experiment_key(t) for t in trials} - {""})
    layout = infer_input_feature_layout_from_loader(
        train_loader,
        include_pelvis_euler=bool(cfg["include_pelvis_euler"]),
        include_ankle_heights=bool(cfg["include_ankle_heights"]),
        include_jacobian_input=bool(cfg["include_jacobian_input"]),
        include_auxiliary_denoising_inputs=bool(cfg["include_auxiliary_denoising_inputs"]),
    )
    cfg["input_feature_layout"] = layout

    with (output_dir / "hyperparameters.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    with (output_dir / "split.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "n_train_trials": len(train_trials),
                "n_val_trials": len(val_trials),
                "held_out_experiments": excluded_experiments,
                "train_experiments": cfg["train_experiments"],
            },
            f,
            indent=2,
        )

    model = KinematicsToDirectTorque(
        input_dim=input_dim,
        static_dim=static_dim,
        output_dim=DIRECT_TORQUE_OUTPUT_DIM,
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout_rate=float(args.dropout_rate),
    )
    total_steps = max(1, int(train_loader.total_windows // int(args.batch_size)) * int(args.epochs))
    rng = jax.random.PRNGKey(int(args.seed))
    rng, init_rng = jax.random.split(rng)
    state, _lr_fn = create_train_state(
        init_rng,
        model,
        input_shape=(1, int(args.window_size), input_dim),
        static_shape=(1, static_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        total_steps=total_steps,
        use_lr_schedule=not bool(args.no_lr_schedule),
    )
    train_step = make_direct_train_step(normalizers, str(args.robust_loss), float(args.huber_delta))
    eval_step = make_direct_eval_step(normalizers, str(args.robust_loss), float(args.huber_delta))

    best_val = float("inf")
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = []
        first_train_raw = None
        for raw_batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} train",
            disable=bool(args.quiet_steps),
        ):
            if first_train_raw is None:
                first_train_raw = raw_batch
            batch = normalize_direct_batch(raw_batch, normalizers)
            rng, drop_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, drop_rng)
            train_metrics.append(metrics)
        train_summary = _mean_metrics(train_metrics)

        val_metrics = []
        first_val_raw = None
        first_val_norm = None
        first_val_pred = None
        for raw_batch in tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{args.epochs} val",
            disable=bool(args.quiet_steps),
        ):
            if first_val_raw is None:
                first_val_raw = raw_batch
            batch = normalize_direct_batch(raw_batch, normalizers)
            metrics, _pred = eval_step(state, batch)
            if first_val_pred is None:
                first_val_norm = batch
                first_val_pred = _pred
            val_metrics.append(metrics)
        val_summary = _mean_metrics(val_metrics)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_summary.items()},
            **{f"val_{k}": v for k, v in val_summary.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train loss={train_summary.get('loss', float('nan')):.4f}, "
            f"val loss={val_summary.get('loss', float('nan')):.4f}, "
            f"val MAE={val_summary.get('mae_bwh', float('nan')):.4f} %BW*H",
            flush=True,
        )
        if not bool(args.no_plots) and first_train_raw is not None and first_val_norm is not None:
            train_norm = normalize_direct_batch(first_train_raw, normalizers)
            _train_metrics_for_plot, train_pred = eval_step(state, train_norm)
            _plot_direct_torque_summary(
                train_norm,
                train_pred,
                first_val_norm,
                first_val_pred,
                normalizers,
                history,
                output_dir,
                epoch,
            )
        if val_summary.get("mae_bwh", float("inf")) < best_val:
            best_val = float(val_summary["mae_bwh"])
            _save_checkpoint(output_dir / "best_model.pkl", state, normalizers, cfg, epoch, val_summary)
            print(f"   ✅ Saved best_model.pkl (val MAE={best_val:.4f} %BW*H)")
        if int(epoch) in set(cfg.get("save_model_epochs", [])):
            _save_checkpoint(
                output_dir / f"model_epoch_{int(epoch):04d}.pkl",
                state,
                normalizers,
                cfg,
                epoch,
                val_summary,
            )

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    _save_checkpoint(output_dir / "final_model.pkl", state, normalizers, cfg, int(args.epochs), history[-1])
    print(f"Done. Best val MAE: {best_val:.4f} %BW*H")


if __name__ == "__main__":
    main()
