"""Fixed-epoch fine-tuning core shared by trusted-layout LOSO folds."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import jax
import numpy as np

import loso_adapters
import loso_from_checkpoint as legacy_loso
import train as train_module
from loso_dataset_utils import make_trusted_loader, validate_noised_inputs

try:
    from train_directTorque import make_direct_eval_step, make_direct_train_step
except Exception:  # pragma: no cover
    make_direct_eval_step = make_direct_train_step = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")


def is_direct(config: Mapping[str, Any]) -> bool:
    return str(config.get("model_structure", "")).lower() == "direct_torque"


def validate_checkpoint(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    for key in ("params", "normalizers"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key {key!r}.")
    normalizers = checkpoint["normalizers"]
    required = {"input", "static", "direct_torque"} if is_direct(config) else {"input", "static", "cop", "grf"}
    missing = sorted(required - set(normalizers))
    if missing:
        raise KeyError(f"Checkpoint normalizers are missing: {missing}")


def run_finetune_fold(
    fold: Mapping[str, Any],
    *,
    fold_dir: Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
) -> Dict[str, Any]:
    """Fine-tune all model parameters and save a checkpoint for one LOSO fold."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    held_out = str(fold["held_out_subject"])
    if held_out in set(fold["train_subjects"]):
        raise ValueError(f"LOSO leakage: {held_out} appears in its training subject list.")
    all_trials = list(fold["train_trials"]) + list(fold["held_out_trials"])
    validate_noised_inputs(all_trials, bool(config.get("use_noised", False)))
    save_json(fold_dir / "split.json", {
        "held_out_subject": held_out,
        "train_subjects": list(fold["train_subjects"]),
        "train_trials": list(fold["train_trials"]),
        "held_out_trials": list(fold["held_out_trials"]),
        "layout": "trusted",
    })

    train_loader = make_trusted_loader(fold["train_trials"], config, batch_size=batch_size, shuffle=True)
    test_loader = make_trusted_loader(fold["held_out_trials"], config, batch_size=batch_size, shuffle=False)
    sample = next(iter(train_loader))
    input_dim = int(sample["input"].shape[-1])
    static_dim = int(sample["static_context"].shape[-1])
    params = checkpoint["params"]
    checkpoint_input_dim = int(np.asarray(params["Dense_0"]["kernel"]).shape[0])
    checkpoint_static_dim = int(np.asarray(params["Dense_1"]["kernel"]).shape[0])
    if (input_dim, static_dim) != (checkpoint_input_dim, checkpoint_static_dim):
        raise ValueError(
            "Checkpoint/loader dimension mismatch: "
            f"checkpoint=({checkpoint_input_dim}, {checkpoint_static_dim}), "
            f"trusted loader=({input_dim}, {static_dim}). Adjust the checkpoint feature flags."
        )

    model = loso_adapters.build_loso_model(config, params)
    rng = jax.random.PRNGKey(int(seed))
    rng, init_rng = jax.random.split(rng)
    state = loso_adapters.create_loso_train_state(
        init_rng, model, params,
        input_shape=(1, int(config["window_size"]), input_dim),
        static_shape=(1, static_dim),
        learning_rate=float(learning_rate), weight_decay=float(weight_decay),
    )
    normalizers = checkpoint["normalizers"]
    loss_weights = legacy_loso._build_loss_weights(config)
    direct = is_direct(config)
    if direct:
        if make_direct_train_step is None:
            raise ImportError("Direct-torque training dependencies could not be imported.")
        train_step = make_direct_train_step(normalizers, str(config.get("robust_loss", "huber")), float(config.get("huber_delta", 1.0)))
        eval_step = make_direct_eval_step(normalizers, str(config.get("robust_loss", "huber")), float(config.get("huber_delta", 1.0)))
    else:
        dof_weights = legacy_loso._build_dof_weights(config)
        common = (
            normalizers, bool(config.get("use_contact_weighting", False)),
            bool(config.get("mag_on_off", False)), bool(config.get("contact_on_off", False)),
            False, float(config.get("contact_weight_multiplier", 1.5)),
            float(config.get("mag_weight", 3.0)), max(1, int(epochs)), dof_weights,
        )
        train_step = train_module.make_train_step(
            *common, cop_mask=bool(config.get("cop_mask", True)),
            use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
            use_full_id_gt_for_torque=bool(config.get("use_OpenSimID_GT", False)),
        )
        eval_step = train_module.make_eval_step(
            *common, cop_mask=bool(config.get("cop_mask", True)),
            use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
            use_full_id_gt_for_torque=bool(config.get("use_OpenSimID_GT", False)),
        )
        calibration_eval_step = train_module.make_eval_step(
            *common, cop_mask=bool(config.get("cop_mask", True)),
            use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
            use_full_id_gt_for_torque=bool(config.get("use_OpenSimID_GT", False)),
        )
        if legacy_loso._configure_kam_loss_weight_for_mode(
            loss_weights,
            config,
            log_prefix=f"[{held_out}]",
        ):
            legacy_loso._calibrate_kam_loss_weight_first_batch(
                state=state,
                train_loader=train_loader,
                eval_step=calibration_eval_step,
                normalizers=normalizers,
                loss_weights=loss_weights,
                config=config,
                epoch=1,
                log_prefix=f"[{held_out}]",
            )

    history = [{"epoch": 0, "train_losses": None}]
    for epoch in range(1, int(epochs) + 1):
        started = time.time()
        state, losses, rng = legacy_loso._run_train_epoch(
            state, train_loader, train_step=train_step, normalizers=normalizers,
            loss_weights=loss_weights, rng=rng, epoch=epoch, direct_torque=direct,
        )
        history.append({"epoch": epoch, "duration_s": time.time() - started, "train_losses": losses})
        print(f"[{held_out}] epoch {epoch}/{epochs} loss={losses['total_loss']:.6g}", flush=True)

    metrics = legacy_loso._evaluate_loader(
        state, test_loader, eval_step=eval_step, normalizers=normalizers,
        loss_weights=loss_weights, config=config, epoch=int(epochs),
    )
    output_bundle = dict(checkpoint)
    output_bundle["params"] = state.params
    output_bundle["normalizers"] = normalizers
    output_bundle["loso_metadata"] = {
        "held_out_subject": held_out, "epochs": int(epochs),
        "model_structure": config["model_structure"], "source_checkpoint": config.get("source_checkpoint"),
        "use_OpenSimID_GT": bool(config.get("use_OpenSimID_GT", False)),
        "torque_target_definition": (
            "qfrc_inverse_raw - aligned_OpenSim_ID"
            if bool(config.get("use_OpenSimID_GT", False))
            else "qfrc_grf_contribution"
        ),
        "torque_weight": float(loss_weights.get("torque", 0.0)),
        "torque_weight_knee_adduction_mode": str(
            config.get("torque_weight_knee_adduction_mode", "absolute")
        ),
        "requested_torque_weight_knee_adduction": float(
            config.get(
                "requested_torque_weight_knee_adduction",
                config.get("torque_weight_knee_adduction", 0.0),
            )
        ),
        "effective_torque_weight_knee_adduction": float(
            config.get(
                "effective_torque_weight_knee_adduction",
                loss_weights.get("torque_knee_adduction", 0.0),
            )
        ),
        "torque_weight_knee_adduction_calibration": config.get(
            "torque_weight_knee_adduction_calibration"
        ),
    }
    with (fold_dir / "best_model.pkl").open("wb") as handle:
        pickle.dump(output_bundle, handle)
    save_json(fold_dir / "training_history.json", {
        "epochs": history,
        "loss_weights": loss_weights,
        "torque_weight_knee_adduction_mode": str(
            config.get("torque_weight_knee_adduction_mode", "absolute")
        ),
        "torque_weight_knee_adduction_calibration": config.get(
            "torque_weight_knee_adduction_calibration"
        ),
    })
    save_json(fold_dir / "metrics.json", metrics)
    return {"held_out_subject": held_out, "checkpoint": str(fold_dir / "best_model.pkl"), "metrics": metrics}
