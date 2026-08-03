#!/usr/bin/env python3
"""LOSO fine-tuning for Older/Younger datasets in trusted layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MJX_DATALOADER_QUIET", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from loso_dataset_utils import (  # noqa: E402
    build_loso_folds, discover_trusted_trials, make_trusted_loader, parse_subject_list,
    validate_noised_inputs, validate_opensim_id_targets,
)


def _parse_optional_bool_arg(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True, help="Root containing Y*/OA*/Trial_*/ProcessedData.")
    p.add_argument("--checkpoint", help="Pretrained best_model.pkl; optional for discovery-only dry runs.")
    p.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "loso_older_younger"))
    p.add_argument("--include_subjects", nargs="*", default=[])
    p.add_argument("--exclude_subjects", nargs="*", default=[])
    p.add_argument("--held_out_subjects", nargs="*", default=[])
    p.add_argument("--max_trials_per_subject", type=int)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument(
        "--torque_weight",
        type=float,
        help="Override the regular torque-loss weight saved with the checkpoint.",
    )
    p.add_argument(
        "--Torque_weight_knee_adduction",
        "--torque_weight_knee_adduction",
        dest="torque_weight_knee_adduction",
        type=float,
        help=(
            "Override the knee-adduction/KAM loss weight. Its interpretation is controlled "
            "by --torque_weight_knee_adduction_mode."
        ),
    )
    p.add_argument(
        "--Torque_weight_knee_adduction_mode",
        "--torque_weight_knee_adduction_mode",
        dest="torque_weight_knee_adduction_mode",
        choices=("absolute", "first_step_ratio"),
        help=(
            "Interpret the KAM weight as an absolute multiplier or as the target first-batch "
            "scaled KAM/scaled regular-torque loss ratio."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_noised", choices=("auto", "true", "false"), default="auto")
    p.add_argument(
        "--use_OpenSimID_GT",
        nargs="?",
        const=True,
        default=None,
        type=_parse_optional_bool_arg,
        help=(
            "Use time-aligned OpenSim inverse-dynamics .sto/.mot moments as the training and "
            "evaluation torque ground truth for direct-torque and torque-informed models. The "
            "two KAM channels remain measured-GRF/moment-arm targets because the OpenSim model "
            "has no knee-adduction coordinate. Fails if any trial cannot be aligned exactly."
        ),
    )
    p.add_argument(
        "--subtractAnkleHeightKneeVecs",
        nargs="?",
        const=True,
        default=False,
        type=_parse_optional_bool_arg,
        help=(
            "Correct legacy KneeToCOP_Vectors.npy in memory by subtracting each ankle's "
            "world-Z height from the corresponding vector Z component. Use this for existing "
            "datasets; do not enable it after regenerating vectors with the corrected ProcessData.py."
        ),
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--discovery_only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite_fold", action="store_true")
    return p.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_bool(raw: Mapping[str, Any], *keys: str, default: bool) -> bool:
    value = next((raw[k] for k in keys if k in raw), default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _load_checkpoint(path: Path):
    try:
        import loso_from_checkpoint as legacy
        checkpoint, config = legacy._load_checkpoint_bundle(path)
        raw = json.loads(path.with_name("hyperparameters.json").read_text(encoding="utf-8"))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The training environment is missing a dependency (usually JAX/Flax). "
            "Run this script in the same environment used for TransformerFinal training."
        ) from exc
    config = dict(config)
    config.update({
        "include_ankle_heights": _raw_bool(raw, "include_ankle_heights", "includeAnkleHeights", default=True),
        "include_jacobian_input": _raw_bool(raw, "include_jacobian_input", "includeJacobianInput", default=True),
        "include_auxiliary_denoising_inputs": _raw_bool(raw, "include_auxiliary_denoising_inputs", default=True),
        "use_os_filtering": _raw_bool(raw, "use_os_filtering", default=False),
        "use_grf_nofilt": _raw_bool(raw, "use_grf_nofilt", default=True),
        "robust_loss": str(raw.get("robust_loss", "huber")),
        "huber_delta": float(raw.get("huber_delta", 1.0)),
        "torque_weight_knee_adduction": float(
            raw.get(
                "Torque_weight_knee_adduction",
                raw.get("torque_weight_knee_adduction", 0.0),
            ) or 0.0
        ),
        "torque_weight_knee_adduction_mode": str(
            raw.get(
                "Torque_weight_knee_adduction_mode",
                raw.get("torque_weight_knee_adduction_mode", "absolute"),
            ) or "absolute"
        ).strip().lower(),
        "source_checkpoint": str(path),
    })
    if not config.get("window_size"):
        raise ValueError("Checkpoint hyperparameters must define a positive window_size.")
    from loso_finetune_core import validate_checkpoint
    validate_checkpoint(checkpoint, config)
    return checkpoint, config, raw


def _compatibility_check(discovery: Mapping[str, Any], checkpoint, config, batch_size: int) -> Dict[str, int]:
    validate_noised_inputs(discovery["trials"], bool(config["use_noised"]))
    loader = make_trusted_loader(discovery["trials"][:1], config, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    actual = {"input_dim": int(batch["input"].shape[-1]), "static_dim": int(batch["static_context"].shape[-1])}
    params = checkpoint["params"]
    expected = {
        "input_dim": int(params["Dense_0"]["kernel"].shape[0]),
        "static_dim": int(params["Dense_1"]["kernel"].shape[0]),
    }
    if actual != expected:
        raise ValueError(f"Checkpoint expects {expected}, but trusted-layout loader produced {actual}.")
    return actual


def _run_paired_inference(fold, fold_dir: Path, checkpoint, config, batch_size: int):
    """Evaluate source and fold checkpoints on the exact same held-out windows."""
    import loso_adapters
    from loso_inference_compare import compare_trial, write_ankle_power_stance_report
    from loso_reporting import write_loso_reports
    from loso_dataset_utils import make_trusted_loader

    with (fold_dir / "best_model.pkl").open("rb") as handle:
        fine_tuned = pickle.load(handle)
    model = loso_adapters.build_loso_model(config, checkpoint["params"])
    loader_factory = lambda trials: make_trusted_loader(  # noqa: E731
        trials if isinstance(trials, list) else [trials], config,
        batch_size=batch_size, shuffle=False,
    )
    inference_dir = fold_dir / "inference"
    trial_metrics = []
    for trial in fold["held_out_trials"]:
        compared = compare_trial(
            trial=trial, model=model, original_checkpoint=checkpoint,
            fine_tuned_checkpoint=fine_tuned, loader_factory=loader_factory,
            model_structure=config["model_structure"], output_dir=inference_dir, config=config,
        )
        write_ankle_power_stance_report(
            trial={**trial, "use_noised": bool(config.get("use_noised", False))},
            comparison=compared,
            model_structure=config["model_structure"],
            output_root=fold_dir / "ankle_power_results",
        )
        trial_metrics.append(compared["metrics"])
    (inference_dir / "trial_metrics.json").write_text(json.dumps(trial_metrics, indent=2), encoding="utf-8")
    write_loso_reports(
        fold_dir, trial_metrics, dataset_path=str(fold["held_out_trials"][0]["dataset_root"]),
        source_checkpoint=str(config.get("source_checkpoint", "")),
        fold_checkpoint=str(fold_dir / "best_model.pkl"),
    )
    return trial_metrics


def main() -> None:
    args = parse_args()
    include = parse_subject_list(args.include_subjects)
    exclude = parse_subject_list(args.exclude_subjects)
    held_out = parse_subject_list(args.held_out_subjects)
    discovery = discover_trusted_trials(
        args.data_dir, include_subjects=include, exclude_subjects=exclude,
        max_trials_per_subject=args.max_trials_per_subject,
    )
    folds = build_loso_folds(discovery["subject_to_trials"], held_out_subjects=held_out)
    report = {
        "data_dir": discovery["data_dir"], "layout": "trusted",
        "subject_count": len(discovery["all_subjects"]),
        "valid_subject_count": len(discovery["subjects"]), "trial_count": len(discovery["trials"]),
        "all_subjects": discovery["all_subjects"],
        "subjects": discovery["subjects"], "trial_counts": discovery["trial_counts"],
        "held_out_subjects": [f["held_out_subject"] for f in folds],
        "skipped_trial_count": len(discovery["skipped_trials"]),
        "skipped_trials": discovery["skipped_trials"],
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.discovery_only or (args.dry_run and not args.checkpoint):
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --discovery_only (or checkpoint-free --dry_run) is used.")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint, config, raw_hparams = _load_checkpoint(checkpoint_path)
    if args.torque_weight is not None:
        config["torque_weight"] = float(args.torque_weight)
    if args.torque_weight_knee_adduction is not None:
        config["torque_weight_knee_adduction"] = float(args.torque_weight_knee_adduction)
    if args.torque_weight_knee_adduction_mode is not None:
        config["torque_weight_knee_adduction_mode"] = str(args.torque_weight_knee_adduction_mode)
    for name in ("torque_weight", "torque_weight_knee_adduction"):
        value = float(config.get(name, 0.0))
        if not value >= 0.0 or not math.isfinite(value):
            raise ValueError(f"--{name} must be a finite, non-negative value; got {value!r}.")
    kam_weight_mode = str(config.get("torque_weight_knee_adduction_mode", "absolute"))
    if kam_weight_mode not in {"absolute", "first_step_ratio"}:
        raise ValueError(
            "Knee-adduction loss mode must be 'absolute' or 'first_step_ratio', "
            f"not {kam_weight_mode!r}."
        )
    if args.use_noised != "auto":
        config["use_noised"] = args.use_noised == "true"
    config["use_OpenSimID_GT"] = bool(args.use_OpenSimID_GT)
    config["subtract_ankle_height_knee_vecs"] = bool(args.subtractAnkleHeightKneeVecs)
    model_structure = str(config.get("model_structure", "")).lower()
    if model_structure == "direct_torque" and any((
        args.torque_weight is not None,
        args.torque_weight_knee_adduction is not None,
        args.torque_weight_knee_adduction_mode is not None,
    )):
        raise ValueError(
            "The torque/KAM loss-weight overrides apply to torque-informed cop_grf_moments "
            "models. Direct-torque checkpoints use their direct output loss instead."
        )
    opensim_supported_structures = {"direct_torque", "cop_grf_moments"}
    if config["use_OpenSimID_GT"] and model_structure not in opensim_supported_structures:
        raise ValueError(
            "--use_OpenSimID_GT training targets are supported for direct_torque and "
            f"torque-informed cop_grf_moments checkpoints, not {model_structure!r}."
        )
    if config["use_OpenSimID_GT"] and bool(config.get("use_recalculated_opensim_id_gt", False)):
        raise ValueError(
            "--use_OpenSimID_GT cannot be combined with use_recalculated_opensim_id_gt."
        )
    opensim_id_audit = None
    if config["use_OpenSimID_GT"]:
        print(
            "Validating timestamp alignment for every OpenSim ID training/evaluation target...",
            flush=True,
        )
        opensim_id_audit = validate_opensim_id_targets(discovery["trials"])
        print(
            f"Validated aligned OpenSim ID targets for {opensim_id_audit['trial_count']} trials.",
            flush=True,
        )
    batch_size = int(args.batch_size or config.get("batch_size") or 64)
    dimensions = _compatibility_check(discovery, checkpoint, config, batch_size)
    report.update({"checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
                   "model_structure": config["model_structure"], **dimensions})
    print(f"model_structure={config['model_structure']} input_dim={dimensions['input_dim']} static_dim={dimensions['static_dim']}", flush=True)
    if args.dry_run:
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config: Dict[str, Any] = {
        **report, "epochs": int(args.epochs), "learning_rate": float(args.learning_rate),
        "batch_size": batch_size, "weight_decay": float(args.weight_decay), "seed": int(args.seed),
        "use_noised": bool(config["use_noised"]),
        "use_OpenSimID_GT": bool(config["use_OpenSimID_GT"]),
        "subtractAnkleHeightKneeVecs": bool(
            config["subtract_ankle_height_knee_vecs"]
        ),
        "torque_weight": float(config.get("torque_weight", 0.0)),
        "torque_weight_knee_adduction": float(config.get("torque_weight_knee_adduction", 0.0)),
        "torque_weight_knee_adduction_mode": str(
            config.get("torque_weight_knee_adduction_mode", "absolute")
        ),
        "opensim_id_alignment": (
            opensim_id_audit["alignment"] if opensim_id_audit is not None else None
        ),
        "torque_ground_truth": (
            "Aligned OpenSim full ID (external-torque target = qfrc_inverse - OpenSim ID); "
            "measured GRF/moment-arm KAM remains separate"
            if config["use_OpenSimID_GT"] else "checkpoint/default trusted-layout targets"
        ),
    }
    run_config_path = output_dir / "run_config.json"
    if args.resume and run_config_path.exists():
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        comparable = (
            "data_dir", "checkpoint_sha256", "epochs", "learning_rate", "batch_size",
            "weight_decay", "use_noised", "use_OpenSimID_GT",
            "subtractAnkleHeightKneeVecs", "torque_weight",
            "torque_weight_knee_adduction", "torque_weight_knee_adduction_mode",
        )
        changed = [key for key in comparable if previous.get(key) != run_config.get(key)]
        if changed:
            raise ValueError(f"Unsafe resume: run configuration changed for {changed}.")
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if opensim_id_audit is not None:
        (output_dir / "opensim_id_alignment_audit.json").write_text(
            json.dumps(opensim_id_audit, indent=2), encoding="utf-8"
        )

    from loso_finetune_core import run_finetune_fold, save_json
    summaries = []
    all_trial_metrics = []
    for index, fold in enumerate(folds):
        subject = fold["held_out_subject"]
        fold_dir = output_dir / f"held_out_{subject}"
        marker = fold_dir / "FINETUNE_COMPLETE.json"
        if marker.exists() and args.resume and not args.overwrite_fold:
            print(f"[{subject}] already complete; skipping", flush=True)
            summaries.append(json.loads(marker.read_text(encoding="utf-8")))
            metrics_path = fold_dir / "inference" / "trial_metrics.json"
            if metrics_path.exists():
                all_trial_metrics.extend(json.loads(metrics_path.read_text(encoding="utf-8")))
            continue
        result = run_finetune_fold(
            fold, fold_dir=fold_dir, checkpoint=checkpoint, config=config,
            epochs=int(args.epochs), learning_rate=float(args.learning_rate), batch_size=batch_size,
            weight_decay=float(args.weight_decay), seed=int(args.seed) + index,
        )
        fold_hparams = dict(raw_hparams)
        fold_hparams.update({
            "use_OpenSimID_GT": bool(config["use_OpenSimID_GT"]),
            "subtractAnkleHeightKneeVecs": bool(
                config["subtract_ankle_height_knee_vecs"]
            ),
            "opensim_id_alignment": run_config.get("opensim_id_alignment"),
            "torque_ground_truth": run_config["torque_ground_truth"],
            "torque_weight": float(config.get("torque_weight", 0.0)),
            "Torque_weight_knee_adduction": float(
                config.get("torque_weight_knee_adduction", 0.0)
            ),
            "Torque_weight_knee_adduction_mode": str(
                config.get("torque_weight_knee_adduction_mode", "absolute")
            ),
            "effective_Torque_weight_knee_adduction": config.get(
                "effective_torque_weight_knee_adduction"
            ),
            "torque_weight_knee_adduction_calibration": config.get(
                "torque_weight_knee_adduction_calibration"
            ),
        })
        (fold_dir / "hyperparameters.json").write_text(
            json.dumps(fold_hparams, indent=2), encoding="utf-8"
        )
        trial_metrics = _run_paired_inference(fold, fold_dir, checkpoint, config, batch_size)
        result["inference_trial_count"] = len(trial_metrics)
        all_trial_metrics.extend(trial_metrics)
        save_json(marker, result)
        summaries.append(result)
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
    save_json(output_dir / "finetune_summary.json", {"folds": summaries})
    if all_trial_metrics:
        from loso_reporting import write_loso_reports
        write_loso_reports(
            output_dir, all_trial_metrics, dataset_path=discovery["data_dir"],
            source_checkpoint=str(checkpoint_path), fold_checkpoint="per-fold best_model.pkl",
        )


if __name__ == "__main__":
    main()
