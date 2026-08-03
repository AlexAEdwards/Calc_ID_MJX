from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from paths import artifact, dataset  # noqa: E402

try:
    from .refine_q_shared import (
        RefineQRecord,
        build_model_from_hyperparams,
        compute_sequence_metrics,
        deserialize_normalizers,
        discover_refine_q_records,
        filter_records_for_selector,
        get_record_quantity_dof_names,
        load_refine_q_sample,
        plot_single_inference_summary,
        plot_open_capval_dashboard,
        predict_refined_sequence,
        sanitize_sample_id,
        save_prediction_bundle,
    )
    from .wandb_utils import configure_runtime_env
except ImportError:
    from refine_q_shared import (
        RefineQRecord,
        build_model_from_hyperparams,
        compute_sequence_metrics,
        deserialize_normalizers,
        discover_refine_q_records,
        filter_records_for_selector,
        get_record_quantity_dof_names,
        load_refine_q_sample,
        plot_single_inference_summary,
        plot_open_capval_dashboard,
        predict_refined_sequence,
        sanitize_sample_id,
        save_prediction_bundle,
    )
    from wandb_utils import configure_runtime_env


RUNTIME_ENV_APPLIED = configure_runtime_env()
OPEN_CAP_SUBJECTS_DIR = "Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects"


def _ts_print(*parts: Any) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}]", *parts, flush=True)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference for the residual refine-q transformer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=OPEN_CAP_SUBJECTS_DIR,
        help="Dataset root (defaults to the OpenCapSubjects dataset root)",
    )
    parser.add_argument("--output", type=str, default=str(artifact("inference_results", "refine_q")))
    parser.add_argument("--trial_name", type=str, default=None, help="subject/trial or subject/trial/sample_name")
    parser.add_argument("--test_json", type=str, default=None, help="JSON list of selectors to run")
    parser.add_argument("--all_val", action="store_true", help="Run all validation records stored in the checkpoint")
    parser.add_argument("--OpenCapVal", action="store_true", help="Use OpenCap ProcessedData inputs with MoCap ground truth")
    parser.add_argument("--OpenCapValDataset", action="store_true", help="Alias for --OpenCapVal")
    parser.add_argument("--OpenCapDataset", action="store_true", help="Alias for --OpenCapVal")
    return parser


def _records_from_checkpoint_payload(items: Sequence[Dict[str, Any]]) -> List[RefineQRecord]:
    return [RefineQRecord(**item) for item in items]


def _selectors_from_test_json(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    selectors: List[str] = []
    for item in payload:
        if isinstance(item, str):
            selectors.append(item)
        elif isinstance(item, dict):
            if "sample_id" in item:
                selectors.append(str(item["sample_id"]))
            elif "trial_name" in item:
                selectors.append(str(item["trial_name"]))
            elif "trial" in item:
                selectors.append(str(item["trial"]))
    return selectors


def _select_records(
    discovered_records: Sequence[RefineQRecord],
    *,
    checkpoint_payload: Dict[str, Any],
    trial_name: str | None,
    test_json: str | None,
    all_val: bool,
) -> List[RefineQRecord]:
    if all_val:
        checkpoint_val = _records_from_checkpoint_payload(checkpoint_payload.get("val_records", []))
        selector_ids = {record.sample_id for record in checkpoint_val}
        selected = [record for record in discovered_records if record.sample_id in selector_ids]
        return selected

    if test_json:
        selectors = _selectors_from_test_json(Path(test_json))
        selected: List[RefineQRecord] = []
        for selector in selectors:
            selected.extend(filter_records_for_selector(discovered_records, selector))
        unique: Dict[str, RefineQRecord] = {record.sample_id: record for record in selected}
        return list(unique.values())

    if trial_name:
        return filter_records_for_selector(discovered_records, trial_name)

    return list(discovered_records)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = _make_parser().parse_args()
    if args.OpenCapValDataset or args.OpenCapDataset:
        args.OpenCapVal = True

    if RUNTIME_ENV_APPLIED:
        _ts_print("Applied runtime env:", RUNTIME_ENV_APPLIED)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint_payload = pickle.load(f)

    hyperparameters = dict(checkpoint_payload.get("hyperparameters", {}))
    params = checkpoint_payload["params"]
    normalizers = deserialize_normalizers(checkpoint_payload["normalizers"])

    discovered_records = discover_refine_q_records(args.data_dir, opencap_val=bool(args.OpenCapVal))
    if not discovered_records:
        raise RuntimeError(f"No compatible refine-q records found under {args.data_dir}")

    selected_records = _select_records(
        discovered_records,
        checkpoint_payload=checkpoint_payload,
        trial_name=args.trial_name,
        test_json=args.test_json,
        all_val=bool(args.all_val),
    )
    if not selected_records:
        raise RuntimeError("No records matched the requested selection")

    preview_sample = load_refine_q_sample(selected_records[0])
    preview_input_dim = int(preview_sample.input_pos.shape[-1] + preview_sample.input_vel.shape[-1] + preview_sample.input_acc.shape[-1])
    preview_output_dim = int(preview_input_dim)

    input_dim = int(hyperparameters.get("input_dim", 69))
    static_dim = int(hyperparameters.get("static_dim", 2))
    output_dim = int(hyperparameters.get("output_dim", 69))
    if preview_input_dim != input_dim or preview_output_dim != output_dim:
        raise RuntimeError(
            "Checkpoint/data layout mismatch: checkpoint expects "
            f"input_dim={input_dim}, output_dim={output_dim}, but selected data resolves to "
            f"input_dim={preview_input_dim}, output_dim={preview_output_dim}. "
            "This usually means the checkpoint was trained before the OpenCap reduced-kinematics alignment change "
            "and needs to be retrained for the current OpenCap layout."
        )
    model = build_model_from_hyperparams(hyperparameters, input_dim=input_dim, static_dim=static_dim, output_dim=output_dim)

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    aggregate_summary: Dict[str, Any] = {"records": []}
    detailed_metrics: List[Dict[str, Any]] = []
    _ts_print(f"Running inference on {len(selected_records)} records...")

    for record in selected_records:
        sample = load_refine_q_sample(record)
        predictions = predict_refined_sequence(model, params, normalizers, sample)
        quantity_dof_names = get_record_quantity_dof_names(record)
        metrics = compute_sequence_metrics(predictions, quantity_dof_names)

        sample_dir = output_base / sanitize_sample_id(record.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = save_prediction_bundle(sample_dir / "refine_q_predictions.npz", predictions, quantity_dof_names, record.sample_id)
        plot_path = plot_single_inference_summary(
            sample_id=record.sample_id,
            predictions=predictions,
            dof_names=quantity_dof_names,
            output_path=sample_dir / "refine_q_summary.png",
        )

        summary_payload = {
            "sample_id": record.sample_id,
            "subject": record.subject,
            "trial": record.trial,
            "sample_name": record.sample_name,
            "source_kind": record.source_kind,
            "OpenCapVal": bool(record.opencap_val),
            "prediction_bundle": str(bundle_path),
            "summary_png": str(plot_path),
            "metrics": metrics,
        }
        _write_json(sample_dir / "refine_q_summary.json", summary_payload)
        detailed_metrics.append(metrics)
        aggregate_summary["records"].append(
            {
                "sample_id": record.sample_id,
                "pos_mean_rmse": float(metrics["pos"]["pred_mean_rmse"]),
                "vel_mean_rmse": float(metrics["vel"]["pred_mean_rmse"]),
                "acc_mean_rmse": float(metrics["acc"]["pred_mean_rmse"]),
                "pos_input_mean_rmse": float(metrics["pos"]["input_mean_rmse"]),
                "vel_input_mean_rmse": float(metrics["vel"]["input_mean_rmse"]),
                "acc_input_mean_rmse": float(metrics["acc"]["input_mean_rmse"]),
            }
        )
        _ts_print(
            f"{record.sample_id}",
            f"pos_mean_rmse={metrics['pos']['pred_mean_rmse']:.4f}",
            f"input_pos_rmse={metrics['pos']['input_mean_rmse']:.4f}",
        )

    if aggregate_summary["records"]:
        aggregate_summary["mean_pos_rmse"] = float(np.mean([item["pos_mean_rmse"] for item in aggregate_summary["records"]]))
        aggregate_summary["mean_vel_rmse"] = float(np.mean([item["vel_mean_rmse"] for item in aggregate_summary["records"]]))
        aggregate_summary["mean_acc_rmse"] = float(np.mean([item["acc_mean_rmse"] for item in aggregate_summary["records"]]))
    _write_json(output_base / "refine_q_aggregate_summary.json", aggregate_summary)
    _ts_print(f"Saved aggregate summary to {output_base / 'refine_q_aggregate_summary.json'}")

    if args.OpenCapVal and selected_records:
        dashboard_dir = output_base / "open_capval_dashboard"
        dashboard_outputs = plot_open_capval_dashboard(
            records=selected_records,
            metrics_list=detailed_metrics,
            output_dir=dashboard_dir,
        )
        _write_json(dashboard_dir / "dashboard_manifest.json", dashboard_outputs)
        _ts_print(f"Saved OpenCapVal dashboard artifacts to {dashboard_dir}")


if __name__ == "__main__":
    main()
