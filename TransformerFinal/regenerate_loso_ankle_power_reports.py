#!/usr/bin/env python3
"""Regenerate LOSO ankle-power JSONs from already-saved paired inference arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from loso_inference_compare import write_ankle_power_stance_report  # noqa: E402


REQUIRED_ARRAYS = (
    "original_torque_nm",
    "fine_tuned_torque_nm",
    "target_torque_nm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("loso_output_dir", type=Path)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Dataset root; defaults to data_dir recorded in run_config.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help=(
            "Optional separate output root. Reports are grouped under held_out_<subject>/"
            "ankle_power_results; defaults to writing inside the LOSO output directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.loso_output_dir.expanduser().resolve()
    report_output_root = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None else output_root
    )
    report_output_root.mkdir(parents=True, exist_ok=True)
    run_config_path = output_root / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    data_root = (args.data_dir or Path(run_config["data_dir"])).expanduser().resolve()
    model_structure = str(run_config["model_structure"])
    ground_truth_label = (
        "Aligned OpenSim ID GT" if bool(run_config.get("use_OpenSimID_GT", False)) else "MJX GT"
    )

    written = 0
    skipped = []
    for fold_dir in sorted(output_root.glob("held_out_*")):
        inference_root = fold_dir / "inference"
        if not inference_root.is_dir():
            continue
        for trial_dir in sorted(path for path in inference_root.iterdir() if path.is_dir()):
            metrics_path = trial_dir / "metrics.json"
            if not metrics_path.is_file():
                skipped.append({"trial_dir": str(trial_dir), "reason": "missing metrics.json"})
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            subject = str(metrics["subject"])
            trial_name = str(metrics["trial"]).split("/", 1)[-1]
            processed = data_root / subject / trial_name / "ProcessedData"
            missing = [name for name in REQUIRED_ARRAYS if not (trial_dir / f"{name}.npy").is_file()]
            if missing or not processed.is_dir():
                reason = f"missing arrays: {missing}" if missing else f"missing processed data: {processed}"
                skipped.append({"trial_dir": str(trial_dir), "reason": reason})
                continue
            comparison = {
                "arrays": {
                    name: np.load(trial_dir / f"{name}.npy") for name in REQUIRED_ARRAYS
                },
                "evaluation_mask": np.load(trial_dir / "evaluation_mask.npy"),
            }
            write_ankle_power_stance_report(
                trial={
                    "subject": subject,
                    "trial": trial_name,
                    "trial_root": str(processed.parent),
                    "training_data_path": str(processed),
                    "use_noised": bool(run_config.get("use_noised", False)),
                    "ground_truth_label": ground_truth_label,
                },
                comparison=comparison,
                model_structure=model_structure,
                output_root=(
                    report_output_root / fold_dir.name / "ankle_power_results"
                    if args.output_dir is not None
                    else fold_dir / "ankle_power_results"
                ),
            )
            written += 1

    report = {
        "source_loso_output_dir": str(output_root),
        "output_dir": str(report_output_root),
        "ground_truth_label": ground_truth_label,
        "written": written,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    report_path = report_output_root / "ankle_power_regeneration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
