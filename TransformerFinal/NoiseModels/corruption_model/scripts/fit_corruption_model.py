from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corruption_model.config import load_config
from corruption_model.evaluation.metrics import (
    compute_per_joint_psd_stats,
    compute_per_joint_residual_stats_with_points,
    compute_real_trialwise_residual_summary,
)
from corruption_model.evaluation.plots import plot_psd_comparison
from corruption_model.io.load_paired import load_paired_trials
from corruption_model.models.full_corruptor import FullCorruptor
from corruption_model.preprocess.align import estimate_global_lag
from corruption_model.preprocess.harmonize import harmonize_trial_pair
from corruption_model.residuals.compute_residuals import compute_residual_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the OpenCap-style corruption model from paired Motion vs trimmed "
            "MoCap kinematics on their shared time overlap."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-model", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paired_trials = [
        harmonize_trial_pair(trial, sample_rate_hz=config.representation.sample_rate_hz)
        for trial in load_paired_trials(config.data.paired_path, metadata_filename=config.data.subject_metadata_filename)
    ]
    residual_trials = []
    for trial in paired_trials:
        alignment = estimate_global_lag(
            trial,
            sample_rate_hz=config.representation.sample_rate_hz,
            max_lag_frames=config.model.lag_max_frames,
        )
        residual_trials.append(
            compute_residual_trial(
                trial=trial,
                q_mocap_aligned=alignment.q_mocap_aligned,
                q_opencap_aligned=alignment.q_opencap_aligned,
                lag_frames=alignment.lag_frames,
                lag_seconds=alignment.lag_seconds,
                alignment_score=alignment.alignment_score,
            )
        )
    corruptor = FullCorruptor(config=config).fit(paired_trials)
    output_dir = Path(config.data.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.output_model) if args.output_model else output_dir / "corruptor.pkl"
    corruptor.save(model_path)
    (output_dir / "fit_summary.json").write_text(json.dumps(corruptor.fit_summary_, indent=2), encoding="utf-8")
    if residual_trials:
        dof_names = residual_trials[0].subject_metadata.dof_names
        stacked_residual = np.concatenate([trial.residual for trial in residual_trials], axis=0)
        residual_stats = compute_per_joint_residual_stats_with_points(stacked_residual, dof_names)
        residual_stats["source"] = "real_opencap_minus_mocap"
        residual_stats["num_trials"] = len(residual_trials)
        (output_dir / "residual_joint_stats.json").write_text(json.dumps(residual_stats, indent=2), encoding="utf-8")
        trialwise_residual_stats = compute_real_trialwise_residual_summary(
            [trial.residual for trial in residual_trials],
            dof_names=dof_names,
        )
        trialwise_residual_stats["source"] = "real_opencap_minus_mocap"
        trialwise_residual_stats["num_trials"] = len(residual_trials)
        (output_dir / "residual_joint_trialwise_stats.json").write_text(json.dumps(trialwise_residual_stats, indent=2), encoding="utf-8")
        psd_stats = compute_per_joint_psd_stats(
            stacked_residual,
            dof_names=dof_names,
            fs_hz=config.representation.sample_rate_hz,
        )
        psd_stats["source"] = "real_opencap_minus_mocap"
        psd_stats["num_trials"] = len(residual_trials)
        (output_dir / "residual_joint_psd.json").write_text(json.dumps(psd_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
