from __future__ import annotations

import argparse
import json
from pathlib import Path

from corruption_model.config import load_config
from corruption_model.evaluation.metrics import compute_basic_metrics, covariance_similarity
from corruption_model.evaluation.plots import plot_trial_overlay
from corruption_model.io.load_paired import load_paired_trials
from corruption_model.models.full_corruptor import FullCorruptor
from corruption_model.preprocess.harmonize import harmonize_trial_pair
from corruption_model.preprocess.phase import compute_stance_swing_phase_positions_from_grf


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the corruption model with leave-one-subject-out folds.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    paired_trials = [
        harmonize_trial_pair(trial, sample_rate_hz=config.representation.sample_rate_hz)
        for trial in load_paired_trials(config.data.paired_path, metadata_filename=config.data.subject_metadata_filename)
    ]
    subjects = sorted({trial.subject_id for trial in paired_trials})
    output_dir = Path(config.data.output_path) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    for subject in subjects:
        train_trials = [trial for trial in paired_trials if trial.subject_id != subject]
        test_trials = [trial for trial in paired_trials if trial.subject_id == subject]
        if not train_trials or not test_trials:
            continue
        corruptor = FullCorruptor(config=config).fit(train_trials)
        per_trial = []
        for trial_idx, trial in enumerate(test_trials):
            phase_positions = None
            if trial.grf is not None:
                phase_positions = compute_stance_swing_phase_positions_from_grf(
                    trial.grf,
                    target_length=trial.q_mocap.shape[0],
                )
            synthetic, _ = corruptor.sample(
                trial.q_mocap,
                meta={"height_m": trial.subject_metadata.height_m, "phase_positions": phase_positions},
                random_state=config.generation.random_seed + trial_idx,
            )
            metrics = compute_basic_metrics(trial.q_opencap, synthetic)
            metrics["covariance_l1"] = covariance_similarity(trial.q_opencap - trial.q_mocap, synthetic - trial.q_mocap)
            metrics["subject_id"] = subject
            metrics["trial_id"] = trial.trial_id
            per_trial.append(metrics)
            if config.evaluation.save_plots and trial_idx < config.evaluation.plots_max_trials:
                plot_trial_overlay(
                    clean=trial.q_mocap,
                    real=trial.q_opencap,
                    synthetic=synthetic,
                    output_path=output_dir / f"{subject}_{trial.trial_id}_overlay.png",
                )
        fold_metrics.extend(per_trial)

    (output_dir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
