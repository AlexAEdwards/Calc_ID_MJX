from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corruption_model.config import load_config
from corruption_model.evaluation.metrics import (
    compute_average_curve_psd_stats,
    compute_average_curve_residual_stats,
    compute_per_joint_residual_stats_with_points,
    compute_trialwise_residual_summary,
    compute_residual_scale_vector_from_stats,
)
from corruption_model.evaluation.mujoco_viewer import (
    build_viewer_qpos_from_motion_pos,
    resolve_subject_mujoco_model_path,
    show_dry_run_mujoco_viewer,
)
from corruption_model.evaluation.plots import (
    plot_noised_curves_against_gt,
    plot_psd_comparison,
    plot_residual_stats_boxplot_comparison,
    plot_residual_stats_comparison,
)
from corruption_model.io.load_mocap_only import load_mocap_trials
from corruption_model.io.save_dataset import save_processeddata_outputs
from corruption_model.models.full_corruptor import FullCorruptor
from corruption_model.preprocess.filter import differentiate_signal
from corruption_model.preprocess.harmonize import harmonize_mocap_trial
from corruption_model.preprocess.phase import compute_stance_swing_phase_positions_from_grf


def _save_trial_visualization_outputs(
    *,
    config,
    trial,
    corrupted_curves: list[dict],
    plot_curves: list[np.ndarray],
    real_residual_stats: dict | None,
    real_psd_stats: dict | None,
    output_dir: Path,
    residual_scale_vector_path: Path | None,
    residual_scale_vector: np.ndarray | None,
    noise_scale: float,
    show_viewer: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_subject_mujoco_model_path(Path(trial.meta["source_dataset_path"]))
    plot_path = plot_noised_curves_against_gt(
        clean=trial.pos,
        synthetic_curves=plot_curves,
        dof_names=trial.subject_metadata.dof_names,
        output_path=output_dir / f"{trial.subject_id}_{trial.trial_id}_noised_vs_gt.png",
    )
    synthetic_residual_stats = compute_average_curve_residual_stats(
        clean=trial.pos,
        synthetic_curves=plot_curves,
        dof_names=trial.subject_metadata.dof_names,
    )
    synthetic_stats_path = output_dir / f"{trial.subject_id}_{trial.trial_id}_synthetic_residual_joint_stats.json"
    synthetic_stats_path.write_text(json.dumps(synthetic_residual_stats, indent=2), encoding="utf-8")
    synthetic_psd_stats = compute_average_curve_psd_stats(
        clean=trial.pos,
        synthetic_curves=plot_curves,
        dof_names=trial.subject_metadata.dof_names,
        fs_hz=config.representation.sample_rate_hz,
    )
    synthetic_psd_stats_path = output_dir / f"{trial.subject_id}_{trial.trial_id}_synthetic_residual_joint_psd.json"
    synthetic_psd_stats_path.write_text(json.dumps(synthetic_psd_stats, indent=2), encoding="utf-8")
    residual_scale_suggestion_path = None
    if real_residual_stats is not None:
        residual_scale_vector_suggestion = compute_residual_scale_vector_from_stats(
            real_stats=real_residual_stats,
            synthetic_stats=synthetic_residual_stats,
            dof_names=trial.subject_metadata.dof_names,
        )
        residual_scale_suggestion_path = output_dir / f"{trial.subject_id}_{trial.trial_id}_residual_scale_vector.npy"
        np.save(residual_scale_suggestion_path, residual_scale_vector_suggestion.astype(np.float32))
    residual_comparison_plot_path = None
    psd_comparison_plot_path = None
    if real_residual_stats is not None:
        residual_comparison_plot_path = plot_residual_stats_comparison(
            real_stats=real_residual_stats,
            synthetic_stats=synthetic_residual_stats,
            output_path=output_dir / f"{trial.subject_id}_{trial.trial_id}_residual_stats_comparison.png",
        )
    if real_psd_stats is not None:
        psd_comparison_plot_path = plot_psd_comparison(
            real_psd_stats=real_psd_stats,
            synthetic_psd_stats=synthetic_psd_stats,
            output_path=output_dir / f"{trial.subject_id}_{trial.trial_id}_psd_comparison.png",
    )
    viewer_qpos_files = []
    viewer_results = []
    for curve_idx, curve in enumerate(corrupted_curves, start=1):
        qpos_matrix = build_viewer_qpos_from_motion_pos(
            curve["pos"],
            model_path=model_path,
        )
        qpos_path = output_dir / f"{trial.subject_id}_{trial.trial_id}_qpos_noised_{curve_idx:03d}.npy"
        np.save(qpos_path, qpos_matrix.astype(np.float32))
        viewer_qpos_files.append(str(qpos_path))
        if show_viewer:
            viewer_results.append(
                show_dry_run_mujoco_viewer(
                    trial_dir=Path(trial.meta["source_dataset_path"]),
                    time_vec=trial.time_for_pos,
                    motion_pos=curve["pos"],
                    source_name=f"synthetic_noised_{curve_idx:03d}",
                )
            )
    summary = {
        "subject_id": trial.subject_id,
        "trial_id": trial.trial_id,
        "activity": trial.activity,
        "source_dataset_path": trial.meta["source_dataset_path"],
        "patient_md_path": trial.meta.get("patient_md_path"),
        "num_noised_curves": int(len(corrupted_curves)),
        "plot_path": str(plot_path),
        "synthetic_residual_stats_path": str(synthetic_stats_path),
        "real_residual_stats_path": str(Path(config.data.output_path) / "residual_joint_stats.json") if real_residual_stats is not None else None,
        "applied_residual_scale_vector_path": str(residual_scale_vector_path) if residual_scale_vector_path is not None else None,
        "generated_residual_scale_vector_path": str(residual_scale_suggestion_path) if residual_scale_suggestion_path is not None else None,
        "applied_noise_scale": float(noise_scale),
        "residual_stats_comparison_plot_path": str(residual_comparison_plot_path) if residual_comparison_plot_path is not None else None,
        "synthetic_psd_stats_path": str(synthetic_psd_stats_path),
        "real_psd_stats_path": str(Path(config.data.output_path) / "residual_joint_psd.json") if real_psd_stats is not None else None,
        "psd_comparison_plot_path": str(psd_comparison_plot_path) if psd_comparison_plot_path is not None else None,
        "viewer_qpos_files": viewer_qpos_files,
        "viewer_results": viewer_results,
    }
    summary_path = output_dir / f"{trial.subject_id}_{trial.trial_id}_dry_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "plot_path": str(plot_path),
        "summary_path": str(summary_path),
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic OpenCap-like kinematics from mocap-only trials.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num_noised_curves", type=int, required=True)
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="Scalar multiplier applied to the generated noise residuals before adding them back to mocap. Use values below 1.0 to reduce the noise strength.",
    )
    parser.add_argument("--residual_scale_vector", type=str, default=None, help="Optional .npy file with per-DOF residual scaling factors applied before adding residuals back to mocap.")
    parser.add_argument("--dry_run", action="store_true", help="Randomly choose one trial, corrupt it, and save comparison plots instead of writing all trial outputs.")
    parser.add_argument("--dry_run_large", type=int, default=0, help="Randomly choose this many trials, generate num_noised_curves for each, and save aggregate residual comparison plots.")
    args = parser.parse_args()

    config = load_config(args.config)
    corruptor = FullCorruptor.load(args.model)
    fit_output_dir = Path(config.data.output_path)
    real_residual_stats_path = fit_output_dir / "residual_joint_stats.json"
    real_trialwise_stats_path = fit_output_dir / "residual_joint_trialwise_stats.json"
    real_psd_stats_path = fit_output_dir / "residual_joint_psd.json"
    real_residual_stats = None
    real_trialwise_stats = None
    real_psd_stats = None
    if real_residual_stats_path.exists():
        real_residual_stats = json.loads(real_residual_stats_path.read_text(encoding="utf-8"))
    if real_trialwise_stats_path.exists():
        real_trialwise_stats = json.loads(real_trialwise_stats_path.read_text(encoding="utf-8"))
    if real_psd_stats_path.exists():
        real_psd_stats = json.loads(real_psd_stats_path.read_text(encoding="utf-8"))
    output_dir = Path(config.data.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_scale = float(args.noise_scale)
    residual_scale_vector = None
    residual_scale_vector_path = None
    if args.residual_scale_vector:
        residual_scale_vector_path = Path(args.residual_scale_vector)
        residual_scale_vector = np.asarray(np.load(residual_scale_vector_path), dtype=np.float32)
    mocap_trials = [
        harmonize_mocap_trial(trial, sample_rate_hz=config.representation.sample_rate_hz)
        for trial in load_mocap_trials(config.data.mocap_only_path, metadata_filename=config.data.subject_metadata_filename)
    ]
    if args.dry_run:
        if not mocap_trials:
            raise RuntimeError("No mocap-only trials were found for dry run.")
        rng = np.random.default_rng(config.generation.random_seed)
        selected_idx = int(rng.integers(0, len(mocap_trials)))
        mocap_trials = [mocap_trials[selected_idx]]
    elif args.dry_run_large:
        if not mocap_trials:
            raise RuntimeError("No mocap-only trials were found for dry_run_large.")
        rng = np.random.default_rng(config.generation.random_seed)
        sample_count = min(int(args.dry_run_large), len(mocap_trials))
        selected_idx = rng.choice(len(mocap_trials), size=sample_count, replace=False)
        mocap_trials = [mocap_trials[int(idx)] for idx in selected_idx]
        visualize_count = min(3, len(mocap_trials))
        visualize_indices = set()
        if visualize_count > 0:
            visualize_indices = set(
                int(idx) for idx in rng.choice(len(mocap_trials), size=visualize_count, replace=False).tolist()
            )
        dry_run_large_dir = output_dir / "dry_run_large"
        dry_run_large_dir.mkdir(parents=True, exist_ok=True)

    generated_trial_dirs = []
    dry_run_outputs = []
    dry_run_large_trial_stats = []
    dry_run_large_residual_stack = []
    dry_run_large_visualized_outputs = []
    for trial_idx, trial in enumerate(mocap_trials):
        if residual_scale_vector is not None and residual_scale_vector.shape[0] != trial.pos.shape[1]:
            raise ValueError(
                f"Residual scale vector length {residual_scale_vector.shape[0]} does not match trial DOF width {trial.pos.shape[1]}"
            )
        phase_positions = None
        if trial.grf is not None:
            phase_positions = compute_stance_swing_phase_positions_from_grf(
                trial.grf,
                target_length=trial.pos.shape[0],
            )
        corrupted_curves = []
        plot_curves = []
        for sample_idx in range(args.num_noised_curves):
            seed = config.generation.random_seed + (trial_idx * 10_000) + sample_idx
            corrupted, aux = corruptor.sample(
                trial.pos,
                activity=trial.activity,
                meta={"height_m": trial.subject_metadata.height_m, "phase_positions": phase_positions},
                random_state=seed,
            )
            residual = (corrupted.astype(np.float32) - trial.pos.astype(np.float32)).astype(np.float32)
            if residual_scale_vector is not None:
                residual = (residual * residual_scale_vector[np.newaxis, :]).astype(np.float32)
            if noise_scale != 1.0:
                residual = (residual * noise_scale).astype(np.float32)
            corrupted = (trial.pos.astype(np.float32) + residual).astype(np.float32)
            filtered_pos = corrupted.astype(np.float32)
            filtered_vel = differentiate_signal(filtered_pos, trial.time_for_pos)
            filtered_accel = differentiate_signal(filtered_vel, trial.time_for_pos)
            corrupted_curves.append(
                {
                    "pos": filtered_pos,
                    "vel": filtered_vel,
                    "accel": filtered_accel,
                    "corruption_params": {"seed": seed, **aux},
                }
            )
            plot_curves.append(filtered_pos)

        if args.dry_run_large:
            dry_run_large_residual_stack.extend(
                [(curve.astype(np.float32) - trial.pos.astype(np.float32)).astype(np.float32) for curve in plot_curves]
            )
            dry_run_large_trial_stats.append(
                {
                    "subject_id": trial.subject_id,
                    "trial_id": trial.trial_id,
                    "trial_stats": compute_trialwise_residual_summary(
                        clean=trial.pos,
                        synthetic_curves=plot_curves,
                        dof_names=trial.subject_metadata.dof_names,
                    ),
                }
            )
            if trial_idx in visualize_indices:
                dry_run_large_visualized_outputs.append(
                    _save_trial_visualization_outputs(
                        config=config,
                        trial=trial,
                        corrupted_curves=corrupted_curves,
                        plot_curves=plot_curves,
                        real_residual_stats=real_residual_stats,
                        real_psd_stats=real_psd_stats,
                        output_dir=dry_run_large_dir / "visualized_trials" / f"{trial.subject_id}_{trial.trial_id}",
                        residual_scale_vector_path=residual_scale_vector_path,
                        residual_scale_vector=residual_scale_vector,
                        noise_scale=noise_scale,
                        show_viewer=False,
                    )
                )
            continue

        if args.dry_run:
            dry_run_dir = output_dir / "dry_run"
            dry_run_summary = _save_trial_visualization_outputs(
                config=config,
                trial=trial,
                corrupted_curves=corrupted_curves,
                plot_curves=plot_curves,
                real_residual_stats=real_residual_stats,
                real_psd_stats=real_psd_stats,
                output_dir=dry_run_dir,
                residual_scale_vector_path=residual_scale_vector_path,
                residual_scale_vector=residual_scale_vector,
                noise_scale=noise_scale,
                show_viewer=True,
            )
            dry_run_outputs.append(dry_run_summary)
            continue

        trial_output_dir = save_processeddata_outputs(
            trial_dir=trial.meta["source_dataset_path"],
            output_subdir_name=config.export.output_subdir_name,
            corrupted_curves=corrupted_curves,
            time=trial.time,
            time_for_pos=trial.time_for_pos,
            trial_metadata={
                "subject_id": trial.subject_id,
                "trial_id": trial.trial_id,
                "activity": trial.activity,
                "source_dataset_path": trial.meta["source_dataset_path"],
                "patient_md_path": trial.meta.get("patient_md_path"),
                "height_m": trial.subject_metadata.height_m,
                "mass_kg": trial.subject_metadata.mass_kg,
                "dof_names": trial.subject_metadata.dof_names,
                "applied_residual_scale_vector_path": str(residual_scale_vector_path) if residual_scale_vector_path is not None else None,
                "applied_noise_scale": float(noise_scale),
            },
        )
        generated_trial_dirs.append(str(trial_output_dir))

    summary = {
        "num_trials": len(mocap_trials),
        "num_noised_curves_per_trial": int(args.num_noised_curves),
        "dry_run": bool(args.dry_run),
        "dry_run_large": int(args.dry_run_large),
        "noise_scale": float(noise_scale),
        "output_subdir_name": config.export.output_subdir_name,
        "generated_trial_dirs": generated_trial_dirs,
        "dry_run_outputs": dry_run_outputs,
        "dry_run_large_visualized_outputs": dry_run_large_visualized_outputs,
    }
    if args.dry_run_large:
        if not mocap_trials:
            raise RuntimeError("No trials selected for dry_run_large.")
        dof_names = mocap_trials[0].subject_metadata.dof_names
        synthetic_trial_rows = []
        curve_counter = 0
        for item in dry_run_large_trial_stats:
            for curve_row in item["trial_stats"]["curve_stats"]:
                synthetic_trial_rows.append(
                    {
                        "trial_idx": int(curve_counter),
                        "subject_id": item["subject_id"],
                        "trial_id": item["trial_id"],
                        "curve_idx": int(curve_row["curve_idx"]),
                        "joint_stats": curve_row["joint_stats"],
                    }
                )
                curve_counter += 1
        synthetic_trialwise_stats = {
            "source": "synthetic_opencap_minus_mocap",
            "num_trials": int(len(synthetic_trial_rows)),
            "num_joints": int(len(dof_names)),
            "trial_stats": synthetic_trial_rows,
        }
        synthetic_trialwise_path = dry_run_large_dir / "synthetic_residual_joint_trialwise_stats.json"
        synthetic_trialwise_path.write_text(json.dumps(synthetic_trialwise_stats, indent=2), encoding="utf-8")
        synthetic_residual_points = np.concatenate(dry_run_large_residual_stack, axis=0) if dry_run_large_residual_stack else np.zeros((0, len(dof_names)), dtype=np.float32)
        synthetic_residual_stats = compute_per_joint_residual_stats_with_points(synthetic_residual_points, dof_names)
        synthetic_residual_stats["source"] = "synthetic_opencap_minus_mocap"
        synthetic_residual_stats["num_trials"] = int(len(synthetic_trial_rows))
        synthetic_residual_stats_path = dry_run_large_dir / "synthetic_residual_joint_stats.json"
        synthetic_residual_stats_path.write_text(json.dumps(synthetic_residual_stats, indent=2), encoding="utf-8")
        residual_scale_vector_path_large = None
        if real_residual_stats is not None:
            residual_scale_vector_large = compute_residual_scale_vector_from_stats(
                real_stats=real_residual_stats,
                synthetic_stats=synthetic_residual_stats,
                dof_names=dof_names,
            )
            residual_scale_vector_path_large = dry_run_large_dir / "residual_scale_vector.npy"
            np.save(residual_scale_vector_path_large, residual_scale_vector_large.astype(np.float32))
        boxplot_path = None
        if real_trialwise_stats is not None:
            boxplot_path = plot_residual_stats_boxplot_comparison(
                real_trialwise_stats=real_trialwise_stats,
                synthetic_trialwise_stats=synthetic_trialwise_stats,
                output_path=dry_run_large_dir / "residual_stats_boxplot_comparison.png",
            )
        summary["dry_run_large_outputs"] = {
            "selected_trials": [
                {"subject_id": trial.subject_id, "trial_id": trial.trial_id, "source_dataset_path": trial.meta["source_dataset_path"]}
                for trial in mocap_trials
            ],
            "real_trialwise_stats_path": str(real_trialwise_stats_path) if real_trialwise_stats is not None else None,
            "synthetic_trialwise_stats_path": str(synthetic_trialwise_path),
            "synthetic_residual_stats_path": str(synthetic_residual_stats_path),
            "residual_scale_vector_path": str(residual_scale_vector_path_large) if residual_scale_vector_path_large is not None else None,
            "residual_stats_boxplot_comparison_path": str(boxplot_path) if boxplot_path is not None else None,
            "noise_scale": float(noise_scale),
        }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
