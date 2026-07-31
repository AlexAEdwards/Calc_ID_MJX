import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from corruption_model.config import CorruptionConfig
from corruption_model.evaluation.metrics import (
    compute_average_curve_psd_stats,
    compute_average_curve_residual_stats,
    compute_per_joint_psd_stats,
    compute_per_joint_residual_stats,
    compute_per_joint_residual_stats_with_points,
    compute_real_trialwise_residual_summary,
    compute_residual_scale_vector_from_stats,
    compute_trialwise_residual_summary,
)
from corruption_model.evaluation.mujoco_viewer import map_motion_pos_to_qpos
from corruption_model.evaluation.plots import (
    plot_noised_curves_against_gt,
    plot_psd_comparison,
    plot_residual_stats_boxplot_comparison,
    plot_residual_stats_comparison,
)
from corruption_model.io.load_mocap_only import load_mocap_trials
from corruption_model.io.load_paired import load_paired_trials
from corruption_model.io.save_dataset import save_processeddata_outputs
from corruption_model.models.full_corruptor import FullCorruptor
from corruption_model.preprocess.filter import butter_lowpass_filter, differentiate_signal
from corruption_model.preprocess.harmonize import harmonize_mocap_trial, harmonize_trial_pair


DOF_NAMES = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]


def _write_patient_md(subject_dir: Path, patient_id: str) -> None:
    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / "Patient_MD.json").write_text(json.dumps({
        "Patient_ID": patient_id,
        "Height_m": 1.75,
        "Mass_kg": 70.0,
        "BiologicalSex": "male",
        "NumDOFs": 23,
        "DOF_names": DOF_NAMES,
        "SubjectTags": ["healthy"],
    }), encoding="utf-8")


class EndToEndTests(unittest.TestCase):
    def test_generation_pipeline_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paired_root = root / "paired"
            mocap_root = root / "mocap"
            subject_dir = paired_root / "subjectA"
            _write_patient_md(subject_dir, "subjectA")
            trial_dir = subject_dir / "Trial_1"
            (trial_dir / "Motion").mkdir(parents=True)
            (trial_dir / "MoCap" / "Untrimmed").mkdir(parents=True)
            T = 30
            time = np.arange(T, dtype=np.float32) / 100.0
            clean = np.zeros((T, 23), dtype=np.float32)
            target = clean + 0.05
            np.save(trial_dir / "Motion" / "Pos.npy", target)
            np.save(trial_dir / "Motion" / "Time.npy", time)
            np.save(trial_dir / "MoCap" / "Untrimmed" / "Pos.npy", clean)

            target_subject = mocap_root / "subjectB"
            _write_patient_md(target_subject, "subjectB")
            motion_dir = target_subject / "Trial_2" / "Motion"
            motion_dir.mkdir(parents=True)
            for name in ("Pos", "Vel", "Accel"):
                np.save(motion_dir / f"{name}.npy", np.zeros((T, 23), dtype=np.float32))
            np.save(motion_dir / "Time.npy", time)
            np.save(motion_dir / "Time_for_pos.npy", time)

            config = CorruptionConfig()
            paired_trials = [harmonize_trial_pair(trial, config.representation.sample_rate_hz) for trial in load_paired_trials(paired_root)]
            corruptor = FullCorruptor(config).fit(paired_trials)
            mocap_trial = harmonize_mocap_trial(load_mocap_trials(mocap_root)[0], config.representation.sample_rate_hz)
            corrupted, aux = corruptor.sample(mocap_trial.pos, meta={"height_m": mocap_trial.subject_metadata.height_m}, random_state=123)
            filtered_pos = butter_lowpass_filter(corrupted, cutoff_hz=6.0, fs_hz=config.representation.sample_rate_hz, order=4)
            filtered_vel = differentiate_signal(filtered_pos, mocap_trial.time_for_pos)
            filtered_accel = differentiate_signal(filtered_vel, mocap_trial.time_for_pos)
            out_dir = save_processeddata_outputs(
                trial_dir=target_subject / "Trial_2",
                output_subdir_name="ProcessedData_1",
                corrupted_curves=[
                    {
                        "pos": filtered_pos,
                        "vel": filtered_vel,
                        "accel": filtered_accel,
                        "corruption_params": aux,
                    }
                ],
                time=mocap_trial.time,
                time_for_pos=mocap_trial.time_for_pos,
                trial_metadata=mocap_trial.meta,
            )
            self.assertEqual(corrupted.shape, (T, 23))
            self.assertTrue((out_dir / "Pos_noised_001.npy").exists())
            self.assertTrue((out_dir / "Vel_noised_001.npy").exists())
            self.assertTrue((out_dir / "Accel_noised_001.npy").exists())

    def test_dry_run_plot_is_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean = np.zeros((20, 23), dtype=np.float32)
            synthetic_curves = [np.ones((20, 23), dtype=np.float32) * 0.1, np.ones((20, 23), dtype=np.float32) * 0.2]
            plot_path = plot_noised_curves_against_gt(
                clean=clean,
                synthetic_curves=synthetic_curves,
                dof_names=DOF_NAMES,
                output_path=root / "dry_run_plot.png",
                max_dofs=4,
            )
            self.assertTrue(plot_path.exists())

    def test_motion_pos_maps_to_mujoco_qpos_width(self):
        motion_pos = np.zeros((10, 23), dtype=np.float32)
        qpos = map_motion_pos_to_qpos(motion_pos, qpos_size=31)
        self.assertEqual(qpos.shape, (10, 31))

    def test_residual_stats_and_comparison_plot_are_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            residual = np.ones((20, 23), dtype=np.float32) * 0.1
            real_stats = compute_per_joint_residual_stats(residual, DOF_NAMES)
            synthetic_stats = compute_average_curve_residual_stats(
                clean=np.zeros((20, 23), dtype=np.float32),
                synthetic_curves=[np.ones((20, 23), dtype=np.float32) * 0.2, np.ones((20, 23), dtype=np.float32) * 0.3],
                dof_names=DOF_NAMES,
            )
            plot_path = plot_residual_stats_comparison(real_stats, synthetic_stats, root / "residual_stats_comparison.png")
            self.assertTrue(plot_path.exists())
            self.assertEqual(len(real_stats["joint_stats"]), 23)
            self.assertEqual(len(synthetic_stats["joint_stats"]), 23)

    def test_residual_stats_with_points_and_boxplot_are_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            residual_a = np.ones((20, 23), dtype=np.float32) * 0.1
            residual_b = np.ones((20, 23), dtype=np.float32) * 0.2
            real_stats = compute_per_joint_residual_stats_with_points(np.concatenate([residual_a, residual_b], axis=0), DOF_NAMES)
            self.assertIn("points", real_stats["joint_stats"][0])
            real_trialwise = compute_real_trialwise_residual_summary([residual_a, residual_b], DOF_NAMES)
            synthetic_trialwise = {
                "trial_stats": compute_trialwise_residual_summary(
                    clean=np.zeros((20, 23), dtype=np.float32),
                    synthetic_curves=[np.ones((20, 23), dtype=np.float32) * 0.15, np.ones((20, 23), dtype=np.float32) * 0.25],
                    dof_names=DOF_NAMES,
                )["curve_stats"]
            }
            plot_path = plot_residual_stats_boxplot_comparison(real_trialwise, synthetic_trialwise, root / "residual_stats_boxplot_comparison.png")
            self.assertTrue(plot_path.exists())

    def test_psd_stats_and_comparison_plot_are_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            t = np.arange(200, dtype=np.float32) / 100.0
            residual = np.stack([np.sin(2 * np.pi * 2.0 * t) for _ in range(23)], axis=1).astype(np.float32)
            real_psd = compute_per_joint_psd_stats(residual, DOF_NAMES, fs_hz=100.0)
            synthetic_psd = compute_average_curve_psd_stats(
                clean=np.zeros_like(residual),
                synthetic_curves=[residual * 0.8, residual * 1.1],
                dof_names=DOF_NAMES,
                fs_hz=100.0,
            )
            plot_path = plot_psd_comparison(real_psd, synthetic_psd, root / "psd_comparison.png", max_dofs=4)
            self.assertTrue(plot_path.exists())
            self.assertEqual(len(real_psd["joint_psd"]), 23)
            self.assertEqual(len(synthetic_psd["joint_psd"]), 23)

    def test_residual_scale_vector_pools_left_right_dofs(self):
        real_rows = []
        synthetic_rows = []
        for joint_idx, joint_name in enumerate(DOF_NAMES):
            if joint_name == "knee_angle_r":
                real_mean, real_std = 0.0, 2.0
                synthetic_mean, synthetic_std = 0.0, 1.0
            elif joint_name == "knee_angle_l":
                real_mean, real_std = 0.0, 4.0
                synthetic_mean, synthetic_std = 0.0, 2.0
            else:
                real_mean, real_std = 0.0, 1.0
                synthetic_mean, synthetic_std = 0.0, 1.0
            real_rows.append({"joint_idx": joint_idx, "joint_name": joint_name, "mean": real_mean, "std": real_std})
            synthetic_rows.append({"joint_idx": joint_idx, "joint_name": joint_name, "mean": synthetic_mean, "std": synthetic_std})
        scale = compute_residual_scale_vector_from_stats(
            real_stats={"joint_stats": real_rows},
            synthetic_stats={"joint_stats": synthetic_rows},
            dof_names=DOF_NAMES,
        )
        idx_r = DOF_NAMES.index("knee_angle_r")
        idx_l = DOF_NAMES.index("knee_angle_l")
        self.assertAlmostEqual(float(scale[idx_r]), 2.0, places=5)
        self.assertAlmostEqual(float(scale[idx_l]), 2.0, places=5)
        self.assertAlmostEqual(float(scale[DOF_NAMES.index("pelvis_tilt")]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
