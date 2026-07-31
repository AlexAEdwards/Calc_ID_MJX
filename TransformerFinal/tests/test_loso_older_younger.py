from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
if str(TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_DIR))

from loso_dataset_utils import build_loso_folds, discover_trusted_trials, natural_key
from loso_inference_compare import ankle_power_dual_source, write_ankle_power_stance_report
from loso_reporting import build_loso_summary
from opensim_id_targets import OpenSimIDAlignmentError, load_aligned_opensim_id_target


class TrustedDatasetTests(unittest.TestCase):
    def _trial(self, root: Path, subject: str, trial: str, length: int = 40) -> None:
        processed = root / subject / trial / "ProcessedData"
        processed.mkdir(parents=True)
        np.save(processed / "pos_inputs.npy", np.zeros((length, 3), dtype=np.float32))

    def test_discovery_natural_sort_and_loso_has_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._trial(root, "Y10", "Trial_2")
            self._trial(root, "Y2", "Trial_10")
            self._trial(root, "OA1", "Trial_1")
            self._trial(root, "OA1", "Trial_short", length=10)
            discovery = discover_trusted_trials(root)
            self.assertEqual(discovery["subjects"], ["OA1", "Y2", "Y10"])
            self.assertEqual(len(discovery["trials"]), 3)
            self.assertEqual(len(discovery["skipped_trials"]), 1)
            folds = build_loso_folds(discovery["subject_to_trials"])
            for fold in folds:
                held_out = fold["held_out_subject"]
                self.assertNotIn(held_out, fold["train_subjects"])
                self.assertTrue(all(t["subject"] != held_out for t in fold["train_trials"]))
                self.assertTrue(all(t["subject"] == held_out for t in fold["held_out_trials"]))

    def test_subject_filter_and_trial_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subject in ("Y1", "Y2", "OA1"):
                self._trial(root, subject, "Trial_1")
                self._trial(root, subject, "Trial_2")
            discovery = discover_trusted_trials(
                root, include_subjects=["Y1", "OA1"], max_trials_per_subject=1
            )
            self.assertEqual(discovery["subjects"], ["OA1", "Y1"])
            self.assertEqual(discovery["trial_counts"], {"OA1": 1, "Y1": 1})

    def test_natural_key(self) -> None:
        self.assertEqual(sorted(["Y10", "Y2", "Y1"], key=natural_key), ["Y1", "Y2", "Y10"])


class OpenSimIDAlignmentTests(unittest.TestCase):
    _TARGET_COLUMNS = (
        "hip_flex_r_torque", "hip_add_r_torque", "hip_rot_r_torque",
        "knee_flex_r_torque", "ankle_flex_r_torque", "subt_angle_r_torque",
        "hip_flex_l_torque", "hip_add_l_torque", "hip_rot_l_torque",
        "knee_flex_l_torque", "ankle_flex_l_torque", "subt_angle_l_torque",
    )

    def _make_trial(self, root: Path, *, omit_column: str | None = None) -> Path:
        trial = root / "Y1" / "Trial_1"
        raw = trial / "Motion" / "Raw"
        processed = trial / "ProcessedData"
        raw.mkdir(parents=True)
        processed.mkdir(parents=True)
        # Motion has one terminal sample not present in the ID file.  The trim
        # metadata's pretrim length is authoritative and must prevent a one-frame shift.
        np.save(trial / "Motion" / "Time.npy", np.arange(11, dtype=np.float64) * 0.01)
        np.save(processed / "pos_inputs.npy", np.zeros((5, 3), dtype=np.float32))
        (processed / "Trial_Processing_Information.json").write_text(json.dumps({
            "core_trim_pretrim_n_frames": 10,
            "core_trim_bounds_motion_aligned": [2, 9],
            "ds_edge_trim_n_frames_before": 7,
            "ds_edge_trim_bounds": [1, 6],
        }), encoding="utf-8")
        columns = [name for name in self._TARGET_COLUMNS if name != omit_column]
        lines = ["name test invdyn", "endheader", "", "time\t" + "\t".join(columns)]
        for frame in range(10):
            values = [f"{frame * 0.01:.2f}"] + [str(frame + 100 * (idx + 1)) for idx in range(len(columns))]
            lines.append("\t".join(values))
        (raw / "testid.mot").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return trial

    def test_timestamp_alignment_applies_recorded_trims_without_length_stretching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = self._make_trial(Path(tmp))
            bundle = load_aligned_opensim_id_target(trial, target_len=5)
            # core [2:9], then DS [1:6] selects original frames [3:8].
            np.testing.assert_allclose(bundle["id"][:, 6], np.arange(3, 8) + 100)
            self.assertEqual(bundle["id"].shape, (5, 23))
            self.assertIn("ProcessData 100-Hz grid", bundle["alignment"])

    def test_missing_required_opensim_moment_fails_instead_of_using_mjx_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = self._make_trial(Path(tmp), omit_column="ankle_flex_l_torque")
            with self.assertRaisesRegex(OpenSimIDAlignmentError, "missing required"):
                load_aligned_opensim_id_target(trial, target_len=5)

    def test_target_length_mismatch_fails_instead_of_normalized_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial = self._make_trial(Path(tmp))
            with self.assertRaisesRegex(OpenSimIDAlignmentError, "did not reproduce"):
                load_aligned_opensim_id_target(trial, target_len=6)


class PairedOutputTests(unittest.TestCase):
    def test_zero_epoch_equivalent_ankle_power_sources_match(self) -> None:
        torque = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        omega = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
        result = ankle_power_dual_source(
            original_torque_nm=torque,
            fine_tuned_torque_nm=torque.copy(),
            ankle_angular_velocity_rad_s=omega,
            mass_kg=2.0,
        )
        self.assertEqual(result["pred_power_valid_w"], result["original_pred_power_valid_w"])
        self.assertEqual(result["original_pred_power_valid_w"], result["fine_tuned_pred_power_valid_w"])
        self.assertEqual(result["original_pred_power_101_w"], result["fine_tuned_pred_power_101_w"])

    def test_reporting_is_subject_weighted(self) -> None:
        metrics = [
            {"subject": "Y1", "trial": "Y1/Trial_1", "original": {"torque": {"mae": 1.0}}, "fine_tuned": {"torque": {"mae": 0.5}}},
            {"subject": "Y1", "trial": "Y1/Trial_2", "original": {"torque": {"mae": 3.0}}, "fine_tuned": {"torque": {"mae": 1.5}}},
            {"subject": "OA1", "trial": "OA1/Trial_1", "original": {"torque": {"mae": 10.0}}, "fine_tuned": {"torque": {"mae": 8.0}}},
        ]
        summary = build_loso_summary(metrics)
        by_subject = {row["subject"]: row for row in summary["subjects"]}
        self.assertAlmostEqual(by_subject["Y1"]["original_torque_mae"], 2.0)
        self.assertAlmostEqual(by_subject["Y1"]["fine_tuned_torque_mae"], 1.0)

    def test_stance_report_contains_original_fine_and_legacy_power(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "Y1" / "Trial_1" / "ProcessedData"
            processed.mkdir(parents=True)
            n = 8
            qvel = np.ones((n, 23), dtype=np.float32)
            contact = np.zeros((n, 2), dtype=np.float32)
            contact[1:7, :] = 1.0
            np.save(processed / "qvel_mjx.npy", qvel)
            np.save(processed / "contactBoolean.npy", contact)
            np.save(processed / "Mass_kg.npy", np.asarray([70.0], dtype=np.float32))
            original = np.ones((n, 14), dtype=np.float32)
            fine = original * 2.0
            target = original * 3.0
            path = write_ankle_power_stance_report(
                trial={
                    "subject": "Y1",
                    "trial": "Trial_1",
                    "trial_root": str(processed.parent),
                    "training_data_path": str(processed),
                },
                comparison={
                    "arrays": {
                        "original_torque_nm": original,
                        "fine_tuned_torque_nm": fine,
                        "target_torque_nm": target,
                    },
                    "evaluation_mask": np.ones(n, dtype=bool),
                },
                model_structure="direct_torque",
                output_root=root / "results",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            ankle = payload["sides"]["right"]["complete_stances"][0]["ankle_power"]
            self.assertEqual(ankle["pred_power_valid_w"], ankle["original_pred_power_valid_w"])
            self.assertNotEqual(ankle["original_pred_power_valid_w"], ankle["fine_tuned_pred_power_valid_w"])
            self.assertIn("fine_tuned_pred_peak_w", ankle["summary"])

    def test_stance_report_maps_full_width_qvel_to_independent_dofs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "Y1" / "Trial_1" / "ProcessedData"
            processed.mkdir(parents=True)
            n = 8
            qvel = np.zeros((n, 31), dtype=np.float32)
            qvel[:, 14] = 2.0  # independent right ankle velocity index 10
            qvel[:, 25] = 4.0  # independent left ankle velocity index 17
            contact = np.zeros((n, 2), dtype=np.float32)
            contact[1:7, :] = 1.0
            np.save(processed / "qvel_mjx.npy", qvel)
            np.save(processed / "contactBoolean.npy", contact)
            np.save(processed / "Mass_kg.npy", np.asarray([70.0], dtype=np.float32))
            torque = np.ones((n, 14), dtype=np.float32)
            path = write_ankle_power_stance_report(
                trial={
                    "subject": "Y1",
                    "trial": "Trial_1",
                    "trial_root": str(processed.parent),
                    "training_data_path": str(processed),
                },
                comparison={
                    "arrays": {
                        "original_torque_nm": torque,
                        "fine_tuned_torque_nm": torque,
                        "target_torque_nm": torque,
                    },
                    "evaluation_mask": np.ones(n, dtype=bool),
                },
                model_structure="direct_torque",
                output_root=root / "results",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            right = payload["sides"]["right"]["complete_stances"][0]["ankle_power"]
            left = payload["sides"]["left"]["complete_stances"][0]["ankle_power"]
            self.assertEqual(right["gt_power_valid_w"], [2.0] * 6)
            self.assertEqual(left["gt_power_valid_w"], [4.0] * 6)

    def test_physics_stance_report_reconstructs_net_joint_moment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "Y1" / "Trial_1" / "ProcessedData"
            processed.mkdir(parents=True)
            n = 8
            qvel = np.zeros((n, 31), dtype=np.float32)
            qvel[:, 14] = 2.0
            qvel[:, 25] = 4.0
            qfrc = np.zeros((n, 31), dtype=np.float32)
            qfrc[:, 14] = 10.0
            qfrc[:, 25] = 20.0
            id_gt = np.zeros((n, 31), dtype=np.float32)
            id_gt[:, 14] = 8.0
            id_gt[:, 25] = 16.0
            contact = np.zeros((n, 2), dtype=np.float32)
            contact[1:7, :] = 1.0
            np.save(processed / "qvel_mjx.npy", qvel)
            np.save(processed / "qfrc_inverse.npy", qfrc)
            np.save(processed / "ID_GT_MJX.npy", id_gt)
            np.save(processed / "contactBoolean.npy", contact)
            np.save(processed / "Mass_kg.npy", np.asarray([70.0], dtype=np.float32))
            original_tau_grf = np.full((n, 23), 3.0, dtype=np.float32)
            fine_tau_grf = np.full((n, 23), 2.0, dtype=np.float32)
            target_tau_grf = np.full((n, 23), 2.0, dtype=np.float32)
            path = write_ankle_power_stance_report(
                trial={
                    "subject": "Y1",
                    "trial": "Trial_1",
                    "trial_root": str(processed.parent),
                    "training_data_path": str(processed),
                },
                comparison={
                    "arrays": {
                        "original_torque_nm": original_tau_grf,
                        "fine_tuned_torque_nm": fine_tau_grf,
                        "target_torque_nm": target_tau_grf,
                    },
                    "evaluation_mask": np.ones(n, dtype=bool),
                },
                model_structure="cop_grf_moments",
                output_root=root / "results",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            right = payload["sides"]["right"]["complete_stances"][0]["ankle_power"]
            left = payload["sides"]["left"]["complete_stances"][0]["ankle_power"]
            self.assertEqual(right["original_pred_power_valid_w"], [14.0] * 6)
            self.assertEqual(right["fine_tuned_pred_power_valid_w"], [16.0] * 6)
            self.assertEqual(right["gt_power_valid_w"], [16.0] * 6)
            self.assertEqual(left["original_pred_power_valid_w"], [68.0] * 6)
            self.assertEqual(left["gt_power_valid_w"], [64.0] * 6)
            self.assertIn("minus_tau_grf", payload["torque_source"])


if __name__ == "__main__":
    unittest.main()
