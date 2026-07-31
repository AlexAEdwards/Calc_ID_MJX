import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from corruption_model.io.load_mocap_only import load_mocap_trials
from corruption_model.io.save_dataset import save_processeddata_outputs


DOF_NAMES = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]


class IoExportTests(unittest.TestCase):
    def test_processeddata_export_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subject_dir = root / "02"
            trial_motion = subject_dir / "Trial_1" / "Motion"
            trial_motion.mkdir(parents=True)
            patient_md = {
                "Patient_ID": "02",
                "Height_m": 1.83,
                "Mass_kg": 71.4,
                "BiologicalSex": "male",
                "NumDOFs": 23,
                "DOF_names": DOF_NAMES,
                "SubjectTags": ["healthy"],
            }
            (subject_dir / "Patient_MD.json").write_text(json.dumps(patient_md), encoding="utf-8")
            T = 12
            for name in ("Pos", "Vel", "Accel"):
                np.save(trial_motion / f"{name}.npy", np.zeros((T, 23), dtype=np.float32))
            np.save(trial_motion / "Time.npy", np.arange(T, dtype=np.float32) / 100.0)
            np.save(trial_motion / "Time_for_pos.npy", np.arange(T, dtype=np.float32) / 100.0)
            trials = load_mocap_trials(root)
            trial = trials[0]
            out_dir = save_processeddata_outputs(
                trial_dir=subject_dir / "Trial_1",
                output_subdir_name="ProcessedData_1",
                corrupted_curves=[
                    {
                        "pos": np.ones((T, 23), dtype=np.float32),
                        "vel": np.full((T, 23), 2.0, dtype=np.float32),
                        "accel": np.full((T, 23), 3.0, dtype=np.float32),
                        "corruption_params": {"seed": 5},
                    }
                ],
                time=trial.time,
                time_for_pos=trial.time_for_pos,
                trial_metadata={
                    "subject_id": trial.subject_id,
                    "trial_id": trial.trial_id,
                    "height_m": trial.subject_metadata.height_m,
                    "mass_kg": trial.subject_metadata.mass_kg,
                },
            )
            metadata = json.loads((out_dir / "corruption_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["trial_metadata"]["subject_id"], "02")
            self.assertEqual(metadata["trial_metadata"]["height_m"], 1.83)
            self.assertTrue((out_dir / "Pos_noised_001.npy").exists())
            self.assertTrue((out_dir / "Vel_noised_001.npy").exists())
            self.assertTrue((out_dir / "Accel_noised_001.npy").exists())


if __name__ == "__main__":
    unittest.main()
