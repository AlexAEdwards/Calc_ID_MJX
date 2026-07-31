import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from corruption_model.io.load_mocap_only import load_mocap_trials


DOF_NAMES = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]


class ShapeTests(unittest.TestCase):
    def test_patient_md_drives_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subject_dir = root / "Subject1"
            trial_dir = subject_dir / "Trial_1" / "Motion"
            trial_dir.mkdir(parents=True)
            (subject_dir / "Patient_MD.json").write_text(json.dumps({
                "Patient_ID": "Subject1",
                "Height_m": 1.8,
                "Mass_kg": 75.0,
                "BiologicalSex": "male",
                "NumDOFs": 23,
                "DOF_names": DOF_NAMES,
                "SubjectTags": ["healthy"],
            }), encoding="utf-8")
            T = 20
            np.save(trial_dir / "Pos.npy", np.zeros((T, 23), dtype=np.float32))
            np.save(trial_dir / "Vel.npy", np.zeros((T, 23), dtype=np.float32))
            np.save(trial_dir / "Accel.npy", np.zeros((T, 23), dtype=np.float32))
            np.save(trial_dir / "Time.npy", np.arange(T, dtype=np.float32) / 100.0)
            np.save(trial_dir / "Time_for_pos.npy", np.arange(T, dtype=np.float32) / 100.0)
            trials = load_mocap_trials(root)
            self.assertEqual(len(trials), 1)
            self.assertEqual(trials[0].subject_metadata.height_m, 1.8)
            self.assertEqual(trials[0].pos.shape, (T, 23))


if __name__ == "__main__":
    unittest.main()
