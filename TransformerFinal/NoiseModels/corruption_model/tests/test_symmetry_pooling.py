import unittest

import numpy as np

from corruption_model.config import CorruptionConfig
from corruption_model.models.full_corruptor import FullCorruptor
from corruption_model.types import SubjectMetadata, TrialPair


DOF_NAMES = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]


class SymmetryPoolingTests(unittest.TestCase):
    def test_left_right_residual_pooling_shares_statistics(self):
        metadata = SubjectMetadata("subject1", 1.75, 70.0, "male", DOF_NAMES, 23)
        t = np.arange(40, dtype=np.float32) / 100.0
        q_mocap = np.zeros((40, 23), dtype=np.float32)
        q_opencap = np.zeros((40, 23), dtype=np.float32)
        q_opencap[:, DOF_NAMES.index("knee_angle_l")] = 0.3
        trial = TrialPair(metadata, "Trial_1", "walking", t, q_mocap, q_opencap)

        corruptor = FullCorruptor(CorruptionConfig()).fit([trial])
        left_idx = DOF_NAMES.index("knee_angle_l")
        right_idx = DOF_NAMES.index("knee_angle_r")
        self.assertAlmostEqual(float(corruptor.bias_model.global_bias[left_idx]), float(corruptor.bias_model.global_bias[right_idx]), places=6)
        self.assertGreater(float(corruptor.bias_model.global_bias[left_idx]), 0.0)
        self.assertEqual(corruptor.fit_summary_["num_fit_trials_after_left_right_pooling"], 2)


if __name__ == "__main__":
    unittest.main()
