import unittest

import numpy as np

from corruption_model.preprocess.align import shift_with_interpolation
from corruption_model.types import SubjectMetadata, TrialPair
from corruption_model.preprocess.align import estimate_global_lag


class AlignmentTests(unittest.TestCase):
    def test_shifted_sequence_recovers_lag(self):
        t = np.linspace(0.0, 1.0, 200, endpoint=False, dtype=np.float32)
        base = np.stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), np.sin(4 * np.pi * t), np.cos(4 * np.pi * t)], axis=1)
        padded = np.concatenate([base, np.zeros((200, 19), dtype=np.float32)], axis=1)
        shifted = shift_with_interpolation(padded, 4.0)
        trial = TrialPair(
            subject_metadata=SubjectMetadata("subject1", 1.75, 70.0, "male", [f"dof_{i}" for i in range(23)], 23),
            trial_id="Trial_1",
            activity="walking",
            time=t,
            q_mocap=padded,
            q_opencap=shifted,
        )
        trial.subject_metadata.dof_names[1] = "pelvis_ty"
        trial.subject_metadata.dof_names[0] = "hip_flexion_r"
        trial.subject_metadata.dof_names[2] = "knee_angle_r"
        result = estimate_global_lag(trial, sample_rate_hz=200.0, max_lag_frames=8)
        self.assertEqual(result.lag_frames, -4)


if __name__ == "__main__":
    unittest.main()
