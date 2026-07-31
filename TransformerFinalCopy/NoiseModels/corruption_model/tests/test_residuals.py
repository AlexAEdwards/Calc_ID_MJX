import unittest

import numpy as np

from corruption_model.residuals.compute_residuals import compute_residual_trial
from corruption_model.types import SubjectMetadata, TrialPair


class ResidualTests(unittest.TestCase):
    def test_residual_shape_and_values(self):
        q_mocap = np.ones((10, 23), dtype=np.float32)
        q_opencap = q_mocap + 2.0
        trial = TrialPair(
            subject_metadata=SubjectMetadata("subject1", 1.7, 65.0, "female", [f"dof_{i}" for i in range(23)], 23),
            trial_id="Trial_1",
            activity="walking",
            time=np.arange(10, dtype=np.float32) / 100.0,
            q_mocap=q_mocap,
            q_opencap=q_opencap,
        )
        residual_trial = compute_residual_trial(
            trial=trial,
            q_mocap_aligned=q_mocap,
            q_opencap_aligned=q_opencap,
            lag_frames=0,
            lag_seconds=0.0,
            alignment_score=1.0,
        )
        self.assertEqual(residual_trial.residual.shape, (10, 23))
        self.assertTrue(np.allclose(residual_trial.residual, 2.0))


if __name__ == "__main__":
    unittest.main()
