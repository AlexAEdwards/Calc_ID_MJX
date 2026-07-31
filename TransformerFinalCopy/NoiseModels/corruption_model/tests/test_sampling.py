import unittest

import numpy as np

from corruption_model.config import CorruptionConfig
from corruption_model.models.full_corruptor import FullCorruptor
from corruption_model.types import SubjectMetadata, TrialPair


class SamplingTests(unittest.TestCase):
    def test_seed_reproducibility(self):
        metadata = SubjectMetadata("subject1", 1.75, 70.0, "male", [f"dof_{i}" for i in range(23)], 23)
        t = np.arange(50, dtype=np.float32) / 100.0
        q_mocap = np.zeros((50, 23), dtype=np.float32)
        q_opencap = np.ones((50, 23), dtype=np.float32) * 0.1
        trial = TrialPair(metadata, "Trial_1", "walking", t, q_mocap, q_opencap)
        corruptor = FullCorruptor(CorruptionConfig()).fit([trial])
        a, _ = corruptor.sample(q_mocap, meta={"height_m": metadata.height_m}, random_state=123)
        b, _ = corruptor.sample(q_mocap, meta={"height_m": metadata.height_m}, random_state=123)
        c, _ = corruptor.sample(q_mocap, meta={"height_m": metadata.height_m}, random_state=124)
        self.assertTrue(np.allclose(a, b))
        self.assertFalse(np.allclose(a, c))
        self.assertFalse(np.isnan(a).any())

    def test_all_disabled_components_leave_signal_unchanged(self):
        config = CorruptionConfig()
        config.model.use_phase_conditioning = False
        config.model.use_phase_residual = False
        config.model.use_lowrank = False
        config.model.use_smoothing = False
        config.model.use_lag = False
        config.model.use_dropout = False
        config.model.phase_residual_sample_scale = 0.0
        config.model.lowrank_sample_scale = 0.0
        config.model.noise_sample_scale = 0.0
        config.model.lag_std_scale = 0.0
        metadata = SubjectMetadata("subject1", 1.75, 70.0, "male", [f"dof_{i}" for i in range(23)], 23)
        t = np.arange(50, dtype=np.float32) / 100.0
        q_mocap = np.linspace(0.0, 1.0, 50, dtype=np.float32)[:, np.newaxis] * np.ones((1, 23), dtype=np.float32)
        q_opencap = q_mocap.copy()
        trial = TrialPair(metadata, "Trial_1", "walking", t, q_mocap, q_opencap)
        corruptor = FullCorruptor(config).fit([trial])
        sample, aux = corruptor.sample(q_mocap, meta={"height_m": metadata.height_m}, random_state=7)
        self.assertTrue(np.allclose(sample, q_mocap))
        self.assertEqual(float(aux["sampled_lag"]), 0.0)
        self.assertFalse(aux["phase_residual_used"])

    def test_phase_residual_influences_sampling(self):
        config = CorruptionConfig()
        config.model.use_phase_residual = True
        config.model.phase_window_frames = 1
        config.model.phase_residual_sample_scale = 0.0
        config.model.lowrank_sample_scale = 0.0
        config.model.noise_sample_scale = 0.0
        config.model.lag_std_scale = 0.0
        config.model.lag_max_frames = 0
        metadata = SubjectMetadata("subject1", 1.75, 70.0, "male", [f"dof_{i}" for i in range(23)], 23)
        t = np.arange(40, dtype=np.float32) / 100.0
        q_mocap = np.zeros((40, 23), dtype=np.float32)
        q_opencap = np.zeros((40, 23), dtype=np.float32)
        q_opencap[:20, 0] = 0.1
        q_opencap[20:, 0] = 0.2
        trial = TrialPair(metadata, "Trial_1", "walking", t, q_mocap, q_opencap)
        corruptor = FullCorruptor(config).fit([trial])
        phase_positions = np.concatenate(
            [np.full((20,), 25, dtype=np.int32), np.full((20,), 125, dtype=np.int32)],
            axis=0,
        )
        sample, aux = corruptor.sample(q_mocap, meta={"height_m": metadata.height_m, "phase_positions": phase_positions}, random_state=7)
        self.assertTrue(aux["phase_residual_used"])
        self.assertGreater(float(np.mean(sample[20:, 0])), float(np.mean(sample[:20, 0])))


if __name__ == "__main__":
    unittest.main()
