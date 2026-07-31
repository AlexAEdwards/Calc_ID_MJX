from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from corruption_model.preprocess.phase import PHASE_CYCLE_LENGTH, nearest_phase_window
from corruption_model.types import ResidualTrial


@dataclass
class PhaseResidualModel:
    enabled: bool = True
    sample_scale: float = 0.35
    gain_std: float = 0.1
    phase_window_frames: int = 50
    phase_mean_lookup: Optional[np.ndarray] = None
    phase_std_lookup: Optional[np.ndarray] = None
    global_phase_mean: Optional[np.ndarray] = None

    def fit(self, residual_trials: list[ResidualTrial], reference_sequences: list[np.ndarray]) -> "PhaseResidualModel":
        if not self.enabled:
            self.phase_mean_lookup = None
            self.phase_std_lookup = None
            self.global_phase_mean = None
            return self
        all_centered = []
        all_positions = []
        for trial, reference in zip(residual_trials, reference_sequences):
            centered = (trial.residual - reference).astype(np.float32)
            all_centered.append(centered)
            if trial.phase_positions is None:
                continue
            all_positions.append(trial.phase_positions.astype(np.int32))
        if all_centered:
            self.global_phase_mean = np.mean(np.concatenate(all_centered, axis=0), axis=0).astype(np.float32)
        else:
            self.global_phase_mean = None
        if not all_centered or not all_positions:
            self.phase_mean_lookup = None
            self.phase_std_lookup = None
            return self
        stacked_centered = np.concatenate(all_centered, axis=0).astype(np.float32)
        stacked_positions = np.concatenate(all_positions, axis=0).astype(np.int32)
        self.phase_mean_lookup = np.zeros((PHASE_CYCLE_LENGTH, stacked_centered.shape[1]), dtype=np.float32)
        self.phase_std_lookup = np.zeros((PHASE_CYCLE_LENGTH, stacked_centered.shape[1]), dtype=np.float32)
        for phase_idx in range(PHASE_CYCLE_LENGTH):
            nearest = nearest_phase_window(stacked_centered, stacked_positions, center=phase_idx, window_frames=self.phase_window_frames)
            if nearest.size == 0:
                self.phase_mean_lookup[phase_idx] = self.global_phase_mean if self.global_phase_mean is not None else 0.0
                self.phase_std_lookup[phase_idx] = 1e-3
                continue
            self.phase_mean_lookup[phase_idx] = np.mean(nearest, axis=0).astype(np.float32)
            self.phase_std_lookup[phase_idx] = np.sqrt(np.maximum(np.var(nearest, axis=0), 1e-6)).astype(np.float32)
        return self

    def sample(self, phase_positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or self.phase_mean_lookup is None or self.phase_std_lookup is None:
            return np.zeros((phase_positions.shape[0], 0), dtype=np.float32) if phase_positions.ndim == 1 else np.zeros_like(phase_positions, dtype=np.float32)
        positions = np.clip(np.asarray(phase_positions, dtype=np.int32), 0, self.phase_mean_lookup.shape[0] - 1)
        dof = self.phase_mean_lookup.shape[1]
        out = np.zeros((positions.shape[0], dof), dtype=np.float32)
        gain = float(np.clip(rng.normal(1.0, self.gain_std), 0.0, 2.0))
        for t, phase_idx in enumerate(positions):
            mean = self.phase_mean_lookup[int(phase_idx)]
            std = self.phase_std_lookup[int(phase_idx)]
            out[t] = gain * (mean + (self.sample_scale * rng.normal(0.0, std).astype(np.float32)))
        return out.astype(np.float32)
