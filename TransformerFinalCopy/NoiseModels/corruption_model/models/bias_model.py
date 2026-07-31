from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from corruption_model.preprocess.phase import PHASE_CYCLE_LENGTH, nearest_phase_window
from corruption_model.types import ResidualTrial


@dataclass
class BiasModel:
    phase_window_frames: int = 50
    global_bias: Optional[np.ndarray] = None
    phase_bias_lookup: Optional[np.ndarray] = None

    def fit(self, residual_trials: list[ResidualTrial]) -> "BiasModel":
        residuals = np.concatenate([trial.residual for trial in residual_trials], axis=0).astype(np.float32)
        self.global_bias = np.mean(residuals, axis=0).astype(np.float32)
        phase_values = []
        phase_positions = []
        for trial in residual_trials:
            if trial.phase_positions is None:
                continue
            phase_values.append(trial.residual.astype(np.float32))
            phase_positions.append(trial.phase_positions.astype(np.int32))
        if not phase_values:
            self.phase_bias_lookup = None
            return self
        stacked_values = np.concatenate(phase_values, axis=0).astype(np.float32)
        stacked_positions = np.concatenate(phase_positions, axis=0).astype(np.int32)
        lookup = np.zeros((PHASE_CYCLE_LENGTH, stacked_values.shape[1]), dtype=np.float32)
        for phase_idx in range(PHASE_CYCLE_LENGTH):
            nearest = nearest_phase_window(stacked_values, stacked_positions, center=phase_idx, window_frames=self.phase_window_frames)
            lookup[phase_idx] = np.mean(nearest, axis=0).astype(np.float32) if nearest.size else self.global_bias
        self.phase_bias_lookup = lookup.astype(np.float32)
        return self

    def predict(self, phase_positions: Optional[np.ndarray]) -> np.ndarray:
        if self.global_bias is None:
            raise RuntimeError("BiasModel must be fit before predict().")
        if phase_positions is None:
            return self.global_bias[np.newaxis, :]
        if self.phase_bias_lookup is None:
            return np.repeat(self.global_bias[np.newaxis, :], phase_positions.shape[0], axis=0).astype(np.float32)
        positions = np.clip(np.asarray(phase_positions, dtype=np.int32), 0, self.phase_bias_lookup.shape[0] - 1)
        return self.phase_bias_lookup[positions].astype(np.float32)
