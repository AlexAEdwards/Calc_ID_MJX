from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from corruption_model.preprocess.phase import PHASE_CYCLE_LENGTH, nearest_phase_window
from corruption_model.types import ResidualTrial


@dataclass
class NoiseModel:
    minimum_variance: float = 1e-6
    sample_scale: float = 1.0
    phase_window_frames: int = 50
    global_std: np.ndarray | None = None
    phase_std_lookup: Optional[np.ndarray] = None
    global_rho: np.ndarray | None = None

    def fit(
        self,
        residual_trials: list[ResidualTrial],
        deterministic_components: list[np.ndarray] | None = None,
        lowrank_components: list[np.ndarray] | None = None,
    ) -> "NoiseModel":
        all_noise = []
        all_positions = []
        for trial_idx, trial in enumerate(residual_trials):
            residual = trial.residual.copy()
            if deterministic_components is not None:
                deterministic = deterministic_components[trial_idx]
                residual = residual - deterministic[: residual.shape[0]]
            if lowrank_components is not None:
                lowrank = lowrank_components[trial_idx]
                residual = residual - lowrank[: residual.shape[0]]
            all_noise.append(residual)
            if trial.phase_positions is not None:
                all_positions.append(trial.phase_positions.astype(np.int32))
        stacked = np.concatenate(all_noise, axis=0)
        self.global_std = np.sqrt(np.maximum(np.var(stacked, axis=0), self.minimum_variance)).astype(np.float32)
        self.global_rho = self._fit_jointwise_rho(all_noise, residual_dim=stacked.shape[1])
        if all_positions:
            stacked_positions = np.concatenate(all_positions, axis=0).astype(np.int32)
            self.phase_std_lookup = np.zeros((PHASE_CYCLE_LENGTH, stacked.shape[1]), dtype=np.float32)
            for phase_idx in range(PHASE_CYCLE_LENGTH):
                nearest = nearest_phase_window(stacked, stacked_positions, center=phase_idx, window_frames=self.phase_window_frames)
                if nearest.size == 0:
                    self.phase_std_lookup[phase_idx] = self.global_std
                else:
                    self.phase_std_lookup[phase_idx] = np.sqrt(np.maximum(np.var(nearest, axis=0), self.minimum_variance)).astype(np.float32)
        else:
            self.phase_std_lookup = None
        return self

    def _fit_jointwise_rho(self, sequences: list[np.ndarray], residual_dim: int) -> np.ndarray:
        rho = np.zeros((residual_dim,), dtype=np.float32)
        for joint_idx in range(residual_dim):
            numerators = []
            denominators = []
            for seq in sequences:
                if seq.shape[0] < 2:
                    continue
                x_prev = seq[:-1, joint_idx]
                x_next = seq[1:, joint_idx]
                numerators.append(float(np.sum(x_prev * x_next)))
                denominators.append(float(np.sum(x_prev ** 2)))
            denom = float(np.sum(denominators))
            rho[joint_idx] = float(np.clip(np.sum(numerators) / denom, -0.995, 0.995)) if denom > 1e-8 else 0.0
        return rho

    def sample(self, phase_positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.global_std is None or self.global_rho is None:
            raise RuntimeError("NoiseModel must be fit before sample().")
        positions = np.asarray(phase_positions, dtype=np.int32)
        std = np.repeat(self.global_std[np.newaxis, :], positions.shape[0], axis=0).astype(np.float32)
        if self.phase_std_lookup is not None:
            clipped = np.clip(positions, 0, self.phase_std_lookup.shape[0] - 1)
            std = self.phase_std_lookup[clipped].astype(np.float32)
        std = std * float(self.sample_scale)
        noise = np.zeros_like(std, dtype=np.float32)
        initial_scale = std[0]
        noise[0] = rng.normal(loc=0.0, scale=initial_scale).astype(np.float32)
        for t in range(1, positions.shape[0]):
            innovation_std = std[t] * np.sqrt(np.maximum(1.0 - (self.global_rho ** 2), 1e-6))
            innovation = rng.normal(loc=0.0, scale=innovation_std).astype(np.float32)
            noise[t] = (self.global_rho * noise[t - 1]) + innovation
        return noise.astype(np.float32)
