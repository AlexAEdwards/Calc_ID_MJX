from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LagModel:
    mean_frames: float = 0.0
    std_frames: float = 0.0
    max_frames: int = 10
    sample_scale: float = 1.0

    def fit(self, lag_frames: list[int]) -> "LagModel":
        if lag_frames:
            arr = np.asarray(lag_frames, dtype=np.float32)
            self.mean_frames = float(np.mean(arr))
            self.std_frames = float(np.std(arr))
        return self

    def sample(self, rng: np.random.Generator) -> float:
        sampled_std = max(self.std_frames * float(self.sample_scale), 1e-6)
        sampled = rng.normal(self.mean_frames, sampled_std)
        return float(np.clip(sampled, -self.max_frames, self.max_frames))
