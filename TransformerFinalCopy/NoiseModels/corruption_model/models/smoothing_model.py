from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corruption_model.preprocess.filter import butter_lowpass_filter
from corruption_model.types import ResidualTrial


@dataclass
class SmoothingModel:
    fs_hz: float
    filter_order: int = 4
    default_cutoff_hz: float = 6.0
    sample_std_scale: float = 1.0
    input_cutoff_mean_hz: float = 6.0
    input_cutoff_std_hz: float = 0.5

    def fit(self, residual_trials: list[ResidualTrial]) -> "SmoothingModel":
        # Keep the model-level cutoff as the baseline; no adaptive fitting from training data.
        self.input_cutoff_mean_hz = float(self.default_cutoff_hz)
        self.input_cutoff_std_hz = 0.0
        return self

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        input_std = max(self.input_cutoff_std_hz * float(self.sample_std_scale), 0.0)
        if input_std > 0.0:
            input_cutoff_hz = float(np.clip(rng.normal(self.input_cutoff_mean_hz, input_std), 0.5, self.fs_hz * 0.45))
        else:
            input_cutoff_hz = float(np.clip(self.input_cutoff_mean_hz, 0.5, self.fs_hz * 0.45))
        return {
            "input_cutoff_hz": input_cutoff_hz,
            "filter_order": int(self.filter_order),
        }

    def apply_input(self, q_clean: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
        params = self.sample_params(rng)
        filtered = butter_lowpass_filter(q_clean, cutoff_hz=params["input_cutoff_hz"], fs_hz=self.fs_hz, order=params["filter_order"])
        return filtered, params
