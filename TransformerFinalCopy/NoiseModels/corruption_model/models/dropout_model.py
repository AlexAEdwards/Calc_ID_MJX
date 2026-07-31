from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DropoutModel:
    enabled: bool = False
    probability: float = 0.0

    def fit(self, masks: list[np.ndarray] | None = None) -> "DropoutModel":
        if not self.enabled or not masks:
            self.probability = 0.0
            return self
        valid = np.concatenate([mask.astype(np.float32).reshape(-1) for mask in masks], axis=0)
        self.probability = float(np.mean(1.0 - valid))
        return self

    def apply(self, q: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        mask = np.ones_like(q, dtype=bool)
        if not self.enabled or self.probability <= 0.0:
            return q.astype(np.float32), mask
        dropout_mask = rng.random(q.shape) < self.probability
        out = q.copy().astype(np.float32)
        for col_idx in range(out.shape[1]):
            for row_idx in range(1, out.shape[0]):
                if dropout_mask[row_idx, col_idx]:
                    out[row_idx, col_idx] = out[row_idx - 1, col_idx]
                    mask[row_idx, col_idx] = False
        return out, mask
