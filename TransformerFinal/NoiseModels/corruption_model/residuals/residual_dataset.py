from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from corruption_model.types import ResidualTrial


@dataclass
class ResidualDataset:
    trials: List[ResidualTrial]

    def by_subject(self) -> Dict[str, List[ResidualTrial]]:
        grouped: Dict[str, List[ResidualTrial]] = defaultdict(list)
        for trial in self.trials:
            grouped[trial.subject_id].append(trial)
        return dict(grouped)

    def residual_matrix(self) -> np.ndarray:
        if not self.trials:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate([trial.residual for trial in self.trials], axis=0).astype(np.float32)

    def phase_bins(self) -> np.ndarray:
        if not self.trials:
            return np.zeros((0,), dtype=np.int32)
        return np.concatenate([trial.phase_bins for trial in self.trials], axis=0).astype(np.int32)

    def group_by_phase_bin(self) -> Dict[int, np.ndarray]:
        grouped: Dict[int, List[np.ndarray]] = defaultdict(list)
        for trial in self.trials:
            if trial.phase_bins is None:
                continue
            for phase_bin in np.unique(trial.phase_bins):
                grouped[int(phase_bin)].append(trial.residual[trial.phase_bins == phase_bin])
        return {key: np.concatenate(value, axis=0) for key, value in grouped.items()}
