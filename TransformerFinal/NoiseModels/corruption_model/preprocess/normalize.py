from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_by_height(q: np.ndarray, height_m: Optional[float]) -> np.ndarray:
    if height_m is None or height_m <= 0.0:
        return q.astype(np.float32)
    return (q / float(height_m)).astype(np.float32)


def denormalize_by_height(q: np.ndarray, height_m: Optional[float]) -> np.ndarray:
    if height_m is None or height_m <= 0.0:
        return q.astype(np.float32)
    return (q * float(height_m)).astype(np.float32)
