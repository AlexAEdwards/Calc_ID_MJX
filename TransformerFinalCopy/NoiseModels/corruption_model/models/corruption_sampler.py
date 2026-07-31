from __future__ import annotations

import numpy as np


def make_rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(int(random_state))
