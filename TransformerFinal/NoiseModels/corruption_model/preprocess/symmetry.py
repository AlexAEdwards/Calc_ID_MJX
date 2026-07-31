from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

import numpy as np

from corruption_model.types import ResidualTrial


def build_left_right_index_map(dof_names: Sequence[str]) -> Dict[int, int]:
    name_to_idx = {name: idx for idx, name in enumerate(dof_names)}
    mapping: Dict[int, int] = {}
    for idx, name in enumerate(dof_names):
        if name.endswith("_r"):
            partner_name = f"{name[:-2]}_l"
        elif name.endswith("_l"):
            partner_name = f"{name[:-2]}_r"
        else:
            continue
        partner_idx = name_to_idx.get(partner_name)
        if partner_idx is not None:
            mapping[idx] = partner_idx
    return mapping


def swap_left_right_columns(array: np.ndarray, index_map: Dict[int, int]) -> np.ndarray:
    swapped = np.asarray(array, dtype=np.float32).copy()
    visited: set[int] = set()
    for idx, partner_idx in index_map.items():
        if idx in visited or partner_idx in visited:
            continue
        swapped[:, idx] = array[:, partner_idx]
        swapped[:, partner_idx] = array[:, idx]
        visited.add(idx)
        visited.add(partner_idx)
    return swapped.astype(np.float32)


def make_mirrored_residual_trial(trial: ResidualTrial) -> ResidualTrial:
    index_map = build_left_right_index_map(trial.subject_metadata.dof_names)
    if not index_map:
        return trial
    mirrored_meta = dict(trial.meta)
    mirrored_meta["mirrored_left_right"] = True
    mirrored_meta["mirrored_from_trial_id"] = trial.trial_id
    return replace(
        trial,
        q_clean=swap_left_right_columns(trial.q_clean, index_map),
        q_target=swap_left_right_columns(trial.q_target, index_map),
        residual=swap_left_right_columns(trial.residual, index_map),
        mask_valid=swap_left_right_columns(trial.mask_valid.astype(np.float32), index_map).astype(bool) if trial.mask_valid is not None else None,
        meta=mirrored_meta,
    )


def augment_with_left_right_mirrors(residual_trials: List[ResidualTrial]) -> List[ResidualTrial]:
    augmented = list(residual_trials)
    for trial in residual_trials:
        mirrored = make_mirrored_residual_trial(trial)
        if mirrored is not trial:
            augmented.append(mirrored)
    return augmented
