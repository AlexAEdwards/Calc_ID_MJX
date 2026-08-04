#!/usr/bin/env python3
"""Stance detection and ankle-power helpers shared by validation scripts."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from paths import REPO_ROOT
INFER_PY = REPO_ROOT / "TransformerFinal" / "infer.py"

ANKLE_DOF_IDX = {"right": 14, "left": 25}
GRAVITY = 9.8067
DEFAULT_STANCE_LOW_N = 5.0
DEFAULT_STANCE_CORE_BW_RATIO = 0.20
DEFAULT_EDGE_EXCLUDE_FRAMES = 30


def _load_infer_helpers() -> Dict[str, Any]:
    src = INFER_PY.read_text()
    tree = ast.parse(src)
    want = {
        "_load_json_dict",
        "_apply_ds_edge_trim_if_needed",
        "_load_opensim_sto",
        "_find_raw_opensim_id_mot_file",
        "_align_motion_raw_series_to_processed_frames",
    }
    ns: Dict[str, Any] = {
        "np": np, "pd": pd, "json": json, "Path": Path,
        "Any": Any, "Dict": Dict, "List": List, "Mapping": Mapping,
        "Optional": Optional, "Tuple": Tuple,
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(INFER_PY), "exec"), ns)
    missing = want - set(ns)
    if missing:
        raise RuntimeError(f"Failed to extract infer.py helpers: {missing}")
    return ns


_H = _load_infer_helpers()
_find_raw_opensim_id_mot_file = _H["_find_raw_opensim_id_mot_file"]
_load_opensim_sto = _H["_load_opensim_sto"]
_align_motion_raw_series_to_processed_frames = _H["_align_motion_raw_series_to_processed_frames"]


def _stance_phases_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    d = np.diff(mask.astype(int), prepend=0)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if len(ends) < len(starts):
        ends = np.append(ends, len(mask))
    if len(starts) > len(ends):
        starts = starts[: len(ends)]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s > 5]


def dual_threshold_stance_intervals(
    grf_raw: np.ndarray,
    side: str,
    body_weight_n: float,
    low_threshold_n: float = DEFAULT_STANCE_LOW_N,
    core_bw_ratio: float = DEFAULT_STANCE_CORE_BW_RATIO,
) -> List[Tuple[int, int]]:
    grf_raw = np.asarray(grf_raw)
    n = len(grf_raw)
    vidx = 2 if str(side).strip().lower() == "right" else 5
    vf = np.abs(grf_raw[:, vidx].astype(np.float64))
    low_mask = vf > float(low_threshold_n)
    high_mask = vf > float(core_bw_ratio) * float(body_weight_n)
    expanded: List[Tuple[int, int]] = []
    for cs, ce in _stance_phases_from_mask(high_mask):
        s = cs
        while s > 0 and low_mask[s - 1]:
            s -= 1
        e = ce
        while e < n and low_mask[e]:
            e += 1
        expanded.append((s, e))
    expanded.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in expanded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def is_edge_peak(peak_frame: int, trial_len: int,
                 edge_exclude_frames: int = DEFAULT_EDGE_EXCLUDE_FRAMES) -> bool:
    return (peak_frame < edge_exclude_frames) or (peak_frame >= trial_len - edge_exclude_frames)


def stance_peak(power: np.ndarray, s: int, e: int,
                last_half: bool = False) -> Tuple[float, int]:
    start = s + (e - s) // 2 if last_half else s
    seg = np.asarray(power[start:e], dtype=np.float64)
    if seg.size == 0 or not np.any(np.isfinite(seg)):
        return float("nan"), -1
    off = int(np.nanargmax(seg))
    return float(seg[off]), start + off
