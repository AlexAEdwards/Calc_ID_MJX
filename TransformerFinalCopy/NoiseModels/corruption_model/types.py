from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SubjectMetadata:
    subject_id: str
    height_m: Optional[float]
    mass_kg: Optional[float]
    biological_sex: Optional[str]
    dof_names: List[str]
    num_dofs: int
    subject_tags: List[str] = field(default_factory=list)
    patient_md_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialPair:
    subject_metadata: SubjectMetadata
    trial_id: str
    activity: str
    time: np.ndarray
    q_mocap: np.ndarray
    q_opencap: np.ndarray
    grf: Optional[np.ndarray] = None
    contact_mask: Optional[np.ndarray] = None
    mask_valid: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def subject_id(self) -> str:
        return self.subject_metadata.subject_id


@dataclass
class ResidualTrial:
    subject_metadata: SubjectMetadata
    trial_id: str
    activity: str
    time: np.ndarray
    q_clean: np.ndarray
    q_target: np.ndarray
    residual: np.ndarray
    phase: Optional[np.ndarray] = None
    phase_positions: Optional[np.ndarray] = None
    phase_bins: Optional[np.ndarray] = None
    speed: Optional[np.ndarray] = None
    lag_frames: int = 0
    lag_seconds: float = 0.0
    alignment_score: float = 0.0
    mask_valid: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def subject_id(self) -> str:
        return self.subject_metadata.subject_id


@dataclass
class MocapTrial:
    subject_metadata: SubjectMetadata
    trial_id: str
    activity: str
    time: np.ndarray
    time_for_pos: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    accel: np.ndarray
    grf: Optional[np.ndarray] = None
    grm: Optional[np.ndarray] = None
    cop: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def subject_id(self) -> str:
        return self.subject_metadata.subject_id


@dataclass
class SyntheticSample:
    subject_metadata: SubjectMetadata
    subject_id: str
    trial_id: str
    activity: str
    time: np.ndarray
    time_for_pos: np.ndarray
    q_input_corrupted: np.ndarray
    q_clean_reference: np.ndarray
    vel_reference: np.ndarray
    accel_reference: np.ndarray
    corruption_params: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)
