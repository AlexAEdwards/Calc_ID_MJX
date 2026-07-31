from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from corruption_model.io.load_paired import CANONICAL_DOF_NAMES, EXPECTED_NUM_DOFS
from corruption_model.types import MocapTrial, SubjectMetadata


def _load_subject_metadata(subject_dir: Path, metadata_filename: str) -> SubjectMetadata:
    patient_md_path = subject_dir / metadata_filename
    if not patient_md_path.exists():
        raise FileNotFoundError(f"Missing subject metadata file: {patient_md_path}")
    payload = json.loads(patient_md_path.read_text(encoding="utf-8"))
    dof_names = list(payload.get("DOF_names", []))
    num_dofs = int(payload.get("NumDOFs", len(dof_names)))
    if not dof_names and num_dofs in {0, EXPECTED_NUM_DOFS}:
        dof_names = list(CANONICAL_DOF_NAMES)
        num_dofs = EXPECTED_NUM_DOFS
    return SubjectMetadata(
        subject_id=str(payload.get("Patient_ID", subject_dir.name)),
        height_m=float(payload["Height_m"]) if payload.get("Height_m") is not None else None,
        mass_kg=float(payload["Mass_kg"]) if payload.get("Mass_kg") is not None else None,
        biological_sex=payload.get("BiologicalSex"),
        dof_names=dof_names,
        num_dofs=num_dofs,
        subject_tags=list(payload.get("SubjectTags", [])),
        patient_md_path=patient_md_path,
        extra={k: v for k, v in payload.items() if k not in {"Patient_ID", "Height_m", "Mass_kg", "BiologicalSex", "DOF_names", "NumDOFs", "SubjectTags"}},
    )


def _required_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    try:
        try:
            loaded = np.load(path, allow_pickle=False)
        except ValueError as exc:
            if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
                raise
            loaded = np.load(path, allow_pickle=True)
        return np.asarray(loaded, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Could not load numeric array from {path}: {exc}") from exc


def _load_time_for_pos(motion_dir: Path) -> np.ndarray:
    time_for_pos_path = motion_dir / "Time_for_pos.npy"
    if time_for_pos_path.exists():
        return np.load(time_for_pos_path).astype(np.float32)
    return _required_array(motion_dir / "Time.npy")


def iter_mocap_trials(root: str | Path, metadata_filename: str = "Patient_MD.json") -> Iterable[MocapTrial]:
    root_path = Path(root)
    for subject_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        try:
            metadata = _load_subject_metadata(subject_dir, metadata_filename)
        except Exception:
            continue
        if metadata.num_dofs != EXPECTED_NUM_DOFS or len(metadata.dof_names) != EXPECTED_NUM_DOFS:
            continue
        for trial_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir() and path.name.startswith("Trial_")):
            motion_dir = trial_dir / "Motion"
            try:
                pos = _required_array(motion_dir / "Pos.npy")
                vel = _required_array(motion_dir / "Vel.npy")
                accel = _required_array(motion_dir / "Accel.npy")
                time = _required_array(motion_dir / "Time.npy")
                time_for_pos = _load_time_for_pos(motion_dir)
            except (FileNotFoundError, ValueError):
                continue
            if pos.ndim != 2 or pos.shape[1] != metadata.num_dofs:
                continue
            if vel.ndim != 2 or accel.ndim != 2:
                continue
            if pos.shape[0] != time_for_pos.shape[0]:
                continue
            grf = np.load(motion_dir / "GRF.npy").astype(np.float32) if (motion_dir / "GRF.npy").exists() else None
            grm = np.load(motion_dir / "GRM.npy").astype(np.float32) if (motion_dir / "GRM.npy").exists() else None
            cop = np.load(motion_dir / "COP.npy").astype(np.float32) if (motion_dir / "COP.npy").exists() else None
            yield MocapTrial(
                subject_metadata=metadata,
                trial_id=trial_dir.name,
                activity="walking",
                time=time,
                time_for_pos=time_for_pos,
                pos=pos,
                vel=vel,
                accel=accel,
                grf=grf,
                grm=grm,
                cop=cop,
                meta={
                    "source_dataset_path": str(trial_dir),
                    "patient_md_path": str(metadata.patient_md_path) if metadata.patient_md_path else None,
                    "position_time_source": "Time_for_pos.npy" if (motion_dir / "Time_for_pos.npy").exists() else "Time.npy",
                },
            )


def load_mocap_trials(root: str | Path, metadata_filename: str = "Patient_MD.json") -> List[MocapTrial]:
    return list(iter_mocap_trials(root=root, metadata_filename=metadata_filename))
