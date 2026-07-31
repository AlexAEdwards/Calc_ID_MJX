from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from corruption_model.types import SubjectMetadata, TrialPair


EXPECTED_NUM_DOFS = 23
CANONICAL_DOF_NAMES = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
]


def _load_subject_metadata(subject_dir: Path, metadata_filename: str) -> SubjectMetadata:
    patient_md_path = subject_dir / metadata_filename
    if patient_md_path.exists():
        payload = json.loads(patient_md_path.read_text(encoding="utf-8"))
        dof_names = list(payload.get("DOF_names", []))
        num_dofs = int(payload.get("NumDOFs", len(dof_names)))
        if not dof_names and num_dofs == EXPECTED_NUM_DOFS:
            dof_names = list(CANONICAL_DOF_NAMES)
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

    fallback_names = list(CANONICAL_DOF_NAMES)
    return SubjectMetadata(
        subject_id=subject_dir.name,
        height_m=None,
        mass_kg=None,
        biological_sex=None,
        dof_names=fallback_names,
        num_dofs=len(fallback_names),
        patient_md_path=None,
        extra={},
    )


def _validate_subject_metadata(metadata: SubjectMetadata) -> None:
    if metadata.num_dofs != len(metadata.dof_names):
        raise ValueError(f"{metadata.subject_id}: NumDOFs={metadata.num_dofs} does not match DOF_names length={len(metadata.dof_names)}")
    if metadata.num_dofs != EXPECTED_NUM_DOFS:
        raise ValueError(f"{metadata.subject_id}: expected {EXPECTED_NUM_DOFS} DOFs, found {metadata.num_dofs}")


def _load_contact_or_grf(trial_dir: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    motion_dir = trial_dir / "Motion"
    grf_path = motion_dir / "GRF.npy"
    grf = np.load(grf_path).astype(np.float32) if grf_path.exists() else None
    contact_path = trial_dir / "MoCap" / "contactBoolean.npy"
    contact_mask = np.load(contact_path).astype(np.float32) if contact_path.exists() else None
    return grf, contact_mask


def _load_optional_time(path: Path, expected_frames: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        time = np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        time = np.load(path, allow_pickle=True)
    time = np.asarray(time, dtype=np.float32)
    if time.ndim != 1 or time.shape[0] != expected_frames:
        return None
    return time


def _resample_to_target_time(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    out = np.empty((target_time.shape[0], values.shape[1]), dtype=np.float32)
    for col_idx in range(values.shape[1]):
        out[:, col_idx] = np.interp(
            target_time,
            source_time,
            values[:, col_idx],
            left=values[0, col_idx],
            right=values[-1, col_idx],
        ).astype(np.float32)
    return out


def _build_common_timebase(time_a: np.ndarray, time_b: np.ndarray, *, dt: float = 0.01) -> Optional[np.ndarray]:
    time_a = np.asarray(time_a, dtype=np.float32).reshape(-1)
    time_b = np.asarray(time_b, dtype=np.float32).reshape(-1)
    if time_a.size == 0 or time_b.size == 0:
        return None
    start = max(float(time_a[0]), float(time_b[0]))
    end = min(float(time_a[-1]), float(time_b[-1]))
    if end < start:
        return None
    n_steps = int(np.floor((end - start) / float(dt))) + 1
    if n_steps <= 0:
        return None
    return (start + np.arange(n_steps, dtype=np.float32) * float(dt)).astype(np.float32)


def align_signal_blocks_to_common_timebase(
    time_a: np.ndarray,
    blocks_a: Dict[str, np.ndarray],
    time_b: np.ndarray,
    blocks_b: Dict[str, np.ndarray],
    *,
    dt: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    common_time = _build_common_timebase(time_a, time_b, dt=dt)
    if common_time is None:
        raise ValueError("Could not build a shared overlap timebase for the provided signals.")
    aligned_a = {
        name: _resample_to_target_time(np.asarray(values, dtype=np.float32), np.asarray(time_a, dtype=np.float32), common_time)
        for name, values in blocks_a.items()
    }
    aligned_b = {
        name: _resample_to_target_time(np.asarray(values, dtype=np.float32), np.asarray(time_b, dtype=np.float32), common_time)
        for name, values in blocks_b.items()
    }
    return common_time.astype(np.float32), aligned_a, aligned_b


def iter_paired_trials(root: str | Path, metadata_filename: str = "Patient_MD.json") -> Iterable[TrialPair]:
    root_path = Path(root)
    for subject_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        metadata = _load_subject_metadata(subject_dir, metadata_filename)
        _validate_subject_metadata(metadata)
        for trial_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir() and path.name.startswith("Trial_")):
            motion_dir = trial_dir / "Motion"
            mocap_dir = trial_dir / "MoCap"
            motion_pos_path = motion_dir / "Pos.npy"
            motion_time_path = motion_dir / "Time.npy"
            mocap_pos_path = mocap_dir / "Pos.npy"
            if not (motion_pos_path.exists() and motion_time_path.exists() and mocap_pos_path.exists()):
                continue

            q_opencap = np.load(motion_pos_path).astype(np.float32)
            time = np.load(motion_time_path).astype(np.float32)
            q_mocap = np.load(mocap_pos_path).astype(np.float32)
            if q_opencap.ndim != 2 or q_mocap.ndim != 2:
                raise ValueError(f"{trial_dir}: expected 2D position arrays")
            if q_opencap.shape[1] != metadata.num_dofs:
                raise ValueError(f"{trial_dir}: expected {metadata.num_dofs} DOFs, found {q_opencap.shape[1]}")
            if time.shape[0] != q_opencap.shape[0]:
                raise ValueError(f"{trial_dir}: time length {time.shape[0]} does not match frames {q_opencap.shape[0]}")
            if q_mocap.shape[1] != metadata.num_dofs:
                raise ValueError(f"{trial_dir}: expected MoCap width {metadata.num_dofs}, found {q_mocap.shape[1]}")

            mocap_time = _load_optional_time(mocap_dir / "Time.npy", q_mocap.shape[0])
            if mocap_time is None:
                mocap_time = np.linspace(float(time[0]), float(time[-1]), q_mocap.shape[0], dtype=np.float32)

            try:
                time, motion_blocks, mocap_blocks = align_signal_blocks_to_common_timebase(
                    time,
                    {"q_opencap": q_opencap},
                    mocap_time,
                    {"q_mocap": q_mocap},
                    dt=0.01,
                )
            except ValueError:
                continue
            q_opencap = motion_blocks["q_opencap"]
            q_mocap = mocap_blocks["q_mocap"]

            grf, contact_mask = _load_contact_or_grf(trial_dir)
            info_path = trial_dir / "MoCap" / "Trial_Processing_Information.json"
            meta = {}
            if info_path.exists():
                meta["trial_processing_information"] = json.loads(info_path.read_text(encoding="utf-8"))
            if mocap_time is not None:
                meta["mocap_time_path"] = str(mocap_dir / "Time.npy")
            meta["aligned_to_common_timebase"] = True
            meta["common_time_start_s"] = float(time[0]) if time.size else 0.0
            meta["common_time_end_s"] = float(time[-1]) if time.size else 0.0
            meta["motion_source"] = str(motion_pos_path)
            meta["mocap_source"] = str(mocap_pos_path)

            yield TrialPair(
                subject_metadata=metadata,
                trial_id=trial_dir.name,
                activity="walking",
                time=time,
                q_mocap=q_mocap,
                q_opencap=q_opencap,
                grf=grf,
                contact_mask=contact_mask,
                mask_valid=np.isfinite(q_mocap) & np.isfinite(q_opencap),
                meta=meta,
            )


def load_paired_trials(root: str | Path, metadata_filename: str = "Patient_MD.json") -> List[TrialPair]:
    return list(iter_paired_trials(root=root, metadata_filename=metadata_filename))
