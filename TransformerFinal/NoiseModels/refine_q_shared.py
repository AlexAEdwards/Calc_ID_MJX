from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from corruption_model.io.load_paired import align_signal_blocks_to_common_timebase
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from corruption_model.io.load_paired import align_signal_blocks_to_common_timebase

STANDARD_DOF_NAMES: List[str] = [
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

TRANSLATIONAL_DOF_NAMES = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
SUMMARY_JOINT_PRIORITY = [
    "pelvis_tilt",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
    "lumbar_extension",
]
OPENCAP_POS_INPUT_IDXS = [0, 1, 2, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22]
OPENCAP_VEL_INPUT_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22]
OPENCAP_ACC_INPUT_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22]
OPENCAP_POS_MTP_REMOVE_IDXS = (8, 14)
OPENCAP_VEL_ACC_MTP_REMOVE_IDXS = (11, 17)


@dataclass
class RefineQRecord:
    subject: str
    trial: str
    sample_name: str
    sample_id: str
    input_pos_path: str
    input_vel_path: str
    input_acc_path: str
    input_time_dir: str
    gt_pos_path: str
    gt_vel_path: str
    gt_acc_path: str
    gt_time_dir: str
    patient_md_path: Optional[str]
    height_m: float
    mass_kg: float
    dof_names: List[str]
    source_kind: str
    opencap_val: bool
    pos_dof_names: Optional[List[str]] = None
    vel_dof_names: Optional[List[str]] = None
    acc_dof_names: Optional[List[str]] = None


@dataclass
class LoadedRefineQSample:
    record: RefineQRecord
    input_time: np.ndarray
    input_pos: np.ndarray
    input_vel: np.ndarray
    input_acc: np.ndarray
    gt_pos: np.ndarray
    gt_vel: np.ndarray
    gt_acc: np.ndarray


class StreamingStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum: Optional[np.ndarray] = None
        self.sumsq: Optional[np.ndarray] = None

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.size == 0:
            return
        if self.sum is None:
            self.sum = np.zeros(arr.shape[-1], dtype=np.float64)
            self.sumsq = np.zeros(arr.shape[-1], dtype=np.float64)
        self.sum += np.sum(arr, axis=0)
        self.sumsq += np.sum(np.square(arr), axis=0)
        self.count += int(arr.shape[0])

    def finalize(self, eps: float = 1e-6) -> Normalizer:
        if self.count <= 0 or self.sum is None or self.sumsq is None:
            raise ValueError("Cannot finalize empty StreamingStats")
        mean = self.sum / float(self.count)
        var = np.maximum(self.sumsq / float(self.count) - np.square(mean), 0.0)
        std = np.sqrt(var)
        std = np.maximum(std, eps)
        return Normalizer(mean.astype(np.float32), std.astype(np.float32), eps=eps)


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < eps, eps, self.std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std + self.mean


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        return np.asarray(np.load(path), dtype=np.float32)
    except Exception:
        return None


def _first_existing_glob(parent: Path, pattern: str) -> Optional[Path]:
    matches = sorted(parent.glob(pattern))
    return matches[0] if matches else None


def _first_existing_path(parent: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def _order_preserving_subset_indices(src: np.ndarray, ref: np.ndarray) -> Optional[np.ndarray]:
    """Return indices of src columns that best match ref columns while preserving order."""
    if src.ndim != 2 or ref.ndim != 2:
        return None
    src_dim = int(src.shape[1])
    ref_dim = int(ref.shape[1])
    if src_dim < ref_dim:
        return None
    if src_dim == ref_dim:
        return np.arange(src_dim, dtype=np.int32)

    t = min(int(src.shape[0]), int(ref.shape[0]))
    if t <= 1:
        return None
    src_t = np.asarray(src[:t], dtype=np.float32)
    ref_t = np.asarray(ref[:t], dtype=np.float32)
    src_c = src_t - np.mean(src_t, axis=0, keepdims=True)
    ref_c = ref_t - np.mean(ref_t, axis=0, keepdims=True)
    src_std = np.std(src_c, axis=0, keepdims=True)
    ref_std = np.std(ref_c, axis=0, keepdims=True)
    src_std = np.where(src_std < 1e-8, 1.0, src_std)
    ref_std = np.where(ref_std < 1e-8, 1.0, ref_std)
    src_n = src_c / src_std
    ref_n = ref_c / ref_std
    score = np.abs((ref_n.T @ src_n) / float(max(t - 1, 1)))
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    m, n = score.shape
    neg_inf = -1e30
    dp = np.full((m, n), neg_inf, dtype=np.float32)
    prev = np.full((m, n), -1, dtype=np.int32)

    j_max0 = n - (m - 1)
    for j in range(0, j_max0):
        dp[0, j] = score[0, j]

    for i in range(1, m):
        running_best = neg_inf
        running_idx = -1
        j_start = i
        j_stop = n - (m - i)
        for j in range(j_start, j_stop):
            k = j - 1
            if dp[i - 1, k] > running_best:
                running_best = dp[i - 1, k]
                running_idx = k
            if running_idx >= 0:
                dp[i, j] = running_best + score[i, j]
                prev[i, j] = running_idx

    end_start = m - 1
    end_j = int(np.argmax(dp[m - 1, end_start:n]) + end_start)
    if dp[m - 1, end_j] <= neg_inf / 2:
        return None

    idx = np.zeros(m, dtype=np.int32)
    idx[m - 1] = end_j
    for i in range(m - 1, 0, -1):
        pj = int(prev[i, idx[i]])
        if pj < 0:
            return None
        idx[i - 1] = pj
    return idx


def _align_or_pad_to_reference_layout(source_arr: np.ndarray, reference_arr: np.ndarray) -> np.ndarray:
    """Align reduced columns to a reference layout, padding missing columns with zeros."""
    source_arr = np.asarray(source_arr, dtype=np.float32)
    reference_arr = np.asarray(reference_arr, dtype=np.float32)
    if source_arr.ndim != 2 or reference_arr.ndim != 2:
        return source_arr.astype(np.float32, copy=False)
    src_dim = int(source_arr.shape[1])
    ref_dim = int(reference_arr.shape[1])
    if src_dim == ref_dim:
        return source_arr.astype(np.float32, copy=False)
    if src_dim < ref_dim:
        ref_idx = _order_preserving_subset_indices(reference_arr, source_arr)
        if ref_idx is not None and ref_idx.shape[0] == src_dim:
            aligned = np.zeros((source_arr.shape[0], ref_dim), dtype=np.float32)
            aligned[:, ref_idx] = source_arr
            return aligned
        aligned = np.zeros((source_arr.shape[0], ref_dim), dtype=np.float32)
        aligned[:, :src_dim] = source_arr
        return aligned
    src_idx = _order_preserving_subset_indices(source_arr, reference_arr)
    if src_idx is not None and src_idx.shape[0] == ref_dim:
        return source_arr[:, src_idx].astype(np.float32, copy=False)
    return source_arr[:, :ref_dim].astype(np.float32, copy=False)


def _record_to_jsonable(record: RefineQRecord) -> Dict[str, Any]:
    return asdict(record)


def records_to_jsonable(records: Sequence[RefineQRecord]) -> List[Dict[str, Any]]:
    return [_record_to_jsonable(record) for record in records]


def _load_patient_metadata(subject_dir: Path) -> Tuple[Optional[Path], float, float, List[str]]:
    md_path = subject_dir / "Patient_MD.json"
    payload = _safe_load_json(md_path) or {}
    height = float(payload.get("Height_m", 1.7))
    mass = float(payload.get("Mass_kg", 70.0))
    dof_names = list(payload.get("DOF_names", STANDARD_DOF_NAMES))
    if len(dof_names) == 0:
        dof_names = list(STANDARD_DOF_NAMES)
    return (md_path if md_path.exists() else None), height, mass, dof_names


def _names_from_indices(indices: Sequence[int]) -> List[str]:
    return [STANDARD_DOF_NAMES[int(idx)] for idx in indices]


def _default_quantity_dof_names(base_dof_names: Sequence[str]) -> Dict[str, List[str]]:
    base = list(base_dof_names)
    return {
        "pos": list(base),
        "vel": list(base),
        "acc": list(base),
    }


def _opencap_quantity_dof_names() -> Dict[str, List[str]]:
    return {
        "pos": _names_from_indices(OPENCAP_POS_INPUT_IDXS),
        "vel": _names_from_indices(OPENCAP_VEL_INPUT_IDXS),
        "acc": _names_from_indices(OPENCAP_ACC_INPUT_IDXS),
    }


def get_record_quantity_dof_names(record: RefineQRecord) -> Dict[str, List[str]]:
    pos_names = list(record.pos_dof_names) if record.pos_dof_names else list(record.dof_names)
    vel_names = list(record.vel_dof_names) if record.vel_dof_names else list(record.dof_names)
    acc_names = list(record.acc_dof_names) if record.acc_dof_names else list(record.dof_names)
    return {
        "pos": pos_names,
        "vel": vel_names,
        "acc": acc_names,
    }


def get_quantity_dims_from_sample(sample: LoadedRefineQSample) -> Dict[str, int]:
    return {
        "pos": int(sample.input_pos.shape[-1]),
        "vel": int(sample.input_vel.shape[-1]),
        "acc": int(sample.input_acc.shape[-1]),
    }


def _fallback_subject_scalars(trial_dir: Path, height_m: float, mass_kg: float) -> Tuple[float, float]:
    for base_dir_name in ("ProcessedData", "MoCap", "Motion"):
        base_dir = trial_dir / base_dir_name
        height_arr = _safe_load_npy(base_dir / "Height_m.npy")
        mass_arr = _safe_load_npy(base_dir / "Mass_kg.npy")
        if height_arr is not None and height_arr.size > 0:
            height_m = float(np.asarray(height_arr).reshape(-1)[0])
        if mass_arr is not None and mass_arr.size > 0:
            mass_kg = float(np.asarray(mass_arr).reshape(-1)[0])
    return height_m, mass_kg


def discover_refine_q_records(data_dir: str, *, opencap_val: bool = False) -> List[RefineQRecord]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    records: List[RefineQRecord] = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        patient_md_path, height_m, mass_kg, dof_names = _load_patient_metadata(subject_dir)
        for trial_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir() and path.name.startswith("Trial_")):
            trial_height_m, trial_mass_kg = _fallback_subject_scalars(trial_dir, height_m, mass_kg)
            if opencap_val:
                processed_dir = trial_dir / "ProcessedData"
                motion_dir = trial_dir / "Motion"
                mocap_dir = trial_dir / "MoCap"
                if not motion_dir.exists() or not mocap_dir.exists():
                    continue
                pos_path = _first_existing_path(motion_dir, ["Pos.npy"])
                vel_path = _first_existing_path(motion_dir, ["Vel.npy"])
                acc_path = _first_existing_path(motion_dir, ["Accel.npy"])
                gt_pos_path = _first_existing_path(mocap_dir, ["Pos.npy"])
                gt_vel_path = _first_existing_path(mocap_dir, ["Vel.npy"])
                gt_acc_path = _first_existing_path(mocap_dir, ["Accel.npy"])
                if not all(path is not None and path.exists() for path in (pos_path, vel_path, acc_path, gt_pos_path, gt_vel_path, gt_acc_path)):
                    continue
                quantity_dof_names = _default_quantity_dof_names(dof_names)
                sample_name = "Motion"
                sample_id = f"{subject_dir.name}/{trial_dir.name}/{sample_name}"
                records.append(
                    RefineQRecord(
                        subject=subject_dir.name,
                        trial=trial_dir.name,
                        sample_name=sample_name,
                        sample_id=sample_id,
                        input_pos_path=str(pos_path),
                        input_vel_path=str(vel_path),
                        input_acc_path=str(acc_path),
                        input_time_dir=str(motion_dir),
                        gt_pos_path=str(gt_pos_path),
                        gt_vel_path=str(gt_vel_path),
                        gt_acc_path=str(gt_acc_path),
                        gt_time_dir=str(mocap_dir),
                        patient_md_path=str(patient_md_path) if patient_md_path is not None else None,
                        height_m=trial_height_m,
                        mass_kg=trial_mass_kg,
                        dof_names=list(quantity_dof_names["pos"]),
                        source_kind="OpenCapSubjects",
                        opencap_val=True,
                        pos_dof_names=list(quantity_dof_names["pos"]),
                        vel_dof_names=list(quantity_dof_names["vel"]),
                        acc_dof_names=list(quantity_dof_names["acc"]),
                    )
                )
                continue

            processed_dir = trial_dir / "ProcessedData"
            motion_dir = trial_dir / "Motion"
            if not processed_dir.exists() or not motion_dir.exists():
                continue
            gt_pos_path = motion_dir / "Pos.npy"
            gt_vel_path = motion_dir / "Vel.npy"
            gt_acc_path = motion_dir / "Accel.npy"
            if not all(path.exists() for path in (gt_pos_path, gt_vel_path, gt_acc_path)):
                continue

            for noised_dir in sorted(path for path in processed_dir.glob("Noised_*") if path.is_dir()):
                pos_path = _first_existing_glob(noised_dir, "Pos_noised_*.npy")
                vel_path = _first_existing_glob(noised_dir, "Vel_noised_*.npy")
                acc_path = _first_existing_glob(noised_dir, "Accel_noised_*.npy")
                if pos_path is None or vel_path is None or acc_path is None:
                    continue
                sample_name = noised_dir.name
                sample_id = f"{subject_dir.name}/{trial_dir.name}/{sample_name}"
                records.append(
                    RefineQRecord(
                        subject=subject_dir.name,
                        trial=trial_dir.name,
                        sample_name=sample_name,
                        sample_id=sample_id,
                        input_pos_path=str(pos_path),
                        input_vel_path=str(vel_path),
                        input_acc_path=str(acc_path),
                        input_time_dir=str(processed_dir),
                        gt_pos_path=str(gt_pos_path),
                        gt_vel_path=str(gt_vel_path),
                        gt_acc_path=str(gt_acc_path),
                        gt_time_dir=str(motion_dir),
                        patient_md_path=str(patient_md_path) if patient_md_path is not None else None,
                        height_m=trial_height_m,
                        mass_kg=trial_mass_kg,
                        dof_names=list(dof_names),
                        source_kind="TrustedDatasetNoisedFromModel",
                        opencap_val=False,
                    )
                )
    return records


def split_records_by_subject(records: Sequence[RefineQRecord], seed: int = 42, train_fraction: float = 0.8) -> Tuple[List[RefineQRecord], List[RefineQRecord]]:
    grouped: Dict[str, List[RefineQRecord]] = {}
    for record in records:
        grouped.setdefault(record.subject, []).append(record)
    subjects = sorted(grouped.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    if len(subjects) <= 1:
        pivot = max(1, int(math.floor(train_fraction * len(records))))
        train_records = list(records[:pivot])
        val_records = list(records[pivot:]) or list(records[:1])
        return train_records, val_records
    pivot = max(1, int(math.floor(train_fraction * len(subjects))))
    pivot = min(pivot, len(subjects) - 1)
    train_subjects = set(subjects[:pivot])
    train_records = [record for record in records if record.subject in train_subjects]
    val_records = [record for record in records if record.subject not in train_subjects]
    return train_records, val_records


def _load_time_candidate(base_dir: Path, filename: str) -> Optional[np.ndarray]:
    arr = _safe_load_npy(base_dir / filename)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None
    return arr


def select_time_vector(base_dir: Path, expected_len: int, *, prefer_pos: bool) -> Optional[np.ndarray]:
    priority = ["Time_for_pos.npy", "Time.npy"] if prefer_pos else ["Time.npy", "Time_for_pos.npy"]
    for filename in priority:
        arr = _load_time_candidate(base_dir, filename)
        if arr is not None and int(arr.shape[0]) == int(expected_len):
            return arr.astype(np.float32)
    for filename in priority:
        arr = _load_time_candidate(base_dir, filename)
        if arr is not None:
            return arr.astype(np.float32)
    return None


def infer_uniform_time(num_steps: int, reference_time: Optional[np.ndarray] = None, dt: float = 0.01) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    start = 0.0
    if reference_time is not None and reference_time.size > 0:
        start = float(reference_time.reshape(-1)[0])
    return (start + np.arange(num_steps, dtype=np.float32) * float(dt)).astype(np.float32)


def resample_timeseries(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    source_time = np.asarray(source_time, dtype=np.float64).reshape(-1)
    target_time = np.asarray(target_time, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != source_time.shape[0]:
        raise ValueError(f"Time/value length mismatch: {source_time.shape[0]} vs {values.shape[0]}")
    if source_time.shape[0] == target_time.shape[0] and np.allclose(source_time, target_time, atol=1e-6, rtol=1e-6):
        return values.astype(np.float32)

    sort_idx = np.argsort(source_time)
    source_time = source_time[sort_idx]
    values = values[sort_idx]
    unique_time, unique_indices = np.unique(source_time, return_index=True)
    values = values[unique_indices]
    if unique_time.shape[0] <= 1:
        return np.repeat(values[:1], repeats=target_time.shape[0], axis=0).astype(np.float32)

    if values.ndim == 1:
        return np.interp(target_time, unique_time, values, left=values[0], right=values[-1]).astype(np.float32)

    out = np.empty((target_time.shape[0], values.shape[1]), dtype=np.float32)
    for channel_idx in range(values.shape[1]):
        out[:, channel_idx] = np.interp(
            target_time,
            unique_time,
            values[:, channel_idx],
            left=values[0, channel_idx],
            right=values[-1, channel_idx],
        ).astype(np.float32)
    return out


def _resolve_input_time(base_dir: Path, signal_len: int) -> np.ndarray:
    time_vec = select_time_vector(base_dir, signal_len, prefer_pos=True)
    if time_vec is not None and int(time_vec.shape[0]) == int(signal_len):
        return time_vec.astype(np.float32)
    reference = time_vec if time_vec is not None else None
    return infer_uniform_time(signal_len, reference_time=reference)


def _align_input_signal(signal: np.ndarray, base_dir: Path, target_time: np.ndarray, *, prefer_pos: bool) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape[0] == target_time.shape[0]:
        return signal
    signal_time = select_time_vector(base_dir, signal.shape[0], prefer_pos=prefer_pos)
    if signal_time is None:
        signal_time = infer_uniform_time(signal.shape[0], reference_time=target_time)
    return resample_timeseries(signal_time, signal, target_time)


def _coerce_opencap_block(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return arr
    if name == "pos" and int(arr.shape[1]) == 18:
        return np.delete(arr, OPENCAP_POS_MTP_REMOVE_IDXS, axis=1).astype(np.float32, copy=False)
    if name in {"vel", "acc"} and int(arr.shape[1]) == 21:
        return np.delete(arr, OPENCAP_VEL_ACC_MTP_REMOVE_IDXS, axis=1).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def load_refine_q_sample(record: RefineQRecord) -> LoadedRefineQSample:
    input_pos = np.asarray(np.load(record.input_pos_path), dtype=np.float32)
    input_vel = np.asarray(np.load(record.input_vel_path), dtype=np.float32)
    input_acc = np.asarray(np.load(record.input_acc_path), dtype=np.float32)
    if record.opencap_val:
        input_pos = _coerce_opencap_block("pos", input_pos)
        input_vel = _coerce_opencap_block("vel", input_vel)
        input_acc = _coerce_opencap_block("acc", input_acc)
    input_time_dir = Path(record.input_time_dir)
    input_time = select_time_vector(input_time_dir, int(input_pos.shape[0]), prefer_pos=True)
    if input_time is None:
        input_time = infer_uniform_time(int(input_pos.shape[0]))
    input_vel = _align_input_signal(input_vel, input_time_dir, input_time, prefer_pos=False)
    input_acc = _align_input_signal(input_acc, input_time_dir, input_time, prefer_pos=False)

    gt_pos = np.asarray(np.load(record.gt_pos_path), dtype=np.float32)
    gt_vel = np.asarray(np.load(record.gt_vel_path), dtype=np.float32)
    gt_acc = np.asarray(np.load(record.gt_acc_path), dtype=np.float32)
    if record.opencap_val:
        gt_pos = _coerce_opencap_block("pos", gt_pos)
        gt_vel = _coerce_opencap_block("vel", gt_vel)
        gt_acc = _coerce_opencap_block("acc", gt_acc)
    gt_time_dir = Path(record.gt_time_dir)

    gt_time = select_time_vector(gt_time_dir, int(gt_pos.shape[0]), prefer_pos=True)
    if gt_time is None:
        gt_time = infer_uniform_time(int(gt_pos.shape[0]))

    if record.opencap_val:
        target_time, input_blocks, gt_blocks = align_signal_blocks_to_common_timebase(
            input_time,
            {"pos": input_pos, "vel": input_vel, "acc": input_acc},
            gt_time,
            {"pos": gt_pos, "vel": gt_vel, "acc": gt_acc},
            dt=0.01,
        )
        input_pos = input_blocks["pos"]
        input_vel = input_blocks["vel"]
        input_acc = input_blocks["acc"]
        gt_pos_resampled = gt_blocks["pos"]
        gt_vel_resampled = gt_blocks["vel"]
        gt_acc_resampled = gt_blocks["acc"]
    else:
        target_time = input_time
        gt_pos_time = select_time_vector(gt_time_dir, int(gt_pos.shape[0]), prefer_pos=True)
        if gt_pos_time is None:
            gt_pos_time = infer_uniform_time(int(gt_pos.shape[0]), reference_time=target_time)
        gt_vel_time = select_time_vector(gt_time_dir, int(gt_vel.shape[0]), prefer_pos=False)
        if gt_vel_time is None:
            gt_vel_time = gt_pos_time if gt_pos.shape[0] == gt_vel.shape[0] else infer_uniform_time(int(gt_vel.shape[0]), reference_time=target_time)
        gt_acc_time = select_time_vector(gt_time_dir, int(gt_acc.shape[0]), prefer_pos=False)
        if gt_acc_time is None:
            gt_acc_time = gt_vel_time if gt_vel.shape[0] == gt_acc.shape[0] else infer_uniform_time(int(gt_acc.shape[0]), reference_time=target_time)
        gt_pos_resampled = resample_timeseries(gt_pos_time, gt_pos, target_time)
        gt_vel_resampled = resample_timeseries(gt_vel_time, gt_vel, target_time)
        gt_acc_resampled = resample_timeseries(gt_acc_time, gt_acc, target_time)

    if input_pos.shape[1] != gt_pos_resampled.shape[1]:
        input_pos = _align_or_pad_to_reference_layout(input_pos, gt_pos_resampled)
    if input_vel.shape[1] != gt_vel_resampled.shape[1]:
        input_vel = _align_or_pad_to_reference_layout(input_vel, gt_vel_resampled)
    if input_acc.shape[1] != gt_acc_resampled.shape[1]:
        input_acc = _align_or_pad_to_reference_layout(input_acc, gt_acc_resampled)

    return LoadedRefineQSample(
        record=record,
        input_time=target_time.astype(np.float32),
        input_pos=input_pos.astype(np.float32),
        input_vel=input_vel.astype(np.float32),
        input_acc=input_acc.astype(np.float32),
        gt_pos=gt_pos_resampled.astype(np.float32),
        gt_vel=gt_vel_resampled.astype(np.float32),
        gt_acc=gt_acc_resampled.astype(np.float32),
    )


def build_full_sequence_arrays(sample: LoadedRefineQSample) -> Dict[str, np.ndarray]:
    input_all = np.concatenate([sample.input_pos, sample.input_vel, sample.input_acc], axis=-1).astype(np.float32)
    pos_res = (sample.gt_pos - sample.input_pos).astype(np.float32)
    vel_res = (sample.gt_vel - sample.input_vel).astype(np.float32)
    acc_res = (sample.gt_acc - sample.input_acc).astype(np.float32)
    residual_all = np.concatenate([pos_res, vel_res, acc_res], axis=-1).astype(np.float32)
    gt_all = np.concatenate([sample.gt_pos, sample.gt_vel, sample.gt_acc], axis=-1).astype(np.float32)
    noisy_all = np.concatenate([sample.input_pos, sample.input_vel, sample.input_acc], axis=-1).astype(np.float32)
    static_context = np.asarray([sample.record.height_m, sample.record.mass_kg], dtype=np.float32)
    return {
        "input": input_all,
        "residual": residual_all,
        "gt": gt_all,
        "noisy": noisy_all,
        "static_context": static_context,
    }


def fit_normalizers(records: Sequence[RefineQRecord]) -> Dict[str, Normalizer]:
    input_stats = StreamingStats()
    static_stats = StreamingStats()
    pos_stats = StreamingStats()
    vel_stats = StreamingStats()
    acc_stats = StreamingStats()

    for record in records:
        sample = load_refine_q_sample(record)
        arrays = build_full_sequence_arrays(sample)
        quantity_names = get_record_quantity_dof_names(record)
        input_stats.update(arrays["input"])
        static_stats.update(arrays["static_context"])
        pos_dim = len(quantity_names["pos"])
        vel_dim = len(quantity_names["vel"])
        acc_dim = len(quantity_names["acc"])
        pos_residual, vel_residual, acc_residual = split_residual_blocks(arrays["residual"], pos_dim, vel_dim, acc_dim)
        pos_stats.update(pos_residual)
        vel_stats.update(vel_residual)
        acc_stats.update(acc_residual)

    return {
        "input": input_stats.finalize(),
        "static": static_stats.finalize(),
        "pos_residual": pos_stats.finalize(),
        "vel_residual": vel_stats.finalize(),
        "acc_residual": acc_stats.finalize(),
    }


def normalize_input_block(input_block: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    return np.asarray(normalizer.normalize(np.asarray(input_block, dtype=np.float32)), dtype=np.float32)


def normalize_static_block(static_block: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    return np.asarray(normalizer.normalize(np.asarray(static_block, dtype=np.float32)), dtype=np.float32)


def split_residual_blocks(array: np.ndarray, pos_dim: int, vel_dim: int, acc_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(array, dtype=np.float32)
    pos = array[..., 0:pos_dim]
    vel = array[..., pos_dim:pos_dim + vel_dim]
    acc = array[..., pos_dim + vel_dim:pos_dim + vel_dim + acc_dim]
    return pos, vel, acc


def combine_residual_blocks(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32), np.asarray(acc, dtype=np.float32)], axis=-1)


def predict_refined_sequence(
    model: Any,
    params: Any,
    normalizers: Dict[str, Normalizer],
    sample: LoadedRefineQSample,
) -> Dict[str, np.ndarray]:
    arrays = build_full_sequence_arrays(sample)
    dims = get_quantity_dims_from_sample(sample)
    model_input = normalize_input_block(arrays["input"], normalizers["input"])[None, ...]
    static_context = normalize_static_block(arrays["static_context"], normalizers["static"])[None, ...]
    pred_z = np.asarray(model.apply({"params": params}, model_input, static_context, train=False))[0]

    pred_pos_z, pred_vel_z, pred_acc_z = split_residual_blocks(pred_z, dims["pos"], dims["vel"], dims["acc"])
    pred_pos_res = np.asarray(normalizers["pos_residual"].unnormalize(pred_pos_z), dtype=np.float32)
    pred_vel_res = np.asarray(normalizers["vel_residual"].unnormalize(pred_vel_z), dtype=np.float32)
    pred_acc_res = np.asarray(normalizers["acc_residual"].unnormalize(pred_acc_z), dtype=np.float32)

    pred_pos = sample.input_pos + pred_pos_res
    pred_vel = sample.input_vel + pred_vel_res
    pred_acc = sample.input_acc + pred_acc_res

    target_pos_res, target_vel_res, target_acc_res = split_residual_blocks(arrays["residual"], dims["pos"], dims["vel"], dims["acc"])

    return {
        "time": sample.input_time,
        "input_pos": sample.input_pos,
        "input_vel": sample.input_vel,
        "input_acc": sample.input_acc,
        "gt_pos": sample.gt_pos,
        "gt_vel": sample.gt_vel,
        "gt_acc": sample.gt_acc,
        "pred_pos_residual": pred_pos_res,
        "pred_vel_residual": pred_vel_res,
        "pred_acc_residual": pred_acc_res,
        "target_pos_residual": target_pos_res,
        "target_vel_residual": target_vel_res,
        "target_acc_residual": target_acc_res,
        "pred_pos": pred_pos.astype(np.float32),
        "pred_vel": pred_vel.astype(np.float32),
        "pred_acc": pred_acc.astype(np.float32),
    }


def build_model_from_hyperparams(hyperparams: Dict[str, Any], input_dim: int, static_dim: int, output_dim: int) -> Any:
    try:
        from .mod_q_shared import build_mod_q_model
    except ImportError:
        from mod_q_shared import build_mod_q_model
    model_cfg = {
        "output_dim": int(output_dim),
        "d_model": int(hyperparams.get("d_model", 256)),
        "num_heads": int(hyperparams.get("num_heads", 4)),
        "num_layers": int(hyperparams.get("num_layers", 4)),
        "ff_dim": int(hyperparams.get("ff_dim", 1024)),
        "dropout_rate": float(hyperparams.get("dropout_rate", 0.1)),
        "use_cnn": bool(hyperparams.get("use_cnn", True)),
        "cnn_num_layers": int(hyperparams.get("cnn_num_layers", 2)),
        "cnn_kernel_sizes": hyperparams.get("cnn_kernel_sizes", [3, 5]),
        "use_multitask": False,
    }
    return build_mod_q_model(input_dim=input_dim, static_dim=static_dim, model_cfg=model_cfg)


def sanitize_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__")


def is_rotational_dof(name: str) -> bool:
    return name not in TRANSLATIONAL_DOF_NAMES


def quantity_unit(name: str, quantity: str) -> str:
    if is_rotational_dof(name):
        if quantity == "pos":
            return "deg"
        if quantity == "vel":
            return "deg/s"
        return "deg/s^2"
    if quantity == "pos":
        return "m"
    if quantity == "vel":
        return "m/s"
    return "m/s^2"


def quantity_scale(name: str, quantity: str) -> float:
    return float(180.0 / np.pi) if is_rotational_dof(name) else 1.0


def compute_per_joint_rmse(
    predicted: np.ndarray,
    target: np.ndarray,
    dof_names: Sequence[str],
    quantity: str,
) -> Dict[str, Dict[str, Any]]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch for RMSE: {predicted.shape} vs {target.shape}")

    out: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(dof_names):
        scale = quantity_scale(name, quantity)
        rmse = float(np.sqrt(np.mean(np.square((predicted[:, idx] - target[:, idx]) * scale))))
        out[str(name)] = {
            "rmse": rmse,
            "unit": quantity_unit(name, quantity),
            "scale": scale,
        }
    return out


def compute_sequence_metrics(predictions: Dict[str, np.ndarray], dof_names: Sequence[str]) -> Dict[str, Any]:
    if isinstance(dof_names, dict):
        quantity_dof_names = {key: list(value) for key, value in dof_names.items()}
    else:
        quantity_dof_names = _default_quantity_dof_names(dof_names)
    metrics: Dict[str, Any] = {}
    for quantity in ("pos", "vel", "acc"):
        pred_key = f"pred_{quantity}"
        input_key = f"input_{quantity}"
        gt_key = f"gt_{quantity}"
        quantity_names = quantity_dof_names[quantity]
        pred_joint = compute_per_joint_rmse(predictions[pred_key], predictions[gt_key], quantity_names, quantity)
        input_joint = compute_per_joint_rmse(predictions[input_key], predictions[gt_key], quantity_names, quantity)
        pred_mean = float(np.mean([payload["rmse"] for payload in pred_joint.values()]))
        input_mean = float(np.mean([payload["rmse"] for payload in input_joint.values()]))
        improvement = input_mean - pred_mean
        for dof_name in pred_joint.keys():
            pred_joint[dof_name]["input_rmse"] = float(input_joint[dof_name]["rmse"])
            pred_joint[dof_name]["rmse_improvement"] = float(input_joint[dof_name]["rmse"] - pred_joint[dof_name]["rmse"])
            pred_joint[dof_name]["improvement_percent"] = (
                float(100.0 * pred_joint[dof_name]["rmse_improvement"] / input_joint[dof_name]["rmse"])
                if float(input_joint[dof_name]["rmse"]) > 1e-12
                else 0.0
            )
        metrics[quantity] = {
            "pred_joint_rmse": pred_joint,
            "input_joint_rmse": input_joint,
            "pred_mean_rmse": pred_mean,
            "input_mean_rmse": input_mean,
            "mean_rmse_improvement": improvement,
        }
    return metrics


def order_summary_dofs(dof_names: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for name in SUMMARY_JOINT_PRIORITY:
        if name in dof_names and name not in ordered:
            ordered.append(name)
    for name in dof_names:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _normalize_quantity_dof_names(dof_names: Sequence[str] | Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    if isinstance(dof_names, dict):
        return {
            "pos": list(dof_names.get("pos", [])),
            "vel": list(dof_names.get("vel", [])),
            "acc": list(dof_names.get("acc", [])),
        }
    return _default_quantity_dof_names(dof_names)


def _draw_single_dof_panel(
    ax: Any,
    time_axis: np.ndarray,
    prediction: np.ndarray,
    noisy_input: np.ndarray,
    target: np.ndarray,
    *,
    dof_name: str,
    dof_idx: int,
    quantity: str,
    metric_payload: Dict[str, Any],
    show_legend: bool = False,
) -> None:
    scale = quantity_scale(dof_name, quantity)
    unit = quantity_unit(dof_name, quantity)
    ax.plot(time_axis, target[:, dof_idx] * scale, color="#1f77b4", linewidth=1.5, label="GT")
    ax.plot(time_axis, prediction[:, dof_idx] * scale, color="#d62728", linewidth=1.2, linestyle="--", label="Pred")
    ax.plot(time_axis, noisy_input[:, dof_idx] * scale, color="#2ca02c", linewidth=1.0, linestyle=":", label="Input")
    joint_metrics = metric_payload["pred_joint_rmse"][dof_name]
    ax.set_title(
        f"{dof_name} | pred {joint_metrics['rmse']:.2f} {unit} "
        f"| input {joint_metrics['input_rmse']:.2f} {unit} "
        f"| Δ {joint_metrics['rmse_improvement']:+.2f} {unit}",
        fontsize=8,
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=6)
    if show_legend:
        ax.legend(fontsize=6, loc="upper right")


def _render_dof_grid(
    axes: np.ndarray,
    *,
    split_name: str,
    time_axis: np.ndarray,
    predictions: Dict[str, np.ndarray],
    dof_names: Sequence[str] | Dict[str, Sequence[str]],
    metrics: Dict[str, Any],
    ncols: int,
) -> None:
    quantity_dof_names = _normalize_quantity_dof_names(dof_names)
    n_dofs_max = max(len(quantity_dof_names["pos"]), len(quantity_dof_names["vel"]), len(quantity_dof_names["acc"]))
    nrows_per_quantity = int(np.ceil(max(n_dofs_max, 1) / float(ncols)))
    quantity_names = ["pos", "vel", "acc"]

    for quantity_idx, quantity in enumerate(quantity_names):
        row_offset = quantity_idx * nrows_per_quantity
        quantity_metrics = metrics[quantity]
        quantity_names_list = list(quantity_dof_names[quantity])
        ordered_dofs = order_summary_dofs(quantity_names_list)
        n_dofs = len(ordered_dofs)
        for dof_pos, dof_name in enumerate(ordered_dofs):
            row = row_offset + (dof_pos // ncols)
            col = dof_pos % ncols
            ax = axes[row][col]
            joint_metrics = quantity_metrics["pred_joint_rmse"][dof_name]
            _draw_single_dof_panel(
                ax,
                time_axis,
                predictions[f"pred_{quantity}"],
                predictions[f"input_{quantity}"],
                predictions[f"gt_{quantity}"],
                dof_name=dof_name,
                dof_idx=quantity_names_list.index(dof_name),
                quantity=quantity,
                metric_payload=quantity_metrics,
                show_legend=(quantity_idx == 0 and dof_pos == 0),
            )
            ax.text(
                0.99,
                0.02,
                f"{joint_metrics['rmse_improvement']:+.2f} {joint_metrics['unit']}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                family="monospace",
                color="#6a3d9a" if joint_metrics["rmse_improvement"] >= 0 else "#b22222",
            )
            if col == 0:
                ax.set_ylabel(f"{split_name}\n{quantity.upper()}", fontsize=9)
            if row == row_offset + nrows_per_quantity - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

        for blank_idx in range(n_dofs, nrows_per_quantity * ncols):
            row = row_offset + (blank_idx // ncols)
            col = blank_idx % ncols
            axes[row][col].set_axis_off()

    for quantity_idx, quantity in enumerate(quantity_names):
        axes[quantity_idx * nrows_per_quantity, 0].set_ylabel(
            f"{split_name}\n{quantity.upper()}",
            fontsize=9,
        )


def _build_dof_grid_figure(
    *,
    split_name: str,
    time_axis: np.ndarray,
    predictions: Dict[str, np.ndarray],
    dof_names: Sequence[str] | Dict[str, Sequence[str]],
    metrics: Dict[str, Any],
    output_path: Path,
    epoch: Optional[int] = None,
    ncols: int = 4,
) -> Path:
    quantity_dof_names = _normalize_quantity_dof_names(dof_names)
    n_dofs_max = max(len(quantity_dof_names["pos"]), len(quantity_dof_names["vel"]), len(quantity_dof_names["acc"]))
    nrows_per_quantity = int(np.ceil(max(n_dofs_max, 1) / float(ncols)))
    quantity_names = ["pos", "vel", "acc"]
    fig_height = max(10.0, 2.4 * nrows_per_quantity * len(quantity_names))
    fig_width = max(18.0, 4.2 * ncols)
    fig, axes = plt.subplots(len(quantity_names) * nrows_per_quantity, ncols, figsize=(fig_width, fig_height), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.25, top=0.95, bottom=0.03, left=0.04, right=0.98)
    _render_dof_grid(
        axes,
        split_name=split_name,
        time_axis=time_axis,
        predictions=predictions,
        dof_names=dof_names,
        metrics=metrics,
        ncols=ncols,
    )
    epoch_prefix = f"Epoch {int(epoch)} | " if epoch is not None else ""
    fig.suptitle(
        f"{epoch_prefix}{split_name} residual refinement by DOF | "
        f"ordered hip/knee/ankle first | "
        f"pos RMSE {metrics['pos']['pred_mean_rmse']:.3f} vs input {metrics['pos']['input_mean_rmse']:.3f} | "
        f"vel RMSE {metrics['vel']['pred_mean_rmse']:.3f} vs input {metrics['vel']['input_mean_rmse']:.3f} | "
        f"acc RMSE {metrics['acc']['pred_mean_rmse']:.3f} vs input {metrics['acc']['input_mean_rmse']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_refine_summary(
    *,
    train_record_id: str,
    train_predictions: Dict[str, np.ndarray],
    val_record_id: str,
    val_predictions: Dict[str, np.ndarray],
    dof_names: Sequence[str] | Dict[str, Sequence[str]],
    output_path: Path,
    epoch: Optional[int] = None,
) -> Path:
    train_metrics = compute_sequence_metrics(train_predictions, dof_names)
    val_metrics = compute_sequence_metrics(val_predictions, dof_names)
    quantity_dof_names = _normalize_quantity_dof_names(dof_names)
    n_dofs = max(len(quantity_dof_names["pos"]), len(quantity_dof_names["vel"]), len(quantity_dof_names["acc"]))
    ncols = 4
    nrows_per_quantity = int(np.ceil(max(n_dofs, 1) / float(ncols)))
    total_rows = len(["pos", "vel", "acc"]) * nrows_per_quantity
    fig_width = max(34.0, 4.4 * ncols * 2)
    fig_height = max(12.0, 2.5 * total_rows)
    fig, axes = plt.subplots(total_rows, ncols * 2, figsize=(fig_width, fig_height), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.18, top=0.94, bottom=0.03, left=0.03, right=0.99)

    left_axes = axes[:, :ncols]
    right_axes = axes[:, ncols:]
    _render_dof_grid(
        left_axes,
        split_name=f"TRAIN {train_record_id}",
        time_axis=train_predictions["time"],
        predictions=train_predictions,
        dof_names=dof_names,
        metrics=train_metrics,
        ncols=ncols,
    )
    _render_dof_grid(
        right_axes,
        split_name=f"VAL {val_record_id}",
        time_axis=val_predictions["time"],
        predictions=val_predictions,
        dof_names=dof_names,
        metrics=val_metrics,
        ncols=ncols,
    )

    epoch_prefix = f"Epoch {int(epoch)} | " if epoch is not None else ""
    fig.suptitle(
        f"{epoch_prefix}Residual q-refinement summary by DOF | hip/knee/ankle prioritized | "
        f"Train pos RMSE {train_metrics['pos']['pred_mean_rmse']:.3f} vs input {train_metrics['pos']['input_mean_rmse']:.3f} | "
        f"Val pos RMSE {val_metrics['pos']['pred_mean_rmse']:.3f} vs input {val_metrics['pos']['input_mean_rmse']:.3f}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_single_inference_summary(
    *,
    sample_id: str,
    predictions: Dict[str, np.ndarray],
    dof_names: Sequence[str] | Dict[str, Sequence[str]],
    output_path: Path,
) -> Path:
    metrics = compute_sequence_metrics(predictions, dof_names)
    return _build_dof_grid_figure(
        split_name=sample_id,
        time_axis=predictions["time"],
        predictions=predictions,
        dof_names=dof_names,
        metrics=metrics,
        output_path=output_path,
        epoch=None,
    )


def _compute_open_capval_aggregate_rows(
    records: Sequence[RefineQRecord],
    metrics_list: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(records) != len(metrics_list):
        raise ValueError("records and metrics_list must have the same length")
    if not records:
        raise ValueError("records must not be empty")

    rows: List[Dict[str, Any]] = []
    per_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    quantity_dof_names = get_record_quantity_dof_names(records[0])
    for record, metrics in zip(records, metrics_list):
        row = {
            "sample_id": record.sample_id,
            "subject": record.subject,
            "trial": record.trial,
            "source_kind": record.source_kind,
        }
        for quantity in ("pos", "vel", "acc"):
            row[f"{quantity}_pred_rmse"] = float(metrics[quantity]["pred_mean_rmse"])
            row[f"{quantity}_input_rmse"] = float(metrics[quantity]["input_mean_rmse"])
            row[f"{quantity}_improvement"] = float(metrics[quantity]["mean_rmse_improvement"])
        rows.append(row)
        per_subject[record.subject].append(row)

    return {
        "rows": rows,
        "per_subject": per_subject,
        "dof_names": quantity_dof_names,
    }


def plot_open_capval_dashboard(
    *,
    records: Sequence[RefineQRecord],
    metrics_list: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, str]:
    aggregate = _compute_open_capval_aggregate_rows(records, metrics_list)
    rows = aggregate["rows"]
    per_subject = aggregate["per_subject"]
    quantity_dof_names = aggregate["dof_names"]
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}
    quantities = ["pos", "vel", "acc"]
    quantity_titles = {"pos": "Position", "vel": "Velocity", "acc": "Acceleration"}
    quantity_labels = {"pos": "deg / m", "vel": "deg/s / m/s", "acc": "deg/s^2 / m/s^2"}

    fig, axes = plt.subplots(3, 3, figsize=(22, 19))
    fig.subplots_adjust(hspace=0.35, wspace=0.22, top=0.93, bottom=0.05, left=0.05, right=0.98)
    subject_names = sorted(per_subject.keys())

    for row_idx, quantity in enumerate(quantities):
        pred_vals = np.asarray([row[f"{quantity}_pred_rmse"] for row in rows], dtype=np.float64)
        input_vals = np.asarray([row[f"{quantity}_input_rmse"] for row in rows], dtype=np.float64)
        improvement_vals = np.asarray([row[f"{quantity}_improvement"] for row in rows], dtype=np.float64)
        subject_improvements = np.asarray(
            [
                float(np.mean([subject_row[f"{quantity}_improvement"] for subject_row in per_subject[subject]]))
                for subject in subject_names
            ],
            dtype=np.float64,
        )

        axes[row_idx, 0].bar(["Input", "Pred"], [float(np.mean(input_vals)), float(np.mean(pred_vals))], color=["#2ca02c", "#d62728"])
        axes[row_idx, 0].set_title(f"Mean {quantity_titles[quantity]} RMSE")
        axes[row_idx, 0].set_ylabel(quantity_labels[quantity])
        axes[row_idx, 0].grid(True, axis="y", alpha=0.25)

        axes[row_idx, 1].hist(improvement_vals, bins=min(24, max(8, len(improvement_vals) // 2)), color="#6a3d9a", alpha=0.85)
        axes[row_idx, 1].axvline(float(np.mean(improvement_vals)), color="black", linestyle="--", linewidth=1.2, label="Mean")
        axes[row_idx, 1].axvline(0.0, color="#666666", linestyle=":", linewidth=1.0, label="No change")
        axes[row_idx, 1].set_title(f"{quantity_titles[quantity]} improvement")
        axes[row_idx, 1].set_xlabel(quantity_labels[quantity])
        axes[row_idx, 1].grid(True, alpha=0.25)
        axes[row_idx, 1].legend(fontsize=7)

        axes[row_idx, 2].boxplot(
            [subject_improvements],
            labels=[f"{quantity.upper()} per-subject"],
            patch_artist=True,
            boxprops=dict(facecolor="#9ecae1"),
        )
        axes[row_idx, 2].axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
        axes[row_idx, 2].set_title(f"Per-subject mean improvement")
        axes[row_idx, 2].grid(True, axis="y", alpha=0.25)

    dashboard_path = output_dir / "open_capval_dashboard.png"
    fig.suptitle("OpenCapVal Refinement Dashboard", fontsize=16, fontweight="bold")
    fig.savefig(dashboard_path, dpi=180)
    plt.close(fig)
    outputs["dashboard_png"] = str(dashboard_path)

    fig2, axes2 = plt.subplots(3, 1, figsize=(24, 18), sharex=True)
    fig2.subplots_adjust(hspace=0.28, top=0.93, bottom=0.12, left=0.05, right=0.98)
    width = 0.25
    for q_idx, quantity in enumerate(quantities):
        dof_names = list(quantity_dof_names[quantity])
        x = np.arange(len(dof_names))
        pred_by_dof = []
        input_by_dof = []
        improvement_by_dof = []
        for dof_name in dof_names:
            pred_series = np.asarray([float(metrics[quantity]["pred_joint_rmse"][dof_name]["rmse"]) for metrics in metrics_list], dtype=np.float64)
            input_series = np.asarray([float(metrics[quantity]["input_joint_rmse"][dof_name]["rmse"]) for metrics in metrics_list], dtype=np.float64)
            pred_by_dof.append(float(np.mean(pred_series)))
            input_by_dof.append(float(np.mean(input_series)))
            improvement_by_dof.append(float(np.mean(input_series - pred_series)))

        axes2[q_idx].bar(x - width, input_by_dof, width=width, color="#2ca02c", label="Input")
        axes2[q_idx].bar(x, pred_by_dof, width=width, color="#d62728", label="Pred")
        axes2[q_idx].bar(x + width, improvement_by_dof, width=width, color="#6a3d9a", label="Improvement")
        axes2[q_idx].set_title(f"Per-DOF mean {quantity_titles[quantity]} RMSE and improvement")
        axes2[q_idx].set_ylabel(quantity_labels[quantity])
        axes2[q_idx].grid(True, axis="y", alpha=0.25)
        axes2[q_idx].legend(fontsize=8, ncol=3)
        axes2[q_idx].set_xticks(x)
        axes2[q_idx].set_xticklabels(dof_names, rotation=45, ha="right", fontsize=7)
    per_dof_path = output_dir / "open_capval_per_dof_summary.png"
    fig2.suptitle("OpenCapVal Per-DOF Summary", fontsize=16, fontweight="bold")
    fig2.savefig(per_dof_path, dpi=180)
    plt.close(fig2)
    outputs["per_dof_png"] = str(per_dof_path)

    summary_path = output_dir / "open_capval_summary.json"
    summary_payload: Dict[str, Any] = {
        "n_records": len(records),
        "n_subjects": len(per_subject),
        "quantities": {},
        "per_subject": {},
    }
    for quantity in quantities:
        pred_vals = np.asarray([row[f"{quantity}_pred_rmse"] for row in rows], dtype=np.float64)
        input_vals = np.asarray([row[f"{quantity}_input_rmse"] for row in rows], dtype=np.float64)
        improvement_vals = np.asarray([row[f"{quantity}_improvement"] for row in rows], dtype=np.float64)
        summary_payload["quantities"][quantity] = {
            "pred_mean_rmse": float(np.mean(pred_vals)),
            "input_mean_rmse": float(np.mean(input_vals)),
            "mean_rmse_improvement": float(np.mean(improvement_vals)),
            "median_rmse_improvement": float(np.median(improvement_vals)),
            "best_improvement": float(np.max(improvement_vals)),
            "worst_improvement": float(np.min(improvement_vals)),
        }
    for subject, subject_rows in per_subject.items():
        summary_payload["per_subject"][subject] = {}
        for quantity in quantities:
            summary_payload["per_subject"][subject][quantity] = {
                "pred_mean_rmse": float(np.mean([row[f"{quantity}_pred_rmse"] for row in subject_rows])),
                "input_mean_rmse": float(np.mean([row[f"{quantity}_input_rmse"] for row in subject_rows])),
                "mean_rmse_improvement": float(np.mean([row[f"{quantity}_improvement"] for row in subject_rows])),
            }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    outputs["summary_json"] = str(summary_path)
    return outputs


def save_prediction_bundle(
    output_path: Path,
    predictions: Dict[str, np.ndarray],
    dof_names: Sequence[str] | Dict[str, Sequence[str]],
    sample_id: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantity_dof_names = _normalize_quantity_dof_names(dof_names)
    np.savez(
        output_path,
        sample_id=np.asarray(sample_id),
        dof_names=np.asarray(list(quantity_dof_names["pos"])),
        pos_dof_names=np.asarray(list(quantity_dof_names["pos"])),
        vel_dof_names=np.asarray(list(quantity_dof_names["vel"])),
        acc_dof_names=np.asarray(list(quantity_dof_names["acc"])),
        **{key: np.asarray(value) for key, value in predictions.items()},
    )
    return output_path


def serialize_normalizers(normalizers: Dict[str, Normalizer]) -> Dict[str, Dict[str, np.ndarray]]:
    payload: Dict[str, Dict[str, np.ndarray]] = {}
    for key, normalizer in normalizers.items():
        payload[key] = {
            "mean": np.asarray(normalizer.mean, dtype=np.float32),
            "std": np.asarray(normalizer.std, dtype=np.float32),
        }
    return payload


def deserialize_normalizers(payload: Dict[str, Dict[str, Any]]) -> Dict[str, Normalizer]:
    out: Dict[str, Normalizer] = {}
    for key, item in payload.items():
        out[key] = Normalizer(
            np.asarray(item["mean"], dtype=np.float32),
            np.asarray(item["std"], dtype=np.float32),
        )
    return out


def filter_records_for_selector(records: Sequence[RefineQRecord], selector: str) -> List[RefineQRecord]:
    selector_norm = str(selector).strip().strip("/")
    if not selector_norm:
        return list(records)
    exact = [record for record in records if record.sample_id == selector_norm]
    if exact:
        return exact
    prefix = selector_norm + "/"
    prefixed = [record for record in records if record.sample_id.startswith(prefix)]
    if prefixed:
        return prefixed
    suffix = "/" + selector_norm
    suffix_matches = [record for record in records if record.sample_id.endswith(suffix) or record.sample_id.endswith(selector_norm)]
    return suffix_matches
