"""Loso_Combined.py — Three-stage LOSO over OpenCapSubjects.

Stage 1 (refine-q fine-tuning):
    For each held-out subject, fine-tune the pre-trained ``QRefineTransformer``
    (``TransformerFinal/train_refine_q.py``) on the remaining subjects.  The
    pre-trained checkpoint is supplied via ``refine_q_base_checkpoint``; its
    architecture hyperparameters (d_model, num_heads, num_layers, ff_dim,
    dropout_rate) are read **directly from the checkpoint** so the fine-tuned
    model exactly mirrors the pre-trained one.  Fine-tuning runs for 5 epochs
    (configurable via ``refine_q.epochs``).  After fine-tuning, inference
    is run on every trial (train / val / held-out) in the fold to produce
    "non-noised" 16-column refined joint-angle predictions (q_prime).
    The refine transformer takes **(pos, vel, acc)** concatenated per-frame inputs
    (see ``train_refine_q.INPUT_DIM``).

Stage 2 (per-trial physics regeneration):
    Filter the refined positions with a 6 Hz, 4th-order zero-phase Butterworth
    low-pass, differentiate twice to obtain velocity and acceleration, build the
    31-DOF MuJoCo qpos by patching the refined values into the per-trial qpos
    scaffold, apply XML coupled-coordinate constraints, and re-run the MuJoCo
    forward+inverse-dynamics chain.  This produces the following *inputs* for
    Stage 3, all derived from the MJX physics simulation on refined kinematics:
        - Jacobians at calcn_r / calcn_l
        - WorldToGroundAlignedCalcnRotation (COP rotation matrix)
        - qfrc_inverse  (= qfrc_inverse_only + qfrc_constraint)
    GRF / Moment ground truth is unchanged and reused from the original MoCap
    directory.

Stage 3 (transformer LOSO fine-tuning):
    For each held-out subject, run the transformer LOSO via
    ``loso_from_checkpoint._run_fold`` with ``input_source="processed"`` and
    ``opencap_val=True``.  This means:
        • *Model inputs*: Jacobians, COP rotation matrix, and qfrc_inverse come
          from the Stage-2 MJX-physics cache (refined kinematics).
        • *Ground-truth targets*: qfrc_inverse, Jacobians, and
          WorldToGroundAlignedCalcnRotation come from the MoCap folder
          (symlinked into the cache), unchanged from the original capture.

Hyperparameters for all stages are hard-coded in ``COMBINED_LOSO_CONFIG``
below.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import random
import zlib
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping so we can import sibling packages without a setup.py.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
TRANSFORMER_DIR = REPO_ROOT / "TransformerFinal"
# NoiseModels lives under TransformerFinal/ in this repo layout; fall back to a
# top-level NoiseModels/ if that ever exists.
_NOISE_MODELS_CANDIDATES = (
    TRANSFORMER_DIR / "NoiseModels",
    REPO_ROOT / "NoiseModels",
)
NOISE_MODELS_DIR = next((p for p in _NOISE_MODELS_CANDIDATES if p.is_dir()),
                       _NOISE_MODELS_CANDIDATES[0])
# Insert in reverse-priority order because each entry is placed at sys.path[0].
# Stage 1 depends on TransformerFinal/train_refine_q.py's lightweight API.
for _path in (NOISE_MODELS_DIR, REPO_ROOT, TRANSFORMER_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# ---------------------------------------------------------------------------
# Hard-coded combined LOSO configuration.
# ---------------------------------------------------------------------------
COMBINED_LOSO_CONFIG: Dict[str, Any] = {
    # Dataset / output paths
    "data_dir": "Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects_NoTrim_NoFilt",
    "output_dir": "outputs/loso_combined",
    "refine_q_base_checkpoint": None,  # Optional pre-trained .pkl checkpoint to fine-tune from.
    "transformer_checkpoint": "outpus/OlderAdultModel_128Window_UnFiltered_UnTrimmed/best_model.pkl",    # REQUIRED at runtime — pre-trained transformer .pkl checkpoint.

    # Reproducibility
    "seed": 42,
    # If True: run only Stage 1 (refine-q LOSO) and skip Stage 2/3.
    "stage1_only": False,
    # If True: skip Stage 1/2 and run Stage 3 on original ProcessedData files.
    "stage3_UsePrecomputed": True,
    # If True: after Stage 1, regenerate Stage-2 physics from q_prime and
    # compare refined vs original ProcessedData physics inputs against MoCap GT.
    "stage2_computeJotQfrcRot_accuracy": False,

    # Stage 1 — refine-q LOSO fine-tuning
    # When `refine_q_base_checkpoint` is set the architecture params below
    # (d_model, num_heads, num_layers, ff_dim, dropout_rate) are IGNORED:
    # they are automatically read from the checkpoint so the fine-tuned model
    # exactly mirrors the pre-trained one.  They serve as fallback defaults
    # only when no checkpoint is supplied (training from scratch).
    "refine_q": {
        "learning_rate": 1e-6,
        "epochs": 4,           # Fine-tune epochs per fold
        # Loss
        # Effective objective in this orchestrator:
        #   pos_loss_weight * recon_loss + reg_loss_weight * (lambda_reg * reg_loss)
        # This is implemented by folding weights into an effective lambda:
        #   lambda_eff = lambda_reg * reg_loss_weight / pos_loss_weight
        # so you can heavily prioritize direct position reconstruction.
        "pos_loss_weight": 1,
        "reg_loss_weight": 0,
        "lambda_reg": 0,
        "reg_fade_epochs": 0,
        # Optional differentiable MJX physics losses during Stage-1 refine-q
        # fine-tuning. These are ignored when stage1_only=True.
        "qfrc_inverse_loss_weight": 0.0,
        "jacobian_loss_weight": 0,
        "rotation_loss_weight": 0,
        # When True: on every Stage-1 fold, sample ``trusted_normalizer_num_windows`` random windows
        # from ``trusted_normalizer_data_dir`` and fit pos/vel/acc mean+std (saved as
        # stage1_refine_q/equiv_kinematic_normalizers.json). Independent of loss weights: fitting always
        # runs when this flag is True. pos_std scales Stage-1 recon when physics losses are off; vel/acc
        # stds participate in kinematic-equiv terms when differentiable physics losses are active.
        "use_train_dataset_normalizers": False,
        "trusted_normalizer_data_dir": "TrustedDataSetNoised12Distributed",
        "trusted_normalizer_num_windows": 10000,
        # If None, uses the global ``COMBINED_LOSO_CONFIG['seed']`` (plus fold tag in logs only).
        "trusted_normalizer_sample_seed": None,
        # True = one stacked batch per epoch (non-physics: all train windows at once;
        # physics: one batch per MuJoCo XML with all its windows).
        "one_batch": False,
        # Sliding-window batch size for Stage-1 refine-q fine-tuning.
        # If set to None, the value is taken from the refine-q checkpoint's
        # `hyperparameters.json` (when available) or the hard-coded defaults.
        "batch_size": 32,
    },

    # Stage 2 — physics regeneration from refined kinematics
    "physics": {
        "fs": 100.0,                 # Sampling rate (Hz) — must match ProcessData
        "filter_cutoff_hz": 6.0,
        "filter_order": 4,
        "id_chunk_size": 200,
        "grf_contact_threshold": 30.0,
        "cop_trim_start_frames": 5,
        "cop_trim_end_frames": 5,
        "cop_filter_pad_width": 15,
        "filter_refined_kinematics": False,
    },

    # Stage 3 — transformer LOSO fine-tuning.
    # All other model/training hyperparameters are read from the transformer's
    # sibling hyperparameters.json. Set any value here to None to use that JSON.
    "transformer_loso": {
        "epochs": 2,
        "learning_rate": 5e-5,
        "weight_decay": 0.001,
        "torque_weight": None,
        "grf_weight": None,
        "cop_weight": None,
        "UseGRFNormCOP": None,
        "includeJacobianInput": None,
    },
}


# ---------------------------------------------------------------------------
# Module imports that depend on sys.path being set first.
# ---------------------------------------------------------------------------
# Lazy / late import the heavy modules so simple --help / config inspection is fast.
def _lazy_imports() -> Dict[str, Any]:
    # Quiet noisy XLA GPU autotuning warnings (dot_search_space, etc.) unless user set explicitly.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import jax
    import jax.numpy as jnp
    import optax  # noqa: F401  (used inside imported helpers)
    from flax.training import train_state  # noqa: F401

    import mujoco
    from mujoco import mjx  # noqa: F401
    from scipy.signal import butter, filtfilt  # noqa: F401  (used inside ProcessData helpers)
    from scipy.spatial.transform import Rotation as R

    process_data = importlib.import_module("ProcessData")
    train_refine_q = importlib.import_module("train_refine_q")
    loso_from_checkpoint = importlib.import_module("loso_from_checkpoint")
    train_module = importlib.import_module("train")
    data_loader_module = importlib.import_module("data_loader")
    mod_q_shared = importlib.import_module("mod_q_shared")

    return {
        "jax": jax,
        "jnp": jnp,
        "mujoco": mujoco,
        "mjx": mjx,
        "R": R,
        "process_data": process_data,
        "train_refine_q": train_refine_q,
        "loso_from_checkpoint": loso_from_checkpoint,
        "train_module": train_module,
        "data_loader_module": data_loader_module,
        "mod_q_shared": mod_q_shared,
    }


# ---------------------------------------------------------------------------
# Logging helpers.
# ---------------------------------------------------------------------------
def _ts_print(*parts: Any) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}]", *parts, flush=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Refined-pos -> 31-DOF MuJoCo qpos column mapping.
#
# pos_inputs.npy columns are documented in NoiseModels/refine_q_shared.py as
# OPENCAP_POS_INPUT_IDXS, which gives the column indices (in the 23-DOF
# STANDARD_DOF_NAMES OpenSim layout) that the 16-column pos_inputs vector
# corresponds to.  The 23-column → 31-DOF MuJoCo mapping is NP_TO_QPOS in
# ProcessData.py.  Composing the two yields a refined-col -> qpos-index map.
#
# We embed both arrays here as constants so that this orchestrator is
# self-contained and does not have to import refine_q_shared (which has a
# different runtime layout from train_refine_q).
# ---------------------------------------------------------------------------
OPENCAP_POS_INPUT_IDXS: Tuple[int, ...] = (
    0, 1, 2, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22,
)  # 16 entries — column j of pos_inputs is STANDARD index OPENCAP_POS_INPUT_IDXS[j]

OPENCAP_VEL_INPUT_IDXS: Tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20, 21, 22,
)  # 19 entries — vel_inputs / acc_inputs share this layout

STANDARD_DOF_NAMES: Tuple[str, ...] = (
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
)
POS_INPUT_DOF_NAMES: Tuple[str, ...] = tuple(
    STANDARD_DOF_NAMES[idx] for idx in OPENCAP_POS_INPUT_IDXS
)

NP_TO_QPOS: Dict[int, int] = {
    0: 3,  1: 4,  2: 5,
    3: 0,  4: 1,  5: 2,
    6: 6,  7: 7,  8: 8,
    9: 11, 10: 14, 11: 15, 12: 16, 13: 17, 14: 18, 15: 19, 16: 22,
    17: 25, 18: 26, 19: 27, 20: 28, 21: 29, 22: 30,
}


def _refined_pos_to_qpos_map() -> Dict[int, int]:
    return {
        refined_idx: NP_TO_QPOS[std_idx]
        for refined_idx, std_idx in enumerate(OPENCAP_POS_INPUT_IDXS)
        if std_idx in NP_TO_QPOS
    }


def _qvel_idxs_for_vel_inputs() -> List[int]:
    return [NP_TO_QPOS[std_idx] for std_idx in OPENCAP_VEL_INPUT_IDXS]


QFRC_INVERSE_DOF_NAMES: Tuple[str, ...] = (
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_flexion_r",
    "knee_angle_r_beta",
    "knee_angle_r_beta_translation2",
    "knee_angle_r_beta_translation1",
    "knee_angle_r_beta_rotation2",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_flexion_l",
    "knee_angle_l_beta",
    "knee_angle_l_beta_translation2",
    "knee_angle_l_beta_translation1",
    "knee_angle_l_beta_rotation2",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
)


def _qfrc_inverse_dof_name(index: int) -> str:
    if 0 <= index < len(QFRC_INVERSE_DOF_NAMES):
        return QFRC_INVERSE_DOF_NAMES[index]
    return f"DOF_{index}"


# ---------------------------------------------------------------------------
# Subject discovery + LOSO fold building.
# ---------------------------------------------------------------------------
def _subject_sort_key(name: str) -> Tuple[int, str]:
    import re
    match = re.search(r"(\d+)", str(name))
    if match:
        return int(match.group(1)), str(name)
    return (10**9, str(name))


def _build_loso_folds_from_trials(
    trials: Sequence[Mapping[str, Any]],
    valid_subjects: Sequence[str],
) -> List[Dict[str, Any]]:
    subjects = sorted(valid_subjects, key=_subject_sort_key)
    if len(subjects) < 2:
        raise ValueError("Combined LOSO requires at least 2 valid subjects.")

    subject_to_trials: Dict[str, List[Mapping[str, Any]]] = {s: [] for s in subjects}
    for trial in trials:
        s = str(trial["subject"])
        if s in subject_to_trials:
            subject_to_trials[s].append(trial)

    folds: List[Dict[str, Any]] = []
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        train_trials = [t for s in train_subjects for t in subject_to_trials[s]]
        folds.append({
            "held_out_subject": test_subject,
            "inner_val_subject": None,
            "train_subjects": train_subjects,
            "train_trials": train_trials,
            # No subject is held out for validation. Downstream trainers may use
            # train_trials for internal model selection/bookkeeping, but no
            # additional subject is removed from the 8-subject training split.
            "inner_val_trials": [],
            "held_out_trials": list(subject_to_trials[test_subject]),
        })
    return folds


def _build_precomputed_transformer_buckets(
    fold: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Use original ProcessedData trial paths directly for Stage 3."""
    buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "held_out": []}
    source_map = {
        "train": "train_trials",
        "val": "inner_val_trials",
        "held_out": "held_out_trials",
    }
    for bucket_name, source_key in source_map.items():
        for trial in fold.get(source_key, []):
            training_data_path = Path(str(trial.get("training_data_path", "")))
            if not training_data_path.exists():
                raise FileNotFoundError(
                    f"Precomputed Stage 3 trial path missing: {training_data_path}"
                )
            buckets[bucket_name].append(
                {
                    "subject": str(trial.get("subject", training_data_path.parent.parent.name)),
                    "trial_name": str(
                        trial.get(
                            "trial_name",
                            f"{training_data_path.parent.parent.name}/{training_data_path.parent.name}",
                        )
                    ),
                    "training_data_path": str(training_data_path),
                    "length": int(trial.get("length", 0)),
                }
            )
    return buckets


# ---------------------------------------------------------------------------
# Stage-1 refine-q cache.
#
# ``train_refine_q._load_trial_for_refine`` uses ``pos_inputs_noised.npy`` + ``pos_inputs.npy``
# (GT) on normal datasets.  Loso_Combined writes **unsuffixed** ``pos_inputs.npy`` (OpenCap
# positions), optional ``vel_inputs.npy`` / ``acc_inputs.npy`` (zeros if absent in source),
# plus ``pos_gt.npy`` (MoCap) and ``loso_combined_trial_info.json`` so the loader selects
# the LOSO cache branch without ``*_noised`` filenames on disk. The refine transformer
# uses **16-D positions only** as model input; vel/acc remain available for physics stages.
#
# Layout produced (per fold):
#     <fold_root>/refine_q_dataset/<subject>/
#         Patient_MD.json (symlink/copy from original)
#         PatientSize.npy (symlink/copy from original)
#         <trial>/ProcessedData/
#             pos_inputs.npy  (= OpenCap positions); vel_inputs.npy / acc_inputs.npy optional
#             pos_gt.npy                                     (= original MoCap pos_inputs)
#             Height_m.npy, Mass_kg.npy
#             forwardVel.npy
# ---------------------------------------------------------------------------
def _read_or_none(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        return np.load(path).astype(np.float32)
    except Exception:
        return None


def _link_or_copy(src: Path, dst: Path) -> None:
    """Create dst as a symlink to src; fall back to file copy if the FS rejects symlinks."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            shutil.copy2(src, dst)
        except Exception:
            shutil.copyfile(src, dst)


def _materialize_refine_q_trial(
    src_subject_dir: Path,
    src_trial_dir: Path,
    dst_trial_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Populate one trial under the refine-q cache.

    OpenCap **positions** (required) come from ``src_trial_dir/ProcessedData``;
    MoCap positions for supervision from ``src_trial_dir/MoCap/pos_inputs.npy`` (written
    as ``pos_gt.npy``). Optional OpenCap ``vel_inputs`` / ``acc_inputs`` are copied when
    present; otherwise zero arrays matching ``train_refine_q`` kinematic widths are written
    for physics/aux consumers.

    Returns a trial-info dict with keys (subject, trial_name, training_data_path,
    length) or None if the trial could not be populated.
    """
    import train_refine_q as _trq

    src_processed = src_trial_dir / "ProcessedData"
    src_mocap = src_trial_dir / "MoCap"
    if not src_processed.exists() or not src_mocap.exists():
        return None

    pos_oc = _read_or_none(src_processed / "pos_inputs.npy")
    if pos_oc is None:
        return None

    pos_mc = _read_or_none(src_mocap / "pos_inputs.npy")
    if pos_mc is None:
        return None

    vel_oc = _read_or_none(src_processed / "vel_inputs.npy")
    acc_oc = _read_or_none(src_processed / "acc_inputs.npy")
    if acc_oc is None:
        acc_oc = _read_or_none(src_processed / "accel_inputs.npy")

    height = _read_or_none(src_processed / "Height_m.npy")
    mass = _read_or_none(src_processed / "Mass_kg.npy")
    forward_vel = _read_or_none(src_processed / "forwardVel.npy")
    if height is None or mass is None:
        return None

    T = min(int(pos_oc.shape[0]), int(pos_mc.shape[0]))
    if T < 32:
        return None

    vd = int(_trq.VEL_INPUT_DIM)
    ad = int(_trq.ACC_INPUT_DIM)
    vel_save = np.zeros((T, vd), dtype=np.float32)
    acc_save = np.zeros((T, ad), dtype=np.float32)
    if vel_oc is not None and vel_oc.ndim == 2:
        n = min(T, int(vel_oc.shape[0]))
        c = min(vd, int(vel_oc.shape[1]))
        vel_save[:n, :c] = vel_oc[:n, :c]
    if acc_oc is not None and acc_oc.ndim == 2:
        n = min(T, int(acc_oc.shape[0]))
        c = min(ad, int(acc_oc.shape[1]))
        acc_save[:n, :c] = acc_oc[:n, :c]

    out_processed = dst_trial_dir / "ProcessedData"
    out_processed.mkdir(parents=True, exist_ok=True)

    # Unsuffixed layout: positions required for refine-q; vel/acc optional (zeros OK).
    np.save(out_processed / "pos_inputs.npy", pos_oc[:T].astype(np.float32))
    np.save(out_processed / "vel_inputs.npy", vel_save)
    np.save(out_processed / "acc_inputs.npy", acc_save)
    # Supervision: MoCap joint positions (same 16-D layout as pos_inputs).
    np.save(out_processed / "pos_gt.npy", pos_mc[:T].astype(np.float32))
    np.save(out_processed / "Height_m.npy", np.asarray(height, dtype=np.float32))
    np.save(out_processed / "Mass_kg.npy", np.asarray(mass, dtype=np.float32))
    if forward_vel is not None:
        np.save(out_processed / "forwardVel.npy", np.asarray(forward_vel, dtype=np.float32))

    # Drop stale * _noised.npy from a previous cache so train_refine_q does not
    # mix layouts (OpenCap must stay in unsuffixed inputs; MoCap in pos_gt.npy).
    for stale_name in (
        "pos_inputs_noised.npy",
        "vel_inputs_noised.npy",
        "acc_inputs_noised.npy",
        "forwardVel_noised.npy",
    ):
        stale_path = out_processed / stale_name
        if stale_path.exists():
            try:
                stale_path.unlink()
            except OSError:
                pass

    # Record original trial root so downstream stages can locate the original
    # ProcessedData / MoCap / XML.
    info = {
        "subject": src_subject_dir.name,
        "trial_name": f"{src_subject_dir.name}/{src_trial_dir.name}",
        "training_data_path": str(out_processed),
        "length": int(T),
        "original_trial_dir": str(src_trial_dir),
        "original_subject_dir": str(src_subject_dir),
    }
    _write_json(out_processed / "loso_combined_trial_info.json", info)
    return info


def _materialize_subject_metadata(
    src_subject_dir: Path,
    dst_subject_dir: Path,
) -> None:
    dst_subject_dir.mkdir(parents=True, exist_ok=True)
    for name in ("Patient_MD.json", "PatientSize.npy", "metadata.json", "subject_metadata.json"):
        src = src_subject_dir / name
        if src.exists():
            _link_or_copy(src, dst_subject_dir / name)
    # train_refine_q.py's loader only looks for metadata.json / subject_metadata.json,
    # but OpenCap dataset uses Patient_MD.json — bridge the names so BiologicalSex is
    # picked up correctly during refine-q training/inference.
    src_md = src_subject_dir / "Patient_MD.json"
    if src_md.exists():
        for alias in ("metadata.json", "subject_metadata.json"):
            dst_alias = dst_subject_dir / alias
            if not dst_alias.exists() and not dst_alias.is_symlink():
                _link_or_copy(src_md, dst_alias)


def _build_refine_q_cache(
    fold: Mapping[str, Any],
    fold_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build the refine-q dataset cache for one LOSO fold.

    Returns a dict {"train": [...], "val": [...], "held_out": [...]} where each
    entry is a trial-info dict pointing into the cache.
    """
    cache_root = fold_dir / "refine_q_dataset"
    cache_root.mkdir(parents=True, exist_ok=True)

    bucket_trials: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "held_out": []}
    seen_subjects: set = set()

    for bucket_name, source_key in (
        ("train", "train_trials"),
        ("val", "inner_val_trials"),
        ("held_out", "held_out_trials"),
    ):
        for trial in fold[source_key]:
            src_trial_dir = Path(str(trial.get("training_data_path", ""))).parent
            if src_trial_dir.name == "":
                continue
            src_subject_dir = src_trial_dir.parent
            dst_subject_dir = cache_root / src_subject_dir.name
            if src_subject_dir.name not in seen_subjects:
                _materialize_subject_metadata(src_subject_dir, dst_subject_dir)
                seen_subjects.add(src_subject_dir.name)

            dst_trial_dir = dst_subject_dir / src_trial_dir.name
            info = _materialize_refine_q_trial(src_subject_dir, src_trial_dir, dst_trial_dir)
            if info is not None:
                bucket_trials[bucket_name].append(info)

    _write_json(fold_dir / "refine_q_dataset_split.json", {
        "train_trials": bucket_trials["train"],
        "val_trials": bucket_trials["val"],
        "held_out_trials": bucket_trials["held_out"],
    })
    return bucket_trials


# ---------------------------------------------------------------------------
# Stage-1 refine-q fold trainer.
# ---------------------------------------------------------------------------
def _extract_arch_from_checkpoint(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    """Return architecture hyperparameters stored inside a QRefineTransformer checkpoint.

    Supports two checkpoint layouts:
    * ``train_refine_q.py`` (TransformerFinal) — stores ``ckpt["args"]`` (vars(argparse.Namespace))
    * ``Loso_Combined.py`` fold checkpoint       — stores ``ckpt["config"]`` (refine_q_cfg dict)

    Returns only the keys that directly govern model architecture so that the
    caller can selectively override its working config without disturbing
    optimisation hyper-parameters.
    """
    arch_keys = ("d_model", "num_heads", "num_layers", "ff_dim", "dropout_rate")
    for source_key in ("args", "config"):
        source = ckpt.get(source_key)
        if isinstance(source, dict):
            extracted = {k: source[k] for k in arch_keys if k in source}
            if extracted:
                return extracted
    return {}


def _load_refine_q_defaults_from_checkpoint(
    ckpt: Mapping[str, Any],
    ckpt_path: Path,
    candidate_keys: Sequence[str],
) -> Dict[str, Any]:
    """Load refine-q config defaults from checkpoint payload and sidecar JSON."""
    defaults: Dict[str, Any] = {}

    # Prefer direct checkpoint metadata.
    for source_key in ("config", "args"):
        source = ckpt.get(source_key)
        if isinstance(source, Mapping):
            for key in candidate_keys:
                if key in source:
                    defaults[key] = source.get(key)

    # Require sibling hyperparameters.json for reproducible defaults.
    sidecar_path = ckpt_path.parent / "hyperparameters.json"
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Required refine-q hyperparameters file not found: {sidecar_path}"
        )
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse required refine-q hyperparameters file: {sidecar_path}"
        ) from exc
    if isinstance(sidecar, Mapping):
        for key in candidate_keys:
            if key not in defaults and key in sidecar:
                defaults[key] = sidecar.get(key)

    return defaults


def _resolve_refine_q_effective_config(
    refine_q_cfg: Mapping[str, Any],
    ckpt: Optional[Mapping[str, Any]],
    ckpt_path: Optional[Path],
) -> Dict[str, Any]:
    """Merge refine_q config with checkpoint defaults.

    Rule:
    - Start from checkpoint/hyperparameters defaults when available.
    - Override only with refine_q_cfg entries that are explicitly non-None.
    - If a refine_q_cfg value is None, keep checkpoint default.
    """
    fallback_defaults = {
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 4,
        "ff_dim": 1024,
        "dropout_rate": 0.1,
        "window_size": 64,
        "stride": 16,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "epochs": 5,
        "lambda_reg": 0.01,
        "pos_loss_weight": 1.0,
        "reg_loss_weight": 1.0,
        "reg_fade_epochs": 0,
        "qfrc_inverse_loss_weight": 0.0,
        "jacobian_loss_weight": 0.0,
        "rotation_loss_weight": 0.0,
        "use_train_dataset_normalizers": False,
        "trusted_normalizer_data_dir": "outputs/TrustedDataSetNoised12Distributed/Noised12Distributed_LetsDoThis",
        "trusted_normalizer_num_windows": 1000,
        "trusted_normalizer_sample_seed": None,
        # When True: Stage-1 refine-q yields one batch per epoch (all sliding windows
        # stacked). Non-physics: single optimizer step per epoch. Physics: one step per
        # MuJoCo XML group (all windows sharing that model together; cannot mix XMLs).
        "one_batch": False,
    }

    effective: Dict[str, Any] = {}
    candidate_keys = sorted(set(fallback_defaults) | set(refine_q_cfg.keys()) | {"num_epochs"})
    if ckpt is not None and ckpt_path is not None:
        effective.update(_load_refine_q_defaults_from_checkpoint(ckpt, ckpt_path, candidate_keys))
    for key, value in refine_q_cfg.items():
        if value is not None:
            effective[key] = value
        elif key not in effective:
            effective[key] = value

    # Backward compatibility: older checkpoints/configs may still use num_epochs.
    if ("epochs" not in effective or effective.get("epochs") is None) and (
        "num_epochs" in effective and effective.get("num_epochs") is not None
    ):
        effective["epochs"] = int(effective["num_epochs"])

    # Hard fallback defaults so missing keys never crash stage 1 when config
    # intentionally omits fields to rely on checkpoint-side values.
    for key, default_value in fallback_defaults.items():
        if key not in effective or effective.get(key) is None:
            effective[key] = default_value

    # Keep a single canonical key.
    if "num_epochs" in effective:
        effective.pop("num_epochs", None)

    return effective


def _train_refine_q_for_fold(
    fold: Mapping[str, Any],
    bucket_trials: Mapping[str, List[Dict[str, Any]]],
    refine_q_cfg: Mapping[str, Any],
    fold_dir: Path,
    base_checkpoint: Optional[str],
    seed: int,
    libs: Mapping[str, Any],
    enable_physics_losses: bool = True,
    physics_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fine-tune a QRefineTransformer for one fold.

    When *base_checkpoint* is supplied the function:
      1. Loads the checkpoint produced by ``TransformerFinal/train_refine_q.py``
         (or a previous Loso_Combined fold checkpoint).
      2. Reads the architecture hyperparameters (d_model, num_heads, num_layers,
         ff_dim, dropout_rate) **directly from the checkpoint** so the fine-tuned
         model exactly mirrors the pre-trained one.
      3. Builds the model with those dimensions, initialises a fresh optimiser
         with the fine-tuning learning rate / weight-decay, and loads the
         pre-trained params.
      4. Fine-tunes for ``refine_q_cfg["epochs"]`` epochs (default 5).

    When no checkpoint is provided the architecture falls back to the values in
    *refine_q_cfg* and the model is trained from scratch.
    """
    train_refine_q = libs["train_refine_q"]
    jax = libs["jax"]
    jnp = libs["jnp"]

    output_dir = fold_dir / "stage1_refine_q"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_trials = bucket_trials["train"]
    val_trials = bucket_trials["val"] or train_trials
    if not train_trials:
        raise RuntimeError(
            f"[{fold['held_out_subject']}] no trials available for refine-q training cache; "
            "check that the OpenCap dataset has both ProcessedData/ and MoCap/ directories per trial."
        )
    if not bucket_trials["val"]:
        _ts_print(
            f"[refine-q fold {fold['held_out_subject']}] no separate validation subject; "
            "using training trials for internal validation metrics."
        )

    # ------------------------------------------------------------------
    # Step 1: load the pre-trained checkpoint (if supplied) and extract
    #         architecture dims so the model is built with matching shapes.
    # ------------------------------------------------------------------
    pretrained_params = None
    checkpoint_payload: Optional[Mapping[str, Any]] = None
    checkpoint_path_used: Optional[Path] = None
    effective_arch: Dict[str, Any] = dict(refine_q_cfg)  # mutable working copy

    if base_checkpoint:
        ckpt_path = Path(str(base_checkpoint))
        if ckpt_path.exists():
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            checkpoint_payload = ckpt
            checkpoint_path_used = ckpt_path
            pretrained_params = ckpt.get("params")
            arch_overrides = _extract_arch_from_checkpoint(ckpt)
            if arch_overrides:
                effective_arch.update(arch_overrides)
                _ts_print(
                    f"[refine-q fold {fold['held_out_subject']}] "
                    f"checkpoint arch: {arch_overrides}"
                )
            else:
                _ts_print(
                    f"[refine-q fold {fold['held_out_subject']}] "
                    f"WARN: no architecture keys found in checkpoint; "
                    f"using config defaults"
                )
            _ts_print(
                f"[refine-q fold {fold['held_out_subject']}] "
                f"loaded pre-trained weights from {ckpt_path}"
            )
        else:
            raise FileNotFoundError(
                f"[refine-q fold {fold['held_out_subject']}] "
                f"base checkpoint not found at {ckpt_path}"
            )

    effective_refine_cfg = _resolve_refine_q_effective_config(
        refine_q_cfg,
        checkpoint_payload,
        checkpoint_path_used,
    )
    full_refine_hparams = {
        **dict(effective_refine_cfg),
        "d_model": int(effective_arch["d_model"]),
        "num_heads": int(effective_arch["num_heads"]),
        "num_layers": int(effective_arch["num_layers"]),
        "ff_dim": int(effective_arch["ff_dim"]),
        "dropout_rate": float(effective_arch["dropout_rate"]),
        "input_dim": int(train_refine_q.INPUT_DIM),
        "static_dim": int(train_refine_q.STATIC_DIM),
        "output_dim": int(train_refine_q.OUTPUT_DIM),
        "base_checkpoint": str(checkpoint_path_used) if checkpoint_path_used is not None else None,
    }
    _ts_print(
        f"[refine-q fold {fold['held_out_subject']}] full resolved fine-tune hyperparameters:\n"
        + json.dumps(full_refine_hparams, indent=2, sort_keys=True)
    )
    _ts_print(
        f"[refine-q fold {fold['held_out_subject']}] resolved config: "
        f"d_model={effective_arch.get('d_model')}, "
        f"num_heads={effective_arch.get('num_heads')}, "
        f"num_layers={effective_arch.get('num_layers')}, "
        f"ff_dim={effective_arch.get('ff_dim')}, "
        f"dropout={effective_arch.get('dropout_rate')}, "
        f"window={effective_refine_cfg.get('window_size')}, "
        f"stride={effective_refine_cfg.get('stride')}, "
        f"batch={effective_refine_cfg.get('batch_size')}, "
        f"epochs={effective_refine_cfg.get('epochs')}, "
        f"lr={effective_refine_cfg.get('learning_rate')}, "
        f"wd={effective_refine_cfg.get('weight_decay')}, "
        f"one_batch={bool(effective_refine_cfg.get('one_batch', False))}"
    )

    rng = jax.random.PRNGKey(int(seed))

    # ------------------------------------------------------------------
    # Step 2: data loaders (windowing params from refine_q_cfg, not arch)
    # ------------------------------------------------------------------
    one_batch = bool(effective_refine_cfg.get("one_batch", False))
    loader_kwargs = dict(
        window_size=int(effective_refine_cfg["window_size"]),
        stride=int(effective_refine_cfg["stride"]),
        batch_size=int(effective_refine_cfg["batch_size"]),
        one_batch=one_batch,
    )
    physics_cfg_effective = physics_cfg or COMBINED_LOSO_CONFIG["physics"]
    physics_loss_weights = _refine_q_physics_loss_weights(
        effective_refine_cfg,
        stage1_only=not enable_physics_losses,
    )
    physics_losses_active = bool(enable_physics_losses and any(v > 0.0 for v in physics_loss_weights.values()))
    fixed_equiv_kinematic_stds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if bool(refine_q_cfg.get("use_train_dataset_normalizers", False)):
        td = _resolve_path_relative_to_repo(str(refine_q_cfg.get("trusted_normalizer_data_dir", "")))
        n_w = int(refine_q_cfg.get("trusted_normalizer_num_windows", 1000))
        se_raw = refine_q_cfg.get("trusted_normalizer_sample_seed")
        if se_raw is None:
            fold_tag = str(fold.get("held_out_subject", ""))
            sample_seed = int(seed) + (zlib.adler32(fold_tag.encode("utf-8")) & 0x7FFFFFFF)
        else:
            sample_seed = int(se_raw)
        _ts_print(
            f"[refine-q fold {fold['held_out_subject']}] use_train_dataset_normalizers: "
            f"sampling {n_w} windows (window={effective_refine_cfg['window_size']}, "
            f"stride={effective_refine_cfg['stride']}) from trusted dir {td} (rng_seed={sample_seed})"
        )
        stats = _fit_trusted_equiv_kinematic_normalizers(
            td,
            train_refine_q=train_refine_q,
            window_size=int(effective_refine_cfg["window_size"]),
            stride=int(effective_refine_cfg["stride"]),
            num_windows=n_w,
            sample_seed=sample_seed,
        )
        _write_json(
            output_dir / "equiv_kinematic_normalizers.json",
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()},
        )
        fixed_equiv_kinematic_stds = (
            np.asarray(stats["pos_std"], dtype=np.float32),
            np.asarray(stats["vel_std"], dtype=np.float32),
            np.asarray(stats["acc_std"], dtype=np.float32),
        )
        _ts_print(
            f"[refine-q fold {fold['held_out_subject']}] wrote trusted equiv normalizers to "
            f"{output_dir / 'equiv_kinematic_normalizers.json'}"
        )

    physics_adapter = None
    if physics_losses_active:
        physics_adapter = libs["mod_q_shared"].ModQPhysicsAdapter()
        if not physics_adapter.available:
            raise RuntimeError("refine-q physics losses require mujoco/mjx, but ModQPhysicsAdapter is unavailable.")
        physics_loader_kwargs = {
            **loader_kwargs,
            "fs": float(physics_cfg_effective.get("fs", 100.0)),
            "libs": libs,
        }
        train_loader = _QRefinePhysicsDataLoader(
            train_trials, shuffle=True, **physics_loader_kwargs
        )
        val_loader = (
            _QRefinePhysicsDataLoader(val_trials, shuffle=False, **physics_loader_kwargs)
            if val_trials else None
        )
        _ts_print(
            f"[refine-q fold {fold['held_out_subject']}] differentiable physics losses enabled: "
            f"qfrc_inverse={physics_loss_weights['qfrc_inverse']}, "
            f"jacobian={physics_loss_weights['jacobian']}, "
            f"rotation={physics_loss_weights['rotation']}"
        )
    else:
        train_loader = train_refine_q.QRefineDataLoader(
            train_trials, shuffle=True, **loader_kwargs
        )
        val_loader = (
            train_refine_q.QRefineDataLoader(val_trials, shuffle=False, **loader_kwargs)
            if val_trials else None
        )

    bs_refine = int(effective_refine_cfg["batch_size"])
    if physics_losses_active:
        steps_per_epoch, physics_windows_by_xml = train_loader.describe_epoch_plan()
        if steps_per_epoch <= 0:
            raise RuntimeError(
                f"[refine-q fold {fold['held_out_subject']}] physics loader produced zero batches per epoch "
                "(no valid windows). Check trials, window_size/stride, and MoCap/ProcessedData physics files."
            )
        if one_batch:
            _ts_print(
                f"[refine-q fold {fold['held_out_subject']}] refine_q.one_batch=True: "
                f"one gradient step per MuJoCo model (all windows sharing the same subject_model_xml stacked; "
                f"`batch_size` is ignored for chunking). {len(physics_windows_by_xml)} model group(s)."
            )
        else:
            _ts_print(
                f"[refine-q fold {fold['held_out_subject']}] physics train batching: sliding windows are pooled "
                f"per unique `subject_model_xml` (one MuJoCo model per subject). Each step stacks up to "
                f"batch_size={bs_refine} windows that share the same XML so a single MJX context applies. "
                f"Steps/epoch = sum over models of ceil(n_windows / batch_size) — not one step per subject, "
                f"and not ceil(total_windows / batch_size) across all subjects."
            )
        _ts_print(
            f"[refine-q fold {fold['held_out_subject']}] physics train epoch plan: "
            f"{steps_per_epoch} gradient steps/epoch across {len(physics_windows_by_xml)} distinct model file(s)"
        )
        for xml_path, nw in sorted(
            physics_windows_by_xml.items(),
            key=lambda kv: (-kv[1], str(kv[0])),
        ):
            if one_batch and nw > 0:
                nb = 1
            else:
                nb = int(math.ceil(float(nw) / float(max(1, bs_refine)))) if nw > 0 else 0
            short = Path(str(xml_path)).name
            _ts_print(f"    · {short}: {nw} windows -> {nb} batches (batch_size={bs_refine})")
        if val_loader is not None:
            val_steps, val_map = val_loader.describe_epoch_plan()
            _ts_print(
                f"[refine-q fold {fold['held_out_subject']}] physics val: {val_steps} batches/epoch, "
                f"{len(val_map)} model(s)"
            )
    else:
        steps_per_epoch = max(1, len(train_loader))
        if one_batch:
            _ts_print(
                f"[refine-q fold {fold['held_out_subject']}] refine_q.one_batch=True: "
                f"one train step per epoch with all {train_loader.total_windows} sliding windows stacked "
                f"(batch_size config is ignored for chunking; GPU memory may limit viability)."
            )

    # ------------------------------------------------------------------
    # Step 3: build model with dims from checkpoint (or config fallback)
    # ------------------------------------------------------------------
    model = train_refine_q.QRefineTransformer(
        input_dim=train_refine_q.INPUT_DIM,
        static_dim=train_refine_q.STATIC_DIM,
        output_dim=train_refine_q.OUTPUT_DIM,
        d_model=int(effective_arch["d_model"]),
        num_heads=int(effective_arch["num_heads"]),
        num_layers=int(effective_arch["num_layers"]),
        ff_dim=int(effective_arch["ff_dim"]),
        dropout_rate=float(effective_arch["dropout_rate"]),
    )

    total_steps = max(1, int(steps_per_epoch) * int(effective_refine_cfg["epochs"]))
    warmup_steps = max(1, min(200, total_steps // 10))

    rng, init_rng = jax.random.split(rng)
    state = train_refine_q.create_train_state(
        init_rng,
        model,
        input_shape=(1, int(effective_refine_cfg["window_size"]), train_refine_q.INPUT_DIM),
        static_shape=(1, train_refine_q.STATIC_DIM),
        learning_rate=float(effective_refine_cfg["learning_rate"]),
        weight_decay=float(effective_refine_cfg["weight_decay"]),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    # ------------------------------------------------------------------
    # Step 4: transplant pre-trained params into the fresh train state
    # ------------------------------------------------------------------
    if pretrained_params is not None:
        try:
            state = state.replace(params=pretrained_params)
            _ts_print(
                f"[refine-q fold {fold['held_out_subject']}] "
                f"pre-trained params loaded — fine-tuning for "
                f"{int(effective_refine_cfg['epochs'])} epochs"
            )
        except Exception as exc:
            _ts_print(
                f"[refine-q fold {fold['held_out_subject']}] "
                f"WARN: param transplant failed ({exc}); "
                f"falling back to random init"
            )

    if physics_losses_active:
        train_step_fn = _make_refine_q_physics_step(
            model,
            physics_loss_weights,
            fs=float(physics_cfg_effective.get("fs", 100.0)),
            train=True,
            libs=libs,
            fixed_equiv_kinematic_stds=fixed_equiv_kinematic_stds,
        )
        eval_step_fn = _make_refine_q_physics_step(
            model,
            physics_loss_weights,
            fs=float(physics_cfg_effective.get("fs", 100.0)),
            train=False,
            libs=libs,
            fixed_equiv_kinematic_stds=fixed_equiv_kinematic_stds,
        )
    elif fixed_equiv_kinematic_stds is not None:
        train_step_fn, eval_step_fn = _make_refine_q_trusted_pos_std_weighted_steps(
            model,
            trusted_pos_std=fixed_equiv_kinematic_stds[0],
            libs=libs,
        )
        _ts_print(
            f"[refine-q fold {fold['held_out_subject']}] Stage-1 train/eval JIT: recon loss uses trusted "
            f"pos_std per joint (z-scored MSE in recon_loss; recon_loss_physical_space_mse is raw rad^2 for "
            f"comparison). vel/acc stds from the same file are used only when physics equiv losses are enabled."
        )
    else:
        train_step_fn = train_refine_q.make_train_step(model)
        eval_step_fn = train_refine_q.make_eval_step(model)

    history: List[Dict[str, Any]] = []
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    best_val_loss = math.inf
    best_params = state.params

    num_epochs = int(effective_refine_cfg["epochs"])
    lambda_reg = float(effective_refine_cfg["lambda_reg"])
    pos_loss_weight = max(1e-8, float(effective_refine_cfg.get("pos_loss_weight", 1.0)))
    reg_loss_weight = max(0.0, float(effective_refine_cfg.get("reg_loss_weight", 1.0)))
    reg_fade_epochs = int(effective_refine_cfg["reg_fade_epochs"])
    log_path = output_dir / "training_log.jsonl"

    _ts_print(f"[refine-q fold {fold['held_out_subject']}] training {num_epochs} epochs "
              f"(steps_per_epoch={steps_per_epoch}, train_trials={len(train_trials)}, val_trials={len(val_trials)})")
    if fixed_equiv_kinematic_stds is not None and physics_losses_active:
        equiv_scale_mode = "trusted_dataset (physics equiv)"
    elif fixed_equiv_kinematic_stds is not None:
        equiv_scale_mode = "trusted_dataset (recon z-MSE via pos_std)"
    elif physics_losses_active:
        equiv_scale_mode = "per_batch"
    else:
        equiv_scale_mode = "n/a"
    _ts_print(
        f"[refine-q fold {fold['held_out_subject']}] loss weights: "
        f"pos_loss_weight={pos_loss_weight:.4f}, reg_loss_weight={reg_loss_weight:.4f}, "
        f"base_lambda={lambda_reg:.6f}, "
        f"qfrc_inverse={physics_loss_weights['qfrc_inverse']:.6f}, "
        f"jacobian={physics_loss_weights['jacobian']:.6f}, "
        f"rotation={physics_loss_weights['rotation']:.6f}, "
        f"kinematic_equiv_std={equiv_scale_mode}"
    )

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        if reg_fade_epochs > 0:
            fade_frac = min(1.0, (epoch - 1) / reg_fade_epochs)
            base_lambda_eff = float(lambda_reg * (1.0 - fade_frac))
        else:
            base_lambda_eff = lambda_reg
        effective_lambda = float(base_lambda_eff * (reg_loss_weight / pos_loss_weight))

        train_log_stride = max(1, int(steps_per_epoch) // 10)
        _ts_print(
            f"[refine-q fold {fold['held_out_subject']}] epoch {epoch}/{num_epochs} "
            f"train: starting pass (~{steps_per_epoch} batches, progress every {train_log_stride})"
        )

        train_acc: Dict[str, float] = {}
        train_count = 0
        jax_device_get = jax.device_get
        for batch in train_loader:
            rng, dropout_rng = jax.random.split(rng)
            jax_batch = {k: jnp.asarray(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
            if physics_losses_active:
                assert physics_adapter is not None
                physics_context = physics_adapter.get_jit_context(str(batch["subject_model_xml"]))
                state, metrics = train_step_fn(state, jax_batch, physics_context, dropout_rng, effective_lambda)
            else:
                state, metrics = train_step_fn(state, jax_batch, dropout_rng, effective_lambda)
            _accumulate_metric_values(train_acc, metrics)
            train_count += 1
            metrics_host = jax_device_get(metrics)
            if (
                train_count == 1
                or train_count == int(steps_per_epoch)
                or train_count % train_log_stride == 0
            ):
                xml_b = Path(str(batch.get("subject_model_xml", ""))).name or "—"
                bsz = int(jax_batch["input"].shape[0])
                tl = float(metrics_host.get("total_loss", 0.0))
                rl = float(metrics_host.get("recon_loss", 0.0))
                phys_tail = ""
                if physics_losses_active:
                    eq_sum = float(metrics_host.get("physics_scaled_equiv_pos_sum", 0.0))
                    eq_q = float(metrics_host.get("qfrc_inverse_scaled_equiv_pos_loss", 0.0))
                    eq_j = float(metrics_host.get("jacobian_scaled_equiv_pos_loss", 0.0))
                    eq_r = float(metrics_host.get("rotation_scaled_equiv_pos_loss", 0.0))
                    sq_q = float(metrics_host.get("qfrc_inverse_scaled_equiv_pos_rmse_phys", 0.0))
                    sq_j = float(metrics_host.get("jacobian_scaled_equiv_pos_rmse_phys", 0.0))
                    sq_r = float(metrics_host.get("rotation_scaled_equiv_pos_rmse_phys", 0.0))
                    phys_tail = (
                        f" qfrc={float(metrics_host.get('qfrc_inverse_loss', 0.0)):.4e} "
                        f"jac={float(metrics_host.get('jacobian_loss', 0.0)):.4e} "
                        f"rot={float(metrics_host.get('rotation_loss', 0.0)):.4e} "
                        f"| kineq_pos_MSE(w*grad) sum={eq_sum:.4e} "
                        f"(qfrc={eq_q:.4e} jac={eq_j:.4e} rot={eq_r:.4e}; "
                        f"~RMSE_phys_rad q/j/r={sq_q:.4e}/{sq_j:.4e}/{sq_r:.4e})"
                    )
                _ts_print(
                    f"[refine-q fold {fold['held_out_subject']}] epoch {epoch}/{num_epochs} "
                    f"batch {train_count}/{steps_per_epoch} B={bsz} xml={xml_b} "
                    f"total={tl:.5f} recon={rl:.5f}{phys_tail}"
                )
        if physics_losses_active and train_count != int(steps_per_epoch):
            _ts_print(
                f"[refine-q fold {fold['held_out_subject']}] epoch {epoch} train: "
                f"completed {train_count} batches (planned {steps_per_epoch}; mismatch is unexpected — "
                f"report if reproducible)"
            )
        train_avg = _average_metric_values(train_acc, train_count)
        train_loss_history.append(train_avg["total_loss"])

        if val_loader is not None:
            val_acc: Dict[str, float] = {}
            val_count = 0
            if physics_losses_active:
                v_plan, _ = val_loader.describe_epoch_plan()
                _ts_print(
                    f"[refine-q fold {fold['held_out_subject']}] epoch {epoch}/{num_epochs} "
                    f"val: evaluating (~{v_plan} batches)"
                )
            for batch in val_loader:
                jax_batch = {k: jnp.asarray(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
                if physics_losses_active:
                    assert physics_adapter is not None
                    physics_context = physics_adapter.get_jit_context(str(batch["subject_model_xml"]))
                    metrics = eval_step_fn(state, jax_batch, physics_context, effective_lambda)
                else:
                    metrics = eval_step_fn(state, jax_batch, effective_lambda)
                _accumulate_metric_values(val_acc, metrics)
                val_count += 1
            val_avg = _average_metric_values(val_acc, val_count) if val_count > 0 else dict(train_avg)
            current_val = float(val_avg["total_loss"])
            val_loss_history.append(current_val)
        else:
            val_avg = dict(train_avg)
            current_val = float(train_avg["total_loss"])

        elapsed = time.time() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "train": train_avg,
            "val": val_avg,
            "loss_weights": {
                "pos_loss_weight": pos_loss_weight,
                "reg_loss_weight": reg_loss_weight,
                "lambda_reg_base": lambda_reg,
                "lambda_reg_base_eff": base_lambda_eff,
                **{f"{key}_loss_weight": value for key, value in physics_loss_weights.items()},
            },
            "physics_losses_active": bool(physics_losses_active),
            "lambda_reg_eff": effective_lambda,
            "elapsed_s": elapsed,
        }
        history.append(epoch_record)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record) + "\n")
        eq_tail = ""
        if physics_losses_active:
            teq = float(train_avg.get("physics_scaled_equiv_pos_sum", 0.0))
            trl = float(train_avg.get("recon_loss", 0.0))
            eq_tail = (
                f" kineq_pos_MSE_sum={teq:.5f} (vs recon_MSE={trl:.5f}; "
                f"same construction as train_mod_q scaled_equiv_* on batch std)"
            )
        _ts_print(
            f"[refine-q fold {fold['held_out_subject']}] epoch {epoch:03d}/{num_epochs}",
            f"train={train_avg['total_loss']:.5f} val={current_val:.5f}",
            f"lambda={effective_lambda:.4f} "
            f"qfrc={train_avg.get('qfrc_inverse_loss', 0.0):.5f} "
            f"jac={train_avg.get('jacobian_loss', 0.0):.5f} "
            f"rot={train_avg.get('rotation_loss', 0.0):.5f}{eq_tail} "
            f"({elapsed:.1f}s)",
        )

        if current_val < best_val_loss:
            best_val_loss = current_val
            best_params = state.params
            # Save checkpoint using `args`-style layout so downstream code
            # (including future folds) can recover the architecture via
            # _extract_arch_from_checkpoint().
            with open(output_dir / "best_model.pkl", "wb") as f:
                pickle.dump(
                    {
                        "params": state.params,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "args": {
                            "d_model": int(effective_arch["d_model"]),
                            "num_heads": int(effective_arch["num_heads"]),
                            "num_layers": int(effective_arch["num_layers"]),
                            "ff_dim": int(effective_arch["ff_dim"]),
                            "dropout_rate": float(effective_arch["dropout_rate"]),
                        },
                        "config": dict(effective_refine_cfg),
                    },
                    f,
                )

    _write_json(output_dir / "history.json", {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "best_val_loss": float(best_val_loss),
    })

    return {
        "model": model,
        "params": best_params,
        "best_val_loss": float(best_val_loss),
        "output_dir": str(output_dir),
        "history": history,
        "effective_refine_cfg": dict(effective_refine_cfg),
    }


# ---------------------------------------------------------------------------
# Stage-1 refined inference: predict q_prime for every trial in the fold.
# ---------------------------------------------------------------------------
def _predict_refined_pos_for_trial(
    model: Any,
    params: Any,
    cache_trial_processed: Path,
    refine_q_cfg: Mapping[str, Any],
    libs: Mapping[str, Any],
) -> Optional[np.ndarray]:
    train_refine_q = libs["train_refine_q"]
    jnp = libs["jnp"]

    sample = train_refine_q._load_trial_for_refine(cache_trial_processed)
    if sample is None:
        return None

    pos_n = sample["pos"]   # (T, 16)
    vel_n = sample["vel"]   # (T, 19)
    acc_n = sample["acc"]   # (T, 19)
    T = int(min(pos_n.shape[0], vel_n.shape[0], acc_n.shape[0]))
    pos_n = pos_n[:T]
    vel_n = vel_n[:T]
    acc_n = acc_n[:T]

    static_ctx = np.array([
        float(np.asarray(sample["height"]).reshape(-1)[0]),
        float(np.asarray(sample["mass"]).reshape(-1)[0]),
        float(sample["gender"]),
    ], dtype=np.float32)

    pos_n = pos_n.astype(np.float32)
    vel_n = vel_n.astype(np.float32)
    acc_n = acc_n.astype(np.float32)

    window_size = int(refine_q_cfg["window_size"])
    stride = int(refine_q_cfg["stride"])
    starts = train_refine_q.build_window_start_indices(T, window_size, stride)
    if not starts:
        return None

    delta_sum = np.zeros((T, train_refine_q.OUTPUT_DIM), dtype=np.float32)
    weight_sum = np.zeros((T, 1), dtype=np.float32)
    static_jax = jnp.asarray(static_ctx[None, ...])

    for start in starts:
        end = min(start + window_size, T)
        valid_len = max(0, end - start)
        if valid_len <= 0:
            continue

        window = np.zeros((window_size, train_refine_q.INPUT_DIM), dtype=np.float32)
        window[:valid_len] = np.concatenate(
            [pos_n[start:end], vel_n[start:end], acc_n[start:end]],
            axis=-1,
        )
        delta_q = model.apply(
            {"params": params},
            jnp.asarray(window[None, ...]),
            static_jax,
            train=False,
        )
        delta_np = np.asarray(delta_q, dtype=np.float32)[0, :valid_len]

        mask = np.ones((valid_len, 1), dtype=np.float32)
        delta_sum[start:end] += delta_np * mask
        weight_sum[start:end] += mask

    has_weight = weight_sum[:, 0] > 0
    if np.any(has_weight):
        delta_sum[has_weight] /= weight_sum[has_weight]

    delta_np = delta_sum  # (T, 16)
    return (pos_n + delta_np).astype(np.float32)


def _predict_refined_for_all_trials(
    bucket_trials: Mapping[str, List[Dict[str, Any]]],
    model: Any,
    params: Any,
    refine_q_cfg: Mapping[str, Any],
    output_dir: Path,
    libs: Mapping[str, Any],
) -> Dict[str, Path]:
    """Run refine-q inference on every trial in the fold.  Saves q_prime to
    `<output_dir>/<subject>/<trial>/q_prime.npy` and returns a map keyed by
    trial training_data_path string for downstream stages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    total_trials = int(sum(len(v) for v in bucket_trials.values()))
    processed = 0
    _ts_print(f"  [predict] stage start: {total_trials} trial(s) across buckets")
    for bucket_name, trials in bucket_trials.items():
        _ts_print(f"  [predict] bucket={bucket_name}: {len(trials)} trial(s)")
        for trial in trials:
            processed += 1
            cache_trial_processed = Path(str(trial["training_data_path"]))
            q_prime = _predict_refined_pos_for_trial(
                model,
                params,
                cache_trial_processed,
                refine_q_cfg,
                libs,
            )
            if q_prime is None:
                _ts_print(f"  [predict] skipped {trial['trial_name']}: missing inputs")
                continue
            sub_dir = output_dir / Path(trial["trial_name"])
            sub_dir.mkdir(parents=True, exist_ok=True)
            out_path = sub_dir / "q_prime.npy"
            np.save(out_path, q_prime)
            saved[str(cache_trial_processed)] = out_path
            if processed % 5 == 0 or processed == total_trials:
                _ts_print(f"  [predict] progress {processed}/{total_trials} (saved={len(saved)})")
    _ts_print(f"  [predict] stage complete: saved {len(saved)}/{total_trials} q_prime files")
    return saved


def _compute_refine_q_held_out_joint_error_summary(
    bucket_trials: Mapping[str, List[Dict[str, Any]]],
    refined_pos_paths: Mapping[str, Path],
    libs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare noised-vs-GT and refined-vs-GT on held-out trials per joint."""
    train_refine_q = libs["train_refine_q"]
    held_out_trials = list(bucket_trials.get("held_out", []))
    output_dim = len(POS_INPUT_DOF_NAMES)
    abs_err_noised_sum = np.zeros((output_dim,), dtype=np.float64)
    abs_err_refined_sum = np.zeros((output_dim,), dtype=np.float64)
    frames_accum = 0
    trials_used = 0
    skipped_trials: List[str] = []

    for trial_info in held_out_trials:
        cache_trial_processed = Path(str(trial_info["training_data_path"]))
        sample = train_refine_q._load_trial_for_refine(cache_trial_processed)
        q_prime_path = refined_pos_paths.get(str(cache_trial_processed))
        if sample is None or q_prime_path is None or not q_prime_path.exists():
            skipped_trials.append(str(trial_info.get("trial_name", cache_trial_processed)))
            continue
        try:
            q_prime = np.asarray(np.load(q_prime_path), dtype=np.float32)
        except Exception:
            skipped_trials.append(str(trial_info.get("trial_name", cache_trial_processed)))
            continue

        # OpenCap LOSO cache stores OpenCap/video kinematics in pos_inputs.npy and
        # MoCap positions in pos_gt.npy. train_refine_q loader exposes the input stream as "pos".
        pos_noised_raw = sample.get("pos", sample.get("pos_noised"))
        if pos_noised_raw is None:
            skipped_trials.append(str(trial_info.get("trial_name", cache_trial_processed)))
            continue
        pos_noised = np.asarray(pos_noised_raw, dtype=np.float32)
        pos_gt = np.asarray(sample["pos_gt"], dtype=np.float32)
        T = int(min(pos_noised.shape[0], pos_gt.shape[0], q_prime.shape[0]))
        if T <= 0:
            skipped_trials.append(str(trial_info.get("trial_name", cache_trial_processed)))
            continue

        noised_abs_err = np.abs(pos_noised[:T] - pos_gt[:T])
        refined_abs_err = np.abs(q_prime[:T] - pos_gt[:T])
        abs_err_noised_sum += np.sum(noised_abs_err, axis=0, dtype=np.float64)
        abs_err_refined_sum += np.sum(refined_abs_err, axis=0, dtype=np.float64)
        frames_accum += T
        trials_used += 1

    if frames_accum <= 0:
        return {
            "held_out_trial_count": len(held_out_trials),
            "used_trial_count": 0,
            "frame_count": 0,
            "skipped_trials": skipped_trials,
            "per_joint": {},
            "overall": {},
        }

    noised_mae = abs_err_noised_sum / float(frames_accum)
    refined_mae = abs_err_refined_sum / float(frames_accum)
    mae_delta = noised_mae - refined_mae
    denom = np.where(np.abs(noised_mae) < 1e-12, np.nan, noised_mae)
    mae_delta_percent = (mae_delta / denom) * 100.0

    per_joint: Dict[str, Dict[str, float]] = {}
    for idx, dof_name in enumerate(POS_INPUT_DOF_NAMES):
        per_joint[str(dof_name)] = {
            "mae_noised_vs_gt": float(noised_mae[idx]),
            "mae_refined_vs_gt": float(refined_mae[idx]),
            "mae_improvement": float(mae_delta[idx]),
            "mae_improvement_percent": (
                float(mae_delta_percent[idx]) if np.isfinite(mae_delta_percent[idx]) else float("nan")
            ),
        }

    overall = {
        "mae_noised_vs_gt_mean": float(np.mean(noised_mae)),
        "mae_refined_vs_gt_mean": float(np.mean(refined_mae)),
        "mae_improvement_mean": float(np.mean(mae_delta)),
        "mae_improvement_percent_mean": float(np.nanmean(mae_delta_percent)),
    }

    return {
        "held_out_trial_count": len(held_out_trials),
        "used_trial_count": trials_used,
        "frame_count": int(frames_accum),
        "skipped_trials": skipped_trials,
        "per_joint": per_joint,
        "overall": overall,
    }


def _load_jacobian_payload(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    try:
        payload = np.load(path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if not isinstance(payload, Mapping):
            return None
        out: Dict[str, np.ndarray] = {}
        for key in ("jacp", "jacr", "body_ids"):
            if key in payload:
                out[key] = np.asarray(payload[key])
        if "jacp" not in out and "jacr" not in out:
            return None
        return out
    except Exception:
        return None


def _resolve_original_trial_xml(trial_info: Mapping[str, Any]) -> Optional[Path]:
    original_subject_dir = trial_info.get("original_subject_dir")
    if not original_subject_dir:
        return None
    subject_dir = Path(str(original_subject_dir))
    for xml_name in ("MyosuiteModel_FIXED.xml", "MyosuiteModel.xml"):
        candidate = subject_dir / xml_name
        if candidate.exists():
            return candidate
    return None


def _refine_q_physics_loss_weights(
    refine_q_cfg: Mapping[str, Any],
    *,
    stage1_only: bool = False,
) -> Dict[str, float]:
    if stage1_only:
        return {"qfrc_inverse": 0.0, "jacobian": 0.0, "rotation": 0.0}
    return {
        "qfrc_inverse": max(0.0, float(refine_q_cfg.get("qfrc_inverse_loss_weight", 0.0) or 0.0)),
        "jacobian": max(0.0, float(refine_q_cfg.get("jacobian_loss_weight", 0.0) or 0.0)),
        "rotation": max(0.0, float(refine_q_cfg.get("rotation_loss_weight", 0.0) or 0.0)),
    }


def _refine_q_physics_losses_active(
    refine_q_cfg: Mapping[str, Any],
    *,
    stage1_only: bool = False,
) -> bool:
    return any(value > 0.0 for value in _refine_q_physics_loss_weights(refine_q_cfg, stage1_only=stage1_only).values())


def _finite_diff_np(x: np.ndarray, *, dt: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[0] < 2:
        return np.zeros_like(arr)
    first = (arr[1:2] - arr[0:1]) / float(dt)
    last = (arr[-1:] - arr[-2:-1]) / float(dt)
    if arr.shape[0] == 2:
        return np.concatenate([first, last], axis=0).astype(np.float32)
    middle = (arr[2:] - arr[:-2]) / (2.0 * float(dt))
    return np.concatenate([first, middle, last], axis=0).astype(np.float32)


def _coerce_time_dim(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] >= int(target_len):
        return arr[: int(target_len)]
    if arr.shape[0] == 0:
        pad_value = np.zeros((1,) + arr.shape[1:], dtype=np.float32)
    else:
        pad_value = arr[-1:]
    pad = np.repeat(pad_value, int(target_len) - int(arr.shape[0]), axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)


class _QRefinePhysicsDataLoader:
    """Subject-grouped refine-q loader with differentiable-MJX supervision fields."""

    def __init__(
        self,
        trials: List[Dict[str, Any]],
        *,
        window_size: int,
        stride: int,
        batch_size: int,
        shuffle: bool,
        fs: float,
        libs: Mapping[str, Any],
        one_batch: bool = False,
    ):
        self.trials = list(trials)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.one_batch = bool(one_batch)
        self.fs = float(fs)
        self.libs = libs
        self.train_refine_q = libs["train_refine_q"]
        self._trial_cache: Dict[str, Tuple[Optional[Dict[str, Any]], List[str]]] = {}
        self._cached_epoch_plan: Optional[Tuple[int, Dict[str, int]]] = None
        self._printed_skip_keys: set[str] = set()

    def describe_epoch_plan(self) -> Tuple[int, Dict[str, int]]:
        """Return (train_steps_per_epoch, windows_per_subject_xml_basename).

        Batches are built **per MuJoCo model** (``subject_model_xml``): all sliding
        windows from every trial that shares the same XML are concatenated, then
        chunked in runs of ``batch_size`` windows. One optimizer step = one chunk
        (partial last chunk is still one step). Different subjects (different XML
        paths) never appear in the same batch, because MJX context is per-model.

        ``train_steps_per_epoch`` is the sum over XML groups of
        ``ceil(n_windows_group / batch_size)``, not ``ceil(total_windows / batch_size)``,
        unless ``one_batch`` is True (then one step per XML group with all windows).
        """
        if self._cached_epoch_plan is not None:
            return self._cached_epoch_plan
        items = self._gather_windows_by_xml(
            shuffle_trials=False,
            shuffle_within_each_xml=False,
            shuffle_xml_blocks=False,
        )
        per_xml: Dict[str, int] = {}
        batch_total = 0
        for xml_path, wins in items:
            label = Path(str(xml_path)).name
            nw = len(wins)
            per_xml[str(xml_path)] = nw
            if nw > 0:
                if self.one_batch:
                    batch_total += 1
                else:
                    batch_total += int(math.ceil(float(nw) / float(max(1, self.batch_size))))
        self._cached_epoch_plan = (int(batch_total), per_xml)
        return self._cached_epoch_plan

    def __len__(self) -> int:
        n_batches, _ = self.describe_epoch_plan()
        return max(1, int(n_batches))

    def _gather_windows_by_xml(
        self,
        *,
        shuffle_trials: bool,
        shuffle_within_each_xml: bool,
        shuffle_xml_blocks: bool,
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        by_xml: Dict[str, List[Dict[str, Any]]] = {}
        trial_list = list(self.trials)
        if shuffle_trials:
            random.shuffle(trial_list)
        for trial_info in trial_list:
            trial_data, skip_reasons = self._load_trial(trial_info)
            if trial_data is None:
                skip_key = str(
                    trial_info.get("training_data_path")
                    or trial_info.get("trial_name")
                    or id(trial_info)
                )
                if skip_key not in self._printed_skip_keys:
                    self._printed_skip_keys.add(skip_key)
                    detail = "; ".join(skip_reasons) if skip_reasons else "unknown reason"
                    _ts_print(
                        f"  [QRefinePhysicsDataLoader] skipping "
                        f"{trial_info.get('trial_name', trial_info.get('training_data_path'))}: {detail}"
                    )
                continue
            windows = self._extract_windows(trial_data)
            by_xml.setdefault(str(trial_data["subject_model_xml"]), []).extend(windows)
        for xml_key in list(by_xml.keys()):
            if shuffle_within_each_xml and by_xml[xml_key]:
                random.shuffle(by_xml[xml_key])
        items = list(by_xml.items())
        if shuffle_xml_blocks:
            random.shuffle(items)
        return items

    def _load_trial(self, trial_info: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        cache_key = str(trial_info.get("training_data_path", ""))
        if cache_key in self._trial_cache:
            return self._trial_cache[cache_key]

        reasons: List[str] = []
        cache_processed = Path(cache_key)
        base = self.train_refine_q._load_trial_for_refine(cache_processed)
        xml_path = _resolve_original_trial_xml(trial_info)
        original_trial_dir = trial_info.get("original_trial_dir")

        if base is None:
            reasons.append(
                "refine cache trial unreadable (_load_trial_for_refine failed; check training cache ProcessedData kinematics)"
            )
        if xml_path is None:
            osub = trial_info.get("original_subject_dir", "?")
            reasons.append(
                f"subject XML not found under {osub} (expected MyosuiteModel_FIXED.xml or MyosuiteModel.xml)"
            )
        if not original_trial_dir:
            reasons.append("trial_info missing original_trial_dir")

        qpos_template: Optional[np.ndarray] = None
        qfrc_inverse_gt: Optional[np.ndarray] = None
        gt_rot_w_to_ga: Optional[np.ndarray] = None
        jac_payload: Optional[Dict[str, np.ndarray]] = None

        if original_trial_dir:
            od = Path(str(original_trial_dir))
            src_processed = od / "ProcessedData"
            src_mocap = od / "MoCap"

            def _rel(p: Path) -> str:
                try:
                    return str(p.relative_to(od))
                except ValueError:
                    return str(p)

            p_pos_mjx = src_processed / "pos_mjx.npy"
            if not p_pos_mjx.is_file():
                reasons.append(f"missing physics file: {_rel(p_pos_mjx)}")
            else:
                try:
                    qpos_template = np.load(p_pos_mjx).astype(np.float32)
                except Exception as exc:
                    reasons.append(f"unreadable {_rel(p_pos_mjx)}: {exc}")

            p_qfrc = src_mocap / "qfrc_inverse.npy"
            if not p_qfrc.is_file():
                reasons.append(f"missing physics file: {_rel(p_qfrc)}")
            else:
                try:
                    qfrc_inverse_gt = np.load(p_qfrc).astype(np.float32)
                except Exception as exc:
                    reasons.append(f"unreadable {_rel(p_qfrc)}: {exc}")

            p_rot = src_mocap / "WorldToGroundAlignedCalcnRotation.npy"
            if not p_rot.is_file():
                reasons.append(f"missing physics file: {_rel(p_rot)}")
            else:
                try:
                    gt_rot_w_to_ga = np.load(p_rot).astype(np.float32)
                except Exception as exc:
                    reasons.append(f"unreadable {_rel(p_rot)}: {exc}")

            p_jac = src_mocap / "Jacobian.npy"
            if not p_jac.is_file():
                reasons.append(f"missing physics file: {_rel(p_jac)}")
            else:
                jac_payload = _load_jacobian_payload(p_jac)
                if jac_payload is None:
                    reasons.append(
                        f"invalid {_rel(p_jac)} (expected pickled/Mapping with jacp and jacr arrays)"
                    )
                elif "jacp" not in jac_payload or "jacr" not in jac_payload:
                    reasons.append(f"{_rel(p_jac)} missing jacp or jacr keys after load")

        if base is None or xml_path is None or not original_trial_dir:
            self._trial_cache[cache_key] = (None, reasons)
            return None, reasons

        if qpos_template is None or qfrc_inverse_gt is None or gt_rot_w_to_ga is None or jac_payload is None:
            self._trial_cache[cache_key] = (None, reasons)
            return None, reasons
        if "jacp" not in jac_payload or "jacr" not in jac_payload:
            self._trial_cache[cache_key] = (None, reasons)
            return None, reasons

        original_trial_dir = Path(str(original_trial_dir))
        src_processed = original_trial_dir / "ProcessedData"
        src_mocap = original_trial_dir / "MoCap"

        qvel_template = _read_or_none(src_processed / "qvel_mjx.npy")
        if qvel_template is None:
            qvel_template = _read_or_none(src_processed / "vel_mjx.npy")
        if qvel_template is None:
            qvel_template = np.zeros_like(qpos_template, dtype=np.float32)
        qacc_template = _read_or_none(src_processed / "qacc_mjx.npy")
        if qacc_template is None:
            qacc_template = _read_or_none(src_processed / "acc_mjx.npy")
        if qacc_template is None:
            qacc_template = np.zeros_like(qpos_template, dtype=np.float32)
        qvel_template = np.asarray(qvel_template, dtype=np.float32)
        qacc_template = np.asarray(qacc_template, dtype=np.float32)

        ankle_heights = _read_or_none(src_processed / "ankle_heights.npy")
        if ankle_heights is None:
            ankle_heights = _read_or_none(src_mocap / "ankle_heights.npy")
        if ankle_heights is None:
            ankle_heights = np.zeros((qpos_template.shape[0], 2), dtype=np.float32)

        stream_lengths: Dict[str, int] = {
            "refine_cache.pos": int(base["pos"].shape[0]),
            "refine_cache.vel": int(base["vel"].shape[0]),
            "refine_cache.acc": int(base["acc"].shape[0]),
            "refine_cache.pos_gt": int(base["pos_gt"].shape[0]),
            "original.ProcessedData.pos_mjx": int(qpos_template.shape[0]),
            "original.ProcessedData.qvel_template": int(qvel_template.shape[0]),
            "original.ProcessedData.qacc_template": int(qacc_template.shape[0]),
            "original.MoCap.qfrc_inverse": int(qfrc_inverse_gt.shape[0]),
            "original.MoCap.WorldToGroundAlignedCalcnRotation": int(gt_rot_w_to_ga.shape[0]),
            "original.MoCap.Jacobian.jacp": int(jac_payload["jacp"].shape[0]),
            "original.MoCap.Jacobian.jacr": int(jac_payload["jacr"].shape[0]),
            "original.(ProcessedData|MoCap).ankle_heights": int(np.asarray(ankle_heights).shape[0]),
        }
        T = min(stream_lengths.values())
        if T < self.window_size:
            shortest_keys = sorted(k for k, v in stream_lengths.items() if v == T)
            lens_compact = ", ".join(f"{k}={stream_lengths[k]}" for k in sorted(stream_lengths))
            reasons.append(
                f"aligned sequence too short for physics loader (T={T} < window_size={self.window_size}); "
                f"shortest stream(s): {', '.join(shortest_keys)}; "
                f"frame_counts [{lens_compact}]"
            )
            self._trial_cache[cache_key] = (None, reasons)
            return None, reasons

        height = _coerce_time_dim(np.asarray(base["height"], dtype=np.float32).reshape(-1, 1), T)
        mass = _coerce_time_dim(np.asarray(base["mass"], dtype=np.float32).reshape(-1, 1), T)
        out = {
            "pos": np.asarray(base["pos"], dtype=np.float32)[:T],
            "vel": np.asarray(base["vel"], dtype=np.float32)[:T],
            "acc": np.asarray(base["acc"], dtype=np.float32)[:T],
            "pos_gt": np.asarray(base["pos_gt"], dtype=np.float32)[:T],
            "height": height,
            "mass": mass,
            "gender": float(base.get("gender", 0.5)),
            "qpos_mjx_input": qpos_template[:T],
            "qvel_mjx_input": qvel_template[:T],
            "qacc_mjx_input": qacc_template[:T],
            "qfrc_inverse_gt": qfrc_inverse_gt[:T],
            "jacp": np.asarray(jac_payload["jacp"], dtype=np.float32)[:T],
            "jacr": np.asarray(jac_payload["jacr"], dtype=np.float32)[:T],
            "gt_rot_w_to_ga": gt_rot_w_to_ga[:T],
            "ankle_heights": np.asarray(ankle_heights, dtype=np.float32)[:T],
            "subject_model_xml": str(xml_path),
            "trial_name": str(trial_info.get("trial_name", cache_processed.parent.name)),
        }
        self._trial_cache[cache_key] = (out, [])
        return out, []

    def _extract_windows(self, trial_data: Mapping[str, Any]) -> List[Dict[str, Any]]:
        T = int(trial_data["pos"].shape[0])
        static_ctx = np.array(
            [
                float(np.asarray(trial_data["height"]).reshape(-1)[0]),
                float(np.asarray(trial_data["mass"]).reshape(-1)[0]),
                float(trial_data["gender"]),
            ],
            dtype=np.float32,
        )
        windows: List[Dict[str, Any]] = []
        for start in self.train_refine_q.build_window_start_indices(T, self.window_size, self.stride):
            end = int(start) + self.window_size
            pos_w = trial_data["pos"][start:end].copy()
            vel_w = trial_data["vel"][start:end].copy()
            acc_w = trial_data["acc"][start:end].copy()
            input_w = np.concatenate([pos_w, vel_w, acc_w], axis=-1)
            windows.append(
                {
                    "input": input_w.astype(np.float32),
                    "pos_noised": pos_w.astype(np.float32),
                    "vel_noised": vel_w.astype(np.float32),
                    "acc_noised": acc_w.astype(np.float32),
                    "pos_gt": trial_data["pos_gt"][start:end].astype(np.float32),
                    "static_context": static_ctx,
                    "supervision_mask": np.ones((self.window_size, 1), dtype=np.float32),
                    "qpos_mjx_input": trial_data["qpos_mjx_input"][start:end].astype(np.float32),
                    "qvel_mjx_input": trial_data["qvel_mjx_input"][start:end].astype(np.float32),
                    "qacc_mjx_input": trial_data["qacc_mjx_input"][start:end].astype(np.float32),
                    "qfrc_inverse_gt": trial_data["qfrc_inverse_gt"][start:end].astype(np.float32),
                    "jacp": trial_data["jacp"][start:end].astype(np.float32),
                    "jacr": trial_data["jacr"][start:end].astype(np.float32),
                    "gt_rot_w_to_ga": trial_data["gt_rot_w_to_ga"][start:end].astype(np.float32),
                    "ankle_heights": trial_data["ankle_heights"][start:end].astype(np.float32),
                    "height": trial_data["height"][start:end].astype(np.float32),
                    "mass": trial_data["mass"][start:end].astype(np.float32),
                    "subject_model_xml": str(trial_data["subject_model_xml"]),
                    "trial_name": str(trial_data["trial_name"]),
                }
            )
        return windows

    @staticmethod
    def _collate(windows: List[Dict[str, Any]], subject_model_xml: str) -> Dict[str, Any]:
        array_keys = [
            "input",
            "pos_noised",
            "vel_noised",
            "acc_noised",
            "pos_gt",
            "static_context",
            "supervision_mask",
            "qpos_mjx_input",
            "qvel_mjx_input",
            "qacc_mjx_input",
            "qfrc_inverse_gt",
            "jacp",
            "jacr",
            "gt_rot_w_to_ga",
            "ankle_heights",
            "height",
            "mass",
        ]
        batch = {key: np.stack([w[key] for w in windows]).astype(np.float32) for key in array_keys}
        batch["subject_model_xml"] = subject_model_xml
        batch["trial_name"] = [str(w["trial_name"]) for w in windows]
        return batch

    def __iter__(self):
        items = self._gather_windows_by_xml(
            shuffle_trials=self.shuffle,
            shuffle_within_each_xml=self.shuffle,
            shuffle_xml_blocks=self.shuffle,
        )
        for subject_model_xml, windows in items:
            step = len(windows) if self.one_batch else self.batch_size
            step = max(1, step)
            for start in range(0, len(windows), step):
                chunk = windows[start : start + step]
                if chunk:
                    yield self._collate(chunk, subject_model_xml)


def _finite_diff_jax(x: Any, *, dt: float, jnp: Any) -> Any:
    if int(x.shape[1]) < 2:
        return jnp.zeros_like(x)
    first = (x[:, 1:2] - x[:, 0:1]) / float(dt)
    last = (x[:, -1:] - x[:, -2:-1]) / float(dt)
    if int(x.shape[1]) == 2:
        return jnp.concatenate([first, last], axis=1)
    middle = (x[:, 2:] - x[:, :-2]) / (2.0 * float(dt))
    return jnp.concatenate([first, middle, last], axis=1)


def _refine_q_predicted_vel_acc_from_pos(
    q_prime: Any,
    vel_template: Any,
    acc_template: Any,
    *,
    dt: float,
    jnp: Any,
) -> Tuple[Any, Any]:
    vel_pred = jnp.asarray(vel_template, dtype=q_prime.dtype)
    acc_pred = jnp.asarray(acc_template, dtype=q_prime.dtype)
    vel_from_pos = _finite_diff_jax(q_prime, dt=dt, jnp=jnp)
    acc_from_pos = _finite_diff_jax(vel_from_pos, dt=dt, jnp=jnp)
    vel_std_to_col = {std_idx: col for col, std_idx in enumerate(OPENCAP_VEL_INPUT_IDXS)}
    for pos_col, std_idx in enumerate(OPENCAP_POS_INPUT_IDXS):
        vel_col = vel_std_to_col.get(std_idx)
        if vel_col is None:
            continue
        vel_pred = vel_pred.at[:, :, vel_col].set(vel_from_pos[:, :, pos_col])
        acc_pred = acc_pred.at[:, :, vel_col].set(acc_from_pos[:, :, pos_col])
    return vel_pred, acc_pred


def _refine_q_equiv_mse_metrics_from_grad(grad_slice: Any, scale_std: Any, jnp: Any) -> Dict[str, Any]:
    """Kinematic-channel MSE equivalent of a loss gradient slice (see train_mod_q._equivalent_mse_metrics_from_grad)."""

    grad_arr = jnp.asarray(grad_slice)
    std_vec = jnp.asarray(scale_std, dtype=grad_arr.dtype)
    if std_vec.ndim == 1 and grad_arr.ndim > 1:
        std_arr = std_vec.reshape((1,) * (grad_arr.ndim - 1) + std_vec.shape)
    else:
        std_arr = std_vec
    std_arr = jnp.broadcast_to(std_arr, grad_arr.shape)
    element_count = jnp.maximum(jnp.asarray(grad_arr.size, dtype=grad_arr.dtype), jnp.asarray(1.0, dtype=grad_arr.dtype))
    equiv_error = 0.5 * element_count * grad_arr
    equiv_loss = jnp.mean(jnp.square(equiv_error))
    equiv_rmse = jnp.sqrt(equiv_loss)
    equiv_rmse_phys = jnp.sqrt(jnp.mean(jnp.square(equiv_error * std_arr)))
    grad_l2 = jnp.linalg.norm(grad_arr)
    return {
        "loss": equiv_loss,
        "rmse": equiv_rmse,
        "rmse_phys": equiv_rmse_phys,
        "grad_l2": grad_l2,
    }


def _refine_q_batch_kinematic_stds(batch: Mapping[str, Any], *, dtype: Any, jnp: Any, eps: float = 1e-6):
    """Per-feature std over (batch, time) for pos/vel/acc — scales for equiv MSE (train_mod_q normalizer role)."""

    pos_gt = jnp.asarray(batch["pos_gt"], dtype=dtype)
    vel_n = jnp.asarray(batch["vel_noised"], dtype=dtype)
    acc_n = jnp.asarray(batch["acc_noised"], dtype=dtype)
    pos_std = jnp.maximum(jnp.std(pos_gt, axis=(0, 1)), eps)
    vel_std = jnp.maximum(jnp.std(vel_n, axis=(0, 1)), eps)
    acc_std = jnp.maximum(jnp.std(acc_n, axis=(0, 1)), eps)
    return pos_std, vel_std, acc_std


def _resolve_path_relative_to_repo(path_str: str) -> Path:
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _fit_trusted_equiv_kinematic_normalizers(
    trusted_data_dir: Path,
    *,
    train_refine_q: Any,
    window_size: int,
    stride: int,
    num_windows: int,
    sample_seed: int,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Sample ``num_windows`` random sliding windows from a trusted dataset and fit per-feature mean/std.

    Uses ``train_refine_q.discover_trials`` / ``_load_trial_for_refine`` so layout matches
    ``train_refine_q`` (position inputs + optional vel/acc for equiv stats, MTP stripping on acc when applicable).
    """
    trusted_data_dir = Path(trusted_data_dir).resolve()
    if not trusted_data_dir.is_dir():
        raise FileNotFoundError(f"trusted normalizer data directory not found: {trusted_data_dir}")

    trials = train_refine_q.discover_trials(str(trusted_data_dir))
    wsz = int(window_size)
    st = int(stride)
    nwin = int(num_windows)
    valid = [t for t in trials if int(t.get("length", 0)) >= wsz]
    if not valid:
        raise RuntimeError(
            f"No refine-q trials with length >= window_size={wsz} under {trusted_data_dir} "
            f"(discovered {len(trials)} trials total)."
        )

    rng = random.Random(int(sample_seed))
    starts_fn = train_refine_q.build_window_start_indices
    pos_dim = int(train_refine_q.POS_INPUT_DIM)
    vel_dim = int(train_refine_q.VEL_INPUT_DIM)
    acc_dim = int(train_refine_q.ACC_INPUT_DIM)
    pos_buf = np.zeros((nwin, wsz, pos_dim), dtype=np.float32)
    vel_buf = np.zeros((nwin, wsz, vel_dim), dtype=np.float32)
    acc_buf = np.zeros((nwin, wsz, acc_dim), dtype=np.float32)

    filled = 0
    max_attempts = max(20000, nwin * 100)
    attempts = 0
    while filled < nwin and attempts < max_attempts:
        attempts += 1
        tinfo = rng.choice(valid)
        length = int(tinfo["length"])
        starts = starts_fn(length, wsz, st)
        if not starts:
            continue
        st0 = int(rng.choice(starts))
        sample = train_refine_q._load_trial_for_refine(Path(str(tinfo["training_data_path"])))
        if sample is None:
            continue
        pos = np.asarray(sample["pos"], dtype=np.float32)
        vel = np.asarray(sample["vel"], dtype=np.float32)
        acc = np.asarray(sample["acc"], dtype=np.float32)
        end = st0 + wsz
        if pos.shape[0] < end or vel.shape[0] < end or acc.shape[0] < end:
            continue
        pos_buf[filled] = pos[st0:end]
        vel_buf[filled] = vel[st0:end]
        acc_buf[filled] = acc[st0:end]
        filled += 1

    if filled < nwin:
        raise RuntimeError(
            f"Could only collect {filled}/{nwin} valid windows from {trusted_data_dir} "
            f"after {attempts} attempts (check pos_inputs, pos_gt, optional vel/acc, and MIN_TRIAL_LENGTH)."
        )

    pos_flat = pos_buf.reshape(-1, pos_dim).astype(np.float64, copy=False)
    vel_flat = vel_buf.reshape(-1, vel_dim).astype(np.float64, copy=False)
    acc_flat = acc_buf.reshape(-1, acc_dim).astype(np.float64, copy=False)
    pos_mean = np.mean(pos_flat, axis=0).astype(np.float32)
    vel_mean = np.mean(vel_flat, axis=0).astype(np.float32)
    acc_mean = np.mean(acc_flat, axis=0).astype(np.float32)
    pos_std = np.maximum(np.std(pos_flat, axis=0), eps).astype(np.float32)
    vel_std = np.maximum(np.std(vel_flat, axis=0), eps).astype(np.float32)
    acc_std = np.maximum(np.std(acc_flat, axis=0), eps).astype(np.float32)

    return {
        "trusted_data_dir": str(trusted_data_dir),
        "window_size": wsz,
        "stride": st,
        "num_windows": nwin,
        "sample_seed": int(sample_seed),
        "pos_mean": pos_mean,
        "vel_mean": vel_mean,
        "acc_mean": acc_mean,
        "pos_std": pos_std,
        "vel_std": vel_std,
        "acc_std": acc_std,
        "pos_dim": pos_dim,
        "vel_dim": vel_dim,
        "acc_dim": acc_dim,
    }


def _make_refine_q_trusted_pos_std_weighted_steps(
    model: Any,
    *,
    trusted_pos_std: np.ndarray,
    libs: Mapping[str, Any],
) -> Tuple[Any, Any]:
    """Train/eval steps matching train_refine_q recon+reg, but recon MSE is per-joint z-scored by trusted pos_std.

    Used when trusted kinematic normalizers are fit but Stage-1 physics losses are off, so equiv stds still
    influence optimization. ``recon_loss`` is the weighted objective; ``recon_loss_physical_space_mse`` matches
    the unweighted rad^2 mean used by make_train_step for logging/comparison.
    """

    jax = libs["jax"]
    jnp = libs["jnp"]
    std_vec = jnp.asarray(np.asarray(trusted_pos_std, dtype=np.float32))

    @jax.jit
    def train_step(
        state: Any,
        batch: Mapping[str, Any],
        dropout_rng: Any,
        lambda_reg: Any,
    ):
        safe_std = jnp.maximum(
            std_vec.astype(batch["pos_noised"].dtype),
            jnp.asarray(1e-8, dtype=batch["pos_noised"].dtype),
        )

        def loss_fn(params):
            delta_q = model.apply(
                {"params": params},
                batch["input"],
                batch["static_context"],
                train=True,
                rngs={"dropout": dropout_rng},
            )
            q_prime = batch["pos_noised"] + delta_q
            mask = batch["supervision_mask"]
            diff = q_prime - batch["pos_gt"]
            z_sq = (diff / safe_std) ** 2
            sq_err = jnp.mean(z_sq, axis=-1, keepdims=True)
            n_valid = jnp.maximum(jnp.sum(mask), 1.0)
            recon_loss = jnp.sum(sq_err * mask) / n_valid

            sq_phys = jnp.mean(diff ** 2, axis=-1, keepdims=True)
            recon_loss_physical = jnp.sum(sq_phys * mask) / n_valid

            reg_loss = jnp.mean(delta_q ** 2)
            total_loss = recon_loss + lambda_reg * reg_loss
            metrics = {
                "recon_loss": recon_loss,
                "recon_loss_physical_space_mse": recon_loss_physical,
                "reg_loss": reg_loss,
                "total_loss": total_loss,
            }
            return total_loss, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @jax.jit
    def eval_step(state: Any, batch: Mapping[str, Any], lambda_reg: Any):
        safe_std = jnp.maximum(
            std_vec.astype(batch["pos_noised"].dtype),
            jnp.asarray(1e-8, dtype=batch["pos_noised"].dtype),
        )
        delta_q = model.apply(
            {"params": state.params},
            batch["input"],
            batch["static_context"],
            train=False,
        )
        q_prime = batch["pos_noised"] + delta_q
        mask = batch["supervision_mask"]
        diff = q_prime - batch["pos_gt"]
        z_sq = (diff / safe_std) ** 2
        sq_err = jnp.mean(z_sq, axis=-1, keepdims=True)
        n_valid = jnp.maximum(jnp.sum(mask), 1.0)
        recon_loss = jnp.sum(sq_err * mask) / n_valid
        sq_phys = jnp.mean(diff ** 2, axis=-1, keepdims=True)
        recon_loss_physical = jnp.sum(sq_phys * mask) / n_valid
        reg_loss = jnp.mean(delta_q ** 2)
        total_loss = recon_loss + lambda_reg * reg_loss
        return {
            "recon_loss": recon_loss,
            "recon_loss_physical_space_mse": recon_loss_physical,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }

    return train_step, eval_step


def _make_refine_q_physics_step(
    model: Any,
    physics_loss_weights: Mapping[str, float],
    *,
    fs: float,
    train: bool,
    libs: Mapping[str, Any],
    fixed_equiv_kinematic_stds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
):
    jax = libs["jax"]
    jnp = libs["jnp"]
    mod_q_shared = libs["mod_q_shared"]
    dt = 1.0 / float(fs)
    qfrc_weight = float(physics_loss_weights.get("qfrc_inverse", 0.0))
    jac_weight = float(physics_loss_weights.get("jacobian", 0.0))
    rot_weight = float(physics_loss_weights.get("rotation", 0.0))
    runner = mod_q_shared._GENERIC_MJX_PHYSICS_RUNNER
    reconstructor = mod_q_shared._GENERIC_KINEMATICS_RECONSTRUCTOR

    def _compute_loss(params: Any, batch: Mapping[str, Any], physics_context: Mapping[str, Any], dropout_rng: Any):
        apply_kwargs = {"train": train}
        if train:
            apply_kwargs["rngs"] = {"dropout": dropout_rng}
        delta_q = model.apply(
            {"params": params},
            batch["input"],
            batch["static_context"],
            **apply_kwargs,
        )
        q_prime = jnp.asarray(batch["pos_noised"], dtype=delta_q.dtype) + delta_q
        mask = jnp.asarray(batch["supervision_mask"], dtype=delta_q.dtype)
        sq_err = jnp.mean((q_prime - jnp.asarray(batch["pos_gt"], dtype=delta_q.dtype)) ** 2, axis=-1, keepdims=True)
        n_valid = jnp.maximum(jnp.sum(mask), 1.0)
        recon_loss = jnp.sum(sq_err * mask) / n_valid
        reg_loss = jnp.mean(delta_q**2)

        vel_pred, acc_pred = _refine_q_predicted_vel_acc_from_pos(
            q_prime,
            batch["vel_noised"],
            batch["acc_noised"],
            dt=dt,
            jnp=jnp,
        )

        def _physics_branch_scalars(q: Any, v: Any, a: Any) -> Tuple[Any, Any, Any]:
            flat = q.shape[0] * q.shape[1]
            dtype = q.dtype
            qpos_flat, qvel_flat, qacc_flat = reconstructor(
                q.reshape((flat, q.shape[-1])),
                v.reshape((flat, v.shape[-1])),
                a.reshape((flat, a.shape[-1])),
                jnp.asarray(batch["qpos_mjx_input"], dtype=dtype).reshape(
                    (flat, batch["qpos_mjx_input"].shape[-1])
                ),
                jnp.asarray(batch["qvel_mjx_input"], dtype=dtype).reshape(
                    (flat, batch["qvel_mjx_input"].shape[-1])
                ),
                jnp.asarray(batch["qacc_mjx_input"], dtype=dtype).reshape(
                    (flat, batch["qacc_mjx_input"].shape[-1])
                ),
                jnp.asarray(physics_context["slave_idx"], dtype=jnp.int32),
                jnp.asarray(physics_context["master_idx"], dtype=jnp.int32),
                jnp.asarray(physics_context["coeffs"], dtype=jnp.float32),
            )
            zeros_cop = jnp.zeros((flat, 4), dtype=dtype)
            zeros_grf = jnp.zeros((flat, 6), dtype=dtype)
            zeros_grm = jnp.zeros((flat, 2), dtype=dtype)
            physics_flat = runner(
                physics_context["mjx_model"],
                jnp.asarray(physics_context["calcn_r_id"], dtype=jnp.int32),
                jnp.asarray(physics_context["calcn_l_id"], dtype=jnp.int32),
                qpos_flat,
                qvel_flat,
                qacc_flat,
                zeros_cop,
                zeros_grf,
                zeros_grm,
                jnp.asarray(batch["ankle_heights"], dtype=dtype).reshape((flat, 2)),
            )
            bsh = q.shape[:2]

            def _rp(val: Any) -> Any:
                return val.reshape(bsh + val.shape[1:])

            qfrc_p = _rp(physics_flat["qfrc_inverse"])
            jacp_p = _rp(physics_flat["jacp"])
            jacr_p = _rp(physics_flat["jacr"])
            rot_p = _rp(physics_flat["rot_w_to_ga"])
            mass_b = jnp.asarray(batch["mass"], dtype=dtype)
            height_b = jnp.asarray(batch["height"], dtype=dtype)
            norm_f = jnp.maximum(mass_b * jnp.asarray(9.8067, dtype=dtype) * height_b, 1e-6)
            qf_l = jnp.mean(
                jnp.square((qfrc_p - jnp.asarray(batch["qfrc_inverse_gt"], dtype=dtype)) / norm_f)
            )
            jac_l = (
                jnp.mean(jnp.square(jacp_p - jnp.asarray(batch["jacp"], dtype=dtype)))
                + jnp.mean(jnp.square(jacr_p - jnp.asarray(batch["jacr"], dtype=dtype)))
            )
            rot_l = mod_q_shared.geodesic_rotation_mse(
                rot_p,
                jnp.asarray(batch["gt_rot_w_to_ga"], dtype=dtype),
                mask,
                xp=jnp,
                project=False,
            )
            return qf_l, jac_l, rot_l

        qfrc_loss, jacobian_loss, rotation_loss = _physics_branch_scalars(q_prime, vel_pred, acc_pred)

        total_loss = (
            recon_loss
            + batch["lambda_reg"] * reg_loss
            + jnp.asarray(qfrc_weight, dtype=delta_q.dtype) * qfrc_loss
            + jnp.asarray(jac_weight, dtype=delta_q.dtype) * jacobian_loss
            + jnp.asarray(rot_weight, dtype=delta_q.dtype) * rotation_loss
        )
        metrics: Dict[str, Any] = {
            "recon_loss": recon_loss,
            "reg_loss": reg_loss,
            "qfrc_inverse_loss": qfrc_loss,
            "jacobian_loss": jacobian_loss,
            "rotation_loss": rotation_loss,
            "qfrc_inverse_loss_scaled": jnp.asarray(qfrc_weight, dtype=delta_q.dtype) * qfrc_loss,
            "jacobian_loss_scaled": jnp.asarray(jac_weight, dtype=delta_q.dtype) * jacobian_loss,
            "rotation_loss_scaled": jnp.asarray(rot_weight, dtype=delta_q.dtype) * rotation_loss,
            "total_loss": total_loss,
        }

        if fixed_equiv_kinematic_stds is not None:
            pos_std = jnp.asarray(fixed_equiv_kinematic_stds[0], dtype=delta_q.dtype)
            vel_std = jnp.asarray(fixed_equiv_kinematic_stds[1], dtype=delta_q.dtype)
            acc_std = jnp.asarray(fixed_equiv_kinematic_stds[2], dtype=delta_q.dtype)
        else:
            pos_std, vel_std, acc_std = _refine_q_batch_kinematic_stds(batch, dtype=delta_q.dtype, jnp=jnp)
        term_cfgs = (
            ("qfrc_inverse", qfrc_weight, 0),
            ("jacobian", jac_weight, 1),
            ("rotation", rot_weight, 2),
        )
        scaled_pos_terms: List[Any] = []
        for term_name, weight, loss_idx in term_cfgs:
            w = float(weight)
            if w <= 0.0:
                continue

            def _term_pick(q: Any, v: Any, a: Any, li: int = loss_idx) -> Any:
                trip = _physics_branch_scalars(q, v, a)
                return trip[li]

            g_q, g_v, g_a = jax.grad(_term_pick, argnums=(0, 1, 2))(q_prime, vel_pred, acc_pred)
            w_j = jnp.asarray(w, dtype=delta_q.dtype)
            for comp, g_slice, std_vec in (
                ("pos", g_q, pos_std),
                ("vel", g_v, vel_std),
                ("acc", g_a, acc_std),
            ):
                raw_m = _refine_q_equiv_mse_metrics_from_grad(g_slice, std_vec, jnp)
                scaled_m = _refine_q_equiv_mse_metrics_from_grad(w_j * g_slice, std_vec, jnp)
                metrics[f"{term_name}_raw_equiv_{comp}_loss"] = raw_m["loss"]
                metrics[f"{term_name}_raw_equiv_{comp}_rmse_phys"] = raw_m["rmse_phys"]
                metrics[f"{term_name}_scaled_equiv_{comp}_loss"] = scaled_m["loss"]
                metrics[f"{term_name}_scaled_equiv_{comp}_rmse_phys"] = scaled_m["rmse_phys"]
            scaled_pos_terms.append(metrics[f"{term_name}_scaled_equiv_pos_loss"])

        if scaled_pos_terms:
            metrics["physics_scaled_equiv_pos_sum"] = sum(scaled_pos_terms)
        else:
            metrics["physics_scaled_equiv_pos_sum"] = jnp.asarray(0.0, dtype=delta_q.dtype)

        return total_loss, metrics

    if train:
        @jax.jit
        def train_step(state: Any, batch: Mapping[str, Any], physics_context: Mapping[str, Any], dropout_rng: Any, lambda_reg: float):
            batch = dict(batch)
            batch["lambda_reg"] = jnp.asarray(lambda_reg, dtype=jnp.float32)

            def loss_fn(params):
                return _compute_loss(params, batch, physics_context, dropout_rng)

            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, metrics

        return train_step

    @jax.jit
    def eval_step(state: Any, batch: Mapping[str, Any], physics_context: Mapping[str, Any], lambda_reg: float):
        batch = dict(batch)
        batch["lambda_reg"] = jnp.asarray(lambda_reg, dtype=jnp.float32)
        _, metrics = _compute_loss(state.params, batch, physics_context, None)
        return metrics

    return eval_step


def _accumulate_metric_values(acc: Dict[str, float], metrics: Mapping[str, Any]) -> None:
    for key, value in metrics.items():
        acc[str(key)] = acc.get(str(key), 0.0) + float(value)


def _average_metric_values(acc: Mapping[str, float], count: int) -> Dict[str, float]:
    denom = max(1, int(count))
    return {str(key): float(value) / denom for key, value in acc.items()}


def _nan_percent_improvement(original_error: float, refined_error: float) -> float:
    if not (np.isfinite(original_error) and np.isfinite(refined_error)):
        return float("nan")
    if abs(original_error) <= 1e-12:
        return float("nan")
    return float((original_error - refined_error) / original_error * 100.0)


def _aligned_mae_by_column(candidate: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, int]:
    cand = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(gt, dtype=np.float64)
    if cand.ndim == 1:
        cand = cand[:, None]
    if truth.ndim == 1:
        truth = truth[:, None]
    T = min(int(cand.shape[0]), int(truth.shape[0]))
    C = min(int(cand.shape[1]), int(truth.shape[1]))
    if T <= 0 or C <= 0:
        return np.zeros((0,), dtype=np.float64), 0
    err = np.abs(cand[:T, :C] - truth[:T, :C])
    return np.nanmean(err, axis=0), T


def _rotation_geodesic_error_deg(candidate: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, int]:
    cand = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(gt, dtype=np.float64)
    if cand.ndim != 4 or truth.ndim != 4:
        return np.zeros((0,), dtype=np.float64), 0
    T = min(int(cand.shape[0]), int(truth.shape[0]))
    B = min(int(cand.shape[1]), int(truth.shape[1]))
    if T <= 0 or B <= 0 or cand.shape[-2:] != (3, 3) or truth.shape[-2:] != (3, 3):
        return np.zeros((0,), dtype=np.float64), 0
    errors = np.zeros((T, B), dtype=np.float64)
    for body_idx in range(B):
        rel = np.einsum(
            "tij,tkj->tik",
            cand[:T, body_idx],
            truth[:T, body_idx],
        )
        trace = np.trace(rel, axis1=1, axis2=2)
        cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        errors[:, body_idx] = np.degrees(np.arccos(cos_angle))
    return np.nanmean(errors, axis=0), T


def _jacobian_mae_by_body(
    candidate: Optional[Dict[str, np.ndarray]],
    gt: Optional[Dict[str, np.ndarray]],
) -> Tuple[Dict[str, float], int]:
    if candidate is None or gt is None:
        return {}, 0
    cand_jacp = candidate.get("jacp")
    gt_jacp = gt.get("jacp")
    cand_jacr = candidate.get("jacr")
    gt_jacr = gt.get("jacr")
    if cand_jacp is None or gt_jacp is None or cand_jacr is None or gt_jacr is None:
        return {}, 0
    cand_p = np.asarray(cand_jacp, dtype=np.float64)
    truth_p = np.asarray(gt_jacp, dtype=np.float64)
    cand_r = np.asarray(cand_jacr, dtype=np.float64)
    truth_r = np.asarray(gt_jacr, dtype=np.float64)
    if cand_p.ndim != 4 or truth_p.ndim != 4 or cand_r.ndim != 4 or truth_r.ndim != 4:
        return {}, 0
    T = min(int(cand_p.shape[0]), int(truth_p.shape[0]), int(cand_r.shape[0]), int(truth_r.shape[0]))
    B = min(int(cand_p.shape[1]), int(truth_p.shape[1]), int(cand_r.shape[1]), int(truth_r.shape[1]))
    R = min(int(cand_p.shape[2]), int(truth_p.shape[2]), int(cand_r.shape[2]), int(truth_r.shape[2]))
    C = min(int(cand_p.shape[3]), int(truth_p.shape[3]), int(cand_r.shape[3]), int(truth_r.shape[3]))
    if T <= 0 or B <= 0 or R <= 0 or C <= 0:
        return {}, 0

    body_ids = candidate.get("body_ids")
    if body_ids is None:
        body_ids = gt.get("body_ids")
    if body_ids is not None:
        body_ids_arr = np.asarray(body_ids).reshape(-1)
    else:
        body_ids_arr = np.arange(B)

    out: Dict[str, float] = {}
    for body_idx in range(B):
        combined = np.concatenate(
            [
                cand_p[:T, body_idx, :R, :C] - truth_p[:T, body_idx, :R, :C],
                cand_r[:T, body_idx, :R, :C] - truth_r[:T, body_idx, :R, :C],
            ],
            axis=1,
        )
        body_id = int(body_ids_arr[body_idx]) if body_idx < body_ids_arr.size else body_idx
        out[f"body_{body_id}"] = float(np.nanmean(np.abs(combined)))
    return out, T


def _mean_metric_triplet_from_lists(
    original_values: Sequence[float],
    refined_values: Sequence[float],
) -> Dict[str, float]:
    orig_arr = np.asarray(original_values, dtype=np.float64)
    refined_arr = np.asarray(refined_values, dtype=np.float64)
    valid = np.isfinite(orig_arr) & np.isfinite(refined_arr)
    if not np.any(valid):
        return {
            "original_mae": float("nan"),
            "refined_mae": float("nan"),
            "percent_improvement": float("nan"),
        }
    original_error = float(np.nanmean(orig_arr[valid]))
    refined_error = float(np.nanmean(refined_arr[valid]))
    return {
        "original_mae": original_error,
        "refined_mae": refined_error,
        "percent_improvement": _nan_percent_improvement(original_error, refined_error),
    }


def _compute_stage2_physics_accuracy_summary(
    bucket_trials: Mapping[str, List[Dict[str, Any]]],
    transformer_buckets: Mapping[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    held_refine_trials = bucket_trials.get("held_out", [])
    held_transformer_trials = transformer_buckets.get("held_out", [])
    refined_by_trial = {
        str(trial.get("trial_name")): Path(str(trial.get("training_data_path", "")))
        for trial in held_transformer_trials
    }

    qfrc_orig_by_joint: Dict[str, List[float]] = {}
    qfrc_refined_by_joint: Dict[str, List[float]] = {}
    jac_orig_by_body: Dict[str, List[float]] = {}
    jac_refined_by_body: Dict[str, List[float]] = {}
    rot_orig_by_body: Dict[str, List[float]] = {}
    rot_refined_by_body: Dict[str, List[float]] = {}
    skipped_trials: List[str] = []
    used_trial_count = 0
    frame_count = 0

    for trial in held_refine_trials:
        trial_name = str(trial.get("trial_name"))
        refined_processed = refined_by_trial.get(trial_name)
        if refined_processed is None or not refined_processed.exists():
            skipped_trials.append(f"{trial_name}: missing refined Stage-2 ProcessedData")
            continue
        original_trial_dir = Path(str(trial.get("original_trial_dir", "")))
        original_processed = original_trial_dir / "ProcessedData"
        mocap_dir = original_trial_dir / "MoCap"
        if not original_processed.exists() or not mocap_dir.exists():
            skipped_trials.append(f"{trial_name}: missing original ProcessedData or MoCap")
            continue

        try:
            original_qfrc = np.load(original_processed / "qfrc_inverse.npy")
            refined_qfrc = np.load(refined_processed / "qfrc_inverse.npy")
            gt_qfrc = np.load(mocap_dir / "qfrc_inverse.npy")
        except Exception as exc:
            skipped_trials.append(f"{trial_name}: qfrc load failed: {exc}")
            continue

        orig_qfrc_mae, qfrc_frames = _aligned_mae_by_column(original_qfrc, gt_qfrc)
        refined_qfrc_mae, _ = _aligned_mae_by_column(refined_qfrc, gt_qfrc)
        qfrc_cols = min(int(orig_qfrc_mae.shape[0]), int(refined_qfrc_mae.shape[0]))
        for idx in range(qfrc_cols):
            joint_name = _qfrc_inverse_dof_name(idx)
            qfrc_orig_by_joint.setdefault(joint_name, []).append(float(orig_qfrc_mae[idx]))
            qfrc_refined_by_joint.setdefault(joint_name, []).append(float(refined_qfrc_mae[idx]))

        original_jac = _load_jacobian_payload(original_processed / "Jacobian.npy")
        refined_jac = _load_jacobian_payload(refined_processed / "Jacobian.npy")
        gt_jac = _load_jacobian_payload(mocap_dir / "Jacobian.npy")
        orig_jac_mae, jac_frames = _jacobian_mae_by_body(original_jac, gt_jac)
        refined_jac_mae, _ = _jacobian_mae_by_body(refined_jac, gt_jac)
        for body_name in sorted(set(orig_jac_mae) & set(refined_jac_mae)):
            jac_orig_by_body.setdefault(body_name, []).append(float(orig_jac_mae[body_name]))
            jac_refined_by_body.setdefault(body_name, []).append(float(refined_jac_mae[body_name]))

        try:
            original_rot = np.load(original_processed / "WorldToGroundAlignedCalcnRotation.npy")
            refined_rot = np.load(refined_processed / "WorldToGroundAlignedCalcnRotation.npy")
            gt_rot = np.load(mocap_dir / "WorldToGroundAlignedCalcnRotation.npy")
            orig_rot_geo, rot_frames = _rotation_geodesic_error_deg(original_rot, gt_rot)
            refined_rot_geo, _ = _rotation_geodesic_error_deg(refined_rot, gt_rot)
            rot_bodies = min(int(orig_rot_geo.shape[0]), int(refined_rot_geo.shape[0]))
            for body_idx in range(rot_bodies):
                body_name = "calcn_r" if body_idx == 0 else "calcn_l" if body_idx == 1 else f"body_slot_{body_idx}"
                rot_orig_by_body.setdefault(body_name, []).append(float(orig_rot_geo[body_idx]))
                rot_refined_by_body.setdefault(body_name, []).append(float(refined_rot_geo[body_idx]))
        except Exception as exc:
            skipped_trials.append(f"{trial_name}: rotation load/eval failed: {exc}")
            rot_frames = 0

        used_trial_count += 1
        frame_count += max(int(qfrc_frames), int(jac_frames), int(rot_frames))

    qfrc_per_joint = {
        joint_name: _mean_metric_triplet_from_lists(
            qfrc_orig_by_joint.get(joint_name, []),
            qfrc_refined_by_joint.get(joint_name, []),
        )
        for joint_name in sorted(set(qfrc_orig_by_joint) | set(qfrc_refined_by_joint))
    }
    jac_per_body = {
        body_name: _mean_metric_triplet_from_lists(
            jac_orig_by_body.get(body_name, []),
            jac_refined_by_body.get(body_name, []),
        )
        for body_name in sorted(set(jac_orig_by_body) | set(jac_refined_by_body))
    }
    rot_per_body = {
        body_name: _mean_metric_triplet_from_lists(
            rot_orig_by_body.get(body_name, []),
            rot_refined_by_body.get(body_name, []),
        )
        for body_name in sorted(set(rot_orig_by_body) | set(rot_refined_by_body))
    }
    return {
        "held_out_trial_count": len(held_refine_trials),
        "used_trial_count": used_trial_count,
        "frame_count": frame_count,
        "skipped_trials": skipped_trials,
        "qfrc_inverse_per_joint_mae": qfrc_per_joint,
        "jacobian_per_body_mae": jac_per_body,
        "rotation_per_body_geodesic_deg": rot_per_body,
    }


def _accumulate_stage2_rolling(
    rolling_store: Dict[str, Dict[str, Dict[str, float]]],
    stage2_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    section_map = {
        "qfrc_inverse_per_joint_mae": "qfrc_inverse_per_joint_mae",
        "jacobian_per_body_mae": "jacobian_per_body_mae",
        "rotation_per_body_geodesic_deg": "rotation_per_body_geodesic_deg",
    }
    for summary_key, store_key in section_map.items():
        metrics = stage2_summary.get(summary_key, {})
        if not isinstance(metrics, Mapping):
            continue
        section_store = rolling_store.setdefault(store_key, {})
        for metric_name, values in metrics.items():
            if not isinstance(values, Mapping):
                continue
            orig = float(values.get("original_mae", float("nan")))
            refined = float(values.get("refined_mae", float("nan")))
            pct = float(values.get("percent_improvement", float("nan")))
            if not (np.isfinite(orig) and np.isfinite(refined) and np.isfinite(pct)):
                continue
            accum = section_store.setdefault(
                str(metric_name),
                {
                    "original_sum": 0.0,
                    "refined_sum": 0.0,
                    "percent_improvement_sum": 0.0,
                    "count": 0.0,
                },
            )
            accum["original_sum"] += orig
            accum["refined_sum"] += refined
            accum["percent_improvement_sum"] += pct
            accum["count"] += 1.0

    rolling_summary: Dict[str, Any] = {}
    for section_key, section_store in rolling_store.items():
        rolling_summary[section_key] = {}
        for metric_name, accum in section_store.items():
            count = float(accum.get("count", 0.0))
            if count <= 0:
                continue
            rolling_summary[section_key][metric_name] = {
                "rolling_original_mae": float(accum["original_sum"] / count),
                "rolling_refined_mae": float(accum["refined_sum"] / count),
                "rolling_percent_improvement": float(accum["percent_improvement_sum"] / count),
                "fold_count": int(count),
            }
    return rolling_summary


# ---------------------------------------------------------------------------
# Stage-2 physics regeneration helper.
# ---------------------------------------------------------------------------
def _butter_lp(data: np.ndarray, cutoff: float, fs: float, order: int) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    if data.size == 0:
        return data
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    if data.ndim == 1:
        return filtfilt(b, a, data).astype(np.float32, copy=False)
    out = np.empty_like(data, dtype=np.float32)
    for col in range(data.shape[1]):
        out[:, col] = filtfilt(b, a, data[:, col]).astype(np.float32, copy=False)
    return out


def _safe_diff_filter(data: np.ndarray, dt: float, cutoff: float, fs: float, order: int) -> np.ndarray:
    derivative = np.gradient(np.asarray(data, dtype=np.float64), dt, axis=0)
    return _butter_lp(derivative.astype(np.float32, copy=False), cutoff, fs, order)


def _build_qpos_from_refined(refined_pos_16: np.ndarray, base_pos_mjx_31: np.ndarray) -> np.ndarray:
    qpos = np.asarray(base_pos_mjx_31, dtype=np.float32).copy()
    refined_pos_16 = np.asarray(refined_pos_16, dtype=np.float32)
    T = min(int(qpos.shape[0]), int(refined_pos_16.shape[0]))
    qpos = qpos[:T]
    refined = refined_pos_16[:T]
    refined_to_qpos = _refined_pos_to_qpos_map()
    for refined_idx, qpos_idx in refined_to_qpos.items():
        qpos[:, qpos_idx] = refined[:, refined_idx]
    return qpos


def _forward_kinematics_features(
    mj_model: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    libs: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute per-frame body kinematics features used downstream."""
    mujoco = libs["mujoco"]

    body_names = (
        "pelvis", "torso",
        "femur_r", "tibia_r", "calcn_r", "toes_r",
        "femur_l", "tibia_l", "calcn_l", "toes_l",
    )
    body_ids = {name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names}

    T = int(qpos.shape[0])
    nv = int(mj_model.nv)
    calcn_pos_r = np.zeros((T, 3), dtype=np.float64)
    calcn_pos_l = np.zeros((T, 3), dtype=np.float64)
    toes_pos_r = np.zeros((T, 3), dtype=np.float64)
    toes_pos_l = np.zeros((T, 3), dtype=np.float64)
    rot_g_to_r = np.zeros((T, 3, 3), dtype=np.float64)
    rot_g_to_l = np.zeros((T, 3, 3), dtype=np.float64)
    jacp_all = np.zeros((T, 2, 3, nv), dtype=np.float64)
    jacr_all = np.zeros((T, 2, 3, nv), dtype=np.float64)

    data = mujoco.MjData(mj_model)
    nq = int(mj_model.nq)
    qpos_eff = qpos[:, :nq]

    for t in range(T):
        data.qpos[:nq] = qpos_eff[t]
        if mj_model.nv > 0 and qvel.shape[1] >= mj_model.nv:
            data.qvel[: mj_model.nv] = qvel[t, : mj_model.nv]
        mujoco.mj_kinematics(mj_model, data)
        mujoco.mj_comPos(mj_model, data)

        if body_ids["calcn_r"] >= 0:
            calcn_pos_r[t] = data.xpos[body_ids["calcn_r"]]
            rot_g_to_r[t] = data.xmat[body_ids["calcn_r"]].reshape(3, 3).T
        if body_ids["calcn_l"] >= 0:
            calcn_pos_l[t] = data.xpos[body_ids["calcn_l"]]
            rot_g_to_l[t] = data.xmat[body_ids["calcn_l"]].reshape(3, 3).T
        if body_ids["toes_r"] >= 0:
            toes_pos_r[t] = data.xpos[body_ids["toes_r"]]
        if body_ids["toes_l"] >= 0:
            toes_pos_l[t] = data.xpos[body_ids["toes_l"]]

        for slot, bid in enumerate((body_ids["calcn_r"], body_ids["calcn_l"])):
            if bid < 0:
                continue
            jp = np.zeros((3, nv), dtype=np.float64)
            jr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacBody(mj_model, data, jp, jr, bid)
            jacp_all[t, slot] = jp
            jacr_all[t, slot] = jr

    return {
        "calcn_pos_r": calcn_pos_r.astype(np.float32),
        "calcn_pos_l": calcn_pos_l.astype(np.float32),
        "toes_pos_r": toes_pos_r.astype(np.float32),
        "toes_pos_l": toes_pos_l.astype(np.float32),
        "rot_g_to_r": rot_g_to_r.astype(np.float32),
        "rot_g_to_l": rot_g_to_l.astype(np.float32),
        "jacp": jacp_all.astype(np.float32),
        "jacr": jacr_all.astype(np.float32),
        "body_ids_calcn": np.array(
            [body_ids["calcn_r"], body_ids["calcn_l"]], dtype=np.int32
        ),
    }


def _build_refined_processed_data(
    refined_pos_16: np.ndarray,
    original_trial_dir: Path,
    xml_path: Path,
    output_dir: Path,
    physics_cfg: Mapping[str, Any],
    libs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Generate transformer-ready ProcessedData/ files from refined kinematics.

    Returns a dict with provenance info.  Output files are written under
    `output_dir` (which becomes the trial's new ProcessedData/).
    """
    process_data = libs["process_data"]
    mujoco = libs["mujoco"]
    mjx = libs["mjx"]
    R = libs["R"]

    output_dir.mkdir(parents=True, exist_ok=True)
    src_processed = original_trial_dir / "ProcessedData"
    src_mocap = original_trial_dir / "MoCap"

    base_pos_mjx_path = src_processed / "pos_mjx.npy"
    if not base_pos_mjx_path.exists():
        raise FileNotFoundError(f"missing pos_mjx.npy at {base_pos_mjx_path}")
    base_pos_mjx = np.load(base_pos_mjx_path).astype(np.float32)

    grf_path = src_mocap / "GRF_Cleaned.npy"
    moment_path = src_mocap / "Moment_Cleaned.npy"
    contact_path = src_processed / "contactBoolean.npy"
    height_path = src_processed / "Height_m.npy"
    mass_path = src_processed / "Mass_kg.npy"
    forward_vel_path = src_processed / "forwardVel.npy"

    GRF_mj = np.load(grf_path).astype(np.float32) if grf_path.exists() else None
    Moment_mj = np.load(moment_path).astype(np.float32) if moment_path.exists() else None
    contact_bool = np.load(contact_path).astype(np.float32) if contact_path.exists() else None
    height_arr = np.load(height_path).astype(np.float32) if height_path.exists() else None
    mass_arr = np.load(mass_path).astype(np.float32) if mass_path.exists() else None
    forward_vel = np.load(forward_vel_path).astype(np.float32) if forward_vel_path.exists() else None

    fs = float(physics_cfg["fs"])
    cutoff = float(physics_cfg["filter_cutoff_hz"])
    order = int(physics_cfg["filter_order"])
    dt = 1.0 / fs

    # Build 31-DOF qpos by patching refined master DOFs into the original scaffold.
    qpos = _build_qpos_from_refined(refined_pos_16, base_pos_mjx)
    T = int(qpos.shape[0])
    if T < 8:
        raise ValueError(f"trial too short after column patching: {T}")

    # Filter qpos, then differentiate twice with filtering after each step.
    # When differentiable physics losses trained the refine-q model, keep this
    # path aligned with that objective by avoiding the non-differentiable filter
    # before differentiation.
    if bool(physics_cfg.get("filter_refined_kinematics", True)):
        qpos = _butter_lp(qpos, cutoff=cutoff, fs=fs, order=order)
        qvel = _safe_diff_filter(qpos, dt=dt, cutoff=cutoff, fs=fs, order=order)
        qacc = _safe_diff_filter(qvel, dt=dt, cutoff=cutoff, fs=fs, order=order)
    else:
        qvel = _finite_diff_np(qpos, dt=dt)
        qacc = _finite_diff_np(qvel, dt=dt)

    # XML-driven coupled-coordinate fix-up (slave knee/walker DOFs derived from masters).
    qpos, qvel, qacc = process_data.calculate_coupled_coordinates_automated(
        qpos.astype(np.float64),
        qvel.astype(np.float64),
        qacc.astype(np.float64),
        Path(str(xml_path)),
    )
    qpos = qpos.astype(np.float32)
    qvel = qvel.astype(np.float32)
    qacc = qacc.astype(np.float32)

    # Forward kinematics to extract body positions, COM, Jacobians at calcn_r/l.
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    nq = int(mj_model.nq)
    nv = int(mj_model.nv)
    if qpos.shape[1] != nq:
        if qpos.shape[1] > nq:
            qpos = qpos[:, :nq]
        else:
            pad = np.zeros((T, nq - qpos.shape[1]), dtype=np.float32)
            qpos = np.concatenate([qpos, pad], axis=1)
    if qvel.shape[1] != nv:
        if qvel.shape[1] > nv:
            qvel = qvel[:, :nv]
        else:
            pad = np.zeros((T, nv - qvel.shape[1]), dtype=np.float32)
            qvel = np.concatenate([qvel, pad], axis=1)
    if qacc.shape[1] != nv:
        if qacc.shape[1] > nv:
            qacc = qacc[:, :nv]
        else:
            pad = np.zeros((T, nv - qacc.shape[1]), dtype=np.float32)
            qacc = np.concatenate([qacc, pad], axis=1)

    fk = _forward_kinematics_features(mj_model, qpos, qvel, qacc, libs)

    # The base pos_mjx.npy was already saved AFTER ProcessData applied the
    # floor-height correction (qpos[:,1] -= floor_height), so calcn-Z from
    # forward kinematics is already a height-above-floor.  We still record the
    # original floor height for provenance.
    floor_height = 0.0
    info_path = src_processed / "Trial_Processing_Information.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            floor_height = float(info.get("floor_height_m", 0.0))
        except Exception:
            floor_height = 0.0
    ankle_heights = np.column_stack(
        [fk["calcn_pos_r"][:, 2], fk["calcn_pos_l"][:, 2]]
    ).astype(np.float32)

    # Inverse dynamics + Jacobian-based GRF contribution.
    mjx_model = mjx.put_model(mj_model)
    qfrc_inv, qfrc_con, com_global_id = process_data.compute_inverse_dynamics_chunked(
        mjx_model,
        qpos.astype(np.float64),
        qvel.astype(np.float64),
        qacc.astype(np.float64),
        chunk_size=int(physics_cfg["id_chunk_size"]),
    )
    qfrc_inverse = (np.asarray(qfrc_inv) + np.asarray(qfrc_con)).astype(np.float32)
    qfrc_inverse_only = np.asarray(qfrc_inv, dtype=np.float32)
    qfrc_constraint = np.asarray(qfrc_con, dtype=np.float32)
    com_global = np.asarray(com_global_id, dtype=np.float32)
    com_vel = _safe_diff_filter(com_global, dt=dt, cutoff=cutoff, fs=fs, order=order)
    com_acc_global = _safe_diff_filter(com_vel, dt=dt, cutoff=cutoff, fs=fs, order=order)
    com_r = (com_global[:T] - fk["calcn_pos_r"][:T]).astype(np.float32)
    com_l = (com_global[:T] - fk["calcn_pos_l"][:T]).astype(np.float32)

    # Match ProcessData.py exactly: pelvis_euler = qpos[:, 3:6] in ZXY order,
    # then flatten the first two rotation-matrix columns.
    pelvis_euler = qpos[:T, 3:6].astype(np.float64)
    pelvis_rot_3x3 = R.from_euler("ZXY", pelvis_euler).as_matrix()
    pelvis_rot_matrix = np.concatenate(
        [pelvis_rot_3x3[:, :, 0], pelvis_rot_3x3[:, :, 1]], axis=1
    ).astype(np.float32)

    # COP cleanup + qfrc_grf contribution (uses original GRF/Moment/COP unchanged).
    # The GT COP was rotated to ground-aligned coordinates using the MoCap-side
    # calcn rotation matrix.  To get COP back in world coords we must use that
    # same MoCap rotation matrix, not the refined one.
    cop_path = src_mocap / "COP_CalcFrame_GroundAligned.npy"
    rot_path = src_mocap / "WorldToGroundAlignedCalcnRotation.npy"
    cop_world = None
    if cop_path.exists() and rot_path.exists():
        cop_calc_frame_ground = np.load(cop_path).astype(np.float32)
        rot_w_to_ga_mocap = np.load(rot_path).astype(np.float32)
        if (
            cop_calc_frame_ground.ndim == 2
            and cop_calc_frame_ground.shape[1] >= 6
            and rot_w_to_ga_mocap.ndim == 4
            and rot_w_to_ga_mocap.shape[1:] == (2, 3, 3)
        ):
            T_use = min(T, int(cop_calc_frame_ground.shape[0]), int(rot_w_to_ga_mocap.shape[0]))
            rot_ga_to_w_r = np.transpose(rot_w_to_ga_mocap[:T_use, 0], (0, 2, 1))
            rot_ga_to_w_l = np.transpose(rot_w_to_ga_mocap[:T_use, 1], (0, 2, 1))
            cop_r_world = np.einsum("tij,tj->ti", rot_ga_to_w_r, cop_calc_frame_ground[:T_use, 0:3])
            cop_l_world = np.einsum("tij,tj->ti", rot_ga_to_w_l, cop_calc_frame_ground[:T_use, 3:6])
            cop_world = np.column_stack([cop_r_world, cop_l_world]).astype(np.float32)

    qfrc_grf = np.zeros_like(qfrc_inverse)
    id_gt_mjx = qfrc_inverse.copy()
    cop_rel = np.zeros((T, 4), dtype=np.float32)

    if (
        GRF_mj is not None
        and Moment_mj is not None
        and cop_world is not None
        and GRF_mj.shape[0] >= T
        and Moment_mj.shape[0] >= T
        and cop_world.shape[0] >= T
    ):
        cop_rel, qfrc_grf, id_gt_mjx = process_data.compute_cop_clean_and_id(
            GRF_mj[:T].astype(np.float32),
            Moment_mj[:T].astype(np.float32),
            cop_world[:T].astype(np.float32),
            fk["calcn_pos_r"][:T].astype(np.float32),
            fk["calcn_pos_l"][:T].astype(np.float32),
            {"jacp": fk["jacp"][:T], "jacr": fk["jacr"][:T]},
            qfrc_inverse[:T].astype(np.float32),
            cfg={
                "GRF_CONTACT_THRESHOLD": float(physics_cfg["grf_contact_threshold"]),
                "COP_TRIM_START_FRAMES": int(physics_cfg["cop_trim_start_frames"]),
                "COP_TRIM_END_FRAMES": int(physics_cfg["cop_trim_end_frames"]),
                "COP_FILTER_PAD_WIDTH": int(physics_cfg["cop_filter_pad_width"]),
                "FILTER_CUTOFF_HZ": float(physics_cfg["filter_cutoff_hz"]),
            },
            fs=fs,
        )

    # Build pos_inputs / vel_inputs / acc_inputs in the transformer's column layout.
    refined_to_qpos = _refined_pos_to_qpos_map()
    pos_inputs = np.zeros((T, len(OPENCAP_POS_INPUT_IDXS)), dtype=np.float32)
    for refined_idx, _qpos_idx in refined_to_qpos.items():
        pos_inputs[:, refined_idx] = qpos[:T, _qpos_idx]
    qvel_idxs_for_vel = _qvel_idxs_for_vel_inputs()
    nv_qvel = qvel.shape[1]
    vel_inputs = np.zeros((T, len(qvel_idxs_for_vel)), dtype=np.float32)
    acc_inputs = np.zeros((T, len(qvel_idxs_for_vel)), dtype=np.float32)
    for col, qvel_idx in enumerate(qvel_idxs_for_vel):
        if 0 <= qvel_idx < nv_qvel:
            vel_inputs[:, col] = qvel[:T, qvel_idx]
            acc_inputs[:, col] = qacc[:T, qvel_idx]

    # Save canonical files.
    np.save(output_dir / "pos_inputs.npy", pos_inputs)
    np.save(output_dir / "vel_inputs.npy", vel_inputs)
    np.save(output_dir / "acc_inputs.npy", acc_inputs)
    np.save(output_dir / "pos_mjx.npy", qpos[:T].astype(np.float32))
    np.save(output_dir / "qvel_mjx.npy", qvel[:T].astype(np.float32))
    np.save(output_dir / "qacc_mjx.npy", qacc[:T].astype(np.float32))
    np.save(output_dir / "pelvis_rot_matrix.npy", pelvis_rot_matrix[:T])
    np.save(output_dir / "ankle_heights.npy", ankle_heights[:T])
    np.save(output_dir / "COM_r.npy", com_r[:T])
    np.save(output_dir / "COM_l.npy", com_l[:T])
    np.save(output_dir / "COM_Acc_Global.npy", com_acc_global[:T])
    np.save(output_dir / "ankle_pos_r.npy", fk["calcn_pos_r"][:T])
    np.save(output_dir / "ankle_pos_l.npy", fk["calcn_pos_l"][:T])
    np.save(output_dir / "toes_pos_r.npy", fk["toes_pos_r"][:T])
    np.save(output_dir / "toes_pos_l.npy", fk["toes_pos_l"][:T])
    np.save(output_dir / "COP_Cleaned_Relative.npy", cop_rel[:T])
    np.save(output_dir / "qfrc_inverse.npy", qfrc_inverse[:T])
    np.save(output_dir / "qfrc_inverse_only.npy", qfrc_inverse_only[:T])
    np.save(output_dir / "qfrc_constraint.npy", qfrc_constraint[:T])
    np.save(output_dir / "qfrc_grf_contribution.npy", qfrc_grf[:T])
    np.save(output_dir / "ID_GT_MJX.npy", id_gt_mjx[:T])

    # Pass-through copies for unchanged quantities.
    if contact_bool is not None:
        cb = contact_bool[:T] if contact_bool.shape[0] >= T else np.pad(contact_bool, ((0, T - contact_bool.shape[0]), (0, 0)))
        np.save(output_dir / "contactBoolean.npy", cb.astype(np.float32))
    if GRF_mj is not None:
        np.save(output_dir / "GRF_Cleaned.npy", GRF_mj[:T])
    if Moment_mj is not None:
        np.save(output_dir / "Moment_Cleaned.npy", Moment_mj[:T])
    if height_arr is not None:
        np.save(output_dir / "Height_m.npy", height_arr)
    if mass_arr is not None:
        np.save(output_dir / "Mass_kg.npy", mass_arr)
    if forward_vel is not None:
        np.save(output_dir / "forwardVel.npy", forward_vel)

    # Jacobian.npy is a pickled dict for backward compatibility with data_loader.
    jac_payload = {
        "jacp": fk["jacp"][:T].astype(np.float32),
        "jacr": fk["jacr"][:T].astype(np.float32),
        "body_ids": fk["body_ids_calcn"],
    }
    np.save(output_dir / "Jacobian.npy", jac_payload, allow_pickle=True)

    # WorldToGroundAligned + Foot progression / CalcnToFloor angles via ProcessData helper.
    ok, msg = process_data.generate_calc_frame_outputs_for_source(output_dir, Path(str(xml_path)), output_dir.name)
    if not ok:
        _ts_print(f"  [physics] WARN calc-frame postprocess: {msg}")

    return {
        "T": T,
        "floor_height_m": floor_height,
        "had_grf": GRF_mj is not None,
        "had_cop": cop_world is not None,
        "calc_frame_ok": bool(ok),
        "calc_frame_msg": msg,
    }


# ---------------------------------------------------------------------------
# Stage-2 cache builder.
# ---------------------------------------------------------------------------
def _build_transformer_cache(
    fold: Mapping[str, Any],
    bucket_trials: Mapping[str, List[Dict[str, Any]]],
    refined_pos_paths: Mapping[str, Path],
    fold_dir: Path,
    physics_cfg: Mapping[str, Any],
    libs: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    cache_root = fold_dir / "transformer_dataset"
    cache_root.mkdir(parents=True, exist_ok=True)

    out_buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "held_out": []}
    seen_subjects: set = set()
    failures: List[str] = []
    total_trials = int(sum(len(v) for v in bucket_trials.values()))
    processed = 0
    successes = 0
    _ts_print(f"  [physics] stage start: {total_trials} trial(s) to regenerate")

    for bucket_name in ("train", "val", "held_out"):
        _ts_print(f"  [physics] bucket={bucket_name}: {len(bucket_trials[bucket_name])} trial(s)")
        for trial_info in bucket_trials[bucket_name]:
            processed += 1
            cache_processed = Path(str(trial_info["training_data_path"]))
            q_prime_path = refined_pos_paths.get(str(cache_processed))
            if q_prime_path is None or not q_prime_path.exists():
                failures.append(f"{trial_info['trial_name']}: no q_prime")
                continue

            original_trial_dir = Path(str(trial_info["original_trial_dir"]))
            original_subject_dir = Path(str(trial_info["original_subject_dir"]))

            xml_path = original_subject_dir / "MyosuiteModel_FIXED.xml"
            if not xml_path.exists():
                xml_path = original_subject_dir / "MyosuiteModel.xml"
            if not xml_path.exists():
                failures.append(f"{trial_info['trial_name']}: missing XML at {original_subject_dir}")
                continue

            subject_name = original_subject_dir.name
            dst_subject_dir = cache_root / subject_name
            if subject_name not in seen_subjects:
                _materialize_subject_metadata(original_subject_dir, dst_subject_dir)
                _link_or_copy(xml_path, dst_subject_dir / "MyosuiteModel_FIXED.xml")
                seen_subjects.add(subject_name)

            dst_trial_dir = dst_subject_dir / original_trial_dir.name
            dst_processed = dst_trial_dir / "ProcessedData"
            dst_processed.mkdir(parents=True, exist_ok=True)

            # Symlink MoCap unchanged (it is the GT source for the transformer loader).
            src_mocap = original_trial_dir / "MoCap"
            if src_mocap.exists():
                dst_mocap_link = dst_trial_dir / "MoCap"
                if not dst_mocap_link.exists() and not dst_mocap_link.is_symlink():
                    try:
                        os.symlink(src_mocap, dst_mocap_link)
                    except OSError:
                        shutil.copytree(src_mocap, dst_mocap_link, symlinks=True)

            try:
                refined_pos = np.load(q_prime_path).astype(np.float32)
                provenance = _build_refined_processed_data(
                    refined_pos_16=refined_pos,
                    original_trial_dir=original_trial_dir,
                    xml_path=xml_path,
                    output_dir=dst_processed,
                    physics_cfg=physics_cfg,
                    libs=libs,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                failures.append(f"{trial_info['trial_name']}: physics build failed: {exc}\n{tb}")
                _ts_print(f"  [physics] {trial_info['trial_name']} FAILED: {exc}")
                continue

            try:
                length = int(np.load(dst_processed / "pos_inputs.npy", mmap_mode="r").shape[0])
            except Exception:
                length = int(provenance.get("T", 0))

            new_info = {
                "subject": subject_name,
                "trial_name": f"{subject_name}/{original_trial_dir.name}",
                "training_data_path": str(dst_processed),
                "length": length,
                "original_trial_dir": str(original_trial_dir),
                "original_subject_dir": str(original_subject_dir),
            }
            out_buckets[bucket_name].append(new_info)
            successes += 1
            if processed % 5 == 0 or processed == total_trials:
                _ts_print(
                    f"  [physics] progress {processed}/{total_trials} "
                    f"(ok={successes}, failed={len(failures)})"
                )

    if failures:
        _write_json(fold_dir / "transformer_cache_failures.json", {"failures": failures})
        _ts_print(f"  [physics] wrote failure report with {len(failures)} item(s)")
    _ts_print(
        "  [physics] stage complete: "
        f"ok={successes}, failed={len(failures)}, "
        f"train={len(out_buckets['train'])}, val={len(out_buckets['val'])}, held_out={len(out_buckets['held_out'])}"
    )

    fold_payload = {
        "held_out_subject": fold["held_out_subject"],
        "inner_val_subject": fold["inner_val_subject"],
        "train_subjects": list(fold["train_subjects"]),
        "train_trials": out_buckets["train"],
        "inner_val_trials": out_buckets["val"],
        "held_out_trials": out_buckets["held_out"],
    }
    _write_json(fold_dir / "transformer_dataset_split.json", fold_payload)
    return out_buckets


# ---------------------------------------------------------------------------
# Stage-3 transformer LOSO runner — delegates to loso_from_checkpoint._run_fold.
# ---------------------------------------------------------------------------
TRANSFORMER_LOSO_OVERRIDE_KEYS: Tuple[str, ...] = (
    "epochs",
    "learning_rate",
    "weight_decay",
    "torque_weight",
    "grf_weight",
    "cop_weight",
    "UseGRFNormCOP",
    "includeJacobianInput",
)


def _resolve_transformer_loso_effective_config(
    base_config: Mapping[str, Any],
    transformer_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    extra_keys = sorted(set(transformer_cfg.keys()) - set(TRANSFORMER_LOSO_OVERRIDE_KEYS))
    if extra_keys:
        raise ValueError(
            "COMBINED_LOSO_CONFIG['transformer_loso'] only supports these override keys: "
            f"{list(TRANSFORMER_LOSO_OVERRIDE_KEYS)}. Remove extra key(s): {extra_keys}"
        )

    resolved = dict(base_config)
    for key in TRANSFORMER_LOSO_OVERRIDE_KEYS:
        if key in transformer_cfg and transformer_cfg[key] is not None:
            if key == "UseGRFNormCOP":
                resolved_key = "use_grf_norm_cop"
            elif key == "includeJacobianInput":
                resolved_key = "include_jacobian_input"
            else:
                resolved_key = key
            resolved[resolved_key] = transformer_cfg[key]
    return resolved


def _run_transformer_fold(
    fold: Mapping[str, Any],
    out_buckets: Mapping[str, List[Dict[str, Any]]],
    fold_dir: Path,
    transformer_checkpoint_path: Path,
    transformer_cfg: Mapping[str, Any],
    seed: int,
    libs: Mapping[str, Any],
) -> Dict[str, Any]:
    loso_module = libs["loso_from_checkpoint"]

    # _load_checkpoint_bundle returns (checkpoint_dict, normalized_hyperparameters).
    checkpoint, base_config = loso_module._load_checkpoint_bundle(transformer_checkpoint_path)
    effective_config = _resolve_transformer_loso_effective_config(
        base_config,
        transformer_cfg,
    )

    fold_for_runner = {
        "held_out_subject": fold["held_out_subject"],
        "inner_val_subject": fold["inner_val_subject"],
        "train_subjects": list(fold["train_subjects"]),
        "train_trials": list(out_buckets["train"]),
        "inner_val_trials": list(out_buckets["val"]) or list(out_buckets["train"]),
        "held_out_trials": list(out_buckets["held_out"]),
    }
    if not fold_for_runner["train_trials"]:
        return {"status": "skipped", "reason": "no train trials in transformer cache"}
    if not fold_for_runner["held_out_trials"]:
        return {"status": "skipped", "reason": "no held-out trials in transformer cache"}
    if not out_buckets["val"]:
        _ts_print(
            f"[stage3 fold {fold['held_out_subject']}] no separate validation subject; "
            "using training trials for internal validation/selection metrics."
        )

    adapter_hidden_dim = int(effective_config.get("adapter_hidden_dim", 0))
    adapter_dropout_rate = float(effective_config.get("adapter_dropout_rate", 0.0))

    fold_runner_dir = fold_dir / "stage3_transformer_loso"
    fold_runner_dir.mkdir(parents=True, exist_ok=True)
    _write_json(fold_runner_dir / "effective_transformer_loso_hyperparameters.json", effective_config)
    _ts_print(
        f"[stage3 fold {fold['held_out_subject']}] start: "
        f"epochs={int(effective_config['epochs'])}, lr={float(effective_config['learning_rate'])}, "
        f"batch_size={int(effective_config['batch_size'])}, "
        f"weight_decay={float(effective_config['weight_decay'])}, "
        f"torque_weight={float(effective_config['torque_weight'])}, "
        f"grf_weight={float(effective_config['grf_weight'])}, "
        f"cop_weight={float(effective_config['cop_weight'])}, "
        f"UseGRFNormCOP={bool(effective_config.get('use_grf_norm_cop', False))}"
    )
    _ts_print(
        f"[stage3 fold {fold['held_out_subject']}] dataset sizes: "
        f"train={len(fold_for_runner['train_trials'])}, val={len(fold_for_runner['inner_val_trials'])}, "
        f"held_out={len(fold_for_runner['held_out_trials'])}"
    )
    stage3_start = time.time()

    metrics = loso_module._run_fold(
        fold_for_runner,
        fold_dir=fold_runner_dir,
        checkpoint=checkpoint,
        config=effective_config,
        epochs=int(effective_config["epochs"]),
        learning_rate=float(effective_config["learning_rate"]),
        batch_size=int(effective_config["batch_size"]),
        weight_decay=float(effective_config["weight_decay"]),
        adapter_hidden_dim=int(adapter_hidden_dim),
        adapter_dropout_rate=float(adapter_dropout_rate),
        seed=int(seed),
    )
    elapsed_s = time.time() - stage3_start
    if isinstance(metrics, dict):
        key_metrics = []
        for key in ("best_val_total_loss", "test_total_loss", "test_torque_loss", "test_torque_mae"):
            if key in metrics:
                key_metrics.append(f"{key}={metrics[key]}")
        if key_metrics:
            _ts_print(
                f"[stage3 fold {fold['held_out_subject']}] complete in {elapsed_s:.1f}s | "
                + ", ".join(key_metrics)
            )
        else:
            _ts_print(f"[stage3 fold {fold['held_out_subject']}] complete in {elapsed_s:.1f}s")
    else:
        _ts_print(f"[stage3 fold {fold['held_out_subject']}] complete in {elapsed_s:.1f}s")
    return {"status": "ok", "metrics": metrics, "fold_dir": str(fold_runner_dir)}


def _write_combined_stage3_infer_summary(
    output_dir: Path,
    stage3_metrics_by_fold: Sequence[Mapping[str, Any]],
    libs: Mapping[str, Any],
) -> None:
    loso_module = libs["loso_from_checkpoint"]
    infer_module = libs["infer_module"] if "infer_module" in libs else None
    if infer_module is None:
        infer_module = importlib.import_module("infer")

    infer_trial_metric_rows: List[Dict[str, Any]] = []
    infer_trial_mae_reports: Dict[str, Dict[str, float]] = {}
    source_mae_reports: Dict[str, Dict[str, Dict[str, float]]] = {
        "LOSO Fine-Tuned": {},
        "Original OpenCap PredInput": {},
        "Original OpenCap OCInput": {},
        "Original Motion Capture": {},
    }
    comparison_rows: List[Dict[str, Any]] = []
    aggregated_stance_stats_groups: Dict[str, List[Mapping[str, Mapping[str, Any]]]] = {}

    source_key_map = {
        "mae_reports": "LOSO Fine-Tuned",
        "mae_reports_original_opencap_predinput": "Original OpenCap PredInput",
        "mae_reports_original_opencap_ocinput": "Original OpenCap OCInput",
        "mae_reports_original_motion_capture": "Original Motion Capture",
        # Backward-compatible keys from older fold outputs.
        "mae_reports_original_opencap": "Original OpenCap PredInput",
        "mae_reports_original_motioncapture": "Original Motion Capture",
    }

    for metrics_payload in stage3_metrics_by_fold:
        infer_summary = metrics_payload.get("held_out_infer_style", {})
        if not isinstance(infer_summary, Mapping):
            continue
        for metric in infer_summary.get("trial_metrics", []):
            if isinstance(metric, Mapping):
                infer_trial_metric_rows.append(dict(metric))
        for summary_key, source_label in source_key_map.items():
            reports = infer_summary.get(summary_key, {})
            if not isinstance(reports, Mapping):
                continue
            for trial_name, mae_report in reports.items():
                if not isinstance(mae_report, Mapping):
                    continue
                normalized = {
                    str(key): float(value)
                    for key, value in dict(mae_report).items()
                    if value is not None
                }
                source_mae_reports.setdefault(source_label, {})[str(trial_name)] = normalized
                if source_label == "LOSO Fine-Tuned":
                    infer_trial_mae_reports[str(trial_name)] = normalized
        for comparison_row in infer_summary.get("comparison_rows", []):
            if isinstance(comparison_row, Mapping):
                comparison_rows.append(dict(comparison_row))
        for source_label, source_stats in infer_summary.get("aggregated_stance_statistics_by_source", {}).items():
            if isinstance(source_stats, Mapping) and source_stats:
                aggregated_stance_stats_groups.setdefault(str(source_label), []).append(source_stats)

    if not infer_trial_metric_rows:
        _ts_print("[combined stage3] no infer-style trial metrics found; combined infer summary skipped.")
        return

    infer_style_summary_dir = output_dir / "infer_style_eval"
    infer_style_summary = loso_module._write_infer_style_summary_artifacts(
        infer_style_summary_dir,
        infer_trial_metric_rows,
        infer_trial_mae_reports,
        {},
        mae_by_source=source_mae_reports,
    )
    loso_module._write_summary_csv(output_dir / "loso_infer_trial_metrics.csv", infer_trial_metric_rows)

    comparison_metric_means: Dict[str, Any] = {}
    comparison_metric_stds: Dict[str, Any] = {}
    if comparison_rows:
        comparison_metric_means, comparison_metric_stds = loso_module._aggregate_metric_dicts(comparison_rows)
        comparison_summary_payload = {
            "trial_count": len(comparison_rows),
            "metric_means": comparison_metric_means,
            "metric_stds": comparison_metric_stds,
            "per_trial": comparison_rows,
        }
        loso_module._save_json(infer_style_summary_dir / "model_comparison_summary.json", comparison_summary_payload)
        loso_module._save_json(output_dir / "loso_model_comparison_summary.json", comparison_summary_payload)

    aggregated_stance_statistics_by_source = {
        source_label: loso_module._merge_stance_summary_stats(source_stats_group)
        for source_label, source_stats_group in aggregated_stance_stats_groups.items()
        if source_stats_group
    }
    if aggregated_stance_statistics_by_source:
        infer_style_summary["aggregated_stance_statistics_by_source"] = aggregated_stance_statistics_by_source
        infer_style_summary["aggregated_stance_statistics"] = aggregated_stance_statistics_by_source.get(
            "LOSO Fine-Tuned",
            {},
        )
        loso_module._save_json(infer_style_summary_dir / "infer_style_summary.json", infer_style_summary)
        loso_module._save_json(
            infer_style_summary_dir / "aggregated_stance_statistics_by_source.json",
            aggregated_stance_statistics_by_source,
        )
        loso_module._save_json(
            output_dir / "aggregated_stance_statistics_by_source.json",
            aggregated_stance_statistics_by_source,
        )
        loso_module._save_json(
            output_dir / "aggregated_stance_statistics.json",
            aggregated_stance_statistics_by_source.get("LOSO Fine-Tuned", {}),
        )

    source_average_mae = loso_module._compute_source_average_mae_per_dof(source_mae_reports)
    source_average_joint_moment_mae = {
        source_label: loso_module._filter_joint_moment_mae_map(source_mae)
        for source_label, source_mae in source_average_mae.items()
        if source_mae
    }
    source_trial_details = {
        source_label: loso_module._build_trial_detail_payloads(source_reports)
        for source_label, source_reports in source_mae_reports.items()
        if source_reports
    }
    source_subject_avg = loso_module._compute_subject_average_torque_mae_by_source(source_trial_details)

    grf_by_source_trial: Dict[str, Dict[str, Dict[str, float]]] = {
        "fine_tuned_opencap_input": {},
        "original_opencap_predinput": {},
        "original_opencap_ocinput": {},
        "motioncapture_input": {},
    }
    comparison_source_specs = (
        ("loso_fine_tuned", "fine_tuned_opencap_input"),
        ("original_checkpoint_opencap_predinput", "original_opencap_predinput"),
        ("original_checkpoint_opencap_ocinput", "original_opencap_ocinput"),
        ("original_checkpoint_mocap", "motioncapture_input"),
    )
    for comparison_row in comparison_rows:
        trial_name = str(comparison_row.get("trial_name", "unknown_trial"))
        for comparison_key, output_key in comparison_source_specs:
            source_payload = comparison_row.get(comparison_key)
            if not isinstance(source_payload, Mapping):
                continue
            grf_summary = loso_module._extract_bilateral_grf_mae_percent_bw(source_payload.get("metrics"))
            if grf_summary:
                grf_by_source_trial[output_key][trial_name] = grf_summary

    compatible_overall_mae_report = {
        "average_mae_per_dof": source_average_mae.get("LOSO Fine-Tuned", {}),
        "average_joint_moment_mae_per_dof": source_average_joint_moment_mae.get("LOSO Fine-Tuned", {}),
        "average_mae_per_dof_opencap_input": source_average_mae.get("Original OpenCap PredInput", {}),
        "average_joint_moment_mae_per_dof_opencap_input": source_average_joint_moment_mae.get("Original OpenCap PredInput", {}),
        "average_mae_per_dof_original_opencap_predinput": source_average_mae.get("Original OpenCap PredInput", {}),
        "average_joint_moment_mae_per_dof_original_opencap_predinput": source_average_joint_moment_mae.get("Original OpenCap PredInput", {}),
        "average_mae_per_dof_original_opencap_ocinput": source_average_mae.get("Original OpenCap OCInput", {}),
        "average_joint_moment_mae_per_dof_original_opencap_ocinput": source_average_joint_moment_mae.get("Original OpenCap OCInput", {}),
        "average_mae_per_dof_fine_tuned_opencap_input": source_average_mae.get("LOSO Fine-Tuned", {}),
        "average_joint_moment_mae_per_dof_fine_tuned_opencap_input": source_average_joint_moment_mae.get("LOSO Fine-Tuned", {}),
        "average_mae_per_dof_motioncapture_input": source_average_mae.get("Original Motion Capture", {}),
        "average_joint_moment_mae_per_dof_motioncapture_input": source_average_joint_moment_mae.get("Original Motion Capture", {}),
        "average_mae_per_dof_by_source": source_average_mae,
        "average_joint_moment_mae_per_dof_by_source": source_average_joint_moment_mae,
        "trial_details": source_trial_details.get("LOSO Fine-Tuned", {}),
        "trial_details_opencap_input": source_trial_details.get("Original OpenCap PredInput", {}),
        "trial_details_original_opencap_predinput": source_trial_details.get("Original OpenCap PredInput", {}),
        "trial_details_original_opencap_ocinput": source_trial_details.get("Original OpenCap OCInput", {}),
        "trial_details_fine_tuned_opencap_input": source_trial_details.get("LOSO Fine-Tuned", {}),
        "trial_details_motioncapture_input": source_trial_details.get("Original Motion Capture", {}),
        "subject_average_torque_mae_bwh_percent": source_subject_avg.get("LOSO Fine-Tuned", {}),
        "subject_average_torque_mae_bwh_percent_opencap_input": source_subject_avg.get("Original OpenCap PredInput", {}),
        "subject_average_torque_mae_bwh_percent_original_opencap_predinput": source_subject_avg.get("Original OpenCap PredInput", {}),
        "subject_average_torque_mae_bwh_percent_original_opencap_ocinput": source_subject_avg.get("Original OpenCap OCInput", {}),
        "subject_average_torque_mae_bwh_percent_fine_tuned_opencap_input": source_subject_avg.get("LOSO Fine-Tuned", {}),
        "subject_average_torque_mae_bwh_percent_motioncapture_input": source_subject_avg.get("Original Motion Capture", {}),
        "subject_average_torque_mae_bwh_percent_by_source": source_subject_avg,
        "grf_mae_percent_bw_bilateral_stance_by_source": {
            source_label: loso_module._average_metric_dicts(trial_metrics_by_source)
            for source_label, trial_metrics_by_source in grf_by_source_trial.items()
            if trial_metrics_by_source
        },
        "trial_grf_mae_percent_bw_bilateral_stance_by_source": grf_by_source_trial,
        "torque_metric_scope": "left_stance_selected_dofs",
        "torque_metric_side": "left",
        "torque_metric_phase": "stance",
        "torque_metric_dof_names": list(infer_module.SELECTED_LEFT_STANCE_DOF_NAMES)
        + [infer_module.LEFT_STANCE_KAM_DOF_NAME],
        "source": "Loso_Combined",
        "source_summary_json": str(infer_style_summary_dir / "infer_style_summary.json"),
        "source_trial_metrics_csv": str(output_dir / "loso_infer_trial_metrics.csv"),
        "comparison_metric_means": comparison_metric_means,
        "comparison_metric_stds": comparison_metric_stds,
    }
    if aggregated_stance_statistics_by_source:
        compatible_overall_mae_report["aggregated_stance_statistics_by_source"] = aggregated_stance_statistics_by_source
        compatible_overall_mae_report["aggregated_stance_statistics"] = aggregated_stance_statistics_by_source.get(
            "LOSO Fine-Tuned",
            {},
        )
    loso_module._save_json(output_dir / "overall_mae_report.json", compatible_overall_mae_report)

    summary_payload = {
        "source": "Loso_Combined",
        "completed_stage3_folds": len(stage3_metrics_by_fold),
        "infer_style_eval_output_dir": str(infer_style_summary_dir),
        "infer_style_trial_count": len(infer_trial_metric_rows),
        "infer_style_trial_metric_rows": infer_trial_metric_rows,
        "infer_style_metric_means": infer_style_summary.get("metric_means", {}),
        "infer_style_metric_stds": infer_style_summary.get("metric_stds", {}),
        "infer_style_average_mae_per_dof": infer_style_summary.get("average_mae_per_dof", {}),
        "infer_style_subject_average_torque_mae_bwh_percent": infer_style_summary.get(
            "subject_average_torque_mae_bwh_percent"
        ),
        "infer_style_subject_average_torque_mae_bwh_percent_by_source": infer_style_summary.get(
            "subject_average_torque_mae_bwh_percent_by_source", {}
        ),
        "infer_style_model_comparison_rows": comparison_rows,
        "infer_style_model_comparison_metric_means": comparison_metric_means,
        "infer_style_model_comparison_metric_stds": comparison_metric_stds,
        "overall_mae_report": str(output_dir / "overall_mae_report.json"),
        "source_summary_json": str(infer_style_summary_dir / "infer_style_summary.json"),
        "source_trial_metrics_csv": str(output_dir / "loso_infer_trial_metrics.csv"),
    }
    loso_module._save_json(output_dir / "loso_summary.json", summary_payload)

    _ts_print(f"[combined stage3] wrote infer-style summary to {infer_style_summary_dir / 'infer_style_summary.json'}")
    _ts_print(f"[combined stage3] wrote trial metrics CSV to {output_dir / 'loso_infer_trial_metrics.csv'}")
    _ts_print(f"[combined stage3] wrote compareMAE summary to {output_dir / 'overall_mae_report.json'}")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def _resolve_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(COMBINED_LOSO_CONFIG))  # deep copy via JSON round-trip
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.refine_q_base_checkpoint:
        cfg["refine_q_base_checkpoint"] = args.refine_q_base_checkpoint
    if args.transformer_checkpoint:
        cfg["transformer_checkpoint"] = args.transformer_checkpoint
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.only_subjects:
        cfg["only_subjects"] = list(args.only_subjects)
    if args.stage1_only is not None:
        cfg["stage1_only"] = bool(args.stage1_only)
    if args.stage3_UsePrecomputed:
        cfg["stage3_UsePrecomputed"] = True
        # stage3_UsePrecomputed is mutually exclusive with stage1_only; flip the
        # default off so the user doesn't have to also pass --no-stage1_only.
        if args.stage1_only is None:
            cfg["stage1_only"] = False
    if args.stage2_computeJotQfrcRot_accuracy:
        cfg["stage2_computeJotQfrcRot_accuracy"] = True
    for arg_name, cfg_name in (
        ("refine_q_qfrc_inverse_loss_weight", "qfrc_inverse_loss_weight"),
        ("refine_q_jacobian_loss_weight", "jacobian_loss_weight"),
        ("refine_q_rotation_loss_weight", "rotation_loss_weight"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            cfg["refine_q"][cfg_name] = float(value)
    if getattr(args, "use_train_dataset_normalizers", False):
        cfg["refine_q"]["use_train_dataset_normalizers"] = True
    if getattr(args, "trusted_normalizer_data_dir", None):
        cfg["refine_q"]["trusted_normalizer_data_dir"] = str(args.trusted_normalizer_data_dir)
    if getattr(args, "trusted_normalizer_num_windows", None) is not None:
        cfg["refine_q"]["trusted_normalizer_num_windows"] = int(args.trusted_normalizer_num_windows)
    if getattr(args, "trusted_normalizer_sample_seed", None) is not None:
        cfg["refine_q"]["trusted_normalizer_sample_seed"] = int(args.trusted_normalizer_sample_seed)
    if getattr(args, "one_batch", False):
        cfg["refine_q"]["one_batch"] = True
    if getattr(args, "UseGRFNormCOP", None) is not None:
        cfg["transformer_loso"]["UseGRFNormCOP"] = bool(args.UseGRFNormCOP)
    if getattr(args, "includeJacobianInput", None) is not None:
        cfg["transformer_loso"]["includeJacobianInput"] = bool(args.includeJacobianInput)
    return cfg


def _filter_folds_by_subject(
    folds: List[Dict[str, Any]],
    only_subjects: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    if not only_subjects:
        return folds
    keep = set(only_subjects)
    return [fold for fold in folds if fold["held_out_subject"] in keep]


def _validate_refine_q_checkpoint_bundle(refine_q_base_checkpoint: Optional[str]) -> None:
    if not refine_q_base_checkpoint:
        return
    ckpt_path = Path(str(refine_q_base_checkpoint))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"refine_q_base_checkpoint not found: {ckpt_path}")
    hyperparams_path = ckpt_path.parent / "hyperparameters.json"
    if not hyperparams_path.exists():
        raise FileNotFoundError(
            f"Required refine-q hyperparameters file not found next to checkpoint: {hyperparams_path}"
        )
    try:
        with open(hyperparams_path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse refine-q hyperparameters file: {hyperparams_path}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined LOSO over OpenCapSubjects (refine-q + transformer).")
    parser.add_argument("--data_dir", type=str, default=None, help="OpenCapSubjects root (overrides config).")
    parser.add_argument("--output_dir", type=str, default=None, help="Top-level output directory.")
    parser.add_argument("--refine_q_base_checkpoint", type=str, default=None,
                        help="Optional pre-trained QRefineTransformer .pkl for fine-tuning.")
    parser.add_argument("--transformer_checkpoint", type=str, default=None,
                        help="Pre-trained transformer .pkl checkpoint (required for stage 3).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--stage1_only",
        dest="stage1_only",
        action="store_true",
        default=None,
        help="Run only Stage 1 refine-q LOSO; skip Stage 2 physics and Stage 3 transformer.",
    )
    parser.add_argument(
        "--no-stage1_only",
        dest="stage1_only",
        action="store_false",
        help="Override config default and disable stage1_only (needed when running Stage 3).",
    )
    parser.add_argument(
        "--stage3_UsePrecomputed",
        action="store_true",
        help="Skip Stage 1/2 and run Stage 3 using original ProcessedData precomputed physics files.",
    )
    parser.add_argument(
        "--stage2_computeJotQfrcRot_accuracy",
        action="store_true",
        help=(
            "After Stage 1 and Stage 2, compare regenerated Jacobian, qfrc_inverse, "
            "and COP rotation against MoCap GT versus original ProcessedData."
        ),
    )
    parser.add_argument(
        "--refine_q_qfrc_inverse_loss_weight",
        type=float,
        default=None,
        help="Stage-1 refine-q differentiable-MJX qfrc_inverse loss weight.",
    )
    parser.add_argument(
        "--refine_q_jacobian_loss_weight",
        type=float,
        default=None,
        help="Stage-1 refine-q differentiable-MJX Jacobian loss weight.",
    )
    parser.add_argument(
        "--refine_q_rotation_loss_weight",
        type=float,
        default=None,
        help="Stage-1 refine-q differentiable-MJX COP rotation geodesic loss weight.",
    )
    parser.add_argument(
        "--use_train_dataset_normalizers",
        action="store_true",
        help=(
            "On each Stage-1 fold, fit pos/vel/acc stats from random windows under refine_q.trusted_normalizer_data_dir "
            "and write equiv_kinematic_normalizers.json. Stds replace per-batch OpenCap stds for kinematic-equiv "
            "metrics when differentiable physics losses are active."
        ),
    )
    parser.add_argument(
        "--trusted_normalizer_data_dir",
        type=str,
        default=None,
        help="Root with <Subject>/<Trial>/ProcessedData (refine_q layout); default from COMBINED_LOSO_CONFIG.",
    )
    parser.add_argument(
        "--trusted_normalizer_num_windows",
        type=int,
        default=None,
        help="Number of random windows to pool for normalizer stats (default from config, typically 1000).",
    )
    parser.add_argument(
        "--trusted_normalizer_sample_seed",
        type=int,
        default=None,
        help="RNG seed for window sampling; default = global --seed plus a fold-dependent tag.",
    )
    parser.add_argument(
        "--one_batch",
        action="store_true",
        help=(
            "Stage-1 refine-q: stack all sliding windows into one batch per epoch (non-physics), or "
            "one batch per MuJoCo XML with all its windows (physics). Matches refine_q.one_batch in config."
        ),
    )
    parser.add_argument(
        "--UseGRFNormCOP",
        nargs="?",
        const=True,
        default=None,
        type=lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        help="Stage 3 transformer: use COP_CalcFrame_GroundAligned_GRFNorm.npy as the COP target.",
    )
    parser.add_argument(
        "--includeJacobianInput",
        nargs="?",
        const=True,
        default=None,
        type=lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y", "on"},
        help="Stage 3 transformer: include flattened preprocessed Jacobian [jacp,jacr] as temporal model inputs.",
    )
    parser.add_argument("--only_subjects", type=str, nargs="*", default=None,
                        help="Optional list of held-out subjects to actually run (e.g. subject2 subject3).")
    args = parser.parse_args()

    cfg = _resolve_runtime_config(args)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "combined_loso_config.json", cfg)

    libs = _lazy_imports()
    train_module = libs["train_module"]
    loso_module = libs["loso_from_checkpoint"]

    data_dir = Path(cfg["data_dir"])
    _ts_print(f"Discovering trials in {data_dir} ...")
    trials = train_module.discover_all_trials(str(data_dir), refresh_cache=True, scan_workers=4)
    if not trials:
        raise RuntimeError(f"no trials discovered under {data_dir}")
    subject_to_trials: Dict[str, List[Mapping[str, Any]]] = {}
    for trial in trials:
        subject_to_trials.setdefault(str(trial["subject"]), []).append(trial)
    valid_subjects = sorted(subject_to_trials.keys(), key=_subject_sort_key)
    if len(valid_subjects) < 2:
        raise RuntimeError(
            f"Combined LOSO requires at least 2 valid subjects, found {valid_subjects}"
        )
    _ts_print(f"Valid LOSO subjects: {valid_subjects}")

    folds = _build_loso_folds_from_trials(trials, valid_subjects)
    folds = _filter_folds_by_subject(folds, cfg.get("only_subjects"))
    _ts_print(f"Running {len(folds)} fold(s).")

    summary_path = output_dir / "loso_summary.jsonl"
    summary_path.unlink(missing_ok=True)

    transformer_ckpt_str = cfg.get("transformer_checkpoint")
    transformer_ckpt = Path(str(transformer_ckpt_str)) if transformer_ckpt_str else None
    stage1_only = bool(cfg.get("stage1_only", False))
    stage3_use_precomputed = bool(cfg.get("stage3_UsePrecomputed", False))
    stage2_compute_accuracy = bool(cfg.get("stage2_computeJotQfrcRot_accuracy", False))
    if stage1_only and stage3_use_precomputed:
        raise ValueError("stage1_only and stage3_UsePrecomputed cannot both be true.")
    if stage1_only and stage2_compute_accuracy:
        raise ValueError("stage2_computeJotQfrcRot_accuracy requires Stage 2 and cannot be used with stage1_only.")
    if stage2_compute_accuracy and stage3_use_precomputed:
        raise ValueError(
            "stage2_computeJotQfrcRot_accuracy requires Stage 1/2 and cannot be used with stage3_UsePrecomputed."
        )
    if not stage3_use_precomputed:
        _validate_refine_q_checkpoint_bundle(cfg.get("refine_q_base_checkpoint"))
    can_run_stage3 = (
        (not stage1_only)
        and (not stage2_compute_accuracy)
        and bool(transformer_ckpt and transformer_ckpt.exists())
    )
    if stage1_only:
        _ts_print("Stage1Only=True — skipping Stage 2 physics regeneration and Stage 3 transformer LOSO.")
    elif stage3_use_precomputed:
        _ts_print(
            "stage3_UsePrecomputed=True — skipping Stage 1/2 and using original ProcessedData physics files."
        )
        if not can_run_stage3:
            _ts_print(
                "WARN: transformer_checkpoint is missing or unset — precomputed Stage 3 will be skipped."
            )
    elif not can_run_stage3:
        _ts_print(
            "WARN: transformer_checkpoint is missing or unset — stage 3 (transformer LOSO) will be skipped."
        )
    if stage2_compute_accuracy:
        _ts_print(
            "stage2_computeJotQfrcRot_accuracy=True — Stage 2 refined physics will be compared against MoCap GT, then Stage 3 will be skipped."
        )
    refine_q_physics_losses_active = _refine_q_physics_losses_active(
        cfg.get("refine_q", {}),
        stage1_only=stage1_only,
    )
    if stage1_only and _refine_q_physics_losses_active(cfg.get("refine_q", {}), stage1_only=False):
        _ts_print(
            "Stage1Only=True — refine-q qfrc_inverse/Jacobian/rotation physics loss weights are configured but will be ignored."
        )
    elif refine_q_physics_losses_active:
        _ts_print(
            "refine-q differentiable physics losses are active; Stage 2 will skip low-pass filtering before differentiation."
        )
    if bool(cfg.get("refine_q", {}).get("use_train_dataset_normalizers")):
        _ts_print(
            "refine_q.use_train_dataset_normalizers=True — each Stage-1 fold will sample the trusted dataset "
            "(refine_q.trusted_normalizer_data_dir), fit pos/vel/acc mean+std, and write "
            "stage1_refine_q/equiv_kinematic_normalizers.json. This runs regardless of physics/reg loss weights; "
            "pos_std scales recon when physics is off, vel/acc stds feed equiv metrics when physics is on."
        )
    if bool(cfg.get("refine_q", {}).get("one_batch")):
        _ts_print(
            "refine_q.one_batch=True — Stage-1 uses one stacked batch per epoch (non-physics) or one batch per "
            "MuJoCo XML (physics); see per-fold logs for window counts and GPU memory caveats."
        )

    # Rolling Stage-1-only evaluation accumulators (across completed folds).
    rolling_joint_sum_noised: Dict[str, float] = {}
    rolling_joint_sum_refined: Dict[str, float] = {}
    rolling_joint_sum_improvement: Dict[str, float] = {}
    rolling_joint_sum_improvement_pct: Dict[str, float] = {}
    rolling_joint_count: Dict[str, int] = {}
    rolling_stage1_fold_count = 0
    rolling_stage2_accuracy: Dict[str, Dict[str, Dict[str, float]]] = {}
    stage3_metrics_by_fold: List[Mapping[str, Any]] = []

    for fold in folds:
        held = fold["held_out_subject"]
        fold_dir = output_dir / f"fold_{held}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _ts_print(f"════════ Fold: {held} (train_subjects={len(fold['train_subjects'])}, no val subject) ════════")
        fold_start = time.time()

        if stage3_use_precomputed:
            _ts_print(f"[fold {held}] stage1/stage2 skipped (stage3_UsePrecomputed=True)")
            out_buckets = _build_precomputed_transformer_buckets(fold)
            _ts_print(
                f"[fold {held}] precomputed Stage 3 dataset: "
                f"train={len(out_buckets['train'])} val={len(out_buckets['val'])} "
                f"held_out={len(out_buckets['held_out'])}"
            )
            stage3_result: Dict[str, Any] = {"status": "skipped"}
            if can_run_stage3 and transformer_ckpt is not None:
                _ts_print(f"[fold {held}] stage3 start: transformer LOSO fine-tuning/eval")
                try:
                    stage3_result = _run_transformer_fold(
                        fold,
                        out_buckets,
                        fold_dir,
                        transformer_ckpt,
                        cfg["transformer_loso"],
                        seed=int(cfg["seed"]) + _subject_sort_key(held)[0],
                        libs=libs,
                    )
                except Exception as exc:
                    tb = traceback.format_exc()
                    _ts_print(f"[fold {held}] transformer LOSO FAILED: {exc}\n{tb}")
                    stage3_result = {"status": "failed", "error": str(exc), "traceback": tb}

            fold_summary = {
                "fold": held,
                "inner_val_subject": fold["inner_val_subject"],
                "train_subjects": fold["train_subjects"],
                "stage1_only": stage1_only,
                "stage3_UsePrecomputed": stage3_use_precomputed,
                "stage2_computeJotQfrcRot_accuracy": stage2_compute_accuracy,
                "transformer_cache_counts": {
                    "train": len(out_buckets["train"]),
                    "val": len(out_buckets["val"]),
                    "held_out": len(out_buckets["held_out"]),
                },
                "stage3_status": stage3_result.get("status"),
                "stage3_error": stage3_result.get("error"),
                "fold_dir": str(fold_dir),
            }
            if "metrics" in stage3_result and isinstance(stage3_result["metrics"], dict):
                for key in (
                    "test_total_loss",
                    "test_torque_loss",
                    "test_torque_mae",
                    "best_val_total_loss",
                ):
                    if key in stage3_result["metrics"]:
                        fold_summary[key] = stage3_result["metrics"][key]
                stage3_metrics_by_fold.append(stage3_result["metrics"])
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fold_summary, default=str) + "\n")
            _write_json(fold_dir / "fold_summary.json", fold_summary)
            _ts_print(
                f"[fold {held}] done in {time.time()-fold_start:.1f}s. "
                f"stage3={stage3_result.get('status')}"
            )
            continue

        # Stage 1a — build refine-q cache
        stage1a_start = time.time()
        _ts_print(f"[fold {held}] stage1a start: building refine-q cache")
        bucket_trials = _build_refine_q_cache(fold, fold_dir)
        _ts_print(
            f"[fold {held}] stage1a complete in {time.time()-stage1a_start:.1f}s: "
            f"train={len(bucket_trials['train'])}, val={len(bucket_trials['val'])}, held_out={len(bucket_trials['held_out'])}"
        )

        # Stage 1b — fine-tune refine-q
        stage1b_start = time.time()
        _ts_print(f"[fold {held}] stage1b start: refine-q fine-tuning")
        try:
            refine_result = _train_refine_q_for_fold(
                fold,
                bucket_trials,
                cfg["refine_q"],
                fold_dir=fold_dir,
                base_checkpoint=cfg.get("refine_q_base_checkpoint"),
                seed=int(cfg["seed"]),
                libs=libs,
                enable_physics_losses=not stage1_only,
                physics_cfg=cfg["physics"],
            )
        except Exception as exc:
            tb = traceback.format_exc()
            _ts_print(f"[fold {held}] refine-q training FAILED: {exc}\n{tb}")
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"fold": held, "status": "refine_q_failed", "error": str(exc)}) + "\n")
            continue
        _ts_print(
            f"[fold {held}] stage1b complete in {time.time()-stage1b_start:.1f}s: "
            f"best_val_loss={refine_result.get('best_val_loss')}"
        )
        refine_cfg_for_inference = refine_result.get("effective_refine_cfg", cfg["refine_q"])

        # Stage 1c — predict refined kinematics for every trial
        stage1c_start = time.time()
        _ts_print(f"[fold {held}] stage1c start: generating refined predictions")
        refined_pred_dir = fold_dir / "refined_predictions"
        refined_pred_paths = _predict_refined_for_all_trials(
            bucket_trials,
            refine_result["model"],
            refine_result["params"],
            refine_cfg_for_inference,
            refined_pred_dir,
            libs,
        )
        _ts_print(
            f"[fold {held}] stage1c complete in {time.time()-stage1c_start:.1f}s: "
            f"refined predictions for {len(refined_pred_paths)} trials"
        )

        held_out_refine_eval = _compute_refine_q_held_out_joint_error_summary(
            bucket_trials,
            refined_pred_paths,
            libs,
        )
        _ts_print(
            f"[fold {held}] post-stage2 held-out refine-q MAE summary: "
            f"trials_used={held_out_refine_eval.get('used_trial_count', 0)}/"
            f"{held_out_refine_eval.get('held_out_trial_count', 0)} "
            f"frames={held_out_refine_eval.get('frame_count', 0)}"
        )
        overall_stage1_eval = held_out_refine_eval.get("overall", {})
        if isinstance(overall_stage1_eval, dict) and overall_stage1_eval:
            _ts_print(
                "  overall mean MAE "
                f"OpenCapInput->MoCapGT={overall_stage1_eval.get('mae_noised_vs_gt_mean', float('nan')):.6f} | "
                f"RefinedQPrime->MoCapGT={overall_stage1_eval.get('mae_refined_vs_gt_mean', float('nan')):.6f} | "
                f"improvement={overall_stage1_eval.get('mae_improvement_mean', float('nan')):.6f} "
                f"({overall_stage1_eval.get('mae_improvement_percent_mean', float('nan')):.2f}%)"
            )
        per_joint_stage1_eval = held_out_refine_eval.get("per_joint", {})
        if isinstance(per_joint_stage1_eval, dict) and per_joint_stage1_eval:
            _ts_print(
                f"[fold {held}] held-out per-joint MAE "
                "(OpenCapInput->MoCapGT vs RefinedQPrime->MoCapGT):"
            )
            for joint_name in POS_INPUT_DOF_NAMES:
                stats = per_joint_stage1_eval.get(str(joint_name))
                if not isinstance(stats, dict):
                    continue
                _ts_print(
                    f"  {joint_name}: "
                    f"{float(stats.get('mae_noised_vs_gt', float('nan'))):.6f} -> "
                    f"{float(stats.get('mae_refined_vs_gt', float('nan'))):.6f} | "
                    f"improvement={float(stats.get('mae_improvement', float('nan'))):.6f} "
                    f"({float(stats.get('mae_improvement_percent', float('nan'))):.2f}%)"
                )

        stage1_rolling_summary: Dict[str, Any] = {}
        if stage1_only and isinstance(per_joint_stage1_eval, dict) and per_joint_stage1_eval:
            rolling_stage1_fold_count += 1
            for joint_name in POS_INPUT_DOF_NAMES:
                stats = per_joint_stage1_eval.get(str(joint_name))
                if not isinstance(stats, dict):
                    continue
                noised_v = float(stats.get("mae_noised_vs_gt", float("nan")))
                refined_v = float(stats.get("mae_refined_vs_gt", float("nan")))
                improv_v = float(stats.get("mae_improvement", float("nan")))
                improv_pct_v = float(stats.get("mae_improvement_percent", float("nan")))
                if not (
                    np.isfinite(noised_v)
                    and np.isfinite(refined_v)
                    and np.isfinite(improv_v)
                    and np.isfinite(improv_pct_v)
                ):
                    continue
                key = str(joint_name)
                rolling_joint_sum_noised[key] = rolling_joint_sum_noised.get(key, 0.0) + noised_v
                rolling_joint_sum_refined[key] = rolling_joint_sum_refined.get(key, 0.0) + refined_v
                rolling_joint_sum_improvement[key] = rolling_joint_sum_improvement.get(key, 0.0) + improv_v
                rolling_joint_sum_improvement_pct[key] = (
                    rolling_joint_sum_improvement_pct.get(key, 0.0) + improv_pct_v
                )
                rolling_joint_count[key] = rolling_joint_count.get(key, 0) + 1

            rolling_per_joint: Dict[str, Dict[str, float]] = {}
            for joint_name in POS_INPUT_DOF_NAMES:
                key = str(joint_name)
                count = int(rolling_joint_count.get(key, 0))
                if count <= 0:
                    continue
                rolling_per_joint[key] = {
                    "rolling_avg_mae_noised_vs_gt": float(rolling_joint_sum_noised[key] / count),
                    "rolling_avg_mae_refined_vs_gt": float(rolling_joint_sum_refined[key] / count),
                    "rolling_avg_mae_improvement": float(rolling_joint_sum_improvement[key] / count),
                    "rolling_avg_mae_improvement_percent": float(
                        rolling_joint_sum_improvement_pct[key] / count
                    ),
                }

            if rolling_per_joint:
                _ts_print(
                    f"[fold {held}] Stage1Only rolling per-joint averages "
                    f"(after {rolling_stage1_fold_count} fold(s)):"
                )
                for joint_name in POS_INPUT_DOF_NAMES:
                    stats = rolling_per_joint.get(str(joint_name))
                    if not isinstance(stats, dict):
                        continue
                    _ts_print(
                        f"  {joint_name}: "
                        f"{stats['rolling_avg_mae_noised_vs_gt']:.6f} -> "
                        f"{stats['rolling_avg_mae_refined_vs_gt']:.6f} | "
                        f"rolling improvement={stats['rolling_avg_mae_improvement']:.6f} "
                        f"({stats['rolling_avg_mae_improvement_percent']:.2f}%)"
                    )

                stage1_rolling_summary = {
                    "folds_accumulated": int(rolling_stage1_fold_count),
                    "per_joint": rolling_per_joint,
                }

        # Stage 2 — physics regeneration cache
        out_buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "held_out": []}
        stage2_accuracy_summary: Dict[str, Any] = {}
        stage2_accuracy_rolling_summary: Dict[str, Any] = {}
        if stage1_only:
            _ts_print(f"[fold {held}] stage2 skipped (Stage1Only=True)")
        else:
            stage2_start = time.time()
            _ts_print(f"[fold {held}] stage2 start: physics regeneration")
            stage2_physics_cfg = dict(cfg["physics"])
            if refine_q_physics_losses_active:
                stage2_physics_cfg["filter_refined_kinematics"] = False
            try:
                out_buckets = _build_transformer_cache(
                    fold,
                    bucket_trials,
                    refined_pred_paths,
                    fold_dir,
                    stage2_physics_cfg,
                    libs,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                _ts_print(f"[fold {held}] physics cache FAILED: {exc}\n{tb}")
                with open(summary_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"fold": held, "status": "physics_failed", "error": str(exc)}) + "\n")
                continue
            _ts_print(
                f"[fold {held}] stage2 complete in {time.time()-stage2_start:.1f}s: "
                f"transformer cache train={len(out_buckets['train'])} "
                f"val={len(out_buckets['val'])} held_out={len(out_buckets['held_out'])}"
            )
            if stage2_compute_accuracy:
                _ts_print(f"[fold {held}] stage2 accuracy start: comparing refined/original physics to MoCap GT")
                try:
                    stage2_accuracy_summary = _compute_stage2_physics_accuracy_summary(
                        bucket_trials,
                        out_buckets,
                    )
                    stage2_accuracy_rolling_summary = _accumulate_stage2_rolling(
                        rolling_stage2_accuracy,
                        stage2_accuracy_summary,
                    )
                    _ts_print(
                        f"[fold {held}] stage2 accuracy complete: "
                        f"trials_used={stage2_accuracy_summary.get('used_trial_count', 0)}/"
                        f"{stage2_accuracy_summary.get('held_out_trial_count', 0)} "
                        f"frames={stage2_accuracy_summary.get('frame_count', 0)}"
                    )
                    qfrc_metrics = stage2_accuracy_summary.get("qfrc_inverse_per_joint_mae", {})
                    if isinstance(qfrc_metrics, Mapping) and qfrc_metrics:
                        _ts_print(f"[fold {held}] qfrc_inverse MAE vs MoCap GT per joint:")
                        for joint_name in sorted(qfrc_metrics):
                            stats = qfrc_metrics.get(joint_name, {})
                            if not isinstance(stats, Mapping):
                                continue
                            _ts_print(
                                f"  {joint_name}: "
                                f"{float(stats.get('original_mae', float('nan'))):.6f} -> "
                                f"{float(stats.get('refined_mae', float('nan'))):.6f} | "
                                f"improvement={float(stats.get('percent_improvement', float('nan'))):.2f}%"
                            )
                    jac_metrics = stage2_accuracy_summary.get("jacobian_per_body_mae", {})
                    if isinstance(jac_metrics, Mapping) and jac_metrics:
                        _ts_print(f"[fold {held}] Jacobian MAE vs MoCap GT per body id:")
                        for body_name in sorted(jac_metrics):
                            stats = jac_metrics.get(body_name, {})
                            if not isinstance(stats, Mapping):
                                continue
                            _ts_print(
                                f"  {body_name}: "
                                f"{float(stats.get('original_mae', float('nan'))):.6f} -> "
                                f"{float(stats.get('refined_mae', float('nan'))):.6f} | "
                                f"improvement={float(stats.get('percent_improvement', float('nan'))):.2f}%"
                            )
                    rot_metrics = stage2_accuracy_summary.get("rotation_per_body_geodesic_deg", {})
                    if isinstance(rot_metrics, Mapping) and rot_metrics:
                        _ts_print(f"[fold {held}] COP rotation geodesic error vs MoCap GT per foot/body:")
                        for body_name in sorted(rot_metrics):
                            stats = rot_metrics.get(body_name, {})
                            if not isinstance(stats, Mapping):
                                continue
                            _ts_print(
                                f"  {body_name}: "
                                f"{float(stats.get('original_mae', float('nan'))):.6f} deg -> "
                                f"{float(stats.get('refined_mae', float('nan'))):.6f} deg | "
                                f"improvement={float(stats.get('percent_improvement', float('nan'))):.2f}%"
                            )

                    if stage2_accuracy_rolling_summary:
                        _ts_print(f"[fold {held}] rolling Stage-2 physics accuracy averages:")
                        for section_key, section_values in stage2_accuracy_rolling_summary.items():
                            if not isinstance(section_values, Mapping) or not section_values:
                                continue
                            _ts_print(f"  {section_key}:")
                            for metric_name in sorted(section_values):
                                stats = section_values.get(metric_name, {})
                                if not isinstance(stats, Mapping):
                                    continue
                                _ts_print(
                                    f"    {metric_name}: "
                                    f"{float(stats.get('rolling_original_mae', float('nan'))):.6f} -> "
                                    f"{float(stats.get('rolling_refined_mae', float('nan'))):.6f} | "
                                    f"rolling improvement="
                                    f"{float(stats.get('rolling_percent_improvement', float('nan'))):.2f}%"
                                )
                except Exception as exc:
                    tb = traceback.format_exc()
                    _ts_print(f"[fold {held}] stage2 accuracy FAILED: {exc}\n{tb}")
                    stage2_accuracy_summary = {
                        "status": "failed",
                        "error": str(exc),
                        "traceback": tb,
                    }

        # Stage 3 — transformer LOSO fold (delegates to loso_from_checkpoint._run_fold)
        stage3_result: Dict[str, Any] = {"status": "skipped"}
        if can_run_stage3:
            _ts_print(f"[fold {held}] stage3 start: transformer LOSO fine-tuning/eval")
            try:
                stage3_result = _run_transformer_fold(
                    fold,
                    out_buckets,
                    fold_dir,
                    transformer_ckpt,
                    cfg["transformer_loso"],
                    seed=int(cfg["seed"]) + _subject_sort_key(held)[0],
                    libs=libs,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                _ts_print(f"[fold {held}] transformer LOSO FAILED: {exc}\n{tb}")
                stage3_result = {"status": "failed", "error": str(exc), "traceback": tb}

        fold_summary = {
            "fold": held,
            "inner_val_subject": fold["inner_val_subject"],
            "train_subjects": fold["train_subjects"],
            "stage1_only": stage1_only,
            "stage3_UsePrecomputed": stage3_use_precomputed,
            "stage2_computeJotQfrcRot_accuracy": stage2_compute_accuracy,
            "refine_q_best_val_loss": refine_result.get("best_val_loss"),
            "stage1_refine_q_held_out_joint_error_summary": held_out_refine_eval,
            "stage1_refine_q_rolling_joint_error_summary": stage1_rolling_summary,
            "stage2_physics_accuracy_summary": stage2_accuracy_summary,
            "stage2_physics_accuracy_rolling_summary": stage2_accuracy_rolling_summary,
            "transformer_cache_counts": {
                "train": len(out_buckets["train"]),
                "val": len(out_buckets["val"]),
                "held_out": len(out_buckets["held_out"]),
            },
            "stage3_status": stage3_result.get("status"),
            "stage3_error": stage3_result.get("error"),
            "fold_dir": str(fold_dir),
        }
        if "metrics" in stage3_result and isinstance(stage3_result["metrics"], dict):
            for key in (
                "test_total_loss",
                "test_torque_loss",
                "test_torque_mae",
                "best_val_total_loss",
            ):
                if key in stage3_result["metrics"]:
                    fold_summary[key] = stage3_result["metrics"][key]
            stage3_metrics_by_fold.append(stage3_result["metrics"])
            held_out_metrics = stage3_result["metrics"].get("held_out_metrics", {})
            held_out_metric_values = held_out_metrics.get("metrics", {}) if isinstance(held_out_metrics, dict) else {}
            if isinstance(held_out_metric_values, dict):
                _ts_print(f"[fold {held}] held-out test performance summary:")
                summary_pairs = []
                for key in (
                    "total_loss",
                    "torque_loss",
                    "cop_rmse_m",
                    "grf_rmse_N",
                    "moments_rmse_Nm",
                    "torque_rmse_selected_dofs_Nm",
                ):
                    if key in held_out_metric_values:
                        summary_pairs.append(f"{key}={held_out_metric_values[key]}")
                if summary_pairs:
                    _ts_print("  " + " | ".join(summary_pairs))
                else:
                    _ts_print("  held-out metrics present but expected summary keys not found.")

            held_out_infer_style = stage3_result["metrics"].get("held_out_infer_style", {})
            infer_overall_report = (
                held_out_infer_style.get("overall_mae_report", {})
                if isinstance(held_out_infer_style, dict)
                else {}
            )
            left_stance_torque_mae_per_dof = {}
            if isinstance(infer_overall_report, dict):
                avg_mae_map = infer_overall_report.get("average_mae_per_dof", {})
                if isinstance(avg_mae_map, dict):
                    left_stance_torque_mae_per_dof = {
                        str(k): float(v)
                        for k, v in avg_mae_map.items()
                        if v is not None
                    }
            if left_stance_torque_mae_per_dof:
                fold_summary["held_out_left_stance_torque_mae_per_dof"] = left_stance_torque_mae_per_dof
                _ts_print(
                    f"[fold {held}] held-out left-stance joint torque MAE "
                    f"(all predicted torque channels):"
                )
                for dof_name, mae_value in sorted(left_stance_torque_mae_per_dof.items()):
                    _ts_print(f"  {dof_name}: {mae_value:.4f}")

                knee_adduction_items = [
                    (name, value)
                    for name, value in left_stance_torque_mae_per_dof.items()
                    if ("knee_adduction" in name.lower()) or ("kam" in name.lower())
                ]
                if knee_adduction_items:
                    fold_summary["held_out_knee_adduction_metric"] = {
                        name: value for name, value in knee_adduction_items
                    }
                    _ts_print(f"[fold {held}] knee adduction metric(s):")
                    for metric_name, metric_value in knee_adduction_items:
                        _ts_print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    _ts_print(
                        f"[fold {held}] knee adduction metric not found in average_mae_per_dof keys."
                    )

        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fold_summary, default=str) + "\n")
        _write_json(fold_dir / "fold_summary.json", fold_summary)
        _ts_print(
            f"[fold {held}] done in {time.time()-fold_start:.1f}s. "
            f"stage3={stage3_result.get('status')}"
        )

    if stage3_metrics_by_fold:
        try:
            _write_combined_stage3_infer_summary(output_dir, stage3_metrics_by_fold, libs)
        except Exception as exc:
            tb = traceback.format_exc()
            _ts_print(f"[combined stage3] infer-style summary FAILED: {exc}\n{tb}")

    _ts_print(f"All folds processed.  Summary at {summary_path}")


if __name__ == "__main__":
    main()
