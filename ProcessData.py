"""
ProcessData.py
==============
Unified data-processing pipeline for the Calc_ID_MJX project.

Default mode reads raw Motion data (Motion_Pelvis_Adjusted/ preferred, Motion/
as fallback) and saves outputs to Trial/ProcessedData/.

With --OC_Mocap, the pipeline reads kinematics from Trial/MoCap and forces from
Trial/Motion, still builds the standard Trial/ProcessedData bundle, saves
trimmed motion-aligned MoCap outputs to Trial/MoCap, writes a raw-MoCap-time
companion bundle to Trial/MoCap_RawTimebase, and optionally writes a pre-trim
snapshot to Trial/MoCap/UntrimmedRaw.

After the core MJX pass, the script can also run the downstream post-processing
pipeline that used to live in ExtractGtoCalcRotation.py, CleanMissSteps.py,
cleanCOPOutliers.py, and extractFPA.py:
  1) calcaneus-frame COP / FPA outputs,
  2) miss-step cleanup with Untrimmed backups,
  3) normalized COP outlier cleanup on ground-aligned calc-frame COP.

Usage
-----
    python ProcessData.py                          # process all trials
    python ProcessData.py --dry-run                # discover only, no processing
    python ProcessData.py --subject 6GC            # single subject (exact match)
    python ProcessData.py --subjects 6GC,S8,S_GAH_1  # comma-separated list
    python ProcessData.py --OC_Mocap               # process MoCap inputs, save to Trial/MoCap/

Outputs
-------
Default mode saves to Trial/ProcessedData/. `--OC_Mocap` saves to Trial/MoCap/.
When `UseNoised` is enabled in default mode, the clean Pos/Vel/Accel-derived
files below remain the canonical ground-truth bundle, and a parallel
prediction-side bundle is also saved with a `_noised` suffix for the
kinematic/geometry files used by training and inference physics.

For `--OC_Mocap`, the main `Trial/MoCap/` bundle remains the motion-aligned
working timeline used by the existing pipeline, while `Trial/MoCap_RawTimebase/`
stores the same processed signals projected onto the trimmed raw MoCap timebase.

  pos_inputs.npy              (T, 18)   – joint angles incl. knees; no pelvis XYZ or MTP joints
  vel_inputs.npy              (T, 21)   – velocities incl. knees; no MTP joints
  acc_inputs.npy              (T, 21)   – accelerations incl. knees; no MTP joints
  pelvis_rot_matrix.npy       (T, 6)    – first 2 cols of R_ZXY per frame
  pos_mjx.npy                 (T, nq)   – full qpos in MuJoCo coords
  qvel_mjx.npy                (T, nv)   – full qvel in MuJoCo coords
  qacc_mjx.npy                (T, nv)   – full qacc in MuJoCo coords
  GRF_Cleaned.npy             (T, 6)    – [Rx,Ry,Rz, Lx,Ly,Lz] MuJoCo Z-up
  Moment_Cleaned.npy          (T, 6)
  COP_Cleaned_Relative.npy    (T, 4)    – [Rx,Ry, Lx,Ly] rel to ankle, floor-corr.
  ankle_heights.npy           (T, 2)    – [R_Z, L_Z] floor-corrected
  ankle_pos_r.npy             (T, 3)    – floor-corrected right ankle XYZ
  ankle_pos_l.npy             (T, 3)    – floor-corrected left ankle XYZ
  contactBoolean.npy          (T, 2)
  COM_r.npy, COM_l.npy        (T, 3)    – COM relative to floor-corrected ankle
  COM_Acc_Global.npy          (T, 3)    – body COM acceleration (double gradient)
  Jacobian.npy                dict      – {jacp, jacr, body_ids}
  ID_GT_MJX.npy               (T, 23)   – independent DOFs, qfrc_inverse − qfrc_grf_contribution
  qfrc_inverse.npy            (T, 23)   – independent DOFs, raw before GRF subtraction
  Height_m.npy                (T,)
  Mass_kg.npy                 (T,)
  Trial_Processing_Information.json
  Trimming_Traceability.json       – source timebases, interpolation map, every trim, output alignment
  FootProgressionAngle.npy    (T, 2)
  Foot_ProgressionAngle.npy   (T, 2)    – legacy-compatible alias
  tosPosition.npy             (2, T, 3) – [left, right] toes world positions
  knee_pos_r.npy              (T, 3)    – right knee global XYZ
  knee_pos_l.npy              (T, 3)    – left knee global XYZ
  KneeToCOP_Vectors.npy       (T, 6)    – [R knee→COP xyz, L knee→COP xyz]
  COP_CalcFrame.npy           (T, 6)
  COP_CalcFrame_GroundAligned.npy              (T, 6)
  COP_CalcFrame_GroundAligned_GRFNorm.npy      (T, 6) – ground-aligned COP/height × |GRF|/BW
  COP_CalcFrame_GroundAligned_YplusAnkleHeight.npy (T, 6)
  COP_CalcFrame_GroundAligned_BackToWorld.npy  (T, 6)
  COP_Cleaned_Relative_RecoveredFromGroundAligned.npy (T, 4)
  WorldToGroundAlignedCalcnRotation.npy        (T, 2, 3, 3)
  CalcnToFloor_AngleDeg.npy   (T, 2)
  Untrimmed/                  backup of pre-miss-step processed outputs
  [optional] GRF_average_reconstructed.npy  (T, 6)
  [optional] COP_average_reconstructed.npy  (T, 4)
  [optional] Moment_average_reconstructed.npy (T, 6)
"""

# ─────────────────────────────────────────────────────────────
#  ENVIRONMENT  (must be set before importing JAX)
# ─────────────────────────────────────────────────────────────
import os
import sys


def _preparse_compute_device(argv: list[str]) -> str:
    """Read --device before importing JAX, without consuming argparse input."""
    value = "cpu"
    for index, argument in enumerate(argv):
        if argument.startswith("--device="):
            value = argument.split("=", 1)[1]
            break
        if argument == "--device" and index + 1 < len(argv):
            value = argv[index + 1]
            break
    value = str(value).strip().lower()
    if value not in {"cpu", "gpu", "auto"}:
        raise SystemExit(
            f"Invalid --device {value!r}; expected one of: cpu, gpu, auto"
        )
    return value


def _preparse_worker_count(argv: list[str], default: int = 1) -> int:
    """Read --workers before importing numerical runtimes."""
    value = default
    for index, argument in enumerate(argv):
        if argument.startswith("--workers="):
            value = argument.split("=", 1)[1]
            break
        if argument == "--workers" and index + 1 < len(argv):
            value = argv[index + 1]
            break
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


_REQUESTED_COMPUTE_DEVICE = _preparse_compute_device(sys.argv[1:])
_REQUESTED_WORKERS = _preparse_worker_count(sys.argv[1:], default=16)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
if _REQUESTED_COMPUTE_DEVICE == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
    if _REQUESTED_WORKERS > 1:
        # A spawned JAX process otherwise creates a CPU thread pool sized for
        # the whole machine. Limit each worker to its fair share of cores so
        # N workers do not create N full-machine thread pools.
        _thread_override = os.environ.get("PROCESSDATA_CPU_THREADS_PER_WORKER")
        if _thread_override is not None:
            try:
                _CPU_THREADS_PER_WORKER = max(1, int(_thread_override))
            except ValueError as exc:
                raise SystemExit(
                    "PROCESSDATA_CPU_THREADS_PER_WORKER must be a positive integer"
                ) from exc
        else:
            _CPU_THREADS_PER_WORKER = max(
                1, (os.cpu_count() or 1) // _REQUESTED_WORKERS
            )
        _thread_count_text = str(_CPU_THREADS_PER_WORKER)
        for _thread_env_name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[_thread_env_name] = _thread_count_text
        _xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_cpu_multi_thread_eigen" not in _xla_flags:
            _xla_flags = f"{_xla_flags} --xla_cpu_multi_thread_eigen=false".strip()
        if "intra_op_parallelism_threads" not in _xla_flags:
            _xla_flags = (
                f"{_xla_flags} intra_op_parallelism_threads="
                f"{_CPU_THREADS_PER_WORKER}"
            ).strip()
        os.environ["XLA_FLAGS"] = _xla_flags
    else:
        _CPU_THREADS_PER_WORKER = None
elif _REQUESTED_COMPUTE_DEVICE == "gpu":
    # JAX's platform selector is named "cuda"; jax.default_backend() reports "gpu".
    os.environ["JAX_PLATFORMS"] = "cuda"
else:
    # Let JAX choose the best available backend.
    os.environ.pop("JAX_PLATFORMS", None)
    _CPU_THREADS_PER_WORKER = None

# ─────────────────────────────────────────────────────────────
#  STANDARD IMPORTS
# ─────────────────────────────────────────────────────────────
import argparse
import gc
import hashlib
import json
import pickle
import shutil
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from scipy.interpolate import interp1d, make_smoothing_spline
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.transform import Rotation as R
try:
    from ProcessAddbiomechnics.updateModel import fix_xml_masses, knee_coupling_is_canonical_xml
except Exception:
    fix_xml_masses = None
    knee_coupling_is_canonical_xml = None

# ─────────────────────────────────────────────────────────────
#  CONFIG  –  edit these values before running
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # "DATA_ROOT":                    "Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects",
    # "DATA_ROOT":                    "Datasets_NAS/DifferentNoisedDataset/TrustedDataSetNoised12DistributedUnFiltered",
    # "DATA_ROOT":                    "Datasets_NAS/OldYoungAdultWalking_MJX_Processed_NoTrim_NoFilt_OSIDFilt",  # for testing: smaller dataset with identical structure
    # "DATA_ROOT":                    "OlderYoungerAdultDataset_PostVisuallyTrimmed",  # for testing: smaller dataset with identical structure
    # "DATA_ROOT":                    "OldYoungAdultWalking_MJX_Processed",
    # "DATA_ROOT":                    "OpenCapSubjects_Filt",
    # "DATA_ROOT":                    "PD_Dataset",   # 20-trial evenly distributed PD subset run
    "DATA_ROOT":                    "TrustedDataSetNoised12Distributed_EdgeHold_AllPatients",
    "NUM_WORKERS":                  16,    # CPU workers use spawn so each gets a clean JAX runtime
    "SUBJECTS_TO_PROCESS":           None,
    # "SUBJECTS_TO_PROCESS":           [
    #     "PD_SUB01_off", "PD_SUB02_off", "PD_SUB03_on", "PD_SUB05_on", "PD_SUB06_on",
    #     "PD_SUB08_off", "PD_SUB10_off", "PD_SUB11_off", "PD_SUB12_off", "PD_SUB13_on",
    #     "PD_SUB14_on", "PD_SUB16_off", "PD_SUB17_on", "PD_SUB18_on", "PD_SUB19_on",
    #     "PD_SUB20_on", "PD_SUB22_off", "PD_SUB23_on", "PD_SUB24_on", "PD_SUB26_off",
    # ],
    # "SUBJECTS_TO_PROCESS":           ["1GC","GaitRetraining_Subject103","S1"],  # None = all subjects, or comma-separated list e.g. "6GC,S8,S_GAH_1"
    # "SUBJECTS_TO_PROCESS":           ["GaitRetraining_SubjectR417","02","03"],  # None = all subjects, or comma-separated list e.g. "6GC,S8,S_GAH_1"
    # "SUBJECTS_TO_PROCESS":           ["GaitRetraining_SubjectR241"],
    # "SUBJECTS_TO_PROCESS":           [
    #     "SUBJ09", "SUBJ10", "SUBJ100", "SUBJ119", "SUBJ120",
    #     "SUBJ123", "SUBJ125", "SUBJ131", "SUBJ25", "SUBJ32",
    #     "SUBJ33", "SUBJ36", "SUBJ40", "SUBJ50",
    #     "TVC03", "TVC04", "TVC36", "TVC53", "TVC60",
    # ],   # None = all subjects, or comma-separated list e.g. "6GC,S8,S_GAH_1"
    # "TRIALS_TO_PROCESS":             [
    #     # 20 evenly distributed trials from PD_Dataset. Set to None or [] to disable trial-level filtering.
    #     "PD_SUB01_off/Trial_1",
    #     "PD_SUB02_off/Trial_14",
    #     "PD_SUB03_on/Trial_17",
    #     "PD_SUB05_on/Trial_5",
    #     "PD_SUB06_on/Trial_8",
    #     "PD_SUB08_off/Trial_17",
    #     "PD_SUB10_off/Trial_12",
    #     "PD_SUB11_off/Trial_15",
    #     "PD_SUB12_off/Trial_4",
    #     "PD_SUB13_on/Trial_10",
    #     "PD_SUB14_on/Trial_12",
    #     "PD_SUB16_off/Trial_5",
    #     "PD_SUB17_on/Trial_11",
    #     "PD_SUB18_on/Trial_13",
    #     "PD_SUB19_on/Trial_17",
    #     "PD_SUB20_on/Trial_4",
    #     "PD_SUB22_off/Trial_10",
    #     "PD_SUB23_on/Trial_17",
    #     "PD_SUB24_on/Trial_6",
    #     "PD_SUB26_off/Trial_9",
    # ],   # None or [] = no per-trial filtering. Format: "<Subject>/<Trial>" exact-match.
    # "TRIALS_TO_PROCESS":             [
    #     # 20 random trials from StrokeDataset (seed=42). Set to None or [] to disable trial-level filtering.
    #     "SUBJ09/Trial_2",   "SUBJ10/Trial_1",   "SUBJ100/Trial_1",
    #     "SUBJ100/Trial_3",  "SUBJ119/Trial_2",  "SUBJ120/Trial_2",
    #     "SUBJ123/Trial_1",  "SUBJ125/Trial_3",  "SUBJ131/Trial_3",
    #     "SUBJ25/Trial_3",   "SUBJ32/Trial_2",   "SUBJ33/Trial_3",
    #     "SUBJ36/Trial_3",   "SUBJ40/Trial_4",   "SUBJ50/Trial_3",
    #     "TVC03/Trial_3",    "TVC04/Trial_3",    "TVC36/Trial_2",
    #     "TVC53/Trial_1",    "TVC60/Trial_2",
    # ],   # None or [] = no per-trial filtering. Format: "<Subject>/<Trial>" exact-match.
    "OC_Mocap":                     False,  # Use Trial/MoCap kinematics + Trial/Motion forces, save to Trial/MoCap
    # Filtering
    "FILTER_CUTOFF_HZ":             6.0,
    "FILTER_ORDER":                 2,
    "SAMPLING_RATE_HZ":             100.0,
    "ENABLE_KINEMATICS_FILTERING":  True,   # if False, skip the 6 Hz Butterworth filter on Pos/Vel/Accel
    "OS_Filtering":                 False,  # if True, ALSO compute MJX ID with OpenSim-style GCVSpline vel/accel -> ProcessedData/OpenSimFiltering/
    # Per-channel kinematics-filter overrides (for the filter-ablation study). Each
    # defaults to None = "follow ENABLE_KINEMATICS_FILTERING / FILTER_CUTOFF_HZ", so
    # leaving them None reproduces the canonical pipeline byte-for-byte.
    "FILTER_POS":                   None,   # None -> global; else bool (filter Pos channel)
    "FILTER_VEL":                   None,   # None -> global; else bool (filter Vel channel)
    "FILTER_ACCEL":                 None,   # None -> global; else bool (filter Accel channel)
    "FILTER_CUTOFF_POS_HZ":         None,   # None -> FILTER_CUTOFF_HZ; else float (Hz)
    "FILTER_CUTOFF_VEL_HZ":         None,   # None -> FILTER_CUTOFF_HZ; else float (Hz)
    "FILTER_CUTOFF_ACCEL_HZ":       None,   # None -> FILTER_CUTOFF_HZ; else float (Hz)
    "OUTPUT_SUBDIR_NAME":           "ProcessedData",  # write target subdir (ablation uses ProcessedData_ablCx)
    "TRIALS_TO_PROCESS":            None,   # None/[] = all; else list of "Subject/Trial" exact-match

    # Floor / contact
    "ENABLE_FLOOR_CORRECTION":      False,  # if False, do not subtract estimated floor height from pelvis/ankle Z
    "FLOOR_TROUGH_PERCENTILE":      10.0,
    "FLOOR_TROUGH_OFFSET_M":        -0.015,
    "FLOOR_MIN_TROUGHS_FOR_DIRECT_PERCENTILE": 5,
    "FLOOR_INTERP_SAMPLES":         200,
    "GRF_STANCE_THRESHOLD":         15.0,   # for outlier-stance removal
    "GRF_CONTACT_THRESHOLD":        1.0,    # for contact boolean & COP masking

    # COP cleaning
    "ENABLE_GRF_FILTERING":         True,   # if False, skip segment-wise GRF/GRM filtering
    "USE_NOFILTER_GRF_FOR_TORQUE":  True,   # if True, use trimmed GRF_NoFilt_Trimmed.npy for qfrc_grf/ID_GT_MJX force terms
    "ENABLE_COP_CLEANING":          True,   # if False, use frame-converted raw COP relative to ankle
    "COP_TRIM_START_FRAMES":        3,
    "COP_TRIM_END_FRAMES":          3,
    "COP_FILTER_PAD_WIDTH":         15,
    "COP_EXTRAPOLATION_FRAMES":     6,
    "COP_EdgeHold":                 True,  # True=edge-hold padding for COP filtfilt; False=zero padding

    "ENABLE_SHORT_STANCE_ZEROING":  True,   # zero GRF/COP for non-edge short/low-peak stances
    "SHORT_STANCE_MAX_FRAMES":      25,
    "SHORT_STANCE_MIN_PEAK_N":      50.0,

    # Processing flags
    "UseNoised":                  True,  # Save clean GT from Pos/Vel/Accel and, when available, a parallel `_noised` prediction bundle from Pos_noised/Vel_noised/Accel_noised; forced off for OC_Mocap
    "OnlyProcessNoised":          False,  # Only rebuild the `_noised` prediction bundle; requires existing clean ProcessedData files when used with --only-new checks
    "OnlyProcessOverGround":      False,   # Pre-filter trials from raw Pos.npy and only process non-treadmill trials
    "ONLY_PROCESS_NEW":             False,  # skip if target outputs already exist
    "TrimGRFMissSteps":             True,    # if False, skip the 1.43x body-weight GRF misstep trim
    "TRIM_TO_DOUBLE_SUPPORT":       False,   # trim trial edges to first/last double-support frame
    "ENABLE_GRF_TRIM":              True,    # primary cut: remove leading/trailing zero-GRF frames (+ optional misstep removal)
    "TRIM_WEAK_EDGE_STANCES":       False,    # secondary pass: trim overground edge stances whose mean vGRF/BW is below threshold
    "TRIM_WEAK_STANCE_MIN_FRAMES":  5,      # minimum stance length (matches visualization stance stats)
    "TRIM_WEAK_STANCE_BW_FRACTION": 0.65,   # mean vGRF threshold as fraction of body weight
    "ENABLE_OUTLIER_STANCE_REMOVAL": False,   # tertiary pass: remove unusually-long outlier stances and keep longest valid segment
    "TIME_ALIGNMENT_TARGET":        "motion",  # For OC_Mocap: "motion" or "mocap"
    "SaveUntrimmedOutputs":         True,   # For OC_Mocap: save original-timebase pre-trim snapshot to MoCap/UntrimmedRaw
    "RUN_CALC_FRAME_POSTPROCESS":   True,   # Run ExtractGtoCalcRotation/extractFPA style outputs after core processing
    "RUN_MISSSTEP_POSTPROCESS":     False,    # Run CleanMissSteps style backup/rettrim pass after calc-frame outputs
    "RUN_COP_OUTLIER_POSTPROCESS":  False,   # Run cleanCOPOutliers style cleanup after miss-step trimming
    "MISSSTEP_MAX_DURATION_S":      5.0,
    "MISSSTEP_FS_HZ":               100.0,
    "MISSSTEP_HALF_RATIO_THRESHOLD": 0.8,
    "MISSSTEP_PEAK_OFFSET_FRAMES":  2,
    "MISSSTEP_DOUBLE_SUPPORT_EDGE_TRIM_FRAMES": 25,
    "COP_OUTLIER_MOVE_THRESHOLD_PCT": 10.0,
    "COP_OUTLIER_MOVE_BAD_TRIALS":  False,  # Disabled by default: moving trials is more destructive than in-place repair
    "COP_OUTLIER_BAD_TRIALS_ROOT_NAME": "TrustedDataSetNoised12DistributedBad",

    # Deviation learning (only used when DO_DEVIATION_LEARNING_PREP=True)
    "DEVIATION_LOAD_AVERAGES_FROM_FILE":  True,
    "DEVIATION_METRICS_PKL_PATH":         "processed_stance_metrics.pkl",
    "DO_DEVIATION_LEARNING_PREP":         False,
    # Outlier stance removal (SecondaryProcessing style)
    "MIN_STANCES_FOR_CHECK":        3,
    "MIN_STANCE_LENGTH_FOR_FLAG":   10,   # frames
    # Memory controls
    "ID_BATCH_CHUNK_SIZE":          100,  # frames per MJX inverse-dynamics chunk
    "MIN_RAM_GB_PER_WORKER":        2.0,  # soft cap for parallel workers (raised from 4.0)
    "MAX_TASKS_PER_CHILD":          4,    # amortize spawn/JAX startup, then release worker caches
    # Model XML handling
    "UsedFIXEDModels":              True,   # Prefer existing MyosuiteModel_FIXED.xml when available
    "DontUseFixed":                 False,  # Use raw MyosuiteModel.xml instead of MyosuiteModel_FIXED.xml
    "RescaleModelsToEstimatedMass": False,  # Rescale the XML actually used by processing to Patient_MD Mass_kg.
}

# ─────────────────────────────────────────────────────────────
#  MuJoCo qpos / save column mapping
# ─────────────────────────────────────────────────────────────
POS_COLUMNS = (
    "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)

POS_ANGLE_COLUMN_IDXS = tuple(i for i in range(len(POS_COLUMNS)) if i not in (3, 4, 5))

CANONICAL_SAVE_DOF_NAMES = (
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "walker_knee_r_translation1", "walker_knee_r_translation2", "walker_knee_r_translation3",
    "knee_angle_r", "walker_knee_r_rotation2", "walker_knee_r_rotation3",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "walker_knee_l_translation1", "walker_knee_l_translation2", "walker_knee_l_translation3",
    "knee_angle_l", "walker_knee_l_rotation2", "walker_knee_l_rotation3",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)

MODEL_SAVE_DOF_NAMES = (
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)

# Coupled-DOF indices filled in by XML at runtime (keys: slave qpos index)
# will be populated in calculate_coupled_coordinates_automated()


# ═══════════════════════════════════════════════════════════════
#                    UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════


# --- Extracted to processing/filtering.py in Stage 6; re-exported so every
# --- existing caller in this file continues to resolve them unchanged.
from processing.filtering import (  # noqa: E402,F401
    apply_kinematics_filtering,
    butter_lowpass_filter,
    filter_segment_wise,
    gcv_derivatives,
)

# --- Extracted to processing/trial_io.py in Stage 6; re-exported unchanged.
from processing.trial_io import (  # noqa: E402,F401
    _copy_outputs_with_suffix,
    _has_noised_prediction_bundle,
    _has_noised_source_inputs,
    _missing_noised_bundle_files,
    _with_file_suffix,
)

# --- Extracted to processing/artifact_names.py in Stage 6; re-exported unchanged.
from processing.artifact_names import (  # noqa: E402,F401
    NOISED_AUX_FILES_TO_COPY,
    NOISED_FILE_SUFFIX,
    NOISED_REQUIRED_BUNDLE_FILENAMES,
    NOISED_STRICT_VALIDATION_FILENAMES,
    TRIMMING_TRACE_FILENAME,
)

# --- Extracted to processing/geometry.py in Stage 6; re-exported unchanged.
from processing.geometry import (  # noqa: E402,F401
    _add_ankle_height_to_ground_aligned_y,
    _apply_rotation_batch,
    _build_ground_aligned_rotation,
    _compose_world_to_ground_aligned,
    _compute_foot_progression_angle_deg,
    _compute_knee_to_cop_vectors,
    _extract_ground_to_calc_rotations_from_qpos,
    _normalize,
    _toe_trough_indices,
    align_myosuite_pelvis,
    estimate_floor_height_from_toe_troughs,
)

# --- Extracted to processing/resampling.py in Stage 6; re-exported unchanged.
from processing.resampling import (  # noqa: E402,F401
    _interpolate_101,
    _interpolate_to_len,
    resample_dataframes_to_uniform_timestep,
)

# --- Extracted to processing/cop.py in Stage 6; re-exported unchanged.
from processing.cop import (  # noqa: E402,F401
    _multiply_cop_by_bodyweight_normalized_grf_magnitude,
    clean_and_filter_cop,
)

# --- Extracted to processing/contact.py in Stage 6; re-exported unchanged.
from processing.contact import (  # noqa: E402,F401
    _detect_stance_phases,
    _stance_segments,
    create_contact_boolean,
    get_stance_phases,
    zero_short_grf_cop_stances,
)









def convert_to_mujoco_coords(vec: np.ndarray) -> np.ndarray:
    """
    Convert OpenSim [X, Y, Z] (Y-up) to MuJoCo [X, -Z, Y] (Z-up).
    Works on 1-D (3,) or 2-D (T, 3) arrays.
    """
    if vec.ndim == 1:
        return np.array([vec[0], -vec[2], vec[1]])
    out = np.empty_like(vec)
    out[:, 0] =  vec[:, 0]
    out[:, 1] = -vec[:, 2]
    out[:, 2] =  vec[:, 1]
    return out


def find_longest_valid_segment(mask: np.ndarray):
    """Returns (start_idx, end_idx) of the longest True run in *mask*."""
    padded = np.concatenate(([False], mask, [False]))
    diffs  = np.diff(padded.astype(int))
    starts = np.where(diffs ==  1)[0]
    ends   = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0, len(mask)
    lengths = ends - starts
    best    = np.argmax(lengths)
    return int(starts[best]), int(ends[best])


def _joint_id(model, joint_name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)


def _joint_dof_count(model, jid: int) -> int:
    jtype = int(model.jnt_type[jid])
    if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1


def _build_name_to_qpos_index(model, joint_names=POS_COLUMNS) -> dict[str, int]:
    """Return validated 1-DOF joint-name -> qpos-index mapping."""
    missing = []
    mapping: dict[str, int] = {}
    for name in joint_names:
        jid = _joint_id(model, name)
        if jid < 0:
            missing.append(name)
            continue
        dof_count = _joint_dof_count(model, jid)
        if dof_count != 1:
            raise ValueError(
                f"Joint '{name}' must be a 1-DOF hinge/slide joint; "
                f"model reports {dof_count} DOFs."
            )
        mapping[name] = int(model.jnt_qposadr[jid])
    if missing:
        raise ValueError(
            "Model is missing required kinematics joints: " + ", ".join(missing)
        )
    if len(set(mapping.values())) != len(mapping):
        duplicates = {}
        for name, idx in mapping.items():
            duplicates.setdefault(idx, []).append(name)
        dup_text = "; ".join(
            f"qpos[{idx}]=" + ",".join(names)
            for idx, names in duplicates.items() if len(names) > 1
        )
        raise ValueError(f"Kinematics joints map to duplicate qpos indices: {dup_text}")
    return mapping


def map_patient_to_qpos(pos_row: np.ndarray,
                        model,
                        pos_columns=POS_COLUMNS,
                        name_to_qpos: dict[str, int] | None = None) -> np.ndarray:
    """Map one named kinematics row to this model's qpos vector."""
    if len(pos_row) < len(pos_columns):
        raise ValueError(
            f"Expected at least {len(pos_columns)} kinematics columns, got {len(pos_row)}"
        )
    if name_to_qpos is None:
        name_to_qpos = _build_name_to_qpos_index(model, pos_columns)
    qpos = np.zeros(int(model.nq), dtype=np.float64)
    for col_idx, joint_name in enumerate(pos_columns):
        qpos[name_to_qpos[joint_name]] = pos_row[col_idx]
    return qpos


def canonical_save_indices(model, dof_names=CANONICAL_SAVE_DOF_NAMES) -> np.ndarray:
    """Return model DOF indices for the canonical lumbar-down save layout."""
    missing = []
    indices = []
    for name in dof_names:
        jid = _joint_id(model, name)
        if jid < 0:
            missing.append(name)
            continue
        dof_count = _joint_dof_count(model, jid)
        if dof_count != 1:
            raise ValueError(
                f"Canonical save joint '{name}' must be 1-DOF; "
                f"model reports {dof_count} DOFs."
            )
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        if qadr != dadr:
            raise ValueError(
                f"Canonical save joint '{name}' has qpos adr {qadr} != dof adr {dadr}; "
                "this pipeline expects nq == nv with scalar joints."
            )
        indices.append(dadr)
    if missing:
        raise ValueError(
            "Model is missing canonical save joints: " + ", ".join(missing)
        )
    if len(set(indices)) != len(indices):
        raise ValueError("Canonical save DOF names resolved to duplicate indices")
    return np.asarray(indices, dtype=np.int64)


def independent_model_save_indices(model) -> np.ndarray:
    """Resolve the independent training/evaluation DOF schema for any source model.

    Trusted/OpenCap XMLs can have different widths because they may include
    patella, coupled knee, trunk, or arm coordinates. We save model-space arrays
    by joint name so qfrc_inverse, qfrc_grf, ID targets, qpos/qvel/qacc, and
    Jacobian columns always share the same independent layout.
    """
    try:
        return canonical_save_indices(model, MODEL_SAVE_DOF_NAMES)
    except ValueError as exc:
        available = []
        try:
            for jid in range(int(model.njnt)):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if name:
                    available.append(str(name))
        except Exception:
            available = []
        preview = ", ".join(available[:80])
        if len(available) > 80:
            preview += ", ..."
        raise ValueError(
            "Unable to resolve independent 23-DOF save schema for this model. "
            f"Required joints: {', '.join(MODEL_SAVE_DOF_NAMES)}. "
            f"Available joints ({len(available)}): {preview}"
        ) from exc


def save_dof_indices_for_model(model, cfg: dict | None = None) -> np.ndarray:
    """Return the independent-DOF model layout used by training/evaluation."""
    return independent_model_save_indices(model)


def jacobian_save_dof_indices_for_model(model) -> np.ndarray:
    """Return the independent-DOF model layout for Jacobian columns."""
    return independent_model_save_indices(model)


def slice_jacobian_dofs(jacobian_data: dict, dof_indices: np.ndarray) -> dict:
    """Slice Jacobian payload on its nv axis, leaving body ids untouched."""
    return {
        "jacp": np.asarray(jacobian_data["jacp"])[..., dof_indices],
        "jacr": np.asarray(jacobian_data["jacr"])[..., dof_indices],
        "body_ids": jacobian_data["body_ids"],
    }


def normalize_kinematic_angle_units(pos: np.ndarray,
                                    vel: np.ndarray | None = None,
                                    accel: np.ndarray | None = None,
                                    *,
                                    context: str = ""):
    """Auto-convert degree-scale angle columns to radians, leaving pelvis XYZ in meters."""
    pos = np.asarray(pos, dtype=np.float64).copy()
    angle_cols = np.asarray(POS_ANGLE_COLUMN_IDXS, dtype=np.int64)
    angle_absmax = float(np.nanmax(np.abs(pos[:, angle_cols]))) if pos.size else 0.0
    if angle_absmax > (2.0 * np.pi + 1e-6):
        pos[:, angle_cols] = np.deg2rad(pos[:, angle_cols])
        if vel is not None:
            vel = np.asarray(vel, dtype=np.float64).copy()
            vel[:, angle_cols] = np.deg2rad(vel[:, angle_cols])
        if accel is not None:
            accel = np.asarray(accel, dtype=np.float64).copy()
            accel[:, angle_cols] = np.deg2rad(accel[:, angle_cols])
        label = f" [{context}]" if context else ""
        print(f"    [Kinematic Units]{label} converted angle columns deg -> rad (absmax={angle_absmax:.2f} deg)")
    return pos, vel, accel


def calculate_coupled_coordinates_automated(qpos:     np.ndarray,
                                             qvel:     np.ndarray,
                                             qacc:     np.ndarray,
                                             xml_path: Path) -> tuple:
    """
    Parse XML equality constraints and apply polynomial coupling.

    The XML format is:
        joint1=<slave>  joint2=<master>  polycoef="c0 c1 c2 c3 c4"
    meaning  q_slave = poly(q_master).

    In this model the walker-knee sub-joints (translation1/2, rotation2/3) are
    the *slaves* listed as joint1, while knee_angle_r/l is the *master* listed
    as joint2.  We therefore always read theta from joint2 (master) and write
    the result into joint1 (slave).

    Returns updated (qpos, qvel, qacc).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    couplings = []
    for eq in root.iter("equality"):
        for weld in eq.iter("joint"):
            slave_name  = weld.get("joint1")
            master_name = weld.get("joint2")
            # Skip single-joint locks (joint2 absent) — passing None to
            # mj_name2id causes a MuJoCo segfault.
            if slave_name is None or master_name is None:
                continue
            poly   = weld.get("polycoef", "0 1 0 0 0").split()
            coeffs = [float(c) for c in poly]
            couplings.append((slave_name, master_name, coeffs))

    if not couplings:
        return qpos, qvel, qacc

    # We need a model to get joint → qpos index mapping
    try:
        mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception:
        return qpos, qvel, qacc

    qpos_out = qpos.copy()
    qvel_out = qvel.copy()
    qacc_out = qacc.copy()

    for slave_name, master_name, coeffs in couplings:
        slave_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, slave_name)
        master_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, master_name)
        if slave_id < 0 or master_id < 0:
            continue

        d_slave  = mj_model.jnt_dofadr[slave_id]
        d_master = mj_model.jnt_dofadr[master_id]

        # theta is the PRIMARY (master) DOF — e.g. knee_angle_r
        theta = qpos_out[:, d_master]

        # Polynomial:  q_slave = c0 + c1*θ + c2*θ² + c3*θ³ + c4*θ⁴
        c = coeffs + [0.0] * (5 - len(coeffs))
        q_slave = c[0] + c[1]*theta + c[2]*theta**2 + c[3]*theta**3 + c[4]*theta**4

        # Velocity:  dq_slave/dt = (c1 + 2c2*θ + 3c3*θ² + 4c4*θ³) * dθ/dt
        dq_dtheta = c[1] + 2*c[2]*theta + 3*c[3]*theta**2 + 4*c[4]*theta**3
        v_slave   = dq_dtheta * qvel_out[:, d_master]

        # Acceleration (chain rule)
        d2q_dtheta2 = 2*c[2] + 6*c[3]*theta + 12*c[4]*theta**2
        a_slave = (dq_dtheta * qacc_out[:, d_master]
                   + d2q_dtheta2 * qvel_out[:, d_master]**2)

        # Write into the SLAVE slot — never overwrite the master (knee_angle)
        qpos_out[:, d_slave] = q_slave
        qvel_out[:, d_slave] = v_slave
        qacc_out[:, d_slave] = a_slave

    return qpos_out, qvel_out, qacc_out


def compute_patient_size(xml_path: Path) -> np.ndarray:
    """
    Compute segment lengths from the FIXED XML model at the default (zero) pose.

    Segments measured:
      - Tibia length  : calcn (heel) → tibia origin, averaged R+L
      - Femur length  : tibia origin → femur origin (hip joint), averaged R+L
      - Foot length   : toes → calcn, averaged R+L
      - Pelvis width  : distance between right and left hip joint centres

    Returns a 1-D array of shape (4,):
        [avg_tibia_length, avg_femur_length, avg_foot_length, pelvis_width]
    """
    try:
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        d = mujoco.MjData(m)
        # Use qpos0 (default pose – all zeros for this model) which is already set
        mujoco.mj_forward(m, d)

        def _body_pos(name: str) -> np.ndarray:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            return d.xpos[bid].copy()

        # ── Joint / body positions ────────────────────────────────
        # Tibia: from knee joint centre (tibia_r/l body) to ankle joint (talus_r/l body)
        tibia_r   = np.linalg.norm(_body_pos("tibia_r")  - _body_pos("talus_r"))
        tibia_l   = np.linalg.norm(_body_pos("tibia_l")  - _body_pos("talus_l"))

        # Femur: from hip joint centre (femur_r/l body) to knee joint (tibia_r/l body)
        femur_r   = np.linalg.norm(_body_pos("femur_r")  - _body_pos("tibia_r"))
        femur_l   = np.linalg.norm(_body_pos("femur_l")  - _body_pos("tibia_l"))

        # Foot: from heel (calcn) to toes
        foot_r    = np.linalg.norm(_body_pos("calcn_r")  - _body_pos("toes_r"))
        foot_l    = np.linalg.norm(_body_pos("calcn_l")  - _body_pos("toes_l"))

        # Pelvis width: distance between femur_r and femur_l origins (hip joint centres)
        pelvis_w  = np.linalg.norm(_body_pos("femur_r")  - _body_pos("femur_l"))

        avg_tibia = (tibia_r + tibia_l) / 2.0
        avg_femur = (femur_r + femur_l) / 2.0
        avg_foot  = (foot_r  + foot_l)  / 2.0

        return np.array([avg_tibia, avg_femur, avg_foot, pelvis_w], dtype=np.float32)

    except Exception as e:
        warnings.warn(f"compute_patient_size failed for {xml_path}: {e}")
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


PIPELINE_SKIP_DIR_NAMES = {"Untrimmed", "UntrimmedRaw"}
OC_MOCAP_RAW_TIMEBASE_DIRNAME = "MoCap_RawTimebase"
PIPELINE_DERIVED_FILENAMES = {
    "COP_CalcFrame.npy",
    "COP_CalcFrame_GroundAligned.npy",
    "COP_CalcFrame_GroundAligned_GRFNorm.npy",
    "COP_CalcFrame_GroundAligned_YplusAnkleHeight.npy",
    "COP_CalcFrame_GroundAligned_BackToWorld.npy",
    "COP_Cleaned_Relative_RecoveredFromGroundAligned.npy",
    "KneeToCOP_Vectors.npy",
    "knee_pos_r.npy",
    "knee_pos_l.npy",
    "WorldToGroundAlignedCalcnRotation.npy",
    "CalcnToFloor_AngleDeg.npy",
    "FootProgressionAngle.npy",
    "Foot_ProgressionAngle.npy",
    "tosPosition.npy",
    "Trial_Processing_Information.json",
    TRIMMING_TRACE_FILENAME,
}
MISSSTEP_ANALYSIS_REQUIRED_FILES = ("GRF_Cleaned.npy", "Mass_kg.npy", "contactBoolean.npy")
COP_OUTLIER_FILENAME = "COP_CalcFrame_GroundAligned.npy"
HEIGHT_FILENAME = "Height_m.npy"
COP_OUTLIER_CHANNEL_BOUNDS = {
    0: (-0.015, 0.145),  # Rx
    3: (-0.015, 0.145),  # Lx
    2: (-0.020, 0.025),  # Rz
    5: (-0.025, 0.020),  # Lz
}


def _load_json_dict(path: Path) -> dict:
    """Best-effort JSON loader that always returns a dict."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without leaving a partially-written provenance record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _array_source_record(path: Path, arr: np.ndarray, trial_path: Path) -> dict[str, Any]:
    """Describe an input array while keeping paths relocatable with the trial."""
    try:
        rel_path = str(path.relative_to(trial_path))
    except ValueError:
        rel_path = str(path)
    stat = path.stat() if path.exists() else None
    sha256 = None
    if path.exists() and path.is_file():
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        sha256 = digest.hexdigest()
    return {
        "path_relative_to_trial": rel_path,
        "shape": [int(v) for v in np.asarray(arr).shape],
        "dtype": str(np.asarray(arr).dtype),
        "file_size_bytes": int(stat.st_size) if stat else None,
        "file_modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else None,
        "sha256": sha256,
    }


def _file_identity_record(path: Path, relative_to: Path) -> dict[str, Any]:
    """Stable identity for a non-array dependency such as the model XML."""
    try:
        display_path = str(path.relative_to(relative_to))
    except ValueError:
        display_path = str(path)
    if not path.exists():
        return {"path": display_path, "exists": False, "sha256": None}
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": display_path,
        "exists": True,
        "file_size_bytes": int(stat.st_size),
        "file_modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256": digest.hexdigest(),
    }


def _time_vector_record(
    path: Path,
    original_time: np.ndarray,
    fitted_time: np.ndarray,
    target_rows: int,
    trial_path: Path,
) -> dict[str, Any]:
    original = np.asarray(original_time, dtype=np.float64).reshape(-1)
    fitted = np.asarray(fitted_time, dtype=np.float64).reshape(-1)
    record = _array_source_record(path, original, trial_path)
    record.update({
        "original_count": int(original.size),
        "data_row_count": int(target_rows),
        "time_was_rebuilt_to_match_data_rows": bool(original.size != int(target_rows)),
        "fit_method": (
            "unchanged"
            if original.size == int(target_rows)
            else "numpy.linspace(original_first, original_last, data_row_count)"
        ),
        "fitted_count": int(fitted.size),
        "fitted_first_time_s": float(fitted[0]) if fitted.size else None,
        "fitted_last_time_s": float(fitted[-1]) if fitted.size else None,
        "fitted_time_strictly_increasing": bool(
            fitted.size < 2 or np.all(np.diff(fitted) > 0)
        ),
        "fitted_median_timestep_s": (
            float(np.median(np.diff(fitted))) if fitted.size > 1 else None
        ),
    })
    return record


def _linear_interpolation_map(source_time: np.ndarray, target_time: np.ndarray) -> dict[str, Any]:
    """
    Return the exact row interpolation used by linear resampling.

    Each target row is source[left] * (1-alpha) + source[right] * alpha.
    The map refers to source data-row indices after any time-vector length fit.
    """
    src = np.asarray(source_time, dtype=np.float64).reshape(-1)
    dst = np.asarray(target_time, dtype=np.float64).reshape(-1)
    if src.size == 0:
        raise ValueError("Cannot trace interpolation from an empty time vector")
    if src.size == 1:
        left = right = np.zeros(dst.size, dtype=np.int64)
        alpha = np.zeros(dst.size, dtype=np.float64)
    else:
        right = np.searchsorted(src, dst, side="left")
        right = np.clip(right, 1, src.size - 1)
        left = right - 1
        exact_left = np.isclose(dst, src[left], rtol=0.0, atol=1e-12)
        exact_right = np.isclose(dst, src[right], rtol=0.0, atol=1e-12)
        right[exact_left] = left[exact_left]
        left[exact_right] = right[exact_right]
        denom = src[right] - src[left]
        alpha = np.divide(
            dst - src[left], denom,
            out=np.zeros(dst.size, dtype=np.float64),
            where=np.abs(denom) > np.finfo(np.float64).eps,
        )
    return {
        "formula": "target[j] = source[left_index[j]]*(1-alpha[j]) + source[right_index[j]]*alpha[j]",
        "left_index": left.astype(int).tolist(),
        "right_index": right.astype(int).tolist(),
        "alpha": alpha.astype(float).tolist(),
    }


def _trace_stage(
    name: str,
    input_count: int,
    keep_start: int,
    keep_end: int,
    cumulative_start: int,
    *,
    enabled: bool,
    parameters: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe one contiguous trim using zero-based half-open bounds."""
    keep_start = int(keep_start)
    keep_end = int(keep_end)
    input_count = int(input_count)
    return {
        "name": name,
        "enabled": bool(enabled),
        "input_frame_count": input_count,
        "keep_bounds_in_input": [keep_start, keep_end],
        "keep_bounds_in_uniform_resampled_timeline": [
            int(cumulative_start + keep_start),
            int(cumulative_start + keep_end),
        ],
        "removed_leading_frames": keep_start,
        "removed_trailing_frames": input_count - keep_end,
        "output_frame_count": keep_end - keep_start,
        "parameters": parameters or {},
        "details": details or {},
    }


def _output_manifest(out_dir: Path, final_count: int) -> dict[str, Any]:
    """Inventory saved arrays and identify every axis matching the final timeline."""
    manifest: dict[str, Any] = {}
    for path in sorted(out_dir.glob("*.npy")):
        try:
            arr = np.load(path, allow_pickle=True)
            entry: dict[str, Any] = {
                "shape": [int(v) for v in arr.shape],
                "dtype": str(arr.dtype),
            }
            if arr.dtype == object and arr.shape == () and isinstance(arr.item(), dict):
                child_shapes = {}
                for key, value in arr.item().items():
                    value_arr = np.asarray(value)
                    child_shapes[str(key)] = [int(v) for v in value_arr.shape]
                entry["object_dictionary_shapes"] = child_shapes
            else:
                entry["axes_matching_final_frame_count"] = [
                    int(axis) for axis, size in enumerate(arr.shape) if int(size) == int(final_count)
                ]
            manifest[path.name] = entry
        except Exception as exc:
            manifest[path.name] = {"inspection_error": str(exc)}
    return manifest


def _refresh_trimming_trace_output_manifest(out_dir: Path) -> None:
    """Refresh output inventory after later post-processing creates more files."""
    trace_path = out_dir / TRIMMING_TRACE_FILENAME
    trace = _load_json_dict(trace_path)
    if not trace:
        return
    final_count = int(trace.get("final_mapping", {}).get("final_frame_count", 0))
    trace["output_files"] = _output_manifest(out_dir, final_count)
    trace["last_updated_at"] = datetime.now().isoformat()
    _write_json_atomic(trace_path, trace)


def _append_postprocess_trim_trace(
    out_dir: Path,
    *,
    stage_name: str,
    input_count: int,
    keep_start: int,
    keep_end: int,
    parameters: dict[str, Any],
    details: dict[str, Any],
) -> None:
    """Update provenance after a post-process rewrites arrays from Untrimmed."""
    trace_path = out_dir / TRIMMING_TRACE_FILENAME
    trace = _load_json_dict(trace_path)
    if not trace:
        return
    previous = trace.get("final_mapping", {})
    uniform_bounds = previous.get("uniform_resampled_frame_bounds", [0, input_count])
    base_start = int(uniform_bounds[0])
    stage = _trace_stage(
        stage_name, input_count, keep_start, keep_end, base_start,
        enabled=True, parameters=parameters, details=details,
    )
    trace.setdefault("timeline_stages", []).append(stage)
    trace.setdefault("postprocessing_history", []).append({
        "stage": stage_name,
        "applied_at": datetime.now().isoformat(),
        "input_frame_count": int(input_count),
        "keep_bounds_in_input": [int(keep_start), int(keep_end)],
        "parameters": parameters,
        "details": details,
    })
    final_start, final_end = stage["keep_bounds_in_uniform_resampled_timeline"]
    uniform = trace.get("uniform_resampling", {})
    grid_start = uniform.get("start_time_s")
    grid_dt = uniform.get("dt_s")
    final_first_time = (
        float(grid_start) + final_start * float(grid_dt)
        if grid_start is not None and grid_dt is not None and final_end > final_start
        else None
    )
    final_last_time = (
        float(grid_start) + (final_end - 1) * float(grid_dt)
        if grid_start is not None and grid_dt is not None and final_end > final_start
        else None
    )
    trace["final_mapping"] = {
        "uniform_resampled_frame_bounds": [final_start, final_end],
        "final_frame_count": int(keep_end - keep_start),
        "final_first_time_s": final_first_time,
        "final_last_time_s": final_last_time,
        "mapping_formula": (
            f"final_frame[j] corresponds to uniform_resampled_frame[{final_start} + j]"
        ),
        "postprocess_trim_applied": True,
    }
    trace["last_updated_at"] = datetime.now().isoformat()
    _write_json_atomic(trace_path, trace)
    _refresh_trimming_trace_output_manifest(out_dir)


def _metadata_timestamp(json_path: Path) -> float | None:
    """Return the best available timestamp for a metadata JSON."""
    if not json_path.exists():
        return None

    payload = _load_json_dict(json_path)
    raw = payload.get("processing_date")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw).timestamp()
        except Exception:
            pass

    try:
        return json_path.stat().st_mtime
    except Exception:
        return None


def _update_trial_info_json(out_dir: Path, updates: dict[str, Any]) -> None:
    """
    Merge extra pipeline metadata into Trial_Processing_Information.json.
    """
    info_path = out_dir / "Trial_Processing_Information.json"
    payload = _load_json_dict(info_path)
    payload.update(updates)
    payload["last_pipeline_update_date"] = datetime.now().isoformat()
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _load_npy_numeric(path: Path) -> np.ndarray:
    """
    Load .npy robustly. If legacy/object arrays are encountered, allow
    pickle and convert to numeric ndarray.
    """
    try:
        arr = np.load(path)
    except ValueError as e:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(e):
            raise
        arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr.astype(np.float64, copy=False)
    return arr


def _fit_time_to_length(t: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ensure a 1-D time vector matches target_len. If not, rebuild an evenly
    spaced vector over the same [start, end].
    """
    t = np.asarray(t).reshape(-1)
    if t.size == target_len:
        return t
    if t.size >= 2:
        return np.linspace(float(t[0]), float(t[-1]), target_len)
    return np.arange(target_len, dtype=float)


def _kinematics_input_filename(base_name: str, cfg: dict | None = None) -> str:
    """
    Select the kinematics input filename for Pos/Vel/Accel based on config.
    """
    if not cfg or not bool(cfg.get("UseNoised", False)):
        return base_name

    mapping = {
        "Pos.npy": "Pos_noised.npy",
        "Vel.npy": "Vel_noised.npy",
        "Accel.npy": "Accel_noised.npy",
    }
    return mapping.get(base_name, base_name)


def _resolve_trial_kinematics_dir(trial_path: Path, cfg: dict | None = None) -> Path:
    """Return the source kinematics directory for the active processing mode."""
    # --OpenCapVal: read kinematics AND forces from Trial/<MoCap|Video>/Motion.
    if cfg and cfg.get("OPENCAPVAL_SOURCE"):
        return trial_path / str(cfg["OPENCAPVAL_SOURCE"]) / "Motion"
    if cfg and bool(cfg.get("OC_Mocap", False)):
        mocap_dir = trial_path / "MoCap"
        preferred_dir = mocap_dir / "Untrimmed"
        required_files = [
            _kinematics_input_filename("Pos.npy", cfg),
            _kinematics_input_filename("Vel.npy", cfg),
            _kinematics_input_filename("Accel.npy", cfg),
            "Time.npy",
        ]
        if preferred_dir.is_dir() and all((preferred_dir / name).exists() for name in required_files):
            return preferred_dir
        return mocap_dir
    motion_dir = trial_path / "Motion" / "Motion_Pelvis_Adjusted"
    if motion_dir.exists():
        return motion_dir
    return trial_path / "Motion"


def _infer_raw_trial_treadmill_flag(
    trial_path: Path,
    cfg: dict | None = None,
) -> tuple[bool | None, str]:
    """
    Infer treadmill-vs-overground from raw Pos.npy before full processing.

    This uses the same pelvis-net-speed threshold as the main pipeline, but
    estimates it directly from the source kinematics after pelvis-yaw alignment.
    """
    fs = float((cfg or {}).get("SAMPLING_RATE_HZ", 100.0))
    kin_dir = _resolve_trial_kinematics_dir(trial_path, cfg)
    pos_path = kin_dir / "Pos.npy"
    if not pos_path.exists():
        return None, f"missing {pos_path}"

    try:
        pos = np.asarray(_load_npy_numeric(pos_path), dtype=np.float64)
    except Exception as e:
        return None, f"failed loading {pos_path} ({e})"

    if pos.ndim != 2:
        return None, f"invalid Pos.npy ndim={pos.ndim}"
    if pos.shape[0] < 2:
        return None, f"too few frames in Pos.npy ({pos.shape[0]})"
    if pos.shape[1] < 6:
        return None, f"invalid Pos.npy shape {pos.shape}"

    time_candidates = [
        kin_dir / "Time_for_pos.npy",
        trial_path / "Motion" / "Time_for_pos.npy",
        kin_dir / "Time.npy",
    ]
    kin_time = None
    for time_path in time_candidates:
        if not time_path.exists():
            continue
        try:
            kin_time = _fit_time_to_length(_load_npy_numeric(time_path), pos.shape[0])
            break
        except Exception:
            continue

    if kin_time is None:
        kin_time = np.arange(pos.shape[0], dtype=np.float64) / max(fs, 1e-6)
    else:
        kin_time = np.asarray(kin_time, dtype=np.float64).reshape(-1)

    pos, _, _ = normalize_kinematic_angle_units(
        pos, context=f"{trial_path.parent.name}/{trial_path.name}/discover"
    )
    aligned_pos, *_ = align_myosuite_pelvis(pos)
    duration_s = float(kin_time[-1] - kin_time[0]) if kin_time.size >= 2 else 0.0
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        duration_s = float(pos.shape[0] - 1) / max(fs, 1e-6)

    pelvis_net_speed = (
        abs(float(aligned_pos[-1, 3]) - float(aligned_pos[0, 3])) / max(duration_s, 1e-6)
    )
    is_treadmill = (duration_s >= 1.0) and (pelvis_net_speed < 0.3)
    return bool(is_treadmill), ""


def _source_dir_ready_for_calc_frame(src_dir: Path) -> tuple[bool, str]:
    required = [
        "pos_mjx.npy",
        "COP_Cleaned_Relative.npy",
        "ankle_heights.npy",
        "contactBoolean.npy",
        "GRF_Cleaned.npy",
        "Mass_kg.npy",
        "Height_m.npy",
    ]
    missing = [name for name in required if not (src_dir / name).exists()]
    if missing:
        return False, f"missing {', '.join(missing)}"
    return True, ""


def generate_calc_frame_outputs_for_source(
    src_dir: Path,
    xml_path: Path,
    trial_id: str,
) -> tuple[bool, str]:
    """
    Write ExtractGtoCalcRotation/extractFPA style outputs for one processed source directory.
    """
    ready, reason = _source_dir_ready_for_calc_frame(src_dir)
    if not ready:
        return False, f"{trial_id} [{src_dir.name}] | {reason}"

    try:
        qpos_matrix = np.asarray(np.load(src_dir / "pos_mjx.npy"), dtype=np.float64)
        cop_rel = np.asarray(np.load(src_dir / "COP_Cleaned_Relative.npy"), dtype=np.float64)
        ankle_h = np.asarray(np.load(src_dir / "ankle_heights.npy"), dtype=np.float64)
        contact_bool = np.asarray(np.load(src_dir / "contactBoolean.npy"), dtype=np.float64)
        grf_mj = np.asarray(np.load(src_dir / "GRF_Cleaned.npy"), dtype=np.float64)
        mass_kg = np.asarray(np.load(src_dir / "Mass_kg.npy"), dtype=np.float64)
        height_m = np.asarray(np.load(src_dir / "Height_m.npy"), dtype=np.float64)
    except Exception as e:
        return False, f"{trial_id} [{src_dir.name}] | failed loading calc-frame inputs ({e})"

    if cop_rel.ndim != 2 or cop_rel.shape[1] < 4:
        return False, f"{trial_id} [{src_dir.name}] | COP_Cleaned_Relative has invalid shape {cop_rel.shape}"
    if ankle_h.ndim != 2 or ankle_h.shape[1] < 2:
        return False, f"{trial_id} [{src_dir.name}] | ankle_heights has invalid shape {ankle_h.shape}"
    if contact_bool.ndim == 1:
        contact_bool = contact_bool[:, np.newaxis]
    if contact_bool.shape[1] == 1:
        contact_bool = np.repeat(contact_bool, 2, axis=1)
    if contact_bool.ndim != 2 or contact_bool.shape[1] < 2:
        return False, f"{trial_id} [{src_dir.name}] | contactBoolean has invalid shape {contact_bool.shape}"

    T = int(qpos_matrix.shape[0])
    if grf_mj.ndim != 2 or grf_mj.shape[1] < 6:
        return False, f"{trial_id} [{src_dir.name}] | GRF_Cleaned has invalid shape {grf_mj.shape}"
    if mass_kg.ndim == 0:
        mass_kg = np.full(T, float(mass_kg), dtype=np.float64)
    mass_kg = mass_kg.reshape(-1)
    if height_m.ndim == 0:
        height_m = np.full(T, float(height_m), dtype=np.float64)
    height_m = height_m.reshape(-1)
    if not (T == cop_rel.shape[0] == ankle_h.shape[0] == contact_bool.shape[0] == grf_mj.shape[0] == mass_kg.shape[0] == height_m.shape[0]):
        return False, (
            f"{trial_id} [{src_dir.name}] | length mismatch "
            f"(qpos={T}, cop={cop_rel.shape[0]}, ankle={ankle_h.shape[0]}, "
            f"contact={contact_bool.shape[0]}, grf={grf_mj.shape[0]}, "
            f"mass={mass_kg.shape[0]}, height={height_m.shape[0]})"
        )

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        rot_g_to_r, rot_g_to_l, calcn_pos_r, calcn_pos_l, toes_pos_r, toes_pos_l = (
            _extract_ground_to_calc_rotations_from_qpos(model, qpos_matrix)
        )

        fpa_deg = _compute_foot_progression_angle_deg(
            calcn_pos_r=calcn_pos_r,
            calcn_pos_l=calcn_pos_l,
            toes_pos_r=toes_pos_r,
            toes_pos_l=toes_pos_l,
        )
        tos_position = np.stack([toes_pos_l, toes_pos_r], axis=0)

        cop_r_ground = np.column_stack([cop_rel[:, 0], cop_rel[:, 1], -ankle_h[:, 0]])
        cop_l_ground = np.column_stack([cop_rel[:, 2], cop_rel[:, 3], -ankle_h[:, 1]])

        cop_r_calc = _apply_rotation_batch(rot_g_to_r, cop_r_ground)
        cop_l_calc = _apply_rotation_batch(rot_g_to_l, cop_l_ground)
        cop_calc_frame = np.column_stack([
            cop_r_calc[:, 0], cop_r_calc[:, 1], cop_r_calc[:, 2],
            cop_l_calc[:, 0], cop_l_calc[:, 1], cop_l_calc[:, 2],
        ])

        rot_w_to_ga_r, angle_r_deg = _compose_world_to_ground_aligned(rot_g_to_r)
        rot_w_to_ga_l, angle_l_deg = _compose_world_to_ground_aligned(rot_g_to_l)
        rot_w_to_ga = np.stack([rot_w_to_ga_r, rot_w_to_ga_l], axis=1)

        cop_r_ga = _apply_rotation_batch(rot_w_to_ga_r, cop_r_ground)
        cop_l_ga = _apply_rotation_batch(rot_w_to_ga_l, cop_l_ground)
        cop_calc_frame_ground_aligned = np.column_stack([
            cop_r_ga[:, 0], cop_r_ga[:, 1], cop_r_ga[:, 2],
            cop_l_ga[:, 0], cop_l_ga[:, 1], cop_l_ga[:, 2],
        ])
        cop_calc_frame_ground_aligned_grf_norm = _multiply_cop_by_bodyweight_normalized_grf_magnitude(
            cop_calc_frame_ground_aligned,
            grf_mj,
            mass_kg,
            height_m,
        )

        cop_r_world_recon = _apply_rotation_batch(np.transpose(rot_w_to_ga_r, (0, 2, 1)), cop_r_ga)
        cop_l_world_recon = _apply_rotation_batch(np.transpose(rot_w_to_ga_l, (0, 2, 1)), cop_l_ga)
        cop_world_recon = np.column_stack([
            cop_r_world_recon[:, 0], cop_r_world_recon[:, 1], cop_r_world_recon[:, 2],
            cop_l_world_recon[:, 0], cop_l_world_recon[:, 1], cop_l_world_recon[:, 2],
        ])
        cop_rel_recon_xy = np.column_stack([
            cop_r_world_recon[:, 0], cop_r_world_recon[:, 1],
            cop_l_world_recon[:, 0], cop_l_world_recon[:, 1],
        ])
        recon_err = float(np.max(np.abs(cop_rel_recon_xy - cop_rel[:, :4])))

        cop_calc_frame_ground_aligned_yplus = _add_ankle_height_to_ground_aligned_y(
            cop_ground_aligned=cop_calc_frame_ground_aligned,
            ankle_h=ankle_h,
        )

        np.save(src_dir / "COP_CalcFrame.npy", cop_calc_frame)
        np.save(src_dir / "WorldToGroundAlignedCalcnRotation.npy", rot_w_to_ga)
        np.save(src_dir / "CalcnToFloor_AngleDeg.npy", np.column_stack([angle_r_deg, angle_l_deg]))
        np.save(src_dir / "COP_CalcFrame_GroundAligned.npy", cop_calc_frame_ground_aligned)
        np.save(src_dir / "COP_CalcFrame_GroundAligned_GRFNorm.npy", cop_calc_frame_ground_aligned_grf_norm)
        np.save(src_dir / "COP_CalcFrame_GroundAligned_BackToWorld.npy", cop_world_recon)
        np.save(src_dir / "COP_Cleaned_Relative_RecoveredFromGroundAligned.npy", cop_rel_recon_xy)
        np.save(src_dir / "COP_CalcFrame_GroundAligned_YplusAnkleHeight.npy", cop_calc_frame_ground_aligned_yplus)
        np.save(src_dir / "FootProgressionAngle.npy", fpa_deg)
        np.save(src_dir / "Foot_ProgressionAngle.npy", fpa_deg)
        np.save(src_dir / "tosPosition.npy", tos_position)

        _update_trial_info_json(
            src_dir,
            {
                "calc_frame_postprocess_ran": True,
                "calc_frame_reconstruction_max_abs_err": recon_err,
                "foot_progression_angle_source": "mujoco_forward_kinematics",
            },
        )
        _refresh_trimming_trace_output_manifest(src_dir)
        if recon_err > 1e-4:
            print(f"[WARN] {trial_id} [{src_dir.name}] | calc-frame reconstruction error = {recon_err:.6e}")
    except Exception as e:
        return False, f"{trial_id} [{src_dir.name}] | calc-frame postprocess failed ({e})"

    return True, f"{trial_id} [{src_dir.name}]"


def get_output_dir_name(cfg: dict | None = None) -> str:
    """Return the per-trial output directory name for the active processing mode."""
    # --OpenCapVal per-pass output: Trial/<MoCap|Video>/ProcessedData.
    if cfg is not None and cfg.get("OPENCAPVAL_SOURCE"):
        return f"{cfg['OPENCAPVAL_SOURCE']}/ProcessedData"
    if cfg is not None and bool(cfg.get("OC_Mocap", False)):
        return "MoCap"
    return "ProcessedData"


def get_output_dir_names(cfg: dict | None = None) -> list[str]:
    """Return all primary per-trial output directory names for the active mode."""
    # --OpenCapVal writes both MoCap and Video ProcessedData bundles per trial.
    if cfg is not None and bool(cfg.get("OpenCapVal", False)):
        return ["MoCap/ProcessedData", "Video/ProcessedData"]
    if cfg is not None and bool(cfg.get("OC_Mocap", False)):
        return ["ProcessedData", "MoCap"]
    return ["ProcessedData"]


def get_output_dir_label(cfg: dict | None = None) -> str:
    """Human-readable label for active primary output directories."""
    return " + ".join(get_output_dir_names(cfg))


def compute_and_apply_column_masks(data_root:      Path,
                                    subject_filter: str | None = None,
                                    subject_list:   list[str] | None = None,
                                    output_dir_name: str = "ProcessedData"):
    """
    Validate acc_inputs.npy against the fixed 21-column knee-inclusive schema.

    The current schema retains knee_angle_r/l and removes only the two MTP
    coordinates. Legacy 19-column files omitted both knees; they cannot be
    repaired safely here without their matching qacc_mjx array and are reported
    for the explicit dataset migration instead.
    """
    # ── Discover all processed trials ────────────────────────────
    allowed: set[str] | None = None
    if subject_list:
        allowed = set(subject_list)
    elif subject_filter:
        allowed = {subject_filter}

    proc_dirs = []
    for subj_dir in sorted(data_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        if allowed is not None and subj_dir.name not in allowed:
            continue
        for trial_dir in sorted(subj_dir.iterdir()):
            pd = trial_dir / output_dir_name
            if pd.is_dir():
                proc_dirs.append(pd)

    if not proc_dirs:
        print(f"  [ColumnMask] No {output_dir_name} directories found — skipping.")
        return

    repaired = 0
    already_ok = 0
    skipped: list[str] = []

    for pd in proc_dirs:
        fpath = pd / "acc_inputs.npy"
        if not fpath.exists():
            continue

        arr = np.load(fpath)
        if arr.ndim != 2:
            skipped.append(f"{pd}: acc_inputs.npy has shape {arr.shape}")
            continue

        if arr.shape[1] == 21:
            already_ok += 1
        elif arr.shape[1] == 19:
            skipped.append(f"{pd}: acc_inputs.npy uses legacy 19-column knee-free schema")
            continue
        else:
            skipped.append(f"{pd}: acc_inputs.npy has unexpected shape {arr.shape}")
            continue

        noised_path = pd / _with_file_suffix("acc_inputs.npy")
        if noised_path.exists():
            noised_arr = np.load(noised_path)
            if noised_arr.ndim != 2:
                skipped.append(f"{pd}: {noised_path.name} has shape {noised_arr.shape}")
            elif noised_arr.shape[1] == 19:
                skipped.append(f"{pd}: {noised_path.name} uses legacy 19-column knee-free schema")
            elif noised_arr.shape[1] != 21:
                skipped.append(f"{pd}: {noised_path.name} has unexpected shape {noised_arr.shape}")

    print(f"  [ColumnMask] acc_inputs.npy: repaired {repaired}, already 21-col {already_ok}, "
          f"skipped {len(skipped)}")
    for msg in skipped[:25]:
        print(f"    [ColumnMask] ↷ {msg}")
    if len(skipped) > 25:
        print(f"    [ColumnMask] ... {len(skipped) - 25} more skipped files")


@dataclass
class MissingStepStanceInfo:
    foot: str
    index: int
    start: int
    end: int
    duration_frames: int
    first_half_mean_bw: float
    second_half_mean_bw: float
    is_edge: bool
    edge_side: str
    ratio_threshold: float
    ratio_value: float
    duration_threshold_frames: float
    non_edge_avg_duration_frames: float


@dataclass
class MissingStepTrialResult:
    subject: str
    trial: str
    processed_dir: Path
    duration_s: float
    fs_hz: float
    vgrf_r_bw: np.ndarray
    vgrf_l_bw: np.ndarray
    non_edge_ds_mean_frames: float
    ds_trim_frames: int
    flagged_stances: list[MissingStepStanceInfo]


def _sync_backup_entries(proc_dir: Path, backup_dir: Path) -> None:
    """
    Keep missing/new derived files available in the backup without overwriting
    its authoritative pre-trim core arrays.
    """
    backup_len = _infer_trial_length(backup_dir)

    for item in proc_dir.iterdir():
        if item.name in PIPELINE_SKIP_DIR_NAMES:
            continue

        dst = backup_dir / item.name
        should_copy = not dst.exists()
        if (
            not should_copy
            and item.name in PIPELINE_DERIVED_FILENAMES
            and item.is_file()
            and dst.is_file()
        ):
            if backup_len is not None and item.suffix.lower() == ".npy":
                try:
                    src_arr = np.load(item, mmap_mode="r", allow_pickle=True)
                    src_len = int(src_arr.shape[0]) if isinstance(src_arr, np.ndarray) and src_arr.ndim >= 1 else None
                except Exception:
                    src_len = None
                if src_len is not None and src_len != backup_len:
                    continue
            try:
                should_copy = item.stat().st_mtime > dst.stat().st_mtime + 1e-6
            except Exception:
                should_copy = False

        if not should_copy:
            continue

        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if item.is_dir():
            shutil.copytree(item, dst)
        elif item.is_file():
            shutil.copy2(item, dst)


def _ensure_untrimmed_backup(proc_dir: Path) -> Path:
    """
    Ensure <source>/Untrimmed exists and contains the current pre-miss-step files.

    Existing backups are preserved unless the source metadata is newer, which
    indicates the trial was reprocessed since the backup was created.
    """
    untrimmed_dir = proc_dir / "Untrimmed"
    refresh = False

    if not untrimmed_dir.exists() or not any(untrimmed_dir.iterdir()):
        refresh = True
    else:
        src_ts = _metadata_timestamp(proc_dir / "Trial_Processing_Information.json")
        backup_ts = _metadata_timestamp(untrimmed_dir / "Trial_Processing_Information.json")
        if src_ts is not None and backup_ts is not None and src_ts > backup_ts + 1e-6:
            refresh = True

    if refresh:
        if untrimmed_dir.exists():
            shutil.rmtree(untrimmed_dir)
        untrimmed_dir.mkdir(parents=True, exist_ok=True)
        for item in proc_dir.iterdir():
            if item.name in PIPELINE_SKIP_DIR_NAMES:
                continue
            dst = untrimmed_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            elif item.is_file():
                shutil.copy2(item, dst)
    else:
        _sync_backup_entries(proc_dir, untrimmed_dir)

    return untrimmed_dir


def _preferred_source_dir(proc_dir: Path) -> Path:
    untrimmed_dir = proc_dir / "Untrimmed"
    if untrimmed_dir.is_dir() and any(untrimmed_dir.iterdir()):
        return untrimmed_dir
    return proc_dir


def _has_required_analysis_files(source_dir: Path | None) -> bool:
    if source_dir is None or not source_dir.is_dir():
        return False
    return all((source_dir / name).exists() for name in MISSSTEP_ANALYSIS_REQUIRED_FILES)


def _select_analysis_source(candidates: list[Path | None]) -> Path | None:
    existing = [p for p in candidates if p is not None and p.is_dir()]
    populated = [p for p in existing if any(p.iterdir())]
    ordered = populated + [p for p in existing if p not in populated]
    for p in ordered:
        if _has_required_analysis_files(p):
            return p
    return ordered[0] if ordered else None


def _infer_trial_length(source_dir: Path) -> int | None:
    candidates = [
        "GRF_Cleaned.npy",
        "contactBoolean.npy",
        "Pos.npy",
        "Vel.npy",
        "COP_Cleaned.npy",
        "COP_CalcFrame_GroundAligned.npy",
    ]
    for fname in candidates:
        path = source_dir / fname
        if not path.exists():
            continue
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                return int(arr.shape[0])
        except Exception:
            try:
                arr = np.asarray(np.load(path, allow_pickle=True))
                if arr.ndim >= 1:
                    return int(arr.shape[0])
            except Exception:
                continue
    return None


def _translate_bounds_by_edge_trim(
    source_len: int,
    target_len: int,
    source_start: int,
    source_end: int,
) -> tuple[int, int]:
    left_trim = int(np.clip(source_start, 0, max(source_len, 0)))
    right_trim = int(np.clip(source_len - source_end, 0, max(source_len, 0)))
    target_start = int(np.clip(left_trim, 0, target_len))
    target_end = int(np.clip(target_len - right_trim, target_start, target_len))
    return target_start, target_end


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1", "yes", "y"}:
            return True
        if norm in {"false", "0", "no", "n"}:
            return False
    return None


def _read_treadmill_flag(proc_dir: Path, source_dir: Path) -> tuple[bool | None, str]:
    proc_meta = proc_dir / "Trial_Processing_Information.json"
    source_meta = source_dir / "Trial_Processing_Information.json"
    candidates = [source_meta]
    if proc_meta != source_meta:
        candidates.append(proc_meta)
    sibling_proc_meta = proc_dir.parent / "ProcessedData" / "Trial_Processing_Information.json"
    if sibling_proc_meta not in candidates:
        candidates.append(sibling_proc_meta)

    errors: list[str] = []
    for path in candidates:
        if not path.exists():
            errors.append(f"{path} not found")
            continue

        payload = _load_json_dict(path)
        if not payload:
            errors.append(f"missing/invalid json in {path}")
            continue

        for key in ("treadmill_flag", "treadmill", "Treadmill flag", "TreadmillFlag"):
            if key not in payload:
                continue
            parsed = _coerce_bool(payload.get(key))
            if parsed is None:
                errors.append(f"invalid boolean for key '{key}' in {path}")
                break
            return parsed, ""
        else:
            errors.append(f"missing treadmill key in {path}")

    return None, "; ".join(errors)


def _load_motion_aligned_trim_reference(processed_dir: Path) -> dict[str, Any] | None:
    """
    Load the authoritative motion-aligned core trim bounds saved by the
    ProcessedData pipeline so the MoCap branch can reuse the exact same window.
    """
    meta_path = processed_dir / "Trial_Processing_Information.json"
    if not meta_path.exists():
        return None

    payload = _load_json_dict(meta_path)
    if not payload:
        return None

    bounds = payload.get("core_trim_bounds_motion_aligned")
    pretrim_n_frames = payload.get("core_trim_pretrim_n_frames")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None
    if not isinstance(pretrim_n_frames, (int, np.integer)):
        return None

    start_idx = int(bounds[0])
    end_idx = int(bounds[1])
    pretrim_n_frames = int(pretrim_n_frames)
    if pretrim_n_frames < 0:
        return None
    if not (0 <= start_idx <= end_idx <= pretrim_n_frames):
        return None

    out: dict[str, Any] = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "pretrim_n_frames": pretrim_n_frames,
    }

    for key in (
        "n_frames_after_grf_trim",
        "n_frames_after_weak_edge_trim",
        "core_trim_reference_space",
        "grf_trim_bounds_motion_aligned",
        "weak_edge_trim_bounds_after_grf",
        "outlier_trim_bounds_after_weak_edge",
    ):
        if key in payload:
            out[key] = payload[key]

    return out


def _trim_array_time_axis(arr: np.ndarray,
                          start_idx: int,
                          end_idx: int,
                          original_len: int) -> tuple[np.ndarray, bool]:
    """
    Trim an array along its time axis.

    Most pipeline arrays are shaped (T, ...), but a few generated files use a
    different convention, most notably tosPosition.npy as (2, T, 3).  Trim axis
    0 when it matches, otherwise trim the only nonzero axis whose length matches
    original_len.  If the match is ambiguous, leave the array unchanged rather
    than guessing.
    """
    if arr.ndim < 1:
        return arr, False
    if arr.shape[0] == original_len:
        return arr[start_idx:end_idx], True

    matching_axes = [axis for axis, size in enumerate(arr.shape) if size == original_len]
    if len(matching_axes) != 1:
        return arr, False

    axis = matching_axes[0]
    indexer = [slice(None)] * arr.ndim
    indexer[axis] = slice(start_idx, end_idx)
    return arr[tuple(indexer)], True


def _trim_object(obj: Any, start_idx: int, end_idx: int, original_len: int) -> tuple[Any, bool]:
    if isinstance(obj, np.ndarray):
        return _trim_array_time_axis(obj, start_idx, end_idx, original_len)
    if isinstance(obj, dict):
        changed = False
        out = {}
        for key, value in obj.items():
            trimmed_value, child_changed = _trim_object(value, start_idx, end_idx, original_len)
            out[key] = trimmed_value
            changed = changed or child_changed
        return out, changed
    if isinstance(obj, list):
        changed = False
        out = []
        for value in obj:
            trimmed_value, child_changed = _trim_object(value, start_idx, end_idx, original_len)
            out.append(trimmed_value)
            changed = changed or child_changed
        return out, changed
    if isinstance(obj, tuple):
        changed = False
        out = []
        for value in obj:
            trimmed_value, child_changed = _trim_object(value, start_idx, end_idx, original_len)
            out.append(trimmed_value)
            changed = changed or child_changed
        return tuple(out), changed
    return obj, False


def _rewrite_trimmed_from_backup(
    proc_dir: Path,
    untrimmed_dir: Path,
    start_idx: int,
    end_idx: int,
    original_len: int,
) -> tuple[int, int]:
    trimmed_files = 0
    total_files = 0

    for existing in proc_dir.iterdir():
        if existing.name in PIPELINE_SKIP_DIR_NAMES:
            continue
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()

    for item in untrimmed_dir.iterdir():
        dst = proc_dir / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
            continue
        if not item.is_file():
            continue

        total_files += 1
        suffix = item.suffix.lower()
        if suffix == ".npy":
            arr = np.load(item, allow_pickle=True)
            changed = False
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
                payload = arr.item()
                payload_trim, changed = _trim_object(payload, start_idx, end_idx, original_len)
                if changed:
                    np.save(dst, payload_trim, allow_pickle=True)
            else:
                arr_trim, changed = _trim_object(arr, start_idx, end_idx, original_len)
                if changed:
                    np.save(dst, arr_trim, allow_pickle=True)
            if changed:
                trimmed_files += 1
            else:
                shutil.copy2(item, dst)
        elif suffix == ".npz":
            changed_any = False
            with np.load(item, allow_pickle=True) as npz:
                out = {}
                for key in npz.files:
                    trimmed_value, child_changed = _trim_object(npz[key], start_idx, end_idx, original_len)
                    out[key] = trimmed_value
                    changed_any = changed_any or child_changed
            if changed_any:
                np.savez(dst, **out)
                trimmed_files += 1
            else:
                shutil.copy2(item, dst)
        else:
            shutil.copy2(item, dst)

    return trimmed_files, total_files


def _load_scalar_mass_kg(mass_arr: np.ndarray) -> float | None:
    mass_arr = np.asarray(mass_arr, dtype=float).reshape(-1)
    if mass_arr.size == 0:
        return None
    finite = mass_arr[np.isfinite(mass_arr)]
    if finite.size == 0:
        return None
    mass = float(np.nanmean(finite))
    if mass <= 0.0:
        return None
    return mass


def _mean_non_edge_double_support_frames(contact: np.ndarray) -> float:
    c = np.asarray(contact)
    if c.ndim != 2 or c.shape[1] < 2 or c.shape[0] == 0:
        return 0.0
    both_stance = (c[:, 0] > 0.5) & (c[:, 1] > 0.5)
    ds_segments = _stance_segments(both_stance)
    t_len = c.shape[0]
    non_edge_ds = [(s, e) for (s, e) in ds_segments if s > 0 and e < t_len]
    if not non_edge_ds:
        return 0.0
    return float(np.mean([e - s for s, e in non_edge_ds]))


def _estimate_duration_seconds(proc_dir: Path, n_frames: int, fs_hz: float) -> float:
    time_candidates = [
        proc_dir / "Time.npy",
        proc_dir.parent / "Motion" / "Time.npy",
        proc_dir.parent / "Motion" / "Time_for_pos.npy",
    ]
    for path in time_candidates:
        if not path.exists():
            continue
        try:
            t = np.asarray(np.load(path), dtype=float).reshape(-1)
            if t.size >= 2:
                span = float(t[-1] - t[0])
                if np.isfinite(span) and span > 0:
                    return span
        except Exception:
            pass
    return float(n_frames) / float(fs_hz)


def _analyze_missing_step_foot(
    vgrf_bw: np.ndarray,
    contact_foot: np.ndarray,
    foot_name: str,
    half_ratio_threshold: float,
    duration_threshold: float,
    non_edge_avg_duration: float,
) -> list[MissingStepStanceInfo]:
    stances = _stance_segments(contact_foot > 0.5)
    if not stances:
        return []

    t_len = len(vgrf_bw)
    flagged: list[MissingStepStanceInfo] = []
    for i, (s, e) in enumerate(stances):
        dur = int(e - s)
        if dur <= 1:
            continue

        is_begin = (s == 0)
        is_end = (e == t_len)
        is_edge = is_begin or is_end
        if not is_edge:
            continue

        seg = vgrf_bw[s:e]
        split = dur // 2
        if split <= 0 or split >= dur:
            continue

        first_half_mean = float(np.mean(seg[:split]))
        second_half_mean = float(np.mean(seg[split:]))
        ratio_value = second_half_mean / (first_half_mean + 1e-8)

        cond_begin = first_half_mean < half_ratio_threshold * second_half_mean
        cond_end = second_half_mean < half_ratio_threshold * first_half_mean

        if is_begin and is_end:
            ratio_condition = cond_begin or cond_end
            edge_side = "both"
        elif is_begin:
            ratio_condition = cond_begin
            edge_side = "begin"
        else:
            ratio_condition = cond_end
            edge_side = "end"

        if (dur > duration_threshold) and ratio_condition:
            flagged.append(
                MissingStepStanceInfo(
                    foot=foot_name,
                    index=i,
                    start=s,
                    end=e,
                    duration_frames=dur,
                    first_half_mean_bw=first_half_mean,
                    second_half_mean_bw=second_half_mean,
                    is_edge=is_edge,
                    edge_side=edge_side,
                    ratio_threshold=half_ratio_threshold,
                    ratio_value=ratio_value,
                    duration_threshold_frames=duration_threshold,
                    non_edge_avg_duration_frames=non_edge_avg_duration,
                )
            )

    return flagged


def _analyze_trial_for_missing_steps(
    proc_dir: Path,
    fs_hz: float,
    max_duration_s: float,
    half_ratio_threshold: float,
    source_dir: Path | None = None,
) -> tuple[MissingStepTrialResult | None, str | None]:
    load_dir = source_dir if source_dir is not None else proc_dir
    grf_path = load_dir / "GRF_Cleaned.npy"
    mass_path = load_dir / "Mass_kg.npy"
    contact_path = load_dir / "contactBoolean.npy"

    if not (grf_path.exists() and mass_path.exists() and contact_path.exists()):
        return None, "missing required files"

    try:
        grf = np.asarray(np.load(grf_path), dtype=float)
        mass = _load_scalar_mass_kg(np.load(mass_path))
        contact = np.asarray(np.load(contact_path), dtype=float)
    except Exception as exc:
        return None, f"failed loading arrays: {exc}"

    if mass is None:
        return None, "invalid mass"
    if grf.ndim != 2 or grf.shape[1] < 6:
        return None, f"invalid GRF shape {grf.shape}"
    if contact.ndim == 1:
        contact = contact[:, np.newaxis]
    if contact.ndim != 2 or contact.shape[1] < 2:
        return None, f"invalid contactBoolean shape {contact.shape}"

    t_len = min(grf.shape[0], contact.shape[0])
    if t_len < 2:
        return None, "too short"

    grf = grf[:t_len]
    contact = contact[:t_len, :2]
    duration_s = _estimate_duration_seconds(load_dir, t_len, fs_hz)
    if np.isfinite(max_duration_s) and duration_s > max_duration_s:
        return None, "trial longer than max duration"

    bw_n = mass * 9.8078
    if bw_n <= 0:
        return None, "invalid BW normalization factor"

    vgrf_r = grf[:, 2] / bw_n
    vgrf_l = grf[:, 5] / bw_n

    stances_r = _stance_segments(contact[:, 0] > 0.5)
    stances_l = _stance_segments(contact[:, 1] > 0.5)
    non_edge_all = [(s, e) for (s, e) in (stances_r + stances_l) if s > 0 and e < t_len]
    if non_edge_all:
        non_edge_avg_duration = float(np.mean([e - s for s, e in non_edge_all]))
        duration_threshold = (1.0 / 3.0) * non_edge_avg_duration
    else:
        non_edge_avg_duration = 0.0
        duration_threshold = float("inf")

    flagged_r = _analyze_missing_step_foot(
        vgrf_r,
        contact[:, 0],
        "R",
        half_ratio_threshold,
        duration_threshold,
        non_edge_avg_duration,
    )
    flagged_l = _analyze_missing_step_foot(
        vgrf_l,
        contact[:, 1],
        "L",
        half_ratio_threshold,
        duration_threshold,
        non_edge_avg_duration,
    )
    flagged = flagged_r + flagged_l
    non_edge_ds_mean_frames = _mean_non_edge_double_support_frames(contact[:, :2])
    ds_trim_frames = int(
        np.clip(
            int(round(non_edge_ds_mean_frames)),
            0,
            int(CONFIG.get("MISSSTEP_DOUBLE_SUPPORT_EDGE_TRIM_FRAMES", 20)),
        )
    )

    return (
        MissingStepTrialResult(
            subject=proc_dir.parent.parent.name,
            trial=proc_dir.parent.name,
            processed_dir=proc_dir,
            duration_s=duration_s,
            fs_hz=fs_hz,
            vgrf_r_bw=vgrf_r,
            vgrf_l_bw=vgrf_l,
            non_edge_ds_mean_frames=non_edge_ds_mean_frames,
            ds_trim_frames=ds_trim_frames,
            flagged_stances=flagged,
        ),
        None,
    )


def _compute_missing_step_trim_bounds(
    trial: MissingStepTrialResult,
    peak_offset_frames: int,
) -> tuple[int, int]:
    t_len = len(trial.vgrf_r_bw)
    begin_markers: list[int] = []
    end_markers: list[int] = []

    for stance in trial.flagged_stances:
        vgrf = trial.vgrf_r_bw if stance.foot == "R" else trial.vgrf_l_bw
        seg = vgrf[stance.start:stance.end]
        if seg.size == 0:
            continue
        peak_global = stance.start + int(np.argmax(seg))
        offset = int(max(0, peak_offset_frames))
        if stance.edge_side in ("begin", "both"):
            begin_markers.append(int(np.clip(peak_global + offset, 0, t_len - 1)))
        if stance.edge_side in ("end", "both"):
            end_markers.append(int(np.clip(peak_global - offset, 0, t_len - 1)))

    start_idx = max(begin_markers) if begin_markers else 0
    end_idx = (min(end_markers) + 1) if end_markers else t_len
    start_idx = int(np.clip(start_idx, 0, t_len))
    end_idx = int(np.clip(end_idx, 0, t_len))
    return start_idx, end_idx


def _compute_final_missing_step_bounds(
    trial: MissingStepTrialResult,
    peak_offset_frames: int,
    edge_trim_frames: int,
) -> tuple[int, int, int, int]:
    original_len = len(trial.vgrf_r_bw)
    logic_start, logic_end = _compute_missing_step_trim_bounds(trial, peak_offset_frames)
    final_start = logic_start if logic_start > 0 else min(logic_end, logic_start + edge_trim_frames)
    final_end = logic_end if logic_end < original_len else max(logic_start, logic_end - edge_trim_frames)
    return logic_start, logic_end, final_start, final_end


def _make_height_vector(height_arr: np.ndarray, n_frames: int) -> np.ndarray:
    h = np.asarray(height_arr).squeeze()
    if h.ndim == 0:
        out = np.full(n_frames, float(h), dtype=np.float64)
    else:
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        if h.size == 1:
            out = np.full(n_frames, float(h[0]), dtype=np.float64)
        elif h.size >= n_frames:
            out = h[:n_frames]
        else:
            out = np.pad(h, (0, n_frames - h.size), mode="edge")

    if not np.all(np.isfinite(out)):
        raise ValueError("Height contains non-finite values.")
    if np.any(out <= 1e-8):
        raise ValueError("Height contains zero or negative values.")
    return out


def _interpolate_outlier_series(series: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, int]:
    x = np.asarray(series, dtype=np.float64)
    valid = np.isfinite(x) & (x >= lower) & (x <= upper)
    n_bad = int((~valid).sum())
    if n_bad == 0:
        return x, 0

    out = x.copy()
    idx = np.arange(x.shape[0], dtype=np.float64)
    if valid.any():
        out[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    else:
        out[:] = np.clip(np.nan_to_num(x, nan=0.0), lower, upper)
    return out, n_bad


def _analyze_cop_outlier_file(cop_path: Path) -> dict:
    try:
        cop = np.load(cop_path)
        if cop.ndim != 2 or cop.shape[1] < 6:
            return {"path": cop_path, "needs_fix": False, "error": f"Unexpected COP shape: {cop.shape}"}

        height_path = cop_path.parent / HEIGHT_FILENAME
        if not height_path.exists():
            return {"path": cop_path, "needs_fix": False, "error": f"Missing {HEIGHT_FILENAME}"}

        height = np.load(height_path)
        h_vec = _make_height_vector(height, cop.shape[0])
        cop_norm = cop / h_vec[:, None]

        channel_hits = {}
        for ch, (lb, ub) in COP_OUTLIER_CHANNEL_BOUNDS.items():
            vals = cop_norm[:, ch]
            bad = ~(np.isfinite(vals) & (vals >= lb) & (vals <= ub))
            n_bad = int(bad.sum())
            if n_bad > 0:
                channel_hits[ch] = {
                    "n_bad": n_bad,
                    "min_val": float(np.nanmin(vals)),
                    "max_val": float(np.nanmax(vals)),
                    "bounds": (lb, ub),
                }

        return {
            "path": cop_path,
            "needs_fix": bool(channel_hits),
            "channels": channel_hits,
            "n_frames": int(cop.shape[0]),
            "max_bad_pct": max(
                (100.0 * info["n_bad"] / max(int(cop.shape[0]), 1) for info in channel_hits.values()),
                default=0.0,
            ),
            "error": None,
        }
    except Exception as e:
        return {"path": cop_path, "needs_fix": False, "error": str(e)}


def _fix_cop_outlier_file(cop_path: Path) -> dict:
    cop = np.load(cop_path)
    height = np.load(cop_path.parent / HEIGHT_FILENAME)
    h_vec = _make_height_vector(height, cop.shape[0])
    cop_norm = cop / h_vec[:, None]
    fixed_counts = {}

    for ch, (lb, ub) in COP_OUTLIER_CHANNEL_BOUNDS.items():
        cleaned, n_fixed = _interpolate_outlier_series(cop_norm[:, ch], lb, ub)
        if n_fixed > 0:
            fixed_counts[ch] = int(n_fixed)
        cop_norm[:, ch] = cleaned

    cop_clean = cop_norm * h_vec[:, None]
    np.save(cop_path, cop_clean)
    return {"path": cop_path, "fixed_counts": fixed_counts}


def _move_trial_folder_to_bad_root(trial_dir: Path, bad_root: Path) -> Path:
    subject_name = trial_dir.parent.name
    trial_name = trial_dir.name
    dst_subject_dir = bad_root / subject_name
    dst_subject_dir.mkdir(parents=True, exist_ok=True)
    dst_trial_dir = dst_subject_dir / trial_name
    if dst_trial_dir.exists():
        raise FileExistsError(f"Destination already exists: {dst_trial_dir}")
    shutil.move(str(trial_dir), str(dst_subject_dir))
    return dst_trial_dir


def extract_patient_metadata(subject_path: Path) -> dict:
    """Read Patient_MD.json; return defaults if absent."""
    md_path = subject_path / "Patient_MD.json"
    defaults = {"Height_m": 1.7, "Mass_kg": 70.0}
    if not md_path.exists():
        return defaults
    try:
        with open(md_path) as f:
            data = json.load(f)
        return {
            "Height_m": float(data.get("Height_m", defaults["Height_m"])),
            "Mass_kg":  float(data.get("Mass_kg",  defaults["Mass_kg"])),
        }
    except Exception:
        return defaults


_MODEL_XML_CACHE: dict[str, Path] = {}
_MODEL_MASS_RESCALE_CACHE: set[str] = set()


class _XmlMassRescaleLock:
    """Small cross-process lock for idempotent XML mass rewrites."""

    def __init__(self, xml_path: Path, timeout_s: float = 300.0):
        self.lock_path = xml_path.with_name(xml_path.name + ".mass_rescale.lock")
        self.timeout_s = timeout_s
        self.fd: int | None = None

    def __enter__(self):
        start = time.monotonic()
        while True:
            try:
                self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"pid={os.getpid()}\n".encode("utf-8"))
                return self
            except FileExistsError:
                if time.monotonic() - start > self.timeout_s:
                    try:
                        age_s = time.time() - self.lock_path.stat().st_mtime
                    except OSError:
                        age_s = 0.0
                    if age_s > self.timeout_s:
                        try:
                            self.lock_path.unlink()
                            continue
                        except OSError:
                            pass
                    raise TimeoutError(f"Timed out waiting for XML mass-rescale lock: {self.lock_path}")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


def _maybe_rescale_model_xml_to_estimated_mass(xml_path: Path, subject_path: Path, cfg: dict) -> Path:
    """Rescale the selected model XML to Patient_MD Mass_kg when requested."""
    if not bool(cfg.get("RescaleModelsToEstimatedMass", False)):
        return xml_path

    xml_path = Path(xml_path)
    subject_path = Path(subject_path)
    cache_key = str(xml_path.resolve())
    if cache_key in _MODEL_MASS_RESCALE_CACHE:
        return xml_path

    md_path = subject_path / "Patient_MD.json"
    if not md_path.exists():
        raise FileNotFoundError(
            f"--rescale-models-to-estimated-mass requested, but {md_path} does not exist."
        )
    with open(md_path) as f:
        md = json.load(f)
    target_mass = md.get("Mass_kg")
    if target_mass is None or not np.isfinite(float(target_mass)) or float(target_mass) <= 0.0:
        raise ValueError(
            f"--rescale-models-to-estimated-mass requested for {subject_path.name}, "
            f"but Patient_MD.json has invalid Mass_kg={target_mass!r}."
        )
    source = md.get("Mass_kg_est_source")
    if source not in (None, "", "GRF_estimated"):
        warnings.warn(
            f"--rescale-models-to-estimated-mass using Patient_MD Mass_kg for {subject_path.name}, "
            f"but Mass_kg_est_source={source!r} is not 'GRF_estimated'.",
            RuntimeWarning,
        )

    try:
        from scripts.rescale_models_to_estimated_mass import rescale_file, sum_inertial_mass
    except Exception as e:
        raise RuntimeError(
            "Could not import scripts.rescale_models_to_estimated_mass; "
            "cannot rescale model XMLs to estimated mass."
        ) from e

    target_mass = float(target_mass)
    with _XmlMassRescaleLock(xml_path):
        current_mass = float(sum_inertial_mass(xml_path.read_text()))
        if current_mass <= 0.0:
            raise ValueError(f"{xml_path}: no positive inertial mass found")
        if abs(current_mass - target_mass) <= max(1e-6, 1e-8 * target_mass):
            _MODEL_MASS_RESCALE_CACHE.add(cache_key)
            return xml_path
        result = rescale_file(str(xml_path), target_mass)
        new_mass = float(result["m_new"])
        if abs(new_mass - target_mass) > max(1e-5, 1e-7 * target_mass):
            raise RuntimeError(
                f"Mass rescale verification failed for {xml_path}: "
                f"target={target_mass:.10g} kg, new={new_mass:.10g} kg."
            )
        print(
            f"  [ModelMassRescale] ✓ {subject_path.name}/{xml_path.name}: "
            f"{current_mass:.3f} kg -> {new_mass:.3f} kg"
        )
    _MODEL_MASS_RESCALE_CACHE.add(cache_key)
    return xml_path


def _fixed_model_has_canonical_knee(xml_path: Path) -> bool:
    """Return True only when an existing fixed XML has the canonical OpenCap knee block."""
    if knee_coupling_is_canonical_xml is None:
        return False
    try:
        return bool(knee_coupling_is_canonical_xml(xml_path))
    except Exception as e:
        print(f"  [ModelKneeCanonical] ↷ {xml_path.name}: canonical check failed ({e}); rebuilding.")
        return False


def _existing_fixed_model_can_be_reused(fixed_xml: Path, raw_xml: Path | None = None) -> bool:
    if not fixed_xml.exists():
        return False
    if _fixed_model_has_canonical_knee(fixed_xml):
        return True
    if raw_xml is not None and raw_xml.exists():
        print(f"  [ModelKneeCanonical] ↷ {fixed_xml.name}: stale/non-canonical knee definition; rebuilding from {raw_xml.name}.")
        return False
    print(f"  [ModelKneeCanonical] ⚠ {fixed_xml.name}: stale/non-canonical knee definition, but raw XML is unavailable; using existing fixed XML.")
    return True


def resolve_subject_model_xml(subject_path: Path, cfg: dict) -> Path:
    """
    Resolve which model XML to use for a subject.

    Behavior:
      - OpenCapVal source models use MyosuiteModel_<source>_FIXED.xml.
      - DontUseFixed=True uses raw MyosuiteModel.xml.
      - Otherwise use MyosuiteModel_FIXED.xml, generating or rebuilding as needed.
      - RescaleModelsToEstimatedMass=True rebuilds the fixed XML from raw XML first,
        then rescales that generated XML to Patient_MD Mass_kg.
    """
    # --OpenCapVal uses source-specific models (MyosuiteModel_MoCap.xml / _Video.xml).
    src = cfg.get("OPENCAPVAL_SOURCE")
    model_stem = f"MyosuiteModel_{src}" if src else "MyosuiteModel"
    mode = (
        "opencapval"
        if src else
        ("raw" if bool(cfg.get("DontUseFixed", False)) else
         ("fixed-rebuild" if not bool(cfg.get("UsedFIXEDModels", True)) else "fixed"))
    )
    mass_mode = "mass-rescale" if bool(cfg.get("RescaleModelsToEstimatedMass", False)) else "mass-native"
    key = f"{subject_path.resolve()}::{src or ''}::{mode}::{mass_mode}"
    if key in _MODEL_XML_CACHE and _MODEL_XML_CACHE[key].exists():
        return _MODEL_XML_CACHE[key]

    raw_xml = subject_path / f"{model_stem}.xml"
    fixed_xml = subject_path / f"{model_stem}_FIXED.xml"
    if src:
        rebuild_fixed = (
            not bool(cfg.get("UsedFIXEDModels", True))
            or bool(cfg.get("RescaleModelsToEstimatedMass", False))
        )
        if fixed_xml.exists() and not rebuild_fixed and _existing_fixed_model_can_be_reused(fixed_xml, raw_xml):
            selected_xml = _maybe_rescale_model_xml_to_estimated_mass(fixed_xml, subject_path, cfg)
            _MODEL_XML_CACHE[key] = selected_xml
            return selected_xml
        if not raw_xml.exists():
            raise FileNotFoundError(
                f"No model XML found for {subject_path.name}. Expected {raw_xml.name}."
            )
        if fix_xml_masses is None:
            raise RuntimeError(
                "Could not import ProcessAddbiomechnics.updateModel.fix_xml_masses; "
                f"cannot build required {fixed_xml.name}."
            )
        try:
            fix_xml_masses(str(raw_xml), str(fixed_xml))
            if not fixed_xml.exists():
                raise FileNotFoundError(f"Expected fixed XML not created: {fixed_xml}")
            selected_xml = _maybe_rescale_model_xml_to_estimated_mass(fixed_xml, subject_path, cfg)
            _MODEL_XML_CACHE[key] = selected_xml
            return selected_xml
        except Exception as e:
            raise RuntimeError(
                f"Failed to build fixed {src} model for {subject_path.name}: {e}."
            ) from e
    if bool(cfg.get("DontUseFixed", False)):
        if not raw_xml.exists():
            raise FileNotFoundError(
                f"No raw model XML found for {subject_path.name}. Expected {raw_xml.name}."
            )
        selected_xml = _maybe_rescale_model_xml_to_estimated_mass(raw_xml, subject_path, cfg)
        _MODEL_XML_CACHE[key] = selected_xml
        return selected_xml

    rebuild_fixed = (
        not bool(cfg.get("UsedFIXEDModels", True))
        or bool(cfg.get("RescaleModelsToEstimatedMass", False))
    )
    if fixed_xml.exists() and not rebuild_fixed and _existing_fixed_model_can_be_reused(fixed_xml, raw_xml):
        selected_xml = _maybe_rescale_model_xml_to_estimated_mass(fixed_xml, subject_path, cfg)
        _MODEL_XML_CACHE[key] = selected_xml
        return selected_xml

    if not raw_xml.exists():
        if fixed_xml.exists() and _existing_fixed_model_can_be_reused(fixed_xml, raw_xml):
            selected_xml = _maybe_rescale_model_xml_to_estimated_mass(fixed_xml, subject_path, cfg)
            _MODEL_XML_CACHE[key] = selected_xml
            return selected_xml
        raise FileNotFoundError(
            f"No model XML found for {subject_path.name}. "
            f"Expected {raw_xml.name} or {fixed_xml.name}."
        )

    if fix_xml_masses is None:
        raise RuntimeError(
            "Could not import ProcessAddbiomechnics.updateModel.fix_xml_masses; "
            "cannot build required MyosuiteModel_FIXED.xml. Pass --DontUseFixed to use raw XML."
        )

    try:
        fix_xml_masses(str(raw_xml), str(fixed_xml))
        if not fixed_xml.exists():
            raise FileNotFoundError(f"Expected fixed XML not created: {fixed_xml}")
        selected_xml = _maybe_rescale_model_xml_to_estimated_mass(fixed_xml, subject_path, cfg)
        _MODEL_XML_CACHE[key] = selected_xml
        return selected_xml
    except Exception as e:
        raise RuntimeError(
            f"Failed to build fixed model for {subject_path.name}: {e}. "
            "Pass --DontUseFixed to use raw XML."
        ) from e


def compute_trim_indices_by_grf(
    grf: np.ndarray,
    body_weight: float = 686.0,
    trim_grf_miss_steps: bool = True,
    trim_to_double_support: bool = False,
) -> np.ndarray:
    """
    Compute keep-indices using the same GRF trim logic used by trim_data_by_grf().

    GRF is expected in OpenSim coordinates where vertical columns are
    right=1 and left=4.
    """
    idx, _ = _compute_trim_indices_by_grf_with_trace(
        grf,
        body_weight=body_weight,
        trim_grf_miss_steps=trim_grf_miss_steps,
        trim_to_double_support=trim_to_double_support,
    )
    return idx


def _compute_trim_indices_by_grf_with_trace(
    grf: np.ndarray,
    body_weight: float = 686.0,
    trim_grf_miss_steps: bool = True,
    trim_to_double_support: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """GRF trimming plus a stage-by-stage audit trail."""
    MISSTEP_THRESHOLD_PCT = 0.70
    n = int(grf.shape[0])
    if n == 0:
        return np.array([], dtype=np.int32), []

    idx = np.arange(n, dtype=np.int32)
    stages: list[dict[str, Any]] = []
    cumulative_start = 0

    v_grf = grf[:, 1] + grf[:, 4]
    nonzero = np.where(v_grf > 0.01)[0]
    if len(nonzero) == 0:
        s1, e1 = 0, n
        details = {"no_nonzero_combined_vertical_grf_found": True}
    else:
        s1, e1 = int(nonzero[0]), int(nonzero[-1]) + 1
        details = {"no_nonzero_combined_vertical_grf_found": False}
    stages.append(_trace_stage(
        "trim_zero_grf_edges", n, s1, e1, cumulative_start,
        enabled=True,
        parameters={"combined_vertical_grf_threshold_n": 0.01},
        details=details,
    ))
    idx = idx[s1:e1]
    grf_work = grf[s1:e1]
    cumulative_start += s1

    if trim_grf_miss_steps:
        misstep_mask = np.ones(grf_work.shape[0], dtype=bool)
        thresh = body_weight * (1.0 / MISSTEP_THRESHOLD_PCT)
        misstep_mask[(grf_work[:, 1] > thresh) | (grf_work[:, 4] > thresh)] = False

        s2, e2 = find_longest_valid_segment(misstep_mask)
    else:
        thresh = body_weight * (1.0 / MISSTEP_THRESHOLD_PCT)
        misstep_mask = np.ones(grf_work.shape[0], dtype=bool)
        s2, e2 = 0, int(grf_work.shape[0])
    invalid_idx = np.where(~misstep_mask)[0]
    stages.append(_trace_stage(
        "trim_high_force_missteps", int(grf_work.shape[0]), s2, e2, cumulative_start,
        enabled=trim_grf_miss_steps,
        parameters={
            "body_weight_n": float(body_weight),
            "misstep_threshold_fraction_body_weight": float(1.0 / MISSTEP_THRESHOLD_PCT),
            "single_foot_vertical_grf_threshold_n": float(thresh),
            "selection": "longest_contiguous_valid_segment",
        },
        details={
            "invalid_frame_count": int(invalid_idx.size),
            "invalid_frame_indices_in_stage_input": invalid_idx.astype(int).tolist(),
        },
    ))
    idx = idx[s2:e2]
    grf_work = grf_work[s2:e2]
    cumulative_start += s2

    if trim_to_double_support:
        both_stance = (grf_work[:, 1] > 10.0) & (grf_work[:, 4] > 10.0)
        ds_idx = np.where(both_stance)[0]
        if len(ds_idx) >= 2:
            s3 = int(ds_idx[0])
            e3 = int(ds_idx[-1]) + 1
        else:
            s3, e3 = 0, int(grf_work.shape[0])
    else:
        both_stance = np.zeros(grf_work.shape[0], dtype=bool)
        ds_idx = np.array([], dtype=int)
        s3, e3 = 0, int(grf_work.shape[0])
    stages.append(_trace_stage(
        "trim_to_double_support", int(grf_work.shape[0]), s3, e3, cumulative_start,
        enabled=trim_to_double_support,
        parameters={
            "right_vertical_grf_threshold_n": 10.0,
            "left_vertical_grf_threshold_n": 10.0,
            "required_detected_frames": 2,
        },
        details={"detected_double_support_frame_count": int(ds_idx.size)},
    ))
    idx = idx[s3:e3]

    return idx, stages


def trim_data_by_grf(pos, vel, accel, grf, moment, cop, time,
                      body_weight: float = 686.0,
                      trim_grf_miss_steps: bool = True,
                      trim_to_double_support: bool = True,
                      trim_weak_edge_stances: bool = False):
    """
    1. Trim leading / trailing frames where BOTH feet have zero GRF.
    2. Optional misstep detection: flag frames where single-foot GRF > 1.43 × body_weight.
    3. (Optional) Trim to first/last double-support frame.

    Note: weak-edge-stance trimming (step 4) is handled separately in
    process_single_trial at step 6b, after treadmill detection.

    Returns trimmed versions of all inputs.
    """
    idx = compute_trim_indices_by_grf(
        grf,
        body_weight=body_weight,
        trim_grf_miss_steps=trim_grf_miss_steps,
        trim_to_double_support=trim_to_double_support,
    )
    return pos[idx], vel[idx], accel[idx], grf[idx], moment[idx], cop[idx], time[idx]


def calculate_treadmill_speed(ankle_pos_r:  np.ndarray,
                               ankle_pos_l:  np.ndarray,
                               grf_r:        np.ndarray,
                               grf_l:        np.ndarray,
                               pelvis_pos:   np.ndarray | None = None,
                               dt:           float = 0.01) -> float:
    """
    Robustly estimate treadmill belt speed from kinematics.

    Strategy (three independent estimates, best one wins):

    1. **Stance-midpoint velocity** — during each individual stance phase the
       foot should be nearly stationary in the lab frame (it moves backward at
       belt speed relative to the treadmill).  We detect each stance phase
       separately, fit a linear regression to ankle-X vs time for the middle
       60 % of that phase, and take the slope.  The negative of the mean slope
       across all stances is the belt speed.

    2. **Step-length / step-time** — the distance between the leading and
       trailing foot at consecutive heel-strikes divided by the stride period
       gives  2 × belt_speed  (both feet move backward at belt_speed while the
       body stays still).

    3. **Pelvis drift fallback** — if pelvis_pos is provided, the long-term
       linear drift of pelvis X over the trial is added to estimate 1 to
       account for any residual forward drift that wasn't removed.

    GRF columns: uses MuJoCo Z-up vertical — index 2 for right, 5 for left.
    Returns a non-negative speed in m/s.
    """
    from scipy.stats import linregress

    STANCE_THRESH   = 20.0   # N – minimum vertical GRF to call stance
    MIN_STANCE_FRAMES = 8    # reject very short stances
    MID_TRIM        = 0.20   # trim 20 % from each end of each stance before fitting

    # ── Helper: per-stance linear regression on ankle X ──────────
    def _stance_speeds(ankle_x: np.ndarray, vgrf: np.ndarray) -> list[float]:
        """Return list of ankle-X slopes (m/s) for each stance phase."""
        in_stance = (vgrf > STANCE_THRESH).astype(int)
        padded    = np.concatenate(([0], in_stance, [0]))
        starts    = np.where(np.diff(padded) ==  1)[0]
        ends      = np.where(np.diff(padded) == -1)[0]
        slopes = []
        for s, e in zip(starts, ends):
            n = e - s
            if n < MIN_STANCE_FRAMES:
                continue
            trim = max(1, int(n * MID_TRIM))
            seg_x = ankle_x[s + trim : e - trim]
            if len(seg_x) < 3:
                continue
            t_seg = np.arange(len(seg_x), dtype=float) * dt
            slope, *_ = linregress(t_seg, seg_x)
            slopes.append(float(slope))
        return slopes

    r_slopes = _stance_speeds(ankle_pos_r[:, 0], grf_r[:, 2])
    l_slopes = _stance_speeds(ankle_pos_l[:, 0], grf_l[:, 2])
    all_slopes = r_slopes + l_slopes

    speed_est1 = 0.0
    if all_slopes:
        # The ankle drifts backward at -belt_speed during stance
        speed_est1 = max(0.0, -float(np.median(all_slopes)))

    # ── Pelvis-drift correction ───────────────────────────────────
    # If the pelvis has a residual linear drift in X, that means the
    # treadmill correction was under-estimated.  Add it back.
    pelvis_drift = 0.0
    if pelvis_pos is not None and len(pelvis_pos) > 10:
        t_full = np.arange(len(pelvis_pos), dtype=float) * dt
        px_slope, *_ = linregress(t_full, pelvis_pos[:, 0])
        pelvis_drift = float(px_slope)   # positive = subject drifting forward

    # Belt speed = ankle regression speed + any residual pelvis drift
    speed_combined = max(0.0, speed_est1 + pelvis_drift)

    return float(speed_combined)


def detect_treadmill_like(
    ankle_pos_r: np.ndarray,
    ankle_pos_l: np.ndarray,
    qpos_matrix: np.ndarray,
    dt: float,
) -> tuple[bool, float, float]:
    """
    Match ProcessData treadmill detection heuristics.

    Returns:
      (is_treadmill, ankle_x_range_m, pelvis_net_speed_mps)
    """
    t_len = int(qpos_matrix.shape[0]) if qpos_matrix is not None else 0
    if t_len <= 1:
        return False, 0.0, 0.0

    ankle_x_range = (np.ptp(ankle_pos_r[:, 0]) + np.ptp(ankle_pos_l[:, 0])) / 2.0
    trial_duration_s = (t_len - 1) * float(dt)
    pelvis_net_speed = (
        abs(float(qpos_matrix[-1, 0]) - float(qpos_matrix[0, 0])) / max(trial_duration_s, 1e-6)
    )
    is_treadmill = (trial_duration_s >= 1.0) and (pelvis_net_speed < 0.3)
    return bool(is_treadmill), float(ankle_x_range), float(pelvis_net_speed)


def compute_weak_edge_trim_slice(
    grf_opensim: np.ndarray,
    body_weight: float,
    contact_threshold: float = 1.0,
    min_frames: int = 5,
    bw_frac_thresh: float = 0.65,
) -> tuple[slice, list[str]]:
    """
    ProcessData-style weak-edge trim on OpenSim-coord GRF (vertical cols 1 and 4).
    Returns a slice and debug messages.
    """
    t_len = int(grf_opensim.shape[0])
    if t_len <= 0:
        return slice(0, 0), []

    ct = float(contact_threshold)
    logs: list[str] = []

    def _stances_col(vgrf_col: np.ndarray) -> list[tuple[int, int]]:
        is_stance = (vgrf_col > ct)
        s_diff = np.diff(is_stance.astype(int), prepend=0)
        starts = np.where(s_diff == 1)[0]
        ends = np.where(s_diff == -1)[0]
        if len(ends) < len(starts):
            ends = np.append(ends, len(is_stance))
        return list(zip(starts, ends))

    def _mean_bw_ratio(vgrf_col: np.ndarray, s_idx: int, e_idx: int) -> float:
        if e_idx <= s_idx or body_weight <= 0:
            return 0.0
        return float(np.mean(vgrf_col[s_idx:e_idx]) / body_weight)

    def _first_contact_after(vgrf_col: np.ndarray, from_frame: int) -> int | None:
        in_stance = (vgrf_col > ct).astype(int)
        pad = np.concatenate(([0], in_stance, [0]))
        rises = np.where(np.diff(pad) == 1)[0]
        future = rises[rises >= from_frame]
        return int(future[0]) if len(future) > 0 else None

    new_start = 0
    new_end = t_len

    for this_col, other_col in [(1, 4), (4, 1)]:
        stances = _stances_col(grf_opensim[:, this_col])
        if not stances:
            continue

        for s0, e0 in stances:
            if (e0 - s0) < int(min_frames):
                continue
            bw_ratio = _mean_bw_ratio(grf_opensim[:, this_col], s0, e0)
            if s0 == 0 and bw_ratio < bw_frac_thresh:
                cut = _first_contact_after(grf_opensim[:, other_col], e0)
                if cut is not None and cut > new_start:
                    new_start = cut
                    logs.append(
                        f"Leading partial col {this_col}: mean vGRF/BW={bw_ratio:.2f} < {bw_frac_thresh:.2f} -> start={new_start}"
                    )
            break

        for sN, eN in reversed(stances):
            if (eN - sN) < int(min_frames):
                continue
            bw_ratio = _mean_bw_ratio(grf_opensim[:, this_col], sN, eN)
            if eN == t_len and bw_ratio < bw_frac_thresh:
                other_stances = _stances_col(grf_opensim[:, other_col])
                preceding = [(s, e) for s, e in other_stances if e <= sN]
                if preceding:
                    cut = int(preceding[-1][1])
                    if cut < new_end:
                        new_end = cut
                        logs.append(
                            f"Trailing partial col {this_col}: mean vGRF/BW={bw_ratio:.2f} < {bw_frac_thresh:.2f} -> end={new_end}"
                        )
            break

    return slice(int(new_start), int(new_end)), logs


# ═══════════════════════════════════════════════════════════════
#                    MJX / MUJOCO FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@jax.jit
def _compute_grf_contribution_jit(jacp, jacr, forces, torques):
    """
    jacp  : (T, B, 3, nv)
    jacr  : (T, B, 3, nv)
    forces: (T, B, 3)
    torques:(T, B, 3)
    Returns (T, nv).
    """
    qfrc_f = jnp.einsum("tbij,tbi->tj", jacp, forces)
    qfrc_t = jnp.einsum("tbij,tbi->tj", jacr,  torques)
    return qfrc_f + qfrc_t


def compute_grf_contribution(jacp, jacr, forces, torques):
    """Batched GRF contribution via JIT-compiled einsum."""
    return np.asarray(_compute_grf_contribution_jit(
        jnp.asarray(jacp), jnp.asarray(jacr),
        jnp.asarray(forces), jnp.asarray(torques),
    ))


def compute_cop_clean_and_id(
    GRF_mj: np.ndarray,
    Moment_mj: np.ndarray,
    COP_mj: np.ndarray,
    ankle_pos_r_corr: np.ndarray,
    ankle_pos_l_corr: np.ndarray,
    jacobian_data: dict,
    qfrc_inverse_batch: np.ndarray,
    cfg: dict,
    fs: float,
    GRF_for_torque_mj: np.ndarray | None = None,
):
    """
    Compute cleaned relative COP and ID_GT_MJX from force/moment/COP/Jacobians.
    """
    r_vec_r = COP_mj[:, 0:3] - ankle_pos_r_corr
    r_vec_l = COP_mj[:, 3:6] - ankle_pos_l_corr

    cop_rel = np.column_stack([
        r_vec_r[:, 0], r_vec_r[:, 1],
        r_vec_l[:, 0], r_vec_l[:, 1],
    ])

    if bool(cfg.get("ENABLE_COP_CLEANING", True)):
        mask_r = GRF_mj[:, 2] < cfg["GRF_CONTACT_THRESHOLD"]
        mask_l = GRF_mj[:, 5] < cfg["GRF_CONTACT_THRESHOLD"]
        r_vec_r[mask_r] = 0.0
        r_vec_l[mask_l] = 0.0

        cop_rel = np.column_stack([
            r_vec_r[:, 0], r_vec_r[:, 1],
            r_vec_l[:, 0], r_vec_l[:, 1],
        ])
        cop_rel = clean_and_filter_cop(
            cop_rel,
            GRF_mj,
            trim_start_frames=cfg["COP_TRIM_START_FRAMES"],
            trim_end_frames=cfg["COP_TRIM_END_FRAMES"],
            extrapolation_frames=int(cfg.get("COP_EXTRAPOLATION_FRAMES", 6)),
            pad_width=cfg["COP_FILTER_PAD_WIDTH"],
            edge_hold=bool(cfg.get("COP_EdgeHold", False)),
            cutoff=cfg["FILTER_CUTOFF_HZ"],
            fs=fs,
            order=int(cfg.get("FILTER_ORDER", 2)),
        )

        cop_rel[mask_r, 0:2] = 0.0
        cop_rel[mask_l, 2:4] = 0.0

        r_vec_r[:, 0] = cop_rel[:, 0]
        r_vec_r[:, 1] = cop_rel[:, 1]
        r_vec_l[:, 0] = cop_rel[:, 2]
        r_vec_l[:, 1] = cop_rel[:, 3]

    grf_torque_mj = GRF_mj if GRF_for_torque_mj is None else np.asarray(GRF_for_torque_mj)
    if grf_torque_mj.shape != GRF_mj.shape:
        raise ValueError(
            f"GRF_for_torque_mj shape {grf_torque_mj.shape} does not match GRF_mj {GRF_mj.shape}"
        )

    grf_r_np = grf_torque_mj[:, 0:3]
    grf_l_np = grf_torque_mj[:, 3:6]
    mom_r_np = Moment_mj[:, 0:3]
    mom_l_np = Moment_mj[:, 3:6]

    mom_added_r = np.cross(r_vec_r, grf_r_np)
    mom_added_l = np.cross(r_vec_l, grf_l_np)

    ext_force = np.zeros((GRF_mj.shape[0], 2, 6), dtype=np.float32)
    ext_force[:, 0, 0:3] = grf_r_np
    ext_force[:, 0, 3:6] = mom_r_np + mom_added_r
    ext_force[:, 1, 0:3] = grf_l_np
    ext_force[:, 1, 3:6] = mom_l_np + mom_added_l

    qfrc_grf = compute_grf_contribution(
        jacobian_data["jacp"],
        jacobian_data["jacr"],
        ext_force[:, :, 0:3],
        ext_force[:, :, 3:6],
    )
    id_gt_mjx = qfrc_inverse_batch - qfrc_grf
    return cop_rel, qfrc_grf, id_gt_mjx


# ── Per-model JIT cache ───────────────────────────────────────────────────────
# JAX traces a new specialisation for each unique mjx_model structure (i.e. each
# subject).  Caching the compiled function avoids re-tracing across chunks of the
# same trial AND across trials of the same subject.
_ID_JIT_CACHE: dict = {}


def _get_id_jit(mjx_model):
    """Return (or build + cache) the jit(vmap(single_id)) fn for this model."""
    key = id(mjx_model)
    if key not in _ID_JIT_CACHE:
        def _single_id(q, v, a):
            d = mjx.make_data(mjx_model)
            d = d.replace(qpos=q, qvel=v, qacc=a)
            d = mjx.inverse(mjx_model, d)
            return d.qfrc_inverse, d.qfrc_constraint, d.subtree_com[0]

        _ID_JIT_CACHE[key] = jax.jit(jax.vmap(_single_id))
    return _ID_JIT_CACHE[key]


def compute_inverse_dynamics_batch(mjx_model, qpos_jnp, qvel_jnp, qacc_jnp):
    """
    Vectorised, JIT-compiled MJX inverse dynamics over a batch of frames.

    The vmapped+JIT function is compiled once per unique mjx_model and then
    reused across all chunks and all trials that share the same model structure,
    avoiding repeated tracing overhead.

    Returns (qfrc_inverse, qfrc_constraint, subtree_com_body0)
    each with a leading time dimension.
    """
    return _get_id_jit(mjx_model)(qpos_jnp, qvel_jnp, qacc_jnp)


def compute_inverse_dynamics_chunked(mjx_model,
                                     qpos_matrix: np.ndarray,
                                     qvel_matrix: np.ndarray,
                                     qacc_matrix: np.ndarray,
                                     chunk_size: int = 200):
    """
    Chunked wrapper to limit peak RAM use for long trials.
    """
    T = qpos_matrix.shape[0]
    if T == 0:
        nv = qvel_matrix.shape[1] if qvel_matrix.ndim == 2 else 0
        return (
            np.empty((0, nv), dtype=np.float32),
            np.empty((0, nv), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    qinv_parts = []
    qcon_parts = []
    com_parts = []
    step = max(1, int(chunk_size))
    for s in range(0, T, step):
        e = min(T, s + step)
        qpos_jnp = jnp.asarray(qpos_matrix[s:e])
        qvel_jnp = jnp.asarray(qvel_matrix[s:e])
        qacc_jnp = jnp.asarray(qacc_matrix[s:e])
        qinv, qcon, com = compute_inverse_dynamics_batch(
            mjx_model, qpos_jnp, qvel_jnp, qacc_jnp
        )
        # Convert to numpy immediately so JAX device buffers can be freed
        qinv_parts.append(np.asarray(qinv))
        qcon_parts.append(np.asarray(qcon))
        com_parts.append(np.asarray(com))
        # Explicitly delete JAX arrays and trigger GC to release device memory
        del qpos_jnp, qvel_jnp, qacc_jnp, qinv, qcon, com
        gc.collect()

    return (
        np.concatenate(qinv_parts, axis=0),
        np.concatenate(qcon_parts, axis=0),
        np.concatenate(com_parts, axis=0),
    )


def setup_and_precompute_jacobians(mj_model, qpos_matrix: np.ndarray):
    """
    Compute per-frame Jacobians for calcn_r and calcn_l.
    Returns (mjx_model, jacobian_data, calcn_r_id, calcn_l_id,
             toes_r_id, toes_l_id).
    """
    calcn_r_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
    calcn_l_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "calcn_l")
    toes_r_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "toes_r")
    toes_l_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "toes_l")
    body_ids   = [calcn_r_id, calcn_l_id]

    T  = qpos_matrix.shape[0]
    nv = mj_model.nv
    jacp_all = np.zeros((T, 2, 3, nv))
    jacr_all = np.zeros((T, 2, 3, nv))

    mj_data = mujoco.MjData(mj_model)
    for t in range(T):
        mj_data.qpos[:] = qpos_matrix[t]
        mujoco.mj_kinematics(mj_model, mj_data)
        mujoco.mj_comPos(mj_model, mj_data)
        for i, bid in enumerate(body_ids):
            jp = np.zeros((3, nv))
            jr = np.zeros((3, nv))
            mujoco.mj_jacBody(mj_model, mj_data, jp, jr, bid)
            jacp_all[t, i] = jp
            jacr_all[t, i] = jr

    mjx_model = mjx.put_model(mj_model)
    jacobian_data = {
        "jacp":     jacp_all,
        "jacr":     jacr_all,
        "body_ids": np.array(body_ids),
    }
    return mjx_model, jacobian_data, calcn_r_id, calcn_l_id, toes_r_id, toes_l_id


# ═══════════════════════════════════════════════════════════════
#                 DEVIATION-LEARNING HELPERS
# ═══════════════════════════════════════════════════════════════

def _extract_trial_stance_metrics(
    grf_data:    np.ndarray,   # (T, 6) MuJoCo coords
    cop_data:    np.ndarray,   # (T, 4) calcaneus-relative
    moment_data: np.ndarray,   # (T, 6) MuJoCo coords
    ankle_pos:   np.ndarray,   # (T, 3) right ankle, MuJoCo coords
    toes_pos:    np.ndarray,   # (T, 3) right toes,  MuJoCo coords
    ankle_pos_l: np.ndarray,   # (T, 3) left ankle
    toes_pos_l:  np.ndarray,   # (T, 3) left toes
    mass:   float,
    height: float,
    threshold:    float = 1.0,
    min_duration: int   = 10,
) -> dict:
    """
    Extract per-stance normalised GRF / COP / Moment curves and FPA for one
    trial.  Mirrors detect_stance_phases() in PrepForDeviationLearning.py.

    Returns a dict matching the PrepForDeviationLearning metrics structure:
      { 'Right': { 'durations', 'fpa', 'grf_curves', 'cop_curves',
                   'moment_curves', 'intervals', 'partials' },
        'Left':  { ... } }
    Full stances are stored in 'durations'/'fpa'/'intervals'; boundary-partial
    stances are stored in 'partials' as (start, end, edge).
    """
    results = {
        side: {"durations": [], "fpa": [], "grf_curves": [], "cop_curves": [],
               "moment_curves": [], "intervals": [], "partials": []}
        for side in ("Right", "Left")
    }

    T = grf_data.shape[0]
    # (side_name, vgrf_col, grf_cols, cop_cols, mz_col, ankle_arr, toes_arr)
    configs = [
        ("Right", 2, [0, 1, 2], [0, 1], 2, ankle_pos,   toes_pos),
        ("Left",  5, [3, 4, 5], [2, 3], 5, ankle_pos_l, toes_pos_l),
    ]

    for side_name, vgrf_idx, grf_cols, cop_cols, mz_col, a_pos, t_pos in configs:
        vgrf      = grf_data[:, vgrf_idx]
        is_stance = vgrf > threshold
        padded    = np.concatenate(([False], is_stance, [False]))
        diffs     = np.diff(padded.astype(int))
        starts    = np.where(diffs ==  1)[0]
        ends      = np.where(diffs == -1)[0]

        for s, e in zip(starts, ends):
            dur = int(e - s)
            if dur < min_duration:
                continue

            # Boundary-partial stances
            if s == 0 or e == T:
                edge = "begin" if s == 0 else "end"
                results[side_name]["partials"].append((int(s), int(e), edge))
                continue

            # ── Foot Progression Angle ──────────────────────────────
            ankle_slice = a_pos[s:e, :2]   # XY only (MuJoCo: forward=X, lateral=Y)
            toes_slice  = t_pos[s:e, :2]
            foot_vecs   = toes_slice - ankle_slice
            angles_deg  = np.degrees(np.arctan2(foot_vecs[:, 1], foot_vecs[:, 0]))
            mean_fpa    = float(np.mean(angles_deg))

            # ── GRF (X,Y,Z) normalised by body weight ───────────────
            grf_slice = grf_data[s:e, grf_cols].copy()
            grf_slice /= (mass * 9.8067)
            grf_interp = _interpolate_101(grf_slice)

            # ── COP (X,Y) normalised by height ──────────────────────
            cop_slice = cop_data[s:e, cop_cols].copy()
            cop_slice /= height
            cop_interp = _interpolate_101(cop_slice)

            # ── Moment (Mz) normalised by BW*height ─────────────────
            mom_slice = moment_data[s:e, mz_col:mz_col + 1].copy()
            mom_slice /= (mass * 9.8067 * height)
            mom_interp = _interpolate_101(mom_slice)

            results[side_name]["durations"].append(dur)
            results[side_name]["fpa"].append(mean_fpa)
            results[side_name]["grf_curves"].append(grf_interp)
            results[side_name]["cop_curves"].append(cop_interp)
            results[side_name]["moment_curves"].append(mom_interp)
            results[side_name]["intervals"].append((int(s), int(e)))

    return results


def _build_deviation_averages_from_trials(
    trials: list,   # list of (subject_path, trial_path) Path tuples
    cfg:    dict,
) -> dict | None:
    """
    Scan all already-processed trials, load their GRF/COP/Moment/ankle/toes
    numpy files, extract per-stance normalised curves, then aggregate into
    duration-binned GRF+Moment averages and FPA-binned COP averages.

    This mirrors the aggregation logic in PrepForDeviationLearning.py exactly.

    The result dict is also saved to cfg["DEVIATION_METRICS_PKL_PATH"] so
    subsequent runs can load it directly with DEVIATION_LOAD_AVERAGES_FROM_FILE=True.

    Returns the same structure as load_deviation_data(), or None on failure.
    """
    print("  [Deviation] Building stance averages from processed trials …")

    all_raw = {
        side: {"durations": [], "fpa": [], "grf_curves": [], "cop_curves": [],
               "moment_curves": []}
        for side in ("Right", "Left")
    }

    output_dir_name = get_output_dir_name(cfg)
    loaded = 0
    skipped = 0
    for sp, tp in trials:
        proc_dir = tp / output_dir_name
        # Need GRF, COP_Cleaned_Relative, Moment_Cleaned, ankle_pos_r/l, and
        # Patient_MD.json for mass/height.
        grf_path    = proc_dir / "GRF_Cleaned.npy"
        cop_path    = proc_dir / "COP_Cleaned_Relative.npy"
        mom_path    = proc_dir / "Moment_Cleaned.npy"
        ank_r_path  = proc_dir / "ankle_pos_r.npy"
        ank_l_path  = proc_dir / "ankle_pos_l.npy"
        meta_path   = sp / "Patient_MD.json"

        # Toes positions — saved by ProcessData as toes_pos_r/l if present,
        # otherwise we fall back to the ankle position (FPA will be 0, safe
        # because global_median_fpa is used for COP lookup anyway).
        toes_r_path = proc_dir / "toes_pos_r.npy"
        toes_l_path = proc_dir / "toes_pos_l.npy"

        required = [grf_path, cop_path, mom_path, ank_r_path, ank_l_path, meta_path]
        if not all(p.exists() for p in required):
            skipped += 1
            continue

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            mass   = float(meta.get("Mass_kg",  70.0))
            height = float(meta.get("Height_m",  1.7))

            grf_data    = np.load(grf_path)
            cop_data    = np.load(cop_path)
            moment_data = np.load(mom_path)
            ankle_r     = np.load(ank_r_path)   # (T, 3)
            ankle_l     = np.load(ank_l_path)   # (T, 3)

            # Toes: use saved file when available, else duplicate ankle
            toes_r = np.load(toes_r_path) if toes_r_path.exists() else ankle_r.copy()
            toes_l = np.load(toes_l_path) if toes_l_path.exists() else ankle_l.copy()

            # Align all arrays to the same length
            min_T = min(grf_data.shape[0], cop_data.shape[0], moment_data.shape[0],
                        ankle_r.shape[0], ankle_l.shape[0])
            grf_data    = grf_data[:min_T]
            cop_data    = cop_data[:min_T]
            moment_data = moment_data[:min_T]
            ankle_r     = ankle_r[:min_T]
            ankle_l     = ankle_l[:min_T]
            toes_r      = toes_r[:min_T]
            toes_l      = toes_l[:min_T]

            # Skip trials whose COP is not calcaneus-relative
            if np.max(np.abs(cop_data)) > 0.4:
                skipped += 1
                continue

            metrics = _extract_trial_stance_metrics(
                grf_data, cop_data, moment_data,
                ankle_r, toes_r, ankle_l, toes_l,
                mass, height,
            )

            for side in ("Right", "Left"):
                for key in ("durations", "fpa", "grf_curves", "cop_curves", "moment_curves"):
                    all_raw[side][key].extend(metrics[side][key])

            loaded += 1

        except Exception as e:
            warnings.warn(f"[Deviation] skipping {sp.name}/{tp.name}: {e}")
            skipped += 1

    print(f"  [Deviation] Loaded {loaded} trials, skipped {skipped}.")

    if loaded == 0:
        print("  [Deviation] No trials loaded — cannot build averages. "
              "Deviation prep will be skipped.")
        return None

    # ── Global medians ────────────────────────────────────────────────────────
    all_durations = all_raw["Right"]["durations"] + all_raw["Left"]["durations"]
    all_fpas      = all_raw["Right"]["fpa"]       + all_raw["Left"]["fpa"]
    global_median_duration = int(np.median(all_durations)) if all_durations else 50
    global_median_fpa      = float(np.median(all_fpas))    if all_fpas      else 0.0
    print(f"  [Deviation] Global median stance duration: {global_median_duration} frames")
    print(f"  [Deviation] Global median FPA:             {global_median_fpa:.2f}°")

    # ── Pool both feet for combined GRF X and Z (bilateral symmetric) ────────
    combined_dur_buckets: dict[int, list] = {}
    for side in ("Right", "Left"):
        for dur, curve in zip(all_raw[side]["durations"], all_raw[side]["grf_curves"]):
            bin_dur = (dur // 2) * 2
            combined_dur_buckets.setdefault(bin_dur, []).append(curve)

    combined_grf_means: dict[int, dict] = {}
    for bin_dur, curves in combined_dur_buckets.items():
        combined_grf_means[bin_dur] = {
            "mean": np.mean(curves, axis=0),
            "n":    len(curves),
        }

    # ── Side-specific aggregation (mirrors PrepForDeviationLearning.py) ──────
    final_output: dict = {
        side: {"grf_by_duration": {}, "cop_by_fpa": {}, "moment_by_duration": {}}
        for side in ("Right", "Left")
    }

    for side in ("Right", "Left"):
        # GRF + Moment — binned by duration
        dur_buckets:    dict[int, list] = {}
        moment_buckets: dict[int, list] = {}
        for dur, grf_c, mom_c in zip(all_raw[side]["durations"],
                                      all_raw[side]["grf_curves"],
                                      all_raw[side]["moment_curves"]):
            bin_dur = (dur // 2) * 2
            dur_buckets.setdefault(bin_dur, []).append(grf_c)
            moment_buckets.setdefault(bin_dur, []).append(mom_c)

        for bin_dur, curves in dur_buckets.items():
            if len(curves) < 5:   # require at least 5 samples (same as PrepForDeviationLearning)
                continue
            side_mean = np.mean(curves, axis=0)
            comb      = combined_grf_means[bin_dur]
            # Inject bilateral-pooled X (forward) and Z (vertical) into the side mean
            side_mean[:, 0] = comb["mean"][:, 0]
            side_mean[:, 2] = comb["mean"][:, 2]
            final_output[side]["grf_by_duration"][bin_dur] = {
                "mean": side_mean,
                "n":    comb["n"],
            }
            final_output[side]["moment_by_duration"][bin_dur] = {
                "mean": np.mean(moment_buckets[bin_dur], axis=0),
                "n":    len(moment_buckets[bin_dur]),
            }

        # COP — binned by FPA (1-degree bins)
        fpa_buckets: dict[int, list] = {}
        for fpa, cop_c in zip(all_raw[side]["fpa"], all_raw[side]["cop_curves"]):
            bin_fpa = round(fpa)
            fpa_buckets.setdefault(bin_fpa, []).append(cop_c)

        for bin_fpa, curves in fpa_buckets.items():
            if len(curves) < 5:
                continue
            final_output[side]["cop_by_fpa"][bin_fpa] = {
                "mean": np.mean(curves, axis=0),
                "n":    len(curves),
            }

    # ── Save pkl so next run can use DEVIATION_LOAD_AVERAGES_FROM_FILE=True ──
    pkl_path = Path(cfg["DEVIATION_METRICS_PKL_PATH"])
    save_dict = {
        "final_output":           final_output,
        "global_median_duration": global_median_duration,
        "global_median_fpa":      global_median_fpa,
    }
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"  [Deviation] Saved averages to {pkl_path}")
    except Exception as e:
        warnings.warn(f"[Deviation] Could not save pkl: {e}")

    return {
        "final_output":           final_output,
        "global_median_duration": global_median_duration,
        "global_median_fpa":      global_median_fpa,
    }


def build_average_reconstructions(grf_data:    np.ndarray,
                                   cop_data:    np.ndarray,
                                   moment_data: np.ndarray,
                                   final_output: dict,
                                   global_median_duration: int,
                                   global_median_fpa:      float,
                                   mass:   float,
                                   height: float,
                                   ankle_pos_r: np.ndarray = None,
                                   toes_pos_r:  np.ndarray = None,
                                   ankle_pos_l: np.ndarray = None,
                                   toes_pos_l:  np.ndarray = None) -> tuple:
    """
    Reconstruct average GRF / COP / Moment curves for a single trial.
    Mirrors the logic in PrepForDeviationLearning.py exactly.
    Returns (recon_grf (T,6), recon_cop (T,4), recon_moment (T,6)).
    """
    T = grf_data.shape[0]
    recon_grf    = np.zeros((T, 6))
    recon_cop    = np.zeros((T, 4))
    recon_moment = np.zeros((T, 6))

    # Detect all stances including boundary partials.
    # Full stances: partial_begin=False, partial_end=False → stored in 'intervals' + 'fpa'
    # Partial stances: boundary stances → stored in 'partials' as (start, end, edge)
    stances = _detect_stance_phases(grf_data, include_boundary=True)

    # Build parallel structures matching PrepForDeviationLearning's metrics dict so we
    # can reuse the same indexing logic for both full and partial stances.
    # Full stances carry a per-stance FPA; partials use the trial-median FPA.
    # We need FPA for COP lookup — compute it here from ankle/toes positions embedded in
    # grf_data's stance windows.  Since ProcessData doesn't pass ankle/toes into this
    # function we use global_median_fpa as the per-stance FPA fallback, which is
    # the same value PrepForDeviationLearning uses when no ankle data is available.

    # ── Helper: align-and-clip for partial stances ──────────────────────────
    def align_and_clip(full_curve, partial_len, edge):
        """Align a full average curve to a partial stance window.

        edge='begin': right-align (end of curve → end of partial window)
        edge='end':   left-align  (start of curve → start of partial window)
        """
        n_full = full_curve.shape[0]
        n_cols = full_curve.shape[1]
        if partial_len <= n_full:
            return full_curve[-partial_len:, :] if edge == "begin" else full_curve[:partial_len, :]
        padded = np.zeros((partial_len, n_cols))
        if edge == "begin":
            padded[-n_full:, :] = full_curve   # right-align, leading frames = 0
        else:
            padded[:n_full, :] = full_curve    # left-align, trailing frames = 0
        return padded

    # ── Full-stance reconstruction ───────────────────────────────────────────
    # Collect per-side intervals from _detect_stance_phases.  Complete stances
    # have partial_begin=False and partial_end=False.
    for side in ["Right", "Left"]:
        grf_off = 0 if side == "Right" else 3
        cop_off = 0 if side == "Right" else 2
        mz_col  = 2 if side == "Right" else 5
        mom_off = 0 if side == "Right" else 3

        full_phases = [ph for ph in stances[side]
                       if not ph.get("partial_begin") and not ph.get("partial_end")]

        for i, ph in enumerate(full_phases):
            s, e   = ph["start"], ph["end"]
            dur    = ph["duration_frames"]
            bin_dur = (dur // 2) * 2

            # 1. GRF — index by duration
            avail_grf = final_output[side]["grf_by_duration"]
            if avail_grf:
                closest_dur = min(avail_grf, key=lambda k: abs(k - bin_dur))
                avg_grf = avail_grf[closest_dur]
                grf_mean = avg_grf["mean"] if isinstance(avg_grf, dict) else avg_grf
                grf_chunk = _interpolate_to_len(np.asarray(grf_mean), dur)
                grf_chunk *= (mass * 9.8067)
                if grf_chunk.shape[1] == 3:
                    recon_grf[s:e, grf_off:grf_off + 3] = grf_chunk

            # 2. Moment — index by duration (same bin as GRF)
            avail_mom = final_output[side]["moment_by_duration"]
            if avail_mom:
                closest_dur_m = min(avail_mom, key=lambda k: abs(k - bin_dur))
                avg_mom = avail_mom[closest_dur_m]
                mom_mean = avg_mom["mean"] if isinstance(avg_mom, dict) else avg_mom
                mom_chunk = _interpolate_to_len(np.asarray(mom_mean), dur)
                mom_chunk *= (mass * 9.8067 * height)
                if mom_chunk.shape[1] == 1:
                    recon_moment[s:e, mz_col:mz_col + 1] = mom_chunk
                elif mom_chunk.shape[1] == 3:
                    recon_moment[s:e, mom_off:mom_off + 3] = mom_chunk

            # 3. COP — index by per-stance FPA
            # Compute FPA from ankle/toes if available (matches PrepForDeviationLearning).
            # Fall back to global_median_fpa when ankle data is not provided.
            avail_cop = final_output[side]["cop_by_fpa"]
            if avail_cop:
                stance_fpa = global_median_fpa  # default fallback
                if side == "Right" and ankle_pos_r is not None and toes_pos_r is not None:
                    ankle_slice = ankle_pos_r[s:e, :2]
                    toes_slice  = toes_pos_r[s:e, :2]
                    foot_vecs   = toes_slice - ankle_slice
                    angles_deg  = np.degrees(np.arctan2(foot_vecs[:, 1], foot_vecs[:, 0]))
                    stance_fpa  = float(np.mean(angles_deg))
                elif side == "Left" and ankle_pos_l is not None and toes_pos_l is not None:
                    ankle_slice = ankle_pos_l[s:e, :2]
                    toes_slice  = toes_pos_l[s:e, :2]
                    foot_vecs   = toes_slice - ankle_slice
                    angles_deg  = np.degrees(np.arctan2(foot_vecs[:, 1], foot_vecs[:, 0]))
                    stance_fpa  = float(np.mean(angles_deg))
                bin_fpa     = round(stance_fpa)
                closest_fpa = min(avail_cop, key=lambda k: abs(k - bin_fpa))
                avg_cop    = avail_cop[closest_fpa]
                cop_mean   = avg_cop["mean"] if isinstance(avg_cop, dict) else avg_cop
                cop_chunk  = _interpolate_to_len(np.asarray(cop_mean), dur)
                cop_chunk  *= height
                if cop_chunk.shape[1] == 2:
                    recon_cop[s:e, cop_off:cop_off + 2] = cop_chunk
                elif cop_chunk.shape[1] == 3:
                    recon_cop[s:e, cop_off:cop_off + 2] = cop_chunk[:, :min(3, 4 - cop_off)]

    # ── Partial-stance reconstruction ────────────────────────────────────────
    # Compute trial-median duration and FPA pooled from BOTH feet's full stances,
    # then fall back to global medians if no full stances exist. This exactly
    # mirrors the PrepForDeviationLearning logic.
    all_trial_durations = (
        [ph["duration_frames"] for ph in stances["Right"]
         if not ph.get("partial_begin") and not ph.get("partial_end")]
        + [ph["duration_frames"] for ph in stances["Left"]
           if not ph.get("partial_begin") and not ph.get("partial_end")]
    )
    trial_median_dur = int(np.median(all_trial_durations)) if all_trial_durations else int(global_median_duration)

    # Compute trial-median FPA from full stances when ankle/toes data is available.
    # This mirrors PrepForDeviationLearning.py's use of per-trial medians for partials.
    all_trial_fpas = []
    for side, a_pos, t_pos in [("Right", ankle_pos_r, toes_pos_r), ("Left", ankle_pos_l, toes_pos_l)]:
        for ph in stances[side]:
            if ph.get("partial_begin") or ph.get("partial_end"):
                continue
            s, e = ph["start"], ph["end"]
            if a_pos is not None and t_pos is not None:
                ankle_sl = a_pos[s:e, :2]
                toes_sl  = t_pos[s:e, :2]
                foot_vecs = toes_sl - ankle_sl
                angles_deg = np.degrees(np.arctan2(foot_vecs[:, 1], foot_vecs[:, 0]))
                all_trial_fpas.append(float(np.mean(angles_deg)))
    trial_median_fpa = float(np.median(all_trial_fpas)) if all_trial_fpas else float(global_median_fpa)

    for side in ["Right", "Left"]:
        grf_off = 0 if side == "Right" else 3
        cop_off = 0 if side == "Right" else 2
        mz_col  = 2 if side == "Right" else 5
        mom_off = 0 if side == "Right" else 3

        partial_phases = [ph for ph in stances[side]
                          if ph.get("partial_begin") or ph.get("partial_end")]

        for ph in partial_phases:
            p_start = ph["start"]
            p_end   = ph["end"]
            partial_len = ph["duration_frames"]
            edge = "begin" if ph.get("partial_begin") else "end"

            bin_dur = (trial_median_dur // 2) * 2
            bin_fpa = round(trial_median_fpa)

            # 1. GRF
            avail_grf = final_output[side]["grf_by_duration"]
            if avail_grf:
                closest_dur = min(avail_grf, key=lambda k: abs(k - bin_dur))
                avg_grf = avail_grf[closest_dur]
                grf_mean = avg_grf["mean"] if isinstance(avg_grf, dict) else avg_grf
                full_grf = _interpolate_to_len(np.asarray(grf_mean), trial_median_dur)
                full_grf *= (mass * 9.8067)
                clipped_grf = align_and_clip(full_grf, partial_len, edge)
                if clipped_grf.shape[1] == 3:
                    recon_grf[p_start:p_end, grf_off:grf_off + 3] = clipped_grf

            # 2. Moment
            avail_mom = final_output[side]["moment_by_duration"]
            if avail_mom:
                closest_dur_m = min(avail_mom, key=lambda k: abs(k - bin_dur))
                avg_mom = avail_mom[closest_dur_m]
                mom_mean = avg_mom["mean"] if isinstance(avg_mom, dict) else avg_mom
                full_mom = _interpolate_to_len(np.asarray(mom_mean), trial_median_dur)
                full_mom *= (mass * 9.8067 * height)
                clipped_mom = align_and_clip(full_mom, partial_len, edge)
                if clipped_mom.shape[1] == 1:
                    recon_moment[p_start:p_end, mz_col:mz_col + 1] = clipped_mom
                elif clipped_mom.shape[1] == 3:
                    recon_moment[p_start:p_end, mom_off:mom_off + 3] = clipped_mom

            # 3. COP
            avail_cop = final_output[side]["cop_by_fpa"]
            if avail_cop:
                closest_fpa = min(avail_cop, key=lambda k: abs(k - bin_fpa))
                avg_cop = avail_cop[closest_fpa]
                cop_mean = avg_cop["mean"] if isinstance(avg_cop, dict) else avg_cop
                full_cop = _interpolate_to_len(np.asarray(cop_mean), trial_median_dur)
                full_cop *= height
                clipped_cop = align_and_clip(full_cop, partial_len, edge)
                if clipped_cop.shape[1] == 2:
                    recon_cop[p_start:p_end, cop_off:cop_off + 2] = clipped_cop
                elif clipped_cop.shape[1] == 3:
                    recon_cop[p_start:p_end, cop_off:cop_off + 2] = clipped_cop[:, :min(3, 4 - cop_off)]

    return recon_grf, recon_cop, recon_moment


# ═══════════════════════════════════════════════════════════════
#                    CORE TRIAL PROCESSOR
# ═══════════════════════════════════════════════════════════════

def _process_single_trial_processed_core(subject_path: Path, trial_path: Path,
                                         cfg: dict,
                                         deviation_data: dict | None = None,
                                         out_dir: Path | None = None) -> dict:
    """
    Non-OC_Mocap processing pipeline for one trial, saving standard filenames
    into `out_dir`.
    """
    trial_id = f"{subject_path.name}/{trial_path.name}"
    if out_dir is None:
        out_dir = trial_path / cfg.get("OUTPUT_SUBDIR_NAME", "ProcessedData")

    # ── Skip if already processed ───────────────────────────────
    if cfg["ONLY_PROCESS_NEW"] and (out_dir / "pos_inputs.npy").exists():
        # When OS_Filtering is also requested, only skip if the OSfilt outputs are already present
        if not cfg.get("OS_Filtering", False) or (out_dir / "ID_GT_MJX_OSfilt.npy").exists():
            return {"id": trial_id, "success": True, "skipped": True}

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        fs  = cfg["SAMPLING_RATE_HZ"]
        dt  = 1.0 / fs
        xml_path = resolve_subject_model_xml(subject_path, cfg)

        # ── 1. Load kinematics ────────────────────────────────────
        motion_dir = _resolve_trial_kinematics_dir(trial_path, cfg)

        _pos_path = motion_dir / _kinematics_input_filename("Pos.npy", cfg)
        pos   = _load_npy_numeric(_pos_path)
        # OpenCap-style inputs (e.g. --OpenCapVal) ship Pos only; derive Vel/Accel from
        # Pos on the kinematics timebase below when the files are absent.
        _vel_path   = motion_dir / _kinematics_input_filename("Vel.npy", cfg)
        _accel_path = motion_dir / _kinematics_input_filename("Accel.npy", cfg)
        _derive_kin = not (_vel_path.exists() and _accel_path.exists())
        vel   = None if _derive_kin else _load_npy_numeric(_vel_path)
        accel = None if _derive_kin else _load_npy_numeric(_accel_path)
        _grf_path = motion_dir / "GRF.npy"
        _grm_path = motion_dir / "GRM.npy"
        _cop_path = motion_dir / "COP.npy"
        _force_time_path = motion_dir / "Time.npy"
        grf   = _load_npy_numeric(_grf_path)
        grm   = _load_npy_numeric(_grm_path)
        cop   = _load_npy_numeric(_cop_path)
        force_time = _load_npy_numeric(_force_time_path)

        # GaitRetraining datasets store separate time vectors:
        #   Time_for_pos.npy -> kinematics (Pos/Vel/Accel)
        #   Time.npy         -> forces (GRF/GRM/COP)
        kin_time_path = motion_dir / "Time_for_pos.npy"
        if not kin_time_path.exists():
            # Fallback: some layouts keep Time_for_pos.npy in Motion root.
            kin_time_path = trial_path / "Motion" / "Time_for_pos.npy"
        if kin_time_path.exists():
            kin_time = _load_npy_numeric(kin_time_path)
        else:
            # Fallback for non-GaitRetraining layouts with a single time vector.
            kin_time = force_time
            kin_time_path = _force_time_path

        original_kin_time = np.asarray(kin_time).copy()
        original_force_time = np.asarray(force_time).copy()
        source_inputs = {
            "kinematics": {
                "position": _array_source_record(_pos_path, pos, trial_path),
                "velocity": (
                    {"derived_from_position": True}
                    if _derive_kin else _array_source_record(_vel_path, vel, trial_path)
                ),
                "acceleration": (
                    {"derived_from_velocity": True}
                    if _derive_kin else _array_source_record(_accel_path, accel, trial_path)
                ),
            },
            "forces": {
                "grf": _array_source_record(_grf_path, grf, trial_path),
                "moment": _array_source_record(_grm_path, grm, trial_path),
                "cop": _array_source_record(_cop_path, cop, trial_path),
            },
        }
        kin_time = _fit_time_to_length(kin_time, pos.shape[0])
        force_time = _fit_time_to_length(force_time, grf.shape[0])
        source_inputs["kinematics"]["time"] = _time_vector_record(
            kin_time_path, original_kin_time, kin_time, pos.shape[0], trial_path
        )
        source_inputs["forces"]["time"] = _time_vector_record(
            _force_time_path, original_force_time, force_time, grf.shape[0], trial_path
        )

        unit_context = f"{subject_path.name}/{trial_path.name}"
        if cfg.get("OPENCAPVAL_SOURCE"):
            unit_context += f"/{cfg.get('OPENCAPVAL_SOURCE')}"
        pos, vel, accel = normalize_kinematic_angle_units(
            pos, vel, accel, context=unit_context
        )

        # Derive Vel/Accel from Pos when not provided (OpenCap-style Pos-only inputs).
        if _derive_kin:
            pos = np.asarray(pos, dtype=np.float64)
            vel = np.gradient(pos, kin_time, axis=0)
            accel = np.gradient(vel, kin_time, axis=0)

        # ── 2. Resample to uniform 100 Hz ────────────────────────
        time_arr, pos, vel, accel, grf, grm, cop = resample_dataframes_to_uniform_timestep(
            kin_time, force_time, pos, vel, accel, grf, grm, cop, dt=dt
        )
        uniform_time_full = np.asarray(time_arr).copy()
        resampling_trace = {
            "method": "linear interpolation (scipy.interpolate.interp1d)",
            "target_grid_formula": "numpy.arange(max(first source times), min(last source times), dt)",
            "dt_s": float(dt),
            "sample_rate_hz": float(fs),
            "start_time_s": float(uniform_time_full[0]) if uniform_time_full.size else None,
            "last_time_s": float(uniform_time_full[-1]) if uniform_time_full.size else None,
            "end_time_exclusive_s": float(min(kin_time[-1], force_time[-1])),
            "frame_count": int(uniform_time_full.size),
            "kinematic_source_row_map": _linear_interpolation_map(kin_time, uniform_time_full),
            "force_source_row_map": _linear_interpolation_map(force_time, uniform_time_full),
        }

        # ── 3. Align pelvis yaw (align_myosuite_pelvis) ───────────
        pos, vel, accel, grf, grm, cop = align_myosuite_pelvis(
            pos, vel, accel, grf, grm, cop
        )
        grf_no_filt = grf.copy()

        # ── 4. Butterworth 6 Hz on kinematics (per-channel; defaults to global) ──
        pos, vel, accel = apply_kinematics_filtering(pos, vel, accel, cfg, fs)

        # ── 5. Segment-wise filter on GRF and GRM (OpenSim Y-up, idx 1 & 4) ──
        if bool(cfg.get("ENABLE_GRF_FILTERING", True)):
            for col in range(grf.shape[1]):
                foot_idx = 1 if col < 3 else 4   # right Y-up vertical=1, left=4
                grf[:, col] = filter_segment_wise(
                    grf[:, col], grf[:, foot_idx],
                    cutoff=cfg["FILTER_CUTOFF_HZ"], fs=fs,
                    order=int(cfg.get("FILTER_ORDER", 2)),
                )
            for col in range(grm.shape[1]):
                foot_idx = 1 if col < 3 else 4
                grm[:, col] = filter_segment_wise(
                    grm[:, col], grf[:, foot_idx],
                    cutoff=cfg["FILTER_CUTOFF_HZ"], fs=fs,
                    order=int(cfg.get("FILTER_ORDER", 2)),
                )
        else:
            print("    [GRF Filter] ENABLE_GRF_FILTERING=False - skipping segment-wise GRF/GRM filtering")

        short_stance_report: dict[str, Any] = {
            "n_flagged": 0, "n_frames_zeroed": 0, "stances": [],
        }
        if bool(cfg.get("ENABLE_SHORT_STANCE_ZEROING", False)):
            grf, grm, cop, short_stance_report = zero_short_grf_cop_stances(
                grf,
                grm,
                cop,
                contact_threshold_n=float(cfg.get("GRF_CONTACT_THRESHOLD", 1.0)),
                max_frames=int(cfg.get("SHORT_STANCE_MAX_FRAMES", 25)),
                min_peak_n=float(cfg.get("SHORT_STANCE_MIN_PEAK_N", 50.0)),
            )
            if short_stance_report["n_flagged"]:
                print(
                    "    [ShortStanceZero] "
                    f"zeroed {short_stance_report['n_flagged']} non-edge stance(s), "
                    f"{short_stance_report['n_frames_zeroed']} frame-foot samples"
                )

        # ── 6. Basic + misstep + double-support trim ──────────────
        meta        = extract_patient_metadata(subject_path)
        body_weight = meta["Mass_kg"] * 9.8067
        pretrim_len_motion_aligned = int(pos.shape[0])
        if bool(cfg.get("ENABLE_GRF_TRIM", True)):
            trim_idx, timeline_stages = _compute_trim_indices_by_grf_with_trace(
                grf,
                body_weight=body_weight,
                trim_grf_miss_steps=bool(cfg.get("TrimGRFMissSteps", True)),
                trim_to_double_support=bool(cfg.get("TRIM_TO_DOUBLE_SUPPORT", False)),
            )
            if trim_idx.size == 0:
                raise ValueError("GRF trim removed all frames")

            grf_trim_start = int(trim_idx[0])
            grf_trim_end = int(trim_idx[-1]) + 1
            pos = pos[trim_idx]
            vel = vel[trim_idx]
            accel = accel[trim_idx]
            grf = grf[trim_idx]
            grf_no_filt = grf_no_filt[trim_idx]
            grm = grm[trim_idx]
            cop = cop[trim_idx]
            time_arr = time_arr[trim_idx]
        else:
            grf_trim_start = 0
            grf_trim_end = pretrim_len_motion_aligned
            timeline_stages = [_trace_stage(
                "primary_grf_trim_disabled",
                pretrim_len_motion_aligned, 0, pretrim_len_motion_aligned, 0,
                enabled=False,
            )]
            print("    [GRF Trim] ENABLE_GRF_TRIM=False — skipping primary GRF trim")

        T = pos.shape[0]
        if T < 20:
            raise ValueError(f"Trial too short after GRF trim: {T} frames")
        t_after_grf_trim = int(T)

        # ── 7. Map to MuJoCo qpos / qvel / qacc ──────────────────
        mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        name_to_qpos = _build_name_to_qpos_index(mj_model)
        qpos_matrix = np.array([map_patient_to_qpos(pos[t], mj_model, name_to_qpos=name_to_qpos) for t in range(T)])
        qvel_matrix = np.array([map_patient_to_qpos(vel[t], mj_model, name_to_qpos=name_to_qpos) for t in range(T)])
        qacc_matrix = np.array([map_patient_to_qpos(accel[t], mj_model, name_to_qpos=name_to_qpos) for t in range(T)])

        # ── 8. Polynomial coupled coordinates ────────────────────
        if xml_path.exists():
            qpos_matrix, qvel_matrix, qacc_matrix = calculate_coupled_coordinates_automated(
                qpos_matrix, qvel_matrix, qacc_matrix, xml_path
            )

        # ── 9. Floor height from toe bodies ──────────────────────
        toes_r_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "toes_r")
        toes_l_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "toes_l")
        calcn_r_id_cpu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
        calcn_l_id_cpu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "calcn_l")
        tibia_r_id_cpu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tibia_r")
        tibia_l_id_cpu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tibia_l")

        mj_data = mujoco.MjData(mj_model)
        toes_z_r_all = np.zeros(T)
        toes_z_l_all = np.zeros(T)
        ankle_pos_r_all = np.zeros((T, 3))
        ankle_pos_l_all = np.zeros((T, 3))
        knee_pos_r_all = np.zeros((T, 3))
        knee_pos_l_all = np.zeros((T, 3))
        toes_pos_r_all  = np.zeros((T, 3))
        toes_pos_l_all  = np.zeros((T, 3))

        for t in range(T):
            mj_data.qpos[:] = qpos_matrix[t]
            mujoco.mj_forward(mj_model, mj_data)
            toes_z_r_all[t] = mj_data.xpos[toes_r_body_id, 2]
            toes_z_l_all[t] = mj_data.xpos[toes_l_body_id, 2]
            ankle_pos_r_all[t] = mj_data.xpos[calcn_r_id_cpu]
            ankle_pos_l_all[t] = mj_data.xpos[calcn_l_id_cpu]
            knee_pos_r_all[t] = mj_data.xpos[tibia_r_id_cpu]
            knee_pos_l_all[t] = mj_data.xpos[tibia_l_id_cpu]
            toes_pos_r_all[t]  = mj_data.xpos[toes_r_body_id]
            toes_pos_l_all[t]  = mj_data.xpos[toes_l_body_id]

        floor_height, trough_count = estimate_floor_height_from_toe_troughs(
            toes_z_r_all,
            toes_z_l_all,
            percentile=cfg["FLOOR_TROUGH_PERCENTILE"],
            offset_m=cfg["FLOOR_TROUGH_OFFSET_M"],
            min_troughs_for_direct_percentile=cfg["FLOOR_MIN_TROUGHS_FOR_DIRECT_PERCENTILE"],
            interp_samples=cfg["FLOOR_INTERP_SAMPLES"],
        )
        del toes_z_r_all, toes_z_l_all  # no longer needed

        # Apply floor correction to pelvis height in both the MuJoCo qpos vector
        # and the raw 23-col pos array (col 4 = pelvis_ty) so that pos_inputs
        # (built later from pos) also reflects the floor-corrected height.
        if bool(cfg.get("ENABLE_FLOOR_CORRECTION", True)):
            qpos_matrix[:, 1] -= floor_height   # qpos[1] = pelvis_ty (MuJoCo Z-up vertical)
            pos[:, 4] -= floor_height           # col 4 = pelvis_ty in 23-col layout
        else:
            print("    [Floor Correction] ENABLE_FLOOR_CORRECTION=False - leaving pelvis Z uncorrected")

        # ── 10. Treadmill detection ───────────────────────────────
        # Convert GRF to MuJoCo coords for treadmill speed check
        grf_r_mj = convert_to_mujoco_coords(grf[:, 0:3])
        grf_l_mj = convert_to_mujoco_coords(grf[:, 3:6])

        # --- Treadmill detection: dual criterion ---
        # Keep ankle X range for logging/metadata, but do not use it for
        # treadmill classification.
        ankle_x_range = (np.ptp(ankle_pos_r_all[:, 0]) + np.ptp(ankle_pos_l_all[:, 0])) / 2.0

        # Net pelvis forward speed below this threshold is treated as treadmill,
        # but only for trials at least 1 second long.
        trial_duration_s   = (T - 1) * dt
        pelvis_net_speed   = (abs(qpos_matrix[-1, 0] - qpos_matrix[0, 0])
                              / max(trial_duration_s, 1e-6))   # m/s

        # Trials shorter than 1 second are always marked non-treadmill.
        is_treadmill = (trial_duration_s >= 1.0 and pelvis_net_speed < 0.3)

        print(f"    [TreadmillDetect] ankle_x_range={ankle_x_range:.3f} m | "
              f"pelvis_net_speed={pelvis_net_speed:.3f} m/s | "
              f"is_treadmill={is_treadmill}")

        treadmill_speed = 0.0
        disp = np.zeros(T)   # initialise so COP block below is always valid

        if is_treadmill:
            # pelvis X position (qpos_matrix[:,0] = pelvis_tx in MuJoCo coords)
            pelvis_pos_for_speed = qpos_matrix[:, 0:3]   # (T, 3)

            treadmill_speed = calculate_treadmill_speed(
                ankle_pos_r_all, ankle_pos_l_all,
                grf_r_mj, grf_l_mj,
                pelvis_pos = pelvis_pos_for_speed,
                dt = dt,
            )
            print(f"    [Treadmill] detected – belt speed estimate: {treadmill_speed:.4f} m/s")
            forward_vel_val = treadmill_speed

            # Displacement correction applied BEFORE ID
            disp = treadmill_speed * (time_arr - time_arr[0])
            qpos_matrix[:, 0] += disp            # pelvis_tx position in MuJoCo qpos
            qvel_matrix[:, 0] += treadmill_speed # pelvis_tx velocity in MuJoCo qvel

            # Mirror into the 23-col kinematics arrays used for saved model inputs.
            # Layout: col 3 = pelvis_tx, col 4 = pelvis_ty, col 5 = pelvis_tz
            pos[:, 3] += disp
            vel[:, 3] += treadmill_speed
            # accel[:, 3] intentionally not corrected (matches qacc behavior)
        else:
            # For overground trials, calculate average forward velocity
            # based on pelvis translation over the trial duration.
            trial_duration_s = (T - 1) * dt
            if trial_duration_s > 0:
                forward_vel_val = abs(qpos_matrix[-1, 0] - qpos_matrix[0, 0]) / trial_duration_s
            else:
                forward_vel_val = 0.0
            print(f"    [Overground] forward velocity estimate: {forward_vel_val:.4f} m/s")

        # ── 6b. Weak-edge-stance trim (overground only) ───────────
        # Runs after treadmill detection so is_treadmill is known.
        # Operates on the OpenSim-coord grf (cols 1=right vert, 4=left vert).
        # Also trims the already-built qpos/qvel/qacc matrices and ankle arrays
        # so all arrays remain aligned.
        weak_trim_start = 0
        weak_trim_end = int(T)
        if bool(cfg.get("TRIM_WEAK_EDGE_STANCES", True)) and not is_treadmill:
            grf_weak, _, _, grm_weak, cop_weak, pos_weak, vel_weak, accel_weak, \
                qpos_weak, qvel_weak, qacc_weak, ankle_r_weak, ankle_l_weak, \
                time_weak = (
                grf, None, None, grm, cop, pos, vel, accel,
                qpos_matrix, qvel_matrix, qacc_matrix,
                ankle_pos_r_all, ankle_pos_l_all, time_arr,
            )

            # Reuse the trim logic by calling trim_data_by_grf with only step 4 active,
            # passing dummy arrays for the ones that don't need trimming here.
            # We need to trim: pos, vel, accel, grf, grm, cop, time_arr,
            #                  qpos_matrix, qvel_matrix, qacc_matrix,
            #                  ankle_pos_r_all, ankle_pos_l_all
            # Strategy: call the helper once to get (new_start, new_end), then apply
            # to all arrays manually.

            bw_frac_thresh = float(cfg.get("TRIM_WEAK_STANCE_BW_FRACTION", 0.65))
            min_frames  = int(cfg.get("TRIM_WEAK_STANCE_MIN_FRAMES", 5))
            ct          = float(cfg["GRF_CONTACT_THRESHOLD"])

            def _stances_col(vgrf_col):
                # Keep stance segmentation identical to visualization logic:
                # is_stance -> diff(prepend=0) -> starts/ends (+append trailing end if needed)
                is_stance = (vgrf_col > ct)
                s_diff    = np.diff(is_stance.astype(int), prepend=0)
                starts    = np.where(s_diff == 1)[0]
                ends      = np.where(s_diff == -1)[0]
                if len(ends) < len(starts):
                    ends = np.append(ends, len(is_stance))
                return list(zip(starts, ends))

            def _mean_bw_ratio(vgrf_col, s_idx, e_idx):
                if e_idx <= s_idx or body_weight <= 0:
                    return 0.0
                return float(np.mean(vgrf_col[s_idx:e_idx]) / body_weight)

            def _last_before_contact(vgrf_col, from_frame):
                in_s = (vgrf_col > ct).astype(int)
                pad  = np.concatenate(([0], in_s, [0]))
                rises = np.where(np.diff(pad) == 1)[0]
                future = rises[rises >= from_frame]
                return int(future[0]) if len(future) > 0 else None

            new_start = 0
            new_end   = T
            # Right=col1, Left=col4 in OpenSim Y-up
            for this_col, other_col in [(1, 4), (4, 1)]:
                stances = _stances_col(grf[:, this_col])
                if not stances:
                    continue

                # ── Leading: only consider the first stance if it is a
                #    *partial* stance (GRF was already active at frame 0,
                #    i.e. a misstep cut off by the start of the recording).
                #    Walk inward past any too-short stances; stop at the
                #    first qualifying one.
                for s0, e0 in stances:
                    if (e0 - s0) < min_frames:
                        continue   # too short to judge – skip past it
                    # Only act on a partial stance: one that begins at frame 0
                    bw_ratio = _mean_bw_ratio(grf[:, this_col], s0, e0)
                    if s0 == 0 and bw_ratio < bw_frac_thresh:
                        cut = _last_before_contact(grf[:, other_col], e0)
                        if cut is not None and cut > new_start:
                            new_start = cut
                            print(f"    [WeakEdgeTrim] Leading partial: foot col {this_col}, "
                                  f"mean vGRF/BW={bw_ratio:.2f} "
                                  f"< {bw_frac_thresh:.2f} → trim start to frame {new_start}")
                    break   # stop after the first qualifying stance regardless

                # ── Trailing: only consider the last stance if it is a
                #    *partial* stance (GRF was still active at frame T-1,
                #    i.e. a misstep cut off by the end of the recording).
                #    Walk inward past any too-short stances; stop at the
                #    first qualifying one.
                for sN, eN in reversed(stances):
                    if (eN - sN) < min_frames:
                        continue   # too short to judge – skip past it
                    # Only act on a partial stance: one that ends at frame T
                    bw_ratio = _mean_bw_ratio(grf[:, this_col], sN, eN)
                    if eN == T and bw_ratio < bw_frac_thresh:
                        other_stances = _stances_col(grf[:, other_col])
                        preceding = [(s, e) for s, e in other_stances if e <= sN]
                        if preceding:
                            cut = preceding[-1][1]
                            if cut < new_end:
                                new_end = cut
                                print(f"    [WeakEdgeTrim] Trailing partial: foot col {this_col}, "
                                      f"mean vGRF/BW={bw_ratio:.2f} "
                                      f"< {bw_frac_thresh:.2f} → trim end to frame {new_end}")
                    break   # stop after the last qualifying stance regardless

            weak_trim_start = int(new_start)
            weak_trim_end = int(new_end)
            if new_start > 0 or new_end < T:
                sl = slice(new_start, new_end)
                pos, vel, accel         = pos[sl],         vel[sl],         accel[sl]
                grf, grm, cop           = grf[sl],         grm[sl],         cop[sl]
                grf_no_filt             = grf_no_filt[sl]
                time_arr                = time_arr[sl]
                qpos_matrix             = qpos_matrix[sl]
                qvel_matrix             = qvel_matrix[sl]
                qacc_matrix             = qacc_matrix[sl]
                ankle_pos_r_all         = ankle_pos_r_all[sl]
                ankle_pos_l_all         = ankle_pos_l_all[sl]
                knee_pos_r_all          = knee_pos_r_all[sl]
                knee_pos_l_all          = knee_pos_l_all[sl]
                toes_pos_r_all          = toes_pos_r_all[sl]
                toes_pos_l_all          = toes_pos_l_all[sl]
                T = qpos_matrix.shape[0]
                if T < 10:
                    raise ValueError(
                        f"Trial too short after weak-edge-stance trim: {T} frames")
        t_after_weak_edge_trim = int(T)
        timeline_stages.append(_trace_stage(
            "trim_weak_edge_stances",
            t_after_grf_trim,
            weak_trim_start,
            weak_trim_end,
            grf_trim_start,
            enabled=bool(cfg.get("TRIM_WEAK_EDGE_STANCES", True)) and not is_treadmill,
            parameters={
                "overground_only": True,
                "body_weight_fraction_threshold": float(
                    cfg.get("TRIM_WEAK_STANCE_BW_FRACTION", 0.65)
                ),
                "minimum_stance_frames": int(cfg.get("TRIM_WEAK_STANCE_MIN_FRAMES", 5)),
                "contact_threshold_n": float(cfg["GRF_CONTACT_THRESHOLD"]),
            },
            details={"treadmill_trial": bool(is_treadmill)},
        ))

        # ── 11. Setup Jacobians + full MJX ID batch ───────────────
        if not xml_path.exists():
            raise FileNotFoundError(f"Model XML not found: {xml_path}")

        # Free temporary treadmill-detection GRF arrays
        del grf_r_mj, grf_l_mj

        # Re-run forward kinematics with corrected qpos to get updated ankle/toes positions
        for t in range(T):
            mj_data.qpos[:] = qpos_matrix[t]
            mujoco.mj_forward(mj_model, mj_data)
            ankle_pos_r_all[t] = mj_data.xpos[calcn_r_id_cpu]
            ankle_pos_l_all[t] = mj_data.xpos[calcn_l_id_cpu]
            toes_pos_r_all[t]  = mj_data.xpos[toes_r_body_id]
            toes_pos_l_all[t]  = mj_data.xpos[toes_l_body_id]

        mjx_model, jacobian_data, calcn_r_id, calcn_l_id, toes_r_id, toes_l_id = \
            setup_and_precompute_jacobians(mj_model, qpos_matrix)

        # MJX inverse dynamics in chunks to reduce peak RAM.
        qfrc_inverse_only, qfrc_constraint_only, com_global = compute_inverse_dynamics_chunked(
            mjx_model,
            qpos_matrix,
            qvel_matrix,
            qacc_matrix,
            chunk_size=cfg["ID_BATCH_CHUNK_SIZE"],
        )
        qfrc_inverse_batch = qfrc_inverse_only + qfrc_constraint_only
        # Free the two component arrays immediately — only the sum is needed later
        del qfrc_inverse_only, qfrc_constraint_only

        # ── 12. Floor-corrected ankle positions ───────────────────
        ankle_pos_r_corr = ankle_pos_r_all.copy()
        ankle_pos_l_corr = ankle_pos_l_all.copy()
        if bool(cfg.get("ENABLE_FLOOR_CORRECTION", True)):
            ankle_pos_r_corr[:, 2] -= floor_height
            ankle_pos_l_corr[:, 2] -= floor_height

        # ── 13. COM relative to floor-corrected ankle ─────────────
        COM_r = com_global - ankle_pos_r_corr
        COM_l = com_global - ankle_pos_l_corr

        # ── 14. COM acceleration (double gradient + filter) ───────
        com_vel = np.gradient(com_global, dt, axis=0)
        com_vel = butter_lowpass_filter(com_vel, cfg["FILTER_CUTOFF_HZ"], fs, order=int(cfg.get("FILTER_ORDER", 2)))
        com_acc = np.gradient(com_vel, dt, axis=0)
        com_acc = butter_lowpass_filter(com_acc, cfg["FILTER_CUTOFF_HZ"], fs, order=int(cfg.get("FILTER_ORDER", 2)))
        COM_Acc_Global = com_acc

        # ── 15. GRF coordinate conversion: OpenSim → MuJoCo ──────
        GRF_mj = np.hstack([
            convert_to_mujoco_coords(grf[:, 0:3]),
            convert_to_mujoco_coords(grf[:, 3:6]),
        ])
        GRF_NoFilt_mj = np.hstack([
            convert_to_mujoco_coords(grf_no_filt[:, 0:3]),
            convert_to_mujoco_coords(grf_no_filt[:, 3:6]),
        ])
        Moment_mj = np.hstack([
            convert_to_mujoco_coords(grm[:, 0:3]),
            convert_to_mujoco_coords(grm[:, 3:6]),
        ])
        COP_mj = np.hstack([
            convert_to_mujoco_coords(cop[:, 0:3]),
            convert_to_mujoco_coords(cop[:, 3:6]),
        ])  # (T, 6)

        # Treadmill COP X correction (already in MuJoCo coords now)
        if is_treadmill:
            COP_mj[:, 0] += disp   # Right X
            COP_mj[:, 3] += disp   # Left  X

        # ── 16. GaitRetraining direction check ────────────────────
        was_negated = {"right": False, "left": False}
        if "GaitRetraining" in str(subject_path):
            right_stance_mask = GRF_mj[:, 2] > cfg["GRF_CONTACT_THRESHOLD"]
            left_stance_mask  = GRF_mj[:, 5] > cfg["GRF_CONTACT_THRESHOLD"]

            if np.any(right_stance_mask):
                right_cop_r = COP_mj[right_stance_mask, 0]
                if len(right_cop_r) > 1:
                    right_slope = np.mean(np.diff(right_cop_r))
                    if right_slope < -1e-5:
                        COP_mj[:, 0:2]  *= -1.0
                        GRF_mj[:, 0:3]   = GRF_mj[:, 0:3] * np.array([-1, -1, 1])
                        GRF_NoFilt_mj[:, 0:3] = GRF_NoFilt_mj[:, 0:3] * np.array([-1, -1, 1])
                        was_negated["right"] = True

            if np.any(left_stance_mask):
                left_cop_x = COP_mj[left_stance_mask, 3]
                if len(left_cop_x) > 1:
                    left_slope = np.mean(np.diff(left_cop_x))
                    if left_slope < -1e-5:
                        COP_mj[:, 3:5]  *= -1.0
                        GRF_mj[:, 3:6]   = GRF_mj[:, 3:6] * np.array([-1, -1, 1])
                        GRF_NoFilt_mj[:, 3:6] = GRF_NoFilt_mj[:, 3:6] * np.array([-1, -1, 1])
                        was_negated["left"] = True

        # ── 17. Outlier stance removal ────────────────────────────
        outlier_flagged_stances: list[dict[str, Any]] = []
        outlier_input_count = int(T)
        if bool(cfg.get("ENABLE_OUTLIER_STANCE_REMOVAL", True)):
            valid_mask = np.ones(T, dtype=bool)
            for grf_idx, min_stances in [(2, cfg["MIN_STANCES_FOR_CHECK"]),
                                          (5, cfg["MIN_STANCES_FOR_CHECK"])]:
                stances = get_stance_phases(GRF_mj, grf_idx, threshold=cfg["GRF_STANCE_THRESHOLD"])
                if len(stances) >= min_stances:
                    durations = [s["duration_frames"] for s in stances]
                    med = np.median(durations)
                    for s in stances:
                        if (s["duration_frames"] > 2 * med and
                                s["duration_frames"] > cfg["MIN_STANCE_LENGTH_FOR_FLAG"]):
                            valid_mask[s["start"]:s["end"]] = False
                            outlier_flagged_stances.append({
                                "foot": "right" if grf_idx == 2 else "left",
                                "vertical_grf_column_mujoco": int(grf_idx),
                                "start": int(s["start"]),
                                "end": int(s["end"]),
                                "duration_frames": int(s["duration_frames"]),
                                "median_stance_duration_frames": float(med),
                            })

            si, ei = find_longest_valid_segment(valid_mask)
            # Trim all arrays to the valid segment
            pos               = pos[si:ei]
            vel               = vel[si:ei]
            accel             = accel[si:ei]
            qpos_matrix       = qpos_matrix[si:ei]
            qvel_matrix       = qvel_matrix[si:ei]
            qacc_matrix       = qacc_matrix[si:ei]
            GRF_mj            = GRF_mj[si:ei]
            GRF_NoFilt_mj     = GRF_NoFilt_mj[si:ei]
            Moment_mj         = Moment_mj[si:ei]
            COP_mj            = COP_mj[si:ei]
            ankle_pos_r_corr  = ankle_pos_r_corr[si:ei]
            ankle_pos_l_corr  = ankle_pos_l_corr[si:ei]
            knee_pos_r_all    = knee_pos_r_all[si:ei]
            knee_pos_l_all    = knee_pos_l_all[si:ei]
            toes_pos_r_all    = toes_pos_r_all[si:ei]
            toes_pos_l_all    = toes_pos_l_all[si:ei]
            COM_r             = COM_r[si:ei]
            COM_l             = COM_l[si:ei]
            com_global        = com_global[si:ei]
            COM_Acc_Global    = COM_Acc_Global[si:ei]
            qfrc_inverse_batch = qfrc_inverse_batch[si:ei]
            jacobian_data = {
                "jacp":     jacobian_data["jacp"][si:ei],
                "jacr":     jacobian_data["jacr"][si:ei],
                "body_ids": jacobian_data["body_ids"],
            }
            time_arr = time_arr[si:ei] if len(time_arr) > ei else time_arr
            T = qpos_matrix.shape[0]
        else:
            si = 0
            ei = T
            print("    [Outlier Stance] ENABLE_OUTLIER_STANCE_REMOVAL=False — skipping tertiary outlier trim")

        if T < 10:
            raise ValueError(f"Trial too short after outlier-stance removal: {T} frames")

        core_trim_start = int(grf_trim_start + weak_trim_start + si)
        core_trim_end = int(grf_trim_start + weak_trim_start + ei)
        timeline_stages.append(_trace_stage(
            "trim_outlier_stances",
            outlier_input_count,
            si,
            ei,
            grf_trim_start + weak_trim_start,
            enabled=bool(cfg.get("ENABLE_OUTLIER_STANCE_REMOVAL", True)),
            parameters={
                "stance_threshold_n": float(cfg["GRF_STANCE_THRESHOLD"]),
                "minimum_stances_for_check": int(cfg["MIN_STANCES_FOR_CHECK"]),
                "duration_threshold": "duration > 2 * median_duration",
                "minimum_duration_for_flag_frames": int(cfg["MIN_STANCE_LENGTH_FOR_FLAG"]),
                "selection_after_flagging": "longest_contiguous_valid_segment",
            },
            details={"flagged_stances": outlier_flagged_stances},
        ))

        # ── 18. COP relative to ankle ─────────────────────────────
        # COP_mj is (T,6): [Rx,Ry,Rz, Lx,Ly,Lz]
        r_vec_r = COP_mj[:, 0:3] - ankle_pos_r_corr   # (T,3)
        r_vec_l = COP_mj[:, 3:6] - ankle_pos_l_corr   # (T,3)

        # Store only X, Y
        COP_Cleaned_Relative = np.column_stack([
            r_vec_r[:, 0], r_vec_r[:, 1],
            r_vec_l[:, 0], r_vec_l[:, 1],
        ])   # (T, 4)

        # ── 19. Clean and filter COP ─────────────────────────────
        if bool(cfg.get("ENABLE_COP_CLEANING", True)):
            # Mask where GRF is below contact threshold
            mask_r = GRF_mj[:, 2] < cfg["GRF_CONTACT_THRESHOLD"]
            mask_l = GRF_mj[:, 5] < cfg["GRF_CONTACT_THRESHOLD"]
            r_vec_r[mask_r] = 0.0
            r_vec_l[mask_l] = 0.0
            COP_Cleaned_Relative = np.column_stack([
                r_vec_r[:, 0], r_vec_r[:, 1],
                r_vec_l[:, 0], r_vec_l[:, 1],
            ])
            COP_Cleaned_Relative = clean_and_filter_cop(
                COP_Cleaned_Relative,
                GRF_mj,
                trim_start_frames = cfg["COP_TRIM_START_FRAMES"],
                trim_end_frames   = cfg["COP_TRIM_END_FRAMES"],
                extrapolation_frames = int(cfg.get("COP_EXTRAPOLATION_FRAMES", 6)),
                pad_width         = cfg["COP_FILTER_PAD_WIDTH"],
                edge_hold         = bool(cfg.get("COP_EdgeHold", False)),
                cutoff            = cfg["FILTER_CUTOFF_HZ"],
                fs                = fs,
                order             = int(cfg.get("FILTER_ORDER", 2)),
            )

            # Re-mask after COP cleaning
            COP_Cleaned_Relative[mask_r, 0:2] = 0.0
            COP_Cleaned_Relative[mask_l, 2:4] = 0.0

        KneeToCOP_Vectors, _, _ = _compute_knee_to_cop_vectors(
            COP_Cleaned_Relative,
            ankle_pos_r_corr,
            ankle_pos_l_corr,
            knee_pos_r_all,
            knee_pos_l_all,
        )

        # Propagate filtered COP back to r_vec for moment computation
        r_vec_r[:, 0] = COP_Cleaned_Relative[:, 0]
        r_vec_r[:, 1] = COP_Cleaned_Relative[:, 1]
        r_vec_l[:, 0] = COP_Cleaned_Relative[:, 2]
        r_vec_l[:, 1] = COP_Cleaned_Relative[:, 3]

        # ── 20. GRF contribution → ID_GT_MJX ─────────────────────
        torque_grf_mj = GRF_NoFilt_mj if bool(cfg.get("USE_NOFILTER_GRF_FOR_TORQUE", False)) else GRF_mj
        grf_r_np = torque_grf_mj[:, 0:3]
        grf_l_np = torque_grf_mj[:, 3:6]
        mom_r_np = Moment_mj[:, 0:3]
        mom_l_np = Moment_mj[:, 3:6]

        mom_added_r = np.cross(r_vec_r, grf_r_np)   # (T,3)
        mom_added_l = np.cross(r_vec_l, grf_l_np)

        ext_force = np.zeros((T, 2, 6))
        ext_force[:, 0, 0:3] = grf_r_np
        ext_force[:, 0, 3:6] = mom_r_np + mom_added_r
        ext_force[:, 1, 0:3] = grf_l_np
        ext_force[:, 1, 3:6] = mom_l_np + mom_added_l

        qfrc_grf = compute_grf_contribution(
            jacobian_data["jacp"],
            jacobian_data["jacr"],
            ext_force[:, :, 0:3],
            ext_force[:, :, 3:6],
        )
        ID_GT_MJX = qfrc_inverse_batch - qfrc_grf

        # ── 20b. OpenSim-filtering variant (optional) ────────────
        # Replace the kinematic velocity/acceleration with OpenSim-style GCVSpline
        # derivatives (the technique OpenSim's ID uses) and recompute everything that
        # depends on them: the MJX ID (reusing the identical qfrc_grf, so the only
        # difference vs the standard ID_GT_MJX is the filtering method) AND the model
        # input features. Saved alongside the standard files with a _OSfilt suffix.
        id_gt_mjx_osfilt = qvel_mjx_osfilt = qacc_mjx_osfilt = qfrc_inverse_osfilt = None
        vel_inputs_osfilt = acc_inputs_osfilt = None
        if bool(cfg.get("OS_Filtering", False)):
            # ID side: GCVSpline derivatives of the 31-col MJX qpos.
            qvel_mjx_osfilt, qacc_mjx_osfilt = gcv_derivatives(qpos_matrix, dt)
            qfrc_inv_os, qfrc_con_os, _ = compute_inverse_dynamics_chunked(
                mjx_model, qpos_matrix, qvel_mjx_osfilt, qacc_mjx_osfilt,
                chunk_size=cfg["ID_BATCH_CHUNK_SIZE"],
            )
            qfrc_inverse_osfilt = qfrc_inv_os + qfrc_con_os
            del qfrc_inv_os, qfrc_con_os
            id_gt_mjx_osfilt = qfrc_inverse_osfilt - qfrc_grf
            # Input side: GCVSpline vel/accel of the 23-col coordinates -> input features
            # (positions are unchanged, so pos_inputs has no OS-filtered variant).
            vel_os23, acc_os23 = gcv_derivatives(pos, dt)
            vel_inputs_osfilt = build_vel_inputs_without_mtp(vel_os23)
            acc_inputs_osfilt = build_acc_inputs_without_mtp(acc_os23)

        # ── 21. Pelvis rotation matrix (T, 6) ─────────────────────
        # After align_myosuite_pelvis, r_new.as_euler("ZXY") is stored back into
        # data[:,0:3] as [Z_angle, X_angle, Y_angle].  map_patient_to_qpos then
        # places those at qpos[3], qpos[4], qpos[5] respectively.
        # TreadmillSpeedAdjust_PelvisRotation.py (authoritative) builds:
        #   Rz from qpos[:,3], Rx from qpos[:,4], Ry from qpos[:,5]  → R = Rz @ Rx @ Ry
        # which is exactly R.from_euler("ZXY", qpos[:,3:6]).
        pelvis_euler = qpos_matrix[:, 3:6]   # (T, 3)  [Z_ang, X_ang, Y_ang]
        R_batch      = R.from_euler("ZXY", pelvis_euler)   # vectorised, no loop
        R_mats       = R_batch.as_matrix()                 # (T, 3, 3)
        pelvis_rot_matrix = np.concatenate(
            [R_mats[:, :, 0], R_mats[:, :, 1]], axis=1    # (T, 6): col-0 + col-1
        )

        # ── 22. pos/vel/acc inputs (retain knees; remove only MTP, plus pelvis XYZ from pos) ─
        # Slice from the original 23-col motion arrays (pos/vel/accel), NOT from the
        # saved model-space qpos_matrix. The 23-col motion layout is:
        #   0-2  : pelvis tilt/list/rotation  → keep
        #   3-5  : pelvis tx/ty/tz            → REMOVE (not useful as features)
        #   6-8  : hip_r (flex/add/rot)       → keep
        #   9    : knee_angle_r                → KEEP
        #   10-15: ankle/subtalar/mtp_r + hip_l → keep, then drop mtp_r
        #   16   : knee_angle_l                → KEEP
        #   17-22: ankle_l .. lumbar           → keep, then drop mtp_l
        # Traceable schemas are declared in POS_INPUT_COLUMN_NAMES and
        # VEL_ACC_INPUT_COLUMN_NAMES below.
        # Result: pos_inputs (T,18), vel_inputs (T,21), acc_inputs (T,21)
        pos_inputs = build_pos_inputs_without_mtp(pos)
        vel_inputs = build_vel_inputs_without_mtp(vel)
        acc_inputs = build_acc_inputs_without_mtp(accel)

        # ── 23. Contact boolean ───────────────────────────────────
        contact_bool = create_contact_boolean(GRF_mj, cfg["GRF_CONTACT_THRESHOLD"])

        # ── 24. Ankle heights (floor-corrected Z) ─────────────────
        ankle_heights = np.column_stack([
            ankle_pos_r_corr[:, 2],
            ankle_pos_l_corr[:, 2],
        ])

        # ── 25. Metadata arrays ───────────────────────────────────
        Height_arr  = np.full(T, meta["Height_m"])
        Mass_arr    = np.full(T, meta["Mass_kg"])
        Forward_Vel = np.full(T, forward_vel_val)

        # ── 26. Compute bad percentage (for quality info) ─────────
        total_frames_raw = pos.shape[0] if hasattr(pos, "shape") else T
        bad_pct = 0.0
        try:
            vgrf_r = GRF_mj[:, 2]
            vgrf_l = GRF_mj[:, 5]
            above  = (np.sum(vgrf_r > 1.0) + np.sum(vgrf_l > 1.0))
            bad_pct = 100.0 * (1.0 - above / (2 * T))
        except Exception:
            pass

        # ── 27. Save all outputs ──────────────────────────────────
        save_dof_idx = save_dof_indices_for_model(mj_model, cfg)
        jac_dof_idx = jacobian_save_dof_indices_for_model(mj_model)
        id_gt_save = ID_GT_MJX[:, save_dof_idx]
        qfrc_inverse_save = qfrc_inverse_batch[:, save_dof_idx]
        qfrc_grf_save = qfrc_grf[:, save_dof_idx]
        jac_save = slice_jacobian_dofs(jacobian_data, jac_dof_idx)

        np.save(out_dir / "pos_inputs.npy",          pos_inputs)
        np.save(out_dir / "vel_inputs.npy",          vel_inputs)
        np.save(out_dir / "acc_inputs.npy",          acc_inputs)
        np.save(out_dir / "pelvis_rot_matrix.npy",   pelvis_rot_matrix)
        np.save(out_dir / "pos_mjx.npy",             qpos_matrix)
        np.save(out_dir / "qvel_mjx.npy",            qvel_matrix)
        np.save(out_dir / "qacc_mjx.npy",            qacc_matrix)
        np.save(out_dir / "GRF_Cleaned.npy",         GRF_mj)
        np.save(out_dir / "GRF_NoFilt_Trimmed.npy",  GRF_NoFilt_mj)
        np.save(out_dir / "Moment_Cleaned.npy",      Moment_mj)
        np.save(out_dir / "COP_Cleaned_Relative.npy",COP_Cleaned_Relative)
        np.save(out_dir / "KneeToCOP_Vectors.npy",   KneeToCOP_Vectors)
        np.save(out_dir / "forwardVel.npy",          Forward_Vel)
        np.save(out_dir / "ankle_heights.npy",       ankle_heights)
        np.save(out_dir / "ankle_pos_r.npy",         ankle_pos_r_corr)
        np.save(out_dir / "ankle_pos_l.npy",         ankle_pos_l_corr)
        np.save(out_dir / "knee_pos_r.npy",          knee_pos_r_all)
        np.save(out_dir / "knee_pos_l.npy",          knee_pos_l_all)
        np.save(out_dir / "toes_pos_r.npy",          toes_pos_r_all)
        np.save(out_dir / "toes_pos_l.npy",          toes_pos_l_all)
        np.save(out_dir / "contactBoolean.npy",      contact_bool)
        np.save(out_dir / "COM_r.npy",               COM_r)
        np.save(out_dir / "COM_l.npy",               COM_l)
        np.save(out_dir / "COM_Acc_Global.npy",      COM_Acc_Global)
        np.save(out_dir / "ID_GT_MJX.npy",           id_gt_save)
        np.save(out_dir / "qfrc_inverse.npy",        qfrc_inverse_save)
        np.save(out_dir / "qfrc_grf_contribution.npy", qfrc_grf_save)
        np.save(out_dir / "Height_m.npy",            Height_arr)
        np.save(out_dir / "Mass_kg.npy",             Mass_arr)

        # OpenSim-filtering variant (optional): saved alongside the standard files with a
        # _OSfilt suffix so training can opt into a self-consistent OS-filtered dataset.
        if id_gt_mjx_osfilt is not None:
            np.save(out_dir / "ID_GT_MJX_OSfilt.npy",      id_gt_mjx_osfilt[:, save_dof_idx])
            np.save(out_dir / "qfrc_inverse_OSfilt.npy",   qfrc_inverse_osfilt[:, save_dof_idx])
            np.save(out_dir / "qvel_mjx_OSfilt.npy",       qvel_mjx_osfilt)
            np.save(out_dir / "qacc_mjx_OSfilt.npy",       qacc_mjx_osfilt)
            np.save(out_dir / "vel_inputs_OSfilt.npy",     vel_inputs_osfilt)
            np.save(out_dir / "acc_inputs_OSfilt.npy",     acc_inputs_osfilt)

        # Jacobian: save as pickle (contains jnp arrays)
        np.save(out_dir / "Jacobian.npy", jac_save, allow_pickle=True)

        # ── 28. Trial info JSON ───────────────────────────────────
        proc_info = {
            "bad_percentage":    round(float(bad_pct), 2),
            "treadmill_flag":    bool(is_treadmill),
            "treadmill_speed":   round(float(treadmill_speed), 4),
            "was_negated":       was_negated,
            "n_frames":          int(T),
            "n_frames_after_grf_trim": int(t_after_grf_trim),
            "n_frames_after_weak_edge_trim": int(t_after_weak_edge_trim),
            "core_trim_reference_space": "motion_aligned_post_filter",
            "core_trim_pretrim_n_frames": int(pretrim_len_motion_aligned),
            "grf_trim_bounds_motion_aligned": [int(grf_trim_start), int(grf_trim_end)],
            "weak_edge_trim_bounds_after_grf": [int(weak_trim_start), int(weak_trim_end)],
            "outlier_trim_bounds_after_weak_edge": [int(si), int(ei)],
            "core_trim_bounds_motion_aligned": [int(core_trim_start), int(core_trim_end)],
            "floor_height_m":    round(float(floor_height), 6),
            "floor_height_method": "toe_trough_10th_percentile_plus_offset",
            "floor_trough_count": int(trough_count),
            "enable_floor_correction": bool(cfg.get("ENABLE_FLOOR_CORRECTION", True)),
            "enable_kinematics_filtering": bool(cfg.get("ENABLE_KINEMATICS_FILTERING", True)),
            "enable_grf_filtering": bool(cfg.get("ENABLE_GRF_FILTERING", True)),
            "use_nofilter_grf_for_torque": bool(cfg.get("USE_NOFILTER_GRF_FOR_TORQUE", False)),
            "grf_torque_source": "GRF_NoFilt_Trimmed.npy" if bool(cfg.get("USE_NOFILTER_GRF_FOR_TORQUE", False)) else "GRF_Cleaned.npy",
            "enable_cop_cleaning": bool(cfg.get("ENABLE_COP_CLEANING", True)),
            "cop_edge_hold": bool(cfg.get("COP_EdgeHold", False)),
            "subject_model_xml": str(resolve_subject_model_xml(subject_path, cfg)),
            "os_filtering": bool(cfg.get("OS_Filtering", False)),
            "processing_date":   datetime.now().isoformat(),
        }
        with open(out_dir / "Trial_Processing_Information.json", "w") as f:
            json.dump(proc_info, f, indent=4)

        # ── 29. Deviation learning reconstruction (optional) ──────
        if cfg["DO_DEVIATION_LEARNING_PREP"] and deviation_data is not None:
            try:
                final_output        = deviation_data["final_output"]
                global_median_dur   = deviation_data["global_median_duration"]
                global_median_fpa   = deviation_data["global_median_fpa"]

                recon_grf, recon_cop, recon_moment = build_average_reconstructions(
                    GRF_mj, COP_Cleaned_Relative, Moment_mj,
                    final_output, global_median_dur, global_median_fpa,
                    meta["Mass_kg"], meta["Height_m"],
                    ankle_pos_r=ankle_pos_r_corr,
                    toes_pos_r=toes_pos_r_all,
                    ankle_pos_l=ankle_pos_l_corr,
                    toes_pos_l=toes_pos_l_all,
                )
                np.save(out_dir / "GRF_average_reconstructed.npy",    recon_grf)
                np.save(out_dir / "COP_average_reconstructed.npy",    recon_cop)
                np.save(out_dir / "Moment_average_reconstructed.npy", recon_moment)
            except Exception as dev_e:
                warnings.warn(f"[{trial_id}] Deviation learning prep failed: {dev_e}")

        trace_payload = {
            "schema_name": "ProcessData trimming traceability",
            "schema_version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "trial_id": trial_id,
            "pipeline": "ProcessData.py",
            "interval_convention": (
                "All frame bounds are zero-based, half-open [start, end); "
                "end is excluded."
            ),
            "source_inputs": source_inputs,
            "model_dependency": _file_identity_record(xml_path, subject_path),
            "uniform_resampling": resampling_trace,
            "timeline_stages": timeline_stages,
            "signal_mutations_without_timeline_removal": {
                "short_stance_zeroing": {
                    "enabled": bool(cfg.get("ENABLE_SHORT_STANCE_ZEROING", False)),
                    "parameters": {
                        "contact_threshold_n": float(cfg.get("GRF_CONTACT_THRESHOLD", 1.0)),
                        "maximum_frames": int(cfg.get("SHORT_STANCE_MAX_FRAMES", 25)),
                        "minimum_peak_n": float(cfg.get("SHORT_STANCE_MIN_PEAK_N", 50.0)),
                    },
                    "report_in_uniform_resampled_frame_space": short_stance_report,
                },
                "cop_cleaning": {
                    "enabled": bool(cfg.get("ENABLE_COP_CLEANING", True)),
                    "parameters": {
                        "contact_threshold_n": float(cfg["GRF_CONTACT_THRESHOLD"]),
                        "trim_start_frames_per_stance": int(cfg["COP_TRIM_START_FRAMES"]),
                        "trim_end_frames_per_stance": int(cfg["COP_TRIM_END_FRAMES"]),
                        "extrapolation_frames": int(cfg.get("COP_EXTRAPOLATION_FRAMES", 6)),
                        "filter_pad_width": int(cfg["COP_FILTER_PAD_WIDTH"]),
                        "edge_hold": bool(cfg.get("COP_EdgeHold", False)),
                        "filter_cutoff_hz": float(cfg["FILTER_CUTOFF_HZ"]),
                    },
                    "note": (
                        "These operations modify COP values within stance but do not remove "
                        "timeline frames."
                    ),
                },
            },
            "final_mapping": {
                "uniform_resampled_frame_bounds": [int(core_trim_start), int(core_trim_end)],
                "final_frame_count": int(T),
                "final_first_time_s": float(time_arr[0]) if len(time_arr) else None,
                "final_last_time_s": float(time_arr[-1]) if len(time_arr) else None,
                "mapping_formula": (
                    f"final_frame[j] corresponds to "
                    f"uniform_resampled_frame[{core_trim_start} + j]"
                ),
                "source_mapping_note": (
                    "Use the uniform frame index above with kinematic_source_row_map "
                    "or force_source_row_map to recover the contributing raw data rows."
                ),
            },
            "output_files": _output_manifest(out_dir, T),
            "postprocessing_history": [],
        }
        _write_json_atomic(out_dir / TRIMMING_TRACE_FILENAME, trace_payload)

        result_dict = {"id": trial_id, "success": True, "n_frames": int(T),
                       "treadmill": is_treadmill}

        # ── 30. Explicit memory cleanup ───────────────────────────
        # Release large arrays that are no longer needed so that forked worker
        # processes return memory to the OS between trials.
        del (qpos_matrix, qvel_matrix, qacc_matrix,
             qfrc_inverse_batch,
             qfrc_grf, jacobian_data, jac_save,
             GRF_mj, Moment_mj, COP_mj, COP_Cleaned_Relative,
             pos_inputs, vel_inputs, acc_inputs,
             pos, vel, accel,
             ankle_pos_r_corr, ankle_pos_l_corr, ankle_pos_r_all, ankle_pos_l_all,
             knee_pos_r_all, knee_pos_l_all,
             toes_pos_r_all, toes_pos_l_all,
             com_global, COM_r, COM_l, COM_Acc_Global, com_vel, com_acc,
             ID_GT_MJX, pelvis_rot_matrix)
        del mj_data, mj_model, mjx_model
        gc.collect()

        return result_dict

    except Exception as exc:
        import traceback
        gc.collect()
        return {"id": trial_id, "success": False, "error": str(exc),
                "traceback": traceback.format_exc()}


def process_single_trial(subject_path: Path, trial_path: Path,
                         cfg: dict,
                         deviation_data: dict | None = None) -> dict:
    """
    Full processing pipeline for one trial.
    Returns a result dict with keys: id, success, error (if any).
    """
    # --OpenCapVal: run the core pipeline twice per trial — once on MoCap/Motion
    # (MoCap model) into MoCap/ProcessedData, once on Video/Motion (Video model)
    # into Video/ProcessedData.
    if bool(cfg.get("OpenCapVal", False)):
        trial_id = f"{subject_path.name}/{trial_path.name}"
        results: dict[str, dict] = {}
        for source in ("MoCap", "Video"):
            pass_cfg = dict(cfg)
            pass_cfg["OpenCapVal"] = False            # prevent recursion
            pass_cfg["OPENCAPVAL_SOURCE"] = source
            out_dir = trial_path / source / "ProcessedData"
            results[source] = _process_single_trial_processed_core(
                subject_path, trial_path, pass_cfg, deviation_data, out_dir=out_dir
            )
        overall_ok = all(bool(r.get("success")) for r in results.values())
        combined = {
            "id": trial_id,
            "success": overall_ok,
            "skipped": all(bool(r.get("skipped")) for r in results.values()) if overall_ok else False,
            "mocap_result": results["MoCap"],
            "video_result": results["Video"],
            "n_frames_mocap": results["MoCap"].get("n_frames"),
            "n_frames_video": results["Video"].get("n_frames"),
        }
        if not overall_ok:
            combined["error"] = " | ".join(
                f"{s}: {r.get('error', '?')}" for s, r in results.items() if not r.get("success")
            )
        return combined

    if bool(cfg.get("OC_Mocap", False)):
        trial_id = f"{subject_path.name}/{trial_path.name}"

        processed_cfg = dict(cfg)
        processed_cfg["OC_Mocap"] = False
        processed_result = process_single_trial(
            subject_path,
            trial_path,
            processed_cfg,
            deviation_data,
        )

        mocap_result = process_single_trial_oc_mocap(
            subject_path,
            trial_path,
            cfg,
            deviation_data,
        )

        combined_success = bool(processed_result.get("success")) and bool(mocap_result.get("success"))
        combined_skipped = bool(processed_result.get("skipped")) and bool(mocap_result.get("skipped"))
        combined_result = {
            "id": trial_id,
            "success": combined_success,
            "skipped": combined_skipped if combined_success else False,
            "processed_result": processed_result,
            "mocap_result": mocap_result,
            "n_frames_processed": processed_result.get("n_frames"),
            "n_frames_mocap": mocap_result.get("n_frames"),
        }

        if mocap_result.get("n_frames") is not None:
            combined_result["n_frames"] = mocap_result.get("n_frames")
        elif processed_result.get("n_frames") is not None:
            combined_result["n_frames"] = processed_result.get("n_frames")

        if combined_success:
            if processed_result.get("used_noised_prediction_bundle"):
                combined_result["used_noised_prediction_bundle"] = True
            return combined_result

        error_parts = []
        if not processed_result.get("success"):
            error_parts.append(
                f"ProcessedData failed: {processed_result.get('error', '?')}"
            )
        if not mocap_result.get("success"):
            error_parts.append(
                f"MoCap failed: {mocap_result.get('error', '?')}"
            )
        combined_result["error"] = " | ".join(error_parts) if error_parts else "combined OC_Mocap processing failed"
        return combined_result

    trial_id = f"{subject_path.name}/{trial_path.name}"
    out_dir = trial_path / cfg.get("OUTPUT_SUBDIR_NAME", "ProcessedData")
    use_noised = bool(cfg.get("UseNoised", False))
    only_process_noised = bool(cfg.get("OnlyProcessNoised", False))

    if only_process_noised:
        if not use_noised:
            return {
                "id": trial_id,
                "success": False,
                "error": "OnlyProcessNoised requires UseNoised=True",
            }
        if cfg["ONLY_PROCESS_NEW"] and _has_noised_prediction_bundle(out_dir):
            return {"id": trial_id, "success": True, "skipped": True}
        if not _has_noised_source_inputs(trial_path):
            return {
                "id": trial_id,
                "success": False,
                "error": (
                    "OnlyProcessNoised requested but one or more noised kinematics inputs are missing "
                    "(expected Pos_noised.npy, Vel_noised.npy, Accel_noised.npy in Motion/ or Motion/Motion_Pelvis_Adjusted/)"
                ),
            }

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = trial_path / "_ProcessedData_NoisedTmp"
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

            noised_cfg = dict(cfg)
            noised_cfg["UseNoised"] = True
            noised_cfg["OnlyProcessNoised"] = False
            noised_cfg["ONLY_PROCESS_NEW"] = False
            noised_cfg["DO_DEVIATION_LEARNING_PREP"] = False
            noised_result = _process_single_trial_processed_core(
                subject_path,
                trial_path,
                noised_cfg,
                deviation_data=None,
                out_dir=tmp_dir,
            )
            if not noised_result.get("success"):
                return {
                    "id": trial_id,
                    "success": False,
                    "error": f"noised bundle failed: {noised_result.get('error', '?')}",
                    "traceback": noised_result.get("traceback"),
                }

            xml_path = resolve_subject_model_xml(subject_path, cfg)
            ok, message = generate_calc_frame_outputs_for_source(tmp_dir, xml_path, f"{trial_id} [noised-only]")
            if not ok:
                return {
                    "id": trial_id,
                    "success": False,
                    "error": f"noised calc-frame generation failed: {message}",
                }

            _copy_outputs_with_suffix(tmp_dir, out_dir, NOISED_AUX_FILES_TO_COPY, NOISED_FILE_SUFFIX)
            missing_noised = _missing_noised_bundle_files(
                out_dir,
                filenames=NOISED_STRICT_VALIDATION_FILENAMES,
            )
            if missing_noised:
                return {
                    "id": trial_id,
                    "success": False,
                    "error": (
                        "noised bundle is incomplete after copy: "
                        + ", ".join(sorted(missing_noised))
                    ),
                }
            _update_trial_info_json(
                out_dir,
                {
                    "use_noised_prediction_bundle": True,
                    "only_process_noised": True,
                    "prediction_bundle_suffix": NOISED_FILE_SUFFIX,
                    "prediction_bundle_source": "Pos_noised.npy / Vel_noised.npy / Accel_noised.npy",
                    "noised_bundle_required_files": list(NOISED_REQUIRED_BUNDLE_FILENAMES),
                },
            )

            return {
                "id": trial_id,
                "success": True,
                "n_frames": noised_result.get("n_frames"),
                "used_noised_prediction_bundle": True,
                "only_process_noised": True,
            }
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if cfg["ONLY_PROCESS_NEW"] and (out_dir / "pos_inputs.npy").exists():
        osfilt_done = not cfg.get("OS_Filtering", False) or (out_dir / "ID_GT_MJX_OSfilt.npy").exists()
        if osfilt_done and ((not use_noised) or _has_noised_prediction_bundle(out_dir)):
            return {"id": trial_id, "success": True, "skipped": True}

    clean_cfg = dict(cfg)
    clean_cfg["UseNoised"] = False
    clean_cfg["ONLY_PROCESS_NEW"] = False
    clean_result = _process_single_trial_processed_core(
        subject_path,
        trial_path,
        clean_cfg,
        deviation_data,
        out_dir=out_dir,
    )
    if not clean_result.get("success") or not use_noised:
        return clean_result

    if not _has_noised_source_inputs(trial_path):
        return {
            "id": trial_id,
            "success": False,
            "error": (
                "UseNoised requested but one or more noised kinematics inputs are missing "
                "(expected Pos_noised.npy, Vel_noised.npy, Accel_noised.npy in Motion/ or Motion/Motion_Pelvis_Adjusted/)"
            ),
        }

    tmp_dir = trial_path / "_ProcessedData_NoisedTmp"
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        noised_cfg = dict(cfg)
        noised_cfg["UseNoised"] = True
        noised_cfg["ONLY_PROCESS_NEW"] = False
        noised_cfg["DO_DEVIATION_LEARNING_PREP"] = False
        noised_result = _process_single_trial_processed_core(
            subject_path,
            trial_path,
            noised_cfg,
            deviation_data=None,
            out_dir=tmp_dir,
        )
        if not noised_result.get("success"):
            return {
                "id": trial_id,
                "success": False,
                "error": f"clean bundle succeeded but noised bundle failed: {noised_result.get('error', '?')}",
                "traceback": noised_result.get("traceback"),
            }

        xml_path = resolve_subject_model_xml(subject_path, cfg)
        ok, message = generate_calc_frame_outputs_for_source(tmp_dir, xml_path, f"{trial_id} [noised]")
        if not ok:
            return {
                "id": trial_id,
                "success": False,
                "error": f"clean bundle succeeded but noised calc-frame generation failed: {message}",
            }

        _copy_outputs_with_suffix(tmp_dir, out_dir, NOISED_AUX_FILES_TO_COPY, NOISED_FILE_SUFFIX)
        missing_noised = _missing_noised_bundle_files(
            out_dir,
            filenames=NOISED_STRICT_VALIDATION_FILENAMES,
        )
        if missing_noised:
            return {
                "id": trial_id,
                "success": False,
                "error": (
                    "clean bundle succeeded but noised bundle is incomplete after copy: "
                    + ", ".join(sorted(missing_noised))
                ),
            }
        _update_trial_info_json(
            out_dir,
            {
                "use_noised_prediction_bundle": True,
                "prediction_bundle_suffix": NOISED_FILE_SUFFIX,
                "prediction_bundle_source": "Pos_noised.npy / Vel_noised.npy / Accel_noised.npy",
                "noised_bundle_required_files": list(NOISED_REQUIRED_BUNDLE_FILENAMES),
            },
        )
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    clean_result = clean_result.copy()
    clean_result["used_noised_prediction_bundle"] = True
    return clean_result


def process_single_trial_oc_mocap(subject_path: Path, trial_path: Path,
                                  cfg: dict,
                                  deviation_data: dict | None = None) -> dict:
    """
    MoCap-specific processing path for ProcessData.

    Inputs:
      - kinematics from Trial/MoCap
      - forces from Trial/Motion
    Outputs:
      - trimmed processed arrays in Trial/MoCap
      - optional pre-trim snapshot in Trial/MoCap/UntrimmedRaw
    """
    trial_id = f"{subject_path.name}/{trial_path.name}"
    mocap_dir = trial_path / "MoCap"
    mocap_source_dir = _resolve_trial_kinematics_dir(trial_path, cfg)
    mocap_raw_timebase_dir = trial_path / OC_MOCAP_RAW_TIMEBASE_DIRNAME
    motion_dir = trial_path / "Motion"
    out_dir = mocap_dir

    if cfg["ONLY_PROCESS_NEW"] and (out_dir / "pos_inputs.npy").exists():
        return {"id": trial_id, "success": True, "skipped": True}

    if not mocap_dir.exists():
        return {"id": trial_id, "success": False, "error": "MoCap directory not found"}
    if not motion_dir.exists():
        return {"id": trial_id, "success": False, "error": "Motion directory not found"}

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        fs = cfg["SAMPLING_RATE_HZ"]
        dt = 1.0 / fs
        xml_path = resolve_subject_model_xml(subject_path, cfg)

        def _load_angle_2ch(base_dir: Path, primary: str,
                            fallback: str | None = None,
                            n_rows: int = 0) -> np.ndarray:
            arr = None
            p1 = base_dir / primary
            if p1.exists():
                arr = np.load(p1)
            elif fallback:
                p2 = base_dir / fallback
                if p2.exists():
                    arr = np.load(p2)
            if arr is None:
                return np.zeros((n_rows, 2), dtype=np.float32)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if arr.shape[1] < 2:
                arr = np.tile(arr, (1, 2))[:, :2]
            elif arr.shape[1] > 2:
                arr = arr[:, :2]
            return arr

        def _resolve_angle_series_to_target(
            primary: str,
            fallback: str | None,
            *,
            mocap_time: np.ndarray,
            motion_time: np.ndarray,
            target_name: str,
            processed_dir: Path,
        ) -> np.ndarray:
            """
            Resolve a 2-channel derived angle series onto the requested target
            timebase. On reruns, MoCap/ may already contain previously-trimmed
            outputs, so these files are not guaranteed to live on the raw MoCap
            timeline.
            """
            mocap_time = np.asarray(mocap_time, dtype=np.float64).reshape(-1)
            motion_time = np.asarray(motion_time, dtype=np.float64).reshape(-1)
            target_name = str(target_name).strip().lower()
            if target_name not in {"motion", "mocap"}:
                raise ValueError(f"Unsupported target timebase '{target_name}'")

            target_time = motion_time if target_name == "motion" else mocap_time
            alt_time = mocap_time if target_name == "motion" else motion_time
            target_label = "motion" if target_name == "motion" else "mocap"
            alt_label = "mocap" if target_name == "motion" else "motion"

            arr = _load_angle_2ch(mocap_dir, primary, fallback, n_rows=0)
            if arr.size == 0:
                candidate_paths = [mocap_dir / primary]
                if fallback:
                    candidate_paths.append(mocap_dir / fallback)
                if processed_dir.is_dir():
                    candidate_paths.append(processed_dir / primary)
                    if fallback:
                        candidate_paths.append(processed_dir / fallback)

                for candidate in candidate_paths:
                    if not candidate.exists():
                        continue
                    cand = np.asarray(np.load(candidate), dtype=np.float32)
                    if cand.ndim == 1:
                        cand = cand[:, np.newaxis]
                    if cand.shape[1] < 2:
                        cand = np.tile(cand, (1, 2))[:, :2]
                    elif cand.shape[1] > 2:
                        cand = cand[:, :2]
                    arr = cand
                    break

            if arr.size == 0:
                return np.zeros((len(target_time), 2), dtype=np.float32)

            arr_len = int(arr.shape[0])
            if arr_len == len(target_time):
                return np.asarray(arr, dtype=np.float32)
            if arr_len == len(alt_time):
                return np.asarray(_resample(arr, alt_time, target_time), dtype=np.float32)

            for candidate_dir, candidate_label, candidate_time in [
                (processed_dir, "ProcessedData", motion_time),
                (mocap_dir, "MoCap", target_time),
            ]:
                if not candidate_dir.is_dir():
                    continue
                for candidate_name in ([primary] + ([fallback] if fallback else [])):
                    candidate_path = candidate_dir / candidate_name
                    if not candidate_path.exists():
                        continue
                    cand = np.asarray(np.load(candidate_path), dtype=np.float32)
                    if cand.ndim == 1:
                        cand = cand[:, np.newaxis]
                    if cand.shape[1] < 2:
                        cand = np.tile(cand, (1, 2))[:, :2]
                    elif cand.shape[1] > 2:
                        cand = cand[:, :2]

                    cand_len = int(cand.shape[0])
                    if cand_len == len(target_time):
                        print(
                            f"    [AngleTimebase] Using {candidate_label}/{candidate_name} "
                            f"directly on {target_label} timeline ({cand_len} frames)"
                        )
                        return cand
                    if cand_len == len(candidate_time):
                        print(
                            f"    [AngleTimebase] Resampling {candidate_label}/{candidate_name} "
                            f"from {candidate_label.lower()} timeline ({cand_len} frames) "
                            f"to {target_label} timeline ({len(target_time)} frames)"
                        )
                        return np.asarray(_resample(cand, candidate_time, target_time), dtype=np.float32)

            warnings.warn(
                f"[AngleTimebase] {primary} length {arr_len} matched neither the {target_label} "
                f"timeline ({len(target_time)}) nor the {alt_label} timeline ({len(alt_time)}). "
                "Falling back to linear index-based resizing."
            )
            if arr_len <= 1 or len(target_time) == 0:
                base_val = arr[:1] if arr_len > 0 else np.zeros((1, 2), dtype=np.float32)
                return np.repeat(base_val.astype(np.float32, copy=False), len(target_time), axis=0)

            src_idx = np.linspace(0.0, 1.0, arr_len, dtype=np.float64)
            tgt_idx = np.linspace(0.0, 1.0, len(target_time), dtype=np.float64)
            return np.asarray(_resample(arr, src_idx, tgt_idx), dtype=np.float32)

        def _resample(data: np.ndarray, src_t: np.ndarray, tgt_t: np.ndarray) -> np.ndarray:
            f = interp1d(src_t, data, axis=0, kind="linear",
                         fill_value="extrapolate", bounds_error=False)
            return f(tgt_t)

        def _project_series_to_timebase(
            data: np.ndarray,
            src_t: np.ndarray,
            tgt_t: np.ndarray,
            *,
            kind: str = "linear",
        ) -> np.ndarray:
            arr = np.asarray(data)
            if arr.ndim == 0:
                return np.array(arr, copy=True)
            if arr.shape[0] != len(src_t):
                return np.array(arr, copy=True)

            src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
            tgt_t = np.asarray(tgt_t, dtype=np.float64).reshape(-1)
            if src_t.shape == tgt_t.shape and np.allclose(src_t, tgt_t):
                return np.array(arr, copy=True)

            interp_input = arr.astype(np.float32, copy=False) if arr.dtype == bool else arr
            f = interp1d(
                src_t,
                interp_input,
                axis=0,
                kind=kind,
                fill_value="extrapolate",
                bounds_error=False,
            )
            out = np.asarray(f(tgt_t))
            if arr.dtype == bool:
                return out > 0.5
            return out

        def _select_trimmed_raw_timebase(src_t: np.ndarray, raw_t: np.ndarray) -> np.ndarray:
            """Pick the raw MoCap time samples that overlap the final trimmed window."""
            src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
            raw_t = np.asarray(raw_t, dtype=np.float64).reshape(-1)
            if src_t.size == 0 or raw_t.size == 0:
                return raw_t[:0]

            eps = max(float(dt) * 0.5, 1e-8)
            mask = (raw_t >= (src_t[0] - eps)) & (raw_t <= (src_t[-1] + eps))
            if np.any(mask):
                return raw_t[mask]

            start_idx = int(np.argmin(np.abs(raw_t - src_t[0])))
            end_idx = int(np.argmin(np.abs(raw_t - src_t[-1])))
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            return raw_t[start_idx:end_idx + 1]

        def _write_snapshot_dir(
            snapshot_dir: Path,
            *,
            time_vec: np.ndarray,
            pos_arr: np.ndarray,
            vel_arr: np.ndarray,
            accel_arr: np.ndarray,
            extra_save_map: dict[str, np.ndarray],
            jac_payload: dict[str, np.ndarray | list[int]],
            info_payload: dict,
        ) -> None:
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            snapshot_map = {
                "Time.npy": np.asarray(time_vec, dtype=np.float32),
                "Pos.npy": np.asarray(pos_arr, dtype=np.float32),
                "Vel.npy": np.asarray(vel_arr, dtype=np.float32),
                "Accel.npy": np.asarray(accel_arr, dtype=np.float32),
            }
            snapshot_map.update(extra_save_map)

            for name, arr in snapshot_map.items():
                np.save(snapshot_dir / name, np.asarray(arr, dtype=np.float32))

            np.save(snapshot_dir / "Jacobian.npy", jac_payload, allow_pickle=True)
            with open(snapshot_dir / "Trial_Processing_Information.json", "w") as f:
                json.dump(info_payload, f, indent=4)

        if mocap_source_dir != mocap_dir:
            print(
                f"    [MoCapSource] Using {mocap_source_dir.relative_to(trial_path)} "
                f"as the kinematics/time source"
            )

        pos = _load_npy_numeric(mocap_source_dir / _kinematics_input_filename("Pos.npy", cfg))
        vel = _load_npy_numeric(mocap_source_dir / _kinematics_input_filename("Vel.npy", cfg))
        accel = _load_npy_numeric(mocap_source_dir / _kinematics_input_filename("Accel.npy", cfg))
        pos, vel, accel = normalize_kinematic_angle_units(
            pos, vel, accel, context=f"{subject_path.name}/{trial_path.name}/OC_Mocap"
        )
        t_mocap_raw = _load_npy_numeric(mocap_source_dir / "Time.npy").reshape(-1)
        grf_raw = _load_npy_numeric(motion_dir / "GRF.npy")
        grm_raw = _load_npy_numeric(motion_dir / "GRM.npy")
        cop_raw = _load_npy_numeric(motion_dir / "COP.npy")
        t_motion = _load_npy_numeric(motion_dir / "Time.npy").reshape(-1)
        time_alignment_target = str(cfg.get("TIME_ALIGNMENT_TARGET", "motion")).strip().lower()
        if time_alignment_target not in {"motion", "mocap"}:
            warnings.warn(
                f"Unknown TIME_ALIGNMENT_TARGET='{time_alignment_target}'. Falling back to 'motion'."
            )
            time_alignment_target = "motion"
        processed_dir = trial_path / "ProcessedData"
        foot_progression_angle = _resolve_angle_series_to_target(
            "FootProgressionAngle.npy",
            "Foot_ProgressionAngle.npy",
            mocap_time=t_mocap_raw,
            motion_time=t_motion,
            target_name=time_alignment_target,
            processed_dir=processed_dir,
        )
        calcn_to_floor_angle = _resolve_angle_series_to_target(
            "CalcnToFloor_AngleDeg.npy",
            None,
            mocap_time=t_mocap_raw,
            motion_time=t_motion,
            target_name=time_alignment_target,
            processed_dir=processed_dir,
        )
        n_frames_source_mocap = int(len(t_mocap_raw))
        n_frames_source_motion = int(len(t_motion))

        if time_alignment_target == "motion":
            pos = _resample(pos, t_mocap_raw, t_motion)
            vel = _resample(vel, t_mocap_raw, t_motion)
            accel = _resample(accel, t_mocap_raw, t_motion)
            grf = grf_raw.copy()
            grm = grm_raw.copy()
            cop = cop_raw.copy()
            time_arr = t_motion.copy()
            resampled_kinematics_to_motion = True
            resampled_forces_to_mocap = False
        else:
            grf = _resample(grf_raw, t_motion, t_mocap_raw)
            grm = _resample(grm_raw, t_motion, t_mocap_raw)
            cop = _resample(cop_raw, t_motion, t_mocap_raw)
            time_arr = t_mocap_raw.copy()
            resampled_kinematics_to_motion = False
            resampled_forces_to_mocap = True

        n_work = int(len(time_arr))
        if not (
            pos.shape[0] == vel.shape[0] == accel.shape[0] ==
            grf.shape[0] == grm.shape[0] == cop.shape[0] ==
            foot_progression_angle.shape[0] == calcn_to_floor_angle.shape[0] == n_work
        ):
            raise ValueError(
                "Time-aligned arrays have inconsistent lengths: "
                f"pos={pos.shape[0]}, vel={vel.shape[0]}, accel={accel.shape[0]}, "
                f"grf={grf.shape[0]}, grm={grm.shape[0]}, cop={cop.shape[0]}, "
                f"fpa={foot_progression_angle.shape[0]}, cfa={calcn_to_floor_angle.shape[0]}, "
                f"time={n_work}"
            )

        pos, vel, accel, grf, grm, cop = align_myosuite_pelvis(
            pos, vel, accel, grf, grm, cop
        )
        grf_no_filt = grf.copy()

        pos, vel, accel = apply_kinematics_filtering(pos, vel, accel, cfg, fs)

        if bool(cfg.get("ENABLE_GRF_FILTERING", True)):
            for col in range(grf.shape[1]):
                foot_idx = 1 if col < 3 else 4
                grf[:, col] = filter_segment_wise(
                    grf[:, col], grf[:, foot_idx],
                    cutoff=cfg["FILTER_CUTOFF_HZ"], fs=fs,
                    order=int(cfg.get("FILTER_ORDER", 2)),
                )
            for col in range(grm.shape[1]):
                foot_idx = 1 if col < 3 else 4
                grm[:, col] = filter_segment_wise(
                    grm[:, col], grf[:, foot_idx],
                    cutoff=cfg["FILTER_CUTOFF_HZ"], fs=fs,
                    order=int(cfg.get("FILTER_ORDER", 2)),
                )
        else:
            print("    [GRF Filter] ENABLE_GRF_FILTERING=False - skipping segment-wise GRF/GRM filtering")

        if bool(cfg.get("ENABLE_SHORT_STANCE_ZEROING", False)):
            grf, grm, cop, short_stance_report = zero_short_grf_cop_stances(
                grf,
                grm,
                cop,
                contact_threshold_n=float(cfg.get("GRF_CONTACT_THRESHOLD", 1.0)),
                max_frames=int(cfg.get("SHORT_STANCE_MAX_FRAMES", 25)),
                min_peak_n=float(cfg.get("SHORT_STANCE_MIN_PEAK_N", 50.0)),
            )
            if short_stance_report["n_flagged"]:
                print(
                    "    [ShortStanceZero] "
                    f"zeroed {short_stance_report['n_flagged']} non-edge stance(s), "
                    f"{short_stance_report['n_frames_zeroed']} frame-foot samples"
                )

        meta = extract_patient_metadata(subject_path)
        body_weight = meta["Mass_kg"] * 9.8067
        t_len = int(pos.shape[0])
        if t_len < 20:
            raise ValueError(f"Trial too short before ID: {t_len} frames")

        mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        save_dof_idx = save_dof_indices_for_model(mj_model, cfg)
        jac_dof_idx = jacobian_save_dof_indices_for_model(mj_model)
        name_to_qpos = _build_name_to_qpos_index(mj_model)
        qpos_matrix = np.array([map_patient_to_qpos(pos[t], mj_model, name_to_qpos=name_to_qpos) for t in range(t_len)])
        qvel_matrix = np.array([map_patient_to_qpos(vel[t], mj_model, name_to_qpos=name_to_qpos) for t in range(t_len)])
        qacc_matrix = np.array([map_patient_to_qpos(accel[t], mj_model, name_to_qpos=name_to_qpos) for t in range(t_len)])

        qpos_matrix, qvel_matrix, qacc_matrix = calculate_coupled_coordinates_automated(
            qpos_matrix, qvel_matrix, qacc_matrix, xml_path
        )

        toes_r_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "toes_r")
        toes_l_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "toes_l")
        calcn_r_id_cpu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
        calcn_l_id_cpu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "calcn_l")
        tibia_r_id_cpu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tibia_r")
        tibia_l_id_cpu = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tibia_l")

        mj_data = mujoco.MjData(mj_model)
        toes_z_r_all = np.zeros(t_len)
        toes_z_l_all = np.zeros(t_len)
        ankle_pos_r_all = np.zeros((t_len, 3))
        ankle_pos_l_all = np.zeros((t_len, 3))
        knee_pos_r_all = np.zeros((t_len, 3))
        knee_pos_l_all = np.zeros((t_len, 3))
        toes_pos_r_all = np.zeros((t_len, 3))
        toes_pos_l_all = np.zeros((t_len, 3))

        for t in range(t_len):
            mj_data.qpos[:] = qpos_matrix[t]
            mujoco.mj_forward(mj_model, mj_data)
            toes_z_r_all[t] = mj_data.xpos[toes_r_body_id, 2]
            toes_z_l_all[t] = mj_data.xpos[toes_l_body_id, 2]
            ankle_pos_r_all[t] = mj_data.xpos[calcn_r_id_cpu]
            ankle_pos_l_all[t] = mj_data.xpos[calcn_l_id_cpu]
            knee_pos_r_all[t] = mj_data.xpos[tibia_r_id_cpu]
            knee_pos_l_all[t] = mj_data.xpos[tibia_l_id_cpu]
            toes_pos_r_all[t] = mj_data.xpos[toes_r_body_id]
            toes_pos_l_all[t] = mj_data.xpos[toes_l_body_id]

        floor_height, trough_count = estimate_floor_height_from_toe_troughs(
            toes_z_r_all,
            toes_z_l_all,
            percentile=cfg["FLOOR_TROUGH_PERCENTILE"],
            offset_m=(cfg["FLOOR_TROUGH_OFFSET_M"]
                      if bool(cfg.get("ENABLE_FLOOR_TROUGH_OFFSET", True)) else 0.0),
            min_troughs_for_direct_percentile=cfg["FLOOR_MIN_TROUGHS_FOR_DIRECT_PERCENTILE"],
            interp_samples=cfg["FLOOR_INTERP_SAMPLES"],
        )
        del toes_z_r_all, toes_z_l_all

        if bool(cfg.get("ENABLE_FLOOR_CORRECTION", True)):
            qpos_matrix[:, 1] -= floor_height
            pos[:, 4] -= floor_height
        else:
            print("    [Floor Correction] ENABLE_FLOOR_CORRECTION=False - leaving pelvis Z uncorrected")

        grf_r_mj = convert_to_mujoco_coords(grf[:, 0:3])
        grf_l_mj = convert_to_mujoco_coords(grf[:, 3:6])
        is_treadmill, ankle_x_range, pelvis_net_speed = detect_treadmill_like(
            ankle_pos_r_all, ankle_pos_l_all, qpos_matrix, dt
        )

        treadmill_speed = 0.0
        disp = np.zeros(t_len)
        if is_treadmill:
            treadmill_speed = calculate_treadmill_speed(
                ankle_pos_r_all, ankle_pos_l_all,
                grf_r_mj, grf_l_mj,
                pelvis_pos=qpos_matrix[:, 0:3],
                dt=dt,
            )
            disp = treadmill_speed * (time_arr - time_arr[0])
            qpos_matrix[:, 0] += disp
            qvel_matrix[:, 0] += treadmill_speed
            pos[:, 3] += disp
            vel[:, 3] += treadmill_speed
            forward_vel_val = treadmill_speed
        else:
            trial_duration_s = (t_len - 1) * dt
            forward_vel_val = (
                abs(qpos_matrix[-1, 0] - qpos_matrix[0, 0]) / trial_duration_s
                if trial_duration_s > 0 else 0.0
            )

        for t in range(t_len):
            mj_data.qpos[:] = qpos_matrix[t]
            mujoco.mj_forward(mj_model, mj_data)
            ankle_pos_r_all[t] = mj_data.xpos[calcn_r_id_cpu]
            ankle_pos_l_all[t] = mj_data.xpos[calcn_l_id_cpu]
            toes_pos_r_all[t] = mj_data.xpos[toes_r_body_id]
            toes_pos_l_all[t] = mj_data.xpos[toes_l_body_id]

        mjx_model, jacobian_data, _, _, _, _ = setup_and_precompute_jacobians(mj_model, qpos_matrix)

        qfrc_inverse_only, qfrc_constraint_only, com_global = compute_inverse_dynamics_chunked(
            mjx_model,
            qpos_matrix,
            qvel_matrix,
            qacc_matrix,
            chunk_size=cfg["ID_BATCH_CHUNK_SIZE"],
        )
        qfrc_inverse_batch = qfrc_inverse_only + qfrc_constraint_only
        del qfrc_inverse_only, qfrc_constraint_only, grf_r_mj, grf_l_mj

        ankle_pos_r_corr = ankle_pos_r_all.copy()
        ankle_pos_l_corr = ankle_pos_l_all.copy()
        if bool(cfg.get("ENABLE_FLOOR_CORRECTION", True)):
            ankle_pos_r_corr[:, 2] -= floor_height
            ankle_pos_l_corr[:, 2] -= floor_height

        COM_r = com_global - ankle_pos_r_corr
        COM_l = com_global - ankle_pos_l_corr

        com_vel = np.gradient(com_global, dt, axis=0)
        com_vel = butter_lowpass_filter(com_vel, cfg["FILTER_CUTOFF_HZ"], fs, order=int(cfg.get("FILTER_ORDER", 2)))
        com_acc = np.gradient(com_vel, dt, axis=0)
        com_acc = butter_lowpass_filter(com_acc, cfg["FILTER_CUTOFF_HZ"], fs, order=int(cfg.get("FILTER_ORDER", 2)))
        COM_Acc_Global = com_acc

        GRF_mj = np.hstack([
            convert_to_mujoco_coords(grf[:, 0:3]),
            convert_to_mujoco_coords(grf[:, 3:6]),
        ])
        GRF_NoFilt_mj = np.hstack([
            convert_to_mujoco_coords(grf_no_filt[:, 0:3]),
            convert_to_mujoco_coords(grf_no_filt[:, 3:6]),
        ])
        Moment_mj = np.hstack([
            convert_to_mujoco_coords(grm[:, 0:3]),
            convert_to_mujoco_coords(grm[:, 3:6]),
        ])
        COP_mj = np.hstack([
            convert_to_mujoco_coords(cop[:, 0:3]),
            convert_to_mujoco_coords(cop[:, 3:6]),
        ])
        if is_treadmill:
            COP_mj[:, 0] += disp
            COP_mj[:, 3] += disp

        was_negated = {"right": False, "left": False}
        if "GaitRetraining" in str(subject_path):
            right_stance_mask = GRF_mj[:, 2] > cfg["GRF_CONTACT_THRESHOLD"]
            left_stance_mask = GRF_mj[:, 5] > cfg["GRF_CONTACT_THRESHOLD"]

            if np.any(right_stance_mask):
                right_cop_x = COP_mj[right_stance_mask, 0]
                if len(right_cop_x) > 1 and np.mean(np.diff(right_cop_x)) < -1e-5:
                    COP_mj[:, 0:2] *= -1.0
                    GRF_mj[:, 0:3] = GRF_mj[:, 0:3] * np.array([-1, -1, 1])
                    GRF_NoFilt_mj[:, 0:3] = GRF_NoFilt_mj[:, 0:3] * np.array([-1, -1, 1])
                    was_negated["right"] = True

            if np.any(left_stance_mask):
                left_cop_x = COP_mj[left_stance_mask, 3]
                if len(left_cop_x) > 1 and np.mean(np.diff(left_cop_x)) < -1e-5:
                    COP_mj[:, 3:5] *= -1.0
                    GRF_mj[:, 3:6] = GRF_mj[:, 3:6] * np.array([-1, -1, 1])
                    GRF_NoFilt_mj[:, 3:6] = GRF_NoFilt_mj[:, 3:6] * np.array([-1, -1, 1])
                    was_negated["left"] = True

        t_untrimmed = int(qpos_matrix.shape[0])
        cop_rel_untrimmed, qfrc_grf_untrimmed, ID_GT_MJX_untrimmed = compute_cop_clean_and_id(
            GRF_mj,
            Moment_mj,
            COP_mj,
            ankle_pos_r_corr,
            ankle_pos_l_corr,
            jacobian_data,
            qfrc_inverse_batch,
            cfg,
            fs,
            GRF_for_torque_mj=GRF_NoFilt_mj if bool(cfg.get("USE_NOFILTER_GRF_FOR_TORQUE", False)) else None,
        )
        KneeToCOP_Vectors_untrimmed, _, _ = _compute_knee_to_cop_vectors(
            cop_rel_untrimmed,
            ankle_pos_r_corr,
            ankle_pos_l_corr,
            knee_pos_r_all,
            knee_pos_l_all,
        )

        pelvis_euler_untrimmed = qpos_matrix[:, 3:6]
        R_batch_untrimmed = R.from_euler("ZXY", pelvis_euler_untrimmed)
        R_mats_untrimmed = R_batch_untrimmed.as_matrix()
        pelvis_rot_matrix_untrimmed = np.concatenate(
            [R_mats_untrimmed[:, :, 0], R_mats_untrimmed[:, :, 1]], axis=1
        )

        pos_inputs_untrimmed = build_pos_inputs_without_mtp(pos)
        vel_inputs_untrimmed = build_vel_inputs_without_mtp(vel)
        acc_inputs_untrimmed = build_acc_inputs_without_mtp(accel)
        contact_bool_untrimmed = create_contact_boolean(GRF_mj, cfg["GRF_CONTACT_THRESHOLD"])
        ankle_heights_untrimmed = np.column_stack([ankle_pos_r_corr[:, 2], ankle_pos_l_corr[:, 2]])
        Height_arr_untrimmed = np.full(t_untrimmed, meta["Height_m"])
        Mass_arr_untrimmed = np.full(t_untrimmed, meta["Mass_kg"])
        Forward_Vel_untrimmed = np.full(t_untrimmed, forward_vel_val)

        if bool(cfg.get("SaveUntrimmedOutputs", True)):
            untrimmed_dir = mocap_dir / "UntrimmedRaw"
            pos_untrimmed_raw = _project_series_to_timebase(pos, time_arr, t_mocap_raw)
            vel_untrimmed_raw = _project_series_to_timebase(vel, time_arr, t_mocap_raw)
            accel_untrimmed_raw = _project_series_to_timebase(accel, time_arr, t_mocap_raw)
            qpos_untrimmed_raw = _project_series_to_timebase(qpos_matrix, time_arr, t_mocap_raw)
            qvel_untrimmed_raw = _project_series_to_timebase(qvel_matrix, time_arr, t_mocap_raw)
            qacc_untrimmed_raw = _project_series_to_timebase(qacc_matrix, time_arr, t_mocap_raw)
            qfrc_inverse_untrimmed_raw = _project_series_to_timebase(qfrc_inverse_batch[:, save_dof_idx], time_arr, t_mocap_raw)
            qfrc_grf_untrimmed_raw = _project_series_to_timebase(qfrc_grf_untrimmed[:, save_dof_idx], time_arr, t_mocap_raw)
            id_gt_untrimmed_raw = _project_series_to_timebase(ID_GT_MJX_untrimmed[:, save_dof_idx], time_arr, t_mocap_raw)
            pelvis_rot_untrimmed_raw = _project_series_to_timebase(
                pelvis_rot_matrix_untrimmed, time_arr, t_mocap_raw
            )
            grf_untrimmed_raw = _project_series_to_timebase(GRF_mj, time_arr, t_mocap_raw)
            grf_nofilt_untrimmed_raw = _project_series_to_timebase(GRF_NoFilt_mj, time_arr, t_mocap_raw)
            moment_untrimmed_raw = _project_series_to_timebase(Moment_mj, time_arr, t_mocap_raw)
            cop_rel_untrimmed_raw = _project_series_to_timebase(cop_rel_untrimmed, time_arr, t_mocap_raw)
            ankle_heights_untrimmed_raw = _project_series_to_timebase(
                ankle_heights_untrimmed, time_arr, t_mocap_raw
            )
            ankle_pos_r_untrimmed_raw = _project_series_to_timebase(ankle_pos_r_corr, time_arr, t_mocap_raw)
            ankle_pos_l_untrimmed_raw = _project_series_to_timebase(ankle_pos_l_corr, time_arr, t_mocap_raw)
            toes_pos_r_untrimmed_raw = _project_series_to_timebase(toes_pos_r_all, time_arr, t_mocap_raw)
            toes_pos_l_untrimmed_raw = _project_series_to_timebase(toes_pos_l_all, time_arr, t_mocap_raw)
            com_r_untrimmed_raw = _project_series_to_timebase(COM_r, time_arr, t_mocap_raw)
            com_l_untrimmed_raw = _project_series_to_timebase(COM_l, time_arr, t_mocap_raw)
            com_acc_untrimmed_raw = _project_series_to_timebase(COM_Acc_Global, time_arr, t_mocap_raw)
            foot_progression_angle_untrimmed_raw = _project_series_to_timebase(
                foot_progression_angle, time_arr, t_mocap_raw
            )
            calcn_to_floor_angle_untrimmed_raw = _project_series_to_timebase(
                calcn_to_floor_angle, time_arr, t_mocap_raw
            )
            jac_save_untrimmed = slice_jacobian_dofs({
                "jacp": _project_series_to_timebase(jacobian_data["jacp"], time_arr, t_mocap_raw),
                "jacr": _project_series_to_timebase(jacobian_data["jacr"], time_arr, t_mocap_raw),
                "body_ids": jacobian_data["body_ids"],
            }, jac_dof_idx)
            contact_bool_untrimmed_raw = create_contact_boolean(
                grf_untrimmed_raw, cfg["GRF_CONTACT_THRESHOLD"]
            )
            untrimmed_save_map = {
                "ID_GT_MJX.npy": id_gt_untrimmed_raw,
                "qfrc_inverse.npy": qfrc_inverse_untrimmed_raw,
                "qfrc_grf_contribution.npy": qfrc_grf_untrimmed_raw,
                "pos_inputs.npy": build_pos_inputs_without_mtp(pos_untrimmed_raw),
                "vel_inputs.npy": build_vel_inputs_without_mtp(vel_untrimmed_raw),
                "acc_inputs.npy": build_acc_inputs_without_mtp(accel_untrimmed_raw),
                "pelvis_rot_matrix.npy": pelvis_rot_untrimmed_raw,
                "pos_mjx.npy": qpos_untrimmed_raw,
                "qvel_mjx.npy": qvel_untrimmed_raw,
                "qacc_mjx.npy": qacc_untrimmed_raw,
                "GRF_Cleaned.npy": grf_untrimmed_raw,
                "GRF_NoFilt_Trimmed.npy": grf_nofilt_untrimmed_raw,
                "Moment_Cleaned.npy": moment_untrimmed_raw,
                "COP_Cleaned_Relative.npy": cop_rel_untrimmed_raw,
                "KneeToCOP_Vectors.npy": _project_series_to_timebase(KneeToCOP_Vectors_untrimmed, time_arr, t_mocap_raw),
                "forwardVel.npy": np.full(len(t_mocap_raw), forward_vel_val, dtype=np.float32),
                "ankle_heights.npy": ankle_heights_untrimmed_raw,
                "ankle_pos_r.npy": ankle_pos_r_untrimmed_raw,
                "ankle_pos_l.npy": ankle_pos_l_untrimmed_raw,
                "knee_pos_r.npy": _project_series_to_timebase(knee_pos_r_all, time_arr, t_mocap_raw),
                "knee_pos_l.npy": _project_series_to_timebase(knee_pos_l_all, time_arr, t_mocap_raw),
                "toes_pos_r.npy": toes_pos_r_untrimmed_raw,
                "toes_pos_l.npy": toes_pos_l_untrimmed_raw,
                "contactBoolean.npy": contact_bool_untrimmed_raw,
                "COM_r.npy": com_r_untrimmed_raw,
                "COM_l.npy": com_l_untrimmed_raw,
                "COM_Acc_Global.npy": com_acc_untrimmed_raw,
                "Height_m.npy": np.full(len(t_mocap_raw), meta["Height_m"], dtype=np.float32),
                "Mass_kg.npy": np.full(len(t_mocap_raw), meta["Mass_kg"], dtype=np.float32),
                "FootProgressionAngle.npy": foot_progression_angle_untrimmed_raw,
                "Foot_ProgressionAngle.npy": foot_progression_angle_untrimmed_raw,
                "CalcnToFloor_AngleDeg.npy": calcn_to_floor_angle_untrimmed_raw,
            }
            untrimmed_info = {
                "n_frames": int(len(t_mocap_raw)),
                "n_frames_source_mocap": int(n_frames_source_mocap),
                "n_frames_source_motion": int(n_frames_source_motion),
                "mocap_input_source_dir": str(mocap_source_dir.relative_to(trial_path)),
                "time_alignment_target": time_alignment_target,
                "resampled_kinematics_to_motion": bool(resampled_kinematics_to_motion),
                "resampled_forces_to_mocap": bool(resampled_forces_to_mocap),
                "treadmill_flag": bool(is_treadmill),
                "treadmill_speed": round(float(treadmill_speed), 4),
                "ankle_x_range_m": float(ankle_x_range),
                "pelvis_net_speed_mps": float(pelvis_net_speed),
                "floor_height_m": round(float(floor_height), 6),
                "floor_height_method": (
                    "toe_trough_10th_percentile_plus_offset"
                    if bool(cfg.get("ENABLE_FLOOR_TROUGH_OFFSET", True))
                    else "toe_trough_10th_percentile_no_offset"
                ),
                "floor_trough_count": int(trough_count),
                "enable_floor_correction": bool(cfg.get("ENABLE_FLOOR_CORRECTION", True)),
                "enable_kinematics_filtering": bool(cfg.get("ENABLE_KINEMATICS_FILTERING", True)),
                "enable_floor_trough_offset": bool(cfg.get("ENABLE_FLOOR_TROUGH_OFFSET", True)),
                "enable_grf_filtering": bool(cfg.get("ENABLE_GRF_FILTERING", True)),
                "enable_cop_cleaning": bool(cfg.get("ENABLE_COP_CLEANING", True)),
                "subject_model_xml": str(resolve_subject_model_xml(subject_path, cfg)),
                "processing_date": datetime.now().isoformat(),
                "pipeline": "ProcessData_OC_Mocap",
                "note": "Pre-GRF-trim snapshot projected back to the original MoCap timeline.",
            }
            _write_snapshot_dir(
                untrimmed_dir,
                time_vec=t_mocap_raw,
                pos_arr=pos_untrimmed_raw,
                vel_arr=vel_untrimmed_raw,
                accel_arr=accel_untrimmed_raw,
                extra_save_map=untrimmed_save_map,
                jac_payload=jac_save_untrimmed,
                info_payload=untrimmed_info,
            )

        processed_trim_ref = None
        processed_dir = trial_path / "ProcessedData"
        if time_alignment_target == "motion" and processed_dir.is_dir():
            processed_trim_ref = _load_motion_aligned_trim_reference(processed_dir)

        used_processed_trim_reference = False
        if (
            processed_trim_ref is not None
            and int(t_len) == int(processed_trim_ref["pretrim_n_frames"])
        ):
            ref_start = int(processed_trim_ref["start_idx"])
            ref_end = int(processed_trim_ref["end_idx"])
            ref_sl = slice(ref_start, ref_end)
            print(
                f"    [TrimSync] Reusing ProcessedData trim window [{ref_start}:{ref_end}] "
                f"for {subject_path.name}/{trial_path.name}"
            )
            pos = pos[ref_sl]
            vel = vel[ref_sl]
            accel = accel[ref_sl]
            grf = grf[ref_sl]
            grf_no_filt = grf_no_filt[ref_sl]
            grm = grm[ref_sl]
            cop = cop[ref_sl]
            time_arr = time_arr[ref_sl]
            foot_progression_angle = foot_progression_angle[ref_sl]
            calcn_to_floor_angle = calcn_to_floor_angle[ref_sl]
            qpos_matrix = qpos_matrix[ref_sl]
            qvel_matrix = qvel_matrix[ref_sl]
            qacc_matrix = qacc_matrix[ref_sl]
            GRF_mj = GRF_mj[ref_sl]
            GRF_NoFilt_mj = GRF_NoFilt_mj[ref_sl]
            Moment_mj = Moment_mj[ref_sl]
            COP_mj = COP_mj[ref_sl]
            ankle_pos_r_corr = ankle_pos_r_corr[ref_sl]
            ankle_pos_l_corr = ankle_pos_l_corr[ref_sl]
            knee_pos_r_all = knee_pos_r_all[ref_sl]
            knee_pos_l_all = knee_pos_l_all[ref_sl]
            toes_pos_r_all = toes_pos_r_all[ref_sl]
            toes_pos_l_all = toes_pos_l_all[ref_sl]
            COM_r = COM_r[ref_sl]
            COM_l = COM_l[ref_sl]
            com_global = com_global[ref_sl]
            COM_Acc_Global = COM_Acc_Global[ref_sl]
            qfrc_inverse_batch = qfrc_inverse_batch[ref_sl]
            jacobian_data = {
                "jacp": jacobian_data["jacp"][ref_sl],
                "jacr": jacobian_data["jacr"][ref_sl],
                "body_ids": jacobian_data["body_ids"],
            }
            t_len = qpos_matrix.shape[0]
            if t_len < 10:
                raise ValueError(
                    f"Trial too short after ProcessedData-synced trim: {t_len} frames"
                )
            t_after_grf_trim = int(
                processed_trim_ref.get("n_frames_after_grf_trim", t_len)
            )
            t_after_weak_edge_trim = int(
                processed_trim_ref.get("n_frames_after_weak_edge_trim", t_len)
            )
            used_processed_trim_reference = True
        else:
            if processed_trim_ref is not None and int(t_len) != int(processed_trim_ref["pretrim_n_frames"]):
                print(
                    f"    [TrimSync] ProcessedData trim reference length mismatch "
                    f"({processed_trim_ref['pretrim_n_frames']} vs {t_len}); "
                    "falling back to MoCap-local trim logic."
                )

            trim_idx = compute_trim_indices_by_grf(
                grf,
                body_weight=body_weight,
                trim_grf_miss_steps=bool(cfg.get("TrimGRFMissSteps", True)),
                trim_to_double_support=bool(cfg.get("TRIM_TO_DOUBLE_SUPPORT", False)),
            )
            if trim_idx.size == 0:
                raise ValueError("GRF trim removed all frames")

            pos = pos[trim_idx]
            vel = vel[trim_idx]
            accel = accel[trim_idx]
            grf = grf[trim_idx]
            grf_no_filt = grf_no_filt[trim_idx]
            grm = grm[trim_idx]
            cop = cop[trim_idx]
            time_arr = time_arr[trim_idx]
            foot_progression_angle = foot_progression_angle[trim_idx]
            calcn_to_floor_angle = calcn_to_floor_angle[trim_idx]
            qpos_matrix = qpos_matrix[trim_idx]
            qvel_matrix = qvel_matrix[trim_idx]
            qacc_matrix = qacc_matrix[trim_idx]
            GRF_mj = GRF_mj[trim_idx]
            GRF_NoFilt_mj = GRF_NoFilt_mj[trim_idx]
            Moment_mj = Moment_mj[trim_idx]
            COP_mj = COP_mj[trim_idx]
            ankle_pos_r_corr = ankle_pos_r_corr[trim_idx]
            ankle_pos_l_corr = ankle_pos_l_corr[trim_idx]
            knee_pos_r_all = knee_pos_r_all[trim_idx]
            knee_pos_l_all = knee_pos_l_all[trim_idx]
            toes_pos_r_all = toes_pos_r_all[trim_idx]
            toes_pos_l_all = toes_pos_l_all[trim_idx]
            COM_r = COM_r[trim_idx]
            COM_l = COM_l[trim_idx]
            com_global = com_global[trim_idx]
            COM_Acc_Global = COM_Acc_Global[trim_idx]
            qfrc_inverse_batch = qfrc_inverse_batch[trim_idx]
            jacobian_data = {
                "jacp": jacobian_data["jacp"][trim_idx],
                "jacr": jacobian_data["jacr"][trim_idx],
                "body_ids": jacobian_data["body_ids"],
            }
            t_len = qpos_matrix.shape[0]
            if t_len < 20:
                raise ValueError(f"Trial too short after GRF trim: {t_len} frames")
            t_after_grf_trim = int(t_len)

            t_after_weak_edge_trim = int(t_len)
            if bool(cfg.get("TRIM_WEAK_EDGE_STANCES", True)) and not is_treadmill:
                weak_sl, weak_logs = compute_weak_edge_trim_slice(
                    grf,
                    body_weight=body_weight,
                    contact_threshold=float(cfg["GRF_CONTACT_THRESHOLD"]),
                    min_frames=int(cfg.get("TRIM_WEAK_STANCE_MIN_FRAMES", 5)),
                    bw_frac_thresh=float(cfg.get("TRIM_WEAK_STANCE_BW_FRACTION", 0.65)),
                )
                for msg in weak_logs:
                    print(f"    [WeakEdgeTrim] {msg}")
                if weak_sl.start > 0 or weak_sl.stop < int(t_len):
                    pos = pos[weak_sl]
                    vel = vel[weak_sl]
                    accel = accel[weak_sl]
                    grf = grf[weak_sl]
                    grf_no_filt = grf_no_filt[weak_sl]
                    grm = grm[weak_sl]
                    cop = cop[weak_sl]
                    time_arr = time_arr[weak_sl]
                    foot_progression_angle = foot_progression_angle[weak_sl]
                    calcn_to_floor_angle = calcn_to_floor_angle[weak_sl]
                    qpos_matrix = qpos_matrix[weak_sl]
                    qvel_matrix = qvel_matrix[weak_sl]
                    qacc_matrix = qacc_matrix[weak_sl]
                    GRF_mj = GRF_mj[weak_sl]
                    GRF_NoFilt_mj = GRF_NoFilt_mj[weak_sl]
                    Moment_mj = Moment_mj[weak_sl]
                    COP_mj = COP_mj[weak_sl]
                    ankle_pos_r_corr = ankle_pos_r_corr[weak_sl]
                    ankle_pos_l_corr = ankle_pos_l_corr[weak_sl]
                    knee_pos_r_all = knee_pos_r_all[weak_sl]
                    knee_pos_l_all = knee_pos_l_all[weak_sl]
                    toes_pos_r_all = toes_pos_r_all[weak_sl]
                    toes_pos_l_all = toes_pos_l_all[weak_sl]
                    COM_r = COM_r[weak_sl]
                    COM_l = COM_l[weak_sl]
                    com_global = com_global[weak_sl]
                    COM_Acc_Global = COM_Acc_Global[weak_sl]
                    qfrc_inverse_batch = qfrc_inverse_batch[weak_sl]
                    jacobian_data = {
                        "jacp": jacobian_data["jacp"][weak_sl],
                        "jacr": jacobian_data["jacr"][weak_sl],
                        "body_ids": jacobian_data["body_ids"],
                    }
                    t_len = qpos_matrix.shape[0]
                    if t_len < 10:
                        raise ValueError(f"Trial too short after weak-edge-stance trim: {t_len} frames")
                t_after_weak_edge_trim = int(t_len)

            valid_mask = np.ones(t_len, dtype=bool)
            for grf_idx, min_stances in [(2, cfg["MIN_STANCES_FOR_CHECK"]), (5, cfg["MIN_STANCES_FOR_CHECK"])]:
                stances = get_stance_phases(GRF_mj, grf_idx, threshold=cfg["GRF_STANCE_THRESHOLD"])
                if len(stances) >= min_stances:
                    durations = [s["duration_frames"] for s in stances]
                    med = np.median(durations)
                    for s in stances:
                        if s["duration_frames"] > 2 * med and s["duration_frames"] > cfg["MIN_STANCE_LENGTH_FOR_FLAG"]:
                            valid_mask[s["start"]:s["end"]] = False

            si, ei = find_longest_valid_segment(valid_mask)
            sl = slice(si, ei)
            pos = pos[sl]
            vel = vel[sl]
            accel = accel[sl]
            foot_progression_angle = foot_progression_angle[sl]
            calcn_to_floor_angle = calcn_to_floor_angle[sl]
            qpos_matrix = qpos_matrix[sl]
            qvel_matrix = qvel_matrix[sl]
            qacc_matrix = qacc_matrix[sl]
            GRF_mj = GRF_mj[sl]
            GRF_NoFilt_mj = GRF_NoFilt_mj[sl]
            Moment_mj = Moment_mj[sl]
            COP_mj = COP_mj[sl]
            ankle_pos_r_corr = ankle_pos_r_corr[sl]
            ankle_pos_l_corr = ankle_pos_l_corr[sl]
            knee_pos_r_all = knee_pos_r_all[sl]
            knee_pos_l_all = knee_pos_l_all[sl]
            toes_pos_r_all = toes_pos_r_all[sl]
            toes_pos_l_all = toes_pos_l_all[sl]
            COM_r = COM_r[sl]
            COM_l = COM_l[sl]
            com_global = com_global[sl]
            COM_Acc_Global = COM_Acc_Global[sl]
            qfrc_inverse_batch = qfrc_inverse_batch[sl]
            jacobian_data = {
                "jacp": jacobian_data["jacp"][sl],
                "jacr": jacobian_data["jacr"][sl],
                "body_ids": jacobian_data["body_ids"],
            }
            t_len = qpos_matrix.shape[0]
            if t_len < 10:
                raise ValueError(f"Trial too short after outlier-stance removal: {t_len} frames")

        COP_Cleaned_Relative, qfrc_grf, ID_GT_MJX = compute_cop_clean_and_id(
            GRF_mj,
            Moment_mj,
            COP_mj,
            ankle_pos_r_corr,
            ankle_pos_l_corr,
            jacobian_data,
            qfrc_inverse_batch,
            cfg,
            fs,
            GRF_for_torque_mj=GRF_NoFilt_mj if bool(cfg.get("USE_NOFILTER_GRF_FOR_TORQUE", False)) else None,
        )

        KneeToCOP_Vectors, _, _ = _compute_knee_to_cop_vectors(
            COP_Cleaned_Relative,
            ankle_pos_r_corr,
            ankle_pos_l_corr,
            knee_pos_r_all,
            knee_pos_l_all,
        )

        pelvis_euler = qpos_matrix[:, 3:6]
        R_batch = R.from_euler("ZXY", pelvis_euler)
        R_mats = R_batch.as_matrix()
        pelvis_rot_matrix = np.concatenate([R_mats[:, :, 0], R_mats[:, :, 1]], axis=1)

        pos_inputs = build_pos_inputs_without_mtp(pos)
        vel_inputs = build_vel_inputs_without_mtp(vel)
        acc_inputs = build_acc_inputs_without_mtp(accel)
        contact_bool = create_contact_boolean(GRF_mj, cfg["GRF_CONTACT_THRESHOLD"])
        ankle_heights = np.column_stack([ankle_pos_r_corr[:, 2], ankle_pos_l_corr[:, 2]])
        Height_arr = np.full(t_len, meta["Height_m"])
        Mass_arr = np.full(t_len, meta["Mass_kg"])
        Forward_Vel = np.full(t_len, forward_vel_val)

        try:
            above = np.sum(GRF_mj[:, 2] > 1.0) + np.sum(GRF_mj[:, 5] > 1.0)
            bad_pct = 100.0 * (1.0 - above / (2 * max(t_len, 1)))
        except Exception:
            bad_pct = 0.0

        save_map = {
            "ID_GT_MJX.npy": ID_GT_MJX[:, save_dof_idx],
            "qfrc_inverse.npy": qfrc_inverse_batch[:, save_dof_idx],
            "qfrc_grf_contribution.npy": qfrc_grf[:, save_dof_idx],
            "pos_inputs.npy": pos_inputs,
            "vel_inputs.npy": vel_inputs,
            "acc_inputs.npy": acc_inputs,
            "pelvis_rot_matrix.npy": pelvis_rot_matrix,
            "pos_mjx.npy": qpos_matrix,
            "qvel_mjx.npy": qvel_matrix,
            "qacc_mjx.npy": qacc_matrix,
            "GRF_Cleaned.npy": GRF_mj,
            "GRF_NoFilt_Trimmed.npy": GRF_NoFilt_mj,
            "Moment_Cleaned.npy": Moment_mj,
            "COP_Cleaned_Relative.npy": COP_Cleaned_Relative,
            "KneeToCOP_Vectors.npy": KneeToCOP_Vectors,
            "forwardVel.npy": Forward_Vel,
            "ankle_heights.npy": ankle_heights,
            "ankle_pos_r.npy": ankle_pos_r_corr,
            "ankle_pos_l.npy": ankle_pos_l_corr,
            "knee_pos_r.npy": knee_pos_r_all,
            "knee_pos_l.npy": knee_pos_l_all,
            "toes_pos_r.npy": toes_pos_r_all,
            "toes_pos_l.npy": toes_pos_l_all,
            "contactBoolean.npy": contact_bool,
            "COM_r.npy": COM_r,
            "COM_l.npy": COM_l,
            "COM_Acc_Global.npy": COM_Acc_Global,
            "Height_m.npy": Height_arr,
            "Mass_kg.npy": Mass_arr,
            "FootProgressionAngle.npy": foot_progression_angle,
            "Foot_ProgressionAngle.npy": foot_progression_angle,
            "CalcnToFloor_AngleDeg.npy": calcn_to_floor_angle,
        }
        for name, arr in save_map.items():
            np.save(out_dir / name, np.asarray(arr, dtype=np.float32))

        jac_save = slice_jacobian_dofs(jacobian_data, jac_dof_idx)
        np.save(out_dir / "Jacobian.npy", jac_save, allow_pickle=True)

        proc_info = {
            "bad_percentage": round(float(bad_pct), 2),
            "treadmill_flag": bool(is_treadmill),
            "treadmill_speed": round(float(treadmill_speed), 4),
            "was_negated": was_negated,
            "n_frames": int(t_len),
            "n_frames_untrimmed": int(t_untrimmed),
            "n_frames_after_grf_trim": int(t_after_grf_trim),
            "n_frames_after_weak_edge_trim": int(t_after_weak_edge_trim),
            "n_frames_source_mocap": int(n_frames_source_mocap),
            "n_frames_source_motion": int(n_frames_source_motion),
            "mocap_input_source_dir": str(mocap_source_dir.relative_to(trial_path)),
            "time_alignment_target": time_alignment_target,
            "resampled_kinematics_to_motion": bool(resampled_kinematics_to_motion),
            "resampled_forces_to_mocap": bool(resampled_forces_to_mocap),
            "trim_synced_from_processeddata": bool(used_processed_trim_reference),
            "ankle_x_range_m": float(ankle_x_range),
            "pelvis_net_speed_mps": float(pelvis_net_speed),
            "floor_height_m": round(float(floor_height), 6),
            "floor_height_method": (
                "toe_trough_10th_percentile_plus_offset"
                if bool(cfg.get("ENABLE_FLOOR_TROUGH_OFFSET", True))
                else "toe_trough_10th_percentile_no_offset"
            ),
            "floor_trough_count": int(trough_count),
            "enable_floor_correction": bool(cfg.get("ENABLE_FLOOR_CORRECTION", True)),
            "enable_kinematics_filtering": bool(cfg.get("ENABLE_KINEMATICS_FILTERING", True)),
            "enable_floor_trough_offset": bool(cfg.get("ENABLE_FLOOR_TROUGH_OFFSET", True)),
            "enable_grf_filtering": bool(cfg.get("ENABLE_GRF_FILTERING", True)),
            "use_nofilter_grf_for_torque": bool(cfg.get("USE_NOFILTER_GRF_FOR_TORQUE", False)),
            "grf_torque_source": "GRF_NoFilt_Trimmed.npy" if bool(cfg.get("USE_NOFILTER_GRF_FOR_TORQUE", False)) else "GRF_Cleaned.npy",
            "enable_cop_cleaning": bool(cfg.get("ENABLE_COP_CLEANING", True)),
            "cop_edge_hold": bool(cfg.get("COP_EdgeHold", False)),
            "subject_model_xml": str(resolve_subject_model_xml(subject_path, cfg)),
            "processing_date": datetime.now().isoformat(),
            "pipeline": "ProcessData_OC_Mocap",
            "companion_raw_timebase_dir": OC_MOCAP_RAW_TIMEBASE_DIRNAME,
        }
        with open(out_dir / "Trial_Processing_Information.json", "w") as f:
            json.dump(proc_info, f, indent=4)

        alignment_time = t_motion if time_alignment_target == "motion" else t_mocap_raw
        aligned_start = (
            int(np.argmin(np.abs(alignment_time - time_arr[0]))) if len(time_arr) else 0
        )
        aligned_end = aligned_start + int(t_len)
        oc_trace = {
            "schema_name": "ProcessData trimming traceability",
            "schema_version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "trial_id": trial_id,
            "pipeline": "ProcessData.py --OC_Mocap",
            "interval_convention": "All frame bounds are zero-based, half-open [start, end).",
            "source_inputs": {
                "kinematics": {
                    "position": _array_source_record(
                        mocap_source_dir / _kinematics_input_filename("Pos.npy", cfg),
                        _load_npy_numeric(
                            mocap_source_dir / _kinematics_input_filename("Pos.npy", cfg)
                        ),
                        trial_path,
                    ),
                    "time": _time_vector_record(
                        mocap_source_dir / "Time.npy",
                        t_mocap_raw, t_mocap_raw, len(t_mocap_raw), trial_path,
                    ),
                },
                "forces": {
                    "grf": _array_source_record(
                        motion_dir / "GRF.npy", grf_raw, trial_path
                    ),
                    "moment": _array_source_record(
                        motion_dir / "GRM.npy", grm_raw, trial_path
                    ),
                    "cop": _array_source_record(
                        motion_dir / "COP.npy", cop_raw, trial_path
                    ),
                    "time": _time_vector_record(
                        motion_dir / "Time.npy",
                        t_motion, t_motion, len(t_motion), trial_path,
                    ),
                },
            },
            "model_dependency": _file_identity_record(xml_path, subject_path),
            "uniform_resampling": {
                "target_timebase": time_alignment_target,
                "target_frame_count_before_trim": int(t_untrimmed),
                "target_time_s": np.asarray(alignment_time, dtype=float).tolist(),
                "kinematic_source_row_map": _linear_interpolation_map(
                    t_mocap_raw, alignment_time
                ),
                "force_source_row_map": _linear_interpolation_map(
                    t_motion, alignment_time
                ),
            },
            "timeline_stages": [
                _trace_stage(
                    (
                        "synchronized_to_processeddata_trim"
                        if used_processed_trim_reference else
                        "oc_mocap_local_grf_weak_edge_and_outlier_trim"
                    ),
                    int(t_untrimmed),
                    aligned_start,
                    aligned_end,
                    0,
                    enabled=True,
                    parameters={
                        "trim_synced_from_processeddata": bool(
                            used_processed_trim_reference
                        ),
                        "grf_misstep_trim_enabled": bool(
                            cfg.get("TrimGRFMissSteps", True)
                        ),
                        "double_support_trim_enabled": bool(
                            cfg.get("TRIM_TO_DOUBLE_SUPPORT", False)
                        ),
                        "weak_edge_trim_enabled": bool(
                            cfg.get("TRIM_WEAK_EDGE_STANCES", True)
                        ),
                        "outlier_stance_removal_enabled": True,
                    },
                    details={
                        "n_frames_after_grf_trim": int(t_after_grf_trim),
                        "n_frames_after_weak_edge_trim": int(
                            t_after_weak_edge_trim
                        ),
                        "processeddata_reference": (
                            processed_trim_ref if used_processed_trim_reference else None
                        ),
                    },
                )
            ],
            "final_mapping": {
                "uniform_resampled_frame_bounds": [aligned_start, aligned_end],
                "final_frame_count": int(t_len),
                "final_first_time_s": float(time_arr[0]) if len(time_arr) else None,
                "final_last_time_s": float(time_arr[-1]) if len(time_arr) else None,
                "mapping_formula": (
                    f"final_frame[j] corresponds to aligned_{time_alignment_target}"
                    f"_frame[{aligned_start} + j]"
                ),
            },
            "output_files": _output_manifest(out_dir, t_len),
            "postprocessing_history": [],
        }
        _write_json_atomic(out_dir / TRIMMING_TRACE_FILENAME, oc_trace)

        raw_timebase = _select_trimmed_raw_timebase(time_arr, t_mocap_raw)
        if raw_timebase.size > 0:
            raw_pos = _project_series_to_timebase(pos, time_arr, raw_timebase)
            raw_vel = _project_series_to_timebase(vel, time_arr, raw_timebase)
            raw_accel = _project_series_to_timebase(accel, time_arr, raw_timebase)
            raw_qpos = _project_series_to_timebase(qpos_matrix, time_arr, raw_timebase)
            raw_qfrc_inverse = _project_series_to_timebase(qfrc_inverse_batch[:, save_dof_idx], time_arr, raw_timebase)
            raw_qfrc_grf = _project_series_to_timebase(qfrc_grf[:, save_dof_idx], time_arr, raw_timebase)
            raw_id_gt = _project_series_to_timebase(ID_GT_MJX[:, save_dof_idx], time_arr, raw_timebase)
            raw_pelvis_rot = _project_series_to_timebase(pelvis_rot_matrix, time_arr, raw_timebase)
            raw_qvel = _project_series_to_timebase(qvel_matrix, time_arr, raw_timebase)
            raw_qacc = _project_series_to_timebase(qacc_matrix, time_arr, raw_timebase)
            raw_grf = _project_series_to_timebase(GRF_mj, time_arr, raw_timebase)
            raw_grf_nofilt = _project_series_to_timebase(GRF_NoFilt_mj, time_arr, raw_timebase)
            raw_moment = _project_series_to_timebase(Moment_mj, time_arr, raw_timebase)
            raw_cop_rel = _project_series_to_timebase(COP_Cleaned_Relative, time_arr, raw_timebase)
            raw_forward_vel = np.full(len(raw_timebase), forward_vel_val, dtype=np.float32)
            raw_ankle_heights = _project_series_to_timebase(ankle_heights, time_arr, raw_timebase)
            raw_ankle_pos_r = _project_series_to_timebase(ankle_pos_r_corr, time_arr, raw_timebase)
            raw_ankle_pos_l = _project_series_to_timebase(ankle_pos_l_corr, time_arr, raw_timebase)
            raw_knee_pos_r = _project_series_to_timebase(knee_pos_r_all, time_arr, raw_timebase)
            raw_knee_pos_l = _project_series_to_timebase(knee_pos_l_all, time_arr, raw_timebase)
            raw_toes_pos_r = _project_series_to_timebase(toes_pos_r_all, time_arr, raw_timebase)
            raw_toes_pos_l = _project_series_to_timebase(toes_pos_l_all, time_arr, raw_timebase)
            raw_com_r = _project_series_to_timebase(COM_r, time_arr, raw_timebase)
            raw_com_l = _project_series_to_timebase(COM_l, time_arr, raw_timebase)
            raw_com_acc = _project_series_to_timebase(COM_Acc_Global, time_arr, raw_timebase)
            raw_contact_bool = create_contact_boolean(raw_grf, cfg["GRF_CONTACT_THRESHOLD"])
            raw_fpa = _project_series_to_timebase(foot_progression_angle, time_arr, raw_timebase)
            raw_cfa = _project_series_to_timebase(calcn_to_floor_angle, time_arr, raw_timebase)
            raw_jac_save = slice_jacobian_dofs({
                "jacp": _project_series_to_timebase(jacobian_data["jacp"], time_arr, raw_timebase),
                "jacr": _project_series_to_timebase(jacobian_data["jacr"], time_arr, raw_timebase),
                "body_ids": jacobian_data["body_ids"],
            }, jac_dof_idx)
            raw_save_map = {
                "ID_GT_MJX.npy": raw_id_gt,
                "qfrc_inverse.npy": raw_qfrc_inverse,
                "qfrc_grf_contribution.npy": raw_qfrc_grf,
                "pos_inputs.npy": build_pos_inputs_without_mtp(raw_pos),
                "vel_inputs.npy": build_vel_inputs_without_mtp(raw_vel),
                "acc_inputs.npy": build_acc_inputs_without_mtp(raw_accel),
                "pelvis_rot_matrix.npy": raw_pelvis_rot,
                "pos_mjx.npy": raw_qpos,
                "qvel_mjx.npy": raw_qvel,
                "qacc_mjx.npy": raw_qacc,
                "GRF_Cleaned.npy": raw_grf,
                "GRF_NoFilt_Trimmed.npy": raw_grf_nofilt,
                "Moment_Cleaned.npy": raw_moment,
                "COP_Cleaned_Relative.npy": raw_cop_rel,
                "KneeToCOP_Vectors.npy": _project_series_to_timebase(KneeToCOP_Vectors, time_arr, raw_timebase),
                "forwardVel.npy": raw_forward_vel,
                "ankle_heights.npy": raw_ankle_heights,
                "ankle_pos_r.npy": raw_ankle_pos_r,
                "ankle_pos_l.npy": raw_ankle_pos_l,
                "knee_pos_r.npy": raw_knee_pos_r,
                "knee_pos_l.npy": raw_knee_pos_l,
                "toes_pos_r.npy": raw_toes_pos_r,
                "toes_pos_l.npy": raw_toes_pos_l,
                "contactBoolean.npy": raw_contact_bool,
                "COM_r.npy": raw_com_r,
                "COM_l.npy": raw_com_l,
                "COM_Acc_Global.npy": raw_com_acc,
                "Height_m.npy": np.full(len(raw_timebase), meta["Height_m"], dtype=np.float32),
                "Mass_kg.npy": np.full(len(raw_timebase), meta["Mass_kg"], dtype=np.float32),
                "FootProgressionAngle.npy": raw_fpa,
                "Foot_ProgressionAngle.npy": raw_fpa,
                "CalcnToFloor_AngleDeg.npy": raw_cfa,
            }
            raw_proc_info = dict(proc_info)
            raw_proc_info.update({
                "n_frames": int(len(raw_timebase)),
                "raw_timebase_bundle": True,
                "raw_timebase_source_dir": "MoCap",
                "raw_timebase_source_window_seconds": [float(time_arr[0]), float(time_arr[-1])],
                "raw_timebase_projection_origin": "trimmed_motion_aligned_bundle",
                "pipeline": "ProcessData_OC_Mocap_RawTimebase",
            })
            _write_snapshot_dir(
                mocap_raw_timebase_dir,
                time_vec=raw_timebase,
                pos_arr=raw_pos,
                vel_arr=raw_vel,
                accel_arr=raw_accel,
                extra_save_map=raw_save_map,
                jac_payload=raw_jac_save,
                info_payload=raw_proc_info,
            )

        pre_missstep_dir = mocap_dir / "Untrimmed"
        pre_missstep_info = dict(proc_info)
        pre_missstep_info["note"] = "Processed snapshot prior to miss-step cleanup."
        _write_snapshot_dir(
            pre_missstep_dir,
            time_vec=time_arr,
            pos_arr=pos,
            vel_arr=vel,
            accel_arr=accel,
            extra_save_map=save_map,
            jac_payload=jac_save,
            info_payload=pre_missstep_info,
        )

        if cfg["DO_DEVIATION_LEARNING_PREP"] and deviation_data is not None:
            try:
                final_output = deviation_data["final_output"]
                global_median_dur = deviation_data["global_median_duration"]
                global_median_fpa = deviation_data["global_median_fpa"]

                recon_grf, recon_cop, recon_moment = build_average_reconstructions(
                    GRF_mj, COP_Cleaned_Relative, Moment_mj,
                    final_output, global_median_dur, global_median_fpa,
                    meta["Mass_kg"], meta["Height_m"],
                    ankle_pos_r=ankle_pos_r_corr,
                    toes_pos_r=toes_pos_r_all,
                    ankle_pos_l=ankle_pos_l_corr,
                    toes_pos_l=toes_pos_l_all,
                )
                np.save(out_dir / "GRF_average_reconstructed.npy", recon_grf)
                np.save(out_dir / "COP_average_reconstructed.npy", recon_cop)
                np.save(out_dir / "Moment_average_reconstructed.npy", recon_moment)
            except Exception as dev_e:
                warnings.warn(f"[{trial_id}] Deviation learning prep failed: {dev_e}")

        result_dict = {"id": trial_id, "success": True, "n_frames": int(t_len), "treadmill": is_treadmill}

        del (
            qpos_matrix, qvel_matrix, qacc_matrix,
            qfrc_inverse_batch, qfrc_grf, jacobian_data, jac_save,
            GRF_mj, Moment_mj, COP_mj, COP_Cleaned_Relative,
            pos_inputs, vel_inputs, acc_inputs,
            pos, vel, accel,
            ankle_pos_r_corr, ankle_pos_l_corr, ankle_pos_r_all, ankle_pos_l_all,
            knee_pos_r_all, knee_pos_l_all,
            toes_pos_r_all, toes_pos_l_all,
            com_global, COM_r, COM_l, COM_Acc_Global, com_vel, com_acc,
            ID_GT_MJX, pelvis_rot_matrix,
        )
        del mj_data, mj_model, mjx_model
        gc.collect()

        return result_dict

    except Exception as exc:
        import traceback
        gc.collect()
        return {"id": trial_id, "success": False, "error": str(exc),
                "traceback": traceback.format_exc()}


# ═══════════════════════════════════════════════════════════════
#              MULTIPROCESSING WORKER WRAPPER
# ═══════════════════════════════════════════════════════════════

def _worker(args):
    """Top-level function (picklable) for multiprocessing.Pool."""
    subject_path, trial_path, cfg, deviation_data = args
    result = process_single_trial(
        Path(subject_path), Path(trial_path), cfg, deviation_data
    )
    # Clear JAX compilation caches accumulated in this worker process so
    # that memory is released before the next trial runs in the same worker.
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()
    return result


# ═══════════════════════════════════════════════════════════════
#                     MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def discover_all_trials(data_root: Path,
                        subject_filter: str | None = None,
                        subject_list:   list[str] | None = None,
                        required_dir_name: str | None = None,
                        cfg: dict | None = None) -> list:
    """Return list of (subject_path, trial_path) tuples.

    subject_filter : exact name of a single subject directory to include.
    subject_list   : list of exact subject directory names to include.
                     If both subject_filter and subject_list are provided,
                     subject_list takes precedence.
    """
    only_overground = bool((cfg or {}).get("OnlyProcessOverGround", False))

    # Build the set of allowed subject names (None = allow all)
    allowed: set[str] | None = None
    if subject_list:
        allowed = set(subject_list)
    elif subject_filter:
        allowed = {subject_filter}

    # Optional trial-level filter: list of "<Subject>/<Trial>" strings.
    trials_to_process = (cfg or {}).get("TRIALS_TO_PROCESS")
    allowed_trial_ids: set[str] | None = (
        set(trials_to_process) if trials_to_process else None
    )

    trials = []
    matched_subjects: set[str] = set()
    scanned_trials = 0
    skipped_treadmill = 0
    skipped_undetermined = 0
    skipped_examples: list[str] = []
    for subject_dir in sorted(data_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        if allowed is not None and subject_dir.name not in allowed:
            continue
        matched_subjects.add(subject_dir.name)
        _opencapval = bool((cfg or {}).get("OpenCapVal", False))
        for trial_dir in sorted(subject_dir.iterdir()):
            _is_trial = trial_dir.name.startswith("Trial") or (
                _opencapval and trial_dir.name.lower().startswith("trial")
            )
            if not (trial_dir.is_dir() and _is_trial):
                continue
            if required_dir_name is not None and not (trial_dir / required_dir_name).is_dir():
                continue
            if allowed_trial_ids is not None and f"{subject_dir.name}/{trial_dir.name}" not in allowed_trial_ids:
                continue
            scanned_trials += 1
            if only_overground:
                treadmill_flag, reason = _infer_raw_trial_treadmill_flag(trial_dir, cfg)
                if treadmill_flag is None:
                    skipped_undetermined += 1
                    if len(skipped_examples) < 10:
                        skipped_examples.append(f"{subject_dir.name}/{trial_dir.name}: {reason}")
                    continue
                if treadmill_flag:
                    skipped_treadmill += 1
                    continue
            trials.append((subject_dir, trial_dir))

    if allowed is not None:
        missing = sorted(allowed - matched_subjects)
        if missing:
            print(f"[Discover] Warning: requested subjects not found: {', '.join(missing)}")
    if only_overground:
        print(
            f"[Discover] OnlyProcessOverGround kept {len(trials)}/{scanned_trials} trials "
            f"(skipped treadmill={skipped_treadmill}, skipped undetermined={skipped_undetermined})"
        )
        for msg in skipped_examples:
            print(f"[Discover]   undetermined -> {msg}")
    return trials


def _discover_existing_output_dirs(
    data_root: Path,
    cfg: dict,
    subject_filter: str | None = None,
    subject_list: list[str] | None = None,
    include_standard_when_oc: bool = False,
) -> list[Path]:
    output_dir_names = (
        get_output_dir_names(cfg)
        if include_standard_when_oc
        else [get_output_dir_name(cfg)]
    )
    required_dir_name = "MoCap" if bool(cfg.get("OC_Mocap", False)) else None
    trials = discover_all_trials(
        data_root,
        subject_filter=subject_filter,
        subject_list=subject_list,
        required_dir_name=required_dir_name,
        cfg=cfg,
    )
    out_dirs = []
    seen: set[Path] = set()
    for _, trial_path in trials:
        for output_dir_name in output_dir_names:
            out_dir = trial_path / output_dir_name
            if not out_dir.is_dir() or out_dir in seen:
                continue
            out_dirs.append(out_dir)
            seen.add(out_dir)
    return out_dirs


def _append_oc_mocap_raw_timebase_dirs(source_dirs: list[Path], cfg: dict) -> list[Path]:
    """Include OC_Mocap raw-time companion directories when present."""
    if not bool(cfg.get("OC_Mocap", False)):
        return source_dirs

    expanded: list[Path] = []
    seen: set[Path] = set()
    for src_dir in source_dirs:
        for candidate in (src_dir, src_dir.parent / OC_MOCAP_RAW_TIMEBASE_DIRNAME):
            if not candidate.is_dir() or candidate in seen:
                continue
            expanded.append(candidate)
            seen.add(candidate)
    return expanded


def _cfg_for_output_source_dir(cfg: dict, src_dir: Path) -> dict:
    """Return source-specific config for OpenCapVal output dirs."""
    if not bool(cfg.get("OpenCapVal", False)):
        return cfg
    parts = src_dir.parts
    source = None
    if "MoCap" in parts:
        source = "MoCap"
    elif "Video" in parts:
        source = "Video"
    if source is None:
        return cfg
    source_cfg = dict(cfg)
    source_cfg["OpenCapVal"] = False
    source_cfg["OPENCAPVAL_SOURCE"] = source
    return source_cfg


def _trial_path_from_output_source_dir(src_dir: Path) -> Path:
    """Resolve a trial directory from a processed-output/source directory.

    Supported layouts include:
      - Subject/Trial/ProcessedData
      - Subject/Trial/MoCap                       (standard/OC_Mocap side output)
      - Subject/Trial/MoCap/ProcessedData         (--OpenCapVal source output)
      - Subject/Trial/Video/ProcessedData         (--OpenCapVal source output)

    The old generic `ProcessedData -> parent.parent` rule only worked for
    source-specific --OpenCapVal outputs and incorrectly collapsed standard
    TrustedDataSet paths from Subject/Trial/ProcessedData to Subject.
    """
    src_dir = Path(src_dir)
    if src_dir.name == "ProcessedData":
        if src_dir.parent.name in {"MoCap", "Video"}:
            return src_dir.parent.parent
        return src_dir.parent
    if src_dir.name in {"MoCap", "Video"}:
        return src_dir.parent
    if src_dir.name == OC_MOCAP_RAW_TIMEBASE_DIRNAME:
        return src_dir.parent.parent
    return src_dir.parent


def run_calc_frame_postprocess(
    data_root: Path,
    cfg: dict,
    subject_filter: str | None = None,
    subject_list: list[str] | None = None,
) -> None:
    source_dirs = _discover_existing_output_dirs(
        data_root,
        cfg,
        subject_filter,
        subject_list,
        include_standard_when_oc=True,
    )
    if not bool(cfg.get("OpenCapVal", False)):
        # OpenCapSubjects-derived datasets often already contain Trial/MoCap folders
        # even when ProcessData.py is run in the standard ProcessedData mode. Include
        # them here so derived calc-frame files, including GRFNorm COP, stay complete.
        # This does not use source-specific model XMLs; those are only selected when
        # --OpenCapVal is active via _cfg_for_output_source_dir().
        seen_source_dirs = set(source_dirs)
        for _, trial_path in discover_all_trials(
            data_root,
            subject_filter=subject_filter,
            subject_list=subject_list,
            required_dir_name=None,
            cfg=cfg,
        ):
            mocap_dir = trial_path / "MoCap"
            if mocap_dir.is_dir() and mocap_dir not in seen_source_dirs:
                source_dirs.append(mocap_dir)
                seen_source_dirs.add(mocap_dir)
    source_dirs = _append_oc_mocap_raw_timebase_dirs(source_dirs, cfg)
    output_dir_name = get_output_dir_label(cfg)
    print(f"\n{'='*60}")
    print(f"  Post-processing: calc-frame COP/FPA outputs ({output_dir_name})")

    if not source_dirs:
        print(f"  [CalcFrame] No {output_dir_name} directories found — skipping.")
        print("="*60)
        return

    updated = 0
    skipped = 0
    for src_dir in source_dirs:
        trial_path = _trial_path_from_output_source_dir(src_dir)
        subject_path = trial_path.parent
        trial_id = f"{subject_path.name}/{trial_path.name}"
        source_cfg = _cfg_for_output_source_dir(cfg, src_dir)
        try:
            xml_path = resolve_subject_model_xml(subject_path, source_cfg)
        except Exception as e:
            skipped += 1
            print(f"  [CalcFrame] ✗ {trial_id}: could not resolve model XML ({e})")
            continue

        ok, message = generate_calc_frame_outputs_for_source(src_dir, xml_path, trial_id)
        if ok:
            updated += 1
            print(f"  [CalcFrame] ✓ {message}")
        else:
            skipped += 1
            print(f"  [CalcFrame] ↷ {message}")

    print(f"  [CalcFrame] Updated: {updated}")
    print(f"  [CalcFrame] Skipped: {skipped}")
    print("="*60)


def run_missing_step_postprocess(
    data_root: Path,
    cfg: dict,
    subject_filter: str | None = None,
    subject_list: list[str] | None = None,
) -> None:
    proc_dirs = _discover_existing_output_dirs(data_root, cfg, subject_filter, subject_list)
    source_dir_name = get_output_dir_name(cfg)
    print(f"\n{'='*60}")
    print(f"  Post-processing: miss-step cleanup ({source_dir_name})")

    if not proc_dirs:
        print(f"  [MissStep] No {source_dir_name} directories found — skipping.")
        print("="*60)
        return

    fs_hz = float(cfg.get("MISSSTEP_FS_HZ", cfg.get("SAMPLING_RATE_HZ", 100.0)))
    max_duration_s = float(cfg.get("MISSSTEP_MAX_DURATION_S", 5.0))
    half_ratio_threshold = float(cfg.get("MISSSTEP_HALF_RATIO_THRESHOLD", 0.8))
    peak_offset_frames = int(cfg.get("MISSSTEP_PEAK_OFFSET_FRAMES", 2))
    edge_trim_frames = int(cfg.get("MISSSTEP_DOUBLE_SUPPORT_EDGE_TRIM_FRAMES", 20))
    if bool(cfg.get("OC_Mocap", False)) and edge_trim_frames != 0:
        print(
            f"  [MissStep] OC_Mocap active: suppressing extra edge trim "
            f"({edge_trim_frames} frames) for MoCap-side cleanup."
        )
        edge_trim_frames = 0

    analyzable_trials: list[MissingStepTrialResult] = []
    flagged_trials: list[MissingStepTrialResult] = []
    skipped_long = 0
    skipped_invalid = 0
    skipped_treadmill = 0
    skipped_missing_treadmill_meta = 0

    for proc_dir in proc_dirs:
        if bool(cfg.get("OC_Mocap", False)):
            mocap_dir = proc_dir
            processed_dir = mocap_dir.parent / "ProcessedData"
            source_dir = _select_analysis_source(
                [
                    processed_dir / "Untrimmed" if (processed_dir / "Untrimmed").is_dir() else None,
                    processed_dir if processed_dir.is_dir() else None,
                    mocap_dir / "Untrimmed" if (mocap_dir / "Untrimmed").is_dir() else None,
                    mocap_dir,
                ]
            )
            if source_dir is None:
                skipped_invalid += 1
                print(f"  [MissStep] ↷ {mocap_dir.parent.parent.name}/{mocap_dir.parent.name}: no valid MoCap/ProcessedData source")
                continue
        else:
            source_dir = _preferred_source_dir(proc_dir)

        treadmill_flag, treadmill_reason = _read_treadmill_flag(proc_dir=proc_dir, source_dir=source_dir)
        if treadmill_flag is None:
            skipped_missing_treadmill_meta += 1
            print(f"  [MissStep] ↷ {proc_dir.parent.parent.name}/{proc_dir.parent.name}: {treadmill_reason}")
            continue
        if treadmill_flag:
            skipped_treadmill += 1
            continue

        result, reason = _analyze_trial_for_missing_steps(
            proc_dir=proc_dir,
            fs_hz=fs_hz,
            max_duration_s=max_duration_s,
            half_ratio_threshold=half_ratio_threshold,
            source_dir=source_dir,
        )
        if result is not None:
            analyzable_trials.append(result)
            if result.flagged_stances:
                flagged_trials.append(result)
            continue
        if reason == "trial longer than max duration":
            skipped_long += 1
        else:
            skipped_invalid += 1

    print(f"  [MissStep] Total {source_dir_name} dirs: {len(proc_dirs)}")
    print(f"  [MissStep] Skipped treadmill: {skipped_treadmill}")
    print(f"  [MissStep] Skipped metadata: {skipped_missing_treadmill_meta}")
    print(f"  [MissStep] Skipped long: {skipped_long}")
    print(f"  [MissStep] Skipped invalid: {skipped_invalid}")
    print(f"  [MissStep] Analyzable non-treadmill trials: {len(analyzable_trials)}")
    print(f"  [MissStep] Flagged trials: {len(flagged_trials)}")

    if not analyzable_trials:
        print("="*60)
        return

    trimmed_trial_count = 0
    for trial in analyzable_trials:
        source_len = len(trial.vgrf_r_bw)
        trial_edge_trim = int(trial.ds_trim_frames)

        if trial.flagged_stances:
            logic_start, logic_end, final_start, final_end = _compute_final_missing_step_bounds(
                trial,
                peak_offset_frames=peak_offset_frames,
                edge_trim_frames=trial_edge_trim,
            )
        else:
            if trial_edge_trim <= 0:
                continue
            logic_start, logic_end = 0, source_len
            final_start = trial_edge_trim
            final_end = source_len - trial_edge_trim

        if final_start >= final_end:
            print(
                f"  [MissStep] ↷ {trial.subject}/{trial.trial}: invalid final bounds "
                f"[{final_start}, {final_end})"
            )
            continue

        trim_targets: list[tuple[str, Path]] = [(source_dir_name, trial.processed_dir)]
        if bool(cfg.get("OC_Mocap", False)):
            processed_dir = trial.processed_dir.parent / "ProcessedData"
            if processed_dir.is_dir():
                trim_targets.append(("ProcessedData", processed_dir))
            raw_timebase_dir = trial.processed_dir.parent / OC_MOCAP_RAW_TIMEBASE_DIRNAME
            if raw_timebase_dir.is_dir():
                trim_targets.append((OC_MOCAP_RAW_TIMEBASE_DIRNAME, raw_timebase_dir))

        per_target_updates: list[str] = []
        for label, target_dir in trim_targets:
            untrimmed_dir = _ensure_untrimmed_backup(target_dir)
            target_len = _infer_trial_length(untrimmed_dir)
            if target_len is None:
                target_len = source_len

            target_start, target_end = _translate_bounds_by_edge_trim(
                source_len=source_len,
                target_len=target_len,
                source_start=final_start,
                source_end=final_end,
            )
            if target_start >= target_end:
                per_target_updates.append(
                    f"{label}: skipped (invalid bounds [{target_start}:{target_end}] for len={target_len})"
                )
                continue

            trimmed_files, total_files = _rewrite_trimmed_from_backup(
                proc_dir=target_dir,
                untrimmed_dir=untrimmed_dir,
                start_idx=target_start,
                end_idx=target_end,
                original_len=target_len,
            )
            _update_trial_info_json(
                target_dir,
                {
                    "n_frames": int(target_end - target_start),
                    "miss_step_cleanup_applied": True,
                    "miss_step_flagged_stances": int(len(trial.flagged_stances)),
                    "miss_step_logic_bounds": [int(logic_start), int(logic_end)],
                    "miss_step_final_bounds": [int(final_start), int(final_end)],
                    "miss_step_edge_trim_frames": int(trial_edge_trim),
                    "miss_step_edge_trim_cap_frames": int(edge_trim_frames),
                    "miss_step_was_flagged": bool(trial.flagged_stances),
                    "miss_step_peak_offset_frames": int(peak_offset_frames),
                    "miss_step_trimmed_files": int(trimmed_files),
                    "miss_step_total_files_seen": int(total_files),
                },
            )
            _append_postprocess_trim_trace(
                target_dir,
                stage_name="postprocess_missing_step_cleanup",
                input_count=int(target_len),
                keep_start=int(target_start),
                keep_end=int(target_end),
                parameters={
                    "sampling_rate_hz": float(fs_hz),
                    "maximum_trial_duration_s": float(max_duration_s),
                    "half_ratio_threshold": float(half_ratio_threshold),
                    "peak_offset_frames": int(peak_offset_frames),
                    "double_support_edge_trim_frames": int(trial_edge_trim),
                },
                details={
                    "source_analysis_frame_count": int(source_len),
                    "source_logic_bounds": [int(logic_start), int(logic_end)],
                    "source_final_bounds": [int(final_start), int(final_end)],
                    "target_label": label,
                    "target_bounds": [int(target_start), int(target_end)],
                    "flagged_stance_count": int(len(trial.flagged_stances)),
                    "trimmed_file_count": int(trimmed_files),
                    "total_files_seen": int(total_files),
                },
            )
            per_target_updates.append(
                f"{label} {target_len}->{target_end - target_start} ({trimmed_files}/{total_files} files)"
            )

        if not per_target_updates:
            continue

        trimmed_trial_count += 1
        print(
            f"  [MissStep] {trial.subject}/{trial.trial}: "
            f"logic [{logic_start}:{logic_end}] -> final [{final_start}:{final_end}] "
            f"(source_len={source_len}) | updates: {', '.join(per_target_updates)}"
        )

    print(f"  [MissStep] Trimmed trials: {trimmed_trial_count}")
    print("="*60)


def run_cop_outlier_postprocess(
    data_root: Path,
    cfg: dict,
    subject_filter: str | None = None,
    subject_list: list[str] | None = None,
) -> None:
    source_dirs = _append_oc_mocap_raw_timebase_dirs(
        _discover_existing_output_dirs(
            data_root,
            cfg,
            subject_filter,
            subject_list,
            include_standard_when_oc=True,
        ),
        cfg,
    )
    output_dir_name = get_output_dir_label(cfg)
    print(f"\n{'='*60}")
    print(f"  Post-processing: COP outlier cleanup ({output_dir_name})")

    reports = []
    errors = []
    for src_dir in source_dirs:
        cop_path = src_dir / COP_OUTLIER_FILENAME
        if not cop_path.exists():
            continue
        report = _analyze_cop_outlier_file(cop_path)
        if report.get("error"):
            errors.append(report)
        elif report.get("needs_fix"):
            reports.append(report)

    if errors:
        print(f"  [COPOutlier] Invalid/unreadable files: {len(errors)}")
        for err in errors[:25]:
            print(f"    {err['path']} -> {err['error']}")
        if len(errors) > 25:
            print(f"    ... {len(errors) - 25} more")

    if not reports:
        print("  [COPOutlier] No out-of-range COP values found.")
        print("="*60)
        return

    move_threshold = float(cfg.get("COP_OUTLIER_MOVE_THRESHOLD_PCT", 10.0))
    move_bad_trials = bool(cfg.get("COP_OUTLIER_MOVE_BAD_TRIALS", False))
    bad_root = data_root.parent / str(cfg.get("COP_OUTLIER_BAD_TRIALS_ROOT_NAME", "BadTrialsFromTrustedDataset"))

    to_fix = [report for report in reports if float(report.get("max_bad_pct", 0.0)) <= move_threshold]
    high_bad = [report for report in reports if float(report.get("max_bad_pct", 0.0)) > move_threshold]

    fixed = 0
    fixed_frames = 0
    failed_fix: list[tuple[Path, str]] = []
    for report in to_fix:
        cop_path = report["path"]
        try:
            result = _fix_cop_outlier_file(cop_path)
            fixed += 1
            fixed_frames += int(sum(result["fixed_counts"].values()))
            _update_trial_info_json(
                cop_path.parent,
                {
                    "cop_outlier_cleanup_applied": True,
                    "cop_outlier_fixed_counts": {str(k): int(v) for k, v in result["fixed_counts"].items()},
                    "cop_outlier_max_bad_pct": float(report.get("max_bad_pct", 0.0)),
                },
            )
        except Exception as e:
            failed_fix.append((cop_path, str(e)))

    moved = 0
    failed_move: list[tuple[Path, str]] = []
    if high_bad:
        print(
            f"  [COPOutlier] High-bad trials above {move_threshold:.2f}%: {len(high_bad)} "
            f"(move_bad_trials={move_bad_trials})"
        )
    if move_bad_trials:
        for report in high_bad:
            trial_dir = report["path"].parent.parent
            try:
                _move_trial_folder_to_bad_root(trial_dir, bad_root)
                moved += 1
            except Exception as e:
                failed_move.append((trial_dir, str(e)))
    else:
        for report in high_bad:
            _update_trial_info_json(
                report["path"].parent,
                {
                    "cop_outlier_cleanup_applied": False,
                    "cop_outlier_high_bad_pct": float(report.get("max_bad_pct", 0.0)),
                    "cop_outlier_high_bad_action": "reported_only",
                },
            )
            try:
                rel = report["path"].relative_to(data_root)
            except ValueError:
                rel = report["path"]
            print(f"    [COPOutlier] REPORT {rel} max_bad_pct={float(report.get('max_bad_pct', 0.0)):.2f}")

    print(f"  [COPOutlier] Fixed files: {fixed}/{len(to_fix)}")
    print(f"  [COPOutlier] Repaired frame-values: {fixed_frames}")
    print(f"  [COPOutlier] High-bad moved: {moved}/{len(high_bad) if move_bad_trials else 0}")
    if failed_fix:
        print(f"  [COPOutlier] Failed fixes: {len(failed_fix)}")
        for path, err in failed_fix[:25]:
            print(f"    {path} -> {err}")
        if len(failed_fix) > 25:
            print(f"    ... {len(failed_fix) - 25} more")
    if failed_move:
        print(f"  [COPOutlier] Failed moves: {len(failed_move)}")
        for path, err in failed_move[:25]:
            print(f"    {path} -> {err}")
        if len(failed_move) > 25:
            print(f"    ... {len(failed_move) - 25} more")
    print("="*60)


def load_deviation_data(cfg: dict,
                        trials: list | None = None) -> dict | None:
    """Load (or build) the stance-metrics averages for deviation learning.

    When DEVIATION_LOAD_AVERAGES_FROM_FILE=True  → load from pkl.
    When DEVIATION_LOAD_AVERAGES_FROM_FILE=False → scan all already-processed
        trials in *trials*, compute per-stance normalised curves, aggregate
        into duration/FPA-binned averages (exactly as PrepForDeviationLearning
        does), save to pkl, and return the result dict.
    """
    if not cfg["DO_DEVIATION_LEARNING_PREP"]:
        return None

    pkl_path = Path(cfg["DEVIATION_METRICS_PKL_PATH"])

    if cfg["DEVIATION_LOAD_AVERAGES_FROM_FILE"]:
        # ── Load from existing pkl ────────────────────────────────
        if not pkl_path.exists():
            print(f"  [Deviation] pkl not found: {pkl_path}. Skipping deviation prep.")
            return None
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)

        # Ensure sub-structure is in the expected {'mean': array, 'n': int} format
        def _ensure_structured(data_dict):
            if not isinstance(data_dict, dict):
                return {}
            out = {}
            for k, v in data_dict.items():
                out[k] = v if (isinstance(v, dict) and "mean" in v) else {"mean": v, "n": 1}
            return out

        final_output = saved.get("final_output", {})
        if final_output:
            for side in ("Right", "Left"):
                for key in ("grf_by_duration", "cop_by_fpa", "moment_by_duration"):
                    if key in final_output.get(side, {}):
                        final_output[side][key] = _ensure_structured(final_output[side][key])

        print(f"  [Deviation] Loaded averages from {pkl_path}")
        return {
            "final_output":           final_output,
            "global_median_duration": saved.get("global_median_duration", 50),
            "global_median_fpa":      saved.get("global_median_fpa", 0.0),
        }

    else:
        # ── Build averages at runtime from processed trials ───────
        if not trials:
            print("  [Deviation] No trials list provided — cannot build averages.")
            return None
        return _build_deviation_averages_from_trials(trials, cfg)


def _available_memory_gb() -> float | None:
    """Best-effort available RAM in GiB (prefers Linux MemAvailable)."""
    gib = 1024.0 ** 3

    # 1) Linux host-level available memory (includes reclaimable cache).
    mem_available_gb = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Format: "MemAvailable:   <kB> kB"
                    parts = line.split()
                    kb = float(parts[1])
                    mem_available_gb = kb / (1024.0 ** 2)
                    break
    except Exception:
        mem_available_gb = None

    # 2) Respect cgroup memory limit if present (container/session caps).
    # cgroup v2: memory.max
    # cgroup v1: memory.limit_in_bytes
    cgroup_limit_gb = None
    cgroup_current_gb = None

    for limit_path in (
        "/sys/fs/cgroup/memory.max",                   # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes", # cgroup v1
    ):
        try:
            with open(limit_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw and raw != "max":
                limit_bytes = float(raw)
                # Ignore absurdly large "no real limit" sentinels.
                if limit_bytes > 0 and limit_bytes < (1 << 60):
                    cgroup_limit_gb = limit_bytes / gib
                    break
        except Exception:
            continue

    for current_path in (
        "/sys/fs/cgroup/memory.current",               # cgroup v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes", # cgroup v1
    ):
        try:
            with open(current_path, "r", encoding="utf-8") as f:
                used_bytes = float(f.read().strip())
            if used_bytes >= 0:
                cgroup_current_gb = used_bytes / gib
                break
        except Exception:
            continue

    cgroup_available_gb = None
    if cgroup_limit_gb is not None and cgroup_current_gb is not None:
        cgroup_available_gb = max(0.0, cgroup_limit_gb - cgroup_current_gb)

    # If both are known, use the tighter bound.
    candidates = [x for x in (mem_available_gb, cgroup_available_gb) if x is not None]
    if candidates:
        return min(candidates)

    # Fallback for non-Linux or restricted environments.
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / gib
    except Exception:
        return None


def choose_worker_count(requested_workers: int,
                        min_ram_gb_per_worker: float = 4.0) -> int:
    """
    Cap requested workers against currently available memory.
    """
    if requested_workers <= 1:
        return 1
    avail_gb = _available_memory_gb()
    if avail_gb is None or min_ram_gb_per_worker <= 0:
        return requested_workers
    max_by_ram = max(1, int(avail_gb // min_ram_gb_per_worker))
    return max(1, min(requested_workers, max_by_ram))


def batch_process_all_subjects(data_root:      Path,
                                cfg:            dict,
                                subject_filter: str | None = None,
                                subject_list:   list[str] | None = None,
                                dry_run:        bool = False):
    """Discover and process all trials under data_root.

    Parallelism strategy:
        CPU workers use multiprocessing "spawn". Each worker starts a fresh
        interpreter and initializes its own JAX/MuJoCo CPU runtime. This is
        intentionally more expensive at startup than Linux "fork", but avoids
        inheriting JAX thread-pool locks, which can leave workers permanently
        blocked in futex_wait_queue.

        NUM_WORKERS=1  → one process; required for CUDA and valid for CPU
        NUM_WORKERS>1  → spawn-based multiprocessing (CPU only)

    GPU mode uses one host process while JAX/MJX parallelizes inverse dynamics
    over frames, bodies, and degrees of freedom on the GPU.
    """
    requested_workers = int(cfg["NUM_WORKERS"])
    backend = jax.default_backend()
    requested_device = str(cfg.get("COMPUTE_DEVICE", "cpu"))
    if requested_device == "gpu" and backend != "gpu":
        raise RuntimeError(
            f"--device gpu requested, but JAX initialized backend {backend!r}. "
            "Verify nvidia-smi and the CUDA-enabled JAX installation."
        )
    print(
        f"[Compute] requested={requested_device} backend={backend} "
        f"devices={jax.devices()}"
    )
    if backend == "gpu":
        n_workers = 1
        if requested_workers != 1:
            print(
                f"[Compute] Limiting workers {requested_workers} -> 1: "
                "multiple CUDA host processes are unsafe here. "
                "MJX kernels remain parallel on the GPU."
            )
    else:
        n_workers = choose_worker_count(
            requested_workers,
            min_ram_gb_per_worker=float(cfg.get("MIN_RAM_GB_PER_WORKER", 4.0)),
        )
    if backend == "cpu" and n_workers > 1:
        print(
            f"[Compute] multiprocessing=spawn workers={n_workers} "
            f"threads_per_worker={_CPU_THREADS_PER_WORKER or 'runtime-default'}"
        )

    output_dir_name = get_output_dir_name(cfg)
    output_dir_label = get_output_dir_label(cfg)
    required_dir_name = "MoCap" if bool(cfg.get("OC_Mocap", False)) else None
    trials = discover_all_trials(
        data_root,
        subject_filter,
        subject_list,
        required_dir_name=required_dir_name,
        cfg=cfg,
    )
    print(f"Found {len(trials)} trials under {data_root} for output mode {output_dir_label}")

    if dry_run:
        print("Dry-run mode — no processing performed.")
        for sp, tp in trials:
            print(f"  {sp.name}/{tp.name}")
        return

    # Prepare model XMLs once in the parent process to avoid race conditions
    # when multiple workers process trials from the same subject.
    subjects_seen: set[Path] = {Path(sp) for sp, _ in trials}
    for subj_path in sorted(subjects_seen):
        prep_cfgs = []
        if bool(cfg.get("OpenCapVal", False)):
            for source in ("MoCap", "Video"):
                source_cfg = dict(cfg)
                source_cfg["OpenCapVal"] = False
                source_cfg["OPENCAPVAL_SOURCE"] = source
                prep_cfgs.append(source_cfg)
        else:
            prep_cfgs.append(cfg)
        for prep_cfg in prep_cfgs:
            try:
                _ = resolve_subject_model_xml(subj_path, prep_cfg)
            except Exception as e:
                source = prep_cfg.get("OPENCAPVAL_SOURCE")
                label = f" {source}" if source else ""
                print(f"  [Model] ⚠  Could not prepare{label} model for {subj_path.name}: {e}")

    deviation_data = load_deviation_data(cfg, trials=trials)
    if backend != "gpu" and n_workers < requested_workers:
        avail_gb = _available_memory_gb()
        if avail_gb is not None:
            print(f"[Memory] Limiting workers {requested_workers} -> {n_workers} "
                  f"(available RAM: {avail_gb:.1f} GiB, "
                  f"target per worker: {cfg['MIN_RAM_GB_PER_WORKER']:.1f} GiB)")
        else:
            print(f"[Memory] Limiting workers {requested_workers} -> {n_workers}")

    # Build picklable arg tuples (Path objects are picklable)
    work_args = [(sp, tp, cfg, deviation_data) for sp, tp in trials]

    from tqdm import tqdm

    results = []
    if n_workers > 1:
        import multiprocessing
        # Never fork after importing/initializing JAX. A forked child inherits
        # synchronization primitives without the parent runtime's background
        # threads and can deadlock as soon as it enters a compiled operation.
        # Spawn pays a one-time import/device-init cost per worker but gives
        # every worker a valid, independent CPU runtime.
        ctx = multiprocessing.get_context("spawn")
        # maxtasksperchild recycles each worker after this many trials so
        # that JAX JIT caches, MuJoCo model handles, and numpy temporaries
        # accumulated across trials are fully released back to the OS.
        # Tune this value lower if RAM is very tight (e.g. 2–4).
        # NOTE: multiprocessing.Pool.maxtasksperchild works on Python 2.7+;
        # ProcessPoolExecutor's equivalent (max_tasks_per_child) requires 3.12+.
        max_tasks = int(cfg.get("MAX_TASKS_PER_CHILD", 8))
        with ctx.Pool(processes=n_workers, maxtasksperchild=max_tasks) as pool:
            for result in tqdm(pool.imap(_worker, work_args),
                               total=len(work_args), desc=f"ProcessData[{output_dir_label}]"):
                results.append(result)
                if not result.get("success"):
                    tqdm.write(f"  ✗ {result['id']}: {result.get('error', '?')}")
                elif result.get("skipped"):
                    tqdm.write(f"  ↷ {result['id']} (skipped)")
                else:
                    tqdm.write(f"  ✓ {result['id']}  ({result.get('n_frames','?')} frames)")
    else:
        for sp, tp in tqdm(trials, desc=f"ProcessData[{output_dir_label}]"):
            result = process_single_trial(sp, tp, cfg, deviation_data)
            results.append(result)
            if not result.get("success"):
                tqdm.write(f"  ✗ {result['id']}: {result.get('error', '?')}")
            elif result.get("skipped"):
                tqdm.write(f"  ↷ {result['id']} (skipped)")
            else:
                tqdm.write(f"  ✓ {result['id']}  ({result.get('n_frames','?')} frames)")

    # ── Summary ──────────────────────────────────────────────────
    ok      = [r for r in results if r.get("success")]
    skipped = [r for r in ok      if r.get("skipped")]
    failed  = [r for r in results if not r.get("success")]

    print(f"\n{'='*60}")
    print(f"  Processed : {len(ok) - len(skipped)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("\nFailed trials:")
        for r in failed:
            print(f"  {r['id']}: {r.get('error', '?')}")
    print("="*60)

    if bool(cfg.get("OnlyProcessNoised", False)):
        print("\n  [OnlyProcessNoised] Rebuilt `_noised` bundles only; skipping PatientSize, column-mask, and clean post-processing passes.")
        return

    # ── Post-pass 1: PatientSize.npy per subject ─────────────────
    # Collect the set of subject directories that had at least one trial
    # attempted (success or failure — we still want the anthropometrics).
    print(f"\n{'='*60}")
    print("  Post-processing: PatientSize.npy")
    subjects_seen: set[Path] = {Path(sp) for sp, _ in trials}
    for subj_path in sorted(subjects_seen):
        try:
            size_cfg = cfg
            if bool(cfg.get("OpenCapVal", False)):
                size_cfg = dict(cfg)
                size_cfg["OpenCapVal"] = False
                size_cfg["OPENCAPVAL_SOURCE"] = "MoCap"
            xml_for_subject = resolve_subject_model_xml(subj_path, size_cfg)
        except Exception as e:
            print(f"  [PatientSize] ⚠  No valid XML for {subj_path.name} — skipping ({e})")
            continue
        patient_size = compute_patient_size(xml_for_subject)
        out_path = subj_path / "PatientSize.npy"
        np.save(out_path, patient_size)
        print(f"  [PatientSize] ✓ {subj_path.name}: "
              f"tibia={patient_size[0]:.3f}m  femur={patient_size[1]:.3f}m  "
              f"foot={patient_size[2]:.3f}m  pelvis_w={patient_size[3]:.3f}m")

    # ── Post-pass 2: normalize acceleration-input schema ─────────
    # Only run if at least one trial was actually processed (not all skipped).
    n_newly_processed = len(ok) - len(skipped)
    if n_newly_processed > 0:
        print(f"\n{'='*60}")
        print("  Post-processing: acc_inputs schema normalization")
        for target_output_dir_name in get_output_dir_names(cfg):
            compute_and_apply_column_masks(
                data_root=data_root,
                subject_filter=subject_filter,
                subject_list=subject_list,
                output_dir_name=target_output_dir_name,
            )
    else:
        print("\n  [ColumnMask] All trials were skipped — column masks not recomputed.")
    print("="*60)

    if bool(cfg.get("RUN_MISSSTEP_POSTPROCESS", True)):
        run_missing_step_postprocess(
            data_root=data_root,
            cfg=cfg,
            subject_filter=subject_filter,
            subject_list=subject_list,
        )

    if bool(cfg.get("RUN_CALC_FRAME_POSTPROCESS", True)):
        run_calc_frame_postprocess(
            data_root=data_root,
            cfg=cfg,
            subject_filter=subject_filter,
            subject_list=subject_list,
        )

    if bool(cfg.get("RUN_COP_OUTLIER_POSTPROCESS", True)):
        run_cop_outlier_postprocess(
            data_root=data_root,
            cfg=cfg,
            subject_filter=subject_filter,
            subject_list=subject_list,
        )


POS_INPUT_COLUMN_NAMES = tuple(
    name for name in POS_COLUMNS
    if name not in {"pelvis_tx", "pelvis_ty", "pelvis_tz", "mtp_angle_r", "mtp_angle_l"}
)
VEL_ACC_INPUT_COLUMN_NAMES = tuple(
    name for name in POS_COLUMNS
    if name not in {"mtp_angle_r", "mtp_angle_l"}
)
POS_INPUT_COLUMN_IDXS = tuple(POS_COLUMNS.index(name) for name in POS_INPUT_COLUMN_NAMES)
VEL_ACC_INPUT_COLUMN_IDXS = tuple(POS_COLUMNS.index(name) for name in VEL_ACC_INPUT_COLUMN_NAMES)


def build_pos_inputs_without_mtp(pos: np.ndarray) -> np.ndarray:
    """Build the traceable 18-column position schema, retaining both knees."""
    pos = np.asarray(pos)
    if pos.ndim != 2:
        raise ValueError(f"pos must be 2D, got shape {pos.shape}")
    if pos.shape[1] == len(POS_INPUT_COLUMN_NAMES):
        return pos
    if pos.shape[1] != len(POS_COLUMNS):
        raise ValueError(
            f"pos must have {len(POS_COLUMNS)} raw columns or "
            f"{len(POS_INPUT_COLUMN_NAMES)} processed columns; got shape {pos.shape}"
        )
    return pos[:, POS_INPUT_COLUMN_IDXS]


def build_vel_inputs_without_mtp(vel: np.ndarray) -> np.ndarray:
    """Build the traceable 21-column velocity schema, retaining both knees."""
    vel = np.asarray(vel)
    if vel.ndim != 2:
        raise ValueError(f"vel must be 2D, got shape {vel.shape}")
    if vel.shape[1] == len(VEL_ACC_INPUT_COLUMN_NAMES):
        return vel
    if vel.shape[1] != len(POS_COLUMNS):
        raise ValueError(
            f"vel must have {len(POS_COLUMNS)} raw columns or "
            f"{len(VEL_ACC_INPUT_COLUMN_NAMES)} processed columns; got shape {vel.shape}"
        )
    return vel[:, VEL_ACC_INPUT_COLUMN_IDXS]


def build_acc_inputs_without_mtp(accel: np.ndarray) -> np.ndarray:
    """Build the traceable 21-column acceleration schema, retaining both knees."""
    if accel.ndim != 2:
        raise ValueError(f"accel must be 2D, got shape {accel.shape}")
    if accel.shape[1] == len(VEL_ACC_INPUT_COLUMN_NAMES):
        return accel
    if accel.shape[1] != len(POS_COLUMNS):
        raise ValueError(
            f"accel must have {len(POS_COLUMNS)} raw columns or "
            f"{len(VEL_ACC_INPUT_COLUMN_NAMES)} processed columns; got shape {accel.shape}"
        )
    return accel[:, VEL_ACC_INPUT_COLUMN_IDXS]


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified MJX inverse-dynamics pipeline – saves to ProcessedData/ or MoCap/")
    parser.add_argument("--data-root",  default=CONFIG["DATA_ROOT"],
                        help="Root dataset directory")
    parser.add_argument("--subject",    default=None,
                        help="Process only this subject (exact directory name)")
    parser.add_argument("--subjects",   default=CONFIG["SUBJECTS_TO_PROCESS"],
                        help="Comma-separated list of subject names, e.g. 6GC,S8,S_GAH_1")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Discover trials only, no processing")
    parser.add_argument("--workers",    type=int, default=CONFIG["NUM_WORKERS"],
                        help="Number of spawn-based trial workers (CPU only; CUDA uses one process)")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default=_REQUESTED_COMPUTE_DEVICE,
        help=(
            "JAX compute device. GPU maps to CUDA and automatically uses one "
            "host process; CPU multiprocessing uses fresh spawned runtimes."
        ),
    )
    parser.add_argument("--only-new",   action="store_true",
                        help="Skip trials that already have the target output files")
    parser.add_argument("--deviation",  action="store_true", default = CONFIG["DO_DEVIATION_LEARNING_PREP"],
                        help="Enable deviation-learning reconstruction")
    parser.add_argument("--UseNoised", action="store_true", default=CONFIG["UseNoised"],
                        help="Also build a `_noised` prediction bundle from Pos_noised/Vel_noised/Accel_noised while keeping clean Pos/Vel/Accel outputs as ground truth.")
    parser.add_argument("--no-UseNoised", dest="UseNoised", action="store_false",
                        help="Use clean Pos/Vel/Accel inputs directly and do not look for Pos_noised/Vel_noised/Accel_noised.")
    parser.add_argument("--OnlyProcessNoised", action="store_true", default=CONFIG["OnlyProcessNoised"],
                        help="Only rebuild the `_noised` prediction bundle from Pos_noised/Vel_noised/Accel_noised and leave existing clean ProcessedData files untouched. Implies --UseNoised.")
    parser.add_argument("--OnlyProcessOverGround", action="store_true", default=CONFIG["OnlyProcessOverGround"],
                        help="Load raw Pos.npy first and only process trials inferred to be overground (non-treadmill).")
    parser.add_argument("--OC_Mocap", action="store_true",
                        help="Run the standard ProcessedData pipeline and, in parallel, a MoCap-input variant using Trial/MoCap kinematics + Trial/Motion forces saved to Trial/MoCap")
    parser.add_argument("--OpenCapVal", action="store_true",
                        help="Process an OpenCapWalkingTrunkSwaySubjects-style dataset: for each trial run the pipeline "
                             "twice — MoCap/Motion (kinematics+forces, MyosuiteModel_MoCap.xml) -> MoCap/ProcessedData, then "
                             "Video/Motion (MyosuiteModel_Video.xml) -> Video/ProcessedData. Only zero-force edge trimming is applied.")
    parser.add_argument("--no-kinematics-filtering", action="store_true",
                        help="Disable the 6 Hz Butterworth filter on Pos/Vel/Accel.")
    parser.add_argument("--no-calc-frame-post", action="store_true",
                        help="Disable the calc-frame / FPA / toes post-pass.")
    parser.add_argument("--no-missstep-post", action="store_true",
                        help="Disable the miss-step cleanup post-pass.")
    parser.add_argument("--no-cop-outlier-post", action="store_true",
                        help="Disable the COP outlier cleanup post-pass.")
    parser.add_argument("--COP_EdgeHold", dest="COP_EdgeHold", action="store_true",
                        default=CONFIG["COP_EdgeHold"],
                        help="Use edge-hold padding for COP segment filtfilt instead of zero padding.")
    parser.add_argument("--no-COP_EdgeHold", dest="COP_EdgeHold", action="store_false",
                        help="Use zero padding for COP segment filtfilt instead of edge-hold padding.")
    parser.add_argument("--use-nofilter-grf-for-torque", dest="USE_NOFILTER_GRF_FOR_TORQUE",
                        action="store_true", default=CONFIG["USE_NOFILTER_GRF_FOR_TORQUE"],
                        help="Use trimmed GRF_NoFilt_Trimmed.npy for qfrc_grf/ID_GT_MJX force terms while keeping cleaned GRF for GRF-dependent processing decisions.")
    parser.add_argument("--no-use-nofilter-grf-for-torque", dest="USE_NOFILTER_GRF_FOR_TORQUE",
                        action="store_false",
                        help="Use GRF_Cleaned.npy for qfrc_grf/ID_GT_MJX force terms.")
    parser.add_argument("--move-bad-cop-trials", action="store_true",
                        help="Move trials above the configured bad-COP threshold into the bad-trials root.")
    parser.add_argument("--use-fixed-models", dest="UsedFIXEDModels", action="store_true",
                        default=CONFIG["UsedFIXEDModels"],
                        help="Prefer existing MyosuiteModel_FIXED.xml files when available.")
    parser.add_argument("--rebuild-fixed-models", dest="UsedFIXEDModels", action="store_false",
                        help="Regenerate MyosuiteModel_FIXED.xml files from MyosuiteModel.xml before processing.")
    parser.add_argument("--DontUseFixed", action="store_true", default=CONFIG["DontUseFixed"],
                        help="Use raw MyosuiteModel.xml instead of MyosuiteModel_FIXED.xml for non-OpenCapVal processing.")
    parser.add_argument("--rescale-models-to-estimated-mass", dest="RescaleModelsToEstimatedMass",
                        action="store_true", default=CONFIG["RescaleModelsToEstimatedMass"],
                        help="Force-regenerate the fixed model XML from the raw model, then rescale that generated "
                             "model's inertial masses and inertias to Patient_MD.json Mass_kg using "
                             "scripts/rescale_models_to_estimated_mass.py. This avoids reusing older scaled fixed "
                             "models that may have stripped upper-body DOFs, while preserving joints, geometry, "
                             "upper-body DOFs, and the knee-fixed model structure from the current updater.")
    # --- Filter-ablation knobs (default None/unspecified -> canonical behavior) ---
    parser.add_argument("--filter-cutoff", type=float, default=None,
                        help="Override FILTER_CUTOFF_HZ (global kinematics Butterworth cutoff, Hz).")
    parser.add_argument("--filter-order", type=int, default=None,
                        help="Override FILTER_ORDER (Butterworth order).")
    parser.add_argument("--filter-channels", default=None,
                        help="Comma-separated subset of pos,vel,accel to filter (others left raw). "
                             "Omit to follow ENABLE_KINEMATICS_FILTERING for all channels.")
    parser.add_argument("--os-filtering", dest="OS_Filtering", action="store_true",
                        default=CONFIG["OS_Filtering"],
                        help="Also compute MJX inverse dynamics with OpenSim-style GCVSpline "
                             "vel/accel and save to ProcessedData/OpenSimFiltering/.")
    parser.add_argument("--no-os-filtering", dest="OS_Filtering", action="store_false",
                        help="Disable the OpenSim-filtering MJX ID pass.")
    parser.add_argument("--output-subdir", default=None,
                        help="Write outputs to trial/<name> instead of trial/ProcessedData (ablation).")
    parser.add_argument("--trials-file", default=None,
                        help="JSON file with a list of 'Subject/Trial' ids to restrict processing to.")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["NUM_WORKERS"]               = args.workers
    cfg["COMPUTE_DEVICE"]            = args.device
    cfg["ONLY_PROCESS_NEW"]          = args.only_new
    cfg["DO_DEVIATION_LEARNING_PREP"] = args.deviation
    cfg["UseNoised"]                 = bool(args.UseNoised)
    cfg["OnlyProcessNoised"]         = bool(args.OnlyProcessNoised)
    if cfg["OnlyProcessNoised"]:
        cfg["UseNoised"] = True
        cfg["DO_DEVIATION_LEARNING_PREP"] = False
    cfg["OnlyProcessOverGround"]     = bool(args.OnlyProcessOverGround)
    cfg["OC_Mocap"]                  = args.OC_Mocap
    cfg["OS_Filtering"]              = bool(args.OS_Filtering)
    if args.no_kinematics_filtering:
        cfg["ENABLE_KINEMATICS_FILTERING"] = False
    # Filter-ablation overrides
    if args.filter_cutoff is not None:
        cfg["FILTER_CUTOFF_HZ"] = float(args.filter_cutoff)
    if args.filter_order is not None:
        cfg["FILTER_ORDER"] = int(args.filter_order)
    if args.filter_channels is not None:
        wanted = {c.strip().lower() for c in args.filter_channels.split(",") if c.strip()}
        for name, key in (("pos", "FILTER_POS"), ("vel", "FILTER_VEL"), ("accel", "FILTER_ACCEL")):
            cfg[key] = name in wanted
    if args.output_subdir:
        cfg["OUTPUT_SUBDIR_NAME"] = str(args.output_subdir)
    if args.trials_file:
        with open(args.trials_file) as _tf:
            _payload = json.load(_tf)
        _ids = _payload.get("all_ids", _payload) if isinstance(_payload, dict) else _payload
        cfg["TRIALS_TO_PROCESS"] = list(_ids)
    cfg["OpenCapVal"] = bool(args.OpenCapVal)
    if cfg["OpenCapVal"]:
        # Real OpenCap data: no synthetic noised bundle.
        cfg["UseNoised"] = False
        cfg["OnlyProcessNoised"] = False
        # Trimming: keep ONLY the leading/trailing zero-force (no-GRF) edge trim.
        cfg["ENABLE_GRF_TRIM"] = True
        cfg["TrimGRFMissSteps"] = False
        cfg["TRIM_TO_DOUBLE_SUPPORT"] = False
        cfg["TRIM_WEAK_EDGE_STANCES"] = False
        # Post-processing: normal bundle minus trimming — calc-frame + COP-outlier on,
        # miss-step (which trims) off.
        cfg["RUN_CALC_FRAME_POSTPROCESS"] = True
        cfg["RUN_COP_OUTLIER_POSTPROCESS"] = True
        cfg["RUN_MISSSTEP_POSTPROCESS"] = False
    if cfg["OC_Mocap"]:
        cfg["UseNoised"] = False
        cfg["OnlyProcessNoised"] = False
    if args.no_calc_frame_post:
        cfg["RUN_CALC_FRAME_POSTPROCESS"] = False
    if args.no_missstep_post:
        cfg["RUN_MISSSTEP_POSTPROCESS"] = False
    if args.no_cop_outlier_post:
        cfg["RUN_COP_OUTLIER_POSTPROCESS"] = False
    cfg["COP_EdgeHold"] = bool(args.COP_EdgeHold)
    cfg["USE_NOFILTER_GRF_FOR_TORQUE"] = bool(args.USE_NOFILTER_GRF_FOR_TORQUE)
    cfg["COP_OUTLIER_MOVE_BAD_TRIALS"] = bool(args.move_bad_cop_trials)
    cfg["UsedFIXEDModels"] = bool(args.UsedFIXEDModels)
    cfg["DontUseFixed"] = bool(args.DontUseFixed)
    cfg["RescaleModelsToEstimatedMass"] = bool(args.RescaleModelsToEstimatedMass)
    if cfg["DontUseFixed"]:
        cfg["UsedFIXEDModels"] = False

    # Build subject list: --subjects takes precedence over --subject.
    # args.subjects may be a list (when the CONFIG default is a list) or a
    # comma-separated string (when passed on the command line), or None.
    raw = args.subjects
    if isinstance(raw, list):
        subject_list = [s.strip() for s in raw if str(s).strip()] or None
    elif isinstance(raw, str) and raw.strip():
        subject_list = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        subject_list = None
    subject_filter = args.subject if not subject_list else None

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).parent / data_root

    batch_process_all_subjects(
        data_root      = data_root,
        cfg            = cfg,
        subject_filter = subject_filter,
        subject_list   = subject_list,
        dry_run        = args.dry_run,
    )
