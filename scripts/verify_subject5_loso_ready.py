#!/usr/bin/env python3
"""Verify OpenCapSubjects_Filt/subject5 is ready for LOSO/inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "TransformerFinal"))

from data_loader import TrialDataLoader, load_single_trial  # noqa: E402
from train import discover_all_trials  # noqa: E402

ROOT = REPO / "OpenCapSubjects_Filt"
SUBJECT = ROOT / "subject5"

REQUIRED_PROCESSED = [
    "pos_inputs.npy",
    "vel_inputs.npy",
    "acc_inputs.npy",
    "pelvis_rot_matrix.npy",
    "pos_mjx.npy",
    "qvel_mjx.npy",
    "qacc_mjx.npy",
    "WorldToGroundAlignedCalcnRotation.npy",
    "Jacobian.npy",
    "forwardVel.npy",
    "COM_l.npy",
    "COM_r.npy",
    "Height_m.npy",
    "Mass_kg.npy",
    "ID_GT_MJX.npy",
    "qfrc_inverse.npy",
]
REQUIRED_MOCAP_GT = [
    "qfrc_inverse.npy",
    "contactBoolean.npy",
    "GRF_NoFilt_Trimmed.npy",
    "GRF_Cleaned.npy",
    "Moment_Cleaned.npy",
    "COP_CalcFrame_GroundAligned.npy",
    "COP_CalcFrame_GroundAligned_GRFNorm.npy",
    "WorldToGroundAlignedCalcnRotation.npy",
    "Mass_kg.npy",
    "Height_m.npy",
    "ID_GT_MJX.npy",
]
SUBJECT_FILES = [
    "PatientSize.npy",
    "Patient_MD.json",
    "MyosuiteModel_FIXED.xml",
]


def main() -> int:
    trials = discover_all_trials(str(ROOT), refresh_cache=False)
    s5 = [t for t in trials if t["subject"] == "subject5"]
    print(f"discovered subject5 trials: {len(s5)}")
    for t in s5:
        print(f"  {t['trial_name']} len={t.get('length')}")

    print("\n=== File checklist ===")
    all_ok = True
    miss_subj = [f for f in SUBJECT_FILES if not (SUBJECT / f).exists()]
    print(f"Subject-level missing: {miss_subj or 'none'}")
    all_ok &= not miss_subj

    for trial in sorted(SUBJECT.glob("Trial_*")):
        pd = trial / "ProcessedData"
        mc = trial / "MoCap"
        miss_p = [f for f in REQUIRED_PROCESSED if not (pd / f).exists()]
        miss_m = [f for f in REQUIRED_MOCAP_GT if not (mc / f).exists()]
        print(f"\n{trial.name}:")
        print(f"  ProcessedData missing: {miss_p or 'none'}")
        print(f"  MoCap GT missing: {miss_m or 'none'}")
        all_ok &= not miss_p and not miss_m

    print("\n=== load_single_trial smoke tests ===")
    for trial in sorted(SUBJECT.glob("Trial_*")):
        for src in ("processed", "mocap"):
            data = load_single_trial(
                trial,
                opencap_val=True,
                input_source=src,
                use_noised=False,
            )
            ok = data is not None
            print(f"  {trial.name} input_source={src}: {'OK' if ok else 'FAILED'}")
            if not ok:
                all_ok = False
            elif src == "processed":
                print(
                    f"    pos/vel/acc shapes: {data['pos'].shape}, "
                    f"{data['vel'].shape}, {data['acc'].shape}"
                )

    print("\n=== TrialDataLoader smoke test (LOSO-like) ===")
    loader = TrialDataLoader(
        s5,
        window_size=128,
        stride=64,
        batch_size=2,
        shuffle=False,
        trim_cop=True,
        deviation_learning=False,
        use_noised=False,
        noised_gt=False,
        predict_jacobian=False,
        opencap_val=True,
        input_source="processed",
        include_pelvis_euler=True,
        include_ankle_heights=True,
        include_jacobian_input=True,
        prediction_margin_frames=0,
        use_grf_norm_cop=False,
        drop_last=False,
    )
    print(f"total_windows={loader.total_windows} trials={len(loader.trial_window_counts)}")
    if loader.total_windows <= 0:
        all_ok = False
    else:
        batch = next(iter(loader))
        print(f"batch keys: {sorted(batch.keys())}")
        print(f"batch pos shape: {batch['pos'].shape}")

    # Compare fold eligibility with other subjects
    from loso_from_checkpoint import _discover_subject_trials, _build_loso_folds

    _trials, all_subjects, valid_subjects, subject_to_trials = _discover_subject_trials(ROOT)
    print("\n=== LOSO discovery ===")
    print(f"valid_subjects ({len(valid_subjects)}): {valid_subjects}")
    print(f"subject5 in valid_subjects: {'subject5' in valid_subjects}")
    if "subject5" in valid_subjects:
        print(f"subject5 trials: {[t['trial_name'] for t in subject_to_trials['subject5']]}")
    try:
        folds = _build_loso_folds(valid_subjects, subject_to_trials)
        s5_folds = [f for f in folds if f.get("held_out_subject") == "subject5"]
        print(f"LOSO folds with subject5 held out: {len(s5_folds)}")
    except Exception as exc:
        print(f"LOSO fold build error: {exc}")
        all_ok = False

    print(f"\nOVERALL: {'READY' if all_ok else 'NOT READY'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
