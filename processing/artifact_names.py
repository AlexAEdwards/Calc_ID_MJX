"""Names of the files a processed trial is expected to contain.

Extracted from ProcessData.py in REFACTOR_PLAN.md Stage 6. These live in their own
module rather than beside the code that reads them because several clusters need
them - trial I/O, trimming and provenance all refer to the same filenames, and a
shared constants module keeps those from having to import each other.

The three tuples are deliberately different lengths and are not interchangeable:

* ``NOISED_AUX_FILES_TO_COPY`` - everything worth mirroring into a noised bundle.
  Missing entries are skipped silently, so this is a best-effort list.
* ``NOISED_REQUIRED_BUNDLE_FILENAMES`` - the subset whose absence means the noised
  bundle is incomplete. This is what decides whether a trial has usable noised
  data at all.
* ``NOISED_STRICT_VALIDATION_FILENAMES`` - the required set plus the derived
  kinetics, used when a stronger guarantee is wanted.

Noised variants are the same names with ``NOISED_FILE_SUFFIX`` inserted before the
extension (``pos_inputs.npy`` -> ``pos_inputs_noised.npy``); build them with
``processing.trial_io._with_file_suffix`` rather than by hand.
"""

from __future__ import annotations


NOISED_FILE_SUFFIX = "_noised"


TRIMMING_TRACE_FILENAME = "Trimming_Traceability.json"


NOISED_AUX_FILES_TO_COPY = (
    "pos_inputs.npy",
    "vel_inputs.npy",
    "acc_inputs.npy",
    "pelvis_rot_matrix.npy",
    "pos_mjx.npy",
    "qvel_mjx.npy",
    "qacc_mjx.npy",
    "COP_Cleaned_Relative.npy",
    "forwardVel.npy",
    "ankle_heights.npy",
    "ankle_pos_r.npy",
    "ankle_pos_l.npy",
    "toes_pos_r.npy",
    "toes_pos_l.npy",
    "COM_r.npy",
    "COM_l.npy",
    "COM_Acc_Global.npy",
    "qfrc_inverse.npy",
    "Jacobian.npy",
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
)


NOISED_REQUIRED_BUNDLE_FILENAMES = (
    TRIMMING_TRACE_FILENAME,
    "pos_inputs.npy",
    "vel_inputs.npy",
    "acc_inputs.npy",
    "pelvis_rot_matrix.npy",
    "pos_mjx.npy",
    "qvel_mjx.npy",
    "qacc_mjx.npy",
    "WorldToGroundAlignedCalcnRotation.npy",
    "Jacobian.npy",
    "ankle_heights.npy",
    "COM_r.npy",
    "COM_l.npy",
    "COM_Acc_Global.npy",
    "forwardVel.npy",
    "Foot_ProgressionAngle.npy",
    "CalcnToFloor_AngleDeg.npy",
)


NOISED_STRICT_VALIDATION_FILENAMES = NOISED_REQUIRED_BUNDLE_FILENAMES + (
    "qfrc_inverse.npy",
    "COP_Cleaned_Relative.npy",
    "COP_CalcFrame_GroundAligned.npy",
    "COP_CalcFrame_GroundAligned_GRFNorm.npy",
)
