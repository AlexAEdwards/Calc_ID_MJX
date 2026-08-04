"""Build KineticVAEDataset/ - a clean, self-describing hand-off copy of TrustedDataSet_ByExperiment.

Keeps only what a downstream user needs, renames inconsistent files to a single
convention, and ships the schema/README material needed to interpret every array.

    # inspect the plan
    python scripts/data_prep/build_kinetic_vae_dataset.py --dest KineticVAEDataset

    # build it
    python scripts/data_prep/build_kinetic_vae_dataset.py --dest KineticVAEDataset --apply

    # fill in one experiment later (e.g. Hip_OA once inference finishes)
    python scripts/data_prep/build_kinetic_vae_dataset.py --dest KineticVAEDataset --apply \
        --experiments Hip_OA --force

Idempotent: trials that already exist at the destination are skipped unless
``--force``. Trials whose predictions have not been produced yet are copied
without a ``prediction/`` folder and flagged in ``dataset_index.csv``, so the
build can run before every experiment has been scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from paths import artifact, dataset, resolve as paths_resolve  # noqa: E402

from paths import REPO_ROOT
DEFAULT_SOURCE = dataset("TrustedDataSet_ByExperiment")
DEFAULT_ACCURACY = artifact("outputs") / "DirectTorque_LOEO_edge70" / "accuracy"

# --------------------------------------------------------------------------
# DOF conventions - derived, never hand-typed, so they cannot drift from
# ProcessData.py's builders.
# --------------------------------------------------------------------------
POS_COLUMNS = (
    "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)
DIRECT_TORQUE_NAMES = (
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_flexion_r",
    "knee_adduction_r", "ankle_flexion_r", "subtalar_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_flexion_l",
    "knee_adduction_l", "ankle_flexion_l", "subtalar_l",
)


def _pos_input_columns() -> List[str]:
    """Mirror ProcessData's 18-column knee-inclusive position schema."""
    excluded = {"pelvis_tx", "pelvis_ty", "pelvis_tz", "mtp_angle_r", "mtp_angle_l"}
    return [name for name in POS_COLUMNS if name not in excluded]


def _vel_input_columns() -> List[str]:
    """Mirror ProcessData's 21-column knee-inclusive velocity schema."""
    return [name for name in POS_COLUMNS if name not in {"mtp_angle_r", "mtp_angle_l"}]


# --------------------------------------------------------------------------
# File routing: source relative path -> (destination subfolder, new name)
# --------------------------------------------------------------------------
RAW_FILES = [
    "Pos.npy", "Vel.npy", "Accel.npy", "Pos_raw.npy",
    "COP.npy", "GRF.npy", "GRM.npy",
    # Kinematics and forces are captured on different clocks in several cohorts;
    # without Time_for_pos.npy the two streams cannot be aligned.
    "Time.npy", "Time_for_pos.npy", "treadmill_speed.npy",
    # Per-cohort data-quality signals (PD, Hip_OA): which foot a plate was assigned
    # to, and where the force trace is contaminated.
    "ContactMask.npy", "ContaminatedMask.npy", "ForceAssignmentConfidence.npy",
]
RAW_JSON = ["extraction_metadata.json", "motion_metadata.json", "force_plate_assignment.json"]

# Original upstream OpenSim/lab files, kept verbatim under raw/source/ for provenance.
# Names are cohort-specific by nature, so these are matched by extension.
SOURCE_SUFFIXES = (".mot", ".sto", ".trc", ".csv", ".forces", ".npz")

# OpenSim inverse dynamics stored alongside the raw motion (OpenCapVal only, which
# has no OpenSimResults/ folder). It is torque ground truth, so it belongs in kinetics/.
MOTION_KINETICS_FILES = ["OpenSim_ID.npy", "OpenSim_ID_recalculated.npy"]

# kinematics/ is *derived* (see build_cleaned_kinematics), not copied.
KINEMATICS_FILES: List[str] = []

# The 21-DOF uniform set: every DOF from lumbar down except the two MTP joints,
# in raw Pos.npy column order so raw/ and kinematics/ share one convention.
CLEANED_DOF = (
    "pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)
# Column order used when an MJX array stores only the independent DOFs
# (OpenCap conversions) rather than the full qpos vector.
MODEL_23_DOF = (
    "pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)

KINETICS_FILES = ["GRF_Cleaned.npy", "GRF_NoFilt_Trimmed.npy", "Moment_Cleaned.npy",
                  "contactBoolean.npy", "ID_GT_MJX.npy", "qfrc_inverse.npy",
                  "qfrc_grf_contribution.npy",
                  # COP representations kept: the model's target, the foot-relative
                  # form, and BackToWorld - which is the foot-origin lever arm needed
                  # to rebuild tau_grf (verified to 0.7% against qfrc_grf_contribution).
                  "COP_CalcFrame_GroundAligned.npy", "COP_Cleaned_Relative.npy",
                  "COP_CalcFrame_GroundAligned_BackToWorld.npy"]

MUJOCO_FILES = ["Jacobian.npy", "KneeToCOP_Vectors.npy",
                "pos_mjx.npy", "qvel_mjx.npy", "qacc_mjx.npy"]

LANDMARK_FILES = ["ankle_pos_r.npy", "ankle_pos_l.npy", "knee_pos_r.npy", "knee_pos_l.npy",
                  "toes_pos_r.npy", "toes_pos_l.npy", "tosPosition.npy",
                  "COM_r.npy", "COM_l.npy", "COM_Acc_Global.npy", "ankle_heights.npy",
                  "pelvis_rot_matrix.npy", "WorldToGroundAlignedCalcnRotation.npy",
                  "FootProgressionAngle.npy", "CalcnToFloor_AngleDeg.npy", "forwardVel.npy"]

PREDICTION_FILES = ["direct_torque_pred_percent_bwh.npy", "direct_torque_gt_percent_bwh.npy",
                    "direct_torque_pred_nm.npy", "direct_torque_gt_nm.npy",
                    # prediction_coverage: a prediction exists here.
                    # scoring_mask: score this frame (coverage minus the trial edges).
                    # evaluation_mask is the historical name and equals scoring_mask.
                    "prediction_coverage.npy", "scoring_mask.npy", "evaluation_mask.npy",
                    "direct_torque_names.json", "metrics.json"]

# OpenSim inverse-dynamics ground truth. Interactive html plots are excluded.
OPENSIM_FILES = ["inverse_dynamics.sto", "inverse_dynamics_no_external_loads.sto",
                 "inverse_dynamics_constraints_disabled.sto", "coordinates.mot",
                 "ground_reaction.mot", "external_loads.xml", "id_setup.xml",
                 "opensim_id_force_diagnostics.csv", "AccuracyMetrics.json"]

# Subject-level model files, in preference order -> normalized destination name.
OSIM_CANDIDATES = ["OpenSimModel.osim", "LaiUhlrich2022_scaled.osim",
                   "OpenSimScaled_MoCap.osim", "OpenSimScaled_Video.osim"]
MJC_CANDIDATES = ["MyosuiteModel_FIXED.xml", "MyosuiteModel_MoCap_FIXED.xml",
                  "MyosuiteModel_Video_FIXED.xml", "MyosuiteModel.xml"]


def _first_existing(directory: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = directory / n
        if p.exists():
            return p
    return None


def _copy_group(src_dir: Path, dst_dir: Path, names: List[str], apply: bool) -> Tuple[int, int]:
    """Copy the named files; returns (copied, bytes)."""
    copied = nbytes = 0
    for n in names:
        s = src_dir / n
        if not s.exists():
            continue
        if apply:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, dst_dir / n)
        copied += 1
        nbytes += s.stat().st_size
    return copied, nbytes


def mjx_qpos_layout(model_path: Path) -> Optional[Dict[str, Any]]:
    """Enumerate the MuJoCo qpos vector so pos_mjx/qvel_mjx columns are interpretable.

    ``pos_mjx`` width equals the model's ``nq`` for most subjects, but the OpenCap
    conversions store only the 23 independent DOFs. Recording nq alongside the
    names lets a consumer detect which convention a subject uses.
    """
    try:
        import mujoco
    except Exception:
        return None
    try:
        m = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception:
        return None
    qpos, qvel = [], []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        jtype = int(m.jnt_type[j])
        nq, nv = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}[jtype]
        for k in range(nq):
            qpos.append(name if nq == 1 else f"{name}[{k}]")
        for k in range(nv):
            qvel.append(name if nv == 1 else f"{name}[{k}]")
    return {
        "model_file": model_path.name,
        "nq": int(m.nq), "nv": int(m.nv), "njnt": int(m.njnt),
        "qpos_names": qpos, "qvel_names": qvel,
        "note": ("pos_mjx.npy columns follow qpos_names when its width == nq. "
                 "Some subjects (OpenCap conversions) instead store only the 23 "
                 "independent DOFs; compare the array width against nq to tell."),
    }


def build_subject(src_subj: Path, dst_subj: Path, apply: bool) -> Dict[str, Any]:
    info: Dict[str, Any] = {"models": {}, "bytes": 0}

    md_path = src_subj / "Patient_MD.json"
    meta: Dict[str, Any] = {}
    if md_path.exists():
        meta = json.loads(md_path.read_text())
    ps = src_subj / "PatientSize.npy"
    if ps.exists():
        try:
            import numpy as np
            meta["PatientSize_m"] = [float(x) for x in np.load(ps)]
            meta["PatientSize_note"] = ("Segment scale factors used by the MuJoCo model "
                                        "(see README). Height_m/Mass_kg above are the "
                                        "authoritative scalars; the per-frame Height_m.npy / "
                                        "Mass_kg.npy of the source dataset were constant and "
                                        "have been dropped.")
        except Exception:
            pass
    if apply:
        dst_subj.mkdir(parents=True, exist_ok=True)
        (dst_subj / "subject_metadata.json").write_text(json.dumps(meta, indent=2))

    models_dir = dst_subj / "models"
    osim = _first_existing(src_subj, OSIM_CANDIDATES)
    mjc = _first_existing(src_subj, MJC_CANDIDATES)
    if apply:
        models_dir.mkdir(parents=True, exist_ok=True)
    if osim:
        info["models"]["opensim"] = osim.name
        info["bytes"] += osim.stat().st_size
        if apply:
            shutil.copy2(osim, models_dir / "opensim_model.osim")
    if mjc:
        info["models"]["mujoco"] = mjc.name
        info["bytes"] += mjc.stat().st_size
        if apply:
            shutil.copy2(mjc, models_dir / "mujoco_model.xml")
        layout = mjx_qpos_layout(mjc)
        if layout:
            info["nq"] = layout["nq"]
            info["layout"] = layout
            if apply:
                (models_dir / "mjx_qpos_layout.json").write_text(json.dumps(layout, indent=2))

    geom = src_subj / "Geometry"
    if geom.is_dir():
        info["bytes"] += sum(f.stat().st_size for f in geom.rglob("*") if f.is_file())
        if apply and not (models_dir / "Geometry").exists():
            shutil.copytree(geom, models_dir / "Geometry")

    # The patella-stripped model used for OpenSim ID lives per-trial in the source;
    # it is identical across trials, so store it once here.
    for t in sorted(src_subj.glob("Trial_*")):
        cand = t / "OpenSimResults" / "OpenSimModel_NoPatel.osim"
        if cand.exists():
            info["bytes"] += cand.stat().st_size
            if apply:
                shutil.copy2(cand, models_dir / "opensim_model_no_patella.osim")
            break
    return info



def _mjx_column_names(width: int, layout: Optional[Dict[str, Any]], kind: str) -> Optional[List[str]]:
    """Column names for an MJX array of the given width.

    Most subjects store the full qpos/qvel vector, so the model's joint enumeration
    applies. OpenCap conversions instead store only the 23 independent DOFs against
    a 43-DOF model, so width is the discriminator, not the model.
    """
    if layout is None:
        return list(MODEL_23_DOF) if width == len(MODEL_23_DOF) else None
    names = layout.get("qpos_names" if kind == "pos" else "qvel_names") or []
    if width == len(names):
        return list(names)
    if width == len(MODEL_23_DOF):
        return list(MODEL_23_DOF)
    return None


def build_cleaned_kinematics(pd_dir: Path, dst_trial: Path, layout: Optional[Dict[str, Any]],
                             apply: bool) -> Tuple[int, Dict[str, Any]]:
    """Derive the uniform 21-DOF pos/vel/acc arrays from the MJX state.

    Sourced from pos_mjx/qvel_mjx/qacc_mjx rather than raw Pos/Vel/Accel because
    those are the arrays the torques were computed from, so the result is
    frame-aligned with kinetics/ and prediction/ by construction.
    """
    import numpy as np
    out: Dict[str, Any] = {}
    nbytes = 0
    for src_name, dst_name, kind in (("pos_mjx.npy", "pos_cleaned.npy", "pos"),
                                     ("qvel_mjx.npy", "vel_cleaned.npy", "vel"),
                                     ("qacc_mjx.npy", "acc_cleaned.npy", "vel")):
        src = pd_dir / src_name
        if not src.exists():
            continue
        arr = np.load(src)
        names = _mjx_column_names(arr.shape[1], layout, kind)
        if names is None or any(c not in names for c in CLEANED_DOF):
            out.setdefault("skipped", []).append(src_name)
            continue
        idx = [names.index(c) for c in CLEANED_DOF]
        sub = np.ascontiguousarray(arr[:, idx])
        nbytes += sub.nbytes
        if apply:
            (dst_trial / "kinematics").mkdir(parents=True, exist_ok=True)
            np.save(dst_trial / "kinematics" / dst_name, sub)
        out["n_frames"] = int(sub.shape[0])
        out["n_columns"] = int(sub.shape[1])
    return nbytes, out


def build_time_vector(src_trial: Path, dst_trial: Path, n_frames: Optional[int],
                      bounds: Optional[List[int]], apply: bool) -> int:
    """Write kinematics/time.npy so processed arrays carry their own timebase."""
    import numpy as np
    if not n_frames:
        return 0
    t = None
    raw_time = src_trial / "Motion" / "Time.npy"
    if raw_time.exists() and bounds and len(bounds) == 2:
        full = np.load(raw_time)
        lo, hi = int(bounds[0]), int(bounds[1])
        if 0 <= lo < hi <= len(full) and (hi - lo) == n_frames:
            t = full[lo:hi]
    if t is None and raw_time.exists():
        full = np.load(raw_time)
        if len(full) >= 2:
            dt = float(np.median(np.diff(full)))
            t = full[0] + np.arange(n_frames) * dt
    if t is None:
        t = np.arange(n_frames) / 100.0
    t = np.asarray(t, dtype=np.float64)
    if apply:
        (dst_trial / "kinematics").mkdir(parents=True, exist_ok=True)
        np.save(dst_trial / "kinematics" / "time.npy", t)
    return int(t.nbytes)


def build_trial(src_trial: Path, dst_trial: Path, apply: bool,
                layout: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"bytes": 0, "has_prediction": False, "has_opensim_id": False,
                           "n_frames": None}
    pd_dir = src_trial / "ProcessedData"

    motion = src_trial / "Motion"
    c, b = _copy_group(motion, dst_trial / "raw", RAW_FILES + RAW_JSON, apply)
    rec["bytes"] += b
    c, b = _copy_group(motion, dst_trial / "mujoco" / "kinetics", MOTION_KINETICS_FILES, apply)
    rec["bytes"] += b

    # Upstream source files (.mot/.sto/.trc/...), including any nested Motion/Raw/.
    if motion.is_dir():
        for p in sorted(motion.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            if "_noised" in p.name:
                continue
            out = dst_trial / "raw" / "source" / p.relative_to(motion)
            rec["bytes"] += p.stat().st_size
            rec["n_source_files"] = rec.get("n_source_files", 0) + 1
            if apply:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
    for sub, names in (("mujoco/kinetics", KINETICS_FILES),
                       ("mujoco", MUJOCO_FILES), ("mujoco/landmarks", LANDMARK_FILES)):
        c, b = _copy_group(pd_dir, dst_trial / sub, names, apply)
        rec["bytes"] += b

    inf = src_trial / "inference_results"
    if inf.is_dir() and (inf / "metrics.json").exists():
        c, b = _copy_group(inf, dst_trial / "prediction", PREDICTION_FILES, apply)
        rec["bytes"] += b
        rec["has_prediction"] = True

    osr = src_trial / "OpenSimResults"
    if osr.is_dir():
        c, b = _copy_group(osr, dst_trial / "opensim_id", OPENSIM_FILES, apply)
        if c:
            rec["bytes"] += b
            rec["has_opensim_id"] = True

    prov: Dict[str, Any] = {}
    for name in ("Trial_Processing_Information.json",):
        p = pd_dir / name
        if p.exists():
            try:
                prov["processing"] = json.loads(p.read_text())
            except Exception:
                pass
    for name in ("trial_metadata.json", "trial_manifest.json", "Visual_Trim_Application.json"):
        p = src_trial / name
        if p.exists():
            try:
                prov[name.replace(".json", "")] = json.loads(p.read_text())
            except Exception:
                pass
    try:
        import numpy as np
        pi = pd_dir / "pos_inputs.npy"
        if pi.exists():
            rec["n_frames"] = int(np.load(pi, mmap_mode="r").shape[0])
    except Exception:
        pass
    nb, kin = build_cleaned_kinematics(pd_dir, dst_trial, layout, apply)
    rec["bytes"] += nb
    if kin.get("n_frames"):
        rec["n_frames"] = kin["n_frames"]

    bounds = None
    proc = prov.get("processing") or {}
    for key in ("core_trim_bounds_motion_aligned", "grf_trim_bounds_motion_aligned"):
        if isinstance(proc.get(key), list) and len(proc[key]) == 2:
            bounds = proc[key]
            break
    rec["bytes"] += build_time_vector(src_trial, dst_trial, rec["n_frames"], bounds, apply)

    prov["n_frames"] = rec["n_frames"]
    prov["alignment"] = {
        "processed_n_frames": rec["n_frames"],
        "aligned_folders": ["kinematics", "mujoco", "mujoco/kinetics",
                            "mujoco/landmarks", "prediction"],
        "note": ("All arrays in the folders above share one frame index, 0..n_frames-1. "
                 "Only raw/ differs."),
        "raw_window_into_motion_aligned_grid": bounds,
        "raw_caveat": ("raw/ is NOT a slice of the processed arrays: kinematics are "
                       "low-pass filtered during processing, so values differ inside the "
                       "window. Use the window for index alignment only."),
        "time_vector": "kinematics/time.npy (seconds, same index as the aligned folders)",
        "prediction_masks": {
            "prediction_coverage.npy": ("True where a prediction exists. Inference windows the "
                                        "FULL trial, so this is now True everywhere and the "
                                        "prediction/GT arrays contain no NaN."),
            "scoring_mask.npy": ("True where the frame is included in the accuracy metrics: "
                                 "coverage minus the trial's first and last 20 frames. Those "
                                 "edge frames ARE predicted, they are just not scored."),
            "evaluation_mask.npy": "historical name, identical to scoring_mask.npy",
        },
        "cleaned_kinematics_columns": list(CLEANED_DOF),
        "cleaned_kinematics_source": ("derived from mujoco/pos_mjx.npy, qvel_mjx.npy and "
                                      "qacc_mjx.npy by DOF name, so it is frame-aligned with "
                                      "the torques by construction. Pelvis translations are "
                                      "in the MuJoCo model frame with treadmill travel "
                                      "accumulated - they are not lab coordinates."),
    }
    if apply:
        dst_trial.mkdir(parents=True, exist_ok=True)
        (dst_trial / "trial_metadata.json").write_text(json.dumps(prov, indent=2, default=str))
    return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--dest", required=True)
    p.add_argument("--experiments", default="", help="Comma-separated subset (default: all).")
    p.add_argument("--apply", action="store_true", help="Write files (default: dry run).")
    p.add_argument("--force", action="store_true", help="Rebuild trials that already exist.")
    p.add_argument("--accuracy_dir", default=str(DEFAULT_ACCURACY))
    p.add_argument("--limit_subjects", type=int, default=0, help="Pilot mode: N subjects per experiment.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = paths_resolve(args.source)
    dst = paths_resolve(args.dest)

    from TransformerFinal.experiment_groups import list_experiment_dirs
    exps = [p.name for p in list_experiment_dirs(src)]
    want = [e.strip() for e in args.experiments.split(",") if e.strip()]
    if want:
        unknown = sorted(set(want) - set(exps))
        if unknown:
            raise SystemExit(f"Unknown experiment(s) {unknown}. Available: {exps}")
        exps = want

    print(f"source : {src}")
    print(f"dest   : {dst}")
    print(f"experiments: {exps}\n")

    index_rows: List[Dict[str, Any]] = []
    total_bytes = 0
    for e in exps:
        subs = sorted(p for p in (src / e).iterdir() if p.is_dir())
        if args.limit_subjects:
            subs = subs[: args.limit_subjects]
        n_tr = n_pred = n_skipped = 0
        exp_bytes = 0
        for s in subs:
            dsub = dst / e / s.name
            sinfo = build_subject(s, dsub, args.apply)
            exp_bytes += sinfo["bytes"]
            for t in sorted(s.glob("Trial_*")):
                # Trial folders that never got processed are raw-only shells: no
                # kinematics, no kinetics, no prediction. Shipping them would leave
                # dead ends in the tree, so they are skipped and counted instead.
                if not (t / "ProcessedData" / "pos_inputs.npy").exists():
                    n_skipped += 1
                    continue
                dtrial = dsub / t.name
                if dtrial.exists() and not args.force:
                    continue
                rec = build_trial(t, dtrial, args.apply, sinfo.get("layout"))
                exp_bytes += rec["bytes"]
                n_tr += 1
                n_pred += int(rec["has_prediction"])
                index_rows.append({
                    "experiment": e, "subject": s.name, "trial": t.name,
                    "n_frames": rec["n_frames"], "has_prediction": rec["has_prediction"],
                    "has_opensim_id": rec["has_opensim_id"],
                    "mjx_nq": sinfo.get("nq"),
                    "path": f"{e}/{s.name}/{t.name}",
                })
        total_bytes += exp_bytes
        print(f"  {e:<16} {len(subs):>4} subj  {n_tr:>5} trials  "
              f"{n_pred:>5} with prediction   {exp_bytes/1e9:>6.2f} GB"
              + (f"   ({n_skipped} unprocessed skipped)" if n_skipped else ""))

    print(f"\nTOTAL: {len(index_rows)} trials, {total_bytes/1e9:.2f} GB")
    missing = sum(1 for r in index_rows if not r["has_prediction"])
    if missing:
        print(f"  note: {missing} trials have no prediction yet (flagged in dataset_index.csv)")

    if not args.apply:
        print("\nDry run - nothing written. Re-run with --apply.")
        return

    dst.mkdir(parents=True, exist_ok=True)
    idx = dst / "dataset_index.csv"
    write_header = not idx.exists()
    with idx.open("a" if idx.exists() else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(index_rows)

    acc = Path(args.accuracy_dir)
    if acc.is_dir() and not (dst / "accuracy").exists():
        shutil.copytree(acc, dst / "accuracy")

    write_schema(dst, src)
    write_example(dst)
    write_root_readme(dst, exps)
    for e in exps:
        write_experiment_readme(dst / e, e)
    print(f"\nWrote schema/, READMEs, dataset_index.csv and accuracy/ to {dst}")


# --------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------
def write_schema(dst: Path, src: Optional[Path] = None) -> None:
    d = dst / "schema"
    d.mkdir(parents=True, exist_ok=True)

    # The source dataset records the exact pos/vel/acc column layout the network
    # consumed. This dataset ships pos_cleaned instead of pos_inputs, so the file
    # is not needed to READ the data - it is here so results can be reproduced.
    if src is not None:
        model_schema = Path(src) / "Kinematic_Input_Schema.json"
        if model_schema.exists():
            shutil.copy2(model_schema, d / "model_input_schema.json")
    schema = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "units": {
            "angles": "radians", "translations": "metres", "forces": "newtons",
            "moments": "newton-metres", "time": "seconds", "gravity_m_s2": 9.8067,
        },
        "columns": {
            "raw/Pos.npy | Vel.npy | Accel.npy": list(POS_COLUMNS),
            "kinematics/pos_cleaned.npy | vel_cleaned.npy | acc_cleaned.npy": list(CLEANED_DOF),
            "kinematics/time.npy": "seconds; same frame index as every aligned folder",
            "mujoco/kinetics/ID_GT_MJX.npy | qfrc_inverse.npy | qfrc_grf_contribution.npy": list(POS_COLUMNS),
            "mujoco/kinetics/GRF_*.npy | Moment_Cleaned.npy | COP_CalcFrame_GroundAligned.npy":
                ["R_x", "R_y", "R_z", "L_x", "L_y", "L_z"],
            "mujoco/kinetics/COP_Cleaned_Relative.npy": ["R_x", "R_z", "L_x", "L_z"],
            "mujoco/kinetics/contactBoolean.npy": ["right_in_contact", "left_in_contact"],
            "prediction/direct_torque_*.npy": list(DIRECT_TORQUE_NAMES),
            "mujoco/pos_mjx.npy | qvel_mjx.npy | qacc_mjx.npy":
                "per-subject; see <Subject>/models/mjx_qpos_layout.json",
        },
        "normalization": {
            "COP": "divided by subject height (m)",
            "GRF": "divided by subject mass (kg)",
            "Moments": "divided by subject mass (kg)",
            "direct_torque_percent_bwh": "torque / (mass * height * 9.8067) * 100",
        },
    }
    (d / "dof_columns.json").write_text(json.dumps(schema, indent=2))
    (d / "coordinate_frames.md").write_text(COORDINATE_FRAMES_MD)
    (d / "file_reference.md").write_text(FILE_REFERENCE_MD)


COORDINATE_FRAMES_MD = """# Coordinate frames

## World frames

| Frame | Up axis | Used by |
|---|---|---|
| **OpenSim world** | +Y | `raw/` (`Pos`, `COP`, `GRF`, `GRM`), `opensim_id/` |
| **MuJoCo world** | +Z | `kinematics/pos_mjx`, `mujoco/`, everything MJX-derived |

Conversion applied during processing (OpenSim -> MuJoCo):

    [x, y, z]_opensim  ->  [x, -z, y]_mujoco

## Ground-aligned calcaneus frame

Most COP quantities are expressed per foot in a frame attached to the calcaneus but
levelled to the ground plane: the foot's forward axis is projected onto the ground
plane (normal `+Z` in MuJoCo), so the frame follows foot progression without
inheriting foot pitch or roll.

`landmarks/WorldToGroundAlignedCalcnRotation.npy` has shape **(T, 2, 3, 3)** -
index 1 is the foot (`0 = right`, `1 = left`), giving the world -> ground-aligned
rotation per frame. Apply its transpose to map a ground-aligned vector back to world.

Variants kept in this dataset:

- `kinetics/COP_CalcFrame_GroundAligned.npy` (T,6) - COP per foot in that frame,
  height-normalised. **This is the model's COP target.**
- `kinetics/COP_Cleaned_Relative.npy` (T,4) - COP relative to each foot, XZ only.

## Side/column convention

All 6-wide force/COP/moment arrays are `[R_x, R_y, R_z, L_x, L_y, L_z]`, right foot
first. 4-wide COP arrays drop the vertical component: `[R_x, R_z, L_x, L_z]`.

## Torque sign convention

Joint torques follow the OpenSim/MuJoCo coordinate sign for that DOF (e.g. positive
`hip_flexion_r` = flexion). `kinetics/ID_GT_MJX.npy` is the net joint torque from MJX
inverse dynamics; `qfrc_grf_contribution.npy` is the part attributable to external
(ground) forces, so `qfrc_inverse - qfrc_grf_contribution` isolates the rest.
"""

FILE_REFERENCE_MD = """# File reference

> `schema/model_input_schema.json` is copied verbatim from the source dataset and
> records the exact position/velocity/acceleration column layout the network was
> trained on (18 / 21 / 21 columns, knee-inclusive, no MTP). You do **not** need it
> to read this dataset - `kinematics/` ships the 21-DOF `pos_cleaned` arrays instead -
> but it is the definitive record for reproducing the model.

Every trial has the same layout. `T` is the trial's frame count (100 Hz).

## `raw/` - untouched source signals
| File | Shape | Notes |
|---|---|---|
| `Pos.npy` `Vel.npy` `Accel.npy` | (T, 23) | OpenSim IK coordinates, radians/metres |
| `COP.npy` `GRF.npy` `GRM.npy` | (T, 6) | OpenSim world frame, un-normalised |
| `Time.npy` | (T,) | seconds |
| `treadmill_speed.npy` | (1,) | m/s, 0 for overground |

Raw arrays are **longer** than the processed ones: processing trims to a clean,
fully-observed window.

## `kinematics/` - cleaned, trimmed model inputs
| File | Shape | Notes |
|---|---|---|
| `pos_cleaned.npy` `vel_cleaned.npy` `acc_cleaned.npy` | (T, 21) | **uniform across every subject and trial.** All DOF from lumbar down except the two MTP joints, in raw `Pos.npy` column order. Derived from the MJX state by DOF name, so frame-aligned with the torques by construction. |
| `time.npy` | (T,) | seconds; shared index with every aligned folder |

Pelvis translations here are in the **MuJoCo model frame with treadmill travel
accumulated** - they are not lab coordinates, and will not match `raw/Pos.npy`.

## `mujoco/` - MJX state and model-derived quantities
| File | Shape | Notes |
|---|---|---|
| `pos_mjx.npy` `qvel_mjx.npy` `qacc_mjx.npy` | (T, nq) | full MJX state; **width varies per subject**, see `models/mjx_qpos_layout.json` |
| `Jacobian.npy` | dict | `jacp`, `jacr`, `body_ids` |
| `KneeToCOP_Vectors.npy` | (T, 6) | knee-to-COP lever arm per foot |

## `mujoco/kinetics/` - ground truth
| File | Shape | Notes |
|---|---|---|
| `ID_GT_MJX.npy` | (T, 23) | net joint torque, N*m, MJX inverse dynamics |
| `qfrc_inverse.npy` | (T, 23) | generalised force, N*m |
| `qfrc_grf_contribution.npy` | (T, 23) | portion from external forces |
| `GRF_Cleaned.npy` `GRF_NoFilt_Trimmed.npy` | (T, 6) | filtered / unfiltered ground reaction force |
| `Moment_Cleaned.npy` | (T, 6) | free moment |
| `COP_CalcFrame_GroundAligned.npy` | (T, 6) | height-normalised COP |
| `COP_Cleaned_Relative.npy` | (T, 4) | foot-relative COP, XZ |
| `contactBoolean.npy` | (T, 2) | per-foot contact flag |

## `mujoco/landmarks/` - derived kinematic quantities
Joint-centre and COM positions, ankle heights, pelvis rotation (6-dim), the
ground-aligned rotation bundle, foot progression and calcaneus-to-floor angles,
forward velocity.

## `prediction/` - model output (leave-one-experiment-out)
| File | Shape | Notes |
|---|---|---|
| `direct_torque_pred_percent_bwh.npy` | (T, 14) | **model prediction**, %BW*H |
| `direct_torque_gt_percent_bwh.npy` | (T, 14) | matching ground truth |
| `direct_torque_pred_nm.npy` / `_gt_nm.npy` | (T, 14) | same in N*m |
| `prediction_coverage.npy` | (T,) bool | a prediction exists here. Inference windows the full trial, so this is True for every frame and the prediction arrays contain **no NaN** |
| `scoring_mask.npy` | (T,) bool | frames included in the accuracy metrics: coverage minus the first/last 20 frames. Those edges are predicted but deliberately unscored |
| `evaluation_mask.npy` | (T,) bool | historical name, identical to `scoring_mask.npy` |
| `metrics.json` | - | per-trial and per-channel MAE/RMSE/bias/r, plus pooling sums |

Channel order is in `direct_torque_names.json`. Predictions come from a model that
**never saw that experiment during training** - see the root README.

Edge frames are predicted but not scored: `prediction_coverage` is True everywhere,
while `scoring_mask` drops the trial's first and last 20 frames. Training trimmed
those same 20 frames off each end before windowing, so the model was never fit on
them - which is exactly why they are reported separately rather than silently mixed
into the accuracy numbers.

## `opensim_id/` - OpenSim inverse dynamics (where available)
`inverse_dynamics.sto` plus the setup and inputs used to produce it. Absent for the
PD, OpenCapVal and Hip_OA experiments.
"""


def write_root_readme(dst: Path, exps: List[str]) -> None:
    (dst / "README.md").write_text(f"""# KineticVAEDataset

Gait kinematics with matched ground-reaction kinetics, MuJoCo/OpenSim models, and
leave-one-experiment-out joint-torque predictions.

Generated {datetime.now():%Y-%m-%d} from `TrustedDataSet_ByExperiment`.

## Layout

```
<Experiment>/<Subject>/<Trial_#>/
    raw/  kinematics/  mujoco/{{kinetics,landmarks}}/  prediction/  opensim_id/
```

- `README.md` in each experiment folder describes that cohort.
- `schema/` - column definitions, coordinate frames, per-file reference.
- `dataset_index.csv` - one row per trial (frames, prediction availability, model nq).
- `accuracy/` - leave-one-experiment-out accuracy tables.

## Start here

1. `schema/file_reference.md` - what every file is.
2. `schema/coordinate_frames.md` - frames, sign and side conventions.
3. `schema/dof_columns.json` - machine-readable column -> DOF map.

## Three things that will bite you

1. **Everything except `raw/` is frame-aligned.** `kinematics/`, `mujoco/`,
   `mujoco/kinetics/`, `mujoco/landmarks/` and `prediction/` all share one index,
   `0..n_frames-1`. `kinematics/time.npy` carries the timebase.
2. **`raw/` is longer and is NOT a slice of the processed arrays.** Processing trims
   *and* low-pass filters, so values differ inside the window. `trial_metadata.json`
   -> `alignment.raw_window_into_motion_aligned_grid` gives the index window; use it
   for alignment only, never assume equality.
3. **`mujoco/pos_mjx` column count varies per subject** (the model's `nq`: 43 with
   patella, 33 without, or 23 independent DOFs for OpenCap subjects). Read
   `<Subject>/models/mjx_qpos_layout.json`. **`kinematics/pos_cleaned.npy` is the
   uniform alternative** - always (T, 21), identical columns for every subject.

## Predictions

Every prediction comes from a model trained **without** that experiment
(leave-one-experiment-out), so predictions are out-of-sample at the cohort level.

**Every frame carries a prediction, but not every frame is scored.** Two masks:

- `prediction_coverage.npy` - a prediction exists here. Inference windows the full
  trial, so this is True on every frame and the prediction arrays contain no NaN.
- `scoring_mask.npy` - frames included in the accuracy numbers: coverage minus the
  trial's first and last 20 frames. (`evaluation_mask.npy` is the historical name
  for the same array.)

Those edge frames are excluded because training trimmed the same 20 frames off each
end before windowing, so the model was never fit on them. They are provided so you
can judge edge behaviour yourself, and kept out of the metrics so the reported
accuracy is not flattered by frames the model saw least context for.
`accuracy/` holds the per-trial, per-subject and per-experiment tables. Pooled
performance across the dataset is R^2 ~ 0.92 on the 14 torque channels.

Experiments present: {', '.join(exps)}.
""")


def write_experiment_readme(exp_dir: Path, name: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    notes = {
        "Numeric": "AddBiomechanics healthy walking subjects (folder names `02`-`20`).",
        "Stroke": "Public stroke cohort (`SUBJ*`).",
        "GaitRetraining": "Gait-retraining study, including the `SubjectR*` arm.",
        "OA_Y": "Silder 2008 older (`OA*`) and younger (`Y*`) adults. **`OA` here means "
                "Older Adult, not osteoarthritis.** Trials carry a visual-QC review; the "
                "`Y*` subgroup has known left-side data-quality issues.",
        "PD": "Parkinson's cohort, `_on`/`_off` medication states.",
        "S_GAH": "Gastrocnemius-avoidance protocol (`S#`, `S_GAH_*`). Unilateral "
                 "intervention, so left/right asymmetry is expected.",
        "OpenCapVal": "OpenCap validation subjects, normal walking only (trunk-sway "
                      "trials excluded). MoCap-derived kinematics. No noised variants existed "
                      "for this cohort.",
        "Hip_OA": "Bertaux hip-osteoarthritis cohort (`HOA*`/`HEA*`), visually trimmed and "
                  "QC-filtered. Never used for training in any model.",
    }
    (exp_dir / "README.md").write_text(f"""# {name}

{notes.get(name, "")}

## Organisation

```
{name}/<Subject>/
├── subject_metadata.json      height, mass, sex, DOF names, segment scales
├── models/
│   ├── opensim_model.osim
│   ├── opensim_model_no_patella.osim   (used for OpenSim inverse dynamics)
│   ├── mujoco_model.xml
│   ├── mjx_qpos_layout.json            qpos index -> DOF name for THIS subject
│   └── Geometry/                       meshes referenced by the models
└── <Trial_#>/
    ├── trial_metadata.json    frame count, processing/trim provenance
    ├── raw/                   untrimmed OpenSim-frame signals (+ source/ originals)
    ├── kinematics/            pos/vel/acc_cleaned.npy (T,21) + time.npy
    ├── mujoco/                MJX state, Jacobian, and everything model-derived
    │   ├── pos_mjx / qvel_mjx / qacc_mjx    full qpos vector (width varies)
    │   ├── Jacobian.npy, KneeToCOP_Vectors.npy
    │   ├── kinetics/          GRF / COP / moments / joint-torque ground truth
    │   └── landmarks/         joint centres, COM, rotations, foot angles
    ├── prediction/            model prediction vs ground truth (14 channels)
    └── opensim_id/            OpenSim inverse dynamics (where available)
```

## Conventions

Full detail is in `../schema/`. The essentials:

- **Units**: radians, metres, newtons, newton-metres, seconds. Gravity 9.8067 m/s^2.
- **Sides**: all 6-wide arrays are `[R_x, R_y, R_z, L_x, L_y, L_z]`, right foot first.
- **Frames**: `raw/` and `opensim_id/` are OpenSim world (**Y-up**); everything MJX-derived
  is MuJoCo world (**Z-up**), related by `[x, y, z] -> [x, -z, y]`. COP targets are in a
  ground-aligned calcaneus frame; see `../schema/coordinate_frames.md`.
- **Normalisation**: COP / height, GRF and moments / mass, predicted torque as
  percent of body-weight x height (`tau / (m * h * 9.8067) * 100`).
- **Column maps**: `../schema/dof_columns.json`. `pos_inputs.npy` is the traceable
  18-column knee-inclusive/no-MTP schema; `pos_mjx.npy` width is per-subject
  (see `mjx_qpos_layout.json`).

## Predictions

Produced by a model trained on every experiment **except this one**. See
`../accuracy/` for per-trial and per-subject error tables.

Every frame has a prediction (`prediction_coverage.npy` is True throughout, no NaN).
Accuracy is computed on `scoring_mask.npy`, which drops the trial's first and last
20 frames - the same frames training trimmed away. `metrics.json` reports both
`n_predicted_frames` and `n_eval_frames`.
""")




EXAMPLE_PY = '''"""Load one trial and rebuild the joint torques from the shipped files.

Run from the dataset root:
    python example_torque_reconstruction.py Numeric/02/Trial_11
"""
import sys, json
import numpy as np
from pathlib import Path

trial = Path(sys.argv[1] if len(sys.argv) > 1 else "Numeric/02/Trial_11")

J = np.load(trial / "mujoco" / "Jacobian.npy", allow_pickle=True).item()
jacp, jacr = np.asarray(J["jacp"]), np.asarray(J["jacr"])   # (T, 2 feet, 3, ndof)

F = np.load(trial / "mujoco" / "kinetics" / "GRF_NoFilt_Trimmed.npy")  # (T, 6) newtons, world
M0 = np.load(trial / "mujoco" / "kinetics" / "Moment_Cleaned.npy")     # (T, 6) free moment
r = np.load(trial / "mujoco" / "kinetics" / "COP_CalcFrame_GroundAligned_BackToWorld.npy")

T = F.shape[0]
F = F.reshape(T, 2, 3); M0 = M0.reshape(T, 2, 3); r = r.reshape(T, 2, 3)

# Moment about each foot body origin, then project through the Jacobians.
M = M0 + np.cross(r, F)
tau_grf = np.einsum("tbij,tbi->tj", jacp, F) + np.einsum("tbij,tbi->tj", jacr, M)

qfrc_inverse = np.load(trial / "mujoco" / "kinetics" / "qfrc_inverse.npy")
id_gt = qfrc_inverse - tau_grf                      # == ID_GT_MJX.npy

ref_tau = np.load(trial / "mujoco" / "kinetics" / "qfrc_grf_contribution.npy")
ref_id = np.load(trial / "mujoco" / "kinetics" / "ID_GT_MJX.npy")
rel = lambda a, b: 100 * np.mean(np.abs(a - b)) / np.mean(np.abs(b))
print(f"tau_grf   vs qfrc_grf_contribution : {rel(tau_grf, ref_tau):.3f}% mean abs error")
print(f"ID_GT_MJX vs shipped ID_GT_MJX     : {rel(id_gt, ref_id):.3f}% mean abs error")

# Model prediction vs ground truth, if this trial has been scored.
pred_dir = trial / "prediction"
if pred_dir.exists():
    names = json.loads((pred_dir / "direct_torque_names.json").read_text())
    pred = np.load(pred_dir / "direct_torque_pred_percent_bwh.npy")
    gt = np.load(pred_dir / "direct_torque_gt_percent_bwh.npy")
    mask = np.load(pred_dir / "scoring_mask.npy")   # score the interior only
    mae = np.nanmean(np.abs(pred[mask] - gt[mask]), axis=0)
    print("\\nper-channel MAE (%BW*H) over evaluated frames:")
    for n, v in zip(names, mae):
        print(f"   {n:<18} {v:.4f}")
'''


def write_example(dst: Path) -> None:
    (dst / "example_torque_reconstruction.py").write_text(EXAMPLE_PY)


if __name__ == "__main__":
    main()
