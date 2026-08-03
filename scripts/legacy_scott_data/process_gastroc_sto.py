import numpy as np
import glob
import json
import argparse
from pathlib import Path
from scipy.interpolate import interp1d

# ──────────────────────────────────────────────────────────────────────
# Target output rate — everything is resampled to this frequency.
# BatchDataProcessingFast.py also resamples to dt=0.01 internally,
# so giving it 100 Hz data avoids any further resampling artefacts.
TARGET_DT = 0.01  # 100 Hz
# ──────────────────────────────────────────────────────────────────────

# 23-DOF order that matches every other dataset in the pipeline
# (Jan7, OpenCap, OA_GaitRetraining, etc.)
TARGET_DOFS = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]

# .mot column-name → output-slot mapping.
# Uses column NAMES (not indices!) to avoid the index-order bug
# that caused Left↔Right↔Moment cross-contamination.
# Convention: output is always [R_x, R_y, R_z, L_x, L_y, L_z].
# In these .mot files  "1_" prefix = Left foot, no prefix = Right foot.
FORCE_MAPPING = {
    "right": ["ground_force_vx", "ground_force_vy", "ground_force_vz"],
    "left":  ["1_ground_force_vx", "1_ground_force_vy", "1_ground_force_vz"],
}
COP_MAPPING = {
    "right": ["ground_force_px", "ground_force_py", "ground_force_pz"],
    "left":  ["1_ground_force_px", "1_ground_force_py", "1_ground_force_pz"],
}
TORQUE_MAPPING = {
    "right": ["ground_torque_x", "ground_torque_y", "ground_torque_z"],
    "left":  ["1_ground_torque_x", "1_ground_torque_y", "1_ground_torque_z"],
}


# ── file parsers ─────────────────────────────────────────────────────

def _parse_opensim_file(file_path):
    """Parse a .sto or .mot file.  Returns (column_names, data_array)."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    header_end = 0
    for i, line in enumerate(lines):
        if "endheader" in line:
            header_end = i + 1
            break

    col_names = lines[header_end].strip().split()
    data = np.loadtxt(lines[header_end + 1 :])
    return col_names, data


def _build_uniform_time(t_orig, dt=TARGET_DT):
    """Create a uniform time vector spanning the same range as *t_orig*."""
    t_start = np.ceil(t_orig[0] / dt) * dt
    t_end = np.floor(t_orig[-1] / dt) * dt
    return np.arange(t_start, t_end + dt / 2, dt)


def _resample(t_orig, data, t_new):
    """Linearly interpolate *data* (N×C) from *t_orig* to *t_new*."""
    f = interp1d(t_orig, data, axis=0, kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    return f(t_new)


# ── kinematics processing ───────────────────────────────────────────

def _process_sto(sto_path, save_dir, t_uniform):
    """
    Read results_ik.sto, extract the 23 target DOFs (skipping
    knee_angle_*_beta), convert degrees→radians, resample to
    *t_uniform*, differentiate, and save Pos/Vel/Accel.
    """
    cols, data = _parse_opensim_file(sto_path)
    cols = [c.strip() for c in cols]
    time_sto = data[:, 0]
    col_to_idx = {name: i for i, name in enumerate(cols)}

    num_frames_orig = data.shape[0]
    pos_orig = np.zeros((num_frames_orig, len(TARGET_DOFS)))

    missing = []
    for i, dof in enumerate(TARGET_DOFS):
        if dof in col_to_idx:
            val = data[:, col_to_idx[dof]]
            # Translational DOFs are already in metres; angles are in degrees
            if dof in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
                pos_orig[:, i] = val
            else:
                pos_orig[:, i] = np.deg2rad(val)
        else:
            missing.append(dof)

    if missing:
        print(f"    ⚠️  DOFs not in STO (filled with 0): {missing}")

    # Resample kinematics to 100 Hz
    pos_100 = _resample(time_sto, pos_orig, t_uniform)

    # Differentiate at the uniform rate
    dt = t_uniform[1] - t_uniform[0]
    vel_100 = np.gradient(pos_100, dt, axis=0)
    acc_100 = np.gradient(vel_100, dt, axis=0)

    np.save(save_dir / "Pos.npy", pos_100.astype(np.float32))
    np.save(save_dir / "Vel.npy", vel_100.astype(np.float32))
    np.save(save_dir / "Accel.npy", acc_100.astype(np.float32))

    print(f"    ✅ Pos/Vel/Accel  {num_frames_orig} frames @ "
          f"{1/np.mean(np.diff(time_sto)):.0f} Hz → {len(t_uniform)} frames @ "
          f"{1/dt:.0f} Hz  ({len(TARGET_DOFS)} DOFs)")

    return time_sto  # return original kin time for reference


# ── force processing ────────────────────────────────────────────────

def _extract_by_name(col_to_idx, data, mapping):
    """
    Pull columns by NAME for Right then Left foot.
    Returns an (N, 6) array: [R_x, R_y, R_z, L_x, L_y, L_z].
    """
    cols = mapping["right"] + mapping["left"]
    idxs = []
    for c in cols:
        if c not in col_to_idx:
            raise KeyError(f"Column '{c}' not found in .mot file.  "
                           f"Available: {list(col_to_idx.keys())}")
        idxs.append(col_to_idx[c])
    return data[:, idxs]


def _process_mot(mot_path, save_dir, t_uniform):
    """
    Read the *_forces.mot file, extract GRF/COP/GRM **by column name**
    (not by index!), resample to *t_uniform*, and save.
    """
    cols, data = _parse_opensim_file(mot_path)
    cols = [c.strip() for c in cols]
    time_mot = data[:, 0]
    col_to_idx = {name: i for i, name in enumerate(cols)}

    grf_raw = _extract_by_name(col_to_idx, data, FORCE_MAPPING)
    cop_raw = _extract_by_name(col_to_idx, data, COP_MAPPING)
    grm_raw = _extract_by_name(col_to_idx, data, TORQUE_MAPPING)

    # Resample to the same uniform 100 Hz grid as kinematics
    grf_100 = _resample(time_mot, grf_raw, t_uniform)
    cop_100 = _resample(time_mot, cop_raw, t_uniform)
    grm_100 = _resample(time_mot, grm_raw, t_uniform)

    np.save(save_dir / "GRF.npy", grf_100.astype(np.float32))
    np.save(save_dir / "COP.npy", cop_100.astype(np.float32))
    np.save(save_dir / "GRM.npy", grm_100.astype(np.float32))

    print(f"    ✅ GRF/COP/GRM   {data.shape[0]} frames @ "
          f"{1/np.mean(np.diff(time_mot)):.0f} Hz → {len(t_uniform)} frames @ "
          f"{1/(t_uniform[1]-t_uniform[0]):.0f} Hz")


# ── main driver ─────────────────────────────────────────────────────

def process_gastroc(single_file=None):
    root = Path("Datasets_NAS/AddBiomechanicsDataset_All_npy/"
                "Gastroc_Avoidance_Healthy_MJX")

    if single_file:
        sto_files = [single_file]
        print(f"Processing single file: {single_file}")
    else:
        # Fixed: glob uses "Trial_*" not "Trail_*"
        sto_files = sorted(glob.glob(
            str(root / "S_GAH_*/Trial_*/Motion/results_ik.sto")))

    if not sto_files:
        print("❌ No STO files found.")
        return

    print(f"🚀 Processing {len(sto_files)} trials …\n")
    total = 0

    for sto_path in sto_files:
        sto_path = Path(sto_path)
        save_dir = sto_path.parent          # …/Motion/
        subj_trial = f"{sto_path.parts[-4]}/{sto_path.parts[-3]}"
        print(f"  📂 {subj_trial}")

        # ── build a single uniform 100 Hz time grid ──────────────
        # Use the UNION of .sto and .mot time ranges so both can be
        # interpolated onto the same axis without extrapolation.
        cols_sto, data_sto = _parse_opensim_file(sto_path)
        time_sto = data_sto[:, 0]

        mot_files = sorted(save_dir.glob("*_forces.mot"))
        if mot_files:
            cols_mot, data_mot = _parse_opensim_file(mot_files[0])
            time_mot = data_mot[:, 0]
            # Restrict to overlapping region
            t_start = max(time_sto[0], time_mot[0])
            t_end = min(time_sto[-1], time_mot[-1])
        else:
            t_start = time_sto[0]
            t_end = time_sto[-1]

        t_start = np.ceil(t_start / TARGET_DT) * TARGET_DT
        t_end = np.floor(t_end / TARGET_DT) * TARGET_DT
        t_uniform = np.arange(t_start, t_end + TARGET_DT / 2, TARGET_DT)

        # ── kinematics ───────────────────────────────────────────
        _process_sto(sto_path, save_dir, t_uniform)

        # ── forces ───────────────────────────────────────────────
        if mot_files:
            _process_mot(mot_files[0], save_dir, t_uniform)
        else:
            print(f"    ⚠️  No *_forces.mot in {save_dir}")

        # ── save time vector ─────────────────────────────────────
        # Single shared time vector — kinematics AND forces are now
        # on the same 100 Hz grid, so no Time_for_pos.npy needed.
        np.save(save_dir / "Time.npy", t_uniform.astype(np.float32))

        # Remove stale Time_for_pos.npy if present (everything is
        # at the same rate now, so a separate kin-time file would
        # confuse BatchDataProcessingFast.py).
        time_for_pos = save_dir / "Time_for_pos.npy"
        if time_for_pos.exists():
            time_for_pos.unlink()

        print(f"    ✅ Time.npy      {len(t_uniform)} frames @ 100 Hz  "
              f"[{t_uniform[0]:.3f} – {t_uniform[-1]:.3f} s]")

        # ── update Patient_MD.json ───────────────────────────────
        subj_dir = sto_path.parents[2]
        md_path = subj_dir / "Patient_MD.json"
        if md_path.exists():
            with open(md_path, "r") as f:
                subj_md = json.load(f)
            subj_md["DOF_names"] = TARGET_DOFS
            subj_md["NumDOFs"] = len(TARGET_DOFS)
            with open(md_path, "w") as f:
                json.dump(subj_md, f, indent=2)

        total += 1
        print()

    print(f"🏁 Done — processed {total} / {len(sto_files)} trials.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process STO + MOT files for Gastroc_Avoidance_Healthy_MJX."
    )
    parser.add_argument("--file", type=str,
                        help="Path to a single .sto file to process.")
    args = parser.parse_args()
    process_gastroc(single_file=args.file)
