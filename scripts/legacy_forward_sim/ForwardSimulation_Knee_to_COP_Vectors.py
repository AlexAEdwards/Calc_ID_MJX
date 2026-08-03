"""
Script to run a forward simulation loading noised or original inputs, calculate coupled coordinates,
run a forward kinematic sim to extract global knee positions, and calculate vectors to COP
across multiple dataset formats.
"""

import sys
import numpy as np
import mujoco
from pathlib import Path
import argparse

# Important: make sure we can import ProcessData from the current directory
sys.path.append(str(Path.cwd()))
import ProcessData as pd

def process_trial(trial_dir: Path, opencap_val: bool = False):
    """Process a single trial and save the knee-to-COP vectors."""
    subject_dir = trial_dir.parent
    proc_dir = trial_dir / "ProcessedData"
    
    # Define sources: list of dicts with paths and target output
    sources = []
    if opencap_val:
        # 1. ProcessedData source (Clean kinematics and kinetics)
        sources.append({
            "name": "Processed",
            "pos": trial_dir / "Motion" / "Pos.npy",
            "vel": trial_dir / "Motion" / "Vel.npy",
            "acc": trial_dir / "Motion" / "Accel.npy",
            "cop": proc_dir / "COP_Cleaned_Relative.npy",
            "ankle_pos_r": proc_dir / "ankle_pos_r.npy",
            "ankle_pos_l": proc_dir / "ankle_pos_l.npy",
            "output": proc_dir / "KneeToCOP_Vectors.npy"
        })
        # 2. Mocap source (Motion capture kinematics)
        mocap_dir = trial_dir / "Motion" / "Mocap"
        if mocap_dir.exists():
            sources.append({
                "name": "Mocap",
                "pos": mocap_dir / "Pos.npy",
                "vel": mocap_dir / "Vel.npy",
                "acc": mocap_dir / "Accel.npy",
                "cop": proc_dir / "COP_Cleaned_Relative_Mocap.npy",
                "ankle_pos_r": proc_dir / "ankle_pos_r_mocap.npy",
                "ankle_pos_l": proc_dir / "ankle_pos_l_mocap.npy",
                "output": proc_dir / "KneeToCOP_Vectors_Mocap.npy"
            })
    else:
        # Default logic for TrustedDataSetNoised12Distributed
        motion_dir = trial_dir / "Motion" / "Motion_Pelvis_Adjusted"
        if not motion_dir.exists():
            motion_dir = trial_dir / "Motion"
        sources.append({
            "name": "Noised",
            "pos": motion_dir / "Pos_noised.npy",
            "vel": motion_dir / "Vel_noised.npy",
            "acc": motion_dir / "Accel_noised.npy",
            "cop": proc_dir / "COP_Cleaned_Relative_noised.npy",
            "ankle_pos_r": proc_dir / "ankle_pos_r_noised.npy",
            "ankle_pos_l": proc_dir / "ankle_pos_l_noised.npy",
            "output": proc_dir / "KneeToCOP_Vectors_noised.npy"
        })

    overall_success = True
    info_list = []

    for src in sources:
        required = [src["pos"], src["vel"], src["acc"], src["cop"], src["ankle_pos_r"], src["ankle_pos_l"]]
        if not all(f.exists() for f in required):
            continue

        try:
            # 1. Resolve XML model path
            xml_path = pd.resolve_subject_model_xml(subject_dir, {"UsedFIXEDModels": False})
            
            # 2. Load inputs
            pos = np.load(src["pos"])
            vel = np.load(src["vel"])
            acc = np.load(src["acc"])
            cop_rel = np.load(src["cop"])
            ankle_r = np.load(src["ankle_pos_r"])
            ankle_l = np.load(src["ankle_pos_l"])

            # 3. Setup MuJoCo model 
            mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
            nq, nv = mj_model.nq, mj_model.nv
            
            # Match lengths
            T = min(pos.shape[0], cop_rel.shape[0], ankle_r.shape[0], ankle_l.shape[0])
            pos, vel, acc = pos[:T], vel[:T], acc[:T]
            cop_rel, ankle_r, ankle_l = cop_rel[:T], ankle_r[:T], ankle_l[:T]
            
            # Map raw columns to MuJoCo qpos/qvel/qacc
            qpos_matrix = np.zeros((T, nq))
            qvel_matrix = np.zeros((T, nv))
            qacc_matrix = np.zeros((T, nv))
            for t in range(T):
                qpos_matrix[t] = pd.map_patient_to_qpos(pos[t], nq)
                qvel_matrix[t] = pd.map_patient_to_qpos(vel[t], nv)
                qacc_matrix[t] = pd.map_patient_to_qpos(acc[t], nv)
            
            # Polynomial coupling logic
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            couplings = []
            for eq in root.iter("equality"):
                for weld in eq.iter("joint"):
                    slave_name  = weld.get("joint1")
                    master_name = weld.get("joint2")
                    if slave_name and master_name:
                        poly = weld.get("polycoef", "0 1 0 0 0").split()
                        coeffs = [float(c) for c in poly]
                        couplings.append((slave_name, master_name, coeffs))

            qpos_coupled = qpos_matrix.copy()
            if couplings:
                qvel_coupled, qacc_coupled = qvel_matrix.copy(), qacc_matrix.copy()
                for slave_name, master_name, coeffs in couplings:
                    slave_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, slave_name)
                    master_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, master_name)
                    if slave_id >= 0 and master_id >= 0:
                        p_slave, p_master = mj_model.jnt_qposadr[slave_id], mj_model.jnt_qposadr[master_id]
                        v_slave, v_master = mj_model.jnt_dofadr[slave_id], mj_model.jnt_dofadr[master_id]
                        theta = qpos_coupled[:, p_master]
                        c = coeffs + [0.0] * (5 - len(coeffs))
                        qpos_coupled[:, p_slave] = c[0] + c[1]*theta + c[2]*theta**2 + c[3]*theta**3 + c[4]*theta**4
                        dq_dtheta = c[1] + 2*c[2]*theta + 3*c[3]*theta**2 + 4*c[4]*theta**3
                        qvel_coupled[:, v_slave] = dq_dtheta * qvel_coupled[:, v_master]
                        d2q_dtheta2 = 2*c[2] + 6*c[3]*theta + 12*c[4]*theta**2
                        qacc_coupled[:, v_slave] = (dq_dtheta * qacc_coupled[:, v_master] + d2q_dtheta2 * qvel_coupled[:, v_master]**2)

            # 4. Global Kinematics
            mj_data = mujoco.MjData(mj_model)
            knee_r_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tibia_r")
            knee_l_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tibia_l")
            
            knee_pos_r_global = np.zeros((T, 3))
            knee_pos_l_global = np.zeros((T, 3))
            for t in range(T):
                mj_data.qpos[:] = qpos_coupled[t]
                mujoco.mj_kinematics(mj_model, mj_data)
                knee_pos_r_global[t] = mj_data.xpos[knee_r_id]
                knee_pos_l_global[t] = mj_data.xpos[knee_l_id]
                
            # 5. World COP
            cop_world_r = np.zeros((T, 3))
            cop_world_l = np.zeros((T, 3))
            cop_world_r[:, :2] = ankle_r[:, :2] + cop_rel[:, :2]
            cop_world_r[:, 2] = ankle_r[:, 2]
            cop_world_l[:, :2] = ankle_l[:, :2] + cop_rel[:, 2:4]
            cop_world_l[:, 2] = ankle_l[:, 2]

            # 6. Calculate & Save
            vec_knee_to_cop_r = cop_world_r - knee_pos_r_global
            vec_knee_to_cop_l = cop_world_l - knee_pos_l_global
            combined = np.column_stack([vec_knee_to_cop_r, vec_knee_to_cop_l])
            np.save(src["output"], combined)
            info_list.append(f"✓ {src['name']}")

        except Exception as e:
            overall_success = False
            info_list.append(f"✗ {src['name']} ({str(e)})")

    if not info_list:
        return False, "No valid data sources found in trial."
    return overall_success, " | ".join(info_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--OpenCapVal", action="store_true", help="Process OpenCapSubjects instead of noised dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory of the dataset")
    args = parser.parse_args()

    if args.OpenCapVal:
        dataset_name = "OpenCapSubjects"
        dataset_path = Path(args.data_dir) if args.data_dir else Path("Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects")
    else:
        dataset_name = "TrustedDataSetNoised12Distributed"
        dataset_path = Path(args.data_dir) if args.data_dir else Path(dataset_name)
    
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} not found.")
        sys.exit(1)

    print(f"Scanning dataset: {dataset_name} at {dataset_path}")
    
    trials_processed = 0
    errors = []
    subjects = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    for subject_dir in subjects:
        print(f"\nProcessing Subject: {subject_dir.name}")
        trials = sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name != "Geometry"])
        for trial_dir in trials:
            if not (trial_dir / "Motion").exists(): continue
            success, info = process_trial(trial_dir, opencap_val=args.OpenCapVal)
            if success:
                print(f"  {trial_dir.name}: {info}")
                trials_processed += 1
            else:
                print(f"  {trial_dir.name}: {info}")
                errors.append(f"{subject_dir.name}/{trial_dir.name}: {info}")

    print(f"\nFinished. Processed {trials_processed} trials.")

if __name__ == "__main__":
    main()
