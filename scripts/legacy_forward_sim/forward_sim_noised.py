import sys
from pathlib import Path
import numpy as np
import mujoco

sys.path.append(str(Path.cwd()))
import ProcessData as pd

def main():
    dataset_dir = Path("TrustedDataSetNoised12Distributed")
    subject_dir = dataset_dir / "GaitRetraining_SubjectR731"
    trial_dir = subject_dir / "Trial_25"
    
    print(f"Running forward simulation for: {trial_dir}")
    
    xml_path = pd.resolve_subject_model_xml(subject_dir, {"UsedFIXEDModels": False})
    
    # 1. Load noised inputs
    pos_noised = np.load(trial_dir / "Motion" / "Motion_Pelvis_Adjusted" / "Pos_noised.npy")
    vel_noised = np.load(trial_dir / "Motion" / "Motion_Pelvis_Adjusted" / "Vel_noised.npy")
    acc_noised = np.load(trial_dir / "Motion" / "Motion_Pelvis_Adjusted" / "Accel_noised.npy")
    
    T = pos_noised.shape[0]
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    nq = mj_model.nq
    
    qpos_matrix = np.array([pd.map_patient_to_qpos(pos_noised[t], nq) for t in range(T)])
    qvel_matrix = np.array([pd.map_patient_to_qpos(vel_noised[t], nq) for t in range(T)])
    qacc_matrix = np.array([pd.map_patient_to_qpos(acc_noised[t], nq) for t in range(T)])
    
    # 2. Calculate the coupled coordinates
    qpos_coup, qvel_coup, qacc_coup = pd.calculate_coupled_coordinates_automated(
        qpos_matrix, qvel_matrix, qacc_matrix, xml_path
    )
    
    # 3. Run forward simulation & extract global position of knee
    mj_data = mujoco.MjData(mj_model)
    
    knee_r_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tibia_r")
    knee_l_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "tibia_l")
    
    # Load ankle positions to reconstruct world COP (ProcessData subtracts ankle_pos from COP in relative)
    # Alternatively we can just load the raw absolute COP from the dataset, but let's use the clean bundle if possible, or calculate it.
    pass

if __name__ == "__main__":
    main()
