import pandas as pd
import numpy as np
import mujoco
from mujoco import mjx
import jax.numpy as jnp

# Load model
model_path = "Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml"
mj_model = mujoco.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.make_data(mjx_model)

# Load data
data_path = "PatientData/Falisse_2017_subject_01/"
pos_data = pd.read_csv(data_path + "pos.csv")
cop_data_raw = pd.read_csv(data_path + "cop.csv", header=None, skiprows=1)
cop_matrix = cop_data_raw.values

# Get body IDs
calcn_l_id = mj_model.body('calcn_l').id
calcn_r_id = mj_model.body('calcn_r').id

# Map qpos for a timestep with COP data
def map_patient_to_qpos(patient_row, qpos_size=39):
    qpos_mapping = {
        'pelvis_tx': 0, 'pelvis_ty': 1, 'pelvis_tz': 2,
        'pelvis_tilt': 3, 'pelvis_list': 4, 'pelvis_rotation': 5,
        'hip_flexion_r': 6, 'hip_adduction_r': 7, 'hip_rotation_r': 8,
        'knee_angle_r': 11, 'ankle_angle_r': 14, 'subtalar_angle_r': 15, 'mtp_angle_r': 16,
        'hip_flexion_l': 21, 'hip_adduction_l': 22, 'hip_rotation_l': 23,
        'knee_angle_l': 26, 'ankle_angle_l': 29, 'subtalar_angle_l': 30, 'mtp_angle_l': 31,
        'lumbar_extension': 36, 'lumbar_bending': 37, 'lumbar_rotation': 38,
    }
    qpos = np.zeros(qpos_size)
    for col_name, qpos_idx in qpos_mapping.items():
        if col_name in patient_row:
            qpos[qpos_idx] = patient_row[col_name]
    return qpos

# Find a timestep with non-zero COP data
for t in range(len(cop_matrix)):
    if cop_matrix[t, 1] != 0 or cop_matrix[t, 4] != 0:
        print(f"\n{'='*70}")
        print(f"Timestep {t} (time={cop_matrix[t, 0]:.3f}s)")
        print(f"{'='*70}")
        
        # Get COP data from CSV
        print(f"\nCOP data from CSV:")
        print(f"  Right foot: col1={cop_matrix[t, 1]:.4f}, col3={cop_matrix[t, 3]:.4f}")
        print(f"  Left foot:  col4={cop_matrix[t, 4]:.4f}, col6={cop_matrix[t, 6]:.4f}")
        
        # Get calcaneus positions in MuJoCo
        qpos = map_patient_to_qpos(pos_data.iloc[t], mj_model.nq)
        mjx_data_t = mjx_data.replace(qpos=jnp.array(qpos))
        mjx_data_t = mjx.forward(mjx_model, mjx_data_t)
        
        calcn_l_pos = np.array(mjx_data_t.xpos[calcn_l_id])
        calcn_r_pos = np.array(mjx_data_t.xpos[calcn_r_id])
        
        print(f"\nCalcaneus positions in MuJoCo world frame:")
        print(f"  Left:  X={calcn_l_pos[0]:.4f}, Y={calcn_l_pos[1]:.4f}, Z={calcn_l_pos[2]:.4f}")
        print(f"  Right: X={calcn_r_pos[0]:.4f}, Y={calcn_r_pos[1]:.4f}, Z={calcn_r_pos[2]:.4f}")
        
        # Try different coordinate mappings
        print(f"\nTrying different COP coordinate mappings:")
        
        print(f"\n1. Current mapping [X, 0, -Z]:")
        cop_r_v1 = np.array([cop_matrix[t, 1], 0.0, -cop_matrix[t, 3]])
        cop_l_v1 = np.array([cop_matrix[t, 4], 0.0, -cop_matrix[t, 6]])
        print(f"  Right COP: {cop_r_v1}, dist to calcn_r: {np.linalg.norm(calcn_r_pos - cop_r_v1):.4f}")
        print(f"  Left COP:  {cop_l_v1}, dist to calcn_l: {np.linalg.norm(calcn_l_pos - cop_l_v1):.4f}")
        
        print(f"\n2. Try [X, 0, Z]:")
        cop_r_v2 = np.array([cop_matrix[t, 1], 0.0, cop_matrix[t, 3]])
        cop_l_v2 = np.array([cop_matrix[t, 4], 0.0, cop_matrix[t, 6]])
        print(f"  Right COP: {cop_r_v2}, dist to calcn_r: {np.linalg.norm(calcn_r_pos - cop_r_v2):.4f}")
        print(f"  Left COP:  {cop_l_v2}, dist to calcn_l: {np.linalg.norm(calcn_l_pos - cop_l_v2):.4f}")
        
        print(f"\n3. Try [Z, 0, X]:")
        cop_r_v3 = np.array([cop_matrix[t, 3], 0.0, cop_matrix[t, 1]])
        cop_l_v3 = np.array([cop_matrix[t, 6], 0.0, cop_matrix[t, 4]])
        print(f"  Right COP: {cop_r_v3}, dist to calcn_r: {np.linalg.norm(calcn_r_pos - cop_r_v3):.4f}")
        print(f"  Left COP:  {cop_l_v3}, dist to calcn_l: {np.linalg.norm(calcn_l_pos - cop_l_v3):.4f}")
        
        print(f"\n4. Try [-Z, 0, X]:")
        cop_r_v4 = np.array([-cop_matrix[t, 3], 0.0, cop_matrix[t, 1]])
        cop_l_v4 = np.array([-cop_matrix[t, 6], 0.0, cop_matrix[t, 4]])
        print(f"  Right COP: {cop_r_v4}, dist to calcn_r: {np.linalg.norm(calcn_r_pos - cop_r_v4):.4f}")
        print(f"  Left COP:  {cop_l_v4}, dist to calcn_l: {np.linalg.norm(calcn_l_pos - cop_l_v4):.4f}")
        
        print(f"\n5. Try [Z, 0, -X]:")
        cop_r_v5 = np.array([cop_matrix[t, 3], 0.0, -cop_matrix[t, 1]])
        cop_l_v5 = np.array([cop_matrix[t, 6], 0.0, -cop_matrix[t, 4]])
        print(f"  Right COP: {cop_r_v5}, dist to calcn_r: {np.linalg.norm(calcn_r_pos - cop_r_v5):.4f}")
        print(f"  Left COP:  {cop_l_v5}, dist to calcn_l: {np.linalg.norm(calcn_l_pos - cop_l_v5):.4f}")
        
        print(f"\n6. Try [-Z, 0, -X]:")
        cop_r_v6 = np.array([-cop_matrix[t, 3], 0.0, -cop_matrix[t, 1]])
        cop_l_v6 = np.array([-cop_matrix[t, 6], 0.0, -cop_matrix[t, 4]])
        print(f"  Right COP: {cop_r_v6}, dist to calcn_r: {np.linalg.norm(calcn_r_pos - cop_r_v6):.4f}")
        print(f"  Left COP:  {cop_l_v6}, dist to calcn_l: {np.linalg.norm(calcn_l_pos - cop_l_v6):.4f}")
        
        break  # Only check first timestep with data
