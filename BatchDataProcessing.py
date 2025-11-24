from pyexpat import model
import mujoco  # Only needed for visualization code at bottom
from mujoco import mjx
import numpy as np
import pandas as pd
# import mujoco.viewer  # Only needed for visualization code at bottom
import time
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib
# matplotlib.use('Agg')  # Commented out to allow interactive plotting
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import mujoco.mjx._src as support
from updateModel import update_model

jax.clear_caches()
# Load 

# Functions:
qpos_mapping = {
    # Pelvis translation (patient data: tx, ty, tz -> MuJoCo: 0, 1, 2)
    'pelvis_tx': 0,
    'pelvis_ty': 1, 
    'pelvis_tz': 2,
    # Pelvis rotation (patient data: tilt, list, rotation -> MuJoCo: 3, 4, 5)
    'pelvis_tilt': 3,
    'pelvis_list': 4,
    'pelvis_rotation': 5,
    # Right leg
    'hip_flexion_r': 6,
    'hip_adduction_r': 7,
    'hip_rotation_r': 8,
    # Knee (only main angle, skip coupled joints 9, 10, 12, 13)
    'knee_angle_r': 11,
    # Ankle and foot
    'ankle_angle_r': 14,
    'subtalar_angle_r': 15,
    'mtp_angle_r': 16,
    # Patella (17-20) are coupled, skip
    # Left leg
    'hip_flexion_l': 21,
    'hip_adduction_l': 22,
    'hip_rotation_l': 23,
    # Knee (only main angle, skip coupled joints 24, 25, 27, 28)
    'knee_angle_l': 26,
    # Ankle and foot
    'ankle_angle_l': 29,
    'subtalar_angle_l': 30,
    'mtp_angle_l': 31,
    # Patella (32-35) are coupled, skip
    # Lumbar
    'lumbar_extension': 36,
    'lumbar_bending': 37,
    'lumbar_rotation': 38,
}

def calculate_coupled_coordinates_automated(qpos_matrix, qvel_matrix, qacc_matrix, xml_path):
    """
    Automatically extract polynomial constraints from XML model and populate coupled coordinates.
    
    This function reads the <equality> section from the MuJoCo XML file, extracts polynomial
    constraints for coupled joints, and calculates the position, velocity, and acceleration
    for all coupled coordinates.
    
    Parameters:
    -----------
    qpos_matrix : numpy array (num_timesteps, nq)
        Position matrix with main joint angles already filled
    qvel_matrix : numpy array (num_timesteps, nv)
        Velocity matrix with main joint velocities already filled
    qacc_matrix : numpy array (num_timesteps, nv)
        Acceleration matrix with main joint accelerations already filled
    xml_path : str
        Path to the MuJoCo XML model file
    
    Returns:
    --------
    tuple : (qpos_matrix, qvel_matrix, qacc_matrix)
        Updated matrices with all coupled coordinates populated
        
    Example:
    --------
    >>> qpos, qvel, qacc = calculate_coupled_coordinates_automated(
    ...     qpos_matrix, qvel_matrix, qacc_matrix,
    ...     "Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml"
    ... )
    
    Notes:
    ------
    The function parses XML joint equality constraints of the form:
        <joint joint1="coupled_joint" joint2="driver_joint" polycoef="c0 c1 c2 c3 c4"/>
    
    where the relationship is: joint1 = c0 + c1*joint2 + c2*joint2^2 + c3*joint2^3 + c4*joint2^4
    
    Velocities and accelerations are computed using the chain rule:
        dq/dt = dq/dtheta * dtheta/dt
        d2q/dt2 = d2q/dtheta2 * (dtheta/dt)^2 + dq/dtheta * d2theta/dt2
    """
    import xml.etree.ElementTree as ET
    
    # Make copies to avoid modifying originals
    qpos_matrix = qpos_matrix.copy()
    qvel_matrix = qvel_matrix.copy()
    qacc_matrix = qacc_matrix.copy()
    
    # Load the MuJoCo model to get joint name to index mapping
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # Create mapping from joint name to qpos/qvel index
    joint_name_to_qpos_idx = {}
    joint_name_to_qvel_idx = {}
    
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_addr = model.jnt_qposadr[i]
        qvel_addr = model.jnt_dofadr[i]
        joint_name_to_qpos_idx[joint_name] = qpos_addr
        joint_name_to_qvel_idx[joint_name] = qvel_addr
    
    # Parse the XML file to extract polynomial constraints
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find all equality constraints with polynomial coefficients
    equality_section = root.find('.//equality')
    if equality_section is None:
        print("Warning: No equality section found in XML")
        return qpos_matrix, qvel_matrix, qacc_matrix
    
    # Dictionary to store constraints grouped by driver joint
    # Format: {driver_joint_name: {coupled_joint_name: [c0, c1, c2, c3, c4]}}
    constraints = {}
    
    for joint_elem in equality_section.findall('joint'):
        joint1_name = joint_elem.get('joint1')  # Coupled joint
        joint2_name = joint_elem.get('joint2')  # Driver joint (can be None for locked joints)
        polycoef_str = joint_elem.get('polycoef')
        
        if polycoef_str is None:
            continue
            
        # Parse polynomial coefficients
        coeffs = [float(x) for x in polycoef_str.split()]
        
        # Skip if not a 5-coefficient polynomial (c0, c1, c2, c3, c4)
        if len(coeffs) != 5:
            continue
        
        # Handle locked joints (no joint2 specified, or all coeffs except c0 are zero)
        if joint2_name is None or all(abs(c) < 1e-10 for c in coeffs[1:]):
            # This is a locked joint - just set to constant value c0
            if joint1_name in joint_name_to_qpos_idx:
                idx = joint_name_to_qpos_idx[joint1_name]
                qpos_matrix[:, idx] = coeffs[0]
                # Locked joints have zero velocity and acceleration
                if joint1_name in joint_name_to_qvel_idx:
                    vel_idx = joint_name_to_qvel_idx[joint1_name]
                    qvel_matrix[:, vel_idx] = 0.0
                    qacc_matrix[:, vel_idx] = 0.0
            continue
        
        # Store the constraint
        if joint2_name not in constraints:
            constraints[joint2_name] = {}
        constraints[joint2_name][joint1_name] = coeffs
    
    print(f"\n{'='*70}")
    print(f"Automated Coupled Coordinate Calculation")
    print(f"{'='*70}")
    print(f"Found {sum(len(v) for v in constraints.values())} polynomial constraints")
    print(f"Driver joints: {list(constraints.keys())}")
    
    # Process each driver joint and its coupled coordinates
    for driver_joint, coupled_joints in constraints.items():
        if driver_joint not in joint_name_to_qpos_idx:
            print(f"Warning: Driver joint '{driver_joint}' not found in model")
            continue
        
        driver_qpos_idx = joint_name_to_qpos_idx[driver_joint]
        driver_qvel_idx = joint_name_to_qvel_idx[driver_joint]
        
        # Get driver joint state
        theta = qpos_matrix[:, driver_qpos_idx]
        theta_dot = qvel_matrix[:, driver_qvel_idx]
        theta_ddot = qacc_matrix[:, driver_qvel_idx]
        
        print(f"\nProcessing driver joint: {driver_joint} (qpos_idx={driver_qpos_idx}, qvel_idx={driver_qvel_idx})")
        print(f"  Coupled joints: {len(coupled_joints)}")
        
        # Calculate each coupled coordinate
        for coupled_joint, coeffs in coupled_joints.items():
            if coupled_joint not in joint_name_to_qpos_idx:
                print(f"  Warning: Coupled joint '{coupled_joint}' not found in model")
                continue
            
            coupled_qpos_idx = joint_name_to_qpos_idx[coupled_joint]
            coupled_qvel_idx = joint_name_to_qvel_idx[coupled_joint]
            
            # Position: q = c0 + c1*θ + c2*θ² + c3*θ³ + c4*θ⁴
            qpos_matrix[:, coupled_qpos_idx] = (
                coeffs[0]
                + coeffs[1] * theta
                + coeffs[2] * theta**2
                + coeffs[3] * theta**3
                + coeffs[4] * theta**4
            )
            
            # Velocity: dq/dt = (c1 + 2*c2*θ + 3*c3*θ² + 4*c4*θ³) * dθ/dt
            dq_dtheta = (
                coeffs[1]
                + 2 * coeffs[2] * theta
                + 3 * coeffs[3] * theta**2
                + 4 * coeffs[4] * theta**3
            )
            qvel_matrix[:, coupled_qvel_idx] = dq_dtheta * theta_dot
            
            # Acceleration: d²q/dt² = d²q/dθ² * (dθ/dt)² + dq/dθ * d²θ/dt²
            d2q_dtheta2 = (
                2 * coeffs[2]
                + 6 * coeffs[3] * theta
                + 12 * coeffs[4] * theta**2
            )
            qacc_matrix[:, coupled_qvel_idx] = (
                d2q_dtheta2 * (theta_dot**2) + dq_dtheta * theta_ddot
            )
            
            print(f"    ✓ {coupled_joint} (qpos_idx={coupled_qpos_idx}, qvel_idx={coupled_qvel_idx})")
    
    print(f"\n{'='*70}")
    print(f"✓ Coupled coordinates calculated successfully")
    print(f"{'='*70}\n")
    
    return qpos_matrix, qvel_matrix, qacc_matrix

def map_patient_to_qpos(patient_row, qpos_size=39):
    """Map a single row of patient data to MuJoCo qpos array"""
    qpos = np.zeros(qpos_size)
    for col_name, qpos_idx in qpos_mapping.items():
        if col_name in patient_row:
            qpos[qpos_idx] = patient_row[col_name]
    return qpos

def resample_dataframes_to_uniform_timestep(pos_data, vel_data, acc_data, moment_matrix=None, grf_matrix=None, cop_data=None, dt=0.01):
    """
    Resample pos_data, vel_data, acc_data, and optionally moment_matrix, grf_matrix, cop_data so that their time vectors are consistent and uniformly spaced at dt (default 0.01s).
    Assumes the first column is the time vector for each DataFrame or matrix.
    Returns new DataFrames for pos/vel/acc/cop and numpy arrays for moment_matrix and grf_matrix, all resampled to the new time base.
    """
    import numpy as np
    import pandas as pd
    # Extract time vectors
    t_pos = pos_data.iloc[:, 0].values
    t_vel = vel_data.iloc[:, 0].values
    t_acc = acc_data.iloc[:, 0].values
    t_list = [t_pos, t_vel, t_acc]
    dfs = [pos_data, vel_data, acc_data]
    names = ['pos_data', 'vel_data', 'acc_data']
    # Convert moment_matrix and grf_matrix to DataFrames if provided
    moment_df = None
    grf_df = None
    if moment_matrix is not None:
        t_mom = moment_matrix[:, 0]
        t_list.append(t_mom)
        moment_df = pd.DataFrame(moment_matrix, columns=[f'col_{i}' for i in range(moment_matrix.shape[1])])
        dfs.append(moment_df)
        names.append('moment_matrix')
    if grf_matrix is not None:
        t_grf = grf_matrix[:, 0]
        t_list.append(t_grf)
        grf_df = pd.DataFrame(grf_matrix, columns=[f'col_{i}' for i in range(grf_matrix.shape[1])])
        dfs.append(grf_df)
        names.append('grf_matrix')
    if cop_data is not None:
        t_cop = cop_data.iloc[:, 0].values
        t_list.append(t_cop)
        dfs.append(cop_data)
        names.append('cop_data')
    # Use the union of all time points, then create a uniform grid
    t_min = max(t[0] for t in t_list)
    t_max = min(t[-1] for t in t_list)
    t_uniform = np.arange(t_min, t_max + dt/2, dt)
    print("\n--- Resampling DataFrames ---")
    for name, df in zip(names, dfs):
        print(f"{name}: before = {len(df)} timesteps, after = {len(t_uniform)} timesteps")
    # Interpolate each DataFrame to the new time base
    def interp_df(df, t_uniform):
        t = df.iloc[:, 0].values
        data_interp = np.empty((len(t_uniform), df.shape[1]))
        data_interp[:, 0] = t_uniform
        for col in range(1, df.shape[1]):
            data_interp[:, col] = np.interp(t_uniform, t, df.iloc[:, col].values)
        return pd.DataFrame(data_interp, columns=df.columns)
    resampled = [interp_df(df, t_uniform) for df in dfs]
    # Convert moment and grf back to numpy arrays if they were provided
    out = []
    idx = 0
    for arg, name in zip([pos_data, vel_data, acc_data, moment_matrix, grf_matrix, cop_data],
                         ['pos_data', 'vel_data', 'acc_data', 'moment_matrix', 'grf_matrix', 'cop_data']):
        if arg is not None:
            if name == 'moment_matrix' or name == 'grf_matrix':
                out.append(resampled[idx].to_numpy())
            else:
                out.append(resampled[idx])
            idx += 1
        else:
            out.append(None)
    return tuple(out)

def calculate_knee_coupled_coords_all(qpos_matrix, qvel_matrix, qacc_matrix):
    """
    Fill coupled knee coordinates for position, velocity, and acceleration using polynomials and their derivatives.
    Returns updated (qpos_matrix, qvel_matrix, qacc_matrix).
    """
    qpos_matrix = qpos_matrix.copy()
    qvel_matrix = qvel_matrix.copy()
    qacc_matrix = qacc_matrix.copy()

    # Right knee polynomials (driven by knee_angle_r at index 11)
    right_knee_polys = {
        9: [9.877e-08, 0.00324, -0.00239, 0.0005816, 5.886e-07],
        10: [7.949e-11, 0.006076, -0.001298, -2.706e-06, 6.452e-07],
        12: [-1.473e-08, 0.0791, -0.03285, -0.02522, 0.01083],
        13: [1.089e-08, 0.3695, -0.1695, 0.02516, 3.505e-07],
        17: [0.05515, -0.0158, -0.03583, 0.01403, -0.000925],
        18: [-0.01121, -0.05052, 0.009607, 0.01364, -0.003621],
        19: [0.00284182, 0, 0, 0, 0],
        20: [0.01051, 0.02476, -1.316, 0.7163, -0.1383],
    }
    # Left knee polynomials (driven by knee_angle_l at index 26)
    left_knee_polys = {
        24: [1.003e-07, 0.00324, -0.00239, 0.0005816, 5.886e-07],
        25: [8.073e-11, 0.006076, -0.001298, -2.706e-06, 6.452e-07],
        27: [-1.473e-08, 0.0791, -0.03285, -0.02522, 0.01083],
        28: [1.089e-08, 0.3695, -0.1695, 0.02516, 3.505e-07],
        32: [0.05515, -0.0158, -0.03583, 0.01403, -0.000925],
        33: [-0.01121, -0.05052, 0.009607, 0.01364, -0.003621],
        34: [0.00284182, 0, 0, 0, 0],
        35: [0.01051, 0.02476, -1.316, 0.7163, -0.1383],
    }

    # Right knee: main angle index 11
    theta_r = qpos_matrix[:, 11]
    theta_dot_r = qvel_matrix[:, 11]
    theta_ddot_r = qacc_matrix[:, 11]
    for idx, coeffs in right_knee_polys.items():
        # Position
        qpos_matrix[:, idx] = (
            coeffs[0]
            + coeffs[1] * theta_r
            + coeffs[2] * theta_r**2
            + coeffs[3] * theta_r**3
            + coeffs[4] * theta_r**4
        )
        # Velocity (chain rule)
        dq_dtheta = (
            coeffs[1]
            + 2 * coeffs[2] * theta_r
            + 3 * coeffs[3] * theta_r**2
            + 4 * coeffs[4] * theta_r**3
        )
        qvel_matrix[:, idx] = dq_dtheta * theta_dot_r
        # Acceleration (chain rule)
        d2q_dtheta2 = (
            2 * coeffs[2]
            + 6 * coeffs[3] * theta_r
            + 12 * coeffs[4] * theta_r**2
        )
        qacc_matrix[:, idx] = d2q_dtheta2 * (theta_dot_r**2) + dq_dtheta * theta_ddot_r

    # Left knee: main angle index 26
    theta_l = qpos_matrix[:, 26]
    theta_dot_l = qvel_matrix[:, 26]
    theta_ddot_l = qacc_matrix[:, 26]
    for idx, coeffs in left_knee_polys.items():
        qpos_matrix[:, idx] = (
            coeffs[0]
            + coeffs[1] * theta_l
            + coeffs[2] * theta_l**2
            + coeffs[3] * theta_l**3
            + coeffs[4] * theta_l**4
        )
        dq_dtheta = (
            coeffs[1]
            + 2 * coeffs[2] * theta_l
            + 3 * coeffs[3] * theta_l**2
            + 4 * coeffs[4] * theta_l**3
        )
        qvel_matrix[:, idx] = dq_dtheta * theta_dot_l
        d2q_dtheta2 = (
            2 * coeffs[2]
            + 6 * coeffs[3] * theta_l
            + 12 * coeffs[4] * theta_l**2
        )
        qacc_matrix[:, idx] = d2q_dtheta2 * (theta_dot_l**2) + dq_dtheta * theta_ddot_l

    return qpos_matrix, qvel_matrix, qacc_matrix

def butter_lowpass_filter(data, cutoff=3, fs=100.5, order=4):
    """Apply Butterworth low-pass filter to data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data, axis=0)

def setup_and_precompute_jacobians(mj_model, qpos_matrix, qvel_matrix):
    """
    One-time setup: compute Jacobians for entire trajectory.
    
    Returns:
        mj_model, mjx_model, body_ids, jacobian_data
    """
    
    # Find calcaneus bodies
    calcn_r_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'calcn_r')
    calcn_l_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'calcn_l')
    body_ids = [calcn_r_id, calcn_l_id]

    
    # Pre-compute Jacobians
    print("\nPre-computing Jacobians for trajectory...")
    n_timesteps = qpos_matrix.shape[0]
    n_bodies = len(body_ids)
    nv = mj_model.nv
    
    # Storage: [timestep, body, direction, dof]
    jacp_all = np.zeros((n_timesteps, n_bodies, 3, nv))
    jacr_all = np.zeros((n_timesteps, n_bodies, 3, nv))
    
    mj_data = mujoco.MjData(mj_model)
    
    for t in range(n_timesteps):
        # Set configuration
        mj_data.qpos[:] = qpos_matrix[t, :]
        mj_data.qvel[:] = qvel_matrix[t, :]
        
        # CRITICAL: Update kinematics AND dynamics so Jacobians are computed correctly
        # mj_kinematics only updates positions, not the quantities needed for Jacobians
        mujoco.mj_forward(mj_model, mj_data)  # Full forward pass
        
        # Compute Jacobian for each foot
        for i, body_id in enumerate(body_ids):
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            # Use MuJoCo's built-in Jacobian computation (correct!)
            mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, body_id)
            
            jacp_all[t, i] = jacp
            jacr_all[t, i] = jacr
            
            # Debug: Check if Jacobian is non-zero
            if t == 0:
                print(f"  Body {body_id} (index {i}): jacp norm = {np.linalg.norm(jacp):.6f}, jacr norm = {np.linalg.norm(jacr):.6f}")
        
        if t % 500 == 0:
            print(f"  Processed {t}/{n_timesteps} timesteps")
    
    print(f"✓ Jacobians computed for all {n_timesteps} timesteps")
    
    # Create MJX model
    mjx_model = mjx.put_model(mj_model)
    
    jacobian_data = {
        'jacp': jnp.array(jacp_all),
        'jacr': jnp.array(jacr_all),
        'body_ids': jnp.array(body_ids)
    }
    
    return mj_model, mjx_model, body_ids, jacobian_data

@jax.jit
def compute_inverse_dynamics(model, data, qpos, qvel, qacc):
    data = data.replace(qpos=qpos, qvel=qvel, qacc=qacc)
    return mjx.inverse(model, data)

# Path to muscle-free MuJoCo model
model_path = "Results/ModelNoMus/scaled_model_no_muscles_cvt2_FIXED.xml"

# Remove Incompatible MJX elements

# Load MuJoCo model and convert to MJX
mj_model = mujoco.MjModel.from_xml_path(model_path)
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.make_data(mjx_model)

# Update Model
mjx_model, mj_model, fixed_xml_path = update_model(xml_path="Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml",
    min_mass=0.25,
    min_inertia=0.01,
    min_armature=0.01
)

# Load in data files:
    # positions, velocities, accelerations 
        # Unit Check
        # map to matrix
            # num_timesteps x 39 (if we include the patella)
        # calc coupled generalized coordinates
        # resample to match model timestep
        # filter data?
        # Trim data to match length of force plate data
    # GRF, Moment, and COP Matrix formation
        # Foot Recognition: Which foot is which.
        # map to matrix (To MuJoCo format)
            # GRF_Left:num_timesteps x 6 (Fx ,Fy, Fz)
            # GRF_Right:num_timesteps x 6 (Fx ,Fy, Fz)
            # Moment_Left:num_timesteps x 6 (Mx ,My, Mz)
            # Moment_Right:num_timesteps x 6 (Mx ,My, Mz)
            # COP:num_timesteps x 4 (L_COPx, L_COPy, R_COPx, R_COPy)
        # Notes:
            # OpenSim:
                # X axis: Forward
                # Y axis: Up
                # Z axis: Right
            # MuJoCo:
                # X axis: Forward (X in openSim)
                # Y axis: Left (negative Z in openSim)
                # Z axis: Up (Y in openSim)
    # Inverse Dynamics Computation
        # Store Jacobian matrices for each foot at each time step
        # Compute ID (Not taking into account external forces yet)
        # Store Ankle Position

# Load motion data
data_path = "PatientData/Falisse_2017_subject_01/"
pos_data = pd.read_csv(data_path + "pos.csv")
vel_data = pd.read_csv(data_path + "vel.csv")
acc_data = pd.read_csv(data_path + "acc.csv")

# Load Force Plate Data
grf_data_raw = pd.read_csv(data_path + "grf.csv", header=None, skiprows=1)
moment_data_raw = pd.read_csv(data_path + "moment.csv", header=None, skiprows=1)
cop_data_raw = pd.read_csv(data_path + "cop.csv", header=None, skiprows=1)

grf_matrix = grf_data_raw.values  # Get all rows and columns (208, 10)
moment_matrix = moment_data_raw.values  # Get all rows and columns (208, 10)
cop_matrix = cop_data_raw.values  # Get all rows and columns (208, 10)

# I had to include this for formatting issues
cop_data = pd.DataFrame(cop_matrix, columns=[f'col_{i}' for i in range(cop_matrix.shape[1])])
cop_data.insert(0, 'time', grf_data_raw.iloc[:, 0].values)  # Ensure time is first column

# Resample all data to uniform timestep
pos_data, vel_data, acc_data, moment_matrix, grf_matrix, cop_data = resample_dataframes_to_uniform_timestep(
    pos_data, vel_data, acc_data, moment_matrix, grf_matrix, cop_data, dt=0.01)

#=============================================
# ==============TRIM DATA ====================
#=============================================

# Mapping pos,vel, and acc data to MuJoCo generalized coordinates
# Notes: If we remove the patella, we'll need to adjust the size of the mapping matrix.
num_timesteps = len(pos_data)
qpos_matrix = np.zeros((num_timesteps, mj_model.nq))
qvel_matrix = np.zeros((num_timesteps, mj_model.nv))
qacc_matrix = np.zeros((num_timesteps, mj_model.nv))

for i in range(num_timesteps):
    qpos_matrix[i, :] = map_patient_to_qpos(pos_data.iloc[i], mj_model.nq)
    qvel_matrix[i, :] = map_patient_to_qpos(vel_data.iloc[i], mj_model.nv)
    qacc_matrix[i, :] = map_patient_to_qpos(acc_data.iloc[i], mj_model.nv)

# Calculate coupled knee joint coordinates, velocities, and accelerations
# NEED TO EXTRACT POLYNOMIALS FROM THE MODEL FILE INSTEAD OF HARD CODING
qpos_matrix, qvel_matrix, qacc_matrix = calculate_coupled_coordinates_automated(qpos_matrix, qvel_matrix, qacc_matrix,model_path)

#Filtering Data:
qacc_matrix_filtered = butter_lowpass_filter(qacc_matrix, cutoff=6, fs=100.5, order=4)
qvel_matrix_filtered = butter_lowpass_filter(qvel_matrix, cutoff=6, fs=100.5, order=4)
qpos_matrix_filtered = butter_lowpass_filter(qpos_matrix, cutoff=6, fs=100.5, order=4)

qacc_matrix = qacc_matrix_filtered
qvel_matrix = qvel_matrix_filtered
qpos_matrix = qpos_matrix_filtered


# NON-AUTOMATED GRF CODE
grf_left_opensim = grf_matrix[:, 4:7]  # Left foot GRF columns [Fx, Fy, Fz] in OpenSim
grf_right_opensim = grf_matrix[:, 1:4]  # Right foot GRF columns [Fx, Fy, Fz] in OpenSim

# Transform to MuJoCo coordinates: [X, Y, Z]_mujoco = [X, Z, Y]_opensim
grf_left = np.column_stack([grf_left_opensim[:, 0],   # X stays X
                             -1*grf_left_opensim[:, 2],   # Z becomes Y
                             grf_left_opensim[:, 1]])  # Y becomes Z (vertical)

grf_right = np.column_stack([grf_right_opensim[:, 0],  # X stays X  
                              -1*grf_right_opensim[:, 2],  # Z becomes Y
                              grf_right_opensim[:, 1]]) # Y becomes Z (vertical)

moment_left_opensim = moment_matrix[:, 4:7]  # Left foot moments
moment_right_opensim = moment_matrix[:, 1:4]  # Right foot moments

# Transform moments to MuJoCo coordinates as well
moment_left = np.column_stack([moment_left_opensim[:, 0],   # X stays X
                                -1*moment_left_opensim[:, 2],   # Z becomes Y
                                moment_left_opensim[:, 1]])  # Y becomes Z

moment_right = np.column_stack([moment_right_opensim[:, 0],  # X stays X
                                 -1*moment_right_opensim[:, 2],  # Z becomes Y
                                 moment_right_opensim[:, 1]]) # Y becomes Z



#Initialize Storage Variables:
joint_forces_over_time = np.zeros((num_timesteps, mjx_model.nv))
ankle_pos_l_all = np.zeros((num_timesteps, 3))
ankle_pos_r_all = np.zeros((num_timesteps, 3))

# Find calcaneus body IDs
calcn_l_id = mj_model.body('calcn_l').id  # Left calcaneus body ID (should be 12)
calcn_r_id = mj_model.body('calcn_r').id  # Right calcaneus body ID (should be 6)


#=============================================
#==============CHECK UNITS====================
#=============================================

#=============================================
#=============Foot Recognition================
#=============================================

a, b, c, jacobian_data = setup_and_precompute_jacobians(mj_model, qpos_matrix, qvel_matrix)

for t in range(num_timesteps):
    current_mjx_data = compute_inverse_dynamics(mjx_model, current_mjx_data, 
                                                    qpos_matrix[t, :], qvel_matrix[t, :], 
                                                    qacc_matrix[t, :])

    ankle_pos_l = current_mjx_data.xpos[calcn_l_id]  # Left ankle (calcaneus) world position
    ankle_pos_r = current_mjx_data.xpos[calcn_r_id]  # Right ankle (calcaneus) world position
    ankle_pos_l_all[t] = ankle_pos_l
    ankle_pos_r_all[t] = ankle_pos_r

#=============================================
    qfrc_inverse = current_mjx_data.qfrc_inverse
    qfrc_constraint = current_mjx_data.qfrc_constraint
    joint_forces_over_time[t, :] = qfrc_inverse + qfrc_constraint # Removes the effects of constraint forces.
