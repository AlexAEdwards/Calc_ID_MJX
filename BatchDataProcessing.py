from pyexpat import model
import shutil
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

# Mapping from .npy file column indices to MuJoCo qpos indices
# Based on pos.csv column order (excluding time column):
# Columns: pelvis_tilt, pelvis_list, pelvis_rotation, pelvis_tx, pelvis_ty, pelvis_tz,
#          hip_flexion_r, hip_adduction_r, hip_rotation_r, knee_angle_r, ankle_angle_r,
#          subtalar_angle_r, mtp_angle_r, hip_flexion_l, hip_adduction_l, hip_rotation_l,
#          knee_angle_l, ankle_angle_l, subtalar_angle_l, mtp_angle_l,
#          lumbar_extension, lumbar_bending, lumbar_rotation
npy_column_to_qpos = {
    # Column index in .npy: qpos index in MuJoCo
    0: 3,   # pelvis_tilt -> qpos[3]
    1: 4,   # pelvis_list -> qpos[4]
    2: 5,   # pelvis_rotation -> qpos[5]
    3: 0,   # pelvis_tx -> qpos[0]
    4: 1,   # pelvis_ty -> qpos[1]
    5: 2,   # pelvis_tz -> qpos[2]
    6: 6,   # hip_flexion_r -> qpos[6]
    7: 7,   # hip_adduction_r -> qpos[7]
    8: 8,   # hip_rotation_r -> qpos[8]
    9: 11,  # knee_angle_r -> qpos[11]
    10: 14, # ankle_angle_r -> qpos[14]
    11: 15, # subtalar_angle_r -> qpos[15]
    12: 16, # mtp_angle_r -> qpos[16]
    13: 21, # hip_flexion_l -> qpos[21]
    14: 22, # hip_adduction_l -> qpos[22]
    15: 23, # hip_rotation_l -> qpos[23]
    16: 26, # knee_angle_l -> qpos[26]
    17: 29, # ankle_angle_l -> qpos[29]
    18: 30, # subtalar_angle_l -> qpos[30]
    19: 31, # mtp_angle_l -> qpos[31]
    20: 36, # lumbar_extension -> qpos[36]
    21: 37, # lumbar_bending -> qpos[37]
    22: 38, # lumbar_rotation -> qpos[38]
}

def trim_data_by_grf(pos, vel, accel, grf, moment, cop, time=None):
    """
    Trims all arrays in-place based on when GRF data is nonzero.
    Uses any nonzero value in GRF (any column) to determine start/end.
    Returns trimmed arrays (views, not copies) and a new time vector starting at zero.
    Assumes all arrays have the same length along axis 0.
    If time is provided, trims and shifts it as well.
    """
    # Find indices where any GRF column is nonzero
    grf_sum = np.abs(grf).sum(axis=1)
    nonzero_indices = np.where(grf_sum > 0)[0]
    if len(nonzero_indices) == 0:
        # No nonzero GRF found, return empty arrays and empty time
        if time is not None:
            return (pos[0:0], vel[0:0], accel[0:0], grf[0:0], moment[0:0], cop[0:0], time[0:0])
        else:
            return (pos[0:0], vel[0:0], accel[0:0], grf[0:0], moment[0:0], cop[0:0], np.array([]))
    start = nonzero_indices[0]
    end = nonzero_indices[-1] + 1  # +1 for slicing
    # Trim time and shift to zero
    if time is not None:
        new_time = time[start:end] - time[start]
    else:
        new_time = np.arange(end - start)
    return (
        pos[start:end],
        vel[start:end],
        accel[start:end],
        grf[start:end],
        moment[start:end],
        cop[start:end],
        new_time
    )

def convert_to_mujoco_coords(vecA, vecB):
    if vecA.ndim == 1:
        vecA_mj = jnp.array([vecA[0], -vecA[2], vecA[1]])
        vecB_mj = jnp.array([vecB[0], -vecB[2], vecB[1]])
        return vecA_mj, vecB_mj
    if vecA.ndim == 2:
        vecA_mj = jnp.array([vecA[:,0], -vecA[:,2], vecA[:,1]])
        vecB_mj = jnp.array([vecB[:,0], -vecB[:,2], vecB[:,1]])
        return vecA_mj, vecB_mj
    else:
        raise ValueError("Input arrays must be 1D or 2D.")

def compute_grf_contribution(model, external_forces,
                          jacp_t, jacr_t, body_ids_array):
    """
    Compute GRF contribution to joint torques using Jacobian transpose method.
    
    Args:
        model: MJX model
        external_forces: (nbody, 6) - GRFs on all bodies [torque(3), force(3)]
        jacp_t: (n_bodies, 3, nv) - Position Jacobians at timestep t
        jacr_t: (n_bodies, 3, nv) - Rotation Jacobians at timestep t
        body_ids_array: (n_bodies,) - Body IDs with GRFs
    
    Returns:
        qfrc_from_grf: (nv,) - Joint torques from GRF contribution
    """
    
    # Compute GRF contribution using pre-computed Jacobians
    qfrc_from_grf = jnp.zeros(model.nv)
    
    for i in range(len(body_ids_array)):
        body_id = body_ids_array[i]  # Get the actual body ID
        
        # Index external_forces by body_id (full array of all bodies)
        xfrc = external_forces[body_id]  # (6,) [torque(3), force(3)]
        
        # Extract torque and force components  
        # NOTE: MuJoCo xfrc format is [torque(3), force(3)]
        force = xfrc[:3]
        torque = xfrc[3:]
        
        # Get the corresponding Jacobian for THIS body
        # jacp_t and jacr_t are shape (n_bodies, 3, nv) where n_bodies=len(body_ids_array)
        # Index by i (0, 1, ...) to get Jacobian for i-th body in body_ids_array
        jacp = jacp_t[i]  # (3, nv) - Jacobian for body_ids_array[i]
        jacr = jacr_t[i]  # (3, nv) - Jacobian for body_ids_array[i]
        
        # Map to joint space: tau = J^T @ F
        # Force contribution: J_p^T @ force
        # Torque contribution: J_r^T @ torque
        qfrc_from_body = jacp.T @ force + jacr.T @ torque
        qfrc_from_grf += qfrc_from_body
    return qfrc_from_grf

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
    """
    Map a single row of patient data to MuJoCo qpos array.
    
    Parameters:
    -----------
    patient_row : numpy array or pd.Series
        Single row of patient data. If numpy array, assumes it's from .npy file with columns
        in the order defined by npy_column_to_qpos mapping.
        If pd.Series, uses column names with qpos_mapping.
    qpos_size : int
        Size of qpos array (default: 39)
    
    Returns:
    --------
    qpos : numpy array
        MuJoCo qpos array with mapped values
    """
    qpos = np.zeros(qpos_size)
    
    # Check if it's a pandas Series (has column names) or numpy array (indexed)
    if hasattr(patient_row, 'index'):
        # Pandas Series - use name-based mapping
        for col_name, qpos_idx in qpos_mapping.items():
            if col_name in patient_row.index:
                qpos[qpos_idx] = patient_row[col_name]
    else:
        # Numpy array - use index-based mapping from .npy files
        for npy_col_idx, qpos_idx in npy_column_to_qpos.items():
            if npy_col_idx < len(patient_row):
                qpos[qpos_idx] = patient_row[npy_col_idx]
    
    return qpos

def resample_dataframes_to_uniform_timestep(pos_data, vel_data, acc_data, time_vector=None, moment_matrix=None, grf_matrix=None, cop_data=None, dt=0.01):
    """
    Resample pos_data, vel_data, acc_data, and optionally moment_matrix, grf_matrix, cop_data 
    so that their time vectors are consistent and uniformly spaced at dt (default 0.01s).
    
    Parameters:
    -----------
    pos_data, vel_data, acc_data : pd.DataFrame
        DataFrames with kinematic data (NO time column - just joint data)
    time_vector : numpy array, optional
        Time vector corresponding to the kinematic data. If None, assumes first column is time.
    moment_matrix, grf_matrix : numpy array, optional
        Force/moment matrices (NO time column - just force/moment data)
    cop_data : pd.DataFrame, optional
        COP data (NO time column)
    dt : float
        Target uniform time step
    
    Returns:
    --------
    Resampled data with uniform time spacing
    """
    import numpy as np
    import pandas as pd
    
    # Handle time vector
    if time_vector is None:
        # Legacy behavior: first column is time
        t_pos = pos_data.iloc[:, 0].values
        t_vel = vel_data.iloc[:, 0].values
        t_acc = acc_data.iloc[:, 0].values
        # Remove time column from data
        pos_data = pos_data.iloc[:, 1:]
        vel_data = vel_data.iloc[:, 1:]
        acc_data = acc_data.iloc[:, 1:]
    else:
        # Use provided time vector for all kinematic data
        t_pos = time_vector
        t_vel = time_vector
        t_acc = time_vector
    
    t_list = [t_pos, t_vel, t_acc]
    data_list = [pos_data, vel_data, acc_data]
    names = ['pos_data', 'vel_data', 'acc_data']
    
    # Handle optional matrices (they also use the same time vector)
    if moment_matrix is not None:
        t_list.append(time_vector if time_vector is not None else moment_matrix[:, 0])
        data_list.append(moment_matrix)
        names.append('moment_matrix')
    
    if grf_matrix is not None:
        t_list.append(time_vector if time_vector is not None else grf_matrix[:, 0])
        data_list.append(grf_matrix)
        names.append('grf_matrix')
    
    if cop_data is not None:
        t_list.append(time_vector if time_vector is not None else cop_data.iloc[:, 0].values)
        data_list.append(cop_data)
        names.append('cop_data')
    
    # Use the union of all time points, then create a uniform grid
    t_min = max(t[0] for t in t_list)
    t_max = min(t[-1] for t in t_list)
    t_uniform = np.arange(t_min, t_max + dt/2, dt)
    
    print("\n--- Resampling Data ---")
    for name, data in zip(names, data_list):
        orig_len = len(data) if isinstance(data, pd.DataFrame) else data.shape[0]
        print(f"{name}: before = {orig_len} timesteps, after = {len(t_uniform)} timesteps")
    
    # Interpolate each dataset to the new time base
    def interp_data(data, t_orig, t_uniform):
        if isinstance(data, pd.DataFrame):
            # DataFrame - interpolate each column
            data_interp = np.empty((len(t_uniform), data.shape[1]))
            for col in range(data.shape[1]):
                data_interp[:, col] = np.interp(t_uniform, t_orig, data.iloc[:, col].values)
            return pd.DataFrame(data_interp, columns=data.columns)
        else:
            # Numpy array - interpolate each column
            data_interp = np.empty((len(t_uniform), data.shape[1]))
            for col in range(data.shape[1]):
                data_interp[:, col] = np.interp(t_uniform, t_orig, data[:, col])
            return data_interp
    
    # Resample all data
    resampled = []
    for data, t_orig in zip(data_list, t_list):
        resampled.append(interp_data(data, t_orig, t_uniform))
    
    # Build output tuple
    out = []
    idx = 0
    for arg, name in zip([pos_data, vel_data, acc_data, moment_matrix, grf_matrix, cop_data],
                         ['pos_data', 'vel_data', 'acc_data', 'moment_matrix', 'grf_matrix', 'cop_data']):
        if arg is not None:
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


def identify_and_organize_grf(grf_data, moment_data, ankle_pos_l_all, ankle_pos_r_all, threshold=0.05, dt=0.01):
    """
    Identify which GRF and moment columns belong to which foot and organize them into a standardized format.
    
    This function reads GRF and moment data (from file paths or numpy arrays), identifies which columns 
    contain non-zero data, groups them into sets of 3 (X, Y, Z), and assigns each set to left or 
    right foot based on correlation with ankle height during ground contact.
    
    Parameters:
    -----------
    grf_data : str or numpy array
        Either path to the GRF .npy file OR numpy array with shape (num_timesteps, 6)
    moment_data : str or numpy array
        Either path to the moment .npy file OR numpy array with shape (num_timesteps, 6)
    ankle_pos_l_all : numpy array (num_timesteps, 3)
        Left ankle positions over time [x, y, z] where z is vertical
    ankle_pos_r_all : numpy array (num_timesteps, 3)
        Right ankle positions over time [x, y, z] where z is vertical
    threshold : float, optional
        Threshold for determining non-zero GRF values (default: 0.05)
    dt : float, optional
        Time step for generating time vector (default: 0.01)
    
    Returns:
    --------
    GRF_Cleaned : numpy array (num_timesteps, 7)
        Organized GRF data in MuJoCo coordinate frame with columns:
        [time, Fx_r, Fy_r, Fz_r, Fx_l, Fy_l, Fz_l]
        where:
        - Columns 1-3: Right foot GRF (X, Y, Z) in MuJoCo frame
        - Columns 4-6: Left foot GRF (X, Y, Z) in MuJoCo frame
    
    Moment_Cleaned : numpy array (num_timesteps, 7)
        Organized moment data in MuJoCo coordinate frame with columns:
        [time, Mx_r, My_r, Mz_r, Mx_l, My_l, Mz_l]
        where:
        - Columns 1-3: Right foot moment (X, Y, Z) in MuJoCo frame
        - Columns 4-6: Left foot moment (X, Y, Z) in MuJoCo frame
    
    Coordinate Frame Transformation:
    --------------------------------
    Input Data (OpenSim-like):
        X-axis: Forward
        Y-axis: Up
        Z-axis: Right
    
    MuJoCo:
        X-axis: Forward (same as input X)
        Y-axis: Left (negative of input Z)
        Z-axis: Up (same as input Y)
    
    Transformation applied:
        MuJoCo_X = Input_X
        MuJoCo_Y = -Input_Z
        MuJoCo_Z = Input_Y
    
    Algorithm:
    ----------
    1. Load the GRF .npy file (must have 6 columns)
    2. Identify columns with non-zero data
    3. Group non-zero columns into sets of 3 (assuming X, Y, Z order)
    4. For each set, determine which foot it belongs to by:
       - Finding timesteps where GRF magnitude is above threshold
       - Computing mean ankle height during those contact periods
       - Foot with lower mean height during contact = that foot's GRF
    5. Organize into standardized format: [time, R_x, R_y, R_z, L_x, L_y, L_z]
    6. Transform from input coordinate frame to MuJoCo coordinate frame
    7. Repeat steps 1-6 for moment data using same foot assignments
    
    Example:
    --------
    >>> GRF_Cleaned, Moment_Cleaned = identify_and_organize_grf(
    ...     "data/GRF.npy",
    ...     "data/GRM.npy",
    ...     ankle_pos_l_all,
    ...     ankle_pos_r_all
    ... )
    """
    
    print(f"\n{'='*70}")
    print(f"GRF and Moment Foot Identification and Organization")
    print(f"{'='*70}")
    
    # Load or use GRF data
    if isinstance(grf_data, str):
        # It's a file path - load from file
        if not grf_data.endswith('.npy'):
            raise ValueError(f"GRF file must be .npy format. Got: {grf_data}")
        print(f"\nLoading GRF from NumPy file: {grf_data}")
        grf_array = np.load(grf_data)
    else:
        # It's already a numpy array
        print(f"\nUsing provided GRF array")
        grf_array = grf_data
    
    print(f"  Data shape: {grf_array.shape}")
    
    # Validate shape
    if len(grf_array.shape) != 2:
        raise ValueError(f"GRF data must be 2D array. Got shape: {grf_array.shape}")
    
    if grf_array.shape[1] != 6:
        raise ValueError(f"GRF data must have exactly 6 columns (3 per foot). Got {grf_array.shape[1]} columns")
    
    # Generate time column
    num_timesteps = grf_array.shape[0]
    time_col = np.arange(num_timesteps) * dt
    
    # Convert to DataFrame for consistent processing
    col_names = [f'GRF_{i}' for i in range(6)]
    grf_data = pd.DataFrame(grf_array, columns=col_names)
    print(f"  Generated time vector: 0 to {time_col[-1]:.2f}s with dt={dt}s")
    print(f"  Number of timesteps: {num_timesteps}")
    
    # Identify columns with non-zero data
    nonzero_cols = []
    for i in range(grf_data.shape[1]):
        col_data = grf_data.iloc[:, i].values
        if np.any(np.abs(col_data) > threshold):
            nonzero_cols.append(i)
    
    print(f"\nNon-zero columns detected: {len(nonzero_cols)}")
    print(f"  Column indices: {nonzero_cols}")
    
    # Group columns into sets of 3 (X, Y, Z)
    if len(nonzero_cols) % 3 != 0:
        print(f"\n⚠️  WARNING: Number of non-zero columns ({len(nonzero_cols)}) is not divisible by 3!")
        print(f"  Expected groups of 3 for X, Y, Z components")
        # Pad or truncate to nearest multiple of 3
        if len(nonzero_cols) > 3:
            nonzero_cols = nonzero_cols[:6]  # Take first 6 (2 feet × 3 components)
            print(f"  Truncated to first 6 columns: {nonzero_cols}")
    
    num_groups = len(nonzero_cols) // 3
    grf_groups = []
    
    for g in range(num_groups):
        group_cols = nonzero_cols[g*3:(g+1)*3]
        group_data = grf_data.iloc[:, group_cols].values  # (num_timesteps, 3)
        grf_groups.append({
            'indices': group_cols,
            'data': group_data,
            'column_names': [grf_data.columns[i] for i in group_cols]
        })
        print(f"\nGroup {g+1}:")
        print(f"  Columns: {group_cols}")
        print(f"  Names: {grf_groups[g]['column_names']}")
    
    # Determine which group belongs to which foot
    print(f"\n{'='*70}")
    print(f"Foot Assignment Based on Ankle Height During Contact")
    print(f"{'='*70}")
    
    foot_assignments = []  # Will store 'left' or 'right' for each group
    
    for g, group in enumerate(grf_groups):
        # Calculate GRF magnitude for this group
        grf_magnitude = np.linalg.norm(group['data'], axis=1)
        
        # Find timesteps where there's significant ground contact
        contact_mask = grf_magnitude > threshold
        num_contact_frames = np.sum(contact_mask)
        
        if num_contact_frames == 0:
            print(f"\nGroup {g+1}: No contact detected (all values below threshold)")
            foot_assignments.append('unknown')
            continue
        
        # Get ankle heights (z-coordinate) during contact
        ankle_l_heights_contact = ankle_pos_l_all[contact_mask, 2]  # z is vertical
        ankle_r_heights_contact = ankle_pos_r_all[contact_mask, 2]
        
        # Calculate mean ankle height during contact for each foot
        mean_height_l = np.mean(ankle_l_heights_contact)
        mean_height_r = np.mean(ankle_r_heights_contact)
        
        # The foot with lower mean height during contact is on the ground
        if mean_height_l < mean_height_r:
            assigned_foot = 'left'
        else:
            assigned_foot = 'right'
        
        foot_assignments.append(assigned_foot)
        
        print(f"\nGroup {g+1} ({group['column_names']}):")
        print(f"  Contact frames: {num_contact_frames}/{num_timesteps}")
        print(f"  Mean left ankle height during contact: {mean_height_l:.4f} m")
        print(f"  Mean right ankle height during contact: {mean_height_r:.4f} m")
        print(f"  → Assigned to: {assigned_foot.upper()} foot")
    
    # Organize into standardized format: [time, R_x, R_y, R_z, L_x, L_y, L_z]
    print(f"\n{'='*70}")
    print(f"Creating GRF_Cleaned Matrix")
    print(f"{'='*70}")
    
    GRF_Cleaned = np.zeros((num_timesteps, 7))
    GRF_Cleaned[:, 0] = time_col
    
    # Assign each group to the correct columns
    right_assigned = False
    left_assigned = False
    
    for g, group in enumerate(grf_groups):
        if foot_assignments[g] == 'right' and not right_assigned:
            GRF_Cleaned[:, 1:4] = group['data']  # Columns 1-3: Right foot
            right_assigned = True
            print(f"✓ Right foot GRF: Columns {group['indices']} → GRF_Cleaned[:, 1:4]")
        elif foot_assignments[g] == 'left' and not left_assigned:
            GRF_Cleaned[:, 4:7] = group['data']  # Columns 4-6: Left foot
            left_assigned = True
            print(f"✓ Left foot GRF: Columns {group['indices']} → GRF_Cleaned[:, 4:7]")
    
    if not right_assigned:
        print(f"⚠️  WARNING: No GRF data assigned to RIGHT foot")
    if not left_assigned:
        print(f"⚠️  WARNING: No GRF data assigned to LEFT foot")
    
    # ============================================================================
    # COORDINATE FRAME TRANSFORMATION
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"Coordinate Frame Transformation")
    print(f"{'='*70}")
    print(f"\nInput Data Coordinate System:")
    print(f"  X-axis: Forward")
    print(f"  Y-axis: Up")
    print(f"  Z-axis: Right")
    print(f"\nMuJoCo Coordinate System:")
    print(f"  X-axis: Forward (same as input X)")
    print(f"  Y-axis: Left (negative of input Z)")
    print(f"  Z-axis: Up (same as input Y)")
    print(f"\nTransformation:")
    print(f"  MuJoCo_X = Input_X  (forward → forward)")
    print(f"  MuJoCo_Y = -Input_Z (right → left, sign flip)")
    print(f"  MuJoCo_Z = Input_Y  (up → up)")
    
    # Store original values for comparison
    GRF_original = GRF_Cleaned.copy()
    
    # Apply coordinate transformation to both feet
    # Right foot: columns 1-3 (Fx, Fy, Fz in input frame)
    if right_assigned:
        input_Fx_r = GRF_Cleaned[:, 1].copy()  # Forward
        input_Fy_r = GRF_Cleaned[:, 2].copy()  # Up
        input_Fz_r = GRF_Cleaned[:, 3].copy()  # Right
        
        GRF_Cleaned[:, 1] = input_Fx_r      # MuJoCo X = Input X (forward)
        GRF_Cleaned[:, 2] = -input_Fz_r     # MuJoCo Y = -Input Z (left)
        GRF_Cleaned[:, 3] = input_Fy_r      # MuJoCo Z = Input Y (up)
    
    # Left foot: columns 4-6 (Fx, Fy, Fz in input frame)
    if left_assigned:
        input_Fx_l = GRF_Cleaned[:, 4].copy()  # Forward
        input_Fy_l = GRF_Cleaned[:, 5].copy()  # Up
        input_Fz_l = GRF_Cleaned[:, 6].copy()  # Right
        
        GRF_Cleaned[:, 4] = input_Fx_l      # MuJoCo X = Input X (forward)
        GRF_Cleaned[:, 5] = -input_Fz_l     # MuJoCo Y = -Input Z (left)
        GRF_Cleaned[:, 6] = input_Fy_l      # MuJoCo Z = Input Y (up)
    
    print(f"\n✓ Coordinate transformation applied")
    print(f"\nExample transformation (first non-zero frame):")
    
    # Find first frame with significant GRF
    if right_assigned:
        right_mag = np.linalg.norm(GRF_original[:, 1:4], axis=1)
        nonzero_idx = np.where(right_mag > threshold)[0]
        if len(nonzero_idx) > 0:
            idx = nonzero_idx[0]
            print(f"\n  Right foot at t={GRF_Cleaned[idx, 0]:.3f}s:")
            print(f"    Input:  Fx={GRF_original[idx, 1]:7.2f}, Fy={GRF_original[idx, 2]:7.2f}, Fz={GRF_original[idx, 3]:7.2f} N")
            print(f"    MuJoCo: Fx={GRF_Cleaned[idx, 1]:7.2f}, Fy={GRF_Cleaned[idx, 2]:7.2f}, Fz={GRF_Cleaned[idx, 3]:7.2f} N")
    
    if left_assigned:
        left_mag = np.linalg.norm(GRF_original[:, 4:7], axis=1)
        nonzero_idx = np.where(left_mag > threshold)[0]
        if len(nonzero_idx) > 0:
            idx = nonzero_idx[0]
            print(f"\n  Left foot at t={GRF_Cleaned[idx, 0]:.3f}s:")
            print(f"    Input:  Fx={GRF_original[idx, 4]:7.2f}, Fy={GRF_original[idx, 5]:7.2f}, Fz={GRF_original[idx, 6]:7.2f} N")
            print(f"    MuJoCo: Fx={GRF_Cleaned[idx, 4]:7.2f}, Fy={GRF_Cleaned[idx, 5]:7.2f}, Fz={GRF_Cleaned[idx, 6]:7.2f} N")
    
    # Verify magnitude is preserved (rotation should preserve vector magnitude)
    if right_assigned:
        mag_original_r = np.linalg.norm(GRF_original[:, 1:4], axis=1)
        mag_transformed_r = np.linalg.norm(GRF_Cleaned[:, 1:4], axis=1)
        max_diff_r = np.max(np.abs(mag_original_r - mag_transformed_r))
        print(f"\n  Right foot magnitude preservation: max diff = {max_diff_r:.2e} N (should be ~0)")
    
    if left_assigned:
        mag_original_l = np.linalg.norm(GRF_original[:, 4:7], axis=1)
        mag_transformed_l = np.linalg.norm(GRF_Cleaned[:, 4:7], axis=1)
        max_diff_l = np.max(np.abs(mag_original_l - mag_transformed_l))
        print(f"  Left foot magnitude preservation: max diff = {max_diff_l:.2e} N (should be ~0)")
    
    # ============================================================================
    
    print(f"\nGRF_Cleaned shape: {GRF_Cleaned.shape}")
    print(f"  Column 0: Time")
    print(f"  Columns 1-3: Right foot (Fx, Fy, Fz) in MuJoCo frame")
    print(f"  Columns 4-6: Left foot (Fx, Fy, Fz) in MuJoCo frame")
    
    # Summary statistics
    print(f"\nGRF Summary Statistics (MuJoCo frame):")
    print(f"  Right foot GRF magnitude: {np.linalg.norm(GRF_Cleaned[:, 1:4], axis=1).max():.2f} N (max)")
    print(f"  Left foot GRF magnitude: {np.linalg.norm(GRF_Cleaned[:, 4:7], axis=1).max():.2f} N (max)")
    
    print(f"\n{'='*70}")
    print(f"✓ GRF Processing Complete")
    print(f"{'='*70}\n")
    
    
    # ============================================================================
    # PROCESS MOMENT DATA (SAME LOGIC AS GRF)
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"Moment Data Processing")
    print(f"{'='*70}")
    
    # Load or use moment data
    if isinstance(moment_data, str):
        # It's a file path - load from file
        if not moment_data.endswith('.npy'):
            raise ValueError(f"Moment file must be .npy format. Got: {moment_data}")
        print(f"\nLoading Moment from NumPy file: {moment_data}")
        moment_array = np.load(moment_data)
    else:
        # It's already a numpy array
        print(f"\nUsing provided Moment array")
        moment_array = moment_data
    
    print(f"  Data shape: {moment_array.shape}")
    
    # Validate shape
    if len(moment_array.shape) != 2:
        raise ValueError(f"Moment data must be 2D array. Got shape: {moment_array.shape}")
    
    if moment_array.shape[1] != 6:
        raise ValueError(f"Moment data must have exactly 6 columns (3 per foot). Got {moment_array.shape[1]} columns")
    
    # Verify timesteps match GRF
    if moment_array.shape[0] != num_timesteps:
        print(f"\n⚠️  WARNING: Moment has {moment_array.shape[0]} timesteps, GRF has {num_timesteps}")
        print(f"  Proceeding with moment data length")
    
    # Convert to DataFrame for consistent processing
    col_names = [f'GRM_{i}' for i in range(6)]
    moment_data = pd.DataFrame(moment_array, columns=col_names)
    
    # Identify columns with non-zero data
    moment_nonzero_cols = []
    for i in range(moment_data.shape[1]):
        col_data = moment_data.iloc[:, i].values
        if np.any(np.abs(col_data) > threshold):
            moment_nonzero_cols.append(i)
    
    print(f"\nNon-zero moment columns detected: {len(moment_nonzero_cols)}")
    print(f"  Column indices: {moment_nonzero_cols}")
    
    # Group columns into sets of 3 (Mx, My, Mz)
    if len(moment_nonzero_cols) % 3 != 0:
        print(f"\n⚠️  WARNING: Number of non-zero moment columns ({len(moment_nonzero_cols)}) is not divisible by 3!")
        print(f"  Expected groups of 3 for Mx, My, Mz components")
        if len(moment_nonzero_cols) > 3:
            moment_nonzero_cols = moment_nonzero_cols[:6]  # Take first 6 (2 feet × 3 components)
            print(f"  Truncated to first 6 columns: {moment_nonzero_cols}")
    
    num_moment_groups = len(moment_nonzero_cols) // 3
    moment_groups = []
    
    for g in range(num_moment_groups):
        group_cols = moment_nonzero_cols[g*3:(g+1)*3]
        group_data = moment_data.iloc[:, group_cols].values  # (num_timesteps, 3)
        moment_groups.append({
            'indices': group_cols,
            'data': group_data,
            'column_names': [moment_data.columns[i] for i in group_cols]
        })
        print(f"\nMoment Group {g+1}:")
        print(f"  Columns: {group_cols}")
        print(f"  Names: {moment_groups[g]['column_names']}")
    
    # Use the SAME foot assignments as GRF (moments should match GRF foot assignment)
    print(f"\n{'='*70}")
    print(f"Moment Foot Assignment (Using GRF-based assignments)")
    print(f"{'='*70}")
    
    moment_foot_assignments = []
    
    for g, group in enumerate(moment_groups):
        # Use GRF magnitude to determine contact (moments correspond to GRF)
        # We'll use the foot_assignments from GRF processing
        if g < len(foot_assignments):
            assigned_foot = foot_assignments[g]
            moment_foot_assignments.append(assigned_foot)
            print(f"\nMoment Group {g+1} ({group['column_names']}):")
            print(f"  → Assigned to: {assigned_foot.upper()} foot (matching GRF assignment)")
        else:
            print(f"\nMoment Group {g+1}: No corresponding GRF assignment")
            moment_foot_assignments.append('unknown')
    
    # Organize moments into standardized format: [time, M_r_x, M_r_y, M_r_z, M_l_x, M_l_y, M_l_z]
    print(f"\n{'='*70}")
    print(f"Creating Moment_Cleaned Matrix")
    print(f"{'='*70}")
    
    Moment_Cleaned = np.zeros((num_timesteps, 7))
    Moment_Cleaned[:, 0] = time_col
    
    # Assign each moment group to the correct columns
    moment_right_assigned = False
    moment_left_assigned = False
    
    for g, group in enumerate(moment_groups):
        if moment_foot_assignments[g] == 'right' and not moment_right_assigned:
            Moment_Cleaned[:, 1:4] = group['data']  # Columns 1-3: Right foot
            moment_right_assigned = True
            print(f"✓ Right foot moment: Columns {group['indices']} → Moment_Cleaned[:, 1:4]")
        elif moment_foot_assignments[g] == 'left' and not moment_left_assigned:
            Moment_Cleaned[:, 4:7] = group['data']  # Columns 4-6: Left foot
            moment_left_assigned = True
            print(f"✓ Left foot moment: Columns {group['indices']} → Moment_Cleaned[:, 4:7]")
    
    if not moment_right_assigned:
        print(f"⚠️  WARNING: No moment data assigned to RIGHT foot")
    if not moment_left_assigned:
        print(f"⚠️  WARNING: No moment data assigned to LEFT foot")
    
    # ============================================================================
    # COORDINATE FRAME TRANSFORMATION FOR MOMENTS
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"Moment Coordinate Frame Transformation")
    print(f"{'='*70}")
    print(f"\nInput Data Coordinate System:")
    print(f"  X-axis: Forward")
    print(f"  Y-axis: Up")
    print(f"  Z-axis: Right")
    print(f"\nMuJoCo Coordinate System:")
    print(f"  X-axis: Forward (same as input X)")
    print(f"  Y-axis: Left (negative of input Z)")
    print(f"  Z-axis: Up (same as input Y)")
    print(f"\nTransformation:")
    print(f"  MuJoCo_Mx = Input_Mx  (forward → forward)")
    print(f"  MuJoCo_My = -Input_Mz (right → left, sign flip)")
    print(f"  MuJoCo_Mz = Input_My  (up → up)")
    
    # Store original values for comparison
    Moment_original = Moment_Cleaned.copy()
    
    # Apply coordinate transformation to both feet
    # Right foot: columns 1-3 (Mx, My, Mz in input frame)
    if moment_right_assigned:
        input_Mx_r = Moment_Cleaned[:, 1].copy()  # Forward
        input_My_r = Moment_Cleaned[:, 2].copy()  # Up
        input_Mz_r = Moment_Cleaned[:, 3].copy()  # Right
        
        Moment_Cleaned[:, 1] = input_Mx_r      # MuJoCo X = Input X (forward)
        Moment_Cleaned[:, 2] = -input_Mz_r     # MuJoCo Y = -Input Z (left)
        Moment_Cleaned[:, 3] = input_My_r      # MuJoCo Z = Input Y (up)
    
    # Left foot: columns 4-6 (Mx, My, Mz in input frame)
    if moment_left_assigned:
        input_Mx_l = Moment_Cleaned[:, 4].copy()  # Forward
        input_My_l = Moment_Cleaned[:, 5].copy()  # Up
        input_Mz_l = Moment_Cleaned[:, 6].copy()  # Right
        
        Moment_Cleaned[:, 4] = input_Mx_l      # MuJoCo X = Input X (forward)
        Moment_Cleaned[:, 5] = -input_Mz_l     # MuJoCo Y = -Input Z (left)
        Moment_Cleaned[:, 6] = input_My_l      # MuJoCo Z = Input Y (up)
    
    print(f"\n✓ Coordinate transformation applied to moments")
    print(f"\nExample moment transformation (first non-zero frame):")
    
    # Find first frame with significant moment
    if moment_right_assigned:
        right_mag = np.linalg.norm(Moment_original[:, 1:4], axis=1)
        nonzero_idx = np.where(right_mag > threshold)[0]
        if len(nonzero_idx) > 0:
            idx = nonzero_idx[0]
            print(f"\n  Right foot at t={Moment_Cleaned[idx, 0]:.3f}s:")
            print(f"    Input:  Mx={Moment_original[idx, 1]:7.2f}, My={Moment_original[idx, 2]:7.2f}, Mz={Moment_original[idx, 3]:7.2f} Nm")
            print(f"    MuJoCo: Mx={Moment_Cleaned[idx, 1]:7.2f}, My={Moment_Cleaned[idx, 2]:7.2f}, Mz={Moment_Cleaned[idx, 3]:7.2f} Nm")
    
    if moment_left_assigned:
        left_mag = np.linalg.norm(Moment_original[:, 4:7], axis=1)
        nonzero_idx = np.where(left_mag > threshold)[0]
        if len(nonzero_idx) > 0:
            idx = nonzero_idx[0]
            print(f"\n  Left foot at t={Moment_Cleaned[idx, 0]:.3f}s:")
            print(f"    Input:  Mx={Moment_original[idx, 4]:7.2f}, My={Moment_original[idx, 5]:7.2f}, Mz={Moment_original[idx, 6]:7.2f} Nm")
            print(f"    MuJoCo: Mx={Moment_Cleaned[idx, 4]:7.2f}, My={Moment_Cleaned[idx, 5]:7.2f}, Mz={Moment_Cleaned[idx, 6]:7.2f} Nm")
    
    # Verify magnitude is preserved (rotation should preserve vector magnitude)
    if moment_right_assigned:
        mag_original_r = np.linalg.norm(Moment_original[:, 1:4], axis=1)
        mag_transformed_r = np.linalg.norm(Moment_Cleaned[:, 1:4], axis=1)
        max_diff_r = np.max(np.abs(mag_original_r - mag_transformed_r))
        print(f"\n  Right foot magnitude preservation: max diff = {max_diff_r:.2e} Nm (should be ~0)")
    
    if moment_left_assigned:
        mag_original_l = np.linalg.norm(Moment_original[:, 4:7], axis=1)
        mag_transformed_l = np.linalg.norm(Moment_Cleaned[:, 4:7], axis=1)
        max_diff_l = np.max(np.abs(mag_original_l - mag_transformed_l))
        print(f"  Left foot magnitude preservation: max diff = {max_diff_l:.2e} Nm (should be ~0)")
    
    # ============================================================================
    
    print(f"\nMoment_Cleaned shape: {Moment_Cleaned.shape}")
    print(f"  Column 0: Time")
    print(f"  Columns 1-3: Right foot (Mx, My, Mz) in MuJoCo frame")
    print(f"  Columns 4-6: Left foot (Mx, My, Mz) in MuJoCo frame")
    
    # Summary statistics
    print(f"\nMoment Summary Statistics (MuJoCo frame):")
    print(f"  Right foot moment magnitude: {np.linalg.norm(Moment_Cleaned[:, 1:4], axis=1).max():.2f} Nm (max)")
    print(f"  Left foot moment magnitude: {np.linalg.norm(Moment_Cleaned[:, 4:7], axis=1).max():.2f} Nm (max)")
    
    print(f"\n{'='*70}")
    print(f"✓ Moment Processing Complete")
    print(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print(f"✓ GRF and Moment Foot Identification Complete")
    print(f"{'='*70}\n")
    
    return GRF_Cleaned, Moment_Cleaned

# Path to muscle-free MuJoCo model
model_path = "Results/ModelNoMus/scaled_model_no_muscles_cvt2_FIXED.xml"

# Remove Incompatible MJX elements

def process_trial(subject_path, trial_path, mj_model, mjx_model):
    """
    Process a single trial for inverse dynamics computation.
    
    Parameters:
    -----------
    subject_path : str
        Path to subject folder (e.g., "Data/P044_split0")
    trial_path : str
        Path to trial folder (e.g., "Data/P044_split0/Trial_1")
    mj_model : MjModel
        MuJoCo model for the subject
    mjx_model : mjx.Model
        MJX model for the subject
    """
    
    print(f"\n  Processing trial: {os.path.basename(trial_path)}")
    
    # Define paths
    motion_path = os.path.join(trial_path, "Motion") + "/"
    calculated_path = os.path.join(trial_path, "calculatedInputs") + "/"
    
    # Ensure calculatedInputs directory exists
    os.makedirs(calculated_path, exist_ok=True)
    
    # Initialize MJX data
    mjx_data = mjx.make_data(mjx_model)
    
    # Load motion data from .npy files
    print("    Loading data from .npy files...")
    pos_array = np.load(motion_path + "Pos.npy")
    vel_array = np.load(motion_path + "Vel.npy")
    acc_array = np.load(motion_path + "Accel.npy")
    time_array = np.load(motion_path + "Time.npy")  # Load time vector

    # Convert to DataFrames WITHOUT time column (time will be passed separately)
    pos_data = pd.DataFrame(pos_array, columns=[f'joint_{i}' for i in range(pos_array.shape[1])])
    vel_data = pd.DataFrame(vel_array, columns=[f'joint_{i}' for i in range(vel_array.shape[1])])
    acc_data = pd.DataFrame(acc_array, columns=[f'joint_{i}' for i in range(acc_array.shape[1])])

    # Load Force Plate Data from .npy files
    print("    Loading GRF/GRM/COP from .npy files...")
    grf_matrix = np.load(motion_path + "GRF.npy")  # Shape: (num_timesteps, 6)
    moment_matrix = np.load(motion_path + "GRM.npy")  # Shape: (num_timesteps, 6)
    cop_matrix = np.load(motion_path + "COP.npy")  # Shape: (num_timesteps, 6 or 4)

    print(f"      GRF shape: {grf_matrix.shape}")
    print(f"      GRM shape: {moment_matrix.shape}")
    print(f"      COP shape: {cop_matrix.shape}")

    # Validate shapes
    if grf_matrix.shape[1] != 6:
        raise ValueError(f"GRF.npy must have 6 columns, got {grf_matrix.shape[1]}")
    if moment_matrix.shape[1] != 6:
        raise ValueError(f"GRM.npy must have 6 columns, got {moment_matrix.shape[1]}")


    # Resample all data to uniform timestep using the time_array
    dt = 0.01
    pos_array, vel_array, acc_array, moment_matrix, grf_matrix, cop_matrix = resample_dataframes_to_uniform_timestep(
        pos_data, vel_data, acc_data, time_vector=time_array, 
        moment_matrix=moment_matrix, grf_matrix=grf_matrix, cop_data=cop_matrix, dt=dt)

    # Filter Pos_vel_acc Data
    pos_array = butter_lowpass_filter(pos_array, cutoff=12, fs=1/dt, order=4)
    vel_array = butter_lowpass_filter(vel_array, cutoff=12, fs=1/dt, order=4)
    acc_array = butter_lowpass_filter(acc_array, cutoff=12, fs=1/dt, order=4)


    # TRIM DATA TO MATCH GRF NON-ZERO FRAMES
    pos_array, vel_array, acc_array, grf_matrix, moment_matrix, cop_matrix, time_array = trim_data_by_grf(pos_array, vel_array, acc_array, grf_matrix, moment_matrix, cop_matrix, time_array)

    # Mapping pos,vel, and acc data to MuJoCo generalized coordinates
    # After resampling, get the number of timesteps from the RESAMPLED data
    if isinstance(pos_array, pd.DataFrame):
        num_timesteps = len(pos_array)
        pos_array = pos_array.values
        vel_array = vel_array.values
        acc_array = acc_array.values
    else:
        num_timesteps = pos_array.shape[0]
    
    qpos_matrix = np.zeros((num_timesteps, mj_model.nq))
    qvel_matrix = np.zeros((num_timesteps, mj_model.nv))
    qacc_matrix = np.zeros((num_timesteps, mj_model.nv))

    # # Extract data arrays directly (no time column to skip)
    # pos_array_resampled = pos_data.values
    # vel_array_resampled = vel_data.values
    # acc_array_resampled = acc_data.values

    # Map using npy_column_to_qpos dictionary (works with numpy arrays)
    for i in range(num_timesteps):
        qpos_matrix[i, :] = map_patient_to_qpos(pos_array[i], mj_model.nq)
        qvel_matrix[i, :] = map_patient_to_qpos(vel_array[i], mj_model.nv)
        qacc_matrix[i, :] = map_patient_to_qpos(acc_array[i], mj_model.nv)

    # Calculate coupled knee joint coordinates, velocities, and accelerations
    model_xml_path = os.path.join(subject_path, "MyosuiteModel_FIXED.xml")
    qpos_matrix, qvel_matrix, qacc_matrix = calculate_coupled_coordinates_automated(
        qpos_matrix, qvel_matrix, qacc_matrix, model_xml_path)

    #Initialize Storage Variables:
    ID_Results_MJX = np.zeros((num_timesteps, mjx_model.nv))
    ankle_pos_l_all = np.zeros((num_timesteps, 3))
    ankle_pos_r_all = np.zeros((num_timesteps, 3))

    r_vec_l_all = np.zeros((num_timesteps, 3))
    r_vec_r_all = np.zeros((num_timesteps, 3))

    # Find calcaneus body IDs
    calcn_l_id = mj_model.body('calcn_l').id  # Left calcaneus body ID
    calcn_r_id = mj_model.body('calcn_r').id  # Right calcaneus body ID

    #=============================================
    #=============Foot Recognition================
    #=============================================

    a, b, c, jacobian_data = setup_and_precompute_jacobians(mj_model, qpos_matrix, qvel_matrix)

    for t in range(num_timesteps):
        current_mjx_data = compute_inverse_dynamics(mjx_model, mjx_data, 
                                                        qpos_matrix[t, :], qvel_matrix[t, :], 
                                                        qacc_matrix[t, :])

        ankle_pos_l = current_mjx_data.xpos[calcn_l_id]  # Left ankle (calcaneus) world position
        ankle_pos_r = current_mjx_data.xpos[calcn_r_id]  # Right ankle (calcaneus) world position
        ankle_pos_l_all[t] = ankle_pos_l
        ankle_pos_r_all[t] = ankle_pos_r

        qfrc_inverse = current_mjx_data.qfrc_inverse
        qfrc_constraint = current_mjx_data.qfrc_constraint
        ID_Results_MJX[t, :] = qfrc_inverse + qfrc_constraint # Removes the effects of constraint forces.

    # Identify and organize GRF/GRM based on ankle positions
    # Use the already-resampled grf_matrix and moment_matrix (not the files)
    # GRF_Cleaned, Moment_Cleaned = identify_and_organize_grf(
    #     grf_matrix, moment_matrix, ankle_pos_l_all, ankle_pos_r_all, threshold=0.05, dt=dt)
    grf_left=grf_matrix[:, 3:6]
    grf_right=grf_matrix[:, 0:3]
    # print size of grf_left and grf_right
    print("grf_left shape:", grf_left.shape)
    print("grf_right shape:", grf_right.shape)
    moment_left=moment_matrix[:, 3:6]
    moment_right=moment_matrix[:, 0:3]
    cop_right_all=cop_matrix[:, 0:3]
    cop_left_all=cop_matrix[:, 3:6]

    ID_Results_MJX_NoGRFContrib = ID_Results_MJX.copy()
    

    # Convert to MuJoCo coordinates
    grf_leftMJX, grf_rightMJX = convert_to_mujoco_coords(grf_left,grf_right)
    moment_leftMJX, moment_rightMJX = convert_to_mujoco_coords(moment_left,moment_right)
    cop_leftMJX, cop_rightMJX = convert_to_mujoco_coords(cop_left_all,cop_right_all)

    # combine grf_leftMJX and grf_rightMJX into GRF_Cleaned
    GRF_Cleaned = jnp.concatenate([grf_rightMJX, grf_leftMJX], axis=0).T
    Moment_Cleaned = jnp.concatenate([moment_rightMJX, moment_leftMJX], axis=0).T
    COP_Cleaned = jnp.concatenate([cop_rightMJX, cop_leftMJX], axis=0).T

    print("GRF_Cleaned shape:", GRF_Cleaned.shape)
    print("Moment_Cleaned shape:", Moment_Cleaned.shape)
    print("COP_Cleaned shape:", COP_Cleaned.shape)
    # Initialize External_Force array for MJX
    nb = mjx_model.nbody
    External_Force=jnp.zeros((nb,6,num_timesteps))

    for t in range(num_timesteps):
        grf_left=GRF_Cleaned[t, 3:6]
        grf_right=GRF_Cleaned[t, 0:3]
        moment_left=Moment_Cleaned[t, 3:6]
        moment_right=Moment_Cleaned[t, 0:3]
        cop_right=cop_matrix[t, 0:3]  # Assuming COP columns 0-2 are right foot
        cop_left=cop_matrix[t, 3:6]  # Assuming COP columns 3-5 are left foot

        # Converts from 
        cop_left, cop_right = convert_to_mujoco_coords(cop_left, cop_right)
        
        # Add to External_Force Array
        External_Force = External_Force.at[calcn_l_id, 0:3, t].set(grf_left.T)  
        External_Force = External_Force.at[calcn_r_id, 0:3, t].set(grf_right.T)

        # External_Force = External_Force.at[calcn_l_id, 3:6, t].set(moment_left.T)  
        # External_Force = External_Force.at[calcn_r_id, 3:6, t].set(moment_right.T)

        r_vec_r = cop_right - ankle_pos_r_all[t]
        r_vec_l = cop_left - ankle_pos_l_all[t]
        #print magnitude of r_vec
        # print("Magnitude of r_vec_l:", jnp.linalg.norm(r_vec_l))
        # print("Magnitude of r_vec_r:", jnp.linalg.norm(r_vec_r))
        
        # If the magnitude of GRF is zero set r_vec to zero otherwise set to r_vel_l/r_vec_r
        if jnp.linalg.norm(grf_left) < 1e-3:
            r_vec_l = jnp.array([0.0, 0.0, 0.0])
        if jnp.linalg.norm(grf_right) < 1e-3:
            r_vec_r = jnp.array([0.0, 0.0, 0.0])
        r_vec_l_all[t] = r_vec_l
        r_vec_r_all[t] = r_vec_r

        moment_added_r = jnp.cross(r_vec_r, grf_right)
        moment_added_l = jnp.cross(r_vec_l, grf_left)


        External_Force = External_Force.at[calcn_l_id, 3:6, t].set(moment_left+moment_added_l)
        External_Force = External_Force.at[calcn_r_id, 3:6, t].set(moment_right+moment_added_r)

        qfrc_grf_contribution = compute_grf_contribution(mjx_model, External_Force[:,:,t], 
                                                        jacobian_data['jacp'][t], jacobian_data['jacr'][t], 
                                                        jacobian_data['body_ids'])
        
        ID_Results_MJX[t, :] -= qfrc_grf_contribution


    # Trim trial based on whether r_vec_l_all is bad
    r_vec_l_magnitudes = jnp.linalg.norm(r_vec_l_all, axis=1)
    r_vec_r_magnitudes = jnp.linalg.norm(r_vec_r_all, axis=1)
    bad_indices_left = jnp.where(r_vec_l_magnitudes > 0.3)[0]
    bad_indices_right = jnp.where(r_vec_r_magnitudes > 0.3)[0]
    num_timesteps = r_vec_l_all.shape[0]
    bad_indices = jnp.unique(jnp.concatenate((bad_indices_left, bad_indices_right)))
    bad_percentage = (len(bad_indices) / num_timesteps) * 100
    if bad_percentage > 10:
        # Delete the trial and all folders associated with it
        print(f"    ❌ Trial has {bad_percentage:.2f}% of frames with bad COP (>30cm). Deleting trial folder.")
        shutil.rmtree(trial_path)
        return
    else:
        print(f"    ✅ Trial has {bad_percentage:.2f}% of frames with bad COP (>30cm). Keeping trial.")

    np.save(os.path.join(calculated_path, "bad_COP_percentage30cm.npy"), np.array(bad_percentage))

    # Save outputs to calculatedInputs folder
    print("    Saving results...")
    np.save(os.path.join(calculated_path, "qfrc_grf_contribution.npy"), qfrc_grf_contribution)
    np.save(os.path.join(calculated_path, "ID_Results_MJX.npy"), ID_Results_MJX)
    np.save(os.path.join(calculated_path, "anklePos.npy"), np.stack([ankle_pos_l_all, ankle_pos_r_all], axis=0))
    np.save(os.path.join(calculated_path, "Jacobian.npy"), jacobian_data)
    np.save(os.path.join(calculated_path, "ID_Results_MJX_NoGRFContrib.npy"), ID_Results_MJX_NoGRFContrib)

    # Save cleaned GRF/GRM to Motion folder
    # Create a subfolder within the Motion folder called mjx
    mjx_path = os.path.join(motion_path, "mjx")
    os.makedirs(mjx_path, exist_ok=True)
    np.save(os.path.join(mjx_path, "GRF_Cleaned.npy"), GRF_Cleaned)
    np.save(os.path.join(mjx_path, "Moment_Cleaned.npy"), Moment_Cleaned)
    np.save(os.path.join(mjx_path, "COP_Cleaned.npy"), COP_Cleaned)
    # Save qpos, qvel, qacc to Motion folder
    np.save(os.path.join(mjx_path, "pos_mjx.npy"), qpos_matrix)
    np.save(os.path.join(mjx_path, "vel_mjx.npy"), qvel_matrix)
    np.save(os.path.join(mjx_path, "acc_mjx.npy"), qacc_matrix)

    np.save(os.path.join(calculated_path, "External_Force.npy"), np.array(External_Force))
    print("    ✓ Trial processing complete! for subject " + os.path.basename(subject_path) + ", trial " + os.path.basename(trial_path))


def batch_process_all_subjects(data_root="Data", num_subjects="all"):
    """
    Batch process subjects and trials in the Data folder.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing all subject folders (default: "Data")
    num_subjects : int or str
        Number of subjects to process. Can be:
        - "all" (default): Process all subjects
        - Integer (e.g., 5): Process only the first N subjects
    
    Examples:
    ---------
    >>> batch_process_all_subjects()  # Process all subjects
    >>> batch_process_all_subjects(num_subjects=5)  # Process only the first 5 subjects
    >>> batch_process_all_subjects(num_subjects="all")  # Process all subjects
    """
    from pathlib import Path
    
    print("="*70)
    print("BATCH PROCESSING - INVERSE DYNAMICS")
    print("="*70)
    
    data_path = Path(data_root)
    
    # Find all subject folders (folders containing MyosuiteModel_FIXED.xml)
    subjects = []
    for item in sorted(data_path.iterdir()):
        if item.is_dir() and (item / "MyosuiteModel.xml").exists():
            subjects.append(item)
    
    if not subjects:
        print(f"\n❌ No subjects with MyosuiteModel_FIXED.xml found in {data_root}")
        print("   Please run the model conversion first!")
        return
    
    # Determine how many subjects to process
    if isinstance(num_subjects, str) and num_subjects.lower() == "all":
        subjects_to_process = subjects
        print(f"\nProcessing ALL {len(subjects)} subjects\n")
    elif isinstance(num_subjects, int):
        subjects_to_process = subjects[:num_subjects]
        print(f"\nProcessing first {len(subjects_to_process)} of {len(subjects)} subjects\n")
    else:
        print(f"\n❌ Invalid num_subjects parameter: {num_subjects}")
        print("   Must be 'all' or an integer")
        return
    
    total_trials = 0
    successful_trials = 0
    failed_trials = 0
    failed_list = []
    
    for subject_idx, subject_path in enumerate(subjects_to_process, 1):
        subject_name = subject_path.name
        print(f"\n[{subject_idx}/{len(subjects_to_process)}] Processing subject: {subject_name}")
        print("-" * 70)
        
        # Load subject's model
        model_xml_path = subject_path / "MyosuiteModel.xml"
        
        try:
            # Load and update model
            print(f"  Loading model: {model_xml_path.name}")
            mjx_model, mj_model, _ = update_model(
                xml_path=str(model_xml_path),
                min_mass=0.25,
                min_inertia=0.01,
                min_armature=0.01
            )
            print(f"  ✓ Model loaded successfully")
            
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            continue
        
        # Find all trial folders
        trial_folders = []
        for item in sorted(subject_path.iterdir()):
            if item.is_dir() and item.name.startswith("Trial_"):
                # Check if Motion folder exists
                motion_folder = item / "Motion"
                if motion_folder.exists():
                    trial_folders.append(item)
        
        if not trial_folders:
            print(f"  ⚠️  No trial folders found for {subject_name}")
            continue
        
        print(f"  Found {len(trial_folders)} trials")
        
        # Process each trial
        for trial_path in trial_folders:
            total_trials += 1
            try:
                process_trial(str(subject_path), str(trial_path), mj_model, mjx_model)
                successful_trials += 1
            except Exception as e:
                failed_trials += 1
                failed_name = f"{subject_name}/{trial_path.name}"
                failed_list.append((failed_name, str(e)))
                print(f"    ❌ Failed: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Subjects processed: {len(subjects_to_process)} of {len(subjects)} total")
    print(f"Total trials processed: {total_trials}")
    print(f"Successful: {successful_trials}")
    print(f"Failed: {failed_trials}")
    
    if failed_list:
        print(f"\nFailed trials:")
        for trial_name, error in failed_list:
            print(f"  - {trial_name}: {error}")
    
    print("="*70)


# Main execution
if __name__ == "__main__":
    num_subjects = "all"
    # You can change num_subjects to:
    # - "all" to process all subjects (default)
    # - An integer (e.g., 5) to process only the first N subjects
    time_start = time.time()
    batch_process_all_subjects(data_root="Data_Full_Cleaned", num_subjects=num_subjects)
    time_end = time.time()
    print(f"Total processing time: {time_end - time_start:.2f} seconds")

    data_root = "Data_Full_Cleaned"
    from pathlib import Path
    data_path = Path(data_root)
    
    # Find all subject folders (folders containing MyosuiteModel_FIXED.xml)
    subjects = []
    for item in sorted(data_path.iterdir()):
        if item.is_dir() and (item / "MyosuiteModel.xml").exists():
            subjects.append(item)
    # Print subject processing Names:
    subject_to_process = subjects[:num_subjects] if isinstance(num_subjects, int) else subjects
    for subject_path in subject_to_process:
        print(f"Processed subject: {subject_path.name}")
    # Examples:
    # batch_process_all_subjects(data_root="Data", num_subjects=1)   # Process only first subject
    # batch_process_all_subjects(data_root="Data", num_subjects=5)   # Process first 5 subjects
    # batch_process_all_subjects(data_root="Data", num_subjects="all")  # Process all subjects



# FUTURE IMPROVEMENTS:
# Check for GRF miss-steps
# Check for GRF and moment are correctly assigned
# Simplify Print Statements
# Organize output is a bit better 
# Add loading bars to both patient and trial processing
# COP convention
# COP 