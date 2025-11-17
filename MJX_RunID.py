
from pyexpat import model
import mujoco
from mujoco import mjx
import numpy as np
import pandas as pd
import mujoco.viewer
import time
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib
# matplotlib.use('Agg')  # Commented out to allow interactive plotting
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def butter_lowpass_filter(data, cutoff=3, fs=100.5, order=4):
    """Apply Butterworth low-pass filter to data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data, axis=0)

def calculate_knee_coupled_velocities(qvel_matrix, qpos_matrix):
    """
    Calculate coupled knee joint velocities from polynomial constraints.
    For each coupled coordinate: dq/dt = dq/dtheta * dtheta/dt
    where dq/dtheta is the derivative of the polynomial w.r.t. the main knee angle.
    """
    qvel_matrix = qvel_matrix.copy()

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

    # Right knee: main angle index 11, velocity index 11
    theta_r = qpos_matrix[:, 11]
    theta_dot_r = qvel_matrix[:, 11]
    for idx, coeffs in right_knee_polys.items():
        # Derivative of polynomial: dq/dtheta = c1 + 2*c2*theta + 3*c3*theta^2 + 4*c4*theta^3
        dq_dtheta = (
            coeffs[1]
            + 2 * coeffs[2] * theta_r
            + 3 * coeffs[3] * theta_r**2
            + 4 * coeffs[4] * theta_r**3
        )
        qvel_matrix[:, idx] = dq_dtheta * theta_dot_r

    # Left knee: main angle index 26, velocity index 26
    theta_l = qpos_matrix[:, 26]
    theta_dot_l = qvel_matrix[:, 26]
    for idx, coeffs in left_knee_polys.items():
        dq_dtheta = (
            coeffs[1]
            + 2 * coeffs[2] * theta_l
            + 3 * coeffs[3] * theta_l**2
            + 4 * coeffs[4] * theta_l**3
        )
        qvel_matrix[:, idx] = dq_dtheta * theta_dot_l

    return qvel_matrix
    """
    Calculate coupled knee joint coordinates from polynomial constraints.
    These are 4th-order polynomials: q = c0 + c1*θ + c2*θ² + c3*θ³ + c4*θ⁴
    where θ is the main knee angle (knee_angle_r or knee_angle_l).
    
    Parameters:
    -----------
    qpos_matrix : numpy array (num_timesteps, 39)
        Position matrix with main knee angles already filled
    
    Returns:
    --------
    qpos_matrix : numpy array (num_timesteps, 39)
        Updated matrix with coupled joint coordinates calculated
    """
    
    # Right knee polynomials (driven by knee_angle_r at index 11)
    right_knee_polys = {
        9: [9.877e-08, 0.00324, -0.00239, 0.0005816, 5.886e-07],  # walker_knee_r_translation1
        10: [7.949e-11, 0.006076, -0.001298, -2.706e-06, 6.452e-07],  # walker_knee_r_translation2
        12: [-1.473e-08, 0.0791, -0.03285, -0.02522, 0.01083],  # walker_knee_r_rotation2
        13: [1.089e-08, 0.3695, -0.1695, 0.02516, 3.505e-07],  # walker_knee_r_rotation3
        17: [0.05515, -0.0158, -0.03583, 0.01403, -0.000925],  # patellofemoral_r_translation1
        18: [-0.01121, -0.05052, 0.009607, 0.01364, -0.003621],  # patellofemoral_r_translation2
        19: [0.00284182, 0, 0, 0, 0],  # patellofemoral_r_translation3 (constant)
        20: [0.01051, 0.02476, -1.316, 0.7163, -0.1383],  # patellofemoral_r_rotation1
    }
    
    # Left knee polynomials (driven by knee_angle_l at index 26)
    left_knee_polys = {
        24: [9.877e-08, 0.00324, -0.00239, 0.0005816, 5.886e-07],  # walker_knee_l_translation1
        25: [-7.949e-11, -0.006076, 0.001298, 2.706e-06, -6.452e-07],  # walker_knee_l_translation2
        27: [-1.473e-08, 0.0791, -0.03285, -0.02522, 0.01083],  # walker_knee_l_rotation2
        28: [-1.089e-08, -0.3695, 0.1695, -0.02516, -3.505e-07],  # walker_knee_l_rotation3
        32: [0.05515, -0.0158, -0.03583, 0.01403, -0.000925],  # patellofemoral_l_translation1
        33: [-0.01121, -0.05052, 0.009607, 0.01364, -0.003621],  # patellofemoral_l_translation2
        34: [-0.00284182, 0, 0, 0, 0],  # patellofemoral_l_translation3 (constant)
        35: [0.01051, 0.02476, -1.316, 0.7163, -0.1383],  # patellofemoral_l_rotation1
    }
    
    # Evaluate polynomials for right knee
    theta_r = qpos_matrix[:, 11]  # knee_angle_r
    for idx, coeffs in right_knee_polys.items():
        c = coeffs
        qpos_matrix[:, idx] = c[0] + c[1]*theta_r + c[2]*theta_r**2 + c[3]*theta_r**3 + c[4]*theta_r**4
    
    # Evaluate polynomials for left knee
    theta_l = qpos_matrix[:, 26]  # knee_angle_l
    for idx, coeffs in left_knee_polys.items():
        c = coeffs
        qpos_matrix[:, idx] = c[0] + c[1]*theta_l + c[2]*theta_l**2 + c[3]*theta_l**3 + c[4]*theta_l**4
    
    return qpos_matrix

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

def calculate_knee_coupled_accelerations(qacc_matrix, qpos_matrix, qvel_matrix):
    """
    Calculate coupled knee joint accelerations from polynomial constraints.
    For each coupled coordinate:
        d2q/dt2 = f''(theta) * (dtheta/dt)^2 + f'(theta) * (d2theta/dt2)
    """
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

    # Right knee: main angle index 11, velocity index 11, acceleration index 11
    theta_r = qpos_matrix[:, 11]
    theta_dot_r = qvel_matrix[:, 11]
    theta_ddot_r = qacc_matrix[:, 11]
    for idx, coeffs in right_knee_polys.items():
        # First derivative: dq/dtheta
        dq_dtheta = (
            coeffs[1]
            + 2 * coeffs[2] * theta_r
            + 3 * coeffs[3] * theta_r**2
            + 4 * coeffs[4] * theta_r**3
        )
        # Second derivative: d2q/dtheta2
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
        dq_dtheta = (
            coeffs[1]
            + 2 * coeffs[2] * theta_l
            + 3 * coeffs[3] * theta_l**2
            + 4 * coeffs[4] * theta_l**3
        )
        d2q_dtheta2 = (
            2 * coeffs[2]
            + 6 * coeffs[3] * theta_l
            + 12 * coeffs[4] * theta_l**2
        )
        qacc_matrix[:, idx] = d2q_dtheta2 * (theta_dot_l**2) + dq_dtheta * theta_ddot_l

    return qacc_matrix


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_calcaneus_forces(external_forces, pos_data, calcn_l_id, calcn_r_id):
    """Plot generalized forces applied to calcaneus bodies."""
    print("\n" + "="*70)
    print("VISUALIZING GENERALIZED FORCES ON CALCANEUS BODIES")
    print("="*70)

    # Extract forces for left and right calcaneus
    time_array = pos_data['time'].values
    forces_calcn_l = np.array(external_forces[calcn_l_id, :, :])  # Shape: (6, num_timesteps)
    forces_calcn_r = np.array(external_forces[calcn_r_id, :, :])  # Shape: (6, num_timesteps)

    # Create figure with subplots for forces and torques
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    fig.suptitle('Generalized Forces Applied to Calcaneus Bodies', fontsize=16, fontweight='bold')

    # Left calcaneus forces [Fx, Fy, Fz]
    ax = axes[0, 0]
    ax.set_facecolor('white')
    ax.plot(time_array, forces_calcn_l[0, :], linewidth=2, label='Fx (Anterior-Posterior)', color='red')
    ax.plot(time_array, forces_calcn_l[1, :], linewidth=2, label='Fy (Medial-Lateral)', color='green')
    ax.plot(time_array, forces_calcn_l[2, :], linewidth=2, label='Fz (Vertical)', color='blue')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Force (N)', fontsize=10)
    ax.set_title('Left Calcaneus - Forces', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    print(f"Left Calcaneus Forces:")
    print(f"  Fx: [{forces_calcn_l[0, :].min():.1f}, {forces_calcn_l[0, :].max():.1f}] N")
    print(f"  Fy: [{forces_calcn_l[1, :].min():.1f}, {forces_calcn_l[1, :].max():.1f}] N")
    print(f"  Fz: [{forces_calcn_l[2, :].min():.1f}, {forces_calcn_l[2, :].max():.1f}] N")

    # Right calcaneus forces [Fx, Fy, Fz]
    ax = axes[0, 1]
    ax.set_facecolor('white')
    ax.plot(time_array, forces_calcn_r[0, :], linewidth=2, label='Fx (Anterior-Posterior)', color='red')
    ax.plot(time_array, forces_calcn_r[1, :], linewidth=2, label='Fy (Medial-Lateral)', color='green')
    ax.plot(time_array, forces_calcn_r[2, :], linewidth=2, label='Fz (Vertical)', color='blue')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Force (N)', fontsize=10)
    ax.set_title('Right Calcaneus - Forces', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    print(f"Right Calcaneus Forces:")
    print(f"  Fx: [{forces_calcn_r[0, :].min():.1f}, {forces_calcn_r[0, :].max():.1f}] N")
    print(f"  Fy: [{forces_calcn_r[1, :].min():.1f}, {forces_calcn_r[1, :].max():.1f}] N")
    print(f"  Fz: [{forces_calcn_r[2, :].min():.1f}, {forces_calcn_r[2, :].max():.1f}] N")

    # Left calcaneus torques [Tx, Ty, Tz]
    ax = axes[1, 0]
    ax.set_facecolor('white')
    ax.plot(time_array, forces_calcn_l[3, :], linewidth=2, label='Tx (Roll)', color='darkred')
    ax.plot(time_array, forces_calcn_l[4, :], linewidth=2, label='Ty (Pitch)', color='darkgreen')
    ax.plot(time_array, forces_calcn_l[5, :], linewidth=2, label='Tz (Yaw)', color='darkblue')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Torque (N·m)', fontsize=10)
    ax.set_title('Left Calcaneus - Torques', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    print(f"Left Calcaneus Torques:")
    print(f"  Tx: [{forces_calcn_l[3, :].min():.1f}, {forces_calcn_l[3, :].max():.1f}] N·m")
    print(f"  Ty: [{forces_calcn_l[4, :].min():.1f}, {forces_calcn_l[4, :].max():.1f}] N·m")
    print(f"  Tz: [{forces_calcn_l[5, :].min():.1f}, {forces_calcn_l[5, :].max():.1f}] N·m")

    # Right calcaneus torques [Tx, Ty, Tz]
    ax = axes[1, 1]
    ax.set_facecolor('white')
    ax.plot(time_array, forces_calcn_r[3, :], linewidth=2, label='Tx (Roll)', color='darkred')
    ax.plot(time_array, forces_calcn_r[4, :], linewidth=2, label='Ty (Pitch)', color='darkgreen')
    ax.plot(time_array, forces_calcn_r[5, :], linewidth=2, label='Tz (Yaw)', color='darkblue')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Torque (N·m)', fontsize=10)
    ax.set_title('Right Calcaneus - Torques', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    print(f"Right Calcaneus Torques:")
    print(f"  Tx: [{forces_calcn_r[3, :].min():.1f}, {forces_calcn_r[3, :].max():.1f}] N·m")
    print(f"  Ty: [{forces_calcn_r[4, :].min():.1f}, {forces_calcn_r[4, :].max():.1f}] N·m")
    print(f"  Tz: [{forces_calcn_r[5, :].min():.1f}, {forces_calcn_r[5, :].max():.1f}] N·m")

    plt.tight_layout()
    print("\n✓ Displaying Calcaneus Forces Plot")
    print("="*70)

    plt.show()

def plot_moment_arm_vectors(time_vec, r_vec_r_all, r_vec_l_all):
    """Plot moment arm vectors (distance from COP to ankle)."""
    r_vec_r_all = jnp.array(r_vec_r_all)
    r_vec_l_all = jnp.array(r_vec_l_all)
    fig, axes = plt.subplots(1,2, figsize=(14, 5))

    axes[0].plot(time_vec, r_vec_r_all[:,0], label='Right Foot X', linewidth=2)
    axes[0].plot(time_vec, r_vec_r_all[:,1], label='Right Foot Y', linewidth=2)
    axes[0].plot(time_vec, r_vec_r_all[:,2], label='Right Foot Z', linewidth=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Distance (m)')
    axes[0].set_title('Moment Arm Vectors: Right Foot (Ankle - COP)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time_vec, r_vec_l_all[:,0], label='Left Foot X', linewidth=2)
    axes[1].plot(time_vec, r_vec_l_all[:,1], label='Left Foot Y', linewidth=2)
    axes[1].plot(time_vec, r_vec_l_all[:,2], label='Left Foot Z', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Distance (m)')
    axes[1].set_title('Moment Arm Vectors: Left Foot (Ankle - COP)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.show()

def plot_calculated_moments(time_vec, moment_added_l_matrix, moment_added_r_matrix):
    """Plot calculated moments from cross product M = r × F."""
    fig_moments, axes_moments = plt.subplots(3, 2, figsize=(14, 10), facecolor='white')
    fig_moments.suptitle('Calculated Moments from Cross Product: M = r × F (Ankle-COP distance × GRF)', 
                         fontsize=14, fontweight='bold')

    # Left foot moments
    axes_moments[0, 0].plot(time_vec, moment_added_l_matrix[:, 0], 'b-', linewidth=2)
    axes_moments[0, 0].set_ylabel('Mx (N⋅m)', fontsize=12)
    axes_moments[0, 0].set_title('Left Foot - X Component', fontsize=12, fontweight='bold')
    axes_moments[0, 0].grid(True, alpha=0.3)

    axes_moments[1, 0].plot(time_vec, moment_added_l_matrix[:, 1], 'g-', linewidth=2)
    axes_moments[1, 0].set_ylabel('My (N⋅m)', fontsize=12)
    axes_moments[1, 0].set_title('Left Foot - Y Component', fontsize=12, fontweight='bold')
    axes_moments[1, 0].grid(True, alpha=0.3)

    axes_moments[2, 0].plot(time_vec, moment_added_l_matrix[:, 2], 'r-', linewidth=2)
    axes_moments[2, 0].set_ylabel('Mz (N⋅m)', fontsize=12)
    axes_moments[2, 0].set_xlabel('Time (s)', fontsize=12)
    axes_moments[2, 0].set_title('Left Foot - Z Component', fontsize=12, fontweight='bold')
    axes_moments[2, 0].grid(True, alpha=0.3)

    # Right foot moments
    axes_moments[0, 1].plot(time_vec, moment_added_r_matrix[:, 0], 'b-', linewidth=2)
    axes_moments[0, 1].set_ylabel('Mx (N⋅m)', fontsize=12)
    axes_moments[0, 1].set_title('Right Foot - X Component', fontsize=12, fontweight='bold')
    axes_moments[0, 1].grid(True, alpha=0.3)

    axes_moments[1, 1].plot(time_vec, moment_added_r_matrix[:, 1], 'g-', linewidth=2)
    axes_moments[1, 1].set_ylabel('My (N⋅m)', fontsize=12)
    axes_moments[1, 1].set_title('Right Foot - Y Component', fontsize=12, fontweight='bold')
    axes_moments[1, 1].grid(True, alpha=0.3)

    axes_moments[2, 1].plot(time_vec, moment_added_r_matrix[:, 2], 'r-', linewidth=2)
    axes_moments[2, 1].set_ylabel('Mz (N⋅m)', fontsize=12)
    axes_moments[2, 1].set_xlabel('Time (s)', fontsize=12)
    axes_moments[2, 1].set_title('Right Foot - Z Component', fontsize=12, fontweight='bold')
    axes_moments[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_moment_magnitudes(time_vec, moment_added_l_matrix, moment_added_r_matrix):
    """Plot magnitude of calculated moments."""
    fig_mag, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig_mag.suptitle('Magnitude of Calculated Moments: ||M|| = ||r × F||', 
                     fontsize=14, fontweight='bold')

    moment_mag_l = np.linalg.norm(moment_added_l_matrix, axis=1)
    moment_mag_r = np.linalg.norm(moment_added_r_matrix, axis=1)

    ax_l.plot(time_vec, moment_mag_l, 'purple', linewidth=2)
    ax_l.set_xlabel('Time (s)', fontsize=12)
    ax_l.set_ylabel('Moment Magnitude (N⋅m)', fontsize=12)
    ax_l.set_title('Left Foot', fontsize=12, fontweight='bold')
    ax_l.grid(True, alpha=0.3)

    ax_r.plot(time_vec, moment_mag_r, 'orange', linewidth=2)
    ax_r.set_xlabel('Time (s)', fontsize=12)
    ax_r.set_ylabel('Moment Magnitude (N⋅m)', fontsize=12)
    ax_r.set_title('Right Foot', fontsize=12, fontweight='bold')
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_ankle_cop_distances(time_vec, distance_ankle_r_all, distance_ankle_l_all):
    """Plot distances between ankle and center of pressure."""
    plt.figure(figsize=(10, 5))
    plt.plot(time_vec, distance_ankle_r_all, label='Right Foot', linewidth=2)
    plt.plot(time_vec, distance_ankle_l_all, label='Left Foot', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title('Distance between Ankle and COP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_3d_trajectories(calcn_l_positions, cop_l_positions, calcn_r_positions, 
                         cop_r_positions, ankle_pos_l_all):
    """Plot 3D trajectories of calcaneus, ankle, and COP positions."""
    calcn_l_array = np.array(calcn_l_positions)
    cop_l_array = np.array(cop_l_positions)
    calcn_r_array = np.array(calcn_r_positions)
    cop_r_array = np.array(cop_r_positions)
    ankle_pos_l_all = np.array(ankle_pos_l_all)

    fig = plt.figure(figsize=(14, 6))

    # Left foot subplot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(calcn_l_array[:, 0], calcn_l_array[:, 1], calcn_l_array[:, 2], 
            label='Left Calcaneus', linewidth=2, color='blue')
    ax1.plot(cop_l_array[:, 0], cop_l_array[:, 1], cop_l_array[:, 2], 
            label='Left COP', linewidth=2, color='red', linestyle='--')
    ax1.plot(ankle_pos_l_all[:, 0], ankle_pos_l_all[:, 1], ankle_pos_l_all[:, 2], 
            label='Left Ankle', linewidth=2, color='green', linestyle='--')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Left Foot: Calcaneus and COP')
    ax1.legend()

    # Set equal aspect ratio for left foot
    all_left_data = np.vstack([calcn_l_array, cop_l_array, ankle_pos_l_all])
    max_range_l = np.array([all_left_data[:,0].max()-all_left_data[:,0].min(),
                            all_left_data[:,1].max()-all_left_data[:,1].min(),
                            all_left_data[:,2].max()-all_left_data[:,2].min()]).max() / 2.0
    mid_x_l = (all_left_data[:,0].max()+all_left_data[:,0].min()) * 0.5
    mid_y_l = (all_left_data[:,1].max()+all_left_data[:,1].min()) * 0.5
    mid_z_l = (all_left_data[:,2].max()+all_left_data[:,2].min()) * 0.5
    ax1.set_xlim(mid_x_l - max_range_l, mid_x_l + max_range_l)
    ax1.set_ylim(mid_y_l - max_range_l, mid_y_l + max_range_l)
    ax1.set_zlim(mid_z_l - max_range_l, mid_z_l + max_range_l)

    # Right foot subplot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(calcn_r_array[:, 0], calcn_r_array[:, 1], calcn_r_array[:, 2], 
            label='Right Calcaneus', linewidth=2, color='blue')
    ax2.plot(cop_r_array[:, 0], cop_r_array[:, 1], cop_r_array[:, 2], 
            label='Right COP', linewidth=2, color='red', linestyle='--')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Right Foot: Calcaneus and COP')
    ax2.legend()

    # Set equal aspect ratio for right foot
    all_right_data = np.vstack([calcn_r_array, cop_r_array])
    max_range_r = np.array([all_right_data[:,0].max()-all_right_data[:,0].min(),
                            all_right_data[:,1].max()-all_right_data[:,1].min(),
                            all_right_data[:,2].max()-all_right_data[:,2].min()]).max() / 2.0
    mid_x_r = (all_right_data[:,0].max()+all_right_data[:,0].min()) * 0.5
    mid_y_r = (all_right_data[:,1].max()+all_right_data[:,1].min()) * 0.5
    mid_z_r = (all_right_data[:,2].max()+all_right_data[:,2].min()) * 0.5
    ax2.set_xlim(mid_x_r - max_range_r, mid_x_r + max_range_r)
    ax2.set_ylim(mid_y_r - max_range_r, mid_y_r + max_range_r)
    ax2.set_zlim(mid_z_r - max_range_r, mid_z_r + max_range_r)

    plt.tight_layout()
    plt.show()

def plot_pelvis_trajectory(pelvis_positions):
    """Plot 3D trajectory of pelvis during gait."""
    pelvis_array = np.array(pelvis_positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pelvis_array[:, 0], pelvis_array[:, 1], pelvis_array[:, 2], 
            linewidth=2, color='purple', marker='o', markersize=2)
    ax.set_xlabel('X (m) - Anterior-Posterior')
    ax.set_ylabel('Y (m) - Medial-Lateral')
    ax.set_zlabel('Z (m) - Vertical')
    ax.set_title('Pelvis 3D Trajectory During Gait')
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    max_range = np.array([pelvis_array[:,0].max()-pelvis_array[:,0].min(),
                          pelvis_array[:,1].max()-pelvis_array[:,1].min(),
                          pelvis_array[:,2].max()-pelvis_array[:,2].min()]).max() / 2.0
    mid_x = (pelvis_array[:,0].max()+pelvis_array[:,0].min()) * 0.5
    mid_y = (pelvis_array[:,1].max()+pelvis_array[:,1].min()) * 0.5
    mid_z = (pelvis_array[:,2].max()+pelvis_array[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

def plot_inverse_dynamics_forces(joint_forces_array, time_array, joint_names):
    """Plot inverse dynamics forces organized by body region."""
    # Group DOFs for organized plotting
    dof_groups = {
        'Pelvis': [
            (3, 'pelvis_tilt'), (4, 'pelvis_list'), (5, 'pelvis_rotation'),
            (0, 'pelvis_tx'), (1, 'pelvis_ty'), (2, 'pelvis_tz')
        ],
        'Right Leg - Hip & Knee': [
            (6, 'hip_flexion_r'), (7, 'hip_adduction_r'), (8, 'hip_rotation_r'),
            (11, 'knee_angle_r')
        ],
        'Right Leg - Ankle & Foot': [
            (14, 'ankle_angle_r'), (15, 'subtalar_angle_r'), (16, 'mtp_angle_r')
        ],
        'Left Leg - Hip & Knee': [
            (21, 'hip_flexion_l'), (22, 'hip_adduction_l'), (23, 'hip_rotation_l'),
            (26, 'knee_angle_l')
        ],
        'Left Leg - Ankle & Foot': [
            (29, 'ankle_angle_l'), (30, 'subtalar_angle_l'), (31, 'mtp_angle_l')
        ],
        'Lumbar': [
            (36, 'lumbar_extension'), (37, 'lumbar_bending'), (38, 'lumbar_rotation')
        ]
    }

    # Create a figure for each DOF group
    for group_name, dof_list in dof_groups.items():
        # Determine subplot layout
        n_dofs = len(dof_list)
        if n_dofs <= 3:
            n_rows, n_cols = n_dofs, 1
        elif n_dofs == 4:
            n_rows, n_cols = 2, 2
        elif n_dofs <= 6:
            n_rows, n_cols = 3, 2
        else:
            n_rows, n_cols = 4, 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows), facecolor='white')
        fig.suptitle(f'Inverse Dynamics Forces: {group_name}', fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_dofs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Plot each DOF in the group
        for idx, (joint_idx, joint_name) in enumerate(dof_list):
            ax = axes[idx]
            ax.set_facecolor('white')
            
            # Get force data for this DOF
            forces = joint_forces_array[:, joint_idx]
            
            # Calculate statistics
            mean_val = np.mean(forces)
            max_val = np.max(forces)
            min_val = np.min(forces)
            std_val = np.std(forces)
            
            # Plot
            ax.plot(time_array, forces, linewidth=2, color='darkblue', label='qfrc_inverse')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axhline(y=mean_val, color='red', linestyle=':', linewidth=1, alpha=0.7, 
                       label=f'Mean: {mean_val:.1f}')
            
            # Labels and formatting
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Force/Torque (N·m or N)', fontsize=10)
            ax.set_title(f'{joint_name}\n[Min: {min_val:.1f}, Max: {max_val:.1f}, Std: {std_val:.1f}]', 
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
            
            print(f"  {joint_name:25s}: [{min_val:8.2f}, {max_val:8.2f}] N·m, Mean: {mean_val:8.2f}, Std: {std_val:8.2f}")
        
        # Hide unused subplots
        for idx in range(n_dofs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        print(f"\n✓ Figure created: {group_name}")

    # Create one additional figure showing all DOFs overlaid for comparison
    fig_all, ax_all = plt.subplots(figsize=(16, 8), facecolor='white')
    ax_all.set_facecolor('white')

    # Plot all DOFs with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(joint_names)))
    for idx in range(min(joint_forces_array.shape[1], len(joint_names))):
        forces = joint_forces_array[:, idx]
        ax_all.plot(time_array, forces, linewidth=1.5, alpha=0.7, 
                    color=colors[idx], label=joint_names[idx])

    ax_all.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax_all.set_xlabel('Time (s)', fontsize=12)
    ax_all.set_ylabel('Force/Torque (N·m or N)', fontsize=12)
    ax_all.set_title('All DOFs - Inverse Dynamics Forces Comparison', fontsize=14, fontweight='bold')
    ax_all.grid(True, alpha=0.3)
    ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    plt.tight_layout()

    print("\n✓ Figure created: All DOFs Comparison")

    # Print overall statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS FOR INVERSE DYNAMICS FORCES")
    print("="*70)
    all_forces = joint_forces_array.flatten()
    print(f"Overall min:  {np.min(all_forces):10.2f} N·m")
    print(f"Overall max:  {np.max(all_forces):10.2f} N·m")
    print(f"Overall mean: {np.mean(all_forces):10.2f} N·m")
    print(f"Overall std:  {np.std(all_forces):10.2f} N·m")

    # Find the DOF with maximum absolute force
    max_abs_forces = np.abs(joint_forces_array).max(axis=0)
    max_dof_idx = np.argmax(max_abs_forces)
    max_force = max_abs_forces[max_dof_idx]
    print(f"\nDOF with largest force: {joint_names[max_dof_idx]} (DOF {max_dof_idx}) = {max_force:.2f} N·m")

    print("\n✓ Displaying all inverse dynamics plots...")
    plt.show()

# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Path to muscle-free MuJoCo model
model_path = "Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml"

# Load MuJoCo model and convert to MJX
mj_model = mujoco.MjModel.from_xml_path(model_path)

# mj_model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
# mj_model.opt.disableflags += mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
# Print gravity settings
print("="*70)
print("MODEL PHYSICS SETTINGS")
print("="*70)
print(f"Gravity: {mj_model.opt.gravity}")
print(f"Total mass: {np.sum(mj_model.body_mass):.2f} kg")
print(f"Expected gravitational force: {np.sum(mj_model.body_mass) * 9.81:.2f} N")
print("="*70 + "\n")

# mj_model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.make_data(mjx_model)

print("MuJoCo model timestep:", mj_model.opt.timestep)

# Load patient data
data_path = "PatientData/Falisse_2017_subject_01/"
pos_data = pd.read_csv(data_path + "pos.csv")
vel_data = pd.read_csv(data_path + "vel.csv")
acc_data = pd.read_csv(data_path + "acc.csv")

# GRF and moment files have malformed headers (7 column names, 10 actual columns)
# Must read without header to get all 10 columns correctly
grf_data_raw = pd.read_csv(data_path + "grf.csv", header=None, skiprows=1)
moment_data_raw = pd.read_csv(data_path + "moment.csv", header=None, skiprows=1)
cop_data_raw = pd.read_csv(data_path + "cop.csv", header=None, skiprows=1)

# Store GRF and moment data in matrices
# Actual structure: [time, Fx_l, Fy_l, Fz_l, Fx_r, Fy_r, Fz_r, 0, 0, 0]
grf_matrix = grf_data_raw.values  # Get all rows and columns (208, 10)
moment_matrix = moment_data_raw.values  # Get all rows and columns (208, 10)
cop_matrix = cop_data_raw.values  # Get all rows and columns (208, 10)

cop_data = pd.DataFrame(cop_matrix, columns=[f'col_{i}' for i in range(cop_matrix.shape[1])])
cop_data.insert(0, 'time', grf_data_raw.iloc[:, 0].values)  # Ensure time is first column

# Resample all data to uniform 0.002s timestep
pos_data, vel_data, acc_data, moment_matrix, grf_matrix, cop_data = resample_dataframes_to_uniform_timestep(
    pos_data, vel_data, acc_data, moment_matrix, grf_matrix, cop_data, dt=0.01)

# Print first several lines of each for verification
print("Resampled pos_data (head):\n", pos_data.head())
print("Resampled vel_data (head):\n", vel_data.head())
print("Resampled acc_data (head):\n", acc_data.head())
if moment_matrix is not None:
    print("Resampled moment_matrix (head):\n", moment_matrix[:5])
if grf_matrix is not None:
    print("Resampled grf_matrix (head):\n", grf_matrix[:5])
print("Resampled cop_data (head):\n", cop_data.head())

# Mapping from patient data columns to MuJoCo qpos indices
# Patient data has 23 DOFs, MuJoCo model has 39 qpos (includes coupled joints)
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

# Initialize matrices and map all timesteps
num_timesteps = len(pos_data)
qpos_matrix = np.zeros((num_timesteps, mj_model.nq))
qvel_matrix = np.zeros((num_timesteps, mj_model.nv))
qacc_matrix = np.zeros((num_timesteps, mj_model.nv))

print("shape" , qpos_matrix.shape, qvel_matrix.shape, qacc_matrix.shape)

for i in range(num_timesteps):
    qpos_matrix[i, :] = map_patient_to_qpos(pos_data.iloc[i], mj_model.nq)
    qvel_matrix[i, :] = map_patient_to_qpos(vel_data.iloc[i], mj_model.nv)
    qacc_matrix[i, :] = map_patient_to_qpos(acc_data.iloc[i], mj_model.nv)

# Calculate coupled knee joint coordinates, velocities, and accelerations
qpos_matrix, qvel_matrix, qacc_matrix = calculate_knee_coupled_coords_all(qpos_matrix, qvel_matrix, qacc_matrix)


# Apply low-pass Butterworth filter to acceleration, velocity, and position data
qacc_matrix_filtered = butter_lowpass_filter(qacc_matrix, cutoff=3, fs=100.5, order=4)
qvel_matrix_filtered = butter_lowpass_filter(qvel_matrix, cutoff=3, fs=100.5, order=4)
qpos_matrix_filtered = butter_lowpass_filter(qpos_matrix, cutoff=3, fs=100.5, order=4)

qacc_matrix = qacc_matrix_filtered
qvel_matrix = qvel_matrix_filtered
qpos_matrix = qpos_matrix_filtered

print(f"Data processing complete: {qpos_matrix.shape[0]} frames mapped with coupled coordinates")

# Set initial frame to MJX model
mjx_data = mjx_data.replace(qpos=jnp.array(qpos_matrix[0, :]), qvel=jnp.array(qvel_matrix[0, :]))

@jax.jit
def compute_inverse_dynamics(model, data,qacc,qvel,qpos, external_forces):
    data = data.replace(qpos=qpos, qvel=qvel,qacc=qacc,xfrc_applied = external_forces)
    # mjx_model = mjx_model.tree_replace({
    # 'opt.enableflags': mjx.EnableBit.INVDISCRETE | mjx_model.opt.enableflags
    # })
    data = mjx.forward(model, data)
    # data = mjx.fwd_position(model, data)
    # data = mjx.fwd_velocity(model, data)
    # data = mjx.step(model,data)
    return mjx.inverse(model, data)

nb = mjx_model.nbody
external_forces = jnp.zeros((nb, 6, num_timesteps))

# Get body IDs for left and right calcaneus
calcn_l_id = mj_model.body('calcn_l').id  # Left calcaneus body ID (should be 12)
calcn_r_id = mj_model.body('calcn_r').id  # Right calcaneus body ID (should be 6)
talus_l_id = mj_model.body('talus_l').id    # Left talus body ID (should be 11)
talus_r_id = mj_model.body('talus_r').id    # Right talus body ID (should be 5)
pelvis_id = mj_model.body('pelvis').id    # Pelvis body ID (should be 1)

body_id_l= calcn_l_id
body_id_r= calcn_r_id

# IMPORTANT COORDINATE TRANSFORMATION:
# OpenSim: X=forward, Y=up, Z=right
# MuJoCo: X=forward, Y=right, Z=up
# Therefore: OpenSim_X -> MuJoCo_X, OpenSim_Y -> MuJoCo_Z, OpenSim_Z -> MuJoCo_Y

grf_left_opensim = grf_matrix[:, 4:7]  # Left foot GRF columns [Fx, Fy, Fz] in OpenSim
grf_right_opensim = grf_matrix[:, 1:4]  # Right foot GRF columns [Fx, Fy, Fz] in OpenSim

# Transform to MuJoCo coordinates: [X, Y, Z]_mujoco = [X, Z, Y]_opensim
neg=-1
grf_left = np.column_stack([neg*grf_left_opensim[:, 0],   # X stays X
                             -1*neg*grf_left_opensim[:, 2],   # Z becomes Y
                             neg*grf_left_opensim[:, 1]])  # Y becomes Z (vertical)

grf_right = np.column_stack([neg*grf_right_opensim[:, 0],  # X stays X  
                              -1*neg*grf_right_opensim[:, 2],  # Z becomes Y
                              neg*grf_right_opensim[:, 1]]) # Y becomes Z (vertical)

moment_left_opensim = moment_matrix[:, 4:7]  # Left foot moments
moment_right_opensim = moment_matrix[:, 1:4]  # Right foot moments

# Transform moments to MuJoCo coordinates as well
moment_left = np.column_stack([neg*moment_left_opensim[:, 0],   # X stays X
                                -1*neg*moment_left_opensim[:, 2],   # Z becomes Y
                                neg*moment_left_opensim[:, 1]])  # Y becomes Z

moment_right = np.column_stack([neg*moment_right_opensim[:, 0],  # X stays X
                                 -1*neg*moment_right_opensim[:, 2],  # Z becomes Y
                                 neg*moment_right_opensim[:, 1]]) # Y becomes Z

external_forces = external_forces.at[body_id_l, 0:3, :].set(grf_left.T)  
external_forces = external_forces.at[body_id_r, 0:3, :].set(grf_right.T) 

# external_forces = external_forces.at[talus_l_id, 0:3, :].set(grf_left.T)
# external_forces = external_forces.at[talus_r_id, 0:3, :].set(grf_right.T)
# Add moments/torques
external_forces = external_forces.at[body_id_l, 3:6, :].set(moment_left.T)  
external_forces = external_forces.at[body_id_r, 3:6, :].set(moment_right.T)  

# external_forces = external_forces.at[talus_l_id, 3:6, :].set(moment_left.T)
# external_forces = external_forces.at[talus_r_id, 3:6, :].set(moment_right.T)

# Initialize mjx_data (immutable structure)
current_mjx_data = mjx_data

# To store muscle forces over time
joint_forces_over_time = []
distance_calcn_r_all = []
distance_calcn_l_all = []
qfrc_from_grf_l_all = []
contact_force_all=[]

# Store calcaneus and COP positions for plotting
calcn_l_positions = []
cop_l_positions = []
calcn_r_positions = []
cop_r_positions = []
pelvis_positions = []
r_vec_r_all=[]
r_vec_l_all=[]
ankle_pos_l_all=[]
acceleration_all=[]
pos_all=[]
vel_all=[]

# Store calculated moments
moment_added_l_all = []
moment_added_r_all = []



contact_force_total_normal = []  # Store total normal contact force at each timestep

# Computing inverse dynamics
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

start = time.time()
for t in tqdm(range(num_timesteps), desc="Computing inverse dynamics"):
    try:
        # COP data from CSV: col1,3 = X,Y for right foot, col4,6 = X,Y for left foot (OpenSim frame)
        # MuJoCo frame: X=anterior-posterior, Y=medial-lateral, Z=vertical
        # COP is on ground plane, so Z=0
        # OpenSim X -> MuJoCo X, OpenSim Y -> MuJoCo Y, Z=0 (ground)
        cop_l = jnp.array([cop_matrix[t, 4], -1*cop_matrix[t, 6], 0.0])
        cop_r = jnp.array([cop_matrix[t, 1], -1*cop_matrix[t, 3], 0.0])

        body_id_l=calcn_l_id
        body_id_r=calcn_r_id

        if t == 0:
            distance_calcn_r_all.append(0.0)
            distance_calcn_l_all.append(0.0)
            pelvis_positions.append(np.array([0, 0, 0]))
            # Store positions for 3D plotting
            calcn_l_positions.append(np.array([0, 0, 0]))
            cop_l_positions.append(np.array([0, 0, 0]))
            calcn_r_positions.append(np.array([0, 0, 0]))
            cop_r_positions.append(np.array([0, 0, 0]))
            r_vec_r_all.append(np.array([0, 0, 0]))
            r_vec_l_all.append(np.array([0, 0, 0]))
            moment_added_l_all.append(np.array([0, 0, 0]))
            moment_added_r_all.append(np.array([0, 0, 0]))
        else:
            ankle_pos_l = current_mjx_data.xpos[calcn_l_id]  # Left ankle (talus) world position
            ankle_pos_r = current_mjx_data.xpos[calcn_r_id]  # Right ankle (talus) world position

            # Calculate distance between ankle (talus) and COP (full 3D distance)
            # COP is on ground plane (Z=0), ankle/talus is above ground
            # This distance is needed for moment arm calculation: M = r × F

            if sum(jnp.abs(cop_r)) > 0.0:
                # Full 3D vector from COP to ankle (talus)
                r_vec_r = cop_r - ankle_pos_r
                distance_ankle_r_to_cop = jnp.linalg.norm(r_vec_r)
                
            else:
                distance_ankle_r_to_cop = 0.0
                r_vec_r = jnp.array([0.0, 0.0, 0.0])

            if sum(jnp.abs(cop_l)) > 0.0:
                # Full 3D vector from COP to ankle (talus)
                r_vec_l = cop_l - ankle_pos_l
                distance_ankle_l_to_cop = jnp.linalg.norm(r_vec_l)
                
            else:
                distance_ankle_l_to_cop = 0.0
                r_vec_l = jnp.array([0.0, 0.0, 0.0])
            
            # Place limits on r_vel_l and r_vel_r to avoid extreme values
            # X lower bound
            if r_vec_l[0] < -.02:
                r_vec_l = r_vec_l.at[0].set(-0.02)
            if r_vec_r[0] < -.02:
                r_vec_r = r_vec_r.at[0].set(-0.02)

            # X upper bound
            if r_vec_l[0] > .13:
                r_vec_l = r_vec_l.at[0].set(0.13)
            if r_vec_r[0] > .13:
                r_vec_r = r_vec_r.at[0].set(0.13)

            # Y bounds
            if r_vec_l[1] > .04:
                r_vec_l = r_vec_l.at[1].set(0.04)
            if r_vec_r[1] < -.04:
                r_vec_r = r_vec_r.at[1].set(-0.04)
            if r_vec_l[1] < -.02:
                r_vec_l = r_vec_l.at[1].set(-0.02)
            if r_vec_r[1] > .02:
                r_vec_r = r_vec_r.at[1].set(0.02)

            # Z bounds
            if r_vec_l[2] < -.13:
                r_vec_l = r_vec_l.at[2].set(-0.13)
            if r_vec_r[2] < -.13:
                r_vec_r = r_vec_r.at[2].set(-0.13)

            distance_calcn_r_all.append(distance_ankle_r_to_cop)
            distance_calcn_l_all.append(distance_ankle_l_to_cop)
            pelvis_positions.append(np.array(pelvis_pos))
            # Store positions for 3D plotting
            calcn_l_positions.append(np.array(calcn_l_pos))
            cop_l_positions.append(np.array(cop_l))
            calcn_r_positions.append(np.array(calcn_r_pos))
            cop_r_positions.append(np.array(cop_r))
            r_vec_r_all.append(np.array(r_vec_r))
            r_vec_l_all.append(np.array(r_vec_l))
            
            ankle_pos_l_all.append(np.array(ankle_pos_l))

            moment_added_l= jnp.cross(r_vec_l,grf_left[t,:].T)
            moment_added_r= jnp.cross(r_vec_r,grf_right[t,:].T)
        
            
            # Store the calculated moments
            moment_added_l_all.append(np.array(moment_added_l))
            moment_added_r_all.append(np.array(moment_added_r))

            external_forces = external_forces.at[body_id_l, 3:6, t+1].set(moment_left[t,:]+moment_added_l)
            external_forces = external_forces.at[body_id_r, 3:6, t+1].set(moment_right[t,:]+moment_added_r)

        # Compute inverse dynamics for current timestep
        current_mjx_data = compute_inverse_dynamics(mjx_model, current_mjx_data,qacc_matrix[t, :],qvel_matrix[t, :],qpos_matrix[t, :], external_forces[:, :, t])

        # Get control forces (joint torques) from inverse dynamics
        ctrl_forces = current_mjx_data.qfrc_inverse

        contact_force_all.append(current_mjx_data.cfrc_ext)

        # Store acceleration, position, and velocity data from current timestep
        acceleration_all.append(current_mjx_data.qacc)
        pos_all.append(current_mjx_data.qpos)
        vel_all.append(current_mjx_data.qvel)

        # Store forces for analysis
        joint_forces_over_time.append(ctrl_forces)

        # Get body positions (shape: (3,) for [x, y, z])
        calcn_l_pos = current_mjx_data.xpos[calcn_l_id]  # Left calcaneus world position
        calcn_r_pos = current_mjx_data.xpos[calcn_r_id]  # Right calcaneus world position
        pelvis_pos = current_mjx_data.xpos[pelvis_id]    # Pelvis world position
        ankle_pos_l = current_mjx_data.xpos[talus_l_id]  # Left ankle (talus) world position
        ankle_pos_r = current_mjx_data.xpos[talus_r_id]  # Right ankle (talus) world position

        # Update ctrl in mjx_data for consistency
        # current_mjx_data = current_mjx_data.replace(ctrl=ctrl_forces)

        # --- Collect MuJoCo contact forces for this timestep ---
        # Create a MuJoCo mjData for this frame (if not already available)
        # (Assumes mj_data and mj_model are available and synchronized)


    except Exception as e:
        print(f"Error at timestep {t}: {e}")
        # Use zeros as fallback for all storage arrays
        joint_forces_over_time.append(jnp.zeros(mjx_model.nv))
        qfrc_from_grf_l_all.append(np.zeros(mjx_model.nv))
        distance_calcn_r_all.append(0.0)
        distance_calcn_l_all.append(0.0)
        pelvis_positions.append(np.zeros(3))
        calcn_l_positions.append(np.zeros(3))
        cop_l_positions.append(np.zeros(3))
        calcn_r_positions.append(np.zeros(3))
        cop_r_positions.append(np.zeros(3))
        r_vec_r_all.append(np.zeros(3))
        r_vec_l_all.append(np.zeros(3))
        moment_added_l_all.append(np.zeros(3))
        moment_added_r_all.append(np.zeros(3))

# ============================================================================
# VISUALIZATION: Plot all results
# ============================================================================

# # Plot moment arm vectors
# plot_moment_arm_vectors(pos_data['time'].values, r_vec_r_all, r_vec_l_all)

# Convert moment lists to numpy arrays for analysis
moment_added_l_matrix = np.array(moment_added_l_all)
moment_added_r_matrix = np.array(moment_added_r_all)

# # Plot calculated moments
# plot_calculated_moments(pos_data['time'].values, moment_added_l_matrix, moment_added_r_matrix)

# # Plot moment magnitudes
# plot_moment_magnitudes(pos_data['time'].values, moment_added_l_matrix, moment_added_r_matrix)

# # Plot ankle-COP distances
# plot_ankle_cop_distances(pos_data['time'].values, distance_calcn_r_all, distance_calcn_l_all)

# # Plot 3D trajectories
# plot_3d_trajectories(calcn_l_positions, cop_l_positions, calcn_r_positions, 
#                      cop_r_positions, ankle_pos_l_all)

# # Plot pelvis trajectory
# plot_pelvis_trajectory(pelvis_positions)

# ============================================================================
# VIEWER: Launch MuJoCo viewer
# ============================================================================

# Launch viewer
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)

import mujoco
import mujoco.viewer
import numpy as np
import time

# Get IDs for the GRF arrow sites and geom
tail_id_L = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "grf_left_tail")
tip_id_L = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "grf_left_tip")
arrow_id_L = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "grf_left_arrow")
# Get IDs for right foot GRF arrow
tail_id_R = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "grf_right_tail")
tip_id_R = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "grf_right_tip")
arrow_id_R = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "grf_right_arrow")

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    t = 0
    scale = 0.001  # Adjust as needed for force arrow length
    while viewer.is_running():
        mj_data.qpos[:] = qpos_matrix[t, :]
        mj_data.qvel[:] = qvel_matrix[t, :]
        mujoco.mj_forward(mj_model, mj_data)

        # Get COP and GRF (in MuJoCo coordinates)
        cop_l = np.array([cop_matrix[t, 4], -1*cop_matrix[t, 6], 0.0])
        grf_l = grf_left[t, :]  # shape (3,)
        arrow_end = cop_l + grf_l * scale

        # Update sites: tail is at COP, tip is at arrow_end
        mj_data.site_xpos[tail_id_L] = cop_l
        mj_data.site_xpos[tip_id_L] = arrow_end

        # Get COP and GRF (in MuJoCo coordinates)
        cop_r = np.array([cop_matrix[t, 1], -1*cop_matrix[t, 3], 0.0])
        grf_r = grf_right[t, :]  # shape (3,)
        arrow_end = cop_r + grf_r * scale

        # Update sites: tail is at COP, tip is at arrow_end
        mj_data.site_xpos[tail_id_R] = cop_r
        mj_data.site_xpos[tip_id_R] = arrow_end

        # Update capsule geom to connect tail and tip
        # mujoco.mj_set_geom_fromto(mj_model, mj_data, arrow_id_L, cop_l, arrow_end)

        # Optional: show contact forces as well
        # with viewer.lock():
        #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

        viewer.sync()
        time.sleep(0.1)
        t = (t + 1) % num_timesteps

# ============================================================================
# PLOTTING: Generate visualizations of inverse dynamics results
# ============================================================================

# Convert forces to numpy array
joint_forces_array = np.array([np.array(f) for f in joint_forces_over_time])
time_array = pos_data['time'].values

# Define joint names (matching qpos_mapping)
joint_names = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',           # 0-2
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', # 3-5
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', # 6-8
    'walker_knee_r_t1', 'walker_knee_r_t2',          # 9-10
    'knee_angle_r',                                   # 11
    'walker_knee_r_r2', 'walker_knee_r_r3',          # 12-13
    'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', # 14-16
    'patella_r_t1', 'patella_r_t2', 'patella_r_t3', 'patella_r_r1', # 17-20
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', # 21-23
    'walker_knee_l_t1', 'walker_knee_l_t2',          # 24-25
    'knee_angle_l',                                   # 26
    'walker_knee_l_r2', 'walker_knee_l_r3',          # 27-28
    'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', # 29-31
    'patella_l_t1', 'patella_l_t2', 'patella_l_t3', 'patella_l_r1', # 32-35
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation' # 36-38
]

# Plot all inverse dynamics results
plot_inverse_dynamics_forces(joint_forces_array, time_array, joint_names)

print("="*70)
print("All plots generated successfully!")
print("="*70)

# =========================================================================
# COMPARISON PLOT: acceleration_all vs qacc_matrix (all DOFs, 6 per slide)
# =========================================================================
import matplotlib.pyplot as plt

acc_all_np = np.array(acceleration_all)  # shape: (num_timesteps, n_dof)
qacc_np = np.array(qacc_matrix)          # shape: (num_timesteps, n_dof)
time_array = pos_data['time'].values if 'pos_data' in locals() else np.arange(acc_all_np.shape[0])

# Use joint_names if available, else generic labels
try:
    dof_names = joint_names
except NameError:
    dof_names = [f'DOF {i}' for i in range(acc_all_np.shape[1])]

n_dof = acc_all_np.shape[1]
n_per_fig = 6
n_figs = (n_dof + n_per_fig - 1) // n_per_fig

for fig_idx in range(n_figs):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Comparison of acceleration_all vs qacc_matrix (DOFs {fig_idx*n_per_fig} to {min((fig_idx+1)*n_per_fig-1, n_dof-1)})', fontsize=16, fontweight='bold')
    for i in range(n_per_fig):
        dof_idx = fig_idx * n_per_fig + i
        if dof_idx >= n_dof:
            axes.flat[i].axis('off')
            continue
        axes.flat[i].plot(time_array, acc_all_np[:, dof_idx], label='acceleration_all', color='blue')
        axes.flat[i].plot(time_array, qacc_np[:, dof_idx], label='qacc_matrix', color='red', linestyle='--')
        axes.flat[i].plot(time_array, acc_all_np[:, dof_idx] - qacc_np[:, dof_idx], label='difference', color='green', linestyle=':')
        axes.flat[i].set_ylabel(f'{dof_names[dof_idx]}')
        axes.flat[i].legend()
        axes.flat[i].grid(True, alpha=0.3)
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# =========================================================================
# COMPARISON PLOT: pos_all vs qpos_matrix (all DOFs, 6 per slide)
# =========================================================================
pos_all_np = np.array(pos_all)  # shape: (num_timesteps, n_dof)
qpos_np = np.array(qpos_matrix) # shape: (num_timesteps, n_dof)
time_array_pos = pos_data['time'].values if 'pos_data' in locals() else np.arange(pos_all_np.shape[0])

for fig_idx in range(n_figs):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Comparison of pos_all vs qpos_matrix (DOFs {fig_idx*n_per_fig} to {min((fig_idx+1)*n_per_fig-1, n_dof-1)})', fontsize=16, fontweight='bold')
    for i in range(n_per_fig):
        dof_idx = fig_idx * n_per_fig + i
        if dof_idx >= n_dof:
            axes.flat[i].axis('off')
            continue
        axes.flat[i].plot(time_array_pos, pos_all_np[:, dof_idx], label='pos_all', color='blue')
        axes.flat[i].plot(time_array_pos, qpos_np[:, dof_idx], label='qpos_matrix', color='red', linestyle='--')
        axes.flat[i].plot(time_array_pos, pos_all_np[:, dof_idx] - qpos_np[:, dof_idx], label='difference', color='green', linestyle=':')
        axes.flat[i].set_ylabel(f'{dof_names[dof_idx]}')
        axes.flat[i].legend()
        axes.flat[i].grid(True, alpha=0.3)
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# =========================================================================
# COMPARISON PLOT: vel_all vs qvel_matrix (all DOFs, 6 per slide)
# =========================================================================
vel_all_np = np.array(vel_all)  # shape: (num_timesteps, n_dof)
qvel_np = np.array(qvel_matrix) # shape: (num_timesteps, n_dof)
time_array_vel = pos_data['time'].values if 'pos_data' in locals() else np.arange(vel_all_np.shape[0])

for fig_idx in range(n_figs):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Comparison of vel_all vs qvel_matrix (DOFs {fig_idx*n_per_fig} to {min((fig_idx+1)*n_per_fig-1, n_dof-1)})', fontsize=16, fontweight='bold')
    for i in range(n_per_fig):
        dof_idx = fig_idx * n_per_fig + i
        if dof_idx >= n_dof:
            axes.flat[i].axis('off')
            continue
        axes.flat[i].plot(time_array_vel, vel_all_np[:, dof_idx], label='vel_all', color='blue')
        axes.flat[i].plot(time_array_vel, qvel_np[:, dof_idx], label='qvel_matrix', color='red', linestyle='--')
        axes.flat[i].plot(time_array_vel, vel_all_np[:, dof_idx] - qvel_np[:, dof_idx], label='difference', color='green', linestyle=':')
        axes.flat[i].set_ylabel(f'{dof_names[dof_idx]}')
        axes.flat[i].legend()
        axes.flat[i].grid(True, alpha=0.3)
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# =========================================================================
# CONTACT FORCE VISUALIZATION: Plot contact forces for all bodies (5 per figure)
# =========================================================================
if 'contact_force_all' in locals() or 'contact_force_all' in globals():
    contact_force_np = np.array(contact_force_all)  # [num_time_steps, nbodies, 6]
    num_timesteps, nbodies, _ = contact_force_np.shape
    time_array = pos_data['time'].values if 'pos_data' in locals() else np.arange(num_timesteps)
    # Get body names from MuJoCo model
    body_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(nbodies)]
    n_per_fig = 5
    n_figs = (nbodies + n_per_fig - 1) // n_per_fig
    for fig_idx in range(n_figs):
        fig, axes = plt.subplots(n_per_fig, 1, figsize=(12, 2.5*n_per_fig), sharex=True)
        if n_per_fig == 1:
            axes = [axes]
        fig.suptitle(f'Contact Forces (bodies {fig_idx*n_per_fig} to {min((fig_idx+1)*n_per_fig-1, nbodies-1)})', fontsize=16, fontweight='bold')
        for i in range(n_per_fig):
            body_idx = fig_idx * n_per_fig + i
            if body_idx >= nbodies:
                axes[i].axis('off')
                continue
            # Plot force norm (or pick a component, e.g., Fz: contact_force_np[:, body_idx, 2])
            force_norm = np.linalg.norm(contact_force_np[:, body_idx, 0:3], axis=1)
            axes[i].plot(time_array, force_norm, label='|F| (N)', color='blue')
            axes[i].plot(time_array, contact_force_np[:, body_idx, 2], label='Fz (vertical)', color='orange', linestyle='--')
            axes[i].set_ylabel('Force (N)')
            axes[i].set_title(f'Body: {body_names[body_idx]}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()