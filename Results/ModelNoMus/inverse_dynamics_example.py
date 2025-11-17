"""
Example: Computing Inverse Dynamics for the muscle-free model

This script demonstrates how to:
1. Load the muscle-free model
2. Set a kinematic trajectory
3. Compute inverse dynamics
4. Extract and analyze joint torques
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Load the muscle-free model
model_path = '/home/mobl/Documents/Classwork/BioSimClass/ConvertOpenSimToMJX/Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml'

print("=" * 80)
print("INVERSE DYNAMICS EXAMPLE")
print("=" * 80)
print()

print("Loading model...")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
print(f"✓ Model loaded: {model.nq} DOFs, {model.nbody} bodies")
print()

# ============================================================================
# Example 1: Static Standing - Gravity Compensation
# ============================================================================
print("=" * 80)
print("EXAMPLE 1: Static Standing (Gravity Compensation)")
print("=" * 80)
print()

# Reset to default standing pose
mujoco.mj_resetDataKeyframe(model, data, 0)

# Zero velocities and accelerations (static)
data.qvel[:] = 0
data.qacc[:] = 0

# Compute inverse dynamics
mujoco.mj_inverse(model, data)

# Extract torques
torques_standing = data.qfrc_inverse.copy()

print("Joint torques required to maintain standing posture:")
print(f"  Right hip flexion:  {torques_standing[6]:8.2f} Nm")
print(f"  Right knee:         {torques_standing[11]:8.2f} Nm")
print(f"  Right ankle:        {torques_standing[14]:8.2f} Nm")
print(f"  Left hip flexion:   {torques_standing[21]:8.2f} Nm")
print(f"  Left knee:          {torques_standing[26]:8.2f} Nm")
print(f"  Left ankle:         {torques_standing[29]:8.2f} Nm")
print(f"  Lumbar extension:   {torques_standing[36]:8.2f} Nm")
print()

# ============================================================================
# Example 2: Dynamic Motion - Knee Flexion
# ============================================================================
print("=" * 80)
print("EXAMPLE 2: Dynamic Knee Flexion")
print("=" * 80)
print()

# Simulate knee flexion from 0 to 90 degrees over 1 second
n_steps = 100
dt = 0.01  # 10ms timesteps
time = np.arange(n_steps) * dt

# Trajectory: smooth knee flexion
knee_angle_traj = 1.57 * (1 - np.cos(np.pi * time / time[-1])) / 2  # 0 to 90 degrees

# Storage for results
hip_torques = []
knee_torques = []
ankle_torques = []

print("Computing inverse dynamics for knee flexion trajectory...")

for i in range(n_steps):
    # Reset to standing
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set knee angle
    data.qpos[11] = knee_angle_traj[i]  # Right knee
    
    # Compute velocity (finite difference)
    if i > 0:
        data.qvel[11] = (knee_angle_traj[i] - knee_angle_traj[i-1]) / dt
    
    # Compute acceleration (finite difference)
    if i > 1:
        v_prev = (knee_angle_traj[i-1] - knee_angle_traj[i-2]) / dt
        v_curr = data.qvel[11]
        data.qacc[11] = (v_curr - v_prev) / dt
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Inverse dynamics
    mujoco.mj_inverse(model, data)
    
    # Store torques
    hip_torques.append(data.qfrc_inverse[6])
    knee_torques.append(data.qfrc_inverse[11])
    ankle_torques.append(data.qfrc_inverse[14])

hip_torques = np.array(hip_torques)
knee_torques = np.array(knee_torques)
ankle_torques = np.array(ankle_torques)

print(f"✓ Computed {n_steps} timesteps")
print()
print("Peak torques during motion:")
print(f"  Hip flexion:   {np.max(np.abs(hip_torques)):8.2f} Nm")
print(f"  Knee:          {np.max(np.abs(knee_torques)):8.2f} Nm")
print(f"  Ankle:         {np.max(np.abs(ankle_torques)):8.2f} Nm")
print()

# ============================================================================
# Example 3: Walking Gait Cycle (Simplified)
# ============================================================================
print("=" * 80)
print("EXAMPLE 3: Simplified Gait Cycle")
print("=" * 80)
print()

# Simple gait pattern: alternating single leg support
n_gait_steps = 200
gait_time = np.arange(n_gait_steps) * 0.01

# Hip angles (sinusoidal pattern)
hip_r_traj = 0.5 * np.sin(2 * np.pi * gait_time / 1.0)  # 1 second cycle
hip_l_traj = 0.5 * np.sin(2 * np.pi * gait_time / 1.0 + np.pi)  # Opposite phase

# Knee angles (non-negative, phase with hip)
knee_r_traj = 0.5 * (1 - np.cos(2 * np.pi * gait_time / 1.0))
knee_l_traj = 0.5 * (1 - np.cos(2 * np.pi * gait_time / 1.0 + np.pi))

# Storage
hip_r_torque = []
knee_r_torque = []
hip_l_torque = []
knee_l_torque = []

print("Computing inverse dynamics for gait cycle...")

for i in range(n_gait_steps):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set joint angles
    data.qpos[6] = hip_r_traj[i]   # Right hip
    data.qpos[11] = knee_r_traj[i]  # Right knee
    data.qpos[21] = hip_l_traj[i]   # Left hip
    data.qpos[26] = knee_l_traj[i]  # Left knee
    
    # Estimate velocities and accelerations (simplified)
    if i > 0:
        data.qvel[6] = (hip_r_traj[i] - hip_r_traj[i-1]) / 0.01
        data.qvel[11] = (knee_r_traj[i] - knee_r_traj[i-1]) / 0.01
        data.qvel[21] = (hip_l_traj[i] - hip_l_traj[i-1]) / 0.01
        data.qvel[26] = (knee_l_traj[i] - knee_l_traj[i-1]) / 0.01
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Inverse dynamics
    mujoco.mj_inverse(model, data)
    
    # Store torques
    hip_r_torque.append(data.qfrc_inverse[6])
    knee_r_torque.append(data.qfrc_inverse[11])
    hip_l_torque.append(data.qfrc_inverse[21])
    knee_l_torque.append(data.qfrc_inverse[26])

hip_r_torque = np.array(hip_r_torque)
knee_r_torque = np.array(knee_r_torque)
hip_l_torque = np.array(hip_l_torque)
knee_l_torque = np.array(knee_l_torque)

print(f"✓ Computed {n_gait_steps} timesteps")
print()
print("Gait cycle torque statistics:")
print(f"  Right hip:  Mean={np.mean(np.abs(hip_r_torque)):6.2f} Nm, Peak={np.max(np.abs(hip_r_torque)):6.2f} Nm")
print(f"  Right knee: Mean={np.mean(np.abs(knee_r_torque)):6.2f} Nm, Peak={np.max(np.abs(knee_r_torque)):6.2f} Nm")
print(f"  Left hip:   Mean={np.mean(np.abs(hip_l_torque)):6.2f} Nm, Peak={np.max(np.abs(hip_l_torque)):6.2f} Nm")
print(f"  Left knee:  Mean={np.mean(np.abs(knee_l_torque)):6.2f} Nm, Peak={np.max(np.abs(knee_l_torque)):6.2f} Nm")
print()

# ============================================================================
# Visualization
# ============================================================================
print("=" * 80)
print("GENERATING PLOTS")
print("=" * 80)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Inverse Dynamics Examples', fontsize=16, fontweight='bold')

# Example 2: Knee flexion
axes[0, 0].plot(time, np.rad2deg(knee_angle_traj), 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Knee Angle (degrees)')
axes[0, 0].set_title('Example 2: Knee Flexion Trajectory')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time, knee_torques, 'r-', linewidth=2, label='Knee')
axes[0, 1].plot(time, hip_torques, 'g-', linewidth=2, label='Hip')
axes[0, 1].plot(time, ankle_torques, 'b-', linewidth=2, label='Ankle')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Joint Torque (Nm)')
axes[0, 1].set_title('Example 2: Joint Torques During Knee Flexion')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Example 3: Gait cycle
axes[1, 0].plot(gait_time, np.rad2deg(hip_r_traj), 'r-', linewidth=2, label='Right Hip')
axes[1, 0].plot(gait_time, np.rad2deg(hip_l_traj), 'b-', linewidth=2, label='Left Hip')
axes[1, 0].plot(gait_time, np.rad2deg(knee_r_traj), 'r--', linewidth=2, label='Right Knee')
axes[1, 0].plot(gait_time, np.rad2deg(knee_l_traj), 'b--', linewidth=2, label='Left Knee')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Joint Angle (degrees)')
axes[1, 0].set_title('Example 3: Gait Cycle Joint Angles')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(gait_time, hip_r_torque, 'r-', linewidth=2, label='Right Hip')
axes[1, 1].plot(gait_time, hip_l_torque, 'b-', linewidth=2, label='Left Hip')
axes[1, 1].plot(gait_time, knee_r_torque, 'r--', linewidth=2, label='Right Knee')
axes[1, 1].plot(gait_time, knee_l_torque, 'b--', linewidth=2, label='Left Knee')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Joint Torque (Nm)')
axes[1, 1].set_title('Example 3: Gait Cycle Joint Torques')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = '/home/mobl/Documents/Classwork/BioSimClass/ConvertOpenSimToMJX/Results/ModelNoMus/inverse_dynamics_examples.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Plots saved to: {output_path}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("✓ Model successfully loaded and tested")
print("✓ Inverse dynamics computed for 3 scenarios:")
print("    1. Static standing (gravity compensation)")
print("    2. Dynamic knee flexion")
print("    3. Simplified gait cycle")
print()
print("Key functions used:")
print("  • mj_forward(model, data)  - Forward kinematics")
print("  • mj_inverse(model, data)  - Inverse dynamics")
print("  • data.qfrc_inverse        - Joint torques output")
print()
print("The model is ready for your inverse dynamics research!")
print("=" * 80)
