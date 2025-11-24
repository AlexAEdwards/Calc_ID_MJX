"""
Example demonstrating the use of calculate_coupled_coordinates_automated function.

This script shows how to automatically extract polynomial constraints from an XML
model and populate coupled coordinates without hardcoding polynomial coefficients.
"""

import numpy as np
from MJX_IDworks import calculate_coupled_coordinates_automated
import mujoco

# Path to your XML model
xml_path = "Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml"

# Load model to get dimensions
model = mujoco.MjModel.from_xml_path(xml_path)
nq = model.nq  # Number of position coordinates
nv = model.nv  # Number of velocity coordinates
num_timesteps = 100  # Example: 100 timesteps

print(f"Model dimensions:")
print(f"  nq (positions): {nq}")
print(f"  nv (velocities/accelerations): {nv}")

# Create example matrices (normally these would come from your motion capture data)
# For this example, we'll create dummy data with some variation in the knee angles
qpos_matrix = np.zeros((num_timesteps, nq))
qvel_matrix = np.zeros((num_timesteps, nv))
qacc_matrix = np.zeros((num_timesteps, nv))

# Set up some example main joint angles (e.g., knee angles)
# Assuming knee_angle_r is at qpos index 11 and knee_angle_l is at index 26
# (adjust these indices based on your actual model)
time = np.linspace(0, 1, num_timesteps)
qpos_matrix[:, 11] = 0.5 * np.sin(2 * np.pi * time)  # Right knee: -0.5 to 0.5 radians
qpos_matrix[:, 26] = 0.5 * np.sin(2 * np.pi * time + np.pi/4)  # Left knee: phase shifted

# Set corresponding velocities (derivative of position)
qvel_matrix[:, 11] = 0.5 * 2 * np.pi * np.cos(2 * np.pi * time)
qvel_matrix[:, 26] = 0.5 * 2 * np.pi * np.cos(2 * np.pi * time + np.pi/4)

# Set corresponding accelerations (derivative of velocity)
qacc_matrix[:, 11] = -0.5 * (2 * np.pi)**2 * np.sin(2 * np.pi * time)
qacc_matrix[:, 26] = -0.5 * (2 * np.pi)**2 * np.sin(2 * np.pi * time + np.pi/4)

print(f"\nBefore automated calculation:")
print(f"  Main knee angles populated: Yes")
print(f"  Coupled coordinates: Not yet calculated")

# Call the automated function
qpos_updated, qvel_updated, qacc_updated = calculate_coupled_coordinates_automated(
    qpos_matrix, 
    qvel_matrix, 
    qacc_matrix, 
    xml_path
)

print(f"\nAfter automated calculation:")
print(f"  All coupled coordinates populated: Yes")
print(f"  qpos_matrix shape: {qpos_updated.shape}")
print(f"  qvel_matrix shape: {qvel_updated.shape}")
print(f"  qacc_matrix shape: {qacc_updated.shape}")

# Compare with hardcoded version (if you want to verify)
print(f"\n{'='*70}")
print(f"Verification (optional)")
print(f"{'='*70}")
print(f"You can compare the results with calculate_knee_coupled_coords_all()")
print(f"to verify that the automated extraction produces the same results.")

# Example: Check some coupled coordinate values
print(f"\nExample coupled coordinate values at timestep 50:")
print(f"  Right knee angle (index 11): {qpos_updated[50, 11]:.6f} rad")

# These indices correspond to some coupled coordinates (adjust based on your model)
# For example, walker_knee_r_translation1 might be at index 9
if nq > 9:
    print(f"  Walker knee R translation1 (index 9): {qpos_updated[50, 9]:.6f}")
if nq > 10:
    print(f"  Walker knee R translation2 (index 10): {qpos_updated[50, 10]:.6f}")

print(f"\n✓ Example complete!")
print(f"\nNow you can use these populated matrices for inverse dynamics:")
print(f"  tau = mjx.inverse(mjx_model, mjx_data)")
