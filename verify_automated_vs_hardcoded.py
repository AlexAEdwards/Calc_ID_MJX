"""
Verification script to compare automated vs hardcoded coupled coordinate calculations.

This script demonstrates that calculate_coupled_coordinates_automated() produces
identical results to the original calculate_knee_coupled_coords_all() function.
"""

import numpy as np
from MJX_IDworks import (
    calculate_knee_coupled_coords_all,
    calculate_coupled_coordinates_automated
)
import mujoco

# Configuration
xml_path = "Results/ModelNoMus/scaled_model_no_muscles_cvt2.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
num_timesteps = 100

print(f"{'='*80}")
print(f"Verification: Automated vs Hardcoded Coupled Coordinate Calculation")
print(f"{'='*80}\n")

# Create test data
qpos_matrix = np.zeros((num_timesteps, model.nq))
qvel_matrix = np.zeros((num_timesteps, model.nv))
qacc_matrix = np.zeros((num_timesteps, model.nv))

# Generate sinusoidal knee motion for testing
time = np.linspace(0, 2, num_timesteps)

# Right knee (index 11) - full flexion/extension cycle
qpos_matrix[:, 11] = 0.5 + 0.5 * np.sin(2 * np.pi * time)
qvel_matrix[:, 11] = 0.5 * 2 * np.pi * np.cos(2 * np.pi * time)
qacc_matrix[:, 11] = -0.5 * (2 * np.pi)**2 * np.sin(2 * np.pi * time)

# Left knee (index 26) - phase shifted
qpos_matrix[:, 26] = 0.5 + 0.5 * np.sin(2 * np.pi * time + np.pi/3)
qvel_matrix[:, 26] = 0.5 * 2 * np.pi * np.cos(2 * np.pi * time + np.pi/3)
qacc_matrix[:, 26] = -0.5 * (2 * np.pi)**2 * np.sin(2 * np.pi * time + np.pi/3)

print("Test Data:")
print(f"  Timesteps: {num_timesteps}")
print(f"  Right knee angle range: [{qpos_matrix[:, 11].min():.3f}, {qpos_matrix[:, 11].max():.3f}] rad")
print(f"  Left knee angle range: [{qpos_matrix[:, 26].min():.3f}, {qpos_matrix[:, 26].max():.3f}] rad")

# Method 1: Hardcoded version
print(f"\n{'='*80}")
print("Method 1: Hardcoded (calculate_knee_coupled_coords_all)")
print(f"{'='*80}\n")

qpos_hardcoded, qvel_hardcoded, qacc_hardcoded = calculate_knee_coupled_coords_all(
    qpos_matrix.copy(), 
    qvel_matrix.copy(), 
    qacc_matrix.copy()
)

print("✓ Hardcoded calculation complete")

# Method 2: Automated version
print(f"\n{'='*80}")
print("Method 2: Automated (calculate_coupled_coordinates_automated)")
print(f"{'='*80}\n")

qpos_automated, qvel_automated, qacc_automated = calculate_coupled_coordinates_automated(
    qpos_matrix.copy(), 
    qvel_matrix.copy(), 
    qacc_matrix.copy(),
    xml_path
)

# Compare results
print(f"\n{'='*80}")
print("Comparison Results")
print(f"{'='*80}\n")

# Check if arrays are identical
qpos_match = np.allclose(qpos_hardcoded, qpos_automated, rtol=1e-10, atol=1e-12)
qvel_match = np.allclose(qvel_hardcoded, qvel_automated, rtol=1e-10, atol=1e-12)
qacc_match = np.allclose(qacc_hardcoded, qacc_automated, rtol=1e-10, atol=1e-12)

print(f"Position (qpos) match: {'✓ YES' if qpos_match else '✗ NO'}")
print(f"Velocity (qvel) match: {'✓ YES' if qvel_match else '✗ NO'}")
print(f"Acceleration (qacc) match: {'✓ YES' if qacc_match else '✗ NO'}")

if not qpos_match:
    qpos_diff = np.abs(qpos_hardcoded - qpos_automated)
    print(f"\n  qpos max difference: {qpos_diff.max():.2e}")
    print(f"  qpos mean difference: {qpos_diff.mean():.2e}")
    
if not qvel_match:
    qvel_diff = np.abs(qvel_hardcoded - qvel_automated)
    print(f"\n  qvel max difference: {qvel_diff.max():.2e}")
    print(f"  qvel mean difference: {qvel_diff.mean():.2e}")
    
if not qacc_match:
    qacc_diff = np.abs(qacc_hardcoded - qacc_automated)
    print(f"\n  qacc max difference: {qacc_diff.max():.2e}")
    print(f"  qacc mean difference: {qacc_diff.mean():.2e}")

# Detailed comparison for specific coupled coordinates
print(f"\n{'='*80}")
print("Detailed Comparison (Sample Coupled Coordinates at t=50)")
print(f"{'='*80}\n")

t = 50  # Middle timestep

# Right knee coupled coordinates (adjust indices based on your model)
coupled_coords = {
    9: "walker_knee_r_translation1",
    10: "walker_knee_r_translation2",
    12: "walker_knee_r_rotation2",
    13: "walker_knee_r_rotation3",
}

print("Right Knee Coupled Coordinates:")
print(f"  Driver: knee_angle_r = {qpos_hardcoded[t, 11]:.6f} rad\n")

for idx, name in coupled_coords.items():
    if idx < model.nq:
        pos_hard = qpos_hardcoded[t, idx]
        pos_auto = qpos_automated[t, idx]
        pos_diff = abs(pos_hard - pos_auto)
        
        print(f"  {name}:")
        print(f"    Hardcoded: {pos_hard:.8f}")
        print(f"    Automated: {pos_auto:.8f}")
        print(f"    Difference: {pos_diff:.2e}")
        print()

# Final verdict
print(f"{'='*80}")
if qpos_match and qvel_match and qacc_match:
    print("✓✓✓ VERIFICATION SUCCESSFUL ✓✓✓")
    print("\nBoth methods produce IDENTICAL results!")
    print("The automated function correctly extracts and applies the polynomial constraints.")
else:
    print("⚠️  VERIFICATION FAILED")
    print("\nThe methods produce DIFFERENT results.")
    print("Please check the XML parsing or polynomial application logic.")
print(f"{'='*80}\n")

# Summary
print("Summary:")
print(f"  • Automated method correctly parses XML: {'✓' if qpos_match else '✗'}")
print(f"  • Polynomial coefficients match: {'✓' if qpos_match else '✗'}")
print(f"  • Velocity derivatives correct: {'✓' if qvel_match else '✗'}")
print(f"  • Acceleration derivatives correct: {'✓' if qacc_match else '✗'}")
print(f"\n  → Recommendation: Use {'automated' if all([qpos_match, qvel_match, qacc_match]) else 'hardcoded (investigate differences)'} method")
