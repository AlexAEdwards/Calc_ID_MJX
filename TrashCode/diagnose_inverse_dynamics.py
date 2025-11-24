"""
Diagnostic script to identify why inverse dynamics produces unrealistically high values.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data_path = "PatientData/Falisse_2017_subject_01/"
pos_data = pd.read_csv(data_path + "pos.csv")
vel_data = pd.read_csv(data_path + "vel.csv")
acc_data = pd.read_csv(data_path + "acc.csv")
grf_data_raw = pd.read_csv(data_path + "grf.csv", header=None, skiprows=1)
moment_data_raw = pd.read_csv(data_path + "moment.csv", header=None, skiprows=1)

print("="*70)
print("DIAGNOSTIC: Checking for unrealistic values in input data")
print("="*70)

# Check accelerations
print("\n1. ACCELERATION DATA:")
acc_matrix = acc_data.drop('time', axis=1).values
print(f"   Shape: {acc_matrix.shape}")
print(f"   Min: {acc_matrix.min():.2f} rad/s²")
print(f"   Max: {acc_matrix.max():.2f} rad/s²")
print(f"   Mean: {acc_matrix.mean():.2f} rad/s²")
print(f"   Std: {acc_matrix.std():.2f} rad/s²")
if np.abs(acc_matrix).max() > 100:
    print("   ⚠️  WARNING: Very high accelerations detected!")
    print(f"   Highest values at:")
    for col_idx in range(acc_matrix.shape[1]):
        max_val = np.abs(acc_matrix[:, col_idx]).max()
        if max_val > 100:
            col_name = acc_data.columns[col_idx+1]
            print(f"      {col_name}: {max_val:.2f} rad/s²")

# Check GRF
print("\n2. GROUND REACTION FORCES:")
grf_matrix = grf_data_raw.values
grf_left = grf_matrix[:, 4:7]
grf_right = grf_matrix[:, 1:4]
print(f"   Left foot GRF range: [{grf_left.min():.1f}, {grf_left.max():.1f}] N")
print(f"   Right foot GRF range: [{grf_right.min():.1f}, {grf_right.max():.1f}] N")
print(f"   Total max GRF: {max(np.abs(grf_left).max(), np.abs(grf_right).max()):.1f} N")

# Check moments from file
print("\n3. MOMENTS FROM FILE:")
moment_matrix = moment_data_raw.values
moment_left = moment_matrix[:, 4:7]
moment_right = moment_matrix[:, 1:4]
print(f"   Left foot moment range: [{moment_left.min():.1f}, {moment_left.max():.1f}] N⋅m")
print(f"   Right foot moment range: [{moment_right.min():.1f}, {moment_right.max():.1f}] N⋅m")
print(f"   Total max moment: {max(np.abs(moment_left).max(), np.abs(moment_right).max()):.1f} N⋅m")

# Check for NaN or Inf
print("\n4. DATA QUALITY CHECKS:")
print(f"   Positions: {np.isnan(pos_data.values).sum()} NaN, {np.isinf(pos_data.values).sum()} Inf")
print(f"   Velocities: {np.isnan(vel_data.values).sum()} NaN, {np.isinf(vel_data.values).sum()} Inf")
print(f"   Accelerations: {np.isnan(acc_matrix).sum()} NaN, {np.isinf(acc_matrix).sum()} Inf")
print(f"   GRF: {np.isnan(grf_matrix).sum()} NaN, {np.isinf(grf_matrix).sum()} Inf")
print(f"   Moments: {np.isnan(moment_matrix).sum()} NaN, {np.isinf(moment_matrix).sum()} Inf")

# Plot accelerations to visualize
print("\n5. GENERATING ACCELERATION PLOT...")
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Joint Accelerations - Checking for Unrealistic Values', fontsize=14, fontweight='bold')

joints_to_check = ['knee_angle_r', 'ankle_angle_r', 'hip_flexion_r', 
                    'knee_angle_l', 'ankle_angle_l', 'hip_flexion_l']

for idx, joint_name in enumerate(joints_to_check):
    if joint_name in acc_data.columns:
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        acc_values = acc_data[joint_name].values
        ax.plot(acc_data['time'], acc_values, linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (rad/s²)')
        ax.set_title(f'{joint_name}\nRange: [{acc_values.min():.1f}, {acc_values.max():.1f}]')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        
        # Highlight if values are too high
        if np.abs(acc_values).max() > 50:
            ax.set_facecolor('#ffeeee')  # Light red background

plt.tight_layout()
plt.savefig('acceleration_diagnostic.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved to: acceleration_diagnostic.png")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
