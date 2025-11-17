"""
Visualize tau.csv data across multiple figures for better visibility.

This script loads the tau (torque/force) data and creates multiple figures
with subplots to display all DOFs clearly.

Author: GitHub Copilot
Date: October 17, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load tau data
tau_path = "PatientData/Falisse_2017_subject_01/tau.csv"
tau_df = pd.read_csv(tau_path)

print("="*70)
print("TAU DATA VISUALIZATION")
print("="*70)
print(f"Data shape: {tau_df.shape}")
print(f"Number of DOFs: {tau_df.shape[1] - 1}")  # Subtract time column
print(f"Time range: [{tau_df['time'].min():.3f}, {tau_df['time'].max():.3f}] s")
print(f"Number of timesteps: {len(tau_df)}")

# Get time array
time = tau_df['time'].values

# Get all DOF columns (exclude time)
dof_columns = tau_df.columns[1:].tolist()

# Split DOFs into logical groups for multiple figures
groups = {
    'Pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 
               'pelvis_tx', 'pelvis_ty', 'pelvis_tz'],
    'Right Leg - Hip & Knee': ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 
                                'knee_angle_r'],
    'Right Leg - Ankle & Foot': ['ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r'],
    'Left Leg - Hip & Knee': ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
                               'knee_angle_l'],
    'Left Leg - Ankle & Foot': ['ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'],
    'Lumbar': ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
}

print("\n" + "="*70)
print("GENERATING PLOTS...")
print("="*70)

# Create a figure for each group
for group_name, dof_list in groups.items():
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
    fig.suptitle(f'Tau (Torques/Forces): {group_name}', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_dofs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plot each DOF in the group
    for idx, dof_name in enumerate(dof_list):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        # Get data for this DOF
        tau_values = tau_df[dof_name].values
        
        # Calculate statistics
        mean_val = np.mean(tau_values)
        max_val = np.max(tau_values)
        min_val = np.min(tau_values)
        std_val = np.std(tau_values)
        
        # Plot
        ax.plot(time, tau_values, linewidth=2, color='darkblue', label='Tau')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(y=mean_val, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'Mean: {mean_val:.1f}')
        
        # Labels and formatting
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Torque/Force (N·m or N)', fontsize=10)
        ax.set_title(f'{dof_name}\n[Min: {min_val:.1f}, Max: {max_val:.1f}, Std: {std_val:.1f}]', 
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        
        print(f"  {dof_name:25s}: [{min_val:8.2f}, {max_val:8.2f}] N·m, Mean: {mean_val:8.2f}, Std: {std_val:8.2f}")
    
    # Hide unused subplots
    for idx in range(n_dofs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    print(f"\n✓ Figure created: {group_name}")

# Create one additional figure showing all DOFs overlaid for comparison
fig_all, ax_all = plt.subplots(figsize=(16, 8), facecolor='white')
ax_all.set_facecolor('white')

# Plot all DOFs with different colors
colors = plt.cm.tab20(np.linspace(0, 1, len(dof_columns)))
for idx, dof_name in enumerate(dof_columns):
    tau_values = tau_df[dof_name].values
    ax_all.plot(time, tau_values, linewidth=1.5, alpha=0.7, 
                color=colors[idx], label=dof_name)

ax_all.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax_all.set_xlabel('Time (s)', fontsize=12)
ax_all.set_ylabel('Torque/Force (N·m or N)', fontsize=12)
ax_all.set_title('All DOFs - Tau (Torques/Forces) Comparison', fontsize=14, fontweight='bold')
ax_all.grid(True, alpha=0.3)
ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
plt.tight_layout()

print("\n✓ Figure created: All DOFs Comparison")

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Overall statistics
all_tau_values = tau_df[dof_columns].values.flatten()
print(f"Overall min:  {np.min(all_tau_values):10.2f} N·m")
print(f"Overall max:  {np.max(all_tau_values):10.2f} N·m")
print(f"Overall mean: {np.mean(all_tau_values):10.2f} N·m")
print(f"Overall std:  {np.std(all_tau_values):10.2f} N·m")

# Find the DOF with maximum absolute torque
max_abs_torques = tau_df[dof_columns].abs().max()
max_dof = max_abs_torques.idxmax()
max_torque = max_abs_torques.max()
print(f"\nDOF with largest torque: {max_dof} = {max_torque:.2f} N·m")

print("\n" + "="*70)
print("Displaying all plots...")
print("Close each plot window to see the next one.")
print("="*70)

# Show all plots
plt.show()
