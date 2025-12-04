"""
Visualization Script for Trial Data Structure

This script cycles through all subjects and trials in the Data folder and generates
visualizations showing:
1. Joint positions over time (from Motion/pos_mjx.npy) - highlighting Knee and Ankle joints
2. Inverse dynamics results over time (from calculatedInputs/ID_Results_MJX.npy) - highlighting Knee and Ankle joints
3. Both ankle positions in a single 3D plot (from calculatedInputs/anklePos.npy)

Figures are displayed interactively (not saved).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import os


# Joint index mappings (based on your model structure)
# These correspond to the qpos indices in your MuJoCo model
JOINT_INDICES = {
    'knee_angle_r': 11,      # Right knee
    'knee_angle_l': 26,      # Left knee
    'ankle_angle_r': 14,     # Right ankle
    'ankle_angle_l': 29,     # Left ankle
}

JOINT_NAMES = {
    11: 'Right Knee',
    26: 'Left Knee',
    14: 'Right Ankle',
    29: 'Left Ankle',
}


def visualize_trial_data(subject_path, trial_path, save_figures=False):
    # =========================================================================
    # SUBPLOT 6: X COP Value for Both Feet
    # =========================================================================

    """
    Generate visualization for a single trial.
    
    Parameters:
    -----------
    subject_path : Path
        Path to subject folder
    trial_path : Path
        Path to trial folder
    save_figures : bool
        Whether to save figures to disk (default: False - display only)
    """
    subject_name = subject_path.name
    trial_name = trial_path.name
    
    print(f"  Visualizing {subject_name}/{trial_name}...")
    
    # Define file paths
    motion_path = trial_path / "Motion"
    mjx_path = motion_path / "mjx"  # MJX-specific files are in Motion/mjx/
    calculated_path = trial_path / "calculatedInputs"
    
    # Load data files
    try:
        pos_mjx = np.load(mjx_path / "pos_mjx.npy")  # Shape: (timesteps, num_joints)
        id_results = np.load(calculated_path / "ID_Results_MJX.npy")  # Shape: (timesteps, num_joints)
        ankle_pos = np.load(calculated_path / "anklePos.npy")  # Shape: (2, timesteps, 3) - [left, right]
        grf_path = mjx_path / "GRF_Cleaned.npy"
        cop_cleaned_path = mjx_path / "COP_Cleaned.npy"
        grf = np.load(grf_path) if grf_path.exists() else None  # Shape: (timesteps, 6)
        cop_cleaned = np.load(cop_cleaned_path) if cop_cleaned_path.exists() else None  # Shape: (timesteps, 6)
    except FileNotFoundError as e:
        print(f"    ⚠️  Missing file: {e}")
        return
    
    # Get dimensions
    num_timesteps = pos_mjx.shape[0]
    num_joints = pos_mjx.shape[1]
    time = np.arange(num_timesteps) * 0.01  # Assuming dt=0.01s
    
    # Extract left and right ankle positions
    ankle_left = ankle_pos[0, :, :]   # Shape: (timesteps, 3) - [x, y, z]
    ankle_right = ankle_pos[1, :, :]  # Shape: (timesteps, 3) - [x, y, z]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'{subject_name} - {trial_name}', fontsize=16, fontweight='bold')
    
    # =========================================================================
    # SUBPLOT 1: Joint Positions Over Time (KNEE AND ANKLE HIGHLIGHTED)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # First, plot all other joints in light gray (background)
    knee_ankle_indices = list(JOINT_INDICES.values())
    for j in range(num_joints):
        if j not in knee_ankle_indices:
            ax1.plot(time, pos_mjx[:, j], color='lightgray', alpha=0.3, linewidth=0.5)
    
    # Now plot KNEE and ANKLE joints with emphasis (thicker, colored lines)
    colors = {'knee_r': 'red', 'knee_l': 'blue', 'ankle_r': 'orange', 'ankle_l': 'cyan'}
    
    if JOINT_INDICES['knee_angle_r'] < num_joints:
        ax1.plot(time, pos_mjx[:, JOINT_INDICES['knee_angle_r']], 
                color=colors['knee_r'], linewidth=2.5, label='Right Knee', alpha=0.9)
    
    if JOINT_INDICES['knee_angle_l'] < num_joints:
        ax1.plot(time, pos_mjx[:, JOINT_INDICES['knee_angle_l']], 
                color=colors['knee_l'], linewidth=2.5, label='Left Knee', alpha=0.9)
    
    if JOINT_INDICES['ankle_angle_r'] < num_joints:
        ax1.plot(time, pos_mjx[:, JOINT_INDICES['ankle_angle_r']], 
                color=colors['ankle_r'], linewidth=2.5, label='Right Ankle', alpha=0.9)
    
    if JOINT_INDICES['ankle_angle_l'] < num_joints:
        ax1.plot(time, pos_mjx[:, JOINT_INDICES['ankle_angle_l']], 
                color=colors['ankle_l'], linewidth=2.5, label='Left Ankle', alpha=0.9)
    
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Position (rad)', fontsize=11)
    ax1.set_title('Joint Positions Over Time (Knee & Ankle Highlighted)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # SUBPLOT 2: Inverse Dynamics Results (KNEE AND ANKLE ONLY)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    # Only plot knee and ankle torque for both legs
    if JOINT_INDICES['knee_angle_r'] < num_joints:
        ax2.plot(time, id_results[:, JOINT_INDICES['knee_angle_r']], 
                color=colors['knee_r'], linewidth=2.5, label='Right Knee', alpha=0.9)
    if JOINT_INDICES['knee_angle_l'] < num_joints:
        ax2.plot(time, id_results[:, JOINT_INDICES['knee_angle_l']], 
                color=colors['knee_l'], linewidth=2.5, label='Left Knee', alpha=0.9)
    if JOINT_INDICES['ankle_angle_r'] < num_joints:
        ax2.plot(time, id_results[:, JOINT_INDICES['ankle_angle_r']], 
                color=colors['ankle_r'], linewidth=2.5, label='Right Ankle', alpha=0.9)
    if JOINT_INDICES['ankle_angle_l'] < num_joints:
        ax2.plot(time, id_results[:, JOINT_INDICES['ankle_angle_l']], 
                color=colors['ankle_l'], linewidth=2.5, label='Left Ankle', alpha=0.9)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Torque (N·m)', fontsize=11)
    ax2.set_title('Inverse Dynamics: Knee & Ankle Torque', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # SUBPLOT 3: Both Ankle 3D Trajectories (COMBINED)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    # =========================================================================
    # SUBPLOT 4: Ground Reaction Force Magnitude Over Time
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    # =========================================================================
    # SUBPLOT 5: Magnitude of r_vel_l and r_vel_r (COP-Ankle distance)
    # =========================================================================
    cop_ankle_r = np.zeros_like(cop_cleaned[:, 0:3])  # To store COP-Ankle differences for right foot
    cop_ankle_l = np.zeros_like(cop_cleaned[:, 3:6])  # To store COP-Ankle differences for left foot
    ax5 = fig.add_subplot(gs[2, 1])
    if cop_cleaned is not None:
        # Right foot: COP_cleaned[:, 0:3], Ankle_Pos right: ankle_right
        # Left foot: COP_cleaned[:, 3:6], Ankle_Pos left: ankle_left
        if cop_cleaned.shape[0] == ankle_left.shape[0] == ankle_right.shape[0]:
            # Only plot where GRF is nonzero
            if grf is not None and grf.shape[0] == cop_cleaned.shape[0]:
                grf_sum = np.abs(grf).sum(axis=1)
                nonzero_mask = grf_sum > 0
                r_vel_r = np.linalg.norm(cop_cleaned[:, 0:3] - ankle_right, axis=1)
                r_vel_l = np.linalg.norm(cop_cleaned[:, 3:6] - ankle_left, axis=1)
                # Magnitude plots
                ax5.plot(time[nonzero_mask], r_vel_r[nonzero_mask], color='orange', label='Right Foot |COP-Ankle|', linewidth=2)
                ax5.plot(time[nonzero_mask], r_vel_l[nonzero_mask], color='cyan', label='Left Foot |COP-Ankle|', linewidth=2)
                # Individual COP-Ankle components for right foot
                # cop_ankle_r[:,1:3] = cop_cleaned[:, 1:3] - ankle_right[:,1:3]
                # cop_ankle_r[:,0] = cop_cleaned[:, 0] -ankle_left[:,0]
                ax5.plot(time[nonzero_mask], cop_ankle_r[nonzero_mask, 0], color='red', linestyle='--', label='Right Foot ΔX', linewidth=1)
                ax5.plot(time[nonzero_mask], cop_ankle_r[nonzero_mask, 1], color='green', linestyle='--', label='Right Foot ΔY', linewidth=1)
                ax5.plot(time[nonzero_mask], cop_ankle_r[nonzero_mask, 2], color='blue', linestyle='--', label='Right Foot ΔZ', linewidth=1)
                # Individual COP-Ankle components for left foot
                cop_ankle_l = cop_cleaned[:, 3:6] - ankle_left
                ax5.plot(time[nonzero_mask], cop_ankle_l[nonzero_mask, 0], color='magenta', linestyle=':', label='Left Foot ΔX', linewidth=1)
                ax5.plot(time[nonzero_mask], cop_ankle_l[nonzero_mask, 1], color='lime', linestyle=':', label='Left Foot ΔY', linewidth=1)
                ax5.plot(time[nonzero_mask], cop_ankle_l[nonzero_mask, 2], color='navy', linestyle=':', label='Left Foot ΔZ', linewidth=1)
            else:
                ax5.plot([], [], color='orange', label='Right Foot |COP-Ankle|')
                ax5.plot([], [], color='cyan', label='Left Foot |COP-Ankle|')
            ax5.set_xlabel('Time (s)', fontsize=11)
            ax5.set_ylabel('Distance (m)', fontsize=11)
            ax5.set_title('COP-Ankle Distance: Magnitude & Components', fontsize=13, fontweight='bold')
            ax5.legend(loc='upper right', fontsize=9)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'COP_Cleaned and Ankle_Pos have mismatched timesteps', ha='center', va='center', fontsize=14, color='red', transform=ax5.transAxes)
            ax5.set_title('COP-Ankle Distance & GRF Components', fontsize=13, fontweight='bold')
            ax5.axis('off')
    else:
        ax5.text(0.5, 0.5, 'COP_Cleaned.npy not found', ha='center', va='center', fontsize=14, color='red', transform=ax5.transAxes)
        ax5.set_title('COP-Ankle Distance & GRF Components', fontsize=13, fontweight='bold')
        ax5.axis('off')
    if grf is not None:
        # Right foot: columns 0,1,2; Left foot: columns 3,4,5
        grf_right = grf[:, 0:3]
        grf_left = grf[:, 3:6]
        grf_right_mag = np.linalg.norm(grf_right, axis=1)
        grf_left_mag = np.linalg.norm(grf_left, axis=1)
        # Magnitude plots
        ax4.plot(time, grf_right_mag, color='orange', label='Right Foot |GRF|', linewidth=2)
        ax4.plot(time, grf_left_mag, color='cyan', label='Left Foot |GRF|', linewidth=2)
        # Individual components for right foot
        ax4.plot(time, grf_right[:, 0], color='red', linestyle='--', label='Right Foot GRF X', linewidth=1)
        ax4.plot(time, grf_right[:, 1], color='green', linestyle='--', label='Right Foot GRF Y', linewidth=1)
        ax4.plot(time, grf_right[:, 2], color='blue', linestyle='--', label='Right Foot GRF Z', linewidth=1)
        # Individual components for left foot
        ax4.plot(time, grf_left[:, 0], color='magenta', linestyle=':', label='Left Foot GRF X', linewidth=1)
        ax4.plot(time, grf_left[:, 1], color='lime', linestyle=':', label='Left Foot GRF Y', linewidth=1)
        ax4.plot(time, grf_left[:, 2], color='navy', linestyle=':', label='Left Foot GRF Z', linewidth=1)
        ax4.set_xlabel('Time (s)', fontsize=11)
        ax4.set_ylabel('GRF (N)', fontsize=11)
        ax4.set_title('Ground Reaction Force: Magnitude & Components', fontsize=13, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'GroundReactionForces.npy not found', ha='center', va='center', fontsize=14, color='red', transform=ax4.transAxes)
        ax4.set_title('Ground Reaction Force: Magnitude & Components', fontsize=13, fontweight='bold')
        ax4.axis('off')
    
    # Plot both ankle trajectories on the same plot
    ax3.plot(ankle_left[:, 0], ankle_left[:, 1], ankle_left[:, 2], 
             'b-', linewidth=2.5, label='Left Ankle', alpha=0.8)
    ax3.plot(ankle_right[:, 0], ankle_right[:, 1], ankle_right[:, 2], 
             'r-', linewidth=2.5, label='Right Ankle', alpha=0.8)
    
    # Mark start points (green)
    ax3.scatter(ankle_left[0, 0], ankle_left[0, 1], ankle_left[0, 2], 
                c='green', s=120, marker='o', label='Start', edgecolors='black', linewidths=2, zorder=10)
    ax3.scatter(ankle_right[0, 0], ankle_right[0, 1], ankle_right[0, 2], 
                c='green', s=120, marker='o', edgecolors='black', linewidths=2, zorder=10)
    
    # Mark end points (red)
    ax3.scatter(ankle_left[-1, 0], ankle_left[-1, 1], ankle_left[-1, 2], 
                c='darkred', s=120, marker='s', label='End', edgecolors='black', linewidths=2, zorder=10)
    ax3.scatter(ankle_right[-1, 0], ankle_right[-1, 1], ankle_right[-1, 2], 
                c='darkred', s=120, marker='s', edgecolors='black', linewidths=2, zorder=10)
    
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_zlabel('Z (m)', fontsize=10)
    ax3.set_title('Both Ankle 3D Trajectories', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    # Get the data ranges
    all_x = np.concatenate([ankle_left[:, 0], ankle_right[:, 0]])
    all_y = np.concatenate([ankle_left[:, 1], ankle_right[:, 1]])
    all_z = np.concatenate([ankle_left[:, 2], ankle_right[:, 2]])
    
    max_range = np.array([all_x.max()-all_x.min(), 
                          all_y.max()-all_y.min(), 
                          all_z.max()-all_z.min()]).max() / 2.0
    
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax6 = fig.add_subplot(gs[2, 0])
    if cop_cleaned is not None:
        # Right foot: COP_cleaned[:, 0] (X), Left foot: COP_cleaned[:, 3] (X)
        ax6.plot(time, cop_cleaned[:, 0], color='orange', label='Right Foot COP X', linewidth=2)
        ax6.plot(time, cop_cleaned[:, 3], color='cyan', label='Left Foot COP X', linewidth=2)
        ax6.set_xlabel('Time (s)', fontsize=11)
        ax6.set_ylabel('COP X (m)', fontsize=11)
        ax6.set_title('COP X Value Over Time', fontsize=13, fontweight='bold')
        ax6.legend(loc='upper right', fontsize=10)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'COP_Cleaned.npy not found', ha='center', va='center', fontsize=14, color='red', transform=ax6.transAxes)
        ax6.set_title('COP X Value Over Time', fontsize=13, fontweight='bold')
        ax6.axis('off')
    # Load COP_Cleaned.npy
    # cop_cleaned_path = trial_path / "Motion" / "COP_Cleaned.npy"
    # cop_cleaned = np.load(cop_cleaned_path) if cop_cleaned_path.exists() else None
    # =========================================================================
    # Display or Save Figure
    # =========================================================================
    if save_figures:
        # Create Visualizations folder
        viz_folder = trial_path / "Visualizations"
        viz_folder.mkdir(exist_ok=True)
        
        # Save figure
        output_file = viz_folder / f"{subject_name}_{trial_name}_visualization.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved: {output_file.name}")
        plt.close(fig)
    else:
        if grf is not None:
            grf_right = grf[:, 0:3]
            grf_left = grf[:, 3:6]
            #Average across time for each column
            avg_grf_right = np.mean(grf_right, axis=0)
            avg_grf_left = np.mean(grf_left, axis=0)
            print("    Average Right Foot GRF (X, Y, Z):", avg_grf_right)
            print("    Average Left Foot GRF (X, Y, Z):", avg_grf_left)
            #Print Shape of GRF array
            print("    GRF array shape:", grf.shape)
            COP_ANKLE_DIFFERENCE_ALL = []
            for t in range(num_timesteps):
                if np.mean(grf_right[t,:]) >.1:
                    # Copy COP for time t 3 times to make a matrix
                    cop_right=cop_cleaned[t, 0:3]
                    cop_right_matrix = np.tile(cop_right, (3, 1))
                    # subtract ankle position row vector from each row of cop_right_matrix
                    ankle_right_vector = ankle_right[t, :].reshape(1, 3)
                    difference_right = cop_right_matrix - ankle_right_vector.T
                    COP_ANKLE_DIFFERENCE_ALL.append(difference_right)
            # print out the average COP-ANKLE difference across all timepoints where GRF is nonzero for each entry
            COP_ANKLE_DIFFERENCE_ALL = np.array(COP_ANKLE_DIFFERENCE_ALL)
            avg_difference_right = np.mean(COP_ANKLE_DIFFERENCE_ALL, axis=0)
            print("    Average COP-Ankle Difference (Right Foot) across all timepoints with nonzero GRF:")
            print(avg_difference_right)

        # Display the figure
        plt.show(block=True)  # block=True waits for user to close window
        # Show average values for each GRF column




def visualize_all_trials(data_root="Data", num_subjects="all", specific_subject=None):
    """
    Cycle through all subjects and trials to generate visualizations.
    Only processes trials that have all required data files.
    
    Parameters:
    -----------
    data_root : str
        Root directory containing subject folders
    num_subjects : int or str
        Number of subjects to process ("all" or integer)
    specific_subject : str or None
        If specified, only visualize this specific subject folder name (e.g., "P002_split0", "subject7")
        When set, num_subjects is ignored.
    """
    print("="*70)
    print("TRIAL DATA VISUALIZATION")
    print("="*70)
    
    data_path = Path(data_root)
    
    # Folders to exclude (not subject folders)
    exclude_folders = {'GeometryAll', 'Geometry', 'geometry', 'GeometryWithMus', 'temp', 'logs'}
    
    # If specific subject is requested, only process that one
    if specific_subject:
        subject_path = data_path / specific_subject
        if not subject_path.exists():
            print(f"\n❌ Subject folder '{specific_subject}' not found in {data_root}")
            return
        if not subject_path.is_dir():
            print(f"\n❌ '{specific_subject}' is not a directory")
            return
        subjects = [subject_path]
        print(f"\nProcessing specific subject: {specific_subject}\n")
    else:
        # Find all subject folders (any folder that's not in exclude list)
        subjects = []
        for item in sorted(data_path.iterdir()):
            if item.is_dir() and item.name not in exclude_folders:
                # Additional check: subject folders should have trial subdirectories or at least an osim file
                has_trials = any(subitem.is_dir() for subitem in item.iterdir() if subitem.name.startswith("Trial"))
                has_osim = (item / "OpenSimModel.osim").exists()
                
                if has_trials or has_osim:
                    subjects.append(item)
    
        if not subjects:
            print(f"\n❌ No subject folders found in {data_root}")
            return
        
        print(f"Found {len(subjects)} subject folders")
        
        # Determine how many subjects to process
        if isinstance(num_subjects, str) and num_subjects.lower() == "all":
            subjects_to_process = subjects
            print(f"\nProcessing ALL {len(subjects)} subjects\n")
        elif isinstance(num_subjects, int):
            subjects_to_process = subjects[:num_subjects]
            print(f"\nProcessing first {len(subjects_to_process)} of {len(subjects)} subjects\n")
        else:
            print(f"\n❌ Invalid num_subjects parameter: {num_subjects}")
            return
    
    # After specific_subject check, subjects list is already set correctly
    subjects_to_process = subjects if specific_subject else (
        subjects if (isinstance(num_subjects, str) and num_subjects.lower() == "all") 
        else subjects[:num_subjects] if isinstance(num_subjects, int) 
        else subjects
    )
    
    total_trials = 0
    successful_trials = 0
    failed_trials = 0
    skipped_subjects = []
    skipped_trials = []
    
    for subject_idx, subject_path in enumerate(subjects_to_process, 1):
        subject_name = subject_path.name
        print(f"\n[{subject_idx}/{len(subjects_to_process)}] Processing subject: {subject_name}")
        print("-" * 70)
        
        # Find all trial folders with complete data
        trial_folders = []
        trial_candidates = []
        
        for item in sorted(subject_path.iterdir()):
            # Look for trial folders (case-insensitive, starts with "Trial" or "trial")
            if item.is_dir() and item.name.lower().startswith("trial"):
                trial_candidates.append(item.name)
                
                # Check if required folders exist
                motion_folder = item / "Motion"
                mjx_folder = motion_folder / "mjx"  # MJX-specific files subfolder
                calculated_folder = item / "calculatedInputs"
                
                if not (motion_folder.exists() and mjx_folder.exists() and calculated_folder.exists()):
                    continue
                
                # Check if all required files exist
                required_files = [
                    mjx_folder / "pos_mjx.npy",
                    calculated_folder / "ID_Results_MJX.npy",
                    calculated_folder / "anklePos.npy"
                ]
                
                all_files_exist = all(f.exists() for f in required_files)
                
                if all_files_exist:
                    trial_folders.append(item)
                else:
                    # Track which files are missing
                    missing = [f.name for f in required_files if not f.exists()]
                    skipped_trials.append((subject_name, item.name, missing))
        
        if len(trial_candidates) > 0:
            print(f"  Found {len(trial_candidates)} trial folder(s) total")
        
        if not trial_folders:
            if len(trial_candidates) > 0:
                print(f"  ⚠️  None of the {len(trial_candidates)} trial(s) have complete data - SKIPPING")
            else:
                print(f"  ⚠️  No trial folders found for {subject_name} - SKIPPING")
            skipped_subjects.append(subject_name)
            continue
        
        print(f"  Found {len(trial_folders)} trial(s) with complete data - VISUALIZING")
        
        # Process each trial
        for trial_path in trial_folders:
            total_trials += 1
            try:
                visualize_trial_data(subject_path, trial_path, save_figures=False)
                successful_trials += 1
            except Exception as e:
                failed_trials += 1
                print(f"    ❌ Failed: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)
    print(f"Subjects processed: {len(subjects_to_process)} of {len(subjects)} total")
    print(f"Subjects skipped (no complete data): {len(skipped_subjects)}")
    print(f"Trials skipped (missing files): {len(skipped_trials)}")
    print(f"Total trials visualized: {total_trials}")
    print(f"Successful: {successful_trials}")
    print(f"Failed: {failed_trials}")
    
    if skipped_subjects:
        print(f"\nSkipped subjects:")
        for subject in skipped_subjects:
            print(f"  - {subject}")
    
    if skipped_trials and len(skipped_trials) <= 20:  # Only show if not too many
        print(f"\nSkipped trials (missing files):")
        for subj, trial, missing in skipped_trials:
            print(f"  - {subj}/{trial}: missing {', '.join(missing)}")
    print("="*70)
    print("\nDone!")


# Main execution
if __name__ == "__main__":
    # You can change parameters to:
    # - specific_subject: Set to a subject folder name (e.g., "P002_split0", "subject7") to visualize only that subject
    # - num_subjects: "all" to process all subjects, or an integer (e.g., 5) to process only the first N subjects

    # IF you want to visualize a specific subject, uncomment below:
    # visualize_all_trials(data_root="Data", specific_subject="P002_split0")  # Visualize only specific subject

    # Otherwise, visualize all subjects:
    visualize_all_trials(data_root="Data_Full_Cleaned", num_subjects="all")  # Process all subjects

