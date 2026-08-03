import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT = "Datasets_NAS/AddBiomechanicsDataset_All_npy/NeedsCleanedFromScott"
RELATIVE_GRF_PATH = "Gait/Week1/Edited/GRFmot"
ZERO_THRESHOLD = 20.0  # Newtons, same as processing script

# OpenSim Coordinates: X=Forward, Y=Up, Z=Right

def visualize_subject_grf(subject_name):
    """
    Visualize GRF and COP data for all trials of a given subject.
    """
    # Get absolute path to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    subject_dir = os.path.join(project_root, DATA_ROOT, subject_name)
    grf_dir = os.path.join(subject_dir, RELATIVE_GRF_PATH)
    
    if not os.path.isdir(grf_dir):
        print(f"Error: GRF data directory not found: {grf_dir}")
        return

    # Find all trial directories (each contains .npy files)
    trial_dirs = sorted([d for d in glob.glob(os.path.join(grf_dir, "*")) if os.path.isdir(d)])
    
    if not trial_dirs:
        print(f"No trial data found in {grf_dir}")
        return

    print(f"Found {len(trial_dirs)} trials for {subject_name}")
    print("Close the plot window to proceed to the next trial. Press Ctrl+C in terminal to exit.")

    for trial_path in trial_dirs:
        trial_name = os.path.basename(trial_path)
        print(f"\nVisualizing trial: {trial_name}")
        
        try:
            # Load Combined Data: GRF.npy, Moment.npy, COP.npy
            # Shape: (N, 6) -> [Right_0, Right_1, Right_2, Left_0, Left_1, Left_2]
            # OpenSim Frame: X=Forward, Y=Up, Z=Right
            
            time = np.load(os.path.join(trial_path, 'time.npy'))
            
            grf_combined = np.load(os.path.join(trial_path, 'GRF.npy'))
            cop_combined = np.load(os.path.join(trial_path, 'COP.npy'))
            moment_combined = np.load(os.path.join(trial_path, 'Moment.npy'))
            
            # Split back into Right/Left for plotting convenience
            # Columns 0-2 = Right, Columns 3-5 = Left
            r_forces = grf_combined[:, 0:3]
            l_forces = grf_combined[:, 3:6]
            
            r_cop = cop_combined[:, 0:3]
            l_cop = cop_combined[:, 3:6]
            
            r_moments = moment_combined[:, 0:3]
            l_moments = moment_combined[:, 3:6]
            
            # No masking with NaNs requested. 
            # User wants to see zeros where data is zero.
            r_f_plot = r_forces
            l_f_plot = l_forces
            r_c_plot = r_cop
            l_c_plot = l_cop
            r_m_plot = r_moments
            l_m_plot = l_moments

            # Create figure
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle(f"GRF and COP Visualization - {subject_name}: {trial_name}", fontsize=16)
            
            # Use GridSpec for better control
            gs = plt.GridSpec(5, 3, height_ratios=[1, 1, 1, 1, 1.5], hspace=0.4, wspace=0.3)

            # --- Row 1 & 2: Forces (Separate subplots for R/L) ---
            axes_f = []
            for i, side in enumerate(['Right', 'Left']):
                data = r_f_plot if side == 'Right' else l_f_plot
                color = 'tab:red' if side == 'Right' else 'tab:blue'
                for j, (comp, ylabel) in enumerate(zip(['Fx', 'Fy', 'Fz'], ['Forward (N)', 'Vertical (N)', 'Lateral (N)'])):
                    ax = fig.add_subplot(gs[i, j])
                    ax.plot(time, data[:, j], color=color, linewidth=1.5)
                    ax.set_title(f"{side} Force {comp}")
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3)
                    if i == 1: ax.set_xlabel("Time (s)")

            # --- Row 3 & 4: Moments (Separate subplots for R/L) ---
            # Usually My (Free Moment) is most relevant, but we'll show all
            for i, side in enumerate(['Right', 'Left']):
                data = r_m_plot if side == 'Right' else l_m_plot
                color = 'tab:orange' if side == 'Right' else 'tab:purple'
                for j, (comp, ylabel) in enumerate(zip(['Mx', 'My', 'Mz'], ['N*m', 'N*m (Free)', 'N*m'])):
                    ax = fig.add_subplot(gs[i+2, j])
                    ax.plot(time, data[:, j], color=color, linewidth=1.5)
                    ax.set_title(f"{side} Moment {comp}")
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3)
                    if i == 1: ax.set_xlabel("Time (s)")

            # --- Row 5: COP Analysis ---
            # Col 0: COP Time Series (Right)
            ax_cr = fig.add_subplot(gs[4, 0])
            ax_cr.plot(time, r_c_plot[:, 0], label='COP X (Fwd)', color='tab:red')
            ax_cr.plot(time, r_c_plot[:, 2], label='COP Z (Lat)', color='tab:green')
            ax_cr.set_title("Right COP vs Time")
            ax_cr.set_ylabel("Position (m)")
            ax_cr.set_xlabel("Time (s)")
            ax_cr.legend()
            ax_cr.grid(True, alpha=0.3)

            # Col 1: COP Time Series (Left)
            ax_cl = fig.add_subplot(gs[4, 1])
            ax_cl.plot(time, l_c_plot[:, 0], label='COP X (Fwd)', color='tab:blue')
            ax_cl.plot(time, l_c_plot[:, 2], label='COP Z (Lat)', color='tab:orange')
            ax_cl.set_title("Left COP vs Time")
            ax_cl.set_ylabel("Position (m)")
            ax_cl.set_xlabel("Time (s)")
            ax_cl.legend()
            ax_cl.grid(True, alpha=0.3)

            # Col 2: Top-down 2D COP (Overlaid)
            ax_2d = fig.add_subplot(gs[4, 2])
            ax_2d.plot(r_c_plot[:, 2], r_c_plot[:, 0], 'r.', markersize=2, label='Right')
            ax_2d.plot(l_c_plot[:, 2], l_c_plot[:, 0], 'b.', markersize=2, label='Left')
            ax_2d.set_title("Top-down COP (X vs Z)")
            ax_2d.set_xlabel("Lateral Position (m)")
            ax_2d.set_ylabel("Forward Position (m)")
            ax_2d.set_aspect('equal', 'datalim')
            ax_2d.legend()
            ax_2d.grid(True, alpha=0.3)

            plt.show()
            
        except Exception as e:
            print(f"Error plotting trial {trial_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        subj = sys.argv[1]
    else:
        subj = input("Enter subject name (e.g. Subject102): ").strip()
    
    if subj:
        visualize_subject_grf(subj)
    else:
        print("No subject name provided.")
