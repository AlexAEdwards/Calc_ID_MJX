import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path

def process_gastroc_forces():
    root_dir = "/home/mobl/Documents/Classwork/BioSimClass/ClonedRepo/Calc_ID_MJX/Datasets_NAS/AddBiomechanicsDataset_All_npy/Gastroc_Avoidance_Healthy_MJX"
    
    # Column mapping based on user specifications:
    # 1_ prefix is Left foot, no prefix is Right foot.
    # OpenSim format: [vx, vy, vz] for force, [px, py, pz] for COP, [torque_x, torque_y, torque_z] for GRM.
    
    # We want standard output order: [R_x, R_y, R_z, L_x, L_y, L_z]
    force_mapping = {
        'right': ['ground_force_vx', 'ground_force_vy', 'ground_force_vz'],
        'left':  ['1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz']
    }
    
    cop_mapping = {
        'right': ['ground_force_px', 'ground_force_py', 'ground_force_pz'],
        'left':  ['1_ground_force_px', '1_ground_force_py', '1_ground_force_pz']
    }
    
    torque_mapping = {
        'right': ['ground_torque_x', 'ground_torque_y', 'ground_torque_z'],
        'left':  ['1_ground_torque_x', '1_ground_torque_y', '1_ground_torque_z']
    }
    
    if not os.path.exists(root_dir):
        print(f"❌ Error: {root_dir} not found.")
        return

    subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"🚀 Processing forces for {len(subjects)} subjects in Gastroc_Avoidance_Healthy_MJX...")

    total_processed = 0
    
    for subj in subjects:
        subj_path = os.path.join(root_dir, subj)
        trials = sorted([d for d in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, d)) and d.startswith("Trial_")])
        
        for trial in trials:
            motion_dir = os.path.join(subj_path, trial, "Motion")
            if not os.path.exists(motion_dir):
                continue
                
            # Find the .mot force file
            force_files = [f for f in os.listdir(motion_dir) if f.endswith("_forces.mot")]
            if not force_files:
                continue
                
            force_path = os.path.join(motion_dir, force_files[0])
            print(f"  📂 Subject {subj}, Trial {trial}: {force_files[0]}")
            
            try:
                # 1. Parse .mot file
                header_end = 0
                column_names = []
                with open(force_path, 'r') as f:
                    for i, line in enumerate(f):
                        if 'endheader' in line:
                            header_end = i + 1
                            # Next line should be column names
                            col_line = next(f).strip()
                            column_names = col_line.split('\t')
                            if len(column_names) <= 1:
                                column_names = col_line.split()
                            break
                
                # Load data using pandas
                df = pd.read_csv(force_path, sep='\s+', skiprows=header_end)
                if len(df.columns) != len(column_names):
                    df.columns = column_names
                
                if 'time' not in df.columns:
                    df.columns = [c.lower() for c in df.columns]

                orig_time = df['time'].values
                
                # 2. Setup Resampling (target 100Hz = 0.01s)
                dt = 0.01
                # Start at a clean multiple of dt
                t_start = np.ceil(orig_time[0] / dt) * dt
                t_end = np.floor(orig_time[-1] / dt) * dt
                t_new = np.arange(t_start, t_end + dt/2, dt)

                def extract_and_resample(mapping):
                    # Combine Right components then Left components
                    cols = mapping['right'] + mapping['left']
                    data = df[cols].values
                    interp_func = interp1d(orig_time, data, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
                    return interp_func(t_new).astype(np.float32)

                # 3. Process each group
                grf_100hz = extract_and_resample(force_mapping)
                cop_100hz = extract_and_resample(cop_mapping)
                grm_100hz = extract_and_resample(torque_mapping)

                # 4. Save directly to Motion folder
                np.save(os.path.join(motion_dir, "GRF.npy"), grf_100hz)
                np.save(os.path.join(motion_dir, "COP.npy"), cop_100hz)
                np.save(os.path.join(motion_dir, "GRM.npy"), grm_100hz)
                
                print(f"    ✅ Saved GRF, COP, GRM ({len(t_new)} frames at 100Hz)")
                total_processed += 1

            except Exception as e:
                print(f"    ❌ Error processing {trial}: {e}")

    print(f"\n✅ Done! Processed {total_processed} trials.")

if __name__ == "__main__":
    process_gastroc_forces()
