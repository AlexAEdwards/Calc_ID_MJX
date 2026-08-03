import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def load_sto(file_path):
    """
    Load an OpenSim .sto file into a pandas DataFrame.
    Detects if data is in degrees from the header.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        start_row = 0
        in_degrees = False
        for i, line in enumerate(lines):
            if 'inDegrees=yes' in line:
                in_degrees = True
            if 'endheader' in line:
                start_row = i + 1
                break
        
        # OpenSim files sometimes have a "Coordinates" line after endheader
        if start_row < len(lines) and "Coordinates" in lines[start_row]:
            start_row += 1

        df = pd.read_csv(file_path, sep='\t', skiprows=start_row)
        
        # If tab separator fails (some exporters use spaces), try whitespace
        if len(df.columns) < 2:
            df = pd.read_csv(file_path, sep='\s+', skiprows=start_row)
            
        return df, in_degrees
    except Exception as e:
        print(f"    Error reading sto file: {e}")
        return None, False

def process_opencap_to_npy():
    root_dir = "/home/mobl/Documents/Classwork/BioSimClass/ClonedRepo/Calc_ID_MJX/Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects"
    
    # Target order based on user request (Indices 0-22)
    npy_column_order = [
        "pelvis_tilt",      # 0
        "pelvis_list",      # 1
        "pelvis_rotation",  # 2
        "pelvis_tx",        # 3
        "pelvis_ty",        # 4
        "pelvis_tz",        # 5
        "hip_flexion_r",    # 6
        "hip_adduction_r",  # 7
        "hip_rotation_r",   # 8
        "knee_angle_r",     # 9
        "ankle_angle_r",    # 10
        "subtalar_angle_r", # 11
        "mtp_angle_r",      # 12
        "hip_flexion_l",    # 13
        "hip_adduction_l",  # 14
        "hip_rotation_l",   # 15
        "knee_angle_l",     # 16
        "ankle_angle_l",    # 17
        "subtalar_angle_l", # 18
        "mtp_angle_l",      # 19
        "lumbar_extension", # 20
        "lumbar_bending",   # 21
        "lumbar_rotation"   # 22
    ]
    
    # Identify which columns are rotations (indices) to convert Degrees -> Radians
    # Index 3, 4, 5 are Translations (pelvis_tx/y/z) - stay in meters
    rot_indices = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found.")
        return

    subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"🚀 Processing {len(subjects)} subjects...")

    for subj in subjects:
        subj_path = os.path.join(root_dir, subj)
        trials = sorted([d for d in os.listdir(subj_path) if d.startswith("Trial_")])
        
        for trial in trials:
            motion_dir = os.path.join(subj_path, trial, "Motion")
            if not os.path.exists(motion_dir):
                continue
                
            # Find the .sto file (typically OC_results_OptFeet_walkingX_ik.sto)
            sto_files = [f for f in os.listdir(motion_dir) if f.endswith(".sto") and "_ik" in f]
            if not sto_files:
                continue
                
            sto_path = os.path.join(motion_dir, sto_files[0])
            print(f"  📂 Subject {subj}, Trial {trial}: {sto_files[0]}")
            
            df, in_degrees = load_sto(sto_path)
            if df is None:
                continue

            # Identify time column
            time_col = next((c for c in df.columns if c.lower() == 'time'), None)
            if not time_col:
                print(f"    ❌ Error: No time column found.")
                continue
            
            orig_time = df[time_col].values
            
            # 1. Extraction and Unit Conversion
            rearranged_data = np.zeros((len(orig_time), len(npy_column_order)))
            for idx, col_name in enumerate(npy_column_order):
                if col_name in df.columns:
                    vals = df[col_name].values
                    if in_degrees and idx in rot_indices:
                        vals = np.deg2rad(vals)
                    rearranged_data[:, idx] = vals
                else:
                    print(f"    ⚠️ Warning: {col_name} missing, filling with 0.")
            
            # 2. Resampling to 100Hz
            # Set target frequency to 100Hz (0.01s intervals)
            dt = 0.01
            t_new = np.arange(orig_time[0], orig_time[-1], dt)
            # Ensure we don't exceed original time range due to float precision
            t_new = t_new[t_new <= orig_time[-1]]
            
            # Spline interpolation (kind='cubic') for smooth differentiation later
            interp_func = interp1d(orig_time, rearranged_data, axis=0, kind='cubic', 
                                  fill_value="extrapolate")
            pos_100hz = interp_func(t_new)
            
            # 3. Differentiation for Vel and Accel
            # First derivative (Velocity)
            vel_100hz = np.gradient(pos_100hz, dt, axis=0)
            
            # Second derivative (Acceleration)
            accel_100hz = np.gradient(vel_100hz, dt, axis=0)
            
            # 4. Save results
            np.save(os.path.join(motion_dir, "Pos.npy"), pos_100hz.astype(np.float32))
            np.save(os.path.join(motion_dir, "Vel.npy"), vel_100hz.astype(np.float32))
            np.save(os.path.join(motion_dir, "Accel.npy"), accel_100hz.astype(np.float32))
            np.save(os.path.join(motion_dir, "Time.npy"), t_new.astype(np.float32))
            
            print(f"    ✅ Saved Pos, Vel, Accel, Time ({len(t_new)} frames)")

if __name__ == "__main__":
    process_opencap_to_npy()
