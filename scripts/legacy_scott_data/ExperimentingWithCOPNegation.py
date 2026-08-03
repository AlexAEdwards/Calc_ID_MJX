import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Hardcoded Dataset Path
DATASET_PATH = "Datasets_NAS/AddBiomechanicsDataset_All_npy/OA_GaitRetrainingSubjects"

def process_dataset(dataset_root):
    dataset_path = Path(dataset_root)
    
    if "OA_GaitRetrainingSubjects" not in dataset_path.name:
        print(f"Skipping dataset: {dataset_path.name} (Not OA_GaitRetrainingSubjects)")
        return

    print(f"Processing dataset: {dataset_path.name}")
    
    # Iterate through subject folders
    for subject_dir in sorted(dataset_path.iterdir()):
        if not subject_dir.is_dir():
            continue
            
        print(f"  Processing Subject: {subject_dir.name}")
        
        # Iterate through trial folders
        for trial_dir in sorted(subject_dir.iterdir()):
            if not (trial_dir.is_dir() and trial_dir.name.startswith("Trial_")):
                continue
            
            try:
                # Define paths
                ankle_pos_path = trial_dir / "TrainingData" / "anklePos.npy"
                cop_global_path = trial_dir / "Motion" / "mjx" / "COP_Cleaned_Global.npy"
                grf_path = trial_dir / "TrainingData" / "GRF_Cleaned.npy"
                output_folder = trial_dir / "Motion" / "mjx"
                output_path = output_folder / "COP_Cleaned_Relative_Negated.npy"
                output_path_global = output_folder / "COP_Cleaned_Negated.npy"
                
                # Check for required files
                if not (ankle_pos_path.exists() and cop_global_path.exists() and grf_path.exists()):
                    continue
                
                # Load data
                ankle_pos = np.load(ankle_pos_path)     # (2, N, 3) 0:R, 1:L
                cop_global = np.load(cop_global_path)   # (N, 6) 0-2:R, 3-5:L
                grf = np.load(grf_path)                 # (N, 6) 0-2:R, 3-5:L

                # 1. Extract X and Y for both feet
                # COP: Right X, Y (0, 1) and Left X, Y (3, 4)
                right_cop_xy = cop_global[:, 0:2].copy()
                left_cop_xy = cop_global[:, 3:5].copy()
                
                # Ankle: Right [0] X, Y (0, 1) and Left [1] X, Y (0, 1)
                right_ankle_xy = ankle_pos[0, :, 0:2].copy()
                left_ankle_xy = ankle_pos[1, :, 0:2].copy()
                
                # 2. Determine and apply negation based on COP slope during stance
                # We also create a copy of the global COP to save as negated version
                cop_global_negated = cop_global.copy()

                # Right Foot (GRF index 2 is vertical, index 0 is COP X)
                right_is_stance = grf[:, 2] > 0
                if np.any(right_is_stance):
                    right_slope = np.mean(np.diff(cop_global[right_is_stance, 0]))
                    if right_slope < -1e-5:
                        right_cop_xy *= -1
                        cop_global_negated[:, 0:2] *= -1
                
                # Left Foot (GRF index 5 is vertical, index 3 is COP X)
                left_is_stance = grf[:, 5] > 0
                if np.any(left_is_stance):
                    left_slope = np.mean(np.diff(cop_global[left_is_stance, 3]))
                    if left_slope < -1e-5:
                        left_cop_xy *= -1
                        cop_global_negated[:, 3:5] *= -1

                # 3. Calculate Relative COP (N, 2 for each foot)
                right_rel = right_cop_xy - right_ankle_xy
                left_rel = left_cop_xy - left_ankle_xy

                # 4. Apply GRF Thresholding (< 5N vertical GRF resets COP to zero)
                right_rel[grf[:, 2] < 5.0] = 0
                left_rel[grf[:, 5] < 5.0] = 0
                
                # 5. Concatenate and save (Shape: num_timesteps, 4)
                cop_relative_result = np.concatenate([right_rel, left_rel], axis=1)
                np.save(output_path, cop_relative_result)

                # 6. Save the negated global COP (Shape: num_timesteps, 6)
                np.save(output_path_global, cop_global_negated)
                
            except Exception as e:
                print(f"    Error processing {subject_dir.name}/{trial_dir.name}: {e}")

if __name__ == "__main__":
    process_dataset(DATASET_PATH)
    print("\nSecondary cleaning complete.")
