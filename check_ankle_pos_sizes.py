import os
import numpy as np

data_root = "Data_Full_Cleaned"

print(f"{'Subject/Trial':<40} | {'AnklePos Shape':<20} | {'COP Shape':<20} | {'Status':<20}")
print("-" * 110)

# Walk through all directories recursively
for root, dirs, files in os.walk(data_root):
    # Check if this directory looks like a trial folder (contains Motion or calculatedInputs)
    has_motion = "Motion" in dirs
    has_calc = "calculatedInputs" in dirs
    
    if has_motion or has_calc:
        # Get relative path for display
        rel_path = os.path.relpath(root, data_root)
        
        # Paths
        ankle_path = os.path.join(root, "calculatedInputs", "anklePos.npy")
        cop_path = os.path.join(root, "Motion", "mjx", "COP_Cleaned.npy")
        
        ankle_shape = "Missing"
        cop_shape = "Missing"
        status = "OK"
        
        ankle_timesteps = -1
        cop_timesteps = -1
        
        if os.path.isfile(ankle_path):
            try:
                ankle_data = np.load(ankle_path)
                ankle_shape = str(ankle_data.shape)
                # Check dimensions to determine timesteps
                if len(ankle_data.shape) == 3:
                    # Heuristic: usually timesteps is the largest dimension
                    # But let's stick to the observation: (timesteps, 2, 3) or (2, timesteps, 3)
                    if ankle_data.shape[0] == 2:
                         ankle_timesteps = ankle_data.shape[1]
                    else:
                         ankle_timesteps = ankle_data.shape[0]
                else:
                    status = "Weird Ankle Shape"
            except Exception as e:
                ankle_shape = "Error"
                
        if os.path.isfile(cop_path):
            try:
                cop_data = np.load(cop_path)
                cop_shape = str(cop_data.shape)
                # Assuming shape is (timesteps, ...)
                if len(cop_data.shape) >= 1:
                    cop_timesteps = cop_data.shape[0]
            except Exception as e:
                cop_shape = "Error"

        if ankle_timesteps != -1 and cop_timesteps != -1:
            if ankle_timesteps != cop_timesteps:
                status = "MISMATCH"
            else:
                status = "MATCH"
        elif ankle_timesteps != -1:
             status = "COP Missing"
        elif cop_timesteps != -1:
             status = "Ankle Missing"
        else:
             status = "Both Missing"

        # Only print interesting cases (mismatches or missing files where one exists)
        if status != "MATCH" and status != "Both Missing":
             print(f"{rel_path:<40} | {ankle_shape:<20} | {cop_shape:<20} | {status:<20}")
