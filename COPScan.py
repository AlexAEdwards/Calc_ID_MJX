import os
import numpy as np
import jax.numpy as jnp
import shutil

data_root = "/home/mobl/Documents/Classwork/BioSimClass/Data_Full_Cleaned"  # Adjust path as needed
Bad_Percentage_all=[]
Scaled_Percentages_all=[]
for subject in os.listdir(data_root):
    subject_path = os.path.join(data_root, subject)
    if os.path.isdir(subject_path):
        for trial in os.listdir(subject_path):
            trial_path = os.path.join(subject_path, trial)
            if os.path.isdir(trial_path):
                motion_path = os.path.join(trial_path, "Motion")
                try:
                    if os.path.isdir(motion_path):
                        cop_clean_path = os.path.join(motion_path, "mjx/COP_Cleaned.npy")
                        grf_clean_path = os.path.join(motion_path, "mjx/GRF_Cleaned.npy")
                        if os.path.isfile(cop_clean_path) and os.path.isfile(grf_clean_path):
                            cop_clean_data = np.load(cop_clean_path)
                            grf_clean_data = np.load(grf_clean_path)
                            # print(f"Loaded COP_Cleaned.npy and GRF_Cleaned.npy for {subject}/{trial}")
                            grf_left = grf_clean_data[:,3:6]
                            grf_right = grf_clean_data[:,0:3]
                    calc_inputs_path = os.path.join(trial_path, "calculatedInputs")
                    if os.path.isdir(calc_inputs_path):
                        ankle_pos_path = os.path.join(calc_inputs_path, "anklePos.npy")
                        if os.path.isfile(ankle_pos_path):
                            ankle_pos_data = np.load(ankle_pos_path)
                            # print(f"Loaded anklePos.npy for {subject}/{trial}")
                            
                            # Check shape of ankle_pos_data
                            # Expected shape: (2, timesteps, 3) OR (timesteps, 2, 3)
                            if ankle_pos_data.shape[0] == 2 and len(ankle_pos_data.shape) == 3:
                                # Shape is (2, timesteps, 3)
                                ankle_pos_r_all = ankle_pos_data[1,:,:]
                                ankle_pos_l_all = ankle_pos_data[0,:,:]
                                print(f"Ankle position shape for {subject}/{trial}: {ankle_pos_data.shape}")
                            elif ankle_pos_data.shape[1] == 2 and len(ankle_pos_data.shape) == 3:
                                # Shape is (timesteps, 2, 3) - DELETE TRIAL
                                print(f"Deleting trial {subject}/{trial} due to incorrect anklePos shape: {ankle_pos_data.shape}")
                                shutil.rmtree(trial_path)
                                continue
                            else:
                                print(f"Unexpected anklePos shape for {subject}/{trial}: {ankle_pos_data.shape}")
                                continue

                            cop_right_all = cop_clean_data[:,0:3]
                            cop_left_all = cop_clean_data[:,3:6]

                            # Handle length mismatch if any
                            min_len = min(ankle_pos_r_all.shape[0], cop_right_all.shape[0])
                            if ankle_pos_r_all.shape[0] != cop_right_all.shape[0]:
                                # print(f"Length mismatch for {subject}/{trial}: Ankle {ankle_pos_r_all.shape[0]}, COP {cop_right_all.shape[0]}. Trimming to {min_len}.")
                                ankle_pos_r_all = ankle_pos_r_all[:min_len]
                                ankle_pos_l_all = ankle_pos_l_all[:min_len]
                                cop_right_all = cop_right_all[:min_len]
                                cop_left_all = cop_left_all[:min_len]
                                grf_right = grf_right[:min_len]
                                grf_left = grf_left[:min_len]
                                print(f"After trimming, data length for {subject}/{trial}: {min_len}")

                            r_vec_r = cop_right_all - ankle_pos_r_all
                            r_vec_l = cop_left_all - ankle_pos_l_all
                            for i in range(len(grf_right)):
                                if jnp.linalg.norm(grf_right[i, :]) < 1e-3:
                                    r_vec_r[i,:] = [0,0,0]
                                if jnp.linalg.norm(grf_left[i, :]) < 1e-3:
                                    r_vec_l[i,:] = [0,0,0]
                            r_vec_l_magnitudes = jnp.linalg.norm(r_vec_l, axis=1)
                            r_vec_r_magnitudes = jnp.linalg.norm(r_vec_r, axis=1)
                            bad_indices_left = jnp.where(r_vec_l_magnitudes > 0.3)[0]
                            bad_indices_right = jnp.where(r_vec_r_magnitudes > 0.3)[0]
                            num_timesteps = r_vec_l.shape[0]
                            # Concatenate bad indices from left and right
                            bad_indices = jnp.unique(jnp.concatenate((bad_indices_left, bad_indices_right)))
                            # if len(bad_indices) > 0.2 * num_timesteps:
                        
                            # Print percentage of bad frames
                            bad_percentage = (len(bad_indices) / num_timesteps) * 100
                            # print(f"Bad COP data percentage for {subject}/{trial}: {bad_percentage:.2f}%")
                            Bad_Percentage_all.append(bad_percentage/10)
                            for i in range(int(np.round(num_timesteps))):
                                Scaled_Percentages_all.append(bad_percentage)
                except Exception as e:
                    print(f"Error processing {subject}/{trial}: {e}")
import matplotlib.pyplot as plt
# After processing all trials, print overall statistics
if Bad_Percentage_all:
    plt.figure(figsize=(8,6))
    plt.hist(Bad_Percentage_all, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Bad COP Data Percentage (%)')
    plt.ylabel('Number of Trials')
    plt.title('Histogram of Bad COP Data Percentage Across Trials')
    plt.grid(True)
    plt.show()
if Scaled_Percentages_all:
    plt.figure(figsize=(8,6))
    plt.hist(Scaled_Percentages_all, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Scaled Bad COP Data Percentage (%)')
    plt.ylabel('Number of Trials')
    plt.title('Histogram of Scaled Bad COP Data Percentage Across Trials')
    plt.grid(True)
    plt.show()