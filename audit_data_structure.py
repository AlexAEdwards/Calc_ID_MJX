import os
import shutil

data_root = "Data_Full_Cleaned"

print(f"{'Subject':<30} | {'Valid Trials':<15}")
print("-" * 50)

total_trials_all = 0

if os.path.exists(data_root):
    for subject in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject)
        
        if not os.path.isdir(subject_path):
            continue
            
        valid_trial_count = 0
        
        # Iterate through items in subject folder
        for item in os.listdir(subject_path):
            item_path = os.path.join(subject_path, item)
            
            # Delete 'combined' folders
            if item == "combined" and os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    # print(f"Deleted 'combined' folder in {subject}")
                except Exception as e:
                    print(f"Error deleting combined in {subject}: {e}")
                continue
                
            # Ignore 'Geometry' folders
            if item == "Geometry":
                continue
                
            # Check if it is a trial folder
            if os.path.isdir(item_path):
                motion_path = os.path.join(item_path, "Motion")
                calc_path = os.path.join(item_path, "calculatedInputs")
                
                # Check if both Motion and calculatedInputs folders exist
                # Note: You might want to check for specific files inside, but the prompt asked for folders.
                if os.path.isdir(motion_path) and os.path.isdir(calc_path):
                    valid_trial_count += 1
        
        print(f"{subject:<30} | {valid_trial_count:<15}")
        
        if valid_trial_count == 0:
            try:
                shutil.rmtree(subject_path)
                print(f"  -> DELETED subject folder: {subject} (0 valid trials)")
            except Exception as e:
                print(f"  -> Error deleting {subject}: {e}")
        else:
            total_trials_all += valid_trial_count

print("-" * 50)
print(f"{'Total Valid Trials':<30} | {total_trials_all:<15}")
