import os
import shutil
from pathlib import Path

# Absolute Path to Workspace Root (adjusted for script location)
# But here I can just use relative paths if I run it from root.
# Let's use absolute paths based on where the user says the folders are relative to CWD.

# Using absolute path for safety based on workspace info
WORKSPACE_ROOT = Path("/home/mobl/Documents/Classwork/BioSimClass/ClonedRepo/Calc_ID_MJX")
DATASETS_DIR = WORKSPACE_ROOT / "Datasets_NAS/AddBiomechanicsDataset_All_npy"

SOURCE_BASE = DATASETS_DIR / "NeedsCleanedFromScott"
DEST_BASE = DATASETS_DIR / "OA_GaitRetrainingSubjects"
MAPPING_FILE = DATASETS_DIR / "OA_GaitRetraining.txt"

FILES_TO_COPY = ["COP.npy", "GRF.npy", "Moment.npy", "time.npy"]

def main():
    print(f"Reading mapping file: {MAPPING_FILE}")
    if not MAPPING_FILE.exists():
        print(f"Error: Mapping file not found at {MAPPING_FILE}")
        return

    with open(MAPPING_FILE, 'r') as f:
        lines = f.readlines()

    print(f"Found {len(lines)-1} entries to process.")
    
    success_count = 0
    missing_count = 0
    
    # Skip header
    for i, line in enumerate(lines[1:], 1):
        if not line.strip(): continue
        
        parts = line.strip().split(',')
        if len(parts) < 4:
            print(f"Skipping malformed line {i}: {line.strip()}")
            continue
            
        subject_id = parts[0].strip()   # e.g., Subject103
        orig_trc = parts[1].strip()     # e.g., rotated_baseline_OG1.trc
        trial_num = parts[3].strip()    # e.g., 1
        
        # 1. Process Source Name
        # Remove "rotated_" prefix
        # Note: If multiple "rotated_" exist, only remove leading? Logic says "beginning of every trial name"
        clean_name = orig_trc
        if clean_name.startswith("rotated_"):
            clean_name = clean_name[len("rotated_"):]
            
        # Remove ".trc" extension
        if clean_name.endswith(".trc"):
            clean_name = clean_name[:-4]
            
        # Construct Source Dir
        # Subject<###>/Gait/Week1/Edited/GRFmot/<TrialName>
        src_dir = SOURCE_BASE / subject_id / "Gait/Week1/Edited/GRFmot" / clean_name
        
        # 2. Construct Destination Dir
        # GaitRetraining_Subject<###>/Trial_<#>/Motion
        # Check naming: GaitRetraining_Subject103
        dest_subj_name = f"GaitRetraining_{subject_id}"
        dest_dir = DEST_BASE / dest_subj_name / f"Trial_{trial_num}" / "Motion"
        
        # 3. Copy Files
        files_found_for_trial = True
        
        # Define file mapping: Source Name -> Destination Name
        # User requested to rename Moment.npy -> GRM.npy
        file_mapping = {
            "COP.npy": "COP.npy",
            "GRF.npy": "GRF.npy",
            "Moment.npy": "GRM.npy",
            "time.npy": "time.npy"
        }

        # 1. Copy NPY files
        for src_fname, dest_fname in file_mapping.items():
            src_file = src_dir / src_fname
            dest_file = dest_dir / dest_fname
            
            if src_file.exists():
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_file)
                except Exception as e:
                    print(f"  [ERR] Failed to copy {src_file}: {e}")
                    files_found_for_trial = False
            else:
                if src_fname == "COP.npy": # Just use one file to check existence of folder essentially
                    print(f"  [MISSING] Source dir not found or empty: {src_dir}")
                files_found_for_trial = False
                missing_count += 1
        
        # 2. Cleanup old Moment.npy if it exists (renaming logic)
        old_moment_file = dest_dir / "Moment.npy"
        if old_moment_file.exists():
            try:
                old_moment_file.unlink()
            except OSError as e:
                print(f"  [ERR] Failed to delete old Moment.npy: {e}")

        # 3. Copy .mot file
        # Expected name: {clean_name}_grf.mot
        mot_filename = f"{clean_name}_grf.mot"
        src_mot = src_dir / mot_filename
        dest_mot = dest_dir / mot_filename
        
        if src_mot.exists():
             try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_mot, dest_mot)
             except Exception as e:
                print(f"  [ERR] Failed to copy mot file {src_mot}: {e}")
        else:
             # Fallback: look for any .mot file?
             # print(f"  [WARN] .mot file not found: {src_mot}")
             pass
        
        if files_found_for_trial:
            success_count += 1
            if success_count % 10 == 0:
                print(f"Processed {success_count} trials successfully...")

    print("="*50)
    print(f"Processing Complete.")
    print(f"Trials successfully copied: {success_count}")
    print(f"Trials with missing source files: {missing_count} (approx)")

if __name__ == "__main__":
    main()
