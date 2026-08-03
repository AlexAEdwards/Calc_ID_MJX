import os
import shutil
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
# Path to the directory containing subject folders
BASE_DIR = Path("/home/mobl/Documents/Classwork/BioSimClass/ClonedRepo/Calc_ID_MJX/Datasets_NAS/AddBiomechanicsDataset_All_npy/NeedsCleanedFromScott")

# The relative path that MUST exist within a subject folder to keep it
REQUIRED_SUBDIR = "Gait/Week1/Edited"

# Flag to control deletion. 
# Set to False (default) to only print patients missing the directory.
# Set to True to actually delete the folders.
REALLY_DELETE = True

def main():
    if not BASE_DIR.exists():
        print(f"Error: Base directory not found: {BASE_DIR}")
        return

    print(f"Scanning subjects in: {BASE_DIR}")
    print(f"Required subdirectory: {REQUIRED_SUBDIR}")
    print(f"Mode: {'DELETION' if REALLY_DELETE else 'DRY RUN (Printing missing only)'}")
    print("-" * 50)

    # Get all direct subdirectories (subject folders)
    subject_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    
    missing_count = 0
    total_count = len(subject_dirs)
    
    for subject_path in sorted(subject_dirs):
        # Check if the required path exists inside
        check_path = subject_path / REQUIRED_SUBDIR
        
        if not check_path.exists():
            missing_count += 1
            if REALLY_DELETE:
                print(f"[DELETING] {subject_path.name} (Missing {REQUIRED_SUBDIR})")
                try:
                    shutil.rmtree(subject_path)
                except Exception as e:
                    print(f"Error deleting {subject_path.name}: {e}")
            else:
                print(f"[MISSING] {subject_path.name}")
    
    print("-" * 50)
    print(f"Summary:")
    print(f"Total subjects scanned: {total_count}")
    print(f"Subjects missing {REQUIRED_SUBDIR}: {missing_count}")
    
    if not REALLY_DELETE and missing_count > 0:
        print("\nTo delete these folders, edit this script and set REALLY_DELETE = True.")

if __name__ == "__main__":
    main()
