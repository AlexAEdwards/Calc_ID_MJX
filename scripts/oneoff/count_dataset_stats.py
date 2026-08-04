import os
import re
import numpy as np
from pathlib import Path

def analyze_dataset(dataset_dir: Path):
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"Error: Directory {dataset_dir} does not exist.")
        return

    # Tracking dictionaries for subjects and total frames
    # Keys: "GaitRetraining", "S_or_S_GAH", "Remaining"
    stats = {
        "GaitRetraining": {"subjects": 0, "total_frames": 0},
        "S_or_S_GAH": {"subjects": 0, "total_frames": 0},
        "Remaining": {"subjects": 0, "total_frames": 0},
    }

    # Group 2 regex: starts with S appended by digits, OR starts with S_GAH_ appended by digits
    group2_pattern = re.compile(r"^(S_GAH_\d+|S\d+)")

    print(f"\nScanning dataset in {dataset_dir}...")
    print("-" * 50)

    # Iterate over subject folders
    for subject_path in dataset_dir.iterdir():
        if not subject_path.is_dir():
            continue

        subj_name = subject_path.name

        # Categorize
        if "GaitRetraining_" in subj_name:
            group = "GaitRetraining"
        elif group2_pattern.match(subj_name):
            group = "S_or_S_GAH"
        else:
            group = "Remaining"

        stats[group]["subjects"] += 1

        # Search for all pos_inputs.npy recursively inside the subject folder
        subject_frames = 0
        for npy_file in subject_path.rglob("ProcessedData/pos_inputs.npy"):
            try:
                # Use mmap_mode="r" to load just the shape without reading into memory
                arr = np.load(npy_file, mmap_mode="r")
                subject_frames += arr.shape[0]  # Assuming first dimension is time/frames
            except Exception as e:
                print(f"Failed to read {npy_file}: {e}")
        
        stats[group]["total_frames"] += subject_frames

    print(f"--- Final Statistics for {dataset_dir.name} ---")
    for group, data in stats.items():
        subjects = data["subjects"]
        frames = data["total_frames"]
        
        minutes = frames / (100 * 60)
        hours = frames / (100 * 3600)
        
        print(f"Group: {group}")
        print(f"  Total Subjects: {subjects}")
        print(f"  Total Duration: {frames:,} frames")
        print(f"  Estimated Time (at 100Hz): {minutes:.2f} minutes ({hours:.2f} hours)")
        print()

def main():
    datasets = [
        Path("TrustedDataSetNoised12Distributed"),
        Path("Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects")
    ]
    
    for ds in datasets:
        analyze_dataset(ds)

if __name__ == "__main__":
    main()
