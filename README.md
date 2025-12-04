# Calc_ID_MJX - Inverse Dynamics Pipeline using MuJoCo MJX

This repository contains a pipeline for computing inverse dynamics of human motion using MuJoCo's JAX-accelerated backend (MJX). The workflow processes motion capture and force plate data through a MuJoCo musculoskeletal model to calculate joint torques.

---

## Python Files Overview

### 1. `BatchDataProcessing.py` (1616 lines)
**Purpose:** Main batch processing script for computing inverse dynamics across multiple subjects and trials.

- Iterates through all subjects/trials in the `Data_Full_Cleaned` directory
- Loads motion data (positions, velocities, accelerations) and force plate data (GRF, COP, moments)
- Transforms coordinates from OpenSim to MuJoCo conventions
- Computes inverse dynamics using MJX with GPU acceleration
- Calculates GRF contributions to joint torques via Jacobian transpose method
- Saves results (`ID_Results_MJX.npy`, `anklePos.npy`, etc.) to each trial's `calculatedInputs/` folder

---

### 2. `MJX_RunID.py` (1692 lines)
**Purpose:** Single-trial inverse dynamics script with visualization and interactive viewer.

- Runs inverse dynamics on a single patient dataset (e.g., `PatientData/Falisse_2017_subject_01/`)
- Includes MuJoCo passive viewer for visualizing the motion with GRF arrows
- Generates comparison plots between computed torques and reference data (`tau.csv`)
- Contains plotting functions for debugging (acceleration, position, velocity comparisons)

---
### RUN THIS TO LOOK AT THE PATIENT DATA. It should be set up to run without too much trouble
### 3. `VisualizeDataStructure.py` (514 lines)
**Purpose:** Visualization tool for inspecting processed trial data.

- Cycles through all subjects/trials and generates 6-panel figures:
  1. Joint positions over time (knee & ankle highlighted)
  2. Inverse dynamics torques (knee & ankle)
  3. 3D ankle trajectories
  4. Ground reaction force magnitude & components
  5. COP X position over time
  6. COP-Ankle distance magnitude & components
- Reads from `Motion/mjx/` and `calculatedInputs/` folders

---

### 4. `updateModel.py` (387 lines)
**Purpose:** Model preparation utility for fixing MuJoCo XML files before simulation.

- Fixes zero/small body masses, inertias, and joint armatures
- Updates mesh geometry paths to subject-specific `Geometry/` folders
- Disables collisions by setting `contype=0` and `conaffinity=0`
- Converts models to MJX format and verifies numerical stability
- Main function: `update_model(xml_path, min_mass, min_inertia, min_armature)`

---

### 5. `COPScan.py` (105 lines)
**Purpose:** Data quality scan for COP and ankle position consistency.

- Scans all trials for COP-ankle distance anomalies
- Identifies trials where COP is >30cm from ankle (indicates bad data)
- Deletes trials with incorrect `anklePos.npy` shape
- Outputs percentage of "bad" COP timesteps per trial

---

### 6. `audit_data_structure.py` (60 lines)
**Purpose:** Audit utility to verify data folder structure.

- Lists all subjects and counts valid trials (those with `Motion/` and `calculatedInputs/` folders)
- Deletes empty `combined/` folders
- Removes subjects with zero valid trials
- Outputs a summary table of subjects and trial counts

---

### 7. `check_ankle_pos_sizes.py` (72 lines)
**Purpose:** Debug utility for checking array shape consistency.

- Scans all trials and reports shapes of `anklePos.npy` and `COP_Cleaned.npy`
- Flags mismatches in timestep counts between files
- Useful for diagnosing data loading issues

---

## Directory Structure

```
Calc_ID_MJX/
├── BatchDataProcessing.py      # Main batch ID computation
├── MJX_RunID.py                 # Single-trial ID with viewer
├── VisualizeDataStructure.py   # Data visualization
├── updateModel.py               # Model XML fixer
├── COPScan.py                   # COP quality scanner
├── audit_data_structure.py     # Data structure auditor
├── check_ankle_pos_sizes.py    # Array shape checker
├── Data_Full_Cleaned/           # Processed subject/trial data
│   └── <SubjectName>/
│       └── <TrialName>/
│           ├── Motion/
│           │   └── mjx/         # MJX-ready motion data
│           └── calculatedInputs/ # ID results
├── PatientData/                 # Raw patient data
├── GeometryWithMus/             # Model geometry files
├── ID_Ultiles/                  # Utility scripts
└── myoconverter/                # OpenSim to MuJoCo converter
```

---

## Typical Workflow

1. **Prepare Model:** Run `updateModel.py` to fix the MuJoCo XML for each subject
2. **Batch Process:** Run `BatchDataProcessing.py` to compute ID for all trials
3. **Visualize:** Run `VisualizeDataStructure.py` to inspect results
4. **Debug (if needed):** Use `audit_data_structure.py`, `COPScan.py`, or `check_ankle_pos_sizes.py`

---

## Dependencies

- `mujoco` and `mujoco.mjx`
- `jax` and `jax.numpy`
- `numpy`, `pandas`, `scipy`
- `matplotlib`
- `tqdm`
