import os
import sys
import subprocess
import json
import shutil
import re
import argparse
from pathlib import Path

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Get paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 1. Dataset Configuration
# DATA_DIR = str(PROJECT_ROOT / "Datasets_NAS" / "AddBiomechanicsDataset_All_npy" / "TrustedDataSet")  # Path to your training data
# DATA_DIR = str(PROJECT_ROOT / "Datasets_NAS" / "DifferentNoisedDataset"/"TrustedDataSetNoised12DistributedUnFiltered_Trimmed")  # Path to your training data
# Note: this workspace does not currently contain a repo-root TrustedDataSet12Distributed folder.
DATA_DIR = str(PROJECT_ROOT / "TrustedDataSetNoised12Distributed_EdgeHold_AllPatients")
# 2. Model Hyperparameters
CONFIG = {
    # Best HPO result from
    # outputs/local_hpo_202606/local_bayes_sharedval_dataset_arch_20260617_153455
    # (HPO_local_bayes_100_equal_knee.yaml, 224 trials, minimize
    #  val/torque_mae_percent_bilateral_mean):
    #   rank 1: trial_0144_dm384_nl4_lr0.000191462_dr0.158504  metric = 0.57432
    # Chosen over the field because it is also the most robust region: rank 3
    # (trial_0103) is nearly identical (dm384/nl4/ws110/lr~0.00021/dr~0.16), and
    # all top-5 trials trained on TrustedDataSetNoised12Distributed_AllPatients_EstimatedWeights.
    # Loss weights below are this trial's exact values; data/balancing flags match
    # the sweep's fixed base_args (use_GRF_NoFilt=True, gender_balance=True,
    # moments_weight=0.02, equal knee L/R weighting).
    "d_model": 384,
    "num_layers": 4,         # Number of transformer layers
    "window_size": 110,      # Context window size (e.g., 64, 128)
    "stride": 16,            # Window stride used for training/evaluation window extraction
    "prediction_margin_frames": 20,  # Only supervise/evaluate frames [margin : window_size - margin)
    "learning_rate": 0.000191462,
    "dropout_rate": 0.158504,
    "ff_dim": 1536,          # Feed-forward dimension (usually 4x d_model)
    "epochs": 40,           # Number of training epochs
    "save_model_epochs": [5,7,9,10,13,15,18,20,22,25,28,30,40],  # Optional extra checkpoint epochs, e.g. [7, 8, 9]
    "batch_size": 64,       # Batch size increased for faster processing
    "trim_cop": False,       # Enable COP trimming
    "includePelvisEuler": False, # Keep pelvis_tilt/list/rotation in pos_inputs model features
    "includeJacobianInput": True, # Add flattened Jacobian [jacp,jacr] as temporal model features
    # Map predicted COP/GRF -> torque/KAM with MoCap Jacobian+rotation+ankle/knee
    # instead of video/input-source kinematics. Default False (video-path physics).
    "use_gt_jacob_and_rot": False,
    "UseNoised": True,      # Use *_noised.npy temporal input bundle
    "NoisedGT": True,       # Use selected *_noised.npy files for GT COP
    "UseGRFNormCOP": False, # Predict COP_CalcFrame_GroundAligned_GRFNorm.npy instead of standard COP/height
    "UseOSFiltering": False, # Train on OpenSim-filtered (_OSfilt) inputs/targets (ProcessData --os-filtering)
    "use_GRF_NoFilt": True,  # Train GRF target on GRF_NoFilt_Trimmed.npy when available
    # Correct legacy KneeToCOP_Vectors.npy in memory. Disable after reprocessing
    # the dataset with the corrected ProcessData.py.
    "subtractAnkleHeightKneeVecs": False,
    "cop_mask": False,           # Apply predicted contact boolean mask to COP/GRF
    "contact_weight": 2.30317,    # BCE weight for predicted contact boolean
    "contact_weight_multiplier": 1,  # Stance-contact upweighting multiplier when contact weighting is enabled
    "use_contact_weighting": False,
    "magOnOff": False,
    "contactOnOff": False,
    "magWeight": 1.0,  # Scaling factor for torque-loss magnitude weighting
    # No-torque ablation: replace the removed torque-loss contribution with
    # roughly equivalent COP/GRF direct supervision pressure based on the
    # torque-trained epoch-40 scaled train-loss proportions.
    "cop_weight": 0.9371,
    "grf_weight": 6.15879,
    "moments_weight": 0.02,
    "torque_weight": 2.751,
    # In first_step_ratio mode, this is the target scaled_KAM / scaled_TauGRF
    # on the first training batch. 0.2 drifted to ~1.0+ mid-run (Tau improves
    # faster than KAM), so val became KAM-dominated; 0.1 keeps a clear nudge
    # without swamping COP/GRF/torque.
    "Torque_weight_knee_adduction": 0.075,
    "Torque_weight_knee_adduction_mode": "first_step_ratio",  # first_step_ratio or absolute
    "grf_correction_weight": 0, # Physics residue loss: m*a = sum(GRF)
    "output_reg_weight": 0, # L1 regularizer on normalized model outputs
    # HPO swept a single "knee_weight" applied equally to L/R (equal-knee sweep).
    "knee_r_weight": 1.57353,
    "knee_l_weight": 1.57353,
    "ankle_l_weight": 1.0,
    "ankle_r_weight": 1.0,
    "subtalar_r_weight": 1.0,
    "subtalar_l_weight": 1.0,
    # None => train.py defaults these DOFs to 1.0 (the HPO winner left lumbar at 1.0).
    "lumbar_extension_weight": None,
    "lumbar_bending_weight": None,
    "lumbar_rotation_weight": None,
    # Speed/Gender window balancing (loss-weighting)
    # When enabled, each training window gets a scalar weight folded into the
    # supervision mask so under-represented groups contribute more to the loss.
    # Weights are computed once at startup from per-trial walking speed + gender,
    # and a histogram/JSON report is written to the output dir. Disabled => all
    # windows weighted 1.0 (original behavior); validation is never weighted.
    "balance_speed_gender": False,   # Master switch for speed+gender loss balancing
    "gender_balance": True,          # Equalize male vs female window mass (unknown sex => neutral 1.0)
    "speed_bin_width": 0.1,        # Walking-speed histogram bin width (m/s) for speed balancing
    "speed_weight_power": .5,      # Speed up-weight exponent: w ∝ (1/bin_count)^power (0=off, 0.5=softened, 1=full inverse-freq)
    "weight_clip_ratio": 3.0,       # Clip combined per-window weight to [1/ratio, ratio]; start conservative (3), raise to weight outliers harder
    # Best-model selection
    # Set False so the "best_model_metric" above (joint_moment_mae) drives selection;
    # True would override it and pick by weighted torque RMSE instead.
    "BestModelByTorque": False,   # If True, best model chosen by weighted torque RMSE, not val loss
    "BestModel_TorqueWeighting": {  # Grouped torque-score weights; bilateral groups average R/L, lumbar groups are single-DOF
        "hip_flexion": 1.0,
        "hip_add": 1.0,
        "knee": 1.0,
        "ankle": 3.0,
        "subtalar": 1.0,
        "lumbar_extension": 1.0,
        "lumbar_bending": 1.0,
        "lumbar_rotation": 1.0,
    },
    # WandB (phase 1: config + scalar epoch metrics)
    "use_wandb": True,
    "wandb_project": "Model_Search",  # Set to your project name
    "wandb_entity": None,       # Set to your team/user if needed
    "wandb_group": None,        # Optional grouping key (e.g., experiment family)
    "wandb_tags": "hpo_best,metric_0.57432,trial_0144,dm384_nl4,noised12,equal_knee",
    "wandb_mode": "online",     # "online", "offline", or "disabled"
    "wandb_resume": None,       # "allow", "must", "never", or "auto"
    "wandb_run_id": None,       # Required if you want deterministic resume
    "wandb_api_key": None,      # Optional override; .env is still supported in train.py
    # -------------------------------------------------------------------------
    # NEW FEATURE TOGGLES (all default to prior behavior; flip to opt in)
    # See MODEL_ISSUES_AND_IMPROVEMENT_PLANS.md. Test_New_Features.py sweeps these.
    # -------------------------------------------------------------------------
    # LR schedule (Plan 2): warmup -> cosine decay. no_lr_schedule=True reverts to constant LR.
    "no_lr_schedule": False,        # True => constant LR (original behavior)
    "warmup_frac": 0.03,            # fraction of total steps spent warming up to peak LR
    "end_lr_frac": 0.05,            # final LR as a fraction of peak
    # Robust loss (Plan 4a): 'mse' (original) or 'huber'.
    "robust_loss": "huber",
    "huber_delta": 1.0,             # Huber knee in Z-scored units (only used when robust_loss='huber')
    # Honest normalizer stats (Plan 4b).
    "normalizer_max_batches": 100,  # batches sampled to estimate Z-score normalizers
    "normalizer_stance_only": True,  # True => COP/GRF stats over in-contact frames only
    # Contact-mask skew (Plan 5): 'gt' (original) | 'pred' | 'mixed'.
    "contact_mask_source": "gt",
    "contact_mix_max_alpha": 0.5,   # max blend toward predicted contact at end of training ('mixed')
    # Best-model selection metric: 'val_loss' | 'torque_rmse' | 'moment_mae_bwh' |
    # 'joint_moment_mae' (mean bilateral stance MAE %BW·h across ankle, hip flexion,
    # hip adduction, knee flexion, and knee adduction/KAM).
    # NOTE: BestModelByTorque (below) overrides this when True — set it False to use this.
    "best_model_metric": "joint_moment_mae",
    # FiLM subject conditioning (Plan 7).
    "subject_film": False,
    # Diagnostics/plots (speed): effect_diagnostics adds extra grad probes/step; no_plots disables all plots.
    "effect_diagnostics": False,
    "no_plots": False,
    # Fast-subset cap for ablation sweeps: 0 = all subjects; N>0 = first N subjects only.
    "max_subjects": 0,
    # Window-level random split (fast subset that mixes all subjects). 0 = disabled
    # (subject-level split). >0 = keep that fraction of ALL windows, then split them.
    "window_split_frac": 0.0,     # e.g. 0.5 => use 50% of all windows
    "window_train_frac": 0.7,     # of the sampled windows, fraction for training
    "window_split_seed": 42,
    # Logging: quiet_steps prints only the per-epoch summary (no per-step loss lines).
    "quiet_steps": False,
    # Multi-agent runtime safety
    "scan_workers": 3,          # Lower trial-discovery thread pressure per process
    "jax_multi_agent_safe": False,
    "jax_gpu_mem_fraction": 0.4,  # Per-process GPU memory cap target
    "jax_cpu_threads": 6,          # BLAS/OpenMP threads per process
    "cuda_visible_devices": None,  # e.g. "0" or "1"; None leaves current visibility unchanged
    
}

# 3. Output Configuration
# The script will create a folder with a descriptive name in this directory
BASE_OUTPUT_DIR = str(PROJECT_ROOT / "outputs")  # Base directory for outputs; will create subfolders per run
# EXPERIMENT_NAME = "BestClean" # Optional custom name tag
EXPERIMENT_NAME = "ReprocessedDataSet_July9_TorqueInformed_KAM_Weight_Corrected" # Optional custom name tag

# 4. Trials/subjects to exclude from training
# Entries are matched against each trial's name. Use "Subject/Trial" (e.g.
# "OA19/Trial_5") to drop a single trial, or a bare "Subject" (e.g. "OA19") to
# drop every trial for that subject. Excluded trials are removed from the entire
# run (both train and validation) before the train/val split.
# Pre-populated with the validation outliers from validation_outliers_epoch_0040.png
# of the AllData_WeightedDataSet run.
#
# EXCLUDE_PREFIXES: subject-name prefixes to drop wholesale. Every patient folder
# whose name starts with one of these strings is removed from the entire run
# (train + validation). For example "SUBJ" drops SUBJ12, SUBJ44, SUBJ108, ...;
# add "OA" or "Y" to drop those cohorts too.
EXCLUDE_PREFIXES = [
    "SUBJ",
    "OA",
    "Y",
]
EXCLUDE_FROM_TRAINING = [
    "OA19/Trial_5",
    "OA19/Trial_6",
    "OA18/Trial_3",
    "OA18/Trial_11",
    "OA10/Trial_6",
    "SUBJ12/Trial_1",
    "SUBJ12/Trial_2",
    "SUBJ44/Trial_1",
#   "GaitRetraining_Subject125/Trial_1",
#  "GaitRetraining_Subject138/Trial_4",
#   "GaitRetraining_Subject138/Trial_28",
#   "GaitRetraining_Subject153/Trial_5",
#   "GaitRetraining_SubjectR583/Trial_12",
   "04/Trial_26",
    "Y21/Trial_14",
    "S11/Trial_1",
    "S11/Trial_2",
    # Validation outliers from outputs/EstimatedWeights_AllPatients_Noised/
    # validation_outliers_epoch_*.png, excluding trials that only appeared in
    # Moment Rz/Lz panels.
    "OA19/Trial_9",
    "OA19/Trial_10",
    "OA19/Trial_11",
    "OA19/Trial_12",
    "OA19/Trial_13",
    "SUBJ48/Trial_1",
#   "GaitRetraining_SubjectR892/Trial_3",
    "Y10/Trial_11",
    "Y10/Trial_12",
    "Y10/Trial_13",
    "Y10/Trial_14",
    "Y10/Trial_15",
    "Y18/Trial_12",
    "Y18/Trial_14",
    "Y18/Trial_15",
    "Y21/Trial_10",
 #  "GaitRetraining_Subject142/Trial_18",
  # "GaitRetraining_Subject108/Trial_1",
#   "GaitRetraining_Subject138/Trial_9",
#   "GaitRetraining_Subject138/Trial_14",
#   "GaitRetraining_Subject138/Trial_19",
#   "GaitRetraining_Subject138/Trial_23",
#   "GaitRetraining_Subject140/Trial_11",
#   "GaitRetraining_Subject140/Trial_25",
#   "GaitRetraining_Subject147/Trial_19",
#   "GaitRetraining_Subject153/Trial_7",
#   "GaitRetraining_Subject153/Trial_11",
#   "GaitRetraining_Subject153/Trial_15",
#   "GaitRetraining_Subject153/Trial_21",
#   "GaitRetraining_Subject153/Trial_23",
#   "GaitRetraining_SubjectR618/Trial_1",
#   "GaitRetraining_SubjectR618/Trial_3",
#   "GaitRetraining_SubjectR618/Trial_6",
#   "GaitRetraining_SubjectR618/Trial_11",
#   "GaitRetraining_SubjectR618/Trial_12",
#   "GaitRetraining_SubjectR728/Trial_1",
#   "GaitRetraining_SubjectR728/Trial_4",
#   "GaitRetraining_SubjectR728/Trial_5",
#   "GaitRetraining_SubjectR728/Trial_21",
#   "GaitRetraining_SubjectR1218/Trial_4",
#   "GaitRetraining_SubjectR1218/Trial_11",
#   "GaitRetraining_SubjectR1218/Trial_14",
#   "GaitRetraining_SubjectR1218/Trial_18",
#   "GaitRetraining_SubjectR1218/Trial_25",
#   "GaitRetraining_SubjectR1218/Trial_30",
#   "GaitRetraining_SubjectR1389/Trial_1",
#   "GaitRetraining_SubjectR1389/Trial_2",
#   "GaitRetraining_SubjectR1389/Trial_3",
#   "GaitRetraining_SubjectR1389/Trial_16",
    # Validation outliers from outputs/EstimatedWeights_AllPatients_NoisedV2/
    # validation_outliers_epoch_*.png, excluding trials that only appeared in
    # Moment Rz/Lz panels.
#   "GaitRetraining_Subject108/Trial_3",
#   "GaitRetraining_Subject136/Trial_6",
#   "GaitRetraining_Subject139/Trial_17",
#   "GaitRetraining_Subject141/Trial_1",
#   "GaitRetraining_Subject155/Trial_1",
#   "GaitRetraining_Subject155/Trial_13",
#   "GaitRetraining_Subject155/Trial_15",
#   "GaitRetraining_Subject155/Trial_16",
#   "GaitRetraining_SubjectR1246/Trial_1",
#   "GaitRetraining_SubjectR1246/Trial_5",
#   "GaitRetraining_SubjectR1246/Trial_8",
#   "GaitRetraining_SubjectR1246/Trial_13",
#   "GaitRetraining_SubjectR1246/Trial_15",
#   "GaitRetraining_SubjectR1246/Trial_17",
#   "GaitRetraining_SubjectR1246/Trial_19",
#   "GaitRetraining_SubjectR1246/Trial_21",
#   "GaitRetraining_SubjectR140/Trial_6",
#   "GaitRetraining_SubjectR140/Trial_8",
#   "GaitRetraining_SubjectR140/Trial_22",
#   "GaitRetraining_SubjectR635/Trial_12",
#   "GaitRetraining_SubjectR635/Trial_25",
#   "GaitRetraining_SubjectR729/Trial_1",
    "OA11/Trial_5",
    "OA11/Trial_6",
    "OA11/Trial_8",
    "OA19/Trial_4",
    "OA19/Trial_7",
    "OA19/Trial_8",
    "SUBJ108/Trial_1",
    "SUBJ108/Trial_2",
    "SUBJ16/Trial_2",
    "SUBJ16/Trial_3",
    "Y11/Trial_11",
    "Y11/Trial_12",
    "Y21/Trial_11",
    "Y21/Trial_12",
    "Y21/Trial_13",
]
# =============================================================================
# SCRIPT LOGIC
# =============================================================================

def run_single_training(force=False):
    # Use the experiment name directly for the run folder.
    run_name = EXPERIMENT_NAME
    output_dir = os.path.join(BASE_OUTPUT_DIR, run_name)
    
    print(f"🚀 Starting Single Model Training")
    print(f"📂 Data Directory: {DATA_DIR}")
    print(f"📂 Output Directory: {output_dir}")
    print(f"⚙️  Configuration:")
    print(json.dumps(CONFIG, indent=2))
    
    # Create output directory
    if os.path.exists(output_dir):
        print(f"⚠️  Warning: Output directory {output_dir} already exists.")
        if force:
            print("    --force set: overwriting.")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            resp = input("    Overwrite? (y/n): ")
            if resp.lower() == 'y':
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
            else:
                print("    Aborting.")
                return
    else:
        os.makedirs(output_dir)

    # Save a bootstrap hyperparameters file before training starts.
    bootstrap_hyperparams = dict(CONFIG)
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(bootstrap_hyperparams, f, indent=2)

    # Construct command - use absolute path to train.py
    train_script = str(SCRIPT_DIR / "train.py")
    cmd = [
        sys.executable, train_script,
        "--data_dir", DATA_DIR,
        "--output_dir", output_dir,
        "--exp_name", run_name,
        "--epochs", str(CONFIG["epochs"]),
        "--d_model", str(CONFIG["d_model"]),
        "--num_layers", str(CONFIG["num_layers"]),
        "--window_size", str(CONFIG["window_size"]),
        "--stride", str(CONFIG["stride"]),
        "--prediction_margin_frames", str(CONFIG["prediction_margin_frames"]),
        "--learning_rate", str(CONFIG["learning_rate"]),
        "--dropout_rate", str(CONFIG.get("dropout_rate", 0.1)),
        "--weight_decay", str(CONFIG.get("weight_decay", 0.001)),
        "--ff_dim", str(CONFIG["ff_dim"]),
        "--batch_size", str(CONFIG["batch_size"]),
        "--log_interval", "10",
        "--vis_interval", "1", # Visualize every 10 epochs
        "--refresh_cache",   # Always scan dataset for changes
        "--trim_cop", str(CONFIG["trim_cop"]),
        "--includePelvisEuler", str(CONFIG.get("includePelvisEuler", True)),
        "--includeJacobianInput", str(CONFIG.get("includeJacobianInput", True)),
        "--useGTJacobAndRot", str(CONFIG.get("use_gt_jacob_and_rot", False)),
        "--UseNoised", str(CONFIG["UseNoised"]),
        "--NoisedGT", str(CONFIG["NoisedGT"]),
        "--UseGRFNormCOP", str(CONFIG.get("UseGRFNormCOP", False)),
        "--UseOSFiltering", str(CONFIG.get("UseOSFiltering", False)),
        "--use_GRF_NoFilt", str(CONFIG.get("use_GRF_NoFilt", None)),
        "--subtractAnkleHeightKneeVecs", str(
            CONFIG.get("subtractAnkleHeightKneeVecs", False)
        ),
        "--cop_mask", str(CONFIG.get("cop_mask", True)),
        "--contact_weight", str(CONFIG.get("contact_weight", 1.0)),
        "--contact_weight_multiplier", str(CONFIG.get("contact_weight_multiplier", 1.5)),
        "--use_contact_weighting", str(CONFIG.get("use_contact_weighting", False)),
        "--magOnOff", str(CONFIG.get("magOnOff", False)),
        "--contactOnOff", str(CONFIG.get("contactOnOff", False)),
        "--magWeight", str(CONFIG.get("magWeight", 3.0)),
    ]

    if CONFIG.get("save_model_epochs"):
        save_model_epochs_arg = ",".join(str(int(epoch)) for epoch in CONFIG["save_model_epochs"])
        cmd.extend(["--save_model_epochs", save_model_epochs_arg])

    if CONFIG.get("scan_workers") is not None:
        cmd.extend(["--scan_workers", str(CONFIG["scan_workers"])])

    cmd.extend([
        "--cop_weight", str(CONFIG.get("cop_weight", 1.0)),
        "--grf_weight", str(CONFIG.get("grf_weight", 1.0)),
        "--moments_weight", str(CONFIG.get("moments_weight", 0.25)),
        "--torque_weight", str(CONFIG.get("torque_weight", 2.0)),
        "--Torque_weight_knee_adduction", str(CONFIG.get("Torque_weight_knee_adduction", 0.0)),
        "--Torque_weight_knee_adduction_mode", str(CONFIG.get("Torque_weight_knee_adduction_mode", "first_step_ratio")),
        "--grf_correction_weight", str(CONFIG.get("grf_correction_weight", 500.0)),
        "--output_reg_weight", str(CONFIG.get("output_reg_weight", 0.0)),
        "--BestModelByTorque", str(CONFIG.get("BestModelByTorque", False)),
        "--BestModel_TorqueWeighting", json.dumps(CONFIG.get("BestModel_TorqueWeighting", {})),
        # Speed/gender window balancing
        "--balance_speed_gender", str(CONFIG.get("balance_speed_gender", False)),
        "--gender_balance", str(CONFIG.get("gender_balance", True)),
        "--speed_bin_width", str(CONFIG.get("speed_bin_width", 0.05)),
        "--speed_weight_power", str(CONFIG.get("speed_weight_power", 0.5)),
        "--weight_clip_ratio", str(CONFIG.get("weight_clip_ratio", 3.0)),
    ])

    # New feature toggles (see CONFIG block above).
    cmd.extend([
        "--warmup_frac", str(CONFIG.get("warmup_frac", 0.03)),
        "--end_lr_frac", str(CONFIG.get("end_lr_frac", 0.05)),
        "--robust_loss", str(CONFIG.get("robust_loss", "mse")),
        "--huber_delta", str(CONFIG.get("huber_delta", 1.0)),
        "--normalizer_max_batches", str(CONFIG.get("normalizer_max_batches", 100)),
        "--normalizer_stance_only", str(CONFIG.get("normalizer_stance_only", False)),
        "--contact_mask_source", str(CONFIG.get("contact_mask_source", "gt")),
        "--contact_mix_max_alpha", str(CONFIG.get("contact_mix_max_alpha", 0.5)),
        "--best_model_metric", str(CONFIG.get("best_model_metric", "val_loss")),
        "--subject_film", str(CONFIG.get("subject_film", False)),
        "--effect_diagnostics", str(CONFIG.get("effect_diagnostics", False)),
    ])
    if CONFIG.get("no_lr_schedule", False):
        cmd.append("--no_lr_schedule")
    if CONFIG.get("no_plots", False):
        cmd.append("--no_plots")
    if CONFIG.get("max_subjects", 0):
        cmd.extend(["--max_subjects", str(int(CONFIG["max_subjects"]))])
    if CONFIG.get("window_split_frac", 0.0):
        cmd.extend([
            "--window_split_frac", str(CONFIG["window_split_frac"]),
            "--window_train_frac", str(CONFIG.get("window_train_frac", 0.7)),
            "--window_split_seed", str(int(CONFIG.get("window_split_seed", 42))),
        ])
    if CONFIG.get("quiet_steps", False):
        cmd.append("--quiet_steps")

    if EXCLUDE_FROM_TRAINING:
        cmd.extend(["--exclude_trials", json.dumps(EXCLUDE_FROM_TRAINING)])

    if EXCLUDE_PREFIXES:
        cmd.extend(["--exclude_prefixes", json.dumps(EXCLUDE_PREFIXES)])

    dof_weight_arg_names = [
        "hip_add_r_weight",
        "knee_r_weight",
        "ankle_r_weight",
        "subtalar_r_weight",
        "hip_add_l_weight",
        "knee_l_weight",
        "ankle_l_weight",
        "subtalar_l_weight",
        "lumbar_extension_weight",
        "lumbar_bending_weight",
        "lumbar_rotation_weight",
    ]
    for arg_name in dof_weight_arg_names:
        if CONFIG.get(arg_name) is not None:
            cmd.extend([f"--{arg_name}", str(CONFIG[arg_name])])

    if CONFIG.get("use_wandb", False):
        cmd.append("--use_wandb")
        cmd.extend(["--wandb_project", str(CONFIG.get("wandb_project", "gait-dynamics-jax"))])
        if CONFIG.get("wandb_entity"):
            cmd.extend(["--wandb_entity", str(CONFIG["wandb_entity"])])
        if CONFIG.get("wandb_group"):
            cmd.extend(["--wandb_group", str(CONFIG["wandb_group"])])
        if CONFIG.get("wandb_tags"):
            cmd.extend(["--wandb_tags", str(CONFIG["wandb_tags"])])
        if CONFIG.get("wandb_mode"):
            cmd.extend(["--wandb_mode", str(CONFIG["wandb_mode"])])
        if CONFIG.get("wandb_resume"):
            cmd.extend(["--wandb_resume", str(CONFIG["wandb_resume"])])
        if CONFIG.get("wandb_run_id"):
            cmd.extend(["--wandb_run_id", str(CONFIG["wandb_run_id"])])
        if CONFIG.get("wandb_api_key"):
            cmd.extend(["--wandb_api_key", str(CONFIG["wandb_api_key"])])
    
    print(f"\n🏃 Running command:\n{' '.join(cmd)}\n")
    
    # Prepare environment variables to allow multiple runs
    env = os.environ.copy()
    if CONFIG.get("jax_multi_agent_safe", True):
        env["JAX_MULTI_AGENT_SAFE"] = "true"
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        gpu_fraction = CONFIG.get("jax_gpu_mem_fraction", 0.35)
        if gpu_fraction is not None:
            env["JAX_GPU_MEM_FRACTION"] = str(gpu_fraction)
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(gpu_fraction))
        cpu_threads = int(CONFIG.get("jax_cpu_threads", 1))
        env["JAX_CPU_THREADS"] = str(cpu_threads)
        for thread_var in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]:
            env.setdefault(thread_var, str(cpu_threads))
    if CONFIG.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(CONFIG["cuda_visible_devices"])

    try:
        # Run training and stream output to console
        # We use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding="utf-8",
            errors="replace",
            bufsize=1, 
            universal_newlines=True,
            env=env
        )
        
        # Capture output for log file
        full_log = []
        
        for line in process.stdout:
            print(line, end='') # Print to console
            full_log.append(line)
            
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ Training complete successfully!")
        else:
            print(f"\n❌ Training failed with return code {process.returncode}")

        # Save full log
        with open(os.path.join(output_dir, "training_log.txt"), "w", encoding="utf-8", errors="replace") as f:
            f.writelines(full_log)

        return int(process.returncode)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        return 1

if __name__ == "__main__":
    def _b(x):
        return str(x).lower() == 'true'

    parser = argparse.ArgumentParser()
    # Existing overrides
    parser.add_argument("--UseNoised", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument("--NoisedGT", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument("--UseGRFNormCOP", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument("--use_GRF_NoFilt", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument(
        "--subtractAnkleHeightKneeVecs",
        nargs="?",
        const=True,
        default=None,
        type=_b,
        help=(
            "Correct legacy KneeToCOP vectors in memory by subtracting ankle world-Z. "
            "Do not enable after reprocessing with the corrected ProcessData.py."
        ),
    )
    parser.add_argument(
        "--useGTJacobAndRot",
        "--use_gt_jacob_and_rot",
        dest="use_gt_jacob_and_rot",
        type=_b,
        default=None,
        help="Use MoCap Jacobian/rotation/ankle-knee for torque+KAM reconstruction.",
    )
    parser.add_argument("--cop_mask", type=lambda x: (str(x).lower() != 'false'), default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--prediction_margin_frames", type=int, default=None)
    parser.add_argument("--save_model_epochs", type=str, default=None)
    # Run-control overrides (used by Test_New_Features.py)
    parser.add_argument("--exp_name", type=str, default=None, help="Override EXPERIMENT_NAME (run folder).")
    parser.add_argument("--data_dir", type=str, default=None, help="Override DATA_DIR.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_subjects", type=int, default=None, help="Fast-subset cap; 0 = all subjects.")
    parser.add_argument("--window_split_frac", type=float, default=None,
                        help="Window-level split: fraction of ALL windows to use (0 = subject split).")
    parser.add_argument("--window_train_frac", type=float, default=None,
                        help="Of the sampled windows, fraction used for training (rest = validation).")
    parser.add_argument("--window_split_seed", type=int, default=None)
    parser.add_argument("--quiet_steps", type=_b, default=None,
                        help="Only print the per-epoch summary (no per-step loss lines).")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB for this run.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output dir without prompting.")
    # New feature toggles (Plans 1-7)
    parser.add_argument("--no_lr_schedule", type=_b, default=None)
    parser.add_argument("--warmup_frac", type=float, default=None)
    parser.add_argument("--end_lr_frac", type=float, default=None)
    parser.add_argument("--robust_loss", type=str, default=None, choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=None)
    parser.add_argument("--normalizer_max_batches", type=int, default=None)
    parser.add_argument("--normalizer_stance_only", type=_b, default=None)
    parser.add_argument("--contact_mask_source", type=str, default=None, choices=["gt", "pred", "mixed"])
    parser.add_argument("--contact_mix_max_alpha", type=float, default=None)
    parser.add_argument("--best_model_metric", type=str, default=None,
                        choices=["val_loss", "torque_rmse", "moment_mae_bwh", "joint_moment_mae"])
    parser.add_argument("--BestModelByTorque", type=_b, default=None,
                        help="Legacy torque-RMSE selection. When True it overrides --best_model_metric.")
    parser.add_argument("--subject_film", type=_b, default=None)
    parser.add_argument("--effect_diagnostics", type=_b, default=None)
    parser.add_argument("--no_plots", type=_b, default=None)
    args = parser.parse_args()

    if args.UseNoised is not None:
        CONFIG["UseNoised"] = args.UseNoised
    if args.NoisedGT is not None:
        CONFIG["NoisedGT"] = args.NoisedGT
    if args.UseGRFNormCOP is not None:
        CONFIG["UseGRFNormCOP"] = args.UseGRFNormCOP
    if args.use_GRF_NoFilt is not None:
        CONFIG["use_GRF_NoFilt"] = args.use_GRF_NoFilt
    if args.subtractAnkleHeightKneeVecs is not None:
        CONFIG["subtractAnkleHeightKneeVecs"] = args.subtractAnkleHeightKneeVecs
    if args.use_gt_jacob_and_rot is not None:
        CONFIG["use_gt_jacob_and_rot"] = args.use_gt_jacob_and_rot
    if args.cop_mask is not None:
        CONFIG["cop_mask"] = args.cop_mask
    if args.stride is not None:
        CONFIG["stride"] = args.stride
    if args.prediction_margin_frames is not None:
        CONFIG["prediction_margin_frames"] = args.prediction_margin_frames
    if args.save_model_epochs is not None:
        raw_text = args.save_model_epochs.strip()
        if raw_text[:1] in "([{" and raw_text[-1:] in ")]}":
            raw_text = raw_text[1:-1].strip()
        CONFIG["save_model_epochs"] = [int(part.strip()) for part in raw_text.split(",") if part.strip()]

    # Run-control overrides
    if args.exp_name is not None:
        EXPERIMENT_NAME = args.exp_name
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
    if args.batch_size is not None:
        CONFIG["batch_size"] = args.batch_size
    if args.max_subjects is not None:
        CONFIG["max_subjects"] = args.max_subjects
    if args.window_split_frac is not None:
        CONFIG["window_split_frac"] = args.window_split_frac
    if args.window_train_frac is not None:
        CONFIG["window_train_frac"] = args.window_train_frac
    if args.window_split_seed is not None:
        CONFIG["window_split_seed"] = args.window_split_seed
    if args.quiet_steps is not None:
        CONFIG["quiet_steps"] = args.quiet_steps
    if args.no_wandb:
        CONFIG["use_wandb"] = False
    if args.BestModelByTorque is not None:
        CONFIG["BestModelByTorque"] = args.BestModelByTorque

    # Feature-toggle overrides
    for _key in ("no_lr_schedule", "warmup_frac", "end_lr_frac", "robust_loss", "huber_delta",
                 "normalizer_max_batches", "normalizer_stance_only", "contact_mask_source",
                 "contact_mix_max_alpha", "best_model_metric", "subject_film", "effect_diagnostics",
                 "no_plots"):
        _val = getattr(args, _key)
        if _val is not None:
            CONFIG[_key] = _val

    _rc = run_single_training(force=args.force)
    sys.exit(int(_rc) if _rc is not None else 0)
