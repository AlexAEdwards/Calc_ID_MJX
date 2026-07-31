import os
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
DATA_DIR = str(PROJECT_ROOT / "TrustedDataSetNoised12Distributed_EdgeHold_OYIncluded")
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
    "UseNoised": True,      # Use *_noised.npy temporal input bundle
    "NoisedGT": True,       # Use selected *_noised.npy files for GT COP
    "UseGRFNormCOP": False, # Predict COP_CalcFrame_GroundAligned_GRFNorm.npy instead of standard COP/height
    "UseOSFiltering": False, # Train on OpenSim-filtered (_OSfilt) inputs/targets (ProcessData --os-filtering)
    "use_GRF_NoFilt": True,  # Train GRF target on GRF_NoFilt_Trimmed.npy when available
    "cop_mask": False,           # Apply predicted contact boolean mask to COP/GRF
    "contact_weight": 2.30317,    # BCE weight for predicted contact boolean
    "contact_weight_multiplier": 1,  # Stance-contact upweighting multiplier when contact weighting is enabled
    "use_contact_weighting": False,
    "magOnOff": False,
    "contactOnOff": False,
    "magWeight": 1.0,  # Scaling factor for torque-loss magnitude weighting
    "cop_weight": 0.937189,
    "grf_weight": 6.15879,
    "moments_weight": 0.02,
    "torque_weight": 2.75091,
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
    "BestModelByTorque": True,   # If True, best model chosen by weighted torque RMSE, not val loss
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
EXPERIMENT_NAME = "BestHPO_057432_trial0144_dm384nl4_SmallJacobian" # Optional custom name tag

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

def run_single_training():
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
        "python", train_script,
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
        "--UseNoised", str(CONFIG["UseNoised"]),
        "--NoisedGT", str(CONFIG["NoisedGT"]),
        "--UseGRFNormCOP", str(CONFIG.get("UseGRFNormCOP", False)),
        "--UseOSFiltering", str(CONFIG.get("UseOSFiltering", False)),
        "--use_GRF_NoFilt", str(CONFIG.get("use_GRF_NoFilt", None)),
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

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--UseNoised", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument("--NoisedGT", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument("--UseGRFNormCOP", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument("--use_GRF_NoFilt", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument("--cop_mask", type=lambda x: (str(x).lower() != 'false'), default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--prediction_margin_frames", type=int, default=None)
    parser.add_argument("--save_model_epochs", type=str, default=None)
    args = parser.parse_args()

    if args.UseNoised is not None:
        CONFIG["UseNoised"] = args.UseNoised
    if args.NoisedGT is not None:
        CONFIG["NoisedGT"] = args.NoisedGT
    if args.UseGRFNormCOP is not None:
        CONFIG["UseGRFNormCOP"] = args.UseGRFNormCOP
    if args.use_GRF_NoFilt is not None:
        CONFIG["use_GRF_NoFilt"] = args.use_GRF_NoFilt
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

    run_single_training()
