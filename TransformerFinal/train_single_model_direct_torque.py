import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from paths import artifact, dataset  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# =============================================================================
# USER CONFIGURATION
# =============================================================================
#
# Edit this dictionary for normal direct-torque training runs.  These defaults
# mirror the current train_single_model.py setup where it applies to the direct
# torque architecture.  COP/GRF loss weights are intentionally absent because
# train_directTorque.py supervises only the 14 direct torque outputs.
CONFIG = {
    # Paths / run identity
    "data_dir": str(dataset("TrustedDataSetNoised12Distributed_EdgeHold_AllPatients")),
    "base_output_dir": str(artifact("outputs")),
    "experiment_name": "DirectTorque_ReprocessedDataset_July8th",

    # Model architecture
    "model_structure": "direct_torque",
    "model_type": "direct_torque",
    "d_model": 384,
    "num_layers": 4,
    "window_size": 110,
    "stride": 16,
    "prediction_margin_frames": 20,
    "learning_rate": 0.000191462,
    "dropout_rate": 0.158504,
    "ff_dim": 1536,
    "epochs": 40,
    "save_model_epochs": [5, 7, 9, 10, 13, 15, 18, 20, 22, 25, 28, 30, 40],
    "batch_size": 64,
    "weight_decay": 0.001,

    # Input bundle / targets
    "includePelvisEuler": False,
    "includeJacobianInput": True,
    "UseNoised": True,
    "NoisedGT": True,
    "UseGRFNormCOP": False,
    "UseOSFiltering": False,
    "use_GRF_NoFilt": True,

    # Training behavior
    "robust_loss": "huber",
    "huber_delta": 1.0,
    "normalizer_max_batches": 100,
    "best_model_metric": "direct_torque_mae_bwh",
    "no_lr_schedule": False,
    "no_plots": False,
    "quiet_steps": True,
    "max_subjects": 0,
    "scan_workers": 3,

    # Runtime safety
    "jax_multi_agent_safe": False,
    "jax_gpu_mem_fraction": 0.4,
    "jax_cpu_threads": 6,
    "cuda_visible_devices": None,

    # WandB bookkeeping only. train_directTorque.py does not currently log to
    # WandB, but these values are saved into hyperparameters.json for traceability.
    "use_wandb": True,
    "wandb_project": "Model_Search",
    "wandb_entity": None,
    "wandb_group": None,
    "wandb_tags": "direct_torque,hpo_best,metric_0.57432,trial_0144,dm384_nl4,noised12,equal_knee",
    "wandb_mode": "online",
    "wandb_resume": None,
    "wandb_run_id": None,
    "wandb_api_key": None,

    # Exclusions copied from the current train_single_model.py defaults.
    "exclude_prefixes": [
        "SUBJ",
        "OA",
        "Y",
    ],
    "exclude_trials": [
        "OA19/Trial_5",
        "OA19/Trial_6",
        "OA18/Trial_3",
        "OA18/Trial_11",
        "OA10/Trial_6",
        "SUBJ12/Trial_1",
        "SUBJ12/Trial_2",
        "SUBJ44/Trial_1",
        "04/Trial_26",
        "Y21/Trial_14",
        "S11/Trial_1",
        "S11/Trial_2",
        "OA19/Trial_9",
        "OA19/Trial_10",
        "OA19/Trial_11",
        "OA19/Trial_12",
        "OA19/Trial_13",
        "SUBJ48/Trial_1",
        "Y10/Trial_11",
        "Y10/Trial_12",
        "Y10/Trial_13",
        "Y10/Trial_14",
        "Y10/Trial_15",
        "Y18/Trial_12",
        "Y18/Trial_14",
        "Y18/Trial_15",
        "Y21/Trial_10",
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
    ],
}

DATA_DIR = str(CONFIG["data_dir"])
BASE_OUTPUT_DIR = str(CONFIG["base_output_dir"])
EXPERIMENT_NAME = str(CONFIG["experiment_name"])
EXCLUDE_PREFIXES = list(CONFIG.get("exclude_prefixes", []))
EXCLUDE_FROM_TRAINING = list(CONFIG.get("exclude_trials", []))


def _bool_arg(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_epoch_list(text):
    raw_text = str(text).strip()
    if raw_text[:1] in "([{" and raw_text[-1:] in ")]}":
        raw_text = raw_text[1:-1].strip()
    return [int(part.strip()) for part in raw_text.split(",") if part.strip()]


def _apply_run_environment(env):
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
        for thread_var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env.setdefault(thread_var, str(cpu_threads))
    if CONFIG.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(CONFIG["cuda_visible_devices"])
    return env


def run_single_training(force=False):
    run_name = EXPERIMENT_NAME
    output_dir = os.path.join(BASE_OUTPUT_DIR, run_name)

    print("Starting direct-torque single-model training")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {output_dir}")
    if CONFIG.get("quiet_steps", False):
        compact_config = {
            key: CONFIG.get(key)
            for key in (
                "d_model",
                "num_layers",
                "window_size",
                "stride",
                "prediction_margin_frames",
                "learning_rate",
                "dropout_rate",
                "ff_dim",
                "epochs",
                "batch_size",
                "UseNoised",
                "NoisedGT",
                "includePelvisEuler",
                "includeJacobianInput",
                "use_GRF_NoFilt",
                "robust_loss",
                "huber_delta",
                "normalizer_max_batches",
                "model_structure",
            )
        }
        print("Configuration summary:")
        print(json.dumps(compact_config, indent=2, default=str))
    else:
        print("Configuration:")
        print(json.dumps(CONFIG, indent=2, default=str))

    if os.path.exists(output_dir):
        print(f"Warning: Output directory {output_dir} already exists.")
        if force:
            print("  --force set: overwriting.")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            resp = input("  Overwrite? (y/n): ")
            if resp.lower() == "y":
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
            else:
                print("  Aborting.")
                return 0
    else:
        os.makedirs(output_dir)

    bootstrap_hyperparams = dict(CONFIG)
    bootstrap_hyperparams.update(
        {
            "data_dir": DATA_DIR,
            "output_dir": output_dir,
            "exp_name": run_name,
            "exclude_prefixes": EXCLUDE_PREFIXES,
            "exclude_trials": EXCLUDE_FROM_TRAINING,
        }
    )
    with open(os.path.join(output_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(bootstrap_hyperparams, f, indent=2, default=str)

    train_script = str(SCRIPT_DIR / "train_directTorque.py")
    cmd = [
        sys.executable,
        train_script,
        "--data_dir",
        DATA_DIR,
        "--output_dir",
        output_dir,
        "--exp_name",
        run_name,
        "--epochs",
        str(CONFIG["epochs"]),
        "--d_model",
        str(CONFIG["d_model"]),
        "--num_layers",
        str(CONFIG["num_layers"]),
        "--window_size",
        str(CONFIG["window_size"]),
        "--stride",
        str(CONFIG["stride"]),
        "--prediction_margin_frames",
        str(CONFIG["prediction_margin_frames"]),
        "--learning_rate",
        str(CONFIG["learning_rate"]),
        "--dropout_rate",
        str(CONFIG.get("dropout_rate", 0.1)),
        "--weight_decay",
        str(CONFIG.get("weight_decay", 0.001)),
        "--ff_dim",
        str(CONFIG["ff_dim"]),
        "--batch_size",
        str(CONFIG["batch_size"]),
        "--normalizer_max_batches",
        str(CONFIG.get("normalizer_max_batches", 100)),
        "--robust_loss",
        str(CONFIG.get("robust_loss", "huber")),
        "--huber_delta",
        str(CONFIG.get("huber_delta", 1.0)),
        "--UseNoised",
        str(CONFIG.get("UseNoised", True)),
        "--NoisedGT",
        str(CONFIG.get("NoisedGT", True)),
        "--includePelvisEuler",
        str(CONFIG.get("includePelvisEuler", False)),
        "--includeJacobianInput",
        str(CONFIG.get("includeJacobianInput", True)),
        "--use_grf_nofilt",
        str(CONFIG.get("use_GRF_NoFilt", True)),
        "--use_os_filtering",
        str(CONFIG.get("UseOSFiltering", False)),
        "--UseGRFNormCOP",
        str(CONFIG.get("UseGRFNormCOP", False)),
        "--scan_workers",
        str(CONFIG.get("scan_workers", 3)),
        "--exclude_prefixes",
        json.dumps(EXCLUDE_PREFIXES),
        "--exclude_trials",
        json.dumps(EXCLUDE_FROM_TRAINING),
    ]

    if CONFIG.get("save_model_epochs"):
        cmd.extend(
            [
                "--save_model_epochs",
                ",".join(str(int(epoch)) for epoch in CONFIG["save_model_epochs"]),
            ]
        )
    if CONFIG.get("max_subjects", 0):
        cmd.extend(["--max_subjects", str(int(CONFIG["max_subjects"]))])
    if CONFIG.get("no_lr_schedule", False):
        cmd.append("--no_lr_schedule")
    if CONFIG.get("no_plots", False):
        cmd.append("--no_plots")
    if CONFIG.get("quiet_steps", False):
        cmd.append("--quiet_steps")

    command_path = os.path.join(output_dir, "run_command.txt")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")

    if CONFIG.get("quiet_steps", False):
        print(f"\nRunning train_directTorque.py with quiet_steps=True")
        print(f"Full command is recorded in {command_path}.\n")
    else:
        print(f"\nRunning command:\n{' '.join(cmd)}\n")

    env = _apply_run_environment(os.environ.copy())
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        full_log = []
        for line in process.stdout:
            print(line, end="")
            full_log.append(line)
        process.wait()

        if process.returncode == 0:
            print("\nDirect-torque training complete successfully.")
        else:
            print(f"\nDirect-torque training failed with return code {process.returncode}")

        with open(os.path.join(output_dir, "training_log.txt"), "w", encoding="utf-8", errors="replace") as f:
            f.writelines(full_log)
        return int(process.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 130
    except Exception as exc:
        print(f"\nAn error occurred: {exc}")
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper for direct-torque single-model training.")
    parser.add_argument("--exp_name", type=str, default=None, help="Override run folder/name.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override DATA_DIR.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--prediction_margin_frames", type=int, default=None)
    parser.add_argument("--save_model_epochs", type=str, default=None)
    parser.add_argument("--UseNoised", type=_bool_arg, default=None)
    parser.add_argument("--NoisedGT", type=_bool_arg, default=None)
    parser.add_argument("--UseGRFNormCOP", type=_bool_arg, default=None)
    parser.add_argument("--use_GRF_NoFilt", type=_bool_arg, default=None)
    parser.add_argument("--includePelvisEuler", type=_bool_arg, default=None)
    parser.add_argument("--includeJacobianInput", type=_bool_arg, default=None)
    parser.add_argument("--robust_loss", choices=["mse", "huber"], default=None)
    parser.add_argument("--huber_delta", type=float, default=None)
    parser.add_argument("--normalizer_max_batches", type=int, default=None)
    parser.add_argument("--no_lr_schedule", type=_bool_arg, default=None)
    parser.add_argument("--no_plots", type=_bool_arg, default=None)
    parser.add_argument("--quiet_steps", type=_bool_arg, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Stored in config for bookkeeping.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output dir without prompting.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.exp_name is not None:
        EXPERIMENT_NAME = args.exp_name
    if args.data_dir is not None:
        DATA_DIR = args.data_dir

    override_map = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_subjects": args.max_subjects,
        "stride": args.stride,
        "prediction_margin_frames": args.prediction_margin_frames,
        "UseNoised": args.UseNoised,
        "NoisedGT": args.NoisedGT,
        "UseGRFNormCOP": args.UseGRFNormCOP,
        "use_GRF_NoFilt": args.use_GRF_NoFilt,
        "includePelvisEuler": args.includePelvisEuler,
        "includeJacobianInput": args.includeJacobianInput,
        "robust_loss": args.robust_loss,
        "huber_delta": args.huber_delta,
        "normalizer_max_batches": args.normalizer_max_batches,
        "no_lr_schedule": args.no_lr_schedule,
        "no_plots": args.no_plots,
        "quiet_steps": args.quiet_steps,
    }
    for key, value in override_map.items():
        if value is not None:
            CONFIG[key] = value
    if args.save_model_epochs is not None:
        CONFIG["save_model_epochs"] = _parse_epoch_list(args.save_model_epochs)
    if args.no_wandb:
        CONFIG["use_wandb"] = False

    rc = run_single_training(force=args.force)
    sys.exit(int(rc) if rc is not None else 0)
