import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


# =============================================================================
# USER CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 1) Dataset configuration
DATA_DIR = str(PROJECT_ROOT / "TrustedDataSetNoised12Distributed")

# 2) Refine-Q model/training configuration
# QRefineTransformer uses 16-D position-only inputs (pos_noised). Older 54-D
# checkpoints (pos+vel+acc) cannot be loaded into this architecture.
CONFIG = {
    # Architecture
    "d_model": 16,
    "num_heads": 2,
    "num_layers": 3,
    "ff_dim": 64,
    "dropout_rate": 0,
    # Windowing
    "window_size": 128,
    "stride": 16,
    # Training
    "batch_size": 32,
    "learning_rate": 5e-6,
    "weight_decay": 0,
    "epochs":20,
    # Loss (effective lambda = lambda_reg * reg_loss_weight / pos_loss_weight)
    "lambda_reg": 0,
    "pos_loss_weight": .1,
    "reg_loss_weight": 1,
    "reg_fade_epochs": 0,
    # Split / misc
    "val_subjects": [],  # e.g. ["subject2", "subject3"]
    "seed": 42,
    # Optional resume checkpoint
    "checkpoint": None,
}

# 3) Output configuration
BASE_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "refine_q_single_run")
EXPERIMENT_NAME = "RefineQ_VerySmallModel"


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def run_single_training() -> None:
    run_name = EXPERIMENT_NAME
    output_dir = os.path.join(BASE_OUTPUT_DIR, run_name)

    print("Starting single refine-q training run")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {output_dir}")
    print("Configuration:")
    print(json.dumps(CONFIG, indent=2))

    if os.path.exists(output_dir):
        print(f"Warning: output directory already exists: {output_dir}")
        resp = input("Overwrite? (y/n): ").strip().lower()
        if resp == "y":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        else:
            print("Aborting.")
            return
    else:
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    train_script = str(SCRIPT_DIR / "train_refine_q.py")
    epochs = int(CONFIG.get("epochs", CONFIG.get("num_epochs", 100)))

    cmd = [
        "python",
        train_script,
        "--data_dir",
        DATA_DIR,
        "--output_dir",
        output_dir,
        "--d_model",
        str(CONFIG["d_model"]),
        "--num_heads",
        str(CONFIG["num_heads"]),
        "--num_layers",
        str(CONFIG["num_layers"]),
        "--ff_dim",
        str(CONFIG["ff_dim"]),
        "--dropout_rate",
        str(CONFIG["dropout_rate"]),
        "--window_size",
        str(CONFIG["window_size"]),
        "--stride",
        str(CONFIG["stride"]),
        "--batch_size",
        str(CONFIG["batch_size"]),
        "--learning_rate",
        str(CONFIG["learning_rate"]),
        "--weight_decay",
        str(CONFIG["weight_decay"]),
        "--num_epochs",
        str(epochs),
        "--lambda_reg",
        str(CONFIG["lambda_reg"]),
        "--pos_loss_weight",
        str(CONFIG["pos_loss_weight"]),
        "--reg_loss_weight",
        str(CONFIG["reg_loss_weight"]),
        "--reg_fade_epochs",
        str(CONFIG["reg_fade_epochs"]),
        "--seed",
        str(CONFIG["seed"]),
    ]

    val_subjects = CONFIG.get("val_subjects") or []
    if val_subjects:
        cmd.append("--val_subjects")
        cmd.extend([str(s) for s in val_subjects])

    checkpoint = CONFIG.get("checkpoint")
    if checkpoint:
        cmd.extend(["--checkpoint", str(checkpoint)])

    print(f"\nRunning command:\n{' '.join(cmd)}\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        full_log = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            full_log.append(line)
        process.wait()

        if process.returncode == 0:
            print("\nTraining complete successfully.")
        else:
            print(f"\nTraining failed with return code {process.returncode}")

        with open(os.path.join(output_dir, "training_log.txt"), "w", encoding="utf-8") as f:
            f.writelines(full_log)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as exc:
        print(f"\nAn error occurred: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-run launcher for train_refine_q.py")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--val_subjects", nargs="*", default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lambda_reg", type=float, default=None)
    parser.add_argument("--pos_loss_weight", type=float, default=None)
    parser.add_argument("--reg_loss_weight", type=float, default=None)
    parser.add_argument("--reg_fade_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.data_dir is not None:
        DATA_DIR = args.data_dir
    if args.output_dir is not None:
        BASE_OUTPUT_DIR = args.output_dir
    if args.experiment_name is not None:
        EXPERIMENT_NAME = args.experiment_name
    if args.val_subjects is not None:
        CONFIG["val_subjects"] = list(args.val_subjects)
    if args.checkpoint is not None:
        CONFIG["checkpoint"] = args.checkpoint
    if args.num_epochs is not None:
        CONFIG["epochs"] = int(args.num_epochs)
    if args.learning_rate is not None:
        CONFIG["learning_rate"] = float(args.learning_rate)
    if args.lambda_reg is not None:
        CONFIG["lambda_reg"] = float(args.lambda_reg)
    if args.pos_loss_weight is not None:
        CONFIG["pos_loss_weight"] = float(args.pos_loss_weight)
    if args.reg_loss_weight is not None:
        CONFIG["reg_loss_weight"] = float(args.reg_loss_weight)
    if args.reg_fade_epochs is not None:
        CONFIG["reg_fade_epochs"] = int(args.reg_fade_epochs)
    if args.seed is not None:
        CONFIG["seed"] = int(args.seed)

    run_single_training()
