import argparse
from collections import deque
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path


# =============================================================================
# USER CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 1. Dataset Configuration
DATA_DIR = str(PROJECT_ROOT / "TrustedDataSetNoised12Distributed")

# 2. Model Hyperparameters
CONFIG = {
    "d_model": 128,
    "num_layers": 5,
    "window_size": 125,
    "stride": 24,
    "prediction_margin_frames": 15,
    "learning_rate": 0.000184,
    "weight_decay": 0.001,
    "dropout_rate": 0.32,
    "ff_dim": 512,
    "epochs": 22,
    "warmup_epochs": 20,
    "batch_size": 32,
    "UseCNN": False,
    "CNN_Num_Layers": 2,
    "CNN_Kernel_Sizes": "3,5",
    "UseNoised": True,
    "includePelvisEuler": True,
    "PredictJacobian": False,
    "DeviationLearning": True,
    "cop_mask": True,
    "use_contact_weighting": False,
    "contact_weight_multiplier": 1.5,
    "magOnOff": False,
    "contactOnOff": False,
    "magWeight": 3.0,
    "cop_weight": 6.2,
    "grf_weight": 60.0,
    "moments_weight": 0.084,
    "contact_weight": 10.0,
    "torque_weight": 0.0,
    "grf_correction_weight": 0.0,
    "output_reg_weight": 0.0,
    "qpos_weight": 10.0,
    "qvel_weight": 1.0,
    "qacc_weight": 0.5,
    "qfrc_inverse_weight": 6.0,
    "jacobian_weight": 10,
    # Geodesic supervision on the physics-derived rot_w_to_ga rotation bundle.
    "rotation_weight": 1.5,
    "full_id_weight": 0.0,
    # WandB
    "use_wandb": False,
    "wandb_project": "Model_Search",
    "wandb_entity": None,
    "wandb_group": None,
    "wandb_tags": "phase2,single_model,mod_q",
    "wandb_mode": "online",
    "wandb_run_id": None,
    # Runtime safety
    "jax_multi_agent_safe": False,
    "jax_gpu_mem_fraction": 0.6,
    "jax_cpu_threads": 2,
    "cuda_visible_devices": None,
    "jax_preallocate": False,
    "jax_platform_allocator": True,
    "tf_force_gpu_allow_growth": True,
    "prefetch_batches": 0,
    "auto_backfill_q_mjx": False,
    "check_mjx_gradients": True,
    "jax_compilation_cache_dir": str(PROJECT_ROOT / ".jax_compilation_cache"),
    "full_stage_cache_limit": 2,
    "full_stage_precompile_max_groups": 2,
    "full_stage_compile_ahead_groups": 1,
    "clear_runtime_cache_every": 75,
    "log_kinematic_equiv": True,
    "kinematic_equiv_interval": 10,
    "low_ram_available_gb": 8.0,
    "low_ram_available_frac": 0.10,
}

# 3. Output Configuration
BASE_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "ModQ")
EXPERIMENT_NAME = "Noised12_ModQ_ActuallyTrain"


# =============================================================================
# SCRIPT LOGIC
# =============================================================================

def _build_run_name() -> str:
    run_name = (
        f"{EXPERIMENT_NAME}_D{CONFIG['d_model']}_L{CONFIG['num_layers']}"
        f"_W{CONFIG['window_size']}_FF{CONFIG['ff_dim']}"
    )
    if CONFIG.get("UseCNN", False):
        run_name += "_CNN"
    if CONFIG.get("DeviationLearning", False):
        run_name += "_DEV"
    return run_name


def _prepare_env() -> dict:
    env = os.environ.copy()
    env["JAX_MULTI_AGENT_SAFE"] = "true" if CONFIG.get("jax_multi_agent_safe", False) else "false"
    gpu_fraction = CONFIG.get("jax_gpu_mem_fraction", 0.8)
    if gpu_fraction is not None:
        env["JAX_GPU_MEM_FRACTION"] = str(gpu_fraction)
    cpu_threads = int(CONFIG.get("jax_cpu_threads", 8))
    env["JAX_CPU_THREADS"] = str(cpu_threads)
    env.setdefault("JAX_ENABLE_X64", "false")
    if CONFIG.get("jax_preallocate") is not None:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if CONFIG.get("jax_preallocate") else "false"
    if CONFIG.get("jax_platform_allocator", False):
        env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    if CONFIG.get("tf_force_gpu_allow_growth", False):
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    cache_dir = CONFIG.get("jax_compilation_cache_dir")
    if cache_dir:
        cache_path = Path(str(cache_dir))
        cache_path.mkdir(parents=True, exist_ok=True)
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_path)
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


def _build_command(output_dir: str, run_name: str) -> list[str]:
    train_script = str(SCRIPT_DIR / "train_mod_q.py")
    cmd = [
        sys.executable,
        train_script,
        "--data_dir", DATA_DIR,
        "--output_dir", output_dir,
        "--exp_name", run_name,
        "--epochs", str(CONFIG["epochs"]),
        "--warmup_epochs", str(CONFIG["warmup_epochs"]),
        "--batch_size", str(CONFIG["batch_size"]),
        "--window_size", str(CONFIG["window_size"]),
        "--stride", str(CONFIG["stride"]),
        "--prefetch_batches", str(CONFIG["prefetch_batches"]),
        "--prediction_margin_frames", str(CONFIG["prediction_margin_frames"]),
        "--learning_rate", str(CONFIG["learning_rate"]),
        "--weight_decay", str(CONFIG["weight_decay"]),
        "--d_model", str(CONFIG["d_model"]),
        "--num_layers", str(CONFIG["num_layers"]),
        "--ff_dim", str(CONFIG["ff_dim"]),
        "--dropout_rate", str(CONFIG["dropout_rate"]),
        "--use_cnn", str(CONFIG["UseCNN"]),
        "--cnn_num_layers", str(CONFIG["CNN_Num_Layers"]),
        "--cnn_kernel_sizes", str(CONFIG["CNN_Kernel_Sizes"]),
        "--log_interval", "1",
        "--refresh_cache",
        "--cop_weight", str(CONFIG["cop_weight"]),
        "--grf_weight", str(CONFIG["grf_weight"]),
        "--moments_weight", str(CONFIG["moments_weight"]),
        "--contact_weight", str(CONFIG["contact_weight"]),
        "--torque_weight", str(CONFIG["torque_weight"]),
        "--grf_correction_weight", str(CONFIG["grf_correction_weight"]),
        "--output_reg_weight", str(CONFIG["output_reg_weight"]),
        "--qpos_weight", str(CONFIG["qpos_weight"]),
        "--qvel_weight", str(CONFIG["qvel_weight"]),
        "--qacc_weight", str(CONFIG["qacc_weight"]),
        "--qfrc_inverse_weight", str(CONFIG["qfrc_inverse_weight"]),
        "--jacobian_weight", str(CONFIG["jacobian_weight"]),
        "--rotation_weight", str(CONFIG["rotation_weight"]),
        "--full_id_weight", str(CONFIG["full_id_weight"]),
        "--full_stage_cache_limit", str(CONFIG["full_stage_cache_limit"]),
        "--full_stage_precompile_max_groups", str(CONFIG["full_stage_precompile_max_groups"]),
        "--full_stage_compile_ahead_groups", str(CONFIG["full_stage_compile_ahead_groups"]),
        "--clear_runtime_cache_every", str(CONFIG["clear_runtime_cache_every"]),
        "--log_kinematic_equiv", str(CONFIG["log_kinematic_equiv"]),
        "--kinematic_equiv_interval", str(CONFIG["kinematic_equiv_interval"]),
        "--low_ram_available_gb", str(CONFIG["low_ram_available_gb"]),
        "--low_ram_available_frac", str(CONFIG["low_ram_available_frac"]),
        "--use_contact_weighting", str(CONFIG["use_contact_weighting"]),
        "--contact_weight_multiplier", str(CONFIG["contact_weight_multiplier"]),
        "--magOnOff", str(CONFIG["magOnOff"]),
        "--contactOnOff", str(CONFIG["contactOnOff"]),
        "--magWeight", str(CONFIG["magWeight"]),
        "--cop_mask", str(CONFIG["cop_mask"]),
    ]

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
        if CONFIG.get("wandb_run_id"):
            cmd.extend(["--wandb_run_id", str(CONFIG["wandb_run_id"])])

    if CONFIG.get("check_mjx_gradients", False):
        cmd.append("--check_mjx_gradients")

    return cmd


def _describe_return_code(return_code: int) -> str:
    if return_code >= 0:
        return f"exit code {return_code}"
    sig_num = -return_code
    try:
        sig_name = signal.Signals(sig_num).name
    except ValueError:
        sig_name = f"SIG{sig_num}"
    if sig_num == int(signal.SIGKILL):
        return (
            f"{sig_name} ({return_code}) - the OS or job manager killed the process. "
            "This often indicates out-of-memory pressure."
        )
    return f"{sig_name} ({return_code})"


def _write_failure_summary(
    output_dir: str,
    *,
    return_code: int,
    cmd: list[str],
    config: dict,
    tail_lines: list[str],
) -> None:
    summary = {
        "return_code": int(return_code),
        "return_code_description": _describe_return_code(int(return_code)),
        "command": cmd,
        "config": config,
        "last_log_lines": tail_lines,
    }
    summary_path = os.path.join(output_dir, "failure_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _count_missing_q_mjx(root_dir: str) -> tuple[int, int]:
    root = Path(root_dir)
    processed_count = 0
    missing_count = 0
    for processed_dir in root.rglob("ProcessedData"):
        if not processed_dir.is_dir():
            continue
        processed_count += 1
        missing_clean = not (processed_dir / "qvel_mjx.npy").exists() or not (processed_dir / "qacc_mjx.npy").exists()
        missing_noised = bool(CONFIG.get("UseNoised", True)) and (
            not (processed_dir / "qvel_mjx_noised.npy").exists()
            or not (processed_dir / "qacc_mjx_noised.npy").exists()
        )
        if missing_clean or missing_noised:
            missing_count += 1
    return processed_count, missing_count


def _list_missing_q_mjx(root_dir: str) -> list[str]:
    root = Path(root_dir)
    missing: list[str] = []
    for processed_dir in root.rglob("ProcessedData"):
        if not processed_dir.is_dir():
            continue
        missing_clean = not (processed_dir / "qvel_mjx.npy").exists() or not (processed_dir / "qacc_mjx.npy").exists()
        missing_noised = bool(CONFIG.get("UseNoised", True)) and (
            not (processed_dir / "qvel_mjx_noised.npy").exists()
            or not (processed_dir / "qacc_mjx_noised.npy").exists()
        )
        if missing_clean or missing_noised:
            try:
                label = str(processed_dir.parent.relative_to(root))
            except Exception:
                label = str(processed_dir.parent)
            missing.append(label)
    return sorted(missing)


def _ensure_q_mjx_backfill(env: dict) -> None:
    processed_count, missing_count = _count_missing_q_mjx(DATA_DIR)
    if processed_count == 0:
        print(f"No ProcessedData directories found under {DATA_DIR}")
        return
    if missing_count == 0:
        print("Found qvel_mjx/qacc_mjx bundles for all processed trials.")
        return

    missing_trials = _list_missing_q_mjx(DATA_DIR)
    print(
        f"Warning: {missing_count} of {processed_count} processed trials are missing clean and/or noised qvel_mjx/qacc_mjx bundles. "
        "Those trials may be excluded from training if the required MJX velocity/acceleration templates are unavailable. "
        + (f"Examples: {', '.join(missing_trials[:5])}" if missing_trials else "")
    )


def run_single_training():
    run_name = _build_run_name()
    output_dir = os.path.join(BASE_OUTPUT_DIR, run_name)

    print("Starting Single Model Mod-Q Training")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {output_dir}")
    print("Configuration:")
    print(json.dumps(CONFIG, indent=2))

    if os.path.exists(output_dir):
        print(f"Warning: Output directory {output_dir} already exists.")
        resp = input("Overwrite? (y/n): ")
        if resp.lower() == "y":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            print("Aborting.")
            return
    else:
        os.makedirs(output_dir)

    bootstrap_hyperparams = dict(CONFIG)
    bootstrap_hyperparams["use_cnn"] = bool(CONFIG.get("UseCNN", False))
    bootstrap_hyperparams["model_type"] = "mod_q"
    bootstrap_hyperparams["physics_backend"] = "mjx_jit_differentiable"
    bootstrap_hyperparams["forced_flags"] = {
        "UseNoised": True,
        "includePelvisEuler": True,
        "PredictJacobian": False,
        "DeviationLearning": bool(CONFIG.get("DeviationLearning", False)),
    }
    with open(os.path.join(output_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(bootstrap_hyperparams, f, indent=2)

    cmd = _build_command(output_dir, run_name)
    print(f"\nRunning command:\n{' '.join(cmd)}\n")
    print(
        "Launcher note: logs will be streamed live to training_log.txt, "
        "and failures will write failure_summary.json with the last log lines."
    )
    if CONFIG.get("jax_compilation_cache_dir"):
        print(f"Persistent JAX compilation cache: {CONFIG['jax_compilation_cache_dir']}")

    env = _prepare_env()
    _ensure_q_mjx_backfill(env)
    log_path = os.path.join(output_dir, "training_log.txt")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        last_lines: deque[str] = deque(maxlen=200)
        best_val_loss = float("inf")
        patterns = (
            r"Training complete\. Best val loss: ([\d\.]+)",
            r"Saved new best checkpoint with val loss ([\d\.]+)",
        )
        assert process.stdout is not None
        with open(log_path, "w", encoding="utf-8") as log_file:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
                last_lines.append(line.rstrip("\n"))
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        best_val_loss = float(match.group(1))

        process.wait()

        if process.returncode == 0:
            print("\nTraining complete successfully.")

            if best_val_loss != float("inf"):
                new_dir_name = f"{run_name}_Loss_{best_val_loss:.4f}"
                new_output_dir = os.path.join(BASE_OUTPUT_DIR, new_dir_name)
                if not os.path.exists(new_output_dir):
                    os.rename(output_dir, new_output_dir)
                    print(f"Renamed output folder to: {new_output_dir}")
                    output_dir = new_output_dir
        else:
            description = _describe_return_code(process.returncode)
            print(f"\nTraining failed with return code {process.returncode}")
            print(f"Failure detail: {description}")
            if process.returncode == -int(signal.SIGKILL):
                print(
                    "Most likely cause: the training subprocess was killed externally, "
                    "often by the OS OOM killer during the full MJX stage."
                )
            _write_failure_summary(
                output_dir,
                return_code=process.returncode,
                cmd=cmd,
                config=CONFIG,
                tail_lines=list(last_lines),
            )
            print(f"Saved live log to: {log_path}")
            print(f"Saved failure summary to: {os.path.join(output_dir, 'failure_summary.json')}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--UseCNN", type=lambda x: str(x).lower() != "false", default=None)
    parser.add_argument("--DeviationLearning", type=lambda x: str(x).lower() != "false", default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--prefetch_batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--prediction_margin_frames", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--ff_dim", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--CNN_Num_Layers", type=int, default=None)
    parser.add_argument("--CNN_Kernel_Sizes", type=str, default=None)
    parser.add_argument("--cop_mask", type=lambda x: str(x).lower() != "false", default=None)
    parser.add_argument("--use_contact_weighting", type=lambda x: str(x).lower() != "false", default=None)
    parser.add_argument("--contact_weight_multiplier", type=float, default=None)
    parser.add_argument("--cop_weight", type=float, default=None)
    parser.add_argument("--grf_weight", type=float, default=None)
    parser.add_argument("--moments_weight", type=float, default=None)
    parser.add_argument("--contact_weight", type=float, default=None)
    parser.add_argument("--torque_weight", type=float, default=None)
    parser.add_argument("--grf_correction_weight", type=float, default=None)
    parser.add_argument("--output_reg_weight", type=float, default=None)
    parser.add_argument("--qpos_weight", type=float, default=None)
    parser.add_argument("--qvel_weight", type=float, default=None)
    parser.add_argument("--qacc_weight", type=float, default=None)
    parser.add_argument("--qfrc_inverse_weight", type=float, default=None)
    parser.add_argument("--jacobian_weight", type=float, default=None)
    parser.add_argument("--rotation_weight", type=float, default=None)
    parser.add_argument("--full_id_weight", type=float, default=None)
    parser.add_argument("--full_stage_cache_limit", type=int, default=None)
    parser.add_argument("--full_stage_precompile_max_groups", type=int, default=None)
    parser.add_argument("--full_stage_compile_ahead_groups", type=int, default=None)
    parser.add_argument("--clear_runtime_cache_every", type=int, default=None)
    parser.add_argument("--log_kinematic_equiv", type=lambda x: str(x).lower() != "false", default=None)
    parser.add_argument("--kinematic_equiv_interval", type=int, default=None)
    parser.add_argument("--low_ram_available_gb", type=float, default=None)
    parser.add_argument("--low_ram_available_frac", type=float, default=None)
    parser.add_argument("--jax_gpu_mem_fraction", type=float, default=None)
    parser.add_argument("--jax_cpu_threads", type=int, default=None)
    parser.add_argument("--jax_preallocate", type=lambda x: str(x).lower() != "false", default=None)
    parser.add_argument("--jax_platform_allocator", type=lambda x: str(x).lower() != "false", default=None)
    parser.add_argument("--tf_force_gpu_allow_growth", type=lambda x: str(x).lower() != "false", default=None)
    parser.add_argument("--auto_backfill_q_mjx", type=lambda x: str(x).lower() != "false", default=None)
    args = parser.parse_args()

    for key in (
        "UseCNN",
        "DeviationLearning",
        "stride",
        "window_size",
        "batch_size",
        "prefetch_batches",
        "epochs",
        "warmup_epochs",
        "prediction_margin_frames",
        "learning_rate",
        "weight_decay",
        "dropout_rate",
        "ff_dim",
        "d_model",
        "num_layers",
        "CNN_Num_Layers",
        "CNN_Kernel_Sizes",
        "cop_mask",
        "use_contact_weighting",
        "contact_weight_multiplier",
        "cop_weight",
        "grf_weight",
        "moments_weight",
        "contact_weight",
        "torque_weight",
        "grf_correction_weight",
        "output_reg_weight",
        "qpos_weight",
        "qvel_weight",
        "qacc_weight",
        "qfrc_inverse_weight",
        "jacobian_weight",
        "rotation_weight",
        "full_id_weight",
        "full_stage_cache_limit",
        "full_stage_precompile_max_groups",
        "full_stage_compile_ahead_groups",
        "clear_runtime_cache_every",
        "log_kinematic_equiv",
        "kinematic_equiv_interval",
        "low_ram_available_gb",
        "low_ram_available_frac",
        "jax_gpu_mem_fraction",
        "jax_cpu_threads",
        "jax_preallocate",
        "jax_platform_allocator",
        "tf_force_gpu_allow_growth",
        "auto_backfill_q_mjx",
    ):
        value = getattr(args, key)
        if value is not None:
            CONFIG[key] = value

    run_single_training()
