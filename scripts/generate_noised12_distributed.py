#!/usr/bin/env python3
"""Generate TrustedDataSetNoised12Distributed-style PD kinematics.

The empirical OpenCap-vs-MoCap error statistics are read from an existing
OCVal_ErrorDistributions summary.  For each DOF, the target noise standard
deviation is

    empirical_error_std / (12 + N(0, 0.35))

using the legacy seed and multisine settings.  Only active
``<dataset>/<subject>/Trial_*/Motion`` folders are considered; nested
``TempRemove`` trials are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NOISE_MODULE_DIR = REPO_ROOT / "NoiseAndPowerAnalOfInputData"
if str(NOISE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(NOISE_MODULE_DIR))

from generate_sine_noise import generate_multisine_noise
from paths import artifact, dataset  # noqa: E402


DEFAULT_DATASET_ROOT = dataset("Datasets_Local") / "PD_Dataset"
DEFAULT_PROFILE_SUMMARY = (
    REPO_ROOT / "NoiseAndPowerAnalOfInputData" / "StrokeDataset_Noised12_summary.json"
)
DEFAULT_SUMMARY_PATH = DEFAULT_DATASET_ROOT / "pd_noised12_distributed_summary.json"
DEFAULT_MANIFEST_PATH = DEFAULT_DATASET_ROOT / "pd_noised12_distributed_manifest.csv"

NUM_DOFS = 23
NUM_WAVES = 100
MIN_FREQUENCY_HZ = 0.1
MAX_FREQUENCY_HZ = 6.0
SYNTHESIS_SAMPLE_RATE_HZ = 100.0
ERROR_SCALER = 12.0
ERROR_SCALER_SPREAD_STD = 0.35
SEED = 42


def active_trial_dirs(dataset_root: Path) -> list[Path]:
    trials: list[Path] = []
    for subject_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        trials.extend(
            sorted(
                path
                for path in subject_dir.iterdir()
                if path.is_dir()
                and path.name.startswith("Trial_")
                and (path / "Motion" / "Pos.npy").is_file()
            )
        )
    return trials


def load_empirical_profile(summary_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("per_dof_summary")
    if not isinstance(rows, list) or len(rows) < NUM_DOFS:
        raise ValueError(
            f"{summary_path} does not contain at least {NUM_DOFS} per_dof_summary rows"
        )
    rows = rows[:NUM_DOFS]
    error_std = np.asarray([float(row["std"]) for row in rows], dtype=np.float64)
    error_mae = np.asarray([float(row["mae"]) for row in rows], dtype=np.float64)
    dof_names = [str(row.get("dof", f"DOF_{idx}")) for idx, row in enumerate(rows)]
    if not np.all(np.isfinite(error_std)) or np.any(error_std < 0):
        raise ValueError("Empirical error standard deviations must be finite and nonnegative")
    if not np.all(np.isfinite(error_mae)) or np.any(error_mae < 0):
        raise ValueError("Empirical error MAEs must be finite and nonnegative")
    return error_std, error_mae, dof_names


def distributed_targets(error_std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + 10_000)
    perturbations = rng.normal(0.0, ERROR_SCALER_SPREAD_STD, NUM_DOFS)
    effective_scalers = ERROR_SCALER + perturbations
    if np.any(effective_scalers <= 0):
        raise RuntimeError("Seeded distributed error scaler produced a nonpositive value")
    return error_std / effective_scalers, effective_scalers, perturbations


def load_kinematic_time(motion_dir: Path, expected_len: int) -> tuple[np.ndarray, str]:
    for filename in ("Time_for_pos.npy", "Time_for_Pos.npy", "Time.npy"):
        path = motion_dir / filename
        if not path.is_file():
            continue
        time_vec = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if (
            time_vec.size == expected_len
            and np.all(np.isfinite(time_vec))
            and (time_vec.size < 2 or np.all(np.diff(time_vec) > 0))
        ):
            return time_vec, filename
    return (
        np.arange(expected_len, dtype=np.float64) / SYNTHESIS_SAMPLE_RATE_HZ,
        "generated_uniform_100Hz",
    )


def differentiate(values: np.ndarray, time_vec: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float64)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, time_vec, axis=0, edge_order=edge_order)


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "status",
        "num_frames",
        "num_dofs",
        "time_source",
        "pos_noised",
        "vel_noised",
        "accel_noised",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--profile-summary",
        type=Path,
        default=DEFAULT_PROFILE_SUMMARY,
        help="Existing OCVal_ErrorDistributions summary containing empirical 23-DOF errors.",
    )
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Pos_noised/Vel_noised/Accel_noised files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the active trial selection without writing arrays.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    profile_summary = args.profile_summary.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not profile_summary.is_file():
        raise FileNotFoundError(f"Empirical profile summary not found: {profile_summary}")

    error_std, error_mae, dof_names = load_empirical_profile(profile_summary)
    target_std, effective_scalers, perturbations = distributed_targets(error_std)
    trials = active_trial_dirs(dataset_root)
    if not trials:
        raise RuntimeError(f"No active Subject/Trial_*/Motion/Pos.npy files under {dataset_root}")

    print(f"Active trials: {len(trials)}")
    print(f"Dataset: {dataset_root}")
    print(f"Empirical profile: {profile_summary}")
    print(
        "Effective DOF error scalers: "
        + ", ".join(f"{value:.6f}" for value in effective_scalers)
    )
    if args.dry_run:
        print("Dry run complete; no files written.")
        return 0

    manifest_rows: list[dict[str, object]] = []
    written = 0
    skipped = 0
    failed = 0
    for trial_index, trial_dir in enumerate(trials):
        motion_dir = trial_dir / "Motion"
        pos_path = motion_dir / "Pos.npy"
        output_paths = {
            "pos": motion_dir / "Pos_noised.npy",
            "vel": motion_dir / "Vel_noised.npy",
            "accel": motion_dir / "Accel_noised.npy",
        }
        relative_trial = str(trial_dir.relative_to(dataset_root))
        row: dict[str, object] = {
            "trial": relative_trial,
            "status": "",
            "num_frames": "",
            "num_dofs": "",
            "time_source": "",
            "pos_noised": str(output_paths["pos"]),
            "vel_noised": str(output_paths["vel"]),
            "accel_noised": str(output_paths["accel"]),
            "note": "",
        }
        try:
            if not args.overwrite and all(path.exists() for path in output_paths.values()):
                row["status"] = "skipped_existing"
                skipped += 1
                manifest_rows.append(row)
                continue

            pos = np.asarray(np.load(pos_path), dtype=np.float64)
            if pos.ndim != 2 or pos.shape[1] != NUM_DOFS:
                raise ValueError(f"expected Pos.npy shape (T, {NUM_DOFS}), got {pos.shape}")
            if not np.all(np.isfinite(pos)):
                raise ValueError("Pos.npy contains nonfinite values")

            pos_noised = pos.copy()
            trial_seed_base = SEED + trial_index * NUM_DOFS
            for dof_idx in range(NUM_DOFS):
                result = generate_multisine_noise(
                    num_waves=NUM_WAVES,
                    min_frequency_hz=MIN_FREQUENCY_HZ,
                    max_frequency_hz=MAX_FREQUENCY_HZ,
                    sample_rate_hz=SYNTHESIS_SAMPLE_RATE_HZ,
                    num_samples=pos.shape[0],
                    amplitude_constant=float(error_mae[dof_idx]),
                    amplitude_std=None,
                    target_std=float(target_std[dof_idx]),
                    seed=trial_seed_base + dof_idx,
                )
                pos_noised[:, dof_idx] += np.asarray(
                    result["noise_signal"], dtype=np.float64
                )

            time_vec, time_source = load_kinematic_time(motion_dir, pos.shape[0])
            vel_noised = differentiate(pos_noised, time_vec)
            accel_noised = differentiate(vel_noised, time_vec)

            np.save(output_paths["pos"], pos_noised)
            np.save(output_paths["vel"], vel_noised)
            np.save(output_paths["accel"], accel_noised)
            row.update(
                {
                    "status": "written",
                    "num_frames": int(pos.shape[0]),
                    "num_dofs": int(pos.shape[1]),
                    "time_source": time_source,
                }
            )
            written += 1
        except Exception as exc:
            row["status"] = "failed"
            row["note"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        manifest_rows.append(row)
        if (trial_index + 1) % 50 == 0 or trial_index + 1 == len(trials):
            print(
                f"[{trial_index + 1}/{len(trials)}] "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    manifest_path = args.manifest_csv.resolve()
    summary_path = args.summary_json.resolve()
    write_manifest(manifest_rows, manifest_path)
    summary_payload = {
        "dataset_root": str(dataset_root),
        "profile_summary": str(profile_summary),
        "active_trial_count": len(trials),
        "tempremove_trials_excluded": True,
        "written": written,
        "skipped_existing": skipped,
        "failed": failed,
        "settings": {
            "num_waves": NUM_WAVES,
            "min_frequency_hz": MIN_FREQUENCY_HZ,
            "max_frequency_hz": MAX_FREQUENCY_HZ,
            "synthesis_sample_rate_hz": SYNTHESIS_SAMPLE_RATE_HZ,
            "error_scaler": ERROR_SCALER,
            "additional_noise_spread_enabled": True,
            "additional_noise_spread_std": ERROR_SCALER_SPREAD_STD,
            "seed": SEED,
            "amplitude_std": None,
            "target_std_formula": (
                "empirical_error_std / "
                "(error_scaler + N(0, additional_noise_spread_std))"
            ),
            "velocity_acceleration_time_source": (
                "Prefer Motion/Time_for_pos.npy; differentiate Pos_noised on the "
                "actual PD kinematic timebase."
            ),
        },
        "dofs": [
            {
                "index": idx,
                "name": dof_names[idx],
                "empirical_error_std": float(error_std[idx]),
                "empirical_error_mae": float(error_mae[idx]),
                "error_scaler_perturbation": float(perturbations[idx]),
                "effective_error_scaler": float(effective_scalers[idx]),
                "target_noise_std": float(target_std[idx]),
            }
            for idx in range(NUM_DOFS)
        ],
        "manifest_csv": str(manifest_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
