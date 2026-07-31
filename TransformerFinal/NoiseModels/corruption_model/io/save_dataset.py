from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from corruption_model.types import SyntheticSample


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_npz_shards(samples: Iterable[SyntheticSample], output_dir: str | Path, shard_size: int = 64) -> List[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_list = list(samples)
    shard_paths: List[Path] = []
    for shard_idx, start in enumerate(range(0, len(sample_list), shard_size)):
        batch = sample_list[start : start + shard_size]
        metadata_rows: List[Dict[str, Any]] = []
        arrays: Dict[str, np.ndarray] = {}
        for local_idx, sample in enumerate(batch):
            prefix = f"sample_{local_idx:04d}"
            arrays[f"{prefix}_q_input_corrupted"] = sample.q_input_corrupted.astype(np.float32)
            arrays[f"{prefix}_q_clean_reference"] = sample.q_clean_reference.astype(np.float32)
            arrays[f"{prefix}_time"] = sample.time.astype(np.float32)
            arrays[f"{prefix}_time_for_pos"] = sample.time_for_pos.astype(np.float32)
            metadata_rows.append(
                {
                    "subject_id": sample.subject_id,
                    "trial_id": sample.trial_id,
                    "activity": sample.activity,
                    "height_m": sample.subject_metadata.height_m,
                    "mass_kg": sample.subject_metadata.mass_kg,
                    "patient_md_path": str(sample.subject_metadata.patient_md_path) if sample.subject_metadata.patient_md_path else None,
                    "dof_names": list(sample.subject_metadata.dof_names),
                    "corruption_params": _json_safe(sample.corruption_params),
                    "meta": _json_safe(sample.meta),
                }
            )

        shard_path = output_path / f"synthetic_shard_{shard_idx:04d}.npz"
        np.savez_compressed(shard_path, metadata_json=np.asarray(json.dumps(metadata_rows), dtype=object), **arrays)
        shard_paths.append(shard_path)

    manifest = {
        "num_samples": len(sample_list),
        "num_shards": len(shard_paths),
        "shards": [path.name for path in shard_paths],
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return shard_paths


def save_processeddata_outputs(
    *,
    trial_dir: str | Path,
    output_subdir_name: str,
    corrupted_curves: List[Dict[str, Any]],
    time: np.ndarray,
    time_for_pos: np.ndarray,
    trial_metadata: Dict[str, Any],
) -> Path:
    output_dir = Path(trial_dir) / output_subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "Time.npy", np.asarray(time, dtype=np.float32))
    np.save(output_dir / "Time_for_pos.npy", np.asarray(time_for_pos, dtype=np.float32))

    summary_rows: List[Dict[str, Any]] = []
    for idx, curve in enumerate(corrupted_curves, start=1):
        suffix = f"{idx:03d}"
        np.save(output_dir / f"Pos_noised_{suffix}.npy", np.asarray(curve["pos"], dtype=np.float32))
        np.save(output_dir / f"Vel_noised_{suffix}.npy", np.asarray(curve["vel"], dtype=np.float32))
        np.save(output_dir / f"Accel_noised_{suffix}.npy", np.asarray(curve["accel"], dtype=np.float32))
        summary_rows.append(
            {
                "curve_index": idx,
                "pos_file": f"Pos_noised_{suffix}.npy",
                "vel_file": f"Vel_noised_{suffix}.npy",
                "accel_file": f"Accel_noised_{suffix}.npy",
                "corruption_params": _json_safe(curve["corruption_params"]),
            }
        )

    payload = {
        "trial_metadata": _json_safe(trial_metadata),
        "generated_curves": summary_rows,
    }
    (output_dir / "corruption_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_dir
