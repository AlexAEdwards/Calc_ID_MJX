#!/usr/bin/env python3
"""Restore knee-angle channels in processed position/velocity/acceleration inputs.

The canonical schemas produced by ProcessData.py are:

* pos_inputs: 18 columns; pelvis angles, bilateral leg angles (including the
  independent knee_angle_r/l coordinates), and lumbar angles. Pelvis XYZ and
  both MTP coordinates are excluded.
* vel_inputs / acc_inputs: 21 columns; the same independent coordinates plus
  pelvis XYZ derivatives. Both MTP coordinates are excluded.

Knee values are always rebuilt from the matching model-space state file:
pos_mjx, qvel_mjx, or qacc_mjx. Clean, _noised, and _OSfilt suffixes remain
paired. Before a file is changed, its existing non-knee channels must match a
recognized legacy schema derived from the same MJX state and timebase.

Run with the project's MuJoCo-capable Python environment:

    python scripts/restore_knee_input_channels.py
    python scripts/restore_knee_input_channels.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    PROJECT_ROOT / "TrustedDataSetNoised12Distributed_EdgeHold_AllPatients",
    PROJECT_ROOT / "TrustedDataSet_ByExperiment",
    PROJECT_ROOT / "OpenCapWalkingTrunkSwaySubjects",
)

POS_COLUMNS = (
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)
MODEL_SAVE_DOF_NAMES = (
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)
LEGACY_31_QPOS_NAMES = (
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "walker_knee_r_translation1", "walker_knee_r_translation2",
    "knee_angle_r", "walker_knee_r_rotation2", "walker_knee_r_rotation3",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "walker_knee_l_translation1", "walker_knee_l_translation2",
    "knee_angle_l", "walker_knee_l_rotation2", "walker_knee_l_rotation3",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)
CANONICAL_33_QPOS_NAMES = (
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "walker_knee_r_translation1", "walker_knee_r_translation2",
    "walker_knee_r_translation3", "knee_angle_r",
    "walker_knee_r_rotation2", "walker_knee_r_rotation3",
    "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "walker_knee_l_translation1", "walker_knee_l_translation2",
    "walker_knee_l_translation3", "knee_angle_l",
    "walker_knee_l_rotation2", "walker_knee_l_rotation3",
    "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
)

POS_INPUT_NAMES = tuple(
    name for name in POS_COLUMNS
    if name not in {"pelvis_tx", "pelvis_ty", "pelvis_tz", "mtp_angle_r", "mtp_angle_l"}
)
VEL_ACC_INPUT_NAMES = tuple(
    name for name in POS_COLUMNS if name not in {"mtp_angle_r", "mtp_angle_l"}
)
LEGACY_POS_NO_KNEES = tuple(name for name in POS_INPUT_NAMES if not name.startswith("knee_angle_"))
LEGACY_POS_MTP_NO_KNEES = tuple(
    name for name in POS_COLUMNS
    if name not in {"pelvis_tx", "pelvis_ty", "pelvis_tz", "knee_angle_r", "knee_angle_l"}
)
LEGACY_VEL_NO_KNEES = tuple(
    name for name in VEL_ACC_INPUT_NAMES if not name.startswith("knee_angle_")
)
LEGACY_VEL_MTP_NO_KNEES = tuple(
    name for name in POS_COLUMNS if name not in {"knee_angle_r", "knee_angle_l"}
)

TARGETS = {
    "pos_inputs": ("pos_mjx", POS_INPUT_NAMES),
    "vel_inputs": ("qvel_mjx", VEL_ACC_INPUT_NAMES),
    "acc_inputs": ("qacc_mjx", VEL_ACC_INPUT_NAMES),
}


def _suffix_for_target(path: Path, target_stem: str) -> str:
    return path.stem[len(target_stem):]


def _fixed_width_name_map(width: int) -> dict[str, int] | None:
    names = {
        len(MODEL_SAVE_DOF_NAMES): MODEL_SAVE_DOF_NAMES,
        len(LEGACY_31_QPOS_NAMES): LEGACY_31_QPOS_NAMES,
        len(CANONICAL_33_QPOS_NAMES): CANONICAL_33_QPOS_NAMES,
    }.get(width)
    return None if names is None else {name: idx for idx, name in enumerate(names)}


def _find_model_xml(directory: Path, dataset_root: Path) -> Path | None:
    modality = next((part for part in reversed(directory.parts) if part in {"Video", "MoCap"}), None)
    modality_names = (
        [f"MyosuiteModel_{modality}_FIXED.xml", f"MyosuiteModel_{modality}.xml"]
        if modality else []
    )
    general_names = [
        "MyosuiteModel_FIXED.xml",
        "MyosuiteModel_Runtime.xml",
        "MyosuiteModel.xml",
    ]
    current = directory
    while True:
        for name in modality_names + general_names:
            candidate = current / name
            if candidate.is_file():
                return candidate
        if current == dataset_root or current == current.parent:
            return None
        current = current.parent


def _model_name_map(
    width: int,
    directory: Path,
    dataset_root: Path,
    *,
    velocity: bool,
    cache: dict[tuple[str, bool], dict[str, int]],
) -> dict[str, int]:
    fixed = _fixed_width_name_map(width)
    if fixed is not None:
        return fixed

    xml_path = _find_model_xml(directory, dataset_root)
    if xml_path is None:
        raise ValueError(f"no model XML found for {width}-column MJX state")
    cache_key = (str(xml_path.resolve()), velocity)
    if cache_key in cache:
        mapping = cache[cache_key]
    else:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is required for full model-space MJX arrays; run with "
                "/home/mobl/miniconda3/envs/myoconverter/bin/python"
            ) from exc
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        expected_width = int(model.nv if velocity else model.nq)
        if expected_width != width:
            raise ValueError(
                f"{xml_path} has {'nv' if velocity else 'nq'}={expected_width}, "
                f"but state width is {width}"
            )
        mapping = {}
        for jid in range(int(model.njnt)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not name:
                continue
            mapping[str(name)] = int(model.jnt_dofadr[jid] if velocity else model.jnt_qposadr[jid])
        cache[cache_key] = mapping
    return mapping


def _select(state: np.ndarray, name_map: dict[str, int], names: Iterable[str]) -> np.ndarray:
    names = tuple(names)
    missing = [name for name in names if name not in name_map]
    if missing:
        raise ValueError("MJX layout is missing coordinates: " + ", ".join(missing))
    return state[:, [name_map[name] for name in names]]


def _candidate_existing_schemas(target_stem: str) -> tuple[tuple[str, ...], ...]:
    if target_stem == "pos_inputs":
        return (
            POS_INPUT_NAMES,
            LEGACY_POS_NO_KNEES,
            LEGACY_POS_MTP_NO_KNEES,
        )
    return (
        VEL_ACC_INPUT_NAMES,
        LEGACY_VEL_NO_KNEES,
        LEGACY_VEL_MTP_NO_KNEES,
        POS_COLUMNS,
    )


def _classify_existing_schema(
    target_stem: str,
    existing: np.ndarray,
    state: np.ndarray,
    name_map: dict[str, int],
) -> tuple[str, ...] | None:
    """Identify the old columns while preserving their stored signal values.

    The dominant 16/19-column layouts are unambiguous. Their derivatives can
    legitimately differ from qvel/qacc because older processing variants used
    different filters, so equal frame count and same-directory suffix pairing
    establish alignment while the original non-knee channels are preserved.
    Ambiguous 18/21-column layouts are resolved by value matching.
    """
    if target_stem == "pos_inputs" and existing.shape[1] == len(LEGACY_POS_NO_KNEES):
        return LEGACY_POS_NO_KNEES
    if target_stem != "pos_inputs" and existing.shape[1] == len(LEGACY_VEL_NO_KNEES):
        return LEGACY_VEL_NO_KNEES

    for names in _candidate_existing_schemas(target_stem):
        if _matches(
            existing,
            state,
            name_map,
            names,
            ignore_pelvis_translation=(target_stem != "pos_inputs"),
        ):
            return names
    return None


def _rebuild_preserving_existing(
    existing: np.ndarray,
    existing_names: tuple[str, ...],
    state: np.ndarray,
    name_map: dict[str, int],
    desired_names: tuple[str, ...],
) -> np.ndarray:
    """Insert missing knees from MJX and retain every existing desired channel."""
    existing_by_name = {name: existing[:, idx] for idx, name in enumerate(existing_names)}
    columns = []
    for name in desired_names:
        if name in existing_by_name:
            columns.append(existing_by_name[name])
        else:
            columns.append(state[:, name_map[name]])
    return np.column_stack(columns).astype(existing.dtype, copy=False)


def _matches(
    existing: np.ndarray,
    state: np.ndarray,
    name_map: dict[str, int],
    names: tuple[str, ...],
    *,
    ignore_pelvis_translation: bool,
) -> bool:
    if existing.shape != (state.shape[0], len(names)):
        return False
    expected = _select(state, name_map, names)
    compare_cols = [
        idx for idx, name in enumerate(names)
        if not (ignore_pelvis_translation and name in {"pelvis_tx", "pelvis_ty", "pelvis_tz"})
    ]
    if not compare_cols:
        return False
    return bool(np.allclose(
        existing[:, compare_cols],
        expected[:, compare_cols],
        rtol=1e-7,
        atol=1e-9,
        equal_nan=True,
    ))


def _atomic_save(path: Path, array: np.ndarray) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _state_and_map(
    state_path: Path,
    dataset_root: Path,
    *,
    velocity: bool,
    model_cache: dict[tuple[str, bool], dict[str, int]],
) -> tuple[np.ndarray, dict[str, int]]:
    state = np.load(state_path, allow_pickle=False)
    if state.ndim != 2:
        raise ValueError(f"{state_path.name} must be 2D, got {state.shape}")
    name_map = _model_name_map(
        int(state.shape[1]),
        state_path.parent,
        dataset_root,
        velocity=velocity,
        cache=model_cache,
    )
    return state, name_map


def _find_aligned_parent_state(
    target_path: Path,
    state_name: str,
    suffix: str,
    dataset_root: Path,
    *,
    velocity: bool,
    model_cache: dict[tuple[str, bool], dict[str, int]],
) -> tuple[Path, np.ndarray, dict[str, int]] | None:
    """Use an ancestor state only when its knee-position timeline matches locally."""
    local_pos_path = target_path.parent / f"pos_mjx{suffix if suffix == '_noised' else ''}.npy"
    if not local_pos_path.exists():
        local_pos_path = target_path.parent / "pos_mjx.npy"
    if not local_pos_path.exists():
        return None
    local_pos, local_map = _state_and_map(
        local_pos_path, dataset_root, velocity=False, model_cache=model_cache
    )
    local_knees = _select(local_pos, local_map, ("knee_angle_r", "knee_angle_l"))

    current = target_path.parent.parent
    while current != current.parent:
        candidate = current / f"{state_name}{suffix}.npy"
        candidate_pos = current / f"pos_mjx{suffix if suffix == '_noised' else ''}.npy"
        if candidate.exists() and candidate_pos.exists():
            cand_pos, cand_pos_map = _state_and_map(
                candidate_pos, dataset_root, velocity=False, model_cache=model_cache
            )
            cand_knees = _select(cand_pos, cand_pos_map, ("knee_angle_r", "knee_angle_l"))
            if (
                cand_knees.shape == local_knees.shape
                and np.allclose(cand_knees, local_knees, rtol=1e-7, atol=1e-9, equal_nan=True)
            ):
                state, name_map = _state_and_map(
                    candidate, dataset_root, velocity=velocity, model_cache=model_cache
                )
                return candidate, state, name_map
        if current == dataset_root:
            break
        current = current.parent
    return None


def migrate_root(root: Path, *, apply: bool) -> dict:
    summary = Counter()
    failures: list[dict[str, str]] = []
    parent_fallbacks: list[dict[str, str]] = []
    model_cache: dict[tuple[str, bool], dict[str, int]] = {}

    target_paths = sorted(
        path
        for target_stem in TARGETS
        for path in root.rglob(f"{target_stem}*.npy")
    )
    for index, target_path in enumerate(target_paths, start=1):
        target_stem = next(stem for stem in TARGETS if target_path.stem.startswith(stem))
        state_stem, desired_names = TARGETS[target_stem]
        suffix = _suffix_for_target(target_path, target_stem)
        state_path = target_path.parent / f"{state_stem}{suffix}.npy"
        velocity = target_stem != "pos_inputs"

        try:
            existing = np.load(target_path, allow_pickle=False)
            if existing.ndim != 2:
                raise ValueError(f"target must be 2D, got {existing.shape}")

            used_parent = False
            if state_path.exists():
                state, name_map = _state_and_map(
                    state_path, root, velocity=velocity, model_cache=model_cache
                )
            else:
                fallback = _find_aligned_parent_state(
                    target_path,
                    state_stem,
                    suffix,
                    root,
                    velocity=velocity,
                    model_cache=model_cache,
                )
                if fallback is None:
                    raise FileNotFoundError(f"missing matching {state_path.name}")
                state_path, state, name_map = fallback
                used_parent = True

            if existing.shape[0] != state.shape[0]:
                raise ValueError(
                    f"frame mismatch: {target_path.name}={existing.shape[0]}, "
                    f"{state_path.name}={state.shape[0]}"
                )

            existing_names = _classify_existing_schema(
                target_stem, existing, state, name_map
            )
            if existing_names is None:
                raise ValueError(
                    "existing non-knee channels do not match a recognized schema "
                    f"from {state_path}"
                )

            rebuilt = _rebuild_preserving_existing(
                existing, existing_names, state, name_map, desired_names
            )
            if existing.shape == rebuilt.shape and np.allclose(
                existing, rebuilt, rtol=1e-7, atol=1e-9, equal_nan=True
            ):
                summary["already_current"] += 1
                continue

            if apply:
                _atomic_save(target_path, rebuilt)
            summary["would_update" if not apply else "updated"] += 1
            summary[f"{target_stem}:{existing.shape[1]}->{rebuilt.shape[1]}"] += 1
            if used_parent:
                summary["aligned_parent_fallback"] += 1
                if len(parent_fallbacks) < 100:
                    parent_fallbacks.append({
                        "target": str(target_path.relative_to(root)),
                        "state": str(state_path.relative_to(root)),
                    })
        except Exception as exc:
            summary["failed"] += 1
            if len(failures) < 500:
                failures.append({
                    "target": str(target_path.relative_to(root)),
                    "error": str(exc),
                })

        if index % 2000 == 0:
            print(
                f"[{root.name}] {index}/{len(target_paths)} "
                f"updated={summary['updated'] or summary['would_update']} "
                f"failed={summary['failed']}",
                flush=True,
            )

    return {
        "root": str(root),
        "mode": "apply" if apply else "dry-run",
        "target_files": len(target_paths),
        "summary": dict(summary),
        "failures": failures,
        "parent_fallbacks": parent_fallbacks,
    }


def write_schema(root: Path, report: dict) -> None:
    schema = {
        "schema_version": "knee_inclusive_no_mtp_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "position_input_columns": list(POS_INPUT_NAMES),
        "velocity_input_columns": list(VEL_ACC_INPUT_NAMES),
        "acceleration_input_columns": list(VEL_ACC_INPUT_NAMES),
        "source_pairing": {
            "pos_inputs": "pos_mjx",
            "vel_inputs": "qvel_mjx",
            "acc_inputs": "qacc_mjx",
            "suffix_policy": "clean/noised/OSfilt suffixes remain paired",
        },
        "migration_summary": report["summary"],
    }
    path = root / "Kinematic_Input_Schema.json"
    text = json.dumps(schema, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".Kinematic_Input_Schema.", suffix=".json", dir=root)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Atomically replace validated input arrays")
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=list(DEFAULT_ROOTS),
        help="Dataset roots to scan",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "restore_knee_input_channels_report.json",
    )
    args = parser.parse_args()

    reports = []
    for raw_root in args.roots:
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            reports.append({
                "root": str(root),
                "mode": "apply" if args.apply else "dry-run",
                "target_files": 0,
                "summary": {"failed": 1},
                "failures": [{"target": ".", "error": "dataset root does not exist"}],
                "parent_fallbacks": [],
            })
            continue
        report = migrate_root(root, apply=args.apply)
        reports.append(report)
        print(json.dumps({"root": root.name, **report["summary"]}, sort_keys=True), flush=True)
        if args.apply:
            write_schema(root, report)

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "schemas": {
            "pos_inputs": list(POS_INPUT_NAMES),
            "vel_inputs": list(VEL_ACC_INPUT_NAMES),
            "acc_inputs": list(VEL_ACC_INPUT_NAMES),
        },
        "datasets": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Report: {args.report}")
    return 1 if any(report["summary"].get("failed", 0) for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
