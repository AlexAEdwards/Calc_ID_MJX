#!/usr/bin/env python3
"""Verify ProcessData.py output dimensions for selected trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


MODEL_OUTPUT_DOF_COUNT = 23
INPUT_SHAPES = {
    "pos_inputs.npy": (18,),
    "vel_inputs.npy": (21,),
    "acc_inputs.npy": (21,),
    "pelvis_rot_matrix.npy": (6,),
    "ankle_heights.npy": (2,),
    "Height_m.npy": (),
    "Mass_kg.npy": (),
    "forwardVel.npy": (),
}
GT_SHAPES = {
    "ID_GT_MJX.npy": (MODEL_OUTPUT_DOF_COUNT,),
    "qfrc_inverse.npy": (MODEL_OUTPUT_DOF_COUNT,),
    "qfrc_grf_contribution.npy": (MODEL_OUTPUT_DOF_COUNT,),
    "GRF_Cleaned.npy": (6,),
    "GRF_NoFilt_Trimmed.npy": (6,),
    "Moment_Cleaned.npy": (6,),
    "COP_Cleaned_Relative.npy": (4,),
    "contactBoolean.npy": (2,),
}
AUX_SHAPES = {
    "ankle_pos_r.npy": (3,),
    "ankle_pos_l.npy": (3,),
    "knee_pos_r.npy": (3,),
    "knee_pos_l.npy": (3,),
    "toes_pos_r.npy": (3,),
    "toes_pos_l.npy": (3,),
    "COM_r.npy": (3,),
    "COM_l.npy": (3,),
    "COM_Acc_Global.npy": (3,),
    "KneeToCOP_Vectors.npy": (6,),
}
POST_SHAPES = {
    "FootProgressionAngle.npy": (2,),
    "Foot_ProgressionAngle.npy": (2,),
    "CalcnToFloor_AngleDeg.npy": (2,),
    "COP_CalcFrame.npy": (6,),
    "COP_CalcFrame_GroundAligned.npy": (6,),
    "COP_CalcFrame_GroundAligned_GRFNorm.npy": (6,),
    "COP_Cleaned_Relative_RecoveredFromGroundAligned.npy": (4,),
    "WorldToGroundAlignedCalcnRotation.npy": (2, 3, 3),
}
STATE_FILES = ("pos_mjx.npy", "qvel_mjx.npy", "qacc_mjx.npy")
NOISED_REQUIRED = (
    "pos_inputs.npy",
    "vel_inputs.npy",
    "acc_inputs.npy",
    "pelvis_rot_matrix.npy",
    "pos_mjx.npy",
    "qvel_mjx.npy",
    "qacc_mjx.npy",
    "qfrc_inverse.npy",
    "Jacobian.npy",
    "ankle_heights.npy",
    "COP_Cleaned_Relative.npy",
    "COP_CalcFrame_GroundAligned.npy",
)


def with_noised_name(name: str) -> str:
    path = Path(name)
    return f"{path.stem}_noised{path.suffix}"


def load_array(path: Path) -> Any:
    arr = np.load(path, allow_pickle=True)
    if getattr(arr, "shape", None) == () and getattr(arr, "dtype", None) == object:
        return arr.item()
    return arr


def expected_shape(frames: int, tail: tuple[int, ...]) -> tuple[int, ...]:
    return (frames,) + tail


def check_array(
    proc_dir: Path,
    name: str,
    tail: tuple[int, ...],
    frames: int,
    errors: list[str],
    *,
    required: bool = True,
) -> None:
    path = proc_dir / name
    if not path.exists():
        if required:
            errors.append(f"missing {name}")
        return
    arr = load_array(path)
    if not hasattr(arr, "shape"):
        errors.append(f"{name}: not an ndarray-like payload")
        return
    exp = expected_shape(frames, tail)
    if tuple(arr.shape) != exp:
        errors.append(f"{name}: shape {tuple(arr.shape)} expected {exp}")


def check_jacobian(proc_dir: Path, frames: int, errors: list[str], *, name: str = "Jacobian.npy") -> None:
    path = proc_dir / name
    if not path.exists():
        errors.append(f"missing {name}")
        return
    payload = load_array(path)
    if not isinstance(payload, dict):
        errors.append(f"{name}: expected dict payload, got {type(payload).__name__}")
        return
    for key in ("jacp", "jacr"):
        arr = np.asarray(payload.get(key))
        exp = (frames, 2, 3, MODEL_OUTPUT_DOF_COUNT)
        if tuple(arr.shape) != exp:
            errors.append(f"{name}.{key}: shape {tuple(arr.shape)} expected {exp}")
    body_ids = np.asarray(payload.get("body_ids"))
    if tuple(body_ids.shape) != (2,):
        errors.append(f"{name}.body_ids: shape {tuple(body_ids.shape)} expected (2,)")


def resolve_model(subject_dir: Path, source: str | None) -> Path | None:
    candidates = []
    if source in {"MoCap", "Video"}:
        candidates.extend([
            subject_dir / f"MyosuiteModel_{source}_FIXED.xml",
            subject_dir / f"MyosuiteModel_{source}.xml",
        ])
    candidates.extend([
        subject_dir / "MyosuiteModel_FIXED.xml",
        subject_dir / "MyosuiteModel.xml",
    ])
    for path in candidates:
        if path.exists():
            return path
    return None


def trial_output_dirs(data_root: Path, trial_id: str, opencapval: bool) -> list[tuple[str | None, Path, Path]]:
    subject, trial = trial_id.split("/", 1)
    subject_dir = data_root / subject
    trial_dir = subject_dir / trial
    if opencapval:
        return [
            ("MoCap", subject_dir, trial_dir / "MoCap" / "ProcessedData"),
            ("Video", subject_dir, trial_dir / "Video" / "ProcessedData"),
        ]
    return [(None, subject_dir, trial_dir / "ProcessedData")]


def verify_output(proc_dir: Path, subject_dir: Path, source: str | None, use_noised: bool) -> list[str]:
    errors: list[str] = []
    model_path = resolve_model(subject_dir, source)
    if model_path is None:
        return [f"missing model XML for {subject_dir.name} source={source or 'standard'}"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    nq = int(model.nq)
    nv = int(model.nv)

    pos_path = proc_dir / "pos_mjx.npy"
    if not pos_path.exists():
        return [f"missing {pos_path}"]
    pos = load_array(pos_path)
    if not hasattr(pos, "shape") or len(pos.shape) != 2:
        return [f"pos_mjx.npy invalid shape {getattr(pos, 'shape', None)}"]
    frames = int(pos.shape[0])
    if tuple(pos.shape) != (frames, nq):
        errors.append(f"pos_mjx.npy: shape {tuple(pos.shape)} expected {(frames, nq)}")

    state_expected = {
        "qvel_mjx.npy": (nv,),
        "qacc_mjx.npy": (nv,),
    }
    for name, tail in state_expected.items():
        check_array(proc_dir, name, tail, frames, errors)

    for shape_map in (INPUT_SHAPES, GT_SHAPES, AUX_SHAPES):
        for name, tail in shape_map.items():
            check_array(proc_dir, name, tail, frames, errors)
    check_jacobian(proc_dir, frames, errors)

    for name, tail in POST_SHAPES.items():
        check_array(proc_dir, name, tail, frames, errors, required=False)

    if use_noised:
        for clean_name in NOISED_REQUIRED:
            noised_name = with_noised_name(clean_name)
            if clean_name == "Jacobian.npy":
                check_jacobian(proc_dir, frames, errors, name=noised_name)
            elif clean_name in INPUT_SHAPES:
                check_array(proc_dir, noised_name, INPUT_SHAPES[clean_name], frames, errors)
            elif clean_name in GT_SHAPES:
                check_array(proc_dir, noised_name, GT_SHAPES[clean_name], frames, errors)
            elif clean_name in POST_SHAPES:
                check_array(proc_dir, noised_name, POST_SHAPES[clean_name], frames, errors)
            elif clean_name == "pos_mjx.npy":
                check_array(proc_dir, noised_name, (nq,), frames, errors)
            elif clean_name in {"qvel_mjx.npy", "qacc_mjx.npy"}:
                check_array(proc_dir, noised_name, (nv,), frames, errors)
            elif clean_name in INPUT_SHAPES:
                check_array(proc_dir, noised_name, INPUT_SHAPES[clean_name], frames, errors)
            elif clean_name == "ankle_heights.npy":
                check_array(proc_dir, noised_name, (2,), frames, errors)
            elif clean_name == "COP_Cleaned_Relative.npy":
                check_array(proc_dir, noised_name, (4,), frames, errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--trials-file", required=True)
    parser.add_argument("--OpenCapVal", action="store_true")
    parser.add_argument("--UseNoised", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    trial_ids = json.loads(Path(args.trials_file).read_text())
    total = 0
    failed = 0
    for trial_id in trial_ids:
        for source, subject_dir, proc_dir in trial_output_dirs(data_root, trial_id, args.OpenCapVal):
            total += 1
            label = f"{trial_id}" if source is None else f"{trial_id}/{source}"
            errors = verify_output(proc_dir, subject_dir, source, args.UseNoised and not args.OpenCapVal)
            if errors:
                failed += 1
                print(f"[FAIL] {label} -> {proc_dir}")
                for err in errors:
                    print(f"  - {err}")
            else:
                print(f"[OK]   {label} -> {proc_dir}")
    print(f"\nChecked outputs: {total}; failed: {failed}; passed: {total - failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
