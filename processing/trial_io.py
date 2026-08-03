"""Locating and copying a trial's processed artifacts, noised or not.

Extracted verbatim from ProcessData.py in REFACTOR_PLAN.md Stage 6. Filename
constants live in ``processing.artifact_names``.

Each trial can carry a parallel "noised" copy of its arrays - the same pipeline
run on kinematics with synthetic marker noise added, used to train models that
have to tolerate imperfect input. The convention is a suffix before the extension
(``pos_inputs.npy`` -> ``pos_inputs_noised.npy``) rather than a separate
directory, so both variants sit side by side in ProcessedData/.

The two "has noised" predicates answer different questions and are not
interchangeable: ``_has_noised_source_inputs`` asks whether noised *motion* exists
to process from, while ``_has_noised_prediction_bundle`` asks whether a complete
set of noised *outputs* already exists. A trial can have the first without the
second, which is exactly the case that means "there is work to do here".

ProcessData.py re-exports every name, so its callers are unchanged.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from processing.artifact_names import (
    NOISED_FILE_SUFFIX,
    NOISED_REQUIRED_BUNDLE_FILENAMES,
)


def _with_file_suffix(filename: str, suffix: str = NOISED_FILE_SUFFIX) -> str:
    path = Path(filename)
    if path.suffix:
        return f"{path.stem}{suffix}{path.suffix}"
    return f"{filename}{suffix}"


def _missing_noised_bundle_files(
    proc_dir: Path,
    filenames: tuple[str, ...] = NOISED_REQUIRED_BUNDLE_FILENAMES,
) -> list[str]:
    return [name for name in filenames if not (proc_dir / _with_file_suffix(name)).exists()]


def _has_noised_prediction_bundle(proc_dir: Path) -> bool:
    return len(_missing_noised_bundle_files(proc_dir)) == 0


def _has_noised_source_inputs(trial_path: Path) -> bool:
    motion_dir = trial_path / "Motion" / "Motion_Pelvis_Adjusted"
    if not motion_dir.exists():
        motion_dir = trial_path / "Motion"
    required = ("Pos_noised.npy", "Vel_noised.npy", "Accel_noised.npy")
    return all((motion_dir / name).exists() for name in required)


def _copy_outputs_with_suffix(src_dir: Path, dst_dir: Path, filenames: tuple[str, ...], suffix: str) -> None:
    for name in filenames:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / _with_file_suffix(name, suffix)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
