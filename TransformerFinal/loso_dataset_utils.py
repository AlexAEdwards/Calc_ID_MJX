"""Dataset-neutral helpers for trusted-layout leave-one-subject-out runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from data_loader import TrialDataLoader


def natural_key(value: object) -> tuple:
    """Sort Y2 before Y10 while keeping cohort prefixes deterministic."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value)))


def parse_subject_list(values: Optional[Iterable[str]]) -> List[str]:
    result: List[str] = []
    for value in values or ():
        result.extend(part.strip() for part in str(value).split(",") if part.strip())
    return sorted(set(result), key=natural_key)


def discover_trusted_trials(
    data_dir: Path | str,
    *,
    include_subjects: Sequence[str] = (),
    exclude_subjects: Sequence[str] = (),
    max_trials_per_subject: Optional[int] = None,
    min_trial_length: int = 30,
) -> Dict[str, Any]:
    """Scan ``Subject/Trial/ProcessedData`` without creating a discovery cache."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    include = set(parse_subject_list(include_subjects))
    exclude = set(parse_subject_list(exclude_subjects))
    subject_dirs = sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: natural_key(p.name))
    subject_to_trials: Dict[str, List[Dict[str, Any]]] = {}
    considered_subjects: List[str] = []
    skipped: List[Dict[str, str]] = []
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        if include and subject not in include:
            continue
        if subject in exclude:
            continue
        considered_subjects.append(subject)
        found: List[Dict[str, Any]] = []
        trial_dirs = sorted((p for p in subject_dir.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name))
        for trial_dir in trial_dirs:
            processed = trial_dir / "ProcessedData"
            pos_path = processed / "pos_inputs.npy"
            if not pos_path.is_file():
                skipped.append({"subject": subject, "trial": trial_dir.name, "reason": f"missing {pos_path.name}"})
                continue
            try:
                length = int(np.load(pos_path, mmap_mode="r").shape[0])
            except Exception as exc:
                skipped.append({"subject": subject, "trial": trial_dir.name, "reason": f"unreadable pos_inputs.npy: {exc}"})
                continue
            if length < int(min_trial_length):
                skipped.append({"subject": subject, "trial": trial_dir.name, "reason": f"length {length} < {min_trial_length}"})
                continue
            found.append({
                "subject": subject,
                "subject_group": subject,
                "trial": trial_dir.name,
                "trial_name": f"{subject}/{trial_dir.name}",
                "dataset_root": str(root),
                "trial_root": str(trial_dir),
                "training_data_path": str(processed),
                "length": length,
            })
        if max_trials_per_subject is not None and int(max_trials_per_subject) > 0:
            found = found[: int(max_trials_per_subject)]
        if found:
            subject_to_trials[subject] = found
        else:
            skipped.append({"subject": subject, "trial": "", "reason": "no valid trials"})
    subjects = sorted(subject_to_trials, key=natural_key)
    return {
        "data_dir": str(root),
        "subjects": subjects,
        "all_subjects": considered_subjects,
        "subject_to_trials": subject_to_trials,
        "trials": [trial for subject in subjects for trial in subject_to_trials[subject]],
        "trial_counts": {subject: len(subject_to_trials[subject]) for subject in subjects},
        "skipped_trials": skipped,
    }


def build_loso_folds(
    subject_to_trials: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    held_out_subjects: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    subjects = sorted(subject_to_trials, key=natural_key)
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least two subjects with valid trials.")
    selected = parse_subject_list(held_out_subjects) or subjects
    unknown = sorted(set(selected) - set(subjects), key=natural_key)
    if unknown:
        raise ValueError(f"Held-out subjects were not discovered: {unknown}")
    folds: List[Dict[str, Any]] = []
    for held_out in selected:
        train_subjects = [subject for subject in subjects if subject != held_out]
        folds.append({
            "held_out_subject": held_out,
            "train_subjects": train_subjects,
            "train_trials": [dict(t) for s in train_subjects for t in subject_to_trials[s]],
            "held_out_trials": [dict(t) for t in subject_to_trials[held_out]],
        })
    return folds


def make_trusted_loader(
    trials: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
) -> "TrialDataLoader":
    from data_loader import TrialDataLoader
    if not trials:
        raise ValueError("Cannot create a loader for an empty trial list.")
    kwargs = dict(
        window_size=int(config["window_size"]),
        stride=int(config.get("stride", 16)),
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        trim_cop=bool(config.get("trim_cop", False)),
        deviation_learning=bool(config.get("deviation_learning", False)),
        use_noised=bool(config.get("use_noised", False)),
        noised_gt=bool(config.get("noised_gt", False)),
        predict_jacobian=bool(config.get("predict_jacobian", False)),
        opencap_val=False,
        input_source="processed",
        include_pelvis_euler=bool(config.get("include_pelvis_euler", False)),
        include_ankle_heights=bool(config.get("include_ankle_heights", True)),
        include_jacobian_input=bool(config.get("include_jacobian_input", True)),
        include_auxiliary_denoising_inputs=bool(config.get("include_auxiliary_denoising_inputs", True)),
        prediction_margin_frames=int(config.get("prediction_margin_frames", 20)),
        use_grf_norm_cop=bool(config.get("use_grf_norm_cop", False)),
        use_os_filtering=bool(config.get("use_os_filtering", False)),
        use_grf_nofilt=bool(config.get("use_grf_nofilt", True)),
        use_opensim_id_gt=bool(config.get("use_OpenSimID_GT", False)),
        subtract_ankle_height_knee_vecs=bool(
            config.get("subtract_ankle_height_knee_vecs", False)
        ),
        drop_last=False,
    )
    loader = TrialDataLoader(list(trials), **kwargs)
    if loader.total_windows <= 0:
        raise ValueError("Trusted-layout loader produced zero windows.")
    if kwargs["batch_size"] > loader.total_windows:
        kwargs["batch_size"] = int(loader.total_windows)
        loader = TrialDataLoader(list(trials), **kwargs)
    return loader


def validate_noised_inputs(trials: Sequence[Mapping[str, Any]], use_noised: bool) -> None:
    if not use_noised:
        return
    missing = []
    for trial in trials:
        processed = Path(str(trial["training_data_path"]))
        if not (processed / "pos_inputs_noised.npy").is_file():
            missing.append(str(processed / "pos_inputs_noised.npy"))
    if missing:
        raise FileNotFoundError(
            f"use_noised=True but {len(missing)} trial(s) lack pos_inputs_noised.npy; first: {missing[0]}"
        )


def validate_opensim_id_targets(trials: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Strictly preflight every requested OpenSim ID target before training starts."""
    from opensim_id_targets import load_aligned_opensim_id_target

    checked = []
    for trial in trials:
        processed = Path(str(trial["training_data_path"]))
        target_len = int(np.load(processed / "pos_inputs.npy", mmap_mode="r").shape[0])
        bundle = load_aligned_opensim_id_target(
            trial.get("trial_root", processed.parent),
            target_len=target_len,
        )
        checked.append({
            "subject": str(trial.get("subject", "")),
            "trial": str(trial.get("trial", trial.get("trial_name", ""))),
            "source_path": str(bundle["source_path"]),
            "target_len": target_len,
            "alignment": str(bundle["alignment"]),
            "available_columns": list(bundle["available_columns"]),
        })
    return {
        "trial_count": len(checked),
        "alignment": "timestamp interpolation to Motion/Time.npy, then recorded ProcessedData trims",
        "trials": checked,
    }
