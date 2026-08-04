#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SOURCE = Path("StrokeDataset")
DEFAULT_OUTPUT = Path("Stroke_Cleaned_Dataset")
REVIEW_JSON = "visual_cleaning_review.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def parse_window(window: dict[str, Any]) -> tuple[float, float]:
    start = float(window.get("keep_start_time_s", window.get("start_time_s")))
    end = float(window.get("keep_end_time_s", window.get("end_time_s")))
    if end <= start:
        raise ValueError(f"Invalid trim window: {window}")
    return start, end


def bounds_from_time_window(length: int, start_s: float, end_s: float, dt: float = 0.01) -> tuple[int, int]:
    time = np.arange(length, dtype=np.float64) * dt
    lo = int(np.searchsorted(time, start_s, side="left"))
    hi = int(np.searchsorted(time, end_s, side="right"))
    lo = max(0, min(lo, length))
    hi = max(lo + 1, min(hi, length))
    return lo, hi


def slice_array(arr: np.ndarray, lo: int, hi: int, expected_len: int) -> np.ndarray:
    if arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            sliced: dict[Any, Any] = {}
            changed = False
            for key, value in obj.items():
                if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == expected_len:
                    sliced[key] = value[lo:hi]
                    changed = True
                else:
                    sliced[key] = value
            return np.array(sliced, dtype=object) if changed else arr
    if arr.ndim >= 1 and arr.shape[0] == expected_len:
        return arr[lo:hi]
    return arr


def copy_npy_with_optional_slice(
    src: Path,
    dst: Path,
    lo: int | None,
    hi: int | None,
    expected_len: int | None,
) -> bool:
    if lo is None or hi is None or expected_len is None:
        shutil.copy2(src, dst)
        return False

    arr = np.load(src, allow_pickle=True)
    sliced = slice_array(arr, lo, hi, expected_len)
    if sliced is arr:
        shutil.copy2(src, dst)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, sliced, allow_pickle=arr.dtype == object)
    shutil.copystat(src, dst)
    return True


def reset_time_if_needed(path: Path) -> None:
    if path.name not in {"Time.npy", "Time_for_pos.npy"}:
        return
    arr = np.load(path)
    if arr.ndim == 1 and arr.size:
        np.save(path, arr - arr[0])


def motion_bounds_for_file(
    src_path: Path,
    motion_len: int,
    motion_base_len: int | None,
    core_bounds: list[int] | None,
    proc_lo: int,
    proc_hi: int,
) -> tuple[int, int, int] | tuple[None, None, None]:
    if not core_bounds or len(core_bounds) != 2:
        return None, None, None

    core_lo, core_hi = int(core_bounds[0]), int(core_bounds[1])
    if motion_base_len is not None and motion_len == motion_base_len:
        return core_lo + proc_lo, core_lo + proc_hi, motion_len

    if motion_base_len is None or motion_base_len <= 0:
        return None, None, None

    ratio = motion_len / motion_base_len
    scaled_lo = int(round((core_lo + proc_lo) * ratio))
    scaled_hi = int(round((core_lo + proc_hi) * ratio))
    if scaled_hi <= scaled_lo:
        return None, None, None
    return scaled_lo, min(scaled_hi, motion_len), motion_len


def update_processing_info(path: Path, label: str, window: dict[str, Any], lo: int, hi: int, before: int) -> None:
    info = load_json(path) if path.exists() else {}
    info.update(
        {
            "n_frames": int(hi - lo),
            "manual_visual_trim_applied": True,
            "manual_visual_trim_source_trial": label,
            "manual_visual_trim_window": window,
            "manual_visual_trim_bounds_processed": [int(lo), int(hi)],
            "manual_visual_trim_n_frames_before": int(before),
            "manual_visual_trim_n_frames_after": int(hi - lo),
            "manual_visual_trim_timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(path, info)


def copy_subject_assets(subject_src: Path, subject_dst: Path) -> None:
    for child in sorted(subject_src.iterdir()):
        if child.name.startswith("Trial_"):
            continue
        target = subject_dst / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def copy_trimmed_trial(src_trial: Path, dst_trial: Path, label: str, window: dict[str, Any]) -> dict[str, Any]:
    proc_grf = src_trial / "ProcessedData" / "GRF_Cleaned.npy"
    if not proc_grf.exists():
        raise FileNotFoundError(f"Missing ProcessedData/GRF_Cleaned.npy for {label}")

    proc_len = int(np.load(proc_grf, mmap_mode="r").shape[0])
    start_s, end_s = parse_window(window)
    proc_lo, proc_hi = bounds_from_time_window(proc_len, start_s, end_s)

    info_path = src_trial / "ProcessedData" / "Trial_Processing_Information.json"
    info = load_json(info_path) if info_path.exists() else {}
    core_bounds = info.get("core_trim_bounds_motion_aligned")
    motion_base_path = src_trial / "Motion" / "Time_for_pos.npy"
    motion_base_len = None
    if motion_base_path.exists():
        motion_base_len = int(np.load(motion_base_path, mmap_mode="r").shape[0])

    files_copied = 0
    npy_trimmed = 0
    for src_path in sorted(src_trial.rglob("*")):
        if not src_path.is_file():
            continue
        rel = src_path.relative_to(src_trial)
        if "Raw" in rel.parts or "Visualizations" in rel.parts:
            continue
        if "backup" in src_path.name.lower():
            continue
        dst_path = dst_trial / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix != ".npy":
            shutil.copy2(src_path, dst_path)
            files_copied += 1
            continue

        expected_len: int | None = None
        lo: int | None = None
        hi: int | None = None
        if rel.parts[0] == "ProcessedData":
            expected_len, lo, hi = proc_len, proc_lo, proc_hi
        elif rel.parts[0] == "Motion":
            arr_len = int(np.load(src_path, mmap_mode="r", allow_pickle=True).shape[0])
            lo, hi, expected_len = motion_bounds_for_file(
                src_path,
                arr_len,
                motion_base_len,
                core_bounds,
                proc_lo,
                proc_hi,
            )

        if copy_npy_with_optional_slice(src_path, dst_path, lo, hi, expected_len):
            npy_trimmed += 1
            reset_time_if_needed(dst_path)
        files_copied += 1

    update_processing_info(dst_trial / "ProcessedData" / "Trial_Processing_Information.json", label, window, proc_lo, proc_hi, proc_len)
    noised_info = dst_trial / "ProcessedData" / "Trial_Processing_Information_noised.json"
    if noised_info.exists():
        update_processing_info(noised_info, label, window, proc_lo, proc_hi, proc_len)

    return {
        "label": label,
        "processed_bounds": [proc_lo, proc_hi],
        "processed_frames_before": proc_len,
        "processed_frames_after": proc_hi - proc_lo,
        "files_copied": files_copied,
        "npy_files_trimmed": npy_trimmed,
    }


def build_cleaned_dataset(source: Path, output: Path, overwrite: bool) -> dict[str, Any]:
    review = load_json(source / REVIEW_JSON)
    trim_windows = review.get("trim_windows", {})
    labels = sorted(label for label in trim_windows if label.split("/", 1)[0].startswith("SUBJ"))

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    labels_by_subject: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        subject, _trial = label.split("/", 1)
        labels_by_subject[subject].append(label)

    trial_reports: list[dict[str, Any]] = []
    for subject in sorted(labels_by_subject):
        subject_src = source / subject
        subject_dst = output / subject
        if not subject_src.is_dir():
            continue
        copy_subject_assets(subject_src, subject_dst)
        for label in labels_by_subject[subject]:
            _subject, trial = label.split("/", 1)
            trial_reports.append(
                copy_trimmed_trial(
                    subject_src / trial,
                    subject_dst / trial,
                    label,
                    trim_windows[label],
                )
            )

    filtered_review = dict(review)
    filtered_review["dataset_root"] = str(output.resolve())
    filtered_review["source_dataset_root"] = str(source.resolve())
    filtered_review["trim_windows"] = {label: trim_windows[label] for label in labels}
    for key in ("keep_trials", "remove_trials", "needs_more_trimming_trials"):
        filtered_review[key] = [label for label in review.get(key, []) if label in labels]
    write_json(output / REVIEW_JSON, filtered_review)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(source.resolve()),
        "output_dataset": str(output.resolve()),
        "subject_filter": "SUBJ*",
        "excluded_subject_prefixes": ["TVC"],
        "trial_source": f"{REVIEW_JSON}: trim_windows",
        "subjects": sorted(labels_by_subject),
        "n_subjects": len(labels_by_subject),
        "n_trials": len(trial_reports),
        "trials": trial_reports,
    }
    write_json(output / "cleaned_dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stroke_Cleaned_Dataset from visual trim windows.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = build_cleaned_dataset(args.source, args.output, args.overwrite)
    print(f"Wrote {manifest['output_dataset']}")
    print(f"Subjects: {manifest['n_subjects']}")
    print(f"Trials: {manifest['n_trials']}")


if __name__ == "__main__":
    main()
