#!/usr/bin/env python3
"""Clean per-leg COP samples using three rules:

  (a) **vGRF mask** — for each frame and leg, if vertical GRF is below
      ``--vgrf_threshold`` (default 1 N), zero the entire leg's COP row.
  (b) **X bounds** — for each frame and leg, if the COP X (anterior-posterior,
      foot frame) is outside ``[--cop_x_min, --cop_x_max]`` (defaults
      -0.10 m, +0.35 m), zero **just** the X column for that frame.
  (c) **Z bounds** — for each frame and leg, if ``|COP_Z|`` (medio-lateral,
      foot frame) exceeds ``--cop_z_abs_max`` (default 0.15 m), zero **just**
      the Z column for that frame.

The X and Z bounds are anatomically motivated and only meaningful for
foot-relative ("calcaneus frame") COP arrays — every COP variant currently
written in this dataset (CalcFrame, GroundAligned, BackToWorld, Relative,
RecoveredFromGroundAligned) lives in that frame.

For every selected source directory (default ``ProcessedData`` and ``MoCap``)
under each trial in the chosen dataset root the script:
  1. Loads ``GRF_Cleaned.npy`` — assumed layout (T, 6) with columns
     [Fx_R, Fy_R, Fz_R, Fx_L, Fy_L, Fz_L] (Z-up; col 2 = right vGRF,
     col 5 = left vGRF).  Verified by column-magnitude inspection and R/L
     anti-correlation during single-leg stance.
  2. Builds per-frame leg masks ``r_off``, ``l_off`` from vGRF.
  3. Walks every non-GRFNorm ``COP_*.npy`` file with shape (T, 6) or (T, 4) whose first
     dim matches the GRF length.  Files that don't match are reported and
     skipped.
  4. **Reads the pristine original** (``<name>_PreCOPClean.npy`` if present, or
     the legacy ``<name>_PreVGRFClean.npy`` backup, else the current file).
     Applies rules (a), (b), (c) in order to the original and writes the
     result back in place.  A one-time backup with the new suffix is created
     the first time a file is cleaned.  Subsequent runs are idempotent.
  5. Regenerates ``COP_CalcFrame_GroundAligned_GRFNorm.npy`` and
     ``COP_CalcFrame_GroundAligned_GRFNorm_noised.npy`` from the cleaned
     length-unit COP files.  These files are dimensionless
     ``(COP / height) * (|GRF| / body_weight)`` and should not receive the
     meter-valued X/Z bounds directly.

Column conventions:
    Shape (T, 6):  R = (x:0, y:1, z:2)        L = (x:3, y:4, z:5)
    Shape (T, 4):  R = (x:0, z:1)             L = (x:2, z:3)

Usage:
    python3 clean_cop_below_vgrf_threshold.py \
        --dataset Datasets_NAS/DifferentNoisedDataset/TrustedDataSetNoised12DistributedUnFiltered_Trimmed
    # add --dry_run to validate without writing
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# --- conventions discovered by probing the dataset --------------------------
GRF_VERT_R_COL = 2     # Fz_R in GRF_Cleaned.npy
GRF_VERT_L_COL = 5     # Fz_L in GRF_Cleaned.npy
GRF_FILENAME = "GRF_Cleaned.npy"
MASS_FILENAME = "Mass_kg.npy"
HEIGHT_FILENAME = "Height_m.npy"
COP_GROUND_ALIGNED = "COP_CalcFrame_GroundAligned.npy"
COP_GROUND_ALIGNED_NOISED = "COP_CalcFrame_GroundAligned_noised.npy"
COP_GRFNORM = "COP_CalcFrame_GroundAligned_GRFNorm.npy"
COP_GRFNORM_NOISED = "COP_CalcFrame_GroundAligned_GRFNorm_noised.npy"
DEFAULT_SOURCE_DIRS = ("ProcessedData", "MoCap")
# New, more general backup suffix.  We also accept the legacy suffix for files
# that were already backed up by the previous (vGRF-only) version.
BACKUP_SUFFIX = "_PreCOPClean.npy"
LEGACY_BACKUP_SUFFIX = "_PreVGRFClean.npy"
BACKUP_SUFFIXES = (BACKUP_SUFFIX, LEGACY_BACKUP_SUFFIX)


def _leg_columns(n_cols: int):
    """Return (r_cols, l_cols, r_x, r_z, l_x, l_z) for a COP array.

    r_cols / l_cols are the *all-component* column tuples (used by the vGRF
    whole-leg zero).  r_x/l_x and r_z/l_z are the single columns used by the
    X- and Z-bound component zeroes.
    """
    if n_cols == 6:
        # (x:0, y:1, z:2) | (x:3, y:4, z:5)
        return (0, 1, 2), (3, 4, 5), 0, 2, 3, 5
    if n_cols == 4:
        # (x:0, z:1) | (x:2, z:3)
        return (0, 1), (2, 3), 0, 1, 2, 3
    return None


def _is_backup(path: Path) -> bool:
    return any(path.name.endswith(s) for s in BACKUP_SUFFIXES)


def _find_backup(cop_path: Path) -> Path | None:
    """Return the existing backup for a cleaned COP file, or None.

    The current and legacy suffixes are both checked so we don't break files
    that were already backed up by the previous version of this script.
    """
    for suf in BACKUP_SUFFIXES:
        candidate = cop_path.with_name(cop_path.stem + suf)
        if candidate.exists():
            return candidate
    return None


def _is_grf_norm_cop(path: Path) -> bool:
    return "_GRFNorm" in path.stem


def _load_vector_or_none(path: Path, T: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 1:
        return np.full(T, float(arr[0]), dtype=np.float64)
    if arr.size == T:
        return arr
    return None


def _build_grf_norm_cop(cop_ground_aligned: np.ndarray,
                        grf: np.ndarray,
                        mass_kg: np.ndarray,
                        height_m: np.ndarray) -> np.ndarray:
    """Return (COP / height) * (|GRF| / BW) for 6-col ground-aligned COP."""
    cop = np.asarray(cop_ground_aligned, dtype=np.float64)
    if cop.ndim != 2 or cop.shape[1] != 6:
        raise ValueError(f"expected 6-col COP, got {cop.shape}")
    body_weight = mass_kg * 9.8067
    if np.any(~np.isfinite(body_weight)) or np.any(body_weight <= 0.0):
        raise ValueError("invalid Mass_kg body-weight factor")
    if np.any(~np.isfinite(height_m)) or np.any(height_m <= 0.0):
        raise ValueError("invalid Height_m factor")

    grf_mag_r = np.linalg.norm(grf[:, 0:3], axis=1)
    grf_mag_l = np.linalg.norm(grf[:, 3:6], axis=1)
    out = cop / height_m[:, np.newaxis]
    out[:, 0:3] *= (grf_mag_r / body_weight)[:, np.newaxis]
    out[:, 3:6] *= (grf_mag_l / body_weight)[:, np.newaxis]
    return out


def _regenerate_grf_norm_cop_files(pdir: Path,
                                   grf: np.ndarray,
                                   T: int,
                                   dry_run: bool,
                                   result: Dict[str, Any]) -> None:
    mass_kg = _load_vector_or_none(pdir / MASS_FILENAME, T)
    height_m = _load_vector_or_none(pdir / HEIGHT_FILENAME, T)
    if mass_kg is None or height_m is None:
        result["skipped"].append(
            f"GRFNorm COP regenerate (missing/invalid {MASS_FILENAME} or {HEIGHT_FILENAME})"
        )
        return

    for source_name, target_name in (
        (COP_GROUND_ALIGNED, COP_GRFNORM),
        (COP_GROUND_ALIGNED_NOISED, COP_GRFNORM_NOISED),
    ):
        source_path = pdir / source_name
        target_path = pdir / target_name
        if not source_path.exists():
            continue
        try:
            cop_source = np.load(source_path)
            regenerated = _build_grf_norm_cop(cop_source, grf, mass_kg, height_m)
        except Exception as e:  # noqa: BLE001
            result["skipped"].append(f"{target_name} (regenerate_err: {e})")
            continue

        if target_path.exists():
            try:
                current = np.load(target_path)
            except Exception:  # noqa: BLE001
                current = None
            target_dtype = current.dtype if current is not None else cop_source.dtype
            if current is not None and np.array_equal(regenerated.astype(target_dtype, copy=False), current):
                result["skipped"].append(f"{target_name} (already clean)")
                continue
        else:
            current = None
            target_dtype = cop_source.dtype

        if not dry_run:
            bak = _find_backup(target_path)
            if target_path.exists() and bak is None:
                try:
                    shutil.copy2(target_path, target_path.with_name(target_path.stem + BACKUP_SUFFIX))
                except OSError as e:
                    result["skipped"].append(f"{target_name} (backup_write_err: {e})")
                    continue
            try:
                np.save(target_path, regenerated.astype(target_dtype, copy=False))
            except OSError as e:
                result["skipped"].append(f"{target_name} (write_err: {e})")
                continue

        action = "regenerated" if target_path.exists() else "created"
        result["cleaned"].append(f"{pdir.name}/{target_name}  shape={regenerated.shape}  {action}_from={source_name}")


def process_source_dir(source_dir: Path,
                       vgrf_threshold: float,
                       cop_x_min: float,
                       cop_x_max: float,
                       cop_z_abs_max: float,
                       dry_run: bool) -> Dict[str, Any]:
    """Clean every matching COP file inside one source directory."""
    pdir = source_dir
    result: Dict[str, Any] = {
        "trial": str(source_dir.parent),
        "source_dir": source_dir.name,
        "status": "ok",
        "cleaned": [],
        "skipped": [],
        "note": "",
    }
    if not pdir.is_dir():
        result["status"] = "skip"
        result["note"] = f"no {source_dir.name}/"
        return result

    grf_path = pdir / GRF_FILENAME
    if not grf_path.exists():
        result["status"] = "skip"
        result["note"] = f"no {GRF_FILENAME}"
        return result

    try:
        grf = np.load(grf_path)
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["note"] = f"GRF load failed: {e}"
        return result

    if grf.ndim != 2 or grf.shape[1] < max(GRF_VERT_R_COL, GRF_VERT_L_COL) + 1:
        result["status"] = "error"
        result["note"] = f"unexpected GRF shape {grf.shape}"
        return result

    T = grf.shape[0]
    r_off = grf[:, GRF_VERT_R_COL] < vgrf_threshold
    l_off = grf[:, GRF_VERT_L_COL] < vgrf_threshold

    cop_files = sorted(
        p for p in pdir.glob("COP_*.npy")
        if not _is_backup(p) and not _is_grf_norm_cop(p)
    )

    for cop_path in cop_files:
        # Always start from the pristine original if we have a backup, so all
        # three rules are applied to the unmodified data each run (idempotent
        # even when only one rule changes between runs).
        bak = _find_backup(cop_path)
        try:
            source = bak if bak is not None else cop_path
            arr = np.load(source)
        except Exception as e:  # noqa: BLE001
            result["skipped"].append(f"{cop_path.name} (load_err: {e})")
            continue

        if arr.ndim != 2 or arr.shape[0] != T:
            result["skipped"].append(
                f"{cop_path.name} (shape={arr.shape} != ({T}, ?))"
            )
            continue

        layout = _leg_columns(arr.shape[1])
        if layout is None:
            result["skipped"].append(
                f"{cop_path.name} (unrecognised n_cols={arr.shape[1]})"
            )
            continue
        r_cols, l_cols, r_x, r_z, l_x, l_z = layout

        cleaned = arr.copy()

        # --- (a) vGRF whole-leg zero -------------------------------------
        cleaned[np.ix_(r_off, r_cols)] = 0.0
        cleaned[np.ix_(l_off, l_cols)] = 0.0
        n_vgrf_r = int(r_off.sum())
        n_vgrf_l = int(l_off.sum())

        # --- (b) X-bound, component-only ---------------------------------
        rx_bad = (cleaned[:, r_x] < cop_x_min) | (cleaned[:, r_x] > cop_x_max)
        lx_bad = (cleaned[:, l_x] < cop_x_min) | (cleaned[:, l_x] > cop_x_max)
        # Don't re-count frames that are already vGRF-zeroed (their X is 0,
        # which sits inside [x_min, x_max] anyway, so this is just for accounting).
        cleaned[rx_bad, r_x] = 0.0
        cleaned[lx_bad, l_x] = 0.0
        n_x_r = int(rx_bad.sum())
        n_x_l = int(lx_bad.sum())

        # --- (c) Z-bound, component-only ---------------------------------
        rz_bad = np.abs(cleaned[:, r_z]) > cop_z_abs_max
        lz_bad = np.abs(cleaned[:, l_z]) > cop_z_abs_max
        cleaned[rz_bad, r_z] = 0.0
        cleaned[lz_bad, l_z] = 0.0
        n_z_r = int(rz_bad.sum())
        n_z_l = int(lz_bad.sum())

        # Idempotent: skip write when result already matches what's on disk.
        try:
            current_on_disk = np.load(cop_path)
        except Exception:  # noqa: BLE001
            current_on_disk = None
        if current_on_disk is not None and np.array_equal(cleaned, current_on_disk):
            result["skipped"].append(f"{cop_path.name} (already clean)")
            continue

        if not dry_run:
            if bak is None:
                # First-time backup of the truly pristine file.
                target_bak = cop_path.with_name(cop_path.stem + BACKUP_SUFFIX)
                try:
                    shutil.copy2(cop_path, target_bak)
                except OSError as e:
                    result["skipped"].append(f"{cop_path.name} (backup_write_err: {e})")
                    continue
            try:
                np.save(cop_path, cleaned.astype(arr.dtype, copy=False))
            except OSError as e:
                result["skipped"].append(f"{cop_path.name} (write_err: {e})")
                continue

        result["cleaned"].append(
            f"{pdir.name}/{cop_path.name}  shape={arr.shape}  "
            f"vGRF(R={n_vgrf_r},L={n_vgrf_l})  "
            f"X_oob(R={n_x_r},L={n_x_l})  "
            f"|Z|_oob(R={n_z_r},L={n_z_l})"
        )

    _regenerate_grf_norm_cop_files(pdir, grf, T, dry_run, result)

    return result


def process_trial(trial_path: Path,
                  source_dir_names: tuple[str, ...],
                  vgrf_threshold: float,
                  cop_x_min: float,
                  cop_x_max: float,
                  cop_z_abs_max: float,
                  dry_run: bool) -> Dict[str, Any]:
    """Clean all selected source directories inside one trial."""
    combined: Dict[str, Any] = {
        "trial": str(trial_path),
        "status": "ok",
        "cleaned": [],
        "skipped": [],
        "note": "",
    }
    attempted = 0
    ok_sources = 0
    errored_sources = 0
    for source_name in source_dir_names:
        source_dir = trial_path / source_name
        if not source_dir.is_dir():
            continue
        attempted += 1
        result = process_source_dir(
            source_dir,
            vgrf_threshold=vgrf_threshold,
            cop_x_min=cop_x_min,
            cop_x_max=cop_x_max,
            cop_z_abs_max=cop_z_abs_max,
            dry_run=dry_run,
        )
        combined["cleaned"].extend(result.get("cleaned", []))
        combined["skipped"].extend(
            f"{source_name}/{msg}" for msg in result.get("skipped", [])
        )
        if result.get("status") == "error":
            errored_sources += 1
            combined["skipped"].append(f"{source_name} ({result.get('note', 'error')})")
        elif result.get("status") == "ok":
            ok_sources += 1

    if attempted == 0:
        combined["status"] = "skip"
        combined["note"] = f"none of source dirs present: {', '.join(source_dir_names)}"
    elif errored_sources and ok_sources == 0:
        combined["status"] = "error"
        combined["note"] = "all source dirs errored"
    return combined


def _iter_trials(dataset_root: Path, source_dir_names: tuple[str, ...]) -> List[Path]:
    trials: List[Path] = []
    seen: set[Path] = set()
    for trial_dir in sorted(
        p for p in dataset_root.rglob("Trial_*")
        if p.is_dir() and any((p / name).is_dir() for name in source_dir_names)
    ):
        resolved = trial_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        trials.append(trial_dir)
    return trials


def main() -> None:
    default_dataset = (
        "Datasets_NAS/DifferentNoisedDataset/"
        "TrustedDataSetNoised12DistributedUnFiltered_Trimmed"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=default_dataset,
                        help="Dataset root containing <subject>/<Trial_*> source folders")
    parser.add_argument("--source_dirs", default=",".join(DEFAULT_SOURCE_DIRS),
                        help="Comma-separated per-trial source directories to clean "
                             f"(default {','.join(DEFAULT_SOURCE_DIRS)})")
    parser.add_argument("--threshold", "--vgrf_threshold", dest="vgrf_threshold",
                        type=float, default=1.0,
                        help="vGRF threshold in N below which whole-leg COP is zeroed "
                             "(default 1.0)")
    parser.add_argument("--cop_x_min", type=float, default=-0.10,
                        help="COP X lower bound (m, foot frame); X below this is zeroed "
                             "component-only (default -0.10)")
    parser.add_argument("--cop_x_max", type=float, default=0.35,
                        help="COP X upper bound (m, foot frame); X above this is zeroed "
                             "component-only (default +0.35)")
    parser.add_argument("--cop_z_abs_max", type=float, default=0.15,
                        help="COP Z magnitude bound (m, foot frame); |Z| above this is "
                             "zeroed component-only (default 0.15)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker processes (default 8)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would change without writing")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-file cleaning details")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = Path(__file__).resolve().parent / dataset_root
    if not dataset_root.is_dir():
        print(f"ERROR: dataset not found: {dataset_root}")
        sys.exit(1)

    source_dir_names = tuple(
        name.strip() for name in str(args.source_dirs).split(",") if name.strip()
    )
    if not source_dir_names:
        print("ERROR: --source_dirs must name at least one source directory")
        sys.exit(1)

    trials = _iter_trials(dataset_root, source_dir_names)
    if not trials:
        print(
            f"No trials discovered under {dataset_root} with source dirs: "
            f"{', '.join(source_dir_names)}"
        )
        sys.exit(1)

    total = len(trials)
    print(f"{'DRY RUN — ' if args.dry_run else ''}"
          f"Cleaning {total} trials under {dataset_root}\n"
          f"  source dirs    = {', '.join(source_dir_names)}\n"
          f"  vGRF threshold = {args.vgrf_threshold} N\n"
          f"  COP X bounds   = [{args.cop_x_min:+.3f}, {args.cop_x_max:+.3f}] m\n"
          f"  COP |Z| bound  = {args.cop_z_abs_max:.3f} m\n"
          f"  workers        = {args.workers}\n")

    ok = skipped_trials = errored = 0
    total_files_cleaned = 0
    total_files_skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_trial, tp,
                      source_dir_names,
                      args.vgrf_threshold,
                      args.cop_x_min, args.cop_x_max,
                      args.cop_z_abs_max,
                      args.dry_run): tp
            for tp in trials
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            tag = r["status"].upper()
            trial_name = "/".join(Path(r["trial"]).parts[-2:])
            n_clean = len(r["cleaned"])
            n_skip = len(r["skipped"])
            total_files_cleaned += n_clean
            total_files_skipped += n_skip
            note = f"  [{r['note']}]" if r["note"] else ""
            summary = f" cleaned={n_clean} skipped={n_skip}"
            print(f"  [{i:4d}/{total}] {tag:<5} {trial_name}{summary}{note}")
            if args.verbose:
                for line in r["cleaned"]:
                    print(f"      ✓ {line}")
                for line in r["skipped"]:
                    print(f"      · {line}")
            if r["status"] == "ok":
                ok += 1
            elif r["status"] == "skip":
                skipped_trials += 1
            else:
                errored += 1

    print(
        f"\nDone — trials ok={ok} skipped={skipped_trials} errored={errored}; "
        f"files cleaned={total_files_cleaned} skipped={total_files_skipped}"
    )
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
