#!/usr/bin/env python3
"""
stage_footopt_dataset.py

Stage the OpenCapWalkingTrunkSwaySubjects dataset into the folder layout that
FootContactOptimizer / gait_analysis / utilsKinematics_GRF hard-code, so the
foot-contact optimizer can run on the *video* markers.

Source per subject (OpenCapWalkingTrunkSwaySubjects/<subject>/):
    OpenSimScaled_Video.osim                                (scaled video model)
    trial_<n>/Video/Motion/Raw/<stem>_video.trc            (video markers)
    trial_<n>/Video/Motion/Raw/<stem>.mot                  (IK Coordinates)
      where <stem> is e.g. "walking1" (normal) or "walkingTS1" (trunk-sway).

Staged per subject (<out>/<subject>/  ==  one OpenCap "session_dir"):
    OpenSimData/Model/LaiUhlrich2022_scaled.osim           (<- OpenSimScaled_Video.osim, renamed
                                                              to the name utilsKinematics_GRF wants)
    OpenSimData/Kinematics/<stem>.mot                      (<- trial_<n> .mot)
    MarkerData/<stem>.trc                                  (<- <stem>_video.trc, suffix dropped so
                                                              the .trc / .mot stems match)

Session dir name == source subject folder name, and each staged <stem> == the
source stem, so the staged data trivially coordinates back to the original.
After running the optimizer, refined markers land in:
    <out>/<subject>/ForGaitDynamics/MarkerData_optfeet_<stem>.trc

Provenance (for re-running IK and copying results back into the original dataset):
    <out>/<subject>/staging_manifest.json   (per subject)
    <out>/staging_index.json                (whole dataset)
Each trial entry records the original files, the staged files, the expected
optimizer output, and `copy_back_dir` = the original trial_<n>/Video/Motion/Raw
folder to copy re-run IK / optimized markers back into.

Copies only (never moves); safe to re-run (idempotent).
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# This script lives in TRC_FootOptimization/; the datasets live at the repo root.
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SOURCE = REPO_ROOT / "OpenCapWalkingTrunkSwaySubjects"
DEFAULT_OUTPUT = REPO_ROOT / "OpenCapFootOptStaging"

# utilsKinematics_GRF.py hard-codes this model filename (modelName = 'LaiUhlrich2022_scaled').
STAGED_MODEL_NAME = "LaiUhlrich2022_scaled.osim"
# Source scaled video model lives at <subject>/*_[Vv]ideo.osim (OpenSimScaled_Video.osim).
SOURCE_MODEL_GLOB = "*_[Vv]ideo.osim"
# Video marker files: <stem>_video.trc under trial_<n>/Video/Motion/Raw/.
VIDEO_TRC_SUFFIX = "_video.trc"


def _copy(src: Path, dst: Path, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"    [dry-run] {src}  ->  {dst}")
        return
    shutil.copy2(src, dst)


def stage_subject(subject_dir: Path, out_root: Path, dry_run: bool) -> dict:
    """Stage one subject; return its manifest dict."""
    subject = subject_dir.name
    session_dir = out_root / subject

    # --- Locate the scaled video model -----------------------------------
    models = sorted(subject_dir.glob(SOURCE_MODEL_GLOB))
    if not models:
        raise FileNotFoundError(f"No '{SOURCE_MODEL_GLOB}' model in {subject_dir}")
    model_src = models[0]
    model_dst = session_dir / "OpenSimData" / "Model" / STAGED_MODEL_NAME

    # --- Find every video .trc + its paired .mot -------------------------
    trc_sources = sorted(subject_dir.glob(f"trial_*/Video/Motion/Raw/*{VIDEO_TRC_SUFFIX}"))
    if not trc_sources:
        raise FileNotFoundError(f"No '*{VIDEO_TRC_SUFFIX}' files under {subject_dir}/trial_*/Video/Motion/Raw")

    trials = []
    for trc_src in trc_sources:
        stem = trc_src.name[: -len(VIDEO_TRC_SUFFIX)]      # walking1  /  walkingTS1
        raw_dir = trc_src.parent                            # trial_<n>/Video/Motion/Raw
        trial_folder = _trial_folder_of(trc_src, subject_dir)
        mot_src = raw_dir / f"{stem}.mot"
        if not mot_src.exists():
            raise FileNotFoundError(f"Missing paired IK .mot for {trc_src}: expected {mot_src}")

        trc_dst = session_dir / "MarkerData" / f"{stem}.trc"
        mot_dst = session_dir / "OpenSimData" / "Kinematics" / f"{stem}.mot"

        _copy(trc_src, trc_dst, dry_run)
        _copy(mot_src, mot_dst, dry_run)

        trials.append({
            "stem": stem,
            "trial_folder": trial_folder,
            "source": {
                "video_trc": str(trc_src),
                "ik_mot": str(mot_src),
                "trial_raw_dir": str(raw_dir),
            },
            "staged": {
                "marker_trc": str(trc_dst),
                "kinematics_mot": str(mot_dst),
            },
            # Where the optimizer will write the refined markers:
            "optimized_trc": str(session_dir / "ForGaitDynamics" / f"MarkerData_optfeet_{stem}.trc"),
            # Where to copy re-run IK / optimized results back into the ORIGINAL dataset:
            "copy_back_dir": str(raw_dir),
        })

    _copy(model_src, model_dst, dry_run)

    manifest = {
        "subject": subject,
        "session_dir": str(session_dir),
        "source_subject_dir": str(subject_dir),
        "model": {"source": str(model_src), "staged": str(model_dst)},
        "marker_type": "video",
        "n_trials": len(trials),
        "trials": trials,
    }

    if not dry_run:
        (session_dir).mkdir(parents=True, exist_ok=True)
        with open(session_dir / "staging_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    return manifest


def _trial_folder_of(path: Path, subject_dir: Path) -> str:
    """Return the 'trial_<n>' component of a path under the subject dir."""
    rel = path.relative_to(subject_dir)
    return rel.parts[0]  # trial_<n>


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help=f"Source dataset root (default: {DEFAULT_SOURCE})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Staged dataset root (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--dry-run", action="store_true", help="Print planned copies without writing.")
    args = ap.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")

    subject_dirs = sorted(p for p in source.iterdir()
                          if p.is_dir() and (p / "OpenSimScaled_Video.osim").exists())
    if not subject_dirs:
        raise SystemExit(f"No subject folders with OpenSimScaled_Video.osim under {source}")

    print(f"Source : {source}")
    print(f"Output : {output}")
    print(f"Subjects: {len(subject_dirs)}\n")

    index = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source),
        "output_root": str(output),
        "staged_model_name": STAGED_MODEL_NAME,
        "marker_type": "video",
        "subjects": {},
    }

    total_trials = 0
    for sd in subject_dirs:
        print(f"[{sd.name}]")
        m = stage_subject(sd, output, args.dry_run)
        total_trials += m["n_trials"]
        index["subjects"][sd.name] = {
            "session_dir": m["session_dir"],
            "source_subject_dir": m["source_subject_dir"],
            "trials": [
                {"stem": t["stem"], "trial_folder": t["trial_folder"],
                 "copy_back_dir": t["copy_back_dir"], "optimized_trc": t["optimized_trc"]}
                for t in m["trials"]
            ],
        }
        print(f"    staged {m['n_trials']} trial(s): {', '.join(t['stem'] for t in m['trials'])}")

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "staging_index.json", "w") as f:
            json.dump(index, f, indent=2)

    print(f"\nDone. {len(subject_dirs)} subjects, {total_trials} trials"
          f"{' (dry-run, nothing written)' if args.dry_run else ''}.")
    if not args.dry_run:
        print(f"Index: {output / 'staging_index.json'}")
        print("\nRun the optimizer per session, e.g.:")
        print("  from FootContactOptimizer import refine_foot_kinematics_for_session")
        print(f"  refine_foot_kinematics_for_session('{output}/subject2',")
        print("      trial_prefix='', gait_style='overground', trimming_start=0.5, trimming_end=0.5)")


if __name__ == "__main__":
    main()
