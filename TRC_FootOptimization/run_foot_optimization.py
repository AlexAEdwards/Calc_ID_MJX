#!/usr/bin/env python3
"""
run_foot_optimization.py

Batch-run the foot-contact optimizer over the staged dataset
(OpenCapFootOptStaging/ by default), one session per subject.

For each session it clears ForGaitDynamics/ (so re-runs keep canonical names,
no "_dup" files), optimizes every <stem>.trc in MarkerData/, verifies the result
(no-trim length + a real vertical offset vs. a fallback), and writes a report.

Defaults to NO trimming (trimming_start = trimming_end = 0).

Examples
--------
# All subjects / trials, no trimming (recommended):
python3 TRC_FootOptimization/run_foot_optimization.py

# Just a couple of subjects:
python3 TRC_FootOptimization/run_foot_optimization.py --subjects subject2 subject3

# Parallel (independent trials), e.g. 4 at once:
python3 TRC_FootOptimization/run_foot_optimization.py --workers 4
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))          # make the optimizer + utils importable
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "OpenCapFootOptStaging"


def _optimize_one(task: dict) -> dict:
    """Optimize a single TRC. Runs in its own process when --workers > 1."""
    # Keep each worker single-threaded so parallel jobs don't oversubscribe cores.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import numpy as np
    sys.path.insert(0, task["script_dir"])
    from FootContactOptimizer import (
        refine_foot_kinematics_trc, get_trc_frame_rate,
        TRCload, extract_marker_names, filter_markers_by_mapping_keys,
    )

    trc_path = task["trc_path"]
    rec = {"subject": task["subject"], "stem": task["stem"], "trc": trc_path}
    t0 = time.time()
    try:
        fr = get_trc_frame_rate(trc_path)
        out_path = refine_foot_kinematics_trc(
            trc_path=trc_path,
            session_dir=task["session_dir"],
            save_dir=task["save_dir"],
            trimming_start=task["trimming_start"],
            trimming_end=task["trimming_end"],
            gait_style=task["gait_style"],
            frame_rate=fr,
            side=task["side"],
            filter_markers_on_save=task["filter_markers"],
        )
        rec["runtime_s"] = round(time.time() - t0, 1)
        rec["frame_rate"] = fr
        rec["output"] = out_path

        # --- verify: length + real offset vs. fallback ---
        _, din, _ = TRCload(trc_path); din = din[:, ~np.all(np.isnan(din), axis=0)]
        hin = TRCload(trc_path)[0]
        _, dout, _ = TRCload(out_path); dout = dout[:, ~np.all(np.isnan(dout), axis=0)]
        n_in, n_out = din.shape[0], dout.shape[0]
        rec["frames_in"], rec["frames_out"] = n_in, n_out
        rec["trimmed"] = bool(n_out < n_in)

        offset_ok, omin, omax = False, None, None
        if n_out == n_in:  # only diff when lengths match (no trimming)
            mnames = [m for m in extract_marker_names(hin) if m not in ("Frame#", "Time")]
            mf, xf = filter_markers_by_mapping_keys(mnames, din[:, 2:])
            yin = xf[:, 1::3]
            yout = dout[:, 2:][:, 1::3]
            m = min(yin.shape[1], yout.shape[1])
            d = yout[:, :m] - yin[:, :m]
            per_frame = np.nanmean(d, axis=1)
            omin, omax = float(np.nanmin(per_frame)), float(np.nanmax(per_frame))
            offset_ok = bool(np.nanmax(np.abs(per_frame)) > 1e-3)
        rec["offset_min_mm"] = None if omin is None else round(omin, 3)
        rec["offset_max_mm"] = None if omax is None else round(omax, 3)

        if rec["trimmed"]:
            rec["status"] = "optimized_trimmed"   # gait auto-retry added trimming
        elif offset_ok:
            rec["status"] = "optimized"
        else:
            rec["status"] = "no_change"           # gait-detect or optimizer fallback (unmodified)
    except Exception as e:  # noqa: BLE001 - report, don't crash the batch
        rec["runtime_s"] = round(time.time() - t0, 1)
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                    help=f"Staged dataset root (default: {DEFAULT_OUTPUT_ROOT})")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Only these session/subject folders (default: all).")
    ap.add_argument("--trimming-start", type=float, default=0.0, help="Seconds trimmed at start (default 0).")
    ap.add_argument("--trimming-end", type=float, default=0.0, help="Seconds trimmed at end (default 0).")
    ap.add_argument("--gait-style", default="overground", choices=["overground", "treadmill"])
    ap.add_argument("--side", default="l", choices=["l", "r"], help="Leg used for gait-event detection.")
    ap.add_argument("--no-marker-filter", action="store_true",
                    help="Keep all markers instead of the standard mapped subset.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel trials (default 1).")
    args = ap.parse_args()

    root = args.output_root.resolve()
    sessions = sorted(p for p in root.iterdir()
                      if p.is_dir() and (p / "MarkerData").is_dir()
                      and (args.subjects is None or p.name in args.subjects))
    if not sessions:
        raise SystemExit(f"No sessions with MarkerData/ under {root}"
                         + (f" matching {args.subjects}" if args.subjects else ""))

    # Build task list; clear each ForGaitDynamics once so re-runs stay canonical.
    tasks = []
    for sess in sessions:
        save_dir = sess / "ForGaitDynamics"
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for trc in sorted((sess / "MarkerData").glob("*.trc")):
            tasks.append({
                "script_dir": str(SCRIPT_DIR),
                "subject": sess.name,
                "stem": trc.stem,
                "trc_path": str(trc),
                "session_dir": str(sess),
                "save_dir": str(save_dir),
                "trimming_start": args.trimming_start,
                "trimming_end": args.trimming_end,
                "gait_style": args.gait_style,
                "side": args.side,
                "filter_markers": not args.no_marker_filter,
            })

    print(f"Sessions: {len(sessions)} | Trials: {len(tasks)} | trimming=({args.trimming_start},{args.trimming_end}) "
          f"| gait={args.gait_style} | workers={args.workers}\n")

    results = []
    def _log(i, rec):
        extra = ""
        if rec["status"].startswith("optimized") and rec.get("offset_max_mm") is not None:
            extra = f"  dY[{rec['offset_min_mm']:+.1f},{rec['offset_max_mm']:+.1f}]mm  {rec['frames_out']}/{rec['frames_in']}f"
        elif rec["status"] == "error":
            extra = f"  {rec.get('error','')}"
        print(f"[{i:>2}/{len(tasks)}] {rec['subject']:<14} {rec['stem']:<12} {rec['status']:<18} {rec.get('runtime_s',0):>5.1f}s{extra}")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_optimize_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result(); results.append(rec); _log(i, rec)
    else:
        for i, t in enumerate(tasks, 1):
            rec = _optimize_one(t); results.append(rec); _log(i, rec)

    results.sort(key=lambda r: (r["subject"], r["stem"]))
    counts: dict = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    report_path = root / "foot_opt_report.json"
    with open(report_path, "w") as f:
        json.dump({"output_root": str(root),
                   "params": {"trimming_start": args.trimming_start, "trimming_end": args.trimming_end,
                              "gait_style": args.gait_style, "side": args.side,
                              "marker_filter": not args.no_marker_filter},
                   "counts": counts, "trials": results}, f, indent=2)

    print("\n=== Summary ===")
    for k in sorted(counts):
        print(f"  {k:<18} {counts[k]}")
    print(f"\nReport: {report_path}")
    if any(r["status"] in ("no_change", "error", "optimized_trimmed") for r in results):
        print("Review non-'optimized' trials in the report (may need trimming/side tweaks).")


if __name__ == "__main__":
    main()
