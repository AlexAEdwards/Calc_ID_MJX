#!/usr/bin/env python3
"""
run_ik.py

OpenSim Inverse Kinematics via the API.

- run_opensim_ik(...)         : generic, model/TRC-agnostic. Give it a model, a
                                marker TRC whose marker names already match the
                                model, and an output .mot path. Reusable anywhere.
- run_ik_footopt_dataset(...) : wrapper for the OpenCapFootOptStaging dataset.
                                Auto-remaps the optimizer's clean marker names back
                                to the model's `_study` names, runs IK on every
                                optimized TRC, writes the .mot into the staged
                                session AND copies it back into the original dataset
                                as <stem>_opt.mot.

CLI:
    python3 TRC_FootOptimization/run_ik.py                     # all subjects
    python3 TRC_FootOptimization/run_ik.py --subjects subject2 # subset
    python3 TRC_FootOptimization/run_ik.py --workers 16        # parallel
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))          # sibling modules (FootContactOptimizer, utils)
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "OpenCapFootOptStaging"
DEFAULT_MODEL_NAME = "LaiUhlrich2022_scaled.osim"
STAGED_IK_SUBDIR = os.path.join("OpenSimData", "Kinematics_FootOpt")


# =============================================================================
# Generic IK
# =============================================================================
def _parse_marker_errors(sto_path):
    """Return (mean RMS, max) marker error in mm from an IK *_ik_marker_errors.sto."""
    import numpy as np
    with open(sto_path) as f:
        lines = f.readlines()
    hdr_end = next((i for i, l in enumerate(lines) if l.strip().lower() == "endheader"), -1)
    start = hdr_end + 1
    header = lines[start].strip().split("\t")
    rows = [l.strip().split("\t") for l in lines[start + 1:] if l.strip()]
    if not rows:
        return None, None
    data = np.array(rows, dtype=float)

    def col(key):
        return next((i for i, h in enumerate(header) if key in h.lower()), None)

    rms_i, max_i = col("rms"), col("max")
    # OpenSim reports marker errors in model length units (m) -> convert to mm.
    rms = round(float(np.nanmean(data[:, rms_i])) * 1000.0, 2) if rms_i is not None else None
    mx = round(float(np.nanmax(data[:, max_i])) * 1000.0, 2) if max_i is not None else None
    return rms, mx


def run_opensim_ik(model_path, marker_trc_path, output_mot_path,
                   start_time=None, end_time=None, marker_weights=None,
                   setup_xml=None, report_errors=True):
    """
    Run OpenSim IK for a single trial.

    Assumes marker names in `marker_trc_path` already match markers in the model
    (the wrapper handles any renaming). Markers common to model and TRC are used,
    each with weight `marker_weights[name]` (default 1.0). If `setup_xml` is given,
    that IK setup's own IKTaskSet is used instead.

    Returns a QC dict: output_mot, n_markers_used, rms_error_mm, max_error_mm, start, end.
    """
    import opensim

    model = opensim.Model(str(model_path))
    model.initSystem()
    ms = model.getMarkerSet()
    model_markers = {ms.get(i).getName() for i in range(ms.getSize())}

    table = opensim.TimeSeriesTableVec3(str(marker_trc_path))
    tcol = table.getIndependentColumn()
    trc_markers = set(table.getColumnLabels())
    if start_time is None:
        start_time = float(tcol[0])
    if end_time is None:
        end_time = float(tcol[-1])

    output_mot_path = str(output_mot_path)
    out_dir = os.path.dirname(os.path.abspath(output_mot_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(output_mot_path))[0]

    ik = opensim.InverseKinematicsTool(str(setup_xml)) if setup_xml else opensim.InverseKinematicsTool()
    ik.setModel(model)
    ik.setName(name)
    ik.setMarkerDataFileName(str(marker_trc_path))
    ik.setStartTime(float(start_time))
    ik.setEndTime(float(end_time))
    ik.setOutputMotionFileName(output_mot_path)
    ik.setResultsDir(out_dir)
    ik.set_report_errors(bool(report_errors))

    common = sorted(model_markers & trc_markers)
    if not setup_xml:
        taskset = ik.getIKTaskSet()
        weights = marker_weights or {}
        for mk in common:
            task = opensim.IKMarkerTask()
            task.setName(mk)
            task.setApply(True)
            task.setWeight(float(weights.get(mk, 1.0)))
            taskset.cloneAndAppend(task)

    ik.run()

    rms = mx = None
    err_sto = os.path.join(out_dir, name + "_ik_marker_errors.sto")
    if report_errors and os.path.exists(err_sto):
        rms, mx = _parse_marker_errors(err_sto)

    return {
        "output_mot": output_mot_path,
        "n_markers_used": len(common),
        "n_model_markers": len(model_markers),
        "n_trc_markers": len(trc_markers),
        "rms_error_mm": rms,
        "max_error_mm": mx,
        "start": float(start_time),
        "end": float(end_time),
    }


# =============================================================================
# OpenCapFootOptStaging wrapper
# =============================================================================
def _inverse_marker_map():
    """clean-name -> model `_study` name (inverse of the optimizer's rename map)."""
    from FootContactOptimizer import DEFAULT_MARKER_MAPPING
    return {clean: study for study, clean in DEFAULT_MARKER_MAPPING.items()}


def _remap_trc_to_model_names(trc_path, out_trc_path, inverse_map):
    """Rewrite a TRC with marker names remapped to the model's names (idempotent)."""
    import numpy as np
    from FootContactOptimizer import TRCload, extract_marker_names, write_trc_file

    header, data, _ = TRCload(str(trc_path))
    data = data[:, ~np.all(np.isnan(data), axis=0)]
    names = [n for n in extract_marker_names(header) if n not in ("Frame#", "Time")]
    new_names = [inverse_map.get(n, n) for n in names]         # already-_study names pass through
    time_col = data[:, 1]
    mrkdata = data[:, 2:]
    out_dir = os.path.dirname(os.path.abspath(out_trc_path))
    base = os.path.splitext(os.path.basename(out_trc_path))[0]
    write_trc_file(time_col, mrkdata, new_names, out_dir, base)
    return os.path.join(out_dir, base + ".trc")


def _ik_one(task: dict) -> dict:
    """Run IK for one optimized TRC (own process when --workers > 1)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    sys.path.insert(0, task["script_dir"])

    rec = {"subject": task["subject"], "stem": task["stem"], "trc": task["trc_path"]}
    t0 = time.time()
    try:
        inverse_map = _inverse_marker_map()
        with tempfile.TemporaryDirectory() as tmp:
            ready_trc = _remap_trc_to_model_names(
                task["trc_path"], os.path.join(tmp, f"{task['stem']}_ikready.trc"), inverse_map)
            res = run_opensim_ik(
                model_path=task["model_path"],
                marker_trc_path=ready_trc,
                output_mot_path=task["staged_mot"],
                marker_weights=None,          # uniform weight 1.0
                report_errors=True,
            )

        # OpenCap-standard post-processing (best-effort).
        if task["reorder"]:
            try:
                from utils import reorder_ik_mot_to_opensim_standard
                reorder_ik_mot_to_opensim_standard(task["staged_mot"])
            except Exception as e:  # noqa: BLE001
                rec["reorder_warning"] = f"{type(e).__name__}: {e}"
        if task["lowpass_pelvis"]:
            try:
                from utils import lowpass_filter_pelvis_ty
                lowpass_filter_pelvis_ty(task["staged_mot"])
            except Exception as e:  # noqa: BLE001
                rec["lowpass_warning"] = f"{type(e).__name__}: {e}"

        # Copy back into the original dataset as <stem>_opt.mot.
        copied_to = None
        if task["copy_back"] and task["copy_back_dir"]:
            os.makedirs(task["copy_back_dir"], exist_ok=True)
            copied_to = os.path.join(task["copy_back_dir"], f"{task['stem']}_opt.mot")
            shutil.copy2(task["staged_mot"], copied_to)

        rec.update(res)
        rec["staged_mot"] = task["staged_mot"]
        rec["copied_to"] = copied_to
        rec["runtime_s"] = round(time.time() - t0, 1)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - report, don't crash the batch
        rec["runtime_s"] = round(time.time() - t0, 1)
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def _load_copy_back_map(session_dir: Path) -> dict:
    """stem -> copy_back_dir, from the session's staging_manifest.json (if present)."""
    mf = session_dir / "staging_manifest.json"
    if not mf.exists():
        return {}
    data = json.loads(mf.read_text())
    return {t["stem"]: t.get("copy_back_dir") for t in data.get("trials", [])}


def run_ik_footopt_dataset(dataset_root=DEFAULT_DATASET_ROOT, subjects=None,
                           model_name=DEFAULT_MODEL_NAME, copy_back=True,
                           reorder=True, lowpass_pelvis=False, workers=1):
    root = Path(dataset_root).resolve()
    sessions = sorted(p for p in root.iterdir()
                      if p.is_dir() and (p / "ForGaitDynamics").is_dir()
                      and (subjects is None or p.name in subjects))
    if not sessions:
        raise SystemExit(f"No sessions with ForGaitDynamics/ under {root}"
                         + (f" matching {subjects}" if subjects else ""))

    tasks = []
    for sess in sessions:
        model_path = sess / "OpenSimData" / "Model" / model_name
        if not model_path.exists():
            print(f"  [skip] {sess.name}: model not found ({model_path})")
            continue
        staged_ik_dir = sess / STAGED_IK_SUBDIR
        if staged_ik_dir.exists():
            shutil.rmtree(staged_ik_dir)
        staged_ik_dir.mkdir(parents=True, exist_ok=True)
        cb_map = _load_copy_back_map(sess)

        for trc in sorted((sess / "ForGaitDynamics").glob("*.trc")):
            stem = trc.stem.replace("MarkerData_optfeet_", "")
            tasks.append({
                "script_dir": str(SCRIPT_DIR),
                "subject": sess.name,
                "stem": stem,
                "trc_path": str(trc),
                "model_path": str(model_path),
                "staged_mot": str(staged_ik_dir / f"{stem}.mot"),
                "copy_back": copy_back,
                "copy_back_dir": cb_map.get(stem),
                "reorder": reorder,
                "lowpass_pelvis": lowpass_pelvis,
            })

    print(f"Sessions: {len(sessions)} | Trials: {len(tasks)} | model={model_name} "
          f"| copy_back={copy_back} | workers={workers}\n")

    results = []

    def _log(i, rec):
        if rec["status"] == "ok":
            extra = f"  rms={rec.get('rms_error_mm')}mm max={rec.get('max_error_mm')}mm  markers={rec.get('n_markers_used')}"
            if rec.get("copied_to"):
                extra += "  +copied"
        else:
            extra = f"  {rec.get('error', '')}"
        print(f"[{i:>2}/{len(tasks)}] {rec['subject']:<14} {rec['stem']:<12} {rec['status']:<6} {rec.get('runtime_s', 0):>5.1f}s{extra}")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_ik_one, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result(); results.append(rec); _log(i, rec)
    else:
        for i, t in enumerate(tasks, 1):
            rec = _ik_one(t); results.append(rec); _log(i, rec)

    results.sort(key=lambda r: (r["subject"], r["stem"]))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    report = root / "ik_report.json"
    report.write_text(json.dumps(
        {"dataset_root": str(root), "model_name": model_name,
         "params": {"copy_back": copy_back, "reorder": reorder, "lowpass_pelvis": lowpass_pelvis,
                    "marker_weights": "uniform_1.0"},
         "counts": counts, "trials": results}, indent=2))

    ok = [r for r in results if r["status"] == "ok" and r.get("rms_error_mm") is not None]
    print("\n=== Summary ===")
    for k in sorted(counts):
        print(f"  {k:<6} {counts[k]}")
    if ok:
        rms_vals = [r["rms_error_mm"] for r in ok]
        print(f"  marker RMS error: mean={sum(rms_vals)/len(rms_vals):.2f}mm  "
              f"min={min(rms_vals):.2f}  max={max(rms_vals):.2f}")
    print(f"\nReport: {report}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--subjects", nargs="*", default=None, help="Only these session folders.")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--no-copy-back", action="store_true",
                    help="Do not copy IK .mot back into the original dataset.")
    ap.add_argument("--no-reorder", action="store_true",
                    help="Skip reordering coordinate columns to OpenSim standard.")
    ap.add_argument("--lowpass-pelvis", action="store_true",
                    help="Low-pass filter pelvis_ty (OpenCap gait-dynamics convention).")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    run_ik_footopt_dataset(
        dataset_root=args.dataset_root,
        subjects=args.subjects,
        model_name=args.model_name,
        copy_back=not args.no_copy_back,
        reorder=not args.no_reorder,
        lowpass_pelvis=args.lowpass_pelvis,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
