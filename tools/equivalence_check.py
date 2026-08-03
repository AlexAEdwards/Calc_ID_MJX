"""Prove a refactor changed no output. REFACTOR_PLAN.md Stage 4.

Records a baseline of everything the pipeline produces for the fixture trials,
and later re-runs the same work and diffs against it. Stages 5-7 are
reorganisations, so a clean report is the pass condition for every commit.

    python tools/equivalence_check.py --record          # baseline from current code
    python tools/equivalence_check.py                   # compare; exit 1 on any diff
    python tools/equivalence_check.py --layers loader   # subset while iterating

Layers, cheapest first:

``discovery``   trial discovery over the fixture - counts, subjects, lengths
``loader``      TrialDataLoader batches: SHA-256 of input / static / masks
``targets``     build_direct_torque_targets output hashes
``metrics``     the masked metric helpers on fixed synthetic arrays
``loss``        compute_total_loss on real batches with a seeded prediction
``aggregate``   LOEO aggregation over an existing sweep - exact metric values

Tolerances are fixed by the plan and are not arguments: arrays must be
**bit-identical**, aggregated metrics **exact to all printed digits**. Anything
that cannot meet that is a behaviour change, not a reorganisation, and needs its
own decision rather than a looser threshold here.

``processdata_roundtrip`` re-runs the real pipeline (about a minute per
subject) and is the Stage 6 gate. The older ``processdata`` layer only hashes
what is already on disk.

Note: it re-runs the
preprocessing pipeline on the fixture's raw ``Motion/`` and compares every
produced ``.npy`` by SHA-256. It is not part of the default layers because it
needs MuJoCo and several minutes per trial; see ``--layers processdata``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from paths import artifact  # noqa: E402

FIXTURE = artifact("test_fixture")
BASELINE = REPO_ROOT / "tests" / "baseline" / "equivalence_baseline.json"
DEFAULT_LAYERS = ("discovery", "loader", "targets", "metrics", "loss", "aggregate")


def _h(*arrays) -> str:
    """SHA-256 over raw array bytes; NaN-safe and dtype/shape sensitive."""
    import numpy as np
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
def layer_discovery() -> Dict[str, Any]:
    from TransformerFinal.train import discover_all_trials
    trials = discover_all_trials(str(FIXTURE), refresh_cache=True, layout="experiment", scan_workers=4)
    rows = sorted(
        {"experiment": t.get("experiment", ""), "subject": t["subject"],
         "trial": t["trial"], "length": int(t["length"])}
        for t in trials
        for _ in [0]
    ) if False else sorted(
        ({"experiment": t.get("experiment", ""), "subject": t["subject"],
          "trial": t["trial"], "length": int(t["length"])} for t in trials),
        key=lambda r: (r["experiment"], r["subject"], r["trial"]),
    )
    return {"n_trials": len(rows), "trials": rows}


def _fixture_trials():
    from TransformerFinal.train import discover_all_trials
    return sorted(
        discover_all_trials(str(FIXTURE), refresh_cache=True, layout="experiment", scan_workers=4),
        key=lambda t: (t.get("experiment", ""), t["subject"], t["trial"]),
    )


def _loader_cfg(**over):
    cfg = dict(window_size=70, stride=16, batch_size=8, shuffle=False, trim_cop=False,
               deviation_learning=False, predict_jacobian=False, opencap_val=False,
               input_source="processed", include_pelvis_euler=False,
               include_ankle_heights=True, include_jacobian_input=True,
               include_auxiliary_denoising_inputs=True, prediction_margin_frames=20,
               use_grf_norm_cop=False, use_os_filtering=False, use_grf_nofilt=True,
               drop_last=False, use_noised=True, noised_gt=True,
               allow_missing_noised=True, edge_mode="train", edge_trim_frames=20)
    cfg.update(over)
    return cfg


def layer_loader() -> Dict[str, Any]:
    """Hash the first batch each trial produces, under both edge modes."""
    import numpy as np
    from TransformerFinal.data_loader import TrialDataLoader
    out: Dict[str, Any] = {}
    for mode in ("train", "infer"):
        per_trial = {}
        for t in _fixture_trials():
            key = f"{t.get('experiment','')}/{t['subject']}/{t['trial']}"
            dl = TrialDataLoader([t], **_loader_cfg(edge_mode=mode))
            batch = next(iter(dl), None)
            if batch is None:
                per_trial[key] = {"windows": int(dl.total_windows), "batch": None}
                continue
            per_trial[key] = {
                "windows": int(dl.total_windows),
                "input": _h(np.asarray(batch["input"])),
                "static_context": _h(np.asarray(batch["static_context"])),
                "supervision_mask": _h(np.asarray(batch["supervision_mask"])),
                "input_dim": int(np.asarray(batch["input"]).shape[-1]),
            }
        out[mode] = per_trial
    return out


def layer_targets() -> Dict[str, Any]:
    import numpy as np
    from TransformerFinal.data_loader import TrialDataLoader
    from TransformerFinal.direct_torque_utils import build_direct_torque_targets
    per_trial = {}
    for t in _fixture_trials():
        key = f"{t.get('experiment','')}/{t['subject']}/{t['trial']}"
        dl = TrialDataLoader([t], **_loader_cfg())
        batch = next(iter(dl), None)
        if batch is None:
            per_trial[key] = None
            continue
        tgt = np.asarray(build_direct_torque_targets(batch, xp_name="numpy"))
        per_trial[key] = {"hash": _h(tgt), "shape": list(tgt.shape),
                          "finite": bool(np.isfinite(tgt).all())}
    return per_trial


def layer_metrics() -> Dict[str, Any]:
    """Deterministic check of the masked metric helpers, no data required."""
    import numpy as np
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((64, 14)).astype(np.float64)
    gt = rng.standard_normal((64, 14)).astype(np.float64)
    from TransformerFinal.infer_directTorque import _per_channel_metrics
    st = _per_channel_metrics(pred, gt)
    return {
        "pooling": _h(np.array(st["pooling"]["sum_abs_err"]),
                      np.array(st["pooling"]["sum_sq_err"]),
                      np.array(st["pooling"]["sum_pred_gt"])),
        "per_channel_mae": {k: round(v["mae_bwh"], 12) for k, v in st["per_channel"].items()},
    }


def layer_loss() -> Dict[str, Any]:
    """Exercise compute_total_loss on real fixture batches with a fixed prediction.

    This is the only layer that reaches the training objective, so it is what
    makes moving compute_total_loss (616 LOC, 11 transitive deps) a verifiable
    change rather than a hopeful one. Uses a seeded prediction array and real
    normalizers so the result is deterministic but exercises the real code path.
    """
    import numpy as np
    import jax.numpy as jnp
    from TransformerFinal.data_loader import TrialDataLoader
    from TransformerFinal.train import (
        Normalizer, compute_normalizers_from_loader, compute_total_loss, normalize_batch,
    )
    from core.layers import STANDARD_OUTPUT_DIM

    out: Dict[str, Any] = {}
    for t in _fixture_trials()[:4]:                      # 4 trials keeps this ~seconds
        key = f"{t.get('experiment','')}/{t['subject']}/{t['trial']}"
        cfg = _loader_cfg(batch_size=4, edge_mode="legacy", edge_trim_frames=0)
        dl = TrialDataLoader([t], **cfg)
        raw = next(iter(dl), None)
        if raw is None:
            out[key] = None
            continue
        try:
            norms = compute_normalizers_from_loader(TrialDataLoader([t], **cfg), max_batches=1)
            batch = normalize_batch(raw, norms)
            n, seq = np.asarray(batch["input"]).shape[:2]
            pred = jnp.asarray(
                np.random.default_rng(0).standard_normal((n, seq, STANDARD_OUTPUT_DIM)) * 0.1
            )
            loss, metrics = compute_total_loss(
                pred, batch, norms,
                {"cop": 1.0, "grf": 1.0, "moments": 1.0, "contact": 1.0},
                False, False, True, False,
            )
            out[key] = {
                "loss": repr(round(float(loss), 10)),
                "metrics": {k: repr(round(float(np.asarray(v)), 10))
                            for k, v in sorted(metrics.items())
                            if np.asarray(v).size == 1},
            }
        except Exception as e:                            # record, do not mask
            out[key] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return out


def layer_aggregate(sweep: Path) -> Dict[str, Any]:
    acc = sweep / "accuracy" / "loeo_accuracy.json"
    if not acc.exists():
        return {"skipped": f"no accuracy report at {acc}"}
    d = json.loads(acc.read_text())
    o = d["overall"]["micro"]
    return {
        "n_trials": d["overall"]["n_trials"],
        "mae_bwh": repr(o["mae_bwh"]), "rmse_bwh": repr(o["rmse_bwh"]),
        "mean_channel_r": repr(o["mean_channel_r"]),
        "per_experiment_mae": {k: repr(v["micro"]["mae_bwh"])
                               for k, v in sorted(d["per_experiment"].items())},
        "per_channel_mae": {k: repr(v["mae_bwh"])
                            for k, v in o["per_channel"].items()},
    }


# Subjects regenerated by the processdata layer: one per MJX model width, kept
# small so the round trip stays about a minute each.
PROCESSDATA_SUBJECTS = [("PD", "PD_SUB01_off"), ("OA_Y", "OA1")]


def layer_processdata_roundtrip() -> Dict[str, Any]:
    """Re-run ProcessData.py from raw Motion/ and hash every array it produces.

    This is the only layer that exercises the preprocessing pipeline, so it is
    what makes Stage 6 verifiable. ProcessData was measured deterministic
    (74/74 byte-identical across two runs), so a difference here means the
    refactor changed behaviour.

    The comparison is strictly before/after within one session on identical
    input. It is NOT a comparison against the shipped datasets: several cohorts
    were post-processed after ProcessData ran, so their Motion/ on disk is
    already trimmed and a fresh run legitimately differs from what is stored.
    """
    import shutil
    import subprocess
    import tempfile

    out: Dict[str, Any] = {}
    for exp, subj in PROCESSDATA_SUBJECTS:
        src = FIXTURE / exp / subj
        if not src.is_dir():
            out[f"{exp}/{subj}"] = {"skipped": "not in fixture"}
            continue
        with tempfile.TemporaryDirectory(prefix="pdeq_") as td:
            root = Path(td)
            dst = root / subj
            dst.mkdir(parents=True)
            # Motion/ plus subject-level model assets only - ProcessData must
            # regenerate ProcessedData/ itself or the check proves nothing.
            for item in src.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst / item.name)
                elif item.name == "Geometry":
                    shutil.copytree(item, dst / "Geometry")
                elif item.name.startswith("Trial_") and (item / "Motion").is_dir():
                    (dst / item.name).mkdir(exist_ok=True)
                    shutil.copytree(item / "Motion", dst / item.name / "Motion")
            r = subprocess.run(
                [sys.executable, str(REPO_ROOT / "ProcessData.py"),
                 "--data-root", str(root), "--subject", subj,
                 "--device", "cpu", "--workers", "1"],
                capture_output=True, text=True, timeout=1800,
            )
            if r.returncode != 0:
                out[f"{exp}/{subj}"] = {"error": f"exit {r.returncode}: {r.stderr[-160:]}"}
                continue
            per_trial = {}
            for pd_dir in sorted(dst.glob("Trial_*/ProcessedData")):
                per_trial[pd_dir.parent.name] = {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in sorted(pd_dir.glob("*.npy"))
                }
            out[f"{exp}/{subj}"] = per_trial
    return out


def layer_processdata() -> Dict[str, Any]:
    """SHA-256 of every ProcessedData .npy in the fixture (Stage 6 target).

    Recording is cheap; *verifying* means regenerating the fixture with
    ProcessData.py first, which is the heavy part and is left to the operator.
    """
    import numpy as np
    per_trial = {}
    for pd in sorted(FIXTURE.glob("*/*/Trial_*/ProcessedData")):
        key = "/".join(pd.parts[-4:-1])
        files = {}
        for f in sorted(pd.glob("*.npy")):
            try:
                files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
            except OSError:
                files[f.name] = "UNREADABLE"
        per_trial[key] = files
    return per_trial


LAYER_FNS = {
    "discovery": lambda a: layer_discovery(),
    "loader": lambda a: layer_loader(),
    "targets": lambda a: layer_targets(),
    "metrics": lambda a: layer_metrics(),
    "loss": lambda a: layer_loss(),
    "aggregate": lambda a: layer_aggregate(Path(a.sweep)),
    "processdata": lambda a: layer_processdata(),
    "processdata_roundtrip": lambda a: layer_processdata_roundtrip(),
}


# --------------------------------------------------------------------------
def _diff(path: str, a: Any, b: Any, out: List[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: out.append(f"  + {path}/{k} (new)")
            elif k not in b: out.append(f"  - {path}/{k} (missing now)")
            else: _diff(f"{path}/{k}", a[k], b[k], out)
    elif isinstance(a, list) and isinstance(b, list):
        if a != b:
            out.append(f"  ~ {path}: list differs ({len(a)} -> {len(b)} entries)")
    elif a != b:
        sa, sb = str(a), str(b)
        out.append(f"  ~ {path}: {sa[:44]} -> {sb[:44]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="Write the baseline instead of comparing.")
    ap.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    ap.add_argument("--sweep", default=str(artifact("outputs", "DirectTorque_LOEO_edge70")))
    ap.add_argument("--baseline", default=str(BASELINE))
    args = ap.parse_args()

    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    unknown = [l for l in layers if l not in LAYER_FNS]
    if unknown:
        raise SystemExit(f"Unknown layer(s) {unknown}. Known: {sorted(LAYER_FNS)}")
    if not FIXTURE.exists():
        raise SystemExit(f"No fixture at {FIXTURE}. Run tools/stage_test_fixture.py --apply first.")

    current = {}
    for name in layers:
        print(f"  running layer: {name} ...", flush=True)
        current[name] = LAYER_FNS[name](args)

    bpath = Path(args.baseline)
    if args.record:
        bpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {"recorded": datetime.now().isoformat(timespec="seconds"),
                   "fixture": str(FIXTURE), "layers": current}
        bpath.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nBaseline written: {bpath}")
        return

    if not bpath.exists():
        raise SystemExit(f"No baseline at {bpath}. Run with --record first.")
    base = json.loads(bpath.read_text())["layers"]

    diffs: List[str] = []
    for name in layers:
        if name not in base:
            diffs.append(f"  + {name}: layer not in baseline"); continue
        _diff(name, base[name], current[name], diffs)

    print()
    if diffs:
        print(f"EQUIVALENCE FAILED - {len(diffs)} difference(s):")
        for d in diffs[:40]:
            print(d)
        if len(diffs) > 40:
            print(f"  ... and {len(diffs)-40} more")
        print("\nStages 5-7 are reorganisations: if output changed, the commit is wrong.")
        sys.exit(1)
    print(f"EQUIVALENCE OK - {len(layers)} layer(s) identical to baseline "
          f"({', '.join(layers)})")


if __name__ == "__main__":
    main()
