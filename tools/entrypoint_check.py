"""Record and compare whether every entry point in the repo still starts.

The Stage 5/6 equivalence gate answers "did the output change". Stage 7 moves
files instead of editing them, so its failure mode is different: a script that no
longer resolves its imports, or a launcher that shells out to a path that moved.
Neither changes any array, so `tools/equivalence_check.py` stays green while the
repo quietly stops working.

This closes that gap by the crudest reliable means: run every entry point with
``--help`` and record what happened. ``--help`` exits before any real work, so
this is safe to run anywhere, but it still forces the whole import chain.

Usage::

    python tools/entrypoint_check.py --record    # write the baseline
    python tools/entrypoint_check.py             # compare against it

A previously-working entry point that now fails is a regression and exits 1. An
entry point that was already broken and still is gets reported but does not fail
the run, so a known-bad script does not block unrelated work. New entry points
are reported and recorded.

Needs the full JAX/MuJoCo environment - see full_env.yml. Without it every
entry point fails identically and the comparison is meaningless, so the run
aborts up front rather than producing a wall of false regressions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "tests" / "baseline" / "entrypoints.json"
MAIN_GUARD = "__main__"
TIMEOUT_S = 180


def discover() -> list[str]:
    """Tracked .py files that are runnable scripts, in a stable order."""
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    found = []
    for rel in sorted(f for f in out.split("\n") if f):
        try:
            text = (REPO_ROOT / rel).read_text(errors="ignore")
        except OSError:
            continue
        if f'__name__ == "{MAIN_GUARD}"' in text or f"__name__ == '{MAIN_GUARD}'" in text:
            found.append(rel)
    return found


def probe(rel: str) -> str:
    """Run one entry point's --help. Returns 'ok' or a short failure label."""
    try:
        r = subprocess.run([sys.executable, str(REPO_ROOT / rel), "--help"],
                           cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return "timeout"
    if r.returncode == 0:
        return "ok"
    # Last meaningful stderr line, normalised so the label is stable across
    # machines (paths and addresses differ, the failure class does not).
    tail = [l for l in r.stderr.strip().split("\n") if l.strip()]
    label = tail[-1] if tail else f"exit {r.returncode}"
    for marker in ("Error:", "error:"):
        if marker in label:
            label = label.split(marker, 1)[0] + marker + label.split(marker, 1)[1][:60]
            break
    return label[:110]


def run_all(jobs: int) -> dict[str, str]:
    entries = discover()
    print(f"  probing {len(entries)} entry point(s) with --help ...", flush=True)
    results: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for rel, status in zip(entries, pool.map(probe, entries)):
            results[rel] = status
    return results


def _require_runtime() -> None:
    missing = [m for m in ("jax", "mujoco") if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(
            f"Cannot probe entry points with {sys.executable}:\n"
            f"  missing module(s): {', '.join(missing)}\n"
            "  Every entry point would fail on import and the comparison would be\n"
            "  meaningless. Re-run with the full JAX/MuJoCo environment.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="Write the baseline instead of comparing.")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    _require_runtime()
    current = run_all(args.jobs)
    ok = sum(1 for v in current.values() if v == "ok")
    print(f"  {ok}/{len(current)} start cleanly")

    bpath = Path(args.baseline)
    if args.record:
        bpath.parent.mkdir(parents=True, exist_ok=True)
        bpath.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"\nBaseline written: {bpath}")
        for rel, status in sorted(current.items()):
            if status != "ok":
                print(f"  known-broken: {rel}: {status}")
        return

    if not bpath.exists():
        raise SystemExit(f"No baseline at {bpath}. Run with --record first.")
    base = json.loads(bpath.read_text())

    regressions, fixed, added, removed = [], [], [], []
    for rel, status in sorted(current.items()):
        was = base.get(rel)
        if was is None:
            added.append(f"{rel}: {status}")
        elif was == "ok" and status != "ok":
            regressions.append(f"{rel}: was ok, now {status}")
        elif was != "ok" and status == "ok":
            fixed.append(rel)
    for rel in sorted(base):
        if rel not in current:
            removed.append(f"{rel} (was {base[rel]})")

    for label, items in (("REGRESSION", regressions), ("no longer present", removed),
                         ("now fixed", fixed), ("new", added)):
        for it in items:
            print(f"  {label}: {it}")

    if regressions or removed:
        raise SystemExit(
            f"\nENTRY POINTS BROKEN - {len(regressions)} regression(s), "
            f"{len(removed)} disappeared.\n"
            "Stage 7 moves files; if something stopped starting, a reference to it "
            "was not updated.")
    print("\nENTRY POINTS OK - nothing that used to start has stopped starting.")


if __name__ == "__main__":
    main()
