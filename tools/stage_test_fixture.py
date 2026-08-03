"""Stage a small, deliberately varied set of real trials as a test fixture.

REFACTOR_PLAN.md Stage 4. The fixture is what every equivalence check runs
against, so it is chosen to cover the shape variation that actually breaks
things, not just to be small:

* **all 8 cohorts** - each has its own provenance and quirks
* **trial length spanning 2 -> ~6,000 frames** - short trials exercise the
  edge-trim drop path and the padding path; long ones exercise multi-window
  accumulation
* **all three MJX widths** - ``pos_mjx`` is 23 (OpenCap), 33 (patella-free) or
  43 (with patella), and code that assumes one width breaks on the others
* **noised bundle present and absent** - OpenCapVal has none, which is what
  ``allow_missing_noised`` exists for
* **OpenSim ID present and absent** - only 5 of 8 cohorts have it
* **two degenerate trials** - a 24-frame trial below ``MIN_TRIAL_LENGTH`` and a
  2-frame trial, both of which must stay excluded rather than crash

The fixture is a *copy*, so tests can never write back into ``datasets/``. It is
gitignored (roughly 180 MB); only the recorded baseline hashes are committed.

    python tools/stage_test_fixture.py            # show the plan
    python tools/stage_test_fixture.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))
from paths import artifact, dataset  # noqa: E402

SOURCE = dataset("TrustedDataSet_ByExperiment")
FIXTURE = artifact("test_fixture")

# (experiment, subject, trial, why it is in the fixture)
SELECTION = [
    ("Numeric",        "02",                        "Trial_11", "long trial (1991f), nq=33, treadmill"),
    ("S_GAH",          "S_GAH_1",                   "Trial_4",  "very long (5732f), nq=33, unilateral protocol"),
    ("GaitRetraining", "GaitRetraining_Subject103", "Trial_1",  "medium (217f), nq=43, has OpenSim ID"),
    ("OA_Y",           "OA1",                       "Trial_1",  "medium (201f), nq=33, visual-QC cohort"),
    ("Stroke",         "SUBJ01",                    "Trial_2",  "short (146f), nq=43"),
    ("OpenCapVal",     "subject10",                 "Trial_1",  "124f, nq=23, NO noised bundle"),
    ("Hip_OA",         "HEA121_Marche",             "Trial_1",  "121f, nq=43, visually trimmed, no OpenSim ID"),
    ("PD",             "PD_SUB01_off",              "Trial_15", "113f, nq=43, just above the edge-trim threshold"),
    ("Stroke",         "SUBJ72",                    "Trial_3",  "24f - below MIN_TRIAL_LENGTH, must stay undiscovered"),
    ("PD",             "PD_SUB10_off",              "Trial_7",  "2f - degenerate, must not crash the loader"),
]


def plan() -> List[Dict[str, str]]:
    rows = []
    for exp, subj, trial, why in SELECTION:
        src = SOURCE / exp / subj / trial
        rows.append({
            "experiment": exp, "subject": subj, "trial": trial, "why": why,
            "src": str(src), "exists": src.is_dir(),
            "dst": str(FIXTURE / exp / subj / trial),
        })
    return rows


def stage(rows: List[Dict[str, str]], *, apply: bool) -> None:
    missing = [r for r in rows if not r["exists"]]
    if missing:
        for r in missing:
            print(f"   MISSING {r['experiment']}/{r['subject']}/{r['trial']}")
        raise SystemExit("Fixture selection references trials that do not exist.")

    for r in rows:
        print(f"   {r['experiment']:<16}{r['subject']:<28}{r['trial']:<10}{r['why']}")
    if not apply:
        print("\nDry run - nothing copied. Re-run with --apply.")
        return

    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)
    experiments = sorted({r["experiment"] for r in rows})
    for r in rows:
        src, dst = Path(r["src"]), Path(r["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        # subject-level assets the loader needs (Patient_MD, models, Geometry)
        for item in Path(r["src"]).parent.iterdir():
            if item.is_file():
                shutil.copy2(item, dst.parent / item.name)
            elif item.name == "Geometry" and not (dst.parent / "Geometry").exists():
                shutil.copytree(item, dst.parent / "Geometry")

    # Register as a nested experiment-layout dataset so discovery accepts it.
    (FIXTURE / "experiment_layout_manifest.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "layout": "experiment",
        "note": "Test fixture staged by tools/stage_test_fixture.py - a copy, never written back to.",
        "experiments": experiments,
        "moves": [{"subject": r["subject"], "experiment": r["experiment"]} for r in rows],
        "selection": [{k: r[k] for k in ("experiment", "subject", "trial", "why")} for r in rows],
    }, indent=2))
    total = sum(f.stat().st_size for f in FIXTURE.rglob("*") if f.is_file())
    print(f"\n   staged {len(rows)} trials across {len(experiments)} experiments -> {FIXTURE}")
    print(f"   size: {total/1e6:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print(f"source:  {SOURCE}\nfixture: {FIXTURE}\n")
    stage(plan(), apply=args.apply)


if __name__ == "__main__":
    main()
