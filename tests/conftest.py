"""Shared fixtures. REFACTOR_PLAN.md Stage 4.

Tests that need real data are skipped rather than failed when the fixture is
absent, so a fresh clone can still run the pure-logic suite.
"""
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import artifact  # noqa: E402

FIXTURE = artifact("test_fixture")


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    if not FIXTURE.is_dir():
        pytest.skip(f"no test fixture at {FIXTURE}; run tools/stage_test_fixture.py --apply")
    return FIXTURE


@pytest.fixture(scope="session")
def fixture_trials(fixture_root):
    from TransformerFinal.train import discover_all_trials
    trials = discover_all_trials(str(fixture_root), refresh_cache=True,
                                 layout="experiment", scan_workers=4)
    if not trials:
        pytest.skip("fixture contains no discoverable trials")
    return sorted(trials, key=lambda t: (t.get("experiment", ""), t["subject"], t["trial"]))
