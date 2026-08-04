"""Tracked code must not import modules that git does not track.

This failure is invisible on the machine where the code was written and total on
every other one. The module sits on disk, so every import succeeds locally and
every test passes; clone the repo somewhere else and the import fails - or
worse, is swallowed by a bare `except ImportError` and the feature silently
stops happening.

Stage 7 found two real instances:

* `ProcessData.py` imported `ProcessAddbiomechnics.updateModel` for
  `fix_xml_masses` and `knee_coupling_is_canonical_xml`, under a bare
  `except Exception` that bound both to None. `ProcessAddbiomechnics/` was
  listed in .gitignore among the cohort *datasets*, though it contained no data.
  A fresh clone therefore lost mass-fixing and knee-coupling validation with no
  error at all.
* `scripts/data_prep/generate_noised12_distributed.py` imported
  `generate_sine_noise` from the ignored `NoiseAndPowerAnalOfInputData/`.

Both are fixed by tracking the imported source. This test is what stops a third.

Deliberately not a check that every file is tracked: analysis outputs, scratch
work and large artifacts are ignored on purpose. The rule is narrower and
non-negotiable - if tracked code imports it, it ships.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Never walked: third-party checkouts, data roots, caches.
SKIP_DIR_PARTS = frozenset({
    ".git", "myoconverter", "datasets", "artifacts", "__pycache__",
    ".venv", "venv", "node_modules", "build", "dist", ".mypy_cache",
})


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return {f for f in out.split("\n") if f}


def _walk_py() -> list[str]:
    """Repo-relative .py paths, pruning skipped directories during the walk.

    Pruning rather than filtering afterwards is the whole point: datasets/ and
    artifacts/ hold hundreds of gigabytes, and a recursive glob descends into
    them before discarding the results, which takes minutes instead of
    milliseconds.
    """
    out: list[str] = []
    stack = [REPO_ROOT]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_symlink():
                continue
            if e.is_dir():
                if e.name not in SKIP_DIR_PARTS:
                    stack.append(e)
            elif e.suffix == ".py":
                out.append(str(e.relative_to(REPO_ROOT)))
    return out


def _by_stem(paths) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for rel in paths:
        idx.setdefault(Path(rel).stem, []).append(rel)
    return idx


def _untracked_py(tracked: set[str]) -> set[str]:
    """Every .py file in the repo that git does not track."""
    return {rel for rel in _walk_py() if rel not in tracked}


def _imported_names(path: Path) -> set[str]:
    """Dotted module names a file imports, absolute imports only.

    Both forms matter and they resolve differently. ``import generate_sine_noise``
    is a bare name found via sys.path, while
    ``from ProcessAddbiomechnics.updateModel import x`` is a *path*: taking only
    the top-level name would look for ProcessAddbiomechnics.py, which does not
    exist, and miss the file actually imported. That is exactly how the
    ProcessData case escaped the first version of this check.
    """
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except (SyntaxError, OSError):
        return set()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


def _candidate_files(dotted: str, by_stem: dict[str, list[str]] | None = None) -> set[str]:
    """Repo-relative .py paths an import of ``dotted`` could resolve to."""
    as_path = dotted.replace(".", "/")
    cands = {f"{as_path}.py", f"{as_path}/__init__.py"}
    if "." not in dotted:
        # A bare name can also be found through a sys.path entry pointing at any
        # directory in the repo, so basename matches count too.
        if by_stem is None:
            by_stem = _by_stem(_walk_py())
        cands.update(by_stem.get(dotted, ()))
    return cands


def _violations() -> dict[str, dict[str, list[str]]]:
    tracked = _tracked()
    all_py = _walk_py()
    untracked = {r for r in all_py if r not in tracked}
    by_stem = _by_stem(all_py)
    out: dict[str, dict[str, list[str]]] = {}
    for rel in sorted(f for f in tracked if f.endswith(".py")):
        for dotted in _imported_names(REPO_ROOT / rel):
            hit = sorted(_candidate_files(dotted, by_stem) & untracked)
            if hit:
                out.setdefault(dotted, {"defined_in": hit, "imported_by": []})
                out[dotted]["imported_by"].append(rel)
    return out


def test_no_tracked_file_imports_untracked_source() -> None:
    v = _violations()
    assert not v, "Tracked code imports modules git does not track:\n" + "\n".join(
        f"  {mod}  defined in {info['defined_in']}\n"
        f"      imported by {', '.join(info['imported_by'])}"
        for mod, info in sorted(v.items())
    )


def test_the_two_known_offenders_are_now_tracked() -> None:
    """Regression pins for the instances this test was written from."""
    tracked = _tracked()
    for rel in ("ProcessAddbiomechnics/updateModel.py",
                "NoiseAndPowerAnalOfInputData/generate_sine_noise.py"):
        assert rel in tracked, f"{rel} must stay tracked - live code imports it"


def test_scan_is_not_vacuous() -> None:
    """The check must actually be looking at something.

    Note this asserts only on the tracked side. A working tree with *no*
    untracked .py is the correct state, not a broken scan - it is exactly what a
    fresh clone and CI look like. The proof that the detector still detects is
    test_detector_resolves_the_two_known_offenders, which does not depend on any
    untracked file being present.
    """
    py = [f for f in _tracked() if f.endswith(".py")]
    assert len(py) > 100, f"only {len(py)} tracked .py files seen"
    assert _walk_py(), "directory walk found no .py at all; check SKIP_DIR_PARTS"


@pytest.mark.parametrize("dotted,rel", [
    ("ProcessAddbiomechnics.updateModel", "ProcessAddbiomechnics/updateModel.py"),
    ("generate_sine_noise", "NoiseAndPowerAnalOfInputData/generate_sine_noise.py"),
])
def test_detector_resolves_the_two_known_offenders(dotted: str, rel: str) -> None:
    """The detector must map each real import back to the file it loads.

    Without this the check silently narrows: an earlier version only looked at
    top-level names, so a dotted import into a package directory resolved to
    nothing and was never flagged.
    """
    importers = [f for f in _tracked() if f.endswith(".py")
                 and dotted in _imported_names(REPO_ROOT / f)]
    assert importers, f"nothing imports {dotted} any more - is this pin stale?"
    assert rel in _candidate_files(dotted), (
        f"detector does not resolve {dotted} to {rel}; it would not catch it "
        "were the file untracked")
