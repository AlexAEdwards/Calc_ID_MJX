"""Every in-repo file path mentioned in code or docs must actually exist.

Stage 7 of REFACTOR_PLAN.md moves files rather than rewriting them, so its
characteristic failure is not "the output changed" - the equivalence gate cannot
see it - but "something still points at where this used to be". That includes
markdown links, `python scripts/foo.py` lines in docstrings and READMEs, and
path literals in code.

Sibling references are the ones that actually bite. A doc sitting next to
`experiment_groups.py` can say exactly that, and it resolves because the doc is in
the same directory. Move the doc to `docs/` and the reference silently rots. So a
reference is accepted if it resolves against the repo root *or* against the
directory of the file that mentions it, which means moving either end of the pair
without fixing the other fails this test.

Runs without jax/mujoco: it reads files, it does not import them.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Extensions worth scanning for references.
SCANNED = (".py", ".md", ".toml", ".yml", ".yaml", ".txt")

#: [text](target) links. Markdown only - in Python this same shape matches
#: ordinary subscript-then-call syntax such as ``LAYER_FNS[name](args)``.
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)#][^)\s]*)\)")

#: Applied to every scanned file.
PATTERNS = (
    # `python foo/bar.py` / `pytest tests/x.py` in docs and docstrings
    re.compile(r"(?:python3?|pytest)\s+([A-Za-z_][\w./-]*\.(?:py|md))\b"),
    # quoted path literals in code
    re.compile(r"""["']([\w.-]+(?:/[\w.-]+)+\.(?:py|md|json|yml|yaml|xml|toml))["']"""),
)

#: Paths that are runtime data or user-supplied, not repo files. These are
#: produced or consumed at run time and are correctly absent from a clean tree.
IGNORED_PREFIXES = (
    "datasets/", "artifacts/", "outputs/", "output/", "Results/",
    "ProcessedData/", "Motion/", "StrokeDataset/", "Datasets_NAS/",
    "inference_results/", "checkpoints/", "logs/",
)

#: Documentation placeholders, and files this repo *generates* elsewhere rather
#: than contains. example_torque_reconstruction.py is written into the built
#: KineticVAEDataset by scripts/build_kinetic_vae_dataset.py.
IGNORED_NAMES = frozenset({
    "my_script.py", "script.py", "scripts/foo.py", "foo.py", "bar.py",
    "example.py", "your_script.py", "path/to/file.py",
    "example_torque_reconstruction.py",
})

#: Files whose examples describe something other than this repo's layout.
#: CHPC_REFERENCE_FOR_AGENTS.txt documents generic Slurm/cluster usage. This file
#: is excluded from its own scan because its patterns and assertions necessarily
#: contain deliberately-nonexistent paths.
IGNORED_SOURCES = frozenset({
    "CHPC_REFERENCE_FOR_AGENTS.txt",
    "tests/test_repo_references.py",
})


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\n") if f.endswith(SCANNED)]


def _references(text: str, path: str = "") -> set[str]:
    found = set()
    for pat in PATTERNS + ((MARKDOWN_LINK,) if path.endswith(".md") else ()):
        for raw in pat.findall(text):
            ref = raw.strip()
            # Regex artifacts: real paths have no whitespace and no format holes.
            if not ref or any(c in ref for c in " \t\n,()[]{}%*") or "://" in ref:
                continue
            if ref.startswith(("http", "#", "<", "$", "~")):
                continue
            found.add(ref)
    return found


def _is_ok(ref: str, referencing_file: str) -> bool:
    if ref in IGNORED_NAMES or ref.startswith(IGNORED_PREFIXES):
        return True
    # Resolvable from the repo root, or from the referencing file's directory.
    if (REPO_ROOT / ref).exists():
        return True
    return (REPO_ROOT / Path(referencing_file).parent / ref).exists()


def _collect_dangling() -> dict[str, list[str]]:
    dangling: dict[str, list[str]] = {}
    for f in _tracked_files():
        if f in IGNORED_SOURCES:
            continue
        try:
            text = (REPO_ROOT / f).read_text(errors="ignore")
        except OSError:
            continue
        for ref in _references(text, f):
            if not _is_ok(ref, f):
                dangling.setdefault(ref, []).append(f)
    return dangling


def test_no_dangling_file_references() -> None:
    """No tracked file points at a repo path that does not exist."""
    dangling = _collect_dangling()
    assert not dangling, "Dangling in-repo references:\n" + "\n".join(
        f"  {ref}  <- referenced by {', '.join(sorted(src))}"
        for ref, src in sorted(dangling.items())
    )


def test_scan_actually_covers_the_repo() -> None:
    """Guard against the test passing because it scanned nothing."""
    files = _tracked_files()
    assert len(files) > 100, f"only {len(files)} files scanned"
    refs = sum(len(_references((REPO_ROOT / f).read_text(errors="ignore"), f))
               for f in files)
    assert refs > 40, f"only {refs} references extracted - patterns may have broken"


def test_the_check_detects_a_moved_file(tmp_path: Path) -> None:
    """A reference to a nonexistent sibling must be reported, not ignored."""
    # A sibling reference that no longer resolves must be caught.
    assert not _is_ok("definitely_not_here.py", "docs/some_doc.md")
    # Root-relative and sibling-relative resolution both count as fine.
    assert _is_ok("README.md", "docs/some_doc.md")
    assert _is_ok("test_paths.py", "tests/test_repo_references.py")
    # And the markdown-link pattern must not fire on Python call syntax.
    assert _references("LAYER_FNS[name](args)\n", "x.py") == set()


@pytest.mark.parametrize("entry", [
    "ProcessData.py", "paths.py", "README.md", "REFACTOR_PLAN.md",
    "pyproject.toml", "tools/equivalence_check.py", "tools/stage_test_fixture.py",
])
def test_key_files_present(entry: str) -> None:
    """Named anchors the docs and CI depend on."""
    assert (REPO_ROOT / entry).exists(), f"{entry} is missing"
