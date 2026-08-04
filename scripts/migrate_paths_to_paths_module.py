"""Stage 3 (completion): route root-relative data/artifact paths through paths.py.

Stage 1 moved 22 directories under ``datasets/`` and ``artifacts/`` and left a
compatibility symlink at each old name. Those symlinks can only be retired once
no code resolves a moved directory by its old root-relative name. This script
performs that rewrite.

Two patterns are handled, both derived from the Stage 1 manifest so the
dataset/artifact split can never drift from what actually moved:

A. ``PROJECT_ROOT / "outputs" / ...``      ->  ``artifact("outputs", ...)``
   ``REPO_ROOT / "Hip_OA"``                ->  ``dataset("Hip_OA")``

B. bare relative defaults, which resolved against the *current working
   directory* and were already fragile before Stage 1::

       default="outputs/mod_q"             ->  default=str(artifact("outputs", "mod_q"))

Deliberately NOT rewritten: string literals that merely happen to equal a moved
directory name but are not paths - ``("Hip_OA", ("HOA", "HEA"))`` is an
experiment rule, and ``default="inference_results"`` in the inference scripts is
a *subfolder name* created inside each trial directory, not a repo path.

    python scripts/migrate_paths_to_paths_module.py            # dry run
    python scripts/migrate_paths_to_paths_module.py --apply
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from paths import REPO_ROOT
MANIFEST = REPO_ROOT / "repo_root_layout_manifest.json"

# Names that look like a moved directory but are never a path in these files.
SKIP_FILES = {
    "TransformerFinal/experiment_groups.py",      # "Hip_OA" is an experiment label
    "scripts/migrate_paths_to_paths_module.py",   # this file
    "scripts/reorganize_repo_root.py",            # owns the move lists verbatim
    "paths.py",
}
SKIP_DIR_PARTS = ("__pycache__", "legacy_forward_sim", "legacy_scott_data",
                  "datasets", "artifacts", "myoconverter", ".git")


def load_groups() -> Dict[str, str]:
    manifest = json.loads(MANIFEST.read_text())
    return {m["name"]: m["group"] for m in manifest["moves"]}


def _call(group: str, name: str, tail: str = "") -> str:
    fn = "dataset" if group == "datasets" else "artifact"
    return f'{fn}("{name}"{tail})'


def _ensure_import(src: str) -> Tuple[str, bool]:
    """Insert `from paths import artifact, dataset` after the last top-level import."""
    if re.search(r'^from paths import', src, re.M):
        return src, False
    lines = src.splitlines(keepends=True)
    last = None
    for i, l in enumerate(lines[:120]):
        if re.match(r'^(import |from )\S', l) and "__future__" not in l:
            last = i
    if last is None:
        for i, l in enumerate(lines[:60]):
            if re.match(r'^from __future__', l):
                last = i
    if last is None:
        return "from paths import artifact, dataset  # noqa: E402\n" + src, True
    lines.insert(last + 1, "from paths import artifact, dataset  # noqa: E402\n")
    return "".join(lines), True


def migrate(groups: Dict[str, str]) -> List[Tuple[str, int, int]]:
    alt = "|".join(sorted(map(re.escape, groups), key=len, reverse=True))
    # A: <ROOT> / "name"  (optionally followed by further / "part" segments)
    pat_a = re.compile(rf'\b(?:PROJECT_ROOT|REPO_ROOT|ROOT|REPO)\s*/\s*"({alt})"')
    # B: a bare relative string starting with a moved dir
    pat_b = re.compile(rf'"({alt})/([^"]*)"')

    results: List[Tuple[str, int, int]] = []
    files = sorted(set(
        list(REPO_ROOT.glob("TransformerFinal/**/*.py"))
        + list(REPO_ROOT.glob("scripts/*.py"))
        + list(REPO_ROOT.glob("*.py"))
    ))
    for f in files:
        rel = str(f.relative_to(REPO_ROOT))
        if rel in SKIP_FILES or any(p in f.parts for p in SKIP_DIR_PARTS):
            continue
        src = f.read_text(errors="ignore")
        orig = src

        src, n_a = pat_a.subn(lambda m: _call(groups[m.group(1)], m.group(1)), src)
        def _b(m):
            name, tail = m.group(1), m.group(2)
            parts = "".join(f', "{p}"' for p in tail.split("/") if p)
            return f'str({_call(groups[name], name, parts)})'
        src, n_b = pat_b.subn(_b, src)

        if src != orig:
            src, _ = _ensure_import(src)
            results.append((rel, n_a, n_b))
            f.write_text(src)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    groups = load_groups()

    if not args.apply:
        print("Dry run is not supported for the rewrite itself; inspect with git diff after --apply.")
        print(f"Would consider {len(groups)} moved directory names.")
        return

    rows = migrate(groups)
    ta = sum(r[1] for r in rows); tb = sum(r[2] for r in rows)
    print(f"rewrote {len(rows)} files: {ta} root-relative constructions, {tb} bare relative strings")
    for rel, a, b in rows:
        print(f"   {rel:<62} A={a} B={b}")


if __name__ == "__main__":
    main()
