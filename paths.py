"""Single source of truth for where datasets and run artifacts live.

Stage 1 of REFACTOR_PLAN.md moved the data out of the repo root::

    <repo>/datasets/<name>     formerly <repo>/<name>
    <repo>/artifacts/<name>    formerly <repo>/<name>

A compatibility symlink still sits at every old path, so existing hardcoded
references keep working. This module exists so new code stops adding to that
pile: resolve through here and the eventual symlink removal is a no-op.

Both roots can be relocated without touching code::

    export CALCID_DATASETS=/mnt/nas/Datasets
    export CALCID_ARTIFACTS=/scratch/runs

Usage::

    from paths import dataset, artifact, REPO_ROOT

    ds  = dataset("TrustedDataSet_ByExperiment")
    out = artifact("DirectTorque_LOEO_edge70", "accuracy", "loeo_accuracy.json")
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT", "DATASETS_ROOT", "ARTIFACTS_ROOT",
    "dataset", "artifact", "resolve", "describe",
]

REPO_ROOT = Path(__file__).resolve().parent

#: Overridable so the data can live on a different volume than the code.
DATASETS_ROOT = Path(os.environ.get("CALCID_DATASETS") or REPO_ROOT / "datasets")
ARTIFACTS_ROOT = Path(os.environ.get("CALCID_ARTIFACTS") or REPO_ROOT / "artifacts")


def dataset(*parts: str | os.PathLike) -> Path:
    """Path under the datasets root. ``dataset("Hip_OA", "HOA059_M0")``."""
    return DATASETS_ROOT.joinpath(*map(str, parts))


def artifact(*parts: str | os.PathLike) -> Path:
    """Path under the artifacts root. ``artifact("outputs", "run", "best.pkl")``."""
    return ARTIFACTS_ROOT.joinpath(*map(str, parts))


def resolve(path: str | os.PathLike) -> Path:
    """Best-effort resolution of a legacy repo-root-relative path.

    Accepts either a new-style path or one written against the pre-Stage-1
    layout (``outputs/foo``, ``Hip_OA/bar``) and returns the real location.
    Absolute paths are returned unchanged. Handy when migrating a caller
    incrementally rather than all at once.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    head, *rest = p.parts
    if head in {"datasets", "artifacts"}:
        return REPO_ROOT.joinpath(*p.parts)
    for root in (DATASETS_ROOT, ARTIFACTS_ROOT):
        candidate = root.joinpath(head, *rest)
        if candidate.exists() or root.joinpath(head).exists():
            return candidate
    return REPO_ROOT.joinpath(*p.parts)


def describe() -> str:
    """One-line summary, useful in run logs so a run records where it read from."""
    return (f"repo={REPO_ROOT} datasets={DATASETS_ROOT}"
            f"{' (env)' if os.environ.get('CALCID_DATASETS') else ''}"
            f" artifacts={ARTIFACTS_ROOT}"
            f"{' (env)' if os.environ.get('CALCID_ARTIFACTS') else ''}")


if __name__ == "__main__":
    print(describe())
    for name in ("TrustedDataSet_ByExperiment", "KineticVAEDataset", "Hip_OA"):
        p = dataset(name)
        print(f"  dataset({name!r}) -> {p}  exists={p.exists()}")
    for name in ("outputs", "inference_results"):
        p = artifact(name)
        print(f"  artifact({name!r}) -> {p}  exists={p.exists()}")
    print(f"  resolve('outputs/DirectTorque_LOEO_edge70') -> {resolve('outputs/DirectTorque_LOEO_edge70')}")
