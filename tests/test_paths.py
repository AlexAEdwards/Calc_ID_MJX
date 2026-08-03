"""paths.py resolution, including the legacy spellings retired in Stage 3."""
import os
from pathlib import Path

import paths


def test_roots_are_absolute():
    assert paths.REPO_ROOT.is_absolute()
    assert paths.DATASETS_ROOT.is_absolute()
    assert paths.ARTIFACTS_ROOT.is_absolute()


def test_dataset_and_artifact_join_under_the_right_root():
    assert paths.dataset("X", "y") == paths.DATASETS_ROOT / "X" / "y"
    assert paths.artifact("X", "y") == paths.ARTIFACTS_ROOT / "X" / "y"


def test_resolve_maps_pre_stage1_spellings():
    """Old root-relative names must still resolve after the symlinks were removed."""
    assert paths.resolve("outputs/foo").parent.parent == paths.ARTIFACTS_ROOT
    assert paths.resolve("TrustedDataSet_ByExperiment").parent == paths.DATASETS_ROOT


def test_resolve_passes_absolute_through():
    p = Path("/tmp/some/where")
    assert paths.resolve(p) == p


def test_resolve_leaves_new_style_paths_alone():
    assert paths.resolve("datasets/Foo") == paths.REPO_ROOT / "datasets" / "Foo"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CALCID_DATASETS", str(tmp_path))
    import importlib
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.DATASETS_ROOT == tmp_path
    finally:
        monkeypatch.delenv("CALCID_DATASETS", raising=False)
        importlib.reload(paths)
