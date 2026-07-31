"""Experiment (source-study) grouping for the nested dataset layout.

The flat training datasets store one folder per subject::

    Dataset/<Subject>/Trial_#/ProcessedData/...

The nested ``experiment`` layout adds the source study as an extra level::

    Dataset/<Experiment>/<Subject>/Trial_#/ProcessedData/...

Subjects are assigned to an experiment purely from their folder name, using the
same folder-prefix convention that ``scripts/estimate_mass_from_grf.py`` uses for
its cohorts.  ``OA``/``Y`` are one experiment, and the bare ``S#`` subjects share
an experiment with ``S_GAH_*``.

Rule order matters: ``SUBJ`` and ``S_GAH_`` must be tested before the bare ``S``
prefix, otherwise every stroke subject would land in the ``S_GAH`` group.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Written by scripts/reorganize_dataset_by_experiment.py. When present it is the
# authoritative list of experiment folders, which matters because a leftover
# directory of quarantined subjects (UnwantedSubjects/) is structurally
# indistinguishable from a real experiment: both contain subject folders.
LAYOUT_MANIFEST_NAME = "experiment_layout_manifest.json"

# Fallback guard for nested datasets built without a manifest. These sit beside
# the experiment folders and hold subjects that must never reach training.
NON_EXPERIMENT_DIR_NAMES: frozenset = frozenset(
    {
        "UnwantedSubjects",
        "BadTrials",
        "Quarantine",
        "OpenSimToMJX_Accuracy",
        "TempRemove",
    }
)

# (experiment_name, prefixes) in evaluation order.  ``None`` prefixes means the
# rule is a callable check handled in ``experiment_of_subject``.
EXPERIMENT_PREFIX_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Stroke", ("SUBJ",)),
    ("GaitRetraining", ("GaitRetraining",)),
    ("PD", ("PD_SUB",)),
    # Bertaux hip-osteoarthritis sessions. Not present in the trusted datasets as
    # of 2026-07-28 (they live in the standalone Hip_OA/ export), but the rule is
    # here so they group correctly if they are ever merged in.
    ("Hip_OA", ("HOA", "HEA")),
    # Silder 2008 older ("OA chars") and younger ("Young Walkers") adults.
    # ``OA`` here means Older Adult, NOT osteoarthritis - see Hip_OA above.
    ("OA_Y", ("OA", "Y")),
    ("S_GAH", ("S_GAH_", "S")),
)

# Subjects whose folder name starts with a digit (02, 04, ... 20).
NUMERIC_EXPERIMENT = "Numeric"

EXPERIMENT_NAMES: Tuple[str, ...] = (
    NUMERIC_EXPERIMENT,
    "Stroke",
    "GaitRetraining",
    "PD",
    "Hip_OA",
    "OA_Y",
    "S_GAH",
)

# Experiments kept out of every training set by default. They are still scored:
# each still gets its own hold-out round in the LOEO sweep.
DEFAULT_ALWAYS_EXCLUDED_EXPERIMENTS: Tuple[str, ...] = ("Hip_OA",)

# A directory is treated as a subject only if it carries this metadata file.
# It keeps non-subject siblings (UnwantedSubjects/, OpenSimToMJX_Accuracy/, ...)
# out of the reorganization and out of experiment discovery.
SUBJECT_MARKER_FILE = "Patient_MD.json"


def experiment_of_subject(subject: str) -> Optional[str]:
    """Return the experiment name for a subject folder, or None if unmatched."""
    name = str(subject).strip()
    if not name:
        return None
    if name[0].isdigit():
        return NUMERIC_EXPERIMENT
    for experiment, prefixes in EXPERIMENT_PREFIX_RULES:
        if name.startswith(prefixes):
            return experiment
    return None


def is_subject_dir(path: Path) -> bool:
    """True when ``path`` looks like a subject folder in either layout."""
    path = Path(path)
    if not path.is_dir():
        return False
    if (path / SUBJECT_MARKER_FILE).exists():
        return True
    return any(child.is_dir() and child.name.startswith("Trial_") for child in path.iterdir())


def detect_layout(data_dir: Path) -> str:
    """Return ``"experiment"`` if ``data_dir`` is already nested, else ``"trusted"``.

    A nested dataset has no subject folders directly under the root, but does have
    subject folders one level deeper.
    """
    data_dir = Path(data_dir)
    top = [p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if any(is_subject_dir(p) for p in top):
        return "trusted"
    for candidate in top:
        try:
            children = [c for c in candidate.iterdir() if c.is_dir() and not c.name.startswith(".")]
        except OSError:
            continue
        if any(is_subject_dir(c) for c in children):
            return "experiment"
    return "trusted"


def experiment_names_from_manifest(data_dir: Path) -> Optional[Set[str]]:
    """Experiment names recorded by the reorganization script, or None if unavailable."""
    manifest_path = Path(data_dir) / LAYOUT_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    names = {
        str(move["experiment"])
        for move in manifest.get("moves", [])
        if isinstance(move, dict) and move.get("experiment")
    }
    # Experiments staged into the dataset after the reorganization (e.g. OpenCapVal)
    # register themselves here rather than by moving subject folders.
    names |= {str(name) for name in manifest.get("experiments", []) if str(name)}
    return names or None


def list_experiment_dirs(data_dir: Path) -> List[Path]:
    """Return the experiment-level directories of a nested dataset, sorted by name.

    The manifest wins when it exists. Without one, fall back to "contains subject
    folders" minus ``NON_EXPERIMENT_DIR_NAMES`` - never treat an unrecognized
    sibling directory as an experiment, because doing so silently pulls
    quarantined subjects into training.
    """
    data_dir = Path(data_dir)
    declared = experiment_names_from_manifest(data_dir)

    found = []
    for candidate in sorted(data_dir.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        if declared is not None:
            if candidate.name in declared:
                found.append(candidate)
            continue
        if candidate.name in NON_EXPERIMENT_DIR_NAMES:
            continue
        try:
            children = [c for c in candidate.iterdir() if c.is_dir() and not c.name.startswith(".")]
        except OSError:
            continue
        if any(is_subject_dir(c) for c in children):
            found.append(candidate)
    return found


def group_subjects_by_experiment(subjects: Sequence[str]) -> Dict[str, List[str]]:
    """Bucket subject names into ``{experiment: [subject, ...]}`` (unmatched under ``None``)."""
    grouped: Dict[str, List[str]] = {}
    for subject in subjects:
        grouped.setdefault(experiment_of_subject(subject), []).append(str(subject))
    for names in grouped.values():
        names.sort()
    return grouped
