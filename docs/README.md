# Documentation

Design notes and implementation plans. Until Stage 7 of
[REFACTOR_PLAN.md](../REFACTOR_PLAN.md) these sat at the repo root and inside
`TransformerFinal/`, and none of them was linked from anywhere — they were
effectively undiscoverable unless you already knew the filename.

| Document | What it covers | State |
|---|---|---|
| [LOEO_DIRECT_TORQUE.md](LOEO_DIRECT_TORQUE.md) | The leave-one-experiment-out sweep: dataset layout, how a round trains and infers, where results land, how accuracy is aggregated. | Describes shipped behaviour |
| [LOSO_OLDER_YOUNGER.md](LOSO_OLDER_YOUNGER.md) | Leave-one-subject-out fine-tuning over the older/younger cohort, for both the direct-torque and COP/GRF/moment model families. | Describes shipped behaviour |
| [TRUNK_SWAY_KAM_COHORT_PLAN.md](TRUNK_SWAY_KAM_COHORT_PLAN.md) | Cohort-level trunk-sway KAM aggregation across LOSO folds, plus new-model compatibility. | Code complete; acceptance runs outstanding |
| [KNEE_COUPLING_FIX_IMPLEMENTATION_PLAN.md](KNEE_COUPLING_FIX_IMPLEMENTATION_PLAN.md) | Knee-coupling correction and name-based qpos mapping in `ProcessData.py`. Self-contained analysis intended to be read before editing. | Implementation plan |

The two repo-level documents stay at the root, because they describe the
repository rather than a piece of work within it:

- [../README.md](../README.md) — what the project is and how to run it.
- [../REFACTOR_PLAN.md](../REFACTOR_PLAN.md) — the staged reorganisation, what is
  measured, and what is deferred.
- [../repo.md](../repo.md) — a denser orientation map aimed at code-reading tools.

## A note on paths

These documents refer to code as `TransformerFinal/train.py` — paths from the
repository root, not from this directory. They previously used bare filenames
that resolved only because the document sat beside the code. Stage 7 rewrote
them, and `tests/test_repo_references.py` fails if any of them stops resolving.
