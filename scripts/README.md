# scripts/

Standalone utilities. Nothing in the training or inference path imports from
here — these are run by hand, and each one takes `--help`.

| Directory | What lives there |
|---|---|
| [data_prep/](data_prep/) | Building, extracting, staging and converting datasets: C3D/`.mot` force extraction, cohort assembly, noised-variant generation, experiment reorganisation, model rescaling. |
| [opensim/](opensim/) | OpenSim inverse dynamics — generating its inputs, running it in batch, and validating MJX ID against it. |
| [analysis/](analysis/) | Read-only reporting: mass estimation from GRF, per-DOF error attribution, LOSO MAE comparison plots, shape verification. |
| [maintenance/](maintenance/) | Repo-structure migrations from REFACTOR_PLAN.md. Run once, kept for provenance. |
| [oneoff/](oneoff/) | Past investigations, kept for the analysis rather than for reuse. See [oneoff/README.md](oneoff/README.md). |
| [legacy_forward_sim/](legacy_forward_sim/), [legacy_scott_data/](legacy_scott_data/) | Superseded pipelines, retained for reference. |

## Two conventions worth knowing

**Paths.** Scripts resolve datasets and outputs through `paths.py`
(`REPO_ROOT`, `dataset()`, `artifact()`), not by walking up from `__file__`.
Before Stage 7 many computed the repo root as `Path(__file__).parents[1]`, which
silently became wrong the moment they were grouped into these subdirectories.
Relative `--dataset` arguments are resolved against the repository root, so it
does not matter what directory you run from.

**Imports inside a group.** `opensim/` scripts import each other by bare module
name (`import batch_opensim_inverse_dynamics`), which works because they sit in
one directory and Python puts the script's own directory on `sys.path`. That
cluster is import-closed and was moved as a unit for exactly this reason — do not
split it without converting those to package imports.

## Flag naming

Flags are inconsistent across these scripts: `--snake_case` dominates, but some
take `--CamelCase` and a few use string-valued booleans such as `--UseNoised
True`. Stage 7 deliberately did not normalise them. Renaming a flag is a
behaviour change that silently breaks saved commands and shell history, which is
a poor trade for cosmetic consistency. Check `--help` rather than assuming.
