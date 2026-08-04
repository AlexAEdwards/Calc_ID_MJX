# Calc_ID_MJX refactor plan

Staged, conservative plan to make this repo understandable and manageable.
Written 2026-07-28. **Re-measured 2026-07-31.**

> ### ⚠️ Read this before Stage 0
>
> **The GitHub remote is PUBLIC** (`AlexAEdwards/Calc_ID_MJX`, `private=false`,
> 1.30 GB, last pushed 2025-12-04) and its history already contains 27
> `Patient_MD.json` files under `Data_Full_Cleaned/`.
>
> The working tree now holds **568** `Patient_MD.json` files carrying
> `Height_m`, `Mass_kg`, `BiologicalSex`, `ImpairedLeg`, `PrePost`, across
> clinical cohorts (Stroke, Parkinson's `PD`, hip-osteoarthritis `Hip_OA`,
> `GaitRetraining`). None of it is gitignored today.
>
> `git add -A` would stage **419.9 GB** and publish that data. Stage 0 must
> therefore fix `.gitignore` *first* and commit by explicit path — never `-A`.
> Whether the repo should stay public is a decision for the owner, not a
> refactor step.

Every stage is independently valuable, independently revertible, and ends with a
concrete verification step. Stages are ordered so that each one makes the next
one safer — in particular **Stage 0 must come first**, because until it lands
nothing in this repo can be undone.

---

## 0. Where the repo actually stands

Measured, not estimated. Second column is the 2026-07-31 re-measurement:

| | 07-28 | **07-31** |
|---|---|---|
| Total size on disk | 614 GB | **644 GB** |
| Actual source code | ~15 MB / 233 files / 177k LOC | **~15 MB / 237 files / 180k LOC** |
| Entries at repo root | 64 (43 dirs) | **60 (39 dirs)** |
| `__main__` entry points | 145 | ~150 |
| Git commits | 4 | **4 (unchanged — still nothing committed)** |

Three days of work added **+30 GB of data and +4 source files, and zero commits.**
That is the plan's thesis restated as a measurement: the repo grows, the history
does not.

### What changed since 2026-07-28

Two datasets were built and one model sweep was rerun. Neither is in git:

| New | Size | Note |
|---|---:|---|
| `TrustedDataSet_ByExperiment/` | 89 GB | nested `Experiment/Subject/Trial` reorganisation of the flat trusted set |
| `KineticVAEDataset/` | 30 GB | curated hand-off copy (5,503 trials, self-describing `schema/`) |
| `outputs/DirectTorque_LOEO_edge70/` | — | 8-experiment LOEO re-sweep, new edge-frame policy |

New source files, all untracked: `TransformerFinal/experiment_groups.py`,
`loeo_direct_torque.py`, `scripts/data_prep/reorganize_dataset_by_experiment.py`,
`stage_opencap_val_experiment.py`, `build_kinetic_vae_dataset.py`,
`build_visual_trimmed_dataset.py`, `promote_hip_oa_experiment.py`,
`restore_knee_input_channels.py`, `verify_processdata_shapes.py`.

`KineticVAEDataset/` is now the strongest argument *for* Stage 1: it demonstrates
that a curated, documented, schema-carrying dataset can be generated from a script
in ~2 minutes. The same discipline applied to the repo root is Stage 1's goal.

### The critical finding: the code is not in version control

```
git tracked files        9,103   of which 8,854 are Data_Full_Cleaned/ (a dataset)
tracked files deleted
    in the working tree  8,877   including README.md
.git pack size           1.26 GiB
remote                   github.com/AlexAEdwards/Calc_ID_MJX (in sync with local HEAD)
```

`ProcessData.py`, `TransformerFinal/`, `scripts/`, `Loso_Combined.py` — the entire
working codebase — are **untracked**. What *is* tracked is a dataset that has since
been deleted from disk, plus ~10 stale root scripts. `myoconverter` is a gitlink
(commit `cadf3805`) with **no `.gitmodules`**, so a fresh clone cannot resolve it.

Consequence: there is currently no way to revert a bad refactor, no history to
bisect, and no diff to review. This dominates the ordering of everything below.

### The other four problems

1. **Data and code are siblings.** 174 GB `inference_results/`, 97 GB
   `Datasets_Local/`, 74 GB `outputs/`, three ~76 GB `TrustedDataSetNoised12*`
   variants, all at the repo root next to `train.py`. `ls` is unreadable.

2. **The code tree is triplicated.**

   | Tree | Python files | Last touched | Drift vs `TransformerFinal/` |
   |---|---:|---|---|
   | `TransformerFinal/` | 72 | 2026-07-28 | — (active) |
   | `TransformerFinalCopy/` | 57 | 2026-07-02 | `train.py` 1,089 diff lines; `loso_from_checkpoint.py` 2,023 |
   | `CHPCRepo/TransformerFinal/` | 6 | 2026-06-09 | `train.py` 1,701 diff lines |

   Plus 20+ byte-identical file pairs, including `TransformerFinal/infer_mod_q.py`
   duplicated verbatim at `TransformerFinal/NoiseModels/infer_mod_q.py`.

3. **God files.**

   | File | LOC | Notes |
   |---|---:|---|
   | `ProcessData.py` | 8,133 | 137 top-level defs, 34 CLI flags, no section headers |
   | `TransformerFinal/infer.py` | 7,923 | |
   | `TransformerFinal/loso_from_checkpoint.py` | 7,172 | |
   | `TransformerFinal/train.py` | 6,597 | also the de-facto shared library |
   | `Loso_Combined.py` | 4,308 | |

4. **No packaging.** No `pyproject.toml`/`setup.py`. 18 `sys.path.insert` calls,
   28 files independently re-deriving `PROJECT_ROOT`, 3 files with hardcoded
   `/home/mobl` paths. Tests (17 files, 7 of them duplicates) have no `conftest.py`
   and no CI.

### The good news, and why this is tractable

An AST scan of every import in `TransformerFinal/` + `scripts/` shows the
**cross-module API surface is tiny**:

| Module | LOC | Symbols imported elsewhere | Importing files |
|---|---:|---:|---:|
| `data_loader.py` | 2,394 | 16 | 13 |
| `train.py` | 6,597 | **14** | 5 |
| `infer.py` | 7,923 | 24 (mostly `_masked_*` helpers and `create_*_plot`) | 2 |
| `direct_torque_utils.py` | 235 | 7 | 6 |

Over 90 % of each god file is script-local. And the dependency graph is already
layered and acyclic:

```
opensim_id_targets ─┐
                    ├─> data_loader ─> train ─> {infer, train_directTorque, loso_*}
direct_torque_utils ┘
```

So the decomposition work is **mechanical extraction behind re-export shims**, not
an architectural redesign. That is what makes a conservative, low-risk sequence
possible.

---

## Stage 0 — Get the code under version control

**Goal:** make every subsequent change revertible and reviewable.
**Why first:** nothing else is safe until this exists.

**Actions**

1. Copy the ~15 MB of source out of the repo to an off-repo backup *before*
   touching git (`tar` the code dirs; exclude the data dirs).
2. Rewrite `.gitignore` to exclude data/artifacts by construction, not by
   enumeration. Note the current file has a dead rule — the
   `!TrustedDataSetNoised12Distributed_EdgeHold/` negation can never take effect,
   because git will not descend into a directory excluded by the preceding
   `TrustedDataSetNoised12Distributed*/` pattern.
3. Fix the `myoconverter` gitlink: either add a proper `.gitmodules`, or drop the
   gitlink and vendor/pin it explicitly.
4. Commit the real code in a few readable commits (core, scripts, analysis, docs).
5. Decide on `Data_Full_Cleaned/` (see *Decisions* below) — either commit its
   deletion, or purge it from history with `git-filter-repo` to take the clone
   from 1.26 GiB to a few MB.
6. Restore or rewrite `README.md` (currently tracked-but-deleted). Keep `repo.md`
   — it is a genuinely good orientation map — and cross-link the two.
7. **Decide the remote's visibility before pushing** (see the banner at the top).
   The repo is public and already carries 27 `Patient_MD.json` files. Pushing the
   code is harmless; pushing anything under a data directory is not.
8. Push. Confirm the remote now carries the code.

**Verification:** `git status` shows no untracked *source*; a fresh `git clone`
into a temp dir yields a working code tree of the expected size, and
`python -c "import ..."` succeeds for the core modules. Additionally
`git count-objects -vH` should not have grown by more than a few MB.

**Never use `git add -A` in this repo.** With the current `.gitignore` that stages
419.9 GB. Stage by explicit path, and re-check `git status --porcelain | grep '^A'`
before every commit until the ignore rules are proven.

**Risk:** Low for the code commit — additive and local until pushed. The optional
history purge rewrites shared history; coordinate first.
**Effort:** ~1 day. **Revert:** the off-repo backup from step 1.

### Status 2026-07-31 — partially done

- `.gitignore` rewritten to deny-by-default (data/artifact roots enumerated, plus
  the new `TrustedDataSet_ByExperiment/`, `KineticVAEDataset/`, `Datasets_Local/`,
  `inference_results/` which the old file did not cover).
- Source committed locally by explicit path.
- **Not pushed** — blocked on the visibility decision in step 7.
- `Data_Full_Cleaned/` deletion **not** committed; still an open decision.
- `myoconverter` gitlink still unresolved.

---

## Stage 1 — Separate data from code at the root

**Goal:** make `ls` at the repo root legible. 64 entries → ~15.

**Actions**

1. Create `datasets/` and `artifacts/` at the root.
2. `mv` the data directories in (same filesystem, so this is instantaneous and
   costs no disk): `Trusted*`, `OldYoungAdult*`, `OlderYounger*`, `OpenCap*`,
   `Hip_OA*`, `Datasets_Local`, `BadTrialsFromTrustedDataset`, `Data_Full_Cleaned`
   → `datasets/`; `outputs`, `output`, `inference_results`, `logs`, `figures`,
   `RMASBFigures`, `CHPC_HPO_results`, `tmp` → `artifacts/`.
3. **Leave a symlink at every old path** for one cycle. Nothing breaks on day one;
   scripts, notebooks, and half-finished runs keep working.
4. Add `paths.py`: one module that resolves dataset and artifact roots from an
   environment variable, defaulting to the new locations. Do not yet rewrite
   callers — just make the indirection available.
5. Consolidate the obvious redundancies: `output/` vs `outputs/` (3 dirs vs 42),
   and the 4.8 GB `OlderYoungerAdultDataset_PostVisuallyTrimmed (copy)/` — a
   parenthesised space in a path name is a latent bug in every shell script.

**Verification:** re-run one short pipeline of each kind (a `--aggregate_only`
metrics pass, a `scripts/` utility) and confirm identical output. Then grep for
readers of the old paths and migrate them to `paths.py` opportunistically.

**Risk:** Low-medium. Mitigated entirely by the compatibility symlinks.
**Effort:** ~half a day. **Revert:** `mv` back; the symlinks make this trivial.

> Deliberately **not** doing: moving 614 GB to another filesystem, or pruning any
> dataset. That is a storage decision, not a refactor, and should be taken
> separately.

---

## Stage 2 — Delete the duplicate code trees

**Goal:** remove ~80 files and ~40k LOC of near-identical code. Safe only *after*
Stage 0, because git history is what makes deletion non-destructive.

**Actions**

1. `TransformerFinalCopy/` is a dated manual backup (frozen 2026-07-02). Commit it
   on a tagged branch (`archive/transformerfinal-2026-07-02`), then delete it from
   `main`. Before deleting, diff it against `TransformerFinal/` and confirm nothing
   valuable lives only in the copy — `loso_from_checkpoint.py` differs by 2,023
   lines, so this diff needs a real read, not a skim.
2. `TransformerFinal/infer_mod_q.py` and `NoiseModels/infer_mod_q.py` are
   byte-identical: keep one, import it from the other location. Same for the 7
   duplicated `corruption_model/tests/`.
3. `CHPCRepo/TransformerFinal/` is a genuine HPC mirror that has drifted 1,701
   lines. Decide (see *Decisions*): either point CHPC at the packaged library from
   Stage 3, or keep it as an explicit vendored snapshot with a `sync_chpc.sh` and a
   header noting the sync date.
4. `Z...CodeIamNotReadyToDelete/` (11 files, untouched since 2026-04-28) — the name
   says it. Once Stage 0 preserves it in history, delete the directory; that *is*
   the "not ready to delete" safety net, made real.

**Verification:** `grep -r` for imports of every deleted path returns nothing; the
test suite and one smoke training run still pass.

**Risk:** Low, post-Stage-0. **Effort:** ~1 day (mostly the `Copy/` diff review).

---

## Stage 3 — Make the code an installable package

**Goal:** delete the 18 `sys.path.insert` hacks and the 28 ad-hoc `PROJECT_ROOT`
derivations; make imports work from any working directory.

**Actions**

1. Add `pyproject.toml` declaring the existing directories as packages. Do **not**
   move files yet — this stage is purely about making the current layout importable.
2. Add `__init__.py` where needed; `pip install -e .` into the working conda env.
3. Replace `sys.path.insert` blocks with real imports, one file at a time.
4. Route the 3 hardcoded `/home/mobl` paths through `paths.py` from Stage 1.
5. Pin the environment properly: `full_env.yml` and `requirements_env.txt` both
   exist and both predate April; reconcile them into one source of truth and
   record that JAX/flax currently live only in the `myoconverter` conda env.

**Verification:** `python -c "import <each top-level module>"` from `/tmp`; every
entry point still runs.

**Risk:** Low. **Effort:** ~1 day.

---

## Stage 4 — Build the safety net before surgery

**Goal:** characterization tests, so Stages 5–6 can be proven behaviour-preserving
rather than hoped to be.

**Actions**

1. Add a root `conftest.py` and a `tests/` layout; deduplicate the corruption_model
   suite (down from 14 files to 7).
2. **Golden-output fixtures**: stage 2–3 real trials in a small fixture dataset
   (the LOEO smoke-test pattern already used in this repo works well — copy real
   trials into a scratch dir so nothing writes back into the datasets). Record the
   `.npy` outputs of `ProcessData.py` and of one short training + inference run.
3. Add fast, GPU-free unit tests for the pieces about to move: trial discovery
   (all three layouts), `Normalizer` round-trip, `build_direct_torque_targets`,
   the `_masked_*` metric family, window/supervision-mask construction.
4. Add a cheap CI job: import-everything + the GPU-free tests. Nothing that needs a
   dataset or a GPU.

5. **`tools/equivalence_check.py`** — the load-bearing piece. A single command
   that re-runs a fixed set of real trials through the pipeline and diffs every
   produced array against a recorded baseline, exiting non-zero on any
   difference. Stages 5–7 each end by running it.

**Verification:** the golden tests pass on today's unmodified code. That is the
whole point — they must be green *before* anything moves.

**Risk:** None (additive). **Effort:** ~1–2 days.

### The equivalence contract for Stages 5–7

Every stage from here on is a **pure reorganisation**: the code must produce
bit-identical output, and that is checked mechanically rather than argued.

**Baseline.** Before Stage 5 begins, record a baseline from the current code:

* `ProcessData.py` outputs for a fixed set of real trials — every `.npy` in
  `ProcessedData/`, stored as SHA-256 per file.
* One short training run (fixed `--seed`, 2 epochs) — `best_model.pkl` metrics
  and the full `history.json`.
* One inference pass — `direct_torque_pred_*.npy`, both masks, `metrics.json`.
* The LOEO aggregation over the existing sweep — `loeo_accuracy.json`.

**Tolerances, stated up front so they cannot be quietly loosened:**

| Output | Required agreement |
|---|---|
| `ProcessData` `.npy` outputs | **byte-identical** (SHA-256) |
| Inference predictions | **byte-identical** — same checkpoint, same input, no RNG |
| Aggregated metrics (MAE, RMSE, R²) | **exact to all printed digits** |
| Training loss/metric history | `atol=0` on a fixed seed; any drift is a bug, not noise |
| Wall-clock, file layout, log text | free to change |

Anything that cannot be made bit-identical is **not** a reorganisation and does
not belong in Stages 5–7. It is a behaviour change and needs its own decision,
its own commit, and a recorded before/after — exactly how the edge-frame change
was handled.

**Per-commit discipline.** Stage 6 in particular moves one function cluster per
commit, and `tools/equivalence_check.py` runs after each. A commit that cannot
show a clean equivalence report is reverted rather than debugged forward, so the
tree is never left in a state where it is unclear whether output changed.

### ProcessData determinism — measured 2026-08-03

Before Stage 6 could be planned honestly, `ProcessData.py` had to be shown
reproducible. It is:

* Re-running it twice on identical input produced **74/74 byte-identical**
  `.npy` files. There is no RNG in the module, and no thread-order sensitivity
  in the output.
* Therefore Stage 6's verification is viable: regenerate a fixture subject with
  the code *before* a commit and *after* it, and compare by SHA-256.

**The protocol is before/after on the same input — not "reproduce the stored
dataset".** Regenerating a stored trial does *not* reproduce it, for a reason
that is not a defect: several cohorts were post-processed after `ProcessData`
ran. `Hip_OA/HEA121_Marche/Trial_1` records
`core_trim_pretrim_n_frames: 394` and `manual_visual_trim_applied: true`, but its
`Motion/` on disk is the already-trimmed 121-frame version, so a fresh run
re-derives floor height from 9 toe troughs instead of 7 and every downstream
geometry array shifts. Feeding post-processed input back through the pipeline is
simply not the same computation.

Two consequences worth recording separately from the refactor:

1. Stage 6 must regenerate **before and after** within the same session, from the
   same inputs, and compare those two — never against the shipped dataset.
2. The shipped datasets are not reproducible from `Motion/` alone. Reproducing
   them needs the raw pre-trim motion plus the post-processing scripts in the
   right order. That is a provenance gap, not a refactor task, but it is worth
   knowing before anyone tries to regenerate a cohort from scratch.

**Why this matters here specifically.** `ProcessData.py` is the preprocessing
spine for every dataset in `datasets/`. A silent change to a filter cutoff or a
column order would not raise an error — it would quietly poison every dataset
regenerated afterwards, and would only surface as unexplained model accuracy
drift weeks later. The equivalence check is what makes that failure mode
impossible rather than unlikely.

---

## Stage 5 — Extract the shared core from `train.py` and `infer.py`

**Goal:** turn the two biggest library-shaped god files into a small core package
plus thin CLIs, without changing a single import site.

This is safe precisely because the measured surface is small: **14 symbols** of
`train.py` and **24** of `infer.py` are used elsewhere.

**Actions**

1. Create `core/` and move, one commit per module:
   - `core/discovery.py` ← `discover_all_trials`
   - `core/normalization.py` ← `Normalizer`, `normalize_batch`, `compute_normalizers_from_loader`
   - `core/layers.py` ← `SinusoidalPosEmb`, `TransformerBlock`, `TaskHead`
   - `core/models.py` ← `KinematicsToCOPGRFMoments` (+ its alias)
   - `core/physics.py` ← `compute_full_external_moments`, `compute_tau_grf_from_predictions`, the rotation/jacobian helpers
   - `core/train_state.py` ← `create_train_state`, `compute_total_loss`
   - `evaluation/metrics.py` ← the `_masked_*` family from `infer.py`
   - `evaluation/plots.py` ← the `create_*_plot` / dashboard functions from `infer.py`
2. **Leave a re-export shim in the original file** (`from core.normalization import
   Normalizer  # re-exported for back-compat`). Every existing
   `from train import Normalizer` keeps working, so no caller changes in this stage.
3. `train.py` and `infer.py` are then CLI + training/eval loop only.
4. Only afterwards, and as a separate commit, migrate call sites to import from
   `core/` directly and drop the shims.

**Verification:** golden tests from Stage 4; plus a 2-epoch training run and an
inference pass compared numerically against a pre-refactor run.

**Risk:** Medium, well-mitigated. The shims mean any missed symbol shows up as an
`ImportError` at startup, not as a silent numerical change.
**Effort:** ~2–3 days.

> Note `loso_from_checkpoint.py` (7,172 LOC) exposes only 2 symbols to 1 file. It is
> a script, not a library — leave it whole in this stage and revisit later.

---

## Stage 6 — Split `ProcessData.py` by pipeline phase

**Goal:** 8,133 lines / 137 functions / 34 CLI flags → a `processing/` package.

`ProcessData.py` has **zero importers** in live code (only the dead `Z...` dir), so
it can be split freely. Its functions already cluster cleanly by phase:

| Proposed module | Content (existing functions) |
|---|---|
| `processing/filtering.py` | `butter_lowpass_filter`, `gcv_derivatives`, `apply_kinematics_filtering`, `filter_segment_wise` |
| `processing/contact.py` | `create_contact_boolean`, `get_stance_phases`, `zero_short_grf_cop_stances` |
| `processing/cop.py` | `clean_and_filter_cop`, `_multiply_cop_by_bodyweight_normalized_grf_magnitude` |
| `processing/geometry.py` | floor-height estimation, pelvis alignment, ground-aligned rotations, foot progression, knee-to-COP |
| `processing/model_indexing.py` | `_joint_id`, `_build_name_to_qpos_index`, `canonical_save_indices`, `map_patient_to_qpos`, coupled coordinates |
| `processing/trimming.py` | backup/restore, trim bounds, `_trim_array_time_axis`, `_rewrite_trimmed_from_backup` |
| `processing/provenance.py` | the `_trace_stage` / manifest / JSON-atomic-write family |
| `processing/cli.py` | argument parsing + orchestration |

**Actions:** move one cluster per commit, running the Stage-4 golden test after
each. Keep `ProcessData.py` as a thin entry point that imports and dispatches.

### Status 2026-08-03 — done

Six clusters moved, one commit each, `processdata_roundtrip` run and clean after
every one. `ProcessData.py` 8,156 → 7,319 lines; `processing/` is 1,118 LOC
across `filtering`, `contact`, `cop`, `geometry`, `resampling`, `trial_io` and
`artifact_names`.

Two things this stage taught, both worth keeping:

1. **One cluster per commit is not bureaucracy.** A first attempt moved five
   clusters at once, the gate reported 148 differences, and there was no way to
   attribute them. Reverted and redone one at a time: all six passed.
2. **The equivalence gate cannot see unexercised code.** `gcv_derivatives` was
   extracted referencing an unimported `make_smoothing_spline` and the gate
   stayed green, because it is the opt-in OpenSim-filtering path and the fixture
   never calls it. `tests/test_extracted_modules_resolve.py` now closes that hole
   statically for every module in `processing/` and `core/`. Any future
   extraction is covered by *both* checks, and only the pair is sufficient.

`equivalence_check.py` also now refuses to run pipeline layers under an
interpreter lacking jax/mujoco - the wrong env made every ProcessData subprocess
fail and rendered as output differences indistinguishable from a real break.

The remaining ~120 definitions in `ProcessData.py` (model indexing, trimming,
provenance, CLI orchestration) are the less separable half; splitting them is
optional and can follow the same protocol whenever it is wanted.

**Verification:** byte-for-byte identical `.npy` outputs on the golden trials after
every single commit. Non-negotiable — this is the preprocessing spine, and errors
here silently poison every downstream dataset.

**Risk:** Medium-high in principle; reduced to low by the golden test.
**Effort:** ~2–3 days.

---

## Stage 7 — Entry points, one-off scripts, and docs

**Goal:** make the 145 entry points navigable.

**Actions**

1. Sort root-level one-offs by last-touched date. `clean_cop_below_vgrf_threshold.py`,
   `Clean_COP_experiment.py`, `find_short_grf_stances.py`, `restore_untrimmed.py`,
   `count_dataset_stats.py`, `replace_kinematics_from_id_mot.py` are all April–May
   experiments; move to `scripts/oneoff/` or delete (Stage 0 preserves them).
2. Group `scripts/` (30 entry points) into `scripts/data_prep/`,
   `scripts/opensim/`, `scripts/validation/`, `scripts/analysis/`.
3. Fold the standalone analysis dirs (`AnklePowerAnalysis/`, `HPOAnalysis/`,
   `AccuracyByGender&Speed/`, `NoiseAndPowerAnalOfInputData/`, `VisAndAnalDataset/`)
   under a single `analysis/` parent.
4. Adopt one CLI convention across `scripts/` (they currently mix `--snake_case`,
   `--CamelCase`, and `--UseNoised`-style string-boolean flags).
5. Update `repo.md` and `README.md` to match the new layout, and consolidate the
   four scattered plan documents (`KNEE_COUPLING_FIX_IMPLEMENTATION_PLAN.md`,
   `TRUNK_SWAY_KAM_COHORT_PLAN.md`, `MODEL_ISSUES_AND_IMPROVEMENT_PLANS.md`,
   `LOEO_DIRECT_TORQUE.md`) into a `docs/` directory.

**Risk:** Low. **Effort:** ~1–2 days.

### Status 2026-08-04 — done

Root went from 41 tracked entries to 16. `scripts/` is grouped into `data_prep/`
(23), `opensim/` (9), `analysis/` (5), `maintenance/` (2) alongside `oneoff/` and
the `legacy_*` directories; the four plan documents moved to `docs/` with an
index; `repo.md` and `README.md` match the layout.

**Stage 7 needed a different gate.** Stages 5–6 were guarded by
`tools/equivalence_check.py`, which asks whether the output changed. Moving files
does not change output — a script that stops resolving its imports produces no
arrays to differ — so two new checks were built *before* anything moved:

* `tools/entrypoint_check.py` runs all 94 entry points with `--help` against a
  recorded baseline. 76 start; the 18 that do not are pre-existing and recorded,
  so they cannot mask a regression. Unchanged across every move.
* `tests/test_repo_references.py` asserts every in-repo path mentioned in code or
  docs resolves — against the **tracked** set, not the working tree.

**Three latent bugs surfaced, none of them about file locations:**

1. `ProcessData.py` imported `ProcessAddbiomechnics.updateModel` under a bare
   `except Exception`, and that directory was gitignored among the cohort
   *datasets* despite holding no data. Every fresh clone silently lost mass
   fixing and knee-coupling validation. The tracked repo-root `updateModel.py`
   was meanwhile a stale 760-line near-duplicate of the live 916-line module,
   with no function unique to it.
2. `scripts/data_prep/generate_noised12_distributed.py` imported
   `generate_sine_noise` from another gitignored directory — the same fault.
   `tests/test_no_untracked_imports.py` now prevents a third.
3. 17 scripts computed the repo root as `Path(__file__).resolve().parents[1]`,
   correct only while sitting one level below the root. Grouping `scripts/` would
   have rebased them all onto `scripts/` with no error raised. Rewired to
   `paths.REPO_ROOT` as a separate commit before any move.

**Deliberately not done.** Item 4, normalising `--CamelCase` and string-boolean
flags: renaming a flag breaks saved commands and shell history for no functional
gain. Recorded in `scripts/README.md`. Item 3, folding the standalone analysis
directories under one `analysis/` parent: they are gitignored working
directories, absent from any clone, so the move buys nothing for the repository
while being unrevertible by git and breaking a live import path. The docs now
state plainly that they are not part of a clone.

---

## Summary

| Stage | Outcome | Risk | Effort | Status |
|---|---|---|---|---|
| 0 | Code in git; every later stage revertible | Low | 1 d | **done** — 3 commits, pushed |
| 1 | Data/code separated | Low-med | 0.5 d | **done** — 642 GB into `datasets/` + `artifacts/` in 0.064 s |
| 2 | Duplicate trees deleted | Low | 1 d | **done** — −62 files, −59,915 LOC (−33 %) |
| 3 | Installable package; no `sys.path` hacks; symlinks retired | Low | 1 d | **done** — hacks 32→3, 52 paths migrated, root 64→41 |
| 4 | Golden + unit tests, equivalence harness, cheap CI | None | 1–2 d | **done** — 199 MB fixture, 8 layers, 80 tests, CI green |
| 5 | `core/` + `evaluation/` extracted behind shims | Medium | 2–3 d | **done** — 18 symbols into `core/` (1,273 LOC), gate clean |
| 6 | `ProcessData.py` → `processing/` package | Med-high | 2–3 d | **done** — 8,156 → 7,319 lines, 7 modules, gate clean per commit |
| 7 | Entry points, scripts, docs organised | Low | 1–2 d | **done** — root 41→16, scripts grouped, 2 latent import bugs fixed |

Roughly 10–14 working days end to end. Stages 0–3 are complete and delivered most
of the "understandable and manageable" benefit: the code is in version control,
data and code are separated, there is one copy of every module, and imports work
from any directory. Measured result: **237 → 175 Python files, 179,877 → 119,962
LOC, root 64 → 41 entries**, with the LOEO metric (MAE 0.3745) verified unchanged
at every step.

Stages 4–7 are a different kind of work: they make the *code* comprehensible
rather than the *repo* navigable, and 5–6 are the only genuinely risky items in
the plan. They are governed by the equivalence contract in Stage 4.

**Stop-the-line rule:** do not start Stage 5 or 6 until Stage 4's golden tests are
green on unmodified code, and do not land any commit in Stages 5–7 whose
`tools/equivalence_check.py` run is not clean. Stages 5–7 are reorganisations:
if output changed, the commit is wrong.

---

## Decisions needed before execution

1. **`Data_Full_Cleaned/` history** — it is 8,854 tracked files, already deleted
   from disk, and is most of the 1.26 GiB pack. Purge it from history (small clone,
   but rewrites shared history) or keep it (simpler, stays heavy)?
2. **`CHPCRepo/`** — still in active use on the cluster? That decides whether
   Stage 2 unifies it against the packaged library or freezes it as a snapshot.
3. **`TransformerFinalCopy/`** — confirm it is purely a backup. Its
   `loso_from_checkpoint.py` differs by 2,023 lines, which is a lot of drift for
   something described as a copy.
4. **Dataset location** — are the ~600 GB of datasets meant to live inside the repo
   long-term, or move to the NAS mount (`Datasets_NAS`) with the repo holding only
   pointers?
5. **In-flight runs** — any long jobs currently reading/writing `outputs/` or
   `inference_results/`? Stage 1 should wait for a quiet window.
