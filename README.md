# Calc_ID_MJX

Predict **joint kinetics from gait kinematics**, and verify the prediction is
physically consistent using MuJoCo/MJX inverse dynamics.

A Transformer takes kinematics (joint angles, velocities, accelerations, plus
subject scale) and predicts joint torques directly, in units normalised by
body-weight × height. Ground truth comes from MJX inverse dynamics, cross-checked
against OpenSim ID where available. The physical consistency check is
`τ = Jᵀ·F`: the external forces implied by a prediction must reproduce the joint
torques actually observed.

> **Data is not in this repository.** The datasets contain per-subject clinical
> metadata and live outside git (see [Data](#data)). Nothing under `datasets/` or
> `artifacts/` is tracked.

## What is here

```
core/            shared model, loss and physics code
processing/      ProcessData pipeline, split by phase
TransformerFinal/  training, inference, LOSO/LOEO drivers
scripts/         data preparation, OpenSim/MJX utilities, validation
tools/           test-fixture staging and the equivalence harness
tests/           unit tests + the recorded equivalence baseline
paths.py         single source of truth for where data and outputs live
ProcessData.py   raw motion -> ProcessedData (the preprocessing spine)
```

Datasets and run outputs live under `datasets/` and `artifacts/`, both gitignored.
Override either location without touching code:

```bash
export CALCID_DATASETS=/mnt/nas/Datasets
export CALCID_ARTIFACTS=/scratch/runs
```

## Install

Requires Python 3.10 and a JAX/MuJoCo environment; `full_env.yml` pins one.

```bash
conda env create -f full_env.yml
conda activate <env>
pip install -e .            # makes `import paths`, `import core.*` work anywhere
```

The package installs editable purely so the existing layout is importable — it
declares no runtime dependencies and will not fight the conda environment.

## Model

14 torque channels, bilateral:

```
hip_flexion, hip_adduction, hip_rotation, knee_flexion,
knee_adduction, ankle_flexion, subtalar          × {right, left}
```

Knee adduction is computed from the knee-to-COP lever arm and GRF; the other
twelve come from the 23-DOF inverse-dynamics bundle. Everything is normalised as
`τ / (mass · height · 9.8067) × 100` — percent body-weight × height.

## Evaluation: leave-one-experiment-out

Subjects are grouped into source studies, and each cohort is scored by a model
that **never saw it during training**:

| Cohort | Source |
|---|---|
| `Numeric` | AddBiomechanics healthy walking |
| `Stroke` | public stroke dataset (`SUBJ*`) |
| `GaitRetraining` | gait-retraining study |
| `OA_Y` | Silder 2008 older + younger adults — **`OA` = Older Adult, not osteoarthritis** |
| `PD` | Parkinson's, on/off medication |
| `S_GAH` | gastrocnemius-avoidance protocol |
| `OpenCapVal` | OpenCap validation subjects, normal walking |
| `Hip_OA` | Bertaux hip osteoarthritis — **never used for training** |

Current result over 5,498 held-out trials / 559 subjects:

| Metric | Value |
|---|---|
| MAE (frame-weighted) | 0.3745 %BW·H |
| MAE (per trial) | 0.4121 |
| MAE (per experiment) | 0.3984 |
| R² pooled | ~0.92 |

Quote the per-experiment mean: two long-trial cohorts hold ~88% of all frames, so
the frame-weighted figure mostly describes them.

```bash
python TransformerFinal/loeo_direct_torque.py \
    --data_dir TrustedDataSet_ByExperiment \
    --output_root outputs/DirectTorque_LOEO --epochs 40

# re-aggregate without retraining
python TransformerFinal/loeo_direct_torque.py --output_root outputs/... --aggregate_only
```

Each round trains excluding one cohort, runs inference on it, and writes results
into `<Subject>/Trial_#/inference_results/` inside the dataset.

## Data

Trials are stored `Dataset/<Experiment>/<Subject>/Trial_#/`, each with
`ProcessedData/` (model-ready arrays) and `Motion/` (raw signals).

`KineticVAEDataset/` is the curated hand-off copy: 5,503 trials with a
self-describing `schema/`, per-subject MuJoCo qpos layouts, model predictions, and
a runnable `example_torque_reconstruction.py`. Build it with:

```bash
python scripts/data_prep/build_kinetic_vae_dataset.py --dest KineticVAEDataset --apply
```

Three things that regularly catch people out, all documented in that dataset's
`schema/`:

- **`pos_mjx` column count varies per subject** (23 / 33 / 43 depending on the
  MuJoCo model). Read `models/mjx_qpos_layout.json`; never assume a width.
- **Pelvis translations are model-frame with treadmill travel accumulated.** They
  are not lab coordinates and will not match `raw/Pos.npy`.
- **`raw/` is longer than the processed arrays and is not a slice of them** —
  processing trims *and* low-pass filters.

## Tests

```bash
pytest tests/ -q                          # unit tests; jax-dependent ones self-skip
python tools/stage_test_fixture.py --apply   # ~200 MB fixture, 10 trials, 8 cohorts
python tools/equivalence_check.py            # diff every output against the baseline
```

`tools/equivalence_check.py` is the refactoring safety net. It re-runs fixed real
trials through discovery, the loader, target construction, the metric helpers, the
training loss, LOEO aggregation, and a full `ProcessData` round trip, hashing raw
array bytes and exiting non-zero on any difference. It has been verified to
*fail* on a deliberate 2% filter change — a gate that has only ever passed proves
nothing.

## Documentation

[docs/](docs/) holds the design notes and implementation plans — the LOEO sweep,
older/younger LOSO, trunk-sway KAM aggregation, and the knee-coupling fix. See
[docs/README.md](docs/README.md) for what each one covers and whether it
describes shipped behaviour or outstanding work.

## Repository state

An ongoing reorganisation is tracked in [REFACTOR_PLAN.md](REFACTOR_PLAN.md),
which records what has been done, what is measured, and what is deferred.
[repo.md](repo.md) is a denser orientation map aimed at code-reading tools.
