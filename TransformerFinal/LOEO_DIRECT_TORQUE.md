# Leave-one-experiment-out (LOEO) direct-torque sweep

Trains one direct-torque model per source experiment, each time holding that
experiment out entirely, runs inference on the held-out experiment with the model
that never saw it, and reports accuracy per experiment and over the whole dataset.

## 1. Reorganize the dataset

The sweep needs the nested layout:

```
Dataset/<Experiment>/<Subject>/Trial_#/ProcessedData/...
```

```bash
# dry run - prints the plan, moves nothing
python scripts/reorganize_dataset_by_experiment.py \
    --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients

# perform the moves (writes experiment_layout_manifest.json at the dataset root)
python scripts/reorganize_dataset_by_experiment.py \
    --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients --apply

# undo, using that manifest
python scripts/reorganize_dataset_by_experiment.py \
    --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients --revert
```

Experiments come from folder-name prefixes, defined in
[experiment_groups.py](experiment_groups.py). For
`TrustedDataSetNoised12Distributed_EdgeHold_AllPatients` the dry run reports:

| Experiment       | Subjects | Folder prefixes                       |
| ---------------- | -------: | ------------------------------------- |
| `Stroke`         |      105 | `SUBJ*`                               |
| `GaitRetraining` |       83 | `GaitRetraining_Subject*` (incl. `R`) |
| `OA_Y`           |       43 | `OA*`, `Y*`                           |
| `PD`             |       39 | `PD_SUB##_on` / `_off`                |
| `S_GAH`          |       24 | `S#`, `S_GAH_*`                       |
| `Numeric`        |       16 | `02` … `20`                           |
| `Hip_OA`         |        0 | `HOA*`, `HEA*` — rule only, see below |

> **`OA` here means Older Adult, not osteoarthritis.** The `OA*`/`Y*` subjects are
> Silder 2008 older and younger walkers. The hip-osteoarthritis cohort is the
> Bertaux export in the standalone `Hip_OA/` directory (`HOA*`/`HEA*` sessions),
> which is *not* part of the trusted datasets. The `Hip_OA` rule exists so those
> sessions group correctly if they are ever merged in.

Only folders containing `Patient_MD.json` are treated as subjects, so
`UnwantedSubjects/` and `OpenSimToMJX_Accuracy/` stay at the dataset root.
Applying the move also deletes the stale `trial_discovery_cache.json`.

### Adding the OpenCapVal experiment

The OpenCap normal-walking trials live in a different layout
(`subjectN/trial_M/{Video,MoCap}/ProcessedData`) and are staged in separately:

```bash
python scripts/stage_opencap_val_experiment.py --dest TrustedDataSet_ByExperiment          # dry run
python scripts/stage_opencap_val_experiment.py --dest TrustedDataSet_ByExperiment --apply
python scripts/stage_opencap_val_experiment.py --dest TrustedDataSet_ByExperiment --revert
```

This copies **non-trunk-sway trials only** — the source `dataset_manifest.json`
splits each subject into `normal` and `trunk_sway` groups, and only `normal` is
taken (the `subjectN_TS` folders are left behind) — using **`MoCap/ProcessedData`**
as the source. Result: 10 subjects × 3 trials = 30 trials, 119 MB.

Because these subjects are copied in rather than moved, the script registers
`OpenCapVal` under an `experiments` key in `experiment_layout_manifest.json`.
Without that registration discovery would skip the folder, exactly as it skips
`UnwantedSubjects/`.

### Trials with no noised bundle

The OpenCap cohort was never run through the noising pipeline, so it has no
`_noised.npy` files. The sweep trains with `use_noised=True`, and by default such a
trial is dropped — silently, since the loader only prints an aggregate
`Skipped N trials` line.

`--allow_missing_noised` (**default `True`** in the wrapper) makes a trial whose
ProcessedData contains *no* `_noised.npy` files at all fall back to its clean
files, logging one line per trial. A **partial** bundle still fails strictly: that
means the noising run broke, and quietly mixing noised and clean sources would be
a data-integrity bug that is very hard to notice later.

The check exists in two places that must stay in agreement —
`TrialDataLoader.__init__`'s required-file pre-scan and `load_single_trial` — since
the pre-scan decides whether a trial is ever handed to the loader at all.

### Non-experiment folders at the dataset root

`UnwantedSubjects/` holds quarantined subjects. In the flat layout they are
invisible to discovery because they sit one level too deep; once the dataset is
nested they become structurally indistinguishable from a real experiment folder
and would silently enter training. Discovery therefore takes the experiment list
from `experiment_layout_manifest.json` when it exists, and otherwise falls back to
"contains subject folders" minus `NON_EXPERIMENT_DIR_NAMES`
([experiment_groups.py](experiment_groups.py)). The scan prints which top-level
directories it skipped.

Verify after reorganizing — the nested dataset must discover exactly the trials the
flat one did:

```python
from train import discover_all_trials
nested = discover_all_trials("<new>", refresh_cache=True, layout="experiment")
flat   = discover_all_trials("<old>", refresh_cache=True, layout="trusted")
assert {(t["subject"], t["trial"]) for t in nested} == {(t["subject"], t["trial"]) for t in flat}
```

## 2. Run the sweep

```bash
python TransformerFinal/loeo_direct_torque.py \
    --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients \
    --output_root outputs/DirectTorque_LOEO \
    --epochs 40
```

Per experiment `E` the wrapper runs, in separate subprocesses so JAX releases GPU
memory between rounds:

1. `train_directTorque.py --layout experiment --exclude_experiments E`
   → `outputs/DirectTorque_LOEO/hold_out_E/best_model.pkl` (+ `train.log`,
   `split.json`, `hyperparameters.json`). The validation split for model
   selection is a random 20 % of the *training* subjects — `--val_fraction`
   — so the held-out experiment is never used for checkpoint selection.
2. `infer_directTorque.py --experiment E --write_to_trial_dir`
   → per-trial results inside the dataset, plus a round summary at
   `hold_out_E/inference/summary_metrics.json`.

Rounds that already have a `best_model.pkl` and an inference summary are skipped,
so an interrupted sweep just needs to be re-launched. `--force` redoes everything;
`--force_inference` keeps the checkpoints and re-runs inference only.

Useful flags: `--experiments A,B` (subset), `--skip_experiments C`,
`--train_arg` (repeatable pass-through to `train_directTorque.py`), plus the usual
architecture/optimizer flags, which are forwarded when set and otherwise leave
`train_directTorque.py`'s defaults alone.

### Cohorts barred from every training set

`--always_exclude_experiments` (default `Hip_OA`) names experiments that must never
appear in *any* model's training data, while still being evaluated — each keeps its
own hold-out round, so it still appears in the accuracy tables. For hold-out `E`
the training set becomes `all − {E} − always_excluded`.

Names that are not present in the dataset produce a one-line notice rather than an
error, which is what makes the `Hip_OA` default safe to carry today: the cohort
lives in the standalone `Hip_OA/` export and is not in the trusted datasets, so
nothing is currently excluded. If it is merged in later, the exclusion starts
applying with no code change. Pass `--always_exclude_experiments ""` to disable.

Both the requested and the actually-applied lists are recorded in
`sweep_config.json`, and each round prints the experiments it trains on.

## 3. Inference results in the dataset

Each held-out trial gets:

```
Dataset/<Experiment>/<Subject>/Trial_#/inference_results/
├── direct_torque_pred_percent_bwh.npy   # (T, 14) prediction, %BW*H
├── direct_torque_gt_percent_bwh.npy     # (T, 14) target,     %BW*H
├── direct_torque_pred_nm.npy            # (T, 14) prediction, N*m
├── direct_torque_gt_nm.npy              # (T, 14) target,     N*m
├── direct_torque_names.json             # channel order
├── evaluation_mask.npy                  # (T,) frames covered by a window
├── direct_torque_timeseries.png         # unless --no_inference_plots
└── metrics.json
```

Frames outside `evaluation_mask` are `NaN` in the waveform files.
`metrics.json` holds the trial's MAE/RMSE (all 14 channels, plus the left-leg-only
pair kept from earlier reports), per-channel MAE/RMSE/bias/`r`, and a `pooling`
block of per-channel sufficient statistics so any subset of trials can be pooled
exactly without re-reading the waveforms.

Use `--results_subdir <name>` to keep several runs side by side under
`inference_results/`.

## 4. Accuracy metrics

Written to `outputs/DirectTorque_LOEO/accuracy/` and printed at the end of the
sweep. Re-computable at any time without retraining:

```bash
python TransformerFinal/loeo_direct_torque.py \
    --data_dir TrustedDataSetNoised12Distributed_EdgeHold_AllPatients \
    --output_root outputs/DirectTorque_LOEO --aggregate_only
```

| File                        | Contents                                                           |
| --------------------------- | ------------------------------------------------------------------ |
| `loeo_accuracy.json`        | full report: per experiment and overall, micro and macro            |
| `per_experiment_metrics.csv`| one row per experiment, incl. per-channel MAE and `r`               |
| `per_subject_metrics.csv`   | one row per subject                                                 |
| `per_trial_metrics.csv`     | one row per trial                                                   |

All values are in %BW·H. Three averages are reported because they answer
different questions:

- **micro** — frame-weighted; every held-out frame counts equally.
- **macro over trials** — every trial counts equally.
- **macro over experiments** — every experiment counts equally, so the
  105-subject `Stroke` group does not drown out the 16-subject `Numeric` group.

## Notes

- `discover_all_trials(..., layout="experiment")` is the nested-layout scanner;
  `layout="trusted"` (flat) and `layout="opencap"` are unchanged. Discovered
  trials now carry an `experiment` key, empty for the flat layouts.
- `train_directTorque.py` gained `--exclude_experiments` / `--include_experiments`.
  Both refuse to run if no discovered trial carries an experiment, so a forgotten
  reorganization fails loudly instead of silently training on everything.
