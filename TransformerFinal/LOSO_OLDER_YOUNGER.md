# Older/Younger trusted-layout LOSO

`loso_older_younger_from_checkpoint.py` fine-tunes either supported Transformer family while holding out one complete subject at a time:

- `direct_torque`: kinematics directly to 14 joint-torque channels.
- `cop_grf_moments`: kinematics to COP/GRF/free moments, with physics-based torque reconstruction.

The dataset must use `Subject/Trial_*/ProcessedData`, for example `Y1/Trial_1/ProcessedData` and `OA1/Trial_1/ProcessedData`. Subject names are not hard-coded, so the post-trimmed, pre-trimmed, and copied datasets can use the same command. The source checkpoint's normalizers are retained for every fold.

Run discovery without JAX or a checkpoint:

```bash
python TransformerFinal/loso_older_younger_from_checkpoint.py \
  --data_dir OlderYoungerAdultDataset_PostVisuallyTrimmed \
  --discovery_only
```

On the current post-trimmed dataset this reports 43 subject directories, 42 subjects with usable trials, and 607 trials of at least 30 frames. Empty subjects and short or unreadable trials are listed as skipped.

Check checkpoint compatibility without training:

```bash
python TransformerFinal/loso_older_younger_from_checkpoint.py \
  --checkpoint /path/to/run/best_model.pkl \
  --data_dir OlderYoungerAdultDataset_PostVisuallyTrimmed \
  --dry_run
```

Use the same Python environment used to train the Transformer; JAX, Flax, and Optax are required for checkpoint validation and fine-tuning. A full direct-torque run is:

```bash
python TransformerFinal/loso_older_younger_from_checkpoint.py \
  --checkpoint /path/to/direct_torque/best_model.pkl \
  --data_dir OlderYoungerAdultDataset_PostVisuallyTrimmed \
  --output_dir outputs/loso_older_younger/direct_torque \
  --epochs 6
```

Run the COP/GRF model in a separate process by changing the checkpoint and output directory. Separate processes bound JAX memory use and make resuming safer.

For a quick smoke run, restrict folds and trials:

```bash
python TransformerFinal/loso_older_younger_from_checkpoint.py \
  --checkpoint /path/to/best_model.pkl \
  --data_dir OlderYoungerAdultDataset_PostVisuallyTrimmed \
  --held_out_subjects Y1 \
  --max_trials_per_subject 1 \
  --epochs 1 \
  --output_dir outputs/loso_smoke
```

Subject list arguments accept spaces or commas. `--include_subjects` limits the entire experiment population, `--exclude_subjects` removes subjects entirely, and `--held_out_subjects` selects which folds run while retaining all other included subjects for training.

`--use_noised auto` follows the checkpoint. Use `--use_noised false` only as an intentional feature-source override; the input dimension is validated before any training. Missing noised inputs fail with the exact first missing path.

Use `--resume` to skip folds with `FINETUNE_COMPLETE.json`. Resume validates the dataset, checkpoint SHA-256, and training settings. `--overwrite_fold` reruns selected folds. Each fold writes `split.json`, `training_history.json`, `metrics.json`, `best_model.pkl`, and `hyperparameters.json`, then performs paired source-versus-fine-tuned inference on identical windows. Raw arrays go under `inference/`; trial, subject, cohort, and overall reports are written as JSON/CSV. The completion marker is created only after paired inference and reporting succeed. `epochs=0` saves unchanged source parameters and should produce zero original-versus-fine-tuned prediction difference.

Each fold also writes stance-level dual-source ankle-power JSON files under
`held_out_<subject>/ankle_power_results/`. These retain legacy `pred_*` fields
as aliases for the original checkpoint and add separate `original_pred_*` and
`fine_tuned_pred_*` values. Run the existing analysis directly on a fold:

```bash
python AnklePowerAnalysis/run_ankle_power_analysis.py \
  outputs/loso_older_younger/direct_torque/held_out_Y1/ankle_power_results \
  --output_dir outputs/loso_older_younger/direct_torque/held_out_Y1/ankle_power_analysis
```

The regression, stance-curve, effect-size, and cohort plots then include both
“Original Prediction” and “Fine-Tuned Prediction.”

To pool every held-out subject into one Older-versus-Younger analysis, point
the command at the LOSO run root:

```bash
python AnklePowerAnalysis/run_ankle_power_analysis.py \
  outputs/loso_olderYounger \
  --output_dir outputs/loso_olderYounger/ankle_power_analysis_all_subjects \
  --included_all_trials --gt_min_wkg 0 --edge_exclude_frames 0
```

The pooled runner also fits random-intercept mixed-effects models for every
available ankle-power source. The models adjust for categorical walking speed,
side, and the cohort-by-speed interaction while clustering repeated trials by
subject. Outputs include CSV tables, a formatted Excel workbook, adjusted-mean
and cohort-contrast figures, diagnostics, and a JSON audit summary. Walking
speed and raw OpenSim power are resolved from the dataset path in
`run_config.json`; use `--dataset_root PATH` to override it or
`--skip_mixed_effects` to disable this step.
