# Plan — Cohort-Level Trunk-Sway KAM Aggregation + New-Model Compatibility

> **IMPLEMENTATION STATUS (code complete):**
> - ✅ Part 0 dataset verification (all files present, shapes match, nv=23, input 370).
> - ✅ Part 2.1/2.2 verified (no MuJoCo model load on eval path → qpos concern N/A; rotation present).
> - ✅ Part 2.4 KAM-file fallback + warning implemented in `TransformerFinal/infer.py`
>   (`analyze_stance_phase_torques`): source-specific variant → falls back to plain
>   `KneeToCOP_Vectors.npy`, warns if none found.
> - ✅ Part 1 cohort aggregation: `_aggregate_trunk_sway_kam_across_folds` +
>   `_plot_trunk_sway_kam_cohort` added to `TransformerFinal/loso_from_checkpoint.py`, wired into `main()`
>   (guarded by `evaluate_on_ts`), written to `loso_summary.json` +
>   `trunk_sway_cohort_summary/`. Unit-validated on synthetic folds (Pearson r, MAE, curve
>   RMSE, per-condition MAE, plots, degenerate cases).
> - ⏳ **Remaining (runtime, needs a free GPU + chosen checkpoint):** Part 2.3 FiLM/non-FiLM
>   smoke LOSO with `--evaluateOnTS`; Part 2.5 full 10-subject acceptance run.


Goal: after a LOSO run over `OpenCapWalkingTrunkSwaySubjects/` (each subject held out once,
`--evaluateOnTS`), produce a **cohort-level** comparison of the Knee Adduction Moment (KAM)
between trunk-sway (TS) and normal walking, for Ground Truth vs the model's prediction — not
just the per-subject summaries that already exist. Also close the compatibility items (gap #4)
for running this with the new `AblatV_JM` checkpoints.

Files: `TransformerFinal/loso_from_checkpoint.py` (all edits), reading the per-fold artifacts it already writes.

---

## Part 0 — Dataset verification (DONE; recorded here as the plan's baseline)

Checked `OpenCapWalkingTrunkSwaySubjects/` against a training (Trusted) trial. Every file the
loader/model needs is present under each trial's `MoCap/ProcessedData/` **and**
`Video/ProcessedData/`, with shapes identical to training (nv = 23):

| File | Trusted (train) | TS subject10 (MoCap/Video) | Status |
|---|---|---|---|
| pos_inputs.npy | (T, 18) | (T, 18) | ✅ (knees included; pos block selects 15) |
| vel_inputs.npy | (T, 21) | (T, 21) | ✅ (knees included) |
| acc_inputs.npy | (T, 21) | (T, 21) | ✅ (knees included) |
| qfrc_inverse.npy | (T, 23) | (T, 23) | ✅ nv=23 |
| qfrc_grf_contribution.npy | (T, 23) | (T, 23) | ✅ |
| WorldToGroundAlignedCalcnRotation.npy | (T, 2, 3, 3) | (T, 2, 3, 3) | ✅ (→ 18 flat; this is the "rotation matrix") |
| Jacobian.npy | dict jacp/jacr (T,2,3,23) | dict jacp/jacr (T,2,3,23) | ✅ (→ 276 flat) |
| KneeToCOP_Vectors.npy | (T, 6) | (T, 6) | ✅ (KAM input) |
| pelvis_rot_matrix.npy | (T, 6) | (T, 6) | ✅ |
| COM_r/COM_l/ankle_heights | (T,3)/(T,3)/(T,2) | same | ✅ |

Counts: 120 files each for Jacobian / qfrc_inverse / KneeToCOP / pos_inputs (= 10 subjects × 3
trials × 2 conditions × 2 sources), i.e. 60 normal + 60 TS trial-condition dirs, all populated.

Derived input dim = 6+13+19+3+3+3+2+276+2+2+23+18 = **370**, matching the `AblatV_JM` models.
The earlier "rot_w_to_ga_matrix.npy: 0 files" was a wrong filename (the real file is
`WorldToGroundAlignedCalcnRotation.npy`, which exists) — **no data gap**.

Conclusion: no preprocessing is required; the dataset is ready.

---

## Part 1 — Cohort-level KAM aggregation (new implementation)

### What each fold already writes (input to the aggregation)
`<fold_dir>/trunk_sway_effect_eval/kam_trunk_sway_effect_summary.json` and the same dict is
stored in `fold_result["trunk_sway_effect_eval"]`. Its `sources` block has, per source
(`"Ground Truth"`, `"LOSO Fine-Tuned"`, `"Original OpenCap"`):
`normal_curve[101]`, `trunk_sway_curve[101]`, `effect_curve[101]`, `normal_mean`,
`trunk_sway_mean`, `mean_effect`, `peak_abs_effect`, `cohens_d`, `normal_n_stances`,
`trunk_sway_n_stances`. Units: %BW·h. (See `_build_trunk_sway_effect_summary`,
`_summarize_kam_condition_effect` ~line 3063/3210.)

### Step 1.1 — Add `_aggregate_trunk_sway_kam_across_folds(fold_results, output_root)`
Place it near the other aggregation helpers (e.g. after `_build_trunk_sway_effect_summary`).

1. Collect usable fold payloads: `p = fr.get("trunk_sway_effect_eval")` where `p` exists,
   `p.get("enabled")` is True, `not p.get("skipped")`, and `p.get("sources")` is non-empty.
   Record `held_out_subject`. If fewer than 2 usable subjects, still emit a summary but mark
   `"n_subjects" < 2` (std/correlation become nan) and return early after saving.
2. For each `source` in `("Ground Truth", "LOSO Fine-Tuned", "Original OpenCap")`:
   - Gather per-subject scalars into lists: `normal_mean`, `trunk_sway_mean`, `mean_effect`,
     `peak_abs_effect`, `cohens_d`.
   - Gather per-subject curves (each length-101) into arrays: `normal_curve`,
     `trunk_sway_curve`, `effect_curve`. Use `np.vstack`; guard curves that are None/wrong
     length by skipping that subject for that source.
   - Compute cohort stats: for each scalar, `mean` and `std` (ddof=1 when n≥2 else 0/nan) via
     `np.nanmean`/`np.nanstd`. For each curve, `cohort_mean_curve = np.nanmean(stack, axis=0)`
     and `cohort_std_curve = np.nanstd(stack, axis=0)` (both length-101, stored as lists).
   - Store under `sources[source] = {"n_subjects", scalar means/stds,
     "normal_curve_mean/std", "trunk_sway_curve_mean/std", "effect_curve_mean/std"}`.
3. Compute **model-vs-GT agreement** (the headline: does the model reproduce the TS effect?):
   - Build paired per-subject arrays for GT and LOSO of: `mean_effect`, `normal_mean`,
     `trunk_sway_mean` (align by held_out_subject; only subjects present in both).
   - Report, in an `"agreement_LOSO_vs_GT"` block:
     - `mean_effect_pearson_r` (correlation of per-subject TS effects, LOSO vs GT),
     - `mean_effect_mae` and `mean_effect_bias` (mean signed LOSO−GT),
     - same three for `normal_mean` and `trunk_sway_mean` (absolute KAM level, not just effect),
     - `effect_curve_rmse` = mean over subjects of RMSE(LOSO effect_curve − GT effect_curve),
     - `n_paired_subjects`.
   - Also a **KAM prediction-accuracy** block `"kam_pred_accuracy_vs_GT"`: per condition
     (`normal`, `trunk_sway`), cohort mean of the per-subject stance-curve MAE
     `mean_t |LOSO_curve − GT_curve|` (i.e. how well the model predicts absolute KAM, split by
     condition). This directly answers "normal predicting metrics" + "TS vs GT" for KAM.
4. Assemble payload: `{"kam_dof", "units": "percent_BW_times_height", "n_subjects",
   "subjects": [...], "sources": {...}, "agreement_LOSO_vs_GT": {...},
   "kam_pred_accuracy_vs_GT": {...}}`. Save to
   `output_root / "trunk_sway_cohort_summary" / "cohort_kam_trunk_sway_summary.json"` via
   `_save_json`. Return the payload.

### Step 1.2 — Cohort plots `_plot_trunk_sway_kam_cohort(cohort_payload, output_dir)`
Guard `matplotlib` import (mirror `_plot_trunk_sway_kam_effects` ~line 3091). Produce:
- **Fig A — KAM by condition (cohort mean ± std)**: x = 0–100 % stance. For GT and LOSO (and
  OpenCap if present), plot `normal_curve_mean` (solid) and `trunk_sway_curve_mean` (dashed)
  with ±std shaded. Shows whether the model's TS-vs-normal shift tracks GT's. Save
  `cohort_kam_by_condition.png`.
- **Fig B — TS effect agreement scatter**: per-subject `mean_effect` GT (x) vs LOSO (y), with
  identity line and annotated Pearson r + MAE. Save `cohort_kam_effect_agreement.png`.
- **Fig C — cohort effect bar**: `mean_effect` mean ± std per source. Save
  `cohort_kam_effect_bars.png`.
- (Optional) a plotly `.html` twin of Fig A to match the existing per-fold style
  (`_plot_trunk_sway_kam_effects` already uses plotly).

### Step 1.3 — Wire into `main()`
After the fold loop where `metric_means, metric_stds = _aggregate_metric_dicts(...)` is computed
(~line 5237), add:
```python
if evaluate_on_ts:
    cohort_ts = _aggregate_trunk_sway_kam_across_folds(fold_results, output_root)
    if cohort_ts is not None:
        summary_payload["trunk_sway_cohort_summary"] = cohort_ts
        print(f"🦵 Cohort KAM TS-vs-normal summary written for {cohort_ts.get('n_subjects', 0)} subjects", flush=True)
```
Ensure `summary_payload` already exists at that point (it's built ~line 5247); if the insert is
before it, instead stash `cohort_ts` in a local and add the key when `summary_payload` is built.
Keep it inside `if evaluate_on_ts:` so non-TS runs are unchanged.

### Step 1.4 — Validation (no full run needed)
- Unit-test the aggregation on the **existing** per-fold JSONs (or 3 synthetic payloads with
  known curves/means): assert cohort mean_effect == mean of per-subject mean_effects, std uses
  ddof=1, the GT-vs-LOSO Pearson r on a planted linear relation ≈ expected, and curve RMSE is 0
  when LOSO==GT. Confirm it degrades gracefully (nan, no crash) with 1 subject or a `skipped`
  fold.
- `python -c "import ast; ast.parse(open('TransformerFinal/loso_from_checkpoint.py').read())"` after edits.

---

## Part 2 — New-model compatibility (gap #4), as concrete checks/fixes

### 2.1 Feature dimensions + qpos concern — RESOLVED (verified). No action.
All ProcessedData files are nv=23 → input_dim 370, matching `AblatV_JM`. The
"31-vs-51-DOF qpos" concern from earlier notes was about the myoconverter model XML used when
*recomputing* MJX ID. **Confirmed**: `grep -nE "MjModel|from_xml|mujoco\.|mjx\.|load_model"
TransformerFinal/loso_from_checkpoint.py` returns nothing on the eval path — LOSO reads the **preprocessed**
ProcessedData (jacp/jacr, qfrc, rotation all nv=23) and never re-runs MJX ID or loads the model
XML. Torque uses `select_torque_jacobians` (TransformerFinal/train.py:1665) → preprocessed jacp/jacr. The concern
does not apply.

### 2.2 Rotation matrix — RESOLVED. `WorldToGroundAlignedCalcnRotation.npy` (T,2,3,3) present in
both conditions/sources. No action.

### 2.3 FiLM checkpoint on the LOSO path — needs a smoke run.
`build_loso_model`/`loso_adapters` already auto-detect `film_mlp` in the checkpoint tree (added
earlier), so a FiLM `AblatV_JM` checkpoint should reconstruct with `use_film=True`. Validate:
1. Pick a **finished FiLM run**, e.g. `outputs/AblatV_JM_A0B0C0D0E1/best_model.pkl` (E1 = FiLM on).
2. Run a **minimal LOSO**: 1 held-out subject, 1–2 epochs, `--evaluateOnTS`, plots off:
   ```
   python TransformerFinal/loso_from_checkpoint.py \
     --checkpoint outputs/AblatV_JM_A0B0C0D0E1/best_model.pkl \
     --data_dir OpenCapWalkingTrunkSwaySubjects \
     --evaluateOnTS \
     --epochs 1 --folds subject10   # or the config's single-subject knob
     --output_dir inference_results/LOSO_TS_smoke
   ```
   (Use whatever single-fold / subject-subset control `LOSO_CONFIG` exposes; if none, add a
   `--only_subjects subject10` guard, or temporarily set the fold list.)
3. Confirm in the log: model reconstructs (input_dim 370), `film_mlp` detected, no shape/param
   errors, `trunk_sway_effect_eval` produced, and `cohort_kam_trunk_sway_summary.json` written.
4. Also smoke a **non-FiLM** checkpoint (e.g. `AblatV_JM_A0B0C0D1E0`) to confirm both load.

### 2.4 KAM vector filename resolution — CONFIRMED BUG, needs a fix.
The infer KAM logic (TransformerFinal/infer.py ~6739) resolves the vector file as:
`input_source=="mocap"` → `KneeToCOP_Vectors_Mocap.npy`; `use_noised` →
`KneeToCOP_Vectors_noised.npy`; else → `KneeToCOP_Vectors.npy`. **Verified on the TS trials:
only `KneeToCOP_Vectors.npy` exists** — the `_Mocap` and `_noised` variants are absent (0 files).
And KAM is guarded by `if kam_path.exists():` (TransformerFinal/infer.py ~6751), so if a run uses
`input_source=="mocap"` (as MoCap-GT evaluation naturally would), the vector path is missing and
**KAM is silently skipped for every trial** — the entire TS-vs-normal-vs-GT comparison would be
empty even though the code "succeeds."

Fix (do this before the smoke run):
1. In the KAM resolver, add a **fallback**: if the source-specific file
   (`_Mocap`/`_noised`) does not exist, fall back to the plain `KneeToCOP_Vectors.npy` before
   giving up. (The plain file is present for both MoCap and Video ProcessedData here.)
2. When the vector file is still missing after fallback, **print a warning** (not a silent
   skip) naming the trial and the paths tried, so a misconfiguration is visible.
3. Confirm which `input_source` the LOSO run uses for OpenCap (see
   `_evaluate_single_trial_infer_style`); ensure inputs come from `Video/ProcessedData` and GT
   from mocap/force, and that the resolved KAM file exists after the fallback. Add the same
   fallback anywhere loso reconstructs the KAM path independently of infer.

### 2.5 End-to-end acceptance (after 2.3 passes)
Run the full LOSO over all 10 subjects with `--evaluateOnTS` and the chosen best `AblatV_JM`
checkpoint; confirm `loso_summary.json` contains `trunk_sway_cohort_summary`, the three cohort
plots exist, and the normal-trial standard metrics are aggregated as before.

---

## Execution order
1. Part 0 (done) — dataset confirmed ready.
2. Part 2.1/2.2 verification greps (cheap; confirm no model-xml load, rotation present).
3. Part 1.1–1.3 implement the aggregation + plots + main() wiring.
4. Part 1.4 unit-validate the aggregation math.
5. Part 2.3 FiLM + non-FiLM smoke LOSO (1 subject, 1 epoch, `--evaluateOnTS`).
6. Part 2.4 input-source/KAM-file check.
7. Part 2.5 full 10-subject acceptance run.

## Deliverables
- `_aggregate_trunk_sway_kam_across_folds` + `_plot_trunk_sway_kam_cohort` in
  `TransformerFinal/loso_from_checkpoint.py`, wired into `main()`.
- `<output_root>/trunk_sway_cohort_summary/cohort_kam_trunk_sway_summary.json` with per-source
  cohort curves/scalars, model-vs-GT agreement (Pearson r, MAE, bias, curve RMSE), and per-
  condition KAM prediction accuracy vs GT.
- `cohort_kam_by_condition.png`, `cohort_kam_effect_agreement.png`, `cohort_kam_effect_bars.png`.
- Confirmation that FiLM and non-FiLM `AblatV_JM` checkpoints run the LOSO/TS path cleanly.
