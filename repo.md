# repo.md — LLM Quick Reference for Calc_ID_MJX

> Purpose: single-file map of this repo so an LLM can orient without scanning everything.
> Many source files are huge (train.py ~5.8k lines, infer.py ~7.8k, loso_from_checkpoint.py
> ~4.9k, data_loader.py ~2k). Use the line-number hints below to jump instead of reading whole files.
> Repo root: `Calc_ID_MJX/`. CHPC agent guide: `CHPC_REFERENCE_FOR_AGENTS.txt`.
> (`CHPCRepo/` was removed in Stage 2 - it lives in its own repo, AlexAEdwards/CHPC_MJX.)

---

## 1. What this project does (one paragraph)

Predicts **ground reaction dynamics** (Center of Pressure, Ground Reaction Forces, free Moments,
and foot-ground Contact) from **gait kinematics** using a Transformer, then verifies the
prediction is physically consistent via **MuJoCo/MJX Jacobian-based Inverse Dynamics (ID)**.
The model is trained so that its predicted external forces reproduce the correct joint torques
(τ = Jᵀ·F). Ground truth comes from OpenSim ID (patella-stripped) and/or MJX ID. Everything is
physiologically normalized (COP by height `h`, GRF/Moments by mass `m`, with gravity 9.8067).

---

## 2. Directory layout (top level)

```
TransformerFinal/     <-- CORE. Model, training, inference, LOSO, HPO. (see §3)
scripts/              Data-prep + OpenSim/MJX ID utilities, validation, mass estimation (see §6)
ProcessData.py        Master preprocessing pipeline: raw Motion -> Trial/ProcessedData/*.npy (see §5)
Loso_Combined.py      3-stage LOSO over OpenCapSubjects (refine-q -> main -> eval)
ProcessAddbiomechnics/updateModel.py
                      Fix MuJoCo XML masses/inertias/armatures; knee-coupling
                      canonicalisation. Imported by ProcessData.py.
configs/              Config YAMLs
Datasets_Local/       Local datasets;  Datasets_NAS -> symlink to NAS shared datasets
TrustedDataSetNoised12Distributed_*   Primary training datasets (variants: EdgeHold, GRF_Filt,
                      OYIncluded, AllPatients_EstimatedWeights, UnFiltered). (see §5 for structure)
OpenCapSubjects_Filt/, OlderYoungerAdultDataset_*   Additional/eval cohorts
outputs/              Training runs, HPO sweeps, checkpoints (best_model.pkl)
inference_results/    Inference + LOSO outputs, metrics, plots
artifacts/CHPC_HPO_results/, HPOAnalysis/   CHPC sweep outputs
                      (CHPC code itself: separate repo AlexAEdwards/CHPC_MJX)
figures/, RMASBFigures/, AnklePowerAnalysis/, AccuracyByGender&Speed/   Analysis + paper figures
GeometryWithMus/, myoconverter/   Musculoskeletal model geometry / OpenSim->MJX conversion
BadTrialsFromTrustedDataset/      Quarantined bad trials
```

---

## 3. TransformerFinal/ — the core (key files)

### Model + training
- **train.py** — Main training script ("V5 full physics pipeline"). Defines the model, losses,
  train/eval steps, normalization, plotting, and `main()` (CLI). Line hints:
  - `class KinematicsToCOPGRFMoments` (~1618): the Transformer. Inputs = temporal feature seq
    (`input_dim=54` default, auto-inferred) + static token (`static_dim=8`:
    [height, mass, gender, PatientSize(4), forwardVel]). Prepends static token, N `TransformerBlock`s,
    outputs the **standard 14** channels.
  - Output layout constants (~1016): `STANDARD_OUTPUT_DIM=14`,
    `COP_SLICE=0:4` [Rx,Rz,Lx,Lz], `GRF_SLICE=4:10` [Rx,Ry,Rz,Lx,Ly,Lz],
    `MOMENTS_SLICE=10:12` [Rz,Lz], `CONTACT_SLICE=12:14` [right,left] (sigmoid).
    NOTE: contact is an OUTPUT now, not an input.
  - `class Normalizer` (~753), `compute_normalizers_from_loader` (~774), `normalize_batch` (~892):
    Z-score stats (applied after physiological scaling).
  - `compute_total_loss` (~1808): the big weighted loss — COP/GRF/Moments/Contact + physics
    (`torque`, `qfrc_inverse`, `jacobian`, `rotation`, `grf_correction`) + per-DOF joint weights.
  - `compute_tau_grf_from_predictions` (~1749), `compute_full_external_moments` (~1695): physics τ=Jᵀ·F.
  - `make_train_step` (~2324) / `make_eval_step` (~2397): jitted steps.
  - `split_model_predictions` (~1498): slice raw output into channel dicts.
  - `plot_predictions` (~2420), `plot_validation_outlier_grid`, `plot_loss_history`: diagnostics.
  - `main` (~3270): argparse CLI (see §4 for flags).
- **train_single_model.py** — Thin launcher: hardcodes a `CONFIG` dict (best-HPO hyperparams:
  d_model=384, num_layers=4, window_size=110, stride=16, prediction_margin_frames=20,
  lr~1.9e-4, dropout~0.16, ff_dim=1536) + dataset path + exclude lists, then calls train.py.
  Start here to see the canonical training recipe.
- **HPOsearch.py** — WandB-free local hyperparameter search. Reads a sweep YAML
  (e.g. `HPO_local_bayes_100_equal_knee.yaml`), runs train.py jobs sequentially, ranks,
  writes `condensed_results/`. Holds the authoritative `EXCLUDE_*` trial/subject lists.

### Inference
- **infer.py** — Inference + physics-consistency evaluation + publication plots. Line hints:
  - `main` (~7359), `resolve_checkpoint_path` (~7262).
  - `run_inference` (~5232): core loop — load trial via data_loader, window it,
    predict, unnormalize to physical units (N, Nm), compare to GT.
  - `_predict_with_train_style_windows` (~1293), `_build_inference_window_starts` (~1274).
  - `_convert_output_to_physical_predictions` (~3591): undo normalization.
  - `_build_jacobian_comparison_stats` (~1598), rotation/Jacobian comparison plots (~1825+).
  - `make_publication_plots` (~6532).
  - CLI flags: `--checkpoint --data_dir --trial_name --test_json --all_val --output`
    `--window_size --stride --d_model --num_layers --ff_dim --no_plots --lightweight
    --make_graph --RestrictMaxVals --OpenCapValDataset --OpenCapDataset`.

### Data
- **data_loader.py** — Memory-efficient on-demand trial loader. Line hints:
  - `load_single_trial` (~531): reads a trial's `ProcessedData/*.npy`, builds temporal feature
    blocks + targets; handles `_noised`, `_OSfilt` variants, GRF-norm COP target, MuJoCo/OpenSim
    6-col conversion. Returns dict of arrays (pos/vel/acc, com_r/com_l/com_accel, jacp/jacr,
    pelvis_rot, ankle_heights, angles, qfrc_inverse, COP/GRF/Moments/contact targets, metadata).
  - `class TrialDataLoader` (~1467): iterates windows into batches (`__iter__` ~1892,
    `_extract_windows_from_trial` ~1745, `_collate_batch` ~1975). Constructs the 54-dim feature vector.
  - Feature block order (see `infer_input_feature_layout_from_loader` in train.py ~562):
    pelvis_rot, pos, vel, com_r, com_l, com_accel, [ankle_heights], [jacobian_input],
    foot_progression_angle, calcn_to_floor_angle, [qfrc_inverse_input], [rot_w_to_ga_input_flat].
  - `select_pos_input_columns` (~103): pos feature selection (optional pelvis Euler).
  - `compute_bodyweight_height_norm_factor` / `(un)normalize_qfrc_inverse_by_bw_height` (~42-81):
    physiological BW·height scaling for torques.
  - `compute_balance_weights` (~201): gender/speed dataset balancing.
  - `OS_FILTER_FILES` (~19): files with `_OSfilt` variants (vel/acc/qvel/qacc/qfrc_inverse/ID_GT).

### LOSO (Leave-One-Subject-Out) fine-tuning
- **loso_from_checkpoint.py** — Full LOSO from a pretrained checkpoint. `LOSO_CONFIG` dict (~62)
  is the control panel: checkpoint path, output_dir, epochs/lr/batch, loss weights (defaults
  emphasize `torque_weight`), `use_OpenSimID_GT`, and **nested inner-epoch selection**
  (`inner_epoch_selection`: each outer fold runs an inner LOSO to pick the best epoch by
  `inner_selection_objective`, e.g. left-stance moment MAE %BWH, with std penalty).
  Key defs: `main` (~4360), `_run_fold` (~3892), `_build_loso_folds` (~803),
  `_run_train_epoch` (~3030), `_run_infer_style_evaluation` (~2832),
  `_run_inner_epoch_validation_fold` (~3488), `_run_original_checkpoint_reference_outputs` (~2169).
  Uses `loso_adapters.py`. CLI mirrors LOSO_CONFIG keys (see §4).
- **loso_adapters.py** — Reuses train.py's exact architecture to load a checkpoint and fine-tune
  a few epochs (no extra adapter layers). Imports model/loss helpers from train.py.
- **loso_from_checkpoint_HPO.py** — Nested LOSO HPO, metric-only (skips infer.py plotting) to
  tune LR + epoch count fast. Uses `loso_from_checkpoint` as `base`. Grid files:
  `loso_nested_hpo_*.json`.

### Secondary "refine-q" / "mod-q" model family (kinematics denoising)
Separate, smaller model that predicts cleaned joint angles from noisy kinematics
(`q_prime = pos_noised + delta_q`). Feeds cleaner inputs into the main model.
- **train_refine_q.py** / **train_single_refine_q.py** — train the Q-residual refiner.
- **infer_mod_q.py** — RETIRED (Stage 6): could not import from either of its two
  locations; `infer.py` is now a standalone entry point with no library consumers.
  Formerly: inference for the mod_q checkpoint family (fixed schema:
  COP4+GRF6+GRM2+Contact2 + clean pos16+vel19+acc19).
- **NoiseModels/** — corruption/noise modeling used to generate `_noised` data:
  - `corruption_model/` package (config, io, models, preprocess, residuals, evaluation, scripts).
  - `train_mod_q.py`, `infer_refine_q.py`, `mod_q_shared.py`,
    `refine_q_shared.py`, `runtime_model_utils.py`, `make_patch.py`.

### Misc in TransformerFinal/
- **create_trimmed_raw_grf.py** — build trimmed unfiltered GRF arrays aligned to ProcessedData frames.
- Config/grid files: `HPO_local_bayes_100_equal_knee.yaml`,
  `HPO_shared_val_subjects_subj_oa_y.json`, `loso_nested_hpo_*.json`.

---

## 4. Important CLI flags (train.py / loso)

**train.py**: `--data_dir --output_dir --epochs --batch_size --learning_rate --dropout_rate
--weight_decay --window_size --stride --min_trial_length --d_model --num_layers`
Loss weights: `--cop_weight --grf_weight --moments_weight --contact_weight --torque_weight
--qfrc_inverse_weight --rotation_weight --jacobian_weight --grf_correction_weight`
Per-DOF weights: `--hip_add_{r,l}_weight --knee_{r,l}_weight --ankle_{r,l}_weight
--subtalar_{r,l}_weight --lumbar_{extension,bending,rotation}_weight`
Toggles: `--magOnOff --contactOnOff --use_contact_weighting --trim_cop --UseNoised --NoisedGT
--includeJacobianInput --UseGRFNormCOP --UseOSFiltering --cop_mask --refresh_cache
--BestModelByTorque --BestModel_TorqueWeighting`
WandB: `--use_wandb --wandb_project --wandb_entity --wandb_group --exp_name` etc.

**loso_from_checkpoint.py**: `--checkpoint --output_dir --epochs --learning_rate --batch_size
--weight_decay --seed` + all loss weights above + `--use_OpenSimID_GT --UseGRFNormCOP
--torque_grad_through_jacob --*_input_reg_weight --rotation_residual_max_deg`.
(Editing `LOSO_CONFIG` in-file is the usual entry point; CLI overrides it.)

---

## 5. Dataset structure

There are TWO dataset families with slightly different layouts. Both share the same
`<Dataset>/<Subject>/Trial_<n>/` nesting and the same per-subject model files, but the
Trusted datasets are the **training** corpus (noised, single ProcessedData GT) while
OpenCapSubjects is the **LOSO/held-out evaluation** corpus (real MoCap GT, extra timebase folders).

### 5a. TrustedDataset family (TRAINING) — `TrustedDataSetNoised12Distributed_*`
Variants: `_EdgeHold`, `_EdgeHold_GRF_Filt`, `_EdgeHold_OYIncluded`,
`_AllPatients_EstimatedWeights`, `UnFiltered`. Data loader / train.py point here.

```
<TrustedDataset>/<Subject>/     Subject = cohort-prefixed: Y5, OA19, SUBJ44 (=Stroke),
  Patient_MD.json                 GaitRetraining_Subject123, S_GAH_8, 02, ...  (prefix = cohort)
  PatientSize.npy               4-dim body-size vector (part of static token)
  Mass_kg.npy / Height_m.npy    (also mirrored inside ProcessedData)
  MyosuiteModel*.xml            MuJoCo/MJX model;  OpenSimModel.osim, Rajagopal_scaled_*.osim
  Geometry/                     bone STLs
  Trial_<n>/
    Motion/                     raw kinematics: Pos, Accel, COP, GRF, GRM, Time (+ _noised)
    ProcessedData/              <-- what data_loader reads. ~hundreds of .npy files, incl:
       pos_inputs / vel_inputs / acc_inputs .npy   (+ _noised, +_OSfilt)
       pos_mjx / qvel_mjx / qacc_mjx .npy          (MJX joint state)
       COM_r, COM_l, COM_Acc_Global, ankle_heights, knee/ankle_pos_*
       COP_CalcFrame_GroundAligned[_GRFNorm/_BackToWorld/_YplusAnkleHeight].npy  (COP target variants)
       GRF_Cleaned, GRF_NoFilt_Trimmed, Moment_Cleaned, contactBoolean.npy       (force/contact targets)
       Jacobian.npy (jacp/jacr), pelvis_rot_matrix.npy, rotation matrices
       qfrc_inverse.npy, ID_GT_MJX.npy  (torque targets; +_OSfilt variants)
       Height_m.npy, Mass_kg.npy, forwardVel.npy, Foot_ProgressionAngle, CalcnToFloor_AngleDeg
    OpenSimResults/             OpenSim ID GT: inverse_dynamics.sto, coordinates.mot,
                                ground_reaction.mot, external_loads.xml, id_setup.xml,
                                OpenSimModel_NoPatel.osim, AccuracyMetrics.json
```

### 5b. OpenCapSubjects family (LOSO / EVALUATION)
Dirs: `OpenCapSubjects_Filt/` (preferred), `OpenCapSubjects_NoTrim_NoFilt/`, or NAS
`Datasets_NAS/AddBiomechanicsDataset_All_npy/OpenCapSubjects`. Resolution order lives in
`loso_from_checkpoint.py::_resolve_loso_data_dir` (~769). OpenCap trials are real MoCap, so
infer.py forces `UseNoised=False` for them (`--OpenCapValDataset`).

```
OpenCapSubjects_Filt/
  opensim_id_batch_manifest.json          } dataset-level manifests / processing params
  opensim_id_recalculated_manifest.json   }
  processing_parameters.json
  OpenSimToMJX_Accuracy_Controlled/        cohort-wide MJX-vs-OpenSim ID accuracy plots+JSON
  OpenSimVsMJX_ID_Summary/
  subject2/  subject3/ ... subject11/       Subject = "subject<N>" (numbered, NOT cohort-prefixed)
    Patient_MD.json, PatientSize.npy, OpenSimModel.osim, Geometry/
    MyosuiteModel.xml / _FIXED / _MOCAP_FIXED / _Runtime(.xml/.meta.json)  (extra XML variants)
    OpenSimVsMJX_ID/                        per-subject OS-vs-MJX torque validation plots+JSON
    Trial_<n>/
      walking<n>.mot / walking<n>.sto       OpenCap source kinematics/forces
      Motion/                               (same role as Trusted Motion/)
      ProcessedData/                        <-- data_loader reads this (same schema as Trusted)
      TrainingData/                         condensed per-trial training arrays (acc_inputs,
                                            anklePos, COM_{l,r}{,_vel,_acc}, contactBoolean, ...)
      MoCap/ , MoCap_RawTimebase/           MoCap-derived GT kinematics (Accel_Filtered, etc.)
      OpenSimResults/ , OpenSimResults_recalculated/   OpenSim ID GT (+ recalculated variant)
      SecondaryProcessing/
      Torque_Comparison_Trial_<n>.png
```

### Variant/suffix conventions (both families)
- `_noised` = noise-corrupted inputs (training realism; Trusted only in practice).
- `_OSfilt` = OpenSim-filtered velocity/accel derivation (see `OS_FILTER_FILES` in data_loader.py).
- `_PreCOPClean` = COP arrays before the COP-cleanup step.
- Trusted cohort = subject-folder prefix (numeric / OA / Y / S / S_GAH / GaitRetraining / SUBJ=Stroke);
  OpenCap subjects are just numbered `subject<N>`.

---

## 6. scripts/ — data prep & ID/validation utilities

- **generate_opensim_id_inputs.py** — build OpenSim ID setup; **strips patella**
  (`create_patella_free_model`) to avoid ~10,000 N·m artifact moments (patellofemoral
  CoordinateCouplerConstraint). Key fix, see auto-memory.
- **batch_opensim_inverse_dynamics.py** — run OpenSim ID in batch.
- **recalculate_opensim_id_mocap_kin_processed_forces.py** — recompute OpenSim ID GT.
- **validate_opensim_vs_mjx.py** — proves MJX ID ≈ OpenSim ID (MAE ~0.06% BW×H, r~0.997);
  writes AccuracyMetrics.json + OpenSimToMJX_Accuracy/ plots.
- **estimate_mass_from_grf.py** — estimate subject mass from vertical GRF/CoM vs Patient_MD.
- **rescale_models_to_estimated_mass.py**, **write_estimated_mass_to_patient_md.py** — apply it.
- **compare_opensim_mjx_id_opencap.py**, **diagnose_opensim_id_forces.py** — diagnostics.
- **extract_{stroke,pd_boari}_c3d_forces.py**, **create_stroke_cleaned_dataset.py** — cohort import.
- **npy_from_force_mot.py**, **prescribed_accel_id.py**, **attribute_kinematic_torque_error.py**,
  **verify_subject5_loso_ready.py**, **ankle_stance_common.py**.

---

## 7. Typical workflows

1. **Preprocess**: `ProcessData.py` (raw Motion -> ProcessedData/*.npy) + scripts/ for OpenSim/MJX ID GT.
2. **Train**: edit + run `TransformerFinal/train_single_model.py` (wraps `train.py`).
   -> checkpoint `best_model.pkl` under `outputs/.../runs/<run>/`.
3. **HPO**: `TransformerFinal/HPOsearch.py` with a sweep YAML -> `outputs/local_hpo_*`.
4. **Infer / evaluate**: `TransformerFinal/infer.py --checkpoint ... --all_val` -> `inference_results/`.
5. **LOSO generalization**: edit `LOSO_CONFIG` in `loso_from_checkpoint.py` (point at checkpoint) & run;
   nested epoch/LR HPO via `loso_from_checkpoint_HPO.py`.

---

## 8. Key conventions / gotchas

- **Channel order is fixed**: right-before-left; COP=[Rx,Rz,Lx,Lz], GRF=[Rx,Ry,Rz,Lx,Ly,Lz],
  Moments=[Rz,Lz], Contact=[R,L]. Y is vertical.
- **Two normalization stages**: physiological (COP/h, GRF/m, Moments/m, gravity 9.8067) then
  statistical Z-score (`Normalizer`). Physics loss unnormalizes back to N/Nm for τ=Jᵀ·F.
- **Contact is predicted** (sigmoid), then hard-thresholded to mask COP/GRF in physical space.
- **prediction_margin_frames**: only frames [margin : window-margin) are supervised/evaluated.
- **GT source flags**: OpenSim ID vs MJX ID vs recalculated; `_OSfilt` vs default kinematics filtering.
- Big files: prefer grep + the line hints above over full reads. `__pycache__/`, `.jax_compilation_cache/`
  are generated. (`Z...CodeIamNotReadyToDelete/` removed in Stage 2; its source was
  rescued to `scripts/legacy_forward_sim/` and `scripts/legacy_scott_data/`.)
