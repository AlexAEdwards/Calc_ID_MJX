# TransformerFinal — Model Issues & Improvement Plans

> Scope: the training/inference model that predicts COP/GRF/free-moments/contact from gait
> kinematics and enforces physics via τ = Jᵀ·F. Files reviewed: `train.py`, `data_loader.py`,
> and the LOSO/infer entry points. This document (1) summarizes what the model does, (2) ranks
> the high-level issues found, and (3) gives step-by-step remediation plans that a Sonnet-level
> model can execute one at a time.

---

## 0. What the model is (as-implemented)

- **Architecture** (`KinematicsToCOPGRFMoments`, train.py ~1642): a plain pre-LN Transformer
  encoder. A temporal feature sequence (`input_dim≈54`) is projected to `d_model`, gets a
  **sinusoidal** positional embedding, and a single **static token**
  (`[height, mass, gender, PatientSize(4), forwardVel]`) is prepended. `num_layers` blocks of
  MHSA + GELU-MLP, then the static token is dropped and a `Dense(14)` head emits
  COP(4) + GRF(6) + Moments(2) + Contact(2, sigmoid).
- **Two-stage normalization**: physiological (COP/height, GRF/(m·g), Moments/(m·g·h)) applied in
  the data loader, then a Z-score `Normalizer` fit on ~100 train batches.
- **Loss** (`compute_total_loss`, ~1832): weighted sum of `cop + grf + moments + contact + torque
  + grf_correction`. The torque term reconstructs τ_grf via `compute_full_external_moments` →
  `compute_tau_grf_from_predictions` and compares against `qfrc_grf_contribution`, per-DOF/magnitude/
  contact weighted.
- **Contact masking**: during training, COP/GRF are hard-masked by **ground-truth** `contactBoolean`
  (`output_mask_r/l = contact > 0.5`); contact itself is trained with BCE.
- **Optimizer**: `optax.adamw` with `clip_by_global_norm(1.0)` at a **constant** learning rate.
- **Best-model selection**: by `val total_loss`, or by a torque RMSE score when `--BestModelByTorque`.

---

## 1. High-level issues (ranked by impact / effort)

### A. Confirmed correctness / efficiency problems

1. **Two gradient-based "effect losses" run every step but are never optimized.** `compute_total_loss`
   computes `torque_cop_effect_loss` and `torque_grf_effect_loss` via **four `jax.grad` calls**
   (train.py ~2252–2272). They are returned in the metrics dict and only *printed/logged* — they are
   **not** part of `total_loss` (~2304). Because they are jitted and consumed downstream, JAX cannot
   eliminate them, so every **training and evaluation** step pays ~4 extra backward passes purely for a
   diagnostic. This is likely the single largest avoidable cost in the training loop.
   *Impact: large wall-clock/compute cost; no accuracy benefit. Effort: low.*

2. **No learning-rate schedule or warmup.** `create_train_state` (~2337) builds `optax.adamw` at a
   fixed LR; the `epoch` passed into `train_step` only feeds a now-dead `output_reg_multiplier`.
   Transformers are known to need warmup + decay; a constant LR both destabilizes early steps and
   leaves accuracy on the table late in training.
   *Impact: medium-high on final accuracy & stability. Effort: low.*

3. **Train/inference contact-masking skew.** Training masks COP/GRF and builds the torque target with
   **GT** `contactBoolean` (~1900–1927, 2025–2039). At inference the model must use its **own**
   predicted contact. The model is therefore never penalized for how its *predicted* contact errors
   corrupt COP/GRF/torque, so swing-phase leakage and contact-boundary errors are unmodeled at train
   time. *Impact: medium on real-world accuracy. Effort: medium.*

### B. Data / normalization quality

4. **Z-score stats are fragile.** Normalizers are fit from only `max_batches=100` (~1841 call site
   `compute_normalizers_from_loader(..., max_batches=100)`), and COP/GRF targets that are zero during
   swing are included in the statistics — pulling means toward zero and inflating std. This biases the
   Z-scale the network learns to hit. *Impact: medium. Effort: low-medium.*

5. **MSE is spike-sensitive.** All primary channels use plain MSE (`mse_loss`, ~1824). GRF/COP have
   heel-strike transients and occasional artifacts; MSE over-weights those outliers. A Huber/smooth-L1
   would be more robust and is a common accuracy win here. *Impact: medium. Effort: low.*

### C. Physics coupling

6. **Torque reconstruction uses noisy video Jacobians/rotations by default.** `select_torque_jacobians`
   returns the preprocessed (video-derived) `jacp/jacr` (~1584), and `compute_full_external_moments`
   uses the video `rot_w_to_ga`. So GRF/COP gradients are contaminated by kinematic-Jacobian noise
   unless `use_gt_jacob_and_rot` is set (LOSO/eval only). Worth making the training default explicit and
   documented, since it changes what "physics-consistent" means. *Impact: medium, subtle. Effort: medium.*

### D. Dead / confusing code (risk + maintainability)

7. **`loss_type` is hardcoded to `"mse"`** (~1933) — the entire `m3e` branch and `cop_target_z`
   machinery are dead. `m3e_loss` also uses `jnp.sum` (not mean), so if ever enabled it would be scaled
   wildly differently from MSE.
8. **Vestigial auxiliary heads.** `split_model_predictions` is always called with
   `qfrc_inverse_output_dim=0, rotation_output_dim=0` (~1851), so `qfrc_inverse_loss`,
   `rotation_loss`, `jacobian_loss` and their reg terms are hard-zeroed twice (~1948–2005, then again
   ~2292–2297). This is a lot of live-but-inert code that obscures the real loss and invites bugs.
9. **Best-model-by-val-loss compares a moving target.** `val total_loss` mixes 6 weighted terms whose
   relative scale depends on the (unswept) weights; a physically meaningful scalar (e.g. moment MAE in
   %BW·h, which LOSO already uses) is a more reliable selection metric.

### E. Minor / worth noting

10. **Fixed 80/20 subject split** (~3838) with a single seed — fine for a run, but there is no k-fold or
    guaranteed cohort balancing in the base trainer (balancing weights exist, but the *split* itself
    isn't stratified by cohort/gender/speed).
11. **Static token is a single prepended vector** shared across the window; the model can only inject
    subject context through attention to one token. FiLM/conditioning per layer is a cheap potential
    upgrade if subject generalization is weak.
12. **`decode_cop_signal_to_length` with `use_grf_norm_cop`** divides COP by per-foot GRF magnitude
    (~1451). Near low force this is numerically softened only by `eps=1e-6`; combined with contact
    masking it's probably fine, but it's a latent instability if contact/GRF get small but nonzero.

---

## 2. Step-by-step remediation plans

Each plan is self-contained, ordered by *recommended sequence* (cheap wins and de-risking first).
Do them one at a time and re-run a short training smoke test (a few hundred steps) after each.

---

### PLAN 1 — Make the diagnostic "effect losses" opt-in (biggest cheap win)

**Goal:** stop paying ~4 extra `jax.grad` passes per train/eval step unless explicitly requested.

**Files:** `TransformerFinal/train.py`

**Steps:**
1. Add a boolean parameter `compute_effect_diagnostics: bool = False` to `compute_total_loss`
   (signature ~1832), and thread it through `make_train_step` and `make_eval_step` (~2348, ~2421) and
   their internal `compute_total_loss(...)` calls (~2376, ~2430).
2. In `compute_total_loss`, wrap the four `jax.grad` blocks (~2252–2272) in
   `if compute_effect_diagnostics:`. In the `else` path, leave
   `torque_cop_effect_loss`/`torque_grf_effect_loss` as the zero tensors already created at ~2248–2249.
3. Add a CLI flag `--effect_diagnostics` (default `False`) in `main()`'s argparse block (near the other
   toggles ~3538) and pass it into `make_train_step`/`make_eval_step`.
4. Verify the two metrics still exist in the returned dict (they do — zeros when off) so the print/log
   code at ~4598, ~4944, ~5016–5025, ~5241–5258 keeps working without edits.

**Validation:** run `train_single_model.py` for ~200 steps with and without the flag; confirm
identical loss curve for the *optimized* terms and a clear step-time drop when off. Expect a
meaningful steps/sec improvement.

**Risk:** very low — the terms are already excluded from `total_loss`, so gradients/updates are
unchanged.

---

### PLAN 2 — Add LR warmup + cosine decay (and optional weight EMA)

**Goal:** stabilize early training and improve final accuracy with a standard transformer schedule.

**Files:** `TransformerFinal/train.py`

**Steps:**
1. In `create_train_state` (~2337), replace the scalar `learning_rate` in `optax.adamw` with an
   `optax.warmup_cosine_decay_schedule`. Add params: `warmup_steps`, `total_steps`, `peak_lr`
   (= current `learning_rate`), `end_lr` (e.g. `peak_lr * 0.05`).
   ```python
   schedule = optax.warmup_cosine_decay_schedule(
       init_value=peak_lr * 0.01, peak_value=peak_lr,
       warmup_steps=warmup_steps, decay_steps=total_steps, end_value=end_lr)
   tx = optax.chain(optax.clip_by_global_norm(1.0),
                    optax.adamw(schedule, weight_decay=weight_decay))
   ```
2. Compute `total_steps = steps_per_epoch * args.epochs` and `warmup_steps = int(0.03 * total_steps)`
   (or `~1–2` epochs) in `main()` *after* `steps_per_epoch = len(train_loader)` is known (~4139), and
   pass them into `create_train_state` at ~4082. (Move the `create_train_state` call to after
   `steps_per_epoch` is computed, or compute `steps_per_epoch` earlier from
   `train_loader.total_windows // batch_size`.)
3. Add CLI flags `--warmup_frac` (default 0.03) and `--end_lr_frac` (default 0.05).
4. (Optional, separable) Add weight EMA: wrap with `optax.ema` or maintain a manual EMA of `params`,
   and **evaluate/checkpoint the EMA weights**. If added, save EMA params in the checkpoint dict at
   ~5443 and load them in infer.

**Validation:** train a short run; confirm the LR actually ramps then decays (log
`schedule(state.step)`), and that val loss at a fixed epoch is ≤ the constant-LR baseline. Keep the
same `--learning_rate` value as `peak_lr` so the comparison is apples-to-apples.

**Risk:** low. Keep a `--no_lr_schedule` escape hatch that falls back to the constant LR for A/B.

---

### PLAN 3 — Remove dead loss code (de-risk before touching the loss)

**Goal:** shrink `compute_total_loss` to only the live terms so later edits are safe and readable.

**Files:** `TransformerFinal/train.py`

**Steps:**
1. Delete the `m3e` branch and the `loss_type` switch (~1933–1946, 2236–2243); keep only the MSE path.
   Remove the now-unused `m3e_loss` (~1828) and `cop_target_z/grf_target_z/moments_target_z`.
2. Delete the always-`None` auxiliary-head machinery: `qfrc_inverse_loss`, `qfrc_inverse_input_reg_loss`,
   `rotation_loss`, `rotation_input_reg_loss`, `jacobian_loss`, `jacobian_input_reg_loss`
   (blocks at ~1948–2005 and the re-zeroing at ~2292–2297). Keep them as literal `0.0` in the returned
   metrics dict **only if** the plotting/logging code references those keys — grep first
   (`grep -n "rotation_loss\|jacobian_loss\|qfrc_inverse_loss" train.py infer.py`) and either keep the
   dict keys as constants or remove the corresponding print lines.
3. Keep `split_model_predictions` but drop the dead `qfrc_inverse_output_dim/rotation_output_dim`
   parameters, or leave the function and just stop passing nonzero dims (it is already 0 everywhere).

**Validation:** `python -c "import ast; ast.parse(open('TransformerFinal/train.py').read())"` then run
a ~100-step smoke train; the optimized loss must be numerically identical to before (these terms were
already zero).

**Risk:** low, but do it as its own commit so a regression is easy to bisect.

---

### PLAN 4 — Robust primary loss (MSE → Huber) + honest normalizer stats

**Goal:** reduce sensitivity to GRF/COP transients and remove swing-zero bias from Z-scores.

**Files:** `TransformerFinal/train.py`

**Steps (4a — Huber):**
1. Add `huber_loss(pred, target, weights, delta=1.0)` next to `mse_loss` (~1824) using
   `optax.huber_loss` or a manual smooth-L1, preserving the `weights` semantics and the same reduction
   (weighted mean).
2. Replace the COP/GRF/Moments MSE calls (~1935–1937) with Huber. Because inputs are Z-scored, a
   `delta≈1.0` is a sensible knee. Add a `--robust_loss {mse,huber}` flag (default `huber`) and a
   `--huber_delta` flag so it is A/B-testable.
3. Keep the `/4`, `/6`, `/2` per-channel normalizers (~2289–2291) unchanged so loss weights stay
   comparable.

**Steps (4b — normalizer honesty):**
4. Raise the sampling in `compute_normalizers_from_loader` from `max_batches=100` to cover a larger,
   representative fraction (e.g. 300–500, or `min(len(train_loader), N)`), at the call site ~3938.
5. For COP/GRF specifically, compute Z-score stats over **contact/stance frames only** (exclude
   swing-zeroed frames) so the mean/std reflect the values the model actually predicts. Concretely, in
   `compute_normalizers_from_loader`, gather a stance mask from `batch["contactBoolean"]` and index
   `cop_samples`/`grf_samples` before `np.mean/np.std`. Leave contact/moments as-is.

**Validation:** print the new normalizer means/stds and confirm COP/GRF means move away from ~0 and
std shrinks. Train short; expect equal-or-better val COP/GRF MAE and fewer loss spikes. Because
changing normalizers changes the Z targets, **retrain from scratch** for the comparison (do not
fine-tune an old checkpoint against new stats).

**Risk:** medium — normalizer changes require a fresh checkpoint and a matching change wherever infer
reloads/reuses stored normalizers (they are saved in the checkpoint, so infer will pick up the new
stats automatically; verify infer reads normalizers from the checkpoint and not a cached file).

---

### PLAN 5 — Close the train/inference contact-masking skew

**Goal:** make training reflect that COP/GRF/torque at inference are gated by *predicted* contact.

**Files:** `TransformerFinal/train.py` (`compute_total_loss`), and confirm inference in `infer.py`.

**Approach (pick one; B is the safer first step):**

- **Option A — Soft masking.** Replace the hard `(contact > 0.5)` GT masks (~1900–1927, 2025–2039)
  with the **predicted** soft contact probability `contact_pred` as a multiplicative gate for the
  *torque* reconstruction path only (keep the direct COP/GRF supervision masked by GT so those targets
  stay well-defined). This lets contact errors flow into the physics loss.
- **Option B — Scheduled sampling.** Add `--contact_mask_source {gt,pred,mixed}` and a schedule that
  starts with GT masking and linearly mixes in predicted contact over epochs
  (`mask = (1-α)·gt + α·pred_hard`, α from 0→~0.5 across training using the `epoch/total_epochs`
  already available at ~2301). This anneals the model toward its inference-time behavior without
  destabilizing early training.

**Steps:**
1. Thread `epoch`/`total_epochs` (already present) into an `alpha` and compute a blended mask.
2. Apply the blended mask in the torque physical-decode block (~2025–2039) and, if using Option A for
   torque only, leave the direct COP/GRF weight masks (~1902–1914) on GT.
3. **Verify inference parity**: in `infer.py`, confirm COP/GRF are gated by the model's predicted
   contact (search `_convert_output_to_physical_predictions` / contact thresholding ~3591). If infer
   uses GT contact anywhere for masking, fix that too — otherwise the metric is optimistic.

**Validation:** compare val moment MAE with GT-masked vs predicted-masked evaluation; the gap between
them should shrink after training with this change. This is the metric that most closely tracks
real-world (OpenCap/LOSO) performance.

**Risk:** medium — can slow convergence; keep default `gt` and make `mixed` opt-in until validated.

---

### PLAN 6 — Physically meaningful best-model selection

**Goal:** stop selecting checkpoints on a weight-dependent composite; select on a stable metric.

**Files:** `TransformerFinal/train.py` (best-model block ~5425–5470)

**Steps:**
1. During validation, already-computed physical predictions exist; compute a single scalar
   **moment MAE in %BW·h** (mirror the LOSO `inner_selection_objective`) averaged over stance frames,
   right+left, and store it per epoch.
2. Add `--best_model_metric {val_loss,torque_rmse,moment_mae_bwh}` (default `moment_mae_bwh`). Extend
   the existing `BestModelByTorque` path (~5192, 5425) rather than replacing it, so current behavior is
   preserved when requested.
3. Save the selected metric value into the checkpoint dict (~5443) for provenance.

**Validation:** confirm the chosen epoch matches the epoch a human would pick from the moment-MAE
curve, and that it is stable across two seeds.

**Risk:** low; it only changes *which* checkpoint is saved as `best_model.pkl`, not training.

---

### PLAN 7 — (Optional accuracy) Per-layer subject conditioning (FiLM)

**Goal:** stronger subject/anthropometry conditioning than a single prepended token.

**Files:** `TransformerFinal/train.py` (`TransformerBlock`, `KinematicsToCOPGRFMoments`)

**Steps:**
1. From the static branch (~1677) produce per-layer FiLM parameters `(γ_l, β_l)` via a small MLP.
2. In `TransformerBlock.__call__` (~1619), after each LayerNorm apply `x = γ_l * x + β_l` (broadcast
   over sequence). Pass the static embedding into the block.
3. Keep the prepended static token too (they are complementary) or ablate it behind a flag.

**Validation:** LOSO / held-out-subject moment MAE should improve if the win is real; if not, revert.
This is exploratory — do it last and only if subject generalization (LOSO) is the bottleneck.

**Risk:** medium (architecture change → new checkpoints, HPO may need re-running). Gate behind a flag
`--subject_film` so the default architecture is unchanged.

---

## 3. Suggested execution order

1. **Plan 1** (remove wasted grad compute) — immediate speedup, zero accuracy risk.
2. **Plan 3** (delete dead loss code) — de-risk the loss before editing it.
3. **Plan 2** (LR schedule) — likely the biggest single accuracy/stability win.
4. **Plan 4** (Huber + normalizer stats) — robustness; needs a fresh checkpoint.
5. **Plan 6** (selection metric) — makes 2/4 comparisons trustworthy.
6. **Plan 5** (contact skew) — targets real-world/LOSO accuracy specifically.
7. **Plan 7** (FiLM) — optional, only if LOSO generalization is the ceiling.

After Plans 1–3 (no accuracy change expected), keep one checkpoint as a numerical-equivalence baseline
to prove those refactors didn't move the loss. Plans 2/4/5/6/7 each need an A/B against that baseline
on the same val split and seed.
