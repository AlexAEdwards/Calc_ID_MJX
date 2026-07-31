"""Training script for Q-position residual refinement model.

Predicts denoised joint angles (q_prime) from kinematics inputs using a residual
formulation (batch keys use ``pos_noised`` for the base position tensor in
``q_prime = pos_noised + delta_q``):

    delta_q = model(pos_noised, static_ctx)
    q_prime = pos_noised + delta_q

**Default dataset layout** (standalone ``train_refine_q`` / trusted HPO trees):

    Model input per frame: **(pos, vel, acc)**
        ``pos_inputs_noised`` (standard) or ``pos_inputs`` (Loso cache) -> ``(T, 16)``

    Ground truth (clean 16-D positions)::
        pos_inputs.npy

    Optional files ``vel_inputs*_noised`` / ``acc_inputs*_noised`` (or Loso unsuffixed
    vel/acc) are loaded when present for downstream physics pipelines; they are **not**
    concatenated into the transformer input.

**Loso_Combined.py refine-q cache:** when ``loso_combined_trial_info.json`` and
``pos_gt.npy`` are present, the loader uses OpenCap **positions** from
``pos_inputs.npy`` and MoCap positions from ``pos_gt.npy``. Optional
``vel_inputs.npy`` / ``acc_inputs.npy`` fill auxiliary arrays (zeros if missing) and
are concatenated into the transformer input.
This branch takes precedence over ``*_noised.npy`` when both exist. Otherwise the
standard ``pos_inputs_noised.npy`` + ``pos_inputs.npy`` (clean GT) layout applies.

Output:
    q_prime  (T, 16)

Loss:
    recon_loss = masked_MSE(q_prime, pos_gt)
    reg_loss   = mean(delta_q ** 2)
    total_loss = recon_loss + lambda_reg * reg_loss

Note: This script does NOT perform MuJoCo forward simulation, Jacobian
computation, or inverse dynamics.

Common files: Height_m.npy, Mass_kg.npy, metadata (see _load_trial_for_refine).
"""

import json
import pickle
import random
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# ---------------------------------------------------------------------------
# Import shared windowing utilities from data_loader.py (same directory).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    build_window_start_indices,
    NOISED_FILE_SUFFIX,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_TRIAL_LENGTH = 30
POS_INPUT_DIM    = 16
OUTPUT_DIM       = 16   # same column layout as pos_inputs.npy
STATIC_DIM       = 3    # height, mass, gender
VEL_INPUT_DIM    = 19
ACC_INPUT_DIM    = 19
INPUT_DIM        = POS_INPUT_DIM + VEL_INPUT_DIM + ACC_INPUT_DIM
ACC_INPUT_MTP_REMOVE_IDXS = (11, 17)

_FEATURE_DIM_WARNINGS_EMITTED = set()


# ===========================================================================
# Model Architecture
#
# TransformerBlock and SinusoidalPosEmb mirror the definitions in train.py
# (lines 1498-1539) but are reproduced here to avoid importing train.py,
# which has side-effects (WandB setup, runtime env config, etc.).
# ===========================================================================

class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len  = x.shape[1]
        position = jnp.arange(seq_len)
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = position[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return x + emb[None, :, :]


class TransformerBlock(nn.Module):
    d_model:      int
    num_heads:    int
    ff_dim:       int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Self-attention with pre-LN
        residual = x
        x = nn.LayerNorm()(x)
        x = residual + nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x, x, deterministic=not train)

        # Feed-forward with pre-LN
        residual = x
        x  = nn.LayerNorm()(x)
        ff = nn.Dense(self.ff_dim)(x)
        ff = nn.gelu(ff)
        ff = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff)
        ff = nn.Dense(self.d_model)(ff)
        ff = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff)
        return residual + ff


class QRefineTransformer(nn.Module):
    """Residual Transformer for joint-angle denoising / refinement.

    Forward pass:
        delta_q = network(pos_noised, static_ctx)
        q_prime = pos_noised + delta_q        (computed outside the module)

    Architecture:
        Dense(d_model) -> GELU -> LayerNorm
        -> SinusoidalPosEmb
        -> static-context token prepended as position 0
        -> N x TransformerBlock
        -> remove static token -> LayerNorm -> Dense(output_dim)
    """

    input_dim:    int   = INPUT_DIM
    static_dim:   int   = STATIC_DIM
    output_dim:   int   = OUTPUT_DIM
    d_model:      int   = 256
    num_heads:    int   = 4
    num_layers:   int   = 4
    ff_dim:       int   = 1024
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x:              jnp.ndarray,   # (B, T, input_dim)
        static_context: jnp.ndarray,   # (B, static_dim)
        train:          bool = True,
    ) -> jnp.ndarray:                  # (B, T, output_dim) -- the residual delta_q

        # Input projection
        x = nn.Dense(self.d_model)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        # Sinusoidal positional embedding
        x = SinusoidalPosEmb(self.d_model)(x)

        # Static-context token prepended at sequence position 0
        tok = nn.Dense(self.d_model)(static_context)
        tok = nn.gelu(tok)
        tok = nn.LayerNorm()(tok)
        tok = tok[:, None, :]                           # (B, 1, d_model)
        x   = jnp.concatenate([tok, x], axis=1)        # (B, T+1, d_model)

        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        # Remove static token, normalize, project to output
        x = x[:, 1:, :]                                # (B, T, d_model)
        x = nn.LayerNorm()(x)
        delta_q = nn.Dense(self.output_dim)(x)         # (B, T, output_dim)
        return delta_q


# ===========================================================================
# Lightweight Trial Loader
#
# Deliberately minimal -- does NOT require Jacobians or qfrc_inverse so that
# this model can be trained as a preprocessing step before the full physics
# pipeline is available.
# ===========================================================================

def _load_npy(path: Path) -> Optional[np.ndarray]:
    """Load a .npy file as float32, returning None on any error."""
    try:
        return np.load(path).astype(np.float32)
    except Exception:
        return None


def _coerce_2d_feature_dim(
    arr: np.ndarray,
    expected_dim: int,
    feature_name: str,
    trial_path: Path,
) -> Optional[np.ndarray]:
    """Force a temporal feature array to expected_dim via trim/pad."""
    if arr.ndim != 2:
        key = (str(trial_path), feature_name, "ndim")
        if key not in _FEATURE_DIM_WARNINGS_EMITTED:
            print(
                f"  [QRefineDataLoader] Skipping {trial_path}: {feature_name} must be 2D, got shape {arr.shape}",
                flush=True,
            )
            _FEATURE_DIM_WARNINGS_EMITTED.add(key)
        return None
    current_dim = int(arr.shape[1])
    if current_dim == int(expected_dim):
        return arr

    # ProcessData compatibility:
    # when acceleration is saved in the 21-column pre-MTP-removal schema,
    # drop the two MTP columns to match the 19-column training layout.
    if (
        feature_name in ("acc_inputs", "acc_inputs_noised")
        and current_dim == 21
        and int(expected_dim) == ACC_INPUT_DIM
    ):
        key = (feature_name, current_dim, int(expected_dim), "drop_mtp_11_17")
        if key not in _FEATURE_DIM_WARNINGS_EMITTED:
            print(
                "  [QRefineDataLoader] acc_inputs / acc_inputs_noised is 21-col pre-MTP layout; "
                "dropping columns 11 and 17 (mtp_r, mtp_l) for 19-col compatibility.",
                flush=True,
            )
            _FEATURE_DIM_WARNINGS_EMITTED.add(key)
        return np.delete(arr, ACC_INPUT_MTP_REMOVE_IDXS, axis=1)

    key = (feature_name, current_dim, int(expected_dim))
    if key not in _FEATURE_DIM_WARNINGS_EMITTED:
        action = "trimming" if current_dim > int(expected_dim) else "padding"
        print(
            f"  [QRefineDataLoader] {feature_name} dim mismatch ({current_dim} -> {expected_dim}); {action} for compatibility.",
            flush=True,
        )
        _FEATURE_DIM_WARNINGS_EMITTED.add(key)

    if current_dim > int(expected_dim):
        return arr[:, : int(expected_dim)]
    pad = np.zeros((arr.shape[0], int(expected_dim) - current_dim), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def _optional_vel_acc_streams(
    td_path: Path,
    *,
    loso_branch: bool,
    sfx: str,
    T: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load vel/acc when present; zero-pad/truncate to length T for physics/aux code."""
    if loso_branch:
        vel_p = td_path / "vel_inputs.npy"
        acc_p = td_path / "acc_inputs.npy"
        acc_p_alt = td_path / "accel_inputs.npy"
        acc_feature_name = "acc_inputs"
    else:
        vel_p = td_path / f"vel_inputs{sfx}.npy"
        acc_p = td_path / f"acc_inputs{sfx}.npy"
        acc_p_alt = td_path / f"accel_inputs{sfx}.npy"
        acc_feature_name = "acc_inputs_noised"
    vel_raw = _load_npy(vel_p)
    acc_raw = _load_npy(acc_p)
    if acc_raw is None:
        acc_raw = _load_npy(acc_p_alt)
    vel: Optional[np.ndarray] = None
    acc: Optional[np.ndarray] = None
    if vel_raw is not None:
        vel = _coerce_2d_feature_dim(vel_raw, VEL_INPUT_DIM, vel_p.name, td_path)
    if acc_raw is not None:
        # Use canonical feature_name so MTP-column dropping logic applies.
        acc = _coerce_2d_feature_dim(acc_raw, ACC_INPUT_DIM, acc_feature_name, td_path)

    def _fill(x: Optional[np.ndarray], dim: int) -> np.ndarray:
        out = np.zeros((T, dim), dtype=np.float32)
        if x is None:
            return out
        n = min(T, int(x.shape[0]))
        out[:n] = np.asarray(x[:n], dtype=np.float32)
        return out

    return _fill(vel, VEL_INPUT_DIM), _fill(acc, ACC_INPUT_DIM)


def _load_trial_for_refine(td_path: Path) -> Optional[Dict]:
    """Lightweight loader for position-refinement training.

    **Loso_Combined cache** when ``loso_combined_trial_info.json`` exists: unsuffixed
    ``pos_inputs.npy`` (OpenCap positions) and ``pos_gt.npy`` (MoCap) as supervision.
    Optional ``vel_inputs.npy`` / ``acc_inputs.npy`` supply auxiliary kinematics (zeros
    if missing). This branch is tried **first** so leftover ``*_noised.npy`` files in
    an old cache cannot override OpenCap vs MoCap assignment.

    **Standard layout** otherwise: ``pos_inputs_noised.npy`` and ``pos_inputs.npy``
    as clean GT. Vel/acc ``*_noised`` files are optional.

    Statics: Height_m.npy, Mass_kg.npy, BiologicalSex from metadata.

    Returns None when any required array is missing.
    """
    td_path = Path(td_path)
    sfx = NOISED_FILE_SUFFIX  # "_noised"
    noised_pos = td_path / f"pos_inputs{sfx}.npy"
    loso_marker = td_path / "loso_combined_trial_info.json"

    loso_branch = (
        loso_marker.exists()
        and (td_path / "pos_gt.npy").exists()
        and (td_path / "pos_inputs.npy").exists()
    )

    if loso_branch:
        pos_noised = _load_npy(td_path / "pos_inputs.npy")
        pos_gt = _load_npy(td_path / "pos_gt.npy")
        if pos_noised is None or pos_gt is None:
            return None
        pos_noised = _coerce_2d_feature_dim(pos_noised, POS_INPUT_DIM, "pos_inputs", td_path)
        pos_gt = _coerce_2d_feature_dim(pos_gt, OUTPUT_DIM, "pos_gt", td_path)
        if pos_noised is None or pos_gt is None:
            return None
    elif noised_pos.exists():
        pos_noised = _load_npy(noised_pos)
        if pos_noised is None:
            return None
        pos_noised = _coerce_2d_feature_dim(pos_noised, POS_INPUT_DIM, "pos_inputs_noised", td_path)
        if pos_noised is None:
            return None
        pos_gt = _load_npy(td_path / "pos_inputs.npy")
        if pos_gt is None:
            return None
        pos_gt = _coerce_2d_feature_dim(pos_gt, OUTPUT_DIM, "pos_inputs", td_path)
        if pos_gt is None:
            return None
    else:
        return None

    T = min(int(pos_noised.shape[0]), int(pos_gt.shape[0]))
    pos_noised = pos_noised[:T]
    pos_gt = pos_gt[:T]
    vel_noised, acc_noised = _optional_vel_acc_streams(td_path, loso_branch=loso_branch, sfx=sfx, T=T)

    # Height and mass -- required scalars stored as (T, 1) arrays
    height = _load_npy(td_path / "Height_m.npy")
    mass   = _load_npy(td_path / "Mass_kg.npy")
    if height is None or mass is None:
        return None
    height = np.asarray(height, dtype=np.float32).reshape(-1, 1)
    mass   = np.asarray(mass,   dtype=np.float32).reshape(-1, 1)

    # BiologicalSex -- searched up the directory tree via metadata JSON
    gender_val = 0.5
    current = td_path
    for _ in range(5):
        for md_name in ("metadata.json", "subject_metadata.json"):
            md_path = current / md_name
            if md_path.exists():
                try:
                    with open(md_path) as f:
                        md = json.load(f)
                    sex = md.get("BiologicalSex", "").lower()
                    if   sex == "male":   gender_val = 1.0
                    elif sex == "female": gender_val = 0.0
                except Exception:
                    pass
        current = current.parent

    return {
        "pos":          pos_noised,
        "vel":          vel_noised,
        "acc":          acc_noised,
        "pos_gt":       pos_gt,
        "height":       height,
        "mass":         mass,
        "gender":       gender_val,
    }


# ===========================================================================
# Trial Discovery
# ===========================================================================

def discover_trials(data_dir: str) -> List[Dict]:
    """Walk <data_dir>/<Subject>/<Trial>/ProcessedData/ and return trial dicts.

    Includes a trial when either:

    * **Loso_Combined cache:** ``loso_combined_trial_info.json`` plus
      ``pos_inputs.npy`` and ``pos_gt.npy`` (checked **before** the standard layout so
      stale ``*_noised`` files do not hide LOSO trials), or
    * **Standard refine-q layout:** ``pos_inputs.npy`` (clean GT) and
      ``pos_inputs_noised.npy`` exist (optional vel/acc at load time).
    """
    data_dir = Path(data_dir)
    trials: List[Dict] = []
    noised_pos = f"pos_inputs{NOISED_FILE_SUFFIX}.npy"

    for subject_path in sorted(data_dir.iterdir()):
        if not subject_path.is_dir():
            continue
        subject_name = subject_path.name
        for trial_path in sorted(subject_path.iterdir()):
            if not trial_path.is_dir():
                continue
            pd = trial_path / "ProcessedData"
            if not pd.exists():
                continue

            has_loso = (
                (pd / "loso_combined_trial_info.json").exists()
                and (pd / "pos_gt.npy").exists()
                and (pd / "pos_inputs.npy").exists()
            )
            has_standard = (pd / "pos_inputs.npy").exists() and (pd / noised_pos).exists()
            if has_loso:
                length_path = pd / "pos_inputs.npy"
            elif has_standard:
                length_path = pd / "pos_inputs.npy"
            else:
                continue

            try:
                length = int(np.load(length_path, mmap_mode="r").shape[0])
            except Exception:
                continue
            if length < MIN_TRIAL_LENGTH:
                continue
            trials.append({
                "subject":            subject_name,
                "trial_name":         f"{subject_name}/{trial_path.name}",
                "training_data_path": str(pd),
                "length":             length,
            })

    return trials


# ===========================================================================
# Data Loader
# ===========================================================================

class QRefineDataLoader:
    """Sliding-window data loader for position-refinement training.

    If ``one_batch`` is True, a single collated batch contains every window from
    every trial (optional global shuffle of windows when ``shuffle`` is True).

    Each yielded batch is a dict of NumPy arrays (not yet on device):
        input            (B, W, INPUT_DIM) -- concat(pos_noised, vel_noised, acc_noised)
        pos_noised       (B, W, 16)         -- base positions for q_prime = pos_noised + delta_q
        pos_gt           (B, W, 16)         -- supervision target (see _load_trial_for_refine)
        vel_noised       (B, W, 19)        -- aux stream (vel inputs to the transformer)
        acc_noised       (B, W, 19)        -- aux stream (acc inputs to the transformer)
        static_context   (B, 3)       -- per-trial statics
        supervision_mask (B, W, 1)    -- all ones; every window frame is supervised
    """

    def __init__(
        self,
        trials:                   List[Dict],
        window_size:              int  = 64,
        stride:                   int  = 16,
        batch_size:               int  = 32,
        shuffle:                  bool = True,
        prediction_margin_frames: int  = 0,
        one_batch:                 bool = False,
    ):
        self.window_size              = window_size
        self.stride                   = stride
        self.batch_size               = batch_size
        self.shuffle                  = shuffle
        self.prediction_margin_frames = 0
        self.one_batch                = bool(one_batch)
        self.trials                   = trials

        # Pre-compute per-trial window counts for __len__
        self.trial_window_counts: List[Tuple[Dict, int]] = []
        self.total_windows = 0
        for t in trials:
            length = t.get("length", 0)
            if length >= window_size:
                n = len(build_window_start_indices(length, window_size, stride))
            else:
                n = 0
            self.trial_window_counts.append((t, n))
            self.total_windows += n

    def __len__(self) -> int:
        if self.one_batch:
            return 1
        return max(1, self.total_windows // self.batch_size)

    # -- internals -------------------------------------------------------------

    def _load_trial(self, trial_info: Dict) -> Optional[Dict]:
        return _load_trial_for_refine(Path(trial_info["training_data_path"]))

    def _extract_windows(self, trial_data: Dict) -> List[Dict]:
        pos_noised = trial_data["pos"]     # (T, 16)
        vel_noised = trial_data["vel"]     # (T, 19)
        acc_noised = trial_data["acc"]     # (T, 19)
        pos_gt     = trial_data["pos_gt"]  # (T, 16)
        T          = pos_noised.shape[0]

        # Static context vector
        static_ctx = np.array([
            float(trial_data["height"].reshape(-1)[0]),
            float(trial_data["mass"].reshape(-1)[0]),
            float(trial_data["gender"]),
        ], dtype=np.float32)

        window_starts = build_window_start_indices(T, self.window_size, self.stride)
        windows: List[Dict] = []

        for start in window_starts:
            end = start + self.window_size
            if end > T:
                continue
            mask = np.ones((self.window_size, 1), dtype=np.float32)
            pos_w = pos_noised[start:end].copy()
            vel_w = vel_noised[start:end].copy()
            acc_w = acc_noised[start:end].copy()
            input_w = np.concatenate([pos_w, vel_w, acc_w], axis=-1).astype(np.float32)

            windows.append({
                "input":            input_w,
                "pos_noised":       pos_w,
                "vel_noised":       vel_w,
                "acc_noised":       acc_w,
                "pos_gt":           pos_gt[start:end].copy(),
                "static_context":   static_ctx,
                "supervision_mask": mask,
            })

        return windows

    @staticmethod
    def _collate(windows: List[Dict]) -> Dict:
        return {
            k: np.stack([w[k] for w in windows])
            for k in (
                "input",
                "pos_noised",
                "vel_noised",
                "acc_noised",
                "pos_gt",
                "static_context",
                "supervision_mask",
            )
        }

    # -- iterator --------------------------------------------------------------

    def __iter__(self):
        trial_list = list(self.trials)
        if self.shuffle:
            random.shuffle(trial_list)

        buffer: List[Dict] = []

        for trial_info in trial_list:
            trial_data = self._load_trial(trial_info)
            if trial_data is None:
                print(
                    f"  [QRefineDataLoader] Skipping "
                    f"{trial_info.get('trial_name', trial_info['training_data_path'])}"
                    f": missing required files.",
                    flush=True,
                )
                continue

            windows = self._extract_windows(trial_data)
            if self.shuffle and not self.one_batch:
                random.shuffle(windows)
            buffer.extend(windows)

        if self.one_batch:
            if buffer and self.shuffle:
                random.shuffle(buffer)
            if buffer:
                yield self._collate(buffer)
            return

        partial: List[Dict] = []
        for w in buffer:
            partial.append(w)
            if len(partial) >= self.batch_size:
                yield self._collate(partial[: self.batch_size])
                partial = partial[self.batch_size :]

        if partial:
            yield self._collate(partial)


# ===========================================================================
# Training Infrastructure
# ===========================================================================

def create_train_state(
    rng,
    model:         QRefineTransformer,
    input_shape:   Tuple,
    static_shape:  Tuple,
    learning_rate: float = 1e-4,
    weight_decay:  float = 0.01,
    total_steps:   int   = 10_000,
    warmup_steps:  int   = 500,
) -> train_state.TrainState:
    """Initialise model parameters and create an Optax TrainState."""
    dummy_input  = jnp.ones(input_shape)
    dummy_static = jnp.ones(static_shape)
    params = model.init(rng, dummy_input, dummy_static, train=False)["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=learning_rate * 1e-2,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=weight_decay),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(model: QRefineTransformer):
    """Return a JIT-compiled training step.

    lambda_reg is a traced JAX scalar so the fade schedule can change its
    value each epoch without triggering recompilation.
    """

    @jax.jit
    def train_step(
        state:       train_state.TrainState,
        batch:       Dict,
        dropout_rng: jnp.ndarray,
        lambda_reg:  float,
    ):
        def loss_fn(params):
            delta_q = model.apply(
                {"params": params},
                batch["input"],
                batch["static_context"],
                train=True,
                rngs={"dropout": dropout_rng},
            )                                             # (B, T, 16)

            q_prime = batch["pos_noised"] + delta_q      # (B, T, 16)
            mask    = batch["supervision_mask"]           # (B, T, 1)

            # Masked mean-squared reconstruction error
            sq_err     = jnp.mean(
                (q_prime - batch["pos_gt"]) ** 2,
                axis=-1, keepdims=True,
            )                                             # (B, T, 1)
            n_valid    = jnp.maximum(jnp.sum(mask), 1.0)
            recon_loss = jnp.sum(sq_err * mask) / n_valid

            # L2 regularisation: penalise residuals deviating from zero
            reg_loss = jnp.mean(delta_q ** 2)

            total_loss = recon_loss + lambda_reg * reg_loss
            metrics = {
                "recon_loss": recon_loss,
                "reg_loss":   reg_loss,
                "total_loss": total_loss,
            }
            return total_loss, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return train_step


def make_eval_step(model: QRefineTransformer):
    """Return a JIT-compiled evaluation step (no gradient, dropout off)."""

    @jax.jit
    def eval_step(
        state:      train_state.TrainState,
        batch:      Dict,
        lambda_reg: float,
    ):
        delta_q = model.apply(
            {"params": state.params},
            batch["input"],
            batch["static_context"],
            train=False,
        )
        q_prime = batch["pos_noised"] + delta_q
        mask    = batch["supervision_mask"]

        sq_err     = jnp.mean(
            (q_prime - batch["pos_gt"]) ** 2,
            axis=-1, keepdims=True,
        )
        n_valid    = jnp.maximum(jnp.sum(mask), 1.0)
        recon_loss = jnp.sum(sq_err * mask) / n_valid
        reg_loss   = jnp.mean(delta_q ** 2)
        total_loss = recon_loss + lambda_reg * reg_loss

        return {
            "recon_loss": recon_loss,
            "reg_loss":   reg_loss,
            "total_loss": total_loss,
        }

    return eval_step


# ===========================================================================
# Helpers
# ===========================================================================

def _zero_metrics() -> Dict[str, float]:
    return {"recon_loss": 0.0, "reg_loss": 0.0, "total_loss": 0.0}


def _avg_metrics(acc: Dict[str, float], count: int) -> Dict[str, float]:
    return {k: v / max(count, 1) for k, v in acc.items()}


def _save_checkpoint(
    path: Path,
    state: train_state.TrainState,
    epoch: int,
    best_val_loss: float,
    args,
) -> None:
    with open(path, "wb") as f:
        pickle.dump({
            "params":        state.params,
            "epoch":         epoch,
            "best_val_loss": best_val_loss,
            "args":          vars(args),
        }, f)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train QRefineTransformer -- residual joint-angle denoising"
    )

    # Paths
    parser.add_argument(
        "--data_dir", required=True,
        help="Root data directory containing Subject/Trial/ProcessedData/",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save checkpoints and loss history",
    )

    # Architecture
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--num_heads",    type=int,   default=4)
    parser.add_argument("--num_layers",   type=int,   default=4)
    parser.add_argument("--ff_dim",       type=int,   default=1024)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Windowing
    parser.add_argument("--window_size",              type=int, default=64)
    parser.add_argument("--stride",                   type=int, default=16)
    parser.add_argument(
        "--prediction_margin_frames",
        type=int,
        default=0,
        help="Deprecated for refine-q; all frames in each window are supervised.",
    )

    # Training
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--num_epochs",    type=int,   default=100)

    # Loss
    parser.add_argument(
        "--lambda_reg", type=float, default=0.01,
        help=(
            "Weight for the L2 regularisation term on delta_q. "
            "Penalises residual predictions that deviate from zero."
        ),
    )
    parser.add_argument(
        "--reg_fade_epochs", type=int, default=0,
        help=(
            "Linearly decay lambda_reg from its initial value to 0 over this many "
            "epochs.  0 = no fade (constant lambda_reg throughout training)."
        ),
    )
    parser.add_argument(
        "--pos_loss_weight", type=float, default=1.0,
        help=(
            "Relative weight on reconstruction loss. Effective objective is represented "
            "as recon + lambda_eff*reg where lambda_eff=lambda_reg*reg_loss_weight/pos_loss_weight."
        ),
    )
    parser.add_argument(
        "--reg_loss_weight", type=float, default=1.0,
        help="Relative weight on regularization loss (paired with --pos_loss_weight).",
    )

    # Validation / subject split
    parser.add_argument(
        "--val_subjects", nargs="*", default=[],
        help="Subject names to hold out for validation (LOSO-style).",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a .pkl checkpoint file to resume training from.",
    )

    args = parser.parse_args()

    rng        = jax.random.PRNGKey(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  Q-Position Residual Refinement -- train_refine_q.py", flush=True)
    print("=" * 70, flush=True)
    print(f"  JAX devices : {jax.devices()}", flush=True)
    print(f"  Output dir  : {output_dir}", flush=True)

    # -----------------------------------------------------------------------
    # 1. Discover trials
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Discovering trials in: {args.data_dir}", flush=True)
    all_trials = discover_trials(args.data_dir)
    print(f"  Found {len(all_trials)} refine-q trials.", flush=True)

    if not all_trials:
        print(
            "ERROR: No trials found. Each ProcessedData/ needs either (1) Loso_Combined cache: "
            "loso_combined_trial_info.json + pos_inputs.npy + pos_gt.npy (MoCap), or "
            "(2) standard layout: pos_inputs.npy (clean GT) + pos_inputs_noised.npy.",
            flush=True,
        )
        sys.exit(1)

    all_trials = [t for t in all_trials if t["length"] > args.window_size]
    print(
        f"  After length filter (>{args.window_size} frames): "
        f"{len(all_trials)} trials.",
        flush=True,
    )
    if not all_trials:
        print("ERROR: No trials pass the length filter.", flush=True)
        sys.exit(1)

    # Train / validation split by subject name
    val_subjects  = set(args.val_subjects)
    train_trials  = [t for t in all_trials if t["subject"] not in val_subjects]
    val_trials    = [t for t in all_trials if t["subject"] in val_subjects]

    if not train_trials:
        print("ERROR: No training trials after subject split.", flush=True)
        sys.exit(1)

    n_train_subj = len({t["subject"] for t in train_trials})
    print(f"  Train: {len(train_trials)} trials / {n_train_subj} subjects", flush=True)
    if val_trials:
        n_val_subj = len({t["subject"] for t in val_trials})
        print(f"  Val  : {len(val_trials)} trials / {n_val_subj} subjects", flush=True)
    else:
        print("  Val  : none (no --val_subjects specified)", flush=True)

    # -----------------------------------------------------------------------
    # 2. Build data loaders
    # -----------------------------------------------------------------------
    print("\n[2/5] Building data loaders ...", flush=True)
    loader_kwargs = dict(
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    train_loader = QRefineDataLoader(train_trials, shuffle=True,  **loader_kwargs)
    val_loader   = (
        QRefineDataLoader(val_trials, shuffle=False, **loader_kwargs)
        if val_trials else None
    )
    print(f"  Train windows : {train_loader.total_windows}", flush=True)
    if val_loader:
        print(f"  Val windows   : {val_loader.total_windows}", flush=True)

    # -----------------------------------------------------------------------
    # 3. Build model and training state
    # -----------------------------------------------------------------------
    print("\n[3/5] Building model ...", flush=True)
    model = QRefineTransformer(
        input_dim=INPUT_DIM,
        static_dim=STATIC_DIM,
        output_dim=OUTPUT_DIM,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout_rate=args.dropout_rate,
    )
    print("  QRefineTransformer", flush=True)
    print(
        f"    input_dim  = {INPUT_DIM}  (position-only, 16-D)",
        flush=True,
    )
    print(
        f"    output_dim = {OUTPUT_DIM}  "
        f"(denoised pos_inputs; pelvis Euler at cols 0:3)",
        flush=True,
    )
    print(
        f"    d_model={args.d_model}, layers={args.num_layers}, "
        f"heads={args.num_heads}, ff_dim={args.ff_dim}",
        flush=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.num_epochs
    warmup_steps    = max(1, min(500, total_steps // 20))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng, model,
        input_shape=(1, args.window_size, INPUT_DIM),
        static_shape=(1, STATIC_DIM),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    # Optional resume from checkpoint
    best_val_loss = float("inf")
    start_epoch   = 1
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            state         = state.replace(params=ckpt["params"])
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            start_epoch   = ckpt.get("epoch", 0) + 1
            print(
                f"  Resumed from {ckpt_path} (last epoch {start_epoch - 1}, "
                f"best_val={best_val_loss:.5f})",
                flush=True,
            )
        else:
            print(
                f"  Checkpoint not found: {ckpt_path}. Training from scratch.",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # 4. Compile step functions
    # -----------------------------------------------------------------------
    print("\n[4/5] Compiling JIT step functions ...", flush=True)
    train_step_fn = make_train_step(model)
    eval_step_fn  = make_eval_step(model)
    print("  Done (first epoch will be slower due to XLA compilation).", flush=True)

    # -----------------------------------------------------------------------
    # 5. Training loop
    # -----------------------------------------------------------------------
    print(
        f"\n[5/5] Training for {args.num_epochs} epochs "
        f"({steps_per_epoch} steps/epoch) ...",
        flush=True,
    )
    print(
        "  "
        f"lambda_reg={args.lambda_reg}, "
        f"pos_loss_weight={args.pos_loss_weight}, "
        f"reg_loss_weight={args.reg_loss_weight}, "
        f"reg_fade_epochs={args.reg_fade_epochs}",
        flush=True,
    )
    print(
        f"  LR={args.learning_rate}, warmup={warmup_steps} steps, "
        f"weight_decay={args.weight_decay}",
        flush=True,
    )
    print("=" * 70, flush=True)

    train_loss_history: List[float] = []
    val_loss_history:   List[float] = []
    pos_loss_weight = max(1e-8, float(args.pos_loss_weight))
    reg_loss_weight = max(0.0, float(args.reg_loss_weight))

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()

        # Compute effective regularisation weight (linear fade toward 0)
        if args.reg_fade_epochs > 0:
            fade_frac        = min(1.0, (epoch - 1) / args.reg_fade_epochs)
            base_lambda      = float(args.lambda_reg * (1.0 - fade_frac))
        else:
            base_lambda      = float(args.lambda_reg)
        effective_lambda = float(base_lambda * (reg_loss_weight / pos_loss_weight))

        # -- Train ------------------------------------------------------------
        train_acc   = _zero_metrics()
        train_count = 0

        for batch in train_loader:
            rng, dropout_rng = jax.random.split(rng)
            state, metrics   = train_step_fn(state, batch, dropout_rng, effective_lambda)
            for k in train_acc:
                train_acc[k] += float(metrics[k])
            train_count += 1

        train_avg = _avg_metrics(train_acc, train_count)
        train_loss_history.append(train_avg["total_loss"])

        # -- Validate ---------------------------------------------------------
        val_str     = ""
        current_val = train_avg["total_loss"]   # fallback when no val set

        if val_loader is not None:
            val_acc   = _zero_metrics()
            val_count = 0
            for batch in val_loader:
                metrics   = eval_step_fn(state, batch, effective_lambda)
                for k in val_acc:
                    val_acc[k] += float(metrics[k])
                val_count += 1
            val_avg     = _avg_metrics(val_acc, val_count)
            current_val = val_avg["total_loss"]
            val_loss_history.append(current_val)
            val_str = (
                f" | Val  total={val_avg['total_loss']:.5f}"
                f"  recon={val_avg['recon_loss']:.5f}"
                f"  reg={val_avg['reg_loss']:.6f}"
            )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:4d}/{args.num_epochs}  [{elapsed:5.1f}s]"
            f"  Train total={train_avg['total_loss']:.5f}"
            f"  recon={train_avg['recon_loss']:.5f}"
            f"  reg={train_avg['reg_loss']:.6f}"
            f"  lambda={effective_lambda:.4f}"
            f"{val_str}",
            flush=True,
        )

        # -- Save best model --------------------------------------------------
        if current_val < best_val_loss:
            best_val_loss = current_val
            best_path = output_dir / "best_model.pkl"
            _save_checkpoint(best_path, state, epoch, best_val_loss, args)
            print(
                f"  [*] New best (val={best_val_loss:.5f}) saved -> {best_path}",
                flush=True,
            )

        # -- Periodic checkpoint every 10 epochs ------------------------------
        if epoch % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch:04d}.pkl"
            _save_checkpoint(ckpt_path, state, epoch, best_val_loss, args)

    # -----------------------------------------------------------------------
    # Save final model and loss history
    # -----------------------------------------------------------------------
    final_path = output_dir / "final_model.pkl"
    _save_checkpoint(final_path, state, args.num_epochs, best_val_loss, args)
    print(f"\nTraining complete.  Final model -> {final_path}", flush=True)
    print(f"Best val loss: {best_val_loss:.5f}", flush=True)

    hist_path = output_dir / "loss_history.json"
    with open(hist_path, "w") as f:
        json.dump({"train": train_loss_history, "val": val_loss_history}, f, indent=2)
    print(f"Loss history   -> {hist_path}", flush=True)


if __name__ == "__main__":
    main()
