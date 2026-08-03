"""Transformer building blocks and the COP/GRF/moment model.

Extracted verbatim from train.py in REFACTOR_PLAN.md Stage 5.
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn

from core.constants import (
    CONTACT_SLICE, COP_SLICE, GRF_SLICE, MOMENTS_SLICE, STANDARD_OUTPUT_DIM,
)




class SinusoidalPosEmb(nn.Module):
    dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        position = jnp.arange(seq_len)
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = position[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return x + emb[None, :, :]


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True,
                 film_gamma: jnp.ndarray = None, film_beta: jnp.ndarray = None) -> jnp.ndarray:
        # Optional FiLM subject conditioning: modulate the normalized features with
        # per-layer (gamma, beta) derived from the static token. Uses (1 + gamma) so it
        # is near-identity at init. When film_gamma is None the block is unchanged.
        def _film(h):
            if film_gamma is None:
                return h
            return h * (1.0 + film_gamma[:, None, :]) + film_beta[:, None, :]

        residual = x
        x = nn.LayerNorm()(x)
        x = _film(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x, x, deterministic=not train)
        x = residual + attn_out

        residual = x
        x = nn.LayerNorm()(x)
        x = _film(x)
        ff_out = nn.Dense(self.ff_dim)(x)
        ff_out = nn.gelu(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
        ff_out = nn.Dense(self.d_model)(ff_out)
        ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
        x = residual + ff_out

        return x


class KinematicsToCOPGRFMoments(nn.Module):
    """Transformer-based model for gait dynamics prediction.
    
    Inputs:
        - Temporal feature vector (constructed in data_loader.py)
          Includes kinematics, reconstructed COP/GRF/GRM, and optional flattened Jacobians.
          NOTE: contactBoolean is NO LONGER an input; the model predicts it as output.
        - Static token:
          [height, mass, gender, PatientSize(4), forwardVel]
    
    Outputs:
        - COP (4): [rx, rz, lx, lz] in ground-aligned calc frame - Unit: m/h  (contact-masked)
        - GRF (6): [rx, ry, rz, lx, ly, lz] - Unit: N/m*9.806                 (contact-masked)
        - Moments (2): [rz, lz] - Unit: Nm/m*h*9.806
        - ContactBoolean (2): [right, left] - soft sigmoid, hard-thresholded for masking
    """
    input_dim: int = 54
    static_dim: int = 8 # height, mass, gender, PatientSize(4), forwardVel
    output_dim: int = STANDARD_OUTPUT_DIM
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout_rate: float = 0.1
    use_film: bool = False  # Plan 7: per-layer FiLM subject conditioning (default off)

    @nn.compact
    def __call__(self, x: jnp.ndarray, static_context: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # 1. Project temporal inputs directly into the transformer width.
        x = nn.Dense(self.d_model)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        # Positional Encoding
        x = SinusoidalPosEmb(dim=self.d_model)(x)

        # 2. Static Branch: MLP Layer
        s = nn.Dense(self.d_model)(static_context)
        s = nn.gelu(s)
        s = nn.LayerNorm()(s)

        # 2b. Optional FiLM conditioning params (per layer, gamma+beta of width d_model).
        film_params = None
        if self.use_film:
            film_params = nn.Dense(self.num_layers * 2 * self.d_model, name="film_mlp")(s)
            film_params = film_params.reshape(s.shape[0], self.num_layers, 2, self.d_model)

        # 3. Prepend Static Token
        s = jnp.expand_dims(s, axis=1)
        x = jnp.concatenate([s, x], axis=1)  # (batch, seq_len + 1, d_model)

        for _layer_idx in range(self.num_layers):
            film_gamma = film_params[:, _layer_idx, 0, :] if film_params is not None else None
            film_beta = film_params[:, _layer_idx, 1, :] if film_params is not None else None
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train, film_gamma=film_gamma, film_beta=film_beta)

        # 4. Remove Static Token
        x = x[:, 1:, :]
        x = nn.LayerNorm()(x)

        # 5. Predict the standard 14 outputs from the shared backbone.
        raw_out = nn.Dense(self.output_dim)(x)  # (batch, seq, 14)

        # 6. Predict contact probabilities (sigmoid)
        contact_logits = raw_out[..., CONTACT_SLICE]          # (batch, seq, 2) — right, left
        contact_prob   = nn.sigmoid(contact_logits)   # soft, used for BCE loss

        cop_raw = raw_out[..., COP_SLICE]    # [rx, rz, lx, lz]
        grf_raw = raw_out[..., GRF_SLICE]   # [rx, ry, rz, lx, ly, lz]
        mom_raw = raw_out[..., MOMENTS_SLICE]  # [rz, lz]

        # 8. Concatenate final raw (normalized) output: COP(4) + GRF(6) + Moments(2) + ContactProb(2)
        #    Contact hard-masking is now applied in physical space within compute_total_loss
        out = jnp.concatenate([cop_raw, grf_raw, mom_raw, contact_prob], axis=-1)
        return out
