from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from corruption_model.types import ResidualTrial


@dataclass
class LowRankModel:
    n_components: int = 6
    explained_variance_threshold: float = 0.9
    sample_scale: float = 1.0
    template_mix: float = 0.8
    template_gain_std: float = 0.1
    template_jitter_scale: float = 0.35
    basis: Optional[np.ndarray] = None
    mean_alpha: Optional[np.ndarray] = None
    alpha_std: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None
    alpha_templates: list[np.ndarray] | None = None

    def fit(self, residual_trials: list[ResidualTrial], bias_sequences: list[np.ndarray]) -> "LowRankModel":
        centered_trials = []
        for trial, bias in zip(residual_trials, bias_sequences):
            centered_trials.append((trial.residual - bias).astype(np.float32))
        residual_matrix = np.concatenate(centered_trials, axis=0)
        _, singular_values, vh = np.linalg.svd(residual_matrix, full_matrices=False)
        explained = (singular_values ** 2) / np.maximum(np.sum(singular_values ** 2), 1e-8)
        cumulative = np.cumsum(explained)
        stable_rank = int(np.searchsorted(cumulative, self.explained_variance_threshold) + 1)
        k = max(1, min(self.n_components, stable_rank, vh.shape[0]))
        self.basis = vh[:k].T.astype(np.float32)
        self.explained_variance_ratio = explained[:k].astype(np.float32)

        alpha_sequences = [centered @ self.basis for centered in centered_trials]
        self.alpha_templates = [alpha.astype(np.float32) for alpha in alpha_sequences]
        alpha_stack = np.concatenate(alpha_sequences, axis=0)
        self.mean_alpha = np.mean(alpha_stack, axis=0).astype(np.float32)
        self.alpha_std = np.sqrt(np.maximum(np.var(alpha_stack, axis=0), 1e-6)).astype(np.float32)

        rho_values = np.zeros((k,), dtype=np.float32)
        for comp_idx in range(k):
            numerators = []
            denominators = []
            for alpha in alpha_sequences:
                if alpha.shape[0] < 2:
                    continue
                x_prev = alpha[:-1, comp_idx]
                x_next = alpha[1:, comp_idx]
                numerators.append(np.sum(x_prev * x_next))
                denominators.append(np.sum(x_prev ** 2))
            denom = float(np.sum(denominators))
            rho_values[comp_idx] = float(np.clip(np.sum(numerators) / denom, -0.99, 0.99)) if denom > 1e-8 else 0.0
        self.rho = rho_values
        return self

    def _resample_alpha_template(self, alpha_template: np.ndarray, target_length: int) -> np.ndarray:
        template = np.asarray(alpha_template, dtype=np.float32)
        if template.shape[0] == target_length:
            return template.astype(np.float32)
        if template.shape[0] <= 1:
            return np.repeat(template.astype(np.float32), target_length, axis=0)
        source_x = np.linspace(0.0, 1.0, template.shape[0], dtype=np.float32)
        target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
        out = np.empty((target_length, template.shape[1]), dtype=np.float32)
        for comp_idx in range(template.shape[1]):
            out[:, comp_idx] = np.interp(
                target_x,
                source_x,
                template[:, comp_idx],
                left=template[0, comp_idx],
                right=template[-1, comp_idx],
            ).astype(np.float32)
        return out

    def _sample_ar1_alpha(self, T: int, rng: np.random.Generator) -> np.ndarray:
        if self.mean_alpha is None or self.alpha_std is None or self.rho is None:
            raise RuntimeError("LowRankModel must be fit before sample().")
        k = self.mean_alpha.shape[0]
        alpha = np.zeros((T, k), dtype=np.float32)
        scaled_alpha_std = self.alpha_std * float(self.sample_scale)
        alpha[0] = rng.normal(self.mean_alpha, scaled_alpha_std).astype(np.float32)
        innovation_std = scaled_alpha_std * np.sqrt(np.maximum(1.0 - (self.rho ** 2), 1e-6))
        for t in range(1, T):
            alpha[t] = self.rho * alpha[t - 1] + rng.normal(0.0, innovation_std).astype(np.float32)
        return alpha.astype(np.float32)

    def sample(self, T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.basis is None or self.mean_alpha is None or self.alpha_std is None or self.rho is None:
            raise RuntimeError("LowRankModel must be fit before sample().")
        alpha_ar = self._sample_ar1_alpha(T=T, rng=rng)
        if self.alpha_templates:
            template_idx = int(rng.integers(0, len(self.alpha_templates)))
            template_alpha = self._resample_alpha_template(self.alpha_templates[template_idx], T)
            template_gain = float(np.clip(rng.normal(1.0, self.template_gain_std), 0.0, 2.0))
            template_alpha = template_gain * float(self.sample_scale) * template_alpha
            alpha = (
                (float(self.template_mix) * template_alpha)
                + ((1.0 - float(self.template_mix)) * alpha_ar)
            ).astype(np.float32)
            if self.template_jitter_scale > 0.0:
                jitter_std = self.alpha_std * float(self.sample_scale) * float(self.template_jitter_scale)
                alpha = (alpha + rng.normal(0.0, jitter_std, size=alpha.shape).astype(np.float32)).astype(np.float32)
        else:
            alpha = alpha_ar
        return (alpha @ self.basis.T).astype(np.float32), alpha.astype(np.float32)

    def project(self, centered_residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.basis is None:
            raise RuntimeError("LowRankModel must be fit before project().")
        centered = np.asarray(centered_residual, dtype=np.float32)
        alpha = centered @ self.basis
        reconstruction = alpha @ self.basis.T
        return reconstruction.astype(np.float32), alpha.astype(np.float32)
