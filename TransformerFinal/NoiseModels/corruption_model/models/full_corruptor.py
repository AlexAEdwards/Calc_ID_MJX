from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from corruption_model.config import CorruptionConfig
from corruption_model.models.bias_model import BiasModel
from corruption_model.models.corruption_sampler import make_rng
from corruption_model.models.dropout_model import DropoutModel
from corruption_model.models.lag_model import LagModel
from corruption_model.models.lowrank_model import LowRankModel
from corruption_model.models.noise_model import NoiseModel
from corruption_model.models.phase_residual_model import PhaseResidualModel
from corruption_model.models.smoothing_model import SmoothingModel
from corruption_model.preprocess.align import estimate_global_lag, shift_with_interpolation
from corruption_model.preprocess.normalize import denormalize_by_height, normalize_by_height
from corruption_model.preprocess.symmetry import augment_with_left_right_mirrors
from corruption_model.residuals.compute_residuals import compute_residual_trial
from corruption_model.residuals.residual_dataset import ResidualDataset
from corruption_model.types import ResidualTrial, TrialPair


@dataclass
class FullCorruptor:
    config: CorruptionConfig
    bias_model: BiasModel = field(default_factory=BiasModel)
    lag_model: LagModel | None = None
    smoothing_model: SmoothingModel | None = None
    noise_model: NoiseModel | None = None
    phase_residual_model: PhaseResidualModel | None = None
    dropout_model: DropoutModel | None = None
    lowrank_model: LowRankModel | None = None
    fitted_: bool = False
    fit_summary_: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lag_model = self.lag_model or LagModel(
            max_frames=self.config.model.lag_max_frames,
            sample_scale=self.config.model.lag_std_scale,
        )
        self.smoothing_model = self.smoothing_model or SmoothingModel(
            fs_hz=self.config.representation.sample_rate_hz,
            filter_order=self.config.model.smoothing_filter_order,
            default_cutoff_hz=self.config.model.smoothing_cutoff_hz_default,
            sample_std_scale=self.config.model.smoothing_std_scale,
        )
        self.noise_model = self.noise_model or NoiseModel(
            minimum_variance=self.config.model.minimum_variance,
            sample_scale=self.config.model.noise_sample_scale,
            phase_window_frames=self.config.model.phase_window_frames,
        )
        self.phase_residual_model = self.phase_residual_model or PhaseResidualModel(
            enabled=self.config.model.use_phase_residual,
            sample_scale=self.config.model.phase_residual_sample_scale,
            gain_std=self.config.model.phase_residual_gain_std,
            phase_window_frames=self.config.model.phase_window_frames,
        )
        self.bias_model = self.bias_model or BiasModel(phase_window_frames=self.config.model.phase_window_frames)
        self.bias_model.phase_window_frames = self.config.model.phase_window_frames
        self.dropout_model = self.dropout_model or DropoutModel(enabled=self.config.model.use_dropout)
        self.lowrank_model = self.lowrank_model or LowRankModel(
            n_components=self.config.model.pca_components,
            explained_variance_threshold=self.config.model.explained_variance_threshold,
            sample_scale=self.config.model.lowrank_sample_scale,
            template_mix=self.config.model.lowrank_template_mix,
            template_gain_std=self.config.model.lowrank_template_gain_std,
            template_jitter_scale=self.config.model.lowrank_template_jitter_scale,
        )

    def fit(self, paired_trials: List[TrialPair]) -> "FullCorruptor":
        residual_trials: List[ResidualTrial] = []
        lag_frames = []
        for trial in paired_trials:
            alignment = estimate_global_lag(
                trial,
                sample_rate_hz=self.config.representation.sample_rate_hz,
                max_lag_frames=self.config.model.lag_max_frames,
            )
            residual_trials.append(
                compute_residual_trial(
                    trial=trial,
                    q_mocap_aligned=alignment.q_mocap_aligned,
                    q_opencap_aligned=alignment.q_opencap_aligned,
                    lag_frames=alignment.lag_frames,
                    lag_seconds=alignment.lag_seconds,
                    alignment_score=alignment.alignment_score,
                )
            )
            lag_frames.append(alignment.lag_frames)

        fit_trials = augment_with_left_right_mirrors(residual_trials)

        self.bias_model.fit(fit_trials)
        fit_phase_positions = [
            trial.phase_positions if self.config.model.use_phase_conditioning else None
            for trial in fit_trials
        ]
        bias_sequences = [self.bias_model.predict(phase_positions) for phase_positions in fit_phase_positions]
        global_bias = np.asarray(self.bias_model.global_bias, dtype=np.float32)
        global_bias_sequences = [
            np.repeat(global_bias[np.newaxis, :], trial.residual.shape[0], axis=0).astype(np.float32)
            for trial in fit_trials
        ]
        self.phase_residual_model.fit(fit_trials, global_bias_sequences)
        lowrank_reconstructions = None
        if self.config.model.use_lowrank:
            self.lowrank_model.fit(fit_trials, bias_sequences)
            lowrank_reconstructions = [
                self.lowrank_model.project((trial.residual - bias).astype(np.float32))[0]
                for trial, bias in zip(fit_trials, bias_sequences)
            ]
        self.noise_model.fit(
            fit_trials,
            deterministic_components=bias_sequences if bias_sequences else None,
            lowrank_components=lowrank_reconstructions,
        )
        if self.config.model.use_smoothing:
            self.smoothing_model.fit(fit_trials)
        self.lag_model.fit(lag_frames)
        self.dropout_model.fit([trial.mask_valid for trial in fit_trials if trial.mask_valid is not None])
        self.fitted_ = True
        dataset = ResidualDataset(residual_trials)
        self.fit_summary_ = {
            "num_trials": len(residual_trials),
            "num_fit_trials_after_left_right_pooling": len(fit_trials),
            "num_subjects": len(dataset.by_subject()),
            "lag_mean_frames": self.lag_model.mean_frames,
            "lag_std_frames": self.lag_model.std_frames,
            "use_lag": bool(self.config.model.use_lag),
            "lag_sample_scale": self.lag_model.sample_scale,
            "use_phase_conditioning": bool(self.config.model.use_phase_conditioning),
            "use_phase_residual": bool(self.phase_residual_model.enabled),
            "phase_window_frames": self.config.model.phase_window_frames,
            "phase_residual_sample_scale": self.phase_residual_model.sample_scale,
            "phase_residual_gain_std": self.phase_residual_model.gain_std,
            "use_smoothing": bool(self.config.model.use_smoothing),
            "input_cutoff_mean_hz": self.smoothing_model.input_cutoff_mean_hz,
            "input_cutoff_std_hz": self.smoothing_model.input_cutoff_std_hz,
            "smoothing_std_scale": self.smoothing_model.sample_std_scale,
            "use_lowrank": bool(self.config.model.use_lowrank),
            "lowrank_sample_scale": self.lowrank_model.sample_scale,
            "lowrank_template_mix": self.lowrank_model.template_mix,
            "lowrank_template_gain_std": self.lowrank_model.template_gain_std,
            "lowrank_template_jitter_scale": self.lowrank_model.template_jitter_scale,
            "noise_sample_scale": self.noise_model.sample_scale,
            "explained_variance_ratio": self.lowrank_model.explained_variance_ratio.tolist() if self.lowrank_model.explained_variance_ratio is not None else [],
        }
        return self

    def sample(self, q_clean: np.ndarray, activity: str = "walking", meta: Dict[str, Any] | None = None, random_state: int | np.random.Generator | None = None) -> tuple[np.ndarray, Dict[str, Any]]:
        if not self.fitted_:
            raise RuntimeError("FullCorruptor must be fit before sample().")
        rng = make_rng(random_state)
        q = np.asarray(q_clean, dtype=np.float32).copy()
        height_m = None if not meta else meta.get("height_m")
        if self.config.representation.normalize_anthropometrics:
            q = normalize_by_height(q, height_m)
        filter_params = {"output_cutoff_hz": None, "filter_order": None, "enabled": False}

        if meta and meta.get("phase_positions") is not None:
            phase_positions = np.asarray(meta["phase_positions"], dtype=np.int32)
        elif meta and meta.get("phase_bins") is not None:
            phase_positions = np.asarray(meta["phase_bins"], dtype=np.int32)
        else:
            phase_positions = np.floor(np.linspace(0.0, 200.0, q.shape[0], endpoint=False, dtype=np.float32)).astype(np.int32)
        phase_positions_for_models = phase_positions if self.config.model.use_phase_conditioning else None
        bias = self.bias_model.predict(phase_positions_for_models)
        q = q + bias

        phase_residual = self.phase_residual_model.sample(phase_positions=phase_positions, rng=rng)
        if phase_residual.ndim == 2 and phase_residual.shape[1] == q.shape[1]:
            q = q + phase_residual

        if self.config.model.use_lowrank:
            r_corr, alpha = self.lowrank_model.sample(T=q.shape[0], rng=rng)
            q = q + r_corr
        else:
            alpha = np.zeros((q.shape[0], 0), dtype=np.float32)

        eps = self.noise_model.sample(
            phase_positions=phase_positions_for_models if phase_positions_for_models is not None else np.zeros((q.shape[0],), dtype=np.int32),
            rng=rng,
        )
        q = q + eps

        if self.config.model.use_lag:
            sampled_lag = self.lag_model.sample(rng)
            q = shift_with_interpolation(q, sampled_lag)
        else:
            sampled_lag = 0.0

        q, dropout_mask = self.dropout_model.apply(q, rng)
        if self.config.representation.normalize_anthropometrics:
            q = denormalize_by_height(q, height_m)
        if self.config.model.use_smoothing:
            q, filter_params = self.smoothing_model.apply_input(q, rng)
        aux = {
            "activity": activity,
            "sampled_lag": float(sampled_lag),
            "sampled_filter_params": filter_params,
            "dropout_mask": dropout_mask.astype(np.int8),
            "phase_positions": phase_positions,
            "phase_residual_used": bool(self.phase_residual_model.enabled),
            "alpha_shape": list(alpha.shape),
        }
        return q.astype(np.float32), aux

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: str | Path) -> "FullCorruptor":
        with Path(path).open("rb") as f:
            return pickle.load(f)
