from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trial_overlay(clean: np.ndarray, real: np.ndarray, synthetic: np.ndarray, output_path: str | Path, dof_idx: int = 0) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(clean[:, dof_idx], label="clean_mocap", linewidth=1.5)
    plt.plot(real[:, dof_idx], label="real_opencap", linewidth=1.2)
    plt.plot(synthetic[:, dof_idx], label="synthetic_opencap", linewidth=1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_noised_curves_against_gt(
    clean: np.ndarray,
    synthetic_curves: list[np.ndarray],
    dof_names: list[str],
    output_path: str | Path,
    max_dofs: int = 8,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if clean.ndim != 2:
        raise ValueError(f"Expected clean array with shape [T, D], got {clean.shape}")
    selected_dofs = min(int(max_dofs), clean.shape[1])
    fig, axes = plt.subplots(selected_dofs, 1, figsize=(12, max(2.5 * selected_dofs, 4)), sharex=True)
    if selected_dofs == 1:
        axes = [axes]
    for dof_idx in range(selected_dofs):
        ax = axes[dof_idx]
        ax.plot(clean[:, dof_idx], label="GT clean pos", color="#111827", linewidth=1.8)
        for curve_idx, curve in enumerate(synthetic_curves, start=1):
            ax.plot(curve[:, dof_idx], linewidth=1.0, alpha=0.75, label=f"Noised {curve_idx}" if dof_idx == 0 else None)
        dof_label = dof_names[dof_idx] if dof_idx < len(dof_names) else f"dof_{dof_idx}"
        ax.set_ylabel(dof_label)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_residual_stats_comparison(
    real_stats: dict,
    synthetic_stats: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_joint_stats = real_stats["joint_stats"]
    synthetic_joint_stats = synthetic_stats["joint_stats"]
    joint_names = [row["joint_name"] for row in real_joint_stats]
    real_means = np.asarray([row["mean"] for row in real_joint_stats], dtype=np.float32)
    real_stds = np.asarray([row["std"] for row in real_joint_stats], dtype=np.float32)
    synthetic_means = np.asarray([row["mean"] for row in synthetic_joint_stats], dtype=np.float32)
    synthetic_stds = np.asarray([row["std"] for row in synthetic_joint_stats], dtype=np.float32)

    x = np.arange(len(joint_names), dtype=np.float32)
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].bar(x - width / 2, real_means, width=width, label="OpenCap residual mean", color="#2563eb")
    axes[0].bar(x + width / 2, synthetic_means, width=width, label="Synthetic residual mean", color="#f97316")
    axes[0].set_ylabel("Mean residual")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x - width / 2, real_stds, width=width, label="OpenCap residual std", color="#2563eb")
    axes[1].bar(x + width / 2, synthetic_stds, width=width, label="Synthetic residual std", color="#f97316")
    axes[1].set_ylabel("Std residual")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(joint_names, rotation=75, ha="right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_residual_stats_boxplot_comparison(
    real_trialwise_stats: dict,
    synthetic_trialwise_stats: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_trials = real_trialwise_stats["trial_stats"]
    synthetic_trials = synthetic_trialwise_stats["trial_stats"]
    if not real_trials or not synthetic_trials:
        raise ValueError("Both real and synthetic trialwise stats must be non-empty.")

    joint_names = [row["joint_name"] for row in real_trials[0]["joint_stats"]]
    x = np.arange(len(joint_names), dtype=np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    def _extract(metric_name: str, trials: list[dict]) -> list[list[float]]:
        values = [[] for _ in range(len(joint_names))]
        for trial in trials:
            for joint_idx, row in enumerate(trial["joint_stats"]):
                values[joint_idx].append(float(row[metric_name]))
        return values

    real_means = _extract("mean", real_trials)
    syn_means = _extract("mean", synthetic_trials)
    real_stds = _extract("std", real_trials)
    syn_stds = _extract("std", synthetic_trials)

    def _draw_box_with_points(ax, left_values, right_values, ylabel: str) -> None:
        left_pos = x - 0.18
        right_pos = x + 0.18
        ax.boxplot(
            left_values,
            positions=left_pos,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#bfdbfe", color="#2563eb"),
            medianprops=dict(color="#1d4ed8", linewidth=1.5),
            whiskerprops=dict(color="#2563eb"),
            capprops=dict(color="#2563eb"),
        )
        ax.boxplot(
            right_values,
            positions=right_pos,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#fed7aa", color="#f97316"),
            medianprops=dict(color="#c2410c", linewidth=1.5),
            whiskerprops=dict(color="#f97316"),
            capprops=dict(color="#f97316"),
        )
        rng = np.random.default_rng(123)
        for joint_idx, values in enumerate(left_values):
            jitter = rng.uniform(-0.04, 0.04, size=len(values))
            ax.scatter(np.full((len(values),), left_pos[joint_idx]) + jitter, values, s=10, alpha=0.35, color="#1d4ed8")
        for joint_idx, values in enumerate(right_values):
            jitter = rng.uniform(-0.04, 0.04, size=len(values))
            ax.scatter(np.full((len(values),), right_pos[joint_idx]) + jitter, values, s=10, alpha=0.35, color="#c2410c")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)

    _draw_box_with_points(axes[0], real_means, syn_means, "Residual mean")
    _draw_box_with_points(axes[1], real_stds, syn_stds, "Residual std")
    axes[0].plot([], [], color="#2563eb", label="OpenCap")
    axes[0].plot([], [], color="#f97316", label="Synthetic")
    axes[0].legend(loc="upper right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(joint_names, rotation=75, ha="right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_psd_comparison(
    real_psd_stats: dict,
    synthetic_psd_stats: dict,
    output_path: str | Path,
    max_dofs: int = 8,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    freqs_real = np.asarray(real_psd_stats["freqs_hz"], dtype=np.float32)
    freqs_syn = np.asarray(synthetic_psd_stats["freqs_hz"], dtype=np.float32)
    if freqs_real.size == 0 or freqs_syn.size == 0:
        raise ValueError("Real and synthetic PSD frequencies must both be non-empty.")

    # Dry runs may aggregate trials with different sequence lengths, which
    # produces different Welch frequency grids. Compare both spectra on the
    # shared frequency support rather than requiring identical bins.
    common_min = float(max(freqs_real.min(), freqs_syn.min()))
    common_max = float(min(freqs_real.max(), freqs_syn.max()))
    if common_max <= common_min:
        raise ValueError("Real and synthetic PSD frequency ranges do not overlap.")

    common_freqs = np.linspace(common_min, common_max, num=min(len(freqs_real), len(freqs_syn)), dtype=np.float32)

    def _interp_psd(freqs_src: np.ndarray, psd_src: np.ndarray, freqs_dst: np.ndarray) -> np.ndarray:
        return np.interp(freqs_dst, freqs_src, psd_src).astype(np.float32)

    real_joint_psd = real_psd_stats["joint_psd"]
    synthetic_joint_psd = synthetic_psd_stats["joint_psd"]
    selected_dofs = min(int(max_dofs), len(real_joint_psd))
    fig, axes = plt.subplots(selected_dofs, 1, figsize=(12, max(2.6 * selected_dofs, 4)), sharex=True)
    if selected_dofs == 1:
        axes = [axes]

    for dof_idx in range(selected_dofs):
        ax = axes[dof_idx]
        joint_name = real_joint_psd[dof_idx]["joint_name"]
        real_psd = _interp_psd(freqs_real, np.asarray(real_joint_psd[dof_idx]["psd"], dtype=np.float32), common_freqs)
        syn_psd = _interp_psd(freqs_syn, np.asarray(synthetic_joint_psd[dof_idx]["psd"], dtype=np.float32), common_freqs)
        ax.plot(common_freqs, real_psd, label="OpenCap residual PSD", color="#2563eb", linewidth=1.6)
        ax.plot(common_freqs, syn_psd, label="Synthetic residual PSD", color="#f97316", linewidth=1.4)
        ax.set_ylabel(joint_name)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
