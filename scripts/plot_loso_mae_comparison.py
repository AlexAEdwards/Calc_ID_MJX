#!/usr/bin/env python3
"""Grouped bar chart comparing per-DOF average MAE across LOSO model variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paths import artifact, dataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OPENCAP_RUN = (
    REPO_ROOT
    / str(artifact("inference_results", "Loso_AvgHPO_MJXGT_OriginalWeightsModel_ExcludeSub5"))
)
DEFAULT_MOCAP_RUN = (
    REPO_ROOT
    / str(artifact("inference_results", "Loso_HPO16h_final_trial0103_ep10_lr3e5_Sub5Excluded_MOCAPFINETUNED_MJXGT"))
)
DEFAULT_OUT_DIR = REPO_ROOT / str(artifact("inference_results", "loso_mae_comparison"))

# Internal key -> display label (anatomical order)
DOF_ORDER: List[Tuple[str, str]] = [
    ("hip_flexion_l", "Hip flexion L"),
    ("hip_adduction_l", "Hip adduction L"),
    ("hip_rotation_l", "Hip rotation L"),
    ("knee_angle_l", "Knee flexion L"),
    ("knee_adduction_moment_l", "Knee adduction L"),
    ("ankle_angle_l", "Ankle dorsiflexion L"),
    ("subtalar_angle_l", "Subtalar supination L"),
    ("lumbar_extension", "Lumbar extension"),
    ("lumbar_bending", "Lumbar bending"),
    ("lumbar_rotation", "Lumbar rotation"),
]

MODEL_SPECS: List[Tuple[str, str, str]] = [
    ("OpenCap Fine Tuned", "opencap", "average_mae_per_dof_fine_tuned_opencap_input"),
    ("OpenCap Original", "opencap", "average_mae_per_dof_original_opencap_predinput"),
    ("MoCap Fine Tuned", "mocap", "average_mae_per_dof"),
    ("MoCap Original", "mocap", "average_mae_per_dof_motioncapture_input"),
    ("Hybrid", "hybrid", "hybrid"),
]

MODEL_COLORS = [
    "#2E86AB",
    "#6BAED6",
    "#E94F37",
    "#FCAE91",
    "#5B8C5A",
]

# Per-DOF Hybrid MAE (% BW×H), same order as DOF_ORDER
HYBRID_MAE: Dict[str, float] = {
    "hip_flexion_l": 0.96,
    "hip_adduction_l": 0.67,
    "hip_rotation_l": 0.26,
    "knee_angle_l": 0.91,
    "knee_adduction_moment_l": 0.41,
    "ankle_angle_l": 0.57,
    "subtalar_angle_l": 0.36,
    "lumbar_extension": 0.74,
    "lumbar_bending": 0.39,
    "lumbar_rotation": 0.18,
}


def _load_aggregate_sections(report_path: Path) -> Dict[str, Dict[str, float]]:
    """Load only top-level aggregate MAE dicts (skip large trial_details)."""
    text = report_path.read_text()
    marker = '"trial_details"'
    if marker in text:
        text = text[: text.index(marker)].rstrip()
        if text.endswith(","):
            text = text[:-1]
        text += "\n}"

    payload = json.loads(text)
    return {
        key: value
        for key, value in payload.items()
        if key.startswith("average_mae_per_dof") and isinstance(value, dict)
    }


def _get_mae(
    sections: Dict[str, Dict[str, float]],
    section_key: str,
    dof_key: str,
) -> Optional[float]:
    values = sections.get(section_key, {})
    if dof_key not in values:
        return None
    return float(values[dof_key])


def build_table(
    opencap_sections: Dict[str, Dict[str, float]],
    mocap_sections: Dict[str, Dict[str, float]],
    hybrid_mae: Dict[str, float],
) -> Tuple[List[str], List[str], np.ndarray]:
    dof_keys = [k for k, _ in DOF_ORDER]
    dof_labels = [label for _, label in DOF_ORDER]
    model_labels = [spec[0] for spec in MODEL_SPECS]

    data = np.full((len(dof_keys), len(MODEL_SPECS)), np.nan, dtype=float)

    for model_idx, (_label, source, section_key) in enumerate(MODEL_SPECS):
        sections = (
            hybrid_mae if source == "hybrid"
            else opencap_sections if source == "opencap"
            else mocap_sections
        )
        for dof_idx, dof_key in enumerate(dof_keys):
            if source == "hybrid":
                data[dof_idx, model_idx] = hybrid_mae[dof_key]
            else:
                value = _get_mae(sections, section_key, dof_key)
                if value is not None:
                    data[dof_idx, model_idx] = value

    return dof_labels, model_labels, data


def write_csv(
    out_path: Path,
    dof_labels: List[str],
    model_labels: List[str],
    data: np.ndarray,
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DOF", *model_labels])
        for dof_label, row in zip(dof_labels, data):
            writer.writerow([dof_label, *[f"{v:.4f}" if np.isfinite(v) else "" for v in row]])


def plot_grouped_bars(
    out_path: Path,
    dof_labels: List[str],
    model_labels: List[str],
    data: np.ndarray,
    title: str,
) -> None:
    n_dofs = len(dof_labels)
    n_models = len(model_labels)
    x = np.arange(n_dofs)
    bar_width = 0.14
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(max(14, n_dofs * 1.2), 7))
    for model_idx, (label, color, offset) in enumerate(
        zip(model_labels, MODEL_COLORS, offsets)
    ):
        values = data[:, model_idx]
        bars = ax.bar(x + offset, values, bar_width, label=label, color=color)
        for bar, value in zip(bars, values):
            if np.isfinite(value) and value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(dof_labels, rotation=35, ha="right")
    ax.set_ylabel("Average MAE (% BW×H)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(np.nanmax(data) * 1.28, 0.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencap-run", type=Path, default=DEFAULT_OPENCAP_RUN)
    parser.add_argument("--mocap-run", type=Path, default=DEFAULT_MOCAP_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    opencap_report = args.opencap_run / "overall_mae_report.json"
    mocap_report = args.mocap_run / "overall_mae_report.json"
    if not opencap_report.exists():
        raise FileNotFoundError(opencap_report)
    if not mocap_report.exists():
        raise FileNotFoundError(mocap_report)

    opencap_sections = _load_aggregate_sections(opencap_report)
    mocap_sections = _load_aggregate_sections(mocap_report)

    dof_labels, model_labels, data = build_table(
        opencap_sections,
        mocap_sections,
        hybrid_mae=HYBRID_MAE,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "loso_mae_comparison.csv"
    png_path = args.out_dir / "loso_mae_comparison_grouped_bar.png"

    write_csv(csv_path, dof_labels, model_labels, data)
    plot_grouped_bars(
        png_path,
        dof_labels,
        model_labels,
        data,
        title="Per-DOF Average MAE by Model (LOSO)",
    )

    print(f"Wrote:\n  {csv_path}\n  {png_path}")


if __name__ == "__main__":
    main()
