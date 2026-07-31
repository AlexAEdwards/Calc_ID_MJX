#!/usr/bin/env python3
"""Batch OpenSim inverse dynamics over subject/trial folders."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Maps OpenSim coordinate name → column index in ID_GT_MJX.npy (MJX qpos order).
# Rotational DOFs only — translations are forces, not torques, so we skip them.
_OPENSIM_TO_MJX_IDX: dict[str, int] = {
    "hip_flexion_r": 6,
    "hip_adduction_r": 7,
    "hip_rotation_r": 8,
    "knee_angle_r": 11,
    "ankle_angle_r": 14,
    "subtalar_angle_r": 15,
    "mtp_angle_r": 16,
    "hip_flexion_l": 17,
    "hip_adduction_l": 18,
    "hip_rotation_l": 19,
    "knee_angle_l": 22,
    "ankle_angle_l": 25,
    "subtalar_angle_l": 26,
    "mtp_angle_l": 27,
    "lumbar_extension": 28,
    "lumbar_bending": 29,
    "lumbar_rotation": 30,
}

from generate_opensim_id_inputs import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_LEFT_BODY,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR_NAME,
    DEFAULT_RIGHT_BODY,
    SOURCE_MOTION,
    SOURCE_PROCESSED,
    TrialPaths,
    discover_trials,
    generate_trial_inputs,
)


def import_opensim():
    try:
        import opensim as osim  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the OpenSim Python API. Install/activate OpenSim's "
            "Python bindings in this environment before running ID."
        ) from exc
    return osim


def expected_id_output(paths: TrialPaths) -> Path:
    return paths.output_dir / "inverse_dynamics.sto"


def find_id_outputs(paths: TrialPaths) -> list[Path]:
    if not paths.output_dir.exists():
        return []
    candidates = []
    for pattern in ("*inverse*dynamics*.sto", "*inverse*dynamics*.mot", "*InverseDynamics*.sto", "*InverseDynamics*.mot"):
        candidates.extend(paths.output_dir.glob(pattern))
    expected = expected_id_output(paths)
    if expected.exists():
        candidates.append(expected)
    return sorted(set(candidates))


def primary_id_output(paths: TrialPaths, outputs: list[Path]) -> Path | None:
    expected = expected_id_output(paths)
    if expected.exists():
        return expected
    if outputs:
        return outputs[0]
    return None


def existing_generation_matches(paths: TrialPaths, *, source: str, use_noised: bool) -> bool:
    manifest_path = paths.output_dir / "input_generation_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    if manifest.get("source") != source or bool(manifest.get("use_noised")) != bool(use_noised):
        return False
    if not _HAS_NUMPY:
        return True

    suffix = "_noised" if use_noised else ""
    if source == SOURCE_PROCESSED:
        current_pos = paths.processed_dir / f"pos_mjx{suffix}.npy"
    elif source == SOURCE_MOTION:
        current_pos = paths.motion_dir / f"Pos{suffix}.npy"
    else:
        return False
    if not current_pos.exists():
        return False
    try:
        current_shape = list(np.load(current_pos, mmap_mode="r").shape)
    except Exception:
        return False

    manifest_pos_shape = (manifest.get("shapes") or {}).get("pos")
    if source == SOURCE_PROCESSED and manifest_pos_shape:
        # Processed OpenSim inputs convert 31-channel qpos to 23 coordinate columns.
        manifest_pos_shape = [manifest_pos_shape[0], current_shape[1] if len(current_shape) > 1 else None]
    return list(manifest_pos_shape or []) == current_shape


def run_inverse_dynamics(setup_file: Path) -> None:
    osim = import_opensim()
    tool = osim.InverseDynamicsTool(str(setup_file))
    tool.run()


def read_storage_file(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(errors="replace").splitlines()
    header_end = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    columns = lines[header_end + 1].split()
    rows = [[float(token) for token in line.split()] for line in lines[header_end + 2 :] if line.strip()]
    return columns, rows


def load_mjx_id_torques(trial_dir: Path, n_frames: int) -> dict[str, list[float]] | None:
    """Load ID_GT_MJX.npy and return {opensim_coord_name: [torque_per_frame]}."""
    if not _HAS_NUMPY:
        return None
    path = trial_dir / "ProcessedData" / "ID_GT_MJX.npy"
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] <= max(_OPENSIM_TO_MJX_IDX.values()):
        return None
    # Align frame count: use the shorter of the two so indices always match.
    n = min(n_frames, data.shape[0])
    result: dict[str, list[float]] = {}
    for coord, col in _OPENSIM_TO_MJX_IDX.items():
        result[coord] = data[:n, col].tolist()
    return result


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4g}"


def _svg_multiline(
    series: list[tuple[list[tuple[float, float]], str, str]],
    width: int,
    height: int,
) -> str:
    """Render multiple (points, css_class, label) series onto one SVG.

    *series* is a list of (points, css_class, label) where points is
    [(x, y), ...].  All series share the same axes so they're comparable.
    """
    all_xs: list[float] = []
    all_ys: list[float] = []
    for points, _, _ in series:
        for x, y in points:
            if math.isfinite(x) and math.isfinite(y):
                all_xs.append(x)
                all_ys.append(y)
    if len(all_xs) < 2:
        return ""
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 1.0)
        y_min -= pad
        y_max += pad
    pad_left, pad_right, pad_top, pad_bottom = 58, 16, 14, 34
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def sx(x: float) -> float:
        return pad_left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return pad_top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_h

    zero_line = ""
    if y_min <= 0.0 <= y_max:
        zy = sy(0.0)
        zero_line = f'<line class="zero" x1="{pad_left}" x2="{width - pad_right}" y1="{zy:.2f}" y2="{zy:.2f}" />'

    polylines = ""
    for points, css_class, _label in series:
        pts = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
        if len(pts) >= 2:
            pt_text = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in pts)
            polylines += f'<polyline class="{html.escape(css_class)}" points="{pt_text}" />\n        '

    # Legend chips (one per series that has a label)
    legend_items = [
        (css_class, label)
        for _, css_class, label in series
        if label
    ]
    legend_svg = ""
    if len(legend_items) > 1:
        lx = pad_left + 6
        ly = pad_top + 6
        for i, (css_class, label) in enumerate(legend_items):
            item_y = ly + i * 16
            legend_svg += (
                f'<line class="{html.escape(css_class)}" x1="{lx}" x2="{lx+18}" y1="{item_y+5}" y2="{item_y+5}" />'
                f'<text class="legend-label" x="{lx+22}" y="{item_y+9}">{html.escape(label)}</text>\n        '
            )

    return f"""
      <svg viewBox="0 0 {width} {height}" role="img">
        <rect class="plot-bg" x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" />
        {zero_line}
        {polylines}
        {legend_svg}
        <line class="axis" x1="{pad_left}" x2="{width - pad_right}" y1="{height - pad_bottom}" y2="{height - pad_bottom}" />
        <line class="axis" x1="{pad_left}" x2="{pad_left}" y1="{pad_top}" y2="{height - pad_bottom}" />
        <text class="tick" x="{pad_left}" y="{height - 10}">{_format_float(x_min)}s</text>
        <text class="tick end" x="{width - pad_right}" y="{height - 10}">{_format_float(x_max)}s</text>
        <text class="tick" x="8" y="{pad_top + 4}">{_format_float(y_max)}</text>
        <text class="tick" x="8" y="{height - pad_bottom}">{_format_float(y_min)}</text>
      </svg>
    """


def write_id_html_report(
    sto_path: Path,
    *,
    output_path: Path | None = None,
    mjx_torques: dict[str, list[float]] | None = None,
) -> Path:
    columns, rows = read_storage_file(sto_path)
    if not rows or "time" not in columns:
        raise ValueError(f"{sto_path} does not look like a time-series OpenSim storage file")
    output_path = output_path or sto_path.with_name("inverse_dynamics_torque_plots.html")
    time_col = columns.index("time")
    time_values = [row[time_col] for row in rows]
    graphs: list[str] = []
    summary_rows: list[str] = []

    has_mjx = bool(mjx_torques)
    legend_note = " — <span class='legend-osim'>blue: OpenSim ID</span>, <span class='legend-mjx'>orange: MJX GT</span>" if has_mjx else ""

    for col_idx, column in enumerate(columns):
        if col_idx == time_col:
            continue
        values = [row[col_idx] for row in rows]
        finite_values = [v for v in values if math.isfinite(v)]
        if finite_values:
            mean = sum(finite_values) / len(finite_values)
            rms = math.sqrt(sum(v * v for v in finite_values) / len(finite_values))
            min_v = min(finite_values)
            max_v = max(finite_values)
        else:
            mean = rms = min_v = max_v = math.nan
        safe_col = html.escape(column)
        graph_id = html.escape(column.replace(" ", "_"))

        osim_points = [(t, v) for t, v in zip(time_values, values) if math.isfinite(t) and math.isfinite(v)]
        all_series: list[tuple[list[tuple[float, float]], str, str]] = [
            (osim_points, "series-osim", "OpenSim ID"),
        ]

        # Strip trailing "_moment" / "_force" suffix to get coord name for MJX lookup.
        coord_key = column.removesuffix("_moment").removesuffix("_force")
        mjx_vals = mjx_torques.get(coord_key) if mjx_torques else None
        mjx_mean_str = ""
        if mjx_vals is not None:
            n = min(len(mjx_vals), len(time_values))
            mjx_points = [(t, v) for t, v in zip(time_values[:n], mjx_vals[:n]) if math.isfinite(t) and math.isfinite(v)]
            if mjx_points:
                all_series.append((mjx_points, "series-mjx", "MJX GT"))
                mjx_mean = sum(v for _, v in mjx_points) / len(mjx_points)
                mjx_mean_str = f" | MJX mean={_format_float(mjx_mean)}"

        meta = (
            f"mean={_format_float(mean)} | rms={_format_float(rms)}"
            f" | min={_format_float(min_v)} | max={_format_float(max_v)}{mjx_mean_str}"
        )
        graphs.append(
            f"""
            <section class="card" id="{graph_id}">
              <h2>{safe_col}</h2>
              <div class="meta">{meta}</div>
              {_svg_multiline(all_series, width=960, height=260)}
            </section>
            """
        )
        summary_rows.append(
            "<tr>"
            f"<td><a href=\"#{graph_id}\">{safe_col}</a></td>"
            f"<td>{_format_float(mean)}</td>"
            f"<td>{_format_float(rms)}</td>"
            f"<td>{_format_float(min_v)}</td>"
            f"<td>{_format_float(max_v)}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OpenSim Inverse Dynamics Torque Plots</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #f7f7f8; color: #1f2933; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .subtle {{ color: #5b6773; margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; background: white; }}
    th, td {{ border: 1px solid #d7dce2; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #edf1f5; position: sticky; top: 0; }}
    a {{ color: #1f6feb; text-decoration: none; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d7dce2; border-radius: 8px; padding: 14px 16px 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .card h2 {{ margin: 0 0 4px; font-size: 1rem; }}
    .meta {{ color: #5b6773; font-size: 0.85rem; margin-bottom: 8px; }}
    svg {{ width: 100%; height: auto; }}
    .plot-bg {{ fill: #fbfcfd; stroke: #e5e9ef; }}
    .axis {{ stroke: #5b6773; stroke-width: 1; }}
    .zero {{ stroke: #c9ced6; stroke-width: 1; stroke-dasharray: 4 4; }}
    .series-osim {{ fill: none; stroke: #1f6feb; stroke-width: 1.5; }}
    .series-mjx {{ fill: none; stroke: #e07b20; stroke-width: 1.5; stroke-dasharray: 6 3; }}
    .tick {{ fill: #5b6773; font-size: 12px; dominant-baseline: middle; }}
    .tick.end {{ text-anchor: end; }}
    .legend-label {{ fill: #1f2933; font-size: 11px; dominant-baseline: middle; }}
    .legend-osim {{ color: #1f6feb; font-weight: 600; }}
    .legend-mjx {{ color: #e07b20; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>OpenSim Inverse Dynamics Torque Plots</h1>
  <p class="subtle">Source: {html.escape(str(sto_path))} | frames: {len(rows)} | DOFs: {len(columns) - 1}{legend_note}</p>
  <table>
    <thead><tr><th>DOF</th><th>Mean</th><th>RMS</th><th>Min</th><th>Max</th></tr></thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>
  <div class="grid">
    {''.join(graphs)}
  </div>
</body>
</html>
"""
    output_path.write_text(doc)
    return output_path


def process_trial(
    paths: TrialPaths,
    *,
    use_noised: bool,
    overwrite: bool,
    dry_run: bool,
    right_body: str,
    left_body: str,
    source: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "subject": paths.subject_dir.name,
        "trial": paths.trial_dir.name,
        "trial_dir": str(paths.trial_dir),
        "output_dir": str(paths.output_dir),
    }

    existing_outputs = find_id_outputs(paths)
    if existing_outputs and not overwrite and existing_generation_matches(paths, source=source, use_noised=use_noised):
        primary_output = primary_id_output(paths, existing_outputs)
        html_report = None
        if primary_output is not None:
            _, existing_rows = read_storage_file(primary_output)
            mjx_torques = load_mjx_id_torques(paths.trial_dir, len(existing_rows))
            html_report = write_id_html_report(primary_output, mjx_torques=mjx_torques)
        result.update(
            {
                "status": "skipped",
                "reason": "ID output already exists",
                "source": source,
                "id_outputs": [str(path) for path in existing_outputs],
                "html_report": str(html_report) if html_report is not None else None,
            }
        )
        return result

    start = time.perf_counter()
    generated = generate_trial_inputs(
        paths,
        use_noised=use_noised,
        overwrite=overwrite,
        dry_run=dry_run,
        right_body=right_body,
        left_body=left_body,
        source=source,
    )
    result["input_generation"] = generated
    setup_file = paths.output_dir / "id_setup.xml"
    result["setup_file"] = str(setup_file)

    if dry_run:
        result.update({"status": "dry_run", "elapsed_sec": time.perf_counter() - start})
        return result

    if not setup_file.exists():
        raise FileNotFoundError(f"generated setup file not found: {setup_file}")

    run_inverse_dynamics(setup_file)
    outputs = find_id_outputs(paths)
    primary_output = primary_id_output(paths, outputs)
    if primary_output is not None:
        _, id_rows = read_storage_file(primary_output)
        mjx_torques = load_mjx_id_torques(paths.trial_dir, len(id_rows))
        html_report = write_id_html_report(primary_output, mjx_torques=mjx_torques)
    else:
        html_report = None
    result.update(
        {
            "status": "ok",
            "elapsed_sec": time.perf_counter() - start,
            "id_outputs": [str(path) for path in outputs],
            "html_report": str(html_report) if html_report is not None else None,
        }
    )
    if not outputs:
        result["warning"] = "OpenSim run completed, but no inverse dynamics output file was found"
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--trial", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir-name", default=DEFAULT_OUTPUT_DIR_NAME)
    parser.add_argument("--right-body", default=DEFAULT_RIGHT_BODY)
    parser.add_argument("--left-body", default=DEFAULT_LEFT_BODY)
    parser.add_argument("--source", choices=(SOURCE_MOTION, SOURCE_PROCESSED), default=SOURCE_PROCESSED)
    parser.add_argument("--use-noised", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest-name",
        default="opensim_id_batch_manifest.json",
        help="Dataset-level manifest filename.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        return 2

    trials = discover_trials(
        dataset_root,
        subject=args.subject,
        trial=args.trial,
        model_name=args.model_name,
        output_dir_name=args.output_dir_name,
    )
    if args.limit is not None:
        trials = trials[: args.limit]

    manifest: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "use_noised": args.use_noised,
        "source": args.source,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "trials_seen": len(trials),
        "trials_ok": 0,
        "trials_skipped": 0,
        "trials_failed": 0,
        "results": [],
        "failures": [],
    }

    for paths in trials:
        try:
            result = process_trial(
                paths,
                use_noised=args.use_noised,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                right_body=args.right_body,
                left_body=args.left_body,
                source=args.source,
            )
            manifest["results"].append(result)
            if result.get("status") == "skipped":
                manifest["trials_skipped"] += 1
            elif result.get("status") in {"ok", "dry_run"}:
                manifest["trials_ok"] += 1
        except Exception as exc:
            manifest["trials_failed"] += 1
            manifest["failures"].append(
                {
                    "subject": paths.subject_dir.name,
                    "trial": paths.trial_dir.name,
                    "trial_dir": str(paths.trial_dir),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
        if manifest["trials_seen"]:
            done = manifest["trials_ok"] + manifest["trials_skipped"] + manifest["trials_failed"]
            print(f"processed {done}/{manifest['trials_seen']}", flush=True)

    manifest_path = dataset_root / args.manifest_name
    if not args.dry_run:
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        manifest["manifest_path"] = str(manifest_path)

    print(json.dumps(manifest, indent=2))
    return 1 if manifest["trials_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
