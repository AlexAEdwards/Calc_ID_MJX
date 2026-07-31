#!/usr/bin/env python3
"""Convert Video and MoCap OpenSim models for OpenCapWalkingTrunkSwaySubjects.

This mirrors the repository's existing MyoConverter settings from
``myoconverter/convert_all_opensim_models.py``: step 1 only, speedy conversion,
ground geom enabled, no validation or PDF generation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "str"):
    np.str = str


def _preload_opensim_bundled_ezc3d() -> None:
    """Prefer OpenSim's bundled ezc3d ABI over any conda-installed ezc3d."""
    for site_packages in Path(sys.prefix).glob("lib/python*/site-packages"):
        bundled = site_packages / "opensim" / "libezc3d.so"
        if bundled.exists():
            ctypes.CDLL(str(bundled), mode=ctypes.RTLD_GLOBAL)
            return


REPO_ROOT = Path(__file__).resolve().parents[1]
MYOCONVERTER_ROOT = REPO_ROOT / "myoconverter"
sys.path.insert(0, str(MYOCONVERTER_ROOT))

_preload_opensim_bundled_ezc3d()

from myoconverter.O2MPipeline import O2MPipeline  # noqa: E402


DEFAULT_DATASET_ROOT = REPO_ROOT / "OpenCapWalkingTrunkSwaySubjects"
DEFAULT_GEOMETRY_FOLDER = REPO_ROOT / "GeometryWithMus"
DEFAULT_TEMP_ROOT = MYOCONVERTER_ROOT / "temp_opencap_walking_trunksway_models"

MODEL_SPECS = {
    "MoCap": ("OpenSimScaled_MoCap.osim", "MyosuiteModel_MoCap.xml"),
    "Video": ("OpenSimScaled_Video.osim", "MyosuiteModel_Video.xml"),
}

PIPELINE_KWARGS = {
    "convert_steps": [1],
    "muscle_list": None,
    "osim_data_overwrite": True,
    "conversion": True,
    "validation": False,
    "speedy": True,
    "generate_pdf": False,
    "add_ground_geom": True,
    "treat_as_normal_path_point": False,
}


def get_opensim_version(osim_path: Path) -> str | None:
    try:
        with osim_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "<OpenSimDocument" in line and 'Version="' in line:
                    start = line.find('Version="') + 9
                    end = line.find('"', start)
                    if end > start:
                        return line[start:end]
                    return None
    except Exception:
        return None
    return None


def upgrade_opensim_40_to_405(osim_path: Path) -> bool:
    """Apply the same OpenSim 4.0 compatibility edits as the existing converter."""
    try:
        content = osim_path.read_text(encoding="utf-8")
        backup = osim_path.with_suffix(osim_path.suffix + ".bak")
        backup.write_text(content, encoding="utf-8")
        content = content.replace('Version="40000"', 'Version="40500"')
        content = content.replace(
            '<GeometryPath name="geometrypath">',
            '<GeometryPath name="path">',
        )
        pattern = (
            r"(<GeometryPath[^>]*>)\s*(<!--[^>]*-->)?\s*"
            r"(<PathPointSet>.*?</PathPointSet>)\s*(<!--[^>]*-->)?\s*"
            r"(<PathWrapSet>.*?</PathWrapSet>)\s*(<!--[^>]*-->)?\s*"
            r"(<Appearance>.*?</Appearance>)\s*(</GeometryPath>)"
        )

        def swap_elements(match: re.Match[str]) -> str:
            geometry_path_open = match.group(1)
            point_comment = match.group(2) or ""
            path_point_set = match.group(3)
            wrap_comment = match.group(4) or ""
            path_wrap_set = match.group(5)
            appearance_comment = match.group(6) or ""
            appearance = match.group(7)
            geometry_path_close = match.group(8)

            result = f"{geometry_path_open}\n"
            if appearance_comment:
                result += f"                    {appearance_comment}\n"
            result += f"                    {appearance}\n"
            if point_comment:
                result += f"                    {point_comment}\n"
            result += f"                    {path_point_set}\n"
            if wrap_comment:
                result += f"                    {wrap_comment}\n"
            result += f"                    {path_wrap_set}\n                {geometry_path_close}"
            return result

        content = re.sub(pattern, swap_elements, content, flags=re.DOTALL)
        osim_path.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False


def discover_subject_dirs(dataset_root: Path) -> list[Path]:
    return sorted(p for p in dataset_root.glob("subject*") if p.is_dir())


def verify_models(subject_dirs: list[Path]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for subject_dir in subject_dirs:
        for model_kind, (source_name, _output_name) in MODEL_SPECS.items():
            source = subject_dir / source_name
            if not source.exists():
                missing.append(
                    {
                        "subject": subject_dir.name,
                        "model_kind": model_kind,
                        "missing": str(source),
                    }
                )
    return missing


def convert_one(
    subject_dir: Path,
    model_kind: str,
    source_name: str,
    output_name: str,
    *,
    geometry_folder: Path,
    temp_root: Path,
    overwrite: bool,
) -> dict:
    osim_path = subject_dir / source_name
    output_xml = subject_dir / output_name
    if output_xml.exists() and not overwrite:
        return {
            "subject": subject_dir.name,
            "model_kind": model_kind,
            "source": str(osim_path),
            "output": str(output_xml),
            "status": "skipped_existing",
        }

    temp_dir = temp_root / f"{subject_dir.name}_{model_kind}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    upgraded = False
    if get_opensim_version(osim_path) == "40000":
        upgraded = upgrade_opensim_40_to_405(osim_path)

    try:
        O2MPipeline(str(osim_path), str(geometry_folder), str(temp_dir), **PIPELINE_KWARGS)
        candidates = sorted(
            list(temp_dir.glob("*_cvt*_FIXED.xml")) + list(temp_dir.glob("*_cvt*.xml")),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise RuntimeError("No converted XML produced by MyoConverter")
        shutil.copyfile(candidates[-1], output_xml)

        temp_geometry = temp_dir / "Geometry"
        if temp_geometry.exists():
            subject_geometry = subject_dir / "Geometry"
            subject_geometry.mkdir(exist_ok=True)
            for geom in temp_geometry.glob("*"):
                if geom.is_file():
                    shutil.copyfile(geom, subject_geometry / geom.name)

        return {
            "subject": subject_dir.name,
            "model_kind": model_kind,
            "source": str(osim_path),
            "output": str(output_xml),
            "status": "converted",
            "upgraded_opensim_40000": upgraded,
        }
    except Exception as exc:
        return {
            "subject": subject_dir.name,
            "model_kind": model_kind,
            "source": str(osim_path),
            "output": str(output_xml),
            "status": "failed",
            "error": str(exc),
            "upgraded_opensim_40000": upgraded,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--geometry-folder", type=Path, default=DEFAULT_GEOMETRY_FOLDER)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    geometry_folder = args.geometry_folder.resolve()
    temp_root = args.temp_root.resolve()
    subject_dirs = discover_subject_dirs(dataset_root)

    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")
    if not geometry_folder.exists():
        raise SystemExit(f"Geometry folder not found: {geometry_folder}")
    if not subject_dirs:
        raise SystemExit(f"No subject folders found under: {dataset_root}")

    missing = verify_models(subject_dirs)
    if missing:
        print(json.dumps({"missing_models": missing}, indent=2))
        return 2

    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    results = []
    started = datetime.now().isoformat(timespec="seconds")
    for subject_dir in subject_dirs:
        for model_kind, (source_name, output_name) in MODEL_SPECS.items():
            print(f"[{subject_dir.name}] converting {model_kind}: {source_name} -> {output_name}", flush=True)
            result = convert_one(
                subject_dir,
                model_kind,
                source_name,
                output_name,
                geometry_folder=geometry_folder,
                temp_root=temp_root,
                overwrite=bool(args.overwrite),
            )
            print(f"  {result['status']}", flush=True)
            results.append(result)

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    report = {
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "geometry_folder": str(geometry_folder),
        "settings": PIPELINE_KWARGS,
        "model_specs": MODEL_SPECS,
        "counts": counts,
        "results": results,
    }
    report_path = dataset_root / "opencap_walking_trunksway_model_conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)

    print(json.dumps({"counts": counts, "report": str(report_path)}, indent=2))
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
