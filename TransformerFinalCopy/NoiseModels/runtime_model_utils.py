"""Utilities for generating mod_q runtime XMLs with harmonized site counts.

The mod_q training/inference path uses ``MyosuiteModel_Runtime.xml`` files to
reduce MJX/XLA recompiles caused by otherwise-identical subject models that
only differ in site inventory. These runtime XMLs are generated from each
subject's ``MyosuiteModel_FIXED.xml`` (or ``MyosuiteModel.xml`` as a fallback)
and are intended for mod_q only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Optional, Tuple
import xml.etree.ElementTree as ET

try:
    import mujoco
except Exception:  # pragma: no cover - import-time safety
    mujoco = None


RUNTIME_XML_NAME = "MyosuiteModel_Runtime.xml"
RUNTIME_META_NAME = "MyosuiteModel_Runtime.meta.json"
_RUNTIME_SITE_PREFIX = "modq_runtime_pad_site_"

_FAMILY_SCAN_CACHE: Dict[str, Dict[Tuple[int, ...], Dict[str, Any]]] = {}
_RUNTIME_INFO_CACHE: Dict[str, "RuntimeModelInfo"] = {}


@dataclass(frozen=True)
class RuntimeModelInfo:
    source_xml: Path
    runtime_xml: Path
    core_shape: Tuple[int, ...]
    runtime_shape: Tuple[int, ...]
    structure_key: str
    target_nsite: int
    added_sites: int


def _core_shape_from_model(model: Any) -> Tuple[int, ...]:
    return (
        int(model.nq),
        int(model.nv),
        int(model.nu),
        int(model.na),
        int(model.nbody),
        int(model.njnt),
        int(model.ngeom),
        int(model.neq),
    )


def _runtime_shape_from_model(model: Any) -> Tuple[int, ...]:
    return (
        int(model.nq),
        int(model.nv),
        int(model.nu),
        int(model.na),
        int(model.nbody),
        int(model.njnt),
        int(model.ngeom),
        int(model.nsite),
        int(model.neq),
    )


def _structure_key_from_shape(runtime_shape: Tuple[int, ...]) -> str:
    digest = hashlib.sha1(repr(runtime_shape).encode("utf-8")).hexdigest()[:16]
    return f"modq_rt_{digest}"


def _load_model(xml_path: Path):
    if mujoco is None:
        raise RuntimeError("mujoco is required to build mod_q runtime XMLs.")
    return mujoco.MjModel.from_xml_path(str(xml_path))


def _resolve_source_xml(path_like: Path | str) -> Path:
    path = Path(path_like)
    if path.is_dir():
        fixed = path / "MyosuiteModel_FIXED.xml"
        raw = path / "MyosuiteModel.xml"
    else:
        subject_dir = path.parent
        fixed = subject_dir / "MyosuiteModel_FIXED.xml"
        raw = subject_dir / "MyosuiteModel.xml"
        if path.name == RUNTIME_XML_NAME and fixed.exists():
            return fixed
        if path.name == "MyosuiteModel_FIXED.xml":
            return path
        if path.name == "MyosuiteModel.xml" and fixed.exists():
            return fixed
    if fixed.exists():
        return fixed
    if raw.exists():
        return raw
    raise FileNotFoundError(
        f"Could not resolve a subject XML from {path_like}. "
        "Expected MyosuiteModel_FIXED.xml or MyosuiteModel.xml."
    )


def _iter_fixed_xmls(dataset_root: Path) -> Iterable[Path]:
    for subject_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        fixed = subject_dir / "MyosuiteModel_FIXED.xml"
        raw = subject_dir / "MyosuiteModel.xml"
        if fixed.exists():
            yield fixed
        elif raw.exists():
            yield raw


def _scan_dataset_families(dataset_root: Path) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    cache_key = str(dataset_root.resolve())
    cached = _FAMILY_SCAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    families: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    if mujoco is None:
        _FAMILY_SCAN_CACHE[cache_key] = families
        return families

    for xml_path in _iter_fixed_xmls(dataset_root):
        try:
            model = _load_model(xml_path)
        except Exception:
            continue
        core_shape = _core_shape_from_model(model)
        runtime_shape = _runtime_shape_from_model(model)
        entry = families.setdefault(
            core_shape,
            {
                "target_nsite": int(runtime_shape[7]),
                "members": [],
            },
        )
        entry["target_nsite"] = max(int(entry["target_nsite"]), int(runtime_shape[7]))
        entry["members"].append(str(xml_path))

    for core_shape, entry in families.items():
        runtime_shape = core_shape[:7] + (int(entry["target_nsite"]), core_shape[7])
        entry["runtime_shape"] = runtime_shape
        entry["structure_key"] = _structure_key_from_shape(runtime_shape)

    _FAMILY_SCAN_CACHE[cache_key] = families
    return families


def _existing_runtime_meta(runtime_xml: Path) -> Optional[Dict[str, Any]]:
    meta_path = runtime_xml.with_name(RUNTIME_META_NAME)
    if not (runtime_xml.exists() and meta_path.exists()):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _write_runtime_meta(runtime_xml: Path, meta: Dict[str, Any]) -> None:
    meta_path = runtime_xml.with_name(RUNTIME_META_NAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _choose_site_anchor(root: ET.Element) -> ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML is missing <worldbody>; cannot add runtime padding sites.")

    for body in worldbody.findall("body"):
        if body.get("name") == "ground":
            return body

    first_body = worldbody.find("body")
    if first_body is not None:
        return first_body

    raise ValueError("MuJoCo XML has no body under <worldbody>; cannot add runtime padding sites.")


def _collect_existing_site_names(root: ET.Element) -> set[str]:
    names: set[str] = set()
    for site in root.iter("site"):
        name = site.get("name")
        if name:
            names.add(name)
    return names


def _add_padding_sites(root: ET.Element, count: int) -> int:
    if count <= 0:
        return 0
    anchor = _choose_site_anchor(root)
    existing_names = _collect_existing_site_names(root)
    added = 0
    next_idx = 0
    while added < count:
        name = f"{_RUNTIME_SITE_PREFIX}{next_idx:04d}"
        next_idx += 1
        if name in existing_names:
            continue
        site = ET.SubElement(anchor, "site")
        site.set("name", name)
        site.set("pos", "0 0 -1000")
        site.set("size", "1e-06")
        site.set("rgba", "0 0 0 0")
        site.set("group", "31")
        existing_names.add(name)
        added += 1
    return added


def ensure_modq_runtime_xml(path_like: Path | str) -> RuntimeModelInfo:
    """Create or refresh ``MyosuiteModel_Runtime.xml`` for a subject XML.

    The returned runtime XML preserves the source model structure and numeric
    parameters, but pads site count up to the family-wide maximum so mod_q can
    share compiled MJX executables across subjects with matching runtime shape.
    """
    source_xml = _resolve_source_xml(path_like)
    cache_key = str(source_xml.resolve())
    cached = _RUNTIME_INFO_CACHE.get(cache_key)
    if cached is not None and cached.runtime_xml.exists():
        source_mtime_ns = int(source_xml.stat().st_mtime_ns)
        meta = _existing_runtime_meta(cached.runtime_xml)
        if meta is not None and int(meta.get("source_mtime_ns", -1)) == source_mtime_ns:
            return cached

    if mujoco is None:
        runtime_xml = source_xml.with_name(RUNTIME_XML_NAME)
        if not runtime_xml.exists() or source_xml.stat().st_mtime_ns > runtime_xml.stat().st_mtime_ns:
            shutil.copyfile(source_xml, runtime_xml)
        info = RuntimeModelInfo(
            source_xml=source_xml,
            runtime_xml=runtime_xml,
            core_shape=tuple(),
            runtime_shape=tuple(),
            structure_key=f"modq_rt_path_{source_xml.parent.name}",
            target_nsite=0,
            added_sites=0,
        )
        _RUNTIME_INFO_CACHE[cache_key] = info
        return info

    source_model = _load_model(source_xml)
    core_shape = _core_shape_from_model(source_model)
    source_runtime_shape = _runtime_shape_from_model(source_model)

    dataset_root = source_xml.parent.parent
    families = _scan_dataset_families(dataset_root)
    family = families.get(core_shape)
    target_nsite = int(family["target_nsite"]) if family is not None else int(source_runtime_shape[7])
    runtime_shape = core_shape[:7] + (target_nsite, core_shape[7])
    structure_key = (
        str(family.get("structure_key"))
        if family is not None and "structure_key" in family
        else _structure_key_from_shape(runtime_shape)
    )

    runtime_xml = source_xml.with_name(RUNTIME_XML_NAME)
    source_mtime_ns = int(source_xml.stat().st_mtime_ns)
    meta = _existing_runtime_meta(runtime_xml)
    if meta is not None:
        meta_matches = (
            int(meta.get("source_mtime_ns", -1)) == source_mtime_ns
            and int(meta.get("target_nsite", -1)) == target_nsite
            and tuple(meta.get("runtime_shape", [])) == runtime_shape
            and str(meta.get("structure_key", "")) == structure_key
        )
        if meta_matches and runtime_xml.exists():
            info = RuntimeModelInfo(
                source_xml=source_xml,
                runtime_xml=runtime_xml,
                core_shape=core_shape,
                runtime_shape=runtime_shape,
                structure_key=structure_key,
                target_nsite=target_nsite,
                added_sites=int(meta.get("added_sites", 0)),
            )
            _RUNTIME_INFO_CACHE[cache_key] = info
            return info

    current_nsite = int(source_runtime_shape[7])
    added_sites = max(0, target_nsite - current_nsite)
    if added_sites <= 0:
        shutil.copyfile(source_xml, runtime_xml)
    else:
        tree = ET.parse(str(source_xml))
        root = tree.getroot()
        actual_added = _add_padding_sites(root, added_sites)
        try:
            ET.indent(tree, space="  ")
        except Exception:
            pass
        tree.write(runtime_xml, encoding="utf-8", xml_declaration=True)
        added_sites = int(actual_added)

    meta = {
        "source_xml": str(source_xml),
        "runtime_xml": str(runtime_xml),
        "source_mtime_ns": source_mtime_ns,
        "core_shape": list(core_shape),
        "runtime_shape": list(runtime_shape),
        "structure_key": structure_key,
        "target_nsite": target_nsite,
        "added_sites": added_sites,
    }
    _write_runtime_meta(runtime_xml, meta)

    info = RuntimeModelInfo(
        source_xml=source_xml,
        runtime_xml=runtime_xml,
        core_shape=core_shape,
        runtime_shape=runtime_shape,
        structure_key=structure_key,
        target_nsite=target_nsite,
        added_sites=added_sites,
    )
    _RUNTIME_INFO_CACHE[cache_key] = info
    return info


def resolve_modq_runtime_xml(path_like: Path | str) -> Path:
    return ensure_modq_runtime_xml(path_like).runtime_xml


def modq_runtime_structure_key(path_like: Path | str) -> str:
    return ensure_modq_runtime_xml(path_like).structure_key
