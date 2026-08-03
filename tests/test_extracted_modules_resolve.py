"""Every global name an extracted module uses must actually resolve.

Stage 6 moves function clusters out of ProcessData.py into `processing/`. A moved
function stops seeing ProcessData's module globals, so any import the cluster
relied on has to travel with it. When one doesn't, the result is a NameError that
only fires when that specific code path runs.

`tools/equivalence_check.py` cannot catch this: it compares outputs of the paths
the fixture exercises, and an unexercised path has no output to differ. That is
exactly how `gcv_derivatives` shipped referencing an unimported
`make_smoothing_spline` - it is the opt-in OpenSim-filtering path, so the gate ran
green over a function that would have raised on first use.

This check is static, so it covers every function whether or not it is called.
"""

from __future__ import annotations

import builtins
import importlib
import pkgutil
import symtable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_PACKAGES = ("processing", "core")


def _module_names(pkg: str) -> list[str]:
    path = REPO_ROOT / pkg
    if not path.is_dir():
        return []
    return [f"{pkg}.{m.name}" for m in pkgutil.iter_modules([str(path)])]


def _global_names(table: symtable.SymbolTable) -> set[str]:
    """Names a block reads from module scope, recursing into nested blocks."""
    found = set()
    for sym in table.get_symbols():
        if sym.is_global() and not sym.is_assigned():
            found.add(sym.get_name())
    for child in table.get_children():
        found |= _global_names(child)
    return found


ALL_MODULES = [m for pkg in EXTRACTED_PACKAGES for m in _module_names(pkg)]


@pytest.mark.parametrize("modname", ALL_MODULES)
def test_extracted_module_globals_resolve(modname: str) -> None:
    """No function in an extracted module references an unimported global."""
    pytest.importorskip("numpy")
    try:
        mod = importlib.import_module(modname)
    except ImportError as exc:  # jax/mujoco not installed in this env
        pytest.skip(f"{modname} needs an unavailable dependency: {exc}")

    src = Path(mod.__file__).read_text()
    used = _global_names(symtable.symtable(src, mod.__file__, "exec"))
    available = set(vars(mod)) | set(dir(builtins))

    unresolved = sorted(used - available)
    assert not unresolved, (
        f"{modname} uses global name(s) that do not resolve: {unresolved}. "
        "An import probably did not travel with the extracted code."
    )


def test_the_check_itself_detects_a_missing_import(tmp_path: Path) -> None:
    """Guard against this test silently passing because it inspects nothing."""
    src = "def f():\n    return some_missing_helper(1)\n"
    used = _global_names(symtable.symtable(src, "x.py", "exec"))
    assert "some_missing_helper" in used


def test_some_modules_were_actually_found() -> None:
    assert ALL_MODULES, "no extracted modules discovered - the check is vacuous"
