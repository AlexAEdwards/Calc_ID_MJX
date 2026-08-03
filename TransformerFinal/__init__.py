"""Model, training, inference and LOSO/LOEO code.

The modules in this package import each other as flat top-level names
(``from data_loader import TrialDataLoader``) because they were written as
scripts. Registering this directory on ``sys.path`` here is what lets that keep
working when the package is imported from elsewhere, e.g.::

    from TransformerFinal.data_loader import TrialDataLoader

Doing it once, here, replaces the ``sys.path.insert`` block that used to be
copied into ~28 files. Converting the flat imports to real relative imports is
Stage 5 of REFACTOR_PLAN.md; until then this single line is the bridge.
"""

import sys as _sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

del _sys, _Path, _HERE
