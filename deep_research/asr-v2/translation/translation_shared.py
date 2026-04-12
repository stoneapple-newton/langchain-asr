from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parent / "shared" / "translation_utils.py"
SPEC = importlib.util.spec_from_file_location("_translation_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

__all__ = [name for name in dir(MODULE) if not name.startswith("_")]

for name in __all__:
    globals()[name] = getattr(MODULE, name)
