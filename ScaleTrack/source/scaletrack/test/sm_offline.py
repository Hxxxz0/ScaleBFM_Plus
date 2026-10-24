"""Load the pure-torch SuperMimic modules without importing IsaacLab (for offline tests)."""

import importlib.util
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "scaletrack" / "tasks" / "tracking" / "supermimic"
ASSET_XML = Path(__file__).resolve().parents[1] / "scaletrack" / "assets" / "robots" / "g1_29dof" / "g1_29dof.xml"


def load(module: str):
    if "sm_offline_pkg" not in sys.modules:
        spec = importlib.util.spec_from_file_location("sm_offline_pkg", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["sm_offline_pkg"] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"sm_offline_pkg.{module}")
