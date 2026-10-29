"""Put the existing map scripts on the import path without importing Habitat."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def install_habitat():
    install()
    package = str((ROOT / "streamvln").resolve())
    sys.path[:] = [item for item in sys.path if str(Path(item).resolve()) != package]
    sys.path.insert(0, package)
    _drop_foreign_habitat_extensions(package)


def _drop_foreign_habitat_extensions(package_root):
    """Open-Nav also ships a habitat_extensions package. Drop it if it was imported first."""
    expected = str(Path(package_root) / "habitat_extensions")
    current = sys.modules.get("habitat_extensions")
    if current is None:
        return
    paths = [str(Path(item).resolve()) for item in (getattr(current, "__path__", None) or [])]
    if any(item == expected for item in paths):
        return
    for name in list(sys.modules):
        if name == "habitat_extensions" or name.startswith("habitat_extensions."):
            del sys.modules[name]
