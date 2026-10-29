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
    package = str(ROOT / "streamvln")
    if package not in sys.path:
        sys.path.insert(0, package)
