"""Put the existing map scripts on the import path without importing Habitat."""

import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = Path("/root/data1/StreamVLN/results/geonav_r2r_100")


def redirect_stdio(path):
    """Write stdout and stderr to a file, and keep running after the terminal closes.

    A remote IDE stops reading its terminal when the laptop sleeps. The next
    print then blocks on a full pipe. A file does not need a reader, and
    ignoring hangup keeps the process alive after that terminal is torn down.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notice = f"日志写到 {path}  pid={os.getpid()}\n".encode()
    for fd in (2, 1):
        try:
            os.write(fd, notice)
            break
        except OSError:
            continue
    hangup = getattr(signal, "SIGHUP", None)
    if hangup is not None:
        signal.signal(hangup, signal.SIG_IGN)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        targets = {1, 2}
        for stream in (sys.stdout, sys.stderr):
            try:
                targets.add(stream.fileno())
            except (AttributeError, OSError, ValueError):
                pass
        for target in targets:
            try:
                os.dup2(log_fd, target)
            except OSError:
                pass
    finally:
        if log_fd > 2:
            os.close(log_fd)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"pid={os.getpid()}", flush=True)


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
