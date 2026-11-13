"""Put the existing map scripts on the import path without importing Habitat."""

import errno
import io
import os
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = Path("/root/data1/StreamVLN/results/geonav_r2r_100")
_WRITE_LOCK = threading.RLock()
_AGAIN = {errno.EAGAIN}
if hasattr(errno, "EWOULDBLOCK"):
    _AGAIN.add(errno.EWOULDBLOCK)


class _Tee(io.TextIOBase):
    """Write the full text to the log, and a non-blocking copy to the terminal."""

    def __init__(self, log_stream, terminal_fd):
        super().__init__()
        self._log = log_stream
        self._terminal = terminal_fd
        self._alive = terminal_fd is not None

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        with _WRITE_LOCK:
            self._log.write(text)
            self._log.flush()
            self._emit(text.encode("utf-8", errors="replace"))
        return len(text)

    def flush(self):
        with _WRITE_LOCK:
            self._log.flush()

    def _emit(self, data):
        if not self._alive or not data:
            return
        view = memoryview(data)
        while len(view):
            try:
                written = os.write(self._terminal, view)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno in _AGAIN:
                    return
                self._alive = False
                return
            if written <= 0:
                self._alive = False
                return
            view = view[written:]

    def fileno(self):
        return self._log.fileno()

    def isatty(self):
        if not self._alive:
            return False
        try:
            return os.isatty(self._terminal)
        except OSError:
            return False

    def readable(self):
        return False

    def writable(self):
        return True

    @property
    def encoding(self):
        return "utf-8"

    @property
    def errors(self):
        return "replace"


def _save_terminal(stream, fallback):
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        fd = fallback
    try:
        return os.dup(fd)
    except OSError:
        return None


def redirect_stdio(path):
    """Copy stdout and stderr to a file and to the terminal.

    The terminal copy does not wait. When the remote IDE stops reading, that
    write is dropped and the process keeps running. The file keeps the full
    text. Hangup does not kill the process.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hangup = getattr(signal, "SIGHUP", None)
    if hangup is not None:
        signal.signal(hangup, signal.SIG_IGN)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    term_out = _save_terminal(sys.stdout, 1)
    term_err = _save_terminal(sys.stderr, 2)
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        targets = {1, 2}
        for stream in (sys.stdout, sys.stderr):
            try:
                targets.add(stream.fileno())
            except (AttributeError, OSError, ValueError):
                pass
        saved = {fd for fd in (term_out, term_err) if fd is not None}
        for target in targets - saved:
            try:
                os.dup2(log_fd, target)
            except OSError:
                pass
    finally:
        if log_fd > 2:
            os.close(log_fd)
    for fd in (term_out, term_err):
        if fd is None:
            continue
        try:
            os.set_blocking(fd, False)
        except (AttributeError, OSError):
            pass
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass
    sys.stdout = _Tee(sys.stdout, term_out)
    sys.stderr = _Tee(sys.stderr, term_err)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"日志写到 {path}  pid={os.getpid()}", flush=True)


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
