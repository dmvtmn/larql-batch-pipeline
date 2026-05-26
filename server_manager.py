"""server_manager.py — Task 3
Subprocess wrapper for the LARQL server with health-check and clean shutdown.
Usage:
    python server_manager.py          # start server, block until ready, then keep alive
    from server_manager import start_server, stop_server  # import API
"""
import atexit
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [server_manager] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
LARQL_BINARY = "./larql/target/release/larql"
VINDEX_PATH = "./data/gemma3-4b.vindex"
PORT = 8080
LOG_FILE = "larql_server.log"
HEALTH_URL = f"http://localhost:{PORT}/health"
HEALTH_POLL_INTERVAL_S = 2

_proc: subprocess.Popen | None = None


def start_server(timeout_s: int = 180) -> subprocess.Popen:
    """Launch larql serve and block until the health endpoint returns 200."""
    global _proc

    if _proc is not None and _proc.poll() is None:
        log.info("Server already running (PID %d).", _proc.pid)
        return _proc

    cmd = [
        LARQL_BINARY,
        "serve",
        VINDEX_PATH,
        "--port", str(PORT),
        "--ffn-only",
        "--release-mmap-after-request",
    ]
    log.info("Launching: %s", " ".join(cmd))

    logfile = open(LOG_FILE, "w")  # noqa: WPS515 — intentional
    _proc = subprocess.Popen(
        cmd,
        stdout=logfile,
        stderr=subprocess.STDOUT,
    )
    log.info("Server PID: %d — stdout/stderr → %s", _proc.pid, LOG_FILE)

    atexit.register(stop_server)
    signal.signal(signal.SIGTERM, lambda *_: stop_server())

    wait_for_ready(timeout_s)
    return _proc


def wait_for_ready(timeout_s: int = 180) -> None:
    """Block until GET /health returns 200 or timeout_s is exceeded."""
    deadline = time.monotonic() + timeout_s
    log.info("Waiting for server at %s (timeout %ds)...", HEALTH_URL, timeout_s)

    while time.monotonic() < deadline:
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200:
                log.info("Server is ready.")
                return
        except requests.exceptions.ConnectionError:
            pass  # still starting up

        if _proc is not None and _proc.poll() is not None:
            raise RuntimeError(
                f"LARQL server exited prematurely (rc={_proc.returncode}). "
                f"Check {LOG_FILE} for details."
            )

        time.sleep(HEALTH_POLL_INTERVAL_S)

    raise TimeoutError(
        f"Server did not become ready within {timeout_s}s. Check {LOG_FILE}."
    )


def stop_server() -> None:
    """Terminate the child process cleanly."""
    global _proc
    if _proc is not None and _proc.poll() is None:
        log.info("Terminating LARQL server (PID %d)...", _proc.pid)
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("Server did not exit cleanly — sending SIGKILL.")
            _proc.kill()
    _proc = None


if __name__ == "__main__":
    start_server()
    log.info("Server running. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(5)
            if _proc and _proc.poll() is not None:
                log.error("Server exited unexpectedly (rc=%d).", _proc.returncode)
                sys.exit(1)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        stop_server()
