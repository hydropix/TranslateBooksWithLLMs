"""Shared fixtures for the end-to-end suite.

Everything here is session-scoped and expensive: one server process, one
browser, one set of input fixtures. Individual tests get a fresh page.

Requirements, all checked at collection time so a missing one skips instead of
failing:
  - playwright installed, with its Chromium downloaded (`playwright install chromium`)
  - a Gemini API key in the environment (E2E runs cost real tokens)
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = int(os.environ.get("E2E_PORT", "5099"))
# Overridable so the suite can be pointed at a cheaper or newer model.
E2E_MODEL = os.environ.get("E2E_MODEL", "gemini-2.5-flash")
E2E_PROVIDER = "gemini"

pytest.importorskip("requests", reason="e2e tests drive the HTTP API")
sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="pip install playwright && playwright install chromium"
).sync_playwright

import requests  # noqa: E402  (after importorskip)


def _has_gemini_key():
    """True when a Gemini key is reachable, from the environment or .env."""
    if (os.environ.get("GEMINI_API_KEY") or "").strip():
        return True
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return False
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "GEMINI_API_KEY" and value.split("#")[0].strip():
            return True
    return False


def _port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


@pytest.fixture(scope="session")
def live_server(tmp_path_factory):
    """Start the web server on a private port; yield (base_url, api_token)."""
    if not _has_gemini_key():
        pytest.skip("GEMINI_API_KEY is not configured; e2e runs need a real LLM")
    if not _port_is_free(DEFAULT_PORT):
        pytest.skip(f"port {DEFAULT_PORT} is already in use; set E2E_PORT to a free one")

    work = tmp_path_factory.mktemp("e2e-server")
    token_file = work / "token.txt"
    log_file = work / "server.log"

    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tests.e2e._server_runner",
             str(DEFAULT_PORT), str(token_file)],
            cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
        )

    base_url = f"http://127.0.0.1:{DEFAULT_PORT}"
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server exited early; log:\n{log_file.read_text(errors='ignore')[-3000:]}")
            if token_file.exists():
                try:
                    if requests.get(f"{base_url}/api/health", timeout=3).ok:
                        break
                except requests.RequestException:
                    pass
            time.sleep(1)
        else:
            pytest.fail(f"server did not come up; log:\n{log_file.read_text(errors='ignore')[-3000:]}")

        yield base_url, token_file.read_text(encoding="utf-8").strip()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def api(live_server):
    """Small authenticated HTTP client for the gated /api/ routes."""
    base_url, token = live_server

    class Api:
        base = base_url
        headers = {"X-API-Token": token}

        def get(self, path, **kw):
            return requests.get(f"{base_url}{path}", headers=self.headers, timeout=30, **kw)

        def delete(self, path, **kw):
            return requests.delete(f"{base_url}{path}", headers=self.headers, timeout=30, **kw)

        def post(self, path, **kw):
            return requests.post(f"{base_url}{path}", headers=self.headers, timeout=120, **kw)

        def job_status(self, translation_id):
            r = self.get(f"/api/translation/{translation_id}")
            return r.json().get("status") if r.ok else None

        def wait_for_terminal(self, translation_id, timeout=300):
            from .helpers import TERMINAL_STATUSES
            deadline = time.time() + timeout
            while time.time() < deadline:
                status = self.job_status(translation_id)
                if status in TERMINAL_STATUSES:
                    return status
                time.sleep(2)
            raise AssertionError(f"{translation_id} did not reach a terminal status in {timeout}s")

    return Api()


@pytest.fixture(scope="session")
def input_files(tmp_path_factory):
    """Deterministic TXT inputs, sized for what each test needs."""
    from .helpers import build_input_files
    return build_input_files(tmp_path_factory.mktemp("e2e-inputs"))


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=os.environ.get("E2E_HEADED") != "1")
        yield b
        b.close()


@pytest.fixture
def context(browser):
    ctx = browser.new_context()
    yield ctx
    ctx.close()


@pytest.fixture
def page(context, live_server):
    """A page with the SPA loaded and the LLM provider/model already chosen."""
    from .helpers import open_app
    base_url, _ = live_server
    p = context.new_page()
    errors = []
    p.on("pageerror", lambda e: errors.append(str(e)))
    # resumeJob() and a few destructive actions gate on window.confirm();
    # Playwright dismisses dialogs by default, which would silently no-op them.
    p.on("dialog", lambda d: d.accept())
    open_app(p, base_url)
    yield p
    assert not errors, f"uncaught page errors: {errors}"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_generated_files():
    """Remove the output, upload and thumbnail files this session produced.

    A before/after diff rather than a name pattern, because uploads and
    thumbnails are stored under a content hash. The consequence is that a
    translation started by hand *while the suite runs* would be swept up too —
    run e2e on a quiet instance.
    """
    from src.config import OUTPUT_DIR

    output_dir = (REPO_ROOT / OUTPUT_DIR).resolve()
    watched = [output_dir, output_dir / "uploads", output_dir / "thumbnails"]

    def snapshot():
        found = set()
        for directory in watched:
            if directory.is_dir():
                found |= {p for p in directory.iterdir() if p.is_file()}
        return found

    before = snapshot()
    yield
    for path in snapshot() - before:
        try:
            path.unlink()
        except OSError:
            pass


@pytest.fixture(autouse=True)
def _cleanup_jobs(request, api):
    """Delete any checkpoint an e2e test leaves behind.

    Only jobs created during the test are removed, so a developer's own
    in-flight job in the same database is never touched.
    """
    if "api" not in request.fixturenames:
        yield
        return

    def ids():
        r = api.get("/api/resumable")
        if not r.ok:
            return set()
        return {j.get("translation_id") for j in r.json().get("resumable_jobs", [])}

    before = ids()
    yield
    for translation_id in ids() - before:
        api.delete(f"/api/checkpoint/{translation_id}")
