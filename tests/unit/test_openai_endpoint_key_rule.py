"""The web UI's "OpenAI needs a key" rule must match the backend guard.

The backend (src/api/services/credential_guard.py) lets a request reach a
keyless OpenAI-compatible server on any host EndpointValidator accepts (LAN
hostnames and addresses included). The frontend used to demand a key unless
the endpoint string contained 'localhost' or '127.0.0.1', which blocked keyless
servers on the LAN before the request was even sent.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.api.services.credential_guard import guard_request_credentials

MODULE = (
    Path(__file__).resolve().parents[2]
    / "src" / "web" / "static" / "js" / "utils" / "openai-endpoint.js"
)

CASES = {
    "": True,
    "   ": True,
    "https://api.openai.com/v1/chat/completions": True,
    "https://API.OpenAI.com/v1/chat/completions": True,
    "http://localhost:8080/v1/chat/completions": False,
    "http://127.0.0.1:1234/v1/chat/completions": False,
    "http://192.168.1.20:1234/v1/chat/completions": False,
    "http://lmstudio:1234/v1/chat/completions": False,
    "http://ai-box.lan:11434/v1/chat/completions": False,
    "https://api.openai.com.evil.example/v1": False,
}

node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_frontend_rule_requires_key_only_for_the_official_api():
    script = (
        f"import({json.dumps(MODULE.as_uri())}).then(m => {{"
        f"const cases = {json.dumps(list(CASES))};"
        "console.log(JSON.stringify(cases.map(c => m.openaiEndpointRequiresKey(c))));"
        "});"
    )
    out = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout
    assert dict(zip(CASES, json.loads(out))) == CASES


@pytest.mark.parametrize("endpoint", [
    "http://192.168.1.20:1234/v1/chat/completions",
    "http://lmstudio:1234/v1/chat/completions",
])
def test_backend_accepts_keyless_lan_openai_endpoint(endpoint, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    api_key, rejection = guard_request_credentials("openai", endpoint, "")
    assert rejection is None
    assert api_key == ""
