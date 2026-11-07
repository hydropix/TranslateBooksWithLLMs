"""
Unit tests for listing models through the OpenAI-compatible provider when the
endpoint is Anthropic's OpenAI compatibility layer.

Anthropic serves /v1/chat/completions with a plain Bearer key, but its
/v1/models endpoint answers HTTP 400 ("anthropic-version: header is required")
without the anthropic-version header. The model picker then showed an error
even though translations worked.
"""
import sys
from pathlib import Path

import pytest
from flask import Flask

# Make the project importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.blueprints import config_routes

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/chat/completions"
LOCAL_ENDPOINT = "http://localhost:1234/v1/chat/completions"
PLACEHOLDER_KEY = "sk-xxxxxxxx"


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {"data": [{"id": "claude-model-b"}, {"id": "claude-model-a"}]}


@pytest.fixture
def captured_requests(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": dict(headers or {})})
        return _FakeResponse()

    monkeypatch.setattr(config_routes.requests, "get", fake_get)
    return calls


@pytest.fixture
def client(monkeypatch):
    # The configured endpoint is always accepted by the endpoint allowlist.
    monkeypatch.setattr(config_routes._config, "OPENAI_API_ENDPOINT", ANTHROPIC_ENDPOINT)
    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint())
    with app.test_client() as c:
        yield c


def test_anthropic_models_request_sends_version_header(client, captured_requests):
    response = client.post("/api/models", json={
        "provider": "openai",
        "api_endpoint": ANTHROPIC_ENDPOINT,
        "api_key": PLACEHOLDER_KEY,
    })

    assert response.status_code == 200
    assert len(captured_requests) == 1
    call = captured_requests[0]
    assert call["url"] == "https://api.anthropic.com/v1/models"
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["headers"]["Authorization"] == f"Bearer {PLACEHOLDER_KEY}"

    data = response.get_json()
    assert data["status"] == "openai_connected"
    assert data["model_names"] == ["claude-model-a", "claude-model-b"]


def test_other_openai_compatible_endpoints_do_not_get_version_header(client, captured_requests):
    response = client.post("/api/models", json={
        "provider": "openai",
        "api_endpoint": LOCAL_ENDPOINT,
    })

    assert response.status_code == 200
    assert len(captured_requests) == 1
    assert captured_requests[0]["url"] == "http://localhost:1234/v1/models"
    assert "anthropic-version" not in captured_requests[0]["headers"]
