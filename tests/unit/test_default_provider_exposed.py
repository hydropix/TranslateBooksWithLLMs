"""
LLM_PROVIDER must reach the web UI through /api/config.

Without it the provider <select> always starts on its first option (Ollama)
in a browser with no saved choice, even when the .env configures another
provider, and the page reports "Ollama is not accessible". Same family as
#108 (DEFAULT_TARGET_LANGUAGE) and #153 (DEFAULT_MODEL).
"""

import importlib
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def make_client(monkeypatch):
    """Build a test client for the config blueprint with LLM_PROVIDER set."""
    from src import config

    def _make(provider):
        if provider is None:
            monkeypatch.delenv('LLM_PROVIDER', raising=False)
        else:
            monkeypatch.setenv('LLM_PROVIDER', provider)
        importlib.reload(config)

        from flask import Flask
        from src.api.blueprints.config_routes import create_config_blueprint

        app = Flask(__name__)
        app.register_blueprint(create_config_blueprint(server_session_id=12345))
        return app.test_client()

    yield _make

    monkeypatch.undo()
    importlib.reload(config)


def test_api_config_exposes_configured_provider(make_client):
    data = make_client('openai').get('/api/config').get_json()

    assert data['llm_provider'] == 'openai'


def test_api_config_provider_defaults_to_ollama(make_client):
    data = make_client(None).get('/api/config').get_json()

    assert data['llm_provider'] == 'ollama'
