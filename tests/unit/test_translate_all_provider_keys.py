"""Regression test: /api/translate must keep the key of every cloud provider.

The route used to build the job config with only the Gemini, OpenAI and
OpenRouter keys, while the frontend sends seven and the job handler reads
mistral/deepseek/poe/nim keys from the config: a key typed in the UI for one of
those four providers was silently dropped.
"""
import pytest
from flask import Flask

from src.api.api_keys import PROVIDER_ENV_VARS
from src.api.blueprints.translation_routes import create_translation_blueprint

# Inert placeholder, never a real credential.
PLACEHOLDER_KEY = 'YOUR_API_KEY_HERE'


class _RecordingStateManager:
    def __init__(self):
        self.created = []

    def create_translation(self, translation_id, config):
        self.created.append((translation_id, dict(config)))


@pytest.fixture
def translate_app(tmp_path):
    state_manager = _RecordingStateManager()
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state_manager, lambda translation_id, config: None, str(tmp_path),
    ))
    with app.test_client() as client:
        yield client, state_manager


def _payload(provider, **overrides):
    body = {
        'text': 'Hello world.',
        'source_language': 'English',
        'target_language': 'French',
        'model': 'some-model',
        # Inert for cloud providers that do not read the request endpoint.
        'llm_api_endpoint': 'http://localhost:11434/api/generate',
        'output_filename': 'out.txt',
        'llm_provider': provider,
    }
    body.update(overrides)
    return body


@pytest.mark.parametrize('provider', ['mistral', 'deepseek', 'poe', 'nim',
                                      'gemini', 'openrouter'])
def test_key_typed_in_ui_reaches_job_config(translate_app, monkeypatch, provider):
    client, state_manager = translate_app
    monkeypatch.delenv(PROVIDER_ENV_VARS[provider], raising=False)
    # The NIM client reads the request endpoint; send its default so the
    # pairing rule does not treat it as an override.
    overrides = {f'{provider}_api_key': PLACEHOLDER_KEY}
    if provider == 'nim':
        import src.config as _config
        overrides['llm_api_endpoint'] = _config.NIM_API_ENDPOINT

    resp = client.post('/api/translate', json=_payload(provider, **overrides))

    assert resp.status_code == 200, resp.get_json()
    _tid, config = state_manager.created[0]
    assert config[f'{provider}_api_key'] == PLACEHOLDER_KEY


def test_every_provider_key_falls_back_to_env(translate_app, monkeypatch):
    client, state_manager = translate_app
    for env_var in PROVIDER_ENV_VARS.values():
        monkeypatch.setenv(env_var, f'env-{env_var.lower()}')

    resp = client.post('/api/translate', json=_payload(
        'ollama', **{f'{p}_api_key': '__USE_ENV__' for p in PROVIDER_ENV_VARS}))

    assert resp.status_code == 200
    _tid, config = state_manager.created[0]
    for provider, env_var in PROVIDER_ENV_VARS.items():
        assert config[f'{provider}_api_key'] == f'env-{env_var.lower()}'
    # Keys are never echoed back to the browser.
    assert not any(k.endswith('_api_key') for k in resp.get_json()['config_received'])
