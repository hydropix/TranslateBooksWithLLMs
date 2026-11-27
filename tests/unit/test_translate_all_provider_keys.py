"""Regression test: /api/translate must keep the key of every cloud provider.

The route used to build the job config with only the Gemini, OpenAI and
OpenRouter keys, while the frontend sends seven and the job handler reads
mistral/deepseek/poe/nim keys from the config: a key typed in the UI for one of
those four providers was silently dropped.

The route now builds a ``JobConfig`` whose provider carries the selected
provider's key in memory only; the state registry gets the secret-free
``to_persisted()`` form.
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
    started = []
    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state_manager, lambda translation_id, job: started.append(job), str(tmp_path),
    ))
    with app.test_client() as client:
        yield client, state_manager, started


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
    if provider == 'nim':
        # The NIM client reads the request endpoint; send its default so the
        # pairing rule does not treat it as an override.
        import src.config as _config
        body['llm_api_endpoint'] = _config.NIM_API_ENDPOINT
    body.update(overrides)
    return body


def _assert_no_key_persisted(config):
    assert not [k for k in config if k == 'api_key' or k.endswith('_api_key')]


CLOUD_PROVIDERS = ['mistral', 'deepseek', 'poe', 'nim', 'llmtr', 'gemini', 'openrouter']


@pytest.mark.parametrize('provider', CLOUD_PROVIDERS)
def test_key_typed_in_ui_reaches_the_job(translate_app, monkeypatch, provider):
    client, state_manager, started = translate_app
    monkeypatch.delenv(PROVIDER_ENV_VARS[provider], raising=False)

    resp = client.post('/api/translate', json=_payload(
        provider, **{f'{provider}_api_key': PLACEHOLDER_KEY}))

    assert resp.status_code == 200, resp.get_json()
    (job,) = started
    assert job.llm_provider == provider
    assert job.provider.api_key == PLACEHOLDER_KEY
    # The key lives in the in-memory job only, never in the registry config.
    _tid, config = state_manager.created[0]
    _assert_no_key_persisted(config)
    assert PLACEHOLDER_KEY not in repr(config)


@pytest.mark.parametrize('provider', CLOUD_PROVIDERS)
def test_provider_key_falls_back_to_env(translate_app, monkeypatch, provider):
    client, state_manager, started = translate_app
    env_var = PROVIDER_ENV_VARS[provider]
    monkeypatch.setenv(env_var, f'env-{env_var.lower()}')

    resp = client.post('/api/translate', json=_payload(
        provider, **{f'{p}_api_key': '__USE_ENV__' for p in PROVIDER_ENV_VARS}))

    assert resp.status_code == 200, resp.get_json()
    (job,) = started
    assert job.provider.api_key == f'env-{env_var.lower()}'
    _tid, config = state_manager.created[0]
    _assert_no_key_persisted(config)
    # Keys are never echoed back to the browser.
    assert not any(k.endswith('_api_key') for k in resp.get_json()['config_received'])
