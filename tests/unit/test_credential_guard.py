"""Regression tests for the shared endpoint/credential guard.

The sample routes accepted a per-column endpoint, filled in the key from the
environment and skipped both the endpoint allowlist and the "a request-chosen
host must bring its own key" rule; POST /api/settings wrote the Ollama/OpenAI
endpoints to .env without any validation. Every entry point now goes through
src/api/services/credential_guard.py.
"""
import socket

import pytest
from flask import Flask

import src.config as _config
from src.api.blueprints import config_routes
from src.api.blueprints import sample_routes
from src.api.blueprints.sample_routes import create_sample_blueprint
from src.api.services import credential_guard as guard
from src.api.services import endpoint_validator as _endpoint_validator

PLACEHOLDER_KEY = 'YOUR_API_KEY_HERE'
STORED_KEY = 'sk-xxxxxxxx'
OPENAI_DEFAULT = 'https://api.openai.com/v1/chat/completions'
OLLAMA_DEFAULT = 'http://localhost:11434/api/generate'
EVIL = 'https://attacker.example.com/v1/chat/completions'
LAN_OPENAI = 'http://192.168.1.50:1234/v1/chat/completions'


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    def no_dns(*_args, **_kwargs):
        raise socket.gaierror(-2, 'Name or service not known')

    monkeypatch.setattr(socket, 'getaddrinfo', no_dns)
    _endpoint_validator._resolution_cache.clear()
    monkeypatch.setattr(_config, 'OPENAI_API_ENDPOINT', OPENAI_DEFAULT)
    monkeypatch.setattr(_config, 'OLLAMA_API_ENDPOINT', OLLAMA_DEFAULT)
    monkeypatch.setattr(_config, 'API_ENDPOINT', OLLAMA_DEFAULT)
    monkeypatch.setattr(_config, 'LLM_ENDPOINT_ALLOWLIST', ())
    monkeypatch.setenv('OPENAI_API_KEY', STORED_KEY)
    yield
    _endpoint_validator._resolution_cache.clear()


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------

def test_default_endpoint_gets_the_stored_key():
    assert guard.guard_request_credentials('openai', OPENAI_DEFAULT, '__USE_ENV__') == (STORED_KEY, None)


def test_disallowed_endpoint_is_rejected():
    key, rejection = guard.guard_request_credentials('openai', EVIL, PLACEHOLDER_KEY)
    assert key == '' and rejection.error == 'Endpoint not allowed'


def test_override_without_own_key_is_rejected_by_default():
    key, rejection = guard.guard_request_credentials('openai', LAN_OPENAI, '__USE_ENV__')
    assert key == '' and 'own API key' in rejection.error


def test_override_without_own_key_can_drop_the_stored_key():
    assert guard.guard_request_credentials(
        'openai', LAN_OPENAI, None, on_missing_own_key=guard.DROP) == ('', None)


def test_override_with_own_key_keeps_it():
    assert guard.guard_request_credentials('openai', LAN_OPENAI, PLACEHOLDER_KEY) == (PLACEHOLDER_KEY, None)


def test_endpoint_field_is_inert_for_cloud_providers(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', STORED_KEY)
    assert guard.guard_request_credentials('gemini', EVIL, '__USE_ENV__') == (STORED_KEY, None)


# ---------------------------------------------------------------------------
# Sample routes
# ---------------------------------------------------------------------------

class _NoopSampleState:
    def create(self, *args, **kwargs):
        raise AssertionError('state must not be created for a rejected request')


@pytest.fixture
def sample_client(tmp_path):
    uploads = tmp_path / 'uploads'
    uploads.mkdir()
    (uploads / 'book.txt').write_text('Some text. ' * 50, encoding='utf-8')
    app = Flask(__name__)
    app.register_blueprint(create_sample_blueprint(_NoopSampleState(), None, str(tmp_path)))
    with app.test_client() as client:
        yield client, str(uploads / 'book.txt')


def _sample_body(file_path, column):
    return {'file_path': file_path, 'file_type': 'txt', 'target_language': 'French',
            'source_language': 'English', 'columns': [column]}


@pytest.mark.parametrize('column, expected_error', [
    ({'provider': 'openai', 'model': 'm', 'api_endpoint': EVIL, 'api_key': PLACEHOLDER_KEY},
     'Endpoint not allowed'),
    ({'provider': 'openai', 'model': 'm', 'api_endpoint': LAN_OPENAI, 'api_key': '__USE_ENV__'},
     'Endpoint override requires its own API key'),
    ({'provider': 'ollama', 'model': 'm', 'endpoint': 'http://169.254.169.254.nip.io/x'},
     'Endpoint not allowed'),
])
def test_sample_run_applies_the_guard(sample_client, column, expected_error):
    client, file_path = sample_client
    resp = client.post('/api/sample/run', json=_sample_body(file_path, column))
    assert resp.status_code == 400
    assert resp.get_json()['error'] == expected_error


def test_instantiate_provider_never_pairs_stored_key_with_override(monkeypatch):
    built = {}
    monkeypatch.setattr(sample_routes, 'create_llm_provider',
                        lambda provider, **kwargs: built.update(kwargs) or object())
    with pytest.raises(ValueError):
        sample_routes._instantiate_provider(
            {'provider': 'openai', 'model': 'm', 'api_endpoint': LAN_OPENAI,
             'api_key': '__USE_ENV__'})
    assert built == {}

    sample_routes._instantiate_provider(
        {'provider': 'openai', 'model': 'm', 'api_endpoint': OPENAI_DEFAULT,
         'api_key': '__USE_ENV__'})
    assert built['api_key'] == STORED_KEY


# ---------------------------------------------------------------------------
# POST /api/settings
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_client(tmp_path, monkeypatch):
    monkeypatch.setattr(config_routes, 'get_config_path', lambda: str(tmp_path))
    monkeypatch.setattr(config_routes, 'reload_config', lambda: None)
    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint())
    with app.test_client() as client:
        yield client, tmp_path / '.env'


@pytest.mark.parametrize('setting', ['OLLAMA_API_ENDPOINT', 'OPENAI_API_ENDPOINT'])
def test_settings_reject_a_disallowed_endpoint(settings_client, setting):
    client, env_file = settings_client
    resp = client.post('/api/settings', json={setting: EVIL, 'OPENAI_API_KEY': PLACEHOLDER_KEY})
    assert resp.status_code == 400
    assert resp.get_json()['fields'] == [setting]
    assert not env_file.exists()


def test_settings_refuse_moving_the_stored_key_to_a_new_host(settings_client):
    client, env_file = settings_client
    resp = client.post('/api/settings', json={'OPENAI_API_ENDPOINT': LAN_OPENAI})
    assert resp.status_code == 400
    assert not env_file.exists()


def test_settings_accept_a_new_host_with_its_own_key(settings_client):
    client, env_file = settings_client
    resp = client.post('/api/settings', json={
        'OPENAI_API_ENDPOINT': LAN_OPENAI, 'OPENAI_API_KEY': PLACEHOLDER_KEY})
    assert resp.status_code == 200
    assert f'OPENAI_API_ENDPOINT={LAN_OPENAI}' in env_file.read_text(encoding='utf-8')


def test_settings_accept_unchanged_endpoints(settings_client):
    client, _env_file = settings_client
    resp = client.post('/api/settings', json={
        'OPENAI_API_ENDPOINT': OPENAI_DEFAULT, 'OLLAMA_API_ENDPOINT': 'http://ollama:11434/api/generate'})
    assert resp.status_code == 200
