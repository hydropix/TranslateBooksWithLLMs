"""
/api/config and /api/settings derive their per-provider key fields from the
single credential table, and /api/model/warning reports failures with a 500.
"""

import pytest
from flask import Flask

import src.config as config
from src.core.llm.credentials import PROVIDER_KEY_SETTINGS

PROVIDERS = ('gemini', 'openai', 'openrouter', 'mistral', 'deepseek', 'poe', 'nim', 'llmtr')


@pytest.fixture
def client(monkeypatch):
    # Inert placeholder keys only: no real credential is ever used here.
    for setting in PROVIDER_KEY_SETTINGS.values():
        monkeypatch.setattr(config, setting, '', raising=False)
    monkeypatch.setattr(config, 'GEMINI_API_KEY', 'placeholder-aaaa1111,placeholder-bbbb2222')
    monkeypatch.setattr(config, 'POE_API_KEY', 'placeholder-cccc3333')

    from src.api.blueprints.config_routes import create_config_blueprint

    app = Flask(__name__)
    app.register_blueprint(create_config_blueprint(server_session_id=12345))
    return app.test_client()


def test_table_covers_the_key_providers():
    assert tuple(PROVIDER_KEY_SETTINGS) == PROVIDERS


def test_api_config_key_fields(client):
    data = client.get('/api/config').get_json()

    for provider in PROVIDERS:
        assert f'{provider}_api_key' in data
        assert f'{provider}_api_key_count' in data
        assert f'{provider}_api_key_configured' in data

    assert data['gemini_api_key'] == '***2222'
    assert data['gemini_api_key_count'] == 2
    assert data['gemini_api_key_configured'] is True
    assert data['poe_api_key'] == '***3333'
    assert data['poe_api_key_count'] == 1
    assert data['openai_api_key'] == ''
    assert data['openai_api_key_count'] == 0
    assert data['openai_api_key_configured'] is False


def test_api_settings_key_fields(client):
    data = client.get('/api/settings').get_json()

    for provider in PROVIDERS:
        assert data[f'{provider}_api_key_configured'] is (provider in ('gemini', 'poe'))


def test_model_warning_failure_returns_500(client, monkeypatch):
    import src.core.llm as llm

    def boom(*args, **kwargs):
        raise RuntimeError('lookup failed')

    monkeypatch.setattr(llm, 'get_model_warning_message', boom)

    response = client.get('/api/model/warning?model=some-model')

    assert response.status_code == 500
    body = response.get_json()
    assert body['warning'] is None
    assert body['behavior'] is None
    assert body['error'] == 'lookup failed'


def test_model_warning_without_model_is_ok(client):
    response = client.get('/api/model/warning')

    assert response.status_code == 200
    assert response.get_json() == {'warning': None, 'behavior': None}
