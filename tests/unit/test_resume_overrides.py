"""Unit tests for the resume model/provider override logic (issue #183).

Covers `_apply_resume_overrides`: backward compatibility (empty body), field
merging, generic API-key routing through `_resolve_api_key`, the
key/endpoint validation guards, and refreshing `auto_pause_on_rate_limit`
from `src.config.DISABLE_AUTO_PAUSE` on every resume.
"""
import pytest
from flask import Flask

import src.api.blueprints.translation_routes as translation_routes
from src.api.blueprints.translation_routes import _apply_resume_overrides


@pytest.fixture
def app_ctx():
    """`_apply_resume_overrides` calls jsonify on failure, which needs a context."""
    app = Flask(__name__)
    with app.app_context():
        yield


def _base_config():
    return {
        'model': 'llama3',
        'llm_provider': 'ollama',
        'llm_api_endpoint': 'http://localhost:11434/api/generate',
    }


def test_empty_overrides_leaves_model_provider_endpoint_untouched(app_ctx, monkeypatch):
    """model/provider/endpoint stay a snapshot; only auto_pause is refreshed."""
    monkeypatch.setattr(translation_routes._config, 'DISABLE_AUTO_PAUSE', 'false')
    config = _base_config()
    snapshot = {k: v for k, v in config.items()}
    assert _apply_resume_overrides(config, {}) is None
    for key, value in snapshot.items():
        assert config[key] == value


def test_none_overrides_is_noop_for_model_provider_endpoint(app_ctx, monkeypatch):
    monkeypatch.setattr(translation_routes._config, 'DISABLE_AUTO_PAUSE', 'false')
    config = _base_config()
    snapshot = {k: v for k, v in config.items()}
    assert _apply_resume_overrides(config, None) is None
    for key, value in snapshot.items():
        assert config[key] == value


def test_auto_pause_refreshed_from_live_setting_when_not_overridden(app_ctx, monkeypatch):
    """A job created while DISABLE_AUTO_PAUSE=false must pick up a later flip
    to true (e.g. saved via /api/settings, which already refreshes this
    module attribute) on resume, even with an empty request body."""
    config = _base_config()
    config['auto_pause_on_rate_limit'] = True  # snapshot taken at job creation

    monkeypatch.setattr(translation_routes._config, 'DISABLE_AUTO_PAUSE', 'true')
    assert _apply_resume_overrides(config, {}) is None
    assert config['auto_pause_on_rate_limit'] is False

    monkeypatch.setattr(translation_routes._config, 'DISABLE_AUTO_PAUSE', 'false')
    assert _apply_resume_overrides(config, None) is None
    assert config['auto_pause_on_rate_limit'] is True


def test_auto_pause_explicit_override_wins_over_live_setting(app_ctx, monkeypatch):
    monkeypatch.setattr(translation_routes._config, 'DISABLE_AUTO_PAUSE', 'false')  # live setting says auto-pause ON
    config = _base_config()
    err = _apply_resume_overrides(config, {'auto_pause_on_rate_limit': False})
    assert err is None
    assert config['auto_pause_on_rate_limit'] is False


def test_simple_model_provider_override(app_ctx, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-from-env')
    config = _base_config()
    err = _apply_resume_overrides(config, {
        'model': 'anthropic/claude-sonnet-4',
        'llm_provider': 'OpenRouter',  # case-insensitive
    })
    assert err is None
    assert config['model'] == 'anthropic/claude-sonnet-4'
    assert config['llm_provider'] == 'openrouter'  # normalized


def test_api_key_routed_to_provider_field(app_ctx):
    config = _base_config()
    err = _apply_resume_overrides(config, {
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash',
        'api_key': 'real-key-123',
    })
    assert err is None
    assert config['gemini_api_key'] == 'real-key-123'


def test_api_key_use_env_sentinel_resolves_from_env(app_ctx, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'env-gemini-key')
    config = _base_config()
    err = _apply_resume_overrides(config, {
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash',
        'api_key': '__USE_ENV__',
    })
    assert err is None
    assert config['gemini_api_key'] == 'env-gemini-key'


def test_cloud_provider_without_key_is_rejected(app_ctx, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    config = _base_config()
    result = _apply_resume_overrides(config, {
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash',
    })
    assert result is not None
    _response, status = result
    assert status == 400


def test_endpoint_provider_without_endpoint_is_rejected(app_ctx, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-openai')
    config = _base_config()
    config['llm_api_endpoint'] = ''  # cleared
    result = _apply_resume_overrides(config, {
        'llm_provider': 'openai',
        'model': 'gpt-4o',
    })
    assert result is not None
    _response, status = result
    assert status == 400


def test_invalid_context_window_is_rejected(app_ctx):
    config = _base_config()
    result = _apply_resume_overrides(config, {'context_window': 'not-a-number'})
    assert result is not None
    _response, status = result
    assert status == 400


def test_multi_key_string_is_preserved(app_ctx):
    """Comma-separated keys must reach the config unchanged for the rotation pool."""
    config = _base_config()
    err = _apply_resume_overrides(config, {
        'llm_provider': 'openrouter',
        'model': 'x',
        'api_key': 'sk-or-1,sk-or-2,sk-or-3',
    })
    assert err is None
    assert config['openrouter_api_key'] == 'sk-or-1,sk-or-2,sk-or-3'


def test_retry_token_aligned_defaults_to_off_on_every_resume(app_ctx, monkeypatch):
    """A routine resume must never retranslate the approximately-tagged chunks.

    Those chunks are translated; only an explicit request from the completion
    card widens the work set to them (design decision D3). The flag is written
    on every resume, including from an empty body, so a value left in a stored
    config can never turn a plain Resume into a repair pass.
    """
    monkeypatch.setattr(translation_routes._config, 'DISABLE_AUTO_PAUSE', 'false')
    config = _base_config()
    config['retry_token_aligned'] = True  # stale value from an earlier pass

    assert _apply_resume_overrides(config, {}) is None
    assert config['retry_token_aligned'] is False


def test_retry_token_aligned_opt_in_is_honoured(app_ctx, monkeypatch):
    monkeypatch.setattr(translation_routes._config, 'DISABLE_AUTO_PAUSE', 'false')
    config = _base_config()
    assert _apply_resume_overrides(config, {'retry_token_aligned': True}) is None
    assert config['retry_token_aligned'] is True


def test_non_boolean_retry_token_aligned_is_rejected(app_ctx):
    config = _base_config()
    result = _apply_resume_overrides(config, {'retry_token_aligned': 'yes'})
    assert result is not None
    _response, status = result
    assert status == 400
