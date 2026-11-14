"""Unit tests for the resume model/provider override logic (issue #183).

Covers ``JobConfig.from_checkpoint`` (which replaced the route-level
``_apply_resume_overrides``): backward compatibility (empty body), field
merging, generic API-key routing, the key/endpoint validation guards, and
refreshing ``auto_pause_on_rate_limit`` from ``src.config.DISABLE_AUTO_PAUSE``
on every resume.
"""
import pytest

import src.config as config_module
from src.core.job import JobConfig, JobConfigError

OUTPUT_DIR = 'translated_files'
ENDPOINT = 'http://localhost:11434/api/generate'


def _no_env_key(monkeypatch, var):
    """No key for this provider in the environment nor in src.config."""
    monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config_module, var, '', raising=False)


def _env_key(monkeypatch, var, value):
    monkeypatch.setenv(var, value)


def _stored(**changes):
    stored = {
        'model': 'llama3',
        'llm_provider': 'ollama',
        'llm_api_endpoint': ENDPOINT,
        'output_filename': 'book_fr.txt',
        'file_type': 'txt',
        'source_language': 'English',
        'target_language': 'French',
    }
    stored.update(changes)
    return stored


def _resume(stored, overrides=None):
    return JobConfig.from_checkpoint(stored, output_dir=OUTPUT_DIR, overrides=overrides)


def _rejected(stored, overrides=None):
    with pytest.raises(JobConfigError) as info:
        _resume(stored, overrides)
    return info.value


def test_empty_overrides_leaves_model_provider_endpoint_untouched(monkeypatch):
    """model/provider/endpoint stay a snapshot; only auto_pause is refreshed."""
    monkeypatch.setattr(config_module, 'DISABLE_AUTO_PAUSE', 'false')
    job = _resume(_stored(), {})
    assert job.model == 'llama3'
    assert job.llm_provider == 'ollama'
    assert job.endpoint == ENDPOINT
    assert job.output_filename == 'book_fr.txt'


def test_none_overrides_is_noop_for_model_provider_endpoint(monkeypatch):
    monkeypatch.setattr(config_module, 'DISABLE_AUTO_PAUSE', 'false')
    job = _resume(_stored(), None)
    assert (job.model, job.llm_provider, job.endpoint) == ('llama3', 'ollama', ENDPOINT)


def test_auto_pause_refreshed_from_live_setting_when_not_overridden(monkeypatch):
    """A job created while DISABLE_AUTO_PAUSE=false must pick up a later flip
    to true (e.g. saved via /api/settings, which already refreshes this
    module attribute) on resume, even with an empty request body."""
    stored = _stored(auto_pause_on_rate_limit=True)  # snapshot taken at job creation

    monkeypatch.setattr(config_module, 'DISABLE_AUTO_PAUSE', 'true')
    assert _resume(stored, {}).auto_pause_on_rate_limit is False

    monkeypatch.setattr(config_module, 'DISABLE_AUTO_PAUSE', 'false')
    assert _resume(stored, None).auto_pause_on_rate_limit is True


def test_auto_pause_explicit_override_wins_over_live_setting(monkeypatch):
    monkeypatch.setattr(config_module, 'DISABLE_AUTO_PAUSE', 'false')  # live setting: auto-pause ON
    job = _resume(_stored(), {'auto_pause_on_rate_limit': False})
    assert job.auto_pause_on_rate_limit is False


def test_simple_model_provider_override(monkeypatch):
    _env_key(monkeypatch, 'OPENROUTER_API_KEY', 'sk-test-not-a-real-key')
    job = _resume(_stored(), {
        'model': 'anthropic/claude-sonnet-4',
        'llm_provider': 'OpenRouter',  # case-insensitive
    })
    assert job.model == 'anthropic/claude-sonnet-4'
    assert job.llm_provider == 'openrouter'  # normalized
    assert job.provider.api_key == 'sk-test-not-a-real-key'


def test_api_key_routed_to_provider(monkeypatch):
    _no_env_key(monkeypatch, 'GEMINI_API_KEY')
    job = _resume(_stored(), {
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash',
        'api_key': 'test-key-not-real',
    })
    assert job.llm_provider == 'gemini'
    assert job.provider.api_key == 'test-key-not-real'


def test_api_key_use_env_sentinel_resolves_from_env(monkeypatch):
    _env_key(monkeypatch, 'GEMINI_API_KEY', 'env-test-key-not-real')
    job = _resume(_stored(), {
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash',
        'api_key': '__USE_ENV__',
    })
    assert job.provider.api_key == 'env-test-key-not-real'


def test_cloud_provider_without_key_is_rejected(monkeypatch):
    _no_env_key(monkeypatch, 'GEMINI_API_KEY')
    error = _rejected(_stored(), {'llm_provider': 'gemini', 'model': 'gemini-2.0-flash'})
    assert error.status == 400
    assert error.payload['error'] == 'Missing API key for provider'
    assert 'GEMINI_API_KEY' in error.payload['message']


def test_endpoint_provider_without_endpoint_is_rejected(monkeypatch):
    _env_key(monkeypatch, 'OPENAI_API_KEY', 'sk-test-not-a-real-key')
    error = _rejected(_stored(llm_api_endpoint=''), {'llm_provider': 'openai', 'model': 'gpt-4o'})
    assert error.status == 400
    assert error.payload['error'] == 'Missing API endpoint for provider'


def test_invalid_context_window_is_rejected():
    error = _rejected(_stored(), {'context_window': 'not-a-number'})
    assert error.status == 400
    assert 'context_window' in str(error)


def test_context_window_override_is_applied():
    job = _resume(_stored(context_window=2048), {'context_window': '8192'})
    assert job.context_window == 8192
    assert job.provider.context_window == 8192


def test_multi_key_string_is_preserved(monkeypatch):
    """Comma-separated keys must reach the provider unchanged for the rotation pool."""
    job = _resume(_stored(), {
        'llm_provider': 'openrouter',
        'model': 'x',
        'api_key': 'sk-test-1,sk-test-2,sk-test-3',
    })
    assert job.provider.api_key == 'sk-test-1,sk-test-2,sk-test-3'


def test_request_key_never_reaches_the_persisted_form():
    job = _resume(_stored(), {
        'llm_provider': 'openrouter', 'model': 'x', 'api_key': 'sk-test-not-a-real-key',
    })
    persisted = job.to_persisted()
    assert 'sk-test-not-a-real-key' not in repr(persisted)
    assert not [k for k in persisted if k == 'api_key' or k.endswith('_api_key')]


def test_retry_token_aligned_defaults_to_off_on_every_resume(monkeypatch):
    """A routine resume must never retranslate the approximately-tagged chunks.

    Those chunks are translated; only an explicit request from the completion
    card widens the work set to them (design decision D3). The flag is written
    on every resume, including from an empty body, so a value left in a stored
    config can never turn a plain Resume into a repair pass.
    """
    monkeypatch.setattr(config_module, 'DISABLE_AUTO_PAUSE', 'false')
    stored = _stored(retry_token_aligned=True)  # stale value from an earlier pass
    assert _resume(stored, {}).retry_token_aligned is False


def test_retry_token_aligned_opt_in_is_honoured(monkeypatch):
    monkeypatch.setattr(config_module, 'DISABLE_AUTO_PAUSE', 'false')
    assert _resume(_stored(), {'retry_token_aligned': True}).retry_token_aligned is True


def test_non_boolean_retry_token_aligned_is_rejected():
    error = _rejected(_stored(), {'retry_token_aligned': 'yes'})
    assert error.status == 400
