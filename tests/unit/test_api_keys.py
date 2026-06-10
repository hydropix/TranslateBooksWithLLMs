"""
Unit tests for the shared API-key resolver (src/api/api_keys.py).

Locks in the behavior that four blueprints rely on, including the
'__USE_ENV__' sentinel resolution that regressed in issue #200.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.api_keys import (
    USE_ENV_SENTINEL,
    provider_env_var,
    resolve_api_key,
)


class TestProviderEnvVar:
    def test_known_providers(self):
        assert provider_env_var('gemini') == 'GEMINI_API_KEY'
        assert provider_env_var('openai') == 'OPENAI_API_KEY'
        assert provider_env_var('nim') == 'NIM_API_KEY'

    def test_case_insensitive(self):
        assert provider_env_var('Gemini') == 'GEMINI_API_KEY'
        assert provider_env_var('OPENAI') == 'OPENAI_API_KEY'

    def test_keyless_or_unknown_provider_returns_empty(self):
        assert provider_env_var('ollama') == ''
        assert provider_env_var('nonexistent') == ''
        assert provider_env_var('') == ''
        assert provider_env_var(None) == ''


class TestResolveApiKey:
    def test_real_key_is_returned_unchanged(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'}):
            assert resolve_api_key('real-key', 'GEMINI_API_KEY') == 'real-key'

    def test_multi_key_string_passes_through(self):
        # Comma-separated keys must survive intact for rotation.
        raw = 'key-a,key-b,key-c'
        assert resolve_api_key(raw, 'GEMINI_API_KEY') == raw

    def test_sentinel_resolves_to_env(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'}):
            assert resolve_api_key(USE_ENV_SENTINEL, 'GEMINI_API_KEY') == 'env-key'

    def test_empty_value_resolves_to_env(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'}):
            assert resolve_api_key('', 'GEMINI_API_KEY') == 'env-key'
            assert resolve_api_key(None, 'GEMINI_API_KEY') == 'env-key'

    def test_config_default_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_api_key(USE_ENV_SENTINEL, 'GEMINI_API_KEY', 'cfg') == 'cfg'

    def test_env_wins_over_config_default(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'}):
            assert resolve_api_key(USE_ENV_SENTINEL, 'GEMINI_API_KEY', 'cfg') == 'env-key'

    def test_no_env_var_name_uses_config_default(self):
        assert resolve_api_key(USE_ENV_SENTINEL, '', 'cfg') == 'cfg'
        assert resolve_api_key('', '', '') == ''


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
