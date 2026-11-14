"""TranslationConfig defaults follow the live src.config values.

Its defaults used to be frozen at import, so reload_config() (after the web UI
saved .env) never reached a new instance.
"""

import dataclasses

import src.config as config
from src.config import TranslationConfig


def test_monkeypatched_setting_reaches_a_new_instance(monkeypatch):
    monkeypatch.setattr(config, 'MAX_TOKENS_PER_CHUNK', 1234)
    monkeypatch.setattr(config, 'SOFT_LIMIT_RATIO', 0.42)
    monkeypatch.setattr(config, 'PARALLEL_TRANSLATIONS', 3)

    cfg = TranslationConfig()

    assert cfg.max_tokens_per_chunk == 1234
    assert cfg.soft_limit_ratio == 0.42
    assert cfg.parallel_workers == 3


def test_reload_config_reaches_a_new_instance(monkeypatch):
    # Snapshot every reloadable global so monkeypatch restores them, and keep
    # the developer's own .env out of the reload.
    for attr, _env, _default in config._RELOADABLE_ENV_SETTINGS:
        monkeypatch.setattr(config, attr, getattr(config, attr))
    monkeypatch.setattr(config, 'load_dotenv', lambda *args, **kwargs: False)
    monkeypatch.setenv('DEFAULT_MODEL', 'reloaded-model')
    monkeypatch.setenv('LLM_PROVIDER', 'mistral')

    config.reload_config()
    cfg = TranslationConfig()

    assert cfg.model == 'reloaded-model'
    assert cfg.llm_provider == 'mistral'


def test_explicit_values_still_win(monkeypatch):
    monkeypatch.setattr(config, 'MAX_TOKENS_PER_CHUNK', 1234)

    assert TranslationConfig(max_tokens_per_chunk=99).max_tokens_per_chunk == 99


def test_no_credentials_on_translation_config():
    names = {f.name for f in dataclasses.fields(TranslationConfig)}
    assert not any(name.endswith('_api_key') for name in names)
    assert not any(key.endswith('_api_key') for key in TranslationConfig().to_dict())
