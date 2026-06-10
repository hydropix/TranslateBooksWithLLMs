"""
Shared API-key resolution for the HTTP layer.

The frontend never sends a real key when one is configured in ``.env``: it
sends the ``__USE_ENV__`` sentinel instead (see ``ApiKeyUtils.getValue`` in
``api-key-utils.js``). Every endpoint that accepts a per-request key must turn
that sentinel — and any empty value — back into the configured key.

This logic was previously copy-pasted into four blueprints
(translation/sample/config/glossary). Diverging copies caused issue #200:
the glossary NER endpoint forwarded the literal ``__USE_ENV__`` to Gemini,
which rejected it as an invalid key. Keep the single source of truth here.
"""
import os

# Marker the frontend sends when the key field is empty but a key is
# configured in .env. Resolved back to the real env value server-side.
USE_ENV_SENTINEL = '__USE_ENV__'

# Provider -> conventional env var holding its API key.
PROVIDER_ENV_VARS = {
    'gemini': 'GEMINI_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
    'mistral': 'MISTRAL_API_KEY',
    'deepseek': 'DEEPSEEK_API_KEY',
    'poe': 'POE_API_KEY',
    'nim': 'NIM_API_KEY',
}


def provider_env_var(provider):
    """Return the env var name conventionally used for a provider's API key.

    Returns ``''`` for providers that need no key (e.g. ``ollama``) or unknown
    provider names.
    """
    return PROVIDER_ENV_VARS.get((provider or '').lower(), '')


def resolve_api_key(value, env_var_name, config_default=''):
    """Resolve a per-request API-key value to the key to actually use.

    A real key (anything truthy that isn't the ``__USE_ENV__`` sentinel) is
    returned unchanged — including multi-key, comma-separated strings, which
    provider constructors split for key rotation. Otherwise the value falls
    back to the environment variable, then to ``config_default``.

    Args:
        value: Value from the request (a real key, ``'__USE_ENV__'``, or empty).
        env_var_name: Env var to fall back to. Empty/None skips the env lookup.
        config_default: Last-resort default (e.g. the value loaded into the
            config module at import time) when the env var is unset.

    Returns:
        The resolved key string (possibly empty if nothing is configured).
    """
    if value and value != USE_ENV_SENTINEL:
        return value
    if not env_var_name:
        return config_default
    return os.getenv(env_var_name, config_default)
