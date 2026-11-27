"""
The one credential table and key resolver.

Every provider that takes an API key is listed once here, with the name of
the ``.env`` / ``src.config`` setting that holds it. The provider factory,
``ProviderConfig``, the HTTP layer (``src/api/api_keys.py`` re-exports this
module) and the job configuration builders all resolve keys through it; there
is no second provider/env-var table anywhere.

Resolution rules (unchanged from the four copies this replaces):

* A real key from a request (anything truthy that is not the ``__USE_ENV__``
  sentinel) is used as is, including a comma-separated multi-key string,
  which the providers split for rotation.
* Otherwise the environment variable is used, then the value loaded into
  ``src.config``, unless the caller forbids the environment fallback: a
  request that chose its own host must bring its own key
  (``src/api/services/credential_guard.py``).
* An explicit empty key handed to the provider factory stays empty. Only
  ``None`` ("no opinion") is re-filled from the environment, by
  :func:`env_key`.
"""

import os
from typing import Optional

#: Marker the frontend sends when its key field is empty but a key is
#: configured in ``.env``. Resolved back to the real value server-side.
USE_ENV_SENTINEL = '__USE_ENV__'

#: Provider -> setting (env var and ``src.config`` attribute) holding its key.
PROVIDER_KEY_SETTINGS = {
    'gemini': 'GEMINI_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
    'mistral': 'MISTRAL_API_KEY',
    'deepseek': 'DEEPSEEK_API_KEY',
    'poe': 'POE_API_KEY',
    'nim': 'NIM_API_KEY',
    'llmtr': 'LLMTR_API_KEY',
}

#: Providers that hold a key in ``PROVIDER_KEY_SETTINGS``, in display order.
KEY_PROVIDERS = tuple(PROVIDER_KEY_SETTINGS)

#: Providers that cannot work without an API key. ``openai`` is absent on
#: purpose: OpenAI-compatible local servers (llama.cpp, LM Studio, vLLM) need
#: none; see :func:`key_required`.
KEY_REQUIRED_PROVIDERS = frozenset({'gemini', 'openrouter', 'mistral', 'deepseek', 'poe', 'nim', 'llmtr'})


def normalize_provider(provider: Optional[str]) -> str:
    """Lower-cased provider name, ``'ollama'`` when empty."""
    return (provider or 'ollama').strip().lower()


def provider_env_var(provider: Optional[str]) -> str:
    """The env var holding ``provider``'s key, or ``''`` when it takes none."""
    return PROVIDER_KEY_SETTINGS.get((provider or '').strip().lower(), '')


def key_required(provider: Optional[str], endpoint: Optional[str] = None) -> bool:
    """True when ``provider`` cannot run without a key.

    ``openai`` only needs one for the official API, mirroring the factory.
    """
    provider = normalize_provider(provider)
    if provider == 'openai':
        return 'api.openai.com' in (endpoint or '')
    return provider in KEY_REQUIRED_PROVIDERS


def env_key(provider: Optional[str]) -> str:
    """The server's stored key for ``provider``, read at call time."""
    setting = provider_env_var(provider)
    if not setting:
        return ''
    from src import config
    return os.getenv(setting) or getattr(config, setting, '') or ''


def resolve_api_key(value, env_var_name, config_default='', allow_env_fallback=True):
    """Resolve a per-request API-key value to the key to actually use.

    Args:
        value: Value from the request (a real key, ``'__USE_ENV__'``, or empty).
        env_var_name: Env var to fall back to. Empty/None skips the env lookup.
        config_default: Last-resort default when the env var is unset.
        allow_env_fallback: When ``False``, an absent or sentinel value resolves
            to ``''`` instead of consulting the environment.

    Returns:
        The resolved key string (possibly empty if nothing is configured).
    """
    if value and value != USE_ENV_SENTINEL:
        return value
    if not allow_env_fallback:
        return ''
    if not env_var_name:
        return config_default
    return os.getenv(env_var_name, config_default)


def resolve_provider_key(provider: Optional[str], request_value=None, *,
                         allow_env_fallback: bool = True) -> str:
    """Resolve the key ``provider`` should use for one job or request.

    ``request_value`` is what the client sent (a key, the sentinel, empty or
    None). Returns ``''`` for providers that take no key.
    """
    setting = provider_env_var(provider)
    if not setting:
        return ''
    if request_value and request_value != USE_ENV_SENTINEL:
        return request_value
    if not allow_env_fallback:
        return ''
    return env_key(provider)
