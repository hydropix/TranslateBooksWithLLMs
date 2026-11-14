"""
API-key resolution for the HTTP layer.

The frontend never sends a real key when one is configured in ``.env``: it
sends the ``__USE_ENV__`` sentinel instead (see ``ApiKeyUtils.getValue`` in
``api-key-utils.js``). Every endpoint that accepts a per-request key must turn
that sentinel, and any empty value, back into the configured key.

The table and the resolver live in ``src/core/llm/credentials.py``, the single
source of truth shared with the provider factory and the job configuration
builders (diverging copies caused issue #200). This module re-exports them
under the names the blueprints already import.
"""
from src.core.llm.credentials import (  # noqa: F401 - re-exported API
    KEY_PROVIDERS,
    PROVIDER_KEY_SETTINGS as PROVIDER_ENV_VARS,
    USE_ENV_SENTINEL,
    provider_env_var,
    resolve_api_key,
    resolve_provider_key,
)
