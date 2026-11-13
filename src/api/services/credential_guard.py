"""
Single guard for every HTTP entry point that lets the request pick an LLM host.

Two rules, applied together (see ``endpoint_validator`` for the threat model):

1. A request-supplied endpoint must pass ``EndpointValidator`` (loopback,
   private network, or an allowlisted host).
2. An endpoint that differs from the server default is an *override*, and an
   override must bring its own key: the server's stored credential never
   travels to a host the client chose.

Rule 2 is enforced by refusing the request rather than by quietly dropping the
stored key, because the provider factory falls back to the environment key on
its own when it receives none (``create_llm_provider`` for 'openai' and
'nim'). Dropping the key at the HTTP layer would therefore not keep it from
following the overridden endpoint. Only callers that use the resolved key
directly, with no factory in between, may ask for the "drop" behaviour.

Providers whose client never reads the request endpoint (Gemini, Mistral,
DeepSeek, Poe, OpenRouter, LiteLLM) are exempt from both rules: the field is
inert for them, and the frontend sends it unconditionally.

This replaces four diverging copies that lived in the translation, config,
glossary and custom-instruction blueprints, and covers the sample routes and
the settings writer that had no guard at all.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import src.config as _config
from src.api.api_keys import USE_ENV_SENTINEL, provider_env_var, resolve_api_key
from src.api.services.endpoint_validator import EndpointValidator

# Providers whose client reads the request endpoint. Source:
# src/core/llm/factory.py (ollama, openai and nim use kwargs['api_endpoint']).
ENDPOINT_CONSUMING_PROVIDERS = ('ollama', 'openai', 'nim')

# What the guard does with an override that brings no key of its own while a
# stored key exists.
REJECT = 'reject'
DROP = 'drop'


@dataclass(frozen=True)
class GuardRejection:
    """Why a request was refused; both strings are safe to return to a client."""
    error: str
    message: str

    def as_dict(self):
        return {"error": self.error, "message": self.message}


def normalize_provider(provider: Optional[str]) -> str:
    return (provider or 'ollama').strip().lower()


def normalize_endpoint(value: Optional[str]) -> str:
    """Normalize an endpoint for comparison (trim whitespace and trailing '/')."""
    return (value or '').strip().rstrip('/')


def server_default_endpoint(provider: Optional[str]) -> str:
    """The .env-configured endpoint of an endpoint-consuming provider.

    Read from src.config at call time so reload_config() is honoured.
    """
    provider = normalize_provider(provider)
    if provider == 'ollama':
        return _config.API_ENDPOINT or ''
    if provider == 'openai':
        return _config.OPENAI_API_ENDPOINT or ''
    if provider == 'nim':
        return _config.NIM_API_ENDPOINT or ''
    return ''


def is_endpoint_override(provider: Optional[str], requested_endpoint: Optional[str],
                         default_endpoint: Optional[str] = None) -> bool:
    """True when the request chose an endpoint the provider actually reads and
    that differs from the server default (or from ``default_endpoint``)."""
    provider = normalize_provider(provider)
    if provider not in ENDPOINT_CONSUMING_PROVIDERS:
        return False
    if default_endpoint is None:
        default_endpoint = server_default_endpoint(provider)
    requested = normalize_endpoint(requested_endpoint)
    return bool(requested) and requested != normalize_endpoint(default_endpoint)


def stored_key_exists(provider: Optional[str]) -> bool:
    """True when the server holds a key for ``provider``.

    python-dotenv loads .env into os.environ (and reload_config() refreshes
    it), so the environment is the single place to look.
    """
    env_var = provider_env_var(normalize_provider(provider))
    return bool(env_var and os.getenv(env_var))


def _own_key(request_key: Optional[str]) -> str:
    if request_key and request_key != USE_ENV_SENTINEL:
        return request_key
    return ''


def guard_request_credentials(
    provider: Optional[str],
    requested_endpoint: Optional[str],
    request_key: Optional[str],
    *,
    on_missing_own_key: str = REJECT,
    default_endpoint: Optional[str] = None,
) -> Tuple[str, Optional[GuardRejection]]:
    """Validate a request's endpoint and resolve the key it may use.

    Args:
        provider: Provider name from the request ('ollama' when empty).
        requested_endpoint: Endpoint the request supplied, if any.
        request_key: Key the request supplied: a real key, the '__USE_ENV__'
            sentinel, empty or None.
        on_missing_own_key: REJECT (default) refuses an override that brings
            no key while a stored key exists. DROP resolves the key to ''
            instead; only for callers that hand the key straight to an HTTP
            client, never through the provider factory.
        default_endpoint: Overrides the server default the request endpoint is
            compared against (rarely needed).

    Returns:
        ``(api_key, None)`` when the request may proceed (``api_key`` may be
        ''), or ``('', GuardRejection)`` when it must be refused.
    """
    provider = normalize_provider(provider)
    env_var = provider_env_var(provider)

    if provider in ENDPOINT_CONSUMING_PROVIDERS:
        ok, err = EndpointValidator.validate(requested_endpoint)
        if not ok:
            return '', GuardRejection("Endpoint not allowed", err)

        if is_endpoint_override(provider, requested_endpoint, default_endpoint):
            own = _own_key(request_key)
            if not own and stored_key_exists(provider) and on_missing_own_key == REJECT:
                return '', GuardRejection(
                    "Endpoint override requires its own API key",
                    f"A custom '{provider}' endpoint was supplied, so the server's "
                    "stored key will not be used. Send the key with the request or "
                    "restore the default endpoint.",
                )
            return own, None

    if not env_var:
        return '', None
    return resolve_api_key(request_key, env_var), None
