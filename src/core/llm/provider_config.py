"""
Typed provider configuration.

``ProviderConfig`` is everything ``create_llm_provider`` needs to build one
provider: the provider type, model, endpoint, the credential for that
provider, the context window, the retry policy and the sampling parameters.
It is the provider-level object a job configuration embeds.

``provider_config_from_args`` builds one from today's loose arguments (the
seven ``*_api_key`` parameters, endpoint, model) so the upper layers can
switch to it in one line; nothing below it sees the loose parameters.

Credential rule: an empty key passed explicitly stays empty. The caller (the
web layer's credential guard in particular) may drop a key on purpose, e.g.
when a request-chosen host must bring its own key; this module never
re-fills it from the environment. ``provider_config_from_args`` only falls
back to the ``.env`` key when the caller passed ``None`` (no opinion).
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional

from .credentials import KEY_REQUIRED_PROVIDERS, env_key as _env_key  # noqa: F401
from .policy import RetryPolicy

#: Every provider type ``create_llm_provider`` can build.
SUPPORTED_PROVIDERS = (
    "ollama", "openai", "gemini", "openrouter", "mistral", "deepseek", "poe", "nim", "litellm",
)

@dataclass(frozen=True)
class ProviderConfig:
    """Configuration of one LLM provider instance.

    Attributes:
        provider: Provider type (one of ``SUPPORTED_PROVIDERS``).
        model: Model identifier. Empty means the provider's configured default.
        endpoint: API endpoint (Ollama, OpenAI-compatible, NIM). Ignored by
            providers with a fixed endpoint unless documented otherwise.
        api_key: Credential(s) for this provider (one key or "k1,k2"). None
            or "" means "no key": never re-filled from the environment here.
        context_window: Context window to request (Ollama, OpenAI-compatible).
        retry: Retry/timeout policy; None resolves the active job's policy at
            request time.
        temperature: Sampling temperature; None reads ``TEMPERATURE`` at
            request time.
        options: Provider-specific switches (``disable_thinking``,
            ``disable_web_search``, ``api_base`` for LiteLLM).
        log_callback: Optional ``(event, message)`` logging callback.
    """

    provider: str = "ollama"
    model: str = ""
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    context_window: Optional[int] = None
    retry: Optional[RetryPolicy] = None
    temperature: Optional[float] = None
    options: Dict[str, Any] = field(default_factory=dict)
    log_callback: Optional[Callable] = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "provider", (self.provider or "").strip().lower())

    def with_changes(self, **changes: Any) -> "ProviderConfig":
        return replace(self, **changes)

    def __repr__(self) -> str:  # never print the key
        return (f"ProviderConfig(provider={self.provider!r}, model={self.model!r}, "
                f"endpoint={self.endpoint!r}, api_key={'<set>' if self.api_key else None}, "
                f"context_window={self.context_window!r})")
