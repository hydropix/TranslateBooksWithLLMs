"""Integration-level regression tests for the Opencode provider wiring.

These cover the two findings from review that the adapter-level tests cannot:
the settings endpoint allow-list (B1) and the ``create_llm_client`` kwargs
forwarding used by the EPUB/refine paths (M1).
"""

from src.api.blueprints.config_routes import SETTINGS_ALLOWED_KEYS
from src.core.llm_client import create_llm_client


def test_settings_allowlist_accepts_opencode_key_and_model():
    """`/api/settings` must not silently drop the provider's key/model."""
    assert "OPENCODE_API_KEY" in SETTINGS_ALLOWED_KEYS
    assert "OPENCODE_MODEL" in SETTINGS_ALLOWED_KEYS


def test_settings_allowlist_keeps_endpoint_out_of_ui_editable_keys():
    """Endpoint is startup-configured, like the other cloud providers."""
    assert "OPENCODE_API_ENDPOINT" not in SETTINGS_ALLOWED_KEYS


def test_create_llm_client_forwards_context_window_and_log_callback():
    def log_callback(*_args, **_kwargs):
        pass

    client = create_llm_client(
        "opencode",
        None,
        "http://endpoint.invalid",
        "m",
        opencode_api_key="ck",
        context_window=9999,
        log_callback=log_callback,
    )

    assert client is not None
    assert client.provider_kwargs.get("context_window") == 9999
    assert client.provider_kwargs.get("log_callback") is log_callback


def test_opencode_has_no_default_pricing():
    """Intentional: no real OpenCode prices are known, so cost is unknown."""
    from src.core.pricing.pricing_data import get_default_pricing

    assert get_default_pricing("opencode", "opencode-go/deepseek-v4.1-flash") is None
