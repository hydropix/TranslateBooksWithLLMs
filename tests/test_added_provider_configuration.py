from pathlib import Path

from src.config import (
    ANTHROPIC_API_ENDPOINT,
    OPENCODE_API_ENDPOINT,
    OPENCODE_GO_API_ENDPOINT,
    OLLAMA_CLOUD_API_ENDPOINT,
    XAI_API_ENDPOINT,
    provider_default_endpoint,
)
from src.core.llm.providers.anthropic import AnthropicProvider
from src.core.llm.providers.ollama_cloud import OllamaCloudProvider
from src.core.llm.providers.opencode import OpenCodeGoProvider, OpenCodeProvider
from src.core.llm.providers.xai import XAIProvider


def test_added_provider_endpoints_and_fallback_models():
    assert ANTHROPIC_API_ENDPOINT == "https://api.anthropic.com/v1"
    assert XAI_API_ENDPOINT == "https://api.x.ai/v1"
    assert OPENCODE_API_ENDPOINT == "https://opencode.ai/zen/v1"
    assert OPENCODE_GO_API_ENDPOINT == "https://opencode.ai/zen/go/v1"
    assert OLLAMA_CLOUD_API_ENDPOINT == "https://ollama.com/v1"
    assert AnthropicProvider.API_URL.endswith("/v1/messages")
    assert "grok-4.5" in XAIProvider.FALLBACK_MODELS
    assert OpenCodeProvider.FALLBACK_MODELS[0] == "deepseek-v4-flash"
    assert OpenCodeGoProvider.FALLBACK_MODELS[0] == "deepseek-v4-pro"
    assert OllamaCloudProvider.FALLBACK_MODELS[0] == "gpt-oss:120b"
    assert OpenCodeProvider.DEFAULT_MAX_OUTPUT_TOKENS == 32768
    assert OpenCodeGoProvider.DEFAULT_MAX_OUTPUT_TOKENS == 32768


def test_opencode_default_endpoints_are_not_ollama():
    assert provider_default_endpoint("opencode") == OPENCODE_API_ENDPOINT
    assert provider_default_endpoint("opencodego") == OPENCODE_GO_API_ENDPOINT
    assert provider_default_endpoint("ollamacloud") == OLLAMA_CLOUD_API_ENDPOINT
    assert provider_default_endpoint("opencode") != provider_default_endpoint("ollama")
    assert provider_default_endpoint("opencodego") != provider_default_endpoint("ollama")
    assert provider_default_endpoint("ollamacloud") != provider_default_endpoint("ollama")


def test_ui_opencode_fallback_contains_all_zen_models():
    source = (Path(__file__).parents[1] / "src/web/static/js/providers/provider-manager.js").read_text(encoding="utf-8")
    assert "loadGenericCloudModels(provider)" in source
    assert "deepseek-v4-flash" in source
    assert "deepseek-v4-pro" in source
    assert "'opencode'" in source
    assert "'opencodego'" in source
    assert "'ollamacloud'" in source
    assert "loadChatGPTModels" in source
    assert "geminiSettings.style.display = 'none'" in source.split("['anthropic', 'xai', 'opencode', 'opencodego', 'ollamacloud']")[1]
    assert "data.status === 'api_key_missing'" in source


def test_frontend_validates_added_provider_keys():
    source = (
        Path(__file__).parents[1] / "src/web/static/js/utils/api-key-utils.js"
    ).read_text(encoding="utf-8")
    assert "anthropic" in source
    assert "opencodego" in source
    assert "ollamacloud" in source
    assert "chatgpt" in source
    assert "errors:api_key_required" in source
    assert "chatgpt_sign_in_to_load" in source


