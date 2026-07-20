import pytest


def test_factory_creates_atlascloud_provider():
    from src.core.llm.factory import create_llm_provider
    from src.core.llm.providers.atlascloud import AtlasCloudProvider

    provider = create_llm_provider(
        "atlascloud",
        model="qwen/qwen3.5-flash",
        api_key="atlas-test-key",
    )

    assert isinstance(provider, AtlasCloudProvider)
    assert provider.model == "qwen/qwen3.5-flash"
    assert provider.api_endpoint == "https://api.atlascloud.ai/v1/chat/completions"
    assert provider.api_key == "atlas-test-key"


def test_factory_requires_atlascloud_api_key(monkeypatch):
    from src.core.llm.factory import create_llm_provider

    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
        create_llm_provider("atlascloud", model="qwen/qwen3.5-flash")


def test_atlascloud_provider_fallback_models_without_key():
    from src.core.llm.providers.atlascloud import AtlasCloudProvider

    provider = AtlasCloudProvider(api_key="", model="qwen/qwen3.5-flash")
    models = provider._get_fallback_models()

    model_ids = {model["id"] for model in models}
    assert "qwen/qwen3.5-flash" in model_ids
    assert "deepseek-ai/deepseek-v4-pro" in model_ids
