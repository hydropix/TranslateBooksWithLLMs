"""Frontend job payloads must send UI-entered cloud API keys.

A key typed only in the form (not stored in .env) used to be dropped for
Anthropic / xAI / OpenCode. Failed generate() calls then kept the source text.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "src" / "web" / "static" / "js" / "translation" / "batch-controller.js"
FORM = ROOT / "src" / "web" / "static" / "js" / "ui" / "form-manager.js"

CLOUD_KEY_FIELDS = (
    "anthropic_api_key",
    "xai_api_key",
    "opencode_api_key",
    "opencodego_api_key",
    "ollamacloud_api_key",
)


def test_batch_controller_forwards_cloud_api_keys():
    source = BATCH.read_text(encoding="utf-8")
    for field in CLOUD_KEY_FIELDS:
        assert f"{field}:" in source, f"batch-controller missing {field}"
        assert "ApiKeyUtils.getValue" in source


def test_form_manager_forwards_cloud_api_keys():
    source = FORM.read_text(encoding="utf-8")
    for field in CLOUD_KEY_FIELDS:
        assert f"{field}:" in source, f"form-manager missing {field}"


def test_batch_controller_forwards_chunk_size():
    source = BATCH.read_text(encoding="utf-8")
    assert "max_tokens_per_chunk:" in source
    assert "maxTokensPerChunk" in source


def test_form_manager_forwards_chunk_size():
    source = FORM.read_text(encoding="utf-8")
    assert "max_tokens_per_chunk:" in source
    assert "maxTokensPerChunk" in source


def test_settings_dirty_tracks_opencode_and_chunk_size():
    settings = (ROOT / "src" / "web" / "static" / "js" / "core" / "settings-manager.js").read_text(
        encoding="utf-8"
    )
    for field_id in (
        "opencodeApiKey",
        "anthropicApiKey",
        "xaiApiKey",
        "opencodegoApiKey",
        "ollamacloudApiKey",
        "maxTokensPerChunk",
    ):
        assert f"id: '{field_id}'" in settings


def test_form_manager_applies_saved_provider():
    source = FORM.read_text(encoding="utf-8")
    assert "config.llm_provider" in source
    assert "providerSelect.value = config.llm_provider" in source


def test_settings_save_uses_env_key_sentinel():
    settings = (ROOT / "src" / "web" / "static" / "js" / "core" / "settings-manager.js").read_text(
        encoding="utf-8"
    )
    assert "ApiKeyUtils.getValue('opencodeApiKey')" in settings
    assert "from '../utils/api-key-utils.js'" in settings


def test_translate_route_forwards_chunk_size():
    routes = (ROOT / "src" / "api" / "blueprints" / "translation_routes.py").read_text(
        encoding="utf-8"
    )
    assert "config['max_tokens_per_chunk']" in routes
    assert "max_tokens_per_chunk" in routes


def test_stale_api_token_triggers_reload():
    client = (ROOT / "src" / "web" / "static" / "js" / "core" / "api-client.js").read_text(
        encoding="utf-8"
    )
    assert "chatgptOAuth" in client or "chatgpt" in client.lower()
    lifecycle = (ROOT / "src" / "web" / "static" / "js" / "utils" / "lifecycle-manager.js").read_text(
        encoding="utf-8"
    )
    assert "LifecycleManager" in lifecycle


def test_frontend_refreshes_glossaries_on_auto_save_event():
    glossary = (ROOT / "src" / "web" / "static" / "js" / "glossary" / "glossary-manager.js").read_text(
        encoding="utf-8"
    )
    assert "refreshDropdown" in glossary

