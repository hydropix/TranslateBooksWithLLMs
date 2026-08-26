from pathlib import Path

import re


HELPERS = Path(__file__).parents[1] / "src" / "web" / "static" / "js" / "providers" / "provider-select-helpers.js"


def provider_logos():
    text = HELPERS.read_text(encoding="utf-8")
    return dict(re.findall(r"^\s*(\w+):\s*'([^']+)'", text, re.MULTILINE))


def test_added_provider_logos_exist():
    static_root = Path(__file__).parents[1] / "src" / "web" / "static"
    for provider in ("anthropic", "xai", "opencode", "opencodego", "ollamacloud", "chatgpt"):
        asset = provider_logos()[provider].removeprefix("/static/")
        assert (static_root / asset).is_file(), f"missing logo for {provider}"


def test_xai_does_not_reuse_openai_logo():
    logos = provider_logos()
    assert logos["xai"] != logos["openai"]


def test_opencode_logos_are_distinct():
    logos = provider_logos()
    assert logos["opencode"] != logos["openai"]
    assert logos["opencodego"] != logos["openai"]
    assert logos["opencode"] != logos["opencodego"]
