"""
Unit tests for `_apply_cli_auto_prep` in `translate.py` (Phase 4 of
plan/PLAN_AutoGlossaryStyle.md — CLI parity for auto glossary / auto style).

Fully offline: `translate.create_provider_from_args` and
`translate.build_auto_prompt_options` are monkeypatched, and
`translate.auto_prep.extract_source_text` is stubbed so no file I/O or
network call happens. `args` is a plain `argparse.Namespace` built by
`_make_args`, matching the attributes `translate.py` reads off the real
parsed CLI arguments.
"""
import argparse

import pytest

import translate


class FakeLogger:
    """Records `.info` / `.warning` calls, mirroring the unified CLI logger."""

    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg)


class FakeClient:
    """Duck-typed stand-in for `LLMClient` — only `close()` is ever awaited."""

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def _make_args(**overrides):
    defaults = dict(
        auto_glossary=False,
        auto_style=False,
        glossary=None,
        refine_only=False,
        input="book.txt",
        source_lang="English",
        target_lang="French",
        provider="ollama",
        gemini_api_key="",
        api_endpoint="http://localhost:11434",
        model="llama3",
        openai_api_key="",
        openrouter_api_key="",
        mistral_api_key="",
        deepseek_api_key="",
        poe_api_key="",
        nim_api_key="",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def _fake_extract_source_text(monkeypatch):
    """No file ever touches disk in this test module."""
    monkeypatch.setattr(
        translate.auto_prep,
        "extract_source_text",
        lambda **kwargs: "Some source text about a wizard.",
    )


def test_auto_glossary_skipped_when_glossary_given(monkeypatch):
    """--auto-glossary --glossary f.json --auto-style: glossary is dropped,
    exactly one warning is logged, and the still-enabled style pass receives
    want_glossary=False."""
    args = _make_args(auto_glossary=True, auto_style=True, glossary="f.json")
    logger = FakeLogger()
    prompt_options = {}
    captured = {}
    client = FakeClient()

    async def fake_build(**kwargs):
        captured.update(kwargs)
        return {"custom_instructions": "be terse"}

    monkeypatch.setattr(translate, "create_provider_from_args", lambda *a, **kw: client)
    monkeypatch.setattr(translate, "build_auto_prompt_options", fake_build)

    translate._apply_cli_auto_prep(args, prompt_options, logger)

    assert len(logger.warnings) == 1
    assert "--auto-glossary ignored: --glossary was provided." in logger.warnings[0]
    assert captured["want_glossary"] is False
    assert captured["want_style"] is True
    assert prompt_options == {"custom_instructions": "be terse"}
    assert client.closed is True


def test_auto_glossary_skipped_in_refine_only_mode(monkeypatch):
    """--auto-glossary --refine-only --auto-style: glossary is dropped with one
    warning; style still runs, with source_language forced to target_language
    (decision D7)."""
    args = _make_args(
        auto_glossary=True, auto_style=True, refine_only=True,
        source_lang="English", target_lang="French",
    )
    logger = FakeLogger()
    prompt_options = {}
    captured = {}
    client = FakeClient()

    async def fake_build(**kwargs):
        captured.update(kwargs)
        return {"refinement_instructions": "polish it"}

    monkeypatch.setattr(translate, "create_provider_from_args", lambda *a, **kw: client)
    monkeypatch.setattr(translate, "build_auto_prompt_options", fake_build)

    translate._apply_cli_auto_prep(args, prompt_options, logger)

    assert len(logger.warnings) == 1
    assert "--auto-glossary ignored in --refine-only mode." in logger.warnings[0]
    assert captured["want_glossary"] is False
    assert captured["want_style"] is True
    assert captured["source_language"] == "French"
    assert captured["target_language"] == "French"
    assert prompt_options == {"refinement_instructions": "polish it"}


def test_auto_style_alone_merges_fragment(monkeypatch):
    """--auto-style with no other flag merges the fake fragment into
    prompt_options and leaves existing keys alone."""
    args = _make_args(auto_style=True)
    logger = FakeLogger()
    prompt_options = {"preserve_technical_content": True}
    client = FakeClient()

    async def fake_build(**kwargs):
        return {"custom_instructions": "keep it dry"}

    monkeypatch.setattr(translate, "create_provider_from_args", lambda *a, **kw: client)
    monkeypatch.setattr(translate, "build_auto_prompt_options", fake_build)

    translate._apply_cli_auto_prep(args, prompt_options, logger)

    assert prompt_options == {
        "preserve_technical_content": True,
        "custom_instructions": "keep it dry",
    }
    assert logger.warnings == []
    assert client.closed is True


def test_raising_build_leaves_prompt_options_untouched(monkeypatch):
    """A raising `build_auto_prompt_options` never escapes and never mutates
    prompt_options; the client is still closed and a warning is logged."""
    args = _make_args(auto_glossary=True)
    logger = FakeLogger()
    prompt_options = {"preserve_technical_content": True}
    client = FakeClient()

    async def fake_build(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(translate, "create_provider_from_args", lambda *a, **kw: client)
    monkeypatch.setattr(translate, "build_auto_prompt_options", fake_build)

    # Must not raise.
    translate._apply_cli_auto_prep(args, prompt_options, logger)

    assert prompt_options == {"preserve_technical_content": True}
    assert any("Auto mode failed" in w for w in logger.warnings)
    assert client.closed is True


def test_neither_flag_never_creates_client(monkeypatch):
    """With neither --auto-glossary nor --auto-style, no client is created and
    no event loop is spun up."""
    args = _make_args()
    logger = FakeLogger()
    prompt_options = {}
    calls = []

    monkeypatch.setattr(translate, "create_provider_from_args", lambda *a, **kw: calls.append("client"))
    monkeypatch.setattr(translate, "build_auto_prompt_options", lambda **kw: calls.append("build"))

    translate._apply_cli_auto_prep(args, prompt_options, logger)

    assert calls == []
    assert prompt_options == {}
    assert logger.warnings == []


def test_close_awaited_even_when_extraction_raises(monkeypatch):
    """`client.close()` runs inside the same coroutine's `finally`, so it is
    awaited even though `build_auto_prompt_options` raised."""
    args = _make_args(auto_style=True)
    logger = FakeLogger()
    prompt_options = {}
    client = FakeClient()

    async def fake_build(**kwargs):
        raise ValueError("kaboom")

    monkeypatch.setattr(translate, "create_provider_from_args", lambda *a, **kw: client)
    monkeypatch.setattr(translate, "build_auto_prompt_options", fake_build)

    translate._apply_cli_auto_prep(args, prompt_options, logger)

    assert client.closed is True
    assert prompt_options == {}
