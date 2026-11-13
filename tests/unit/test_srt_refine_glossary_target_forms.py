"""SRT refine must match the glossary on target forms (C4).

The refine pass sees target-language text, so filtering the glossary on
source-language keys (e.g. ZH->FR) matched nothing and the refine prompt got
no glossary at all.
"""

import asyncio

from src.core import subtitle_translator
from tests.unit.test_srt_refine_indexing import FakeLLMClient


def test_srt_refine_requests_target_form_glossary_matching(monkeypatch):
    calls = []

    def fake_block(chunk_content, prompt_options, **kwargs):
        calls.append(kwargs)
        return ""

    monkeypatch.setattr(subtitle_translator, "_build_chunk_glossary_block", fake_block)

    asyncio.run(subtitle_translator.refine_subtitle_translations(
        translations={0: "Bonjour Li Wei", 1: "Au revoir"},
        target_language="French",
        model_name="fake",
        llm_client=FakeLLMClient(),
        prompt_options={"glossary_terms": {"李伟": "Li Wei"}},
    ))

    assert calls, "refine pass never built a glossary block"
    assert all(call.get("match_target_forms") is True for call in calls)
