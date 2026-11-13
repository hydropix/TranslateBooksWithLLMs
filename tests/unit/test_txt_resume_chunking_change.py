"""TXT resume must survive a chunk-size change between pause and resume.

Completed chunks were restored by index after re-chunking the input with the
current chunk size, without checking that the stored source still matched the
chunk at that index: with a different size, paragraphs were duplicated or
dropped. The job's own chunking parameters now win on resume, and a stored
unit is only restored onto the same source text.
"""

import pytest

import src.core.llm.factory as llm_factory_module
import src.core.translator as translator_module
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.txt_adapter import TxtAdapter
from src.persistence.checkpoint_manager import CheckpointManager

PARAGRAPHS = [f"Paragraph number {k} tells a short story about item {k}." for k in range(12)]
TEXT = "\n\n".join(PARAGRAPHS) + "\n"


class _Client:
    def __init__(self, **kwargs):
        pass

    async def close(self):
        pass


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


async def _run(tmp_path, cm, monkeypatch, max_tokens, fail_marker=None):
    async def fake_request(main_content=None, **kwargs):
        if fail_marker and fail_marker in main_content:
            return None
        return main_content.upper()

    monkeypatch.setattr(llm_factory_module, "create_provider_from_args", _fake_factory(_Client))
    monkeypatch.setattr(translator_module, "generate_translation_request", fake_request)
    adapter = TxtAdapter(
        str(tmp_path / "in.txt"), str(tmp_path / "out.txt"),
        {"max_tokens_per_chunk": max_tokens},
    )
    translator = GenericTranslator(adapter=adapter, checkpoint_manager=cm, translation_id="job")
    ok = await translator.translate(
        source_language="English", target_language="French",
        model_name="m", llm_provider="openai",
    )
    return ok, adapter


def _paragraphs(path):
    import re
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip() and "item" in p.lower()]


@pytest.mark.asyncio
async def test_resume_with_another_chunk_size_keeps_every_paragraph_once(tmp_path, cm, monkeypatch):
    (tmp_path / "in.txt").write_text(TEXT, encoding="utf-8")

    ok, first = await _run(tmp_path, cm, monkeypatch, max_tokens=40, fail_marker="number 7 ")
    assert ok is False
    assert len(first.chunks) > 2

    ok, second = await _run(tmp_path, cm, monkeypatch, max_tokens=400)
    assert ok is True
    # The stored chunking won: same cut as the first run.
    assert [c["main_content"] for c in second.chunks] == [c["main_content"] for c in first.chunks]
    assert _paragraphs(tmp_path / "out.txt") == [p.upper() for p in PARAGRAPHS]


@pytest.mark.asyncio
async def test_txt_adapter_does_not_restore_onto_other_source(tmp_path):
    (tmp_path / "in.txt").write_text(TEXT, encoding="utf-8")
    adapter = TxtAdapter(str(tmp_path / "in.txt"), str(tmp_path / "o.txt"),
                         {"max_tokens_per_chunk": 400})
    await adapter.prepare_for_translation()
    stale = {"chunks": [{
        "status": "completed", "chunk_index": 0, "original_text": "something else",
        "translated_text": "STALE", "chunk_data": {"chunk_index": 0},
    }]}
    await adapter.resume_from_checkpoint(stale)
    assert adapter.translated_chunks[0] is None


def _fake_factory(client_cls):
    """Stand-in for `create_provider_from_args` that builds the test's fake client."""
    def build(llm_provider, model=None, endpoint=None, **kwargs):
        return client_cls(provider_type=llm_provider, model=model, **kwargs)
    return build
