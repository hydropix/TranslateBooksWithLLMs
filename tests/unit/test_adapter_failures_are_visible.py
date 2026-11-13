"""Adapter failures must fail the job visibly.

The SRT and PGN adapters' reconstruct_output returned the untranslated source
file on any exception, and the generic translator then wrote it and marked the
job completed. Bare `except: return False/0` in prepare/resume hid the cause
(and a half-restored checkpoint produced source text for units the job counted
as translated).
"""

import pytest

import src.core.llm.factory as llm_factory_module
import src.core.translator as translator_module
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.pgn_adapter import PgnAdapter
from src.core.adapters.srt_adapter import SrtAdapter
from src.core.adapters.txt_adapter import TxtAdapter
from src.persistence.checkpoint_manager import CheckpointManager

SRT = "1\n00:00:01,000 --> 00:00:02,000\nHello world.\n\n2\n00:00:03,000 --> 00:00:04,000\nGoodbye.\n"
PGN = '[Event "x"]\n\n1. e4 {A strong opening move.} e5 *\n'


class _FakeLLMClient:
    def __init__(self, **kwargs):
        pass


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls,ext,content", [
    (SrtAdapter, "srt", SRT),
    (PgnAdapter, "pgn", PGN),
])
async def test_reconstruct_failure_does_not_write_the_source(
        tmp_path, cm, monkeypatch, adapter_cls, ext, content):
    src = tmp_path / f"in.{ext}"
    src.write_text(content, encoding="utf-8")
    out = tmp_path / f"out.{ext}"

    async def fake_request(main_content=None, **kwargs):
        return main_content.upper()

    monkeypatch.setattr(llm_factory_module, "create_provider_from_args", _fake_factory(_FakeLLMClient))
    monkeypatch.setattr(translator_module, "generate_translation_request", fake_request)

    def broken(*args, **kwargs):
        raise ValueError("boom")

    if ext == "srt":
        from src.core.srt_processor import SRTProcessor
        monkeypatch.setattr(SRTProcessor, "reconstruct_srt", broken)
    else:
        import src.core.adapters.pgn_adapter as pgn_module
        monkeypatch.setattr(pgn_module, "_column", broken)

    adapter = adapter_cls(str(src), str(out), {})
    logs = []
    translator = GenericTranslator(adapter=adapter, checkpoint_manager=cm, translation_id="job")
    ok = await translator.translate(
        source_language="English", target_language="French",
        model_name="m", llm_provider="ollama",
        log_callback=lambda kind, msg, data=None: logs.append(kind),
    )

    assert ok is False
    assert not out.exists()
    assert "reconstruct_failed" in logs
    assert cm.get_job("job")["status"] != "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls,ext", [
    (SrtAdapter, "srt"), (PgnAdapter, "pgn"), (TxtAdapter, "txt"),
])
async def test_prepare_failure_carries_its_cause(tmp_path, adapter_cls, ext):
    adapter = adapter_cls(str(tmp_path / f"missing.{ext}"), str(tmp_path / f"o.{ext}"), {})
    with pytest.raises(Exception) as info:
        await adapter.prepare_for_translation()
    assert "missing" in str(info.value) or "No such file" in str(info.value)


@pytest.mark.asyncio
async def test_resume_failure_is_not_swallowed(tmp_path):
    src = tmp_path / "in.srt"
    src.write_text(SRT, encoding="utf-8")
    adapter = SrtAdapter(str(src), str(tmp_path / "o.srt"), {})
    await adapter.prepare_for_translation()
    bad = {"chunks": [{"status": "completed", "translated_text": "[0]x",
                       "chunk_data": {"local_to_global": {"not-an-int": 0}}}]}
    with pytest.raises(Exception):
        await adapter.resume_from_checkpoint(bad)


def _fake_factory(client_cls):
    """Stand-in for `create_provider_from_args` that builds the test's fake client."""
    def build(llm_provider, model=None, endpoint=None, **kwargs):
        return client_cls(provider_type=llm_provider, model=model, **kwargs)
    return build
