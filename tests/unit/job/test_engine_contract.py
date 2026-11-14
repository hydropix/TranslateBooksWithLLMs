"""Engines take (JobConfig, JobContext) and entry points return a JobResult."""

import pytest

from src.core.adapters import refine_file, translate_file
from src.core.adapters.exceptions import UnsupportedFormatError
from src.core.job import JobStatus
from src.core.llm.exceptions import LLMClientConfigError
from src.core.refine.client_setup import build_refine_client
from tests.characterization import fake_llm
from tests.job_helpers import make_ctx, make_job


def test_build_refine_client_raises_on_a_bad_provider():
    events = []
    with pytest.raises(LLMClientConfigError):
        build_refine_client(make_job(provider="no-such-provider", model="m"),
                            log_callback=lambda key, msg: events.append(key))
    assert events == ["llm_client_error"]


def test_build_refine_client_uses_the_refine_provider(monkeypatch):
    from src.core.llm.provider_config import ProviderConfig

    seen = []
    monkeypatch.setattr("src.core.llm.factory.create_llm_provider",
                        lambda cfg: seen.append(cfg) or object())
    job = make_job(provider="ollama", model="translator",
                   refine_provider=ProviderConfig(provider="openai", model="polisher",
                                                  endpoint="http://localhost:1"))
    client, context_manager = build_refine_client(job)

    assert client is not None
    assert context_manager is None  # adaptive context is Ollama-only
    assert [(c.provider, c.model) for c in seen] == [("openai", "polisher")]


@pytest.mark.asyncio
async def test_refine_file_rejects_unsupported_formats(tmp_path):
    src = tmp_path / "book.pgn"
    src.write_text('[Event "x"]\n\n1. e4 e5 *\n', encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        await refine_file(make_job(str(src), str(tmp_path / "out.pgn"), operation="refine"),
                          make_ctx())


@pytest.mark.asyncio
async def test_refine_file_fails_cleanly_when_the_client_cannot_be_built(tmp_path):
    src = tmp_path / "book.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\nBonjour.\n", encoding="utf-8")
    events = []
    result = await refine_file(
        make_job(str(src), str(tmp_path / "out.srt"), operation="refine",
                 provider="no-such-provider"),
        make_ctx(log_callback=lambda key, msg, *a: events.append(key)),
    )
    assert result.status is JobStatus.FAILED
    assert "srt_refine_client_failed" in events


@pytest.mark.asyncio
async def test_translate_file_reports_an_interrupted_run(tmp_path, monkeypatch):
    fake_llm.install(monkeypatch)
    src = tmp_path / "book.txt"
    src.write_text("\n\n".join(f"Paragraph number {i} of the book." for i in range(6)),
                   encoding="utf-8")
    from src.persistence.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    try:
        result = await translate_file(
            make_job(str(src), str(tmp_path / "out.txt"), provider="poe", model="fake",
                     api_key="UNUSED_FAKE_KEY", max_tokens_per_chunk=20),
            make_ctx(checkpoint_manager=manager, translation_id="job",
                     check_interruption=lambda: True),
        )
    finally:
        manager.close()
    assert result.status is JobStatus.INTERRUPTED
    assert not result
