"""Per-job provider keys must reach the provider factory on every format
(issue #230).

`translate_file` once forwarded per-job API keys for the EPUB path only: the
TXT/SRT adapter path and the DOCX/PDF call sites dropped `nim_api_key`, so
with `llm_provider='nim'`, a per-job NIM key and no NIM key in `.env`, the
factory raised "NVIDIA NIM provider requires an API key". The key now
travels inside the job's `ProviderConfig`, and every engine builds its client
with `create_llm_provider(job.provider_for(...))`, so there is no per-key
argument left to drop.

The tests drive the real entry point with the real factory (its `_require_key`
gate decides pass/fail); only the network boundary (`provider.generate`) is
stubbed, and the factory calls are recorded around the real ones.
"""

import asyncio

import pytest

import src.core.llm.factory as factory_module
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager
from src.utils.file_detector import detect_file_type
from tests.job_helpers import make_ctx, make_job

JOB_KEY = "nvfake-key-per-job"
MODEL = "moonshotai/kimi-k2-instruct-0905"

TXT_CONTENT = "Hello world, this is a short line.\n"

SRT_CONTENT = """1
00:00:01,000 --> 00:00:02,000
Hello world.

2
00:00:03,000 --> 00:00:04,000
This is a second line.
"""


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


def _patch_boundaries(monkeypatch):
    """Stub the network edge and record every real factory call."""
    from src.core.llm.base import LLMResponse
    from src.core.llm.providers.openai import OpenAICompatibleProvider

    async def fake_generate(self, prompt, *args, **kwargs):
        # [0]/[1] markers satisfy the SRT adapter's per-unit validation
        # (missing-marker responses are retried then marked failed).
        return LLMResponse(content="<translate>[0]ok [1]ok</translate>")

    monkeypatch.setattr(OpenAICompatibleProvider, "generate", fake_generate)
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    # The legacy factory path falls back to the `.env` value kept in
    # src.config; blank it so a developer's local NIM key cannot mask a
    # dropped key.
    import src.config as config_module
    monkeypatch.setattr(config_module, "NIM_API_KEY", "")

    real_create = factory_module.create_llm_provider
    factory_calls = []

    def recording_create(provider=None, **kwargs):
        factory_calls.append((provider, kwargs))
        return real_create(provider, **kwargs)

    monkeypatch.setattr(factory_module, "create_llm_provider", recording_create)
    return factory_calls


def _nim_job(src, out):
    return make_job(str(src), str(out), provider="nim", model=MODEL, api_key=JOB_KEY)


def _assert_job_key_reached_the_factory(factory_calls):
    assert factory_calls, "the provider factory was never called"
    for cfg, kwargs in factory_calls:
        # Engines pass a ProviderConfig and nothing else: no loose keys.
        assert isinstance(cfg, ProviderConfig)
        assert kwargs == {}
        assert cfg.provider == "nim"
        assert cfg.api_key == JOB_KEY


async def _translate_txt_or_srt(tmp_path, cm, ext):
    from src.core.adapters.translate_file import translate_file

    src = tmp_path / f"in.{ext}"
    content = TXT_CONTENT if ext == "txt" else SRT_CONTENT
    src.write_text(content, encoding="utf-8")

    return await translate_file(
        _nim_job(src, tmp_path / f"out.{ext}"),
        make_ctx(checkpoint_manager=cm, translation_id=f"job-{ext}"),
    )


@pytest.mark.asyncio
async def test_txt_forwards_the_per_job_nim_key(tmp_path, cm, monkeypatch):
    assert detect_file_type(str(tmp_path / "in.txt")) == "txt"
    factory_calls = _patch_boundaries(monkeypatch)

    result = await _translate_txt_or_srt(tmp_path, cm, "txt")

    assert result.ok
    _assert_job_key_reached_the_factory(factory_calls)
    assert cm.get_job("job-txt")["status"] == "completed"


@pytest.mark.asyncio
async def test_srt_forwards_the_per_job_nim_key(tmp_path, cm, monkeypatch):
    assert detect_file_type(str(tmp_path / "in.srt")) == "srt"
    factory_calls = _patch_boundaries(monkeypatch)

    result = await _translate_txt_or_srt(tmp_path, cm, "srt")

    assert result.ok
    _assert_job_key_reached_the_factory(factory_calls)
    assert cm.get_job("job-srt")["status"] == "completed"


@pytest.mark.asyncio
async def test_docx_forwards_the_per_job_nim_key(tmp_path, cm, monkeypatch):
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hello world, this is a short line.")
    src = tmp_path / "in.docx"
    doc.save(str(src))
    assert detect_file_type(str(src)) == "docx"

    factory_calls = _patch_boundaries(monkeypatch)

    from src.core.adapters.translate_file import translate_file

    # On the old code this raised ValueError at provider creation.
    result = await translate_file(
        _nim_job(src, tmp_path / "out.docx"),
        make_ctx(checkpoint_manager=cm, translation_id="job-docx"),
    )

    assert result.ok
    _assert_job_key_reached_the_factory(factory_calls)


@pytest.mark.asyncio
async def test_pdf_forwards_the_per_job_nim_key(tmp_path, cm, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (72, 100), "Hello world, this is a short line.",
        fontsize=11, fontname="helv",
    )
    src = tmp_path / "in.pdf"
    doc.save(str(src))
    doc.close()
    assert detect_file_type(str(src)) == "pdf"

    factory_calls = _patch_boundaries(monkeypatch)

    from src.core.adapters.translate_file import translate_file

    result = await translate_file(
        _nim_job(src, tmp_path / "out.pdf"),
        make_ctx(checkpoint_manager=cm, translation_id="job-pdf"),
    )

    assert result.ok
    _assert_job_key_reached_the_factory(factory_calls)


def test_missing_job_key_fails_instead_of_reading_another_provider(tmp_path, cm, monkeypatch):
    """Without the job key the real factory gate refuses the NIM provider."""
    _patch_boundaries(monkeypatch)
    from src.core.adapters.translate_file import translate_file

    src = tmp_path / "in.txt"
    src.write_text(TXT_CONTENT, encoding="utf-8")
    result = asyncio.run(translate_file(
        make_job(str(src), str(tmp_path / "out.txt"), provider="nim", model=MODEL),
        make_ctx(checkpoint_manager=cm, translation_id="job-nokey"),
    ))

    assert not result.ok
