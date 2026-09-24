"""Per-job provider keys must reach the provider factory on every format
(issue #230).

`translate_file` forwarded per-job API keys for the EPUB path only. The
TXT/SRT adapter path built `llm_config` and the DOCX path called
`create_llm_provider` without `nim_api_key`, so with `llm_provider='nim'`, a
per-job NIM key and no NIM key in `.env`, the factory raised "NVIDIA NIM
provider requires an API key": a TXT/SRT job failed mid-run (the key was
silently dropped) and a DOCX job failed at provider creation. The key is now
threaded at both call sites, matching the EPUB path.

The tests drive the real entry point with the real factory (its `_require_key`
gate decides pass/fail); only the network boundary (`provider.generate`) is
stubbed, and the factory kwargs are recorded around the real calls.
"""

import asyncio

import pytest

import src.core.llm as llm_package
import src.core.llm_client as llm_client_module
from src.persistence.checkpoint_manager import CheckpointManager
from src.utils.file_detector import detect_file_type

JOB_KEY = "nvfake-key-per-job"

TXT_CONTENT = "Hello world, this is a short line.\n"

SRT_CONTENT = """1
00:00:01,000 --> 00:00:02,000
Hello world.

2
00:00:03,000 --> 00:00:04,000
This is a second line.
"""

ALL_JOB_KEYS = (
    "gemini_api_key",
    "openai_api_key",
    "openrouter_api_key",
    "mistral_api_key",
    "deepseek_api_key",
    "poe_api_key",
    "nim_api_key",
)


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


def _patch_boundaries(monkeypatch):
    """Stub the network edge and record the factory kwargs of the real calls.

    The DOCX path resolves `create_llm_provider` from the `src.core.llm`
    package at call time; the TXT/SRT path reaches it through `LLMClient` in
    `src.core.llm_client`. Both attributes point at the recorded wrapper, and
    the real factory still runs (its `_require_key` gate is the oracle).
    """
    from src.core.llm.base import LLMResponse
    from src.core.llm.providers.openai import OpenAICompatibleProvider

    async def fake_generate(self, prompt, *args, **kwargs):
        # [0]/[1] markers satisfy the SRT adapter's per-unit validation
        # (missing-marker responses are retried then marked failed).
        return LLMResponse(content="<translate>[0]ok [1]ok</translate>")

    monkeypatch.setattr(OpenAICompatibleProvider, "generate", fake_generate)
    monkeypatch.delenv("NIM_API_KEY", raising=False)

    real_create = llm_client_module.create_llm_provider
    factory_calls = []

    def recording_create(provider_type, **kwargs):
        factory_calls.append((provider_type, kwargs))
        return real_create(provider_type, **kwargs)

    monkeypatch.setattr(
        llm_client_module, "create_llm_provider", recording_create
    )
    monkeypatch.setattr(llm_package, "create_llm_provider", recording_create)
    return factory_calls


async def _translate_txt_or_srt(tmp_path, cm, ext):
    from src.core.adapters.translate_file import translate_file

    src = tmp_path / f"in.{ext}"
    content = TXT_CONTENT if ext == "txt" else SRT_CONTENT
    src.write_text(content, encoding="utf-8")

    return await translate_file(
        input_filepath=str(src),
        output_filepath=str(tmp_path / f"out.{ext}"),
        source_language="English",
        target_language="French",
        model_name="moonshotai/kimi-k2-instruct-0905",
        llm_provider="nim",
        checkpoint_manager=cm,
        translation_id=f"job-{ext}",
        nim_api_key=JOB_KEY,
    )


@pytest.mark.asyncio
async def test_txt_forwards_the_per_job_nim_key(tmp_path, cm, monkeypatch):
    assert detect_file_type(str(tmp_path / "in.txt")) == "txt"
    factory_calls = _patch_boundaries(monkeypatch)

    # On the old code the factory raises "NVIDIA NIM provider requires an
    # API key", every unit fails and the job ends partial (returns False).
    assert await _translate_txt_or_srt(tmp_path, cm, "txt") is True

    provider_type, kwargs = factory_calls[0]
    assert provider_type == "nim"
    assert kwargs["nim_api_key"] == JOB_KEY
    assert cm.get_job("job-txt")["status"] == "completed"


@pytest.mark.asyncio
async def test_srt_forwards_the_per_job_nim_key(tmp_path, cm, monkeypatch):
    assert detect_file_type(str(tmp_path / "in.srt")) == "srt"
    factory_calls = _patch_boundaries(monkeypatch)

    assert await _translate_txt_or_srt(tmp_path, cm, "srt") is True

    provider_type, kwargs = factory_calls[0]
    assert provider_type == "nim"
    assert kwargs["nim_api_key"] == JOB_KEY
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

    # On the old code this raises ValueError at provider creation.
    result = await translate_file(
        input_filepath=str(src),
        output_filepath=str(tmp_path / "out.docx"),
        source_language="English",
        target_language="French",
        model_name="moonshotai/kimi-k2-instruct-0905",
        llm_provider="nim",
        checkpoint_manager=cm,
        translation_id="job-docx",
        nim_api_key=JOB_KEY,
    )

    assert result is True
    provider_type, kwargs = factory_calls[0]
    assert provider_type == "nim"
    # The DOCX call site forwards every per-job provider key, not just NIM.
    for key in ALL_JOB_KEYS:
        assert key in kwargs, f"{key} dropped from the DOCX provider call"
    assert kwargs["nim_api_key"] == JOB_KEY


def test_txt_llm_config_keeps_every_per_job_key(tmp_path, cm, monkeypatch):
    """The TXT/SRT `llm_config` must carry the full per-job key set, so a
    provider key added to `translate_file` later cannot be silently dropped
    here again (that is how NIM regressed)."""
    factory_calls = _patch_boundaries(monkeypatch)

    assert asyncio.run(_translate_txt_or_srt(tmp_path, cm, "txt")) is True

    _provider_type, kwargs = factory_calls[0]
    for key in ALL_JOB_KEYS:
        assert key in kwargs, f"{key} dropped from llm_config"
    assert kwargs["nim_api_key"] == JOB_KEY
