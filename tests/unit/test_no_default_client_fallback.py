"""No silent fallback to the module-global Ollama client.

The request helpers used `llm_client or default_client`, so a job whose
client was missing silently talked to the import-time Ollama default. Both
the global client and the None convention are gone: the factory raises, and
client creation failures fail the pass clearly.
"""

import pytest

import src.core.llm.factory as llm_factory
import src.core.refine.srt_refiner as srt_refiner
import src.core.translator as translator
from tests.job_helpers import make_ctx, make_job


@pytest.mark.asyncio
async def test_translation_request_without_client_fails_clearly():
    with pytest.raises(ValueError, match="No LLM client"):
        await translator.generate_translation_request(
            "Hello world, a sentence.", "", "", "", llm_client=None,
        )


@pytest.mark.asyncio
async def test_refine_chunks_fails_when_the_client_cannot_be_created(monkeypatch):
    def factory(*args, **kwargs):
        raise ValueError("NVIDIA NIM provider requires an API key")

    monkeypatch.setattr(llm_factory, "create_llm_provider", factory)
    logs = []
    with pytest.raises(RuntimeError, match="Could not create the LLM client"):
        await translator.refine_chunks(
            ["Un brouillon."], [{}],
            make_job(provider="nim", model="m", target_language="French"),
            make_ctx(log_callback=lambda kind, msg, data=None: logs.append(kind)),
        )
    assert "refinement_client_failed" in logs


@pytest.mark.asyncio
async def test_srt_refine_fails_when_the_client_cannot_be_created(tmp_path, monkeypatch):
    src = tmp_path / "in.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\nBonjour.\n", encoding="utf-8")

    def factory(*args, **kwargs):
        raise ValueError("missing key")

    monkeypatch.setattr(llm_factory, "create_llm_provider", factory)
    logs = []
    ok = await srt_refiner.refine_srt_file(
        make_job(str(src), str(tmp_path / "out.srt"), operation="refine",
                 provider="nim", model="m", target_language="French"),
        make_ctx(log_callback=lambda kind, msg, data=None: logs.append(kind)),
    )
    assert ok is False
    assert "srt_refine_client_failed" in logs
