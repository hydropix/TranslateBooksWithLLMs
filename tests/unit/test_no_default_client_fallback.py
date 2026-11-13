"""No silent fallback to the module-global Ollama client.

The request helpers used `llm_client or default_client`, so a job whose
client was missing (e.g. create_llm_client returned None for a cloud
provider) silently talked to the import-time Ollama default. Client creation
failures must now fail the pass clearly.
"""

import pytest

import src.core.refine.srt_refiner as srt_refiner
import src.core.translator as translator


@pytest.mark.asyncio
async def test_translation_request_without_client_fails_clearly():
    with pytest.raises(ValueError, match="No LLM client"):
        await translator.generate_translation_request(
            "Hello world, a sentence.", "", "", "", llm_client=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("factory_result", ["raise", None])
async def test_refine_chunks_fails_when_the_client_cannot_be_created(monkeypatch, factory_result):
    def factory(*args, **kwargs):
        if factory_result == "raise":
            raise ValueError("NVIDIA NIM provider requires an API key")
        return None

    monkeypatch.setattr(translator, "create_llm_client", factory)
    logs = []
    with pytest.raises(RuntimeError, match="Could not create the LLM client"):
        await translator.refine_chunks(
            ["Un brouillon."], [{}], target_language="French", model_name="m",
            api_endpoint="", llm_provider="nim",
            log_callback=lambda kind, msg, data=None: logs.append(kind),
        )
    assert "refinement_client_failed" in logs


@pytest.mark.asyncio
async def test_srt_refine_fails_when_the_client_cannot_be_created(tmp_path, monkeypatch):
    src = tmp_path / "in.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\nBonjour.\n", encoding="utf-8")

    def factory(*args, **kwargs):
        raise ValueError("missing key")

    monkeypatch.setattr(srt_refiner, "create_llm_client", factory)
    logs = []
    ok = await srt_refiner.refine_srt_file(
        input_filepath=str(src), output_filepath=str(tmp_path / "out.srt"),
        target_language="French", model_name="m", llm_provider="nim",
        log_callback=lambda kind, msg, data=None: logs.append(kind),
    )
    assert ok is False
    assert "srt_refine_client_failed" in logs
