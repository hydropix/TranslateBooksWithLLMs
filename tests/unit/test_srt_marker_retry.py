"""SRT translate path must retry when the LLM drops [N] index markers.

Before this fix the live SRT path (SrtAdapter + GenericTranslator) accepted
any non-empty LLM response: missing markers were only logged to the server
console, the unit was counted completed, and the affected cues silently kept
the source-language text in the output file.

Now the orchestrator validates each unit via the adapter hook, retries with
a reinforced prompt up to UNIT_VALIDATION_RETRIES times, and after
exhaustion keeps the best-effort content while marking the unit failed
(job ends 'partial', retryable like issue #204).
"""

import pytest

import src.config as config_module
import src.core.llm.factory as llm_factory_module
import src.core.translator as translator_module
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.srt_adapter import SrtAdapter
from src.persistence.checkpoint_manager import CheckpointManager

SRT = """1
00:00:01,000 --> 00:00:02,000
alpha one

2
00:00:03,000 --> 00:00:04,000
bravo two

3
00:00:05,000 --> 00:00:06,000
charlie three
"""

COMPLETE = "[0]ALPHA UN\n[1]BRAVO DEUX\n[2]CHARLIE TROIS"
MISSING_LAST = "[0]ALPHA UN\n[1]BRAVO DEUX"


class FakeLLMClient:
    def __init__(self, **kwargs):
        pass


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


def _patch_llm(monkeypatch, responses, calls):
    """LLM stub returning responses[i] for call i (last one repeats)."""

    async def fake_request(main_content=None, **kwargs):
        calls.append({
            "content": main_content,
            "prompt_options": kwargs.get("prompt_options") or {},
        })
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(llm_factory_module, "create_provider_from_args", _fake_factory(FakeLLMClient))
    monkeypatch.setattr(
        translator_module, "generate_translation_request", fake_request
    )


async def _run(tmp_path, cm, logs=None):
    input_file = tmp_path / "in.srt"
    output_file = tmp_path / "out.srt"
    input_file.write_text(SRT, encoding="utf-8")
    adapter = SrtAdapter(str(input_file), str(output_file), {})
    translator = GenericTranslator(
        adapter=adapter, checkpoint_manager=cm, translation_id="job"
    )

    def log_callback(msg_type, message, data=None):
        if logs is not None:
            logs.append((msg_type, message))

    success = await translator.translate(
        source_language="English",
        target_language="French",
        model_name="fake-model",
        llm_provider="ollama",
        log_callback=log_callback,
    )
    return success, output_file


@pytest.mark.asyncio
async def test_validate_unit_translation_detects_missing_markers(tmp_path):
    input_file = tmp_path / "in.srt"
    input_file.write_text(SRT, encoding="utf-8")
    adapter = SrtAdapter(str(input_file), str(tmp_path / "out.srt"), {})
    assert await adapter.prepare_for_translation()

    assert adapter.validate_unit_translation("block_0", COMPLETE) is None
    feedback = adapter.validate_unit_translation("block_0", MISSING_LAST)
    assert feedback is not None and "[2]" in feedback
    feedback = adapter.validate_unit_translation("block_0", "no markers at all")
    assert feedback is not None
    assert "[0]" in feedback and "[1]" in feedback and "[2]" in feedback


@pytest.mark.asyncio
async def test_missing_marker_retries_then_succeeds(tmp_path, cm, monkeypatch):
    calls = []
    logs = []
    _patch_llm(monkeypatch, [MISSING_LAST, COMPLETE], calls)

    success, output_file = await _run(tmp_path, cm, logs)

    assert success is True
    assert len(calls) == 2
    # The retry prompt is reinforced with the exact missing markers.
    retry_instructions = calls[1]["prompt_options"].get("custom_instructions", "")
    assert "[2]" in retry_instructions
    # The validation failure is reported through the log callback.
    assert any(t == "unit_validation_failed" for t, _ in logs)

    output = output_file.read_text(encoding="utf-8")
    assert "CHARLIE TROIS" in output
    assert cm.get_job("job")["status"] == "completed"


@pytest.mark.asyncio
async def test_persistent_missing_marker_marks_unit_failed(
    tmp_path, cm, monkeypatch
):
    monkeypatch.setattr(config_module, "UNIT_VALIDATION_RETRIES", 2)
    calls = []
    logs = []
    _patch_llm(monkeypatch, [MISSING_LAST], calls)

    success, output_file = await _run(tmp_path, cm, logs)

    assert success is False
    assert len(calls) == 3  # 1 attempt + 2 retries
    assert any(t == "unit_validation_exhausted" for t, _ in logs)

    # The job is partial and the unit is recorded failed, hence retryable.
    assert cm.get_job("job")["status"] == "partial"
    assert cm.load_checkpoint("job")["failed_chunk_indices"] == [0]

    # Best effort: the cues that did come back are translated, the missing
    # one keeps its source text instead of inheriting a neighbor's.
    output = output_file.read_text(encoding="utf-8")
    assert "ALPHA UN" in output
    assert "BRAVO DEUX" in output
    assert "charlie three" in output


@pytest.mark.asyncio
async def test_complete_response_does_not_retry(tmp_path, cm, monkeypatch):
    calls = []
    _patch_llm(monkeypatch, [COMPLETE], calls)

    success, output_file = await _run(tmp_path, cm)

    assert success is True
    assert len(calls) == 1
    assert cm.get_job("job")["status"] == "completed"


def _fake_factory(client_cls):
    """Stand-in for `create_provider_from_args` that builds the test's fake client."""
    def build(llm_provider, model=None, endpoint=None, **kwargs):
        return client_cls(provider_type=llm_provider, model=model, **kwargs)
    return build
