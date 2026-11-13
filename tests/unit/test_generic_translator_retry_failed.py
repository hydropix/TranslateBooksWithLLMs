"""Resume must retry failed chunks instead of skipping them (issue #204).

For TXT/SRT jobs, the progress pointer (current_chunk_index) advances past
failed units, so a resume based solely on it never re-enqueued them: the
retry pass was a no-op, the job was marked 'completed' and the checkpoint
(the only record of which chunks failed) was deleted. Pending work is now
derived from per-chunk statuses, and a job is only marked completed when
every unit is genuinely translated.
"""

import pytest

import src.core.llm.factory as llm_factory_module
import src.core.translator as translator_module
from src.core.adapters.format_adapter import FormatAdapter
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.translation_unit import TranslationUnit
from src.persistence.checkpoint_manager import CheckpointManager

N_UNITS = 4
FAILING_CONTENT = "content 1"


class FakeAdapter(FormatAdapter):
    """In-memory TXT-like adapter with deterministic units."""

    def __init__(self, tmp_path):
        super().__init__(
            input_file_path=str(tmp_path / "in.txt"),
            output_file_path=str(tmp_path / "out.txt"),
            config={},
        )
        self.units = [
            TranslationUnit(
                unit_id=f"chunk_{i}",
                content=f"content {i}",
                metadata={"chunk_index": i},
            )
            for i in range(N_UNITS)
        ]
        self.saved = {}

    async def prepare_for_translation(self):
        return True

    def get_translation_units(self):
        return self.units

    async def save_unit_translation(self, unit_id, translated_content):
        self.saved[unit_id] = translated_content
        return True

    async def reconstruct_output(self, bilingual=False):
        return "\n".join(
            self.saved.get(u.unit_id, u.content) for u in self.units
        ).encode("utf-8")

    async def resume_from_checkpoint(self, checkpoint_data):
        for chunk in checkpoint_data.get("chunks", []):
            if chunk.get("status") == "completed" and chunk.get("translated_text"):
                self.saved[f"chunk_{chunk['chunk_index']}"] = chunk["translated_text"]
        return checkpoint_data.get("resume_from_index", 0)

    async def cleanup(self):
        pass

    @property
    def format_name(self):
        return "txt"


class FakeLLMClient:
    def __init__(self, **kwargs):
        pass


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


def _patch_llm(monkeypatch, fail_contents, calls):
    """LLM stub: returns None (= unit failure) for contents in fail_contents."""

    async def fake_request(main_content=None, **kwargs):
        calls.append(main_content)
        if main_content in fail_contents:
            return None
        return f"translated {main_content}"

    monkeypatch.setattr(llm_factory_module, "create_provider_from_args", _fake_factory(FakeLLMClient))
    monkeypatch.setattr(
        translator_module, "generate_translation_request", fake_request
    )


async def _run(adapter, cm):
    translator = GenericTranslator(
        adapter=adapter, checkpoint_manager=cm, translation_id="job"
    )
    return await translator.translate(
        source_language="English",
        target_language="French",
        model_name="fake-model",
        llm_provider="ollama",
    )


@pytest.mark.asyncio
async def test_run_with_failure_marks_partial_and_keeps_checkpoint(
    tmp_path, cm, monkeypatch
):
    calls = []
    _patch_llm(monkeypatch, {FAILING_CONTENT}, calls)

    success = await _run(FakeAdapter(tmp_path), cm)

    assert success is False
    job = cm.get_job("job")
    assert job is not None
    assert job["status"] == "partial"
    checkpoint = cm.load_checkpoint("job")
    assert checkpoint["failed_chunk_indices"] == [1]


@pytest.mark.asyncio
async def test_resume_retries_failed_chunk_then_completes(
    tmp_path, cm, monkeypatch
):
    calls = []
    _patch_llm(monkeypatch, {FAILING_CONTENT}, calls)
    await _run(FakeAdapter(tmp_path), cm)

    # Second run: the LLM now succeeds. Only the failed unit must be retried.
    calls = []
    _patch_llm(monkeypatch, set(), calls)
    adapter = FakeAdapter(tmp_path)
    success = await _run(adapter, cm)

    assert calls == [FAILING_CONTENT]
    assert success is True
    assert cm.get_job("job")["status"] == "completed"
    # The retried unit ends up translated in the reconstructed output.
    output = (await adapter.reconstruct_output()).decode("utf-8")
    assert f"translated {FAILING_CONTENT}" in output
    assert len(adapter.saved) == N_UNITS


@pytest.mark.asyncio
async def test_resume_with_persistent_failure_stays_partial(
    tmp_path, cm, monkeypatch
):
    calls = []
    _patch_llm(monkeypatch, {FAILING_CONTENT}, calls)
    await _run(FakeAdapter(tmp_path), cm)

    # Second run: the unit fails again. The job must NOT be marked completed
    # and the checkpoint (failure evidence) must survive.
    calls = []
    _patch_llm(monkeypatch, {FAILING_CONTENT}, calls)
    success = await _run(FakeAdapter(tmp_path), cm)

    assert calls == [FAILING_CONTENT]
    assert success is False
    job = cm.get_job("job")
    assert job is not None
    assert job["status"] == "partial"
    assert cm.load_checkpoint("job")["failed_chunk_indices"] == [1]


def _fake_factory(client_cls):
    """Stand-in for `create_provider_from_args` that builds the test's fake client."""
    def build(llm_provider, model=None, endpoint=None, **kwargs):
        return client_cls(provider_type=llm_provider, model=model, **kwargs)
    return build
