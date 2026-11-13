"""A failed or partial translation must be reported as a failure.

`translate_file` used to return True unconditionally for EPUB ("Legacy
function doesn't return success status"), and the CLI ignored the adapters'
return value altogether: it logged "Translation Completed Successfully",
sent the success notification and exited 0 even when chunks had failed or the
output was never written. A raised exception also exited 0.

The CLI now runs every job through the shared job runner
(``translate.run_cli_job``); its exit code is ``JobResult.exit_code``.
"""

import pytest

import translate
from src.core.adapters.translate_file import translate_file as adapter_translate_file
from src.core.job import OP_REFINE, JobConfig, JobResult, JobStatus
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager
from src.utils.notifier import EVENT_FAILURE, EVENT_SUCCESS


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg)

    def warning(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg)

    def update_progress(self, *args, **kwargs):
        pass

    def create_legacy_callback(self):
        return lambda key, message="", data=None: None


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A CLI job on tmp_path; returns (job builder, notified events, manager)."""
    monkeypatch.chdir(tmp_path)
    notified = []
    monkeypatch.setattr("src.utils.notifier.notify", lambda event, payload: notified.append(event))
    source = tmp_path / "book.txt"
    source.write_text("Hello.", encoding="utf-8")
    manager = CheckpointManager(db_path=str(tmp_path / "data" / "jobs.db"))

    def make_job(**changes):
        base = dict(operation='translate', file_type='txt', input_path=str(source),
                    output_path=str(tmp_path / "book (French).txt"),
                    source_language="English", target_language="French",
                    provider=ProviderConfig(provider="ollama", model="m",
                                            endpoint="http://localhost:11434"),
                    origin='cli')
        base.update(changes)
        return JobConfig(**base)

    return make_job, notified, manager


def _run(job, manager, logger, **engines):
    return translate.run_cli_job(job, logger, translation_id="cli_test",
                                 checkpoint_manager=manager, engines=engines)


async def _no_output(job, ctx):
    """An engine that reports failure and never wrote the output."""
    stats = {'total_chunks': 2, 'completed_chunks': 2, 'failed_chunks': 2}
    ctx.emit(stats)
    return JobResult.from_run(stats, job.output_path, engine_ok=False)


async def _partial(job, ctx):
    """An engine that wrote the output but some units failed."""
    with open(job.output_path, "w", encoding="utf-8") as handle:
        handle.write("Bonjour.")
    stats = {'total_chunks': 2, 'completed_chunks': 2, 'failed_chunks': 1}
    ctx.emit(stats)
    return JobResult.from_run(stats, job.output_path, engine_ok=False)


@pytest.mark.parametrize("name,operation", [
    ("translate_file", 'translate'),
    ("refine_file", OP_REFINE),
])
def test_cli_reports_a_failed_result_as_failure(cli, name, operation):
    make_job, notified, manager = cli
    logger = _FakeLogger()

    result = _run(make_job(operation=operation), manager, logger, **{name: _no_output})

    assert result.status is JobStatus.FAILED
    assert result.exit_code != 0
    assert EVENT_FAILURE in notified and EVENT_SUCCESS not in notified
    assert not any("Completed Successfully" in m for m in logger.infos)
    assert logger.errors


@pytest.mark.parametrize("name,operation", [
    ("translate_file", 'translate'),
    ("refine_file", OP_REFINE),
])
def test_cli_reports_a_partial_result_as_failure(cli, name, operation):
    make_job, notified, manager = cli
    logger = _FakeLogger()

    result = _run(make_job(operation=operation), manager, logger, **{name: _partial})

    assert result.status is JobStatus.PARTIAL
    assert result.exit_code != 0
    assert EVENT_SUCCESS not in notified
    assert not any("Completed Successfully" in m for m in logger.infos)
    assert logger.errors


def test_cli_exits_non_zero_on_exception(cli):
    make_job, notified, manager = cli

    async def boom(job, ctx):
        raise RuntimeError("boom")

    result = _run(make_job(), manager, _FakeLogger(), translate_file=boom)
    assert result.exit_code != 0
    assert "boom" in result.error
    assert notified == [EVENT_FAILURE]


def test_cli_success_still_exits_zero(cli):
    make_job, notified, manager = cli

    async def ok(job, ctx):
        with open(job.output_path, "w", encoding="utf-8") as handle:
            handle.write("Bonjour.")
        stats = {'total_chunks': 1, 'completed_chunks': 1, 'failed_chunks': 0}
        ctx.emit(stats)
        return JobResult.from_run(stats, job.output_path, engine_ok=True)

    logger = _FakeLogger()
    result = _run(make_job(), manager, logger, translate_file=ok)
    assert result.exit_code == 0
    assert notified == [EVENT_SUCCESS]
    assert any("Completed Successfully" in m for m in logger.infos)


@pytest.mark.asyncio
@pytest.mark.parametrize("epub_result,expected", [(True, True), (False, False), (None, False)])
async def test_translate_file_propagates_the_epub_result(tmp_path, monkeypatch, epub_result, expected):
    import src.core.epub.translator as epub_translator

    async def fake_translate_epub_file(**kwargs):
        return epub_result

    monkeypatch.setattr(epub_translator, "translate_epub_file", fake_translate_epub_file)
    src = tmp_path / "book.epub"
    src.write_bytes(b"PK\x03\x04")  # detection goes by extension

    result = await adapter_translate_file(
        input_filepath=str(src),
        output_filepath=str(tmp_path / "out.epub"),
        source_language="English",
        target_language="French",
        model_name="m",
        llm_provider="ollama",
        checkpoint_manager=None,
        translation_id=None,
    )

    assert result is expected
