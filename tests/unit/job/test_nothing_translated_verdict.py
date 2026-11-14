"""A job in which no unit got a translation ends 'failed', and stays resumable.

Before, a provider failing every request produced a 'partial' job whose
output was 100% source text. The verdict now calls it a failure (CLI exit
code 1, failure notification, web status 'error'), while a job with at least
one translated unit and at least one failed unit stays 'partial'. Resume must
not get worse: the failed job keeps its checkpoint, appears in the resumable
list, and a resumed run with a working provider completes it.
"""

import asyncio

import pytest

from src.core.job import JobConfig, JobStatus
from src.core.job.result import JobResult
from src.core.job.runner import JobRunner, JobSinks
from src.core.job.verdict import classify_completion
from src.core.llm.exceptions import ProviderError
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm, fixtures


# -- verdict --------------------------------------------------------------


@pytest.fixture
def output(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text("untouched source text", encoding="utf-8")
    return str(path)


def test_every_chunk_failed_is_an_error(output):
    stats = {'total_chunks': 6, 'completed_chunks': 6, 'failed_chunks': 6, 'unfinished_chunks': 6}
    verdict = classify_completion(stats, output)
    assert verdict.status == 'error'
    assert verdict.nothing_translated
    assert verdict.error


def test_every_chunk_left_as_source_is_an_error_even_with_text_free_chunks(output):
    # EPUB: completed_chunks counts the fallbacks; one chunk was text-free
    # (passed through without an LLM call), so it is neither owed nor translated.
    stats = {'total_chunks': 65, 'completed_chunks': 65, 'failed_chunks': 0,
             'fallback_used': 64, 'unfinished_chunks': 64, 'skipped_chunks': 1}
    verdict = classify_completion(stats, output)
    assert verdict.status == 'error' and verdict.nothing_translated
    result = JobResult.from_run(stats, output)
    assert result.status is JobStatus.FAILED
    assert result.units_translated == 0
    assert result.exit_code == 1


def test_docx_fallback_counter_without_unfinished_key(output):
    stats = {'total_chunks': 22, 'completed_chunks': 22, 'fallback_used': 22}
    assert classify_completion(stats, output).status == 'error'


@pytest.mark.parametrize("stats", [
    {'total_chunks': 6, 'completed_chunks': 6, 'failed_chunks': 5, 'unfinished_chunks': 5},
    {'total_chunks': 65, 'completed_chunks': 65, 'failed_chunks': 0,
     'unfinished_chunks': 63, 'skipped_chunks': 1},
])
def test_one_translated_unit_keeps_the_job_partial(output, stats):
    verdict = classify_completion(stats, output)
    assert verdict.status == 'partial'
    assert not verdict.nothing_translated
    assert JobResult.from_run(stats, output).units_translated == 1


def test_a_book_with_nothing_to_translate_is_still_completed(output):
    stats = {'total_chunks': 3, 'completed_chunks': 3, 'failed_chunks': 0,
             'unfinished_chunks': 0, 'skipped_chunks': 3}
    assert classify_completion(stats, output).status == 'completed'


# -- the full chain: real engines, a provider failing every request -------


class _FailingProvider(fake_llm.FakeEchoProvider):
    async def _send(self, request):
        raise ProviderError("upstream is down", status_code=500, retryable=False)


class _Sinks(JobSinks):
    def __init__(self):
        self.merged = {}
        self.logs = []
        self.notices = []

    def log(self, key, message="", data=None):
        self.logs.append(key)

    def emit_stats(self, stats):
        self.merged.update(stats)

    def current_stats(self):
        return dict(self.merged)

    def notify(self, event, context):
        self.notices.append(event)


def _install(monkeypatch, provider_cls):
    def factory(provider_type="ollama", **_kwargs):
        return provider_cls()
    for target in fake_llm._PATCH_TARGETS:
        monkeypatch.setattr(target, factory, raising=True)


def _job(fmt, source, tmp_path):
    return JobConfig(
        file_type=fmt, input_path=str(source), output_path=str(tmp_path / f"out.{fmt}"),
        source_language="English", target_language="French",
        provider=ProviderConfig(provider="poe", model="fake-echo", api_key="UNUSED_FAKE_KEY"),
        context_window=4096, auto_adjust_context=False, max_tokens_per_chunk=60,
        max_attempts=1, retry_delay=0, origin='cli',
    )


@pytest.mark.parametrize("fmt,builder", [("txt", fixtures.build_txt), ("srt", fixtures.build_srt),
                                         ("epub", fixtures.build_epub), ("docx", fixtures.build_docx)])
def test_provider_failing_every_request_fails_the_job_and_resume_completes_it(
        fmt, builder, tmp_path, monkeypatch):
    from src.utils.notifier import EVENT_FAILURE

    monkeypatch.chdir(tmp_path)
    source = builder(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    tid = f"cli_all_failed_{fmt}"

    _install(monkeypatch, _FailingProvider)
    sinks = _Sinks()
    result = asyncio.run(JobRunner(_job(fmt, source, tmp_path), translation_id=tid,
                                   checkpoint_manager=manager, sinks=sinks).run())

    assert result.status is JobStatus.FAILED, (result.status, sinks.merged)
    assert result.verdict is not None and result.verdict.nothing_translated
    assert result.exit_code == 1
    assert result.units_translated == 0
    assert 'summary_nothing_translated' in sinks.logs
    assert sinks.notices == [EVENT_FAILURE]
    # Resumable exactly like a partial job: checkpoint kept, listed.
    assert manager.get_job(tid)['status'] == 'error'
    assert tid in [job['translation_id'] for job in manager.get_resumable_jobs()]

    # Resume with a working provider: the job completes.
    fake_llm.install(monkeypatch)
    data = manager.load_checkpoint(tid)
    assert data is not None
    resumed = asyncio.run(JobRunner(_job(fmt, source, tmp_path), translation_id=tid,
                                    checkpoint_manager=manager, sinks=_Sinks(),
                                    is_resume=True).run())
    if fmt == 'docx':
        # Pre-existing DOCX limitation, shared with partial jobs: a resume
        # restores the finished document and does not retry the chunks that
        # fell back to their source text. The job stays failed and resumable.
        assert resumed.status is JobStatus.FAILED
        assert manager.get_job(tid)['status'] == 'error'
        return
    assert resumed.status is JobStatus.COMPLETED, (resumed.status, resumed.error)
    assert manager.get_job(tid)['status'] == 'completed'
