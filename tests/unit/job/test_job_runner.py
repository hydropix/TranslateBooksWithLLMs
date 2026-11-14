"""JobRunner phase transitions, driven with fake engines and a real checkpoint DB."""

import asyncio

import pytest

from src.core.job import JobConfig, JobResult, JobStatus, OP_REFINE
from src.core.job.runner import JobRunner, JobSinks
from src.core.llm.exceptions import RateLimitError
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager


class RecordingSinks(JobSinks):
    def __init__(self):
        self.logs = []
        self.stats = []
        self.statuses = []
        self.finished = None
        self.rate_limited = None
        self.failed = None
        self.interrupt = False
        self.notified = []

    def log(self, key, message="", data=None):
        self.logs.append((key, message))

    def emit_stats(self, stats):
        self.stats.append(dict(stats))

    def is_interrupted(self):
        return self.interrupt

    def set_status(self, status, message=None):
        self.statuses.append(status)

    def job_finished(self, job, result, elapsed):
        self.finished = result

    def job_rate_limited(self, job, message):
        self.rate_limited = message

    def job_failed(self, job, error):
        self.failed = error

    def notify(self, event, context):
        self.notified.append(event)

    def keys(self):
        return [k for k, _ in self.logs]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "book.txt"
    source.write_text("one\n\ntwo\n\nthree\n", encoding="utf-8")
    manager = CheckpointManager(db_path=str(tmp_path / "data" / "jobs.db"))
    return tmp_path, source, manager


def _job(tmp_path, source, **changes):
    base = dict(
        operation='translate', file_type='txt', input_path=str(source),
        output_path=str(tmp_path / "out.txt"), source_language="English",
        target_language="French", provider=ProviderConfig(provider="poe", model="fake", api_key="unused"),
        origin='cli', auto_pause_on_rate_limit=True,
    )
    base.update(changes)
    return JobConfig(**base)


def _completed(job, ctx, text="translated"):
    with open(job.output_path, "w", encoding="utf-8") as f:
        f.write(text)
    stats = {'total_chunks': 3, 'completed_chunks': 3, 'failed_chunks': 0, 'unfinished_chunks': 0}
    ctx.emit(stats)
    return JobResult.from_run(stats, job.output_path, engine_ok=True)


def _status(manager, tid):
    return manager.get_job(tid)['status']


def test_translate_then_refine_runs_refine_exactly_once(env):
    tmp_path, source, manager = env
    calls = []

    async def translate_file(job, ctx):
        calls.append(('translate', dict(job.prompt_options), ctx.checkpoint_phase))
        return _completed(job, ctx)

    async def refine_file(job, ctx):
        calls.append(('refine', job.input_path, ctx.checkpoint_phase, ctx.resume_from_index))
        assert manager.get_job('cli_1')['job_phase'] == 'refine'
        return _completed(job, ctx, "refined")

    sinks = RecordingSinks()
    # Both refine requests set at once, like the web UI does.
    job = _job(tmp_path, source, refine_after=True, prompt_options={'refine': True})
    result = asyncio.run(JobRunner(job, translation_id='cli_1', checkpoint_manager=manager,
                                   sinks=sinks, engines={'translate_file': translate_file,
                                                         'refine_file': refine_file}).run())

    assert result.status is JobStatus.COMPLETED
    assert [c[0] for c in calls] == ['translate', 'refine']
    assert calls[0][1]['refine'] is False          # the engine never refines inline
    # Refine reads a snapshot of the translated output, removed once it completes.
    assert calls[1][1] == str(tmp_path / "out.txt") + ".pre-refine"
    assert not (tmp_path / "out.txt.pre-refine").exists()
    assert calls[1][2] == 'refine' and calls[1][3] == 0
    assert _status(manager, 'cli_1') == 'completed'
    # Phase-tagged progress: [0, 50] then [50, 100].
    assert any(s.get('current_phase') == 1 and s.get('enable_refinement') for s in sinks.stats)
    assert any(s.get('current_phase') == 2 for s in sinks.stats)
    assert sinks.finished is result


def test_fresh_refine_pass_ignores_a_leftover_pre_refine_copy(env):
    """A copy left by an earlier job on the same output must not be refined."""
    tmp_path, source, manager = env
    (tmp_path / "out.txt.pre-refine").write_text("stale text", encoding="utf-8")
    seen = []

    async def translate_file(job, ctx):
        return _completed(job, ctx, "fresh translation")

    async def refine_file(job, ctx):
        with open(job.input_path, encoding="utf-8") as f:
            seen.append(f.read())
        return _completed(job, ctx, "refined")

    job = _job(tmp_path, source, refine_after=True)
    result = asyncio.run(JobRunner(job, translation_id='cli_stale', checkpoint_manager=manager,
                                   sinks=RecordingSinks(), engines={'translate_file': translate_file,
                                                                    'refine_file': refine_file}).run())

    assert result.status is JobStatus.COMPLETED
    assert seen == ["fresh translation"]


def test_interrupted_refine_resumes_refine_not_translation(env):
    tmp_path, source, manager = env
    calls = []
    sinks = RecordingSinks()

    async def translate_file(job, ctx):
        calls.append('translate')
        return _completed(job, ctx)

    async def refine_interrupted(job, ctx):
        calls.append(('refine', ctx.resume_from_index))
        for i in range(2):
            manager.save_checkpoint(ctx.translation_id, i, f"t{i}", f"r{i}", phase=ctx.checkpoint_phase)
        sinks.interrupt = True
        return JobResult.interrupted(output_path=job.output_path)

    job = _job(tmp_path, source, refine_after=True, origin='web',
               output_path=str(tmp_path / "out.txt"))
    first = asyncio.run(JobRunner(job, translation_id='trans_1', checkpoint_manager=manager,
                                  sinks=sinks, temp_dir=str(tmp_path),
                                  engines={'translate_file': translate_file,
                                           'refine_file': refine_interrupted}).run())
    assert first.status is JobStatus.INTERRUPTED
    assert _status(manager, 'trans_1') == 'interrupted'
    # The translation chunk table is untouched by the refine units.
    assert manager.load_checkpoint('trans_1')['chunks'] == []

    async def refine_rest(job, ctx):
        calls.append(('refine', ctx.resume_from_index))
        return _completed(job, ctx, "refined")

    stored = manager.get_job('trans_1')['config']
    resumed_job = JobConfig.from_persisted(stored, output_dir=str(tmp_path), input_path=str(source))
    second = asyncio.run(JobRunner(resumed_job, translation_id='trans_1', checkpoint_manager=manager,
                                   sinks=RecordingSinks(), is_resume=True, temp_dir=str(tmp_path),
                                   engines={'translate_file': translate_file,
                                            'refine_file': refine_rest}).run())
    assert second.status is JobStatus.COMPLETED
    assert calls == ['translate', ('refine', 0), ('refine', 2)]


def test_rate_limit_auto_resume_is_a_loop(env, monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "RATE_LIMIT_AUTO_RESUME_DELAY", 0)
    tmp_path, source, manager = env
    attempts = []

    async def translate_file(job, ctx):
        attempts.append(ctx.resume_from_index)
        if len(attempts) == 1:
            manager.save_checkpoint(ctx.translation_id, 0, "one", "un", total_chunks=3, completed_chunks=1)
            raise RateLimitError("429", retry_after=0, provider="poe")
        return _completed(job, ctx)

    sinks = RecordingSinks()
    job = _job(tmp_path, source, auto_pause_on_rate_limit=False)
    result = asyncio.run(JobRunner(job, translation_id='cli_rl', checkpoint_manager=manager,
                                   sinks=sinks, engines={'translate_file': translate_file}).run())
    assert result.status is JobStatus.COMPLETED
    assert attempts == [0, 1]
    assert sinks.statuses == ['rate_limited', 'running']
    assert _status(manager, 'cli_rl') == 'completed'


def test_rate_limit_pause_keeps_the_job_resumable(env):
    tmp_path, source, manager = env

    async def translate_file(job, ctx):
        raise RateLimitError("429", retry_after=30, provider="poe")

    sinks = RecordingSinks()
    result = asyncio.run(JobRunner(_job(tmp_path, source), translation_id='cli_p',
                                   checkpoint_manager=manager, sinks=sinks,
                                   engines={'translate_file': translate_file}).run())
    assert result.status is JobStatus.RATE_LIMITED
    assert sinks.rate_limited and "Rate limited by poe" in sinks.rate_limited
    assert _status(manager, 'cli_p') == 'interrupted'


def test_failure_sets_a_terminal_status(env):
    tmp_path, source, manager = env

    async def translate_file(job, ctx):
        raise ValueError("boom")

    sinks = RecordingSinks()
    result = asyncio.run(JobRunner(_job(tmp_path, source), translation_id='cli_f',
                                   checkpoint_manager=manager, sinks=sinks,
                                   engines={'translate_file': translate_file}).run())
    assert result.status is JobStatus.FAILED
    assert "boom" in result.error and "boom" in sinks.failed
    assert _status(manager, 'cli_f') == 'error'
    assert 'failure' in sinks.notified


def test_cancellation_still_leaves_a_terminal_status(env):
    tmp_path, source, manager = env

    async def translate_file(job, ctx):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(JobRunner(_job(tmp_path, source), translation_id='cli_c',
                              checkpoint_manager=manager, sinks=RecordingSinks(),
                              engines={'translate_file': translate_file}).run())
    assert _status(manager, 'cli_c') == 'error'


def test_partial_translation_skips_refine_and_keeps_checkpoint(env):
    tmp_path, source, manager = env
    refined = []

    async def translate_file(job, ctx):
        with open(job.output_path, "w", encoding="utf-8") as f:
            f.write("x")
        stats = {'total_chunks': 3, 'completed_chunks': 3, 'failed_chunks': 1, 'unfinished_chunks': 1}
        return JobResult.from_run(stats, job.output_path, engine_ok=False)

    async def refine_file(job, ctx):
        refined.append(True)

    result = asyncio.run(JobRunner(_job(tmp_path, source, refine_after=True), translation_id='cli_part',
                                   checkpoint_manager=manager, sinks=RecordingSinks(),
                                   engines={'translate_file': translate_file,
                                            'refine_file': refine_file}).run())
    assert result.status is JobStatus.PARTIAL and result.exit_code == 1
    assert refined == []
    assert _status(manager, 'cli_part') == 'partial'


def test_refine_only_job_uses_the_refine_namespace(env):
    tmp_path, source, manager = env
    seen = {}

    async def refine_file(job, ctx):
        seen['phase'] = ctx.checkpoint_phase
        seen['input'] = job.input_path
        return _completed(job, ctx, "polished")

    job = _job(tmp_path, source, operation=OP_REFINE)
    result = asyncio.run(JobRunner(job, translation_id='cli_ro', checkpoint_manager=manager,
                                   sinks=RecordingSinks(),
                                   engines={'refine_file': refine_file}).run())
    assert result.ok
    assert seen == {'phase': 'refine', 'input': str(source)}


def test_glossary_and_instructions_are_snapshotted_before_start_job(env):
    tmp_path, source, manager = env

    class Glossary:
        name = "Names"
        terms = [1]
        terms_dict = {"Alice": "Alicia"}
        terms_metadata = {}

    class Store:
        def get_glossary(self, glossary_id):
            assert glossary_id == 7
            return Glossary()

    captured = {}

    async def translate_file(job, ctx):
        captured.update(job.prompt_options)
        return _completed(job, ctx)

    job = _job(tmp_path, source, prompt_options={'glossary_id': 7})
    sinks = RecordingSinks()
    asyncio.run(JobRunner(job, translation_id='cli_g', checkpoint_manager=manager, sinks=sinks,
                          glossary_store=lambda: Store(),
                          engines={'translate_file': translate_file}).run())
    assert captured['glossary_terms'] == {"Alice": "Alicia"}
    stored = manager.get_job('cli_g')['config']['prompt_options']
    assert stored['glossary_terms'] == {"Alice": "Alicia"}
    assert 'glossary_loaded' in sinks.keys()
