"""A refine-after pass paused by a rate limit and resumed must give the same
output as an uninterrupted run, for every format the runner refines.

Real JobRunner, real translate/refine engines; only the LLM transport is fake
(echo). The provider raises RateLimitError once a request budget is spent,
which lands the pause inside the refine phase; the resume re-enters that phase.
"""

import asyncio

import pytest

from src.core.job import JobConfig, JobStatus
from src.core.job.runner import JobRunner, JobSinks
from src.core.llm.exceptions import RateLimitError
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm, fixtures
from tests.characterization.recorder import fingerprint_output


class _Budget:
    def __init__(self):
        self.limit = None
        self.calls = 0


class _Sinks(JobSinks):
    def __init__(self):
        self.logs = []

    def log(self, key, message="", data=None):
        self.logs.append(key)

    def notify(self, event, context):
        pass


@pytest.fixture
def budget(monkeypatch):
    state = _Budget()

    class LimitedEcho(fake_llm.FakeEchoProvider):
        async def _send(self, request):
            state.calls += 1
            if state.limit is not None and state.calls > state.limit:
                raise RateLimitError("429 Too Many Requests (test)", provider="fake")
            return await super()._send(request)

    def factory(provider_type="ollama", **kwargs):
        cw = getattr(provider_type, "context_window", None) or kwargs.get("context_window")
        return LimitedEcho(model="fake-echo", context_window=cw)

    for target in fake_llm._PATCH_TARGETS:
        monkeypatch.setattr(target, factory, raising=True)
    fake_llm.allow_echo(monkeypatch)
    monkeypatch.setattr("src.utils.notifier.notify", lambda *a, **k: None)
    return state


def _job(fmt, source, output):
    return JobConfig(
        file_type=fmt, input_path=str(source), output_path=str(output),
        source_language="English", target_language="French",
        provider=ProviderConfig(provider="poe", model="fake-echo", api_key="UNUSED_FAKE_KEY"),
        context_window=4096, auto_adjust_context=False, max_tokens_per_chunk=60,
        refine_after=True, origin='cli', auto_pause_on_rate_limit=True,
    )


def _run(job, tid, manager, is_resume=False):
    return asyncio.run(JobRunner(job, translation_id=tid, checkpoint_manager=manager,
                                 sinks=_Sinks(), is_resume=is_resume).run())


@pytest.mark.parametrize("fmt,builder", [("txt", fixtures.build_txt), ("srt", fixtures.build_srt),
                                         ("docx", fixtures.build_docx), ("epub", fixtures.build_epub)])
def test_refine_pause_and_resume_matches_a_clean_run(fmt, builder, budget, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = builder(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))

    clean_out = tmp_path / f"clean.{fmt}"
    assert _run(_job(fmt, source, clean_out), "clean", manager).status is JobStatus.COMPLETED
    total_calls = budget.calls

    # Translation-only run, to count the translate requests.
    budget.calls = 0
    probe = _job(fmt, source, tmp_path / f"probe.{fmt}").with_changes(refine_after=False)
    assert _run(probe, "probe", manager).status is JobStatus.COMPLETED
    translate_calls = budget.calls
    refine_calls = total_calls - translate_calls
    if refine_calls < 2:
        pytest.skip(f"{fmt}: the refine pass sends {refine_calls} request(s), cannot pause inside it")

    budget.calls = 0
    budget.limit = translate_calls + 1          # pause after one refined unit
    paused_out = tmp_path / f"paused.{fmt}"
    first = _run(_job(fmt, source, paused_out), "paused", manager)
    assert first.status is JobStatus.RATE_LIMITED, first.status

    budget.limit = None
    stored = manager.get_job("paused")["config"]
    resumed_job = JobConfig.from_persisted(stored, input_path=str(source))
    second = _run(resumed_job, "paused", manager, is_resume=True)
    assert second.status is JobStatus.COMPLETED, second.error

    assert fingerprint_output(paused_out) == fingerprint_output(clean_out)
