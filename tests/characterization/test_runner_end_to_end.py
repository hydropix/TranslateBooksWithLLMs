"""The job runner driving the real engines with the echo provider.

Checks what only the full chain can show: a translate + refine job refines
exactly once (no in-engine EPUB/DOCX refine on top of the runner's refine
phase), the phase-tagged progress reaches 100%, the verdict is 'completed'
and the checkpoint ends in a terminal status.
"""

import asyncio

import pytest

from src.core.job import JobConfig, JobStatus
from src.core.job.runner import JobRunner, JobSinks
from src.core.llm.provider_config import ProviderConfig
from src.core.progress import snapshot_from_job_stats
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm, fixtures


class Sinks(JobSinks):
    def __init__(self):
        self.stats = []
        self.logs = []

    def log(self, key, message="", data=None):
        self.logs.append(key)

    def emit_stats(self, stats):
        self.stats.append(dict(stats))

    def notify(self, event, context):
        pass


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    fake_llm.install(monkeypatch)


@pytest.mark.parametrize("fmt,builder", [("txt", fixtures.build_txt), ("epub", fixtures.build_epub),
                                         ("srt", fixtures.build_srt), ("docx", fixtures.build_docx)])
def test_translate_then_refine_once(fmt, builder, tmp_path):
    source = builder(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    job = JobConfig(
        file_type=fmt, input_path=str(source), output_path=str(tmp_path / f"out.{fmt}"),
        source_language="English", target_language="French",
        provider=ProviderConfig(provider="poe", model="fake-echo", api_key="UNUSED_FAKE_KEY"),
        context_window=4096, auto_adjust_context=False, max_tokens_per_chunk=60,
        refine_after=True, prompt_options={'refine': True}, origin='cli',
    )
    sinks = Sinks()
    result = asyncio.run(JobRunner(job, translation_id=f"cli_{fmt}", checkpoint_manager=manager,
                                   sinks=sinks).run())

    assert result.status is JobStatus.COMPLETED, (result.error, sinks.logs[-5:])
    assert manager.get_job(f"cli_{fmt}")['status'] == 'completed'
    # One refine pass: the runner's (phase 2), never the engine's inline one.
    assert sinks.logs.count('refine_after_start') == 1
    assert 'epub_refinement_start' not in sinks.logs
    phase2 = [s for s in sinks.stats if s.get('current_phase') == 2]
    assert phase2, "the refine phase emitted no progress"
    assert all(s.get('enable_refinement') for s in sinks.stats if 'current_phase' in s)
    last = {}
    for s in sinks.stats:
        last.update(s)
    assert snapshot_from_job_stats(last).percent == 100.0
