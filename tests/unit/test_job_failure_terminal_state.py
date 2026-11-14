"""Regression tests: a failed web job must reach a terminal DB status and
leave no temporary input file behind.

The generic `except Exception` path of `perform_actual_translation` used to set
the in-memory status to 'error' without calling `checkpoint_manager.mark_error`,
so the SQLite row stayed 'running' (not resumable until a restart). The
temporary text file created for inline-text jobs was only removed on success.
Both are now the job runner's (``src.core.job.runner``) ``try/finally``
guarantees; ``perform_actual_translation`` is its web adapter.
"""
import time

import pytest

from src.api import handlers
from src.core import adapters
from src.core.job import JobConfig, JobResult
from src.core.llm.provider_config import ProviderConfig


class FakeCheckpointManager:
    def __init__(self):
        self.calls = []

    def start_job(self, translation_id, file_type, config, input_file_path):
        self.calls.append(('start_job', translation_id))

    def update_job_config(self, translation_id, config):
        return True

    def __getattr__(self, name):
        if name.startswith('mark_') or name in ('prune_job_data', 'set_job_phase'):
            return lambda translation_id, *args: self.calls.append((name, translation_id))
        raise AttributeError(name)

    def load_checkpoint(self, translation_id):
        return None


class FakeStateManager:
    def __init__(self, checkpoint_manager):
        self._checkpoint_manager = checkpoint_manager
        self.fields = {'logs': [], 'stats': {'start_time': time.time()}}

    def exists(self, translation_id):
        return True

    def set_translation_field(self, translation_id, field, value):
        self.fields[field] = value

    def get_translation_field(self, translation_id, field):
        return self.fields.get(field)

    def update_stats(self, translation_id, new_stats):
        self.fields.setdefault('stats', {}).update(new_stats)

    def get_checkpoint_manager(self):
        return self._checkpoint_manager


class FakeSocketIO:
    def emit(self, *args, **kwargs):
        pass


def _job(output_dir):
    return JobConfig(
        text='Hello world.',
        file_type='txt',
        source_language='English',
        target_language='French',
        provider=ProviderConfig(provider='ollama', model='some-model',
                                endpoint='http://localhost:11434/api/generate'),
        output_path=str(output_dir / 'out.txt'),
        prompt_options={},
        origin='web',
    )


@pytest.fixture
def seams(monkeypatch):
    monkeypatch.setattr(handlers, 'emit_update', lambda *a, **k: None)
    monkeypatch.setattr('src.utils.notifier.notify', lambda *a, **k: None)


async def _run(tmp_path, monkeypatch, translate_impl):
    checkpoint_manager = FakeCheckpointManager()
    state_manager = FakeStateManager(checkpoint_manager)
    seen = {}

    async def fake_translate_file(job, ctx):
        seen['input'] = job.input_path
        return await translate_impl(job, ctx)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapters, 'translate_file', fake_translate_file)
    output_dir = tmp_path / 'out'
    output_dir.mkdir()
    await handlers.perform_actual_translation(
        'trans_1', _job(output_dir), state_manager, str(output_dir), FakeSocketIO())
    return state_manager, checkpoint_manager, seen, output_dir


@pytest.mark.asyncio
async def test_unexpected_exception_marks_db_error_and_removes_temp_file(
        tmp_path, monkeypatch, seams):
    async def boom(job, ctx):
        raise RuntimeError('provider exploded')

    state, checkpoint_manager, seen, output_dir = await _run(tmp_path, monkeypatch, boom)

    assert state.fields['status'] == 'error'
    assert 'provider exploded' in state.fields['error']
    assert ('mark_error', 'trans_1') in checkpoint_manager.calls
    # The temp input lived in the output dir and must be gone.
    assert seen['input'].startswith(str(output_dir))
    assert list(output_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_engine_failure_result_is_mirrored_to_db(tmp_path, monkeypatch, seams):
    async def fails_softly(job, ctx):
        return JobResult.failed('chunk loop failed')

    state, checkpoint_manager, _seen, output_dir = await _run(
        tmp_path, monkeypatch, fails_softly)

    assert state.fields['status'] == 'error'
    assert state.fields['error'] == 'chunk loop failed'
    assert ('mark_error', 'trans_1') in checkpoint_manager.calls
    assert list(output_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_success_still_removes_temp_file(tmp_path, monkeypatch, seams):
    async def writes(job, ctx):
        with open(job.output_path, 'w', encoding='utf-8') as handle:
            handle.write('Bonjour le monde.')
        stats = {'total_chunks': 1, 'completed_chunks': 1, 'failed_chunks': 0}
        ctx.emit(stats)
        return JobResult.from_run(stats, job.output_path, engine_ok=True)

    state, checkpoint_manager, _seen, output_dir = await _run(tmp_path, monkeypatch, writes)

    assert state.fields['status'] == 'completed'
    assert ('mark_error', 'trans_1') not in checkpoint_manager.calls
    assert ('mark_completed', 'trans_1') in checkpoint_manager.calls
    assert [p.name for p in output_dir.iterdir()] == ['out.txt']


def test_wrapper_error_marks_db_error(monkeypatch, tmp_path):
    checkpoint_manager = FakeCheckpointManager()
    state = FakeStateManager(checkpoint_manager)
    monkeypatch.setattr(handlers, 'emit_update', lambda *a, **k: None)

    async def crash(*args, **kwargs):
        raise RuntimeError('crash before the job body')

    monkeypatch.setattr(handlers, 'perform_actual_translation', crash)
    handlers.run_translation_async_wrapper('trans_2', _job(tmp_path), state, '.', FakeSocketIO())

    assert state.fields['status'] == 'error'
    assert ('mark_error', 'trans_2') in checkpoint_manager.calls
