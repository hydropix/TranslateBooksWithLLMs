"""Regression test: a resume reuses the preset text the job started with.

The custom-instruction preset used to be re-read from disk after start_job and
again on every resume, so editing or deleting the preset mid-book changed the
instructions for the second half. The resolved text is now snapshotted into the
persisted config (``src.core.job.prepare.snapshot_custom_instructions``, run by
the job runner's prepare phase), like the glossary.
"""
import copy
import time

import pytest

from src.api import handlers
from src.core import adapters
from src.core.job import JobConfig, JobResult
from src.core.job.prepare import CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG, snapshot_custom_instructions
from src.core.llm.provider_config import ProviderConfig


class FakeCheckpointManager:
    def __init__(self):
        self.persisted = None

    def start_job(self, translation_id, file_type, config, input_file_path):
        self.persisted = copy.deepcopy(config)

    def update_job_config(self, translation_id, config):
        self.persisted = copy.deepcopy(config)
        return True

    def __getattr__(self, name):
        if name.startswith('mark_') or name in ('prune_job_data', 'set_job_phase'):
            return lambda translation_id, *args: None
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


async def _run(job, output_dir, monkeypatch, is_resume=False):
    checkpoint_manager = FakeCheckpointManager()
    seen = {}

    async def fake_translate_file(engine_job, ctx):
        seen['prompt_options'] = copy.deepcopy(dict(engine_job.prompt_options))
        with open(engine_job.output_path, 'w', encoding='utf-8') as handle:
            handle.write('done')
        stats = {'total_chunks': 1, 'completed_chunks': 1, 'failed_chunks': 0}
        ctx.emit(stats)
        return JobResult.from_run(stats, engine_job.output_path, engine_ok=True)

    monkeypatch.setattr(adapters, 'translate_file', fake_translate_file)
    await handlers.perform_actual_translation(
        'trans_1', job, FakeStateManager(checkpoint_manager), str(output_dir), FakeSocketIO(),
        is_resume=is_resume)
    return checkpoint_manager, seen


@pytest.mark.asyncio
async def test_resume_keeps_the_preset_text_the_job_started_with(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(handlers, 'emit_update', lambda *a, **k: None)
    monkeypatch.setattr('src.utils.notifier.notify', lambda *a, **k: None)
    presets = tmp_path / 'Custom_Instructions'
    presets.mkdir()
    preset = presets / 'tone.txt'
    preset.write_text('Keep the original tone.', encoding='utf-8')
    output_dir = tmp_path / 'out'
    output_dir.mkdir()

    job = JobConfig(
        text='Hello.', file_type='txt', source_language='English', target_language='French',
        provider=ProviderConfig(provider='ollama', model='m',
                                endpoint='http://localhost:11434/api/generate'),
        output_path=str(output_dir / 'out.txt'),
        prompt_options={'custom_instruction_file': 'tone.txt'},
        origin='web',
    )
    checkpoint_manager, first = await _run(job, output_dir, monkeypatch)

    persisted = checkpoint_manager.persisted
    assert 'Keep the original tone.' in persisted['prompt_options']['custom_instructions']
    assert persisted['prompt_options'][CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG] is True
    assert 'Keep the original tone.' in first['prompt_options']['custom_instructions']

    # The user edits, then deletes, the preset mid-book.
    preset.write_text('Rewrite everything as pirate speech.', encoding='utf-8')
    resumed_job = JobConfig.from_persisted(copy.deepcopy(persisted), output_dir=str(output_dir))
    _cm, resumed = await _run(resumed_job, output_dir, monkeypatch, is_resume=True)
    assert 'Keep the original tone.' in resumed['prompt_options']['custom_instructions']

    preset.unlink()
    resumed_job = JobConfig.from_persisted(copy.deepcopy(persisted), output_dir=str(output_dir))
    _cm, resumed = await _run(resumed_job, output_dir, monkeypatch, is_resume=True)
    assert 'Keep the original tone.' in resumed['prompt_options']['custom_instructions']


def test_snapshot_is_taken_once_and_flagged(tmp_path):
    (tmp_path / 'tone.txt').write_text('Be formal.', encoding='utf-8')
    options = {'custom_instruction_file': 'tone.txt'}
    logs = []

    snapshot_custom_instructions(options, tmp_path, lambda *a, **k: logs.append(a[0]))

    assert options['custom_instructions'] == 'Be formal.'
    assert options[CUSTOM_INSTRUCTIONS_SNAPSHOT_FLAG] is True
    assert logs == ['custom_instructions']

    # A flagged snapshot is never re-read, even when the file changes.
    (tmp_path / 'tone.txt').write_text('Be casual.', encoding='utf-8')
    snapshot_custom_instructions(options, tmp_path, lambda *a, **k: logs.append(a[0]))
    assert options['custom_instructions'] == 'Be formal.'
    assert logs == ['custom_instructions']


def test_legacy_config_without_the_flag_is_resolved_from_disk(tmp_path):
    (tmp_path / 'tone.txt').write_text('Be formal.', encoding='utf-8')
    options = {'custom_instruction_file': 'tone.txt'}

    snapshot_custom_instructions(options, tmp_path, lambda *a, **k: None)

    assert options['custom_instructions'] == 'Be formal.'


def test_no_preset_selected_sets_no_flag(tmp_path):
    options = {}
    snapshot_custom_instructions(options, tmp_path, lambda *a, **k: None)
    assert options == {}
