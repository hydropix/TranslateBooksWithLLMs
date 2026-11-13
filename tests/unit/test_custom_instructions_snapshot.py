"""Regression test: a resume reuses the preset text the job started with.

The custom-instruction preset used to be re-read from disk after start_job and
again on every resume, so editing or deleting the preset mid-book changed the
instructions for the second half. The resolved text is now snapshotted into the
persisted config, like the glossary.
"""
import copy
import time

import pytest

from src.api import handlers


class FakeCheckpointManager:
    def __init__(self):
        self.persisted = None

    def start_job(self, translation_id, file_type, config, input_file_path):
        self.persisted = copy.deepcopy(config)

    def update_job_config(self, translation_id, config):
        return True

    def __getattr__(self, name):
        if name.startswith('mark_') or name == 'prune_job_data':
            return lambda translation_id: None
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


async def _run(config, output_dir, monkeypatch):
    checkpoint_manager = FakeCheckpointManager()
    seen = {}

    async def fake_translate_file(**kwargs):
        seen['prompt_options'] = copy.deepcopy(kwargs['prompt_options'])
        with open(kwargs['output_filepath'], 'w', encoding='utf-8') as handle:
            handle.write('done')

    monkeypatch.setattr(handlers, 'translate_file', fake_translate_file)
    await handlers.perform_actual_translation(
        'trans_1', config, FakeStateManager(checkpoint_manager), str(output_dir), FakeSocketIO())
    return checkpoint_manager, seen


@pytest.mark.asyncio
async def test_resume_keeps_the_preset_text_the_job_started_with(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(handlers, 'emit_update', lambda *a, **k: None)
    monkeypatch.setattr(handlers, 'notify', lambda *a, **k: None)
    presets = tmp_path / 'Custom_Instructions'
    presets.mkdir()
    preset = presets / 'tone.txt'
    preset.write_text('Keep the original tone.', encoding='utf-8')
    output_dir = tmp_path / 'out'
    output_dir.mkdir()

    config = {
        'text': 'Hello.', 'file_type': 'txt', 'source_language': 'English',
        'target_language': 'French', 'model': 'm', 'llm_provider': 'ollama',
        'llm_api_endpoint': 'http://localhost:11434/api/generate',
        'output_filename': 'out.txt',
        'prompt_options': {'custom_instruction_file': 'tone.txt'},
    }
    checkpoint_manager, first = await _run(config, output_dir, monkeypatch)

    persisted = checkpoint_manager.persisted
    assert 'Keep the original tone.' in persisted['prompt_options']['custom_instructions']
    assert 'Keep the original tone.' in first['prompt_options']['custom_instructions']

    # The user edits, then deletes, the preset mid-book.
    preset.write_text('Rewrite everything as pirate speech.', encoding='utf-8')
    resume_config = copy.deepcopy(persisted)
    resume_config.update({'is_resume': True, 'resume_from_index': 1})
    _cm, resumed = await _run(resume_config, output_dir, monkeypatch)
    assert 'Keep the original tone.' in resumed['prompt_options']['custom_instructions']

    preset.unlink()
    resume_config = copy.deepcopy(persisted)
    resume_config.update({'is_resume': True, 'resume_from_index': 1})
    _cm, resumed = await _run(resume_config, output_dir, monkeypatch)
    assert 'Keep the original tone.' in resumed['prompt_options']['custom_instructions']
