"""Issue #341: CLI jobs stored before v1.6.0 must still resume.

Those jobs persisted the slim config written by ``GenericTranslator``
(``input_file_path``, ``output_file_path``, ``model_name``, no ``origin``,
no ``output_filename``/``output_dir``, no ``file_type``).
``JobConfig.from_persisted`` and ``translate._resume_job`` rebuild them.
"""
import os
from types import SimpleNamespace

from src.core.job import JobConfig

import translate


def _legacy_config(tmp_path):
    src = tmp_path / "book.epub"
    src.write_bytes(b"x")
    return {
        'input_file_path': str(src),
        'output_file_path': str(tmp_path / "out" / "book (French).epub"),
        'source_language': 'English',
        'target_language': 'French',
        'model_name': 'qwen3:14b',
        'llm_provider': 'ollama',
        'max_tokens_per_chunk': 450,
    }


def test_from_persisted_rebuilds_legacy_cli_config(tmp_path):
    stored = _legacy_config(tmp_path)
    job = JobConfig.from_persisted(stored)

    assert job.origin == 'cli'
    assert job.output_path == stored['output_file_path']
    assert job.model == 'qwen3:14b'
    assert job.input_path == stored['input_file_path']
    assert job.max_tokens_per_chunk == 450
    # The legacy keys are consumed, not carried as extras and re-persisted.
    persisted = job.to_persisted()
    for key in ('output_file_path', 'input_file_path', 'model_name'):
        assert key not in persisted
    assert persisted['output_filename'] == 'book (French).epub'
    assert persisted['output_dir'] == str(tmp_path / "out")


def test_from_persisted_does_not_mutate_caller_dict(tmp_path):
    stored = _legacy_config(tmp_path)
    before = dict(stored)
    JobConfig.from_persisted(stored)
    assert stored == before


def test_current_config_is_unchanged(tmp_path):
    stored = {
        'output_filename': 'a (French).txt', 'output_dir': str(tmp_path),
        'origin': 'cli', 'model': 'm', 'llm_provider': 'ollama', 'file_type': 'txt',
        # A stale legacy key must not override the current shape.
        'output_file_path': str(tmp_path / 'other.txt'),
    }
    job = JobConfig.from_persisted(stored)
    assert job.output_path == os.path.join(str(tmp_path), 'a (French).txt')


def test_resume_job_takes_file_type_from_job_row(tmp_path):
    stored = _legacy_config(tmp_path)

    class FakeCheckpoints:
        def load_checkpoint(self, _id):
            return {'job': {'file_type': 'epub', 'config': stored}, 'chunks': []}

    args = SimpleNamespace(resume='cli_legacy')
    job = translate._resume_job(args, FakeCheckpoints())
    assert job.file_type == 'epub'
    assert job.origin == 'cli'
    assert job.output_path == stored['output_file_path']
