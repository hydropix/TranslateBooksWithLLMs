"""Unit tests for issue #213 follow-up: the jobs database must never store API keys.

Covers `sanitize_config_secrets` (the persistence-side strip), the actual DB
write paths (`create_job`, `update_job_config`), the resume-time credential
validation that replaces reading keys back from the checkpoint
(`JobConfig.from_checkpoint`), and `JobConfig.to_persisted`.
"""
import json
import sqlite3

import pytest

import src.config as config_module
from src.api.blueprints.translation_routes import _strip_api_keys
from src.core.job import JobConfig, JobConfigError
from src.persistence.database import Database, sanitize_config_secrets


SECRET_CONFIG = {
    'model': 'gemini-2.0-flash',
    'llm_provider': 'gemini',
    'llm_api_endpoint': '',
    'output_filename': 'book_fr.epub',
    'api_key': 'generic-secret',
    'gemini_api_key': 'AIza-secret',
    'openai_api_key': 'sk-secret',
    'openrouter_api_key': 'sk-or-1,sk-or-2',
    'mistral_api_key': 'mistral-secret',
    'deepseek_api_key': 'ds-secret',
    'poe_api_key': 'poe-secret',
    'nim_api_key': 'nim-secret',
    'prompt_options': {'glossary_id': 3},
}


def _read_raw_config(db_path, translation_id):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT config FROM translation_jobs WHERE translation_id = ?",
            (translation_id,),
        ).fetchone()
        return json.loads(row[0])
    finally:
        conn.close()


def _assert_no_secrets(config):
    leaked = [k for k in config if k == 'api_key' or k.endswith('_api_key')]
    assert leaked == [], f"API keys persisted in config: {leaked}"


class TestSanitizeConfigSecrets:
    def test_removes_every_api_key_field(self):
        sanitized = sanitize_config_secrets(SECRET_CONFIG)
        _assert_no_secrets(sanitized)

    def test_preserves_non_secret_fields(self):
        sanitized = sanitize_config_secrets(SECRET_CONFIG)
        assert sanitized['model'] == 'gemini-2.0-flash'
        assert sanitized['llm_provider'] == 'gemini'
        assert sanitized['output_filename'] == 'book_fr.epub'
        assert sanitized['prompt_options'] == {'glossary_id': 3}

    def test_does_not_mutate_input(self):
        config = dict(SECRET_CONFIG)
        sanitize_config_secrets(config)
        assert config['gemini_api_key'] == 'AIza-secret'


class TestDatabaseNeverStoresKeys:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "jobs.db")
        database = Database(db_path)
        yield database
        database.close()

    def test_create_job_strips_keys_from_persisted_config(self, db):
        assert db.create_job('trans_1', 'epub', dict(SECRET_CONFIG))
        raw = _read_raw_config(db.db_path, 'trans_1')
        _assert_no_secrets(raw)
        assert raw['model'] == 'gemini-2.0-flash'

    def test_create_job_leaves_caller_config_intact(self, db):
        config = dict(SECRET_CONFIG)
        db.create_job('trans_2', 'epub', config)
        # The live job still needs its keys in memory.
        assert config['gemini_api_key'] == 'AIza-secret'

    def test_update_job_config_strips_keys_too(self, db):
        db.create_job('trans_3', 'txt', {'model': 'llama3'})
        assert db.update_job_config('trans_3', dict(SECRET_CONFIG))
        raw = _read_raw_config(db.db_path, 'trans_3')
        _assert_no_secrets(raw)

    def test_get_job_returns_config_without_keys(self, db):
        db.create_job('trans_4', 'srt', dict(SECRET_CONFIG))
        job = db.get_job('trans_4')
        _assert_no_secrets(job['config'])


class TestResumeCredentialValidation:
    """Checkpoints carry no keys, so resume must validate .env/request instead."""

    @staticmethod
    def _no_env_key(monkeypatch, var):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(config_module, var, '', raising=False)

    def _restored_config(self):
        # What load_checkpoint now yields: a cloud-provider config, no keys.
        return {
            'model': 'gemini-2.0-flash',
            'llm_provider': 'gemini',
            'llm_api_endpoint': '',
            'output_filename': 'book_fr.epub',
            'file_type': 'epub',
        }

    @staticmethod
    def _resume(stored, overrides):
        return JobConfig.from_checkpoint(stored, output_dir='out', overrides=overrides)

    def test_empty_body_resume_without_env_key_is_rejected(self, monkeypatch):
        self._no_env_key(monkeypatch, 'GEMINI_API_KEY')
        with pytest.raises(JobConfigError) as info:
            self._resume(self._restored_config(), {})
        assert info.value.status == 400

    def test_empty_body_resume_with_env_key_passes(self, monkeypatch):
        monkeypatch.setenv('GEMINI_API_KEY', 'env-key')
        job = self._resume(self._restored_config(), {})
        assert job.provider.api_key == 'env-key'
        _assert_no_secrets(job.to_persisted())

    def test_key_in_resume_request_passes_without_env(self, monkeypatch):
        self._no_env_key(monkeypatch, 'GEMINI_API_KEY')
        job = self._resume(self._restored_config(), {'api_key': 'user-key'})
        assert job.provider.api_key == 'user-key'
        _assert_no_secrets(job.to_persisted())
        assert 'user-key' not in json.dumps(job.to_persisted())

    def test_local_openai_compatible_resume_needs_no_key(self, monkeypatch):
        self._no_env_key(monkeypatch, 'OPENAI_API_KEY')
        config = {
            'model': 'qwen3-14b',
            'llm_provider': 'openai',
            'llm_api_endpoint': 'http://localhost:1234/v1/chat/completions',
            'output_filename': 'book_fr.txt',
        }
        job = self._resume(config, {})
        assert not job.provider.api_key

    def test_official_openai_resume_without_key_is_rejected(self, monkeypatch):
        self._no_env_key(monkeypatch, 'OPENAI_API_KEY')
        config = {
            'model': 'gpt-4o',
            'llm_provider': 'openai',
            'llm_api_endpoint': 'https://api.openai.com/v1/chat/completions',
            'output_filename': 'book_fr.txt',
        }
        with pytest.raises(JobConfigError) as info:
            self._resume(config, {})
        assert info.value.status == 400


class TestJobConfigPersistence:
    def test_to_persisted_never_carries_a_key(self):
        stored = dict(SECRET_CONFIG)
        job = JobConfig.from_persisted(stored, output_dir='out')
        job = job.with_changes(provider=job.provider.with_changes(api_key='live-secret'))
        persisted = job.to_persisted()
        _assert_no_secrets(persisted)
        assert 'live-secret' not in json.dumps(persisted)
        assert 'AIza-secret' not in json.dumps(persisted)


class TestResponseStripping:
    def test_strip_api_keys_removes_all_keys_in_place(self):
        config = dict(SECRET_CONFIG)
        _strip_api_keys(config)
        _assert_no_secrets(config)
        assert config['model'] == 'gemini-2.0-flash'

    def test_strip_api_keys_tolerates_none(self):
        assert _strip_api_keys(None) is None
