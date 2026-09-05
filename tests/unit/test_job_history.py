"""
Tests for the completed-job history write path (issue #271, Tier 3).

A successful job must leave a 'completed' row in translation_jobs so it can be
listed as history, while its checkpoint chunks and its data/uploads/<id>/
directory are pruned. The user-initiated delete path stays a full delete.

Sections:
  1. Persistence layer: Database / CheckpointManager (this file, below).
  2. HTTP layer: GET /api/history (added by a later phase; append it at the
     bottom so the fixtures above can be reused untouched).
"""

import json
import sqlite3

import pytest
from flask import Flask

from src.api.blueprints.translation_routes import (
    _history_item,
    create_translation_blueprint,
)
from src.api.translation_state import TranslationStateManager
from src.persistence.checkpoint_manager import CheckpointManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cm(tmp_path, monkeypatch):
    """
    A CheckpointManager fully contained in tmp_path.

    CheckpointManager hardcodes a RELATIVE uploads directory ("data/uploads")
    and mkdir()s it in __init__, so the working directory is moved first to
    keep the test from writing into the repository, and the attribute is then
    pinned explicitly for readability.
    """
    monkeypatch.chdir(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / 'jobs.db'))
    manager.uploads_dir = tmp_path / 'uploads'
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)
    yield manager
    manager.close()


def _make_job(cm, translation_id, config=None, file_type='txt'):
    """Create a running job with a chunk and a populated upload directory."""
    assert cm.db.create_job(translation_id, file_type, config or {})
    cm.db.save_chunk(translation_id, 0, 'source text', 'translated text')

    upload_dir = cm.uploads_dir / translation_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / 'input.txt').write_text('source text', encoding='utf-8')
    return upload_dir


def _raw_row(cm, translation_id):
    """Read a translation_jobs row straight from SQLite, bypassing the API."""
    conn = sqlite3.connect(cm.db.db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM translation_jobs WHERE translation_id = ?",
            (translation_id,)
        )
        return cursor.fetchone()
    finally:
        conn.close()


def _set_completed_at(cm, translation_id, value):
    conn = sqlite3.connect(cm.db.db_path)
    try:
        conn.execute(
            "UPDATE translation_jobs SET completed_at = ? WHERE translation_id = ?",
            (value, translation_id)
        )
        conn.commit()
    finally:
        conn.close()


def _age_job(cm, translation_id, days):
    conn = sqlite3.connect(cm.db.db_path)
    try:
        conn.execute(
            "UPDATE translation_jobs "
            "SET created_at = datetime('now', ? || ' days') "
            "WHERE translation_id = ?",
            (f'-{days}', translation_id)
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. Persistence layer
# ---------------------------------------------------------------------------

class TestPruneJobData:
    """The success path keeps the row and drops only the resume data."""

    def test_completed_job_keeps_its_row_and_loses_its_resume_data(self, cm):
        upload_dir = _make_job(cm, 'job-a')

        assert cm.mark_completed('job-a') is True
        assert cm.prune_job_data('job-a') is True

        row = _raw_row(cm, 'job-a')
        assert row is not None, "the completed row must survive pruning"
        assert row['status'] == 'completed'
        assert row['completed_at'] is not None

        assert cm.db.get_chunks('job-a') == []
        assert not upload_dir.exists()

        # A finished job must never show up in a list of unfinished work.
        assert cm.get_resumable_jobs() == []

    def test_prune_job_data_never_touches_the_jobs_table(self, cm):
        _make_job(cm, 'job-a')
        cm.mark_completed('job-a')

        # Even called twice, and even with no upload directory left.
        assert cm.prune_job_data('job-a') is True
        assert cm.prune_job_data('job-a') is True
        assert _raw_row(cm, 'job-a') is not None

    def test_prune_job_data_tolerates_a_missing_upload_directory(self, cm):
        assert cm.db.create_job('job-a', 'txt', {})
        assert cm.prune_job_data('job-a') is True

    def test_delete_checkpoint_still_removes_the_row(self, cm):
        """User-initiated cleanup stays a full delete."""
        upload_dir = _make_job(cm, 'job-a')
        cm.mark_completed('job-a')

        assert cm.delete_checkpoint('job-a') is True

        assert _raw_row(cm, 'job-a') is None
        assert cm.db.get_chunks('job-a') == []
        assert not upload_dir.exists()


class TestJobHistoryQuery:
    """Database.get_job_history ordering, filtering and clamping."""

    def test_only_completed_jobs_newest_first(self, cm):
        for job_id in ('older', 'newer', 'paused-job', 'error-job'):
            cm.db.create_job(job_id, 'txt', {})

        cm.mark_completed('older')
        cm.mark_completed('newer')
        cm.db.update_job_progress('paused-job', status='paused')
        cm.db.update_job_progress('error-job', status='error')

        _set_completed_at(cm, 'older', '2024-01-01 10:00:00')
        _set_completed_at(cm, 'newer', '2024-06-01 10:00:00')

        history = cm.db.get_job_history()
        ids = [job['translation_id'] for job in history]

        assert ids == ['newer', 'older']
        assert 'paused-job' not in ids
        assert 'error-job' not in ids
        assert all(job['status'] == 'completed' for job in history)
        assert history[0]['completed_at'] == '2024-06-01 10:00:00'

    def test_limit_is_honoured_and_clamped(self, cm):
        for index in range(3):
            job_id = f'job-{index}'
            cm.db.create_job(job_id, 'txt', {})
            cm.mark_completed(job_id)
            _set_completed_at(cm, job_id, f'2024-01-0{index + 1} 10:00:00')

        assert len(cm.db.get_job_history(limit=1)) == 1

        # Lower bound: 0 is clamped up to 1, it must not mean "no rows".
        assert len(cm.db.get_job_history(limit=0)) == 1

        # Upper bound: 1000 is clamped down to 200. Asserted on the SQL
        # parameter, since materialising 201 rows would only be slower.
        captured = {}
        original_connection = cm.db._get_connection

        class _SpyConnection:
            def __init__(self, inner):
                self._inner = inner

            def cursor(self):
                inner_cursor = self._inner.cursor()

                class _SpyCursor:
                    def execute(_self, sql, params=()):
                        if 'FROM translation_jobs' in sql and 'LIMIT ?' in sql:
                            captured['limit'] = params[-1]
                        return inner_cursor.execute(sql, params)

                    def __getattr__(_self, name):
                        return getattr(inner_cursor, name)

                return _SpyCursor()

            def __getattr__(self, name):
                return getattr(self._inner, name)

        cm.db._get_connection = lambda: _SpyConnection(original_connection())
        try:
            cm.db.get_job_history(limit=1000)
            assert captured['limit'] == 200
            cm.db.get_job_history(limit=0)
            assert captured['limit'] == 1
        finally:
            cm.db._get_connection = original_connection


class TestJobHistoryEnrichment:
    """CheckpointManager.get_job_history display fields."""

    def _history_entry(self, cm, config):
        cm.db.create_job('job-a', 'epub', config)
        cm.mark_completed('job-a')
        cm.db.update_job_progress('job-a', total_chunks=10, completed_chunks=9)
        history = cm.get_job_history()
        assert len(history) == 1
        return history[0]

    def test_persisted_input_filename_wins(self, cm):
        entry = self._history_entry(cm, {
            'input_filename': 'Book.epub',
            'file_path': 'data/uploads/0123456789abcdef_Book.epub',
            'output_filename': 'Book_fr.epub',
        })

        assert entry['input_filename'] == 'Book.epub'
        assert entry['output_filename'] == 'Book_fr.epub'
        assert entry['total_chunks'] == 10
        assert entry['completed_chunks'] == 9

    def test_legacy_row_falls_back_to_the_path_without_its_upload_prefix(self, cm):
        entry = self._history_entry(cm, {
            'file_path': 'data/uploads/0123456789abcdef_old.txt',
        })

        assert entry['input_filename'] == 'old.txt'
        assert entry['output_filename'] == 'unknown'

    def test_preserved_input_path_is_used_when_file_path_is_absent(self, cm):
        entry = self._history_entry(cm, {
            'preserved_input_path': 'data/uploads/job-a/0123456789abcdef_kept.srt',
        })

        assert entry['input_filename'] == 'kept.srt'

    def test_missing_everything_reports_unknown(self, cm):
        entry = self._history_entry(cm, {})

        assert entry['input_filename'] == 'unknown'
        assert entry['output_filename'] == 'unknown'
        assert entry['total_chunks'] == 10

    def test_api_keys_are_never_exposed_by_the_history(self, cm):
        entry = self._history_entry(cm, {
            'input_filename': 'Book.epub',
            'api_key': 'sk-xxxxxxxx',
            'gemini_api_key': 'sk-xxxxxxxx',
        })

        serialized = json.dumps(entry)
        assert 'sk-xxxxxxxx' not in serialized


class TestCompletedRowInvariants:
    """Guardrails on the lifecycle of a kept 'completed' row."""

    def test_reset_running_jobs_leaves_a_completed_row_alone(self, cm):
        _make_job(cm, 'job-a')
        cm.mark_completed('job-a')
        cm.prune_job_data('job-a')

        cm.db.reset_running_jobs('a-brand-new-server-session')

        assert _raw_row(cm, 'job-a')['status'] == 'completed'
        assert cm.get_resumable_jobs() == []

    def test_cleanup_old_jobs_still_sweeps_old_completed_rows(self, cm):
        _make_job(cm, 'job-old')
        cm.mark_completed('job-old')
        cm.prune_job_data('job-old')
        _age_job(cm, 'job-old', days=40)

        _make_job(cm, 'job-recent')
        cm.mark_completed('job-recent')
        cm.prune_job_data('job-recent')

        deleted = cm.db.cleanup_old_jobs(max_age_days=30)

        assert deleted == 1
        assert _raw_row(cm, 'job-old') is None
        assert _raw_row(cm, 'job-recent') is not None
        assert [j['translation_id'] for j in cm.get_job_history()] == ['job-recent']


# ---------------------------------------------------------------------------
# 2. HTTP layer — GET /api/history
#    (added by a later phase; append below without touching the section above)
# ---------------------------------------------------------------------------

#: The complete set of fields the browser is allowed to see. Spelled out here
#: rather than imported from the route module so a silent widening of the
#: projection fails this test instead of following it.
HISTORY_KEYS = {
    'translation_id',
    'status',
    'file_type',
    'input_filename',
    'output_filename',
    'created_at',
    'updated_at',
    'completed_at',
    'total_chunks',
    'completed_chunks',
}


@pytest.fixture
def history_client(cm):
    """Flask test client exposing the translation blueprint over `cm`'s db.

    No auth gate is registered: /api/history stays gated in production by
    src/api/auth.py, which is exercised by its own tests. The `cm` fixture has
    already chdir'd into tmp_path, so no state lands in the repository.
    """
    state_manager = TranslationStateManager(checkpoint_manager=cm)

    app = Flask(__name__)
    app.register_blueprint(create_translation_blueprint(
        state_manager,
        lambda *args, **kwargs: None,
        str(cm.uploads_dir.parent),
    ))

    with app.test_client() as client:
        yield client


def _complete_job(cm, translation_id, config=None, file_type='txt', completed_at=None):
    """Seed a completed job through the manager, not through HTTP."""
    assert cm.db.create_job(translation_id, file_type, config or {})
    cm.mark_completed(translation_id)
    if completed_at is not None:
        _set_completed_at(cm, translation_id, completed_at)


class TestHistoryItemProjection:
    """_history_item is an allowlist, not a denylist."""

    def test_only_the_ten_display_fields_survive(self):
        item = _history_item({
            'translation_id': 'job-a',
            'status': 'completed',
            'file_type': 'epub',
            'input_filename': 'Book.epub',
            'output_filename': 'Book_fr.epub',
            'created_at': '2024-01-01 09:00:00',
            'updated_at': '2024-01-01 10:00:00',
            'completed_at': '2024-01-01 10:00:00',
            'total_chunks': 10,
            'completed_chunks': 10,
            # Everything below must be dropped.
            'config': {'file_path': '/srv/data/uploads/x_Book.epub'},
            'progress': {'total_chunks': 10},
            'file_path': '/srv/data/uploads/x_Book.epub',
            'server_session_id': 'session-1',
        })

        assert set(item.keys()) == HISTORY_KEYS
        assert item['translation_id'] == 'job-a'
        assert item['input_filename'] == 'Book.epub'
        assert item['total_chunks'] == 10

    def test_missing_fields_become_none_rather_than_raising(self):
        item = _history_item({'translation_id': 'job-a'})

        assert set(item.keys()) == HISTORY_KEYS
        assert item['completed_at'] is None


class TestHistoryEndpoint:
    """GET /api/history."""

    def test_empty_database_returns_an_empty_list(self, history_client):
        response = history_client.get('/api/history')

        assert response.status_code == 200
        assert response.get_json() == {"history": []}

    def test_completed_job_is_projected_without_config_paths_or_keys(self, cm, history_client):
        _complete_job(cm, 'job-a', {
            'input_filename': 'Book.epub',
            'output_filename': 'Book_fr.epub',
            'file_path': 'data/uploads/0123456789abcdef_Book.epub',
            'gemini_api_key': 'sk-xxxxxxxx',
            'llm_api_endpoint': 'https://generativelanguage.googleapis.com',
        }, file_type='epub')
        cm.db.update_job_progress('job-a', total_chunks=12, completed_chunks=12)

        response = history_client.get('/api/history')

        assert response.status_code == 200
        history = response.get_json()['history']
        assert len(history) == 1

        item = history[0]
        assert set(item.keys()) == HISTORY_KEYS
        assert item['translation_id'] == 'job-a'
        assert item['status'] == 'completed'
        assert item['file_type'] == 'epub'
        assert item['input_filename'] == 'Book.epub'
        assert item['output_filename'] == 'Book_fr.epub'
        assert item['total_chunks'] == 12
        assert item['completed_chunks'] == 12
        assert item['completed_at'] is not None

        # The raw wire bytes, not just the parsed item: nothing anywhere in the
        # response may hint at the server's filesystem or at a credential.
        body = response.get_data(as_text=True)
        assert 'file_path' not in body
        assert 'sk-xxxxxxxx' not in body
        assert '_api_key' not in body

    def test_non_integer_limit_is_rejected(self, history_client):
        response = history_client.get('/api/history?limit=abc')

        assert response.status_code == 400
        assert response.get_json() == {"error": "Invalid limit"}

    def test_limit_selects_the_most_recently_completed_job(self, cm, history_client):
        _complete_job(cm, 'older', {'input_filename': 'old.txt'},
                      completed_at='2024-01-01 10:00:00')
        _complete_job(cm, 'newer', {'input_filename': 'new.txt'},
                      completed_at='2024-06-01 10:00:00')

        response = history_client.get('/api/history?limit=1')

        assert response.status_code == 200
        history = response.get_json()['history']
        assert [item['translation_id'] for item in history] == ['newer']
