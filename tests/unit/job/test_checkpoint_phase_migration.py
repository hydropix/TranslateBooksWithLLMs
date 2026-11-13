"""Schema migration: job phase, owner heartbeat and the refine chunk namespace."""

import os
import sqlite3
import time

from src.persistence.checkpoint_manager import CheckpointManager
from src.persistence.database import owner_still_running


def _old_schema_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE translation_jobs (
            translation_id TEXT PRIMARY KEY, status TEXT NOT NULL, file_type TEXT NOT NULL,
            config JSON NOT NULL, progress JSON NOT NULL, translation_context JSON,
            server_session_id TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, paused_at TIMESTAMP,
            completed_at TIMESTAMP, owner_pid INTEGER);
        CREATE TABLE checkpoint_chunks (
            translation_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
            original_text TEXT NOT NULL, translated_text TEXT, chunk_data JSON,
            status TEXT NOT NULL, completed_at TIMESTAMP,
            PRIMARY KEY (translation_id, chunk_index),
            FOREIGN KEY (translation_id) REFERENCES translation_jobs(translation_id)
                ON DELETE CASCADE);
        INSERT INTO translation_jobs (translation_id, status, file_type, config, progress)
        VALUES ('trans_old', 'interrupted', 'srt', '{"output_filename": "o.srt"}',
                '{"current_chunk_index": 1, "total_chunks": 4, "completed_chunks": 2,
                  "failed_chunks": 0, "resume_index_semantics": "completed"}');
        INSERT INTO checkpoint_chunks VALUES ('trans_old', 0, 'a', 'A', NULL, 'completed', NULL);
        INSERT INTO checkpoint_chunks VALUES ('trans_old', 1, 'b', 'B', NULL, 'completed', NULL);
    """)
    conn.commit()
    conn.close()


def test_old_database_is_migrated_and_stays_resumable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = str(tmp_path / "data" / "jobs.db")
    os.makedirs(os.path.dirname(db_path))
    _old_schema_db(db_path)

    manager = CheckpointManager(db_path=db_path)
    data = manager.load_checkpoint("trans_old")
    assert data["resume_from_index"] == 2
    assert [c["translated_text"] for c in data["chunks"]] == ["A", "B"]
    assert data["job_phase"] == "translate"
    assert data["refine_resume_index"] == 0


def test_refine_units_never_overwrite_translation_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / "data" / "jobs.db"))
    manager.start_job("trans_1", "srt", {"output_filename": "o.srt"})
    manager.save_checkpoint("trans_1", 0, "src0", "fr0", total_chunks=3, completed_chunks=1)
    manager.save_checkpoint("trans_1", 0, "fr0", "refined0", phase="refine")
    manager.save_checkpoint("trans_1", 1, "fr1", "refined1", phase="refine")
    manager.set_job_phase("trans_1", "refine")

    data = manager.load_checkpoint("trans_1")
    assert [c["translated_text"] for c in data["chunks"]] == ["fr0"]
    assert data["resume_from_index"] == 1  # the translation pointer did not move
    assert data["job_phase"] == "refine"
    assert data["refine_resume_index"] == 2


def test_owner_check_survives_pid_reuse(monkeypatch):
    import src.persistence.database as database
    monkeypatch.setattr(database, "is_process_alive", lambda pid: True)
    now = time.time()
    other = os.getpid() + 1
    assert owner_still_running(other, now - 10, now=now)
    # Same PID alive but no heartbeat for a long time: a reused PID, not our owner.
    assert not owner_still_running(other, now - 3600, now=now)
    # Legacy rows without a heartbeat keep the PID-only rule.
    assert owner_still_running(other, None, now=now)
    assert not owner_still_running(os.getpid(), now, now=now)
