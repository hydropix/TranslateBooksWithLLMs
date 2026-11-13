"""Regression tests: server startup must not destroy or flip other processes' jobs.

cleanup_orphan_uploads deleted every data/uploads folder not named trans_*,
including the folder of a CLI job (cli_<id>) that still had a DB row, and
reset_running_jobs flipped CLI rows (no server session) to 'interrupted'
while the CLI process could still be running.
"""
import os
import subprocess
import sys
import time

import pytest

from src.persistence.checkpoint_manager import CheckpointManager
from src.persistence.database import is_process_alive

OLD = time.time() - 2 * CheckpointManager.ORPHAN_GRACE_SECONDS


@pytest.fixture
def manager(tmp_path, monkeypatch):
    # CheckpointManager resolves data/uploads against the cwd.
    monkeypatch.chdir(tmp_path)
    cm = CheckpointManager(db_path=str(tmp_path / 'data' / 'jobs.db'), server_session_id='server_2')
    yield cm
    cm.close()


def _make_dir(path, age=OLD):
    path.mkdir(parents=True)
    (path / 'book.txt').write_text('x', encoding='utf-8')
    os.utime(path, (age, age))
    return path


def test_cleanup_keeps_cli_job_folder_with_db_row(manager):
    manager.db.create_job('cli_1234abcd', 'txt', {}, server_session_id=None)
    cli_dir = _make_dir(manager.uploads_dir / 'cli_1234abcd')
    web_dir = _make_dir(manager.uploads_dir / 'trans_1')
    manager.db.create_job('trans_1', 'txt', {}, server_session_id='server_1')
    orphan = _make_dir(manager.uploads_dir / 'cli_deadbeef')
    orphan_web = _make_dir(manager.uploads_dir / 'trans_2')

    deleted = manager.cleanup_orphan_uploads()

    assert cli_dir.exists() and web_dir.exists()
    assert not orphan.exists() and not orphan_web.exists()
    assert deleted == 2


def test_cleanup_keeps_freshly_created_folders(manager):
    # Another process creates the folder just before inserting its job row.
    fresh = _make_dir(manager.uploads_dir / 'cli_fresh000', age=time.time())
    assert manager.cleanup_orphan_uploads() == 0
    assert fresh.exists()


def test_is_process_alive():
    assert is_process_alive(os.getpid())
    assert not is_process_alive(None)
    assert not is_process_alive(0)


def test_reset_leaves_jobs_of_a_live_process_alone(manager):
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        manager.db.create_job('cli_live0000', 'txt', {}, server_session_id=None,
                              owner_pid=other.pid)
        manager.db.create_job('trans_old', 'txt', {}, server_session_id='server_1')

        assert manager.reset_running_jobs_on_startup() == 1
        assert manager.get_job('cli_live0000')['status'] == 'running'
        assert manager.get_job('trans_old')['status'] == 'interrupted'
    finally:
        other.kill()
        other.wait()


def test_reset_flips_jobs_of_a_dead_process(manager):
    finished = subprocess.Popen([sys.executable, '-c', 'pass'])
    finished.wait()
    manager.db.create_job('cli_dead0000', 'txt', {}, server_session_id=None,
                          owner_pid=finished.pid)
    assert manager.reset_running_jobs_on_startup() == 1
    assert manager.get_job('cli_dead0000')['status'] == 'interrupted'


def test_start_job_records_the_owner_pid(manager):
    manager.start_job('trans_9', 'txt', {})
    row = manager.db._get_connection().execute(
        "SELECT owner_pid FROM translation_jobs WHERE translation_id = 'trans_9'").fetchone()
    assert row['owner_pid'] == os.getpid()
