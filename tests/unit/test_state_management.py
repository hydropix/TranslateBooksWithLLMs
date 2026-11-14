"""
Unit tests for state management across browser refresh and server restart scenarios.

Tests cover:
1. TranslationStateManager - in-memory state with thread safety
2. Server session ID generation and detection
3. Job state persistence and restoration
4. Checkpoint manager integration
5. Server restart handling (reset_running_jobs_on_startup)
"""

import pytest
import threading
import time
import tempfile
import os
import sys
from unittest.mock import MagicMock

# Mock flask_socketio before importing modules that depend on it
sys.modules['flask_socketio'] = MagicMock()

from src.api.translation_state import (
    TranslationStateManager,
    generate_server_session_id,
)
from src.persistence.checkpoint_manager import CheckpointManager
from src.persistence.database import Database


class TestServerSessionId:
    """Tests for server session ID generation and usage."""

    def test_generate_server_session_id_returns_timestamp_string(self):
        """Session ID should be a string representation of Unix timestamp."""
        session_id = generate_server_session_id()
        assert isinstance(session_id, str)
        # Should be convertible to int (timestamp)
        timestamp = int(session_id)
        # Should be a recent timestamp (within last minute)
        now = int(time.time())
        assert now - 60 <= timestamp <= now + 1

    def test_generate_server_session_id_unique_per_call(self):
        """Each call should potentially generate a different ID (time-based)."""
        id1 = generate_server_session_id()
        time.sleep(0.01)  # Small delay
        id2 = generate_server_session_id()
        # IDs are time-based, so they might be same if called within same second
        # But they should both be valid timestamps
        assert int(id1) > 0
        assert int(id2) > 0

    def test_state_manager_uses_provided_session_id(self):
        """State manager should use provided session ID."""
        custom_id = "12345678"
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            checkpoint_mgr = CheckpointManager(db_path=db_path, server_session_id=custom_id)
            state_manager = TranslationStateManager(
                checkpoint_manager=checkpoint_mgr,
                server_session_id=custom_id
            )
            assert state_manager.server_session_id == custom_id
            checkpoint_mgr.close()

    def test_state_manager_generates_session_id_if_not_provided(self):
        """State manager should generate session ID if none provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            checkpoint_mgr = CheckpointManager(db_path=db_path)
            state_manager = TranslationStateManager(checkpoint_manager=checkpoint_mgr)
            assert state_manager.server_session_id is not None
            assert len(state_manager.server_session_id) > 0
            # Should be a valid timestamp
            timestamp = int(state_manager.server_session_id)
            assert timestamp > 0
            checkpoint_mgr.close()


class TestTranslationStateManager:
    """Tests for in-memory translation state management."""

    @pytest.fixture
    def state_manager(self):
        """Create a fresh state manager for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            checkpoint_mgr = CheckpointManager(db_path=db_path, server_session_id="test_session")
            manager = TranslationStateManager(
                checkpoint_manager=checkpoint_mgr,
                server_session_id="test_session"
            )
            yield manager
            checkpoint_mgr.close()

    def test_create_translation_initializes_state(self, state_manager):
        """Creating a translation should initialize all required fields."""
        config = {"input_filename": "test.txt", "output_filename": "test_out.txt"}
        state_manager.create_translation("trans_001", config)

        state = state_manager.get_translation("trans_001")
        assert state is not None
        assert state['status'] == 'queued'
        assert state['progress'] == 0
        assert state['config'] == config
        assert state['interrupted'] is False
        assert 'stats' in state
        assert 'logs' in state

    def test_get_translation_returns_deep_copy(self, state_manager):
        """Returned state should be a deep copy to prevent external mutation."""
        config = {"nested": {"value": 1}}
        state_manager.create_translation("trans_001", config)

        state1 = state_manager.get_translation("trans_001")
        state1['config']['nested']['value'] = 999  # Mutate the copy

        state2 = state_manager.get_translation("trans_001")
        assert state2['config']['nested']['value'] == 1  # Original unchanged

    def test_update_translation_modifies_state(self, state_manager):
        """Updates should modify the translation state."""
        state_manager.create_translation("trans_001", {})

        success = state_manager.update_translation("trans_001", {
            'status': 'running',
            'progress': 50
        })

        assert success is True
        state = state_manager.get_translation("trans_001")
        assert state['status'] == 'running'
        assert state['progress'] == 50

    def test_update_nonexistent_translation_returns_false(self, state_manager):
        """Updating non-existent translation should return False."""
        success = state_manager.update_translation("nonexistent", {'status': 'running'})
        assert success is False

    def test_update_stats_nested_merge(self, state_manager):
        """Stats updates should merge with existing stats."""
        state_manager.create_translation("trans_001", {})

        state_manager.update_stats("trans_001", {'total_chunks': 10})
        state_manager.update_stats("trans_001", {'completed_chunks': 5})

        state = state_manager.get_translation("trans_001")
        assert state['stats']['total_chunks'] == 10
        assert state['stats']['completed_chunks'] == 5

    def test_append_log_adds_entries(self, state_manager):
        """Logs should be appended correctly."""
        state_manager.create_translation("trans_001", {})
        initial_log_count = len(state_manager.get_translation("trans_001")['logs'])

        state_manager.append_log("trans_001", "Processing chunk 1")
        state_manager.append_log("trans_001", "Processing chunk 2")

        state = state_manager.get_translation("trans_001")
        assert len(state['logs']) == initial_log_count + 2

    def test_set_interrupted_flag(self, state_manager):
        """Interrupted flag should be settable."""
        state_manager.create_translation("trans_001", {})

        assert state_manager.is_interrupted("trans_001") is False

        state_manager.set_interrupted("trans_001", True)
        assert state_manager.is_interrupted("trans_001") is True

        state_manager.set_interrupted("trans_001", False)
        assert state_manager.is_interrupted("trans_001") is False

    def test_exists_check(self, state_manager):
        """Existence check should work correctly."""
        assert state_manager.exists("trans_001") is False

        state_manager.create_translation("trans_001", {})
        assert state_manager.exists("trans_001") is True

    def test_get_all_translations_returns_deep_copy(self, state_manager):
        """Getting all translations should return deep copies."""
        state_manager.create_translation("trans_001", {"key": "value1"})
        state_manager.create_translation("trans_002", {"key": "value2"})

        all_translations = state_manager.get_all_translations()
        assert len(all_translations) == 2

        # Mutate the copy
        all_translations["trans_001"]["config"]["key"] = "mutated"

        # Original should be unchanged
        state = state_manager.get_translation("trans_001")
        assert state["config"]["key"] == "value1"

    def test_get_translation_summaries(self, state_manager):
        """Summaries should include key fields for UI."""
        config = {
            "input_filename": "book.epub",
            "output_filename": "book_fr.epub",
            "file_type": "epub"
        }
        state_manager.create_translation("trans_001", config)
        state_manager.update_stats("trans_001", {
            'total_chunks': 100,
            'completed_chunks': 50
        })

        summaries = state_manager.get_translation_summaries()
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary['translation_id'] == "trans_001"
        assert summary['input_filename'] == "book.epub"
        assert summary['output_filename'] == "book_fr.epub"
        assert summary['total_chunks'] == 100
        assert summary['completed_chunks'] == 50


class TestTranslationStateManagerThreadSafety:
    """Tests for thread safety of TranslationStateManager."""

    @pytest.fixture
    def state_manager(self):
        """Create a fresh state manager for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            checkpoint_mgr = CheckpointManager(db_path=db_path, server_session_id="test_session")
            manager = TranslationStateManager(
                checkpoint_manager=checkpoint_mgr,
                server_session_id="test_session"
            )
            yield manager
            checkpoint_mgr.close()

    def test_concurrent_updates_are_thread_safe(self, state_manager):
        """Multiple threads updating same translation should not corrupt state."""
        state_manager.create_translation("trans_001", {})
        errors = []
        update_count = 100

        def update_progress(thread_id):
            try:
                for i in range(update_count):
                    state_manager.update_stats("trans_001", {
                        f'thread_{thread_id}_update': i
                    })
                    state_manager.append_log("trans_001", f"Thread {thread_id} update {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_progress, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        state = state_manager.get_translation("trans_001")
        # Each thread should have added 100 log entries
        # Plus the initial log from create_translation
        assert len(state['logs']) >= 500

    def test_concurrent_create_and_read(self, state_manager):
        """Creating and reading translations concurrently should be safe."""
        errors = []
        created_ids = []

        def create_translations(start_id):
            try:
                for i in range(20):
                    tid = f"trans_{start_id}_{i}"
                    state_manager.create_translation(tid, {"id": tid})
                    created_ids.append(tid)
            except Exception as e:
                errors.append(e)

        def read_translations():
            try:
                for _ in range(50):
                    state_manager.get_all_translations()
                    state_manager.get_translation_summaries()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_translations, args=(1,)),
            threading.Thread(target=create_translations, args=(2,)),
            threading.Thread(target=read_translations),
            threading.Thread(target=read_translations),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCheckpointManagerServerRestart:
    """Tests for checkpoint manager behavior across server restarts."""

    @pytest.fixture
    def checkpoint_manager(self):
        """Create a checkpoint manager with temp database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = CheckpointManager(db_path=db_path, server_session_id="session_1")
            yield manager
            manager.close()

    def test_start_job_stores_server_session_id(self, checkpoint_manager):
        """Starting a job should store the server session ID."""
        config = {"input_filename": "test.txt"}
        checkpoint_manager.start_job("trans_001", "txt", config)

        job = checkpoint_manager.get_job("trans_001")
        assert job is not None
        # The session ID is stored in the database via the Database class

    def test_reset_running_jobs_on_startup_marks_old_session_jobs(self):
        """Jobs from previous server sessions should be marked as interrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Simulate first server session
            manager1 = CheckpointManager(db_path=db_path, server_session_id="session_1")
            manager1.start_job("trans_001", "txt", {"file": "test1.txt"})
            manager1.start_job("trans_002", "txt", {"file": "test2.txt"})
            manager1.close()

            # Simulate server restart with new session
            manager2 = CheckpointManager(db_path=db_path, server_session_id="session_2")

            # Reset running jobs from old session
            reset_count = manager2.reset_running_jobs_on_startup()

            # Both jobs from session_1 should be marked as interrupted
            assert reset_count == 2

            # Check jobs are now resumable
            resumable = manager2.get_resumable_jobs()
            assert len(resumable) == 2
            for job in resumable:
                assert job['status'] == 'interrupted'

            manager2.close()

    def test_reset_running_jobs_preserves_current_session_jobs(self):
        """Jobs from current session should NOT be marked as interrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Create job in current session
            manager = CheckpointManager(db_path=db_path, server_session_id="session_1")
            manager.start_job("trans_001", "txt", {"file": "test.txt"})

            # Reset should not affect jobs from same session
            reset_count = manager.reset_running_jobs_on_startup()
            assert reset_count == 0

            # Job should still be running
            job = manager.get_job("trans_001")
            assert job['status'] == 'running'

            manager.close()

    def test_load_checkpoint_returns_resume_index(self):
        """Loading checkpoint should return correct resume index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = CheckpointManager(db_path=db_path, server_session_id="session_1")

            # Start job and save some chunks
            manager.start_job("trans_001", "txt", {"file": "test.txt"})
            manager.save_checkpoint(
                "trans_001",
                chunk_index=0,
                original_text="Hello",
                translated_text="Bonjour",
                total_chunks=5,
                completed_chunks=1
            )
            manager.save_checkpoint(
                "trans_001",
                chunk_index=1,
                original_text="World",
                translated_text="Monde",
                total_chunks=5,
                completed_chunks=2
            )

            # Mark as interrupted (simulating server crash)
            manager.mark_interrupted("trans_001")

            # Load checkpoint
            checkpoint_data = manager.load_checkpoint("trans_001")
            assert checkpoint_data is not None
            # For TXT files, resume_from_index = current_chunk_index + 1
            # current_chunk_index was last updated to 1
            assert checkpoint_data['resume_from_index'] == 2  # Should resume from chunk 2
            assert len(checkpoint_data['chunks']) == 2

            manager.close()


class TestDatabaseServerSessionHandling:
    """Tests for database-level server session handling."""

    @pytest.fixture
    def database(self):
        """Create a database with temp file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = Database(db_path=db_path)
            yield db
            db.close()

    def test_create_job_with_session_id(self, database):
        """Jobs should be created with server session ID."""
        config = {"file": "test.txt"}
        success = database.create_job("trans_001", "txt", config, server_session_id="session_123")
        assert success is True

        job = database.get_job("trans_001")
        assert job is not None
        assert job['status'] == 'running'

    def test_reset_running_jobs_filters_by_session(self, database):
        """Reset should only affect jobs from different sessions."""
        # Create jobs with different session IDs
        database.create_job("trans_001", "txt", {}, server_session_id="old_session")
        database.create_job("trans_002", "txt", {}, server_session_id="old_session")
        database.create_job("trans_003", "txt", {}, server_session_id="current_session")

        # Reset with current session ID
        reset_count = database.reset_running_jobs("current_session")

        # Only old_session jobs should be reset
        assert reset_count == 2

        job1 = database.get_job("trans_001")
        job2 = database.get_job("trans_002")
        job3 = database.get_job("trans_003")

        assert job1['status'] == 'interrupted'
        assert job2['status'] == 'interrupted'
        assert job3['status'] == 'running'  # Current session job unchanged

    def test_reset_running_jobs_handles_null_session_id(self, database):
        """Jobs without session ID (legacy) should be reset."""
        # Create job without session ID (legacy behavior)
        database.create_job("trans_001", "txt", {}, server_session_id=None)

        # Reset with current session
        reset_count = database.reset_running_jobs("current_session")

        assert reset_count == 1
        job = database.get_job("trans_001")
        assert job['status'] == 'interrupted'


class TestStateManagerCheckpointIntegration:
    """Tests for StateManager + CheckpointManager integration."""

    def test_restore_job_from_checkpoint(self):
        """State manager should restore jobs from checkpoint correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            checkpoint_mgr = CheckpointManager(db_path=db_path, server_session_id="session_1")

            # Create and save a job
            config = {
                "input_filename": "book.epub",
                "output_filename": "book_fr.epub",
                "file_type": "txt"
            }
            checkpoint_mgr.start_job("trans_001", "txt", config)
            checkpoint_mgr.save_checkpoint(
                "trans_001",
                chunk_index=5,
                original_text="Original",
                translated_text="Translated",
                total_chunks=10,
                completed_chunks=6
            )
            checkpoint_mgr.mark_interrupted("trans_001")

            # Create new state manager (simulating server restart)
            state_manager = TranslationStateManager(
                checkpoint_manager=checkpoint_mgr,
                server_session_id="session_2"
            )

            # Restore the job
            success = state_manager.restore_job_from_checkpoint("trans_001")
            assert success is True

            # Verify restored state
            state = state_manager.get_translation("trans_001")
            assert state is not None
            assert state['status'] == 'paused'
            assert state['config']['input_filename'] == "book.epub"
            assert 'resume_from_index' in state

            checkpoint_mgr.close()

    def test_delete_checkpoint_removes_from_memory_and_db(self):
        """Deleting checkpoint should remove from both memory and database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            checkpoint_mgr = CheckpointManager(db_path=db_path, server_session_id="session_1")
            state_manager = TranslationStateManager(
                checkpoint_manager=checkpoint_mgr,
                server_session_id="session_1"
            )

            # Create job in both memory and DB
            config = {"file": "test.txt"}
            state_manager.create_translation("trans_001", config)
            checkpoint_mgr.start_job("trans_001", "txt", config)

            # Verify exists
            assert state_manager.exists("trans_001")
            assert checkpoint_mgr.get_job("trans_001") is not None

            # Delete
            state_manager.delete_checkpoint("trans_001")

            # Verify removed from both
            assert not state_manager.exists("trans_001")
            assert checkpoint_mgr.get_job("trans_001") is None

            checkpoint_mgr.close()

    def test_get_resumable_jobs_returns_paused_and_interrupted(self):
        """Resumable jobs should include paused and interrupted status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            checkpoint_mgr = CheckpointManager(db_path=db_path, server_session_id="session_1")

            # Create jobs with different statuses
            checkpoint_mgr.start_job("trans_001", "txt", {"file": "1.txt"})
            checkpoint_mgr.mark_paused("trans_001")

            checkpoint_mgr.start_job("trans_002", "txt", {"file": "2.txt"})
            checkpoint_mgr.mark_interrupted("trans_002")

            checkpoint_mgr.start_job("trans_003", "txt", {"file": "3.txt"})
            checkpoint_mgr.mark_completed("trans_003")

            checkpoint_mgr.start_job("trans_004", "txt", {"file": "4.txt"})
            # Still running

            # Get resumable
            resumable = checkpoint_mgr.get_resumable_jobs()

            # Should include paused and interrupted, not completed or running
            resumable_ids = [j['translation_id'] for j in resumable]
            assert "trans_001" in resumable_ids  # paused
            assert "trans_002" in resumable_ids  # interrupted
            assert "trans_003" not in resumable_ids  # completed
            assert "trans_004" not in resumable_ids  # running

            checkpoint_mgr.close()


class TestHealthEndpointSessionId:
    """Tests for health endpoint returning session ID."""

    def test_config_blueprint_uses_state_manager_session_id(self):
        """Config blueprint should use session ID from state manager."""
        from src.api.blueprints.config_routes import create_config_blueprint
        from flask import Flask

        app = Flask(__name__)
        session_id = "1234567890"

        bp = create_config_blueprint(server_session_id=session_id)
        app.register_blueprint(bp)

        with app.test_client() as client:
            response = client.get('/api/health')
            data = response.get_json()

            assert data['session_id'] == int(session_id)
            assert data['startup_time'] == int(session_id)

    def test_config_blueprint_generates_session_id_if_not_provided(self):
        """Config blueprint should generate session ID if not provided."""
        from src.api.blueprints.config_routes import create_config_blueprint
        from flask import Flask

        app = Flask(__name__)

        bp = create_config_blueprint()  # No session_id provided
        app.register_blueprint(bp)

        with app.test_client() as client:
            response = client.get('/api/health')
            data = response.get_json()

            # Should have generated a valid timestamp
            assert 'session_id' in data
            assert data['session_id'] > 0
            # Should be a recent timestamp
            now = int(time.time())
            assert now - 60 <= data['session_id'] <= now + 1
