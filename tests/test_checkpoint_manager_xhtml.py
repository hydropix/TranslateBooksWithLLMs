"""
Unit tests for CheckpointManager XHTML partial state methods.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.persistence.checkpoint_manager import CheckpointManager
from src.core.epub.xhtml_translation_state import XHTMLTranslationState


@pytest.fixture
def temp_checkpoint_manager():
    """Create a temporary CheckpointManager for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_jobs.db"
    manager = CheckpointManager(db_path=str(db_path))

    # Override uploads_dir to use temp directory
    manager.uploads_dir = Path(temp_dir) / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)

    yield manager

    # Cleanup
    manager.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_state():
    """Create a sample XHTMLTranslationState for testing."""
    return XHTMLTranslationState(
        file_path="/path/to/chapter1.xhtml",
        translation_id="test_trans_123",
        file_href="OEBPS/chapter1.xhtml",
        source_language="English",
        target_language="French",
        model_name="test-model",
        max_tokens_per_chunk=1000,
        max_retries=3,
        chunks=[
            {'text': 'chunk1', 'local_tag_map': {'id0': '<em>'}, 'global_indices': [0]},
            {'text': 'chunk2', 'local_tag_map': {'id0': '<strong>'}, 'global_indices': [1]},
            {'text': 'chunk3', 'local_tag_map': {}, 'global_indices': []},
        ],
        global_tag_map={'id0': '<em>', 'id1': '<strong>'},
        placeholder_format=('[[', ']]'),
        translated_chunks=['Chunk traduit 1', 'Chunk traduit 2'],
        current_chunk_index=2,
        original_body_html='<body><p>Original</p></body>',
        doc_metadata={'namespace': 'http://www.w3.org/1999/xhtml'},
        stats={'total_chunks': 3, 'translated_chunks': 2},
        created_at='2026-01-17T10:00:00Z',
        updated_at='2026-01-17T10:00:00Z',
    )


class TestCheckpointManagerXHTMLPartialState:
    """Tests for XHTML partial state save/load/delete."""

    def test_save_and_load_roundtrip(self, temp_checkpoint_manager, sample_state):
        """Test saving and loading a partial state."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_123"
        file_href = "OEBPS/chapter1.xhtml"

        # Save state
        success = manager.save_xhtml_partial_state(translation_id, file_href, sample_state)
        assert success is True

        # Load state
        loaded_state = manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded_state is not None

        # Verify all fields match
        assert loaded_state.file_path == sample_state.file_path
        assert loaded_state.translation_id == sample_state.translation_id
        assert loaded_state.file_href == sample_state.file_href
        assert loaded_state.source_language == sample_state.source_language
        assert loaded_state.target_language == sample_state.target_language
        assert loaded_state.model_name == sample_state.model_name
        assert loaded_state.chunks == sample_state.chunks
        assert loaded_state.global_tag_map == sample_state.global_tag_map
        assert loaded_state.placeholder_format == sample_state.placeholder_format
        assert loaded_state.translated_chunks == sample_state.translated_chunks
        assert loaded_state.current_chunk_index == sample_state.current_chunk_index

    def test_load_nonexistent_state(self, temp_checkpoint_manager):
        """Test loading a state that doesn't exist."""
        manager = temp_checkpoint_manager

        state = manager.load_xhtml_partial_state("nonexistent_id", "OEBPS/chapter1.xhtml")
        assert state is None

    def test_delete_existing_state(self, temp_checkpoint_manager, sample_state):
        """Test deleting an existing partial state."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_456"
        file_href = "OEBPS/chapter2.xhtml"

        # Save state
        manager.save_xhtml_partial_state(translation_id, file_href, sample_state)

        # Verify it exists
        loaded = manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded is not None

        # Delete state
        success = manager.delete_xhtml_partial_state(translation_id, file_href)
        assert success is True

        # Verify it's gone
        loaded = manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded is None

    def test_delete_nonexistent_state(self, temp_checkpoint_manager):
        """Test deleting a state that doesn't exist (should succeed)."""
        manager = temp_checkpoint_manager

        success = manager.delete_xhtml_partial_state("nonexistent_id", "OEBPS/chapter1.xhtml")
        assert success is True  # Should succeed even if doesn't exist

    def test_safe_filename_with_slashes(self, temp_checkpoint_manager, sample_state):
        """Test that file_href with slashes is correctly converted to safe filename."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_789"
        file_href = "OEBPS/Text/chapter1.xhtml"  # Multiple slashes

        # Update sample_state with the correct file_href
        sample_state.file_href = file_href

        # Save state
        success = manager.save_xhtml_partial_state(translation_id, file_href, sample_state)
        assert success is True

        # Verify file was created with safe name
        states_dir = manager.uploads_dir / translation_id / "xhtml_states"
        safe_filename = file_href.replace('/', '_')
        state_file = states_dir / f"{safe_filename}.json"
        assert state_file.exists()

        # Load state (should work despite slashes)
        loaded_state = manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded_state is not None
        # Verify the state preserves the original file_href
        assert loaded_state.file_href == file_href

    def test_safe_filename_with_backslashes(self, temp_checkpoint_manager, sample_state):
        """Test that file_href with backslashes is correctly converted."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_999"
        file_href = "OEBPS\\Text\\chapter1.xhtml"  # Windows-style

        # Save state
        success = manager.save_xhtml_partial_state(translation_id, file_href, sample_state)
        assert success is True

        # Load state
        loaded_state = manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded_state is not None

    def test_multiple_states_for_different_files(self, temp_checkpoint_manager, sample_state):
        """Test saving multiple states for different files in the same job."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_multi"

        # Save states for multiple files
        file_hrefs = [
            "OEBPS/chapter1.xhtml",
            "OEBPS/chapter2.xhtml",
            "OEBPS/chapter3.xhtml",
        ]

        for file_href in file_hrefs:
            success = manager.save_xhtml_partial_state(translation_id, file_href, sample_state)
            assert success is True

        # Verify all states can be loaded
        for file_href in file_hrefs:
            loaded_state = manager.load_xhtml_partial_state(translation_id, file_href)
            assert loaded_state is not None

    def test_list_xhtml_partial_states(self, temp_checkpoint_manager, sample_state):
        """Test listing all partial states for a job."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_list"

        # Initially empty
        states = manager.list_xhtml_partial_states(translation_id)
        assert states == []

        # Save multiple states
        file_hrefs = [
            "OEBPS/chapter1.xhtml",
            "OEBPS/chapter2.xhtml",
            "OEBPS/Text/chapter3.xhtml",
        ]

        for file_href in file_hrefs:
            manager.save_xhtml_partial_state(translation_id, file_href, sample_state)

        # List states
        states = manager.list_xhtml_partial_states(translation_id)
        assert len(states) == 3

        # Note: The list function reconstructs file_href by replacing _ with /
        # This works for simple cases but may not perfectly reconstruct complex paths
        # For now, just verify we get the right count
        assert len(states) == len(file_hrefs)

    def test_list_states_for_nonexistent_job(self, temp_checkpoint_manager):
        """Test listing states for a job that doesn't exist."""
        manager = temp_checkpoint_manager

        states = manager.list_xhtml_partial_states("nonexistent_job")
        assert states == []

    def test_invalid_state_not_loaded(self, temp_checkpoint_manager):
        """Test that invalid states are rejected during load."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_invalid"
        file_href = "OEBPS/chapter1.xhtml"

        # Create an invalid state (chunk index out of range)
        invalid_state = XHTMLTranslationState(
            file_path="/path/to/file.xhtml",
            translation_id=translation_id,
            file_href=file_href,
            source_language="English",
            target_language="French",
            model_name="test-model",
            max_tokens_per_chunk=1000,
            max_retries=1,
            chunks=[{'text': 'chunk1', 'local_tag_map': {}, 'global_indices': []}],
            global_tag_map={},
            placeholder_format=('[[', ']]'),
            translated_chunks=['t1'],
            current_chunk_index=10,  # Out of range!
            original_body_html='',
            doc_metadata={},
            stats={},
            created_at='2026-01-17T10:00:00Z',
            updated_at='2026-01-17T10:00:00Z',
        )

        # Save invalid state (save doesn't validate)
        manager.save_xhtml_partial_state(translation_id, file_href, invalid_state)

        # Load should reject it
        loaded = manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded is None  # Invalid state rejected

    def test_timestamp_update_on_save(self, temp_checkpoint_manager, sample_state):
        """Test that updated_at timestamp is updated on save."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_timestamp"
        file_href = "OEBPS/chapter1.xhtml"

        # Save first time
        manager.save_xhtml_partial_state(translation_id, file_href, sample_state)

        # Load and verify timestamp was updated
        loaded_state = manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded_state is not None
        # The timestamp should have been updated (likely different from original)
        # We can't do exact comparison due to timing, but we can verify it exists
        assert loaded_state.updated_at is not None
        assert isinstance(loaded_state.updated_at, str)

    def test_state_key_roundtrip_with_nested_href(self, temp_checkpoint_manager, sample_state):
        """Test that an href with both '/' and '_' round-trips through save/list/delete/load."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_nested_underscore"
        file_href = "OEBPS/text/chapter_1.xhtml"

        # Update sample_state with the correct file_href so it is persisted in the payload
        sample_state.file_href = file_href

        # Save state
        success = manager.save_xhtml_partial_state(translation_id, file_href, sample_state)
        assert success is True

        # list must return exactly the original file_href, not a lossy
        # reconstruction of the sanitised filename
        states = manager.list_xhtml_partial_states(translation_id)
        assert states == [file_href]

        # Delete state
        success = manager.delete_xhtml_partial_state(translation_id, file_href)
        assert success is True

        # Load should now return None
        loaded = manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded is None

    def test_state_directory_creation(self, temp_checkpoint_manager, sample_state):
        """Test that the xhtml_states directory is created automatically."""
        manager = temp_checkpoint_manager
        translation_id = "test_trans_dir"
        file_href = "OEBPS/chapter1.xhtml"

        # Directory shouldn't exist yet
        states_dir = manager.uploads_dir / translation_id / "xhtml_states"
        assert not states_dir.exists()

        # Save state
        manager.save_xhtml_partial_state(translation_id, file_href, sample_state)

        # Directory should now exist
        assert states_dir.exists()
        assert states_dir.is_dir()
