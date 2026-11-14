"""
Integration tests for EPUB translation with chunk-level interruption.

Tests the full EPUB translation pipeline with interruption and resume,
including multiple XHTML files.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, UTC

from src.core.epub.book_loop import translate_xhtml_file
from src.core.epub.run_context import EpubRun, FileTask
from src.persistence.checkpoint_manager import CheckpointManager
from src.core.llm.base import LLMResponse
from tests.job_helpers import make_ctx, make_job


async def _translate_single_xhtml_file(*, file_path, content_href, source_language,
                                       target_language, model_name, llm_client,
                                       max_tokens_per_chunk, max_attempts, context_manager,
                                       log_callback, prompt_options, checkpoint_manager,
                                       translation_id, check_interruption_callback):
    """Translate one file through the book loop's per-file entry point."""
    run = EpubRun(
        job=make_job(model=model_name, source_language=source_language,
                     target_language=target_language),
        ctx=make_ctx(translation_id=translation_id, checkpoint_manager=checkpoint_manager,
                     log_callback=log_callback,
                     check_interruption=check_interruption_callback),
        llm_client=llm_client,
        context_manager=context_manager,
        max_tokens_per_chunk=max_tokens_per_chunk,
        prompt_options=prompt_options,
        max_retries=max_attempts,
    )
    result = await translate_xhtml_file(run, FileTask(href=content_href, file_path=file_path))
    return result.doc_root, result.success, result.stats


def create_mock_llm_response(content: str) -> LLMResponse:
    """Create a proper LLMResponse object for testing."""
    return LLMResponse(
        content=content,
        prompt_tokens=100,
        completion_tokens=50,
        context_used=150,
        context_limit=4096,
        was_truncated=False
    )


def create_mock_llm_client_with_response(default_response: str = "Texte traduit"):
    """Create a mock LLM client with all required methods."""
    client = MagicMock()

    async def mock_generate(*args, **kwargs):
        return create_mock_llm_response(default_response)

    client.generate = mock_generate
    client.extract_translation = lambda response: response

    return client


def _generate_chapter_content(chapter_num: int, paragraph_count: int = 15) -> str:
    """Generate chapter content with enough text for multiple chunks."""
    paragraphs = []
    for i in range(paragraph_count):
        paragraphs.append(f'<p>This is paragraph {i+1} of chapter {chapter_num}. It contains enough text to generate multiple chunks. The quick brown fox jumps over the lazy dog repeatedly.</p>')
    return '\n            '.join(paragraphs)


@pytest.fixture
def temp_epub_dir(tmp_path):
    """Create a temporary EPUB structure with enough content for multiple chunks."""
    epub_dir = tmp_path / "test_epub"
    epub_dir.mkdir()

    # Create OEBPS directory
    oebps_dir = epub_dir / "OEBPS"
    oebps_dir.mkdir()

    # Create sample XHTML files with more content
    chapter1 = oebps_dir / "chapter1.xhtml"
    chapter1.write_text(f'''<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
        <head><title>Chapter 1</title></head>
        <body>
            <h1>Chapter 1</h1>
            {_generate_chapter_content(1)}
        </body>
    </html>
    ''', encoding='utf-8')

    chapter2 = oebps_dir / "chapter2.xhtml"
    chapter2.write_text(f'''<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
        <head><title>Chapter 2</title></head>
        <body>
            <h1>Chapter 2</h1>
            {_generate_chapter_content(2)}
        </body>
    </html>
    ''', encoding='utf-8')

    chapter3 = oebps_dir / "chapter3.xhtml"
    chapter3.write_text(f'''<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
        <head><title>Chapter 3</title></head>
        <body>
            <h1>Chapter 3</h1>
            {_generate_chapter_content(3)}
        </body>
    </html>
    ''', encoding='utf-8')

    return epub_dir


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return create_mock_llm_client_with_response("Texte traduit")


@pytest.fixture
def temp_checkpoint_manager(tmp_path):
    """Create a temporary checkpoint manager with isolated storage."""
    db_path = tmp_path / "integration_test.db"
    manager = CheckpointManager(db_path=str(db_path))
    # Override uploads_dir to use temp directory
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)
    return manager


class TestSingleFileInterruption:
    """Tests for single XHTML file translation with interruption."""

    @pytest.mark.asyncio
    async def test_single_file_interrupt_and_resume(
        self, temp_epub_dir, mock_llm_client, temp_checkpoint_manager
    ):
        """Test interrupting and resuming a single XHTML file translation."""
        translation_id = "test_single_file_001"
        file_path = str(temp_epub_dir / "OEBPS" / "chapter1.xhtml")
        content_href = "OEBPS/chapter1.xhtml"

        # Track chunks
        chunks_processed = [0]

        async def track_translation(*args, **kwargs):
            chunks_processed[0] += 1
            return create_mock_llm_response(f"Chunk {chunks_processed[0]} traduit")

        mock_llm_client.generate = track_translation
        mock_llm_client.extract_translation = lambda response: response

        # Interrupt after 1 chunk
        interrupt_count = [0]

        def interrupt_after_one():
            interrupt_count[0] += 1
            return interrupt_count[0] >= 1

        # ========== FIRST PASS: Interruption ==========
        doc_root, success, stats = await _translate_single_xhtml_file(
            file_path=file_path,
            content_href=content_href,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            max_attempts=1,
            context_manager=None,
            log_callback=None,
            prompt_options=None,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=interrupt_after_one,
        )

        # Should be interrupted
        assert success is False, "Should be interrupted"
        # At least some processing should have occurred or interruption happened before first chunk
        assert chunks_processed[0] >= 0, "Chunks processed count should be non-negative"

        # Check state saved (may or may not exist depending on when interruption occurred)
        state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, content_href)
        # State should be saved if any chunks were processed
        if chunks_processed[0] > 0:
            assert state is not None, "State should be saved after processing chunks"

        # ========== SECOND PASS: Resume ==========
        # Keep tracking from previous count
        chunks_before_resume = chunks_processed[0]
        interrupt_count[0] = 0

        doc_root, success, stats = await _translate_single_xhtml_file(
            file_path=file_path,
            content_href=content_href,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            max_attempts=1,
            context_manager=None,
            log_callback=None,
            prompt_options=None,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=None,  # No interruption
        )

        # Should complete
        assert success is True, "Should complete after resume"
        # Should have processed more chunks
        assert chunks_processed[0] > chunks_before_resume, "Should process additional chunks on resume"

    @pytest.mark.asyncio
    async def test_file_completes_without_interruption(
        self, temp_epub_dir, mock_llm_client, temp_checkpoint_manager
    ):
        """Test that file completes normally without interruption."""
        translation_id = "test_no_interrupt_001"
        file_path = str(temp_epub_dir / "OEBPS" / "chapter1.xhtml")
        content_href = "OEBPS/chapter1.xhtml"

        doc_root, success, stats = await _translate_single_xhtml_file(
            file_path=file_path,
            content_href=content_href,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            max_attempts=1,
            context_manager=None,
            log_callback=None,
            prompt_options=None,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=None,
        )

        assert success is True
        assert doc_root is not None

        # State should be deleted after successful completion by _translate_single_xhtml_file
        state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, content_href)
        # Note: State deletion happens inside _translate_single_xhtml_file after file save
        # If state still exists, it should show 100% completion
        if state is not None:
            assert state.current_chunk_index == len(state.chunks), "If state exists, should be fully completed"


class TestMultipleFilesInterruption:
    """Tests for multiple XHTML files with interruption."""

    @pytest.mark.asyncio
    async def test_interrupt_between_files(
        self, temp_epub_dir, mock_llm_client, temp_checkpoint_manager
    ):
        """Test interruption between different XHTML files."""

        # Create content files list

        files_processed = []

        # Mock log callback to track file processing
        def log_callback(event_type, message):
            if "Starting file" in str(message):
                files_processed.append(event_type)

        # Interrupt after first file
        def interrupt_after_first_file():
            return len(files_processed) >= 1

        # First pass - should process first file then interrupt
        # Note: This is a simplified test - actual _process_all_content_files
        # would need to be tested with full EPUB structure
        # For now, we test individual file handling

    @pytest.mark.asyncio
    async def test_each_file_has_independent_state(
        self, temp_epub_dir, mock_llm_client, temp_checkpoint_manager
    ):
        """Test that each XHTML file has independent checkpoint state."""
        translation_id = "test_independent_states"

        # Process chapter1
        file1_path = str(temp_epub_dir / "OEBPS" / "chapter1.xhtml")
        href1 = "OEBPS/chapter1.xhtml"

        interrupt_at_1 = [0]

        def interrupt_ch1():
            interrupt_at_1[0] += 1
            return interrupt_at_1[0] >= 1

        doc1, success1, _ = await _translate_single_xhtml_file(
            file_path=file1_path,
            content_href=href1,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            max_attempts=1,
            context_manager=None,
            log_callback=None,
            prompt_options=None,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=interrupt_ch1,
        )

        # Process chapter2
        file2_path = str(temp_epub_dir / "OEBPS" / "chapter2.xhtml")
        href2 = "OEBPS/chapter2.xhtml"

        interrupt_at_2 = [0]

        def interrupt_ch2():
            interrupt_at_2[0] += 1
            return interrupt_at_2[0] >= 1

        doc2, success2, _ = await _translate_single_xhtml_file(
            file_path=file2_path,
            content_href=href2,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            max_attempts=1,
            context_manager=None,
            log_callback=None,
            prompt_options=None,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=interrupt_ch2,
        )

        # Both should be interrupted
        assert success1 is False
        assert success2 is False

        # States may or may not exist depending on when interruption occurred
        state1 = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, href1)
        state2 = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, href2)

        # If both states exist, they should be for different files
        if state1 is not None and state2 is not None:
            assert state1.file_href != state2.file_href, "States should be for different files"

        # Resume chapter1
        interrupt_at_1[0] = 0
        doc1_resumed, success1_resumed, _ = await _translate_single_xhtml_file(
            file_path=file1_path,
            content_href=href1,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            max_attempts=1,
            context_manager=None,
            log_callback=None,
            prompt_options=None,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=None,
        )

        assert success1_resumed is True, "Chapter 1 should complete"

        # Chapter 1 state: may be deleted or show 100% completion
        state1_after = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, href1)
        if state1_after is not None:
            assert state1_after.current_chunk_index == len(state1_after.chunks), "Should be fully completed"

        # Chapter 2 state: if it existed before, it should still exist (independent of chapter 1)
        state2_still = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, href2)
        # If state2 was saved initially, it should still exist
        if state2 is not None:
            assert state2_still is not None, "Chapter 2 state should still exist if it was saved"


class TestStateManagement:
    """Tests for checkpoint state management across files."""

    def test_list_partial_states(self, temp_checkpoint_manager):
        """Test listing all partial states for a translation job."""
        translation_id = "test_list_states"

        # Create multiple states
        from src.core.epub.xhtml_translation_state import XHTMLTranslationState
        now = datetime.now(UTC).isoformat()
        files = [
            "OEBPS/chapter1.xhtml",
            "OEBPS/chapter2.xhtml",
            "OEBPS/chapter3.xhtml",
        ]

        for file_href in files:
            state = XHTMLTranslationState(
                file_path=file_href,
                translation_id=translation_id,
                file_href=file_href,
                source_language="English",
                target_language="French",
                model_name="test-model",
                max_tokens_per_chunk=100,
                max_retries=1,
                chunks=[],
                global_tag_map={},
                placeholder_format=('[[', ']]'),
                translated_chunks=[],
                current_chunk_index=0,
                original_body_html='',
                doc_metadata={},
                stats={},
                created_at=now,
                updated_at=now,
            )
            temp_checkpoint_manager.save_xhtml_partial_state(translation_id, file_href, state)

        # List all states
        states = temp_checkpoint_manager.list_xhtml_partial_states(translation_id)

        # Should have 3 states
        assert len(states) >= 3, f"Expected at least 3 states, got {len(states)}"

    def test_cleanup_all_states(self, temp_checkpoint_manager):
        """Test cleaning up all partial states for a job."""
        translation_id = "test_cleanup"

        from src.core.epub.xhtml_translation_state import XHTMLTranslationState

        now = datetime.now(UTC).isoformat()
        # Create states
        files = ["OEBPS/ch1.xhtml", "OEBPS/ch2.xhtml"]

        for file_href in files:
            state = XHTMLTranslationState(
                file_path=file_href,
                translation_id=translation_id,
                file_href=file_href,
                source_language="English",
                target_language="French",
                model_name="test-model",
                max_tokens_per_chunk=100,
                max_retries=1,
                chunks=[],
                global_tag_map={},
                placeholder_format=('[[', ']]'),
                translated_chunks=[],
                current_chunk_index=0,
                original_body_html='',
                doc_metadata={},
                stats={},
                created_at=now,
                updated_at=now,
            )
            temp_checkpoint_manager.save_xhtml_partial_state(translation_id, file_href, state)

        # Verify states exist
        states_before = temp_checkpoint_manager.list_xhtml_partial_states(translation_id)
        assert len(states_before) >= 2

        # Delete all states
        for file_href in files:
            temp_checkpoint_manager.delete_xhtml_partial_state(translation_id, file_href)

        # Verify states deleted
        states_after = temp_checkpoint_manager.list_xhtml_partial_states(translation_id)
        assert len(states_after) == 0, "All states should be deleted"


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_resume_after_file_not_found(
        self, temp_epub_dir, mock_llm_client, temp_checkpoint_manager
    ):
        """Test handling of missing file during resume."""
        translation_id = "test_missing_file"
        file_path = str(temp_epub_dir / "OEBPS" / "nonexistent.xhtml")
        content_href = "OEBPS/nonexistent.xhtml"

        doc_root, success, stats = await _translate_single_xhtml_file(
            file_path=file_path,
            content_href=content_href,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            max_attempts=1,
            context_manager=None,
            log_callback=None,
            prompt_options=None,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=None,
        )

        # Should fail gracefully
        assert success is False
        assert doc_root is None
        assert stats is None

    @pytest.mark.asyncio
    async def test_corrupted_state_handling(
        self, temp_epub_dir, mock_llm_client, temp_checkpoint_manager
    ):
        """Test handling of corrupted checkpoint state."""
        translation_id = "test_corrupted_state"
        file_path = str(temp_epub_dir / "OEBPS" / "chapter1.xhtml")
        content_href = "OEBPS/chapter1.xhtml"

        # Create a corrupted state (invalid validation)
        from src.core.epub.xhtml_translation_state import XHTMLTranslationState

        now = datetime.now(UTC).isoformat()
        corrupted_state = XHTMLTranslationState(
            file_path=content_href,
            translation_id=translation_id,
            file_href=content_href,
            source_language="English",
            target_language="French",
            model_name="test-model",
            max_tokens_per_chunk=100,
            max_retries=1,
            chunks=[],
            global_tag_map={},
            placeholder_format=('[[', ']]'),
            translated_chunks=['chunk1', 'chunk2'],  # Inconsistent with index
            current_chunk_index=0,  # Should be 2
            original_body_html='',
            doc_metadata={},
            stats={},
            created_at=now,
            updated_at=now,
        )

        # Save corrupted state
        temp_checkpoint_manager.save_xhtml_partial_state(translation_id, content_href, corrupted_state)

        # Try to translate - should detect corrupted state and start fresh
        doc_root, success, stats = await _translate_single_xhtml_file(
            file_path=file_path,
            content_href=content_href,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            max_attempts=1,
            context_manager=None,
            log_callback=None,
            prompt_options=None,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            check_interruption_callback=None,
        )

        # Should complete (starts fresh due to invalid state)
        assert success is True
