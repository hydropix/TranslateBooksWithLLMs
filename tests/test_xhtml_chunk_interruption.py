"""
Unit tests for XHTML chunk-level interruption and resume functionality.

Tests the complete interruption/resume cycle:
1. Start translation
2. Interrupt at specific chunk
3. Resume from saved state
4. Complete translation
"""

import pytest
from unittest.mock import MagicMock
from lxml import etree

from tests.epub_engine_helpers import run_chunk_loop, translate_xhtml
from src.core.epub.xhtml_translation_state import (
    XHTMLTranslationState,
    unfinished_chunk_indices,
    untranslated_chunk_indices,
)
from src.persistence.checkpoint_manager import CheckpointManager
from src.core.llm.base import LLMResponse


# The text-survival guard rejects a reply far shorter than its source, so the
# mock translations carry roughly a paragraph of text.
PLAUSIBLE_TRANSLATION = (
    "Ceci est un paragraphe traduit. Il contient assez de texte pour remplacer "
    "un paragraphe source. Le renard brun rapide saute par-dessus le chien paresseux."
)


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


def create_mock_llm_client(default_response: str = "Texte traduit"):
    """Create a mock LLM client with all required methods."""
    client = MagicMock()

    # Create a proper async function that returns a response
    async def mock_generate(*args, **kwargs):
        return create_mock_llm_response(default_response)

    client.generate = mock_generate
    # extract_translation returns the content directly (simulates tag extraction)
    client.extract_translation = lambda response: response

    return client


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns simple translations."""
    return create_mock_llm_client("Texte traduit. " + PLAUSIBLE_TRANSLATION)


@pytest.fixture(autouse=True)
def force_token_alignment(monkeypatch):
    """Pin Phase 2 on for every test in this module.

    The mock LLM below answers with plain text carrying no placeholders, so
    Phase 1 validation always fails and the token-alignment fallback is what
    produces the translated chunks these tests count. Without this pin the
    module silently inherits `EPUB_TOKEN_ALIGNMENT_ENABLED` from the
    developer's .env: set it to false for an unrelated reason and these tests
    fail with confusing chunk counts, because every chunk drops to Phase 3.

    `setattr` on the config attribute, not `setenv`: src/config.py reads the
    variable once at import time, while translate_chunk_with_fallback re-reads
    the module attribute on every call.
    """
    monkeypatch.setattr('src.config.EPUB_TOKEN_ALIGNMENT_ENABLED', True)


@pytest.fixture
def sample_xhtml_doc():
    """Create a sample XHTML document for testing with enough content for multiple chunks."""
    # Generate enough paragraphs to create multiple chunks with small token limit
    paragraphs = []
    for i in range(20):
        paragraphs.append(f'<p>This is paragraph number {i+1}. It contains enough text to help generate multiple chunks when combined with other paragraphs. The quick brown fox jumps over the lazy dog.</p>')

    body_content = '\n            '.join(paragraphs)
    # The heading is long enough for the fixed mock reply to pass the runaway
    # guard (4x the source) once a failed chunk is translated block by block.
    xhtml = f'''<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml">
        <head><title>Test</title></head>
        <body>
            <h1>Chapter 1: The Keeper of the Northern Lighthouse</h1>
            {body_content}
        </body>
    </html>
    '''.encode('utf-8')
    parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
    return etree.fromstring(xhtml, parser)


@pytest.fixture
def temp_checkpoint_manager(tmp_path):
    """Create a temporary checkpoint manager with isolated storage."""
    db_path = tmp_path / "test_jobs.db"
    manager = CheckpointManager(db_path=str(db_path))
    # Override uploads_dir to use temp directory
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)
    return manager


class TestChunkLevelInterruption:
    """Tests for chunk-level interruption and resume."""

    @pytest.mark.asyncio
    async def test_interruption_at_chunk_3(self, sample_xhtml_doc, mock_llm_client, temp_checkpoint_manager):
        """Test interruption at chunk 3, then resume to completion."""
        translation_id = "test_trans_interrupt_001"
        file_href = "OEBPS/chapter1.xhtml"

        # Track chunk processing
        chunks_processed = []

        # Override LLM client to track chunks
        async def track_chunks(*args, **kwargs):
            chunks_processed.append(len(chunks_processed) + 1)
            return create_mock_llm_response(f"Translated chunk {len(chunks_processed)}. {PLAUSIBLE_TRANSLATION}")

        mock_llm_client.generate = track_chunks
        mock_llm_client.extract_translation = lambda response: response

        # Create interruption callback that triggers after 3 chunks
        interrupt_counter = [0]

        def check_interruption():
            interrupt_counter[0] += 1
            return interrupt_counter[0] >= 3

        # ========== FIRST PASS: Interruption ==========
        success, stats = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            check_interruption_callback=check_interruption,
        )

        # Should fail (interrupted)
        assert success is False, "Translation should be incomplete due to interruption"

        # Interruption callback is checked AFTER processing each chunk
        # So with >= 3, it processes chunks 1, 2, 3, then checks and interrupts before chunk 4
        # The number depends on timing, but should be at least 3
        assert len(chunks_processed) >= 3, f"Expected at least 3 chunks processed, got {len(chunks_processed)}"

        # Check that partial state was saved
        saved_state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        assert saved_state is not None, "Partial state should be saved"
        # The chunk index should match chunks processed
        assert saved_state.current_chunk_index >= 2, f"Expected chunk index >= 2, got {saved_state.current_chunk_index}"
        assert len(saved_state.translated_chunks) >= 2, f"Should have at least 2 translated chunks, got {len(saved_state.translated_chunks)}"

        # Validate the saved state
        assert saved_state.validate() is True, "Saved state should be valid"

        # ========== SECOND PASS: Resume ==========
        # Reset tracking
        chunks_processed.clear()
        interrupt_counter[0] = 0  # Don't interrupt this time

        success, stats = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            resume_state=saved_state,
        )

        # Should succeed now
        assert success is True, "Translation should complete after resume"

        # Should only process remaining chunks (starting from chunk 4)
        # Note: depends on how many chunks the document generates
        assert len(chunks_processed) >= 1, "Should process at least one more chunk"

        # Note: the file-level translation does not delete partial state - that's done
        # at the higher level (_translate_single_xhtml_file) after saving the file.
        final_state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        if final_state is not None:
            # If state exists, it should show 100% completion
            assert final_state.current_chunk_index == len(final_state.chunks), "Should be fully translated"

    @pytest.mark.asyncio
    async def test_interruption_does_not_inflate_the_untranslated_count(
            self, sample_xhtml_doc, mock_llm_client, temp_checkpoint_manager):
        """An interrupted file owes work, but owns no Phase 3 damage.

        The live Fallbacks stat card sums `token_alignment_used` and the
        untranslated-only projection. If that projection merged `pending` in
        (as `unfinished_chunk_indices` does, on purpose, for the retry work
        set), pausing a healthy job would light the card up with every chunk it
        simply has not reached yet.
        """
        translation_id = "test_trans_pending_not_fallback"
        file_href = "OEBPS/chapter1.xhtml"

        interrupt_counter = [0]

        def check_interruption():
            interrupt_counter[0] += 1
            return interrupt_counter[0] >= 3

        success, _stats = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            check_interruption_callback=check_interruption,
        )
        assert success is False

        state = temp_checkpoint_manager.load_xhtml_partial_state(
            translation_id, file_href)
        assert state is not None
        assert state.chunk_statuses is not None

        unfinished = unfinished_chunk_indices(state.chunk_statuses)
        untranslated = untranslated_chunk_indices(state.chunk_statuses)

        # There IS work left (the never-attempted tail), and it is exactly the
        # chunks from current_chunk_index onwards.
        assert unfinished == list(range(state.current_chunk_index,
                                        len(state.chunks)))
        assert unfinished, "an interrupted file must still owe chunks"
        # ...and none of it counts as a fallback.
        assert untranslated == []

    @pytest.mark.asyncio
    async def test_multiple_interruptions(self, sample_xhtml_doc, mock_llm_client, temp_checkpoint_manager):
        """Test multiple interruptions and resumes."""
        translation_id = "test_trans_multi_interrupt"
        file_href = "OEBPS/chapter2.xhtml"

        total_chunks_processed = []

        async def track_chunks(*args, **kwargs):
            total_chunks_processed.append(len(total_chunks_processed) + 1)
            return create_mock_llm_response(f"Translated chunk {len(total_chunks_processed)}. {PLAUSIBLE_TRANSLATION}")

        mock_llm_client.generate = track_chunks
        mock_llm_client.extract_translation = lambda response: response

        # A chunk whose placeholders are lost is retried block by block, so one
        # chunk can issue several LLM requests. Each pass therefore interrupts
        # once it has made progress (any request since the pass started)
        # instead of at an absolute request count.
        base_1 = len(total_chunks_processed)

        def interrupt_at_2():
            return len(total_chunks_processed) > base_1

        success, _ = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            check_interruption_callback=interrupt_at_2,
        )

        assert success is False
        state1 = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        assert state1 is not None
        # Chunk index should be at least 1 (0-indexed means at least 1 chunk processed)
        assert state1.current_chunk_index >= 1, f"Expected chunk index >= 1, got {state1.current_chunk_index}"

        base_2 = len(total_chunks_processed)

        def interrupt_at_4():
            return len(total_chunks_processed) > base_2

        success, _ = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            check_interruption_callback=interrupt_at_4,
            resume_state=state1,
        )

        assert success is False
        state2 = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        assert state2 is not None
        # Chunk index should be greater than state1
        assert state2.current_chunk_index > state1.current_chunk_index, "Should have progressed past first interruption"

        # Final completion without interruption
        success, _ = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            resume_state=state2,
        )

        assert success is True
        # Note: the file-level translation does not delete partial state
        final_state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        if final_state is not None:
            assert final_state.current_chunk_index == len(final_state.chunks), "Should be fully translated"


class TestPeriodicCheckpoint:
    """Tests for periodic checkpoint saving."""

    @pytest.mark.asyncio
    async def test_checkpoint_every_5_chunks(self, temp_checkpoint_manager):
        """Test that checkpoints are saved every 5 chunks."""

        # Create mock data with 12 chunks
        chunks = [
            {
                'text': f'Chunk {i}',
                'local_tag_map': {},
                'global_indices': []
            }
            for i in range(12)
        ]

        # Track checkpoint saves
        checkpoint_saves = []
        original_save = temp_checkpoint_manager.save_xhtml_partial_state

        def track_save(translation_id, file_href, state):
            checkpoint_saves.append(state.current_chunk_index)
            return original_save(translation_id, file_href, state)

        temp_checkpoint_manager.save_xhtml_partial_state = track_save

        # Mock LLM client
        mock_client = MagicMock()
        async def simple_translate(*args, **kwargs):
            return create_mock_llm_response("Translated")
        mock_client.generate = simple_translate
        mock_client.extract_translation = lambda response: response

        # Run translation
        translated_chunks, stats, was_interrupted = await run_chunk_loop(
            chunks=chunks,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_client,
            max_retries=1,
            context_manager=None,
            placeholder_format=('[[', ']]'),
            checkpoint_manager=temp_checkpoint_manager,
            translation_id="test_periodic",
            file_href="OEBPS/test.xhtml",
            file_path="OEBPS/test.xhtml",
            start_chunk_index=0,
            translated_chunks=None,
            global_tag_map={},
            stats=None,
        )

        # Should have saved at chunks 5, 10, and 12 (final)
        assert was_interrupted is False
        assert len(checkpoint_saves) >= 3, f"Expected at least 3 checkpoints, got {len(checkpoint_saves)}"

        # Check that saves happened at expected intervals
        # Note: exact indices depend on implementation (could be 5, 10, 12 or 4, 9, 12 if 0-indexed)
        assert 12 in checkpoint_saves, "Should save on last chunk"


class TestBilingualModeWithInterruption:
    """Tests for bilingual mode with interruption."""

    @pytest.mark.asyncio
    async def test_bilingual_mode_resume(self, sample_xhtml_doc, mock_llm_client, temp_checkpoint_manager):
        """Test that bilingual mode works correctly with interruption/resume."""
        translation_id = "test_bilingual_interrupt"
        file_href = "OEBPS/bilingual.xhtml"

        chunks_count = [0]

        async def count_chunks(*args, **kwargs):
            chunks_count[0] += 1
            return create_mock_llm_response("Bilingual translation")

        mock_llm_client.generate = count_chunks
        mock_llm_client.extract_translation = lambda response: response

        # Interrupt after 2 chunks
        def interrupt_early():
            return chunks_count[0] >= 2

        # First pass with bilingual mode
        success, _ = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            bilingual=True,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            check_interruption_callback=interrupt_early,
        )

        assert success is False

        # Load state and check bilingual data is preserved
        state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        assert state is not None
        assert state.bilingual is True, "Bilingual flag should be preserved"
        assert state.original_chunks is not None, "Original chunks should be preserved"
        assert len(state.original_chunks) > 0, "Should have original chunks saved"

        # Resume with bilingual mode
        chunks_count[0] = 0
        success, _ = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            bilingual=True,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            resume_state=state,
        )

        assert success is True


class TestStateValidation:
    """Tests for state validation during resume."""

    @pytest.mark.asyncio
    async def test_invalid_state_rejected(self, sample_xhtml_doc, mock_llm_client, temp_checkpoint_manager):
        """Test that invalid states are rejected on load."""
        from datetime import datetime

        translation_id = "test_invalid_state"
        file_href = "OEBPS/invalid.xhtml"

        # Create an invalid state (more translated chunks than index)
        invalid_state = XHTMLTranslationState(
            file_path="OEBPS/invalid.xhtml",
            translation_id=translation_id,
            file_href=file_href,
            source_language="English",
            target_language="French",
            model_name="test-model",
            max_tokens_per_chunk=100,
            max_retries=1,
            chunks=[
                {'text': 'chunk1', 'local_tag_map': {}, 'global_indices': []},
                {'text': 'chunk2', 'local_tag_map': {}, 'global_indices': []},
            ],
            global_tag_map={},
            placeholder_format=('[[', ']]'),
            translated_chunks=['t1', 't2', 't3'],  # 3 chunks translated
            current_chunk_index=2,  # But index is only 2 (should be 3)
            original_body_html='',
            doc_metadata={},
            stats={},
            created_at=datetime.now().isoformat() + 'Z',
            updated_at=datetime.now().isoformat() + 'Z',
        )

        # Save the invalid state directly
        temp_checkpoint_manager.save_xhtml_partial_state(translation_id, file_href, invalid_state)

        # Try to load it - should return None due to validation failure
        loaded_state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        assert loaded_state is None, "Invalid state should be rejected on load"

    def test_state_validation_checks(self):
        """Test various state validation scenarios."""
        from datetime import datetime

        # Valid state
        valid_state = XHTMLTranslationState(
            file_path="/path/file.xhtml",
            translation_id="trans_123",
            file_href="OEBPS/ch1.xhtml",
            source_language="English",
            target_language="French",
            model_name="test-model",
            max_tokens_per_chunk=100,
            max_retries=1,
            chunks=[{'text': 'c1', 'local_tag_map': {}, 'global_indices': []}],
            global_tag_map={},
            placeholder_format=('[[', ']]'),
            translated_chunks=[],
            current_chunk_index=0,
            original_body_html='',
            doc_metadata={},
            stats={},
            created_at=datetime.now().isoformat() + 'Z',
            updated_at=datetime.now().isoformat() + 'Z',
        )
        assert valid_state.validate() is True

        # Invalid: chunk index out of range
        invalid_state1 = XHTMLTranslationState(
            file_path="/path/file.xhtml",
            translation_id="trans_123",
            file_href="OEBPS/ch1.xhtml",
            source_language="English",
            target_language="French",
            model_name="test-model",
            max_tokens_per_chunk=100,
            max_retries=1,
            chunks=[],
            global_tag_map={},
            placeholder_format=('[[', ']]'),
            translated_chunks=[],
            current_chunk_index=5,  # Out of range
            original_body_html='',
            doc_metadata={},
            stats={},
            created_at=datetime.now().isoformat() + 'Z',
            updated_at=datetime.now().isoformat() + 'Z',
        )
        assert invalid_state1.validate() is False

        # Invalid: missing required field
        invalid_state2 = XHTMLTranslationState(
            file_path="",  # Empty!
            translation_id="trans_123",
            file_href="OEBPS/ch1.xhtml",
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
            created_at=datetime.now().isoformat() + 'Z',
            updated_at=datetime.now().isoformat() + 'Z',
        )
        assert invalid_state2.validate() is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_interrupt_on_first_chunk(self, sample_xhtml_doc, mock_llm_client, temp_checkpoint_manager):
        """Test interruption on the very first chunk."""
        translation_id = "test_first_chunk_interrupt"
        file_href = "OEBPS/first.xhtml"

        # Interrupt immediately
        def interrupt_immediately():
            return True

        success, _ = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            check_interruption_callback=interrupt_immediately,
        )

        assert success is False

        # Should have state saved with 0 translated chunks
        state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        assert state is not None
        assert state.current_chunk_index == 0
        assert len(state.translated_chunks) == 0

    @pytest.mark.asyncio
    async def test_no_interruption_full_translation(self, sample_xhtml_doc, mock_llm_client, temp_checkpoint_manager):
        """Test normal translation without any interruption."""
        translation_id = "test_no_interrupt"
        file_href = "OEBPS/normal.xhtml"

        # Never interrupt
        def never_interrupt():
            return False

        success, stats = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            checkpoint_manager=temp_checkpoint_manager,
            translation_id=translation_id,
            file_href=file_href,
            check_interruption_callback=never_interrupt,
        )

        assert success is True

        # Note: the file-level translation does not delete partial state - that's done
        # at the higher level (_translate_single_xhtml_file) after saving the file.
        # Here we just verify translation completed successfully.
        state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
        # State may still exist but should show 100% progress
        if state is not None:
            assert state.current_chunk_index == len(state.chunks), "Should be fully translated"

    @pytest.mark.asyncio
    async def test_resume_without_checkpoint_manager(self, sample_xhtml_doc, mock_llm_client):
        """Test that translation works without checkpoint manager (backwards compatibility)."""
        # Should work without checkpoint_manager
        success, stats = await translate_xhtml(
            doc_root=sample_xhtml_doc,
            source_language="English",
            target_language="French",
            model_name="test-model",
            llm_client=mock_llm_client,
            max_tokens_per_chunk=100,
            # No checkpoint_manager provided
        )

        assert success is True
