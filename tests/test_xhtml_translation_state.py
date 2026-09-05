"""
Unit tests for XHTMLTranslationState serialization and validation.
"""

import pytest
from datetime import datetime
from src.core.epub.xhtml_translation_state import (
    CHUNK_PENDING,
    CHUNK_TOKEN_ALIGNED,
    CHUNK_TRANSLATED,
    CHUNK_UNTRANSLATED,
    XHTMLTranslationState,
    token_aligned_chunk_indices,
    unfinished_chunk_indices,
    untranslated_chunk_indices,
)


def create_sample_state() -> XHTMLTranslationState:
    """Create a sample state for testing."""
    return XHTMLTranslationState(
        file_path="/path/to/file.xhtml",
        translation_id="test_trans_123",
        file_href="OEBPS/chapter1.xhtml",
        source_language="English",
        target_language="French",
        model_name="test-model",
        max_tokens_per_chunk=4000,
        max_retries=3,
        chunks=[
            {
                'text': 'Chunk 1 with [[0]] placeholder',
                'local_tag_map': {'[[0]]': '<p>'},
                'global_indices': [0]
            },
            {
                'text': 'Chunk 2 with [[0]] and [[1]]',
                'local_tag_map': {'[[0]]': '<strong>', '[[1]]': '</strong>'},
                'global_indices': [1, 2]
            },
            {
                'text': 'Chunk 3',
                'local_tag_map': {},
                'global_indices': []
            }
        ],
        global_tag_map={
            '[[0]]': '<p>',
            '[[1]]': '<strong>',
            '[[2]]': '</strong>'
        },
        placeholder_format=('[[', ']]'),
        translated_chunks=['Translated chunk 1', 'Translated chunk 2'],
        current_chunk_index=2,
        original_body_html='<body><p>Original content</p></body>',
        doc_metadata={'namespace': 'http://www.w3.org/1999/xhtml'},
        stats={
            'total_chunks': 3,
            'chunks_completed': 2,
            'total_tokens': 1000
        },
        prompt_options={'temperature': 0.7},
        bilingual=False,
        original_chunks=None,
        protect_technical=True,
        created_at='2026-01-17T10:00:00Z',
        updated_at='2026-01-17T10:05:00Z',
    )


def test_serialization_roundtrip():
    """Test that to_dict() → from_dict() preserves all data."""
    # Create original state
    original_state = create_sample_state()

    # Serialize to dict
    data = original_state.to_dict()

    # Deserialize back
    restored_state = XHTMLTranslationState.from_dict(data)

    # Verify all fields are preserved
    assert restored_state.file_path == original_state.file_path
    assert restored_state.translation_id == original_state.translation_id
    assert restored_state.file_href == original_state.file_href
    assert restored_state.source_language == original_state.source_language
    assert restored_state.target_language == original_state.target_language
    assert restored_state.model_name == original_state.model_name
    assert restored_state.max_tokens_per_chunk == original_state.max_tokens_per_chunk
    assert restored_state.max_retries == original_state.max_retries
    assert restored_state.chunks == original_state.chunks
    assert restored_state.global_tag_map == original_state.global_tag_map
    assert restored_state.placeholder_format == original_state.placeholder_format
    assert restored_state.translated_chunks == original_state.translated_chunks
    assert restored_state.current_chunk_index == original_state.current_chunk_index
    assert restored_state.original_body_html == original_state.original_body_html
    assert restored_state.doc_metadata == original_state.doc_metadata
    assert restored_state.stats == original_state.stats
    assert restored_state.prompt_options == original_state.prompt_options
    assert restored_state.bilingual == original_state.bilingual
    assert restored_state.original_chunks == original_state.original_chunks
    assert restored_state.protect_technical == original_state.protect_technical
    assert restored_state.created_at == original_state.created_at
    assert restored_state.updated_at == original_state.updated_at


def test_tuple_preservation():
    """Test that placeholder_format tuple is preserved through serialization."""
    state = create_sample_state()

    # Verify it's a tuple initially
    assert isinstance(state.placeholder_format, tuple)
    assert state.placeholder_format == ('[[', ']]')

    # Serialize and deserialize
    data = state.to_dict()
    restored = XHTMLTranslationState.from_dict(data)

    # Verify it's still a tuple after deserialization
    assert isinstance(restored.placeholder_format, tuple)
    assert restored.placeholder_format == ('[[', ']]')


def test_optional_fields():
    """Test handling of optional fields (None values)."""
    state = XHTMLTranslationState(
        file_path="/path/to/file.xhtml",
        translation_id="test_trans_456",
        file_href="OEBPS/chapter2.xhtml",
        source_language="English",
        target_language="Spanish",
        model_name="test-model",
        max_tokens_per_chunk=4000,
        max_retries=3,
        chunks=[],
        global_tag_map={},
        placeholder_format=('[[', ']]'),
        translated_chunks=[],
        current_chunk_index=0,
        original_body_html='',
        doc_metadata={},
        stats={},
        prompt_options=None,  # Optional
        bilingual=False,
        original_chunks=None,  # Optional
        protect_technical=True,
        created_at='2026-01-17T10:00:00Z',
        updated_at='2026-01-17T10:00:00Z',
    )

    # Serialize and deserialize
    data = state.to_dict()
    restored = XHTMLTranslationState.from_dict(data)

    # Verify optional fields
    assert restored.prompt_options is None
    assert restored.original_chunks is None


def test_validation_valid_state():
    """Test validation of a valid state."""
    state = create_sample_state()
    assert state.validate() is True


def test_validation_invalid_chunk_index():
    """Test validation fails when current_chunk_index is out of bounds."""
    state = create_sample_state()

    # Test index too large
    state.current_chunk_index = 10  # Only 3 chunks
    assert state.validate() is False

    # Test negative index
    state.current_chunk_index = -1
    assert state.validate() is False


def test_validation_mismatched_translated_chunks():
    """Test validation fails when translated_chunks doesn't match current_chunk_index."""
    state = create_sample_state()

    # current_chunk_index is 2, but only 1 translated chunk
    state.translated_chunks = ['Only one chunk']
    assert state.validate() is False


def test_validation_invalid_placeholder_format():
    """Test validation fails with invalid placeholder_format."""
    state = create_sample_state()

    # Not a tuple
    state.placeholder_format = ['[[', ']]']  # type: ignore
    assert state.validate() is False

    # Wrong length
    state.placeholder_format = ('[[',)  # type: ignore
    assert state.validate() is False


def test_validation_missing_required_fields():
    """Test validation fails with empty required fields."""
    state = create_sample_state()

    state.file_path = ''
    assert state.validate() is False

    state = create_sample_state()
    state.translation_id = ''
    assert state.validate() is False

    state = create_sample_state()
    state.file_href = ''
    assert state.validate() is False


def test_validation_invalid_chunks_structure():
    """Test validation fails with malformed chunks."""
    state = create_sample_state()

    # Chunks not a list
    state.chunks = {}  # type: ignore
    assert state.validate() is False

    # Chunk missing required keys
    state.chunks = [{'text': 'test'}]  # Missing local_tag_map and global_indices
    assert state.validate() is False


def test_progress_percentage():
    """Test progress percentage calculation."""
    state = create_sample_state()

    # 2 out of 3 chunks completed
    progress = state.get_progress_percentage()
    assert progress == pytest.approx(66.67, rel=0.01)

    # All chunks completed
    state.current_chunk_index = 3
    progress = state.get_progress_percentage()
    assert progress == 100.0

    # No chunks completed
    state.current_chunk_index = 0
    progress = state.get_progress_percentage()
    assert progress == 0.0

    # Empty chunks
    state.chunks = []
    progress = state.get_progress_percentage()
    assert progress == 0.0


def test_remaining_chunks():
    """Test remaining chunks calculation."""
    state = create_sample_state()

    # 2 completed, 1 remaining
    assert state.get_remaining_chunks() == 1

    # All completed
    state.current_chunk_index = 3
    assert state.get_remaining_chunks() == 0

    # None completed
    state.current_chunk_index = 0
    assert state.get_remaining_chunks() == 3


def test_repr():
    """Test string representation."""
    state = create_sample_state()
    repr_str = repr(state)

    # Should include key information
    assert 'OEBPS/chapter1.xhtml' in repr_str
    assert '66.7%' in repr_str or '66.67%' in repr_str  # Progress
    assert '2/3' in repr_str  # Chunk progress
    assert '2026-01-17T10:05:00Z' in repr_str  # Updated time


def test_bilingual_mode():
    """Test state with bilingual mode enabled."""
    state = create_sample_state()
    state.bilingual = True
    state.original_chunks = [
        {'text': 'Original chunk 1', 'local_tag_map': {}, 'global_indices': []},
        {'text': 'Original chunk 2', 'local_tag_map': {}, 'global_indices': []},
    ]

    # Serialize and deserialize
    data = state.to_dict()
    restored = XHTMLTranslationState.from_dict(data)

    # Verify bilingual fields preserved
    assert restored.bilingual is True
    assert restored.original_chunks == state.original_chunks


def test_default_values():
    """Test that default values are applied correctly."""
    # Create state with minimal required fields
    data = {
        'file_path': '/path/to/file.xhtml',
        'translation_id': 'test_123',
        'file_href': 'OEBPS/ch1.xhtml',
        'source_language': 'English',
        'target_language': 'French',
        'model_name': 'test-model',
        'max_tokens_per_chunk': 4000,
        'max_retries': 3,
        'chunks': [],
        'global_tag_map': {},
        'placeholder_format': ['[[', ']]'],
        'translated_chunks': [],
        'current_chunk_index': 0,
        'original_body_html': '',
        'doc_metadata': {},
        'stats': {},
        'created_at': '2026-01-17T10:00:00Z',
        'updated_at': '2026-01-17T10:00:00Z',
        # Optional fields not provided
    }

    state = XHTMLTranslationState.from_dict(data)

    # Verify defaults
    assert state.prompt_options is None
    assert state.bilingual is False
    assert state.original_chunks is None
    assert state.protect_technical is True  # Default should be True


def test_complex_chunks():
    """Test state with complex chunk structures."""
    state = create_sample_state()

    # Add complex chunk with many placeholders
    complex_chunk = {
        'text': 'Complex [[0]] text [[1]] with [[2]] many [[3]] placeholders [[4]]',
        'local_tag_map': {
            '[[0]]': '<p>',
            '[[1]]': '<strong>',
            '[[2]]': '</strong>',
            '[[3]]': '<em>',
            '[[4]]': '</em>',
        },
        'global_indices': [10, 11, 12, 13, 14]
    }
    state.chunks.append(complex_chunk)

    # Serialize and deserialize
    data = state.to_dict()
    restored = XHTMLTranslationState.from_dict(data)

    # Verify complex chunk preserved
    assert restored.chunks[-1] == complex_chunk
    assert restored.chunks[-1]['local_tag_map'] == complex_chunk['local_tag_map']
    assert restored.chunks[-1]['global_indices'] == complex_chunk['global_indices']


def test_empty_state():
    """Test state with no chunks (edge case)."""
    state = XHTMLTranslationState(
        file_path="/path/to/empty.xhtml",
        translation_id="test_empty",
        file_href="OEBPS/empty.xhtml",
        source_language="English",
        target_language="French",
        model_name="test-model",
        max_tokens_per_chunk=4000,
        max_retries=3,
        chunks=[],
        global_tag_map={},
        placeholder_format=('[[', ']]'),
        translated_chunks=[],
        current_chunk_index=0,
        original_body_html='<body></body>',
        doc_metadata={},
        stats={},
        created_at='2026-01-17T10:00:00Z',
        updated_at='2026-01-17T10:00:00Z',
    )

    # Should validate
    assert state.validate() is True

    # Progress should be 0
    assert state.get_progress_percentage() == 0.0
    assert state.get_remaining_chunks() == 0

    # Serialize and deserialize
    data = state.to_dict()
    restored = XHTMLTranslationState.from_dict(data)
    assert restored.validate() is True


# ---------------------------------------------------------------------------
# Per-chunk statuses (issue #261)
# ---------------------------------------------------------------------------

def test_chunk_statuses_roundtrip():
    """Test that chunk_statuses survives to_dict() → from_dict()."""
    state = create_sample_state()
    state.chunk_statuses = [CHUNK_UNTRANSLATED, CHUNK_TOKEN_ALIGNED, CHUNK_PENDING]

    data = state.to_dict()
    assert data['chunk_statuses'] == state.chunk_statuses

    restored = XHTMLTranslationState.from_dict(data)
    assert restored.chunk_statuses == [CHUNK_UNTRANSLATED, CHUNK_TOKEN_ALIGNED,
                                       CHUNK_PENDING]
    assert restored.validate() is True
    assert unfinished_chunk_indices(restored.chunk_statuses) == [0, 2]


def test_legacy_payload_without_chunk_statuses_is_migrated():
    """A state written before #261 reports 'nothing to retry' (D12).

    The translated prefix becomes 'translated', the rest 'pending', which is
    exactly the behaviour the state had before the field existed.
    """
    data = create_sample_state().to_dict()
    del data['chunk_statuses']  # legacy payload

    state = XHTMLTranslationState.from_dict(data)

    # 2 translated chunks out of 3.
    assert state.chunk_statuses == [CHUNK_TRANSLATED, CHUNK_TRANSLATED, CHUNK_PENDING]
    assert state.validate() is True
    assert unfinished_chunk_indices(state.chunk_statuses) == [2]


def test_null_chunk_statuses_is_migrated_like_a_legacy_payload():
    data = create_sample_state().to_dict()
    data['chunk_statuses'] = None

    state = XHTMLTranslationState.from_dict(data)

    assert state.chunk_statuses == [CHUNK_TRANSLATED, CHUNK_TRANSLATED, CHUNK_PENDING]
    assert state.validate() is True


def test_stale_length_chunk_statuses_is_rebuilt():
    """A statuses list that no longer lines up with chunks is not trusted."""
    data = create_sample_state().to_dict()
    data['chunk_statuses'] = [CHUNK_UNTRANSLATED]  # 1 status, 3 chunks

    state = XHTMLTranslationState.from_dict(data)

    assert state.chunk_statuses == [CHUNK_TRANSLATED, CHUNK_TRANSLATED, CHUNK_PENDING]
    assert state.validate() is True


def test_migration_clamps_an_out_of_range_chunk_index():
    data = create_sample_state().to_dict()
    del data['chunk_statuses']
    data['current_chunk_index'] = 99  # defensive: more than len(chunks)

    state = XHTMLTranslationState.from_dict(data)

    assert state.chunk_statuses == [CHUNK_TRANSLATED] * 3


def test_validation_chunk_statuses_wrong_length():
    """Test validation fails when chunk_statuses does not cover every chunk."""
    state = create_sample_state()
    state.chunk_statuses = [CHUNK_TRANSLATED, CHUNK_TRANSLATED]  # 3 chunks
    assert state.validate() is False


def test_validation_chunk_statuses_unknown_value():
    """Test validation fails on a status outside the four known constants."""
    state = create_sample_state()
    state.chunk_statuses = [CHUNK_TRANSLATED, CHUNK_TRANSLATED, 'in_progress']
    assert state.validate() is False


def test_validation_chunk_statuses_pending_below_current_index():
    """Invariant D4: a chunk below current_chunk_index is never 'pending'."""
    state = create_sample_state()
    # current_chunk_index is 2, so index 1 must already have a terminal status.
    state.chunk_statuses = [CHUNK_TRANSLATED, CHUNK_PENDING, CHUNK_PENDING]
    assert state.validate() is False


def test_validation_accepts_untranslated_below_current_index():
    """A failed chunk keeps its source text in the slot, so it is not pending."""
    state = create_sample_state()
    state.chunk_statuses = [CHUNK_UNTRANSLATED, CHUNK_TRANSLATED, CHUNK_PENDING]
    assert state.validate() is True


def test_validation_ignores_chunk_statuses_when_none():
    """A state that carries no statuses stays valid (nothing to check)."""
    state = create_sample_state()
    assert state.chunk_statuses is None
    assert state.validate() is True


def test_unfinished_chunk_indices_on_none():
    assert unfinished_chunk_indices(None) == []
    assert unfinished_chunk_indices([]) == []


def test_untranslated_chunk_indices_on_none():
    assert untranslated_chunk_indices(None) == []
    assert untranslated_chunk_indices([]) == []


def test_untranslated_chunk_indices_excludes_pending():
    """The narrow projection: only Phase 3 fallbacks, never never-attempted work.

    Both helpers read the same statuses list; they must disagree exactly on the
    pending chunks. Otherwise the live Fallbacks stat card would report every
    chunk an interrupted job has not reached yet as a fallback.
    """
    statuses = [
        CHUNK_UNTRANSLATED,   # 0 - fell back to source text
        CHUNK_TRANSLATED,     # 1 - Phase 1
        CHUNK_PENDING,        # 2 - never attempted
        CHUNK_TOKEN_ALIGNED,  # 3 - Phase 2, translated, never retried (D3)
        CHUNK_UNTRANSLATED,   # 4 - fell back to source text
        CHUNK_PENDING,        # 5 - never attempted
    ]

    assert unfinished_chunk_indices(statuses) == [0, 2, 4, 5]
    assert untranslated_chunk_indices(statuses) == [0, 4]


def test_untranslated_chunk_indices_is_empty_for_a_pending_only_backlog():
    """An interrupted-but-healthy file owes work, but has no damage to report."""
    statuses = [CHUNK_TRANSLATED, CHUNK_TOKEN_ALIGNED, CHUNK_PENDING,
                CHUNK_PENDING]

    assert unfinished_chunk_indices(statuses) == [2, 3]
    assert untranslated_chunk_indices(statuses) == []


def test_token_aligned_chunk_indices_on_none():
    assert token_aligned_chunk_indices(None) == []
    assert token_aligned_chunk_indices([]) == []


def test_token_aligned_chunk_indices_is_disjoint_from_the_work_set():
    """The third projection of the same statuses list.

    Token-aligned chunks are translated, so they are the one status that must
    appear in NEITHER of the other two projections: putting them in the
    unfinished set would make every such job 'partial' (D3/D10), and putting
    them in the untranslated set would claim source-language content the book
    does not contain. They get their own list precisely so an explicit retry can
    address them without any of that.
    """
    statuses = [
        CHUNK_TOKEN_ALIGNED,  # 0 - translated, approximate tag positions
        CHUNK_TRANSLATED,     # 1 - Phase 1
        CHUNK_UNTRANSLATED,   # 2 - fell back to source text
        CHUNK_PENDING,        # 3 - never attempted
        CHUNK_TOKEN_ALIGNED,  # 4 - translated, approximate tag positions
    ]

    assert token_aligned_chunk_indices(statuses) == [0, 4]
    assert unfinished_chunk_indices(statuses) == [2, 3]
    assert untranslated_chunk_indices(statuses) == [2]


def test_token_aligned_chunk_indices_is_empty_for_a_healthy_file():
    statuses = [CHUNK_TRANSLATED, CHUNK_TRANSLATED, CHUNK_PENDING]

    assert token_aligned_chunk_indices(statuses) == []
