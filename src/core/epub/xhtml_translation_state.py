"""
XHTML Translation State Management

This module provides serializable state management for XHTML translation,
enabling interruption and resume at the chunk level.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


# === Per-chunk translation statuses (issue #261) ===
# A chunk of an XHTML file is in exactly one of these four states. They are
# persisted inside the partial state so a resume can tell "already translated"
# from "still to do" at chunk granularity instead of at file granularity.
CHUNK_PENDING = 'pending'  # Never attempted
CHUNK_TRANSLATED = 'translated'  # Phase 1 success, or text-free pass-through
CHUNK_TOKEN_ALIGNED = 'token_aligned'  # Phase 2 (token alignment) succeeded
CHUNK_UNTRANSLATED = 'untranslated'  # Phase 3 fallback: source text kept

ALL_CHUNK_STATUSES = frozenset({
    CHUNK_PENDING, CHUNK_TRANSLATED, CHUNK_TOKEN_ALIGNED, CHUNK_UNTRANSLATED,
})

# Only these two statuses mean "there is still work to do on this chunk".
#
# CHUNK_TOKEN_ALIGNED is deliberately NOT unfinished: those chunks ARE
# translated, only their placeholder positions were approximated by
# proportional reinsertion. Retrying them would re-translate healthy content
# and could make the result worse. This is the same standing rule as the
# docstring of src/api/completion_status.py ("token_alignment_used ... must
# never be added to the rules"), applied at chunk level.
UNFINISHED_CHUNK_STATUSES = frozenset({CHUNK_PENDING, CHUNK_UNTRANSLATED})


def unfinished_chunk_indices(statuses: Optional[List[str]]) -> List[int]:
    """Return the ascending indices of chunks that still need translating.

    Args:
        statuses: Per-chunk statuses, or None when a state carries none.

    Returns:
        Ascending list of indices whose status is in UNFINISHED_CHUNK_STATUSES.
        An empty list when statuses is None (nothing known, nothing to retry).
    """
    if not statuses:
        return []
    return [i for i, status in enumerate(statuses)
            if status in UNFINISHED_CHUNK_STATUSES]


def untranslated_chunk_indices(statuses: Optional[List[str]]) -> List[int]:
    """Return the ascending indices of chunks that fell back to source text.

    Narrower than :func:`unfinished_chunk_indices` on purpose: CHUNK_PENDING is
    excluded. A pending chunk was never attempted, so it is work still owed, not
    damage done. The live Fallbacks stat card reads this counter, and merging
    the two would make a merely interrupted job display every chunk it has not
    reached yet as a fallback.

    Args:
        statuses: Per-chunk statuses, or None when a state carries none.

    Returns:
        Ascending list of indices whose status is exactly CHUNK_UNTRANSLATED.
        An empty list when statuses is None (nothing known, nothing to report).
    """
    if not statuses:
        return []
    return [i for i, status in enumerate(statuses)
            if status == CHUNK_UNTRANSLATED]


def token_aligned_chunk_indices(statuses: Optional[List[str]]) -> List[int]:
    """Return the ascending indices of chunks repaired by token alignment.

    These chunks ARE translated: Phase 2 retranslated their text without
    placeholders and reinserted the inline tags proportionally, so only the tag
    positions are approximate. They are therefore never part of the automatic
    work set (design decision D3) - neither :func:`unfinished_chunk_indices` nor
    the resume path ever lists them, and a job whose only imperfection is an
    approximate tag placement stays 'completed'.

    This projection exists so an explicit, user-initiated retry can widen its
    work set to them, and so the UI can tell "3 chunks are still approximately
    tagged" from the accumulated ``token_alignment_used`` counter, which keeps
    counting every Phase 2 event of every pass even after those chunks have been
    repaired.

    Args:
        statuses: Per-chunk statuses, or None when a state carries none.

    Returns:
        Ascending list of indices whose status is exactly CHUNK_TOKEN_ALIGNED.
        An empty list when statuses is None (nothing known, nothing to report).
    """
    if not statuses:
        return []
    return [i for i, status in enumerate(statuses)
            if status == CHUNK_TOKEN_ALIGNED]


@dataclass
class XHTMLTranslationState:
    """
    Serializable state for partial XHTML translation.

    This class captures the complete translation state at any point during
    XHTML processing, allowing for interruption and exact resume from the
    last translated chunk.
    """

    # Identification
    file_path: str
    translation_id: str
    file_href: str  # Relative path in EPUB (e.g., "OEBPS/chapter1.xhtml")

    # Translation Configuration
    source_language: str
    target_language: str
    model_name: str
    max_tokens_per_chunk: int
    max_retries: int

    # Chunking State
    chunks: List[Dict[str, Any]]  # Complete list of chunks
    # Each chunk contains:
    #   - text: str (with local placeholders)
    #   - local_tag_map: Dict[str, str]
    #   - global_indices: List[int]

    global_tag_map: Dict[str, str]  # Global placeholder → HTML tag mapping
    placeholder_format: Tuple[str, str]  # (prefix, suffix) e.g., ("[[", "]]")

    # Translation Progress
    translated_chunks: List[str]  # Already translated chunks (with global indices)
    current_chunk_index: int  # Next chunk to translate (0-based)

    # Original Document Metadata
    original_body_html: str  # Original body HTML (for reference)
    doc_metadata: Dict[str, Any]  # Namespaces, attributes, etc.

    # Statistics
    stats: Dict[str, Any]  # Serialized TranslationMetrics (file-local)

    # Timestamps
    created_at: str  # ISO 8601 format
    updated_at: str  # ISO 8601 format

    # Global Statistics (for EPUB with multiple XHTML files)
    global_stats: Optional[Dict[str, Any]] = None  # Global stats across all files

    # Options (with defaults - must come after non-default fields)
    prompt_options: Optional[Dict[str, Any]] = None
    bilingual: bool = False
    original_chunks: Optional[List[Dict[str, Any]]] = None  # For bilingual mode

    # Per-chunk statuses (issue #261), one entry per chunk in `chunks`.
    # None means "unknown" (legacy state); from_dict() rebuilds it in that case.
    chunk_statuses: Optional[List[str]] = None

    # Technical Content Protection (always enabled)
    protect_technical: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state to JSON-compatible dictionary.

        Returns:
            Dictionary containing all state information
        """
        return {
            'file_path': self.file_path,
            'translation_id': self.translation_id,
            'file_href': self.file_href,
            'source_language': self.source_language,
            'target_language': self.target_language,
            'model_name': self.model_name,
            'max_tokens_per_chunk': self.max_tokens_per_chunk,
            'max_retries': self.max_retries,
            'chunks': self.chunks,
            'global_tag_map': self.global_tag_map,
            'placeholder_format': list(self.placeholder_format),  # Convert tuple to list for JSON
            'translated_chunks': self.translated_chunks,
            'current_chunk_index': self.current_chunk_index,
            'original_body_html': self.original_body_html,
            'doc_metadata': self.doc_metadata,
            'stats': self.stats,
            'prompt_options': self.prompt_options,
            'bilingual': self.bilingual,
            'original_chunks': self.original_chunks,
            'chunk_statuses': self.chunk_statuses,
            'protect_technical': self.protect_technical,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'global_stats': self.global_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'XHTMLTranslationState':
        """
        Deserialize state from dictionary.

        Args:
            data: Dictionary containing serialized state

        Returns:
            XHTMLTranslationState instance
        """
        return cls(
            file_path=data['file_path'],
            translation_id=data['translation_id'],
            file_href=data['file_href'],
            source_language=data['source_language'],
            target_language=data['target_language'],
            model_name=data['model_name'],
            max_tokens_per_chunk=data['max_tokens_per_chunk'],
            max_retries=data['max_retries'],
            chunks=data['chunks'],
            global_tag_map=data['global_tag_map'],
            placeholder_format=tuple(data['placeholder_format']),  # Convert list back to tuple
            translated_chunks=data['translated_chunks'],
            current_chunk_index=data['current_chunk_index'],
            original_body_html=data['original_body_html'],
            doc_metadata=data['doc_metadata'],
            stats=data['stats'],
            prompt_options=data.get('prompt_options'),
            bilingual=data.get('bilingual', False),
            original_chunks=data.get('original_chunks'),
            chunk_statuses=cls._migrate_chunk_statuses(
                data.get('chunk_statuses'),
                data['chunks'],
                data['current_chunk_index'],
            ),
            protect_technical=data.get('protect_technical', True),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            global_stats=data.get('global_stats'),
        )

    @staticmethod
    def _migrate_chunk_statuses(
        statuses: Optional[List[str]],
        chunks: List[Dict[str, Any]],
        current_chunk_index: int,
    ) -> List[str]:
        """Rebuild chunk_statuses for a payload that has none or a stale one.

        There is no schema migration on disk (design decision D12): a state
        written before issue #261 simply has no `chunk_statuses` key, and it is
        reconstructed here, in memory, every time it is loaded. The rebuilt list
        marks the contiguous translated prefix as CHUNK_TRANSLATED and the rest
        as CHUNK_PENDING, so a legacy state reports "nothing to retry" — exactly
        the behaviour it had before this field existed.

        The same rebuild covers a length mismatch (a state whose chunk list was
        re-chunked): trusting a list that no longer lines up with `chunks` would
        retry arbitrary indices.

        Args:
            statuses: Statuses read from the payload, possibly None.
            chunks: The payload's chunk list.
            current_chunk_index: The payload's next-chunk-to-translate index.

        Returns:
            A statuses list of length len(chunks).
        """
        total = len(chunks)
        if statuses is not None and len(statuses) == total:
            return list(statuses)

        translated = max(0, min(int(current_chunk_index), total))
        return [CHUNK_TRANSLATED] * translated + [CHUNK_PENDING] * (total - translated)

    def validate(self) -> bool:
        """
        Validate the consistency of the state.

        Returns:
            True if state is valid, False otherwise
        """
        # Check that current_chunk_index is within bounds
        if self.current_chunk_index < 0 or self.current_chunk_index > len(self.chunks):
            return False

        # Check that translated_chunks matches current_chunk_index
        if len(self.translated_chunks) != self.current_chunk_index:
            return False

        # Check that placeholder_format is valid
        if not isinstance(self.placeholder_format, tuple) or len(self.placeholder_format) != 2:
            return False

        # Check required fields are not empty
        if not self.file_path or not self.translation_id or not self.file_href:
            return False

        # Check chunks structure
        if not isinstance(self.chunks, list):
            return False

        for chunk in self.chunks:
            if not isinstance(chunk, dict):
                return False
            if 'text' not in chunk or 'local_tag_map' not in chunk or 'global_indices' not in chunk:
                return False

        # Check chunk_statuses consistency (issue #261)
        if self.chunk_statuses is not None:
            if len(self.chunk_statuses) != len(self.chunks):
                return False
            for index, status in enumerate(self.chunk_statuses):
                if status not in ALL_CHUNK_STATUSES:
                    return False
                # Invariant D4: any status other than 'pending' means the chunk
                # has text in translated_chunks, so 'pending' can never appear
                # below current_chunk_index.
                if status == CHUNK_PENDING and index < self.current_chunk_index:
                    return False

        return True

    def get_progress_percentage(self) -> float:
        """
        Calculate translation progress as percentage.

        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if not self.chunks:
            return 0.0
        return (self.current_chunk_index / len(self.chunks)) * 100.0

    def get_remaining_chunks(self) -> int:
        """
        Get number of remaining chunks to translate.

        Returns:
            Number of chunks remaining
        """
        return len(self.chunks) - self.current_chunk_index

    def __repr__(self) -> str:
        """String representation for debugging."""
        progress = self.get_progress_percentage()
        return (
            f"XHTMLTranslationState("
            f"file_href='{self.file_href}', "
            f"progress={progress:.1f}%, "
            f"chunks={self.current_chunk_index}/{len(self.chunks)}, "
            f"updated_at='{self.updated_at}'"
            f")"
        )
