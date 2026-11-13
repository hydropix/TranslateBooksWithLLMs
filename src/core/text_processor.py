"""
Text processing module for chunking and context management
"""
from typing import List, Dict, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from src.config import TranslationConfig


# Legacy line-based chunking functions removed
# All text chunking now uses token-based approach via split_text_into_chunks()


def split_text_into_chunks(
    text: str,
    config: Optional['TranslationConfig'] = None,
    max_tokens_per_chunk: Optional[int] = None,
    soft_limit_ratio: Optional[float] = None
) -> List[Dict[str, str]]:
    """
    Split text into chunks with context preservation using token-based chunking.

    Args:
        text: Input text to split
        config: TranslationConfig object (optional, for default values)
        max_tokens_per_chunk: Override for max tokens per chunk
        soft_limit_ratio: Override for soft limit ratio

    Returns:
        List of chunk dictionaries with context_before, main_content, context_after
    """
    from src.config import MAX_TOKENS_PER_CHUNK, SOFT_LIMIT_RATIO

    # Determine settings from config or defaults
    if config is not None:
        _max_tokens = max_tokens_per_chunk if max_tokens_per_chunk is not None else config.max_tokens_per_chunk
        _soft_limit = soft_limit_ratio if soft_limit_ratio is not None else config.soft_limit_ratio
    else:
        _max_tokens = max_tokens_per_chunk if max_tokens_per_chunk is not None else MAX_TOKENS_PER_CHUNK
        _soft_limit = soft_limit_ratio if soft_limit_ratio is not None else SOFT_LIMIT_RATIO

    # Token-based chunking
    from src.core.chunking.token_chunker import TokenChunker
    chunker = TokenChunker(
        max_tokens=_max_tokens,
        soft_limit_ratio=_soft_limit
    )
    return chunker.chunk_text(text)