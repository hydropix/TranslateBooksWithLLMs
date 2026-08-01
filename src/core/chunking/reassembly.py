"""
Reassembly helpers shared by the TXT translate and refine paths.

The chunker records, on every chunk, the separator that re-attaches it to the
previous chunk (`join_with`). Joining on that separator restores the source
paragraph structure without inventing a paragraph break in the middle of a
paragraph that was split at sentence level.
"""

from typing import Any, Mapping, Optional, Sequence

DEFAULT_JOINER = "\n\n"


def join_translated_chunks(
    parts: Sequence[Optional[str]],
    chunks: Sequence[Mapping[str, Any]],
    fallbacks: Optional[Sequence[str]] = None,
) -> str:
    """Join per-chunk output back into a single document.

    Args:
        parts: Per-chunk output text. A falsy entry falls back to `fallbacks`
            at the same index, then to an empty string.
        chunks: The chunk dictionaries the parts came from. Each one may carry
            a `join_with` key holding the separator that re-attaches it to the
            previous chunk.
        fallbacks: Optional per-chunk replacement text used when `parts[i]` is
            falsy (typically the untranslated source content).

    Returns:
        The joined text. A missing, empty or out-of-range `join_with` falls back
        to DEFAULT_JOINER, so chunk dictionaries produced by older checkpoints
        or by any other producer keep working.
    """
    pieces = []

    for i, part in enumerate(parts):
        fallback = fallbacks[i] if fallbacks is not None and i < len(fallbacks) else ""
        text = (part or fallback or "").strip()

        if i > 0:
            joiner = DEFAULT_JOINER
            if i < len(chunks):
                joiner = chunks[i].get("join_with") or DEFAULT_JOINER
            pieces.append(joiner)

        pieces.append(text)

    return "".join(pieces)
