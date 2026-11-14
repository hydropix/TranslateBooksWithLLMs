"""
Placeholder renumbering between document (global) and chunk (local) indices.

The single home of every global <-> local placeholder conversion:

- ``PlaceholderRenumberer.create_chunk_with_local_placeholders`` builds a
  chunk: it numbers each placeholder *occurrence* of a slice of the document
  0, 1, 2... and records the global index of each (``global_indices``).
- ``globalize_placeholders`` maps a chunk's local placeholders back to their
  global indices (after translation, before reassembly).
- ``localize_placeholders`` maps global placeholders back to local ones (a
  translated chunk sent to the refine pass).

Placeholder levels:

    Document (TagPreserver): "<body><p>Hello</p></body>" -> "[id0]Hello[id1]"
    Chunk (HtmlChunker):     "[id5]Hello[id6]" -> "[id0]Hello[id1]",
                             global_indices [5, 6]
    Translation:             the LLM returns "[id0]Bonjour[id1]",
                             globalized to "[id5]Bonjour[id6]"
"""
from typing import Dict, List, Optional, Sequence, Tuple

from src.common.placeholder_format import PlaceholderFormat
from src.config import detect_placeholder_format_in_text


def globalize_placeholders(
    text: str,
    global_indices: Sequence[int],
    placeholder_format: Optional[Tuple[str, str]] = None,
) -> str:
    """Rewrite local placeholders (0, 1, 2...) as their global indices.

    Local placeholder ``i`` becomes ``global_indices[i]``. Every local
    placeholder is first swapped for a temporary marker, so a local index
    that equals another position's global index is never renumbered twice.

    Args:
        text: Text holding local placeholders.
        global_indices: Global index of each local placeholder, in order.
        placeholder_format: ``(prefix, suffix)``; detected from ``text``
            when None.
    """
    if not global_indices:
        return text
    prefix, suffix = placeholder_format or detect_placeholder_format_in_text(text)
    result = text
    for local_idx in range(len(global_indices)):
        local_ph = f"{prefix}{local_idx}{suffix}"
        if local_ph in result:
            result = result.replace(local_ph, f"__RESTORE_{local_idx}__")
    for local_idx, global_idx in enumerate(global_indices):
        result = result.replace(f"__RESTORE_{local_idx}__", f"{prefix}{global_idx}{suffix}")
    return result


def localize_placeholders(
    text: str,
    global_indices: Sequence[int],
    placeholder_format: Tuple[str, str],
) -> str:
    """Rewrite global placeholders as local ones, the inverse of
    :func:`globalize_placeholders`: ``global_indices[i]`` becomes ``i``,
    through temporary markers for the same reason."""
    prefix, suffix = placeholder_format
    result = text
    for local_idx, global_idx in enumerate(global_indices):
        result = result.replace(f"{prefix}{global_idx}{suffix}", f"__TEMP_PH_{local_idx}__")
    for local_idx in range(len(global_indices)):
        result = result.replace(f"__TEMP_PH_{local_idx}__", f"{prefix}{local_idx}{suffix}")
    return result


def globalize_chunks(
    chunks: List[Dict],
    placeholder_format: Optional[Tuple[str, str]] = None,
) -> List[str]:
    """Each chunk dict's ``text`` with its placeholders globalized.

    Turns freshly chunked source (or draft) text into the globally numbered
    form the reassembly and the refine pass work on.
    """
    return [
        globalize_placeholders(chunk.get('text', ''), chunk.get('global_indices', []),
                               placeholder_format)
        for chunk in chunks
    ]


class PlaceholderRenumberer:
    """
    Renumbers global placeholders to local indices within chunks.

    When text is split into chunks, each chunk needs its own local placeholder
    numbering (0, 1, 2, ...) for translation. This class handles:
    - Converting global placeholders to local indices
    - Maintaining mapping between local and global placeholders
    - Tracking global indices for reconstruction

    Example:
        Global text: "[id5]Hello[id6]world[id7]"
        After renumbering: "[id0]Hello[id1]world[id2]"
        With mapping: {
            '[id0]': '<p>',
            '[id1]': '<b>',
            '[id2]': '</b></p>'
        }
    """

    def __init__(self):
        """Initialize the renumberer with placeholder format from config."""
        self.placeholder_format = PlaceholderFormat.from_config()

    def create_chunk_with_local_placeholders(
        self,
        text: str,
        global_tag_map: Dict[str, str],
        global_offset: int
    ) -> Dict:
        """
        Create chunk with locally renumbered placeholders (0, 1, 2...).

        This method takes text with global placeholders and converts them to
        local placeholders starting from 0. Each occurrence of a placeholder
        gets a unique local index, even if the same global placeholder appears
        multiple times.

        Args:
            text: Text with global placeholders (e.g., "[id5]Hello[id6]")
            global_tag_map: Mapping of global placeholders to HTML tags
            global_offset: Starting global offset for this chunk

        Returns:
            Dictionary containing:
                - text: Text with local placeholders (e.g., "[id0]Hello[id1]")
                - local_tag_map: Mapping of local placeholders to tags
                - global_offset: The global offset value (preserved)
                - global_indices: List of global indices in order

        Example:
            >>> renumberer = PlaceholderRenumberer()
            >>> global_map = {"[id5]": "<p>", "[id6]": "</p>"}
            >>> result = renumberer.create_chunk_with_local_placeholders(
            ...     "[id5]Hello[id6]",
            ...     global_map,
            ...     0
            ... )
            >>> result['text']
            '[id0]Hello[id1]'
            >>> result['local_tag_map']
            {'[id0]': '<p>', '[id1]': '</p>'}
            >>> result['global_indices']
            [5, 6]
        """
        # Find all global placeholders in this chunk (including duplicates)
        # Each occurrence will get a unique local index
        placeholder_occurrences = []

        for start, end, global_placeholder, global_idx in self.placeholder_format.find_all(text):
            placeholder_occurrences.append((start, end, global_placeholder, global_idx))

        # Step 1: Replace each occurrence with a unique temporary marker
        # This prevents issues with overlapping replacements
        temp_markers = []
        for i, (start, end, global_placeholder, global_idx) in enumerate(placeholder_occurrences):
            temp_marker = f"__TEMP_PH_{i}__"
            temp_markers.append(temp_marker)

        # Apply temp markers in REVERSE order to avoid position shifts
        temp_text = text
        for i in range(len(placeholder_occurrences) - 1, -1, -1):
            start, end, _, _ = placeholder_occurrences[i]
            temp_text = temp_text[:start] + temp_markers[i] + temp_text[end:]

        # Step 2: Replace temp markers with local placeholders (0, 1, 2, ...)
        renumbered_text = temp_text
        local_tag_map = {}
        global_indices = []

        for local_idx, (_, _, global_placeholder, global_idx) in enumerate(placeholder_occurrences):
            local_placeholder = self.placeholder_format.create(local_idx)
            temp_marker = temp_markers[local_idx]

            # Replace temp marker with local placeholder
            renumbered_text = renumbered_text.replace(temp_marker, local_placeholder, 1)

            # Build mapping
            local_tag_map[local_placeholder] = global_tag_map.get(global_placeholder, "")
            global_indices.append(global_idx)

        return {
            'text': renumbered_text,
            'local_tag_map': local_tag_map,
            'global_offset': global_offset,
            'global_indices': global_indices
        }
