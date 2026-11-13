"""
Placeholder renumbering for HTML chunking.

This module handles the conversion of global placeholder indices to local
indices within chunks, enabling independent translation of each chunk.
"""
from typing import Dict

from src.common.placeholder_format import PlaceholderFormat


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
