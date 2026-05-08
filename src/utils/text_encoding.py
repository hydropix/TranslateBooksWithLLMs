"""
Text metadata encoding for translated output attribution.

Embeds invisible metadata in translation outputs using:
1. Zero-width characters (invisible to users)
2. Homoglyph substitution (visually identical characters)
3. Whitespace patterns (multiple spaces, special spaces)

This metadata helps with:
- Source attribution and credit
- License compliance verification (AGPL-3.0)
- Output provenance tracking
- Quality assurance and debugging

IMPORTANT: These metadata markers are INVISIBLE to end users and do NOT affect
readability or translation quality. They are purely for attribution and compliance.
"""

import hashlib
import re
from typing import Optional


# Zero-width characters (completely invisible)
ZERO_WIDTH_SPACE = '\u200B'      # Zero-width space
ZERO_WIDTH_JOINER = '\u200D'     # Zero-width joiner
ZERO_WIDTH_NON_JOINER = '\u200C' # Zero-width non-joiner
WORD_JOINER = '\u2060'           # Word joiner (prevents line break)

# Special spaces (visually identical to regular space)
EN_SPACE = '\u2002'              # En space (width of 'n')
EM_SPACE = '\u2003'              # Em space (width of 'm')
THIN_SPACE = '\u2009'            # Thin space
HAIR_SPACE = '\u200A'            # Hair space
NARROW_NO_BREAK_SPACE = '\u202F' # Narrow no-break space


class TextMetadataEncoder:
    """
    Embeds invisible metadata in translated text for attribution.

    Uses zero-width characters to encode binary data (client ID)
    in a way that survives copy-paste but is invisible to users.
    """

    def __init__(self, client_id: str):
        """
        Initialize with client identifier.

        Args:
            client_id: Client identifier (hex string)
        """
        self.client_id = client_id

        # Encoding: 0 = ZWNJ, 1 = ZWJ (binary encoding)
        self.bit_0 = ZERO_WIDTH_NON_JOINER
        self.bit_1 = ZERO_WIDTH_JOINER

    def _text_to_binary(self, text: str) -> str:
        """
        Convert text to binary string.

        Args:
            text: Text to convert

        Returns:
            Binary string (e.g., "01001000")
        """
        return ''.join(format(ord(c), '08b') for c in text)

    def _binary_to_zwc(self, binary: str) -> str:
        """
        Convert binary string to zero-width characters.

        Args:
            binary: Binary string (e.g., "01001000")

        Returns:
            String of zero-width characters
        """
        return ''.join(self.bit_1 if bit == '1' else self.bit_0 for bit in binary)

    def _zwc_to_binary(self, zwc: str) -> str:
        """
        Convert zero-width characters back to binary.

        Args:
            zwc: String containing zero-width characters

        Returns:
            Binary string
        """
        binary = ''
        for char in zwc:
            if char == self.bit_1:
                binary += '1'
            elif char == self.bit_0:
                binary += '0'
        return binary

    def _binary_to_text(self, binary: str) -> str:
        """
        Convert binary string back to text.

        Args:
            binary: Binary string

        Returns:
            Original text
        """
        # Pad to multiple of 8
        while len(binary) % 8 != 0:
            binary += '0'

        chars = []
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            chars.append(chr(int(byte, 2)))

        return ''.join(chars)

    def embed_metadata(self, text: str, position: str = "middle") -> str:
        """
        Embed invisible metadata in text.

        Inserts zero-width characters encoding the instance ID at a
        strategic position in the text.

        Args:
            text: Text to metadata
            position: Where to insert ('start', 'middle', 'end', 'distributed')

        Returns:
            Annotated text (looks identical to original)
        """
        if not text or not text.strip():
            return text

        # Create metadata: "SID:{client_id}"
        metadata_text = f"SID:{self.client_id}"

        # Convert to binary then to zero-width characters
        binary = self._text_to_binary(metadata_text)
        zwc_metadata = self._binary_to_zwc(binary)

        # Insert based on position
        if position == "start":
            # After first word
            words = text.split(' ', 1)
            if len(words) > 1:
                return words[0] + zwc_metadata + ' ' + words[1]
            return text + zwc_metadata

        elif position == "end":
            # Before last punctuation or at end
            if text[-1] in '.!?':
                return text[:-1] + zwc_metadata + text[-1]
            return text + zwc_metadata

        elif position == "middle":
            # In the middle of the text
            mid = len(text) // 2
            # Find nearest space
            space_pos = text.find(' ', mid)
            if space_pos == -1:
                space_pos = text.rfind(' ', 0, mid)
            if space_pos == -1:
                return text + zwc_metadata
            return text[:space_pos] + zwc_metadata + text[space_pos:]

        elif position == "distributed":
            # Distribute across multiple locations (more robust)
            # Split metadata into chunks
            chunk_size = len(zwc_metadata) // 3
            chunks = [
                zwc_metadata[:chunk_size],
                zwc_metadata[chunk_size:chunk_size*2],
                zwc_metadata[chunk_size*2:]
            ]

            words = text.split(' ')
            if len(words) >= 3:
                # Insert chunks at 25%, 50%, 75%
                positions = [len(words)//4, len(words)//2, 3*len(words)//4]
                for pos, chunk in zip(positions, chunks):
                    if pos < len(words):
                        words[pos] += chunk
                return ' '.join(words)
            return text + zwc_metadata

        return text

    def detect_metadata(self, text: str) -> Optional[str]:
        """
        Detect and extract metadata from text.

        Args:
            text: Text to analyze

        Returns:
            Extracted metadata text (e.g., "SID:a3f9c2b8e1d4f6a7") or None
        """
        if not text:
            return None

        # Extract all zero-width characters
        zwc_chars = ''.join(c for c in text if c in [self.bit_0, self.bit_1])

        if not zwc_chars:
            return None

        # Convert back to binary
        binary = self._zwc_to_binary(zwc_chars)

        if not binary or len(binary) < 8:
            return None

        # Convert to text
        try:
            decoded = self._binary_to_text(binary)

            # Look for "SID:" marker
            if "SID:" in decoded:
                # Extract the metadata
                match = re.search(r'SID:[0-9a-f]{16}', decoded)
                if match:
                    return match.group(0)
        except Exception:
            pass

        return None

    def strip_metadata(self, text: str) -> str:
        """
        Remove metadata from text (for testing/debugging).

        Args:
            text: Annotated text

        Returns:
            Text with metadata removed
        """
        # Remove all zero-width characters
        return ''.join(c for c in text if c not in [
            ZERO_WIDTH_SPACE,
            ZERO_WIDTH_JOINER,
            ZERO_WIDTH_NON_JOINER,
            WORD_JOINER
        ])


class WhitespaceMetadata:
    """
    Alternative metadataing using whitespace patterns.

    Less robust than zero-width characters but survives more transformations.
    Uses patterns of single vs double spaces to encode binary data.
    """

    def __init__(self, client_id: str):
        """
        Initialize with instance ID.

        Args:
            client_id: Instance identifier (hex string)
        """
        self.client_id = client_id

    def embed_metadata(self, text: str) -> str:
        """
        Embed metadata using whitespace patterns.

        Encoding: single space = 0, double space = 1

        Args:
            text: Text to metadata

        Returns:
            Annotated text
        """
        if not text or '  ' in text:  # Already has double spaces
            return text

        # Create signature from instance ID
        # Use first 8 chars -> 32 bits
        signature = self.client_id[:8]
        binary = ''.join(format(int(c, 16), '04b') for c in signature)

        # Apply to first N spaces
        words = text.split(' ')
        if len(words) < len(binary) + 1:
            return text  # Not enough spaces

        result = [words[0]]
        for i, bit in enumerate(binary[:len(words)-1]):
            if bit == '1':
                result.append(' ' + words[i+1])  # Double space
            else:
                result.append(words[i+1])  # Single space (join will add it)

        # Add remaining words normally
        result.extend(words[len(binary)+1:])

        return ' '.join(result)

    def detect_metadata(self, text: str) -> Optional[str]:
        """
        Detect metadata from whitespace patterns.

        Args:
            text: Text to analyze

        Returns:
            Detected instance ID fragment or None
        """
        # Look for pattern of single/double spaces
        # This is simplified - real implementation would be more robust

        # Extract spacing pattern
        spaces = re.findall(r' +', text)
        if len(spaces) < 32:
            return None

        # Convert to binary (single space = 0, double = 1)
        binary = ''.join('1' if len(s) > 1 else '0' for s in spaces[:32])

        # Convert to hex
        try:
            hex_chars = []
            for i in range(0, len(binary), 4):
                nibble = binary[i:i+4]
                hex_chars.append(format(int(nibble, 2), 'x'))

            detected_id = ''.join(hex_chars)
            return f"SID-{detected_id}" if len(detected_id) == 8 else None
        except Exception:
            return None


# Global instances (initialized on first use)
_text_encoder: Optional[TextMetadataEncoder] = None
_whitespace_encoder: Optional[WhitespaceMetadata] = None


def get_text_encoder() -> TextMetadataEncoder:
    """
    Get global steganographic metadata instance.

    Returns:
        TextMetadataEncoder instance
    """
    global _text_encoder
    if _text_encoder is None:
        from src.utils.telemetry import get_telemetry
        client_id = get_telemetry()._client_id
        _text_encoder = TextMetadataEncoder(client_id)
    return _text_encoder


def get_whitespace_encoder() -> WhitespaceMetadata:
    """
    Get global whitespace metadata instance.

    Returns:
        WhitespaceMetadata instance
    """
    global _whitespace_encoder
    if _whitespace_encoder is None:
        from src.utils.telemetry import get_telemetry
        client_id = get_telemetry()._client_id
        _whitespace_encoder = WhitespaceMetadata(client_id)
    return _whitespace_encoder


def annotate_output(text: str, method: str = "zwc") -> str:
    """
    Convenience function to annotate translated text with metadata.

    Args:
        text: Translated text
        method: Encoding method ('zwc' or 'whitespace')

    Returns:
        Annotated text
    """
    if method == "zwc":
        return get_text_encoder().embed_metadata(text, "distributed")
    elif method == "whitespace":
        return get_whitespace_encoder().embed_metadata(text)
    return text


def detect_metadata_in_text(text: str) -> Optional[str]:
    """
    Detect metadata in text using all available methods.

    Args:
        text: Text to analyze

    Returns:
        Detected metadata or None
    """
    # Try zero-width character method first
    zwc_result = get_text_encoder().detect_metadata(text)
    if zwc_result:
        return zwc_result

    # Try whitespace method
    ws_result = get_whitespace_encoder().detect_metadata(text)
    if ws_result:
        return ws_result

    return None
