"""
Centralized file type detection with content-based analysis.

This module provides file type detection based on:
1. File extension (fast path for known extensions)
2. Magic bytes (for binary formats like EPUB, DOCX, ZIP)
3. Content heuristics (for text-based formats like SRT, plain text)

This allows translation of files with non-standard extensions (e.g., .log, .md, .text)
by analyzing their actual content.
"""
import os
import re
import zipfile
from typing import Literal, Optional, Tuple
from pathlib import Path

FileType = Literal["txt", "epub", "srt", "docx"]

# Known text file extensions that should be treated as plain text
KNOWN_TEXT_EXTENSIONS = {
    '.txt', '.text', '.log', '.md', '.markdown', '.rst', '.asc',
    '.cfg', '.ini', '.conf', '.csv', '.tsv', '.json', '.xml', '.html', '.htm',
    '.yaml', '.yml', '.toml', '.properties', '.env'
}

# Extensions with dedicated processors
PROCESSOR_EXTENSIONS = {
    '.epub': 'epub',
    '.srt': 'srt',
    '.docx': 'docx'
}


def detect_file_type(file_path: str) -> FileType:
    """
    Detect file type from extension or content analysis.

    This function first checks the file extension. For unknown extensions,
    it analyzes the file content to determine if it's a supported format.

    Args:
        file_path: Path to the file

    Returns:
        File type as string ('txt', 'epub', 'srt', 'docx')

    Raises:
        ValueError: If file type cannot be determined or is not supported
    """
    _, ext = os.path.splitext(file_path.lower())

    # Fast path: known processor extensions
    if ext in PROCESSOR_EXTENSIONS:
        return PROCESSOR_EXTENSIONS[ext]

    # Fast path: known text extensions -> treat as txt
    if ext in KNOWN_TEXT_EXTENSIONS:
        return "txt"

    # For .txt specifically, return immediately
    if ext == '.txt':
        return "txt"

    # Unknown extension: analyze content
    detected_type = detect_file_type_by_content(file_path)
    if detected_type:
        return detected_type

    # Could not determine type
    raise ValueError(
        f"Cannot determine file type for: {ext}. "
        f"Supported types: .txt, .epub, .srt, .docx, "
        f"or plain text files with any extension."
    )


def detect_file_type_by_content(file_path: str) -> Optional[FileType]:
    """
    Detect file type by analyzing file content.

    Uses magic bytes for binary formats and heuristics for text formats.
    This is useful for files with non-standard or missing extensions.

    Args:
        file_path: Path to the file

    Returns:
        Detected file type or None if cannot be determined
    """
    if not os.path.exists(file_path):
        return None

    # Check binary formats first (magic bytes)
    binary_type = _detect_binary_format(file_path)
    if binary_type:
        return binary_type

    # Check if it's a text file and detect format
    text_type = _detect_text_format(file_path)
    if text_type:
        return text_type

    return None


def _detect_binary_format(file_path: str) -> Optional[FileType]:
    """
    Detect binary file formats using magic bytes.

    Args:
        file_path: Path to the file

    Returns:
        Detected file type or None
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(64)  # Read first 64 bytes for magic detection

        if len(header) < 4:
            return None

        # ZIP-based formats (EPUB, DOCX, ODT are all ZIP archives)
        # ZIP magic bytes: PK\x03\x04 or PK\x05\x06 (empty) or PK\x07\x08 (spanned)
        if header[:2] == b'PK':
            return _identify_zip_format(file_path)

        return None

    except Exception:
        return None


def _identify_zip_format(file_path: str) -> Optional[FileType]:
    """
    Identify the specific format of a ZIP-based file.

    EPUB and DOCX are ZIP archives with specific structures.

    Args:
        file_path: Path to the ZIP file

    Returns:
        'epub', 'docx', or None
    """
    try:
        if not zipfile.is_zipfile(file_path):
            return None

        with zipfile.ZipFile(file_path, 'r') as zf:
            namelist = zf.namelist()

            # EPUB: has mimetype file containing "application/epub+zip"
            if 'mimetype' in namelist:
                try:
                    mimetype_content = zf.read('mimetype').decode('utf-8', errors='ignore').strip()
                    if 'epub' in mimetype_content.lower():
                        return 'epub'
                except Exception:
                    pass

            # Also check for META-INF/container.xml (EPUB standard)
            if 'META-INF/container.xml' in namelist:
                return 'epub'

            # DOCX: has [Content_Types].xml and word/ directory
            if '[Content_Types].xml' in namelist:
                has_word = any(n.startswith('word/') for n in namelist)
                if has_word:
                    return 'docx'

        return None

    except Exception:
        return None


def _detect_text_format(file_path: str) -> Optional[FileType]:
    """
    Detect text-based file formats by analyzing content.

    Checks for:
    - SRT subtitle format (numbered entries with timecodes)
    - Plain text (readable text content)

    Args:
        file_path: Path to the file

    Returns:
        'srt', 'txt', or None
    """
    try:
        # Try to read as text with multiple encodings
        content = _read_text_file_safe(file_path, max_bytes=8192)
        if content is None:
            return None

        # Check for SRT format
        if _is_srt_format(content):
            return 'srt'

        # Check if it's readable text (not binary garbage)
        if _is_readable_text(content):
            return 'txt'

        return None

    except Exception:
        return None


def _read_text_file_safe(file_path: str, max_bytes: int = 8192) -> Optional[str]:
    """
    Safely read a text file trying multiple encodings.

    Args:
        file_path: Path to the file
        max_bytes: Maximum bytes to read

    Returns:
        File content as string or None if not readable as text
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

    # Read raw bytes first
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read(max_bytes)
    except Exception:
        return None

    # Check for BOM
    if raw_content.startswith(b'\xef\xbb\xbf'):
        raw_content = raw_content[3:]  # UTF-8 BOM
    elif raw_content.startswith(b'\xff\xfe'):
        try:
            return raw_content.decode('utf-16-le')
        except Exception:
            pass
    elif raw_content.startswith(b'\xfe\xff'):
        try:
            return raw_content.decode('utf-16-be')
        except Exception:
            pass

    # Try encodings in order
    for encoding in encodings:
        try:
            return raw_content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    return None


def _is_srt_format(content: str) -> bool:
    """
    Check if content appears to be SRT subtitle format.

    SRT format has:
    - Numbered subtitle entries (1, 2, 3, ...)
    - Timecodes in format: HH:MM:SS,mmm --> HH:MM:SS,mmm
    - Text content

    Args:
        content: Text content to analyze

    Returns:
        True if content appears to be SRT format
    """
    # SRT timecode pattern: 00:00:00,000 --> 00:00:00,000
    srt_timecode_pattern = re.compile(
        r'\d{1,2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{3}',
        re.MULTILINE
    )

    # SRT entry number pattern: single digit on its own line
    srt_number_pattern = re.compile(r'^\d+\s*$', re.MULTILINE)

    # Count matches
    timecode_matches = len(srt_timecode_pattern.findall(content))
    number_matches = len(srt_number_pattern.findall(content))

    # Need at least 2 timecode patterns and corresponding numbers to be confident
    return timecode_matches >= 2 and number_matches >= 2


def _is_readable_text(content: str) -> bool:
    """
    Check if content is readable text (not binary garbage).

    Uses heuristics:
    - High ratio of printable characters
    - Contains word-like patterns
    - Low ratio of control characters (except newlines/tabs)

    Args:
        content: Text content to analyze

    Returns:
        True if content appears to be readable text
    """
    if not content or len(content) < 10:
        return False

    # Count character types
    printable_count = 0
    control_count = 0

    for char in content:
        code = ord(char)
        if char.isprintable() or char in '\n\r\t':
            printable_count += 1
        elif code < 32 and code not in (9, 10, 13):  # Control chars except tab, newline, CR
            control_count += 1

    total = len(content)
    printable_ratio = printable_count / total
    control_ratio = control_count / total

    # Text should be mostly printable with very few control characters
    # Threshold: 95% printable, less than 1% control characters
    return printable_ratio > 0.95 and control_ratio < 0.01


def detect_file_type_safe(file_path: str) -> Tuple[Optional[FileType], Optional[str]]:
    """
    Safely detect file type, returning error message instead of raising.

    This is useful for validation flows where you want to provide
    user-friendly error messages.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (file_type, error_message)
        - If successful: (file_type, None)
        - If failed: (None, error_message)
    """
    try:
        file_type = detect_file_type(file_path)
        return file_type, None
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error detecting file type: {str(e)}"


def generate_output_filename(input_path: str, target_language: str) -> str:
    """
    Generate output filename based on input and target language
    
    Args:
        input_path: Input file path
        target_language: Target language
        
    Returns:
        Generated output filename
    """
    base, ext = os.path.splitext(input_path)
    lang_suffix = target_language.lower().replace(' ', '_')
    return f"{base}_{lang_suffix}{ext}"