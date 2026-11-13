"""
File utilities for translation operations
"""
import os
import re
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable, Tuple

from src.core.srt_processor import SRTProcessor

if TYPE_CHECKING:
    from src.tts.tts_config import TTSConfig


PARTIAL_PREFIX = "[partial] "
# Accept both the current `[partial] ` form and the legacy `[partial NN%] ` form
# so cleanup also catches files left behind by older versions.
_PARTIAL_RE = re.compile(r'^\[partial(?:\s+\d+%)?\]\s+')


def get_partial_output_path(output_path):
    """Return the path with a `[partial] ` prefix on the basename, used to mark
    an interrupted EPUB output so it cannot be confused with a completed file."""
    p = Path(output_path)
    base = _PARTIAL_RE.sub('', p.name)
    return str(p.parent / f"{PARTIAL_PREFIX}{base}")


def find_partial_output_paths(output_path):
    """Return all sibling files in the same directory that match the partial
    naming convention for ``output_path`` (current and legacy formats)."""
    p = Path(output_path)
    parent = p.parent if str(p.parent) else Path('.')
    if not parent.exists():
        return []
    target = _PARTIAL_RE.sub('', p.name)
    pattern = re.compile(r'^\[partial(?:\s+\d+%)?\]\s+' + re.escape(target) + r'$')
    return [str(entry) for entry in parent.iterdir()
            if entry.is_file() and pattern.match(entry.name)]


def get_unique_output_path(output_path):
    """
    Generate a unique output path by adding a number suffix if the file already exists.

    Args:
        output_path (str): Desired output path

    Returns:
        str: Unique output path (original or with numeric suffix)

    Examples:
        book.epub -> book.epub (if doesn't exist)
        book.epub -> book (1).epub (if book.epub exists)
        book.epub -> book (2).epub (if book.epub and book (1).epub exist)
    """
    path = Path(output_path)

    # If the file doesn't exist, return the original path
    if not path.exists():
        return output_path

    # Extract components
    parent = path.parent
    stem = path.stem  # filename without extension
    suffix = path.suffix  # .epub, .txt, .srt, etc.

    # Try incrementing numbers until we find a free filename
    counter = 1
    while True:
        new_stem = f"{stem} ({counter})"
        new_path = parent / f"{new_stem}{suffix}"

        if not new_path.exists():
            return str(new_path)

        counter += 1

        # Safety check to avoid infinite loops (highly unlikely)
        if counter > 9999:
            raise RuntimeError(f"Could not find unique filename after 9999 attempts for: {output_path}")




def _extract_text_from_txt(filepath: str) -> str:
    """Extract text from a plain text file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def _extract_text_from_epub(filepath: str) -> str:
    """
    Extract readable text from an EPUB file.

    Parses all HTML/XHTML content files and extracts text,
    removing HTML tags and keeping only readable content.
    """
    text_parts = []

    with zipfile.ZipFile(filepath, 'r') as epub:
        for name in epub.namelist():
            if name.endswith(('.html', '.xhtml', '.htm')):
                try:
                    content = epub.read(name).decode('utf-8')
                    # Remove HTML tags
                    clean_text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    clean_text = re.sub(r'<style[^>]*>.*?</style>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
                    clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
                    # Clean up whitespace
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    # Decode HTML entities
                    clean_text = clean_text.replace('&nbsp;', ' ')
                    clean_text = clean_text.replace('&amp;', '&')
                    clean_text = clean_text.replace('&lt;', '<')
                    clean_text = clean_text.replace('&gt;', '>')
                    clean_text = clean_text.replace('&quot;', '"')
                    clean_text = clean_text.replace('&#39;', "'")

                    if clean_text:
                        text_parts.append(clean_text)
                except Exception:
                    continue

    return '\n\n'.join(text_parts)


def _extract_text_from_srt(filepath: str) -> str:
    """
    Extract readable text from an SRT subtitle file.

    Extracts only the subtitle text, removing timing information
    and index numbers.
    """
    srt_processor = SRTProcessor()

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    subtitles = srt_processor.parse_srt(content)

    # Extract just the text from each subtitle
    text_parts = [sub.get('text', '') for sub in subtitles if sub.get('text')]

    return ' '.join(text_parts)


def extract_text_from_file(filepath: str) -> str:
    """
    Extract readable text from a translated file.

    Supports txt, epub, and srt files. Used for TTS generation
    after translation is complete.

    Args:
        filepath: Path to the translated file

    Returns:
        Extracted text content

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    _, ext = os.path.splitext(filepath.lower())

    if ext == '.txt':
        return _extract_text_from_txt(filepath)
    elif ext == '.epub':
        return _extract_text_from_epub(filepath)
    elif ext == '.srt':
        return _extract_text_from_srt(filepath)
    else:
        raise ValueError(f"Unsupported file type for TTS: {ext}")


async def generate_tts_for_translation(
    translated_filepath: str,
    target_language: str,
    tts_config: 'TTSConfig',
    log_callback: Optional[Callable] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Generate TTS audio from a translated file.

    Extracts text from the translated file (txt, epub, or srt),
    then generates audio using the configured TTS provider.

    Args:
        translated_filepath: Path to the translated file
        target_language: Target language (for voice selection)
        tts_config: TTS configuration object
        log_callback: Optional logging callback
    Returns:
        Tuple of (success: bool, message: str, audio_path: Optional[str])
    """
    from src.tts.audio_processor import generate_tts_for_text

    if log_callback:
        log_callback("tts_start", f"Starting TTS generation for: {translated_filepath}")

    # Generate output audio path
    base, _ = os.path.splitext(translated_filepath)
    audio_extension = tts_config.get_output_extension()
    audio_path = f"{base}_audio{audio_extension}"

    # Ensure unique path
    audio_path = get_unique_output_path(audio_path)

    try:
        # Extract text from translated file
        if log_callback:
            log_callback("tts_extract", "Extracting text from translated file...")

        text = extract_text_from_file(translated_filepath)

        if not text.strip():
            return False, "No text found in translated file", None

        text_length = len(text)
        if log_callback:
            log_callback("tts_text_extracted", f"Extracted {text_length:,} characters for TTS")

        # Set target language in config
        tts_config.target_language = target_language

        # Create progress wrapper for TTS
        def tts_progress(current, total, message):
            if log_callback:
                log_callback("tts_progress", f"TTS: {message} ({current}/{total})")
            if progress_callback:  # Pass all arguments to the callback
                progress_callback(current, total, message)

        # Generate audio
        if log_callback:
            log_callback("tts_synthesize", f"Synthesizing audio with voice: {tts_config.get_effective_voice(target_language)}")

        success, message = await generate_tts_for_text(
            text=text,
            output_path=audio_path,
            config=tts_config,
            language=target_language,
            progress_callback=tts_progress
        )

        if success:
            if log_callback:
                log_callback("tts_complete", f"TTS audio saved: {audio_path}")
            return True, message, audio_path
        else:
            if log_callback:
                log_callback("tts_error", f"TTS generation failed: {message}")
            return False, message, None

    except FileNotFoundError as e:
        error_msg = f"Translated file not found: {e}"
        if log_callback:
            log_callback("tts_error", error_msg)
        return False, error_msg, None
    except ValueError as e:
        error_msg = f"Unsupported file type: {e}"
        if log_callback:
            log_callback("tts_error", error_msg)
        return False, error_msg, None
    except Exception as e:
        error_msg = f"TTS generation error: {e}"
        if log_callback:
            log_callback("tts_error", error_msg)
        return False, error_msg, None