"""
Audio Processor for TTS Generation

Handles text chunking, synthesis orchestration, audio concatenation,
and encoding to Opus format via ffmpeg.
"""
import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from .tts_config import TTSConfig
from .providers.base import TTSProvider, TTSError, ProgressCallback
from .providers.edge_tts import EdgeTTSProvider
from .providers.chatterbox_tts import ChatterboxProvider, is_chatterbox_available, MAX_TEXT_LENGTH as CHATTERBOX_MAX_LENGTH

logger = logging.getLogger(__name__)

# Sentence-ending punctuation for chunking
SENTENCE_ENDINGS = re.compile(r'[.!?。！？…]+[\s\n]*')
PARAGRAPH_BREAK = re.compile(r'\n\s*\n')


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in the system PATH"""
    return shutil.which('ffmpeg') is not None


def get_ffmpeg_install_instructions() -> str:
    """
    Get platform-specific installation instructions for ffmpeg.

    Returns:
        Human-readable installation instructions
    """
    import platform
    system = platform.system().lower()

    instructions = "\n" + "=" * 60 + "\n"
    instructions += "FFmpeg is required for audio encoding but was not found.\n"
    instructions += "=" * 60 + "\n\n"

    if system == "windows":
        instructions += "WINDOWS - Choose one method:\n\n"
        instructions += "Option 1 - WinGet (Recommended, Windows 10/11):\n"
        instructions += "  Open PowerShell/Terminal as Administrator and run:\n"
        instructions += "  > winget install Gyan.FFmpeg\n"
        instructions += "  Then restart your terminal/application.\n\n"
        instructions += "Option 2 - Chocolatey:\n"
        instructions += "  > choco install ffmpeg\n\n"
        instructions += "Option 3 - Manual:\n"
        instructions += "  1. Download from https://ffmpeg.org/download.html\n"
        instructions += "  2. Extract to C:\\ffmpeg\n"
        instructions += "  3. Add C:\\ffmpeg\\bin to your PATH\n"
    elif system == "darwin":
        instructions += "macOS - Using Homebrew:\n"
        instructions += "  $ brew install ffmpeg\n\n"
        instructions += "If you don't have Homebrew:\n"
        instructions += "  $ /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\n"
    else:  # Linux
        instructions += "LINUX:\n\n"
        instructions += "Ubuntu/Debian:\n"
        instructions += "  $ sudo apt update && sudo apt install ffmpeg\n\n"
        instructions += "Fedora:\n"
        instructions += "  $ sudo dnf install ffmpeg\n\n"
        instructions += "Arch:\n"
        instructions += "  $ sudo pacman -S ffmpeg\n"

    instructions += "\n" + "-" * 60 + "\n"
    instructions += "After installation, restart your terminal/application.\n"
    instructions += "=" * 60 + "\n"

    return instructions


def check_ffmpeg_with_instructions() -> Tuple[bool, str]:
    """
    Check if ffmpeg is available and return installation instructions if not.

    Returns:
        Tuple of (is_available: bool, message: str)
        - If available: (True, "ffmpeg found: <version>")
        - If not: (False, "<installation instructions>")
    """
    if check_ffmpeg_available():
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_line = result.stdout.split('\n')[0] if result.stdout else "unknown version"
            return True, f"ffmpeg found: {version_line}"
        except Exception:
            return True, "ffmpeg found"

    return False, get_ffmpeg_install_instructions()


def get_ffmpeg_status() -> dict:
    """
    Get detailed FFmpeg status for API responses.

    Returns:
        Dict with availability status and version info
    """
    import platform
    available = check_ffmpeg_available()
    result = {
        "available": available,
        "platform": platform.system().lower(),
        "version": None,
        "can_auto_install": platform.system().lower() == "windows"
    }

    if available:
        try:
            proc = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.stdout:
                result["version"] = proc.stdout.split('\n')[0]
        except Exception:
            result["version"] = "unknown"

    return result


def install_ffmpeg_windows() -> Tuple[bool, str]:
    """
    Attempt to install FFmpeg on Windows using winget.

    Returns:
        Tuple of (success: bool, message: str)
    """
    import platform
    if platform.system().lower() != "windows":
        return False, "Auto-installation is only supported on Windows"

    logger.info("Attempting to install FFmpeg via winget...")

    try:
        # Check if winget is available
        winget_check = subprocess.run(
            ['winget', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if winget_check.returncode != 0:
            return False, "winget is not available. Please install FFmpeg manually."

        # Install FFmpeg using winget
        result = subprocess.run(
            ['winget', 'install', 'Gyan.FFmpeg', '--accept-package-agreements', '--accept-source-agreements'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout for download/install
        )

        if result.returncode == 0:
            logger.info("FFmpeg installed successfully via winget")
            return True, "FFmpeg installed successfully! Please restart the application to use TTS."
        else:
            # Check if already installed
            if "already installed" in result.stdout.lower() or "already installed" in result.stderr.lower():
                return True, "FFmpeg is already installed. Please restart the application."

            error_msg = result.stderr or result.stdout or "Unknown error"
            logger.error(f"winget install failed: {error_msg}")
            return False, f"Installation failed: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Installation timed out. Please try installing FFmpeg manually."
    except FileNotFoundError:
        return False, "winget not found. Please install FFmpeg manually or install winget first."
    except Exception as e:
        logger.exception("Error during FFmpeg installation")
        return False, f"Installation error: {str(e)}"


def chunk_text_for_tts(text: str, max_chunk_size: int = 5000) -> List[str]:
    """
    Split text into chunks suitable for TTS synthesis.

    Respects sentence boundaries to avoid cutting words mid-sentence.
    Aims for natural pauses at paragraph and sentence breaks.

    Args:
        text: Full text to chunk
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # Normalize whitespace
    text = text.strip()

    # If text is small enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by paragraphs first
    paragraphs = PARAGRAPH_BREAK.split(text)

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If paragraph fits in current chunk, add it
        if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            # Paragraph doesn't fit - need to split by sentences
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # If single paragraph is too long, split by sentences
            if len(paragraph) > max_chunk_size:
                sentences = SENTENCE_ENDINGS.split(paragraph)
                sentence_ends = SENTENCE_ENDINGS.findall(paragraph)

                for i, sentence in enumerate(sentences):
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    # Add back the punctuation
                    if i < len(sentence_ends):
                        sentence += sentence_ends[i].strip()

                    if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                        if current_chunk:
                            current_chunk += " "
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # If single sentence is too long, force split
                        if len(sentence) > max_chunk_size:
                            for j in range(0, len(sentence), max_chunk_size):
                                chunks.append(sentence[j:j + max_chunk_size])
                            current_chunk = ""
                        else:
                            current_chunk = sentence
            else:
                current_chunk = paragraph

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class AudioProcessor:
    """
    Orchestrates TTS generation for complete documents.

    Handles:
    - Text chunking optimized for TTS
    - Chunk-by-chunk synthesis with progress callbacks
    - Audio concatenation
    - Opus encoding via ffmpeg
    """

    def __init__(self, config: TTSConfig, provider: Optional[TTSProvider] = None):
        """
        Initialize the AudioProcessor.

        Args:
            config: TTS configuration
            provider: TTS provider instance (defaults to EdgeTTSProvider)
        """
        self.config = config
        self.provider = provider or EdgeTTSProvider()
        self._temp_dir: Optional[Path] = None

        # Adjust chunk size based on provider
        # Chatterbox has strict tokenizer limits
        if isinstance(self.provider, ChatterboxProvider):
            # Use smaller chunks for Chatterbox (with safety margin)
            self._effective_chunk_size = min(config.chunk_size, CHATTERBOX_MAX_LENGTH - 50)
            logger.info(f"Using Chatterbox-optimized chunk size: {self._effective_chunk_size}")
        else:
            self._effective_chunk_size = config.chunk_size

    async def generate_audio(
        self,
        text: str,
        output_path: str,
        language: str = "",
        progress_callback: Optional[ProgressCallback] = None
    ) -> Tuple[bool, str]:
        """
        Generate audio from text and save to output file.

        Args:
            text: Text to convert to speech
            output_path: Destination path for audio file
            language: Target language for voice selection
            progress_callback: Optional callback(current, total, message)

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not text.strip():
            return False, "No text provided for TTS generation"

        # Determine voice
        voice = self.config.get_effective_voice(language)
        if not voice:
            return False, f"Could not determine voice for language: {language}"

        logger.info(f"Starting TTS generation with voice: {voice}")

        # Check if we need opus encoding
        needs_encoding = self.config.output_format.lower() == 'opus'
        if needs_encoding:
            ffmpeg_available, ffmpeg_message = check_ffmpeg_with_instructions()
            if not ffmpeg_available:
                return False, ffmpeg_message

        try:
            # Create temp directory for intermediate files
            self._temp_dir = Path(tempfile.mkdtemp(prefix="tts_"))

            # Chunk the text (use provider-appropriate chunk size)
            chunks = chunk_text_for_tts(text, self._effective_chunk_size)
            total_chunks = len(chunks)

            if total_chunks == 0:
                return False, "Text produced no chunks for synthesis"

            logger.info(f"Text split into {total_chunks} chunks")

            # Synthesize each chunk
            temp_audio_files: List[Path] = []

            for i, chunk in enumerate(chunks):
                if progress_callback:
                    progress_callback(i + 1, total_chunks, f"Synthesizing chunk {i + 1}/{total_chunks}")

                # Generate temp file path for this chunk
                temp_file = self._temp_dir / f"chunk_{i:04d}.mp3"

                # Synthesize
                result = await self.provider.synthesize_to_file(
                    text=chunk,
                    output_path=str(temp_file),
                    voice=voice,
                    rate=self.config.rate,
                    volume=self.config.volume,
                    pitch=self.config.pitch
                )

                if not result.success:
                    return False, f"Failed to synthesize chunk {i + 1}: {result.error_message}"

                temp_audio_files.append(temp_file)

            # Concatenate and encode
            if progress_callback:
                progress_callback(total_chunks, total_chunks, "Concatenating audio...")

            if needs_encoding:
                success, message = await self._concatenate_and_encode_opus(
                    temp_audio_files,
                    output_path
                )
            else:
                success, message = await self._concatenate_mp3(
                    temp_audio_files,
                    output_path
                )

            if success:
                logger.info(f"TTS generation complete: {output_path}")

            return success, message

        except TTSError as e:
            return False, str(e)
        except Exception as e:
            logger.exception("Unexpected error during TTS generation")
            return False, f"TTS generation failed: {e}"
        finally:
            # Cleanup temp directory
            self._cleanup_temp()

    async def _concatenate_mp3(
        self,
        input_files: List[Path],
        output_path: str
    ) -> Tuple[bool, str]:
        """
        Concatenate MP3 files without re-encoding.

        Args:
            input_files: List of MP3 files to concatenate
            output_path: Destination path

        Returns:
            Tuple of (success, message)
        """
        if len(input_files) == 1:
            # Just copy the single file
            shutil.copy(input_files[0], output_path)
            return True, "Audio saved successfully"

        try:
            # Create concat file list for ffmpeg
            concat_file = self._temp_dir / "concat.txt"
            with open(concat_file, 'w', encoding='utf-8') as f:
                for audio_file in input_files:
                    # Escape single quotes in path
                    escaped_path = str(audio_file).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                output_path
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await result.communicate()

            if result.returncode != 0:
                return False, f"ffmpeg concatenation failed: {stderr.decode()}"

            return True, "Audio concatenated successfully"

        except Exception as e:
            return False, f"MP3 concatenation failed: {e}"

    async def _concatenate_and_encode_opus(
        self,
        input_files: List[Path],
        output_path: str
    ) -> Tuple[bool, str]:
        """
        Concatenate MP3 files and encode to Opus.

        Args:
            input_files: List of MP3 files to concatenate
            output_path: Destination path for Opus file

        Returns:
            Tuple of (success, message)
        """
        try:
            # Create concat file list
            concat_file = self._temp_dir / "concat.txt"
            with open(concat_file, 'w', encoding='utf-8') as f:
                for audio_file in input_files:
                    escaped_path = str(audio_file).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # ffmpeg command to concat and encode to opus
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:a', 'libopus',
                '-b:a', self.config.bitrate,
                '-ar', str(self.config.sample_rate),
                '-ac', '1',  # Mono for speech
                '-application', 'voip',  # Optimized for speech
                output_path
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await result.communicate()

            if result.returncode != 0:
                return False, f"Opus encoding failed: {stderr.decode()}"

            return True, "Audio encoded to Opus successfully"

        except Exception as e:
            return False, f"Opus encoding failed: {e}"

    def _cleanup_temp(self):
        """Remove temporary directory and files"""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
            self._temp_dir = None


def create_tts_provider(provider_name: str = "edge-tts", **kwargs) -> TTSProvider:
    """
    Factory function to create a TTS provider.

    Args:
        provider_name: Name of the provider to create
        **kwargs: Additional arguments for provider (e.g., voice_prompt_path for chatterbox)

    Returns:
        TTSProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider_name == "edge-tts":
        return EdgeTTSProvider()

    elif provider_name == "chatterbox":
        if not is_chatterbox_available():
            raise ValueError(
                "Chatterbox TTS is not available. Install with: "
                "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && "
                "pip install chatterbox-tts"
            )
        return ChatterboxProvider(
            voice_prompt_path=kwargs.get('voice_prompt_path'),
            exaggeration=kwargs.get('exaggeration', 0.5),
            cfg_weight=kwargs.get('cfg_weight', 0.5)
        )

    else:
        supported = ["edge-tts"]
        if is_chatterbox_available():
            supported.append("chatterbox")
        raise ValueError(f"Unknown TTS provider: {provider_name}. Supported: {supported}")


async def generate_tts_for_text(
    text: str,
    output_path: str,
    config: Optional[TTSConfig] = None,
    language: str = "",
    progress_callback: Optional[ProgressCallback] = None
) -> Tuple[bool, str]:
    """
    High-level function to generate TTS audio from text.

    Args:
        text: Text to convert to speech
        output_path: Destination path for audio file
        config: TTS configuration (uses defaults if None)
        language: Target language for voice selection
        progress_callback: Optional progress callback

    Returns:
        Tuple of (success: bool, message: str)
    """
    if config is None:
        config = TTSConfig.from_env()
        config.target_language = language

    provider = create_tts_provider(
        config.provider,
        voice_prompt_path=config.voice_prompt_path,
        exaggeration=config.exaggeration,
        cfg_weight=config.cfg_weight
    )
    processor = AudioProcessor(config, provider)

    return await processor.generate_audio(
        text=text,
        output_path=output_path,
        language=language,
        progress_callback=progress_callback
    )
