"""
Base TTS Provider Abstract Class

Defines the interface that all TTS providers must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable, AsyncIterator


@dataclass
class VoiceInfo:
    """Information about an available voice"""
    name: str  # Full voice name (e.g., "zh-CN-XiaoxiaoNeural")
    short_name: str  # Short name for display
    language: str  # Language code (e.g., "zh-CN")
    gender: str  # "Male" or "Female"
    locale: str  # Full locale (e.g., "Chinese (Mandarin, Simplified)")


@dataclass
class TTSResult:
    """Result of a TTS synthesis operation"""
    success: bool
    audio_data: Optional[bytes] = None
    output_path: Optional[str] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


# Type alias for progress callback
# callback(current_chunk: int, total_chunks: int, message: str)
ProgressCallback = Callable[[int, int, str], None]


class TTSProvider(ABC):
    """
    Abstract base class for TTS providers.

    All TTS providers must implement these methods to ensure
    consistent behavior across different TTS engines.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name"""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming audio generation"""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ) -> bytes:
        """
        Synthesize text to audio bytes.

        Args:
            text: Text to synthesize
            voice: Voice name to use
            rate: Speed adjustment (e.g., "+10%", "-20%")
            volume: Volume adjustment (e.g., "+0%")
            pitch: Pitch adjustment (e.g., "+0Hz")

        Returns:
            Raw audio bytes (typically MP3 format from Edge-TTS)

        Raises:
            TTSError: If synthesis fails
        """
        pass

    @abstractmethod
    async def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ) -> TTSResult:
        """
        Synthesize text directly to a file.

        Args:
            text: Text to synthesize
            output_path: Path to save the audio file
            voice: Voice name to use
            rate: Speed adjustment
            volume: Volume adjustment
            pitch: Pitch adjustment

        Returns:
            TTSResult with success status and metadata
        """
        pass

    @abstractmethod
    async def get_available_voices(self, language_filter: Optional[str] = None) -> List[VoiceInfo]:
        """
        Get list of available voices.

        Args:
            language_filter: Optional language code to filter voices (e.g., "zh", "en")

        Returns:
            List of VoiceInfo objects
        """
        pass

    async def stream_synthesis(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ) -> AsyncIterator[bytes]:
        """
        Stream audio synthesis chunk by chunk.

        Default implementation returns single chunk.
        Override for providers that support true streaming.

        Args:
            text: Text to synthesize
            voice: Voice name to use
            rate: Speed adjustment
            volume: Volume adjustment
            pitch: Pitch adjustment

        Yields:
            Audio data chunks
        """
        audio_data = await self.synthesize(text, voice, rate, volume, pitch)
        yield audio_data

    def validate_voice(self, voice: str) -> bool:
        """
        Check if a voice name is valid format.

        Args:
            voice: Voice name to validate

        Returns:
            True if voice appears valid
        """
        # Basic validation - most voices follow pattern: xx-XX-NameNeural
        if not voice:
            return False
        parts = voice.split('-')
        return len(parts) >= 3


class TTSError(Exception):
    """Exception raised for TTS-related errors"""

    def __init__(self, message: str, provider: str = "", recoverable: bool = False):
        self.message = message
        self.provider = provider
        self.recoverable = recoverable
        super().__init__(f"[{provider}] {message}" if provider else message)
