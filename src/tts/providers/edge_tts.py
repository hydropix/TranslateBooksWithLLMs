"""
Edge-TTS Provider Implementation

Uses Microsoft Edge's text-to-speech API via the edge-tts library.
Provides high-quality neural voices for free without API key.
"""
import logging
from typing import Optional, List, AsyncIterator
from pathlib import Path

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

from .base import TTSProvider, TTSResult, VoiceInfo, TTSError

logger = logging.getLogger(__name__)


class EdgeTTSProvider(TTSProvider):
    """
    TTS provider using Microsoft Edge's neural voices.

    Features:
    - Free, no API key required
    - High-quality neural voices
    - Supports 300+ voices in 100+ languages
    - Streaming support
    - Adjustable rate, volume, and pitch
    """

    def __init__(self):
        if not EDGE_TTS_AVAILABLE:
            raise TTSError(
                "edge-tts library not installed. Install with: pip install edge-tts",
                provider=self.name,
                recoverable=False
            )
        self._voices_cache: Optional[List[VoiceInfo]] = None

    @property
    def name(self) -> str:
        return "edge-tts"

    @property
    def supports_streaming(self) -> bool:
        return True

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
            voice: Voice name (e.g., "zh-CN-XiaoxiaoNeural")
            rate: Speed adjustment (e.g., "+10%", "-20%")
            volume: Volume adjustment
            pitch: Pitch adjustment

        Returns:
            MP3 audio bytes
        """
        if not text.strip():
            raise TTSError("Cannot synthesize empty text", provider=self.name)

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch
            )

            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            if not audio_chunks:
                raise TTSError("No audio data received", provider=self.name)

            return b"".join(audio_chunks)

        except Exception as e:
            if "edge_tts" in str(type(e).__module__):
                raise TTSError(f"Edge-TTS error: {e}", provider=self.name, recoverable=True)
            raise TTSError(f"Synthesis failed: {e}", provider=self.name)

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
            output_path: Path to save the audio file (MP3)
            voice: Voice name
            rate: Speed adjustment
            volume: Volume adjustment
            pitch: Pitch adjustment

        Returns:
            TTSResult with success status
        """
        if not text.strip():
            return TTSResult(
                success=False,
                error_message="Cannot synthesize empty text"
            )

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch
            )

            await communicate.save(output_path)

            # Verify file was created
            path = Path(output_path)
            if not path.exists():
                return TTSResult(
                    success=False,
                    error_message="Output file was not created"
                )

            return TTSResult(
                success=True,
                output_path=output_path,
            )

        except Exception as e:
            logger.error(f"Edge-TTS synthesis to file failed: {e}")
            return TTSResult(
                success=False,
                error_message=str(e)
            )

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

        Yields MP3 audio chunks as they are generated.
        """
        if not text.strip():
            return

        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            logger.error(f"Edge-TTS streaming failed: {e}")
            raise TTSError(f"Streaming synthesis failed: {e}", provider=self.name)

    async def get_available_voices(self, language_filter: Optional[str] = None) -> List[VoiceInfo]:
        """
        Get list of available voices from Edge-TTS.

        Args:
            language_filter: Optional language code to filter (e.g., "zh", "en", "fr")

        Returns:
            List of VoiceInfo objects
        """
        # Use cache if available
        if self._voices_cache is None:
            try:
                voices_raw = await edge_tts.list_voices()
                self._voices_cache = [
                    VoiceInfo(
                        name=v["ShortName"],
                        short_name=v["ShortName"].split("-")[-1].replace("Neural", ""),
                        language=v["Locale"],
                        gender=v["Gender"],
                        locale=v.get("LocaleName", v["Locale"])
                    )
                    for v in voices_raw
                ]
            except Exception as e:
                logger.error(f"Failed to fetch voices: {e}")
                return []

        if not language_filter:
            return self._voices_cache

        # Filter by language code
        filter_lower = language_filter.lower()
        return [
            v for v in self._voices_cache
            if filter_lower in v.language.lower() or filter_lower in v.locale.lower()
        ]

    async def get_voice_by_language(self, language: str, gender: str = "Female") -> Optional[str]:
        """
        Get a recommended voice for a language.

        Args:
            language: Language name or code
            gender: Preferred gender ("Male" or "Female")

        Returns:
            Voice name or None if not found
        """
        voices = await self.get_available_voices(language)
        if not voices:
            return None

        # Prefer specified gender
        gender_match = [v for v in voices if v.gender == gender]
        if gender_match:
            return gender_match[0].name

        # Fallback to any voice
        return voices[0].name


def create_edge_tts_provider() -> EdgeTTSProvider:
    """Factory function to create an EdgeTTSProvider instance"""
    return EdgeTTSProvider()
