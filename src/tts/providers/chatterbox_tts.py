"""
Chatterbox TTS Provider Implementation

Uses ResembleAI's Chatterbox TTS model for local GPU-accelerated speech synthesis.
Supports voice cloning and 23 languages.

Key Features:
- Local GPU inference (CUDA)
- Voice cloning from audio samples
- Emotion/exaggeration control
- 23 language support
"""
import asyncio
import logging
import os
import re
import unicodedata
from typing import Optional, List, AsyncIterator
from pathlib import Path
import io

from .base import TTSProvider, TTSResult, VoiceInfo, TTSError

logger = logging.getLogger(__name__)

# The heavy optional dependencies (torch, torchaudio, chatterbox-tts) are only
# imported on first real use (_load_backend), never when this module is
# imported: importing torch costs seconds and the module is reachable from
# every CLI and server start even with TTS disabled.
torch = None
torchaudio = None
ChatterboxTTS = None
_BACKEND: Optional[dict] = None
_BACKEND_FLAGS = {
    "TORCH_AVAILABLE": "torch",
    "CHATTERBOX_AVAILABLE": "chatterbox",
    "TORCHAUDIO_AVAILABLE": "torchaudio",
}


def _load_backend() -> dict:
    """Import torch, chatterbox-tts and torchaudio once; return what is available."""
    global torch, torchaudio, ChatterboxTTS, _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    backend = {"torch": False, "chatterbox": False, "torchaudio": False}
    try:
        import torch as _torch
        torch = _torch
        backend["torch"] = True
    except ImportError:
        pass
    try:
        from chatterbox.tts import ChatterboxTTS as _ChatterboxTTS
        ChatterboxTTS = _ChatterboxTTS
        backend["chatterbox"] = True
    except ImportError:
        pass
    try:
        import torchaudio as _torchaudio
        torchaudio = _torchaudio
        backend["torchaudio"] = True
    except ImportError:
        pass
    _BACKEND = backend
    return backend


def __getattr__(name: str):
    """Keep the former TORCH_AVAILABLE-style module flags readable (probed lazily)."""
    if name in _BACKEND_FLAGS:
        return _load_backend()[_BACKEND_FLAGS[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Maximum text length per synthesis call (characters)
# Chatterbox has internal tokenizer limits - exceeding them causes CUDA index errors
MAX_TEXT_LENGTH = 500

# Characters that may cause tokenization issues
PROBLEMATIC_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def sanitize_text_for_tts(text: str) -> str:
    """
    Sanitize text to prevent CUDA index errors in Chatterbox.

    Removes/replaces characters that may cause tokenization issues
    leading to out-of-bounds tensor indices.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text safe for TTS synthesis
    """
    if not text:
        return ""

    # Remove control characters except newlines and tabs
    text = PROBLEMATIC_CHARS_PATTERN.sub('', text)

    # Normalize unicode (NFC form)
    text = unicodedata.normalize('NFC', text)

    # Replace problematic unicode characters with ASCII equivalents
    replacements = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2013': '-',   # En dash
        '\u2014': '-',   # Em dash
        '\u2026': '...', # Ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\u200b': '',    # Zero-width space
        '\u200c': '',    # Zero-width non-joiner
        '\u200d': '',    # Zero-width joiner
        '\ufeff': '',    # BOM
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# Chatterbox supported languages (23 languages)
CHATTERBOX_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "id": "Indonesian",
    "el": "Greek",
}


class ChatterboxProvider(TTSProvider):
    """
    TTS provider using ResembleAI's Chatterbox model.

    Features:
    - Local GPU inference with CUDA support
    - Voice cloning from audio prompts
    - Emotion/exaggeration control
    - 23 language support
    - Automatic GPU/CPU fallback
    """

    def __init__(
        self,
        voice_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5
    ):
        """
        Initialize ChatterboxProvider.

        Args:
            voice_prompt_path: Path to audio file for voice cloning (optional)
            exaggeration: Emotion exaggeration level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight for stability
        """
        self._validate_dependencies()

        self.voice_prompt_path = voice_prompt_path
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

        # Determine device
        self.device = self._get_device()
        logger.info(f"Chatterbox TTS using device: {self.device}")

        # Model will be lazy-loaded
        self._model: Optional[ChatterboxTTS] = None
        self._model_loading = False

    def _validate_dependencies(self):
        """Validate that all required dependencies are available."""
        missing = []
        backend = _load_backend()

        if not backend["torch"]:
            missing.append("torch")
        if not backend["chatterbox"]:
            missing.append("chatterbox-tts")
        if not backend["torchaudio"]:
            missing.append("torchaudio")

        if missing:
            raise TTSError(
                f"Missing dependencies: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}",
                provider="chatterbox",
                recoverable=False
            )

    def _get_device(self) -> str:
        """Determine the best available device (GPU/CPU)."""
        if not _load_backend()["torch"]:
            return "cpu"

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA available: {device_name} ({vram_gb:.1f} GB VRAM)")
            return "cuda"

        logger.warning("CUDA not available, falling back to CPU (will be slow)")
        return "cpu"

    async def _load_model(self):
        """Lazy load the Chatterbox model."""
        if self._model is not None:
            return

        if self._model_loading:
            # Wait for model to finish loading
            while self._model_loading:
                await asyncio.sleep(0.1)
            return

        self._model_loading = True
        try:
            logger.info("Loading Chatterbox TTS model...")

            # Load model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: ChatterboxTTS.from_pretrained(device=self.device)
            )

            logger.info("Chatterbox TTS model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Chatterbox model: {e}")
            raise TTSError(
                f"Failed to load Chatterbox model: {e}",
                provider=self.name,
                recoverable=False
            )
        finally:
            self._model_loading = False

    @property
    def name(self) -> str:
        return "chatterbox"

    @property
    def supports_streaming(self) -> bool:
        # Chatterbox generates full audio at once, no streaming support
        return False

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used."""
        return self.device == "cuda"

    @property
    def device_info(self) -> dict:
        """Get information about the current device."""
        info = {
            "device": self.device,
            "gpu_available": torch.cuda.is_available() if _load_backend()["torch"] else False,
        }
        if info["gpu_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["vram_used_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
        return info

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
            voice: Language code (e.g., "en", "fr") or path to voice prompt audio
            rate: Speed adjustment (not directly supported, ignored)
            volume: Volume adjustment (not directly supported, ignored)
            pitch: Pitch adjustment (not directly supported, ignored)

        Returns:
            WAV audio bytes
        """
        # Sanitize text to prevent CUDA index errors
        text = sanitize_text_for_tts(text)

        if not text:
            raise TTSError("Cannot synthesize empty text", provider=self.name)

        # Validate text length
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(
                f"Text too long ({len(text)} chars), truncating to {MAX_TEXT_LENGTH}. "
                f"Use audio_processor chunking for longer texts."
            )
            text = text[:MAX_TEXT_LENGTH]

        # Load model if needed
        await self._load_model()

        try:
            # Determine voice prompt path
            audio_prompt = self._resolve_voice_prompt(voice)

            # Run synthesis in executor
            loop = asyncio.get_event_loop()
            wav_tensor = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                audio_prompt
            )

            # Convert tensor to bytes
            audio_bytes = self._tensor_to_wav_bytes(wav_tensor)

            return audio_bytes

        except RuntimeError as e:
            error_str = str(e)
            # Handle CUDA errors specifically
            if "CUDA" in error_str or "device-side assert" in error_str:
                logger.error(f"CUDA error during synthesis: {e}")
                self._clear_cuda_state()
                raise TTSError(
                    f"CUDA error during synthesis. Text may contain unsupported characters. "
                    f"Original error: {e}",
                    provider=self.name,
                    recoverable=True
                )
            raise TTSError(f"Synthesis failed: {e}", provider=self.name, recoverable=True)

        except Exception as e:
            logger.error(f"Chatterbox synthesis failed: {e}")
            raise TTSError(f"Synthesis failed: {e}", provider=self.name, recoverable=True)

    def _clear_cuda_state(self):
        """Clear CUDA state after an error to allow recovery."""
        if _load_backend()["torch"] and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("CUDA state cleared after error")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA state: {e}")

    def _resolve_voice_prompt(self, voice: str) -> Optional[str]:
        """
        Resolve voice parameter to audio prompt path.

        Args:
            voice: Language code or path to audio file

        Returns:
            Path to audio file for voice cloning, or None
        """
        # Check if voice is a file path
        if voice and os.path.isfile(voice):
            return voice

        # Check if instance has a voice prompt path
        if self.voice_prompt_path and os.path.isfile(self.voice_prompt_path):
            return self.voice_prompt_path

        # No voice cloning - use default model voice
        return None

    def _synthesize_sync(self, text: str, audio_prompt: Optional[str]) -> 'torch.Tensor':
        """
        Synchronous synthesis (runs in executor).

        Args:
            text: Text to synthesize
            audio_prompt: Optional path to voice prompt audio

        Returns:
            Audio tensor
        """
        if audio_prompt:
            logger.debug(f"Using voice cloning from: {audio_prompt}")
            wav = self._model.generate(
                text,
                audio_prompt_path=audio_prompt,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight
            )
        else:
            logger.debug("Using default model voice")
            wav = self._model.generate(
                text,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight
            )

        return wav

    def _tensor_to_wav_bytes(self, wav_tensor: 'torch.Tensor') -> bytes:
        """
        Convert audio tensor to WAV bytes.

        Args:
            wav_tensor: Audio tensor from Chatterbox

        Returns:
            WAV file bytes
        """
        buffer = io.BytesIO()

        # Chatterbox outputs at 24kHz
        sample_rate = self._model.sr

        # Ensure tensor is on CPU and has correct shape
        wav_cpu = wav_tensor.cpu()
        if wav_cpu.dim() == 1:
            wav_cpu = wav_cpu.unsqueeze(0)

        torchaudio.save(buffer, wav_cpu, sample_rate, format="wav")
        buffer.seek(0)

        return buffer.read()

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
            output_path: Path to save the audio file (WAV)
            voice: Language code or path to voice prompt
            rate: Speed adjustment (ignored)
            volume: Volume adjustment (ignored)
            pitch: Pitch adjustment (ignored)

        Returns:
            TTSResult with success status
        """
        # Sanitize text to prevent CUDA index errors
        text = sanitize_text_for_tts(text)

        if not text:
            return TTSResult(
                success=False,
                error_message="Cannot synthesize empty text (after sanitization)"
            )

        # Validate text length
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(
                f"Text too long ({len(text)} chars), truncating to {MAX_TEXT_LENGTH}. "
                f"Use audio_processor chunking for longer texts."
            )
            text = text[:MAX_TEXT_LENGTH]

        try:
            # Load model if needed
            await self._load_model()

            # Determine voice prompt path
            audio_prompt = self._resolve_voice_prompt(voice)

            # Run synthesis in executor
            loop = asyncio.get_event_loop()
            wav_tensor = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                audio_prompt
            )

            # Save to file
            await loop.run_in_executor(
                None,
                self._save_audio,
                wav_tensor,
                output_path
            )

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

        except RuntimeError as e:
            error_str = str(e)
            # Handle CUDA errors specifically
            if "CUDA" in error_str or "device-side assert" in error_str:
                logger.error(f"CUDA error during synthesis to file: {e}")
                self._clear_cuda_state()
                return TTSResult(
                    success=False,
                    error_message=f"CUDA error: Text may contain unsupported characters. Try shorter text or different content."
                )
            logger.error(f"Chatterbox synthesis to file failed: {e}")
            return TTSResult(
                success=False,
                error_message=str(e)
            )

        except Exception as e:
            logger.error(f"Chatterbox synthesis to file failed: {e}")
            return TTSResult(
                success=False,
                error_message=str(e)
            )

    def _save_audio(self, wav_tensor: 'torch.Tensor', output_path: str):
        """
        Save audio tensor to file.

        Args:
            wav_tensor: Audio tensor
            output_path: Destination path
        """
        sample_rate = self._model.sr

        wav_cpu = wav_tensor.cpu()
        if wav_cpu.dim() == 1:
            wav_cpu = wav_cpu.unsqueeze(0)

        torchaudio.save(output_path, wav_cpu, sample_rate)
        logger.debug(f"Audio saved to: {output_path}")

    async def get_available_voices(self, language_filter: Optional[str] = None) -> List[VoiceInfo]:
        """
        Get list of supported languages as "voices".

        Chatterbox supports 23 languages. When using voice cloning,
        any of these languages can be used with the cloned voice.

        Args:
            language_filter: Optional language code to filter (e.g., "zh", "en")

        Returns:
            List of VoiceInfo objects representing supported languages
        """
        voices = []

        for code, name in CHATTERBOX_LANGUAGES.items():
            voice = VoiceInfo(
                name=code,
                short_name=code.upper(),
                language=code,
                gender="Neutral",  # Depends on voice prompt used
                locale=name
            )
            voices.append(voice)

        if not language_filter:
            return voices

        # Filter by language code
        filter_lower = language_filter.lower()
        return [
            v for v in voices
            if filter_lower in v.language.lower() or filter_lower in v.locale.lower()
        ]

    async def stream_synthesis(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ) -> AsyncIterator[bytes]:
        """
        Chatterbox doesn't support true streaming.
        Returns full audio as single chunk.
        """
        audio_data = await self.synthesize(text, voice, rate, volume, pitch)
        yield audio_data

    def set_voice_prompt(self, audio_path: str):
        """
        Set the voice prompt for voice cloning.

        Args:
            audio_path: Path to audio file (WAV/MP3)
        """
        if not os.path.isfile(audio_path):
            raise TTSError(f"Voice prompt file not found: {audio_path}", provider=self.name)

        self.voice_prompt_path = audio_path
        logger.info(f"Voice prompt set to: {audio_path}")

    def set_emotion(self, exaggeration: float):
        """
        Set emotion/exaggeration level.

        Args:
            exaggeration: Level from 0.0 (neutral) to 1.0 (expressive)
        """
        self.exaggeration = max(0.0, min(1.0, exaggeration))
        logger.debug(f"Exaggeration set to: {self.exaggeration}")

    def clear_voice_prompt(self):
        """Clear the voice prompt, returning to default voice."""
        self.voice_prompt_path = None
        logger.info("Voice prompt cleared, using default voice")


def create_chatterbox_provider(
    voice_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5
) -> ChatterboxProvider:
    """
    Factory function to create a ChatterboxProvider instance.

    Args:
        voice_prompt_path: Optional path to voice prompt audio
        exaggeration: Emotion exaggeration level (0.0-1.0)
        cfg_weight: Classifier-free guidance weight

    Returns:
        ChatterboxProvider instance
    """
    return ChatterboxProvider(
        voice_prompt_path=voice_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight
    )


def is_chatterbox_available() -> bool:
    """Check if Chatterbox TTS dependencies are available (imports them on first call)."""
    return all(_load_backend().values())


def get_gpu_status() -> dict:
    """
    Get GPU status information.

    Returns:
        Dictionary with GPU availability and details
    """
    backend = _load_backend()
    status = {
        "torch_available": backend["torch"],
        "chatterbox_available": backend["chatterbox"],
        "torchaudio_available": backend["torchaudio"],
        "cuda_available": False,
        "gpu_name": None,
        "vram_total_gb": None,
    }

    if backend["torch"] and torch.cuda.is_available():
        status["cuda_available"] = True
        status["gpu_name"] = torch.cuda.get_device_name(0)
        status["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )

    return status
