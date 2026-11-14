"""
TTS (Text-to-Speech) Configuration Module

Provides centralized configuration for TTS generation including
provider selection, voice settings, and audio encoding options.
"""
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict

from src.utils.lang_normalize import normalize_lang_key

# Default voice mappings by language code
# These are high-quality neural voices from Edge-TTS
# Supports 55+ languages with premium neural voices
DEFAULT_VOICES: Dict[str, str] = {
    # ===== Asian Languages =====
    # Chinese
    "chinese": "zh-CN-XiaoxiaoNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "zh-tw": "zh-TW-HsiaoChenNeural",
    "zh-hk": "zh-HK-HiuMaanNeural",

    # Japanese
    "japanese": "ja-JP-NanamiNeural",
    "ja": "ja-JP-NanamiNeural",

    # Korean
    "korean": "ko-KR-SunHiNeural",
    "ko": "ko-KR-SunHiNeural",

    # Hindi
    "hindi": "hi-IN-SwaraNeural",
    "hi": "hi-IN-SwaraNeural",

    # Vietnamese
    "vietnamese": "vi-VN-HoaiMyNeural",
    "vi": "vi-VN-HoaiMyNeural",

    # Thai
    "thai": "th-TH-PremwadeeNeural",
    "th": "th-TH-PremwadeeNeural",

    # Indonesian
    "indonesian": "id-ID-GadisNeural",
    "id": "id-ID-GadisNeural",

    # Malay
    "malay": "ms-MY-YasminNeural",
    "ms": "ms-MY-YasminNeural",

    # Filipino/Tagalog
    "filipino": "fil-PH-BlessicaNeural",
    "tl": "fil-PH-BlessicaNeural",

    # ===== European Languages =====
    # English
    "english": "en-US-AriaNeural",
    "en": "en-US-AriaNeural",
    "en-us": "en-US-AriaNeural",
    "en-gb": "en-GB-SoniaNeural",
    "en-au": "en-AU-NatashaNeural",
    "en-ca": "en-CA-ClaraNeural",
    "en-in": "en-IN-NeerjaNeural",

    # French
    "french": "fr-FR-DeniseNeural",
    "fr": "fr-FR-DeniseNeural",
    "fr-ca": "fr-CA-SylvieNeural",

    # German
    "german": "de-DE-KatjaNeural",
    "de": "de-DE-KatjaNeural",

    # Spanish
    "spanish": "es-ES-ElviraNeural",
    "es": "es-ES-ElviraNeural",
    "es-mx": "es-MX-DaliaNeural",

    # Italian
    "italian": "it-IT-ElsaNeural",
    "it": "it-IT-ElsaNeural",

    # Portuguese
    "portuguese": "pt-BR-FranciscaNeural",
    "portuguese (brazil)": "pt-BR-FranciscaNeural",
    "portuguese (portugal)": "pt-PT-RaquelNeural",
    "pt": "pt-BR-FranciscaNeural",
    "pt-br": "pt-BR-FranciscaNeural",
    "pt-pt": "pt-PT-RaquelNeural",

    # Russian
    "russian": "ru-RU-SvetlanaNeural",
    "ru": "ru-RU-SvetlanaNeural",

    # Dutch
    "dutch": "nl-NL-ColetteNeural",
    "nl": "nl-NL-ColetteNeural",

    # Polish
    "polish": "pl-PL-AgnieszkaNeural",
    "pl": "pl-PL-AgnieszkaNeural",

    # Swedish
    "swedish": "sv-SE-SofieNeural",
    "sv": "sv-SE-SofieNeural",

    # Norwegian
    "norwegian": "nb-NO-PernilleNeural",
    "no": "nb-NO-PernilleNeural",
    "nb": "nb-NO-PernilleNeural",

    # Danish
    "danish": "da-DK-ChristelNeural",
    "da": "da-DK-ChristelNeural",

    # Finnish
    "finnish": "fi-FI-NooraNeural",
    "fi": "fi-FI-NooraNeural",

    # Greek
    "greek": "el-GR-AthinaNeural",
    "el": "el-GR-AthinaNeural",

    # Czech
    "czech": "cs-CZ-VlastaNeural",
    "cs": "cs-CZ-VlastaNeural",

    # Hungarian
    "hungarian": "hu-HU-NoemiNeural",
    "hu": "hu-HU-NoemiNeural",

    # Romanian
    "romanian": "ro-RO-AlinaNeural",
    "ro": "ro-RO-AlinaNeural",

    # Turkish
    "turkish": "tr-TR-EmelNeural",
    "tr": "tr-TR-EmelNeural",

    # Ukrainian
    "ukrainian": "uk-UA-PolinaNeural",
    "uk": "uk-UA-PolinaNeural",

    # Bulgarian
    "bulgarian": "bg-BG-KalinaNeural",
    "bg": "bg-BG-KalinaNeural",

    # Croatian
    "croatian": "hr-HR-GabrijelaNeural",
    "hr": "hr-HR-GabrijelaNeural",

    # Slovak
    "slovak": "sk-SK-ViktoriaNeural",
    "sk": "sk-SK-ViktoriaNeural",

    # Slovenian
    "slovenian": "sl-SI-PetraNeural",
    "sl": "sl-SI-PetraNeural",

    # Lithuanian
    "lithuanian": "lt-LT-OnaNeural",
    "lt": "lt-LT-OnaNeural",

    # Latvian
    "latvian": "lv-LV-EveritaNeural",
    "lv": "lv-LV-EveritaNeural",

    # Estonian
    "estonian": "et-EE-AnuNeural",
    "et": "et-EE-AnuNeural",

    # ===== Semitic / RTL Languages =====
    # Arabic
    "arabic": "ar-SA-ZariyahNeural",
    "ar": "ar-SA-ZariyahNeural",
    "ar-eg": "ar-EG-SalmaNeural",

    # Hebrew
    "hebrew": "he-IL-HilaNeural",
    "he": "he-IL-HilaNeural",

    # Persian/Farsi
    "persian": "fa-IR-DilaraNeural",
    "fa": "fa-IR-DilaraNeural",

    # ===== Other Languages =====
    # Bengali
    "bengali": "bn-BD-NabanitaNeural",
    "bn": "bn-BD-NabanitaNeural",

    # Tamil
    "tamil": "ta-IN-PallaviNeural",
    "ta": "ta-IN-PallaviNeural",

    # Telugu
    "telugu": "te-IN-ShrutiNeural",
    "te": "te-IN-ShrutiNeural",

    # Urdu
    "urdu": "ur-PK-UzmaNeural",
    "ur": "ur-PK-UzmaNeural",
}

# Load TTS settings from environment
TTS_ENABLED = os.getenv('TTS_ENABLED', 'false').lower() == 'true'
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'edge-tts')  # edge-tts or chatterbox
TTS_VOICE = os.getenv('TTS_VOICE', '')  # Empty = auto-select based on language
TTS_RATE = os.getenv('TTS_RATE', '+0%')  # Speed adjustment: -50% to +100%
TTS_VOLUME = os.getenv('TTS_VOLUME', '+0%')  # Volume adjustment: -50% to +50%
TTS_PITCH = os.getenv('TTS_PITCH', '+0Hz')  # Pitch adjustment

# Audio encoding settings
TTS_OUTPUT_FORMAT = os.getenv('TTS_OUTPUT_FORMAT', 'opus')  # opus, mp3, wav
TTS_BITRATE = os.getenv('TTS_BITRATE', '64k')  # For opus/mp3 encoding
TTS_SAMPLE_RATE = int(os.getenv('TTS_SAMPLE_RATE', '24000'))  # Hz

# Chunking settings for TTS
TTS_CHUNK_SIZE = int(os.getenv('TTS_CHUNK_SIZE', '5000'))  # Max chars per TTS chunk
TTS_PAUSE_BETWEEN_CHUNKS = float(os.getenv('TTS_PAUSE_BETWEEN_CHUNKS', '0.5'))  # Seconds

# Chatterbox-specific settings (GPU local TTS)
TTS_VOICE_PROMPT_PATH = os.getenv('TTS_VOICE_PROMPT_PATH', '')  # Audio file for voice cloning
TTS_EXAGGERATION = float(os.getenv('TTS_EXAGGERATION', '0.5'))  # Emotion level 0.0-1.0
TTS_CFG_WEIGHT = float(os.getenv('TTS_CFG_WEIGHT', '0.5'))  # Classifier-free guidance weight

# Chatterbox supported languages (23 languages)
CHATTERBOX_VOICES: Dict[str, str] = {
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


def get_voice_for_language(language: str) -> str:
    """
    Get the default voice for a given language.

    Args:
        language: Language name or code (e.g., 'Chinese', 'zh', 'en-US')

    Returns:
        Voice name for Edge-TTS, or empty string if not found
    """
    normalized = language.lower().strip()
    return DEFAULT_VOICES.get(normalized, '')


@dataclass
class TTSConfig:
    """Configuration for TTS generation"""

    # Core settings
    enabled: bool = TTS_ENABLED
    provider: str = TTS_PROVIDER

    # Voice settings
    voice: str = TTS_VOICE
    rate: str = TTS_RATE
    volume: str = TTS_VOLUME
    pitch: str = TTS_PITCH

    # Output settings
    output_format: str = TTS_OUTPUT_FORMAT
    bitrate: str = TTS_BITRATE
    sample_rate: int = TTS_SAMPLE_RATE

    # Processing settings
    chunk_size: int = TTS_CHUNK_SIZE
    pause_between_chunks: float = TTS_PAUSE_BETWEEN_CHUNKS

    # Chatterbox-specific settings
    voice_prompt_path: str = TTS_VOICE_PROMPT_PATH
    exaggeration: float = TTS_EXAGGERATION
    cfg_weight: float = TTS_CFG_WEIGHT

    # Runtime settings (set during execution)
    target_language: str = ''
    output_path: Optional[str] = None

    @classmethod
    def from_cli_args(cls, args) -> 'TTSConfig':
        """Create config from CLI arguments"""
        config = cls(
            enabled=getattr(args, 'tts', False),
            provider=getattr(args, 'tts_provider', None) or TTS_PROVIDER,
            voice=getattr(args, 'tts_voice', '') or TTS_VOICE,
            rate=getattr(args, 'tts_rate', None) or TTS_RATE,
            bitrate=getattr(args, 'tts_bitrate', None) or TTS_BITRATE,
            output_format=getattr(args, 'tts_format', None) or TTS_OUTPUT_FORMAT,
            # Chatterbox-specific
            voice_prompt_path=getattr(args, 'tts_voice_prompt', '') or TTS_VOICE_PROMPT_PATH,
            exaggeration=getattr(args, 'tts_exaggeration', None) or TTS_EXAGGERATION,
            cfg_weight=getattr(args, 'tts_cfg_weight', None) or TTS_CFG_WEIGHT,
        )
        return config

    @classmethod
    def from_env(cls) -> 'TTSConfig':
        """Create config from environment variables only"""
        return cls()

    @classmethod
    def from_web_request(cls, request_data: dict) -> 'TTSConfig':
        """Create config from web request data"""
        return cls(
            enabled=request_data.get('tts_enabled', False),
            provider=request_data.get('tts_provider', TTS_PROVIDER),
            voice=request_data.get('tts_voice', '') or TTS_VOICE,
            rate=request_data.get('tts_rate', TTS_RATE),
            volume=request_data.get('tts_volume', TTS_VOLUME),
            bitrate=request_data.get('tts_bitrate', TTS_BITRATE),
            output_format=request_data.get('tts_format', TTS_OUTPUT_FORMAT),
            # Chatterbox-specific
            voice_prompt_path=request_data.get('tts_voice_prompt_path', '') or TTS_VOICE_PROMPT_PATH,
            exaggeration=float(request_data.get('tts_exaggeration', TTS_EXAGGERATION)),
            cfg_weight=float(request_data.get('tts_cfg_weight', TTS_CFG_WEIGHT)),
        )

    def get_effective_voice(self, language: str = '') -> str:
        """
        Get the voice to use, auto-selecting if not specified.

        Args:
            language: Target language for auto-selection

        Returns:
            Voice name to use
        """
        if self.voice:
            return self.voice

        lang = language or self.target_language
        if lang:
            auto_voice = get_voice_for_language(lang)
            if auto_voice:
                return auto_voice

        # Fallback to English
        return DEFAULT_VOICES['english']

    def get_output_extension(self) -> str:
        """Get file extension for the output format"""
        format_extensions = {
            'opus': '.opus',
            'mp3': '.mp3',
            'wav': '.wav',
            'ogg': '.ogg',
        }
        return format_extensions.get(self.output_format.lower(), '.opus')

    @classmethod
    def from_dict(cls, data: dict) -> 'TTSConfig':
        """Rebuild an enabled config from ``to_dict()`` output (a job's TTS options).

        Unknown keys are ignored; missing ones keep their defaults.
        """
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in (data or {}).items() if k in known}
        values['enabled'] = True
        return cls(**values)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'enabled': self.enabled,
            'provider': self.provider,
            'voice': self.voice,
            'rate': self.rate,
            'volume': self.volume,
            'pitch': self.pitch,
            'output_format': self.output_format,
            'bitrate': self.bitrate,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'pause_between_chunks': self.pause_between_chunks,
            'target_language': self.target_language,
            # Chatterbox-specific
            'voice_prompt_path': self.voice_prompt_path,
            'exaggeration': self.exaggeration,
            'cfg_weight': self.cfg_weight,
        }

    def get_chatterbox_voice(self, language: str = '') -> str:
        """
        Get the language code for Chatterbox TTS.

        Chatterbox has no regional variants, so both regional codes
        ("pt-br", "pt-pt") and regional names ("Portuguese (Brazil)")
        resolve to the base language it does support ("pt").

        Args:
            language: Target language name or code

        Returns:
            Language code for Chatterbox (e.g., 'en', 'fr', 'zh')
        """
        lang = language.lower().strip() if language else self.target_language.lower().strip()

        # Direct match
        if lang in CHATTERBOX_VOICES:
            return lang

        # Regional code: "pt-br" / "en_US" -> "pt" / "en"
        base_code = re.split(r'[-_]', lang, maxsplit=1)[0]
        if base_code in CHATTERBOX_VOICES:
            return base_code

        # Try to match by full name, with or without a regional suffix
        base_name = normalize_lang_key(lang)
        for code, name in CHATTERBOX_VOICES.items():
            if name.lower() == base_name:
                return code

        # Fallback to English
        return 'en'
