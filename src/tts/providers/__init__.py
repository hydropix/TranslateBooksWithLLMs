"""
TTS Providers Package

Factory functions for creating TTS provider instances.
"""
from .base import TTSProvider, TTSResult, VoiceInfo, TTSError, ProgressCallback
from .edge_tts import EdgeTTSProvider, create_edge_tts_provider

# The Chatterbox names are resolved on first access (PEP 562) so that importing
# this package never loads the Chatterbox provider module.
_CHATTERBOX_EXPORTS = {
    'ChatterboxProvider': 'ChatterboxProvider',
    'create_chatterbox_provider': 'create_chatterbox_provider',
    'is_chatterbox_available': 'is_chatterbox_available',
    'get_gpu_status': 'get_gpu_status',
    'CHATTERBOX_LANGUAGES': 'CHATTERBOX_LANGUAGES',
    'CHATTERBOX_MAX_TEXT_LENGTH': 'MAX_TEXT_LENGTH',
    'sanitize_text_for_tts': 'sanitize_text_for_tts',
}


def __getattr__(name: str):
    if name in _CHATTERBOX_EXPORTS:
        from . import chatterbox_tts
        value = getattr(chatterbox_tts, _CHATTERBOX_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Base classes
    'TTSProvider',
    'TTSResult',
    'VoiceInfo',
    'TTSError',
    'ProgressCallback',
    # Edge-TTS
    'EdgeTTSProvider',
    'create_edge_tts_provider',
    # Chatterbox TTS
    'ChatterboxProvider',
    'create_chatterbox_provider',
    'is_chatterbox_available',
    'get_gpu_status',
    'CHATTERBOX_LANGUAGES',
    'CHATTERBOX_MAX_TEXT_LENGTH',
    'sanitize_text_for_tts',
]


def create_provider(provider_name: str = "edge-tts", **kwargs) -> TTSProvider:
    """
    Factory function to create a TTS provider by name.

    Args:
        provider_name: Name of the provider ("edge-tts", "chatterbox")
        **kwargs: Additional arguments passed to provider constructor
            - For chatterbox: voice_prompt_path, exaggeration, cfg_weight

    Returns:
        TTSProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider_name == "edge-tts":
        return create_edge_tts_provider()

    elif provider_name == "chatterbox":
        from .chatterbox_tts import create_chatterbox_provider
        return create_chatterbox_provider(
            voice_prompt_path=kwargs.get('voice_prompt_path'),
            exaggeration=kwargs.get('exaggeration', 0.5),
            cfg_weight=kwargs.get('cfg_weight', 0.5)
        )

    else:
        available = "edge-tts, chatterbox"
        raise ValueError(f"Unknown TTS provider: {provider_name}. Available: {available}")
