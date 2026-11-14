"""
Dependency injection container for EPUB translation pipeline.

Provides centralized creation and configuration of translation components.
"""

from dataclasses import dataclass, field
from typing import Optional

from .tag_preservation import TagPreserver
from .html_chunker import HtmlChunker
from .placeholder_validator import PlaceholderValidator


def _configured_max_tokens_per_chunk() -> int:
    """Read MAX_TOKENS_PER_CHUNK when a config is created, not at import time."""
    from src import config
    return config.MAX_TOKENS_PER_CHUNK


@dataclass
class TranslationConfig:
    """Configuration for translation pipeline.

    Attributes:
        max_tokens_per_chunk: Maximum tokens per chunk
        max_retries: Maximum translation retry attempts
        enable_debug: Enable debug logging
        placeholder_prefix: Placeholder prefix (e.g., "[id")
        placeholder_suffix: Placeholder suffix (e.g., "]")
    """
    max_tokens_per_chunk: int = field(default_factory=_configured_max_tokens_per_chunk)
    max_retries: int = 3
    enable_debug: bool = False
    placeholder_prefix: str = "[id"
    placeholder_suffix: str = "]"

    def __post_init__(self):
        """Validate configuration."""
        if self.max_tokens_per_chunk < 50:
            raise ValueError("max_tokens_per_chunk must be >= 50")
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")


class TranslationContainer:
    """Dependency injection container for translation components."""

    def __init__(self, config: Optional[TranslationConfig] = None):
        """Initialize container with configuration.

        Args:
            config: Translation configuration (uses defaults if None)
        """
        self.config = config or TranslationConfig()

        # Initialize components (lazy loading)
        self._tag_preserver = None
        self._chunker = None
        self._validator = None

    @property
    def tag_preserver(self) -> TagPreserver:
        """Get or create TagPreserver instance."""
        if self._tag_preserver is None:
            self._tag_preserver = TagPreserver()
        return self._tag_preserver

    @property
    def chunker(self) -> HtmlChunker:
        """Get or create HtmlChunker instance."""
        if self._chunker is None:
            self._chunker = HtmlChunker(
                max_tokens=self.config.max_tokens_per_chunk
            )
        return self._chunker

    def chunker_for(self, max_tokens: Optional[int] = None) -> HtmlChunker:
        """Get an HtmlChunker sized for ``max_tokens``.

        The caller's chunk size (the user's setting for this job) always wins
        over the container's default configuration. ``None`` means "no
        preference" and returns the container's own chunker.

        Args:
            max_tokens: Maximum tokens per chunk requested by the caller

        Returns:
            The shared chunker when the size matches the configuration,
            otherwise a new chunker with the requested size.
        """
        if max_tokens is None or max_tokens == self.config.max_tokens_per_chunk:
            return self.chunker
        return HtmlChunker(max_tokens=max_tokens)

    @property
    def validator(self) -> PlaceholderValidator:
        """Get or create PlaceholderValidator instance."""
        if self._validator is None:
            self._validator = PlaceholderValidator()
        return self._validator

    def create_translator(self, llm_client, context_manager):
        """Create configured translator instance.

        Args:
            llm_client: LLM client instance
            context_manager: Context window manager

        Returns:
            Configured translator ready for use

        Note:
            To be implemented in Phase 3 with new architecture.
        """
        # Placeholder for Phase 3 implementation
        raise NotImplementedError(
            "create_translator() will be implemented in Phase 3 "
            "with the new pipeline architecture"
        )
