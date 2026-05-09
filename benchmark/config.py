"""
Benchmark configuration module.

Defines configuration settings for the benchmark system including:
- Ollama/OpenAI-compatible settings for translation
- OpenRouter settings for evaluation
- File paths and defaults
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# Default languages for quick benchmark (representative sample)
DEFAULT_QUICK_LANGUAGES = [
    # European
    "fr",       # French
    "de",       # German
    "es",       # Spanish
    "it",       # Italian
    "pt",       # Portuguese
    "pl",       # Polish
    # Asian
    "zh-Hans",  # Chinese (Simplified)
    "zh-Hant",  # Chinese (Traditional)
    "ja",       # Japanese
    "ko",       # Korean
    "vi",       # Vietnamese
    "th",       # Thai
    # South Asian
    "hi",       # Hindi
    "bn",       # Bengali
    "ta",       # Tamil
    # Cyrillic
    "ru",       # Russian
    "uk",       # Ukrainian
    # Semitic (RTL)
    "ar",       # Arabic
    "he",       # Hebrew
]

# Default evaluator model
DEFAULT_EVALUATOR_MODEL = "google/gemini-3-flash-preview"

# Default evaluator provider ("openrouter" or "poe")
DEFAULT_EVALUATOR_PROVIDER = "poe"

# Default POE model for evaluation
# Available models: gemini-3.1-flash-lite, gemini-3.1-pro, Claude-Sonnet-4, GPT-4o, etc.
DEFAULT_POE_EVALUATOR_MODEL = "gemini-3.1-flash-lite"

# Score thresholds for visual indicators
SCORE_THRESHOLDS = {
    "excellent": 9,   # 🟢 9-10
    "good": 7,        # 🟡 7-8
    "acceptable": 5,  # 🟠 5-6
    "poor": 3,        # 🔴 3-4
    # Below 3: ⚫ Failed (1-2)
}


@dataclass
class OllamaConfig:
    """Configuration for Ollama translation provider."""

    endpoint: str = field(
        default_factory=lambda: os.getenv("API_ENDPOINT", "http://ai_server.mds.com:11434/api/generate")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "mistral-small:24b")
    )
    num_ctx: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "900"))
    )


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter evaluation provider."""

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    default_model: str = DEFAULT_EVALUATOR_MODEL
    timeout: int = 120

    # Request headers
    site_url: str = "https://github.com/yourusername/TranslateBookWithLLM"
    site_name: str = "TranslateBookWithLLM Benchmark"


@dataclass
class OpenAICompatibleConfig:
    """Configuration for OpenAI-compatible translation provider."""

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    endpoint: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_API_ENDPOINT",
            "https://api.openai.com/v1/chat/completions"
        )
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    context_window: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_NUM_CTX", os.getenv("OLLAMA_NUM_CTX", "2048")))
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_REQUEST_TIMEOUT", os.getenv("REQUEST_TIMEOUT", "900")))
    )


@dataclass
class PoeConfig:
    """Configuration for Poe evaluation provider."""

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("POE_API_KEY")
    )
    endpoint: str = "https://api.poe.com/v1/chat/completions"
    default_model: str = DEFAULT_POE_EVALUATOR_MODEL
    timeout: int = 120


@dataclass
class PathConfig:
    """Configuration for file paths."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    # GitHub wiki repository URL
    wiki_repo_url: str = field(
        default_factory=lambda: os.getenv(
            "WIKI_REPO_URL",
            "https://github.com/hydropix/TranslateBookWithLLM.wiki.git"
        )
    )

    @property
    def languages_file(self) -> Path:
        return self.base_dir / "languages.yaml"

    @property
    def reference_texts_file(self) -> Path:
        return self.base_dir / "reference_texts.yaml"

    @property
    def results_dir(self) -> Path:
        return self.base_dir.parent / "benchmark_results"

    @property
    def wiki_output_dir(self) -> Path:
        return self.base_dir.parent / "wiki"

    @property
    def templates_dir(self) -> Path:
        return self.base_dir / "wiki" / "templates"

    @property
    def wiki_clone_dir(self) -> Path:
        return self.base_dir.parent / ".wiki_repo"


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration aggregating all sub-configs."""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAICompatibleConfig = field(default_factory=OpenAICompatibleConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    poe: PoeConfig = field(default_factory=PoeConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Benchmark settings
    source_language: str = "English"
    quick_languages: list = field(default_factory=lambda: DEFAULT_QUICK_LANGUAGES.copy())

    # Translation provider ("ollama", "openai", or "openrouter")
    translation_provider: str = "ollama"

    # Evaluator provider ("openrouter" or "poe")
    evaluator_provider: str = DEFAULT_EVALUATOR_PROVIDER

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0

    @classmethod
    def from_env(cls) -> "BenchmarkConfig":
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_cli_args(
        cls,
        openrouter_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        openai_endpoint: Optional[str] = None,
        evaluator_model: Optional[str] = None,
        ollama_endpoint: Optional[str] = None,
        translation_provider: Optional[str] = None,
        evaluator_provider: Optional[str] = None,
        poe_key: Optional[str] = None,
        **kwargs
    ) -> "BenchmarkConfig":
        """Create configuration from CLI arguments with env fallbacks."""
        config = cls()

        if openrouter_key:
            config.openrouter.api_key = openrouter_key

        if openai_key:
            config.openai.api_key = openai_key

        if poe_key:
            config.poe.api_key = poe_key

        if evaluator_model:
            config.openrouter.default_model = evaluator_model
            config.poe.default_model = evaluator_model

        if ollama_endpoint:
            config.ollama.endpoint = ollama_endpoint

        if openai_endpoint:
            config.openai.endpoint = openai_endpoint

        if translation_provider:
            config.translation_provider = translation_provider.lower()

        if evaluator_provider:
            config.evaluator_provider = evaluator_provider.lower()

        return config

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate evaluator provider
        if self.evaluator_provider not in ("openrouter", "poe"):
            errors.append(
                f"Invalid evaluator provider: {self.evaluator_provider}. "
                "Must be 'openrouter' or 'poe'"
            )

        # Check API key for evaluation provider
        if self.evaluator_provider == "poe":
            if not self.poe.api_key:
                errors.append(
                    "Poe API key not configured. Required for evaluation. "
                    "Set POE_API_KEY in .env or use --poe-key"
                )
        else:  # openrouter
            if not self.openrouter.api_key:
                errors.append(
                    "OpenRouter API key not configured. Required for evaluation. "
                    "Set OPENROUTER_API_KEY in .env or use --openrouter-key"
                )

        # Check translation provider API key if needed
        if self.translation_provider == "openrouter" and not self.openrouter.api_key:
            errors.append(
                "OpenRouter API key not configured. Required for translation. "
                "Set OPENROUTER_API_KEY in .env or use --openrouter-key"
            )

        if self.translation_provider == "openai" and not self.openai.endpoint:
            errors.append(
                "OpenAI-compatible endpoint not configured. Required for translation. "
                "Set OPENAI_API_ENDPOINT in .env or use --openai-endpoint"
            )

        if self.translation_provider == "poe" and not self.poe.api_key:
            errors.append(
                "Poe API key not configured. Required for translation. "
                "Set POE_API_KEY in .env or use --poe-key"
            )

        # Accept either the split layout or the legacy monolithic YAMLs.
        split_lang_dir = self.paths.base_dir / "data" / "languages"
        if not split_lang_dir.is_dir() and not self.paths.languages_file.exists():
            errors.append(
                f"Languages data not found at {split_lang_dir} or {self.paths.languages_file}"
            )

        split_texts_dir = self.paths.base_dir / "data" / "reference_texts"
        if not split_texts_dir.is_dir() and not self.paths.reference_texts_file.exists():
            errors.append(
                f"Reference texts not found at {split_texts_dir} or {self.paths.reference_texts_file}"
            )

        # Validate translation provider
        if self.translation_provider not in ("ollama", "openai", "openrouter", "poe"):
            errors.append(
                f"Invalid translation provider: {self.translation_provider}. "
                "Must be 'ollama', 'openai', 'openrouter', or 'poe'"
            )

        return errors


def get_score_indicator(score: float) -> str:
    """Get visual indicator emoji for a score."""
    if score >= SCORE_THRESHOLDS["excellent"]:
        return "🟢"
    elif score >= SCORE_THRESHOLDS["good"]:
        return "🟡"
    elif score >= SCORE_THRESHOLDS["acceptable"]:
        return "🟠"
    elif score >= SCORE_THRESHOLDS["poor"]:
        return "🔴"
    else:
        return "⚫"


def get_score_label(score: float) -> str:
    """Get text label for a score."""
    if score >= SCORE_THRESHOLDS["excellent"]:
        return "Excellent"
    elif score >= SCORE_THRESHOLDS["good"]:
        return "Good"
    elif score >= SCORE_THRESHOLDS["acceptable"]:
        return "Acceptable"
    elif score >= SCORE_THRESHOLDS["poor"]:
        return "Poor"
    else:
        return "Failed"
