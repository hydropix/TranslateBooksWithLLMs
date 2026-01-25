"""
Centralized configuration class
"""
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Setup debug logger for configuration
_config_logger = logging.getLogger('config')

# Check for DEBUG_MODE early (before .env is loaded, check environment)
_debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
if _debug_mode:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _config_logger.setLevel(logging.DEBUG)
    _config_logger.debug("🔍 DEBUG_MODE enabled - verbose logging active")

# Get config directory (current working directory)
_config_dir = Path.cwd()

# Check if .env file exists and provide helpful guidance
_env_file = _config_dir / '.env'
_env_example = _config_dir / '.env.example'
_env_exists = _env_file.exists()
_cwd = Path.cwd()

if _debug_mode:
    _config_logger.debug(f"📁 Current working directory: {_cwd}")
    _config_logger.debug(f"📁 Looking for .env at: {_env_file.absolute()}")
    _config_logger.debug(f"📁 .env exists: {_env_exists}")

if not _env_exists:
    # Check if running as PyInstaller executable
    _is_frozen = getattr(sys, 'frozen', False)

    if not _is_frozen:
        # Only show the interactive prompt when NOT running as executable
        print("\n" + "="*70)
        print("⚠️  WARNING: .env configuration file not found")
        print("="*70)
        print("\nThe application will run with default settings, but you may need to")
        print("configure it for your specific setup.\n")

        if _env_example.exists():
            print("📋 QUICK SETUP:")
            print(f"   1. Copy the template: copy .env.example .env")
            print(f"   2. Edit .env to match your configuration")
            print(f"   3. Restart the application\n")
        else:
            print("📋 MANUAL SETUP:")
            print(f"   1. Create a .env file in: {Path.cwd()}")
            print(f"   2. Add your configuration (see documentation)")
            print(f"   3. Restart the application\n")

        print("🔧 DEFAULT SETTINGS BEING USED:")
        print(f"   • API Endpoint: http://localhost:11434/api/generate")
        print(f"   • LLM Provider: ollama")
        print(f"   • Model: qwen3:14b")
        print(f"   • Port: 5000")
        print(f"\n💡 TIP: If using a remote server or different provider, you MUST")
        print(f"   create a .env file with the correct settings.\n")
        print("="*70)
        print("Press Ctrl+C to stop and configure, or wait 5 seconds to continue...")
        print("="*70 + "\n")

        # Give user time to read and react
        import time
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n\n⏹️  Startup cancelled by user. Please configure .env and try again.\n")
            sys.exit(0)
    else:
        # Running as executable - silently use defaults
        if _debug_mode:
            _config_logger.debug("⚠️  .env not found, using defaults (executable mode)")

# Load .env file if it exists
_dotenv_result = load_dotenv(_env_file)
if _debug_mode:
    _config_logger.debug(f"📁 load_dotenv() returned: {_dotenv_result}")
    _config_logger.debug(f"📁 Loaded .env from: {_env_file.absolute()}")

# Load from environment variables with defaults
API_ENDPOINT = os.getenv('API_ENDPOINT', 'http://localhost:11434/api/generate')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'qwen3:14b')
PORT = int(os.getenv('PORT', '5000'))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '900'))
OLLAMA_NUM_CTX = int(os.getenv('OLLAMA_NUM_CTX', '4096'))

# =============================================================================
# THINKING MODEL CONFIGURATION
# =============================================================================
# Models are classified based on their behavior with the 'think' parameter:
#
# 1. UNCONTROLLABLE: Models that think even with think=false (need WARNING)
# 2. CONTROLLABLE: Models that respect think=false (no warning needed)
# 3. STANDARD: Models that don't think at all (no think param needed)
#
# Auto-detection at runtime will classify models by testing with think=true/false

# Models that CANNOT be prevented from thinking - show WARNING to user
# These models either ignore think=false or don't support the param but still think
UNCONTROLLABLE_THINKING_MODELS = [
    "qwen3:30b",      # Qwen3 30B ignores think=false (tested)
    "qwen3-vl",       # Qwen3 Vision models ignore think=false
    "phi4-reasoning", # Phi4 reasoning doesn't support think param but always thinks
    "deepseek-r1",    # DeepSeek R1 reasoning model
    "qwq",            # Qwen QwQ reasoning model
    "marco-o1",       # Alibaba reasoning model
    "exaone-deep",    # LG reasoning model
]

# Models that respect think=false - controllable, no warning needed
CONTROLLABLE_THINKING_MODELS = [
    "qwen3:8b",       # Respects think=false (tested)
    "qwen3:14b",      # Respects think=false (tested)
    "qwen3:4b",       # Smaller Qwen3 models likely controllable
    "qwen3:1.7b",     # Smaller Qwen3 models likely controllable
    "qwen3:0.6b",     # Smaller Qwen3 models likely controllable
]

# Legacy alias for backward compatibility
THINKING_MODELS = UNCONTROLLABLE_THINKING_MODELS + CONTROLLABLE_THINKING_MODELS
MAX_TRANSLATION_ATTEMPTS = int(os.getenv('MAX_TRANSLATION_ATTEMPTS', '2'))

# Adaptive context optimization settings
# The new strategy starts at a small context and grows as needed based on actual token usage
AUTO_ADJUST_CONTEXT = os.getenv("AUTO_ADJUST_CONTEXT", "true").lower() == "true"
ADAPTIVE_CONTEXT_INITIAL = int(os.getenv("ADAPTIVE_CONTEXT_INITIAL", "2048"))  # Starting context size
ADAPTIVE_CONTEXT_INITIAL_THINKING = int(os.getenv("ADAPTIVE_CONTEXT_INITIAL_THINKING", "6144"))  # Starting context for thinking models (need more space for reasoning)
ADAPTIVE_CONTEXT_STEP = int(os.getenv("ADAPTIVE_CONTEXT_STEP", "2048"))  # Step size for increases
ADAPTIVE_CONTEXT_STABILITY_WINDOW = int(os.getenv("ADAPTIVE_CONTEXT_STABILITY_WINDOW", "5"))  # Chunks to track before reducing

# Repetition loop detection settings
# Thinking models may have natural repetitions in their reasoning, so we use higher thresholds
REPETITION_MIN_PHRASE_LENGTH = int(os.getenv("REPETITION_MIN_PHRASE_LENGTH", "5"))  # Min phrase length to detect
REPETITION_MIN_COUNT = int(os.getenv("REPETITION_MIN_COUNT", "10"))  # Min repetitions for standard models
REPETITION_MIN_COUNT_THINKING = int(os.getenv("REPETITION_MIN_COUNT_THINKING", "15"))  # Min repetitions for thinking models (more lenient)
REPETITION_MIN_COUNT_STREAMING = int(os.getenv("REPETITION_MIN_COUNT_STREAMING", "12"))  # Min repetitions during streaming (early detection)

# Legacy settings (kept for compatibility)
MIN_RECOMMENDED_NUM_CTX = 4096  # Minimum recommended context for chunk_size=25
SAFETY_MARGIN = 1.1  # 10% safety margin for token estimation
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "5"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "100"))

# Token-based chunking configuration
# All file types use token-based chunking with tiktoken for consistent chunk sizes
MAX_TOKENS_PER_CHUNK = int(os.getenv('MAX_TOKENS_PER_CHUNK', '450'))
SOFT_LIMIT_RATIO = float(os.getenv('SOFT_LIMIT_RATIO', '0.8'))

# === Translation Buffer Configuration ===
TRANSLATION_OUTPUT_MULTIPLIER = 2
"""Multiplicateur pour la longueur de sortie estimée (certaines langues cibles
peuvent être 2x plus longues que la source)"""

TRANSLATION_TAG_OVERHEAD = 50
"""Tokens réservés pour les balises XML de traduction (<Translated>...</Translated>)"""

# === Placeholder Validation ===
MAX_PLACEHOLDER_RETRIES = 3
"""Nombre maximum de tentatives de validation des placeholders"""

MAX_PLACEHOLDER_CORRECTION_ATTEMPTS = 2
"""Nombre maximum de tentatives de correction LLM pour les placeholders malformés"""

# === Chunking Limits ===
MIN_CHUNK_SIZE_TOKENS = 50
"""Taille minimale d'un chunk pour éviter la sur-fragmentation"""

# LLM Provider configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')  # 'ollama', 'gemini', 'openai', or 'openrouter'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# OpenRouter configuration (access to 200+ models)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'anthropic/claude-sonnet-4')
OPENROUTER_API_ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'

# Mistral AI configuration
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY', '')
MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
MISTRAL_API_ENDPOINT = os.getenv('MISTRAL_API_ENDPOINT', 'https://api.mistral.ai/v1/chat/completions')

# SRT-specific configuration
SRT_LINES_PER_BLOCK = int(os.getenv('SRT_LINES_PER_BLOCK', '5'))
SRT_MAX_CHARS_PER_BLOCK = int(os.getenv('SRT_MAX_CHARS_PER_BLOCK', '500'))

# Translation Attribution
# This adds a discrete attribution to your translations (metadata for EPUB, footer for TXT, comment for SRT)
# Please consider keeping this enabled to support the project and help others discover this free tool!
# The attribution is non-intrusive and placed at the end of files. Thank you for your support!
ATTRIBUTION_ENABLED = os.getenv('ATTRIBUTION_ENABLED', os.getenv('SIGNATURE_ENABLED', 'true')).lower() == 'true'
GENERATOR_NAME = "TranslateBook with LLM (TBL)"
GENERATOR_SOURCE = "https://github.com/hydropix/TranslateBookWithLLM"
METADATA_VERSION = "1.0"

# Default languages from environment (optional)
# Source language: Auto-detected from file content (langdetect)
# Target language: Auto-detected from browser language in UI
DEFAULT_SOURCE_LANGUAGE = os.getenv('DEFAULT_SOURCE_LANGUAGE', '')  # Empty = auto-detect
DEFAULT_TARGET_LANGUAGE = os.getenv('DEFAULT_TARGET_LANGUAGE', '')  # Empty = use browser language

# ============================================================================
# PROMPT OPTIONS CONFIGURATION
# ============================================================================
# These options control which optional sections are included in the system prompt.
# Each option can be enabled/disabled via the web interface or CLI.

# Technical Content Preservation (always enabled)
# Automatically detects and preserves code, paths, URLs, formulas, etc.
# This is always active as it has no negative impact on literary texts.
PROMPT_PRESERVE_TECHNICAL_CONTENT = True

# Server configuration
HOST = os.getenv('HOST', '127.0.0.1')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'translated_files')

# Debug mode (reload after .env is loaded)
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Log loaded configuration in debug mode
if DEBUG_MODE or _debug_mode:
    _config_logger.setLevel(logging.DEBUG)
    _config_logger.debug("="*60)
    _config_logger.debug("📋 LOADED CONFIGURATION VALUES:")
    _config_logger.debug("="*60)
    _config_logger.debug(f"   API_ENDPOINT: {API_ENDPOINT}")
    _config_logger.debug(f"   DEFAULT_MODEL: {DEFAULT_MODEL}")
    _config_logger.debug(f"   LLM_PROVIDER: {LLM_PROVIDER}")
    _config_logger.debug(f"   PORT: {PORT}")
    _config_logger.debug(f"   HOST: {HOST}")
    _config_logger.debug(f"   DEFAULT_SOURCE_LANGUAGE: {DEFAULT_SOURCE_LANGUAGE}")
    _config_logger.debug(f"   DEFAULT_TARGET_LANGUAGE: {DEFAULT_TARGET_LANGUAGE}")
    _config_logger.debug(f"   OLLAMA_NUM_CTX: {OLLAMA_NUM_CTX}")
    _config_logger.debug(f"   REQUEST_TIMEOUT: {REQUEST_TIMEOUT}")
    _config_logger.debug(f"   GEMINI_API_KEY: {'***' + GEMINI_API_KEY[-4:] if GEMINI_API_KEY else '(not set)'}")
    _config_logger.debug(f"   OPENAI_API_KEY: {'***' + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else '(not set)'}")
    _config_logger.debug(f"   OPENROUTER_API_KEY: {'***' + OPENROUTER_API_KEY[-4:] if OPENROUTER_API_KEY else '(not set)'}")
    _config_logger.debug(f"   OPENROUTER_MODEL: {OPENROUTER_MODEL}")
    _config_logger.debug(f"   MISTRAL_API_KEY: {'***' + MISTRAL_API_KEY[-4:] if MISTRAL_API_KEY else '(not set)'}")
    _config_logger.debug(f"   MISTRAL_MODEL: {MISTRAL_MODEL}")
    _config_logger.debug("="*60)

# Translation tags - Improved for LLM clarity and reliability
TRANSLATE_TAG_IN = "<TRANSLATION>"
TRANSLATE_TAG_OUT = "</TRANSLATION>"
INPUT_TAG_IN = "<SOURCE_TEXT>"
INPUT_TAG_OUT = "</SOURCE_TEXT>"

# ============================================================================
# TAG PLACEHOLDER CONFIGURATION
# ============================================================================
# These placeholders are used to temporarily replace HTML/XML tags during
# translation. The LLM must preserve them exactly in its output.
#
# Unified format: [id0], [id1], [id2], ...
# - Semantic naming helps LLM understand these are identifiers
# - Compact format reduces token usage
# - Adjacent tags are grouped into single placeholders
# - Strict validation ensures placeholder integrity

# Single unified format
PLACEHOLDER_PREFIX = "[id"
"""Prefix for tag placeholders (e.g., [id in [id0])"""

PLACEHOLDER_SUFFIX = "]"
"""Suffix for tag placeholders (e.g., ] in [id0])"""

PLACEHOLDER_PATTERN = r'\[id(\d+)\]'
"""Regex pattern for placeholders (e.g., [id0])"""

# Maximum retries for placeholder validation before falling back to source text
MAX_PLACEHOLDER_RETRIES = 0
"""Number of retry attempts when placeholder validation fails"""

MAX_PLACEHOLDER_CORRECTION_ATTEMPTS = 0
"""Number of LLM correction attempts before falling back to proportional insertion (0 = skip correction phase entirely)"""

# =============================================================================
# TOKEN ALIGNMENT FALLBACK CONFIGURATION (Phase 2)
# =============================================================================
# When LLM fails to preserve placeholders correctly, use word-level alignment
# to reinsert them at semantically correct positions.

EPUB_TOKEN_ALIGNMENT_ENABLED = os.getenv('EPUB_TOKEN_ALIGNMENT_ENABLED', 'true').lower() == 'true'
"""Enable token alignment fallback for EPUB translation (Phase 2)"""

EPUB_TOKEN_ALIGNMENT_METHOD = os.getenv('EPUB_TOKEN_ALIGNMENT_METHOD', 'proportional')
"""
Alignment method to use:
- 'proportional': Simple position-based alignment (fast, no dependencies)
- 'advanced': Future - could add ML-based alignment
"""


def detect_placeholder_mode(text: str) -> tuple:
    """
    Returns the unified placeholder format [idN].

    Args:
        text: Text (parameter kept for backward compatibility but unused)

    Returns:
        Tuple of (prefix, suffix, pattern) for the [idN] format
    """
    return (PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX, PLACEHOLDER_PATTERN)


def create_placeholder(tag_num: int, prefix: str = None, suffix: str = None) -> str:
    """
    Create a placeholder string for a given tag number.

    Args:
        tag_num: Tag number
        prefix: Optional custom prefix (defaults to PLACEHOLDER_PREFIX)
        suffix: Optional custom suffix (defaults to PLACEHOLDER_SUFFIX)

    Returns:
        Placeholder string like [id0]
    """
    if prefix is None:
        prefix = PLACEHOLDER_PREFIX
    if suffix is None:
        suffix = PLACEHOLDER_SUFFIX
    return f"{prefix}{tag_num}{suffix}"


def create_example_placeholder(prefix: str = None, suffix: str = None) -> str:
    """
    Create an example placeholder for documentation/prompts.

    Args:
        prefix: Optional custom prefix
        suffix: Optional custom suffix

    Returns:
        Example placeholder like [id0]
    """
    return create_placeholder(0, prefix, suffix)


def detect_format_from_placeholder(sample_placeholder: str) -> str:
    """
    Returns the unified format name.

    Args:
        sample_placeholder: A sample placeholder (parameter kept for backward compatibility)

    Returns:
        Format name (always "id" for [idN] format)
    """
    return "id"


# Sentence terminators
SENTENCE_TERMINATORS = tuple(list(".!?") + ['."', '?"', '!"', '."', ".'", "?'", "!'", ":", ".)"])

# EPUB-specific configuration
NAMESPACES = {
    'opf': 'http://www.idpf.org/2007/opf',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'xhtml': 'http://www.w3.org/1999/xhtml',
    'epub': 'http://www.idpf.org/2007/ops'
}

IGNORED_TAGS_EPUB = [
    '{http://www.w3.org/1999/xhtml}script',
    '{http://www.w3.org/1999/xhtml}style',
    '{http://www.w3.org/1999/xhtml}meta',
    '{http://www.w3.org/1999/xhtml}link'
]

CONTENT_BLOCK_TAGS_EPUB = [
    '{http://www.w3.org/1999/xhtml}p', '{http://www.w3.org/1999/xhtml}div',
    '{http://www.w3.org/1999/xhtml}li', '{http://www.w3.org/1999/xhtml}h1',
    '{http://www.w3.org/1999/xhtml}h2', '{http://www.w3.org/1999/xhtml}h3',
    '{http://www.w3.org/1999/xhtml}h4', '{http://www.w3.org/1999/xhtml}h5',
    '{http://www.w3.org/1999/xhtml}h6', '{http://www.w3.org/1999/xhtml}blockquote',
    '{http://www.w3.org/1999/xhtml}td', '{http://www.w3.org/1999/xhtml}th',
    '{http://www.w3.org/1999/xhtml}caption',
    '{http://www.w3.org/1999/xhtml}dt', '{http://www.w3.org/1999/xhtml}dd'
]

# Model family context size defaults (shared across providers)
# Used as fallback when context size cannot be detected from the server
# NOTE: Order matters! More specific patterns (gpt-4) must come before generic ones (gpt)
MODEL_FAMILY_CONTEXT_DEFAULTS = {
    "gpt-4": 128000,  # Must come before "gpt"
    "gpt": 8192,
    "claude": 100000,
    "deepseek": 16384,
    "mistral": 32000,  # Mistral small has 32K, large/medium have 128K
    "gemma": 8192,
    "qwen": 8192,
    "llama": 4096,
    "phi": 2048,
}
DEFAULT_CONTEXT_FALLBACK = 2048


@dataclass
class TranslationConfig:
    """Unified configuration for both CLI and web interfaces"""
    
    # Core settings
    source_language: str = DEFAULT_SOURCE_LANGUAGE
    target_language: str = DEFAULT_TARGET_LANGUAGE
    model: str = DEFAULT_MODEL
    api_endpoint: str = API_ENDPOINT
    
    # LLM Provider settings
    llm_provider: str = LLM_PROVIDER
    gemini_api_key: str = GEMINI_API_KEY
    openai_api_key: str = OPENAI_API_KEY
    openrouter_api_key: str = OPENROUTER_API_KEY
    mistral_api_key: str = MISTRAL_API_KEY

    # LLM parameters
    timeout: int = REQUEST_TIMEOUT
    max_attempts: int = MAX_TRANSLATION_ATTEMPTS
    retry_delay: int = 2  # Fixed retry delay in seconds
    context_window: int = OLLAMA_NUM_CTX

    # Context optimization
    auto_adjust_context: bool = AUTO_ADJUST_CONTEXT
    min_chunk_size: int = MIN_CHUNK_SIZE
    max_chunk_size: int = MAX_CHUNK_SIZE

    # Token-based chunking
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK
    soft_limit_ratio: float = SOFT_LIMIT_RATIO

    # Interface-specific
    interface_type: str = "cli"  # or "web"
    enable_colors: bool = True
    enable_interruption: bool = False

    @classmethod
    def from_cli_args(cls, args) -> 'TranslationConfig':
        """Create config from CLI arguments"""
        return cls(
            source_language=args.source_lang,
            target_language=args.target_lang,
            model=args.model,
            api_endpoint=args.api_endpoint,
            interface_type="cli",
            enable_colors=not args.no_color,
            llm_provider=getattr(args, 'provider', LLM_PROVIDER),
            gemini_api_key=getattr(args, 'gemini_api_key', GEMINI_API_KEY),
            openai_api_key=getattr(args, 'openai_api_key', OPENAI_API_KEY),
            openrouter_api_key=getattr(args, 'openrouter_api_key', OPENROUTER_API_KEY),
            mistral_api_key=getattr(args, 'mistral_api_key', MISTRAL_API_KEY),
            max_tokens_per_chunk=getattr(args, 'max_tokens_per_chunk', MAX_TOKENS_PER_CHUNK),
            soft_limit_ratio=getattr(args, 'soft_limit_ratio', SOFT_LIMIT_RATIO)
        )

    @classmethod
    def from_web_request(cls, request_data: dict) -> 'TranslationConfig':
        """Create config from web request data"""
        return cls(
            source_language=request_data.get('source_language', DEFAULT_SOURCE_LANGUAGE),
            target_language=request_data.get('target_language', DEFAULT_TARGET_LANGUAGE),
            model=request_data.get('model', DEFAULT_MODEL),
            api_endpoint=request_data.get('llm_api_endpoint', API_ENDPOINT),
            timeout=request_data.get('timeout', REQUEST_TIMEOUT),
            max_attempts=request_data.get('max_attempts', MAX_TRANSLATION_ATTEMPTS),
            retry_delay=request_data.get('retry_delay', 2),
            context_window=request_data.get('context_window', OLLAMA_NUM_CTX),
            auto_adjust_context=request_data.get('auto_adjust_context', AUTO_ADJUST_CONTEXT),
            min_chunk_size=request_data.get('min_chunk_size', MIN_CHUNK_SIZE),
            max_chunk_size=request_data.get('max_chunk_size', MAX_CHUNK_SIZE),
            interface_type="web",
            enable_interruption=True,
            llm_provider=request_data.get('llm_provider', LLM_PROVIDER),
            gemini_api_key=request_data.get('gemini_api_key', GEMINI_API_KEY),
            openai_api_key=request_data.get('openai_api_key', OPENAI_API_KEY),
            openrouter_api_key=request_data.get('openrouter_api_key', OPENROUTER_API_KEY),
            mistral_api_key=request_data.get('mistral_api_key', MISTRAL_API_KEY),
            max_tokens_per_chunk=request_data.get('max_tokens_per_chunk', MAX_TOKENS_PER_CHUNK),
            soft_limit_ratio=request_data.get('soft_limit_ratio', SOFT_LIMIT_RATIO)
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'source_language': self.source_language,
            'target_language': self.target_language,
            'model': self.model,
            'api_endpoint': self.api_endpoint,
            'timeout': self.timeout,
            'max_attempts': self.max_attempts,
            'retry_delay': self.retry_delay,
            'context_window': self.context_window,
            'llm_provider': self.llm_provider,
            'gemini_api_key': self.gemini_api_key,
            'openai_api_key': self.openai_api_key,
            'openrouter_api_key': self.openrouter_api_key,
            'mistral_api_key': self.mistral_api_key,
            'max_tokens_per_chunk': self.max_tokens_per_chunk,
            'soft_limit_ratio': self.soft_limit_ratio
        }


def detect_placeholder_format_in_text(text: str) -> tuple:
    """
    Returns the unified placeholder format [idN].

    Args:
        text: Text (parameter kept for backward compatibility but unused)

    Returns:
        (prefix, suffix) tuple for [idN] format

    Example:
        >>> detect_placeholder_format_in_text("Hello [id0] world [id1]")
        ("[id", "]")
    """
    return PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX


def detect_existing_placeholder_format(text: str) -> tuple:
    """
    Returns the unified placeholder format [idN].

    Args:
        text: Text (parameter kept for backward compatibility but unused)

    Returns:
        Tuple of (prefix, suffix, pattern) for the [idN] format

    Example:
        >>> detect_existing_placeholder_format("Hello [id0] world [id1]")
        ("[id", "]", r'\\[id(\\d+)\\]')
    """
    return (PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX, PLACEHOLDER_PATTERN)