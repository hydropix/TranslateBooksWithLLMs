"""
Benchmark translation wrapper.

Provides a simplified interface for translating reference texts
using Ollama, OpenAI-compatible, or OpenRouter models for benchmark testing.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable, Union

from benchmark.config import BenchmarkConfig
from benchmark.models import ReferenceText, TranslationResult
from src.core.llm import OllamaProvider, OpenAICompatibleProvider, OpenRouterProvider, PoeProvider, LLMProvider
from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT


# Map BCP-47 codes to display names used in the translation prompt.
_LANG_NAME = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean",
    "zh-Hans": "Chinese (Simplified)", "zh-Hant": "Chinese (Traditional)",
    "ru": "Russian", "ar": "Arabic", "vi": "Vietnamese", "id": "Indonesian",
}


def code_to_language_name(code: str, fallback: Optional[str] = None) -> str:
    """Best-effort BCP-47 code → human display name. Falls back to the code."""
    return _LANG_NAME.get(code, fallback or code)


@dataclass
class TranslationRequest:
    """Request for a single translation."""

    text: ReferenceText
    target_language: str
    target_language_name: str
    model: str
    source_language: str = "en"
    source_language_name: str = "English"


class BenchmarkTranslator:
    """
    Wrapper for translation in benchmark context.

    Simplified interface focusing on:
    - Single text translation
    - Timing measurement
    - Error handling for benchmark purposes
    - Support for multiple providers (Ollama, OpenAI-compatible, OpenRouter)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        log_callback: Optional[Callable[[str, str], None]] = None,
        provider_type: str = "ollama"
    ):
        """
        Initialize the benchmark translator.

        Args:
            config: Benchmark configuration
            log_callback: Optional callback for logging (level, message)
            provider_type: Provider to use ("ollama", "openai", or "openrouter")
        """
        self.config = config
        self.log_callback = log_callback
        self.provider_type = provider_type.lower()
        self._providers: dict[str, LLMProvider] = {}

    def _provider_label(self) -> str:
        """Get a human-readable label for the active provider."""
        labels = {
            "ollama": "Ollama",
            "openai": "OpenAI-compatible provider",
            "openrouter": "OpenRouter",
            "poe": "Poe",
        }
        return labels.get(self.provider_type, self.provider_type)

    def _log(self, level: str, message: str) -> None:
        """Log a message using the callback if available."""
        if self.log_callback:
            self.log_callback(level, message)
        else:
            print(f"[{level.upper()}] {message}")

    def _get_provider(self, model: str) -> LLMProvider:
        """Get or create a provider for the given model."""
        if model not in self._providers:
            if self.provider_type == "openrouter":
                if not self.config.openrouter.api_key:
                    raise ValueError("OpenRouter API key is required for translation. "
                                   "Set OPENROUTER_API_KEY in .env or use --openrouter-key")
                self._providers[model] = OpenRouterProvider(
                    api_key=self.config.openrouter.api_key,
                    model=model
                )
            elif self.provider_type == "poe":
                if not self.config.poe.api_key:
                    raise ValueError("Poe API key is required for translation. "
                                     "Set POE_API_KEY in .env or use --poe-key")
                self._providers[model] = PoeProvider(
                    api_key=self.config.poe.api_key,
                    model=model,
                )
            elif self.provider_type == "openai":
                self._providers[model] = OpenAICompatibleProvider(
                    api_endpoint=self.config.openai.endpoint,
                    api_key=self.config.openai.api_key,
                    model=model,
                    context_window=self.config.openai.context_window,
                    log_callback=lambda level, msg: self._log(level, msg)
                )
            else:
                # Default to Ollama
                self._providers[model] = OllamaProvider(
                    api_endpoint=self.config.ollama.endpoint,
                    model=model,
                    context_window=self.config.ollama.num_ctx,
                    log_callback=lambda level, msg: self._log(level, msg)
                )
        return self._providers[model]

    def _build_prompt(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> tuple[str, str]:
        """
        Build system and user prompts for literary translation.

        Similar to the main app's prompts but focused on literary texts
        without technical elements (no placeholders, HTML, code, etc.).

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = f"""You are a professional {target_language} translator and writer.

# TRANSLATION PRINCIPLES

Translate {source_language} to {target_language}. Output only the translation.

**PRIORITY ORDER:**
1. Preserve exact names
2. Match original tone and formality
3. Use natural {target_language} phrasing - never word-for-word
4. Fix grammar/spelling errors in output
5. Translate idioms to {target_language} equivalents
6. Preserve the author's literary style and emotional impact

**QUALITY CHECK:**
- Does it sound natural to a native {target_language} speaker?
- Are all details from the original included?
- Does punctuation follow {target_language} conventions?

If unsure between literal and natural phrasing: **choose natural**.

**LAYOUT PRESERVATION:**
- Keep the exact text layout, spacing, line breaks, and indentation
- **WRITE YOUR TRANSLATION IN {target_language.upper()} - THIS IS MANDATORY**

# FINAL REMINDER: YOUR OUTPUT LANGUAGE

**YOU MUST TRANSLATE INTO {target_language.upper()}.**
Your entire translation output must be written in {target_language}.
Do NOT write in {source_language} or any other language - ONLY {target_language.upper()}.

# OUTPUT FORMAT

**CRITICAL OUTPUT RULES:**
1. Your response MUST start with {TRANSLATE_TAG_IN} (first characters, no text before)
2. Your response MUST end with {TRANSLATE_TAG_OUT} (last characters, no text after)
3. Include NOTHING before {TRANSLATE_TAG_IN} and NOTHING after {TRANSLATE_TAG_OUT}
4. Do NOT add explanations, comments, notes, or greetings

**INCORRECT examples (DO NOT do this):**
❌ "Here is the translation: {TRANSLATE_TAG_IN}Text...{TRANSLATE_TAG_OUT}"
❌ "{TRANSLATE_TAG_IN}Text...{TRANSLATE_TAG_OUT} (Additional comment)"
❌ "Sure! {TRANSLATE_TAG_IN}Text...{TRANSLATE_TAG_OUT}"
❌ "Text..." (missing tags entirely)
❌ "{TRANSLATE_TAG_IN}Text..." (missing closing tag)

**CORRECT format (ONLY this):**
✅ {TRANSLATE_TAG_IN}
Your translated text here
{TRANSLATE_TAG_OUT}"""

        user_prompt = f"""# TEXT TO TRANSLATE

{text}

REMINDER: Output ONLY your translation in this exact format:
{TRANSLATE_TAG_IN}
your translation here
{TRANSLATE_TAG_OUT}

Start with {TRANSLATE_TAG_IN} and end with {TRANSLATE_TAG_OUT}. Nothing before or after.

Provide your translation now:"""

        return system_prompt, user_prompt

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """
        Translate a single reference text.

        Args:
            request: Translation request details

        Returns:
            TranslationResult with translation or error
        """
        start_time = time.perf_counter()

        try:
            provider = self._get_provider(request.model)

            system_prompt, user_prompt = self._build_prompt(
                text=request.text.content,
                source_language=request.source_language_name,
                target_language=request.target_language_name
            )

            # Make the translation request
            request_timeout = self.config.openai.timeout if self.provider_type == "openai" else self.config.ollama.timeout
            llm_response = await provider.generate(
                prompt=user_prompt,
                timeout=request_timeout,
                system_prompt=system_prompt
            )

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            if not llm_response:
                return TranslationResult(
                    source_text_id=request.text.id,
                    target_language=request.target_language,
                    model=request.model,
                    translated_text="",
                    translation_time_ms=elapsed_ms,
                    error=f"No response from {self._provider_label()}"
                )

            # Extract response content from LLMResponse object
            response_content = llm_response.content

            # Extract translation from response
            translated_text = provider.extract_translation(response_content)

            if not translated_text:
                # Fallback: use the raw response if extraction fails
                self._log("warning", f"Could not extract translation tags, using raw response")
                translated_text = response_content.strip()

            return TranslationResult(
                source_text_id=request.text.id,
                target_language=request.target_language,
                model=request.model,
                translated_text=translated_text,
                translation_time_ms=elapsed_ms
            )

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            self._log("error", f"Translation failed: {e}")

            return TranslationResult(
                source_text_id=request.text.id,
                target_language=request.target_language,
                model=request.model,
                translated_text="",
                translation_time_ms=elapsed_ms,
                error=str(e)
            )

    async def translate_batch(
        self,
        requests: list[TranslationRequest],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[TranslationResult]:
        """
        Translate multiple texts sequentially.

        Args:
            requests: List of translation requests
            progress_callback: Optional callback (completed, total)

        Returns:
            List of TranslationResults
        """
        results = []
        total = len(requests)

        for i, request in enumerate(requests):
            self._log("info", f"Translating {request.text.id} to {request.target_language_name} with {request.model}")

            result = await self.translate(request)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

            # Small delay to avoid overwhelming the selected backend
            if i < total - 1:
                await asyncio.sleep(0.5)

        return results

    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


async def test_ollama_connection(config: BenchmarkConfig) -> tuple[bool, str]:
    """
    Test if Ollama is accessible and the default model is available.

    Args:
        config: Benchmark configuration

    Returns:
        Tuple of (success, message)
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Test Ollama API endpoint
            response = await client.get(
                config.ollama.endpoint.replace("/api/generate", "/api/tags")
            )
            response.raise_for_status()

            data = response.json()
            models = [m["name"] for m in data.get("models", [])]

            if not models:
                return False, "No models found in Ollama. Run 'ollama pull <model>' first."

            return True, f"Ollama connected. Available models: {', '.join(models[:5])}..."

    except httpx.ConnectError:
        return False, f"Cannot connect to Ollama at {config.ollama.endpoint}. Is Ollama running?"
    except Exception as e:
        return False, f"Ollama connection test failed: {e}"


async def get_available_ollama_models(config: BenchmarkConfig) -> list[str]:
    """
    Get list of available Ollama models.

    Args:
        config: Benchmark configuration

    Returns:
        List of model names
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                config.ollama.endpoint.replace("/api/generate", "/api/tags")
            )
            response.raise_for_status()

            data = response.json()
            return [m["name"] for m in data.get("models", [])]

    except Exception:
        return []


async def get_available_openrouter_models(config: BenchmarkConfig, text_only: bool = True) -> list[dict]:
    """
    Get list of available OpenRouter models.

    Args:
        config: Benchmark configuration
        text_only: If True, filter out vision/multimodal models

    Returns:
        List of model dicts with id, name, pricing info
    """
    if not config.openrouter.api_key:
        print("⚠️ OpenRouter API key not configured. Using fallback model list.")
        return OpenRouterProvider.FALLBACK_MODELS

    try:
        provider = OpenRouterProvider(
            api_key=config.openrouter.api_key,
            model="dummy"  # Model doesn't matter for listing
        )
        models = await provider.get_available_models(text_only=text_only)
        await provider.close()
        return models

    except Exception as e:
        print(f"⚠️ Failed to fetch OpenRouter models: {e}")
        return [{"id": m, "name": m} for m in OpenRouterProvider.FALLBACK_MODELS]


async def _fetch_openai_models(config: BenchmarkConfig) -> list[dict]:
    """
    Fetch and filter models from an OpenAI-compatible endpoint.

    Args:
        config: Benchmark configuration

    Returns:
        List of model dicts with id, name, and owned_by fields.
        Empty list if the endpoint is unreachable or returns no models.
    """
    import httpx

    base_url = config.openai.endpoint.replace("/chat/completions", "").rstrip("/")
    headers = {}
    if config.openai.api_key:
        headers["Authorization"] = f"Bearer {config.openai.api_key}"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()

        data = response.json()
        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            if not model_id:
                continue
            if "embedding" in model_id.lower() or "whisper" in model_id.lower():
                continue
            models.append({
                "id": model_id,
                "name": model_id,
                "owned_by": model.get("owned_by", "unknown"),
            })

        models.sort(key=lambda item: item["name"].lower())
        return models


async def get_available_openai_models(config: BenchmarkConfig) -> list[dict]:
    """
    Get list of available models from an OpenAI-compatible endpoint.

    Args:
        config: Benchmark configuration

    Returns:
        List of model dicts with id and name. Empty list on failure.
    """
    try:
        return await _fetch_openai_models(config)
    except Exception:
        return []


async def test_openai_translation_connection(config: BenchmarkConfig) -> tuple[bool, str]:
    """
    Test if an OpenAI-compatible endpoint is accessible for translation.

    Args:
        config: Benchmark configuration

    Returns:
        Tuple of (success, message)
    """
    import httpx

    try:
        models = await _fetch_openai_models(config)

        if not models:
            return False, "No OpenAI-compatible models available"

        model_names = [m["id"] for m in models[:5]]
        return True, (
            f"OpenAI-compatible endpoint connected ({config.openai.endpoint}). "
            f"Available models: {', '.join(model_names)}..."
        )
    except httpx.ConnectError:
        return False, f"Cannot connect to OpenAI-compatible endpoint at {config.openai.endpoint}"
    except httpx.HTTPStatusError as e:
        return False, f"OpenAI-compatible HTTP error: {e.response.status_code}"
    except Exception as e:
        return False, f"OpenAI-compatible connection test failed: {e}"


async def test_openrouter_translation_connection(config: BenchmarkConfig) -> tuple[bool, str]:
    """
    Test if OpenRouter is accessible for translation.

    Args:
        config: Benchmark configuration

    Returns:
        Tuple of (success, message)
    """
    if not config.openrouter.api_key:
        return False, "OpenRouter API key not configured"

    try:
        models = await get_available_openrouter_models(config)
        if not models:
            return False, "No OpenRouter models available"

        model_names = [m["id"] if isinstance(m, dict) else m for m in models[:5]]
        return True, f"OpenRouter connected. Available models: {', '.join(model_names)}..."

    except Exception as e:
        return False, f"OpenRouter connection test failed: {e}"
