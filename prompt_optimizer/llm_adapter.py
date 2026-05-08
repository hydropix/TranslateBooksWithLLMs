"""
LLM adapters for translation (Ollama), evaluation (OpenRouter), and mutation (OpenRouter).
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional, Callable

import httpx

from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.logger import get_logger, ConsoleLogger


@dataclass
class TranslationResult:
    """Result of a translation request."""
    text: str
    success: bool
    error: Optional[str] = None
    elapsed_ms: int = 0
    tokens_used: int = 0


@dataclass
class EvaluationResult:
    """Result of an evaluation request."""
    accuracy: float
    fluency: float
    style: float
    overall: float
    feedback: str
    success: bool
    error: Optional[str] = None
    elapsed_ms: int = 0
    cost: float = 0.0

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score using default weights."""
        return (
            self.accuracy * 0.35 +
            self.fluency * 0.30 +
            self.style * 0.20 +
            self.overall * 0.15
        )

    @classmethod
    def failed(cls, error: str) -> 'EvaluationResult':
        """Create a failed evaluation result."""
        return cls(
            accuracy=0.0, fluency=0.0, style=0.0, overall=0.0,
            feedback=error, success=False, error=error
        )


class TranslationAdapter:
    """
    Adapter for translation using Ollama.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        log_callback: Optional[Callable[[str, str], None]] = None,
        console_logger: Optional[ConsoleLogger] = None
    ):
        self.config = config
        self.log_callback = log_callback
        self.console = console_logger or get_logger()
        self._client: Optional[httpx.AsyncClient] = None

    def _log(self, level: str, message: str) -> None:
        """Log a message."""
        if self.log_callback:
            self.log_callback(level, message)
        else:
            print(f"[{level.upper()}] {message}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.ollama.timeout)
            )
        return self._client

    async def translate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> TranslationResult:
        """
        Translate text using Ollama.

        Args:
            system_prompt: The system prompt (instructions)
            user_prompt: The user prompt (text to translate)

        Returns:
            TranslationResult with the translation
        """
        import time
        start_time = time.perf_counter()

        # Log request
        self.console.ollama_request(self.config.ollama.model, system_prompt, user_prompt)

        try:
            client = await self._get_client()

            payload = {
                "model": self.config.ollama.model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "num_ctx": self.config.ollama.num_ctx,
                    "truncate": False
                },
                "think": False  # Disable thinking mode for Qwen3, etc.
            }

            response = await client.post(
                self.config.ollama.endpoint,
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            response_text = result.get("response", "")
            tokens_used = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)

            # Clean up response (remove think blocks if present)
            response_text = self._clean_response(response_text)

            # Log response
            self.console.ollama_response(response_text, elapsed_ms, tokens_used)

            return TranslationResult(
                text=response_text,
                success=True,
                elapsed_ms=elapsed_ms,
                tokens_used=tokens_used
            )

        except httpx.TimeoutException:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return TranslationResult(
                text="", success=False,
                error="Translation request timed out",
                elapsed_ms=elapsed_ms
            )
        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"HTTP error {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", error_msg)
            except Exception:
                pass
            return TranslationResult(
                text="", success=False,
                error=error_msg,
                elapsed_ms=elapsed_ms
            )
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return TranslationResult(
                text="", success=False,
                error=str(e),
                elapsed_ms=elapsed_ms
            )

    def _clean_response(self, response: str) -> str:
        """Clean up response by removing think blocks."""
        if not response:
            return response

        # Remove <think>...</think> blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Remove orphan </think> tags (when opening was truncated)
        response = re.sub(r'^.*?</think>\s*', '', response, flags=re.DOTALL | re.IGNORECASE)

        return response.strip()

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class EvaluationAdapter:
    """
    Adapter for evaluation using OpenRouter.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        log_callback: Optional[Callable[[str, str], None]] = None,
        console_logger: Optional[ConsoleLogger] = None
    ):
        self.config = config
        self.log_callback = log_callback
        self.console = console_logger or get_logger()
        self._client: Optional[httpx.AsyncClient] = None

        # Cost tracking
        self.total_cost: float = 0.0
        self.total_evaluations: int = 0

    def _log(self, level: str, message: str) -> None:
        """Log a message."""
        if self.log_callback:
            self.log_callback(level, message)
        else:
            print(f"[{level.upper()}] {message}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.openrouter.timeout)
            )
        return self._client

    def _build_evaluation_prompt(
        self,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        text_style: str,
        text_title: str,
        text_author: str
    ) -> tuple[str, str]:
        """Build the evaluation prompt."""
        system_prompt = """You are an expert literary translation evaluator with deep knowledge of multiple languages and translation theory.

Your task is to evaluate the quality of a translation objectively and precisely.

# EVALUATION CRITERIA

Score each criterion from 1-10:

1. **Accuracy (1-10)**: How faithfully does the translation preserve the meaning?
   - 10: Perfect meaning preservation, all nuances captured
   - 7-9: Minor omissions or slight meaning shifts
   - 4-6: Some meaning lost or distorted
   - 1-3: Significant meaning errors or omissions

2. **Fluency (1-10)**: How natural does the translation read in the target language?
   - 10: Reads like original prose written by a native speaker
   - 7-9: Natural with minor awkward phrasings
   - 4-6: Understandable but clearly translated
   - 1-3: Unnatural, difficult to read

3. **Style (1-10)**: How well is the literary style/tone preserved?
   - 10: Perfect style match (irony, formality, era, voice)
   - 7-9: Style mostly preserved with minor deviations
   - 4-6: Noticeable style changes
   - 1-3: Style completely different from original

4. **Overall (1-10)**: Your holistic quality assessment.
   - Consider all factors and give an overall grade

# OUTPUT FORMAT

You MUST respond with ONLY a valid JSON object. No text before or after.

{
  "accuracy": <number 1-10>,
  "fluency": <number 1-10>,
  "style": <number 1-10>,
  "overall": <number 1-10>,
  "feedback": "<brief feedback explaining scores, 1-2 sentences>"
}"""

        user_prompt = f"""# TRANSLATION EVALUATION REQUEST

**Source Language**: {source_language}
**Target Language**: {target_language}
**Source Text**: "{text_title}" by {text_author}
**Style**: {text_style}

## Original Text ({source_language}):

{source_text}

## Translation ({target_language}):

{translated_text}

---

Evaluate this translation. Respond with ONLY the JSON object:"""

        return system_prompt, user_prompt

    def _parse_evaluation_response(self, response: str) -> Optional[dict]:
        """Parse the JSON evaluation response."""
        try:
            cleaned = response.strip()

            # Remove markdown code blocks if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Try to find JSON object in response
            json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            data = json.loads(cleaned)

            # Validate and clamp scores to 1-10 range
            def clamp_score(value) -> float:
                try:
                    score = float(value)
                    return max(1.0, min(10.0, score))
                except (TypeError, ValueError):
                    return 5.0

            return {
                'accuracy': clamp_score(data.get("accuracy", 5)),
                'fluency': clamp_score(data.get("fluency", 5)),
                'style': clamp_score(data.get("style", 5)),
                'overall': clamp_score(data.get("overall", 5)),
                'feedback': str(data.get("feedback", ""))[:500]
            }

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    async def evaluate(
        self,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        text_style: str = "",
        text_title: str = "",
        text_author: str = ""
    ) -> EvaluationResult:
        """
        Evaluate a translation using OpenRouter.

        Args:
            source_text: Original text
            translated_text: Translated text to evaluate
            source_language: Source language name
            target_language: Target language name
            text_style: Style description
            text_title: Title of the work
            text_author: Author of the work

        Returns:
            EvaluationResult with scores
        """
        import time
        start_time = time.perf_counter()

        if not self.config.openrouter.api_key:
            return EvaluationResult.failed("OpenRouter API key not configured")

        if not translated_text or not translated_text.strip():
            return EvaluationResult.failed("Empty translation")

        # Log request
        self.console.openrouter_eval_request(self.config.openrouter.model, source_text, translated_text)

        try:
            client = await self._get_client()

            system_prompt, user_prompt = self._build_evaluation_prompt(
                source_text=source_text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                text_style=text_style,
                text_title=text_title,
                text_author=text_author
            )

            headers = {
                "Authorization": f"Bearer {self.config.openrouter.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.config.openrouter.site_url,
                "X-Title": self.config.openrouter.site_name,
            }

            payload = {
                "model": self.config.openrouter.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500,
            }

            response = await client.post(
                self.config.openrouter.endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            # Track cost
            cost = 0.0
            if "usage" in result:
                usage = result["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cost = float(result.get("cost", 0))
                if cost == 0:
                    # Fallback estimate
                    cost = (prompt_tokens * 0.50 / 1_000_000) + (completion_tokens * 1.50 / 1_000_000)
                self.total_cost += cost
                self.total_evaluations += 1

            # Extract response content
            if "choices" not in result or len(result["choices"]) == 0:
                return EvaluationResult.failed("No response from OpenRouter")

            response_text = result["choices"][0].get("message", {}).get("content", "")

            if not response_text:
                return EvaluationResult.failed("Empty response from OpenRouter")

            # Parse the evaluation
            scores = self._parse_evaluation_response(response_text)

            if scores is None:
                return EvaluationResult.failed("Failed to parse evaluation response")

            # Log response
            self.console.openrouter_eval_response(
                accuracy=scores['accuracy'],
                fluency=scores['fluency'],
                style=scores['style'],
                overall=scores['overall'],
                feedback=scores['feedback'],
                elapsed_ms=elapsed_ms,
                cost=cost
            )

            return EvaluationResult(
                accuracy=scores['accuracy'],
                fluency=scores['fluency'],
                style=scores['style'],
                overall=scores['overall'],
                feedback=scores['feedback'],
                success=True,
                elapsed_ms=elapsed_ms,
                cost=cost
            )

        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            error_msg = f"HTTP error: {e.response.status_code}"

            if e.response.status_code == 401:
                error_msg = "Invalid OpenRouter API key"
            elif e.response.status_code == 402:
                error_msg = "Insufficient OpenRouter credits"
            elif e.response.status_code == 404:
                error_msg = f"Model not found: {self.config.openrouter.model}"

            return EvaluationResult.failed(error_msg)

        except httpx.TimeoutException:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return EvaluationResult.failed("Evaluation request timed out")

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return EvaluationResult.failed(str(e))

    def get_cost_summary(self) -> dict:
        """Get summary of evaluation costs."""
        return {
            "total_cost_usd": self.total_cost,
            "total_evaluations": self.total_evaluations,
            "avg_cost_per_evaluation": (
                self.total_cost / self.total_evaluations
                if self.total_evaluations > 0 else 0
            ),
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class MutationAdapter:
    """
    Adapter for LLM-based prompt mutation using OpenRouter.
    Uses the frontier model to intelligently improve prompts.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        log_callback: Optional[Callable[[str, str], None]] = None,
        console_logger: Optional[ConsoleLogger] = None
    ):
        self.config = config
        self.log_callback = log_callback
        self.console = console_logger or get_logger()
        self._client: Optional[httpx.AsyncClient] = None

        # Cost tracking
        self.total_cost: float = 0.0
        self.total_mutations: int = 0

        # Track last mutation for logging
        self._last_parent_tokens: int = 0

    def _log(self, level: str, message: str) -> None:
        """Log a message."""
        if self.log_callback:
            self.log_callback(level, message)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.openrouter.timeout)
            )
        return self._client

    async def mutate_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        strategy: str = "",
        parent_id: str = "",
        parent_fitness: float = 0.0,
        parent_tokens: int = 0,
        feedbacks: list = None
    ) -> tuple[str, bool, str]:
        """
        Call the frontier LLM to mutate/improve a prompt.

        Args:
            system_prompt: The system prompt for mutation instructions
            user_prompt: The user prompt containing the current prompt and feedback
            strategy: Name of the mutation strategy (for logging)
            parent_id: ID of parent prompt (for logging)
            parent_fitness: Fitness of parent (for logging)
            parent_tokens: Token count of parent (for logging)
            feedbacks: List of evaluation feedbacks (for logging)

        Returns:
            Tuple of (new_prompt, success, error_message)
        """
        import time
        start_time = time.perf_counter()

        self._last_parent_tokens = parent_tokens

        # Log mutation request
        self.console.mutation_request(strategy, parent_id, parent_fitness)
        if feedbacks:
            self.console.mutation_context(feedbacks)

        if not self.config.openrouter.api_key:
            return "", False, "OpenRouter API key not configured"

        try:
            client = await self._get_client()

            headers = {
                "Authorization": f"Bearer {self.config.openrouter.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.config.openrouter.site_url,
                "X-Title": self.config.openrouter.site_name,
            }

            payload = {
                "model": self.config.openrouter.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,  # Higher temp for creative mutations
                "max_tokens": 1500,  # Allow longer prompts
            }

            response = await client.post(
                self.config.openrouter.endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            # Track cost
            if "usage" in result:
                usage = result["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cost = float(result.get("cost", 0))
                if cost == 0:
                    cost = (prompt_tokens * 0.50 / 1_000_000) + (completion_tokens * 1.50 / 1_000_000)
                self.total_cost += cost
                self.total_mutations += 1

            # Extract response
            if "choices" not in result or len(result["choices"]) == 0:
                return "", False, "No response from OpenRouter"

            new_prompt = result["choices"][0].get("message", {}).get("content", "")

            if not new_prompt or not new_prompt.strip():
                return "", False, "Empty response from mutation LLM"

            # Clean up the response
            new_prompt = self._clean_mutation_response(new_prompt)

            # Validate that placeholders are preserved
            if "{source_language}" not in new_prompt or "{target_language}" not in new_prompt:
                self._log("warning", "Mutation lost language placeholders, adding them back")
                if "{source_language}" not in new_prompt:
                    new_prompt = new_prompt.replace(
                        "source language", "{source_language}"
                    ).replace(
                        "Source language", "{source_language}"
                    )
                if "{target_language}" not in new_prompt:
                    new_prompt = new_prompt.replace(
                        "target language", "{target_language}"
                    ).replace(
                        "Target language", "{target_language}"
                    )

            # Log mutation response
            new_tokens = len(new_prompt) // 4
            token_change = new_tokens - self._last_parent_tokens
            self.console.mutation_response(new_prompt, elapsed_ms, token_change)

            return new_prompt, True, ""

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error: {e.response.status_code}"
            self.console.error(f"Mutation failed: {error_msg}")
            return "", False, error_msg

        except httpx.TimeoutException:
            self.console.error("Mutation request timed out")
            return "", False, "Mutation request timed out"

        except Exception as e:
            self.console.error(f"Mutation error: {e}")
            return "", False, str(e)

    def _clean_mutation_response(self, response: str) -> str:
        """Clean up the mutation response."""
        response = response.strip()

        # Remove markdown code blocks if the LLM wrapped the prompt
        if response.startswith("```"):
            lines = response.split('\n')
            # Remove first line (```yaml or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = '\n'.join(lines)

        # Remove any leading/trailing quotes
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]

        return response.strip()

    def get_cost_summary(self) -> dict:
        """Get summary of mutation costs."""
        return {
            "total_cost_usd": self.total_cost,
            "total_mutations": self.total_mutations,
            "avg_cost_per_mutation": (
                self.total_cost / self.total_mutations
                if self.total_mutations > 0 else 0
            ),
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class LLMAdapter:
    """
    Combined adapter for translation, evaluation, and mutation.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        log_callback: Optional[Callable[[str, str], None]] = None,
        console_logger: Optional[ConsoleLogger] = None
    ):
        self.config = config
        self.log_callback = log_callback
        self.console = console_logger or get_logger()
        self.translation = TranslationAdapter(config, log_callback, self.console)
        self.evaluation = EvaluationAdapter(config, log_callback, self.console)
        self.mutation = MutationAdapter(config, log_callback, self.console)

    async def translate_and_evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        source_text: str,
        source_language: str,
        target_language: str,
        text_style: str = "",
        text_title: str = "",
        text_author: str = ""
    ) -> tuple[TranslationResult, EvaluationResult]:
        """
        Translate text and evaluate the translation.

        Args:
            system_prompt: System prompt for translation
            user_prompt: User prompt for translation
            source_text: Original text
            source_language: Source language name
            target_language: Target language name
            text_style: Style description
            text_title: Title of the work
            text_author: Author of the work

        Returns:
            Tuple of (TranslationResult, EvaluationResult)
        """
        # Translate
        translation = await self.translation.translate(system_prompt, user_prompt)

        if not translation.success or not translation.text:
            return translation, EvaluationResult.failed("Translation failed")

        # Rate limit between calls
        await asyncio.sleep(0.5)

        # Evaluate
        evaluation = await self.evaluation.evaluate(
            source_text=source_text,
            translated_text=translation.text,
            source_language=source_language,
            target_language=target_language,
            text_style=text_style,
            text_title=text_title,
            text_author=text_author
        )

        return translation, evaluation

    async def close(self) -> None:
        """Close all HTTP clients."""
        await self.translation.close()
        await self.evaluation.close()
        await self.mutation.close()
