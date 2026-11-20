"""
Ollama provider implementation.

This module provides the OllamaProvider class for interacting with local
Ollama servers with full thinking model detection and handling.
"""

import json
import logging
import re
from typing import Callable, Optional

import httpx

from ..base import (
    LLMProvider, LLMRequest, LLMResponse, is_output_limit_finish, provider_error_from_response,
)
from ..exceptions import ProviderError, RateLimitError, RepetitionLoopError
from ..policy import RetryPolicy
from ..rate_limit_handler import compute_wait_time, wait_for_request_slot
from ..thinking.cache import get_thinking_cache
from ..thinking.detection import detect_repetition_loop
from ..thinking.behavior import ThinkingBehavior, _model_matches_pattern
from ..utils.context_detection import ContextDetector

from src.config import (
    API_ENDPOINT,
    DEFAULT_MODEL,
    OLLAMA_NUM_CTX,
    UNCONTROLLABLE_THINKING_MODELS,
    CONTROLLABLE_THINKING_MODELS,
    REPETITION_MIN_COUNT_STREAMING
)


class OllamaProvider(LLMProvider):
    """Ollama API provider - uses /api/chat for proper think parameter support"""

    DISPLAY_NAME = "Ollama"
    STATUS_HINTS = {404: "model not found, check `ollama list`"}

    def _status_hint(self, status_code: Optional[int]) -> str:
        return self.STATUS_HINTS.get(status_code or 0) or super()._status_hint(status_code)

    def __init__(self, api_endpoint: str = API_ENDPOINT, model: str = DEFAULT_MODEL,
                 context_window: int = OLLAMA_NUM_CTX, log_callback: Optional[Callable] = None,
                 *, retry_policy: Optional[RetryPolicy] = None,
                 temperature: Optional[float] = None):
        super().__init__(model, provider_name="ollama", retry_policy=retry_policy,
                         temperature=temperature, log_callback=log_callback)
        # Convert /api/generate endpoint to /api/chat for proper think support
        self.api_endpoint = api_endpoint.replace('/api/generate', '/api/chat')
        self.context_window = context_window
        self._context_detector = ContextDetector()
        # Will be detected on first request via _detect_thinking_behavior()
        self._thinking_behavior: Optional[ThinkingBehavior] = None
        self._supports_think_param: bool = True
        # Quick check against known model lists (fallback if detection fails)
        self._known_uncontrollable = any(_model_matches_pattern(model, tm) for tm in UNCONTROLLABLE_THINKING_MODELS)
        self._known_controllable = any(_model_matches_pattern(model, tm) for tm in CONTROLLABLE_THINKING_MODELS)

    def _check_known_model_lists(self) -> Optional[ThinkingBehavior]:
        """Check if model matches known model lists for quick classification."""
        # Check uncontrollable list first (more specific matches)
        for pattern in UNCONTROLLABLE_THINKING_MODELS:
            if _model_matches_pattern(self.model, pattern):
                return ThinkingBehavior.UNCONTROLLABLE

        # Check controllable list
        for pattern in CONTROLLABLE_THINKING_MODELS:
            if _model_matches_pattern(self.model, pattern):
                return ThinkingBehavior.CONTROLLABLE

        return None

    async def _test_thinking(self, think_param: Optional[bool] = None) -> tuple[bool, bool, bool]:
        """
        Test model thinking behavior with specific think parameter.

        Args:
            think_param: True/False/None (None = don't include param)

        Returns:
            (has_thinking_field, has_think_tags, supports_param)
        """
        test_prompt = "What is 2+2? Reply with just the number."

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": test_prompt}],
            "stream": False,
            "options": {"num_ctx": 2048},
        }

        if think_param is not None:
            payload["think"] = think_param

        client = await self._get_client()
        # A real request to the provider: paced like the ones generate() sends.
        await wait_for_request_slot(self.provider_name)
        response = await client.post(self.api_endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")

        has_thinking_field = bool(thinking)
        has_think_tags = bool(re.search(r'<think>|</think>', content, re.IGNORECASE))

        return has_thinking_field, has_think_tags, True

    async def _detect_thinking_behavior(self) -> ThinkingBehavior:
        """
        Detect model's thinking behavior by testing with different think parameters.

        Classification:
        - STANDARD: Model never thinks
        - CONTROLLABLE: Model thinks but respects think=false
        - UNCONTROLLABLE: Model thinks even with think=false (needs WARNING)

        Uses persistent cache to avoid re-testing on every startup.

        Returns:
            ThinkingBehavior classification
        """
        # Check cache first (instant)
        cache = get_thinking_cache()
        cached = cache.get(self.model, self.api_endpoint)
        if cached:
            if self.log_callback:
                self.log_callback("info", f"[MODEL] {self.model}: {cached.value} (from cache)")
            return cached

        # Check known model lists (instant fallback)
        known_behavior = self._check_known_model_lists()
        if known_behavior:
            if self.log_callback:
                self.log_callback("info", f"[MODEL] {self.model}: {known_behavior.value} (from known list)")
            # Cache for future use
            cache.set(self.model, known_behavior, True, self.api_endpoint)
            return known_behavior

        # Need to run dynamic tests (slow - 3 LLM requests)
        if self.log_callback:
            self.log_callback("info", f"[MODEL] {self.model}: Testing thinking behavior (first time)...")

        try:
            # Test 1: Without think parameter (baseline)
            field_none, tags_none, _ = await self._test_thinking(think_param=None)
            thinks_without_param = field_none or tags_none

            # Test 2: With think=true (does model support thinking?)
            try:
                field_true, tags_true, _ = await self._test_thinking(think_param=True)
                thinks_when_enabled = field_true or tags_true
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    # Model doesn't support think param
                    self._supports_think_param = False
                    if thinks_without_param:
                        return ThinkingBehavior.UNCONTROLLABLE
                    return ThinkingBehavior.STANDARD
                raise

            # Test 3: With think=false (can we disable thinking?)
            try:
                field_false, tags_false, _ = await self._test_thinking(think_param=False)
                thinks_when_disabled = field_false or tags_false
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    self._supports_think_param = False
                    if thinks_without_param:
                        return ThinkingBehavior.UNCONTROLLABLE
                    return ThinkingBehavior.STANDARD
                raise

            # Classify based on test results
            if thinks_when_disabled:
                # Model thinks even with think=false - UNCONTROLLABLE
                behavior = ThinkingBehavior.UNCONTROLLABLE
            elif thinks_when_enabled or thinks_without_param:
                # Model can think but respects think=false - CONTROLLABLE
                behavior = ThinkingBehavior.CONTROLLABLE
            else:
                # Model never thinks - STANDARD
                behavior = ThinkingBehavior.STANDARD

            # Cache the result for future use
            cache.set(self.model, behavior, self._supports_think_param, self.api_endpoint)
            if self.log_callback:
                self.log_callback("info", f"[MODEL] {self.model}: {behavior.value} (tested & cached)")

            return behavior

        except Exception as e:
            # If detection fails, use known lists or default to standard
            if self.log_callback:
                self.log_callback("warning", f"[MODEL DETECTION] Failed for {self.model}: {e}")

            if self._known_uncontrollable:
                return ThinkingBehavior.UNCONTROLLABLE
            elif self._known_controllable:
                return ThinkingBehavior.CONTROLLABLE
            return ThinkingBehavior.STANDARD

    def _show_thinking_warning(self):
        """Log a warning for uncontrollable thinking models."""
        lines = [
            f"[WARNING] UNCONTROLLABLE THINKING MODEL: {self.model}",
            "This model produces <think> blocks that CANNOT be disabled:",
            "  - SLOWER translations (model thinks before answering)",
            "  - MORE tokens consumed (reasoning uses context window)",
            "  - LESS consistent results for translation tasks",
        ]
        model_lower = self.model.lower()
        if "qwen3" in model_lower and "instruct" not in model_lower:
            size_match = re.search(r':(\d+b)', model_lower)
            size = size_match.group(1) if size_match else ""
            if size:
                lines.append(f"RECOMMENDATION: use 'qwen3:{size}-instruct' instead "
                             f"(direct answers, same quality, faster, fewer tokens)")
            else:
                lines.append("RECOMMENDATION: use a Qwen3 instruct variant "
                             "(e.g. qwen3:14b-instruct, qwen3:30b-instruct)")
        elif "phi4-reasoning" in model_lower:
            lines.append("RECOMMENDATION: use 'phi4:latest' (no reasoning mode)")
        elif "deepseek" in model_lower or "qwq" in model_lower:
            lines.append("RECOMMENDATION: use a non-reasoning model for translation")
        lines.append("[INFO] Using think=true to cleanly separate thinking from content")
        self._log("thinking_model_warning", "\n".join(lines))

    async def _ensure_thinking_behavior(self) -> None:
        """Classify the model's thinking behavior once (cache, lists or tests)."""
        if self._thinking_behavior is not None:
            return
        self._thinking_behavior = await self._detect_thinking_behavior()
        if self._thinking_behavior == ThinkingBehavior.UNCONTROLLABLE:
            self._show_thinking_warning()
        elif self._thinking_behavior == ThinkingBehavior.CONTROLLABLE:
            self._logger.info(f"[MODEL] {self.model}: controllable thinking model - using think=false")
        else:
            self._logger.info(f"[MODEL] {self.model}: standard model (no thinking)")

    async def detect_thinking_model(self) -> Optional[bool]:
        await self._ensure_thinking_behavior()
        return self.get_is_thinking_model()

    async def _before_request(self, model: str) -> None:
        await self._ensure_thinking_behavior()

    def _on_keyless_rate_limit(self, error: ProviderError, request: LLMRequest) -> None:
        # Ollama Cloud enforces a monthly quota, so retrying cannot clear a
        # 429. Raise immediately so the pipeline pauses and checkpoints
        # (issue #279). Without a server hint, leave retry_after unset: the
        # header-less fallback (a few seconds) would hammer an exhausted
        # quota, while None lets the caller use its own auto-resume delay.
        headers = error.headers
        hinted = any(
            header in headers
            for header in ("Retry-After", "retry-after", "X-RateLimit-Reset", "x-ratelimit-reset")
        )
        retry_after = compute_wait_time(headers, 0) if hinted else None
        self._log("llm_rate_limit",
                  f"⚠️ Rate limit reached (HTTP 429) for {request.model}: {error.message}. "
                  f"Translation is being paused - a checkpoint is saved so you can resume later")
        raise RateLimitError(error.message, retry_after=retry_after, provider="ollama")

    def _build_payload(self, request: LLMRequest, context_window: int) -> dict:
        # Think parameter by behavior:
        # - UNCONTROLLABLE: think=true to cleanly separate thinking into its field
        # - CONTROLLABLE / STANDARD: think=false
        # Only sent when the model supports the parameter at all.
        #
        # No temperature is sent: local models keep the sampling parameters of
        # their Modelfile. Thinking models ship tuned values there, and their
        # model cards warn that low temperatures cause endless repetition.
        payload = {
            "model": request.model,
            "messages": request.messages(),
            "stream": True,  # Streaming for real-time token monitoring
            "options": {
                "num_ctx": context_window,
                "truncate": False
            },
        }
        if self._supports_think_param:
            payload["think"] = self._thinking_behavior == ThinkingBehavior.UNCONTROLLABLE
        return payload

    async def _send(self, request: LLMRequest) -> LLMResponse:
        """
        One Ollama chat request, streamed for real-time token monitoring.

        Streaming lets us catch a context overflow while it happens (Ollama
        does not error when the window is exceeded - it keeps generating
        garbage) and abort a repetition loop early.
        """
        context_window = request.context_window or self.context_window
        payload = self._build_payload(request, context_window)
        client = await self._get_client()

        content_chunks = []
        thinking_chunks = []
        prompt_tokens = 0
        completion_tokens = 0
        abort_reason = None
        done_reason = None

        # Safe limit for completion tokens, refined once the prompt size is known.
        max_completion_tokens = int(context_window * 0.85)

        async with client.stream("POST", self.api_endpoint, json=payload,
                                 timeout=request.timeout) as response:
            if response.is_error:
                # The body of a streamed response is not fetched yet: read it
                # while the stream is open so Ollama's error message (num_ctx
                # overflow, unknown model, ...) survives into the classification.
                await response.aread()
                raise provider_error_from_response(response)

            try:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Prompt tokens arrive once, in the first chunk.
                    if chunk_data.get("prompt_eval_count"):
                        prompt_tokens = chunk_data["prompt_eval_count"]
                        max_completion_tokens = int((context_window - prompt_tokens) * 0.90)

                    message = chunk_data.get("message", {})
                    if message.get("content"):
                        content_chunks.append(message["content"])
                    if message.get("thinking"):
                        thinking_chunks.append(message["thinking"])

                    if chunk_data.get("eval_count"):
                        completion_tokens = chunk_data["eval_count"]

                    # Context overflow during streaming (~3 chars per token).
                    current_content = "".join(content_chunks)
                    current_thinking = "".join(thinking_chunks)
                    estimated_tokens = (len(current_content) + len(current_thinking)) // 3
                    if estimated_tokens > max_completion_tokens:
                        abort_reason = (f"estimated {estimated_tokens} tokens exceeds the safe "
                                        f"limit {max_completion_tokens} (context: {context_window})")
                        break

                    # Repetition checks every ~500 chars, with the streaming
                    # thresholds (slightly more sensitive for early detection).
                    if len(current_content) > 500 and len(current_content) % 500 < 50:
                        if detect_repetition_loop(
                            current_content,
                            min_repetitions=REPETITION_MIN_COUNT_STREAMING,
                            is_thinking_content=False
                        ):
                            abort_reason = "repetition loop detected in content"
                            break

                    # Thinking content gets more lenient detection.
                    if len(current_thinking) > 800 and len(current_thinking) % 500 < 50:
                        if detect_repetition_loop(
                            current_thinking,
                            min_repetitions=REPETITION_MIN_COUNT_STREAMING,
                            is_thinking_content=True
                        ):
                            abort_reason = "repetition loop detected in thinking"
                            break

                    if chunk_data.get("done"):
                        prompt_tokens = chunk_data.get("prompt_eval_count", prompt_tokens)
                        completion_tokens = chunk_data.get("eval_count", completion_tokens)
                        done_reason = chunk_data.get("done_reason")
                        break
            finally:
                # Close the stream so no "Task was destroyed but it is pending".
                await response.aclose()

        if abort_reason:
            self._log("stream_abort", f"[STREAM ABORT] {abort_reason}")
            # Caller retries with a larger context.
            raise RepetitionLoopError(
                f"Context overflow detected during streaming ({abort_reason}). "
                f"Context window ({context_window}) is too small. "
                f"Prompt used ~{prompt_tokens} tokens, leaving insufficient space for response."
            )

        content = "".join(content_chunks)
        thinking = "".join(thinking_chunks)

        # Ollama's eval_count may or may not include thinking tokens depending
        # on the version: estimate (~3 chars per token) and keep the larger
        # value, since underestimating causes premature context reduction.
        thinking_tokens_estimate = len(thinking) // 3 if thinking else 0
        content_tokens_estimate = len(content) // 3 if content else 0
        total_completion_estimate = thinking_tokens_estimate + content_tokens_estimate
        effective_completion_tokens = max(completion_tokens, total_completion_estimate)

        if thinking and effective_completion_tokens > completion_tokens * 1.2:
            self._log("token_usage_warning",
                      f"⚠️ Thinking tokens likely not in eval_count: reported={completion_tokens}, "
                      f"estimated={total_completion_estimate} (thinking~{thinking_tokens_estimate}, "
                      f"content~{content_tokens_estimate})", level=logging.DEBUG)

        context_used = prompt_tokens + effective_completion_tokens
        was_truncated = context_used >= context_window * 0.95

        if self.log_callback:
            status = "⚠️ NEAR LIMIT" if was_truncated else "✓"
            thinking_info = f" (incl. ~{thinking_tokens_estimate} thinking)" if thinking else ""
            self.log_callback("token_usage",
                f"Tokens: prompt={prompt_tokens}, response={effective_completion_tokens}{thinking_info}, "
                f"total={context_used}/{context_window} {status}")

        if thinking:
            self._logger.debug(f"[THINKING FIELD] {request.model} produced thinking "
                               f"({len(thinking)} chars):\n{thinking}")
        if "<think>" in content.lower():
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
            if think_match:
                self._logger.debug(f"[THINK BLOCK IN CONTENT] {request.model} embedded thinking "
                                   f"({len(think_match.group(1))} chars):\n{think_match.group(1)}")

        # Final repetition check on the complete response.
        loop_detected_in = None
        if thinking and detect_repetition_loop(thinking, is_thinking_content=True):
            loop_detected_in = "thinking"
        elif content and detect_repetition_loop(content, is_thinking_content=False):
            loop_detected_in = "content"
        if loop_detected_in:
            error_msg = (
                f"Repetition loop detected in {loop_detected_in}! "
                f"This usually means the context window ({context_window}) is too small for "
                f"thinking models. Try increasing OLLAMA_NUM_CTX or reducing chunk size."
            )
            self._log("repetition_loop", f"[REPETITION LOOP DETECTED] {error_msg}")
            raise RepetitionLoopError(error_msg)

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=effective_completion_tokens,  # Includes the thinking estimate
            context_used=context_used,
            context_limit=context_window,
            was_truncated=was_truncated,
            finish_reason=done_reason,
            output_truncated=is_output_limit_finish(done_reason)
        )

    async def get_model_context_size(self) -> int:
        """
        Query Ollama API to get the model's context size.

        Returns:
            int: Maximum context size in tokens
        """
        try:
            client = await self._get_client()
            # Use ContextDetector for Ollama-specific detection
            return await self._context_detector.detect_ollama(
                client=client,
                model=self.model,
                endpoint=self.api_endpoint,
                log_callback=self.log_callback,
                fallback_context=self.context_window
            )
        except Exception as e:
            if self.log_callback:
                self.log_callback("warning",
                    f"Failed to query model context size: {e}. Using configured value.")
            return self.context_window


