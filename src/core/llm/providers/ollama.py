"""
Ollama provider implementation.

This module provides the OllamaProvider class for interacting with local
Ollama servers with full thinking model detection and handling.
"""

from typing import Optional, Callable
import asyncio
import json
import re
import httpx

from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError, RepetitionLoopError
from ..thinking.cache import get_thinking_cache
from ..thinking.detection import detect_repetition_loop
from ..thinking.behavior import ThinkingBehavior, _model_matches_pattern
from ..utils.context_detection import ContextDetector

from src.config import (
    API_ENDPOINT,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT,
    OLLAMA_NUM_CTX,
    MAX_TRANSLATION_ATTEMPTS,
    UNCONTROLLABLE_THINKING_MODELS,
    CONTROLLABLE_THINKING_MODELS,
    REPETITION_MIN_COUNT_STREAMING
)


class OllamaProvider(LLMProvider):
    """Ollama API provider - uses /api/chat for proper think parameter support"""

    def __init__(self, api_endpoint: str = API_ENDPOINT, model: str = DEFAULT_MODEL,
                 context_window: int = OLLAMA_NUM_CTX, log_callback: Optional[Callable] = None):
        super().__init__(model)
        # Convert /api/generate endpoint to /api/chat for proper think support
        self.api_endpoint = api_endpoint.replace('/api/generate', '/api/chat')
        self.context_window = context_window
        self.log_callback = log_callback
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
        """Display warning for uncontrollable thinking models."""
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

        print(f"\n{RED}{'='*70}{RESET}")
        print(f"{RED}{BOLD}[WARNING] UNCONTROLLABLE THINKING MODEL: {self.model}{RESET}")
        print(f"{RED}{'='*70}{RESET}")
        print(f"{YELLOW}This model produces <think> blocks that CANNOT be disabled.{RESET}")
        print(f"{YELLOW}Consequences:{RESET}")
        print(f"{YELLOW}  - SLOWER translations (model thinks before answering){RESET}")
        print(f"{YELLOW}  - MORE tokens consumed (reasoning uses context window){RESET}")
        print(f"{YELLOW}  - LESS consistent results for translation tasks{RESET}")
        print()

        # Suggest alternatives based on model
        model_lower = self.model.lower()
        if "qwen3" in model_lower and "instruct" not in model_lower:
            size_match = re.search(r':(\d+b)', model_lower)
            size = size_match.group(1) if size_match else ""
            if size:
                print(f"{GREEN}{BOLD}RECOMMENDATION: Use 'qwen3:{size}-instruct' instead{RESET}")
                print(f"{GREEN}  → Instruct models give direct answers without thinking{RESET}")
                print(f"{GREEN}  → Same quality, faster speed, less token usage{RESET}")
            else:
                print(f"{GREEN}{BOLD}RECOMMENDATION: Use a Qwen3 instruct variant{RESET}")
                print(f"{GREEN}  → Example: qwen3:14b-instruct, qwen3:30b-instruct{RESET}")
        elif "phi4-reasoning" in model_lower:
            print(f"{GREEN}{BOLD}RECOMMENDATION: Use 'phi4:latest' instead{RESET}")
            print(f"{GREEN}  → Standard Phi4 doesn't use reasoning mode{RESET}")
        elif "deepseek" in model_lower or "qwq" in model_lower:
            print(f"{GREEN}{BOLD}RECOMMENDATION: Use a non-reasoning model{RESET}")
            print(f"{GREEN}  → Reasoning models are for complex problems, not translation{RESET}")

        print(f"{RED}{'='*70}{RESET}\n")
        print(f"{CYAN}[INFO] Using think=true to cleanly separate thinking from content{RESET}\n")

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                      system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """
        Generate text using Ollama Chat API with streaming for real-time token monitoring.

        Uses streaming to detect context overflow in real-time (Ollama doesn't return
        errors when context is exceeded - it just keeps generating garbage).

        Args:
            prompt: The user prompt (content to translate)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed
        """
        # Detect thinking behavior on first request
        if self._thinking_behavior is None:
            self._thinking_behavior = await self._detect_thinking_behavior()

            # Show warning only for uncontrollable thinking models
            if self._thinking_behavior == ThinkingBehavior.UNCONTROLLABLE and self.log_callback:
                self._show_thinking_warning()
            elif self._thinking_behavior == ThinkingBehavior.CONTROLLABLE and self.log_callback:
                GREEN = '\033[92m'
                RESET = '\033[0m'
                print(f"\n{GREEN}[MODEL] {self.model}: Controllable thinking model - using think=false{RESET}")
            elif self._thinking_behavior == ThinkingBehavior.STANDARD and self.log_callback:
                GREEN = '\033[92m'
                RESET = '\033[0m'
                print(f"\n{GREEN}[MODEL] {self.model}: Standard model (no thinking){RESET}")

        # Build messages array for chat API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Determine think parameter based on behavior:
        # - UNCONTROLLABLE: use think=true to cleanly separate thinking into dedicated field
        # - CONTROLLABLE: use think=false to disable thinking
        # - STANDARD: don't include think param (model doesn't support it)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,  # Enable streaming for real-time token monitoring
            "options": {
                "num_ctx": self.context_window,
                "truncate": False
            },
        }

        # Only add think param if model supports it
        if self._supports_think_param:
            if self._thinking_behavior == ThinkingBehavior.UNCONTROLLABLE:
                # For uncontrollable models, use think=true to get clean separation
                payload["think"] = True
            else:
                # For controllable and standard models, use think=false
                payload["think"] = False

        client = await self._get_client()
        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
            try:
                # Use streaming to monitor tokens in real-time
                content_chunks = []
                thinking_chunks = []
                prompt_tokens = 0
                completion_tokens = 0
                exceeded_context = False

                # Calculate safe limit for completion tokens
                # Reserve space for prompt (we'll get actual count from first chunk)
                # Use 90% of remaining context as safety margin
                max_completion_tokens = int(self.context_window * 0.85)

                async with client.stream("POST", self.api_endpoint, json=payload, timeout=timeout) as response:
                    response.raise_for_status()

                    try:
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue

                            try:
                                chunk_data = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            # Get prompt tokens from first chunk (Ollama sends this once)
                            if chunk_data.get("prompt_eval_count"):
                                prompt_tokens = chunk_data["prompt_eval_count"]
                                # Recalculate max completion tokens based on actual prompt size
                                max_completion_tokens = int((self.context_window - prompt_tokens) * 0.90)

                            # Accumulate content
                            message = chunk_data.get("message", {})
                            if message.get("content"):
                                content_chunks.append(message["content"])
                            if message.get("thinking"):
                                thinking_chunks.append(message["thinking"])

                            # Update completion token count
                            if chunk_data.get("eval_count"):
                                completion_tokens = chunk_data["eval_count"]

                            # Check for context overflow during streaming
                            # This catches the case where Ollama keeps generating past the limit
                            current_completion_len = len("".join(content_chunks)) + len("".join(thinking_chunks))

                            # Heuristic: ~4 chars per token on average
                            estimated_tokens = current_completion_len // 3

                            if estimated_tokens > max_completion_tokens:
                                exceeded_context = True
                                if self.log_callback:
                                    RED = '\033[91m'
                                    RESET = '\033[0m'
                                    print(f"\n{RED}[STREAM ABORT] Estimated {estimated_tokens} tokens exceeds "
                                          f"safe limit {max_completion_tokens} (context: {self.context_window}){RESET}")
                                break

                            # Also check for repetition in real-time during streaming
                            current_content = "".join(content_chunks)
                            current_thinking = "".join(thinking_chunks)

                            # Only check periodically (every ~500 chars) to avoid overhead
                            # Use streaming thresholds (slightly more sensitive for early detection)
                            if len(current_content) > 500 and len(current_content) % 500 < 50:
                                if detect_repetition_loop(
                                    current_content,
                                    min_repetitions=REPETITION_MIN_COUNT_STREAMING,
                                    is_thinking_content=False
                                ):
                                    exceeded_context = True
                                    if self.log_callback:
                                        RED = '\033[91m'
                                        RESET = '\033[0m'
                                        print(f"\n{RED}[STREAM ABORT] Repetition loop detected in content{RESET}")
                                    break

                            # For thinking content, use more lenient detection
                            if len(current_thinking) > 800 and len(current_thinking) % 500 < 50:
                                if detect_repetition_loop(
                                    current_thinking,
                                    min_repetitions=REPETITION_MIN_COUNT_STREAMING,
                                    is_thinking_content=True
                                ):
                                    exceeded_context = True
                                    if self.log_callback:
                                        RED = '\033[91m'
                                        RESET = '\033[0m'
                                        print(f"\n{RED}[STREAM ABORT] Repetition loop detected in thinking{RESET}")
                                    break

                            # Check if stream is done
                            if chunk_data.get("done"):
                                # Get final token counts
                                prompt_tokens = chunk_data.get("prompt_eval_count", prompt_tokens)
                                completion_tokens = chunk_data.get("eval_count", completion_tokens)
                                break
                    finally:
                        # Ensure the stream is properly closed and all data is consumed
                        # This prevents "Task was destroyed but it is pending" errors
                        await response.aclose()

                # If we exceeded context, raise error for retry with larger context
                if exceeded_context:
                    raise RepetitionLoopError(
                        f"Context overflow detected during streaming. "
                        f"Context window ({self.context_window}) is too small. "
                        f"Prompt used ~{prompt_tokens} tokens, leaving insufficient space for response."
                    )

                # Combine chunks
                content = "".join(content_chunks)
                thinking = "".join(thinking_chunks)

                # Estimate thinking tokens if present
                # Ollama's eval_count may or may not include thinking tokens depending on version
                # We estimate thinking tokens (~3.5 chars per token for English text) and use
                # the MAX of reported completion_tokens or our estimate to be safe
                thinking_tokens_estimate = len(thinking) // 3 if thinking else 0
                content_tokens_estimate = len(content) // 3 if content else 0
                total_completion_estimate = thinking_tokens_estimate + content_tokens_estimate

                # Use the higher value to avoid underestimating (which causes premature context reduction)
                effective_completion_tokens = max(completion_tokens, total_completion_estimate)

                # If there's a significant difference, the reported count likely excludes thinking
                if thinking and effective_completion_tokens > completion_tokens * 1.2:
                    if self.log_callback:
                        self.log_callback("token_usage_warning",
                            f"⚠️ Thinking tokens likely not in eval_count: reported={completion_tokens}, "
                            f"estimated={total_completion_estimate} (thinking~{thinking_tokens_estimate}, content~{content_tokens_estimate})")

                context_used = prompt_tokens + effective_completion_tokens

                # Detect if context was nearly exhausted
                truncation_threshold = self.context_window * 0.95
                was_truncated = context_used >= truncation_threshold

                # Log token usage
                if self.log_callback:
                    status = "⚠️ NEAR LIMIT" if was_truncated else "✓"
                    thinking_info = f" (incl. ~{thinking_tokens_estimate} thinking)" if thinking else ""
                    self.log_callback("token_usage",
                        f"Tokens: prompt={prompt_tokens}, response={effective_completion_tokens}{thinking_info}, "
                        f"total={context_used}/{self.context_window} {status}")

                # Log thinking content if present
                CYAN = '\033[96m'
                RESET = '\033[0m'

                if thinking and self.log_callback:
                    print(f"\n{CYAN}{'='*80}")
                    print(f"[THINKING FIELD] Model produced thinking ({len(thinking)} chars):")
                    print(f"{thinking}")
                    print(f"{'='*80}{RESET}\n")

                # Check for <think> blocks in content
                if "<think>" in content.lower() and self.log_callback:
                    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
                    if think_match:
                        think_content = think_match.group(1)
                        print(f"\n{CYAN}{'='*80}")
                        print(f"[THINK BLOCK IN CONTENT] Model embedded thinking ({len(think_content)} chars):")
                        print(f"{think_content}")
                        print(f"{'='*80}{RESET}\n")

                # Final repetition loop check on complete response
                # Use appropriate thresholds for thinking vs content
                loop_detected_in = None
                if thinking and detect_repetition_loop(thinking, is_thinking_content=True):
                    loop_detected_in = "thinking"
                elif content and detect_repetition_loop(content, is_thinking_content=False):
                    loop_detected_in = "content"

                if loop_detected_in:
                    RED = '\033[91m'
                    error_msg = (
                        f"Repetition loop detected in {loop_detected_in}! "
                        f"This usually means the context window ({self.context_window}) is too small for thinking models. "
                        f"Try increasing OLLAMA_NUM_CTX or reducing chunk size."
                    )
                    print(f"\n{RED}{'='*80}")
                    print(f"[REPETITION LOOP DETECTED] {error_msg}")
                    print(f"{'='*80}{RESET}\n")
                    raise RepetitionLoopError(error_msg)

                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=effective_completion_tokens,  # Use effective count (includes thinking estimate)
                    context_used=context_used,
                    context_limit=self.context_window,
                    was_truncated=was_truncated
                )

            except httpx.TimeoutException as e:
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    self.log_callback("llm_timeout",
                        f"{YELLOW}⚠️ LLM request timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Possible causes:{RESET}\n"
                        f"{YELLOW}   - Model crashed or became unresponsive{RESET}\n"
                        f"{YELLOW}   - Server overloaded or out of memory{RESET}\n"
                        f"{YELLOW}   - Network connectivity issues{RESET}")
                else:
                    print(f"{YELLOW}⚠️ LLM timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}{RESET}")

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue

                # All retry attempts exhausted
                if self.log_callback:
                    self.log_callback("llm_timeout_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   Translation failed - unable to reach LLM server{RESET}\n"
                        f"{RED}   Recommendations:{RESET}\n"
                        f"{RED}   1. Check if Ollama/llama.cpp server is running{RESET}\n"
                        f"{RED}   2. Verify model is loaded: ollama list{RESET}\n"
                        f"{RED}   3. Check server logs for crashes{RESET}\n"
                        f"{RED}   4. Try reducing context size or chunk size{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None
            except httpx.HTTPStatusError as e:
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                error_message = str(e)
                if e.response:
                    try:
                        error_data = e.response.json()
                        error_message = error_data.get("error", str(e))
                    except Exception:
                        pass

                # Handle context overflow errors
                if any(keyword in error_message.lower()
                       for keyword in ["context", "truncate", "length", "too long"]):
                    if self.log_callback:
                        self.log_callback("llm_context_overflow",
                            f"{RED}❌ Context size exceeded!{RESET}\n"
                            f"{RED}   Prompt is too large for model's context window{RESET}\n"
                            f"{RED}   Current context window: {self.context_window} tokens{RESET}\n"
                            f"{RED}   Error: {error_message}{RESET}\n"
                            f"{YELLOW}   Solutions:{RESET}\n"
                            f"{YELLOW}   1. Reduce max_tokens_per_chunk (current chunk may be too large){RESET}\n"
                            f"{YELLOW}   2. Increase OLLAMA_NUM_CTX in .env file{RESET}\n"
                            f"{YELLOW}   3. Use a model with larger context window{RESET}")
                    else:
                        print(f"{RED}Context size exceeded: {error_message}{RESET}")
                    raise ContextOverflowError(error_message)

                # Handle other HTTP errors
                if self.log_callback:
                    self.log_callback("llm_http_error",
                        f"{YELLOW}⚠️ HTTP error from LLM server (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Status: {e.response.status_code if e.response else 'unknown'}{RESET}\n"
                        f"{YELLOW}   Error: {error_message}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}")
                else:
                    print(f"{YELLOW}HTTP error (attempt {attempt + 1}): {error_message}{RESET}")

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue

                # All retries exhausted
                if self.log_callback:
                    self.log_callback("llm_http_error_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   HTTP error persists - translation failed{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None
            except (RepetitionLoopError, ContextOverflowError):
                # These errors should propagate up for handling by translator
                raise
            except json.JSONDecodeError as e:
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    self.log_callback("llm_json_error",
                        f"{YELLOW}⚠️ Invalid JSON response from LLM (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Error: {str(e)}{RESET}\n"
                        f"{YELLOW}   This may indicate:{RESET}\n"
                        f"{YELLOW}   - Server returned malformed response{RESET}\n"
                        f"{YELLOW}   - Model output corrupted{RESET}\n"
                        f"{YELLOW}   - API endpoint incompatibility{RESET}")
                else:
                    print(f"{YELLOW}JSON decode error (attempt {attempt + 1}): {e}{RESET}")

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue

                if self.log_callback:
                    self.log_callback("llm_json_error_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   Unable to parse LLM response - translation failed{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None
            except Exception as e:
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    self.log_callback("llm_unexpected_error",
                        f"{YELLOW}⚠️ Unexpected error during LLM request (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Error type: {type(e).__name__}{RESET}\n"
                        f"{YELLOW}   Error: {str(e)}{RESET}")
                else:
                    print(f"{YELLOW}Unexpected error (attempt {attempt + 1}): {type(e).__name__}: {e}{RESET}")

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue

                if self.log_callback:
                    self.log_callback("llm_unexpected_error_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   Unexpected error persists - translation failed{RESET}\n"
                        f"{RED}   Please report this issue with the error details above{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None

        return None

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


