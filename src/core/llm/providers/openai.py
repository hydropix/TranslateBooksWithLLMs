"""
OpenAI-compatible provider implementation.

This module provides the OpenAICompatibleProvider class for interacting with
OpenAI API and compatible endpoints (llama.cpp, LM Studio, vLLM, OpenAI, etc.).
"""

from typing import Optional, Callable
import asyncio
import json
import httpx

from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError
from ..utils.context_detection import ContextDetector

from src.config import (
    REQUEST_TIMEOUT,
    OLLAMA_NUM_CTX,
    MAX_TRANSLATION_ATTEMPTS
)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible API provider (works with llama.cpp, LM Studio, vLLM, OpenAI, etc.)"""

    def __init__(self, api_endpoint: str, model: str, api_key: Optional[str] = None,
                 context_window: int = OLLAMA_NUM_CTX, log_callback: Optional[Callable] = None):
        super().__init__(model)
        self.api_endpoint = self._normalize_endpoint(api_endpoint)
        self.api_key = api_key
        self.context_window = context_window
        self.log_callback = log_callback
        self._detected_context_size: Optional[int] = None
        self._context_detector = ContextDetector()

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """
        Normalize API endpoint URL for OpenAI-compatible APIs.
        
        Automatically adds '/chat/completions' if the URL ends with '/v1' or '/v1/'
        but not with the full path. This handles common user mistakes like:
        - http://localhost:11434/v1 -> http://localhost:11434/v1/chat/completions
        - https://api.example.com/v1/ -> https://api.example.com/v1/chat/completions
        
        Args:
            endpoint: Raw endpoint URL provided by user
            
        Returns:
            Normalized endpoint URL with complete path
        """
        if not endpoint:
            return endpoint
        
        # Remove trailing slash for consistent processing
        endpoint = endpoint.rstrip('/')
        
        # If already ends with /v1/chat/completions, keep as-is
        if endpoint.endswith('/v1/chat/completions'):
            return endpoint
        
        # If ends with /v1, append /chat/completions
        if endpoint.endswith('/v1'):
            return endpoint + '/chat/completions'
        
        # Otherwise return as-is (user provided custom path)
        return endpoint

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                      system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """
        Generate text using an OpenAI compatible API.

        Args:
            prompt: The user prompt (content to translate)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Build messages array with optional system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Disable thinking/reasoning mode for local servers and compatible APIs
            # This prevents models from outputting <think>...</think> blocks
            "thinking": False,
            "enable_thinking": False,
            # Some servers use chat_template_kwargs
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

        client = await self._get_client()
        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
            try:
                response = await client.post(
                    self.api_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                response.raise_for_status()

                response_json = response.json()
                response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Extract token usage if available
                usage = response_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=self.context_window,
                    was_truncated=False  # OpenAI API doesn't provide truncation info
                )

            except httpx.TimeoutException as e:
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    self.log_callback("llm_timeout",
                        f"{YELLOW}⚠️ LLM request timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Endpoint: {self.api_endpoint}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Possible causes:{RESET}\n"
                        f"{YELLOW}   - llama.cpp/LM Studio server crashed or became unresponsive{RESET}\n"
                        f"{YELLOW}   - Server overloaded or out of memory{RESET}\n"
                        f"{YELLOW}   - Network connectivity issues{RESET}\n"
                        f"{YELLOW}   - Model too large for available VRAM{RESET}")
                else:
                    print(f"{YELLOW}⚠️ OpenAI-compatible API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}{RESET}")

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
                        f"{RED}   1. Check if llama.cpp/LM Studio server is running{RESET}\n"
                        f"{RED}   2. Verify server is accessible at: {self.api_endpoint}{RESET}\n"
                        f"{RED}   3. Check server console/logs for crashes or OOM errors{RESET}\n"
                        f"{RED}   4. Try reducing context size or chunk size{RESET}\n"
                        f"{RED}   5. Ensure model fits in available VRAM/RAM{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None
            except httpx.HTTPStatusError as e:
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                error_message = str(e)
                error_body = ""
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    error_body = e.response.text[:500]
                    try:
                        # Try to parse JSON error for better messages
                        error_json = e.response.json()
                        if "error" in error_json:
                            if isinstance(error_json["error"], dict):
                                error_message = error_json["error"].get("message", str(e))
                            else:
                                error_message = str(error_json.get("error", e))
                    except:
                        error_message = f"{e} - {error_body}"

                # Detect context overflow errors
                context_overflow_keywords = ["context_length", "maximum context", "token limit",
                                              "too many tokens", "reduce the length", "max_tokens",
                                              "context", "truncate", "length", "too long"]
                if any(keyword in error_message.lower() for keyword in context_overflow_keywords):
                    RED = '\033[91m'
                    YELLOW = '\033[93m'
                    RESET = '\033[0m'

                    if self.log_callback:
                        self.log_callback("llm_context_overflow",
                            f"{RED}❌ Context size exceeded!{RESET}\n"
                            f"{RED}   Prompt is too large for model's context window{RESET}\n"
                            f"{RED}   Current context window: {self.context_window} tokens{RESET}\n"
                            f"{RED}   Error: {error_message}{RESET}\n"
                            f"{YELLOW}   Solutions:{RESET}\n"
                            f"{YELLOW}   1. Reduce max_tokens_per_chunk (current chunk may be too large){RESET}\n"
                            f"{YELLOW}   2. Increase context size in server configuration{RESET}\n"
                            f"{YELLOW}   3. Use a model with larger context window{RESET}\n"
                            f"{YELLOW}   4. For llama.cpp: increase -c/--ctx-size parameter{RESET}")
                    else:
                        print(f"{RED}Context size exceeded: {error_message}{RESET}")
                    raise ContextOverflowError(error_message)

                # Handle other HTTP errors with detailed information
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    status_code = e.response.status_code if e.response else 'unknown'
                    self.log_callback("llm_http_error",
                        f"{YELLOW}⚠️ HTTP error from LLM server (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Endpoint: {self.api_endpoint}{RESET}\n"
                        f"{YELLOW}   Status: {e.response.status_code if e.response else 'unknown'}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Error: {error_message}{RESET}")
                    if error_body:
                        self.log_callback("llm_http_error_detail", f"{YELLOW}   Response: {error_body[:200]}...{RESET}")
                else:
                    print(f"{YELLOW}⚠️ OpenAI-compatible API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}{RESET}")
                    if error_body:
                        print(f"{YELLOW}   Response: {error_body[:200]}...{RESET}")

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue

                # All retries exhausted
                if self.log_callback:
                    self.log_callback("llm_http_error_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   HTTP error persists - translation failed{RESET}\n"
                        f"{RED}   Status: {e.response.status_code if e.response else 'unknown'}{RESET}\n"
                        f"{RED}   Check server logs for more details{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None
            except json.JSONDecodeError as e:
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    self.log_callback("llm_json_error",
                        f"{YELLOW}⚠️ Invalid JSON response from LLM (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Endpoint: {self.api_endpoint}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Error: {str(e)}{RESET}\n"
                        f"{YELLOW}   This may indicate:{RESET}\n"
                        f"{YELLOW}   - Server returned malformed response{RESET}\n"
                        f"{YELLOW}   - llama.cpp server crashed mid-response{RESET}\n"
                        f"{YELLOW}   - API endpoint incompatibility{RESET}\n"
                        f"{YELLOW}   - Server configuration issues{RESET}")
                else:
                    print(f"{YELLOW}⚠️ OpenAI-compatible API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}{RESET}")

                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue

                if self.log_callback:
                    self.log_callback("llm_json_error_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   Unable to parse LLM response - translation failed{RESET}\n"
                        f"{RED}   Verify server is running and endpoint is correct{RESET}")
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
                        f"{YELLOW}   Endpoint: {self.api_endpoint}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Error type: {type(e).__name__}{RESET}\n"
                        f"{YELLOW}   Error: {str(e)}{RESET}")
                else:
                    print(f"{YELLOW}⚠️ OpenAI-compatible API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {type(e).__name__}: {e}{RESET}")

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
        """Query server to get model's context size using ContextDetector."""
        if self._detected_context_size:
            return self._detected_context_size

        client = await self._get_client()
        ctx = await self._context_detector.detect(
            client=client,
            model=self.model,
            endpoint=self.api_endpoint,
            api_key=self.api_key,
            log_callback=self.log_callback
        )

        self._detected_context_size = ctx
        return ctx
