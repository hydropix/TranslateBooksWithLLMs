"""ChatGPT provider — OAuth access token against the Codex Responses backend."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import json
import uuid

import httpx

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS
from src.core.llm.chatgpt_oauth import (
    CODEX_CLIENT_VERSION,
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    ensure_fresh_tokens,
    request_headers,
    save_tokens,
)
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError
from ..rate_limit_handler import is_retryable_http_status


def _models_request_url() -> str:
    return f"{CODEX_MODELS_URL}?client_version={CODEX_CLIENT_VERSION}"


def parse_sse_response(lines: List[str]) -> Tuple[str, Dict[str, Any], bool]:
    """Collect output text and usage from Codex Responses SSE frames."""
    text_parts: List[str] = []
    completed_text = ""
    usage: Dict[str, Any] = {}
    was_truncated = False
    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw.strip() in ("", "[DONE]"):
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type") or "")
        if etype == "response.output_text.delta":
            delta = event.get("delta") or event.get("text") or ""
            if delta:
                text_parts.append(str(delta))
        elif etype == "response.output_text.done":
            done_text = event.get("text") or ""
            if done_text:
                completed_text = str(done_text)
        elif etype == "response.completed":
            resp_obj = event.get("response") or {}
            if isinstance(resp_obj, dict):
                usage = resp_obj.get("usage") or {}
                if str(resp_obj.get("status") or "") == "incomplete":
                    was_truncated = True
        elif etype == "response.failed":
            resp_obj = event.get("response") or {}
            error = (resp_obj.get("error") if isinstance(resp_obj, dict) else None) or event.get("error")
            message = ""
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or "")
            elif error:
                message = str(error)
            if any(word in message.lower() for word in ("context", "token limit", "too long")):
                raise ContextOverflowError(message or "ChatGPT context overflow")
    text = "".join(text_parts) or completed_text
    return text, usage if isinstance(usage, dict) else {}, was_truncated


class ChatGPTProvider(LLMProvider):
    FALLBACK_MODELS = [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
    ]

    def __init__(
        self,
        model: str = FALLBACK_MODELS[0],
        context_window: Optional[int] = None,
        log_callback: Optional[Callable] = None,
        tokens: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model, api_keys=None, provider_name="chatgpt")
        self.context_window = context_window or 128000
        self.log_callback = log_callback
        self._tokens = tokens

    async def _fresh_tokens(self) -> Dict[str, Any]:
        if self._tokens and self._tokens.get("access_token"):
            return self._tokens
        self._tokens = await ensure_fresh_tokens()
        return self._tokens

    def _extract_output_text(self, body: Dict[str, Any]) -> str:
        direct = body.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        chunks: List[str] = []
        output = body.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") not in (None, "message", "output_text"):
                    continue
                content = item.get("content")
                if isinstance(content, str) and content:
                    chunks.append(content)
                    continue
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, str) and part:
                            chunks.append(part)
                        elif isinstance(part, dict):
                            text = part.get("text") or part.get("content") or ""
                            if text and str(part.get("type") or "") not in ("reasoning", "thinking"):
                                chunks.append(str(text))
        return "".join(chunks)

    async def get_available_models(self) -> list:
        try:
            tokens = await self._fresh_tokens()
            response = await (await self._get_client()).get(
                _models_request_url(),
                headers=request_headers(tokens),
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models") or payload.get("data") or payload.get("items") or []
            parsed = []
            if isinstance(models, list):
                for item in models:
                    if isinstance(item, str):
                        parsed.append({"id": item, "name": item})
                    elif isinstance(item, dict):
                        visibility = str(item.get("visibility") or "list").lower()
                        if visibility in ("hidden", "none"):
                            continue
                        model_id = item.get("slug") or item.get("id") or item.get("model") or ""
                        if model_id:
                            parsed.append({
                                "id": model_id,
                                "name": item.get("display_name") or item.get("title") or item.get("name") or model_id,
                            })
            if parsed:
                return parsed
        except Exception:
            pass
        return [{"id": m, "name": m} for m in self.FALLBACK_MODELS]

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                       system_prompt: Optional[str] = None,
                       **generation_options) -> Optional[LLMResponse]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }],
            "store": False,
            "stream": True,
        }
        if system_prompt:
            payload["instructions"] = system_prompt
        max_tokens = generation_options.get("max_tokens")
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        temperature = generation_options.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature

        client = await self._get_client()
        attempt = 0
        while attempt < MAX_TRANSLATION_ATTEMPTS:
            try:
                tokens = await self._fresh_tokens()
                headers = request_headers(tokens)
                headers["session_id"] = str(uuid.uuid4())
                async with client.stream(
                    "POST",
                    CODEX_RESPONSES_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                ) as response:
                    if response.status_code == 401:
                        from src.core.llm.chatgpt_oauth import refresh_tokens
                        refreshed = await refresh_tokens(tokens["refresh_token"])
                        save_tokens(refreshed)
                        self._tokens = refreshed
                        attempt += 1
                        continue
                    response.raise_for_status()
                    lines = [line async for line in response.aiter_lines()]
                text, usage, was_truncated = parse_sse_response(lines)
                prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
                if self.log_callback and (prompt_tokens or completion_tokens):
                    self.log_callback(
                        "token_usage",
                        f"Tokens: prompt={prompt_tokens}, response={completion_tokens}, "
                        f"total={prompt_tokens + completion_tokens}",
                    )
                if not text.strip():
                    attempt += 1
                    if attempt < MAX_TRANSLATION_ATTEMPTS:
                        await asyncio.sleep(2)
                    continue
                return LLMResponse(
                    content=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=self.context_window,
                    was_truncated=was_truncated,
                )
            except httpx.TimeoutException:
                attempt += 1
            except httpx.HTTPStatusError as exc:
                message = str(exc)
                if any(word in message.lower() for word in ("context", "token limit", "too long")):
                    raise ContextOverflowError(message) from exc
                if exc.response is not None and not is_retryable_http_status(exc.response.status_code):
                    return None
                attempt += 1
            except ContextOverflowError:
                raise
            except (ValueError, httpx.DecodingError):
                attempt += 1
            if attempt < MAX_TRANSLATION_ATTEMPTS:
                await asyncio.sleep(2)
        return None
