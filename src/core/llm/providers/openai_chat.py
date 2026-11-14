"""
Shared implementation of the OpenAI chat-completions wire format.

OpenAI (and every OpenAI-compatible server: llama.cpp, LM Studio, vLLM,
NVIDIA NIM), DeepSeek, Mistral, Poe and OpenRouter all speak the same
``POST .../chat/completions`` protocol. ``OpenAIChatProvider`` implements it
once; each provider module is a thin configuration on top of it:

    * ``API_URL`` / ``MODELS_URL``: endpoints;
    * ``MODEL_CONTEXT_SIZES`` / ``DEFAULT_CONTEXT_LIMIT``: context table;
    * ``_headers()``: extra request headers;
    * ``_fixed_params()``: payload entries always sent (thinking switches);
    * ``_optional_params()``: payload entries a model may reject (sampling,
      reasoning / bot overrides). On an HTTP 400 that names one of them, the
      request is re-sent once without them, consuming no attempt;
    * ``_parse_cost()``: billed cost, for gateways that return one.

The request loop itself (attempts, backoff, 429 rotation, error
classification, usage reporting) lives in ``LLMProvider``.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..base import (
    LLMProvider, LLMRequest, LLMResponse, is_output_limit_finish, provider_error_from_response,
)
from ..exceptions import ProviderError
from ..policy import RetryPolicy


def _error_in_ok_answer(body_error: Dict[str, Any]) -> ProviderError:
    """Classify an ``error`` object found in an HTTP 200 answer.

    OpenRouter sends 200 as soon as the upstream accepts the request, so a
    later failure lands in the body: only an ``error`` object (failed before
    any output), or partial ``choices`` with ``finish_reason: "error"``
    (failed mid-answer). ``error.code`` carries the HTTP-style status, so the
    failure is classified like that status (429 rate limit, 402 not
    retryable, 5xx retried); without a usable code it is retried. The
    partial text is never returned as an answer.
    """
    code = body_error.get("code")
    status: Optional[int] = None
    if isinstance(code, int) and not isinstance(code, bool):
        status = code
    elif isinstance(code, str) and code.strip().isdigit():
        status = int(code.strip())
    if status is not None and not 400 <= status <= 599:
        status = None
    metadata = body_error.get("metadata")
    error_type = metadata.get("error_type") if isinstance(metadata, dict) else None
    error_code = error_type or (None if status is not None or code is None else str(code))
    message = body_error.get("message") or str(body_error)[:300]
    return ProviderError(
        f"provider error in an HTTP 200 answer: {message}",
        status_code=status,
        error_code=str(error_code) if error_code is not None else None,
        retryable=True if status is None else None,
    )


class OpenAIChatProvider(LLMProvider):
    """Base for every provider speaking the OpenAI chat-completions protocol."""

    API_URL: str = ""
    MODELS_URL: str = ""
    #: Substring of a model id -> context window, first match wins.
    MODEL_CONTEXT_SIZES: Dict[str, int] = {}
    #: Context limit reported for a model missing from the table (0 = unknown).
    DEFAULT_CONTEXT_LIMIT: int = 0
    #: Whether ``temperature`` is sent (as an optional, droppable parameter).
    SEND_TEMPERATURE: bool = True
    #: On a 400 that names none of the optional parameters, drop them all
    #: anyway and retry once (for gateways whose 400s rarely name the field).
    DROP_ALL_OPTIONAL_ON_UNNAMED_400: bool = False
    #: Extra advice per HTTP status, appended to the failure log line.
    STATUS_HINTS: Dict[int, str] = {}

    def __init__(
        self,
        api_key: Optional[Union[str, List[str]]] = None,
        model: str = "",
        api_endpoint: Optional[str] = None,
        *,
        provider_name: str = "openai-compatible",
        context_window: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        temperature: Optional[float] = None,
        log_callback=None,
    ):
        super().__init__(
            model, api_keys=api_key, provider_name=provider_name,
            retry_policy=retry_policy, temperature=temperature, log_callback=log_callback,
        )
        self.api_endpoint = api_endpoint or self.API_URL
        self.context_window = context_window
        # Optional parameters a model rejected by name: never sent to that
        # model again by this instance (one failed 400 per run, not per chunk).
        self._rejected_optional: Dict[str, set] = {}

    # ------------------------------------------------------------ config
    def _context_limit_for(self, model_name: str) -> int:
        """Context window for ``model_name`` from ``MODEL_CONTEXT_SIZES``."""
        model_lower = (model_name or "").lower()
        for prefix, limit in self.MODEL_CONTEXT_SIZES.items():
            if prefix in model_lower:
                return limit
        return self.DEFAULT_CONTEXT_LIMIT

    def _get_context_limit(self) -> int:
        """Context window for the provider's own model."""
        return self._context_limit_for(self.model)

    def _response_context_limit(self, request: LLMRequest) -> int:
        """``LLMResponse.context_limit`` for this request."""
        return self._context_limit_for(request.model)

    def _headers(self, request: LLMRequest) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if request.api_key:
            headers["Authorization"] = f"Bearer {request.api_key}"
        return headers

    async def _fixed_params(self, request: LLMRequest) -> Dict[str, Any]:
        """Payload entries always sent for this request."""
        return {}

    async def _optional_params(self, request: LLMRequest) -> Dict[str, Any]:
        """Payload entries a model may reject; dropped once on a naming 400."""
        if self.SEND_TEMPERATURE:
            return {"temperature": self.effective_temperature()}
        return {}

    def _parse_cost(self, usage: Dict[str, Any], prompt_tokens: int,
                    completion_tokens: int) -> Tuple[Optional[float], bool]:
        """(billed cost in USD or None, cost is an estimate)."""
        return None, False

    def _on_optional_params_rejected(self, rejected: Iterable[str], request: LLMRequest) -> None:
        """Hook: the model named ``rejected`` optional parameters in a 400."""

    def _status_hint(self, status_code: Optional[int]) -> str:
        return self.STATUS_HINTS.get(status_code or 0) or super()._status_hint(status_code)

    # -------------------------------------------------------------- send
    async def _build_payload(self, request: LLMRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages(),
            "stream": False,
        }
        payload.update(await self._fixed_params(request))
        if "optional" not in request.state:
            rejected = self._rejected_optional.get(request.model, set())
            request.state["optional"] = {
                k: v for k, v in (await self._optional_params(request)).items()
                if k not in rejected
            }
        payload.update(request.state["optional"])
        return payload

    async def _send(self, request: LLMRequest) -> LLMResponse:
        payload = await self._build_payload(request)
        client = await self._get_client()
        response = await client.post(
            self.api_endpoint,
            headers=self._headers(request),
            json=payload,
            timeout=request.timeout,
        )
        if (getattr(response, "status_code", 200) or 200) >= 400:
            raise provider_error_from_response(response)
        return self._parse_response(response.json(), request)

    def _parse_response(self, data: Dict[str, Any], request: LLMRequest) -> LLMResponse:
        choices = data.get("choices") if isinstance(data, dict) else None
        body_error = data.get("error") if isinstance(data, dict) else None
        if isinstance(body_error, dict):
            finish = ((choices or [{}])[0] or {}).get("finish_reason")
            if not choices or finish == "error":
                raise _error_in_ok_answer(body_error)
        if not choices:
            raise ProviderError(
                f"unexpected response format (no choices): {str(data)[:300]}",
                retryable=False,
            )
        choice = choices[0] or {}
        # A present-but-null "content" must still give a string downstream.
        content = (choice.get("message") or {}).get("content") or ""
        finish_reason = choice.get("finish_reason")

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        cost, estimated = self._parse_cost(usage, prompt_tokens, completion_tokens)
        self._inspect_usage(usage, request)

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_used=prompt_tokens + completion_tokens,
            context_limit=self._response_context_limit(request),
            was_truncated=False,  # Context overflow is reported as an error instead
            finish_reason=finish_reason,
            output_truncated=is_output_limit_finish(finish_reason),
            cost=cost,
            cost_estimated=estimated,
        )

    def _inspect_usage(self, usage: Dict[str, Any], request: LLMRequest) -> None:
        """Hook to look at the raw usage block (reasoning tokens, ...)."""

    # ------------------------------------------- optional-parameter fallback
    async def _retry_without_optional_params(self, error: ProviderError,
                                             request: LLMRequest) -> bool:
        optional = request.state.get("optional") or {}
        if error.status_code != 400 or not optional or request.state.get("optional_dropped"):
            return False
        text = f"{error.message} {error.body}"
        rejected = [name for name in optional if name in text]
        if not rejected and not self.DROP_ALL_OPTIONAL_ON_UNNAMED_400:
            return False
        request.state["optional_dropped"] = True
        if rejected:
            self._log("llm_params_rejected",
                      f"⚠️ [{self.display_name}] '{request.model}' rejected {sorted(rejected)}; "
                      f"retrying without them", level=logging.INFO)
            self._rejected_optional.setdefault(request.model, set()).update(rejected)
            self._on_optional_params_rejected(rejected, request)
            request.state["optional"] = {k: v for k, v in optional.items() if k not in rejected}
        else:
            self._log("llm_params_rejected",
                      f"⚠️ [{self.display_name}] HTTP 400 for '{request.model}'; retrying once "
                      f"without {sorted(optional)} to rule them out", level=logging.INFO)
            request.state["optional"] = {}
        return True
