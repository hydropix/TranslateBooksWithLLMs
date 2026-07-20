"""Shared LLM client/context bootstrap for the refine-only refiners.

The EPUB and DOCX refiners built their LLM client and adaptive context manager
with byte-identical ~30-line blocks. That setup lives here once so adding a
provider arg or tweaking the context heuristic is a one-place change.
"""

from typing import Any, Optional, Tuple

from src.config import (
    THINKING_MODELS,
    ADAPTIVE_CONTEXT_INITIAL_THINKING,
)
from src.core.context_optimizer import INITIAL_CONTEXT_SIZE
from src.core.epub.translator import _create_llm_client, _create_context_manager


def build_refine_client(
    *,
    model_name: str,
    llm_provider: str,
    cli_api_endpoint: str,
    auto_adjust_context: bool,
    context_window: int,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    atlascloud_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    log_callback: Optional[Any] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Create the refine LLM client and (optional) context manager.

    Returns ``(llm_client, context_manager)``. ``llm_client`` is ``None`` when
    creation failed (callers should bail out).
    """
    is_thinking_model = any(tm in model_name.lower() for tm in THINKING_MODELS)
    if auto_adjust_context:
        initial_context = (
            ADAPTIVE_CONTEXT_INITIAL_THINKING if is_thinking_model else INITIAL_CONTEXT_SIZE
        )
    else:
        initial_context = context_window

    llm_client = _create_llm_client(
        llm_provider=llm_provider,
        model_name=model_name,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        atlascloud_api_key=atlascloud_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        cli_api_endpoint=cli_api_endpoint,
        initial_context=initial_context,
        log_callback=log_callback,
    )
    if llm_client is None:
        return None, None

    context_manager = _create_context_manager(
        llm_provider=llm_provider,
        auto_adjust_context=auto_adjust_context,
        initial_context=initial_context,
        is_thinking_model=is_thinking_model,
        log_callback=log_callback,
    )
    return llm_client, context_manager
