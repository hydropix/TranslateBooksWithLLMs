"""LLM client and context sizing for a job.

Shared by the EPUB book loop, the generic adapter translator, the DOCX and
PDF engines and the refine-only refiners, so the context heuristic lives in
one place.
"""

from typing import Any, Callable, Optional, Tuple

from src.config import ADAPTIVE_CONTEXT_INITIAL_THINKING, OLLAMA_NUM_CTX, THINKING_MODELS

from ..context_optimizer import (
    CONTEXT_STEP,
    INITIAL_CONTEXT_SIZE,
    MAX_CONTEXT_SIZE,
    AdaptiveContextManager,
)
from ..job import JobConfig
from ..llm.exceptions import LLMClientConfigError


def initial_context_size(model_name: str, auto_adjust_context: bool,
                         context_window: Optional[int]) -> Tuple[int, bool]:
    """(initial context window, model is a known thinking model) for a job.

    With adaptive context the window starts small (larger for thinking
    models) and grows on demand; otherwise it is the user's setting.
    """
    is_thinking_model = any(tm in (model_name or "").lower() for tm in THINKING_MODELS)
    if auto_adjust_context:
        initial = ADAPTIVE_CONTEXT_INITIAL_THINKING if is_thinking_model else INITIAL_CONTEXT_SIZE
    else:
        initial = context_window or OLLAMA_NUM_CTX
    return initial, is_thinking_model


def job_max_tokens_per_chunk(job: JobConfig) -> int:
    """The job's chunk size; the current ``MAX_TOKENS_PER_CHUNK`` when unset.

    Read at call time, so a ``reload_config()`` between jobs is honoured.
    """
    if job.max_tokens_per_chunk is not None:
        return job.max_tokens_per_chunk
    import src.config as config
    return config.MAX_TOKENS_PER_CHUNK


def create_llm_client(
    job: JobConfig,
    initial_context: int,
    log_callback: Optional[Callable] = None,
    *,
    refine: bool = False,
) -> Any:
    """Create the LLM provider for this job (its refine provider with ``refine``).

    Raises ``LLMClientConfigError`` for an unknown provider or a missing
    required key, after reporting it to the job's log. There is no default
    client to fall back to.
    """
    from ..llm.factory import create_llm_provider

    provider = job.provider_for(context_window=initial_context,
                                log_callback=log_callback, refine=refine)
    try:
        return create_llm_provider(provider)
    except LLMClientConfigError as e:
        if log_callback:
            log_callback("llm_client_error",
                         f"ERROR: Could not create LLM client for provider "
                         f"'{provider.provider or 'ollama'}': {e}")
        raise


def create_context_manager(
    llm_provider: str,
    auto_adjust_context: bool,
    initial_context: int,
    is_thinking_model: bool,
    log_callback: Optional[Callable] = None
) -> Optional[AdaptiveContextManager]:
    """Adaptive context manager for Ollama with adaptive context, else None."""
    context_manager = None
    if llm_provider == "ollama" and auto_adjust_context:
        context_manager = AdaptiveContextManager(
            initial_context=initial_context,
            context_step=CONTEXT_STEP,
            max_context=MAX_CONTEXT_SIZE,
            log_callback=log_callback
        )
        model_type = "thinking" if is_thinking_model else "standard"
        if log_callback:
            log_callback("context_adaptive",
                f"🎯 Adaptive context enabled for EPUB ({model_type} model): starting at {initial_context} tokens, "
                f"max={MAX_CONTEXT_SIZE}, step={CONTEXT_STEP}")

    return context_manager
