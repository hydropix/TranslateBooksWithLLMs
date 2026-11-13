"""Shared LLM client/context bootstrap for the refine-only refiners.

The EPUB and DOCX refiners (and the smart term-fix pass) build their LLM
client and adaptive context manager the same way; that setup lives here once
so tweaking the context heuristic is a one-place change.
"""

from typing import Any, Optional, Tuple

from src.core.context_optimizer import REFINEMENT_MIN_CONTEXT
from src.core.epub.translator import (
    _create_context_manager,
    _create_llm_client,
    initial_context_size,
)
from src.core.job import JobConfig


def build_refine_client(
    job: JobConfig,
    *,
    log_callback: Optional[Any] = None,
) -> Tuple[Any, Optional[Any]]:
    """Create the refine LLM client and (optional) context manager.

    The client is built from the job's refine provider
    (``job.provider_for(refine=True)``), sized from ``job.context_window``
    (2048 when unset) and ``job.auto_adjust_context``.

    Returns:
        ``(llm_client, context_manager)``; the context manager is None unless
        the provider is Ollama with adaptive context.

    Raises:
        LLMClientConfigError: the client cannot be created (unknown provider,
            missing key). It has already been reported to ``log_callback``.
    """
    provider = job.refine_provider_config()
    llm_provider = provider.provider or 'ollama'
    initial_context, is_thinking_model = initial_context_size(
        provider.model, job.auto_adjust_context, job.context_window or 2048)
    # Same floor as the TXT/SRT refine path (refine_chunks in translator.py).
    initial_context = max(initial_context, REFINEMENT_MIN_CONTEXT)

    llm_client = _create_llm_client(job, initial_context, log_callback, refine=True)

    context_manager = _create_context_manager(
        llm_provider=llm_provider,
        auto_adjust_context=job.auto_adjust_context,
        initial_context=initial_context,
        is_thinking_model=is_thinking_model,
        log_callback=log_callback,
    )
    return llm_client, context_manager
