"""Build a JobConfig / JobContext by hand for engine tests.

Engines take ``(job: JobConfig, ctx: JobContext)``. These helpers keep the
tests short: every argument has a test-friendly default and the provider
fields (``provider``, ``model``, ``endpoint``, ``api_key``) are folded into
the job's ``ProviderConfig``.
"""

import os
from typing import Any, Optional

from src.core.job import JobConfig, JobContext
from src.core.llm.provider_config import ProviderConfig


def make_job(
    input_path: Optional[str] = None,
    output_path: str = "",
    *,
    provider: str = "ollama",
    model: str = "test-model",
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    file_type: Optional[str] = None,
    source_language: str = "English",
    target_language: str = "French",
    **fields: Any,
) -> JobConfig:
    """A JobConfig with a ProviderConfig built from the loose provider fields."""
    if file_type is None:
        ext = os.path.splitext(str(input_path or ""))[1].lstrip(".").lower()
        file_type = ext or "txt"
    if 'operation' not in fields and fields.get('term_fix'):
        fields['operation'] = 'term_fix'
    return JobConfig(
        input_path=None if input_path is None else str(input_path),
        output_path=str(output_path),
        file_type=file_type,
        source_language=source_language,
        target_language=target_language,
        provider=ProviderConfig(provider=provider, model=model, endpoint=endpoint,
                                api_key=api_key),
        **fields,
    )


def make_ctx(
    *,
    translation_id: Optional[str] = None,
    checkpoint_manager: Any = None,
    log_callback=None,
    stats_callback=None,
    check_interruption=None,
    resume_from_index: int = 0,
    checkpoint_phase: str = 'translate',
) -> JobContext:
    """A JobContext; every hook is optional."""
    return JobContext(
        translation_id=translation_id,
        checkpoint_manager=checkpoint_manager,
        log_callback=log_callback,
        stats_callback=stats_callback,
        check_interruption=check_interruption,
        resume_from_index=resume_from_index,
        checkpoint_phase=checkpoint_phase,
    )
