"""Unified refine-only entry point.

Mirrors translate_file() but skips the translation phase: the input file
is assumed to already be in the target language, and only a refinement
pass is applied.
"""

import importlib
import os

from .exceptions import UnsupportedFormatError
from src.core.job import JobConfig, JobContext, JobResult
from src.utils.file_detector import detect_file_type, detect_file_type_by_content

#: Detected file type -> (refiner module, refiner function). Every refiner is
#: called as ``await refiner(job, ctx)`` and returns True when its output was
#: written in full.
_REFINERS = {
    'txt': ('src.core.refine.txt_refiner', 'refine_txt_file'),
    'epub': ('src.core.refine.epub_refiner', 'refine_epub_file'),
    'docx': ('src.core.refine.docx_refiner', 'refine_docx_file'),
    'srt': ('src.core.refine.srt_refiner', 'refine_srt_file'),
}


def _detect_type(input_filepath: str, ctx: JobContext) -> str:
    _, ext = os.path.splitext(input_filepath.lower())
    try:
        detected_type = detect_file_type(input_filepath)
    except ValueError:
        detected_type = detect_file_type_by_content(input_filepath)
        if detected_type is None:
            raise UnsupportedFormatError(
                f"Cannot determine file type for: {ext}. "
                f"The file does not appear to be a supported format."
            )

    if detected_type != ext.lstrip('.'):
        ctx.log("file_type_detected",
                f"📄 File with extension '{ext}' detected as '{detected_type.upper()}' format")
    return detected_type


async def refine_file(job: JobConfig, ctx: JobContext) -> JobResult:
    """Run a refinement-only pass on an already-translated file.

    Reads ``job.input_path`` and writes ``job.output_path``. The client is
    built from the job's refine provider (``job.provider_for(refine=True)``).
    ``job.target_language`` names the language the file is already in:
    refinement is monolingual and does not translate.

    Returns:
        The run's :class:`JobResult` (INTERRUPTED when the user stopped it;
        an interrupted EPUB refine reports the ``[partial ...]`` file it
        wrote as ``output_path``).

    Raises:
        UnsupportedFormatError: the file format cannot be refined.
        RateLimitError: propagated unchanged so the caller can pause.
    """
    # Resolve max_tokens_per_chunk lazily so a reload_config() between calls is
    # honoured for subsequent runs (the .env value can change at runtime via
    # the /api/settings endpoint).
    if job.max_tokens_per_chunk is None:
        from src.config import MAX_TOKENS_PER_CHUNK as _DEFAULT_MAX_TOKENS
        job = job.with_changes(max_tokens_per_chunk=_DEFAULT_MAX_TOKENS)

    detected_type = _detect_type(job.input_path, ctx)
    target = _REFINERS.get(detected_type)
    if target is None:
        supported = ', '.join(_REFINERS)
        raise UnsupportedFormatError(
            f"Unsupported file format for refine-only: {detected_type}. "
            f"Supported formats: {supported}"
        )

    module_name, function_name = target
    refiner = getattr(importlib.import_module(module_name), function_name)

    ctx, seen_stats = ctx.tap_stats()
    ok = await refiner(job, ctx)

    interrupted = ctx.interrupted()
    output_path = job.output_path
    if interrupted and detected_type == 'epub':
        from src.utils.file_utils import get_partial_output_path
        output_path = get_partial_output_path(job.output_path)
    return JobResult.from_run(seen_stats, output_path, engine_ok=bool(ok),
                              interrupted=interrupted)
