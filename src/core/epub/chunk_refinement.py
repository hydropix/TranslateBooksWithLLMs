"""Refinement of placeholder chunks (second LLM pass over a translation).

Used by the refine package (EPUB and DOCX refine-only modes and the
refine-after-translate phase) and, for its request helper, by term-fix.
Refined chunks keep their placeholders: each chunk is sent with local
indices, validated against its local tag map, and brought back to global
indices; any failure keeps the draft translation.
"""

import re
import time
from dataclasses import replace as _replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import MAX_REFINE_RETRIES
from src.core.progress import ProgressReporter
from src.prompts.prompts import PromptPair

from .chunk_fallback import keep_edge_whitespace, validate_placeholders
from .engine_log import report_error
from .placeholder_renumberer import globalize_placeholders, localize_placeholders
from ..context_optimizer import AdaptiveContextManager


async def request_refinement(
    llm_client: Any,
    context_manager: Optional[AdaptiveContextManager],
    prompt_pair: Any,
    model_name: str,
    log_callback: Optional[Callable],
    *,
    extract: bool = True,
) -> Any:
    """Send one refinement request through the shared request core.

    Returns the provider response when it passed every guard of
    ``request_chunk`` (the caller extracts and validates the placeholders),
    or None when the draft must be kept. The context grows on overflow,
    truncation or a repetition loop until the context manager refuses; the
    call has its own growth budget, given back when it fails.
    ``extract=False`` hands back the raw answer of callers with their own
    response format (term-fix).
    """
    from ..request_core import ContextBudget, REFINEMENT_LABELS, request_chunk

    # The caller logs the response itself (with its execution time).
    labels = _replace(REFINEMENT_LABELS, response=None)
    with ContextBudget(context_manager) as budget:
        outcome = await request_chunk(
            llm_client, prompt_pair,
            source_text="",
            allow_raw_fallback=False,
            context_manager=context_manager,
            log_callback=log_callback,
            labels=labels,
            extract=extract,
        )
        if not outcome.ok:
            return None
        budget.succeeded()
        response = outcome.response
        if context_manager and response is not None and response.prompt_tokens > 0:
            context_manager.record_success(
                response.prompt_tokens,
                response.completion_tokens,
                response.context_limit
            )
        return response


def build_refine_placeholder_reminder(local_tag_map: Dict) -> str:
    """Build the reminder appended to a refinement prompt after placeholder corruption."""
    def _ph_index(key: Any) -> Tuple[int, str]:
        m = re.search(r"(\d+)", str(key))
        return (int(m.group(1)) if m else 0, str(key))

    expected = ", ".join(str(k) for k in sorted(local_tag_map, key=_ph_index))
    return (
        "\n\nIMPORTANT: your previous answer altered, dropped, duplicated or reordered "
        "the placeholders. Keep EVERY placeholder exactly as written, once each, in the "
        "same order and position. "
        + (f"Expected placeholders: {expected}." if expected else "")
    )


class _ChunkRefiner:
    """Refines chunks one by one with the settings shared by all of them."""

    def __init__(self, target_language: str, model_name: str, llm_client: Any,
                 context_manager: Optional[AdaptiveContextManager],
                 placeholder_format: Tuple[str, str], log_callback: Optional[Callable],
                 prompt_options: Optional[Dict], max_refine_retries: int, total_chunks: int):
        self.target_language = target_language
        self.model_name = model_name
        self.llm_client = llm_client
        self.context_manager = context_manager
        self.placeholder_format = placeholder_format
        self.log_callback = log_callback
        self.prompt_options = prompt_options
        self.max_refine_retries = max_refine_retries
        self.total_chunks = total_chunks

    def _request(self, prompt_pair: Any):
        return request_refinement(self.llm_client, self.context_manager, prompt_pair,
                                  self.model_name, self.log_callback)

    async def _retry_corrupted_placeholders(self, llm_response: Any, prompt_pair: Any,
                                            local_tag_map: Dict, idx: int) -> Any:
        """Re-prompt with an explicit reminder (bounded) while the refined
        text corrupts the placeholders; the last response."""
        refine_attempt = 0
        while (refine_attempt < self.max_refine_retries and local_tag_map
               and llm_response and llm_response.content):
            candidate = self.llm_client.extract_translation(llm_response.content)
            if not candidate or validate_placeholders(candidate, local_tag_map):
                break
            refine_attempt += 1
            report_error(self.log_callback, "epub_refinement_placeholder_retry",
                         f"Chunk {idx + 1}/{self.total_chunks}: refinement corrupted placeholders, "
                         f"retry {refine_attempt}/{self.max_refine_retries}")
            retry_prompt = PromptPair(
                system=prompt_pair.system,
                user=prompt_pair.user + build_refine_placeholder_reminder(local_tag_map),
            )
            llm_response = await self._request(retry_prompt)
        return llm_response

    def _log_request(self, prompt_pair: Any) -> None:
        if self.log_callback:
            self.log_callback("llm_request", "Sending refinement request to LLM", data={
                'type': 'llm_request',
                'system_prompt': prompt_pair.system,
                'user_prompt': prompt_pair.user,
                'model': self.model_name
            })

    def _log_response(self, llm_response: Any, execution_time: float) -> None:
        if self.log_callback and llm_response:
            self.log_callback("llm_response", "LLM Response received", data={
                'type': 'llm_response',
                'response': llm_response.content,
                'execution_time': execution_time,
                'model': self.model_name,
                'tokens': {
                    'prompt': llm_response.prompt_tokens,
                    'completion': llm_response.completion_tokens,
                    'total': llm_response.context_used,
                    'limit': llm_response.context_limit
                }
            })

    def _accept(self, llm_response: Any, translated_text: str, local_tag_map: Dict,
                global_indices: List[int], idx: int) -> str:
        """The refined chunk (global indices), or the draft when unusable."""
        log_callback = self.log_callback
        if not (llm_response and llm_response.content):
            report_error(log_callback, "epub_refinement_failed",
                         f"Chunk {idx + 1}/{self.total_chunks}: refinement failed, using original")
            return translated_text

        refined_text = self.llm_client.extract_translation(llm_response.content)
        if not refined_text:
            if log_callback:
                log_callback("epub_refinement_fallback",
                             f"Chunk {idx + 1}/{self.total_chunks}: using original translation")
            return translated_text

        # refined_text should have LOCAL indices (0, 1, 2...) matching local_tag_map.
        if local_tag_map and not validate_placeholders(refined_text, local_tag_map):
            report_error(log_callback, "epub_refinement_placeholder_corruption",
                         f"Chunk {idx + 1}/{self.total_chunks}: refinement corrupted placeholders, using original translation")
            return translated_text

        refined_global = globalize_placeholders(refined_text, global_indices, self.placeholder_format)
        if log_callback:
            log_callback("epub_chunk_refined", f"Chunk {idx + 1}/{self.total_chunks} refined successfully")
        return keep_edge_whitespace(translated_text, refined_global)

    async def refine(self, idx: int, translated_text: str, chunk_dict: Dict) -> str:
        """Refine one chunk; the draft comes back on any failure."""
        from src.prompts.prompts import generate_post_processing_prompt

        refinement_instructions = (self.prompt_options.get('refinement_instructions', '')
                                   if self.prompt_options else '')
        local_tag_map = chunk_dict.get('local_tag_map', {})
        global_indices = chunk_dict.get('global_indices', [])

        # translated_text has GLOBAL indices; the prompt expects placeholders
        # starting at 0. Convert global -> local before sending to the LLM,
        # and local -> global after receiving the refined result.
        text_for_refinement = localize_placeholders(translated_text, global_indices,
                                                    self.placeholder_format)
        prompt_pair = generate_post_processing_prompt(
            translated_text=text_for_refinement,
            target_language=self.target_language,
            additional_instructions=refinement_instructions,
            has_placeholders=True,
            placeholder_format=self.placeholder_format,
            prompt_options=self.prompt_options
        )
        try:
            self._log_request(prompt_pair)
            start_time = time.time()
            llm_response = await self._request(prompt_pair)
            execution_time = time.time() - start_time
            llm_response = await self._retry_corrupted_placeholders(
                llm_response, prompt_pair, local_tag_map, idx)
            self._log_response(llm_response, execution_time)
            return self._accept(llm_response, translated_text, local_tag_map, global_indices, idx)
        except Exception as e:
            # Re-raise RateLimitError to trigger auto-pause
            from src.core.llm.exceptions import RateLimitError as _RLE
            if isinstance(e, _RLE):
                raise
            report_error(self.log_callback, "epub_refinement_error",
                         f"Chunk {idx + 1}/{self.total_chunks}: error during refinement: {e}")
            return translated_text


async def refine_chunks(
    translated_chunks: List[str],
    chunks: List[Dict],
    target_language: str,
    model_name: str,
    llm_client: Any,
    context_manager: Optional[AdaptiveContextManager],
    placeholder_format: Tuple[str, str],
    log_callback: Optional[Callable],
    prompt_options: Optional[Dict],
    progress: Optional[ProgressReporter] = None,
    max_refine_retries: Optional[int] = None
) -> List[str]:
    """Refine translated placeholder chunks with a second LLM pass.

    Args:
        translated_chunks: Translated chunk texts (GLOBAL placeholders); see
            ``placeholder_renumberer.globalize_chunks`` for chunks fresh from
            the chunker
        chunks: Chunk dicts (``local_tag_map``, ``global_indices``)
        target_language: Target language
        model_name: LLM model name
        llm_client: LLM client instance
        context_manager: Optional context manager
        placeholder_format: Placeholder format tuple (prefix, suffix)
        log_callback: The job's log callback
        prompt_options: Prompt options dict
        progress: The caller's ProgressReporter, advanced and emitted once per
            refined chunk (the caller has begun it with its own total)
        max_refine_retries: Re-prompts allowed per chunk when the refined text
            corrupts placeholders (default: MAX_REFINE_RETRIES from config)

    Returns:
        Refined chunk texts (GLOBAL placeholders); a chunk that could not be
        refined keeps its draft.

    Raises:
        RateLimitError: with ``partial_result`` set to a complete chunk list
            (the chunks refined before the limit, then the remaining drafts),
            so the caller can save partial output before pausing.
    """
    if max_refine_retries is None:
        max_refine_retries = MAX_REFINE_RETRIES
    total_chunks = len(translated_chunks)

    if log_callback:
        log_callback("epub_refinement_info",
                     f"Refining {total_chunks} EPUB chunks (original chunks: {len(chunks)})...")
    if len(translated_chunks) != len(chunks):
        report_error(log_callback, "epub_refinement_warning",
                     f"Warning: Length mismatch - translated_chunks: {len(translated_chunks)}, chunks: {len(chunks)}")

    refiner = _ChunkRefiner(target_language, model_name, llm_client, context_manager,
                            placeholder_format, log_callback, prompt_options,
                            max_refine_retries, total_chunks)
    refined_chunks: List[str] = []
    from src.core.llm.exceptions import RateLimitError
    try:
        for idx, (translated_text, chunk_dict) in enumerate(zip(translated_chunks, chunks)):
            refined_chunks.append(await refiner.refine(idx, translated_text, chunk_dict))
            if progress is not None:
                progress.record()
                progress.emit()
    except RateLimitError as e:
        e.partial_result = refined_chunks + list(translated_chunks[len(refined_chunks):])
        raise

    if log_callback:
        successful_refinements = sum(1 for orig, ref in zip(translated_chunks, refined_chunks) if orig != ref)
        log_callback("epub_refinement_complete",
                     f"✨ Refinement complete: {successful_refinements}/{total_chunks} chunks improved")
    return refined_chunks
