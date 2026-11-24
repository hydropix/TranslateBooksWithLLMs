"""
Translation module for LLM communication
"""
import re

from tqdm.auto import tqdm

from src.config import (
    DEFAULT_MODEL, SENTENCE_TERMINATORS,
    THINKING_MODELS, ADAPTIVE_CONTEXT_INITIAL_THINKING,
    PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX,
)
from src.core.common.source_residue import detect_source_residue, residue_log_message, residue_score
from src.prompts.prompts import generate_translation_prompt, generate_refinement_prompt
from .llm import RateLimitError, LLMResponse
from .request_core import (
    ChunkStatus,
    ContextBudget,
    REFINEMENT_LABELS,
    TRANSLATION_LABELS,
    request_chunk,
    # Re-exported: the raw-fallback guards used to live here.
    RAW_FALLBACK_MAX_RATIO,  # noqa: F401
    RAW_FALLBACK_MIN_RATIO,  # noqa: F401
    RAW_FALLBACK_MIN_SOURCE_LENGTH,  # noqa: F401
    _weighted_length,
    raw_fallback_rejection_reason,  # noqa: F401
    strip_raw_preamble,  # noqa: F401
)
from .post_processor import clean_translated_text
from .context_optimizer import (
    AdaptiveContextManager,
    INITIAL_CONTEXT_SIZE,
    CONTEXT_STEP,
    REFINEMENT_MIN_CONTEXT,
)
from .progress import Phase, ProgressReporter
from typing import Callable, List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .job import JobConfig, JobContext


# Configuration for context overflow recovery
MAX_CHUNK_REDUCTION_ATTEMPTS = 3
CHUNK_REDUCTION_FACTOR = 0.6  # Reduce to 60% of original size each attempt
MIN_CHUNK_CHARACTERS = 200  # Minimum chunk size to attempt translation


def _build_chunk_glossary_block(
    chunk_content: str,
    prompt_options: Optional[dict],
    log_callback=None,
    runtime_state: Optional[dict] = None,
    *,
    target_language: str = "",
    match_target_forms: bool = False,
) -> str:
    """
    Filter the active glossary against the current chunk and render a prompt block.

    Reads `glossary_terms` (dict source -> target) and optional `glossary_config`
    (GlossaryConfig) from prompt_options. Returns "" when no glossary is active
    or when the glossary has neither a term matching this chunk nor any gendered
    entity.

    Two sections may be emitted:
      - the cast block (gendered entities), which is NOT chunk-filtered, since
        a passage referring to a character by pronoun alone contains no match
        yet is exactly where the model would otherwise guess the gender;
      - the glossary block, restricted to terms present in this chunk.

    When the per-chunk cap is hit and `warn_on_cap` is enabled, logs a single
    warning per job. The dedupe flag lives in `runtime_state` (a transient dict
    owned by the caller) so it never leaks into the persisted prompt_options
    snapshot. If runtime_state is None, a fresh local dict is used (warning
    won't be deduped across calls — fine for ad-hoc uses).

    Keyword-only args:
        target_language: the job's target language, forwarded to
            `build_glossary_block`. It selects the target-side inflection
            instruction, emitted only for the languages listed in
            `INFLECTED_TARGET_LANGUAGES` (see src/core/glossary/inflection.py).
            Defaults to "" so existing callers keep the previous output.
        match_target_forms: the chunk is a target-language draft (refinement
            pass): match the glossary's target renderings instead of its
            source terms, and word the block as "keep these exact forms".
            Matching source keys against a ZH->FR draft would find nothing.
    """
    if not prompt_options:
        return ""
    terms = prompt_options.get("glossary_terms")
    if not terms:
        return ""
    try:
        from src.core.glossary import (
            build_cast_block,
            build_glossary_block,
            build_refinement_glossary_block,
            filter_glossary,
            GlossaryConfig,
            target_language_is_inflected,
        )
    except ImportError:
        return ""
    config = prompt_options.get("glossary_config") or GlossaryConfig()
    if match_target_forms:
        sources_by_target: Dict[str, List[str]] = {}
        for source, target in terms.items():
            if target and target.strip():
                sources_by_target.setdefault(target.strip(), []).append(source)
        # An inflected target rarely shows its dictionary form verbatim, so
        # an inflected ending after the rendering still counts as a match.
        matched_targets, capped = filter_glossary(
            chunk_content, {t: t for t in sources_by_target}, config,
            prefix=target_language_is_inflected(target_language),
        )
        filtered = {
            source: terms[source]
            for target in matched_targets
            for source in sources_by_target[target]
        }
    else:
        filtered, capped = filter_glossary(chunk_content, terms, config)

    if runtime_state is None:
        runtime_state = {}

    if capped and config.warn_on_cap and not runtime_state.get("glossary_cap_warned"):
        runtime_state["glossary_cap_warned"] = True
        if log_callback:
            log_callback(
                "glossary_capped",
                f"⚠️ Glossary cap reached: more than {config.max_entries} terms matched in a single chunk. "
                f"Excess entries are dropped — increase `max_entries` if you need full coverage."
            )

    metadata = prompt_options.get("glossary_term_metadata") or None

    cast_block, cast_capped = build_cast_block(
        terms,
        term_metadata=metadata,
        max_entries=getattr(config, "max_cast_entries", None) or 0,
    )

    if cast_capped and config.warn_on_cap and not runtime_state.get("cast_cap_warned"):
        runtime_state["cast_cap_warned"] = True
        if log_callback:
            log_callback(
                "glossary_cast_capped",
                f"⚠️ Cast block cap reached: more than {config.max_cast_entries} glossary entries "
                f"carry a gender. Entries beyond the cap are dropped from the per-chunk gender "
                f"reference — raise `max_cast_entries` if the whole cast must be listed."
            )

    render = build_refinement_glossary_block if match_target_forms else build_glossary_block
    glossary_block = (
        render(
            filtered, target_language=target_language, term_metadata=metadata
        ) if filtered else ""
    )

    if not cast_block and not glossary_block:
        return ""
    if not cast_block:
        return glossary_block
    if not glossary_block:
        return cast_block
    return f"{cast_block}\n{glossary_block}"


# Previous-translation context handed to the next chunk's prompt: at most
# this many words, and at most this weighted length (CJK counts 3, Hangul 2),
# so unspaced scripts get a bounded tail too instead of the whole chunk.
PREVIOUS_CONTEXT_MAX_WORDS = 25
PREVIOUS_CONTEXT_MAX_WEIGHTED_CHARS = 300


def previous_context_tail(text: Optional[str]) -> str:
    """The end of a translation, used as the next chunk's previous context.

    The single policy for every pipeline: the last PREVIOUS_CONTEXT_MAX_WORDS
    whitespace-separated words, further capped at
    PREVIOUS_CONTEXT_MAX_WEIGHTED_CHARS. A Chinese or Japanese translation has
    no spaces, so the word rule alone would pass the whole previous chunk.
    """
    text = text or ""
    words = text.split()
    tail = " ".join(words[-PREVIOUS_CONTEXT_MAX_WORDS:]) if len(words) > PREVIOUS_CONTEXT_MAX_WORDS else text
    if _weighted_length(tail) <= PREVIOUS_CONTEXT_MAX_WEIGHTED_CHARS:
        return tail
    budget = PREVIOUS_CONTEXT_MAX_WEIGHTED_CHARS
    start = len(tail)
    while start > 0:
        weight = _weighted_length(tail[start - 1])
        if weight > budget:
            break
        budget -= weight
        start -= 1
    cut = tail[start:]
    # Do not start in the middle of a word when the text has spaces.
    if start > 0 and not tail[start - 1].isspace() and " " in cut:
        cut = cut.split(" ", 1)[1]
    return cut.lstrip()


def split_chunk_for_retry(main_content: str, target_ratio: float = 0.5) -> Tuple[str, str]:
    """
    Split a chunk into two parts for retry after context overflow.

    Tries to split at a sentence boundary near the target ratio.

    Args:
        main_content: The text content to split
        target_ratio: Target position for split (0.5 = middle)

    Returns:
        Tuple of (first_half, second_half), both stripped. Use
        split_chunk_preserving_separator to also get the whitespace between them.
    """
    first, _, second = split_chunk_preserving_separator(main_content, target_ratio)
    return first, second


def split_chunk_preserving_separator(
    main_content: str, target_ratio: float = 0.5
) -> Tuple[str, str, str]:
    """
    Split a chunk in two for a context-overflow retry, keeping the separator.

    Returns (first, separator, second) with first + separator + second equal
    to main_content minus its outer whitespace, so the two translations can be
    joined back with the original separator (a space inside a paragraph, a
    blank line between paragraphs) instead of reflowing the text. Whenever the
    content holds more than one character of text, both parts are non-empty.
    """
    content = main_content.strip()
    if not content:
        return content, "", ""
    pos = _split_position(content, target_ratio)
    # A split at either edge would leave one side empty and lose nothing only
    # by luck; force a real split so the overflow loop always makes progress.
    if pos <= 0 or pos >= len(content):
        pos = max(1, min(len(content) - 1, len(content) // 2))
    first = content[:pos].rstrip()
    second = content[pos:].lstrip()
    if not first or not second:
        # Whitespace on one side only: split in the middle of the text instead.
        mid = max(1, len(content) // 2)
        first, second = content[:mid].rstrip(), content[mid:].lstrip()
    separator = content[len(first):len(content) - len(second)]
    return first, separator, second


def _split_position(main_content: str, target_ratio: float) -> int:
    """Index in main_content where split_chunk_preserving_separator cuts."""
    lines = main_content.split('\n')
    if len(lines) <= 1:
        # For single line, split at sentence boundary or middle
        target_pos = int(len(main_content) * target_ratio)

        # Look for sentence terminators near target position
        best_split = target_pos
        for terminator in SENTENCE_TERMINATORS:
            # Search in a window around target position
            search_start = max(0, target_pos - 100)
            search_end = min(len(main_content), target_pos + 100)
            search_area = main_content[search_start:search_end]

            term_pos = search_area.rfind(terminator)
            if term_pos != -1:
                actual_pos = search_start + term_pos + len(terminator)
                if abs(actual_pos - target_pos) < abs(best_split - target_pos):
                    best_split = actual_pos

        return best_split

    # For multi-line content, split at line boundaries
    target_line = int(len(lines) * target_ratio)

    # Look for a sentence-ending line near target
    best_line = target_line
    for i in range(max(0, target_line - 5), min(len(lines), target_line + 5)):
        line_stripped = lines[i].strip()
        if line_stripped and line_stripped.endswith(SENTENCE_TERMINATORS):
            best_line = i + 1
            break

    # Never cut before the first or after the last non-empty line.
    non_empty = [k for k, line in enumerate(lines) if line.strip()]
    if len(non_empty) >= 2:
        best_line = max(non_empty[0] + 1, min(best_line, non_empty[-1]))
    return len('\n'.join(lines[:best_line]))





def _require_client(llm_client):
    """Return llm_client, or fail clearly when none was given.

    There is no module-global fallback client: a request without the job's
    provider must fail clearly rather than reach an arbitrary endpoint.
    """
    if llm_client is None:
        raise ValueError(
            "No LLM client was provided for this request. Create one with "
            "create_llm_provider() for the job's provider."
        )
    return llm_client


def _join_split_translations(parts: List[Tuple[str, str]]) -> Optional[str]:
    """Join (separator_before, translation) pieces back into one text."""
    if not parts:
        return None
    return parts[0][1] + "".join(sep + text for sep, text in parts[1:])


async def _make_llm_request_with_adaptive_context(
    main_content: str,
    previous_translation_context: str,
    source_language: str,
    target_language: str,
    model: str,
    llm_client,
    log_callback,
    has_placeholders: bool,
    prompt_options: dict = None,
    context_manager: AdaptiveContextManager = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    runtime_state: Optional[dict] = None,
    raw_response_callback: Optional[Callable[[LLMResponse], None]] = None,
    answer_check: Optional[Callable[[str, str], Optional[str]]] = None,
    on_rejected: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], str, Optional[LLMResponse]]:
    """
    Translate one chunk through the shared request core, splitting it on a
    context overflow the context manager cannot absorb.

    Every response guard (empty, truncated, unclosed tag, extraction, raw
    fallback, adaptive context growth) lives in ``request_chunk``; this
    function only owns the overflow split: the chunk is cut at a sentence or
    line boundary, each piece is translated in order and the translations are
    joined back with the source separators.

    Args:
        main_content: Text to translate
        previous_translation_context: Previous translation for consistency
        source_language: Source language
        target_language: Target language
        model: LLM model name (logs only; the provider carries its model)
        llm_client: LLM provider instance
        log_callback: Logging callback function
        has_placeholders: If True, includes placeholder preservation
            instructions and forbids the raw-response fallback
        prompt_options: Optional dict with prompt customization options
        context_manager: AdaptiveContextManager for context sizing
        raw_response_callback: Optional callable receiving every LLMResponse
            as it arrives, before extraction or any retry decision. Errors it
            raises are swallowed: recording must never break a translation.
        answer_check: Optional ``(source, answer) -> reason | None`` applied
            by the request core to every tagged answer (see
            ``request_core.tagged_answer_problem``).
        on_rejected: Optional callable receiving the reason when the chunk
            failed because its answer was rejected or cut off (as opposed to
            the provider failing), so the caller can retry with a reinforced
            prompt.

    Returns:
        Tuple of (translated_text or None, main_content, last LLMResponse);
        the response's ``was_fallback`` is True when any piece came from an
        untagged raw answer.
    """
    client = _require_client(llm_client)
    current_content = main_content
    # Pieces still to translate after a context-overflow split, each with the
    # source whitespace that separated it from the piece before it.
    remaining: List[Tuple[str, str]] = []
    current_sep = ""
    # (separator_before, translation) per translated piece.
    all_translations: List[Tuple[str, str]] = []
    reduction_attempt = 0
    last_response: Optional[LLMResponse] = None
    used_raw_fallback = False

    while current_content.strip():
        glossary_block = _build_chunk_glossary_block(
            current_content, prompt_options, log_callback=log_callback,
            runtime_state=runtime_state, target_language=target_language,
        )
        prompt_pair = generate_translation_prompt(
            current_content,
            previous_translation_context,
            source_language,
            target_language,
            has_placeholders=has_placeholders,
            prompt_options=prompt_options,
            placeholder_format=placeholder_format,
            glossary_block=glossary_block,
        )
        if log_callback and reduction_attempt == 0:
            log_callback("llm_request", "Sending request to LLM", data={
                'type': 'llm_request',
                'system_prompt': prompt_pair.system,
                'user_prompt': prompt_pair.user,
                'model': model
            })

        outcome = await request_chunk(
            client, prompt_pair,
            source_text=current_content,
            allow_raw_fallback=not has_placeholders,
            context_manager=context_manager,
            log_callback=log_callback,
            labels=TRANSLATION_LABELS,
            raw_response_callback=raw_response_callback,
            answer_check=answer_check,
        )
        if outcome.response is not None:
            last_response = outcome.response

        if outcome.status is ChunkStatus.OVERFLOW:
            reduction_attempt += 1
            if reduction_attempt > MAX_CHUNK_REDUCTION_ATTEMPTS:
                if log_callback:
                    log_callback("context_overflow_fatal",
                        f"⚠️ Context overflow: Max reduction attempts ({MAX_CHUNK_REDUCTION_ATTEMPTS}) "
                        f"exceeded. Original error: {outcome.detail}")
                else:
                    tqdm.write(f"\n⚠️ Context overflow after {MAX_CHUNK_REDUCTION_ATTEMPTS} reduction attempts")
                return None, main_content, last_response

            reduction_factor = CHUNK_REDUCTION_FACTOR ** reduction_attempt
            if log_callback:
                log_callback("context_overflow_retry",
                    f"⚠️ Context overflow detected! Reducing chunk to {reduction_factor*100:.0f}% "
                    f"(attempt {reduction_attempt}/{MAX_CHUNK_REDUCTION_ATTEMPTS})")
            else:
                tqdm.write(f"\n⚠️ Context overflow - reducing chunk (attempt {reduction_attempt})")

            first_part, separator, second_part = split_chunk_preserving_separator(
                current_content, reduction_factor)
            if not first_part or (len(first_part) < MIN_CHUNK_CHARACTERS and not all_translations):
                if log_callback:
                    log_callback("context_overflow_fatal",
                        f"⚠️ Cannot reduce chunk further (min size: {MIN_CHUNK_CHARACTERS} chars)")
                return None, main_content, last_response

            current_content = first_part
            # The second part goes first in the queue: it precedes every piece
            # already waiting there. Its separator travels with it, so the
            # translations are joined back exactly as the source was.
            if second_part:
                remaining.insert(0, (separator, second_part))
            continue

        if not outcome.ok:
            if outcome.status is ChunkStatus.REPETITION and not log_callback:
                tqdm.write("\n⚠️ Repetition loop detected - increase OLLAMA_NUM_CTX")
            if on_rejected and outcome.status in (ChunkStatus.REFUSED, ChunkStatus.TRUNCATED):
                on_rejected(outcome.detail or outcome.status.value)
            return None, main_content, last_response

        all_translations.append((current_sep, outcome.text))
        if outcome.status is ChunkStatus.FALLBACK:
            used_raw_fallback = True

        if remaining:
            current_sep, current_content = remaining.pop(0)
            previous_translation_context = previous_context_tail(all_translations[-1][1])
            reduction_attempt = 0
            continue
        break

    # was_fallback must survive a later split request overwriting
    # last_response, so it is applied once, on the response returned.
    if used_raw_fallback and last_response is not None:
        last_response.was_fallback = True
    return _join_split_translations(all_translations), main_content, last_response


# Legacy wrapper for backward compatibility

def _placeholder_tokens(text: str, placeholder_format: Optional[Tuple[str, str]]) -> List[str]:
    """Sorted placeholder tokens of `text`, used to compare two answers."""
    prefix, suffix = placeholder_format or (PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX)
    return sorted(re.findall(re.escape(prefix) + r"\d+" + re.escape(suffix), text or ""))


async def _retry_on_source_residue(
    translated_text: str,
    *,
    main_content: str,
    source_language: str,
    target_language: str,
    log_callback,
    has_placeholders: bool,
    prompt_options: Optional[dict],
    placeholder_format: Optional[Tuple[str, str]],
    on_raw_fallback: Optional[Callable[[], None]],
    **request_kwargs,
) -> Tuple[str, bool]:
    """Re-ask once when an answer still holds source-language text (issue #270).

    Returns ``(answer, retry_won)``. The retry states what went wrong in the
    prompt. The answer with the least residue wins; in placeholder mode a
    retry that lost or invented a placeholder never replaces an answer that
    kept them all, since the caller would then throw away a usable chunk.
    Every finding earns this retry, including the retry-only ones (echo,
    leftover Latin text, single lowercase source words); callers then run `detect_blocking_residue` on what
    comes back and record the chunk as untranslated only for a blocking
    finding.

    The retry reports a raw-response fallback through `on_raw_fallback` only
    when its answer is the one kept; when the first answer is kept, the
    caller reports that one as usual.
    """
    if (prompt_options or {}).get('source_residue_retry'):
        return translated_text, False
    # Glossary terms may legitimately stay in their source form (spells,
    # coined words): never treat them as leftover lowercase words (#353).
    glossary_terms = (prompt_options or {}).get('glossary_terms') or {}
    allowed_terms = [*glossary_terms.keys(), *glossary_terms.values()]
    report = detect_source_residue(main_content, translated_text, source_language,
                                   target_language, allowed_terms=allowed_terms)
    if report is None:
        return translated_text, False

    if log_callback:
        log_callback("source_residue_retry",
                     f"⚠️ {residue_log_message(report, source_language)} - retrying once")

    retry_was_fallback = False

    def _mark_retry_fallback():
        nonlocal retry_was_fallback
        retry_was_fallback = True

    retry_options = {**(prompt_options or {}), 'source_residue_retry': True}
    retried = await generate_translation_request(
        main_content,
        source_language=source_language,
        target_language=target_language,
        log_callback=log_callback,
        has_placeholders=has_placeholders,
        prompt_options=retry_options,
        placeholder_format=placeholder_format,
        on_raw_fallback=_mark_retry_fallback,
        **request_kwargs,
    )
    if not retried:
        return translated_text, False

    def _accept_retry() -> Tuple[str, bool]:
        if retry_was_fallback and on_raw_fallback:
            try:
                on_raw_fallback()
            except Exception as count_error:
                if log_callback:
                    log_callback("raw_fallback_count_failed",
                        f"Could not record the raw-response fallback: {count_error}")
        return retried, True

    if has_placeholders:
        expected = _placeholder_tokens(main_content, placeholder_format)
        first_ok = _placeholder_tokens(translated_text, placeholder_format) == expected
        retry_ok = _placeholder_tokens(retried, placeholder_format) == expected
        if first_ok and not retry_ok:
            return translated_text, False
        if retry_ok and not first_ok:
            return _accept_retry()

    retry_report = detect_source_residue(main_content, retried, source_language,
                                         target_language, allowed_terms=allowed_terms)
    if residue_score(retry_report) < residue_score(report):
        if log_callback and retry_report is None:
            log_callback("source_residue_recovered", "✓ Retry translated the leftover text")
        return _accept_retry()
    return translated_text, False


async def generate_translation_request(main_content, previous_translation_context,
                                       source_language="English", target_language="Chinese", model=DEFAULT_MODEL,
                                       llm_client=None, log_callback=None, has_placeholders=False,
                                       prompt_options=None, context_manager: AdaptiveContextManager = None,
                                       placeholder_format: Optional[Tuple[str, str]] = None,
                                       raw_response_callback: Optional[Callable[[LLMResponse], None]] = None,
                                       on_raw_fallback: Optional[Callable[[], None]] = None,
                                       answer_check: Optional[Callable[[str, str], Optional[str]]] = None,
                                       on_rejected: Optional[Callable[[str], None]] = None):
    """
    Generate translation request to LLM API with automatic context overflow handling.

    Args:
        main_content (str): Text to translate
        previous_translation_context (str): Previous translation for consistency
        source_language (str): Source language
        target_language (str): Target language
        model (str): LLM model name
        llm_client: LLM provider instance
        log_callback (callable): Logging callback function
        has_placeholders (bool): If True, includes placeholder preservation instructions
        prompt_options (dict): Optional dict with prompt customization options
        context_manager (AdaptiveContextManager): Optional context manager for adaptive retry on overflow
        placeholder_format (Tuple[str, str]): Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format
        raw_response_callback (callable): Optional callable receiving every raw
            LLMResponse (see _make_llm_request_with_adaptive_context)
        on_raw_fallback (callable): Optional callable invoked (no arguments)
            when the returned translation came from an untagged raw response,
            so callers can count it in their stats.
        answer_check / on_rejected: See _make_llm_request_with_adaptive_context.

    Returns:
        str: Translated text or None if failed
    """
    # Skip LLM translation for single character or empty chunks
    if len(main_content.strip()) <= 1:
        if log_callback:
            log_callback("skip_translation", f"Skipping LLM for single/empty character: '{main_content}'")
        return main_content

    # Each chunk gets its own context growth budget, and a chunk that still
    # fails gives back what it grew.
    with ContextBudget(context_manager) as budget:
        translated_text, _, last_response = await _make_llm_request_with_adaptive_context(
            main_content=main_content,
            previous_translation_context=previous_translation_context,
            source_language=source_language,
            target_language=target_language,
            model=model,
            llm_client=llm_client,
            log_callback=log_callback,
            has_placeholders=has_placeholders,
            prompt_options=prompt_options,
            context_manager=context_manager,
            placeholder_format=placeholder_format,
            raw_response_callback=raw_response_callback,
            answer_check=answer_check,
            on_rejected=on_rejected,
        )
        if translated_text:
            budget.succeeded()

    if translated_text:
        translated_text, retry_won = await _retry_on_source_residue(
            translated_text,
            main_content=main_content,
            previous_translation_context=previous_translation_context,
            source_language=source_language,
            target_language=target_language,
            model=model,
            llm_client=llm_client,
            log_callback=log_callback,
            has_placeholders=has_placeholders,
            prompt_options=prompt_options,
            context_manager=context_manager,
            placeholder_format=placeholder_format,
            raw_response_callback=raw_response_callback,
            on_raw_fallback=on_raw_fallback,
            answer_check=answer_check,
            on_rejected=on_rejected,
        )
        if retry_won:
            return translated_text
        if on_raw_fallback and getattr(last_response, "was_fallback", False) is True:
            try:
                on_raw_fallback()
            except Exception as count_error:
                if log_callback:
                    log_callback("raw_fallback_count_failed",
                        f"Could not record the raw-response fallback: {count_error}")
        return translated_text
    else:
        err_msg = "ERROR: LLM API request failed"
        if log_callback:
            log_callback("llm_api_error", err_msg)
        else:
            tqdm.write(f"\n{err_msg}")
        return None



async def _refinement_request_attempt(
    draft_translation: str,
    context_before: str,
    context_after: str,
    previous_refined_context: str,
    target_language: str,
    model: str,
    llm_client,
    log_callback,
    has_placeholders: bool,
    prompt_options: dict = None,
    context_manager: AdaptiveContextManager = None,
    runtime_state: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[LLMResponse]]:
    """
    Refine one plain-text chunk through the shared request core.

    Returns:
        Tuple of (refined_text or None, last LLMResponse). None means "keep
        the draft": every guard of ``request_chunk`` applies.
    """
    refinement_instructions = prompt_options.get('refinement_instructions', '') if prompt_options else ''

    # Filter the glossary against the DRAFT (target language), on the terms'
    # target forms: the renderings that survived the first pass are the ones
    # to keep stable through refinement.
    glossary_block = _build_chunk_glossary_block(
        draft_translation, prompt_options, log_callback=log_callback,
        runtime_state=runtime_state, target_language=target_language,
        match_target_forms=True,
    )

    prompt_pair = generate_refinement_prompt(
        draft_translation=draft_translation,
        previous_refined_context=previous_refined_context,
        target_language=target_language,
        has_placeholders=False,
        prompt_options=prompt_options,
        additional_instructions=refinement_instructions,
        glossary_block=glossary_block,
    )

    client = _require_client(llm_client)
    if log_callback:
        log_callback("refinement_request", "Sending refinement request to LLM", data={
            'type': 'refinement_request',
            'system_prompt': prompt_pair.system,
            'user_prompt': prompt_pair.user,
            'model': model
        })

    outcome = await request_chunk(
        client, prompt_pair,
        source_text=draft_translation,
        allow_raw_fallback=True,
        context_manager=context_manager,
        log_callback=log_callback,
        labels=REFINEMENT_LABELS,
    )
    return (outcome.text if outcome.ok else None), outcome.response


async def _make_refinement_request(**kwargs) -> Tuple[Optional[str], Optional[LLMResponse]]:
    """Refine one chunk with its own context growth budget.

    Wraps _refinement_request_attempt: the context manager (if any) starts a
    fresh per-chunk budget, and a chunk that still fails gives back the
    context it grew, so one bad chunk cannot inflate every later request.
    """
    with ContextBudget(kwargs.get('context_manager')) as budget:
        refined_text, llm_response = await _refinement_request_attempt(**kwargs)
        if refined_text is not None:
            budget.succeeded()
    return refined_text, llm_response


async def refine_chunks(
    translated_chunks: List[str],
    original_chunks: List[Dict],
    job: "JobConfig",
    ctx: "JobContext",
) -> List[str]:
    """
    Refine translated chunks with a second pass for literary quality improvement.

    This function takes already-translated chunks and runs them through a
    refinement prompt that focuses on improving literary quality, natural flow,
    and stylistic excellence.

    Args:
        translated_chunks: List of translated text strings from first pass
        original_chunks: Original chunk dictionaries (for context structure)
        job: The job. Its refine provider (``job.provider_for(refine=True)``)
            builds the client; ``target_language``, ``context_window`` (2048
            when unset), ``auto_adjust_context`` and ``prompt_options`` are
            read from it.
        ctx: Log, stats and interruption hooks.

    Returns:
        List of refined text strings

    Raises:
        RuntimeError: the LLM client cannot be created.
        RateLimitError: re-raised with ``partial_result`` set to the complete
            list (chunks from the failing one on are the unrefined drafts).
    """
    provider = job.refine_provider_config()
    llm_provider = provider.provider or "ollama"
    model_name = provider.model or DEFAULT_MODEL
    target_language = job.target_language
    context_window = job.context_window or 2048
    auto_adjust_context = job.auto_adjust_context
    prompt_options = job.engine_prompt_options()
    log_callback = ctx.log_callback
    check_interruption_callback = ctx.check_interruption

    total_chunks = len(translated_chunks)
    refined_parts = []
    last_refined_context = ""
    # Transient per-job state (e.g. glossary cap warning dedupe) — never persisted.
    runtime_state: dict = {}

    # Chunk-counted, like every other engine. Whether this pass is the second
    # phase of a translate→refine job is the job runner's business.
    progress = ProgressReporter(ctx.stats_callback, phase=Phase.REFINING)
    progress.begin(total_chunks)
    progress.emit()

    if log_callback:
        log_callback("refinement_start", f"✨ Starting refinement pass ({total_chunks} chunks)...")

    # Determine if model is a thinking model for initial context sizing
    is_known_thinking_model = any(tm in model_name.lower() for tm in THINKING_MODELS)

    # Refinement needs MORE context than translation because:
    # - The prompt includes the already-translated text (input)
    # - Plus context before/after
    # - Plus instructions
    # So we start with at least REFINEMENT_MIN_CONTEXT or the user's context_window,
    # whichever is larger

    if auto_adjust_context:
        if is_known_thinking_model:
            initial_context = max(ADAPTIVE_CONTEXT_INITIAL_THINKING, REFINEMENT_MIN_CONTEXT)
        else:
            initial_context = max(INITIAL_CONTEXT_SIZE * 2, REFINEMENT_MIN_CONTEXT)
    else:
        initial_context = max(context_window, REFINEMENT_MIN_CONTEXT)

    # Create the provider. A failure must stop the pass: there is no safe
    # fallback client for an arbitrary provider.
    try:
        from .llm.factory import create_llm_provider
        llm_client = create_llm_provider(job.provider_for(
            context_window=initial_context, log_callback=log_callback, refine=True))
    except Exception as e:
        if log_callback:
            log_callback("refinement_client_failed",
                f"❌ Could not create the LLM client for '{llm_provider}': {e}")
        raise RuntimeError(f"Could not create the LLM client for '{llm_provider}': {e}") from e

    # Create adaptive context manager for Ollama
    context_manager = None
    if llm_provider == "ollama" and auto_adjust_context:
        from .context_optimizer import MAX_CONTEXT_SIZE
        context_manager = AdaptiveContextManager(
            initial_context=initial_context,
            context_step=CONTEXT_STEP,
            max_context=MAX_CONTEXT_SIZE,
            log_callback=log_callback
        )
        if log_callback:
            log_callback("refinement_context", f"📐 Refinement context: starting at {initial_context} tokens (min for refinement: {REFINEMENT_MIN_CONTEXT})")

    # Detect thinking model status
    if llm_provider == "ollama":
        await llm_client.detect_thinking_model()

    try:
        iterator = tqdm(
            enumerate(translated_chunks),
            total=total_chunks,
            desc=f"Refining {target_language} translation",
            unit="seg"
        ) if not log_callback else enumerate(translated_chunks)

        for i, draft_text in iterator:
            # Check for interruption
            if check_interruption_callback and check_interruption_callback():
                if log_callback:
                    log_callback("refinement_interrupted",
                        f"Refinement interrupted at chunk {i+1}/{total_chunks}")
                else:
                    tqdm.write(f"\nRefinement interrupted at chunk {i+1}/{total_chunks}")
                # Add remaining unrefined chunks as-is
                for remaining in translated_chunks[i:]:
                    refined_parts.append(remaining)
                break

            # Skip empty or very short content
            if len(draft_text.strip()) <= 1:
                refined_parts.append(draft_text)
                progress.record()
                progress.emit()
                continue

            # Get context from original chunks if available
            context_before = ""
            context_after = ""
            if i < len(original_chunks):
                context_before = original_chunks[i].get("context_before", "")
                context_after = original_chunks[i].get("context_after", "")

            # Make refinement request
            try:
                refined_text, llm_response = await _make_refinement_request(
                    draft_translation=draft_text,
                    context_before=context_before,
                    context_after=context_after,
                    previous_refined_context=last_refined_context,
                    target_language=target_language,
                    model=model_name,
                    llm_client=llm_client,
                    log_callback=log_callback,
                    has_placeholders=False,
                    prompt_options=prompt_options,
                    context_manager=context_manager,
                    runtime_state=runtime_state,
                )
            except RateLimitError as e:
                if log_callback:
                    retry_msg = f" (retry after ~{e.retry_after}s)" if e.retry_after else ""
                    log_callback("rate_limit_pause",
                        f"⏸️ Rate limited by {e.provider or 'API'}{retry_msg}. "
                        f"Auto-pausing refinement at chunk {i+1}/{total_chunks}...")
                # Add remaining unrefined chunks as-is
                for remaining in translated_chunks[i:]:
                    refined_parts.append(remaining)
                # Invariant: len(refined_parts) == len(translated_chunks) here —
                # the list is complete, chunks at index >= i are the unrefined
                # drafts. Hand it to the caller so the partial pass can be saved.
                e.partial_result = list(refined_parts)
                raise  # Re-raise to handlers.py

            # Record success in context manager
            if refined_text is not None and llm_response and context_manager:
                context_manager.record_success(
                    prompt_tokens=llm_response.prompt_tokens,
                    completion_tokens=llm_response.completion_tokens,
                    context_limit=llm_response.context_limit
                )

            if refined_text is not None:
                refined_text = clean_translated_text(refined_text, target_language=target_language)
                if not refined_text.strip():
                    # Never let an empty refinement erase the chunk.
                    refined_text = None

            if refined_text is not None:
                refined_parts.append(refined_text)
                progress.record()

                # Update context for next chunk
                last_refined_context = previous_context_tail(refined_text)
            else:
                # Keep original translation if refinement fails
                if log_callback:
                    log_callback("refinement_chunk_failed",
                        f"Refinement failed for chunk {i+1}, keeping original translation")
                refined_parts.append(draft_text)
                progress.record(succeeded=False)
                last_refined_context = ""

            progress.emit()

    finally:
        if llm_client:
            await llm_client.close()

    if log_callback:
        log_callback("refinement_complete",
            f"✨ Refinement complete: {progress.succeeded} refined, {progress.failed} kept original")

    return refined_parts


# Subtitle translation functions moved to subtitle_translator.py