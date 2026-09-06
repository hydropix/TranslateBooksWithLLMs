"""
Plain-text translation pipeline used by Plain Text Mode.

Skips placeholder preservation and HTML chunking entirely. Paragraphs are
grouped into token-budgeted segments that remember which source paragraph
indices they cover, translated with has_placeholders=False, then written back
to those exact indices. Empty source paragraphs (image-only blocks) are never
sent to the LLM and keep their slot; a paragraph larger than the token budget
is split into sentence pieces that all collapse back into its single slot
(issue #203: count-only realignment shifted every paragraph after an empty or
oversized block).

Used by the EPUB and DOCX adapters when prompt_options['plain_text_mode'] is True.
"""
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from src.core.chunking.token_chunker import TokenChunker
from src.core.translator import generate_translation_request
from src.core.post_processor import clean_translated_text
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.common.parallel import aclosing, iter_ordered_concurrent
from src.core.llm.exceptions import RateLimitError
from src.prompts.prompts import PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION


PARAGRAPH_SEPARATOR = "\n\n"
_RESPLIT_REGEX = re.compile(r"\n{2,}")
_MARKUP_TAG_REGEX = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*?)?/?>")
# A markdown heading marker at the START OF A PARAGRAPH: the blob is split on
# blank lines, so \A or a blank-line separator is what anchors the match, and
# the separator is put back by the replacement. Requiring at least one space
# followed by a non-space keeps "#1 bestseller", "#MeToo" and a lone "#" out
# of it, and keeps the paragraph count of the segment unchanged.
_MARKDOWN_HEADING_MARKER_REGEX = re.compile(r"(\A|\n{2,})[ \t]*#{1,6}[ \t]+(?=\S)")


def strip_hallucinated_markup(translated: str, source: str) -> str:
    """Remove HTML-like tags the model invented in Plain Text Mode.

    Plain Text Mode never sends markup to the LLM, so a tag in the output is
    model noise (e.g. small models wrap ordinals or footnote numbers in
    <sup>...</sup>). Only the tags are dropped; their inner text is kept.
    Chunks whose source legitimately contains '<' (code samples inside <pre>
    blocks) are left untouched to avoid damaging real content.
    """
    if "<" not in translated or "<" in source:
        return translated
    return _MARKUP_TAG_REGEX.sub("", translated)


def strip_hallucinated_markdown_markers(translated: str, source: str) -> str:
    """Remove markdown heading markers the model invented in Plain Text Mode.

    Plain Text Mode sends the LLM a bare heading line like "Chapter 6: Define
    Secrets and Clues" with no structure, and models conditioned on markdown
    titles tend to echo a "# " prefix back. The heading tag already carries the
    structure, so the marker is content-adding noise.

    Every paragraph of the blob is checked, not just the first: a segment holds
    many paragraphs and the decorated one is rarely the leading one. Like
    strip_hallucinated_markup, a source that legitimately uses the marker (a
    markdown sample inside a <pre> block) disarms the whole thing rather than
    risk damaging real content.
    """
    if not translated or "#" not in translated:
        return translated
    if _MARKDOWN_HEADING_MARKER_REGEX.search(source or ""):
        return translated
    return _MARKDOWN_HEADING_MARKER_REGEX.sub(r"\1", translated)


def _split_translated_back_to_paragraphs(translated_text: str) -> List[str]:
    """Split a translated blob into paragraphs (tolerates 2+ newlines)."""
    return [p.strip() for p in _RESPLIT_REGEX.split(translated_text) if p.strip()]


def _paragraph_count_mismatch(
    translated_text: str,
    expected_count: int,
) -> Optional[Tuple[int, int]]:
    """Return (got, expected) when the model's paragraph count differs, else None.

    Counting goes through _split_translated_back_to_paragraphs so detection and
    _reconcile_paragraph_counts can never disagree on what a paragraph is: a
    mismatch reported here is exactly a case where the reconciliation below
    pads or merges, which is what silently injects source text (issue #253).
    """
    got = len(_split_translated_back_to_paragraphs(translated_text))
    if got == expected_count:
        return None
    return got, expected_count


def _reconcile_paragraph_counts(
    translated_paragraphs: List[str],
    expected_count: int,
) -> List[str]:
    """
    Best-effort alignment when the LLM merged or split paragraphs inside one
    segment. The blast radius is the segment, never the whole document.

    - translated == expected: return as-is
    - translated < expected: pad with empty strings
    - translated > expected: merge surplus into the last slot
    """
    got = len(translated_paragraphs)
    if got == expected_count:
        return translated_paragraphs
    if got < expected_count:
        return translated_paragraphs + [""] * (expected_count - got)
    head = translated_paragraphs[:expected_count - 1]
    tail = " ".join(translated_paragraphs[expected_count - 1:])
    return head + [tail]


async def _retry_segment_with_paragraph_count(
    main_content: str,
    expected_count: int,
    *,
    translate_segment: Callable[[str, Dict[str, Any]], Awaitable[Optional[str]]],
    prompt_options: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Re-translate a mismatched segment once, as one call, with the count stated.

    This is the cheap half of the #253 repair. The per-paragraph repair below
    makes the alignment exact by construction, but it costs one LLM call per
    paragraph the segment covers - on a book whose chapters open with front
    matter the model treats as metadata, that is most first segments, and a
    segment holds tens of paragraphs. A single retry that names the expected
    count recovers the common case for one call instead of N.

    Returns the cleaned translation when it comes back with exactly
    `expected_count` paragraphs, otherwise None - and None always means "fall
    through to the per-paragraph repair", never "accept as is". The acceptance
    test is the same _paragraph_count_mismatch used everywhere else, so a retry
    can never widen what counts as an aligned segment.

    The count is passed through a *copy* of prompt_options: the caller's dict is
    shared by every other chunk in the run and must not learn about this segment.
    """
    options = dict(prompt_options or {})
    options[PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION] = expected_count

    answer = await translate_segment(main_content, options)
    if not answer:
        return None

    cleaned = clean_translated_text(answer)
    cleaned = strip_hallucinated_markup(cleaned, main_content)
    cleaned = strip_hallucinated_markdown_markers(cleaned, main_content)
    if not cleaned.strip():
        return None
    if _paragraph_count_mismatch(cleaned, expected_count) is not None:
        return None
    return cleaned


async def _repair_segment_by_paragraph(
    segment: Dict[str, Any],
    source_paragraphs: List[str],
    *,
    translate_one: Callable[[str, str, str], Awaitable[Optional[str]]],
    log_callback: Optional[Callable],
    segment_index: int,
    context_before: str = "",
    context_after: str = "",
) -> Tuple[str, int]:
    """Re-translate each paragraph of a segment individually.

    Returns (joined_text, failed_count). joined_text contains exactly
    len(segment['indices']) paragraphs separated by PARAGRAPH_SEPARATOR.
    A paragraph whose translation fails or comes back empty contributes its
    source text and increments failed_count.

    One LLM call per paragraph is what makes the alignment exact: the mapping
    is positional by construction, so nothing has to guess which output
    paragraph belongs to which input one (issue #253).

    translate_one(text, context_before, context_after) is injected so this stays
    testable without an LLM client. The neighbouring source paragraphs of the
    same segment are the context; at the edges the segment's own context is
    used, which is why it is passed in rather than read from the segment dict
    (that dict is what the checkpoint serializes and its shape must not move).
    """
    indices = segment['indices']
    texts = [source_paragraphs[idx] for idx in indices]

    repaired: List[str] = []
    failed = 0
    for j, text in enumerate(texts):
        before = texts[j - 1] if j > 0 else context_before
        after = texts[j + 1] if j + 1 < len(texts) else context_after

        translated = await translate_one(text, before, after)
        cleaned = ""
        if translated:
            cleaned = clean_translated_text(translated)
            cleaned = strip_hallucinated_markup(cleaned, text)
            cleaned = strip_hallucinated_markdown_markers(cleaned, text)
            # One paragraph in, one paragraph out, always: a model that split
            # this single paragraph would otherwise re-introduce the very
            # misalignment the repair exists to remove. Never recurse.
            cleaned = " ".join(_split_translated_back_to_paragraphs(cleaned))

        if not cleaned:
            # The only remaining path that puts source text in the output, so
            # it is counted and logged instead of silently padded.
            cleaned = text
            failed += 1
        repaired.append(cleaned)

    if failed and log_callback:
        log_callback(
            "plain_text_paragraph_repair_failed",
            f"⚠️ Segment {segment_index + 1}: {failed} of {len(texts)} paragraph(s) "
            "could not be re-translated - they keep their source text"
        )

    return PARAGRAPH_SEPARATOR.join(repaired), failed


def build_plain_segments(
    paragraphs: List[str],
    max_tokens_per_chunk: int,
) -> List[Dict[str, Any]]:
    """
    Group source paragraphs into translation segments that track their indices.

    Each segment is {'indices': [int, ...], 'text': str, 'partial': bool}:
    - whole-paragraph segments cover consecutive non-empty paragraphs joined
      with PARAGRAPH_SEPARATOR ('partial' False, one index per paragraph);
    - an oversized paragraph yields several sentence-piece segments that share
      the same single index ('partial' True).

    Empty/whitespace-only paragraphs are skipped here and restored by index at
    reassembly time.
    """
    chunker = TokenChunker(max_tokens=max_tokens_per_chunk)
    sep_tokens = chunker.count_tokens(PARAGRAPH_SEPARATOR)

    segments: List[Dict[str, Any]] = []
    cur_indices: List[int] = []
    cur_texts: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cur_indices, cur_texts, cur_tokens
        if cur_indices:
            segments.append({
                'indices': cur_indices,
                'text': PARAGRAPH_SEPARATOR.join(cur_texts),
                'partial': False,
            })
            cur_indices, cur_texts, cur_tokens = [], [], 0

    for idx, paragraph in enumerate(paragraphs):
        text = paragraph or ""
        if not text.strip():
            continue

        tokens = chunker.count_tokens(text)

        if tokens > chunker.max_tokens:
            flush()
            sentences = chunker.split_paragraph_into_sentences(text)
            if len(sentences) > 1:
                # _chunk_units returns {"text", "join_before"} records; this
                # pipeline tracks paragraph identity itself and only needs the text.
                pieces = [p["text"] for p in chunker._chunk_units(sentences, separator=" ")]
            else:
                pieces = [text]
            for piece in pieces:
                segments.append({'indices': [idx], 'text': piece, 'partial': True})
            continue

        potential = cur_tokens + tokens + (sep_tokens if cur_indices else 0)
        if cur_indices and potential > chunker.max_tokens:
            flush()
        cur_indices.append(idx)
        cur_texts.append(text)
        cur_tokens = cur_tokens + tokens + (sep_tokens if len(cur_indices) > 1 else 0)

    flush()
    return segments


def _reassemble(
    segments: List[Dict[str, Any]],
    translated_parts: List[str],
    source_paragraphs: List[str],
) -> List[str]:
    """
    Write each segment's translation back to the source indices it covers.

    Empty source slots keep their original (empty) value; pieces of an
    oversized paragraph are concatenated in order into its single slot.
    """
    out: List[Optional[str]] = [None] * len(source_paragraphs)
    partial_pieces: Dict[int, List[str]] = {}

    for segment, translated in zip(segments, translated_parts):
        text = translated or ""
        if segment['partial']:
            partial_pieces.setdefault(segment['indices'][0], []).append(text.strip())
        else:
            parts = _split_translated_back_to_paragraphs(text)
            parts = _reconcile_paragraph_counts(parts, len(segment['indices']))
            for k, idx in enumerate(segment['indices']):
                out[idx] = parts[k]

    for idx, pieces in partial_pieces.items():
        out[idx] = " ".join(p for p in pieces if p)

    return [
        slot if slot is not None else source_paragraphs[i]
        for i, slot in enumerate(out)
    ]


async def translate_paragraphs_plain(
    paragraphs: List[str],
    source_language: str,
    target_language: str,
    model_name: str,
    llm_client: Any,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    context_manager: Optional[Any] = None,
    check_interruption_callback: Optional[Callable] = None,
    prompt_options: Optional[Dict] = None,
    parallel_workers: int = 1,
    *,
    resume_segments: Optional[List[Dict[str, Any]]] = None,
    resume_translated: Optional[List[str]] = None,
    checkpoint_hook: Optional[Callable[[List[Dict[str, Any]], List[str], int, Dict[str, Any]], None]] = None,
    checkpoint_every: int = 5,
) -> Tuple[List[str], TranslationMetrics, bool]:
    """
    Translate a list of plain-text paragraphs without placeholder preservation.

    Args:
        paragraphs: source paragraphs (one string per block)
        source_language, target_language: language names
        model_name, llm_client: LLM config
        max_tokens_per_chunk: chunking budget
        log_callback, stats_callback: callbacks (stats_callback receives
            file-local stats via TranslationMetrics.to_dict(); callers that
            aggregate across files are responsible for adding their global
            offset to completed_chunks).
        context_manager: AdaptiveContextManager (Ollama)
        check_interruption_callback: returns True to abort
        prompt_options: prompt customization (text_cleanup, glossary, etc.)
        parallel_workers: number of chunks translated concurrently (already
            resolved against the provider by the caller). When 1, behavior is
            identical to the legacy sequential loop, including previous-chunk
            context chaining; > 1 drops that chaining.
        resume_segments: segment list from a previous run's checkpoint, replayed
            verbatim instead of re-deriving it from `paragraphs`. Storing the
            segmentation rather than rebuilding it makes resume immune to a
            token-budget change between the pause and the resume.
        resume_translated: translations already produced for the first
            len(resume_translated) segments. Those segments are never retried,
            including ones that fell back to source text after a failure.
        checkpoint_hook: called as
            hook(segments, prefix, next_index, stats_dict) whenever the
            contiguous translated prefix advances far enough to be worth
            persisting. `prefix` is always exactly `next_index` items long and
            contains no None. A hook that raises is logged and ignored.
        checkpoint_every: how many segments between periodic hook calls.

    Returns:
        (translated_paragraphs, stats, was_interrupted)
    """
    stats = TranslationMetrics()

    source = list(paragraphs)
    if not source or all(not (p or "").strip() for p in source):
        if stats_callback:
            stats_callback(stats.to_dict())
        return source, stats, False

    # === RESUME ===
    # build_plain_segments is deterministic for a given (paragraphs, budget),
    # but the stored segmentation always wins so a budget change between pause
    # and resume cannot shift the indices the prefix was written against.
    prefix: List[str] = list(resume_translated or [])
    segments = (
        list(resume_segments) if resume_segments is not None
        else build_plain_segments(source, max_tokens_per_chunk)
    )
    if prefix and (resume_segments is None or len(prefix) > len(segments)):
        # More translated segments than segments to translate means the source
        # changed under the checkpoint; nothing about the prefix can be trusted.
        if log_callback:
            log_callback(
                "plain_text_resume_discarded",
                "⚠️ Plain-text checkpoint does not match the source, restarting this file"
            )
        prefix = []
        segments = build_plain_segments(source, max_tokens_per_chunk)

    # Chunk dicts mirror split_text_into_chunks() output; context comes from
    # the neighboring segments.
    chunks: List[Dict[str, str]] = []
    for i, segment in enumerate(segments):
        if i > 0:
            context_before = segments[i - 1]['text'].split(PARAGRAPH_SEPARATOR)[-1]
        else:
            context_before = ""
        if i < len(segments) - 1:
            context_after = segments[i + 1]['text'].split(PARAGRAPH_SEPARATOR)[0]
        else:
            context_after = ""
        chunks.append({
            'context_before': context_before,
            'main_content': segment['text'],
            'context_after': context_after,
        })

    stats.total_chunks = len(chunks)

    workers = max(1, int(parallel_workers))
    sequential = workers == 1

    # Index-addressed results so out-of-order completion still reassembles in
    # source order.
    translated_parts: List[Optional[str]] = [None] * len(chunks)
    previous_translation_context = ""

    # Restored work counts as processed so the progress bar does not rewind.
    for k, done in enumerate(prefix):
        translated_parts[k] = done
        stats.record_processed()

    if stats_callback:
        stats_callback(stats.to_dict())

    async def _request(main_content, context_before, context_after, previous, options=None):
        """Single Plain Text Mode LLM call, shared by the segment loop, the
        count-stating retry and the per-paragraph repair so the three can never
        drift apart. `options` overrides prompt_options for that one call only.
        """
        return await generate_translation_request(
            main_content=main_content,
            context_before=context_before,
            context_after=context_after,
            previous_translation_context=previous,
            source_language=source_language,
            target_language=target_language,
            model=model_name,
            llm_client=llm_client,
            log_callback=log_callback,
            has_placeholders=False,
            prompt_options=prompt_options if options is None else options,
            context_manager=context_manager,
            placeholder_format=None,
        )

    async def _translate_chunk(i):
        """Translate one chunk. Reads previous_translation_context only in
        sequential mode (parallel runs have no stable previous chunk)."""
        main_content = chunks[i].get('main_content', '')
        if not main_content.strip():
            return ('empty', main_content)
        translated = await _request(
            main_content,
            chunks[i].get('context_before', ''),
            chunks[i].get('context_after', ''),
            previous_translation_context if sequential else "",
        )
        return ('done', translated)

    async def _translate_one_paragraph(text, context_before, context_after):
        """Repair call for a single paragraph.

        previous_translation_context is empty on purpose: the repair replays
        paragraphs the sequential chain has already moved past, so feeding it
        the running context would describe the wrong position in the text.
        """
        return await _request(text, context_before, context_after, "")

    def _fill_remaining_with_source():
        for j in range(len(chunks)):
            if translated_parts[j] is None:
                translated_parts[j] = chunks[j].get('main_content', '')

    def _run_checkpoint_hook(next_index):
        """Hand the contiguous prefix [0, next_index) to the caller's hook.

        A failing hook degrades persistence, not the translation, so every call
        is isolated.
        """
        if checkpoint_hook is None:
            return
        try:
            checkpoint_hook(
                segments, list(translated_parts[:next_index]), next_index, stats.to_dict()
            )
        except Exception as exc:  # noqa: BLE001 - checkpointing is best-effort
            if log_callback:
                log_callback(
                    "plain_text_checkpoint_failed",
                    f"⚠️ Plain-text checkpoint failed at segment {next_index}/{len(chunks)}: {exc}"
                )

    checkpoint_step = max(1, int(checkpoint_every))
    pending = list(range(len(prefix), len(chunks)))
    rate_limit_error = None
    processed = len(prefix)

    # Continuous concurrency with in-order delivery (see iter_ordered_concurrent).
    # aclosing() is required: the rate-limit branch breaks out of the loop, and
    # only closing the generator cancels the requests still in flight.
    async with aclosing(iter_ordered_concurrent(
        pending, workers, _translate_chunk, check_interruption_callback
    )) as stream:
        async for i, result in stream:
            main_content = chunks[i].get('main_content', '')

            if isinstance(result, RateLimitError):
                rate_limit_error = result
                break

            if isinstance(result, Exception):
                if log_callback:
                    log_callback(
                        "plain_text_chunk_failed",
                        f"Chunk {i + 1}/{len(chunks)} failed ({result}) - keeping original text"
                    )
                translated_parts[i] = main_content
                stats.failed_chunks += 1
            else:
                kind, value = result
                if kind == 'empty':
                    translated_parts[i] = value
                    stats.successful_first_try += 1
                elif value is None:
                    if log_callback:
                        log_callback(
                            "plain_text_chunk_failed",
                            f"Chunk {i + 1}/{len(chunks)} failed - keeping original text"
                        )
                    translated_parts[i] = main_content
                    stats.failed_chunks += 1
                else:
                    cleaned = clean_translated_text(value)
                    cleaned = strip_hallucinated_markup(
                        cleaned, chunks[i].get('main_content', ''))
                    cleaned = strip_hallucinated_markdown_markers(
                        cleaned, chunks[i].get('main_content', ''))
                    translated_parts[i] = cleaned
                    stats.successful_first_try += 1
                    # A wrong paragraph count is reconciled silently at
                    # reassembly time (padding with empty slots, which then fall
                    # back to source text). Count it, report it, and repair it
                    # here, where the segment is still identifiable.
                    # Partial segments are pieces of one oversized paragraph:
                    # they are joined back by design and have no count contract.
                    if not segments[i]['partial']:
                        expected = len(segments[i]['indices'])
                        mismatch = _paragraph_count_mismatch(cleaned, expected)
                        if mismatch is not None:
                            got, _ = mismatch
                            # Counted whether or not the repair runs, so the
                            # metric measures model behaviour, not our reaction.
                            stats.paragraph_count_mismatches += 1
                            if got > expected and expected == 1:
                                # Benign: the model split the one paragraph this
                                # segment owns, and _reconcile_paragraph_counts
                                # merges the surplus straight back into its
                                # single slot - no source text can survive. The
                                # plan asks for debug level; log_callback has no
                                # level parameter, so the level maps to a quiet
                                # event id instead of the mismatch warning.
                                if log_callback:
                                    log_callback(
                                        "plain_text_paragraph_split_benign",
                                        f"Chunk {i + 1}/{len(chunks)} (segment {i}): the model "
                                        f"split the paragraph into {got} parts - merged back "
                                        "into its single slot"
                                    )
                            else:
                                if log_callback:
                                    log_callback(
                                        "plain_text_paragraph_mismatch",
                                        f"⚠️ Chunk {i + 1}/{len(chunks)} (segment {i}): the model "
                                        f"returned {got} paragraph(s) instead of {expected} - "
                                        "re-translating this segment"
                                    )
                                # One retry of the whole segment first, stating
                                # the count the answer must have. It recovers the
                                # common case for a single call; the exact-by-
                                # construction repair below costs one call per
                                # paragraph and stays as the fallback. Skipped
                                # for a one-paragraph segment, where the repair
                                # is already a single call and a retry would only
                                # add one. Accepted only on an exact count match,
                                # so the alignment guarantee is unchanged.
                                retried = None
                                if expected > 1:
                                    retried = await _retry_segment_with_paragraph_count(
                                        main_content,
                                        expected,
                                        translate_segment=lambda text, options: _request(
                                            text,
                                            chunks[i].get('context_before', ''),
                                            chunks[i].get('context_after', ''),
                                            previous_translation_context if sequential else "",
                                            options,
                                        ),
                                        prompt_options=prompt_options,
                                    )
                                if retried is not None:
                                    # The common tail below picks `cleaned` up
                                    # for the sequential context chain and runs
                                    # the checkpoint hook, exactly as it does for
                                    # a segment that never mismatched.
                                    cleaned = retried
                                    translated_parts[i] = retried
                                    stats.paragraph_retry_recovered += 1
                                    if log_callback:
                                        log_callback(
                                            "plain_text_paragraph_retry_recovered",
                                            f"Chunk {i + 1}/{len(chunks)} (segment {i}): the retry "
                                            f"returned the expected {expected} paragraph(s) - no "
                                            "per-paragraph repair needed"
                                        )
                                else:
                                    if log_callback:
                                        log_callback(
                                            "plain_text_paragraph_repair_started",
                                            f"⚠️ Chunk {i + 1}/{len(chunks)} (segment {i}): the "
                                            f"retry did not return {expected} paragraph(s) - "
                                            "re-translating this segment paragraph by paragraph"
                                        )
                                    # Repaired here, inside the ordered consumer
                                    # loop and before the checkpoint hook below,
                                    # so the persisted prefix always holds the
                                    # repaired text. Never inside
                                    # _translate_chunk: that runs under the
                                    # worker semaphore and would deadlock.
                                    repaired, repair_failed = await _repair_segment_by_paragraph(
                                        segments[i],
                                        source,
                                        translate_one=_translate_one_paragraph,
                                        log_callback=log_callback,
                                        segment_index=i,
                                        context_before=chunks[i].get('context_before', ''),
                                        context_after=chunks[i].get('context_after', ''),
                                    )
                                    if _paragraph_count_mismatch(repaired, expected) is None:
                                        # Single joined string, as the checkpoint
                                        # format and _reassemble both expect.
                                        cleaned = repaired
                                        translated_parts[i] = repaired
                                        if repair_failed:
                                            stats.paragraph_repair_failed += 1
                                    else:
                                        # Forbidden by the helper's contract;
                                        # keep the pre-repair value and say so.
                                        # Attempted once per segment - no second
                                        # round.
                                        stats.paragraph_repair_failed += 1
                                        if log_callback:
                                            log_callback(
                                                "plain_text_paragraph_repair_failed",
                                                f"⚠️ Chunk {i + 1}/{len(chunks)} (segment {i}): "
                                                "the per-paragraph repair still returned the wrong "
                                                "count - keeping the original translation"
                                            )
                    if sequential:
                        words = cleaned.split()
                        previous_translation_context = (
                            " ".join(words[-25:]) if len(words) > 25 else cleaned
                        )

            stats.record_processed()
            if stats_callback:
                stats_callback(stats.to_dict())
            processed += 1

            # === CONTIGUITY INVARIANT ===
            # iter_ordered_concurrent yields indices strictly in ascending order
            # and every branch above assigns translated_parts[i] (success, empty,
            # None result, or exception -> source fallback). Therefore, once index
            # i has been handled, slots 0..i are all non-None and i + 1 is a
            # gap-free resume point. Every hook call below relies on this: the
            # prefix handed to the checkpoint is never sparse.
            next_index = i + 1
            if next_index % checkpoint_step == 0 or next_index == len(chunks):
                _run_checkpoint_hook(next_index)

    if rate_limit_error is not None:
        # Persist the contiguous prefix translated before the limit, then keep
        # source text for everything else and propagate: the caller's auto-pause
        # depends on the exception reaching it.
        _run_checkpoint_hook(processed)
        _fill_remaining_with_source()
        safe_parts = [p if p is not None else "" for p in translated_parts]
        rate_limit_error.partial_result = _reassemble(segments, safe_parts, source)
        raise rate_limit_error

    # Interruption: the scheduler stopped launching new chunks; keep source text
    # for the uncommitted tail and report the interruption.
    if processed < len(chunks) and check_interruption_callback and check_interruption_callback():
        if log_callback:
            log_callback(
                "plain_text_translation_interrupted",
                f"⏸️ Plain-text translation interrupted at chunk {processed + 1}/{len(chunks)}"
            )
        _run_checkpoint_hook(processed)
        _fill_remaining_with_source()
        safe_parts = [p if p is not None else "" for p in translated_parts]
        return _reassemble(segments, safe_parts, source), stats, True

    # Any None left (shouldn't happen) falls back to empty string.
    safe_parts = [p if p is not None else "" for p in translated_parts]
    return _reassemble(segments, safe_parts, source), stats, False
