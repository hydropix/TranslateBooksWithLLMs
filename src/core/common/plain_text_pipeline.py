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
import math
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from src.core.chunking.token_chunker import TokenChunker
from src.core.translator import generate_translation_request
from src.core.post_processor import clean_translated_text
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.xhtml_translation_state import (
    CHUNK_PENDING,
    CHUNK_TRANSLATED,
    CHUNK_UNTRANSLATED,
    unfinished_chunk_indices,
)
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


# A range this small is repaired paragraph by paragraph straight away: bisecting
# it cannot beat its own paragraph count (issue #259).
BISECTION_PER_PARAGRAPH_MAX = 3


def bisection_repair_call_budget(paragraph_count: int) -> int:
    """Most LLM calls the bisection repair may spend on one segment.

    The old one-call-per-paragraph cost plus two calls per bisection level.
    """
    if paragraph_count <= 1:
        return paragraph_count
    return paragraph_count + 2 * math.ceil(math.log2(paragraph_count))


async def _repair_segment_by_bisection(
    segment: Dict[str, Any],
    source_paragraphs: List[str],
    *,
    translate_range: Callable[[str, str, str, Dict[str, Any]], Awaitable[Optional[str]]],
    translate_one: Callable[[str, str, str], Awaitable[Optional[str]]],
    prompt_options: Optional[Dict[str, Any]] = None,
    log_callback: Optional[Callable],
    segment_index: int,
    context_before: str = "",
    context_after: str = "",
) -> Tuple[str, int, int]:
    """Repair a mismatched segment by locating the divergence (issue #259).

    The segment already failed twice (first attempt and the count-stating
    retry), but a wrong count is almost always local: most mismatches are off
    by exactly one paragraph. Re-translating every paragraph to fix one
    boundary is what made the old repair cost one call per paragraph.

    The segment is split in two halves, each re-translated as one call with
    its expected count stated. A half that comes back with the exact count is
    accepted as is; a half that does not is split again, so the search narrows
    onto the misbehaving paragraphs. A range of BISECTION_PER_PARAGRAPH_MAX
    paragraphs or fewer goes through _repair_segment_by_paragraph, and a
    one-paragraph half is translated directly with its surplus merged back.

    A model failing across the board (both halves mismatch at every level)
    would make blind bisection cost more than the old repair, so the search
    runs under a call budget, bisection_repair_call_budget(len): a split is
    only made while every still-unresolved paragraph could be finished one
    call each within it, otherwise the range goes paragraph by paragraph. The
    worst case is therefore the old cost plus two calls per bisection level.
    A +1/-1 mismatch on a 53-paragraph segment costs 2 calls when the halves
    obey the stated count, and at most 12 when one paragraph keeps failing.

    The alignment guarantee is unchanged: a half is only accepted on an exact
    count match (the same _paragraph_count_mismatch test as everywhere else),
    and everything below that is positional by construction.

    translate_range(text, context_before, context_after, options) and
    translate_one(text, context_before, context_after) are injected so this
    stays testable without an LLM client.

    Returns (joined_text, failed_count, calls_made). joined_text holds exactly
    len(segment['indices']) paragraphs separated by PARAGRAPH_SEPARATOR.
    """
    texts = [source_paragraphs[idx] for idx in segment['indices']]
    calls = 0
    failed = 0

    def _context(lo: int, hi: int) -> Tuple[str, str]:
        before = texts[lo - 1] if lo > 0 else context_before
        after = texts[hi] if hi < len(texts) else context_after
        return before, after

    async def _counting_one(text, before, after):
        nonlocal calls
        calls += 1
        return await translate_one(text, before, after)

    async def _per_paragraph(lo: int, hi: int) -> List[str]:
        nonlocal failed
        before, after = _context(lo, hi)
        joined, range_failed = await _repair_segment_by_paragraph(
            {'indices': segment['indices'][lo:hi]},
            source_paragraphs,
            translate_one=_counting_one,
            log_callback=None,
            segment_index=segment_index,
            context_before=before,
            context_after=after,
        )
        failed += range_failed
        return joined.split(PARAGRAPH_SEPARATOR)

    async def _attempt(lo: int, hi: int) -> Optional[List[str]]:
        """One call for texts[lo:hi]; the paragraphs on an exact count, else None."""
        nonlocal calls
        if hi - lo == 1:
            return await _per_paragraph(lo, hi)
        before, after = _context(lo, hi)
        main_content = PARAGRAPH_SEPARATOR.join(texts[lo:hi])

        async def _translate_segment(text, options):
            return await translate_range(text, before, after, options)

        calls += 1
        answer = await _retry_segment_with_paragraph_count(
            main_content,
            hi - lo,
            translate_segment=_translate_segment,
            prompt_options=prompt_options,
        )
        if answer is None:
            return None
        return _split_translated_back_to_paragraphs(answer)

    async def _resolve(lo: int, hi: int) -> List[str]:
        """Translate texts[lo:hi], a range already known to mismatch."""
        nonlocal resolved
        # Every unresolved paragraph can always be finished at one call each,
        # so a split (two calls) is only made while that fallback still fits.
        affordable = calls + 2 + (total - resolved) <= budget
        if hi - lo <= BISECTION_PER_PARAGRAPH_MAX or not affordable:
            out = await _per_paragraph(lo, hi)
            resolved += hi - lo
            return out
        mid = lo + (hi - lo) // 2
        left = await _attempt(lo, mid)
        if left is not None:
            resolved += mid - lo
        right = await _attempt(mid, hi)
        if right is not None:
            resolved += hi - mid
        if left is None:
            left = await _resolve(lo, mid)
        if right is None:
            right = await _resolve(mid, hi)
        return left + right

    total = len(texts)
    resolved = 0
    budget = bisection_repair_call_budget(total)

    repaired = await _resolve(0, len(texts))

    if failed and log_callback:
        log_callback(
            "plain_text_paragraph_repair_failed",
            f"⚠️ Segment {segment_index + 1}: {failed} of {len(texts)} paragraph(s) "
            "could not be re-translated - they keep their source text"
        )

    return PARAGRAPH_SEPARATOR.join(repaired), failed, calls


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
    resume_statuses: Optional[List[str]] = None,
    resume_stats: Optional[Dict[str, Any]] = None,
    checkpoint_hook: Optional[Callable[..., None]] = None,
    checkpoint_every: int = 5,
    count_failures_as_fallback: bool = False,
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
            len(resume_translated) segments. Those segments are not sent again,
            except the ones `resume_statuses` marks as CHUNK_UNTRANSLATED.
        resume_statuses: per-segment statuses from the same checkpoint (the
            CHUNK_* constants of xhtml_translation_state). A segment marked
            CHUNK_UNTRANSLATED fell back to its source text in an earlier pass
            and is re-translated even though it sits inside the restored prefix
            (issue #285). None, or a list that does not line up with the
            segments, means the legacy reading: the whole prefix is translated.
        resume_stats: the checkpoint's serialized TranslationMetrics. When
            given, the file-local counters continue from it instead of starting
            at zero, exactly like the placeholder pipeline's resume.
        checkpoint_hook: called as
            hook(segments, prefix, next_index, stats_dict, chunk_statuses=...)
            whenever the contiguous translated prefix advances far enough to be
            worth persisting, and after every retried segment. `prefix` is
            always exactly `next_index` items long and contains no None;
            `chunk_statuses` has one entry per segment and is CHUNK_PENDING
            exactly from `next_index` on. A hook that raises is logged and
            ignored.
        checkpoint_every: how many segments between periodic hook calls.
        count_failures_as_fallback: record a segment that failed (and kept its
            source text) in `fallback_used` instead of `failed_chunks`. The EPUB
            adapter sets it: there the failure is also persisted as
            CHUNK_UNTRANSLATED and is retryable, i.e. the same outcome as the
            placeholder pipeline's Phase 3 fallback, and the job verdict must
            be able to clear once the retry succeeds - which a `failed_chunks`
            count restored across passes never does. DOCX keeps the historical
            counter.

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

    # Per-segment statuses: what decides which segments this pass sends. The
    # restored prefix is translated unless the checkpoint says a segment fell
    # back to its source text; everything past it has never been attempted. A
    # list that does not line up with the segments describes something else
    # and is ignored rather than trusted (same rule as the placeholder
    # pipeline's chunk loop).
    statuses: List[str] = [CHUNK_TRANSLATED] * len(prefix) + [CHUNK_PENDING] * (len(segments) - len(prefix))
    if prefix and resume_statuses is not None and len(resume_statuses) == len(segments):
        for k in range(len(prefix)):
            if resume_statuses[k] == CHUNK_UNTRANSLATED:
                statuses[k] = CHUNK_UNTRANSLATED
    retry_indices = [k for k in range(len(prefix)) if statuses[k] == CHUNK_UNTRANSLATED]

    if prefix and resume_stats:
        stats = TranslationMetrics.from_dict(resume_stats)

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
    # Restored stats already count it; counting it again would double it.
    for k, done in enumerate(prefix):
        translated_parts[k] = done
        if not (prefix and resume_stats):
            stats.record_processed()

    if retry_indices and log_callback:
        log_callback(
            "plain_text_resume_retry",
            f"🔁 Retrying {len(retry_indices)} segment(s) that kept their source "
            f"text in an earlier pass: {[k + 1 for k in retry_indices]}"
        )

    if stats_callback:
        stats_callback(stats.to_dict())

    def _context_tail(text):
        """The last 25 words of a translation, as sequential context."""
        words = (text or "").split()
        return " ".join(words[-25:]) if len(words) > 25 else (text or "")

    def _previous_context(i):
        """previous_translation_context for segment i (sequential mode only).

        A retried segment sits inside the restored prefix, where the running
        chain describes a different position; its own predecessor, when that
        one is translated, is the right context.
        """
        if not sequential:
            return ""
        if i < len(prefix):
            if i > 0 and statuses[i - 1] == CHUNK_TRANSLATED:
                return _context_tail(translated_parts[i - 1])
            return ""
        return previous_translation_context

    def _contiguous_len():
        """Length of the gap-free translated prefix (the resume point)."""
        for j, part in enumerate(translated_parts):
            if part is None:
                return j
        return len(translated_parts)

    def _record_failure(i):
        statuses[i] = CHUNK_UNTRANSLATED
        if count_failures_as_fallback:
            stats.fallback_used += 1
        else:
            stats.failed_chunks += 1

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
            _previous_context(i),
        )
        return ('done', translated)

    async def _translate_one_paragraph(text, context_before, context_after):
        """Repair call for a single paragraph.

        previous_translation_context is empty on purpose: the repair replays
        paragraphs the sequential chain has already moved past, so feeding it
        the running context would describe the wrong position in the text.
        """
        return await _request(text, context_before, context_after, "")

    async def _translate_paragraph_range(text, context_before, context_after, options):
        """Repair call for a sub-range of a segment, with its count stated.

        Same empty previous_translation_context as _translate_one_paragraph,
        for the same reason.
        """
        return await _request(text, context_before, context_after, "", options)

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
        # Everything from next_index on is persisted as never attempted, which
        # is what validate() requires of a state (no text past the pointer).
        persisted_statuses = statuses[:next_index] + [CHUNK_PENDING] * (len(chunks) - next_index)
        try:
            checkpoint_hook(
                segments, list(translated_parts[:next_index]), next_index, stats.to_dict(),
                chunk_statuses=persisted_statuses,
            )
        except Exception as exc:  # noqa: BLE001 - checkpointing is best-effort
            if log_callback:
                log_callback(
                    "plain_text_checkpoint_failed",
                    f"⚠️ Plain-text checkpoint failed at segment {next_index}/{len(chunks)}: {exc}"
                )

    checkpoint_step = max(1, int(checkpoint_every))
    # The work set: every segment that fell back to its source text in an
    # earlier pass, plus every segment never attempted (issue #285).
    pending = unfinished_chunk_indices(statuses)
    rate_limit_error = None
    delivered = 0
    # Extra LLM calls spent by the mismatch repair, surfaced in the log so a
    # metered key can see the overhead while the run is going (issue #259).
    repair_calls_total = 0

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
                _record_failure(i)
            else:
                kind, value = result
                if kind == 'empty':
                    translated_parts[i] = value
                    statuses[i] = CHUNK_TRANSLATED
                    stats.successful_first_try += 1
                elif value is None:
                    if log_callback:
                        log_callback(
                            "plain_text_chunk_failed",
                            f"Chunk {i + 1}/{len(chunks)} failed - keeping original text"
                        )
                    translated_parts[i] = main_content
                    _record_failure(i)
                else:
                    cleaned = clean_translated_text(value)
                    cleaned = strip_hallucinated_markup(
                        cleaned, chunks[i].get('main_content', ''))
                    cleaned = strip_hallucinated_markdown_markers(
                        cleaned, chunks[i].get('main_content', ''))
                    translated_parts[i] = cleaned
                    statuses[i] = CHUNK_TRANSLATED
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
                                            _previous_context(i),
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
                                            "locating the divergence to repair this segment"
                                        )
                                    # Repaired here, inside the ordered consumer
                                    # loop and before the checkpoint hook below,
                                    # so the persisted prefix always holds the
                                    # repaired text. Never inside
                                    # _translate_chunk: that runs under the
                                    # worker semaphore and would deadlock.
                                    repaired, repair_failed, repair_calls = await _repair_segment_by_bisection(
                                        segments[i],
                                        source,
                                        translate_range=_translate_paragraph_range,
                                        translate_one=_translate_one_paragraph,
                                        prompt_options=prompt_options,
                                        log_callback=log_callback,
                                        segment_index=i,
                                        context_before=chunks[i].get('context_before', ''),
                                        context_after=chunks[i].get('context_after', ''),
                                    )
                                    repair_calls_total += repair_calls
                                    if log_callback:
                                        log_callback(
                                            "plain_text_paragraph_repair_done",
                                            f"Chunk {i + 1}/{len(chunks)} (segment {i}): repaired "
                                            f"{expected} paragraph(s) in {repair_calls} extra "
                                            f"call(s) ({repair_calls_total} repair call(s) so far "
                                            "in this file)"
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
                                                "the repair still returned the wrong "
                                                "count - keeping the original translation"
                                            )
                    # A retried segment sits inside the restored prefix and
                    # must not move the running chain the new segments use.
                    if sequential and i >= len(prefix):
                        previous_translation_context = _context_tail(cleaned)

            stats.record_processed()
            if stats_callback:
                stats_callback(stats.to_dict())
            delivered += 1

            # === CONTIGUITY INVARIANT ===
            # iter_ordered_concurrent yields indices strictly in ascending order
            # and every branch above assigns translated_parts[i] (success, empty,
            # None result, or exception -> source fallback). The restored prefix
            # is pre-filled, so a retried segment below it never opens a gap,
            # and once a new index i has been handled, slots 0..i are all
            # non-None. _contiguous_len() is therefore always a gap-free resume
            # point, and the prefix handed to the checkpoint is never sparse.
            # It is NOT i + 1 for a retried segment: that would cut the
            # restored prefix short and throw the work behind it away.
            if i < len(prefix):
                # Retries are rare and each one repairs damage: persist it now.
                _run_checkpoint_hook(_contiguous_len())
            else:
                next_index = i + 1
                if next_index % checkpoint_step == 0 or next_index == len(chunks):
                    _run_checkpoint_hook(next_index)

    if rate_limit_error is not None:
        # Persist the contiguous prefix translated before the limit, then keep
        # source text for everything else and propagate: the caller's auto-pause
        # depends on the exception reaching it.
        _run_checkpoint_hook(_contiguous_len())
        _fill_remaining_with_source()
        safe_parts = [p if p is not None else "" for p in translated_parts]
        rate_limit_error.partial_result = _reassemble(segments, safe_parts, source)
        raise rate_limit_error

    # Interruption: the scheduler stopped launching new chunks; keep source text
    # for the uncommitted tail and report the interruption.
    if delivered < len(pending) and check_interruption_callback and check_interruption_callback():
        resume_point = _contiguous_len()
        if log_callback:
            log_callback(
                "plain_text_translation_interrupted",
                f"⏸️ Plain-text translation interrupted at chunk "
                f"{min(resume_point + 1, len(chunks))}/{len(chunks)}"
            )
        _run_checkpoint_hook(resume_point)
        _fill_remaining_with_source()
        safe_parts = [p if p is not None else "" for p in translated_parts]
        return _reassemble(segments, safe_parts, source), stats, True

    # Any None left (shouldn't happen) falls back to empty string.
    safe_parts = [p if p is not None else "" for p in translated_parts]
    return _reassemble(segments, safe_parts, source), stats, False
