"""
One request/validate core for every chunk sent to an LLM.

``request_chunk`` performs "send prompt, validate, extract" for one chunk and
owns every response guard exactly once:

    * provider failure (no response) vs. an empty response;
    * adaptive context growth through ``AdaptiveContextManager`` on a
      truncated window, an unclosed translation tag, a repetition loop or a
      context overflow (until the manager refuses);
    * ``output_truncated`` (the provider says the output limit was hit);
    * extraction through the provider's ``TranslationExtractor``;
    * the plain-text raw fallback rules: no fallback for structured content,
      the source echoed back, a refusal or a reply whose length is far off.

It returns a typed ``ChunkOutcome`` instead of ``None`` / ``""`` conventions.
Callers keep their own policy for a failed chunk (retry with another prompt,
keep the draft, mark the chunk untranslated, split it on overflow).
"""

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT

from .llm import ContextOverflowError, RepetitionLoopError
from .llm.base import LLMResponse


# --- Raw-response fallback guards (plain-text mode) ---------------------------
#
# When the model drops the translation tags, plain-text mode falls back to the
# raw reply. That reply can be a preamble, the model's reasoning or a refusal
# rather than a translation, and saving it would silently replace the chunk.

# "Here is the translation:", "Sure! Here's the French translation:" ...
_RAW_PREAMBLE_RE = re.compile(
    r"\A\s*(?:here(?:'s|\s+is|\s+are)|below\s+is|sure|certainly|okay|ok)\b"
    r"[^\n:]{0,80}?\btranslat\w*[^\n:]{0,40}:[ \t]*\n?",
    re.IGNORECASE,
)
_REFUSAL_OPENER_RE = re.compile(
    r"\A\s*(?:I['’]?m\s+sorry|I\s+am\s+sorry|sorry\b|I\s+apologi[sz]e|"
    r"I\s+can(?:not|['’]t)\b|I['’]?m\s+(?:not\s+able|unable)|I\s+am\s+(?:not\s+able|unable)|"
    r"as\s+an\s+AI\b|unfortunately,?\s+I\b)",
    re.IGNORECASE,
)
_REFUSAL_TOPIC_RE = re.compile(
    r"translat|request|assist|help|content|comply|provide|policy|guidelines",
    re.IGNORECASE,
)
# Weighted length bounds for target/source. CJK and Hangul characters carry
# about two to three Latin characters of meaning, so they are weighted up to
# keep CJK<->Latin pairs inside the bounds.
RAW_FALLBACK_MIN_RATIO = 0.2
RAW_FALLBACK_MAX_RATIO = 4.0
RAW_FALLBACK_MIN_SOURCE_LENGTH = 40


def _weighted_length(text: str) -> float:
    """Length of `text` with CJK ideographs/kana weighted 3 and Hangul 2."""
    total = 0.0
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        if (0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF
                or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2FFFF):
            total += 3
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            total += 2
        else:
            total += 1
    return total


def strip_raw_preamble(raw: str) -> str:
    """Drop a leading "Here is the translation:" line from an untagged reply."""
    stripped = _RAW_PREAMBLE_RE.sub("", raw, count=1).strip()
    return stripped or raw.strip()


def raw_fallback_rejection_reason(raw: str, source: str) -> Optional[str]:
    """Why an untagged raw reply cannot be saved as a translation, or None.

    Rejects an empty reply, a refusal, and a reply whose length is wildly
    off the source's (reasoning dumps, one-line answers). Short sources skip
    the length test: their ratio says nothing.
    """
    text = (raw or "").strip()
    if not text:
        return "empty response"
    head = text[:300]
    if (_REFUSAL_OPENER_RE.search(head) and _REFUSAL_TOPIC_RE.search(head)
            and not _REFUSAL_OPENER_RE.search(source or "")):
        return "the response looks like a refusal"
    source_len = _weighted_length(source or "")
    if source_len >= RAW_FALLBACK_MIN_SOURCE_LENGTH:
        ratio = _weighted_length(text) / source_len
        if ratio < RAW_FALLBACK_MIN_RATIO or ratio > RAW_FALLBACK_MAX_RATIO:
            return f"its length is {ratio:.1f}x the source's"
    return None


# --- Tagged-answer acceptance (TXT / SRT / PGN) --------------------------------
#
# A well-tagged answer is not necessarily a translation: it can be a refusal
# wrapped in the tags, a cut-off fragment, a runaway commentary or the source
# copied back. ``tagged_answer_problem`` applies the same rules the raw
# fallback and the EPUB/DOCX survival guard apply, so callers that opt in
# (``request_chunk(answer_check=...)``) never save such an answer as done.

_MARKER_RE = re.compile(r"\[(?:id)?\d+\]")
_SOUND_TAG_RE = re.compile(r"\[[^\[\]\n]*\]|\([^()\n]*\)|[♪♫♩♬]")
_SAN_TOKEN_RE = re.compile(
    r"^(?:\d+\.(?:\.\.)?|O-O(?:-O)?|0-0(?:-0)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|"
    r"[KQRBN][a-h1-8]?x?[a-h][1-8]|1-0|0-1|1/2-1/2|\*)[+#]?[!?]{0,2}$"
)
_WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*", re.UNICODE)
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
ECHO_SINGLE_TOKEN_MAX_LENGTH = 24
"""A one-word line up to this long may legitimately come back unchanged."""
ECHO_NAME_MAX_WORDS = 3
"""A line of at most this many capitalized words is treated as a proper name."""
REFUSAL_MAX_SHARE_OF_SOURCE = 0.6
"""A refusal-shaped answer is rejected only when it is this much shorter than
the source: a short dialogue line that happens to read like a refusal ("I'm
sorry, I can't help you.") is a translation of a line of the same length."""


def _normalize_for_echo(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def _line_may_stay_unchanged(line: str) -> bool:
    """True for a line a translator may legitimately return as is."""
    text = _MARKER_RE.sub(" ", line)
    if not _LETTER_RE.search(text):
        return True  # numbers, punctuation, music notes only
    if not _LETTER_RE.search(_SOUND_TAG_RE.sub(" ", text)):
        return True  # pure sound tags: "[MUSIC]", "(laughs)", "♪ ... ♪" without words
    tokens = text.split()
    if tokens and all(_SAN_TOKEN_RE.match(tok.strip(",;:()")) for tok in tokens
                      if tok.strip(",;:()")):
        return True  # chess move text
    words = _WORD_RE.findall(text)
    if len(words) == 1 and len(words[0]) <= ECHO_SINGLE_TOKEN_MAX_LENGTH:
        return True  # a single short token
    if 0 < len(words) <= ECHO_NAME_MAX_WORDS and all(w[:1].isupper() for w in words):
        return True  # a proper name ("John Smith", "Paris")
    return False


def source_may_stay_unchanged(source: str) -> bool:
    """True when every non-empty line of ``source`` may come back unchanged.

    Numbers, punctuation, music notes, pure sound tags ("[MUSIC]",
    "(laughs)"), chess move text, a single short token and a short proper
    name are correct untranslated in any target language, so an answer equal
    to such a source is not an echo.
    """
    lines = [ln for ln in (source or "").splitlines() if ln.strip()]
    return all(_line_may_stay_unchanged(ln) for ln in lines)


def languages_differ(source_language: Optional[str], target_language: Optional[str]) -> bool:
    """True when both languages are known and are not the same."""
    src = (source_language or "").strip().casefold()
    tgt = (target_language or "").strip().casefold()
    return bool(src and tgt and src not in ("auto", "auto-detect") and src != tgt)


def tagged_answer_problem(answer: str, source: str, *, reject_echo: bool) -> Optional[str]:
    """Why an extracted (tagged) answer cannot be saved as a translation, or None.

    Rules, in order:
      * a refusal-shaped answer far shorter than the source;
      * the text-survival bounds shared with EPUB/DOCX (too short, runaway);
      * with ``reject_echo`` (source and target languages differ), an answer
        identical to the source after whitespace and case normalisation,
        unless the source may legitimately stay unchanged
        (:func:`source_may_stay_unchanged`).
    """
    from src.core.epub.text_survival import text_survival_problem

    text = (answer or "").strip()
    head = _MARKER_RE.sub("", text[:300]).strip()
    if (_REFUSAL_OPENER_RE.search(head) and _REFUSAL_TOPIC_RE.search(head)
            and not _REFUSAL_OPENER_RE.search(_MARKER_RE.sub("", source or "").strip())
            and _weighted_length(text) < REFUSAL_MAX_SHARE_OF_SOURCE * _weighted_length(source or "")):
        return "the answer looks like a refusal"
    problem = text_survival_problem(source or "", text)
    if problem:
        return problem
    if (reject_echo and _normalize_for_echo(text) == _normalize_for_echo(source)
            and not source_may_stay_unchanged(source)):
        return "the answer is the source text copied back"
    return None


# --- Outcome -------------------------------------------------------------------

class ChunkStatus(str, Enum):
    """What happened to one chunk request."""
    OK = "ok"                        # Tagged answer extracted
    FALLBACK = "fallback"            # Untagged raw answer accepted (plain text)
    EMPTY = "empty"                  # Empty answer or empty tags
    TRUNCATED = "truncated"          # Output limit hit, or tag opened but never closed
    REFUSED = "refused"              # Untagged answer rejected (refusal, echo, length)
    UNEXTRACTABLE = "unextractable"  # No tags and the raw fallback is not allowed
    OVERFLOW = "overflow"            # Prompt too large, context cannot grow further
    REPETITION = "repetition"        # Repetition loop the context could not cure
    FAILED = "failed"                # The provider gave no response at all


@dataclass
class ChunkOutcome:
    """Result of ``request_chunk``.

    ``text`` is set only when ``ok``. ``response`` is the last provider
    response (None when the provider never answered). ``error`` holds the
    ``ContextOverflowError`` / ``RepetitionLoopError`` behind an OVERFLOW /
    REPETITION status.
    """
    status: ChunkStatus
    text: Optional[str] = None
    response: Optional[LLMResponse] = None
    detail: str = ""
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.status in (ChunkStatus.OK, ChunkStatus.FALLBACK)


@dataclass(frozen=True)
class PassLabels:
    """Log event names and wording of one kind of pass (translate / refine)."""
    response: Optional[str]
    empty: str
    truncated: str
    unclosed: str
    unclosed_retry: str
    extraction_failed: Optional[str]
    unextractable: str
    rejected: str
    fallback: Optional[str]
    echo: str
    repetition_retry: str
    repetition_fatal: str
    overflow_retry: Optional[str]
    on_failure: str  # What happens to the chunk, appended to failure messages
    noun: str        # "Model" / "Refinement"


TRANSLATION_LABELS = PassLabels(
    response="llm_response",
    empty="empty_llm_response",
    truncated="output_truncated",
    unclosed="unclosed_translation_tag",
    unclosed_retry="unclosed_tag_retry",
    extraction_failed="translation_extraction_failed",
    unextractable="epub_extraction_critical_fail",
    rejected="raw_response_fallback_rejected",
    fallback="using_raw_response_fallback",
    echo="llm_prompt_in_response_warning",
    repetition_retry="repetition_loop_retry",
    repetition_fatal="repetition_loop_fatal",
    overflow_retry=None,
    on_failure="Chunk marked failed.",
    noun="Model",
)

REFINEMENT_LABELS = PassLabels(
    response="refinement_response",
    empty="refinement_empty_response",
    truncated="refinement_output_truncated",
    unclosed="refinement_unclosed_tag",
    unclosed_retry="refinement_unclosed_tag_retry",
    extraction_failed=None,
    unextractable="refinement_extraction_failed",
    rejected="refinement_fallback_rejected",
    fallback=None,
    echo="refinement_warning",
    repetition_retry="refinement_repetition_retry",
    repetition_fatal="refinement_error",
    overflow_retry="refinement_overflow_retry",
    on_failure="Keeping the draft.",
    noun="Refinement",
)


def _log(log_callback, event: Optional[str], message: str, **kwargs) -> None:
    if log_callback and event:
        log_callback(event, message, **kwargs)


# --- The core ------------------------------------------------------------------

async def request_chunk(
    client: Any,
    prompt: Any,
    *,
    source_text: str,
    allow_raw_fallback: bool,
    context_manager: Any = None,
    model: Optional[str] = None,
    log_callback: Optional[Callable] = None,
    labels: PassLabels = TRANSLATION_LABELS,
    raw_response_callback: Optional[Callable[[LLMResponse], None]] = None,
    extract: bool = True,
    answer_check: Optional[Callable[[str, str], Optional[str]]] = None,
) -> ChunkOutcome:
    """Send one prompt pair, validate the answer and extract the text.

    Args:
        client: An ``LLMProvider`` (anything with ``generate`` and
            ``extract_translation``).
        prompt: A prompt pair with ``.system`` and ``.user``.
        source_text: The text the prompt asks the model to process (source
            chunk or draft): used to spot an echoed prompt and to bound the
            raw fallback's length.
        allow_raw_fallback: Accept an untagged answer that passes the raw
            fallback guards (plain text only; never for placeholders or
            structured output).
        context_manager: Optional ``AdaptiveContextManager``: the request is
            sent with its current context size and the context grows on the
            guards that call for it. Budget bookkeeping (begin / abandon) is
            the caller's, see ``ContextBudget``.
        model: Model for this request (the provider's own when None).
        log_callback: Optional ``(event, message, data=...)`` logger.
        labels: Event names and wording of the pass.
        raw_response_callback: Receives every response as it arrives, before
            any guard. Errors it raises are logged and swallowed.
        extract: False returns the raw answer as ``text`` (status OK) once
            the transport guards passed (no response, empty answer, context
            growth), for callers with their own answer format that validate
            each item themselves (term-fix snippets).
        answer_check: Optional ``(source_text, answer) -> reason | None``
            run on a tagged answer once it is extracted (see
            :func:`tagged_answer_problem`). A reason rejects the answer
            (status REFUSED), with a log line naming it.
    """
    last_response: Optional[LLMResponse] = None
    # One plain repeat of the request for an unclosed tag, once the context
    # cannot (or need not) grow: on cloud providers this cut-off is usually
    # transient and the same prompt goes through on the next call (#354).
    unclosed_retry_left = True
    while True:
        kwargs = {"system_prompt": prompt.system}
        if model:
            kwargs["model"] = model
        if context_manager is not None:
            new_ctx = context_manager.get_context_size()
            current = getattr(client, "context_window", None)
            if current is not None and current != new_ctx:
                _log(log_callback, "context_update", f"📐 Context window for this request: {new_ctx}")
            kwargs["context_window"] = new_ctx

        start_time = time.time()
        try:
            response = await client.generate(prompt.user, **kwargs)
        except RepetitionLoopError as e:
            # Typically a thinking model whose window is too small. Grow it
            # (two steps), a bounded number of times: a loop that survives
            # larger windows is the model looping on this chunk.
            if context_manager is not None and context_manager.should_retry_repetition_loop():
                old = context_manager.get_context_size()
                context_manager.increase_context()
                context_manager.increase_context()
                new = context_manager.get_context_size()
                if new > old:
                    _log(log_callback, labels.repetition_retry,
                         f"🔄 Repetition loop detected! Increasing context from {old} to {new} tokens")
                    continue
            _log(log_callback, labels.repetition_fatal,
                 f"⚠️ Repetition loop detected and cannot recover. "
                 f"Try increasing OLLAMA_NUM_CTX or lowering the chunk size. Error: {e}")
            return ChunkOutcome(ChunkStatus.REPETITION, response=last_response,
                                detail=str(e), error=e)
        except ContextOverflowError as e:
            if context_manager is not None and context_manager.should_retry_with_larger_context(True, 0):
                context_manager.increase_context()
                _log(log_callback, labels.overflow_retry,
                     f"⚠️ Context overflow! Retrying with context "
                     f"{context_manager.get_context_size()}")
                continue
            return ChunkOutcome(ChunkStatus.OVERFLOW, response=last_response,
                                detail=str(e), error=e)
        execution_time = time.time() - start_time

        if not response:
            return ChunkOutcome(ChunkStatus.FAILED, response=last_response,
                                detail="the provider returned no response")
        last_response = response

        if raw_response_callback:
            try:
                raw_response_callback(response)
            except Exception as record_error:
                _log(log_callback, "raw_response_record_failed",
                     f"Could not record raw LLM response: {record_error}")

        raw = response.content
        if not raw or not raw.strip():
            # Almost always a refusal or a provider-side moderation block, not
            # a truncation: a larger context would not help.
            _log(log_callback, labels.empty,
                 f"⚠️ {labels.noun} returned an empty response. This usually means the "
                 f"model refused or filtered this chunk (sensitive/policy-flagged content "
                 f"or provider-side moderation). {labels.on_failure}")
            return ChunkOutcome(ChunkStatus.EMPTY, response=response, detail="empty response")

        if (context_manager is not None and getattr(response, "was_truncated", False)
                and context_manager.should_retry_with_larger_context(
                    True, getattr(response, "context_used", 0))):
            context_manager.increase_context()
            continue

        if labels.response:
            _log(log_callback, labels.response, "LLM Response received", data={
                'type': labels.response,
                'response': raw,
                'execution_time': execution_time,
                'model': model or getattr(client, "model", None),
                'tokens': {
                    'prompt': getattr(response, "prompt_tokens", 0),
                    'completion': getattr(response, "completion_tokens", 0),
                    'total': getattr(response, "context_used", 0),
                    'limit': getattr(response, "context_limit", 0),
                },
            })

        if not extract:
            return ChunkOutcome(ChunkStatus.OK, text=raw, response=response)

        # The provider says the output limit was hit: what came back is only
        # the beginning of the chunk. Saving it would drop the rest silently.
        if getattr(response, "output_truncated", False) is True:
            _log(log_callback, labels.truncated,
                 f"⚠️ {labels.noun} hit the output token limit before finishing "
                 f"(finish_reason: {getattr(response, 'finish_reason', None)}). "
                 f"{labels.on_failure} "
                 f"If this keeps happening, lower the chunk size or disable reasoning "
                 f"for this model.")
            return ChunkOutcome(ChunkStatus.TRUNCATED, response=response,
                                detail="output token limit")

        text = client.extract_translation(raw)
        if text and text.strip():
            problem = answer_check(source_text, text) if answer_check else None
            if problem:
                _log(log_callback, labels.rejected,
                     f"⚠️ {labels.noun} answer rejected ({problem}). {labels.on_failure}")
                return ChunkOutcome(ChunkStatus.REFUSED, response=response, detail=problem)
            return ChunkOutcome(ChunkStatus.OK, text=text, response=response)

        stripped = raw.strip()
        if stripped.startswith(TRANSLATE_TAG_IN) and not stripped.endswith(TRANSLATE_TAG_OUT):
            # The model opened the tag and stopped: implicit truncation.
            if (context_manager is not None
                    and context_manager.should_retry_with_larger_context(
                        True, getattr(response, "context_used", 0))):
                _log(log_callback, "implicit_truncation_retry",
                     "🔄 Model stopped before closing tag. Retrying with larger context...")
                context_manager.increase_context()
                continue
            finish_reason = getattr(response, "finish_reason", None)
            if unclosed_retry_left:
                unclosed_retry_left = False
                _log(log_callback, labels.unclosed_retry,
                     f"🔄 Response opened the translation tag but never closed it "
                     f"(finish_reason: {finish_reason}). Retrying once...")
                continue
            _log(log_callback, labels.unclosed,
                 f"⚠️ Response opened the translation tag but never closed it "
                 f"(finish_reason: {finish_reason}): the output was cut off. "
                 f"{labels.on_failure}")
            return ChunkOutcome(ChunkStatus.TRUNCATED, response=response, detail="unclosed tag")

        if TRANSLATE_TAG_IN in stripped and TRANSLATE_TAG_OUT in stripped:
            _log(log_callback, labels.empty,
                 f"⚠️ {labels.noun} returned empty translation tags. {labels.on_failure}")
            return ChunkOutcome(ChunkStatus.EMPTY, response=response, detail="empty tags")

        if labels.extraction_failed:
            _log(log_callback, labels.extraction_failed,
                 "⚠️ WARNING: Failed to extract translation (tags not found or malformed)")
            _log(log_callback, f"{labels.extraction_failed}_preview",
                 f"Response preview (first 300 chars): {raw[:300]}")

        if not allow_raw_fallback:
            # Structured content (placeholders, numbered blocks): the raw
            # answer would corrupt the output.
            _log(log_callback, labels.unextractable,
                 f"CRITICAL: translation tags missing and the raw response cannot be used "
                 f"for structured content. {labels.on_failure}")
            return ChunkOutcome(ChunkStatus.UNEXTRACTABLE, response=response,
                                detail="tags missing")

        if source_text and source_text in raw:
            _log(log_callback, labels.echo,
                 f"WARNING: response contains the input. {labels.on_failure}")
            return ChunkOutcome(ChunkStatus.REFUSED, response=response, detail="input echoed")

        candidate = strip_raw_preamble(raw)
        rejection = raw_fallback_rejection_reason(candidate, source_text)
        if rejection:
            _log(log_callback, labels.rejected,
                 f"⚠️ Untagged response rejected ({rejection}). {labels.on_failure}")
            return ChunkOutcome(ChunkStatus.REFUSED, response=response, detail=rejection)

        _log(log_callback, labels.fallback,
             "⚠️ Translation tags missing: using the raw response as the "
             "translation (plain text mode)")
        return ChunkOutcome(ChunkStatus.FALLBACK, text=candidate, response=response,
                            detail="raw response")


class ContextBudget:
    """Per-request context growth budget of an ``AdaptiveContextManager``.

    ``with ContextBudget(cm) as budget: ...; budget.succeeded()`` starts a
    fresh budget, and gives back the context grown for a request that failed
    (or raised), so one bad chunk cannot inflate every later request.
    A ``None`` manager makes it a no-op.
    """

    def __init__(self, context_manager: Any):
        self.context_manager = context_manager
        self._ok = False

    def succeeded(self) -> None:
        self._ok = True

    def __enter__(self) -> "ContextBudget":
        if self.context_manager is not None:
            self.context_manager.begin_request()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.context_manager is not None and (exc_type is not None or not self._ok):
            self.context_manager.abandon_request()
        return False
