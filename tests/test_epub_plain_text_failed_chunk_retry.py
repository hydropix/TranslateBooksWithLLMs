"""Retrying Plain Text Mode segments that failed in an EPUB (issue #285).

Plain Text Mode used to replace a failed segment with its source text and save
the file as finished: nothing recorded which segment it was, so the job ended
`partial` with "N failed", and Resume re-entered no file, retried nothing and
reported the same failures again, forever. The resume also double-counted the
restored files and pre-counted the chunks on the already translated copies.

These tests pin the fix end to end on the 3-chapter EPUB of issue #261, in
Plain Text Mode, with no network (`generate_translation_request` is stubbed in
the plain-text pipeline module):

1. Pass 1 starves chapter 2: the segment is recorded as untranslated (a
   retryable fallback, not a terminal failure) and chapter 2 keeps its partial
   state, source body included.
2. Pass 2 retries exactly that segment from the source body, even though the
   copy on disk is now the translated rebuild, and the job comes back clean.
3. Pass 2 still starved retries once, stays partial and keeps its ticket.
4. Bilingual output, where re-extracting the rebuilt copy would find twice the
   paragraphs, retries the segment without duplicating the source twins.

A last family pins the pipeline contract directly: a retried segment below the
restored prefix must never shorten the persisted prefix.
"""

import asyncio
import re
from unittest.mock import MagicMock

import pytest

import src.core.common.plain_text_pipeline as plain_pipeline
import src.core.epub.translator as epub_translator
from src.api.completion_status import classify_completion
from src.core.epub.xhtml_translation_state import (
    CHUNK_PENDING,
    CHUNK_TRANSLATED,
    CHUNK_UNTRANSLATED,
)
from src.core.llm.exceptions import RateLimitError
from src.persistence.checkpoint_manager import CheckpointManager
from tests.test_epub_retry_failed_chunks import (
    MODEL,
    SENTINEL,
    TRANSLATED_MARKER,
    _build_epub,
    _chapter_text,
)


MAX_TOKENS_PER_CHUNK = 2000


@pytest.fixture
def plain_job(tmp_path, monkeypatch):
    """A registered EPUB job with an isolated checkpoint store."""
    input_path = tmp_path / "seventh_lamp.epub"
    output_path = tmp_path / "seventh_lamp_fr.epub"
    _build_epub(input_path)

    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)

    translation_id = "retry285"
    manager.start_job(
        translation_id=translation_id,
        file_type="epub",
        config={
            'file_path': str(input_path),
            'output_filename': output_path.name,
            'source_language': "English",
            'target_language': "French",
            'model': MODEL,
            'llm_provider': "ollama",
            'file_type': "epub",
        },
        input_file_path=str(input_path),
    )

    monkeypatch.setattr(epub_translator, "_create_llm_client",
                        lambda **kwargs: MagicMock())
    monkeypatch.setattr(epub_translator, "EPUB_TRANSLATE_METADATA_ENABLED", False)

    return {
        'input': input_path,
        'output': output_path,
        'manager': manager,
        'translation_id': translation_id,
    }


async def _run_pass(job, monkeypatch, resume_from_index, starve, bilingual=False,
                    max_tokens=MAX_TOKENS_PER_CHUNK, check_interruption=None):
    """Run one Plain Text Mode pass; return (stats, requests, logs)."""
    requests = []
    logs = []

    async def fake_generate_translation_request(main_content, *args, **kwargs):
        requests.append(main_content)
        if starve and SENTINEL in main_content:
            return None
        return "\n\n".join("%s %s" % (TRANSLATED_MARKER, p)
                           for p in main_content.split("\n\n"))

    monkeypatch.setattr(plain_pipeline, "generate_translation_request",
                        fake_generate_translation_request)

    stats = {}

    await epub_translator.translate_epub_file(
        input_filepath=str(job['input']),
        output_filepath=str(job['output']),
        source_language="English",
        target_language="French",
        model_name=MODEL,
        llm_provider="ollama",
        checkpoint_manager=job['manager'],
        translation_id=job['translation_id'],
        resume_from_index=resume_from_index,
        log_callback=lambda kind, message, **kwargs: logs.append((kind, message)),
        stats_callback=stats.update,
        check_interruption_callback=check_interruption,
        max_tokens_per_chunk=max_tokens,
        max_attempts=1,
        prompt_options={'plain_text_mode': True, 'bilingual': bilingual},
    )
    return stats, requests, logs


def _progress(job):
    return job['manager'].get_job(job['translation_id'])['progress']


def _resume_index(job):
    return job['manager'].load_checkpoint(job['translation_id'])['resume_from_index']


def _files_translated(logs):
    """The file count of the final 'EPUB translation complete' log line."""
    for kind, message in logs:
        if kind == "epub_save_success":
            return int(re.search(r"(\d+) files translated", message).group(1))
    raise AssertionError("no completion log line")


async def _starved_first_pass(job, monkeypatch, bilingual=False,
                              max_tokens=MAX_TOKENS_PER_CHUNK):
    stats, requests, logs = await _run_pass(
        job, monkeypatch, resume_from_index=0, starve=True, bilingual=bilingual,
        max_tokens=max_tokens)

    verdict = classify_completion(stats, str(job['output']))
    assert verdict.status == 'partial'
    # A retryable fallback, not a terminal failure: a failed_chunks count
    # survives every resume and can never clear the verdict.
    assert verdict.failed_chunks == 0
    assert verdict.unfinished_chunks == 1
    assert stats['fallback_used'] == 1

    job['manager'].mark_partial(job['translation_id'])
    return stats, requests, logs


@pytest.mark.asyncio
async def test_failed_segment_is_recorded_for_retry(plain_job, monkeypatch):
    await _starved_first_pass(plain_job, monkeypatch)
    manager, tid = plain_job['manager'], plain_job['translation_id']

    assert _progress(plain_job)['epub_unfinished_units'] == {'chapter2.xhtml': [0]}
    assert manager.list_xhtml_partial_states(tid) == ['chapter2.xhtml']

    state = manager.load_xhtml_partial_state(tid, 'chapter2.xhtml')
    assert state.chunk_statuses == [CHUNK_UNTRANSLATED]
    assert state.current_chunk_index == 1
    # The source body travels with the state: the copy on disk is about to be
    # replaced by the translated rebuild.
    assert SENTINEL in state.original_body_html
    assert TRANSLATED_MARKER not in state.original_body_html


@pytest.mark.asyncio
async def test_resume_retries_only_the_failed_segment(plain_job, monkeypatch):
    await _starved_first_pass(plain_job, monkeypatch)
    manager, tid = plain_job['manager'], plain_job['translation_id']

    stats, requests, logs = await _run_pass(
        plain_job, monkeypatch, resume_from_index=_resume_index(plain_job), starve=False)

    # Exactly one request, for the failed segment, sent as SOURCE text: the
    # translated copy on disk was not what got re-extracted.
    assert len(requests) == 1
    assert SENTINEL in requests[0]
    assert TRANSLATED_MARKER not in requests[0]
    assert any(kind == 'epub_retry_file' for kind, _ in logs)

    chapter2 = _chapter_text(plain_job['output'], "chapter2.xhtml")
    assert "%s %s" % (TRANSLATED_MARKER, SENTINEL) in chapter2
    assert "plain-text-untranslated" not in chapter2
    # Structure survives the round-trip through the stored source body.
    assert "<h1" in chapter2

    verdict = classify_completion(stats, str(plain_job['output']))
    assert verdict.status == 'completed'
    assert _progress(plain_job)['epub_unfinished_units'] == {}
    assert manager.list_xhtml_partial_states(tid) == []

    # Counters: the book has 3 files and 3 chunks, however many passes ran.
    assert _files_translated(logs) == 3
    assert stats['total_chunks'] == 3

    for href in ("chapter1.xhtml", "chapter3.xhtml"):
        assert TRANSLATED_MARKER in _chapter_text(plain_job['output'], href)


@pytest.mark.asyncio
async def test_retry_that_fails_again_keeps_the_ticket(plain_job, monkeypatch):
    await _starved_first_pass(plain_job, monkeypatch)
    manager, tid = plain_job['manager'], plain_job['translation_id']

    stats, requests, _logs = await _run_pass(
        plain_job, monkeypatch, resume_from_index=_resume_index(plain_job), starve=True)

    assert len(requests) == 1
    verdict = classify_completion(stats, str(plain_job['output']))
    assert verdict.status == 'partial'
    assert verdict.failed_chunks == 0
    assert _progress(plain_job)['epub_unfinished_units'] == {'chapter2.xhtml': [0]}
    state = manager.load_xhtml_partial_state(tid, 'chapter2.xhtml')
    assert state.chunk_statuses == [CHUNK_UNTRANSLATED]
    assert SENTINEL in state.original_body_html


@pytest.mark.asyncio
async def test_bilingual_retry_does_not_duplicate_the_source(plain_job, monkeypatch):
    await _starved_first_pass(plain_job, monkeypatch, bilingual=True)

    stats, requests, _logs = await _run_pass(
        plain_job, monkeypatch, resume_from_index=_resume_index(plain_job),
        starve=False, bilingual=True)

    assert len(requests) == 1
    assert TRANSLATED_MARKER not in requests[0]
    chapter2 = _chapter_text(plain_job['output'], "chapter2.xhtml")
    # One source twin per source block (h1 + 2 paragraphs), not six.
    assert chapter2.count("plain-text-source") == 3
    assert chapter2.count("plain-text-target") == 3
    assert classify_completion(stats, str(plain_job['output'])).status == 'completed'


@pytest.mark.asyncio
async def test_interrupted_retry_keeps_the_translated_chapter(plain_job, monkeypatch):
    """Pausing a retry pass must not ship the swapped-in source body.

    The adapter replaces the translated copy on disk with the stored source
    body before retrying. An interruption returns that document to the EPUB
    translator, which writes it into the partial output: without putting the
    translated copy back, the whole chapter would come out in the source
    language, including the segments pass 1 had translated.
    """
    import inspect

    # Small budget: chapter 2 splits into several segments, only one of which
    # (the sentinel paragraph) fails in pass 1.
    tokens = 25
    await _starved_first_pass(plain_job, monkeypatch, max_tokens=tokens)

    paused = []

    def interrupt_inside_the_pipeline():
        # The pause lands once the pipeline's scheduler asks, so the file is
        # re-entered and then stopped before its retry is launched. Like a real
        # pause, it stays on from then on.
        if not paused and any(frame.function == 'iter_ordered_concurrent'
                              for frame in inspect.stack()[1:4]):
            paused.append(True)
        return bool(paused)

    _stats, requests, logs = await _run_pass(
        plain_job, monkeypatch, resume_from_index=_resume_index(plain_job),
        starve=False, max_tokens=tokens,
        check_interruption=interrupt_inside_the_pipeline)

    assert requests == []
    assert any(kind == 'plain_text_translation_interrupted' for kind, _ in logs)
    outputs = [p for p in plain_job['output'].parent.glob("*.epub")
               if p != plain_job['input']]
    assert outputs
    for output in outputs:
        assert TRANSLATED_MARKER in _chapter_text(output, "chapter2.xhtml")

    # The debt is intact: the next pass retries it and finishes the book.
    stats, requests, _logs = await _run_pass(
        plain_job, monkeypatch, resume_from_index=_resume_index(plain_job),
        starve=False, max_tokens=tokens)
    assert len(requests) == 1 and SENTINEL in requests[0]
    assert classify_completion(stats, str(plain_job['output'])).status == 'completed'


# ---------------------------------------------------------------------------
# Pipeline contract: a retry below the restored prefix
# ---------------------------------------------------------------------------

PARAGRAPHS = [
    f"Paragraph number {i} with enough words to be its own chunk."
    for i in range(6)
]
PIPELINE_TOKENS = 20


class _Hook:
    def __init__(self):
        self.calls = []

    def __call__(self, segments, prefix, next_index, stats_dict, chunk_statuses=None):
        self.calls.append((list(prefix), next_index, list(chunk_statuses or [])))


async def _pipeline(fake, **overrides):
    kwargs = dict(
        paragraphs=PARAGRAPHS,
        source_language="English",
        target_language="French",
        model_name="m",
        llm_client=object(),
        max_tokens_per_chunk=PIPELINE_TOKENS,
        parallel_workers=1,
    )
    kwargs.update(overrides)
    return await plain_pipeline.translate_paragraphs_plain(**kwargs)


@pytest.fixture
def identity_cleanup(monkeypatch):
    monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s, **kw: s)


def _llm(monkeypatch, seen, fail_on=(), rate_limit_on=()):
    async def fake(*, main_content, **kwargs):
        await asyncio.sleep(0)
        seen.append(main_content)
        if main_content in rate_limit_on:
            raise RateLimitError("429", retry_after=1, provider="test")
        if main_content in fail_on:
            return None
        return f"T::{main_content}"
    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake)


@pytest.mark.asyncio
async def test_failures_count_as_failed_by_default_and_as_fallback_on_request(
        monkeypatch, identity_cleanup):
    """DOCX keeps the historical counter; EPUB opts into the fallback one."""
    _llm(monkeypatch, [], fail_on={PARAGRAPHS[2]})
    _, stats, _ = await _pipeline(None)
    assert (stats.failed_chunks, stats.fallback_used) == (1, 0)

    _llm(monkeypatch, [], fail_on={PARAGRAPHS[2]})
    _, stats, _ = await _pipeline(None, count_failures_as_fallback=True)
    assert (stats.failed_chunks, stats.fallback_used) == (0, 1)


@pytest.mark.asyncio
async def test_resume_retries_untranslated_segments_and_keeps_the_prefix(
        monkeypatch, identity_cleanup):
    seen = []
    _llm(monkeypatch, seen, fail_on={PARAGRAPHS[1]})
    hook = _Hook()
    await _pipeline(None, checkpoint_hook=hook)
    prefix, next_index, statuses = hook.calls[-1]
    assert next_index == len(PARAGRAPHS)
    assert statuses[1] == CHUNK_UNTRANSLATED
    assert statuses.count(CHUNK_TRANSLATED) == len(PARAGRAPHS) - 1

    seen.clear()
    _llm(monkeypatch, seen)
    hook2 = _Hook()
    segments = plain_pipeline.build_plain_segments(PARAGRAPHS, PIPELINE_TOKENS)
    out, _, interrupted = await _pipeline(
        None,
        resume_segments=segments,
        resume_translated=prefix,
        resume_statuses=statuses,
        checkpoint_hook=hook2,
    )

    assert not interrupted
    assert seen == [PARAGRAPHS[1]]
    assert out == [f"T::{p}" for p in PARAGRAPHS]
    # The retry persisted the WHOLE prefix, not the two segments up to it.
    last_prefix, last_index, last_statuses = hook2.calls[-1]
    assert last_index == len(PARAGRAPHS)
    assert len(last_prefix) == len(PARAGRAPHS)
    assert last_statuses == [CHUNK_TRANSLATED] * len(PARAGRAPHS)


@pytest.mark.asyncio
async def test_rate_limit_during_a_retry_keeps_the_prefix_and_the_debt(
        monkeypatch, identity_cleanup):
    """A limit hit while retrying segment 1 must not drop segments 2..5."""
    segments = plain_pipeline.build_plain_segments(PARAGRAPHS, PIPELINE_TOKENS)
    prefix = [f"T::{p}" for p in PARAGRAPHS]
    prefix[1] = PARAGRAPHS[1]
    prefix[3] = PARAGRAPHS[3]
    statuses = [CHUNK_TRANSLATED] * len(PARAGRAPHS)
    statuses[1] = statuses[3] = CHUNK_UNTRANSLATED

    _llm(monkeypatch, [], rate_limit_on={PARAGRAPHS[3]})
    hook = _Hook()
    with pytest.raises(RateLimitError):
        await _pipeline(
            None,
            resume_segments=segments,
            resume_translated=prefix,
            resume_statuses=statuses,
            checkpoint_hook=hook,
        )

    last_prefix, last_index, last_statuses = hook.calls[-1]
    assert last_index == len(PARAGRAPHS)
    assert last_prefix[1] == f"T::{PARAGRAPHS[1]}"
    assert last_prefix[4] == prefix[4]
    # Segment 1 was repaired, segment 3 is still owed.
    assert last_statuses[1] == CHUNK_TRANSLATED
    assert last_statuses[3] == CHUNK_UNTRANSLATED
    assert CHUNK_PENDING not in last_statuses


@pytest.mark.asyncio
async def test_parallel_resume_retries_and_translates_the_tail(monkeypatch, identity_cleanup):
    """Retries below the prefix and new segments above it share one pool."""
    segments = plain_pipeline.build_plain_segments(PARAGRAPHS, PIPELINE_TOKENS)
    prefix = [f"T::{p}" for p in PARAGRAPHS[:4]]
    prefix[0] = PARAGRAPHS[0]
    prefix[2] = PARAGRAPHS[2]
    statuses = [CHUNK_TRANSLATED] * 4 + [CHUNK_PENDING] * (len(PARAGRAPHS) - 4)
    statuses[0] = statuses[2] = CHUNK_UNTRANSLATED

    seen = []
    _llm(monkeypatch, seen)
    hook = _Hook()
    out, _, interrupted = await _pipeline(
        None,
        parallel_workers=3,
        resume_segments=segments,
        resume_translated=prefix,
        resume_statuses=statuses,
        checkpoint_hook=hook,
    )

    assert not interrupted
    assert sorted(seen) == sorted([PARAGRAPHS[0], PARAGRAPHS[2], PARAGRAPHS[4], PARAGRAPHS[5]])
    assert out == [f"T::{p}" for p in PARAGRAPHS]
    last_prefix, last_index, last_statuses = hook.calls[-1]
    assert last_index == len(PARAGRAPHS)
    assert last_statuses == [CHUNK_TRANSLATED] * len(PARAGRAPHS)
    # No persisted prefix ever shrank below the restored one.
    assert all(index >= 4 for _, index, _ in hook.calls)
