"""Unit tests for per-chunk translation outcomes (issue #261, plan Phase 2).

A job that ends with one chunk in the Phase 3 fallback used to leave no trace of
*which* chunk was left in the source language, so it could never be retried.
The pipeline now records a terminal status per chunk in
`TranslationMetrics.chunk_outcomes`, and the XHTML partial state persists it as
`chunk_statuses`.

These tests pin the four terminal branches and the persistence, with no network:
`generate_translation_request` is monkeypatched in the xhtml_translator module.
"""
import pytest

import src.core.epub.xhtml_translator as xt
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.xhtml_translation_state import (
    CHUNK_PENDING,
    CHUNK_TOKEN_ALIGNED,
    CHUNK_TRANSLATED,
    CHUNK_UNTRANSLATED,
    unfinished_chunk_indices,
)
from src.persistence.checkpoint_manager import CheckpointManager


PLACEHOLDER_TUPLE = ('[id', ']')

# A plain-prose chunk: no placeholder, so Phase 1 validation cannot fail.
PLAIN_CHUNK = {
    'text': 'Bonjour le monde.',
    'local_tag_map': {},
    'global_indices': [],
}

# A marked-up chunk: Phase 1 validation fails whenever the LLM drops the tags.
TAGGED_CHUNK = {
    'text': '[id0]Bonjour le monde.[id1]',
    'local_tag_map': {'[id0]': '<p>', '[id1]': '</p>'},
    'global_indices': [4, 5],
}


def _stub_generate(monkeypatch, answer):
    """Replace the LLM request with a callable returning `answer(text)`."""
    calls = []

    async def fake_generate_translation_request(main_content, *args, **kwargs):
        calls.append(main_content)
        return answer(main_content)

    monkeypatch.setattr(xt, "generate_translation_request",
                        fake_generate_translation_request)
    return calls


async def _translate(chunk, stats, chunk_index=0, max_retries=1):
    return await xt.translate_chunk_with_fallback(
        chunk_text=chunk['text'],
        local_tag_map=chunk['local_tag_map'],
        global_indices=chunk['global_indices'],
        source_language="French",
        target_language="English",
        model_name="test-model",
        llm_client=object(),
        stats=stats,
        max_retries=max_retries,
        placeholder_format=PLACEHOLDER_TUPLE,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# 1. Phase 1 success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_try_success_records_translated(monkeypatch):
    _stub_generate(monkeypatch, lambda text: "Hello world.")
    stats = TranslationMetrics()

    result = await _translate(PLAIN_CHUNK, stats, chunk_index=3)

    assert result == "Hello world."
    assert stats.successful_first_try == 1
    assert stats.chunk_outcomes == {3: CHUNK_TRANSLATED}
    assert unfinished_chunk_indices([CHUNK_TRANSLATED]) == []


@pytest.mark.asyncio
async def test_success_after_retry_records_translated(monkeypatch):
    """Second attempt succeeds: still a translated chunk, nothing to retry."""
    attempts = {'n': 0}

    def answer(text):
        attempts['n'] += 1
        return None if attempts['n'] == 1 else "Hello world."

    _stub_generate(monkeypatch, answer)
    stats = TranslationMetrics()

    result = await _translate(PLAIN_CHUNK, stats, chunk_index=0, max_retries=2)

    assert result == "Hello world."
    assert stats.successful_after_retry == 1
    assert stats.chunk_outcomes == {0: CHUNK_TRANSLATED}


# ---------------------------------------------------------------------------
# 2. Phase 3 fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exhausted_phases_record_untranslated(monkeypatch):
    """The LLM never answers: Phase 1 and Phase 2 both fail, Phase 3 kicks in."""
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", True)
    _stub_generate(monkeypatch, lambda text: None)
    stats = TranslationMetrics()

    result = await _translate(TAGGED_CHUNK, stats, chunk_index=1)

    # The source text comes back with its global indices restored.
    assert result == '[id4]Bonjour le monde.[id5]'
    assert stats.fallback_used == 1
    assert stats.chunk_outcomes == {1: CHUNK_UNTRANSLATED}
    assert unfinished_chunk_indices([CHUNK_TRANSLATED, CHUNK_UNTRANSLATED]) == [1]


# ---------------------------------------------------------------------------
# 3. Phase 2 token alignment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_alignment_records_token_aligned_and_is_not_unfinished(monkeypatch):
    """Phase 1 loses the placeholders, Phase 2 puts them back.

    Design decision D3: such a chunk IS translated (only the placeholder
    positions were approximated), so it must never be listed as unfinished.
    """
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", True)
    # The answer never carries the placeholders, so Phase 1 validation fails and
    # Phase 2's clean translation succeeds.
    _stub_generate(monkeypatch, lambda text: "Hello world.")
    stats = TranslationMetrics()

    result = await _translate(TAGGED_CHUNK, stats, chunk_index=2)

    assert stats.token_alignment_used == 1
    assert stats.token_alignment_success == 1
    assert stats.fallback_used == 0
    assert stats.chunk_outcomes == {2: CHUNK_TOKEN_ALIGNED}
    # Both placeholders were reinserted with their global indices.
    assert '[id4]' in result and '[id5]' in result

    statuses = [CHUNK_TRANSLATED, CHUNK_TRANSLATED, CHUNK_TOKEN_ALIGNED]
    assert unfinished_chunk_indices(statuses) == []


# ---------------------------------------------------------------------------
# 4. record_chunk_outcome contract
# ---------------------------------------------------------------------------

def test_record_chunk_outcome_is_a_no_op_without_index():
    stats = TranslationMetrics()
    stats.record_chunk_outcome(None, CHUNK_UNTRANSLATED)
    assert stats.chunk_outcomes == {}


def test_record_chunk_outcome_overwrites_previous_status():
    """A retry supersedes the earlier outcome of the same chunk."""
    stats = TranslationMetrics()
    stats.record_chunk_outcome(7, CHUNK_UNTRANSLATED)
    stats.record_chunk_outcome(7, CHUNK_TRANSLATED)
    assert stats.chunk_outcomes == {7: CHUNK_TRANSLATED}


def test_chunk_outcomes_stay_out_of_the_stats_payload():
    """to_dict()/from_dict() and merge() must ignore chunk_outcomes.

    The statuses live in XHTMLTranslationState.chunk_statuses; the stats dict is
    a UI/aggregation payload, and its indices would collide across files.
    """
    stats = TranslationMetrics()
    stats.record_chunk_outcome(0, CHUNK_UNTRANSLATED)

    payload = stats.to_dict()
    assert 'chunk_outcomes' not in payload
    assert TranslationMetrics.from_dict(payload).chunk_outcomes == {}

    other = TranslationMetrics()
    other.record_chunk_outcome(0, CHUNK_TRANSLATED)
    other.merge(stats)
    assert other.chunk_outcomes == {0: CHUNK_TRANSLATED}


# ---------------------------------------------------------------------------
# 5. The chunk loop: text-free pass-through and persistence
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_checkpoint_manager(tmp_path):
    """Checkpoint manager with isolated storage (same pattern as
    tests/unit/epub/test_text_free_chunk_passthrough.py)."""
    manager = CheckpointManager(db_path=str(tmp_path / "test_jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)
    return manager


async def _run_loop(chunks, checkpoint_manager=None, translation_id=None,
                    file_href=None):
    return await xt._translate_all_chunks_with_checkpoint(
        chunks=chunks,
        source_language="French",
        target_language="English",
        model_name="test-model",
        llm_client=object(),
        max_retries=1,
        context_manager=None,
        placeholder_format=PLACEHOLDER_TUPLE,
        checkpoint_manager=checkpoint_manager,
        translation_id=translation_id,
        file_href=file_href,
        file_path=file_href,
        parallel_workers=1,
    )


@pytest.mark.asyncio
async def test_text_free_chunk_records_translated(monkeypatch):
    """A chunk with nothing to translate never reaches the LLM, yet it counts as
    translated: it must not be retried on resume."""
    calls = _stub_generate(monkeypatch, lambda text: "unexpected")

    text_free = {'text': '[id0]', 'local_tag_map': {'[id0]': '<div><svg/></div>'},
                 'global_indices': [0]}
    _translated, stats, was_interrupted = await _run_loop([text_free])

    assert was_interrupted is False
    assert calls == []  # nothing was sent
    assert stats.chunk_outcomes == {0: CHUNK_TRANSLATED}


@pytest.mark.asyncio
async def test_saved_state_persists_chunk_statuses(monkeypatch, temp_checkpoint_manager):
    """The partial state remembers which chunk stayed untranslated."""
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", True)

    def answer(text):
        # Only the second chunk gets a translation; the first starves.
        return "Second chunk translated." if 'Deuxieme' in text else None

    _stub_generate(monkeypatch, answer)

    chunks = [
        {'text': 'Premier paragraphe.', 'local_tag_map': {}, 'global_indices': []},
        {'text': 'Deuxieme paragraphe.', 'local_tag_map': {}, 'global_indices': []},
    ]
    translation_id, file_href = "outcomes_job", "OEBPS/chapter1.xhtml"

    translated, stats, _ = await _run_loop(
        chunks, temp_checkpoint_manager, translation_id, file_href)

    assert len(translated) == 2
    assert stats.chunk_outcomes == {0: CHUNK_UNTRANSLATED, 1: CHUNK_TRANSLATED}

    state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
    assert state is not None
    assert state.validate() is True
    assert state.chunk_statuses == [CHUNK_UNTRANSLATED, CHUNK_TRANSLATED]
    assert unfinished_chunk_indices(state.chunk_statuses) == [0]


@pytest.mark.asyncio
async def test_interrupted_run_persists_pending_tail(monkeypatch, temp_checkpoint_manager):
    """Chunks never attempted stay CHUNK_PENDING in the persisted statuses."""
    calls = _stub_generate(monkeypatch, lambda text: "Translated.")

    chunks = [
        {'text': f'Paragraphe {i}.', 'local_tag_map': {}, 'global_indices': []}
        for i in range(4)
    ]
    translation_id, file_href = "outcomes_interrupt", "OEBPS/chapter2.xhtml"

    translated, _stats, was_interrupted = await xt._translate_all_chunks_with_checkpoint(
        chunks=chunks,
        source_language="French",
        target_language="English",
        model_name="test-model",
        llm_client=object(),
        max_retries=1,
        context_manager=None,
        placeholder_format=PLACEHOLDER_TUPLE,
        checkpoint_manager=temp_checkpoint_manager,
        translation_id=translation_id,
        file_href=file_href,
        file_path=file_href,
        check_interruption_callback=lambda: len(calls) >= 1,
        parallel_workers=1,
    )

    assert was_interrupted is True
    state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
    assert state is not None
    assert state.validate() is True
    assert len(state.chunk_statuses) == len(chunks)
    done = len(translated)
    assert state.chunk_statuses[:done] == [CHUNK_TRANSLATED] * done
    assert state.chunk_statuses[done:] == [CHUNK_PENDING] * (len(chunks) - done)
    assert unfinished_chunk_indices(state.chunk_statuses) == list(range(done, len(chunks)))


@pytest.mark.asyncio
async def test_interrupted_retry_pass_reports_interrupted(
    monkeypatch, temp_checkpoint_manager
):
    """A pause mid-repair must not look like a completed pass.

    Repair work is CHUNK_UNTRANSLATED below the resume pointer, so
    `_next_pending_index()` is already len(chunks). The caller uses
    was_interrupted to skip reconstruction and keep [partial] outputs.
    """
    calls = _stub_generate(monkeypatch, lambda text: "Retried.")

    chunks = [
        {'text': f'Paragraphe {i}.', 'local_tag_map': {}, 'global_indices': []}
        for i in range(3)
    ]
    translation_id, file_href = "outcomes_retry_interrupt", "OEBPS/chapter3.xhtml"

    translated, _stats, was_interrupted = await xt._translate_all_chunks_with_checkpoint(
        chunks=chunks,
        source_language="French",
        target_language="English",
        model_name="test-model",
        llm_client=object(),
        max_retries=1,
        context_manager=None,
        placeholder_format=PLACEHOLDER_TUPLE,
        checkpoint_manager=temp_checkpoint_manager,
        translation_id=translation_id,
        file_href=file_href,
        file_path=file_href,
        start_chunk_index=len(chunks),
        translated_chunks=["Source 0.", "Source 1.", "Source 2."],
        chunk_statuses=[CHUNK_UNTRANSLATED, CHUNK_UNTRANSLATED, CHUNK_UNTRANSLATED],
        check_interruption_callback=lambda: len(calls) >= 1,
        parallel_workers=1,
    )

    assert was_interrupted is True
    assert len(calls) == 1
    assert translated == ["Retried.", "Source 1.", "Source 2."]
    state = temp_checkpoint_manager.load_xhtml_partial_state(translation_id, file_href)
    assert state is not None
    assert state.validate() is True
    assert state.chunk_statuses[0] == CHUNK_TRANSLATED
    assert state.chunk_statuses[1:] == [CHUNK_UNTRANSLATED, CHUNK_UNTRANSLATED]
