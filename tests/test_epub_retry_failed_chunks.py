"""Integration tests for retrying failed EPUB chunks on resume (issue #261).

A job that ends `partial` because one chunk fell back to its source text used to
be unfinishable: the resume pointer is a file index, so the file holding the bad
chunk counted as done and Resume translated nothing. These tests pin the fix end
to end, on a 3-chapter in-memory EPUB and with no network at all
(`generate_translation_request` is stubbed inside the xhtml_translator module).

Three scenarios, in order:

1. Pass 1 starves chapter 2's only chunk: the verdict is `partial`, the job
   progress lists the unfinished chunk under `epub_unfinished_units`, and the
   per-file partial state of chapter 2 - and only chapter 2 - survives (which
   also pins the state-key unification: chapters 1 and 3 are cleaned up).
2. Pass 2 with the starvation lifted retries exactly that one chunk, nothing
   else, and the job comes back clean: no ticket, no partial state, chapter 2
   translated in the output EPUB.
3. Pass 2 with the starvation still in place retries exactly once, stays
   `partial` and keeps its ticket - no false success, no retry loop.

A second family covers the token-aligned (Phase 2) chunks: they are translated
with approximate tag positions, so they never enter the automatic work set
(design decision D3) and never move the verdict away from `completed`. They are
retryable only through the explicit `retry_token_aligned` opt-in the completion
card sends.
"""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.core.epub.translator as epub_translator
import src.core.epub.xhtml_translator as xhtml_translator
from src.api.completion_status import classify_completion
from src.common.placeholder_format import PlaceholderFormat
from src.persistence.checkpoint_manager import CheckpointManager


# The sabotaged chunk is identified by this sentinel: the stub returns None for
# any request containing it, in Phase 1 (with placeholders) and in Phase 2
# (placeholders stripped), which is what pushes the chunk into Phase 3.
SENTINEL = "The keeper refused to name the seventh lamp"

# Marker the stub prepends to a successful "translation". A chapter carrying it
# has been through the LLM; a chapter without it is still source text.
TRANSLATED_MARKER = "[FR]"

MODEL = "test-model"
# Large enough that each short chapter is exactly one chunk.
MAX_TOKENS_PER_CHUNK = 2000

PARAGRAPHS = [
    "The lighthouse keeper's journal had grown thick with salt and ink.",
    "Every evening the lamp was wound and the log signed in the same hand.",
]


# ---------------------------------------------------------------------------
# Fixture: a 3-chapter EPUB whose chapter 2 carries the sentinel
# ---------------------------------------------------------------------------

def _chapter_xhtml(title, paragraphs):
    body = "\n".join("    <p>%s</p>" % p for p in paragraphs)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">\n'
        '<head><title>%s</title></head>\n'
        '<body>\n'
        '  <h1>%s</h1>\n'
        '%s\n'
        '</body>\n'
        '</html>\n' % (title, title, body)
    )


def _build_epub(path):
    """3 chapters; chapter 2 carries the paragraph that will fail to translate."""
    hrefs = ["chapter%d.xhtml" % (i + 1) for i in range(3)]
    manifest = "\n".join(
        '    <item id="ch%d" href="%s" media-type="application/xhtml+xml"/>'
        % (i + 1, href) for i, href in enumerate(hrefs)
    )
    spine = "\n".join('    <itemref idref="ch%d"/>' % (i + 1) for i in range(3))
    content_opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package version="3.0" xmlns="http://www.idpf.org/2007/opf" '
        'unique-identifier="bookid">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        '    <dc:identifier id="bookid">urn:uuid:261</dc:identifier>\n'
        '    <dc:title>The Seventh Lamp</dc:title>\n'
        '    <dc:language>en</dc:language>\n'
        '    <meta property="dcterms:modified">2024-01-01T00:00:00Z</meta>\n'
        '  </metadata>\n'
        '  <manifest>\n' + manifest + '\n  </manifest>\n'
        '  <spine>\n' + spine + '\n  </spine>\n'
        '</package>\n'
    )
    container_xml = (
        '<?xml version="1.0"?>\n'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles>\n'
        '    <rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/>\n'
        '  </rootfiles>\n'
        '</container>\n'
    )
    sabotaged = list(PARAGRAPHS)
    sabotaged[1] = SENTINEL + ", and the assistant never asked twice."

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", container_xml)
        z.writestr("OEBPS/content.opf", content_opf)
        for i, href in enumerate(hrefs):
            paragraphs = sabotaged if i == 1 else PARAGRAPHS
            z.writestr("OEBPS/" + href,
                       _chapter_xhtml("Chapter %d" % (i + 1), paragraphs))


def _chapter_text(epub_path, href):
    """Read one chapter out of an output EPUB (empty string when absent)."""
    if not Path(epub_path).exists():
        return ""
    with zipfile.ZipFile(epub_path) as z:
        for name in z.namelist():
            if name.endswith(href):
                return z.read(name).decode("utf-8", errors="replace")
    return ""


@pytest.fixture
def epub_job(tmp_path, monkeypatch):
    """An extracted-and-registered EPUB job with an isolated checkpoint store."""
    input_path = tmp_path / "seventh_lamp.epub"
    output_path = tmp_path / "seventh_lamp_fr.epub"
    _build_epub(input_path)

    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)

    translation_id = "retry261"
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

    # No LLM client is ever needed: chunk requests are stubbed per pass and the
    # packaging-metadata pass (the only other LLM caller) is switched off.
    monkeypatch.setattr(epub_translator, "_create_llm_client",
                        lambda **kwargs: MagicMock())
    monkeypatch.setattr(epub_translator, "EPUB_TRANSLATE_METADATA_ENABLED", False)

    return {
        'input': input_path,
        'output': output_path,
        'manager': manager,
        'translation_id': translation_id,
    }


# ---------------------------------------------------------------------------
# One translation pass
# ---------------------------------------------------------------------------

async def _run_pass(job, monkeypatch, resume_from_index, starve,
                    payload_sink=None, interrupt_after=None, degrade=False,
                    retry_token_aligned=False):
    """Run one EPUB pass and return (stats, chunk_requests, log_kinds).

    `starve` decides whether the sentinel chunk is answered. Every chunk-level
    LLM request is recorded, so a test can assert exactly which chunks were
    translated.

    `payload_sink`, when given, collects every stats payload in emission order.
    The returned `stats` dict is the merge of all of them (which is what the web
    layer keeps), so it can only answer "what did the panel end on"; the sink is
    how a test can look at the *first* emit of a pass.

    `interrupt_after`, when given, makes the interruption callback return True
    once that many chunk-level requests have been issued - the machinery of
    tests/test_xhtml_chunk_interruption.py, lifted to the book level so a plain
    interrupted resume can be told apart from a repair pass.

    `degrade` pushes the sentinel chunk into Phase 2 instead of Phase 3: the
    Phase 1 answer comes back with every placeholder dropped (so validation
    fails) while the placeholder-free Phase 2 request is answered normally, and
    token alignment reinserts the tags proportionally. The chunk ends up
    TRANSLATED with approximate tag positions - the state this feature is about.

    `retry_token_aligned` is the explicit opt-in the completion card sends: it
    widens the pass's work set to those chunks.
    """
    requests = []
    log_kinds = []

    async def fake_generate_translation_request(main_content, *args, **kwargs):
        requests.append(main_content)
        if starve and SENTINEL in main_content:
            return None
        if degrade and SENTINEL in main_content and kwargs.get('has_placeholders'):
            # Phase 1: a plausible translation that lost every placeholder, so
            # validation fails. Phase 2 asks again without placeholders (and
            # with has_placeholders=False), which this stub answers normally.
            return "%s %s" % (TRANSLATED_MARKER,
                              PlaceholderFormat.from_config().remove_all(main_content))
        return "%s %s" % (TRANSLATED_MARKER, main_content)

    monkeypatch.setattr(xhtml_translator, "generate_translation_request",
                        fake_generate_translation_request)

    # Which repair phase a starved chunk goes through is a property of the
    # scenario, not of the developer's .env: translate_chunk_with_fallback reads
    # EPUB_TOKEN_ALIGNMENT_ENABLED from src.config at call time, so without
    # pinning it here the same test measures Phase 2 on one machine and Phase 3
    # on another. `degrade` needs Phase 2 (that is the whole scenario); every
    # other pass in this file is about the Phase 3 fallback and must not have a
    # Phase 2 attempt counted into its numbers. Same technique as
    # tests/test_xhtml_chunk_interruption.py.
    monkeypatch.setattr('src.config.EPUB_TOKEN_ALIGNMENT_ENABLED', bool(degrade))

    stats = {}

    def stats_callback(payload):
        if payload_sink is not None:
            payload_sink.append(dict(payload))
        stats.update(payload)

    def log_callback(kind, message, **kwargs):
        log_kinds.append(kind)

    check_interruption = None
    if interrupt_after is not None:
        def check_interruption():
            return len(requests) >= interrupt_after

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
        log_callback=log_callback,
        stats_callback=stats_callback,
        check_interruption_callback=check_interruption,
        max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
        max_attempts=1,
        prompt_options={},
        retry_token_aligned=retry_token_aligned,
    )

    return stats, requests, log_kinds


def _job_progress(job):
    return job['manager'].get_job(job['translation_id'])['progress']


def _sentinel_requests(requests):
    return [text for text in requests if SENTINEL in text]


async def _first_pass(job, monkeypatch):
    """Pass 1: chapter 2's chunk is starved and ends up as source text."""
    stats, requests, _kinds = await _run_pass(
        job, monkeypatch, resume_from_index=0, starve=True)

    verdict = classify_completion(stats, str(job['output']))
    assert verdict.status == 'partial'
    assert verdict.fallback_chunks == 1

    # A fresh pass restores nothing, so the per-run counters (emitted
    # unconditionally, never only-sometimes) equal the accumulated ones.
    assert stats['run_processed_chunks'] == stats['processed_chunks'] == 3
    assert stats['run_fallback_used'] == stats['fallback_used'] == 1

    # Scope of the pass (the live progress panel's denominator). A fresh pass
    # is the whole book, and it is explicitly not a repair.
    assert stats['run_is_repair'] is False
    assert stats['run_total_chunks'] == stats['total_chunks'] == 3

    # Live count of chunks currently sitting in their source text. Unlike
    # `fallback_used` (accumulated across passes) this one is a projection of
    # the per-chunk statuses, which is what lets the Fallbacks stat card count
    # down when a retry heals a chunk.
    assert stats['untranslated_chunks'] == 1

    # Mirror what handlers.py does with a partial job: keep the checkpoint.
    job['manager'].mark_partial(job['translation_id'])
    return stats, requests


# ---------------------------------------------------------------------------
# 1 + 2. The fallback chunk is recorded, then retried and healed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_retries_only_the_unfinished_chunk(epub_job, monkeypatch):
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    # === PASS 1: chapter 2 starves ===
    await _first_pass(epub_job, monkeypatch)

    progress = _job_progress(epub_job)
    assert progress['epub_unfinished_units'] == {'chapter2.xhtml': [0]}
    # Only the file with unfinished work keeps its partial state; the two clean
    # chapters were deleted after being saved (state-key unification).
    assert manager.list_xhtml_partial_states(translation_id) == ['chapter2.xhtml']
    # The file pointer still says "all three files done".
    assert manager.load_checkpoint(translation_id)['resume_from_index'] == 3
    assert TRANSLATED_MARKER not in _chapter_text(epub_job['output'],
                                                  "chapter2.xhtml")

    # === PASS 2: resume with the starvation lifted ===
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']
    _stats2, requests2, kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index, starve=False)

    # Exactly one chunk was translated, and it is the sabotaged one.
    assert len(requests2) == 1
    assert SENTINEL in requests2[0]
    assert 'epub_retry_file' in kinds2
    assert 'epub_retry_state_missing' not in kinds2

    assert TRANSLATED_MARKER in _chapter_text(epub_job['output'],
                                              "chapter2.xhtml")

    # Chunk accounting: the re-entered file must be counted once, not twice
    # (it is excluded from the pre-loop sum and added back after processing).
    assert _stats2['completed_chunks'] == _stats2['total_chunks'] == 3

    # Per-run counters. The accumulated ones are rehydrated across passes on
    # purpose (issue #180: the Fallbacks card must not reset to zero), which
    # makes any percentage derived from them a cross-pass average - and worse,
    # a re-entered file replays its own restored metrics on top of the snapshot
    # that already counted them (processed_chunks: 3 restored + 1 replayed + 1
    # new = 5 for a 3-chunk book, fallback_used: 1 counted twice). The `run_*`
    # twins are what the completion card divides by, and they describe exactly
    # this pass: one chunk retried, cleanly.
    assert _stats2['run_processed_chunks'] == 1
    assert _stats2['run_fallback_used'] == 0
    assert _stats2['run_token_alignment_used'] == 0
    assert _stats2['run_successful_after_retry'] == 0
    assert _stats2['run_placeholder_errors'] == 0
    assert _stats2['processed_chunks'] == 5
    assert _stats2['fallback_used'] == 2

    # The live Fallbacks card counts down as the retry heals the chunk: the
    # accumulated counter still remembers both fallbacks (issue #180 - it must
    # not reset on resume), while `untranslated_chunks` describes the book as it
    # now stands, and nothing is in the source language any more.
    assert _stats2['untranslated_chunks'] == 0

    # Nothing is owed any more: no ticket, no partial state left.
    assert _job_progress(epub_job)['epub_unfinished_units'] == {}
    assert manager.list_xhtml_partial_states(translation_id) == []
    assert manager.load_xhtml_partial_state(translation_id,
                                            "chapter2.xhtml") is None

    # The resume pointer must not have rewound to the re-entered file: it stays
    # at "all files done", so a further resume cannot re-enter files 1 and 3
    # without a partial state.
    assert manager.load_checkpoint(translation_id)['resume_from_index'] == 3

    # The chapters that were already fine were neither re-translated nor lost.
    for href in ("chapter1.xhtml", "chapter3.xhtml"):
        assert TRANSLATED_MARKER in _chapter_text(epub_job['output'], href)


# ---------------------------------------------------------------------------
# 2b. The live Fallbacks card: hydrated on resume, then counting down
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_untranslated_chunks_is_hydrated_before_the_retry_runs(
        epub_job, monkeypatch):
    """The very first emit of a resumed pass already knows about the fallback.

    This is the issue #180 guard, transposed to the new counter: the Fallbacks
    card must show the damage inherited from the previous pass, not 0, before a
    single chunk of the retry has been translated. The pre-loop emit reads it
    from the per-file partial states the retry tickets are built from - the
    stored `epub_unfinished_units` map merges pending with untranslated and
    could not answer this.
    """
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    await _first_pass(epub_job, monkeypatch)
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']

    payloads = []
    stats2, _requests2, _kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index,
        starve=False, payload_sink=payloads)

    # First emit of the pass: nothing has been retried yet, and the card is
    # already at 1 (the chunk pass 1 left in English), not at 0.
    assert payloads
    assert payloads[0]['untranslated_chunks'] == 1
    # The re-entered file is deliberately excluded from the pre-loop chunk sum
    # (the loop adds it back once processed), so 2 of the 3 chunks are counted:
    # this really is the emit that precedes any retry work.
    assert payloads[0]['completed_chunks'] == 2

    # ... and by the end of the pass the same key has counted down to 0.
    assert payloads[-1]['untranslated_chunks'] == 0
    assert stats2['untranslated_chunks'] == 0
    # The card's other term is unaffected: no chunk ever needed Phase 2 here.
    assert stats2['token_alignment_used'] == 0


# ---------------------------------------------------------------------------
# 3. A retry that fails again: still partial, ticket kept, one attempt only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_retry_that_fails_again_keeps_the_ticket(epub_job, monkeypatch):
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    _stats1, requests1 = await _first_pass(epub_job, monkeypatch)
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']

    # === PASS 2': resume with the starvation still in place ===
    stats2, requests2, kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index, starve=True)

    # The sabotaged chunk walked the same ladder exactly once (Phase 1 with the
    # placeholders, Phase 2 without), and no other chunk was touched.
    assert requests2 == _sentinel_requests(requests1)
    assert len(requests2) >= 1
    assert 'epub_retry_file' in kinds2

    # This pass retried one chunk and it fell back again: 1 of 1, not 3 of 5.
    assert stats2['run_processed_chunks'] == 1
    assert stats2['run_fallback_used'] == 1
    assert stats2['fallback_used'] == 3

    # Still one chunk in the source language, so the card holds at 1 - it does
    # not follow the accumulated counter up to 3.
    assert stats2['untranslated_chunks'] == 1

    verdict2 = classify_completion(stats2, str(epub_job['output']))
    assert verdict2.status == 'partial'

    # The ticket and its payload are still there, so the user can retry again.
    assert _job_progress(epub_job)['epub_unfinished_units'] == {'chapter2.xhtml': [0]}
    assert manager.list_xhtml_partial_states(translation_id) == ['chapter2.xhtml']
    state = manager.load_xhtml_partial_state(translation_id, "chapter2.xhtml")
    assert state is not None
    assert state.validate() is True
    assert SENTINEL in state.translated_chunks[0]
    assert manager.load_checkpoint(translation_id)['resume_from_index'] == 3
    assert TRANSLATED_MARKER not in _chapter_text(epub_job['output'],
                                                 "chapter2.xhtml")


# ---------------------------------------------------------------------------
# 4. The live panel reports the repair pass, not the book
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repair_pass_reports_its_own_scope(epub_job, monkeypatch):
    """`run_total_chunks` / `run_is_repair` describe the pass in flight.

    The panel used to report a one-chunk repair of a 3-chapter book as
    "3 TOTAL / 2 COMPLETED", creeping to 3/3 in one tiny step with an ETA paced
    by chunks nobody was translating. These two keys are what let it read
    "1 TOTAL / 1 COMPLETED" instead - and they are live-payload only: the
    persisted progress keeps describing the book, because that is what the
    checkpoint and the resumable-job card need.
    """
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    await _first_pass(epub_job, monkeypatch)
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']

    payloads = []
    stats2, requests2, _kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index,
        starve=False, payload_sink=payloads)

    # Exactly one chunk was retried, and the pass says so.
    assert len(requests2) == 1
    assert stats2['run_is_repair'] is True
    assert stats2['run_total_chunks'] == 1
    assert stats2['run_processed_chunks'] == 1

    # The denominator is known before the first chunk of the pass: the very
    # first emit already carries 1, not 0 (no denominator yet) and not 3 (the
    # book). Without that the bar would have nothing to divide by on its first
    # frame, which is exactly when the user is looking at it.
    assert payloads
    assert payloads[0]['run_total_chunks'] == 1
    assert payloads[0]['run_is_repair'] is True
    assert payloads[0]['run_processed_chunks'] == 0

    # The book-level pair is untouched in the same payloads - the panel picks,
    # the payload never has to choose.
    assert stats2['total_chunks'] == 3
    assert stats2['completed_chunks'] == 3

    # ... and what got persisted is the book, not the repair. This is what the
    # resumable-job card renders as "Progress: X/Y chunks (Z%)": a 3-chunk book
    # must never read "1/1 chunks (100%)" there.
    progress = _job_progress(epub_job)
    assert progress['total_chunks'] == 3
    assert progress['completed_chunks'] == 3
    assert 'run_total_chunks' not in progress
    assert 'run_is_repair' not in progress


@pytest.mark.asyncio
async def test_repair_pass_that_fails_again_still_scopes_to_the_retry(
        epub_job, monkeypatch):
    """A retry that falls back again keeps the pass-level denominator.

    The failure mode this guards is a bar that can exceed 100% or stall below
    it: `run_processed_chunks` must count the same things `run_total_chunks`
    counts, whatever the outcome of the chunk.
    """
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    await _first_pass(epub_job, monkeypatch)
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']

    stats2, _requests2, _kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index, starve=True)

    assert stats2['run_is_repair'] is True
    assert stats2['run_total_chunks'] == 1
    assert stats2['run_processed_chunks'] == 1
    assert _job_progress(epub_job)['total_chunks'] == 3


# ---------------------------------------------------------------------------
# 5. A plain interrupted resume is NOT a repair pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interrupted_resume_is_not_a_repair_pass(epub_job, monkeypatch):
    """Resuming an interrupted book keeps book-level progress.

    Someone resuming a book at 33% expects 33 -> 100% of the book, not
    0 -> 100% of the remainder, so the work set derived from the file pointer
    must never set `run_is_repair`. Only retry tickets do.
    """
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    # Pass 1 stops right after chapter 1: nothing fell back, so nothing is
    # owed at the chunk level - the file pointer alone describes the remainder.
    stats1, requests1, _kinds1 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=0, starve=False,
        interrupt_after=1)
    assert len(requests1) == 1
    assert stats1['run_is_repair'] is False

    progress = _job_progress(epub_job)
    assert progress.get('epub_unfinished_units') in (None, {})
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']
    assert resume_from_index == 1

    payloads = []
    stats2, requests2, kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index,
        starve=False, payload_sink=payloads)

    # Chapters 2 and 3 were translated, and no file was re-entered by ticket.
    assert len(requests2) == 2
    assert 'epub_retry_file' not in kinds2
    assert stats2['run_is_repair'] is False
    assert payloads[0]['run_is_repair'] is False
    # Book-level pair still ends the pass at 3/3, exactly as before this change.
    assert stats2['total_chunks'] == 3
    assert stats2['completed_chunks'] == 3


# ---------------------------------------------------------------------------
# The repair denominator must not over-count the file the previous pass was
# interrupted inside
# ---------------------------------------------------------------------------

def test_pending_chunk_count_uses_the_partial_state_when_there_is_one(
        epub_job):
    """`_pending_chunk_count` answers "how many chunks will this file attempt".

    A file the previous pass was interrupted inside carries a partial state and
    resumes from `current_chunk_index`, so it attempts fewer chunks than the
    pre-count says. That file gets no ticket (a ticket is written by
    `_save_checkpoint`, which only runs for files that completed), so its state
    has to be read directly.

    It matters when a book was interrupted *and* an earlier file left a fallback
    behind: the tickets make the pass a repair, so the panel uses
    `run_total_chunks`, and an over-counted denominator leaves the repair bar
    stalled below 100%. The fixture here has one chunk per chapter, so only a
    direct test can distinguish the two values.
    """
    from src.core.epub.translator import _pending_chunk_count
    from src.core.epub.xhtml_translation_state import (
        CHUNK_PENDING,
        CHUNK_TRANSLATED,
        CHUNK_UNTRANSLATED,
        XHTMLTranslationState,
    )

    manager = epub_job['manager']
    translation_id = epub_job['translation_id']
    href = 'chapter1.xhtml'
    precounted = 6

    # No state at all: the file has never been entered, so the pre-count is the
    # only answer available and must be returned untouched.
    assert _pending_chunk_count(
        manager, translation_id, href, precounted) == precounted

    # Interrupted after 4 of 6 chunks, one of which fell back to source text.
    # Two chunks are still pending and the untranslated one is owed again, so
    # the file will attempt 3 - not the 6 the pre-count reports.
    chunks = [{'text': f'c{i}', 'local_tag_map': {}, 'global_indices': []}
              for i in range(precounted)]
    statuses = [CHUNK_TRANSLATED, CHUNK_UNTRANSLATED, CHUNK_TRANSLATED,
                CHUNK_TRANSLATED, CHUNK_PENDING, CHUNK_PENDING]
    state = XHTMLTranslationState(
        file_path=href, translation_id=translation_id, file_href=href,
        source_language='English', target_language='French',
        model_name=MODEL, max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
        max_retries=1, chunks=chunks, global_tag_map={},
        placeholder_format=('[id', ']'),
        translated_chunks=['t0', 't1', 't2', 't3'], current_chunk_index=4,
        original_body_html='', doc_metadata={}, stats={},
        created_at='2026-01-01T00:00:00Z', updated_at='2026-01-01T00:00:00Z',
        chunk_statuses=statuses,
    )
    assert state.validate() is True
    assert manager.save_xhtml_partial_state(translation_id, href, state) is True

    assert _pending_chunk_count(manager, translation_id, href, precounted) == 3

    # Without a checkpoint manager there is nothing to read, so the pre-count
    # stands rather than silently collapsing to 0.
    assert _pending_chunk_count(None, translation_id, href, precounted) == precounted


# ---------------------------------------------------------------------------
# 6. Token-aligned chunks: translated, never automatic, retryable on demand
# ---------------------------------------------------------------------------

async def _degraded_first_pass(job, monkeypatch):
    """Pass 1 where chapter 2's chunk is repaired by token alignment.

    The chunk comes out TRANSLATED with approximate tag positions, which is a
    'completed' book (D3): nothing is owed, nothing is in the source language.
    """
    stats, requests, kinds = await _run_pass(
        job, monkeypatch, resume_from_index=0, starve=False, degrade=True)

    # Phase 2 really is what happened: one alignment, no Phase 3 fallback.
    assert stats['token_alignment_used'] == 1
    assert stats['fallback_used'] == 0
    return stats, requests, kinds


@pytest.mark.asyncio
async def test_token_aligned_chunk_is_kept_and_indexed_without_moving_the_verdict(
        epub_job, monkeypatch):
    """The payload a later repair needs survives, and the job stays completed.

    Two things used to make this impossible: the verdict said 'completed', so
    handlers.py destroyed the checkpoint, and `_save_checkpoint` deleted the
    per-file XHTML state of every file that owed nothing. The state is now kept
    for a degraded file too, and the job-level index records where those chunks
    are - in its OWN map, so `unfinished_chunks` stays 0 and the verdict does
    not move to 'partial' (which is what D3/D10 forbid).
    """
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    stats, _requests, kinds = await _degraded_first_pass(epub_job, monkeypatch)

    # The verdict does not move: an approximate tag placement is not unfinished
    # work.
    verdict = classify_completion(stats, str(epub_job['output']))
    assert verdict.status == 'completed'
    assert stats['unfinished_chunks'] == 0
    assert stats['untranslated_chunks'] == 0
    assert stats['unfinished_files'] == {}

    # ... but the live degraded map knows exactly which chunk is approximate.
    assert stats['degraded_chunks'] == 1
    assert stats['degraded_files'] == {'chapter2.xhtml': [0]}

    # Persisted, separately from the unfinished index.
    progress = _job_progress(epub_job)
    assert progress['epub_degraded_units'] == {'chapter2.xhtml': [0]}
    assert progress['epub_unfinished_units'] == {}

    # The retention that makes the retry possible at all (D7): only the degraded
    # file keeps its partial state, and the log says why.
    assert manager.list_xhtml_partial_states(translation_id) == ['chapter2.xhtml']
    assert 'xhtml_partial_state_kept_degraded' in kinds

    # The chunk IS translated - this is not a fallback left in English.
    assert TRANSLATED_MARKER in _chapter_text(epub_job['output'],
                                              "chapter2.xhtml")


@pytest.mark.asyncio
async def test_normal_resume_never_retries_a_token_aligned_chunk(
        epub_job, monkeypatch):
    """Design decision D3, pinned: the automatic work set ignores them.

    A plain Resume of the same job must translate nothing at all - no ticket is
    granted for a file whose only imperfection is tag placement - and it must
    leave the degraded index and its payload untouched, so the explicit action
    is still available afterwards.
    """
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    await _degraded_first_pass(epub_job, monkeypatch)
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']

    stats2, requests2, kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index, starve=False)

    assert requests2 == []
    assert 'epub_retry_file' not in kinds2
    assert 'epub_retry_token_aligned_requested' not in kinds2

    # Nothing moved: same index, same payload, same verdict.
    assert _job_progress(epub_job)['epub_degraded_units'] == {'chapter2.xhtml': [0]}
    assert manager.list_xhtml_partial_states(translation_id) == ['chapter2.xhtml']
    assert stats2['degraded_files'] == {'chapter2.xhtml': [0]}
    assert stats2['unfinished_chunks'] == 0
    assert classify_completion(stats2, str(epub_job['output'])).status == 'completed'


@pytest.mark.asyncio
async def test_retry_token_aligned_retries_exactly_that_chunk(
        epub_job, monkeypatch):
    """The opt-in pass: exactly the degraded chunk, and the map empties.

    `token_alignment_used` stays positive afterwards - it is an accumulated
    tally of Phase 2 events and must not be reset (D10) - which is precisely why
    the button, the chip and the note are driven by the map instead.
    """
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    await _degraded_first_pass(epub_job, monkeypatch)
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']

    stats2, requests2, kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index,
        starve=False, retry_token_aligned=True)

    # Exactly one chunk was retried, and it is the degraded one.
    assert len(requests2) == 1
    assert SENTINEL in requests2[0]
    assert 'epub_retry_token_aligned_requested' in kinds2
    assert 'epub_retry_file' in kinds2

    # The repair pass is scoped to itself, like every other repair pass.
    assert stats2['run_is_repair'] is True
    assert stats2['run_total_chunks'] == 1
    assert stats2['run_processed_chunks'] == 1

    # The map empties while the accumulated counter keeps its memory: this is
    # THE trap the frontend precedence rule exists for.
    assert stats2['degraded_chunks'] == 0
    assert stats2['degraded_files'] == {}
    assert stats2['token_alignment_used'] > 0
    assert stats2['unfinished_chunks'] == 0
    assert classify_completion(stats2, str(epub_job['output'])).status == 'completed'

    # Nothing is owed and nothing is degraded any more, so the payload goes
    # away too - both the index and the per-file state.
    assert _job_progress(epub_job)['epub_degraded_units'] == {}
    assert manager.list_xhtml_partial_states(translation_id) == []

    # The other chapters were neither re-translated nor lost.
    assert len(_sentinel_requests(requests2)) == 1
    for href in ("chapter1.xhtml", "chapter2.xhtml", "chapter3.xhtml"):
        assert TRANSLATED_MARKER in _chapter_text(epub_job['output'], href)

    # D4: the retried chunk sits below the file pointer, so the pointer must not
    # rewind and the persisted state must keep validating (no hole, no pending
    # below current_chunk_index).
    assert manager.load_checkpoint(translation_id)['resume_from_index'] == 3


@pytest.mark.asyncio
async def test_retry_token_aligned_that_falls_back_keeps_the_translation(
        epub_job, monkeypatch):
    """A failed repair must never make the book worse than it was.

    The chunk was translated (approximate tags only). If the retry ends in
    Phase 3 - source text - overwriting the slot would replace a translation
    with English AND move the verdict to 'partial'. The previous translation is
    kept instead; only the accumulated fallback counter records the attempt.
    """
    manager = epub_job['manager']
    translation_id = epub_job['translation_id']

    await _degraded_first_pass(epub_job, monkeypatch)
    resume_from_index = manager.load_checkpoint(translation_id)['resume_from_index']

    stats2, requests2, kinds2 = await _run_pass(
        epub_job, monkeypatch, resume_from_index=resume_from_index,
        starve=True, retry_token_aligned=True)

    assert len(requests2) >= 1
    assert 'chunk_retry_kept_previous' in kinds2

    # Verdict unchanged, nothing in the source language, chunk still degraded.
    assert stats2['unfinished_chunks'] == 0
    assert stats2['untranslated_chunks'] == 0
    assert stats2['degraded_files'] == {'chapter2.xhtml': [0]}
    assert classify_completion(stats2, str(epub_job['output'])).status == 'completed'
    assert TRANSLATED_MARKER in _chapter_text(epub_job['output'],
                                              "chapter2.xhtml")

    # Still retryable: the index and the payload survive another round.
    assert _job_progress(epub_job)['epub_degraded_units'] == {'chapter2.xhtml': [0]}
    state = manager.load_xhtml_partial_state(translation_id, "chapter2.xhtml")
    assert state is not None
    assert state.validate() is True
