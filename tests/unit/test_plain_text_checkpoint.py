"""
Segment-level checkpoint and resume for Plain Text Mode (backlog item 2.4).

Before this, a rate-limit pause or an interruption during Plain Text Mode threw
away every segment already translated: the pipeline reassembled a partial result
and the adapters dropped it on the floor. These tests pin the replacement
behaviour:

- the pipeline hands a contiguous, gap-free prefix to a checkpoint hook;
- rate limit and interruption both persist that prefix before giving up;
- a resume replays the stored segmentation and never retries a stored segment;
- a checkpoint that does not match the source is discarded, not trusted;
- a failing checkpoint backend degrades persistence, never the translation;
- the EPUB adapter round-trips the whole thing through a real CheckpointManager.
"""
import asyncio

import pytest
from lxml import etree

import src.core.common.plain_text_pipeline as plain_pipeline
from src.core.llm.exceptions import RateLimitError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
MAX_TOKENS = 20

# Each paragraph is big enough that the token chunker gives it its own segment,
# so segment index == paragraph index and the assertions stay readable.
PARAGRAPHS = [
    f"Paragraph number {i} with enough words to be its own chunk."
    for i in range(12)
]


def _fake_llm(seen, fail_at=None):
    """Fake generate_translation_request recording every main_content it gets.

    `fail_at` is a 0-based call ordinal that raises RateLimitError instead of
    translating (sequential mode only, where call order == segment order).
    """
    async def fake_request(*, main_content, **kwargs):
        await asyncio.sleep(0)
        ordinal = len(seen)
        seen.append(main_content)
        if fail_at is not None and ordinal == fail_at:
            raise RateLimitError("429 Too Many Requests", retry_after=1, provider="test")
        return f"T::{main_content}"
    return fake_request


class _HookRecorder:
    """Records every checkpoint_hook invocation; optionally always fails."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, segments, prefix, next_index, stats_dict, chunk_statuses=None):
        self.calls.append({
            'segments': segments,
            'prefix': list(prefix),
            'next_index': next_index,
            'stats': stats_dict,
            'statuses': list(chunk_statuses) if chunk_statuses is not None else None,
        })
        if self.fail:
            raise RuntimeError("checkpoint backend unavailable")

    @property
    def indices(self):
        return [c['next_index'] for c in self.calls]

    @property
    def last(self):
        return self.calls[-1]


@pytest.fixture
def patched_llm(monkeypatch):
    """Install the fake LLM + identity post-processing, return the seen list."""
    seen = []

    def install(fail_at=None):
        monkeypatch.setattr(
            plain_pipeline, "generate_translation_request", _fake_llm(seen, fail_at)
        )
        monkeypatch.setattr(plain_pipeline, "clean_translated_text", lambda s, **kw: s)
        return seen

    return install


async def _translate(**overrides):
    kwargs = dict(
        paragraphs=PARAGRAPHS,
        source_language="English",
        target_language="French",
        model_name="m",
        llm_client=object(),
        max_tokens_per_chunk=MAX_TOKENS,
        parallel_workers=1,
    )
    kwargs.update(overrides)
    return await plain_pipeline.translate_paragraphs_plain(**kwargs)


def _segment_count():
    return len(plain_pipeline.build_plain_segments(PARAGRAPHS, MAX_TOKENS))


# ---------------------------------------------------------------------------
# 1. Hook contract, happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hook_receives_contiguous_increasing_prefixes(patched_llm):
    patched_llm()
    hook = _HookRecorder()
    total = _segment_count()
    assert total >= 6, "the fixture must produce enough segments to be interesting"

    out, stats, interrupted = await _translate(checkpoint_hook=hook, checkpoint_every=2)

    assert not interrupted
    assert out == [f"T::{p}" for p in PARAGRAPHS]
    assert hook.calls, "the hook was never called"
    # Strictly increasing resume points.
    assert hook.indices == sorted(set(hook.indices))
    for call in hook.calls:
        assert len(call['prefix']) == call['next_index']
        assert all(part is not None for part in call['prefix'])
    assert hook.last['next_index'] == total


# ---------------------------------------------------------------------------
# 2. Rate limit persists the prefix
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rate_limit_persists_prefix_and_propagates(patched_llm):
    seen = patched_llm(fail_at=3)
    hook = _HookRecorder()

    with pytest.raises(RateLimitError) as excinfo:
        await _translate(checkpoint_hook=hook)

    assert hook.last['next_index'] == 3
    assert hook.last['prefix'] == [f"T::{p}" for p in PARAGRAPHS[:3]]
    # The reassembled partial travels with the exception (P4a attribute).
    assert excinfo.value.partial_result is not None
    assert excinfo.value.partial_result[:3] == [f"T::{p}" for p in PARAGRAPHS[:3]]
    # Segment 3 was attempted (and failed); nothing past it was translated.
    assert seen == PARAGRAPHS[:4]


# ---------------------------------------------------------------------------
# 3. Interruption persists the prefix
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_interruption_persists_prefix(patched_llm):
    seen = patched_llm()
    hook = _HookRecorder()

    out, stats, interrupted = await _translate(
        checkpoint_hook=hook,
        check_interruption_callback=lambda: len(seen) >= 2,
    )

    assert interrupted is True
    assert hook.last['next_index'] == 2
    assert hook.last['prefix'] == [f"T::{p}" for p in PARAGRAPHS[:2]]
    # The untranslated tail keeps its source text.
    assert out[2] == PARAGRAPHS[2]


# ---------------------------------------------------------------------------
# 4. Resume skips completed segments
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resume_skips_completed_segments(patched_llm):
    # Run 1: rate-limited after three segments.
    patched_llm(fail_at=3)
    hook = _HookRecorder()
    with pytest.raises(RateLimitError):
        await _translate(checkpoint_hook=hook)

    segments = hook.last['segments']
    prefix = hook.last['prefix']
    assert len(prefix) == 3

    # Run 2: resume from the checkpoint with a fresh, always-succeeding LLM.
    seen = patched_llm()
    seen.clear()

    out, stats, interrupted = await _translate(
        resume_segments=segments,
        resume_translated=prefix,
    )

    assert not interrupted
    assert len(seen) == len(segments) - 3
    # None of the restored segments was sent to the LLM again.
    for done in segments[:3]:
        assert done['text'] not in seen
    # The restored translations are present in the final output...
    assert out[:3] == prefix
    # ...and the remainder was translated in this run.
    assert out[3:] == [f"T::{p}" for p in PARAGRAPHS[3:]]
    # Progress accounts for the restored work, not just this run's segments.
    assert stats.processed_chunks == len(segments)


# ---------------------------------------------------------------------------
# 5. Mismatched resume is discarded
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mismatched_resume_is_discarded(patched_llm):
    seen = patched_llm()
    logs = []
    total = _segment_count()

    segments = plain_pipeline.build_plain_segments(PARAGRAPHS, MAX_TOKENS)
    out, stats, interrupted = await _translate(
        resume_segments=segments[:2],
        resume_translated=["stale a", "stale b", "stale c"],
        log_callback=lambda key, msg: logs.append((key, msg)),
    )

    assert not interrupted
    assert len(seen) == total
    assert out == [f"T::{p}" for p in PARAGRAPHS]
    assert any(key == "plain_text_resume_discarded" for key, _ in logs)


@pytest.mark.asyncio
async def test_no_resume_does_not_log_a_discard(patched_llm):
    """A plain run without a checkpoint must not warn about a mismatch."""
    patched_llm()
    logs = []
    await _translate(log_callback=lambda key, msg: logs.append((key, msg)))
    assert not any(key == "plain_text_resume_discarded" for key, _ in logs)


# ---------------------------------------------------------------------------
# 6. Hook failure is non-fatal
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_failing_hook_does_not_abort_translation(patched_llm):
    patched_llm()
    hook = _HookRecorder(fail=True)
    logs = []

    out, stats, interrupted = await _translate(
        checkpoint_hook=hook,
        checkpoint_every=2,
        log_callback=lambda key, msg: logs.append((key, msg)),
    )

    assert not interrupted
    assert out == [f"T::{p}" for p in PARAGRAPHS]
    assert len(hook.calls) >= 2, "the hook kept being called after it failed"
    assert any(key == "plain_text_checkpoint_failed" for key, _ in logs)


# ---------------------------------------------------------------------------
# EPUB adapter round-trip (7) and resume rejection (8, 9)
# ---------------------------------------------------------------------------
def _epub_doc():
    body = "".join(f"<p>{p}</p>" for p in PARAGRAPHS)
    return etree.fromstring(f"<html><body>{body}</body></html>")


def _checkpoint_manager(tmp_path):
    from src.persistence.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)
    return manager


async def _epub_plain_run(resume_state=None, **overrides):
    from src.core.epub.body_serializer import find_body
    from src.core.epub.plain_text_document import translate_plain_text
    from tests.epub_engine_helpers import make_settings

    kwargs = dict(
        source_language="English",
        target_language="French",
        model_name="m",
        llm_client=object(),
        max_tokens_per_chunk=MAX_TOKENS,
        log_callback=None,
        context_manager=None,
        prompt_options={'plain_text_mode': True},
        stats_callback=None,
        bilingual=False,
        file_href="OEBPS/chapter1.xhtml",
        parallel_workers=1,
    )
    kwargs.update(overrides)
    return await translate_plain_text(find_body(_epub_doc()), make_settings(**kwargs), resume_state)


def _spy_on_pipeline(monkeypatch):
    """Record the kwargs the adapter passes to translate_paragraphs_plain."""
    recorded = []
    real = plain_pipeline.translate_paragraphs_plain

    async def spy(**kwargs):
        recorded.append(kwargs)
        return await real(**kwargs)

    monkeypatch.setattr(plain_pipeline, "translate_paragraphs_plain", spy)
    return recorded


@pytest.mark.asyncio
async def test_epub_adapter_checkpoint_roundtrip(patched_llm, tmp_path):
    """7. Rate limit -> real checkpoint on disk -> resume translates the rest."""
    manager = _checkpoint_manager(tmp_path)
    tid, href = "job-plain", "OEBPS/chapter1.xhtml"
    total = _segment_count()

    seen = patched_llm(fail_at=3)
    with pytest.raises(RateLimitError):
        await _epub_plain_run(
            checkpoint_manager=manager,
            translation_id=tid,
            file_href=href,
        )

    state = manager.load_xhtml_partial_state(tid, href)
    assert state is not None, "the persisted state failed validate() and was dropped"
    assert state.doc_metadata['plain_text_mode'] is True
    assert state.doc_metadata['paragraph_count'] == len(PARAGRAPHS)
    assert state.current_chunk_index == 3
    assert len(state.chunks) == total
    assert len(state.translated_chunks) == 3

    # Resume: only the remainder must reach the LLM.
    seen = patched_llm()
    seen.clear()
    success, stats = await _epub_plain_run(
        checkpoint_manager=manager,
        translation_id=tid,
        file_href=href,
        resume_state=state,
    )

    assert success is True
    assert len(seen) == total - 3
    reloaded = manager.load_xhtml_partial_state(tid, href)
    assert reloaded is not None and reloaded.current_chunk_index == total


@pytest.mark.asyncio
async def test_epub_adapter_rejects_placeholder_mode_state(patched_llm, monkeypatch, tmp_path):
    """8. A state without the plain_text_mode marker must not be replayed."""
    from src.core.epub.xhtml_translation_state import XHTMLTranslationState

    seen = patched_llm()
    recorded = _spy_on_pipeline(monkeypatch)
    logs = []

    placeholder_state = XHTMLTranslationState(
        file_path="OEBPS/chapter1.xhtml",
        translation_id="job-plain",
        file_href="OEBPS/chapter1.xhtml",
        source_language="English",
        target_language="French",
        model_name="m",
        max_tokens_per_chunk=MAX_TOKENS,
        max_retries=1,
        chunks=[{'text': "[id0]x[id1]", 'local_tag_map': {}, 'global_indices': []}],
        global_tag_map={},
        placeholder_format=("[id", "]"),
        translated_chunks=["already"],
        current_chunk_index=1,
        original_body_html="",
        doc_metadata={'namespaces': {}},
        stats={},
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    success, stats = await _epub_plain_run(
        resume_state=placeholder_state,
        log_callback=lambda key, msg: logs.append((key, msg)),
    )

    assert success is True
    assert recorded[0]['resume_segments'] is None
    assert recorded[0]['resume_translated'] is None
    assert len(seen) == _segment_count()
    assert any(key == "plain_text_resume_ignored" for key, _ in logs)


@pytest.mark.asyncio
async def test_epub_adapter_rejects_paragraph_count_mismatch(patched_llm, monkeypatch, tmp_path):
    """9. A plain-text state built for a different source must be ignored."""
    from src.core.epub.xhtml_translation_state import XHTMLTranslationState

    seen = patched_llm()
    recorded = _spy_on_pipeline(monkeypatch)
    logs = []

    segments = plain_pipeline.build_plain_segments(PARAGRAPHS, MAX_TOKENS)
    stale_state = XHTMLTranslationState(
        file_path="OEBPS/chapter1.xhtml",
        translation_id="job-plain",
        file_href="OEBPS/chapter1.xhtml",
        source_language="English",
        target_language="French",
        model_name="m",
        max_tokens_per_chunk=MAX_TOKENS,
        max_retries=1,
        chunks=segments,
        global_tag_map={},
        placeholder_format=("", ""),
        translated_chunks=["restored 0", "restored 1"],
        current_chunk_index=2,
        original_body_html="",
        doc_metadata={
            'plain_text_mode': True,
            'paragraph_count': len(PARAGRAPHS) + 1,  # source changed under us
        },
        stats={},
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    success, stats = await _epub_plain_run(
        resume_state=stale_state,
        log_callback=lambda key, msg: logs.append((key, msg)),
    )

    assert success is True
    assert recorded[0]['resume_segments'] is None
    assert len(seen) == _segment_count()
    assert any(key == "plain_text_resume_ignored" for key, _ in logs)


# ---------------------------------------------------------------------------
# DOCX adapter: same checkpoint, plus the explicit completion delete
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_docx_adapter_checkpoints_then_deletes_on_success(patched_llm, tmp_path):
    """DOCX has no post-save seam like EPUB, so it must clean up itself."""
    from docx import Document

    from src.core.docx.docx_translation_adapter import DocxTranslationAdapter

    doc = Document()
    for paragraph in PARAGRAPHS:
        doc.add_paragraph(paragraph)
    source_path = str(tmp_path / "sample.docx")
    doc.save(source_path)

    manager = _checkpoint_manager(tmp_path)
    tid, href = "job-docx-plain", "sample.docx"

    async def run(**overrides):
        kwargs = dict(
            source_path=source_path,
            source_language="English",
            target_language="French",
            model_name="m",
            llm_client=object(),
            max_tokens_per_chunk=MAX_TOKENS,
            log_callback=None,
            context_manager=None,
            prompt_options={'plain_text_mode': True},
            stats_callback=None,
            check_interruption_callback=None,
            parallel_workers=1,
            file_href=href,
            checkpoint_manager=manager,
            translation_id=tid,
        )
        kwargs.update(overrides)
        return await DocxTranslationAdapter()._translate_plain_text(**kwargs)

    # A rate limit leaves a resumable checkpoint behind.
    patched_llm(fail_at=3)
    with pytest.raises(RateLimitError):
        await run()

    state = manager.load_xhtml_partial_state(tid, href)
    assert state is not None
    assert state.doc_metadata['plain_text_mode'] is True
    assert state.current_chunk_index == 3

    # Resuming to completion translates only the tail and clears the state.
    seen = patched_llm()
    seen.clear()
    docx_bytes, stats = await run(resume_state=state)

    assert docx_bytes
    assert len(seen) == len(state.chunks) - 3
    assert manager.load_xhtml_partial_state(tid, href) is None


# ---------------------------------------------------------------------------
# The mirror guard: a plain-text state must never reach a placeholder pipeline
# ---------------------------------------------------------------------------
def _plain_state(**overrides):
    """Build a Plain Text Mode partial state, as the hook would persist it."""
    from src.core.epub.xhtml_translation_state import XHTMLTranslationState

    segments = plain_pipeline.build_plain_segments(PARAGRAPHS, MAX_TOKENS)
    fields = dict(
        file_path="OEBPS/chapter1.xhtml",
        translation_id="job-plain",
        file_href="OEBPS/chapter1.xhtml",
        source_language="English",
        target_language="French",
        model_name="m",
        max_tokens_per_chunk=MAX_TOKENS,
        max_retries=1,
        chunks=segments,
        global_tag_map={},
        placeholder_format=("", ""),
        translated_chunks=["restored 0", "restored 1"],
        current_chunk_index=2,
        original_body_html="",
        doc_metadata={'plain_text_mode': True, 'paragraph_count': len(PARAGRAPHS)},
        stats={},
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    fields.update(overrides)
    return XHTMLTranslationState(**fields)


class TestPlainStateIsRejectedByPlaceholderPipelines:
    """
    Plain Text Mode now writes an XHTMLTranslationState where it previously
    wrote none. Pausing a plain-text job and resuming it with Plain Text Mode
    switched off would otherwise hand that state to the placeholder pipeline,
    which would replay the prefix under an empty placeholder scheme.
    """

    def test_predicate_identifies_each_state_kind(self):
        from src.core.common.plain_text_checkpoint import is_plain_text_state

        assert is_plain_text_state(_plain_state()) is True
        assert is_plain_text_state(None) is False
        assert is_plain_text_state(object()) is False
        assert is_plain_text_state(_plain_state(doc_metadata={})) is False
        assert is_plain_text_state(
            _plain_state(doc_metadata={'plain_text_mode': False})
        ) is False

    @pytest.mark.asyncio
    async def test_xhtml_translator_ignores_a_plain_state(self, monkeypatch):
        """The EPUB placeholder path restarts the file instead of replaying it."""
        from src.core.epub import document_translation
        from tests.epub_engine_helpers import translate_xhtml

        captured = {}

        async def _fake_chunk_loop(*args, **kwargs):
            captured['start_index'] = kwargs.get('start_chunk_index')
            raise _StopTranslation()

        monkeypatch.setattr(
            document_translation,
            "translate_chunks",
            _fake_chunk_loop,
        )

        logs = []
        with pytest.raises(_StopTranslation):
            await translate_xhtml(
                doc_root=_epub_doc(),
                source_language="English",
                target_language="French",
                model_name="m",
                llm_client=object(),
                max_tokens_per_chunk=MAX_TOKENS,
                log_callback=lambda key, msg: logs.append((key, msg)),
                resume_state=_plain_state(),
            )

        assert any(key == "xhtml_resume_ignored" for key, _ in logs)
        assert not any(key == "xhtml_resume_partial" for key, _ in logs)
        # Restarted from the top rather than from the plain state's index 2.
        assert captured['start_index'] == 0


class _StopTranslation(Exception):
    """Sentinel used to stop a pipeline once the assertion point is reached."""


    @pytest.mark.asyncio
    async def test_docx_adapter_ignores_a_plain_state(self, monkeypatch):
        """The DOCX placeholder path re-extracts instead of replaying."""
        from src.core.docx.docx_translation_adapter import DocxTranslationAdapter

        def _fake_extract(self, source_path, log_callback=None):
            raise _StopTranslation()

        monkeypatch.setattr(
            DocxTranslationAdapter, "extract_content", _fake_extract
        )

        logs = []
        with pytest.raises(_StopTranslation):
            await DocxTranslationAdapter().translate_content(
                raw_content="unused.docx",
                source_language="English",
                target_language="French",
                model_name="m",
                llm_client=object(),
                max_tokens_per_chunk=MAX_TOKENS,
                log_callback=lambda key, msg: logs.append((key, msg)),
                prompt_options={},
                resume_state=_plain_state(),
            )

        assert any(key == "docx_resume_ignored" for key, _ in logs)
        assert not any(key == "docx_resume_partial" for key, _ in logs)
