"""Pin the stats fields the web frontend reads against the real payload.

Every format runs through the real :class:`JobRunner` with the echo provider
and the real :class:`WebJobSinks` (state-manager merge, canonical
``percent``/``phase`` derivation, monotonic floor); the ``stats`` payloads
Socket.IO would carry are captured. For each format the test asserts that
every engine-derived field the frontend reads, and that the format emitted
before the engines moved onto ``ProgressReporter``, is still present, plus
the sink's ``percent`` and ``phase``.

The frontend readers are ``src/web/static/js/translation/progress-manager.js``
and ``src/web/static/js/translation/translation-tracker.js`` (field list
cross-checked against the ``refactor-phase4b-frontend-js`` branch). The
per-format sets below were recorded from the payloads at commit 2a785d2.
"""

import asyncio
import pytest

from src.api.handlers import WebJobSinks
from src.api.translation_state import TranslationStateManager
from src.core.job import JobConfig
from src.core.job.runner import JobRunner
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm, fixtures

# Engine-derived fields the frontend reads.
FRONTEND_ENGINE_FIELDS = frozenset({
    "total_chunks", "completed_chunks", "failed_chunks", "processed_chunks",
    "successful_after_retry", "fallback_used", "placeholder_errors",
    "quality_warning_fired", "token_alignment_used",
    "paragraph_count_mismatches", "paragraph_repair_failed",
    "unfinished_chunks", "unfinished_files", "untranslated_chunks",
    "degraded_chunks", "degraded_files",
    "run_total_chunks", "run_is_repair", "run_processed_chunks",
    "run_successful_after_retry", "run_token_alignment_used",
    "run_fallback_used", "run_placeholder_errors",
})

# Canonical fields the sink derives on every emit.
SINK_FIELDS = frozenset({"percent", "phase"})

PGN = """[Event "Test"]
[Result "*"]

1. d4 {The queen's pawn opening, a solid choice.} d5
2. c4 {Offering a pawn for the centre.} e6 *
"""


def _build_pgn(dest_dir):
    path = dest_dir / "sample.pgn"
    path.write_text(PGN, encoding="utf-8")
    return path


def _build_pdf(dest_dir):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "A Simple Title", fontsize=20, fontname="hebo")
    for i in range(4):
        page.insert_text((72, 130 + 60 * i), fixtures._para(i)[:80], fontsize=11, fontname="helv")
    path = dest_dir / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path


_BUILDERS = {
    "txt": fixtures.build_txt,
    "srt": fixtures.build_srt,
    "docx": fixtures.build_docx,
    "epub": fixtures.build_epub,
    "pdf": _build_pdf,
    "pgn": _build_pgn,
}


class _FakeSocketIO:
    def __init__(self):
        self.stats = []

    def emit(self, event, payload, namespace=None):
        if isinstance(payload, dict) and isinstance(payload.get("stats"), dict):
            self.stats.append(dict(payload["stats"]))


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    fake_llm.install(monkeypatch)


def run_web_job(tmp_path, fmt, mode):
    """Run one job through the runner and the web sink; return the emitted stats."""
    source = _BUILDERS[fmt](tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    state = TranslationStateManager(checkpoint_manager=manager)
    tid = f"web_{fmt}_{mode}"
    state.create_translation(tid, {})
    socketio = _FakeSocketIO()
    fields = dict(
        file_type=fmt, input_path=str(source),
        output_path=str(tmp_path / f"out{source.suffix}"),
        source_language="English", target_language="French",
        provider=ProviderConfig(provider="poe", model="fake-echo", api_key="UNUSED_FAKE_KEY"),
        context_window=4096, auto_adjust_context=False, max_tokens_per_chunk=60,
        origin='cli',
    )
    if mode == "refine":
        fields["operation"] = "refine"
    elif mode == "translate_refine":
        fields.update(refine_after=True, prompt_options={"refine": True})
    job = JobConfig(**fields)
    asyncio.run(JobRunner(job, translation_id=tid, checkpoint_manager=manager,
                          sinks=WebJobSinks(tid, state, socketio)).run())
    return socketio.stats


_GENERIC = frozenset({
    "total_chunks", "completed_chunks", "failed_chunks", "unfinished_chunks",
    "run_processed_chunks", "run_successful_after_retry", "run_token_alignment_used",
    "run_fallback_used", "run_placeholder_errors",
})
_METRICS = frozenset({
    "total_chunks", "completed_chunks", "failed_chunks", "processed_chunks",
    "successful_after_retry", "fallback_used", "placeholder_errors",
    "quality_warning_fired", "token_alignment_used",
    "paragraph_count_mismatches", "paragraph_repair_failed",
})
_COUNTS = frozenset({"total_chunks", "completed_chunks", "failed_chunks"})

# (format, mode) -> engine-derived frontend fields the payload carried at
# 2a785d2. A translate + refine job keeps the translate phase's fields (the
# web state merges key by key).
EXPECTED_FIELDS = {
    ("txt", "translate"): _GENERIC,
    ("srt", "translate"): _GENERIC,
    ("pgn", "translate"): _GENERIC,
    ("docx", "translate"): _METRICS,
    ("pdf", "translate"): _METRICS,
    ("epub", "translate"): FRONTEND_ENGINE_FIELDS,
    ("txt", "refine"): _COUNTS,
    ("srt", "refine"): _COUNTS,
    ("docx", "refine"): _COUNTS,
    ("epub", "refine"): _COUNTS,
    ("txt", "translate_refine"): _GENERIC,
    ("srt", "translate_refine"): _GENERIC,
    ("docx", "translate_refine"): _METRICS,
    ("epub", "translate_refine"): FRONTEND_ENGINE_FIELDS,
}


@pytest.mark.parametrize("fmt,mode", sorted(EXPECTED_FIELDS))
def test_frontend_fields_survive(fmt, mode, tmp_path):
    stats = run_web_job(tmp_path, fmt, mode)
    assert stats, f"{fmt}/{mode}: no stats emitted"
    keys = set()
    for payload in stats:
        keys.update(payload)
    missing = (EXPECTED_FIELDS[(fmt, mode)] | SINK_FIELDS) - keys
    assert not missing, f"{fmt}/{mode}: fields the frontend reads disappeared: {sorted(missing)}"

    percents = [payload["percent"] for payload in stats]
    assert percents == sorted(percents), f"{fmt}/{mode}: percent regressed"
    assert percents[-1] == 100.0
    last = stats[-1]
    assert last["completed_chunks"] == last["total_chunks"] > 0
