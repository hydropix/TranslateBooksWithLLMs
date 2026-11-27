"""Pipeline integration of the Plain Text Mode emphasis option (issue #352).

The extractor and the rebuild are covered by test_plain_emphasis.py. These
tests pin everything between them and around them:

- strip_hallucinated_markup keeps bare emphasis markers only with keep_emphasis;
- the sentence splitter keeps a closing marker with the sentence it closes;
- the option only reaches the EPUB engine (JobConfig.engine_prompt_options),
  and the DOCX plain path never turns the emphasis post-processing on;
- a checkpoint written with the other setting is not reused;
- the precount extracts with the same flag as the translation;
- translate_plain_text end to end with a fake LLM: markers kept, broken
  markers falling back to plain, hallucinated tags stripped, stats logged.
"""
import pytest
from lxml import etree

import src.core.common.plain_text_pipeline as plain_pipeline
from src.core.common.emphasis_markers import remove_markers
from src.core.common.plain_text_checkpoint import (
    KEEP_EMPHASIS_METADATA_KEY,
    resume_plain_segments,
)
from src.core.epub.body_serializer import find_body
from src.core.epub.book_loop import _precount_chunks_plain_text
from src.prompts.prompts import PLAIN_TEXT_KEEP_EMPHASIS_OPTION
from tests.epub_engine_helpers import make_settings
from tests.job_helpers import make_job

XHTML_NS = "http://www.w3.org/1999/xhtml"
KEEP = {'plain_text_mode': True, PLAIN_TEXT_KEEP_EMPHASIS_OPTION: True}


# ---------------------------------------------------------------------------
# strip_hallucinated_markup
# ---------------------------------------------------------------------------
class TestStripHallucinatedMarkup:
    def test_default_path_unchanged(self):
        source = "He said hello."
        assert plain_pipeline.strip_hallucinated_markup(
            "Il a dit <i>bonjour</i><sup>1</sup>.", source) == "Il a dit bonjour1."

    def test_default_path_disarmed_by_markers_in_source(self):
        # Without the option, any '<' in the source disarms the strip.
        source = "He said <i>hello</i>."
        out = "Il a dit <i>bonjour</i><sup>1</sup>."
        assert plain_pipeline.strip_hallucinated_markup(out, source) == out

    def test_keeps_bare_markers_and_strips_other_tags(self):
        source = "He said <i>hello</i> and <em>left</em>."
        out = "Il a dit <i>bonjour</i><sup>1</sup> et <b>est</b> <em>parti</em>."
        assert plain_pipeline.strip_hallucinated_markup(out, source, keep_emphasis=True) == (
            "Il a dit <i>bonjour</i>1 et est <em>parti</em>.")

    def test_markers_with_attributes_are_not_markers(self):
        source = "A <i>word</i>."
        out = 'Un <i class="x">mot</i>.'
        assert plain_pipeline.strip_hallucinated_markup(out, source, keep_emphasis=True) == (
            "Un mot</i>.")

    def test_real_angle_bracket_in_source_still_disarms(self):
        source = "if a < b then <i>swap</i>"
        out = "si a < b alors <i>échange</i><sup>2</sup>"
        assert plain_pipeline.strip_hallucinated_markup(out, source, keep_emphasis=True) == out

    def test_no_tag_in_translation_is_untouched(self):
        assert plain_pipeline.strip_hallucinated_markup(
            "rien", "<i>nothing</i>", keep_emphasis=True) == "rien"


# ---------------------------------------------------------------------------
# Oversized paragraphs split by sentence
# ---------------------------------------------------------------------------
def test_sentence_split_keeps_closer_with_its_sentence():
    paragraph = ("He said <i>Hello there. How are you?</i> Then he left. "
                 "<em>Fine.</em> End of it all here, finally.")
    pieces = plain_pipeline.build_plain_segments([paragraph], 20, keep_emphasis=True)
    assert len(pieces) > 1 and all(p['partial'] for p in pieces)
    joined = " ".join(p['text'] for p in pieces)
    assert "you?</i> Then" in joined
    assert "Fine.</em> End" in joined
    assert not any(p['text'].startswith("</") for p in pieces)
    # Nothing but whitespace moved.
    assert joined.split() == paragraph.split()


def test_sentence_split_default_path_unchanged():
    paragraph = "One sentence here. </i> Two more words here. Three is the last one."
    assert (plain_pipeline.build_plain_segments([paragraph], 10)
            == plain_pipeline.build_plain_segments([paragraph], 10, keep_emphasis=False))


# ---------------------------------------------------------------------------
# Scoping: only the EPUB plain path ever sees the option
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("file_type", ["docx", "txt", "pdf", "srt", "md", "fb2", "pgn"])
def test_engine_options_drop_the_key_outside_epub(file_type):
    job = make_job(input_path=f"book.{file_type}", file_type=file_type, prompt_options=dict(KEEP))
    options = job.engine_prompt_options()
    assert PLAIN_TEXT_KEEP_EMPHASIS_OPTION not in options
    assert options['plain_text_mode'] is True
    # The job's own options are not mutated.
    assert job.prompt_options[PLAIN_TEXT_KEEP_EMPHASIS_OPTION] is True


def test_engine_options_keep_the_key_for_epub_plain_mode_only():
    epub = make_job(input_path="book.epub", file_type="epub", prompt_options=dict(KEEP))
    assert epub.engine_prompt_options()[PLAIN_TEXT_KEEP_EMPHASIS_OPTION] is True

    placeholder = make_job(input_path="book.epub", file_type="epub",
                           prompt_options={PLAIN_TEXT_KEEP_EMPHASIS_OPTION: True})
    assert PLAIN_TEXT_KEEP_EMPHASIS_OPTION not in placeholder.engine_prompt_options()


def test_docx_prompt_is_byte_identical_with_the_option_sent():
    from src.prompts.prompts import generate_translation_prompt

    docx = make_job(input_path="a.docx", file_type="docx", prompt_options=dict(KEEP))
    args = ("Some text.", "", "English", "French")
    with_key = generate_translation_prompt(
        *args, has_placeholders=False, prompt_options=docx.engine_prompt_options())
    without = generate_translation_prompt(
        *args, has_placeholders=False, prompt_options={'plain_text_mode': True})
    assert with_key == without


@pytest.mark.asyncio
async def test_docx_plain_path_never_turns_emphasis_on(monkeypatch, tmp_path):
    """Even if the key leaked into its options, DOCX post-processing stays default."""
    from docx import Document

    from src.core.docx.docx_translation_adapter import DocxTranslationAdapter

    doc = Document()
    doc.add_paragraph("A paragraph.")
    source_path = str(tmp_path / "sample.docx")
    doc.save(source_path)

    recorded = []

    async def spy(**kwargs):
        recorded.append(kwargs)
        return list(kwargs['paragraphs']), plain_pipeline.TranslationMetrics(), False

    monkeypatch.setattr(plain_pipeline, "translate_paragraphs_plain", spy)
    await DocxTranslationAdapter()._translate_plain_text(
        source_path=source_path, source_language="English", target_language="French",
        model_name="m", llm_client=object(), max_tokens_per_chunk=200, log_callback=None,
        context_manager=None, prompt_options=dict(KEEP), stats_callback=None,
        check_interruption_callback=None, parallel_workers=1, file_href="sample.docx",
    )
    assert recorded and recorded[0].get('keep_emphasis', False) is False


# ---------------------------------------------------------------------------
# Checkpoint flag
# ---------------------------------------------------------------------------
class _State:
    def __init__(self, metadata):
        self.doc_metadata = {'plain_text_mode': True, 'paragraph_count': 2, **metadata}
        self.chunks = [{'indices': [0, 1], 'text': "a\n\nb", 'partial': False}]
        self.translated_chunks = ["A\n\nB"]


@pytest.mark.parametrize("stored, current, reused", [
    (None, False, True),     # written before the feature == off
    (None, True, False),
    (False, False, True),
    (False, True, False),
    (True, True, True),
    (True, False, False),
])
def test_checkpoint_flag_must_match(stored, current, reused):
    metadata = {} if stored is None else {KEEP_EMPHASIS_METADATA_KEY: stored}
    logs = []
    segments, prefix = resume_plain_segments(
        _State(metadata), 2, lambda key, msg: logs.append(key), keep_emphasis=current)
    assert (segments is not None) is reused
    assert (prefix is not None) is reused
    assert ("plain_text_resume_ignored" in logs) is (not reused)


def test_checkpoint_hook_records_the_flag(tmp_path):
    from src.core.common.plain_text_checkpoint import build_plain_checkpoint_hook

    saved = []

    class _Manager:
        def save_xhtml_partial_state(self, tid, href, state):
            saved.append(state)

    for flag in (False, True):
        hook = build_plain_checkpoint_hook(
            checkpoint_manager=_Manager(), translation_id="t", file_href="c.xhtml",
            source_language="English", target_language="French", model_name="m",
            max_tokens_per_chunk=100, max_retries=1, paragraph_count=1, keep_emphasis=flag)
        hook([{'indices': [0], 'text': "x", 'partial': False}], ["X"], 1, {})
    assert [s.doc_metadata[KEEP_EMPHASIS_METADATA_KEY] for s in saved] == [False, True]


# ---------------------------------------------------------------------------
# End to end through translate_plain_text
# ---------------------------------------------------------------------------
BODY = (
    f'<html xmlns="{XHTML_NS}"><body>'
    "<p>He was <i>very</i> tired.</p>"
    "<p>She read <em>Dune</em> again.</p>"
    "<p><i>The whole line is italic.</i></p>"
    "<p>Plain line with no emphasis.</p>"
    "</body></html>"
)
TRANSLATIONS = {
    "He was <i>very</i> tired.": "Il était <i>très</i> fatigué.",
    "She read <em>Dune</em> again.": "Elle a relu <em>Dune</em>.",
    "The whole line is italic.": "Toute la ligne est en italique.",
    "Plain line with no emphasis.": "Ligne simple sans emphase.",
}


def _install_llm(monkeypatch, translate):
    seen = []

    async def fake_request(main_content, *args, **kwargs):
        seen.append(main_content)
        return "\n\n".join(translate(p) for p in main_content.split("\n\n"))

    monkeypatch.setattr(plain_pipeline, "generate_translation_request", fake_request)
    return seen


async def _run(prompt_options, max_tokens=2000):
    from src.core.epub.plain_text_document import translate_plain_text

    root = etree.fromstring(BODY)
    logs, totals = [], {}
    settings = make_settings(
        prompt_options=prompt_options, max_tokens_per_chunk=max_tokens,
        log_callback=lambda key, msg: logs.append((key, msg)), emphasis_totals=totals)
    success, _ = await translate_plain_text(find_body(root), settings)
    assert success
    paragraphs = [etree.tostring(p, encoding="unicode", with_tail=False)
                  for p in find_body(root)]
    return paragraphs, logs, totals


def _inner(p_markup):
    return p_markup[p_markup.index(">") + 1:p_markup.rindex("<")]


@pytest.mark.asyncio
async def test_end_to_end_markers_kept(monkeypatch):
    seen = _install_llm(monkeypatch, lambda p: TRANSLATIONS[p])
    paragraphs, logs, totals = await _run(dict(KEEP))

    # The whole-line span travels without markers.
    assert seen == ["\n\n".join(TRANSLATIONS)]
    inner = [_inner(p) for p in paragraphs]
    assert inner[0] == "Il était <i>très</i> fatigué."
    assert inner[1] == "Elle a relu <em>Dune</em>."
    assert inner[2] == "<i>Toute la ligne est en italique.</i>"
    assert inner[3] == "Ligne simple sans emphase."
    assert totals == {"source_spans": 3, "kept_spans": 3, "rejected_paragraphs": 0}
    assert ("plain_text_emphasis",
            "Italics: 3/3 spans kept, 0 paragraph(s) fell back to plain") in logs


@pytest.mark.asyncio
async def test_end_to_end_broken_markers_fall_back_to_plain(monkeypatch):
    broken = {**TRANSLATIONS, "He was <i>very</i> tired.": "Il était <i>très fatigué."}
    _install_llm(monkeypatch, lambda p: broken[p])
    paragraphs, logs, totals = await _run(dict(KEEP))

    assert _inner(paragraphs[0]) == "Il était très fatigué."
    assert _inner(paragraphs[1]) == "Elle a relu <em>Dune</em>."
    assert totals == {"source_spans": 3, "kept_spans": 2, "rejected_paragraphs": 1}


@pytest.mark.asyncio
async def test_end_to_end_hallucinated_tags_are_stripped(monkeypatch):
    noisy = {
        **TRANSLATIONS,
        "Plain line with no emphasis.": "Ligne <b>simple</b> sans emphase<sup>1</sup>.",
        "She read <em>Dune</em> again.": "Elle a relu <em>Dune</em> <sup>2</sup>.",
    }
    _install_llm(monkeypatch, lambda p: noisy[p])
    paragraphs, _, totals = await _run(dict(KEEP))

    assert _inner(paragraphs[1]) == "Elle a relu <em>Dune</em> 2."
    assert _inner(paragraphs[3]) == "Ligne simple sans emphase1."
    assert totals["kept_spans"] == 3


@pytest.mark.asyncio
async def test_end_to_end_option_off_is_plain(monkeypatch):
    seen = _install_llm(monkeypatch, lambda p: "T " + p)
    paragraphs, logs, totals = await _run({'plain_text_mode': True})

    assert "<i>" not in seen[0] and "<em>" not in seen[0]
    assert all("<i>" not in p and "<em>" not in p for p in paragraphs)
    assert totals == {}
    assert not any(key == "plain_text_emphasis" for key, _ in logs)


# ---------------------------------------------------------------------------
# Precount consistency
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [False, True])
async def test_precount_matches_segments_sent(monkeypatch, keep):
    long_italic = " ".join(f"Sentence <i>number {k}</i> goes on." for k in range(12))
    body = (f'<html xmlns="{XHTML_NS}"><body>'
            + "".join(f"<p>Short <em>para</em> {k}.</p>" for k in range(15))
            + f"<p>{long_italic}</p></body></html>")
    max_tokens = 40

    counted = _precount_chunks_plain_text(etree.fromstring(body), max_tokens, keep_emphasis=keep)

    from src.core.epub.plain_text_document import translate_plain_text
    seen = _install_llm(monkeypatch, lambda p: p)
    options = dict(KEEP) if keep else {'plain_text_mode': True}
    root = etree.fromstring(body)
    await translate_plain_text(
        find_body(root), make_settings(prompt_options=options, max_tokens_per_chunk=max_tokens))

    assert counted == len(seen) > 1
    assert any("<i>" in text for text in seen) is keep


# ---------------------------------------------------------------------------
# Glossary and residue scans ignore the markers
# ---------------------------------------------------------------------------
def test_glossary_matches_a_term_split_by_markers():
    from src.core.translator import _build_chunk_glossary_block

    options = {'glossary_terms': {"Lord Kar": "Seigneur Kar"}}
    chunk = "Then Lord <i>Kar</i> spoke."
    assert _build_chunk_glossary_block(chunk, options) == ""
    block = _build_chunk_glossary_block(chunk, {**options, **KEEP})
    assert "Seigneur Kar" in block


@pytest.mark.asyncio
async def test_markers_do_not_trigger_a_residue_retry(monkeypatch):
    import src.core.translator as translator

    calls = []

    async def fake_request(*args, **kwargs):
        calls.append(args)
        return None

    monkeypatch.setattr(translator, "generate_translation_request", fake_request)
    source = "<i>Dune</i>, <i>Emma</i>, <i>Ulysses</i>, <i>Kim</i>, <i>Ivanhoe</i>"
    answer = "<i>デューン</i>、<i>エマ</i>、<i>ユリシーズ</i>、<i>キム</i>、<i>アイヴァンホー</i>"
    assert remove_markers(answer) == "デューン、エマ、ユリシーズ、キム、アイヴァンホー"

    common = dict(main_content=source, source_language="English", target_language="Japanese",
                  log_callback=None, has_placeholders=False, placeholder_format=None,
                  on_raw_fallback=None)
    # Without the option the markers read as leftover English words.
    await translator._retry_on_source_residue(answer, prompt_options={}, **common)
    assert len(calls) == 1
    calls.clear()
    kept, retried = await translator._retry_on_source_residue(
        answer, prompt_options=dict(KEEP), **common)
    assert calls == [] and kept == answer and retried is False


# ---------------------------------------------------------------------------
# Book-level summary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("keep, totals, expected", [
    (True, {"source_spans": 7, "kept_spans": 5, "rejected_paragraphs": 1},
     "Italics kept: 5/7 spans (1 paragraphs fell back to plain)"),
    (True, {}, None),
    (False, {"source_spans": 7, "kept_spans": 5, "rejected_paragraphs": 1}, None),
])
def test_book_summary_line(keep, totals, expected):
    from types import SimpleNamespace

    from src.core.epub.book_loop import _log_emphasis_summary

    logs = []
    run = SimpleNamespace(plain_text_keep_emphasis=keep, emphasis_totals=totals,
                          log=lambda key, msg: logs.append((key, msg)))
    _log_emphasis_summary(run)
    assert logs == ([] if expected is None else [("plain_text_emphasis_summary", expected)])
