"""
Tests for Markdown support (issue #243).

Only the inline text of headings, paragraphs, list items, blockquotes,
footnotes and table cells may change. Everything else (front matter, code,
markers, indentation, pipes, rules, URLs, HTML) must come back byte for
byte, and protected spans must never reach the LLM.
"""

import re
from pathlib import Path

import pytest

import src.core.llm.factory as llm_factory_module
import src.core.translator as translator_module
from src.api.blueprints import sample_routes
from src.core.adapters import MarkdownAdapter
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.translate_file import get_file_type_from_path
from src.core.markdown import (
    decode_chunk, group_chunks, plan_chunk, rebuild, segment_markdown, segments_of,
)
from src.persistence.checkpoint_manager import CheckpointManager
from src.utils.file_detector import detect_file_type
from tests.job_helpers import make_ctx, make_job

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "markdown" / "sample.md"
PLACEHOLDER = re.compile(r'(\[id\d+\])')

# Bytes that must never be sent to the LLM.
NEVER_SENT = [
    "title: A field guide",
    "This comment looks like prose",
    "Watering {plant}",
    "the cat sat on the mat",
    "Indented code block that looks like a sentence",
    "pip install garden[extras]",
    "https://example.com/manual",
    "Manual title",
    "https://example.org/autolink",
    "https://example.net/bare/path",
    "images/robin.png",
    "trim | sort",
    "HTML block content",
    "A comment that must stay",
    "E = mc^2",
    "https://example.com/guide",
    "<b>",
]


def _source() -> str:
    # LF-normalized, whatever line endings the checkout gave the fixture
    # (CRLF is covered by its own tests).
    return FIXTURE.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _fake_translate(content: str) -> str:
    """Upper-case every piece of text outside the placeholders."""
    return ''.join(part if PLACEHOLDER.fullmatch(part) else part.upper()
                   for part in PLACEHOLDER.split(content))


def _raw_pieces(text: str):
    return [p for p in segment_markdown(text) if isinstance(p, str)]


async def _adapter(tmp_path, text=None, raw=None, **config):
    src = tmp_path / "in.md"
    if raw is not None:
        src.write_bytes(raw)
    else:
        src.write_bytes((text if text is not None else _source()).encode("utf-8"))
    adapter = MarkdownAdapter(str(src), str(tmp_path / "out.md"), config)
    assert await adapter.prepare_for_translation()
    return adapter


async def _translate_all(adapter, translate):
    for unit in adapter.get_translation_units():
        assert await adapter.save_unit_translation(unit.unit_id, translate(unit.content))
    return (await adapter.reconstruct_output()).decode("utf-8")


class TestSegmentation:
    def test_rebuild_without_translation_is_byte_identical(self):
        text = _source()
        assert rebuild(segment_markdown(text)) == text

    def test_raw_structure_is_kept(self):
        raw = ''.join(_raw_pieces(_source()))
        for kept in ("---\ntitle: A field guide to the garden\n", "# ", " #\n",
                     "\n============\n", "\n  1. ", "\n     ", "\n- [ ] ",
                     "\n> - ", "\n>\n> > ", "```python\n", "~~~\n",
                     "|:------|:------:|--------:|", " | `trim | sort` |",
                     "<div align=\"center\">", "$$\nE = mc^2\n$$", "***",
                     "[guide]: https://example.com/guide \"Guide\"", "[^1]: "):
            assert kept in raw, kept

    def test_segments_hold_only_prose(self):
        segments = segments_of(segment_markdown(_source()))
        prose = [s.prose for s in segments]
        assert "A Field Guide to the Garden" in prose
        assert "Setext Title" in prose
        assert "Winter is cold" in prose
        assert "The footnote text is translated too." in prose
        # Cells without letters (numbers, dashes) are not segments.
        assert "42" not in prose and "7" not in prose
        joined = '\n'.join(prose)
        for secret in NEVER_SENT:
            assert secret not in joined, secret

    def test_link_text_is_prose_and_destination_is_protected(self):
        [seg] = segments_of(segment_markdown(
            'See [the manual](https://x.io/a_(b) "T") and ![alt text](i.png).\n'))
        assert seg.tokens == [
            (False, 'See '), (True, '['), (False, 'the manual'),
            (True, '](https://x.io/a_(b) "T")'), (False, ' and '), (True, '!['),
            (False, 'alt text'), (True, '](i.png)'), (False, '.'),
        ]

    def test_brackets_inside_inline_code_are_not_a_link(self):
        [seg] = segments_of(segment_markdown('Call `f[x](y)` here.\n'))
        assert seg.tokens == [(False, 'Call '), (True, '`f[x](y)`'), (False, ' here.')]

    def test_unclosed_fence_keeps_the_rest_raw(self):
        text = "Intro text.\n\n```\ncode that never ends\nmore code\n"
        pieces = segment_markdown(text)
        assert [s.prose for s in segments_of(pieces)] == ["Intro text."]
        assert rebuild(pieces) == text

    def test_crlf_and_tabs_round_trip(self):
        text = _source().replace("\n", "\r\n")
        assert rebuild(segment_markdown(text)) == text


class TestChunking:
    def test_tables_stay_in_one_chunk(self):
        segments = segments_of(segment_markdown(_source()))
        chunks = group_chunks(segments, 60, lambda t: len(t.split()))
        assert len(chunks) > 1
        table_group = next(s.group for s in segments if s.prose == "Plant")
        holders = {i for i, chunk in enumerate(chunks)
                   for idx in chunk if segments[idx].group == table_group}
        assert len(holders) == 1

    def test_large_table_is_split_at_row_boundaries_only(self):
        rows = '\n'.join(f"| Name number {i} | Value number {i} |" for i in range(10))
        text = "| Head one | Head two |\n|---|---|\n" + rows + "\n"
        segments = segments_of(segment_markdown(text))
        chunks = group_chunks(segments, 30, lambda t: len(t.split()) + 1)
        assert len(chunks) > 1
        for chunk in chunks:
            assert segments[chunk[0]].row_start

    def test_every_segment_is_in_exactly_one_chunk(self):
        segments = segments_of(segment_markdown(_source()))
        chunks = group_chunks(segments, 25, lambda t: len(t.split()))
        assert [i for chunk in chunks for i in chunk] == list(range(len(segments)))


class TestCodec:
    def test_identity_answer_decodes_to_source(self):
        segments = segments_of(segment_markdown(_source()))
        plan, rendered = plan_chunk(segments, list(range(len(segments))))
        translations, failed = decode_chunk(plan, rendered)
        assert failed == []
        assert translations == {i: s.text for i, s in enumerate(segments)}

    def test_dropped_inline_placeholder_fails_only_its_segment(self):
        text = "First paragraph here.\n\nSecond with `code` inside.\n\nThird paragraph.\n"
        segments = segments_of(segment_markdown(text))
        plan, rendered = plan_chunk(segments, [0, 1, 2])
        assert rendered == ("[id0]First paragraph here.\n\n[id1]Second with [id2] inside."
                            "\n\n[id3]Third paragraph.")
        answer = "[id0]Premier paragraphe.\n\n[id1]Second sans code.\n\n[id3]Troisième."
        translations, failed = decode_chunk(plan, answer)
        assert failed == [1]
        assert translations == {0: "Premier paragraphe.", 2: "Troisième."}

    def test_lost_boundary_rejects_the_merged_segment(self):
        text = "Alpha one.\n\nBeta two.\n"
        segments = segments_of(segment_markdown(text))
        plan, _ = plan_chunk(segments, [0, 1])
        translations, failed = decode_chunk(plan, "[id0]Alpha un. Beta deux.")
        assert failed == [0, 1]
        assert translations == {}

    def test_added_line_breaks_keep_the_structure(self):
        text = "# Short title\n\n> Quoted line one\n> line two.\n\n| A cell | B cell |\n|---|---|\n"
        pieces = segment_markdown(text)
        segs = segments_of(pieces)
        out = rebuild(pieces, {0: "Short\ntitle", 1: "One\n\n\nTwo\nThree", 2: "A\ncell"})
        assert out == ("# Short title\n\n> One\n> Two\n> Three\n\n"
                       "| A cell | B cell |\n|---|---|\n")
        assert len(segs) == 4


class TestAdapter:
    @pytest.mark.asyncio
    async def test_identity_round_trip_is_byte_identical(self, tmp_path):
        adapter = await _adapter(tmp_path)
        assert await _translate_all(adapter, lambda c: c) == _source()

    @pytest.mark.asyncio
    async def test_crlf_and_bom_round_trip_is_byte_identical(self, tmp_path):
        raw = b'\xef\xbb\xbf' + _source().replace("\n", "\r\n").encode("utf-8")
        adapter = await _adapter(tmp_path, raw=raw)
        for unit in adapter.get_translation_units():
            await adapter.save_unit_translation(unit.unit_id, unit.content)
        assert await adapter.reconstruct_output() == raw

    @pytest.mark.asyncio
    async def test_translation_changes_only_prose(self, tmp_path):
        source = _source()
        adapter = await _adapter(tmp_path, max_tokens_per_chunk=60)
        assert len(adapter.get_translation_units()) > 1
        output = await _translate_all(adapter, _fake_translate)

        # Same skeleton: every raw piece identical, in order.
        assert _raw_pieces(output) == _raw_pieces(source)
        for kept in NEVER_SENT:
            assert kept in output, kept
        assert "# A FIELD GUIDE TO THE GARDEN #\n" in output
        assert "[THE USER MANUAL](https://example.com/manual?lang=en \"Manual title\")" in output
        assert "![A ROBIN ON A BRANCH](images/robin.png)" in output
        assert "  2. CUT THE DRY BRANCHES, NOT THE GREEN ONES.\n     A CONTINUATION" in output
        assert "> A QUOTE ABOUT GARDENS: PATIENCE IS THE\n> FIRST TOOL" in output
        assert "| ROSE BUSH | SPRING | `trim | sort` |" in output
        assert "| 7 | WINTER IS COLD | -- |" in output
        assert "LITERAL [id7] TEXT STAYS." in output
        assert "A HARD BREAK ENDS THIS LINE  \nAND THE PARAGRAPH" in output

    @pytest.mark.asyncio
    async def test_protected_spans_are_never_sent(self, tmp_path):
        adapter = await _adapter(tmp_path, max_tokens_per_chunk=60)
        sent = '\n'.join(u.content for u in adapter.get_translation_units())
        for secret in NEVER_SENT:
            assert secret not in sent, secret

    @pytest.mark.asyncio
    async def test_validation_reports_broken_placeholders(self, tmp_path):
        adapter = await _adapter(tmp_path)
        unit = adapter.get_translation_units()[0]
        assert adapter.validate_unit_translation(unit.unit_id, unit.content) is None
        broken = PLACEHOLDER.sub('', unit.content, count=3)
        assert "placeholders" in adapter.validate_unit_translation(unit.unit_id, broken)

    @pytest.mark.asyncio
    async def test_bilingual_keeps_original_and_structure(self, tmp_path):
        adapter = await _adapter(tmp_path, text="## Hello world\n\n| Cell one |\n|---|\n")
        for unit in adapter.get_translation_units():
            await adapter.save_unit_translation(unit.unit_id, _fake_translate(unit.content))
        out = (await adapter.reconstruct_output(bilingual=True)).decode("utf-8")
        assert out == "## Hello world / HELLO WORLD\n\n| Cell one / CELL ONE |\n|---|\n"

    @pytest.mark.asyncio
    async def test_resume_restores_matching_chunks_only(self, tmp_path):
        adapter = await _adapter(tmp_path, max_tokens_per_chunk=60)
        units = adapter.get_translation_units()
        checkpoint = {
            'resume_from_index': 2,
            'chunks': [
                {'status': 'completed', 'chunk_data': {'chunk_index': 0},
                 'original_text': units[0].content,
                 'translated_text': _fake_translate(units[0].content)},
                {'status': 'completed', 'chunk_data': {'chunk_index': 1},
                 'original_text': "stale source from another chunking",
                 'translated_text': _fake_translate(units[1].content)},
            ],
        }
        assert await adapter.resume_from_checkpoint(checkpoint) == 2
        assert set(adapter.translations) == set(adapter.plans[0].segment_indices)


class TestRouting:
    def test_sample_routes_load_markdown_as_raw_text(self):
        assert FIXTURE.is_file(), f"Missing Markdown fixture: {FIXTURE}"
        file_type = detect_file_type(str(FIXTURE))
        assert file_type == "md"

        units = sample_routes._load_source_units(str(FIXTURE), file_type)

        assert units
        assert "Welcome to the garden." in units[0]["main_content"]

    @pytest.mark.parametrize("name", ["notes.md", "README.MARKDOWN"])
    def test_markdown_extensions_are_detected(self, name):
        assert detect_file_type(name) == "md"
        assert get_file_type_from_path(name) == "md"

    def test_job_config_from_cli_args_uses_md(self):
        import argparse
        from src.core.job import JobConfig
        args = argparse.Namespace(
            input="book.md", output=None, source_lang="English",
            target_lang="French", provider="ollama", model="m",
        )
        job = JobConfig.from_cli_args(args)
        assert job.file_type == "md"
        assert job.output_path == "book (French).md"


class FakeLLMClient:
    def __init__(self, **kwargs):
        pass

    async def close(self):
        pass


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


@pytest.mark.asyncio
async def test_end_to_end_uses_placeholder_mode_without_extra_instructions(tmp_path, cm, monkeypatch):
    calls = []

    async def fake_request(main_content=None, **kwargs):
        calls.append({'content': main_content, **kwargs})
        return _fake_translate(main_content)

    monkeypatch.setattr(llm_factory_module, "create_llm_provider",
                        lambda *args, **kwargs: FakeLLMClient())
    monkeypatch.setattr(translator_module, "generate_translation_request", fake_request)

    src = tmp_path / "in.md"
    src.write_bytes(_source().encode("utf-8"))
    adapter = MarkdownAdapter(str(src), str(tmp_path / "out.md"), {'max_tokens_per_chunk': 80})
    translator = GenericTranslator(adapter=adapter, checkpoint_manager=cm, translation_id="job")
    success = await translator.translate(
        make_job(source_language="English", target_language="French", model="fake-model",
                 provider="ollama"),
        make_ctx(log_callback=lambda *args, **kwargs: None),
    )
    assert success is True
    assert calls
    for call in calls:
        assert call['has_placeholders'] is True
        assert call['placeholder_format'] == ("[id", "]")
        # No Markdown-specific instruction is injected into the prompt.
        assert not (call.get('prompt_options') or {}).get('custom_instructions')
        # The previous-chunk context carries no stale placeholder numbers.
        assert not PLACEHOLDER.search(call.get('previous_translation_context') or '')
    output = (tmp_path / "out.md").read_text(encoding="utf-8")
    assert _raw_pieces(output) == _raw_pieces(_source())
    assert cm.get_job("job")["status"] == "completed"
