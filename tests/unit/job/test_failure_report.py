"""The opt-in failure report written next to the output (issue #232)."""

import argparse
import re
import asyncio
import zipfile
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.core.job import JobConfig, JobResult, JobStatus
from src.core.job.failure_report import (
    FileIssues,
    UnitIssue,
    collect_epub_issues,
    collect_unit_issues,
    epub_chapter_info,
    make_snippet,
    render_report,
    report_path_for,
    write_failure_report,
)
from src.core.job.runner import JobRunner, JobSinks
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager

FIXED_TIME = datetime(2026, 11, 24, 15, 2)


def _squash(text):
    """The report with runs of spaces collapsed (column widths are cosmetic)."""
    return re.sub(r' {2,}', ' ', text)


def _state(statuses, texts, placeholder_format=("[[", "]]")):
    return SimpleNamespace(chunk_statuses=list(statuses),
                           chunks=[{'text': t} for t in texts],
                           placeholder_format=placeholder_format)


def _epub(path, chapters):
    """A minimal EPUB: ``chapters`` is a list of (href, body html)."""
    manifest = "".join(
        f'<item id="c{i}" href="{href}" media-type="application/xhtml+xml"/>'
        for i, (href, _) in enumerate(chapters))
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('mimetype', 'application/epub+zip')
        z.writestr('META-INF/container.xml',
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr('OEBPS/content.opf',
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
                   f'version="3.0"><manifest>{manifest}</manifest><spine>{spine}</spine></package>')
        for href, body in chapters:
            z.writestr(f'OEBPS/{href}',
                       '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
                       f'<head><title>Head title</title></head><body>{body}</body></html>')
    return str(path)


# -- helpers -----------------------------------------------------------------


def test_report_path_is_named_after_the_output():
    assert report_path_for("/out/Book (French).epub") == "/out/Book (French).report.txt"
    assert report_path_for("subs.srt") == "subs.report.txt"


def test_snippet_drops_placeholders_and_truncates():
    assert make_snippet("[[0]]Hello[[1]]   world\n again", ("[[", "]]")) == "Hello world again"
    long = make_snippet("word " * 50, length=20)
    assert len(long) == 20 and long.endswith("…")
    assert make_snippet(None) == ""


# -- EPUB --------------------------------------------------------------------


def test_epub_issues_use_the_partial_state_statuses():
    stats = {'unfinished_files': {'Text/ch2.xhtml': [1, 3]},
             'degraded_files': {'Text/ch2.xhtml': [2], 'Text/ch5.xhtml': [0]}}
    states = {'Text/ch2.xhtml': _state(
        ['translated', 'untranslated', 'token_aligned', 'pending'],
        ['a', '[[0]]It was a dark[[1]] night', 'Some [[2]]italic[[3]] text', 'Later on'])}

    entries = collect_epub_issues(stats, states.get)

    assert [e.href for e in entries] == ['Text/ch2.xhtml', 'Text/ch5.xhtml']
    ch2, ch5 = entries
    assert (ch2.untranslated, ch2.not_reached, ch2.token_aligned) == ([1], [3], [2])
    assert ch2.total_chunks == 4
    assert ch2.snippets == {1: "It was a dark night", 2: "Some italic text", 3: "Later on"}
    # No state on disk: the stats maps are reported as they are.
    assert (ch5.untranslated, ch5.token_aligned, ch5.total_chunks) == ([], [0], None)


def test_epub_issues_ignore_malformed_maps_and_a_failing_loader():
    def broken(_href):
        raise OSError("unreadable")

    stats = {'unfinished_files': {'a.xhtml': [2, 'x', -1, 2], 'b.xhtml': []},
             'degraded_files': 'not a map'}
    entries = collect_epub_issues(stats, broken)
    assert [(e.href, e.untranslated) for e in entries] == [('a.xhtml', [2])]
    assert collect_epub_issues({}, None) == []


def test_epub_chapter_info_gives_reading_order_and_heading(tmp_path):
    path = _epub(tmp_path / "book.epub", [
        ("Text/cover.xhtml", "<p>no heading</p>"),
        ("Text/ch1.xhtml", "<h2>Chapter <i>One</i></h2><p>x</p>"),
    ])
    info, total = epub_chapter_info(path)
    assert total == 2
    assert info["Text/cover.xhtml"] == (1, "Head title")
    assert info["Text/ch1.xhtml"] == (2, "Chapter One")
    assert epub_chapter_info(str(tmp_path / "missing.epub")) == ({}, 0)


def test_render_epub_report_lists_files_in_reading_order():
    files = [
        FileIssues(href="Text/ch9.xhtml", token_aligned=[0], position=9, title="Nine"),
        FileIssues(href="Text/ch2.xhtml", untranslated=[1], not_reached=[3], token_aligned=[2],
                   total_chunks=4, snippets={1: "It was a dark night"}, position=2,
                   title="Chapter Two"),
    ]
    text = render_report(output_name="Book (French).epub", source_name="Book.epub",
                         status="partial", model="ollama / qwen3:14b", file_type="epub",
                         epub_files=files, spine_total=12, generated_at=FIXED_TIME)

    assert "Output:    Book (French).epub" in text
    assert "Status:    partial" in text
    assert "Generated: 2026-11-24 15:02" in text
    assert ("Summary: 1 untranslated chunk(s), 1 chunk(s) not reached, "
            "2 chunk(s) with approximate tag placement in 2 file(s).") in text
    assert text.index('[2/12] Text/ch2.xhtml  "Chapter Two"') < text.index('[9/12] Text/ch9.xhtml')
    assert 'UNTRANSLATED chunk 2/4 "It was a dark night"' in _squash(text)
    assert "NOT REACHED   chunk 4/4" in text
    assert "APPROX. TAGS  chunk 3/4" in text
    # Rows of a file are in chunk order, whatever their kind.
    block = text[text.index("[2/12]"):text.index("[9/12]")]
    assert block.index("chunk 2/4") < block.index("chunk 3/4") < block.index("chunk 4/4")


def test_render_epub_report_without_issues_says_so():
    text = render_report(output_name="b.epub", file_type="epub", epub_files=[],
                         generated_at=FIXED_TIME)
    assert "No untranslated chunks and no chunks with approximate tag placement" in text
    assert "Legend" not in text


def test_write_epub_report_reads_states_and_the_output_book(tmp_path):
    output = _epub(tmp_path / "Book (French).epub", [
        ("Text/ch1.xhtml", "<h1>One</h1>"), ("Text/ch2.xhtml", "<h1>Two</h1>")])
    loaded = []

    class Checkpoints:
        def load_xhtml_partial_state(self, translation_id, href):
            loaded.append((translation_id, href))
            return _state(['untranslated', 'translated'], ['Source sentence', 'b'])

    job = JobConfig(file_type='epub', input_filename='Book.epub', output_path=output,
                    provider=ProviderConfig(provider='ollama', model='qwen3:14b'))
    path = write_failure_report(job, 'partial', output,
                                {'unfinished_files': {'Text/ch2.xhtml': [0]}},
                                Checkpoints(), 'job1')

    assert path == str(tmp_path / "Book (French).report.txt")
    text = open(path, encoding='utf-8').read()
    assert loaded == [('job1', 'Text/ch2.xhtml')]
    assert '[2/2] Text/ch2.xhtml  "Two"' in text
    assert 'UNTRANSLATED chunk 1/2 "Source sentence"' in _squash(text)
    assert "Model:     ollama / qwen3:14b" in text


# -- other formats -----------------------------------------------------------


def test_unit_issues_keep_only_failed_chunks():
    rows = [{'chunk_index': 0, 'status': 'completed', 'original_text': 'a'},
            {'chunk_index': 4, 'status': 'failed', 'original_text': 'Fourth  line'},
            {'chunk_index': 2, 'status': 'failed', 'original_text': None}]
    assert collect_unit_issues(rows) == [UnitIssue(2, ''), UnitIssue(4, 'Fourth line')]


def test_render_unit_report():
    text = render_report(output_name="b.txt", file_type="txt",
                         failed_units=[UnitIssue(4, "Fourth line")], raw_fallback_count=2,
                         generated_at=FIXED_TIME)
    assert "Summary: 1 chunk(s) failed" in text
    assert 'FAILED chunk 5 "Fourth line"' in _squash(text)
    assert "2 chunk(s) came back from the model without a usable translation" in text

    clean = render_report(output_name="b.txt", file_type="srt", generated_at=FIXED_TIME)
    assert "No failed chunks were recorded for this job." in clean


# -- runner and configuration --------------------------------------------------


class _Sinks(JobSinks):
    def __init__(self):
        self.logs = []

    def log(self, key, message="", data=None):
        self.logs.append((key, message))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "book.txt"
    source.write_text("one\n\ntwo\n\nthree\n", encoding="utf-8")
    manager = CheckpointManager(db_path=str(tmp_path / "data" / "jobs.db"))
    return tmp_path, source, manager


def _run(tmp_path, source, manager, failure_report):
    async def translate_file(job, ctx):
        with open(job.output_path, "w", encoding="utf-8") as f:
            f.write("un\n\ntwo\n\ntrois\n")
        ctx.checkpoint_manager.save_checkpoint(
            translation_id=ctx.translation_id, chunk_index=1, original_text="two",
            translated_text=None, total_chunks=3, failed_chunks=1)
        stats = {'total_chunks': 3, 'completed_chunks': 3, 'failed_chunks': 1,
                 'unfinished_chunks': 1}
        ctx.emit(stats)
        return JobResult.from_run(stats, job.output_path, engine_ok=False)

    job = JobConfig(operation='translate', file_type='txt', input_path=str(source),
                    output_path=str(tmp_path / "out.txt"), source_language="English",
                    target_language="French", origin='cli', failure_report=failure_report,
                    provider=ProviderConfig(provider="poe", model="fake", api_key="unused"))
    sinks = _Sinks()
    result = asyncio.run(JobRunner(job, translation_id='cli_r', checkpoint_manager=manager,
                                   sinks=sinks, engines={'translate_file': translate_file}).run())
    return result, sinks


def test_runner_writes_the_report_when_enabled(env):
    tmp_path, source, manager = env
    result, sinks = _run(tmp_path, source, manager, failure_report=True)

    report = tmp_path / "out.report.txt"
    assert result.status is JobStatus.PARTIAL
    assert str(report) in result.output_paths
    text = report.read_text(encoding="utf-8")
    assert "Status:    partial" in text
    assert 'FAILED chunk 2 "two"' in _squash(text)
    assert any(key == "failure_report_written" for key, _ in sinks.logs)


def test_runner_writes_nothing_when_disabled(env):
    tmp_path, source, manager = env
    result, _ = _run(tmp_path, source, manager, failure_report=False)
    assert not (tmp_path / "out.report.txt").exists()
    assert result.output_paths == (str(tmp_path / "out.txt"),)


def test_failure_report_option_round_trips_and_defaults(tmp_path, monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "WRITE_FAILURE_REPORT", "false")
    monkeypatch.setattr(cfg, "DEFAULT_MODEL", "qwen3:14b")

    job = JobConfig(failure_report=True, output_path=str(tmp_path / "o.txt"))
    stored = job.to_persisted()
    assert stored['failure_report'] is True
    assert JobConfig.from_persisted(stored, output_dir=str(tmp_path)).failure_report is True

    args = argparse.Namespace(
        input=str(tmp_path / "book.txt"), output=None, source_lang="English",
        target_lang="French", model="qwen3:14b", api_endpoint="http://localhost:11434/api/generate",
        provider="ollama", gemini_api_key="", openai_api_key="", openrouter_api_key="",
        mistral_api_key="", deepseek_api_key="", poe_api_key="", nim_api_key="", llmtr_api_key="",
        parallel=1, text_cleanup=False, refine=False, refine_only=False, glossary=None,
        auto_glossary=False, auto_style=False, fix_terms=None, fix_terms_mode="smart",
        fix_terms_scan=False, tts=False)
    assert JobConfig.from_cli_args(args).failure_report is False
    args.failure_report = True
    assert JobConfig.from_cli_args(args).failure_report is True
    del args.failure_report
    monkeypatch.setattr(cfg, "WRITE_FAILURE_REPORT", "true")
    assert JobConfig.from_cli_args(args).failure_report is True


def test_cli_parser_has_the_flag():
    from translate import build_parser
    args = build_parser().parse_args(["-i", "book.txt", "--failure-report"])
    assert args.failure_report is True
    assert build_parser().parse_args(["-i", "book.txt"]).failure_report is False


def test_web_request_and_resume_carry_the_option(tmp_path, monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "WRITE_FAILURE_REPORT", "false")
    monkeypatch.setattr(cfg, "POE_API_KEY", "", raising=False)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    upload = uploads / "0123456789abcdef_book.txt"
    upload.write_text("Hello.", encoding="utf-8")
    body = {"file_path": str(upload), "file_type": "txt", "source_language": "English",
            "target_language": "French", "model": "m1", "llm_provider": "poe",
            "llm_api_endpoint": "https://api.openai.com/v1/chat/completions",
            "poe_api_key": "YOUR_API_KEY_HERE", "output_filename": "book (French).txt"}

    def build(**extra):
        return JobConfig.from_web_request({**body, **extra}, output_dir=str(tmp_path),
                                          uploads_dir=uploads)

    assert build().failure_report is False
    assert build(failure_report=True).failure_report is True
    monkeypatch.setattr(cfg, "WRITE_FAILURE_REPORT", "true")
    assert build().failure_report is True          # the saved setting is the default
    assert build(failure_report=False).failure_report is False

    stored = {"llm_provider": "ollama", "model": "a", "output_filename": "o.txt",
              "file_type": "txt", "llm_api_endpoint": "http://localhost:11434/api/generate"}
    # Turned on after the job started: a resumed job still writes the report.
    assert JobConfig.from_checkpoint(stored, output_dir="out").failure_report is True
    assert JobConfig.from_checkpoint(stored, output_dir="out",
                                     overrides={"failure_report": False}).failure_report is False
