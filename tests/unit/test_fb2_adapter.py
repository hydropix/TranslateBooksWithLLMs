"""
Tests for FictionBook 2 (.fb2) support (discussion #364).

Only the text of the translatable blocks may change (body paragraphs, verses,
subtitles, text authors, table cells, titles, epigraphs, poems, cites, notes,
and book-title / annotation / keywords in the metadata), and <lang> is set to
the target language. Ids, links, note references, images, binaries, author
names, document-info and attributes must come back unchanged.
"""

import argparse
import re
from pathlib import Path

import pytest
from lxml import etree

import src.core.epub.chunk_fallback as chunk_fallback_module
import src.core.llm.factory as llm_factory_module
from src.core.adapters.translate_file import get_file_type_from_path, translate_file
from src.core.epub.container import TranslationContainer
from src.core.fb2.document import (
    Fb2Error, apply_block_translation, apply_translations, build_translation_html,
    decode_fb2, parse_fb2, serialize_fb2, split_translated_html,
)
from src.persistence.checkpoint_manager import CheckpointManager
from src.utils.file_detector import detect_file_type
from tests.job_helpers import make_ctx, make_job

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "fb2" / "sample.fb2"
FB = "http://www.gribuser.ru/xml/fictionbook/2.0"
XLINK = "http://www.w3.org/1999/xlink"
NS = {"fb": FB}
HREF = f"{{{XLINK}}}href"
# Placeholders and XML entities must survive the fake translation untouched.
PROTECTED = re.compile(r'(\[id\d+\]|&#?\w+;)')


def _source_bytes() -> bytes:
    # LF-normalized, whatever line endings the checkout gave the fixture.
    return FIXTURE.read_bytes().replace(b"\r\n", b"\n")


def _fake_translate(content: str) -> str:
    """Upper-case every piece of text outside placeholders and entities."""
    return ''.join(part if PROTECTED.fullmatch(part) else part.upper()
                   for part in PROTECTED.split(content))


def _tree(data: bytes) -> etree._Element:
    return etree.fromstring(decode_fb2(data)[0].encode("utf-8"))


def _texts(root, xpath):
    return [''.join(el.itertext()) for el in root.xpath(xpath, namespaces=NS)]


class FakeLLMClient:
    async def close(self):
        pass


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    async def fake_request(text, **kwargs):
        calls.append(text)
        return _fake_translate(text)

    monkeypatch.setattr(llm_factory_module, "create_llm_provider",
                        lambda *args, **kwargs: FakeLLMClient())
    monkeypatch.setattr(chunk_fallback_module, "generate_translation_request", fake_request)
    return calls


async def _run(tmp_path, data: bytes, *, cm=None, job_id=None, check_interruption=None,
               **fields):
    src = tmp_path / "in.fb2"
    src.write_bytes(data)
    out = tmp_path / "out.fb2"
    fields.setdefault("max_tokens_per_chunk", 120)
    job = make_job(str(src), str(out), source_language="English",
                   target_language="French", **fields)
    ctx = make_ctx(translation_id=job_id, checkpoint_manager=cm,
                   check_interruption=check_interruption)
    result = await translate_file(job, ctx)
    return result, out


class TestDocument:
    def test_identity_round_trip_is_byte_identical(self):
        data = _source_bytes()
        document = parse_fb2(data)
        preserver = TranslationContainer().tag_preserver
        text, tag_map = preserver.preserve_tags(build_translation_html(document.blocks))
        restored = preserver.restore_tags(text, tag_map)
        assert apply_translations(document, split_translated_html(restored)) == []
        assert serialize_fb2(document) == data

    def test_blocks_cover_every_text_location_and_nothing_else(self):
        names = [b.name for b in parse_fb2(_source_bytes()).blocks]
        for name in ("book-title", "keywords", "p", "v", "subtitle", "text-author", "th", "td"):
            assert name in names
        html = build_translation_html(parse_fb2(_source_bytes()).blocks)
        for never in ("Fixture Press", "translatebook-tests", "Hand-written", "Marta</",
                      "iVBORw0KGgo", "sf_fantasy", "Island Tales", "ring_bell", "[1]"):
            assert never not in html

    def test_unparsable_translation_keeps_the_source(self):
        document = parse_fb2(_source_bytes())
        block = next(b for b in document.blocks if "<emphasis>" in b.inner)
        before = etree.tostring(block.element)
        assert apply_block_translation(block, "broken <emphasis>text") is False
        assert etree.tostring(block.element) == before

    def test_lost_note_reference_keeps_the_source(self):
        document = parse_fb2(_source_bytes())
        block = next(b for b in document.blocks if b.kept)
        assert apply_block_translation(block, "no marker left") is False

    def test_bare_ampersand_in_translation_is_escaped(self):
        document = parse_fb2(_source_bytes())
        block = next(b for b in document.blocks if b.name == "td")
        assert apply_block_translation(block, "Tom & Jerry &amp; co")
        assert block.element.text == "Tom & Jerry & co"

    def test_not_fictionbook_is_rejected(self):
        with pytest.raises(Fb2Error):
            parse_fb2(b'<?xml version="1.0"?><html/>')
        with pytest.raises(Fb2Error):
            parse_fb2(b'<FictionBook><body><p>unclosed</body></FictionBook>')

    def test_entities_are_not_expanded(self):
        data = (b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e "EXPANDED">]>'
                b'<FictionBook xmlns="' + FB.encode() + b'"><body><p>&e;</p></body></FictionBook>')
        document = parse_fb2(data)
        assert b"EXPANDED" not in serialize_fb2(document)


@pytest.mark.asyncio
class TestTranslation:
    async def test_structure_ids_links_and_binaries_are_unchanged(self, tmp_path, fake_llm):
        result, out = await _run(tmp_path, _source_bytes())
        assert result.status.value == "completed"
        source, output = _tree(_source_bytes()), _tree(out.read_bytes())

        def skeleton(root):
            return [(el.tag, dict(el.attrib)) for el in root.iter() if isinstance(el.tag, str)]

        assert skeleton(output) == skeleton(source)
        for xpath in ("//fb:binary", "//fb:description/fb:document-info",
                      "//fb:title-info/fb:author", "//fb:publish-info", "//fb:image",
                      "//fb:a[@type='note']", "//fb:code", "//fb:empty-line",
                      "//fb:genre", "//fb:date", "//fb:sequence"):
            assert [etree.tostring(e, with_tail=False) for e in output.xpath(xpath, namespaces=NS)] == \
                   [etree.tostring(e, with_tail=False) for e in source.xpath(xpath, namespaces=NS)], xpath
        # The xlink prefix is kept as written.
        assert b'l:href="#n1"' in out.read_bytes()

    async def test_every_text_location_is_translated(self, tmp_path, fake_llm):
        _, out = await _run(tmp_path, _source_bytes())
        output = _tree(out.read_bytes())
        expected = {
            "//fb:title-info/fb:book-title": "THE LIGHTHOUSE KEEPER'S LEDGER",
            "//fb:title-info/fb:keywords": "LIGHTHOUSE, ISLAND, LEDGER, TEST",
            "//fb:annotation/fb:p[2]": "A SHORT TEST BOOK WITH EVERY FB2 STRUCTURE IN IT.",
            "/fb:FictionBook/fb:body[1]/fb:title/fb:p[2]": "THE LIGHTHOUSE KEEPER'S LEDGER",
            "/fb:FictionBook/fb:body[1]/fb:epigraph/fb:text-author": "OLD HARBOUR SAYING",
            "//fb:section[@id='ch1']/fb:epigraph//fb:v[2]": "AND NOBODY ASKS IT WHY.",
            "//fb:section[@id='ch2']/fb:cite/fb:text-author": "LEDGER ENTRY, PAGE 212",
            "//fb:th[1]": "DATE",
            "//fb:tr[2]/fb:td[3]": "PASSED SAFELY",
            "//fb:poem/fb:title/fb:p": "LAMP ROOM",
            "//fb:poem/fb:epigraph/fb:p": "FOR WHOEVER COMES AFTER.",
            "//fb:stanza[2]/fb:v[3]": "DO NOT BE AFRAID OF THEM:",
            "//fb:poem/fb:text-author": "E. HALE",
            "//fb:body[@name='notes']/fb:title/fb:p": "NOTES",
            "//fb:section[@id='n3']/fb:p[2]":
                "A SECOND PARAGRAPH INSIDE THE SAME NOTE, TO TEST MULTI-PARAGRAPH NOTES.",
        }
        for xpath, text in expected.items():
            assert _texts(output, xpath) == [text], xpath
        # Inline markup stays, with its text translated.
        assert _texts(output, "//fb:section[@id='ch1']/fb:p[3]/fb:strong[2]/fb:emphasis") == ["CHURCH"]
        links = output.xpath("//fb:a[not(@type)]", namespaces=NS)
        assert [''.join(a.itertext()) for a in links] == ["CHAPTER 3", "REFERENCE LIST"]
        # Note references and code are never translated nor sent.
        assert _texts(output, "//fb:a[@type='note']") == ["[1]", "[2]", "[3]"]
        assert _texts(output, "//fb:code") == ["ring_bell(times=3)"]
        assert not any("ring_bell" in call for call in fake_llm)
        # Author names and publish-info are not translated.
        assert _texts(output, "//fb:title-info/fb:author/fb:last-name") == ["Fixture"]
        assert _texts(output, "//fb:publish-info/fb:book-name") == ["The Lighthouse Keeper's Ledger"]

    async def test_lang_is_set_to_the_target_language(self, tmp_path, fake_llm):
        _, out = await _run(tmp_path, _source_bytes())
        output = _tree(out.read_bytes())
        assert _texts(output, "//fb:title-info/fb:lang") == ["fr"]
        assert _texts(output, "//fb:title-info/fb:src-lang") == ["en"]

    async def test_windows_1251_input_gives_utf8_output(self, tmp_path, fake_llm):
        text = _source_bytes().decode("utf-8")
        text = text.replace('encoding="utf-8"', 'encoding="windows-1251"')
        text = text.replace("Old harbour saying", "Старая поговорка")
        data = text.encode("cp1251")
        assert parse_fb2(data).source_encoding == "windows-1251"
        _, out = await _run(tmp_path, data)
        raw = out.read_bytes()
        assert raw.startswith(b'<?xml version="1.0" encoding="utf-8"?>')
        output = _tree(raw)
        assert _texts(output, "/fb:FictionBook/fb:body[1]/fb:epigraph/fb:text-author") == \
               ["СТАРАЯ ПОГОВОРКА"]

    async def test_utf8_bom_is_accepted_and_dropped(self, tmp_path, fake_llm):
        _, out = await _run(tmp_path, b"\xef\xbb\xbf" + _source_bytes())
        raw = out.read_bytes()
        assert raw.startswith(b'<?xml version="1.0" encoding="utf-8"?>')
        assert _texts(_tree(raw), "//fb:th[1]") == ["DATE"]

    async def test_xlink_prefix_is_kept(self, tmp_path, fake_llm):
        data = _source_bytes().replace(b"xmlns:l=", b"xmlns:xlink=").replace(b"l:href", b"xlink:href")
        _, out = await _run(tmp_path, data)
        raw = out.read_bytes()
        assert b'xlink:href="#n2"' in raw and b"l:href" not in raw.replace(b"xlink:href", b"")
        assert [a.get(HREF) for a in _tree(raw).iter(f"{{{FB}}}a")] == \
               [a.get(HREF) for a in _tree(data).iter(f"{{{FB}}}a")]

    async def test_bilingual_keeps_the_source_block_first(self, tmp_path, fake_llm):
        _, out = await _run(tmp_path, _source_bytes(), bilingual_output=True)
        output = _tree(out.read_bytes())
        verses = _texts(output, "//fb:section[@id='ch1']/fb:epigraph//fb:v")
        assert verses[:2] == ["The tide comes in, the tide goes out,",
                              "THE TIDE COMES IN, THE TIDE GOES OUT,"]
        ids = output.xpath("//@id")
        assert len(ids) == len(set(ids))

    async def test_resume_translates_only_the_remaining_chunks(self, tmp_path, cm, fake_llm):
        stop_after = {"n": 3}

        def check_interruption():
            return len(fake_llm) >= stop_after["n"]

        result, out = await _run(tmp_path, _source_bytes(), cm=cm, job_id="job",
                                 check_interruption=check_interruption)
        assert not out.exists()
        first_pass = len(fake_llm)
        assert first_pass >= 3
        stop_after["n"] = 10_000

        _, out = await _run(tmp_path, _source_bytes(), cm=cm, job_id="job",
                            check_interruption=check_interruption)
        sent = fake_llm[first_pass:]
        assert sent and not set(sent) & set(fake_llm[:first_pass])
        assert _texts(_tree(out.read_bytes()), "//fb:th[1]") == ["DATE"]
        assert cm.load_xhtml_partial_state("job", "in.fb2") is None


@pytest.mark.asyncio
async def test_failed_chunks_keep_the_checkpoint_for_a_retry(tmp_path, cm, monkeypatch):
    llm_up = {"value": False}

    async def flaky_request(text, **kwargs):
        return _fake_translate(text) if llm_up["value"] else None

    monkeypatch.setattr(llm_factory_module, "create_llm_provider",
                        lambda *args, **kwargs: FakeLLMClient())
    monkeypatch.setattr(chunk_fallback_module, "generate_translation_request", flaky_request)

    _, out = await _run(tmp_path, _source_bytes(), cm=cm, job_id="job")
    # The source text is written back, still as valid FB2, and the state stays.
    assert _texts(_tree(out.read_bytes()), "//fb:th[1]") == ["Date"]
    assert cm.load_xhtml_partial_state("job", "in.fb2") is not None

    llm_up["value"] = True
    _, out = await _run(tmp_path, _source_bytes(), cm=cm, job_id="job")
    assert _texts(_tree(out.read_bytes()), "//fb:th[1]") == ["DATE"]
    assert cm.load_xhtml_partial_state("job", "in.fb2") is None


BLOCK_XPATH = ("//fb:body//fb:p | //fb:body//fb:v | //fb:body//fb:subtitle"
               " | //fb:body//fb:text-author | //fb:body//fb:th | //fb:body//fb:td")


@pytest.mark.asyncio
async def test_lost_placeholders_never_move_text_across_blocks(tmp_path, monkeypatch):
    """A model that drops every placeholder of a multi-block chunk must not
    smear text from one FB2 block into its neighbours (proportional token
    alignment over the whole chunk did): each block is retried on its own."""
    placeholder = re.compile(r'\[id\d+\]')
    # No single block is longer than this: a longer request spans blocks.
    longest = max(len(''.join(b.element.itertext())) for b in parse_fb2(_source_bytes()).blocks)
    calls = {"multi_block": 0}

    async def careless_request(text, **kwargs):
        clean = placeholder.sub('', text)
        if len(clean) <= longest:
            return _fake_translate(text)
        calls["multi_block"] += 1
        # Every placeholder dropped, and a reply whose length no longer
        # matches the source, so proportional reinsertion lands mid-word.
        return "NOTE DU TRADUCTEUR, TRÈS LONGUE: " + _fake_translate(clean)

    monkeypatch.setattr(llm_factory_module, "create_llm_provider",
                        lambda *args, **kwargs: FakeLLMClient())
    monkeypatch.setattr(chunk_fallback_module, "generate_translation_request",
                        careless_request)
    result, out = await _run(tmp_path, _source_bytes(), max_tokens_per_chunk=300)

    assert calls["multi_block"] > 0, "the fixture must yield multi-block chunks"
    source = _tree(_source_bytes())
    output = _tree(out.read_bytes())
    kept = {"[1]", "[2]", "[3]", "ring_bell(times=3)"}  # note refs and code
    pairs = list(zip(source.xpath(BLOCK_XPATH, namespaces=NS),
                     output.xpath(BLOCK_XPATH, namespaces=NS)))
    assert len(pairs) == len(source.xpath(BLOCK_XPATH, namespaces=NS))
    for src_el, out_el in pairs:
        expected = ''.join(t if t in kept else t.upper() for t in src_el.itertext())
        assert ''.join(out_el.itertext()) == expected
    assert "TRADUCTEUR" not in out.read_bytes().decode("utf-8")


class TestRouting:
    @pytest.mark.parametrize("name", ["book.fb2", "BOOK.FB2"])
    def test_fb2_extension_is_detected(self, name):
        assert detect_file_type(name) == "fb2"
        assert get_file_type_from_path(name) == "fb2"

    def _args(self, **overrides):
        values = dict(input="book.fb2", output=None, source_lang="English",
                      target_lang="French", provider="ollama", model="m")
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_job_config_from_cli_args_uses_fb2(self):
        from src.core.job import JobConfig
        job = JobConfig.from_cli_args(self._args())
        assert job.file_type == "fb2"
        assert job.output_path == "book (French).fb2"

    def test_refine_is_ignored_and_refine_only_rejected(self):
        from src.core.job import JobConfig, JobConfigError
        job = JobConfig.from_cli_args(self._args(refine=True))
        assert any("FB2" in notice for notice in job.notices)
        with pytest.raises(JobConfigError, match="FB2"):
            JobConfig.from_cli_args(self._args(refine_only=True))

    def test_upload_validator_accepts_fb2_and_rejects_other_xml(self, tmp_path):
        from src.utils.security import SecureFileHandler
        handler = SecureFileHandler.__new__(SecureFileHandler)
        good = tmp_path / "a.fb2"
        good.write_bytes(_source_bytes())
        assert handler._validate_fb2_file(good).is_valid
        bad = tmp_path / "b.fb2"
        bad.write_bytes(b"<html><body>nope</body></html>")
        assert not handler._validate_fb2_file(bad).is_valid
