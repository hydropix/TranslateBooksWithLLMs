"""Regression guard: no per-install fingerprint may come back.

TBL used to ship a machine-derived install identifier (`src/utils/telemetry.py`,
`src/utils/text_encoding.py`, both deleted in Phase 0 of
`plan/PLAN_UsageStatistics.md`). It was hashed from the MAC address and then:

* sent to every LLM provider as `X-Session-Token` / `X-Client-Agent` /
  `X-Request-ID` HTTP headers;
* steganographically encoded into translated output text as zero-width
  joiner/non-joiner runs, so it travelled inside files users distribute;
* written into document metadata as an EPUB `dc:identifier` of the form
  `urn:tbl:{12 hex}` (with `id="render-uid"`) and as a hexadecimal tail on the
  DOCX `core_properties.last_modified_by` stamp.

There was no notice, no consent and no opt-out, and the module docstrings
described the mechanism as "Unicode normalization". The removal is the
precondition for the lawful, opt-in usage statistics the rest of that plan
builds.

This module drives the *real* writers and asserts the observable traces are
gone. It is deliberately behavioural rather than a grep over the source: it
must fail for any reintroduction, whatever the new code is called. Do not
delete it as redundant with the unit tests of the individual writers — none of
those pin the absence of an identifier.

The honest, non-identifying attribution (`ATTRIBUTION_ENABLED`,
`GENERATOR_NAME`, `GENERATOR_SOURCE`) is explicitly *kept*, so every test here
forces attribution on: the fingerprint used to ride along with that branch, and
an assertion taken with attribution off would pass vacuously.
"""

import re
import zipfile
from pathlib import Path

import pytest
from docx import Document
from lxml import etree

from src import __version__
from src.config import NAMESPACES
from src.core.docx.converter import DocxHtmlConverter
from src.core.docx.plain_extractor import build_minimal_docx, extract_plain_paragraphs
from tests.characterization import fake_llm, fixtures, recorder


# The four codepoints the old encoder used to carry the identifier.
ZERO_WIDTH = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
}

# The install token was a 16-character hex string (and its 12-character
# prefix). Match any standalone hex run long enough to be one.
HEX_TOKEN = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{12,}(?![0-9a-fA-F])")

# Exactly `GENERATOR_NAME`, nothing appended. Spelled out as a literal rather
# than read from src.config so that widening the constant itself is caught.
GENERATOR_NAME_ONLY = re.compile(r"^TranslateBook with LLM \(TBL\)$")

FORBIDDEN_HEADERS = {"x-session-token", "x-client-agent", "x-request-id"}

# httpx sets these on every AsyncClient; `user-agent` is the one TBL overrides.
EXPECTED_HEADER_NAMES = {"accept", "accept-encoding", "connection", "user-agent"}


@pytest.fixture
def echo_llm(monkeypatch):
    """Replace every `create_llm_provider` lookup with the deterministic echo."""
    fake_llm.install(monkeypatch)


@pytest.fixture
def attribution_on(monkeypatch):
    """Force attribution on regardless of the developer's local .env.

    The DOCX writers read the flag live off `src.config`; the EPUB metadata
    writer imported it into its own namespace at module load, so it needs its
    own patch.
    """
    monkeypatch.setattr("src.config.ATTRIBUTION_ENABLED", True)
    monkeypatch.setattr("src.core.epub.opf_metadata.ATTRIBUTION_ENABLED", True)
    monkeypatch.setattr("src.core.epub.attribution_page.ATTRIBUTION_ENABLED", True)
    monkeypatch.setattr("src.core.epub.attribution_page.ATTRIBUTION_PAGE_ENABLED", True)


def _describe_zero_width(text: str) -> str:
    """Report the offending codepoints and their offsets, for a useful failure."""
    hits = [(i, hex(ord(c))) for i, c in enumerate(text) if c in ZERO_WIDTH]
    return f"{len(hits)} zero-width mark(s), first at {hits[:5]}"


# ---------------------------------------------------------------------------
# 1. Text output — no zero-width payload
# ---------------------------------------------------------------------------

# TXT translate is the criterion named in the plan. The refine path and the two
# SRT writers are added because those are the three places the encoder was
# actually wired into: a translate-only assertion would have passed even before
# the removal.
_TEXT_RUNS = {
    "translate_txt": (recorder.record_translation, fixtures.build_txt),
    "refine_txt": (recorder.record_refine, fixtures.build_txt),
    "translate_srt": (recorder.record_translation, fixtures.build_srt),
    "refine_srt": (recorder.record_refine, fixtures.build_srt),
}


@pytest.mark.parametrize("run_name", sorted(_TEXT_RUNS))
def test_text_output_carries_no_zero_width_payload(
    run_name, tmp_path, echo_llm, attribution_on
):
    """End-to-end text runs must not add U+200B/200C/200D/2060 to the output."""
    record, build_input = _TEXT_RUNS[run_name]
    input_path = build_input(tmp_path)
    output_path = tmp_path / f"out{input_path.suffix}"

    source_text = input_path.read_text(encoding="utf-8")
    assert not (set(source_text) & ZERO_WIDTH), "fixture is not zero-width-free"

    record(tmp_path, input_path, output_path)

    output_text = output_path.read_text(encoding="utf-8")
    assert output_text.strip(), f"{run_name}: produced an empty output"
    assert not (set(output_text) & ZERO_WIDTH), (
        f"{run_name}: output carries a zero-width payload — "
        f"{_describe_zero_width(output_text)}"
    )


# ---------------------------------------------------------------------------
# 2. EPUB — no auxiliary dc:identifier
# ---------------------------------------------------------------------------


def _read_opf(epub_path: Path) -> etree._Element:
    """Return the parsed OPF root of an EPUB, resolved through container.xml."""
    with zipfile.ZipFile(epub_path) as zf:
        container = etree.fromstring(zf.read("META-INF/container.xml"))
        rootfile = container.find(
            ".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile"
        )
        assert rootfile is not None, "EPUB has no rootfile entry"
        return etree.fromstring(zf.read(rootfile.get("full-path")))


def test_translated_epub_has_no_install_identifier(tmp_path, echo_llm, attribution_on):
    """A real EPUB translation must emit no per-install dc:identifier."""
    input_path = fixtures.build_epub(tmp_path)
    output_path = tmp_path / "out.epub"

    recorder.record_translation(tmp_path, input_path, output_path)

    opf_root = _read_opf(output_path)
    metadata = opf_root.find(".//opf:metadata", namespaces=NAMESPACES)
    assert metadata is not None, "translated EPUB lost its OPF metadata block"

    # The attribution branch — the one the identifier used to ride along with —
    # really did run, so the assertions below are not vacuous.
    contributors = metadata.findall(
        ".//{http://purl.org/dc/elements/1.1/}contributor"
    )
    assert contributors, "attribution contributor missing; the branch did not run"

    identifiers = metadata.findall(
        ".//{http://purl.org/dc/elements/1.1/}identifier"
    )
    for identifier in identifiers:
        text = (identifier.text or "").strip()
        assert not text.startswith("urn:tbl:"), (
            f"per-install dc:identifier reintroduced: {text!r}"
        )
        assert identifier.get("id") != "render-uid", (
            "dc:identifier id='render-uid' reintroduced"
        )


# ---------------------------------------------------------------------------
# 3. DOCX — last_modified_by carries no hexadecimal tail
# ---------------------------------------------------------------------------


def _assert_clean_stamp(docx_path: Path, writer: str) -> None:
    stamp = Document(str(docx_path)).core_properties.last_modified_by or ""
    assert not HEX_TOKEN.search(stamp), (
        f"{writer}: last_modified_by carries a hex tail: {stamp!r}"
    )
    assert GENERATOR_NAME_ONLY.match(stamp), (
        f"{writer}: last_modified_by must be exactly the generator name, "
        f"got {stamp!r}"
    )


def test_plain_extractor_docx_stamp_has_no_hex_tail(tmp_path, attribution_on):
    """`build_minimal_docx` stamps the constant generator name and nothing else."""
    source = fixtures.build_docx(tmp_path)
    content = extract_plain_paragraphs(str(source))
    output_path = tmp_path / "plain_out.docx"

    build_minimal_docx(list(content.paragraphs_text), content, str(output_path))

    _assert_clean_stamp(output_path, "build_minimal_docx")


def test_html_converter_docx_stamp_has_no_hex_tail(tmp_path, attribution_on):
    """`DocxHtmlConverter.from_html` stamps the constant generator name only."""
    source = fixtures.build_docx(tmp_path)
    converter = DocxHtmlConverter()
    html_content, metadata = converter.to_html(str(source))
    output_path = tmp_path / "html_out.docx"

    converter.from_html(html_content, metadata, str(output_path))

    _assert_clean_stamp(output_path, "DocxHtmlConverter.from_html")


# ---------------------------------------------------------------------------
# 4. LLM client — no identifying request headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_client_sends_no_identifying_headers():
    """`LLMProvider._get_client()` must attach only a constant User-Agent."""
    provider = fake_llm.FakeEchoProvider()
    try:
        client = await provider._get_client()
        headers = dict(client.headers)
    finally:
        await provider.close()

    names = set(headers)
    assert names & FORBIDDEN_HEADERS == set(), (
        f"identifying header(s) reintroduced: {sorted(names & FORBIDDEN_HEADERS)}"
    )
    assert names == EXPECTED_HEADER_NAMES, (
        f"unexpected header(s) on the LLM client: "
        f"{sorted(names ^ EXPECTED_HEADER_NAMES)}"
    )

    for name, value in headers.items():
        assert not HEX_TOKEN.search(value), (
            f"header {name!r} carries a hex token: {value!r}"
        )

    assert headers["user-agent"] == f"TranslateBookWithLLM/{__version__}", (
        f"User-Agent must be the per-release constant, got "
        f"{headers['user-agent']!r}"
    )
