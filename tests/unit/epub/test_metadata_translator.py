"""
Phase 6 of plan/PLAN_CjkSourceRendering.md: localize the packaging metadata of a
translated EPUB (OPF `dc:title` / `dc:description`, every NCX `docTitle/text`),
behind `EPUB_TRANSLATE_METADATA_ENABLED`, in exactly one LLM call, without ever
failing the job.

Two layers:
  - unit tests driving `translate_opf_metadata` directly against a small OPF +
    NCX pair on disk, with a stub client whose answer each test controls;
  - one integration test through `translate_epub_file`, which is what pins the
    step ordering: the output OPF must carry BOTH the translated title (step
    5.5, in-memory tree) AND no `duokan-body-font` meta (step 6.6, re-parsed
    tree). Swap the two steps and one of the halves is silently lost.

The integration layer reuses the Phase 5 harness (echo LLM client, EPUB
zipping, attribution toggle) and the container builder
(`_build_cjk_epub_dir`), both shared through `tests/unit/epub/conftest.py`
rather than a cross-module import chain.

No API key and no network access anywhere: every client here is a local stub.
"""
import zipfile
from pathlib import Path

import pytest
from lxml import etree

import src.core.epub.translator as translator_module
from src.config import (
    GENERATOR_NAME,
    GENERATOR_SOURCE,
    INPUT_TAG_IN,
    INPUT_TAG_OUT,
    TRANSLATE_TAG_IN,
    TRANSLATE_TAG_OUT,
)
from src.core.epub.metadata_translator import (
    DESCRIPTION_MAX_CHARS,
    translate_opf_metadata,
)
from src.core.epub.translator import translate_epub_file
from src.core.llm.base import LLMResponse
from src.core.llm.utils.extraction import TranslationExtractor

from tests.unit.epub.conftest import _disable_attribution, _echo_llm_client, _read
from tests.job_helpers import make_ctx, make_job


SOURCE_TITLE = "被渣后和前夫破镜重圆了"
SOURCE_DESCRIPTION = "婚后三年，她终于看清了这个男人。"
SOURCE_CREATOR = "林清欢"

FRENCH_TITLE = "Réconciliée avec son ex-mari après la trahison"
FRENCH_DESCRIPTION = "Après trois ans de mariage, elle voit enfin cet homme tel qu'il est."


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _opf_text(description: str = None) -> str:
    """A minimal OPF carrying a CJK title, a CJK author and an optional description."""
    description_element = (
        f"    <dc:description>{description}</dc:description>\n"
        if description is not None else ""
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0" '
        'unique-identifier="book-id">\n'
        '  <metadata>\n'
        f'    <dc:title>{SOURCE_TITLE}</dc:title>\n'
        f'    <dc:creator>{SOURCE_CREATOR}</dc:creator>\n'
        '    <dc:language>fr</dc:language>\n'
        f'{description_element}'
        '  </metadata>\n'
        '  <manifest>\n'
        '    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>\n'
        '  </manifest>\n'
        '  <spine toc="ncx"/>\n'
        '</package>\n'
    )


NCX_TEXT = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" xml:lang="zh">\n'
    f'  <docTitle><text>{SOURCE_TITLE}</text></docTitle>\n'
    '  <navMap/>\n'
    '</ncx>\n'
)


class StubClient:
    """Local stand-in for the pipeline's LLM client.

    Implements only what `translate_opf_metadata` uses: `generate`,
    `extract_translation` and a settable `context_window`. Counts its calls so a
    test can assert "exactly one" or "never".
    """

    def __init__(self, answer: str = None, exception: Exception = None,
                 response: LLMResponse = None):
        self._answer = answer
        self._exception = exception
        self._response = response
        self.context_window = 2048
        self.calls = []
        self.call_kwargs = []
        self._extractor = TranslationExtractor(TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT)

    async def generate(self, prompt, system_prompt=None, **kwargs):
        self.calls.append((prompt, system_prompt, self.context_window))
        self.call_kwargs.append(dict(kwargs))
        if self._exception is not None:
            raise self._exception
        if self._response is not None:
            return self._response
        return LLMResponse(content=self._answer, prompt_tokens=10,
                           completion_tokens=10, context_used=20,
                           context_limit=4096)

    def extract_translation(self, response):
        return self._extractor.extract(response)


def _tagged_answer(title: str = None, description: str = None) -> str:
    """A well-formed model answer, in the shape the prompt asks for."""
    body = ""
    if title is not None:
        body += f"<BOOK_TITLE>{title}</BOOK_TITLE>\n"
    if description is not None:
        body += f"<BOOK_DESCRIPTION>{description}</BOOK_DESCRIPTION>\n"
    return f"{TRANSLATE_TAG_IN}\n{body}{TRANSLATE_TAG_OUT}"


def _signature() -> str:
    """The signature `_update_epub_metadata` appends, built from the same constants."""
    return f"\n\nTranslated using {GENERATOR_NAME}\n{GENERATOR_SOURCE}"


class Book:
    """An OPF + NCX pair on disk, parsed the way the pipeline parses them."""

    def __init__(self, root: Path, description: str = None):
        root.mkdir(parents=True, exist_ok=True)
        self.opf_path = root / "content.opf"
        self.ncx_path = root / "toc.ncx"
        self.opf_path.write_text(_opf_text(description), encoding="utf-8")
        self.ncx_path.write_text(NCX_TEXT, encoding="utf-8")
        self.opf_dir = str(root)
        self.tree = etree.parse(str(self.opf_path))

    async def run(self, client, target_language="French", **overrides):
        self.events = []

        def log_callback(event, message, **_kwargs):
            self.events.append((event, message))

        kwargs = dict(
            opf_tree=self.tree,
            opf_path=str(self.opf_path),
            opf_dir=self.opf_dir,
            source_language="Chinese",
            target_language=target_language,
            llm_client=client,
            model_name="stub-model",
            log_callback=log_callback,
        )
        kwargs.update(overrides)
        return await translate_opf_metadata(**kwargs)

    def written_opf(self) -> str:
        return self.opf_path.read_text(encoding="utf-8")

    def written_ncx(self) -> str:
        return self.ncx_path.read_text(encoding="utf-8")

    def field(self, local_name: str) -> str:
        """Text of a dc: element as re-read from disk."""
        tree = etree.parse(str(self.opf_path))
        for element in tree.getroot().iter():
            if isinstance(element.tag, str) and element.tag.endswith(f"}}{local_name}"):
                return element.text or ""
        return None

    def ncx_doctitle(self) -> str:
        tree = etree.parse(str(self.ncx_path))
        for element in tree.getroot().iter():
            if isinstance(element.tag, str) and element.tag.endswith("}text"):
                return element.text or ""
        return None

    def event_names(self):
        return [event for event, _ in self.events]


@pytest.fixture
def book(tmp_path: Path) -> Book:
    return Book(tmp_path / "book", description=SOURCE_DESCRIPTION)


# ---------------------------------------------------------------------------
# Criterion 1 -- a good answer replaces both fields and reaches the NCX
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_good_answer_translates_both_fields_and_the_ncx_title(book):
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    result = await book.run(client)

    assert result['title_translated'] is True
    assert result['description_translated'] is True
    assert result['ncx_doctitle_updated'] == 1
    assert result['skipped_reason'] is None

    assert book.field("title") == FRENCH_TITLE
    assert book.field("description") == FRENCH_DESCRIPTION
    assert book.ncx_doctitle() == FRENCH_TITLE
    # The author is never translated nor transliterated.
    assert book.field("creator") == SOURCE_CREATOR
    assert SOURCE_TITLE not in book.written_opf()
    assert SOURCE_TITLE not in book.written_ncx()
    assert "epub_metadata_translated" in book.event_names()


@pytest.mark.asyncio
async def test_large_prompt_sizes_its_window_per_request_without_mutating_the_client(book):
    """A prompt larger than the client's window is sent with a per-request
    ``context_window``; the shared client keeps its own window throughout."""
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    client.context_window = 16
    await book.run(client)

    assert len(client.calls) == 1
    # The client was never mutated, not even during the call.
    assert client.calls[0][2] == 16
    assert client.context_window == 16
    sent = client.call_kwargs[0].get('context_window')
    assert isinstance(sent, int) and sent > 16


@pytest.mark.asyncio
async def test_small_prompt_uses_the_client_window(book):
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    client.context_window = 32768
    await book.run(client)

    assert 'context_window' not in client.call_kwargs[0]


# ---------------------------------------------------------------------------
# Criterion 8 -- exactly one LLM call per book
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exactly_one_llm_call_for_both_fields(book):
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    await book.run(client)

    assert len(client.calls) == 1
    prompt = client.calls[0][0]
    # Both fields travelled in that single request, inside the source tags.
    assert SOURCE_TITLE in prompt
    assert SOURCE_DESCRIPTION in prompt
    assert INPUT_TAG_IN in prompt and INPUT_TAG_OUT in prompt
    # The author never leaves the OPF.
    assert SOURCE_CREATOR not in prompt


# ---------------------------------------------------------------------------
# Criterion 2 -- an answer still in the source script is rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cjk_answer_is_rejected_and_originals_are_preserved(book):
    client = StubClient(_tagged_answer(SOURCE_TITLE, SOURCE_DESCRIPTION))
    result = await book.run(client)

    assert result['title_translated'] is False
    assert result['description_translated'] is False
    assert result['ncx_doctitle_updated'] == 0
    assert book.field("title") == SOURCE_TITLE
    assert book.field("description") == SOURCE_DESCRIPTION
    assert book.ncx_doctitle() == SOURCE_TITLE
    assert "epub_metadata_title_rejected" in book.event_names()
    assert "epub_metadata_description_rejected" in book.event_names()


@pytest.mark.asyncio
async def test_cjk_answer_is_accepted_when_the_target_is_cjk(tmp_path):
    """The CJK guard is about the *target*, not about the characters as such."""
    book = Book(tmp_path / "cjk_target", description="A three-year marriage.")
    client = StubClient(_tagged_answer("破镜重圆", "婚后三年。"))
    result = await book.run(client, target_language="Chinese")

    assert result['title_translated'] is True
    assert result['description_translated'] is True
    assert book.field("title") == "破镜重圆"


# ---------------------------------------------------------------------------
# Criterion 3 -- an empty answer is rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_answer_is_rejected(book):
    client = StubClient(_tagged_answer("", "   \n  "))
    result = await book.run(client)

    assert result['title_translated'] is False
    assert result['description_translated'] is False
    assert book.field("title") == SOURCE_TITLE
    assert book.field("description") == SOURCE_DESCRIPTION


@pytest.mark.asyncio
async def test_unparseable_answer_keeps_both_originals(book):
    client = StubClient("Sure! Here is the translation you asked for.")
    result = await book.run(client)

    assert result['title_translated'] is False
    assert result['description_translated'] is False
    assert book.field("title") == SOURCE_TITLE
    assert book.field("description") == SOURCE_DESCRIPTION


@pytest.mark.asyncio
async def test_multiline_title_is_rejected_but_the_description_still_lands(book):
    client = StubClient(_tagged_answer("Line one\nLine two", FRENCH_DESCRIPTION))
    result = await book.run(client)

    assert result['title_translated'] is False
    assert result['description_translated'] is True
    assert book.field("title") == SOURCE_TITLE
    assert book.field("description") == FRENCH_DESCRIPTION
    # No accepted title means no NCX propagation: a title is never invented.
    assert book.ncx_doctitle() == SOURCE_TITLE


@pytest.mark.asyncio
async def test_runaway_answer_is_rejected(book):
    runaway = "Here is what this description means, at length. " * 40
    client = StubClient(_tagged_answer(FRENCH_TITLE, runaway))
    result = await book.run(client)

    assert len(runaway) > 4 * len(SOURCE_DESCRIPTION)
    assert result['description_translated'] is False
    assert result['skipped_reason'] == 'description_rejected'
    assert book.field("description") == SOURCE_DESCRIPTION
    # The title in the same answer is judged on its own merits.
    assert result['title_translated'] is True


# ---------------------------------------------------------------------------
# Criterion 4 -- a raising client never fails anything
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_raising_client_preserves_everything_and_logs(book):
    client = StubClient(exception=RuntimeError("provider exploded"))
    result = await book.run(client)

    assert result == {
        'title_translated': False,
        'description_translated': False,
        'ncx_doctitle_updated': 0,
        'skipped_reason': 'llm_error',
    }
    assert book.field("title") == SOURCE_TITLE
    assert book.field("description") == SOURCE_DESCRIPTION
    assert book.ncx_doctitle() == SOURCE_TITLE
    assert "epub_metadata_translation_failed" in book.event_names()


@pytest.mark.asyncio
async def test_no_response_preserves_everything(book):
    # Content-less response: the model answered nothing usable.
    client = StubClient(answer=None)
    result = await book.run(client)

    assert result['skipped_reason'] == 'llm_no_response'
    assert book.field("title") == SOURCE_TITLE


# ---------------------------------------------------------------------------
# Criterion 5 -- the attribution signature round-trips exactly once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_signature_is_stripped_before_the_call_and_reappended_once(tmp_path):
    signature = _signature()
    book = Book(tmp_path / "signed", description=SOURCE_DESCRIPTION + signature)
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    result = await book.run(client)

    assert result['description_translated'] is True

    # It was not sent to the model...
    prompt = client.calls[0][0]
    assert GENERATOR_NAME not in prompt
    assert GENERATOR_SOURCE not in prompt

    # ...and it came back verbatim, appended exactly once.
    description = book.field("description")
    assert description == FRENCH_DESCRIPTION + signature
    assert description.count(GENERATOR_SOURCE) == 1
    assert description.count("Translated using") == 1


@pytest.mark.asyncio
async def test_signature_only_description_is_not_sent_at_all(tmp_path):
    """`_update_epub_metadata` writes the bare signature when a book had no
    description. There is nothing to translate then -- only the title is."""
    signature_only = _signature().strip()
    book = Book(tmp_path / "sig_only", description=signature_only)
    client = StubClient(_tagged_answer(FRENCH_TITLE))
    result = await book.run(client)

    assert result['title_translated'] is True
    assert result['description_translated'] is False
    assert book.field("description") == signature_only
    assert "BOOK_DESCRIPTION" not in client.calls[0][0]


@pytest.mark.asyncio
async def test_missing_description_element_translates_the_title_only(tmp_path):
    book = Book(tmp_path / "no_desc", description=None)
    client = StubClient(_tagged_answer(FRENCH_TITLE))
    result = await book.run(client)

    assert result['title_translated'] is True
    assert result['description_translated'] is False
    assert book.field("description") is None


# ---------------------------------------------------------------------------
# Criterion 6 -- an over-long description is skipped, the title is not
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_over_long_description_is_skipped_but_the_title_is_translated(tmp_path):
    long_description = "婚" * (DESCRIPTION_MAX_CHARS + 1)
    assert len(long_description) == 4001
    book = Book(tmp_path / "long_desc", description=long_description)
    client = StubClient(_tagged_answer(FRENCH_TITLE))
    result = await book.run(client)

    assert result['skipped_reason'] == 'description_too_long'
    assert result['description_translated'] is False
    assert result['title_translated'] is True
    assert book.field("description") == long_description
    assert book.field("title") == FRENCH_TITLE
    # The over-long text never left the process.
    assert long_description not in client.calls[0][0]
    assert "epub_metadata_description_skipped" in book.event_names()


@pytest.mark.asyncio
async def test_description_at_the_limit_is_still_sent(tmp_path):
    at_limit = "婚" * DESCRIPTION_MAX_CHARS
    book = Book(tmp_path / "limit_desc", description=at_limit)
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    result = await book.run(client)

    assert result['skipped_reason'] is None
    assert result['description_translated'] is True
    assert at_limit in client.calls[0][0]


# ---------------------------------------------------------------------------
# Robustness -- nothing to do, and a broken NCX
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_metadata_means_no_llm_call(tmp_path):
    opf_path = tmp_path / "empty.opf"
    opf_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0">\n'
        '  <metadata/>\n'
        '</package>\n', encoding="utf-8")
    client = StubClient(_tagged_answer(FRENCH_TITLE))

    result = await translate_opf_metadata(
        opf_tree=etree.parse(str(opf_path)),
        opf_path=str(opf_path),
        opf_dir=str(tmp_path),
        source_language="Chinese",
        target_language="French",
        llm_client=client,
        model_name="stub-model",
    )

    assert result['skipped_reason'] == 'no_translatable_metadata'
    assert client.calls == []


@pytest.mark.asyncio
async def test_broken_ncx_does_not_fail_the_pass(tmp_path):
    book = Book(tmp_path / "broken_ncx", description=SOURCE_DESCRIPTION)
    book.ncx_path.write_bytes(b"\x00\x01 not xml at all <<<")
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    result = await book.run(client)

    assert result['title_translated'] is True
    assert result['ncx_doctitle_updated'] == 0
    assert book.field("title") == FRENCH_TITLE


@pytest.mark.asyncio
async def test_unwritable_opf_reports_nothing_translated(book):
    """A failed OPF write must not be reported as a success, and must not reach
    the NCX: the book stays consistently in its source language.

    The write is made to fail by pointing `opf_path` at a directory, which is
    the closest stand-in for the real cases (read-only mount, locked file).
    """
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    result = await book.run(client, opf_path=book.opf_dir)

    assert result['title_translated'] is False
    assert result['description_translated'] is False
    assert result['ncx_doctitle_updated'] == 0
    assert result['skipped_reason'] == 'opf_write_failed'
    assert book.field("title") == SOURCE_TITLE
    assert book.ncx_doctitle() == SOURCE_TITLE
    assert "epub_metadata_opf_write_failed" in book.event_names()


@pytest.mark.asyncio
async def test_every_ncx_under_the_opf_dir_is_updated(tmp_path):
    book = Book(tmp_path / "multi_ncx", description=SOURCE_DESCRIPTION)
    (Path(book.opf_dir) / "nested").mkdir()
    (Path(book.opf_dir) / "nested" / "other.ncx").write_text(NCX_TEXT, encoding="utf-8")
    client = StubClient(_tagged_answer(FRENCH_TITLE, FRENCH_DESCRIPTION))
    result = await book.run(client)

    assert result['ncx_doctitle_updated'] == 2


# ---------------------------------------------------------------------------
# Integration -- ordering vs step 6.6, and the flag
# ---------------------------------------------------------------------------

def _pipeline_client():
    """Echo stub for body chunks, French metadata for the single metadata call.

    Wraps the Phase 5 echo client instead of redefining it: only the metadata
    request (recognizable by its field tags) is answered differently.
    """
    client = _echo_llm_client()
    echo_generate = client.generate
    client.metadata_calls = []

    async def generate(user_prompt, system_prompt=None, **kwargs):
        if "BOOK_TITLE" in user_prompt:
            client.metadata_calls.append(user_prompt)
            return LLMResponse(
                content=_tagged_answer(FRENCH_TITLE),
                prompt_tokens=10, completion_tokens=10,
                context_used=20, context_limit=4096,
            )
        return await echo_generate(user_prompt, system_prompt=system_prompt, **kwargs)

    client.generate = generate
    # The pipeline's extract_translation is identity in this harness; the
    # metadata pass tolerates that and parses the field tags out of the payload.
    return client


# `input_epub` is a fixture defined in conftest.py -- pytest injects it
# automatically, no import needed.


async def _translate(input_path: Path, output_path: Path, monkeypatch, client,
                     attribution: bool = False):
    """Drive translate_epub_file with `client` as the pipeline's LLM client.

    Attribution is off by default (it defaults from the developer's local .env,
    so pinning it keeps the assertions deterministic); one test turns it on to
    exercise the real `_update_epub_metadata` signature interplay.
    """
    monkeypatch.setattr(translator_module, "_create_llm_client",
                        lambda *args, **kwargs: client)
    if attribution:
        monkeypatch.setattr(translator_module, "ATTRIBUTION_ENABLED", True)
        monkeypatch.setattr(
            "src.core.epub.attribution_page.ATTRIBUTION_ENABLED", False)
    else:
        _disable_attribution(monkeypatch)

    events = []

    def log_callback(event, message, **_kwargs):
        events.append((event, message))

    await translate_epub_file(
        make_job(
            input_path=str(input_path),
            output_path=str(output_path),
            source_language="Chinese",
            target_language="French",
        ),
        make_ctx(log_callback=log_callback),
    )
    return events


@pytest.mark.asyncio
async def test_pipeline_writes_the_translated_title_without_undoing_step_6_6(
    tmp_path, input_epub, monkeypatch
):
    """The ordering test.

    Step 5.5 writes the in-memory OPF tree; step 6.6 re-parses the OPF from disk
    and writes its own tree. Both halves must be present in the output: the
    translated `dc:title` (5.5) and the removed `duokan-body-font` meta (6.6).
    Running 5.5 after 6.6 would revert the meta removal.
    """
    monkeypatch.setattr(translator_module, "EPUB_TRANSLATE_METADATA_ENABLED", True)
    client = _pipeline_client()
    output_epub = tmp_path / "output.epub"
    events = await _translate(input_epub, output_epub, monkeypatch, client)

    opf = _read(output_epub, "OEBPS/content.opf")
    assert f"<dc:title>{FRENCH_TITLE}</dc:title>" in opf
    assert "<dc:title>被渣后和前夫破镜重圆了</dc:title>" not in opf
    # Only dc:title is rewritten: the fixture's calibre:title_sort meta carries
    # the same source string and is out of scope, which is why the assertion
    # above is element-scoped rather than a bare substring check.
    assert 'name="calibre:title_sort"' in opf
    assert "duokan-body-font" not in opf

    # The NCX carries the translated docTitle (step 5.5) and the target
    # language (step 6.6), so neither pass clobbered the other's NCX write.
    ncx = _read(output_epub, "OEBPS/toc.ncx")
    assert f"<text>{FRENCH_TITLE}</text>" in ncx
    assert 'xml:lang="fr"' in ncx

    assert len(client.metadata_calls) == 1
    event_names = [event for event, _ in events]
    assert "epub_metadata_translated" in event_names
    assert "epub_save_success" in event_names


@pytest.mark.asyncio
async def test_pipeline_flag_off_makes_no_llm_call_and_keeps_the_source_title(
    tmp_path, input_epub, monkeypatch
):
    monkeypatch.setattr(translator_module, "EPUB_TRANSLATE_METADATA_ENABLED", False)
    client = _pipeline_client()
    output_epub = tmp_path / "output_disabled.epub"
    events = await _translate(input_epub, output_epub, monkeypatch, client)

    assert client.metadata_calls == []
    opf = _read(output_epub, "OEBPS/content.opf")
    assert "<dc:title>被渣后和前夫破镜重圆了</dc:title>" in opf
    assert not any(event.startswith("epub_metadata") for event, _ in events)
    assert any(event == "epub_save_success" for event, _ in events)


@pytest.mark.asyncio
async def test_pipeline_signature_only_description_survives_untouched(
    tmp_path, input_epub, monkeypatch
):
    """End-to-end version of the signature round-trip.

    The fixture book has no dc:description, so the real `_update_epub_metadata`
    creates one holding nothing but the attribution signature. Step 5.5 must
    recognize that shape, translate the title only, and leave the signature in
    place exactly once.
    """
    monkeypatch.setattr(translator_module, "EPUB_TRANSLATE_METADATA_ENABLED", True)
    client = _pipeline_client()
    output_epub = tmp_path / "output_signed.epub"
    await _translate(input_epub, output_epub, monkeypatch, client,
                     attribution=True)

    opf = _read(output_epub, "OEBPS/content.opf")
    assert f"<dc:title>{FRENCH_TITLE}</dc:title>" in opf
    assert opf.count("Translated using") == 1
    assert opf.count(GENERATOR_SOURCE) == 1
    assert f"<dc:description>{_signature().strip()}</dc:description>" in opf


@pytest.mark.asyncio
async def test_pipeline_metadata_failure_never_fails_the_job(
    tmp_path, input_epub, monkeypatch
):
    monkeypatch.setattr(translator_module, "EPUB_TRANSLATE_METADATA_ENABLED", True)

    async def _raise(**kwargs):
        raise RuntimeError("simulated failure inside the metadata pass")

    monkeypatch.setattr(translator_module, "translate_opf_metadata", _raise)

    output_epub = tmp_path / "output_failure.epub"
    events = await _translate(input_epub, output_epub, monkeypatch,
                              _pipeline_client())

    assert output_epub.exists()
    with zipfile.ZipFile(output_epub) as archive:
        assert "OEBPS/content.opf" in archive.namelist()
    event_names = [event for event, _ in events]
    assert "epub_metadata_translation_failed" in event_names
    assert "epub_save_success" in event_names
    # Step 6.6 still ran: the two passes are independent.
    assert "duokan-body-font" not in _read(output_epub, "OEBPS/content.opf")
