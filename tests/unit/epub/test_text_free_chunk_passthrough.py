"""Unit tests for text-free chunk passthrough and the empty-body guard.

A chunk that carries no translatable character (pure markup, a '==' separator,
a bare number, a CJK ellipsis) must never reach the LLM: the model has nothing
to translate, returns nothing usable, and the empty result used to wipe the
document body. The cover page of a CJK EPUB — whose whole body is
`<div><svg><image xlink:href="…"/></svg></div>` — was destroyed exactly this
way: TagPreserver reduces it to '[id0]', so the single chunk contained no text
at all.

Two independent defenses are tested here:
  1. `is_text_free_chunk` + the skip in the chunk loop (nothing is sent).
  2. `replace_body_content` refusing to empty a populated <body>.
"""
import pytest
from lxml import etree
from unittest.mock import MagicMock

from src.common.placeholder_format import PlaceholderFormat
from src.core.epub.body_serializer import extract_body_html, replace_body_content
from src.core.epub.exceptions import BodyExtractionError
from src.core.epub.html_chunker import HtmlChunker
from src.core.epub.html_utils import is_text_free_chunk
from src.core.epub.tag_preservation import TagPreserver
from src.core.epub.translation_metrics import TranslationMetrics
from tests.epub_engine_helpers import run_chunk_loop, translate_xhtml
from src.core.llm.base import LLMResponse
from src.persistence.checkpoint_manager import CheckpointManager


# The real cover page markup from the reported book (issue F2).
COVER_XHTML = '''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
    <head><title>封面</title></head>
    <body>
        <div>
            <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="100%" height="100%" viewBox="0 0 1200 1600" preserveAspectRatio="none">
                <image width="1200" height="1600" xlink:href="../Images/cover.jpg"/>
            </svg>
        </div>
    </body>
</html>
'''

PLACEHOLDER_TUPLE = ('[id', ']')


def _parse(xhtml: str) -> etree._Element:
    parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
    return etree.fromstring(xhtml.encode('utf-8'), parser)


# ---------------------------------------------------------------------------
# 1. The predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_text, expected", [
    ('[id0]', True),                    # cover page: pure markup
    ('[id0]==[id1]', True),             # separator paragraph
    ('[id0]……[id1]', True),             # CJK ellipsis only
    ('[id0]第1章[id1]', False),          # has letters
    ('[id0]2024[id1]', True),           # digits are not letters
    ('[id0]归墟，海中无底之谷。[id1]', False),  # normal CJK prose
    ('[id0]The quick brown fox.[id1]', False),  # normal Latin prose
])
def test_is_text_free_chunk_table(chunk_text, expected):
    assert is_text_free_chunk(chunk_text, PLACEHOLDER_TUPLE) is expected


def test_is_text_free_chunk_defaults_to_config_format():
    assert is_text_free_chunk('[id0]') is True
    assert is_text_free_chunk('[id0]texte[id1]') is False


def test_is_text_free_chunk_accepts_placeholder_format_instance():
    fmt = PlaceholderFormat.from_config()
    assert is_text_free_chunk('[id0]', fmt) is True
    assert is_text_free_chunk('[id0]texte[id1]', fmt) is False


def test_is_text_free_chunk_on_empty_and_whitespace():
    assert is_text_free_chunk('', PLACEHOLDER_TUPLE) is True
    assert is_text_free_chunk('   \n ', PLACEHOLDER_TUPLE) is True


# ---------------------------------------------------------------------------
# 2. Cover-page regression through the real extraction pipeline
# ---------------------------------------------------------------------------

def test_cover_page_produces_a_single_text_free_chunk():
    body_html, body_element = extract_body_html(_parse(COVER_XHTML))
    assert body_element is not None
    assert 'xlink:href="../Images/cover.jpg"' in body_html

    text_with_placeholders, tag_map = TagPreserver().preserve_tags(body_html)
    chunks = HtmlChunker(max_tokens=400).chunk_html_with_placeholders(
        text_with_placeholders, tag_map
    )

    assert len(chunks) == 1
    assert is_text_free_chunk(chunks[0]['text'], PLACEHOLDER_TUPLE) is True


# ---------------------------------------------------------------------------
# 3. End-to-end: the cover page survives a translation with no LLM available
# ---------------------------------------------------------------------------

def _exploding_llm_client():
    """An LLM client that fails the test if it is ever called."""
    client = MagicMock()

    async def explode(*args, **kwargs):
        raise AssertionError("the LLM must not be called for a text-free chunk")

    client.generate = explode
    client.extract_translation = lambda response: response
    return client


@pytest.mark.asyncio
async def test_cover_page_survives_translation_without_any_llm_call():
    doc_root = _parse(COVER_XHTML)

    success, stats = await translate_xhtml(
        doc_root=doc_root,
        source_language="Chinese",
        target_language="French",
        model_name="test-model",
        llm_client=_exploding_llm_client(),
        max_tokens_per_chunk=400,
    )

    assert success is True

    body = doc_root.find('.//{http://www.w3.org/1999/xhtml}body')
    serialized = etree.tostring(body, encoding='unicode')

    assert len(body.findall('.//{http://www.w3.org/1999/xhtml}div')) == 1
    assert serialized.count('<svg') == 1
    assert serialized.count('image') >= 1
    assert '../Images/cover.jpg' in serialized

    # The xlink href must survive namespace round-tripping.
    image = body.find('.//{http://www.w3.org/2000/svg}image')
    assert image is not None
    assert image.get('{http://www.w3.org/1999/xlink}href') == '../Images/cover.jpg'

    # The chunk still counts as completed, but no retry/fallback was recorded.
    assert stats.processed_chunks == 1
    assert stats.successful_first_try == 1
    assert stats.retry_attempts == 0
    assert stats.fallback_used == 0
    assert stats.token_alignment_used == 0


# ---------------------------------------------------------------------------
# 4. The body guard
# ---------------------------------------------------------------------------

def test_replace_body_content_refuses_to_empty_a_populated_body():
    doc = _parse('<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                 '<div><p>content</p></div></body></html>')
    body = doc.find('.//{http://www.w3.org/1999/xhtml}body')
    before = etree.tostring(body, encoding='unicode')

    with pytest.raises(BodyExtractionError):
        replace_body_content(body, "")

    assert len(body) == 1
    assert etree.tostring(body, encoding='unicode') == before


def test_replace_body_content_refuses_whitespace_only_replacement():
    doc = _parse('<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                 '<p>content</p></body></html>')
    body = doc.find('.//{http://www.w3.org/1999/xhtml}body')

    with pytest.raises(BodyExtractionError):
        replace_body_content(body, "  \n  ")

    assert len(body) == 1


def test_replace_body_content_refuses_to_drop_bare_body_text():
    doc = _parse('<html xmlns="http://www.w3.org/1999/xhtml">'
                 '<body>bare text</body></html>')
    body = doc.find('.//{http://www.w3.org/1999/xhtml}body')

    with pytest.raises(BodyExtractionError):
        replace_body_content(body, "")

    assert body.text == 'bare text'


def test_replace_body_content_empty_to_empty_is_a_no_op():
    doc = _parse('<html xmlns="http://www.w3.org/1999/xhtml"><body/></html>')
    body = doc.find('.//{http://www.w3.org/1999/xhtml}body')

    replace_body_content(body, "")  # must not raise

    assert len(body) == 0
    assert not (body.text or "").strip()


# ---------------------------------------------------------------------------
# 5. Interrupt / resume parity with a text-free chunk in the mix
# ---------------------------------------------------------------------------

# Chunk 0 is pure markup (skipped), chunks 1-3 are real prose.
MIXED_CHUNKS = [
    {'text': '[id0]', 'local_tag_map': {'[id0]': '<div><svg/></div>'},
     'global_indices': [0]},
    {'text': 'Premier paragraphe source.', 'local_tag_map': {}, 'global_indices': []},
    {'text': 'Deuxieme paragraphe source.', 'local_tag_map': {}, 'global_indices': []},
    {'text': 'Troisieme paragraphe source.', 'local_tag_map': {}, 'global_indices': []},
]

MIXED_RESPONSES = {
    'Premier paragraphe source.': 'First source paragraph.',
    'Deuxieme paragraphe source.': 'Second source paragraph.',
    'Troisieme paragraphe source.': 'Third source paragraph.',
}


def _deterministic_llm_client(call_log):
    """LLM client whose answer depends only on the chunk inside the prompt."""
    client = MagicMock()

    async def generate(user_prompt, system_prompt=None, **kwargs):
        for source, translation in MIXED_RESPONSES.items():
            if source in user_prompt:
                call_log.append(source)
                return LLMResponse(
                    content=translation,
                    prompt_tokens=10,
                    completion_tokens=10,
                    context_used=20,
                    context_limit=4096,
                    was_truncated=False,
                )
        raise AssertionError(f"unexpected chunk sent to the LLM: {user_prompt!r}")

    client.generate = generate
    client.extract_translation = lambda response: response
    return client


@pytest.fixture
def temp_checkpoint_manager(tmp_path):
    """Checkpoint manager with isolated storage (same pattern as
    tests/test_xhtml_chunk_interruption.py)."""
    manager = CheckpointManager(db_path=str(tmp_path / "test_jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir(parents=True, exist_ok=True)
    return manager


async def _run_mixed(checkpoint_manager, translation_id, file_href, call_log,
                     check_interruption=None, resume_state=None):
    kwargs = dict(
        chunks=MIXED_CHUNKS,
        source_language="French",
        target_language="English",
        model_name="test-model",
        llm_client=_deterministic_llm_client(call_log),
        max_retries=1,
        context_manager=None,
        placeholder_format=PLACEHOLDER_TUPLE,
        checkpoint_manager=checkpoint_manager,
        translation_id=translation_id,
        file_href=file_href,
        file_path=file_href,
        check_interruption_callback=check_interruption,
    )
    if resume_state is None:
        kwargs.update(start_chunk_index=0, translated_chunks=None,
                      global_tag_map={}, stats=None)
    else:
        kwargs.update(
            start_chunk_index=resume_state.current_chunk_index,
            translated_chunks=list(resume_state.translated_chunks),
            global_tag_map=resume_state.global_tag_map,
            stats=TranslationMetrics.from_dict(resume_state.stats),
        )
    return await run_chunk_loop(**kwargs)


@pytest.mark.asyncio
async def test_text_free_chunk_is_not_sent_to_the_llm(temp_checkpoint_manager):
    call_log = []
    translated, stats, was_interrupted = await _run_mixed(
        temp_checkpoint_manager, "tf_plain", "OEBPS/mixed.xhtml", call_log
    )

    assert was_interrupted is False
    assert call_log == list(MIXED_RESPONSES)  # the markup chunk never appears
    assert translated[0] == '[id0]'
    # Every chunk counts as completed, so the progress totals stay exact.
    assert stats.processed_chunks == len(MIXED_CHUNKS)
    assert stats.successful_first_try == len(MIXED_CHUNKS)


@pytest.mark.asyncio
async def test_interrupted_run_resumes_identically(temp_checkpoint_manager):
    file_href = "OEBPS/mixed.xhtml"

    # Reference: uninterrupted run.
    reference, _, _ = await _run_mixed(
        temp_checkpoint_manager, "tf_reference", file_href, []
    )

    # Interrupt right after the text-free chunk (no new work is launched once
    # the callback returns True).
    call_log = []

    def interrupt_after_first_llm_call():
        return len(call_log) >= 1

    partial, _, was_interrupted = await _run_mixed(
        temp_checkpoint_manager, "tf_resume", file_href, call_log,
        check_interruption=interrupt_after_first_llm_call,
    )

    assert was_interrupted is True
    assert 0 < len(partial) < len(MIXED_CHUNKS)
    assert partial[0] == '[id0]'  # the skipped chunk was persisted

    state = temp_checkpoint_manager.load_xhtml_partial_state("tf_resume", file_href)
    assert state is not None
    assert state.current_chunk_index == len(partial)

    resumed, _, was_interrupted = await _run_mixed(
        temp_checkpoint_manager, "tf_resume", file_href, [], resume_state=state
    )

    assert was_interrupted is False
    assert resumed == reference
