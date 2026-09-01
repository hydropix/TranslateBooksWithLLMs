"""Diagnosis tests for F3: `class="head"` dropped from every translated heading.

Finding F3 of `plan/PLAN_CjkSourceRendering.md` measured
`LOST ATTRS: {('h3','class'): 28}` over a real Chinese -> French EPUB: every
translated `<h3 class="head">` came out as a bare `<h3>`.

These tests localize the loss (and its opposite, the preservation controls).
Both translation paths of the EPUB pipeline are exercised on the same body
HTML, with an LLM client stubbed at the transport level only (`generate` /
`extract_translation`); every module under test is the real one, so the
assertions describe production behaviour:

  * Plain Text Mode (`prompt_options['plain_text_mode']`) PRESERVES the
    attribute. `plain_extractor.extract_plain_paragraphs` records each block's
    tag NAME *and* attributes, and `replace_body_with_paragraphs` carries the
    attributes onto the rebuilt element (see plain_extractor.py). This is the
    regression test below.

  * The placeholder path (the default) PRESERVES the attribute, both when the
    model echoes the placeholders and when it emits literal HTML tags instead
    and the token-alignment fallback has to repair the chunk. Those two are
    passing controls, and they pin any future loss to Plain Text Mode rather
    than to `TagPreserver`, `PlaceholderRenumberer`, `TokenAlignmentFallback`
    or the entity-escaping step of `_reconstruct_html`.
"""
import re
from typing import List

import pytest
from lxml import etree

from src.core.epub.epub_translation_adapter import EpubTranslationAdapter
from src.core.epub.xhtml_translator import translate_xhtml_simplified
from src.core.llm.base import LLMResponse


# The reproduction input of finding F3, taken verbatim from the reported book
# (OEBPS/Text/chapter_*.html of the Chinese source): a `class="head"` heading
# followed by one paragraph.
BODY_HTML = (
    '<h3 class="head">第1章</h3>\n'
    '<p>归墟，海中无底之谷。</p>'
)

XHTML_DOC = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<html xmlns="http://www.w3.org/1999/xhtml">'
    '<head><title>Chapter</title></head>'
    f'<body>{BODY_HTML}</body>'
    '</html>'
).encode('utf-8')

# What a model returns for the placeholder chunk '[id0]第1章[id1]...[id2]'
# when it behaves: same placeholders, same order, translated text between them.
WELL_BEHAVED_RESPONSE = (
    '<TRANSLATION>'
    '[id0]Chapitre 1[id1]Guixu, la vallée sans fond au milieu de la mer.[id2]'
    '</TRANSLATION>'
)

# What a model plausibly returns instead: the markup it inferred, as literal
# tags, with the placeholders dropped. The attributes it never saw are gone.
LITERAL_TAGS_RESPONSE = (
    '<TRANSLATION>'
    '<h3>Chapitre 1</h3>\n'
    '<p>Guixu, la vallée sans fond au milieu de la mer.</p>'
    '</TRANSLATION>'
)


class StubLLMClient:
    """Transport-level stub: replays canned responses, records the prompts.

    Only the two methods `src.core.translator` calls on a client are provided
    (`generate`, `extract_translation`), plus the `context_window` attribute the
    adaptive-context code reads. Nothing inside `src/core/epub/` is mocked.
    """

    def __init__(self, responses: List[str]):
        self._responses = responses
        self.context_window = 8192
        self.prompts: List[str] = []

    async def generate(self, prompt, system_prompt=None, **kwargs):
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self._responses) - 1)
        return LLMResponse(
            content=self._responses[index],
            prompt_tokens=1,
            completion_tokens=1,
            context_used=2,
            context_limit=self.context_window,
            was_truncated=False,
        )

    def extract_translation(self, response):
        match = re.search(r'<TRANSLATION>(.*)</TRANSLATION>', response, re.DOTALL)
        return match.group(1) if match else None


class EchoPlaceholdersClient(StubLLMClient):
    """Stub that keeps every `[idN]` of the prompt and translates the rest.

    Used by the plain-text case, where the prompt carries no placeholder at all
    and the response must simply be a plausible translation of each segment.
    """

    def __init__(self):
        super().__init__(responses=[''])

    async def generate(self, prompt, system_prompt=None, **kwargs):
        self.prompts.append(prompt)
        source = _source_text_of(prompt)
        pieces = []
        for piece in re.split(r'(\[id\d+\])', source):
            if re.fullmatch(r'\[id\d+\]', piece):
                pieces.append(piece)
            elif piece.strip():
                pieces.append('Texte traduit.')
        return LLMResponse(
            content='<TRANSLATION>' + ''.join(pieces) + '</TRANSLATION>',
            prompt_tokens=1,
            completion_tokens=1,
            context_used=2,
            context_limit=self.context_window,
            was_truncated=False,
        )


def _source_text_of(prompt: str) -> str:
    """Return the <SOURCE_TEXT> block of a translation prompt."""
    match = re.search(r'<SOURCE_TEXT>\n(.*)\n</SOURCE_TEXT>', prompt, re.DOTALL)
    return match.group(1) if match else prompt


def _parse_doc() -> etree._Element:
    parser = etree.XMLParser(encoding='utf-8', recover=True, remove_blank_text=False)
    return etree.fromstring(XHTML_DOC, parser)


def _heading_open_tags(doc_root: etree._Element) -> List[str]:
    """Serialized opening tags of every h1..h6 of the document, in order."""
    serialized = etree.tostring(doc_root, encoding='unicode')
    return re.findall(r'<h[1-6][^>]*>', serialized)


@pytest.mark.asyncio
async def test_placeholder_path_preserves_heading_class(monkeypatch):
    """Control: the default path keeps `class="head"` end to end.

    This is the round-trip the planner already verified on strings, re-run
    through `translate_xhtml_simplified` with a stubbed LLM: body extraction,
    tag preservation, chunking, placeholder renumbering, reconstruction and
    body reinjection all keep the attribute.
    """
    # setattr, not setenv: src/config.py reads this variable once at import
    # time, so setting the environment variable here has no effect on the
    # already-computed module attribute. translate_chunk_with_fallback re-reads
    # `src.config.EPUB_TOKEN_ALIGNMENT_ENABLED` on every call, which is what
    # makes patching the attribute work.
    monkeypatch.setattr('src.config.EPUB_TOKEN_ALIGNMENT_ENABLED', True)
    doc_root = _parse_doc()
    client = StubLLMClient([WELL_BEHAVED_RESPONSE])

    success, _stats = await translate_xhtml_simplified(
        doc_root=doc_root,
        source_language='Chinese',
        target_language='French',
        model_name='stub-model',
        llm_client=client,
        max_retries=1,
    )

    assert success is True
    # The model saw placeholders, not markup: the prompt must not leak the class.
    assert 'class="head"' not in client.prompts[0]
    assert _heading_open_tags(doc_root) == ['<h3 class="head">']
    assert 'Chapitre 1' in etree.tostring(doc_root, encoding='unicode')


@pytest.mark.asyncio
async def test_placeholder_path_preserves_heading_class_when_model_emits_literal_tags(monkeypatch):
    """Control: literal-tag output does not lose the attribute either.

    When the model answers with `<h3>Chapitre 1</h3>` instead of the
    placeholders, strict validation fails and the token-alignment fallback
    reinserts `[id0]`/`[id1]`/`[id2]` into the clean translation. The heading
    tag still comes from the tag map, so `class="head"` survives — which rules
    out "a recovery path accepts the model's literal tags" as the cause of F3.
    """
    # This test IS the token-alignment path, so it must not inherit the flag
    # from the developer's .env: with EPUB_TOKEN_ALIGNMENT_ENABLED=false the
    # chunk drops straight to Phase 3 and only one prompt is ever sent.
    monkeypatch.setattr('src.config.EPUB_TOKEN_ALIGNMENT_ENABLED', True)
    doc_root = _parse_doc()
    client = StubLLMClient([LITERAL_TAGS_RESPONSE])

    success, _stats = await translate_xhtml_simplified(
        doc_root=doc_root,
        source_language='Chinese',
        target_language='French',
        model_name='stub-model',
        llm_client=client,
        max_retries=1,
    )

    assert success is True
    # Two calls: the placeholder attempt, then the placeholder-free fallback.
    assert len(client.prompts) == 2
    assert _heading_open_tags(doc_root) == ['<h3 class="head">']


@pytest.mark.asyncio
async def test_plain_text_mode_preserves_heading_class():
    """Regression: Plain Text Mode keeps `class="head"` on the translated heading."""
    doc_root = _parse_doc()
    client = EchoPlaceholdersClient()
    adapter = EpubTranslationAdapter()

    success, _stats = await adapter.translate_content(
        raw_content=doc_root,
        structure_map={},
        context={},
        source_language='Chinese',
        target_language='French',
        model_name='stub-model',
        llm_client=client,
        max_tokens_per_chunk=2000,
        prompt_options={'plain_text_mode': True},
    )

    assert success is True
    # The heading text IS translated, so the block was processed; only the
    # attribute is missing. Asserted first so the xfail is about the attribute
    # and not about a broken harness.
    assert 'Texte traduit.' in etree.tostring(doc_root, encoding='unicode')
    assert _heading_open_tags(doc_root) == ['<h3 class="head">']
