"""Drop caps and styled initials stay attached to their word (issue #333)."""
import pytest

from src.core.epub.html_chunker import HtmlChunker
from src.core.epub.initial_letters import close_initial_letters, merge_initial_letters
from src.core.epub.tag_preservation import TagPreserver
from src.core.epub.xhtml_translator import PlaceholderManager, _reconstruct_html

DROPCAP_SPLIT = '<p class="noindent"><span class="dropcap">A</span>lice was beginning to get tired.</p>'
DROPCAP_WORD = '<p><span class="dropcap">A</span> good translation, it has been said.</p>'


@pytest.fixture(params=[False, True], ids=["plain", "technical"])
def preserve(request):
    preserver = TagPreserver(protect_technical=request.param)
    if request.param:
        return preserver, preserver.preserve_tags_and_technical_content
    return preserver, preserver.preserve_tags


@pytest.mark.parametrize("html, model_text", [
    (DROPCAP_SPLIT, "Alice was beginning to get tired."),
    (DROPCAP_WORD, "A good translation, it has been said."),
])
def test_model_sees_the_whole_word(preserve, html, model_text):
    _, preserve_fn = preserve
    text, _ = preserve_fn(html)
    assert model_text in text


@pytest.mark.parametrize("html", [DROPCAP_SPLIT, DROPCAP_WORD])
def test_untranslated_round_trip_is_identical(preserve, html):
    preserver, preserve_fn = preserve
    text, tag_map = preserve_fn(html)
    assert preserver.restore_tags(text, tag_map) == html


@pytest.mark.parametrize("html, source, translation, expected", [
    (DROPCAP_SPLIT, "Alice was beginning to get tired.", "Alice começava a se cansar.",
     '<p class="noindent"><span class="dropcap">A</span>lice começava a se cansar.</p>'),
    (DROPCAP_WORD, "A good translation, it has been said.", "Uma boa tradução, como se diz.",
     '<p><span class="dropcap">U</span>ma boa tradução, como se diz.</p>'),
    (DROPCAP_SPLIT, "Alice was beginning to get tired.", "« Alice commençait à fatiguer.",
     '<p class="noindent"><span class="dropcap">« A</span>lice commençait à fatiguer.</p>'),
])
def test_translation_first_letter_becomes_the_initial(preserve, html, source, translation, expected):
    preserver, preserve_fn = preserve
    text, tag_map = preserve_fn(html)
    assert preserver.restore_tags(text.replace(source, translation), tag_map) == expected


def test_model_drops_the_text_keeps_document_well_formed():
    preserver = TagPreserver()
    text, tag_map = preserver.preserve_tags(DROPCAP_SPLIT)
    restored = preserver.restore_tags(text.replace("Alice was beginning to get tired.", ""), tag_map)
    assert restored == '<p class="noindent"><span class="dropcap"></span></p>'


@pytest.mark.parametrize("html", [
    # Mid-sentence: a styled letter that does not start the block.
    '<p>He said <b>N</b>o way.</p>',
    # A whole word in bold at the start, without an initial class: emphasis.
    '<p><b>I</b> was there.</p>',
    # Three letters is a word, not an initial.
    '<p><span class="dropcap">The</span> end.</p>',
    # Followed by punctuation, not by a word.
    '<p><span class="dropcap">A</span>, b, c.</p>',
])
def test_non_initials_are_left_alone(html):
    assert merge_initial_letters(html) == html


def test_initial_after_nested_wrappers_is_detected():
    html = '<div class="chapter"><p><em><span class="initial">T</span>he</em> sea.</p></div>'
    merged = merge_initial_letters(html)
    assert '</span>he' not in merged
    assert close_initial_letters(merged) == html


def test_full_pipeline_through_chunker_and_reconstruction():
    """Preserve, chunk, translate each chunk, reconstruct: the initial follows the translation."""
    body = (
        '<div><h1>Chapter I</h1>'
        f'{DROPCAP_SPLIT}'
        '<p>Second paragraph.</p>'
        f'{DROPCAP_WORD}</div>'
    )
    preserver = TagPreserver()
    text, global_tag_map = preserver.preserve_tags(body)
    chunks = HtmlChunker(max_tokens=20).chunk_html_with_placeholders(text, global_tag_map)
    assert len(chunks) > 1

    translations = {
        "Alice was beginning to get tired.": "Alice começava a se cansar.",
        "A good translation, it has been said.": "Uma boa tradução, como se diz.",
    }
    translated_chunks = []
    for chunk in chunks:
        chunk_text = chunk['text']
        for source, target in translations.items():
            chunk_text = chunk_text.replace(source, target)
        translated_chunks.append(PlaceholderManager.restore_to_global(chunk_text, chunk['global_indices']))

    html = _reconstruct_html(translated_chunks, global_tag_map, preserver)

    assert '<span class="dropcap">A</span>lice começava' in html
    assert '<span class="dropcap">U</span>ma boa tradução' in html
    assert 'tbl-initial' not in html
