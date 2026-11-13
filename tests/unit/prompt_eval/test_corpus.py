"""Corpus schema and content checks for the prompt regression harness."""

from dataclasses import replace

import pytest

from tests.prompt_eval.corpus import load_corpus, validate_case
from tests.prompt_eval.model import Case

REQUIRED_IDS = {
    "T01_plain_paragraphs_en_fr",
    "T02_plain_repeated_quotes_en_fr",
    "T03_retry_variant_en_fr",
    "T04_dialogue_short_lines_en_fr",
    "T05_placeholders_en_fr",
    "T06_placeholders_dense_en_de",
    "T07_names_en_fr",
    "T08_glossary_en_fr",
    "T09_glossary_en_de",
    "T10_script_residue_ko_en",
    "T11_script_residue_en_ja",
    "T12_script_residue_ru_fr",
    "T13_text_cleanup_fr_en",
    "R01_clean_draft_fr",
    "R02_draft_with_english_sentence_fr",
    "R03_draft_placeholders_fr",
    "R04_draft_names_fr",
    "R05_good_draft_fr",
}

VALID = Case(
    id="X01",
    builder="translate",
    source_language="English",
    target_language="French",
    text="First paragraph with Alba in it.\n\nSecond paragraph.",
    has_placeholders=False,
    expect_paragraphs=True,
    must_keep=("Alba",),
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_committed_corpus_loads_and_has_every_required_case(corpus):
    ids = {c.id for c in corpus}
    assert REQUIRED_IDS <= ids


def test_corpus_texts_are_within_word_budget(corpus):
    for case in corpus:
        words = len(case.text.split())
        assert 60 <= words <= 250, f"{case.id}: {words} words"


def test_t03_is_t02_with_the_retry_option(corpus):
    by_id = {c.id: c for c in corpus}
    t02, t03 = by_id["T02_plain_repeated_quotes_en_fr"], by_id["T03_retry_variant_en_fr"]
    assert t03.text == t02.text
    assert t03.prompt_options == {"source_residue_retry": True}


def test_glossary_terms_occur_in_text(corpus):
    for case in corpus:
        for term in case.prompt_options.get("glossary_terms", {}):
            assert term.lower() in case.text.lower(), f"{case.id}: {term}"


def test_builder_and_case_filters(corpus):
    refine = load_corpus(builders=["refine"])
    assert refine and all(c.builder == "refine" for c in refine)
    assert [c.id for c in load_corpus(case_filter="^T0[12]_")] == [
        "T01_plain_paragraphs_en_fr", "T02_plain_repeated_quotes_en_fr"]


def test_valid_case_has_no_error():
    assert validate_case(VALID) == []


@pytest.mark.parametrize("change, fragment", [
    ({"builder": "summarise"}, "builder must be one of"),
    ({"has_placeholders": True, "text": "Only [id0] here."}, "at least two"),
    ({"has_placeholders": True, "text": "[id0] a [id2] b"}, "without gaps"),
    ({"has_placeholders": True, "text": "[id0] a [id1] b [id1]"}, "exactly once"),
    ({"text": "Alba [id0] and [id1].\n\nSecond."}, "has_placeholders is false"),
    ({"text": "Only one paragraph with Alba."}, "at least 2 paragraphs"),
    ({"must_keep": ("Zorvath",)}, "does not occur in text"),
    ({"length_ratio": (1.5, 1.0)}, "0 < lo < hi"),
    ({"length_ratio": (0.0, 1.0)}, "0 < lo < hi"),
    ({"target_language": "English"}, "different source and target"),
])
def test_validate_case_rules(change, fragment):
    errors = validate_case(replace(VALID, **change))
    assert any(fragment in e for e in errors), errors


def test_refine_case_may_use_same_language():
    case = replace(VALID, builder="refine", source_language="French", target_language="French")
    assert validate_case(case) == []


def _write(tmp_path, name, body):
    (tmp_path / name).write_text(body, encoding="utf-8")


CASE_YAML = """cases:
  - id: {id}
    builder: translate
    source_language: English
    target_language: French
    has_placeholders: false
    text: Some text.
"""


def test_duplicate_ids_raise(tmp_path):
    _write(tmp_path, "a.yaml", CASE_YAML.format(id="DUP"))
    _write(tmp_path, "b.yaml", CASE_YAML.format(id="DUP"))
    with pytest.raises(ValueError, match="Duplicate case ids"):
        load_corpus(corpus_dir=tmp_path)


def test_invalid_case_raises(tmp_path):
    _write(tmp_path, "a.yaml", CASE_YAML.format(id="BAD").replace("French", "English"))
    with pytest.raises(ValueError, match="Invalid corpus"):
        load_corpus(corpus_dir=tmp_path)


def test_unknown_key_raises(tmp_path):
    _write(tmp_path, "a.yaml", CASE_YAML.format(id="K") + "    colour: blue\n")
    with pytest.raises(ValueError, match="unknown case keys"):
        load_corpus(corpus_dir=tmp_path)


def test_lists_become_tuples(tmp_path):
    _write(tmp_path, "a.yaml", CASE_YAML.format(id="L") + "    must_contain_any:\n      - [a, b]\n"
           "    length_ratio: [0.5, 2]\n")
    (case,) = load_corpus(corpus_dir=tmp_path)
    assert case.must_contain_any == (("a", "b"),)
    assert case.length_ratio == (0.5, 2.0)
