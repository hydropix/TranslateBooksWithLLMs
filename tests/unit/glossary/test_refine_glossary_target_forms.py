"""The refinement pass must match the glossary on its target forms.

Refinement filtered the glossary (source-language keys) against the
target-language draft, so for e.g. ZH->FR nothing ever matched and the refine
pass got no glossary at all.
"""

from src.core.translator import _build_chunk_glossary_block

TERMS = {"李明": "Li Ming", "青云山": "le mont Qingyun", "王": "Wang"}


def test_zh_to_fr_draft_matches_target_forms():
    draft = "Li Ming gravit le mont Qingyun avant l'aube."
    block = _build_chunk_glossary_block(
        draft, {"glossary_terms": TERMS}, target_language="French",
        match_target_forms=True,
    )
    assert "李明 -> Li Ming" in block
    assert "青云山 -> le mont Qingyun" in block
    assert "Wang" not in block
    assert "KEEP THESE EXACT FORMS" in block


def test_source_key_matching_is_unchanged_for_translation():
    block = _build_chunk_glossary_block(
        "李明走进了房间。", {"glossary_terms": TERMS}, target_language="French",
    )
    assert "李明 -> Li Ming" in block
    assert "REQUIRED TRANSLATIONS" in block


def test_target_form_matching_finds_nothing_in_a_source_text():
    block = _build_chunk_glossary_block(
        "李明走进了房间。", {"glossary_terms": TERMS}, target_language="French",
        match_target_forms=True,
    )
    assert block == ""
