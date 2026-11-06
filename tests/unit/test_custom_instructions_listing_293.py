"""Regression tests for issue #293: presets silently missing from the list.

A user with 5-6 hand-made presets only saw 3 of them: names with commas,
apostrophes, brackets or ampersands failed the filename whitelist, and `.txt`
files saved as Windows-1252 ("ANSI") failed UTF-8 decoding. Both were skipped
without any message.
"""

import pytest

from src.utils.custom_instructions import (
    is_safe_filename,
    list_custom_instructions,
    read_preset,
)


def test_listing_keeps_all_six_mixed_presets(tmp_path):
    (tmp_path / "Web Novel (v2).txt").write_text("one", encoding="utf-8")
    (tmp_path / "Romance, tome 1.txt").write_text("two", encoding="utf-8")
    (tmp_path / "L'été [draft].txt").write_text("three", encoding="utf-8")
    (tmp_path / "Série & co.yaml").write_text("translation: four\n", encoding="utf-8")
    (tmp_path / "ansi.txt").write_bytes("café, été".encode("cp1252"))
    (tmp_path / "plain.yml").write_text("refinement: six\n", encoding="utf-8")

    names = {e["filename"] for e in list_custom_instructions(tmp_path)}

    assert names == {
        "Web Novel (v2).txt",
        "Romance, tome 1.txt",
        "L'été [draft].txt",
        "Série & co.yaml",
        "ansi.txt",
        "plain.yml",
    }


def test_cp1252_text_preset_is_decoded(tmp_path):
    (tmp_path / "ansi.txt").write_bytes("café".encode("cp1252"))
    assert read_preset("ansi.txt", tmp_path)["translation"] == "café"


@pytest.mark.parametrize(
    "name",
    ["a/b.txt", "a\\b.txt", "../x.txt", ".hidden.txt", "a|b.txt", "a:b.yaml", "x.exe", "a\x01b.txt"],
)
def test_unsafe_filenames_still_rejected(name):
    assert not is_safe_filename(name)
