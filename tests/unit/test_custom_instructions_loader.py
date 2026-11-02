"""Tests for the custom instructions loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.custom_instructions import (
    is_safe_filename,
    list_custom_instructions,
    load_custom_instructions,
)


@pytest.fixture
def instructions_dir(tmp_path: Path) -> Path:
    """Provide an empty Custom_Instructions directory."""
    d = tmp_path / "Custom_Instructions"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# is_safe_filename
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("foo.txt", True),
        ("foo_bar.yaml", True),
        ("foo-bar.yml", True),
        ("noir_detective.yaml", True),
        ("foo.md", False),
        ("foo.json", False),
        ("../escape.txt", False),
        ("foo/bar.txt", False),
        ("foo\\bar.txt", False),
        # Hand-renamed files must stay listed (#293): spaces, parentheses,
        # any script, upper-case extensions.
        ("foo bar.yaml", True),
        ("Romance (v2).txt", True),
        ("武侠.yaml", True),
        ("novela_ñandú.yml", True),
        ("FOO.YAML", True),
        (".hidden.yaml", False),
        (" leading.yaml", False),
        ("a:b.yaml", False),
        ("a*b.txt", False),
        ('a"b.yaml', False),
        ("", False),
        ("foo", False),
    ],
)
def test_is_safe_filename(name: str, expected: bool) -> None:
    assert is_safe_filename(name) is expected


# ---------------------------------------------------------------------------
# load_custom_instructions: .txt legacy
# ---------------------------------------------------------------------------

def test_txt_legacy_applies_to_both_phases(instructions_dir: Path) -> None:
    (instructions_dir / "noir.txt").write_text(
        "Hardboiled noir. Cynical voice.\n", encoding="utf-8"
    )
    result = load_custom_instructions("noir.txt", instructions_dir)
    assert result["translation"] == "Hardboiled noir. Cynical voice."
    assert result["refinement"] == "Hardboiled noir. Cynical voice."


def test_txt_empty_returns_none_for_both(instructions_dir: Path) -> None:
    (instructions_dir / "blank.txt").write_text("   \n\n", encoding="utf-8")
    result = load_custom_instructions("blank.txt", instructions_dir)
    assert result == {"translation": None, "refinement": None}


# ---------------------------------------------------------------------------
# load_custom_instructions: YAML
# ---------------------------------------------------------------------------

def test_yaml_both_phases(instructions_dir: Path) -> None:
    (instructions_dir / "noir.yaml").write_text(
        "translation: |\n  Translate noir.\nrefinement: |\n  Polish noir.\n",
        encoding="utf-8",
    )
    result = load_custom_instructions("noir.yaml", instructions_dir)
    assert result["translation"].strip() == "Translate noir."
    assert result["refinement"].strip() == "Polish noir."


def test_yaml_translation_only(instructions_dir: Path) -> None:
    (instructions_dir / "t.yaml").write_text(
        "translation: |\n  Only at translate time.\n", encoding="utf-8"
    )
    result = load_custom_instructions("t.yaml", instructions_dir)
    assert result["translation"].strip() == "Only at translate time."
    assert result["refinement"] is None


def test_yaml_refinement_only(instructions_dir: Path) -> None:
    (instructions_dir / "r.yml").write_text(
        "refinement: Polish only.\n", encoding="utf-8"
    )
    result = load_custom_instructions("r.yml", instructions_dir)
    assert result["translation"] is None
    assert result["refinement"] == "Polish only."


def test_yaml_empty_string_field_normalises_to_none(instructions_dir: Path) -> None:
    (instructions_dir / "e.yaml").write_text(
        "translation: ''\nrefinement: '   '\n", encoding="utf-8"
    )
    result = load_custom_instructions("e.yaml", instructions_dir)
    assert result == {"translation": None, "refinement": None}


def test_yaml_top_level_list_rejected(instructions_dir: Path) -> None:
    (instructions_dir / "bad.yaml").write_text(
        "- one\n- two\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_custom_instructions("bad.yaml", instructions_dir)


def test_yaml_malformed_raises(instructions_dir: Path) -> None:
    (instructions_dir / "broken.yaml").write_text(
        "translation: |\n  unclosed\n  - item\nbad:\n   :\n", encoding="utf-8"
    )
    with pytest.raises(yaml.YAMLError):
        load_custom_instructions("broken.yaml", instructions_dir)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def test_unsafe_filename_rejected(instructions_dir: Path) -> None:
    with pytest.raises(ValueError):
        load_custom_instructions("../escape.txt", instructions_dir)


def test_unsupported_extension_rejected(instructions_dir: Path) -> None:
    (instructions_dir / "foo.md").write_text("ignored", encoding="utf-8")
    with pytest.raises(ValueError):
        load_custom_instructions("foo.md", instructions_dir)


def test_missing_file_raises(instructions_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_custom_instructions("ghost.txt", instructions_dir)


# ---------------------------------------------------------------------------
# list_custom_instructions
# ---------------------------------------------------------------------------

def test_list_reports_phase_availability(instructions_dir: Path) -> None:
    (instructions_dir / "both.txt").write_text("Legacy.", encoding="utf-8")
    (instructions_dir / "t_only.yaml").write_text(
        "translation: Hi.\n", encoding="utf-8"
    )
    (instructions_dir / "r_only.yml").write_text(
        "refinement: Hello.\n", encoding="utf-8"
    )
    (instructions_dir / "full.yaml").write_text(
        "translation: T.\nrefinement: R.\n", encoding="utf-8"
    )

    entries = list_custom_instructions(instructions_dir)
    by_name = {e["display_name"]: e for e in entries}

    assert by_name["both"]["format"] == "txt"
    assert by_name["both"]["has_translation"] is True
    assert by_name["both"]["has_refinement"] is True

    assert by_name["t_only"]["format"] == "yaml"
    assert by_name["t_only"]["has_translation"] is True
    assert by_name["t_only"]["has_refinement"] is False

    assert by_name["r_only"]["format"] == "yaml"
    assert by_name["r_only"]["has_translation"] is False
    assert by_name["r_only"]["has_refinement"] is True

    assert by_name["full"]["has_translation"] is True
    assert by_name["full"]["has_refinement"] is True


def test_list_silently_skips_malformed(instructions_dir: Path) -> None:
    (instructions_dir / "ok.yaml").write_text("translation: OK.\n", encoding="utf-8")
    (instructions_dir / "broken.yaml").write_text(": : : :\nbad:\n  ::\n", encoding="utf-8")

    entries = list_custom_instructions(instructions_dir)
    display_names = {e["display_name"] for e in entries}
    assert "ok" in display_names
    assert "broken" not in display_names


def test_list_sorted_alphabetically(instructions_dir: Path) -> None:
    for name in ("zeta.yaml", "alpha.yaml", "mu.yaml"):
        (instructions_dir / name).write_text("translation: x\n", encoding="utf-8")
    entries = list_custom_instructions(instructions_dir)
    assert [e["display_name"] for e in entries] == ["alpha", "mu", "zeta"]


def test_list_empty_dir_returns_empty(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist"
    assert list_custom_instructions(nonexistent) == []
