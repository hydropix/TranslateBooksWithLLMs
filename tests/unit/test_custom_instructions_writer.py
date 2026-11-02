"""Tests for the extended-schema preset persistence layer.

Covers `write_preset`, `read_preset`, `delete_preset`, `resolve_inside`,
`slugify_preset_name`, and the three new keys added to
`list_custom_instructions`. The plain loader contract itself is covered by
`test_custom_instructions_loader.py`, which this suite does not touch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.custom_instructions import (
    delete_preset,
    filename_for_name,
    list_custom_instructions,
    load_custom_instructions,
    read_preset,
    resolve_inside,
    slugify_preset_name,
    write_preset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def instructions_dir(tmp_path: Path) -> Path:
    """Provide an empty Custom_Instructions directory."""
    d = tmp_path / "Custom_Instructions"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# 1. write -> load_custom_instructions round-trip
# ---------------------------------------------------------------------------

def test_write_then_load_round_trips_both_phases(instructions_dir: Path) -> None:
    payload = {
        "description": "Hardboiled register",
        "mode": "source",
        "rules": [{"dimension": "register", "instruction": "Stay cynical."}],
        "translation": "Match the following writing style in the translation.",
        "refinement": "Polish the translation to match the following writing style.",
    }
    write_preset("noir.yaml", payload, instructions_dir)

    loaded = load_custom_instructions("noir.yaml", instructions_dir)
    assert loaded["translation"] == payload["translation"]
    assert loaded["refinement"] == payload["refinement"]


# ---------------------------------------------------------------------------
# 2. write -> read_preset round-trip on rules and metadata
# ---------------------------------------------------------------------------

def test_write_then_read_preset_round_trips_metadata(instructions_dir: Path) -> None:
    payload = {
        "description": "Hardboiled register extracted from 2 books",
        "mode": "source",
        "source_files": ["novel_a.epub", "novel_b.epub"],
        "generated_at": "2026-08-02T14:12:00Z",
        "rules": [
            {"dimension": "register", "instruction": "Keep a cynical tone."},
            {"dimension": "sentence_rhythm", "instruction": "Favor short sentences."},
        ],
        "translation": "Match the style.",
        "refinement": "Polish to match the style.",
    }
    write_preset("full.yaml", payload, instructions_dir)

    preset = read_preset("full.yaml", instructions_dir)
    assert preset["description"] == payload["description"]
    assert preset["mode"] == payload["mode"]
    assert preset["source_files"] == payload["source_files"]
    assert preset["generated_at"] == payload["generated_at"]
    assert preset["rules"] == payload["rules"]
    assert preset["translation"] == payload["translation"]
    assert preset["refinement"] == payload["refinement"]
    assert preset["format"] == "yaml"
    assert preset["filename"] == "full.yaml"
    assert preset["display_name"] == "full"


# ---------------------------------------------------------------------------
# 3. multi-line prose with ':' and '#' survives the round trip
# ---------------------------------------------------------------------------

def test_multiline_prose_with_colon_and_hash_survives(instructions_dir: Path) -> None:
    translation = (
        "Rules to apply:\n"
        "- Keep it terse. # not a comment, this is prose\n"
        "- Use a colon: like this one, freely."
    )
    write_preset("special_chars.yaml", {"translation": translation}, instructions_dir)

    loaded = load_custom_instructions("special_chars.yaml", instructions_dir)
    assert loaded["translation"] == translation


# ---------------------------------------------------------------------------
# 4. non-ASCII prose survives with allow_unicode=True
# ---------------------------------------------------------------------------

def test_non_ascii_prose_survives(instructions_dir: Path) -> None:
    translation = "Garde un ton feutré et mélancolique, à la française."
    refinement = "保持忧郁而细腻的语气，如同古老的物语。"
    write_preset(
        "unicode.yaml",
        {"translation": translation, "refinement": refinement},
        instructions_dir,
    )

    raw = (instructions_dir / "unicode.yaml").read_text(encoding="utf-8")
    assert "\\u" not in raw  # allow_unicode=True: no escape sequences

    loaded = load_custom_instructions("unicode.yaml", instructions_dir)
    assert loaded["translation"] == translation
    assert loaded["refinement"] == refinement


# ---------------------------------------------------------------------------
# 4b. context: write -> read round trip, multi-line/accented, and omission
# ---------------------------------------------------------------------------

def test_context_write_then_read_round_trips(instructions_dir: Path) -> None:
    payload = {
        "description": "Hardboiled register",
        "context": "A rain-soaked 1940s American city, no smartphones, no internet.",
        "translation": "Match the style.",
    }
    write_preset("with_context.yaml", payload, instructions_dir)

    preset = read_preset("with_context.yaml", instructions_dir)
    assert preset["context"] == payload["context"]


def test_multiline_accented_context_survives_round_trip(instructions_dir: Path) -> None:
    context = (
        "Un vieux port breton, années 1950 :\n"
        "pas d'électricité dans les hameaux, pêche à la voile, hiver rude."
    )
    write_preset("context_multiline.yaml", {"context": context}, instructions_dir)

    preset = read_preset("context_multiline.yaml", instructions_dir)
    assert preset["context"] == context


def test_write_without_context_omits_context_key(instructions_dir: Path) -> None:
    write_preset(
        "no_context.yaml",
        {"description": "No setting given", "translation": "Match the style."},
        instructions_dir,
    )

    raw = (instructions_dir / "no_context.yaml").read_text(encoding="utf-8")
    assert "context" not in yaml.safe_load(raw)
    assert "context:" not in raw


def test_write_with_empty_context_omits_context_key(instructions_dir: Path) -> None:
    write_preset(
        "empty_context.yaml",
        {"context": "", "translation": "Match the style."},
        instructions_dir,
    )

    raw = (instructions_dir / "empty_context.yaml").read_text(encoding="utf-8")
    assert "context" not in yaml.safe_load(raw)


def test_overwrite_without_touching_context_preserves_it(instructions_dir: Path) -> None:
    write_preset(
        "keep_context.yaml",
        {"context": "A remote alpine village, pre-industrial.", "translation": "First."},
        instructions_dir,
    )

    write_preset(
        "keep_context.yaml",
        {"translation": "Second."},
        instructions_dir,
        overwrite=True,
    )

    preset = read_preset("keep_context.yaml", instructions_dir)
    assert preset["context"] == "A remote alpine village, pre-industrial."
    assert preset["translation"] == "Second."


def test_read_preset_defaults_context_to_empty_string(instructions_dir: Path) -> None:
    path = instructions_dir / "legacy.yaml"
    path.write_text("translation: Hello.\n", encoding="utf-8")

    preset = read_preset("legacy.yaml", instructions_dir)
    assert preset["context"] == ""


def test_legacy_txt_preset_reports_empty_context(instructions_dir: Path) -> None:
    path = instructions_dir / "legacy.txt"
    path.write_text("Some plain instructions.", encoding="utf-8")

    preset = read_preset("legacy.txt", instructions_dir)
    assert preset["context"] == ""


# ---------------------------------------------------------------------------
# 5. overwrite=False on an existing file -> FileExistsError
# ---------------------------------------------------------------------------

def test_write_without_overwrite_raises_on_existing_file(instructions_dir: Path) -> None:
    write_preset("dup.yaml", {"translation": "First."}, instructions_dir)

    with pytest.raises(FileExistsError):
        write_preset("dup.yaml", {"translation": "Second."}, instructions_dir, overwrite=False)


# ---------------------------------------------------------------------------
# 6. overwrite preserves an unknown top-level key
# ---------------------------------------------------------------------------

def test_overwrite_preserves_unknown_key(instructions_dir: Path) -> None:
    path = instructions_dir / "custom.yaml"
    path.write_text(
        "translation: Original.\ncustom_field: 42\n", encoding="utf-8"
    )

    write_preset(
        "custom.yaml",
        {"translation": "Updated."},
        instructions_dir,
        overwrite=True,
    )

    raw = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    assert parsed["custom_field"] == 42
    assert parsed["translation"] == "Updated."

    # Unknown key appears after the known PRESET_KEY_ORDER keys.
    keys = list(parsed.keys())
    assert keys.index("translation") < keys.index("custom_field")


# ---------------------------------------------------------------------------
# 7. path traversal / unsafe filenames raise ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_name", ["../evil.yaml", "/etc/passwd", "evil.sh"])
def test_unsafe_filenames_rejected_everywhere(instructions_dir: Path, bad_name: str) -> None:
    with pytest.raises(ValueError):
        write_preset(bad_name, {"translation": "x"}, instructions_dir)
    with pytest.raises(ValueError):
        read_preset(bad_name, instructions_dir)
    with pytest.raises(ValueError):
        delete_preset(bad_name, instructions_dir)


# ---------------------------------------------------------------------------
# 8. atomicity: no .tmp file left behind, success or failure
# ---------------------------------------------------------------------------

def test_no_tmp_file_remains_after_success(instructions_dir: Path) -> None:
    write_preset("atomic.yaml", {"translation": "Hello."}, instructions_dir)
    tmp_files = list(instructions_dir.glob("*.tmp"))
    assert tmp_files == []
    assert (instructions_dir / "atomic.yaml").exists()


def test_no_tmp_file_remains_after_simulated_dump_failure(
    instructions_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.utils.custom_instructions as ci_module

    def boom(*args, **kwargs):
        raise RuntimeError("simulated safe_dump failure")

    monkeypatch.setattr(ci_module.yaml, "safe_dump", boom)

    with pytest.raises(RuntimeError):
        write_preset("boom.yaml", {"translation": "Hello."}, instructions_dir)

    tmp_files = list(instructions_dir.glob("*.tmp"))
    assert tmp_files == []
    assert not (instructions_dir / "boom.yaml").exists()


# ---------------------------------------------------------------------------
# 9. slugify_preset_name
# ---------------------------------------------------------------------------

def test_slugify_preset_name_collapses_and_strips() -> None:
    assert slugify_preset_name("Mon style — noir!") == "Mon_style_noir"


def test_slugify_preset_name_keeps_non_latin_scripts() -> None:
    """A Chinese/Japanese/Korean name must not slugify to empty (#296)."""
    assert slugify_preset_name("新建空白风格") == "新建空白风格"
    assert slugify_preset_name("空のスタイル 2") == "空のスタイル_2"
    assert slugify_preset_name("Nuevo estilo ñandú") == "Nuevo_estilo_ñandú"


def test_slugify_preset_name_empty_raises() -> None:
    with pytest.raises(ValueError):
        slugify_preset_name("!!!")


def test_filename_for_name_appends_yaml_extension() -> None:
    assert filename_for_name("My Style") == "My_Style.yaml"


# ---------------------------------------------------------------------------
# 10. list_custom_instructions on the presets shipped in the repository
# ---------------------------------------------------------------------------

def test_list_shipped_presets_has_five_original_and_three_new_keys() -> None:
    """The three new keys must appear on every entry of the real preset folder.

    The folder's contents are curated over time, so this asserts on the shape
    of each entry rather than on how many presets happen to ship today — and
    skips when the repository ships none at all, which is a valid state: the
    app creates the folder on demand and `/api/custom-instructions` answers
    `folder_not_found` until then. The shape itself is covered on synthetic
    presets by the tests above; this one only guards the real folder.
    """
    custom_instructions_dir = REPO_ROOT / "Custom_Instructions"
    if not custom_instructions_dir.is_dir():
        pytest.skip("the repository ships no Custom_Instructions folder")

    entries = list_custom_instructions(custom_instructions_dir)
    if not entries:
        pytest.skip("the repository ships no preset to inspect")

    original_keys = {"filename", "display_name", "format", "has_translation", "has_refinement"}
    new_keys = {"description", "mode", "updated_at"}

    for entry in entries:
        assert original_keys.issubset(entry.keys())
        assert new_keys.issubset(entry.keys())
        assert isinstance(entry["description"], str)
        assert entry["mode"] is None or isinstance(entry["mode"], str)
        assert isinstance(entry["updated_at"], str)


# ---------------------------------------------------------------------------
# resolve_inside: extra coverage for the renamed public helper
# ---------------------------------------------------------------------------

def test_resolve_inside_rejects_escape(instructions_dir: Path) -> None:
    assert resolve_inside(instructions_dir, "../escape.yaml") is None


def test_resolve_inside_accepts_valid_name(instructions_dir: Path) -> None:
    result = resolve_inside(instructions_dir, "ok.yaml")
    assert result == instructions_dir / "ok.yaml"


def test_delete_preset_returns_false_when_missing(instructions_dir: Path) -> None:
    assert delete_preset("ghost.yaml", instructions_dir) is False


def test_delete_preset_removes_existing_file(instructions_dir: Path) -> None:
    write_preset("to_delete.yaml", {"translation": "x"}, instructions_dir)
    assert delete_preset("to_delete.yaml", instructions_dir) is True
    assert not (instructions_dir / "to_delete.yaml").exists()
