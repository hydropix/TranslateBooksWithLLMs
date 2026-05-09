"""
Loaders for the split benchmark data layout.

Reads:
  - `benchmark/data/languages/<code>.yaml`
  - `benchmark/data/reference_texts/<source_lang>/<id>.yaml`

Falls back to the legacy monolithic YAMLs (`benchmark/languages.yaml`,
`benchmark/reference_texts.yaml`) when the split layout is absent, so existing
runs keep working during the migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from .models import Language, LanguageCategory, ReferenceText


_CATEGORY_MAP = {
    "european_major": LanguageCategory.EUROPEAN_MAJOR,
    "asian": LanguageCategory.ASIAN,
    "semitic": LanguageCategory.SEMITIC,
    "cyrillic": LanguageCategory.CYRILLIC,
    "classical": LanguageCategory.CLASSICAL,
    "minority": LanguageCategory.MINORITY,
}


def _split_languages_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "languages"


def _split_reference_texts_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "reference_texts"


def load_languages(
    base_dir: Path,
    legacy_file: Path,
) -> dict[str, Language]:
    """Load languages from the split layout, or the legacy YAML as fallback."""
    split_dir = _split_languages_dir(base_dir)
    if split_dir.is_dir():
        return _load_languages_split(split_dir)
    if legacy_file.exists():
        return _load_languages_legacy(legacy_file)
    raise FileNotFoundError(
        f"No language data found. Looked in {split_dir} and {legacy_file}."
    )


def load_reference_texts(
    base_dir: Path,
    legacy_file: Path,
) -> dict[str, ReferenceText]:
    """Load reference texts from the split layout, or the legacy YAML as fallback."""
    split_dir = _split_reference_texts_dir(base_dir)
    if split_dir.is_dir():
        return _load_reference_texts_split(split_dir)
    if legacy_file.exists():
        return _load_reference_texts_legacy(legacy_file)
    raise FileNotFoundError(
        f"No reference text data found. Looked in {split_dir} and {legacy_file}."
    )


def _iter_yaml_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*.yaml")):
        if path.is_file():
            yield path


def _load_languages_split(directory: Path) -> dict[str, Language]:
    languages: dict[str, Language] = {}
    for path in _iter_yaml_files(directory):
        with path.open("r", encoding="utf-8") as f:
            entry = yaml.safe_load(f) or {}
        code = str(entry["code"])
        category = _CATEGORY_MAP.get(
            entry.get("category", "european_major"),
            LanguageCategory.EUROPEAN_MAJOR,
        )
        languages[code] = Language(
            code=code,
            name=entry["name"],
            category=category,
            native_name=entry.get("native_name", entry["name"]),
            is_rtl=bool(entry.get("rtl", False)),
            script=entry.get("script", "Latin"),
        )
    return languages


def _load_languages_legacy(yaml_path: Path) -> dict[str, Language]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    languages: dict[str, Language] = {}
    for cat_key, cat_data in (data.get("categories") or {}).items():
        category = _CATEGORY_MAP.get(cat_key, LanguageCategory.EUROPEAN_MAJOR)
        for lang in cat_data.get("languages", []):
            code = str(lang["code"])
            languages[code] = Language(
                code=code,
                name=lang["name"],
                category=category,
                native_name=lang.get("native_name", lang["name"]),
                is_rtl=bool(lang.get("rtl", False)),
                script=lang.get("script", "Latin"),
            )
    return languages


def _load_reference_texts_split(directory: Path) -> dict[str, ReferenceText]:
    texts: dict[str, ReferenceText] = {}
    for path in _iter_yaml_files(directory):
        with path.open("r", encoding="utf-8") as f:
            entry = yaml.safe_load(f) or {}
        text_id = entry["id"]
        texts[text_id] = ReferenceText(
            id=text_id,
            title=entry["title"],
            author=entry.get("author", "Unknown"),
            year=int(entry.get("year") or 0),
            content=str(entry["text"]).strip(),
            style=entry.get("style", ""),
            source_language=entry.get("source_language", "en"),
            challenges=list(entry.get("challenges", [])),
            license=entry.get("license", "public-domain"),
        )
    return texts


def _load_reference_texts_legacy(yaml_path: Path) -> dict[str, ReferenceText]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    texts: dict[str, ReferenceText] = {}
    for entry in data.get("texts", []):
        text_id = entry["id"]
        source = entry.get("source_language", "English")
        # Normalize legacy "English" -> "en"
        if isinstance(source, str) and len(source) > 3:
            source = {"english": "en", "french": "fr"}.get(source.lower(), source.lower()[:2])

        texts[text_id] = ReferenceText(
            id=text_id,
            title=entry["title"],
            author=entry.get("author", "Unknown"),
            year=int(entry.get("year") or 0),
            content=str(entry["text"]).strip(),
            style=entry.get("style", ""),
            source_language=source,
            challenges=list(entry.get("challenges", [])),
            license=entry.get("license", "public-domain"),
        )
    return texts
