"""
One-shot migration: split monolithic benchmark YAML files into per-entry files.

Reads:
  - benchmark/languages.yaml         (categorized list of languages)
  - benchmark/reference_texts.yaml   (list of reference texts)

Writes:
  - benchmark/data/languages/<code>.yaml
  - benchmark/data/reference_texts/<source_lang>/<id>.yaml

Idempotent: re-running overwrites existing files with the latest source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_LANGUAGES = REPO_ROOT / "benchmark" / "languages.yaml"
LEGACY_REFERENCE_TEXTS = REPO_ROOT / "benchmark" / "reference_texts.yaml"
DATA_DIR = REPO_ROOT / "benchmark" / "data"
LANGUAGES_OUT = DATA_DIR / "languages"
REFERENCE_TEXTS_OUT = DATA_DIR / "reference_texts"


_LANG_TO_CODE = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "chinese": "zh-Hans",
}


def _normalize_source_language(value: str) -> str:
    """Accept either a language code or a language name; return a code."""
    if not value:
        return "en"
    v = value.strip()
    if len(v) <= 8 and "-" in v or (len(v) <= 3 and v.isalpha()):
        # Looks like a code already.
        return v
    return _LANG_TO_CODE.get(v.lower(), v.lower()[:2])


def split_languages(src: Path, dst_dir: Path) -> int:
    if not src.exists():
        print(f"[skip] languages source not found: {src}", file=sys.stderr)
        return 0

    with src.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for cat_key, cat_data in (data.get("categories") or {}).items():
        for lang in cat_data.get("languages") or []:
            entry = {
                "code": str(lang["code"]),
                "name": lang["name"],
                "native_name": lang.get("native_name", lang["name"]),
                "script": lang.get("script", "Latin"),
                "category": cat_key,
                "rtl": bool(lang.get("rtl", False)),
            }
            if "difficulty" in lang:
                entry["difficulty"] = lang["difficulty"]

            out_path = dst_dir / f"{entry['code']}.yaml"
            with out_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(entry, f, allow_unicode=True, sort_keys=False)
            count += 1

    return count


def split_reference_texts(src: Path, dst_dir: Path) -> int:
    if not src.exists():
        print(f"[skip] reference texts source not found: {src}", file=sys.stderr)
        return 0

    with src.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    count = 0

    for text in data.get("texts") or []:
        source_lang = _normalize_source_language(text.get("source_language", "en"))
        text_id = text["id"]

        entry = {
            "id": text_id,
            "title": text["title"],
            "author": text.get("author", "Unknown"),
            "year": text.get("year"),
            "source_language": source_lang,
            "style": text.get("style", ""),
            "challenges": list(text.get("challenges", [])),
            "text": text["text"].strip(),
            "license": text.get("license", "public-domain"),
        }
        if "era" in text:
            entry["era"] = text["era"]
        if "character_count" in text:
            entry["character_count"] = text["character_count"]

        lang_dir = dst_dir / source_lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        out_path = lang_dir / f"{text_id}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(entry, f, allow_unicode=True, sort_keys=False, width=4096)
        count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages-src",
        type=Path,
        default=LEGACY_LANGUAGES,
        help="Path to the legacy languages YAML.",
    )
    parser.add_argument(
        "--reference-texts-src",
        type=Path,
        default=LEGACY_REFERENCE_TEXTS,
        help="Path to the legacy reference_texts YAML.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Destination data directory.",
    )
    args = parser.parse_args()

    languages_dir = args.data_dir / "languages"
    reference_texts_dir = args.data_dir / "reference_texts"

    n_lang = split_languages(args.languages_src, languages_dir)
    n_text = split_reference_texts(args.reference_texts_src, reference_texts_dir)

    print(f"Wrote {n_lang} language files to {languages_dir}")
    print(f"Wrote {n_text} reference text files to {reference_texts_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
