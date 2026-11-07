"""
Load a glossary from a JSON or CSV file.

JSON shapes accepted:
  - {"terms": [{"source": "...", "target": "...", "category": "...", "gender": "..."}, ...]}
  - {"name": "...", "terms": [...]}              (other top-level fields ignored)
  - [{"source": "...", "target": "..."}, ...]    (bare list of terms)

CSV shape accepted:
  - Header row required, must include at least `source` and `target` columns.
  - Optional `category` and `gender` columns are read when present.
"""
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple

from src.core.glossary.models import normalize_gender


def _iter_term_entries(raw_terms: Iterable) -> Iterable[Dict[str, str]]:
    for entry in raw_terms or []:
        if not isinstance(entry, dict):
            continue
        source = (entry.get("source") or entry.get("source_term") or "").strip()
        target = (entry.get("target") or entry.get("translated_term") or "").strip()
        if not source or not target:
            continue
        category = (entry.get("category") or "").strip()
        gender = normalize_gender(entry.get("gender") or entry.get("sex"))
        forbidden = entry.get("forbidden") or entry.get("forbidden_translations") or []
        if isinstance(forbidden, str):
            forbidden = [part.strip() for part in forbidden.split("|") if part.strip()]
        else:
            forbidden = [str(part).strip() for part in forbidden if str(part).strip()]
        yield {
            "source": source,
            "target": target,
            "category": category,
            "gender": gender or "",
            "forbidden": forbidden,
        }


def _read_file(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return list(_iter_term_entries(data))
        if isinstance(data, dict):
            return list(_iter_term_entries(data.get("terms") or []))
        raise ValueError("Unsupported JSON structure: expected list or dict with 'terms'.")

    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "source" not in reader.fieldnames or "target" not in reader.fieldnames:
                raise ValueError("CSV must contain 'source' and 'target' columns.")
            return list(_iter_term_entries(reader))

    raise ValueError(f"Unsupported glossary file extension: {ext} (expected .json or .csv)")


def load_glossary_terms_from_file(path: str) -> Dict[str, str]:
    """Back-compat: return only the {source: target} mapping."""
    entries = _read_file(path)
    return {e["source"]: e["target"] for e in entries}


def load_glossary_from_file(path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Load a glossary file and return:
      - terms: {source: target}                  (consumed by the filter)
      - metadata: {source: {category, gender}}   (consumed by the injector)

    A term contributes a metadata entry only for the fields it actually has,
    so a glossary with no genders produces exactly the metadata it did before
    the field existed.
    """
    entries = _read_file(path)
    terms: Dict[str, str] = {}
    metadata: Dict[str, Dict[str, str]] = {}
    for e in entries:
        source = e["source"]
        terms[source] = e["target"]
        entry_meta = {}
        if e["category"]:
            entry_meta["category"] = e["category"]
        if e["gender"]:
            entry_meta["gender"] = e["gender"]
        if e.get("forbidden"):
            entry_meta["forbidden"] = "|".join(e["forbidden"])
        if entry_meta:
            metadata[source] = entry_meta
    return terms, metadata
