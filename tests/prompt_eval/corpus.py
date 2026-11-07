"""Corpus loader and validator."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Iterable, Optional

import yaml

from tests.prompt_eval.model import Case

CORPUS_DIR = Path(__file__).parent / "corpus"
BUILDERS = frozenset({"translate", "refine"})
PLACEHOLDER_RE = re.compile(r"\[id(\d+)\]")

_CASE_FIELDS = {f.name for f in fields(Case)}


def _split_paragraphs(text: str) -> list[str]:
    from src.core.common.plain_text_pipeline import _split_translated_back_to_paragraphs
    return _split_translated_back_to_paragraphs(text)


def case_from_dict(data: dict, origin: str = "") -> Case:
    unknown = set(data) - _CASE_FIELDS
    if unknown:
        raise ValueError(f"{origin}: unknown case keys {sorted(unknown)}")
    data = dict(data)
    for key in ("must_keep", "must_not_contain"):
        if key in data and data[key] is not None:
            data[key] = tuple(data[key])
    if data.get("must_contain_any") is not None:
        data["must_contain_any"] = tuple(tuple(group) for group in data["must_contain_any"])
    if data.get("length_ratio") is not None:
        data["length_ratio"] = tuple(float(x) for x in data["length_ratio"])
    if data.get("prompt_options") is None:
        data["prompt_options"] = {}
    try:
        return Case(**data)
    except TypeError as exc:
        raise ValueError(f"{origin}: {exc}") from exc


def validate_case(case: Case) -> list[str]:
    """Return human-readable errors for one case, empty when valid."""
    errors: list[str] = []
    prefix = f"{case.id}:"

    if case.builder not in BUILDERS:
        errors.append(f"{prefix} builder must be one of {sorted(BUILDERS)}, got {case.builder!r}")

    indices = [int(n) for n in PLACEHOLDER_RE.findall(case.text)]
    if case.has_placeholders:
        counts = Counter(indices)
        if len(indices) < 2:
            errors.append(f"{prefix} has_placeholders requires at least two [idN] tokens")
        elif sorted(counts) != list(range(len(counts))):
            errors.append(f"{prefix} [idN] tokens must be numbered from 0 without gaps, got {sorted(counts)}")
        duplicated = sorted(n for n, c in counts.items() if c > 1)
        if duplicated:
            errors.append(f"{prefix} [idN] tokens must appear exactly once, duplicated: {duplicated}")
    elif indices:
        errors.append(f"{prefix} text contains [idN] tokens but has_placeholders is false")

    if case.expect_paragraphs and len(_split_paragraphs(case.text)) < 2:
        errors.append(f"{prefix} expect_paragraphs requires at least 2 paragraphs")

    for keep in case.must_keep:
        if keep not in case.text:
            errors.append(f"{prefix} must_keep string {keep!r} does not occur in text")

    if case.length_ratio is not None:
        if len(case.length_ratio) != 2:
            errors.append(f"{prefix} length_ratio must be a (lo, hi) pair")
        else:
            lo, hi = case.length_ratio
            if not (0 < lo < hi):
                errors.append(f"{prefix} length_ratio must satisfy 0 < lo < hi, got {case.length_ratio}")

    if case.builder == "translate" and case.source_language == case.target_language:
        errors.append(f"{prefix} translate cases need different source and target languages")

    return errors


def load_corpus(
    builders: Optional[Iterable[str]] = None,
    case_filter: Optional[str] = None,
    corpus_dir: Path = CORPUS_DIR,
) -> list[Case]:
    """Load, validate and filter every ``*.yaml`` corpus file."""
    cases: list[Case] = []
    for path in sorted(corpus_dir.glob("*.yaml")):
        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        for i, entry in enumerate(doc.get("cases") or []):
            cases.append(case_from_dict(entry, origin=f"{path.name}[{i}]"))

    duplicates = sorted(i for i, c in Counter(c.id for c in cases).items() if c > 1)
    if duplicates:
        raise ValueError(f"Duplicate case ids: {duplicates}")

    errors = [err for case in cases for err in validate_case(case)]
    if errors:
        raise ValueError("Invalid corpus:\n" + "\n".join(errors))

    if builders is not None:
        wanted = set(builders)
        cases = [c for c in cases if c.builder in wanted]
    if case_filter:
        pattern = re.compile(case_filter)
        cases = [c for c in cases if pattern.search(c.id)]
    return cases
