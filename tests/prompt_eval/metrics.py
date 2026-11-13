"""Deterministic per-output metrics.

Each metric returns "pass", "fail" or "n/a". They decide the verdict; the
optional LLM judge never does.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import langdetect
from langdetect.lang_detect_exception import LangDetectException

from src import config
from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.common.plain_text_pipeline import _split_translated_back_to_paragraphs
from src.core.common.source_residue import (
    detect_source_residue,
    get_language_code,
    language_scripts,
)
from tests.prompt_eval.model import Case

langdetect.DetectorFactory.seed = 0
config.SOURCE_RESIDUE_CHECK_ENABLED = True

PASS, FAIL, NA = "pass", "fail", "n/a"

METRIC_NAMES = (
    "format",
    "chatter",
    "placeholders",
    "paragraphs",
    "residue",
    "target_language",
    "must_keep",
    "must_contain",
    "must_not_contain",
    "length",
)

CHATTER_MAX_CHARS = 20
MIN_DETECT_LETTERS = 40
CJK_CODES = frozenset({"zh", "ja", "ko"})

_PLACEHOLDER_RE = re.compile(r"\[id\d+\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LEADING_UNCLOSED_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


def _verdict(ok: bool) -> str:
    return PASS if ok else FAIL


def _primary(code: Optional[str]) -> Optional[str]:
    return code.split("-")[0].lower() if code else None


def _strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub("", text)


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def chatter_chars(raw: str, extracted: Optional[str]) -> int:
    """Non-whitespace characters outside the think blocks and the translation span."""
    text = _THINK_RE.sub("", raw)
    if "</think>" in text.lower():
        # Some models omit the opening tag and only close their reasoning.
        text = _LEADING_UNCLOSED_THINK_RE.sub("", text, count=1)
    start = text.find(TRANSLATE_TAG_IN)
    end = text.rfind(TRANSLATE_TAG_OUT)
    if start != -1 and end > start:
        text = text[:start] + text[end + len(TRANSLATE_TAG_OUT):]
    elif extracted and extracted in text:
        text = text.replace(extracted, "", 1)
    return sum(1 for ch in text if not ch.isspace())


def default_length_ratio(source_language: str, target_language: str) -> Optional[tuple[float, float]]:
    src_code = _primary(get_language_code(source_language))
    tgt_code = _primary(get_language_code(target_language))
    src_cjk, tgt_cjk = src_code in CJK_CODES, tgt_code in CJK_CODES
    if tgt_cjk and not src_cjk:
        return (0.15, 0.9)
    if src_cjk and not tgt_cjk:
        return (1.2, 6.0)
    src_scripts, tgt_scripts = language_scripts(source_language), language_scripts(target_language)
    for script in ("latin", "cyrillic"):
        if src_scripts == {script} and tgt_scripts == {script}:
            return (0.6, 1.8)
    return None


def _target_language(case: Case, extracted: str) -> str:
    expected = _primary(get_language_code(case.target_language))
    if not expected:
        return NA
    qualifying = [
        p for p in (_strip_placeholders(p) for p in _split_translated_back_to_paragraphs(extracted))
        if _letters(p) >= MIN_DETECT_LETTERS
    ]
    if not qualifying:
        return NA
    for paragraph in qualifying:
        try:
            detected = _primary(langdetect.detect(paragraph))
        except LangDetectException:
            return FAIL
        if detected != expected:
            return FAIL
    return PASS


def score(case: Case, raw: Optional[str], extracted: Optional[str]) -> dict[str, str]:
    """Score one output. Returns every name in METRIC_NAMES."""
    if raw is None:
        return {name: NA for name in METRIC_NAMES}

    is_translate = case.builder == "translate"
    ratio = case.length_ratio or default_length_ratio(case.source_language, case.target_language)
    applies = {
        "placeholders": case.has_placeholders,
        "paragraphs": case.expect_paragraphs,
        "residue": is_translate,
        "target_language": True,
        "must_keep": bool(case.must_keep),
        "must_contain": bool(case.must_contain_any),
        "must_not_contain": bool(case.must_not_contain),
        "length": ratio is not None,
    }

    result = {"format": _verdict(bool(extracted and extracted.strip())),
              "chatter": _verdict(chatter_chars(raw, extracted) <= CHATTER_MAX_CHARS)}

    if result["format"] == FAIL:
        for name, applicable in applies.items():
            result[name] = FAIL if applicable else NA
        return result

    out = extracted
    out_lower = out.lower()

    result["placeholders"] = (
        _verdict(Counter(_PLACEHOLDER_RE.findall(out)) == Counter(_PLACEHOLDER_RE.findall(case.text)))
        if applies["placeholders"] else NA
    )
    result["paragraphs"] = (
        _verdict(len(_split_translated_back_to_paragraphs(out))
                 == len(_split_translated_back_to_paragraphs(case.text)))
        if applies["paragraphs"] else NA
    )
    result["residue"] = (
        _verdict(detect_source_residue(case.text, out, case.source_language, case.target_language) is None)
        if applies["residue"] else NA
    )
    result["target_language"] = _target_language(case, out)
    result["must_keep"] = (
        _verdict(all(s in out for s in case.must_keep)) if applies["must_keep"] else NA
    )
    result["must_contain"] = (
        _verdict(all(any(alt.lower() in out_lower for alt in group) for group in case.must_contain_any))
        if applies["must_contain"] else NA
    )
    result["must_not_contain"] = (
        _verdict(not any(s.lower() in out_lower for s in case.must_not_contain))
        if applies["must_not_contain"] else NA
    )
    if applies["length"]:
        source_len = len(_strip_placeholders(case.text))
        lo, hi = ratio
        result["length"] = _verdict(source_len > 0 and lo <= len(_strip_placeholders(out)) / source_len <= hi)
    else:
        result["length"] = NA
    return result
