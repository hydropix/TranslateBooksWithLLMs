"""
Check that the translatable text of an EPUB chunk survived translation.

Placeholder validation only proves the markup came back. A reply made of
placeholders alone ("[id0][id1]" for a whole paragraph), or a runaway answer
(a commentary, the prompt echoed back), passes it while the text is lost or
polluted. This guard compares the amount of text on both sides.

Lengths are weighted so that the bounds hold across scripts: one CJK
character (Han, kana, Hangul) carries roughly as much meaning as three Latin
characters, so an EN->ZH translation is not mistaken for a truncation and a
ZH->EN one is not mistaken for runaway output. Mirrors the runaway guard of
metadata_translator.py (MAX_EXPANSION_RATIO / MIN_LENGTH_ALLOWANCE).
"""
import re
from typing import Optional

from src.common.placeholder_format import PlaceholderFormat

CJK_CHAR_WEIGHT = 3
"""Weight of one Han/kana/Hangul character, in Latin-character units."""

MIN_SURVIVAL_RATIO = 0.2
"""Reject a translation whose weighted text is below this share of the source."""

MIN_SOURCE_WEIGHT_FOR_RATIO = 40
"""The lower ratio only applies to sources at least this heavy: on a few words
the ratio is meaningless (an abbreviation, a one-word exclamation)."""

MAX_EXPANSION_RATIO = 4
"""Reject a translation whose weighted text is more than this many times longer."""

MIN_LENGTH_ALLOWANCE = 120
"""Absolute allowance on top of MAX_EXPANSION_RATIO for short sources."""

_CJK_RE = re.compile(r'[぀-ゟ゠-ヿ一-鿿가-힯㐀-䶿豈-﫿]')
_WHITESPACE_RE = re.compile(r'\s+')


def weighted_text_length(text: str, placeholder_format: Optional[PlaceholderFormat] = None) -> int:
    """Weighted count of the non-whitespace characters outside placeholders."""
    fmt = placeholder_format or PlaceholderFormat.from_config()
    bare = _WHITESPACE_RE.sub('', fmt.remove_all(text))
    cjk = len(_CJK_RE.findall(bare))
    return (len(bare) - cjk) + cjk * CJK_CHAR_WEIGHT


def text_survival_problem(
    source: str,
    translated: str,
    placeholder_format: Optional[PlaceholderFormat] = None,
) -> Optional[str]:
    """Why ``translated`` cannot stand for ``source``, or None when it can.

    Both texts may contain placeholders; they are ignored. The returned string
    is a short English label for the log line.
    """
    source_weight = weighted_text_length(source, placeholder_format)
    if source_weight == 0:
        return None
    translated_weight = weighted_text_length(translated, placeholder_format)
    if translated_weight == 0:
        return "the translation contains no text"
    if (source_weight >= MIN_SOURCE_WEIGHT_FOR_RATIO
            and translated_weight < source_weight * MIN_SURVIVAL_RATIO):
        return (f"the translation is too short ({translated_weight} vs "
                f"{source_weight} weighted characters)")
    if translated_weight > max(MAX_EXPANSION_RATIO * source_weight, MIN_LENGTH_ALLOWANCE):
        return (f"the translation is too long ({translated_weight} vs "
                f"{source_weight} weighted characters, runaway output)")
    return None
