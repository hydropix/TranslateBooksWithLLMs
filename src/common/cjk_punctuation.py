"""
Full-width (CJK) closing punctuation and the block boundaries it must not cross.

In Chinese and Japanese text, sentence-final and clause-final punctuation is
written without a space and always belongs to the text before it. When a
translation puts a block boundary (a closing ``</p>`` placeholder, a blank line
between plain-text paragraphs) right before such a character, the character is
rendered at the START of the next paragraph:

    expected   <p>こんにちは。</p><p>お元気ですか？</p>
    observed   <p>こんにちは</p><p>。お元気ですか？</p>

This module holds the one shared set of such characters and two helpers:

  * ``skip_closing_punctuation`` - used by the proportional placeholder
    re-insertion (token alignment) to place a placeholder after, never before,
    a run of closing punctuation;
  * ``reattach_closing_punctuation`` - used on finished model answers to move a
    run of closing punctuation that follows a block boundary back before it.

Deliberately NOT in the set:

  * opening brackets and quotes (「『（【〔〈《), which belong to the next block;
  * the ellipsis (… ‥) and the wave dash (〜 ～): a Japanese or Chinese
    paragraph routinely starts with them ("……はい。", "〜序章〜"), so moving
    them would damage correct text;
  * ASCII punctuation and curly quotes, which Latin scripts share and which are
    separated from the next block by a space anyway.
"""
import re
from typing import Optional

# Unambiguous closing punctuation of CJK scripts: it never opens a paragraph.
CJK_CLOSING_PUNCTUATION = frozenset(
    "。"  # 。 ideographic full stop
    "．"  # ． fullwidth full stop
    "｡"  # ｡ halfwidth ideographic full stop
    "？"  # ？ fullwidth question mark
    "！"  # ！ fullwidth exclamation mark
    "、"  # 、 ideographic comma
    "，"  # ， fullwidth comma
    "､"  # ､ halfwidth ideographic comma
    "："  # ： fullwidth colon
    "；"  # ； fullwidth semicolon
    "」"  # 」 right corner bracket
    "』"  # 』 right white corner bracket
    "｣"  # ｣ halfwidth right corner bracket
    "）"  # ） fullwidth right parenthesis
    "】"  # 】 right black lenticular bracket
    "〕"  # 〕 right tortoise shell bracket
    "〗"  # 〗 right white lenticular bracket
    "〙"  # 〙 right white tortoise shell bracket
    "〛"  # 〛 right white square bracket
    "〉"  # 〉 right angle bracket
    "》"  # 》 right double angle bracket
    "］"  # ］ fullwidth right square bracket
    "｝"  # ｝ fullwidth right curly bracket
    "〞"  # 〞 double prime quotation mark
    "〟"  # 〟 low double prime quotation mark
)

_CLASS = "[" + re.escape("".join(sorted(CJK_CLOSING_PUNCTUATION))) + "]"


def is_cjk_closing_punctuation(char: str) -> bool:
    """True when ``char`` is one character of ``CJK_CLOSING_PUNCTUATION``."""
    return char in CJK_CLOSING_PUNCTUATION


def skip_closing_punctuation(text: str, pos: int) -> int:
    """Return the first index at or after ``pos`` that is not closing punctuation.

    An insertion point that lands on "。」" is moved past it, so whatever is
    inserted there opens the next sentence instead of splitting the punctuation
    off the previous one.
    """
    while 0 <= pos < len(text) and text[pos] in CJK_CLOSING_PUNCTUATION:
        pos += 1
    return pos


def reattach_closing_punctuation(text: str, boundary_pattern: str,
                                 source: Optional[str] = None) -> str:
    """Move closing punctuation that follows a block boundary back before it.

    ``boundary_pattern`` is a regex (no capturing groups) matching one block
    boundary in ``text``. Every "<text><boundary><closing punctuation>" becomes
    "<text><closing punctuation><boundary>". A boundary at the very start of the
    text or preceded by whitespace has no sentence to give the punctuation back
    to and is left alone.

    When ``source`` is given and one of its own blocks starts with closing
    punctuation, the text is returned unchanged: the source shows such a block
    start is real content here, and a guess could damage it.
    """
    if not text or not boundary_pattern:
        return text
    if source and re.search(rf"(?:\A|{boundary_pattern})\s*{_CLASS}", source):
        return text
    pattern = re.compile(rf"(?<=\S)({boundary_pattern})({_CLASS}+)")
    return pattern.sub(lambda m: m.group(2) + m.group(1), text)


# A blank line between two plain-text paragraphs (spaces/tabs tolerated).
PARAGRAPH_BREAK_PATTERN = r"[ \t]*\n[ \t]*\n\s*"
