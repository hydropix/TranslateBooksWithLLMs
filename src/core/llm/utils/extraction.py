"""
Translation extraction from LLM responses.

This module provides utilities for extracting translations from LLM responses,
handling various response formats including thinking blocks.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Reasoning tag names some models emit around their chain of thought.
_THINKING_TAG_NAMES = ("think", "thinking", "reasoning")
_THINKING_NAMES_RE = "|".join(_THINKING_TAG_NAMES)

_COMPLETE_THINKING_BLOCK = re.compile(
    rf"<({_THINKING_NAMES_RE})>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_THINKING_OPEN = re.compile(rf"<(?:{_THINKING_NAMES_RE})>", re.IGNORECASE)
_THINKING_CLOSE = re.compile(rf"</(?:{_THINKING_NAMES_RE})>", re.IGNORECASE)

# Placeholder contents of the example blocks embedded in the prompts
# (src/prompts/prompts.py). When a model echoes the prompt, these blocks
# appear in the response next to the real translation and must never win.
_PROMPT_EXAMPLE_CONTENTS = frozenset({
    "text...",
    "your translation here",
    "your translated text here",
    "your refined text here",
})


class TranslationExtractor:
    """
    Extracts translation text from LLM responses.

    Handles:
        - Extraction between custom tags (e.g., <Translated>...</Translated>)
        - Removal of <think>, <thinking> and <reasoning> blocks, including
          unclosed ones and orphan closing tags
        - Responses with several tagged blocks (e.g. the model echoed the
          prompt's example blocks): the last real block wins and the
          returned text never contains a literal tag
        - Malformed closing tags (fuzzy match on the tag prefix)

    Example:
        >>> extractor = TranslationExtractor("<Translated>", "</Translated>")
        >>> response = "<think>reasoning</think><Translated>Hello</Translated>"
        >>> extractor.extract(response)
        'Hello'
    """

    def __init__(self, tag_in: str, tag_out: str):
        """
        Initialize the extractor with custom tags.

        Args:
            tag_in: Opening tag (e.g., "<Translated>")
            tag_out: Closing tag (e.g., "</Translated>")
        """
        self._tag_in = tag_in
        self._tag_out = tag_out

    def extract(self, response: str) -> Optional[str]:
        """
        Extract the translation from a raw LLM response.

        Reasoning blocks are removed first, so translation tags quoted inside
        a model's chain of thought are ignored. When the response holds more
        than one complete block, the last block whose content is not one of
        the prompt's example placeholders is returned: models that echo the
        prompt reproduce its example blocks before the real answer.

        Args:
            response: Raw LLM response text

        Returns:
            Extracted translation, or None if extraction fails
        """
        if not response:
            return None

        response = response.strip()
        response = self._remove_think_blocks(response)
        # Some providers (e.g. Gemini) wrap the whole answer in ``` fences
        response = self._remove_markdown_code_blocks(response).strip()

        blocks = self._find_blocks(response)
        if blocks:
            if len(blocks) > 1:
                logger.debug("Response contains %d translation blocks; using the last real one",
                             len(blocks))
            return self._pick_block(blocks)

        # FUZZY FALLBACK: opening tag present but closing tag is malformed.
        # Some models (e.g. Gemini without thinking) write </TRANATION>
        # instead of </TRANSLATION>.
        if response.startswith(self._tag_in):
            closing_tag_match = re.match(r'</(\w+)>', self._tag_out)
            if closing_tag_match:
                prefix = closing_tag_match.group(1)[:3]
                fuzzy_close = re.compile(rf'</\w*{re.escape(prefix)}\w*>')
                fuzzy_match = fuzzy_close.search(response, len(self._tag_in))
                if fuzzy_match:
                    content = response[:fuzzy_match.start()]
                    content = content[content.rfind(self._tag_in) + len(self._tag_in):]
                    logger.warning("Fuzzy tag match: closing tag was malformed, extracted "
                                   "content using prefix '</%s...'", prefix)
                    return content.strip()

        return None

    def _find_blocks(self, text: str) -> List[str]:
        """Return the content of every complete tag_in ... tag_out block.

        Each block runs from the opening tag nearest to its closing tag, so
        an unbalanced opening tag earlier in the text can never leak into a
        block's content.
        """
        blocks = []
        cursor = 0
        while True:
            close = text.find(self._tag_out, cursor)
            if close < 0:
                break
            segment = text[cursor:close]
            open_idx = segment.rfind(self._tag_in)
            if open_idx >= 0:
                blocks.append(segment[open_idx + len(self._tag_in):].strip())
            cursor = close + len(self._tag_out)
        return blocks

    def _has_block(self, text: str) -> bool:
        return bool(self._find_blocks(text))

    @staticmethod
    def _pick_block(blocks: List[str]) -> str:
        """Pick the last block that is not a prompt example placeholder."""
        for content in reversed(blocks):
            if content and content.lower() not in _PROMPT_EXAMPLE_CONTENTS:
                return content
        return blocks[-1]

    def _remove_markdown_code_blocks(self, response: str) -> str:
        """
        Remove markdown code block wrappers from response.

        Some providers (notably Gemini) may wrap responses in markdown code blocks
        like ```xml\\n...\\n``` which prevents tag extraction.
        """
        match = re.match(r'^```\w*\s*\n(.*?)\n```\s*$', response, re.DOTALL)
        if match:
            return match.group(1)
        return response

    def _remove_think_blocks(self, response: str) -> str:
        """
        Remove reasoning blocks (<think>, <thinking>, <reasoning>) from response.

        These blocks contain the model's internal reasoning; translation tags
        inside them must never be extracted.

        Handles:
            - Complete blocks
            - Orphan closing tags (the opening tag was truncated, e.g. by
              Ollama), including reasoning that itself quotes the closing tag
            - Unclosed opening tags (reasoning that never ended)
        """
        response = _COMPLETE_THINKING_BLOCK.sub('', response)

        # Orphan closing tags: everything before them is reasoning. Prefer the
        # last closing tag that is followed by a complete translation block
        # (reasoning may quote the closing tag). Without such a block, keep
        # the legacy rule: strip only if the prefix holds no opening
        # translation tag, so a truncated answer stays inspectable (#170).
        closes = list(_THINKING_CLOSE.finditer(response))
        if closes:
            for match in reversed(closes):
                if self._has_block(response[match.end():]):
                    return response[match.end():].lstrip()
            first = closes[0]
            if self._tag_in.lower() not in response[:first.end()].lower():
                response = response[first.end():].lstrip()

        # Unclosed opening tag: the reasoning runs up to the last opening
        # translation tag after it (if any), else to the end of the text.
        opening = _THINKING_OPEN.search(response)
        if opening:
            last_tag_in = response.rfind(self._tag_in)
            if last_tag_in > opening.start():
                response = response[:opening.start()] + response[last_tag_in:]
            else:
                response = response[:opening.start()]

        return response
