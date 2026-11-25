"""HTML tag classification and priority detection.

This module provides utilities for classifying HTML tags by type and
determining split priorities for HTML-aware chunking.
"""
import re


class TagClassifier:
    """Classifies HTML tags by type and determines split priorities.

    This class provides methods to identify block-level tags, determine
    split priorities for chunking, and detect chapter headings.
    """

    BLOCK_TAGS = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                  'blockquote', 'section', 'article', 'li', 'tr', 'td', 'th'}

    CHAPTER_HEADINGS = {'h1', 'h2', 'h3'}
    MAJOR_SECTIONS = {'h4', 'h5', 'h6', 'section', 'article'}
    PARAGRAPHS = {'p', 'div', 'blockquote'}

    def get_split_priority(self, tag: str) -> int:
        """Get priority for splitting at this tag.

        Lower number = higher priority (preferred split point).

        Priority levels:
        1: Chapter headings (h1, h2, h3)
        2: Major sections (h4, h5, h6, section, article)
        3: Paragraphs and divs (p, div, blockquote)
        4: Other blocks (li, tr, td, th)

        Args:
            tag: HTML tag string (e.g., "</p>", "<div>")

        Returns:
            Priority level (1-4)
        """
        tag_lower = tag.lower()

        # Priority 1: Chapter headings
        if any(f'</{ht}>' in tag_lower for ht in self.CHAPTER_HEADINGS):
            return 1

        # Priority 2: Major sections
        if any(f'</{ht}>' in tag_lower for ht in self.MAJOR_SECTIONS):
            return 2

        # Priority 3: Paragraphs and divs
        if any(f'</{ht}>' in tag_lower for ht in self.PARAGRAPHS):
            return 3

        # Priority 4: Other blocks
        return 4

    def is_block_closing_tag(self, tag: str) -> bool:
        """Check if tag is a block closing tag.

        Examples: "</p>", "</div>", "</h1>"

        Args:
            tag: HTML tag string

        Returns:
            True if tag is a block closing tag
        """
        tag_lower = tag.lower()
        for bt in self.BLOCK_TAGS:
            if f'</{bt}>' in tag_lower or f'</{bt} ' in tag_lower:
                return True
        return False

    def is_block_opening_tag(self, tag: str) -> bool:
        """Check if tag is a block opening tag.

        Examples: "<p>", "<div>", "<h1 class='title'>"

        Args:
            tag: HTML tag string

        Returns:
            True if tag is a block opening tag
        """
        tag_lower = tag.lower()
        for bt in self.BLOCK_TAGS:
            if f'<{bt}>' in tag_lower or f'<{bt} ' in tag_lower:
                return True
        return False

    def is_chapter_heading(self, tag: str) -> bool:
        """Check if tag is a chapter heading (h1-h3).

        Chapter headings are high-priority split points as they typically
        mark major structural boundaries in EPUB content.

        Args:
            tag: HTML tag string

        Returns:
            True if tag is a chapter heading
        """
        tag_lower = tag.lower()
        return any(f'</{ht}>' in tag_lower for ht in self.CHAPTER_HEADINGS)


#: Elements whose start or end separates two runs of text that must never
#: exchange words: the chunking split points (``TagClassifier.BLOCK_TAGS``)
#: plus the other HTML block elements met in EPUB XHTML and mammoth's DOCX
#: HTML (definition lists, captions, preformatted text, list and table
#: containers, sectioning elements). Inline markup (em, span, a, br, FB2
#: emphasis...) is deliberately absent: text may move across it within a block.
BLOCK_BOUNDARY_TAGS = frozenset(TagClassifier.BLOCK_TAGS | {
    'address', 'aside', 'body', 'caption', 'dd', 'details', 'dl', 'dt',
    'figcaption', 'figure', 'footer', 'header', 'hr', 'main', 'nav', 'ol',
    'pre', 'summary', 'table', 'tbody', 'tfoot', 'thead', 'ul',
})

_TAG_NAME_RE = re.compile(r'<\s*/?\s*(?:[A-Za-z][\w.-]*:)?([A-Za-z][\w.-]*)')


def is_block_boundary_markup(markup: str) -> bool:
    """True when ``markup`` (a placeholder's tags) opens or closes a block.

    The fallback ladder never lets text cross such a placeholder (see
    ``chunk_fallback``). Namespace prefixes (``<html:p>``) are ignored.
    """
    return any(name.lower() in BLOCK_BOUNDARY_TAGS
               for name in _TAG_NAME_RE.findall(markup))
