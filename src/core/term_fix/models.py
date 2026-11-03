"""
Result types shared by the term-fix correctors, the fixer and the report.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: Replaced deterministically, no LLM.
STATUS_LITERAL = "literal"
#: Corrected by the LLM and accepted by the guard.
STATUS_FIXED = "fixed"
#: The LLM correction failed the guard twice; the original was kept.
STATUS_NEEDS_REVIEW = "needs_review"
#: Not processed (job interrupted); the original was kept.
STATUS_SKIPPED = "skipped"

Edit = Tuple[int, int, str]


@dataclass(frozen=True)
class SnippetItem:
    """A snippet handed to a corrector. `spans` are (start, end, rule_index)
    relative to `text`."""
    text: str
    spans: Tuple[Tuple[int, int, int], ...]


@dataclass(frozen=True)
class Correction:
    """A corrector's answer for one SnippetItem. `text` is None when the
    original must be kept (see `status`)."""
    text: Optional[str]
    status: str
    reason: str = ""


@dataclass(frozen=True)
class Occurrence:
    """One occurrence, with its sentence before and after the change."""
    rule_index: int
    alternative_index: int
    location: str
    matched: str
    replacement: str
    before: str
    after: str
    status: str
    reason: str = ""


@dataclass
class TermFixResult:
    """Outcome over a list of text units (one for TXT, many for EPUB/DOCX)."""
    texts: List[str]
    edits: List[List[Edit]]
    occurrences: List[Occurrence] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Convenience for single-unit inputs."""
        return self.texts[0] if self.texts else ""

    @property
    def changed(self) -> bool:
        return any(self.edits)

    def count(self, status: str) -> int:
        return sum(1 for o in self.occurrences if o.status == status)
