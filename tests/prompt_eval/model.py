"""Data model shared by every part of the prompt regression harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Case:
    """One corpus entry: a fixed input rendered through one prompt builder."""

    id: str
    builder: str                       # "translate" | "refine"
    source_language: str               # full name, e.g. "English"
    target_language: str               # full name, e.g. "French"
    text: str                          # main_content (translate) or draft_translation (refine)
    has_placeholders: bool
    prompt_options: dict = field(default_factory=dict)  # may include glossary_terms
    context_before: str = ""
    context_after: str = ""
    previous_context: str = ""         # previous_translation_context / previous_refined_context
    expect_paragraphs: bool = False
    must_keep: tuple[str, ...] = ()
    must_contain_any: tuple[tuple[str, ...], ...] = ()
    must_not_contain: tuple[str, ...] = ()
    length_ratio: Optional[tuple[float, float]] = None
    reference_language_of_text: Optional[str] = None
    notes: str = ""

    # prompt_options is a dict, so the default dataclass hash would fail.
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class Target:
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    dropped_kwargs: tuple[str, ...] = ()


@dataclass
class RunRecord:
    """One LLM call; serialised as one JSONL line.

    Not frozen: the optional judge fills ``judge`` after the run.
    """

    case_id: str
    arm: str                           # "baseline" | "candidate"
    target: str                        # "provider:model"
    repeat: int                        # 0-based
    raw: Optional[str]
    extracted: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    elapsed_s: float
    metrics: dict[str, str]
    judge: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        return cls(**data)

    @property
    def request_failed(self) -> bool:
        return self.raw is None
