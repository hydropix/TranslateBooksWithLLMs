"""
Glossary module for consistent translation of recurring terms.

Provides:
- models: Glossary, GlossaryTerm, GlossaryConfig dataclasses
- store: SQLite CRUD operations
- filter: chunk-aware glossary filtering (Latin word-boundary, CJK substring)
- injector: build the glossary and cast blocks injected into the prompt
- inflection: which target languages need the target-side inflection instruction
"""
from src.core.glossary.models import (
    ALLOWED_GENDERS,
    BulkReplaceResult,
    DEFAULT_MAX_CAST_ENTRIES,
    Glossary,
    GlossaryConfig,
    GlossaryTerm,
    KNOWN_GENDERS,
    normalize_gender,
)
from src.core.glossary.filter import filter_glossary
from src.core.glossary.inflection import (
    INFLECTED_TARGET_LANGUAGES,
    target_language_is_inflected,
)
from src.core.glossary.injector import (
    build_cast_block,
    build_glossary_block,
    build_refinement_glossary_block,
)
from src.core.glossary.store import GlossaryStore
from src.core.glossary.ner import parse_ner_response, suggest_terms

__all__ = [
    "ALLOWED_GENDERS",
    "BulkReplaceResult",
    "DEFAULT_MAX_CAST_ENTRIES",
    "Glossary",
    "GlossaryTerm",
    "GlossaryConfig",
    "GlossaryStore",
    "INFLECTED_TARGET_LANGUAGES",
    "KNOWN_GENDERS",
    "filter_glossary",
    "build_cast_block",
    "build_glossary_block",
    "build_refinement_glossary_block",
    "normalize_gender",
    "parse_ner_response",
    "suggest_terms",
    "target_language_is_inflected",
]
