"""
Build the glossary block to inject into the system prompt.

The block style mirrors the existing prompt voice in src/prompts/prompts.py
(numbered priorities, MANDATORY phrasing) so the LLM treats glossary entries
with the same weight as the rest of the instructions.

Two blocks live here:

- ``build_glossary_block`` renders the terms that match the current chunk.
- ``build_cast_block`` renders every gendered entity in the glossary,
  regardless of the current chunk. Gender is the one piece of glossary data
  that is needed precisely when the term is *absent*: in languages that omit
  the subject or do not mark gender (Chinese, Japanese, Korean, Turkish,
  Finnish), a passage can be full of pronouns without naming anyone, and a
  chunk-filtered glossary says nothing there. That is exactly where a model
  falls back to masculine for everyone.
"""
from typing import Dict, List, Optional, Tuple

from src.core.glossary.inflection import target_language_is_inflected
from src.core.glossary.models import DEFAULT_MAX_CAST_ENTRIES, normalize_gender

# Target-side counterpart of the '|' source-variants line: in an inflected
# target the citation form listed in the glossary is routinely NOT the form a
# grammatical sentence needs, so "EXACT" has to be scoped to the choice of
# rendering rather than to the letters. Emitted as a single line, only for the
# languages in INFLECTED_TARGET_LANGUAGES.
#
# It grants the whole morphology on purpose, rather than "keep the stem,
# inflect the ending": that narrower licence is false for several of the gated
# languages (Turkish kitap->kitabı, Finnish katu->kadun, Russian отец->отца,
# German Vater->Väter) and would not cover derivation, which is where the drift
# reported in issue #255 actually shows up.
TARGET_INFLECTION_INSTRUCTION = (
    "Each target above is given in its dictionary form. Apply whatever "
    "morphological changes {target_language} grammar requires — case, number, "
    "agreement, derived forms, and the regular stem alternations these entail. "
    "What must never change is the choice of rendering: do not substitute a "
    "different translation, a different transliteration, or a more familiar "
    "variant of the term."
)


def build_glossary_block(
    filtered_terms: Dict[str, str],
    target_language: str = "",
    term_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """
    Render the glossary block. Empty string if no terms.

    Args:
        filtered_terms: {source: target} of terms that match the current chunk.
        target_language: selects the target-side inflection instruction, emitted
            only for the languages in ``INFLECTED_TARGET_LANGUAGES``. The
            exclusion criterion (does inflection change the written form of the
            term?) is documented in ``src/core/glossary/inflection.py``.
        term_metadata: optional {source: {category, gender}} mapping. When a
            term has a category, it is rendered as a bracketed hint after the
            arrow so the LLM can disambiguate homonyms (e.g. a name vs a
            place sharing the same spelling). A gender is appended to the same
            hint; the authoritative gender instruction lives in the cast block
            built by ``build_cast_block``.

    The block lives between the optional sections and the placeholder section
    in the system prompt — close enough to the input text that the model will
    not forget it, but not after the FINAL REMINDER so the output-language
    reminder stays last.
    """
    if not filtered_terms:
        return ""

    metadata = term_metadata or {}

    def _forbidden_values(source: str) -> List[str]:
        raw = (metadata.get(source) or {}).get("forbidden") or ""
        if isinstance(raw, str):
            return [part.strip() for part in raw.split("|") if part.strip()]
        return [str(part).strip() for part in raw if str(part).strip()]

    has_forbidden = any(_forbidden_values(source) for source in filtered_terms)
    lines = [
        "# GLOSSARY - REQUIRED TRANSLATIONS",
        "",
        "MANDATORY: use these EXACT translations whenever the source term appears.",
        "Do NOT paraphrase, transliterate differently, or invent alternatives.",
        *([TARGET_INFLECTION_INSTRUCTION.format(target_language=target_language)]
          if target_language_is_inflected(target_language) else []),
        "Apply each rule consistently every time the term occurs.",
        "When several source forms are listed before the arrow (comma-separated), they are inflected variants of the same entity — translate any of them with the single target on the right.",
        "Bracketed hints after the arrow (e.g. [character, female]) describe the entity type and, where known, its gender — use them to disambiguate and to pick pronouns, never as part of the translation.",
        *(["If a forbidden rendering is listed, never use it, even if it sounds natural in the target language."]
          if has_forbidden else []),
        "",
    ]

    for source, target in filtered_terms.items():
        meta = metadata.get(source) or {}
        category = (meta.get("category") or "").strip()
        gender = normalize_gender(meta.get("gender"))
        forbidden = _forbidden_values(source)
        # Render alternatives (declined forms separated by '|' in storage) as
        # a comma-separated list so the LLM reads them as a natural set.
        display_source = ", ".join(
            a.strip() for a in source.split("|") if a.strip()
        ) or source
        hints = [h for h in (category, gender) if h]
        line = f"  - {display_source} -> {target}"
        if hints:
            line += f"  [{', '.join(hints)}]"
        if forbidden:
            line += f"  [forbidden: {', '.join(forbidden)}]"
        lines.append(line)

    return "\n".join(lines) + "\n"


def build_cast_block(
    glossary_terms: Dict[str, str],
    term_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    max_entries: int = DEFAULT_MAX_CAST_ENTRIES,
) -> Tuple[str, bool]:
    """
    Render the cast block listing every entity with a known gender.

    Unlike the glossary block, this is NOT filtered against the current chunk:
    a chunk that refers to a character only by pronoun contains no glossary
    match, yet that is the chunk that most needs the gender.

    Args:
        glossary_terms: {source: target} for the whole glossary (unfiltered).
        term_metadata: {source: {category, gender}}. Entries whose gender is
            absent or unrecognized are skipped.
        max_entries: cap on the number of listed entities. The block is
            injected into every chunk, so an unbounded cast on a 300-character
            saga would be a permanent token tax.

    Returns:
        (block, capped). ``block`` is "" when no entity has a known gender —
        which is the case for every glossary predating this field, so the
        injected prompt is unchanged until someone fills a gender in.
    """
    metadata = term_metadata or {}
    if not glossary_terms or not metadata:
        return "", False

    entries: List[Tuple[str, str, str]] = []
    for source, target in glossary_terms.items():
        gender = normalize_gender((metadata.get(source) or {}).get("gender"))
        if not gender:
            continue
        # The first alternative is the canonical form; listing every declined
        # variant here would triple the block for no gain, since the variants
        # are already spelled out in the glossary block.
        canonical = next(
            (a.strip() for a in source.split("|") if a.strip()),
            source,
        )
        entries.append((canonical, target, gender))

    if not entries:
        return "", False

    capped = False
    if max_entries and len(entries) > max_entries:
        capped = True
        # Truncate in glossary order rather than by relevance: the block must
        # be byte-identical across every chunk of a job, both for prompt
        # caching and so a character does not change gender halfway through.
        entries = entries[:max_entries]

    lines = [
        "# CAST - CHARACTER GENDERS",
        "",
        "MANDATORY: each entity below has a fixed gender for the entire text.",
        "Use pronouns and gendered agreement matching the gender listed here, "
        "even when the source passage omits the subject or does not mark gender.",
        "Never default to masculine for an entity listed as female, and never "
        "switch an entity's gender between passages.",
        "This list is the reference for the whole work — an entity may be "
        "referred to here by pronoun only in the passage you are translating.",
        "",
    ]
    for canonical, target, gender in entries:
        label = f"{canonical} ({target})" if target and target != canonical else canonical
        lines.append(f"  - {label} — {gender}")

    return "\n".join(lines) + "\n", capped
