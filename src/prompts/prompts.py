from typing import Any, Dict, List, NamedTuple, Tuple, Optional

from src.prompts.examples import (build_placeholder_section,
                              get_output_format_example, get_subtitle_example,
                              TAG0)
from src.config import (INPUT_TAG_IN, INPUT_TAG_OUT, TRANSLATE_TAG_IN,
                        TRANSLATE_TAG_OUT, PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX,
                        create_placeholder)

# Tags for placeholder correction responses
CORRECTED_TAG_IN = "<CORRECTED_TAG_IN>"
CORRECTED_TAG_OUT = "<CORRECTED_TAG_OUT>"


class PromptPair(NamedTuple):
    """A pair of system and user prompts for LLM translation."""
    system: str
    user: str


# ============================================================================
# SHARED PROMPT SECTIONS
# ============================================================================

def _get_output_format_section(
    translate_tag_in: str,
    translate_tag_out: str,
    input_tag_in: str,
    input_tag_out: str,
    additional_rules: str = "",
    example_format: str = "Your translated text here"
) -> str:
    """
    Generate standardized output format instructions.

    Args:
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        input_tag_in: Opening tag for input text
        input_tag_out: Closing tag for input text
        additional_rules: Optional additional formatting rules
        example_format: Example text to show in correct format

    Returns:
        str: Formatted output format instructions
    """
    additional_rules_text = f"\n{additional_rules}" if additional_rules else ""

    return f"""# OUTPUT FORMAT

**CRITICAL OUTPUT RULES:**
1. Translate ONLY the text between "{input_tag_in}" and "{input_tag_out}" tags
2. Your response MUST start with {translate_tag_in} (first characters, no text before)
3. Your response MUST end with {translate_tag_out} (last characters, no text after)
4. Include NOTHING before {translate_tag_in} and NOTHING after {translate_tag_out}
5. Do NOT add explanations, comments, notes, or greetings{additional_rules_text}

**INCORRECT examples (DO NOT do this):**
❌ "Here is the translation: {translate_tag_in}Text...{translate_tag_out}"
❌ "{translate_tag_in}Text...{translate_tag_out} (Additional comment)"
❌ "Sure! {translate_tag_in}Text...{translate_tag_out}"
❌ "Text..." (missing tags entirely)
❌ "{translate_tag_in}Text..." (missing closing tag)

**CORRECT format (ONLY this):**
✅ {translate_tag_in}
{example_format}
{translate_tag_out}
"""


# Numbering starts at 6 because rules 1-5 are emitted by _get_output_format_section.
_SUBTITLE_FORMAT_RULES = (
    "\n6. Each subtitle has an index marker: [index]text - PRESERVE these markers exactly"
    "\n7. Keep ONE [index] per subtitle - do NOT merge or split subtitles"
    "\n8. Maintain line breaks between indexed subtitles"
    "\n9. Preserve inline tags (<i>, <b>, <u>, <font ...>, {\\an8}, etc.) and any \\n line breaks INSIDE a subtitle exactly as in the source"
)

# Numbering starts at 6 because rules 1-5 are emitted by _get_output_format_section.
# Emitted only in Plain Text Mode (has_placeholders is False): the paragraph-count
# reconciliation downstream cannot absorb a model that merges or splits paragraphs,
# so the contract is stated explicitly here instead of left implicit.
_PLAIN_TEXT_FORMAT_RULES = (
    "\n6. The input contains paragraphs separated by ONE BLANK LINE"
    "\n7. Output EXACTLY the same number of paragraphs, in the same order"
    "\n8. Separate every output paragraph with ONE BLANK LINE - never a single newline"
    "\n9. Do NOT merge two paragraphs into one, and do NOT split one paragraph into two"
    "\n10. An empty input paragraph stays empty - do not fill it"
    "\n11. Do NOT add markdown heading markers (#, ##, ###, etc.) or any other "
    "markdown formatting symbols to your output - output plain translated text only"
)

# prompt_options key carrying the paragraph count of the segment being retried.
# Set by the plain-text pipeline on the single retry it issues after a
# paragraph-count mismatch, and never on a first attempt (issue #253).
PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION = 'plain_text_expected_paragraphs'


def _plain_text_format_rules(prompt_options: Optional[Dict[str, Any]] = None) -> str:
    """Return the Plain Text Mode paragraph contract, hardened on a retry.

    The base contract is what every Plain Text Mode call carries. When the
    caller sets PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION, the model has already
    answered this exact text with the wrong paragraph count, so the retry names
    the number it must produce and spells out the failure observed in practice:
    disclaimers, author's notes and headings folded away as if they were
    metadata (issue #253). Numbering continues from the base rules.
    """
    expected = (prompt_options or {}).get(PLAIN_TEXT_EXPECTED_PARAGRAPHS_OPTION)
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
        return _PLAIN_TEXT_FORMAT_RULES
    return _PLAIN_TEXT_FORMAT_RULES + (
        "\n12. RETRY: your previous answer for this exact text had the WRONG number of paragraphs"
        f"\n13. This input contains EXACTLY {expected} paragraph(s); your output MUST contain "
        f"EXACTLY {expected} paragraph(s), separated by ONE BLANK LINE"
        "\n14. Count the paragraphs before answering. A disclaimer, an author's note, a chapter "
        "heading or any short standalone line IS a paragraph and IS content: translate each one "
        "as its own paragraph - never drop it, never fold it into the next one, never summarize it"
    )


# ============================================================================
# OPTIONAL PROMPT SECTIONS
# ============================================================================

# Technical content preservation section (for technical documents)
TECHNICAL_CONTENT_SECTION = """
**Technical Content (DO NOT TRANSLATE):**
- Code snippets and syntax: `function()`, `variable_name`, `class MyClass`
- Command lines: `npm install`, `git commit -m "message"`
- File paths: `/usr/bin/`, `C:/Users/Documents/`
- URLs: `https://example.com`, `www.site.org`
- Programming identifiers, API names, and technical terms"""

# Text cleanup section (for OCR or poorly formatted source texts)
TEXT_CLEANUP_SECTION = """
# TEXT CLEANUP (Source Defects Correction)

The source text may contain OCR errors, formatting artifacts, or typographic defects.
**CORRECT THESE ISSUES during translation:**

- **Line breaks**: Fix broken words (e.g., "trans-\\nlation" → "translation")
- **Spacing**: Remove double spaces, fix missing spaces after punctuation
- **Punctuation**: Correct misplaced or missing punctuation marks
- **Paragraph flow**: Merge incorrectly split paragraphs, preserve intentional breaks

**DO NOT** add content, remove meaningful text, or alter the author's style."""

# Plain Text Mode variant of TEXT_CLEANUP_SECTION: the downstream pipeline reassembles
# a segment's paragraphs by position (see plain_text_pipeline.py), so it cannot absorb
# a model that merges or splits paragraphs. Derived with .replace() rather than a second
# hand-written block so the two constants can never drift apart on everything but that
# one bullet.
TEXT_CLEANUP_SECTION_PLAIN = TEXT_CLEANUP_SECTION.replace(
    "- **Paragraph flow**: Merge incorrectly split paragraphs, preserve intentional breaks",
    "- **Paragraph flow**: Keep the paragraph breaks exactly as they are - never merge or split paragraphs",
)


def _build_optional_prompt_sections(prompt_options: dict, *, plain_text: bool = False) -> str:
    """
    Build optional prompt sections based on the provided options.

    Args:
        prompt_options: Dictionary containing prompt customization flags:
            - preserve_technical_content: DEPRECATED - Technical content is now protected
              via placeholder system (no prompt section needed)
            - text_cleanup: Include OCR/typographic defect correction instructions
        plain_text: If True, use the Plain Text Mode variant of the text-cleanup section,
            which forbids merging/splitting paragraphs instead of encouraging it. Keyword-only
            so existing callers (e.g. generate_refinement_prompt) keep today's behavior unchanged.

    Returns:
        str: Concatenated optional sections to include in the system prompt
    """
    if prompt_options is None:
        prompt_options = {}

    sections = []

    # Technical content preservation is now handled by the placeholder system
    # (TagPreserver with protect_technical=True), so no prompt instructions are needed.
    # The LLM never sees technical content - it's hidden in placeholders like [id0], [id1].
    # Leaving this commented for reference:
    # if prompt_options.get('preserve_technical_content', False):
    #     sections.append(TECHNICAL_CONTENT_SECTION)

    # Text cleanup for OCR or poorly formatted sources
    if prompt_options.get('text_cleanup', False):
        sections.append(TEXT_CLEANUP_SECTION_PLAIN if plain_text else TEXT_CLEANUP_SECTION)

    # Join sections with double newline for proper separation
    return '\n\n'.join(sections)


# ============================================================================
# TRANSLATION PROMPT FUNCTIONS
# ============================================================================

def generate_translation_prompt(
    main_content: str,
    context_before: str,
    context_after: str,
    previous_translation_context: str,
    source_language: str = "English",
    target_language: str = "English",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    glossary_block: str = "",
) -> PromptPair:
    """
    Generate the translation prompt with all contextual elements.

    Args:
        main_content: The text to translate
        context_before: Text appearing before main_content for context
        context_after: Text appearing after main_content for context
        previous_translation_context: Previously translated text for consistency
        source_language: Source language name
        target_language: Target language name
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        has_placeholders: If True, includes placeholder preservation instructions (for EPUB HTML tags)
        prompt_options: Optional dict with prompt customization options:
            - preserve_technical_content: If True, includes instructions to NOT translate
              code, paths, URLs, etc. (for technical documents)
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    # Initialize prompt_options if not provided
    if prompt_options is None:
        prompt_options = {}

    # Extract custom instructions if provided
    custom_instructions = prompt_options.get('custom_instructions', '')

    # Get target-language-specific example text for output format
    example_texts = {
        "chinese": "您翻译的文本在这里" if not has_placeholders else f"您翻译的文本在这里，所有{TAG0}标记都精确保留",
        "french": "Votre texte traduit ici" if not has_placeholders else f"Votre texte traduit ici, tous les marqueurs {TAG0} sont préservés exactement",
        "spanish": "Su texto traducido aquí" if not has_placeholders else f"Su texto traducido aquí, todos los marcadores {TAG0} se preservan exactamente",
        "german": "Ihr übersetzter Text hier" if not has_placeholders else f"Ihr übersetzter Text hier, alle {TAG0}-Markierungen werden genau beibehalten",
        "japanese": "翻訳されたテキストはこちら" if not has_placeholders else f"翻訳されたテキストはこちら、すべての{TAG0}マーカーは正確に保持されます",
        "italian": "Il tuo testo tradotto qui" if not has_placeholders else f"Il tuo testo tradotto qui, tutti i marcatori {TAG0} sono conservati esattamente",
        "portuguese": "Seu texto traduzido aqui" if not has_placeholders else f"Seu texto traduzido aqui, todos os marcadores {TAG0} são preservados exatamente",
        "russian": "Ваш переведенный текст здесь" if not has_placeholders else f"Ваш переведенный текст здесь, все маркеры {TAG0} сохранены точно",
        "korean": "번역된 텍스트는 여기에" if not has_placeholders else f"번역된 텍스트는 여기에, 모든 {TAG0} 마커는 정확히 보존됩니다",
    }

    # Try to match target language to get appropriate example
    from src.utils.lang_normalize import normalize_lang_key
    target_lang_lower = normalize_lang_key(target_language)
    example_format_text = example_texts.get(target_lang_lower, "Your translated text here")

    # Build the output format section outside the f-string to avoid backslash issues in Python 3.11.
    # The paragraph-structure contract is only emitted in Plain Text Mode (has_placeholders is
    # False): the placeholder path has its own structural contract and must not be perturbed.
    output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules=_plain_text_format_rules(prompt_options) if not has_placeholders else "",
        example_format=example_format_text
    )

    # Build placeholder preservation section dynamically based on languages
    if has_placeholders:
        placeholder_section = build_placeholder_section(source_language, target_language, placeholder_format)
    else:
        placeholder_section = ""

    # Build optional prompt sections based on prompt_options
    optional_sections = _build_optional_prompt_sections(prompt_options, plain_text=not has_placeholders)

    # Build custom instructions section.
    #
    # Deliberately firm but not absolutist. An earlier wording ("ABSOLUTE
    # PRIORITY", "Non-compliance = FAILURE", "Zero exceptions") contradicted
    # the presets themselves: a style preset ends with a guard telling the
    # model to favour natural phrasing whenever a rule fights the passage, so
    # the wrapper was ordering the opposite of its own payload. What the
    # emphasis actually needs to buy is persistence — models apply a style to
    # the opening lines and then drift — which the closing line states without
    # claiming the instructions outrank meaning.
    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = f"""# STYLE INSTRUCTIONS

**Apply these throughout the translation. They take precedence over the general style guidance in this prompt.**

{custom_instructions.strip()}

Keep them in force across the whole passage, not only in its opening lines.

"""

    # SYSTEM PROMPT - Role and instructions (stable across requests)
    system_prompt = f"""You are a professional {target_language} translator and writer.

{custom_instructions_section}# TRANSLATION PRINCIPLES

Translate {source_language} to {target_language}. Output only the translation.

**PRIORITY ORDER:**
1. Preserve exact names
2. Match original tone and formality
3. Use natural {target_language} phrasing - never word-for-word
4. Fix grammar/spelling errors in output
5. Translate idioms to {target_language} equivalents

**QUALITY CHECK:**
- Does it sound natural to a native {target_language} speaker?
- Are all details from the original included?
- Does punctuation follow {target_language} conventions?

If unsure between literal and natural phrasing: **choose natural**.

**LAYOUT PRESERVATION:**
- Keep the exact text layout, spacing, line breaks, and indentation
- **WRITE YOUR TRANSLATION IN {target_language.upper()} - THIS IS MANDATORY**
{optional_sections}
{placeholder_section}

# FINAL REMINDER: YOUR OUTPUT LANGUAGE

**YOU MUST TRANSLATE INTO {target_language.upper()}.**
Your entire translation output must be written in {target_language}.
Do NOT write in {source_language} or any other language - ONLY {target_language.upper()}.

{output_format_section}"""

    # USER PROMPT - Context and content to translate (varies per request)
    previous_translation_block_text = ""
    if previous_translation_context and previous_translation_context.strip():
        previous_translation_block_text = f"""# CONTEXT - Previous Paragraph

For consistency and natural flow, here's what came immediately before:

{previous_translation_context}

"""

    # Glossary block lives in the user prompt: it changes per chunk, so
    # keeping it out of the system prompt lets the system prompt stay
    # stable and cacheable across chunks.
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""

    user_prompt = f"""{previous_translation_block_text}{glossary_section}# TEXT TO TRANSLATE

{INPUT_TAG_IN}
{main_content}
{INPUT_TAG_OUT}

REMINDER: Output ONLY your translation in this exact format:
{translate_tag_in}
your translation here
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your translation now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# GLOSSARY: NER EXTRACTION PROMPT (Phase 2)
# ============================================================================

NER_TAG_IN = "<NER_JSON>"
NER_TAG_OUT = "</NER_JSON>"


def generate_ner_extraction_prompt(
    text: str,
    source_language: str = "Chinese",
    target_language: str = "English",
) -> PromptPair:
    """
    Build a prompt that asks the LLM to extract recurring proper-noun entities
    (characters, locations, organizations/sects, items) from a sample of source
    text, along with a suggested target-language translation for each and, for
    characters, the gender evidenced by the passage.

    The gender field is deliberately evidence-only with an explicit "unknown"
    escape hatch: a sampled excerpt often carries no gender marker at all, and
    a confidently wrong gender is worse than a blank the user reviews.

    Output is wrapped in <NER_JSON>...</NER_JSON> with a strict schema. The
    parser is permissive (handles markdown fences, missing tags, partial JSON).
    """
    system_prompt = f"""You are a literary entity extractor. Your job is to read a passage written in {source_language} and identify recurring proper nouns that a translator would want to keep consistent across an entire book.

# CATEGORIES (use exactly these labels)

- "character"     — named persons (李凡, Li Fan, Captain Ahab)
- "location"      — places, regions, buildings (青玄宗大殿, Mount Tai)
- "organization"  — sects, schools, clans, factions, companies (青玄宗, Heavenly Sword Gate)
- "item"          — named artifacts, weapons, treasures, techniques (混沌珠, Excalibur)
- "title"         — honorifics or named ranks tied to a person (Elder, 长老, Master)
- "other"         — anything else worth keeping consistent (events, magical formulas)

# RULES

1. Extract ONLY proper nouns or named concepts that look likely to recur. Skip generic words.
2. Do NOT translate common nouns or descriptive phrases — only named entities.
3. For each entity, propose ONE canonical {target_language} translation. Use the standard romanization or the most natural literary rendering. Keep the proposal concise.
4. Deduplicate: if the same entity appears multiple times in the passage, list it once.
5. If you are unsure about an entry, omit it rather than guessing.
6. Preserve the original {source_language} form exactly as it appears in the text (no extra spaces, no normalization).

# GENDER

For entities in the "character" category, also report the character's gender. This matters because {source_language} may not mark gender the way {target_language} does: when a passage omits the subject or uses an unmarked pronoun, a translator with no gender information defaults every character to "he" and silently corrupts the cast across a whole book.

Determine gender ONLY from evidence in the passage: gendered pronouns (她 / 彼女 / she), gendered forms of address or kinship terms (姐姐, 母亲, Miss, Lady, sir), or explicit description. Do NOT guess from the name itself, and do NOT assume a protagonist is male.

Report exactly one of:
  - "female"  — the passage shows this character is female
  - "male"    — the passage shows this character is male
  - "unknown" — no gender evidence in the passage, or the entity is not a character

"unknown" is the correct, expected answer whenever the evidence is absent. A wrong guess is far more damaging than an honest "unknown", because the user reviews unknowns but trusts stated genders.

# OUTPUT FORMAT

Return ONLY a JSON array wrapped between {NER_TAG_IN} and {NER_TAG_OUT}. No prose, no explanations.

Each array element MUST be an object with these keys:
  - "source"   (string, required) — the entity in {source_language}
  - "target"   (string, required) — the proposed {target_language} translation
  - "category" (string, required) — one of the labels listed above
  - "gender"   (string, required) — "female", "male", or "unknown"

Example:
{NER_TAG_IN}
[
  {{"source": "李凡", "target": "Li Fan", "category": "character", "gender": "male"}},
  {{"source": "林月", "target": "Lin Yue", "category": "character", "gender": "female"}},
  {{"source": "沈青", "target": "Shen Qing", "category": "character", "gender": "unknown"}},
  {{"source": "青玄宗", "target": "Qingxuan Sect", "category": "organization", "gender": "unknown"}}
]
{NER_TAG_OUT}

If no entities are found, return an empty array: {NER_TAG_IN}[]{NER_TAG_OUT}.

Do NOT wrap the JSON in markdown code fences. Do NOT add commentary before or after the tags."""

    user_prompt = f"""# SOURCE TEXT ({source_language})

{INPUT_TAG_IN}
{text}
{INPUT_TAG_OUT}

Extract the recurring named entities now. Output the JSON array between {NER_TAG_IN} and {NER_TAG_OUT}, nothing else."""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# STYLE EXTRACTION PROMPT (Phase 2)
# ============================================================================

STYLE_TAG_IN = "<STYLE_JSON>"
STYLE_TAG_OUT = "</STYLE_JSON>"


def generate_style_extraction_prompt(
    text: str,
    mode: str,
    source_language: str = "English",
    target_language: str = "English",
) -> PromptPair:
    """
    Build a prompt that asks the LLM to characterize the literary style of a
    sample of text and emit a strict JSON list of atomic, abstract style
    rules, in English.

    Two modes are supported:
      - "source": the passages are the source text about to be translated;
        the rules must make the translation read like the original.
      - "model": the passages are a reference work chosen as a stylistic
        model; the rules must make an unrelated text read like this author.

    Output is wrapped in <STYLE_JSON>...</STYLE_JSON> with a strict schema.
    The rules are deliberately abstract (no quoted vocabulary) so they do
    not turn into a lexical tic once applied across a whole book.

    The schema also carries a "context" field describing the narrative
    setting (era, technological level, social frame) so a translation does
    not reach for anachronistic vocabulary. In "source" mode it is required
    and bounded; in "model" mode it must be the empty string, since a
    stylistic model's own setting must never be imposed on the target text.
    """
    if mode not in ("source", "model"):
        raise ValueError(f"mode must be 'source' or 'model', got {mode!r}")

    if mode == "source":
        role_framing = (
            "The passages below are the SOURCE TEXT that is about to be translated. "
            "Produce rules a translator must follow so that the translation reads like the original."
        )
        context_directive = """# CONTEXT FIELD (mandatory)

The "context" field must be 1 to 3 sentences, at most 400 characters, written in English. Describe the world the passages take place in: the historical period (or its secondary-world equivalent), the technological level, the social and cultural frame, and anything else that would make a modern word feel out of place if used in the translation.

Forbidden in "context": proper nouns, character names, place names, and any summary of the plot.

This field exists so the translator never reaches for a word that belongs to a later era or a different technological level than this setting, even when it would otherwise be the most direct equivalent."""
        context_example = (
            '"context": "A late-medieval frontier town under martial law, with no gunpowder '
            'weapons and a rigid guild hierarchy.",'
        )
    else:
        role_framing = (
            "The passages below are a REFERENCE WORK chosen as a stylistic model. "
            "Produce rules a translator must follow to make an unrelated text read like this author."
        )
        context_directive = """# CONTEXT FIELD (mandatory)

The "context" field MUST be the empty string "". These passages are only a stylistic model for an unrelated text: the reference work's own setting must never be imposed on the text that will actually be translated."""
        context_example = '"context": "",'

    system_prompt = f"""You are a literary style analyst. Your job is to read the passages below and characterize HOW the text is written, never WHAT it says. Ignore plot, characters, and setting entirely; focus only on the craft choices a translator could reproduce.

{role_framing}

# DIMENSIONS (use exactly these labels)

- "register"         — formality, distance, irony, emotional temperature
- "narrative_voice"  — person, tense, focalization, narrator presence
- "sentence_rhythm"  — length distribution, parataxis vs subordination, cadence
- "lexicon"          — concrete vs abstract, recurring lexical fields, archaisms
- "imagery"          — metaphors, similes, recurring figurative motifs
- "dialogue"         — speech tags, orality, idiolects, interruption handling
- "punctuation"      — em-dashes, semicolons, ellipses, exclamation frequency
- "formatting"       — paragraph length, italics usage, section breaks
- "other"            — anything else worth capturing that does not fit above

# LANGUAGE OF THE INSTRUCTIONS

Write every instruction in English, regardless of the language of the passages. This directive holds even when the passages are in one language and the translation target is a different, third language.

# ABSTRACTION DIRECTIVE (mandatory)

Every instruction must describe a PROPERTY of the writing, never the specific words that realize it. Naming specific vocabulary is a trap: the translator will repeat that exact vocabulary across the whole book, and it will read as a tic.

Forbidden, in every instruction:
1. No quoted material from the passages, and no quotation marks at all.
2. No example words, phrases, idioms or turns of phrase to use.
3. No "such as", "e.g.", "for example", "words like", "expressions like".
4. No proper nouns, no invented terminology, no lexical field named as a word list.
5. No instruction that can be satisfied by inserting one specific token.

Describe the CHOICE being made instead: proportion, frequency, position, contrast, degree, consistency.

Example:
- rejected: Use metaphors of darkness and shadow, and words like "dusk" and "gloom".
- accepted: Draw figurative language from a single consistent sensory field rather than varying its source from one image to the next.

# RULES FOR EACH INSTRUCTION

1. One imperative sentence, self-contained, actionable by a translator who has not read the passages.
2. At most 240 characters.
3. Must not mention characters, places, or plot from the passages.
4. Return between 6 and 14 rules in total, at most 3 rules per dimension. Omit a dimension rather than padding it — 6 abstract rules are better than 14 that name vocabulary.

{context_directive}

# OUTPUT FORMAT

Return ONLY a JSON object wrapped between {STYLE_TAG_IN} and {STYLE_TAG_OUT}. No prose, no explanations, no markdown code fences.

The JSON object MUST have these keys:
  - "summary"         (string, required) — one sentence, at most 120 characters, usable as a preset description
  - "suggested_name"  (string, required) — 2 to 4 lowercase words joined by underscores, ASCII only (e.g. "dry_hardboiled_noir")
  - "context"         (string, required) — see the CONTEXT FIELD section above for the exact requirement
  - "rules"           (array, required) — a list of objects, each with:
      - "dimension"   (string, required) — one of the dimension labels listed above
      - "instruction" (string, required) — the abstract, imperative style rule, in English
      - "evidence"    (string, required, at most 120 characters) — a short quotation from the passages that justifies the rule

"evidence" is the ONLY field allowed to quote the passages. It is shown to the human reviewer as justification for the rule and is discarded afterwards — it never becomes part of the preset. This separation is what lets "instruction" stay fully abstract without losing the reviewer's ability to check the claim.

Example:
{STYLE_TAG_IN}
{{
  "summary": "Terse, present-tense narration with clipped dialogue and sparse punctuation.",
  "suggested_name": "terse_present_tense",
  {context_example}
  "rules": [
    {{"dimension": "sentence_rhythm", "instruction": "Favor short, paratactic sentences over long subordinated ones, especially in action passages.", "evidence": "He ran. He did not look back."}}
  ]
}}
{STYLE_TAG_OUT}

Do NOT wrap the JSON in markdown code fences. Do NOT add commentary before or after the tags."""

    if mode == "source":
        language_note = f"The passages are in {source_language}. The rules you produce will be applied to a translation into {target_language}."
    else:
        language_note = f"The passages are already in {target_language}, the target language of the translation these rules will later be applied to."

    user_prompt = f"""# PASSAGES

{language_note}

{INPUT_TAG_IN}
{text}
{INPUT_TAG_OUT}

Produce the JSON now. Output it between {STYLE_TAG_IN} and {STYLE_TAG_OUT}, nothing else."""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_refinement_prompt(
    draft_translation: str,
    context_before: str = "",
    context_after: str = "",
    previous_refined_context: str = "",
    target_language: str = "English",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    additional_instructions: str = "",
    glossary_block: str = "",
) -> PromptPair:
    """
    Generate a refinement prompt to polish a draft translation.

    This is used for a second pass where the LLM improves a first-pass translation,
    focusing on literary quality, natural flow, and stylistic excellence.

    Args:
        draft_translation: The first-pass translation to refine
        context_before: Previously refined text for context (default: "")
        context_after: Text appearing after for context (default: "")
        previous_refined_context: Last refined text for consistency (default: "")
        target_language: Target language name
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        has_placeholders: If True, includes placeholder preservation instructions
        prompt_options: Optional dict with prompt customization options
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format
        additional_instructions: Additional refinement instructions to include in the prompt (default: "")

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    if prompt_options is None:
        prompt_options = {}

    # Get target-language-specific example text for output format
    example_texts = {
        "chinese": "您润色后的文本在这里",
        "french": "Votre texte affiné ici",
        "spanish": "Su texto refinado aquí",
        "german": "Ihr verfeinerter Text hier",
        "japanese": "洗練されたテキストはこちら",
        "italian": "Il tuo testo raffinato qui",
        "portuguese": "Seu texto refinado aqui",
        "russian": "Ваш улучшенный текст здесь",
        "korean": "다듬어진 텍스트는 여기에",
    }

    from src.utils.lang_normalize import normalize_lang_key
    target_lang_lower = normalize_lang_key(target_language)
    example_format_text = example_texts.get(target_lang_lower, "Your refined text here")

    output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules="",
        example_format=example_format_text
    )

    # Build placeholder preservation section if needed
    if has_placeholders:
        placeholder_section = build_placeholder_section(target_language, target_language, placeholder_format)
    else:
        placeholder_section = ""

    # Build optional prompt sections
    optional_sections = _build_optional_prompt_sections(prompt_options)

    # Add additional instructions section if provided
    additional_instructions_section = ""
    if additional_instructions and additional_instructions.strip():
        additional_instructions_section = f"""

# ADDITIONAL REFINEMENT INSTRUCTIONS

{additional_instructions.strip()}"""

    # SYSTEM PROMPT for refinement
    system_prompt = f"""You are an elite {target_language} literary editor and prose stylist.

# YOUR TASK: REFINE AND POLISH

You will receive a DRAFT {target_language} translation that needs significant improvement.
Your job is to REWRITE it with perfect literary {target_language} style.

**THE INPUT IS:**
- A amator, literal, or awkward {target_language} translation
- It may have unnatural phrasing, stilted expressions, or poor flow
- Consider it a "bad" first draft that probably needs substantial reworking

**YOUR OUTPUT MUST BE:**
- Fluent, natural {target_language} prose
- Stylistically excellent - as if written by a skilled {target_language} author

# REFINEMENT PRINCIPLES

**PRIORITY ORDER:**
1. **Natural flow** - Sentences should flow beautifully in {target_language}
2. **Idiomatic expressions** - Use natural {target_language} idioms and phrasings
3. **Elegant word choice** - Select the most appropriate and refined vocabulary
4. **Rhythm and cadence** - The text should have pleasant reading rhythm
5. **Preserve meaning** - Keep the original meaning intact while improving style

**WHAT TO FIX:**
- Awkward literal translations → Natural {target_language} expressions
- Repetitive or dull vocabulary → Rich, varied word choices
- Unnatural word order → Proper {target_language} syntax
- **Lexical repetitions and cacophony** → Use synonyms to avoid same-root word repetition
  (e.g., "the singer sang a song" → "the singer performed a song" or "the vocalist sang a melody")

**WHAT TO PRESERVE:**
- All factual content and meaning
- Character names and proper nouns
- Technical terms (if any)
{optional_sections}
{placeholder_section}
{additional_instructions_section}

# CRITICAL REMINDER

You are NOT translating - you are REWRITING in {target_language.upper()}.
The input is already in {target_language}, but poorly written.
Your output must be polished, literary-quality {target_language}.

**⚠️ PLACEHOLDER PRESERVATION IS ABSOLUTELY CRITICAL:**
If the input contains ANY placeholders (like [id0], [id1], etc.), you MUST preserve them EXACTLY.
Removing or corrupting placeholders will corrupt the document structure.
Your refinement MUST maintain the exact same placeholders in the exact same positions.

{output_format_section}"""

    # USER PROMPT
    previous_context_block = ""
    if previous_refined_context and previous_refined_context.strip():
        previous_context_block = f"""# CONTEXT - Previous Refined Paragraph

For consistency and natural flow, here's what came immediately before:

{previous_refined_context}

"""

    # Glossary block injected here (per-chunk dynamic) so the system prompt
    # stays cacheable across chunks.
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""

    user_prompt = f"""{previous_context_block}{glossary_section}# DRAFT TO REFINE

The following is a rough {target_language} translation that needs significant improvement.
Rewrite it with elegant, literary-quality {target_language} prose:

{INPUT_TAG_IN}
{draft_translation}
{INPUT_TAG_OUT}

REMINDER: Output ONLY your refined text in this exact format:
{translate_tag_in}
your refined text here
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your refined version now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_subtitle_refinement_block_prompt(
    subtitle_blocks: List[Tuple[int, str]],
    previous_refined_block: str = "",
    target_language: str = "English",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    additional_instructions: str = "",
    glossary_block: str = "",
) -> PromptPair:
    """
    Generate a refinement prompt for multiple subtitles in a single LLM call.

    Mirrors generate_subtitle_block_prompt but rewrites each draft subtitle into
    polished target-language prose while preserving the [index] markers.

    Args:
        subtitle_blocks: List of tuples (local_index, draft_translated_text)
        previous_refined_block: Last refined block for continuity
        target_language: Target language
        translate_tag_in: Opening tag for refinement output
        translate_tag_out: Closing tag for refinement output
        additional_instructions: Extra refinement guidance
        glossary_block: Optional glossary block

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    subtitle_additional_rules = _SUBTITLE_FORMAT_RULES
    subtitle_example_format = "[0]Première ligne affinée\n[1]Deuxième ligne affinée"
    subtitle_output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules=subtitle_additional_rules,
        example_format=subtitle_example_format,
    )

    additional_instructions_section = ""
    if additional_instructions and additional_instructions.strip():
        additional_instructions_section = f"""

# ADDITIONAL REFINEMENT INSTRUCTIONS

{additional_instructions.strip()}"""

    system_prompt = f"""You are an elite {target_language} subtitle editor and dialogue stylist.

# YOUR TASK: REFINE A BLOCK OF SUBTITLES

You will receive a block of DRAFT {target_language} subtitles, each prefixed with an [index] marker.
Your job is to REWRITE each subtitle with natural, idiomatic {target_language} dialogue while
preserving the index markers and the one-subtitle-per-marker structure.

**THE INPUT IS:**
- A block of draft {target_language} subtitles, possibly literal or awkward
- Each subtitle is prefixed with [N] where N is its local index

**YOUR OUTPUT MUST BE:**
- The same number of subtitles, each prefixed with the SAME [N] marker
- Fluent, natural spoken {target_language} suited to subtitling

# REFINEMENT PRINCIPLES

**PRIORITY ORDER:**
1. **Natural dialogue** - sound like real {target_language} speech, not translation
2. **Reading speed** - keep subtitle length viewer-friendly
3. **Continuity** - terminology and tone consistent across the block
4. **Preserve meaning** - keep the original meaning intact while improving style

**WHAT TO FIX:**
- Awkward literal phrasing -> natural {target_language} expressions
- Repetitive vocabulary that is clearly an artefact of literal translation -> varied word choices
- Unnatural word order -> proper {target_language} syntax

**WHAT TO PRESERVE:**
- The [index] markers exactly as given
- All factual content and meaning
- Character names and proper nouns
- The one-subtitle-per-[index] structure (no merging, no splitting)
- Intentional repetitions (e.g. "No. No. No.") and dialogue dashes ("- ...\\n- ...") when present in the draft
- Inline formatting tags and any \\n line breaks inside a subtitle{additional_instructions_section}

# CRITICAL REMINDERS

You are NOT translating - you are REWRITING in {target_language.upper()}.
The input is already in {target_language}, but possibly poorly written.
Your output must be polished, natural {target_language} dialogue.

**Index markers are MANDATORY:** every input [N] must appear exactly once in the output,
in the same order, followed by the refined text for that subtitle.

{subtitle_output_format_section}"""

    previous_refined_block_text = ""
    if previous_refined_block and previous_refined_block.strip():
        previous_refined_block_text = f"""# CONTEXT - Previous Refined Block

For continuity and consistency, here's the previous refined block:

{previous_refined_block}

"""

    formatted_subtitles = [f"[{idx}]{text}" for idx, text in subtitle_blocks]
    formatted_subtitles_text = "\n".join(formatted_subtitles)

    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""

    user_prompt = f"""{previous_refined_block_text}{glossary_section}# SUBTITLES TO REFINE

{INPUT_TAG_IN}
{formatted_subtitles_text}
{INPUT_TAG_OUT}

REMINDER: Output format must be:
{translate_tag_in}
[0]refined subtitle 0
[1]refined subtitle 1
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your refined block now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


def generate_subtitle_block_prompt(
    subtitle_blocks: List[Tuple[int, str]],
    previous_translation_block: str,
    source_language: str = "English",
    target_language: str = "English",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    custom_instructions: str = "",
    glossary_block: str = "",
) -> PromptPair:
    """
    Generate translation prompt for multiple subtitle blocks with index markers.

    Args:
        subtitle_blocks: List of tuples (index, text) for subtitles to translate
        previous_translation_block: Previous translated block for context
        source_language: Source language
        target_language: Target language
        translate_tag_in: Opening tag for translation output
        translate_tag_out: Closing tag for translation output
        custom_instructions: Additional custom translation instructions

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    # Build the output format section outside the f-string to avoid backslash issues in Python 3.11
    subtitle_additional_rules = _SUBTITLE_FORMAT_RULES
    subtitle_example_format = "[1]第一行翻译文本\n[2]第二行翻译文本"
    subtitle_output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules=subtitle_additional_rules,
        example_format=subtitle_example_format
    )

    # Build custom instructions section if provided
    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = f"""

# STYLE INSTRUCTIONS

**Apply these throughout the translation. They take precedence over the general style guidance in this prompt.**

{custom_instructions.strip()}

Keep them in force across every subtitle in the batch, not only the first few.
"""

    # SYSTEM PROMPT - Role and instructions for subtitle translation
    system_prompt = f"""You are a professional {target_language} subtitle translator and dialogue adaptation specialist.

# CRITICAL: TARGET LANGUAGE IS {target_language.upper()}

**YOUR SUBTITLE TRANSLATION MUST BE WRITTEN ENTIRELY IN {target_language.upper()}.**

You are translating subtitles FROM {source_language} TO {target_language}.
Your output must be in {target_language} ONLY - do NOT use any other language.

# SUBTITLE TRANSLATION PRINCIPLES

**Quality Standards:**
- Translate dialogues naturally and conversationally for {target_language} viewers
- Adapt expressions, slang, and cultural references appropriately
- Keep subtitle length readable (typically 40-42 characters per line)
- Restructure sentences naturally (avoid word-by-word translation)
- Maintain speaker's tone, personality, and emotion
- **WRITE YOUR TRANSLATION IN {target_language.upper()} - THIS IS MANDATORY**

**Subtitle-Specific Rules:**
- Prioritize clarity and reading speed over literal accuracy
- Condense when necessary without losing meaning
- Use natural, spoken {target_language} (not formal written style)
- Preserve intentional repetitions (e.g. "No. No. No.") and dialogue dashes ("- ...\\n- ...") from the source
- Preserve inline formatting tags (<i>, <b>, <font ...>, {{\\an8}}, etc.) and any \\n line breaks inside a subtitle{custom_instructions_section}

# FINAL REMINDER: YOUR OUTPUT LANGUAGE

**YOU MUST TRANSLATE INTO {target_language.upper()}.**
Your entire subtitle translation must be written in {target_language}.
Do NOT write in {source_language} or any other language - ONLY {target_language.upper()}.

{subtitle_output_format_section}"""

    # USER PROMPT - Context and subtitles to translate
    previous_translation_block_text = ""
    if previous_translation_block and previous_translation_block.strip():
        previous_translation_block_text = f"""# CONTEXT - Previous Subtitle Block

For continuity and consistency, here's the previous subtitle block:

{previous_translation_block}

"""

    # Format subtitle blocks with indices
    formatted_subtitles = [f"[{idx}]{text}" for idx, text in subtitle_blocks]

    # Join subtitles outside f-string to avoid Python 3.11 backslash issues
    formatted_subtitles_text = "\n".join(formatted_subtitles)

    # Glossary block in user prompt (dynamic per chunk).
    glossary_section = f"{glossary_block}\n" if glossary_block and glossary_block.strip() else ""

    user_prompt = f"""{previous_translation_block_text}{glossary_section}# SUBTITLES TO TRANSLATE

{INPUT_TAG_IN}
{formatted_subtitles_text}
{INPUT_TAG_OUT}

REMINDER: Output format must be:
{translate_tag_in}
[1]translated subtitle 1
[2]translated subtitle 2
{translate_tag_out}

Start with {translate_tag_in} and end with {translate_tag_out}. Nothing before or after.

Provide your translation now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# PLACEHOLDER CORRECTION PROMPT
# ============================================================================

def generate_placeholder_correction_prompt(
    original_text: str,
    translated_text: str,
    specific_errors: str,
    source_language: str,
    target_language: str,
    expected_count: int,
    placeholder_format: Optional[Tuple[str, str]] = None
) -> PromptPair:
    """
    Generate a prompt for correcting placeholder errors in a translation.

    This prompt is used when a translation has placeholder issues (missing,
    duplicated, mutated, or out of order). It asks the LLM to fix ONLY the
    placeholder positions without modifying the translated text.

    Args:
        original_text: Source text with correct placeholders
        translated_text: Translation with placeholder errors
        specific_errors: Detailed error description (generated by build_specific_error_details)
        source_language: Source language name (e.g., "English")
        target_language: Target language name (e.g., "French")
        expected_count: Number of placeholders expected (0 to expected_count-1)
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format.
            If None, uses default [[0]] format

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    # Use custom format if provided, otherwise use defaults
    if placeholder_format:
        prefix, suffix = placeholder_format
    else:
        prefix, suffix = PLACEHOLDER_PREFIX, PLACEHOLDER_SUFFIX

    # Generate dynamic placeholder examples using the correct format
    def make_placeholder(idx: int) -> str:
        return f"{prefix}{idx}{suffix}"

    max_index = expected_count - 1 if expected_count > 0 else 0
    placeholder_format_str = f"{prefix}N{suffix}"
    example_range = f"{make_placeholder(0)} to {make_placeholder(max_index)}"
    placeholder_list = ", ".join(make_placeholder(i) for i in range(min(3, expected_count)))
    if expected_count > 3:
        placeholder_list += ", etc."

    # SYSTEM PROMPT
    system_prompt = f"""You are a technical placeholder correction specialist.

## YOUR TASK

A {source_language} to {target_language} translation was performed, but the placeholders were corrupted.
You must fix the placeholder positions to match the original text structure.

## PLACEHOLDER FORMAT

**CORRECT format:** {make_placeholder(0)}, {make_placeholder(1)}, {make_placeholder(2)}, etc.
- Brackets: {prefix} and {suffix}
- Sequential numbering starting from 0
- Expected range for this text: {example_range}

**FORMAT VARIATIONS:**
The system uses different placeholder formats based on text content:
- [id0], [id1], [id2]... (default - semantic markers, highest accuracy)
- /0, /1, /2... (when text contains brackets)
- $0$, $1$, $2$... (when text contains brackets and slashes)
- [[0]], [[1]], [[2]]... (legacy format)

All formats follow the same rules: preserve exact format, maintain sequential order, keep position.

## HOW TO POSITION PLACEHOLDERS

Placeholders represent HTML/XML tags. To position them correctly:

1. **Look at the ORIGINAL text** to see what content each placeholder surrounds
2. **Find the equivalent content** in the translation
3. **Place the placeholder at the same logical position** around that content

**Example:**
- Original: "{make_placeholder(0)}Hello{make_placeholder(1)} world"
- If translation is "Bonjour monde", the placeholders mark "Hello"
- Correct: "{make_placeholder(0)}Bonjour{make_placeholder(1)} monde"

## VALIDATION RULES

1. **EXACT COUNT**: Must contain exactly {expected_count} placeholders
2. **SEQUENTIAL ORDER**: Placeholders must appear in order: {placeholder_list}
3. **NO DUPLICATES**: Each placeholder must appear exactly once
4. **NO MUTATIONS**: Use ONLY the {placeholder_format_str} format
5. **POSITION MATCHING**: Each placeholder must surround the translated equivalent of what it surrounded in the original

## CRITICAL INSTRUCTIONS

- Analyze the ORIGINAL to understand what each placeholder marks
- Position placeholders around the SAME semantic content in the translation
- Do NOT add or remove words from the translation
- Keep the {target_language} text intact, only fix placeholder positions

## OUTPUT FORMAT

Your response MUST start with {CORRECTED_TAG_IN} and end with {CORRECTED_TAG_OUT}.
Include NOTHING before or after these tags."""

    # USER PROMPT
    user_prompt = f"""## ORIGINAL TEXT ({source_language}) - Reference for placeholder positions:

<ORIGINAL_TAG_IN>
{original_text}
<ORIGINAL_TAG_OUT>

## TRANSLATION WITH ERRORS ({target_language}):

<TRANSLATION_TAG_IN>
{translated_text}
<TRANSLATION_TAG_OUT>

## DETECTED ERRORS:

{specific_errors}

## YOUR TASK:

Reposition the placeholders {example_range} in the translation above.
Keep the translated text unchanged - only fix placeholder positions.

Provide your corrected version now:"""

    return PromptPair(system=system_prompt.strip(), user=user_prompt.strip())


# ============================================================================
# ALIAS FOR BACKWARDS COMPATIBILITY
# ============================================================================

def generate_post_processing_prompt(
    translated_text: str,
    target_language: str = "English",
    context_before: str = "",
    context_after: str = "",
    additional_instructions: str = "",
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    glossary_block: str = "",
) -> PromptPair:
    """
    Alias for generate_refinement_prompt with parameter name mapping.

    This function exists for backwards compatibility and to provide a more intuitive
    API for post-processing/refinement use cases.

    Args:
        translated_text: The draft translation to refine (mapped to draft_translation)
        target_language: Target language name
        context_before: Previously refined text for context
        context_after: Text appearing after for context
        additional_instructions: Additional refinement instructions
        has_placeholders: If True, includes placeholder preservation instructions
        prompt_options: Optional dict with prompt customization options
        placeholder_format: Optional tuple of (prefix, suffix) for placeholders

    Returns:
        PromptPair: A named tuple with 'system' and 'user' prompts
    """
    return generate_refinement_prompt(
        draft_translation=translated_text,
        context_before=context_before,
        context_after=context_after,
        previous_refined_context="",  # Not used in post-processing calls
        target_language=target_language,
        has_placeholders=has_placeholders,
        prompt_options=prompt_options,
        placeholder_format=placeholder_format,
        additional_instructions=additional_instructions,
        glossary_block=glossary_block,
    )
