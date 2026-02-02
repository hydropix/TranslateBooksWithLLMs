from typing import List, NamedTuple, Tuple, Optional

from prompts.examples import (build_placeholder_section,
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


def _build_optional_prompt_sections(prompt_options: dict) -> str:
    """
    Build optional prompt sections based on the provided options.

    Args:
        prompt_options: Dictionary containing prompt customization flags:
            - preserve_technical_content: DEPRECATED - Technical content is now protected
              via placeholder system (no prompt section needed)
            - text_cleanup: Include OCR/typographic defect correction instructions

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
        sections.append(TEXT_CLEANUP_SECTION)

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
    target_language: str = "Chinese",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None
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
    target_lang_lower = target_language.lower()
    example_format_text = example_texts.get(target_lang_lower, "Your translated text here")

    # Build the output format section outside the f-string to avoid backslash issues in Python 3.11
    output_format_section = _get_output_format_section(
        translate_tag_in,
        translate_tag_out,
        INPUT_TAG_IN,
        INPUT_TAG_OUT,
        additional_rules="",
        example_format=example_format_text
    )

    # Build placeholder preservation section dynamically based on languages
    if has_placeholders:
        placeholder_section = build_placeholder_section(source_language, target_language, placeholder_format)
    else:
        placeholder_section = ""

    # Build optional prompt sections based on prompt_options
    optional_sections = _build_optional_prompt_sections(prompt_options)

    # Build custom instructions section
    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = f"""# ⚠️ MANDATORY STYLE INSTRUCTIONS - ABSOLUTE PRIORITY ⚠️

**These instructions override ALL other guidelines. Non-compliance = FAILURE.**

{custom_instructions.strip()}

⚠️ Apply to EVERY word you translate. Zero exceptions. ⚠️

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

    user_prompt = f"""{previous_translation_block_text}# TEXT TO TRANSLATE

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


def generate_refinement_prompt(
    draft_translation: str,
    context_before: str = "",
    context_after: str = "",
    previous_refined_context: str = "",
    target_language: str = "Chinese",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    additional_instructions: str = ""
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

    target_lang_lower = target_language.lower()
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

    user_prompt = f"""{previous_context_block}# DRAFT TO REFINE

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


def generate_subtitle_block_prompt(
    subtitle_blocks: List[Tuple[int, str]],
    previous_translation_block: str,
    source_language: str = "English",
    target_language: str = "Chinese",
    translate_tag_in: str = TRANSLATE_TAG_IN,
    translate_tag_out: str = TRANSLATE_TAG_OUT,
    custom_instructions: str = ""
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
    subtitle_additional_rules = "\n6. Each subtitle has an index marker: [index]text - PRESERVE these markers exactly\n7. Maintain line breaks between indexed subtitles"
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

# ⚠️ MANDATORY STYLE INSTRUCTIONS - ABSOLUTE PRIORITY ⚠️

**These instructions override ALL other guidelines. Non-compliance = FAILURE.**

{custom_instructions.strip()}

⚠️ Apply to EVERY subtitle. Zero exceptions. ⚠️
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
- Use natural, spoken {target_language} (not formal written style){custom_instructions_section}

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

    user_prompt = f"""{previous_translation_block_text}# SUBTITLES TO TRANSLATE

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
    target_language: str = "Chinese",
    context_before: str = "",
    context_after: str = "",
    additional_instructions: str = "",
    has_placeholders: bool = True,
    prompt_options: dict = None,
    placeholder_format: Optional[Tuple[str, str]] = None
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
        additional_instructions=additional_instructions
    )
