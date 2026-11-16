"""
The ``{translatedName}`` placeholder of an output file name.

The naming pattern of the web UI (or a CLI ``-o`` value) may carry the literal
token ``{translatedName}``: the source file's name, translated into the target
language by the job's own model. The runner resolves it once, in the prepare
phase and before the checkpoint is created, so the name is fixed for the whole
job: a pause/resume, a ``[partial]`` EPUB, the refine pass and TTS never see
the token.

One LLM call per job. It can never fail the job: a refine-only or term-fix
job, a missing source name, a failed request or a rejected answer all fall
back to the original name.
"""

from __future__ import annotations

import os
import re
from pathlib import PureWindowsPath
from typing import Callable, List, Optional, Tuple

from src.config import (
    DEFAULT_MODEL,
    INPUT_TAG_IN,
    INPUT_TAG_OUT,
    TRANSLATE_TAG_IN,
    TRANSLATE_TAG_OUT,
)

from .config import INPUT_FILENAME_MAX_CHARS, OP_TRANSLATE, JobConfig

TOKEN = '{translatedName}'

# The file is written when the job ends, possibly hours later, so a name the
# filesystem refuses must be impossible here. Most filesystems cap a name at
# 255 bytes; this leaves room for a " (1)" collision suffix, the "[partial] "
# prefix and ".pre-refine".
MAX_NAME_BYTES = 220

# Replaced in a name: path separators, the characters Windows refuses, control
# characters, braces (an answer can never smuggle in another placeholder) and
# the bidirectional controls that can make a name display as something else.
_UNSAFE_CHARS_RE = re.compile(r'[<>:"|?*/\\{}\x00-\x1f\u202a-\u202e\u2066-\u2069]+')

# The example the prompt shows in its output format; a model that only echoes
# the prompt returns it as the answer.
_EXAMPLE_ANSWER = "your translation here"

# A small model often wraps its answer in quotes despite the prompt.
_QUOTE_PAIRS = ('""', "''", '“”', '‘’', '«»', '「」', '『』')


def source_stem(job: JobConfig) -> str:
    """The source file's name without its extension ('' when the job has none)."""
    return os.path.splitext(os.path.basename(job.input_filename or ''))[0].strip()


def safe_stem(text: str) -> str:
    """Reduce a name to a safe file-name stem ('' when nothing is left)."""
    # A lone surrogate (half an emoji from a streamed answer, a file name that
    # is not valid UTF-8) cannot be written to disk: drop what does not encode.
    cleaned = text.encode('utf-8', errors='ignore').decode('utf-8')
    cleaned = ' '.join(cleaned.split())  # tabs and line breaks become single spaces
    cleaned = _UNSAFE_CHARS_RE.sub('_', cleaned).strip('. ')
    # A device name ('CON', 'NUL', ...) opens the device instead of a file on
    # older Windows.
    if cleaned and PureWindowsPath(cleaned).is_reserved():
        cleaned = f"_{cleaned}"
    return cleaned


def provisional_name(job: JobConfig) -> str:
    """The output file name with the placeholder filled by the original name:
    what the job is called until (and unless) the translation succeeds."""
    name = job.output_filename
    if TOKEN not in name:
        return name
    stem = safe_stem(source_stem(job)) or 'translation'
    # Only the original name is clipped (on a character boundary): the rest
    # of the pattern, language and extension included, is kept whole.
    stem = stem.encode('utf-8')[:_stem_budget(name)].decode('utf-8', errors='ignore')
    return name.replace(TOKEN, stem.rstrip('. '))


def _stem_budget(name: str) -> int:
    """Bytes each placeholder of ``name`` may take for the whole file name to
    stay within MAX_NAME_BYTES."""
    fixed = len(name.replace(TOKEN, '').encode('utf-8', errors='ignore'))
    return max(MAX_NAME_BYTES - fixed, 0) // name.count(TOKEN)


def build_name_prompt(stem: str, source_language: str, target_language: str) -> Tuple[str, str]:
    """The (system, user) prompt pair of the single file-name request."""
    system_prompt = f"""You are translating the file name of a document from {source_language} into {target_language}.

RULES
1. Translate the meaning into natural {target_language}. Do not romanize, do not explain, do not comment.
2. It stays a file name: a single line, no file extension, no quotation marks, no path.
3. Keep every number exactly as it is (volume, chapter or episode numbers, years, identifiers) and translate the words around it.
4. Do not translate a person's name: write it the way it is usually written in {target_language}.
5. The answer must be written entirely in {target_language}.

OUTPUT FORMAT
{TRANSLATE_TAG_IN}{_EXAMPLE_ANSWER}{TRANSLATE_TAG_OUT}

Emit exactly this and nothing else: no preamble, no explanation, no markdown code fences."""

    # Same closing reminder as the chunk prompts (src/prompts/prompts.py):
    # without it a small model often answers with the bare name, untagged.
    user_prompt = f"""# FILE NAME TO TRANSLATE ({source_language})

{INPUT_TAG_IN}
{stem}
{INPUT_TAG_OUT}

REMINDER: Output ONLY the translated file name in this exact format:
{TRANSLATE_TAG_IN}{_EXAMPLE_ANSWER}{TRANSLATE_TAG_OUT}

Start with {TRANSLATE_TAG_IN} and end with {TRANSLATE_TAG_OUT}. Nothing before or after.

Provide your translation now:"""

    return system_prompt.strip(), user_prompt.strip()


def _unquote(candidate: str) -> str:
    """The answer without the one pair of quotes a model may wrap it in."""
    candidate = candidate.strip()
    if len(candidate) > 2 and candidate[0] + candidate[-1] in _QUOTE_PAIRS:
        candidate = candidate[1:-1]
    return candidate


def _numbers(text: str) -> List[int]:
    return [int(n) for n in re.findall(r'\d+', text)]


def _rejection(candidate: Optional[str], stem: str, target_language: str) -> Optional[str]:
    """Why ``candidate`` must not name the file, or None when it is acceptable."""
    # The acceptance rules of the EPUB title (empty, several lines, source
    # script left over, runaway length), plus one of this module's own.
    from src.core.epub.metadata_translator import _rejection_reason
    reason = _rejection_reason(candidate, stem, target_language, single_line=True)
    if reason is None and candidate.strip().lower() == _EXAMPLE_ANSWER:
        reason = "the model echoed the prompt's example"
    if reason is None and not set(_numbers(stem)) <= set(_numbers(candidate)):
        # "Vol. 2" must not become "Vol. II" or lose its number: two volumes
        # of a series would end up under the same name.
        reason = "a number of the original name is missing from the answer"
    return reason


def _wants_translation(job: JobConfig, stem: str) -> bool:
    """Only a real translation has a name to translate: a refine-only or
    term-fix job writes the language it read."""
    if len(job.input_filename or '') >= INPUT_FILENAME_MAX_CHARS:
        # The name may have been cut at that length and have lost its end (a
        # volume number): a clipped name is never sent for translation.
        return False
    source = (job.source_language or '').strip().lower()
    target = (job.target_language or '').strip().lower()
    return bool(stem) and job.operation == OP_TRANSLATE and bool(target) and source != target


async def translate_stem(job: JobConfig, stem: str, max_bytes: int,
                         log: Callable) -> Optional[str]:
    """Ask the job's model for the translated name. None on any failure,
    a name longer than ``max_bytes`` included."""
    from src.core.epub.llm_setup import create_llm_client, initial_context_size

    try:
        # The context window the engines start the job with: any other size
        # would make a local server (Ollama) load the model twice.
        initial_context, _ = initial_context_size(
            job.model or DEFAULT_MODEL, job.auto_adjust_context, job.context_window or 2048)
        client = create_llm_client(job, initial_context, log)
        try:
            system_prompt, user_prompt = build_name_prompt(
                stem, job.source_language, job.target_language)
            response = await client.generate(user_prompt, system_prompt=system_prompt)
            # Only a tagged answer counts: the raw text of a reply without
            # tags may be a one-line refusal or a model's reasoning.
            content = getattr(response, 'content', None)
            candidate = client.extract_translation(content) if content else None
            if candidate:
                candidate = _unquote(candidate)
        finally:
            await client.close()
    except Exception as exc:  # incl. RateLimitError: a file name never pauses a job
        log("output_name_translation_failed",
            f"⚠️ Output file name kept in {job.source_language}, the request failed: {exc}")
        return None

    reason = _rejection(candidate, stem, job.target_language)
    translated = ''
    if reason is None:
        # A model that echoes the source extension would give "Name.epub (English).epub".
        echoed_ext = os.path.splitext(job.input_filename or '')[1]
        if echoed_ext and candidate.lower().endswith(echoed_ext.lower()):
            candidate = candidate[:-len(echoed_ext)]
        translated = safe_stem(candidate)
        if not translated.strip('_ '):  # '???' would otherwise name the file '_'
            reason = "nothing usable was left of the answer"
        elif len(translated.encode('utf-8')) > max_bytes:
            # Clipping it could cut off the volume number: keep the original.
            reason = "the answer is too long for a file name"
    if reason is not None:
        log("output_name_translation_rejected",
            f"⚠️ Output file name kept in {job.source_language}: {reason}")
        return None
    return translated


async def resolve_output_name(job: JobConfig, log: Callable) -> JobConfig:
    """Replace ``{translatedName}`` in the job's output file name.

    Only the file name is touched, never its directory. Returns the job
    unchanged when the name does not carry the token.
    """
    if TOKEN not in job.output_filename:
        return job
    stem = source_stem(job)
    resolved = provisional_name(job)
    budget = _stem_budget(job.output_filename)  # 0: the pattern alone fills the name
    if budget and _wants_translation(job, stem):
        log("output_name_translation_start", f"🏷️ Translating the output file name with {job.model}...")
        translated = await translate_stem(job, stem, budget, log)
        if translated:
            log("output_name_translated", f"🏷️ Output file name: {stem} → {translated}")
            resolved = job.output_filename.replace(TOKEN, translated)
    return job.with_changes(output_path=os.path.join(job.output_dir, resolved))
