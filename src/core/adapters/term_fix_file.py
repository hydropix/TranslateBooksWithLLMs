"""Unified term-fix entry point.

Applies correction rules to an already-translated file: only the occurrences
of the wrong renderings change, every other byte is written back as read.

Each format exposes its editable text as a list of units:
- TXT and SRT: the whole file (cue numbers and timings cannot match a term);
- EPUB: every text segment of every XHTML/NCX file, edited in the raw markup;
  zip entries without a change are copied as they are;
- DOCX: every paragraph, edited run by run so formatting stays in place.

Literal mode replaces deterministically. Smart mode sends sentence snippets
to the LLM so the term is inflected; see docs/design/TERM_FIX_PASS.md.
"""

import os
import shutil
import zipfile
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .exceptions import UnsupportedFormatError
from src.core.term_fix import (
    MODE_LITERAL,
    MODE_SMART,
    MODES,
    STATUS_NEEDS_REVIEW,
    STATUS_SKIPPED,
    RuleScan,
    SmartCorrector,
    TermFixResult,
    TermRule,
    check_literal_rules,
    fix_units,
    line_locator,
    render_report,
    report_path_for,
    scan_units,
)
from src.core.term_fix.formats import (
    decode_segment,
    distribute_edits,
    encode_segment,
    xml_text_segments,
)
from src.utils.file_detector import detect_file_type, detect_file_type_by_content

SUPPORTED_TYPES = ("txt", "srt", "epub", "docx")

_EPUB_MARKUP_EXTENSIONS = (".xhtml", ".html", ".htm")
_EPUB_NCX_EXTENSION = ".ncx"


def _resolve_type(input_filepath: str) -> str:
    try:
        detected = detect_file_type(input_filepath)
    except ValueError:
        detected = detect_file_type_by_content(input_filepath)
    if detected not in SUPPORTED_TYPES:
        raise UnsupportedFormatError(
            f"Term fix does not support '{detected or 'unknown'}' files "
            f"(supported: {', '.join(SUPPORTED_TYPES)}).",
            file_type=detected,
        )
    return detected


class _TextDocument:
    """TXT / SRT: one unit, the whole file."""

    def __init__(self, path: str):
        # newline='' keeps CRLF/LF exactly as in the file.
        with open(path, "r", encoding="utf-8", newline="") as f:
            self.units = [f.read()]
        self.locate = line_locator(self.units[0])

    def write(self, output_path: str, result: TermFixResult) -> None:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write(result.texts[0])


class _EpubDocument:
    """EPUB: text segments of the content files, edited in the raw markup."""

    def __init__(self, path: str):
        self.path = path
        self.raw: Dict[str, str] = {}
        # unit index -> (entry name, segment start, segment end, raw segment)
        self.slots: List[Tuple[str, int, int, str]] = []
        self.units: List[str] = []
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                name = info.filename
                lower = name.lower()
                is_markup = lower.endswith(_EPUB_MARKUP_EXTENSIONS)
                if not (is_markup or lower.endswith(_EPUB_NCX_EXTENSION)):
                    continue
                try:
                    markup = zf.read(info).decode("utf-8")
                except UnicodeDecodeError:
                    continue  # not UTF-8: leave the entry untouched
                self.raw[name] = markup
                for start, end in xml_text_segments(markup, body_only=is_markup):
                    segment = markup[start:end]
                    self.slots.append((name, start, end, segment))
                    self.units.append(decode_segment(segment))

    def locate(self, unit_index: int, _pos: int) -> str:
        return self.slots[unit_index][0]

    def write(self, output_path: str, result: TermFixResult) -> None:
        changed: Dict[str, List[Tuple[int, int, str]]] = {}
        for idx, (name, start, end, segment) in enumerate(self.slots):
            if not result.edits[idx]:
                continue
            new_text = result.texts[idx]
            if "&" in segment or any(c in new_text for c in "&<>"):
                new_segment = encode_segment(new_text)
            else:
                new_segment = new_text
            changed.setdefault(name, []).append((start, end, new_segment))

        with zipfile.ZipFile(self.path) as zin, \
                zipfile.ZipFile(output_path, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info)
                if info.filename in changed:
                    markup = self.raw[info.filename]
                    for start, end, new_segment in sorted(changed[info.filename], reverse=True):
                        markup = markup[:start] + new_segment + markup[end:]
                    data = markup.encode("utf-8")
                # writestr(ZipInfo) keeps the entry's name, date and compression
                # (the 'mimetype' entry stays first and stored).
                zout.writestr(info, data)


def _docx_paragraphs(document) -> List:
    """Every paragraph of the body, tables (nested), headers and footers."""
    seen, out = set(), []

    def walk(container) -> None:
        for paragraph in container.paragraphs:
            if id(paragraph._p) not in seen:
                seen.add(id(paragraph._p))
                out.append(paragraph)
        for table in getattr(container, "tables", []):
            for row in table.rows:
                for cell in row.cells:
                    walk(cell)

    walk(document)
    for section in document.sections:
        for part in (section.header, section.footer):
            if part is not None and not part.is_linked_to_previous:
                walk(part)
    return out


class _DocxDocument:
    """DOCX: one unit per paragraph, edits distributed over its runs."""

    def __init__(self, path: str):
        from docx import Document
        self.path = path
        self.document = Document(path)
        self.paragraphs = _docx_paragraphs(self.document)
        self.runs = [list(p.runs) for p in self.paragraphs]
        self.units = ["".join(r.text for r in runs) for runs in self.runs]

    def locate(self, unit_index: int, _pos: int) -> str:
        return f"paragraph {unit_index + 1}"

    def write(self, output_path: str, result: TermFixResult) -> None:
        if not result.changed:
            shutil.copyfile(self.path, output_path)
            return
        for runs, unit_edits in zip(self.runs, result.edits):
            if not unit_edits:
                continue
            pieces = [r.text for r in runs]
            for run, old, new in zip(runs, pieces, distribute_edits(pieces, unit_edits)):
                if new != old:
                    run.text = new
        self.document.save(output_path)


def _load(input_filepath: str):
    file_type = _resolve_type(input_filepath)
    if file_type == "epub":
        return _EpubDocument(input_filepath)
    if file_type == "docx":
        return _DocxDocument(input_filepath)
    return _TextDocument(input_filepath)


async def scan_file(input_filepath: str, rules: Sequence[TermRule]) -> List[RuleScan]:
    """Count matches per rule without changing anything."""
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(input_filepath)
    return scan_units(_load(input_filepath).units, rules)


async def term_fix_file(
    input_filepath: str,
    output_filepath: str,
    rules: Sequence[TermRule],
    mode: str = MODE_LITERAL,
    target_language: str = "",
    model_name: Optional[str] = None,
    llm_provider: str = "ollama",
    llm_api_endpoint: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    context_window: int = 2048,
    auto_adjust_context: bool = True,
    parallel_workers: int = 1,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    report_filepath: Optional[str] = None,
    source_name: Optional[str] = None,
) -> TermFixResult:
    """Apply `rules` to `input_filepath`, write the result and a change report.

    The report goes to `report_filepath`, or `<output stem>.term-fix.md` by
    default; its title names `source_name` (the input's base name by default). Raises UnsupportedFormatError for unsupported formats, ValueError
    for an unknown mode or a rule literal mode cannot apply, and RuntimeError
    when the smart-mode LLM client cannot be created.
    """
    def log(key: str, message: str) -> None:
        if log_callback:
            log_callback(key, message)

    if mode not in MODES:
        raise ValueError(f"Unknown term-fix mode '{mode}' (expected one of {', '.join(MODES)}).")
    if mode == MODE_LITERAL:
        check_literal_rules(rules)
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(input_filepath)

    document = _load(input_filepath)
    log("term_fix_start",
        f"Term fix ({mode}): {len(rules)} rule(s) over {len(document.units)} text unit(s)")

    llm_client = None
    corrector = None
    try:
        if mode == MODE_SMART:
            from src.config import DEFAULT_MODEL, API_ENDPOINT
            from src.core.refine.client_setup import build_refine_client
            llm_client, context_manager = build_refine_client(
                model_name=model_name or DEFAULT_MODEL,
                llm_provider=llm_provider,
                cli_api_endpoint=llm_api_endpoint or API_ENDPOINT,
                auto_adjust_context=auto_adjust_context,
                context_window=context_window,
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                openrouter_api_key=openrouter_api_key,
                mistral_api_key=mistral_api_key,
                deepseek_api_key=deepseek_api_key,
                poe_api_key=poe_api_key,
                nim_api_key=nim_api_key,
                log_callback=log_callback,
            )
            if llm_client is None:
                raise RuntimeError(f"Could not create an LLM client for provider '{llm_provider}'.")
            # Local providers serve one request at a time.
            parallel = 1 if llm_provider == "ollama" else max(1, parallel_workers)
            corrector = SmartCorrector(
                llm_client=llm_client,
                model_name=model_name or DEFAULT_MODEL,
                target_language=target_language or "the same language",
                rules=rules,
                context_manager=context_manager,
                parallel=parallel,
                log_callback=log_callback,
                stats_callback=stats_callback,
                check_interruption_callback=check_interruption_callback,
            )
        result = await fix_units(document.units, rules, document.locate, corrector=corrector)
        if corrector is None and stats_callback:
            # Literal mode is a single instant step; report it as done.
            stats_callback({'total_chunks': 1, 'completed_chunks': 1, 'failed_chunks': 0})
    finally:
        if llm_client is not None and hasattr(llm_client, "close"):
            try:
                await llm_client.close()
            except Exception:
                pass

    document.write(output_filepath, result)
    changed = sum(1 for o in result.occurrences if o.after != o.before)
    log("term_fix_saved",
        f"Term fix: {len(result.occurrences)} occurrence(s) found, {changed} changed, "
        f"saved '{output_filepath}'")
    review = result.count(STATUS_NEEDS_REVIEW)
    if review:
        log("term_fix_needs_review",
            f"⚠️ Term fix: {review} occurrence(s) left unchanged (needs_review), see the report")
    skipped = result.count(STATUS_SKIPPED)
    if skipped:
        log("term_fix_interrupted",
            f"⚠️ Term fix interrupted: {skipped} occurrence(s) not processed")

    report_filepath = report_filepath or report_path_for(output_filepath)
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(render_report(source_name or os.path.basename(input_filepath),
                              rules, result.occurrences))
    log("term_fix_report", f"Term fix report: '{report_filepath}'")
    return result
