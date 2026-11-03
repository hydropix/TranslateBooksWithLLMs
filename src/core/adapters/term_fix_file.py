"""Unified term-fix entry point.

Applies correction rules to an already-translated file: only the occurrences
of the wrong renderings change, every other byte is written back as read.
Phase 1 supports TXT with literal (no LLM) replacement; see
docs/design/TERM_FIX_PASS.md for the EPUB/SRT/DOCX and smart-mode phases.
"""

import os
from typing import Callable, List, Optional, Sequence

import aiofiles

from .exceptions import UnsupportedFormatError
from src.core.term_fix import (
    RuleScan,
    TermFixResult,
    TermRule,
    apply_literal_fix,
    render_report,
    report_path_for,
    scan_text,
)
from src.utils.file_detector import detect_file_type, detect_file_type_by_content

SUPPORTED_TYPES = ("txt",)


def _resolve_type(input_filepath: str) -> str:
    try:
        detected = detect_file_type(input_filepath)
    except ValueError:
        detected = detect_file_type_by_content(input_filepath)
    if detected not in SUPPORTED_TYPES:
        raise UnsupportedFormatError(
            f"Term fix does not support '{detected or 'unknown'}' files yet "
            f"(supported: {', '.join(SUPPORTED_TYPES)}).",
            file_type=detected,
        )
    return detected


async def _read_text(path: str) -> str:
    # newline='' keeps CRLF/LF exactly as in the file, so untouched lines are
    # written back byte-identical.
    async with aiofiles.open(path, "r", encoding="utf-8", newline="") as f:
        return await f.read()


async def _write_text(path: str, text: str) -> None:
    async with aiofiles.open(path, "w", encoding="utf-8", newline="") as f:
        await f.write(text)


async def scan_file(input_filepath: str, rules: Sequence[TermRule]) -> List[RuleScan]:
    """Count matches per rule without changing anything."""
    _resolve_type(input_filepath)
    return scan_text(await _read_text(input_filepath), rules)


async def term_fix_file(
    input_filepath: str,
    output_filepath: str,
    rules: Sequence[TermRule],
    log_callback: Optional[Callable] = None,
    report_filepath: Optional[str] = None,
) -> TermFixResult:
    """Apply `rules` to `input_filepath`, write the result and a change report.

    The report goes to `report_filepath`, or `<output stem>.term-fix.md` by
    default. Raises UnsupportedFormatError for formats not handled yet, and
    ValueError when a rule needs the (not yet available) smart mode.
    """
    def log(key: str, message: str) -> None:
        if log_callback:
            log_callback(key, message)

    if not os.path.exists(input_filepath):
        raise FileNotFoundError(input_filepath)
    _resolve_type(input_filepath)

    text = await _read_text(input_filepath)
    result = apply_literal_fix(text, rules)

    await _write_text(output_filepath, result.text)
    log("term_fix_saved",
        f"Term fix: {len(result.occurrences)} occurrence(s) changed, saved '{output_filepath}'")

    report_filepath = report_filepath or report_path_for(output_filepath)
    report = render_report(os.path.basename(input_filepath), rules, result.occurrences)
    await _write_text(report_filepath, report)
    log("term_fix_report", f"Term fix report: '{report_filepath}'")
    return result
