"""
Term-fix pass: correct terms in an already-translated text without retranslating.

Provides:
- rules: parse correction rules ("wrong|other -> correct")
- matcher: locate the wrong renderings (word boundaries, prefix wildcard, CJK)
- snippets: sentence windows around matches, splicing edits back
- literal: deterministic case-preserving replacement (no LLM)
- corrector: LLM replacement with inflection, behind a guard (smart mode)
- fixer: apply a corrector to a list of text units, build occurrences
- report: pre-run scan summary and post-run change report

See docs/design/TERM_FIX_PASS.md.
"""
from src.core.term_fix.rules import (
    Alternative,
    RuleParseError,
    TermRule,
    load_rules_file,
    parse_rules_json,
    parse_rules_text,
)
from src.core.term_fix.matcher import Span, find_spans
from src.core.term_fix.snippets import Snippet, build_snippets, splice
from src.core.term_fix.models import (
    STATUS_FIXED,
    STATUS_LITERAL,
    STATUS_NEEDS_REVIEW,
    STATUS_SKIPPED,
    Correction,
    Occurrence,
    SnippetItem,
    TermFixResult,
)
from src.core.term_fix.literal import LiteralCorrector, check_literal_rules, match_case
from src.core.term_fix.corrector import SmartCorrector, check_correction
from src.core.term_fix.fixer import apply_literal_fix, fix_units, line_locator
from src.core.term_fix.report import (
    RuleScan,
    format_scan,
    render_report,
    report_path_for,
    scan_text,
    scan_to_dict,
    scan_units,
)

MODE_LITERAL = "literal"
MODE_SMART = "smart"
MODES = (MODE_LITERAL, MODE_SMART)

__all__ = [
    "Alternative",
    "Correction",
    "LiteralCorrector",
    "MODES",
    "MODE_LITERAL",
    "MODE_SMART",
    "Occurrence",
    "RuleParseError",
    "RuleScan",
    "STATUS_FIXED",
    "STATUS_LITERAL",
    "STATUS_NEEDS_REVIEW",
    "STATUS_SKIPPED",
    "SmartCorrector",
    "Snippet",
    "SnippetItem",
    "Span",
    "TermFixResult",
    "TermRule",
    "apply_literal_fix",
    "build_snippets",
    "check_correction",
    "check_literal_rules",
    "find_spans",
    "fix_units",
    "format_scan",
    "line_locator",
    "load_rules_file",
    "match_case",
    "parse_rules_json",
    "parse_rules_text",
    "render_report",
    "report_path_for",
    "scan_text",
    "scan_to_dict",
    "scan_units",
    "splice",
]
