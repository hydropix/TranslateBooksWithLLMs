"""
Term-fix pass: correct terms in an already-translated text without retranslating.

Provides:
- rules: parse correction rules ("wrong|other -> correct")
- matcher: locate the wrong renderings (word boundaries, prefix wildcard, CJK)
- snippets: sentence windows around matches, splicing edits back
- literal: deterministic case-preserving replacement (no LLM)
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
from src.core.term_fix.literal import (
    Occurrence,
    TermFixResult,
    apply_literal_fix,
    check_literal_rules,
    match_case,
)
from src.core.term_fix.report import (
    RuleScan,
    format_scan,
    render_report,
    report_path_for,
    scan_text,
)

__all__ = [
    "Alternative",
    "Occurrence",
    "RuleParseError",
    "RuleScan",
    "Snippet",
    "Span",
    "TermFixResult",
    "TermRule",
    "apply_literal_fix",
    "build_snippets",
    "check_literal_rules",
    "find_spans",
    "format_scan",
    "load_rules_file",
    "match_case",
    "parse_rules_json",
    "parse_rules_text",
    "render_report",
    "report_path_for",
    "scan_text",
    "splice",
]
