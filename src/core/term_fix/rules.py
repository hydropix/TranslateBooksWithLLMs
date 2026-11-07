"""
Correction rules for the term-fix pass.

A rule names the wrong renderings found in an already-translated text and the
correct term that should replace them. Text form, one rule per line:

    Drakbrand*|Drakenbrännaren -> Eldsdraken
    # comments and blank lines are ignored

The left side reuses the glossary's '|' alternatives convention. A trailing
'*' turns an alternative into a prefix match, so it also covers inflected or
suffixed forms ("Drakbrand*" matches "Drakbranden", "Drakbrandens").

JSON form: {"rules": [{"find": ["Drakbrand*", "Drakenbrännaren"],
"replace": "Eldsdraken"}]} or a bare list of such objects.
"""
import json
import os
from dataclasses import dataclass
from typing import Any, List, Tuple

RULE_SEPARATOR = "->"
ALTERNATIVE_SEPARATOR = "|"
PREFIX_WILDCARD = "*"


class RuleParseError(ValueError):
    """Raised when correction rules are malformed. The message names the rule."""


@dataclass(frozen=True)
class Alternative:
    """One wrong rendering to look for."""
    text: str
    prefix: bool = False

    def __str__(self) -> str:
        return f"{self.text}{PREFIX_WILDCARD}" if self.prefix else self.text


@dataclass(frozen=True)
class TermRule:
    """Wrong renderings (any of `alternatives`) -> `replacement`."""
    alternatives: Tuple[Alternative, ...]
    replacement: str

    @property
    def has_prefix_alternative(self) -> bool:
        return any(alt.prefix for alt in self.alternatives)

    def __str__(self) -> str:
        finds = ALTERNATIVE_SEPARATOR.join(str(a) for a in self.alternatives)
        return f"{finds} {RULE_SEPARATOR} {self.replacement}"


def _parse_alternative(raw: str, where: str) -> Alternative:
    text = raw.strip()
    prefix = text.endswith(PREFIX_WILDCARD)
    if prefix:
        text = text[:-1].rstrip()
    if not text:
        raise RuleParseError(f"{where}: empty alternative on the left side.")
    if PREFIX_WILDCARD in text:
        raise RuleParseError(
            f"{where}: '{PREFIX_WILDCARD}' is only allowed at the end of an alternative."
        )
    return Alternative(text=text, prefix=prefix)


def _build_rule(finds: List[str], replacement: str, where: str) -> TermRule:
    replacement = (replacement or "").strip()
    if not replacement:
        raise RuleParseError(f"{where}: the replacement (right side) is empty.")
    alternatives = tuple(_parse_alternative(f, where) for f in finds if f.strip())
    if not alternatives:
        raise RuleParseError(f"{where}: nothing to find (left side is empty).")
    rule = TermRule(alternatives=alternatives, replacement=replacement)

    # A rule whose own replacement it would match again is not idempotent: a
    # second run would rewrite already-correct occurrences, and the smart-mode
    # guard could never confirm the wrong rendering is gone.
    from src.core.term_fix.matcher import find_spans
    if find_spans(replacement, [rule]):
        raise RuleParseError(
            f"{where}: the replacement '{replacement}' is itself matched by the "
            f"rule's left side. Make the alternatives more specific."
        )
    return rule


def _check_rule_set(rules: List[TermRule]) -> List[TermRule]:
    """Reject an empty set, and a replacement that another rule would match
    (applying the set twice would then rewrite already-fixed text)."""
    if not rules:
        raise RuleParseError("No correction rule found.")
    from src.core.term_fix.matcher import find_spans
    for idx, rule in enumerate(rules):
        for span in find_spans(rule.replacement, rules):
            if span.rule_index != idx:
                raise RuleParseError(
                    f"The replacement '{rule.replacement}' is matched by rule "
                    f"'{rules[span.rule_index]}'. Make the alternatives more specific."
                )
    return rules


def parse_rules_text(text: str) -> List[TermRule]:
    """Parse the one-rule-per-line text form. Raises RuleParseError."""
    rules: List[TermRule] = []
    for lineno, line in enumerate((text or "").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        where = f"Rule line {lineno}"
        if RULE_SEPARATOR not in stripped:
            raise RuleParseError(
                f"{where}: missing '{RULE_SEPARATOR}' (expected 'wrong|other -> correct')."
            )
        left, right = stripped.split(RULE_SEPARATOR, 1)
        if RULE_SEPARATOR in right:
            raise RuleParseError(f"{where}: more than one '{RULE_SEPARATOR}'.")
        rules.append(_build_rule(left.split(ALTERNATIVE_SEPARATOR), right, where))
    return _check_rule_set(rules)


def parse_rules_json(data: Any) -> List[TermRule]:
    """Parse the JSON form (already decoded). Raises RuleParseError."""
    entries = data.get("rules") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise RuleParseError("Expected a list of rules or an object with a 'rules' list.")
    rules: List[TermRule] = []
    for idx, entry in enumerate(entries, start=1):
        where = f"Rule #{idx}"
        if not isinstance(entry, dict):
            raise RuleParseError(f"{where}: expected an object with 'find' and 'replace'.")
        finds = entry.get("find")
        if isinstance(finds, str):
            finds = finds.split(ALTERNATIVE_SEPARATOR)
        if not isinstance(finds, list) or not all(isinstance(f, str) for f in finds):
            raise RuleParseError(f"{where}: 'find' must be a string or a list of strings.")
        rules.append(_build_rule(finds, entry.get("replace") or "", where))
    return _check_rule_set(rules)


def load_rules_file(path: str) -> List[TermRule]:
    """Load rules from a .json file or a plain-text file (any other extension)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()
    if os.path.splitext(path)[1].lower() == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuleParseError(f"Invalid JSON in {path}: {e}") from e
        return parse_rules_json(data)
    return parse_rules_text(content)
