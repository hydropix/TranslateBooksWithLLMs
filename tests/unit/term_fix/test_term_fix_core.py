"""
Unit tests for the term-fix core: rule parsing, matching, snippets, literal
replacement and the scan/report helpers.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.term_fix import (
    RuleParseError,
    apply_literal_fix,
    build_snippets,
    find_spans,
    format_scan,
    match_case,
    parse_rules_json,
    parse_rules_text,
    render_report,
    report_path_for,
    scan_text,
    splice,
)


class TestParseRules:

    def test_text_form_with_alternatives_and_prefix(self):
        rules = parse_rules_text(
            "# comment\n\nDrakbrand*|Drakenbrännaren -> Eldsdraken\n"
        )
        assert len(rules) == 1
        alts = rules[0].alternatives
        assert [a.text for a in alts] == ["Drakbrand", "Drakenbrännaren"]
        assert [a.prefix for a in alts] == [True, False]
        assert rules[0].replacement == "Eldsdraken"

    @pytest.mark.parametrize("text", [
        "no separator here",
        " -> Eldsdraken",
        "Drakbrand -> ",
        "a -> b -> c",
        "Dra*k -> Eldsdraken",
        "* -> Eldsdraken",
        "",
        "# only comments",
    ])
    def test_text_form_rejects_malformed(self, text):
        with pytest.raises(RuleParseError):
            parse_rules_text(text)

    def test_error_names_the_line(self):
        with pytest.raises(RuleParseError, match="line 2"):
            parse_rules_text("a -> b\nbroken\n")

    def test_rejects_replacement_matched_by_own_rule(self):
        with pytest.raises(RuleParseError, match="itself matched"):
            parse_rules_text("Eld* -> Eldsdraken")
        with pytest.raises(RuleParseError, match="itself matched"):
            parse_rules_text("eldsdraken -> Eldsdraken")

    def test_json_forms(self):
        a = parse_rules_json({"rules": [{"find": ["Ashe", "Ashen"], "replace": "Askan"}]})
        b = parse_rules_json([{"find": "Ashe|Ashen", "replace": "Askan"}])
        assert str(a[0]) == str(b[0]) == "Ashe|Ashen -> Askan"

    @pytest.mark.parametrize("data", [
        {"rules": "nope"},
        [{"find": 3, "replace": "x"}],
        [{"find": ["a"]}],
        ["a -> b"],
        [],
    ])
    def test_json_rejects_malformed(self, data):
        with pytest.raises(RuleParseError):
            parse_rules_json(data)


class TestFindSpans:

    def test_word_boundary(self):
        rules = parse_rules_text("Fan -> Fen")
        assert [s.matched for s in find_spans("Fan likes Fantasy. fan!", rules)] == ["Fan", "fan"]

    def test_case_sensitive(self):
        rules = parse_rules_text("Fan -> Fen")
        assert len(find_spans("Fan fan", rules, case_sensitive=True)) == 1

    def test_prefix_wildcard_covers_suffixes(self):
        rules = parse_rules_text("Drakbrand* -> Eldsdraken")
        spans = find_spans("Drakbranden såg Drakbrandens eld. Odrakbrand.", rules)
        assert [s.matched for s in spans] == ["Drakbranden", "Drakbrandens"]

    def test_longest_alternative_wins_at_same_start(self):
        rules = parse_rules_text("Lord Ashe|Lord Ashen -> Lord Askan")
        spans = find_spans("Lord Ashen spoke.", rules)
        assert [(s.matched, s.alternative_index) for s in spans] == [("Lord Ashen", 1)]

    def test_overlap_across_rules_keeps_earliest(self):
        rules = parse_rules_text("Red Moon -> Crimson Moon\nMoon Gate -> Lunar Gate")
        spans = find_spans("the Red Moon Gate", rules)
        assert [(s.matched, s.rule_index) for s in spans] == [("Red Moon", 0)]

    def test_cjk_substring(self):
        assert len(find_spans("他是李凡啊", parse_rules_text("李凡 -> 凡青"))) == 1
        # Substring matching means a replacement containing the wrong form is
        # rejected, since a second run would match it again.
        with pytest.raises(RuleParseError, match="itself matched"):
            parse_rules_text("李凡 -> 李凡青")

    def test_non_word_edge(self):
        rules = parse_rules_text("Mr. Ashe -> Mr. Askan")
        assert [s.matched for s in find_spans("Hi Mr. Ashe.", rules)] == ["Mr. Ashe"]


class TestSnippets:

    def test_sentence_window(self):
        text = "First one. The Drakbrand roared! Last one."
        spans = find_spans(text, parse_rules_text("Drakbrand -> X"))
        [snip] = build_snippets(text, spans)
        assert text[snip.start:snip.end] == "The Drakbrand roared!"

    def test_line_break_is_a_boundary(self):
        text = "Title\nDrakbrand came\nNext"
        spans = find_spans(text, parse_rules_text("Drakbrand -> X"))
        [snip] = build_snippets(text, spans)
        assert text[snip.start:snip.end] == "Drakbrand came"

    def test_two_matches_in_one_sentence_merge(self):
        text = "Ashe met Ashe. Other."
        spans = find_spans(text, parse_rules_text("Ashe -> Askan"))
        snippets = build_snippets(text, spans)
        assert len(snippets) == 1 and len(snippets[0].spans) == 2

    def test_long_sentence_is_capped_on_word_boundary(self):
        text = "word " * 100 + "Ashe " + "word " * 100
        spans = find_spans(text, parse_rules_text("Ashe -> Askan"))
        [snip] = build_snippets(text, spans)
        piece = text[snip.start:snip.end]
        assert "Ashe" in piece and len(piece) <= 2 * 200 + 4
        assert piece.startswith("word") and piece.endswith("word")

    def test_splice_rejects_overlap(self):
        assert splice("abcdef", [(0, 1, "X"), (4, 6, "YZ")]) == "XbcdYZ"
        with pytest.raises(ValueError):
            splice("abcdef", [(0, 3, "X"), (2, 4, "Y")])


class TestMatchCase:

    @pytest.mark.parametrize("matched,replacement,expected", [
        ("drakbrand", "Eldsdraken", "Eldsdraken"),
        ("DRAKBRAND", "Eldsdraken", "ELDSDRAKEN"),
        ("Drakbrand", "eldsdrake", "Eldsdrake"),
        ("drakbrand", "eldsdrake", "eldsdrake"),
        ("A", "eldsdrake", "Eldsdrake"),
    ])
    def test_match_case(self, matched, replacement, expected):
        assert match_case(matched, replacement) == expected


class TestApplyLiteralFix:

    def test_only_matches_change(self):
        text = "Ashe came.\r\nTHE ASHE ROSE!\r\n  ashe   spacing kept  \r\n"
        result = apply_literal_fix(text, parse_rules_text("Ashe -> Askan"))
        assert result.text == "Askan came.\r\nTHE ASKAN ROSE!\r\n  Askan   spacing kept  \r\n"
        assert [o.line for o in result.occurrences] == [1, 2, 3]
        assert result.occurrences[0].before == "Ashe came."
        assert result.occurrences[0].after == "Askan came."

    def test_no_match_is_identity(self):
        text = "Nothing here.\n"
        result = apply_literal_fix(text, parse_rules_text("Ashe -> Askan"))
        assert result.text == text and not result.changed

    def test_refuses_prefix_rules(self):
        with pytest.raises(ValueError, match="wildcard"):
            apply_literal_fix("x", parse_rules_text("Drakbrand* -> Eldsdraken"))

    def test_idempotent(self):
        rules = parse_rules_text("Ashe|Ashen -> Askan")
        once = apply_literal_fix("Ashe and Ashen.", rules).text
        assert apply_literal_fix(once, rules).text == once


class TestScanAndReport:

    def test_scan_counts_and_samples(self):
        text = "Ashe one. Ashen two. Ashe three."
        [res] = scan_text(text, parse_rules_text("Ashe|Ashen|Asha -> Askan"), samples_per_rule=2)
        assert res.counts == [2, 1, 0] and res.total == 3
        assert res.samples == ["Ashe one.", "Ashen two."]
        lines = format_scan([res])
        assert "Asha: 0" in "\n".join(lines)

    def test_report_escapes_pipes(self):
        rules = parse_rules_text("Ashe|Ashen -> Askan")
        result = apply_literal_fix("a | Ashe", rules)
        report = render_report("book.txt", rules, result.occurrences)
        assert "Ashe\\|Ashen -> Askan | 1 |" in report
        assert "a \\| Askan" in report

    def test_report_path(self):
        assert report_path_for("out/book (term-fix).txt") == "out/book (term-fix).term-fix.md"
