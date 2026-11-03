"""
Tests for the smart-mode pieces: token diff, guard, response parsing, the
SmartCorrector batching/retry logic, and the format helpers.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.term_fix import (
    STATUS_FIXED,
    STATUS_NEEDS_REVIEW,
    STATUS_SKIPPED,
    SmartCorrector,
    SnippetItem,
    check_correction,
    parse_rules_text,
    splice,
)
from src.core.term_fix.corrector import mark_snippet, parse_response
from src.core.term_fix.diffing import changes_outside, minimal_edits
from src.core.term_fix.formats import distribute_edits, xml_text_segments
from src.prompts.prompts import generate_term_fix_prompt

RULES = parse_rules_text("Drakbrand* -> Eldsdraken")


def _item(text: str, word: str) -> SnippetItem:
    start = text.index(word)
    return SnippetItem(text=text, spans=((start, start + len(word), 0),))


class TestDiffing:

    @pytest.mark.parametrize("a,b", [
        ("Le Drakbrand rugit.", "La Eldsdraken rugit."),
        ("abc def", "abc def"),
        ("one two", "one  two three"),
        ("", "x"),
    ])
    def test_minimal_edits_round_trip(self, a, b):
        assert splice(a, minimal_edits(a, b)) == b

    def test_minimal_edits_touch_only_changed_tokens(self):
        edits = minimal_edits("The old Drakbrand roared loudly.", "The old Eldsdraken roared loudly.")
        assert edits == [(8, 17, "Eldsdraken")]

    def test_changes_near_match_allowed(self):
        text = "Il vit le grand Drakbrand noir au loin dans la vallée."
        fixed = "Il vit la grande Eldsdraken noire au loin dans la vallée."
        start = text.index("Drakbrand")
        assert not changes_outside(text, fixed, [(start, start + 9)], 4)

    def test_changes_far_from_match_rejected(self):
        text = "Drakbrand came. Then many other words follow here and finally the end."
        fixed = "Eldsdraken came. Then many other words follow here and finally a conclusion."
        assert changes_outside(text, fixed, [(0, 9)], 4)

    def test_whitespace_only_change_ignored(self):
        text = "Drakbrand came. Far away  there were   spaces."
        fixed = "Eldsdraken came. Far away there were spaces."
        assert not changes_outside(text, fixed, [(0, 9)], 2)


class TestGuard:

    def test_accepts_inflected_answer(self):
        item = _item("Drakbrandens eld brann.", "Drakbrandens")
        assert check_correction(item, "Eldsdrakens eld brann.", RULES) is None

    @pytest.mark.parametrize("answer,reason", [
        ("Drakbrandens eld brann.", "still present"),
        ("⟦Eldsdrakens⟧ eld brann.", "markers"),
        ("Hans eld brann.", "missing"),
        ("Eldsdrakens eld brann. [[3]]", "placeholders"),
    ])
    def test_rejections(self, answer, reason):
        item = _item("Drakbrandens eld brann.", "Drakbrandens")
        assert reason in check_correction(item, answer, RULES)

    def test_rewrite_away_from_term_rejected(self):
        text = "Drakbranden vaknade och sedan gick han långt bort över de höga bergen i norr."
        item = _item(text, "Drakbranden")
        answer = "Eldsdraken vaknade och sedan gick han långt bort över de mäktiga topparna i norr."
        assert "away from the term" in check_correction(item, answer, RULES)


class TestPromptAndParsing:

    def test_mark_snippet(self):
        assert mark_snippet(_item("Se Drakbranden nu", "Drakbranden")) == "Se ⟦Drakbranden⟧ nu"

    def test_parse_response_ignores_noise(self):
        content = 'Sure!\n<snippet id="2">two</snippet>\n<snippet id="1">one\nline</snippet>\n<snippet id="1">dup</snippet>'
        assert parse_response(content) == {1: "one\nline", 2: "two"}

    def test_parse_response_tolerates_missing_closing_tags(self):
        # Seen with gemma3 on Ollama: the last item is never closed, and an
        # inner item can lose its closing tag too.
        content = '<snippet id="1">Eldsdraken vaknade.\n<snippet id="2">Alla fruktade Eldsdrakens eld.\n'
        assert parse_response(content) == {
            1: "Eldsdraken vaknade.", 2: "Alla fruktade Eldsdrakens eld."}

    def test_parse_response_strips_leftover_markers(self):
        content = '<snippet id="1">Alla fruktade ⟦Eldsdrakens⟧ eld.</snippet>'
        assert parse_response(content) == {1: "Alla fruktade Eldsdrakens eld."}

    def test_prompt_lists_rules_and_ids(self):
        prompt = generate_term_fix_prompt(["a ⟦Drakbranden⟧", "b"], RULES, "Swedish")
        assert "Drakbrand*  ->  correct: Eldsdraken" in prompt.system
        assert "Swedish" in prompt.system
        assert '<snippet id="1">a ⟦Drakbranden⟧</snippet>\n<snippet id="2">b</snippet>' in prompt.user


class _Response:
    def __init__(self, content):
        self.content = content
        self.was_truncated = False
        self.prompt_tokens = 0


class ScriptedLLM:
    """Returns the next scripted answer builder for each request."""

    def __init__(self, *builders):
        self.builders = list(builders)
        self.prompts = []

    async def make_request(self, prompt, model=None, system_prompt=None):
        self.prompts.append(prompt)
        items = re.findall(r'<snippet id="(\d+)">(.*?)</snippet>', prompt, re.DOTALL)
        builder = self.builders.pop(0) if self.builders else (lambda i, t: None)
        out = []
        for ident, text in items:
            answer = builder(int(ident), text)
            if answer is not None:
                out.append(f'<snippet id="{ident}">{answer}</snippet>')
        return _Response("\n".join(out))


def _good(_i, text):
    return re.sub(r"⟦.*?⟧", "Eldsdraken", text)


class TestSmartCorrector:

    @pytest.mark.asyncio
    async def test_batches_and_progress(self):
        llm = ScriptedLLM(_good, _good)
        stats = []
        corrector = SmartCorrector(llm, "m", "Swedish", RULES, batch_size=2,
                                   stats_callback=stats.append)
        items = [_item(f"Drakbrand {n}.", "Drakbrand") for n in range(3)]
        results = await corrector.correct(items)
        assert [r.status for r in results] == [STATUS_FIXED] * 3
        assert len(llm.prompts) == 2
        assert stats[-1] == {'total_chunks': 2, 'completed_chunks': 2, 'failed_chunks': 0}

    @pytest.mark.asyncio
    async def test_rejected_item_retried_alone_then_flagged(self):
        # Batch: item 2 keeps the wrong form. Retry of item 2: still wrong.
        bad_second = lambda i, t: t.replace("⟦", "").replace("⟧", "") if i == 2 else _good(i, t)
        llm = ScriptedLLM(bad_second, lambda i, t: t.replace("⟦", "").replace("⟧", ""))
        corrector = SmartCorrector(llm, "m", "Swedish", RULES)
        items = [_item("Drakbrand ett.", "Drakbrand"), _item("Drakbrand två.", "Drakbrand")]
        results = await corrector.correct(items)
        assert results[0].status == STATUS_FIXED
        assert results[1].status == STATUS_NEEDS_REVIEW and results[1].text is None
        assert "still present" in results[1].reason
        assert len(llm.prompts) == 2 and "två" in llm.prompts[1] and "ett" not in llm.prompts[1]

    @pytest.mark.asyncio
    async def test_retry_can_succeed(self):
        llm = ScriptedLLM(lambda i, t: None, _good)  # first answer empty
        corrector = SmartCorrector(llm, "m", "Swedish", RULES)
        [result] = await corrector.correct([_item("Drakbrand ett.", "Drakbrand")])
        assert result.status == STATUS_FIXED and result.text == "Eldsdraken ett."

    @pytest.mark.asyncio
    async def test_interruption_skips_remaining_batches(self):
        calls = {"n": 0}

        def interrupted():
            calls["n"] += 1
            return calls["n"] > 1

        corrector = SmartCorrector(ScriptedLLM(_good), "m", "Swedish", RULES, batch_size=1,
                                   check_interruption_callback=interrupted)
        items = [_item("Drakbrand ett.", "Drakbrand"), _item("Drakbrand två.", "Drakbrand")]
        results = await corrector.correct(items)
        assert [r.status for r in results] == [STATUS_FIXED, STATUS_SKIPPED]


class TestFormats:

    def test_xml_segments_body_only_skip_script(self):
        markup = ("<html><head><title>T</title></head><body>A<script>x</script>"
                  "<p>B</p><!-- c --></body></html>")
        assert [markup[s:e] for s, e in xml_text_segments(markup, body_only=True)] == ["A", "B"]
        assert [markup[s:e] for s, e in xml_text_segments(markup, body_only=False)] == ["T", "A", "B"]

    @pytest.mark.parametrize("pieces,edits,expected", [
        (["ab", "cd"], [(1, 3, "X")], ["aX", "d"]),
        (["ab", "", "cd"], [(0, 4, "Z")], ["Z", "", ""]),
        (["ab", "cd"], [(4, 4, "!")], ["ab", "cd!"]),
        (["ab", "cd", "ef"], [(0, 1, "A"), (5, 6, "F")], ["Ab", "cd", "eF"]),
    ])
    def test_distribute_edits(self, pieces, edits, expected):
        assert distribute_edits(pieces, edits) == expected
