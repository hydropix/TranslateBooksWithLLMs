"""
Unit tests for `src.core.auto_prep` (Phase 1 of plan/PLAN_AutoGlossaryStyle.md).

Fully offline: the LLM is a fake async client that records its calls and
returns canned payloads. Each numbered class below matches the matching item
in the "Validation criteria (Phase 1)" list of the plan.
"""
import asyncio
import json

import pytest

from src.core import auto_prep


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, content):
        self.content = content


def _ner_payload(entries):
    return f"<NER_JSON>{json.dumps(entries)}</NER_JSON>"


def _style_payload(rules, context=""):
    payload = json.dumps({
        "summary": "A dry, clipped voice.",
        "suggested_name": "dry_voice",
        "context": context,
        "rules": rules,
    })
    return f"<STYLE_JSON>{payload}</STYLE_JSON>"


class FakeLLMClient:
    """Duck-typed stand-in for `LLMClient` — only `generate` is ever used.

    `ner` / `style` may each be a string (the response content) or an
    exception instance (raised when that pass calls in). Routing is done on
    the prompt itself, so a single client can serve both concurrent passes.
    """

    def __init__(self, ner="", style="", delay=0.0):
        self.ner = ner
        self.style = style
        self.delay = delay
        self.calls = 0
        self.prompts = []

    async def generate(self, prompt, system_prompt=None, timeout=None):
        self.calls += 1
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.style if "STYLE_JSON" in prompt else self.ner
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeResponse(outcome)


SOURCE_TEXT = (
    "The rain kept falling on the empty avenue, and nobody came looking for answers. "
) * 40

UNFLAGGED_RULE = {
    "dimension": "sentence_rhythm",
    "instruction": "Keep sentences short and rhythmic, favouring concrete verbs over abstractions",
    "evidence": "The rain kept falling.",
}
UNFLAGGED_RULE_2 = {
    "dimension": "register",
    "instruction": "Preserve a dry and understated ironic register in the dialogue",
    "evidence": "Nobody came looking.",
}
# The linter flags this one (quoted_example + example_marker) — see
# src/core/style/lint.py; D8 says it must never reach the assembled block.
FLAGGED_RULE = {
    "dimension": "lexicon",
    "instruction": 'Use words like "shadow" and "dust" to set the mood',
    "evidence": "shadow, dust",
}


# ---------------------------------------------------------------------------
# 1. normalize_auto_flags
# ---------------------------------------------------------------------------
class TestNormalizeAutoFlags:
    def test_glossary_auto_flag(self):
        options = {"glossary_auto": True}
        assert auto_prep.normalize_auto_flags(options) == (True, False)
        assert options["glossary_auto"] is True

    def test_style_auto_flag(self):
        options = {"style_auto": True}
        assert auto_prep.normalize_auto_flags(options) == (False, True)
        assert options["style_auto"] is True

    def test_glossary_sentinel_is_deleted(self):
        options = {"glossary_id": auto_prep.AUTO_SENTINEL}
        assert auto_prep.normalize_auto_flags(options) == (True, False)
        assert "glossary_id" not in options

    def test_instruction_sentinel_becomes_empty_string(self):
        options = {"custom_instruction_file": auto_prep.AUTO_SENTINEL}
        assert auto_prep.normalize_auto_flags(options) == (False, True)
        assert options["custom_instruction_file"] == ""

    def test_both_sentinels(self):
        options = {
            "glossary_id": auto_prep.AUTO_SENTINEL,
            "custom_instruction_file": auto_prep.AUTO_SENTINEL,
        }
        assert auto_prep.normalize_auto_flags(options) == (True, True)
        assert "glossary_id" not in options
        assert options["custom_instruction_file"] == ""

    def test_real_values_are_untouched(self):
        options = {"glossary_id": 7, "custom_instruction_file": "noir.yaml"}
        assert auto_prep.normalize_auto_flags(options) == (False, False)
        assert options == {"glossary_id": 7, "custom_instruction_file": "noir.yaml"}

    def test_empty_and_none(self):
        assert auto_prep.normalize_auto_flags({}) == (False, False)
        assert auto_prep.normalize_auto_flags(None) == (False, False)


# ---------------------------------------------------------------------------
# 2. candidates_to_glossary
# ---------------------------------------------------------------------------
class TestCandidatesToGlossary:
    def test_empty_list(self):
        assert auto_prep.candidates_to_glossary([]) == ({}, {})

    def test_strips_and_keeps_metadata(self):
        terms, metadata = auto_prep.candidates_to_glossary([
            {"source": "  Li Wei  ", "target": " Li Wei le Grand ", "category": "character",
             "gender": "male"},
        ])
        assert terms == {"Li Wei": "Li Wei le Grand"}
        assert metadata == {"Li Wei": {"category": "character", "gender": "male"}}

    def test_drops_empty_target_and_empty_source(self):
        terms, metadata = auto_prep.candidates_to_glossary([
            {"source": "Ghost", "target": "", "category": "character", "gender": ""},
            {"source": "   ", "target": "Rien", "category": "other", "gender": ""},
            {"source": "Sword", "target": "Épée", "category": "item", "gender": ""},
        ])
        assert terms == {"Sword": "Épée"}
        assert metadata == {"Sword": {"category": "item"}}

    def test_drops_case_insensitive_identity_mapping(self):
        terms, _ = auto_prep.candidates_to_glossary([
            {"source": "Paris", "target": "paris", "category": "location", "gender": ""},
            {"source": "Sword", "target": "Épée", "category": "item", "gender": ""},
        ])
        assert terms == {"Sword": "Épée"}

    def test_first_source_wins(self):
        terms, metadata = auto_prep.candidates_to_glossary([
            {"source": "Mei", "target": "Mei-Ling", "category": "character", "gender": "female"},
            {"source": "Mei", "target": "Meï", "category": "location", "gender": "male"},
        ])
        assert terms == {"Mei": "Mei-Ling"}
        assert metadata == {"Mei": {"category": "character", "gender": "female"}}

    def test_metadata_omitted_when_it_would_be_empty(self):
        terms, metadata = auto_prep.candidates_to_glossary([
            {"source": "Rain", "target": "Pluie", "category": "", "gender": ""},
            {"source": "Snow", "target": "Neige", "category": "", "gender": "unknown"},
            {"source": "Wind", "target": "Vent", "category": "", "gender": "   "},
        ])
        assert terms == {"Rain": "Pluie", "Snow": "Neige", "Wind": "Vent"}
        assert metadata == {}

    def test_preserves_order_and_truncates_after_filtering(self):
        candidates = [{"source": "S0", "target": "", "category": "", "gender": ""}]
        candidates += [
            {"source": f"S{i}", "target": f"T{i}", "category": "other", "gender": ""}
            for i in range(1, 6)
        ]
        terms, metadata = auto_prep.candidates_to_glossary(candidates, max_terms=3)
        assert list(terms.items()) == [("S1", "T1"), ("S2", "T2"), ("S3", "T3")]
        assert set(metadata) == {"S1", "S2", "S3"}

    def test_default_cap(self):
        candidates = [
            {"source": f"S{i}", "target": f"T{i}", "category": "other", "gender": ""}
            for i in range(auto_prep.GLOSSARY_MAX_TERMS + 25)
        ]
        terms, _ = auto_prep.candidates_to_glossary(candidates)
        assert len(terms) == auto_prep.GLOSSARY_MAX_TERMS

    def test_non_dict_entries_are_ignored(self):
        terms, _ = auto_prep.candidates_to_glossary(
            ["nope", None, {"source": "Sword", "target": "Épée"}]
        )
        assert terms == {"Sword": "Épée"}


# ---------------------------------------------------------------------------
# 3. extract_source_text
# ---------------------------------------------------------------------------
class TestExtractSourceText:
    def test_inline_text_wins_over_file_path(self, tmp_path):
        path = tmp_path / "book.txt"
        path.write_text("from the file", encoding="utf-8")
        assert auto_prep.extract_source_text(
            file_path=str(path), text="inline wins"
        ) == "inline wins"

    def test_txt_round_trip(self, tmp_path):
        path = tmp_path / "book.txt"
        # Bytes, not write_text: Windows would translate "\n" to "\r\n".
        path.write_bytes("Chapter One\nThe rain fell.".encode("utf-8"))
        assert auto_prep.extract_source_text(file_path=str(path)) == "Chapter One\nThe rain fell."

    def test_missing_path_returns_empty(self, tmp_path):
        assert auto_prep.extract_source_text(file_path=str(tmp_path / "nope.txt")) == ""

    def test_unsupported_extension_returns_empty(self, tmp_path):
        path = tmp_path / "book.odt"
        path.write_bytes(b"PK fake")
        assert auto_prep.extract_source_text(file_path=str(path)) == ""

    def test_no_input_returns_empty(self):
        assert auto_prep.extract_source_text() == ""
        assert auto_prep.extract_source_text(text="   ") == ""

    def test_inline_text_is_capped(self):
        assert auto_prep.extract_source_text(text="abcdef", hard_cap=3) == "abc"


# ---------------------------------------------------------------------------
# 4. auto_glossary
# ---------------------------------------------------------------------------
class TestAutoGlossary:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        client = FakeLLMClient(ner=_ner_payload([
            {"source": "Li Wei", "target": "Li Wei", "category": "character", "gender": "male"},
            {"source": "Cloud Sect", "target": "Secte des Nuages", "category": "organization",
             "gender": "unknown"},
            {"source": "Mei", "target": "Mei", "category": "character", "gender": "female"},
        ]))
        terms, metadata, excerpts, warnings = await auto_prep.auto_glossary(
            SOURCE_TEXT, "English", "French", client
        )
        # "Li Wei"/"Mei" are identity mappings and are dropped.
        assert terms == {"Cloud Sect": "Secte des Nuages"}
        assert metadata == {"Cloud Sect": {"category": "organization"}}
        assert excerpts >= 1
        assert client.calls == 1

    @pytest.mark.asyncio
    async def test_raising_client_is_caught(self):
        client = FakeLLMClient(ner=RuntimeError("provider exploded"))
        terms, metadata, excerpts, warnings = await auto_prep.auto_glossary(
            SOURCE_TEXT, "English", "French", client
        )
        assert (terms, metadata) == ({}, {})
        assert excerpts >= 1
        assert any("auto glossary failed" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_prose_response_yields_warnings(self):
        client = FakeLLMClient(ner="Sure! I could not find any named entities, sorry.")
        terms, metadata, excerpts, warnings = await auto_prep.auto_glossary(
            SOURCE_TEXT, "English", "French", client
        )
        assert (terms, metadata) == ({}, {})
        assert warnings

    @pytest.mark.asyncio
    async def test_secondary_name_in_long_text_reaches_the_prompt(self):
        # Evenly spaced excerpts would miss a name confined to one stretch.
        filler = "The road went on through the quiet hills under a grey sky. "
        middle = "Then the guide spoke with Ignatius by the fire. " * 4
        text = filler * 1200 + middle + filler * 3000
        client = FakeLLMClient(ner=_ner_payload([]))
        await auto_prep.auto_glossary(text, "English", "French", client)
        assert client.calls == 1
        assert "Ignatius" in client.prompts[0]

    @pytest.mark.asyncio
    async def test_blank_text_makes_no_call(self):
        client = FakeLLMClient(ner=_ner_payload([]))
        result = await auto_prep.auto_glossary("   ", "English", "French", client)
        assert result == ({}, {}, 0, [])
        assert client.calls == 0


# ---------------------------------------------------------------------------
# 5. auto_style
# ---------------------------------------------------------------------------
class TestAutoStyle:
    @pytest.mark.asyncio
    async def test_flagged_rule_is_dropped(self):
        client = FakeLLMClient(style=_style_payload(
            [UNFLAGGED_RULE, FLAGGED_RULE, UNFLAGGED_RULE_2], context="A rainy port city."
        ))
        translation, refinement, kept, warnings = await auto_prep.auto_style(
            SOURCE_TEXT, "English", "French", client
        )
        assert kept == 2
        assert translation and refinement
        for block in (translation, refinement):
            assert UNFLAGGED_RULE["instruction"] in block
            assert UNFLAGGED_RULE_2["instruction"] in block
            assert "shadow" not in block
            assert FLAGGED_RULE["instruction"] not in block
        # The narrative setting is carried through to both blocks.
        assert "A rainy port city." in translation

    @pytest.mark.asyncio
    async def test_every_rule_flagged_assembles_nothing(self):
        client = FakeLLMClient(style=_style_payload([FLAGGED_RULE]))
        translation, refinement, kept, warnings = await auto_prep.auto_style(
            SOURCE_TEXT, "English", "French", client
        )
        assert (translation, refinement, kept) == (None, None, 0)

    @pytest.mark.asyncio
    async def test_zero_rules(self):
        client = FakeLLMClient(style=_style_payload([]))
        translation, refinement, kept, warnings = await auto_prep.auto_style(
            SOURCE_TEXT, "English", "French", client
        )
        assert (translation, refinement, kept) == (None, None, 0)
        assert warnings

    @pytest.mark.asyncio
    async def test_raising_client_is_caught(self):
        client = FakeLLMClient(style=RuntimeError("provider exploded"))
        translation, refinement, kept, warnings = await auto_prep.auto_style(
            SOURCE_TEXT, "English", "French", client
        )
        assert (translation, refinement, kept) == (None, None, 0)
        assert any("auto style failed" in w for w in warnings)


# ---------------------------------------------------------------------------
# 6 & 7. build_auto_prompt_options
# ---------------------------------------------------------------------------
class _Recorder:
    def __init__(self):
        self.entries = []

    def __call__(self, key, message):
        self.entries.append((key, message))

    def keys(self):
        return [key for key, _ in self.entries]


def _full_client():
    return FakeLLMClient(
        ner=_ner_payload([
            {"source": "Cloud Sect", "target": "Secte des Nuages", "category": "organization",
             "gender": ""},
        ]),
        style=_style_payload([UNFLAGGED_RULE]),
    )


class TestBuildAutoPromptOptions:
    @pytest.mark.asyncio
    async def test_both_wants_false_makes_no_call(self):
        client = _full_client()
        log = _Recorder()
        fragment = await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT,
            source_language="English",
            target_language="French",
            want_glossary=False,
            want_style=False,
            llm_client=client,
            log=log,
        )
        assert fragment == {}
        assert client.calls == 0
        assert log.entries == []

    @pytest.mark.asyncio
    async def test_blank_text_and_none_client_make_no_call(self):
        client = _full_client()
        assert await auto_prep.build_auto_prompt_options(
            source_text="   ", source_language="English", target_language="French",
            want_glossary=True, want_style=True, llm_client=client,
        ) == {}
        assert client.calls == 0
        assert await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT, source_language="English", target_language="French",
            want_glossary=True, want_style=True, llm_client=None,
        ) == {}

    @pytest.mark.asyncio
    async def test_glossary_only_fragment(self):
        client = _full_client()
        log = _Recorder()
        fragment = await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT,
            source_language="English",
            target_language="French",
            want_glossary=True,
            want_style=False,
            llm_client=client,
            log=log,
        )
        assert fragment["glossary_terms"] == {"Cloud Sect": "Secte des Nuages"}
        assert fragment["glossary_name"] == auto_prep.AUTO_GLOSSARY_NAME
        assert fragment["glossary_source"] == "auto"
        assert fragment["glossary_term_metadata"] == {
            "Cloud Sect": {"category": "organization"}
        }
        assert "custom_instructions" not in fragment
        assert "refinement_instructions" not in fragment
        assert client.calls == 1
        # The start line is emitted before the passes so the job never looks
        # hung while they run; then exactly one line per enabled pass.
        assert log.keys() == ["auto_prep_start", "auto_glossary"]
        assert "a glossary" in dict(log.entries)["auto_prep_start"]
        assert "style instructions" not in dict(log.entries)["auto_prep_start"]

    @pytest.mark.asyncio
    async def test_style_only_fragment(self):
        client = _full_client()
        log = _Recorder()
        fragment = await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT,
            source_language="English",
            target_language="French",
            want_glossary=False,
            want_style=True,
            llm_client=client,
            log=log,
        )
        assert set(fragment) == {"custom_instructions", "refinement_instructions"}
        assert UNFLAGGED_RULE["instruction"] in fragment["custom_instructions"]
        assert client.calls == 1
        assert log.keys() == ["auto_prep_start", "auto_style"]
        assert "style instructions" in dict(log.entries)["auto_prep_start"]

    @pytest.mark.asyncio
    async def test_both_passes_run_and_log_once_each(self):
        client = _full_client()
        log = _Recorder()
        fragment = await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT,
            source_language="English",
            target_language="French",
            want_glossary=True,
            want_style=True,
            llm_client=client,
            log=log,
        )
        assert set(fragment) == {
            "glossary_terms", "glossary_term_metadata", "glossary_name",
            "glossary_source", "custom_instructions", "refinement_instructions",
        }
        assert client.calls == 2
        # Start line first, then one line per enabled pass.
        assert log.keys()[0] == "auto_prep_start"
        assert sorted(log.keys()[1:]) == ["auto_glossary", "auto_style"]
        messages = dict(log.entries)
        assert "a glossary and style instructions" in messages["auto_prep_start"]
        assert messages["auto_glossary"].startswith("🧠 Auto glossary: 1 terms extracted from")
        assert messages["auto_style"].startswith("🧠 Auto style: 1 rules extracted from")

    @pytest.mark.asyncio
    async def test_empty_results_log_the_warning_variant(self):
        client = FakeLLMClient(ner=_ner_payload([]), style=_style_payload([]))
        log = _Recorder()
        fragment = await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT,
            source_language="English",
            target_language="French",
            want_glossary=True,
            want_style=True,
            llm_client=client,
            log=log,
        )
        assert fragment == {}
        messages = dict(log.entries)
        assert messages["auto_glossary"] == (
            "⚠️ Auto glossary: no usable terms found — translating without a glossary."
        )
        assert messages["auto_style"] == (
            "⚠️ Auto style: no usable style rules found — translating without style instructions."
        )

    @pytest.mark.asyncio
    async def test_one_failing_pass_still_returns_the_other(self):
        client = FakeLLMClient(
            ner=RuntimeError("provider exploded"),
            style=_style_payload([UNFLAGGED_RULE]),
        )
        log = _Recorder()
        fragment = await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT,
            source_language="English",
            target_language="French",
            want_glossary=True,
            want_style=True,
            llm_client=client,
            log=log,
        )
        assert "glossary_terms" not in fragment
        assert UNFLAGGED_RULE["instruction"] in fragment["custom_instructions"]
        assert log.keys()[0] == "auto_prep_start"
        assert sorted(log.keys()[1:]) == ["auto_glossary", "auto_style"]

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_without_raising(self, monkeypatch):
        monkeypatch.setattr(auto_prep, "AUTO_PREP_TIMEOUT_S", 0.05)
        client = FakeLLMClient(
            ner=_ner_payload([{"source": "A", "target": "B"}]),
            style=_style_payload([UNFLAGGED_RULE]),
            delay=2.0,
        )
        fragment = await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT,
            source_language="English",
            target_language="French",
            want_glossary=True,
            want_style=True,
            llm_client=client,
        )
        assert fragment == {}

    @pytest.mark.asyncio
    async def test_raising_log_callback_is_swallowed(self):
        def boom(key, message):
            raise ValueError("log is broken")

        fragment = await auto_prep.build_auto_prompt_options(
            source_text=SOURCE_TEXT,
            source_language="English",
            target_language="French",
            want_glossary=True,
            want_style=False,
            llm_client=_full_client(),
            log=boom,
        )
        assert fragment["glossary_terms"] == {"Cloud Sect": "Secte des Nuages"}

    @pytest.mark.asyncio
    async def test_inputs_are_not_mutated(self):
        client = _full_client()
        text = SOURCE_TEXT
        await auto_prep.build_auto_prompt_options(
            source_text=text,
            source_language="English",
            target_language="French",
            want_glossary=True,
            want_style=True,
            llm_client=client,
        )
        assert text == SOURCE_TEXT
