"""Optional judge: fills translate records only and tolerates failures."""

import asyncio

import pytest

from benchmark.evaluator import TranslationEvaluator
from benchmark.models import EvaluationScores
from tests.prompt_eval.judge import count_judge_failures, judge_records
from tests.prompt_eval.model import Case, RunRecord


def make_case(case_id, builder):
    return Case(id=case_id, builder=builder, source_language="English" if builder == "translate" else "French",
                target_language="French", text="Some text.", has_placeholders=False, notes="why")


def make_record(case_id, extracted="Du texte."):
    return RunRecord(case_id=case_id, arm="candidate", target="ollama:m", repeat=0, raw="r",
                     extracted=extracted, prompt_tokens=None, completion_tokens=None,
                     elapsed_s=0.0, metrics={})


@pytest.fixture
def patched_evaluator(monkeypatch):
    calls = []

    def fake_init(self, config, log_callback=None, provider=None):
        self.provider = provider

    async def fake_evaluate(self, source_text, translated_text, target_language, target_language_name):
        calls.append((source_text.id, source_text.source_language, target_language, target_language_name))
        if source_text.id == "T_BAD":
            raise RuntimeError("judge down")
        return EvaluationScores(accuracy=8, fluency=7, style=6, overall=7, feedback="fine"), 42

    async def fake_close(self):
        pass

    monkeypatch.setattr(TranslationEvaluator, "__init__", fake_init)
    monkeypatch.setattr(TranslationEvaluator, "evaluate", fake_evaluate)
    monkeypatch.setattr(TranslationEvaluator, "close", fake_close)
    return calls


def test_judge_fills_translate_records_only(patched_evaluator):
    cases = [make_case("T_OK", "translate"), make_case("T_BAD", "translate"), make_case("R1", "refine")]
    records = [make_record("T_OK"), make_record("T_BAD"), make_record("R1"), make_record("T_OK", extracted=None)]
    asyncio.run(judge_records(records, cases, "openrouter"))

    assert records[0].judge["overall"] == 7
    assert records[1].judge is None          # exception tolerated
    assert records[2].judge is None          # refine not judged
    assert records[3].judge is None          # nothing extracted
    assert patched_evaluator == [("T_OK", "en", "fr", "French"), ("T_BAD", "en", "fr", "French")]
    assert count_judge_failures(records, cases) == 1
