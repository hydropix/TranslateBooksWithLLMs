"""Runner matrix checks with a fake provider (no LLM call)."""

import asyncio
import json
from typing import Optional

import pytest

from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.llm.base import LLMProvider, LLMResponse
from tests.prompt_eval import __main__ as cli
from tests.prompt_eval.model import Case, RenderedPrompt, Target
from tests.prompt_eval.runner import parse_target, run_matrix


def make_case(case_id: str) -> Case:
    return Case(id=case_id, builder="translate", source_language="English",
                target_language="French", text=f"Text of {case_id}.", has_placeholders=False)


class FakeProvider(LLMProvider):
    """Returns the user prompt wrapped in translation tags; can fail on demand."""

    def __init__(self, model: str = "fake", fail_on: Optional[set] = None):
        super().__init__(model=model, api_keys=None, provider_name="fake")
        self.calls: list[str] = []
        self.fail_on = fail_on or set()
        self.closed = False

    async def generate(self, prompt, timeout=0, system_prompt=None):
        self.calls.append(prompt)
        if len(self.calls) in self.fail_on:
            raise ConnectionError("boom")
        return LLMResponse(content=f"{TRANSLATE_TAG_IN}{prompt}{TRANSLATE_TAG_OUT}",
                           prompt_tokens=10, completion_tokens=5)

    async def _detect_thinking_model(self):
        return False

    async def close(self):
        self.closed = True


CASES = [make_case("A"), make_case("B"), make_case("C")]
SAME = RenderedPrompt(system="sys", user="Texte de B.")
BASELINE = {"A": RenderedPrompt("sys", "Texte de A, version un."), "B": SAME,
            "C": RenderedPrompt("sys", "Texte de C, version un.")}
CANDIDATE = {"A": RenderedPrompt("sys2", "Texte de A, version un."), "B": SAME,
             "C": RenderedPrompt("sys", "Texte de C, version deux.")}


def factory_for(provider):
    def factory(provider_type, **kwargs):
        provider.factory_args = (provider_type, kwargs)
        return provider
    return factory


def run(tmp_path, *, include_unchanged=False, repeat=2, targets=None, provider=None):
    provider = provider or FakeProvider()
    targets = targets or [Target("ollama", "fake:1b")]
    records = asyncio.run(run_matrix(CASES, BASELINE, CANDIDATE, targets, repeat,
                                     include_unchanged, tmp_path, provider_factory=factory_for(provider)))
    return records, provider


def test_parse_target_splits_on_first_colon():
    assert parse_target("ollama:qwen3:8b") == Target("ollama", "qwen3:8b")
    assert parse_target("openrouter:mistralai/mistral-medium-3.1").model == "mistralai/mistral-medium-3.1"


@pytest.mark.parametrize("spec", ["bogus:model", "ollama", "ollama:"])
def test_parse_target_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_target(spec)


def test_unchanged_cases_are_skipped(tmp_path):
    records, provider = run(tmp_path)
    assert {r.case_id for r in records} == {"A", "C"}
    assert len(records) == 2 * 2 * 2          # changed cases x arms x repeat
    assert provider.closed
    assert provider.factory_args == ("ollama", {"model": "fake:1b"})


def test_include_unchanged_runs_everything(tmp_path):
    records, _ = run(tmp_path, include_unchanged=True, repeat=1)
    assert {r.case_id for r in records} == {"A", "B", "C"}


def test_record_count_and_order(tmp_path):
    targets = [Target("ollama", "m1"), Target("ollama", "m2")]
    records, _ = run(tmp_path, repeat=3, targets=targets)
    assert len(records) == 2 * 2 * 3 * 2
    assert [r.arm for r in records] == ["baseline", "candidate"] * (len(records) // 2)
    assert [(r.target, r.repeat, r.case_id) for r in records[:4]] == [
        ("ollama:m1", 0, "A"), ("ollama:m1", 0, "A"), ("ollama:m1", 0, "C"), ("ollama:m1", 0, "C")]
    assert records[-1].target == "ollama:m2" and records[-1].repeat == 2


def test_records_are_scored_and_persisted(tmp_path):
    records, _ = run(tmp_path)
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    first = json.loads(lines[0])
    assert first["case_id"] == "A" and first["arm"] == "baseline"
    assert first["metrics"]["format"] == "pass"
    assert first["prompt_tokens"] == 10


def test_provider_exception_records_raw_none_and_continues(tmp_path, capsys):
    records, _ = run(tmp_path, provider=FakeProvider(fail_on={2}))
    failed = [r for r in records if r.raw is None]
    assert len(failed) == 1 and failed[0].arm == "candidate"
    assert set(failed[0].metrics.values()) == {"n/a"}
    assert len(records) == 8
    err = capsys.readouterr().err
    assert "ConnectionError" in err and "boom" not in err


def test_cli_run_end_to_end_with_fake_provider(tmp_path, capsys):
    """Full ``run`` command against the real corpus and worktree, fake model."""

    class EchoSource(FakeProvider):
        async def generate(self, prompt, timeout=0, system_prompt=None):
            from tests.characterization.fake_llm import _extract_source_block
            self.calls.append(prompt)
            return LLMResponse(content=f"{TRANSLATE_TAG_IN}{_extract_source_block(prompt)}"
                                       f"{TRANSLATE_TAG_OUT}")

    provider = EchoSource()
    out = tmp_path / "run"
    code = cli.main(["run", "--baseline", "HEAD", "--cases", "^(T0[13]|R01)_", "--repeat", "1",
                     "--target", "ollama:fake-model", "--include-unchanged", "--out", str(out)],
                    provider_factory=factory_for(provider))
    assert code in (cli.EXIT_OK, cli.EXIT_REGRESSION)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "**Verdict:" in report and "`ollama:fake-model`" in report
    records = (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 3 * 2
    assert (out / "prompts" / "T03_retry_variant_en_fr.candidate.txt").exists()
    assert "===== USER =====" in (out / "prompts" / "R01_clean_draft_fr.baseline.txt").read_text(encoding="utf-8")
    assert capsys.readouterr().out.startswith("# Prompt regression report")
