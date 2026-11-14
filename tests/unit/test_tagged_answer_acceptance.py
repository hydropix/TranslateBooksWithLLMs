"""TXT / SRT / PGN reject a well-tagged answer that is not a translation.

Through ``GenericTranslator`` a properly tagged answer used to be saved even
when it was cut off by the output limit, a refusal, or the source copied
back. The shared request core now applies the acceptance rules of the other
paths (``request_core.tagged_answer_problem``): the answer is retried with a
reinforced prompt and, if every attempt is rejected, the unit keeps its source
text, is counted as failed, logged, and retranslated on resume.
"""

import asyncio
import re

import pytest

from src.config import INPUT_TAG_IN, INPUT_TAG_OUT, TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.job import JobConfig, JobStatus
from src.core.job.runner import JobRunner, JobSinks
from src.core.llm.base import LLMProvider, LLMResponse
from src.core.llm.provider_config import ProviderConfig
from src.core.request_core import (
    languages_differ,
    source_may_stay_unchanged,
    tagged_answer_problem,
)
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fixtures

LONG_SOURCE = ("The quick brown fox jumps over the lazy dog while the patient translator "
               "carefully preserves the meaning of every clause in the chapter.")


# -- the rules ------------------------------------------------------------


def test_a_real_translation_is_accepted():
    answer = ("Le renard brun rapide saute par-dessus le chien paresseux pendant que le "
              "traducteur patient préserve le sens de chaque proposition du chapitre.")
    assert tagged_answer_problem(answer, LONG_SOURCE, reject_echo=True) is None


def test_a_tagged_refusal_is_rejected():
    reason = tagged_answer_problem("I'm sorry, I can't help with that request.", LONG_SOURCE,
                                   reject_echo=True)
    assert reason and "refusal" in reason


def test_a_short_dialogue_line_that_reads_like_a_refusal_is_accepted():
    # FR -> EN: the translation of a line of the same length is not a refusal.
    assert tagged_answer_problem("I'm sorry, I can't help you.", "Désolé, je ne peux pas t'aider.",
                                 reject_echo=True) is None


def test_a_cut_off_answer_is_rejected():
    assert tagged_answer_problem("Le renard", LONG_SOURCE, reject_echo=True)


def test_the_source_copied_back_is_rejected_when_languages_differ():
    echoed = "  " + LONG_SOURCE.upper().replace(" ", "  ") + "\n"
    assert tagged_answer_problem(echoed, LONG_SOURCE, reject_echo=True) == \
        "the answer is the source text copied back"
    assert tagged_answer_problem(echoed, LONG_SOURCE, reject_echo=False) is None


@pytest.mark.parametrize("source", [
    "1984",
    "...!?",
    "[0]♪ ♪\n[1]♫",
    "[0][MUSIC]\n[1](laughs)",
    "[0]OK",
    "[0]Sherlock Holmes",
    "[0]12. Nf3 Nc6 13. O-O-O Qxd4+",
    "e4 e5 Nf3",
])
def test_legitimately_unchanged_sources_are_exempt_from_the_echo_rule(source):
    assert source_may_stay_unchanged(source)
    assert tagged_answer_problem(source, source, reject_echo=True) is None


@pytest.mark.parametrize("source", [
    "[0]Where are you going tonight?",
    "[0]Hello there\n[1][MUSIC]",
    LONG_SOURCE,
])
def test_ordinary_text_is_not_exempt(source):
    assert not source_may_stay_unchanged(source)


@pytest.mark.parametrize("src,tgt,expected", [
    ("English", "French", True), ("French", "french", False),
    ("", "French", False), ("auto", "French", False),
])
def test_languages_differ(src, tgt, expected):
    assert languages_differ(src, tgt) is expected


# -- end to end through the job runner -------------------------------------


def _source(prompt):
    start = prompt.rfind(INPUT_TAG_IN) + len(INPUT_TAG_IN)
    return prompt[start:prompt.find(INPUT_TAG_OUT, start)].strip("\n")


def _fr(text):
    return re.sub(r"(?<![\[\w])[A-Za-z]{3,}(?![\]\w])", lambda m: "FR_" + m.group(0), text)


def _wrap(text):
    return f"{TRANSLATE_TAG_IN}{text}{TRANSLATE_TAG_OUT}"


BAD = {
    'truncated': lambda s: (_wrap(_fr(s)[: len(s) // 2]), True),
    'cut_short': lambda s: (_wrap(_fr(s)[: len(s) // 10]), False),
    'tagged_refusal': lambda s: (_wrap("I'm sorry, I can't help with that."), False),
    'source_echo': lambda s: (_wrap(s), False),
}


class _Sinks(JobSinks):
    def __init__(self):
        self.logs = []

    def log(self, key, message="", data=None):
        self.logs.append(key)

    def notify(self, event, context):
        pass


def _install(monkeypatch, bad, *, bad_attempts):
    """Unit 2 (the third distinct source) gets ``bad`` for its first
    ``bad_attempts`` requests; every other request is translated."""
    seen, calls = {}, {}

    class Scripted(LLMProvider):
        def __init__(self, **_kw):
            super().__init__(model="fake", api_keys=None, provider_name="fake")
            self.context_window = 8192

        async def _send(self, request):
            src = _source(request.prompt)
            seen.setdefault(src, len(seen))
            calls[src] = calls.get(src, 0) + 1
            truncated = False
            if seen[src] == 2 and calls[src] <= bad_attempts:
                content, truncated = bad(src)
            else:
                content = _wrap(_fr(src))
            response = LLMResponse(content=content, prompt_tokens=1, completion_tokens=1,
                                   context_used=2, context_limit=8192, was_truncated=False,
                                   was_fallback=False)
            response.output_truncated = truncated
            response.finish_reason = "length" if truncated else "stop"
            return response

    def factory(*_a, **_kw):
        return Scripted()
    for target in ("src.core.llm.factory.create_llm_provider", "src.core.llm.create_llm_provider"):
        monkeypatch.setattr(target, factory)
    return seen


def _job(tmp_path, source, fmt='txt'):
    return JobConfig(file_type=fmt, input_path=str(source), output_path=str(tmp_path / f"out.{fmt}"),
                     source_language="English", target_language="French",
                     provider=ProviderConfig(provider="poe", model="fake", api_key="sk-xxxxxxxx"),
                     context_window=8192, auto_adjust_context=False, max_tokens_per_chunk=60,
                     max_attempts=1, retry_delay=0, origin='cli')


@pytest.mark.parametrize("kind", sorted(BAD))
def test_a_rejected_answer_is_retried_and_the_retry_is_kept(kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _install(monkeypatch, BAD[kind], bad_attempts=1)
    source = fixtures.build_txt(tmp_path)
    sinks = _Sinks()
    result = asyncio.run(JobRunner(_job(tmp_path, source), translation_id=f"retry_{kind}",
                                   checkpoint_manager=CheckpointManager(db_path=str(tmp_path / "j.db")),
                                   sinks=sinks).run())
    assert result.status is JobStatus.COMPLETED, (result.status, sinks.logs[-5:])
    assert 'unit_answer_rejected' in sinks.logs
    output = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "I'm sorry" not in output
    assert "FR_paragraph FR_number 3" in output


@pytest.mark.parametrize("kind", sorted(BAD))
def test_an_answer_rejected_every_time_fails_the_unit_and_resume_retries_it(kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _install(monkeypatch, BAD[kind], bad_attempts=99)
    source = fixtures.build_txt(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / "j.db"))
    sinks = _Sinks()
    tid = f"rejected_{kind}"
    result = asyncio.run(JobRunner(_job(tmp_path, source), translation_id=tid,
                                   checkpoint_manager=manager, sinks=sinks).run())

    assert result.status is JobStatus.PARTIAL, result.status
    assert result.units_failed == 1
    assert 'unit_answer_rejected_exhausted' in sinks.logs
    output = (tmp_path / "out.txt").read_text(encoding="utf-8")
    # The unit keeps its source text; the bad answer is never written.
    assert "This is paragraph number 3." in output
    assert "I'm sorry" not in output

    # Resume with a model that now answers properly: the unit is retranslated.
    _install(monkeypatch, BAD[kind], bad_attempts=0)
    resumed = asyncio.run(JobRunner(_job(tmp_path, source), translation_id=tid,
                                    checkpoint_manager=manager, sinks=_Sinks(),
                                    is_resume=True).run())
    assert resumed.status is JobStatus.COMPLETED
    assert "FR_paragraph FR_number 3" in (tmp_path / "out.txt").read_text(encoding="utf-8")


def test_srt_cue_block_copied_back_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _install(monkeypatch, BAD['source_echo'], bad_attempts=99)
    source = fixtures.build_srt(tmp_path)
    sinks = _Sinks()
    result = asyncio.run(JobRunner(_job(tmp_path, source, 'srt'), translation_id="srt_echo",
                                   checkpoint_manager=CheckpointManager(db_path=str(tmp_path / "j.db")),
                                   sinks=sinks).run())
    assert result.status is JobStatus.PARTIAL
    assert 'unit_answer_rejected_exhausted' in sinks.logs


def test_srt_sound_tags_copied_back_are_accepted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    srt = tmp_path / "tags.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n[MUSIC]\n\n"
                   "2\n00:00:03,000 --> 00:00:04,000\n♪ ♪\n", encoding="utf-8")

    class Echo(LLMProvider):
        def __init__(self, **_kw):
            super().__init__(model="fake", api_keys=None, provider_name="fake")
            self.context_window = 8192

        async def _send(self, request):
            return LLMResponse(content=_wrap(_source(request.prompt)), prompt_tokens=1,
                               completion_tokens=1, context_used=2, context_limit=8192,
                               was_truncated=False, was_fallback=False)

    for target in ("src.core.llm.factory.create_llm_provider", "src.core.llm.create_llm_provider"):
        monkeypatch.setattr(target, lambda *a, **k: Echo())
    result = asyncio.run(JobRunner(_job(tmp_path, srt, 'srt'), translation_id="srt_tags",
                                   checkpoint_manager=CheckpointManager(db_path=str(tmp_path / "j.db")),
                                   sinks=_Sinks()).run())
    assert result.status is JobStatus.COMPLETED
