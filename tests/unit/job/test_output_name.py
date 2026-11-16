"""The ``{translatedName}`` output-name placeholder: one LLM call, never fatal."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from src.core.job import JobConfig, JobResult, JobStatus
from src.core.job.output_name import MAX_NAME_BYTES, resolve_output_name, safe_stem
from src.core.job.runner import JobRunner
from src.core.llm.exceptions import RateLimitError
from src.core.llm.provider_config import ProviderConfig
from src.core.llm.utils.extraction import TranslationExtractor
from src.persistence.checkpoint_manager import CheckpointManager


class FakeClient:
    """Answers every request with ``answer`` (an exception instance is raised)."""

    context_window = 2048

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []
        self.closed = False

    async def generate(self, prompt, **kwargs):
        self.prompts.append((kwargs.get('system_prompt'), prompt))
        if isinstance(self.answer, Exception):
            raise self.answer
        return SimpleNamespace(content=self.answer)

    def extract_translation(self, content):
        return TranslationExtractor('<TRANSLATION>', '</TRANSLATION>').extract(content)

    async def close(self):
        self.closed = True


def _answer(name):
    return f"<TRANSLATION>{name}</TRANSLATION>"


def _job(tmp_path, output_name="{translatedName} (English).epub", **changes):
    base = dict(
        operation='translate', file_type='epub', input_path=str(tmp_path / "upload.epub"),
        input_filename="86―エイティシックス― 第01巻.epub",
        output_path=str(tmp_path / output_name),
        source_language="Japanese", target_language="English",
        provider=ProviderConfig(provider="ollama", model="fake"), origin='cli',
    )
    base.update(changes)
    return JobConfig(**base)


@pytest.fixture
def resolve(monkeypatch):
    """Resolve a job's output name against a fake model answering ``answer``."""
    def _resolve(job, answer):
        client = FakeClient(answer)
        client.config = None

        def create(config):
            client.config = config
            return client

        monkeypatch.setattr("src.core.llm.factory.create_llm_provider", create)
        logs = []
        resolved = asyncio.run(resolve_output_name(
            job, lambda key, message="", data=None: logs.append(key)))
        return resolved, client, logs
    return _resolve


def test_token_is_replaced_by_the_translated_name(tmp_path, resolve):
    resolved, client, logs = resolve(_job(tmp_path), _answer("86—Eighty-Six, Volume 01"))

    assert resolved.output_filename == "86—Eighty-Six, Volume 01 (English).epub"
    assert resolved.output_dir == str(tmp_path)
    assert "86―エイティシックス― 第01巻" in client.prompts[0][1]
    assert client.closed
    assert "output_name_translated" in logs
    # The window the engines start the job with (no second model load on Ollama).
    assert client.config.context_window == 2048


def test_a_name_without_the_token_makes_no_request(tmp_path, resolve):
    job = _job(tmp_path, output_name="book (English).epub")
    resolved, client, _ = resolve(job, _answer("unused"))

    assert resolved is job
    assert client.prompts == []


@pytest.mark.parametrize("answer", [
    RateLimitError("slow down"),                # a file name never pauses a job
    RuntimeError("server down"),
    "",                                         # no usable answer
    "86—Eighty-Six, Volume 01",                 # untagged: could be a refusal
    "<think>Maybe <TRANSLATION>Guess 01</TRANSLATION>",   # reasoning that never ended
    _answer("line one\nline two"),              # not a single line
    _answer("86―エイティシックス― 第01巻"),       # still in the source script
    _answer("x" * 500),                         # runaway output
    _answer("86—Eighty-Six, Volume I"),         # the volume number was lost
    _answer("Your translation here"),           # the prompt's example, echoed
    _answer("86 Volume 01 of " + "a very long title " * 12),   # runaway output for this source
    _answer('"your translation here"'),         # the echoed example, wrapped in quotes
    _answer(". . ."),                           # nothing usable left
])
def test_any_failure_falls_back_to_the_original_name(tmp_path, resolve, answer):
    resolved, client, logs = resolve(_job(tmp_path), answer)

    assert resolved.output_filename == "86―エイティシックス― 第01巻 (English).epub"
    assert client.closed
    assert "output_name_translated" not in logs


@pytest.mark.parametrize("answer, expected", [
    # Reasoning is dropped, the tagged answer after it wins.
    ("<think>Is it <TRANSLATION>Guess</TRANSLATION>?</think>\n" + _answer("86—Eighty-Six, Volume 01"),
     "86—Eighty-Six, Volume 01"),
    (_answer("86—Eighty-Six, Volume 1"), "86—Eighty-Six, Volume 1"),        # zero padding may go
    (_answer("86—Eighty-Six, Volume 01.EPUB"), "86—Eighty-Six, Volume 01"),  # echoed extension
    (_answer("AC/DC: 86 \t 01?"), "AC_DC_ 86 01_"),
    (_answer('"86—Eighty-Six, Volume 01"'), "86—Eighty-Six, Volume 01"),     # wrapped in quotes
    (_answer("「86—Eighty-Six」 Volume 01"), "「86—Eighty-Six」 Volume 01"),   # inner quotes stay
    (_answer("86, Volume 01 \ud83d"), "86, Volume 01"),                     # half an emoji
    (_answer("Dr. 86, Volume 01"), "Dr. 86, Volume 01"),
])
def test_accepted_answers_are_cleaned(tmp_path, resolve, answer, expected):
    resolved, _, _ = resolve(_job(tmp_path), answer)

    assert resolved.output_filename == f"{expected} (English).epub"


@pytest.mark.parametrize("pattern", [
    "{translatedName} (Japanese) (some-model)_2026-01-01.epub",
    "{translatedName} - {translatedName} (Japanese).epub",
    "{translatedName}",                         # no extension: a dotted name must not escape the cap
])
def test_a_long_original_name_is_clipped_but_the_pattern_is_kept_whole(tmp_path, resolve, pattern):
    """The file is written hours later: its name must fit the filesystem, and
    clipping must never eat the language, the model or the extension."""
    long_name = "Dr. " + "長い物語の題名" * 30 + " 第2巻"
    job = _job(tmp_path, output_name=pattern, source_language="English",
               target_language="Japanese", input_filename=long_name + ".epub")
    resolved, _, logs = resolve(job, _answer(long_name))     # the answer does not fit either

    name = resolved.output_filename
    assert "output_name_translated" not in logs
    assert len(name.encode('utf-8')) <= MAX_NAME_BYTES
    fixed_parts = pattern.split("{translatedName}")
    assert name.startswith(fixed_parts[0] + "Dr. 長い") and name.endswith(fixed_parts[-1])
    assert all(part in name for part in fixed_parts)


def test_a_name_that_cannot_be_encoded_never_fails_the_job(tmp_path, resolve):
    job = _job(tmp_path, input_filename="Tome 1 \ud83d.epub", source_language="French")
    resolved, _, _ = resolve(job, RuntimeError("unreachable"))

    assert resolved.output_filename == "Tome 1 (English).epub"


@pytest.mark.parametrize("answer", [_answer("???"), _answer('""')])
def test_an_answer_made_of_unusable_characters_keeps_the_original_name(tmp_path, resolve, answer):
    job = _job(tmp_path, input_filename="物語.epub")       # no number: only the content rules apply
    resolved, _, logs = resolve(job, answer)

    assert resolved.output_filename == "物語 (English).epub"
    assert "output_name_translated" not in logs


@pytest.mark.parametrize("changes", [
    {'operation': 'refine'},
    {'operation': 'term_fix', 'term_fix': {'mode': 'literal', 'rules_text': 'a => b'}},
    {'target_language': 'japanese'},            # same language, different case
    # A display name cut at its length limit may have lost its volume number.
    {'input_filename': ("A" * 194 + " Vol. 12.epub")[:200], 'expected': "A" * 194 + " Vol (English).epub"},
    # The pattern alone fills the name: nothing a translation could add.
    {'output_path': "x" * 230 + "{translatedName}.epub", 'expected': "x" * 230 + ".epub"},
])
def test_jobs_that_do_not_translate_keep_the_original_name(tmp_path, resolve, changes):
    expected = changes.pop('expected', "86―エイティシックス― 第01巻 (English).epub")
    resolved, client, _ = resolve(_job(tmp_path, **changes), _answer("unused"))

    assert resolved.output_filename == expected
    assert client.prompts == []


def test_a_job_without_a_source_name_gets_a_generic_one(tmp_path, resolve):
    job = _job(tmp_path, output_name="{translatedName}.txt", input_filename=None, file_type='txt')
    resolved, client, _ = resolve(job, _answer("unused"))

    assert resolved.output_filename == "translation.txt"
    assert client.prompts == []


def test_safe_stem_neutralizes_what_a_model_could_return():
    assert safe_stem('../../etc/passwd') == '_.._etc_passwd'
    assert safe_stem('a\\b:c*d?"e<f>g|h') == 'a_b_c_d_e_f_g_h'
    assert safe_stem('  spaced \t out\x00. ') == 'spaced out_'
    assert safe_stem('{targetLang}') == '_targetLang_'
    assert safe_stem('invoice\u202excod') == 'invoice_xcod'      # right-to-left override
    assert safe_stem('CON') == '_CON'
    assert safe_stem(' . ') == ''


def test_runner_persists_the_resolved_name_before_the_job_starts(tmp_path, monkeypatch):
    """A resume reads the stored name: it must never hold the placeholder."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "upload.txt"
    source.write_text("one\n", encoding="utf-8")
    manager = CheckpointManager(db_path=str(tmp_path / "data" / "jobs.db"))
    client = FakeClient(_answer("The Little Prince"))
    monkeypatch.setattr("src.core.llm.factory.create_llm_provider", lambda config: client)
    seen = {}

    async def translate_file(job, ctx):
        seen['stored'] = manager.get_job("job-1")['config']['output_filename']
        with open(job.output_path, "w", encoding="utf-8") as f:
            f.write("translated")
        stats = {'total_chunks': 1, 'completed_chunks': 1, 'failed_chunks': 0, 'unfinished_chunks': 0}
        ctx.emit(stats)
        return JobResult.from_run(stats, job.output_path, engine_ok=True)

    job = _job(tmp_path, output_name="{translatedName} (English).txt", file_type='txt',
               input_path=str(source), input_filename="Le Petit Prince.txt",
               source_language="French")
    result = asyncio.run(JobRunner(job, translation_id="job-1", checkpoint_manager=manager,
                                   engines={'translate_file': translate_file}).run())

    assert result.status is JobStatus.COMPLETED
    assert seen['stored'] == "The Little Prince (English).txt"
    assert os.path.basename(result.output_path) == "The Little Prince (English).txt"
    assert os.path.exists(tmp_path / "The Little Prince (English).txt")
