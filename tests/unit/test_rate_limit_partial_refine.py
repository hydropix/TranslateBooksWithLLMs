"""A rate limit during a refine pass must keep the work already done.

Before this change, `refine_chunks` re-raised `RateLimitError` bare and the
list of chunks it had already refined was dropped on the floor: resuming after
the pause re-refined everything from scratch. The refined parts now ride on
`RateLimitError.partial_result`, and `refine_txt_file` writes them to disk
before letting the error propagate to the pause / auto-resume logic.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.chunking.reassembly import join_translated_chunks
from src.core.llm.exceptions import RateLimitError
from src.core.text_processor import split_text_into_chunks
from tests.job_helpers import make_ctx, make_job


# Three ~11-token paragraphs; max_tokens=25 splits them into exactly two chunks.
PARAGRAPHS = [
    "First paragraph with a reasonable amount of content in it.",
    "Second paragraph with a reasonable amount of content in it.",
    "Third paragraph with a reasonable amount of content in it.",
]
MULTI_PARAGRAPH_TEXT = "\n\n".join(PARAGRAPHS)
CHUNK_CONFIG = {"max_tokens_per_chunk": 25, "soft_limit_ratio": 0.5}


_ZERO_WIDTH = re.compile('[\u200b\u200c\u200d\u2060\ufeff]')


def _refined(draft: str) -> str:
    """The refinement the fake LLM applies to a draft chunk."""
    return draft.upper()


def _visible(text: str) -> str:
    """Drop the invisible characters normalization adds to the written file."""
    return _ZERO_WIDTH.sub("", text)


class _StubLLMClient:
    """Stands in for a provider client: refine_chunks only opens and closes it."""

    async def detect_thinking_model(self):
        return False

    async def close(self):
        return None


class _FakeResponse:
    prompt_tokens = 10
    completion_tokens = 10
    context_limit = 4096


def _install_fakes(monkeypatch, fail_at_index: int):
    """Patch the translator seams so chunk `fail_at_index` hits a 429.

    Returns the list of drafts handed to the refinement request, in order.
    """
    import src.core.translator as translator

    seen = []

    async def fake_request(*args, **kwargs):
        draft = kwargs["draft_translation"]
        index = len(seen)
        seen.append(draft)
        if index == fail_at_index:
            raise RateLimitError("429 Too Many Requests", provider="test")
        return _refined(draft), _FakeResponse()

    monkeypatch.setattr(translator, "_make_refinement_request", fake_request)
    import src.core.llm.factory as llm_factory
    monkeypatch.setattr(
        llm_factory, "create_llm_provider", lambda *a, **kw: _StubLLMClient()
    )
    return seen


def _reference_finalization(parts, structured_chunks) -> str:
    """The finalization sequence exactly as it was before the extraction.

    Frozen oracle for the `_write_refined_output` refactor: join, then the
    optional attribution footer. Any drift in the helper shows up as a diff
    here.
    """
    from src.config import ATTRIBUTION_ENABLED, GENERATOR_NAME, GENERATOR_SOURCE

    final_text = join_translated_chunks(parts, structured_chunks)
    if ATTRIBUTION_ENABLED:
        footer = f"\n\n{'=' * 60}\n"
        footer += f"Refined with {GENERATOR_NAME}\n"
        footer += f"{GENERATOR_SOURCE}\n"
        footer += f"{'=' * 60}\n"
        final_text += footer

    return final_text


async def _run_refine_txt_file(tmp_path, log_events=None):
    from src.core.refine.txt_refiner import refine_txt_file

    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text(MULTI_PARAGRAPH_TEXT, encoding="utf-8")

    log_callback = None
    if log_events is not None:
        log_callback = lambda event, message: log_events.append((event, message))

    result = await refine_txt_file(
        make_job(str(input_file), str(output_file), operation="refine",
                 target_language="English",
                 max_tokens_per_chunk=CHUNK_CONFIG["max_tokens_per_chunk"]),
        make_ctx(log_callback=log_callback),
        soft_limit_ratio=CHUNK_CONFIG["soft_limit_ratio"],
    )
    return result, output_file


class TestRateLimitErrorPartialResult:
    """Criterion 1 — the new attribute is additive and backward compatible."""

    def test_partial_result_defaults_to_none(self):
        assert RateLimitError("x").partial_result is None

    def test_partial_result_is_carried(self):
        assert RateLimitError("x", partial_result=[1]).partial_result == [1]

    def test_retry_after_and_provider_stay_positional(self):
        err = RateLimitError("x", 30, "gemini")
        assert err.retry_after == 30
        assert err.provider == "gemini"
        assert err.partial_result is None
        assert str(err) == "x"

    def test_all_four_arguments_positional(self):
        err = RateLimitError("x", 30, "gemini", ["a"])
        assert (err.retry_after, err.provider, err.partial_result) == (
            30, "gemini", ["a"]
        )


class TestRefineChunksAttachesPartial:
    """Criterion 2 — refine_chunks hands the whole list over on a 429."""

    @pytest.mark.asyncio
    async def test_partial_result_is_complete_and_ordered(self, monkeypatch):
        from src.core.translator import refine_chunks

        drafts = [f"Draft chunk number {i} with enough words to refine." for i in range(5)]
        _install_fakes(monkeypatch, fail_at_index=2)

        with pytest.raises(RateLimitError) as excinfo:
            await refine_chunks(
                drafts,
                [{} for _ in drafts],
                make_job(model="test-model", target_language="English",
                         endpoint="http://localhost:11434/api/generate"),
                make_ctx(),
            )

        parts = excinfo.value.partial_result
        assert parts is not None
        assert len(parts) == len(drafts)
        assert parts[0] == _refined(drafts[0])
        assert parts[1] == _refined(drafts[1])
        # Index 2 is the chunk that hit the limit; 3-4 were never attempted.
        assert parts[2:] == drafts[2:]

    @pytest.mark.asyncio
    async def test_partial_result_is_a_copy(self, monkeypatch):
        """Mutating the returned list must not reach refine_chunks' internals."""
        from src.core.translator import refine_chunks

        drafts = [f"Draft chunk number {i} with enough words to refine." for i in range(3)]
        _install_fakes(monkeypatch, fail_at_index=0)

        with pytest.raises(RateLimitError) as excinfo:
            await refine_chunks(
                drafts,
                [{} for _ in drafts],
                make_job(model="test-model", target_language="English",
                         endpoint="http://localhost:11434/api/generate"),
                make_ctx(),
            )

        parts = excinfo.value.partial_result
        assert parts == drafts
        assert parts is not drafts


class TestRefineTxtFilePartialSave:
    """Criterion 3 — the partial reaches disk and the error still propagates."""

    @pytest.mark.asyncio
    async def test_partial_output_is_written_and_error_propagates(
        self, tmp_path, monkeypatch
    ):
        structured_chunks = split_text_into_chunks(
            MULTI_PARAGRAPH_TEXT, **CHUNK_CONFIG
        )
        assert len(structured_chunks) == 2, "fixture must produce two chunks"
        drafts = [c["main_content"] for c in structured_chunks]

        _install_fakes(monkeypatch, fail_at_index=1)
        log_events = []

        with pytest.raises(RateLimitError):
            await _run_refine_txt_file(tmp_path, log_events)

        output_file = tmp_path / "output.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert content.strip()
        visible = _visible(content)
        assert _refined(drafts[0]) in visible
        # The chunk that hit the limit is kept as its unrefined draft.
        assert drafts[1] in visible
        assert content == _reference_finalization(
            [_refined(drafts[0]), drafts[1]], structured_chunks
        )

        assert any(event == "refine_partial_saved" for event, _ in log_events)

    @pytest.mark.asyncio
    async def test_no_partial_result_leaves_no_output(self, tmp_path, monkeypatch):
        """An unusable partial re-raises untouched and writes nothing."""
        import src.core.refine.txt_refiner as txt_refiner

        async def raise_bare(*args, **kwargs):
            raise RateLimitError("429", provider="test")

        monkeypatch.setattr(txt_refiner, "refine_chunks", raise_bare)

        with pytest.raises(RateLimitError):
            await _run_refine_txt_file(tmp_path)

        assert not (tmp_path / "output.txt").exists()

    @pytest.mark.asyncio
    async def test_mismatched_partial_length_re_raises(self, tmp_path, monkeypatch):
        """A partial that does not line up with the chunks is not written."""
        import src.core.refine.txt_refiner as txt_refiner

        async def raise_with_short_partial(*args, **kwargs):
            raise RateLimitError("429", provider="test", partial_result=["only one"])

        monkeypatch.setattr(txt_refiner, "refine_chunks", raise_with_short_partial)

        with pytest.raises(RateLimitError):
            await _run_refine_txt_file(tmp_path)

        assert not (tmp_path / "output.txt").exists()


class TestRefineTxtFileHappyPathUnchanged:
    """Criterion 4 — the _write_refined_output extraction is behaviour-neutral."""

    @pytest.mark.asyncio
    async def test_output_matches_the_pre_change_finalization(
        self, tmp_path, monkeypatch
    ):
        structured_chunks = split_text_into_chunks(
            MULTI_PARAGRAPH_TEXT, **CHUNK_CONFIG
        )
        drafts = [c["main_content"] for c in structured_chunks]

        _install_fakes(monkeypatch, fail_at_index=-1)  # never fails
        log_events = []

        result, output_file = await _run_refine_txt_file(tmp_path, log_events)

        assert result is True
        content = output_file.read_text(encoding="utf-8")
        assert content == _reference_finalization(
            [_refined(d) for d in drafts], structured_chunks
        )
        assert any(event == "refine_save_success" for event, _ in log_events)
        assert not any(event == "refine_partial_saved" for event, _ in log_events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
