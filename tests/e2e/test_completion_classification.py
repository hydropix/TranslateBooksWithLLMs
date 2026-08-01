"""Backend end-to-end checks for the completion classifier and TXT reassembly.

No browser here: these drive the HTTP API so the whole of
`run_translation_task`'s finalization branch runs for real.

- `src/api/completion_status.py` is unit-tested in isolation
  (`tests/unit/test_completion_status.py`); this file proves the wiring, and
  covers the plan's highest-blast-radius risk: every format must still end
  `completed` on a clean run.
- The paragraph round-trip is the end-to-end form of issue #208.
"""
from pathlib import Path

import pytest

from .conftest import E2E_MODEL, E2E_PROVIDER

pytestmark = pytest.mark.e2e


def _upload(api, path):
    with open(path, "rb") as handle:
        response = api.post("/api/upload", files={"file": (path.name, handle)})
    assert response.ok, response.text
    return response.json()


def _start(api, file_path, file_type, output_filename, model=None):
    payload = {
        "file_path": file_path,
        "file_type": file_type,
        "source_language": "English",
        "target_language": "French",
        "model": model or E2E_MODEL,
        "llm_provider": E2E_PROVIDER,
        "gemini_api_key": "__USE_ENV__",
        "llm_api_endpoint": "https://generativelanguage.googleapis.com",
        "output_filename": output_filename,
        "parallel_workers": 2,
    }
    response = api.post("/api/translate", json=payload)
    assert response.ok, response.text
    return response.json()["translation_id"]


def _run(api, path, file_type, output_filename, model=None):
    uploaded = _upload(api, path)
    translation_id = _start(api, uploaded["file_path"], file_type, output_filename, model)
    status = api.wait_for_terminal(translation_id)
    return translation_id, status


@pytest.mark.parametrize("file_type", ["txt", "srt", "docx", "epub"])
def test_clean_run_of_every_format_is_completed(api, tmp_path, file_type):
    """No format may regress to `partial` or `error` on a healthy run.

    The classifier reads the stats directly instead of being gated by file
    type, which is what makes this worth checking for all four.
    """
    from tests.characterization import fixtures

    builder = getattr(fixtures, f"build_{file_type}")
    source = builder(tmp_path)

    _, status = _run(api, source, file_type, f"e2e_clean_{file_type}.{file_type}")
    assert status == "completed", f"{file_type} ended as {status}"


def test_fallback_chunks_end_as_partial_and_keep_the_checkpoint(api, tmp_path):
    """A run whose chunks fall back to source text must not claim success.

    This is the backlog's named case for item 1.1: before the classifier, an
    EPUB where every chunk hit the Phase-3 fallback was reported `completed`
    and its checkpoint was deleted, losing the only way to retry.
    """
    from tests.characterization import fixtures

    source = fixtures.build_epub(tmp_path)
    translation_id, status = _run(
        api, source, "epub", "e2e_fallback.epub",
        model="this-model-does-not-exist-9999")

    assert status == "partial", f"expected partial, got {status}"

    resumable = api.get("/api/resumable").json().get("resumable_jobs", [])
    assert translation_id in {j.get("translation_id") for j in resumable}, (
        "the checkpoint was cleaned up; the job can no longer be resumed")


def test_txt_paragraph_breaks_survive_the_chunk_seams(api, tmp_path):
    """Issue #208: reassembly must restore the source paragraph structure.

    The input is deliberately large enough to span several chunks — with a
    single chunk there is no seam and the assertion would prove nothing.
    """
    paragraphs = [
        f"Paragraph {n}. The quick brown fox jumps over the lazy dog while the "
        "patient translator carefully preserves the meaning of every clause."
        for n in range(1, 16)
    ]
    source = tmp_path / "paragraphs.txt"
    source.write_text("\n\n".join(paragraphs), encoding="utf-8")

    translation_id, status = _run(api, source, "txt", "e2e_paragraphs.txt")
    assert status == "completed"

    job = api.get(f"/api/translation/{translation_id}").json()
    assert job["stats"]["total_chunks"] > 1, (
        f"the input did not span several chunks: {job['stats']}")

    output = Path(job["output_filepath"]).read_text(encoding="utf-8")
    # Provenance marks are invisible and irrelevant to paragraph structure.
    output = output.translate({c: None for c in (0x200b, 0x200c, 0x200d, 0x2060, 0xfeff)})

    assert len([p for p in output.split("\n\n") if p.strip()]) == len(paragraphs)
    assert "\n\n\n" not in output, "reassembly introduced a blank line the source lacked"
