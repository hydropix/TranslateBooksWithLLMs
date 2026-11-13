"""Characterization baseline for the per-format progress signal.

Pins the sequence of (stable) stats dicts emitted by ``translate_file`` and
``refine_file`` for each format, plus a deterministic fingerprint of the
output. This is the safety net for the progress-system refactoring: any step
that changes the emitted progress trace will fail here, forcing a conscious
golden update (``UPDATE_CHARACTERIZATION=1``) with a reviewable diff.

The four-format comparison these goldens enable is the whole point: the
current system diverges per format, and these snapshots make that divergence
(and any future convergence) explicit and reviewable.
"""

import json
import os
from pathlib import Path

import pytest

from tests.characterization import fake_llm, fixtures, recorder

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.getenv("UPDATE_CHARACTERIZATION") == "1"

_BUILDERS = {
    "txt": fixtures.build_txt,
    "srt": fixtures.build_srt,
    "docx": fixtures.build_docx,
    "epub": fixtures.build_epub,
}


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def _compare_or_write(name: str, payload: dict):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = _golden_path(name)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if UPDATE or not path.exists():
        path.write_text(serialized + "\n", encoding="utf-8")
        if not UPDATE:
            pytest.skip(f"golden bootstrapped: {path.name} (re-run to compare)")
        return
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert payload == expected, (
        f"Progress trace for '{name}' diverged from golden {path.name}. "
        f"If intentional, re-run with UPDATE_CHARACTERIZATION=1 and review the diff."
    )


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    fake_llm.install(monkeypatch)


@pytest.mark.parametrize("fmt", list(_BUILDERS))
def test_translate_progress_baseline(fmt, tmp_path):
    input_path = _BUILDERS[fmt](tmp_path)
    suffix = input_path.suffix
    output_path = tmp_path / f"out{suffix}"
    sequence, output = recorder.record_translation(tmp_path, input_path, output_path)
    assert sequence, f"no progress callbacks emitted for {fmt} translate"
    _compare_or_write(f"translate_{fmt}", {"sequence": sequence, "output": output})


@pytest.mark.parametrize("fmt", list(_BUILDERS))
def test_refine_progress_baseline(fmt, tmp_path):
    input_path = _BUILDERS[fmt](tmp_path)
    suffix = input_path.suffix
    output_path = tmp_path / f"refined{suffix}"
    sequence, output = recorder.record_refine(tmp_path, input_path, output_path)
    assert sequence, f"no progress callbacks emitted for {fmt} refine"
    _compare_or_write(f"refine_{fmt}", {"sequence": sequence, "output": output})
