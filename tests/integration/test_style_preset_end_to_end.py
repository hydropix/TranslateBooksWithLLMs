"""
End-to-end test proving a generated style preset actually reaches the
translation prompt (Phase 8 of plan/PLAN_StyleExtraction.md).

Chain exercised, with no network and no real LLM provider:

1. `POST /api/custom-instructions/extract-style` with a small `.txt` upload,
   a stubbed provider returning a canned `<STYLE_JSON>` payload of 4 rules.
2. `POST /api/custom-instructions` with those rules, writing a YAML preset.
3. `load_custom_instructions(filename, dir)` returns the assembled prose.
4. `src.core.job.prepare.resolve_custom_instructions` — the helper the job
   runner's prepare phase uses to snapshot the preset at translation time —
   yields the same strings as `translation_instructions` /
   `refinement_instructions`.
5. The written YAML file, read raw and parsed, carries no `evidence` and no
   `flags` key on any rule.

Fixture shape (fake provider + monkeypatched `get_config_path`) mirrors
tests/unit/style/test_style_extract_endpoint.py.
"""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from flask import Flask

# Make the project importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.blueprints import custom_instruction_routes as cir
from src.core.job.prepare import resolve_custom_instructions
from src.core.style.assembler import assemble_instructions
from src.utils.custom_instructions import load_custom_instructions

# Long enough that a single-file sample still clears the extractor's
# MIN_SAMPLE_SIZE (1200 chars).
SAMPLE_TEXT = (
    "The rain kept falling on the empty avenue, and nobody came looking for answers. "
) * 120

# Four rules, none of which trip the abstraction-violation lint (no quotes,
# no example markers, no word lists, no proper nouns, all >= 25 chars) so
# every rule survives into the assembled prose unfiltered.
FOUR_RULES = [
    {
        "dimension": "register",
        "instruction": "Keep the narrator emotionally distant from violent events throughout.",
    },
    {
        "dimension": "sentence_rhythm",
        "instruction": "Alternate short declarative sentences with one longer subordinate clause.",
    },
    {
        "dimension": "dialogue",
        "instruction": "Let characters answer questions with another question instead of a reply.",
    },
    {
        "dimension": "punctuation",
        "instruction": "Favor em dashes over commas when interrupting a thought mid sentence.",
    },
]


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _CannedProvider:
    """Stand-in provider returning a fixed response string."""

    def __init__(self, content, **kwargs):
        self._content = content

    async def generate(self, user_prompt, system_prompt=None, **kwargs):
        return _FakeResponse(self._content)

    async def close(self):
        pass


CONTEXT = "A rain-soaked 1940s port city, no modern telecommunications."


def _canned_style_response(rules, summary="A dry noir style.", suggested_name="dry_noir", context=""):
    payload = json.dumps(
        {"summary": summary, "suggested_name": suggested_name, "context": context, "rules": rules}
    )
    return f"<STYLE_JSON>{payload}</STYLE_JSON>"


@pytest.fixture
def presets_dir(tmp_path, monkeypatch):
    """Point the blueprint's presets directory at an isolated tmp_path.

    Never exercise this suite against the real Custom_Instructions/ folder —
    it holds tracked presets that a stray write/delete would destroy.
    """
    monkeypatch.setattr(cir, "get_config_path", lambda: str(tmp_path))
    return tmp_path / "Custom_Instructions"


@pytest.fixture
def client(presets_dir):
    app = Flask(__name__)
    app.register_blueprint(cir.create_custom_instruction_blueprint())
    with app.test_client() as c:
        yield c


def test_generated_preset_reaches_the_translation_prompt(client, presets_dir):
    # 1. Extract style from an uploaded sample via a stubbed provider.
    canned = _canned_style_response(FOUR_RULES, context=CONTEXT)
    with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
        extract_response = client.post(
            "/api/custom-instructions/extract-style",
            data={"files": [(io.BytesIO(SAMPLE_TEXT.encode()), "novel.txt")], "mode": "source"},
            content_type="multipart/form-data",
        )

    assert extract_response.status_code == 200, extract_response.get_json()
    extracted = extract_response.get_json()
    assert len(extracted["rules"]) == 4
    assert all("evidence" in r and "flags" in r for r in extracted["rules"])
    assert extracted["context"] == CONTEXT
    assert "## Setting" in extracted["assembled"]["translation"]
    assert CONTEXT in extracted["assembled"]["translation"]

    bullet_lines = [
        line for line in extracted["assembled"]["translation"].splitlines() if line.startswith("- ")
    ]
    assert len(bullet_lines) == 4

    # 2. Create a preset from those rules, including the extracted context.
    create_response = client.post(
        "/api/custom-instructions",
        json={
            "name": "Noir End To End",
            "mode": "source",
            "rules": extracted["rules"],
            "context": extracted["context"],
        },
    )
    assert create_response.status_code == 201, create_response.get_json()
    filename = create_response.get_json()["filename"]
    assert filename == "Noir_End_To_End.yaml"

    expected = assemble_instructions("source", FOUR_RULES, CONTEXT)
    assert expected["translation"] is not None
    assert expected["refinement"] is not None

    # 3. load_custom_instructions returns exactly the assembled prose.
    loaded = load_custom_instructions(filename, presets_dir)
    assert loaded["translation"] == expected["translation"]
    assert loaded["refinement"] == expected["refinement"]

    # 4. The prepare-phase helper that feeds the translation job yields the
    #    same strings as translation_instructions / refinement_instructions.
    log_calls = []

    def _log_callback(message_key, message_content="", data=None):
        log_calls.append((message_key, message_content))

    prompt_options = {"custom_instruction_file": filename}
    translation_instructions, refinement_instructions = resolve_custom_instructions(
        prompt_options, presets_dir, _log_callback
    )

    assert translation_instructions == expected["translation"]
    assert refinement_instructions == expected["refinement"]
    assert any(key == "custom_instructions" for key, _ in log_calls)

    # 5. No `evidence` or `flags` key leaked into the written YAML file, and
    #    the context reached the persisted preset verbatim.
    raw = (presets_dir / filename).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    assert parsed["context"] == CONTEXT
    assert len(parsed["rules"]) == 4
    for rule in parsed["rules"]:
        assert "evidence" not in rule
        assert "flags" not in rule
        assert set(rule.keys()) == {"dimension", "instruction"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
