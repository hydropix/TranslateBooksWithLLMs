"""
Unit tests for POST /api/custom-instructions/extract-style.

Mirrors the fake-provider Flask-test-client shape of
tests/unit/glossary/test_ner_endpoint.py: `create_llm_provider` is
monkeypatched at its source module (`src.core.llm.factory`), since the
blueprint imports it fresh inside the request handler.

Each numbered test below corresponds to the matching item in the
`test_style_extract_endpoint.py` validation-criteria list of
plan/PLAN_StyleExtraction.md, Phase 5.
"""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

# Make the project importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.api.blueprints import custom_instruction_routes as cir
from src.core.llm.exceptions import RateLimitError

# Long enough that, split across up to 3 files, each excerpt budget still
# clears the extractor's MIN_SAMPLE_SIZE (1200 chars).
SAMPLE_TEXT = (
    "The rain kept falling on the empty avenue, and nobody came looking for answers. "
) * 120


def _upload(content: bytes, filename: str):
    return (io.BytesIO(content), filename)


def _canned_style_response(rules, summary="A dry noir style.", suggested_name="dry_noir", context=""):
    payload = json.dumps(
        {"summary": summary, "suggested_name": suggested_name, "context": context, "rules": rules}
    )
    return f"<STYLE_JSON>{payload}</STYLE_JSON>"


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _CannedProvider:
    """Stand-in provider returning a fixed response string."""

    def __init__(self, content, **kwargs):
        self._content = content
        self.kwargs = kwargs

    async def generate(self, user_prompt, system_prompt=None, **kwargs):
        return _FakeResponse(self._content)

    async def close(self):
        pass


class _RateLimitedProvider:
    """Stand-in provider whose generate() always raises RateLimitError."""

    def __init__(self, retry_after=None, provider_name="testprov"):
        self._retry_after = retry_after
        self._provider_name = provider_name

    async def generate(self, user_prompt, system_prompt=None, **kwargs):
        raise RateLimitError(
            "rate limit reached", retry_after=self._retry_after, provider=self._provider_name
        )

    async def close(self):
        pass


class _CapturingProvider:
    """Stand-in provider that records init kwargs and returns an empty style."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def generate(self, user_prompt, system_prompt=None, **kwargs):
        return _FakeResponse(_canned_style_response([]))

    async def close(self):
        pass


@pytest.fixture
def presets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cir, "get_config_path", lambda: str(tmp_path))
    return tmp_path / "Custom_Instructions"


@pytest.fixture
def client(presets_dir):
    app = Flask(__name__)
    app.register_blueprint(cir.create_custom_instruction_blueprint())
    with app.test_client() as c:
        yield c


def _post_extract(client, files, **fields):
    data = {"files": files}
    data.update(fields)
    return client.post(
        "/api/custom-instructions/extract-style",
        data=data,
        content_type="multipart/form-data",
    )


CLEAN_RULES = [
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
]

QUOTING_RULES = [
    {
        "dimension": "lexicon",
        "instruction": 'Prefer words like "dusk" and "gloom" whenever describing the weather.',
    },
    {
        "dimension": "imagery",
        "instruction": 'Use recurring images such as "a veil of night" for figurative darkness.',
    },
]


class TestExtractStyleHappyPath:
    def test_1_single_upload_returns_rules_and_one_per_file_entry(self, client):
        """1. single .txt upload -> 200, rules parsed, per_file has 1 entry."""
        canned = _canned_style_response(CLEAN_RULES)
        with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
            response = _post_extract(client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")])

        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert len(body["rules"]) == 3
        assert len(body["per_file"]) == 1
        assert body["per_file"][0]["filename"] == "novel.txt"

    def test_2_three_uploads_split_budget_across_files(self, client):
        """2. three uploads -> budget split, sample_chars <= max_chars + separators, per_file has 3."""
        canned = _canned_style_response(CLEAN_RULES)
        files = [
            _upload(SAMPLE_TEXT.encode(), "a.txt"),
            _upload(SAMPLE_TEXT.encode(), "b.txt"),
            _upload(SAMPLE_TEXT.encode(), "c.txt"),
        ]
        with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
            response = _post_extract(client, files=files, max_chars="9000")

        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert len(body["per_file"]) == 3
        # Separators ("===== EXCERPTS FROM: ... =====") add a bounded amount
        # of overhead per file on top of the raw sampled budget.
        assert body["sample_chars"] <= 9000 + 3 * 200

    def test_3_one_unreadable_file_among_three_warns_and_continues(self, client):
        """3. one unreadable file among three -> 200 with a warning naming it, 2 in per_file."""
        canned = _canned_style_response(CLEAN_RULES)
        files = [
            _upload(SAMPLE_TEXT.encode(), "a.txt"),
            _upload(b"", "empty.txt"),
            _upload(SAMPLE_TEXT.encode(), "c.txt"),
        ]
        with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
            response = _post_extract(client, files=files)

        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert len(body["per_file"]) == 2
        assert any("empty.txt" in warning for warning in body["warnings"])

    def test_8_max_chars_is_clamped_without_error(self, client):
        """8. max_chars: 99999 -> clamped to 12000, no error."""
        canned = _canned_style_response(CLEAN_RULES)
        with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
            response = _post_extract(
                client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")], max_chars="99999"
            )
        assert response.status_code == 200, response.get_json()

    def test_10_garbage_response_yields_empty_rules_and_warnings(self, client):
        """10. LLM returning garbage -> 200 with rules: [] and non-empty warnings."""
        with patch(
            "src.core.llm.factory.create_llm_provider",
            return_value=_CannedProvider("this is not json at all, just plain prose."),
        ):
            response = _post_extract(client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")])

        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert body["rules"] == []
        assert len(body["warnings"]) > 0

    def test_12_mixed_clean_and_quoting_rules(self, client):
        """12. 3 clean + 2 quoting rules -> all 5 returned; assembled has 3 bullet lines."""
        canned = _canned_style_response(CLEAN_RULES + QUOTING_RULES)
        with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
            response = _post_extract(client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")])

        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert len(body["rules"]) == 5

        flagged = [r for r in body["rules"] if r["flags"]]
        unflagged = [r for r in body["rules"] if not r["flags"]]
        assert len(flagged) == 2
        assert len(unflagged) == 3

        bullet_lines = [
            line for line in body["assembled"]["translation"].splitlines() if line.startswith("- ")
        ]
        assert len(bullet_lines) == 3

    def test_13_no_flags_or_evidence_survive_into_a_created_preset(self, client):
        """13. no flags/evidence key survives into a preset created from that response."""
        canned = _canned_style_response(CLEAN_RULES)
        with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
            extract_response = _post_extract(client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")])
        rules = extract_response.get_json()["rules"]
        assert rules and all("flags" in r and "evidence" in r for r in rules)

        create_response = client.post(
            "/api/custom-instructions",
            json={"name": "From Extraction", "mode": "source", "rules": rules},
        )
        assert create_response.status_code == 201

        stored = client.get("/api/custom-instructions/From_Extraction.yaml").get_json()
        for rule in stored["rules"]:
            assert "flags" not in rule
            assert "evidence" not in rule
            assert set(rule.keys()) == {"dimension", "instruction"}

    def test_14_response_carries_context_and_assembled_includes_it(self, client):
        """extract-style's 200 payload gains 'context', and 'assembled' is
        computed with that context plus the unflagged rules — so the modal's
        first preview matches what a create would store."""
        context = "A rain-soaked 1940s port city, no modern telecommunications."
        canned = _canned_style_response(CLEAN_RULES, context=context)
        with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
            response = _post_extract(client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")])

        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert body["context"] == context
        assert "## Setting" in body["assembled"]["translation"]
        assert context in body["assembled"]["translation"]


class TestExtractStyleValidation:
    def test_4_all_files_unreadable_is_400(self, client):
        """4. all files unreadable -> 400."""
        files = [_upload(b"", "a.txt"), _upload(b"", "b.txt")]
        response = _post_extract(client, files=files)
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_5_unsupported_extension_is_400(self, client):
        """5. .odt upload -> 400 listing the supported extensions."""
        response = _post_extract(client, files=[_upload(b"PK", "document.odt")])
        assert response.status_code == 400
        body = response.get_json()
        assert ".odt" in body["error"] or "odt" in body["error"].lower()
        assert ".txt" in body["error"]

    def test_6_six_files_is_400(self, client):
        """6. 6 files -> 400."""
        files = [_upload(SAMPLE_TEXT.encode(), f"f{i}.txt") for i in range(6)]
        response = _post_extract(client, files=files)
        assert response.status_code == 400

    def test_7_bogus_mode_is_400(self, client):
        """7. mode: "bogus" -> 400."""
        response = _post_extract(
            client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")], mode="bogus"
        )
        assert response.status_code == 400


class TestExtractStyleProviderHandling:
    def test_9_rate_limit_returns_429_with_retry_after_header(self, client):
        """9. provider raising RateLimitError -> 429 with a Retry-After header."""
        provider = _RateLimitedProvider(retry_after=17, provider_name="ollama")
        with patch("src.core.llm.factory.create_llm_provider", return_value=provider):
            response = _post_extract(client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")])

        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "17"
        body = response.get_json()
        assert body["provider"] == "ollama"
        assert body["retry_after"] == 17

    def test_11_endpoint_override_with_empty_api_key_uses_none(self, client):
        """11. api_endpoint override + empty api_key -> provider gets api_key=None."""
        captured = {}

        def _fake_factory(**kwargs):
            captured.update(kwargs)
            return _CapturingProvider(**kwargs)

        with patch("src.core.llm.factory.create_llm_provider", side_effect=_fake_factory):
            response = _post_extract(
                client,
                files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")],
                api_endpoint="http://127.0.0.1:11500/api/generate",
                api_key="",
            )

        assert response.status_code == 200, response.get_json()
        assert captured.get("api_key") is None


class TestExtractStyleStreaming:
    @staticmethod
    def _events(response):
        return [json.loads(line) for line in response.get_data(as_text=True).splitlines() if line.strip()]

    def test_stream_emits_stages_then_result(self, client):
        """stream=1 -> NDJSON stage events in order, ending with the usual payload."""
        canned = _canned_style_response(CLEAN_RULES)
        with patch("src.core.llm.factory.create_llm_provider", return_value=_CannedProvider(canned)):
            response = _post_extract(
                client,
                files=[_upload(SAMPLE_TEXT.encode(), "a.txt"), _upload(SAMPLE_TEXT.encode(), "b.txt")],
                stream="1",
            )
            events = self._events(response)

        assert response.status_code == 200
        assert response.mimetype == "application/x-ndjson"
        stages = [e["stage"] for e in events if e["type"] == "stage"]
        assert stages == ["sampling", "sampling", "llm", "parsing"]
        sampled = [e for e in events if e["type"] == "file_sampled"]
        assert [e["file"] for e in sampled] == ["a.txt", "b.txt"]
        assert events[-1]["type"] == "result"
        assert len(events[-1]["data"]["rules"]) == 3

    def test_stream_rate_limit_ends_with_error_event(self, client):
        """A provider error mid-stream ends with an `error` event carrying the status."""
        provider = _RateLimitedProvider(retry_after=17, provider_name="ollama")
        with patch("src.core.llm.factory.create_llm_provider", return_value=provider):
            response = _post_extract(
                client, files=[_upload(SAMPLE_TEXT.encode(), "novel.txt")], stream="1"
            )
            events = self._events(response)

        assert events[-1]["type"] == "error"
        assert events[-1]["status"] == 429
        assert events[-1]["data"]["retry_after"] == 17


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
