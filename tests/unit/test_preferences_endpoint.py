"""
Unit tests for the GET/PUT /api/preferences endpoints (Phase 3 of the
multi-device sharing plan).

Each numbered test below corresponds to the matching item in the
`test_preferences_endpoint.py` validation-criteria list of
plan/PLAN_Issue271_MultiDeviceSharing.md, Phase 3.
"""
import json
import sys
from pathlib import Path

import pytest
from flask import Flask

# Make the project importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.blueprints import config_routes


@pytest.fixture
def preferences_file(tmp_path, monkeypatch):
    """Point the blueprint's path resolution at an isolated tmp_path.

    The blueprint resolves `Path(get_config_path()) / 'data' / 'preferences.json'`
    at request time, so monkeypatching the module-level `get_config_path` name
    it actually calls is enough — the directory need not exist up front.
    """
    monkeypatch.setattr(config_routes, "get_config_path", lambda: str(tmp_path))
    return tmp_path / "data" / "preferences.json"


@pytest.fixture
def client(preferences_file):
    app = Flask(__name__)
    app.register_blueprint(config_routes.create_config_blueprint())
    with app.test_client() as c:
        yield c


class TestReadPath:
    def test_1_fresh_dir_returns_empty_document(self, client, preferences_file):
        """1. GET on a fresh dir returns {"preferences": {}} and writes nothing."""
        response = client.get("/api/preferences")
        assert response.status_code == 200
        assert response.get_json() == {"preferences": {}}
        assert not preferences_file.exists()
        assert not preferences_file.parent.exists()


class TestWritePath:
    def test_2_put_then_get_round_trips(self, client, preferences_file):
        """2. PUT a document, then GET returns it; the file lands under data/."""
        prefs = {"lastSourceLanguage": "French", "ttsEnabled": True, "n": 3}

        put = client.put("/api/preferences", json=prefs)
        assert put.status_code == 200
        assert put.get_json() == {"success": True, "preferences": prefs}

        assert preferences_file.exists()
        assert json.loads(preferences_file.read_text(encoding="utf-8")) == prefs

        get = client.get("/api/preferences")
        assert get.status_code == 200
        assert get.get_json()["preferences"] == prefs

    def test_3_put_replaces_the_whole_document(self, client):
        """3. A second PUT replaces rather than merges."""
        assert client.put(
            "/api/preferences", json={"a": "one", "b": 2}
        ).status_code == 200
        assert client.put(
            "/api/preferences", json={"c": False}
        ).status_code == 200

        assert client.get("/api/preferences").get_json()["preferences"] == {"c": False}


class TestValidation:
    def test_4_rejects_non_object_bodies_and_non_scalar_values(self, client):
        """4. A JSON array, a null value and a nested object are all rejected."""
        array_response = client.put("/api/preferences", json=["a", "b"])
        assert array_response.status_code == 400
        body = array_response.get_json()
        assert body["error"] == "Invalid preferences"
        assert body["message"] == "Preferences must be a JSON object"

        assert client.put("/api/preferences", json={"a": None}).status_code == 400
        assert client.put("/api/preferences", json={"a": {"b": 1}}).status_code == 400
        assert client.put("/api/preferences", json={"a": [1, 2]}).status_code == 400

    def test_5_rejects_oversized_documents_keys_and_values(self, client):
        """5. Too many keys, an over-long value, and an over-long key are rejected."""
        too_many_keys = {f"k{i}": i for i in range(101)}
        assert client.put("/api/preferences", json=too_many_keys).status_code == 400

        assert client.put(
            "/api/preferences", json={"a": "x" * 2049}
        ).status_code == 400

        assert client.put(
            "/api/preferences", json={"k" * 65: "value"}
        ).status_code == 400

    def test_5b_accepts_the_documents_just_within_the_caps(self, client):
        """The caps are inclusive: 100 keys, 2048 chars, a 64-char key all pass."""
        assert client.put(
            "/api/preferences", json={f"k{i}": i for i in range(100)}
        ).status_code == 200
        assert client.put(
            "/api/preferences", json={"a": "x" * 2048}
        ).status_code == 200
        assert client.put(
            "/api/preferences", json={"k" * 64: "value"}
        ).status_code == 200

    def test_5c_rejects_an_empty_key(self, client):
        """An empty key carries no meaning and is rejected by the 1-char floor."""
        assert client.put("/api/preferences", json={"": "value"}).status_code == 400

    def test_no_body_is_rejected(self, client):
        """A PUT with no JSON body parses to None and is rejected, not persisted."""
        response = client.put("/api/preferences")
        assert response.status_code == 400
        assert response.get_json()["message"] == "Preferences must be a JSON object"


class TestCorruptFile:
    def test_6_corrupt_file_degrades_to_empty_and_stays_writable(
        self, client, preferences_file
    ):
        """6. A corrupt file reads as {} and the next PUT repairs it."""
        preferences_file.parent.mkdir(parents=True, exist_ok=True)
        preferences_file.write_text("not json", encoding="utf-8")

        assert client.get("/api/preferences").get_json() == {"preferences": {}}

        assert client.put(
            "/api/preferences", json={"theme": "dark"}
        ).status_code == 200
        assert client.get("/api/preferences").get_json()["preferences"] == {
            "theme": "dark"
        }

    def test_json_scalar_file_is_not_treated_as_a_document(
        self, client, preferences_file
    ):
        """Valid JSON that is not an object also degrades to {}."""
        preferences_file.parent.mkdir(parents=True, exist_ok=True)
        preferences_file.write_text('["a", "b"]', encoding="utf-8")

        assert client.get("/api/preferences").get_json() == {"preferences": {}}


class TestAtomicWrite:
    def test_write_leaves_no_temporary_file_behind(self, client, preferences_file):
        """The atomic write must not litter data/ with .tmp leftovers."""
        assert client.put("/api/preferences", json={"a": 1}).status_code == 200
        leftovers = list(preferences_file.parent.glob("preferences-*.tmp"))
        assert leftovers == []
