"""
Unit tests for the /api/custom-instructions* CRUD, assemble, and
duplicate/export endpoints (Phase 5 of the style-extraction plan).

Each numbered test below corresponds to the matching item in the
`test_custom_instruction_endpoints.py` validation-criteria list of
plan/PLAN_StyleExtraction.md, Phase 5.
"""
import sys
from pathlib import Path

import pytest
import yaml
from flask import Flask

# Make the project importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.api.blueprints import custom_instruction_routes as cir
from src.core.style.assembler import assemble_instructions


@pytest.fixture
def presets_dir(tmp_path, monkeypatch):
    """Point the blueprint's directory resolution at an isolated tmp_path.

    The blueprint resolves `Path(get_config_path()) / 'Custom_Instructions'`
    at request time, so monkeypatching the module-level `get_config_path`
    name the blueprint actually calls is enough — no need for the directory
    to exist up front.
    """
    monkeypatch.setattr(cir, "get_config_path", lambda: str(tmp_path))
    return tmp_path / "Custom_Instructions"


@pytest.fixture
def client(presets_dir):
    app = Flask(__name__)
    app.register_blueprint(cir.create_custom_instruction_blueprint())
    with app.test_client() as c:
        yield c


def _create(client, **overrides):
    body = {"name": "Noir Style", "translation": "Write like a noir novel."}
    body.update(overrides)
    return client.post("/api/custom-instructions", json=body)


class TestListing:
    def test_1_fresh_dir_returns_empty_listing(self, client):
        """1. GET on a fresh dir returns {"files": [], "count": 0}."""
        response = client.get("/api/custom-instructions")
        assert response.status_code == 200
        body = response.get_json()
        assert body["files"] == []
        assert body["count"] == 0

    def test_2_created_preset_is_listed_with_description(self, client):
        """2. POST then GET lists the preset with its description."""
        create_resp = _create(client, description="Hardboiled register")
        assert create_resp.status_code == 201

        response = client.get("/api/custom-instructions")
        body = response.get_json()
        assert body["count"] == 1
        assert body["files"][0]["filename"] == "Noir_Style.yaml"
        assert body["files"][0]["description"] == "Hardboiled register"


class TestCreate:
    def test_3_duplicate_name_conflicts_then_overwrite_replaces(self, client, presets_dir):
        """3. POST with the same name -> 409; with overwrite -> replaced."""
        first = _create(client, translation="First version")
        assert first.status_code == 201

        conflict = _create(client, translation="Second version")
        assert conflict.status_code == 409
        assert "error" in conflict.get_json()

        overwritten = _create(client, translation="Second version", overwrite=True)
        assert overwritten.status_code in (200, 201)

        preset = yaml.safe_load((presets_dir / "Noir_Style.yaml").read_text(encoding="utf-8"))
        assert preset["translation"].strip() == "Second version"

    def test_4_rules_win_over_bogus_client_translation(self, client):
        """4. Rules + a bogus translation -> stored prose is the assembled one."""
        rules = [{"dimension": "register", "instruction": "Stay cynical and terse."}]
        response = client.post(
            "/api/custom-instructions",
            json={
                "name": "Assembled Style",
                "mode": "source",
                "rules": rules,
                "translation": "this is bogus and must be ignored",
            },
        )
        assert response.status_code == 201

        get_resp = client.get("/api/custom-instructions/Assembled_Style.yaml")
        stored = get_resp.get_json()
        expected = assemble_instructions("source", rules)
        assert stored["translation"] == expected["translation"]
        assert "bogus" not in stored["translation"]

    def test_5_manual_true_stores_client_prose_verbatim(self, client):
        """5. Rules + manual:true -> the client prose is stored verbatim."""
        rules = [{"dimension": "register", "instruction": "Stay cynical and terse."}]
        response = client.post(
            "/api/custom-instructions",
            json={
                "name": "Manual Style",
                "mode": "source",
                "rules": rules,
                "translation": "Exactly this text, verbatim.",
                "refinement": "Also exactly this text.",
                "manual": True,
            },
        )
        assert response.status_code == 201

        stored = client.get("/api/custom-instructions/Manual_Style.yaml").get_json()
        assert stored["translation"] == "Exactly this text, verbatim."
        assert stored["refinement"] == "Also exactly this text."
        assert stored["rules"] == [{"dimension": "register", "instruction": "Stay cynical and terse."}]

    def test_6_neither_rules_nor_prose_is_400(self, client):
        """6. POST with neither rules nor prose -> 400."""
        response = client.post("/api/custom-instructions", json={"name": "Empty Style"})
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_context_is_stored_and_reassembled(self, client):
        """Create with a context stores it and re-assembles the setting section into the prose."""
        rules = [{"dimension": "register", "instruction": "Stay cynical and terse."}]
        context = "Victorian London, gaslit streets, no electricity."
        response = client.post(
            "/api/custom-instructions",
            json={
                "name": "Contextual Style",
                "mode": "source",
                "rules": rules,
                "context": context,
            },
        )
        assert response.status_code == 201

        stored = client.get("/api/custom-instructions/Contextual_Style.yaml").get_json()
        assert stored["context"] == context
        expected = assemble_instructions("source", rules, context)
        assert stored["translation"] == expected["translation"]
        assert "## Setting" in stored["translation"]
        assert context in stored["translation"]

    def test_context_over_600_chars_is_400(self, client):
        """A context longer than 600 characters -> 400 naming the field."""
        response = client.post(
            "/api/custom-instructions",
            json={
                "name": "Too Long Context",
                "translation": "Some prose.",
                "context": "x" * 601,
            },
        )
        assert response.status_code == 400
        assert "context" in response.get_json()["error"]


class TestReadUpdateDelete:
    def test_7_get_round_trips_every_field(self, client):
        """7. GET /<f> round-trips every field."""
        rules = [{"dimension": "lexicon", "instruction": "Prefer concrete, plain nouns."}]
        source_files = ["book_one.epub", "book_two.txt"]
        context = "1920s Chicago, prohibition era, no cell phones."
        create_resp = client.post(
            "/api/custom-instructions",
            json={
                "name": "Full Style",
                "description": "A complete preset",
                "mode": "model",
                "context": context,
                "source_files": source_files,
                "rules": rules,
                "translation": "Verbatim translation prose.",
                "refinement": "Verbatim refinement prose.",
                "manual": True,
            },
        )
        assert create_resp.status_code == 201

        stored = client.get("/api/custom-instructions/Full_Style.yaml").get_json()
        assert stored["description"] == "A complete preset"
        assert stored["mode"] == "model"
        assert stored["context"] == context
        assert stored["source_files"] == source_files
        assert stored["rules"] == rules
        assert stored["translation"] == "Verbatim translation prose."
        assert stored["refinement"] == "Verbatim refinement prose."
        assert stored["filename"] == "Full_Style.yaml"
        assert stored["display_name"] == "Full_Style"

    def test_8_put_partial_update_leaves_other_fields_intact(self, client):
        """8. PUT partial update leaves untouched fields intact."""
        rules = [{"dimension": "lexicon", "instruction": "Prefer concrete, plain nouns."}]
        client.post(
            "/api/custom-instructions",
            json={
                "name": "Partial Style",
                "description": "Original description",
                "mode": "source",
                "rules": rules,
                "translation": "Original translation prose.",
                "refinement": "Original refinement prose.",
                "manual": True,
            },
        )

        put_resp = client.put(
            "/api/custom-instructions/Partial_Style.yaml",
            json={"description": "Updated description only"},
        )
        assert put_resp.status_code == 200

        stored = client.get("/api/custom-instructions/Partial_Style.yaml").get_json()
        assert stored["description"] == "Updated description only"
        assert stored["mode"] == "source"
        assert stored["rules"] == rules
        assert stored["translation"] == "Original translation prose."
        assert stored["refinement"] == "Original refinement prose."

    def test_9_delete_then_get_is_404(self, client):
        """9. DELETE -> 200, then GET -> 404."""
        _create(client)

        delete_resp = client.delete("/api/custom-instructions/Noir_Style.yaml")
        assert delete_resp.status_code == 200
        assert delete_resp.get_json() == {"deleted": True}

        get_resp = client.get("/api/custom-instructions/Noir_Style.yaml")
        assert get_resp.status_code == 404

    def test_get_missing_file_is_404(self, client):
        """Extra coverage: GET on a name that was never created is 404."""
        response = client.get("/api/custom-instructions/does_not_exist.yaml")
        assert response.status_code == 404

    def test_get_malformed_yaml_is_422(self, client, presets_dir):
        """Extra coverage: malformed YAML on GET /<f> is 422."""
        presets_dir.mkdir(parents=True, exist_ok=True)
        (presets_dir / "broken.yaml").write_text("translation: [unterminated", encoding="utf-8")

        response = client.get("/api/custom-instructions/broken.yaml")
        assert response.status_code == 422


class TestDuplicate:
    def test_10_duplicate_twice_produces_copy_then_copy2(self, client):
        """10. duplicate twice -> x_copy, x_copy2."""
        _create(client)

        first = client.post("/api/custom-instructions/Noir_Style.yaml/duplicate", json={})
        assert first.status_code == 201
        assert first.get_json()["filename"] == "Noir_Style_copy.yaml"

        second = client.post("/api/custom-instructions/Noir_Style.yaml/duplicate", json={})
        assert second.status_code == 201
        assert second.get_json()["filename"] == "Noir_Style_copy2.yaml"

    def test_duplicate_preserves_manual_prose(self, client):
        """A duplicate of a manually-overridden preset must not re-assemble."""
        rules = [{"dimension": "register", "instruction": "Stay cynical and terse."}]
        client.post(
            "/api/custom-instructions",
            json={
                "name": "Manual Source",
                "mode": "source",
                "rules": rules,
                "translation": "Hand-edited translation prose.",
                "refinement": "Hand-edited refinement prose.",
                "manual": True,
            },
        )

        dup = client.post("/api/custom-instructions/Manual_Source.yaml/duplicate", json={})
        assert dup.status_code == 201

        stored = client.get(f"/api/custom-instructions/{dup.get_json()['filename']}").get_json()
        assert stored["translation"] == "Hand-edited translation prose."
        assert stored["refinement"] == "Hand-edited refinement prose."
        assert stored["rules"] == rules

    def test_duplicate_missing_source_is_404(self, client):
        response = client.post("/api/custom-instructions/does_not_exist.yaml/duplicate", json={})
        assert response.status_code == 404

    def test_duplicate_copies_context(self, client):
        """A duplicate must copy the source preset's `context` like the other metadata."""
        context = "A far-future space station, no gravity, artificial daylight cycles."
        client.post(
            "/api/custom-instructions",
            json={
                "name": "Context Source",
                "translation": "Some prose.",
                "context": context,
            },
        )

        dup = client.post("/api/custom-instructions/Context_Source.yaml/duplicate", json={})
        assert dup.status_code == 201

        stored = client.get(f"/api/custom-instructions/{dup.get_json()['filename']}").get_json()
        assert stored["context"] == context


class TestExport:
    def test_11_export_has_attachment_header_and_parses_as_yaml(self, client):
        """11. export -> Content-Disposition: attachment, body parses as YAML."""
        _create(client, description="Exportable preset")

        response = client.get("/api/custom-instructions/Noir_Style.yaml/export")
        assert response.status_code == 200
        assert "attachment" in response.headers.get("Content-Disposition", "")

        parsed = yaml.safe_load(response.get_data(as_text=True))
        assert isinstance(parsed, dict)
        assert parsed["description"] == "Exportable preset"


class TestPathTraversal:
    """12. Path traversal on every <filename> route -> 400, nothing escapes the dir.

    Forward-slash traversal never reaches the route handler at all (Werkzeug's
    default string converter does not match a literal '/' in a path segment,
    and normalizes '../' away before matching), so it can't be used to probe
    this endpoint. A backslash is not a URL path separator, so it *does*
    reach the handler as a literal filename — that's the vector `is_safe_filename`
    (which forbids backslashes) must reject.
    """

    TRAVERSAL_NAME = "..\\..\\evil.yaml"

    def test_get_traversal_is_400(self, client, presets_dir):
        response = client.get(f"/api/custom-instructions/{self.TRAVERSAL_NAME}")
        assert response.status_code == 400
        assert not presets_dir.parent.parent.joinpath("evil.yaml").exists()

    def test_put_traversal_is_400(self, client):
        response = client.put(
            f"/api/custom-instructions/{self.TRAVERSAL_NAME}",
            json={"translation": "hello"},
        )
        assert response.status_code == 400

    def test_delete_traversal_is_400(self, client):
        response = client.delete(f"/api/custom-instructions/{self.TRAVERSAL_NAME}")
        assert response.status_code == 400

    def test_duplicate_traversal_is_400(self, client):
        response = client.post(f"/api/custom-instructions/{self.TRAVERSAL_NAME}/duplicate", json={})
        assert response.status_code == 400

    def test_export_traversal_is_400(self, client):
        response = client.get(f"/api/custom-instructions/{self.TRAVERSAL_NAME}/export")
        assert response.status_code == 400

    def test_no_file_created_or_removed_outside_dir(self, client, presets_dir, tmp_path):
        client.put(
            f"/api/custom-instructions/{self.TRAVERSAL_NAME}",
            json={"translation": "hello"},
        )
        # Nothing should exist anywhere under tmp_path except possibly the
        # (empty) Custom_Instructions dir itself.
        created_files = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert created_files == []


class TestAssemble:
    def test_13_assemble_matches_assemble_instructions(self, client):
        """13. POST /assemble returns the same strings as assemble_instructions."""
        rules = [
            {"dimension": "register", "instruction": "Stay cynical, dry, and emotionally guarded throughout."},
            {"dimension": "punctuation", "instruction": "Favor short declarative sentences over long ones."},
        ]
        response = client.post(
            "/api/custom-instructions/assemble", json={"mode": "source", "rules": rules}
        )
        assert response.status_code == 200
        body = response.get_json()

        expected = assemble_instructions("source", rules)
        assert body["translation"] == expected["translation"]
        assert body["refinement"] == expected["refinement"]
        assert body["flags"] == [[], []]

    def test_assemble_bogus_mode_is_400(self, client):
        response = client.post(
            "/api/custom-instructions/assemble", json={"mode": "bogus", "rules": []}
        )
        assert response.status_code == 400

    def test_assemble_with_context_returns_setting_section(self, client):
        """assemble with a context returns the '## Setting' section in the prose."""
        rules = [{"dimension": "register", "instruction": "Stay cynical and emotionally guarded throughout."}]
        context = "Feudal Japan, pre-industrial, no firearms."
        response = client.post(
            "/api/custom-instructions/assemble",
            json={"mode": "source", "rules": rules, "context": context},
        )
        assert response.status_code == 200
        body = response.get_json()

        expected = assemble_instructions("source", rules, context)
        assert body["translation"] == expected["translation"]
        assert body["refinement"] == expected["refinement"]
        assert "## Setting" in body["translation"]
        assert context in body["translation"]
        assert body["flags"] == [[]]

    def test_assemble_context_over_600_chars_is_400(self, client):
        response = client.post(
            "/api/custom-instructions/assemble",
            json={"mode": "source", "rules": [], "context": "x" * 601},
        )
        assert response.status_code == 400
        assert "context" in response.get_json()["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestStylesRegressions:
    """Regressions reported against the 1.5.15 Styles tab (#293, #296)."""

    def test_create_with_chinese_name_succeeds(self, client, presets_dir):
        """#296: the zh-CN default name '新建空白风格' used to slugify to empty."""
        response = _create(client, name="新建空白风格")
        assert response.status_code == 201
        assert response.get_json()["filename"] == "新建空白风格.yaml"
        assert (presets_dir / "新建空白风格.yaml").exists()

    def test_hand_renamed_file_is_listed_and_readable(self, client, presets_dir):
        """#293: a preset renamed by hand with spaces must still be listed."""
        presets_dir.mkdir(parents=True)
        (presets_dir / "Romance Novel (v2).txt").write_text("Be lyrical.", encoding="utf-8")

        listing = client.get("/api/custom-instructions").get_json()
        assert [f["filename"] for f in listing["files"]] == ["Romance Novel (v2).txt"]
        preset = client.get("/api/custom-instructions/Romance Novel (v2).txt").get_json()
        assert preset["translation"] == "Be lyrical."

    def test_put_with_unchanged_name_keeps_hand_named_file(self, client, presets_dir):
        presets_dir.mkdir(parents=True)
        (presets_dir / "My Style.yaml").write_text("translation: old\n", encoding="utf-8")

        response = client.put(
            "/api/custom-instructions/My Style.yaml",
            json={"name": "My Style", "translation": "new", "manual": True},
        )
        assert response.status_code == 200
        assert response.get_json()["filename"] == "My Style.yaml"
        assert yaml.safe_load((presets_dir / "My Style.yaml").read_text(encoding="utf-8"))["translation"] == "new"

    def test_put_with_new_name_renames_and_keeps_content(self, client, presets_dir):
        assert _create(client).status_code == 201

        response = client.put(
            "/api/custom-instructions/Noir_Style.yaml",
            json={"name": "Hardboiled", "translation": "Short sentences.", "manual": True},
        )
        assert response.status_code == 200
        assert response.get_json() == {"filename": "Hardboiled.yaml", "display_name": "Hardboiled"}
        assert not (presets_dir / "Noir_Style.yaml").exists()
        data = yaml.safe_load((presets_dir / "Hardboiled.yaml").read_text(encoding="utf-8"))
        assert data["translation"] == "Short sentences."

    def test_rename_onto_existing_preset_conflicts_and_changes_nothing(self, client, presets_dir):
        """#293: saving must never silently overwrite another preset."""
        assert _create(client, name="First", translation="one").status_code == 201
        assert _create(client, name="Second", translation="two").status_code == 201

        response = client.put(
            "/api/custom-instructions/Second.yaml",
            json={"name": "First", "translation": "changed", "manual": True},
        )
        assert response.status_code == 409
        first = yaml.safe_load((presets_dir / "First.yaml").read_text(encoding="utf-8"))
        second = yaml.safe_load((presets_dir / "Second.yaml").read_text(encoding="utf-8"))
        assert first["translation"] == "one"
        assert second["translation"] == "two"

    def test_put_on_txt_preset_saves_plain_text_in_place(self, client, presets_dir):
        """Saving a legacy .txt used to fail with a 400 (write_preset is YAML-only)."""
        presets_dir.mkdir(parents=True)
        (presets_dir / "legacy.txt").write_text("old text", encoding="utf-8")

        response = client.put(
            "/api/custom-instructions/legacy.txt",
            json={"name": "legacy", "translation": "new text", "refinement": "new text", "manual": True},
        )
        assert response.status_code == 200
        assert response.get_json()["filename"] == "legacy.txt"
        assert (presets_dir / "legacy.txt").read_text(encoding="utf-8").strip() == "new text"

    def test_put_on_txt_with_distinct_phases_converts_to_yaml(self, client, presets_dir):
        presets_dir.mkdir(parents=True)
        (presets_dir / "legacy.txt").write_text("old text", encoding="utf-8")

        response = client.put(
            "/api/custom-instructions/legacy.txt",
            json={"translation": "translate this way", "refinement": "refine that way", "manual": True},
        )
        assert response.status_code == 200
        assert response.get_json()["filename"] == "legacy.yaml"
        assert not (presets_dir / "legacy.txt").exists()
        data = yaml.safe_load((presets_dir / "legacy.yaml").read_text(encoding="utf-8"))
        assert data["translation"] == "translate this way"
        assert data["refinement"] == "refine that way"

    def test_delete_txt_preset(self, client, presets_dir):
        presets_dir.mkdir(parents=True)
        (presets_dir / "legacy.txt").write_text("text", encoding="utf-8")

        response = client.delete("/api/custom-instructions/legacy.txt")
        assert response.status_code == 200
        assert not (presets_dir / "legacy.txt").exists()

    def test_export_non_ascii_name_sets_rfc5987_filename(self, client):
        assert _create(client, name="武侠").status_code == 201

        response = client.get("/api/custom-instructions/武侠.yaml/export")
        assert response.status_code == 200
        disposition = response.headers["Content-Disposition"]
        assert "filename*=UTF-8''%E6%AD%A6%E4%BE%A0.yaml" in disposition
