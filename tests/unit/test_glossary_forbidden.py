import json
from pathlib import Path

from src.core.glossary.cli_loader import load_glossary_from_file
from src.core.glossary.injector import build_glossary_block


def test_cli_loader_preserves_forbidden_and_injector_renders_it(tmp_path: Path):
    path = tmp_path / "glossary.json"
    path.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "source": "Fitz",
                        "target": "Fitz",
                        "category": "character",
                        "forbidden": ["Fitz Chivalry", "Fitz, o Farseer"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    terms, metadata = load_glossary_from_file(str(path))
    assert terms == {"Fitz": "Fitz"}
    assert metadata["Fitz"]["forbidden"] == "Fitz Chivalry|Fitz, o Farseer"
    block = build_glossary_block(terms, "Portuguese", metadata)
    assert "forbidden" in block.lower()
    assert "Fitz Chivalry" in block


def test_cli_loader_accepts_pipe_separated_forbidden_translations(tmp_path: Path):
    path = tmp_path / "glossary.json"
    path.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "source": "Fitz",
                        "target": "Fitz",
                        "forbidden_translations": "Fitz Chivalry|Fitz, o Farseer",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _, metadata = load_glossary_from_file(str(path))
    assert metadata["Fitz"]["forbidden"] == "Fitz Chivalry|Fitz, o Farseer"


def test_injector_has_no_forbidden_line_without_forbidden_values():
    block = build_glossary_block({"Fitz": "Fitz"}, "Portuguese")
    assert "forbidden" not in block.lower()
