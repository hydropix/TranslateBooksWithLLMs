"""
Tests for the CLI glossary loader (JSON + CSV, rich vs back-compat shapes).
"""
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.glossary.cli_loader import (
    load_glossary_terms_from_file,
    load_glossary_from_file,
)


@pytest.fixture
def tmp_path_factory_for_files(tmp_path):
    return tmp_path


def test_json_list_shape(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps([
        {"source": "李凡", "target": "Li Fan", "category": "character"},
        {"source": "青玄宗", "target": "Qingxuan Sect", "category": "organization"},
    ]), encoding="utf-8")

    terms = load_glossary_terms_from_file(str(p))
    assert terms == {"李凡": "Li Fan", "青玄宗": "Qingxuan Sect"}

    rich_terms, metadata = load_glossary_from_file(str(p))
    assert rich_terms == terms
    assert metadata["李凡"] == {"category": "character"}
    assert metadata["青玄宗"] == {"category": "organization"}


def test_json_dict_with_terms_key(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({
        "name": "Whatever",
        "terms": [{"source": "a", "target": "b"}],
    }), encoding="utf-8")
    terms, metadata = load_glossary_from_file(str(p))
    assert terms == {"a": "b"}
    assert metadata == {}


def test_csv_with_header(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text(
        "source,target,category\n"
        "李凡,Li Fan,character\n"
        "苏婉清,Su Wanqing,character\n",
        encoding="utf-8",
    )
    terms, metadata = load_glossary_from_file(str(p))
    assert terms == {"李凡": "Li Fan", "苏婉清": "Su Wanqing"}
    assert metadata["李凡"] == {"category": "character"}
    assert metadata["苏婉清"] == {"category": "character"}


def test_csv_extra_columns_ignored(tmp_path):
    """Unknown columns (e.g. legacy 'notes') don't break loading."""
    p = tmp_path / "g.csv"
    p.write_text(
        "source,target,category,notes\n"
        "李凡,Li Fan,character,protagonist\n",
        encoding="utf-8",
    )
    terms, metadata = load_glossary_from_file(str(p))
    assert terms == {"李凡": "Li Fan"}
    assert metadata["李凡"] == {"category": "character"}


def test_csv_minimal_columns(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("source,target\nhello,bonjour\n", encoding="utf-8")
    terms, metadata = load_glossary_from_file(str(p))
    assert terms == {"hello": "bonjour"}
    assert metadata == {}


def test_missing_target_skipped(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps([
        {"source": "valid", "target": "ok"},
        {"source": "no_target", "target": ""},
        {"source": "", "target": "no_source"},
    ]), encoding="utf-8")
    terms, _ = load_glossary_from_file(str(p))
    assert terms == {"valid": "ok"}


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("source,target\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_glossary_from_file(str(p))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_glossary_from_file(str(tmp_path / "nope.json"))


def test_csv_missing_required_columns(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_glossary_from_file(str(p))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
