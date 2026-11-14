"""The Files tab must not list the refine pass's .pre-refine working copy."""
from src.api.services.file_service import FileService


def test_pre_refine_copy_is_hidden(tmp_path):
    (tmp_path / "book (French).txt").write_text("done", encoding="utf-8")
    (tmp_path / "book (French).txt.pre-refine").write_text("draft", encoding="utf-8")

    names = [f['filename'] for f in FileService(str(tmp_path)).list_all_files()]

    assert names == ["book (French).txt"]
