"""The server's startup banner lists every format the engine accepts.

It used to print "Supported formats: .txt, .epub, .srt" although DOCX, PDF
and PGN are supported. The banner now reads the engine's own extension map.
"""

import ast
from pathlib import Path

from src.core.adapters.translate_file import FILE_TYPE_BY_EXTENSION, get_file_type_from_path

REPO = Path(__file__).resolve().parents[2]


def test_extension_map_covers_every_supported_format():
    assert set(FILE_TYPE_BY_EXTENSION) == {'.txt', '.md', '.markdown', '.srt', '.pgn',
                                           '.epub', '.docx', '.pdf', '.fb2'}
    for ext, file_type in FILE_TYPE_BY_EXTENSION.items():
        assert get_file_type_from_path(f"book{ext.upper()}") == file_type


def test_banner_reads_the_extension_map():
    source = (REPO / "translation_api.py").read_text(encoding="utf-8")
    start_server = next(node for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.FunctionDef) and node.name == "start_server")
    body = ast.get_source_segment(source, start_server)
    assert "Supported formats" in body
    assert "FILE_TYPE_BY_EXTENSION" in body
    assert ".txt, .epub, .srt\"" not in body
