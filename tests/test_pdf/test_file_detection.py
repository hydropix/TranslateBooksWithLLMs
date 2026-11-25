"""
Tests for PDF support in the file type detector and the PDF package import.
"""

import importlib
import sys
import zipfile

import pytest

from src.utils.file_detector import detect_file_type, detect_file_type_by_content


def _make_epub(path):
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('mimetype', 'application/epub+zip')
        zf.writestr('META-INF/container.xml', '<container/>')


def _make_docx(path):
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('[Content_Types].xml', '<Types/>')
        zf.writestr('word/document.xml', '<document/>')


class TestPdfDetection:
    def test_pdf_extension_without_reading_file(self):
        # The file does not exist: the extension fast path must be enough
        assert detect_file_type("does_not_exist.pdf") == "pdf"

    def test_pdf_extension_is_case_insensitive(self):
        assert detect_file_type("Report.PDF") == "pdf"

    def test_pdf_magic_bytes_with_unknown_extension(self, tmp_path):
        path = tmp_path / "doc.bin"
        path.write_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n")
        assert detect_file_type(str(path)) == "pdf"
        assert detect_file_type_by_content(str(path)) == "pdf"

    def test_pdf_header_after_junk_bytes(self, tmp_path):
        path = tmp_path / "junk_prefix.bin"
        path.write_bytes(b"\x00\x01\x02" * 33 + b"\x00" + b"%PDF-1.4\n" + b"\x00" * 50)
        assert detect_file_type(str(path)) == "pdf"

    def test_zip_file_is_not_detected_as_pdf(self, tmp_path):
        path = tmp_path / "archive.bin"
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr('hello.txt', 'hello')
        assert path.read_bytes()[:4] == b"PK\x03\x04"
        assert detect_file_type_by_content(str(path)) != "pdf"

    def test_fake_zip_header_is_not_detected_as_pdf(self, tmp_path):
        path = tmp_path / "fake.bin"
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
        assert detect_file_type_by_content(str(path)) != "pdf"

    def test_pdf_extension_not_in_known_text_extensions(self):
        from src.utils.file_detector import KNOWN_TEXT_EXTENSIONS
        assert '.pdf' not in KNOWN_TEXT_EXTENSIONS

    def test_unsupported_error_message_lists_pdf(self, tmp_path):
        path = tmp_path / "blob.bin"
        path.write_bytes(bytes(range(256)) * 4)
        with pytest.raises(ValueError, match=r"\.docx, \.pdf, \.fb2, or plain text"):
            detect_file_type(str(path))


class TestContentDetectionUnchanged:
    """The 64 -> 1024 byte header read must not change other formats."""

    def test_epub(self, tmp_path):
        path = tmp_path / "book.bin"
        _make_epub(path)
        assert detect_file_type_by_content(str(path)) == "epub"

    def test_docx(self, tmp_path):
        path = tmp_path / "document.bin"
        _make_docx(path)
        assert detect_file_type_by_content(str(path)) == "docx"

    def test_txt(self, tmp_path):
        path = tmp_path / "notes.bin"
        path.write_text("This is a plain text file with enough readable content.\n", encoding="utf-8")
        assert detect_file_type_by_content(str(path)) == "txt"

    def test_srt(self, tmp_path):
        path = tmp_path / "subs.bin"
        path.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nWorld\n",
            encoding="utf-8",
        )
        assert detect_file_type_by_content(str(path)) == "srt"

    def test_txt_mentioning_pdf_marker_late_is_still_txt(self, tmp_path):
        # '%PDF-' beyond the first 1024 bytes must not trigger PDF detection
        path = tmp_path / "late.bin"
        path.write_text("word " * 300 + "%PDF- appears here\n", encoding="utf-8")
        assert detect_file_type_by_content(str(path)) == "txt"


class TestPackageImport:
    def test_import_without_pymupdf(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'pymupdf', None)
        monkeypatch.setitem(sys.modules, 'fitz', None)
        for name in [m for m in sys.modules if m == 'src.core.pdf' or m.startswith('src.core.pdf.')]:
            monkeypatch.delitem(sys.modules, name)

        module = importlib.import_module('src.core.pdf')

        assert module.PdfPlainContent is not None
        assert issubclass(module.PdfNoTextLayerError, module.PdfError)
        assert set(module.PARAGRAPH_STYLES) == {"heading1", "heading2", "heading3", "list", "normal", "cell"}
