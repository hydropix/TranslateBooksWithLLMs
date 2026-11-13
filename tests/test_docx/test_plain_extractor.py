"""Regression tests for the DOCX Plain Text Mode extractor and rebuilder."""
import io
import struct
import zlib

from docx import Document
from docx.shared import Emu

from src.core.docx.plain_extractor import extract_plain_paragraphs


def _png_bytes(width: int = 2, height: int = 2) -> bytes:
    """Return a tiny valid RGB PNG built in memory."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_each_image_keeps_its_own_size(tmp_path):
    source = tmp_path / "images.docx"
    doc = Document()
    paragraph = doc.add_paragraph("Two pictures:")
    run = paragraph.add_run()
    run.add_picture(io.BytesIO(_png_bytes()), width=Emu(914400), height=Emu(914400))
    run = paragraph.add_run()
    run.add_picture(io.BytesIO(_png_bytes()), width=Emu(1828800), height=Emu(457200))
    doc.save(source)

    content = extract_plain_paragraphs(str(source))

    images = content.images_by_paragraph[0]
    assert [(i.width_emu, i.height_emu) for i in images] == [
        (914400, 914400),
        (1828800, 457200),
    ]
