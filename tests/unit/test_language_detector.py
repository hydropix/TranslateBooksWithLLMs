"""Regression tests for source-language auto-detection (src/utils/language_detector.py)."""
import io
import zipfile

from docx import Document

from src.utils.language_detector import LanguageDetector

ENGLISH = (
    "The old lighthouse keeper climbed the stairs every evening to light the "
    "lamp, and he watched the ships pass far away on the dark and quiet sea. "
)
FRENCH = (
    "Le vieux gardien du phare montait les escaliers chaque soir pour allumer "
    "la lampe, et il regardait les navires passer au loin sur la mer sombre. "
)


def _docx_bytes(paragraphs, table_rows=()) -> bytes:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    if table_rows:
        table = doc.add_table(rows=len(table_rows), cols=1)
        for index, text in enumerate(table_rows):
            table.cell(index, 0).text = text
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _xhtml(text: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        f"<body><p>{text}</p></body></html>"
    )


def _epub_bytes(chapters_in_zip_order, spine_ids) -> bytes:
    """Build a minimal EPUB whose zip order differs from its spine order."""
    manifest = "".join(
        f'<item id="{item_id}" href="text/{item_id}.xhtml" media-type="application/xhtml+xml"/>'
        for item_id, _ in chapters_in_zip_order
    )
    spine = "".join(f'<itemref idref="{item_id}"/>' for item_id in spine_ids)
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title></metadata>'
        f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>"
    )
    container = (
        '<?xml version="1.0"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        for item_id, text in chapters_in_zip_order:
            archive.writestr(f"OEBPS/text/{item_id}.xhtml", _xhtml(text))
        archive.writestr("OEBPS/content.opf", opf)
    return buffer.getvalue()


def test_docx_language_is_detected_from_document_text():
    data = _docx_bytes([FRENCH * 3], table_rows=[FRENCH])

    language, confidence = LanguageDetector.detect_language_from_file(data, "book.docx")

    assert language == "French"
    assert confidence >= 0.7


def test_docx_text_extraction_reads_paragraphs_and_tables():
    data = _docx_bytes(["First paragraph."], table_rows=["A table cell."])

    text = LanguageDetector._extract_text_from_docx(data)

    assert "First paragraph." in text
    assert "A table cell." in text
    assert "PK" not in text and "word/document.xml" not in text


def test_epub_text_is_read_in_spine_order():
    data = _epub_bytes(
        [("appendix", "Appendix text."), ("ch1", "Chapter one."), ("ch2", "Chapter two.")],
        spine_ids=["ch1", "ch2", "appendix"],
    )

    text = " ".join(LanguageDetector._extract_text_from_epub(data).split())

    assert text == "Chapter one. Chapter two. Appendix text."


def test_epub_sample_comes_from_the_start_of_the_spine():
    # The zip-first file is a huge French appendix; the book itself (spine
    # first) is English. Zip order would fill the whole sample with French.
    data = _epub_bytes(
        [("appendix", FRENCH * 200), ("ch1", ENGLISH * 200)],
        spine_ids=["ch1", "appendix"],
    )

    text = LanguageDetector._extract_text_from_epub(data)

    assert text.lstrip().startswith(ENGLISH.strip()[:40])


def test_detection_is_deterministic():
    # Short, mixed input: unseeded langdetect returns different probabilities
    # from one call to the next.
    sample = "Bonjour hello ciao hola guten Tag the le la der die das el " * 2
    results = {LanguageDetector.detect_language_from_text(sample, 0.0) for _ in range(15)}

    assert len(results) == 1
