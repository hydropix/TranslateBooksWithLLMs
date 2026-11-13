"""Regression tests for source-language auto-detection (src/utils/language_detector.py)."""
import io

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


def test_detection_is_deterministic():
    # Short, mixed input: unseeded langdetect returns different probabilities
    # from one call to the next.
    sample = "Bonjour hello ciao hola guten Tag the le la der die das el " * 2
    results = {LanguageDetector.detect_language_from_text(sample, 0.0) for _ in range(15)}

    assert len(results) == 1
