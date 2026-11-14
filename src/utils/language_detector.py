"""
Language detection utility for automatic source language identification

Uses langdetect library (based on Google's language-detection library)
for fast and accurate language identification from text content.
"""
import posixpath
import re
from typing import List, Optional, Tuple
from urllib.parse import unquote

from langdetect import DetectorFactory, detect_langs, LangDetectException
from lxml import etree

# langdetect is randomised: without a fixed seed the same text can yield a
# different language or confidence on every call.
DetectorFactory.seed = 0

_OPF_NS = "http://www.idpf.org/2007/opf"
_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"


# Mapping from ISO 639-1 codes (langdetect output) to full language names
LANGUAGE_CODE_MAP = {
    'en': 'English',
    'zh-cn': 'Chinese',
    'zh-tw': 'Chinese',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ru': 'Russian',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'fi': 'Finnish',
    'cs': 'Czech',
    'el': 'Greek',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'uk': 'Ukrainian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
}


class LanguageDetector:
    """Detects language from file content"""

    # Minimum text length for reliable detection (characters)
    MIN_TEXT_LENGTH = 50

    # Maximum text to analyze (to avoid performance issues with large files)
    MAX_SAMPLE_LENGTH = 10000

    @staticmethod
    def _epub_spine_order(epub_zip) -> List[str]:
        """
        Return the content documents of an EPUB in reading (spine) order.

        Follows META-INF/container.xml to the OPF package document and resolves
        its spine against the manifest. Returns [] when any step is missing or
        malformed, so the caller can fall back to the archive order.
        """
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
        try:
            container = etree.fromstring(epub_zip.read("META-INF/container.xml"), parser)
            rootfile = container.find(f".//{{{_CONTAINER_NS}}}rootfile")
            opf_path = rootfile.get("full-path") if rootfile is not None else None
            if not opf_path:
                return []
            opf = etree.fromstring(epub_zip.read(opf_path), parser)
        except Exception:
            return []
        if opf is None:
            return []

        opf_dir = posixpath.dirname(opf_path)
        hrefs = {
            item.get("id"): item.get("href")
            for item in opf.iter(f"{{{_OPF_NS}}}item")
            if item.get("id") and item.get("href")
        }
        names = set(epub_zip.namelist())
        ordered = []
        for itemref in opf.iter(f"{{{_OPF_NS}}}itemref"):
            href = hrefs.get(itemref.get("idref"))
            if not href:
                continue
            path = posixpath.normpath(posixpath.join(opf_dir, unquote(href.split("#")[0])))
            if path in names and path not in ordered:
                ordered.append(path)
        return ordered

    @staticmethod
    def _extract_text_from_epub(file_data: bytes) -> str:
        """
        Extract readable text from EPUB file, in spine (reading) order

        Args:
            file_data: Raw EPUB file bytes

        Returns:
            Extracted text content
        """
        import zipfile
        import io

        try:
            with zipfile.ZipFile(io.BytesIO(file_data)) as epub_zip:
                text_parts = []

                # Reading order from the OPF spine; archive order only as a
                # fallback for a package without a usable spine.
                file_names = LanguageDetector._epub_spine_order(epub_zip) or epub_zip.namelist()
                for file_name in file_names:
                    if file_name.endswith(('.xhtml', '.html', '.htm')):
                        try:
                            content = epub_zip.read(file_name)
                            # Parse HTML and extract text
                            parser = etree.HTMLParser()
                            tree = etree.fromstring(content, parser)

                            # Extract text from body only (ignore metadata)
                            body = tree.find('.//body')
                            if body is not None:
                                text = etree.tostring(body, method='text', encoding='unicode')
                                text_parts.append(text)

                            # Stop if we have enough text
                            combined = ' '.join(text_parts)
                            if len(combined) >= LanguageDetector.MAX_SAMPLE_LENGTH:
                                break

                        except Exception:
                            continue

                return ' '.join(text_parts)

        except Exception:
            return ""

    @staticmethod
    def _extract_text_from_docx(file_data: bytes) -> str:
        """
        Extract readable text from a DOCX file (body paragraphs and tables)

        Args:
            file_data: Raw DOCX file bytes

        Returns:
            Extracted text content, or "" if the document cannot be read
        """
        import io

        try:
            from docx import Document
            from docx.oxml.ns import qn

            document = Document(io.BytesIO(file_data))
            text_parts = []
            length = 0
            # Every <w:p> of the body in document order, table cells included.
            for paragraph in document.element.body.iter(qn("w:p")):
                text = "".join(node.text or "" for node in paragraph.iter(qn("w:t")))
                if text.strip():
                    text_parts.append(text)
                    length += len(text) + 1
                    if length >= LanguageDetector.MAX_SAMPLE_LENGTH * 3:
                        break
            return " ".join(text_parts)
        except Exception:
            return ""

    @staticmethod
    def _extract_text_from_srt(file_data: bytes) -> str:
        """
        Extract readable text from SRT subtitle file

        Args:
            file_data: Raw SRT file bytes

        Returns:
            Extracted subtitle text
        """
        try:
            # Try common encodings
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    text = file_data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return ""

            # Remove subtitle numbers and timestamps
            # SRT format: number -> timestamp -> text -> blank line
            lines = text.split('\n')
            text_lines = []

            for line in lines:
                line = line.strip()
                # Skip empty lines, numbers, and timestamps
                if not line:
                    continue
                if line.isdigit():
                    continue
                if '-->' in line:
                    continue
                # This is actual subtitle text
                text_lines.append(line)

            return ' '.join(text_lines)

        except Exception:
            return ""

    @staticmethod
    def _extract_text_from_pdf(file_data: bytes) -> str:
        """
        Extract readable text from PDF file

        Args:
            file_data: Raw PDF file bytes

        Returns:
            Extracted text content, or "" if the PDF has no usable text layer
            (the package wrapper never raises, even without PyMuPDF installed)
        """
        try:
            from src.core.pdf import extract_pdf_text

            return extract_pdf_text(file_data, hard_cap=20000)
        except Exception:
            return ""

    @staticmethod
    def _clean_text_for_detection(text: str) -> str:
        """
        Clean text to improve detection accuracy

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common markup artifacts
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        return text.strip()

    @staticmethod
    def detect_language_from_file(
        file_data: bytes,
        filename: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[Optional[str], float]:
        """
        Detect language from file content

        Args:
            file_data: Raw file bytes
            filename: Original filename (used to determine file type)
            confidence_threshold: Minimum confidence level (0.0-1.0)

        Returns:
            Tuple of (language_name, confidence) or (None, 0.0) if detection fails
        """
        try:
            # Extract text based on file type
            filename_lower = filename.lower()

            if filename_lower.endswith('.epub'):
                text = LanguageDetector._extract_text_from_epub(file_data)
            elif filename_lower.endswith('.srt'):
                text = LanguageDetector._extract_text_from_srt(file_data)
            elif filename_lower.endswith('.pdf'):
                text = LanguageDetector._extract_text_from_pdf(file_data)
            elif filename_lower.endswith('.docx'):
                text = LanguageDetector._extract_text_from_docx(file_data)
            else:
                # Plain text file
                # Try common encodings
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        text = file_data.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return None, 0.0

            # Clean and prepare text
            text = LanguageDetector._clean_text_for_detection(text)

            # Check if we have enough text
            if len(text) < LanguageDetector.MIN_TEXT_LENGTH:
                return None, 0.0

            # Sample text if too long (for performance)
            if len(text) > LanguageDetector.MAX_SAMPLE_LENGTH:
                # Take samples from beginning, middle, and end
                chunk_size = LanguageDetector.MAX_SAMPLE_LENGTH // 3
                text = (
                    text[:chunk_size] + ' ' +
                    text[len(text)//2 - chunk_size//2:len(text)//2 + chunk_size//2] + ' ' +
                    text[-chunk_size:]
                )

            # Detect language with confidence scores
            detected_langs = detect_langs(text)

            if not detected_langs:
                return None, 0.0

            # Get most probable language
            best_match = detected_langs[0]
            lang_code = best_match.lang
            confidence = best_match.prob

            # Map to full language name
            language_name = LANGUAGE_CODE_MAP.get(lang_code)

            # Return result only if confidence is high enough
            if language_name and confidence >= confidence_threshold:
                return language_name, confidence

            return None, 0.0

        except LangDetectException:
            # Language detection library couldn't determine language
            return None, 0.0
        except Exception as e:
            # Log error but don't fail
            print(f"Language detection error: {str(e)}")
            return None, 0.0

    @staticmethod
    def detect_language_from_text(
        text: str,
        confidence_threshold: float = 0.7
    ) -> Tuple[Optional[str], float]:
        """
        Detect language from plain text

        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence level (0.0-1.0)

        Returns:
            Tuple of (language_name, confidence) or (None, 0.0) if detection fails
        """
        try:
            # Clean text
            text = LanguageDetector._clean_text_for_detection(text)

            # Check if we have enough text
            if len(text) < LanguageDetector.MIN_TEXT_LENGTH:
                return None, 0.0

            # Detect with confidence
            detected_langs = detect_langs(text)

            if not detected_langs:
                return None, 0.0

            best_match = detected_langs[0]
            lang_code = best_match.lang
            confidence = best_match.prob

            language_name = LANGUAGE_CODE_MAP.get(lang_code)

            if language_name and confidence >= confidence_threshold:
                return language_name, confidence

            return None, 0.0

        except LangDetectException:
            return None, 0.0
        except Exception:
            return None, 0.0
