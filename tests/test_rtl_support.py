"""
Tests for RTL (Right-to-Left) support module

Tests RTL language detection, CSS generation, and OPF updates.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.epub.rtl_support import (
    is_rtl_language,
    get_language_code,
    generate_rtl_css,
    inject_rtl_css_to_html,
    update_opf_for_rtl,
    apply_rtl_to_epub_directory,
)


class TestRTLLanguageDetection:
    """Test RTL language detection"""

    def test_arabic_is_rtl(self):
        """Arabic should be detected as RTL"""
        assert is_rtl_language('Arabic') is True
        assert is_rtl_language('arabic') is True
        assert is_rtl_language('ARABIC') is True
        assert is_rtl_language('ar') is True
        assert is_rtl_language('ar-SA') is True
        assert is_rtl_language('ar-EG') is True

    def test_hebrew_is_rtl(self):
        """Hebrew should be detected as RTL"""
        assert is_rtl_language('Hebrew') is True
        assert is_rtl_language('hebrew') is True
        assert is_rtl_language('he') is True
        assert is_rtl_language('he-IL') is True
        assert is_rtl_language('iw') is True  # Legacy code

    def test_persian_is_rtl(self):
        """Persian/Farsi should be detected as RTL"""
        assert is_rtl_language('Persian') is True
        assert is_rtl_language('persian') is True
        assert is_rtl_language('Farsi') is True
        assert is_rtl_language('fa') is True

    def test_urdu_is_rtl(self):
        """Urdu should be detected as RTL"""
        assert is_rtl_language('Urdu') is True
        assert is_rtl_language('ur') is True

    def test_non_rtl_languages(self):
        """Non-RTL languages should return False"""
        assert is_rtl_language('English') is False
        assert is_rtl_language('French') is False
        assert is_rtl_language('Chinese') is False
        assert is_rtl_language('Spanish') is False
        assert is_rtl_language('German') is False
        assert is_rtl_language('Japanese') is False
        assert is_rtl_language('Russian') is False
        assert is_rtl_language('en') is False
        assert is_rtl_language('fr') is False
        assert is_rtl_language('zh') is False

    def test_empty_and_none(self):
        """Empty and None should return False"""
        assert is_rtl_language('') is False
        assert is_rtl_language(None) is False


class TestLanguageCodeMapping:
    """Test language code mapping"""

    def test_get_code_for_rtl_languages(self):
        """Should return correct codes for RTL languages"""
        assert get_language_code('Arabic') == 'ar'
        assert get_language_code('arabic') == 'ar'
        assert get_language_code('Hebrew') == 'he'
        assert get_language_code('Persian') == 'fa'
        assert get_language_code('Urdu') == 'ur'

    def test_get_code_with_locale(self):
        """Should extract base code from locale"""
        assert get_language_code('ar-SA') == 'ar'
        assert get_language_code('he-IL') == 'he'
        assert get_language_code('fa-IR') == 'fa'


class TestRTLCSSGeneration:
    """Test RTL CSS generation"""

    def test_css_contains_rtl_rules(self):
        """Generated CSS should contain RTL-specific rules"""
        css = generate_rtl_css('Arabic')
        
        # Check for key RTL properties
        assert 'direction: rtl' in css
        assert 'unicode-bidi: isolate' in css
        assert 'text-align: right' in css

    def test_css_contains_ltr_protection(self):
        """CSS should protect technical content with LTR"""
        css = generate_rtl_css('Arabic')
        
        # Check for code protection
        assert 'direction: ltr' in css
        assert 'unicode-bidi: embed' in css

    def test_css_contains_pre_styling(self):
        """CSS should style pre/code blocks"""
        css = generate_rtl_css('Arabic')
        
        assert 'pre' in css
        assert 'code' in css
        assert 'font-family: monospace' in css

    def test_css_language_comment(self):
        """CSS should include language identifier"""
        css = generate_rtl_css('Arabic')
        assert 'ar' in css  # Language code should be in CSS


class TestHTMLInjection:
    """Test HTML CSS injection"""

    def test_inject_into_html_with_head(self):
        """Should inject CSS into existing head"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Test</title>
</head>
<body>
    <p>Hello</p>
</body>
</html>"""
        
        result = inject_rtl_css_to_html(html, 'Arabic')
        
        # Check CSS was injected
        assert '<style' in result
        assert 'direction: rtl' in result
        assert 'rtl !important' in result
        
        # Check dir attribute added
        assert 'dir="rtl"' in result

    def test_inject_into_html_without_head(self):
        """Should create head and inject CSS"""
        html = """<!DOCTYPE html>
<html>
<body>
    <p>Hello</p>
</body>
</html>"""
        
        result = inject_rtl_css_to_html(html, 'Arabic')
        
        # Check head was created
        assert '<head>' in result
        assert '<style' in result

    def test_preserves_existing_content(self):
        """Should preserve existing HTML content"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Test Book</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Chapter 1</h1>
    <p>This is a paragraph.</p>
</body>
</html>"""
        
        result = inject_rtl_css_to_html(html, 'Arabic')
        
        # Check original content preserved
        assert '<title>Test Book</title>' in result
        assert '<h1>Chapter 1</h1>' in result
        assert 'This is a paragraph.' in result
        assert 'style.css' in result


class TestOPFUpdate:
    """Test OPF file updates"""

    def test_update_opf_spine(self, tmp_path):
        """Should add page-progression-direction to spine"""
        opf_content = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
    <metadata>
        <dc:title>Test Book</dc:title>
        <dc:language>en</dc:language>
    </metadata>
    <manifest>
        <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    </manifest>
    <spine toc="ncx">
        <itemref idref="chapter1"/>
    </spine>
</package>"""
        
        opf_path = tmp_path / "content.opf"
        opf_path.write_text(opf_content, encoding='utf-8')
        
        result = update_opf_for_rtl(str(opf_path), 'Arabic')
        
        assert result is True
        
        # Check content was updated
        updated_content = opf_path.read_text(encoding='utf-8')
        assert 'page-progression-direction="rtl"' in updated_content

    def test_update_opf_without_spine(self, tmp_path):
        """Should return False if no spine found"""
        opf_content = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <metadata>
        <dc:title>Test Book</dc:title>
    </metadata>
</package>"""
        
        opf_path = tmp_path / "content.opf"
        opf_path.write_text(opf_content, encoding='utf-8')
        
        result = update_opf_for_rtl(str(opf_path), 'Arabic')
        
        assert result is False


class TestApplyRTLToEPUBDirectory:
    """Test full EPUB directory processing"""

    def test_apply_rtl_to_directory_arabic(self, tmp_path):
        """Should apply RTL to all HTML files for Arabic"""
        # Create mock EPUB structure
        (tmp_path / "OEBPS").mkdir()
        
        # Create HTML files
        html1 = tmp_path / "OEBPS" / "chapter1.xhtml"
        html1.write_text("""<?xml version="1.0"?>
<html><head><title>Ch1</title></head>
<body><p>Chapter 1</p></body></html>""", encoding='utf-8')
        
        html2 = tmp_path / "OEBPS" / "chapter2.xhtml"
        html2.write_text("""<?xml version="1.0"?>
<html><head><title>Ch2</title></head>
<body><p>Chapter 2</p></body></html>""", encoding='utf-8')
        
        # Create OPF
        opf = tmp_path / "OEBPS" / "content.opf"
        opf.write_text("""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <metadata></metadata>
    <manifest></manifest>
    <spine toc="ncx"></spine>
</package>""", encoding='utf-8')
        
        result = apply_rtl_to_epub_directory(str(tmp_path), 'Arabic')
        
        assert result['is_rtl'] is True
        assert result['css_injected'] == 2
        assert result['opf_updated'] is True
        
        # Check CSS was injected
        updated_html1 = html1.read_text(encoding='utf-8')
        assert 'direction: rtl' in updated_html1
        assert 'rtl !important' in updated_html1

    def test_apply_rtl_skips_non_rtl(self, tmp_path):
        """Should skip processing for non-RTL languages"""
        result = apply_rtl_to_epub_directory(str(tmp_path), 'English')
        
        assert result['is_rtl'] is False
        assert result['css_injected'] == 0
        assert result['opf_updated'] is False

    def test_apply_rtl_to_hebrew(self, tmp_path):
        """Should apply RTL for Hebrew"""
        (tmp_path / "OEBPS").mkdir()
        
        html = tmp_path / "OEBPS" / "page.xhtml"
        html.write_text("""<html><body><p>Text</p></body></html>""", encoding='utf-8')
        
        opf = tmp_path / "OEBPS" / "content.opf"
        opf.write_text("""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
    <metadata><dc:title>Test</dc:title></metadata>
    <spine></spine>
</package>""", encoding='utf-8')
        
        result = apply_rtl_to_epub_directory(str(tmp_path), 'Hebrew')
        
        assert result['is_rtl'] is True
        assert result['css_injected'] == 1


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_whitespace_in_language_name(self):
        """Should handle whitespace in language names"""
        assert is_rtl_language('  Arabic  ') is True
        assert is_rtl_language('  ar  ') is True

    def test_mixed_case(self):
        """Should handle mixed case"""
        assert is_rtl_language('ArAbIc') is True
        assert is_rtl_language('HeBrEw') is True

    def test_invalid_html_content(self):
        """Should handle invalid HTML gracefully"""
        html = "This is not valid HTML < unclosed tag"
        result = inject_rtl_css_to_html(html, 'Arabic')
        # Should not raise exception
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
