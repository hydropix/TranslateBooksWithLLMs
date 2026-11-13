"""
Integration tests for RTL support in EPUB translation workflow

Tests the complete flow from RTL detection to CSS injection and OPF updates.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.epub.rtl_support import (
    is_rtl_language,
    apply_rtl_to_epub_directory,
)


class TestRTLIntegration:
    """Integration tests for RTL workflow"""

    def test_complete_rtl_workflow_arabic(self, tmp_path):
        """Test complete RTL workflow for Arabic translation"""
        # Create a minimal EPUB structure
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        oebps_dir = epub_dir / "OEBPS"
        oebps_dir.mkdir()
        
        # Create chapter files
        chapter1 = oebps_dir / "chapter1.xhtml"
        chapter1.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter 1</title>
</head>
<body>
    <h1>Introduction</h1>
    <p>This is a test paragraph.</p>
    <pre><code>&lt;div&gt;Example code&lt;/div&gt;</code></pre>
</body>
</html>""", encoding='utf-8')
        
        chapter2 = oebps_dir / "chapter2.xhtml"
        chapter2.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Chapter 2</title>
</head>
<body>
    <h1>Next Chapter</h1>
    <p>Another paragraph.</p>
</body>
</html>""", encoding='utf-8')
        
        # Create OPF file
        opf = oebps_dir / "content.opf"
        opf.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>Test Book</dc:title>
        <dc:language>en</dc:language>
    </metadata>
    <manifest>
        <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
        <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
    </manifest>
    <spine toc="ncx">
        <itemref idref="chapter1"/>
        <itemref idref="chapter2"/>
    </spine>
</package>""", encoding='utf-8')
        
        # Verify RTL detection
        assert is_rtl_language('Arabic') is True
        
        # Apply RTL
        result = apply_rtl_to_epub_directory(str(epub_dir), 'Arabic')
        
        # Verify results
        assert result['is_rtl'] is True
        assert result['css_injected'] == 2
        assert result['opf_updated'] is True
        
        # Verify chapter 1 CSS
        ch1_content = chapter1.read_text(encoding='utf-8')
        assert 'direction: rtl' in ch1_content
        assert 'dir="rtl"' in ch1_content
        assert 'lang="ar"' in ch1_content or 'lang="ar"' in ch1_content.replace("'", '"')
        
        # Verify technical content is protected (LTR)
        assert 'direction: ltr' in ch1_content  # Code blocks should be LTR
        
        # Verify chapter 2 CSS
        ch2_content = chapter2.read_text(encoding='utf-8')
        assert 'direction: rtl' in ch2_content
        
        # Verify OPF
        opf_content = opf.read_text(encoding='utf-8')
        assert 'page-progression-direction="rtl"' in opf_content

    def test_non_rtl_language_no_changes(self, tmp_path):
        """Test that non-RTL languages don't get RTL treatment"""
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        oebps_dir = epub_dir / "OEBPS"
        oebps_dir.mkdir()
        
        # Create original file
        original_html = """<?xml version="1.0"?>
<html>
<head><title>Test</title></head>
<body><p>Content</p></body>
</html>"""
        chapter = oebps_dir / "chapter.xhtml"
        chapter.write_text(original_html, encoding='utf-8')
        
        opf = oebps_dir / "content.opf"
        opf.write_text("""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <spine></spine>
</package>""", encoding='utf-8')
        
        # Apply for non-RTL language
        result = apply_rtl_to_epub_directory(str(epub_dir), 'French')
        
        # Should not apply RTL
        assert result['is_rtl'] is False
        assert result['css_injected'] == 0
        
        # File should be unchanged
        current_content = chapter.read_text(encoding='utf-8')
        assert current_content == original_html

    def test_hebrew_rtl_workflow(self, tmp_path):
        """Test RTL workflow for Hebrew"""
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        oebps_dir = epub_dir / "OEBPS"
        oebps_dir.mkdir()
        
        chapter = oebps_dir / "page.xhtml"
        chapter.write_text("""<html><head></head><body><p>Text</p></body></html>""", encoding='utf-8')
        
        opf = oebps_dir / "content.opf"
        opf.write_text("""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <spine toc="ncx"></spine>
</package>""", encoding='utf-8')
        
        result = apply_rtl_to_epub_directory(str(epub_dir), 'Hebrew')
        
        assert result['is_rtl'] is True
        assert result['css_injected'] == 1
        
        # Check Hebrew lang code
        content = chapter.read_text(encoding='utf-8')
        assert 'lang="he"' in content or 'lang="iw"' in content

    def test_persian_rtl_workflow(self, tmp_path):
        """Test RTL workflow for Persian/Farsi"""
        epub_dir = tmp_path / "epub"
        epub_dir.mkdir()
        oebps_dir = epub_dir / "OEBPS"
        oebps_dir.mkdir()
        
        chapter = oebps_dir / "page.xhtml"
        chapter.write_text("""<html><head></head><body><p>Text</p></body></html>""", encoding='utf-8')
        
        opf = oebps_dir / "content.opf"
        opf.write_text("""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
    <spine toc="ncx"></spine>
</package>""", encoding='utf-8')
        
        result = apply_rtl_to_epub_directory(str(epub_dir), 'Persian')
        
        assert result['is_rtl'] is True
        assert result['css_injected'] == 1
        
        # Check Persian lang code
        content = chapter.read_text(encoding='utf-8')
        assert 'lang="fa"' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
