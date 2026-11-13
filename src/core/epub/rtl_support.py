"""
RTL (Right-to-Left) Support Module for EPUB Translation

Provides CSS injection and OPF metadata updates for RTL languages
such as Arabic, Hebrew, Persian, Urdu, etc.

This module ensures proper text direction and layout for RTL languages
while protecting technical content (code, URLs) from direction reversal.
"""

from typing import Set, Optional
import os


# RTL Languages supported with their ISO 639-1 codes
RTL_LANGUAGES: Set[str] = {
    'ar',   # Arabic
    'he',   # Hebrew
    'iw',   # Hebrew (legacy code)
    'fa',   # Persian/Farsi
    'ur',   # Urdu
    'ps',   # Pashto
    'sd',   # Sindhi
    'yi',   # Yiddish
    'ug',   # Uyghur
    'ku',   # Kurdish
}

# Language name to code mapping for detection
LANGUAGE_NAME_TO_CODE = {
    'arabic': 'ar',
    'hebrew': 'he',
    'persian': 'fa',
    'farsi': 'fa',
    'urdu': 'ur',
    'pashto': 'ps',
    'sindhi': 'sd',
    'yiddish': 'yi',
    'uyghur': 'ug',
    'kurdish': 'ku',
}

# CSS template for RTL support
# Based on best practices for mixed RTL/LTR content
RTL_CSS_TEMPLATE = """/* RTL Support - Auto-generated for {language} */

/* Base RTL direction for document */
html, body, .calibre, .body {{
    direction: rtl !important;
    unicode-bidi: isolate !important;
    text-align: right !important;
}}

/* Isolated text elements to prevent spillover */
p, span, div, li, h1, h2, h3, h4, h5, h6 {{
    unicode-bidi: isolate !important;
}}

/* Technical content - force LTR to prevent reversal */
.programlisting,
.fm-code-in-text,
.fm-code-in-text1,
.fm-code-in-text2,
code, pre, tt,
.fm-code-continuation-arrow,
.fm-code-continuation-arrow1,
samp, kbd, var {{
    direction: ltr !important;
    unicode-bidi: embed !important;
    display: inline-block;
    text-align: left !important;
}}

/* URLs and paths - always LTR */
.url, .url1, .filepath, .filename {{
    color: #4080BF;
    text-decoration: none;
    direction: ltr !important;
    display: inline-block;
}}

/* Block-level code */
pre, .programlisting {{
    background-color: #f9f9f9;
    font-family: monospace;
    font-size: 0.83333em;
    white-space: pre-wrap;
    margin: 1em 0;
    padding: 10px 5px;
    border-radius: 4px;
    direction: ltr !important;
    text-align: left !important;
}}

/* Tables - RTL direction */
.contenttable, .fm-contenttable3, table {{
    border-collapse: collapse;
    border-spacing: 2px;
    display: table;
    line-height: 1.4;
    margin-bottom: 5px;
    margin-top: 0;
    width: 100%;
    direction: rtl;
}}

/* Table cells */
.fm-contenttable, .fm-contenttable1, .fm-contenttable2, td, th {{
    display: table-cell;
    vertical-align: top;
    padding: 2px 5px;
    border: black solid 2px;
    text-align: right;
}}

/* Headings - RTL alignment */
.co-summary-head, .fm-head, .fm-head1, .fm-head2, .tochead,
h1, h2, h3, h4, h5, h6 {{
    color: #005;
    display: block;
    font-weight: bold;
    line-height: 1.4;
    text-align: right;
}}

/* Copyright sections - usually LTR */
.copyright, .copyrighta, .copyrightb, .copyrightbody, .copyrightbody1 {{
    text-align: center;
    direction: ltr;
}}

/* Body text styles */
.body {{
    color: black;
    display: block;
    font-family: "Verdana", "Arial", "sans-serif";
    font-size: 1em;
    line-height: 1.4;
    margin: 1em 0;
}}

/* Calibre compatibility */
.calibre {{
    display: block;
    font-size: 1em;
    line-height: 1.4;
    padding-left: 0;
    padding-right: 0;
    margin: 0 5pt;
}}

/* Lists - RTL alignment */
ul, ol {{
    direction: rtl;
    padding-right: 2em;
    padding-left: 0;
}}

li {{
    text-align: right;
}}

/* Blockquotes */
blockquote {{
    direction: rtl;
    border-right: 3px solid #ccc;
    border-left: none;
    margin: 1em 0;
    padding-right: 1em;
    padding-left: 0;
}}
"""


def is_rtl_language(language: str) -> bool:
    """
    Check if a language requires RTL (Right-to-Left) layout.
    
    Args:
        language: Language name (e.g., 'Arabic', 'Hebrew') or code (e.g., 'ar', 'he')
    
    Returns:
        True if the language is RTL, False otherwise
    """
    if not language:
        return False
    
    lang_lower = language.lower().strip()
    
    # Direct code match (e.g., 'ar', 'he')
    if lang_lower in RTL_LANGUAGES:
        return True
    
    # Name match (e.g., 'arabic', 'hebrew')
    if lang_lower in LANGUAGE_NAME_TO_CODE:
        return True
    
    # Handle locale codes like 'ar-SA', 'he-IL'
    base_code = lang_lower.split('-')[0]
    if base_code in RTL_LANGUAGES:
        return True
    
    return False


def get_language_code(language: str) -> Optional[str]:
    """
    Get ISO 639-1 code for a language name or code.
    
    Args:
        language: Language name or code
    
    Returns:
        ISO 639-1 code or None if not found
    """
    if not language:
        return None
    
    lang_lower = language.lower().strip()
    
    # Already a code
    if lang_lower in RTL_LANGUAGES:
        return lang_lower
    
    # Name to code
    if lang_lower in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[lang_lower]
    
    # Handle locale codes
    base_code = lang_lower.split('-')[0]
    if base_code in RTL_LANGUAGES:
        return base_code
    
    return None


def generate_rtl_css(language: str) -> str:
    """
    Generate RTL CSS for a specific language.
    
    Args:
        language: Target language name or code
    
    Returns:
        CSS string for RTL support
    """
    code = get_language_code(language) or language
    return RTL_CSS_TEMPLATE.format(language=code)


def inject_rtl_css_to_html(html_content: str, language: str) -> str:
    """
    Inject RTL CSS into HTML content.
    
    Args:
        html_content: Original HTML content
        language: Target language for CSS comment
    
    Returns:
        Modified HTML with RTL CSS in head
    """
    from lxml import etree
    
    css_content = generate_rtl_css(language)
    style_tag = f"<style type=\"text/css\">\n{css_content}\n</style>"
    lang_code = get_language_code(language) or 'ar'
    
    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_content.encode('utf-8'), parser)
        
        # Find or create head
        head = tree.find('.//head')
        if head is None:
            # Create head element
            head = etree.Element('head')
            # Insert before body if exists
            body = tree.find('.//body')
            if body is not None:
                body.addprevious(head)
            else:
                tree.insert(0, head)
        
        # Create style element
        style_elem = etree.Element('style')
        style_elem.set('type', 'text/css')
        style_elem.text = f"\n{css_content}\n"
        
        # Add to head
        head.append(style_elem)
        
        # Add dir="rtl" and lang to html element
        # lxml's HTMLParser puts everything under a root <html> element
        root = tree
        if root is not None:
            root.set('dir', 'rtl')
            root.set('lang', lang_code)
        
        # Convert back to string
        result = etree.tostring(tree, encoding='unicode', method='html')
        
        # Ensure DOCTYPE is preserved if present
        if '<!DOCTYPE' in html_content.upper() or html_content.strip().startswith('<?xml'):
            # Add DOCTYPE back if it was stripped
            if '<!DOCTYPE' not in result.upper():
                result = '<!DOCTYPE html>\n' + result
        
        return result
    
    except Exception:
        # If parsing fails, inject CSS manually
        if '<head>' in html_content:
            result = html_content.replace('<head>', f'<head>\n{style_tag}\n')
        elif '<html' in html_content.lower():
            result = html_content.replace('<html', f'<head>\n{style_tag}\n</head>\n<html')
        else:
            result = f"<head>\n{style_tag}\n</head>\n{html_content}"
        
        # Add dir and lang to html tag
        if '<html' in result.lower():
            result = result.replace('<html', f'<html dir="rtl" lang="{lang_code}"')
        
        return result


def update_opf_for_rtl(opf_path: str, language: str) -> bool:
    """
    Update content.opf file for RTL language support.
    
    Sets page-progression-direction="rtl" on the spine element.
    
    Args:
        opf_path: Path to content.opf file
        language: Target language
    
    Returns:
        True if successful, False otherwise
    """
    from lxml import etree
    
    try:
        # Parse OPF file with recovery mode for malformed XML
        parser = etree.XMLParser(recover=True, remove_blank_text=False)
        tree = etree.parse(opf_path, parser)
        root = tree.getroot()
        
        # Find spine element - try multiple approaches
        spine = None
        
        # Try with common namespaces
        namespaces = {
            'opf': 'http://www.idpf.org/2007/opf',
            'dc': 'http://purl.org/dc/elements/1.1/'
        }
        
        for ns_prefix, ns_uri in namespaces.items():
            if spine is None:
                spine = root.find(f'.//{{{ns_uri}}}spine')
        
        # Try with opf prefix directly
        if spine is None:
            spine = root.find('.//opf:spine', namespaces=namespaces)
        
        # Try without namespace
        if spine is None:
            spine = root.find('.//spine')
        
        # Search all elements for spine tag
        if spine is None:
            for elem in root.iter():
                if elem.tag.endswith('spine') or elem.tag == 'spine':
                    spine = elem
                    break
        
        if spine is not None:
            # Set page-progression-direction
            spine.set('page-progression-direction', 'rtl')
            
            # Write back
            tree.write(opf_path, encoding='utf-8', xml_declaration=True, pretty_print=True)
            return True
        
        return False
    
    except Exception as e:
        print(f"Warning: Could not update OPF for RTL: {e}")
        return False


def remove_rtl_from_html(html_content: str) -> str:
    """
    Remove RTL styles and reset to LTR layout.
    
    This is used when translating FROM an RTL language TO an LTR language
    (e.g., Arabic -> French). It removes existing RTL CSS and resets direction.
    
    Args:
        html_content: HTML content with potential RTL styles
    
    Returns:
        Modified HTML with LTR layout
    """
    from lxml import etree
    import re
    
    ltr_css = """/* LTR Reset - Auto-generated */
html, body { direction: ltr !important; text-align: left !important; }
.calibre, .body { direction: ltr !important; text-align: left !important; }
h1, h2, h3, h4, h5, h6, p, div { text-align: left !important; }
ul, ol { direction: ltr; padding-left: 2em; padding-right: 0; }
table { direction: ltr; }
td, th { text-align: left; }
blockquote { border-left: 3px solid #ccc; border-right: none; margin: 1em 0; padding-left: 1em; padding-right: 0; }
"""
    
    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_content.encode('utf-8'), parser)
        
        # Find head
        head = tree.find('.//head')
        
        # Remove existing RTL style elements (identified by comment or content)
        if head is not None:
            for style in head.findall('.//style'):
                style_text = style.text or ''
                # Check if it's an RTL style (contains our marker or RTL-specific rules)
                if ('RTL Support' in style_text or 
                    'direction: rtl' in style_text or
                    'text-align: right' in style_text):
                    head.remove(style)
        
        # Reset html attributes to LTR
        root = tree
        if root is not None:
            # Remove dir attribute if set to rtl, or set to ltr
            current_dir = root.get('dir', '').lower()
            if current_dir == 'rtl':
                root.set('dir', 'ltr')
            # Update lang to indicate western/left-to-right context
            # (lang will be updated by translation anyway)
        
        # Add LTR reset CSS
        if head is not None:
            style_elem = etree.Element('style')
            style_elem.set('type', 'text/css')
            style_elem.text = f"\n{ltr_css}\n"
            head.insert(0, style_elem)  # Insert at beginning
        
        # Convert back to string
        result = etree.tostring(tree, encoding='unicode', method='html')
        
        # Preserve DOCTYPE
        if '<!DOCTYPE' in html_content.upper() or html_content.strip().startswith('<?xml'):
            if '<!DOCTYPE' not in result.upper():
                result = '<!DOCTYPE html>\n' + result
        
        return result
    
    except Exception:
        # Fallback: use regex to remove RTL styles
        # Remove style tags containing RTL
        pattern = r'<style[^>]*>[^<]*(?:direction:\s*rtl|RTL Support)[^<]*</style>'
        result = re.sub(pattern, '', html_content, flags=re.IGNORECASE | re.DOTALL)
        
        # Replace dir="rtl" with dir="ltr"
        result = re.sub(r'dir\s*=\s*["\']rtl["\']', 'dir="ltr"', result, flags=re.IGNORECASE)
        
        # Add LTR CSS if head exists
        if '<head>' in result:
            result = result.replace('<head>', f'<head>\n<style type="text/css">\n{ltr_css}\n</style>\n')
        
        return result


def update_opf_for_ltr(opf_path: str) -> bool:
    """
    Update content.opf file for LTR language (remove RTL progression).
    
    Removes or sets page-progression-direction="ltr" on the spine element.
    
    Args:
        opf_path: Path to content.opf file
    
    Returns:
        True if successful, False otherwise
    """
    from lxml import etree
    
    try:
        parser = etree.XMLParser(recover=True, remove_blank_text=False)
        tree = etree.parse(opf_path, parser)
        root = tree.getroot()
        
        # Find spine element
        spine = None
        namespaces = {
            'opf': 'http://www.idpf.org/2007/opf',
            'dc': 'http://purl.org/dc/elements/1.1/'
        }
        
        for ns_prefix, ns_uri in namespaces.items():
            if spine is None:
                spine = root.find(f'.//{{{ns_uri}}}spine')
        
        if spine is None:
            spine = root.find('.//opf:spine', namespaces=namespaces)
        if spine is None:
            spine = root.find('.//spine')
        
        if spine is not None:
            # Check current direction
            current_dir = spine.get('page-progression-direction', '').lower()
            
            if current_dir == 'rtl':
                # Change to LTR or remove attribute (default is LTR)
                spine.set('page-progression-direction', 'ltr')
                
                # Write back
                tree.write(opf_path, encoding='utf-8', xml_declaration=True, pretty_print=True)
                return True
        
        return False
    
    except Exception as e:
        print(f"Warning: Could not update OPF for LTR: {e}")
        return False


def apply_rtl_to_epub_directory(temp_dir: str, target_language: str, source_language: str = None) -> dict:
    """
    Apply RTL or LTR support to all files in an extracted EPUB directory.
    
    Automatically detects the direction change:
    - If target is RTL → Inject RTL CSS
    - If target is LTR but source was RTL → Reset to LTR (remove RTL styles)
    - If both are LTR → No changes needed
    - If both are RTL → Just update metadata
    
    Args:
        temp_dir: Path to extracted EPUB directory
        target_language: Target language name or code
        source_language: Source language name or code (optional, for detecting RTL->LTR transitions)
    
    Returns:
        Dictionary with results:
        {
            'css_injected': int,      # Number of files with CSS injected (RTL)
            'css_removed': int,       # Number of files with RTL CSS removed (LTR reset)
            'opf_updated': bool,      # Whether OPF was updated
            'is_rtl': bool,           # Whether target language is RTL
            'was_transition': bool    # Whether this was an RTL->LTR transition
        }
    """
    result = {
        'css_injected': 0,
        'css_removed': 0,
        'opf_updated': False,
        'is_rtl': False,
        'was_transition': False
    }
    
    target_is_rtl = is_rtl_language(target_language)
    source_is_rtl = is_rtl_language(source_language) if source_language else False
    
    result['is_rtl'] = target_is_rtl
    result['was_transition'] = source_is_rtl and not target_is_rtl
    
    # Case 1: RTL -> LTR transition (e.g., Arabic -> French)
    # Need to remove RTL styles and reset to LTR
    if source_is_rtl and not target_is_rtl:
        return _apply_ltr_reset(temp_dir, result)
    
    # Case 2: LTR -> RTL transition (e.g., French -> Arabic)
    # Need to inject RTL styles
    elif target_is_rtl and not source_is_rtl:
        return _apply_rtl_styles(temp_dir, target_language, result)
    
    # Case 3: Both same direction - no layout changes needed
    # But if target is RTL, still apply/update RTL styles
    elif target_is_rtl:
        return _apply_rtl_styles(temp_dir, target_language, result)
    
    # Case 4: Both LTR - no changes needed
    return result


def _apply_rtl_styles(temp_dir: str, target_language: str, result: dict) -> dict:
    """Apply RTL styles to all HTML files."""
    result['is_rtl'] = True
    
    # Find and update OPF file
    opf_path = None
    for root_dir, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.opf'):
                opf_path = os.path.join(root_dir, file)
                break
        if opf_path:
            break
    
    if opf_path:
        result['opf_updated'] = update_opf_for_rtl(opf_path, target_language)
    
    # Inject CSS into all XHTML/HTML files
    for root_dir, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(('.xhtml', '.html', '.htm')):
                file_path = os.path.join(root_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Inject RTL CSS
                    modified_content = inject_rtl_css_to_html(content, target_language)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    result['css_injected'] += 1
                
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
    
    return result


def _apply_ltr_reset(temp_dir: str, result: dict) -> dict:
    """Remove RTL styles and reset to LTR for all HTML files."""
    result['was_transition'] = True
    
    # Find and update OPF file
    opf_path = None
    for root_dir, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.opf'):
                opf_path = os.path.join(root_dir, file)
                break
        if opf_path:
            break
    
    if opf_path:
        result['opf_updated'] = update_opf_for_ltr(opf_path)
    
    # Remove RTL CSS from all XHTML/HTML files
    for root_dir, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(('.xhtml', '.html', '.htm')):
                file_path = os.path.join(root_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Remove RTL CSS and reset to LTR
                    modified_content = remove_rtl_from_html(content)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    result['css_removed'] += 1
                
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
    
    return result
