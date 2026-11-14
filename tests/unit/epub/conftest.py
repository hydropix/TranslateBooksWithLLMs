"""
Shared fixtures and helpers for the CJK typography / metadata translation test
modules (Phase 3-7 of plan/PLAN_CjkSourceRendering.md).

Consolidated here (Phase 7, Task 1) from what used to be a cross-module import
chain: `test_cjk_typography_apply.py` originally defined `_build_cjk_epub_dir`,
`cjk_epub_dir` and the container-shape constants; `test_cjk_typography_pipeline.py`
imported them and added the echo-LLM/zip/attribution helpers;
`test_metadata_translator.py` imported from both. Every piece below is used by
at least two of those three modules -- anything used by only one of them was
deliberately left in that module instead of being moved here.

The container fixtures reproduce the shape of the reported book (a Chinese
EPUB produced by "Ag2S EpubLib" with duokan reader metadata); their markup
fragments are verbatim from it.
"""
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import INPUT_TAG_IN, INPUT_TAG_OUT
from src.core.llm.base import LLMResponse


REAL_CSS = Path(__file__).resolve().parents[2] / "fixtures" / "cjk_epub" / "main.css"


CONTENT_OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0" unique-identifier="duokan-book-id">
  <metadata>
    <dc:title>被渣后和前夫破镜重圆了</dc:title>
    <dc:language>zh</dc:language>
    <meta name="cover" content="cover"/>
    <meta name="generator" content="Ag2S EpubLib"/>
    <meta name="duokan-body-font" content="DK-SONGTI"/>
    <meta name="calibre:title_sort" content="被渣后和前夫破镜重圆了"/>
  </metadata>
  <manifest>
    <item id="css" href="Styles/main.css" media-type="text/css"/>
    <item id="cover" href="Text/cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="intro" href="Text/intro.xhtml" media-type="application/xhtml+xml"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="font" href="Fonts/zdy2.ttf" media-type="application/x-font-ttf"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="cover"/>
    <itemref idref="intro"/>
  </spine>
</package>
"""

TOC_NCX = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" xml:lang="zh">
  <docTitle><text>被渣后和前夫破镜重圆了</text></docTitle>
  <navMap>
    <navPoint id="n1" playOrder="1">
      <navLabel><text>内容简介</text></navLabel>
      <content src="Text/intro.xhtml"/>
    </navPoint>
  </navMap>
</ncx>
"""

# Verbatim from the reported book: the cover page's override stylesheet and its
# SVG-wrapped cover image. Also the shape that guards against `method='html'`.
COVER_XHTML = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>Cover</title>
<style type="text/css" title="override_css">
@page {padding: 0pt; margin:0pt}
body { text-align: center; padding:0pt; margin: 0pt; }
</style>
</head>
<body><div><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="100%" height="100%" viewBox="0 0 1200 1600" preserveAspectRatio="none"><image width="1200" height="1600" xlink:href="../Images/cover.jpg"/></svg></div></body>
</html>
"""

# The <style> block carries a CJK signal; the <h1> carries the book's real
# inline style attribute, extended with the two declarations of the plan's
# validation criterion 1.
INTRO_XHTML = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>内容简介</title>
<link rel="stylesheet" type="text/css" href="../Styles/main.css"/>
<style type="text/css">
p.quote { font-family: "楷体", serif; line-height: 130%; }
</style>
</head>
<body>
<h1 class="head" style="margin-bottom:2em;">内容简介</h1>
<p style="font-family:宋体;line-height:120%">正文<br/></p>
<p><img src="x.jpg"/></p>
</body>
</html>
"""


def _write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


def _build_cjk_epub_dir(root: Path, css_text: str, opf_text: str = CONTENT_OPF) -> Path:
    """Minimal extracted EPUB reproducing the reported book's container shape."""
    oebps = root / "OEBPS"
    _write(root / "META-INF" / "container.xml",
           '<?xml version="1.0" encoding="utf-8"?>\n'
           '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">\n'
           '  <rootfiles><rootfile full-path="OEBPS/content.opf" '
           'media-type="application/oebps-package+xml"/></rootfiles>\n'
           '</container>\n')
    _write(root / "mimetype", "application/epub+zip")
    _write(oebps / "content.opf", opf_text)
    _write(oebps / "toc.ncx", TOC_NCX)
    _write(oebps / "Styles" / "main.css", css_text)
    _write(oebps / "Text" / "cover.xhtml", COVER_XHTML)
    _write(oebps / "Text" / "intro.xhtml", INTRO_XHTML)
    (oebps / "Fonts").mkdir(parents=True, exist_ok=True)
    # 4 KB stand-in for the book's zdy2.ttf (a 15-glyph CJK subset).
    (oebps / "Fonts" / "zdy2.ttf").write_bytes(b"\x00\x01\x00\x00" + b"\x00" * 4092)
    return root


@pytest.fixture
def cjk_epub_dir(tmp_path: Path) -> Path:
    """The reported book's container, with its real stylesheet."""
    return _build_cjk_epub_dir(tmp_path / "epub",
                               REAL_CSS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pipeline harness: identity-echo LLM client, EPUB zipping, attribution toggle
# ---------------------------------------------------------------------------

def _echo_llm_client() -> MagicMock:
    """Identity-translation stub: echoes the source content back unchanged.

    Keeps every placeholder byte-identical across the "translation", so the
    pipeline never needs a retry or the token-alignment fallback -- this
    fixture is about CJK typography wiring, not translation quality. No API
    key or network access is used anywhere in this stub.
    """
    client = MagicMock()
    client.context_window = 2048

    async def generate(user_prompt, system_prompt=None, **kwargs):
        start = user_prompt.find(INPUT_TAG_IN)
        end = user_prompt.find(INPUT_TAG_OUT)
        if start != -1 and end != -1:
            content = user_prompt[start + len(INPUT_TAG_IN):end].strip("\n")
        else:
            content = user_prompt
        return LLMResponse(
            content=content,
            prompt_tokens=10,
            completion_tokens=10,
            context_used=20,
            context_limit=4096,
            was_truncated=False,
        )

    client.generate = generate
    client.extract_translation = lambda response: response
    return client


def _zip_dir_as_epub(root: Path, dest: Path) -> Path:
    """Package an extracted-EPUB directory tree into a real .epub file."""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        mimetype_path = root / "mimetype"
        if mimetype_path.exists():
            archive.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "mimetype":
                archive.write(path, str(path.relative_to(root)))
    return dest


def _read(epub_path, arcname: str) -> str:
    with zipfile.ZipFile(epub_path) as archive:
        return archive.read(arcname).decode("utf-8")


def _disable_attribution(monkeypatch):
    """Keep the attribution page/signature out of the picture.

    Unrelated to this phase, and its flags default from the developer's local
    .env -- pinning them off keeps the test deterministic regardless of that.
    """
    monkeypatch.setattr("src.core.epub.opf_metadata.ATTRIBUTION_ENABLED", False)
    monkeypatch.setattr("src.core.epub.attribution_page.ATTRIBUTION_ENABLED", False)


@pytest.fixture
def input_epub(tmp_path: Path) -> Path:
    """A real .epub file built from the reported book's container shape."""
    container_root = _build_cjk_epub_dir(
        tmp_path / "src_epub", REAL_CSS.read_text(encoding="utf-8"))
    return _zip_dir_as_epub(container_root, tmp_path / "input.epub")
