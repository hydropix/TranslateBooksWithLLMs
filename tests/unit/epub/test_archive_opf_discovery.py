"""OPF discovery: META-INF/container.xml first, directory walk as fallback."""
import os

from src.core.epub.archive import find_opf_file
from src.core.epub.rtl_support import apply_rtl_to_epub_directory

OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata/><manifest/><spine/>
</package>
"""


def _container(full_path: str, media_type: str = "application/oebps-package+xml") -> str:
    return f"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="other/alt.xml" media-type="application/xhtml+xml"/>
    <rootfile full-path="{full_path}" media-type="{media_type}"/>
  </rootfiles>
</container>
"""


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_container_xml_rootfile_wins_over_the_walk_order(tmp_path):
    # "A" sorts and walks before "Z": the walk alone would pick the decoy.
    _write(tmp_path, "A/decoy.opf", OPF)
    real = _write(tmp_path, "Z/package.opf", OPF)
    _write(tmp_path, "META-INF/container.xml", _container("Z/package.opf"))

    assert os.path.normpath(find_opf_file(str(tmp_path))) == os.path.normpath(str(real))


def test_percent_encoded_full_path_is_resolved(tmp_path):
    real = _write(tmp_path, "OEBPS/my book.opf", OPF)
    _write(tmp_path, "META-INF/container.xml", _container("OEBPS/my%20book.opf"))

    assert os.path.normpath(find_opf_file(str(tmp_path))) == os.path.normpath(str(real))


def test_no_container_falls_back_to_the_first_opf(tmp_path):
    real = _write(tmp_path, "OEBPS/content.opf", OPF)

    assert find_opf_file(str(tmp_path)) == os.path.join(str(tmp_path / "OEBPS"), "content.opf")
    assert os.path.normpath(find_opf_file(str(tmp_path))) == os.path.normpath(str(real))


def test_container_pointing_at_a_missing_file_falls_back(tmp_path):
    real = _write(tmp_path, "OEBPS/content.opf", OPF)
    _write(tmp_path, "META-INF/container.xml", _container("OPS/missing.opf"))

    assert os.path.normpath(find_opf_file(str(tmp_path))) == os.path.normpath(str(real))


def test_container_escaping_the_book_falls_back(tmp_path):
    book = tmp_path / "book"
    real = _write(book, "OEBPS/content.opf", OPF)
    _write(tmp_path, "outside.opf", OPF)
    _write(book, "META-INF/container.xml", _container("../outside.opf"))

    assert os.path.normpath(find_opf_file(str(book))) == os.path.normpath(str(real))


def test_container_without_a_package_rootfile_falls_back(tmp_path):
    real = _write(tmp_path, "OEBPS/content.opf", OPF)
    _write(tmp_path, "META-INF/container.xml", _container("OEBPS/content.opf", "text/plain"))

    assert os.path.normpath(find_opf_file(str(tmp_path))) == os.path.normpath(str(real))


def test_no_opf_at_all(tmp_path):
    _write(tmp_path, "META-INF/container.xml", "not xml at all <<<")
    assert find_opf_file(str(tmp_path)) is None


def test_rtl_pass_updates_the_package_container_xml_names(tmp_path):
    decoy = _write(tmp_path, "A/decoy.opf", OPF)
    real = _write(tmp_path, "Z/package.opf", OPF)
    _write(tmp_path, "META-INF/container.xml", _container("Z/package.opf"))

    result = apply_rtl_to_epub_directory(str(tmp_path), "Arabic", "English")

    assert result["opf_updated"]
    assert "page-progression-direction" in real.read_text(encoding="utf-8")
    assert "page-progression-direction" not in decoy.read_text(encoding="utf-8")
