"""
Unit tests for scripts/check_tag_version.py.

The script is the CI gate that refuses to build a release artifact when the
pushed tag name disagrees with the version declared in src/__version__.py.
"""
import importlib.util
import re
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the script directly from file: scripts/ is not an importable package.
spec = importlib.util.spec_from_file_location(
    "check_tag_version",
    project_root / "scripts" / "check_tag_version.py"
)
check_tag_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_tag_version)

extract_version = check_tag_version.extract_version
normalize_tag = check_tag_version.normalize_tag
main = check_tag_version.main


def test_extract_version_double_quotes():
    """A plain double-quoted assignment is parsed."""
    assert extract_version('__version__ = "1.4.11"\n') == "1.4.11"


def test_extract_version_single_quotes():
    """Single quotes are accepted too."""
    assert extract_version("__version__ = '1.4.11'\n") == "1.4.11"


def test_extract_version_extra_whitespace():
    """Arbitrary surrounding whitespace does not defeat the parser."""
    assert extract_version('   __version__   =   "1.4.11"   \n') == "1.4.11"


def test_extract_version_no_assignment():
    """A file with no __version__ assignment yields None."""
    assert extract_version("# nothing to see here\nVERSION = '1.0.0'\n") is None


def test_normalize_tag_strips_leading_v():
    """A leading 'v' is stripped; a bare version is left alone."""
    assert normalize_tag("v1.4.11") == "1.4.11"
    assert normalize_tag("1.4.11") == "1.4.11"
    assert normalize_tag("v1.4.11") == normalize_tag("1.4.11")


def test_normalize_tag_uppercase_and_whitespace():
    """An uppercase 'V' and surrounding whitespace are handled."""
    assert normalize_tag(" V2.0.0 ") == "2.0.0"


def test_main_matching_tag_returns_zero(tmp_path):
    """An exact match after normalization exits 0."""
    version_file = tmp_path / "__version__.py"
    version_file.write_text('__version__ = "1.4.11"\n', encoding="utf-8")
    assert main(["x", "v1.4.11", str(version_file)]) == 0


def test_main_mismatched_tag_returns_one(tmp_path):
    """A tag that disagrees with the version file exits 1."""
    version_file = tmp_path / "__version__.py"
    version_file.write_text('__version__ = "1.4.11"\n', encoding="utf-8")
    assert main(["x", "v1.4.10", str(version_file)]) == 1


def test_main_prerelease_suffix_is_a_mismatch(tmp_path):
    """Comparison is exact string equality: a suffix must fail."""
    version_file = tmp_path / "__version__.py"
    version_file.write_text('__version__ = "1.4.11"\n', encoding="utf-8")
    assert main(["x", "v1.4.11-rc1", str(version_file)]) == 1


def test_main_without_tag_argument_returns_one():
    """No tag argument at all is an error."""
    assert main(["x"]) == 1


def test_main_missing_version_file_returns_one():
    """A missing version file is an error, not a pass."""
    assert main(["x", "v1.0.0", "/does/not/exist"]) == 1


def test_main_unparseable_version_file_returns_one(tmp_path):
    """A version file with no __version__ assignment is an error."""
    version_file = tmp_path / "__version__.py"
    version_file.write_text("VERSION = '1.0.0'\n", encoding="utf-8")
    assert main(["x", "v1.0.0", str(version_file)]) == 1


def test_main_writes_diagnosis_to_stderr(tmp_path, capsys):
    """The mismatch diagnosis goes to stderr on one line."""
    version_file = tmp_path / "__version__.py"
    version_file.write_text('__version__ = "1.4.11"\n', encoding="utf-8")
    main(["x", "v1.4.10", str(version_file)])
    captured = capsys.readouterr()
    assert "1.4.10" in captured.err
    assert "1.4.11" in captured.err


def test_real_version_file_parses():
    """The live src/__version__.py must parse to a dotted numeric version.

    Deliberately not asserting a literal version so this test does not need
    editing on every release.
    """
    source = (project_root / "src" / "__version__.py").read_text(encoding="utf-8")
    version = extract_version(source)
    assert version is not None
    assert re.fullmatch(r"\d+(\.\d+)+", version), version
