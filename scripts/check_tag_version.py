#!/usr/bin/env python3
"""Verify that a git tag name matches the version declared in src/__version__.py.

Used as a blocking CI gate on every tag-triggered release workflow: a tag whose
name disagrees with the shipped version string must never produce an artifact.

Usage:
    python scripts/check_tag_version.py v1.4.11 [path/to/__version__.py]

Exit code 0 means the tag and the version file agree, 1 means they do not (or
the version file is missing / unparseable / no tag was given).
"""

import os
import re
import sys

DEFAULT_VERSION_FILE = os.path.join("src", "__version__.py")

_VERSION_RE = re.compile(
    r"""^\s*__version__\s*=\s*(['"])(?P<version>.*?)\1\s*$""",
    re.MULTILINE,
)


def extract_version(source):
    """Parse `__version__ = "X.Y.Z"` out of the text of src/__version__.py.

    Returns None when no assignment is found. Accepts single or double quotes
    and arbitrary surrounding whitespace.
    """
    match = _VERSION_RE.search(source)
    if match is None:
        return None
    return match.group("version")


def normalize_tag(tag):
    """Strip a single leading 'v' or 'V' and surrounding whitespace.

    Everything else is preserved verbatim -- no semver parsing, no suffix
    stripping.
    """
    stripped = tag.strip()
    if stripped[:1] in ("v", "V"):
        stripped = stripped[1:]
    return stripped


def main(argv):
    """Compare a tag name against the declared version.

    argv[1] = the tag name (e.g. 'v1.4.11').
    argv[2] = optional path to the version file, defaults to 'src/__version__.py'.

    Prints a one-line diagnosis to stderr and returns 1 on mismatch, on a
    missing/unparseable file, or when no tag argument is given. Returns 0 on
    an exact string match after normalization.
    """
    if len(argv) < 2 or not argv[1].strip():
        print(
            "check_tag_version: missing tag argument. "
            "Usage: check_tag_version.py <tag> [version_file]",
            file=sys.stderr,
        )
        return 1

    tag = argv[1]
    version_file = argv[2] if len(argv) > 2 else DEFAULT_VERSION_FILE

    try:
        with open(version_file, "r", encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        print(
            f"check_tag_version: cannot read version file '{version_file}': {exc}",
            file=sys.stderr,
        )
        return 1

    declared = extract_version(source)
    if declared is None:
        print(
            f"check_tag_version: no __version__ assignment found in '{version_file}'",
            file=sys.stderr,
        )
        return 1

    expected = normalize_tag(tag)
    if expected != declared:
        print(
            f"check_tag_version: tag '{tag}' does not match __version__ "
            f"'{declared}' in '{version_file}' (expected version '{expected}')",
            file=sys.stderr,
        )
        return 1

    print(f"check_tag_version: tag '{tag}' matches __version__ '{declared}'")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
