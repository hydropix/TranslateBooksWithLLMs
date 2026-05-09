"""
Validate one or more benchmark submission JSON files against the schema.

Usage:
    python scripts/validate_submission.py path/to/sub1.json [path/to/sub2.json ...]

Exit code 0 if all submissions are valid, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import jsonschema

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "benchmark" / "schemas" / "submission.schema.json"


def _load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_error(err: jsonschema.ValidationError) -> str:
    location = "/".join(str(p) for p in err.absolute_path) or "<root>"
    return f"{location}: {err.message}"


def validate(path: Path, validator: jsonschema.Draft202012Validator) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Failed to read JSON: {exc}"]

    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    return [_format_error(e) for e in errors]


def run(paths: Iterable[Path]) -> int:
    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)

    failed = 0
    paths = list(paths)
    if not paths:
        print("No submission files provided.")
        return 0

    for path in paths:
        if not path.is_file():
            print(f"[FAIL] {path}: file not found")
            failed += 1
            continue

        errors = validate(path, validator)
        if errors:
            print(f"[FAIL] {path} ({len(errors)} error(s)):")
            for err in errors[:25]:
                print(f"  - {err}")
            if len(errors) > 25:
                print(f"  ... and {len(errors) - 25} more")
            failed += 1
        else:
            print(f"[OK]   {path}")

    if failed:
        print(f"\n{failed} of {len(paths)} submission(s) failed validation.")
        return 1
    print(f"\nAll {len(paths)} submission(s) are valid.")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 0
    paths = [Path(p) for p in sys.argv[1:]]
    return run(paths)


if __name__ == "__main__":
    sys.exit(main())
