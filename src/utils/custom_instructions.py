"""Custom instructions loader and preset persistence layer.

Loads presets from the `Custom_Instructions/` folder for use in translation
and refinement prompts. Two formats are supported:

- `.yaml` / `.yml`: structured file with optional `translation` and
  `refinement` top-level keys. Either or both may be present; the missing
  phase is left unset. The extended schema additionally supports
  `description`, `mode`, `context`, `source_files`, `generated_at`, and
  `rules` — all metadata for the Styles tab, never read at translation time.
- `.txt` (legacy): plain text, applied to both phases identically.

`load_custom_instructions` returns the translation-time shape
`{"translation": str | None, "refinement": str | None}`. `read_preset`
returns the full extended-schema mapping for editing. `write_preset` and
`delete_preset` provide atomic, sandboxed persistence for that mapping.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

import yaml


# Letters and digits from any script (`\w` is Unicode-aware), plus `_-.`,
# spaces and parentheses, so a preset renamed by hand to "Romance (v2).yaml"
# or "武侠.yaml" is still listed. Path separators and every character Windows
# forbids in a filename stay excluded, and a leading dot or space is refused
# so hidden files and `..` never match. `resolve_inside` remains the
# authoritative containment check.
SAFE_FILENAME_RE = re.compile(r"^(?![.\s])[\w\-. ()]+\.(?:txt|ya?ml)$", re.IGNORECASE)
SUPPORTED_EXTENSIONS = (".txt", ".yaml", ".yml")

# Key order enforced when write_preset serializes a preset. Unknown keys
# preserved from a pre-existing file (read-merge-write) are appended after
# these, in their original order.
PRESET_KEY_ORDER = (
    "description",
    "mode",
    "context",
    "source_files",
    "generated_at",
    "rules",
    "translation",
    "refinement",
)


class CustomInstructions(TypedDict, total=False):
    translation: Optional[str]
    refinement: Optional[str]


class Preset(TypedDict):
    """Full extended-schema mapping returned by `read_preset`."""

    filename: str
    display_name: str
    format: str
    description: str
    mode: Optional[str]
    context: str
    source_files: list
    generated_at: Optional[str]
    rules: list
    translation: Optional[str]
    refinement: Optional[str]


class LiteralStr(str):
    """A string that must be serialized as a YAML block-literal (`|-`).

    Used for `translation`/`refinement` prose so `yaml.safe_dump` never
    folds multi-line instructions into a single wrapped line.
    """


def _represent_literal_str(dumper: yaml.Dumper, data: str):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralStr, _represent_literal_str, Dumper=yaml.SafeDumper)


def is_safe_filename(filename: str) -> bool:
    """Whitelist filenames to letters/digits (any script), `_-. ()` and a supported extension."""
    return bool(SAFE_FILENAME_RE.match(filename or ""))


def resolve_inside(directory: Path, filename: str) -> Optional[Path]:
    """Return the file path if it resolves inside `directory`, else None."""
    candidate = directory / filename
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


# Kept for any external caller still importing the private name.
_resolve_inside = resolve_inside


def slugify_preset_name(name: str) -> str:
    """Turn a free-form preset name into a filesystem-safe slug.

    e.g. "Mon style — noir!" -> "Mon_style_noir", "新建空白风格" -> "新建空白风格".
    Case is preserved and letters/digits from any script are kept; runs of
    any other character collapse to a single underscore, and the result is
    truncated to 64 characters.
    """
    slug = re.sub(r"[^\w-]+", "_", name or "").strip("_")[:64]
    if not slug:
        raise ValueError("Preset name cannot be empty")
    return slug


def filename_for_name(name: str) -> str:
    """Derive a `.yaml` filename from a free-form preset name."""
    return f"{slugify_preset_name(name)}.yaml"


def _normalize_phase_value(value) -> Optional[str]:
    """Coerce a YAML scalar into a stripped string, or None if empty."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    return stripped or None


def _normalize_str_field(value) -> Optional[str]:
    """Coerce a YAML scalar into a string, preserving None."""
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _normalize_rules(rules: list) -> list:
    """Keep only the `dimension`/`instruction` fields of each rule mapping.

    Non-mapping entries are dropped rather than raising, matching the
    loader's "never reject the file, just skip what can't be understood"
    posture used elsewhere in this module.
    """
    normalized = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        normalized.append(
            {
                "dimension": str(rule.get("dimension", "")),
                "instruction": str(rule.get("instruction", "")),
            }
        )
    return normalized


def load_custom_instructions(
    filename: str, custom_instructions_dir: Path
) -> CustomInstructions:
    """Load a preset and return `{"translation": ..., "refinement": ...}`.

    Raises:
        ValueError: filename is unsafe or escapes the directory.
        FileNotFoundError: file does not exist.
        yaml.YAMLError: YAML file is malformed.
    """
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: letters, digits, "
            f"spaces, `_`, `-`, `.`, `(`, `)`; extension must be .txt, .yaml, or .yml."
        )

    file_path = resolve_inside(custom_instructions_dir, filename)
    if file_path is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Custom_Instructions directory."
        )

    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Custom instructions file not found: {filename}")

    suffix = file_path.suffix.lower()
    # `utf-8-sig` transparently strips a leading BOM (Windows Notepad / Word
    # exports default to UTF-8-with-BOM). Plain `utf-8` would leave the BOM
    # as a literal `﻿` character at the start of the parsed value, and
    # would also raise UnicodeDecodeError on a Latin-1 file.
    raw = file_path.read_text(encoding="utf-8-sig")

    if suffix == ".txt":
        text = raw.strip()
        if not text:
            return {"translation": None, "refinement": None}
        return {"translation": text, "refinement": text}

    # YAML
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        raise

    if parsed is None:
        return {"translation": None, "refinement": None}

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Custom instructions YAML '{filename}' must be a mapping with "
            f"optional 'translation' and 'refinement' keys."
        )

    return {
        "translation": _normalize_phase_value(parsed.get("translation")),
        "refinement": _normalize_phase_value(parsed.get("refinement")),
    }


def read_preset(filename: str, custom_instructions_dir: Path) -> dict:
    """Load a preset and return the full extended-schema mapping.

    A legacy `.txt` preset maps its whole content to both `translation` and
    `refinement`, reports `format: "txt"`, and always has `rules: []` — it
    carries no metadata.

    Raises:
        ValueError: filename is unsafe or escapes the directory.
        FileNotFoundError: file does not exist.
        yaml.YAMLError: YAML file is malformed.
    """
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: letters, digits, "
            f"spaces, `_`, `-`, `.`, `(`, `)`; extension must be .txt, .yaml, or .yml."
        )

    file_path = resolve_inside(custom_instructions_dir, filename)
    if file_path is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Custom_Instructions directory."
        )

    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Custom instructions file not found: {filename}")

    suffix = file_path.suffix.lower()
    raw = file_path.read_text(encoding="utf-8-sig")

    result: dict = {
        "filename": filename,
        "display_name": file_path.stem,
        "format": "txt" if suffix == ".txt" else "yaml",
        "description": "",
        "mode": None,
        "context": "",
        "source_files": [],
        "generated_at": None,
        "rules": [],
        "translation": None,
        "refinement": None,
    }

    if suffix == ".txt":
        text = raw.strip() or None
        result["translation"] = text
        result["refinement"] = text
        return result

    parsed = yaml.safe_load(raw)

    if parsed is None:
        return result

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Custom instructions YAML '{filename}' must be a mapping with "
            f"optional 'translation' and 'refinement' keys."
        )

    if parsed.get("description") is not None:
        result["description"] = _normalize_str_field(parsed.get("description")) or ""

    if parsed.get("mode") is not None:
        result["mode"] = _normalize_str_field(parsed.get("mode"))

    if parsed.get("context") is not None:
        result["context"] = _normalize_str_field(parsed.get("context")) or ""

    source_files = parsed.get("source_files")
    if isinstance(source_files, list):
        result["source_files"] = list(source_files)

    if parsed.get("generated_at") is not None:
        result["generated_at"] = _normalize_str_field(parsed.get("generated_at"))

    rules = parsed.get("rules")
    if isinstance(rules, list):
        result["rules"] = _normalize_rules(rules)

    result["translation"] = _normalize_phase_value(parsed.get("translation"))
    result["refinement"] = _normalize_phase_value(parsed.get("refinement"))

    return result


def write_preset(
    filename: str,
    payload: dict,
    custom_instructions_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write an extended-schema YAML preset.

    Writes to `<filename>.tmp` in `custom_instructions_dir` and `os.replace`s
    it onto the target, so a crash mid-write never leaves a half-written
    preset in place. The temp file is removed if anything goes wrong.

    When overwriting an existing file, unknown top-level keys already present
    in it are preserved (read-merge-write) and appended after the known
    `PRESET_KEY_ORDER` keys, in their original order.

    Raises:
        ValueError: filename is unsafe, escapes the directory, or isn't
            `.yaml`/`.yml`.
        FileExistsError: the target exists and `overwrite` is False.
    """
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: letters, digits, "
            f"spaces, `_`, `-`, `.`, `(`, `)`; extension must be .txt, .yaml, or .yml."
        )

    target = resolve_inside(custom_instructions_dir, filename)
    if target is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Custom_Instructions directory."
        )

    if target.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(
            f"write_preset only supports .yaml/.yml files, got '{filename}'."
        )

    if target.exists() and not overwrite:
        raise FileExistsError(f"Preset '{filename}' already exists.")

    existing_data: dict = {}
    if overwrite and target.exists():
        try:
            existing_raw = target.read_text(encoding="utf-8-sig")
            existing_parsed = yaml.safe_load(existing_raw)
            if isinstance(existing_parsed, dict):
                existing_data = existing_parsed
        except (OSError, yaml.YAMLError):
            # A malformed pre-existing file has nothing worth preserving.
            existing_data = {}

    merged = dict(existing_data)
    merged.update(payload)

    # `context` is omitted entirely when empty rather than written as a
    # noisy `context: ""` — this also lets a pre-existing file with no
    # setting round-trip unchanged when nothing sets it.
    if not merged.get("context"):
        merged.pop("context", None)

    ordered: dict = {}
    for key in PRESET_KEY_ORDER:
        if key in merged:
            ordered[key] = merged[key]

    # Unknown keys: existing file's own order first, then any new ones
    # introduced by the payload itself. Keys from PRESET_KEY_ORDER are
    # already handled above (or deliberately dropped, e.g. empty
    # `context`) and must not be reconsidered here.
    for key in existing_data:
        if key not in ordered and key not in PRESET_KEY_ORDER:
            ordered[key] = merged[key]
    for key in payload:
        if key not in ordered and key not in PRESET_KEY_ORDER:
            ordered[key] = merged[key]

    if isinstance(ordered.get("rules"), list):
        ordered["rules"] = _normalize_rules(ordered["rules"])

    for phase in ("translation", "refinement", "context"):
        value = ordered.get(phase)
        if isinstance(value, str) and value:
            ordered[phase] = LiteralStr(value.rstrip("\n"))

    tmp_path = target.with_name(target.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                ordered,
                fh,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
                width=100000,
            )
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return target


def write_txt_preset(filename: str, text: str, custom_instructions_dir: Path) -> Path:
    """Atomically overwrite a legacy `.txt` preset with plain text.

    Lets the Styles editor save a `.txt` preset in place, so users who also
    edit their prompts in an external editor keep a plain text file.

    Raises:
        ValueError: filename is unsafe, escapes the directory, or isn't `.txt`.
    """
    target = _checked_target(filename, custom_instructions_dir)
    if target.suffix.lower() != ".txt":
        raise ValueError(f"write_txt_preset only supports .txt files, got '{filename}'.")

    tmp_path = target.with_name(target.name + ".tmp")
    try:
        tmp_path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return target


def rename_preset(old_filename: str, new_filename: str, custom_instructions_dir: Path) -> Path:
    """Rename a preset file without ever clobbering another preset.

    A case-only rename ("noir.yaml" -> "Noir.yaml") is allowed even on a
    case-insensitive filesystem, where the target already "exists" as the
    source file itself.

    Raises:
        ValueError: either filename is unsafe or escapes the directory.
        FileNotFoundError: the source does not exist.
        FileExistsError: another preset already uses the target name.
    """
    source = _checked_target(old_filename, custom_instructions_dir)
    target = _checked_target(new_filename, custom_instructions_dir)

    if not source.is_file():
        raise FileNotFoundError(f"Custom instructions file not found: {old_filename}")
    if target.exists() and not os.path.samefile(source, target):
        raise FileExistsError(f"Preset '{new_filename}' already exists.")

    os.replace(source, target)
    return target


def _checked_target(filename: str, custom_instructions_dir: Path) -> Path:
    """Validate `filename` and return its path inside the directory."""
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: letters, digits, "
            f"spaces, `_`, `-`, `.`, `(`, `)`; extension must be .txt, .yaml, or .yml."
        )
    target = resolve_inside(custom_instructions_dir, filename)
    if target is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Custom_Instructions directory."
        )
    return target


def delete_preset(filename: str, custom_instructions_dir: Path) -> bool:
    """Delete a preset file (`.yaml`, `.yml` or legacy `.txt`).

    Returns:
        True if a file was deleted, False if it did not exist.

    Raises:
        ValueError: filename is unsafe or escapes the directory.
    """
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: letters, digits, "
            f"spaces, `_`, `-`, `.`, `(`, `)`; extension must be .txt, .yaml, or .yml."
        )

    target = resolve_inside(custom_instructions_dir, filename)
    if target is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Custom_Instructions directory."
        )

    if not target.exists():
        return False

    target.unlink()
    return True


def _mtime_to_iso(file_path: Path) -> str:
    """Convert a file's last-modified time to an ISO-8601 UTC timestamp."""
    dt = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def list_custom_instructions(custom_instructions_dir: Path) -> list[dict]:
    """List presets in the directory with phase availability metadata.

    Returns a list of dicts:
        {
            "filename": "noir_detective.yaml",
            "display_name": "noir_detective",
            "format": "yaml" | "txt",
            "has_translation": bool,
            "has_refinement": bool,
            "description": str,
            "mode": str | None,
            "updated_at": str,  # ISO-8601 UTC, from the file's mtime
        }
    Malformed files are silently skipped.
    """
    if not custom_instructions_dir.exists():
        return []

    entries: list[dict] = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in custom_instructions_dir.glob(f"*{ext}"):
            try:
                file_path.resolve().relative_to(custom_instructions_dir.resolve())
            except ValueError:
                continue

            try:
                preset = read_preset(file_path.name, custom_instructions_dir)
            except (ValueError, yaml.YAMLError, FileNotFoundError, OSError, UnicodeDecodeError):
                continue

            entries.append(
                {
                    "filename": file_path.name,
                    "display_name": file_path.stem,
                    "format": preset["format"],
                    "has_translation": preset["translation"] is not None,
                    "has_refinement": preset["refinement"] is not None,
                    "description": preset["description"],
                    "mode": preset["mode"],
                    "updated_at": _mtime_to_iso(file_path),
                }
            )

    entries.sort(key=lambda e: e["display_name"].lower())
    return entries
