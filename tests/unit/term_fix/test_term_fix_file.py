"""
Tests for the term_fix_file / scan_file entry points (TXT, literal mode).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.adapters import scan_file, term_fix_file
from src.core.adapters.exceptions import UnsupportedFormatError
from src.core.term_fix import parse_rules_text


@pytest.mark.asyncio
async def test_txt_bytes_outside_matches_are_preserved(tmp_path):
    src = tmp_path / "book.txt"
    raw = "﻿Chapter 1\r\nAshe came.\r\n\r\nNo match here — ünïcode.\r\n"
    src.write_bytes(raw.encode("utf-8"))
    out = tmp_path / "book (term-fix).txt"
    logs = []

    result = await term_fix_file(
        str(src), str(out), parse_rules_text("Ashe -> Askan"),
        log_callback=lambda key, msg: logs.append(key),
    )

    assert out.read_bytes() == raw.replace("Ashe", "Askan").encode("utf-8")
    assert len(result.occurrences) == 1
    report = (tmp_path / "book (term-fix).term-fix.md").read_text(encoding="utf-8")
    assert "| 2 | Ashe | Ashe came. | Askan came. | literal |" in report
    assert logs == ["term_fix_saved", "term_fix_report"]


@pytest.mark.asyncio
async def test_scan_file_writes_nothing(tmp_path):
    src = tmp_path / "book.txt"
    src.write_text("Ashe. Ashe.", encoding="utf-8")
    [res] = await scan_file(str(src), parse_rules_text("Ashe -> Askan"))
    assert res.total == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["book.txt"]


@pytest.mark.asyncio
async def test_unsupported_format(tmp_path):
    src = tmp_path / "book.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:02,000\nAshe\n", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        await term_fix_file(str(src), str(tmp_path / "o.srt"), parse_rules_text("Ashe -> Askan"))


@pytest.mark.asyncio
async def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        await term_fix_file(str(tmp_path / "nope.txt"), str(tmp_path / "o.txt"),
                            parse_rules_text("Ashe -> Askan"))
