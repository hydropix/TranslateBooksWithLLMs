"""A UTF-8 BOM must not cost an SRT file its first cue.

Pre-existing on main (2ecce2d) and unchanged by the refactor: parse_srt reads
the file as 'utf-8', so the first block starts with U+FEFF, "\ufeff1".isdigit()
is False and cue 1 is silently skipped (translate, refine and term-fix all
lose it from the output).
"""
import asyncio

from src.core.srt_processor import SRTProcessor

SRT = (
    "1\n00:00:01,000 --> 00:00:02,000\nHello there.\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nGeneral Kenobi.\n"
)


def test_parse_srt_keeps_first_cue_after_bom():
    subtitles = SRTProcessor().parse_srt("\ufeff" + SRT)
    assert [s["number"] for s in subtitles] == ["1", "2"]
    assert subtitles[0]["text"] == "Hello there."


def test_srt_adapter_keeps_first_cue_of_bom_file(tmp_path):
    from src.core.adapters.srt_adapter import SrtAdapter

    path = tmp_path / "bom.srt"
    path.write_bytes(b"\xef\xbb\xbf" + SRT.encode("utf-8"))
    adapter = SrtAdapter(str(path), str(tmp_path / "out.srt"), {})
    assert asyncio.run(adapter.prepare_for_translation())
    assert [s["number"] for s in adapter.subtitles] == ["1", "2"]
