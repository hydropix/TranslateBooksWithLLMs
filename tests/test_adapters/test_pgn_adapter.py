"""
Tests for PgnAdapter.

Only the prose of {...} and ; comments may change; tag pairs, move text,
variations, NAGs, escape lines and embedded commands ([%cal], [#], ...)
must come back byte for byte, and moves quoted inside comments must
survive the translation verbatim.
"""

import re

import pytest

import src.config as config_module
import src.core.llm_client as llm_client_module
import src.core.translator as translator_module
from src.core.adapters import PgnAdapter
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.pgn_adapter import _comment_spans
from src.persistence.checkpoint_manager import CheckpointManager

# The Opera Game (Morphy, Paris 1858, public domain) with annotations
# written for these tests, followed by a short second game exercising
# ; comments, an escape line and tag values containing comment characters.
PGN = """[Event "Paris Opera {casual}"]
[Site "Paris FRA"]
[Date "1858.??.??"]
[White "Morphy, Paul"]
[Black "Duke Karl / Count Isouard; allies"]
[Result "1-0"]

{A famous lesson in rapid development.} 1. e4 e5 2. Nf3 d6 3. d4 Bg4 $2
{[%csl Rg4] This pin is a mistake: after 4.dxe5 Black must give up the
bishop, as 4...dxe5 5.Qxd8+ Kxd8 6.Nxe5 loses a pawn.} 4. dxe5 Bxf3 5. Qxf3
dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 (8. Qxb7 Qb4+ {trades queens and lets Black breathe.}) 8... c6 9. Bg5 b5 10. Nxb5
cxb5 11. Bxb5+ Nbd7 12. O-O-O {[%cal Gd1d8] Every white piece now joins
the attack.} Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7 {} 16. Qb8+ $1
{The queen sacrifice opens the d-file. [#] White mates next move.} 16...
Nxb8 17. Rd8# {[%clk 0:00:10]} 1-0

% The next game is a short test fragment.
[Event "Test"]
[Result "*"]

1. d4 ; The queen's pawn opening.
1... d5 2. c4 {Offering a pawn for the centre.} *
"""

# Comment prose in file order, as sent to the LLM.
PROSE = [
    "A famous lesson in rapid development.",
    ("This pin is a mistake: after 4.dxe5 Black must give up the bishop, "
     "as 4...dxe5 5.Qxd8+ Kxd8 6.Nxe5 loses a pawn."),
    "trades queens and lets Black breathe.",
    "Every white piece now joins the attack.",
    "The queen sacrifice opens the d-file.",
    "White mates next move.",
    "The queen's pawn opening.",
    "Offering a pawn for the centre.",
]


def _write(tmp_path, content=PGN, name="game.pgn", encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(content.encode(encoding))
    return path


async def _prepared(tmp_path, content=PGN, config=None, **kwargs):
    adapter = PgnAdapter(
        str(_write(tmp_path, content, **kwargs)),
        str(tmp_path / "out.pgn"),
        config or {},
    )
    assert await adapter.prepare_for_translation()
    return adapter


def _translate_lines(unit_content, translate):
    """Fake LLM response: apply translate() to every [N] line."""
    lines = []
    for line in unit_content.split('\n'):
        match = re.match(r"^\[(\d+)\](.*)$", line)
        lines.append(f"[{match.group(1)}]{translate(match.group(2))}")
    return '\n'.join(lines)


def _strip_comment_prose(text):
    """Remove every comment body, so only the game structure remains."""
    parts, pos = [], 0
    for start, end in _comment_spans(text):
        parts.append(text[pos:start])
        pos = end
    parts.append(text[pos:])
    return ''.join(parts)


@pytest.fixture(autouse=True)
def no_attribution(monkeypatch):
    monkeypatch.setattr(config_module, "ATTRIBUTION_ENABLED", False)


class TestParsing:

    def test_format_name(self, tmp_path):
        adapter = PgnAdapter(str(tmp_path / "a.pgn"), str(tmp_path / "b.pgn"), {})
        assert adapter.format_name == "pgn"

    @pytest.mark.asyncio
    async def test_only_comment_prose_is_extracted(self, tmp_path):
        adapter = await _prepared(tmp_path)
        assert [p['text'] for p in adapter.pieces] == PROSE

    def test_tag_values_and_escape_lines_are_not_comments(self):
        bodies = [PGN[s:e] for s, e in _comment_spans(PGN)]
        assert not any("casual" in b or "allies" in b for b in bodies)
        assert not any("test fragment" in b for b in bodies)

    def test_unterminated_comment_leaves_rest_untouched(self):
        text = '1. e4 {Good move. 1... e5 {Fine.}'
        # The first { swallows up to the only } (PGN comments do not nest);
        # a lone { without any } yields no span.
        assert _comment_spans('1. e4 {never closed 1... e5') == []
        assert len(_comment_spans(text)) == 1

    @pytest.mark.asyncio
    async def test_commands_and_empty_comments_are_skipped(self, tmp_path):
        adapter = await _prepared(tmp_path)
        sent = '\n'.join(u.content for u in adapter.get_translation_units())
        assert "[%" not in sent and "[#]" not in sent
        assert "0:00:10" not in sent

    @pytest.mark.asyncio
    async def test_blocks_respect_count_and_token_bounds(self, tmp_path):
        adapter = await _prepared(tmp_path, config={'lines_per_block': 3})
        units = adapter.get_translation_units()
        assert [len(u.metadata['local_to_global']) for u in units] == [3, 3, 2]
        assert units[1].context_before == PROSE[2]
        assert units[1].context_after == PROSE[6]

        adapter = await _prepared(tmp_path, config={'max_tokens_per_chunk': 1})
        assert len(adapter.get_translation_units()) == len(PROSE)


class TestReconstruction:

    @pytest.mark.asyncio
    async def test_untranslated_round_trip_is_byte_identical(self, tmp_path):
        adapter = await _prepared(tmp_path)
        assert await adapter.reconstruct_output() == PGN.encode('utf-8')

    @pytest.mark.asyncio
    async def test_crlf_round_trip_is_byte_identical(self, tmp_path):
        crlf = PGN.replace('\n', '\r\n')
        adapter = await _prepared(tmp_path, crlf)
        assert [p['text'] for p in adapter.pieces] == PROSE
        assert await adapter.reconstruct_output() == crlf.encode('utf-8')

    @pytest.mark.asyncio
    async def test_translation_changes_only_comment_prose(self, tmp_path):
        adapter = await _prepared(tmp_path)
        for unit in adapter.get_translation_units():
            response = _translate_lines(unit.content, lambda t: f"PT {t}")
            assert await adapter.save_unit_translation(unit.unit_id, response)

        output = (await adapter.reconstruct_output()).decode('utf-8')

        assert _strip_comment_prose(output) == _strip_comment_prose(PGN)
        assert "{PT A famous lesson in rapid development.}" in output
        assert "{[%csl Rg4] PT This pin is a mistake" in output
        assert "{[%cal Gd1d8] PT Every white piece" in output
        assert ("{PT The queen sacrifice opens the d-file. [#] "
                "PT White mates next move.}") in output
        assert "{[%clk 0:00:10]}" in output
        assert "Nxd7 {} 16." in output
        assert "1. d4 ; PT The queen's pawn opening.\n" in output

    @pytest.mark.asyncio
    async def test_bilingual_keeps_original_prose(self, tmp_path):
        adapter = await _prepared(tmp_path)
        unit = adapter.get_translation_units()[0]
        await adapter.save_unit_translation(
            unit.unit_id, _translate_lines(unit.content, lambda t: f"PT {t}")
        )
        output = (await adapter.reconstruct_output(bilingual=True)).decode('utf-8')
        assert ("{A famous lesson in rapid development. / "
                "PT A famous lesson in rapid development.}") in output

    @pytest.mark.asyncio
    async def test_braces_and_line_breaks_in_translation_are_neutralized(self, tmp_path):
        adapter = await _prepared(tmp_path)
        unit = adapter.get_translation_units()[0]
        response = _translate_lines(unit.content, lambda t: f"{{x}} {t}")
        await adapter.save_unit_translation(unit.unit_id, response)
        output = (await adapter.reconstruct_output()).decode('utf-8')
        assert "{(x) A famous lesson" in output
        assert _strip_comment_prose(output) == _strip_comment_prose(PGN)

    @pytest.mark.asyncio
    async def test_cp1252_input_is_written_as_utf8(self, tmp_path):
        content = '[Event "Café"]\n\n1. e4 {Très bien.} *\n'
        adapter = await _prepared(tmp_path, content, encoding="cp1252")
        assert adapter.pieces[0]['text'] == "Très bien."
        assert await adapter.reconstruct_output() == content.encode('utf-8')

    @pytest.mark.asyncio
    async def test_attribution_is_a_pgn_escape_line(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config_module, "ATTRIBUTION_ENABLED", True)
        adapter = await _prepared(tmp_path)
        output = (await adapter.reconstruct_output()).decode('utf-8')
        assert output.startswith(PGN)
        assert all(line.startswith('%') for line in output[len(PGN):].splitlines())


class TestNotationGuard:

    @pytest.mark.asyncio
    async def test_validation_accepts_reordered_prose(self, tmp_path):
        adapter = await _prepared(tmp_path, config={'lines_per_block': 2})
        response = ("[0]Uma lição famosa de desenvolvimento rápido.\n"
                    "[1]Esta cravada é um erro: o Bispo deve ser entregue "
                    "depois de 4.dxe5, pois 4...dxe5 5.Qxd8+ Kxd8 6.Nxe5 "
                    "perde um peão.")
        assert adapter.validate_unit_translation("block_0", response) is None

    @pytest.mark.asyncio
    async def test_validation_rejects_translated_piece_letters(self, tmp_path):
        adapter = await _prepared(tmp_path, config={'lines_per_block': 2})
        response = ("[0]Uma lição famosa.\n"
                    "[1]Esta cravada é um erro: depois de 4.dxe5 as pretas "
                    "perdem o bispo, pois 4...dxe5 5.Dxd8+ Rxd8 6.Cxe5 perde "
                    "um peão.")
        feedback = adapter.validate_unit_translation("block_0", response)
        assert feedback is not None
        assert "notation" in feedback and "[1]" in feedback

    @pytest.mark.asyncio
    async def test_validation_reports_missing_markers(self, tmp_path):
        adapter = await _prepared(tmp_path, config={'lines_per_block': 2})
        feedback = adapter.validate_unit_translation("block_0", "[0]Só um.")
        assert feedback is not None and "[1]" in feedback

    @pytest.mark.asyncio
    async def test_altered_piece_keeps_original_others_are_saved(self, tmp_path):
        adapter = await _prepared(tmp_path, config={'lines_per_block': 2})
        response = "[0]Uma lição famosa.\n[1]Erro: depois de 4.dxe6 tudo muda."
        assert await adapter.save_unit_translation("block_0", response)
        output = (await adapter.reconstruct_output()).decode('utf-8')
        assert "{Uma lição famosa.}" in output
        assert "after 4.dxe5 Black must give up the\nbishop, as" in output

    @pytest.mark.asyncio
    async def test_resume_restores_saved_blocks(self, tmp_path):
        adapter = await _prepared(tmp_path, config={'lines_per_block': 2})
        checkpoint = {
            'resume_from_index': 1,
            'chunks': [
                {'status': 'completed',
                 'chunk_data': {'local_to_global': {'0': 0, '1': 1}},
                 'translated_text': ("[0]Uma lição.\n[1]Erro depois de 4.dxe5: "
                                     "4...dxe5 5.Qxd8+ Kxd8 6.Nxe5.")},
                {'status': 'failed',
                 'chunk_data': {'local_to_global': {'0': 2, '1': 3}},
                 'translated_text': None},
            ],
        }
        assert await adapter.resume_from_checkpoint(checkpoint) == 1
        assert adapter.translations == {
            0: "Uma lição.",
            1: "Erro depois de 4.dxe5: 4...dxe5 5.Qxd8+ Kxd8 6.Nxe5.",
        }


class FakeLLMClient:
    def __init__(self, **kwargs):
        pass


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    yield manager
    manager.close()


async def _run_translation(tmp_path, cm, monkeypatch, translate_by_attempt):
    """Run GenericTranslator on PGN with a fake LLM.

    translate_by_attempt(n) returns the per-line translate function used for
    the n-th LLM call (0-based).
    """
    calls = []

    async def fake_request(main_content=None, **kwargs):
        calls.append(kwargs.get("prompt_options") or {})
        return _translate_lines(main_content, translate_by_attempt(len(calls) - 1))

    monkeypatch.setattr(llm_client_module, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(translator_module, "generate_translation_request", fake_request)

    adapter = PgnAdapter(str(_write(tmp_path)), str(tmp_path / "out.pgn"),
                         {'lines_per_block': 100})
    translator = GenericTranslator(adapter=adapter, checkpoint_manager=cm,
                                   translation_id="job")
    success = await translator.translate(
        source_language="English", target_language="Portuguese",
        model_name="fake-model", llm_provider="ollama",
        log_callback=lambda *args, **kwargs: None,
    )
    output = (tmp_path / "out.pgn").read_text(encoding="utf-8")
    return success, output, calls


def _alter_move(text):
    return "PT " + text.replace("dxe5", "dxe6")


@pytest.mark.asyncio
async def test_altered_notation_is_retried_with_chess_instructions(tmp_path, cm, monkeypatch):
    success, output, calls = await _run_translation(
        tmp_path, cm, monkeypatch,
        lambda n: _alter_move if n == 0 else (lambda t: f"PT {t}"),
    )
    assert success is True
    assert len(calls) == 2
    assert "Never translate piece letters" in calls[1].get("custom_instructions", "")
    assert "{[%csl Rg4] PT This pin is a mistake: after 4.dxe5" in output
    assert cm.get_job("job")["status"] == "completed"


@pytest.mark.asyncio
async def test_persistent_notation_change_keeps_source_comment(tmp_path, cm, monkeypatch):
    monkeypatch.setattr(config_module, "UNIT_VALIDATION_RETRIES", 1)
    success, output, calls = await _run_translation(
        tmp_path, cm, monkeypatch, lambda n: _alter_move,
    )
    assert success is False
    assert len(calls) == 2
    # The altered comment stays in English, every other one is translated.
    assert "after 4.dxe5 Black must give up the\nbishop, as" in output
    assert "{PT A famous lesson in rapid development.}" in output
    assert _strip_comment_prose(output) == _strip_comment_prose(PGN)
    assert cm.get_job("job")["status"] == "partial"
