"""
PGN Adapter for the generic translation system.

Translates the annotations of chess game files (Portable Game Notation)
while leaving the games themselves untouched: tag pairs, move text, move
numbers, variations, NAGs, results, escape lines and the commands embedded
in comments ([%cal ...], [%csl ...], [%clk ...], ChessBase's [#] diagram
marker) are copied byte for byte. Only the prose of {...} and ; comments is
sent to the LLM, one [N]-marked line per piece of prose, using the same
block protocol as the SRT adapter.

Moves quoted inside a comment ("after 12.Nf3 Black...") must survive the
translation verbatim. Every translated piece is checked against the
notation fingerprint of its source; a piece whose moves or squares changed
is retried, and kept in the source language if it still differs.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .format_adapter import FormatAdapter
from .translation_unit import TranslationUnit

# Commands embedded in comments that are not prose and never reach the LLM.
_COMMAND_RE = re.compile(r"\[%[^\]]*\]|\[#\]")

# SAN moves, castling and bare squares quoted in prose. Compared as a
# multiset, so the translation may reorder words but not add, drop or
# alter a move (e.g. Nf3 -> Cf3).
_NOTATION_RE = re.compile(
    r"\b(?:O-O-O|O-O"
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?"
    r"|[a-h][1-8](?:=[QRBN])?)"
    r"[+#]?"
)


def _notation_fingerprint(text: str) -> Counter:
    """Multiset of the chess notation tokens quoted in a piece of prose."""
    return Counter(_NOTATION_RE.findall(text))


def _sanitize(text: str) -> str:
    """Make translated prose safe to splice back into a comment.

    Braces would close or open a comment, and a line break would end a
    ; comment, so both are neutralized.
    """
    text = text.replace("{", "(").replace("}", ")")
    return " ".join(text.split())


def _skip_tag_pair(text: str, start: int) -> int:
    """Return the index just past the tag pair opened at text[start] == '['.

    Quoted values may contain ']', '{' or ';' (with backslash escapes), so
    they are skipped as a whole. A '[' that is not closed on its line is
    consumed alone.
    """
    i, n = start + 1, len(text)
    in_quotes = False
    while i < n:
        c = text[i]
        if c == "\n":
            break
        if in_quotes:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_quotes = False
        elif c == '"':
            in_quotes = True
        elif c == "]":
            return i + 1
        i += 1
    return start + 1


def _comment_spans(text: str) -> List[Tuple[int, int]]:
    """Return the (start, end) spans of every comment body in a PGN text.

    {...} bodies exclude the braces; ; bodies run to the end of the line,
    excluding the line break. Tag pairs and % escape lines are skipped, so
    braces or semicolons inside them are not mistaken for comments. An
    unterminated { leaves the rest of the file untouched.
    """
    spans = []
    i, n = 0, len(text)
    at_line_start = True
    while i < n:
        c = text[i]
        if c == "%" and at_line_start:
            newline = text.find("\n", i)
            i = n if newline < 0 else newline
            continue
        if c == "[":
            i = _skip_tag_pair(text, i)
            at_line_start = False
            continue
        if c == "{":
            close = text.find("}", i + 1)
            if close < 0:
                break
            spans.append((i + 1, close))
            i = close + 1
            at_line_start = False
            continue
        if c == ";":
            newline = text.find("\n", i + 1)
            end = n if newline < 0 else newline
            if end > i + 1 and text[end - 1] == "\r":
                end -= 1
            spans.append((i + 1, end))
            i = end
            continue
        at_line_start = c == "\n"
        i += 1
    return spans


class PgnAdapter(FormatAdapter):
    """Adapter for PGN (chess game) files: translates comments only."""

    def __init__(self, input_file_path: str, output_file_path: str, config: Dict[str, Any]):
        super().__init__(input_file_path, output_file_path, config)
        self.text: str = ""
        # One entry per piece of comment prose, in file order:
        # {'start', 'end', 'text', 'fingerprint'} where start/end delimit the
        # prose in self.text (surrounding whitespace excluded) and 'text' is
        # the whitespace-collapsed version sent to the LLM.
        self.pieces: List[Dict[str, Any]] = []
        self.blocks: List[List[int]] = []
        self.translations: Dict[int, str] = {}  # piece index -> translated prose
        self.processor = None
        self._units: Optional[List[TranslationUnit]] = None

    async def prepare_for_translation(self) -> bool:
        """Decode the file, locate the comment prose and group it into blocks."""
        try:
            # Import here to avoid circular dependency
            from src.core.srt_processor import SRTProcessor
            self.processor = SRTProcessor()

            with open(self.input_file_path, 'rb') as f:
                raw = f.read()
            # ChessBase and older tools export Windows-1252; everything else
            # is UTF-8. Line endings are preserved because bytes are decoded
            # without newline translation.
            try:
                self.text = raw.decode('utf-8-sig')
            except UnicodeDecodeError:
                self.text = raw.decode('cp1252', errors='replace')

            self.pieces = []
            for body_start, body_end in _comment_spans(self.text):
                pos = body_start
                for command in _COMMAND_RE.finditer(self.text, body_start, body_end):
                    self._add_piece(pos, command.start())
                    pos = command.end()
                self._add_piece(pos, body_end)

            self.blocks = self._group_pieces()
            self._units = None
            return True

        except Exception:
            return False

    def _add_piece(self, start: int, end: int):
        """Register text[start:end] as a piece of prose if it has any words."""
        raw = self.text[start:end]
        prose = " ".join(raw.split())
        if not any(ch.isalpha() for ch in prose):
            return
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        self.pieces.append({
            'start': start + leading,
            'end': end - trailing,
            'text': prose,
            'fingerprint': _notation_fingerprint(prose),
        })

    def _group_pieces(self) -> List[List[int]]:
        """Group consecutive pieces into blocks bounded by count and tokens.

        The count bound (SRT_LINES_PER_BLOCK, shared with SRT) keeps [N]
        marker accounting reliable; the token bound keeps long book
        annotations within the usual chunk size. A piece larger than the
        token bound goes alone in its block.
        """
        from src.config import SRT_LINES_PER_BLOCK, MAX_TOKENS_PER_CHUNK
        from src.core.chunking.token_chunker import count_tokens

        max_items = self.config.get('lines_per_block') or SRT_LINES_PER_BLOCK
        max_tokens = self.config.get('max_tokens_per_chunk') or MAX_TOKENS_PER_CHUNK

        blocks: List[List[int]] = []
        current: List[int] = []
        current_tokens = 0
        for index, piece in enumerate(self.pieces):
            tokens = count_tokens(piece['text'])
            if current and (len(current) >= max_items
                            or current_tokens + tokens > max_tokens):
                blocks.append(current)
                current, current_tokens = [], 0
            current.append(index)
            current_tokens += tokens
        if current:
            blocks.append(current)
        return blocks

    def get_translation_units(self) -> List[TranslationUnit]:
        """Create one unit per block of comment prose (cached after first build)."""
        if self._units is not None:
            return self._units

        units = []
        for block_idx, block in enumerate(self.blocks):
            local_to_global = dict(enumerate(block))
            block_text = '\n'.join(
                f"[{local_idx}]{self.pieces[global_idx]['text']}"
                for local_idx, global_idx in local_to_global.items()
            )

            context_before = ""
            context_after = ""
            if block_idx > 0:
                context_before = self.pieces[self.blocks[block_idx - 1][-1]]['text']
            if block_idx < len(self.blocks) - 1:
                context_after = self.pieces[self.blocks[block_idx + 1][0]]['text']

            units.append(TranslationUnit(
                unit_id=f"block_{block_idx}",
                content=block_text,
                context_before=context_before,
                context_after=context_after,
                metadata={
                    'block_index': block_idx,
                    'local_to_global': local_to_global,
                }
            ))

        self._units = units
        return units

    def _extract(self, translated_content: str, local_to_global: Dict[int, int]) -> Dict[int, str]:
        """Map an LLM response back to sanitized prose per global piece index."""
        extracted = self.processor.extract_block_translations_with_remapping(
            translated_content, local_to_global
        )
        return {
            global_idx: _sanitize(text)
            for global_idx, text in extracted.items()
            if global_idx < len(self.pieces)
        }

    def _keeps_notation(self, global_idx: int, translated: str) -> bool:
        """True when the translation quotes exactly the moves of its source."""
        return _notation_fingerprint(translated) == self.pieces[global_idx]['fingerprint']

    def _store(self, translated_content: str, local_to_global: Dict[int, int]):
        """Keep every non-empty translated piece whose notation is intact."""
        for global_idx, text in self._extract(translated_content, local_to_global).items():
            if text and self._keeps_notation(global_idx, text):
                self.translations[global_idx] = text

    def validate_unit_translation(self, unit_id: str, translated_content: str) -> Optional[str]:
        """Check that every [N] marker came back and no quoted move changed."""
        try:
            block_idx = int(unit_id.split('_')[1])
            unit = self.get_translation_units()[block_idx]
        except (IndexError, ValueError):
            return None

        local_to_global = unit.metadata['local_to_global']
        extracted = self._extract(translated_content, local_to_global)

        missing = []
        altered = []
        for local_idx, global_idx in local_to_global.items():
            text = extracted.get(global_idx)
            if not text:
                missing.append(local_idx)
            elif not self._keeps_notation(global_idx, text):
                altered.append(local_idx)

        problems = []
        if missing:
            problems.append("missing comment index markers: "
                            + ', '.join(f'[{n}]' for n in missing))
        if altered:
            problems.append("chess notation altered in comments: "
                            + ', '.join(f'[{n}]' for n in altered))
        return '; '.join(problems) or None

    def get_validation_retry_instructions(self, feedback: str) -> str:
        """Reinforce the [N] marker contract and the notation rule on retry."""
        return (
            f"CRITICAL: Your previous response was invalid ({feedback}). "
            f"You MUST reproduce every [N] index marker from the input exactly "
            f"once, in order, each followed by its translation. Do NOT merge, "
            f"drop or renumber markers. These lines are annotations of chess "
            f"games: copy every move, square and castling notation (e.g. Nf3, "
            f"exd5, Qxh7+, O-O, e4) exactly as written, letters included. "
            f"Never translate piece letters into the target language."
        )

    async def save_unit_translation(self, unit_id: str, translated_content: str) -> bool:
        """Store the valid pieces of a translated block; the rest stay untranslated."""
        try:
            block_idx = int(unit_id.split('_')[1])
            units = self.get_translation_units()
            if block_idx >= len(units):
                return False

            self._store(translated_content, units[block_idx].metadata['local_to_global'])
            return True

        except Exception:
            return False

    async def reconstruct_output(self, bilingual: bool = False) -> bytes:
        """
        Rebuild the PGN file with translated comments.

        Everything outside the translated prose is copied unchanged. The
        output is always UTF-8, since target languages may not fit the
        source encoding.

        Args:
            bilingual: If True, each translated comment keeps the original
                      prose followed by " / " and the translation.

        Returns:
            Complete PGN file as bytes
        """
        try:
            from src.config import ATTRIBUTION_ENABLED, GENERATOR_NAME, GENERATOR_SOURCE

            parts = []
            pos = 0
            for global_idx, piece in enumerate(self.pieces):
                translated = self.translations.get(global_idx)
                if translated is None or translated == piece['text']:
                    continue
                original = self.text[piece['start']:piece['end']]
                parts.append(self.text[pos:piece['start']])
                parts.append(f"{original} / {translated}" if bilingual else translated)
                pos = piece['end']
            parts.append(self.text[pos:])
            pgn_content = ''.join(parts)

            # Signature as % escape lines, which PGN readers ignore
            if ATTRIBUTION_ENABLED:
                newline = '\r\n' if '\r\n' in self.text else '\n'
                if pgn_content and not pgn_content.endswith('\n'):
                    pgn_content += newline
                pgn_content += f"% Translated with {GENERATOR_NAME}{newline}"
                pgn_content += f"% {GENERATOR_SOURCE}{newline}"

            return pgn_content.encode('utf-8')

        except Exception:
            # Fallback: return original file
            with open(self.input_file_path, 'rb') as f:
                return f.read()

    async def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> int:
        """Restore translations from checkpoint."""
        try:
            for chunk_data in checkpoint_data.get('chunks', []):
                if chunk_data.get('status') == 'completed':
                    metadata = chunk_data.get('chunk_data', {})
                    translated_text = chunk_data.get('translated_text', '')

                    if 'local_to_global' in metadata and translated_text:
                        # Convert JSON string keys back to integers
                        local_to_global = {
                            int(k): v for k, v in metadata['local_to_global'].items()
                        }
                        self._store(translated_text, local_to_global)

            return checkpoint_data.get('resume_from_index', 0)

        except Exception:
            return 0

    async def cleanup(self):
        """No cleanup needed for PGN files."""
        pass

    @property
    def format_name(self) -> str:
        """Format identifier."""
        return "pgn"
