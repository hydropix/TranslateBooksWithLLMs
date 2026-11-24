"""
The failure report: a plain-text list of the chunks a job left behind (issue #232).

Opt-in (``JobConfig.failure_report``). When a translation job finishes, the
runner writes ``<output stem>.report.txt`` next to the output file. It names,
per source file, the chunks that kept their source text and the chunks whose
inline tags were only approximately repositioned, so a reader can jump to them
(in Calibre, for an EPUB) instead of scrolling the job log.

Nothing is re-detected here. The report only reads what the pipeline already
recorded:

- EPUB: the job stats' ``unfinished_files`` / ``degraded_files`` maps
  (``{file_href: [chunk_index, ...]}``), refined by the per-chunk statuses of
  each file's XHTML partial state, which is kept on disk exactly for the files
  that appear in those maps. The partial state also supplies the chunk's source
  text for a short, searchable snippet.
- Other formats: the checkpoint chunks stored with status ``failed`` and the
  ``raw_fallback_used`` counter (whose unit positions are not recorded).

The data gathering (``collect_*``) is separated from the rendering
(:func:`render_report`) so the latter stays a pure function.
"""

from __future__ import annotations

import os
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote

REPORT_SUFFIX = '.report.txt'

#: Longest snippet / chapter title quoted in the report.
SNIPPET_LENGTH = 70
TITLE_LENGTH = 60

_WHITESPACE_RE = re.compile(r'\s+')
_HEADINGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')


def report_path_for(output_path: str) -> str:
    """``<output stem>.report.txt``, next to the output file."""
    stem, _ext = os.path.splitext(output_path)
    return f"{stem}{REPORT_SUFFIX}"


# -- data ------------------------------------------------------------------


@dataclass
class FileIssues:
    """What one EPUB document still carries. Chunk indices are 0-based.

    Attributes:
        href: The document's href, as listed in the OPF manifest.
        untranslated: Chunks that fell back to their source text.
        not_reached: Chunks never attempted (the job stopped first).
        token_aligned: Translated chunks with approximate tag positions.
        total_chunks: Chunk count of the document, when known.
        snippets: Source text excerpt per chunk index, when known.
        position: 1-based position of the document in reading order.
        title: First heading (or ``<title>``) of the translated document.
    """

    href: str
    untranslated: List[int] = field(default_factory=list)
    not_reached: List[int] = field(default_factory=list)
    token_aligned: List[int] = field(default_factory=list)
    total_chunks: Optional[int] = None
    snippets: Dict[int, str] = field(default_factory=dict)
    position: Optional[int] = None
    title: Optional[str] = None

    @property
    def empty(self) -> bool:
        return not (self.untranslated or self.not_reached or self.token_aligned)


@dataclass
class UnitIssue:
    """A failed unit of a non-EPUB job (0-based index)."""

    index: int
    snippet: str = ''


def _units_map(value: Any) -> Dict[str, List[int]]:
    """A ``{href: [index, ...]}`` stats map, cleaned of anything malformed."""
    if not isinstance(value, Mapping):
        return {}
    cleaned: Dict[str, List[int]] = {}
    for href, indices in value.items():
        if not isinstance(indices, (list, tuple)):
            continue
        ints = sorted({int(i) for i in indices if isinstance(i, int) and i >= 0})
        if ints:
            cleaned[str(href)] = ints
    return cleaned


def make_snippet(text: Any, placeholder_format: Optional[Sequence[str]] = None,
                 length: int = SNIPPET_LENGTH) -> str:
    """One line of a chunk's text, placeholders removed, truncated."""
    if not isinstance(text, str):
        return ''
    if placeholder_format and len(placeholder_format) == 2 and all(placeholder_format):
        prefix, suffix = placeholder_format
        text = re.sub(re.escape(prefix) + r'\w+' + re.escape(suffix), ' ', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    if len(text) > length:
        text = text[:length - 1].rstrip() + '…'
    return text


def collect_epub_issues(stats: Mapping[str, Any],
                        load_state: Optional[Callable[[str], Any]] = None) -> List[FileIssues]:
    """Per-file issues of an EPUB job, from its stats and partial states.

    Args:
        stats: The job's merged stats (``unfinished_files`` /
            ``degraded_files``).
        load_state: ``href -> XHTMLTranslationState | None``. When a state
            with per-chunk statuses is found it is authoritative for that
            file (it tells a fallback from a chunk never reached, and gives
            the source text). Without it, the stats maps are used as they are.

    Returns:
        One entry per file with something to report, in href order (callers
        re-sort by reading order once it is known).
    """
    from src.core.epub.xhtml_translation_state import (
        CHUNK_PENDING,
        token_aligned_chunk_indices,
        untranslated_chunk_indices,
    )

    unfinished = _units_map(stats.get('unfinished_files'))
    degraded = _units_map(stats.get('degraded_files'))
    entries: List[FileIssues] = []
    for href in sorted(set(unfinished) | set(degraded)):
        entry = FileIssues(href=href)
        state = None
        if load_state is not None:
            try:
                state = load_state(href)
            except Exception:
                state = None
        statuses = getattr(state, 'chunk_statuses', None) if state is not None else None
        if statuses:
            entry.untranslated = untranslated_chunk_indices(statuses)
            entry.not_reached = [i for i, s in enumerate(statuses) if s == CHUNK_PENDING]
            entry.token_aligned = token_aligned_chunk_indices(statuses)
            entry.total_chunks = len(statuses)
            chunks = getattr(state, 'chunks', None) or []
            placeholder_format = getattr(state, 'placeholder_format', None)
            for i in entry.untranslated + entry.not_reached + entry.token_aligned:
                if i < len(chunks) and isinstance(chunks[i], Mapping):
                    snippet = make_snippet(chunks[i].get('text'), placeholder_format)
                    if snippet:
                        entry.snippets[i] = snippet
        else:
            entry.untranslated = list(unfinished.get(href, []))
            entry.token_aligned = list(degraded.get(href, []))
        if not entry.empty:
            entries.append(entry)
    return entries


def _local(tag: Any) -> str:
    return tag.rsplit('}', 1)[-1].lower() if isinstance(tag, str) else ''


def _element_text(element) -> str:
    return _WHITESPACE_RE.sub(' ', ''.join(element.itertext())).strip()


def epub_chapter_info(epub_path: str) -> Tuple[Dict[str, Tuple[int, Optional[str]]], int]:
    """Reading-order position and title of every spine document of an EPUB.

    Best effort: any unreadable part simply yields no position / no title.

    Returns:
        ``({href: (1-based position, title or None)}, spine length)``.
    """
    from lxml import etree, html as lxml_html

    info: Dict[str, Tuple[int, Optional[str]]] = {}
    try:
        with zipfile.ZipFile(epub_path) as archive:
            container = etree.fromstring(archive.read('META-INF/container.xml'))
            rootfile = next((el.get('full-path') for el in container.iter()
                             if _local(el.tag) == 'rootfile' and el.get('full-path')), None)
            if not rootfile:
                return info, 0
            opf = etree.fromstring(archive.read(rootfile))
            opf_dir = posixpath.dirname(rootfile)
            manifest = {el.get('id'): (el.get('href'), el.get('media-type'))
                        for el in opf.iter() if _local(el.tag) == 'item'}
            spine = [manifest.get(el.get('idref')) for el in opf.iter()
                     if _local(el.tag) == 'itemref']
            hrefs = [item[0] for item in spine
                     if item and item[0] and item[1] in ('application/xhtml+xml', 'text/html')]
            names = set(archive.namelist())
            for position, href in enumerate(hrefs, start=1):
                title = None
                member = posixpath.normpath(posixpath.join(opf_dir, unquote(href)))
                if member in names:
                    try:
                        doc = lxml_html.fromstring(archive.read(member))
                        title = _document_title(doc)
                    except Exception:
                        title = None
                info[href] = (position, title)
            return info, len(hrefs)
    except Exception:
        return info, 0


def _document_title(doc) -> Optional[str]:
    fallback = None
    for element in doc.iter():
        name = _local(element.tag)
        if name in _HEADINGS:
            text = _element_text(element)
            if text:
                return make_snippet(text, length=TITLE_LENGTH)
        elif name == 'title' and fallback is None:
            fallback = _element_text(element) or None
    return make_snippet(fallback, length=TITLE_LENGTH) if fallback else None


def collect_unit_issues(chunks: Iterable[Mapping[str, Any]]) -> List[UnitIssue]:
    """Failed units of a non-EPUB job, from its checkpoint chunk rows."""
    issues = []
    for chunk in chunks or ():
        if chunk.get('status') != 'failed':
            continue
        try:
            index = int(chunk.get('chunk_index'))
        except (TypeError, ValueError):
            continue
        issues.append(UnitIssue(index=index, snippet=make_snippet(chunk.get('original_text'))))
    return sorted(issues, key=lambda issue: issue.index)


# -- rendering -------------------------------------------------------------

_LABEL_UNTRANSLATED = 'UNTRANSLATED'
_LABEL_NOT_REACHED = 'NOT REACHED'
_LABEL_TOKEN_ALIGNED = 'APPROX. TAGS'
_LABEL_FAILED = 'FAILED'
_LABEL_WIDTH = max(len(_LABEL_UNTRANSLATED), len(_LABEL_NOT_REACHED),
                   len(_LABEL_TOKEN_ALIGNED), len(_LABEL_FAILED))


def _chunk_line(label: str, index: int, total: Optional[int], snippet: str) -> str:
    number = f"chunk {index + 1}" + (f"/{total}" if total else "")
    line = f"    {label:<{_LABEL_WIDTH}}  {number:<14}"
    if snippet:
        line += f'  "{snippet}"'
    return line.rstrip()


def render_report(*, output_name: str, source_name: Optional[str] = None,
                  status: str = '', model: Optional[str] = None,
                  file_type: str = 'txt',
                  epub_files: Optional[Sequence[FileIssues]] = None,
                  spine_total: int = 0,
                  failed_units: Optional[Sequence[UnitIssue]] = None,
                  raw_fallback_count: int = 0,
                  generated_at: Optional[datetime] = None) -> str:
    """The report text. Pure: everything it prints is passed in."""
    generated_at = generated_at or datetime.now()
    lines = ["TranslateBook failure report", "=" * 28, ""]
    lines.append(f"Output:    {output_name}")
    if source_name:
        lines.append(f"Source:    {source_name}")
    if status:
        lines.append(f"Status:    {status}")
    if model:
        lines.append(f"Model:     {model}")
    lines.append(f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    if file_type == 'epub':
        lines.extend(_render_epub(epub_files or (), spine_total))
    else:
        lines.extend(_render_units(failed_units or (), raw_fallback_count))
    return "\n".join(lines).rstrip() + "\n"


def _render_epub(files: Sequence[FileIssues], spine_total: int) -> List[str]:
    files = [f for f in files if not f.empty]
    if not files:
        return ["No untranslated chunks and no chunks with approximate tag placement",
                "were recorded for this job."]
    untranslated = sum(len(f.untranslated) for f in files)
    not_reached = sum(len(f.not_reached) for f in files)
    aligned = sum(len(f.token_aligned) for f in files)
    parts = []
    if untranslated:
        parts.append(f"{untranslated} untranslated chunk(s)")
    if not_reached:
        parts.append(f"{not_reached} chunk(s) not reached")
    if aligned:
        parts.append(f"{aligned} chunk(s) with approximate tag placement")
    lines = [f"Summary: {', '.join(parts)} in {len(files)} file(s).", "",
             "Legend:"]
    if untranslated:
        lines.append(f"  {_LABEL_UNTRANSLATED:<{_LABEL_WIDTH}}  the translation was rejected; the source text was kept")
    if not_reached:
        lines.append(f"  {_LABEL_NOT_REACHED:<{_LABEL_WIDTH}}  the job stopped before this chunk was translated")
    if aligned:
        lines.append(f"  {_LABEL_TOKEN_ALIGNED:<{_LABEL_WIDTH}}  translated, but inline formatting (italics, links, ...)")
        lines.append(f"  {'':<{_LABEL_WIDTH}}  was repositioned approximately")
    lines += ["",
              "Files are listed in reading order with their path inside the EPUB",
              "(the name shown by Calibre's editor). Chunks count from 1 within each",
              "file; the quoted text is the start of the chunk's source text.",
              ""]

    def order(entry: FileIssues):
        return (entry.position is None, entry.position or 0, entry.href)

    for entry in sorted(files, key=order):
        if entry.position is not None:
            where = f"[{entry.position}/{spine_total}] " if spine_total else f"[{entry.position}] "
        else:
            where = ""
        header = f"{where}{entry.href}"
        if entry.title:
            header += f'  "{entry.title}"'
        lines.append(header)
        rows = ([(i, _LABEL_UNTRANSLATED) for i in entry.untranslated]
                + [(i, _LABEL_NOT_REACHED) for i in entry.not_reached]
                + [(i, _LABEL_TOKEN_ALIGNED) for i in entry.token_aligned])
        for index, label in sorted(rows):
            lines.append(_chunk_line(label, index, entry.total_chunks,
                                     entry.snippets.get(index, '')))
        lines.append("")
    return lines


def _render_units(failed: Sequence[UnitIssue], raw_fallback_count: int) -> List[str]:
    if not failed and raw_fallback_count <= 0:
        return ["No failed chunks were recorded for this job."]
    lines = []
    if failed:
        lines += [f"Summary: {len(failed)} chunk(s) failed and were left untranslated or",
                  "only partly translated in the output.", "",
                  "Chunks count from 1; the quoted text is the start of the chunk's source text.",
                  ""]
        for issue in failed:
            lines.append(_chunk_line(_LABEL_FAILED, issue.index, None, issue.snippet))
        lines.append("")
    if raw_fallback_count > 0:
        lines.append(f"{raw_fallback_count} chunk(s) came back from the model without a usable "
                     "translation and kept")
        lines.append("their source text; their positions are listed in the job log.")
    return lines


# -- entry point -------------------------------------------------------------


def write_failure_report(job, status: str, output_path: str, stats: Mapping[str, Any],
                         checkpoint_manager=None, translation_id: Optional[str] = None,
                         report_path: Optional[str] = None) -> str:
    """Gather the recorded issues of a finished job and write the report.

    Args:
        job: The :class:`JobConfig` (file type, names, model).
        status: The job's final status, printed in the header.
        output_path: The file the job wrote; the report goes next to it.
        stats: The job's merged stats.
        checkpoint_manager / translation_id: Where the per-file partial states
            (EPUB) and the failed checkpoint chunks (other formats) are read.
            Must be called before the checkpoint is pruned.
        report_path: Overrides :func:`report_path_for`.

    Returns:
        The path of the written report.
    """
    file_type = (job.file_type or 'txt').lower()
    epub_files: List[FileIssues] = []
    spine_total = 0
    failed_units: List[UnitIssue] = []
    if file_type == 'epub':
        load_state = None
        if checkpoint_manager is not None and translation_id:
            def load_state(href):
                return checkpoint_manager.load_xhtml_partial_state(translation_id, href)
        epub_files = collect_epub_issues(stats, load_state)
        if epub_files:
            chapters, spine_total = epub_chapter_info(output_path)
            for entry in epub_files:
                position, title = chapters.get(entry.href, (None, None))
                entry.position, entry.title = position, title
    elif checkpoint_manager is not None and translation_id:
        db = getattr(checkpoint_manager, 'db', None)
        if db is not None:
            failed_units = collect_unit_issues(db.get_chunks(translation_id))

    try:
        raw_fallback = max(0, int(stats.get('raw_fallback_used') or 0))
    except (TypeError, ValueError):
        raw_fallback = 0

    text = render_report(
        output_name=os.path.basename(output_path),
        source_name=job.input_filename,
        status=status,
        model=f"{job.llm_provider} / {job.model}" if job.model else None,
        file_type=file_type,
        epub_files=epub_files,
        spine_total=spine_total,
        failed_units=failed_units,
        raw_fallback_count=raw_fallback,
    )
    path = report_path or report_path_for(output_path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path
