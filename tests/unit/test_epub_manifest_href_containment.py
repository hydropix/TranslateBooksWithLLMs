"""A crafted EPUB must not make the engine read or overwrite files outside its
extraction directory.

safe_extract_zip() refuses archive entries such as '../../evil.txt', but the
OPF manifest hrefs are attacker-controlled too. resolve_content_path() joins
them to the OPF directory with no containment check, so a spine item whose
href is absolute ('C:/Users/x/notes.xhtml') or climbs out with '../' is read
from the host, "translated", and written back in place by
save_translated_files(): an EPUB upload becomes a write primitive on any
existing XHTML/HTML file the server process can reach. Same code on main
(src/core/epub/translator.py), so this is pre-existing, not a refactor
regression.
"""
import asyncio
import zipfile
from pathlib import Path

import pytest

from src.core.job import JobConfig
from src.core.job.runner import JobRunner, JobSinks
from src.core.llm.provider_config import ProviderConfig
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm

VICTIM_XHTML = (
    "<html xmlns='http://www.w3.org/1999/xhtml'><head><title>t</title></head>"
    "<body><p>Private notes that must never be touched by a translation job.</p></body></html>"
)

CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

OPF = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">x</dc:identifier><dc:title>T</dc:title><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="evil" href="{href}" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/><itemref idref="evil"/></spine>
</package>"""

CH1 = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>c</title></head>
<body><p>The quick brown fox jumps over the lazy dog.</p></body></html>"""


class _Sinks(JobSinks):
    def log(self, key, message="", data=None):
        pass

    def emit_stats(self, stats):
        pass

    def notify(self, event, context):
        pass


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    fake_llm.install(monkeypatch)


def _build_epub(path: Path, href: str) -> Path:
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
        z.writestr('META-INF/container.xml', CONTAINER)
        z.writestr('OEBPS/content.opf', OPF.format(href=href))
        z.writestr('OEBPS/ch1.xhtml', CH1)
    return path


def _hrefs(victim: Path):
    absolute = victim.resolve().as_posix()                     # 'C:/Users/.../notes.xhtml'
    climbing = '../' * 40 + absolute.split(':', 1)[-1].lstrip('/')  # '../../..../Users/.../notes.xhtml'
    return {'absolute': absolute, 'dotdot': climbing}


@pytest.mark.parametrize('kind', ['absolute', 'dotdot'])
def test_manifest_href_cannot_reach_files_outside_the_book(tmp_path, monkeypatch, kind):
    # The engine keeps partial state under the relative 'data/uploads'; run
    # in tmp_path so nothing lands in the repository.
    monkeypatch.chdir(tmp_path)
    victim_dir = tmp_path / 'victim'
    victim_dir.mkdir()
    victim = victim_dir / 'notes.xhtml'
    victim.write_text(VICTIM_XHTML, encoding='utf-8')
    before = victim.read_bytes()

    work = tmp_path / 'work'
    work.mkdir()
    source = _build_epub(work / 'book.epub', _hrefs(victim)[kind])
    manager = CheckpointManager(db_path=str(work / 'jobs.db'))
    job = JobConfig(
        file_type='epub', input_path=str(source), output_path=str(work / 'out.epub'),
        source_language='English', target_language='French',
        provider=ProviderConfig(provider='poe', model='fake-echo', api_key='UNUSED_FAKE_KEY'),
        context_window=4096, auto_adjust_context=False, max_tokens_per_chunk=200, origin='cli',
    )
    asyncio.run(JobRunner(job, translation_id=f'cli_href_{kind}', checkpoint_manager=manager,
                          sinks=_Sinks()).run())

    assert victim.read_bytes() == before, 'a file outside the extracted book was rewritten'
    # Nor may its text end up in the output book.
    out = work / 'out.epub'
    if out.exists():
        with zipfile.ZipFile(out) as z:
            blob = b''.join(z.read(n) for n in z.namelist())
        assert b'Private notes' not in blob


@pytest.mark.parametrize('href', ['C:/Users/x/notes.xhtml', 'C:notes.xhtml', 'D:..' + chr(92) + 'x.xhtml',
                                  '../../../../x.xhtml'])
def test_xhtml_state_path_stays_under_the_job_folder(tmp_path, monkeypatch, href):
    """The partial-state file name only replaced '/' and '\'. A drive
    prefix ('C:') survived, and pathlib treats 'C:_Users_..json' as a
    drive-relative path: the state file was written to the process working
    directory (observed: the repository root) instead of
    data/uploads/<id>/xhtml_states/."""
    monkeypatch.chdir(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / 'jobs.db'))
    state_dir = (tmp_path / 'data' / 'uploads' / 'job1' / 'xhtml_states').resolve()
    path = Path(manager._xhtml_state_path('job1', href)).resolve()
    assert path.parent == state_dir, path
