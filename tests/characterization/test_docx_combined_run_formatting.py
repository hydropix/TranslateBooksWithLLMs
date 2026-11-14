"""A DOCX run that is both bold and italic must keep both after translation.

Found during the real-model CLI verification (gemma4:31b, English -> French):
the model returned every [idN] placeholder intact, yet the run
"additional funding" (bold + italic) came back bold only. The echo provider
reproduces it, so the loss is in the DOCX extract/reconstruct path, not in the
model. Pre-existing on main (2ecce2d) as well as on the refactor branch.
"""

import docx
import pytest

from tests.characterization import fake_llm
from tests.characterization.recorder import record_translation


def _build(path):
    document = docx.Document()
    p = document.add_paragraph("We will need ")
    run = p.add_run("additional funding")
    run.bold = True
    run.italic = True
    p.add_run(" and ")
    p.add_run("bold only").bold = True
    p.add_run(" and ")
    p.add_run("italic only").italic = True
    p.add_run(".")
    document.save(str(path))
    return path


def _runs(path):
    para = docx.Document(str(path)).paragraphs[0]
    return {r.text: (bool(r.bold), bool(r.italic)) for r in para.runs if r.text.strip()}


@pytest.fixture(autouse=True)
def _echo(monkeypatch):
    fake_llm.install(monkeypatch)


def test_bold_italic_run_survives_translation(tmp_path):
    src = _build(tmp_path / "in.docx")
    out = tmp_path / "out.docx"
    record_translation(tmp_path, src, out)

    runs = _runs(out)
    assert runs.get("bold only") == (True, False)
    assert runs.get("italic only") == (False, True)
    # Italic used to be dropped when the run was bold as well.
    assert runs.get("additional funding") == (True, True), runs
