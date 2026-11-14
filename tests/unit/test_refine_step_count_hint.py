"""The refine step of a translate + refine job explains its own chunk total.

An EPUB translated in 65 chunks can be refined in 66: the refinement pass
re-chunks the translated text. The operation label of the refine step carries
a localized tooltip saying so, marked with data-i18n-attr so a UI language
switch re-translates it.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = "refine_step_chunk_count_hint"
LOCALES = ("en", "fr", "es", "de", "zh-CN", "ja", "ko")


def test_refine_step_label_carries_a_reactive_tooltip():
    src = (ROOT / "src/web/static/js/translation/progress-manager.js").read_text(encoding="utf-8")
    assert "syncRefineCountHint(labelEl, stats)" in src
    assert f"'title:translation:{KEY}'" in src
    assert f"t('translation:{KEY}')" in src


def test_hint_exists_in_every_locale():
    for lang in LOCALES:
        data = json.loads((ROOT / f"src/web/static/locales/{lang}/translation.json")
                          .read_text(encoding="utf-8"))
        assert data.get(KEY), lang
