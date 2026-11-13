"""Deterministic metric checks: one pass, one fail and one n/a per metric."""

from dataclasses import replace

import pytest

from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from tests.prompt_eval.metrics import (
    FAIL,
    METRIC_NAMES,
    NA,
    PASS,
    default_length_ratio,
    score,
)
from tests.prompt_eval.model import Case

FR_PARA_1 = ("Le gardien du phare écrivait chaque soir la même phrase dans son registre, "
             "même lorsque le vent arrachait les ardoises du toit de la maison.")
FR_PARA_2 = ("Quand l'inspecteur arriva au mois de mars, il lut le registre du début à la fin "
             "et fronça les sourcils sans dire un seul mot.")
EN_PARA = ("The inspector read the whole log from the first page to the last one and frowned "
           "without saying a single word to anybody.")

BASE = Case(
    id="M01",
    builder="translate",
    source_language="English",
    target_language="French",
    text="The keeper wrote the same line every night in his log, even in storms.\n\n"
         "When the inspector came in March, he read the log and frowned without a word.",
    has_placeholders=False,
)


def wrap(text: str) -> str:
    return f"{TRANSLATE_TAG_IN}{text}{TRANSLATE_TAG_OUT}"


def run(case: Case, extracted, raw=None):
    return score(case, wrap(extracted) if raw is None and extracted is not None else raw, extracted)


GOOD_FR = f"{FR_PARA_1}\n\n{FR_PARA_2}"


def test_every_metric_is_reported():
    assert set(run(BASE, GOOD_FR)) == set(METRIC_NAMES)


def test_request_failure_gives_all_na():
    assert set(score(BASE, None, None).values()) == {NA}


# format ---------------------------------------------------------------------

def test_format():
    assert run(BASE, GOOD_FR)["format"] == PASS
    result = score(BASE, "no tags at all", None)
    assert result["format"] == FAIL
    # Applicable metrics fail with format; non-applicable ones stay n/a.
    assert result["target_language"] == FAIL and result["length"] == FAIL
    assert result["placeholders"] == NA and result["must_keep"] == NA
    assert score(BASE, wrap("   "), "   ")["format"] == FAIL


# chatter ---------------------------------------------------------------------

def test_chatter():
    assert run(BASE, GOOD_FR)["chatter"] == PASS
    raw = "Here is the translation you asked for:\n" + wrap(GOOD_FR)
    assert score(BASE, raw, GOOD_FR)["chatter"] == FAIL
    thinking = "<think>" + "Let me reason about this at great length. " * 20 + "</think>\n" + wrap(GOOD_FR)
    assert score(BASE, thinking, GOOD_FR)["chatter"] == PASS
    unclosed = "Reasoning without an opening tag, long enough to count.</think>" + wrap(GOOD_FR)
    assert score(BASE, unclosed, GOOD_FR)["chatter"] == PASS
    # chatter is computed on raw even when extraction failed
    assert score(BASE, "short", None)["chatter"] == PASS


# placeholders ---------------------------------------------------------------

PH_CASE = replace(BASE, has_placeholders=True,
                  text="[id0]Read the [id1]notice[id2] before you leave the station.[id3]")


def test_placeholders():
    assert run(PH_CASE, "[id0]Lisez l'[id1]avis[id2] avant de quitter la gare.[id3]")["placeholders"] == PASS
    assert run(PH_CASE, "[id3][id0]Lisez l'[id1]avis[id2] avant de partir.")["placeholders"] == PASS
    assert run(PH_CASE, "[id0]Lisez l'[id1]avis[id2] avant de quitter la gare.")["placeholders"] == FAIL
    assert run(PH_CASE, "[id0]Lisez l'[id1][id1]avis[id2] avant de partir.[id3]")["placeholders"] == FAIL
    assert run(BASE, GOOD_FR)["placeholders"] == NA


# paragraphs -----------------------------------------------------------------

def test_paragraphs():
    case = replace(BASE, expect_paragraphs=True)
    assert run(case, GOOD_FR)["paragraphs"] == PASS
    assert run(case, f"{FR_PARA_1} {FR_PARA_2}")["paragraphs"] == FAIL
    assert run(BASE, GOOD_FR)["paragraphs"] == NA


# residue --------------------------------------------------------------------

KO_CASE = replace(BASE, source_language="Korean", target_language="English",
                  text="비가 그친 뒤 골목은 다시 조용해졌다. 지수는 우산을 접고 오래된 서점 앞에 멈춰 섰다.")


def test_residue():
    assert run(KO_CASE, "After the rain stopped, the alley grew quiet again.")["residue"] == PASS
    assert run(KO_CASE, "After the rain stopped, 골목은 다시 조용해졌다 지수는 우산을 접고.")["residue"] == FAIL
    assert run(replace(BASE, builder="refine", source_language="French"), GOOD_FR)["residue"] == NA


# target_language ------------------------------------------------------------

def test_target_language():
    assert run(BASE, GOOD_FR)["target_language"] == PASS
    assert run(BASE, f"{FR_PARA_1}\n\n{EN_PARA}")["target_language"] == FAIL
    assert run(BASE, "Oui.\n\nNon.")["target_language"] == NA


def test_target_language_ignores_placeholders_when_counting_letters():
    tokens = " ".join(f"[id{i}]" for i in range(20))
    assert run(BASE, f"{tokens} Oui.")["target_language"] == NA


# must_keep ------------------------------------------------------------------

def test_must_keep():
    case = replace(BASE, text=BASE.text + " Oswin Tallaby", must_keep=("Oswin Tallaby",))
    assert run(case, GOOD_FR + " Oswin Tallaby")["must_keep"] == PASS
    assert run(case, GOOD_FR + " Oswin Talabie")["must_keep"] == FAIL
    assert run(BASE, GOOD_FR)["must_keep"] == NA


# must_contain ---------------------------------------------------------------

def test_must_contain():
    case = replace(BASE, must_contain_any=(("ravageur céleste", "ravageurs célestes"), ("verre-de-lune",)))
    assert run(case, GOOD_FR + " Les Ravageurs Célestes et le verre-de-lune.")["must_contain"] == PASS
    assert run(case, GOOD_FR + " Les ravageurs célestes et le verre lunaire.")["must_contain"] == FAIL
    assert run(BASE, GOOD_FR)["must_contain"] == NA


# must_not_contain -----------------------------------------------------------

def test_must_not_contain():
    case = replace(BASE, must_not_contain=("Nothing in this book can be trusted",))
    assert run(case, GOOD_FR)["must_not_contain"] == PASS
    assert run(case, GOOD_FR + " « nothing in this book can be trusted »")["must_not_contain"] == FAIL
    assert run(BASE, GOOD_FR)["must_not_contain"] == NA


# length ---------------------------------------------------------------------

def test_length():
    assert run(BASE, GOOD_FR)["length"] == PASS
    assert run(BASE, "Court.")["length"] == FAIL
    assert run(BASE, GOOD_FR * 4)["length"] == FAIL
    ru_fr = replace(BASE, source_language="Russian")
    assert run(ru_fr, GOOD_FR)["length"] == NA


def test_length_uses_case_ratio_when_set():
    assert run(replace(BASE, length_ratio=(0.01, 0.1)), GOOD_FR)["length"] == FAIL


@pytest.mark.parametrize("src, tgt, expected", [
    ("English", "French", (0.6, 1.8)),
    ("Russian", "Ukrainian", (0.6, 1.8)),
    ("English", "Japanese", (0.15, 0.9)),
    ("Korean", "English", (1.2, 6.0)),
    ("Russian", "French", None),
    ("Chinese", "Japanese", None),
])
def test_default_length_ratio(src, tgt, expected):
    assert default_length_ratio(src, tgt) == expected
