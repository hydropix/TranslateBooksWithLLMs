"""
Live A/B/C measurement of the glossary block's target-side inflection line (issue #255).

Phases 1-3 of the plan prove the instruction *string* is emitted. They prove
nothing about whether it *works*, and the whole change is a behavioral bet on a
model. This script settles that on the reporter's own model: it translates a
committed fixture of 15 short English paragraphs three times over, once per arm,
and prints how often the model produced the glossary rendering versus the
competing established rendering.

    A (control)   glossary block built with target_language=""   - byte-identical
                  to the pre-change block, i.e. what shipped before this plan.
    B (treatment) glossary block built with target_language="Russian" - the new
                  target-side inflection line is present.
    C (baseline)  no glossary block at all, and no glossary in prompt_options.
                  Not optional: without it the A/B cannot tell "the new line
                  helps" apart from "any change to a harmful block helps". If
                  C beats B, the block itself is inducing the drift.

The arms differ in exactly one input. Every arm goes through the real prompt
path - `_build_chunk_glossary_block` then `generate_translation_prompt`, the
same two calls `_make_llm_request_with_adaptive_context` makes in
src/core/translator.py - so what is measured is what ships, not a hand-assembled
block. `target_language` is passed to `generate_translation_prompt` unchanged in
all three arms; only the value handed to `_build_chunk_glossary_block` differs.

The script asserts nothing and gates nothing. It prints. Read the result as the
plan's Phase 5 validation criteria say: C ~ A < B means ship, A ~ B means ship
on prompt-correctness grounds only and claim no improvement, C > B means stop.

METRIC
    The reporter's metric, so the numbers compare to their 24/14: per arm, count
    occurrences whose stem matches the glossary target (магл…, крестраж…) against
    the competing established Russian rendering (маггл…, хоркрукс…). The магл…
    family deliberately includes derived forms (маглорождённых, магловский),
    since D3 widened the licence specifically to cover them.

    Counting is one vote per word: the output is split into words and each word
    is attributed to the *longest* stem that prefixes it, so overlapping stems
    can never double-count. (маггл is not a substring of магл anyway - the double
    г breaks it - but hortkuluk/hortkuluğ in the Turkish set do overlap, and the
    longest-prefix rule is what keeps that honest.) Arm C is counted against the
    same targets even though it was never given them; that is the comparison of
    interest.

PROVIDER AND MODEL
    Defaults to OpenRouter with the literal `mistralai/mistral-medium-3.1`, the
    exact slug from #255. The model is HARDCODED below and is deliberately NOT
    read from OPENROUTER_MODEL: that variable holds a different model, so reading
    it would silently measure the wrong thing. `3.1` is pinned exactly -
    `mistralai/mistral-medium-3` and `mistralai/mistral-medium-3-5` are different
    deployments (the latter ~4x the price).

    `--provider` / `--model` exist for cross-checking only. A Poe Mistral bot is
    not the same deployment as the OpenRouter slug, and its numbers must not be
    reported as the #255 measurement.

NO SAMPLING KNOBS, ON PURPOSE
    No temperature, no seed, no reasoning control. The provider payload sets
    none of them, so both arms inherit the same model defaults and the same
    sampling noise, which is what keeps the comparison fair and what makes it
    match the path a real user runs. The model was measured emitting 0 reasoning
    tokens three different ways, so variable-length reasoning traces cannot
    perturb the A/B either. `--repeat` is the answer to noise here; pinning
    temperature in the harness would measure a configuration nobody runs.

Requires OPENROUTER_API_KEY in .env (already wired through
src/core/llm/factory.py). No key is read into this file and none is printed.
Never run in CI, never imported by tests/unit or tests/e2e.

Cost: 15 paragraphs x 3 arms x 3 repeats = 135 short requests, under $0.10 for a
full run at the verified mistral-medium-3.1 rates ($0.40/M prompt, $2.00/M
completion). Requests are sequential, so a full run takes several minutes. Cost
is not a reason to cut --repeat, drop arm C, or trim the fixture.

Run from repo root:
    python tests/standalone/manual_glossary_inflection.py

Cheap smoke run before spending the full budget:
    python tests/standalone/manual_glossary_inflection.py --limit 2 --repeat 1

Secondary Turkish arm (its glossary lives in the same fixture file, under
targets.turkish; it also reports final-consonant voicing, which is where "keep
the stem" would have been false):
    python tests/standalone/manual_glossary_inflection.py --target turkish

Cross-check on another deployment (NOT the #255 measurement):
    python tests/standalone/manual_glossary_inflection.py --provider poe --model Mistral-Medium
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.console import ensure_utf8_stdio

ensure_utf8_stdio()

from src.core.glossary import GlossaryConfig
from src.core.llm.factory import create_llm_provider
from src.core.translator import _build_chunk_glossary_block
from src.prompts.prompts import generate_translation_prompt

# The exact slug from issue #255. Hardcoded on purpose: OPENROUTER_MODEL holds a
# different model, and `-3` / `-3-5` are different deployments.
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "mistralai/mistral-medium-3.1"

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "glossary_inflection_ru.json"

SOURCE_LANGUAGE = "English"

# (arm id, label, does the arm get a glossary, target_language handed to the
# glossary builder). The third field is the only thing that separates A from B.
ARMS = [
    ("A", "control (no inflection line)", True, ""),
    ("B", "treatment (inflection line)", True, None),  # None -> the real target
    ("C", "baseline (no glossary)", False, ""),
]

WORD_RE = re.compile(r"\w+", re.UNICODE)


def load_fixture(target_key: str):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    targets = data["targets"]
    if target_key not in targets:
        raise SystemExit(
            f"Unknown --target {target_key!r}. Available: {', '.join(sorted(targets))}"
        )
    return data["paragraphs"], targets[target_key]


def build_stem_index(families):
    """[(stem, family_source, side)] lowercased, for longest-prefix attribution."""
    index = []
    for family in families:
        for stem in family["glossary_stems"]:
            index.append((stem.lower(), family["source"], "glossary"))
        for stem in family["competing_stems"]:
            index.append((stem.lower(), family["source"], "competing"))
    return index


def count_renderings(text: str, stem_index):
    """One vote per word, attributed to the longest stem that prefixes it.

    Structurally prevents double-counting when one stem is a prefix of another
    (hortkuluk / hortkuluğ), and keeps a word out of both sides of a family.
    """
    counts = {(source, side): 0 for _, source, side in stem_index}
    for word in WORD_RE.findall(text.lower()):
        best = None
        for stem, source, side in stem_index:
            if word.startswith(stem) and (best is None or len(stem) > len(best[0])):
                best = (stem, source, side)
        if best:
            counts[(best[1], best[2])] += 1
    return counts


def count_voicing(text: str, voicing):
    """Turkish only: correctly voiced stem vs unvoiced stem carrying a suffix."""
    if not voicing:
        return None
    unvoiced = voicing["unvoiced_stem"].lower()
    voiced = voicing["voiced_stem"].lower()
    vowels = set(voicing["suffix_vowels"])
    correct = wrong = 0
    for word in WORD_RE.findall(text.lower()):
        if word.startswith(voiced):
            correct += 1
        elif word.startswith(unvoiced) and len(word) > len(unvoiced) and word[len(unvoiced)] in vowels:
            wrong += 1
    return correct, wrong


def arm_prompt_options(spec, with_glossary: bool):
    if not with_glossary:
        # Arm C: no glossary anywhere in the prompt, not merely an empty block.
        return {}
    return {
        "glossary_terms": spec["glossary_terms"],
        "glossary_config": GlossaryConfig(),
    }


async def translate_one(provider, paragraph: str, spec, with_glossary: bool, block_language):
    """One request through the real prompt path. Returns (text, prompt_tok, completion_tok)."""
    target_language = spec["target_language"]
    prompt_options = arm_prompt_options(spec, with_glossary)
    glossary_block = _build_chunk_glossary_block(
        paragraph,
        prompt_options,
        target_language=(target_language if block_language is None else block_language),
    )
    prompt_pair = generate_translation_prompt(
        paragraph,
        "",
        source_language=SOURCE_LANGUAGE,
        target_language=target_language,
        has_placeholders=False,
        prompt_options=prompt_options,
        glossary_block=glossary_block,
    )
    response = await provider.generate(prompt_pair.user, system_prompt=prompt_pair.system)
    if response is None or not response.content:
        return None, 0, 0
    text = provider.extract_translation(response.content) or response.content.strip()
    return text, response.prompt_tokens or 0, response.completion_tokens or 0


def signature(counts, families):
    return tuple(
        (counts[(f["source"], "glossary")], counts[(f["source"], "competing")])
        for f in families
    )


def print_block_preview(spec, paragraph):
    """Show the blocks under test once, so the run is self-documenting.

    The glossary block is chunk-filtered, so this preview lists only the terms
    that occur in the first paragraph; the instruction lines are the same for
    every chunk of the run.
    """
    print("Blocks under test (built from paragraph 1; the block is chunk-filtered, "
          "the instruction lines are not):")
    print()
    for arm_id, label, with_glossary, block_language in ARMS:
        options = arm_prompt_options(spec, with_glossary)
        block = _build_chunk_glossary_block(
            paragraph,
            options,
            target_language=(spec["target_language"] if block_language is None else block_language),
        )
        print(f"--- arm {arm_id}: {label} ---")
        print(block.rstrip() if block.strip() else "(no glossary block)")
        print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        help="cross-check only; the #255 measurement is openrouter")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="cross-check only; the #255 measurement is " + DEFAULT_MODEL)
    parser.add_argument("--target", default="russian",
                        help="fixture target key: russian (default) or turkish")
    parser.add_argument("--repeat", type=int, default=3,
                        help="samples per arm per paragraph (default 3; there is no seed)")
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N paragraphs (for a cheap smoke run)")
    args = parser.parse_args()

    paragraphs, spec = load_fixture(args.target.lower())
    if args.limit:
        paragraphs = paragraphs[: args.limit]
    families = spec["families"]
    stem_index = build_stem_index(families)
    voicing = spec.get("voicing")

    total_requests = len(paragraphs) * len(ARMS) * args.repeat
    print(f"Provider     : {args.provider}")
    print(f"Model        : {args.model}")
    if args.provider != DEFAULT_PROVIDER or args.model != DEFAULT_MODEL:
        print("               NOTE: not the #255 deployment. Cross-check only - do not "
              "report these numbers as the issue #255 measurement.")
    print(f"Target       : {spec['target_language']} (fixture key '{args.target.lower()}')")
    print(f"Fixture      : {FIXTURE.name}, {len(paragraphs)} paragraph(s)")
    print(f"Repeats      : {args.repeat}")
    print(f"Requests     : {total_requests}")
    print(f"Sampling     : provider defaults (no temperature, no seed, no reasoning knob)")
    print()
    print_block_preview(spec, paragraphs[0])

    provider = create_llm_provider(args.provider, model=args.model)

    # outputs[(arm, paragraph_index, repeat)] = text or None
    outputs = {}
    totals = {arm: {(f["source"], side): 0 for f in families for side in ("glossary", "competing")}
              for arm, _, _, _ in ARMS}
    per_repeat = {arm: [[0, 0] for _ in range(args.repeat)] for arm, _, _, _ in ARMS}
    voicing_totals = {arm: [0, 0] for arm, _, _, _ in ARMS}
    tokens = {"prompt": 0, "completion": 0}
    failures = []

    try:
        for repeat in range(args.repeat):
            for index, paragraph in enumerate(paragraphs):
                line = [f"repeat {repeat + 1}/{args.repeat}  para {index + 1:2d}"]
                for arm_id, _, with_glossary, block_language in ARMS:
                    text, ptok, ctok = await translate_one(
                        provider, paragraph, spec, with_glossary, block_language
                    )
                    tokens["prompt"] += ptok
                    tokens["completion"] += ctok
                    outputs[(arm_id, index, repeat)] = text
                    if text is None:
                        failures.append((arm_id, index, repeat))
                        line.append(f"{arm_id}: FAILED")
                        continue
                    counts = count_renderings(text, stem_index)
                    for key, value in counts.items():
                        totals[arm_id][key] += value
                    glossary_hits = sum(v for (_, side), v in counts.items() if side == "glossary")
                    competing_hits = sum(v for (_, side), v in counts.items() if side == "competing")
                    per_repeat[arm_id][repeat][0] += glossary_hits
                    per_repeat[arm_id][repeat][1] += competing_hits
                    if voicing:
                        ok, bad = count_voicing(text, voicing)
                        voicing_totals[arm_id][0] += ok
                        voicing_totals[arm_id][1] += bad
                    line.append(f"{arm_id}: {glossary_hits}/{competing_hits}")
                print("  ".join(line))
    finally:
        await provider.close()

    print()
    print("=" * 78)
    print("THREE-ARM TABLE  (counts over the whole run; glossary target vs competing rendering)")
    print("=" * 78)
    header = f"{'family':<10}{'rendering':<28}" + "".join(f"{a:>12}" for a, _, _, _ in ARMS)
    print(header)
    for arm_id, label, _, _ in ARMS:
        print(f"{'':<38}{arm_id} = {label}")
    print("-" * 78)
    for family in families:
        for side, label_key in (("glossary", "glossary_label"), ("competing", "competing_label")):
            cells = "".join(f"{totals[a][(family['source'], side)]:>12}" for a, _, _, _ in ARMS)
            print(f"{family['source']:<10}{family[label_key]:<28}{cells}")
    print("-" * 78)
    grand = {}
    for arm_id, _, _, _ in ARMS:
        g = sum(v for (_, side), v in totals[arm_id].items() if side == "glossary")
        c = sum(v for (_, side), v in totals[arm_id].items() if side == "competing")
        grand[arm_id] = (g, c)
    print(f"{'TOTAL':<10}{'glossary rendering':<28}" +
          "".join(f"{grand[a][0]:>12}" for a, _, _, _ in ARMS))
    print(f"{'TOTAL':<10}{'competing rendering':<28}" +
          "".join(f"{grand[a][1]:>12}" for a, _, _, _ in ARMS))
    shares = []
    for arm_id, _, _, _ in ARMS:
        g, c = grand[arm_id]
        shares.append(f"{(100.0 * g / (g + c)):>11.1f}%" if (g + c) else f"{'n/a':>12}")
    print(f"{'TOTAL':<10}{'glossary share':<28}" + "".join(shares))

    if voicing:
        print()
        print("Turkish final-consonant voicing (D3's counter-example to 'keep the stem'):")
        print(f"{'':<38}" + "".join(f"{a:>12}" for a, _, _, _ in ARMS))
        print(f"{'':<10}{'voiced ' + voicing['voiced_stem'] + '…':<28}" +
              "".join(f"{voicing_totals[a][0]:>12}" for a, _, _, _ in ARMS))
        print(f"{'':<10}{'unvoiced ' + voicing['unvoiced_stem'] + '+vowel':<28}" +
              "".join(f"{voicing_totals[a][1]:>12}" for a, _, _, _ in ARMS))

    print()
    print("Per repeat (glossary/competing), so the sampling noise is visible:")
    for arm_id, _, _, _ in ARMS:
        cells = "  ".join(f"r{i + 1}: {g}/{c}" for i, (g, c) in enumerate(per_repeat[arm_id]))
        print(f"  {arm_id}  {cells}")

    print()
    print(f"Tokens: prompt={tokens['prompt']} completion={tokens['completion']}  "
          f"(~${tokens['prompt'] * 0.40e-6 + tokens['completion'] * 2.00e-6:.4f} at "
          f"mistral-medium-3.1 rates)")
    if failures:
        print(f"Failed requests: {len(failures)} -> {failures}")

    print()
    print("=" * 78)
    print("PARAGRAPHS WHERE THE ARMS DISAGREE  (raw output, for human judgement)")
    print("=" * 78)
    disagreements = 0
    for repeat in range(args.repeat):
        for index, paragraph in enumerate(paragraphs):
            texts = {a: outputs.get((a, index, repeat)) for a, _, _, _ in ARMS}
            sigs = {
                a: signature(count_renderings(t, stem_index), families) if t else None
                for a, t in texts.items()
            }
            if len(set(sigs.values())) == 1:
                continue
            disagreements += 1
            print()
            print(f"[repeat {repeat + 1}, paragraph {index + 1}]  EN: {paragraph}")
            for arm_id, _, _, _ in ARMS:
                text = texts[arm_id]
                print(f"  {arm_id}: {text.strip() if text else '(no response)'}")
    if not disagreements:
        print()
        print("(none - every arm produced the same rendering counts on every paragraph)")

    print()
    print("Read the table with the plan's Phase 5 criteria: C ~ A < B means the block was "
          "diluting an ability the model already had and the new line restores it; A ~ B means "
          "the line does nothing measurable, ship on prompt-correctness grounds only; C > B "
          "means the glossary block itself harms terminology consistency - stop and re-open the "
          "question. Do not re-roll runs until a favorable one appears.")


asyncio.run(main())
