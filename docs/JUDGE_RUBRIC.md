# TranslateBookWithLLM — Judge Rubric (v1)

**Version:** `v1`
**Identifier to record in submissions:** `<judge-id>-rubric-v1` (e.g.
`claude-opus-4-7-rubric-v1`, `gemini-3-pro-rubric-v1`)

This document defines how a translation is scored in the benchmark. Any judge —
human, LLM-as-judge, or otherwise — must apply this rubric to make scores
comparable across submissions and over time.

If you change the rubric (new dimensions, new penalty values), bump the version
to `v2` and start a new wiki series. Never silently change the meaning of `v1`.

---

## 1. Dimensions

Each translation is scored on four dimensions, each in `[1.0, 10.0]` with
decimal precision:

| Dimension | What it measures |
|---|---|
| **accuracy** | Preservation of meaning. No additions, no omissions, no contresens, no hallucinations. |
| **fluency** | Naturalness in the target language. Grammar, syntax, idiomaticity, no untranslated source words, no script mismatches. |
| **style** | Register, tone, period vocabulary, literary voice, rhetorical devices (irony, parallelism, etc.). |
| **overall** | Holistic judgement. **Not the average.** Weighted by what matters most for this passage, with hard ceilings (see §4). |

---

## 2. Anchored scale

Scores are anchored to **professional human translation as the reference**.
This is a hard, non-negotiable framing — every judge calibrates against it.

| Score | Anchor description |
|---|---|
| **10** | Equivalent to a published reference translation by a recognized literary translator (e.g. Lydia Davis, Penguin Classics, Pléiade). **Effectively unreachable** by an LLM in current state of the art. |
| **9** | Excellent professional translation, non-literary tier. Could be published with light editing. Quibbles are stylistic, not factual. |
| **8** | Very good. The reader gets the full meaning and tone, but loses 1–2 nuances or makes a faux-sens mineur. |
| **7** | Good in the main, but 2–3 notable errors of register, idiom, or specialized terminology. A reader misses something they shouldn't. |
| **6** | Comprehensible but clearly clumsy. Multiple problematic word choices. |
| **5** | Usable only with editorial intervention. Real meaning errors or fluency breaks. |
| **4** | Significantly impaired. Frequent contresens or sentence-level breakdowns. |
| **3** | Passages incomprehensible or radically distorted. |
| **2** | Mostly wrong, hallucinations dominant. |
| **1** | Non-translation: wrong target language, source copied verbatim, refusal, or empty. |

**Hard cap at 9.0** when no reference human translation is consulted. Reserve
9.5–10 only when the judge has a published reference in front of them and the
LLM output matches or surpasses it.

---

## 3. Penalty table — start from 10, deduct

Apply these per detected issue. Multiple issues compound (additively) within a
dimension. Cap the result at the anchor scale (no negative scores).

### Accuracy penalties

| Issue | Penalty |
|---|---|
| Contresens / radical meaning reversal on a key sentence | **−2.0** |
| Hallucination of facts not in the source (place name, date, number) | **−2.0** |
| Source word/phrase left untranslated in target (e.g. English "Providence" inside Chinese) | **−1.5** |
| Significant omission (a clause or named entity dropped) | **−1.0** |
| Specialized term wrong (botanical, marine, scientific, legal, etc.) | **−0.5** to **−1.0** |
| Minor semantic drift on a non-load-bearing word | **−0.3** |

### Fluency penalties

| Issue | Penalty |
|---|---|
| Grammar error breaking the sentence | **−1.5** |
| Mixed scripts in target (traditional chars in zh-Hans, Latin word inside CJK, etc.) | **−1.0** |
| Awkward construction reading as machine-translated | **−0.5** to **−1.0** |
| Wrong target-language punctuation (e.g. English quotes in French dialogue) | **−0.3** |
| Pronoun inconsistency or unnecessary subject restatement | **−0.3** to **−0.5** |
| Calque/anglicism that has a native equivalent | **−0.5** |

### Style penalties

| Issue | Penalty |
|---|---|
| Lost a signature rhetorical device (irony, parallelism, archaic register) | **−1.0** to **−1.5** |
| Period-inappropriate vocabulary (modern slang in 19th-c. text) | **−1.0** |
| Register flattened (formal → neutral, gnomic → narrative) | **−0.5** to **−1.0** |
| Weakened idiom into bland equivalent | **−0.3** |

---

## 4. Overall — hard ceilings

`overall` is the judge's holistic call, but constrained:

- If `accuracy < 6.0` → `overall ≤ 6.0`. A translation that distorts meaning is not "good", whatever its prose.
- If `fluency < 5.0` → `overall ≤ 6.0`. An unreadable translation is not usable.
- If any dimension is `≤ 3.0` → `overall ≤ 4.0`.
- `overall` should not exceed the **minimum** of `accuracy, fluency, style` by more than 0.5. (Prevents a high overall masking a single damning weakness.)
- Without a published reference comparison, **`overall` cap is 9.0**.

---

## 5. Comparative dispersion (when judging multiple models on the same triple)

When a judge scores N translations of the **same `(text_id, target_lang)`**
triple from N different models, it must:

1. **Rank** them 1st, 2nd, ... Nth.
2. Reflect this ranking in the scores: minimum **0.3** points difference between adjacent ranks on `overall`.
3. If two outputs are genuinely indistinguishable, they may share a rank — but document this in the feedback ("tied with model X").

This forces the judge out of a flat "everything is 8" comfort zone.

---

## 6. Worked example

**Source (English, *The Picture of Dorian Gray*, 1890):**
> The studio was filled with the rich odour of roses, and when the light summer wind stirred amidst the trees of the garden, there came through the open door the heavy scent of the lilac, or the more delicate perfume of the pink-flowering thorn.

**Translation (gemma3:27b → French):**
> L'atelier était empli du riche parfum des roses, et lorsque la légère brise d'été s'agitait parmi les arbres du jardin, une lourde senteur de lilas, ou le parfum plus délicat de l'églantine rose, parvenait par la porte ouverte. Du coin du divan recouvert de sacs à dos persans sur lequel il était allongé...

**Issues:**

| Issue | Dimension | Penalty |
|---|---|---|
| `pink-flowering thorn` → `églantine rose` (wild rose, not hawthorn) | accuracy | −0.5 |
| `Persian saddle-bags` → `sacs à dos persans` (= backpacks; should be "sacoches de selle") | accuracy | −1.0 |
| `laburnum` left untranslated (FR = "cytise") | accuracy | −1.5 |
| `miel-sucrées et miel-colorées` — clumsy compound coinage | fluency | −0.5 |
| `sacs à dos` — modern term in late-19th-c. literary text | style | −0.5 |

**Scores:**
- accuracy: `10 − 0.5 − 1.0 − 1.5 = 7.0`
- fluency: `10 − 0.5 = 9.5`, but capped at 9.0 (no reference). → **9.0**. Actually one bigger issue isn't there, so 9.0 is justified. Re-check: any awkwardness? The sentence reads but the compound is clunky → **8.5**.
- style: `10 − 0.5 = 9.5` → cap at 9.0. The piece overall reads ornate. **8.5** to leave room for a better model.
- overall: floor by min(7.0, 8.5, 8.5) + 0.5 = **7.0** seems right; the botanical/material errors are noticeable in a Wilde passage where the imagery IS the point.

**Final:** `accuracy=7.0, fluency=8.5, style=8.5, overall=7.0`

---

## 7. Feedback field

Each evaluation MUST include a 1–2 sentence `feedback` documenting the
deductions. Future judges and contributors should be able to audit the score.
Examples:

- ✅ "Hallucinates 'Jongno' for 동소문; misreads 문안에 parenthetical as 'her grounds'. (−2.0 accuracy)"
- ✅ "Gnomic Stoic register flattened to plain narrative; '为所欲为' for 'Do, soul, do' is a contresens. (−2.0 acc, −1.0 style)"
- ❌ "Good translation overall." (no audit trail)
- ❌ "Captures the meaning well." (no specifics)

---

## 8. Operational notes

- **Don't average.** `overall` is a holistic call within the ceilings, not `(accuracy + fluency + style) / 3`.
- **Don't be charitable to LLM output.** If a human professional wouldn't ship it, it isn't a 9.
- **Be willing to use 5, 4, 3.** Most LLM outputs in cross-language pairs deserve 6–8. Reserve 9 for excellence.
- **Re-test yourself.** Every 50 evaluations, re-judge 3 of your earliest scores. If they drift more than ±0.5 from your original, recalibrate.

---

## 9. Versioning

- This is **v1**. Recorded as `claude-opus-4-7-rubric-v1` (or any `<judge-id>-rubric-v1`).
- A `v2` rubric must be a separate document. Wiki tables surface the rubric version next to scores.
- Submissions made under different rubric versions are aggregated separately on the wiki ("scores under rubric v1" / "scores under rubric v2"). Don't mix.
