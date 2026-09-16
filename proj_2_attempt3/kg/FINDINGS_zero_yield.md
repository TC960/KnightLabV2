# The extractor's silence: 42 papers that produced no edge

**2026-09-16. Cloud, CPU-only, no `MAIN_DATA.json`, no taxdump.**
Instrument: `zero_yield_audit.py` → `zero_yield.json`, `zero_yield_verdicts.json`.

> ## ⚠️ CORRECTION, same session, before this was quoted anywhere
>
> **The first version of this document claimed 4 confirmed misses and a headline
> paper-level recall of 98.5%. That was wrong, and the reason is worth more than the
> number was.** The adjudication was run without showing the adjudicators the
> extraction prompt's actual inclusion rules (`eval-v2/run_eval.py`,
> `PROMPT_VERSION = "samgated-v1"`), which are stricter than "states a direction":
>
> - **SIGNIFICANCE** — "include a taxon ONLY if the paper reports it as statistically
>   significant (p / FDR / q < 0.05, or a significant LEfSe/LDA or
>   differential-abundance result). **If significance is unclear or unreported for a
>   taxon, omit it.**"
> - **MAIN TEXT ONLY** — ignore taxa appearing only in tables, figures, captions or
>   supplementary material.
> - **DISEASE vs HEALTHY CONTROL ONLY** — ignore disease-vs-disease, severity
>   gradients, subgroup-vs-subgroup and symptom-correlation findings.
>
> Re-tested against that gate, **3 of the 4 "misses" state a direction with no
> reported significance anywhere in the available sentences** — one of them says
> outright *"a **tendency towards** a reduction"*, and one carries a citation marker
> (`[ 32 ]`) that makes it background. Under the prompt's own rule those are
> **correct refusals, not misses.** Only the stroke/depression paper, which carries
> LDA values and p-values inline (*Roseburia* LDA 3.894, P=0.007), is confirmable.
>
> **Confirmed misses: 1, not 4.** And a second-order limit applies even to that:
> `relation_sentences_clean.json` keeps only sentences carrying a taxon *and* a
> direction cue — about 10% of corpus text — so a sentence reporting significance
> without a direction word is not visible here. **This audit can therefore confirm a
> miss but cannot refute one.** The honest output is a bound, not a point estimate:
>
> | reading | recall | 95% CI |
> |---|---|---|
> | confirmed miss only | **271/272 = 99.6%** | [97.9, 99.9] |
> | + 3 direction-without-reported-significance | 271/275 = 98.5% | [96.3, 99.4] |
> | + 7 unclear (worst case) | 271/282 = 96.1% | [93.2, 97.8] |
>
> **Quote the range, or quote ≥96.1% as the defensible floor. Do not quote 98.5% as
> the headline** — that row assumes the three unconfirmable cases are misses, which
> the gate says they are not.
>
> The "intervention-framed paper with a baseline case-control comparison" failure mode
> described below **survives as a description of what the three look like, but is no
> longer evidence of an extraction bug** — with no reported significance, refusing
> them is what the prompt asks for. It remains worth a look only if the full text
> turns out to report statistics for those taxa, which needs `MAIN_DATA.json`.
>
> Everything below this box is the original text; the deterministic results in it
> (the funnel, the build-loses-nothing null, the disease permutation, the triage
> characterisation) are unaffected — none of them depended on the miss count.

## The gap this closes

Every fidelity instrument in `kg/` scores edges that **exist**: reading fidelity
(`audit_direction_witness.py`, ≥86.6%), taxon mention rate
(`verify_taxon_mentions.py`, 99.57%), agreement with Disbiome/Peryton (73.0/72.5%).
The mirror question had never been asked — **are there relations stated in a paper
that produced no edge at all?**

That is *recall*, and the only recall number this project has ever had (F1 ≈ 0.59)
is scored against the in-house gold standard, which is itself under audit and known
unreliable (162 of 250 papers have blank taxa columns). This measures recall against
**the papers' own sentences**, so it depends on neither the gold nor either curated
database — the same move that made the direction audit the best fidelity number here.

## The funnel (deterministic, reproducible)

```
348  extraction rows            (extractions_corrected.json)
325  survive the corpus screen  (extractions_screened.json)
313  after build_kg title dedup
271  contribute ≥1 edge         = 86.6% paper yield
 42  contribute nothing         <- the subject of this document
```

## Result 1 — the graph build loses nothing. Deterministic null.

Of the 54 screened papers absent from the graph, **10 are dedup twins whose
surviving copy IS in the graph** (their content is represented; the dedup docstring
names one of them verbatim). After collapsing those, the number of papers where the
extractor returned taxa that then failed to become an edge is **0**.

This was worth checking: `build_kg.py` has several `continue` branches and a
`min_papers` filter, and a silent drop there would have been invisible to every
existing instrument. It does not happen. Nothing is lost between extraction and the
graph.

## Result 2 — paper-level recall ≥ 98.5%, and the refusals are mostly correct

Of the 42 distinct zero-yield papers, **9 have no relation-bearing sentence at all**
— `relation_sentences` finds nothing naming a resolved taxon near a direction cue,
so there was nothing to extract. The remaining **33** were adjudicated against their
own sentences, in four independent batches, with verbatim quotes required for any
"miss" verdict:

| verdict | papers | meaning |
|---|---|---|
| `CORRECT_REFUSAL_DESIGN` | 14 | no disease-vs-healthy-control contrast exists — intervention/probiotic trial, animal model, longitudinal stability, symptom-correlation, patient-subgroup vs patient-subgroup |
| `UNCLEAR` | 7 | sentences insufficient to decide |
| `BACKGROUND_ONLY` | 4 | every taxon+direction sentence cites other studies |
| `CORRECT_REFUSAL_NEGATIVE` | 4 | the paper explicitly reports no significant difference |
| **`MISS`** | **4** | **a genuine extraction failure** |

**Paper-level recall = 271/275 = 98.5%**, 95% CI [96.3, 99.4].
Worst case, counting all 7 `UNCLEAR` as misses: **271/282 = 96.1%** [93.2, 97.8].

**This is a paper-level number and must not be quoted as edge-level recall.**
It says the extractor rarely refuses a paper it should have read. It says *nothing*
about whether it caught every taxon *within* the 271 papers it did read — that
quantity is still unmeasured, and is the obvious next instrument.

## The 4 confirmed misses

All four were verified by reading the sentences directly, not taken on the
adjudicator's word — the standing rule, earned when an LLM adjudication of 18
self-contradictions got 4 of its 6 "extraction error" verdicts wrong. Several
supplied quotes turned out to be **paraphrases**; each miss below rests on a quote
confirmed to occur verbatim in the paper.

1. **`Immunotherapy-mediated modulation of the gut microbiota in multiple sclerosis…`**
   (DMF therapy) — *"Some Lachnospiraceae genera had lower abundance in PwMS compared
   to HC"*, *"Prevotella 7 + 9 genera had lower abundance in PwMS compared to HC"*,
   *"the Tannerellaceae family was lower in PwMS than HC"*. Unambiguous MS-vs-healthy
   -control statements. The paper's *framing* is a therapy study, which is likely why
   it was refused — but it carries a baseline case-control comparison.

2. **`Interventional Influence of the Intestinal Microbiome Through Dietary
   Intervention…`** (PD) — *"Moreover, we could show a relative increase of
   Actinobacteria and Firmicutes compared to healthy controls."* Same pattern: an
   intervention paper that also reports a baseline PD-vs-HC contrast.

3. **`Relationships of gut microbiota, short-chain fatty acids, inflammation…`** (PD)
   — *"PD patients in this cohort had reduced abundance of Prevotella"*, *"Bacteroides
   is more abundant in controls, and Bifidobacterium is more abundant in PD patients
   in this cohort"*. The clearest of the four.

4. **`Altered Gut Microbiota and Plasma Metabolome Profiles Characterize Depression
   Individuals with Ischemic Stroke`** — LDA/p-value results throughout
   (*Enterobacteriaceae* LDA 4.177 P=0.014; *Roseburia* LDA 3.894 P=0.007). Its
   "control" is **non-stroke individuals who also have depression**, i.e. depression
   held constant across both arms. By the precedent set on 2026-09-15 for the
   HIV/`Neurocognitive impairment` node, that is correctly typed as a Stroke edge.
   **Caveat, logged: this paper contradicts itself on direction.** It states
   *"the IS group showed increased levels of … Enterobacteriaceae, LDA = 4.177,
   P = 0.014"* and, two sentences later, *"Enterobacteriaceae showing greater
   abundance in the non-IS group (LDA value=4.177, P =0.014)"* — the same statistic,
   the opposite direction. Refusing this paper is defensible; it is listed as a miss
   on the strength of its other, self-consistent taxa.

**The dominant failure mode is a single, nameable one: a paper whose headline design
is an intervention or therapy study, but which reports a baseline disease-vs-control
comparison anyway.** Three of the four are that. This is a prompt-level fix, not a
model-level one, and it is the cheapest recall lever available.

## Result 3 — zero-yield is NOT concentrated in any disease. Null, with power.

Per-disease yield ranges from 60% (Dementia, 10 papers) to 100% (SCI, HD,
Encephalopathy). Tested with a **paper-level permutation** (N=20,000) using a
**max-statistic** over the 11 diseases with n≥5, so the multiple comparison across
diseases is controlled by construction:

> worst per-disease zero-yield rate 0.40 (Dementia), **p = 0.1281 — NULL.**

Power statement: with 313 papers spread over 11 disease groups, the largest group
(PD) has n=77 and the smallest tested has n=5. The observed spread is what
independent binomial draws around an 86.6% base rate look like at these group sizes.
**No disease is systematically under-represented in the graph by silent extraction
failure** — at an MDE that a 10-paper group cannot push below roughly 30 points.
So this is "consistent with no disease bias", not "proven none".

## Result 4 — the deterministic provenance screen is triage, not adjudication

Reusing the `CITATION` / `THIRD_PARTY` / `RESULT_CUE` / `CONTROL_FRAME` regexes from
`audit_direction_witness.py`, a sentence counts as an extraction *candidate* only if
it carries a result cue AND a control frame AND no citation. Over the 33 papers:
400 sentences → 88 cited, 141 own-result, **23 own-result-and-control-framed**,
concentrated in 9 papers.

Against the read verdicts that filter scores **3 of 4 misses caught (75% recall),
3 of 9 flagged papers genuine (33% precision)**. Its one false negative is
instructive: `RESULT_CUE` requires a statistic or an explicit "we found", so it
misses *"the Tannerellaceae family was lower in PwMS than HC"* — a plain-language
result sentence with no number in it.

**So the regex screen must not be used as an adjudicator.** Same conclusion the
abstract screen reached on 2026-09-11 (0-for-4 on drops, 0-for-14 on unclears): its
value is triage. Recorded here so the next session does not mistake 23 candidate
sentences for 23 missing edges.

## Reliability of the adjudication

Two papers entered the adjudication **twice**, as dedup twins with identical text,
in different batches. Both received **identical verdicts** (2/2). That is a small
check — it bounds nothing at n=2 — but it is a real one, and it is the only
inter-rater signal available without a second full pass.

## What this does NOT license

- It is **not** an accuracy gain and must never be cited as one. **Nothing in the
  graph was changed by this work.** `graph.json`, `rag_corpus.jsonl`, `kg.html` and
  `docs/` are untouched, so none of these numbers can be miscited as moving
  agreement.
- Adding the 4 missed papers would grow the corpus by 4 of 275 (1.5%) and, per the
  six structural corrections before it, would move agreement by **less than this
  corpus can resolve** (~0.013). Do it for correctness, not for the number.
- It says nothing about edge-level recall inside contributing papers.

## Next lever

**Measure edge-level recall inside the 271 contributing papers** — the same
instrument, one level down: for each contributing paper, does it contain an
own-result, control-framed sentence naming a resolved taxon for which no edge
exists? `relation_sentences_clean.json` covers 271/271 contributing papers, so this
is answerable in the cloud with no GPU, no taxdump and no `MAIN_DATA.json`. It is
the natural successor to this document and would give the project its first
edge-level recall figure that does not depend on the flawed gold.
