# Edge-level recall: what the extractor missed inside papers it did read

**2026-09-16. Cloud, CPU-only, no `MAIN_DATA.json`, no taxdump.**
Instrument: `edge_recall_audit.py` → `edge_recall.json`,
`edge_recall_packets.json`, `edge_recall_verdicts.json`.
Companion to `FINDINGS_zero_yield.md` (the paper-level question).

## The question

`FINDINGS_zero_yield.md` established that the extractor almost never refuses a
whole paper it should have read. This is the harder question one level down:
**inside the 271 papers it did read, did it catch every taxon?**

That number has never existed. The only recall figure this project has ever had
(F1 ≈ 0.59) is scored against the in-house gold standard, which is under audit and
known unreliable. This scores against the papers' own sentences.

## Headline

> **Observation-level recall ≈ 98.1%, 95% CI [96.4, 99.5]** — roughly **60 missed
> observations [14, 116]** against the 3,077 in the graph.
>
> **This is an UPPER bound. Quote it as "at most ~98%", never as "98% and done".**

## Method, and the step that decides the answer

1. **Candidate generation (deterministic).** For every contributing paper, keep the
   sentences that pass the provenance screen from `audit_direction_witness.py`
   (`RESULT_CUE` and `CONTROL_FRAME` present, `CITATION` and `THIRD_PARTY` absent),
   then collect every taxon in them that resolves to a taxid and for which that
   paper has **no** edge. Matching is on taxid **and** normalised label **and** every
   node alias, so renames and synonyms do not create phantom misses.
   → **378 candidate (paper, taxon) pairs across 94 papers.**

2. **Adjudication.** A random sample of **24 papers / 95 candidates**, read in four
   independent batches, verbatim quotes required for any miss verdict.

3. **Gate application — this is the step that decides the answer, and the first
   pass of this audit got it wrong.** The extraction prompt
   (`eval-v2/run_eval.py`, `samgated-v1`) does not extract anything that merely
   states a direction:

   - **SIGNIFICANCE** — *"include a taxon ONLY if the paper reports it as
     statistically significant … If significance is unclear or unreported for a
     taxon, **omit it**."*
   - **MAIN TEXT ONLY** — not tables, figures, captions, supplementary.
   - **DISEASE vs HEALTHY CONTROL ONLY.**

   A candidate is a confirmed miss only if its quote occurs verbatim **and** some
   sentence naming that taxon carries a significance cue that is not negated
   ("no significant", "tendency", "trend toward") and is not a citation.

   **36 raw `REAL_MISS` verdicts → 29 verbatim → 15 confirmed.**

   Scoring an extractor without applying its own gate manufactures misses. It cost
   this session a wrong headline in `FINDINGS_zero_yield.md` (4 misses claimed, 1
   real) before the same test was applied there.

## Why this is an upper bound on recall

Two biases push the same way:

- The candidate generator only sees taxa that resolve to a taxid **and** sit in a
  sentence the regex screen keeps. That screen was measured at **~75% paper-level
  recall** in `FINDINGS_zero_yield.md`, so misses stated in dropped sentences are
  invisible here.
- `relation_sentences_clean.json` holds ~10% of corpus text, and the significance
  gate can be **confirmed** from visible text but never **refuted** — a taxon whose
  p-value lives in a sentence with no direction word scores as a non-miss.

So true recall is **at or below** 98.1%. The honest reading is "no evidence of a
large recall hole", not "recall is 98%".

## The 15 confirmed misses

Highly clustered: **12 of 15 come from 3 papers** — an HBV-associated liver
cirrhosis study (5: *Coprococcus*, *Dorea*, *Cardiobacteriaceae*, *Anaerofustis*,
*Finegoldia*), an Egyptian Parkinson's cohort (4: *Prevotella*, *Bacteroidota*,
*Ruminococcus*, *Lactobacillus*), and a Ugandan Alzheimer's cohort (3:
*Acidobacteriota*, *Bacillota*, *Pseudomonadota*). That clustering is why the
paper-level bootstrap CI is wide: [0.038, 0.308].

**This concentration is the actionable part, not the average.** If missed edges
cluster in a handful of papers rather than spreading thinly, then re-running
extraction on a *short list* recovers most of the loss — which is a far cheaper
intervention than re-extracting the corpus, and it does not need a GPU-scale run.

## Null: high-rank taxa are NOT missed more often. Tested because it looked obvious.

Five of the 15 confirmed misses are phylum-level (*Bacteroidota*, *Bacillota*,
*Pseudomonadota*, *Actinomycetota*, *Acidobacteriota*), and one adjudicator
independently volunteered "all 3 misses are higher-rank taxa" as a pattern. Against
the graph's own rank distribution that looks dramatic — **33% of confirmed misses at
phylum/class vs 6.5% of graph edges**.

It does not survive:

> **p = 0.0537**, permutation at the paper level preserving each paper's confirmed
> count, N=20,000. A single uncorrected test on 15 events from 24 papers.

And the reason is the useful part: **27% of the candidates are already
phylum/class** — the candidate generator is itself rank-skewed, because high-rank
taxa appear in the summary sentences the provenance screen preferentially keeps.
Compared against the pool it was actually drawn from, 33% vs 27% is nothing. The
comparison against the graph's 6.5% is the wrong denominator and would have been a
false finding.

Power: with 15 events, this test could not resolve anything below roughly a
doubling. "No rank effect visible at n=15 from 24 papers" is the result.

## What this does NOT license

- **Nothing in the graph was changed.** `graph.json`, `rag_corpus.jsonl`, `kg.html`
  and `docs/` are untouched. None of these numbers may be cited as an accuracy gain.
- Adding ~60 observations to 3,077 (+2%) would, on the record of the six structural
  corrections before it, move agreement by **less than this corpus can resolve**
  (~0.013). Fix it for correctness, not for the number.
- The 98.1% figure is **observation-level** (paper × taxon × disease), not
  edge-level: a missed observation on an existing edge changes that edge's evidence
  count, not its existence.

## Next lever

**Re-extract the short list, not the corpus.** The 94 candidate papers — or, more
cheaply, the ~20 with the most candidates — are where the recoverable loss is
concentrated. That is a bounded re-run, and the candidates are already enumerated in
`edge_recall_packets.json`.

Second, and free: the misses are dominated by taxa stated with a significance cue in
a *separate* sentence from the direction. A generation-2 filter that links a
direction sentence to a neighbouring significance sentence (rather than requiring
both in one sentence) would raise this instrument's own sensitivity and shrink the
"cannot refute" gap described above.
