# 208 — The Extraction Frontier: what, if anything, is left to gain

*Research review, 2026-09-19. Skeptical/technical. Status: IN PROGRESS — sections are written into
this file as they are finished, in priority order. If a later section is missing, the run was cut
short, not the question dropped.*

**The setup under review.** Qwopus3.5-27B-v3 (Qwen-3.5-27B distilled on Claude-Opus traces), Q4_K_M
GGUF, llama.cpp on a DSMLP GPU, ~20–22 s/paper. Output forced by a hand-written GBNF grammar to
`{"disease", "taxa_enriched":[…], "taxa_depleted":[…]}`. Prompt `samgated-v1`
(`eval-v2/run_eval.py:47`) with three hard gates: reported statistical significance, main text only,
disease-vs-healthy-control only (plus human-hosts-only and verbatim-copy instructions, which are
really schema hygiene rather than gates).

## Sections

1. [What to actually do](#1-what-to-actually-do) — ranked next experiments
2. [Is the model the limiting factor?](#2-is-the-model-the-limiting-factor) — open-weight landscape *(pending)*
3. [Tool use / agentic extraction](#3-tool-use--agentic-extraction) *(pending)*
4. [The three gates as an ML problem](#4-the-three-gates-as-an-ml-problem) *(pending)*
5. [Quantization](#5-quantization) *(pending)*
6. [Constrained decoding beyond GBNF](#6-constrained-decoding-beyond-gbnf) *(pending)*

---

## 1. What to actually do

### 1.1 The conclusion, stated so it cannot be buried

**The extractor is close enough to its ceiling that the project can no longer measure movement in
it.** Both halves of that sentence matter, and the second half is the one that determines what to do
next.

The defensible version of the claim is not "the model is perfect." It is this: *every remaining
deficiency this project has successfully measured is smaller than the resolution of every instrument
it has for detecting an improvement.* That is a measurement-limited regime, and in a
measurement-limited regime the correct move is to spend on the instrument and on the inputs, not on
the thing being measured.

Two arithmetic facts drive the whole section.

**Fact 1 — the measured headroom is ~2%, and it is concentrated.**

| instrument | figure | residual deficit |
|---|---|---|
| reading fidelity (`FINDINGS_direction_audit.md`) | ≥86.6% [81.7, 91.3], 181/209 scoreable obs | 28 disagreements, **all 28 adjudicated twice to "not an extraction error"** → confirmed error rate 0/209 |
| paper-level recall (`FINDINGS_zero_yield.md`) | ≥96.1%; 99.6% [97.9, 99.9] confirmable | **1 confirmed miss in 313 screened papers** |
| edge-level recall (`FINDINGS_edge_recall.md`) | ≤98.1% [96.4, 99.5] | ~60 missed observations [14, 116] against 3,077 — and **12 of 15 confirmed misses come from 3 papers** |

The only quantified loss is roughly sixty observations out of three thousand, most of it in a
handful of papers. Note also which direction each bound points: fidelity and paper recall are
*lower* bounds (they can only be better than stated), edge recall is an *upper* bound biased
downward by the candidate generator. A perfect extractor would move the headline agreement numbers
by an amount the corpus cannot resolve.

**Fact 2 — compute is not the scarce resource; adjudication is.**

A full corpus pass is 271 papers × 22 s ≈ **1.7 GPU-hours**. The 80-paper prompt experiment was
~30 min per variant. This is important and slightly counter-intuitive: the "expected gain per
GPU-hour" ranking the brief asks for has a *tiny denominator for almost every candidate*, so the
ranking is driven almost entirely by the numerator — and by the human adjudication hours each
experiment generates, which is the genuinely scarce input. An experiment that costs 2 GPU-hours and
30 adjudication-hours is more expensive than one that costs 10 GPU-hours and 2.

### 1.2 Why the instrument, not the extractor, is the binding constraint

Four independent measurements of the project's own resolution:

1. **The 80-paper gate experiment** (`prompt_exp_results.json`, the most recent controlled test).
   Baseline A: P .695 / R .699 / **F1 .6968**. Variant C (drop the main-text gate): P .673 / R .730 /
   **F1 .7002**. Variant B (soften the significance gate): **F1 .6671**. Verdict on both: *keep
   baseline* — the paired 95% CI on per-paper ΔF1 excluded neither zero nor, in C's case, a
   meaningful effect. An 80-paper paired design, which is a *good* design for this project, could
   not resolve +0.003 aggregate F1.
2. **The external-agreement numbers cannot resolve 0.013.** Root `CLAUDE.md` records that five
   structural corrections to the graph — species splitting, punctuation defragmentation, typo
   folding, MONDO ids, deduplication — each moved Disbiome/Peryton agreement by less than the corpus
   can resolve (~0.013). Those were *real, verified* corrections and the headline metric could not
   see any of them.
3. **The in-house gold is a broken ruler and is known to be.** 162 of 250 papers have blank taxa
   columns; an Opus 4.8 re-annotation found 72 taxa the humans missed (55% of which sit with a hard
   statistical cue); correcting the gold on 15 papers moved F1 0.64 → 0.84. Any F1 measured against
   it is agreement-with-a-flawed-reference. The eval-v2 leaderboard and every number in
   `RESULTS.md` inherit this.
4. **`test_set_v2` is n=15.** It is the only held-out set, and it is far too small to separate models
   that differ by a few points. Note also that the 80-paper subset scores baseline at .697 while
   `test_set_v2` scores the same model at .751 — a 5-point spread between two evaluation sets of the
   same extractor, which is itself a measure of how much of the headline number is set selection.

Put these together: **the smallest effect the project can currently detect is ~3–5 F1 points, and
the largest remaining defect it has been able to find is ~2% of edges.** There is no experiment in
the model-improvement direction whose positive result could be believed. This is the single most
important sentence in this document.

### 1.3 The base rate on method interventions here is 0 for 5

Worth stating plainly before proposing anything, because it should set the prior:

| intervention | outcome |
|---|---|
| RELATE (retrieve → rerank-with-reject) | **backfired** on all three local models (−.046/−.075/−.069); FPs rose because quantized models could not reject noisy regex candidates |
| grounded extraction (verbatim sentence per taxon) | helped **only** the under-extractor (Qwen2.5-32B); neutral on Qwopus |
| normalization | **exactly zero** F1 change; it is a node-dedup tool |
| standalone LLM judge | first prototype was a **data leak** (used the Opus-4.8 gold as oracle); the gold-free rebuild is a precision↔recall trade with net F1 ≈ 0 at +42 s/paper. Dropped. |
| prompt-gate variants B and C | both null under a pre-registered rule |
| (lab history) open-schema BioBERT RE, 22 labels | **6% precision, 88% FP**, confidence anti-calibrated |

Five controlled method experiments, five nulls or regressions, and the one apparent large win was an
artifact of leakage. A sixth method experiment proposed on the same instrument should be expected to
return the same answer. The one thing that *did* produce large movement was **fixing the reference**
(+.13–.14 F1 from judge-recovered FPs — i.e. the gold under-counted), which is the same lesson from
a different angle: the reference, not the extractor, is where the points are.

### 1.4 The ranking

Ordered by *expected defensible gain per GPU-hour, discounted by adjudication hours*. Each entry
states what would falsify it.

---

**#1 — Finish the corpus screen. 0 GPU-hours.**

Only **23 of 271 contributing papers** have ever been screened for study design
(`FINDINGS_corpus_screen.md`). `maindata_screen.json` covers the 45 title-matched MAIN_DATA
additions; the ~250 datasheet papers were never put through it. An animal-study prefilter validated
at recall 15/15 returns a null on the rest — but animal studies were only 15 of 22 drop reasons.
**No-healthy-control, case-report and review designs remain unscreened across 248 papers.**

This is a *precision* lever denominated in whole papers, and it dominates everything below it on
arithmetic. The `samgated-v1` DISEASE-vs-HC gate instructs the model to return empty lists when
there is no healthy control group — but that gate can only fire when the model correctly identifies
the design from a truncated full text, and it has never been audited as a design classifier. If even
5% of the 248 unscreened papers are the wrong design, that is ~12 papers × ~7 edges ≈ **85 spurious
edges — larger than the entire measured edge-recall deficit (~60)**, and removing them is
deterministic rather than probabilistic.

It also has a property nothing else here has: it improves the *graph* without needing an instrument
sensitive enough to see a 2-point F1 change, because the unit of action is "this paper should not be
in the corpus," which a human can adjudicate directly.

*Falsification:* screen a random 60 of the 248 and count design violations. If the violation rate is
<2% (≤1 paper in 60), the lever is dead and the remaining 188 do not need screening. The confidence
interval on 0/60 tops out around 5%, so a clean null here is genuinely informative.

---

**#2 — Re-extract the enumerated edge-recall shortlist. ~0.1 GPU-hours.**

`kg/edge_recall_packets.json` already holds 378 candidate (paper, taxon) pairs across 94 papers, and
the finding is that the misses **cluster**: 12 of 15 confirmed misses in 3 papers. A shortlist
re-extraction over those 94 papers is ~35 minutes of GPU time and recovers most of a deficit that a
corpus-scale run would cost 1.7 GPU-hours to address no better.

The important design constraint, from `FINDINGS_edge_recall.md` and restated in root `CLAUDE.md`:
**apply the gate when scoring.** An audit that scores the extractor without applying its own
significance/main-text/HC gates manufactures misses — the first recall pass reported 4 misses where
the true confirmable count was 1.

*Falsification:* if adjudicating the recovered pairs against paper sentences yields <10 genuinely
missed observations, the clustering hypothesis is wrong and the ≤98.1% bound is already effectively
tight. Also worth pre-registering: the tempting "high-rank taxa are missed more often" story was
already **tested and rejected** (p=0.054; the apparent phylum skew is a property of the candidate
generator, 27% of whose candidates are phylum/class). Do not re-find it.

---

**#3 — Run a second, architecturally different extractor as a disagreement generator, not a
replacement. ~2–4 GPU-hours.**

This is the one model-side experiment worth running, and the reframing is what makes it worth
running. Asking "is model X better than Qwopus?" is unanswerable here — §1.2 shows no instrument
resolves the likely difference. Asking "where do two independent extractors disagree?" is answerable
and produces a *finite, adjudicable list* instead of a number nobody can believe.

Run a second model over the same 271 papers under the identical prompt and grammar. Every
(paper, taxon, direction) triple where the two disagree becomes an adjudication packet scored
against the paper's own sentences with the gates applied — the same instrument that produced the
only two trustworthy quality numbers this project has (fidelity 86.6%, paper recall 96.1%). The
output is not "model B wins"; it is a set of specific corrections to the graph, plus an *unbiased*
estimate of each model's error rate on the disagreement set.

Model selection for this is §2's job. Two properties matter more than benchmark scores: (a)
architecturally different from Qwen-3.5-27B, so the errors are not correlated — a second Qwen
derivative will agree with Qwopus precisely where both are wrong; (b) able to consume ~25k tokens of
paper at usable throughput.

*Falsification:* if disagreement is <5% of triples, the exercise yields too few packets to be worth
the adjudication hours and the two models are effectively the same instrument. If disagreement is
>30%, something is wrong with the harness (prompt not actually held constant, grammar mismatch,
truncation differing) — debug before adjudicating.

*The trap to avoid:* do **not** let the second model's output become a silent oracle. That is exactly
the failure mode of the first judge prototype, which "recovered" false positives by consulting the
Opus-4.8 gold. Disagreements go to text-grounded adjudication, never to a model vote.

---

**#4 — The Q8-vs-Q4_K_M control. ~1 GPU-hour.**

The cheapest possible one-variable model-side experiment, and it settles §5 empirically for this
task rather than by literature survey: same weights, same prompt, same grammar, same 80 papers, one
bit-width change. It is worth running *before* anyone proposes a larger model, because if Q4_K_M is
costing nothing on this task, the entire "bigger model / better quantization" axis is much less
attractive, and if it is costing something, that is the cheapest fix available (no new model, no new
prompt, no new validation).

There is a confound in the existing record worth clearing up rather than inheriting: `eval-v1`
ran Qwopus3.5-27B-v3 at **Q8_0** and `eval-v2` runs it at **Q4_K_M**, but the prompts also differ,
so the existing "v3 underperformed v1" note is not a quantization result.

*Falsification:* paired per-paper ΔF1 CI includes zero (the likely outcome — see §5). Then the
quantization axis is closed and should be written up as closed.

---

**#5 — A significance-gate classifier over candidate sentences. GPU cost small; design risk high.**

Treated properly in §4. Ranked here rather than higher because of the asymmetry that constrains it:
**significance can be confirmed from visible text but never refuted** — no p-value in a sentence does
not mean no significance. A classifier therefore has a one-sided decision boundary and cannot be
trained to a clean negative class, and its only available training labels come from the flawed
in-house gold. Its defensible use is as a *precision* filter (flag graph edges whose supporting
sentences carry no confirmable significance cue, for human review), not as a recall lever inside the
extraction loop.

*Falsification:* if a held-out, hand-labelled set of ~200 candidate sentences shows the classifier
and the extractor's own gate agree >95% of the time, the classifier adds nothing the prompt clause
is not already doing.

---

**#6 — Tool-augmented / agentic extraction. Not recommended now.**

Treated in §3. The short version: the one tool with a defensible role — NCBI taxonomy lookup —
already exists as deterministic *post*-processing (`taxonomy.py`, `taxon_typos.py`,
`species_synonyms.py`) at zero GPU cost, and moving it inside the loop would actively hurt, because
33 of the unresolved taxon labels are **the papers' own misspellings, all 33 verified verbatim in
source text**. A mid-extraction validator that "corrects" `Fecalibacterium` destroys provenance the
project deliberately preserves. The text-search-to-verify-significance tool re-creates the RELATE
failure mode: it adds a decision point where a quantized model must *reject*, and rejection under
noise is the specific thing this model class was measured to be bad at.

---

**#7 — Alternative constrained-decoding backends. Not recommended.**

§6. The GBNF grammar reports **0 parse errors on 80 papers** across three prompt variants (1 in 240
runs total, in variant C). Parse reliability is not a live problem, so a different backend can only
offer convenience or speed, not quality. The one real hazard is already documented and is a
footgun rather than a framework problem: every GBNF rule must be on one line, because
`LlamaGrammar.from_string()` does not validate and a multi-line `root` rule segfaults at sampling
time (rc=139, no traceback).

### 1.5 What this means for the project's time

The ranking above puts two zero-or-near-zero GPU items at the top and everything model-shaped below
them. That is not pessimism about models; it is what the project's own five-for-five null record and
its measured 2% headroom imply.

The larger point, which belongs in whatever gets written up: **the binding constraint on this
project stopped being extraction quality some time ago.** The open questions in root `CLAUDE.md` —
why papers disagree (24 variables, 24 nulls, paper-level SD of discordance 3.4 points against MDEs
of 4–7), whether contested edges have a design explanation, whether disease subtypes cohere — are
all limited by **n = 271 papers**, not by whether the extractor reads those papers at 96% or 99%.
Improving the extractor from 98% to 99% edge recall adds ~30 observations to a corpus that needs
thousands more to answer its open questions. Corpus size and screening quality are the live levers;
see `203_corpus_scaling.md` for the power arithmetic on the first of those, with the caveat that
scaling is not automatically the answer either.

---

## 2. Is the model the limiting factor?

*(pending)*

## 3. Tool use / agentic extraction

*(pending)*

## 4. The three gates as an ML problem

*(pending)*

## 5. Quantization

*(pending)*

## 6. Constrained decoding beyond GBNF

*(pending)*
