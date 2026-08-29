# What the literature will tell you, and what it won't

**Lab work · technical report · Projects 2 & 3 · August 2026**

Two pipelines built this year, over two different corpora, that ended up making the same argument
from opposite directions: the hard part of turning papers into data is almost never the model or the
network. It's knowing which of your own answers you are actually entitled to believe.

| | |
|---|---|
| **Project 3 · codes recovered** | **+1,176** — accession codes added to the corpus, 4,405 → 5,581 rows |
| **Project 3 · runtime** | **5 h → 1 h** — multi-day Apps Script job → one local run; re-extraction now takes seconds |
| **Project 3 · triaged** | **7,511** code-less rows sorted into audited-empty vs. a curator worklist. None dropped |
| **Project 2 · best local F1** | **0.84** — free 27B model, fairly scored, up from an apparent 0.64 |
| **Project 2 · models benchmarked** | **5** open-weight GGUF extractors on H100, one harness, one prompt |
| **Project 2 · methods rejected** | **4 of 5** — RELATE, grounding, normalization, LLM-judge: measured, then dropped |

13,092 papers · 5,581 answered · 3,434 audited-empty · 4,077 flagged for a human.

---

## Contents

**Part I — Project 2: extraction, and the benchmark that was lying**
[I.1 Building a gold standard worth failing against](#i1-building-a-gold-standard-worth-failing-against) ·
[I.2 The harness](#i2-the-harness) ·
[I.3 The 0.64 that was really 0.84](#i3-the-064-that-was-really-084) ·
[I.4 The leaderboard](#i4-the-leaderboard) ·
[I.5 Five methods, four rejected](#i5-five-methods-four-rejected) ·
[I.6 The judge that cheated](#i6-the-judge-that-cheated) ·
[I.7 Two decisions that aren't mine](#i7-two-decisions-that-arent-mine)

**Part II — Project 3: 13,092 papers, one column**
[II.1 The wall, and the shape of the fix](#ii1-the-wall-and-the-shape-of-the-fix) ·
[II.2 The recovery ladder](#ii2-the-recovery-ladder) ·
[II.3 What we're entitled to claim](#ii3-what-were-entitled-to-claim) ·
[II.4 Audits designed to break our own output](#ii4-audits-designed-to-break-our-own-output) ·
[II.5 Three assumptions the data killed](#ii5-three-assumptions-the-data-killed) ·
[II.6 The finding: 59 → 34 → 14](#ii6-the-finding-59--34--14)

**Part III — [What the two projects share](#part-iii--what-the-two-projects-share)**

---

# Part I — Extraction, and the benchmark that was lying

The end goal is a microbe–disease knowledge graph: edges that say *this taxon is enriched, this one
depleted, in this disease*. Before any of that can be assembled, the edges have to be trustworthy —
which means an extractor good enough to run over the corpus, and a way to know it is. Both turned
out to be measurement problems.

## I.1 Building a gold standard worth failing against

A labmate's hand annotations were the seed: ~342 papers with enriched/depleted taxa columns. That is
not a benchmark yet — it's a spreadsheet. Turning it into one meant deciding, explicitly, which
papers were even readable enough to be scored on.

Full texts were scraped from PMC, then filtered by a completeness score rather than by eye. Each
paper got 0–5 points for length, p-value density, figure/table references, and methods vocabulary
(`QIIME`, `DADA2`, `16S`, `LEfSe`). Anything below 4 was thrown out — not because the paper is bad,
but because a truncated or figure-only paper punishes an extractor for something it can't see. The
survivors were split disease-stratified (`seed=42`) so no single disease could dominate the test set.

| Stage | Papers | Note |
|---|---:|---|
| Annotated rows | 342 | labmate's master sheet |
| Full text scraped | 250 | PMC retrieval succeeded |
| Non-empty taxa | 88 | has an actual annotation |
| Completeness ≥ 4/5 | 72 | readable as a whole paper |
| **Held-out test set** | **15** | `test_set_v2` — read-only, used for all scoring |
| Reserve pool | 57 | untouched holdout |

The 57-paper reserve exists so that when the test set eventually gets contaminated by too much
iteration, there is a clean set left to re-check against.

## I.2 The harness

Models run as quantized GGUF weights on UCSD DSMLP GPUs under llama.cpp, driven by a single entry
point — `run_eval.py --model <key>` — so that a model swap is a flag, not a new notebook.
Temperature 0, a shared prompt, and a GBNF grammar to force the output schema, which removes an
entire class of "the model was right but the JSON was malformed" noise.

The scoring metric is where the real work went. The original metric compared taxon names like a
spell-checker: character n-gram cosine similarity. That treats *Bacteroides* (genus) and
*Bacteroidaceae* (family) as different answers, when biologically one contains the other. The
replacement resolves every name to its NCBI lineage and counts a match when one taxon is an
**ancestor of the other**, falling back to the old string metric only for names NCBI can't resolve.

> **Changing the metric moved the best model from 0.642 to 0.751 without touching a single model
> output. That gap was never the model's — it was ours.**

*Design detail worth keeping:* if the taxonomy tooling isn't installed, the harness prints a warning
and silently falls back to the character metric instead of hard-failing. An eval that crashes on a
fresh pod is an eval nobody runs.

## I.3 The 0.64 that was really 0.84

For months the local models looked mediocre. The best of them scored ~0.64 F1 and no amount of
prompt engineering moved it. The obvious read was "open-weight models can't do this." That read was
wrong, and proving it wrong is the most useful thing this project produced.

The test: take a frontier model and use it not as a competitor but as an **annotator**, re-reading
all 15 test papers thoroughly to produce a second, independent benchmark. Then re-score every model
against both. If our models were simply bad, their scores wouldn't move much. They moved a lot —
because the original benchmark listed 119 taxa, and a thorough pass found roughly **70 more that are
genuinely stated in the papers**. Correct extractions were being counted as false positives.

Same model, same outputs, four scorings (Qwopus3.5-27B):

| Scored against | F1 |
|---|---:|
| Original benchmark (hand-annotated, incomplete) | 0.64 |
| + complete benchmark (thorough re-annotation) | 0.77 |
| + taxonomy-aware matching (rank-aware, not spell-check) | 0.82 |
| + scope correction (excluding 2 out-of-scope papers) | **0.84** |

The model's predictions are byte-identical across all four rows; only the measurement changed.

**The falsification check that makes this credible:** the one model that *under*-extracts
(Qwen2.5-32B) scored *worse* on the fuller benchmark — 0.606 → 0.502. A completeness fix should
reward models that find more and punish models that find less. It did. This isn't the benchmark
flattering our favourite model.

There is still an unresolved question underneath this, and it is being flagged rather than papered
over: the two benchmarks disagree on roughly 70 taxa, and it is **not settled whether the human gold
is incomplete or the frontier annotator over-extracts**. On at least one paper the frontier
annotator was demonstrably wrong. Everywhere else its extras sit next to real significance cues.
That adjudication needs a human with the annotation guidelines in hand — it is not a thing to
resolve by picking whichever benchmark makes the numbers nicer.

## I.4 The leaderboard

Five open-weight models, one prompt, one harness, 15 papers each. The frontier model appears in the
write-ups as a reference annotator, not a contestant — scoring it against a benchmark it wrote would
be circular.

**Taxonomy-aware F1:**

| Model | vs. original benchmark | vs. thorough re-annotation |
|---|---:|---:|
| **Qwopus3.5-27B** (q4_k_m · 22.5 s/paper) | **0.751** | **0.806** |
| Qwythos-9B (q8 · 19.7 s/paper) | 0.665 | 0.742 |
| Qwen2.5-32B (q4_k_m · 18.9 s/paper) | 0.606 | 0.502 |

Two further models (Qwen3.6-35B-A3B, Qwopus3.6-35B-MTP) ran on the character metric only and placed
last — both over-extract heavily.

**Character-metric run — all five models** (the older, stricter string metric, kept for continuity):

| Model | Quant | TP | FP | FN | Precision | Recall | F1 | s/paper |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwopus3.5-27B-v3** (Qwen-3.5 distilled on Opus traces) | q4_k_m | 103 | 93 | 22 | 0.526 | 0.824 | **0.642** | 22.5 |
| Qwen2.5-32B-Instruct (the under-extractor) | q4_k_m | 70 | 57 | 51 | 0.551 | 0.579 | 0.565 | 18.9 |
| Qwythos-9B | q8 | 102 | 128 | 24 | 0.444 | 0.810 | 0.573 | 19.7 |
| Qwen3.6-35B-A3B | q4_k_m | 105 | 148 | 29 | 0.415 | 0.784 | 0.543 | 15.3 |
| Qwopus3.6-35B-A3B-MTP | q4_k_m | 107 | 165 | 25 | 0.393 | 0.811 | 0.530 | 18.5 |

Read the FP column against the story above: every model is recall-heavy and precision-light, and a
large share of those "false positives" were later shown to be real taxa the benchmark had missed.
Newer ≠ better — the two 35B models are the worst performers here.

**Decision: Qwopus3.5-27B is the corpus-scale extractor.** It is free, runs locally, is the
strongest local model on both benchmarks, and at ~22 s/paper covers 100 papers in about 35 minutes —
inside budget for a full-corpus run.

## I.5 Five methods, four rejected

With a fair metric in place, the next question was whether a smarter extraction strategy beats a
plain single-shot prompt. Five approaches were implemented and run as real GGUF inference on an H100
— 31 minutes of GPU time, three models, logged per-run so the results can't be hand-waved. Four lost.

F1 vs. the original benchmark:

| Method | Qwopus3.5 | Qwythos-9B | Qwen2.5-32B | Verdict |
|---|---:|---:|---:|---|
| **Single-shot** (one prompt, grammar-constrained) | **0.751** | **0.665** | 0.606 | **kept** |
| RELATE (retrieve candidates → rerank with a reject option) | 0.705 (▼.046) | 0.590 (▼.075) | 0.537 (▼.069) | backfired on every model |
| Grounded (must quote the sentence supporting each taxon) | 0.668 | 0.644 | **0.653** (▲.047) | helps only under-extractors |
| Normalize (canonicalize taxon names before scoring) | 0.745 | 0.665 | 0.608 | no effect (±0.00) |
| LLM-judge (second pass verifies each extraction) | 0.772 (▲.021) | 0.602 (▼.063) | 0.621 (▲.015) | too slow for +0.02 |

The judge row is the honest gold-free rerun, not the leaked prototype — see I.6.

**Why RELATE backfired — the most useful negative result.** RELATE is a published, sensible idea:
instead of asking a model to free-form extract, retrieve candidate mentions first, then have the
model rerank them with an explicit *reject* option. On a frontier model in a pilot, it helped. On
the quantized 9–27B local models it made everything worse, and the mechanism is instructive: stage-1
retrieval was a crude regex producing ~200 noisy candidates, and the small models **could not say
no**. Qwopus's false positives went from 65 to 114. Handing a weak model a longer list of things to
approve gets you a longer list of approvals.

*Conclusion carried forward:* RELATE isn't wrong, it's mis-staffed. It needs real NER or dense
retrieval for candidate generation and a strong reranker. It is not plug-and-play on quantized local
weights.

**Grounding, and what it's actually for.** Requiring a verbatim supporting sentence per taxon lifted
only Qwen2.5 — the model that under-extracts — from 0.606 to 0.653, with precision climbing
0.598 → 0.681. For the stronger models it was neutral or negative. Grounding is a discipline for
models that make things up or miss the significance framing; it isn't a general-purpose upgrade.

**Normalization: right idea, wrong stage.** 18% of taxon names (43 of 236) fail NCBI resolution
outright — SILVA/GTDB placeholders, misspellings, bracket junk. An LLM normalization pass recovered
**30 of the 43** (`Oscillobacter` → *Oscillibacter*, `c-Actinobacteria` → *Actinomycetia*,
`Bacteroides vulgatus` → *Phocaeicola vulgatus*, which was genuinely reclassified). The remaining 13
are real placeholders like `RB41` and `UBA1819`, plus "lactic acid bacteria" — a functional group,
not a taxon.

And it moved F1 by **+0.000**. The reason is boring and important: a tail name that appears in both
the prediction and the benchmark already matches itself by string. The metric never needed NCBI to
see that `Butyricoccus` equals `Butyricoccus`. Normalization isn't a scoring lever — it's a
**knowledge-graph node-canonicalization lever**, and there it's mandatory, because *Bacteroides
vulgatus* and *Phocaeicola vulgatus* must collapse to one node or the graph fragments into synonyms.
So it ships — at graph assembly, not in the eval.

## I.6 The judge that cheated

The most promising idea in the sweep was a verification pass: let the model re-read the paper and
check its own extractions, dropping the ones it can't confirm. The first prototype showed +0.13 to
+0.14 F1 across every model — by far the biggest lever measured.

It was a data leak. The prototype "verified" candidates against the frontier re-annotation, which is
the answer key. It was measuring how incomplete the original benchmark is, not whether a model can
verify anything. That result was labelled as a leak in the write-up rather than quietly deleted, and
the experiment was rebuilt properly: each model re-reads the paper and answers *is taxon X
significantly enriched/depleted in disease vs. healthy control — yes or no?* for its own
extractions. No gold standard anywhere in the loop.

| Model | Base F1 | + judge | Δ | Judge cost /paper | What happened |
|---|---:|---:|---:|---:|---|
| Qwopus3.5-27B | 0.751 | 0.761 | +0.02 (best case) | 41.8 s | small gain, big bill |
| Qwythos-9B | 0.665 | 0.624 | −0.06 | 16.7 s | recall 0.846 → 0.60 — rejected real findings |
| Qwen2.5-32B | 0.606 | 0.621 | +0.02 | 42.7 s | precision up, recall down |

Batching 3 claims per call is ~2.6× faster than 1 with essentially identical quality — worth knowing
if a judge is ever revived.

Two findings, both cleaner than the leaked version's headline. First, a judge is a
**precision↔recall trade, not free accuracy**: it can only remove taxa, so precision rises and
recall falls, and F1 barely moves. Second — the line worth remembering — **a model is only as good a
judge as it is an extractor**. The weak model was the worst judge, confidently deleting correct
findings.

On cost: even batched, verification adds ~42 s/paper on top of ~22 s extraction, against a
30–40 s/paper budget for a corpus-scale run. Tripling the runtime to buy +0.02 F1 is not a trade
worth making. **Decision: drop the standalone judge and fold the significance check into the
extraction prompt** — zero extra calls.

> **No method beat plain single-shot on the local models. The ceiling was never the prompt — it was
> the benchmark's completeness.**

## I.7 Two decisions that aren't mine

Both are annotation-policy questions, and both change what "correct" means for every model at once,
so guessing at them would silently bake a preference into the benchmark.

1. **Scope.** Two of the 15 test papers have no clean healthy-vs-disease control arm — an
   HIV-cognition study and a memory-clinic study reporting symptom correlations and
   subgroup-vs-subgroup differences. Under the current rule (*disease vs. healthy control only*) the
   correct output for both is *nothing*, so they penalize every model regardless of quality. Should
   papers like these be in scope at all?
2. **Uniformity.** At least one paper appears to be annotated group A vs. group B rather than
   control vs. experimental. Are the guidelines meant to apply strictly and identically to every
   paper, or is per-paper judgement expected? This determines whether the ~70 disputed taxa are
   benchmark gaps or extractor over-reach.

Everything else in this project is now downstream of a knowledge graph that hasn't been built yet:
run the chosen extractor over the full corpus, canonicalize taxa to NCBI nodes, emit directed
microbe→disease edges, and validate them against curated databases (Peryton, Disbiome) that already
cover the neurodegenerative disease mix — external ground truth instead of 15 papers.

---

# Part II — 13,092 papers, one column

A different corpus and a much blunter task: for every microbiome paper in the sheet, find the **data
accession code** — the BioProject, SRA, GEO, EGA or dbGaP identifier under which the study's
sequence data was deposited — and say whether the authors deposited it or reused someone else's. The
governing instruction from the PI: **answer every row, flag ambiguity**. Nothing is silently
dropped; a flagged row is a working row, not a failure.

## II.1 The wall, and the shape of the fix

The inherited pipeline ran in Google Apps Script: a 6-minute execution cap, a ~90 min/day trigger
quota, one-cell-at-a-time writes, no debugger. A 5,712-row job took about five hours spread over
multiple days, and any change to a regex meant re-running the whole thing.

The port to Python kept the extraction logic and changed the two things that made the old version
expensive to be wrong in:

1. **One dictionary, never copied.** Every repository pattern, guard, provenance cue and floor
   reason lives in `dictionary.py`, imported by the pipeline and by any audit. The old codebase had
   three divergent copies, one of which was missing a repository pattern entirely. That class of bug
   is now structurally impossible rather than merely fixed.
2. **Fetch once, extract infinitely.** A content-addressed disk cache sits between the network phase
   and the extraction phase. The slow part runs once; extraction reads only from cache. Tweaking a
   regex re-runs the whole 13k-paper corpus in seconds instead of re-fetching it — which is what
   made the audit work later in this document affordable at all.

Result: ~5 hours across days → **~1 hour, once**, then seconds per re-extraction.

The extraction dictionary itself covers 19 repository formats with two guards. A **standalone-token
guard** rejects a match when it's glued to surrounding alphanumerics — this is what stops grant IDs
like `JUSRP51710A` being read as SRA accessions, a mistake the old pipeline made. A **lookalike
guard** rejects matches whose ±60-character neighbourhood mentions primers, probes, catalog numbers
or RefSeq IDs. Note that the non-INSDC repositories (GSA, CNGB, NODE, BioStudies, HRA) plus
DOI-based archives (figshare, Zenodo, Dryad) account for roughly **450 codes — over 10% of
everything found**. They were added after the original spec and are not busywork.

## II.2 The recovery ladder

Rows are worked cheapest-channel-first, and each channel exists because the one above it left
something behind. The interesting part is how steeply the yield drops — that gradient turns out to
be the project's actual scientific result, discussed in II.6.

| Channel | What it reaches | Yield |
|---|---|---|
| **Europe PMC full text** (body XML by PMCID) | the open-access core of the corpus | 7,781 read · 58.7% contain a code |
| **EPMC text-mined annotations** (keyed by PMID, not just PMCID) | papers PMC cannot redistribute at all — the only channel for closed-access, PMID-only rows | ~11% hit · every mined string re-checked against the dictionary |
| **NCBI E-utilities fallback** (for the EPMC-404 rows) | non-OA papers return front matter with a 200, not an error — mining it yields real codes | 647 read · 34.3% contain a code |
| **Unpaywall** (legal OA copies outside PMC) | publisher gold/hybrid, institutional repositories, bioRxiv | 264 read · 14.0% contain a code |
| **403 Tier-1** (alternate OA location for bot-blocked pages) | publishers that refuse the crawler but have a green/repository copy | 1,462 read · 27 codes |
| **403 Tier-2** (headful, fingerprint-masked Chromium) | Akamai/Cloudflare walls at Wiley, OUP, MDPI that headless can't clear | 648 attempted · 8 codes · 70 clean full reads |
| **Supplementary files** (EPMC supplementaryFiles) | rows whose text says "the data is in the supplement" | 15 of 166 served · 0 codes |

Every response is cached, so the extraction rules can be revised against all of this without
touching the network again.

*Worth flagging as a real cost:* the invalid NCBI API key inherited from the old pipeline returned
HTTP 400 on every request, which for hours looked exactly like rate-limiting. The Python port runs
unauthenticated on purpose and **asserts a 200 in preflight** before trusting any credential.

## II.3 What we're entitled to claim

Recovering codes is the easy half. The harder half is the 7,511 rows that still have no code —
because "we found nothing" and "we didn't really look" are completely different statements to a
curator, and the old pipeline collapsed them into the same blank cell.

So the deliverable carries a `certainty` column with exactly three values and a deterministic rule
behind them. A no-code row is only called **verified-empty** if we hold a full text of **≥ 40,000
characters** that was read end-to-end and in which both the production dictionary and an independent
looser detector found nothing. Anything short of that is **needs-review**: not "empty", but "not yet
read well enough." There is no probabilistic middle ground and no judgement call.

*Why 40k:* real microbiome papers run 30k–80k characters of body text. A "read" that returns less is
a landing page, a cookie wall, an abstract, or a silently blocked fetch. This threshold was adopted
after discovering that **170 of 262 browser reads had returned under 500 characters** while being
counted as successful reads. The rule eliminates that entire error class by construction.

**The corpus, by what we can defend** (`articles.out.csv` · 13,092 rows · recomputed 2026-08-10):

| `certainty` | rows | share |
|---|---:|---:|
| `coded` — an accession code is in the cell | 5,581 | 42.6% |
| `verified-empty` — a ≥40k full text read end-to-end, no accession | 3,434 | 26.2% |
| `needs-review` — paywalled, restricted, blocked, or read too shallow to trust | 4,077 | 31.1% |

**Every code-less bucket** (7,511 rows):

| Flag | Rows | Verified-empty | Needs-review | What it means |
|---|---:|---:|---:|---|
| `OA_AVAILABLE` | 2,678 | 1,902 | 776 | a free copy exists outside PMC — the click-through URL is in the row |
| `PAYWALLED` | 2,319 | 41 | 2,278 | full text only at the publisher; needs library access |
| `ON_REQUEST` | 991 | 844 | 147 | the paper says data is available on request — a true floor |
| `NO_ACCESSION` | 813 | 482 | 331 | read it, found none; the 331 were read too shallow to trust |
| `HUMAN_CAN_GET` | 434 | 0 | 434 | a restricted PMC copy exists; a human can retrieve it, a machine can't |
| `SUPPLEMENT` | 188 | 140 | 48 | "data is in the supplement" — supplements fetched, nothing found |
| `NO_IDENTIFIER` | 36 | 0 | 36 | no DOI, PMID or PMCID to act on |
| `NO_DATA` | 28 | 25 | 3 | the paper states no new data was generated |
| `NO_FULLTEXT` | 24 | 0 | 24 | publisher restricts full text everywhere; metadata only |

The practical output of this is that the no-code pile is no longer a dead column. It is a
**prioritized curator worklist**: thousands of rows carry a URL that opens the free full text in one
click, the paywalled rows are named as paywalled, and the rows where the paper itself says the data
was never deposited are kept as such instead of being mislabeled *go open this*.

*A subtle ordering fix that mattered:* a definitive read (`ON_REQUEST`, `NO_DATA`, or
`NO_ACCESSION` from a genuine full-text read) now outranks "an open copy exists." Without that
precedence, **822 rows the old pipeline had already read** and marked on-request/no-data would have
been relabeled `OA_AVAILABLE` — sending curators to open papers that already say the data isn't
deposited. The OA URL is still appended to those rows so the claim can be checked.

## II.4 Audits designed to break our own output

A confidence column is only worth the audit behind it. Three separate passes were run specifically
to falsify the pipeline's own "we found nothing" claims.

**1 · The looser detector.** Every floor row's exact cached text was replayed through a detector
deliberately *weaker* than production — guards disabled, whitespace-tolerant — so that anything the
real pipeline had rejected would resurface. Across 1,853 floor rows it produced 127 candidate hits.
**95 of them were the journal name "Cancers" glued to a citation number** (`cancERS14040909`
matching the SRA pattern) — the guards were correctly rejecting them. Only about 8 were genuine
misses, roughly **0.4%**, nearly all caused by a space splitting an accession (`PRJNA 795467`).
Fixed and merged.

**2 · The read-the-papers review.** Automated audits share the automation's blind spots, so six
independent reviewers were given 24 randomly sampled `verified-empty` papers to read in full and try
to disprove. **23 of 24 confirmed empty.** The single miss was the interesting one: an SRA
*submission* ID of the form `SUB######`, a format the dictionary had never contained. Sweeping the
corpus for that pattern found 34 more rows, **29 of which had been wrongly marked verified-empty**.
Pattern added, rows merged, count moved 5,547 → 5,581.

This is the audit that justifies the others: a 96% confirmation rate is a real number, and the 4%
failure was a **whole missing format class** — precisely the kind of error a looser version of your
own regex can never find, because it shares your assumptions.

**3 · Provenance, corrected by chronology.** The `own` vs `reused` label came from regex over the
text around each code, and it had a known systematic bias: a cross-check found **23 conflicts, every
single one in the same direction** — the regex said "deposited," the evidence said "reused" — and
they clustered in controlled-access repositories (EGA 12, dbGaP 4, legacy SRA 3, HRA 1). The cause
is linguistic, not technical: in EGA and dbGaP, depositing and citing are phrased identically.
*"Data are available in the EGA under accession EGAS00001002702"* is what you write in both cases.

The fix needs no network and no model — only the corpus itself. If an accession appears on multiple
rows, **only the earliest paper can be the depositor**; the rest are reusing. That offline pass
produced **66 provenance corrections**, concentrated in EGA/dbGaP exactly as predicted, plus **9
dbGaP version collapses** where `PHS001768` and `PHS001768.V1.P1` were being written as two codes
with contradictory provenance.

Throughout all of this, one invariant held: **a human-entered code is never overwritten**, verified
at zero changed. Three recovered codes scraped from repository landing pages rather than stated in a
paper body are tagged `[review]` instead of being counted, because their evidence is weaker than the
rest.

## II.5 Three assumptions the data killed

The inherited handoff document named three high-confidence opportunities. Each was tested against
real papers rather than accepted, and all three were wrong. Two would have been substantial wasted
builds.

| The assumption | What the data showed |
|---|---|
| "DOI-only rows are dead ends." | **False.** 86% resolve to a PMID and 68% reach full text via Europe PMC. The largest supposedly-dead bucket was mostly reachable. |
| "Crossref is the obvious next lever for the ~2,400 DOI-only rows." | **False.** Its dataset-relation field was empty on every probe. Abstract-only fallback; not a channel. |
| "~600–800 codes are sitting in supplementary files" — described as the single biggest opportunity. | **False.** Measured hit rate 2–5%; of 166 attempted, only 15 supplements were served at all and they contained *zero* accessions. Supplements hold the results, not the accession. |

The supplement lead was the one the handoff was most confident about. Probing it cost an afternoon;
building it would have cost a week.

## II.6 The finding: 59 → 34 → 14

Across everything the pipeline fully parsed, **55.5% of papers contain an accession code**. But that
headline rate collapses as you move down the recovery ladder toward the papers that are still
unanswered — and the shape of that collapse is the result worth putting in a discussion section.

| Channel (population it handles) | Parsed full texts | Contain an accession |
|---|---:|---:|
| Europe PMC — all PMC papers | 7,781 | **58.7%** |
| NCBI fallback — papers EPMC lacks | 647 | 34.3% |
| Unpaywall, first read — OA fallback for the floor | 264 | **14.0%** |

The naive reading is that open-access copies are somehow worse. They aren't — a repository PDF is
the same text as the publisher's. What the gradient measures is **selection**. By the time a paper
reaches the fourth channel, it is one that no easier channel could answer, and those are
disproportionately clinical and subscription studies that never deposited sequence data under a
citable accession at all.

> **Availability is largely solvable. Presence is not. The papers that stay unanswered are the ones
> where reading them doesn't help — increasingly because there was never an accession to find.**

This splits the remaining work cleanly in two, and only one half is an engineering problem. Roughly
55% of the unanswered rows with a DOI have a free copy somewhere outside PMC, and nearly all the
rest are ordinary paywalled papers a human with library access opens in seconds — that's a
**curation** task, and the deliverable now hands it over pre-sorted with URLs attached. The other
half is a **fact about data-sharing practice in the field**: the papers we *can* read mostly just
don't contain an accession. That is a finding, not a pipeline limitation, and it is the more
interesting of the two.

### Honest limits

- The PMID-only bucket — the majority of target rows — is closed-access with no PMC full text. Even
  when opened through other channels it rarely contains an accession. This is the bulk of the
  residual.
- `PAYWALLED` rows are essentially unverified by machine (41 of 2,319 read to the 40k bar). They are
  labeled as needing a human, not as checked.
- The provenance `own`/`reused` label remains regex-derived outside the chronology correction. The
  systematic bias is documented and partially fixed, not eliminated.
- Three recovered codes rest on landing-page evidence rather than the paper body, and carry a
  `[review]` tag for that reason.

---

# Part III — What the two projects share

One is an LLM benchmarking problem over 15 papers; the other is a network-and-regex problem over
13,092. They converged on the same four habits, which is the part most likely to transfer to
whatever comes next.

1. **Put a cache between the expensive step and the uncertain one.** In Project 3 it's the disk
   cache splitting fetch from extract; in Project 2 it's cached model outputs scored repeatedly
   under different metrics. Both turn "we should re-run everything" from a multi-day decision into a
   coffee-length one — and cheap re-runs are what make auditing your own work affordable enough to
   actually do.
2. **One source of truth for rules.** One dictionary module; one eval harness with the model as a
   flag. Divergent copies aren't a bug you fix, they're a bug you keep re-fixing.
3. **Measure the measurement before blaming the system.** Project 2 spent months believing the
   models were mediocre when the benchmark was incomplete and the metric was a spell-checker.
   Project 3's inherited document listed three confident opportunities, all three of which
   evaporated on contact with real papers. In both cases the cheap experiment came first and changed
   the plan.
4. **Separate "no" from "we don't know."** This is the same idea as *answer every row, flag
   ambiguity* — and it's why the leaked judge experiment is written up as a leak, why
   `verified-empty` requires 40,000 characters of evidence, and why 4,077 rows are labeled as
   unresolved rather than quietly counted as empty. A pipeline that overstates its confidence is
   worse than one that answers less.

> **Both projects' best results came from an experiment designed to prove the pipeline wrong.**

### Where this goes next

- **Project 2:** settle the two annotation-policy questions, then run the chosen extractor over the
  full corpus and assemble the first knowledge-graph edges — with taxon canonicalization at
  assembly, and validation against Peryton and Disbiome rather than against 15 papers.
- **Project 3:** the extraction side is done and audited. What remains is a curation pass over the
  flagged worklist, in priority order — the 776 open-access rows with a URL and no deep read first,
  the paywalled bulk with library access second.

---

## Provenance of the numbers

- **Project 2 · source** — `proj_2_attempt3/dsmlp_model_prompting/eval-v2/`:
  `results/leaderboard.csv`, `experiments/RESULTS.md`, `experiments/JUDGE_RESULTS.md`,
  `improvement_ideas_findings.md`. Runs on UCSD DSMLP (H100, llama.cpp GGUF).
- **Project 3 · source** — `proj3-MMC2-scraping/`: the `accession/` pipeline, `SUMMARY-for-Sam.md`,
  `NO-ACCESSION-REPORT.md`. Deliverable `articles.out.csv`; the live sheet is never written to.
- **Corpus counts** — recomputed directly from `articles.out.csv`: 13,092 rows — coded 5,581,
  verified-empty 3,434, needs-review 4,077. `SUMMARY-for-Sam.md` and `NO-ACCESSION-REPORT.md` were
  written against an earlier snapshot (5,117 and 5,185 coded, before the browser-recovery and
  `SUB`-format merges) and have been brought up to these figures, so all three agree.
- **Status** — Project 2 extraction is benchmarked and a model is chosen; knowledge-graph
  construction has not started. Project 3 extraction is complete and audited; the remaining work is
  human curation.
