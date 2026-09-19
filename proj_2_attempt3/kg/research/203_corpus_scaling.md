# 203 — Corpus scaling: what 10x actually buys

**Status: §1 measured and complete. §2–3 are scoped notes, not a full survey.**

*2026-09-19. §1 is an original measurement on `graph.json`, not a literature
review — five research agents died mid-run on this brief, and the crux turned out
to be arithmetic we already had the data to settle. Measuring beat surveying.*

---

## 0. The answer in three lines

**10x the corpus is almost exactly the threshold at which the discordance
question becomes answerable, and it lands at z ≈ 2.3 — significant, not
comfortable.** The reason is not the number of papers. It is that each paper
yields only ~11 observations, so **per-paper precision saturates**, and past
~10x the only remaining gain is √K.

**Do not scale for breadth. Scale for depth** — target papers on taxa and
diseases already in the graph. §1.4.

---

## 1. What 10x actually buys

### 1.1 The structure, measured

From `graph.json`:

| | |
|---|---|
| association edges | 2,008 |
| paper–edge observations | 3,077 |
| contributing papers | 271 |
| observations per paper | **11.35** |
| papers per edge | **1.532** |
| edges resting on exactly 1 paper | **1,561 (77.7%)** |

Consistent with `206_edge_confidence.md`, independently recomputed.

### 1.2 Rarefaction — how the structure scales

Subsampling k papers, 40 draws each:

| k | edges | obs/edge | % 1-paper | replicated (≥2) | decisive obs | decisive/paper |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 311 | 1.088 | 92.4% | 24 | 51 | 1.70 |
| 60 | 589 | 1.171 | 87.6% | 73 | 173 | 2.88 |
| 120 | 1,024 | 1.316 | 82.4% | 180 | 503 | 4.19 |
| 180 | 1,450 | 1.403 | 80.2% | 287 | 870 | 4.83 |
| 240 | 1,820 | 1.493 | 78.5% | 391 | 1,289 | 5.37 |
| **271** | **2,008** | **1.532** | **77.7%** | **447** | **1,516** | **5.59** |

Fitted exponents (`y ~ k^b`):

- **all edges: b = 0.840** — sublinear. New papers bring new taxa faster than they
  revisit old ones, so edge count grows slower than the corpus.
- **decisive observations: b = 1.525** — *superlinear*, and this is the
  load-bearing number.

**Why decisive observations grow superlinearly.** An observation only becomes
decisive when its edge acquires a *second* paper. The pool of decisive
observations is therefore a coincidence count, and coincidences grow roughly
quadratically until saturation. This is the opposite of the intuition I started
with — that new papers would mostly mint new single-paper edges and leave the
graph as thin as before. **That intuition is wrong and the data say so:** the
1-paper share falls monotonically 92.4% → 77.7%, and decisive observations grew
**29.7x** while the corpus grew 9x.

### 1.3 The ceiling that kills the extrapolation

Naively extrapolating b = 1.525 to 10x gives 33x more decisive observations and a
5.8x MDE improvement. **That extrapolation is invalid**, and the reason is a hard
structural bound:

> An observation cannot be decisive more often than it is observed. Decisive
> observations per paper are capped by **observations per paper = 11.35**.

Currently 5.59 — **49% of the ceiling**. So the superlinear regime is already
half spent. Growth must bend to linear (b → 1) as the 1-paper share approaches
zero. Any power calculation that rides b = 1.525 out to 10x is fantasy.

### 1.4 The power calculation, done as variance components

The quantity of interest is the **between-paper SD of discordance: 3.4pp on a
27.6% base** (`FINDINGS_paper_discordance.md`), whose cluster-bootstrap CI
[0.0, 6.0] includes zero. That is a variance-component problem, not a
difference-in-means problem, and it must be posed as one.

- signal: τ² = 0.034² = **0.001156**
- per-paper binomial noise: σ² = p(1−p)/n_dec, p = 0.276
- SE(τ²) ≈ σ²·√(2/(K−1))

With `decisive/paper` held to its 11.35 ceiling:

| corpus | K | dec/paper | σ² | I² | SE(τ²) | **z** |
|---:|---:|---:|---:|---:|---:|---:|
| 271 | 271 | 5.59 | 0.0357 | 3.1% | 0.00308 | **0.38** |
| 542 (2x) | 542 | 8.0 | 0.0250 | 4.4% | 0.00152 | **0.76** |
| 1,355 (5x) | 1,355 | 10.3 | 0.0194 | 5.6% | 0.00075 | **1.55** |
| **2,710 (10x)** | 2,710 | 11.0 | 0.0182 | 6.0% | 0.00049 | **2.34** |
| 5,420 (20x) | 5,420 | 11.2 | 0.0178 | 6.1% | 0.00034 | **3.37** |

**z = 0.38 at the current corpus.** That is the arithmetic explanation for 24
variables and 24 nulls: the design has essentially no power, and the null was
never evidence of absence. It also independently reproduces the reported
cluster-bootstrap CI that includes zero, which is a good sign the model is right.

**Read the table carefully, because it does not say "scale up and win":**

- **2x and 5x are wasted effort** for this question. z = 0.76 and 1.55.
- **10x reaches z = 2.34** — past 1.96, but a result you would not want to stake a
  claim on.
- **20x gives z = 3.37**, and buys it almost entirely through K, since dec/paper
  is pinned at the ceiling by then.
- I² only moves 3.1% → 6.1% across the entire range. **The heterogeneity really is
  tiny**, and no corpus size makes it large. Scaling buys the precision to
  *measure* a small effect; it does not make the effect important.

### 1.5 The actionable consequence: scale for depth, not breadth

Because edges grow at b = 0.840 while decisive observations grow at b = 1.525,
**a paper's value depends entirely on whether it lands on an edge that already
exists.** A paper introducing 11 new taxa in a new disease adds 11 edges of
confidence `provisional` and **zero** decisive observations. A paper on a
well-studied taxon–disease pair adds decisive observations directly.

This reframes the sourcing question. The standard move — broaden the query,
maximise recall, screen everything — is the **wrong** strategy here. The right
one is to enumerate the 447 replicated and 1,561 single-paper edges already in
the graph and retrieve literature *targeted at those pairs*.

> **Falsifiable test, ~1 day, no GPU:** take the 45 MAIN_DATA papers added most
> recently and classify each observation as landing on a pre-existing edge or
> minting a new one. **If targeted retrieval does not raise the pre-existing
> share well above the 49% baseline implied by `dec/paper = 5.59 / 11.35`, the
> depth strategy is not working and breadth is no worse.**

### 1.6 Verdict

**Scaling to 10x makes the discordance question answerable and nothing more —
z ≈ 2.3, I² ≈ 6%.** If that question is the reason for scaling, the honest
statement is that it costs a 10x corpus to earn a marginal result about an effect
that is small even when detected.

Two other things are worth saying plainly:

1. **The 24 nulls were never evidence of absence.** z = 0.38 means the study could
   not have found the effect it was looking for. That belongs in
   `FINDINGS_paper_discordance.md` as a power statement, and it is a cheap
   correction available today with no new papers at all.
2. **The better reasons to scale are not this question.** More papers thicken
   `provisional` edges (77.7% of the graph, and per `206` their disjoint-source
   agreement is 0.475/0.383 — the weakest thing the project ships). Raising
   per-edge evidence is a direct quality improvement and does not depend on the
   discordance question resolving at all.

---

## 2. Screening is the bottleneck, not extraction

Per `208_extraction_frontier.md` §1, a full corpus extraction pass is
**~1.7 GPU-hours**. Extraction is effectively free. At 3,000 papers it is
~18 GPU-hours — still not the constraint.

**The constraint is screening**, and it is human. The project's validated screen
adjudicates human-case-control vs animal / review / case-report / no-control,
with **24 blinded controls** shuffled in so its own reliability is measurable —
the standard being that *an unvalidated classification is not a result, it is a
pile of opinions*. It has run on 273 papers.

Scaling that screen 10x is the real cost, and the validation cannot be dropped:
the project has already had **one screened-out rat study leak into the graph**
through a punctuation-level dedup gap, so the screen's failure modes are known to
be live rather than theoretical.

The live design question is what fraction can be automated while keeping blinded
controls in every batch. Established options are ASReview, Rayyan, and LLM-based
screening; this section is **not** a survey of them and one is still owed. What
can be said without it: the screen already has a measured-reliability harness, so
any automation proposal has a ready-made evaluation, and **no automated screen
should be adopted without running it against the existing 24 controls first.**

---

## 3. Sourcing and deduplication — scoped notes

**Licence and provenance: see `205_dataset_release.md`, which supersedes anything
here.** The key fact for planning: the existing 2,026-paper `MAIN_DATA.json` was
assembled by scraping with **no licence provenance tracked**, so it is not a clean
base to build on. §1.5 also implies breadth-first bulk acquisition is the wrong
strategy regardless.

**Full-text availability is the practical limit.** This project already found 31
of its own target papers have no free full text, and 20 open-access recoveries
are still outstanding. Any 10x plan must assume a substantial fraction of
identified papers are unreachable, and should measure that fraction before
committing.

**Deduplication is a known live failure mode, not a hypothetical.** 13 papers sat
in the extraction table under two spellings differing only by punctuation —
`filter_maindata.norm()` folded curly quotes and dashes but kept a trailing full
stop, while `build_kg.dedup_rows` stripped every non-alphanumeric. At 3,000
papers this gets worse, and it is the mechanism that let a screened-out animal
study into the graph. Canonicalise on DOI/PMID rather than title, and treat
title-based matching as a fallback that must be audited.

---

## What this document does not cover

Owed, and not written: a survey of automated screening tools with measured
recall/precision; retrieval-strategy comparison (boolean/MeSH vs embedding vs
citation-graph snowballing); per-source coverage figures for PMC OA / Europe PMC /
OpenAlex. §1 was prioritised because it is the question that decides whether any
of the rest is worth doing, and because it could be settled by measurement rather
than survey.
