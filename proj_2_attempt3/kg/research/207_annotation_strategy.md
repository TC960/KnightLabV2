# 207 — Annotation Strategy: Where to Spend Scarce Annotator-Hours

*Status: in progress. Sections are written in priority order and committed as they are finished.*

**Scope.** This is a review of how to allocate the project's genuinely scarce input — a few
biology-student annotators for tens of hours, not hundreds. It assumes the economic fact
established in `research/208_extraction_frontier.md` §1: a full-corpus extraction pass is
~1.7 GPU-hours, so **compute is cheap and adjudication-hours are the bottleneck**. An
experiment costing 2 GPU-hours and 30 annotator-hours is *more expensive* than one costing
10 GPU-hours and 2.

## Contents

1. Annotate what, exactly — the ranked allocation
2. A concrete protocol for the single highest-value task
3. Measuring the annotators
4. Active learning, honestly assessed
5. LLM-assisted annotation
6. Tooling

---

## 1. Annotate what, exactly — the ranked allocation

**The criterion is not volume of labels. It is: what could the project say afterwards that it
cannot say today?** Applied honestly, that criterion kills most of the candidate list. Below,
each option is costed in annotator-hours and scored by the claim it buys.

### The ranking

| # | task | hours | verdict | the claim it buys |
|---|---|---:|---|---|
| **1** | **(g)** AIS citation-support — does the cited paper support the edge? | **~20** | **DO THIS** | the first fidelity number that covers the **whole** graph, including the 580 edges no existing instrument can see |
| **2** | **(c)** the 11 pairs contradicted by *both* Disbiome and Peryton | **~6** | **DO, second** | turns the "disjoint-literature vs misreading" decomposition from an inference into an adjudicated verdict, at the hardest tail |
| **3** | **(b)** the 3 papers holding 12 of 15 confirmed edge-recall misses | **~3** | **DO, as onboarding** | no new claim — but it is the only task with *pre-existing adjudicated answers*, so it doubles as the annotator calibration test |
| **4** | **(e)** zero-edge papers | **~1** | **finish the tail only** | closes the 7 `unclear` papers; the paper-level recall range ≥96.1–99.6% stops being a range |
| **5** | **(d)** the 212 unresolved taxa | **2, triage only** | **DEFER** | blocked on the NCBI taxdump, not on hours |
| — | **(a)** more papers for a general gold | 20–40 | **REJECT** | a better estimate of a number nobody should quote |
| — | **(f)** effect-size extraction pilot | 10+ | **REJECT** | a ceiling that `research/11_effect_size_extraction.md` has already estimated at 10–15% usable yield |

Total for tasks 1–4: **~30 hours**, inside a 20–40 hour budget, with 1–2 hours of slack.

### 1 — (g) AIS citation-support. Do this first, and give it two thirds of the budget.

The argument is not that attribution is interesting. It is that **every fidelity instrument this
project owns is restricted to the same narrow slice of the graph, and a human with the PDF is the
only instrument that escapes it.**

- The direction audit scores **209 of 3,077 observations** — the strict-witness subset where one
  sentence names exactly one taxon with one polarity. The rejected T2 tier established that regex
  cannot widen this (`FINDINGS_direction_audit.md`).
- The Disbiome/Peryton same-paper figures cover only the ~half of decisive pairs the curators
  happen to share (`FINDINGS_independence.md`).
- **580 edges — 28.9% of the graph — rest on one paper *and* have no own-result prose witness at
  all.** For 301 of them no kept sentence names the taxon. That is not a silent paper; it is a
  blind instrument. `relation_sentences.json` keeps 7.5k of 106k sentences, and a taxon reported
  only in a LEfSe table appears in none of them.

**A biology student with the paper open can read the table.** No automated instrument in this repo
can, and none is planned. That is the entire case, and it is decisive.

What the project could then claim, and cannot today:
- *"Reading fidelity is X% across the whole graph, including the table-only and single-paper
  edges"* — today the honest statement is "≥86.6% on the 7% of observations we can see."
- *"The `provisional` tier is/is not a genuinely weaker tier."* The discordance null
  (`FINDINGS_direction_audit.md`) compared `own` vs `background` and found nothing at an MDE of 8.4
  points — but it could say **nothing at all** about `silent`, because a leave-one-out majority
  does not exist for single-paper edges. This task is the only way to test the tier that 580 edges
  sit in.
- A corpus-wide interval from ~19 hours of labels, via **prediction-powered inference**
  (Angelopoulos et al., Science 2023) — machine-label the remaining thousands of triples, correct
  with the human sample, report a CI that stays valid even if the machine checker is mediocre.
  Nothing else on this list has that multiplier.

**It is runnable tomorrow and does not wait on the QA layer.** §4.6 of `201_kg_grounded_qa.md`
frames this as auditing RAG answers, but the annotator's unit there is *(claim, cited paper)* with
the answer prose deliberately hidden. That unit **is edge provenance** — it exists in `graph.json`
today. Building the answer layer changes nothing about what the student does.

### 2 — (c) the 11 doubly-contradicted pairs. Six hours, at the sharpest tail.

These are the 11 taxon-disease pairs where our direction contradicts **both** Disbiome and Peryton.
Today they are an undifferentiated residue inside the 73.0%/72.5% agreement figures. Each pair
needs its source papers read on both sides: is our reading of *our* paper wrong, or did the
curators read *different papers*?

That is the exact question `FINDINGS_independence.md` answered by decomposition (≈90% reading
fidelity × ≈55% cross-literature reproducibility) but never by adjudication — and it answers it
where the decomposition is most likely to be wrong. What it buys: *"of the worst 11 disagreements
in the graph, k are cross-literature and 11−k are ours."* Even a rough split here is more
informative than the same hours spent anywhere in the agreeing majority.

**Be honest about the ceiling: n = 11.** A 10/11 result carries a 95% CI of roughly [0.59, 0.99].
This is a case-series, and must be written up as one. It is worth six hours *because* it is six
hours, not because it is conclusive.

### 3 — (b) the 3 papers with 12 of 15 misses. Do it, but for the wrong-looking reason.

As a data fix this is near-worthless and `FINDINGS_edge_recall.md` says so: adding ~60 observations
to 3,077 moves agreement by **less than this corpus can resolve** (~0.013). Do not sell it as an
accuracy gain.

Do it because it is the only task on this list where **the right answers already exist**: 15
confirmed misses, each with a required verbatim quote, adjudicated with the extraction gate
applied. Hand a new annotator those 3 papers and the gate in writing, and their output is directly
scoreable. Three hours buys a *measured* annotator, which every subsequent hour then depends on.
This is the training set for the humans, not for a model.

### 4 — (e) zero-edge papers. One hour. The tail, and only the tail.

Already 97% done: of 42 non-contributing papers, 9 have no relation-bearing sentence and 33
adjudicate to 14 correct refusals, 4 explicit negative results, 4 background-only, **7 unclear**,
1 confirmed miss. Spend an hour on the 7 unclear and the paper-level recall figure collapses from a
range (≥96.1%, or 99.6% counting confirmable misses) to a single defensible number. Do not
re-adjudicate the other 35.

### 5 — (d) the 212 unresolved taxa. Two hours of triage, then stop.

The tempting framing — "212 of 883 taxa have no taxid, a student can look them up" — was already
tested and mostly refuted (`FINDINGS_taxon_spelling.md`). The tractable part is done: 12
punctuation splits and 33 verified paper misspellings were folded by a curated table. What remains
is **clade labels** (`SMB53`, `cc115`, `PAC000195_g`) which have no taxid by construction, and real
taxa **absent from the cached taxdump** (`Anaerostignum`, `Mogibacteriaceae`) — blocked on
`ftp.ncbi.nih.gov` returning 403, not on human time. A student cannot unblock a network fetch.

Two hours to sort the 212 into *(unresolvable clade label / needs taxdump / genuinely missed)* is
worth it, because the third bucket is probably small and would otherwise never be found. Anything
beyond triage is misallocated. Note also the standing rule: resolution must go through
`taxonomy.py`, never through another database's stored id.

### Rejected — (a) more papers for a general gold. This is the default answer and it is wrong.

At a realistic 45–90 minutes per full-text paper, 30 hours buys **20–40 papers**. Added to a
250-paper gold in which **162 papers have blank taxa columns**, that changes nothing about the
reference's reliability — it enlarges a corrupted set by ~15%. The evidence that the set is
corrupted is threefold and already in hand: the blank columns, 72 taxa an Opus 4.8 re-annotation
found that the humans missed, and a 15-paper benchmark that rose **0.64 → 0.84 F1 once the gold was
corrected**. "F1 0.680" is agreement with a flawed reference, not accuracy.

There is a *coherent* version of this: re-annotate the existing 250 rather than extend them. That
is 190–375 hours. It is off the table by an order of magnitude, and it would buy a metric the
project has already replaced with three gold-free instruments. **Skip it. If a reviewer demands a
gold-standard F1, report the existing one with the caveat, and point at the gold-free numbers.**

### Rejected — (f) an effect-size extraction pilot.

`research/11_effect_size_extraction.md` costed this end-to-end: the dominant reported statistic
(LEfSe LDA) is not a standardised effect size and is not poolable even if extracted perfectly;
magnitude numbers are reachable from body text for at most **20–30%** of observations; the best
comparable LLM benchmark (RCT numerical extraction — a far more standardised genre) tops out at
~65%/~49% exact match; compounding gives **10–15% usable harmonised effect sizes**. Annotator hours
spent here would be spent confirming a ceiling that has already been estimated, on a capability the
project has decided not to build. If magnitude ever becomes a priority, the narrow move is
alpha-diversity SMDs for the subset of papers reporting them — a different, smaller task.

---

## 2. The protocol for task 1 — edge–source support, blinded

**Hypothesis under test.** *The 580 edges that rest on one paper with no own-result prose witness
are supported by their source paper at the same rate as edges the automated instruments can see.*

This is deliberately not "measure fidelity again." Fidelity on prose-witnessed observations is
already ≥86.6% [81.7, 91.3] by machine, for free, on 3,077 observations. Human hours must be spent
where the machine is blind — and the blind region is a quarter of the graph.

The protocol below is `201_kg_grounded_qa.md` §4.6 with the sampling frame changed from RAG answers
to graph edges, and the adjudication mechanics lifted verbatim from `FINDINGS_direction_audit.md`,
which ran twice-independent adjudication in this exact domain and got 25/28 exact agreement.

### 2.1 Sampling frame — two arms, blinded to the annotator

Every observation in `graph.json` already carries a provenance class from
`audit_direction_witness.py`:

| arm | class | population | sampled |
|---|---|---:|---:|
| **A — WITNESSED** | `own` (≥1 own-result sentence names the taxon) | 2,109 | **150** |
| **B — BLIND** | `background` (585) + `silent` (383) | 968 | **150** (≥90 from `silent`) |

Sample **papers first, then ≤5 claims per paper** — capping claims-per-paper holds the design
effect near 1.5, and the annotator reads each paper once, which is where the time saving comes
from. Target ~70–90 distinct papers across both arms.

**The arm label is never shown.** If an annotator knows an item is "the hard kind," they hunt
harder, and the comparison dies.

### 2.2 What the annotator sees

One screen per item:

1. **The claim**, rendered plainly: *"In Parkinson's disease, **Prevotella** is DEPLETED relative to
   healthy controls."*
2. **A link to the paper** (PMC full text, pre-fetched; supplementary files included if the
   publisher hosts them).
3. **The extraction gate, in writing, on every screen** — reported statistical significance
   required; disease vs healthy control only; the paper's own cohort, not a cited one. Without this
   the audit manufactures errors: the first pass of the recall audit reported 4 misses where 1 was
   real, precisely by skipping this step.

**What they never see:** the answer prose, `n_papers`, the contested flag, the confidence tier, the
extractor's own supporting sentence, the arm, or whether the item is a decoy. Showing the
extractor's sentence would make the reference a function of the system under test — the
contamination failure this project has already hit once (§3).

### 2.3 Response, and the decoy

Each claim is presented **twice in the session, once inverted** (the decoy: same taxon, same
disease, opposite direction), in randomised order, separated by other items. Responses:

- **SUPPORTED** — requires a verbatim quote, *or* a locator (`Table 2, row 14`, `Fig 3B`).
- **CONTRADICTED** — same evidence requirement.
- **NOT-ADDRESSED** — no evidence either way that passes the gate.

Plus one mandatory field on every non-`NOT-ADDRESSED` verdict: **where the evidence lived** —
`main text` / `table` / `figure` / `supplementary`.

That last field is free and it answers a question nobody has answered: *how much of this graph
rests on evidence a prose-level pipeline structurally cannot reach?* If the BLIND arm resolves
overwhelmingly to `table`, table extraction moves from "someday" to "the single highest-value
engineering task," and the 580 `provisional` edges are vindicated rather than suspect. If it
resolves to `NOT-ADDRESSED`, the opposite.

**NOT-ADDRESSED is never folded into CONTRADICTED.** The gate means an annotator can *confirm* that
a paper supports a claim but can never *establish* that it does not — the sentence may have been
gated out rather than absent. This instrument can confirm support; it cannot refute it. Every
write-up must restate that asymmetry.

### 2.4 Volume and hours

| component | items | rate | hours |
|---|---:|---|---:|
| primary annotation (A + B, decoys interleaved) | 300 claims | ~3 min/claim | **15.0** |
| double-annotation overlap (second annotator) | 100 claims | ~3 min | **5.0** |
| training + calibration on task 3's 3 papers | — | — | **1.0** |
| **total** | | | **~21** |

At 150 per arm the per-arm 95% CI is ±4.8 points nominal at a base rate near 0.90, ±5.9 after a
design effect of 1.5 from paper clustering. The **between-arm MDE is ≈12 points at 80% power**
(≈15 after clustering). Sizing the arms at 150 each rather than putting all 300 into one
whole-graph estimate is the deliberate trade: a ±5-point global number that cannot see *where* the
weakness is, versus a ±6-point number per arm that can.

All intervals are **paper-cluster bootstrapped**, as in `FINDINGS_direction_audit.md`. Item-level
CIs on clustered data are wrong and this repo has already learned that once.

### 2.5 Agreement and quality control

- **Primary agreement statistic: raw agreement on the 100 doubled items, always reported with the
  base rate and n.** Cohen's κ reported alongside, with its CI, as a secondary. §3 explains why
  that ordering, and why the decoys make κ usable here when it usually is not.
- **Decoy specificity, per annotator, pre-registered as an exclusion rule.** A decoy should read
  CONTRADICTED. An annotator whose decoy-correct rate falls below **0.85** is acquiescing, and
  their SUPPORTED labels are uninterpretable — retrain and re-run that block. This is a free
  within-annotator specificity estimate, which is the main reason the decoys exist.
- **Verbatim-quote enforcement.** Every miss verdict in the recall audit required a verbatim quote,
  and **36 raw verdicts became 29 verbatim and then 15 confirmed** once the quote and the gate were
  checked. Roughly 60% of unenforced verdicts did not survive enforcement. Quote strings are checked
  against the paper text by script, not by eye.
- **Adjudication.** Disagreements on the doubled items are resolved by a third read that sees both
  verdicts and both quotes, with categories fixed in advance.

### 2.6 What falsifies what — stated before any data is collected

> **H1 (primary).** Support rate in arm B is not materially below arm A.
> **Falsified if** arm B is **≥12 points below** arm A (the design's MDE), cluster-bootstrapped.
> **Consequence if falsified:** the `provisional` tier is a genuinely weaker tier and must be
> reported as a defect rate, not a label; the 580 single-paper unwitnessed edges need downweighting
> in the QA answer policy and in any downstream analysis.
> **Consequence if not falsified:** the tier reflects instrument blindness, not evidence weakness —
> which is the currently-assumed-but-untested reading in `FINDINGS_direction_audit.md`.

> **H2 (secondary, absolute).** Whole-sample support rate is consistent with the extractor's
> measured reading fidelity, ≥86.6% [81.7, 91.3].
> **Falsified if** the 95% CI for support rate lies **entirely below** that interval. That means the
> graph attributes claims to papers that do not make them — a provenance defect, and the only
> failure this instrument uniquely detects.
> **A rate *at* the extractor's fidelity is the ceiling, not a disappointment.** The graph cannot be
> more faithful than the extraction that built it.

> **Negative result that is still publishable:** a high support rate in both arms with the BLIND arm
> resolving mostly to `table`/`figure`. That is a clean, quantified statement that the project's
> prose-level instruments are blind to ~29% of its own evidence base while that evidence is sound —
> which is both a limitation section and a roadmap.

**Pre-register §2.6 in the repo before the first item is annotated.** The prompt-gate experiment was
pre-registered here for the same reason, and the recall audit's wrong headline is what happens
without it.

---
