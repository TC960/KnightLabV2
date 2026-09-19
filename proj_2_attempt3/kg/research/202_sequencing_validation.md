# Validating the KG against raw sequencing data — a skeptical feasibility review

**Status: DRAFT IN PROGRESS (2026-09-19).** Sections are filled in order; anything marked
`[pending]` has not been written yet.

Question asked: every validation this project has is literature-derived, and
`FINDINGS_independence.md` showed the two curated-database numbers (73.0% / 72.5%) are a blend of
~90% reading fidelity and ~55% cross-literature reproducibility. Can we validate instead against
*public microbiome sequencing data*, which is not literature at all?

## 0. Answer up front

**No, not as asked — and yes, in a narrower form that is worth ~10 person-days.**

The idea is right in principle: sequencing data is the only reference available to this project that
is not literature, and it is the only instrument that could say whether the 1,574 `provisional`
edges (78% of the graph) are weak evidence or no evidence — a question
`research/206_edge_confidence.md` sharpened to **0.475 / 0.383 agreement on disjoint sources** and
could not close. Nothing else on the project's list addresses that.

But the validation as normally conceived does not survive scrutiny, for four reasons that compound:

1. **The realised graph is a neuro graph** — 37 of 40 diseases, 60% of edges in five neurological
   diseases (§1.1). Public microbiome repositories are deep in CRC, IBD, T2D and cirrhosis and thin
   in exactly these. The mismatch, not the method, is the binding constraint (§2).
2. **Edges carry direction only** (§1.5), so the comparison is a binary-vs-binary sign test — the
   least efficient test available — on a ceiling of **1,098** testable edges before any external
   coverage loss (§1.3).
3. **It is not powered for the answer we expect.** The honest prior is 55–70% true agreement; 55% is
   **undetectable at any n this graph can supply** (§6.2), and measurement noise attenuates the
   result so hard that a 90%-correct graph reads out at 66% when the DA method's sign-error rate is
   0.3 (§6.3). The pooled percentage is therefore a lower bound at best and uninterpretable at
   worst.
4. **Independence is not free here either.** A public dataset deposited by one of our 271 papers is
   the *same study one layer down* (§5.5) — the identical error `FINDINGS_independence.md` caught in
   Disbiome and Peryton, re-made in a new medium, and easy to ship unnoticed because the join key
   never touches study identity. Microbiome case/control papers routinely deposit their reads, so
   the public neuro cohorts are *disproportionately likely* to be our own papers' cohorts.

**What to do instead.** Three steps, each able to kill the next (§7). **R1: run the provenance
screen — 1–2 person-days, no compute — and stop if fewer than ~8 independent cohorts or ~600
independent case+control samples survive across the top five diseases.** That single output is worth
having on its own: "the public sequencing data for neurological microbiome studies is the same
studies we read" is a real finding about the field. **R2: if it clears, use pre-computed abundance
profiles (curatedMetagenomicData, GMrepo), never raw reads**, and gate on a half-split sign
concordance of ≥0.70 before computing any agreement number. **R3: make the confidence-tier contrast
the primary endpoint, not the pooled agreement rate** — it is the one comparison in this document
that is adequately powered (MDE 16.8 pts at deff 2.27 against an already-measured ~28-point tier
gap), attenuation largely cancels in a difference where it destroys a level, and it validates the
calibration the viewer actually ships.

Total: **~8–13 person-days, under 100 CPU-hours, no GPU, no bulk download.** The remaining work —
uniform SRA reprocessing, per-disease heterogeneity, per-edge verdicts — is 20–40 person-days for
results §6 establishes *in advance* will be uninterpretable, and is recommended against (R5, R6).

**The one thing that must not happen** is the naive design: pull public cohorts, run a differential
abundance test, score "not significant" as disagreement. At the cohort sizes available, DA recall is
~0.18, so that design would report something like 20–30% agreement that is almost entirely the
method's power and would read as a damning finding about the graph (§5.1). Given this project's
history of correctly refusing to interpret underpowered nulls, shipping a spuriously *low* number
that looks like a result would be the worst available outcome.

## 1. What the graph actually contains (and why it decides everything below)

Before surveying anything external, measure the asset. Every number below is computed from the
shipped `kg/graph.json` (2,008 edges, 883 taxa, 40 diseases, 271 contributing papers).

### 1.1 The disease mix is the binding constraint

| disease | edges | | disease | edges |
|---|---:|---|---|---:|
| Parkinson's disease | 326 | | Dementia | 47 |
| Alzheimer's disease | 316 | | Huntington's disease | 47 |
| Multiple sclerosis | 252 | | Anti-NMDAR encephalitis | 39 |
| Stroke | 154 | | CADASIL | 20 |
| Mild cognitive impairment | 154 | | Multiple system atrophy | 20 |
| Epilepsy | 103 | | iNPH | 18 |
| Amyotrophic lateral sclerosis | 92 | | Rett syndrome | 13 |
| Spinal cord injury | 81 | | ADHD | 13 |
| Cognitive impairment | 55 | | *(22 more)* | ≤13 each |
| Autism spectrum disorder | 55 | | | |
| Intracerebral hemorrhage | 50 | | | |

**The top five diseases hold 1,202 of 2,008 edges (60%), and all five are neurological.** 37 of
the 40 disease nodes are neurological or neuro-adjacent; the exceptions are two hepatic
encephalopathy nodes and HBV-associated liver cirrhosis. The repo-root `CLAUDE.md` is right that
the *schema* is broad-scoped, but the *realised* graph is a neuro graph.

This single fact dominates the whole feasibility question, and it cuts the opposite way from
intuition. Public microbiome sequencing repositories are deep in exactly the diseases this graph
does not contain — colorectal cancer, IBD, T2D, obesity, cirrhosis — and thin-to-absent in the
diseases it does. Section 2 quantifies that mismatch; it is the reason the honest answer in
§0 is what it is.

### 1.2 The body-site and assay profile is, fortunately, uniform

- **1,905 of 2,008 edges (94.9%) are gut-only**; 1,957 edges have at least one stool observation.
  Oral (91 edges), gut biopsy (15), nasal (7), blood (6) are the rest.
- Source papers are **162 × 16S, 35 × shotgun, 6 × qPCR, 1 other, 67 unrecorded** (n=271).
  So the graph is roughly **4:1 amplicon-derived**, which matters for §4 (nomenclature) and for
  what a validation cohort must look like.
- Cohorts are small: median 40 cases (IQR 24–71, n=199 papers reporting). **78 of 271 papers are
  from China**, 30 from the USA. Geography is itself a known driver of gut composition, so a
  pooled-public-data comparison is partly a geography comparison unless it is matched.

Uniformity is the good news here: a stool-only, mostly-16S graph can in principle be compared to
stool 16S/shotgun cohorts without a body-site join.

### 1.3 The checkable-edge funnel

An edge is only testable against abundance data if it is (a) **decisive** — the graph asserts a
direction, which contested edges deliberately do not; (b) backed by **stool** evidence; (c)
resolved to an **NCBI taxid**, else there is nothing to join on; and (d) at a **rank an abundance
table actually reports**.

| filter | edges remaining |
|---|---:|
| all edges | 2,008 |
| decisive (drop 220 contested) | 1,788 |
| + has stool evidence | 1,738 |
| + taxon resolved to an NCBI taxid | 1,612 |
| + rank ∈ {genus, species} | **1,098** |

Two observations about this funnel:

- **Taxid resolution is far less lossy at the edge level than the node level.** The headline "76%
  of taxa resolve" (671/883) understates it: unresolved taxa are overwhelmingly low-degree clade
  labels, so **1,873 of 2,008 edges (93.3%) sit on a resolved taxon**. The 212 unresolved taxa
  carry 135 edges between them.
- **Dropping to genus/species costs 514 edges.** Retaining family and above (190 family, 47 order,
  32 class, 74 phylum in the decisive/stool/resolved set) is possible against shotgun profiles,
  where higher ranks can be summed from species; §4 covers why summing is not free.

**1,098 is the optimistic ceiling on testable edges, before any disease- or taxon-coverage loss
from the external side.** It is not the realistic number. The realistic number is set in §2 and
it is one to two orders of magnitude smaller.

### 1.4 The confidence tiers change what a disagreement would mean

| tier | edges | share |
|---|---:|---:|
| well-supported (≥3 agreeing papers) | 76 | 3.8% |
| supported (2 agreeing papers) | 138 | 6.9% |
| provisional (1 paper, or a `discriminating` taxon) | 1,574 | 78.4% |
| contested | 220 | 11.0% |

**78% of the graph is a single paper's claim.** `FINDINGS_independence.md` already measured what
that is worth against disjoint literature: **47.4% / 38.3%** — a coin flip. So a sequencing
validation that returns "our single-paper edges reproduce ~50% of the time" would be *confirming
something already known*, at a very large cost. The scientifically new information lives in the
214 well-supported+supported edges, and §6 will show that this is precisely where the power
problem bites.

### 1.5 What an edge does *not* carry

Direction only. **No effect size, no p-value, no cohort, no baseline abundance, no prevalence.**
This is a deliberate, well-argued design decision (`README.md`: the source papers report
incommensurable statistics), and it is the right one for a KG — but it means the only possible
comparison against sequencing data is a **sign test on direction**, never a correlation of effect
magnitudes. Everything in §5 and §6 follows from that: we are comparing a binary to a binary, the
least statistically efficient comparison available, on a few hundred units.

## 2. The public resources, concretely

`[pending]`

## 3. Disease label alignment

`[pending]`

## 4. Taxonomic nomenclature alignment

`[pending]`

## 5. The statistical design

**This section is the crux.** If there is no defensible test, nothing in §2–§4 matters. The short
answer: **a defensible test exists, but it is not the test anyone would write down first, and the
version most people would write down first is actively misleading.**

### 5.1 The naive design, and exactly how it fails

The obvious design: pull public case/control stool cohorts for our diseases, run a differential
abundance (DA) method, and for each graph edge ask *"is this taxon significantly different, in the
same direction?"* Score agreement.

This is broken in three independent ways.

**(a) Recall at realistic cohort sizes is catastrophic, so "not significant" means nothing.**
The MaAsLin 3 benchmark ([Nickols et al., *Nature Methods* 23:554–564, 2026](https://doi.org/10.1038/s41592-025-02923-9);
[PMC12982127](https://pmc.ncbi.nlm.nih.gov/articles/PMC12982127/)) reports, on simulated data with
known ground truth, **recall of 0.18 at 50 samples**, rising to 0.85 only at 1,000 samples.
ALDEx2 — the method Nearing et al. found most reproducible — **never exceeds recall 0.27 at any
sample size** in that benchmark. Our source cohorts have a **median of 40 cases**, and §2 will show
the public neuro cohorts are no larger. At n≈50–100 a DA method finds roughly one true difference
in five.

So the naive design has no good branch:

- Score "not significant in public data" as **disagreement** → you measure the DA method's power,
  not the graph's correctness, and you will report something like 20–30% agreement that is almost
  entirely an artefact. This is precisely the failure mode this project has been bitten by before
  (24 variables, 24 nulls — `FINDINGS_paper_discordance.md`), except worse, because here it would
  produce a spuriously *low* number that looks like a damning finding.
- **Drop** the non-significant taxa and score only the significant ones → you condition on large
  effects. Large effects are the ones already replicated across many papers, i.e. exactly the
  `well-supported` tier that already agrees with Disbiome at 93.8%. You would rediscover the
  calibration finding and call it independent validation.

**(b) The DA methods do not agree with each other.** [Nearing et al., *Nature Communications*
13:342, 2022](https://doi.org/10.1038/s41467-022-28034-z) ([PMC8763921](https://pmc.ncbi.nlm.nih.gov/articles/PMC8763921/);
note the [Author Correction](https://doi.org/10.1038/s41467-022-28401-w), Nat Commun 13:777)
benchmarked **14 DA methods across 38 datasets / 9,405 samples**. The share of ASVs called
significant ranged from **0.8% to 40.5%** depending on the tool. In unfiltered data only **17.3%**
(SD 22.1) of the significant ASVs were called by more than 12 of the 14 tools; with 10% prevalence
filtering that rises to **38.6%** (SD 15.8) — better, still not agreement. They explicitly advise
avoiding **edgeR** and **LEfSe without p-value correction** (the latter being the method a large
share of our own source papers used), and recommend a **consensus across methods**.

The consequence for us is direct: **the choice of DA method is a researcher degree of freedom
large enough to move the headline agreement number by tens of points.** Any validation run without
pre-registering the method is not a validation.

**(c) Compositionality means the sign itself is frame-dependent.** Relative abundances sum to 1, so
a taxon can appear depleted purely because something else bloomed. ANCOM-BC's bias correction and
MaAsLin 3's *median comparison* both exist to convert a relative-abundance coefficient into an
absolute-scale claim, and MaAsLin 3's authors are explicit that the median-comparison shortcut only
holds when "fewer than half of the community's features are changing in absolute abundance."

There is one genuinely favourable wrinkle here, and it should be stated because it is the strongest
argument *for* feasibility: **our graph inherits the same compositional frame as the validation
data.** 162 of 271 source papers are 16S relative-abundance studies; their "enriched"/"depleted"
claims are relative-abundance claims. So comparing them against a relative-abundance DA coefficient
is a like-for-like comparison. This is not a defence of biological truth — both sides can be
compositionally wrong together — but it does mean compositionality is **not** a reason the
comparison is invalid. It is a reason the comparison cannot be called a test of *absolute* biology.

### 5.2 The design that survives: a sign test on the coefficient, not on significance

Replace "is it significant and in the same direction" with **"what is the sign of the estimated
effect"**, computed for every candidate (taxon, disease) pair regardless of significance.

- **Unit of observation:** one graph edge (taxon, disease) that the graph calls decisive, for which
  at least one qualifying public case/control stool cohort yields an estimable coefficient.
- **Measurement:** the sign of the pooled log-fold-change from a **pre-registered primary DA
  method**, fitted with study as a covariate or in a random-effects meta-analysis across cohorts.
- **Statistic:** the fraction of edges whose public-data sign matches the graph's direction.

This design is strictly better than the naive one because **an underpowered estimate still has an
informative sign**. Shrinkage toward zero adds noise and pushes agreement toward the null, so the
result is *conservative* — it can understate but not overstate agreement. That asymmetry needs
stating in advance, in the same spirit as the recall-audit rule already in the root `CLAUDE.md`
("this instrument can confirm a miss but cannot refute one").

**Primary method recommendation: ANCOM-BC2 as primary, with ALDEx2 and MaAsLin 3 as pre-registered
sensitivity analyses, and the Nearing consensus rule (sign agreed by ≥2 of 3) as the reported
robustness check.** Rationale: ANCOM-BC2 provides an explicit sampling-fraction bias correction
(the right frame for a relative-abundance comparison), ALDEx2 is the tool Nearing et al. found most
cross-study-consistent (1.35-fold over random expectation vs edgeR's 1.10), and MaAsLin 3 adds the
prevalence/abundance split that matters when a taxon differs in *presence* rather than *amount* —
in IBDMDB, **77% of MaAsLin 3's associations were prevalence-based**, and a pseudo-count model
would have missed or mis-signed them. Prevalence-vs-abundance is a real hazard here: our extractor
reads sentences like "*Akkermansia* was reduced in patients", which conflates the two.

**Apply 10% prevalence filtering before testing**, per Nearing et al. Pre-register it; do not tune it.

### 5.3 The null is not 50%, and it must be established empirically

Two separate reasons the naive 50% null is wrong:

1. **Marginal imbalance.** Our decisive edges are **1,038 enriched / 750 depleted (58.1% / 41.9%)**.
   If public-data signs carried the same marginal and were otherwise independent of our graph,
   expected agreement would be **51.3%**, not 50%. Small, but it is the difference between a
   correct null and a free 1.3 points. (On the genus/species stool-resolved subset the figure is
   the same, 51.28%.)
2. **Shared compositional and technical structure.** Both sides analyse relative abundances of the
   same community, so a community-wide shift (e.g. Bacillota/Bacteroidota ratio) induces correlated
   signs across many taxa *for reasons unrelated to our extraction being correct*.

**The fix is a label permutation null, not an analytic one.** Permute case/control labels *within
each public cohort*, re-run the primary DA method, recompute agreement, repeat. This yields the
distribution of agreement under "the graph's directions are unrelated to this data" while
preserving compositional structure, taxon correlation, and cohort composition. The observed
agreement is then compared to that distribution. This is computationally the expensive part
(permutations × cohorts × DA fits) and is the main compute-time driver in §7.

This is the same guard the project already uses — taxon-block permutation plus taxon cluster
bootstrap in `check_independence.py` / `calibrate_agreement.py` — and it should be reused
verbatim rather than reinvented.

### 5.4 Multiple testing, and what the unit actually is

**There are not 1,098 tests. There should be one primary test.**

- **Primary endpoint:** a single pooled agreement rate against the permutation null. One p-value.
- **Secondary, pre-specified:** stratification by (i) confidence tier and (ii) the five diseases
  with ≥50 testable edges. That is at most ~8 tests; BH across that family at q=0.05.
- **Nothing else.** The temptation here is per-edge testing — "which of our edges are contradicted
  by the data" — and it is a trap: 1,098 tests at n≈100 samples each will produce ~0 discoveries
  after correction, exactly as happened with the disease-hierarchy pairs (**0 of 16 survived BH**,
  `FINDINGS_disease_ontology.md`). Per-edge verdicts can be *listed as review targets*, as the
  11 doubly-contradicted pairs already are, but they must not be counted as findings.

**Clustering must be two-way.** Edges sharing a taxon are correlated (one taxon, many diseases);
edges sharing a public cohort are correlated (one cohort drives many taxa). Use a cluster bootstrap
over taxa **and** a permutation blocked by cohort, and report whichever is wider. A naive binomial
CI on 1,098 edges would be roughly ±3 points and would be wrong by a factor of two or more.

### 5.5 The independence screen is mandatory, and it may end the project

This is the one that matters most, and it is the cheapest to run.

`FINDINGS_independence.md` established that Disbiome and Peryton are not independent of our corpus
because they read our papers. **A public sequencing dataset deposited by one of our 271 papers is
not independent either** — it is the *same study*, one layer down. Validating a paper's claim
against that paper's own raw reads measures reanalysis stability, not replication. It would be the
identical error, rediscovered in a new medium, and it would be very easy to ship without noticing
because the join key (taxon, disease) never touches the study identity.

**So the design must include a provenance screen on the public side**, excluding any cohort whose
PMID, DOI, BioProject/SRA accession, or title matches one of our 271 contributing papers — matching
on *any* key, per the established rule that under-matching is the dangerous direction.

And here is the sting: **microbiome case/control papers routinely deposit their reads, so the
public neuro cohorts are disproportionately likely to be exactly our papers' cohorts.** If, after
the screen, the surviving independent sample count is small, the project is dead — and that is
knowable in roughly a day of metadata work, before any data is downloaded or any DA method is run.

> **Falsifiable first step (do this before anything else):** assemble the set of public stool
> case/control cohorts for Parkinson's, Alzheimer's, MS, MCI and Stroke, resolve each to a PMID, and
> intersect with `graph.json`'s 271 paper titles via the existing `paper_keys()` matcher in
> `validate_external.py`. **Kill criterion: if fewer than ~8 independent cohorts or ~600 independent
> case+control samples survive across the top five diseases, stop.** §6 shows why those are the
> thresholds.

### 5.6 Remaining confounders that cannot be designed away

- **Geography.** 78 of our 271 papers are Chinese cohorts; public repositories skew North American
  and European. Gut composition differs by population strongly enough that a disjoint-geography
  comparison partly measures geography. Mitigation: include country as a covariate and report a
  China-only sensitivity analysis — but note the project already found `country=China` splitting
  45/44 on edge-level discordance, i.e. no detectable effect at that MDE.
- **Study effect / batch.** Pooling cohorts requires either a random-effects meta-analysis
  (preferred: keeps study as the unit and is the honest structure) or explicit batch correction —
  [MMUPHin](https://doi.org/10.1186/s13059-022-02753-4) (Ma et al., *Genome Biology* 23:208, 2022)
  and [ConQuR](https://doi.org/10.1101/2021.09.23.461592) (Ling et al.) are the standard options.
  **Prefer meta-analysis over batch correction**: batch correction on confounded designs (where
  study and disease are nearly collinear — which is exactly our situation, since most diseases here
  have one or two cohorts) can remove the signal along with the batch.
- **16S vs shotgun frame mismatch.** Our graph is 4:1 amplicon-derived; the best-curated public
  resource (curatedMetagenomicData) is shotgun-only. §4 covers what this breaks.
- **Medication, diet, age matching.** Uncontrolled and mostly unrecorded on both sides.

### 5.7 Verdict for this section

**A defensible test exists.** It is: *pooled sign agreement between graph direction and a
pre-registered DA coefficient, over provenance-screened independent public stool cohorts, against a
within-cohort label-permutation null, with two-way clustering, one primary endpoint and ≤8
pre-specified secondaries.*

**It is defensible only under conditions that §2 and §6 must be checked against**, and the
conditions are demanding: enough independent cohorts, enough samples per cohort, and enough taxon
overlap. The naive version of this experiment is not merely weaker — it produces a number that
looks like a finding and is not one, which given this project's history would be the worst possible
outcome.

## 6. Power

**Stated before proposing the experiment, per the standing project rule.** This project has
repeatedly reported underpowered comparisons as nulls (24 variables / 24 nulls; 0 of 16
disease-pairs surviving BH; the MS counter-example at MDE 28.7). The numbers below are computed
against the §5.2 design and the §5.3 null of **51.3%**, at 80% power, α=0.05 two-sided.

### 6.1 Design effect first

Edges are not independent. In the 1,098-edge testable subset there are **484 distinct taxa**, mean
**2.27 edges per taxon** (largest taxon cluster: 19 edges). Taking the naive
`deff = 1 + (m̄−1)·ICC` with ICC≈1 as the pessimistic bound gives **deff ≈ 2.27** from taxon
clustering alone. Cohort clustering is *additional* and potentially far worse (§6.4). Three columns
are therefore reported throughout: deff = 1 (wrong, shown only to expose what a naive binomial CI
would claim), 2.27 (taxon clustering), and 5 (taxon + modest cohort clustering).

### 6.2 Primary endpoint MDE

Minimum detectable agreement **above the 51.3% null**, in percentage points:

| testable edges | deff = 1 | deff = 2.27 | deff = 5 |
|---:|---:|---:|---:|
| 50 | 19.3 | 28.1 | 38.9 |
| 100 | 13.8 | 20.5 | 29.3 |
| 200 | 9.8 | 14.7 | 21.4 |
| 400 | 7.0 | 10.5 | 15.4 |
| 600 | 5.7 | 8.6 | 12.6 |
| **1,098** (ceiling from §1.3) | 4.2 | **6.3** | 9.4 |

Read this against the effect we should actually expect. `FINDINGS_independence.md` measured
cross-literature reproducibility at **~53–59%** against disjoint sources, and sequencing data is a
*harsher* instrument than a second curator, not a gentler one. **So the honest prior is that true
agreement lands somewhere in 55–70%.**

- Detecting 60% vs the 51.3% null (8.7 pts) needs **≈600 edges at deff 2.27**, or ≈1,100 at deff 5.
- Detecting 65% (13.7 pts) needs **≈230 edges at deff 2.27**.
- Detecting 55% (3.7 pts) is **out of reach at any n available here** — it would need ~3,200 edges
  at deff 2.27, more edges than the graph has.

**That last line is the single most important number in this document.** If the truth is that our
edges reproduce in sequencing data at 55%, this experiment cannot distinguish that from chance. It
would return "not significant" and someone would read it as "the graph is noise." A pre-registered
statement that 55% is undetectable is the only thing preventing that misreading.

### 6.3 Attenuation makes it worse, and by how much is computable

The DA estimate has its own sign-error rate *q* — the probability it assigns the wrong sign to a
taxon that genuinely differs — driven by finite reads, small cohorts, and the compositional frame.
Observed agreement is then `p(1−q) + (1−p)q`:

| true agreement *p* | q = 0.05 | q = 0.10 | q = 0.20 | q = 0.30 |
|---:|---:|---:|---:|---:|
| 0.90 | 0.860 | 0.820 | 0.740 | 0.660 |
| 0.75 | 0.725 | 0.700 | 0.650 | 0.600 |
| 0.70 | 0.680 | 0.660 | 0.620 | 0.580 |
| 0.60 | 0.590 | 0.580 | 0.560 | 0.540 |

At the cohort sizes available (§2), *q* in the 0.2–0.3 range is realistic — MaAsLin 3's recall of
0.18 at n=50 means most true effects are barely estimated at all, and a barely-estimated effect has
a near-coin-flip sign. **At q = 0.3, a graph that is 75% correct reads out as 60%, and a graph that
is 90% correct reads out as 66%.** The experiment cannot distinguish "our graph is 75% right" from
"our graph is 90% right and the data is noisy."

**Consequence: the result is a lower bound on agreement, never a point estimate, and it must be
reported that way** — the same asymmetry the recall audit already carries ("can confirm a miss but
cannot refute one"). *q* is estimable, not merely assertable: split each public cohort in half,
run the DA method on both halves, and measure how often the two halves agree on sign. That
half-split concordance is a direct empirical handle on *q* and should be a **required deliverable
before the headline number is computed**. If half-split sign concordance is below ~0.7, the
instrument is too noisy and the run should be abandoned.

### 6.4 The failure mode that actually kills it: the cohort is the unit

If a disease has one public cohort, then every edge tested for that disease is scored against one
study's idiosyncrasies. `FINDINGS_paper_discordance.md` established that **disagreement with the
literature majority is a property of the paper** — 27.6% of observations are in the minority, and
which papers hold the minority is clustered far beyond chance (p = 0.0003). A validation cohort is
just another paper. With one cohort per disease, "our graph disagrees with the data" and "this
cohort is one of the ~28% minority-holders" are **not separable**.

If clustering is severe enough that the cohort is effectively the unit of replication:

| independent cohorts | MDE (pts above null) |
|---:|---:|
| 3 | 48.6 |
| 5 | 47.6 |
| 8 | 42.0 |
| 12 | 36.3 |
| 20 | 29.3 |
| 40 | 21.4 |

**This is the regime where the experiment is worthless**, and it is the likely regime for every
disease in this graph except possibly Parkinson's. Hence the §5.5 kill criterion of ~8 independent
cohorts as an absolute floor — and 8 is a floor, not a target; 20+ is where per-disease claims
become possible.

### 6.5 What *is* well-powered — the one genuinely good news item

The **confidence-tier contrast is powered.** Comparing the 144 non-provisional testable edges
against the 954 provisional ones, at a 70% baseline:

| deff | MDE |
|---:|---:|
| 1.0 | 11.3 pts |
| 2.27 | **16.8 pts** |
| 5.0 | 24.4 pts |

The tier gap already measured against Disbiome is **93.8% vs 66.1% ≈ 28 points**. That is larger
than the MDE at deff 2.27 and comparable to it even at deff 5. **So if the sequencing data
reproduces the tier ordering, that is a real, adequately-powered result** — and it is arguably a
more valuable one than the headline agreement number, because it validates the *calibration* the
viewer ships (the thing a reader actually uses when looking at one edge) rather than a single
pooled percentage that §6.3 shows is uninterpretable anyway.

### 6.6 What is not powered, stated in advance

| secondary analysis | n edges | MDE @ deff 2.27 |
|---|---:|---:|
| Parkinson's disease | 149 | 16.9 pts |
| Multiple sclerosis | 152 | 16.8 pts |
| Alzheimer's disease | 131 | 18.0 pts |
| Mild cognitive impairment | 110 | 19.6 pts |
| Stroke | 63 | 25.4 pts |

Every per-disease test sits at an MDE of 17–25 points — the **same regime as the MS null that
`FINDINGS_independence.md` correctly refused to interpret** (MDE 28.7). Per-disease results from
this experiment will be nulls, and must be pre-registered as uninterpretable rather than written
up as heterogeneity.

Likewise, a 95% CI half-width on the primary number is **±4.1 points at deff 2.27** on the full
1,098 edges, and **±13.5 points** if only 100 edges survive the §2/§5.5 screens. A result of
"62% [48, 76]" is not a result.

### 6.7 Power summary

| requirement | threshold | source |
|---|---|---|
| minimum testable edges for a meaningful primary | **≥600** | detects 60% vs 51.3% at deff 2.27 |
| minimum independent cohorts | **≥8** absolute floor, ≥20 for per-disease | §6.4 |
| minimum independent case+control samples | **≈600** across the top five diseases | ~75/cohort × 8 |
| required instrument check | half-split sign concordance **≥0.7** | §6.3 |
| pre-declared undetectable | true agreement ≤55% | §6.2 |

## 7. Recommendation (ranked, costed in person-time and compute-time)

**The headline recommendation is a conditional stop.** §6.2 shows the primary endpoint most people
would want — "what fraction of our edges reproduce in sequencing data" — needs **≥600 testable
edges** to detect 60% against a 51.3% null at deff 2.27, and §6.3 shows that even if it clears that
bar the number is **attenuated into uninterpretability** (a 90%-correct graph reads out at 66% when
*q*=0.3). So the full validation as normally conceived is **not defensible**, and should not be
scheduled. What *is* defensible is a three-step sequence where each step can kill the next, and
where the genuinely powered result (§6.5) is reachable in **step 2 of 3**.

Effort below is in **person-days** (one person, focused) and **compute-hours** (ordinary CPU cores
unless stated; nothing here needs a GPU). No step requires downloading raw reads until R5, which is
the step recommended *against*.

---

### R1 — Run the provenance screen. Do this first, do nothing else until it returns.

**Effort: 1–2 person-days. Compute: negligible (metadata API calls, minutes).**

§5.5's screen is both the cheapest thing in this document and the one most likely to end the
project. Assemble every public stool case/control cohort for Parkinson's, Alzheimer's, MS, MCI and
Stroke from the §2 resources; resolve each to a PMID/DOI/BioProject; intersect against
`graph.json`'s 271 contributing papers using the existing `paper_keys()` matcher in
`validate_external.py`, matching on **any** key.

- **Falsifiable test:** after the screen, count surviving independent cohorts and total
  independent case+control samples across the top five diseases.
- **Kill criterion:** **fewer than ~8 independent cohorts, or fewer than ~600 independent
  case+control samples → stop, and write up R1 itself as the finding.** Per §6.4, at ≤8 cohorts the
  MDE is ≥42 points when the cohort is the unit of replication; there is no analysis that rescues
  that.
- **Why this is worth doing even if it kills everything:** "the public sequencing data for
  neurological microbiome studies is *the same studies we read*" is a publishable, citable negative
  result about the field, and it is the direct generalisation of `FINDINGS_independence.md` from
  curated databases to primary data. It costs two days.

A second, equally cheap output falls out of R1 at no extra cost and should be a required
deliverable: the **per-disease independent-cohort table**. §6.6 pre-registers every per-disease test
as uninterpretable; R1 tells you in advance whether even Parkinson's — the only plausible candidate
— clears 20 cohorts.

---

### R2 — If R1 clears: use pre-computed abundance profiles, not raw reads.

**Effort: 5–8 person-days. Compute: 20–60 CPU-hours (DA fits + permutations), no read processing.**

The single largest cost saving available, and it costs almost nothing in validity. `curatedMetagenomicData`
and GMrepo both ship **already-profiled relative-abundance tables with curated case/control labels**
(§2). Fitting a DA model to those is hours of compute. Re-processing SRA reads through a uniform
pipeline is weeks of compute and introduces a pipeline choice that is itself a researcher degree of
freedom — and §5.1(b) already establishes that method choice moves the answer by tens of points, so
adding a second such lever is strictly bad.

The cost is real and must be stated: profiled tables fix the taxonomic frame to whatever profiler
the resource used (§4), and the 16S/shotgun split is not negotiable at this layer. Accept it.

- **Falsifiable test — the instrument check from §6.3, run before any agreement number is
  computed:** split each surviving cohort in half at random, fit the primary DA method to both
  halves, measure sign concordance between halves on the taxa in our graph.
- **Kill criterion:** **half-split sign concordance < 0.70 → abandon.** The instrument is then too
  noisy to distinguish the hypotheses in §6.3's attenuation table, and any number produced
  downstream is a measurement of the DA method.
- Pre-register the method stack from §5.2 before looking at any agreement: **ANCOM-BC2 primary;
  ALDEx2 and MaAsLin 3 as sensitivity; ≥2-of-3 sign consensus as the robustness check; 10%
  prevalence filter, not tuned.**

---

### R3 — Make the **confidence-tier contrast** the primary endpoint, not the pooled agreement rate.

**Effort: +2–3 person-days on top of R2. Compute: +10–30 CPU-hours (the permutation null dominates).**

This is the recommendation that actually changes what gets done, and it follows directly from §6.5
being the only adequately-powered comparison in the document. Ask **"does sequencing data reproduce
the ordering of our confidence tiers?"** rather than **"what fraction of our edges are right?"**

Three reasons it is the better endpoint:

1. **It is powered.** MDE 16.8 points at deff 2.27 against an already-measured 28-point tier gap
   (93.8% vs 66.1% on Disbiome). The pooled rate is not powered for anything in the honest 55–70%
   prior (§6.2) and is undetectable at 55%.
2. **Attenuation largely cancels.** *q* degrades both tiers, so a *difference* between tiers
   survives noise that destroys the *level*. §6.3's table is a statement about levels.
3. **It validates the thing a reader uses.** The viewer ships a per-edge tier; the pooled percentage
   is not on screen anywhere. And it speaks directly to `research/206_edge_confidence.md`, which
   found the shipped `provisional` tier — **1,574 of 2,008 edges** — agrees at only **0.475
   [0.354, 0.598] / 0.383 [0.256, 0.528]** once recomputed on disjoint sources. An independent
   non-literature reference is the only instrument that can say whether that tier is weak evidence
   or no evidence, and it is the strongest scientific argument for doing any of this.

- **Falsifiable test:** sign-agreement of the 144 non-provisional testable edges minus that of the
  954 provisional ones, against the §5.3 within-cohort label-permutation null, with the two-way
  clustering of §5.4 (taxon cluster bootstrap + cohort-blocked permutation, report the wider).
- **What would kill it:** a tier gap whose CI includes zero at an MDE of ~17 points means the tiers
  are **not** distinguishable by this instrument. That is a real negative result about the shipped
  calibration and must be reported as one — not reframed as "trending in the right direction."
- **Report the pooled agreement rate as a secondary, as a lower bound, with the §6.2 MDE table
  adjacent to it, and with the explicit pre-registered sentence that true agreement ≤55% is
  undetectable here.** The failure mode this guards against is someone reading a non-significant
  pooled number as "the graph is noise."

---

### R4 — Ship the per-edge disagreements as a review list, never as findings.

**Effort: 0.5 person-days (it is a join and a sort, once R2/R3 exist). Compute: none.**

§5.4 is unambiguous that 1,098 per-edge tests produce ~0 discoveries after correction — the
disease-hierarchy pairs already demonstrated this (**0 of 16 survived BH**). But the *list* has
curatorial value in the same way the 11 doubly-contradicted Disbiome+Peryton pairs do.

- **Falsifiable test:** none. This is deliberately not a test, and the document must say so where
  the list is published.
- **What would kill it:** any attempt to attach a p-value or a count to it.

---

### R5 — Do **not** reprocess raw reads from SRA/ENA.

**Effort avoided: 20–40 person-days. Compute avoided: thousands of CPU-hours plus terabytes of
storage.**

Listed explicitly because it is the default instinct and it is wrong here. It buys uniform
processing, which is worth something; it costs an order of magnitude more effort than R2, adds a
pipeline degree of freedom, and — decisively — **cannot fix the binding constraint, which is cohort
count, not cohort quality.** §6.4's table does not improve because the reads were processed better.
If R1 returns 5 independent cohorts, uniform reprocessing of those 5 yields an MDE of 47.6 points.

Revisit only if R1 returns a surprise: ≥20 independent cohorts for a single disease where the
public profiles are 16S and our graph's edges for that disease are dense.

---

### R6 — Do **not** run the naive significance-matching design at any scale.

**Effort avoided: irrelevant — the objection is not cost.**

§5.1(a) shows both branches are broken: scoring "not significant" as disagreement measures
MaAsLin 3's recall of 0.18 at n=50 and would report something near 20–30% that looks like a damning
finding about the graph; dropping non-significant taxa conditions on large effects and rediscovers
the tier calibration while calling it independent validation. This design should be named and
rejected in writing anywhere this work is described, because it is what a reviewer will ask for.

---

### The cheaper 80% version, in one paragraph

**R1 + R2 + R3 = roughly 8–13 person-days and under 100 CPU-hours**, no GPU, no bulk download, no
raw-read pipeline — and it delivers the only adequately-powered result available (the tier
contrast), a pre-registered and correctly-caveated lower bound on pooled agreement, and a hard
answer to the `206_edge_confidence.md` question about the 1,574 provisional edges. The remaining
20% — uniform reprocessing, per-disease heterogeneity, per-edge verdicts — is 20–40 person-days for
results §6 shows in advance will be uninterpretable. **R1 alone is 1–2 days and is the correct next
action regardless of whether anything after it is ever run.**
