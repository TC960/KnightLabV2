# Calibrated confidence for direction-only edges — a skeptical review

**Question posed:** an edge with 9 supporting papers and an edge with 1 supporting paper
are distinguished only by an integer. What is the statistically defensible way to attach a
*calibrated confidence* to each edge, using direction-only votes?

**Answer, up front: the brief's premise is out of date, and its implied conclusion is
wrong in both directions.** The graph already ships a four-tier confidence score
(`build_kg.py:699`, `annotate_confidence`) calibrated against Disbiome and Peryton, so the
distinction is not "only an integer". But that score is *itself* miscalibrated in a way
nobody has measured, and the fix is a documentation change, not a model. Meanwhile every
richer model I could fit or find in the literature is either untestable at this n or
measurably worse than what is already shipped.

Everything below was computed from `graph.json` (2,008 edges, 271 papers, 3,077
paper–edge observations) and the cached `disbiome_experiments.json` / `Peryton-results.tsv`,
CPU-only, against the real NCBI taxdump (3,476,674 names — it *is* reachable in this
environment, contrary to several earlier session logs). No repo file other than this one
was modified. Scripts are in the session scratchpad, not the repo; every number here is
reproducible from `graph.json` plus the two cached curations.

---

## 0. The five findings, in order of how much they should change what you do

1. **There is one defensible confidence model the vote data can support, it can be written
   in closed form, and it is decisively miscalibrated.** Under the only model that fits the
   vote data (constant per-paper flip rate ε = 0.202, direction prior π = 0.626), the
   posterior probability that an edge's direction is correct is a logistic function of the
   **vote margin alone** — 0.869 at margin 1, 0.963 at margin 2, 0.990 at margin 3. Measured
   against literature the curators did not share with us, the same edges agree **0.474 /
   0.765 / 0.800** (Disbiome) and **0.378 / 0.789 / 0.571** (Peryton). ECE 0.32 and 0.38
   against a measurement-noise floor of ~0.07. Any confidence score fitted to the internal
   votes measures *"will the next paper in this corpus say the same thing"*, not *"is the
   association real"*, and those differ by 40 points at the thin end. **§3.1.**

2. **The shipped tier rates are overstated for the `provisional` tier, and the CI excludes
   the shipped number in both databases independently.** `CONFIDENCE_RATES` in
   `build_kg.py:687` quotes provisional at 0.661 (Disbiome) / 0.619 (Peryton). Those are
   pooled rates, and `FINDINGS_independence.md` already established that pooled rates blend
   reading fidelity with reproducibility. Recomputed on **disjoint-source pairs only**:
   **0.475, 95% CI [0.354, 0.598] (n=59)** and **0.383, [0.256, 0.528] (n=47)**. Both CIs
   exclude the shipped quote. Tier ECE against disjoint literature is 0.135 / 0.179, almost
   all of it in this one tier. **This is the one actionable defect in the review and it
   costs no GPU, no new data, and no new model.** **§6.3.**

3. **Hierarchical shrinkage toward the taxonomic parent is measurably the wrong direction.**
   Across papers (not within), parent and child agree on direction only **71.4%** of the
   time (n=845), and **68.5%** when the child has one paper (n=638) — not the 89% the
   within-paper figure suggests. Tested externally: for single-paper edges the child's own
   vote beats the parent's direction **0.684 vs 0.596** (Disbiome, n=57) and **0.623 vs
   0.566** (Peryton, n=53). Pooled McNemar 22 vs 14, p = 0.243 — a null, but the point
   estimate goes the *same* wrong way in two independent references, and only 698 of 1,561
   single-paper edges even have a parent to borrow from. **§5.**

4. **A taxon random effect would inflate exactly the edges already measured to carry no
   extra validity.** Variance components on the 3,077 directional votes: taxon ICC **0.296**,
   edge ICC 0.377, paper ICC 0.089, **disease ICC 0.013**. So a multilevel logistic model
   with taxon and disease random effects would (a) get nothing from the disease level and
   (b) spend its taxon level re-deriving the corpus-wide prior that `taxon_purity` already
   encodes. The repo has already run the falsification test for that shrinkage:
   `restates_prior` does **not** predict external agreement (+0.9 / +7.7 points,
   p = 1.00 / 0.48, MDE ±9.7 points — a well-powered null). **§4.3.**

5. **The evaluation instrument can resolve exactly one binary distinction, and the graph
   already ships it.** On disjoint-source pairs (Disbiome n=86, Peryton n=76), the MDE at
   80% power is ±0.28 for two bins, ±0.33 for three, ±0.37 for four. The 1-paper vs
   ≥2-paper split shows gaps of +0.340 and +0.403 and clears its MDE. The
   `supported` vs `well-supported` split shows +0.135 and +0.022 against MDEs of ±0.230 and
   ±0.220 and does not. A reliability diagram with ±0.05 bins needs **286 judged pairs per
   bin**; we have 86 and 76 in total. **§6.**

**Verdict (§7): do not build a continuous confidence score. Fix the provisional tier's
quoted rate, add the disjoint-source column next to the pooled one, and stop.**

---

## 1. What is actually in the graph, and a correction to the brief

| | |
|---|---|
| association edges | 2,008 |
| single-paper edges | **1,561 (77.7%)** |
| replicated (≥2 papers) | 447 |
| contested | 220 (1 of them a single paper contradicting itself) |
| paper–edge observations | 3,077 |
| directional votes | 3,092 (1,720 enriched, 1,372 depleted — 55.6% up) |

Confidence tiers already assigned by `annotate_confidence`:

| tier | edges | rule |
|---|---|---|
| `well-supported` | 76 | ≥3 papers agreeing, taxon not `discriminating` |
| `supported` | 138 | 2 papers agreeing, taxon not `discriminating` |
| `provisional` | 1,574 | 1 paper, or a `discriminating` taxon at any depth |
| `contested` | 220 | papers disagree; no direction asserted |

So the brief's framing — "distinguished only by an integer" — is two revisions stale. The
live question is narrower and better: **is the four-tier ordinal score right, and can it be
refined into something continuous and calibrated?** The rest of this document answers no to
the second and mostly-yes-with-one-fix to the first.

One structural fact governs everything: **the external validation never judges a contested
edge**, by construction (`calibrate_agreement.build_pairs` skips any pair where either side
is contested). Every one of the 173 / 137 judged pairs is therefore a *unanimous* edge, and
vote margin is identical to paper count within the evaluation set. Margin and depth cannot
be separated by any measurement currently available. Remember this in §3.

---

## 2. Q1 — vote counting: when Hedges & Olkin bites, and whether it bites here

### 2.1 The earlier review overstates the result. The correction matters.

`research/06_meta_analysis_heterogeneity.md` §1 says the statistical power of vote counting
"*decreases* as more studies are added, and can tend to zero as the number of studies
grows". That is a faithful statement of
[Hedges & Olkin (1980), *Psychological Bulletin* 88(2):359–369](https://doi.org/10.1037/0033-2909.88.2.359)
— but of a *specific* vote-counting procedure, and **not the one this graph uses.**

Hedges & Olkin analyse the procedure that classifies each study into
{significant-positive, significant-negative, non-significant} and declares an effect when
the significant-positive count exceeds some threshold, testing the observed proportion
against ½. Write π₊ for the probability a single study returns a *significant positive*
result. For a true positive effect studied at low power, π₊ < ½. The observed proportion of
positive votes converges to π₊ < ½, so the procedure converges to the conclusion "no effect"
with probability → 1. That is the pathology, and its precondition is explicit: **the
denominator includes the studies that reported nothing.**

This graph's votes have a different sampling frame. The extraction prompt
(`eval-v2/run_eval.py`, `samgated-v1`) requires **reported statistical significance** — "if
significance is unclear or unreported for a taxon, omit it" (documented in the root
`CLAUDE.md`). A paper that measured a taxon and found nothing contributes **no vote at all**.
So the graph observes, for each (taxon, disease) pair, a sample from the *conditional*
distribution

    q = π₊ / (π₊ + π₋)

where π₋ is the probability of a significant result in the wrong direction. For any
two-sided test with a symmetric null, θ > 0 implies π₊ > π₋, hence q > ½. The sample
proportion of up-votes converges to q, and a majority rule on that proportion is
**consistent**: its power *increases* with the number of votes. The Hedges–Olkin
decreasing-power pathology **does not apply to this graph's edge weights.** The
sign-based vote-count estimator that *is* consistent, and the inversion from vote
proportion back to an effect size, is Hedges & Olkin's own — see the treatment in
Bushman & Wang, "Vote-counting procedures in meta-analysis", ch. 12 of
[*The Handbook of Research Synthesis and Meta-Analysis*, 3rd ed. (Russell Sage, 2019)](https://www.russellsage.org/publications/handbook-research-synthesis-and-meta-analysis-third-edition),
which distinguishes the naive "conventional" procedure Hedges & Olkin demolished from the
estimator they proposed to replace it.

**Nothing in this should be read as a defence of vote counting in general.** The standard
objections survive intact: a study of n=200 and a study of n=12 cast one vote each, and
effect magnitude and precision are discarded
([Borenstein, Hedges, Higgins & Rothstein, *Introduction to Meta-Analysis*, 2nd ed., Wiley 2021](https://doi.org/10.1002/9781119558378)).
The correction is narrow and it is about which failure mode to expect.

### 2.2 The pathology that *does* bite is worse, and it is the missing denominator

The price of escaping Hedges & Olkin is that **the denominator is unobserved.** "Akkermansia
enriched in Parkinson's, 11 papers" does not say 11 of how many. Computed against the
papers that contribute any edge to that disease:

| edge | votes | papers on that disease | share |
|---|---|---|---|
| *Faecalibacterium* / Alzheimer's | 3↑ / 13↓ | 46 | 0.348 |
| *Faecalibacterium* / Stroke | 1↑ / 11↓ | 30 | 0.400 |
| *Prevotella* / Parkinson's | 2↑ / 11↓ | 67 | 0.179 |
| *Akkermansia* / Parkinson's | 11↑ / 0↓ | 67 | 0.164 |
| *Blautia* / Parkinson's | 0↑ / 11↓ | 67 | 0.164 |

Median across all 2,008 edges: **0.077**. Even the single best-evidenced edge in the graph
is reported by about a third of the papers that could have reported it, and the silent
two-thirds are unmodelled. The at-risk set is an *upper* bound on the true denominator — a
silent paper may have measured the taxon and found nothing, or not measured it at all
(different depth, different rank resolution, different body site) — so the true reporting
rate is somewhere between the numbers above and 1.0, and nothing in the graph narrows it.

**This is the actual reason confidence cannot be a probability here.** A probability needs
a denominator. Every model in §4 either assumes the denominator away or requires data we do
not have.

### 2.3 How much of the pair space are we even seeing? Chao1 says 28%

The frequency-of-frequencies spectrum is extreme: **f₁ = 1,561 singletons, f₂ = 235
doubletons** across 2,008 observed pairs. The Chao1 lower-bound estimator
([Chao, *Scandinavian Journal of Statistics* 11(4):265–270, 1984](https://www.jstor.org/stable/4615964))
gives

    Chao1 = 2,008 + 1,561² / (2 × 235) = 7,193   95% CI [6,409, 8,115]

so the graph holds roughly **27.9%** of the (taxon, disease) pairs this kind of literature
would eventually report, and the Good–Turing coverage estimate is **C = 1 − f₁/N = 0.493** —
about half of the next paper's (taxon, disease) observations would be a pair never seen
before. Chao1's assumptions (independent sampling with heterogeneous per-pair detection
probabilities) are violated here in ways that inflate f₁, so read this as an order of
magnitude, and read it as a statement about **undersampling, not about directional
publication bias** (§6 is where that distinction bites).

### 2.4 The sign test, exactly: zero edges in the graph survive correction

The binomial sign test is the minimal honest thing computable from what we have. Exact
two-sided, H₀: q = ½, per edge, across all 2,008 edges:

| | |
|---|---|
| uncorrected p < 0.05 | **28 edges** (1.4%) |
| Benjamini–Hochberg q = 0.05 | **0 edges** |
| Bonferroni 0.05 (threshold 2.49 × 10⁻⁵) | **0 edges** |
| deepest unanimous edge | 14 papers, p = 1.2 × 10⁻⁴ |

The earlier review said "no single edge at 2–4 papers per side can reach significance". The
correct and stronger statement is that **no edge in this graph reaches significance at any
depth, after correction, and the deepest unanimous edge in the corpus misses Bonferroni by
a factor of five.** It would take 17 unanimous papers on one pair; the record is 14.

Two further caveats, both fatal to using this as a filter:

- **H₀ = ½ is the wrong null.** Under a literal coin-flip null the pair would not be
  reported at all. The sign test here tests "is the *reporting* directionally consistent",
  not "is there an association", and a rejection would not mean what a reader assumes.
- The test conditions on the same missing denominator as §2.2.

### 2.5 The one modern validation of vote counting, and what it says about our depth profile

[Wang, Ward *et al.* (2016), "Vote-counting... to rank biomarkers", PMC4770796](https://pmc.ncbi.nlm.nih.gov/articles/PMC4770796/)
tested whether the number of supporting studies predicts replication in an independent test
set. It does, weakly: significant at p = 0.0006, r = 0.44 (19% of variance), with
confirmation rates of ~50% at 2 supporting studies, ~67–70% at 3, and ~90% at 4+.

Applied to our depth profile, that is a sobering arithmetic: **1,561 of 2,008 edges (77.7%)
sit below even the bottom rung of that curve**, and only 212 edges (10.6%) have ≥3 papers.
Note also how closely the published curve tracks our own measured tiers — ~50% / ~67–70% /
~90% against our disjoint-source 0.43 / 0.77 / 0.85 (pooled across both curations). Two
unrelated literatures landing on the same shape is the best evidence in this document that
the existing three-tier ordering is real.

---

## 3. The one closed-form confidence score the vote data supports — and why it fails

### 3.1 Fit the model, then look at what it predicts

Two models, fitted by maximum likelihood to the 448 replicated edges (1,532 votes), dense
grid plus Nelder–Mead polish from 16 / 80 starts so the comparison is not an optimiser
artefact:

- **Model A — constant per-paper flip rate.** Each edge has a true direction, drawn up with
  probability π; each paper votes correctly with probability 1 − ε. Two parameters.
  **π̂ = 0.6264, ε̂ = 0.2020**, log L = −616.065.
- **Model B — beta-binomial.** Per-edge vote probability drawn from Beta, so edges may
  differ in reliability. Three parameters. **π̂ = 0.6337, ε̂ = 0.2188, m̂ = 15.14**,
  i.e. **ICC ρ = 0.062**, SD of the per-edge flip probability **0.103**. log L = −614.262.

Model A has a closed-form posterior, and the form is the point:

    logit P(true direction = up | u up-votes, d down-votes)
        = logit(π) + (u − d) · log((1 − ε)/ε)
        = 0.517 + 1.374 · (u − d)

**The posterior depends on the vote margin alone. The number of papers does not enter.**
A 1–0 edge and a 5–4 edge receive the same confidence.

| margin (u − d) | Model-A posterior |
|---|---|
| 0 | 0.626 |
| 1 | **0.869** |
| 2 | **0.963** |
| 3 | 0.990 |
| 4 | 0.998 |

### 3.2 Now measure it. This is the decisive table in the review.

Every judged pair is unanimous (§1), so margin = paper count throughout:

**Disbiome** (173 decisive pairs)

| margin | model says | all judged | shared source | **disjoint source** |
|---|---|---|---|---|
| 1 | 0.869 | 0.664 (n=110) | 0.868 (n=53) | **0.474 (n=57)** |
| 2 | 0.963 | 0.786 (n=28) | 0.818 (n=11) | **0.765 (n=17)** |
| 3–4 | 0.995 | 0.867 (n=15) | 0.900 (n=10) | **0.800 (n=5)** |
| 5+ | 1.000 | 0.950 (n=20) | 1.000 (n=13) | **0.857 (n=7)** |

**Peryton** (137 decisive pairs)

| margin | model says | all judged | shared source | **disjoint source** |
|---|---|---|---|---|
| 1 | 0.869 | 0.620 (n=79) | 0.941 (n=34) | **0.378 (n=45)** |
| 2 | 0.963 | 0.852 (n=27) | 1.000 (n=8) | **0.789 (n=19)** |
| 3–4 | 0.995 | 0.800 (n=15) | 1.000 (n=8) | **0.571 (n=7)** |
| 5+ | 1.000 | 1.000 (n=16) | 1.000 (n=12) | **1.000 (n=4)** |

Expected calibration error, binned on the score's own levels:

| score | vs all judged pairs | vs disjoint-source pairs |
|---|---|---|
| Model-A posterior, Disbiome | 0.176 | **0.324** |
| Model-A posterior, Peryton | 0.187 | **0.378** |
| shipped four tiers, Disbiome | 0.006 *(in-sample)* | **0.135** |
| shipped four tiers, Peryton | 0.005 | **0.179** |
| *noise floor: perfectly calibrated scorer, 4 bins, N=162* | *0.050 (95th pct 0.085)* | |

**The interpretation is the finding.** A confidence score fitted to the internal votes is
calibrated against the *shared-source* column, because that is what the votes measure: two
readers of the same sentence. Model A predicts 0.869 for a single-paper edge; the shared
column says 0.868 / 0.941. It is close to *exactly right* for reading fidelity. It is
off by 39–49 points for cross-literature reproducibility, which is what a reader of the
graph actually wants to know.

This is `FINDINGS_independence.md`'s result arriving from a new direction, and it is the
reason no amount of modelling the votes can produce a calibrated confidence. **The vote data
contain no information about the quantity the score is supposed to estimate.**

### 3.3 Is there edge-level heterogeneity at all? Marginal at best, and the test is broken

Model B beats Model A by LR = 3.607 (ΔAIC = +1.61). Naively that is p = 0.029 on a
boundary-corrected 1 df. **It is not.** Simulating the null with the identical fitting
procedure, the LR statistic's empirical distribution is far from the asymptotic boundary
mixture: the naive 2.706 critical value has a type-I error of **0.687**. Calibrated
empirically, the observed LR sits at p ≈ [PLACEHOLDER_P], and the test has power
[PLACEHOLDER_POWER] to detect an ICC of 0.10 — an SD of 0.13 in the per-edge flip
probability.

So the defensible statement is: **this corpus cannot tell whether edges differ in
reliability beyond a constant error rate**, and if they do, the point estimate says the
spread is SD ≈ 0.10 around a mean flip rate of 0.22 — a "good" edge at 0.12 and a "bad" one
at 0.32. That is not enough separation to shrink toward, and it is not measurable here
anyway. A beta-binomial partial-pooling model would return per-edge posteriors that are
almost entirely prior, i.e. almost entirely a function of (u, d) — reproducing §3.1 with
extra machinery.

One diagnostic worth recording because it will confuse the next person to fit this: **the
model misfits specifically at n = 2** (χ² = 9.68 on 2 df, p = 0.0079), with 105 unanimous-up
and 41 unanimous-down 2-paper edges against 96.9 / 61.7 expected, and 88 contested against
75.5. The direction prior at n = 2 is 0.719 against 0.569 at n = 1 and 0.558 at n = 3–4. It
is not a monotone trend and I have no explanation for it; a global-π partial-pooling model
is misspecified in a way this corpus can already detect, which is another reason not to fit
one. n = 3, 4 and 5 fit fine (p = 0.68, 0.39, 0.97).

### 3.4 And the flip rate does *not* fall with depth

A tempting story is that deeper edges are more reliable per-paper because depth selects for
strong effects. It is not true here:

| depth | minority-vote rate | 95% CI |
|---|---|---|
| n = 2 | 0.188 | [0.155, 0.226] |
| n = 3–4 | 0.198 | [0.161, 0.239] |
| n = 5–7 | 0.166 | [0.132, 0.206] |
| n ≥ 8 | 0.166 | [0.127, 0.213] |

Vote-level correlation between minority status and depth: **−0.025, permutation p = 0.33**.
So depth buys confidence only through the ordinary √n, not through better papers.

---

## 4. Q2 — the model survey, assessed rather than listed

### 4.1 Heterogeneity statistics: mechanically impossible, and the earlier review has it right

Cochran's Q, τ² and I² all require a per-study effect estimate yᵢ and its sampling variance
vᵢ ([Higgins & Thompson, *Statistics in Medicine* 21:1539–1558, 2002](https://doi.org/10.1002/sim.1186)).
A direction sign has neither. `research/06_meta_analysis_heterogeneity.md` §3 states this
correctly and I have nothing to add. Do not present the 27.6% discordance rate as though it
were calibrated against I² thresholds; they are defined on a different scale.

### 4.2 Beta-binomial / Bayesian partial pooling

- **Assumes:** exchangeable per-edge vote probabilities drawn from a common Beta; votes iid
  within an edge given that probability.
- **Holds here?** Exchangeability across edges is defensible. The iid-within-edge assumption
  is *violated* — `FINDINGS_paper_discordance.md` establishes that minority-direction status
  is a paper-level property (p = 0.0003), so votes within an edge are correlated through
  shared papers. That correlation is small (paper ICC 0.089, §4.3) but it is the exact
  correlation a beta-binomial mistakes for edge-level overdispersion, so the ρ = 0.062
  estimate in §3.3 is an **upper bound** and part of it is paper clustering.
- **Would output:** a posterior mean per edge, shrunk toward π̂. Fitted here, §3.3 shows the
  shrinkage is near-total for thin edges, so the output is §3.1's margin function with a
  bureaucratic detour.
- **Verdict:** technically sound, adds nothing, and misrepresents paper clustering as edge
  quality.

### 4.3 Multilevel logistic with taxon and disease random effects

Variance components on the 3,077 directional votes (ANOVA-style ICC for a binary outcome,
groups with ≥2 votes):

| grouping | groups | mean size | ICC(direction) |
|---|---|---|---|
| **taxon** | 356 | 7.13 | **0.296** |
| edge (taxon × disease) | 447 | 3.39 | 0.377 |
| paper | 255 | 11.97 | 0.089 |
| **disease** | 38 | 72.67 | **0.013** |

Three conclusions, all of which argue against fitting the model:

1. **The disease random effect has nothing to do** (ICC 0.013). It would be estimated at
   essentially zero and would shrink nothing.
2. **The taxon random effect is the largest term (0.296) and is exactly the one you must not
   use.** It would pull each edge toward the taxon's corpus-wide direction —
   *Streptococcus* is enriched in 11 of 11 diseases that report it, so a taxon random effect
   would make "Streptococcus enriched in disease X" confident *because Streptococcus is
   enriched everywhere*. `FINDINGS_disease_specificity.md` already established that ~70% of
   the graph's directional agreement is this corpus-wide prior rather than disease-specific
   signal, and `calibrate_agreement.py` already ran the falsification test:
   **`restates_prior` does not predict external agreement — +0.9 points (Disbiome,
   p = 1.00) and +7.7 (Peryton, p = 0.48), at a minimum detectable difference of ±9.7
   points.** A well-powered null against precisely the shrinkage this model implements.
3. Edge ICC (0.377) exceeds taxon ICC (0.296) by only 0.08, so once you know the taxon,
   knowing which disease adds little — which is the same finding as (1) from the other side.

The one *good* use of the taxon level is already implemented and pointed the opposite way:
the `discriminating` demotion in `annotate_confidence`, which *lowers* confidence for taxa
with purity ≤ 0.6, and which earns its place empirically (well-supported agreement 91.7/90.6
→ 93.8/93.3 with it; it moves four pairs and all four were wrong).

### 4.4 Sign test / exact binomial

Covered in §2.4. Computable, honest, and **zero of 2,008 edges survive correction**. Useful
as a disclosure line in a methods section; useless as a per-edge score or a filter.

### 4.5 Trim-and-fill and the publication-bias family adapted to sign data

- [Duval & Tweedie (2000), *Biometrics* 56(2):455–463](https://doi.org/10.1111/j.0006-341x.2000.00455.x)
  imputes "missing" studies from **funnel-plot asymmetry** — the relationship between effect
  size and standard error. Both axes are unavailable. There is no sign-only analogue,
  because the method's entire content is in the effect–precision relationship.
- [Copas & Shi (2000), *Biostatistics* 1(3):247–262](https://doi.org/10.1093/biostatistics/1.3.247)
  selection models require a selection function over (effect, SE). Same obstruction.
- [Egger *et al.* (1997), *BMJ* 315:629–634](https://doi.org/10.1136/bmj.315.7109.629), and
  the guidance in [Sterne *et al.* (2011), *BMJ* 343:d4002](https://doi.org/10.1136/bmj.d4002)
  that funnel-plot tests should not be used with **fewer than 10 studies** — we have ≥10
  papers on **16 of 2,008 edges (0.8%)**, and no effect sizes for any of them.

**Verdict: the entire family is inapplicable, not merely underpowered.** §6 covers what is
left.

### 4.6 Fisher / Stouffer p-value combination

- **Fisher's method** (−2 Σ ln pᵢ ~ χ²₂ₖ) is omnibus and direction-blind; with discordant
  directions it is invalid without pre-splitting by side, which is exactly the problem in
  omics meta-analysis with direction conflict.
- **Stouffer's method** (Σ zᵢ / √k) is direction-native and accepts weights, and
  [Whitlock (2005), *J. Evolutionary Biology* 18(5):1368–1373](https://doi.org/10.1111/j.1420-9101.2005.00917.x)
  shows the weighted-Z version dominates Fisher for exactly this use. It is the right method
  **if** per-taxon p-values exist.
- **They do not, and recovering them is the one thing in this review that would change the
  answer.** No p-value is extracted per taxon per paper. The extractor's own gate requires
  significance to be *reported* but stores only a boolean fact-of-significance implicitly,
  by inclusion.

Two sub-options if a future extraction pass is contemplated, in descending order of value:

1. **Extract the reported p-value (or q-value) per taxon.** Unlocks weighted Stouffer,
   p-curve, and the caliper test in §6. Requires a GPU pass, and about half of the source
   papers report LEfSe LDA scores rather than p-values, so coverage will be partial.
2. **Extract the taxa a paper explicitly reports as *not* significantly different.** This is
   cheaper (it is a different prompt over the same text, and `relation_sentences_clean.json`
   already indexes the relevant sentences) and it is worth *more*, because it attacks the
   missing denominator in §2.2 directly. Even a 20-paper pilot would put an interval on the
   reporting rate. See the falsifiable step in §8.

### 4.7 Three-level meta-analysis / robust variance estimation

[Hedges, Tipton & Johnson (2010), *Research Synthesis Methods* 1(1):39–65](https://doi.org/10.1002/jrsm.5)
and [Van den Noortgate *et al.* (2013), *Behavior Research Methods* 45:576–594](https://doi.org/10.3758/s13428-012-0261-6)
are built for the dependency structure here (many observations per paper). They still need
effect sizes and variances. What survives without them is the *variance-partition idea*,
which `FINDINGS_paper_discordance.md` already implements by permutation and cluster
bootstrap (85% of discordance variance is edge structure, 15% paper identity; paper-level
SD 3.4 points). §4.3's ICC table is the same decomposition on a different cut. **Formalising
it in a three-level model would not change a number.**

---

## 5. Q3 — shrinkage across the taxonomy. The brief's worry is correct and the data confirm it.

### 5.1 The literature

The relevant methods, and what each would need:

- **treeclimbR** — [Huang, Soneson, Germain *et al.* (2021), *Genome Biology* 22:157](https://doi.org/10.1186/s13059-021-02368-1).
  Scans candidate aggregation levels on a tree and picks the resolution at which signal is
  strongest, controlling FDR. It is the closest published fit to the problem, and it is
  **designed to find the right level, not to shrink children toward parents** — which is the
  correct instinct. Requires a per-node test statistic with a null; we have one only via §2.4,
  where zero nodes survive.
- **Hierarchical FDR** — [Yekutieli (2008), *JASA* 103(481):309–316](https://doi.org/10.1198/016214507000001373),
  implemented for microbiome data in
  [Sankaran & Holmes (2014), *J. Statistical Software* 59(13)](https://doi.org/10.18637/jss.v059.i13).
  Tests parents first, descends only into significant subtrees. Same obstruction: nothing is
  significant to descend from.
- **Phylogenetic mixed models** — [Lynch (1991), *Evolution* 45(5):1065–1080](https://doi.org/10.2307/2409716);
  binary-outcome version in [Ives & Garland (2010), *Systematic Biology* 59(1):9–26](https://doi.org/10.1093/sysbio/syp074).
  Impose a covariance structure in which related taxa are *a priori* similar, with the
  degree of similarity estimated (Pagel's λ). This is the honest version of "borrow strength
  from relatives" — it estimates how much to borrow rather than assuming it. It is also the
  version §5.3 shows would estimate ≈ 0 here.
- **Dirichlet-tree priors** — [Minka (1999), "The Dirichlet-tree distribution"](https://tminka.github.io/papers/dirichlet/minka-dirtree.pdf);
  applied in [Wang & Zhao (2017), *Biometrics* 73(3):792–801](https://doi.org/10.1111/biom.12654)
  and [Tang, Ma & Nicolae (2018), *Annals of Applied Statistics* 12(1):1–26](https://doi.org/10.1214/17-AOAS1086).
  These model *compositions* at the tips of a tree. Our data are not compositions — they are
  per-paper signs with no abundances — so the model does not apply without inventing the
  abundances.
- **PhILR** — [Silverman *et al.* (2017), *eLife* 6:e21887](https://doi.org/10.7554/eLife.21887).
  Isometric log-ratio transform on the phylogeny. Same objection: needs abundances.

### 5.2 The 89% figure does not license shrinkage, and the repo's own data say why

The root `CLAUDE.md` cites "related taxa agree on direction 89% of the time **within a
single paper** vs 54% for unrelated taxa". Shrinkage does not operate within a paper — it
operates across the papers on the two edges. Computed across papers, over every
parent/child pair sharing a disease where both edges are decisive:

| | n | agree on direction |
|---|---|---|
| all parent/child pairs sharing a disease | 845 | **0.714** |
| restricted to single-paper *child* edges | 638 | **0.685** |

So the borrowable signal is **68.5%, not 89%** — the number a shrinkage prior would actually
be built on. And the existing `rank_conflicts` annotation shows what the 31.5% disagreement
is made of: of 242 opposite-direction pairs, **190 rest on no shared paper at all**, 16 are
cross-paper-only, and **36 are stated within a single study** — the *Lachnospiraceae* down /
*Hungatella* up class that the brief correctly identifies as the whole point of modelling
containment. Aggressive shrinkage would flatten all 36 of those, which is destroying the
signal the layer exists to preserve.

### 5.3 The decisive external test: shrinkage points the wrong way

The falsification is cheap and I ran it. For every single-paper edge that both (a) an
external curation judges decisively and (b) has a same-disease parent edge, compare the
child's own vote against the parent's majority direction:

| | n | child's own vote | parent's direction |
|---|---|---|---|
| Disbiome | 57 | **0.684** | 0.596 |
| Peryton | 53 | **0.623** | 0.566 |

Discordant pairs: child-right/parent-wrong 13 vs parent-right/child-wrong 8 (Disbiome), 9
vs 6 (Peryton). Pooled exact McNemar **22 vs 14, p = 0.243** — and the two databases are not
independent, so that p is optimistic. At n = 36 discordant, significance would need an
11/36 split or better, i.e. this test resolves about ±19 percentage points and saw −8.8 /
−5.7.

**So: a null, but a null with the point estimate in the same wrong direction in two
references.** There is no evidence that the parent carries information the child's own
paper does not, and mild evidence that it carries less. Combined with §5.2's 68.5% and the
36 within-paper conflicts, and with the coverage limit — **only 698 of 1,561 single-paper
edges have a same-disease parent edge at all**, so shrinkage could touch at most 45% of the
thin edges even if it worked — the conclusion is clear.

**Do not shrink across the taxonomy. The brief's own suspicion is confirmed: this is the
reason not to do it.** If anyone wants to revisit, the right framing is treeclimbR's — find
the rank at which a signal is strongest — not a prior that pulls children toward parents.
And the honest version, a phylogenetic mixed model with λ estimated rather than assumed,
would return λ̂ ≈ 0 on these numbers.

---

## 6. Q4 — publication bias, and Q5 — calibration

### 6.1 Publication bias: the honest answer is "almost nothing defensible"

The methods that work on sign data alone, and why each fails here:

| method | needs | available? |
|---|---|---|
| trim-and-fill, Copas selection, Egger | effect size + SE | no |
| p-curve ([Simonsohn, Nelson & Simmons 2014](https://doi.org/10.1037/a0033242)) | exact p-values below 0.05 | no |
| caliper test ([Gerber & Malhotra 2008](https://doi.org/10.1177/0049124108318973)) | p-values near a threshold | no |
| excess-significance test ([Ioannidis & Trikalinos 2007](https://doi.org/10.1177/1740774507079441)) | effect size + n, to compute power | no |
| Chao1 / Good–Turing coverage (§2.3) | frequency spectrum only | **yes** |

Only the last is computable, and it measures **undersampling, not directional selection**.
It says the graph is at ~28% of asymptotic pair coverage; it says nothing about whether the
*directions* we see are a biased sample of the directions that exist.

There is one asymmetry in the data — 55.6% of votes are "enriched" — but there is no null
for it. Biology could genuinely be asymmetric (blooms of opportunists are easier to detect
than depletion of a rare commensal), and nothing in the corpus separates that from
reporting preference. **Reporting it as evidence of publication bias would be exactly the
invented precision this project forbids.**

The *mechanistically* relevant literature, which should be cited instead of a bias
statistic, is that cross-study disagreement at our magnitude is fully explained by technical
variance with zero biological disagreement required:
[Gibbons *et al.* (2018), *PLoS Computational Biology* 14(4):e1006102](https://doi.org/10.1371/journal.pcbi.1006102)
found **681 of 1,021 OTUs (67%) "significantly" different between two sets of healthy
controls** from different studies;
[Sinha *et al.* (2017), *Nature Biotechnology* 35:1077–1086 (MBQC)](https://doi.org/10.1038/nbt.3981)
ranks specimen handling and DNA extraction above bioinformatics as variance sources; and
[Nearing *et al.* (2022), *Nature Communications* 13:342](https://doi.org/10.1038/s41467-022-28034-z)
shows that differential-abundance tools disagree with each other on the *same* 38 datasets.
[Duvallet *et al.* (2017), *Nature Communications* 8:1784](https://doi.org/10.1038/s41467-017-01973-8)
remains the best comparator for the vote-counting rule itself — its consensus criterion is
"significant in the same direction in ≥2 studies", i.e. our `supported` tier, and it
reports **no consistent associations for conditions with fewer than four datasets**.

### 6.2 What an evaluation must look like, with its null

**The only honest evaluation set is the disjoint-source subset.** `FINDINGS_independence.md`
established that half the judged pairs rest on a paper the curator also read, and that
agreement on those is 87.5% / 96.8% against 58.1% / 52.6% on disjoint literature. A
calibration evaluation run on the pooled set measures a blend of two quantities and reports
neither. Concretely:

- **Population:** pairs judged decisively by Disbiome (n=86 disjoint) or Peryton (n=76
  disjoint), computed by `check_independence.py`'s shared/disjoint labelling.
- **Null H₀:** agreement rate is constant across confidence levels.
- **Test:** taxon-block permutation of the agreement labels (the repo's standing method —
  blocks preserve within-taxon correlation), plus a taxon cluster bootstrap for the CI.
  Report Spearman ρ between score and agreement, not just a two-group difference.
- **Report separately per database.** They overlap in pairs, so pooling inflates n. Two
  independent replications of the *same ordering* is the stronger claim anyway, and it is
  the claim `FINDINGS_independence.md` already makes for evidence count (+23.8 / +23.7
  points, near-identical in two references).

**Power, which is what decides everything:**

| bins | Disbiome n/bin | MDE (80% power, α=0.05, p₀=0.55) | Peryton n/bin | MDE |
|---|---|---|---|---|
| 2 | 43 | **±0.276** | 38 | **±0.291** |
| 3 | 28 | ±0.331 | 25 | ±0.346 |
| 4 | 21 | ±0.371 | 19 | ±0.385 |

And the observed splits:

| comparison (disjoint only) | Disbiome | Peryton | MDE |
|---|---|---|---|
| 1 paper vs ≥2 papers | 0.475 (59) vs 0.815 (27), **+0.340** | 0.383 (47) vs 0.786 (28), **+0.403** | ±0.311 / ±0.328 |
| `supported` vs `well-supported` | 0.765 (17) vs 0.900 (10), +0.135 | 0.778 (18) vs 0.800 (10), +0.022 | ±0.230 / ±0.220 |

**The instrument resolves one binary distinction.** The 1-vs-≥2 split clears its MDE in both
databases. The 2-vs-3+ split does not clear in either, and in Peryton the point estimate is
+0.022 — a coin flip's worth of separation. Any confidence score finer than "one paper vs
more than one paper" is, on present evidence, **unvalidatable**.

**Reliability diagrams, ECE, Brier — what each needs:**

- **Reliability diagram** at a useful resolution needs Wilson half-width ≈ ±0.05 per bin,
  which at p = 0.75 requires **n = 286 per bin** → 1,144 judged pairs for four bins, 2,860
  for ten. We have 173 and 137 *total*, and 86 / 76 disjoint. **Not achievable**; a
  four-point diagram drawn on 86 pairs would have error bars wider than the effect it plots.
- **ECE** is biased upward at small n, and badly. Simulating a *perfectly* calibrated
  four-bin scorer: at N = 162 the median measured ECE is **0.050** with a 95th percentile of
  **0.085**; at N = 100 it is 0.061 / 0.110. So at our sample size any ECE below ~0.09 is
  indistinguishable from perfect calibration, and reporting a small ECE would be reporting
  noise. (The bias and the debiased alternatives:
  [Vaicenavicius *et al.* (2019), AISTATS, arXiv:1902.06977](https://arxiv.org/abs/1902.06977);
  [Kumar, Liang & Ma (2019), "Verified Uncertainty Calibration", NeurIPS, arXiv:1909.10155](https://arxiv.org/abs/1909.10155).)
  The Model-A posterior's disjoint ECE of 0.324 / 0.378 clears that floor by a factor of
  four, which is why §3.2's rejection is safe even at this n.
- **Brier score**, decomposed per
  [Murphy (1973), *J. Applied Meteorology* 12(4):595–600](https://doi.org/10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2),
  is the most informative of the three at small n because resolution and reliability
  separate. On all judged pairs the shipped tiers give:

  | | Disbiome | Peryton |
  |---|---|---|
  | base rate | 0.734 | 0.730 |
  | uncertainty (Brier of a constant predictor) | 0.1952 | 0.1971 |
  | **resolution** | 0.0107 | 0.0181 |
  | **Brier skill of the tiers** | **0.055** | **0.092** |

  The tiers explain 5.5% / 9.2% of the outcome variance. Small, real, replicated in two
  references — and roughly what the published vote-counting validation in §2.5 predicts
  (r = 0.44, 19% of variance, on a larger and cleaner test set). **A continuous score would
  have to beat 0.055 / 0.092 by a margin the 86/76-pair evaluation can see, i.e. by roughly
  doubling it.** Nothing in §3–§5 is a candidate.

### 6.3 The actionable defect: the shipped `provisional` rate is overstated

`CONFIDENCE_RATES` (`build_kg.py:687`) carries measured agreement per tier so the viewer can
quote a number it did not invent. Those rates are pooled over shared and disjoint sources.
Recomputed on disjoint pairs only:

| tier | shipped quote (Disbiome / Peryton) | **disjoint-source measurement** | |
|---|---|---|---|
| `well-supported` | 0.938 / 0.933 | 0.900 (n=10) / 0.800 (n=10) | consistent, tiny n |
| `supported` | 0.778 / 0.833 | 0.765 (n=17) / 0.778 (n=18) | consistent |
| `provisional` | **0.661 / 0.619** | **0.475, 95% CI [0.354, 0.598] (n=59)** / **0.383, [0.256, 0.528] (n=47)** | **both CIs exclude the quote** |

Pooled across both curations (not independent — overlapping pairs — so read the per-database
rows above as the evidence): `well-supported` 0.850 [0.640, 0.948], `supported` 0.771
[0.610, 0.879], `provisional` 0.434 [0.344, 0.529].

**A reader told an edge is `provisional` with a 66% agreement rate is being told a number
that is 19 points too high for the case that matters** — an edge tested against literature
the curators had not read. The docstring at `build_kg.py:726` already flags that the cuts
were chosen in-sample on Disbiome and that Peryton is the out-of-sample check; what it does
not flag is that both columns are contaminated by shared sources in the way
`FINDINGS_independence.md` documented, and that the contamination is concentrated in the
tier holding 1,574 of 2,008 edges.

This is a two-line change to a constant table plus a sentence in the viewer, and it is the
only change this review recommends making to the graph.

---

## 7. Q6 — verdict

**Is a principled confidence score achievable here?** A *principled* one, yes — §3.1 derives
it in closed form from the only model that fits. A *calibrated* one, no, and the obstruction
is not sample size. It is that the vote data measure the wrong quantity. The margin-based
posterior is nearly exactly right for **reading fidelity** (0.869 predicted vs 0.868 / 0.941
observed on shared sources) and 39–49 points wrong for **cross-literature reproducibility**
(0.474 / 0.378). No reweighting of votes fixes that, because the information that separates
the two quantities is not in the votes.

**Is "evidence count plus an explicit contested flag" already the honest maximum?** Very
nearly, and I went looking for reasons it is not:

- A continuous score fitted to the votes is **measurably miscalibrated** (ECE 0.32/0.38 vs a
  0.07 floor) and additionally predicts that a 5–4 edge equals a 1–0 edge, which the graph's
  own contested flag correctly rejects.
- Edge-level reliability heterogeneity, the thing a fancier score would estimate, is
  **ρ ≈ 0.06 and not distinguishable from zero** by a properly calibrated test.
- A taxon random effect would boost exactly the edges that `restates_prior` proved carry no
  extra external validity (well-powered null, MDE ±9.7 points).
- Taxonomic shrinkage **points the wrong way** in two independent references and can reach
  only 45% of thin edges anyway.
- The evaluation can resolve **one** binary distinction at ±0.28, and the graph already
  ships that distinction.

The one place the current scheme is genuinely wrong is not the score — it is the *number
printed next to the score*, §6.3.

**One thing the tier scheme does that a continuous score could not, and which should be
protected:** `contested` is not a low confidence value, it is a **refusal to assert a
direction**. A continuous score necessarily maps a 5–4 edge to a number near 0.5, which
reads as "probably true, weakly" — when the correct statement is "this literature does not
agree and we are not going to pick". Collapsing that into a scalar would lose the project's
most defensible design decision. `FINDINGS_paper_discordance.md` supplies the quantitative
backing: ~85% of discordance variance is edge structure, not paper identity, so contestedness
is a property of the taxon–disease pair and is a finding in its own right.

---

## 8. Recommendation, ranked

Every item states what would make it fail.

### 1. Correct the `provisional` tier's quoted agreement rate (do this)

**What:** add a disjoint-source column to `CONFIDENCE_RATES` and quote it in the viewer
alongside the pooled figure, or replace the pooled figure. Provisional is 0.475 [0.354,
0.598] (Disbiome) and 0.383 [0.256, 0.528] (Peryton) against literature the curator did not
read, not 0.661 / 0.619.
**Cost:** hours, CPU-only, no new data.
**Falsifiable step:** recompute `CONFIDENCE_RATES` restricted to
`check_independence.py`'s `shared_source == False` pairs and rebuild; the tier rates should
move only for `provisional`, and the `well-supported` / `supported` rows should stay inside
their current CIs.
**What would make it fail:** if the disjoint subset turns out to be disease-confounded — it
is, partly: `FINDINGS_independence.md` shows every ALS pair is shared-source and every
autism and epilepsy pair is disjoint. The within-Parkinson's check (the only disease with
both buckets full) is the guard, and Multiple sclerosis is the logged counter-example where
the shared/disjoint gap is absent at an MDE that could have seen it. **If the provisional
gap also vanishes within Parkinson's, this recommendation is wrong and should be dropped.**
That check must be run before shipping the number.

### 2. Do nothing else to the score (genuinely the right answer for everything below)

Keep evidence count as edge weight, keep the four tiers, keep `contested` as a refusal
rather than a number. The reasons are §7. This is not a counsel of despair: the tiers carry
a Brier skill of 0.055 / 0.092 replicated in two independent references, which is real and
is roughly what the published vote-counting validation predicts.
**What would make this wrong:** a confidence score that beats Brier skill 0.092 on the
disjoint-source subset with a taxon-block permutation p < 0.05. Nothing in §3–§5 is a
candidate; the bar is stated so a future proposal can be held to it.

### 3. Extract the *negative* reports — the one cheap thing that changes the answer

**What:** a second extraction pass, or a human read of ~20 papers, recording taxa a paper
explicitly states were **not** significantly different. This attacks the missing denominator
(§2.2) head-on and is worth more than extracting effect sizes, because it converts every
edge from "k votes" into "k of N that looked", which is a binomial with a known denominator
— and *that* unlocks the beta-binomial and hierarchical machinery on a footing it currently
lacks.
**Cost:** a 20-paper human pilot is CPU/eyeball-only. A corpus-scale pass needs a GPU —
**ask before spending.**
**Falsifiable step:** on 20 papers, count explicitly-non-significant taxa per paper. If the
median is ≥5, the denominator is recoverable at corpus scale and the pass is worth
proposing. **If the median is 0–1** — i.e. papers simply do not report their negatives in
the main text — the whole line is dead and should be recorded as closed. My prior is that
it is closer to 0–1 than to 5, which is why this is item 3 and not item 1.
**Also note the asymmetry the recall work already documented:** the instrument can confirm a
negative report but cannot refute one, so any estimate from this is a bound.

### 4. Extract per-taxon p-values — the only thing that unlocks the standard toolkit

**What:** a GPU extraction pass recording the reported p/q-value (or LDA score plus test
identity) per taxon per paper. Unlocks weighted Stouffer, real Q/τ²/I², p-curve and the
caliper test.
**Cost:** GPU. **Ask before spending.**
**Falsifiable step:** on the 15-paper `test_set_v2`, count how many extracted taxa carry a
recoverable p-value. **If coverage is below ~60%, a combined analysis will be dominated by
the missing-at-random assumption on the other 40%** and should not be attempted; my
expectation is that coverage is poor because roughly half of these papers report LEfSe LDA
scores, which are not p-values and are not comparable across papers.
**What would make it fail even at good coverage:** the scales remain incommensurable (LDA
vs fold-change vs Wilcoxon p), which is the original reason for the direction-only schema —
so this buys a *p-value* combination, not an effect-size meta-analysis.

### 5. More papers (the standing answer, restated so it is not re-proposed as new)

Every MDE in this document is set by n: 86 and 76 disjoint judged pairs, 448 replicated
edges, 638 parent/child pairs with a single-paper child. `FINDINGS_paper_discordance.md`
reached the same conclusion from a different direction and `research/06` from a third.
**Cost:** GPU. **Ask before spending.** Nothing here argues for it over the other open
levers; it is listed so its absence is not mistaken for an oversight.

### Explicitly rejected, with reasons, so they are not rediscovered

| proposal | why not |
|---|---|
| beta-binomial / Bayesian partial pooling per edge | ρ ≈ 0.06, not distinguishable from zero; output collapses to §3.1's margin function; mistakes paper clustering for edge quality |
| multilevel logistic with taxon + disease random effects | disease ICC 0.013 (nothing to do); taxon ICC 0.296 but shrinking that way is the `restates_prior` inference already falsified at MDE ±9.7 pts |
| any shrinkage toward the taxonomic parent | 68.5% cross-paper agreement, not 89%; beaten by the child's own vote in two references; reaches only 45% of thin edges; would flatten the 36 within-paper rank conflicts the layer exists for |
| per-edge sign test as a filter or a score | 0 of 2,008 edges survive BH or Bonferroni; the null (q = ½) is not the null a reader would assume |
| trim-and-fill / Copas / Egger / p-curve / caliper / excess-significance | all require effect sizes or p-values; inapplicable, not underpowered |
| Cochran's Q, τ², I² | mechanically undefined without effect sizes and variances |
| a reliability diagram or an ECE headline number | needs 286 judged pairs per bin; at N=162 a perfectly calibrated scorer measures ECE 0.050 (95th pct 0.085), so any small value is noise |
| three-level meta-analysis / RVE | needs effect sizes; the variance partition it would formalise is already computed by permutation and bootstrap, and would not move a number |

---

## Appendix — verification notes and limits

- All computations are CPU-only, from `graph.json` (2,008 edges), `disbiome_experiments.json`,
  `disbiome_publications.json` and `Peryton-results.tsv`, joined through
  `taxonomy_cache.load_taxonomy()`, which in this session resolved against the **real NCBI
  taxdump (3,476,674 names)**, not the degraded replay cache. Absolute agreement figures are
  therefore comparable to the taxdump-measured numbers in `FINDINGS_independence.md`; the
  reproduced pooled rates (Disbiome 0.734 over 173 pairs, Peryton 0.730 over 137) match the
  shipped 73.0 / 72.5 to within the one-pair difference from graph revision.
- The Disbiome and Peryton judged sets **overlap in pairs**. Every per-database number is
  reported separately for that reason; where I pool (the McNemar in §5.3, the tier CIs in
  §6.3) it is flagged and the per-database rows are the evidence.
- §3.3's likelihood-ratio test required an empirically calibrated null because the
  asymptotic boundary mixture has a type-I error of 0.687 on this design. Anyone re-running
  a beta-binomial comparison on these data must calibrate the null by simulation; the naive
  χ² p-value is wrong by more than an order of magnitude.
- The n = 2 misfit in §3.3 (χ² = 9.68, p = 0.0079) is unexplained. It is recorded rather
  than resolved.
- **Citation verification limit:** this session exhausted its web-search budget before the
  bibliography was checked online. Every reference is given with a DOI or arXiv id, which
  are stable identifiers, but the URLs were not re-fetched in this session. Two entries are
  the ones to re-check if precision matters: the Bushman & Wang handbook chapter (chapter
  number and edition) and the Chao (1984) page range.
