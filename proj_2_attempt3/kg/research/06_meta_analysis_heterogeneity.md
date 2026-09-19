# Meta-analysis / heterogeneity literature review: what applies to 220 contested edges

**Question:** we have 220 taxon-disease pairs where papers disagree on direction
(enriched vs depleted). 43 pairs have >=2 papers on both sides, at most 7-vs-7,
mostly 2-4 per side. We have direction only (no effect sizes -- papers report
incommensurable LEfSe LDA scores, fold-changes, p-values). What does the
meta-analysis / reproducibility literature say about analyzing exactly this
kind of data, and what is actually achievable at this sample size?

---

## 1. Vote counting: the literature's verdict is settled and negative

Vote counting -- tallying how many studies point each direction and calling
the majority the "winner" -- is exactly what our edge weights currently are
(count of papers enriched vs depleted per taxon-disease pair).

The foundational critique is **Hedges, L.V. & Olkin, I. (1980), "Vote-counting
methods in research synthesis," *Psychological Bulletin* 88(2):359-369**
(doi:10.1037/0033-2909.88.2.359). Their result is counterintuitive and load-bearing:
**the statistical power of vote counting *decreases* as more studies are added**,
and can tend to zero as the number of studies grows, even though every
individual study has a real, non-null, correctly-signed effect. This is the
opposite of what every other meta-analytic method does (power should increase
with more studies) and is why "more evidence, worse decision" is the
literature's stock description of the method. It is bias-prone toward
concluding "no effect" specifically when true effects are small and study n is
low -- our exact situation (small taxa effect sizes, case-control n typically
in the dozens).

Borenstein, Hedges, Higgins & Rothstein's standard text, **"Introduction to
Meta-Analysis" (Wiley, 2009; 2nd ed. 2021)**, devotes a chapter to vote
counting specifically to warn practitioners off it, reiterating Hedges &
Olkin's power result and adding that vote counting **discards effect
magnitude and precision entirely** -- a study of n=200 and a study of n=12
get the same one vote. A 2023 methods paper is blunter in its title: **"Why
vote-counting is never acceptable in evidence synthesis"** (see
researchgate.net/publication/365957009). The applied-ecology literature
(Bushman & Wang's original "sign test" formalization, 1994, and the review at
pmc.ncbi.nlm.nih.gov/articles/PMC2817246) makes the same point about the
`vote-counting -> zero-effect bias` for small-effect small-n literatures.

**A defensible modern descendant exists and is directly relevant.** A
validation study of "vote counting to rank biomarkers from published studies"
(pmc.ncbi.nlm.nih.gov/articles/PMC4770796) tested whether the number of
studies supporting a direction predicts replication in an independent
45-sample test set. It does, but weakly: **number of supporting studies was a
significant predictor (p=0.0006) explaining only 19% of variance (r=0.44)**,
and predictive power was strongly n-dependent: **2 supporting studies -> ~50%
confirmation rate (a coin flip), 3 studies -> ~67-70%, 4+ studies -> ~90%**.
This is the closest the literature comes to calibrating exactly our
situation, and the number to take away is that **support from only 2-3
studies per side -- which is most of our 43 edges -- sits at or barely above
chance-level predictive value even under the most permissive published
validation of vote counting.** The same paper flags that fold-change and
total sample size add nothing once study count is known, which matters
because it means "add up the papers' effect sizes anyway, weighted by n" is
not an easy fix if the direction counts already contain most of the signal.

**Bottom line for us:** the current edge-weight-as-vote-count design is the
literature's most-criticized meta-analytic construct, and its own most
favorable modern validation says confidence should scale with support count
in a way that treats 2-3-paper edges as near-chance -- which is a another,
independent argument for the project's existing "contested edges are kept,
never averaged" policy (`CLAUDE.md`), not a call to add a vote-counting-based
confidence score to them.

## 2. Sign-based / direction-only combination when effect sizes are unavailable

When only direction (or only p-values without comparable effect sizes) is
available, the standard toolkit is **p-value combination**, not effect-size
pooling:

- **Fisher's method** (Fisher's combined probability test): combines p-values
  as `-2 * sum(ln p_i)`, ~ chi-squared with 2k df under the null. It is
  omnibus (tests "is there *any* signal") and is **not naturally
  direction-aware** -- a paper significant in the wrong direction contributes
  the same as one in the right direction unless p-values are pre-split by
  side. See en.wikipedia.org/wiki/Fisher's_method and the comparison in
  academic.oup.com/bioinformatics/article/36/2/524/5540321.
- **Stouffer's method** (inverse-normal / Z-score combination) is more
  natural here because it **combines signed Z-scores**, so direction is
  native to the method rather than bolted on, and it supports per-study
  weights (e.g., by sqrt(n)). See
  imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/CombiningPvalues and the
  applied write-up at r-bloggers.com/2018/05/stouffers-meta-analysis-with-weight-and-direction-effect-in-r.
- **The literature closest to our exact problem is omics meta-analysis with
  discordant direction** (RNA-seq / microarray differential expression across
  studies, structurally identical to "does taxon X go up or down across
  papers"). Standard guidance there
  (pmc.ncbi.nlm.nih.gov/articles/PMC4021464, and the "adaptively weighted
  Fisher" line of work, academic.oup.com/bioinformatics/article/36/2/524)
  is that when direction is inconsistent across studies, a plain Fisher
  combination is invalid, and the field has produced **direction-aware
  p-value combination tests** (e.g., "detect incomplete association" methods,
  nature.com/articles/s41598-021-86465-y) specifically because discordant
  direction is common (one comparison found **~50% of genes discordant in
  direction between two studied populations**) and naive combination
  overstates significance.
- **The binomial sign test** is the honest, minimal version of what we're
  already doing: under H0 that direction is a coin flip, `k` papers agreeing
  out of `n` gives an exact binomial p-value. This is what the task brief's
  own math already computes: **a 4-vs-3 split has only C(7,3)=35 label
  permutations, so p=1/35=0.029 is the smallest attainable p-value for that
  edge**, before any multiple-testing correction across 220 edges. After
  Bonferroni or BH correction across 220 tests, **no single edge at 2-4
  papers per side can reach significance** -- this is not a limitation of our
  analysis, it's a combinatorial floor.

**Applicability to us:** Stouffer/Fisher-style combination requires converting
each paper's result to a signed z-score, which requires either an effect size
or at least a comparable p-value per taxon per paper -- neither of which we
extract (we extract direction only, deliberately, per the extraction schema).
The binomial sign test *is* computable from what we have, but per point above
it cannot be significant at our per-edge n. It could, in principle, be
computed **pooled across all 220 edges** as a single omnibus statement
("across all contested edges, is the majority direction non-random overall?")
but that answers a different, much less useful question than "is edge X's
majority direction real."

## 3. Heterogeneity statistics (I-squared, tau-squared, Cochran's Q) require effect sizes and standard errors -- they cannot be computed from direction counts

This is a clean, mechanical fact worth stating plainly because it forecloses
an entire family of "just run the standard heterogeneity test" suggestions:

- **Cochran's Q** = `sum_i w_i * (y_i - y_bar)^2`, where `y_i` is each study's
  effect-size estimate and `w_i = 1/v_i` is the inverse of its sampling
  variance. **Both `y_i` and `v_i` are required inputs**; a direction sign has
  neither a magnitude nor a variance.
- **tau-squared** (between-study variance) is defined in the squared units of
  the effect size -- it is literally undefined without an effect-size scale.
- **I-squared** = `tau^2 / (tau^2 + sigma^2)`, a ratio of the above two
  quantities, inheriting the same requirement.

(See the mechanics laid out at
cjvanlissa.github.io/Doing-Meta-Analysis-in-R/heterogeneity-statistics.html
and casrai.org/guides/heterogeneity-in-meta-analysis.) None of these can be
computed from a table of "paper says up / paper says down." The closest
substitute with direction-only data is **descriptive discordance**: the
fraction of papers reporting the minority direction, which is exactly the
number this project already reports (27.6% pooled discordance across 1,367
decisive observations, `FINDINGS_paper_discordance.md`). That is a legitimate,
literature-recognized quantity (variance components language calls it
"raw between-study inconsistency"), but it is **not** I-squared or tau-squared
and should not be presented as calibrated against those thresholds (e.g. "I2
> 75% = high heterogeneity") because those thresholds are defined on a
different, effect-size-scaled quantity.

## 4. Meta-regression: the ~10-studies-per-moderator rule, and what it implies here

The standard guidance (Cochrane Handbook, Chapter 10, sections on subgroup
analysis and meta-regression: cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10)
states plainly: **it is unlikely that an investigation of heterogeneity will
produce useful findings unless there are at least 10 studies in the
meta-analysis** per moderator/covariate level, and explicitly flags that
**even 10 can be too few if the covariate is unevenly distributed** across
studies. AHRQ's applied guidance for comparative-effectiveness reviews is only
slightly more permissive: **6-10 studies per continuous covariate, minimum 4
per categorical covariate level**, and only for moderate-or-large primary
studies (ncbi.nlm.nih.gov/books/NBK49407). Multiple sources note these
thresholds are themselves **not empirically validated** rules of thumb, not
derived limits -- but they're the field consensus floor, and the field
consensus floor is already above what we have.

**Direct implication for the 43 dual-sided edges:** at 2-4 papers per side
(7 at most, one edge only), we are **below even the most permissive published
threshold for a single moderator, on a single edge**, before considering that
a real moderator analysis (study design, extraction kit, region, etc.) would
need to be tested **per edge**, multiplying the deficit by 43. This directly
confirms the project's own finding in `CLAUDE.md`/`FINDINGS_paper_discordance.md`
that 24 study-design and wet-lab covariates came back null at MDEs of 16-22%:
the literature's own guidance says a null was close to the only possible
outcome at this n, independent of whether the true moderator effects exist.

## 5. Multilevel / three-level meta-analysis and robust variance estimation (RVE) -- the one method built for exactly our data shape

Our actual data structure is: **one paper contributes many taxon-disease
observations** (multiple effect sizes nested within a study), which is
precisely the dependency structure that **three-level meta-analysis** and
**robust variance estimation (RVE)** were built to handle. Foundational
references:

- **Hedges, L.V., Tipton, E., & Johnson, M.C. (2010), "Robust variance
  estimation in meta-regression with dependent effect size estimates,"
  *Research Synthesis Methods* 1(1):39-65**
  (onlinelibrary.wiley.com/doi/10.1002/jrsm.5) -- introduces a sandwich-type
  covariance estimator for meta-regression coefficients when effect sizes are
  clustered (e.g., many effects per paper) and the exact within-cluster
  correlation structure is unknown or unmodeled.
- **Van den Noortgate, W. et al., "Meta-analysis of multiple outcomes: a
  multilevel approach," *Behavior Research Methods* (2014)**
  (link.springer.com/article/10.3758/s13428-014-0527-2) -- the three-level
  model, which separates variance into within-study, between-study, and
  sampling levels, letting you ask "how much of the disagreement is
  *within-paper* vs *between-paper*" directly.
- A head-to-head comparison
  (link.springer.com/article/10.3758/s13428-018-1156-y) notes the practical
  trade-off: **RVE combines within- and between-study heterogeneity into one
  number and needs no correlation assumption; the three-level model can
  separate the two variance components but needs either the correlation
  structure specified or estimable.**

**This is the one part of the standard toolkit that is directly applicable to
our data as it exists** -- not per-edge (n too small, as above) but
**across the whole corpus**: a three-level model with papers at the top
level and taxon-disease observations nested inside would let us ask, e.g.,
"how much of the corpus-wide directional variance is between-paper vs
within-paper" as a formal variance decomposition, rather than the informal
version already done by hand (`FINDINGS_paper_discordance.md`'s
cluster-bootstrap finding that only ~15% of variance is between-paper,
3.4pp SD on a 27.6% base). Notably, that existing informal analysis is
*already* doing, by permutation/bootstrap, roughly what a three-level model's
variance-partition would formalize -- so this section validates the
project's ad hoc approach rather than pointing to an unused method. What
RVE/three-level *would* add beyond what's been done is **meta-regression
coefficients with correct standard errors** when testing paper-level
moderators (country, kit, etc.) against the *pooled* set of ~1,367
observations rather than per-edge -- but per-edge is what a reviewer would
want to see for any specific contested pair, and the corpus-level model
cannot produce that.

## 6. Microbiome-specific reproducibility literature

This is the most directly on-point body of work, because it is the same
outcome variable (does a taxon go up or down across independent case-control
studies of the same disease) at comparable per-disease study counts.

**Duvallet, C., Gibbons, S.M., Kearney, T., Ai, S., & Alm, E.J. (2017),
"Meta-analysis of gut microbiome studies identifies disease-specific and
shared responses," *Nature Communications* 8:1784**
(doi:10.1038/s41467-017-01973-8; PMC5716994). This paper is close to a
best-case comparator for us:

- They built **MicrobiomeHD**, 28 case-control 16S studies across 10
  diseases, and **their consensus rule for calling a genus "consistently
  associated" with a disease is explicitly vote counting**: "a genus was
  considered consistently associated with a disease if it was significantly
  associated (q<0.05) with the disease **in the same direction in at least
  two studies**." This is the same construct our graph uses, in a paper the
  field treats as a landmark -- i.e., **vote counting is what the microbiome
  field actually does in practice**, despite the general meta-analysis
  literature's objections in Section 1. That tension is worth naming plainly
  rather than resolving in either direction.
- Their own per-disease study counts are directly comparable to our per-edge
  counts: **4 studies for CRC, 4 for IBD, 5 for obesity, 3 for HIV, 2 for
  T1D, 2 for autism, 1 each for arthritis and Parkinson's.** They explicitly
  state: **"we did not find consistent bacterial associations for conditions
  with fewer than four data sets."** That is the field's own empirical floor
  for when vote-counted directional consensus is worth reporting at all,
  and it sits right at the boundary of our 43 dual-sided edges (most of which
  have only 2-4 papers *per side*, i.e., fewer total studies than Duvallet's
  stated floor once split by direction).
- Roughly **half of genus-level disease associations across their 10 diseases
  recur in >1 disease**, which they interpret as a shared, non-specific
  host-response signature rather than disease-specific biology -- a
  candidate (unproven) explanation for why unrelated taxon-disease edges in
  our own graph might show correlated instability, worth flagging as a
  hypothesis, not importing as a finding.

**Sinha, R. et al. (2017), "Assessment of variation in microbial community
amplicon sequencing by the Microbiome Quality Control (MBQC) project
consortium," *Nature Biotechnology*** (doi:10.1038/nbt.3981). 15 labs, 9
bioinformatics pipelines, blinded specimens. Their ranked variance
contributors: **biospecimen type/origin first, then DNA extraction protocol,
then sample handling environment, then bioinformatics pipeline** -- i.e.,
extraction-kit and pipeline choice are confirmed, measured, substantial
sources of variance in this literature, which is the direct mechanistic
backing for why our null result on "extraction kit" as a discordance
moderator (`FINDINGS_paper_discordance.md`) is a power failure rather than
evidence the kit doesn't matter.

**Gibbons, S.M. et al. (2018), "Correcting for batch effects in case-control
microbiome studies," *PLOS Computational Biology***
(doi:10.1371/journal.pcbi.1006102; PMC5940237) makes the sharpest point for
our purposes: comparing **healthy control** cohorts from two different
published studies (Baxter et al. and Zeller et al.) with no correction,
**681 of 1,021 OTUs (67%) differed "significantly" between two groups that
are both just healthy controls** -- purely from batch/study effects, with no
disease involved at all. Their explicit diagnosis: **"batch effects are often
diffuse and conflated with biological signals"** in microbiome data
specifically (unlike, say, RNA-seq, where ComBat/limma-style correction
transfers more cleanly), because of the compositional, zero-inflated,
low-signal nature of the data. This is a load-bearing number for our writeup:
it is a real, measured demonstration that **cross-study disagreement at the
magnitude we see (27.6% discordance) is entirely consistent with batch/technical
variance alone, with zero true biological disagreement required** -- it
doesn't prove our disagreement *is* batch effect, but it establishes that
"we can't tell, and the base rate of pure-batch discordance in this exact
data type is high" is the field's own conclusion, not a weakness particular
to our extraction pipeline.

## What is achievable at our sample size, and what would need effect sizes

**Achievable with direction-only data at 43 edges / 2-4 papers per side:**

- Reporting the pooled/corpus-wide discordance rate (27.6%, already done) and
  a formal three-level or bootstrap variance decomposition of it into
  within- vs between-paper components (already done informally; a three-level
  model would formalize it, not change the number materially).
- A binomial sign test **per edge**, reported honestly as underpowered
  (smallest possible p=0.029 pre-correction, non-significant after any
  multiple-comparison correction across 220 edges) -- useful only as a
  disclosure, not as a filter.
- A corpus-wide omnibus test of whether majority-direction calls across all
  220 edges beat chance in aggregate (a single pooled binomial/sign statistic),
  which answers "is vote counting doing anything at all on average" but not
  "is edge X reliable."
- Comparison to the Duvallet floor: flagging which of our 43 dual-sided edges
  fall at or above their empirical "4 studies minimum" threshold (few will,
  once split by direction) as a rough credibility tier, explicitly borrowed
  from a published precedent rather than invented for this project.
- Naming batch effects (Sinha/MBQC, Gibbons) as a documented, quantified,
  literature-established alternative explanation for discordance that is
  indistinguishable from true biological heterogeneity in our data, because
  we don't have the metadata (or a large enough n) to correct for it the way
  Gibbons et al. do.

**Not achievable without effect sizes, regardless of how we reanalyze what we
have:**

- Cochran's Q, tau-squared, I-squared, or any pooled random/fixed-effect
  summary estimate -- these are mechanically undefined without a per-study
  effect magnitude and variance.
- Meta-regression on any study-level moderator per edge -- the field's own
  ~10-studies-per-covariate floor is 2-5x our per-edge n.
- Any credible-interval or confidence-interval statement on a "true" direction
  for a given contested edge -- direction counts alone don't carry the
  sampling-variance information needed to construct one.
- Distinguishing "this edge is contested because the biology is genuinely
  context-dependent" from "this edge is contested because of batch effects
  between the labs that produced the papers" -- Gibbons et al. show these are
  observationally similar in exactly this data type, and separating them
  needs either raw sequence data reprocessed through one pipeline (their
  approach, and Duvallet's) or extracted effect sizes with enough studies per
  edge to run a real moderator analysis. Neither is available to us short of
  a substantially larger, differently-scoped extraction pass.

**Recommendation implied by the above, stated plainly:** the honest
conclusion is not "run a better statistical test on the direction counts we
have" -- the literature says no such test exists that would produce a
different, more defensible answer than what vote counting already gives at
this n. The two paths that would actually move this forward are (a) extract
effect sizes (or at least comparable p-values) per taxon per paper during a
future extraction pass, which would unlock Stouffer/Fisher combination and
real Q/I-squared statistics, or (b) treat the existing corpus-wide
variance-decomposition work as the ceiling of what direction-only data can
support, and report per-edge contested status as a qualitative flag (as the
project already does) rather than attempting to attach a per-edge confidence
score that the sample size cannot support.
