# The external validation is only half independent — and the two halves measure different things

**Session of 2026-09-09. Scripts: `check_independence.py`, `calibrate_agreement.py`,
`calibrate_confounds.py`. Data: `independence.json`, `calibration.json`,
`calibration_confounds.json`.**

## The claim being checked

Everything this project publishes leans on two numbers: agreement with **Disbiome
(73.0%)** and **Peryton (72.5%)**. They carry the weight they do *because those
curations are independent of this pipeline* — the repo says so explicitly, and
prefers them to the in-house gold standard for exactly that reason.

That independence has never been checked. It is an assumption, and it is the
load-bearing one.

Both databases curate the primary literature. So do we. If Disbiome's entry for
*Roseburia* / Parkinson's was read out of the **same paper our extractor read**,
then "agreement" measures whether two readers read one sentence the same way.
That is a fair and useful test of the extractor. It is not the independent
replication the number is presented as.

## It is measurable

Our extraction rows carry PubMed links; Disbiome ships a publications table with
PubMed urls, DOIs and titles; Peryton ships PMID, DOI and title columns. So the
overlap can simply be looked up.

Matching is on **PMID *or* DOI *or* normalised title**, a match on any one
counting. PMID alone covers only 189 of our 272 papers (69%), and under-matching
is the dangerous direction here: it would file genuinely shared papers as
"disjoint" and flatter the result. Widening the match moved Disbiome's shared set
36 → 43 papers and made the effect below *larger*, as predicted.

Match provenance, for audit: Disbiome 36 by PMID, 1 by DOI, 6 by title only;
Peryton 20 by PMID, 3 by DOI, 1 by title only. The title-only matches are long,
distinctive titles ("Gut inflammation and dysbiosis in human motor neuron
disease."), not generic strings.

## A small overlap that carries half the evidence

| | Disbiome | Peryton |
|---|---|---|
| our corpus papers they also cite | **43 / 272 (15.8%)** | **24 / 272 (8.8%)** |
| decisive pairs resting on ≥1 shared paper | **88 / 174 (50.6%)** | **62 / 138 (44.9%)** |

A 9–16% overlap in *papers* produces a ~50% overlap in *pairs*, because the
shared papers are the heavily-reported ones. This is the whole problem in one
line: the validation set is dominated by the small slice of literature both sides
read.

## Agreement splits sharply on that line

| | shared source | disjoint source | difference |
|---|---|---|---|
| Disbiome | **87.5%** (n=88) | **58.1%** (n=86) | +29.4 pts, p=0.0001, 95% CI [+17.4, +41.1] |
| Peryton | **96.8%** (n=62) | **52.6%** (n=76) | +44.1 pts, p=0.0001, 95% CI [+32.1, +56.2] |

p-values are taxon-block permutations (labels shuffled in whole taxon blocks, so
within-taxon correlation is preserved); CIs are taxon cluster bootstraps. Both
guards agree, in both databases.

**It is not the evidence-count signal in disguise.** Stratified on our own
evidence, the split survives (Disbiome within-1-paper +37.8, p=0.0002; Peryton
+55.8, p=0.0001).

## Disease is a real confounder, and pooling overstates the effect

The buckets are **not comparable across diseases**. Every ALS pair here is
shared-source (both curations cite the ALS papers we used); every autism and
epilepsy pair is disjoint. So the pooled difference partly measures "ALS versus
autism", which is not the question.

Held fixed within a single disease, only Parkinson's has both buckets full — and
there the effect is smaller than pooled, still large, and **replicated across two
databases that arrived at the same disjoint rate without knowing about each
other**:

| Parkinson's disease | shared | disjoint | difference |
|---|---|---|---|
| Disbiome | **100.0%** (n=40) | **59.0%** (n=39) | +41.0 pts, p=0.0001, MDE 17.9 |
| Peryton | **95.8%** (n=48) | **59.6%** (n=47) | +36.3 pts, p=0.0001, MDE 16.9 |

59.0 and 59.6, from two separate curations. That is the number this graph gets
when it is actually tested against literature it did not read.

### The counter-example, logged rather than buried

In **Multiple sclerosis the gap is absent**: shared 72.7% (n=22) vs disjoint
70.6% (n=17), +2.1 pts, p=1.00. The minimum detectable effect there is 28.7
points, so that test **could** have seen a Parkinson's-sized gap and did not.
This is heterogeneity, not lack of power. The finding is established *in
Parkinson's*, not corpus-wide, and MS is evidence against the general version.

## The reframe: 73% is a blend of two different quantities

Crossing source-sharing with evidence count gives the table that actually matters
(cells are agreement, Disbiome / Peryton):

| | 1 paper | ≥2 papers |
|---|---|---|
| **shared source** | 85.2% / 94.1% | 91.2% / 100% |
| **disjoint source** | **47.4% / 38.3%** | **79.3% / 75.9%** |

Read the corners:

- **Top-left is extraction fidelity.** A single-paper edge whose one paper *is*
  the curated source: our reading versus an expert curator's reading of the same
  text. **85–94%.** That is a good result and it is the cleanest measurement of
  the extractor this project has — it does not depend on the in-house gold
  standard, which is under audit and unreliable.
- **Bottom-left is a coin flip.** A single-paper edge tested against different
  literature: **47.4% / 38.3%**.
- **Bottom-right is the real replication rate.** A well-evidenced edge tested
  against disjoint literature: **~76–79%**.

So **73.0% / 72.5% measures neither quantity.** It is a mixture of ~90% reading
fidelity and ~55% cross-literature replication, in a ratio set by how much of our
corpus the curators happened to have read.

## What the disjoint number does and does not mean

It is **not** an extraction accuracy figure, and it should not be reported as
one. Two studies of the same taxon in the same disease genuinely disagree in this
literature — this project has documented that repeatedly (217 contested edges;
~1 taxon in 3 flips sign between cohorts). Disagreement across disjoint sources
is partly real biological and methodological heterogeneity, and 59% may be near
the ceiling the field itself sets.

There is also a selection argument in the other direction: pairs where both sides
cite the same landmark papers may be the *well-established* ones, so "disjoint"
may select for genuinely contested biology. That would not rescue the headline
either — under either reading, 73% is not a measure of independent replication.

The honest split:

- **Extraction fidelity (shared source): 85–97%.** Good news, and better
  evidence for the extractor than anything derived from the in-house gold.
- **Literature reproducibility (disjoint source): ~53–59%.** Sobering, and a
  finding about the field, not only about this pipeline.

## Which edge properties predict agreement

From `calibrate_agreement.py` / `calibrate_confounds.py`, 24 tests,
Benjamini-Hochberg corrected across the family; taxon-block permutation plus
taxon cluster bootstrap throughout.

**Survives (q < 0.05):**

- **Evidence count.** 1-paper edges agree 65.8% / 61.7%; ≥3-paper edges **91.7% /
  90.6%**. Difference +23.8 / +23.7 pts — near-identical in two independent
  databases — both CIs excluding zero, both q < 0.05. Edge weight, the graph's
  primary visual encoding, **is** a calibration signal.
- **Disease specificity.** Taxa the graph classes `discriminating` (purity ≤ 0.6 —
  the taxon points different ways in different diseases) agree far worse: 36.4%
  vs 82.4% for `mixed` (Disbiome, +41.9 pts, q=0.0024). It is not evidence count
  in disguise — within single-paper edges alone it is +45.3 pts (q=0.0038). The
  graph's own internal inconsistency predicts external error. Peryton points the
  same way at the same magnitude (+35.0) but has only 10 such pairs and does not
  survive correction (q=0.055).
- **Species rank.** Species-level edges agree 93.2% vs 66.7% for genus (Disbiome,
  +27.0 pts, q=0.0038). This one **cannot** be an evidence artefact: species
  edges carry *fewer* papers on average (1.61 vs 2.26) and agree *more*. Peryton
  agrees in direction (+20.4) but does not survive correction (q=0.103).

**Nulls, with power:**

- **The reference's own evidence depth does not predict agreement.** The
  hypothesis that the 27% disagreement is mostly thin single-record curated
  entries is **not supported**: Disbiome +6.2 pts, CI [−16.5, +25.4], p=0.49.
  Peryton points positive but does not replicate. A hypothesis this session
  proposed and the data killed.
- **`restates_prior` does not predict agreement** (+0.9 / +7.7 pts, p=1.00 /
  0.48). An edge that restates its taxonomic parent is no less reliable.
- **Within-paper rank conflict**: Disbiome −30.1 pts but p=0.065 on 9 pairs;
  Peryton points the *other* way. Undetermined at n=9.

## Two other things worth having

- **Where both curations concur with each other (79 pairs), we agree with them
  82.3%** — versus 73.0/72.5 against either alone. Concurrence between the two
  references is itself a quality filter on the reference.
- **5 pairs where Disbiome and Peryton flatly contradict each other**, all in
  ALS, each resting on one record per side: *Eubacteriales*, *Lachnospiraceae*,
  *Dorea*, *Anaerostipes*, *Oscillibacter*. These are review targets of the same
  class as the 11 doubly-contradicted pairs, but stronger: two curations reading
  the same small literature and disagreeing.

## What this changes

1. **Report agreement stratified, not pooled.** The single number obscures both
   of the quantities it is made of, and the useful ones are the corners of that
   2×2.
2. **The extractor looks better than the headline suggests**, on evidence that
   does not depend on the in-house gold standard.
3. **The graph's calibration is real and shippable.** Evidence count, taxon
   specificity and rank all predict external agreement, so the viewer can
   honestly tell a reader which edges to trust — a single-paper edge from a
   `discriminating` genus is a coin flip and should say so.
4. **This is not a "structural correction".** The standing rule — do not run
   another correction expecting agreement to move — is untouched. Nothing about
   the graph changed here; the same 174/138 decisive pairs were decomposed.

## Limits

- 40 diseases, but only Parkinson's has enough pairs in both buckets for a
  within-disease test; MS disagrees. Generalising beyond Parkinson's is not
  supported by this data.
- Peryton covers only 3 of our diseases, so its pooled figures are close to a
  Parkinson's/Alzheimer's statement already.
- The join runs on the replay taxonomy cache, not the NCBI taxdump (blocked in
  this environment). It reproduces the shipped 73.0% / 72.5% exactly, so the
  subsetting here is like-for-like; absolute figures measured with a full taxdump
  could shift the overlap slightly.
- Shared/disjoint is per-*pair*, and an edge counts as shared if **any** of its
  papers is cited by the curation. For multi-paper edges the shared bucket
  therefore mixes in independent evidence; the clean same-paper cell is the
  single-paper row.
