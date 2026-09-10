# Discordance is a property of the paper — and study design does not explain it

**Session of 2026-09-10. CPU-only; no `MAIN_DATA.json`, no NCBI taxdump
(`ftp.ncbi.nih.gov` re-probed once, still `CONNECT → 403`, third session running).**

Scripts: `paper_inversion.py`, `paper_inversion_control.py`,
`paper_inversion_decompose.py`, `paper_inversion_power.py`,
`paper_discordance_predictors.py`, `paper_discordance_offset.py`,
`paper_discordance_endogeneity.py`. All outputs verified byte-identical on a
second run.

---

## Why this was worth asking

`FINDINGS_db_conflicts.md` closed last session by showing that all five places
where Disbiome and Peryton contradict each other trace to **one** paper
(PMID 27703453), and that Disbiome has that paper's case/control assignment
swapped — every taxon flipped together, which is the fingerprint of a group-label
error rather than five independent curation mistakes.

That is a failure mode *our* extractor can have too. An LLM reading a paper whose
groups are labelled "Group A / Group B", or whose control column comes second in
the table, can invert the entire paper at once. Nobody had checked. It is
checkable without any new data: the graph carries, per edge, which paper voted
which way.

## Method

For every (taxon, disease) edge with ≥2 contributing papers, each paper's vote is
scored against the **leave-one-out majority** of the other papers on that edge.
Ties are not decisive and are dropped. That gives 440 multi-paper edges, 1,493
observations, **1,367 decisive, of which 377 (27.6%) disagree** with the rest of
the literature.

The null shuffles directions **within each edge**. Every edge keeps its exact
up/down counts and every paper keeps its exact set of edges; the only thing
destroyed is any paper-level consistency in *which* papers hold the minority
direction — precisely the alternative hypothesis. Randomisation is at the paper
level throughout, per the standing rule in this repo.

---

## Result 1 — no paper is inverted. This is a well-powered null.

**Zero of the 134 testable papers survive BH correction** (best q = 0.41; the
leading candidate, "Analysis of the Gut Microflora in Patients With Parkinson's",
is 9 disagreements out of 23 against a null mean of 4.65, p = 0.0053).

The power is measured, not asserted. Flipping every observation of each eligible
paper in turn and re-running its test:

| a fully inverted copy would reach | papers |
|---|---|
| raw p ≤ 0.05 | **100 / 134** |
| Bonferroni-strict (p ≤ 0.00037) | **81 / 134** |

So for roughly three-quarters of the testable corpus, a whole-paper inversion
would have been caught, and none was. **The extractor does not have the failure
mode Disbiome has on PMID 27703453.** This is the strongest statement about
extraction fidelity in the repo that needs neither the in-house gold standard
(under audit) nor an external curation (now known to be only half independent).

## Result 2 — but discordance *is* a paper-level property

Although no single paper is an outlier, minority-direction status is clustered by
paper far beyond what the within-edge null allows. Dispersion statistic
Σ(observed − expected)²/expected over papers:

| set | decisive obs | observed | null mean | p |
|---|---|---|---|---|
| full | 1,367 | 164.2 | 132.7 | **0.0003** |
| containment-controlled | 975 | 125.7 | 102.7 | **0.0013** |
| ≥3 papers per edge | 907 | 121.8 | 87.3 | **0.0003** |
| both controls at once | 637 | 70.5 | 58.8 | 0.058 |

Two artifacts were ruled out individually:

- **Within-paper taxonomic correlation.** This corpus is known to carry it
  (related taxa agree 89% within a paper vs 54% for unrelated), and the
  independent per-edge shuffle cannot reproduce it, so it would manufacture
  exactly this excess. Dropping every within-paper relative across the 723
  containment links — 28.4% of observations — leaves p = 0.0013.
- **Two-paper edges are structurally forced.** On a 1e/1d edge *both* papers are
  scored as disagreeing under every permutation. Restricting to edges with ≥3
  papers removes those contributions entirely and leaves p = 0.0003.

**Applying both at once gives p = 0.058, and that is a power limit, not a
refutation.** Subsampling the containment-controlled set down to the strict set's
637 decisive observations, 200 times, the test reaches p < 0.05 only **58%** of
the time, with median p = 0.035 (46% and median 0.063 for the
excess-disagreement half). An observed 0.058 sits squarely inside what an
underpowered look at a real effect produces. Both tails move: some papers disagree
with consensus far more than chance, some far less.

---

## Result 3 — three "explanations" that were edge depth in disguise

The obvious next question is what makes a paper discordant, and it reframes a
question this project has failed to answer four times. Study design (FDR 0.243),
body site (p = 0.120), ASD (p = 0.211) and taxon co-occurrence (p = 0.14) were all
tested **edge**-level — "do a contested edge's up-papers differ from its
down-papers" — across 130 contested edges averaging ~5 papers. The unit here is
the paper, and every paper contributes: 112 testable, 89 carrying extracted study
metadata.

Tested naively, on each paper's raw disagreement rate, **three predictors survive
BH**:

| predictor | rate | vs | p | q |
|---|---|---|---|---|
| disease is Parkinson's | 0.154 (n=38) | 0.340 (n=74) | 0.0001 | **0.0005** |
| n_cases above median | 0.209 (n=45) | 0.388 (n=42) | 0.0002 | **0.0010** |
| cohort total above median | 0.223 (n=44) | 0.368 (n=43) | 0.0013 | **0.0043** |

**All three are artifacts, and the mechanism is arithmetic.** The disagreement
rate is not comparable across edges of different depth: a 2-paper contested edge
scores *both* its papers as disagreeing (rate 1.00), while a 10-paper 8/2 edge
scores only the two in the minority (rate 0.20). Parkinson's is the most-reported
disease in this corpus and large cohorts study well-studied diseases, so all three
are the same confound wearing three hats.

The fix is not another covariate. The within-edge null defines each paper's
expected disagreements **exactly**, in closed form — for an edge with n_e up and
n_d down out of n,

```
P(decisive) = 1 if |n_e − n_d| ≠ 1 else min(n_e, n_d)/n
P(disagree) = 1 if n_e == n_d      else min(n_e, n_d)/n
```

— which absorbs edge depth, contestedness and the tie structure at once. The
closed form was checked against 4,000 simulated permutations before use (max
absolute error 0.052, consistent with Monte Carlo noise), and globally it
calibrates: 214 observed disagreements against 215.0 expected, **O/E = 0.995**.

Re-running every predictor as a difference in observed/expected ratio:

| predictor | O/E | vs | diff | p | q |
|---|---|---|---|---|---|
| diet controlled | 0.908 (n=28) | 1.081 (n=59) | −0.173 | 0.062 | 0.234 |
| n_cases above median | 0.950 (n=44) | 1.105 (n=41) | −0.156 | 0.078 | 0.234 |
| 16S vs shotgun | 1.068 (n=69) | 0.880 (n=17) | +0.189 | 0.085 | 0.234 |
| n_controls above median | 0.963 (n=42) | 1.100 (n=41) | −0.137 | 0.128 | 0.281 |
| cohort total above median | 0.971 (n=43) | 1.091 (n=42) | −0.119 | 0.181 | 0.332 |
| country = China | 1.097 (n=27) | 0.998 (n=60) | +0.099 | 0.291 | 0.457 |
| disease is Parkinson's | 0.935 (n=36) | 1.011 (n=73) | −0.075 | 0.362 | 0.498 |
| medication controlled | 1.017 (n=71) | 1.105 (n=16) | −0.088 | 0.438 | 0.535 |
| 16S region V3-V4 | 1.045 (n=37) | 0.983 (n=21) | +0.062 | 0.569 | 0.626 |
| *(control)* sits on deep edges | 0.990 (n=56) | 0.999 (n=53) | −0.009 | **0.904** | 0.904 |

**Every study-design variable is null**, best q = 0.234, with minimum detectable
differences of ±0.16 to ±0.22 in O/E — i.e. these tests could have seen a 16–22%
relative shift in discordance and did not. The planted control confirms the offset
does its job: edge depth, which drove all three false positives, is now flat at
p = 0.90 with the tightest MDE of the set (±0.15).

### The one survivor is disqualified on construction

`sits_on_contested_edges` survives BH (O/E 1.056 vs 0.830, p = 0.0029, q = 0.032)
and is the result most likely to be written up. It should not be. It is
endogenous — an edge is contested *because* the papers on it disagreed, so a
discordant paper manufactures the contested edges it is then observed to sit on.

The check is decisive: recompute each edge's contestedness **leave-one-out**, from
the other papers only, so a paper's own vote cannot inflate its own predictor.

| construction | O/E high vs low | diff | p |
|---|---|---|---|
| self-inclusive | 1.056 vs 0.830 | **+0.225** | 0.0031 |
| leave-one-out | 0.935 vs 1.090 | **−0.154** | 0.0428 |
| leave-one-out, ties dropped | 0.714 vs 1.446 | **−0.733** | 0.0004 |

**The effect reverses sign.** That is the signature of the outcome driving the
predictor, and neither direction is interpretable: the self-inclusive version
counts the paper's own dissent as evidence that its ground was disputed, while the
leave-one-out version is coupled the opposite way (a paper that disagrees makes
the *others* look more unanimous by comparison). Recorded here so the next session
does not rediscover it as a finding. Contestedness cannot serve as a predictor of
discordance in either construction.

---

## What this adds up to

1. **The extractor is not inverting papers.** A well-powered null (81/134 would
   have been caught), and it is independent of both compromised references.
2. **Consensus-disagreement is real, paper-level variance** — the first structure
   anything in this project has found in contested edges after four edge-level
   nulls.
3. **No measured study-design variable explains it.** Country, cohort size,
   sequencing platform, 16S region, medication and diet control, and disease
   identity are all null at MDEs of 16–22%.

Point 3 is the fifth documented false positive this corpus has produced and the
first caught by an exact offset rather than a permutation — the raw-rate analysis
would have shipped "Parkinson's papers are more reliable" at q = 0.0005. Worth
noting that the previous four nulls were all edge-level; this one is not weaker
for being paper-level, it is the same answer at a better-conditioned denominator.

## What would move it

The variance is real and unexplained by anything currently extracted, which makes
it a **measurement** problem rather than an analysis one. Two levers, in order:

- **Extract variables we do not currently have.** `metadata.jsonl` carries country,
  cohort size, sequencing, body site, 16S region, medication and diet. It does not
  carry the things most likely to drive a whole-paper direction offset: DNA
  extraction kit, primer set beyond the region label, bioinformatics pipeline
  (QIIME/DADA2/mothur), OTU-vs-ASV, rarefaction depth, differential-abundance
  method (LEfSe vs DESeq2 vs Wilcoxon), and whether abundance is relative or
  absolute. Batch effects of exactly this kind are the standard explanation in the
  microbiome methods literature for cohorts disagreeing. This is a second metadata
  pass over papers already in hand — **CPU-cheap if done by prompt, and the single
  highest-value unblocked item.**
- **More papers.** 109 papers with ≥4 decisive observations is what sets the
  ±0.16–0.22 MDE. Needs a GPU; ask before spending.
