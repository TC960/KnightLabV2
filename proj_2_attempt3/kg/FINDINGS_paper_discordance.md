# No paper is inverted; discordance is paper-level, small, and unexplainable at this n

**Session of 2026-09-10. CPU-only; no `MAIN_DATA.json`, no NCBI taxdump
(`ftp.ncbi.nih.gov` re-probed once, still `CONNECT → 403`, third session running).**

Scripts: `paper_inversion.py`, `paper_inversion_control.py`,
`paper_inversion_decompose.py`, `paper_inversion_power.py`,
`paper_discordance_predictors.py`, `paper_discordance_offset.py`,
`paper_discordance_endogeneity.py`, `methods_metadata.py`,
`methods_discordance.py`, `paper_effect_size.py`. All outputs verified
byte-identical on a second run.

**Read Result 4 before quoting Result 2.** The paper-level effect is statistically
solid and practically small — 3.4 percentage points of discordance, 95% CI
[0.0, 6.0] — and that magnitude is what makes the twenty-four accompanying nulls
uninformative rather than reassuring.

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

---

## Result 4 — the effect is real and small, and that retires the whole line of inquiry

`paper_discordance_offset.py` tested nine study-design variables and found nothing.
The obvious reply is that we were testing the wrong variables: the microbiome
methods literature blames DNA extraction kit, primer set, pipeline, OTU-vs-ASV and
the differential-abundance test for cohorts disagreeing, and `metadata.jsonl`
carries none of them. Every recent session filed that under "needs a GPU".

**It did not.** Full text for **all 272 contributing papers** was already in the
repo, split across `all_usable_papers.json` (250), `extract_input.json` (98) and
`new_papers.json` (53); the union covers 272/272. And the variables of interest
are tool names — literal strings a regex reads deterministically, giving the same
answer twice, which an LLM pass would not. `methods_metadata.py` does it.

**Detector validated before use, two ways.** Against the existing LLM-extracted
labels: 16S-vs-shotgun agreement 74.8% (n=202), 16S region 80.8% (n=125). Against
an independent read of 12 sampled methods sections: overall **recall 0.90,
precision 0.77**, and for the specific named tools the analysis keys on
(LEfSe, DESeq2, ANCOM, ALDEx2, edgeR, metagenomeSeq) **recall 1.00, precision
0.83**. The false positives concentrate in PERMANOVA (5) and Wilcoxon (3) — tests
named in the methods for beta-diversity or baseline comparisons, which the reader
excluded as "not the differential-abundance test". That is a definitional
difference, not a detection failure. The weak family is `normalisation`
(recall 0.57), so the `rarefied` and `absolute_or_CLR` nulls are attenuated and
should be read as such. Publication year, parsed from the article header, is exact
for 21 of the 22 papers carrying an explicit year field.

Fifteen predictors, plus two planted controls, on the same offset:

| predictor | O/E | vs | diff | p | q |
|---|---|---|---|---|---|
| kit uses bead-beating | 0.894 (n=29) | 1.064 (n=36) | −0.171 | 0.042 | 0.606 |
| uses DESeq2 / ANCOM / ALDEx2 | 0.860 (n=22) | 0.987 (n=77) | −0.127 | 0.179 | 0.606 |
| platform MiSeq | 0.952 (n=55) | 1.076 (n=26) | −0.124 | 0.190 | 0.606 |
| uses LEfSe | 1.003 (n=54) | 0.912 (n=45) | +0.091 | 0.245 | 0.606 |
| rarefied | 1.029 (n=14) | 0.887 (n=50) | +0.142 | 0.247 | 0.606 |
| kit QIAamp vs other | 1.042 (n=23) | 0.944 (n=42) | +0.099 | 0.259 | 0.606 |
| absolute quantification or CLR | 0.831 (n=15) | 0.961 (n=49) | −0.129 | 0.281 | 0.606 |
| pipeline QIIME2 / DADA2 | 1.056 (n=35) | 0.962 (n=37) | +0.094 | 0.285 | 0.606 |
| published recently | 1.048 (n=61) | 0.987 (n=31) | +0.062 | 0.477 | 0.901 |
| feature ASV vs OTU | 1.023 (n=23) | 0.976 (n=53) | +0.047 | 0.615 | 0.937 |
| reports multiple-testing correction | 0.986 (n=59) | 1.006 (n=50) | −0.020 | 0.793 | 0.937 |
| cohort imbalanced ≥1.5× | 1.021 (n=35) | 1.038 (n=48) | −0.017 | 0.857 | 0.937 |
| pipeline legacy (QIIME1/mothur/UPARSE) | 1.015 (n=35) | 0.999 (n=37) | +0.016 | 0.859 | 0.937 |
| reports many taxa | 0.998 (n=63) | 0.987 (n=46) | +0.012 | 0.884 | 0.937 |
| nonparametric tests only | 0.958 (n=22) | 0.969 (n=77) | −0.011 | 0.907 | 0.937 |
| *(control)* found methods section | 0.997 (n=71) | 0.991 (n=38) | +0.006 | 0.937 | 0.937 |
| *(control)* sits on deep edges | 0.990 (n=56) | 0.999 (n=53) | −0.009 | 0.902 | 0.937 |

**No survivors.** Both planted controls behave. That is **24 variables tested
against paper discordance across two passes, and 24 nulls.**

### Why they were all going to be null

Rather than test a twenty-fifth, put a magnitude on the thing being explained.
Modelling each paper as carrying a multiplicative discordance propensity r_p with
mean 1 and SD σ, and solving Var(dis_p) = Var_null(dis_p) + (E_p·σ)² by moments:

| quantity | value |
|---|---|
| observed Σ(observed − expected)² | 66.6 |
| expected under the within-edge null | 56.9 |
| **excess** | **9.7 (17%)** |
| σ (SD of the propensity multiplier) | **0.123** |
| paper-level SD of discordance | **3.4 percentage points** on a 27.6% base |
| cluster bootstrap over papers, 95% CI | **[0.0, 6.0] points** — includes zero |

A paper one SD above the mean disagrees 31.0% of the time; one SD below, 24.2%.
15% of bootstrap resamples show no excess variance at all.

**This is the result that matters, and it cuts against the earlier framing in this
document.** The permutation test and the magnitude estimate are answering different
questions and both are right: the permutation conditions on the actual papers and
asks whether the minority-direction labels are exchangeable across them (they are
not, p = 0.0003), while the moment estimator asks how large the resulting spread
is, and that is small and unstably estimated.

Two consequences:

1. **The 24 nulls were foreordained.** The tests have MDEs of ±0.15 to ±0.24 in
   O/E — roughly ±4 to ±7 percentage points of discordance — against a total
   paper-level spread of ±3.4 points. No subdivision of a 3.4-point effect was ever
   going to clear a 4-to-7-point threshold. The honest conclusion is not "kit and
   pipeline do not matter" but **"this corpus cannot answer that question, and
   could not have"**.
2. **Roughly 83% of the variance in disagreement is edge structure, not paper
   identity.** Papers are close to interchangeable; the disagreement lives in the
   taxon–disease pairs themselves. That is quantitative support for the standing
   design decision — contested edges are kept and never averaged because
   disagreement is a finding about the evidence base — and it argues against
   spending further effort on paper-level covariates of any kind.

## What this adds up to

1. **The extractor is not inverting papers.** A well-powered null — 0 of 134
   survive BH and a fully inverted copy would have been caught for 81 of them —
   and it depends on neither the in-house gold (under audit) nor the external
   curations (only half independent). This is the most useful thing in this
   document.
2. **Consensus-disagreement is genuinely paper-level, and genuinely small.**
   Minority-direction labels are not exchangeable across papers (p = 0.0003), but
   the spread is only **3.4 percentage points** on a 27.6% base, with a cluster
   bootstrap CI of [0.0, 6.0] that includes zero. Both statements are true and
   they are not in tension: the permutation test detects the non-exchangeability,
   the moment estimator sizes it.
3. **Twenty-four variables, twenty-four nulls — and the nulls were foreordained.**
   Nine study-design plus fifteen wet-lab/bioinformatics, at MDEs of ±4 to ±7
   percentage points of discordance against a total paper-level spread of ±3.4.
   The defensible conclusion is that **this corpus cannot answer whether extraction
   kit or pipeline drives disagreement**, not that they do not.
4. **About 83% of the variance in disagreement is edge structure, not paper
   identity.** Papers are close to interchangeable. Disagreement lives in the
   taxon–disease pairs themselves.

Point 3 also produced the fifth documented false positive in this corpus, and the
first caught by an exact offset rather than a permutation: on raw disagreement
rates the analysis would have shipped "Parkinson's papers are more reliable" at
q = 0.0005, along with two cohort-size effects, all three of them edge depth in
disguise.

## What would move it — and what would not

**Not more covariates.** That lever is now measured and it is too short. A
paper-level spread of 3.4 points cannot be subdivided by a corpus that resolves
4-to-7-point differences, and adding a twenty-fifth variable changes nothing about
that arithmetic. The second metadata pass was the item every recent session called
the highest-value unblocked step; it has now been done — CPU-only, no GPU, from
full text that was in the repo all along — and it is null. Recording that as a
closed lever is the point of writing it down.

**Not paper-level modelling at all**, for the same reason. If 83% of the variance
is edge structure, the remaining question is about taxon–disease pairs, not about
studies.

**More papers, and only more papers.** 109 papers with ≥4 decisive observations is
what sets every MDE here. This is the same conclusion the last three sessions
reached from different directions, now with the alternative explicitly closed off
rather than left open. Extraction needs a GPU — **ask before spending.**

**One thing worth doing cheaply first:** `methods_metadata.py` extracts nine
families of study-methods variables for all 272 papers at 0.90 recall, and nothing
in the graph consumes them. They are weak predictors of *discordance*, which is
what was tested, but they are perfectly good **provenance** for a reader deciding
whether to trust an edge — "these 6 papers all used the same kit and pipeline" is
information a biologist wants, and it is already computed.
