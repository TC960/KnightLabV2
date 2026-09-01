# Independent validation — embedding / contrast analysis

Adversarial re-derivation of the numeric claims in `proj_2_attempt3/kg`, run 2026-09-01.
Everything below was recomputed with **my own code** (scratchpad, outside the repo) reading
only `chunks.jsonl`, `chunk_vecs.npy`, `disease_basis.npy`, `graph.json`, `probe_bank.json`,
`audit_variable_coverage.json` and the datasheet CSV. Where an existing helper was under test
it was imported and compared against `sklearn` / `scipy`, never trusted. CPU only, no GPU.
No repo file other than this report was modified.

**Bottom line.** The index is sound and the arithmetic is sound — 18 of the 22 claims are
correct as stated. But there is **one real bug** (the disease subspace is not the subspace the
code says it is), **one reproducibility failure** (the shipped `contrast_experiment.json` is not
the run being quoted), and **two overstatements** (the minimum-detectable-effect figure, and the
"disease can't explain this" guarantee). None of them overturn the headline null result; the bug
makes the deconfounding *weaker* than advertised, and fixing it improves both of its own tests.

---

## A. Index integrity

| # | claim | reported | my value | verdict |
|---|---|---|---|---|
| 1 | rows / vector shape | 20,905 and (20905, 384) | 20,905 rows; (20905, 384) float32 | **CONFIRMED** |
| 2 | all vectors unit-norm | unit | max abs deviation 1.76e-07 (raw), 8.65e-09 (deconfounded); 0 rows off by >1e-5 | **CONFIRMED** |
| 3 | row *i* of jsonl == row *i* of npy | aligned | 24/24 re-embedded chunks give cosine **1.000000** against the stored vector at the same index | **CONFIRMED** |

Claim 3 is the one that mattered, so I tested it three ways rather than one. Random sample of 24
chunks (seed 12345), re-embedded from scratch with `all-MiniLM-L6-v2`:

- cosine at the **same index**: min 1.0000000, mean 1.0000001 — all 24;
- cosine against the **whole 20,905-row matrix**, taking the argmax: the argmax is the chunk's
  own index for **24 of 24**, zero mismatches;
- deliberate off-by-one probe: mean cosine at shift −1 is 0.687, at +1 is 0.641, at ±2 is
  0.58/0.49. So a one-row misalignment would have been unmissable.

The earlier interrupted-build misalignment is **not** present. Two extra consistency checks
passed as well: `disease_basis.npy` is orthonormal to 6.4e-09, and `chunk_vecs_deconfounded.npy`
is exactly reproducible as `normalize(V − (V·B)Bᵀ)` (max abs difference 1.3e-08), so the three
arrays are mutually consistent.

## B. Contested-edge counts

| # | claim | reported | my value | verdict |
|---|---|---|---|---|
| 4 | contested edges | 226 | 226 (of 1,927 total) | **CONFIRMED** |
| 5 | contested observations / papers | 957 from 204 | 957 from 204 (442 `e`, 515 `d`) | **CONFIRMED** |
| 6 | papers reporting both directions | 132 of 204 (65%) | 132 of 204 = **64.7%** | **CONFIRMED** |
| 7 | floor / ceiling | 0.538 / 0.706 | 0.5381 / 0.7064 (band 0.1682) | **CONFIRMED** |
| 8 | edges with ≥2 per side / pairs | 43 / 509 | 43 edges, 509 comparable pairs | **CONFIRMED** |

**Claim 8's reconciliation is correct, and I can account for the gap exactly.** 226 contested
edges → applying only the ≥2-per-side filter leaves **47 edges / 592 pairs** → additionally
dropping papers absent from the chunk index leaves **43 edges / 509 pairs**. So the 226-vs-43
difference is precisely the two filters named, nothing else.

**Papers dropped for not being in the chunk index: 18 of the 204 contributing papers** (186
usable). They are real papers with real titles — mostly MS, ALS and nasal/oral-site PD studies
(e.g. *"Alterations of the human gut microbiome in multiple sclerosis"*, *"Gut microbiome
alterations in preclinical Alzheimer's disease"*). Full list available on request; the loss is
~9% of contributing papers and is not random with respect to disease, which is worth knowing
but does not invalidate anything.

Note: `contrast_experiment.py`'s docstring says *"47 of them together may not be"* — that is the
pre-index-filter count. The run uses 43. Cosmetic, but the docstring and the code disagree.

## C. Statistical machinery

| # | claim | reported | my value | verdict |
|---|---|---|---|---|
| 9 | `auc_and_p()` == `roc_auc_score` | matches | max abs diff **1.11e-16** over 500 randomised trials | **CONFIRMED** |
| 10 | `bh()` is correct Benjamini–Hochberg | correct | **exact** (0.0) vs from-definition BH; 2.2e-16 vs `scipy.stats.false_discovery_control` | **CONFIRMED** |
| 11 | `auc(up, dn)` = P(up>dn), ties 0.5 | correct | max abs diff **1.11e-16** vs sklearn over 500 trials | **CONFIRMED** |
| 12 | permutation shuffles within edge, preserves counts | as described | code read + verified empirically | **CONFIRMED** |

Claim 9 was stressed on the tie cases specifically: 500 trials across four regimes including
integer scores drawn from `{0,1,2}` (brutal ties) and vectors where **every score is identical**.
It agrees with sklearn to machine precision in all of them.

`statsmodels` is not installed in this environment, so claim 10 was checked against two other
references instead: a from-definition BH (sort, `p·n/rank`, reverse cumulative-min, clip at 1)
and `scipy.stats.false_discovery_control(method='bh')`. Agreement is exact and 2.2e-16
respectively, including on p-vectors with heavy ties. As a third check, the `q` column in the
shipped `contrast_experiment.json` is reproduced exactly from its own `p` column.

Claim 12 — the code pools each edge's `up + dn` paper indices, permutes, and re-splits at
`len(up)`, so per-edge n_up/n_dn and paper composition are preserved. There is a latent hazard
in `u + d`: if a paper appeared on **both** sides of one edge it would enter the pool twice and
could be assigned to both piles. I checked — **zero** contested edges have a paper on both sides,
so the hazard never fires. Worth an assert, not a bug today.

## D. Headline results

| # | claim | reported | my value | verdict |
|---|---|---|---|---|
| 13a | cohort_longitudinal effect | +0.1031 | **+0.1031** | **CONFIRMED** |
| 13b | stats_correction effect | −0.0992 | **−0.0992** | **CONFIRMED** |
| 13c | p-values | 0.0115 / 0.0149 | 0.0123 / 0.0174 | CONFIRMED (Monte-Carlo noise) |
| 13d | q-values | 0.253 / 0.253 | **0.296 / 0.296** | discrepancy — see below |
| 13e | 0 of 34 survive q<0.05 | 0 of 34 | 0 of 34 | **CONFIRMED** |
| 13f | these numbers are in `contrast_experiment.json` | implied | **the file holds the RAW run** | **WRONG** |
| 14 | min detectable effect ≈ 0.080 = 1.96 × 0.0408 | 0.080 | 1.96 × 0.0411 = **0.0805** | CONFIRMED as arithmetic; **label is wrong** (G4) |
| 15a | raw space: nc_probiotic_intervention 5th | 5th | **5th** | **CONFIRMED** |
| 15b | deconfounded: negative controls 22nd and 27th of 34 | 22 / 27 | **23 / 28** — exact ties | CONFIRMED in substance |

The effect sizes reproduce **exactly** — my implementation recomputes the disease projection
itself, re-encodes the probe bank, and uses `sklearn.roc_auc_score` rather than the project's AUC
helper, and still lands on +0.1031 and −0.0992 to four decimals. The vectorised statistic was
cross-checked against a sklearn-in-a-loop version (max abs diff 2.1e-17). Raw-space effects
reproduce the shipped JSON exactly too (−0.09136, +0.08153, −0.06582, +0.05992, +0.05599, …).

**13f is the reproducibility failure.** `contrast_experiment.json` on disk reads `"space": "raw"`
and reports stats_correction −0.0914 / p 0.027 / q 0.776 and cohort_longitudinal +0.0815 / p 0.047
/ q 0.776. Those are the `--raw-space` sanity-check numbers, not the headline. Anyone reading the
artifact gets different numbers from the ones being quoted. Cause: `--raw-space` writes to the
same `OUT` path, so running the sanity check destroys the primary result. The deconfounded run
(mtime 00:50 for the script, 01:01 for the JSON) was overwritten. **Fix: give the raw run its own
output filename.**

**13d.** My q-values are 0.296, not 0.253. The BH machinery is correct (claim 10) — the gap is
entirely inherited from the rank-2 Monte-Carlo p (0.0174 vs 0.0149; q = p × 34/2). The p
difference is ~2 MC standard errors at 10,000 draws, so it is seed noise rather than an error,
but it means **0.253 is not a reproducible number** and should be quoted as "q ≈ 0.25–0.30, far
above 0.05" or the seed should be pinned and stated.

**15b.** In the deconfounded space I get nc_probiotic_intervention at rank **23** and
nc_animal_model at rank **28**. Both are *exact ties in |effect|* with their neighbour
(0.0147 with `stats_confounder_adjusted`; 0.0069 with `seq_16s_v34`), so the tie-break is
arbitrary argsort order. The statistic is quantised in units of 0.5/509 = 0.00098 and **11 of 34
probes share a |effect| value with another probe**, so ranks in the lower half are not
well-defined at all. The substantive claim — negative controls sink to the bottom third under
deconfounding and one rises to 5th in raw space — is confirmed; the specific ordinals are not
meaningful and should not be quoted as 22nd/27th.

## E. Nuisance removal

| # | claim | reported | my value | verdict |
|---|---|---|---|---|
| 16a | disease accuracy before / after / chance | 0.820 / 0.586 / 0.275 | 0.8198 / 0.5856 / 0.2754 | **CONFIRMED** |
| 16b | split is honest — no test paper informs the basis | honest | honest | **CONFIRMED** |
| 17 | mean study-design AUC delta | +0.004 | **+0.0044** | **CONFIRMED** |

**16b is genuinely clean and I looked for the leak deliberately.** `rng.permutation(usable)` splits
**papers** 60/40 (165 train, 111 test). The centroids that build the basis come from `train` only;
the nearest-centroid classifier's centroids also come from `train` only; accuracy is evaluated on
`test` only. Test papers are projected through a basis they never contributed to. `paper_matrix()`
touches all papers but only averages a paper's own chunks — no cross-paper information. No leakage.

Two small notes: 11 diseases pass the ≥4-paper filter but only **10** get a centroid (`Multiple
system atrophy (MSA), Parkinson's disease (PD)` has <2 training papers), while the classifier still
scores 11 classes — a harmless inconsistency between the basis and the evaluation. And
`disease_acc` at 0.586 is contaminated by the bug in G1.

## F. Coverage audit

| # | claim | reported | my value | verdict |
|---|---|---|---|---|
| 18 | VALUED counts over 303 papers | 105 / 98 / 78 / 53 / 47 / 22 / 13 / 10 | **105 / 98 / 78 / 53 / 47 / 22 / 13 / 10**, `n_papers` = 303 | **CONFIRMED** |
| 19 | "Differential Abundance Test" non-blank | 306 of 337 | **306 of 337** | **CONFIRMED** |

Claim 18 is internally consistent too — each row's `valued_titles` list length equals its `valued`
count, so the labels the downstream AUC work consumes match the headline percentages.

Claim 19's column is actually named `Differential Abundance Test (try Ctrl F for these)`; 337 data
rows, 306 non-blank, 31 blank, and no "NA"-style placeholder values inflating the count. See G7
for a caveat on what those 306 values contain.

---

# G. Judgement

## G1. The disease subspace is not the subspace the code says it is — a real bug

`nuisance_removal.py:162-166`:

```python
U, S, _ = np.linalg.svd(C, full_matrices=False)
ev = (S ** 2) / (S ** 2).sum()
k = int(np.searchsorted(np.cumsum(ev), 0.95) + 1)
B = np.linalg.qr(C.T[:, :])[0][:, :k]     # orthonormal basis, 384 x k
```

The SVD's **right** singular vectors — the ones that live in the 384-dim embedding space and
define the principal directions — are discarded into `_`. Only the singular values are kept, and
they are used solely to choose `k`. The basis actually used is the first `k` columns of a QR
decomposition of `C.T`, which is Gram–Schmidt over the disease centroids **in alphabetical order
of disease name**. `span(Q[:, :7])` is the span of the first seven alphabetically-sorted
centroids, which has no reason to be the top-7 variance subspace.

It isn't. Principal-angle cosines between the shipped basis and the true top-7 SVD subspace:

```
[1.000  1.000  1.000  1.000  1.000  0.161  0.007]
angles: 0°  0°  0°  0°  0°  80.7°  89.6°
```

Five directions coincide; **two are nearly orthogonal**. I confirmed the shipped
`disease_basis.npy` *is* the QR basis (min principal-angle cosine 1.000000) and is *not* the SVD
basis (min cosine 0.0067).

Consequences:

- **The printed "96% of between-disease variance" is wrong for the subspace actually removed.**
  0.967 is `ev[:7].sum()`, a property of the SVD subspace. The QR subspace captures **83.3%**.
- **The removal is much weaker than intended.** Held-out disease accuracy after removal:

  | basis | disease acc | vs chance 0.275 |
  |---|---:|---|
  | none (before) | 0.8198 | — |
  | QR, k=7 (shipped) | 0.5856 | still far above chance |
  | **SVD top-7 (intended)** | **0.1712** | **below chance — fully removed** |
  | SVD all 9 | 0.2342 | below chance |

- **It is not a precision/recall trade — the correct basis is better on *both* of the script's own
  tests.** Mean study-design AUC delta is **+0.0075** with the SVD basis versus **+0.0044** with
  the shipped QR basis. So the fix removes more disease *and* preserves more study-design signal.

Two downstream claims need retracting or rewording:

1. `HANDOFF_embeddings.md`'s caveat — *"0.532 is still above the 0.275 chance line. One projection
   removes most of the disease signal, not all. Iterating (INLP) would go further."* This
   **misdiagnoses a bug as a method limitation.** A single correct projection takes it to 0.171.
   INLP is not needed for this.
2. `contrast_experiment.py`'s docstring — *"Runs on the DISEASE-DECONFOUNDED vectors by default, so
   a hit cannot simply be 'these two piles are about different diseases'."* **Not supported.** The
   space the experiment actually ran in still predicts disease at 0.586 against a 0.275 chance
   line. Disease is attenuated, not eliminated, and a hit could still partly be a disease effect.

The whole embedding-side analysis should be re-run with `B = Vt[:k].T`. I did not do so — that
would mean writing `chunk_vecs_deconfounded.npy` and `disease_basis.npy`, which is outside my
brief.

## G2. The stratified per-edge design is valid (G20a) — with an unstated power cost

**(a) Yes, it genuinely fixes the 0.706 problem.** The cap arises because a paper-level feature
vector carries opposite labels. Within a single edge that cannot happen — and I verified it
empirically rather than taking it on the argument: **zero** of the 226 contested edges have any
paper on both sides. So within an edge the paper→label mapping is a function, the feature and the
label agree, and the Mann-Whitney AUC is a well-posed statistic. The van Elteren / stratified
aggregation with `w = n_up · n_dn` weights is the standard and correct way to pool it. This part
of the design is right, and the reasoning in the docstring is honest and accurate.

**The unstated cost:** across (not within) edges, **50 of the 124 papers in the analysis — 40% —
carry label "up" in one edge and "down" in another**. That is not a validity problem, but each
such paper contributes with opposite sign in different strata, so a substantial fraction of the
evidence self-cancels during aggregation. It is a large part of why the design is underpowered,
and it is not mentioned anywhere. Median edge is 3-up vs 2-down.

## G3. The permutation null is anti-conservative — quantified (G20b)

The docstring flags this honestly (*"edges share papers … this null is mildly anti-conservative"*)
but never sizes it. I sized it.

Overlap first: the 43 edges have 312 paper-slots over **124 distinct papers**; a paper appears in
2.52 edges on average, up to 8; **72 of 124 papers appear in more than one edge**. So the
dependence is not marginal.

Calibration test: I built a **true null** by randomly re-assigning the paper→score rows — this
destroys any paper/label association while preserving the edge structure and the paper-sharing
pattern exactly — then ran the script's own p-value machinery. 200 simulated null datasets × 34
probes = 6,800 p-values:

| nominal α | observed P(p ≤ α) | inflation |
|---|---:|---:|
| 0.01 | 0.0169 | **1.7×** |
| 0.05 | 0.0649 | **1.3×** |
| 0.10 | 0.1209 | 1.2× |
| 0.20 | 0.2229 | 1.1× |

**Direction: anti-conservative — p-values are too small, and increasingly so in the tail that
matters.** The docstring's instinct was right. Applied to the headline, p 0.0123 is realistically
~0.02.

**A counter-weight the analysis does not claim credit for.** The 34 probes are heavily correlated:
mean |r| between their null statistics is 0.332, max 0.786, and the effective number of
independent tests is **≈ 6.6, not 34**. BH is valid under this kind of positive dependence, so the
correction is not wrong — but it is *over*-correcting relative to the true multiplicity. The two
errors point in opposite directions and roughly offset. **Net: the "0 of 34 survive" conclusion is
safe.** Nothing was close enough for either correction to change the verdict.

## G4. The minimum-detectable-effect figure is not sound as labelled (G21)

The arithmetic is right (1.96 × 0.0408 ≈ 0.080; I get mean null SD 0.0407–0.0411 across seeds),
and one assumption behind it checks out — per-probe null SDs are tight (0.0400 to 0.0419, ratio
1.05) and the null is centred (max |mean| 0.0012), so a single pooled SD is fine.

But **1.96·SD is the threshold for p < 0.05 on a *single, uncorrected* test at 50% power.** The
experiment's actual decision rule is **BH q < 0.05 across 34 probes**. On the same scale:

| threshold | value |
|---|---:|
| 1.96·SD — single test, 50% power *(what is reported)* | 0.080 |
| 2.80·SD — single test, 80% power | 0.115 |
| 3.18·SD — BH q<0.05, best case, 50% power | **0.131** |
| 4.02·SD — BH q<0.05, best case, 80% power | **0.165** |

So the design is **1.6–2× less sensitive than "≈0.080" implies**. Note the largest observed
effect, 0.1031, sits *below even the 50%-power BH threshold*. That actually **strengthens** the
honest reading — this is "underpowered for the stated decision rule", not "no effect exists" — but
0.080 should not be quoted as the minimum detectable effect. Quote ~0.13 (50% power) or ~0.17
(80% power) against the rule actually used, or quote 0.080 explicitly as an uncorrected
single-test threshold.

## G5. The paper score is a length proxy — the confound is real, it just doesn't bite here

Paper score is `max` cosine over the paper's chunks. **The max of n draws is monotone in n**, and
papers in this index range from **2 to 172 chunks** (median 72). So a long paper scores higher on
every probe for purely mechanical reasons.

Measured: Spearman correlation between chunk count and probe max-score is **median 0.558**, min
0.398, max 0.698 — **32 of 34 probes above 0.5**. The paper score is substantially a length
measurement.

I then ran the identical stratified statistic using **chunk count itself as the probe**:

```
effect −0.0059,  p 0.898,  would rank 29 of 35 by |effect|
```

So the up- and down-piles do **not** differ in length, and the headline results are not
length artifacts. **This time.** Two things follow: (i) it should be stated as a checked-and-cleared
confound rather than left unmentioned, and (ii) it explains a large share of the inter-probe
correlation in G3, and it will not automatically clear for a different edge set, a different probe
bank, or the per-relation-sentence unit planned next. The `max` should be replaced with something
length-stable (rank-within-paper, or mean of top-k) before this score is reused.

The choice of `max` over `mean` is otherwise well-justified and I agree with it — a paper states
its storage protocol once in ~72 chunks.

## G6. `contrast_experiment.json` cannot reproduce the claims made from it

Restating D13f as a process finding because it is the most likely thing to burn a future reader:
the artifact on disk is the `--raw-space` run. `--raw-space` and the default run share one `OUT`
path, so the sanity check silently overwrites the primary result. Anyone opening the JSON to check
the headline will find different numbers and conclude the headline is fabricated. It isn't — I
reproduced it — but the artifact does not support it. One-line fix: separate output filenames.

## G7. Two stale-artifact problems in the surrounding documentation

**(a) `variable_sweep.json` and `probe_results.json` predate the chunk rebuild.** Timestamps:
`variable_sweep.json` 16:14 and `probe_results.json` 16:06, versus `chunks.jsonl` 19:48 and
`chunk_vecs.npy` 19:54. They were computed against the **old 19,483-chunk index**, not the current
20,905-chunk one. This is directly visible: for probe sets that are character-identical between
`variable_sweep.py` and `nuisance_removal.py`, the two files disagree —

| variable (identical queries) | variable_sweep.json | nuisance_removal.json `auc_before` |
|---|---:|---:|
| sample storage | 0.6949 | 0.6359 |
| bmi | 0.7988 | 0.7798 |
| antibiotic use | 0.7212 | 0.7161 |
| differential abundance method | 0.8382 | 0.8429 |

The AUC/lift table in `HANDOFF_embeddings.md` §4 and the probe table in §4b therefore describe a
retired index. The *conclusions* (read lift not AUC; recruitment setting is the worst) are
unlikely to flip, but the numbers are not current and `variable_sweep.py` should be re-run.

**(b) `HANDOFF_embeddings.md` §5 quotes superseded nuisance-removal numbers.** It reports
0.793 → 0.532, mean delta +0.006, bmi 0.799 → 0.821. The current `nuisance_removal.json` says
0.8198 → 0.5856, mean delta +0.0044, bmi 0.7798 → 0.8141. **None of the four match.** The claims
I was given (0.820 / 0.586 / +0.004) track the JSON, so the JSON is current and the handoff is
stale. Anyone quoting the handoff will quote wrong numbers.

## G8. "306 of 337" overstates the datasheet's usable coverage

Claim 19 is arithmetically correct, but of the 306 non-blank values, **117 are the literal string
`"Other"`**. Actual named methods: LEfSe (LDA) 120, Linear/MaAsLin 23, DESeq 13, ANCOM 7 — i.e.
**189 of 337 rows (56%) name a method**, not 91%. The recommendation in
`FINDINGS_variable_coverage.md` — *"`differential abundance method` is not new — the correct
datasheet already has it hand-curated at 306/337. Join to it, don't extract it"* — is right in
direction but the join recovers a usable method name for **56%** of rows, not 91%. Against the
audit's own VALUED extraction rate of 32.3%, joining still wins; the margin is just smaller than
advertised.

## G9. Things I checked that are fine

- **The `probe_B_sequencing` "FAIL" artifact is correctly diagnosed** in the handoff. Confirmed:
  accuracy 0.80819, null mean 0.80811, and the base rate 240/297 = 0.80808 are the same number.
  The handoff already says not to quote it as a negative result. Correct call, no new error.
- **No circularity in the retrieval evaluation.** The regex labels in
  `audit_variable_coverage.json` are produced by string matching over raw text with no reference to
  embeddings, so scoring embedding retrieval against them is a genuine agreement measure between
  two independent methods. `variable_sweep.py`'s own docstring already states this correctly,
  including that a retrieval "false positive" may be a regex under-count.
- **The probe-bank projection is handled correctly.** `contrast_experiment.py` and
  `nuisance_removal.py` both project probe vectors through the same basis before comparing them to
  projected chunks, and re-normalise afterwards. Comparing a raw probe to deconfounded chunks
  would be meaningless and neither script does it.
- **The permutation counts are adequate.** 10,000 draws gives a p-floor of 1e-4, well below
  anything that matters here; there is no per-edge discreteness problem at the aggregate level
  (unlike the single-edge case the docstring correctly rules out).

---

## Recommended actions, in priority order

1. **Fix `nuisance_removal.py:162-166`** — use `Vt[:k].T`, not `qr(C.T)[0][:, :k]`. Re-run, then
   re-run `contrast_experiment.py` on the corrected space. Everything downstream of the
   deconfounded vectors is provisional until this is done.
2. **Give `--raw-space` its own output file** so the sanity check stops destroying the headline.
3. **Correct the MDE statement** to ~0.13 (50% power) / ~0.17 (80% power) under BH q<0.05, or
   label 0.080 as an uncorrected single-test threshold.
4. **Soften the "cannot simply be different diseases" claim** in `contrast_experiment.py` — until
   (1) lands, it isn't true.
5. **Re-run `variable_sweep.py`** against the current index and update `HANDOFF_embeddings.md`
   §4, §4b and §5, all of which quote retired numbers.
6. Report the anti-conservatism as measured (1.3× at α=0.05, 1.7× at α=0.01) alongside the
   offsetting note that effective multiplicity is ~6.6, not 34.
7. Note the 18 dropped papers and the 40% cross-edge label conflict as stated power limitations.
8. Replace `max`-over-chunks with a length-stable aggregation before reusing the paper scores.
