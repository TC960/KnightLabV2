# Accuracy of the knowledge graph against the new test set

*2026-09-17. Reproduce with `score_lca.py` / `run_lca_eval.py`.*

The gold standard (`high_confidence - final_constrained_override.csv`, 334 DOIs,
hand-curated by Emily Song with no LLM assistance) is a **test set**. It measures
the extractor. It is **not** a source of graph content, and it must not be merged
into the graph — per Sam, *"No Emily's data since this is manually verified."*
The graph must remain purely what the model extracted from the literature, or the
claim "this is what the papers say" stops being true.

The graph is therefore **unchanged**: 271 papers, 883 taxa, 40 diseases, 2,008
edges, 220 contested, 727 containment links. Only the accuracy numbers move.

---

## 1. Extraction accuracy

Cached extractions, 260 papers scoreable against the new gold.

| metric | P | R | **F1** |
|---|---:|---:|---:|
| char-ngram | 0.714 | 0.752 | **0.733** |
| taxonomy-aware (LCA) | 0.721 | 0.759 | **0.739** |

Permutation, 1,000 draws: observed 0.739, null mean 0.148, **p = 0.001**.

Against the *old* reference the same extractions scored **0.639**. Nothing about
the model changed — the old annotations were incomplete.

Two corrections are already folded into the numbers above:

- **The matcher double-counted.** `eval-v2/run_eval.py` and both matchers in
  `score_lca.py` let two predictions claim the same gold taxon and each score a
  true positive. Fixed. This inflated everything on record, including
  `leaderboard.csv`: the pre-fix figures were 0.755 / 0.780.
- **LCA is worth +0.007, not +0.025.** Of 112 LCA "rescues", only 9 claimed a gold
  taxon nothing else had matched; the rest were redundant credit created by the
  bug above.

## 2. Model vs human curator, scored against a third party

The cleanest statement of extractor quality available, because it depends on
neither the in-house gold nor the model. Same 259 papers, model output and human
annotation each compared independently to Disbiome and Peryton.

| | edges | Disbiome | Peryton |
|---|---:|---:|---:|
| human (Emily) | 1,643 | **80.5%** | **80.9%** |
| model (Qwopus3.5) | 1,831 | 74.8% | 73.5% |
| gap | | +5.7 pts | +7.4 pts |
| Fisher exact | | p = 0.275 | p = 0.186 |

**Neither gap is significant.** On the same papers the model sits within ~6 points
of a human curator, and at this sample size that difference is not distinguishable
from chance.

So the inter-annotator agreement between Emily's curation and Disbiome is
**80.5%** — and the model reaches **74.8%** on the same papers.

### A retracted claim

An earlier version of this file reported human-backed edges at ~89% and
model-only edges at ~68%, a 21-point gap at p < 0.001. **That comparison was
invalid.** The "model-only" bucket was defined as *the edges the human did not
confirm*, so it is the residual after removing every point of agreement — a
selection effect, not a measurement. It compared "corroborated vs uncorroborated",
not "human vs model". The head-to-head above is the correct test and gives 6
points, not significant.

## 3. Direction errors

19 reported direction flips reduce to **15** real ones. Four were fuzzy-match
artifacts pairing different organisms — `parabacteroides` (genus) against
`bacteroidetes` (phylum) at char similarity 0.545; `streptococcaceae` against
`peptococcaceae` at 0.617. That the matcher pairs different organisms above its own
0.5 threshold is itself an argument for taxid comparison over string similarity.

The 15 are in `DIRECTION_FLIPS_for_review.csv` for adjudication. They are worth a
human pass because the KG stores direction and nothing else: a missed taxon is an
absent edge, a flipped direction is a *wrong* edge.

Direction errors are 2.7% of all error charges — the smallest bucket, and the
most consequential.

## 4. Where the remaining error is

1,402 error charges: 49.6% unexplained false positives, 46.6% unexplained false
negatives, 2.7% direction, 1.1% rank.

- **647 of 695 false positives (93.1%) appear verbatim in the paper text.** A taxon
  we "invented" that is sitting in the paper is not a hallucination; it is a real
  finding the gold did not record.
- **Only 295 of 653 false negatives (45.2%) appear in the text at all.**

Two hypotheses about that second number were tested and both were wrong:

- *Tables.* Measured across 50 papers: **zero** gold taxa appear in a table but not
  the body. Parsing tables would recover nothing. The scraper change proposed on
  this basis was dropped.
- *Naming variants.* `normalize_taxa.py` strips GTDB suffixes (`Firmicutes_A`),
  SILVA numbered clades (`Prevotella 9`), QIIME prefixes (`g__Blautia`) and
  classifier placeholders (`unclassified Veillonella`). It recovers **34 of 2,729
  gold taxa — 1.3%**, taking findability 82.1% → 83.4%. Real, but small.

The bulk of the false negatives remain unexplained and are the open question.

## 5. Model choice

Re-ranked on the new gold with the fixed matcher, testv2 (n=15):

| model | new gold, LCA F1 |
|---|---:|
| **qwopus3.5-27b-v3** | **0.722** |
| qwythos-9b | 0.650 |
| qwen3.6-35b-a3b | 0.614 |
| qwopus3.6-35b-a3b-mtp | 0.596 |
| qwen2.5-32b-instruct | 0.577 |

Qwopus 3.5 is first under every gold/metric combination. Bootstrap over 2,000
resamples: P(rank 1) = 0.837. Qwopus **3.6** was unfairly penalised by the old
reference and gains most from the correction, but does not overtake 3.5.

## 6. What this does not tell us

The Disbiome/Peryton agreement figures are **not independent replication**. 43 of
our papers are cited by Disbiome and 24 by Peryton, and because those are the
heavily-reported ones they back about half the decisive pairs. Agreement splits
hard on that line — ~90% where both sides read the same paper, ~55% where the
literatures are disjoint. The headline blends *reading fidelity* with
*cross-literature reproducibility* and measures neither cleanly. See
`FINDINGS_independence.md`.
