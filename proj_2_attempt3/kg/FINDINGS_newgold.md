# Re-scoring against the new high-confidence gold

*2026-09-12. Reproduce with `score_newgold.py` (full harness + error analysis) and
`build_kg_gold.py` (gold-derived graph). Three of the numbers below were also
re-derived independently in-session with a separate scorer; see "Reconciliation".*

Gold: `high_confidence - final_constrained_override.csv` — 334 DOIs, two rows each
(Enriched / Depleted), 40 with both cells blank (excluded from scoring). Joins to
our corpus via the Main Datasheet's DOI+Title columns; all 334 gold DOIs are present
there.

---

## 1. The extractor was never as bad as we were reporting

Same model, same prompt, same cached extractions. **Only the reference changes.**

| reference | P | R | **F1** |
|---|---:|---:|---:|
| OLD (datasheet columns) | 0.609 | 0.672 | **0.639** |
| **NEW (high_confidence)** | **0.750** | **0.766** | **0.758** |

253 papers scoreable under both. Precision moves **+0.141** with no change to the
system under test.

Permutation (shuffle gold→paper, 1,000 draws): observed **0.755**, null mean 0.097,
null max 0.123, **p = 0.001**. The score is real work, not matching noise.

**This is the audit in the root `CLAUDE.md` landing.** The long-standing
"over-extraction problem" was largely the old reference being incomplete.

### It is NOT a density effect — a claim to retract

An earlier draft (`GPU_RUN_PLAN.md` §2) said the new gold is **2.1× denser**
(10.5 vs 5.1 taxa/paper). That compares against the **64%-blank export**, not the
corrected sheet, and is wrong. Measured against the corrected sheet on identical
papers: **9.6 vs 10.5 taxa/paper — 1.09×**.

So the gain is not "more labels". The new gold names *the taxa the extractor
actually found*. It is better curated, not bigger.

## 2. Error analysis — where the remaining error actually is

1,402 total error charges (FP + FN):

| bucket | events | charges | share |
|---|---:|---:|---:|
| false positive (unexplained) | 695 | 695 | **49.6%** |
| false negative (unexplained) | 653 | 653 | **46.6%** |
| direction flip | 19 | 38 | **2.7%** |
| rank mismatch | 8 | 16 | 1.1% |

**Direction is nearly solved.** Only 19 flips — 2.7% of all error. That matters more
than F1 for this project, because the KG encodes direction only.

Corroborated independently: comparing `graph_gold.json` against `graph.json`,
direction agrees on **925 of 970 decidable shared edges = 95.4%**. Two different
routes, same conclusion.

### The false positives are mostly not our fault

**647 of 695 false positives (93.1%) appear VERBATIM in the paper text.**

A taxon we "invented" that is sitting in the paper is not a hallucination — it is
almost certainly a real finding the gold did not record. Near-zero hallucination,
consistent with the eval-v2 finding that Opus 4.8's "extra" taxa were mostly
under-annotation in the human gold.

### The false negatives are concentrated at genus, and half are not in the text

FN rank profile: **genus 334**, species 92, family 77, phylum 55, class 32, order 26,
unresolved 33.

Only **295 of 653 (45.2%)** of the taxa we missed appear verbatim in the paper text.
So the majority of "misses" are for taxa that are not stated in the body at all —
likely curated from tables, figures, or supplements that our text pipeline strips.
That is a **corpus** limitation, not a prompt limitation, and no prompt change will
fix it.

## 3. The gold-derived knowledge graph

`build_kg_gold.py` → `graph_gold.json`.

| metric | extraction | GOLD |
|---|---:|---:|
| papers | 285 | 294 |
| taxa | 833 | 676 |
| diseases | 43 | **25** |
| edges | 1,927 | 1,740 |
| **contested** | 226 (11.7%) | **275 (15.8%)** |
| containment links | 625 | 584 |
| taxid resolution | 82% | **87%** |

### Contested edges are not an extraction artifact

The human gold produces a **higher** contested rate than our extractor, and 152 edges
are contested in **both** independently. Disagreement is a property of the literature.

This retires an earlier framing of mine. In the gold standard itself, **246 of 294
non-blank papers (84%) report both an enrichment and a depletion**. A paper reporting
both directions is the norm, not a defect.

What remains true is narrower and is a standard meta-analytic limit: a study-level
moderator can only explain *between*-study contrasts. It says nothing about the
papers being wrong. Treat it as a scope note on `contrast_experiment.py`, not a
finding about the literature.

## 4. Decision: do not buy GPU time

- Only **5 of 38** missing gold papers could be fetched; the other 33 are not open
  access.
- 5 papers ≈ **79 s** of compute against ~15 min of provisioning — 99% overhead.
- Re-running the 260 already-extracted papers needs a hypothesis to test. There isn't
  one: the prompt's supposed weakness was an artifact of the old reference.
- The lab has free UCSD DSMLP GPUs, where eval-v2 already ran.

`GPU_RUN_PLAN.md` holds the launch commands and guardrails if that changes.

## 5. Reconciliation — three independent scorings

| source | old-gold F1 | new-gold F1 |
|---|---:|---:|
| `score_newgold.py` (253 papers) | 0.639 | 0.758 |
| in-session check (257 papers, separate scorer) | 0.588 | 0.737 |
| `GPU_RUN_PLAN.md` §2 (260 papers) | 0.476 | 0.751 |

New-gold F1 is stable at **0.74–0.76** across three implementations. Old-gold varies
more (0.48–0.64) because each used a different "old" reference — the 64%-blank export
versus the corrected sheet. **Quote the new-gold figure as ~0.75 and the improvement
as +0.12 to +0.15**, not the largest available delta.

Metric caveat: all three use char-ngram cosine ≥ 0.5, not the taxonomy-aware LCA
metric (which needs the NCBI taxdump). LCA scores strictly higher, so **0.75 is a
lower bound**.

## 6. What to do next, in order

1. **Re-score with the taxonomy-aware LCA metric** for a comparable, less punitive
   number. `score_newgold.py` already loads the taxdump.
2. **Adjudicate the 19 direction flips.** Small, high-value: a human and the model
   read the same paper and disagreed on direction. Highest-signal error set available.
3. **Rebuild and republish the KG on the gold**, or on a gold+extraction union.
   `graph_gold.json` is ready; `validate_external.py` has NOT yet been run against it
   (the agent died first), so Disbiome/Peryton agreement for the gold graph is unknown.
4. **Retrieval-augmented few-shot** before any weight update: for each paper, retrieve
   the 3 most similar gold papers and use them as in-context examples. Uses all 334
   gold papers at inference with nothing to overfit.
5. Do **not** chase the genus-level false negatives with prompt edits until it is known
   how many live only in tables/supplements — 55% of misses are not in the body text.
