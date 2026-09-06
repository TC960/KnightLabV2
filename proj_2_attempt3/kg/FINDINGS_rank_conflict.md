# Are contested edges rank confusion? — Task 3.2, the containment links

*2026-09-06. Scripts: `analyze_rank_conflict.py`, `attack_rank_conflict.py`.
Data: `rank_conflict.json`, `rank_conflict_attacks.json`.*

## The question

The project's most load-bearing design decision is that taxonomic containment is
**modelled, never collapsed**, and it has been justified by a single example:
in Parkinson's, *Lachnospiraceae* (family) is depleted while *Hungatella* (a
genus inside it) is enriched. 708 containment links exist and, until now,
nothing consumed them. The GraphRAG session counted the general case — 903
(taxon, disease) edges have a parent edge in the same disease and **229 of those
point the opposite way** — and stopped there. Nobody asked whether those
opposite-direction pairs are biology or bookkeeping.

Two questions, kept apart because only one of them is inferential.

## Q1 — Who actually asserts the conflict? (deterministic)

For every parent/child pair that shares a disease, compare majority directions;
where they differ, ask whether a **single paper** reports both. A within-paper
conflict cannot be rank confusion: the same authors, the same cohort, the same
pipeline produced both numbers. A conflict that appears only after pooling
papers is a much weaker object.

952 parent/child pairs share a disease:

| | count |
|---|---|
| same direction | 586 |
| **opposite direction** | **241** |
| no majority (exact tie) | 125 |

Splitting the 241 opposite pairs by who asserts them:

| verdict | named parent+child | placeholder involved |
|---|---|---|
| **within_paper** — one study reports both directions | **28** | 5 |
| cross_paper_only — shared papers exist and all *agree*; conflict comes from papers measuring only one side | 18 | 1 |
| no_shared_paper — no study ever measured both | 156 | 33 |
| total | 202 | 39 |

**This reframes the 229 figure.** Only **33 of 241 opposite pairs (14%)** are
asserted inside a single study. **189 of 241 (78%) rest on no shared paper at
all** — the family was measured by one set of studies, the genus by another, and
the "conflict" is an artefact of pooling. That is not evidence of rank confusion
in the extractor, but it is also not the strong biological claim the number
looks like at first glance. Anyone citing "25% of containment pairs conflict"
should cite 14% instead, and say what it means.

**The flagship example survives, and is now properly witnessed.**
*Lachnospiraceae*↓ / *Hungatella*↑ in Parkinson's is a `within_paper` conflict:
one study reports both. Other well-evidenced within-paper conflicts include
*Oscillospiraceae*↑ / *Faecalibacterium*↓ (Parkinson's), *Bacteroidaceae*↑ /
*Bacteroides*↓ (Stroke), and *Oscillospiraceae*↓ / *Anaerotruncus*↑
(Alzheimer's). These 33 pairs are the concrete argument for the containment
layer and are the highest-value review targets after the doubly-contradicted 11.

### A stale number, corrected

Checking the flagship example against the built graph turned up a factual error
on the **published page**: `kg.html`, `CLAUDE.md`, `build_kg.py` and
`viz_network.js` all claimed *Lachnospiraceae* is "depleted in Parkinson's
across **15 papers**". The graph says **9 papers, 8 down / 1 up**, and the edge
is **contested**; *Hungatella* is 7 papers, 6 up / 1 down, also contested. The
15 predates the 2026-09-03 deduplication and was never updated. All four are
corrected, and the viewer now states the within-paper witness explicitly.

## Q2 — Do related taxa agree more, within a paper? (inferential)

Unit: (paper × disease × taxon-pair). Both arms come from the **same paper**, so
cohort, country, pipeline and that paper's own enrichment propensity are
differenced out by construction. Null: shuffle each paper's direction labels
across the taxa it reported, preserving that paper's up/down counts — a
**paper-level** null, since pair-level shuffling has produced three false
positives on record here.

| | related | unrelated | gap |
|---|---|---|---|
| agreement | **0.8903** | 0.5367 | **+0.3536** |

z = 15.5, p = 0.0001 (10,000 permutations, p at the floor), null SD 0.022,
minimum detectable effect ±0.044. 474 related pairs, 23,673 unrelated, 102
paper×disease units.

A z of 15 on this corpus is a reason for suspicion, not celebration. Four
attacks (`attack_rank_conflict.py`, 2,000 permutations each):

| attack | result | verdict |
|---|---|---|
| **A. Pipeline self-agreement** — drop all 100 split placeholder nodes and their links, since the split *created* both nodes from one mention | +0.3592, z=15.5 | survives, slightly **stronger** |
| **B. One loud paper** — cluster-robust: one gap per paper×disease unit, unweighted mean, permuted the same way (top paper contributes 35 of 474 pairs) | +0.3611, z=11.4, 100 units | survives |
| **C. Direction skew** — papers average 0.68 of calls in one direction | controlled by the within-paper shuffle by construction | not the cause |
| **D. Siblings, not ancestry** — compare against *same-rank* unrelated pairs, in case the effect is really "co-mentioned taxa agree" | +0.3502, z=14.6 | survives; it is ancestry specifically |

**Surviving claim: within a single paper, taxonomically related taxa agree on
direction 89% of the time against a 54% baseline for unrelated taxa.**

This is the expected direction — a family's abundance is largely the sum of its
genera, so agreement is close to arithmetic — which makes it **face validity for
the extractor** rather than a discovery. Its value is the complement: the
**11% that disagree within a paper** are not noise and not rank confusion, and
they are precisely what merging ranks would destroy. The project's refusal to
collapse ranks now rests on a measured 11%, not on one anecdote.

## What this does and does not license

- **Do not** collapse containment. Confirmed, with a number.
- **Do** discount the "229 / 25% of containment pairs conflict" framing. The
  honest figure for *asserted* conflict is 33 pairs, 14%.
- **Do not** read Q2 as a discovery. 89% vs 54% is close to what taxonomy
  arithmetic predicts; it is a sanity check the extractor passes.
- The 125 exact-tie pairs are excluded from Q1 as having no majority to
  conflict with. They are not evidence of agreement either.

## Power

Q2 is the first well-powered question asked of this graph in several sessions:
MDE ±0.044 against an observed +0.354, so the effect is ~8× the resolution
limit. Q1 needs no power statement — it is a string comparison over the full
set, not a sample. The weak spot is that most within-paper conflicts rest on
**exactly one** shared paper (`n_shared = 1` for 32 of 33), so any individual
pair is a single-study claim even though the aggregate is not.
