# Do papers that agree on direction share a taxon vocabulary?

**Answer: no, at this corpus size — and the version of the result that said "yes"
was 12 duplicate papers.**

Session of 2026-09-03. Code: `cooccur_direction.py`, `cooccur_diagnostics.py`,
`cooccur_followup.py`. Results: `cooccur_direction.json`,
`cooccur_diagnostics.json`, `cooccur_followup.json`.

---

## The question

The pooled Task 1 analysis, named by the previous session as the highest-value
next step. Within a *fixed* (taxon, disease) contested edge — so disease, taxon
and rank are all controlled by construction — do the papers reporting
**enrichment** differ from the papers reporting **depletion** in what *other*
taxa they talk about?

If they did, the co-occurrence profile would be a first interpretable
explanatory variable for contested edges, which nothing has yet explained.

**Substrate.** `relation_sentences.json`, the filter validated at 94.8% recall
of a 97.3% ceiling. A paper's profile is the binary incidence vector over the
taxids named in its *kept* (relation-bearing) sentences — 609 taxids, median 20
per paper. The edge's own focal taxon is removed from the vectors: it is in
every paper of the edge by construction.

**Statistic.** Every pair of papers on the same edge is same-direction or
cross-direction; the statistic is mean profile cosine of same-direction pairs
minus mean cosine of cross-direction pairs. Reported pooled over all pairs, and
averaged per edge so one well-papered edge cannot dominate.

**Nulls, both at the PAPER level.** Observations are not independent — one paper
contributes to many edges — and pair-level shuffling is exactly what produced
the `diet_controlled` (p=0.002 → FDR 0.243) and ASD (p=0.211) false positives
already on record.

- **null A** — permute the paper→profile map, *within disease*. Every edge keeps
  its exact up/down group sizes; only the profile association breaks.
- **null B** — flip a whole paper's directions together, preserving its internal
  structure and re-partitioning the edges.

---

## What happened: a strong positive, then a bug

The first run looked like the best result the project had produced.

| | before dedup | final corrected graph |
|---|---:|---:|
| usable contested edges | 138 | 130 |
| within-edge pairs | 2,091 | 1,848 |
| **pooled** | **+0.0155** (p=0.066 A / 0.035 B) | **+0.0023** (p=0.794 / 0.513) |
| **per-edge** | **+0.0472** (p=0.001 / 0.001) | **+0.0137** (p=0.141 / 0.061) |
| per-edge, balanced edges only | +0.0244 (p=0.010) | +0.0080 (p=0.405) |
| rank-based AUC | 0.5596 (p=0.014) | 0.5347 (p=0.206) |

("final corrected graph" = after deduplication, the NMDAR disease merge and the
placeholder split; the numbers were stable across all three — deduplication is
what moved them.)

Per-edge +0.047 against a null SD of 0.009 is five standard deviations, under
*both* paper-level nulls. It survived the first four attacks:

- **Profile size is not the driver.** The obvious mechanical story — most
  contested edges are unbalanced, so every cross-direction pair contains the
  lone dissenter, and if dissenters are merely atypical the effect appears with
  no relation to profile *content*. Refuted twice over: majority-side members
  average 28.6 taxa and minority-side 29.5 (no asymmetry), and restricting the
  permutation to within profile-size quintile left the result untouched
  (p=0.001).
- **Not a study-design confound.** Permuting within country (p=0.001) or within
  sequencing type (p=0.001) did not move it.
- **Not the similarity metric.** Jaccard gave +0.0452, p=0.001.
- **Not one outlier edge.** The per-edge *median* was +0.0150, p=0.009, and 79
  of 138 edges were positive.

The attack that worked was deleting near-duplicate pairs — pairs with cosine
≥ 0.8. That is **1.1% of pairs (23 of 2,091)** and it dropped the effect to
+0.0140, p=0.113.

Deleting the high-similarity tail is a *biased* operation: that tail is exactly
where a real same-direction signal would live, so this alone proves nothing. But
it was worth asking what those 23 pairs *were*, and the answer was not
statistical.

**All eight distinct paper pairs had cosine of exactly 1.00.** They were the
same paper, in the corpus twice.

---

## The bug

12 papers were scraped once from a PubMed link and again from a PMC or publisher
link. The two copies' titles differ only by a trailing period and/or a
curly-versus-straight apostrophe:

```
"The gut microbiota in multiple sclerosis varies with disease activity"
"The gut microbiota in multiple sclerosis varies with disease activity."

"Metagenome-assembled microbial genomes from Parkinson's disease fecal samples."
"Metagenome-assembled microbial genomes from Parkinson’s disease fecal samples"
```

Every paper key in this pipeline is the raw title string, so nothing ever
collapsed them. **Because edge weight is defined as paper count, each duplicate
voted twice.** Measured on the rebuild:

- contributing papers **281 → 272**
- **`n_replicated` 472 → 437: 35 edges presented as replicated but rest on a
  single paper** — 7.4% of every replicated edge in the graph. Among them
  *Flavonifractor plautii*/MS, *Hungatella*/MS, *family Lachnospiraceae*/PD.
- 76 edges change vote counts. **4 lose a majority direction they only had
  because one paper voted twice**: *Bacteroides*/Dementia, *[Clostridium]
  leptum*/MS and *Bifidobacterium longum*/MS were each 1-vs-2 where the 2 was
  one paper counted twice, and *Butyricimonas*/MS resolves a false 2-2 tie.
- 0 contested-status changes, necessarily: `contested = bool(up and dn)`, which
  a duplicate can neither create nor destroy.

The three edges the failed analysis ranked as its strongest signal — +0.944,
+0.870, +0.796 — are precisely three of those four. **The statistic was
measuring the duplication.**

Fixed in `build_kg.py` (`--keep-duplicate-papers` restores the old behaviour),
so a rebuild cannot silently revert it; verified a fixed point by rebuilding
twice and diffing nodes, edges, hierarchy, papers and meta.

**Agreement is unmoved**, as with the three structural corrections before it:
Disbiome −0.0024 (p=0.689), Peryton −0.0006 (p=0.881). The null there is
deliberately mismatched and therefore conservative — it drops 12 *random*
papers, deleting their evidence outright, where dedup deletes only redundant
evidence. This is a **correctness** fix and must not be cited as an accuracy
gain.

---

## The honest answer, with power

On the corrected graph the co-occurrence effect is **null**, in every variant:

| variant | pooled | p | per-edge | p |
|---|---:|---:|---:|---:|
| all edges, null A (within disease) | +0.0023 | 0.794 | +0.0137 | 0.141 |
| all edges, null B (sign flip) | +0.0023 | 0.513 | +0.0137 | 0.061 |
| balanced edges only (37, ≥2 a side) | −0.0034 | 0.757 | +0.0080 | 0.405 |
| permute within profile-size quintile | +0.0023 | 0.610 | +0.0137 | 0.066 |
| permute within country | +0.0023 | 0.832 | +0.0137 | 0.195 |
| permute within sequencing type | +0.0023 | 0.825 | +0.0137 | 0.105 |
| Jaccard instead of cosine | +0.0013 | 0.900 | +0.0088 | 0.125 |
| per-edge median (outlier-robust) | +0.0023 | 0.791 | +0.0009 | 0.353 |
| rank-based per-edge AUC | — | — | 0.5347 | 0.206 |

**66 of 130 edges are positive — 51%, chance.** Before the fix it was 79 of 138
(57%) with a per-edge median of +0.0150; after, the median is +0.0009.

**Power.** At 130 contested edges the minimum detectable per-edge effect is
≈ 0.017 and the minimum detectable |AUC − 0.5| is ≈ 0.053. The observed AUC
deviation is 0.035 — *below* what this corpus can resolve. So this is
"no effect visible at 130 edges / 1,848 pairs", not "no effect".

The one number that still reads borderline is per-edge under null B (p=0.061).
It is not enough: null A on the same statistic gives 0.141, the balanced subset
gives 0.405, the rank-based test gives 0.206, and nine variants were tested. A
single 0.06 among them is what multiplicity looks like.

---

## Secondary result: no *other* pseudo-replication is detectable

Exact duplicates are fixed, but edge weight assumes papers are independent
cohorts, and a same-cohort re-analysis would break that without any title
collision. Screened directly on the hand-curated cohort fields — papers sharing
country *and* n_cases *and* n_controls:

- 4 candidate groups, all coincidental (balanced Chinese designs — 45/45, 20/20,
  50/50, 8/8 — pairing papers on *different* diseases).
- **0 edges draw more than one paper from any candidate group.**

Null. **Power limit: only 192 of 272 papers (71%) carry a full cohort
signature**, so this rules out the blatant cases, not all of them. After dedup
the maximum profile cosine between any two papers is 0.816 (median 0.178), with
no cluster near 1.

**Two independent checks confirm the title dedup is complete**, which matters
because the fix keys on a normalised title and could have missed a copy whose
title differs more than punctuation:

- **By identifier.** 318 of 326 rows carry a resolvable PMID, PMC id or DOI.
  **No two rows share an identifier under different normalised titles** — so
  there is no duplicate the title key missed. (Nor would an identifier key have
  caught the original 12: their two copies were fetched under *different* links,
  which is how they arose. The two keys are complementary and now agree.)
- **By fuzzy title.** **No pair of distinct normalised titles exceeds 0.92
  similarity** — no "Alzheimer's Disease" versus "Alzheimer Disease" case is
  hiding below the exact-normalisation threshold.

---

## What this means for the graph

1. **35 edges were advertising replication they do not have.** That is the
   largest concrete defect found in the graph this month, and it was invisible
   to every check that reads titles.
2. **Contested edges remain unexplained.** Study design does not explain them
   (previous sessions), body site does not (previous session), and taxon
   co-occurrence does not either. Four explanatory variables, four nulls.
3. **The false positive was found by attacking a result, not by inspection.**
   The duplicate papers had survived a screening pass that read all 45
   title-matched full texts. A statistical anomaly pointed at them in one step.

## Next lever

The remaining Task 1 questions (does profile predict disagreement with
Disbiome/Peryton; are there taxon modules aligned with disease or geography) run
on the same substrate and are cheap, but they are the same shape as the four
questions that have now returned nulls, at the same n.

The binding constraint is **n**, not method. Every per-edge question here is
underpowered — 134 contested edges averaging ~5 papers, and the minimum
detectable effects above are set by that, not by the statistic. More papers is
the only lever that changes them, and extraction needs a GPU.
