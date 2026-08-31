# Rescoring against the correct datasheet

*All numbers reproducible from `rescore_correct_sheet.py`, `analyze_drop.py`, `analyze_drop2.py`.*

## 1. We had been scoring against the wrong export

| sheet | rows | blank on BOTH taxa columns |
|---|---:|---:|
| `ALL_EMILY_PAPERS_WITH_(inGoldStd)_COLUMN.csv` (used until now) | 340 | **218 (64%)** |
| `Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv` | 337 | **5 (1%)** |

Every F1 previously reported used the first one.

## 2. Corrected scores

Metric: char-ngram cosine ≥ 0.5. (The taxonomy/LCA metric needs linux-only binaries,
so it is unavailable on this machine — figures here are **not** comparable to the
0.753 quoted earlier, which used the taxonomy metric on the GPU box. Compare only
within this table.)

| gold | subset | n | P | R | **F1** |
|---|---|---:|---:|---:|---:|
| old | all papers | 250 | 0.258 | 0.641 | **0.368** |
| **correct** | all papers | 249 | 0.566 | 0.616 | **0.590** |
| old | papers with non-blank gold | 88 | 0.653 | 0.641 | **0.647** |
| **correct** | papers with non-blank gold | **227** | 0.577 | 0.616 | **0.596** |
| old | test_set_v2 benchmark | 15 | 0.517 | 0.847 | 0.642 |
| correct | test_set_v2 benchmark | 15 | 0.522 | 0.848 | 0.646 |

**The 0.390 artifact was self-inflicted and is gone.** It came from scoring
extractions against blank cells, which counts every extraction as a false positive.

**But the 0.680 figure was flattering.** The scoreable set grew 88 → 227 papers and
F1 *fell* to 0.596. Honest current performance is **~0.59, not ~0.68**.

Extraction is nonetheless doing real work: permutation on F1 gives observed 0.596
against a null mean of 0.072 (1,000 permutations, **p = 0.001**).

## 3. Why the drop? It is the papers, not the annotation

Decomposed:

| | n | F1 |
|---|---:|---:|
| A. the old 88, scored under OLD gold | 88 | 0.647 |
| B. the old 88, scored under NEW gold | 88 | 0.644 |
| C. the 139 newly-scoreable papers, NEW gold | 139 | **0.560** |
| D. all 227, NEW gold | 227 | 0.596 |

- **annotation effect (A→B) = −0.003.** Re-annotating the *same* papers changed
  essentially nothing. The new sheet is not simply a stricter grader.
- **paper effect (B→C) = −0.084.** The 139 papers that were previously unscoreable
  are genuinely harder.

### …but that difference is not statistically established

Papers with larger gold sets score better (gold-size vs precision r = 0.168,
p = 0.015). Permuting the group label **within gold-size strata**, the F1 gap falls
to 0.068 with **p = 0.18** — not significant.

**Power:** the null SD of the F1 gap is 0.051, so the minimum detectable gap at
α = 0.05 is **0.099**. The observed gap is 0.084. *The test cannot resolve a gap
this size.* Report as "the newly-scoreable papers look harder, but at n = 227 we
cannot distinguish that from gold-set-size confounding."

## 4. LLM vs human metadata — the LLM is competitive, and the human errors have a pattern

`metadata.jsonl` holds `country` and `sequencing` extracted from full text by an
LLM. The correct sheet has both hand-curated. 246 papers have both.

| field | agreement |
|---|---:|
| country | **221/241 (91.7%)** |
| sequencing | **216/239 (90.4%)** |

Sampling the disagreements and reading the source text:

| paper | human | LLM | evidence | verdict |
|---|---|---|---|---|
| *Gut Microbiome Features of **Chinese** Patients Newly Diagnosed with AD* | United States | China | *"Sun Yat-sen Memorial Hospital, Sun Yat-sen University, Guangzhou, China"* — the cohort site. A co-author is at *"University of Virginia Health System… USA"* | **LLM right** |
| *Gut Microbial Ecosystem in Parkinson Disease* | Canada | Malaysia | *"University of Malaya, Kuala Lumpur, Malaysia"*, *"Monash University Malaysia"* | **LLM right** |
| *Metabolic modeling links gut microbiota to metabolic markers of PD* | Ireland | USA | re-analysis of a previously published cohort; modelling group and cohort differ | **ambiguous — definitional** |

**The two human errors share a cause: recording an author's affiliation rather than
the cohort's location.** In the first case the paper's own title says "Chinese
Patients" and the curator recorded "United States".

Relevance: a collaborator is preparing an error-rate report on the human
annotations. This is an independent, quantified check on two fields, and it shows
the disagreements are not uniformly human-correct. The `country` field also needs a
written definition — cohort location or corresponding-author affiliation — before
either party can be called wrong on cases like the third.

## 5. Reconciliation

- 1 paper in our 250 does not join the new sheet: *"Metagenome-assembled microbial
  genomes from Parkinson's disease fecal samples."*
- 5 papers present in the new sheet have blank gold, including *"Signature of
  Alzheimer's Disease in Intestinal Microbiome: Results From the AlzBiom Study"*.

## 6. What to do

1. **Requote everything as ~0.59, not 0.68.** The 0.680 was measured on an easier
   88-paper subset.
2. **Rebuild the graph from the correct sheet.** `graph.json` still derives from the
   old export.
3. **Install the taxonomy metric** (taxonkit + NCBI taxdump on a linux box) so
   corpus figures are comparable to the 0.753 benchmark number.
4. **Define `country`** in the datasheet before treating disagreements as errors.
