# Screening the 45 title-matched MAIN_DATA papers

*Reproduce with `prep_filter_snippets.py` → `filter_maindata.py` → `build_kg.py` →
`analyze_filter_effect.py`. Screen verdicts with evidence quotes: `maindata_screen.json`.
Paired test output: `filter_effect.json`.*

## The question

45 of the 348 papers entered the corpus by matching neuro keywords against
`MAIN_DATA.json` titles. Unlike the 303 datasheet papers they were never screened
for study design, and agreement with Disbiome and Peryton fell ~4 points when they
were added. The standing hypothesis: reviews and animal studies are contaminating
the graph, so filtering them should recover agreement.

**The screen confirms the contamination. The contamination is not what moved the
number.** Both halves of that matter.

## 1. Half the batch should not be in a human microbe–disease graph

All 45 full texts were read and classified on study design, each verdict carrying a
verbatim quote (`maindata_screen.json`). 22 of 45 (49%) fail:

| verdict | n | examples |
|---|---:|---|
| **KEEP** — human case-control w/ healthy arm | 23 | *Metagenomics of PD* (490 PD vs 234 controls) |
| DROP — animal | 15 | 3xTgAD mice, R6/1 HD mice, Wistar rats, germ-free recolonisation |
| DROP — no healthy control | 3 | 160 stroke patients stratified by outcome; MTT pre/post |
| DROP — case report | 2 | n=1 dementia; n=2 monozygotic twins |
| DROP — no primary cohort | 2 | a review; a Mendelian-randomisation GWAS re-analysis |

These are not extraction errors. The extractor was asked which taxa move, not
whether the study is human case-control; pointed at 3xTgAD mice versus wild-type
littermates it faithfully returns that contrast, and the edge enters the graph as
though it were a human finding.

## 2. Filtering them changed the agreement rate by nothing at all

The naive comparison — run `validate_external.py` per variant and read the headline
— is confounded: dropping papers drops whole diseases, so Disbiome's reference
denominator moves 506 → 364 and the variants score *different question sets*.

So the comparison is paired per (taxid, disease), restricted to pairs decisive in
both variants, tested with exact McNemar (`analyze_filter_effect.py`):

| comparison | pairs in common | agree before | agree after | flips | p |
|---|---:|---:|---:|---:|---:|
| Disbiome, all348 → screened | 167 | 122 (73.1%) | 122 (73.1%) | 0 | 1.00 |
| Peryton, all348 → screened | 134 | 96 (71.6%) | 96 (71.6%) | 0 | 1.00 |
| Disbiome, all348 → no_maindata | 147 | 112 (76.2%) | 112 (76.2%) | 0 | 1.00 |
| Peryton, all348 → no_maindata | 132 | 96 (72.7%) | 96 (72.7%) | 0 | 1.00 |

**Zero decisive pairs flipped, in any comparison.** This is not an underpowered
null — it is an exact zero. Removing the animal and review papers did not change
our answer on a single taxon–disease pair that both we and a curated database call
decisively.

The reason is mechanical: the 22 failing papers contribute almost nothing. They
remove 11 of 1,927 edges, and only 4 of 285 contributing papers — 18 of the 22
produced no usable extraction at all.

## 3. What actually moved the headline number: disease mix

The headline rate moves because pairs *enter and leave* the decisive set, not
because any pair changes its answer. Dropping all 45 MAIN_DATA papers removes 21
decisive pairs:

- **19 of the 21 are Autism spectrum disorder.** The MAIN_DATA batch effectively
  introduced autism into the graph.
- We agree with Disbiome on **10/21 (48%)** of the departing pairs, versus
  **112/147 (76%)** of the pairs that stay.
- 17 of the 21 rest on 1–2 papers, and every one is supported *only* by MAIN_DATA
  papers.

So "agreement fell 4 points" = "we added a new disease whose thinly-supported edges
agree with curated databases at roughly chance." It is a composition effect.

### …and the autism effect is NOT established

Per-disease agreement with Disbiome: Parkinson's 79.7% (n=79), ALS 81.2% (n=16),
MS 68.6% (n=35), **ASD 52.6% (n=19)**, Alzheimer's 50.0% (n=8).

Tempting, and it does not survive testing. Those 19 ASD pairs come from just
**5 papers**, so the observations are heavily clustered. Permuting the group label
at the **paper** level (10,000 permutations):

- observed gap **−0.225**
- null SD **0.139** → minimum detectable gap at α=.05 is **0.273**
- observed is **1.62 SD**, **p = 0.211**

**The test cannot resolve a gap this size at 5 contributing papers.** Report as
"ASD edges look weaker, but at n=5 papers this is indistinguishable from noise."
Shuffling at the pair level instead of the paper level would have returned a
comfortable false positive here — the same trap that produced the earlier
`diet_controlled` and "explanatory terms" artifacts.

## 4. What was done anyway, and why

The screened corpus is now the graph (`graph.json`, `kg.html`, `docs/index.html`):
**326 papers, 281 contributing, 832 taxa, 1,916 edges, 225 contested, 625
containment links.**

The justification is **construct validity, not the metric**. A graph of human
microbe–disease associations should not carry edges whose evidence is a contrast
between transgenic and wild-type mice, regardless of whether removing them moves
agreement. The honest claim is: the filter makes the graph mean what it says it
means, and it costs nothing — it does not improve agreement, and nobody should
cite it as though it did.

## 5. Caveats

- **The taxonomy join is a replay cache, not the taxdump.** This environment's
  network policy denies `ftp.ncbi.nih.gov` (CONNECT → 403), so `taxonomy_cache.py`
  replays the resolution already recorded in `graph.json`. It reproduces the
  committed graph **exactly** — nodes, edges, all 625 containment links, paper
  table — verified by full rebuild diff. On the external join it is very slightly
  degraded (Disbiome overlap 269 vs 272 with the taxdump; Peryton 221 vs 221),
  because external databases use spellings absent from our alias set. All
  before/after comparisons above run under this *identical* join, so the deltas are
  sound; the absolute rates (72.6% / 71.6%) are ~0.2–1.1 points off the
  taxdump-measured 72.8% / 72.7% and should not be quoted as new absolute figures.
- 3 KEEP verdicts are borderline and were kept deliberately: *preclinical AD*
  (Aβ+ vs Aβ− cognitively normal — comparison arm is non-diseased, not healthy),
  the *ketogenic diet* epilepsy study (controls are the patients' parents,
  age-mismatched), and *MCI→AD conversion* (longitudinal, control arm thin).
- The 23 KEEP papers were not re-extracted; only the paper set changed.

## 6. Next lever

The autism edges are the weakest part of the graph — new disease, 1–2 papers each,
chance-level agreement — and they are also the cheapest to fix, because the
question is now specific: *is ASD genuinely less replicable, or did 5 papers land
badly?* That needs more ASD papers, not more analysis of these 5. Until then the
graph should surface per-disease evidence depth so a reader does not treat a
1-paper autism edge as equivalent to an 18-paper Parkinson's edge.
