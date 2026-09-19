# Knowledge graph — microbe–disease associations

Built from the corpus-scale extraction over the screened 326-paper corpus
(Emily's 250 usable papers plus the screened MAIN_DATA expansion).

## Pipeline

```
extractions_screened.json                        (extraction, 326 rows -> 314 after dedup)
   -> build_kg.py    -> graph.json   (nodes + aggregated edges)
   -> build_viz.py   -> kg.html      (self-contained explorer)
```

## The graph

| | |
|---|---:|
| taxon–disease edges | 2,034 |
| distinct taxa | 925 |
| diseases (normalized) | 40 |
| edges seen in >1 paper | 440 |
| **contested** (papers disagree on direction) | **217** |
| containment links | 723 |
| rank-placeholder nodes | 100 |
| papers contributing ≥1 association | 272 / 326 |

**Confidence tiers** (`annotate_confidence` in `build_kg.py`, added 2026-09-09).
Every edge is tiered from its own properties, and each tier carries a *measured*
agreement rate with the two curated databases rather than an asserted one:

| tier | edges | share | Disbiome | Peryton |
|---|---:|---:|---:|---:|
| well-supported (≥3 agreeing papers) | 75 | 3.7% | **93.8%** | **93.3%** |
| supported (2 agreeing papers) | 135 | 6.6% | 77.8% | 83.3% |
| provisional (1 paper, or a `discriminating` taxon) | 1,607 | **79.0%** | 66.1% | 61.9% |
| contested (papers disagree; no direction asserted) | 217 | 10.7% | — | — |

Monotone in both databases. The cuts were chosen after seeing the Disbiome split,
so Peryton is the out-of-sample check. **79% of this graph is provisional** — that
is the case for more papers, quantified, and it is the honest thing to show a
reader looking at any one edge.

*Updated 2026-09-03. Three structural corrections that session — 12 duplicate
papers removed, three NMDAR disease nodes folded into one, and 32 more SILVA
rank placeholders split out of their parents — changed these counts; see
`SESSION_LOG.md`. None of them moved agreement with the curated databases, and
none should be cited as an accuracy improvement.*

## Design decisions, and why

**Edge weight is evidence count, not effect size.** The extractor returns direction only. Even
with effect sizes the source papers report incommensurable statistics (LEfSe LDA scores,
fold-changes, p-values) that cannot be pooled into one magnitude — a unified "strength" number
would be invented precision. Bar length = number of papers.

**Contested edges are kept, never merged.** 217 pairs are contested — 215 with papers
pointing both ways, plus 2 where a single paper contradicts itself. *(Was written as 151,
a count from three corpus revisions ago; corrected 2026-09-10.)* The
microbiome replication literature reports ~1 taxon in 3 flipping sign between cohorts, so
disagreement is a finding about the evidence base, not noise. Disbiome and Peryton both store
conflicting entries separately for the same reason.

**Direction is encoded by position AND color** (depleted left / enriched right), so the chart
survives colorblindness, greyscale and print. Colors are a validated diverging pair — blue/red
poles with a neutral gray midpoint, ΔE 18.5 under protanopia. Red/green was rejected: ~8% of men
have red-green colorblindness.

**Ranks are preserved, not collapsed.** Papers report phylum, genus, species and OTU-level labels
as peers; there is no accepted convention for merging them. Rank is a node attribute.

**Not a node-link diagram.** 2,034 edges over 925 taxa is a hairball that answers no question. The
question the data serves — "for this disease, which taxa, how replicated, where do papers
disagree" — is a diverging bar chart.

## Known gaps

*The first and last bullets here described the graph as it stood before taxonomy
resolution and external validation landed. Corrected 2026-09-10.*

- **254 of 925 taxa never resolve to an NCBI taxid.** The rest do — `taxonomy.py`
  reads `names.dmp` directly and folds synonyms (Bacteroidetes + Bacteroidota → 976,
  Firmicutes + Bacillota → 1239). What does not resolve is mostly 16S clade labels
  (`[Eubacterium] ventriosum group`), including 100 deliberately-split SILVA rank
  placeholders (`Prevotella 9`).
- **Diseases carry MONDO ids only for the 16 mapped patterns**; anything else keeps its cleaned
  label with `mondo: null` rather than being dropped.
- **Associations only.** No causal claim, no direction of causality.
- **Validated against Disbiome (73.0%) and Peryton (72.5%)** — but those are *not*
  independent replication. 43 of our 272 papers are cited by Disbiome and 24 by
  Peryton, and agreement splits hard on that line (87.5%/96.8% shared-source vs
  58.1%/52.6% disjoint). See `FINDINGS_independence.md`.
- **Reading fidelity, measured without the gold or either database: ≥86.6%**
  (181/209, 95% CI [81.7, 91.3], 122 papers). `audit_direction_witness.py` goes
  back to each paper's own sentences for every observation backing an edge and
  compares the direction words beside the taxon with what we extracted. All 28
  residual disagreements were adjudicated twice independently (25/28 exact
  agreement) and **none is an extraction error** — 12 attribute the abundance to
  the control group, 9 compare something other than disease-vs-control, 4 carry a
  direction belonging to a different taxon — so 86.6% is a **lower bound**. This
  is a different quantity from the 73%/72.5% headline and **must not be quoted as
  moving it**. See `FINDINGS_direction_audit.md`.
- **Precision of SCOPE, 2026-09-19 — the one property no other instrument here
  scores.** Everything above asks whether an edge's *taxon and direction* are
  right. This asks whether the *comparison* was admissible: `samgated-v1` allows
  only disease vs healthy control, so a taxon genuinely higher in ICH survivors
  than in ICH deceased is a correct reading of its paper, scores as a hit
  everywhere above, and still should not be an edge. All 271 contributing papers
  read against their own sentences by adjudicators **blinded** to the disease
  label and to everything graph-side. **2,871 of 3,077 observations (93.3%) are
  confirmed disease-vs-healthy-control; out-of-gate is bounded at 1.8%–2.7% —
  quote the range.** Out-of-gate papers disagree with the leave-one-out
  literature **1.75×** as often (O/E 1.747 vs 0.989, p = 0.00015 over 20,000
  paper-level permutations, MDE ±0.303; leave-one-paper-out worst p = 0.00070;
  the 8 cleanly-out papers alone give 1.460 at p = 0.0118). That is the project's
  first **validated** quality flag — independent in the way the 2026-09-17
  provenance tier was not — and it is worth ~6 excess disagreements, so it is
  **a flag, not an accuracy gain.** Two named errors fell out, both on the
  Alzheimer's node and both **open, waiting on a human**: the SILCODE amyloid
  paper contributes 13 edges and no subject in it has Alzheimer's, and a second
  contributes 7 from AD-with vs AD-without neuropsychiatric symptoms. See
  `FINDINGS_contrast_scope.md`; the tiered list ships opt-in as
  `contrast_out_of_gate.json` and `graph.json` is untouched.
- **Disagreement with the rest of the literature is a property of the PAPER, and
  nothing we extract explains it.** Scored against the leave-one-out majority, 377
  of 1,367 decisive observations (27.6%) disagree, and which papers hold the
  minority direction is clustered far beyond a within-edge null (p = 0.0003;
  p = 0.0013 after dropping within-paper taxonomic relatives). No paper is
  systematically *inverted* — that is a well-powered null, 0 of 134 survive BH and
  a fully inverted copy would have been caught for 81 of them — but country,
  cohort size, sequencing platform, 16S region, medication and diet control, and
  disease identity are all null at MDEs of 16–22%, and so are 15 wet-lab and
  bioinformatics variables added in a second pass (extraction kit, pipeline,
  OTU-vs-ASV, LEfSe vs DESeq2, rarefaction, platform, publication year) —
  **24 variables, 24 nulls**, now **27**: the textual-provenance test of
  2026-09-12 (whether the taxon is named in an own-result sentence or only in
  background/citation: 27.6% vs 27.1%, pooled difference −0.6 points at an MDE of
  8.4; `FINDINGS_direction_audit.md`), and the two contrast-scope variables of
  2026-09-19 — whether a paper ALSO reports a within-disease subgroup contrast
  (94 of 241 in-scope papers do; O/E 1.006 vs 0.973, p = 0.650, MDE ±0.140) and
  whether its disease label is free text (p = 1.000, MDE ±8.5 points;
  `FINDINGS_contrast_scope.md`). Being *out* of gate is the one predictor that
  is NOT null. The magnitude explains why: the paper-level SD of
  discordance is only **3.4 percentage points** on a 27.6% base
  (cluster-bootstrap CI [0.0, 6.0]) against MDEs of ±4–7 points, so this corpus
  *cannot answer* whether kit or pipeline matters. Corollary: **~83% of the
  variance in disagreement is edge structure, not paper identity.** See
  `FINDINGS_paper_discordance.md`.
- **`methods_metadata.py` extracts nine families of study-methods variables for
  all 272 papers (recall 0.90 against an independent read) and nothing consumes
  them yet.** They are weak predictors of discordance but good provenance for a
  reader judging an edge.

## Disease ontology (`mondo.py`) — the disease-side analog of `taxonomy.py`

The taxon half of every edge resolves against NCBI; until 2026-09-14 the disease
half resolved against nothing but a hand-written table of 16 ids. `mondo.py`
parses the MONDO release (cached at `~/.mondo/mondo.obo`, **not committed**,
53 MB) and resolves **28 of the 40** disease labels by exact label/synonym match
plus 3 curated aliases, with 12 documented refusals. Fetch the .obo with:

```bash
mkdir -p ~/.mondo && curl -sSL -o ~/.mondo/mondo.obo \
  https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo
python3 mondo.py --validate      # positive control against build_kg.DISEASE_MAP
```

`ftp.ncbi.nih.gov`, `eutils`, `purl.obolibrary.org` and `ebi.ac.uk` are all
blocked in the cloud environment; **the MONDO GitHub release is not.**

Three things to know before touching this:

- **Do not give `Mild cognitive impairment` a MONDO id.** MONDO has no term of
  that name (0 of 104,643 index keys); `None` is the correct value. It previously
  carried `MONDO:0005453` = *congenital heart disease*, live on 154 edges.
- **Resolution is exact-match only.** Unresolved labels are reported, never
  guessed — edit distance would merge `Cognitive impairment` into `specific
  language impairment`. Aliases live in `mondo.ALIASES` with reasons; refusals in
  `disease_hierarchy.REFUSED_ALIASES`.
- **`mondo.py --validate` reads `DISEASE_MAP` live from `build_kg.py`**, so it
  detects drift in the table it checks. Keep it that way.

| script | what it does |
|---|---|
| `mondo.py` | MONDO parser/resolver, is-a DAG, self-test + positive control |
| `disease_hierarchy.py` | grades the hand-written tiers against MONDO, derives the link layer, ontology-shuffled null |
| `disease_hierarchy_power.py` | pair-clustered null and the MDE arithmetic (4 pairs needed; 2 exist) |
| `disease_cluster_coherence.py` | do clinically adjacent nodes behave like one disease? + ubiquity confound + negative control |

Findings: `FINDINGS_disease_ontology.md`. Headline for a reader: the 71-paper
cognitive-decline cluster **does not cohere** (0.592 vs a 0.672 background, every
MCI pair at or below chance), so those nodes should be linked for *retrieval* and
their evidence should **not** be pooled.

## RAG layer (`build_rag.py`)

`rag_corpus.jsonl` — one document per graph edge, ready for any vector store
(Chroma, FAISS, pgvector, LanceDB): embed `text`, keep `meta`, filter on `meta.*`.

Chunking per **edge**, not per paper: the edge is the unit of claim. A paper-level
chunk buries "Akkermansia is enriched in Parkinson's" inside 50k characters of
methods; an edge-level chunk states it, says how many papers agree, how many
disagree, and names them — which is what you want a model quoting.

```bash
python build_rag.py                                        # build corpus
python build_rag.py --query "what is depleted in parkinson's" -k 5
python build_rag.py --query "contested findings in Alzheimer's"
```

### Retrieval is hybrid, because pure BM25 measurably failed

First version indexed the paper titles and ranked by BM25 alone. Asking "what
bacteria are depleted in parkinson's disease" returned **NMDAR encephalitis**:

| token | df | idf |
|---|---:|---:|
| `parkinson's` | 289 | 1.58 |
| `bacteria` | 53 | **3.26** |
| `are` | 68 | **3.02** |

The NMDAR paper is titled *"Disturbance of Gut **Bacteria** and Metabolites **Are**
Associated…"*, so its incidental title vocabulary outscored the actual disease —
disease names are common *inside* this corpus, so their idf is low. Three fixes:

1. **Titles are not indexed** (they stay in the display text).
2. **Stopwords removed.**
3. **Entities matched explicitly.** Every disease and taxon is known exactly, so a
   named entity in the query is a hard filter, not a bag-of-words hint. Direction
   words ("depleted", "enriched") filter too, and an edge whose *majority*
   direction matches ranks above a contested edge that only partly matches.

This is more accurate and cheaper than embeddings for a 1.4k-doc corpus of proper
nouns. Swap in dense retrieval over the same `text` field if paraphrase matching
is later needed; the corpus format does not change.

## Validation against Disbiome (`validate_disbiome.py`)

Disbiome (https://disbiome.ugent.be) is a hand-curated microbe–disease database:
~10.9k experiments, each recording a taxon Elevated or Reduced in a disease vs
healthy controls. Its API is open (`:8080/experiment`) and the response is cached
to `disbiome_experiments.json`.

**11 diseases overlap** with our corpus (Parkinson's, Alzheimer's, MS, ALS, stroke,
Huntington's, MCI, epilepsy, migraine, myasthenia gravis, neuromyelitis optica).

| | |
|---|---:|
| pairs in both | **268** |
| of our in-scope pairs corroborated | 268/1282 (20.9%) |
| of Disbiome's pairs we recovered | 268/506 (**53.0%**) |
| direction **agreement** (both decisive) | **127/174 (73.0%)** |
| direction disagreement | 47 (26.9%) |

*(Peryton, same join: 223 overlapping pairs, 73.4% recall, direction agreement
100/138 = **72.5%**.)*

> **These two numbers are NOT independent replication, and should not be quoted as
> such** (measured 2026-09-09, `FINDINGS_independence.md`, `check_independence.py`).
> 43 of our 272 papers are also cited by Disbiome and 24 by Peryton; because those
> are the heavily-reported papers they back **half** the decisive pairs. Agreement
> splits on that line — **87.5% / 96.8%** where both sides read the same paper,
> **58.1% / 52.6%** where the literature is disjoint (taxon-block permutation
> p=0.0001, taxon cluster bootstrap CI excluding 0, and it survives stratifying on
> evidence count). Disease is a confounder and pooling overstates it; held fixed
> within Parkinson's, both databases independently give **59.0%** and **59.6%**
> disjoint against 100% and 95.8% shared. In Multiple sclerosis the gap is absent
> (72.7 vs 70.6, n=39, MDE 28.7) — heterogeneity, not power.
>
> The honest split: **extraction fidelity 85–97%** (same paper, two readers — and
> the best evidence for the extractor that does not lean on the in-house gold),
> **literature reproducibility ~53–59%** (disjoint sources). 73% is a mixture of
> the two in a ratio set by how much of our corpus the curators happened to read.
> Report agreement stratified, not pooled.

*These figures are measured with the replay taxonomy cache
plus `species_synonyms.json`, not the NCBI taxdump — this environment's network
policy denies `ftp.ncbi.nih.gov` — so they run ~0.2–1.1 points off taxdump-measured
runs and are sound for before/after deltas rather than as new absolute numbers.*

*The 2026-09-08 species split moved these, and the counts are better evidence than
the ratios: **11 decisive Disbiome pairs entered, all 11 agreeing, 0 disagreements
added and 0 verdicts flipped** (Peryton +1, agreeing). Eleven-for-eleven is
p = 0.026 against a 0.717 baseline, but the 11 come from only **7 distinct taxa**,
and clustering on taxon gives **p = 0.097** — suggestive, not significant. Cite the
coverage gain (+9 net decisive pairs, no new disagreement), not an accuracy gain;
the correction is justified on correctness of meaning. See
`FINDINGS_species_split.md`.*

**Normalize both sides with the same resolver — do not trust their taxid.**
Disbiome's `organism_ncbi_id` is not consistently at the rank the paper reported:
for a paper saying just "Prevotella" its curators recorded **59823** (*Prevotella
sp.*, a SPECIES) where the genus is **838**. Joining on their stored id silently
missed Prevotella/Parkinson's — one of the most replicated findings in the field
(16 papers here, and 3-of-4 Reduced in Disbiome, i.e. an *agreement*). Re-resolving
their `organism_name` through our own resolver lifted overlap 188 → 238 and recall
41.4% → 53.6%. A join is only meaningful when both sides share a normalizer.

**How to read these numbers.** None is pure accuracy. 22% of ours being corroborated
is mostly a coverage difference — Disbiome curates a different, partly older paper
set, and 846 of our pairs are simply absent from it (the top ones by evidence, like
*Faecalibacterium* depleted in Alzheimer's at 14 papers, look like Disbiome gaps
rather than our errors). The 206 pairs they have and we lack are our recall gap.
The 34 direction disagreements are the useful output: specific, checkable claims,
and in most of them our edge rests on more papers than their single record.

### Peryton
Not validated. Peryton (https://dianalab.e-ce.uth.gr/peryton/) is a client-side JS
app with no reachable data endpoint — every API path probed returns 404, and the
page ships ~1 KB of HTML. It needs a manual download through a browser; drop the
export beside this script and the same join logic applies.
