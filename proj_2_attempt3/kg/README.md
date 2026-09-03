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
| taxon–disease edges | 2,011 |
| distinct taxa | 918 |
| diseases (normalized) | 40 |
| edges seen in >1 paper | 438 |
| **contested** (papers disagree on direction) | **219** |
| containment links | 708 |
| rank-placeholder nodes | 100 |
| papers contributing ≥1 association | 272 / 326 |

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

**Contested edges are kept, never merged.** 151 pairs have papers pointing both ways. The
microbiome replication literature reports ~1 taxon in 3 flipping sign between cohorts, so
disagreement is a finding about the evidence base, not noise. Disbiome and Peryton both store
conflicting entries separately for the same reason.

**Direction is encoded by position AND color** (depleted left / enriched right), so the chart
survives colorblindness, greyscale and print. Colors are a validated diverging pair — blue/red
poles with a neutral gray midpoint, ΔE 18.5 under protanopia. Red/green was rejected: ~8% of men
have red-green colorblindness.

**Ranks are preserved, not collapsed.** Papers report phylum, genus, species and OTU-level labels
as peers; there is no accepted convention for merging them. Rank is a node attribute.

**Not a node-link diagram.** 2,011 edges over 918 taxa is a hairball that answers no question. The
question the data serves — "for this disease, which taxa, how replicated, where do papers
disagree" — is a diverging bar chart.

## Known gaps

- **Taxa are surface strings, not NCBI taxids.** Case and rank prefixes are folded; synonyms are
  not (Bacteroidetes/Bacteroidota remain distinct nodes). Wiring in `taxonomy_match.TaxResolver`
  would fix this and needs taxonkit + the NCBI taxdump.
- **Diseases carry MONDO ids only for the 16 mapped patterns**; anything else keeps its cleaned
  label with `mondo: null` rather than being dropped.
- **Associations only.** No causal claim, no direction of causality.
- Not yet validated against Disbiome / Peryton — that is the obvious next step.

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
| pairs in both | **264** |
| of our in-scope pairs corroborated | 264/1254 (21.1%) |
| of Disbiome's pairs we recovered | 264/501 (**52.7%**) |
| direction **agreement** (both decisive) | **120/167 (71.9%)** |
| direction disagreement | 47 (28.1%) |

*(Peryton, same join: 220 overlapping pairs, 72.8% recall, direction agreement
100/138 = **72.5%**.) These figures are measured with the replay taxonomy cache,
not the NCBI taxdump — this environment's network policy denies
`ftp.ncbi.nih.gov` — so they run ~0.2–1.1 points off taxdump-measured runs and
are sound for before/after deltas rather than as new absolute numbers.*

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
