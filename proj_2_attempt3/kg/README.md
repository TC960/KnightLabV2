# Knowledge graph — microbe–disease associations

Built from the corpus-scale extraction over Emily's 250 usable papers.

## Pipeline

```
eval-v2/results/qwopus3.5-27b-v3__q4km__samgated-v1__all250.json   (extraction, 250 papers)
   -> build_kg.py    -> graph.json   (nodes + aggregated edges)
   -> build_viz.py   -> kg.html      (self-contained explorer)
```

## The graph

| | |
|---|---:|
| taxon–disease edges | 1,556 |
| distinct taxa | 873 |
| diseases (normalized) | 27 |
| edges seen in >1 paper | 323 |
| **contested** (papers disagree on direction) | **151** |
| papers contributing ≥1 association | 211 / 250 |

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

**Not a node-link diagram.** 1,556 edges over 873 taxa is a hairball that answers no question. The
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
