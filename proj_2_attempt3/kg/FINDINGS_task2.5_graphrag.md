# GraphRAG: personalized PageRank retrieval over the KG

*`graphrag.py` (retriever), `compare_retrieval.py` (evaluation),
`retrieval_comparison.json` (numbers). Run `python graphrag.py --compare`.*

## What was built

Retrieval as graph traversal rather than keyword scoring:

1. **Entity-link the query** against the closed vocabulary — 832 taxon nodes with
   their aliases, 43 disease nodes — by exact longest-match. Disease surface forms
   ("PD", "Parkinson's") fold through the same map `build_kg.py` uses.
2. **Personalized PageRank** from the matched nodes (damping 0.85, power iteration
   to 1e-10). ~900 nodes, milliseconds, no library.
3. **Return a connected subgraph** — seeds, high-PPR neighbours, the edges between
   them with direction and evidence counts, the containment links used, and the
   papers backing each edge.

Two decisions carry the behaviour:

- **Containment links are traversable** (625 of them, previously consumed by
  nothing). Modest fixed weight, and *not* treated as evidence: containment is a
  taxonomy fact, so it should move probability without voting on direction.
- **Multi-entity queries use a product, not a union.** Seeding both diseases into
  one PPR run ranks taxa near *either* — mostly just the taxa near the bigger
  disease. Instead each seed gets its own run and nodes are ranked by the
  **geometric mean**, so a node scores well only if it is close to *all* seeds.

## The honest comparison: it is a tie, not a win

Ground truth is computed from the graph and fixed before either retriever runs — no
hand-labelled relevance, which would just measure my own expectations. Bridge truth
= taxa with ≥2 papers on **both** diseases.

| query | \|truth\| | GraphRAG P@10 | BM25 P@10 |
|---|---:|---:|---:|
| links Parkinson's ↔ Alzheimer's | 64 | 1.00 | 1.00 |
| links Multiple sclerosis ↔ Stroke | 25 | **1.00** | 0.80 |
| links Alzheimer's ↔ MCI | 8 | 0.40 | 0.40 |
| links MS ↔ ALS | 12 | **0.60** | 0.50 |
| Akkermansia (1 hop) | 14 | 1.00 | 1.00 |
| depleted in Parkinson's (directional) | 116 | 0.80 | **1.00** |
| **mean** | | **0.800** | 0.783 |

**0.800 vs 0.783 over six queries is a tie.** GraphRAG wins two, loses one, ties
three. Nobody should call this a retrieval-accuracy improvement.

### The premise on record was wrong

The plan asserted that "*What links Parkinson's and Alzheimer's?* is a graph query
and BM25 structurally cannot answer it." It can, and does, at P@10 = 1.00. The
reason: `build_rag.py` is **not pure BM25** — it already bolts on exact entity
matching and evidence weighting. Given two diseases it filters to edges about
either and ranks by paper count, and because hub taxa are bridges, that
approximates the graph answer. The structural-impossibility claim should be
retired.

### Two self-inflicted measurement errors, recorded so they are not repeated

- **The first metric was saturated.** Bridge truth as bare co-membership makes 125
  of 832 taxa "correct" for Parkinson's/Alzheimer's, so any ten well-connected
  hubs scored 1.00 and both systems tied trivially. Requiring ≥2 papers per side
  cut it to 64 (and to 8 for Alzheimer's/MCI) and the systems separated. A metric
  that cannot distinguish the systems is not evidence that they are equivalent.
- **PPR is direction-blind, and that was a real bug.** Proximity has no sign, so
  "what is *depleted* in Parkinson's" returned everything near Parkinson's,
  enriched taxa included — P@10 = 0.60, an outright loss to BM25. Direction is now
  an explicit filter on the seed disease, exactly as `build_rag.py` does, which
  lifted it to 0.80. It still loses this query type; keyword matching is simply
  good at it.

## Where the graph is genuinely irreplaceable: containment

The real differentiator is not ranking. Query **Hungatella**:

```
GraphRAG                                    BM25
  Lachnospiraceae depleted  in PD  16p        Hungatella enriched in PD   7p
  Hungatella      enriched  in PD   7p        Hungatella enriched in MS   2p
  Lachnospiraceae depleted  in AD  10p        Hungatella enriched in CI   1p
```

GraphRAG surfaces the parent family, depleted across 16 papers, next to the genus
inside it, enriched across 7 — the rank-conflict this project has flagged as
load-bearing. BM25 returns only Hungatella documents and *cannot* reach the family,
because no document contains both claims. A reader gets half the picture and would
not know to look.

**This is measurable, not anecdotal.** Across the graph:

- **903** (taxon, disease) edges have a parent that also has an edge in the *same*
  disease — i.e. containment context exists and is retrievable.
- **229 of those (25%) point in the opposite direction to their parent.**

Top conflicts by combined evidence:

| disease | parent | child |
|---|---|---|
| Parkinson's | *Lachnospiraceae* depleted (16p) | *Hungatella* enriched (7p) |
| Parkinson's | *Oscillospiraceae* enriched (5p) | *Faecalibacterium* depleted (16p) |
| Parkinson's | *Prevotella* depleted (18p) | *Prevotella pallens* enriched (2p) |
| Multiple sclerosis | *Lachnospiraceae* depleted (9p) | *Blautia* enriched (9p) |
| Alzheimer's | *Bacillota* depleted (11p) | *Clostridia* enriched (6p) |

## Recommendation

**Ship GraphRAG as the retriever, but for its output, not its ranking.** The
defensible claims are: it returns a connected subgraph with direction, evidence
counts and provenance rather than a flat document list, and it is the only way to
retrieve the containment context that changes the interpretation of 229 edges.
The indefensible claim would be that it retrieves more relevant items than BM25 —
it does not, at n=6 queries.

Keep `build_rag.py` as the baseline. It is genuinely better at directional
one-hop queries and should not be deleted.

## Caveats

- **Six queries is a small evaluation.** The 0.017 mean difference is noise; treat
  the per-query wins as illustrative, not established.
- Ground truth is defined *from the graph*, so it measures internal consistency —
  whether a retriever finds what the graph contains — not biological correctness.
- The bridge metric rewards recall of a large set at k=10; recall@10 is capped at
  10/64 for the biggest truth set, so precision is doing most of the work.
