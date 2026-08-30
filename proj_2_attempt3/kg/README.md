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
