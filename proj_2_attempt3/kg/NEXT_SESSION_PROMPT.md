# Session prompt — KG usefulness + embeddings

Paste everything below into a fresh Claude Code session in `/Users/mohak/Desktop/Lab Work`.

---

You are continuing the Knight Lab microbe–disease knowledge graph. Read
`CLAUDE.md` and `proj_2_attempt3/kg/README.md` first. The graph is built,
validated and published (https://www.mohakprakash.com/KnightLabV2/). Everything
below runs **locally — no GPU, no API credits**.

## Ground rules

- **Permutation-test everything.** This corpus is small and has already produced
  two false positives that survived until tested: 198 "explanatory" terms that a
  random split matched (p=0.41), and a `diet_controlled` result that went from
  p=0.002 to FDR 0.243 once clustering and multiple testing were handled.
  Observations are **not independent** — 533 come from only 136 papers, so shuffle
  labels at the **paper** level.
- **Report honest nulls.** A null with a power statement is a result. A cluster
  without a permutation test is not.
- Verify by **executing**, not by "it parses". A prior blank-canvas bug passed
  every static check.
- Commit as you go with real reasoning in the messages.

## TASK 0 — Rebuild on the correct datasheet (do this first, everything depends on it)

We discovered we have been scoring against the **wrong export**.

- Used: `EmilySong_GoldStandardPaper/ALL_EMILY_PAPERS_WITH_(inGoldStd)_COLUMN.csv`
  — 340 rows, **218 (64%) blank on both taxa columns**.
- Correct: `kg/Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv`
  — 337 rows, **only 5 (1%) blank**, and it additionally carries hand-curated
  `Year`, `Country`, `Continent`, `SequencingType`, `PublicationJournal`,
  `Differential Abundance Test`.

Do:
1. Rebuild the paper set and gold from the correct sheet (246 of our 250 papers
   are in it; reconcile the other 4 and the 7/4 title deltas between sheets).
2. **Re-score the extraction.** Every F1 on record is suspect. Expect the "0.390
   over all 250" artifact to disappear, since it was caused by scoring against
   blank cells. Report old vs new side by side.
3. **Cross-check the LLM metadata pass.** `kg/metadata.jsonl` extracted `country`
   and `sequencing` with an LLM; the sheet has them hand-curated at 99%. Compute
   agreement. This is a free, real accuracy measurement of an LLM extraction
   against human labels — report it as such, and note where the LLM was *right*
   and the human wrong, if that happens.
4. Rebuild `graph.json`, `kg.html`, `docs/index.html`, and re-run
   `validate_external.py`. Report how the Disbiome/Peryton numbers move.

## TASK 1 — Taxon-vocabulary features (the promising lead)

Full-text vocabulary is ~8,800 distinct terms per 12 papers, which is hopeless at
this n. Restricting to terms that resolve to an NCBI microbial taxon via
`kg/taxonomy.py` cuts it to **122 — a 72× reduction** — and the surviving feature
is meaningful: *which other taxa does this paper report*, i.e. a co-occurrence
profile.

Build a paper × taxon incidence matrix over all papers and test:
- **Pooled across edges** (the version with power, ~530 observations): do papers
  reporting *enrichment* differ from papers reporting *depletion* in their taxon
  co-occurrence profile? Cluster-robust permutation at paper level.
- **Per contested edge** (expect underpowered; report the power honestly): within
  a fixed taxon–disease pair, do the up-papers and down-papers separate?
- Does co-occurrence profile predict **disagreement with Disbiome/Peryton**?
- Are there taxon *modules* — groups reported together — and do they align with
  disease, geography, or sequencing type (all now available from the sheet)?

## TASK 2 — Embeddings, done properly

Only after Task 1, and only where Task 1 shows signal worth pursuing.

- Embed full texts locally (`sentence-transformers`, e.g. `all-MiniLM-L6-v2` or a
  biomedical model). CPU is fine for ~340 documents.
- **Control for disease.** The dominant axis will be "which disease this paper is
  about", which is already known and useless. Compare within-disease or residualise.
- Compare embedding-based separation against the taxon-vocabulary baseline from
  Task 1. **If the interpretable baseline does as well, prefer it** — 768 unnamed
  dimensions cannot answer "what is the explanatory variable", which is the actual
  question being asked.
- Sanity check: do embeddings reproduce the known structure (same disease cluster,
  same sequencing type cluster)? If not, they are not encoding what we need.

## TASK 3 — Make the graph useful

Concrete, in rough priority order:

1. **Adjudicate the 11 pairs contradicted by BOTH Disbiome and Peryton** —
   *Erysipelotrichaceae* and *Paraprevotella* in Parkinson's, *Dorea* in
   Alzheimer's, and 8 more. Two independent curations disagreeing is the strongest
   error signal available. Read the source papers and determine who is right.
2. **Use the containment links.** 551 exist and nothing consumes them yet. Are
   some contested edges actually *rank confusion* — a family and a genus inside it
   being conflated? This is now answerable and nobody has asked it.
3. **Expand the corpus.** ~47 recoverable Emily papers (43 of the 90 unused have
   no link) plus ~57 neuro-titled papers in `MAIN_DATA.json` that are NOT in our
   set (only 7 of 2,026 overlap). Filter to human case-control studies first.
   Realistic ceiling ~350 papers. Extraction needs a GPU — **ask before spending**.
4. **Ship something a biologist would use**: per-disease evidence summaries,
   a "what's contested and why" view, exportable citations.

## What NOT to do

- Don't re-run extraction on the 250 — it is done and cached.
- Don't trust `in_gold_standard`; it holds the *strings* `'Yes'`/`'No'`, and
  `'No'` is truthy in Python. That bug already produced a wrong claim once.
- Don't join on an external database's taxid. Disbiome records "Prevotella" as
  59823 (*Prevotella sp.*, species) where the genus is 838. Both sides must pass
  through `kg/taxonomy.py`.
- Don't collapse taxonomic ranks. In Parkinson's, *Lachnospiraceae* is depleted
  across 15 papers while *Hungatella* inside it is enriched across 7.

## Deliverable

A written findings doc in `kg/`, committed, that a PI could read: what was tested,
what survived correction, what did not, and what the next lever is. State power
limits plainly — with 174 contested edges averaging ~4 papers per side, most
per-edge questions cannot be answered at this corpus size, and saying so is more
useful than a cluster that does not replicate.
