# Session prompt — KG usefulness + embeddings

Paste everything below into a fresh Claude Code session in `/Users/mohak/Desktop/Lab Work`.

---

You are continuing the Knight Lab microbe–disease knowledge graph. Read
`CLAUDE.md` and `proj_2_attempt3/kg/README.md` first. The graph is built,
validated and published (https://www.mohakprakash.com/KnightLabV2/). Everything
below runs **locally — no GPU, no API credits**.

## Model policy (important — this controls cost)

You are the **orchestrator**. Stay on Opus for judgement: deciding what to test,
reading statistical output, catching when a result is noise, writing findings.

**Delegate all information-gathering to Haiku subagents** — `Agent` with
`model: "haiku"`. Haiku is for: reading papers and pulling quotes, grepping files,
tabulating, fetching, mechanical checks. Do not spend Opus tokens on retrieval.

Rule of thumb: if the task is "go find/read/count X", it is Haiku. If it is
"decide whether X means anything", it is you.

Subagents die often (this project has seen network drops and session limits kill
six in a row). So: give each one a NARROW task with a single output file, have it
write results to disk as it goes, and check for partial output before relaunching.
If agents keep failing, do the work yourself in-session rather than burning tokens
on repeated spawns.

## Mode: full agency, run until you run out

Work continuously. **Do not stop to ask permission** for anything local — reading,
writing, rebuilding, committing, pushing, running analyses. Do not end your turn to
report progress and wait; report *and keep going*. Pick the next task yourself when
one finishes. Keep going until the context/token budget is genuinely exhausted.

The only things worth pausing for: spending money (GPU instances), anything
outward-facing beyond this repo, or destructive operations outside `proj_2_attempt3/`.

Maintain `kg/SESSION_LOG.md` as you go — append a dated line per finding, including
the nulls. Commit after each meaningful step so nothing is lost if the session ends
abruptly. If a task turns out to be a dead end, write down *why* and move to the
next one; a documented dead end is a result.

Parallelise where tasks are independent. Task 0, Task 1 and Task 3.1 do not depend
on each other and can be fanned out to subagents. Task 2 depends on Task 1.

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

## STATE AS OF 2026-08-31 (already done — do not redo)

- Graph rebuilt on the CORRECT datasheet: **348 papers, 299 scoreable** (was 250/88),
  833 taxa, 43 diseases, 1,927 edges, 226 contested, 625 containment links.
- Revalidated: Disbiome **72.8%** agreement, Peryton **72.7%** (272 / 221 overlapping pairs).
- Rescoring finding: the old "F1 0.390" was an artifact of blank gold cells; the old
  "0.680" was flattering (measured on an easy 88-paper subset). **Honest F1 ~0.59.**
  Permutation confirms it is real work: observed 0.596 vs null 0.072, p=0.001.
- LLM-vs-human metadata: country 91.7%, sequencing 90.4% agreement over 246 papers.
  **CAVEAT: only 3 disagreements were actually read.** That is an anecdote, not a
  finding. Re-do at n>=20 before anyone cites it.

### Highest-value cleanup, do early
45 of the 348 papers came from MAIN_DATA by TITLE KEYWORD only and are unfiltered
for reviews and animal studies. Agreement with both curated databases fell ~4 points
when they were added. Filter them (human case-control studies with a healthy control
arm only), rebuild, and report whether agreement recovers. If it does, that is direct
evidence those papers are contaminating the graph.

## TASK 0 — Rebuild on the correct datasheet (DONE — see state above; skip)

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

## TASK 1 — Relation-bearing sentences (the promising lead)

**We care about relations, not metadata. Reduce every paper to the sentences that
could actually state one.**

Two stacked filters, both measured on this corpus:

1. **Entities via NCBI.** Full-text vocabulary is ~8,800 distinct terms per 12
   papers — hopeless at this n. Terms resolving to an NCBI microbial taxon through
   `kg/taxonomy.py`: **122. A 72× reduction**, and interpretable.
2. **Relations via direction words.** Keep only sentences containing *both* an NCBI
   taxon *and* a direction cue (increase/decrease and synonyms: elevated, reduced,
   enriched, depleted, higher, lower, abundance, over/under-represented,
   up/down-regulated, greater, diminished, expanded). Measured over 25 papers:
   **13,082 sentences -> 312 (2.4%), a 41× reduction; 1.2M chars -> 28.5k.**

A relation can only be stated in that 2.4%. Everything else is background by
construction. Build this as `kg/relation_sentences.py` producing, per paper, the
filtered sentences with their taxa and direction cues tagged — it is the shared
substrate for everything below, and it is also a far better RAG chunk than what
`build_rag.py` currently emits.

Validate the filter before trusting it: for edges we already extracted, does the
filtered set still contain the sentence supporting the known relation? Report
recall. If the filter drops real relations, loosen the cue list — **do not** quietly
accept a filter that discards signal.

Then build a paper × taxon incidence matrix from the filtered sentences and test:
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

- Embed the **filtered relation sentences** from Task 1, not raw full texts. 28.5k
  chars of relation-bearing text per 25 papers is a far better signal-to-noise ratio
  than 1.2M chars of methods and references, and it is what the question is about.
- `sentence-transformers` locally (`all-MiniLM-L6-v2`, or a biomedical model). CPU
  is fine at this scale.
- **Control for disease.** The dominant axis will be "which disease this paper is
  about", which is already known and useless. Compare within-disease or residualise.
- Compare embedding-based separation against the taxon-vocabulary baseline from
  Task 1. **If the interpretable baseline does as well, prefer it** — 768 unnamed
  dimensions cannot answer "what is the explanatory variable", which is the actual
  question being asked.
- Sanity check: do embeddings reproduce the known structure (same disease cluster,
  same sequencing type cluster)? If not, they are not encoding what we need.

## TASK 2.5 — Replace BM25 retrieval with GraphRAG (do this; the current design is wrong)

`build_rag.py` currently retrieves with BM25 + entity matching. That is the wrong
primitive **because we have a graph and it ignores it**. Keyword scoring cannot
answer "what else is connected to this", which is the entire reason the graph exists.

Rebuild retrieval as graph traversal:

- **Entity-link the query** to taxon/disease nodes (the vocabulary is closed and
  known, so this is exact matching, not guessing).
- **Personalized PageRank** seeded on the matched nodes, run over the graph. Edge
  weights should combine evidence count and directional consistency; containment
  links (625 of them, currently unused by anything) let a query about a genus reach
  its family and vice versa. Damping ~0.85, a few dozen power iterations — the graph
  is ~900 nodes, this is milliseconds and needs no library.
- **Return a connected subgraph**, not a flat list of documents: the seed nodes,
  their high-PPR neighbours, the edges between them, and the papers backing each
  edge. That is a context block an LLM can actually reason over.
- **Multi-hop is the payoff.** "What links Parkinson's and Alzheimer's?" is a graph
  query — taxa adjacent to both — and BM25 structurally cannot answer it. Make sure
  that query works.

Keep BM25 only as a **baseline to beat**, and report the comparison honestly on a
handful of realistic queries. Do not ship it as the primary retriever.

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
