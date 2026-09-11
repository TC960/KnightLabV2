# Session prompt — KG usefulness + embeddings

> ## ⚠️ READ THIS BEFORE THE TASK LIST BELOW — updated 2026-09-10
>
> **The numbered tasks in this file are all DONE and have been for four sessions.**
> The scheduled routine still fires the old priority list (MAIN_DATA filter,
> Task 1, Task 2.5, Task 3.1), and three consecutive sessions have each opened by
> confirming they were already complete. If you are reading this because that
> prompt sent you here: **do not redo any of them.** Read `SESSION_LOG.md` — its
> top entry is the current state — and pick from the short list below.
>
> **Genuinely open, in order:**
>
> 0. **NEW 2026-09-11 — full-text screen of 18 named papers. Top item, above
>    "more papers", because it needs nothing you do not have and the list is
>    already written.** Only 23 of 271 contributing papers had ever been checked
>    for study design. The 249 others were screened in-session at ABSTRACT level
>    and **the method failed its own validation**: 20/24 keep-vs-drop on blinded
>    controls, and it marked a known rat-FMT study KEEP. All 4 of its proposed
>    drops were read against full text and **none survived** — the graph was NOT
>    modified, correctly. Do not re-run the abstract screen; the extractor bug
>    found alongside it is fixed but does not touch the judgement problem.
>    **What is left is 18 papers at full-text resolution** — the 14 UNCLEAR plus
>    the 4 nominations, listed in `corpus_screen.json`. The animal axis is CLOSED
>    (deterministic, recall 15/15, corroborated independently by the LLM screen
>    finding zero animal papers). Only no-healthy-control, case reports and
>    reviews remain. See `FINDINGS_corpus_screen.md`. Expect agreement not to
>    move (the 22-paper screen moved it by nothing); justify on correctness.
>
> 1. **More papers.** The binding constraint on every *statistical* question, and
>    has been for four sessions. 109 papers with ≥4 decisive observations sets
>    every MDE in the project. **Needs a GPU — ask before spending.**
> 2. **Two modelling calls that want a human, not a script.** Should a joint 16S
>    signal from an assay that cannot separate two genera
>    (`Escherichia-Shigella`) be attributed to one, split, or held apart as it now
>    is? And should disease subtypes be modelled as containment the way taxa are
>    (`Intracerebral hemorrhage` beside `Stroke`)? Both are design decisions.
> 3. **Report PMID 27703453 upstream to Disbiome** — their case/control assignment
>    is inverted on all five records, per `FINDINGS_db_conflicts.md`.
> 4. Re-run `adjudicate_db_conflicts.py` whenever the corpus grows.
>
> **Closed levers — do not reopen without new data:**
>
> - **Paper-level covariates of discordance.** 24 variables tested across two
>   passes (9 study-design, 15 wet-lab/bioinformatics), 24 nulls — and the
>   arithmetic says why: the paper-level SD of discordance is 3.4 points against
>   MDEs of 4–7. See `FINDINGS_paper_discordance.md`.
> - **Another structural correction expecting agreement to move.** SIX have now
>   moved it by less than this corpus can resolve, and the 2026-09-11 pair moved
>   the two databases in OPPOSITE directions (+0.004 Disbiome, −0.002 Peryton),
>   which is what noise looks like. Corrections are still worth making — they are
>   justified on correctness of meaning — but never report one as an accuracy gain.
> - **Rebuilding on top of a failed build.** `taxonomy_cache.py` replays
>   `graph.json`'s own resolution and `build_kg.py` overwrites `graph.json`, so a
>   bad intermediate silently becomes the authority for the next build. Restore a
>   known-good `graph.json` before re-running. This bit on 2026-09-11.
> - **The NCBI taxdump.** `ftp.ncbi.nih.gov` is blocked in the cloud environment
>   (CONNECT → 403, probed in four sessions). `taxonomy_cache.py` replays
>   `graph.json`'s own resolution and is valid for rebuilds over a SUBSET of the
>   current papers — which is every rebuild that does not add papers.

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

## STATE AS OF 2026-09-03 (already done — do not redo)

Read `SESSION_LOG.md` first; it is the running record and its top entry is the
current state. In short, **every numbered task below has been done**, and the
findings docs (`FINDINGS_*.md`) carry the results. What is left is listed under
"THE ACTUAL NEXT STEPS" at the end of this section.

- Graph: **272 contributing papers, 925 taxa, 40 diseases, 2,034 edges, 440
  replicated, 217 contested, 723 containment links, 100 placeholder nodes.**
  Disbiome **73.0%**, Peryton **72.5%** — but **2026-09-09: do not quote those as
  independent replication.** 43 of our 272 papers are also cited by Disbiome and
  24 by Peryton, and those back half the decisive pairs. Agreement is 87.5%/96.8%
  where both sides read the same paper and 58.1%/52.6% where the literature is
  disjoint (within Parkinson's: 100%/95.8% vs **59.0%/59.6%**, p=0.0001 each).
  73% is a blend of ~90% reading fidelity and ~55% cross-literature
  reproducibility. See `FINDINGS_independence.md`. Every edge now carries a
  measured `confidence` tier; **79% of the graph is `provisional`**.
- **Task 0 (rebuild on the correct datasheet): done.** Honest F1 ~0.59
  (permutation p=0.001). The old 0.390 was a blank-cell artifact; the old 0.680
  was an easy-subset figure.
- **MAIN_DATA screen: done.** 22 of 45 papers are not human case-control
  studies. Filtering them changed agreement by nothing (McNemar p=1.00 —
  and see the warning below about that metric).
- **Task 1 (relation-sentence filter): done and validated.** 94.8% recall of a
  97.3% ceiling, 281/281 papers covered, `relation_sentences.json`.
- **Task 1 pooled analysis: done, NULL.** See `FINDINGS_cooccurrence.md`.
- **Task 2.5 (GraphRAG): done.** Ties BM25 on ranking (0.800 vs 0.783 over 6
  queries); the real win is containment traversal, not ranking accuracy.
- **Task 3.1 (11 doubly-contradicted pairs): done.** 11 of 12 adjudicable pairs
  faithfully report what the paper says. One extraction error in fourteen.

### Five structural corrections, and a property of the validation

Paper screen, placeholder split, body-site keying, paper deduplication, and the
extended placeholder split have each moved agreement by **less than this corpus
can resolve** (minimum detectable change ~0.013). That is a property of the
validation — the decisive set is dominated by well-evidenced, unambiguously
named taxa — not a coincidence. All are justified on correctness of meaning.
**None may be cited as an accuracy gain.** Do not run another correction
expecting the agreement number to move.

Also retired: the binary "decisive pairs flipped / McNemar" metric is
**incapable** of responding to a paper-removal correction (a unanimous edge that
loses papers stays unanimous). Use `agreement_metric.py`'s signed concordance
with a paper-level null.

### What has been tested against contested edges, and returned null

Study design (FDR 0.243), body site (p=0.120, and the corpus is 97.9% gut), ASD
being worse (p=0.211), and taxon co-occurrence pooled and per-edge (p=0.14,
AUC 0.535 p=0.21). **Four explanatory variables, four nulls.** The minimum
detectable effects are set by n — 130 contested edges averaging ~5 papers — not
by the statistics. Expect the two remaining Task 1 questions (does profile
predict disagreement with the curated databases; are there taxon modules) to
return the same answer at the same n.

### The method that HAS worked, twice

**Anomaly-hunt the graph's own structure instead of testing hypotheses about
it.** Both real results this month came that way: the placeholder rank collapse,
and this session's 12 duplicate papers (found because three "signal" edges had
paper-profile cosine of exactly 1.00). Ask cheap structural questions — does a
paper contradict itself, do two nodes mean one thing, does a surface string
extend the name it resolved to — and verify by rebuilding and diffing.

**Verify every fix by rebuilding TWICE and diffing.** Two fixes in this repo
have silently erased themselves on rebuild while printing success.

### THE ACTUAL NEXT STEPS

1. ~~Split the 54 named species out of their genera. NEEDS THE NCBI TAXDUMP.~~
   **DONE 2026-09-08 — and every part of that framing was wrong. Do not redo it.**
   See `FINDINGS_species_split.md`. It was **24** species, not 54; it needed no
   taxdump (a taxid is stable across a rename, so Disbiome's pre-rename names join
   to NCBI on it — `species_synonyms.py`); and the other 91 child folds **must not
   be split** (`Escherichia / Shigella` names two taxa, `Clostridium_XlVa` is a
   cluster label). The blocking fear — a split species losing its containment link
   — was real and is handled by storing NCBI lineages per entry; one candidate that
   could not be given ancestry is refused rather than shipped detached.

2. **More papers — this is now the top item.** The binding constraint on every
   remaining question is n, not method. Extraction needs a GPU — **ask before
   spending.**

3. **One open item is a human decision, not an analysis.** The two-genus labels no
   longer vote as one of their genera — `Escherichia-Shigella` was fragmented across
   four nodes by punctuation alone and is now a single joint node (`multi_taxon.py`,
   2026-09-08). What is left is the modelling call itself: should a joint 16S signal
   from an assay that cannot separate two genera be attributed to one, split across
   both, or held apart as it now is? That wants a PI, not a script. The same class of
   question: model disease subtypes as containment rather than separate nodes, the way
   taxa already are: `Intracerebral hemorrhage` / `Hypertensive intracerebral
   hemorrhage` sit beside `Stroke`, and `Chronic traumatic complete spinal cord
   injury` beside `Spinal cord injury`, with no link between them. (The one pure
   *synonym* case, three spellings of anti-NMDAR encephalitis, is already
   folded.) This is a design decision, not a bug — it needs a human call.

## TASK 1 — Relation-bearing sentences (DONE — kept for the design rationale)

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

## TASK 2.5 — GraphRAG (DONE — and its premise was wrong; see FINDINGS_task2.5_graphrag.md)

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

## TASK 3 — Make the graph useful (3.1 DONE; 3.2-3.4 open)

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
- **Don't run another structural correction expecting agreement to move.** Five
  have now moved it by less than the corpus can resolve. Justify corrections on
  correctness and say so; never report one as an accuracy gain.
- **Don't shuffle observations.** Every permutation test here must randomise at
  the PAPER level. Pair-level shuffling has produced three false positives on
  record.
- **Don't trust a subagent's judgement call where a deterministic test exists.**
  An LLM adjudication of 18 self-contradictions got 4 of its 6 "extraction
  error" verdicts wrong; a one-line string comparison settled it.
- **Don't collapse ranks** — and note that `X sp.`/`X spp.`/`X unclassified`
  folding into the genus is CORRECT and is not an instance of this.
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
