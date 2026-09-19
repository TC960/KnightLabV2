# Session prompt — KG usefulness + embeddings

> ## ⚠️ READ THIS BEFORE THE TASK LIST BELOW — updated 2026-09-19
>
> **The numbered tasks in this file are all DONE and have been for six sessions.**
> The scheduled routine still fires the old priority list (MAIN_DATA filter,
> Task 1, Task 2.5, Task 3.1), and **five** consecutive sessions have each opened by
> confirming they were already complete. If you are reading this because that
> prompt sent you here: **do not redo any of them.** Read `SESSION_LOG.md` — its
> top entry is the current state — and pick from the short list below.
>
> ## ⛔ 2026-09-17 — STOP BELIEVING "NO MAIN_DATA IN THE CLOUD". IT IS A `unzip` AWAY.
>
> **`proj_2_attempt3/MAIN_DATA.json.zip` is TRACKED IN GIT** (33 MB). Only the
> unzipped 105 MB `MAIN_DATA.json` is gitignored — which is what every session,
> including this file, mistook for the corpus being unavailable. **Eleven
> sessions deferred item 0 below to "the Mac" on that false premise.** Run:
>
> ```
> unzip -o proj_2_attempt3/MAIN_DATA.json.zip MAIN_DATA.json -d proj_2_attempt3/
> ```
>
> 2,026 papers, ~2.4 s, 105 MB on a disk with 30 GB free. **Item 0 is now DONE
> (see below) and any future "needs the Mac for paper text" claim is wrong
> unless it is about something other than MAIN_DATA.**
>
> **Two other standing instructions in that routine prompt are also stale:**
> 1. It tells you to download the NCBI taxdump from `ftp.ncbi.nih.gov`. That host
>    has returned **403 in eleven consecutive sessions** (re-probed 2026-09-16);
>    `eutils.ncbi.nlm.nih.gov` is 403 too. Use `taxonomy_cache.py`, which replays
>    `graph.json`'s own resolution and is valid for any rebuild that does not add
>    papers. **Do not spend a tool call re-probing.**
> 2. It says paper text is unavailable in the cloud because `MAIN_DATA.json` is
>    gitignored. That is true only of FULL text — see the 2026-09-15 correction
>    below. `relation_sentences_clean.json` is committed and covers 271/271
>    contributing papers.
>
> **NEW 2026-09-16 — the recall direction is now open, measured, and it changes the
> top lever.** Two new instruments: `zero_yield_audit.py` / `FINDINGS_zero_yield.md`
> (paper-level recall **≥96.1%**, 99.6% counting only confirmable misses) and
> `edge_recall_audit.py` / `FINDINGS_edge_recall.md` (**edge-level recall at most
> ~98.1%** [96.4, 99.5], ~60 missed observations of 3,077 — the first edge-level
> recall number that does not depend on the flawed gold).
>
> **READ THIS BEFORE ANY FURTHER RECALL WORK — it cost this session a wrong
> headline.** The extraction prompt (`eval-v2/run_eval.py`, `samgated-v1`) does NOT
> extract anything that merely states a direction. It requires **reported
> statistical significance** (*"if significance is unclear or unreported for a taxon,
> omit it"*), **main text only**, and **disease vs healthy control only**. An audit
> that scores the extractor without applying its own gate **manufactures misses**:
> the first pass reported 4 paper-level misses where 1 is confirmable, and 36
> edge-level miss verdicts where 15 survive. Also: adjudicators' "verbatim" quotes
> are often paraphrases — 7 of 36 failed a machine verbatim check. Always
> machine-check the quote, then apply the gate.
>
> **The top lever is no longer "more papers" in general — it is a SHORT LIST.**
> 12 of the 15 confirmed missed observations come from just 3 papers, and the 94
> candidate papers are already enumerated in `edge_recall_packets.json`. A bounded
> re-extraction over the worst ~20 recovers most of the recoverable loss and is far
> cheaper than a corpus-scale run. Still needs a GPU — **ask before spending** — but
> it is a much smaller ask than item 1 below.
>
> **NEW 2026-09-19 — the first precision-of-SCOPE number, and two shipped errors
> it found.** Every instrument before it scored whether an edge's taxon and
> direction are right; none asked whether the COMPARISON was admissible.
> `contrast_census.py` / `FINDINGS_contrast_scope.md`: **2,871 of 3,077
> observations (93.3%) are confirmed disease-vs-healthy-control**, and
> out-of-gate is **bounded at 1.8–2.7%** (quote the range). Out-of-gate papers
> disagree with the literature **1.75×** as often (O/E 1.747 vs 0.989,
> p = 0.00015, MDE ±0.303, four attacks survived) — a **validated** quality flag,
> built to avoid the flaw that got the 2026-09-17 provenance tier retracted, and
> worth ~6 excess disagreements, so **never quote it as an accuracy gain.**
>
> **A SECOND, SEPARATE number from the same pass — do not add it to the first.**
> Within papers that ARE in scope, observations whose taxon is never reported
> against a control in any sentence naming it are **~1.3% of the graph, CI
> [0.8%, 1.8%]** (48.7% of an 84-observation flagged set adjudicate as genuinely
> out-of-gate, CI [29.0, 65.8], paper-clustered bootstrap). It bounds
> contamination **inside the flagged set only**, so it is a floor, not a ceiling.
> Getting there needed **16 of 20 adjudicator verdicts overruled** — they inferred
> healthy-control comparators the text never states; quotes offered as control
> contrasts turned out to be mice, Aβ+ vs Aβ− cognitively normal, sALS vs bALS.
> The re-read was run in **both** directions (all 18 opposing verdicts survived)
> and every override carries its reason in `contrast_candidate_override.json`.
>
> **(c) A third method rule, and it cost two bugs to learn.** Running the chain
> twice caught a BH sort that ordered p-values **by predictor name** (giving both
> nulls q = 0.0001) and a sampler that silently **redrew** when its pool changed,
> orphaning all 46 adjudications while exiting 0 and printing a plausible count.
> **A pipeline that re-samples on every run cannot be checked by diffing** — freeze
> the draw. All ten scripts here now reproduce byte-identically.
>
> **TWO EDGE-CONTENT DECISIONS ARE NOW WAITING ON A HUMAN, and they are the only
> things in the backlog that change the graph.** Both on the Alzheimer's node:
> the SILCODE amyloid paper contributes **13 edges** and *no subject in it has
> Alzheimer's* (every result sentence is cognitively-normal amyloid-positive vs
> amyloid-negative; 10 of the 13 land on contested pairs), and a second paper
> contributes **7 edges** from AD-with vs AD-without neuropsychiatric symptoms.
> Not fixed here: dropping a paper is a corpus-inclusion call and there is no
> correct node for "amyloid-positive but cognitively normal".
> `contrast_out_of_gate.json` ships the tiered list **opt-in**.
>
> **Two nulls from the same pass, with power.** The free-text disease label does
> NOT predict out-of-gate design (5.9% vs 7.5%, p = 1.000, MDE ±8.5 pts) — the
> 19 out-of-gate papers sit under *canonical* nodes. And mixed provenance (**94
> of 241 in-scope papers also report a subgroup contrast**) does not degrade
> agreement (p = 0.650, MDE ±0.140). Variables 26 and 27, nulls 26 and 27.
>
> **Two method rules updated.** (a) The 2026-09-16 verbatim-quote rule stands but
> its count overstates: 16 quotes failed byte-for-byte here and only **5** are
> real paraphrases; the other 11 differ by PDF-extraction whitespace. Report both
> tiers. (b) **A title naming an animal model does not make the paper
> animal-only** — one ANIMAL verdict was overturned by reading, and the
> 2026-09-11 animal null still stands.
>
> **Also fixed 2026-09-19:** the 2026-09-17 session was never written into
> `SESSION_LOG.md`, so for two days this file's "read the top entry" instruction
> pointed at a state whose accuracy numbers were superseded. Logged retroactively;
> `CLAUDE.md` now carries F1 **0.739** against the new gold, the
> double-counting-matcher correction, the gold-is-a-test-set rule and the
> provenance retraction.
>
> **Genuinely open, in order:**
>
> 0. ~~**Re-run `silent_edge_mentions.py` where `MAIN_DATA.json` exists** (i.e. on
>    the Mac, not in the cloud).~~ **DONE 2026-09-17, IN THE CLOUD — do not redo.**
>    The premise was wrong (see the banner above: the corpus zip is tracked).
>    **The hole is CLOSED: 271/271 contributing papers, 0 unscoreable
>    observations** (795 → 581 → 0). MAIN_DATA got it to 581; the last 43 papers
>    are covered by `extract_input.json` / `new_papers.json`, which are
>    **gitignored on purpose — this repo is PUBLIC and the corpus holds
>    non-open-access articles (254b0a8)**. Restore them locally, read-only, and
>    NEVER `git add` them or quote their text into a tracked file:
>
>    ```
>    git show 254b0a8^:proj_2_attempt3/kg/extract_input.json > extract_input.json
>    git show 254b0a8^:proj_2_attempt3/kg/new_papers.json    > new_papers.json
>    ```
>
>    **The result: 100% mention rate over the WHOLE graph** — every one of the
>    3,077 observations names its taxon in its source paper (own 2,109/2,109,
>    background 585/585, silent 381/381), and **all 12 "absences" are matcher
>    false negatives, none a fabrication.** ~25% of the graph had never been
>    scored by any instrument; scoring it turned up nothing new, which is the
>    finding. Source choice cannot change a verdict — 5 of 6 pairwise source
>    comparisons agree exactly (`validate_text_sources.py`). Three new
>    deterministic tiers absorb the false negatives (`genus_factored`,
>    `gloss_gap`, `defined_abbrev`), each keyed to how the paper writes the
>    name rather than to a loosened threshold. **The
>    "one confirmed extraction error" recorded on 2026-09-13 is RETRACTED** — the
>    paper does report `Clostridiales incerte sedis XIII`, with its own r and
>    p-value; the 2026-09-13 quote was truncated one clause short of the
>    disproof. **There are now zero confirmed extraction errors in the graph.**
>    Method rule earned here: *a true quote can mislead by where it stops* — check
>    the disputed token with a whole-word count, which cannot be truncated.
>
> **CORRECTION 2026-09-15 — "no paper text in the cloud" is too broad, and it has
> been costing sessions.** `relation_sentences_clean.json` is **committed** and
> covers **271 of 271 contributing papers (100%)** — 6,294 relation-bearing
> sentences, 1.74 MB, each tagged with resolved taxa and direction cues. Any
> question of the form *"what did this paper compare?"*, *"what is the cohort?"*,
> or *"which direction does it report for taxon X?"* **is answerable in the
> cloud.** The 2026-09-15 HIV/`Neurocognitive impairment` question was settled
> entirely from it with no `MAIN_DATA.json`. The limit is real and specific: it
> holds ~10% of corpus text (16.78 M → 1.74 M chars), so it **cannot** answer
> "does taxon X appear *anywhere* in this paper". ~~which is precisely why item 0
> above still needs the Mac.~~ **2026-09-17: that last clause is wrong — item 0
> never needed the Mac; unzip the tracked corpus (banner at the top). For
> whole-document questions use `MAIN_DATA.json`; `relation_sentences_clean.json`
> remains the right, cheaper tool for sentence-level questions.**
>
> 1. **More papers.** The binding constraint on every statistical question, and
>    has been for five sessions. 109 papers with ≥4 decisive observations sets
>    every MDE in the project. **Needs a GPU — ask before spending.** The
>    corpus-screening alternative below is now SPENT, so this really is the top
>    item again.
> 2. **ONE modelling call left, and it is now precisely scoped.** The
>    disease-subtype half of this is **largely ANSWERED, 2026-09-14** — see
>    `FINDINGS_disease_ontology.md` and do not redo it. MONDO is reachable here
>    (GitHub release, 200; `purl.obolibrary.org` and EBI are blocked), so
>    `mondo.py` resolves 28 of 40 disease labels and **confirmed 2 of 2 checkable
>    is-a claims and upheld 2 of 2 rejections** — the hand-written tiers were
>    right. The 2 MONDO links are shipped (`disease_hierarchy_links.json`,
>    opt-in). And the 71-paper cognitive cluster **does not cohere**: 0.592 vs a
>    0.672 background, with every MCI pair at or below a coin flip while the
>    MONDO-confirmed AD/Dementia link runs 0.938. **So: link those nodes for
>    retrieval, do NOT pool their evidence, do NOT fold MCI into Alzheimer's.**
>    What genuinely remains for a human: (a) the `Escherichia-Shigella` question,
>    and **2026-09-14 reframes it** — containment is the WRONG primitive, because
>    a joint label denotes reads that could be either genus and so is not a
>    subset of either. The three options on record (attribute-to-one / split /
>    hold-apart) all miss what a user wants, which is the joint node reachable
>    from either parent without either absorbing its evidence. That needs a new
>    **edge type** ("ambiguous assay", not `parent_of`) — a schema decision. 5
>    nodes, 13 edges. See `FINDINGS_orphan_parents.md`. (b) whether the
>    MCI / `Cognitive impairment` / `Neurocognitive impairment` labels are
>    *synonyms of each other* (a folding question) given MONDO carries none of
>    them. Not (b)-as-containment — that is now answered no.
>
>    **(b) is now ANSWERED for `Neurocognitive impairment`, 2026-09-15: DO NOT
>    FOLD — for a cohort reason, not a data reason.** Its one paper studies NCI⁺
>    vs NCI⁻ *within an HIV-infected population* (verified from the paper's own
>    sentences, in the cloud). The edges are correctly typed, so this is **not** an
>    error — but the source population is HIV⁺ and transfer to non-HIV cognitive
>    impairment is untested. The graph's data cannot decide it: n=5, **MDE 46.9
>    points**. Its MONDO id stays `None` — MONDO has no "neurocognitive" term, and
>    `MONDO:0020689` "AIDS dementia complex" is one rank too narrow (the ASD
>    mistake). **`MCI` vs `Cognitive impairment` remains open and is UNRESOLVABLE
>    at this n**: 9/18 = 0.500 vs a 0.669 background, p=0.138, **MDE 28.0 points**
>    against an observed gap of 16.9. It needs a clinical call or more papers.
>
> 2b. **NEW 2026-09-15 — the cheapest remaining non-GPU lever: a curated disease
>    label table.** `norm_disease` falls through to the extractor's free-text label
>    whenever its 17 regexes miss, so **25 of 40 disease nodes (399 edges, 19.9% of
>    the graph, 38 papers) are unvalidated strings**, 11 of them (145 edges) with
>    no MONDO id either. The taxon half has `taxon_typos.py` and `multi_taxon.py`;
>    the disease half has **nothing**. 4 genuine candidate families carry 383 edges
>    (`[cognitive impairment]` 209, `[cord injury]` 101, `[intracerebral
>    hemorrhage]` 61, `[hepatic encephalopathy]` 12). Census and instrument:
>    `disease_node_provenance.py`, `FINDINGS_disease_node_provenance.md`. **Do NOT
>    automate the folds** — that script's own head-noun heuristic grouped
>    Parkinson's + Alzheimer's + Huntington's (689 edges) on a shared suffix, which
>    is exactly why the taxon side uses curated tables and not thresholds.
>
> 3. **Report PMID 27703453 upstream to Disbiome** — their case/control assignment
>    is inverted on all five records, per `FINDINGS_db_conflicts.md`.
> 4. Re-run `adjudicate_db_conflicts.py` whenever the corpus grows.
>
> **Closed levers — do not reopen without new data:**
>
> - **Sentence-level direction audit. DONE 2026-09-12, and it is the best
>   fidelity number the project has.** Reading fidelity **≥86.6%** [81.7, 91.3]
>   against the papers' own sentences, independent of the gold AND of both
>   curated databases. All 28 residual disagreements adjudicated twice — **zero
>   extraction errors** — so it is a lower bound. Do NOT rebuild the
>   comparison-frame corrector: it was built, measured, and made agreement worse
>   (0.866 → 0.774), because 41 of 54 "control-framed" sentences name controls as
>   the *reference*, not the subject. See `FINDINGS_direction_audit.md`.
> - **Textual provenance as a predictor of discordance.** own-result 27.6% vs
>   background-only 27.1%, pooled difference −0.6 points at an MDE of 8.4. The
>   **25th** variable, the 25th null. The 585 background-only observations are
>   not worse evidence, so there is no defective subset to review.
>
> - **Corpus screening for study design. DONE 2026-09-11, and it is clean.**
>   Only 23 of 271 papers had ever been screened; the other 249 now have been.
>   Animal studies: deterministic full-text filter, recall 15/15 on the existing
>   gold set, and a null on the rest. The other three failure modes: an abstract
>   screen with 24 blinded controls flagged 18 papers as droppable or unclear,
>   **all 18 were read against full text, and all 18 are KEEP.** Zero papers
>   removed. Do NOT re-run the abstract screen — it was 0-for-4 on drops and
>   0-for-14 on unclears, and it marked a known rat-FMT study KEEP; its value is
>   triage, not adjudication. Residual: 231 papers were cleared from an abstract
>   and never re-read, against a measured 2/12 missed-drop rate, so "consistent
>   with zero, not proven zero". See `FINDINGS_corpus_screen.md`.
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
> - ~~**One confirmed extraction error is open and unfixed (2026-09-13).**~~
>   **RETRACTED 2026-09-17 — it was never an error, and NOTHING needs rebuilding.**
>   Whole-word counts in the source paper: `xii` 1, `xiii` **1**. The full
>   sentence reads "...clostridiales incerte sedis xii ( r = -0.2625, p = 0.0396)
>   **and xiii ( r = -0.2113, p = 0.0495)** were negatively correlated..." — a
>   prefix-factored enumeration, extracted correctly. The 2026-09-13 quote stopped
>   one clause before the words that disprove it. **The graph has zero confirmed
>   extraction errors.** See `FINDINGS_mention_audit.md`.
>
> - **Orphan containment. DONE 2026-09-14.** 24 taxon nodes had their parent
>   written in their own label and were detached anyway (the placeholder branch
>   only fires for labels the taxonomy resolved; these resolve to nothing). 14
>   linked, 13 refused with reasons in `orphan_parents.py`, 91 correctly left
>   detached. Orphans 170 → 156, hierarchy 713 → 727. **Do not replace that
>   curated table with a substring rule** — four of the 24 candidates are
>   bacteriophages, and a phage is not contained in the genus it infects.
>   Re-run `orphan_parents.py` after any taxon-normalisation change. The 156
>   still-orphaned nodes are mostly 16S clade labels with no recoverable parent;
>   that residue is not worth another pass without new data.
> - **Disease ontology resolution. DONE 2026-09-14 and it found shipped errors.**
>   Two wrong MONDO ids were live on 209 of 2,008 edges and in `kg.html`: the MCI
>   node carried `MONDO:0005453` = *congenital heart disease*, and ASD carried the
>   id for narrow *autism*. Fixed at source, graph/RAG/viz/docs rebuilt under a
>   strict diff gate. **Do not re-add an id for MCI** — MONDO has no such term,
>   `None` is correct. Coverage is 28/40 with 12 documented refusals; don't
>   fuzzy-match the rest. Note for any future statistical work here: **paper
>   overlap between disease nodes is ZERO for every pair**, so cross-disease
>   comparisons are not subject to the shared-paper inflation that makes
>   73.0/72.5 a blend. And **4 is-a disease pairs would be needed** to resolve a
>   20-point ontology-proximity effect; the graph can form 2.
> - **`rag_corpus.jsonl` staleness. FIXED 2026-09-14** — it had never been
>   regenerated after the 2026-09-08 punctuation or 2026-09-11 spelling folds and
>   was wrong on 293 documents (7%). If you change taxon normalisation again,
>   **re-run `build_rag.py`**; the retrieval ground truth is derived from that
>   file, so a stale corpus silently rescores the retriever. Re-running the
>   comparison on the fixed corpus flips Task 2.5's ordering (GraphRAG 0.683 vs
>   BM25 0.700, exact permutation **p=1.000**) — which confirms that comparison is
>   a tie and that **neither number is a ranking**. Six queries cannot resolve a
>   gap below ~0.17.
>
> - **The NCBI taxdump.** `ftp.ncbi.nih.gov` is blocked in the cloud environment
>   (CONNECT → 403, probed in four sessions). **2026-09-13: `eutils.ncbi.nlm.nih.gov`
>   is blocked too (403), so there is no HTTPS API workaround — stop probing.** `taxonomy_cache.py` replays
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
