# Session log

Newest first. Nulls and dead ends are logged as results.

---

# SUMMARY — session of 2026-09-06 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Shipped the specificity layer into the viewer, then consumed the containment
links for the first time and found the project's flagship claim was both true
and misstated.** Write-up: `FINDINGS_rank_conflict.md`.

The scheduled prompt's priority list was stale — Tasks 1, 2.5 and 3.1 and the
MAIN_DATA filter are all already done per the entries below, so none were
redone. The taxdump route was re-probed once and is still shut
(`ftp.ncbi.nih.gov` CONNECT → 403), so the 54 named-species split remains the
top blocked defect. Work went to the two genuinely open items.

### Shipped: the viewer now says what an edge is worth

The 2026-09-05 session put `specificity` / `taxon_breadth` / `taxon_purity` /
`restates_prior` into `graph.json` and deliberately stopped short of the UI.
`build_viz.py` and `viz_network.js` had **zero** references to any of them, so
the published page still showed "*Streptococcus* enriched in Parkinson's, 5
papers" with no way to see it is enriched in eleven other diseases too.

- **Hollow bar = `restates_prior`** (252 edges). Encoded as fill-vs-outline, not
  opacity, so it stays distinct from `.faded` (contested) and survives
  greyscale; label text keeps full contrast.
- **"Sort by: disease specificity"** orders discriminating → mixed → narrow →
  generic. Deliberately **not** `taxon_purity` descending as the last session
  suggested: purity 1.0 *is* the generic case, so that sort surfaces exactly the
  edges the control exists to bury. Narrow outranks generic because a narrow
  taxon is *unjudged* (<3 diseases vote) while a generic one is known
  uninformative.
- "Hide edges that restate a prior" filters **both** views; scope chip per row;
  plain-English specificity sentence in the detail panel; two new table columns;
  a 155-discriminating tile; network links at 0.45 alpha when they restate a
  prior — damped, not hidden.
- Done **without touching `graph.json`**: the per-taxon disease counts live on
  nodes, which the payload does not ship, so `build_viz.py` sends a compact spec
  map keyed by `taxon_key`. The graph and its byte-for-byte fixed point are
  unchanged.
- **`verify_viz.py` is new and is the point.** "It parses" is not verification
  here — two fixes have silently erased themselves on rebuild while printing
  success, and a blank-canvas bug passed every static check. It drives the page
  in Chromium and reads the DOM and canvas pixels back *after real clicks*: 14
  assertions, all passing, including that the canvas is non-blank before and
  after the new filter. `kg.html` rebuilds to a fixed point; `docs/index.html`
  is back in sync.

### Bug 5 — the flagship example's numbers were stale on the published site

`kg.html`, `CLAUDE.md`, `build_kg.py` and `viz_network.js` all claimed
*Lachnospiraceae* is "depleted in Parkinson's across **15 papers**". The graph
says **9 papers, 8 down / 1 up, and the edge is contested**; *Hungatella* is 7
papers, 6 up / 1 down, also contested. The 15 predates the 2026-09-03
deduplication and was never updated — a published number that no longer matched
the artifact it described. All four corrected. Found by checking the anecdote
against the data rather than by looking for it, which is the method that keeps
working here.

### What survived: related taxa agree within a paper, and it is ancestry

**Within a single paper, taxonomically related taxa agree on direction 0.8903 of
the time against 0.5367 for unrelated taxa — gap +0.3536, z=15.5, p=0.0001**
(10,000 permutations, paper-level null shuffling each paper's direction labels
across the taxa it reported, preserving its up/down counts; MDE ±0.044). Both
arms come from the same paper, so cohort, country, pipeline and enrichment
propensity are differenced out by construction.

A z of 15 here is a reason for suspicion, so it was attacked four ways
(`attack_rank_conflict.py`) and survived all of them: dropping every split
placeholder node, since the split *created* both nodes from one mention
(+0.3592, **stronger**); a cluster-robust one-gap-per-paper statistic, since the
top paper contributes 35 of 474 related pairs (+0.3611, z=11.4 over 100 units);
direction skew (0.68, controlled by the shuffle by construction); and
same-rank-unrelated pairs, in case the effect was really "co-mentioned taxa
agree" (+0.3502, z=14.6 — it is ancestry specifically).

**This is face validity, not a discovery.** A family's abundance is largely the
sum of its genera, so ~89% is close to what taxonomy arithmetic predicts; the
extractor passing that check is the result. The value is the complement: the
**11% that disagree inside one paper** are neither noise nor rank confusion, and
they are exactly what collapsing ranks would destroy. The project's refusal to
collapse ranks now rests on a measured 11% rather than one anecdote.

### What got smaller under scrutiny: the "229 opposite pairs" figure

952 parent/child pairs share a disease — 586 same direction, 241 opposite, 125
exact ties. Of the 241 opposite pairs, only **33 (14%) are asserted inside a
single study**; **189 (78%) rest on no shared paper at all**, the family
measured by one set of studies and the genus by another. The GraphRAG session's
"903 edges have a parent edge, 229 point the opposite way" is real but reads as
a much stronger claim than it supports. Cite 14%, not 25%, and say what it
means. The 33 within-paper conflicts — *Lachnospiraceae*↓/*Hungatella*↑ in
Parkinson's among them — are the concrete case for the containment layer and the
best review targets after the doubly-contradicted 11. Caveat: 32 of the 33 rest
on exactly one shared paper, so each individual pair is a single-study claim
even though the aggregate is not.

### Bug 6 — `python3 build_kg.py` silently rebuilt a three-revisions-old graph

Found by following the project's own "rebuild twice and diff" rule. `DEFAULT_IN`
still pointed at the raw 250-paper extraction
(`eval-v2/results/qwopus3.5-27b-v3__q4km__samgated-v1__all250.json`), which is
**not even present in a fresh clone**, while the shipped graph has been built
from `extractions_screened.json` since the paper screen landed — as
`graph.json`'s own `meta.source` has recorded the whole time. So running
`build_kg.py` with no arguments **overwrote `graph.json` with a 773-taxon /
1,462-edge / 211-paper graph** against the shipped 918 / 2,011 / 272, and
printed a normal success summary while doing it.

This is worse than the two fixes that previously erased themselves on rebuild,
because the verification ritual adopted to catch *those* is "run this command
twice" — the rule told you to run the thing that destroys the artifact it
verifies. It is also why the cloud environment looked like it could not rebuild
the graph: it can, perfectly. `DEFAULT_IN` now points at
`extractions_screened.json`; with it, a rebuild reproduces the committed graph
with **zero drift in any pre-existing field** (meta, all 918 nodes, all 708
hierarchy links, all 272 papers and all 2,011 edges identical), and two rebuilds
are byte-identical. The taxdump is NOT required for `build_kg.py` — only for the
still-blocked species split.

### Shipped: the 33 rank conflicts are now findable

`annotate_rank_conflicts()` in `build_kg.py` adds `rank_conflicts` and
`has_within_paper_conflict` per edge, computed inside `build()` from the edges
just built rather than as a sidecar reading `rank_conflict.json`, so it cannot
drift or self-erase — the same reasoning as `annotate_specificity`. **59 edges**
carry a within-paper conflict (the 33 pairs, counted from both sides). The
viewer gains a `rank ↕` chip, a "Rank conflicts only" filter, and a detail-panel
block naming the counterpart taxon and the study that reports both directions.
Deliberately, only `within_paper` conflicts are chipped: the 189 pairs resting
on no shared paper are an artefact of pooling and flagging them would relaunch
the overstatement this session just corrected. `verify_viz.py` grew to **19
assertions**, all passing — and it earned its keep immediately by catching a
regression from a layout change of mine, where moving the `split` chip into the
scope cell silently changed which chip the specificity sort was read from.

### Highest-value next step

**The 54 named-species split, on a machine with the taxdump** — unchanged as the
top defect, and now the only blocked item that is purely mechanical. Everything
else not needing a GPU is either done or known to be underpowered at n=272.
Second choice, and unblocked here: per-disease evidence summaries with
exportable citations (Task 3.4), the last unstarted item on the useful-output
list.

---

# SUMMARY — session of 2026-09-05 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**The disease dimension of this graph carries reproducible directional
information for Parkinson's disease and, at n=272 papers, for nothing else.**
Write-up: `FINDINGS_disease_specificity.md`.

### What I tested

The last open item needing neither the taxdump nor a GPU: quantify the
disease-side fragmentation nobody had measured (40 disease nodes, zero
containment links, so `Intracerebral hemorrhage` sits beside `Stroke`
unconnected while the taxon side models containment with 708 links).

### What did NOT survive

- **"A clinical subtype resembles its parent disease."** NULL across all seven
  Tier-A is-a pairs (p=0.19–0.75; ICH→Stroke 19/22 decisive at p=0.43,
  Poststroke aphasia→Stroke 0/3 at p=0.75). With 3–22 decisive shared taxa per
  pair nothing could have survived — a power statement, not evidence of absence.
  Tier B (AD→Dementia, the cognitive-decline continuum) behaves the same.
  So the disease-containment layer is a **bookkeeping** decision, justifiable on
  correctness of meaning but **not** a signal gain. It still needs a human call;
  `disease_containment.py` records the tiering and the Tier-C rejections
  (Multiple system atrophy is a *sibling* of PD, not a subtype; MCI is a stage,
  not an AD subtype) so they are not re-proposed.
- **"Disease specificity is a corpus-wide property."** Refuted — see below.

### What survived

- **Disease identity does predict edge direction — p=0.0014.** Over 23,627
  same-taxon paper pairs: same-disease agreement **0.716**, different-disease
  **0.657**, gap **+0.0591**, z=3.39, MDE +0.0296, under a **paper-level**
  permutation of the disease label (pair-level shuffling would have been the
  fourth false positive on record here).
- **It is not country and not method.** Country's own gap is −0.0112 (p=0.64)
  and sequencing platform's −0.0278 (p=0.86) — **neither produces any agreement
  at all.** The disease gap holds inside same-country pairs (+0.078) and
  different-country pairs (+0.086); permuting disease within country blocks
  keeps it (+0.0846, z=2.32, p=0.0128). Both surviving p-values clear BH over
  the four inferential tests.
- **But it is ONE DISEASE.** Per-disease internal agreement: Parkinson's
  **0.807** on 1,220 pairs (lift +0.150 over the 0.657 cross-disease baseline),
  Stroke 0.725, MS 0.689, **Alzheimer's 0.608 — BELOW the cross-disease
  baseline, on 806 pairs with ample power** — Epilepsy 0.492. Zero of the other
  four match Parkinson's. Drop its 67 papers and the gap falls to **+0.0187,
  p=0.179 against MDE +0.0350**: effects above +0.035 are excluded outside PD,
  smaller ones are not. That PD is the standout is the field's own consensus, so
  this is **face validity** for the extraction, not a coincidence.
- **~70% of the graph's directional agreement is a generic dysbiosis prior.**
  Decomposing agreement above the 51.4% marginal chance rate: +14.3 points is
  disease-independent, +5.9 is disease-specific and almost all of that is PD.
  **59 of 187 taxa reported in ≥3 diseases never flip direction** —
  *Streptococcus* enriched in all 12 diseases reporting it, *Butyricicoccus*
  depleted in all 8. For those, "enriched in disease X" is near-contentless.
  Meanwhile *Prevotella* (6↑/8↓ over 14 diseases, 50 papers) and *Bacteroides*
  (6↑/8↓ over 14, 54 papers) are simultaneously the highest-evidence and least
  directionally consistent taxa: **high weight is not high information.**
- **This explains why five structural corrections could not move agreement.** If
  70% of directional agreement is a prior shared with Disbiome and Peryton, the
  validation is largely measuring that prior, not the graph's disease-specific
  content — and the decisive set is dominated by exactly these generic
  well-evidenced taxa. Sixth finding in a row the ~0.013 minimum detectable
  change cannot see; now with a mechanism rather than a shrug.

### Dead end closed: no PyPI package substitutes for the taxdump

The 54 named-species split stayed blocked. `ftp.ncbi.nih.gov` and
`ftp.ncbi.nlm.nih.gov` both give CONNECT → 403; EBI, Ensembl, GBIF, UniProt and
LPSN are denied too; only `pypi.org` / `files.pythonhosted.org` are reachable.
The one offline candidate, **`taxoniq`** (bundles an 89 MB NCBI database), was
extracted and tested: the full tree (2,609,295 taxa with parent and rank) and all
scientific names come out of its marisa tries, **but synonyms are deliberately
excluded — `taxoniq/build.py` indexes only `scientific name`, `common name`,
`genbank common name`, `blast name`.** Verified: `Bacteroidota`→976 resolves,
`Bacteroidetes`→**not found**; `Bacillota`→1239 resolves, `Firmicutes`→**not
found**. Since synonym folding is what stops evidence splitting across duplicate
nodes, and the Disbiome/Peryton join needs both sides through `taxonomy.py`, a
synonym-less table would silently break the graph. **Do not re-run this probe.**

### Bug 4 — the build was never byte-deterministic, so the verification rule cried wolf

Tried to *use* the project's own rule (rebuild twice and diff) before trusting a
rebuild here, and it failed: two builds of the same input differed at byte
278347. **Content was not the difference** — every node, edge, direction,
paper-count and meta field matched across two rebuilds and matched the committed
`graph.json`. The only variation was JSON key order in the `sites` dict on ~30
multi-site edges, because `sites` was `Counter(... for p in papers)` over a
**set** of title strings, whose iteration order is randomised per process by
`PYTHONHASHSEED`. Everything else in that block already went through `sorted()`.

This matters more than a key order sounds: the countermeasure adopted after two
fixes silently erased themselves on rebuild was *rebuild twice and diff*, and
that countermeasure was firing a **false positive on every single build** — which
is exactly how a real regression gets waved through as the usual noise. Fixed by
iterating `sorted(papers)`; three independent rebuilds are now byte-identical,
and `graph.json` is regenerated so a future diff against the committed file is
meaningful. **No number moves.**

### Shipped: generic vs discriminating edges are now IN the graph

Acted on the finding rather than only writing it up. `build_kg.py` now annotates,
per taxon node, `specificity` (breadth, n_diseases_enriched/depleted, purity,
consensus, class) and per edge `taxon_breadth`, `taxon_purity`, `taxon_class`,
`restates_prior`. Over 2,011 edges: **291 generic, 155 discriminating, 664 mixed,
901 narrow** (<3 diseases, nothing can be said); **252 edges restate the taxon's
corpus-wide tendency outright**; 61 taxa generic, 19 discriminating.

Computed inside `build()` from the edges just built, deliberately not as a
sidecar, so it cannot drift out of sync or self-erase on rebuild. A contested
edge casts **no** vote (stricter than the exploratory script, which voted by
majority — *Streptococcus* is generic over 11 diseases here, 12 there; documented
at the code), and the 8 taxa whose every edge is contested get breadth 0 rather
than a missing field.

Verified by executing: two rebuilds byte-identical, all 2,011 edges and 918 taxa
annotated, every pre-existing field unchanged. So the published graph's numbers
are untouched — still 272 papers, 918 taxa, 2,011 edges, Disbiome 71.9%, Peryton
72.5% — and `kg.html` / `docs/index.html` were **not** regenerated.

### Highest-value next step

**Surface the new specificity fields in the viewer** (`build_viz.py` +
`viz_network.js`, then regenerate `kg.html` / `docs/index.html`). The data layer
landed this session; the UI change is deliberately separate because it is
outward-facing. A biologist reading "*Streptococcus* enriched in Parkinson's, 5
papers" still cannot see that it is enriched in eleven other diseases too, and
`restates_prior` on 252 edges is exactly the flag that fixes it. Suggested
treatment: de-emphasise `restates_prior` edges and let the ranked bars sort by
`taxon_purity`, so the discriminating edges surface instead of the loudest ones.

(The 54 named-species split remains the top *defect*, unchanged and still needing
the taxdump on a machine that can reach NCBI — the offline route is now a closed
dead end, see above.)

---

# SUMMARY — session of 2026-09-03

**The assigned analysis returned a null. Attacking it, and then attacking the
graph the same way, found three structural defects that four sessions of
screening had missed: 12 duplicate papers, three disease nodes that are one
disease, and 115 child taxa still folded into their parents.** Write-ups:
`FINDINGS_cooccurrence.md`, `FINDINGS_rank_collapse.md`.

### What I tested

The pooled Task 1 question, named by the last session as the highest-value next
step: within a fixed (taxon, disease) contested edge, do papers reporting
enrichment differ from papers reporting depletion in their taxon co-occurrence
profile? Substrate `relation_sentences.json`; nulls at the paper level.

### What did NOT survive

- **"Same-direction papers share a taxon vocabulary."** First run: per-edge
  +0.047 against a null SD of 0.009 — five sigma, under *both* paper-level
  nulls, and it survived four attacks (profile size, country, sequencing type,
  Jaccard, per-edge median). It was **12 duplicate papers**. On the corrected
  graph: pooled +0.0023 (p=0.79), per-edge +0.0137 (p=0.14), balanced edges
  +0.0080 (p=0.41), rank-based AUC 0.5347 (p=0.21), and 66 of 130 edges
  positive — 51%, chance. Null in all nine variants, and stable across all
  three of this session's corrections. Reported with power: the minimum
  detectable per-edge effect is ~0.017 and |AUC−0.5| ~0.053, so this is "no
  effect visible at 130 edges / 1,848 pairs", not "no effect".
- **"The lone dissenter is just an atypical paper."** The mechanical explanation
  I expected to find. Refuted: majority-side members average 28.6 taxa,
  minority-side 29.5, and permuting within profile-size quintile changed
  nothing.

### What survived

- **Bug 1 — duplicate papers.** 12 papers were scraped once from a PubMed link and again from a
  PMC or publisher link; the copies' titles differ only by a trailing period or
  a curly-vs-straight apostrophe, and every paper key in this pipeline is the
  raw title string. Because edge weight IS paper count, each duplicate voted
  twice. Contributing papers 281 → 272; **`n_replicated` 472 → 437, so 35 edges
  (7.4% of all replicated edges) rest on a single paper**; 76 edges change vote
  counts; **4 lose a majority direction they only had because one paper voted
  twice** (*Bacteroides*/Dementia, *[Clostridium] leptum*/MS, *Bifidobacterium
  longum*/MS, and *Butyricimonas*/MS resolves a false 2-2 tie). Fixed in
  `build_kg.py`, verified a fixed point by rebuilding twice and diffing.
- **The anomaly pointed straight at the bug.** The three edges the failed
  analysis ranked strongest (+0.944, +0.870, +0.796) are three of those four
  direction changes. A statistic found in one step what a full-text screening
  pass over all 45 title-matched papers had not.
- **Bug 2 — one disease, three nodes.** `Anti-N-methyl-D-aspartate receptor
  encephalitis` (22 edges), `NMDAR encephalitis` (16) and `Anti-NMDAR
  encephalitis` (7) were three separate disease nodes, one paper each. This is
  the Bacteroidetes/Bacteroidota case in the disease dimension. Folding it: 45
  edges become 38, and **4 edges that every view showed as single-paper become
  replicated, 3 of them CONTESTED** — real inter-study disagreement the
  fragmentation was hiding, plus a 3-paper unanimous *Faecalibacterium* edge
  displayed as three singletons. Fragmentation hides replication AND
  contradiction; duplication invents it.
- **Bug 3 — the placeholder split was half a fix.** Asking which surface strings
  EXTEND the scientific name they resolved to: **115 strings over 52 nodes, none
  flagged**; 285 edges touch one, 76 contested. 29 are `X sp./spp.` where
  folding is correct; **32 are SILVA placeholders the 2026-09-01 pattern
  missed** (`Prevotella 9`, `Coprococcus_1`, `Clostridium IV`, `Clostridiaceae
  1`); 54 are real named species (`Prevotella copri`, `Klebsiella pneumonia`).
  *Prevotella*/Parkinson's — the graph's highest-weight edge at 17 papers — had
  **13 surface strings folded into one node**, five of them distinct SILVA
  genera. Fixing the placeholder class: taxa 892→918, edges 1,978→2,011,
  placeholder nodes 74→100, containment 684→708, and **5 contested edges were
  contested only because placeholder children were folded in**. `Prevotella_9`
  emerges as its own 2-paper contested edge. The 54 named species are NOT fixed
  — that needs real taxids and this environment's network policy denies the NCBI
  taxdump; `child_folds.json` carries all 115 classified, ready for a machine
  that has it.
- **The self-erasing fix, again — and killed properly this time.** The extended
  placeholder split silently decayed on rebuild (106 → 102 placeholder nodes)
  because a placeholder's parent was recoverable only from a containment link,
  which exists only when the parent is itself a node. Now recorded as
  `parent_taxid` on the node; two consecutive builds are byte-identical. The
  pattern had also over-matched bacteriophages (`Enterococcus phage EFAP 1`);
  added a `NOT_PLACEHOLDER` guard agreeing with the cache's existing one.
- **Self-contradicting papers: 18 claims, and 4 were not contradictions.** Does
  a paper ever call the same taxon both enriched and depleted for one disease?
  18 do, all on contested edges, one contested by a single paper alone. 14 are
  genuine (body site, subgroup, different comparator); **4 are two different
  strings folded onto one key** (*Eubacterium biforme* vs *E. rectale*) — which
  is what exposed Bug 3.
- **No other pseudo-replication is detectable.** Same-cohort screen on the
  curated fields (country + n_cases + n_controls): 4 candidate groups, all
  coincidental, **0 edges drawing >1 paper from any of them**. Power limit: only
  192 of 272 papers (71%) carry a full cohort signature. Dedup completeness
  double-checked two ways — no two rows share a PMID/PMC/DOI under different
  titles (318 of 326 have a resolvable id), and no fuzzy title pair exceeds 0.92.

### Agreement, again, moved by nothing

Headline rates after all three corrections: Disbiome **71.9%**, Peryton
**72.5%** (from 71.9% / 72.8%). The dedup correction measured on the sensitive
metric: −0.0024 (p=0.665) and −0.0007 (p=0.885), against a minimum detectable
change of ~0.013. Its null is deliberately mismatched and conservative — it
drops 12 *random* papers, deleting their evidence outright, where dedup deletes
only redundant evidence.

That makes **five** structural corrections in a row that agreement cannot see.
Treat it as a property of the validation, not a coincidence: the decisive set is
dominated by well-evidenced, unambiguously-named taxa, and every correction so
far acts on the margins. All three are justified on correctness of meaning;
**none may be cited as an accuracy gain.**

### Shipped

`graph.json` / `kg.html` / `docs/index.html` / `rag_corpus.jsonl` rebuilt:
**272 contributing papers, 918 taxa, 2,011 edges, 438 replicated (was 472), 219
contested, 708 containment links, 100 placeholder nodes, 40 disease nodes.**
Disbiome 71.9%, Peryton 72.5%. New: `cooccur_direction.py`,
`cooccur_diagnostics.py`, `cooccur_followup.py`, `child_folds.json`,
`selfcontra_packet.json`, `selfcontra_verdicts.json`, `agreement_dedup.json`,
`FINDINGS_cooccurrence.md`, `FINDINGS_rank_collapse.md`. `build_kg.py` gains
`dedup_rows()` / `--keep-duplicate-papers`, an NMDAR synonym entry, an extended
`PLACEHOLDER` + `NOT_PLACEHOLDER` guard, and `parent_taxid` on placeholder
nodes; `taxonomy_cache.py` prefers it; `agreement_metric.py` gains `--drop
dedup` and `--out`.

### A method note worth keeping

An LLM subagent asked to adjudicate the 18 self-contradictions from the source
sentences returned 6 "extraction errors"; **4 of the 6 were wrong**, and its own
quoted evidence showed an oral-vs-gut contrast or two distinct species. A
one-line deterministic test (was the same surface string on both sides?)
settled it. Where a mechanical test exists, prefer it to a judgement call — and
check the subagent.

### Single highest-value next step

**Split the 54 named species out of their genera — on a machine with the NCBI
taxdump.** It is diagnosed, classified and listed in `child_folds.json`, it
touches the graph's flagship edge, and it is the only item this environment was
blocked from finishing (the network policy denies `ftp.ncbi.nih.gov`). Expect
it to move no agreement number, like the five corrections before it.

After that: **more papers. The binding constraint is n, not method.** Four explanatory
variables have now been tested against contested edges — study design, body
site, and taxon co-occurrence pooled and per-edge — and all four are null. The
minimum detectable effects here (per-edge ≈0.017, |AUC−0.5| ≈0.053, mean
concordance ≈0.01–0.02) are set by 134 contested edges averaging ~5 papers, not
by the statistics. The remaining Task 1 questions (does profile predict
disagreement with the curated databases; are there taxon modules) are the same
shape at the same n and should be expected to return the same answer.

Extraction needs a GPU — **ask before spending.** The CPU-only alternative worth
doing first is cheap and was validated this session as a *method*: anomaly-hunt
the graph's own structure for defects rather than testing hypotheses about it.
That is what actually produced a result twice now (the placeholder collapse, and
this).

---

# SUMMARY — session of 2026-09-02

**The headline is that the instrument was broken.** The test used to evaluate the
last four structural corrections could not, by construction, return anything but
zero. Full write-up: `FINDINGS_validation_metric.md`.

### What I tested

1. Body site as an edge key — the top lever handed over by the previous session.
2. Whether the four "zero flips, p = 1.00" results were real nulls.
3. Both paper-removal corrections, re-run on a metric that can move.

### What did NOT survive

- **"Zero decisive pairs flipped — not underpowered, a true zero."** A tautology.
  A pair is decisive only when our edge is *unanimous* (`contested = bool(up and
  dn)`; verified, all 1,765 non-contested edges have minority vote 0), and every
  correction only removes papers. A unanimous edge that loses papers stays
  unanimous in the same direction, so **no paper-removal correction can ever flip
  a decisive pair**. Confirmed empirically: across the gut restriction 17 edges
  change `direction` and every one is a contested↔decisive transition, never
  enriched↔depleted. Two sessions of "did this recover agreement?" were asked
  with an instrument incapable of answering.
- **"Body site is the highest-value next step."** Wrong on the numbers. Once all
  281 contributing papers are labelled, the corpus is **97.9% gut** — six non-gut
  papers. Restricting to gut moves mean concordance **−0.0073** with Disbiome
  (p = 0.120, min detectable 0.0093) and **−0.0048** with Peryton (p = 0.285) —
  null, and in the *opposite* direction to the hypothesis. Rejected as an edge
  key; shipped as an edge attribute instead.
- **"Rothia/Parkinson's is two saliva studies."** It is one oral and one stool.
- **The MAIN_DATA screen, re-tested honestly:** +0.0015 (p = 0.852) / +0.0043
  (p = 0.620), with only 2–3 pairs moving. Still justified on construct validity,
  still not an accuracy gain.

### What survived

- **A metric that can detect a change.** Signed concordance
  `(n_up−n_down)/(n_up+n_down) × reference_direction`, with a **paper-level**
  resampling null (2,000 draws). Sensitive where the old one was blind: the gut
  restriction moves 16 of 242 Disbiome pairs and 19 of 188 Peryton pairs, where
  McNemar saw 0. It refuses to run unless its tally reproduces `graph.json`.
- **Body site for all 281 papers**, via a keyword scanner *scored before it was
  trusted*: 84.3% by argmax, 92.4% once any stool cue wins outright. The failure
  mode was co-sampling (stool studies drawing serum for metabolomics), not noise.
- **Two bugs, both found by verifying rather than reading.** (1) The
  rank-placeholder fix was **erasing itself on every rebuild** in this
  environment — 77 placeholder nodes → 0, 670 containment links → 610 — while
  printing a successful build. Now a verified fixed point. (2) Edge weight
  counted **observations, not papers**, contradicting the comment directly above
  it; 44 edges inflated, 7 contested edges change their majority label
  (*Bacteroides*/Parkinson's flips depleted → enriched).

### Shipped

`graph.json` / `kg.html` / `docs/index.html` rebuilt: 892 taxa, 1,985 edges, 220
contested, **684 containment links** (+14 correct ones the old build missed), 74
placeholder nodes, and every edge now carries `sites` + `gut_only` with a body
site for all 281 papers. New: `body_site.py`, `analyze_bodysite.py`,
`analyze_bodysite_effect.py`, `agreement_metric.py`,
`FINDINGS_validation_metric.md`. Disbiome 71.9%, Peryton 72.8%.

### Also done this session

**`relation_sentences.json` rebuilt on the full corpus** — the prerequisite the
previous session named. Coverage 211 → **281 of 281** contributing papers, recall
re-validated at **94.8%** of a 97.3% ceiling over 3,132 relations. Details in the
dated entry below.

### Single highest-value next step

**Stop correcting the graph and run Task 1's analysis on the substrate that now
exists.** Three structural corrections in a row have moved agreement by less than
this corpus can resolve — the minimum detectable effect (~0.01–0.02 mean
concordance) is set by *n*, not by the metric — so further cleanup cannot be
shown to help, and the honest options are analysis or more papers.

Analysis is the CPU-only one and is now unblocked: build the paper × taxon
incidence matrix from the filtered sentences and run the **pooled** test (do
papers reporting enrichment differ from papers reporting depletion in their taxon
co-occurrence profile?), with cluster-robust permutation at the paper level.
Pooled, because that is the version with power — per-contested-edge tests average
~4 papers a side and cannot be answered at this corpus size.

Expect it to be hard: this corpus has already produced two false positives that
survived until tested, and a third (the four "true zero" agreement results)
survived until this session. Anything that looks like a finding gets shuffled at
the paper level before it is believed.

More papers needs a GPU for extraction — **ask before spending.**

---

## 2026-09-02 — relation_sentences rebuilt on the full corpus: 211 -> 281 papers

The prerequisite the previous session flagged as blocking the embedding work.
`relation_sentences.py` read only `all_usable_papers.json` (the original 250), so
the filtered-sentence substrate covered **211 of 281** contributing papers and
excluded every paper the MAIN_DATA expansion added. It now merges every corpus
file carrying full text (348 papers), and covers **281 of 281**.

**Recall re-validated at corpus scale, and it holds.** Replaying all **3,132**
extracted relations: headline recall **94.8%** against a **97.3%** ceiling — so
97.4% of what any sentence filter could recover, with the direction-cue step
costing 1.9%. That is marginally *better* than the 250-paper measurement (93.9%
of a 96.7% ceiling). Reduction **14.0x** on sentences, 9.7x on characters —
consistent with the corrected 14.4x, and still nothing like the retired "41x".
162 misses: 67 taxa never appear literally in the paper, 56 dropped by the cue
filter, 39 missed by the matcher.

**Deliberate trade-off, recorded.** The old file was taxdump-built; this one is
replay-cache built, which on the shared 250 papers keeps 5,210 -> 5,102
sentences, **−2.1%**. That independently reproduces the exact 2.1% bias measured
last session. Consistency wins here: a mixed file would apply two different
matchers to different papers, and every downstream use (embeddings, per-paper
co-occurrence) compares papers to each other. Reduction ratios from this file are
therefore an upper bound, as the build warns.

---

# SUMMARY — session of 2026-09-01

**Tested four things and one follow-on fix. The headline is that three of the four
premises I was given turned out to be wrong, and the corrections are the result.**

### What survived

- **The MAIN_DATA corpus really is contaminated.** Reading all 45 title-matched
  full texts, **22 (49%) are not human case-control studies** — 15 animal, 3 with no
  healthy control, 2 case reports, 2 with no primary cohort. Every verdict carries a
  verbatim quote.
- **The relation-sentence filter is safe to build on.** 93.9% recall over 2,262
  extracted relations in `loose` mode, against a 96.7% ceiling — 97.1% of what is
  recoverable, with the direction-cue step costing only 2.0%. Use loose, not strict
  (84.8%).
- **Our extraction is right where two curated databases both say it is wrong.**
  Of 14 doubly-contradicted pairs, **11 of 12 adjudicable ones faithfully report
  what the paper says**. One extraction error in fourteen. Three of the disputes are
  explicitly acknowledged by the source papers themselves.
- **Two structural defects, both larger than the error rate**: rank placeholders
  folded into parents (now fixed), and body site missing from the edge key.

### What did NOT survive

- **"Filtering the contaminated papers will recover agreement."** It changed
  **zero** decisive pairs against either database (exact McNemar p = 1.00). Not an
  underpowered null — a true zero. The 22 papers supply 11 of 1,927 edges; 18 of
  them produced no usable extraction at all. The ~4-point drop was a **composition
  effect**: the batch added ~21 mostly-autism, mostly-single-paper pairs.
- **"Autism edges are worse."** 52.6% agreement looks damning but comes from **5
  papers**. Paper-level permutation: gap −0.225, null SD 0.139, minimum detectable
  0.273, **p = 0.211**. Pair-level shuffling would have returned a false positive.
- **"BM25 structurally cannot answer *what links PD and AD*."** It answers at
  P@10 = 1.00. GraphRAG ties it overall (0.800 vs 0.783 over 6 queries).
- **"41× sentence reduction."** That was a 25-paper pilot generalised ~3× too far;
  at corpus scale it is **14.4×**.
- **Two structural fixes moved agreement by nothing.** Both the paper screen and
  the placeholder split flip **zero** decisive pairs. Agreement rate is insensitive
  to structural corrections here, because decisive pairs are dominated by
  well-evidenced unambiguous taxa. Both were applied on **construct validity**, and
  neither should ever be cited as an accuracy gain.

### Shipped

`graph.json` / `kg.html` / `docs/index.html` rebuilt: **326 papers, 281
contributing, 892 taxa, 1,985 edges, 220 contested, 670 containment links, 77
placeholder nodes.** New: `taxonomy_cache.py`, `graphrag.py`, `compare_retrieval.py`,
`filter_maindata.py`, `analyze_filter_effect.py`, `build_adjudication_packets.py`,
plus four findings docs.

### Single highest-value next step

**Put body site into the edge key.** It is diagnosed, cheap, and currently
manufacturing false contradictions: *Rothia*/PD and almost certainly *Gemella*/PD
are saliva studies colliding with gut records on one node — 2 of the 14
doubly-contradicted pairs are this, not disagreement. `metadata.jsonl` already
carries `body_site` per paper, so this is a keying change, not new extraction. It
should also make the Disbiome/Peryton comparison honest, since both are gut-weighted
and we are currently scoring oral findings against them.

*(Runner-up, and a prerequisite for the embedding work: `relation_sentences.json`
still covers only the original 250 papers, not the current 326. That is why 2 of the
14 pairs could not be adjudicated at all.)*

---

## 2026-09-01 — Fixed the placeholder rank collapse; agreement again moved by zero

Acting on the top lever from the adjudication. `taxonomy.py` resolves a rank
placeholder by trimming its qualifier tail, so **"Erysipelotrichaceae UCG-003"** — an
uncultured *genus-level* label INSIDE the family — landed on the family taxid and
pooled as if it were the family. `build_kg.py` now gives placeholders their own node
with a **containment link** to the parent (`--merge-placeholders` restores the old
behaviour). Default ON, so a rebuild cannot silently revert it.

**The motivating case is fixed.** Erysipelotrichaceae/Parkinson's went from a
4-paper "depleted" family edge — which no paper actually measured — to a 1-paper
family edge plus a separate 3-paper *Erysipelotrichaceae UCG-003* edge beneath it.
The apparent 4-paper contradiction of both curated databases evaporates.

**Effect on agreement: zero, again.** Paired McNemar on pairs decisive in both
graphs: **0 flips**, p = 1.00, against both Disbiome and Peryton. Headline rates
wobble (Disbiome 73.1→71.9, Peryton 71.9→72.8) purely through which pairs are
decisive. That is now three structural corrections in a row that change no decisive
pair — worth treating as a property of this validation, not a coincidence: the
decisive set is dominated by well-evidenced unambiguous taxa, so it cannot see
changes at the margins. **Justified on correctness, not the metric.**

Graph: 892 taxa (+60), 1,985 edges (+69), 670 containment links (+45), contested
225 → **220**, 77 placeholder nodes. `resolved` still means "has an NCBI taxid", so
placeholders are `resolved: false, placeholder: true` rather than inflating the
resolved count.

---

## 2026-09-01 — Adjudicated the doubly-contradicted pairs: 11 of 12 were OUR reading, correctly

Full write-up: `FINDINGS_task3_adjudication.md`. Verdicts + quotes:
`adjudication_verdicts.json`.

**Tested.** The pairs contradicted by BOTH Disbiome and Peryton — the strongest
error signal available. On the screened graph there are **14** (the 11 on record
predates the rebuild). Each read against its source papers' own sentences.

**Survived: our extraction.** 11 of 12 adjudicable pairs faithfully report what the
paper says. **1 extraction error in fourteen.** The doubly-contradicted set is not a
pile of our mistakes — it is mostly the literature disagreeing with itself. Nine are
genuine disputes, and **three are acknowledged by the source papers themselves**:
Dorea *"contrary to Liu's findings (2019)"*; Dialister *"previously shown to have a
higher relative abundance ... in a Southern China population ... may reflect dietary
or other geographical differences"*; Halomonas *"Different from Vogt's and Liu's
studies"*. Dialister is the model contested edge — correct, >10-fold, and the paper
names both the conflict and a mechanism.

**The one real error: Phascolarctobacterium / Parkinson's — DROP.** Its only
supporting sentence says the genus was *"correlated with disease stage"* — a
severity correlation within patients, with no direction and no case-vs-control
contrast. The direction was manufactured. A specific, auditable failure mode:
reading a severity correlation as a disease-vs-healthy direction, despite the
extraction prompt being gated on exactly that contrast.

**Two structural defects, both bigger than the error.**
(1) **Rank placeholders are folded into their parent.** Erysipelotrichaceae/PD looked
like a 4-paper contradiction; in fact 3 of 4 papers report *"Erysipelotrichaceae
UCG-003"*, a genus-level SILVA placeholder INSIDE the family, folded onto the family
taxid, and the 4th attributes the change to a member species. No paper measures the
family aggregate. Systematic: **74 placeholder strings onto 37 taxids, 21 edges named
only by a placeholder, 170 mixed, 52 of those contested.** Lachnospiraceae alone
absorbs ND3007/ND3008/NK4A136/UCG-001/UCG-004/UCG-008. This violates the project's own
rule that synonym folding and containment are different operations — a UCG label is a
*child*, not a synonym.
(2) **Body site is not in the edge key.** Rothia/PD is not a contradiction: both our
papers are saliva studies, the curated records are gut. Gemella/PD is the same paper
and almost certainly the same story.

**Next lever (highest value in the project right now).** Stop folding rank
placeholders into parents — give `X UCG-003` its own node as a containment child of
`X`. It is a bug fix rather than a judgement call, touches ~191 edges, and resolves
the worst-looking contradiction in the set.

---

## 2026-09-01 — GraphRAG built; ties BM25 on ranking, wins only on containment

Full write-up: `FINDINGS_task2.5_graphrag.md`. Code: `graphrag.py`,
`compare_retrieval.py`.

**Built.** Personalized PageRank retrieval (damping 0.85, ~900 nodes, no library):
closed-vocabulary entity linking → PPR from the seeds → a connected subgraph with
directions, evidence counts, containment links and backing papers. Multi-entity
queries rank by the **geometric mean** of per-seed PPR, not a joint run, so a node
must be close to *all* seeds rather than merely near the bigger disease.

**Did NOT survive: the claim that GraphRAG beats BM25.** Over 6 queries with truth
computed from the graph (bridge = ≥2 papers on both diseases), mean precision@10 is
**GraphRAG 0.800 vs BM25 0.783** — a tie. GraphRAG wins 2, loses 1, ties 3.

**Did NOT survive: "BM25 structurally cannot answer *what links PD and AD*".** It
answers at P@10 = 1.00. `build_rag.py` is not pure BM25 — it already has entity
matching and evidence weighting, so it filters to edges about either disease and
ranks by paper count, and hub taxa *are* the bridges. Retire that claim.

**Two measurement errors I made and corrected, logged so they are not repeated.**
(1) The first bridge metric was *saturated*: bare co-membership makes 125 of 832
taxa correct for PD/AD, so any ten hubs scored 1.00 and the systems tied trivially.
A metric that cannot separate them is not evidence they are equal. (2) **PPR is
direction-blind** — proximity has no sign, so "what is depleted in Parkinson's"
returned enriched taxa too (P@10 0.60, a real loss). Direction is now an explicit
filter on the seed disease; 0.60 → 0.80. It still loses that query type to BM25.

**Survived — the actual case for the graph is containment, and it is quantified.**
Query *Hungatella*: GraphRAG returns *Lachnospiraceae* depleted in PD (16 papers)
beside *Hungatella* enriched (7), the rank conflict this project calls load-bearing;
BM25 returns only Hungatella docs and cannot reach the family, because no document
holds both claims. Corpus-wide: **903 (taxon, disease) edges have a parent edge in
the same disease, and 229 of those (25%) point the opposite way.** That is the
retrievable context BM25 structurally misses — and it doubles as a first result for
the "are contested edges rank confusion?" question.

**Recommendation.** Ship GraphRAG for its *output* (connected subgraph with
provenance) and for containment traversal — not on a ranking-accuracy claim, which
the data does not support. Keep `build_rag.py`: it is genuinely better on
directional one-hop queries. Caveat: 6 queries is small and the 0.017 gap is noise.

---

## 2026-09-01 — Relation-sentence filter: recall validated at 93.9%; the "41x reduction" was pilot noise

**Tested.** Whether the relation-bearing sentence filter (`relation_sentences.py`)
keeps the sentences that actually support the relations we extracted — the check
that had to pass before anything downstream is allowed to use it.

**Survived — the filter is safe to build on, in `loose` mode.** Replaying all
**2,262** extracted relations across 250 papers:

| | strict | **loose** |
|---|---:|---:|
| sentence reduction | 22.2x | **14.7x** |
| taxon string anywhere in raw paper (ceiling) | 96.7% | 96.7% |
| taxon seen by matcher | 94.8% | 95.8% |
| **taxon in a KEPT sentence (headline recall)** | 84.8% | **93.9%** |
| share of what the matcher saw (= cue-filter cost) | 89.4% | **98.0%** |
| kept-sentence cue agrees with direction | 86.7% | 96.4% |

Against a **ceiling of 96.7%** — 3.3% of extracted relations name a taxon that never
appears literally in the paper, so no sentence filter can reach them — loose mode
recovers **97.1% of what is recoverable**, and the direction cue filter costs only
2.0%. Strict mode is the wrong trade: it buys 1.5x more reduction for 9 points of
recall, discarding 226 real relations at the cue step alone. **Use loose.**

**Correction: the reduction ratio was badly overstated.** The 25-paper pilot on
record claimed *"13,082 sentences -> 312 (2.4%), a 41x reduction"*. At corpus scale
it is **14.4x** (75,004 -> 5,210 sentences; 10.2x on characters). The pilot
generalised from 25 papers and was off by ~3x. Anything reasoning from "2.4% of
sentences" should be redone at 6.9%.

**Kept the existing `relation_sentences.json`** (taxdump-built, 5,210 sentences)
rather than overwriting it with a cache-built one (5,102). That 2.1% gap is a clean
empirical bound on the replay cache's bias for this task — smaller than expected,
and it confirms the cache understates sentences kept, so measured reduction ratios
are an upper bound.

---

## 2026-09-01 — MAIN_DATA screen: contamination confirmed, but it is not what moved agreement

Full write-up: `FINDINGS_task1_maindata_filter.md`.

**Tested.** Whether screening the 45 title-matched MAIN_DATA papers to human
case-control studies recovers the ~4-point agreement drop with Disbiome/Peryton.

**Survived.** The contamination itself is real and large: reading all 45 full texts,
**22 of 45 (49%) are not human case-control studies** — 15 animal (3xTgAD, R6/1,
Wistar rats, germ-free recolonisation), 3 with no healthy control arm, 2 case
reports (n=1, n=2), 2 with no primary cohort (a review, a Mendelian-randomisation
re-analysis). Every verdict carries a verbatim quote in `maindata_screen.json`.

**Did NOT survive — the headline hypothesis is refuted.** Filtering them changed the
agreement rate by **exactly nothing**: 0 decisive pairs flipped, against either
database, in any variant (exact McNemar p = 1.00 throughout). Not an underpowered
null — a true zero. The 22 papers contribute only 11 of 1,927 edges and 4 of 285
contributing papers; 18 of the 22 yielded no usable extraction at all.

**Why the number ever moved: disease mix, not quality.** The naive per-variant
comparison is confounded — dropping papers drops whole diseases, moving Disbiome's
reference denominator 506 → 364, so variants score different question sets. Paired
per (taxid, disease), the movement is entirely pairs entering/leaving the decisive
set. Dropping all 45 removes 21 decisive pairs, **19 of them Autism spectrum
disorder**, on which we agree with Disbiome 10/21 (48%) versus 112/147 (76%) for
pairs that stay. 17 of the 21 rest on 1–2 papers.

**Null, with power.** ASD's low agreement (52.6%, n=19 pairs) does **not** survive
paper-level permutation: those 19 pairs come from only **5 papers**. Observed gap
−0.225, null SD 0.139, minimum detectable gap 0.273, **p = 0.211**. The test cannot
resolve a gap this size at 5 papers. Pair-level shuffling would have given a false
positive — same trap as the earlier `diet_controlled` (FDR 0.243) and
"198 explanatory terms" (p=0.41) artifacts.

**Applied anyway, on construct-validity grounds, not the metric.** `graph.json`,
`kg.html`, `docs/index.html` rebuilt on the screened corpus: **326 papers, 281
contributing, 832 taxa, 1,916 edges, 225 contested, 625 containment links.** A human
microbe–disease graph should not carry edges whose evidence is transgenic-vs-wildtype
mice. The filter costs nothing and buys correctness of meaning — **it does not improve
agreement and must not be cited as though it did.**

**Infrastructure.** This environment's network policy denies `ftp.ncbi.nih.gov`
(CONNECT → 403), so the NCBI taxdump is unavailable. `build_kg.py` previously fell
straight through to string folding when the taxdump was missing — a silent
regression costing 681 taxid resolutions and all 625 containment links while still
printing a successful build. Added `taxonomy_cache.py`, which replays the resolution
recorded in `graph.json`; verified it reproduces the committed graph **exactly**
(nodes, edges, hierarchy, paper table all identical on a full rebuild diff). It is
valid **only for subsets** of that graph and reports cache misses rather than
silently under-resolving. On the external join it is marginally degraded (Disbiome
overlap 269 vs 272; Peryton 221 vs 221), so absolute agreement rates measured here
are ~0.2–1.1 points off taxdump-measured ones and are not new absolute figures;
all before/after deltas use the identical join and are sound.

**Next lever.** ASD is the weakest region of the graph (new disease, 1–2 papers per
edge, chance-level agreement) and the question is now specific: is ASD genuinely
less replicable, or did 5 papers land badly? That needs more ASD papers, not more
analysis of these 5.
