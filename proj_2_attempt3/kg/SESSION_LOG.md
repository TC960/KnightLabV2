# Session log

Newest first. Nulls and dead ends are logged as results.

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
