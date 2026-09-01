# Session log

Newest first. Nulls and dead ends are logged as results.

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
