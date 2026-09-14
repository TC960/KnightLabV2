# The disease half of the graph, resolved against an ontology

**2026-09-14. Cloud session, CPU only. No GPU, no `MAIN_DATA.json`, no NCBI taxdump.**

**Headline.** The disease dimension now has the external authority the taxon
dimension has always had: MONDO is reachable from this environment even though
NCBI is not. Three results follow.

1. **Two wrong MONDO ids were shipped on 209 of 2,008 edges (10.4%)** and are
   fixed at source. The `Mild cognitive impairment` node — 13 papers, 154 edges —
   carried `MONDO:0005453`, which is *congenital heart disease*.
2. **MONDO confirms the hand-written containment tiers wherever it can see them**:
   2 of 2 checkable is-a claims CONFIRMED, 2 of 2 checkable rejections UPHELD.
   The human judgement call from 2026-09-10 was right.
3. **The 71-paper cognitive-decline cluster does NOT cohere microbially, and
   MONDO independently declines to model it.** This answers the three-session-old
   design question in a direction nobody had proposed: link those nodes for
   *retrieval*, do not pool their *evidence*, and specifically **do not fold MCI
   into Alzheimer's**. Details in §4 — this is the actionable part.

---

## 0. Why this was possible now

`ftp.ncbi.nih.gov` and `eutils.ncbi.nlm.nih.gov` are both blocked here (CONNECT →
403, probed in eight sessions; re-probed today, still 403). That has been read as
"no external ontology in the cloud". It is narrower than that:
`github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo`
returns **200 and 53 MB**. `purl.obolibrary.org` and `ebi.ac.uk` are blocked;
the GitHub release is not.

So `mondo.py` is the disease-side analog of `taxonomy.py`, with the same
conventions deliberately copied: exact label/synonym matching only, obsolete
terms followed through `replaced_by`, unresolved labels **reported rather than
guessed**, and a curated alias table that records its refusals.

**Coverage: 28 of 40 disease labels resolve** (25 by exact match, 3 by curated
alias). The 12 refusals are recorded with reasons in `disease_hierarchy.py` and
are not a defect — they are mostly graded or cause-specified cohort labels
(`Minimal hepatic encephalopathy`, `Hypertensive intracerebral hemorrhage`,
`Traumatic thoracic spinal cord injury`) that MONDO genuinely does not carry.

## 1. The positive control found errors in the graph, not in the resolver

`build_kg.py`'s 16 hand-curated ids must come back out of an independent
resolver. They did, 14/16 — and **both disagreements are errors in the curated
table**:

| node | was | what that id actually is | now |
|---|---|---|---|
| Mild cognitive impairment (13 papers, 154 edges) | `MONDO:0005453` | **congenital heart disease** | `None` |
| Autism spectrum disorder (55 edges) | `MONDO:0005260` | *autism* — a **child** of ASD | `MONDO:0005258` |

`MONDO:0005453` is not a near-miss or a deprecated id. MONDO contains **no term
named "mild cognitive impairment" at all** — zero of 104,643 normalised index
keys match the phrase. MCI is a clinical *stage*, not a MONDO disease, so `None`
is the correct value and must not be "fixed" back. The nearest term, `cognitive
disorder` (`MONDO:0002039`), is broader and would merge MCI with a dozen
unrelated conditions.

A third id was also settled. The NMDAR entry carried a comment saying the id was
"deliberately None rather than guessed" because EBI/OLS is blocked and a wrong
id is worse than no id. Both halves of that reasoning were right; the id is now
`MONDO:0021081` by lookup, because MONDO spells it *anti-NMDA receptor
encephalitis*.

`CLAUDE.md` already forbids joining on another database's taxid — Disbiome
records *Prevotella* as a species id where the genus was meant, and joining on it
silently dropped a 16-paper agreement. **This is the same failure one dimension
over, and it shipped because nothing had ever checked the disease half of an
edge against an ontology.** Every fidelity instrument in this repo scores the
taxon half.

### The control also caught my own parser first

MONDO's `is_a` lines carry trailing OBO qualifiers —
`is_a: MONDO:0005071 {source="https://orcid.org/..."} ! nervous system disorder`
— so ids were being read with the qualifier attached and **every ancestor lookup
silently returned nothing**. The self-test asserting "Parkinson's is under
nervous system disease" failed and stopped the run. Fifth time an instrument
here has been weaker than the thing it audited, and the **second consecutive
time a built-in control caught it rather than a spot-check**. The pattern is
cheap and should stay standard.

The control now reads `DISEASE_MAP` **live from `build_kg.py`** rather than
holding a copy. A control carrying its own copy of the table it checks cannot
detect drift in that table — which is how "174 contested edges" survived three
sessions after the number became 217. Control: **16/16**.

## 2. MONDO grades the hand-written tiers — and upholds them

`disease_containment.py` proposed 11 containment pairs in judgement tiers and
explicitly rejected four others. MONDO is an external check on that call, in
both directions:

| claim | MONDO |
|---|---|
| Intracerebral hemorrhage → Stroke (Tier A) | **CONFIRMED** (descendant, dist 2) |
| Alzheimer's disease → Dementia (Tier B) | **CONFIRMED** (descendant, dist 1) |
| Multiple system atrophy → Parkinson's (**rejected**) | **rejection upheld** (cousin) |
| Essential tremor → Parkinson's (**rejected**) | **rejection upheld** (sibling) |
| Sporadic CJD → Dementia (Tier B) | not is-a (cousin, dist 6) |
| the other 8 | unresolvable — MONDO has no such term |

Two confirmations and two upheld rejections out of four checkable claims. The 8
unresolvable ones cut both ways: the hand-written tiers are doing work MONDO
cannot do, and equally **cannot be validated by it**.

## 3. What a containment layer buys, and what it does not

Only **2 is-a links** can be formed between the graph's 40 disease nodes.

**For a node-scoped view the gap is large.** A view that filters edges by
`disease == "Dementia"` — which is what the per-disease panels and any naive
disease filter do — returns its own 6 papers / 47 taxa and misses the **46
papers and 289 taxa** on `Alzheimer's disease`, which MONDO says is a kind of
dementia. Likewise `Stroke` misses 3 papers on `Intracerebral hemorrhage`.

**For the PPR retriever the layer buys nothing, and this was measured rather
than assumed.** The layer was implemented in `graphrag.py` (opt-in, loaded from
`disease_hierarchy_links.json`) and the retrieved subgraph compared with and
without it:

| query | k | child node retrieved? | papers | diseases |
|---|---|---|---|---|
| `Dementia` (child: Alzheimer's) | 12 | **yes both ways** | 79 = 79 | 9 = 9 |
| `Dementia` | 25 | yes both ways | 152 = 152 | 10 = 10 |
| `Stroke` (child: ICH) | 12 | no both ways | 85 = 85 | 8 = 8 |
| `Stroke` | 25 | yes both ways | 157 = 157 | 11 = 11 |

**Identical in every cell.** The only measurable effect is a PPR score nudge on
one node (Alzheimer's 0.0499 → 0.0532) that changes no rank. The reason is
structural: personalized PageRank already connects any two diseases that share a
single taxon, and at the disease level this graph is nearly complete — 453 of the
possible pairs share enough taxa to be compared at all. **Adding 2 links to an
already-connected layer cannot add reach.**

So the layer is kept for a narrower and honest reason: it makes the relation
**explicit and attributed** in the returned subgraph (`Dementia contains
Alzheimer's disease (disease) [MONDO]`), which tells a reader *why* the two are
related and licenses a deliberate is-a expansion in a node-scoped view. It is
**not** a retrieval improvement and must not be reported as one. This is the
same shape as Task 2.5's own conclusion — the value of containment traversal is
expressiveness, not ranking.

**Does ontological proximity predict directional agreement?** No, not at this
size. Bucketing all 23 resolved multi-taxon diseases by MONDO relation:

| relation | pairs | decisive obs | agreement |
|---|---|---|---|
| is-a | 2 | 38 | **0.895** |
| sibling | 11 | 50 | 0.680 |
| cousin | 174 | 1,610 | 0.692 |

The is-a rate looks like a 20-point effect and **is not evidence**: those 38
observations come from exactly **two** disease pairs. Clustering the null at the
pair level (draw 2 pairs from the 185 non-is-a pairs, 20,000 draws):

> observed **34/38 = 0.895**, null **0.664 ± 0.178**, **p = 0.082**.
> A true is-a rate would have to exceed **0.963** to clear the null — an
> **MDE of +29.9 points** on a 66.4% base.

**Four is-a disease pairs would put a 20-point effect above the null. The graph
can form two.** That is an unusually concrete unblock: not "more papers", but
*two more linkable disease pairs*.

One confound was checked and is flat. Pairs with little overlap can only overlap
on widely-reported genera, where any two neurological cohorts tend to agree — so
agreement might track taxon ubiquity rather than disease relatedness. Over 249
pairs: **corr(mean shared-taxon breadth, agreement) = +0.000**, corr(overlap
size, agreement) = +0.043, and the breadth tertiles are flat (0.681 / 0.636 /
0.673). The metric is measuring disease, not fame.

**Paper overlap between disease nodes is zero.** Every pair. `build_kg.py` files
each paper under one predicted disease, so the shared-paper inflation that makes
the 73.0%/72.5% Disbiome/Peryton figures a blend of reading fidelity and
reproducibility **cannot operate here**. That trap was the main statistical risk
in this analysis and it does not apply.

## 4. The finding that changes the design call

The 2026-09-13 audit flagged the cognitive-decline cluster as the largest
modelling gap in the graph — six nodes, 71 papers, zero hierarchy links — and
recommended raising its priority. It was right to, and the answer is the opposite
of what "fold the subtypes" would predict.

**Within-cluster directional agreement, pair as the unit:**

| cluster | nodes | papers | pairs | pooled agreement | null | p |
|---|---|---|---|---|---|---|
| cognitive-decline | 7 | 72 | 16 | **0.592** (61/103) | 0.669 | 0.883 |
| cerebrovascular | 5 | 36 | 10 | 0.759 (41/54) | 0.668 | 0.126 |
| spinal-cord-injury | 3 | 8 | 2 | 0.625 (5/8) | 0.655 | 0.600 |
| hepatic encephalopathy | 2 | 2 | 0 | — | — | — |

Background (453 cross-cluster pairs): **0.672**.

**The cognitive cluster agrees LESS than two unrelated diseases do** (0.592 vs
0.672, p=0.883 — it fails in the wrong direction, so no power statement can
rescue it). And the structure inside it is sharp:

| pair | agreement |
|---|---|
| Alzheimer's / Dementia — *the MONDO-confirmed is-a link* | **15/16 = 0.938** |
| Dementia / Cognitive impairment | 7/8 = 0.875 |
| Alzheimer's / Cognitive impairment | 5/8 = 0.625 |
| **Alzheimer's / MCI** | **13/26 = 0.500** |
| **MCI / Cognitive impairment** | **9/18 = 0.500** |
| **MCI / Dementia** | **4/13 = 0.308** |

Every MCI pair sits at or below a coin flip. The one pair MONDO endorses sits at
93.8%.

**Two independent authorities agree, having been consulted separately.** MONDO's
vocabulary refuses to carry MCI as a disease; the graph's own microbial data
says MCI's directions are uncorrelated with Alzheimer's. Neither was derived from
the other.

**And it is not one contrarian paper.** The obvious artifact — a single deviant
MCI study driving every disagreement — was checked, because paper-level
clustering of discordance is a known property of this corpus (p=0.0003). It does
not hold here: disagreements are spread over **at least 6 of the 13 MCI papers**,
and the largest contributor (*Gut Microbiota and Neurovascular Patterns in
Amnestic MCI*) backs agreements too (3 vs 2 against Alzheimer's). The
disagreement is a property of the node, not of a study.

### So, for the PI

- **Link the cognitive nodes for retrieval. Do not pool their evidence.** A
  query for Alzheimer's should surface the 13 MCI papers as *related cohorts*.
  Merging their direction counts would average two populations that disagree half
  the time and would manufacture contested edges out of a real biological
  distinction.
- **Specifically, do not fold MCI into Alzheimer's.** Note this also
  retrospectively supports the Tier-C rejection of exactly that fold, which was
  made on clinical grounds ("MCI is a stage that may or may not convert") before
  any of this was measured.
- **Ship the 2 MONDO is-a links** (AD→Dementia, ICH→Stroke) — shipped, as an
  opt-in `disease_hierarchy_links.json` that leaves `graph.json` untouched. They
  are lookups, not judgements. But note §3: they buy **no** retrieval reach in
  the PPR retriever (measured, identical subgraphs), only explicit attribution.
  The 8× figure applies to node-scoped views, not to GraphRAG.
- The containment layer is justified **on retrieval and on correctness of
  meaning, not as an accuracy gain** — the standing rule for structural
  corrections in this repo, and it applies here.

## 5. Honest limits

- **The negative control has zero power and must not be read as passing or
  failing.** MSA/Parkinson's agrees 8/8 and essential tremor/Parkinson's 1/2,
  pooled 9/10 = 0.900, p=0.130 — but its MDE requires a rate above **1.000**, so
  the test cannot reject anything at 2 pairs. The 8/8 is *interesting* — MSA and
  Parkinson's are both α-synucleinopathies, so shared gut dysbiosis cutting
  across the clinical distinction is biologically plausible — but at n=8 from one
  pair it is a hypothesis, not a result.
- **The cerebrovascular cluster is directionally consistent with containment and
  underpowered**: +9.1 points observed against an MDE of +13.0.
- Only the cognitive cluster (16 pairs) has enough clusters to have any power at
  all. The SCI and hepatic tests are formally void and are reported as such.
- **Nothing here changes `graph.json`.** The three id corrections are in
  `build_kg.py` only; the rebuild is deliberately left for a machine with the
  taxdump, because rebuilding on a cloud checkout is how this repo lost work on
  2026-09-11. `graph.json` and `rag_corpus.jsonl` still carry the old ids.
- These are **identifier** corrections. They touch no edge, no direction and no
  count, so agreement with Disbiome/Peryton cannot move and was not re-measured.

## Artifacts

| file | what |
|---|---|
| `mondo.py` | MONDO parser/resolver + self-test + positive control (`--validate`) |
| `mondo_resolution.json` | all 40 labels, resolved or refused with reasons |
| `disease_hierarchy.py` → `.json` | tier grading, derived layer, retrieval payoff, ontology-shuffled null |
| `disease_hierarchy_power.py` → `.json` | pair-clustered null and the MDE arithmetic |
| `disease_cluster_coherence.py` → `.json` | within-cluster coherence, ubiquity confound, negative control |

`~/.mondo/mondo.obo` is cached, not committed (53 MB). Re-fetch:
`curl -sSL -o ~/.mondo/mondo.obo https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo`
