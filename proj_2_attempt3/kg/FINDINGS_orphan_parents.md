# 24 taxon nodes had their parent written in their own label and were detached anyway

**2026-09-14. Cloud session, CPU only. Deterministic detection; curated decisions.**

**Headline.** 170 of 883 taxon nodes have no parent containment link. 118 are
unresolved, and for **24 of those the parent is named in the label itself** —
`unclassified_f_Lachnospiraceae`, `norank_f_Christensenellaceae`,
`gut_metagenome_g_Faecalibacterium` — with that parent already present as a
resolved node. **14 are now linked; 13 are refused with recorded reasons**, and
the refusals are the more useful half.

## Why they were detached

`norm_taxon`'s placeholder branch adds a containment link by noticing that a
resolved name differs from the scientific name it landed on — so it only fires
for labels the taxonomy **did** resolve. These labels resolve to nothing at all
(`taxonomy.py` returns no taxid), so `taxonomy.py` can give them no lineage and
the placeholder branch never sees them. They fell between the two mechanisms.

Note the contrast that shows this is a real gap rather than a policy choice: the
*resolved* `unclassified X` nodes are handled correctly already. NCBI mints real
taxids for those subtrees (`unclassified Bacteroides` = 2646097, `unclassified
Pasteurellaceae` = 67757) and `build_kg.py` links each to its parent. The same
concept expressed in a spelling NCBI does not carry was orphaned.

This matters for the reason `build_kg.py` gives for modelling containment at all:
containment is not redundancy. *Lachnospiraceae* is depleted in 8 of 9 papers
while *Hungatella* inside it is enriched in 6 of 7, and that only survives if the
nesting is represented. An orphan node is invisible to that layer entirely.

## The refusals are the product

A generic "link to the taxon named in the label" rule would be wrong, and not
subtly. **Four of the 24 candidates are bacteriophages** — `Klebsiella virus
KP36`, `Streptococcus phage EJ 1`, `Enterococcus phage EFAP 1`, `Escherichia
virus JES2013`. A phage *infects* Klebsiella; it is not contained in it. Linking
them would assert four false taxonomic relations and let a query rolling up
Klebsiella absorb virus evidence.

This is the same lesson as `taxon_typos.py`, where edit distance would have
merged `Oscillospirales` into `Oscillospira`, and `species_synonyms.py`, where 91
of 115 child folds must *not* be split. Hence a curated table with its refusals
recorded in `orphan_parents.py`, not a rule.

| decision | n | examples |
|---|---|---|
| **LINK** | 14 | `unclassified_f_Lachnospiraceae` → Lachnospiraceae; `Ruminiclostridium-5` → Ruminiclostridium |
| **REFUSE — bacteriophage** | 4 | `Klebsiella virus KP36` |
| **REFUSE — joint two-taxon label** | 5 | `Escherichia-Shigella` (10 edges) |
| **REFUSE — parent not a node** | 4 | `norank_p_Parcubacteria`, `order SHA-98` |
| not in either table, correctly detached | 91 | no parent recoverable from the label |

Orphans: **170 → 156**. Hierarchy links: **713 → 727**.

## This sharpens the last open modelling call rather than resolving it

The joint labels are refused **on a substantive argument, not for lack of time**.
`Escherichia-Shigella` denotes 16S reads that could be *either* genus, so it is
**not a subset of Escherichia**: rolling it up would absorb evidence that may be
Shigella. Containment is the wrong primitive here, in both directions.

That is worth stating plainly because it names the third option the repo's
framing of this question has been missing. The choice has been posed as
*attribute to one genus / split across both / hold apart*, and it is held apart.
What a user actually wants — the joint node reachable from **either** parent
without either parent absorbing its evidence — needs a **different edge type**
("ambiguous assay", not `parent_of`). That is a schema decision for a human, and
with the disease-subtype question now answered
(`FINDINGS_disease_ontology.md`) it is **the last genuinely open modelling call
in the graph**. Five nodes and 13 edges depend on it.

## Two cross-rank disagreements became visible

Adding the links changed `rank_conflicts` on exactly 4 edges — 2 parent/child
pairs, each recorded from both sides. That is the containment layer doing its
job, not a side effect:

- **`Faecalibacterium` depleted in Alzheimer's across 16 papers, while
  `gut_metagenome_g_Faecalibacterium` is reported enriched** (1 paper, no shared
  paper with the parent). Worth a reviewer's attention: a metagenome-assembled
  entry of the genus pointing the opposite way to sixteen studies of the genus.
- `Flavobacteriaceae` depleted in Stroke vs `norank_p_Flavobacteriaceae`
  enriched (1 paper each, disjoint).

Both are correctly tagged `no_shared_paper` — different studies disagreeing
across ranks, which this project treats as a finding rather than an error.

## Verification

The rebuild was gated in advance: nodes, edges and the papers table must be
**identical**, hierarchy must grow by exactly 14, and any other change must be
explained. Result: nodes/edges/papers byte-identical as objects, edge *set*
identical, 0 hierarchy links removed and 14 added, no node field changed, the
only edge field touched is `rank_conflicts` on the 4 edges above, and the only
meta change is `n_hierarchy_links` 713 → 727. Second rebuild bit-identical to the
first. `rag_corpus.jsonl` and `kg.html`/`docs/index.html` regenerated; the corpus
matches the graph edge set by set equality.

The published page was then checked by actually rendering it —
`verify_viz.py` under Playwright, **32 passed, 0 failed** — because a blank-canvas
bug in this repo once passed every static check.

**Scope.** These are containment links. They change no edge direction and no
paper count, so agreement with Disbiome/Peryton cannot move and was not
re-measured. Justified on correctness of representation, not as an accuracy gain.

## Artifacts

`orphan_parents.py` → `orphan_parents.json` (links + refusals with reasons),
consumed by `build_kg.py`. Re-run after any change to taxon normalisation.
