# The named-child rank collapse, and what fixing it did

*2026-09-04. Code: `resolve_named_children.py`, `build_kg.py`,
`analyze_species_split.py`. Data: `named_child_taxids.json`,
`species_split_effect.json`, `species_extdb_coverage.json`.*

## The defect

`taxonomy.py` resolves a taxon string by trimming the qualifier tail until
something matches NCBI. For a real binomial the genus always matches, so
`Prevotella copri` resolved to *Prevotella* (838), `Eubacterium rectale` to
*Eubacterium* (1730), `Klebsiella pneumonia` to *Klebsiella* (570). The species
never got a node; its evidence was counted as the genus's.

This is the rank collapse the project explicitly forbids, and it was the top
open defect. `child_folds.json` had all 115 instances classified since
2026-09-03; 54 were left unfixed because splitting them needs real species
taxids, and the analysis environment's network policy denies
`ftp.ncbi.nih.gov` (CONNECT → 403).

**Scale before the fix:** 95 observations over **64 edges, 33 of them
contested**. Two examples fix the character of it:

- ***Eubacterium* / Multiple sclerosis** was 1 up / 3 down — contested. All four
  papers named a species, *E. rectale* or *E. biforme*. **No paper measured the
  genus.** The edge and its contradiction were both artefacts of the fold.
- ***Prevotella* / Parkinson's**, the graph's flagship edge, drew 8 of its 17
  observations from *P. copri* and `Prevotella VZCB`.

## Getting taxids without the taxdump

`taxoniq` ships NCBI's taxon database as a PyPI data package, and PyPI is
reachable where NCBI's FTP is not. It resolves 2.6M scientific names to taxid,
rank and parent.

**It is not a taxdump.** It carries scientific names only — no synonyms, no
`merged.dmp`. `Bacteroidetes` and `Firmicutes` both raise `KeyError`, and
folding exactly those synonyms is `taxonomy.py`'s whole job. So it is used as an
**additive lookup for 54 named strings**, never as a resolver, and every taxid it
supplied is recorded in `named_child_taxids.json` with its provenance.

## `named_child` was not one category

| verdict | n | treatment |
|---|---:|---|
| **species** — a real binomial with an NCBI taxid | 25 | own node, keyed on the species taxid, contained by the genus |
| **clade** — strain / bin / pipeline id (`Clostridium_XlVa`, `Dorea asp: CAG:317`, `Turicibacter sp001543345`) | 24 | own node, **no taxid** |
| **group_label** — names more than one taxon (`Escherichia_Shigella`, `Streptococcus salivarius/thermophilus`) | 5 | own node, **no taxid** |

The last two classes get no taxid rather than a guessed one, so they stay out of
the resolved count and out of the external join. Keeping `Escherichia_Shigella`
as *Escherichia* asserts a genus the paper did not name — SILVA reports the pair
precisely because 16S cannot separate them.

## Adjudicated, then verified — not fuzzy-matched

Automatic matching was tried first and failed in both directions:

- **Constrained to the folded-into genus**, it lost four of the five
  highest-evidence species, because the genus is the thing that has moved:
  NCBI has reassigned *B. dorei*, *B. vulgatus* and *B. plebeius* to
  *Phocaeicola*, and *P. copri* to *Segatella*.
- **Widened to the family**, it matched `Eubacterium_g4` onto the genus
  *Eubacterium* itself — reintroducing the exact collapse being fixed — and
  `Lachnospiraceae_Eubacterium` onto an unrelated species at ratio 0.86.

A similarity score cannot tell a misspelling from a different taxon, because it
does not know what the string means. So the 54 are adjudicated by hand into a
table of *proposals*, and `verify()` accepts none of them on the proposal's
authority: the name must exist in NCBI, be species rank, and share its epithet
with the surface string. **It rejected six of my own proposals** — `Prevotella
buccae` is now *Segatella buccae*, `P. shahii` is *Hoylesella shahii*, and
`Lawsonibacter phoceensis` is not in NCBI at all and stays an unresolved clade.
That is what the gate is for.

The species-rank requirement is the load-bearing one: it is what stops a clade
label collapsing onto a genus a second time.

## The result

| | before | after |
|---|---:|---:|
| taxa | 918 | **946** |
| edges | 2,011 | **2,059** |
| containment links | 708 | **732** |
| contested edges | 219 | **209** |
| Disbiome agreement | 71.9% | **73.3%** |
| Peryton agreement | 72.5% | **73.2%** |

**24 edges vanished.** Every one is an edge no paper ever measured — the whole
of *Eubacterium*/MS, *Escherichia*/Alzheimer's, *Prevotella*/Dementia.

**12 edges went contested → decisive; 2 went the other way.** *Lachnospiraceae*/
Alzheimer's (1 up / 6 down) and *Faecalibacterium*/MS (1 up / 6 down) were each
contested by a single dissenting vote that turned out to be a species or clade
string. With it moved to its own node, both are unanimous and both agree with
the curated databases. **The fold was manufacturing contradictions.**

The two that went the other way are equally real: merging the misspelling
`Faecalibacterium prauznitzii` into *F. prausnitzii* brought one enrichment
paper onto a 7-paper depleted edge. That contest is genuine and was previously
hidden inside the genus.

## The agreement gain is coverage, not accuracy — and this is testable

The rate moved more than any of the five previous corrections, which is exactly
when to be most careful. `agreement_metric.py` does not apply: its null
resamples dropped papers, and this correction drops none — it re-keys strings.
So decompose the rate instead. A rate can rise three ways and only one of them
is the graph getting something right.

**Disbiome:** 166 → 176 decisive pairs, 71.7% → 73.3%.

| | |
|---|---|
| COVERAGE | **14 pairs added** (13 agree, 1 disagrees) |
| ATTRITION | 4 pairs dropped (3 agreed, 1 disagreed) |
| CORRECTION | **0 pairs present in both changed verdict** |
| **on the 162 pairs present in both** | **71.605% → 71.605%** |

**Peryton:** 137 → 138 pairs; on the 136 present in both, 72.794% → 72.794%.

**Unchanged to three decimal places on both, and not one pair flipped.** The
entire headline movement is new pairs entering the comparison. So this is the
**sixth** structural correction that cannot be cited as an accuracy gain, and it
strengthens rather than weakens the standing conclusion: the decisive set is
dominated by well-evidenced, unambiguously named taxa, and corrections act on
the margins.

What it *is* is a coverage gain, and that is worth having on its own terms:
**10 net new checkable pairs**, most of them edges that were unusable because a
folded species was manufacturing a contest.

**The one statistical claim available, tested and NOT significant.** 13 of the
14 added Disbiome pairs agree, against a base rate of 71.6% — one-sided binomial
**p = 0.061**. Suggestive, not significant, and post-hoc besides: the pairs were
identified after seeing which ones moved. At n = 14 the test cannot resolve
anything smaller than a very large effect. Reported because it was run, not
because it supports anything.

## What this changes downstream

- ***Prevotella*/Parkinson's**, the edge the README calls load-bearing, is now
  16 papers → 11, with *P. copri* (6 observations) standing as its own node.
  Anyone quoting "17 papers" for the genus was quoting a rank collapse.
- **The external join now agrees on rank.** Disbiome files 34 records under
  `Prevotella copri` and Peryton 14; while our node was the genus, those records
  joined onto our *genus* edge — the same collapse on the reference side.
  `validate_external.py` now applies the same table, so both sides key on the
  same taxid. (With the split graph this is belt-and-braces — the node's aliases
  already carry the strings — but it must not depend on that.)
- **The graph is a fixed point again.** Building twice caught this repo's
  recurring failure mode in a new form: build 2 asserted 11 containment links a
  second time, because the replay cache had by then read the split parents back
  out of `graph.json`. Deduplicated on (parent, child); two consecutive builds
  are now byte-identical.

## What is left

- **`Lawsonibacter phoceensis`** (1 observation) has no NCBI name and stays an
  unresolved clade. A real taxdump with synonyms would settle it, and would also
  let the 24 clade-verdict strings be checked rather than accepted.
- The **two ambiguous picks** — NCBI holds two taxids each for *Blautia
  massiliensis* and *Ruminococcus bicirculans*, disambiguated by describing
  author. The earlier (gut isolate) was taken in both cases. Only consistency
  matters for the join, since both sides route through the same table, but it is
  a judgement call and is named in the source.
- **258 → 283 taxa still carry no taxid**, now including the 29 clade and group
  labels this split deliberately declines to guess at. That number going *up* is
  the fix working.

---

## Postscript: auditing the fix found two bugs in the fix

`taxoniq` also makes it possible, for the first time, to check the graph's
**containment hierarchy** against an independent source. `taxonomy_cache` stores
"nearest-present-ancestor links, not full NCBI lineages" by its own docstring, so
containment has only ever been as good as whatever taxonomy built the graph.

`audit_containment.py`, over 616 taxid-to-taxid links:

| | |
|---|---:|
| confirmed by NCBI ancestry | 601 |
| deliberate split links (the paper's own naming) | 11 |
| true but not the nearest ancestor | 7 |
| taxid NCBI 2024 no longer carries | 2 |
| **not an ancestor at all — DEFECT** | **2** |

The two defects are `Oscillospiraceae ⊃ Gemmiger` and `Clostridium ⊃
[Clostridium] innocuum`, where the bracket in the name is NCBI saying the
placement is wrong. Both are taxonomy drift rather than build errors, and both
are **left in place**: fixing two links by overriding the graph's own taxonomy
with a second, synonym-less source would trade a 0.3% error rate for a
mixed-provenance hierarchy. Mechanical once the taxdump is reachable.

Writing the audit paid for itself immediately by finding two bugs in this
session's own change:

- **The split flag was lost on the three most interesting nodes.**
  `split_from_parent` keyed off `taxon_how`, which is a `setdefault` — the first
  surface string to reach a node wins. For *Phocaeicola dorei*, *Holdemanella
  biformis* and *Enterocloster clostridioformis* the corpus **already had a node
  under the current name**, the legacy string merged into it, and the flag never
  got set on exactly the cases most worth auditing. (Checked whether those
  merges pooled evidence: they did not — no shared disease between the two
  names. Only the *F. prausnitzii* misspelling pooled, and that created the one
  genuine new contest.)
- **A misspelling became a node label.** Taxid 573 displayed as *"Klebsiella
  pneumonia"*. Labelling a split node with its surface string is right for a
  superseded name and wrong for a typo, and no string property separates them:
  `rectale` → `rectalis` scores 0.80 and is a real reclassification,
  `pneumonia` → `pneumoniae` scores 0.95 and is a typo. The 13 misspellings now
  carry an explicit display name.

Re-running the child-fold detection on the rebuilt graph leaves **36 strings
that still extend their node, 24 of which are `X sp.` / `X spp.` /
`X unclassified` where folding is correct.** The ~10 residue — GTDB sub-genus
suffixes (`Blautia_A`, `Fusobacterium_A`), truncated placeholder tails, and
forms the regex misses (`Christensenellaceae group R7`, `Atopobium cluster`) —
is logged and deliberately not chased. It would be a seventh structural
correction, and this session has just re-confirmed that agreement cannot see one
of that size.
