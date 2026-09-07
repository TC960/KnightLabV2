# The species split puts renamed organisms in the wrong place, and does it inconsistently

*Session of 2026-09-07 (cloud, CPU-only, no MAIN_DATA, no taxdump).*
*Audit target: `origin/claude/kg-species-split` @ c5f8de2, which is **not merged** into `main`.*

## Summary

The unmerged `claude/kg-species-split` branch closes the project's stated top open
defect — the 54 named children folded into their parents — and does it well: the
taxid resolutions are correct, and the write-up is honest about the agreement
number being coverage rather than accuracy. **The resolutions should be kept.**

But the branch's *containment* links are wrong for the renamed species, and wrong
in an inconsistent way that its own audit reports as "deliberate":

| | `main` @ 6b0a131 | branch @ c5f8de2 |
|---|---|---|
| taxid→taxid containment links | 608 | 616 |
| **FALSE** (parent is not an NCBI ancestor) | **2** | **13** |
| children with **more than one** parent | **0** | **4** |

The split introduces **11 new false containment links and all 4 multi-parent
children**. Verified with `audit_containment_ncbi.py`, an independent check that
asks NCBI one question per link — *is the parent anywhere in the child's
lineage?* — and shares no logic with the build.

This is a deterministic structural check against an external reference, not a
statistical claim about the corpus, so no permutation test applies. That is also
why it is trustworthy at this n: it does not depend on the corpus size at all.

## What is actually wrong

Ten species were split out. NCBI has since moved every one of them into a
different genus, and in three cases a different family. The branch parents them
by the genus in their **obsolete binomial**:

| node label in branch | NCBI current name | parented under | true genus |
|---|---|---|---|
| Prevotella copri | *Segatella copri* | Prevotella | Segatella |
| Prevotella buccae | *Segatella buccae* | Prevotella | Segatella |
| Prevotella shahii | *Hoylesella shahii* | Prevotella | Hoylesella |
| Prevotella timonensis | *Hoylesella timonensis* | Prevotella | Hoylesella |
| Bacteroides vulgatus | *Phocaeicola vulgatus* | Bacteroides | Phocaeicola |
| Bacteroides coprophilus | *Phocaeicola coprophilus* | Bacteroides | Phocaeicola |
| Eubacterium rectale | *Agathobacter rectalis* | Eubacterium | Agathobacter |
| Phocaeicola dorei | *Phocaeicola dorei* | Bacteroides **and** Phocaeicola | Phocaeicola |
| Bacteroides plebeius | *Phocaeicola plebeius* | Bacteroides **and** Phocaeicola | Phocaeicola |
| Holdemanella biformis | *Holdemanella biformis* | Eubacterium **and** Holdemanella | Holdemanella |
| Enterocloster clostridioformis | *Enterocloster clostridioformis* | Clostridium **and** Lachnospiraceae | Enterocloster |

`audit_containment.py`'s docstring defends the old-genus link as deliberate — the
graph records "the genus the paper named it under," a claim about the corpus
rather than about NCBI. That is a coherent position. **The defect is that the
branch does not actually implement it.**

Four of the eleven get the correct link *as well*, and the choice of which four
is arbitrary. *Phocaeicola dorei* and *Phocaeicola plebeius* are linked to
*Phocaeicola*; *Phocaeicola vulgatus* and *Phocaeicola coprophilus* are not —
same genus, same situation, same build, opposite treatment. *Agathobacter* is
present in the graph as a node, yet *Agathobacter rectalis* has no link to it.

The labels split the same way and just as arbitrarily: some nodes carry the
obsolete binomial (`Prevotella copri`, `Bacteroides vulgatus`, `Eubacterium
rectale`), others the current one (`Phocaeicola dorei`, `Holdemanella biformis`,
`Enterocloster clostridioformis`).

### The mechanism

The parent a species gets depends on **which spelling the corpus happened to
use**, not on the organism. Nodes whose aliases include the current name
(`['Bacteroides dorei', 'Phocaeicola dorei']`) picked up the correct-genus link
through the ordinary resolution path; nodes mentioned only under the obsolete
name (`['Prevotella copri']`) got only the obsolete-genus link. So two papers
reporting the same organism under different names place it at different points
in the hierarchy — and a node acquires a second parent whenever both spellings
appear.

That is the "does a surface string extend the name it resolved to" class of
structural anomaly this project has twice found real defects in. It is not a
policy about provenance; it is an accident of vocabulary.

## Why it matters downstream

Containment is not decorative any more. Two things consume it:

- **`analyze_rank_conflict.py`**, whose result — related taxa agree on direction
  0.8903 of the time within a paper, z=15.5 — enumerates parent/child pairs. Its
  pair set is only meaningful if ancestry is consistent. Multi-parent nodes are
  counted twice, under two different families.
- **GraphRAG traversal**, where containment is the whole reason a query about a
  genus reaches its family.

The concrete case is *Eubacterium rectale* → *Agathobacter rectalis*. It carries
**12 papers, 12 depleted / 0 enriched, across 8 diseases** — one of the most
consistently reported taxa in the corpus, and three of those papers are
Parkinson's. NCBI places *Agathobacter* in **Lachnospiraceae**; the branch files
it under *Eubacterium*, i.e. Eubacteriaceae. *Enterocloster clostridioformis*
belongs in Lachnospiraceae too.

So the correction is not merely tidiness — it moves a block of unanimous
depletion evidence *into* Lachnospiraceae, the family behind the project's
flagship claim (*Lachnospiraceae* depleted in Parkinson's, 8 of 9 papers). The
branch as written keeps that evidence out of the family it belongs to.

## The fix, and why it costs nothing

Parent each split species by its **true nearest ancestor already present in the
graph**. Every one of the ten has one — no new nodes are required:

| species | correct nearest present ancestor |
|---|---|
| *Segatella copri*, *Segatella buccae*, *Hoylesella shahii* | Prevotellaceae (171552) |
| *Phocaeicola vulgatus / dorei / coprophilus / plebeius* | Phocaeicola (909656) |
| *Agathobacter rectalis* | Agathobacter (1766253) |
| *Holdemanella biformis* | Holdemanella (1573535) |
| *Enterocloster clostridioformis* | Lachnospiraceae (186803) |

This removes all 11 false links and all 4 multi-parent children, and returns the
graph to one containment parent per node.

**Nothing about provenance is lost.** The obsolete binomial is already kept in
the node's `aliases` — that is where "the paper said *Prevotella copri*" is
recorded, and it is a better place for it than a containment edge that asserts an
ancestry NCBI contradicts. Labels should follow the same rule: current
scientific name, obsolete binomial as an alias, applied to all ten.

## Method note, and a warning about the obvious shortcut

`ftp.ncbi.nih.gov` is still denied here (CONNECT → 403), so there is no taxdump.
Both this audit and the branch use the `ncbi-taxon-db` wheel behind `taxoniq`,
installed from PyPI, which bundles an NCBI 2024-09 snapshot: 2.6M scientific
names with ranks and full lineages. **It has no synonym table**, so
`Bacteroidetes` and `Firmicutes` do not resolve in it and it must never be
substituted for `taxonomy.py`. Ancestry, which is all this audit needs, is
complete.

One dead end worth recording, because it is the obvious way to do this and it is
dangerous. `resolve_named_children.py` (this session's version) also tries fuzzy
matching *within the parent genus* at a 0.86 similarity threshold. It produces
confident, wrong answers on exactly the species that matter:

| surface | fuzzy answer | truth |
|---|---|---|
| Prevotella copri | *Prevotella corporis* (28128) | *Segatella copri* (165179) |
| Bacteroides vulgatus | *Bacteroides ovatus* (28116) | *Phocaeicola vulgatus* (821) |
| Eubacterium biforme | *Eubacterium uniforme* (39495) | *Holdemanella biformis* (1735) |
| Bacteroides coprophilus | *Bacteroides coprosuis* (151276) | *Phocaeicola coprophilus* (387090) |
| Prevotella shaii | *Prevotella amnii* (419005) | *Hoylesella shahii* (228603) |

Five wrong in twenty, and the failure is **systematic, not random**: when a
species is moved out of a genus, what remains in that genus are its former
congeners, so a nearest-string search inside the old genus is biased toward a
wrong sibling precisely for the reclassified species. Each wrong answer is a real
organism, so it would join cleanly against Disbiome and Peryton under the wrong
taxon and never look like an error. Do not resolve renamed taxa by
similarity-within-genus.

## Status and recommendation

`claude/kg-species-split` is **8 commits behind `main`** (branched at a7d4437;
`main` is now 6b0a131) and both sides rewrote `build_kg.py`, `graph.json`,
`kg.html` and `docs/index.html`. It cannot be fast-forwarded, and `graph.json` is
generated, so the merge must take the *code* and then rebuild the artifact with
the repo's own tooling rather than resolving the JSON by hand.

Recommended order:

1. Merge `main` into the branch, taking `main`'s side for generated artifacts.
2. Apply the parentage fix above in `build_kg.py`.
3. Rebuild, then **rebuild again and diff** — two fixes in this repo have
   silently erased themselves on rebuild while printing success.
4. Gate on `audit_containment_ncbi.py` returning **2** false links (the `main`
   baseline: *Gemmiger*/Oscillospiraceae and *[Clostridium] innocuum*, both
   pre-existing and unrelated to the split) and **0** multi-parent children.
5. Re-run `verify_viz.py` and `validate_external.py`.

## Applied and verified (same session)

All five steps were carried out on `claude/kg-species-split-merged`.

`main` merged in cleanly for **`build_kg.py`** — the only conflicts were generated
artifacts (`graph.json`, `kg.html`, `docs/index.html`) plus `SESSION_LOG.md`, so
the artifacts were regenerated with the repo's own tooling rather than resolved by
hand. `add_true_ancestors.py` (new) writes each split species' true NCBI ancestor
chain into `named_child_taxids.json`; `build_kg.py` links each to the nearest of
those present in the graph and no longer lets the lineage walk add a second
parent.

Verification, all by execution:

| check | result |
|---|---|
| `audit_containment_ncbi.py` (independent) | 613 links, **2** false, **0** multi-parent — the `main` baseline |
| `audit_containment.py` (theirs) | 609 confirmed, **0** deliberate, same 2 defects |
| `build_kg.py` twice, diffed | **byte-identical** — fixed point |
| `build_viz.py` twice, diffed | **byte-identical** |
| `verify_viz.py` (Chromium, real clicks) | **19 passed, 0 failed** |
| `validate_external.py` | Disbiome 73.3%, Peryton 73.4% |

**The fix is agreement-neutral by construction, and that was verified rather than
asserted.** A controlled build with the fix removed gives an identical node id
set and identical taxids, and edges differing only in the display `taxon` string
and in `rank_conflicts` (which is derived from the hierarchy, so it *should*
change). The hierarchy differs by exactly **11 links removed, 0 added**. Since
the external join is on taxid, it cannot have moved.

So the 73.3% / 73.4% above is the *species split's* number, inherited from the
branch — not this correction's. Ten node labels now show NCBI's current name
(`Prevotella copri` → `Segatella copri`), with the obsolete binomial in `aliases`;
NCBI's nomenclatural authority suffix (`Blautia massiliensis (ex Durand et al.
2017)`) is stripped for display.

Do not report the result as an accuracy gain. The branch's own decomposition
already shows agreement on the *common* pairs is unchanged to four decimal places
(Disbiome 0.7160 → 0.7160, Peryton 0.7279 → 0.7279); the headline movement is
added coverage. This correction is justified on correctness of meaning, like the
five structural corrections before it.
