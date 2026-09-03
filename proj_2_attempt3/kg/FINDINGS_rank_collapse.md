# Rank collapse, part 2: 115 child strings are still folded into their parents

Session of 2026-09-03. Code: `build_kg.py` (`PLACEHOLDER`, `NOT_PLACEHOLDER`,
`parent_taxid`), `taxonomy_cache.py`. Data: `child_folds.json`,
`selfcontra_packet.json`, `selfcontra_verdicts.json`.

The 2026-09-01 session split rank placeholders (`Erysipelotrichaceae UCG-003`)
out of their parent taxa and called it fixed. It was not fixed; it was
*started*. The pattern it used caught the UCG / ND / `group` forms and missed
the commonest family of all.

---

## How it surfaced

Not by inspection. By asking a cheap structural question of the extraction:
**does any paper call the same taxon both enriched and depleted for the same
disease?**

18 such claims, over 9 papers, and **every one of them lands on a contested
edge** — one of them (*Porphyromonas endodontalis* / Parkinson's) is contested
by a single paper voting both ways.

Splitting those 18 on a purely mechanical criterion — was the *same surface
string* written on both sides, or two *different* strings that folded to one
key?

- **14 are genuine paper-level contradictions.** The paper really does report
  both directions: at two body sites (this corpus contains oral-and-gut
  comparison papers), across subgroups or stages, or against different
  comparators. Adjudicated against the papers' own sentences with verbatim
  quotes in `selfcontra_verdicts.json`: 3 body site, 4 subgroup, 4 different
  comparison, 1 unclear, 2 genuine extraction errors.
- **4 are not contradictions at all.** They are two *different* taxa folded onto
  one node:

| enriched string | depleted string | both became |
|---|---|---|
| `Bifidobacterium` | `Bifidobacterium brevis` | genus *Bifidobacterium* |
| `Clostridiales Family XI` | `Clostridiales Ambiguous taxa` | order *Eubacteriales* |
| `Eubacterium biforme` | `Eubacterium rectale` | genus *Eubacterium* |
| `Lachnospiraceae` | `Lachnospiraceae species` | family *Lachnospiraceae* |

*Eubacterium biforme* and *E. rectale* are different organisms. Collapsing them
onto the genus manufactures a contradiction that no paper made.

> A note on method: this classification was first attempted by an LLM subagent
> reading the source sentences, which returned 6 "extraction errors". Four of
> those six were wrong — its own quoted evidence showed an oral-versus-gut
> contrast, or two distinct species. The deterministic same-string test settled
> it in one line. Where a mechanical test exists, prefer it to a judgement call.

## The general measurement

If a surface string resolves to a taxid whose scientific name it **extends**,
the string named a child and got filed under the parent. Corpus-wide:

**115 such strings, over 52 nodes, none of them flagged as placeholders.**
285 edges touch one; 76 of those are contested.

Three sub-classes, needing three different treatments:

| class | n | example | correct treatment |
|---|---:|---|---|
| `unspecified_member` | 29 | `Blautia spp`, `Agathobacter sp`, `Blautia unclassified` | **fold into the parent — already correct, do not change** |
| `placeholder_child` | 32 | `Prevotella 9`, `Coprococcus_1`, `Clostridium IV`, `Clostridiaceae 1` | own node + containment link |
| `named_child` | 54 | `Prevotella copri`, `Klebsiella pneumonia`, `Bacteroides vulgatus`, `Faecalibacterium prauznitzii` | own node at its real taxid + containment |

## The flagship edge

*Prevotella* / Parkinson's is the highest-weight edge in the graph — 17 papers —
and the README calls *Prevotella* load-bearing for the external join. Its node
had folded in **13 distinct surface strings**:

```
Prevotella          Prevotella 2      Prevotella 9      Prevotella_6
Prevotella_9        g_Prevotella_9    Prevotella spp    Prevotella VZCB
Prevotella buccae   Prevotella copri  Prevotella jejunii
Prevotella shaii    Prevotella timonensis
```

Five are *distinct SILVA genera*, not the genus *Prevotella*. Five are named
species.

## What was fixed

The `placeholder_child` class, using the mechanism the project already endorses
(own node, `resolved: false`, containment link to the parent). Extending the
pattern to bare numeric and roman-numeral suffixes, `cluster`, and
`Family <roman>`:

| | before | after |
|---|---:|---:|
| taxa | 892 | **918** |
| edges | 1,978 | **2,011** |
| placeholder nodes | 74 | **100** |
| containment links | 684 | **708** |
| contested edges | 223 | **219** |
| replicated edges | 441 | 438 |

**Five contested edges were contested only because placeholder children were
folded in** — false contradictions, now gone. *Prevotella*/PD drops 17 → 16
papers, and `Prevotella_9` emerges as its own **2-paper contested edge** whose
disagreement was invisible inside the genus.

Two follow-on defects, both found by executing rather than reading:

- **The new pattern over-matched bacteriophages.** `Enterococcus phage EFAP 1`
  and `Streptococcus phage EJ 1` end in a bare number. A phage is not a member
  of the genus it infects — `taxonomy_cache` already refused to hang them there,
  which is exactly why they broke. Added `NOT_PLACEHOLDER`
  (`virus|phage|bacteriophage|uncultured`), agreeing with the cache's existing
  `NOT_CONTAINED`.
- **The fix erased itself on rebuild — the same failure mode as last session's.**
  Placeholder nodes carry no taxid, and the parent was recoverable only from the
  containment link, which exists only when the parent is *itself* a node. So
  `Polaribacter_1` (no other *Polaribacter* edge in the corpus) lost its parent
  on the second build and decayed into a plain unresolved string node: 106 → 102
  placeholder nodes, while printing a successful build. Fixed durably by
  recording `parent_taxid` **on the node**, which `taxonomy_cache` now prefers.
  **Verified: two consecutive builds are now byte-identical on nodes, edges,
  hierarchy, papers and meta.**

## What was NOT fixed, and why

**The 54 `named_child` strings.** Splitting *Prevotella copri* out of
*Prevotella* correctly requires resolving it to its own species taxid — and this
environment's network policy denies `ftp.ncbi.nih.gov` (CONNECT → 403), so the
NCBI taxdump is unavailable and `taxonomy_cache` can only replay taxids already
in the graph. Splitting them here would create 54 nodes marked unresolved, which
would *lose* the external join with Disbiome and Peryton for exactly the species
that matter. That trade is not worth making blind.

**This is the top item for a machine with the taxdump** (the user's Mac).
`child_folds.json` carries all 115 with their classification, so the work is
mechanical: re-resolve each `named_child` string, give it its own node, link it
to the parent. Several are misspellings (`Faecalibacterium prauznitzii`,
`Bacteroides uniforms`, `Klebsiella pneumonia`) that will need fuzzy resolution
or they will simply become unresolved singletons.

## Agreement, for the fourth and fifth time, moved by nothing

Disbiome **71.9%**, Peryton **72.5%** after all of this session's corrections
(from 71.9% / 72.8%). The dedup correction alone: −0.0024 (p=0.665) and −0.0007
(p=0.885), against a minimum detectable change of ~0.013.

That is now five structural corrections that agreement cannot see. It is a
property of the validation, not a coincidence: the decisive set is dominated by
well-evidenced, unambiguously-named taxa, and every correction so far acts on
the margins. **These fixes are justified on correctness of meaning. None of them
should ever be cited as an accuracy improvement.**
