# The papers' own misspellings were splitting 41 taxa across two nodes

*2026-09-11. Method: anomaly-hunt the graph's structure, not test a hypothesis
about it. Third result this method has produced, after the placeholder rank
collapse and the 12 duplicate papers.*

---

## The question

254 of 925 taxon nodes never resolved to an NCBI taxid. That number has been
carried in `CLAUDE.md` for three sessions as "16S clade labels like
`[Eubacterium] ventriosum group`" — i.e. as a known, acceptable residue.

Nobody had asked whether it *is* all clade labels. It is not.

## What the sweep found

`taxon_typos.py --sweep` normalises every unresolved label (lowercase, strip
rank words and NCBI's `[misplaced genus]` brackets, collapse every separator to
a single space) and asks three questions. Two of them return real defects.

**A. Twelve concepts split across two nodes by punctuation alone.**

| one node | the other node |
|---|---|
| `Ruminococcaceae_UCG_002` | `Ruminococcaceae UCG-002` |
| `[Ruminococcus] gnavus group` | `Ruminococcus_gnavus_group` |
| `Christensenellaceae_R_7_group` | `Christensenellaceae R-7 group` |
| `[Eubacterium]_ventriosum_group` | `Eubacterium_ventriosum_group` |
| `Ruminiclostridium-5` | `Ruminiclostridium 5` |

…and seven more. This is a build defect, not a data problem. The unresolved
path in `norm_taxon` collapses every separator style — that fix was made on
2026-09-08, when `Escherichia-Shigella` turned out to be split over four nodes.
But the **placeholder branch returns before reaching it**, and keys on
`.replace("_", " ")` and nothing else. Hyphens and brackets therefore still
fragmented concepts, in exactly the way the earlier fix was written to prevent.

Two of the twelve are worse than a split node. In *Relationship between gut
microbiota and lymphocyte subsets in Chinese Han patients with spinal cord
injury*, one paper names `Christensenellaceae_R_7_group` in one figure and
`Christensenellaceae R-7 group` in another, and likewise for
`Eubacterium_ruminantium _group`. **A single study was casting two votes on one
edge.**

**B. Thirty-three misspellings.** `Fecalibacterium`, `Subdogranulum`,
`pesudobeautyrivibrio`, `Verruocomicrobiota`, `Enterobacteriaeae`,
`Lachinospiracea`, `Megamonus`, `Morganelia`, `Christensenea`, `Tuzzerella`,
`Parabacteroids`, `Diallister invisus`, and more.

## The fact the fix rests on, verified rather than assumed

**All 33 misspellings occur verbatim in their own source paper's full text**
(`taxon_typos.py --verify`, 33/33, checked against the 335 recoverable texts in
`all_usable_papers.json` + `extract_input.json` + `new_papers.json`).

These are the *papers'* spelling errors, copied faithfully by the extractor.
Not one is an extraction error. One paper alone —*Dysbiosis of the Beneficial
Gut Bacteria in Patients with Parkinson's Disease from India* — misspells three
taxa in a single abstract sentence:

> "…certain bacterial families like **Clostridia UCG 014**, **Cristensenellaceae**,
> and Oscillospiraceae are higher in abundance, and **Lachinospiracea**,
> Coriobacteriaceae and genera associated with short-chain fatty acids…"

That is a fidelity result in its own right, and it is why a curated table is the
right instrument: **there is nothing upstream to fix.** It also adds a third
independent line of evidence on extraction quality, alongside "no paper is
inverted" (2026-09-10) and the 85–94% single-paper agreement where our paper is
the curated source — none of which depend on the compromised in-house gold.

## Why it is curated and not an edit-distance rule

Because the near-misses are exactly the cases that matter, and a distance
threshold gets them wrong in both directions:

- **`Oscillospirales` vs `Oscillospira`** — an order and a genus, both real.
  So are **`Thermoactinomycetales` vs `Thermoactinomycetaceae`**; one paper
  names both in the same sentence.
- **`Prevotella_9`, `Ruminococcus_1`, `Coprococcus_2`, `Tyzzerella 4`,
  `Clostridiaceae 1`** are all ~1 edit from their parent genus and are
  **deliberately held apart** (`FINDINGS_rank_collapse.md`). An edit-distance
  rule would have silently undone the placeholder split — the single most
  consequential correction in this graph's history.
- **`Enterococcus phage EFAP 1` vs `Enterococcus phage EFRM31`** — two phages.
- **`Corynebacteria`** *is* a misspelling, but of what is undecidable: the paper
  writes "class Corynebacteria among phylum Actinobacteria", and no such class
  exists. Genus *Corynebacterium* and order *Corynebacteriales* are both
  plausible. **Refused.**
- **`lactic acid bacteria`** is not a taxon at all — a physiological guild
  spanning several genera, measured in that paper with group-specific primers.
  **Refused**, and flagged as a human decision: it should probably not be a node.
- **bare `UCG-002`** is a real SILVA placeholder that does not say whose. In one
  of its two papers it sits beside *Ruminococcus*; in the other it does not
  appear in the text at all. **Refused.**

All 13 refusals are recorded in `taxon_typos.py` with their reasons, so a later
sweep re-derives the same candidates and reads the verdicts instead of
re-litigating them.

## Two bugs I introduced and the checks that caught them

Recorded because both are the kind that pass every static reading.

**1. My first regex was too loose.** Letting the SILVA numeric suffix be joined
by a hyphen (`[ _-]\d{1,3}$`) swallowed strain designations: `Azospirillum sp.
47-25` and `Lachnospiraceae bacterium MC-35` both resolve to real taxids and
were demoted to unresolved placeholders. It surfaced as `n_taxa_resolved`
falling by 2 in the rebuild diff — **not** by reading the pattern. Narrowed to a
single-word stem (`^[A-Za-z]+-\d{1,3}$`), which is what the two genuine cases
(`Ruminiclostridium-5`, `Ruminiclostridium-1`) actually look like.

**2. The build is self-referential, and a bad intermediate poisons the next
build.** `taxonomy_cache.py` replays `graph.json`'s own resolution — and
`build_kg.py` *overwrites* `graph.json`. So the buggy intermediate became the
authority for the following build, which silently changed two display labels
(`Azospirillum sp.` → `Azospirillum sp. 47-25`). Fixed by restoring the pristine
graph before the real rebuild. **This hazard is not specific to this change and
will bite again:** any future correction must rebuild from a known-good
`graph.json`, never from the output of a failed attempt.

## Effect on the graph

| | before | after |
|---|---|---|
| taxa | 925 | 883 |
| edges | 2,034 | 2,008 |
| replicated (>1 paper) | 440 | 447 |
| contested | 217 | 221 |
| placeholder nodes | 104 | 94 |
| resolved to a taxid | 671 | 671 |

No node lost resolution. No edge lost contested status. **Four edges gained
it** — and these are the substantive payoff, because each was unanimous *only*
because the dissenting paper's spelling put it on a node of its own:

- *Bifidobacterium* / Parkinson's — a paper reporting "a significant decrease
  of … **Bifidobacteria** in PD patients" was on a separate node.
- *Butyricicoccus* / Multiple sclerosis — `Butyricococcus` enriched vs
  `Butyricicoccus` depleted, in two different cohorts.
- *Clostridia_UCG-014* / Parkinson's.
- *Verrucomicrobiota* / Alzheimer's.

Four more merges created **new replication** where the graph had seen two
unrelated singletons (*Subdoligranulum*/ALS, *Faecalibacterium prausnitzii*/
Dementia, *Ruminococcus gauvreauii group*/Parkinson's, *Pseudobutyrivibrio*/SCI).

## What this is NOT

**Not an accuracy gain, and the numbers say so.**

| | before | after | Δ |
|---|---|---|---|
| Disbiome agreement | 73.0% | 73.4% | +0.004 |
| Peryton agreement | 72.5% | 72.3% | −0.002 |

Both moves are well under the ~0.013 this corpus can resolve, and **the two
databases move in opposite directions** — the signature of noise, not signal.
This is the **sixth** structural correction to land below resolution, which is
now clearly a property of the validation set (dominated by well-evidenced,
unambiguously named taxa) rather than a run of coincidences.

The honest gain is **recall**: +1 overlapping pair against each database
(Disbiome 268→269, Peryton 223→224), because folding a misspelling reconnects a
taxon that previously failed to join.

The justification is correctness of meaning: *Fecalibacterium* and
*Faecalibacterium* are one genus, and the graph said two.

## Verification

- Three consecutive rebuilds byte-identical — a fixed point, not a drifting one.
- `kg.html` byte-identical on rebuild; `docs/index.html` re-synced.
- `verify_viz.py` 32/32 assertions passing in Chromium.
- `taxon_typos.py --verify` 33/33 against source full text.

## Open, and handed to a human

- **`lactic acid bacteria` should probably not be a taxon node.** It is a
  functional guild. Removing it is a vocabulary decision, not a spelling fix.
- **`Corynebacteria` and bare `UCG-002`** are unresolvable from the text alone.
- **254 → 221 unresolved labels.** The remainder really are clade labels
  (`SMB53`, `cc115`, `PAC000195_g`, `Marine_Methylotrophic_Group_3`) plus real
  taxa the taxdump copy does not cover (`Anaerostignum`, `Mogibacteriaceae`,
  `Erysipelatoclostridiaceae`). The second group would shrink with a current
  taxdump, which `ftp.ncbi.nih.gov` has refused in five consecutive cloud
  sessions (CONNECT → 403).
