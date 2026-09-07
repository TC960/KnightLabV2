# Does the disease dimension carry information? Parkinson's yes, the rest no

**Session of 2026-09-05.** All *analysis* below is read-only on `graph.json` (272
contributing papers, 918 taxa, 40 diseases, 2,011 edges), so no result here
depends on a rebuild. The graph was rebuilt afterwards to ship the annotation
described in the addendum; that rebuild changed no pre-existing field and no
published number.

Scripts: `disease_containment.py`, `disease_specificity.py`,
`disease_specificity_confounds.py`, `disease_specificity_pd.py`.

---

## The headline

Two papers that report the same taxon agree on its direction **71.6%** of the
time when they study the **same** disease and **65.7%** when they study
**different** ones. The gap is +0.0591, z=3.39, **p=0.0014** under a paper-level
permutation of the disease label. It is not country and not sequencing platform.

**But it is one disease.** Parkinson's papers agree with each other at **0.807**;
no other disease comes close, and Alzheimer's — with 47 papers and ample power —
sits at **0.608**, *below* the rate for papers studying entirely different
diseases. Remove Parkinson's 67 papers and the gap falls to +0.0187, **p=0.179**.

> The disease dimension of this graph carries reproducible directional
> information for Parkinson's disease and, at n=272 papers, for nothing else.

This is the most consequential caveat found so far on how the graph should be
read, and it is a caveat about the *literature*, not about the extractor.

---

## How this was arrived at (the null came first)

The assigned open item was the disease-side fragmentation nobody had quantified:
40 disease nodes with no containment links, so `Intracerebral hemorrhage` sits
beside `Stroke` and `Chronic traumatic complete spinal cord injury` beside
`Spinal cord injury`, unconnected. The taxon dimension models containment with
708 links; the disease dimension has none.

The precedent said this should be productive. Folding three spellings of
anti-NMDAR encephalitis turned 4 apparently single-paper edges into replicated
ones, 3 of them contested — fragmentation hides both replication and
contradiction.

**It was a null.** Seven Tier-A clinical is-a pairs were tested for whether a
subtype's directional profile resembles its parent's more than an unrelated
disease's does:

| child → parent | shared taxa | decisive agreement | p |
|---|---|---|---|
| Intracerebral hemorrhage → Stroke | 35 | 19/22 = 0.86 | 0.43 |
| Hypertensive ICH → Intracerebral hemorrhage | 5 | 4/5 = 0.80 | 0.51 |
| Poststroke aphasia → Stroke | 7 | 0/3 = 0.00 | 0.75 |
| Hemorrhagic transformation → Stroke | 9 | 4/6 = 0.67 | 0.50 |
| Chronic traumatic complete SCI → SCI | 6 | 3/5 = 0.60 | 0.28 |
| Traumatic thoracic SCI → SCI | 4 | 3/4 = 0.75 | 0.19 |
| Minimal hepatic encephalopathy → Hepatic encephalopathy | 0 | untestable | — |

Nothing survives, and with 3–22 decisive shared taxa per pair nothing could have.
**This is a power statement, not evidence of absence.** Tier B (AD → Dementia,
the cognitive-decline continuum) behaves the same way.

The reason the null is uninformative, though, is the interesting part: the
comparison baseline was already high. **Arbitrary disease pairs agree 67.2%**
(1,625/2,419 decisive shared taxa over 251 disease pairs) against a marginal
chance rate of **51.4%** (1,045 enriched / 747 depleted decisive edges). A
subtype could not look special because *nothing* looks special against that
baseline.

That reframed the question from "are subtypes related?" to **"does disease
identity explain direction at all?"** — which, unlike the subtype question, has
power.

---

## The test

**Statistic.** Over all 23,627 unordered pairs of distinct papers that report the
same taxon: (agreement rate among same-disease pairs) − (rate among
different-disease pairs).

**Null.** Permute the disease label across the 272 papers. Each paper keeps every
one of its taxon–direction calls intact, so within-taxon structure is fully
preserved and only the taxon × disease association is broken. This is a
cluster-level permutation — the 23,627 pairs come from only 272 papers, and
pair-level shuffling has produced three false positives in this project on
record.

**Observations** are the 3,071 per-paper calls in each edge's `ev`. 17 of 2,011
edges have `n_obs > len(ev)` because one paper reported both directions for one
taxon (the known self-contradiction set); those contribute one call rather than
two, which is conservative throughout.

### Result

| | rate | n pairs |
|---|---|---|
| same disease | 0.7161 | 2,952 |
| different disease | 0.6570 | 20,675 |
| **gap** | **+0.0591** | z=3.39, **p=0.0014**, MDE +0.0296 |

A secondary check confirms the premise: taxon identity does predict direction at
all (directional purity 0.774 vs null 0.642, z=21.2, p=0.0002 over 2,493 calls,
under a within-paper shuffle of direction).

---

## It is not a confound

Disease is entangled with country and method here, because the corpus was
assembled disease by disease. This project has twice been burned by exactly that
shape (198 "explanatory" terms a random split reproduced at p=0.41; a
`diet_controlled` effect that went from p=0.002 to FDR 0.243). So:

| label | own gap | same / diff | p |
|---|---|---|---|
| disease | **+0.0591** | 0.716 / 0.657 | **0.0018** |
| country | −0.0112 | 0.642 / 0.653 | 0.639 |
| sequencing platform | −0.0278 | 0.641 / 0.669 | 0.857 |

(The disease row is the same test as the headline, re-run inside this script;
p=0.0018 here vs 0.0014 above is Monte-Carlo noise on 5,000 permutations, not a
different result.)

**Neither covariate produces any agreement at all** — both gaps are slightly
negative. The disease gap also holds *inside* strata: +0.0782 among same-country
pairs, +0.0862 among different-country pairs. Permuting disease only within
country blocks — the honest "disease, holding country fixed" test — gives
+0.0846 against a null mean of +0.0355, z=2.32, **p=0.0128**.

Both surviving p-values clear Benjamini–Hochberg across the four inferential
tests (0.0018 → 0.0072; 0.0128 → 0.0256). Metadata covers 205/272 papers, so the
stratified tests run on a subset and their MDE is correspondingly wider
(+0.0709).

---

## …but it is one disease

Leave-one-disease-out, then per-disease internal agreement:

| disease | internal agreement | pairs | lift over cross-disease 0.657 |
|---|---|---|---|
| **Parkinson's disease** | **0.807** | 1,220 | **+0.150** |
| Stroke | 0.725 | 455 | +0.068 |
| Multiple sclerosis | 0.689 | 315 | +0.032 |
| Alzheimer's disease | 0.608 | 806 | **−0.049** |
| Epilepsy | 0.492 | 67 | −0.165 |

Parkinson's supplies 42.6% of all same-disease pairs and is the only disease
whose papers agree with each other more than chance-plus-generic-dysbiosis would
predict. **Zero of the other four match or beat it.**

Dropping Parkinson's entirely:

| | gap | same / diff | n pairs | p | MDE |
|---|---|---|---|---|---|
| full corpus | +0.0591 | 0.716 / 0.657 | 2,952 / 20,675 | 0.0014 | +0.0296 |
| **Parkinson's removed** | **+0.0187** | 0.652 / 0.634 | 1,732 / 12,521 | **0.179** | +0.0350 |

**Power statement.** Outside Parkinson's, the observed gap is half the minimum
detectable one. Disease-specificity effects **larger than +0.035 are excluded**
at 205 papers; smaller ones are not. This is "no effect visible", not "no
effect".

That Parkinson's is the standout is the published consensus in the field — the PD
gut signature is the most replicated result in microbiome neurology. Recovering
it from an independent extraction pipeline is **face validity for the graph**,
not a coincidence to explain away.

Alzheimer's sitting *below* the cross-disease baseline on 806 pairs is the
mirror-image observation and deserves its own look: it is consistent with real
cohort heterogeneity in AD microbiome studies, and it is not a power artifact.

---

## What this means for the graph

The graph's 2,011 edges are **not** 2,011 independent disease-specific facts. A
decomposition of the agreement signal above chance (51.4%):

- **+14.3 points** is disease-independent — the same taxa going the same way in
  whatever disease is being studied.
- **+5.9 points** is disease-specific, and essentially all of it is Parkinson's.

So roughly **70% of the directional agreement in this graph is a corpus-wide
generic dysbiosis prior.** (This is a decomposition of agreement rates, a
heuristic, not a variance decomposition.)

Concretely, **59 of 187 taxa reported in ≥3 diseases never flip direction**
(31.6%) — *Streptococcus* enriched in all 12 diseases that report it,
*Butyricicoccus* depleted in all 8, *Enterococcus* enriched in all 8,
*Lactobacillaceae* enriched in all 7. For these, "enriched in disease X" is
close to contentless: they are enriched in everything.

The taxa that *do* discriminate are the ones that flip: *Lachnoclostridium*
(4↑/4↓ across 8 diseases), *Dorea* (3↑/3↓ over 6), *Prevotella* (6↑/8↓ over 14
diseases, 50 papers), *Bacteroides* (6↑/8↓ over 14, 54 papers). Note that
*Prevotella* and *Bacteroides* are simultaneously the graph's highest-evidence
taxa and its least directionally consistent — high weight is not high
information.

**Actionable:** the viewer should distinguish generic from discriminating edges.
A biologist reading "*Streptococcus* enriched in Parkinson's, 5 papers" cannot
currently tell that *Streptococcus* is enriched in eleven other diseases too.
`disease_specificity.json` carries the per-taxon table needed to render this.

---

## Also closed this session: the offline-taxdump route is a dead end

The top open defect — 54 named species folded into their genera (`Prevotella
copri` into `Prevotella`) — needs the NCBI taxdump, and this environment's
network policy denies `ftp.ncbi.nih.gov` and `ftp.ncbi.nlm.nih.gov` (CONNECT →
403). Only `pypi.org` / `files.pythonhosted.org` are reachable; EBI, Ensembl,
GBIF, UniProt and LPSN are all denied too.

The one offline candidate on PyPI was **`taxoniq`**, which bundles an 89 MB NCBI
taxonomy database. It was extracted and tested: the full tree (2,609,295 taxa
with parent and rank) and all scientific names ARE recoverable from its marisa
tries, **but synonyms are not — `taxoniq/build.py` indexes only `scientific
name`, `common name`, `genbank common name` and `blast name`, deliberately
excluding `synonym` and `equivalent name`.** Verified empirically:
`Bacteroidota` → 976 resolves, `Bacteroidetes` → **not found**; `Bacillota` →
1239 resolves, `Firmicutes` → **not found**.

Synonym folding is load-bearing here — Bacteroidetes+Bacteroidota → 976 and
Firmicutes+Bacillota → 1239 are exactly what stops evidence from splitting across
duplicate nodes, and the Disbiome/Peryton join depends on both sides passing
through `taxonomy.py`. A synonym-less name table would silently break that.

**So the species split still requires a machine with the real taxdump, and no
PyPI package substitutes.** Recorded so no future session re-runs this probe.

---

## Highest-value next step

**Split the 54 named species (unchanged, and still needs the taxdump — run it on
the Mac).** `child_folds.json` has all 115 child-folds classified: 54
`named_child` to fix, 32 `placeholder_child` already fixed, 29
`unspecified_member` correct as-is. Several need fuzzy resolution
(`Faecalibacterium prauznitzii`, `Bacteroides uniforms`) or they become
unresolved singletons.

**Second: mark generic vs discriminating edges in the viewer.** This is the first
result in five sessions that changes what the graph *means* rather than
correcting it, it needs no new data, and it is the difference between a graph a
biologist can use and one that overstates its own specificity.

Do **not** expect either to move the Disbiome/Peryton agreement rate — that is
now six structural findings in a row that the validation cannot resolve
(minimum detectable change ~0.013), and it is a property of the decisive set,
which is dominated by exactly the well-evidenced generic taxa identified above.
Indeed, this session explains *why*: if 70% of directional agreement is a
corpus-wide prior shared with the curated databases, agreement is measuring the
prior, not the graph's disease-specific content.

---

## Addendum — acted on, same session

**The finding is now in the graph.** `build_kg.py` annotates each taxon node with
a `specificity` block and each edge with `taxon_breadth`, `taxon_purity`,
`taxon_class` and `restates_prior`. Over 2,011 edges: 291 generic, 155
discriminating, 664 mixed, 901 narrow (fewer than 3 diseases, so nothing can be
said), and **252 edges restate their taxon's corpus-wide tendency outright**.

One definitional caveat for anyone comparing numbers across the two artifacts:
`build_kg.py` lets a **contested** edge cast no directional vote, while the
exploratory `disease_specificity.py` voted by majority. The build's rule is the
stricter one — a disease whose own papers disagree has no settled direction to
contribute — and it shifts a few counts (*Streptococcus* is generic across 11
diseases under the build, 12 under the analysis). Neither is wrong; the build's
is the one the graph ships.

**A determinism bug surfaced on the way.** Using this project's own verification
rule — rebuild twice and diff — revealed that two builds of identical input had
never been byte-identical: `sites` was built by iterating a **set** of paper
titles, so its JSON key order varied per process with `PYTHONHASHSEED` on ~30
multi-site edges. Content was always identical. But it meant the very
countermeasure adopted after two fixes silently erased themselves on rebuild was
emitting a false positive on every build. Fixed; three rebuilds are now
byte-identical. No number moved.
