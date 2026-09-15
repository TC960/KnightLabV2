# Where disease node labels come from — and one suspected error that wasn't

**2026-09-15. Cloud, CPU-only. No `MAIN_DATA.json`, no NCBI taxdump (`ftp.ncbi.nih.gov`
still 403 — tenth session, stop probing).**

Nothing in the graph was changed by this session. One suspected shipped error was
investigated and **cleared**; one structural property of the disease dimension was
measured for the first time; one shipped claim had a missing multiple-comparison
correction supplied.

---

## 1. The suspected error: `Neurocognitive impairment` — NOT an error

`Neurocognitive impairment` is a disease node with **1 paper and 7 edges**, sitting
in the cognitive-decline cluster beside Alzheimer's, MCI and Dementia. Its single
paper is:

> *Gut Microbiota and Fecal Metabolites Associated With Neurocognitive Impairment in
> **HIV-Infected Population*** (doi:10.3389/fcimb.2021.723840)

That looked like the disease-dimension twin of the taxid rule in `CLAUDE.md`: a node
whose label reads as generic cognitive decline while its cohort is defined by an
entirely different primary condition. The 2026-09-14 audit could not have caught it —
it compared the extractor's `predicted_disease` against the datasheet's `disease`
(label vs label) and never against the cohort the paper actually studied. It does
not mention HIV anywhere.

**It is not an error.** The paper's own relation-bearing sentences settle it:

> "The species abundance of Spirochaetes and Epsilonbacteraeota was higher in the
> **NCI group than in the non-NCI group** at the phylum level."

> "Among the genera with observed differences **between the NCI and non-NCI groups**,
> *Faecalibacterium*, Corprococcus_2, Ruminococcaceae_NK4A214_group, and
> *Ruminococcus_1* contain butyrate-producing bacteria."

> "Our results show that BPB and *Klebsiella* and the associated metabolites are
> associated with NCI **in people with HIV**."

HIV is the **background population, held constant across both arms**. The contrast is
NCI⁺ vs NCI⁻ *within* HIV-infected people, so the edges are correctly typed as
neurocognitive impairment. The extractor was right and the datasheet (which said
"Cognitive Impairment") was coarser. **Hypothesis rejected by reading the paper, not
by reasoning from its title** — the title alone would have supported the wrong call.

What survives is narrower and is an *interpretation* caveat, not a typing error: every
observation on this node comes from an HIV⁺ population in which microbial composition
and immune status are entangled (the paper reports *Treponema_2* inversely correlated
with CD4 count). Whether "*Klebsiella* enriched in NCI" transfers to non-HIV cognitive
impairment is untested here.

### Its MONDO id is `None`, and that is a recorded refusal

MONDO 2026-09 (63,278 term stanzas, 36,017 with names) contains **no term whose name
contains "neurocognitive"**. The nearest candidate reachable by synonym is
`MONDO:0020689` **"AIDS dementia complex"** (synonyms "HIV-associated dementia",
"HIV Encephalopathy"), which `is_a MONDO:0001627 dementia`.

Assigning it would repeat the **ASD/autism mistake fixed on 2026-09-14** — an id one
rank too narrow. AIDS dementia complex is the dementia-stage endpoint; this cohort is
neurocognitive impairment across its range, most of it not dementia. `None` is
correct. **Do not "fix" it to `MONDO:0020689`.**

---

## 2. What the investigation actually turned up: 20% of the graph's edges hang off
   free text

`build_kg.norm_disease` tries 17 `DISEASE_MAP` regexes and, on a miss, falls through
to `label = s[:1].upper() + s[1:]` — the extractor's `predicted_disease` string,
title-cased, becomes a disease node verbatim. The fallback is deliberate (the comment
says unmapped labels keep their label "rather than being silently dropped") and is
defensible. It had never been measured.

Measured (`disease_node_provenance.py`, self-test 11/11, `DISEASE_MAP` read **live**
out of `build_kg.py` rather than copied):

| origin | nodes | edges | edge share | papers |
|---|---|---|---|---|
| `DISEASE_MAP` | 15 | 1,609 | 80.1% | 233 |
| **fallback (free text)** | **25** | **399** | **19.9%** | **38** |

Of the 25 fallback nodes, **11 (145 edges) carry no MONDO id either** — neither a
regex nor an ontology has ever seen those strings. They are: `Cognitive impairment`,
`Hepatitis B virus-associated liver cirrhosis`, `Chronic traumatic complete spinal
cord injury`, `Hemorrhagic transformation`, `Hypertensive intracerebral hemorrhage`,
`Minimal hepatic encephalopathy`, `Poststroke aphasia`, `Traumatic thoracic spinal
cord injury`, `Neurocognitive impairment`, `Neuroinfection`, `REM sleep behavior
disorder–Lewy body disease continuum`.

**This is not a census artifact — it is a census.** All 2,008 edges were enumerated,
so no permutation test applies and none is reported.

### The asymmetry this exposes

The **taxon** half of the graph has two curation instruments — `taxon_typos.py` (the
papers' own misspellings; 33 labels folded, 13 refusals recorded) and `multi_taxon.py`
(punctuation fragmenting one concept over four nodes). The **disease** half has
**none**. `Cognitive impairment` exists as a node distinct from `Mild cognitive
impairment` not because anyone judged them distinct, but because the extractor emitted
a string that matched no regex. All four of its papers were labelled **Dementia,
Alzheimer's or Other** by the human datasheet.

### Candidate label families — for human review, nothing folded

`disease_node_provenance.py` flags nodes sharing a head noun. **4 of 5 families are
genuine; the 5th is the heuristic's own false positive** and is left in the output on
purpose:

| family | edges | nodes |
|---|---|---|
| `[cognitive impairment]` | 209 | Mild cognitive impairment (13p) · Cognitive impairment (4p) |
| `[cord injury]` | 101 | Spinal cord injury (6p) · Chronic traumatic complete SCI (1p) · Traumatic thoracic SCI (1p) |
| `[intracerebral hemorrhage]` | 61 | Intracerebral hemorrhage (3p) · Hypertensive ICH (1p) |
| `[hepatic encephalopathy]` | 12 | Minimal hepatic encephalopathy (1p) · Hepatic encephalopathy (1p) |
| ~~`[s disease]`~~ | ~~689~~ | **FALSE POSITIVE** — Parkinson's / Alzheimer's / Huntington's share a suffix, nothing else |

That false positive is the argument for keeping this a *candidate generator* rather
than a folding rule, and it is the same lesson the taxon side already paid for: edit
distance would have merged `Oscillospirales` into `Oscillospira`, which is why
`taxon_typos.py` ships a curated table instead of a similarity threshold. **A rule that
folds on string similarity would have merged three unrelated neurodegenerative
diseases carrying 689 edges.**

**Folding any of these remains a human call** (`CLAUDE.md`: "What remains is a
decision, not an analysis"). What this session adds is that it is *also* a data-quality
question, and that the decision is worth 383 edges across 4 real families.

---

## 3. Correction supplied: no cognitive-cluster pair survives multiple-comparison
   correction

The 2026-09-14 finding — the 71-paper cognitive cluster does not cohere, 0.592 against
a 0.672 background — **stands, and is robust**. Removing `Neurocognitive impairment`
(the HIV node, 4 pairs / 5 observations) moves it to **57/98 = 0.582**, i.e. *further*
from the background, not toward it.

But that session's per-pair language ("the structure is sharp inside it";
`CLAUDE.md`'s "every MCI pair at or below a coin flip while the MONDO-confirmed
Alzheimer's/Dementia link runs 0.938") presents pair-level contrasts as established.
Tested individually against the 0.6687 background with an exact two-sided binomial and
Benjamini–Hochberg across all 16 pairs at q=0.05:

| pair | k/n | rate | p | BH crit | survives |
|---|---|---|---|---|---|
| MCI / Dementia | 4/13 | 0.308 | 0.014 | 0.0031 | no |
| Alzheimer's / Dementia | 15/16 | 0.938 | 0.030 | 0.0063 | no |
| Alzheimer's / MCI | 13/26 | 0.500 | 0.093 | 0.0094 | no |
| MCI / Cognitive impairment | 9/18 | 0.500 | 0.138 | 0.0156 | no |
| *(12 further pairs)* | | | ≥0.257 | | no |

**0 of 16 survive.** The cluster-level conclusion is unaffected — it was a single
pooled test and remains one — but **no individual pair contrast should be quoted as
established**, including the 0.938 AD/Dementia link.

---

## 4. Power, stated plainly

The folding questions this was meant to inform are **underpowered, not answered**:

| question | observations | rate vs 0.669 | exact p | MDE (downward) |
|---|---|---|---|---|
| Is `Cognitive impairment` the same node as `MCI`? | 18 | 0.500 | 0.138 | **28.0 points** |
| Is `Neurocognitive impairment` one of them? | 5 | 0.800 | 1.000 | **46.9 points** |

The observed MCI/CI gap is 16.9 points against an MDE of 28.0 — **this corpus cannot
resolve it**. The NCI question has essentially no power at all (MDE 46.9 points on a
66.9% base) and was decided on cohort grounds — the population is HIV⁺ — not on data.

So the honest statement for item 2(b) of `NEXT_SESSION_PROMPT.md`:

- **`Neurocognitive impairment` should not be folded into MCI or `Cognitive
  impairment`** — but because its source population is HIV⁺ and its
  generalisability is untested, **not** because the graph's data says so. The data
  says nothing at n=5.
- **`MCI` vs `Cognitive impairment` is unresolved and unresolvable here.** It needs
  either a clinical call or more papers.
- Both are downstream of the real issue, which is that **`Cognitive impairment` is a
  free-text node nobody created on purpose.**

---

## 5. Standing assumption corrected: paper text IS available in the cloud

Every session since 2026-09-11 has recorded that cohort- and sentence-level questions
are blocked in the cloud because `MAIN_DATA.json` is gitignored. That is true of
**full** text and false of the text most of these questions need:

**`relation_sentences_clean.json` is committed and covers 271 of 271 contributing
papers (100%) — 6,294 relation-bearing sentences, 1.74 MB**, each tagged with its
resolved taxa and direction cues.

Section 1 of this document was settled entirely from it, in the cloud, with no
`MAIN_DATA.json`. Any future question of the form *"what did this paper compare?"*,
*"which direction does it report for taxon X?"*, or *"what is the cohort?"* is
answerable here.

**The limit is real and must not be overstated.** It holds ~10% of the corpus text
(16.78 M chars → 1.74 M kept). It therefore **cannot** answer *"does taxon X appear
**anywhere** in this paper?"*, which is exactly what `silent_edge_mentions.py` needs.
**Item 0 of `NEXT_SESSION_PROMPT.md` genuinely does still require the Mac.**

---

## 6. Dead end, tested and rejected: the datasheet label is NOT a hierarchy source

The obvious cheap follow-up is tempting enough that it is worth closing off
explicitly. For 3 of the 4 candidate families, **the human datasheet already assigns
every member paper to one label**:

| family | datasheet label on every member paper |
|---|---|
| `[cord injury]` (3 nodes, 8 papers) | `Spinal cord injury (SCI)` |
| `[intracerebral hemorrhage]` (2 nodes, 4 papers) | `Stroke` |
| `[hepatic encephalopathy]` (2 nodes, 2 papers) | `Encephalopathy` |

So the datasheet looks like a free, human-authored source of parent links for the
fallback nodes — the disease-side analog of taxonomic containment. **It is not, and
the check that kills it is already on record.**

Scoring "every backing paper's datasheet label maps to one `DISEASE_MAP` node" as an
is-a claim yields 9 nodes / 219 edges. Inspecting them:

| proposed link | verdict |
|---|---|
| Spinal cord injury → **Amyotrophic lateral sclerosis** | absurd — one member is a *comparative* ALS-vs-SCI study carrying both labels, and "Spinal cord injury" matches no `DISEASE_MAP` regex, so only ALS survived |
| Essential tremor → Parkinson's | **contradicts MONDO** — `FINDINGS_disease_ontology.md`: *rejection upheld (sibling)* |
| Multiple system atrophy → Parkinson's | **contradicts MONDO** — *rejection upheld (cousin)* |
| CADASIL → Stroke | wrong direction of reasoning — CADASIL *causes* strokes |
| Tuberous sclerosis complex → Epilepsy | backwards — TSC causes epilepsy |
| Chronic traumatic complete SCI → *(none)* | inconsistent with plain SCI above, on identical datasheet labels |

**On the only two cases where an external authority exists, the heuristic goes 0 for
2** — and it proposes precisely the two folds the 2026-09-14 MONDO pass independently
*rejected*. The reason is simple once seen: the datasheet's `disease` column is a
**topic/cohort tag** ("what disease area is this paper about"), not a taxonomy.
Comparative studies carry two tags, and a tag naming the condition a cohort *has* says
nothing about whether the node is a subtype of it.

**Do not build disease hierarchy from the datasheet label.** MONDO remains the only
authority here, and it resolves 28 of 40 labels with 12 documented refusals.

This also sharpens §2: the fallback nodes are not merely unvalidated, they are
**unvalidatable from the material already in the repo**. Closing the gap needs either
MONDO coverage these labels do not have, or a human.

## Files

- `disease_node_provenance.py` — the census. Reads `DISEASE_MAP` live from
  `build_kg.py`; self-test 11/11 (6 positive, 5 fallback-arm negative controls);
  report-only, folds nothing.
- `disease_node_provenance.json` — per-node origin, edge mass, MONDO id, candidate
  families.

## What did not change

`graph.json`, `rag_corpus.jsonl`, `kg.html` and `docs/` are **untouched**. No
correction was made, so none can be miscited as an accuracy gain — and per the
standing rule, six structural corrections have now each moved agreement by less than
this corpus can resolve (~0.013).
