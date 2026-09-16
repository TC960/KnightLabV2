# Is each paper filed under the disease it actually studies?

**2026-09-13. Cloud session, CPU only. Deterministic; no model judgement used.**

**Headline.** **Zero** of 325 extraction records assign a paper a disease that
never appears in its own text. Of 15 genuine family-level conflicts with the
human datasheet, **7 are the extractor being right and the datasheet coarse, 5
are comparative studies naming both diseases, 2 are ambiguous, and 1 is
suspect.** No confirmed disease misassignment. The audit's more useful output is
a **size for a modelling decision the repo has flagged as "needs a human" for
three sessions**: 6 separate cognitive-decline disease nodes carry 71 papers, and
the graph contains **0 hierarchy links between any two disease nodes**.

## Why this had never been checked

`build_kg.py:383` reads

```python
dis_raw = (r.get("predicted_disease") or r.get("disease") or "")
```

so **the disease half of every edge in the graph is an LLM output**, with the
human datasheet label used only when the model returned nothing. Every fidelity
instrument in this repo — the gold benchmark, Disbiome/Peryton agreement, the
direction audit, the mention audit — scores the **taxon** half. A wrong disease
is invisible to all of them, and it does not corrupt one edge: it misfiles an
entire paper's edges onto the wrong node.

## Method

`verify_disease_assignment.py`. Both the datasheet label and the predicted label
are canonicalised onto a set of disease *families*, which discards the
differences that are not errors: curly apostrophes, `(PD)` suffixes, case, and
granularity (`Stroke` → `acute ischemic stroke` is the extractor being more
specific, not wrong — counted separately as `refinement`). Only records whose
family sets are **disjoint** are treated as conflicts.

Each conflict is then adjudicated against the paper itself, title first:

- **Title evidence is near-decisive** for a case-control paper and is checked
  before anything else. This matters because raw body-text frequency favours the
  **wrong** answer here: an MCI study discusses Alzheimer's throughout, since MCI
  is its prodrome. The first version of this script, scoring on body counts
  alone, called six correct MCI assignments "suspect".
- Body-text family frequency is the fallback when the title names neither.
- A paper with no full text in git is still adjudicated **from its title** when
  the title settles it — `"...cerebral autosomal dominant arteriopathy with
  subcortical..."` is the CADASIL study whatever the datasheet says.

## Results (325 records)

| outcome | n |
|---|---|
| agree (same family) | 222 |
| `refinement` — prediction narrower, same family | 33 |
| datasheet uninformative (`Other`/blank) | 33 |
| prediction outside this script's vocabulary | 21 |
| **family-level conflict** | **15** |
| no prediction (falls back to datasheet) | 1 |

Conflicts, adjudicated:

| verdict | n |
|---|---|
| prediction correct by title, datasheet coarse | 7 |
| comparative study — both diseases in the title | 5 |
| ambiguous (title names neither) | 2 |
| **datasheet in title, prediction suspect** | **1** |
| **prediction unsupported by the text** | **0** |

The 21 unmapped predictions are not errors; they are real diseases outside this
script's family list (*Lewy body disease*, *REM sleep behaviour disorder*, *Rett
syndrome*, *ADHD*, *traumatic brain injury*, *sporadic Creutzfeldt-Jakob*,
*hepatitis B-associated liver cirrhosis*). The last confirms the graph is
broad-scoped, as `CLAUDE.md` states, rather than neuro-only.

### One datasheet error, found by the audit

*"Faecal mucoprotein MUC2 is decreased in multiple sclerosis and is associated
with mucin degrading bacteria"* is labelled **ALS + Parkinson's** in the human
datasheet. `amyotrophic lateral sclerosis` occurs **0 times** in the paper;
`multiple sclerosis` occurs 21 times plus 98 abbreviations, and is in the title.
**The extractor is right and the human sheet is wrong.** This is a fourth
independent sign that the in-house annotations are unreliable, consistent with
the audit already recorded in `CLAUDE.md`.

### The one suspect assignment

*"Effects of a ketogenic and low-fat diet on the human metabolome, microbiome,
and foodome in adults at risk for Alzheimer's disease"* — datasheet says
Alzheimer's (and the title agrees); the extractor filed it under **mild cognitive
impairment**. The cohort is at-risk/MCI subjects, so both readings are
defensible. Flagged rather than changed: it is one paper, and it is exactly the
boundary case the next section is about.

## The real finding: a modelling gap, now sized

**11 of the 15 conflicts sit on the MCI / Alzheimer's / Dementia boundary.** The
pattern is systematic and is not the extractor erring — it consistently prefers
the specific cohort label where the datasheet uses the coarse family. But the
graph has nowhere to put that relationship:

| disease node | papers |
|---|---|
| Alzheimer's disease | 46 |
| Mild cognitive impairment | 13 |
| Dementia | 6 |
| Cognitive impairment | 4 |
| Neurocognitive impairment | 1 |
| Subjective cognitive decline | 1 |

**Six nodes, 71 papers, one clinical continuum — and 0 hierarchy links between
any two disease nodes in the entire graph**, while taxa carry 708. A query for
Alzheimer's silently misses 13 MCI papers; a contested Alzheimer's edge cannot be
compared against its own prodrome.

This is the same open question already logged for `Intracerebral hemorrhage`
beside `Stroke` and for taxonomic containment — *should disease subtypes be
modelled as containment the way taxa are?* It was recorded as "a design decision,
needing a PI, not a script". It still is. What is new is the magnitude: **this is
not a tidy-up affecting a couple of edge cases, it is 71 papers on the single
largest disease cluster in the graph.** That should raise its priority.

## Limits

- Family canonicalisation is a hand-written vocabulary; 21 predictions fall
  outside it and are reported as unmapped rather than silently passed.
- "Agree" means *same family*, not that the assignment is correct — a paper both
  sources mislabel identically passes. This audit can only catch disagreement
  and text-absence, and text-absence returned zero.
- One paper is unscoreable beyond its title (full text only in the gitignored
  `MAIN_DATA.json`).
- **Nothing here changes the graph.** No rebuild was run; rebuilding on a cloud
  checkout is how this repo lost work on 2026-09-11.

## Artifacts

`verify_disease_assignment.py` → `disease_assignment.json`.
