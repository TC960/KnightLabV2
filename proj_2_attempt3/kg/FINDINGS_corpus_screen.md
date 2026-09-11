# Is the corpus actually human case-control? A leak, and a powered null

*2026-09-11. Found by following a quote, not by auditing a function.*

---

## How this started

While verifying an unrelated taxon merge, the supporting sentence for
`Ruminiclostridium-5` read:

> "…AD Tg mice (statistical significance) and ADWT mice (trending) also
> exhibited increased abundance of **Ruminiclostridium-5**…"

Transgenic mice. In a graph whose edges are supposed to mean "taxon X differs
between human patients and human controls".

## Part 1 — one screened-out rat study was in the graph

`maindata_screen.json` had already adjudicated 45 papers and marked 22 as not
human case-control (15 animal, 3 no control arm, 2 case reports, 2 reviews).
`filter_maindata.py` drops those before the build. One got through anyway.

**The mechanism.** `filter_maindata.norm()` folded curly quotes and dashes but
kept a trailing full stop. `build_kg.dedup_rows` strips *every* non-alphanumeric.
**13 papers sit in `extractions_corrected.json` under two spellings that differ
only by such punctuation** — the same paper scraped twice under two links.

For a paper that FAILS the screen, that gap is a leak:

1. the screen matches one spelling and drops it;
2. the other spelling hashes to a different key here, so it survives;
3. the deduper that would have folded the two copies never sees them together,
   because the filter has already removed one.

The paper that got through:

> **"Microbiota from Alzheimer's patients induce deficits in cognition and
> hippocampal neurogenesis"** — `DROP_ANIMAL`. It transplants human faeces into
> microbiota-depleted rats and reports **the rats'** microbiome.

It contributed **6 Alzheimer's edges** as if they were human findings.

**The assertion that should have caught it made it invisible instead.**
`assert seen == 45` counts *screen entries matched*, not *paper copies dropped* —
so a paper present under two spellings satisfies it while one copy leaks. It is
now an assertion on the set of screen entries matched, plus a post-condition that
no surviving row carries a dropped key. That post-condition fails loudly on the
old normaliser.

**Effect:** 326 → 325 rows in, 272 → 271 contributing papers. Exactly 6 edges
lose one paper each (*Bacteroidota*, *Coprococcus*, *Desulfovibrio*, *Bacillota*,
*Verrucomicrobiota*, *Clostridium sensu stricto 1*, all Alzheimer's). No edge
added or removed. The last drops from 1-up/2-down to 1-up/1-down and is now
correctly **contested** rather than depleted.

Agreement unmoved, as always: Disbiome 73.4% (unchanged), Peryton 72.3% → 72.5%.
A correctness fix — rat microbiome is not human microbiome — not an accuracy gain.

## Part 2 — the gap behind the leak

The leak was one paper. The finding behind it is larger:

> **Only 23 of the 271 contributing papers have EVER been screened for study
> design.**

`maindata_screen.json` was built for the 45 title-matched MAIN_DATA additions,
because those entered by keyword-matching titles and were known to be unvetted.
The ~250 datasheet papers were never put through it. `filter_maindata.py`'s own
docstring says the 45 were unscreened "unlike the 303 datasheet papers" — that
contrast was assumed, not established.

### The screen, and its validation

A deterministic prefilter over all 271 full texts counts animal-model markers
(`mice|mouse|murine|rats?|C57BL|Sprague-Dawley|germ-free|gnotobiotic|3xTg|
APP/PS1|5xFAD|…`) against human-cohort markers, and flags a paper if an animal
term appears in the title, or animal mentions outnumber human ones, or animal
mentions are heavy relative to human ones.

**Recall was validated before the filter was used for anything**, against the
only gold set available — the 45 papers `maindata_screen.json` already
adjudicated:

| | |
|---|---|
| DROP_ANIMAL papers caught | **15 / 15 (recall 1.000)** |
| precision | 15 / 17 (0.882) |
| the 2 false positives | one `KEEP`, one `DROP_REVIEW` (arguably a true positive) |

**Power:** zero misses in 15 bounds the miss rate at **≤20% with 95%
confidence** (rule of three). That is the honest limit — the gold set is small.

A first version of the filter also flagged review language (`meta-analysis`,
`systematic review`). It was **discarded before use**: it fired on 87 papers,
almost all of them good human case-control studies that merely cite a
meta-analysis in their discussion. A filter with that precision would have
buried the signal.

### The result: null

The filter flags **13 of 271** contributing papers. All 13 were read against
full text. **All 13 report a human disease-case-vs-healthy-control microbiome
comparison.** Most also run mouse experiments, which does not disqualify them —
the human contrast is what the extractor read.

| paper (abbrev) | human cohort, verbatim |
|---|---|
| Phthalates / DLB | "A total of 43 DLB patients and 45 normal subjects were included in this study" |
| MS / human T cells | "analyzed the microbiomes of 71 MS patients not undergoing treatment and 71 healthy controls" |
| Blautia coccoides / ICH | "fecal samples from 35 healthy individuals and 36 patients with ICH" |
| SCI dysbiosis | "We enrolled 59 SCI patients and 27 healthy control subjects" |
| *F. prausnitzii* / MCI | "the healthy group (n = 21 …) and the MCI group (n = 15 …)" |
| Transmission of AD dysbiosis | "157 participants were initially screened … 108 participants were included" |
| Akkermansia / AD | "fecal and serum samples obtained from 19 AD patients and 18 healthy volunteers" |
| Hemorrhagic transformation | "32 patients with first-ever acute ischemic stroke … as well as 16 h[ealthy]…" |
| Familial dysautonomia | "a cohort of FD patients … and their cohab[itating relatives]" |
| MS microbial ratio | "45 patients with RRMS were recruited from the Neuroimmunology Clinic" |

**So no additional animal-only study is in the graph.** The single leak in
Part 1 was the only one.

### A note on how this was checked, because it nearly went wrong

The 13 papers were read by Haiku subagents. Their verdicts were then checked by
requiring every supporting quote to appear **verbatim** in the source file —
9 of 13 failed that check, which initially looked like a reliability problem.
It was not: the agents had *paraphrased* cohort descriptions rather than
fabricated them, and spot-reading the failures (the phthalates paper claims
"43 DLB patients vs 45 controls"; the text says "A total of 43 DLB patients and
45 normal subjects were included") confirmed every verdict.

Two lessons, in opposite directions. The verbatim check was worth running —
it is cheap and would have caught a fabrication. But **"quote not found" is not
evidence of a wrong verdict**, and a stricter reading of that signal would have
thrown away 9 correct adjudications. The deterministic cohort regex written to
replace the agents was *also* wrong, and more quietly: it missed
"43 **DLB** patients" because it required the number to sit adjacent to the
noun. Both instruments were weaker than the thing they were auditing.

## Part 3 — screening the other 249, and why the result is "do not act"

The gap above was then actually attacked: all 249 never-screened papers were put
through an abstract-level screen (`screen_corpus.py`, Haiku subagents, 9 batches
of 34) against the four failure modes, with **24 blinded controls** drawn from
the already-adjudicated 45 and shuffled in under the same id scheme.

### The headline, and the number that matters more

| | |
|---|---|
| exact category agreement on controls | 19/24 = **0.792** |
| keep-vs-drop agreement | 20/24 = **0.833** |
| false drops (gold KEEP → predicted drop) | 2/12 = **0.167** |
| missed drops (gold drop → predicted KEEP) | 2/12 = **0.167** |

The control set is balanced by construction (12 KEEP, 12 drop) and includes the
rat-FMT paper from Part 1. **The screen marked that paper KEEP** — the one case
this whole line of work exists to catch, missed from its abstract, on an axis
where the deterministic full-text filter scored 15/15.

### The result on the 249

**231 KEEP · 14 UNCLEAR · 4 proposed drops · 0 animal drops.**

Zero animal drops is worth stating plainly: an LLM reading abstracts and a
regex reading full texts independently agree there is no animal-only study left
in the corpus. Two methods, one conclusion.

### Every proposed drop was then checked against full text, and none survived

All four fell in `DROP_NO_HEALTHY_CONTROL` — **the single category both false
positives landed in.** That is not a coincidence to shrug at, so all four were
read:

| paper | verdict | evidence |
|---|---|---|
| Probiotics/rifaximin/lactulose in MHE | **false positive** | "stool samples from healthy controls and MHE patients" |
| Gut microbiota in dementia with Lewy bodies | **false positive** | "224 PD patients, 26 iRBD patients, 28 DLB patients, and **147 controls**" |
| MCI in patients without dementia | **arguable** | 82 patients, "61 were classified as MCI, and 21 were classified as NC" — a cognitively-normal comparison group exists |
| Neurocognitive impairment in HIV | **arguable** | "122 controls (**HIV-infected patients without NCI**)" — not healthy, but a matched disease-vs-no-disease contrast, which is a valid control for NCI |

**Zero of four are unambiguous. The graph was not modified.**

### So the method does not work well enough, and that is the finding

At a false-discovery rate of 50–100% on the drop list, an abstract-level screen
cannot prune this corpus. It can only *nominate*, and every nomination costs a
full-text read anyway — which is most of the work the screen was meant to save.

Two honest qualifications in opposite directions:

- **The controls are harder than the population.** They are drawn from the 45
  unvetted keyword-matched MAIN_DATA papers, which are far messier than the
  datasheet papers being screened. A 0.167 false-positive rate applied to 245
  KEEP papers predicts ~41 false drops; only 4 drops were proposed in total. So
  0.833 is probably a **lower bound** on this population — but by how much
  cannot be measured, because no representative gold set exists.
- **Some of the failure was mine.** The abstract extractor anchored on the first
  "Abstract" marker and returned pure front matter — journal navigation, author
  lists, affiliations — for **12.9% of the corpus (35 of 271 papers)**. Those
  papers came back UNCLEAR because the screen never saw an abstract. One agent
  diagnosed it unprompted: *"unclear abstracts due to heavy metadata in the
  source text."* Replaced with a sliding best-scoring window, which cuts
  zero-cue spans from 35 papers to 5 (1.8%). **The run reported here used the
  old extractor**, so its 14 UNCLEAR are inflated and a re-run would do better.

### What to do instead

1. **Do not re-run this screen expecting a cleaner answer.** The extractor fix
   will reduce UNCLEAR, but it does not touch the false-positive rate, which is
   a judgement problem and not an input problem — both false positives had
   perfectly good abstracts.
2. **The animal axis is solved** and needs nothing further: deterministic,
   full-text, recall 15/15, and corroborated independently here.
3. **The remaining three axes need full text, not abstracts.** That is what
   produced the trustworthy `maindata_screen.json`. It is the expensive option
   and it is the one that works.
4. **The 14 UNCLEAR papers and the 4 nominations are a ready-made worklist** —
   18 papers, full text, for a human or a stronger model. That is a tractable
   afternoon, and it is recorded in `corpus_screen.json`.

## What is still open, and it is not small

This screen covers **one** failure mode. In the 45-paper gold set, animal
studies were **15 of 22** drops (68%). The rest were:

- **3 no healthy-control arm**
- **2 case reports (n ≤ 2)**
- **2 reviews / no primary cohort**

None of those has a deterministic detector with usable recall, and **248 papers
remain unscreened for them.** If the gold set's rate carried over — 7 non-animal
drops in 45 papers, 15.6% — that would be on the order of **35 more papers** that
do not belong. That number is an extrapolation from 45 papers, not a measurement,
and it should be treated as a reason to screen rather than as a finding.

**This was the cheapest unblocked lever in the project, and Part 3 spent it.**
The answer is that 248 abstracts are not enough — the screen nominates, it does
not decide. What remains is the same work at full-text resolution, which is
what produced the one screen in this repo that can be trusted.

The prior is still against a large effect on agreement: the MAIN_DATA screen
removed 22 papers and moved it by nothing (McNemar p = 1.00, and below
resolution on the signed concordance metric). Removing a handful more will not
move it either. Justify the work on correctness, not on the number.
