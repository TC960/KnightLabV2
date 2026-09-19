# Was the comparison in scope? A contrast census of all 271 contributing papers

*2026-09-19. Reproduce with `contrast_packets.py` → `contrast_census.py` →
`contrast_robustness.py`. Nothing in `graph.json`, `rag_corpus.jsonl`, `kg.html`
or `docs/` was changed by any of it.*

## The gap this fills

Every fidelity instrument in this project scores whether an edge's **taxon and
direction** are right:

| instrument | what it scores | number |
|---|---|---|
| `FINDINGS_direction_audit.md` | direction vs the paper's own sentences | ≥86.6% |
| `FINDINGS_mention_audit.md` | does the taxon appear in the paper | 99.57% |
| `validate_external.py` | direction vs two curated databases | 73.0 / 72.5% |
| `FINDINGS_newgold.md` | taxa vs a hand-curated gold | F1 0.739 |
| `FINDINGS_zero_yield.md` / `FINDINGS_edge_recall.md` | what was missed | ≥96.1% / ≤98.1% |

**None of them asks whether the comparison the edge came from was admissible.**
The extraction prompt (`eval-v2/run_eval.py`, `samgated-v1`) admits only
**disease vs healthy control**. A taxon genuinely higher in ICH survivors than in
ICH deceased is a *correct reading of its paper* and is still not an edge this
graph should carry — and every instrument above would score it as a hit. This is
a **precision-of-scope** question and it had never been asked.

## Method

One packet per contributing paper: its title plus the sentences from
`relation_sentences_clean.json` most likely to name the study's arms. That file
covers **271 of 271** contributing papers, so this needed neither `MAIN_DATA.json`
nor a GPU.

**Adjudicators were blinded** to the disease node label, to whether that label came
from `DISEASE_MAP` or the free-text fallback, and to anything from the graph. That
is not decoration: the second test below compares exactly those two groups, and the
third correlates the verdict with a graph-derived outcome. The provenance tier
retracted on 2026-09-17 failed *because* its two buckets were defined by the
outcome; this one is built so they cannot be.

**Every quote is machine-checked** against the sentences actually supplied, in two
tiers. Byte-for-byte is the headline. A second tier collapses whitespace, because
these sentences carry PDF-extraction artifacts (`" , "`, `"[ 32 ]"`) that a reader
silently tidies — a re-spaced quote is still the paper's own words; an invented one
is not.

| quote check | papers |
|---|---:|
| byte-for-byte exact | 232 |
| exact after collapsing whitespace | 9 |
| partial (≥1 quote verifies) | 24 |
| **supported, used below** | **265 / 271** |
| all quotes paraphrased | 5 |
| no quotes | 1 |

**This revises the 2026-09-16 rule, it does not repeal it.** That session found 7
of 36 "verbatim" quotes were paraphrases and made machine-checking mandatory —
correctly. But the raw count overstates: here 16 failed byte-for-byte and only
**5** are genuine paraphrases; the other 11 are re-spacing. Always machine-check;
report both tiers.

## Result 1 — the census. The gate holds, and now there is a number for it.

Of the graph's **3,077 observations**:

| the paper's main reported contrast | papers | observations |
|---|---:|---:|
| **disease vs healthy control (in scope)** | **241** | **2,871 (93.3%)** |
| treatment / intervention arms | 7 | 40 |
| within-disease subgroup | 8 | 38 |
| animal | 1 | 9 |
| disease vs a different disease | 2 | 3 |
| longitudinal, no control arm | 1 | 3 |
| unclear from the supplied sentences | 5 | 21 |
| verdict failed the quote check | 6 | 92 |

All 19 out-of-gate verdicts were then **read a second time in-session** and tiered:

| tier | papers | obs |
|---|---:|---:|
| cleanly out of scope | 13 | 55 |
| out of scope **but also reports a healthy-control arm**, so some edges are legitimate | 5 | 29 |
| **overturned** — the verdict is wrong (see below) | 1 | 9 |

**So: out-of-gate observations are bounded at 55–84 of 3,077, i.e. 1.8%–2.7%.
Quote the range.** 93.3% of the graph is confirmed in scope, the remainder is
unreadable from these sentences rather than known bad.

## Result 2 — out-of-gate papers disagree with the literature ~1.75× as often

This is the payoff, and it is a *validated* quality flag — the thing the retracted
provenance tier was not.

Outcome and null are `paper_discordance_offset.py` unchanged: the closed-form
within-edge expectation that absorbs edge depth, contestedness and tie structure,
and a permutation that shuffles the predictor **across papers** holding each
paper's (observed, expected) fixed.

| group | papers | observed disagreements | expected | O/E |
|---|---:|---:|---:|---:|
| out of gate | 11 | 13 | 7.44 | **1.747** |
| in gate | 157 | 250 | 252.80 | 0.989 |

**diff +0.758, p = 0.00015** (N = 20,000 paper-level permutations), **MDE ±0.303**.
The effect is 2.5× the minimum this design can resolve.

`paper_discordance_offset.py`'s own `MIN_DECISIVE = 4` filter is dropped here on
purpose and the reason is stated: it discards 17 of the 19 out-of-gate papers —
they are small contributors, which is exactly why nobody noticed them. The O/E
offset already absorbs depth, so pooling every observation and permuting the
**paper** label is valid without it. With the filter kept, n_true = 2 and the
result is not quotable (p = 0.078 against an MDE of ±1.00); that version is in the
JSON and should not be cited.

### Four attacks, all survived (`contrast_robustness.py`)

1. **Leave-one-paper-out** over all 11: worst p = **0.00070**. No single paper
   carries it.
2. **The one overturned verdict is not load-bearing** — and not because dropping it
   helps: that paper has `e_dis = 0`, so it never entered the test at all.
3. **Keep only the 8 cleanly out-of-scope papers**, discarding the 5 that also
   report a control arm: O/E **1.460** vs 0.989, 9 observed vs 6.17 expected,
   diff +0.470, **p = 0.0118**, MDE ±0.364. Weaker, still above MDE, still
   significant.
4. **Seed sensitivity**: p ∈ {0.00025, 0.00005, 0.00010} over three further seeds.

**Independence, which is the whole point.** The predictor is read from the paper's
own sentences by an adjudicator with no access to the disease label, the label's
provenance, the graph, or any agreement figure. The outcome is computed from the
graph. The 2026-09-17 tier reported human-backed edges at ~89% against model-only
at ~68% and was retracted because "model-only" is *defined* as the residual after
removing every point of agreement. Nothing here has that shape.

**And the honest size.** 13 excess disagreements against 7.4 expected is ~6
observations. The *rate* difference is large, robust and real; the *absolute*
correction is negligible. This is a flag worth having, not a fix worth celebrating —
and per the standing rule, it is **not** an accuracy gain and must never be quoted
as one.

## Result 3 — two named errors, both on the Alzheimer's node

The two largest contributors to the signal above are also the two clearest errors,
which is a convergence worth noting: the structural test and a hand read agree.

**`Combination of gut microbiota and plasma amyloid-β … SILCODE study` — 13 edges
filed under `Alzheimer's disease`, and no subject in the study has Alzheimer's
disease.** Every result sentence contrasts **CN+ vs CN−**: cognitively normal
amyloid-positive against cognitively normal amyloid-negative. The paper is about
*preclinical* AD.

> "The relative abundance of phylum Bacteroidetes was significantly enriched,
> whereas phylum Firmicutes and class Deltaproteobacteria were significantly
> decreased in CN+ individuals in comparison with that in CN− individuals."

10 of those 13 edges sit on **contested** AD pairs, including `Faecalibacterium`
(16 papers) and `Bacillota` (10). This is the disease-dimension analogue of the
2026-09-14 MONDO-id errors, found by a different instrument.

**`Gut Microbiota Changes and Their Correlation with Cognitive and Neuropsychiatric
Symptoms …` — 7 edges under `Alzheimer's disease`**, from AD patients **with** vs
**without** neuropsychiatric symptoms. Everyone in both arms has AD, so AD is the
background, not the contrast — the same shape as the HIV / `Neurocognitive
impairment` case, except that one turned out correct and this one does not.

**Neither was fixed here.** Dropping a paper is a corpus-inclusion decision, and
there is no correct node to move "amyloid-positive but cognitively normal" to.
`contrast_out_of_gate.json` ships the full tiered list **opt-in**, the way the
MONDO disease links did; `graph.json` is untouched.

## Result 4 — a verdict overturned by reading, and the lesson

The blinded reader labelled *"Multiple sclerosis and gut microbiota:
Lachnospiraceae from the ileum of MS twins trigger MS-like disease in germfree
transgenic mice"* as **ANIMAL**, citing a real mouse sentence. Reading the rest of
the packet overturns it:

> "Using pairwise comparison, we found a significant increase of E. tayi in the MS
> twins compared to their healthy twins."

That is a human MS-vs-healthy-co-twin comparison — arguably the best-controlled
design in the corpus — with a mouse transfer experiment alongside. **A title naming
an animal model does not make the paper animal-only**, and a one-label-per-paper
instrument forces a choice on papers that legitimately report several contrasts.
This also means the 2026-09-11 animal-study filter's null still stands: **no
animal-only study is in the graph.**

## Nulls, with power

**The free-text disease label does NOT predict an out-of-gate design.** This was
the session's opening hypothesis and it is wrong — which is why the adjudicators
were blinded to it.

| | papers | out of gate | rate |
|---|---:|---:|---:|
| free-text fallback label | 34 | 2 | 5.9% |
| `DISEASE_MAP` label | 226 | 17 | 7.5% |

diff **−0.016**, Fisher **p = 1.000**, paper-level permutation **p = 1.000**,
**MDE ±8.5 points**. The fallback flag is not a cheap screen for scope, and the
19 out-of-gate papers sit overwhelmingly under *canonical* disease nodes
(Alzheimer's, MCI, MS, PD) — the well-populated ones nobody thinks to check.

**Mixed provenance does NOT predict discordance — and it is the dominant residual
risk.** **94 of 241 in-scope papers (39%) also report a within-disease subgroup
contrast**, so a paper can be in scope overall and still contribute an edge from
the wrong comparison. That is invisible to a paper-level instrument. Tested
anyway: O/E **1.006** (n = 69) vs 0.973 (n = 88), diff +0.033, **p = 0.650**,
**MDE ±0.140**. At a resolution of 14 O/E points, reporting a subgroup contrast
alongside a control contrast does not measurably degrade a paper's agreement with
the rest of the literature. That is the 26th and 27th variable tested against
discordance in this project, and the 26th and 27th null.

## Result 5 — the edge-level version: one strong validation, one honest null

The stated limit of everything above is that it is a **paper-level** instrument
while the dominant residual risk is **within** a paper. `contrast_edge_probe.py`
pushes it down to the observation. It is **deterministic** — no adjudicator, so no
paraphrase risk and no blinding needed. For each of the 3,077 (paper, taxon)
observations it finds the sentences in that paper that name that taxon (joining on
**taxid**, since `relation_sentences_clean.json` already resolves taxa per
sentence) and asks what comparison those sentences describe:

| label | meaning | n |
|---|---|---:|
| `CLEAN` | ≥1 sentence naming the taxon states a healthy/normal-control contrast | 1,272 |
| `UNRESOLVED` | the sentences name no comparison | 1,234 |
| `NO_SENTENCE` | no kept sentence names that taxon | 457 |
| `CANDIDATE` | no control contrast, but ≥1 subgroup or treatment-arm contrast | **114** |

**The same asymmetry the 2026-09-16 recall audit had to state applies here.**
`relation_sentences_clean.json` keeps only sentences carrying a taxon *and* a
direction cue — about 10% of corpus text — so a paper can state its comparison in a
sentence this file never kept. `CANDIDATE` means *"no visible control contrast for
this taxon"*, never *"this observation is wrong"*. **The instrument can flag; it
cannot convict.** 114 is an upper bound.

### The validation, and it is strong

`contrast_probe_validate.py`. These two instruments share nothing: the census
verdict comes from a reader judging the study's **arms**, blinded to the graph; the
probe is a regex over the sentences naming one **taxon**, with no reader and no
notion of study design. If the flag measures what the reader measured, `CANDIDATE`
should concentrate in the papers the reader called out of gate.

| paper class | papers | observations | `CANDIDATE` rate |
|---|---:|---:|---:|
| out of gate | 19 | 88 | **28.4%** |
| in gate, also reports a subgroup contrast | 93 | 997 | 4.5% |
| in gate, control contrast only | 157 | 1,535 | 2.9% |

Out of gate vs in-gate-clean: **diff +0.255, p = 0.00005** (20,000 paper-level
permutations), **MDE ±0.082**, BH q = 0.0001 over the two tests here. A **9.9×
enrichment** that two independent instruments converge on. That is the strongest
construct-validity evidence either of them has.

In-gate-mixed vs in-gate-clean: 4.5% vs 2.9%, diff +0.017, **p = 0.301**, MDE
±0.032 — **null**, and it agrees with the paper-level mixed-provenance null.

### The null, and why its power statement is the finding

`contrast_edge_test.py` asks the question the whole edge-level exercise is for:
**holding the paper fixed**, does an observation with no visible control contrast
disagree with the literature more than one that has it?

The null here is deliberately **not** the project's standing paper-level shuffle,
and the reason is worth recording. That rule exists for paper-level predictors.
This predictor varies *within* a paper — that is the point of it — so a paper-level
shuffle would destroy nothing and test nothing. The exchangeable null is a
**within-paper label permutation** holding each paper's count of each label fixed.

| group | observations | observed | expected | O/E |
|---|---:|---:|---:|---:|
| `CANDIDATE` | 16 | 8 | 7.09 | 1.129 |
| `CLEAN` | 294 | 131 | 132.77 | 0.987 |

diff +0.142, **p = 0.647**, **MDE ±0.473**.

**The power statement is the whole result.** Only **5 of 128 papers** contribute
both a `CANDIDATE` and a `CLEAN` observation among their *decisive* ones, and those
5 are the only exchangeable units the null has. 16 scoreable `CANDIDATE`
observations against an MDE of 0.47 cannot resolve an effect the size of the
paper-level one (0.76). **So this is "the corpus cannot answer it", not "there is
no effect"** — and it explains why the paper-level instrument is the one that
worked. A future session should not spend effort here without more papers.

## What this does and does not say about the 2026-09-11 corpus screen

That screen reported **zero** droppable papers over 249, with the residual caveat
that 231 were cleared from an abstract and never re-read against a measured 2/12
missed-drop rate — "consistent with zero, not proven zero". This is a targeted
re-read of all 271, and it finds 13 papers whose design is outside the extractor's
own gate.

**That is not a contradiction and should not be reported as one.** The screen was
hunting reviews, animal studies, case reports and missing-control-arm designs; it
found none, and Result 4 independently confirms the animal half. Treatment-arm and
within-disease-subgroup designs on otherwise ordinary human cohorts were not on its
list. The extractor itself *does* refuse these when it recognises them — 14 of the
33 zero-yield refusals in `FINDINGS_zero_yield.md` are exactly "intervention trial,
animal model, longitudinal stability, subgroup-vs-subgroup". These 13 are the cases
where it did not.

## Files

| file | what |
|---|---|
| `contrast_packets.py` / `contrast_packets.json` | blinded per-paper packets, 271/271 |
| `contrast_census.py` / `contrast_census.json` | quote verification, census, both tests |
| `contrast_robustness.py` / `contrast_robustness.json` | the four attacks |
| `contrast_out_of_gate.json` | **opt-in** tiered list of out-of-gate papers |
| `contrast_edge_probe.py` / `.json` | deterministic per-observation contrast label |
| `contrast_probe_validate.py` / `.json` | probe vs blinded reader, the 9.9x agreement |
| `contrast_edge_test.py` / `.json` | within-paper test; null, and underpowered by 5 papers |
| `disease_label_packets.py` / `disease_label_audit.py` | the 25 free-text disease nodes (below) |

## Appendix — the 25 free-text disease nodes

`norm_disease` falls through its 17 `DISEASE_MAP` regexes to the extractor's own
string for **25 of 40 nodes, 399 edges, 19.9% of the graph**; 11 of those carry no
MONDO id either. The 2026-09-15 session adjudicated exactly one of them and flagged
that the disease half of the graph has **no curation instrument at all**. All 25
have now been read against their source papers' sentences.

**The result is a near-null and that is good news:** of 33 supported (node, paper)
verdicts, **32 say the label names the condition that actually differs between the
study's arms.** The one exception is the MHE probiotics/rifaximin/lactulose trial.
Five verdicts failed the quote check; three of those are papers with only one or two
relation-bearing sentences in the whole file, which is this instrument's blind spot
rather than a finding.

The audit **independently reproduced the 2026-09-15 `Neurocognitive impairment`
verdict by a different route** — SUBGROUP within an HIV-positive population, label
correct — which is the only inter-instrument agreement check available for that
finding.

Three families where a qualifier creates a separate node (`Spinal cord injury` ×3,
`Intracerebral hemorrhage` ×2, `Hepatic encephalopathy` ×2) are **not** folded here.
They remain what `NEXT_SESSION_PROMPT.md` item 2 says they are: a modelling call for
a human, not a threshold for a script.
