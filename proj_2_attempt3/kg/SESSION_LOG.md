# Session log

Newest first. Nulls and dead ends are logged as results.

---

## TL;DR — 2026-09-19 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Tested.** The one property of an edge no instrument here had ever scored: not
whether its taxon and direction are right, but **whether the COMPARISON it came
from was admissible at all.** `samgated-v1` allows only disease vs healthy
control. A taxon genuinely higher in ICH survivors than in ICH deceased is a
correct reading of its paper, scores as a hit on reading fidelity, the mention
audit, Disbiome/Peryton and the new gold alike, and is still not an edge this
graph should carry. All 271 contributing papers read against their own sentences
(`relation_sentences_clean.json`, 271/271, so no `MAIN_DATA` and no GPU).
**Nothing in `graph.json`, `rag_corpus.jsonl`, `kg.html` or `docs/` was changed.**
Instruments: `contrast_packets.py` → `contrast_census.py` →
`contrast_robustness.py`, `FINDINGS_contrast_scope.md`.

**Adjudicators were blinded** to the disease label, to whether it came from
`DISEASE_MAP` or the free-text fallback, and to everything graph-side — because
two of the three tests compare exactly those things. The tier retracted on
2026-09-17 failed *because* its buckets were defined by the outcome.

**Survived.**
- *The census, and it is a census not a sample.* **2,871 of 3,077 observations
  (93.3%)** come from a confirmed disease-vs-healthy-control contrast. Every
  out-of-gate verdict was read a second time and tiered: 13 papers / 55 obs
  cleanly out of scope, 5 papers / 29 obs out of scope but *also* reporting a
  control arm, 1 verdict overturned. **Out-of-gate is bounded at 1.8%–2.7% —
  quote the range.**
- *Out-of-gate papers disagree with the rest of the literature **1.75×** as
  often.* O/E **1.747** (11 papers, 13 observed vs 7.44 expected) against
  **0.989**, diff +0.758, **p = 0.00015** over 20,000 **paper-level**
  permutations, **MDE ±0.303** — 2.5× what the design can resolve. Outcome and
  null are `paper_discordance_offset.py` unchanged, so it is directly comparable
  to the 25 variables already tested that way. Four attacks: leave-one-paper-out
  worst **p = 0.00070**; the overturned verdict never entered the test
  (`e_dis = 0`); the 8 cleanly-out-of-scope papers alone give O/E **1.460**,
  p = 0.0118 against MDE ±0.364; three further seeds agree.
  **It is independent in the way the retracted tier was not** — predictor read
  from the paper's text by a blinded reader, outcome computed from the graph.
  **And the honest size: ~6 excess disagreements.** A flag worth having, **not**
  an accuracy gain, never to be quoted as one.
- *Two named errors, both on the Alzheimer's node — and they are the two largest
  contributors to that signal, so the structural test and a hand read agree.*
  The SILCODE amyloid paper contributes **13 edges to `Alzheimer's disease` and
  no subject in it has Alzheimer's**: every result sentence contrasts cognitively
  normal amyloid-**positive** against cognitively normal amyloid-**negative**.
  10 of the 13 land on contested AD pairs, `Faecalibacterium` (16 papers) among
  them. A second paper contributes **7 AD edges** from AD-with vs AD-without
  neuropsychiatric symptoms, so AD is the background, not the contrast.
  **Neither was fixed:** dropping a paper is a corpus-inclusion decision and
  there is no correct node for "amyloid-positive but cognitively normal".
  `contrast_out_of_gate.json` ships the tiered list **opt-in**, as the MONDO
  links did.

**Did not survive / null / corrected.**
- *The session's opening hypothesis was wrong, and the blinding is why that is
  knowable.* The free-text disease label does **NOT** predict an out-of-gate
  design: **5.9% (2/34) vs 7.5% (17/226)**, Fisher **p = 1.000**, paper-level
  permutation **p = 1.000**, **MDE ±8.5 points**. The 19 out-of-gate papers sit
  overwhelmingly under *canonical* nodes — Alzheimer's, MCI, MS, PD — the
  well-populated ones nobody thinks to check.
- *Mixed provenance does not degrade agreement, and it is the dominant residual
  risk.* **94 of 241 in-scope papers (39%) ALSO report a within-disease subgroup
  contrast**, so a paper can be in scope overall and still contribute an edge
  from the wrong comparison — invisible to any paper-level instrument. O/E
  **1.006** (n=69) vs 0.973 (n=88), diff +0.033, **p = 0.650**, **MDE ±0.140**.
  Variables **26 and 27**, nulls 26 and 27.
- *One verdict overturned by reading, and the 2026-09-11 animal null still
  stands.* The reader called the MS-twin germfree-mouse paper **ANIMAL** from its
  title; the packet says *"a significant increase of E. tayi in the MS twins
  compared to their healthy twins"* — a human co-twin-controlled study with a
  mouse experiment alongside. **A title naming an animal model does not make the
  paper animal-only**, and a one-label-per-paper instrument forces a choice on
  papers that legitimately report several contrasts.
- *The 2026-09-16 verbatim rule is revised, not repealed.* Machine-checking
  adjudicator quotes stays mandatory, but the raw failure count overstates: **16
  quotes failed byte-for-byte and only 5 are real paraphrases** — the other 11
  differ by the whitespace of PDF extraction (`" , "`, `"[ 32 ]"`), which a reader
  tidies silently. Report both tiers.
- *The free-text disease nodes are a near-null, which is good news.* All 25 read
  against their sources (`disease_label_audit.py`); **32 of 33 supported verdicts
  say the label names the condition that actually differs**. The one exception is
  the MHE probiotics/rifaximin/lactulose trial. It also **independently reproduced
  the 2026-09-15 `Neurocognitive impairment` verdict by a different route** — the
  only inter-instrument check that finding has.
- *Bookkeeping that mattered.* The 2026-09-17 session (commits `30120ea`,
  `09b341a`) was never logged, so for two days this log's top entry — which the
  scheduled routine tells every session is the current state — described a project
  whose accuracy numbers were still the old ones. Entry reconstructed and marked
  retroactive. `CLAUDE.md` still told readers to prefer Disbiome/Peryton because
  "F1 0.680" was agreement with a flawed reference; it now carries F1 0.739, the
  double-counting-matcher correction, the gold-is-a-test-set rule, and the
  retraction. Contested count corrected 217 → **220**.

**Highest-value next step.** **Put the two Alzheimer's papers in front of a
human** — 20 edges, a one-line decision each, and the only thing this session
found that changes graph content. Then, if a cheap lever is wanted: the
out-of-gate flag is validated but paper-level, and the dominant residual is
*within*-paper (39% of in-scope papers report a subgroup contrast too). An
edge-level version — which comparison does *this* sentence describe — is the
natural successor and needs no GPU, only the same packets at sentence
granularity.

---

## TL;DR — 2026-09-17 (local, Mac — logged retroactively 2026-09-19)

*This session's work reached the repo as commits `30120ea` and `09b341a` but was
never written into this log, so for two days the log's top entry (2026-09-16)
described a state in which the project's accuracy numbers were still the old
ones. Reconstructed here from those two commit messages and `FINDINGS_newgold.md`,
which are the authoritative record.*

**A new hand-curated gold standard replaced the old reference, and the extractor's
F1 went 0.639 → 0.739 with nothing about the model changing.**
`high_confidence - final_constrained_override.csv`, 334 DOIs, curated by Emily
Song with no LLM assistance, scored over the 260 cached-extraction papers that are
scoreable against it. Taxonomy-aware (LCA) P 0.721 / R 0.759 / **F1 0.739**;
char-ngram 0.733. Permutation over 1,000 draws: null mean 0.148, **p = 0.001**.
The old 0.639 was agreement with an incomplete reference, and `CLAUDE.md`'s
long-standing "F1 0.680, agreement with a flawed reference" caveat is superseded
by it. Instruments: `run_lca_eval.py`, `score_lca.py`, `score_newgold.py`,
`FINDINGS_newgold.md`.

**Two corrections folded into that number before it was quoted.**
- *The taxon matcher double-counted, and every F1 in this project was inflated.*
  `eval-v2/run_eval.py` and both matchers in `score_lca.py` allowed two predictions
  to claim the same gold taxon and each score a true positive. Fixed; the pre-fix
  figures were 0.755 / 0.780, including on `leaderboard.csv`.
- *LCA matching is worth +0.007, not +0.025.* Of 112 LCA "rescues" only 9 claimed a
  gold taxon nothing else had matched; the rest were redundant credit created by
  the double-counting bug.

**The cleanest extractor-quality statement the project has, because it depends on
neither the in-house gold nor the model.** Same 259 papers, human annotation and
model output each scored *independently* against a third party:

| | edges | Disbiome | Peryton |
|---|---:|---:|---:|
| human (Emily) | 1,643 | 80.5% | 80.9% |
| model (Qwopus3.5) | 1,831 | 74.8% | 73.5% |
| gap | | +5.7 | +7.4 |
| Fisher exact | | p = 0.275 | p = 0.186 |

**Neither gap is significant.** On the same papers the model sits within ~6 points
of a human curator, and at this n that is not distinguishable from chance. It also
puts a ceiling on the room above: a human curator's own agreement with Disbiome is
**80.5%**, not 100%.

**Retracted in the same 24 hours: the per-edge provenance "quality filter".**
`30120ea` built a union graph (extraction ∪ gold, 2,500 edges) and reported
human-backed edges at ~89% agreement against model-only at ~68% — a 21-point gap
at p<0.001, presented as a validated quality tier. `09b341a` **retracts it**: the
"model-only" bucket is *defined* as the edges the human did not confirm, i.e. the
residual after removing every point of agreement. That measures corroborated vs
uncorroborated, not human vs model. The independent three-way comparison above is
the replacement, and it shows no significant gap.

**The union graph itself was also reverted, on a constraint that was already on
record** (*"No Emily's data since this is manually verified"*). The gold MEASURES
the extractor; merging it would make the graph partly hand-curated and the claim
the graph exists to support — *this is what a model extracted from the literature*
— would stop being true. `graph_union.json`, `kg_union.html`,
`extractions_union.json`, `build_union_input.py` and `annotate_provenance.py` were
removed. **`graph.json` is unchanged**: 271 papers, 883 taxa, 40 diseases, 2,008
edges, 220 contested, 727 containment links.

**Two hypotheses about the residual false negatives, tested and rejected**: tables
recover nothing, and taxon-name normalisation recovers 1.3%.

---

## TL;DR — 2026-09-16 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Tested.** The one direction no instrument here had ever pointed: not whether the
edges in the graph are right, but **whether relations the papers state produced no
edge at all.** Every existing fidelity number scores edges that exist (reading
fidelity 86.6%, mention rate 99.57%, Disbiome/Peryton 73.0/72.5%). The only recall
number the project has ever had, F1 ~0.59, is scored against the in-house gold —
which is under audit and known unreliable. Both audits below score against **the
papers' own sentences**. **Nothing in the graph was changed:** `graph.json`,
`rag_corpus.jsonl`, `kg.html` and `docs/` are untouched. Instruments:
`zero_yield_audit.py` / `FINDINGS_zero_yield.md`, `edge_recall_audit.py` /
`FINDINGS_edge_recall.md`.

**THE LESSON OF THE SESSION, and it is a method rule, not a number.**
**An audit that scores the extractor without applying the extractor's own gate
manufactures misses.** The extraction prompt (`eval-v2/run_eval.py`,
`samgated-v1`) does not extract anything that merely states a direction — it
requires **reported statistical significance** (*"if significance is unclear or
unreported for a taxon, omit it"*), **main text only**, and **disease vs healthy
control only**. The first pass of the paper-level audit ignored this and reported
**4 confirmed misses; the true confirmable count is 1.** The same gate cut the
edge-level audit from **36 raw miss verdicts to 15**. Corrected in-session, before
either number was quoted anywhere. Any future recall work must apply the gate and
state the asymmetry below.

**Survived.**
- *The funnel, measured for the first time.* 325 screened → 313 after `build_kg`
  title dedup → **271 contributing = 86.6% paper yield**. The 42 papers that
  contribute nothing had never been looked at.
- *Paper-level recall: **≥96.1%**, and **99.6%** [97.9, 99.9] counting only the one
  confirmable miss.* Of 42 zero-yield papers, 9 have no relation-bearing sentence at
  all; of the 33 that do — 14 correct refusals on design (intervention/probiotic
  trial, animal model, longitudinal stability, subgroup-vs-subgroup), 4 explicit
  negative results, 4 background-only, 7 unclear, **1 confirmed miss**, and 3 that
  state a direction with **no reported significance** (one says outright *"a
  tendency towards a reduction"*, one carries a citation marker `[ 32 ]`) — correct
  refusals under the prompt's own rule. **Quote the range, not a point estimate.**
- *Edge-level recall — the number that never existed: **~98.1%**, 95% CI
  [96.4, 99.5]*, ~60 missed observations [14, 116] of 3,077. From 378 candidate
  (paper, taxon) pairs across 94 papers, a random sample of 24 papers / 95
  candidates adjudicated, gate applied. **State it as an UPPER bound:** the
  candidate generator only sees taxa in sentences the provenance screen keeps (~75%
  paper-level recall) and the significance gate can be confirmed from visible text
  but never refuted — both biases push the same way.
- *The actionable part is the clustering, not the average.* **12 of the 15 confirmed
  missed observations come from 3 papers** (an HBV-cirrhosis study, an Egyptian PD
  cohort, a Ugandan AD cohort). So a **short-list re-extraction recovers most of the
  loss without a corpus-scale GPU run** — the candidates are already enumerated in
  `edge_recall_packets.json`.

**Did not survive / null / corrected.**
- *Deterministic null, and it was worth checking: `build_kg.py` loses nothing.* Of
  the 54 screened papers absent from the graph, **10 are dedup twins whose surviving
  copy IS in the graph**, and after collapsing those the number of papers whose
  extractor returned taxa that never became an edge is **0**. A silent drop in one of
  that script's `continue` branches or its `min_papers` filter would have been
  invisible to every existing instrument.
- *High-rank taxa are NOT missed more often — one keystroke from being a false
  finding.* 33% of confirmed misses are phylum/class against **6.5%** of graph
  edges, and an adjudicator independently volunteered the pattern. But **27% of the
  CANDIDATES are phylum/class** — the generator is itself rank-skewed, because
  high-rank taxa live in the summary sentences the screen preferentially keeps.
  Against the pool it was actually drawn from: **p = 0.0537**, paper-level
  permutation preserving each paper's confirmed count, N=20,000, single uncorrected
  test, 15 events. The graph's 6.5% is the wrong denominator.
- *Zero-yield is NOT concentrated in any disease.* Paper-level permutation
  (N=20,000), **max-statistic over the 11 diseases with n≥5** so the multiple
  comparison is controlled by construction: worst rate 0.40 (Dementia, n=10),
  **p = 0.128 — NULL**. A 10-paper group cannot push the MDE below roughly 30 points,
  so this is "consistent with no disease bias", not "proven none".
- *The deterministic provenance screen is triage, not adjudication.* Reusing
  `audit_direction_witness.py`'s regexes scores **75% recall, 33% precision** against
  the read verdicts. Its false negative is instructive: `RESULT_CUE` demands a
  statistic or an explicit "we found", so it misses *"the Tannerellaceae family was
  lower in PwMS than HC"*. Same conclusion the abstract screen reached 2026-09-11.
- *Dedup has no gap.* The only title pair sharing 60 leading characters without
  collapsing is two genuinely different papers (phlegm-heat syndrome vs
  ischemic/hemorrhagic stroke). Correctly not merged.
- *Adjudicator reliability, such as it is.* Two papers entered the paper-level
  adjudication twice as dedup twins with identical text, in different batches, and
  received **identical verdicts (2/2)**. Bounds nothing at n=2, but it is the only
  inter-rater signal available. Separately, **several "verbatim" quotes supplied by
  adjudicators were paraphrases** — 7 of 36 edge-level miss claims failed the
  verbatim check outright. Always machine-check the quote.
- *`ftp.ncbi.nih.gov` re-probed, still 403 — eleventh session.* The scheduled
  routine's prompt still tells sessions to download the taxdump and still lists the
  spent Task 0/1/2.5/3 priority order. **That prompt is stale and is costing the
  opening of every run.** `NEXT_SESSION_PROMPT.md`'s header now says so explicitly.

**Highest-value next step.** **Re-extract the short list, not the corpus.** The
recoverable recall loss is concentrated in a handful of the 94 candidate papers,
already enumerated. Second, and free: the misses are dominated by taxa whose
significance cue sits in a *different sentence* from the direction, so a
generation-2 filter that links a direction sentence to a neighbouring significance
sentence would raise this instrument's own sensitivity and shrink the
"cannot refute" gap.

---

## TL;DR — 2026-09-15 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Tested.** Whether a disease node's LABEL describes the cohort its papers actually
studied — the one question the 2026-09-14 assignment audit did not ask (that audit
compared predicted label vs datasheet label, never label vs cohort, and does not
mention HIV anywhere). **Nothing in the graph was changed.** `graph.json`,
`rag_corpus.jsonl`, `kg.html` and `docs/` are untouched, so nothing here can be
miscited as an accuracy gain.

**The suspected error was NOT an error — and the title would have fooled me.**
`Neurocognitive impairment` (1 paper, 7 edges) sits in the cognitive-decline cluster
and its one paper is *"...Neurocognitive Impairment in **HIV-Infected Population**"*.
That looked like the disease-dimension twin of the never-join-on-another-database's-id
rule. It is not. The paper's own sentences say the contrast is **"the NCI group" vs
"the non-NCI group"**, and *"associated with NCI **in people with HIV**"* — HIV is the
background population **held constant across both arms**. The edges are correctly
typed; the extractor was right and the datasheet ("Cognitive Impairment") was coarser.
**Rejected by reading the paper, not by reasoning from its title.** Residual caveat is
interpretive only: every observation comes from an HIV⁺ population where composition
and immune status are entangled (the paper has *Treponema_2* inversely correlated with
CD4 count), so transfer to non-HIV cognitive impairment is untested.

**Survived.**
- *20% of the graph's edges hang off strings no vocabulary ever approved.*
  `norm_disease` tries 17 `DISEASE_MAP` regexes and on a miss falls through to the
  extractor's `predicted_disease`, title-cased, as a node label. Never measured
  before: **`DISEASE_MAP` 15 nodes / 1,609 edges (80.1%) vs fallback 25 nodes / 399
  edges (19.9%) over 38 papers**, and **11 of those 25 nodes (145 edges) carry no
  MONDO id either** — neither a regex nor an ontology has ever seen them. This is a
  census of all 2,008 edges, not a sample, so no permutation test applies and none is
  reported. The fallback is deliberate and defensible (don't silently drop papers);
  what is new is its size.
- *The asymmetry that exposes.* The taxon half has `taxon_typos.py` and
  `multi_taxon.py`. The disease half has **no curation instrument at all**.
  `Cognitive impairment` is a node distinct from `Mild cognitive impairment` not
  because anyone judged them distinct but because a string matched no regex — and
  **all four of its papers were labelled Dementia / Alzheimer's / Other by the human
  datasheet.**
- *4 genuine candidate label families, 383 edges*: `[cognitive impairment]` 209,
  `[cord injury]` 101, `[intracerebral hemorrhage]` 61, `[hepatic encephalopathy]` 12.
  **Nothing folded — that stays a human call.**
- *The 5th family is the heuristic's own false positive and is left in the output on
  purpose.* `[s disease]` groups Parkinson's + Alzheimer's + Huntington's — **689
  edges — on a shared suffix.** A string-similarity folding rule would have merged
  three unrelated neurodegenerative diseases. Same lesson the taxon side paid for with
  `Oscillospirales`/`Oscillospira`; it is why this ships as a candidate generator and
  why `taxon_typos.py` uses a curated table, not a threshold.

**Did not survive / null / corrected.**
- *Correction supplied to a shipped claim: **0 of 16 cognitive-cluster pairs survive
  BH** at q=0.05.* The 2026-09-14 cluster-level conclusion stands and is robust —
  dropping the HIV node moves 0.592 → **0.582**, i.e. *further* from the 0.669
  background, not toward it. But that session's per-pair language ("the structure is
  sharp inside it", and `CLAUDE.md`'s "0.938" line) reads as established pair-level
  structure. Tested individually with an exact two-sided binomial + Benjamini–Hochberg
  over all 16 pairs: MCI/Dementia 4/13 p=0.014 (crit 0.0031), AD/Dementia 15/16
  p=0.030 (crit 0.0063), AD/MCI 13/26 p=0.093, MCI/CI 9/18 p=0.138 — **none survives,
  including the MONDO-confirmed AD/Dementia link.** The pooled test is still a result;
  **no individual pair contrast should be quoted as established.**
- *Both folding questions are underpowered, not answered.* MCI vs `Cognitive
  impairment`: 9/18 = 0.500 vs 0.669, p=0.138, **MDE 28.0 points** against an observed
  gap of 16.9 — this corpus cannot resolve it. `Neurocognitive impairment`: n=5, **MDE
  46.9 points**, essentially no power. So NCI should not be folded **because its
  source population is HIV⁺**, not because the data says so; the data says nothing.
- *`Neurocognitive impairment`'s MONDO id stays `None`, as a recorded refusal.* MONDO
  2026-09 has **no term whose name contains "neurocognitive"** (0 of 36,017 named
  terms). The nearest reachable candidate is `MONDO:0020689` "AIDS dementia complex"
  (`is_a` dementia) — the dementia-stage endpoint, so assigning it would repeat the
  **ASD/autism one-rank-too-narrow mistake fixed on 2026-09-14**. Do not "fix" it.
- *A standing cross-session assumption is wrong, and this unblocks cloud work.* Since
  2026-09-11 every session has recorded that cohort/sentence questions are blocked in
  the cloud because `MAIN_DATA.json` is gitignored. True of **full** text, false of
  the text these questions need: **`relation_sentences_clean.json` is committed and
  covers 271 of 271 contributing papers (100%) — 6,294 sentences, 1.74 MB**, taxa and
  direction cues tagged. The HIV question above was settled entirely from it, in the
  cloud. **The limit is real though:** it holds ~10% of corpus text (16.78 M → 1.74 M
  chars), so it cannot answer "does taxon X appear *anywhere* in this paper", which is
  what `silent_edge_mentions.py` needs — **item 0 really does still require the Mac.**
- *`ftp.ncbi.nih.gov` re-probed, still 403.* Tenth session. The scheduled routine's
  prompt still instructs sessions to download the taxdump; that instruction is stale.

- *Dead end closed, because it is tempting: the datasheet label is NOT a hierarchy
  source.* For 3 of the 4 candidate families the human datasheet already puts every
  member paper under one label (all 8 cord-injury papers = `Spinal cord injury (SCI)`,
  all 4 ICH papers = `Stroke`, both HE papers = `Encephalopathy`), which makes it look
  like a free source of parent links. Scoring it that way yields 9 nodes / 219 edges
  and is **wrong**: it proposes `Spinal cord injury is-a ALS` (one member is a
  comparative ALS-vs-SCI study carrying both tags), `CADASIL is-a Stroke` (CADASIL
  *causes* strokes), `Tuberous sclerosis is-a Epilepsy` (backwards), and — decisively
  — `Essential tremor is-a Parkinson's` and `MSA is-a Parkinson's`, **the exact two
  folds the 2026-09-14 MONDO pass independently REJECTED** (sibling and cousin,
  rejections upheld). **0 for 2 on the only cases with an external authority.** The
  datasheet's `disease` column is a topic/cohort tag, not a taxonomy; comparative
  studies carry two. So the fallback nodes are not just unvalidated, they are
  **unvalidatable from material already in the repo** — closing the gap needs MONDO
  coverage these labels lack, or a human.

**Highest-value next step.** Unchanged and now better evidenced: **more papers.** Both
folding questions above, and 0-of-16 surviving BH, are n-limited rather than
method-limited. The cheapest thing that does not need a GPU is a **curated disease
label table** — the disease-side analog of `taxon_typos.py`, covering the 25 fallback
nodes — but the folds themselves are a clinical call and must not be automated.

---

## TL;DR — 2026-09-14 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Tested.** Whether the disease half of the graph survives contact with a disease
ontology. The scheduled routine's priority list (MAIN_DATA filter, Task 1, 2.5,
3.1) is spent — five sessions running — and the top open lever needs
`MAIN_DATA.json`, which is gitignored and absent here. So this went at the #2
item: the disease-subtype modelling call, open three sessions and logged as
"needs a PI, not a script" **because there was no authority to appeal to**.

**The unblock.** There is one, and it was a network assumption that was too
broad. `ftp.ncbi.nih.gov` and `eutils` are blocked (403, re-probed today, ninth
session) — but
`github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo` returns
**200 and 53 MB**. `purl.obolibrary.org` and `ebi.ac.uk` are blocked; the GitHub
release is not. So `mondo.py` is now the disease-side analog of `taxonomy.py`:
exact label/synonym matching, obsolete terms followed through `replaced_by`,
unresolved labels reported rather than guessed, curated aliases with recorded
refusals. **28 of 40 disease labels resolve.**

**Survived.**
- *Two wrong MONDO ids were shipped on 209 of 2,008 edges, and the published
  page carried them.* `Mild cognitive impairment` — 13 papers, **154 edges** —
  carried `MONDO:0005453` = **congenital heart disease**. `Autism spectrum
  disorder` carried `MONDO:0005260` = *autism*, a **child** of ASD. MONDO holds
  **no term named "mild cognitive impairment" at all** (0 of 104,643 index keys),
  so `None` is now correct and must not be "fixed" back. A third id was settled
  rather than guessed: Anti-NMDAR encephalitis → `MONDO:0021081`. `CLAUDE.md`
  already forbids joining on another database's taxid; **this is that failure one
  dimension over**, and it shipped because every fidelity instrument in this repo
  scores the taxon half.
- *MONDO grades the hand-written tiers, and upholds them.* 2 of 2 checkable is-a
  claims **CONFIRMED** (Alzheimer's under Dementia, ICH under Stroke); 2 of 2
  checkable Tier-C **rejections upheld** (MSA cousin, essential tremor sibling of
  Parkinson's). The 2026-09-10 human judgement call was right. The other 8 are
  unresolvable — MONDO carries no graded or cause-specified subtype terms — which
  cuts both ways: the tiers do work MONDO cannot, and cannot be validated by it.
- *The 71-paper cognitive-decline cluster does NOT cohere, and that settles the
  design call.* Pair as the unit: **61/103 = 0.592** against a background of
  **0.672** over 453 cross-cluster pairs, p=0.883 — it fails in the **wrong
  direction**, so no power statement rescues it. Inside it the structure is
  sharp: the MONDO-confirmed Alzheimer's/Dementia link runs **15/16 = 0.938**
  while **every MCI pair is at or below a coin flip** (AD/MCI 13/26, MCI/CI 9/18,
  **MCI/Dementia 4/13**). Two authorities consulted separately agree — MONDO's
  vocabulary refuses MCI as a disease, and the graph's own microbial data says
  MCI's directions are uncorrelated with Alzheimer's. Neither was derived from
  the other, and it retrospectively supports the Tier-C rejection of exactly that
  fold, made on clinical grounds before any of this was measured.
  **Recommendation: link the cognitive nodes for RETRIEVAL, do not pool their
  EVIDENCE, do not fold MCI into Alzheimer's.**
- *Three confounds checked before believing it, all null.* (a) Not one contrarian
  paper — discordance clusters by paper here (p=0.0003), and this is spread over
  **at least 6 of 13** MCI papers, the largest contributor backing agreements too
  (3 vs 2). (b) Not taxon ubiquity — over 249 pairs, **corr(mean shared-taxon
  breadth, agreement) = +0.000**, tertiles flat (0.681/0.636/0.673). (c) Not
  shared papers — **paper overlap between disease nodes is ZERO for every pair**,
  so the inflation that makes 73.0/72.5 a blend cannot operate here. That was the
  main statistical risk and it does not apply.

**Did not survive / null / corrected.**
- *My own parser, caught by its own self-test before any result was read.*
  MONDO's `is_a` lines carry trailing `{source="..."}` qualifiers, so ids were
  read with the qualifier attached and **every ancestor lookup silently returned
  nothing**. Fifth time an instrument here was weaker than what it audited;
  **second consecutive time a built-in control caught it instead of a
  spot-check.** Keep doing this. The control now reads `DISEASE_MAP` *live* from
  `build_kg.py` rather than copying it — a control holding its own copy cannot
  detect drift in the table it checks, which is how "174 contested" survived
  three sessions after the number became 217. 16/16.
- *Does ontological proximity predict directional agreement? NO, and the power
  statement is unusually actionable.* is-a pairs agree 34/38 = 0.895 vs 0.692 for
  distant pairs — which looks like a 20-point effect and **is not evidence**,
  because those 38 observations come from exactly **2** disease pairs. Pair-
  clustered null: p=0.082, **MDE +29.9 points** on a 66.4% base. **4 is-a disease
  pairs would resolve a 20-point effect; the graph can form 2.** Not "more
  papers" — two more linkable disease pairs.
- *The shipped disease layer buys NO retrieval reach, measured not assumed.*
  `graphrag.py` now loads the 2 links (opt-in `disease_hierarchy_links.json`, so
  `graph.json` is untouched and `build_kg.py` cannot drop it). Subgraph with vs
  without: **identical in every cell** — Dementia k=12 79=79 papers, k=25
  152=152; Stroke k=25 157=157 — one PPR nudge (0.0499→0.0532) that changes no
  rank. PPR already connects any two diseases sharing one taxon, and at the
  disease level this graph is nearly complete. **I corrected my own claim from
  earlier the same session:** the "8× Dementia expansion" is real for a
  *node-scoped* view and is **not** a GraphRAG gain.
- *`rag_corpus.jsonl` was 7% stale and nobody had noticed.* Never regenerated
  after the 2026-09-08 punctuation fold or the 2026-09-11 spelling fold: **148
  phantom documents** for taxa that are no longer nodes (`[ eubacterium ]`,
  `[eubacterium]_rectale_group`) and **145 real edges missing outright**. The
  retriever served them *and the retrieval ground truth was derived from them*
  (per-query truth counts moved 64→66, 25→21, 8→5, 12→11, 116→120 on rebuild).
  Rebuilt; now matches the graph's edge set by set equality.
- *Which REVERSES Task 2.5's ordering, confirming that gap was always noise.*
  GraphRAG 0.800→**0.683**, BM25 0.783→**0.700**. A 1.7-point gap whose sign
  flips under an unrelated data correction is noise. **Exact** two-sided
  sign-flip permutation over all 2⁶=64 assignments: **p = 1.000**, with only 2 of
  6 queries differing at all and in opposite directions; the comparison cannot
  resolve a mean difference below ~0.17, ten times the gap. Neither number should
  ever have been quoted as a ranking. Addendum appended to
  `FINDINGS_task2.5_graphrag.md`.
- *The negative control has ZERO power and is reported as void, not as passing.*
  MSA/Parkinson's agrees 8/8 and essential tremor/Parkinson's 1/2 (pooled 9/10,
  p=0.130) — but its MDE requires a rate above **1.000**, so it cannot reject
  anything at 2 pairs. The 8/8 is a hypothesis worth noting (both are
  α-synucleinopathies) not a result. The SCI and hepatic clusters are void too.
  Only the cognitive cluster (16 pairs) has any power at all.

- *Backfilled the ids the regex table never had.* `build_kg.py`'s 17 regexes
  only ever carried ids for 16 labels, so 24 disease nodes shipped `mondo=None`
  even where MONDO has an exact term. A small committed table
  (`disease_mondo_ids.json`, regenerated by `mondo.py`, so the builder never
  depends on the 53 MB .obo) takes **28 of 40 disease nodes to a MONDO id where
  15 had one** — 254 further edges, largest being Spinal cord injury 81, ICH 50,
  MSA and CADASIL 20 each. `DISEASE_MAP` still wins outright where it has an
  entry *including where its id is deliberately None*, so the table cannot put an
  id back on MCI; `mondo.py` asserts that, and the rebuild confirms it. Verified
  before shipping that **all 25 automatic resolutions matched a primary name or
  an `EXACT` synonym — none a loose `RELATED` synonym**, which is where
  "Lewy body disease → Lewy body dementia" slippage would come from.
- *A SECOND bug in my own resolver, found only because two numbers disagreed by
  one.* Resolved count said 27; exact-matches-plus-aliases said 28. Curated
  aliases were consulted only when a label had **no** index hits — but `CADASIL`
  *does* hit two MONDO terms (general term and type 1), came back `ambiguous`,
  and silently ignored its curated entry. Aliases now apply whenever exact
  matching fails to yield exactly one live id. **Nothing flagged this but an
  arithmetic disagreement**; that check costs nothing and caught what no
  spot-check would have.
- *NULL, and logged so nobody re-hunts it: MONDO says NO two of the 40 disease
  labels are the same term.* This is the disease analog of the
  Bacteroidetes/Bacteroidota fold, and the repo has been bitten by it once (three
  spellings of anti-NMDAR encephalitis were three nodes; folding them turned 4
  apparently-single-paper edges into replicated ones, 3 contested). **Zero
  collisions** among resolvable labels — there is no second such case hiding.

- *Anomaly hunt, and it paid off for the third time: 24 taxon nodes had their
  parent written in their own label and were detached anyway.* 170 of 883 taxon
  nodes have no parent containment link, 118 of those unresolved; for 24 the
  label names a parent that is already a resolved node
  (`unclassified_f_Lachnospiraceae`, `norank_f_Christensenellaceae`,
  `gut_metagenome_g_Faecalibacterium`). They fell between two mechanisms — the
  placeholder branch only fires for labels the taxonomy DID resolve, and these
  resolve to nothing, so `taxonomy.py` gives them no lineage either. The
  contrast proving it is a gap and not a policy: the *resolved* `unclassified X`
  nodes are already linked, because NCBI mints real taxids for those subtrees
  (`unclassified Bacteroides` = 2646097). **14 linked, 13 refused with reasons,
  91 correctly left detached. Orphans 170 → 156, hierarchy 713 → 727.**
- *And the refusals are the more useful half — FOUR of the 24 are
  bacteriophages.* `Klebsiella virus KP36`, `Streptococcus phage EJ 1`,
  `Enterococcus phage EFAP 1`, `Escherichia virus JES2013`. A phage *infects*
  Klebsiella; it is not contained in it, and a generic "link to the taxon named
  in the label" rule would assert four false taxonomic relations and let a query
  rolling up Klebsiella absorb virus evidence. Same lesson as `taxon_typos.py`
  (edit distance would have merged `Oscillospirales` into `Oscillospira`). Hence
  a curated table with recorded refusals, `orphan_parents.py`.
- *Two cross-rank disagreements became visible — the layer working, not a side
  effect.* **`Faecalibacterium` depleted in Alzheimer's across 16 papers while
  `gut_metagenome_g_Faecalibacterium` is reported enriched** (1 paper, no shared
  paper), and `Flavobacteriaceae` depleted vs `norank_p_Flavobacteriaceae`
  enriched in Stroke. Both correctly tagged `no_shared_paper`. The first is worth
  a reviewer's attention.
- *The joint-label refusal SHARPENS the last open modelling call rather than
  resolving it.* `Escherichia-Shigella` denotes reads that could be **either**
  genus, so it is not a subset of Escherichia and containment is the wrong
  primitive in both directions. The question has always been posed as
  *attribute-to-one / split / hold-apart*; what a user actually wants — the joint
  node reachable from either parent without either absorbing its evidence —
  needs a **different edge type** ("ambiguous assay", not `parent_of`). That is a
  schema decision for a human, and with the disease-subtype call now answered it
  is **the last genuinely open modelling call in the graph**: 5 nodes, 13 edges.
  See `FINDINGS_orphan_parents.md`.
- *The published page was verified by RENDERING it, not by reading it.* Installed
  playwright and ran `verify_viz.py` against the rebuilt `kg.html`: **32 passed,
  0 failed.** A blank-canvas bug here once passed every static check, so this is
  the check that counts.

**The rebuild was gated, not trusted**, because two fixes here have silently
erased themselves on rebuild while printing success. Acceptance condition set in
advance — *nothing but the `mondo` field may change* — and met exactly: 923→923
nodes, 2008→2008 edges, 713→713 hierarchy, 271→271 papers, hierarchy and papers
tables identical, no meta diffs, 248 edges differing in `mondo` **and no other
field**, and a second rebuild bit-identical to the first. `kg.html` differs on
exactly 496 lines, all `"mondo"`; `docs/index.html` re-synced and md5-identical.

**Scope discipline.** These are **identifier** corrections. They touch no edge,
no direction, no count, so agreement with Disbiome/Peryton **cannot move** and
was not re-measured. Do not report any of this as an accuracy gain.

**Also.** Container came up on a **detached HEAD** for the fourth session
running; re-attached to `main` before any work.

**Highest-value next step, unchanged and now the only cheap one left: re-run
`silent_edge_mentions.py` on a machine that has `MAIN_DATA.json`.** It closes 144
unscoreable edges and 795 unscoreable observations (25.8%) with no GPU, no
taxdump, no new papers. Below that: more papers (needs a GPU — **ask first**),
and two precisely scoped design calls: the MCI/Cognitive-impairment boundary
(MONDO carries none of those labels), and an "ambiguous assay" edge type for the
joint two-genus nodes. Write-up: `FINDINGS_disease_ontology.md`.

---

## TL;DR — 2026-09-13 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Tested.** Whether the 580 edges the last session sized as beyond any prose
instrument are fabrications or merely table-only reports. The scheduled routine's
priority list (MAIN_DATA filter, Task 1, 2.5, 3.1) is spent — four sessions
running — so this went after the largest unverified block in the graph instead.

**Survived.**
- *The "unverifiable" 580 are overwhelmingly real.* **429 of the 436 scoreable
  (98.4%)** name their taxon in their own source paper; 144 are unscoreable here
  because their full text lives only in the gitignored MAIN_DATA.json. Across the
  whole extraction, **2,291/2,301 claims (99.57%)**. `silent` meant "invisible to
  the relation filter", never "absent from the paper" — those are now separated.
- *And it is not a vacuous test.* Paper-level shuffle, 100 permutations: null
  **14.96% ± 0.81** vs true **99.58%**, gap **84.6 points**, p=0.0099 (the floor
  at 100 perms; the gap is ~105 null SDs). Rarity-weighted rate **99.52%** and
  only 2.7% of claims name a taxon in over half the corpus, so ubiquitous genera
  are not carrying it. `relation_sentences.py --validate` has reported this
  ceiling since Task 1 with **no null attached**; now it has one.
- *One confirmed extraction error in 2,301 claims (0.04%).*
  `Clostridiales incerte sedis XIII` / Parkinson's: the paper says `incerte sedis
  **xii**` once and XIII **zero** times. Both are real RDP labels, so this is a
  moved roman numeral, not a parse artifact — the same class as the 2026-09-12
  one-letter genus bug. Blast radius: one placeholder-node edge.

**Did not survive / corrected.**
- *My own matcher, caught by a POSITIVE control.* `own` and `background`
  observations name their taxon in a sentence by construction and must score
  100%. The first join scored `own` at 79.7% — impossible. It fed
  `taxon_matchers()`'s `norm_surface`'d forms (whitespace stripped,
  `akkermansiamuciniphila`) into unstripped text, forcing every multi-word taxon
  absent. Final: **1520/1520 and 474/474**. Fourth time an instrument here was
  weaker than what it audited; **first time a built-in control caught it instead
  of a spot-check.** Cheap, and should be standard.
- *A subagent's adjudication, overturned by one line of code.* Haiku called
  `Lachnospiraceae_UCG-001` a genuine fabrication ("paper has UCG-004, not
  UCG-001"). The paper has both — `g_ lachnospiraceae _ucg-001`, with a space
  before the underscore. Its other nine verdicts were correct and diagnostic.
  The standing rule held.
- *Seven of ten residuals are matcher limits, not graph errors*: genus factored
  over a list (`blautia species wexlerae , faecis and massiliensis`),
  parenthetical abbreviation (`gemmiger (ge.) formicilis`), and short epithets
  (`cl. sp cag 273`). **Deliberately not absorbed** — the matcher understates by
  design rather than being tuned to flatter its own result.

**Second finding — the disease half of every edge had never been audited.**
`build_kg.py:383` keys the disease node on the LLM's `predicted_disease`, using
the human label only as fallback, so every fidelity instrument in this repo
scores the taxon half and none scores this one. A wrong disease misfiles a whole
paper's edges. Audited deterministically (`verify_disease_assignment.py`):
**zero of 325 records name a disease absent from the paper's text.** Of 15
family-level conflicts with the datasheet — 7 are the extractor right and the
sheet coarse, 5 comparative studies, 2 ambiguous, 1 suspect, **0 unsupported.**
One is a *datasheet* error: the MUC2 paper is labelled ALS+Parkinson's but
`amyotrophic lateral sclerosis` occurs **0 times** in it and `multiple sclerosis`
119 times including the title — a fourth independent sign the in-house
annotations are unreliable. Note the instrument trap: scoring on body-text
frequency alone called six CORRECT MCI assignments suspect, because an MCI study
discusses Alzheimer's throughout (MCI is its prodrome). Title evidence first.
- *And it sizes a modelling decision that has been open for three sessions.*
  **11 of the 15 conflicts sit on the MCI/Alzheimer's/Dementia boundary.** The
  graph holds **six** cognitive-decline nodes over **71 papers** — AD 46, MCI 13,
  Dementia 6, Cognitive impairment 4, Neurocognitive impairment 1, Subjective
  cognitive decline 1 — and **0 hierarchy links between any two disease nodes**,
  against 708 for taxa. A query for Alzheimer's silently misses 13 MCI papers.
  "Should disease subtypes be modelled as containment?" is still a PI's call, but
  it is not a tidy-up of a few edge cases: it is the largest disease cluster in
  the graph. **Raise its priority.** See `FINDINGS_disease_assignment.md`.

**Scope discipline.** The mention audit bounds **fabrication only**. It does NOT move
73.0/72.5, and is a different quantity from the 86.6% reading fidelity. The 429
edges keep their `provisional` tier: their direction is still unverified by
anything in this repo.

**Also.** `ftp.ncbi.nih.gov` AND `eutils.ncbi.nlm.nih.gov` both probed: CONNECT →
403 (**seventh** session; eutils is newly confirmed blocked too, so the HTTPS
workaround is dead). Container again came up on a **detached HEAD**, and local
`main` was **29 commits behind** it — checking out `main` silently reverted the
tree. Fast-forwarded before any work; this is the third session to hit it.

**Highest-value next step: re-run `silent_edge_mentions.py` on a machine that has
`MAIN_DATA.json`.** It closes the 144 unscoreable edges and the 795 unscoreable
observations (25.8%) with no GPU, no taxdump and no new papers — the cheapest
open lever in the project. Below that, unchanged: more papers (needs a GPU, ask
first) and two human design calls. Write-up: `FINDINGS_mention_audit.md`.

---

## TL;DR — 2026-09-12 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Tested.** Whether the extractor actually read each paper correctly, measured
against the papers' OWN sentences — depending on neither the compromised
in-house gold nor the half-independent curated databases. And whether an
observation's textual provenance (own result vs background/citation) predicts
disagreement with the rest of the literature.

**Survived.**
- *Reading fidelity is **>= 86.6%** (181/209), 95% CI [81.7, 91.3]*, paper-cluster
  bootstrap over 122 papers, disagreement NOT clustered by paper (p=0.28). This
  is the project's **fourth gold-free fidelity signal** and the first that covers
  the whole graph rather than a subset. It corroborates the ~90% reading-fidelity
  half of `FINDINGS_independence.md` by an independent route. It does NOT move
  73.0/72.5 and must not be quoted as doing so — different quantity.
- *All 28 residual disagreements are audit artefacts, not extraction errors.*
  Adjudicated twice independently (orchestrator wrote its verdicts before reading
  the second pass): **25/28 exact category agreement, 0 and 0 extraction errors**.
  12 are sentences attributing abundance to the CONTROL group (so the graph's
  "depleted" is right and the cue is the inverted one), 9 compare something other
  than disease-vs-control (treatment arms, timepoints, symptom subgroups, a
  regression on a continuous score), 4 carry a direction belonging to a different
  taxon, 2 are background leaks, 1 is truncated. So 86.6% is a **lower bound**.
- *One letter was deciding which genus a paper meant.* `relation_sentences.py`
  expanded abbreviated binomials via `alias.setdefault(sci[0].upper(), sci)` —
  first genus with that initial wins. An oral/gut Alzheimer's paper filed
  *P. gingivalis* (**Porphyromonas**, the periodontal pathogen it is largely
  about) under **Phascolarctobacterium**, a gut genus. 774 mentions came from
  that path, 362 in papers with a letter clash, 78 of 348 papers. Fixed at source
  and repaired in place — 84 reassigned, 29 already right, 117 dropped.

**Did not survive / null.**
- *Textual provenance does not predict discordance.* own 27.6% (285/1032),
  background 27.1% (76/281), silent 28.6%. Pooled difference −0.6 points at an
  **MDE of 8.4 points**; paired within paper +5.5 points, p=0.28, MDE 14.1.
  **The 25th variable tested against discordance, the 25th null.** Good news for
  the graph — the 585 background-only observations are not worse evidence — and
  the screen is spent as a lever.
- *A comparison-frame corrector was built, measured and REJECTED.* Flipping
  control-framed sentences moved agreement 0.866 -> 0.774, because the dominant
  construction names controls as the REFERENCE ("lower in PD patients compared
  with the healthy controls"), not the subject. **41 of 54** control-framed
  witnesses were already correct unflipped. Do not rebuild it.
- *My own curated table was briefly weaker than what it audited*, for the third
  time in this repo. A first draft keyed on the epithet alone would have
  rewritten the correctly species-resolved "L. salivarius" (*Ligilactobacillus*)
  to *Streptococcus* and "R. hominis" (*Roseburia*) to *Dialister*. Caught in
  spot-check; two guards added (never override a species-rank resolution; the
  candidate genus must share the mention's initial).

**Also.** `ftp.ncbi.nih.gov` re-probed: still CONNECT -> 403 (**sixth** session).
Container again came up on a detached HEAD; `main` re-attached and pushed.

**A structural lesson worth keeping.** Re-running `cooccur_direction.py` on the
repaired corpus reproduced every statistic **bit-for-bit**, and that was checked
rather than assumed: **zero of 348 papers change their taxid SET**. Reassignment
*requires* the true genus to be named in full in the same paper, and the wrongly
assigned genus was too — that is how it entered the alias map. The bug moved
mentions between taxa the paper names anyway, so a binary incidence profile
cannot see it. *A representation immune to a class of error is also blind to it.*
The graph itself was never affected: `build_kg.py` does not read this file.

**Sizing the evidence base while there.** 1,561 of 2,008 edges (77.7%) rest on a
single paper; of those, 981 have an own-result sentence witness, 279 only
background, 301 none at all. So **580 edges (28.9%) rest on one paper AND have no
own-result prose witness** — all 580 already tagged `provisional`, which is the
correct place for them. `silent` is not `unsupported`: a study listing twenty
taxa in a table names most of them in no sentence, and the filter keeps 7.5k of
106k sentences by design. Verifying these needs more papers or table/figure
extraction; prose filtering cannot reach them.

**Single highest-value next step is unchanged: more papers, which needs a GPU —
ask before spending.** Everything below it is two human design decisions and
reporting PMID 27703453 upstream to Disbiome. Write-up:
`FINDINGS_direction_audit.md`.

---

## TL;DR — 2026-09-11

**Tested.** Whether the 254 "unresolved 16S clade labels" really are all clade
labels; whether the extractor or the papers own the misspellings in them;
whether any animal study is in a graph that claims to be human case-control.

**Survived.**
- *The unresolved residue is not all clade labels.* 12 concepts were split
  across two nodes by punctuation alone, and 33 labels are misspellings.
  41 taxa merged, 925 -> 883.
- *The misspellings are the PAPERS', not the extractor's* — 33/33 occur verbatim
  in their own source paper's full text. A third independent fidelity signal,
  depending on neither the compromised in-house gold nor the half-independent
  curations.
- *A screened-out rat FMT study was in the graph*, contributing 6 Alzheimer's
  edges, because `filter_maindata.py` normalised titles more weakly than the
  deduper does and the paper existed under two spellings.
- *Four edges were unanimous only because of a spelling.* Bifidobacterium/PD,
  Butyricicoccus/MS, Clostridia_UCG-014/PD, Verrucomicrobiota/AD are now
  correctly contested.

**Did not survive / null.**
- *No further animal study is in the corpus.* An animal prefilter validated at
  **recall 15/15** against the existing 45-paper screen flags 13 of 271 papers;
  all 13 read out as genuine human case-control. Rule-of-three bound: miss rate
  <= 20% at 95%.
- *Agreement moved by nothing, for the sixth time.* Disbiome 73.0 -> 73.4%,
  Peryton 72.5 -> 72.5%, both under the ~0.013 this corpus resolves and in
  OPPOSITE directions. Recall is the honest gain: +1 pair against each database.
- *A review-language filter* was built and discarded before use — 87 papers
  flagged, almost all good studies citing a meta-analysis.
- *Two of my own instruments were weaker than what they audited*: a regex that
  demoted `Azospirillum sp. 47-25` from a real taxid (caught by the rebuild
  diff), and a verbatim-quote check that failed 9 of 13 correct adjudications
  because the agents paraphrased.

**Also.** The graph passes a full internal-consistency audit (edge arithmetic,
hierarchy acyclicity, paper table) with zero defects. `ftp.ncbi.nih.gov`
re-probed, still CONNECT -> 403 (fifth session). The container again came up on
a detached HEAD; `main` re-attached and pushed.

**The 248-paper screen was then DONE in-session, and the method failed its own
validation.** All 249 never-screened papers went through an abstract-level
screen with 24 blinded controls: **19/24 exact, 20/24 keep-vs-drop**, false
drops 2/12 and missed drops 2/12 — and it marked the Part-1 rat-FMT paper KEEP,
the single case the work exists to catch. It returned 231 KEEP, 14 UNCLEAR and
4 proposed drops, all 4 in the one category both false positives landed in.
**All 4 were read against full text and none survived**: two are flatly wrong
(one names "147 controls"), two are arguable. **The graph was NOT modified.**

Corroboration worth keeping: **zero animal drops among the 249**, which agrees
with the deterministic full-text sweep by an independent method.

**That 18-paper worklist was then done in-session too: 18/18 KEEP, zero drops,
graph unchanged.** Every one names an explicit control group ("64 patients with
ICH, 46 coronary heart disease controls, and 23 healthy controls"). So the
never-screened half of the corpus shows no evidence of any of the four failure
modes. Residual, stated honestly: 231 papers were called KEEP from an abstract
and never re-read, against a measured 2/12 missed-drop rate on harder controls
— the count of remaining bad papers is consistent with zero, not proven zero.
The abstract screen's value was TRIAGE (249 -> 18 worth a human's time), not
adjudication; it was 0-for-4 on drops and 0-for-14 on unclears.

**Single highest-value next step is now back to: more papers, which needs a
GPU — ask before spending.** The corpus-screening lever is spent. What is left
below it is two human design decisions and reporting one Disbiome record
upstream.

---

# SUMMARY — session of 2026-09-11 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Anomaly-hunted the graph's own structure, which is the method that has now
worked three times. The 254 unresolved taxon labels, carried for three sessions
as "16S clade labels", turned out to contain 12 punctuation-split concepts and
33 misspellings; and following one merge's supporting quote ("AD Tg mice")
surfaced a screened-out rat study in the graph and the fact that 248 of 271
papers were never screened at all.** Write-ups:
`FINDINGS_taxon_spelling.md`, `FINDINGS_corpus_screen.md`.

The scheduled prompt's priority list was stale for the FOURTH session running;
its four priorities were confirmed done and none were redone.

### The spelling sweep

`taxon_typos.py` re-derives the candidates, holds 33 curated folds and **13
recorded refusals**, and verifies every fold against source full text.

Curated rather than edit-distance on purpose: `Oscillospirales`/`Oscillospira`
and `Thermoactinomycetales`/`Thermoactinomycetaceae` are distinct real taxa, and
`Prevotella_9`, `Ruminococcus_1`, `Coprococcus_2` are ~1 edit from their parent
genus and **deliberately held apart** — an edit-distance rule would have
silently undone the placeholder split. `Corynebacteria` (the paper writes "class
Corynebacteria", and no such class exists), bare `UCG-002`, and `lactic acid
bacteria` (a physiological guild, not a taxon) are refused as undecidable from
the text; the last is flagged as a node that probably should not exist.

Root cause of the 12 splits: the placeholder branch of `norm_taxon` returns
before reaching the separator collapse added on 2026-09-08, so it still keyed on
`_`->space alone. Two of the twelve are **one paper reporting one taxon under
two spellings in two figures**, i.e. a single study casting two votes on an edge.

### The screening leak, and the gap behind it

`filter_maindata.norm()` kept trailing punctuation where `dedup_rows` strips
every non-alphanumeric; 13 papers exist under two such spellings. A `DROP_ANIMAL`
paper was dropped under one spelling and survived under the other, and the
deduper never saw the pair because the filter had already removed one copy. The
guarding `assert seen == 45` counted screen entries matched rather than copies
dropped, so it passed throughout.

Behind it: **only 23 of 271 contributing papers have ever been screened.** The
animal prefilter built to test this was validated at recall 15/15 / precision
0.882 on the existing gold set BEFORE use, per the standing rule, and returns a
null on the remaining corpus — 13 flagged, 13 genuine human case-control.

### Two instrument failures worth remembering

**My regex was looser than the concept.** Allowing a hyphen before a trailing
number swallowed strain designations; `Azospirillum sp. 47-25` and
`Lachnospiraceae bacterium MC-35` were demoted from real taxids to unresolved
placeholders. Visible only as `n_taxa_resolved` falling by 2 in the rebuild diff.

**The build is self-referential.** `taxonomy_cache.py` replays `graph.json`'s own
resolution and `build_kg.py` overwrites `graph.json`, so the bad intermediate
became the authority for the next build and silently changed two display labels.
Any future correction must rebuild from a known-good graph, never from the
output of a failed attempt. This hazard is general and will recur.

**The verbatim-quote check on subagent output cried wolf.** 9 of 13 adjudications
failed it; all 9 were correct and merely paraphrased. Worth running — it would
catch a fabrication — but "quote not found" is not evidence of a wrong verdict.

### Verified by executing

Three consecutive rebuilds byte-identical (a fixed point); `kg.html`
byte-identical on rebuild; `docs/index.html` re-synced; `verify_viz.py` 32/32 in
Chromium; `taxon_typos.py --verify` 33/33; full internal-consistency audit of
`graph.json` clean (edge arithmetic, hierarchy acyclicity, no orphan papers).

### The 249-paper screen, and a bug of mine inside it

`screen_corpus.py --prepare/--score` is reproducible (fixed seed, byte-identical
batches on re-run, verified). The blinded controls are the entire validation,
which is why they exist: an unvalidated 249-paper classification is a pile of
opinions. They said do not act, and so the graph was left alone.

**The controls are harder than the population**, and this cuts against the
headline: they are drawn from the 45 unvetted keyword-matched MAIN_DATA papers,
far messier than the datasheet papers being screened. A 0.167 false-positive
rate over 245 KEEP papers predicts ~41 false drops; only 4 drops were proposed
in total. So 0.833 is probably a LOWER bound here — by an unmeasurable amount,
since no representative gold set exists.

**And part of the failure was mine.** The abstract extractor anchored on the
first "Abstract" marker and returned pure front matter — journal navigation,
author lists, affiliations — for **12.9% of the corpus (35 of 271 papers)**,
which is why they came back UNCLEAR. A screening agent diagnosed it unprompted:
"unclear abstracts due to heavy metadata in the source text". Replaced with a
sliding best-scoring window: zero-cue spans 35 -> 5 (1.8%). The run reported
above used the OLD extractor, so its 14 UNCLEAR are inflated. Note this does
NOT explain the false positives — both had perfectly good abstracts. The
extractor fix would reduce UNCLEAR and leave the judgement problem untouched.

### Sized but NOT decided — the disease-subtype question

`Cognitive impairment` and `Mild cognitive impairment` share 18 decisive taxa and
agree on direction for only **9 of them (50%)**, the 13.7th percentile of all 182
disease pairs with >=5 shared taxa (mean 0.670, sd 0.186). `Spinal cord injury` /
`Chronic traumatic complete SCI` sit at 0.600 (29th pct);
`Intracerebral haemorrhage` / `Hypertensive ICH` at 0.800 (70th pct). So the
subtype labels carry **no consistent extra similarity** over unrelated disease
pairs, and merging them would manufacture contested edges rather than resolve
them. This is a number for the standing human decision, not the decision.

---

## TL;DR — 2026-09-10

**Tested.** Whether any of our own papers is systematically inverted (the failure
mode found in Disbiome last session); whether discordance with the literature is a
paper-level property; and 24 candidate explanations for it — 9 study-design, then
15 wet-lab/bioinformatics extracted fresh from full text.

**Survived correction.**
- *No paper is inverted.* 0 of 134 survive BH, and the power is measured: a fully
  inverted copy would have been caught for 81 of them. Best extraction-fidelity
  statement in the repo that depends on neither compromised reference.
- *Discordance is paper-level* — minority-direction labels are not exchangeable
  across papers, p = 0.0003, robust to containment thinning and to dropping
  two-paper edges.

**Did not survive.**
- *Three predictors that would have shipped* — "Parkinson's papers are more
  reliable" (q = 0.0005) and two cohort-size effects — all edge-depth artifacts,
  killed by using the exact within-edge expectation as an offset.
- *All 24 explanations.* Best q = 0.234 and 0.61 across the two passes.
- *The paper-level effect's practical size.* 3.4 percentage points of discordance,
  cluster-bootstrap CI [0.0, 6.0] **including zero**. That is why the 24 nulls
  were foreordained (MDEs of 4–7 points), and it means ~85% of the variance in
  disagreement is edge structure, not paper identity.
- *Methods diversity as a quality signal.* p = 1.00 on both databases; shipped as
  provenance only.

**Also.** The "needs a GPU" blocker on the methods pass was false — full text for
all 272 papers was in the repo. Ten commits from prior sessions were stranded on a
detached HEAD and are now pushed. Stale published claims corrected in `README.md`
and `CLAUDE.md` (contested count 151/174 → 217; "taxa are unresolved strings";
"not yet validated against Disbiome/Peryton").

**Single highest-value next step: more papers, and it needs a GPU — ask before
spending.** Every remaining question is n-limited, and this session closed the
main alternative (paper-level covariates) rather than leaving it open.

---

# SUMMARY — session of 2026-09-10 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Asked of our own extractions the question that caught Disbiome last session —
is any single paper systematically inverted? — and got a well-powered no. Then
found paper-level structure in contested edges, sized it, and found it too small
to chase: 3.4 percentage points, CI including zero.** The 24 explanatory variables
tested against it are all null *because of* that size, not independently of it.
Write-up: `FINDINGS_paper_discordance.md`.

The scheduled prompt's priority list was stale for the third session running:
its four priorities (MAIN_DATA filter, Task 1, Task 2.5, Task 3.1) are all done
per the entries below and none were redone. `ftp.ncbi.nih.gov` re-probed once,
still `CONNECT → 403`. Ten commits from the previous two sessions were sitting
on a detached HEAD; `main` was re-pointed at them and confirmed pushed.

### What was tested

Every paper's vote on every multi-paper edge, scored against the leave-one-out
majority of the other papers: 440 edges, 1,493 observations, **1,367 decisive, 377
(27.6%) disagreeing**. Null shuffles directions WITHIN each edge, so every edge
keeps its up/down counts and every paper keeps its exact edge set.

### Survived

**Discordance is a paper-level property.** Minority-direction status is clustered
by paper well beyond the null: dispersion 164.2 vs 132.7, **p = 0.0003**. Not the
known within-paper taxonomic correlation — dropping every within-paper relative
across the 723 containment links (28.4% of observations) leaves p = 0.0013 — and
not the structurally forced two-paper edges, where both papers are scored as
disagreeing under every permutation: restricting to ≥3-paper edges leaves
p = 0.0003. Both tails move.

Both controls at once give p = 0.058, and that is **power, not refutation**:
subsampling the containment-controlled set to the strict set's 637 decisive
observations 200 times, the test reaches p<0.05 only **58%** of the time, median
p 0.035. This is the first explanatory structure found in contested edges after
four edge-level nulls.

### Did not survive — and one of them would have shipped

**No paper is inverted.** 0 of 134 testable survive BH (best q = 0.41), and the
power is measured: a fully inverted copy would have been caught for **100/134**
at raw p<0.05 and **81/134** Bonferroni-strict. So the extractor does not have
Disbiome's PMID 27703453 failure mode — a fidelity statement that needs neither
the in-house gold (under audit) nor the external curations (half independent).

**No study-design variable explains the paper-level variance.** Tested naively on
raw disagreement rate, three predictors survived BH: Parkinson's (q=0.0005),
n_cases (q=0.0010), cohort total (q=0.0043). **All three are edge-depth artifacts**
— a 2-paper contested edge scores both papers as disagreeing (rate 1.00) while a
10-paper 8/2 edge scores only two (rate 0.20), and Parkinson's is the
most-reported disease while large cohorts study well-studied diseases.

Replacing the rate with the **exact closed-form within-edge expectation**
(`P(disagree) = 1 if n_e==n_d else min(n_e,n_d)/n`, verified against 4,000
simulated permutations, global O/E = 0.995) makes all nine metadata predictors
null, best **q = 0.234**, MDEs ±0.16–0.22 in O/E. The planted control confirms it:
edge depth, which drove all three false positives, goes flat at **p = 0.90**.

**Fifth documented false positive in this corpus, and the first caught by an
offset rather than a permutation.** The raw-rate analysis would have published
"Parkinson's papers are more reliable" at q = 0.0005.

### Disqualified on construction, recorded so it is not rediscovered

`sits_on_contested_edges` is the lone BH survivor of the offset analysis
(p = 0.0029) and is endogenous: an edge is contested *because* its papers
disagreed. Recomputing contestedness leave-one-out **reverses the sign**
(+0.225 → −0.154, and −0.733 with ties dropped, p = 0.0004). Neither construction
is interpretable. Contestedness cannot predict discordance.

### The methods metadata pass — done in the same session, and null

The step above was written mid-session as the next lever, then done, so it is
recorded as closed rather than pending. **It did not need a GPU and it never
did.** Full text for all 272 contributing papers was already in the repo across
`all_usable_papers.json` (250), `extract_input.json` (98) and `new_papers.json`
(53) — union 272/272 — and the variables of interest are tool names, which a
regex reads deterministically where an LLM would not give the same answer twice.
`methods_metadata.py` extracts nine families; the first run scoped 272/272 papers
to the full body because the cleaned texts have no line breaks and headings sit
inline, so an anchored `^heading$` could never fire. Fixed, 155/272 now resolve to
a real Methods section and assay agreement rose 61.4% → 74.8%.

**Detector validated before use**, as the rule requires: 74.8% on 16S-vs-shotgun
and 80.8% on 16S region against the existing LLM labels; against an independent
read of 12 sampled methods sections, **recall 0.90 / precision 0.77 overall** and
**1.00 / 0.83** on the named tools the analysis keys on. Publication year, parsed
from the header, is exact for 21 of 22 checkable papers.

Fifteen predictors — LEfSe, DESeq2/ANCOM, nonparametric-only, multiple-testing
correction, ASV vs OTU, QIAamp, bead-beating kits, QIIME2/DADA2, legacy pipelines,
MiSeq, rarefaction, CLR/absolute quantification, publication year, cohort
imbalance, breadth of reporting — plus two planted controls. **No survivors**;
best raw p = 0.042 (bead-beating), q = 0.61. Both controls behave (p = 0.90, 0.94).
**24 variables tested across two passes, 24 nulls.**

### And the reason they were all going to be null

`paper_effect_size.py` puts a magnitude on the thing being explained instead of
testing a twenty-fifth variable. Excess variance over the within-edge null is 9.7
of 66.6 (17%), giving σ = 0.123 — a paper-level SD of **3.4 percentage points** of
discordance on a 27.6% base, cluster-bootstrap 95% CI **[0.0, 6.0], including
zero**, with 15% of resamples showing no excess at all.

**This cuts against the framing above and is the honest headline.** The
permutation test and the magnitude estimate answer different questions and both
are right: labels are not exchangeable across papers (p = 0.0003), and the spread
that produces is small. Consequences:

- **The 24 nulls were foreordained.** MDEs of ±4 to ±7 points against a total
  spread of ±3.4. The defensible claim is "this corpus cannot answer whether kit
  or pipeline drives disagreement", NOT "they do not".
- **~85% of the variance in disagreement is edge structure, not paper identity.**
  Papers are close to interchangeable; disagreement lives in the taxon–disease
  pairs. That is quantitative support for the standing decision to keep contested
  edges rather than average them — and an argument against any further
  paper-level covariate hunt.

### Shipped: the viewer now tells a reader what protocol the evidence used

The methods variables would have been another JSON file nothing reads, so they are
wired through. `build_kg.py` joins them onto the paper table **on the normalised
title, deliberately not on the `paper` index the records also carry** — that index
points into whatever paper table existed when the file was written and goes stale
silently the moment the corpus changes. Every edge gains `n_methods_papers`,
`n_pipelines`, `n_kits`, `n_platforms` and a `methods_diversity` label:
**313 multi-method, 116 single-method, 1,605 unknown** (single-paper edges cannot
have one by construction). The detail panel gains a protocol note and the study
table a Protocol column ("DADA2 · FastDNA", "QIIME1/UPARSE · Omega").

**And the panel says out loud that this is provenance, not quality.**
`methods_diversity_calibration.py` asked whether multi-method edges agree with
curated databases more often, as `annotate_confidence` had to earn its tiers:
86.0% vs 87.5% (Disbiome), 89.5% vs 87.5% (Peryton), taxon-block permutation
**p = 1.00 both**. Stratifying on evidence count is mandatory — multi-method edges
average 4.81 papers against 2.44 — and leaves cells of 4–12 pairs with nothing
significant. The structural reason is worth keeping: **there are zero
single-method edges with ≥5 papers**, so diversity and evidence count are
near-collinear exactly where the evidence is strong. It must not become a
confidence tier.

Verified by executing: zero drift in every pre-existing field (meta identical, 965
nodes, 723 hierarchy links, all 272 paper rows and 2,034 edges unchanged bar the
new keys), two rebuilds byte-identical, rebuild from the installed graph a fixed
point, `kg.html` byte-identical on rebuild, `docs/index.html` re-synced.
`verify_viz.py` **26 → 32 assertions**, all passing in Chromium.

### Highest-value next step

**More papers, and now the alternative is closed rather than merely unexplored.**
Every recent session named the methods metadata pass as the cheap unblocked lever;
it has been done and it is null, for a reason that is arithmetic rather than
contingent. 109 papers with ≥4 decisive observations is what sets every MDE here.
Extraction needs a GPU — **ask before spending.**

~~One cheap thing worth doing first~~ — DONE in this session, see "Shipped" above: `methods_metadata.py` has nine families of
study-methods variables for all 272 papers at 0.90 recall and **nothing in the
graph consumes them**. They are weak predictors of discordance — that was tested —
but they are good *provenance* for a reader judging an edge ("these 6 papers used
the same kit and pipeline"), which is what Task 3.4 ("ship something a biologist
would use") has always wanted.

---

# SUMMARY — session of 2026-09-09 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Checked the assumption the whole project rests on and it does not hold: the
"independent" external validation is only half independent, and the two halves
measure different things.** Write-up: `FINDINGS_independence.md`.

The scheduled prompt's priority list was stale again — its four priorities (the
MAIN_DATA filter, Task 1, Task 2.5, Task 3.1) are all done per the entries below
and none were redone. `ftp.ncbi.nih.gov` was re-probed once and is still shut
(CONNECT → 403, confirmed in the proxy's own failure log), which no longer blocks
anything: the species split landed without it.

### The question nobody had asked

Disbiome 73.0% and Peryton 72.5% carry the weight they do *because those
curations are independent of this pipeline* — the repo says so, and prefers them
to the in-house gold for exactly that reason. Nobody had checked. Both databases
curate the primary literature; so do we. It is checkable, because our rows carry
PubMed links and both databases ship PMIDs, DOIs and titles.

**43 of our 272 papers are also cited by Disbiome (16%), 24 by Peryton (9%) — and
because the shared ones are the heavily-reported papers, they back HALF the
decisive pairs** (50.6% / 44.9%). Agreement splits hard on that line:

| | shared source | disjoint source |
|---|---|---|
| Disbiome | **87.5%** (n=88) | **58.1%** (n=86) |
| Peryton | **96.8%** (n=62) | **52.6%** (n=76) |

+29.4 / +44.1 pts, taxon-block permutation p=0.0001, taxon cluster-bootstrap CIs
excluding zero, and it survives stratifying on evidence count — so it is not the
evidence-count signal in disguise.

**Disease is a confounder and pooling overstates it.** Every ALS pair here is
shared-source and every autism pair disjoint, so the pooled gap partly measures
"ALS vs autism". Held fixed within Parkinson's — the only disease with both
buckets full — the effect is smaller, still large, and **two databases that know
nothing about each other land on the same disjoint rate**: Disbiome 100.0% (n=40)
vs **59.0%** (n=39); Peryton 95.8% (n=48) vs **59.6%** (n=47); p=0.0001 each.

### The reframe, and the good news inside it

Crossing source-sharing with evidence count (Disbiome / Peryton):

| | 1 paper | ≥2 papers |
|---|---|---|
| shared source | 85.2% / 94.1% | 91.2% / 100% |
| disjoint source | **47.4% / 38.3%** | **79.3% / 75.9%** |

So **73% measures neither quantity** — it is a blend of ~90% *reading fidelity*
and ~55% *cross-literature reproducibility*, mixed in a ratio set by how much of
our corpus the curators happened to read. The top-left corner is the good news
and is worth more than the headline: a single-paper edge whose one paper **is**
the curated source agrees **85–94%**, which is the cleanest measurement of the
extractor this project has and the only one that does not depend on the in-house
gold standard that is under audit.

The disjoint number is **not** an extraction-accuracy figure and must not be
reported as one: this literature genuinely disagrees with itself (217 contested
edges; ~1 taxon in 3 flips sign between cohorts), so ~59% may be near the ceiling
the field sets.

### The counter-example, kept in the text

**In Multiple sclerosis the gap is absent** — 72.7% (n=22) vs 70.6% (n=17),
p=1.00, MDE 28.7 pts. That test could have seen a Parkinson's-sized gap and did
not. The finding is established *in Parkinson's*, not corpus-wide.

### What predicts agreement (24 tests, BH-corrected, block-permuted)

Survives: **evidence count** (1 paper 65.8/61.7 → ≥3 papers **91.7/90.6**;
+23.8/+23.7 pts, near-identical in two databases, both q<0.05 — edge weight IS a
calibration signal); **disease specificity** (`discriminating` taxa agree 36.4%
vs 82.4% `mixed`, Disbiome q=0.0024, and it is not evidence count in disguise —
+45.3 pts within single-paper edges alone); **species rank** (93.2% vs 66.7%
genus, q=0.0038 — and it cannot be an evidence artefact, since species edges
carry *fewer* papers, 1.61 vs 2.26, and agree *more*).

Nulls with power: **the reference's own evidence depth does NOT predict
agreement** (Disbiome +6.2 pts, CI [−16.5, +25.4], p=0.49) — a hypothesis this
session proposed, that the 27% disagreement was mostly thin single-record curated
entries, and the data killed it. **`restates_prior` does not predict agreement**
(+0.9/+7.7 pts). Within-paper rank conflict is undetermined at n=9 and the two
databases point opposite ways.

### Shipped: the viewer now says which edges to trust

`annotate_confidence()` tiers every edge from its own properties only — no
external data — so it covers all 2,034 edges including the ~1,800 no curation
judges: **contested 217 (10.7%), provisional 1,607 (79.0%), supported 135 (6.6%),
well-supported 75 (3.7%)**. Each tier carries its measured rate, and the detail
panel quotes it rather than asserting quality. The `discriminating` demotion
earns its place empirically (without it well-supported is 91.7/90.6, with it
93.8/93.3; it moves four pairs and all four were wrong).

**The number the tiles now show is the sobering one: 79% of this graph is
provisional**, a tier that agrees ~62–66% — and, on disjoint literature, 38–47%.
That is the argument for more papers, quantified rather than asserted.

Verified by executing, not by parsing: rebuild gives **zero drift in every
pre-existing field** (meta, 965 nodes, 723 hierarchy links, 272 papers, all 2,034
edges identical bar the new key), two rebuilds byte-identical, `kg.html` likewise,
`docs/index.html` re-synced. `verify_viz.py` 19 → **26 assertions**, all passing
in Chromium. One new assertion failed first time and was right to: it asserted
tier diversity in the confidence-sorted view, where 75 well-supported edges fill a
60-row chart, so correct code shows exactly one tier.

Also fixed a latent trap: `load_disbiome` emitted Disbiome's own row id under the
key `"pmid"`. Nothing read it yet — which is exactly when to fix it.

### Corrected published claims

`CLAUDE.md` and `kg/README.md` both said the graph "agrees with two **independent**
hand-curated databases". Both now carry the decomposition. Same class of fix as
Bug 5 (the stale *Lachnospiraceae* "15 papers"): a published claim that no longer
matched the artifact.

### Then: five contradictions turned out to be one mis-curated paper

Write-up: `FINDINGS_db_conflicts.md`. Where Disbiome and Peryton contradict *each
other*, one is wrong by construction — a stronger signal than either disagreeing
with us. There were 5, all in ALS. Asking the structural question before reading
anything: **all ten records come from ONE paper (PMID 27703453)**, so it is a
single curation error, not five contradictions; calling it five would overstate
the evidence fivefold, the mistake the "229 opposite pairs" figure made.

We extracted that paper too, so `relation_sentences.json` settles it verbatim,
and the decisive sentence needs no decoding of the paper's group labels:
"significant increased genus *Dorea* ... and significant reduced genus
*Oscillibacter*, *Anaerostipes*, *Lachnospiraceae* ... **in ALS patients**".
**Peryton and this graph are right on all five; Disbiome is inverted on all
five** — and systematically, since *Dorea* flips with the rest, which is what a
swapped group assignment produces.

**5 of our 47 Disbiome disagreements (10.6%) trace to this one paper.** Correcting
them gives 132/174 = **75.9%** — offered as an error in the REFERENCE, not an
accuracy gain for the graph, and not to be quoted unlabelled. It is also the first
demonstrated case that some of the 27% disagreement is the reference being wrong
rather than us. Worth reporting upstream to Disbiome.

### Highest-value next step

**More papers, and now the case is quantified rather than asserted**: 79% of the
graph sits in a tier that is a coin flip against disjoint literature, while the
≥2-paper tier reaches 76–79% on genuinely independent sources. Moving edges from
the first bucket to the second is the entire remaining lever, and it needs a GPU
(ask before spending). The 5 mutually-contradicted pairs, listed here earlier as the
cheapest unblocked item, were **done in this session** (above). Next cheapest:
re-run `adjudicate_db_conflicts.py` whenever the corpus grows — any cluster of
disagreements tracing to one publication is a candidate curation error rather
than N findings — and report PMID 27703453 upstream to Disbiome.

---

# SUMMARY — session of 2026-09-08 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Fixed the top open defect — the species folding into their genus — and found
that all three things the docs said about it were wrong.** Write-up:
`FINDINGS_species_split.md`.

The scheduled prompt's priority list was stale again (Tasks 1, 2.5, 3.1 and the
MAIN_DATA filter are all done per the entries below); none were redone. Work
went to the one item every recent session has recorded as blocked.

### It was not blocked on the taxdump

`ftp.ncbi.nih.gov` is still shut here (CONNECT → 403, re-probed along with
`ftp.ncbi.nlm.nih.gov`, eutils, `api.ncbi.nlm.nih.gov`, Ensembl, EBI). It did not
matter. **An NCBI taxid is stable across a rename**, so it is a join key renaming
cannot move: Disbiome — already committed in this repo — was curated before the
2024-25 reclassifications and holds `"Prevotella copri" → 165179`, and
`ncbi-taxon-db` (the NCBI taxonomy redistributed on PyPI, behind `taxoniq`;
PyPI is reachable) resolves `165179 → Segatella copri [species]`. Two sources
that know nothing about each other. Every entry in the new
`species_synonyms.json` records which route produced it and the evidence string.

Caveat for the next session: **`ncbi-taxon-db` is NOT a general taxdump
substitute** — current scientific names only, no synonym table. Every genuinely-
species string here failed a direct lookup on it. It serves the *taxid* side of
the join and only that.

### It was 24 species, not 54, and the other 91 folds must NOT be split

A mechanical split of the 54 `named_child` strings would have damaged the graph.
`Escherichia / Shigella` and `Streptococcus salivarius/thermophilus` name two
taxa each (a 16S assay that cannot separate two genera has measured neither);
`Clostridium_XlVa`, `Prevotella VZCB`, `Turicibacter sp001543345` are pipeline
cluster labels; `Neisseria multispecies` names no organism; several are strain
codes under a parent that is *already* a species; two are phages. The resolution
ladder in `species_synonyms.py` IS the classifier — those fail it and stay put.
Nothing was sorted by hand.

### The cause was not a missing taxdump either — and three organisms were duplicated

The shipped graph resolves renamed binomials fine (`Clostridium aldenense` is a
node labelled *Enterocloster aldenensis*). What failed is a subset of the 2024-25
renames whose old binomials the lookup does not return, so `resolve()` fell
through to its qualifier-tail trim, **threw the epithet away**, and landed the
mention on the genus. **Three of these organisms were ALREADY nodes under their
new names**: papers writing "Phocaeicola dorei" built a species node, papers
writing "Bacteroides dorei" were folded into genus *Bacteroides*. One organism,
two nodes, two ranks. Found by asking a cheap structural question of the graph,
not by inspection — the method that keeps working here.

### Result, and the honest size of it

918 → **929** taxa, 2,011 → **2,043** edges, 708 → **719** containment links,
219 → **215** contested, 660 → **671** resolved.

The headline case is ***Eubacterium*, not *Prevotella copri***: *Eubacterium* /
Parkinson's goes **4up/5dn (contested) → 2up/1dn**, because that contest was two
species pulling opposite ways inside one genus node — *E. rectale* (now
*Agathobacter rectalis*, Lachnospiraceae) depleted, *E. biforme* (now
*Holdemanella biformis*, Erysipelotrichaceae) enriched. NCBI places neither in
*Eubacterium*, or even in the same family. **NULL on the flagship:** *Prevotella*
/ Parkinson's goes 3up/14dn → 2up/11dn — direction unchanged, still contested;
*P. copri* agrees with its genus (5 of 6 depleted) and neither carries nor flips
that edge.

External validation, reported as counts because the ratio is at the edge of what
this corpus resolves: Disbiome overlap **260 → 269** (recall 51.2% → 53.0%),
Peryton **220 → 224** (72.6% → 73.9%). **11 decisive Disbiome pairs entered, all
11 agree, 0 disagreements added, 0 verdicts flipped**; Peryton +1, agreeing.
Ratios moved 71.7% → 73.1% and 72.5% → 72.7%. Eleven-for-eleven is p = 0.026
against the 0.717 baseline, **but those 11 pairs come from only 7 distinct taxa,
and clustering on taxon gives p = 0.097 — suggestive, not significant.** The
defensible claim is coverage (+9 net decisive pairs, no new disagreement), not
accuracy. Consistent with the standing rule: this correction is justified on
correctness of meaning, and must not be cited as an accuracy gain.

One real cost, logged rather than hidden: *F. prausnitzii* / MS went 0up/7dn →
1up/7dn (became contested, left the decisive set) because the fuzzy route folded
in the misspelling `Faecalibacterium prauznitzii`, from a paper reporting
enrichment. Correct behaviour; de-contesting is not the objective.

### The blocking fear was real, and is handled

The warning that splitting these in the cloud "would have LOST the
Disbiome/Peryton join" was right, and the mechanism was **ancestry**:
`build_kg.py` builds containment by walking `tax.lineage()`, and the replay cache
holds only graph-local links, so a split species would have shipped **detached**.
`species_synonyms.json` therefore stores the full NCBI lineage per entry. One
candidate (*Lawsonibacter phoceensis*, absent from the 2024 snapshot) is
**refused rather than split detached**. Note the right ancestor is usually not the
old genus: *Segatella copri* links to **Prevotellaceae**.

Subtle bit in the diff: the supplement is consulted AFTER `names.dmp` in
`taxonomy.py` (NCBI must win) but BEFORE the cached lookup in
`taxonomy_cache.py`, where the "authority" is a replay of `graph.json` and these
24 entries are exactly what it got wrong. Checked cache-first it is dead code —
which it was for one run, caught by executing the resolver.

Verified: rebuilt twice, **byte-identical fixed point**; `verify_viz.py` 19/19
in Chromium; `docs/index.html` re-synced.

### Then: punctuation was fragmenting concepts across nodes (second fix, same day)

Trying to size that "~20 ambiguous labels" claim instead of estimating it turned
up a smaller number and a **bigger defect underneath**. Asking which concepts land
on more than one node returns **17**. The worst: *Escherichia-Shigella*, the
standard SILVA label for two genera 16S cannot separate, written seven ways and
filed under **four different nodes** — `Escherichia-Shigella` (11 mentions),
`Escherichia/Shigella` (5), `Escherichia–Shigella` (2, EN DASH), and **8 mentions
folded into *Escherichia* itself**. Two bugs at once: the concept fragments by
punctuation, and inconsistently, because when the separator happens to be a space
or underscore `resolve()` trims the trailing token as though it were a qualifier.
A signal from an assay that cannot tell two genera apart was recorded as evidence
about one of them, depending on the authors' typography.

Three narrow rules, each measured before shipping: **refuse the trim when the
discarded token is itself a taxon name** (2 strings change out of 1,090);
**collapse all separator styles in the unresolved key only** (cannot merge
anything NCBI resolved); **strip square brackets as a FALLBACK, never as
pre-normalisation** — brackets are NCBI's own convention, `[Eubacterium] siraeum`
IS a scientific name, and pre-stripping scores 6 gains and 1 loss where the
fallback scores 6 and 0 (it also newly resolves `[Ruminococcus] gnavus group` →
*Mediterraneibacter gnavus*).

929 → **925** taxa, 2,043 → **2,034** edges, 719 → **723** containment, 215 →
**217** contested. *Escherichia* drops from 10 edges / 21 papers to 7 / 13.
**Agreement unchanged** as the standing rule predicts (Disbiome 73.1 → 73.0,
Peryton 72.7 → 72.5, disagreements identical at 47 / 38); the one lost Disbiome
overlap pair is the correction working — we stop crediting *Escherichia* with
evidence the assay could not attribute.

**A regression this nearly shipped:** collapsing separators before the rank
heuristic ("two words means a binomial") silently reranked **~50 unresolved nodes
from genus to species**, because "Escherichia-Shigella" is one written token and
becomes two. Nothing errored. Caught only by diffing the rebuild — reading the
patch would not have found it.

### And a trap that would have reverted the whole species split

`species_synonyms.py` reads `child_folds.json`. Refreshing that file after the
split — the obvious tidy-up — drops the 24 species from it, so the table
regenerates EMPTY and the next `build_kg.py` un-splits them while printing
success. The second-order version fails too: asking the live resolver with the
supplement off also yields nothing, because the split is baked into `graph.json`'s
aliases and the cache is built from those. So `child_folds.json` is now an
explicitly FROZEN input with a guard that refuses to run below 100 rows (verified
by truncating it), and `child_folds.py` — a committed generator at last — answers
the present-tense question into `child_folds_current.json`.

### Nulls from this session

- **The 14 doubly-contradicted pairs are completely unchanged by the split.** Same
  14, no entries, no exits, no flips.
- **Zero remaining duplicate-organism collisions.** Re-running the check that found
  the three (an alias naming an organism that is already another node) returns
  nothing.
- **The flagship edge is unmoved.** *Prevotella*/Parkinson's keeps its direction
  and contested status; *P. copri* agrees with its genus.

### Highest-value next step

**More papers** — the binding constraint on every remaining question is n, not
method, and that needs a GPU (ask before spending). What is left on CPU is a
decision, not an analysis: should a joint two-genus 16S signal be attributed to one
genus, split, or held apart on its own node as it now is? The graph no longer
decides that by accident; a human should decide it on purpose. Same class as
modelling disease subtypes as containment.

---

# SUMMARY — session of 2026-09-06 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**Shipped the specificity layer into the viewer, then consumed the containment
links for the first time and found the project's flagship claim was both true
and misstated.** Write-up: `FINDINGS_rank_conflict.md`.

The scheduled prompt's priority list was stale — Tasks 1, 2.5 and 3.1 and the
MAIN_DATA filter are all already done per the entries below, so none were
redone. The taxdump route was re-probed once and is still shut
(`ftp.ncbi.nih.gov` CONNECT → 403), so the 54 named-species split remains the
top blocked defect. Work went to the two genuinely open items.

### Shipped: the viewer now says what an edge is worth

The 2026-09-05 session put `specificity` / `taxon_breadth` / `taxon_purity` /
`restates_prior` into `graph.json` and deliberately stopped short of the UI.
`build_viz.py` and `viz_network.js` had **zero** references to any of them, so
the published page still showed "*Streptococcus* enriched in Parkinson's, 5
papers" with no way to see it is enriched in eleven other diseases too.

- **Hollow bar = `restates_prior`** (252 edges). Encoded as fill-vs-outline, not
  opacity, so it stays distinct from `.faded` (contested) and survives
  greyscale; label text keeps full contrast.
- **"Sort by: disease specificity"** orders discriminating → mixed → narrow →
  generic. Deliberately **not** `taxon_purity` descending as the last session
  suggested: purity 1.0 *is* the generic case, so that sort surfaces exactly the
  edges the control exists to bury. Narrow outranks generic because a narrow
  taxon is *unjudged* (<3 diseases vote) while a generic one is known
  uninformative.
- "Hide edges that restate a prior" filters **both** views; scope chip per row;
  plain-English specificity sentence in the detail panel; two new table columns;
  a 155-discriminating tile; network links at 0.45 alpha when they restate a
  prior — damped, not hidden.
- Done **without touching `graph.json`**: the per-taxon disease counts live on
  nodes, which the payload does not ship, so `build_viz.py` sends a compact spec
  map keyed by `taxon_key`. The graph and its byte-for-byte fixed point are
  unchanged.
- **`verify_viz.py` is new and is the point.** "It parses" is not verification
  here — two fixes have silently erased themselves on rebuild while printing
  success, and a blank-canvas bug passed every static check. It drives the page
  in Chromium and reads the DOM and canvas pixels back *after real clicks*: 14
  assertions, all passing, including that the canvas is non-blank before and
  after the new filter. `kg.html` rebuilds to a fixed point; `docs/index.html`
  is back in sync.

### Bug 5 — the flagship example's numbers were stale on the published site

`kg.html`, `CLAUDE.md`, `build_kg.py` and `viz_network.js` all claimed
*Lachnospiraceae* is "depleted in Parkinson's across **15 papers**". The graph
says **9 papers, 8 down / 1 up, and the edge is contested**; *Hungatella* is 7
papers, 6 up / 1 down, also contested. The 15 predates the 2026-09-03
deduplication and was never updated — a published number that no longer matched
the artifact it described. All four corrected. Found by checking the anecdote
against the data rather than by looking for it, which is the method that keeps
working here.

### What survived: related taxa agree within a paper, and it is ancestry

**Within a single paper, taxonomically related taxa agree on direction 0.8903 of
the time against 0.5367 for unrelated taxa — gap +0.3536, z=15.5, p=0.0001**
(10,000 permutations, paper-level null shuffling each paper's direction labels
across the taxa it reported, preserving its up/down counts; MDE ±0.044). Both
arms come from the same paper, so cohort, country, pipeline and enrichment
propensity are differenced out by construction.

A z of 15 here is a reason for suspicion, so it was attacked four ways
(`attack_rank_conflict.py`) and survived all of them: dropping every split
placeholder node, since the split *created* both nodes from one mention
(+0.3592, **stronger**); a cluster-robust one-gap-per-paper statistic, since the
top paper contributes 35 of 474 related pairs (+0.3611, z=11.4 over 100 units);
direction skew (0.68, controlled by the shuffle by construction); and
same-rank-unrelated pairs, in case the effect was really "co-mentioned taxa
agree" (+0.3502, z=14.6 — it is ancestry specifically).

**This is face validity, not a discovery.** A family's abundance is largely the
sum of its genera, so ~89% is close to what taxonomy arithmetic predicts; the
extractor passing that check is the result. The value is the complement: the
**11% that disagree inside one paper** are neither noise nor rank confusion, and
they are exactly what collapsing ranks would destroy. The project's refusal to
collapse ranks now rests on a measured 11% rather than one anecdote.

### What got smaller under scrutiny: the "229 opposite pairs" figure

952 parent/child pairs share a disease — 586 same direction, 241 opposite, 125
exact ties. Of the 241 opposite pairs, only **33 (14%) are asserted inside a
single study**; **189 (78%) rest on no shared paper at all**, the family
measured by one set of studies and the genus by another. The GraphRAG session's
"903 edges have a parent edge, 229 point the opposite way" is real but reads as
a much stronger claim than it supports. Cite 14%, not 25%, and say what it
means. The 33 within-paper conflicts — *Lachnospiraceae*↓/*Hungatella*↑ in
Parkinson's among them — are the concrete case for the containment layer and the
best review targets after the doubly-contradicted 11. Caveat: 32 of the 33 rest
on exactly one shared paper, so each individual pair is a single-study claim
even though the aggregate is not.

### Bug 6 — `python3 build_kg.py` silently rebuilt a three-revisions-old graph

Found by following the project's own "rebuild twice and diff" rule. `DEFAULT_IN`
still pointed at the raw 250-paper extraction
(`eval-v2/results/qwopus3.5-27b-v3__q4km__samgated-v1__all250.json`), which is
**not even present in a fresh clone**, while the shipped graph has been built
from `extractions_screened.json` since the paper screen landed — as
`graph.json`'s own `meta.source` has recorded the whole time. So running
`build_kg.py` with no arguments **overwrote `graph.json` with a 773-taxon /
1,462-edge / 211-paper graph** against the shipped 918 / 2,011 / 272, and
printed a normal success summary while doing it.

This is worse than the two fixes that previously erased themselves on rebuild,
because the verification ritual adopted to catch *those* is "run this command
twice" — the rule told you to run the thing that destroys the artifact it
verifies. It is also why the cloud environment looked like it could not rebuild
the graph: it can, perfectly. `DEFAULT_IN` now points at
`extractions_screened.json`; with it, a rebuild reproduces the committed graph
with **zero drift in any pre-existing field** (meta, all 918 nodes, all 708
hierarchy links, all 272 papers and all 2,011 edges identical), and two rebuilds
are byte-identical. The taxdump is NOT required for `build_kg.py` — only for the
still-blocked species split.

### Shipped: the 33 rank conflicts are now findable

`annotate_rank_conflicts()` in `build_kg.py` adds `rank_conflicts` and
`has_within_paper_conflict` per edge, computed inside `build()` from the edges
just built rather than as a sidecar reading `rank_conflict.json`, so it cannot
drift or self-erase — the same reasoning as `annotate_specificity`. **59 edges**
carry a within-paper conflict (the 33 pairs, counted from both sides). The
viewer gains a `rank ↕` chip, a "Rank conflicts only" filter, and a detail-panel
block naming the counterpart taxon and the study that reports both directions.
Deliberately, only `within_paper` conflicts are chipped: the 189 pairs resting
on no shared paper are an artefact of pooling and flagging them would relaunch
the overstatement this session just corrected. `verify_viz.py` grew to **19
assertions**, all passing — and it earned its keep immediately by catching a
regression from a layout change of mine, where moving the `split` chip into the
scope cell silently changed which chip the specificity sort was read from.

### Highest-value next step

**The 54 named-species split, on a machine with the taxdump** — unchanged as the
top defect, and now the only blocked item that is purely mechanical. Everything
else not needing a GPU is either done or known to be underpowered at n=272.
Second choice, and unblocked here: per-disease evidence summaries with
exportable citations (Task 3.4), the last unstarted item on the useful-output
list.

---

# SUMMARY — session of 2026-09-05 (cloud, CPU-only, no MAIN_DATA, no taxdump)

**The disease dimension of this graph carries reproducible directional
information for Parkinson's disease and, at n=272 papers, for nothing else.**
Write-up: `FINDINGS_disease_specificity.md`.

### What I tested

The last open item needing neither the taxdump nor a GPU: quantify the
disease-side fragmentation nobody had measured (40 disease nodes, zero
containment links, so `Intracerebral hemorrhage` sits beside `Stroke`
unconnected while the taxon side models containment with 708 links).

### What did NOT survive

- **"A clinical subtype resembles its parent disease."** NULL across all seven
  Tier-A is-a pairs (p=0.19–0.75; ICH→Stroke 19/22 decisive at p=0.43,
  Poststroke aphasia→Stroke 0/3 at p=0.75). With 3–22 decisive shared taxa per
  pair nothing could have survived — a power statement, not evidence of absence.
  Tier B (AD→Dementia, the cognitive-decline continuum) behaves the same.
  So the disease-containment layer is a **bookkeeping** decision, justifiable on
  correctness of meaning but **not** a signal gain. It still needs a human call;
  `disease_containment.py` records the tiering and the Tier-C rejections
  (Multiple system atrophy is a *sibling* of PD, not a subtype; MCI is a stage,
  not an AD subtype) so they are not re-proposed.
- **"Disease specificity is a corpus-wide property."** Refuted — see below.

### What survived

- **Disease identity does predict edge direction — p=0.0014.** Over 23,627
  same-taxon paper pairs: same-disease agreement **0.716**, different-disease
  **0.657**, gap **+0.0591**, z=3.39, MDE +0.0296, under a **paper-level**
  permutation of the disease label (pair-level shuffling would have been the
  fourth false positive on record here).
- **It is not country and not method.** Country's own gap is −0.0112 (p=0.64)
  and sequencing platform's −0.0278 (p=0.86) — **neither produces any agreement
  at all.** The disease gap holds inside same-country pairs (+0.078) and
  different-country pairs (+0.086); permuting disease within country blocks
  keeps it (+0.0846, z=2.32, p=0.0128). Both surviving p-values clear BH over
  the four inferential tests.
- **But it is ONE DISEASE.** Per-disease internal agreement: Parkinson's
  **0.807** on 1,220 pairs (lift +0.150 over the 0.657 cross-disease baseline),
  Stroke 0.725, MS 0.689, **Alzheimer's 0.608 — BELOW the cross-disease
  baseline, on 806 pairs with ample power** — Epilepsy 0.492. Zero of the other
  four match Parkinson's. Drop its 67 papers and the gap falls to **+0.0187,
  p=0.179 against MDE +0.0350**: effects above +0.035 are excluded outside PD,
  smaller ones are not. That PD is the standout is the field's own consensus, so
  this is **face validity** for the extraction, not a coincidence.
- **~70% of the graph's directional agreement is a generic dysbiosis prior.**
  Decomposing agreement above the 51.4% marginal chance rate: +14.3 points is
  disease-independent, +5.9 is disease-specific and almost all of that is PD.
  **59 of 187 taxa reported in ≥3 diseases never flip direction** —
  *Streptococcus* enriched in all 12 diseases reporting it, *Butyricicoccus*
  depleted in all 8. For those, "enriched in disease X" is near-contentless.
  Meanwhile *Prevotella* (6↑/8↓ over 14 diseases, 50 papers) and *Bacteroides*
  (6↑/8↓ over 14, 54 papers) are simultaneously the highest-evidence and least
  directionally consistent taxa: **high weight is not high information.**
- **This explains why five structural corrections could not move agreement.** If
  70% of directional agreement is a prior shared with Disbiome and Peryton, the
  validation is largely measuring that prior, not the graph's disease-specific
  content — and the decisive set is dominated by exactly these generic
  well-evidenced taxa. Sixth finding in a row the ~0.013 minimum detectable
  change cannot see; now with a mechanism rather than a shrug.

### Dead end closed: no PyPI package substitutes for the taxdump

The 54 named-species split stayed blocked. `ftp.ncbi.nih.gov` and
`ftp.ncbi.nlm.nih.gov` both give CONNECT → 403; EBI, Ensembl, GBIF, UniProt and
LPSN are denied too; only `pypi.org` / `files.pythonhosted.org` are reachable.
The one offline candidate, **`taxoniq`** (bundles an 89 MB NCBI database), was
extracted and tested: the full tree (2,609,295 taxa with parent and rank) and all
scientific names come out of its marisa tries, **but synonyms are deliberately
excluded — `taxoniq/build.py` indexes only `scientific name`, `common name`,
`genbank common name`, `blast name`.** Verified: `Bacteroidota`→976 resolves,
`Bacteroidetes`→**not found**; `Bacillota`→1239 resolves, `Firmicutes`→**not
found**. Since synonym folding is what stops evidence splitting across duplicate
nodes, and the Disbiome/Peryton join needs both sides through `taxonomy.py`, a
synonym-less table would silently break the graph. **Do not re-run this probe.**

### Bug 4 — the build was never byte-deterministic, so the verification rule cried wolf

Tried to *use* the project's own rule (rebuild twice and diff) before trusting a
rebuild here, and it failed: two builds of the same input differed at byte
278347. **Content was not the difference** — every node, edge, direction,
paper-count and meta field matched across two rebuilds and matched the committed
`graph.json`. The only variation was JSON key order in the `sites` dict on ~30
multi-site edges, because `sites` was `Counter(... for p in papers)` over a
**set** of title strings, whose iteration order is randomised per process by
`PYTHONHASHSEED`. Everything else in that block already went through `sorted()`.

This matters more than a key order sounds: the countermeasure adopted after two
fixes silently erased themselves on rebuild was *rebuild twice and diff*, and
that countermeasure was firing a **false positive on every single build** — which
is exactly how a real regression gets waved through as the usual noise. Fixed by
iterating `sorted(papers)`; three independent rebuilds are now byte-identical,
and `graph.json` is regenerated so a future diff against the committed file is
meaningful. **No number moves.**

### Shipped: generic vs discriminating edges are now IN the graph

Acted on the finding rather than only writing it up. `build_kg.py` now annotates,
per taxon node, `specificity` (breadth, n_diseases_enriched/depleted, purity,
consensus, class) and per edge `taxon_breadth`, `taxon_purity`, `taxon_class`,
`restates_prior`. Over 2,011 edges: **291 generic, 155 discriminating, 664 mixed,
901 narrow** (<3 diseases, nothing can be said); **252 edges restate the taxon's
corpus-wide tendency outright**; 61 taxa generic, 19 discriminating.

Computed inside `build()` from the edges just built, deliberately not as a
sidecar, so it cannot drift out of sync or self-erase on rebuild. A contested
edge casts **no** vote (stricter than the exploratory script, which voted by
majority — *Streptococcus* is generic over 11 diseases here, 12 there; documented
at the code), and the 8 taxa whose every edge is contested get breadth 0 rather
than a missing field.

Verified by executing: two rebuilds byte-identical, all 2,011 edges and 918 taxa
annotated, every pre-existing field unchanged. So the published graph's numbers
are untouched — still 272 papers, 918 taxa, 2,011 edges, Disbiome 71.9%, Peryton
72.5% — and `kg.html` / `docs/index.html` were **not** regenerated.

### Highest-value next step

**Surface the new specificity fields in the viewer** (`build_viz.py` +
`viz_network.js`, then regenerate `kg.html` / `docs/index.html`). The data layer
landed this session; the UI change is deliberately separate because it is
outward-facing. A biologist reading "*Streptococcus* enriched in Parkinson's, 5
papers" still cannot see that it is enriched in eleven other diseases too, and
`restates_prior` on 252 edges is exactly the flag that fixes it. Suggested
treatment: de-emphasise `restates_prior` edges and let the ranked bars sort by
`taxon_purity`, so the discriminating edges surface instead of the loudest ones.

(The 54 named-species split remains the top *defect*, unchanged and still needing
the taxdump on a machine that can reach NCBI — the offline route is now a closed
dead end, see above.)

---

# SUMMARY — session of 2026-09-03

**The assigned analysis returned a null. Attacking it, and then attacking the
graph the same way, found three structural defects that four sessions of
screening had missed: 12 duplicate papers, three disease nodes that are one
disease, and 115 child taxa still folded into their parents.** Write-ups:
`FINDINGS_cooccurrence.md`, `FINDINGS_rank_collapse.md`.

### What I tested

The pooled Task 1 question, named by the last session as the highest-value next
step: within a fixed (taxon, disease) contested edge, do papers reporting
enrichment differ from papers reporting depletion in their taxon co-occurrence
profile? Substrate `relation_sentences.json`; nulls at the paper level.

### What did NOT survive

- **"Same-direction papers share a taxon vocabulary."** First run: per-edge
  +0.047 against a null SD of 0.009 — five sigma, under *both* paper-level
  nulls, and it survived four attacks (profile size, country, sequencing type,
  Jaccard, per-edge median). It was **12 duplicate papers**. On the corrected
  graph: pooled +0.0023 (p=0.79), per-edge +0.0137 (p=0.14), balanced edges
  +0.0080 (p=0.41), rank-based AUC 0.5347 (p=0.21), and 66 of 130 edges
  positive — 51%, chance. Null in all nine variants, and stable across all
  three of this session's corrections. Reported with power: the minimum
  detectable per-edge effect is ~0.017 and |AUC−0.5| ~0.053, so this is "no
  effect visible at 130 edges / 1,848 pairs", not "no effect".
- **"The lone dissenter is just an atypical paper."** The mechanical explanation
  I expected to find. Refuted: majority-side members average 28.6 taxa,
  minority-side 29.5, and permuting within profile-size quintile changed
  nothing.

### What survived

- **Bug 1 — duplicate papers.** 12 papers were scraped once from a PubMed link and again from a
  PMC or publisher link; the copies' titles differ only by a trailing period or
  a curly-vs-straight apostrophe, and every paper key in this pipeline is the
  raw title string. Because edge weight IS paper count, each duplicate voted
  twice. Contributing papers 281 → 272; **`n_replicated` 472 → 437, so 35 edges
  (7.4% of all replicated edges) rest on a single paper**; 76 edges change vote
  counts; **4 lose a majority direction they only had because one paper voted
  twice** (*Bacteroides*/Dementia, *[Clostridium] leptum*/MS, *Bifidobacterium
  longum*/MS, and *Butyricimonas*/MS resolves a false 2-2 tie). Fixed in
  `build_kg.py`, verified a fixed point by rebuilding twice and diffing.
- **The anomaly pointed straight at the bug.** The three edges the failed
  analysis ranked strongest (+0.944, +0.870, +0.796) are three of those four
  direction changes. A statistic found in one step what a full-text screening
  pass over all 45 title-matched papers had not.
- **Bug 2 — one disease, three nodes.** `Anti-N-methyl-D-aspartate receptor
  encephalitis` (22 edges), `NMDAR encephalitis` (16) and `Anti-NMDAR
  encephalitis` (7) were three separate disease nodes, one paper each. This is
  the Bacteroidetes/Bacteroidota case in the disease dimension. Folding it: 45
  edges become 38, and **4 edges that every view showed as single-paper become
  replicated, 3 of them CONTESTED** — real inter-study disagreement the
  fragmentation was hiding, plus a 3-paper unanimous *Faecalibacterium* edge
  displayed as three singletons. Fragmentation hides replication AND
  contradiction; duplication invents it.
- **Bug 3 — the placeholder split was half a fix.** Asking which surface strings
  EXTEND the scientific name they resolved to: **115 strings over 52 nodes, none
  flagged**; 285 edges touch one, 76 contested. 29 are `X sp./spp.` where
  folding is correct; **32 are SILVA placeholders the 2026-09-01 pattern
  missed** (`Prevotella 9`, `Coprococcus_1`, `Clostridium IV`, `Clostridiaceae
  1`); 54 are real named species (`Prevotella copri`, `Klebsiella pneumonia`).
  *Prevotella*/Parkinson's — the graph's highest-weight edge at 17 papers — had
  **13 surface strings folded into one node**, five of them distinct SILVA
  genera. Fixing the placeholder class: taxa 892→918, edges 1,978→2,011,
  placeholder nodes 74→100, containment 684→708, and **5 contested edges were
  contested only because placeholder children were folded in**. `Prevotella_9`
  emerges as its own 2-paper contested edge. The 54 named species are NOT fixed
  — that needs real taxids and this environment's network policy denies the NCBI
  taxdump; `child_folds.json` carries all 115 classified, ready for a machine
  that has it.
- **The self-erasing fix, again — and killed properly this time.** The extended
  placeholder split silently decayed on rebuild (106 → 102 placeholder nodes)
  because a placeholder's parent was recoverable only from a containment link,
  which exists only when the parent is itself a node. Now recorded as
  `parent_taxid` on the node; two consecutive builds are byte-identical. The
  pattern had also over-matched bacteriophages (`Enterococcus phage EFAP 1`);
  added a `NOT_PLACEHOLDER` guard agreeing with the cache's existing one.
- **Self-contradicting papers: 18 claims, and 4 were not contradictions.** Does
  a paper ever call the same taxon both enriched and depleted for one disease?
  18 do, all on contested edges, one contested by a single paper alone. 14 are
  genuine (body site, subgroup, different comparator); **4 are two different
  strings folded onto one key** (*Eubacterium biforme* vs *E. rectale*) — which
  is what exposed Bug 3.
- **No other pseudo-replication is detectable.** Same-cohort screen on the
  curated fields (country + n_cases + n_controls): 4 candidate groups, all
  coincidental, **0 edges drawing >1 paper from any of them**. Power limit: only
  192 of 272 papers (71%) carry a full cohort signature. Dedup completeness
  double-checked two ways — no two rows share a PMID/PMC/DOI under different
  titles (318 of 326 have a resolvable id), and no fuzzy title pair exceeds 0.92.

### Agreement, again, moved by nothing

Headline rates after all three corrections: Disbiome **71.9%**, Peryton
**72.5%** (from 71.9% / 72.8%). The dedup correction measured on the sensitive
metric: −0.0024 (p=0.665) and −0.0007 (p=0.885), against a minimum detectable
change of ~0.013. Its null is deliberately mismatched and conservative — it
drops 12 *random* papers, deleting their evidence outright, where dedup deletes
only redundant evidence.

That makes **five** structural corrections in a row that agreement cannot see.
Treat it as a property of the validation, not a coincidence: the decisive set is
dominated by well-evidenced, unambiguously-named taxa, and every correction so
far acts on the margins. All three are justified on correctness of meaning;
**none may be cited as an accuracy gain.**

### Shipped

`graph.json` / `kg.html` / `docs/index.html` / `rag_corpus.jsonl` rebuilt:
**272 contributing papers, 918 taxa, 2,011 edges, 438 replicated (was 472), 219
contested, 708 containment links, 100 placeholder nodes, 40 disease nodes.**
Disbiome 71.9%, Peryton 72.5%. New: `cooccur_direction.py`,
`cooccur_diagnostics.py`, `cooccur_followup.py`, `child_folds.json`,
`selfcontra_packet.json`, `selfcontra_verdicts.json`, `agreement_dedup.json`,
`FINDINGS_cooccurrence.md`, `FINDINGS_rank_collapse.md`. `build_kg.py` gains
`dedup_rows()` / `--keep-duplicate-papers`, an NMDAR synonym entry, an extended
`PLACEHOLDER` + `NOT_PLACEHOLDER` guard, and `parent_taxid` on placeholder
nodes; `taxonomy_cache.py` prefers it; `agreement_metric.py` gains `--drop
dedup` and `--out`.

### A method note worth keeping

An LLM subagent asked to adjudicate the 18 self-contradictions from the source
sentences returned 6 "extraction errors"; **4 of the 6 were wrong**, and its own
quoted evidence showed an oral-vs-gut contrast or two distinct species. A
one-line deterministic test (was the same surface string on both sides?)
settled it. Where a mechanical test exists, prefer it to a judgement call — and
check the subagent.

### Single highest-value next step

**Split the 54 named species out of their genera — on a machine with the NCBI
taxdump.** It is diagnosed, classified and listed in `child_folds.json`, it
touches the graph's flagship edge, and it is the only item this environment was
blocked from finishing (the network policy denies `ftp.ncbi.nih.gov`). Expect
it to move no agreement number, like the five corrections before it.

After that: **more papers. The binding constraint is n, not method.** Four explanatory
variables have now been tested against contested edges — study design, body
site, and taxon co-occurrence pooled and per-edge — and all four are null. The
minimum detectable effects here (per-edge ≈0.017, |AUC−0.5| ≈0.053, mean
concordance ≈0.01–0.02) are set by 134 contested edges averaging ~5 papers, not
by the statistics. The remaining Task 1 questions (does profile predict
disagreement with the curated databases; are there taxon modules) are the same
shape at the same n and should be expected to return the same answer.

Extraction needs a GPU — **ask before spending.** The CPU-only alternative worth
doing first is cheap and was validated this session as a *method*: anomaly-hunt
the graph's own structure for defects rather than testing hypotheses about it.
That is what actually produced a result twice now (the placeholder collapse, and
this).

---

# SUMMARY — session of 2026-09-02

**The headline is that the instrument was broken.** The test used to evaluate the
last four structural corrections could not, by construction, return anything but
zero. Full write-up: `FINDINGS_validation_metric.md`.

### What I tested

1. Body site as an edge key — the top lever handed over by the previous session.
2. Whether the four "zero flips, p = 1.00" results were real nulls.
3. Both paper-removal corrections, re-run on a metric that can move.

### What did NOT survive

- **"Zero decisive pairs flipped — not underpowered, a true zero."** A tautology.
  A pair is decisive only when our edge is *unanimous* (`contested = bool(up and
  dn)`; verified, all 1,765 non-contested edges have minority vote 0), and every
  correction only removes papers. A unanimous edge that loses papers stays
  unanimous in the same direction, so **no paper-removal correction can ever flip
  a decisive pair**. Confirmed empirically: across the gut restriction 17 edges
  change `direction` and every one is a contested↔decisive transition, never
  enriched↔depleted. Two sessions of "did this recover agreement?" were asked
  with an instrument incapable of answering.
- **"Body site is the highest-value next step."** Wrong on the numbers. Once all
  281 contributing papers are labelled, the corpus is **97.9% gut** — six non-gut
  papers. Restricting to gut moves mean concordance **−0.0073** with Disbiome
  (p = 0.120, min detectable 0.0093) and **−0.0048** with Peryton (p = 0.285) —
  null, and in the *opposite* direction to the hypothesis. Rejected as an edge
  key; shipped as an edge attribute instead.
- **"Rothia/Parkinson's is two saliva studies."** It is one oral and one stool.
- **The MAIN_DATA screen, re-tested honestly:** +0.0015 (p = 0.852) / +0.0043
  (p = 0.620), with only 2–3 pairs moving. Still justified on construct validity,
  still not an accuracy gain.

### What survived

- **A metric that can detect a change.** Signed concordance
  `(n_up−n_down)/(n_up+n_down) × reference_direction`, with a **paper-level**
  resampling null (2,000 draws). Sensitive where the old one was blind: the gut
  restriction moves 16 of 242 Disbiome pairs and 19 of 188 Peryton pairs, where
  McNemar saw 0. It refuses to run unless its tally reproduces `graph.json`.
- **Body site for all 281 papers**, via a keyword scanner *scored before it was
  trusted*: 84.3% by argmax, 92.4% once any stool cue wins outright. The failure
  mode was co-sampling (stool studies drawing serum for metabolomics), not noise.
- **Two bugs, both found by verifying rather than reading.** (1) The
  rank-placeholder fix was **erasing itself on every rebuild** in this
  environment — 77 placeholder nodes → 0, 670 containment links → 610 — while
  printing a successful build. Now a verified fixed point. (2) Edge weight
  counted **observations, not papers**, contradicting the comment directly above
  it; 44 edges inflated, 7 contested edges change their majority label
  (*Bacteroides*/Parkinson's flips depleted → enriched).

### Shipped

`graph.json` / `kg.html` / `docs/index.html` rebuilt: 892 taxa, 1,985 edges, 220
contested, **684 containment links** (+14 correct ones the old build missed), 74
placeholder nodes, and every edge now carries `sites` + `gut_only` with a body
site for all 281 papers. New: `body_site.py`, `analyze_bodysite.py`,
`analyze_bodysite_effect.py`, `agreement_metric.py`,
`FINDINGS_validation_metric.md`. Disbiome 71.9%, Peryton 72.8%.

### Also done this session

**`relation_sentences.json` rebuilt on the full corpus** — the prerequisite the
previous session named. Coverage 211 → **281 of 281** contributing papers, recall
re-validated at **94.8%** of a 97.3% ceiling over 3,132 relations. Details in the
dated entry below.

### Single highest-value next step

**Stop correcting the graph and run Task 1's analysis on the substrate that now
exists.** Three structural corrections in a row have moved agreement by less than
this corpus can resolve — the minimum detectable effect (~0.01–0.02 mean
concordance) is set by *n*, not by the metric — so further cleanup cannot be
shown to help, and the honest options are analysis or more papers.

Analysis is the CPU-only one and is now unblocked: build the paper × taxon
incidence matrix from the filtered sentences and run the **pooled** test (do
papers reporting enrichment differ from papers reporting depletion in their taxon
co-occurrence profile?), with cluster-robust permutation at the paper level.
Pooled, because that is the version with power — per-contested-edge tests average
~4 papers a side and cannot be answered at this corpus size.

Expect it to be hard: this corpus has already produced two false positives that
survived until tested, and a third (the four "true zero" agreement results)
survived until this session. Anything that looks like a finding gets shuffled at
the paper level before it is believed.

More papers needs a GPU for extraction — **ask before spending.**

---

## 2026-09-02 — relation_sentences rebuilt on the full corpus: 211 -> 281 papers

The prerequisite the previous session flagged as blocking the embedding work.
`relation_sentences.py` read only `all_usable_papers.json` (the original 250), so
the filtered-sentence substrate covered **211 of 281** contributing papers and
excluded every paper the MAIN_DATA expansion added. It now merges every corpus
file carrying full text (348 papers), and covers **281 of 281**.

**Recall re-validated at corpus scale, and it holds.** Replaying all **3,132**
extracted relations: headline recall **94.8%** against a **97.3%** ceiling — so
97.4% of what any sentence filter could recover, with the direction-cue step
costing 1.9%. That is marginally *better* than the 250-paper measurement (93.9%
of a 96.7% ceiling). Reduction **14.0x** on sentences, 9.7x on characters —
consistent with the corrected 14.4x, and still nothing like the retired "41x".
162 misses: 67 taxa never appear literally in the paper, 56 dropped by the cue
filter, 39 missed by the matcher.

**Deliberate trade-off, recorded.** The old file was taxdump-built; this one is
replay-cache built, which on the shared 250 papers keeps 5,210 -> 5,102
sentences, **−2.1%**. That independently reproduces the exact 2.1% bias measured
last session. Consistency wins here: a mixed file would apply two different
matchers to different papers, and every downstream use (embeddings, per-paper
co-occurrence) compares papers to each other. Reduction ratios from this file are
therefore an upper bound, as the build warns.

---

# SUMMARY — session of 2026-09-01

**Tested four things and one follow-on fix. The headline is that three of the four
premises I was given turned out to be wrong, and the corrections are the result.**

### What survived

- **The MAIN_DATA corpus really is contaminated.** Reading all 45 title-matched
  full texts, **22 (49%) are not human case-control studies** — 15 animal, 3 with no
  healthy control, 2 case reports, 2 with no primary cohort. Every verdict carries a
  verbatim quote.
- **The relation-sentence filter is safe to build on.** 93.9% recall over 2,262
  extracted relations in `loose` mode, against a 96.7% ceiling — 97.1% of what is
  recoverable, with the direction-cue step costing only 2.0%. Use loose, not strict
  (84.8%).
- **Our extraction is right where two curated databases both say it is wrong.**
  Of 14 doubly-contradicted pairs, **11 of 12 adjudicable ones faithfully report
  what the paper says**. One extraction error in fourteen. Three of the disputes are
  explicitly acknowledged by the source papers themselves.
- **Two structural defects, both larger than the error rate**: rank placeholders
  folded into parents (now fixed), and body site missing from the edge key.

### What did NOT survive

- **"Filtering the contaminated papers will recover agreement."** It changed
  **zero** decisive pairs against either database (exact McNemar p = 1.00). Not an
  underpowered null — a true zero. The 22 papers supply 11 of 1,927 edges; 18 of
  them produced no usable extraction at all. The ~4-point drop was a **composition
  effect**: the batch added ~21 mostly-autism, mostly-single-paper pairs.
- **"Autism edges are worse."** 52.6% agreement looks damning but comes from **5
  papers**. Paper-level permutation: gap −0.225, null SD 0.139, minimum detectable
  0.273, **p = 0.211**. Pair-level shuffling would have returned a false positive.
- **"BM25 structurally cannot answer *what links PD and AD*."** It answers at
  P@10 = 1.00. GraphRAG ties it overall (0.800 vs 0.783 over 6 queries).
- **"41× sentence reduction."** That was a 25-paper pilot generalised ~3× too far;
  at corpus scale it is **14.4×**.
- **Two structural fixes moved agreement by nothing.** Both the paper screen and
  the placeholder split flip **zero** decisive pairs. Agreement rate is insensitive
  to structural corrections here, because decisive pairs are dominated by
  well-evidenced unambiguous taxa. Both were applied on **construct validity**, and
  neither should ever be cited as an accuracy gain.

### Shipped

`graph.json` / `kg.html` / `docs/index.html` rebuilt: **326 papers, 281
contributing, 892 taxa, 1,985 edges, 220 contested, 670 containment links, 77
placeholder nodes.** New: `taxonomy_cache.py`, `graphrag.py`, `compare_retrieval.py`,
`filter_maindata.py`, `analyze_filter_effect.py`, `build_adjudication_packets.py`,
plus four findings docs.

### Single highest-value next step

**Put body site into the edge key.** It is diagnosed, cheap, and currently
manufacturing false contradictions: *Rothia*/PD and almost certainly *Gemella*/PD
are saliva studies colliding with gut records on one node — 2 of the 14
doubly-contradicted pairs are this, not disagreement. `metadata.jsonl` already
carries `body_site` per paper, so this is a keying change, not new extraction. It
should also make the Disbiome/Peryton comparison honest, since both are gut-weighted
and we are currently scoring oral findings against them.

*(Runner-up, and a prerequisite for the embedding work: `relation_sentences.json`
still covers only the original 250 papers, not the current 326. That is why 2 of the
14 pairs could not be adjudicated at all.)*

---

## 2026-09-01 — Fixed the placeholder rank collapse; agreement again moved by zero

Acting on the top lever from the adjudication. `taxonomy.py` resolves a rank
placeholder by trimming its qualifier tail, so **"Erysipelotrichaceae UCG-003"** — an
uncultured *genus-level* label INSIDE the family — landed on the family taxid and
pooled as if it were the family. `build_kg.py` now gives placeholders their own node
with a **containment link** to the parent (`--merge-placeholders` restores the old
behaviour). Default ON, so a rebuild cannot silently revert it.

**The motivating case is fixed.** Erysipelotrichaceae/Parkinson's went from a
4-paper "depleted" family edge — which no paper actually measured — to a 1-paper
family edge plus a separate 3-paper *Erysipelotrichaceae UCG-003* edge beneath it.
The apparent 4-paper contradiction of both curated databases evaporates.

**Effect on agreement: zero, again.** Paired McNemar on pairs decisive in both
graphs: **0 flips**, p = 1.00, against both Disbiome and Peryton. Headline rates
wobble (Disbiome 73.1→71.9, Peryton 71.9→72.8) purely through which pairs are
decisive. That is now three structural corrections in a row that change no decisive
pair — worth treating as a property of this validation, not a coincidence: the
decisive set is dominated by well-evidenced unambiguous taxa, so it cannot see
changes at the margins. **Justified on correctness, not the metric.**

Graph: 892 taxa (+60), 1,985 edges (+69), 670 containment links (+45), contested
225 → **220**, 77 placeholder nodes. `resolved` still means "has an NCBI taxid", so
placeholders are `resolved: false, placeholder: true` rather than inflating the
resolved count.

---

## 2026-09-01 — Adjudicated the doubly-contradicted pairs: 11 of 12 were OUR reading, correctly

Full write-up: `FINDINGS_task3_adjudication.md`. Verdicts + quotes:
`adjudication_verdicts.json`.

**Tested.** The pairs contradicted by BOTH Disbiome and Peryton — the strongest
error signal available. On the screened graph there are **14** (the 11 on record
predates the rebuild). Each read against its source papers' own sentences.

**Survived: our extraction.** 11 of 12 adjudicable pairs faithfully report what the
paper says. **1 extraction error in fourteen.** The doubly-contradicted set is not a
pile of our mistakes — it is mostly the literature disagreeing with itself. Nine are
genuine disputes, and **three are acknowledged by the source papers themselves**:
Dorea *"contrary to Liu's findings (2019)"*; Dialister *"previously shown to have a
higher relative abundance ... in a Southern China population ... may reflect dietary
or other geographical differences"*; Halomonas *"Different from Vogt's and Liu's
studies"*. Dialister is the model contested edge — correct, >10-fold, and the paper
names both the conflict and a mechanism.

**The one real error: Phascolarctobacterium / Parkinson's — DROP.** Its only
supporting sentence says the genus was *"correlated with disease stage"* — a
severity correlation within patients, with no direction and no case-vs-control
contrast. The direction was manufactured. A specific, auditable failure mode:
reading a severity correlation as a disease-vs-healthy direction, despite the
extraction prompt being gated on exactly that contrast.

**Two structural defects, both bigger than the error.**
(1) **Rank placeholders are folded into their parent.** Erysipelotrichaceae/PD looked
like a 4-paper contradiction; in fact 3 of 4 papers report *"Erysipelotrichaceae
UCG-003"*, a genus-level SILVA placeholder INSIDE the family, folded onto the family
taxid, and the 4th attributes the change to a member species. No paper measures the
family aggregate. Systematic: **74 placeholder strings onto 37 taxids, 21 edges named
only by a placeholder, 170 mixed, 52 of those contested.** Lachnospiraceae alone
absorbs ND3007/ND3008/NK4A136/UCG-001/UCG-004/UCG-008. This violates the project's own
rule that synonym folding and containment are different operations — a UCG label is a
*child*, not a synonym.
(2) **Body site is not in the edge key.** Rothia/PD is not a contradiction: both our
papers are saliva studies, the curated records are gut. Gemella/PD is the same paper
and almost certainly the same story.

**Next lever (highest value in the project right now).** Stop folding rank
placeholders into parents — give `X UCG-003` its own node as a containment child of
`X`. It is a bug fix rather than a judgement call, touches ~191 edges, and resolves
the worst-looking contradiction in the set.

---

## 2026-09-01 — GraphRAG built; ties BM25 on ranking, wins only on containment

Full write-up: `FINDINGS_task2.5_graphrag.md`. Code: `graphrag.py`,
`compare_retrieval.py`.

**Built.** Personalized PageRank retrieval (damping 0.85, ~900 nodes, no library):
closed-vocabulary entity linking → PPR from the seeds → a connected subgraph with
directions, evidence counts, containment links and backing papers. Multi-entity
queries rank by the **geometric mean** of per-seed PPR, not a joint run, so a node
must be close to *all* seeds rather than merely near the bigger disease.

**Did NOT survive: the claim that GraphRAG beats BM25.** Over 6 queries with truth
computed from the graph (bridge = ≥2 papers on both diseases), mean precision@10 is
**GraphRAG 0.800 vs BM25 0.783** — a tie. GraphRAG wins 2, loses 1, ties 3.

**Did NOT survive: "BM25 structurally cannot answer *what links PD and AD*".** It
answers at P@10 = 1.00. `build_rag.py` is not pure BM25 — it already has entity
matching and evidence weighting, so it filters to edges about either disease and
ranks by paper count, and hub taxa *are* the bridges. Retire that claim.

**Two measurement errors I made and corrected, logged so they are not repeated.**
(1) The first bridge metric was *saturated*: bare co-membership makes 125 of 832
taxa correct for PD/AD, so any ten hubs scored 1.00 and the systems tied trivially.
A metric that cannot separate them is not evidence they are equal. (2) **PPR is
direction-blind** — proximity has no sign, so "what is depleted in Parkinson's"
returned enriched taxa too (P@10 0.60, a real loss). Direction is now an explicit
filter on the seed disease; 0.60 → 0.80. It still loses that query type to BM25.

**Survived — the actual case for the graph is containment, and it is quantified.**
Query *Hungatella*: GraphRAG returns *Lachnospiraceae* depleted in PD (16 papers)
beside *Hungatella* enriched (7), the rank conflict this project calls load-bearing;
BM25 returns only Hungatella docs and cannot reach the family, because no document
holds both claims. Corpus-wide: **903 (taxon, disease) edges have a parent edge in
the same disease, and 229 of those (25%) point the opposite way.** That is the
retrievable context BM25 structurally misses — and it doubles as a first result for
the "are contested edges rank confusion?" question.

**Recommendation.** Ship GraphRAG for its *output* (connected subgraph with
provenance) and for containment traversal — not on a ranking-accuracy claim, which
the data does not support. Keep `build_rag.py`: it is genuinely better on
directional one-hop queries. Caveat: 6 queries is small and the 0.017 gap is noise.

---

## 2026-09-01 — Relation-sentence filter: recall validated at 93.9%; the "41x reduction" was pilot noise

**Tested.** Whether the relation-bearing sentence filter (`relation_sentences.py`)
keeps the sentences that actually support the relations we extracted — the check
that had to pass before anything downstream is allowed to use it.

**Survived — the filter is safe to build on, in `loose` mode.** Replaying all
**2,262** extracted relations across 250 papers:

| | strict | **loose** |
|---|---:|---:|
| sentence reduction | 22.2x | **14.7x** |
| taxon string anywhere in raw paper (ceiling) | 96.7% | 96.7% |
| taxon seen by matcher | 94.8% | 95.8% |
| **taxon in a KEPT sentence (headline recall)** | 84.8% | **93.9%** |
| share of what the matcher saw (= cue-filter cost) | 89.4% | **98.0%** |
| kept-sentence cue agrees with direction | 86.7% | 96.4% |

Against a **ceiling of 96.7%** — 3.3% of extracted relations name a taxon that never
appears literally in the paper, so no sentence filter can reach them — loose mode
recovers **97.1% of what is recoverable**, and the direction cue filter costs only
2.0%. Strict mode is the wrong trade: it buys 1.5x more reduction for 9 points of
recall, discarding 226 real relations at the cue step alone. **Use loose.**

**Correction: the reduction ratio was badly overstated.** The 25-paper pilot on
record claimed *"13,082 sentences -> 312 (2.4%), a 41x reduction"*. At corpus scale
it is **14.4x** (75,004 -> 5,210 sentences; 10.2x on characters). The pilot
generalised from 25 papers and was off by ~3x. Anything reasoning from "2.4% of
sentences" should be redone at 6.9%.

**Kept the existing `relation_sentences.json`** (taxdump-built, 5,210 sentences)
rather than overwriting it with a cache-built one (5,102). That 2.1% gap is a clean
empirical bound on the replay cache's bias for this task — smaller than expected,
and it confirms the cache understates sentences kept, so measured reduction ratios
are an upper bound.

---

## 2026-09-01 — MAIN_DATA screen: contamination confirmed, but it is not what moved agreement

Full write-up: `FINDINGS_task1_maindata_filter.md`.

**Tested.** Whether screening the 45 title-matched MAIN_DATA papers to human
case-control studies recovers the ~4-point agreement drop with Disbiome/Peryton.

**Survived.** The contamination itself is real and large: reading all 45 full texts,
**22 of 45 (49%) are not human case-control studies** — 15 animal (3xTgAD, R6/1,
Wistar rats, germ-free recolonisation), 3 with no healthy control arm, 2 case
reports (n=1, n=2), 2 with no primary cohort (a review, a Mendelian-randomisation
re-analysis). Every verdict carries a verbatim quote in `maindata_screen.json`.

**Did NOT survive — the headline hypothesis is refuted.** Filtering them changed the
agreement rate by **exactly nothing**: 0 decisive pairs flipped, against either
database, in any variant (exact McNemar p = 1.00 throughout). Not an underpowered
null — a true zero. The 22 papers contribute only 11 of 1,927 edges and 4 of 285
contributing papers; 18 of the 22 yielded no usable extraction at all.

**Why the number ever moved: disease mix, not quality.** The naive per-variant
comparison is confounded — dropping papers drops whole diseases, moving Disbiome's
reference denominator 506 → 364, so variants score different question sets. Paired
per (taxid, disease), the movement is entirely pairs entering/leaving the decisive
set. Dropping all 45 removes 21 decisive pairs, **19 of them Autism spectrum
disorder**, on which we agree with Disbiome 10/21 (48%) versus 112/147 (76%) for
pairs that stay. 17 of the 21 rest on 1–2 papers.

**Null, with power.** ASD's low agreement (52.6%, n=19 pairs) does **not** survive
paper-level permutation: those 19 pairs come from only **5 papers**. Observed gap
−0.225, null SD 0.139, minimum detectable gap 0.273, **p = 0.211**. The test cannot
resolve a gap this size at 5 papers. Pair-level shuffling would have given a false
positive — same trap as the earlier `diet_controlled` (FDR 0.243) and
"198 explanatory terms" (p=0.41) artifacts.

**Applied anyway, on construct-validity grounds, not the metric.** `graph.json`,
`kg.html`, `docs/index.html` rebuilt on the screened corpus: **326 papers, 281
contributing, 832 taxa, 1,916 edges, 225 contested, 625 containment links.** A human
microbe–disease graph should not carry edges whose evidence is transgenic-vs-wildtype
mice. The filter costs nothing and buys correctness of meaning — **it does not improve
agreement and must not be cited as though it did.**

**Infrastructure.** This environment's network policy denies `ftp.ncbi.nih.gov`
(CONNECT → 403), so the NCBI taxdump is unavailable. `build_kg.py` previously fell
straight through to string folding when the taxdump was missing — a silent
regression costing 681 taxid resolutions and all 625 containment links while still
printing a successful build. Added `taxonomy_cache.py`, which replays the resolution
recorded in `graph.json`; verified it reproduces the committed graph **exactly**
(nodes, edges, hierarchy, paper table all identical on a full rebuild diff). It is
valid **only for subsets** of that graph and reports cache misses rather than
silently under-resolving. On the external join it is marginally degraded (Disbiome
overlap 269 vs 272; Peryton 221 vs 221), so absolute agreement rates measured here
are ~0.2–1.1 points off taxdump-measured ones and are not new absolute figures;
all before/after deltas use the identical join and are sound.

**Next lever.** ASD is the weakest region of the graph (new disease, 1–2 papers per
edge, chance-level agreement) and the question is now specific: is ASD genuinely
less replicable, or did 5 papers land badly? That needs more ASD papers, not more
analysis of these 5.
