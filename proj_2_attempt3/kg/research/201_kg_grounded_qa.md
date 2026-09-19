# 201 — KG-Grounded QA over a Contested Microbe–Disease Graph

**Status: COMPLETE** (2026-09-19)

Scope: what to build next for question answering over `graph.json` (883 taxa,
40 diseases, 2,008 association edges, 727 containment links, 271 contributing
papers), given that 217 taxon–disease pairs are *contested* and the project's
load-bearing design decision is that contested edges are kept, never averaged.

## Sections

1. Current state of this repo's RAG (what exists, what it proves)
2. Architecture survey — GraphRAG, LightRAG, HippoRAG/2, PathRAG, RAPTOR, GNN-RAG
3. Representing disagreement — abstention, conflicting-evidence QA, citation-grounded generation
4. Evaluation without a human-labelled QA set
5. Query classes the repaired taxonomy tree unlocks (and where symbolic wins)
6. What NOT to build
7. Recommendation (ranked, with effort and falsifiable tests)

---

## 0. Numbers this document assumes (measured, 2026-09-19)

Recomputed from `graph.json` rather than quoted from prose, because several
figures in the repo's markdown are stale:

| quantity | value |
|---|---:|
| association edges | 2,008 |
| contested edges | **220** (not 217 — that figure is stale) |
| edges with exactly 1 paper | **1,561 (77.7%)** |
| edges with ≥2 papers | 447 (22.3%) |
| edges with ≥3 papers | 212 (10.6%) |
| edges with ≥4 papers | 128 (6.4%) |
| contested edges with ≥4 papers | 85 |
| contested edges at exactly 2 papers (1–1 split) | **89** |
| taxon nodes / disease nodes | 883 / 40 |
| repaired taxonomy tree | 916 nodes, 1 root, 245 Steiner ancestors |

**The single most important number for QA design is 77.7%.** Three quarters of
this graph's edges rest on one paper. Any answer-generation policy must have a
defined behaviour for "one paper says X, nothing else in the corpus speaks to
it", and that behaviour is not "X is true". This is a *bigger* design pressure
than the contested edges, which are only 11% of the graph — and it is invisible
in every paper surveyed below, because public GraphRAG benchmarks are built on
corpora where the same fact is restated many times.

Also note the shape of the contested set: 89 of 220 contested edges are a 1–1
split. Per `00_SYNTHESIS.md` §6, vote counting at n=2 is a coin flip. So for 40%
of contested edges the honest answer is not "the literature leans up" but "two
papers, one each way, and this corpus cannot rank them."

---

## 1. Current state of this repo's RAG — what exists, and what it actually proves

Three retrieval systems exist. They are not variants of one design; they answer
different questions and two of them are unconnected to the graph.

| file | index unit | retrieval | has an answer step? |
|---|---|---|---|
| `build_rag.py` | one doc per **graph edge** (2,008) | BM25 + hard entity filter + direction filter + evidence boost | **no** |
| `graphrag.py` | the graph itself | entity-link → personalised PageRank over 883+40 nodes + containment links → connected subgraph | **no** |
| `rag_query.py` | **paper chunks** (20,905, MiniLM vectors) | LLM query expansion → dense max-over-expansions → best chunk per paper | yes (`--answer`, Sonnet, forced `[n]` citations) |

Four things follow that shape everything below.

**(a) There is no QA system yet, only retrieval.** `rag_query.py` is the only
component with a generation step, and it reads *paper chunks*, not the graph. So
the graph — the asset, the thing with direction, provenance and contestedness —
has never been on the answering path. That gap, not the choice of GraphRAG
variant, is the actual next piece of work.

**(b) `build_rag.py` is not a BM25 baseline; it is already a structured
retriever.** It hard-filters on entity match, hard-filters on direction, boosts
by paper count, and prefers majority-direction edges. `FINDINGS_task2.5_graphrag.md`
records the consequence: the pre-registered claim that "*what links PD and AD* is
a graph query BM25 structurally cannot answer" was **false**, at P@10 = 1.00.
Anyone proposing a new retriever here must beat *this*, not textbook BM25, and
must say so explicitly.

**(c) The retrieval comparison is underpowered and the repo already knows it.**
GraphRAG vs BM25 over six queries: 0.800 vs 0.783 on a stale corpus, reversing
to 0.683 vs 0.700 after `rag_corpus.jsonl` was rebuilt. Exact sign-flip
permutation over all 2⁶ assignments: **p = 1.000**, MDE ≈ 0.17 against an
observed gap of 0.017. **Do not run another 6-query retrieval comparison.** Any
evaluation proposed below must state its MDE before it runs; see §4.

**(d) The one defensible graph-only capability is containment traversal.**
903 (taxon, disease) edges have a parent with an edge in the same disease;
229 (25%) point the opposite way to that parent. BM25 over edge documents
cannot reach the parent claim because no document contains both. This is a
*capability* claim, not a ranking claim, and it is the correct thing to build
the QA layer around.

One more piece of the repo is load-bearing and easy to miss: `rag_query.py`'s
docstring reports that querying a **variable name** fails (top cosine 0.347,
mostly junk) while querying a **sentence a paper would contain** works (0.707,
4/4 correct). That is the strongest local evidence that query→corpus vocabulary
mismatch, not retriever architecture, is where the wins are.

---

## 2. Architecture survey — and why most of it does not apply

### 2.1 The systems

| system | paper | what it actually does | what it assumes |
|---|---|---|---|
| **Microsoft GraphRAG** | *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, Edge et al., [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) (v1 2024-04-24, v2 2025-02-19) | LLM extracts an entity graph from raw text → Leiden community detection → **pre-generates an LLM summary per community at every hierarchy level** → a global query maps over community summaries and reduces to one answer | a corpus large enough that no one can read it (evaluated "in the 1 million token range"), and *global sensemaking* questions ("what are the main themes?") |
| **LightRAG** | Guo, Xia, Yu, Ao, Huang, [arXiv:2410.05779](https://arxiv.org/abs/2410.05779) (2024-10-08; v3 2025-04-28) | dual-level retrieval — low-level (entity/relation) and high-level (theme keywords) — over a graph index plus vectors, with an **incremental update** algorithm | graph built by LLM from text; the win is cost/latency vs GraphRAG and the ability to add documents without a full reindex |
| **HippoRAG** | Gutiérrez et al., NeurIPS 2024, [arXiv:2405.14831](https://arxiv.org/abs/2405.14831) | OpenIE → KG → **Personalised PageRank from query entities** as a single-step multi-hop retriever | a KG dense enough that PPR mass lands on the right passages |
| **HippoRAG 2** | *From RAG to Memory: Non-Parametric Continual Learning for LLMs*, Gutiérrez, Shu, Qi, Zhou, Su, [arXiv:2502.14802](https://arxiv.org/abs/2502.14802) (2025-02-20) | PPR again, plus deeper passage integration and online LLM filtering of triples; reports +7% on associative-memory tasks over the SOTA embedding model | same, at multi-document corpus scale |
| **PathRAG** | Chen, Guo, Yang, Chen, Chen, Liu, Shi, Yang, [arXiv:2502.14902](https://arxiv.org/abs/2502.14902) (2025-02-18; v2 2025-11-17) | extracts **relational paths** between query nodes, prunes them with a flow-based score, and renders paths (not node sets) into the prompt | redundancy is the problem — i.e. the graph is big enough that naive neighbourhoods overflow the context window |
| **RAPTOR** | Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning, ICLR 2024, [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) (2024-01-31) | recursively embed → cluster → **summarize** text chunks into a tree; retrieve at any abstraction level. +20% absolute on QuALITY with GPT-4 | long *documents* with latent hierarchy; it is a text-tree method, not a KG method |
| **GNN-RAG** | Mavromatis & Karypis, [arXiv:2405.20139](https://arxiv.org/abs/2405.20139) (2024-05-30) | a GNN scores answer candidates over a dense KG subgraph; shortest paths from question entity to candidate are verbalised for the LLM | Freebase-scale KGQA (WebQSP, CWQ) with **labelled train sets of question→answer pairs** |

### 2.2 Which assumptions fail here, concretely

**Community summarization is ceremony at 923 nodes.** GraphRAG's entire value
proposition is that you cannot read the corpus, so you pre-summarize it into a
hierarchy. This graph has 923 nodes and 2,735 edges; `graphrag.py` already notes
PPR over it runs in "milliseconds, no library". More importantly, the communities
are *already known and named*: the graph is near-bipartite taxon↔disease, so
Leiden will recover approximately "one community per disease", which is the
`disease` node's neighbourhood — obtainable with a dict lookup. And the
pre-generated summary is the wrong artifact for this data: a natural-language
community summary **averages**, which is the one operation this project has
decided is forbidden. *Falsifiable version*: run Leiden on `graph.json`, compute
adjusted mutual information between the community assignment and the disease
label of each edge. If AMI > 0.7, community detection has discovered nothing
that `e["disease"]` does not already say. I expect it to be high; if it comes
back low and the communities are cross-disease taxon guilds, that is a real
finding and worth following.

**The KG-construction half of GraphRAG/LightRAG is already done, better.** Both
systems spend most of their token budget having an LLM build a graph from text
with an open schema. This project has a *closed* schema validated against two
curated databases, with per-paper provenance, significance gating, and a
measured reading fidelity of ≥86.6%. Adopting GraphRAG wholesale would mean
throwing that away and re-extracting with an open schema — which is precisely
what `mega_dump/proj_2` did, at 6% precision. **Take the query-side ideas; do
not take the indexing side.**

**HippoRAG is already implemented here, and it tied.** `graphrag.py` is
personalised PageRank from entity-linked seeds — the HippoRAG mechanism, minus
the OpenIE step this project does not need. The repo's own comparison says it
does not beat the lexical baseline on ranking (p = 1.000, n = 6). The remaining
HippoRAG 2 delta is LLM-in-the-loop triple filtering, which buys little when the
triple set is 2,008 curated edges rather than noisy OpenIE output. **Do not port
HippoRAG 2.**

**PathRAG is the one with a transferable idea, and it is not the pruning.**
PathRAG's flow-based pruning solves context overflow, which does not occur at
this scale. But its *representation* — render a **path** into the prompt, not a
bag of nodes — is exactly right for the containment story. The answer to
"Hungatella in Parkinson's" wants the path
`Hungatella —is-a→ Lachnospiraceae —depleted(16p)→ Parkinson's` alongside
`Hungatella —enriched(7p)→ Parkinson's`. That is a two-path rendering, and it is
a template, not a model. Steal the format; skip the machinery.

**RAPTOR is about text trees; this project's tree is a real taxonomy.** RAPTOR
*infers* a hierarchy by clustering embeddings because none exists. Here one
exists at 100% precision (916 nodes, 1 root, NCBI lineages —
`repair_taxonomy_tree.py`). Using RAPTOR would mean replacing a correct
hierarchy with a guessed one. The only RAPTOR-shaped question that is still open
is whether *paper chunks* (`chunks.jsonl`, 20,905) benefit from a summary tree —
a separate system from the KG, and low priority.

**GNN-RAG requires supervision this project does not have and should not
manufacture.** WebQSP/CWQ come with thousands of labelled question→answer pairs.
There is no labelled QA set here, and `00_SYNTHESIS.md` §2 already rules out GNNs
on this graph for independent reasons (KATZHMDA, a zero-learning topology index,
scores 0.84–0.86 on a comparable benchmark; published MDA-GNN evaluations use
leaky random-negative CV). **Do not build GNN-RAG.**

### 2.3 Two 2026 results that should temper enthusiasm

- **Knowledge-graph grounding helps LLMs mainly for knowledge the model does not
  already have.** *Knowledge-Graph Grounding Helps LLMs Only for Out-of-Training
  Knowledge*, [arXiv:2606.22419](https://arxiv.org/abs/2606.22419) (2026-06-21),
  tests GraphRAG on clinical QA and finds the grounding benefit concentrates on
  facts outside the model's training data. *This cuts both ways for this
  project.* A published microbiome graph built from 271 papers, most of them in
  any frontier model's pretraining set, is at risk of the null: the model may
  answer "is *Akkermansia* enriched in Parkinson's?" correctly with no retrieval
  at all. **The closed-book baseline is therefore mandatory** — see §4. But the
  project's genuinely out-of-training content is *the aggregate*: "how many
  papers in this corpus, and which way do they split." No pretrained model knows
  that, and that is the query class to target.
- **No graph method closes the reasoning gap.** *GraphInfer-Bench*,
  [arXiv:2606.11562](https://arxiv.org/abs/2606.11562) (2026-06-10), evaluates
  GraphRAG among others and reports "no method family closes the gap" on novel
  graph inference. Treat any claim that a graph retriever will make the LLM
  *reason* better as unsupported; the defensible claim is that it puts the right
  facts, with provenance, in front of the model.

### 2.4 Verdict

The right architecture here is not on this list. It is: **deterministic
graph→text rendering with a template, plus a strictly constrained generator.**
Entity-link the query (exists), pull the exact subgraph including containment
parents and children (exists in `graphrag.py`), render every retrieved edge with
its direction, paper count, up/down split and paper titles (exists in
`build_rag.py:edge_text`), and let the LLM do the one thing it is needed for —
turning that into English without adding anything. Every component is already
written. **What is missing is the answer policy, not the retriever.** That is §3.

---

## 3. Representing disagreement

### 3.1 What the literature has established

The relevant body of work is "knowledge conflict in RAG", and it is mostly about
*conflict between the model's parametric memory and a retrieved passage*, or
about *misinformation*. This project's conflict is a third kind — **two
legitimate, correctly-read, mutually contradictory empirical findings** — which
is much closer to systematic-review methodology than to fact-checking. That
mismatch matters and most of the benchmarks below do not cover this case.

**The single most actionable published result for this project:**

> *When Evidence Conflicts: Uncertainty and Order Effects in Retrieval-Augmented
> Biomedical Question Answering*, Han, Lan & Kilicoglu,
> [arXiv:2605.14115](https://arxiv.org/abs/2605.14115) (2026-05-13). Six
> open-weight LLMs, five evidence conditions, the HealthContradict benchmark.
> **Reversing the order of the same two contradictory documents drops accuracy
> for every model and flips 11.4%–25.2% of predictions.** Their proposed fix is a
> conflict-aware abstention score (confidence × an explicit conflict detector),
> which beats confidence-only abstention by 7.2–33.4 points in the
> incorrect-only condition.

Read that against this project's data: 220 contested edges, 89 of them 1–1
splits. An LLM handed two contradictory papers will give an answer that depends
on **which one you paste first**, a quarter of the time. This is not a
hypothetical risk; it is a measured property of the exact model class that would
be used here, on biomedical contradictions. **Any generator this project ships
must be order-randomised and order-tested.** That test is cheap (§4.4) and it is
the single highest-value evaluation in this document.

Other relevant work:

- ***Retrieval-Augmented Generation with Conflicting Evidence***, Wang, Prasad,
  Stengel-Eskin & Bansal, [arXiv:2504.13079](https://arxiv.org/abs/2504.13079)
  (2025-04-17). Introduces **RAMDocs** (ambiguity + misinformation + noise
  jointly) and **MADAM-RAG**, a multi-agent debate with an aggregator.
  Llama-3.3-70B-Instruct scores **32.60 exact match** on RAMDocs — i.e. frontier
  open models mostly fail at conflicting evidence. MADAM-RAG gains up to 11.40
  (AmbigDocs) / 15.80 (FaithEval), and the authors state a substantial gap
  remains as evidence imbalance grows. **Do not adopt multi-agent debate here.**
  Debate is designed to *adjudicate* — to decide which side is right — and this
  project's position is that the corpus cannot decide. Debate would manufacture
  a winner. It is the right tool for misinformation and the wrong tool for
  genuine heterogeneity.
- ***Enabling LLMs to Generate Text with Citations*** (ALCE), Gao, Yen, Yu &
  Chen, EMNLP 2023, [arXiv:2305.14627](https://arxiv.org/abs/2305.14627).
  Defines the fluency / correctness / **citation precision & recall** triple
  that is now standard. Headline: on ELI5 even the best models lack complete
  citation support **50% of the time**. Citation precision/recall is directly
  computable here and is the backbone of the proposed eval in §4.
- ***Adaptive Chameleon or Stubborn Sloth***,
  [arXiv:2305.13300](https://arxiv.org/abs/2305.13300) (ICLR 2024) — the
  foundational result that LLMs are simultaneously over-receptive to external
  evidence and subject to confirmation bias toward their parametric prior.
- ***Query-driven Document-level Scientific Evidence Extraction from Biomedical
  Studies***, [arXiv:2505.06186](https://arxiv.org/abs/2505.06186) (2025-05-09) —
  **CochraneForest**, built from forest plots in Cochrane systematic reviews.
  This is the closest existing dataset to what this project is doing: clinical
  questions where the evidence genuinely conflicts. Worth reading for its
  *presentation* conventions, which are the medical field's 40-year-old answer to
  "how do you show a reader that studies disagree": you show every study, its
  effect and its interval, and you show the pooled estimate only when
  heterogeneity permits.
- Conflict-aware **routing** is converging as the design pattern: *EvidentialRAG*
  ([arXiv:2607.10491](https://arxiv.org/abs/2607.10491), 2026-07-11) routes to
  direct answering / conflict-aware answering / **abstention**; *PassiveQA*
  ([arXiv:2604.04565](https://arxiv.org/abs/2604.04565), 2026-04-06) uses
  Answer / Clarify / Abstain. *Two Axes of LLM Abstention*
  ([arXiv:2607.08456](https://arxiv.org/abs/2607.08456), 2026-07-09) makes the
  distinction this project needs: **answer-confidence and question-answerability
  are separate axes** and need separate scores. Here they map cleanly — "is the
  edge in the graph?" (answerability) vs "does the evidence decide the
  direction?" (confidence). Conflating them is how a system ends up saying "no
  data" when it means "contested".

**What is missing from all of it:** none of these benchmarks contains the case
that dominates this graph — **one source, no contradiction, and no corroboration
either** (1,561 of 2,008 edges). The conflict literature assumes ≥2 documents.
The abstention literature assumes the question has a right answer that is either
supported or not. A single-paper directional finding in microbiome literature is
neither confirmed nor contested; it is *unreplicated*, and the field's own base
rates (Gibbons et al. 2018, PLOS Comput Biol: 67% of OTUs differ significantly
between two studies' **healthy-control cohorts alone**) say an unreplicated
directional finding should carry very little weight. **This project will have to
define that category itself.** That is a contribution, not a gap.

### 3.2 The answer policy this project should implement

Not a model — a deterministic classifier over graph fields, computed before the
LLM sees anything. Five states, with fixed rendering rules:

| state | condition (on the edge) | count | the answer says |
|---|---|---:|---|
| **UNREPLICATED** | `n_papers == 1`, not contested | **1,560** | "One paper reports X. No other paper in this corpus examined it. This is a single unreplicated observation." |
| **SPLIT** | contested, `n_up == n_down` | **101** | "The literature is split: N papers each way. This corpus cannot resolve the direction." |
| **LEANING** | contested, `n_up != n_down` | **119** | "N of M papers report X; the other report the opposite. Majority direction is X, but disagreement is substantial." |
| **CONSISTENT-WEAK** | not contested, 2 ≤ `n_papers` ≤ 3 | **185** | "N papers agree on X, none disagree. At N studies, vote counting is near chance (~50% at 2, ~67% at 3)." |
| **CONSISTENT-STRONG** | not contested, `n_papers` ≥ 4 | **43** | "N papers agree on X with no paper reporting the opposite." |
| **NOT-IN-GRAPH** | no edge | — | "This corpus contains no significance-gated, disease-vs-control observation for that pair." Distinct from "no association exists." |

*(Counts computed from `graph.json` 2026-09-19; they sum to 2,008. One contested
edge has `n_papers == 1` — a paper contradicting itself — and falls in SPLIT.)*

**Note that only 43 edges in the entire graph — 2.1% — reach the tier where vote
counting is ~90% reliable.** That number should be on the system's front page. A
QA layer that answers 2,008 questions with equal confidence is misrepresenting
its own asset by a factor of fifty.

Also: `n_up + n_down` can exceed `n_papers` (a paper reporting both directions
for one pair). The renderer must not compute percentages that assume they sum.

Three rules that are not negotiable:

1. **The state is computed, not generated.** The LLM receives the state label as
   part of the context and is instructed to use it verbatim. It never infers
   contestedness from reading the passages — that is exactly the operation the
   order-effect result says it does unreliably.
2. **Evidence order is randomised per call and the answer must be invariant.**
   See §4.4.
3. **NOT-IN-GRAPH ≠ no association.** The extraction prompt gates on reported
   statistical significance, main text only, disease-vs-healthy-control only
   (`eval-v2/run_eval.py`, `samgated-v1`). Absence from this graph means
   "unreported under that gate", and the QA layer must say so, every time. The
   repo's own recall work is the authority here: paper-level recall ≥96.1%, but
   *edge-level* recall is at most ~98.1% with roughly 60 missed observations
   [14, 116] — and the instrument "can confirm a miss but cannot refute one."

The calibration numbers in the CONSISTENT-WEAK row come from `00_SYNTHESIS.md`
§6 (2 supporting studies → ~50% confirmation, 3 → ~67–70%, 4+ → ~90%; Hedges &
Olkin 1980 on vote-counting power). Quoting them in the answer is what makes the
system honest rather than merely hedged.

### 3.3 What a good contested answer literally looks like

Query: **"Is *Prevotella* increased in Parkinson's disease?"**

All figures below are the real contents of `graph.json` as of 2026-09-19, not an
illustration.

> **Short answer: the literature disagrees, and this corpus does not resolve it.**
>
> ***Prevotella*** (genus) — **Parkinson's disease**: **CONTESTED / LEANING
> DEPLETED.** 12 papers report this pair: **11 report it depleted**, **2 report
> it enriched** (directional consistency 0.846; the counts sum to 13 because one
> paper reports both directions for this pair).
>
> **Treat the majority as a lean, not a result.** Across this corpus, 27.6% of
> decisive observations disagree with the leave-one-out majority of their own
> edge, and *no* study-design variable examined — 24 of them, including
> DNA-extraction kit, sequencing platform, 16S region, country, cohort size,
> medication and diet control — explains which papers hold the minority
> direction (`FINDINGS_paper_discordance.md`). The paper-level SD of discordance
> is 3.4 points on a 27.6% base, so this corpus cannot attribute the
> disagreement to any measured cause.
>
> **Reporting depleted (11 papers):** ⟨titles + links⟩
> **Reporting enriched (2 papers):** ⟨titles + links⟩
>
> **The answer changes with the rank you ask at. All four are in this graph:**
>
> | rank | node | direction | papers (up/down) | state |
> |---|---|---|---|---|
> | family | *Prevotellaceae* | depleted | 5 (1↑/4↓) | CONTESTED |
> | genus | *Prevotella* | depleted | 12 (2↑/11↓) | CONTESTED |
> | clade (16S) | *Prevotella_9* | — | 2 (1↑/1↓) | **SPLIT — undecidable** |
> | species | *Prevotella pallens* | **enriched** | 2 (2↑/0↓) | CONSISTENT-WEAK |
> | species | *Prevotella melaninogenica* | **enriched** | 1 (1↑/0↓) | UNREPLICATED |
>
> **The species inside the genus point the other way.** 25% of parent-child
> pairs in this graph that share a disease point opposite ways, so a genus-level
> answer does not transfer down. Do not read "*Prevotella* is depleted in PD" as
> covering *P. pallens*. Note also that the two species rows rest on 2 and 1
> papers respectively — at that count, vote counting is at or near chance.
>
> **What this answer is not.** It is not an effect size: this graph stores
> direction only, and the source papers report incommensurable statistics
> (LEfSe LDA, fold change, p-values), so no pooled magnitude exists and no
> heterogeneity statistic (Q, τ², I²) can be computed. It is not independent of
> curated databases: Disbiome and Peryton cite 43 and 24 of our papers.
> It covers only findings that this corpus's extraction gate admits — reported
> statistical significance, main text, disease vs healthy control.

Four properties make that a good answer and they are all checkable by a script:
every directional claim carries a paper count; both sides are enumerated with
titles; the containment conflict is surfaced unprompted; and the scope
limitations are stated rather than implied. Note that **nothing in it required
the LLM to judge anything** — it is a template fill. That is deliberate, and it
is what makes §4 tractable.

---

## 4. Evaluation without a human-labelled QA set

### 4.0 The constraint, stated honestly

There is no labelled QA set, there will not be one, and annotator time is tens of
hours of biology-student attention. That is the *entire* budget. Every design
below is sized against it.

Two prior attempts in this repo failed, and both failures are more informative
than the literature:

1. **The LLM-judge prototype was a data leak.** It "recovered FPs" by consulting
   the Opus 4.8 gold as an oracle — i.e. it was scored on information the
   deployed system would not have (`proj_2_attempt3/CLAUDE.md`, eval-v2 method
   experiments). The number it produced looked good and meant nothing.
2. **The rebuilt, genuinely gold-free judge was not worth running.**
   `experiments/JUDGE_RESULTS.md`: each model verifying its own extractions
   against the paper is a **precision↔recall trade with net F1 barely moving**
   (Qwopus +0.02, Qwythos −0.06), at ~42 s/paper on top of ~22 s extraction. The
   one-line diagnosis in that file is the most portable thing this project knows
   about LLM judges: ***"a model is only as good a judge as it's an extractor."***

So the prior here is not "LLM-as-judge is unproven"; it is "**LLM-as-judge has
been tried twice in this project and produced one leak and one null.**" A third
attempt needs a reason, and "RAGAS has a faithfulness metric" is not one.

There is also a third, subtler contamination already on record: the retrieval
ground truth in `compare_retrieval.py` is **computed from `graph.json`**. That is
the right call for a consistency check and it is labelled as such in
`FINDINGS_task2.5_graphrag.md` ("it measures internal consistency … not
biological correctness"). But it means the graph is simultaneously the system
under test and the reference, and any future eval that quietly reuses that
pattern for a *correctness* claim is circular. Name the reference every time.

### 4.1 What the automated-RAG-eval toolkits actually measure

| tool | paper | what it scores | does it transfer here? |
|---|---|---|---|
| **RAGAS** | Es, James, Espinosa-Anke & Schockaert, *Ragas: Automated Evaluation of Retrieval Augmented Generation*, [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) (2023-09-26; v2 2025-04-28), EACL 2024 demo | reference-free triple: **faithfulness** (answer entailed by retrieved context), **answer relevance**, **context relevance** — all LLM-scored | **mostly redundant, see below** |
| **ARES** | Saad-Falcon, Khattab, Potts & Zaharia, [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) (2023-11-16; v2 2024-03-31), NAACL 2024 | same three dimensions, but with **fine-tuned lightweight judges** trained on synthetic queries, plus **prediction-powered inference** over "only a few hundred human annotations" to give statistically valid confidence intervals | **the PPI half transfers; the judge half does not** |
| **RAGChecker** | Ru et al., [arXiv:2408.08067](https://arxiv.org/abs/2408.08067) (2024-08-15) | **claim-level entailment** — decompose the answer into claims and check each against retrieved context and ground truth; reports better human correlation than prior metrics | the decomposition idea is right; here the claims are already atomic by construction |
| **AIS / attribution** | Rashkin, Nikolaev, Lamm, Aroyo, Collins, Das, Petrov, Tomar, Turc & Reitter, *Measuring Attribution in Natural Language Generation Models*, [arXiv:2112.12870](https://arxiv.org/abs/2112.12870) (2021-12-23; rev. 2022-08-02), Computational Linguistics 2023 | "Attributable to Identified Sources" — a two-stage human annotation protocol for whether a generated statement is supported by a cited source | **this is the right protocol for the human hours**, see §4.6 |
| **ALCE citation P/R** | Gao, Yen, Yu & Chen, EMNLP 2023, [arXiv:2305.14627](https://arxiv.org/abs/2305.14627) | citation precision (does each cited source support the claim) and citation recall (is each claim cited) | **directly computable, and cheap — §4.3** |

**The critical observation, and it is specific to this project's §3 design.**
RAGAS/ARES faithfulness asks: *is the answer entailed by the retrieved context?*
Here the retrieved context is **not** free text. It is a deterministic rendering
of graph fields — `n_up`, `n_down`, `n_papers`, `consistency`, `contested`,
`direction`, `papers[]`, `ev[]` — produced by code this project wrote. So
"faithfulness" collapses to *did the generator reproduce integers and a direction
label that were handed to it verbatim?* That is decidable by string comparison,
exactly, in microseconds, at zero annotator cost and with zero judge variance.

**Running an LLM judge on it would be replacing an exact instrument with a noisy
one.** That is the single most important design call in this section, and it is
only available because §3 made the answer a template fill. Do not give it up.

RAGAS still has one non-redundant use: **answer relevance** (did the system
answer the question asked, or a neighbouring one?) is genuinely a judgement call
and is not checkable by assertion. Note it is also the metric most exposed to the
biases in §4.2. Scope any RAGAS use to that one metric and say so.

What ARES contributes that nothing else does is **prediction-powered inference**
(Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, *Prediction-powered inference*,
Science 382(6671):669–674, 2023-11-09,
[arXiv:2301.09633](https://arxiv.org/abs/2301.09633)). PPI gives a *valid*
confidence interval from a large machine-labelled sample corrected by a small
human-labelled one — which is precisely the shape of this project's budget:
thousands of answers, tens of hours of biologist. **Take PPI. Skip the
synthetic-query judge training**, which needs a query distribution this project
does not have and would be trained on the same graph it is meant to audit.

### 4.2 Known failure modes of LLM judges, and which ones bite here

| failure mode | source | bites here? |
|---|---|---|
| **position / order bias** | Wang, Li, Chen, Cai, Zhu, Lin, Cao, Liu, Liu & Sui, *Large Language Models are not Fair Evaluators*, [arXiv:2305.17926](https://arxiv.org/abs/2305.17926) (2023-05-29; rev. 2023-08-30) — ranking "can be easily hacked by simply altering their order of appearance"; Vicuna-13B beat ChatGPT on 66/80 queries purely by reordering, with ChatGPT judging | **yes, severely** — and not only in the judge. Han, Lan & Kilicoglu ([arXiv:2605.14115](https://arxiv.org/abs/2605.14115)) show it in the *generator* on biomedical contradictions: 11.4–25.2% of predictions flip when two contradictory documents swap order. With 220 contested edges this is the dominant risk. **§4.4 tests it.** |
| **verbosity bias** | Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023 D&B, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) (2023-06-09) | **yes, and it is adversarial to this project's goal.** The correct answer for 1,560 UNREPLICATED edges is short and hedged; the wrong answer is long and confident. A verbosity-biased judge systematically prefers the wrong one. |
| **self-enhancement / self-preference** | Zheng et al. (above); Panickssery, Bowman & Feng, *LLM Evaluators Recognize and Favor Their Own Generations*, [arXiv:2404.13076](https://arxiv.org/abs/2404.13076) (2024-04-15) — self-recognition accuracy correlates linearly with self-preference strength | **yes** — and this repo already has the applied version of it: the gold-free judge was the same model as the extractor and measured a net-zero F1 trade. **Never let the answering model judge its own answers.** |
| **12-bias taxonomy** | Ye, Wang, Huang, Chen, Zhang, Moniz, Gao, Geyer, Huang, Chen, Chawla & Zhang, *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge*, [arXiv:2410.02736](https://arxiv.org/abs/2410.02736) (2024-10-03) — CALM framework; advanced models still carry significant task-specific bias | background; the specific three above are the ones with a mechanism here |
| **agreement ≈ 80%** | Zheng et al.: GPT-4 matches human preference >80%, "the same level of agreement between humans" | this is the *ceiling*, and it is quoted on preference ranking, not on biomedical factual support. A judge at 80% cannot audit a template renderer that is right ~100% of the time by construction — **the instrument would be less accurate than the thing it measures.** |

**Design rules that follow, non-negotiable:**

- **R1.** No LLM judge for anything decidable by assertion. (Kills ~80% of the
  proposed eval surface — see §4.3.)
- **R2.** The judge, where one is used at all, is a *different model family* from
  the generator, and neither is the extractor (Qwopus3.5). Three distinct roles,
  three distinct models.
- **R3.** Every judged comparison is run in both orders and scored on the
  agreement of the two runs, not on one run. Balanced position calibration, per
  Wang et al.
- **R4.** Length is controlled: candidate answers in any pairwise judgement are
  truncated or padded to comparable length, or the judgement is made on extracted
  claims rather than prose.
- **R5.** No reference that the system under test produced, contributed to, or was
  tuned on. Write the provenance of the reference into the results file.

### 4.3 The primary instrument: an assertion suite, not a judge

This is the recommendation. It costs **zero annotator hours**, runs in seconds,
and it is *more* sensitive than any judge because it is exact.

Because §3.2 computes the state label deterministically and §3.3's answer is a
template fill, the following are all decidable by script against `graph.json`.
Every one is a regression test, and every one has a null of **0 violations**.

| # | assertion | field(s) checked | violation means |
|---|---|---|---|
| A1 | every integer in the answer appears in the edge record | `n_papers`, `n_up`, `n_down`, `n_obs` | fabricated count |
| A2 | every cited paper title is in `edge["papers"]` | `papers[]` | fabricated citation (ALCE **citation precision**) |
| A3 | every directional claim carries a citation | — | uncited claim (ALCE **citation recall**) |
| A4 | the state word in the answer == the computed state label | `contested`, `n_papers`, `n_up`, `n_down` | the generator re-judged contestedness — the exact operation §3.2 rule 1 forbids |
| A5 | if `contested`, both directions appear in the answer with counts | `n_up`, `n_down` | one-sided rendering of a contested edge — the most damaging failure this system can have |
| A6 | if `n_up == n_down`, the answer contains no majority claim | — | manufactured winner at a 1–1 split (89 edges) |
| A7 | if `n_papers == 1`, the answer contains the unreplicated qualifier | `n_papers` | 1,560 edges silently asserted as fact |
| A8 | no edge is reported with a percentage computed as `n_up/(n_up+n_down)` where `n_up+n_down != n_obs` | `n_obs` | the both-directions-in-one-paper arithmetic bug flagged in §3.2 |
| A9 | if a containment parent or child with an edge in the same disease exists, it appears in the answer | `hierarchy`, `taxonomy_tree.json` | the 903-edge / 229-conflict capability from §1(d) is not actually delivered |
| A10 | if no edge exists, the answer uses the NOT-IN-GRAPH wording and does **not** assert absence of association | — | the gate is being misrepresented as biology |
| A11 | no effect size, fold change, *p*-value, or pooled magnitude appears in the answer | — | invented precision; the graph stores direction only |
| A12 | no heterogeneity statistic (Q, τ², I²) appears | — | mechanically uncomputable from direction counts (`00_SYNTHESIS.md` §6) |

**Coverage.** Run all 12 over **all 2,008 edges** — the whole graph is the test
set, because the reference is the graph record itself and the claim under test is
"the renderer did not distort it." This is not circular: the assertions do not
test whether the *graph* is right (that is what the fidelity, recall and
Disbiome/Peryton work measures); they test whether the *answer layer* preserves
it. Those are different claims and the results file must say which one it is
making.

**Stated null and falsifier.**
> **Null:** the answer layer introduces zero distortions — 0 violations of A1–A12
> across 2,008 edges.
> **Falsifies "the RAG works":** any violation of **A4, A5, A6 or A7**. These four
> are the answer policy. A system that fails them is not a weaker version of the
> design in §3; it is a different system that happens to cite papers.
> A1–A3 and A8–A12 violations are bugs to fix, not design failures.

**Power.** This is where the assertion suite dominates any judge. With 2,008
edges and 0 observed violations, the rule of three gives a 95% upper bound on the
true violation rate of **3/2008 = 0.15%**. Compare the retrieval comparison this
repo already ran: n=6, MDE ≈ 0.17 (17 points), p = 1.000. The assertion suite is
**roughly a hundredfold more sensitive**, and it costs nothing. §1(c) says do not
run another 6-query comparison; this is what to run instead.

**What this does not measure**, stated plainly so nobody overclaims it: it does
not measure whether the answer is *useful*, whether it answers the question
asked, or whether the graph is *true*. It measures non-distortion. That is a
smaller claim than "the RAG works" and it is the part that can be established
for free.

### 4.4 The order-invariance test — the highest-value single experiment here

Motivated by Han, Lan & Kilicoglu ([arXiv:2605.14115](https://arxiv.org/abs/2605.14115),
2026-05-13): on biomedical contradictions, reversing two documents flips
11.4%–25.2% of predictions across six open-weight models.

**Design.** For each of the 220 contested edges, build the answer twice: once
with the supporting-evidence list in `ev[]` order, once reversed. (Optionally
also with a random shuffle, seeded, for a third arm.) Compare the two outputs on
three binary features, all extracted by regex, no judge:

1. the state label (SPLIT / LEANING / …),
2. the asserted majority direction, if any,
3. the set of cited paper titles.

**Null:** flip rate = 0 on all three. The §3.2 design *guarantees* this, because
the state and the majority are computed before the LLM is called. So this test is
a direct check that rule 1 of §3.2 is actually implemented rather than merely
written down.

**Falsifier and power.** Observing 0 flips in 220 gives a 95% upper bound of
**1.4%** on the true flip rate. Han et al.'s lowest reported figure is 11.4% —
so this test has *overwhelming* power against the published effect: a system
whose generator is inferring contestedness from the passages would flip ~25 of
220 and be detected with certainty. **Any non-zero flip rate on feature 1 or 2
falsifies the answer policy**, regardless of how good the answers look.

Run the same protocol with the *evidence list order* held fixed but the
**question paraphrased** (5 paraphrases × 220 edges) as a second arm. Same null,
same falsifier. This costs 1,100 generations — hours of compute, no annotator
time, no GPU (the generator is an API call or a local model already running).

**This is the cheapest experiment in the document with the largest published
effect to detect. Run it first.**

### 4.5 The closed-book ablation, which is mandatory and will probably be
uncomfortable

§2.3 flagged *Knowledge-Graph Grounding Helps LLMs Only for Out-of-Training
Knowledge* ([arXiv:2606.22419](https://arxiv.org/abs/2606.22419), 2026-06-21):
the grounding benefit concentrates on facts the model does not already have. This
corpus is 271 published microbiome papers, most of them in any frontier model's
pretraining data. **The null hypothesis that the graph adds nothing on
directional questions is live and must be tested before anything is claimed.**

Four arms, same question set, exact-match scoring, no judge:

| arm | context given | what it isolates |
|---|---|---|
| **CB** closed book | none | parametric knowledge |
| **CH** chunk RAG | `rag_query.py` output (paper chunks) | text retrieval, the system that exists today |
| **KG** graph-grounded | §3 template rendering | the graph |
| **KG−** graph, counts stripped | template with `n_papers`/`n_up`/`n_down` removed | whether the *aggregate* is what helps, or just the fact list |

Two question families, generated programmatically from `graph.json` — no human
writes them:

- **D-questions (direction).** "In {disease}, is {taxon} reported enriched or
  depleted?" Scored 3-way: enriched / depleted / **contested-or-split**. 2,008
  available; sample stratified by state label.
- **A-questions (aggregate).** "How many papers in this corpus report {taxon}
  {direction} in {disease}?" and "Do any papers in this corpus report the
  opposite direction for {taxon} in {disease}?" Scored by exact integer / yes-no.
  **No pretrained model can know these** — they are properties of *this corpus*,
  assembled 2026, under *this* extraction gate.

**Pre-register the expected outcome**, because it is the honest one and
post-hoc rationalisation is the failure mode here:

> On **D-questions restricted to CONSISTENT-STRONG edges about famous pairs**
> (*Akkermansia*/Parkinson's, *Faecalibacterium*/IBD-adjacent), CB is expected to
> do well, possibly as well as KG. **That is predicted, not a failure**, and it
> must be reported rather than buried.
> On **D-questions restricted to SPLIT and UNREPLICATED edges**, CB is expected
> to confidently assert a direction that this corpus does not support. The
> measurable quantity is CB's rate of unhedged assertion on the 89 1–1 splits.
> On **A-questions**, CB is expected to be at chance and KG near-perfect.

**Falsifiers.**
- If **KG ≈ CB on A-questions**, something is broken in the harness — the model
  cannot know corpus-specific counts, so this outcome means the counts leaked
  into the prompt or the scoring.
- If **KG ≈ CH on both families**, the graph adds nothing over plain chunk
  retrieval and §7 should rank the graph QA layer below improving
  `rag_query.py`. This is a real possible outcome and it must be allowed to win.
- If **KG ≈ KG−**, the aggregate is not doing the work and the whole
  evidence-counting apparatus is decoration.
- If **CB unhedged-assertion rate on 1–1 splits is low** (say <20%), the
  headline framing "ungrounded models manufacture consensus" is not supported by
  this data and must be dropped.

State the MDE before running, per §1(c). At n=200 questions per arm and an
expected accuracy near 0.7, a paired two-sided test at α=0.05, 80% power detects
roughly an 8–10 point difference; anything smaller is out of reach and should not
be discussed. n=200 D-questions is a programmatic sample, so scale it to 800 if
compute allows and the MDE drops to ~4–5 points.

### 4.6 Where the human hours go — and only here

Budget: assume **20 hours** of biology-student time. Everything in §4.3–4.5 uses
zero. The hours buy one thing the machine cannot produce: **an external reference
for whether a cited paper actually supports the claim the answer attributes to
it.** That is AIS (Rashkin et al. 2021/2023) applied to this system's output, and
it is the only metric in this document that is not computed from `graph.json`.

**Protocol** (each design choice exists to block one contamination channel from
§4.2/R5):

1. Sample **(answer, cited paper, directional claim)** triples, stratified across
   the five state labels so the rare tiers are represented, not sampled in
   proportion.
2. The annotator sees **the claim and the paper, never the answer prose and never
   the graph record.** No `n_papers`, no state label, no system confidence. This
   blocks anchoring and blocks the reference from being contaminated by the
   system under test.
3. Present each claim alongside **one decoy** — the same taxon/disease with the
   direction inverted — in randomised order. The annotator marks
   supported / contradicted / not-addressed for each. The decoy gives a
   within-annotator specificity estimate for free, and catches acquiescence.
4. Record per-item time. Adjudicate a **20% overlap** between two annotators and
   report Cohen's κ. This repo has done exactly this before and it worked:
   the direction audit adjudicated all 28 residual disagreements twice, with
   25/28 exact agreement (`FINDINGS_direction_audit.md`). Reuse that protocol
   verbatim; it is already validated in this domain.

**Sample size, computed rather than guessed.** To estimate citation-support rate
within ±7 points at 95% confidence, assuming the true rate is near 0.90:
n = 1.96²·(0.9)(0.1)/0.07² ≈ **71 claims**. Within ±5 points: **139 claims**. At
~3 minutes per claim with the paper's PMC link pre-fetched, 139 claims is
**~7 hours**, plus ~1.4 hours for the 20% double-annotation. That fits the budget
with room to spare — so spend the surplus on stratification depth, not on a
larger single stratum.

**Then apply PPI** (Angelopoulos et al., Science 2023): label the remaining
thousands of triples with a machine checker, correct the machine's bias with the
139 human labels, and report an interval that is *valid* even if the machine
checker is mediocre. This is the ARES contribution, used without ARES's judge
training. It is what turns 7 hours of student time into a corpus-wide number with
an honest CI.

**Stated null and falsifier for this arm.**
> **Null / target:** citation-support rate ≥ 0.90, consistent with the extractor's
> already-measured reading fidelity of ≥86.6% [81.7, 91.3]
> (`FINDINGS_direction_audit.md`).
> **Falsifies "the RAG works":** a citation-support rate whose 95% CI lies
> **below** the extractor's fidelity interval. That would mean the answer layer is
> losing accuracy the extraction already achieved — attributing to paper *i* a
> claim that paper *j* made — which is a pure answer-layer defect and the one
> thing this instrument uniquely detects.
> A rate *at* the extractor's fidelity is the expected ceiling. The answer layer
> cannot be more accurate than the graph it reads; do not report it as if it
> could.

Note the asymmetry inherited from the repo's recall work and restated here
because it applies verbatim: the extraction gate (reported significance, main
text only, disease-vs-healthy-control) means an annotator **can confirm that a
paper supports a claim but cannot establish that it does not**, since the
sentence may be gated out rather than absent. Annotators must be given the gate
in writing, and "not-addressed" must be reported as its own category rather than
folded into "contradicted." The recall audit already manufactured misses by
skipping this step — 4 reported, 1 real.

### 4.7 The falsification summary

"The RAG works" is not one claim. It is four, and they fail independently:

| # | claim | instrument | falsified by | annotator hours |
|---|---|---|---|---:|
| C1 | the answer layer does not distort the graph | assertion suite, 2,008 edges (§4.3) | any A4/A5/A6/A7 violation | 0 |
| C2 | the answer is invariant to evidence order and question phrasing | order test, 220 contested × 2 orders + 5 paraphrases (§4.4) | any state-label or majority-direction flip | 0 |
| C3 | the graph contributes something the model does not already have | 4-arm ablation, D- and A-questions (§4.5) | KG ≈ CH on both families | 0 |
| C4 | citations attribute claims to the papers that made them | blinded AIS sample + PPI (§4.6) | support-rate CI below the extractor's fidelity CI | ~8 |

**C1 and C2 should be run this week.** They are free, they have enormous power
against published effect sizes, and they test the exact design decisions §3 makes.
C3 is a day of compute. C4 is the only one needing people, and it needs eight
hours, not tens.

### 4.8 Three things not to do

- **Do not build a RAGAS/ARES dashboard.** Four of the five scores would be
  measuring an exact template with a noisy model. If a number is wanted for a
  paper, report the assertion-suite pass rate and the AIS interval — both are
  stronger and neither can be hacked by verbosity.
- **Do not have the answering model grade its own answers.** Twice-demonstrated
  here (leak, then null) and twice-demonstrated in the literature
  (self-preference, Panickssery et al.). If a judge is needed for answer
  relevance, it is a different family and it runs in both orders.
- **Do not write a hand-labelled QA set of 50 questions.** It costs most of the
  annotator budget, it encodes the author's expectations as ground truth (the
  precise objection `FINDINGS_task2.5_graphrag.md` raises against hand-labelled
  relevance), and at n=50 its MDE is worse than the assertion suite's by two
  orders of magnitude. Spend the hours on C4, where the machine genuinely cannot
  substitute.

---

## 5. Query classes the repaired taxonomy tree unlocks — and where symbolic wins

### 5.0 What the repair actually bought, measured

Sam's ask was *"we can ask questions like you are asking at species level or genus
level or family level"*. `repair_taxonomy_tree.py` turned the 20-fragment forest
into a tree, and `FINDINGS_hyperbolic.md` then got a Poincaré embedding to recover
rank at ρ = +0.648. Both are real. Neither is the same thing as answering Sam's
question, and the tree — not the embedding — is what does the work.

Recomputed from `taxonomy_tree.json` + `graph.json` (2026-09-19):

| quantity | value |
|---|---:|
| tree nodes / edges / roots | 916 / 915 / **1** |
| observed taxa in the tree | **671** |
| inserted Steiner ancestors (no evidence of their own) | 245 |
| observed taxa by rank | 263 species, 242 genus, 80 family, 29 order, 20 phylum, 18 class, 11 no-rank, 5 strain, 1 each subspecies/kingdom/clade |
| observed taxa with ≥1 observed descendant | **220** (84 genus, 66 family, 28 order, 17 phylum, 15 class, 5 species, 4 no-rank, 1 kingdom) |
| median descendants among those 220 | 2 |
| **(ancestor, disease) pairs reachable by descendant expansion** | **1,454** |
| …where the ancestor **also** has a direct edge in that disease | 468 |
| …of those, ≥1 descendant **disagrees** with the ancestor's direction | **226 (48.3%)** |
| …**pure rollups**: ancestor has *no* direct edge, descendants do | **986** |
| total descendant-edge attachments | 6,699 |

Three readings of that table, in order of importance.

**(1) 986 pure rollups are the capability.** These are (taxon, disease) pairs
where the graph is silent at the rank you asked about but speaks at a rank below.
"What does this corpus say about *Lachnospiraceae* in Alzheimer's?" has a
different answer from "what does it say about the genera inside
*Lachnospiraceae* in Alzheimer's", and today the system can only answer the
first. This is ~49% again on top of the 2,008 direct edges — the largest single
expansion of answerable surface available, and it needs **no model, no GPU and no
new data**. It is a tree walk.

**(2) 48.3% is not a contradiction of the 25% already on record — different
denominator.** `FINDINGS_task2.5_graphrag.md` reports 903 (taxon, disease) edges
with a parent edge in the same disease, of which **229 (25%)** point the opposite
way. That counts **immediate parents inside the pre-repair observed graph**, edge
by edge. The 226/468 = 48.3% above counts **(ancestor, disease) pairs over the
full repaired ancestor set**, flagged if *any* descendant disagrees. Both are
correct; they answer different questions. Quote them with their denominators
attached or they will be read as a revision. Neither is a new finding about
biology — the repair widened the aperture, it did not change the graph.

**(3) 212 of 883 taxa (24%) are not in the tree at all.** They never resolved to a
taxid — 16S clade labels (`SMB53`, `cc115`, `PAC000195_g`) plus real taxa absent
from the cached taxdump. They carry **135 edges (6.7%)**, so they are low-degree,
but they are *completely invisible* to every tree method and to every geometric
method, because a node with no taxid has no lineage and therefore no position.
**No amount of embedding work reaches them.** Any claim of the form "rank-flexible
retrieval now works" must carry "for the 76% of taxa with an NCBI lineage."

And a rule that follows directly from the repair: **the 245 Steiner nodes are
scaffolding, never answers.** They exist to make lineages contiguous. A query that
lands on one must roll to its observed descendants and say so, never report the
Steiner node as a finding. `FINDINGS_hyperbolic.md` already applies this rule to
scoring (659 of 916 nodes scored); the QA layer needs the same rule at answer
time.

### 5.1 Why the symbolic baseline is not just competitive — it is the ceiling

This needs saying bluntly because "we used hyperbolic embeddings" sounds better
than "we walked a tree", and `00_SYNTHESIS.md` §3 already warned about exactly
that framing.

**The Poincaré embedding was trained on the containment tree and nothing else.**
Its training signal *is* the parent–child relation in `taxonomy_tree.json`. A
lossy reconstruction of a structure cannot contain more information than the
structure. So on any question whose ground truth is defined by ancestry —
descendants-of, ancestors-of, lowest-common-ancestor, rank-of — the tree walk is
**exact by construction**, and the geometry can at best tie and will in practice
lose to distortion. There is no experiment to run here; it is an information
argument.

The supporting numbers say the same thing. `FINDINGS_hyperbolic.md`: most of the
ρ = +0.648 is tree depth (depth↔rank is ρ = +0.721 *by construction*, since
`repair_taxonomy_tree.py` builds depth from NCBI lineages); controlling for depth,
radius retains only **ρ = +0.228** of rank. So the embedding's marginal
contribution over "read the depth off the tree" is modest, and its plausible
mechanism — smoothing over uneven branch lengths where lineages carry different
numbers of no-rank intermediates — is a problem the symbolic side can also fix,
by reading `rank` directly instead of using depth as a proxy. `rank` is stored in
`taxonomy_tree.json`. There is no need to infer it from anything.

Scale removes the other usual argument for geometry. 916 nodes, 915 edges. A full
descendant set is a recursive dict lookup; the computation above enumerated all
1,454 (ancestor, disease) pairs and all 6,699 attachments in well under a second
on a laptop. Approximate nearest-neighbour search exists because exact search is
expensive. Here it is not.

**There is a measured precedent for ontology layers adding nothing to retrieval in
this exact system.** `FINDINGS_disease_ontology.md`: the 2 MONDO-confirmed is-a
links add **no** retrieval reach in the PPR retriever — *measured, identical
subgraphs* — only explicit attribution. The disease hierarchy is small (40 nodes)
so that is unsurprising, but the lesson generalises: in this graph, structure that
is already reachable by other paths does not increase reach when you name it. Test
reach before assuming it.

So the bar for any geometric retriever, stated so it cannot be quietly lowered:

> **It must beat exact descendant expansion over a 916-node tree — a method that
> is instant, 100% precise, 100% recall on its own definition, and explainable as
> a printed lineage. Beating "nothing" is not the comparison.**

The only honest ways to clear that bar are to change the task so ancestry is not
the ground truth, or to bring in information the tree does not have (association
direction, evidence counts, paper text). Both are possible. Neither is what the
current embedding does.

### 5.2 The query classes, enumerated and adjudicated

Verdicts: **SYMBOLIC** = build it as a tree/dict operation, done;
**GEOMETRY** = geometry is a real candidate but must beat a stated symbolic
baseline; **NOT WORTH IT** = do not build.

| # | query class | example | mechanism | verdict |
|---|---|---|---|---|
| **Q1** | point lookup | "Is *Prevotella* depleted in PD?" | dict on `(taxon_key, disease)` + §3.2 state label | **SYMBOLIC** |
| **Q2** | **rank rollup / descendant expansion** | "What does this corpus say about *Lachnospiraceae* in AD?" | `desc(node) ∩ observed`, gather edges in that disease | **SYMBOLIC** — 1,454 pairs, **986 answerable no other way**. Highest value in the table. |
| **Q3** | **rank-conflict surfacing** | "Do the genera inside *Prevotellaceae* agree with the family in PD?" | Q2 + direction comparison | **SYMBOLIC** — 468 pairs have both levels, **226 (48.3%) disagree**. This is the containment story §1(d) identified and the one thing BM25 provably cannot do. |
| **Q4** | resolution selection | "What is the right rank to ask this at?" | homogeneity flag over descendants + permutation null (`00_SYNTHESIS.md` §8b, CBEA-style) | **SYMBOLIC**, but needs the refusal list: never aggregate when the edge is contested, when homogeneity fails, when n is small, or when children share provenance — and **never let aggregate evidence override direct evidence**. |
| **Q5** | aggregate / counting | "How many papers report this, and which way do they split?" | read `n_papers`, `n_up`, `n_down`, `ev[]` | **SYMBOLIC** — and per §2.3 this is the project's genuinely out-of-training content, so it is also the query class with the clearest grounding benefit. |
| **Q6** | cross-disease bridging | "What links PD and AD?" | PPR geometric-mean seeding (`graphrag.py`) or entity-filtered BM25 | **SYMBOLIC** — already built, and the premise that BM25 cannot do it was **falsified** at P@10 = 1.00. Do not re-litigate; do not run another 6-query comparison (§1(c), MDE ≈ 0.17). |
| **Q7** | taxonomic neighbourhood | "Taxa closely related to *Blautia*" | LCA distance on the tree | **SYMBOLIC** — exact, and the geometry is a lossy copy of it (§5.1). |
| **Q8** | disease-side rollup | "Everything under cognitive decline" | MONDO links | **NOT WORTH IT for retrieval.** Measured: no added reach, identical subgraphs. Ship the 2 is-a links as opt-in attribution (`disease_hierarchy_links.json`), link for retrieval, **do not pool evidence**, **do not fold MCI into Alzheimer's** — the 71-paper cognitive cluster does not cohere microbially (0.592 vs 0.672 background). |
| **Q9** | disease similarity | "Which diseases share a microbial signature?" | directional agreement over shared taxa | **SYMBOLIC**, with the correction already on record: pooled numbers only — **0 of 16 pairs survive BH at q=0.05**, including AD/Dementia at 0.938 (p=0.030). |
| **Q10** | **out-of-vocabulary entity linking** | user types "Faecalibacterium prausnitzi", "*E. coli*", "Bacteroidetes" | fuzzy / embedding match from query string → node | **GEOMETRY — and this is the one with real evidence behind it.** See §5.3. |
| **Q11** | conflict-severity ranking | "Show me the worst rank conflicts, ordered" | evidence-weighted + tree-distance-weighted score | **GEOMETRY optional.** A symbolic severity score (min paper count on each side × tree distance) is available today and must be the baseline. `FINDINGS_hyperbolic.md` item 2 proposes the geometric version; it is unstarted and it is *not* obviously better. |
| **Q12** | the 212 unresolved taxa | "*SMB53* in PD?" | none of the above | **GAP, not a method choice.** No lineage → no tree position → no embedding. Needs curation or a taxdump refresh, not modelling. 135 edges. |
| **Q13** | "why do studies disagree about X?" | — | — | **NOT WORTH IT.** 24 study-design variables, 24 nulls; paper-level SD of discordance is 3.4 points on a 27.6% base against MDEs of 4–7. The system must **report** the disagreement (§3.3) and must **not** attempt to explain it. Any QA feature that offers an explanation is fabricating one. |
| **Q14** | effect size / magnitude | "How much is it depleted?" | — | **NOT WORTH IT — permanently, at this schema.** Direction only; incommensurable source statistics; Q/τ²/I² mechanically uncomputable from direction counts. Assertion A11/A12 in §4.3 enforces this. |

Ten of fourteen classes are SYMBOLIC. Two are GEOMETRY candidates and only one of
those has supporting evidence. One is a data gap. One is forbidden.

### 5.3 The one place an embedding clearly earns its place — and it is not the tree

**Q10, out-of-vocabulary entity linking.** `graphrag.py` links queries to nodes by
**exact longest-match** against 832 taxon aliases and 43 disease surface forms.
Exact match fails on the things users actually type: misspellings, abbreviations,
rank prefixes (`g__Blautia`), pre-rename synonyms, and species names when the
graph holds the genus.

The local evidence that this — not retriever architecture — is where the wins are
is already in the repo and is quoted in §1: `rag_query.py`'s docstring records
that querying a **variable name** retrieves junk at top cosine 0.347 while
querying **a sentence a paper would contain** retrieves 4/4 correct at 0.707.
That is a vocabulary-mismatch result, and it is the largest measured effect
anywhere in this project's retrieval work — about twice the size of the entire
GraphRAG-vs-BM25 gap, which was noise.

Note precisely what kind of embedding this calls for: a **name/string** embedding
(the retrofitting route in `00_SYNTHESIS.md` §4c, Faruqui et al. 2015 — closed
form, no retraining, no GPU; SapBERT/BioLORD train on *synonymy*, which is exactly
the right relation for this task even though `00_SYNTHESIS.md` correctly notes it
is the wrong relation for hierarchy). It is **not** the Poincaré tree embedding.
Conflating "we have a hyperbolic embedding" with "we can do fuzzy entity linking"
would be a category error, and the two are solving unrelated problems.

Falsifiable test, cheap and honest: take the **33 misspellings already verified to
occur verbatim in their source papers** (`FINDINGS_taxon_spelling.md`,
`taxon_typos.py`) plus the 12 punctuation-split concepts as a held-out linking set
with known correct targets. Measure top-1 linking accuracy for (a) exact match —
which scores ~0 by construction, since these are the strings that failed —
(b) character n-gram similarity, (c) a name embedding. **This is a real labelled
set that already exists, was built for another purpose, and cost nothing.**
The refusals matter too: `taxon_typos.py` records 13 deliberate non-merges, and
any method that merges `Oscillospirales` into `Oscillospira` fails regardless of
its accuracy number. Report accuracy *and* the refusal violations.

### 5.4 What the tree does not fix

Stated so §7 does not over-promise:

- **It does not make single-paper edges more reliable.** 1,561 of 2,008 edges rest
  on one paper. Rolling a family up over three genera each with one paper gives a
  family-level statement backed by three papers *that never examined the family*.
  Q4's refusal list exists for this; without it, Q2 manufactures confidence.
- **It does not resolve contested edges.** Ancestry has no opinion on direction.
- **It does not extend recall.** The gate is unchanged: reported significance,
  main text, disease vs healthy control. Rollup reaches edges the extractor
  already found, at ranks it already assigned.
- **It does not touch the 24% of taxa with no lineage.**
- **The 89% within-paper concordance among related taxa cuts against novelty
  here.** Related taxa in a single paper agree on direction 89% of the time
  (vs 54% for unrelated). So most of what Q2 rolls up is *redundant* with the
  direct edge where one exists — which is precisely why the **986 pure rollups**,
  where no direct edge exists, are the part worth building, and why the **226
  disagreements** are the part worth surfacing. The agreeing middle is the least
  interesting output the feature produces.

---

## 6. What NOT to build

Six directions that sound reasonable, cost weeks, and would each be a mistake.
Each is stated with the reason it fails *here* specifically — not a general
objection to the technique.

### 6.1 Do not fine-tune anything on 271 papers

The pitch: LoRA a 7B model on the corpus, or on the 2,008 edges, so it "knows the
microbiome literature."

**Why it fails.**

- **Scale.** The closest method sibling, MINERVA (Briefings in Bioinformatics
  2025), fine-tunes Biomistral-7B over **129,719 papers** → 66,400 edges. This
  corpus is **271 papers / 2,008 edges** — roughly 1/478 the documents. There is
  no reading of the fine-tuning literature in which that is enough to teach a
  domain rather than to memorise a sample.
- **It destroys the asset.** This project's differentiator is not that a model
  can recite microbe–disease pairs; it is **per-paper provenance, direction
  counts, and an explicit contested flag**. Baking the graph into weights deletes
  all three. A fine-tuned model cannot say "11 papers depleted, 2 enriched, here
  are the titles" — it can only produce a direction, which is the averaging
  operation this project has refused since `mega_dump/proj_2`.
- **It makes the graph un-updatable.** The graph has had five structural
  corrections and three corpus revisions in six weeks (punctuation fold, spelling
  fold, species split, deduplication, MONDO id fixes). LightRAG's stated
  contribution is *incremental update without reindexing*; a fine-tune is the
  opposite — every correction invalidates the weights.
- **The repo already measured the ceiling and it is not the model.** eval-v2:
  RELATE backfired on every local model, grounded prompting helped only the
  under-extractor, normalize moved F1 by 0.00, and **no method beat plain
  single-shot.** The recorded conclusion is *"the benchmark's completeness is the
  ceiling, not the prompt."* Adding a fine-tune to a pipeline whose measured
  bottleneck is the reference set is spending compute on the wrong term.
- **It cannot be evaluated.** Held-out-paper generalisation at n=271 with a
  neuro-skewed disease mix has no defensible split. Whatever number came out
  would be a leak or a coin flip, and §4.2/R5 forbids reporting it.

**What would change this:** a corpus in the tens of thousands of papers. That is
a corpus-scale GPU question, not a modelling question, and it belongs in
`203_corpus_scaling.md`, not here.

### 6.2 Do not run community detection + community summarization

The pitch: Microsoft GraphRAG's global mode — Leiden communities, an LLM summary
per community per level, map-reduce over summaries.

**Why it fails.** Three independent reasons, any one sufficient:

1. **The communities are already known.** The graph is near-bipartite
   taxon↔disease with 923 nodes and 2,735 edges. Leiden will recover
   approximately one community per disease, which is `e["disease"]` — a field
   that already exists. *Falsifiable version, and it is worth actually running
   because it is twenty minutes:* compute adjusted mutual information between the
   Leiden assignment and the disease label of each edge. **AMI > 0.7 ⇒ community
   detection discovered nothing the dict lookup does not already give**, and the
   direction is dead. If it comes back low and the communities are *cross-disease
   taxon guilds*, that is a genuine finding and §7 should be revised. I expect
   high.
2. **A community summary averages.** That is what a natural-language summary of a
   set of contradictory findings *is*. It is the single operation this project
   has decided is forbidden, and it would silently destroy the 220 contested
   edges and the 1,560 unreplicated ones — the whole point of §3.
3. **Pre-generated summaries go stale invisibly.** `FINDINGS_task2.5_graphrag.md`
   records the exact failure mode already: `rag_corpus.jsonl` was scored for
   weeks while disagreeing with `graph.json` on **293 documents** (148 phantom,
   145 missing), and it silently reversed the sign of a headline comparison. A
   cache of LLM-written prose is far harder to set-equality-check than a JSONL of
   edges. Do not add a second, less checkable cache.

GraphRAG's value proposition is explicitly that the corpus is too large to read —
Edge et al. evaluate "in the 1 million token range." This graph's entire edge set
renders to a few hundred KB. **The premise does not hold.**

### 6.3 Do not train a learned reranker

The pitch: a cross-encoder over retrieved edges, trained to rank.

**Why it fails.**

- **There is no relevance label and there is no honest way to make one.** Labels
  derived from `graph.json` make the graph both the training target and the
  system under test — the circularity §4.0 names. Hand-labelled relevance
  "would just measure my own expectations", which is
  `FINDINGS_task2.5_graphrag.md`'s own stated reason for *not* doing it.
- **The headroom is unmeasurable, so the objective is undefined.** GraphRAG vs
  BM25: 0.683 vs 0.700, exact sign-flip permutation over all 2⁶ assignments
  **p = 1.000**, MDE ≈ 0.17 against an observed gap of 0.017. You cannot train
  toward a difference the evaluation cannot resolve, and you certainly cannot
  demonstrate you succeeded.
- **Cross-encoders need order-10⁴ labelled pairs.** The available budget is tens
  of hours of biology-student time, which §4.6 spends on the one measurement no
  machine can make.
- **Ranking is not the bottleneck.** §1(d) and §5.2 both land on the same place:
  the defensible graph-only capability is *reach* (containment traversal, 986
  pure rollups), not *order*. A reranker improves order.

### 6.4 Do not build a chat UI before the answer policy is settled

The pitch: a conversational front end so biologists can explore the graph.

**Why it fails — and this is the least obvious item on the list.** A chat
interface is where the hedging gets stripped, by three mechanisms:

1. **Follow-up pressure.** The natural second turn after a correct SPLIT answer
   is *"okay but which is more likely?"* Every conversational system is trained
   to be helpful to that, and being helpful to it means manufacturing a winner
   from 1 paper vs 1 paper. §3.1 rejected multi-agent debate for precisely this
   reason — debate adjudicates, and this corpus cannot. A chat loop is debate
   with the user as the second agent.
2. **Order effects reopen.** §4.4's order-invariance test closes the
   11.4%–25.2% flip risk (Han et al.) for *single-turn, template-rendered*
   answers, because the state is computed before generation. In multi-turn, the
   context window accumulates evidence in an order the system no longer controls,
   and the invariance guarantee — and the test that verifies it — no longer
   applies. The assertion suite in §4.3 is also single-answer; it has no defined
   behaviour over a transcript.
3. **It inverts the build order.** The UI is a renderer for the answer policy. If
   the policy lands first, the UI is a week of front-end work over a tested core.
   If the UI lands first, the policy gets negotiated against interaction design,
   and the five states become a confidence bar.

**Build order:** answer policy (§3.2) → assertion suite (§4.3) → order test
(§4.4) → *then* an interface, and make it single-turn with a visible state badge
before it is a chat.

### 6.5 Do not attach a numeric confidence score to each edge

The pitch: a 0–1 calibrated probability per edge, so the UI can show a bar.

**Why it fails.** `00_SYNTHESIS.md` §6 settles this and the verdict is stronger
than "not yet":

- Vote counting — which *is* this project's edge weight — has power that
  **decreases** as studies accumulate (Hedges & Olkin 1980, Psych Bull
  88(2):359–369).
- The calibration that exists is coarse and already honest: 2 studies → ~50%,
  3 → ~67–70%, 4+ → ~90%. **Only 43 edges (2.1%) reach the 4+ tier.**
- Heterogeneity statistics are **mechanically uncomputable** from direction
  counts. This is foreclosed permanently at this schema, not pending more data.
- UKGE-style graded confidence collapses two separately-true relations into one —
  the averaging error again.

The §3.2 five-state label *is* the confidence representation, and it is the most
defensible one the data supports. A decimal would be invented precision, and
assertion A11 exists to catch it.

### 6.6 Do not build an agentic / iterative multi-hop retrieval loop

The pitch: a planner that issues successive retrievals, reflects, and re-queries.

**Why it fails.** The graph is **two hops deep**: taxon→disease, and
taxon→ancestor. Every query class in §5.2 is answered by one traversal plus one
set operation; the computation enumerating all 1,454 ancestor–disease pairs ran in
under a second. Iterative retrieval machinery exists to handle graphs where the
relevant subgraph cannot be identified in one shot, and PathRAG's flow-based
pruning exists to handle context overflow — **neither condition occurs here**
(§2.2). Meanwhile each additional LLM turn is another exposure to the order and
verbosity effects of §4.2, and a further step away from the assertion-checkable
template that makes §4 cheap. It adds latency, variance and audit surface in
exchange for hops that do not exist.

### 6.7 Already ruled out elsewhere — do not revisit without new evidence

Listed so they are not rediscovered as fresh ideas:

| direction | ruled out in | one-line reason |
|---|---|---|
| GNN over the KG | `00_SYNTHESIS.md` §2; §2.2 here | KATZHMDA (zero learning) scores 0.84–0.86; published evals use leaky random-negative CV; no external features available |
| GNN-RAG | §2.2 | needs thousands of labelled question→answer pairs |
| HippoRAG 2 port | §2.2 | PPR is already implemented (`graphrag.py`) and tied at p = 1.000; the delta is LLM triple-filtering over 2,008 curated edges |
| RAPTOR over the KG | §2.2 | replaces a 100%-precision NCBI hierarchy with a guessed one |
| multi-agent debate | §3.1 | adjudicates; this corpus cannot decide, so debate manufactures a winner |
| open-schema re-extraction | root `CLAUDE.md`; §2.2 | tried in `mega_dump/proj_2` at **6% precision**, 88% FP |
| another 6-query retrieval comparison | §1(c) | MDE ≈ 0.17 vs a 0.017 gap; p = 1.000 |
| explaining *why* papers disagree | §5.2 Q13 | 24 variables, 24 nulls; paper-level SD 3.4 points on a 27.6% base |
| pooling MCI into Alzheimer's | `FINDINGS_disease_ontology.md` | cluster does not cohere microbially (0.592 vs 0.672); 0 of 16 pairs survive BH |

---

## 7. Recommendation

Ranked by value per hour of the resource that is actually scarce. Per
`208_extraction_frontier.md` §1.2 that resource is **adjudication-hours, not
GPU-hours** — a full corpus extraction pass is ~1.7 GPU-hours, so compute is
effectively free here and the ranking is driven by the numerator and by human
time. Eight of the nine items below use **zero** annotator hours.

The shape of the recommendation follows from §5.2: **ten of fourteen query
classes are symbolic**, one is a data gap, one is forbidden, and only one
geometry candidate has evidence behind it. So the build is mostly a careful
dict-and-tree layer with a strict answer policy on top — not a retrieval
architecture.

### R1 — Build the symbolic answer layer (Q1, Q2, Q3, Q5) + §3.2 state labels

**Do this first and possibly only this.** Q2 alone covers 1,454 taxon–disease
rollup pairs of which **986 are answerable no other way**, and Q3 surfaces the
468 pairs where a family and its genera are both observed — **226 (48.3%)
disagree**. That is the containment story this project has protected since the
beginning, and it is the one thing a flat retriever provably cannot produce.

Effort: ~2–3 days of engineering. Zero annotator hours. No GPU, no embedding,
no index.
**Falsifiable test:** every answer is a template fill over `graph.json` fields
(§3.3), so correctness is decidable by assertion, not judgment — see R2.
**What kills it:** nothing plausible. If the symbolic layer is wrong, it is a bug,
not a failed hypothesis. This is the low-risk core and everything else is
optional relative to it.

### R2 — Ship the assertion suite (§4.3) as CI, before any answer reaches a user

A1–A12 run over all 2,008 edges with no model in the loop.

> **Null:** zero violations.
> **Falsifies "the RAG works":** any violation of **A4, A5, A6, A7** — the four
> that encode the answer policy. A system failing those is not a weaker version
> of §3; it is a different system that happens to cite papers.

Effort: ~1 day. Zero annotator hours. This is the highest value-per-hour item in
the document and it is worth stating why: it converts the project's central
design commitment — never average, never assert a direction on a contested edge —
from a norm someone must remember into a test that fails the build.

### R3 — Run the order-invariance test (§4.4)

The highest-value *single experiment* here. Permute the evidence order fed to the
answer layer and assert the answer's directional content is unchanged. A system
whose verdict depends on which paper it read first is not reporting the
literature.

Effort: hours. Zero annotator hours.
**What kills it:** order-dependence that survives a fix. That would mean the
answer layer is doing inference rather than reporting, and §3's policy would have
to be rebuilt.

### R4 — Run the closed-book ablation (§4.5), and report it whichever way it lands

Mandatory, and **expected to be partly unflattering**. On famous pairs
(*Akkermansia*/Parkinson's) a closed-book LLM will likely match the KG. §4.5 says
that is predicted, not a failure, and must be reported rather than buried.

The number that justifies the whole project is a different one: **closed-book
rate of unhedged assertion on the 89 1–1 split edges.** A model with no access to
provenance will confidently pick a side the corpus does not support; the KG
answers "the literature does not agree". That contrast is the product.

Effort: ~1 day. Zero annotator hours.
**What kills it:** closed-book matching the KG on split and unreplicated edges
*and* hedging appropriately. Unlikely, but it would mean the graph's marginal
value is limited to attribution.

### R5 — Spend the 20 annotator-hours on AIS citation support (§4.6), and nowhere else

The only quantity in this document not computable from `graph.json`: does the
cited paper actually support the claim attributed to it? Protocol is already
specified — annotator sees claim and paper but never the answer prose or the
graph record, one inverted decoy per item, 20% double-annotated for κ. That
design is not new here; it is the direction audit's protocol, already validated
in this domain at 25/28 exact agreement.

> **Target:** citation-support ≥ 0.90, consistent with measured reading fidelity
> ≥86.6% [81.7, 91.3].
> **Falsifies "the RAG works":** a rate whose 95% CI lies **below** the
> extractor's fidelity interval — the answer layer losing accuracy extraction had
> already achieved, i.e. attributing paper *i*'s claim to paper *j*. That is a
> pure answer-layer defect and the one thing this instrument uniquely detects.

Effort: 20 annotator-hours, ±7 points at 95%.

### R6 — Entity linking for out-of-vocabulary queries (Q10) — the one geometry item

The only place §5.3 finds real evidence for an embedding, and the evidence is
local: querying a variable name retrieves junk at cosine 0.347 while querying a
paper-like sentence retrieves 4/4 at 0.707 — a vocabulary-mismatch effect about
twice the size of the entire GraphRAG-vs-BM25 gap, which was noise.

Use a **name/string** embedding (retrofitting, or SapBERT/BioLORD, which train on
synonymy — the right relation for this task). **This is not the Poincaré tree
embedding**, and treating "we have a hyperbolic embedding" as "we can do fuzzy
linking" is a category error.

**Falsifiable test, on a labelled set that already exists and cost nothing:** the
33 misspellings verified verbatim in their source papers plus the 12
punctuation-split concepts. Measure top-1 linking for exact match (~0 by
construction), char-n-gram, and a name embedding.
**What kills it:** char-n-gram matching the embedding. Then ship the cheap one.
**Hard constraint:** `taxon_typos.py` records 13 deliberate non-merges. Any
method that merges `Oscillospirales` into `Oscillospira` fails regardless of its
accuracy number. Report accuracy **and** refusal violations.

### R7 — Close out community detection empirically (20 minutes)

§6.2 argues Leiden will recover approximately one community per disease — a field
that already exists. Rather than argue, measure: **AMI between the Leiden
assignment and `e["disease"]`. AMI > 0.7 ⇒ the direction is dead.** If it comes
back low and the communities are cross-disease taxon guilds, that is a genuine
finding and this section should be revised.

Effort: 20 minutes. Worth it purely to convert an argument into a number.

### R8 — Conflict-severity ranking (Q11), symbolic baseline only

Rank the 226 rank-conflicts by `min(paper count on each side) × tree distance`.
That baseline is available today. The geometric version proposed in
`FINDINGS_hyperbolic.md` item 2 is unstarted and **not obviously better** — it
must beat this, not beat nothing.

### R9 — Do not build, in priority order of temptation

| direction | why | ref |
|---|---|---|
| fine-tune anything | 271 papers vs MINERVA's 129,719; deletes provenance, counts, contested flag; un-updatable; unevaluable at n=271 | §6.1 |
| community summarization | a summary of contradictory findings **is** the averaging operation this project forbids | §6.2 |
| learned reranker | no labelled relevance data | §6.3 |
| chat UI before the answer policy is settled | ships the averaging failure behind a nice interface | §6.4 |
| numeric confidence per edge | measurably miscalibrated (ECE 0.32 vs 0.07 floor); collapses `contested` — a *refusal to assert* — into ~0.5, which reads as "probably true, weakly" | §6.5, `206` §7 |
| agentic multi-hop loop | RELATE failed for the adjacent reason: weak local models handle extra decision points badly | §6.6 |
| another retrieval comparison | MDE ≈ 0.17 vs a 0.017 gap; p = 1.000 | §1(c) |
| explaining *why* studies disagree | 24 variables, 24 nulls; SD 3.4 points on a 27.6% base vs MDEs 4–7. The system must **report** disagreement and must not explain it. Any feature offering an explanation is fabricating one. | Q13 |
| effect sizes | direction only; source statistics incommensurable; Q/τ²/I² mechanically uncomputable. Permanent at this schema. | Q14 |

### The one-line version

**Build R1 and R2. Run R3, R4, R7 — they are hours, not days. Spend the 20
annotator-hours on R5 and nothing else. Treat R6 as the only geometry work worth
starting, and only against the free labelled set.** Everything in R9 is a way of
spending weeks to make the graph worse at the thing it is uniquely good at.

The deeper point, and it is the document's: this project's asset is not retrieval
quality. It is **per-paper provenance, direction counts, and an explicit refusal
to assert on contested pairs**. Almost every fashionable addition to a RAG stack
degrades at least one of those three. The recommendation is therefore mostly
restraint, enforced by R2 so restraint does not depend on memory.
