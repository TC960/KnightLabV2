# Explainable link prediction / disagreement explanation for the microbe–disease KG

**Scope of this note.** The PI's goal is explainability a biologist can read, not raw predictive
accuracy. The graph is small (883 taxa, 40 diseases, 2,008 direction-only edges, 220 contested)
and the thing that needs explaining is *disagreement between papers on an edge's direction*, not
a missing link. Two prior attempts at this — 26 study-design variables under cluster-robust
permutation + BH (best FDR 0.234, `FINDINGS_paper_discordance.md`), and 36 named study-design
concepts scored against paper embeddings via a stratified Mann-Whitney contrast
(`contrast_experiment.py`, best q ≈ 0.41) — both came back null. This note surveys the
explainability literature to (a) place those two attempts on a map of methods, (b) diagnose
whether the second one (which is a crude TCAV) failed for a fixable reason, and (c) recommend
what to try next given the graph's actual size.

---

## 1. Path-based explanation (PRA, DeepPath, MINERVA, AMIE, AnyBURL)

These are the methods whose *native output* is a sentence a domain expert can read without a
second translation step: "A relates to B via path/rule P."

**Path Ranking Algorithm (PRA)** — Lao & Cohen, *"Relational retrieval using a combination of
path-constrained random walks,"* 2010, and Lao, Mitchell & Cohen, *"Random Walk Inference and
Learning in A Large Scale Knowledge Base,"* EMNLP 2011. PRA enumerates bounded-length relation-type
paths connecting instances of a target edge type, uses random-walk-with-restart probabilities along
each path type as a feature, and fits logistic regression over those features to score candidate
edges. **Output:** a ranked list of path *types* (sequences of edge labels) with regression
weights — directly readable as "the edge is likely because paths of type X, Y are frequent between
positive pairs." **Human-readable without translation:** yes, though the weights are a linear
model over paths, so the "why this instance" story is "which paths of positive weight actually
exist between these two specific nodes," which requires one more lookup. **Minimum scale:** works
on graphs far smaller than deep methods — PRA's own experiments ran on NELL and Freebase subsets,
but the algorithm is just counting-based random walks over relation types, so it degrades
gracefully; the practical floor is having enough co-occurring path types to fit a stable regression
per edge type, typically hundreds of positive/negative instances per relation, not millions.
([Semantic Scholar](https://www.semanticscholar.org/paper/19acbce6e3c26e600368d84b38ae770eb0380c10),
[ProPPR / PRA follow-up, arXiv:1404.3301](https://arxiv.org/pdf/1404.3301))

**DeepPath** — Xiong, Hoang & Wang, *"DeepPath: A Reinforcement Learning Method for Knowledge Graph
Reasoning,"* EMNLP 2017. A policy-gradient (REINFORCE) agent walks the KG in embedding space, picking
relations to extend a path from source to target, with a reward balancing accuracy, path diversity
and path efficiency. **Output:** one or more discovered relation-paths per predicted fact — same
readable form as PRA, but the paths are found by a trained agent rather than enumerated, so it
scales to bigger graphs and finds longer/rarer paths PRA would miss. **Human-readable:** yes,
same caveat as PRA (path type, not necessarily grounded in the query pair without a check).
**Minimum scale:** needs enough graph structure and training signal (positive path traces) to
train the RL policy — the paper uses NELL and FB15k-scale graphs (tens of thousands of entities);
below that, the policy has too little exploration surface to learn anything better than PRA's
enumeration. ([arXiv:1707.06690](https://arxiv.org/pdf/1707.06690),
[ACL Anthology D17-1060](https://aclanthology.org/D17-1060/))

**MINERVA** — Das et al., *"Go for a Walk and Arrive at the Answer,"* ICLR 2018. Also
REINFORCE-trained, but answers a query (source, relation, ?) by walking to the answer directly
rather than scoring a precomputed path between a *known* pair — i.e., it does query answering, not
just path scoring, and needs no path features precomputed offline. **Output:** the actual walked
path (sequence of (relation, entity) hops) taken to reach the predicted answer — directly readable,
and unlike DeepPath it is *the actual trace for this specific prediction*, not a path type mined
from many instances. **Minimum scale:** similar or larger than DeepPath (FB15k-237, WN18RR, NELL-995
scale); needs the reward signal (successfully reaching known correct answers during training) to be
dense enough to bootstrap the policy — sparse-reward RL is exactly why several follow-ups (e.g.
reward shaping, arXiv:1808.10568) exist. ([ACL Anthology D18-1362](https://aclanthology.org/D18-1362.pdf))

**AMIE / AMIE+** — Galárraga, Teflioudi, Hose & Suchanek, *"AMIE: Association Rule Mining under
Incomplete Evidence in Ontological Knowledge Bases,"* WWW 2013; *"Fast Rule Mining in Ontological
Knowledge Bases with AMIE+,"* VLDB Journal 2015. Top-down, confidence-driven search over Horn-rule
refinements (e.g. `marriedTo(x,y) ∧ livesIn(y,z) ⇒ livesIn(x,z)`) that explicitly handles the
open-world assumption (no true negatives) via a PCA-confidence metric. **Output:** literal Horn
rules with head/body predicates, support and confidence — the single most legible artifact of any
method surveyed here; nothing to decode. **Minimum scale:** rule mining is a counting/enumeration
method, not gradient-trained, so it has no minimum data-scale requirement in principle; it does
need enough repeated relation *instances* to make a rule's confidence statistically meaningful
(a rule fired twice is not trustworthy regardless of graph size). ([GitHub](https://github.com/dig-team/amie),
[AMIE paper PDF](https://resources.mpi-inf.mpg.de/yago-naga/amie/amie.pdf))

**AnyBURL** — Meilicke, Chekol, Ruffinelli & Stuckenschmidt, *"Anytime Bottom-Up Rule Learning for
Knowledge Graph Completion,"* IJCAI 2019; extended in *"Reinforced Anytime Bottom-Up Rule
Learning,"* arXiv:2004.04412. Instead of AMIE's top-down search, AnyBURL samples random paths from
the graph and generalizes them bottom-up into Horn rules, anytime (can be stopped early with a
best-effort rule set). Outperforms AMIE+ on standard KGC benchmarks while being markedly faster and
lighter, and — because it is sampling-based — scales down as easily as it scales up: it has been run
successfully on graphs orders of magnitude smaller than YAGO/Freebase. **Output:** same as AMIE,
literal rules with confidence, plus (in later work) the specific rule instantiation that fired for
a given prediction. Biomedical applications exist directly on point: AnyBURL rules have been
combined with embedding models (TransE/DistMult/ComplEx/RotatE) and a confidence-calibration
step to screen "critical mechanistic paths" for drug-disease association, and a 2024 Scientific
Reports paper, *"Explainable drug repurposing via path-based knowledge graph completion,"* runs
exactly this recipe on Hetionet (the Rephetio drug-repurposing KG) — the closest published analogue
to what this graph would need. ([AnyBURL site](https://web.informatik.uni-mannheim.de/AnyBURL/),
[IJCAI19 PDF](https://web.informatik.uni-mannheim.de/AnyBURL/meilicke19anyburl.pdf),
[Sci Reports 2024](https://www.nature.com/articles/s41598-024-67163-x),
[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258358/))

**Relevance to the disagreement question, specifically:** path/rule methods explain *why an edge
should exist*, not *why two papers disagree on its sign*. To use them for the contested-edge
problem, the graph would need to encode paper-level context as graph structure (e.g., a paper node
with edges to its study-design attributes, connected to the taxon–disease edge it reports) so that
a rule could read "up-papers connect to `recruitment=hospital`, down-papers don't" as a literal
Horn rule instead of a numerical contrast. That reframing is itself the main design idea worth
testing (see recommendation).

---

## 2. Attention-based explanation — and why it's a weak claim

Attention weights over a GNN's neighbors (or a relation-path encoder's attention over path steps)
are routinely shown as bar charts implying "the model looked here." The literature has a real,
unresolved fight about whether that is a legitimate explanation:

- **Jain & Wallace, "Attention is not Explanation,"** NAACL 2019
  ([arXiv:1902.10186](https://arxiv.org/abs/1902.10186)). Tested attention on several NLP tasks and
  found (a) attention weights correlate weakly with gradient-based feature-importance scores, and
  (b) "adversarial" attention distributions — permuted or substituted attention weights very
  different from the learned ones — can produce nearly identical model outputs. Their conclusion:
  attention weights are not uniquely tied to the prediction, so pointing at them as *the* reason is
  unjustified — many different attention patterns explain the output equally well, meaning the
  weights are not identifiable as "the" explanation.
- **Wiegreffe & Pinter, "Attention is not not Explanation,"** EMNLP 2019
  ([arXiv:1908.04626](https://arxiv.org/pdf/1908.04626)). Rebuttal: the claim depends entirely on
  the definition of "explanation." If "explanation" means the unique causal reason, attention fails
  the bar few methods pass; if it means a *plausible* post-hoc account consistent with the model's
  actual computation, attention can pass a weaker but still useful bar. They propose four
  diagnostics (uniform-weight baseline, seed-variance calibration, frozen-attention training,
  adversarial-attention training) to test whether attention in a *given* trained model is doing
  real work before trusting it, rather than treating "is/isn't explanation" as a universal verdict.

**Practical read for this project:** attention is not free explainability — it requires the
Wiegreffe/Pinter diagnostics to earn trust *per model*, and even then it answers "where did the
model attend," not "what causal factor drove the label," which is what the PI's disagreement
question needs. Given the graph's size (2,008 edges), there also is not enough data to train an
attentional GNN with any confidence its attention patterns are stable across seeds — the debate's
premises assume large supervised training sets, which this project does not have. **Recommendation
weight: low**, and not just because of the debate — because there is no attention model here at all
yet, and the debate says building one would buy a contestable explanation even if training worked.

---

## 3. Post-hoc graph explainers (GNNExplainer, PGExplainer, SubgraphX)

These take a **trained** GNN as given and, for one prediction, find the minimal subgraph/features
"responsible" for it — the graph analogue of LIME/SHAP.

- **GNNExplainer** — Ying, Bourgeois, You, Zitnik & Leskovec, NeurIPS 2019
  ([arXiv:1903.03894](https://arxiv.org/abs/1903.03894)). Model-agnostic; for a given node/edge/graph
  prediction, learns a soft mask over edges (and optionally node features) that maximizes mutual
  information between the masked subgraph and the model's prediction. **Output:** a small
  edge-weighted subgraph — human-readable as "these N edges, in this rough proportion, drove the
  call" but the story is still "the model attended to this local structure," not a causal or
  mechanistic account; a domain expert has to interpret the subgraph themselves. **Known failure
  modes:** optimizes fidelity to the *model*, not to ground truth, so it can produce a
  confident-looking subgraph that is unfaithful to how the model actually reasons; reported to work
  well for node classification but underperform for graph-level tasks; requires re-solving the
  mask optimization per instance, so it does not scale to explaining many predictions at once.
- **PGExplainer** — Luo et al., NeurIPS 2020
  ([arXiv:2011.04573](https://arxiv.org/pdf/2011.04573)). Trains a single parametric (MLP) explainer
  network across *many* instances instead of re-optimizing per instance, so it generalizes and is
  fast at inference — but that also means it inherits whatever spurious correlations the trained
  explainer network itself picks up, and it **fails when the underlying classifier relies mainly on
  node features rather than structure** (its whole mechanism is edge selection). Reported to be the
  most *stable* explainer across runs but not the most accurate.
- **SubgraphX** — Yuan et al., *"On Explainability of Graph Neural Networks via Subgraph
  Explorations,"* ICML 2021 ([arXiv:2102.05152](https://arxiv.org/pdf/2102.05152)). Uses Monte Carlo
  Tree Search to directly search the space of connected subgraphs, scored with a game-theoretic
  (Shapley-like) value — the most accurate and most "reliable" of the three in most surveyed
  comparisons, but MCTS over subgraphs **does not scale to large graphs** and is the slowest of the
  three at inference time.

**Faithfulness is itself contested.** Multiple recent surveys and benchmark papers — Amara et al.,
*"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for GNNs,"* LoG 2022; Faber,
K. Amara et al., *"Evaluating Explainability for Graph Neural Networks,"* Scientific Data 2023
([nature.com](https://www.nature.com/articles/s41597-023-01974-x)); *"Probing GNN Explainers,"*
2021 ([arXiv:2106.09078](https://arxiv.org/pdf/2106.09078)) — find that standard fidelity metrics
(Fid+, Fid−, Fid_Δ) **disagree with each other**, that explainers are measurably less faithful on
heterophilic graphs (structure where connected nodes often differ, which describes a
taxon–disease bipartite-ish graph reasonably well), and that for some GNN architectures a
*perfectly* faithful explanation is provably uninformative (the model is injective, so the "true"
explanation is the whole input).

**Minimum data scale and fit to this project:** all three require a trained GNN to explain in the
first place. With 2,008 edges over 40 disease classes and one relation type, there is not enough
data to train a GNN whose *predictions* are trustworthy, which makes explaining those predictions a
second-order problem on top of a first-order one that isn't solved. These methods also explain
**why a specific prediction was made**, not **why two labeled facts about the same pair disagree**
— they are built for the "missing edge" question the PI has explicitly said is *not* the target
question here. **Recommendation weight: low for this graph, for both scale and mismatch-of-question
reasons.**

---

## 4. Concept-based explanation (TCAV, concept bottleneck models) — and grading our own experiment

**TCAV** — Kim et al., *"Interpretability Beyond Feature Attribution: Quantitative Testing with
Concept Activation Vectors,"* ICML 2018
([PMLR v80](https://proceedings.mlr.press/v80/kim18d/kim18d.pdf)). Method: pick a human-named
concept (e.g. "striped"), gather a set of positive examples and a random negative set, get their
activations at some internal layer of the trained network, and fit a linear classifier between
them — the vector orthogonal to that decision boundary is the Concept Activation Vector (CAV). The
directional derivative of the class-prediction logit along the CAV, evaluated over many class
examples, gives the TCAV score: the fraction of examples for which the concept "pushes toward" the
class.

**What TCAV requires to be trustworthy (the entry bar most crude reimplementations skip):**
1. **A validated, linearly-separable concept direction**, not a single hand-written vector — the
   CAV is *trained*, and its own training accuracy is a diagnostic (a concept that can't be
   linearly separated in that layer's activation space is not a valid concept there).
2. **Enough concept examples to train that classifier reliably.** Public syntheses of the method
   (e.g. the *Interpretable Machine Learning* book's TCAV chapter) recommend on the order of **50+
   diverse examples per concept set**, not one.
3. **Repeated CAV retraining against different random negative sets** — commonly cited as **N ≥ 10
   retrainings** (the original paper's own ablations go up to hundreds) — because a single CAV can
   be an artifact of which random negatives happened to be drawn.
4. **A two-sided statistical test** (t-test or z-test) of the resulting *N* TCAV scores against
   TCAV scores from CAVs built on a random, meaningless "concept," at the conventional α = 0.05,
   to certify the concept's effect is not noise.
5. Sensitivity to **which layer** the CAV is computed at, and TCAV is reported to work poorly on
   shallow networks where concepts aren't yet linearly separable.
   ([christophm.github.io chapter](https://christophm.github.io/interpretable-ml-book/detecting-concepts.html))

**What this project actually did (`contrast_experiment.py`, and its precursor
`probe_embeddings.py`), graded against that bar:**

| TCAV requirement | This project's implementation | Gap |
|---|---|---|
| Concept = trained linear classifier (CAV), validated for separability | Concept = **one hand-written probe sentence's embedding**, or a raw `mean(up) − mean(down)` contrast vector | No classifier, no separability check — the "concept direction" was never certified to exist in the space at all |
| ≥50 diverse examples per concept | **1** exemplar sentence per named concept (`probe_bank.json`, 36 concepts) | Off by ~50× on the concept side |
| ≥10 CAV retrainings against random negatives | **0** — a single fixed probe vector | No estimate of vector instability |
| Per-class (up vs down) examples | **2–7 papers per side, per edge** (documented directly in `contrast_experiment.py`'s own header: a 4-vs-3 edge admits only C(7,3)=35 label splits, so the smallest possible permutation p is 0.029) | Off by ~10× on the *outcome* side too — and this is a property of the phenomenon (contested edges are thinly evidenced), not a fixable implementation choice |
| Two-sided test at α=0.05 | Stratified Mann-Whitney (van Elteren) statistic, 10,000-draw within-edge permutation, **BH-corrected across 36 probes** | This part is *methodologically sound* — arguably more rigorous than vanilla TCAV's t-test, because it's rank-based (appropriate at n=2–7) and multiple-comparison corrected |

**Verdict: this was not a valid TCAV instantiation, so its null cannot be read as "TCAV says no."**
It is best described as **a method failure compounding a power failure**, not a correct negative:
- *Method failure*: representing a concept with one sentence, instead of a validated CAV built
  from tens of examples, means the "concept direction" being tested was never shown to correspond
  to anything the embedding space actually separates. A concept that TCAV would reject outright
  for insufficient exemplars was tested anyway.
- *Power failure, and this part is real and not fixable by better TCAV hygiene*: even a perfectly
  built CAV needs a reasonable number of up-labeled and down-labeled *instances* to correlate
  against, and contested edges cap that at 2–7 per side by construction — this is the same
  constraint `contrast_experiment.py`'s own docstring already derives (only 1 of 226 contested
  edges, the 7-vs-7 Bacteroides/Alzheimer's pair, could ever reach q<0.05 alone). Aggregating
  across 43 edges to 509 pairs does not fully rescue this because the 211 contributing papers
  are shared across edges — not independent replicates.
- The one thing that *was* done well — BH-corrected, edge-stratified, rank-based significance
  testing — means that if a real, large, concept-level effect existed on the *sentence-embedding*
  representation used, this pipeline had a fair chance to find it. So the null is informative about
  "there is no huge signal visible to a single-sentence probe on paper-level MiniLM embeddings,"
  but it is silent on whether a properly-built concept vector (50+ examples, validated separability,
  applied per-sentence rather than per-paper) would find something. **That is the specific,
  cheap follow-up worth doing before concluding the concepts genuinely don't matter.**

**Concept bottleneck models (CBMs)** — Koh et al., *"Concept Bottleneck Models,"* ICML 2020, force
a network to predict human concepts first and the label from those concepts second, which makes the
concept layer inherently readable ("predicted `diet_controlled=yes` → predicted `enriched`").
Known failure mode directly relevant here: **concept leakage** — Mahinpei et al. (2021) and Havasi
et al., *"Addressing Leakage in Concept Bottleneck Models,"* NeurIPS 2022
([paper](https://papers.neurips.cc/paper_files/paper/2022/file/944ecf65a46feb578a43abfd5cddd960-Paper-Conference.pdf)),
show that when concepts and the task are learned jointly, the concept layer's soft outputs smuggle
task-relevant information that has nothing to do with the *named* concept, so the readable concept
labels overstate how much the model is "really" using them. A CBM would require a labeled training
set of concept-per-paper at a scale (hundreds of positive instances per concept) this project's
audited coverage numbers (`HANDOFF_embeddings.md` — e.g. `probiotic use` valued at 3.3%, `bmi` at
7.3% of papers) show the corpus does not have for most candidate concepts. **Recommendation weight:
TCAV done properly is worth one more try (see below); full CBM training is not — too little labeled
concept data.**

---

## 5. Counterfactual explanation on graphs

"This edge would flip if X" is the most literal match to the PI's disagreement question, so this
section gets a longer look.

- **CF-GNNExplainer** — Lucic, Ter Hoeve, Tolomei, de Rijke & Silvestri, AISTATS 2022
  ([arXiv:2102.03322](https://arxiv.org/abs/2102.03322)). First GNN counterfactual method: finds the
  minimal set of **edge deletions** that flips a prediction, via a differentiable relaxation trained
  per-instance. Reports flipping the majority of instances across three benchmark datasets by
  removing **fewer than 3 edges on average**, with ≥94% accuracy that the counterfactual graph does
  flip the real (non-relaxed) model. **Output:** literally "delete these ≤3 edges and the model's
  answer changes" — the single most direct match to "this edge would flip if X" of anything
  surveyed. **Failure mode:** edge-deletion-only counterfactuals can't express "if this paper had
  used shotgun instead of 16S," only "if this connection didn't exist," which is a narrower
  vocabulary than the PI's question needs unless study-design facts are themselves graph nodes.
- **RCExplainer** — Bajaj et al., NeurIPS 2021
  ([proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2021/file/2c8c3a57383c63caef6724343eb62257-Paper.pdf)).
  Addresses a specific weakness of instance-by-instance counterfactual search (including
  CF-GNNExplainer): optimizing per-instance can overfit to noise in that one instance. RCExplainer
  instead learns shared decision-region boundaries across many same-class instances first, then
  reads counterfactuals off those boundaries, trading some per-instance flexibility for robustness
  and speed (faster than CF-GNNExplainer, GNNExplainer, PGM-Explainer and SubgraphX).
- **Kelpie** — Rossi, Barbosa & Firmani, *"Explaining Link Prediction Systems based on Knowledge
  Graph Embeddings,"* SIGMOD 2022 / VLDB
  ([vldb.org](https://www.vldb.org/pvldb/vol15/p3566-rossi.pdf)). Purpose-built for **KG embedding**
  models specifically (TransE/ComplEx/etc. — the right family for a taxon–disease graph, unlike the
  GNN-focused methods above). Finds the smallest set of *training facts* that is **necessary**
  (remove them, retrain, prediction flips to false) or **sufficient** (add them to a fresh entity,
  prediction becomes true), using a "post-training" approximation so it doesn't literally retrain
  the whole embedding model per candidate explanation. **Output:** a small set of existing KG facts,
  labeled necessary/sufficient — again directly readable, and it is explicitly framed as "which
  facts is this prediction counterfactually dependent on," the closest published framing to "this
  edge would flip if paper X's evidence were removed."
- **"Imagine" / additive counterfactuals** — *"Additive Counterfactuals for Explaining Link
  Predictions on Knowledge Graphs,"* EKAW 2024
  ([Springer](https://link.springer.com/chapter/10.1007/978-3-031-77792-9_21)). Complements Kelpie's
  subtractive framing by finding facts whose *addition* would most change a prediction — useful
  framing for "what evidence, if it existed, would resolve this contested edge," i.e. a literal
  prospective research suggestion rather than a retrospective attribution.

**Minimum data scale:** all of these need a *trained* link-prediction model (KGE or GNN) to probe.
Kelpie in particular is designed for exactly the graph type here (a small-to-medium relational KG
with embedding-based scoring, not a giant heterophilic GNN benchmark), and post-training
approximation is specifically meant to make the "retrain and check" step cheap enough to try many
candidate facts — this is more compatible with a 2,008-edge graph than GNN-based counterfactual
methods that assume enough data to train a GNN with generalizable structure in the first place.

**Fit to the disagreement question:** counterfactual framing is the most natural fit of anything
surveyed, *if* the target model is reframed correctly. Applied to the current edge-existence model
("does taxon T associate with disease D"), a counterfactual answers "which papers make this edge
exist at all" — useful but not new. Applied to a **direction-prediction model trained per contested
edge** (predict up vs. down from paper-level features), a Kelpie-style necessary/sufficient
explanation would read as "paper P's reported `recruitment=hospital` is necessary for this edge's
majority direction" — which is exactly the sentence the PI wants, *if* such a direction-prediction
model can be trained at all. Given `ceiling_direction_probe.py`'s finding that paper-level features
cap direction accuracy at 0.706 (arithmetic ceiling, not a modeling failure) because 65% of papers
report both directions, this reframing needs the unit of prediction to be the
(paper, taxon, disease) triple, not the paper — matching the fix already identified in
`HANDOFF_embeddings.md` §2.

---

## Ranked recommendation for THIS graph (2,008 edges, question = disagreement, not missing links)

1. **Rule mining (AnyBURL/AMIE) over an augmented graph, not the taxon–disease graph alone —
   highest priority.** Add paper nodes and study-design-attribute nodes/edges (the ones already
   audited in `HANDOFF_embeddings.md`: differential-abundance method, antibiotic use, BMI, etc.) so
   that "up-paper" and "down-paper" become graph facts a rule can quantify over. AnyBURL is
   sampling-based, has no meaningful minimum-scale floor, requires no GPU, and its output — a Horn
   rule with support/confidence — is the literal form of "A relates to B because of P" the PI asked
   for. This directly reuses the corpus's existing bottleneck (thin per-edge evidence) as
   *aggregate* rule support across all contested edges at once, rather than testing one probe on
   one edge at a time, which is the structural weakness that capped both prior attempts. This is
   the same recipe already published for exactly this kind of biomedical KG (Hetionet/Rephetio drug
   repurposing, Sci Reports 2024).
2. **Redo the TCAV-style concept probe properly, once, before abandoning the concept idea.** The
   two priors here were not valid TCAV instantiations (single-sentence "concepts," no CAV
   validation, no retraining against random negatives) — build actual CAVs from the ≥50-example
   sentence sets `variable_sweep.py` already retrieves per concept (it exists and reports AUC/lift
   already), retrain ≥10 CAVs against random negative chunk sets, and re-run the existing
   edge-stratified permutation test on the resulting scores instead of on raw probe-sentence
   cosine. This is a low-cost fix to a real methodological gap, not a new research direction, and it
   will settle whether the prior null was the corpus (real power ceiling: 2–7 papers/side) or the
   method (invalid concept vectors) — which the write-up above could not fully separate.
3. **Kelpie-style necessary/sufficient counterfactuals, but only after re-scoping the target model
   to (paper, taxon, disease) triples**, per `ceiling_direction_probe.py`'s already-diagnosed 0.706
   ceiling on paper-level features. This is the best conceptual match to "this edge would flip if
   X," and Kelpie's post-training trick keeps it cheap at this graph's size, but it is gated on
   building a working direction-prediction model first — currently blocked, not currently null.
4. **Post-hoc GNN explainers (GNNExplainer/PGExplainer/SubgraphX) — low priority.** They require a
   trained GNN this graph is too small to trust, they answer "why does this edge exist" rather than
   "why do papers disagree," and the faithfulness-metric literature (GraphFramEx et al.) says even
   their answers to the question they *do* answer are metric-dependent and unreliable on
   heterophilic structure.
5. **Attention-based explanation — lowest priority.** No attention model exists here yet, building
   one needs more labeled/trained structure than the corpus offers, and even a well-trained one
   would inherit the unresolved Jain-and-Wallace/Wiegreffe-and-Pinter dispute about whether its
   weights mean anything without per-model diagnostics this project has no spare data budget to run.

**One line for the write-up:** the study-design-variable and concept-embedding attempts failed
because they tested one weak signal against one edge at a time (or a single-sentence proxy for a
concept); rule mining and Kelpie-style counterfactuals both work by aggregating evidence *across*
all 220 contested edges simultaneously and outputting a sentence, which is the structural fix both
prior negative results were pointing at without saying so directly.

---

## Sources

- Lao & Cohen, *Relational retrieval using a combination of path-constrained random walks*, 2010;
  Lao, Mitchell & Cohen, EMNLP 2011 — [Semantic Scholar](https://www.semanticscholar.org/paper/19acbce6e3c26e600368d84b38ae770eb0380c10), [arXiv:1404.3301](https://arxiv.org/pdf/1404.3301)
- Xiong, Hoang & Wang, *DeepPath*, EMNLP 2017 — [arXiv:1707.06690](https://arxiv.org/pdf/1707.06690), [ACL D17-1060](https://aclanthology.org/D17-1060/)
- Das et al., *MINERVA*, ICLR 2018 — [ACL D18-1362](https://aclanthology.org/D18-1362.pdf)
- Galárraga, Teflioudi, Hose & Suchanek, *AMIE*, WWW 2013 / *AMIE+*, VLDB J. 2015 — [GitHub](https://github.com/dig-team/amie), [paper PDF](https://resources.mpi-inf.mpg.de/yago-naga/amie/amie.pdf)
- Meilicke, Chekol, Ruffinelli & Stuckenschmidt, *AnyBURL*, IJCAI 2019; *Reinforced AnyBURL*, arXiv:2004.04412 — [site](https://web.informatik.uni-mannheim.de/AnyBURL/), [IJCAI PDF](https://web.informatik.uni-mannheim.de/AnyBURL/meilicke19anyburl.pdf)
- *Explainable drug repurposing via path-based knowledge graph completion*, Sci. Reports 2024 — [nature.com](https://www.nature.com/articles/s41598-024-67163-x), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258358/)
- Jain & Wallace, *Attention is not Explanation*, NAACL 2019 — [arXiv:1902.10186](https://arxiv.org/abs/1902.10186)
- Wiegreffe & Pinter, *Attention is not not Explanation*, EMNLP 2019 — [arXiv:1908.04626](https://arxiv.org/pdf/1908.04626)
- Ying, Bourgeois, You, Zitnik & Leskovec, *GNNExplainer*, NeurIPS 2019 — [arXiv:1903.03894](https://arxiv.org/abs/1903.03894)
- Luo et al., *PGExplainer*, NeurIPS 2020 — [arXiv:2011.04573](https://arxiv.org/pdf/2011.04573)
- Yuan et al., *SubgraphX*, ICML 2021 — [arXiv:2102.05152](https://arxiv.org/pdf/2102.05152)
- Amara et al., *GraphFramEx*, LoG 2022; Faber et al., *Evaluating Explainability for GNNs*, Scientific Data 2023 — [nature.com](https://www.nature.com/articles/s41597-023-01974-x); *Probing GNN Explainers*, 2021 — [arXiv:2106.09078](https://arxiv.org/pdf/2106.09078)
- Kim et al., *TCAV*, ICML 2018 — [PMLR v80](https://proceedings.mlr.press/v80/kim18d/kim18d.pdf); synthesis: [christophm.github.io](https://christophm.github.io/interpretable-ml-book/detecting-concepts.html)
- Mahinpei et al. 2021; Havasi et al., *Addressing Leakage in Concept Bottleneck Models*, NeurIPS 2022 — [paper](https://papers.neurips.cc/paper_files/paper/2022/file/944ecf65a46feb578a43abfd5cddd960-Paper-Conference.pdf)
- Lucic et al., *CF-GNNExplainer*, AISTATS 2022 — [arXiv:2102.03322](https://arxiv.org/abs/2102.03322)
- Bajaj et al., *RCExplainer*, NeurIPS 2021 — [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2021/file/2c8c3a57383c63caef6724343eb62257-Paper.pdf)
- Rossi, Barbosa & Firmani, *Kelpie*, SIGMOD/VLDB 2022 — [vldb.org](https://www.vldb.org/pvldb/vol15/p3566-rossi.pdf)
- *Additive Counterfactuals for Explaining Link Predictions on Knowledge Graphs*, EKAW 2024 — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-77792-9_21)

## Internal artifacts referenced

- `proj_2_attempt3/kg/FINDINGS_paper_discordance.md` — 26-variable study-design test, best FDR 0.234
- `proj_2_attempt3/kg/contrast_experiment.py`, `contrast_experiment.json` — 36-probe concept-embedding contrast, best q ≈ 0.405, n_edges=43, n_pairs=509
- `proj_2_attempt3/kg/probe_embeddings.py`, `HANDOFF_embeddings.md` — earlier single-vector probes, and `ceiling_direction_probe.py`'s 0.706 arithmetic ceiling on paper-level direction prediction
- `proj_2_attempt3/kg/variable_sweep.py` — existing retrieval benchmark (AUC/lift per concept) that could supply the ≥50-example concept sets a proper TCAV redo needs
