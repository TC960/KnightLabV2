# Classical knowledge-graph embedding (KGE) methods on a small, contested, hierarchical biomedical KG

Status: DRAFT — first full pass, all five sections researched and written.

Motivating constraint (this project): 883 taxa + 40 diseases = 923 entities, 2,008
association edges (`ENRICHED_IN` / `DEPLETED_IN`) + 727 containment edges (`CONTAINS`),
i.e. **3 relation types total**. 220 taxon-disease pairs are CONTESTED — both directions
asserted, each backed by a paper count, kept deliberately unaveraged (see root
`CLAUDE.md`: "contested edges are kept, never averaged... disagreement is a finding, not
noise"). Any embedding approach has to survive contact with both of those facts: almost
no relation diversity, and a chunk of the edge set that is *supposed* to look
self-contradictory.

---

## 1. How TransE / DistMult / ComplEx / RotatE / ConvE handle 2-3 relation types + a strong hierarchy

All five were designed and benchmarked on graphs with **hundreds** of relation types
(FB15k-237: 237 relations; WN18RR: 11) where most of the modeling burden is
*disentangling relation semantics*. With 3 relation types, that burden almost
disappears — you're closer to plain entity embedding with a 3-way linear readout than to
"knowledge graph completion" in the sense these papers were written for. That reframes
the comparison: the interesting differences between the five models on this graph are
not about relation capacity, but about which ones can represent an **asymmetric**
`CONTAINS` edge and **compose** it along a taxonomic chain (phylum→class→order→family→
genus→species).

- **TransE** (Bordes et al., NeurIPS 2013) — scores `h + r ≈ t`. Cannot represent
  1-to-many or symmetric relations well (a classic weakness: `CONTAINS` is exactly
  1-to-many, one family containing several genera). Composition is additive, so
  multi-hop containment chains are in principle representable (`r_family→genus +
  r_genus→species ≈ r_family→species`), but TransE is also known to collapse entities
  that share many relations into near-identical points, which is a real risk for
  monotypic genera that contain exactly one child.
- **DistMult** (Yang et al., ICLR 2015) — bilinear-diagonal, `score(h,r,t) =
  score(t,r,h)` by construction, i.e. **structurally symmetric**. `CONTAINS` is
  antisymmetric (family contains genus, never the reverse) — DistMult cannot represent
  that at all without a workaround (e.g., adding an explicit inverse `CONTAINED_BY`
  relation, which papers using DistMult on hierarchies routinely do). Weakest fit here
  on paper.
- **ComplEx** (Trouillon et al., ICML 2016) — extends DistMult to complex-valued
  embeddings with a Hermitian dot product, which restores the ability to model
  antisymmetric relations. Good fit for `CONTAINS`; a standard strong baseline in every
  benchmark survey (see §3).
- **RotatE** (Sun et al., ICLR 2019, arXiv:1902.10197) — relations are rotations in
  complex space. The paper's own selling point is explicit support for symmetry,
  antisymmetry, inversion, **and composition** — and its worked examples are
  hierarchy-shaped relations on WN18RR (hypernym/hyponym chains). This is the one model
  of the five whose design argument is *literally* "handles the pattern your
  containment edges need."
- **ConvE** (Dettmers et al., AAAI 2018) — 2D convolution over reshaped
  entity/relation embeddings, more parameters per relation than the others. It's
  general-purpose rather than hierarchy-specific, and its extra capacity is the kind
  that needs more data to not overfit (see §3) — a bad match for 2,008 edges regardless
  of relation semantics.

Net: **RotatE and ComplEx are the two with an actual architectural argument for this
graph's shape** (antisymmetric containment + short compositional chains); DistMult is
structurally wrong for `CONTAINS`; TransE is a reasonable cheap baseline; ConvE is
probably over-parameterized for the data volume here.

(Separately, hyperbolic/Poincaré-style embeddings — Nickel & Kiela, NeurIPS 2017 — are
purpose-built for tree-like hierarchies rather than adapted to them; that's a
substantially different modeling bet and is being tracked in parallel by another agent
on this team (`r1-hyperbolic`), not duplicated here.)

## 2. Contradictory / weighted edges — the harder problem

This is the part where the standard toolkit doesn't fit cleanly, and it's worth being
precise about *why*, because there are two different problems that look similar:

**Problem A — "this fact is uncertain."** A single relation fact `(h, r, t)` that may or
may not be true, with a confidence in `[0,1]`. This is what the uncertain-KG-embedding
literature solves.

**Problem B — "two mutually exclusive facts are both asserted, each with its own
evidence count."** `(Bacteroides, ENRICHED_IN, Alzheimers)` with 11 papers and
`(Bacteroides, DEPLETED_IN, Alzheimers)` with 9 papers. This is **not** the same
problem, and forcing it into Problem A's framing is actively harmful to what this
project has decided matters.

### What Problem A's literature offers

- **UKGE** (Chen, Chen, Shi, Sun, Zaniolo, *Embedding Uncertain Knowledge Graphs*, AAAI
  2019; arXiv:1811.10667; code: github.com/stasl0217/UKGE). Learns embeddings whose dot
  product regresses onto a confidence score rather than a binary label, and uses
  **probabilistic soft logic (PSL)** to propagate confidence to triples not seen during
  training (e.g., via transitivity rules). Evaluated on three uncertain KGs, one of
  which — **PPI5k**, built from STRING protein-protein interaction confidence scores
  (Szklarczyk et al. 2017) — is a genuine biological analog: 4,999 proteins, 7 relation
  types, **271,666** confidence-scored triples. That's ~135x our edge count for a
  similar entity count, which is itself informative (see §3).
- **PSL itself** (Bach, Broecheler, Huang, Getoor, *Hinge-Loss Markov Random Fields and
  Probabilistic Soft Logic*, JMLR 2017; arXiv:1505.04406; code:
  github.com/stephenbach/bach-jmlr17-code) is the general soft-logic inference engine
  UKGE borrows. It represents truth values continuously in `[0,1]` and lets you write
  logical rules (e.g., "if X contains Y and X is enriched, Y is more likely enriched")
  as soft constraints solved via convex optimization. Usable standalone as a
  rule-based smoothing layer over a graph, independent of embeddings.
- **Probabilistic box embeddings** (BEUrRE — *Probabilistic Box Embeddings for
  Uncertain Knowledge Graph Reasoning*, arXiv:2104.04597) — represents entities as
  regions/boxes rather than points, so containment and overlap of boxes encode
  confidence and calibrated uncertainty directly. A newer alternative to UKGE's
  regression-based approach.
- Also newer (2025): *Certainty in Uncertainty: Reasoning over Uncertain Knowledge
  Graphs with Statistical Guarantees* (arXiv:2510.24754) — adds formal statistical
  guarantees on top of the UKGE-style setup. Signals this is still an active area, not
  a solved-and-abandoned one.

### Why none of this should be imported wholesale

Every method above assumes there is **one** relation per `(h,t)` pair whose truth is a
matter of degree. Our contested pairs have **two** relation instances — `ENRICHED_IN`
and `DEPLETED_IN` — that are separately, discretely true (each is a real reported
finding in some papers), and the project's own explicit design decision is that a single
"confidence" number would erase exactly the signal being preserved: "~1 taxon in 3 flips
sign between cohorts in this literature, so disagreement is a finding, not noise" (root
`CLAUDE.md`). Collapsing `(11 papers enriched, 9 depleted)` into a single soft-truth
value of ~0.55 for one merged "is-enriched" relation is precisely the averaging this
project has already rejected — twice, per the git history (`09b341a kg: drop the union
— the gold is a test set, not graph content`, `30120ea kg: union graph with per-edge
provenance`).

**The better fit is much simpler and doesn't require adopting an uncertain-KG
architecture at all**: keep `ENRICHED_IN` and `DEPLETED_IN` as two separate hard
relation types (as the graph already does), which means a standard KGE model already
represents both facts as independently true — there is no "contradiction" at the level
the model sees, only two triples with different relations. The one piece of information
that plain unweighted training would throw away is the *evidence count* behind each
triple (11 vs. 9 is a near-tie; 11 vs. 1 is not). That is a **loss-weighting** problem,
not an uncertain-fact-embedding problem, and it has a direct, off-the-shelf answer in
PyKEEN (§4): `pykeen.triples.weights` supports arbitrary per-triple weights, and v1.11
added a general `LossWeighter` interface plus a built-in `RelationLossWeighter`. Set
each triple's weight to (a monotonic function of, e.g. `log1p`) its paper count and
train normally — contested pairs stay exactly as contested as the data says, nothing is
merged, and the model is told which triples have more evidence behind them without
being told to resolve the disagreement.

If graph-consistency propagation is wanted later (e.g., "genus X is probably enriched
because its whole family is, in the 9 papers that don't mention X directly"), PSL is the
right tool for that *specific* sub-problem, applied as a rule-based post-processing/
smoothing step — not as the embedding objective itself.

## 3. Minimum graph size — where we actually fall

| Graph | Entities | Relations | Triples/edges | Notes |
|---|---:|---:|---:|---|
| FB15k | 14,951 | 1,345 | 592,213 | classic KGE benchmark |
| FB15k-237 | 14,541 | 237 | 310,116 | leakage-filtered FB15k |
| WN18RR | 40,943 | 11 | 93,003 | leakage-filtered WN18, hierarchy-heavy |
| PPI5k (UKGE) | 4,999 | 7 | 271,666 | closest *uncertain* biological analog |
| HMDAD (microbe-disease) | 292 microbes + 39 diseases | 1 | 450 (after dedup) | closest *domain* analog |
| **This project** | **883 taxa + 40 diseases = 923** | **3** | **2,735** (2,008 assoc + 727 containment) | |

We sit at roughly **1/34 of WN18RR's triple count and 1/113 of FB15k-237's**, with an
entity count in the same range as WN18RR but two orders of magnitude fewer edges.
Average node degree here is ~5.9 (`2 × 2,735 / 923`) — low enough that most entities
have only a handful of neighbors to disambiguate their embedding from every other
entity's. This matters concretely: Ali et al.'s large benchmark study (*Bringing Light
into the Dark: A Large-scale Evaluation of Knowledge Graph Embedding Models Under a
Unified Framework*, arXiv:2006.13365 — the same group that built PyKEEN) ran ~35,000
KGE experiments and found that reported model-vs-model differences on standard
benchmarks are frequently **dominated by hyperparameter search budget**, not
architecture. At our scale, that effect gets worse, not better: fewer triples means
noisier gradient estimates and a wider gap between "this model is better" and "this
random seed got lucky." **Any result from this graph needs a k-fold or repeated-split
estimate with a variance/CI, not a single train/test number** — consistent with how
this project already reports everything else (ranges, not point estimates).

The domain-specific comparison is more useful than the generic-benchmark one. **HMDAD**
(292 microbes, 39 diseases, 450 associations after dedup — Ma et al./multiple
downstream papers) is almost exactly our shape, just ~4-4.5x smaller in edges, and
1 relation type. Every successful published method on HMDAD-scale graphs (§5) avoids
plain translational/bilinear KGE and instead **injects external similarity structure**
— microbe functional similarity (Gaussian-kernel or taxonomy-derived), disease semantic
similarity (from disease ontology) — fused via GCN or matrix factorization, precisely
because a bare bipartite/near-bipartite graph this small doesn't give a multi-relational
KGE model enough structure to learn from on its own. That's the strongest
size-calibrated evidence available: **at HMDAD/our scale, the field's answer has not
been "run TransE," it's been "add side information."** We already have an obvious,
free source of side information a classical KGE model can't otherwise see: the NCBI
taxonomy tree and (per the disease-ontology work in root `CLAUDE.md`) partial MONDO
structure. A plain KGE run without that is a reasonable first cut, but expectations
should be calibrated to it needing an assist, not to it matching FB15k-237-style
numbers.

## 4. Library status (checked directly against GitHub, 2026-09-18)

| Library | Maintained? | Latest release | Python | Backend | Notes |
|---|---|---|---|---|---|
| **PyKEEN** (`pykeen/pykeen`) | Yes — active | v1.11.1 (Apr 2025); repo pushed to Sep 6 2026 | `>=3.11` | PyTorch `>=2.0` | 2,036★, 129 open issues; most complete model zoo (TransE/DistMult/ComplEx/RotatE/ConvE all built in); has per-triple loss weighting (`pykeen.triples.weights`, `LossWeighter`/`RelationLossWeighter`, added v1.11) — directly usable for the evidence-count weighting in §2; same group published the HPO-variance study in §3 |
| **AmpliGraph** (`Accenture/AmpliGraph`) | Yes — active | 2.2.0 (Jul 2026) | `>=3.9,<3.12` | TensorFlow `>=2.15,<2.16` (pinned — last Keras-2-era TF release) | 2,238★, 30 open issues; healthy but TF-locked, and the TF pin caps how long it stays easy to install alongside current TF |
| **DGL-KE** (`awslabs/dgl-ke`) | Effectively stalled | 0.1.1 (Aug 2020) | — | DGL | Repo not archived, but real commits stopped ~2023; the only 2025/2026 activity is a README edit and one PR merge to a training script. Would not start new work on this. |
| **LibKGE** (`uma-pi1/kge`) | Alive, low-traffic | rolling (last commit Apr 2026) | — | PyTorch | 835★, 31 open issues; University of Mannheim reproducible-research focus, narrower model zoo than PyKEEN, mostly used for HPO-methodology papers rather than as a general library |

**Recommendation: PyKEEN.** It's the only one of the four combining (a) active
maintenance into the present, (b) all five target models pre-implemented, and (c) a
native mechanism for per-triple evidence weighting — which §2 established is the actual
lever this graph needs, not an uncertain-KG architecture swap.

## 5. Published work on microbe-disease KG embedding specifically

None of the published microbe-disease work uses plain TransE/DistMult/ComplEx/RotatE/
ConvE off the shelf — they all build custom GCN/attention/matrix-factorization
architectures that fuse the association graph with computed similarity networks. That
itself is evidence for the §3 conclusion (bare small graphs need an assist):

- **LGRSH** — *Predicting Microbe-Disease Association by Learning Graph
  Representations and Rule-Based Inference on the Heterogeneous Network* (PMC7174569).
  Node2vec over a heterogeneous network combining microbe-similarity, disease-similarity,
  and known-association subgraphs, plus rule-based inference.
- **KGNMDA** — Ma et al., *KGNMDA: A Knowledge Graph Neural Network Method for
  Predicting Microbe-Disease Associations* (IEEE/ACM TCBB, 2022; PMID 35724280).
  Builds a KG from several source databases (not HMDAD alone), learns representations
  via a GNN, and fuses in Gaussian-kernel similarity features before scoring.
- **FGCNMF** — *Fast graph convolutional models incorporating matrix factorization for
  predicting microbe-disease associations* (PMC12780053). Graph convolution + matrix
  factorization on the bipartite association graph.
- **GCATCMDA** — graph convolution + attention with a dual-fusion feature module.
- **GNN + contrastive learning** — *Predicting microbe-disease associations via graph
  neural network and contrastive learning* (Frontiers in Microbiology, 2024;
  PMC11671253).
- **KGCLMDA** — *a computational model for predicting latent associations of microbial
  drugs using knowledge graphs and contrastive learning* (Bioinformatics, 2025) — same
  family of technique, applied to microbe-drug rather than microbe-disease, included
  because it's the same design pattern (KG + contrastive learning) one rung over.
- **MNNMDA** — matrix-nuclear-norm minimization; not an embedding method per se, closer
  to matrix completion, listed for completeness since it targets the identical HMDAD
  task.

Worth flagging as related-project context rather than a method to copy: **MINERVA**
(*MINERVA — microbiome network research and visualization atlas: a scalable knowledge
graph for mapping microbiome-disease associations*, Briefings in Bioinformatics,
2025/2026) is the closest sibling project to ours in spirit (LLM-extracted
microbiome-disease KG at larger scale). It has since drawn a published methodological
critique (Chirumbolo, *Methodological and statistical concerns in MINERVA
microbiome-disease knowledge graph*, Briefings in Bioinformatics, 2026) — selection bias
from open-access-only sourcing, sentence-level extraction ignoring hedging/negation, and
**no FDR correction across the many implicit hypothesis tests a large edge set
represents** — with an author response (same issue). None of this is about embeddings,
but it's a direct precedent for the multiple-comparisons discipline this project already
applies elsewhere (root `CLAUDE.md`'s BH-correction habit on the paper-discordance and
disease-hierarchy findings), and worth citing if this project's KG is ever compared
publicly against MINERVA's.

---

## Ranked shortlist + the minimal experiment

**Ranked shortlist for a first pass on this graph:**

1. **RotatE (via PyKEEN)** — top pick. Only model of the five with an explicit
   architectural argument for asymmetric + compositional relations, which is exactly
   what `CONTAINS` needs, at a moderate parameter count.
2. **ComplEx (via PyKEEN)** — cheap, robust baseline; also asymmetric-capable; useful as
   a second opinion against RotatE and as the "boring but hard to beat" reference point
   the HPO-variance literature (§3) says you need.
3. **TransE (via PyKEEN)** — simplest possible floor; expect it to underperform on
   `CONTAINS` composition but it's nearly free to run alongside the above two.
4. **ConvE** — deprioritize; its extra capacity is the wrong trade at 2,735 edges.
5. **Do not adopt UKGE/PSL/box-embedding architectures wholesale** — the contradiction
   they're built to solve isn't our contradiction (§2). Reuse only the *mechanism*
   (per-triple confidence-as-weight) via PyKEEN's native loss weighting.

**Exact minimal experiment (RotatE, PyKEEN):**

1. Load the 2,735 triples (2,008 `ENRICHED_IN`/`DEPLETED_IN` + 727 `CONTAINS`) into a
   `TriplesFactory`, attaching a per-triple weight column = `log1p(paper_count)`
   (`pykeen.triples.weights`).
2. Split with `TriplesFactory.split(ratios=[.8, .1, .1], random_state=0,
   method="cleanup")` to guarantee no orphan entities in val/test — non-optional at
   this entity count.
3. Because 923 entities / 2,735 edges is small enough for the split itself to be a
   meaningful source of variance, **repeat the split 5x with different seeds** (or use
   PyKEEN's k-fold support) and report mean ± spread, not a single number — per §3.
4. Model: `RotatE`, embedding dim small — start at 32, sweep {16, 32, 64} only (not
   FB15k's typical 200-1000; we have ~100x fewer triples to constrain that many
   parameters).
5. Negative sampling: **type-constrained** — corrupt the taxon slot only with other
   taxa, the disease slot only with other diseases. Uncontrolled corruption on a
   near-bipartite graph this small makes negatives trivially easy to reject and
   inflates every metric.
6. Loss: self-adversarial negative sampling loss (as in the original RotatE paper),
   using the per-triple weights from step 1.
7. Evaluate filtered Hits@1/3/10 and MRR, **restricted to the correct entity type per
   slot** (rank taxa against taxa, diseases against diseases) — unrestricted ranking
   against all 923 entities overstates performance for the same reason as step 5.
8. **Mandatory sanity baseline before trusting any of the above**: compare against a
   trivial frequency/degree heuristic (e.g., predict a taxon's most-evidenced known
   direction for any disease it's already linked to; predict association existence
   from shared-paper co-occurrence). At this edge count there is a real risk a full KGE
   model does no better than this baseline — check it explicitly, first, before
   investing further.

This whole experiment runs on CPU in minutes (2,735 triples is trivial for any of these
models) — no GPU allocation needed, unlike the project's other current experiment track.

---

**Sources**

- [Bordes et al., Translating Embeddings for Modeling Multi-relational Data (TransE), NeurIPS 2013](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)
- [Yang et al., Embedding Entities and Relations for Learning and Inference in Knowledge Bases (DistMult), ICLR 2015](https://arxiv.org/abs/1412.6575)
- [Trouillon et al., Complex Embeddings for Simple Link Prediction (ComplEx), ICML 2016](https://arxiv.org/abs/1606.06357)
- [Sun et al., RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space, ICLR 2019](https://arxiv.org/pdf/1902.10197)
- [Dettmers et al., Convolutional 2D Knowledge Graph Embeddings (ConvE), AAAI 2018](https://arxiv.org/abs/1707.01476)
- [Chen, Chen, Shi, Sun, Zaniolo, Embedding Uncertain Knowledge Graphs (UKGE), AAAI 2019 / arXiv:1811.10667](https://arxiv.org/abs/1811.10667) — [code](https://github.com/stasl0217/UKGE)
- [Bach, Broecheler, Huang, Getoor, Hinge-Loss Markov Random Fields and Probabilistic Soft Logic, JMLR 2017 / arXiv:1505.04406](https://arxiv.org/abs/1505.04406) — [code](https://github.com/stephenbach/bach-jmlr17-code)
- [Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning, arXiv:2104.04597](https://arxiv.org/pdf/2104.04597)
- [Certainty in Uncertainty: Reasoning over Uncertain Knowledge Graphs with Statistical Guarantees, arXiv:2510.24754](https://arxiv.org/html/2510.24754)
- [Ali et al., Bringing Light Into the Dark: A Large-scale Evaluation of Knowledge Graph Embedding Models Under a Unified Framework, arXiv:2006.13365](https://arxiv.org/pdf/2006.13365)
- [Chang et al., Benchmark and Best Practices for Biomedical Knowledge Graph Embeddings, BioNLP 2020 / arXiv:2006.13774](https://arxiv.org/abs/2006.13774)
- [PyKEEN GitHub](https://github.com/pykeen/pykeen) / [PyKEEN docs — Loss Weighting](https://pykeen.readthedocs.io/en/stable/reference/loss_weighting.html)
- [AmpliGraph GitHub](https://github.com/Accenture/AmpliGraph)
- [DGL-KE GitHub](https://github.com/awslabs/dgl-ke)
- [LibKGE GitHub](https://github.com/uma-pi1/kge)
- [Human Microbe-Disease Association Database (HMDAD)](https://bio.tools/hmdad)
- [KGNMDA: A Knowledge Graph Neural Network Method for Predicting Microbe-Disease Associations, IEEE/ACM TCBB 2022](https://pubmed.ncbi.nlm.nih.gov/35724280/)
- [Predicting Microbe-Disease Association by Learning Graph Representations and Rule-Based Inference on the Heterogeneous Network (LGRSH)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7174569/)
- [Predicting microbe-disease associations via graph neural network and contrastive learning, Frontiers in Microbiology 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11671253/)
- [KGCLMDA, Bioinformatics 2025](https://academic.oup.com/bioinformatics/article/41/9/btaf457/8237362)
- [MINERVA — microbiome network research and visualization atlas, Briefings in Bioinformatics 2025/2026](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12454267/)
- [Methodological and statistical concerns in MINERVA microbiome-disease knowledge graph, Briefings in Bioinformatics 2026](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12780759/)
