# Hyperbolic / non-Euclidean embeddings for taxonomic hierarchies

Scoped to Sam's question: can we embed the ~900 microbe/disease nodes such that (a) rank
(species/genus/family/...) is queryable in the embedding space and (b) the geometry itself is
the reason to use it, not just a nicer visualization. Written against the Knight Lab graph:
883 taxa, 40 diseases, 2,008 direction-only edges, 727 containment links.

---

## 1. Why hyperbolic space fits trees — the actual geometric reason

The argument is a volume-growth mismatch, not an aesthetic one.

A tree with branching factor `b` has `(b+1)b^(l-1)` nodes at depth exactly `l`, and
`((b+1)b^l - 2)/(b-1)` nodes within depth `l` — **exponential in `l`**. Euclidean space of any
fixed dimension `d` has volume growing only **polynomially** in radius (`r^d`). So a ball of
radius `r` in `R^d` can only ever contain polynomially many things, but a tree needs
exponentially many slots at distance `r` from the root. The only way to fit an exponentially
branching tree into Euclidean space without crushing far-apart subtrees together is to keep
adding dimensions — which is exactly what plain node2vec/GloVe-style Euclidean embeddings do:
they need large `d` to avoid distortion, and even then get it approximately.

Hyperbolic space (constant negative curvature `K = -1`) has the opposite property: a disc of
radius `r` has circumference `2π·sinh(r)` and area `2π(cosh(r) - 1)` — **both exponential in
`r`**. This is the literal geometric analogue of tree growth, so a tree embeds into 2-dimensional
hyperbolic space with almost no distortion, independent of branching factor. This is stated
explicitly in Nickel & Kiela's original paper (Section 2, "Embeddings and Hyperbolic Geometry")
and is the reason the Poincaré ball needs far fewer dimensions than Euclidean space for the same
fidelity — their Table 1 shows Poincaré at `d=10` (WordNet reconstruction MAP 0.851) already
beating Euclidean at `d=200` (MAP 0.168).

This is not a hand-wavy analogy — it's backed by a hard result: **any finite tree can be embedded
into hyperbolic space (even 2-D) such that path distances are preserved up to a multiplicative
`(1+ε)` distortion**, for any `ε`, via Sarkar's construction (Sarkar, *"Low Distortion Delaunay
Embedding of Trees in Hyperbolic Plane,"* 2011 — [PDF](https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf)).
The construction places the root at the origin and recursively places each node's children evenly
spaced on a hyperbolic sphere around the parent, with the sphere's hyperbolic radius scaled so
that a child, its parent, and its own children remain metrically consistent — you're
literally exploiting the exponential circumference to give every node "room" for its subtree
without moving distant subtrees closer together. No finite-dimensional Euclidean embedding has
an analogous exact-embedding theorem for trees with distortion independent of size; the best
Euclidean bounds (Bourgain-type) scale with `log(n)`.

Practical dimension-vs-fidelity tradeoffs (precision needed, not just accuracy) are analyzed in
De Sa et al., *"Representation Tradeoffs for Hyperbolic Embeddings"* (ICML 2018) — [arXiv:1804.03329](https://arxiv.org/pdf/1804.03329)
— which formalizes the scaling factor / bits-of-precision tradeoff Sarkar's construction leaves
open (precision scales with max path length, log-scales with branching factor).

## 2. Encoding rank/depth as a queryable quantity

This is the part directly load-bearing for Sam's "ask at species vs genus vs family level."

**Yes — Euclidean norm (distance from the origin) is the depth signal, by construction of the
loss, not by post-hoc correlation.** Nickel & Kiela place the root near the origin (Poincaré
distance to the origin is small when `||θ|| ≈ 0`) and let leaves migrate toward the boundary,
because the metric itself blows up near `||x|| → 1` (Eq. 1: `d(u,v) = arcosh(1 + 2‖u−v‖² /
((1−‖u‖²)(1−‖v‖²)))`). Fitting this metric to observed is-a/parent-child pairs pulls
general/shallow concepts toward the center and specific/deep ones outward, because that's the
only way to simultaneously satisfy "children close to their one parent" and "unrelated deep
nodes far apart" — the tree literally self-organizes by depth without depth ever being given
as a label. This is empirically confirmed, not just theoretical: the paper's own worked example
(cited in a follow-up analysis of their WordNet embedding) shows norm ("h-norm") tracking depth
directly — `sport` (depth 9) → h-norm 0.55, `skateboarding` (depth 12) → h-norm 2.25.

**You can threshold on it.** Nickel & Kiela use exactly this for their lexical-entailment
experiment (HyperLex): to score whether `is-a(u, v)` holds (i.e., which of u/v is the more
general concept), they use

```
score(is-a(u, v)) = −(1 + α(‖v‖ − ‖u‖)) · d(u, v)
```

with `α = 10³` — i.e., a term that explicitly penalizes the case where the (candidate) parent
`v` has a *larger* norm than the child `u`, on top of raw distance. This is precisely
"threshold/compare on radius to answer a rank question," already validated in the source paper,
and it beat every WordNet-based baseline on HyperLex (Spearman's ρ = 0.512 vs 0.253 for the next
best WN-based method, Table 3).

For a rank-name query specifically (species vs. genus vs. family, not just "which is more
general"), the more relevant successor is **hyperbolic entailment cones** (Ganea, Bécigneul,
Hofmann, *"Hyperbolic Entailment Cones for Learning Hierarchical Embeddings,"* ICML 2018 —
[arXiv:1804.01882](https://arxiv.org/abs/1804.01882), [PMLR](https://proceedings.mlr.press/v80/ganea18a.html),
code: [dalab/hyperbolic_cones](https://github.com/dalab/hyperbolic_cones)). Instead of a scalar
norm proxy for "more general," they embed the *partial order* itself as nested convex cones: node
`v` entails node `u` (`u` is-a `v`) iff `u` falls inside the cone rooted at `v`. This gives you a
hard containment test, not just a soft ranking — closer to what "give me everything under
Lachnospiraceae" needs. The cones have a closed-form optimal aperture in both Euclidean and
hyperbolic space, derived in the paper. This is the more correct tool if the actual query is
"list all descendants of node X in the embedding," rather than "is A more general than B."

**Caveat on both approaches:** they encode the tree-order relation you *give them as training
pairs*. Your graph already has both a symbolic rank field (species/genus/family from NCBI taxids)
and 727 explicit containment edges — so this machinery would be *learning something you already
know exactly*, from noisier, gradient-descent-approximate positions, unless the goal is
specifically to fuse rank with something else (e.g. embedding position also reflecting the
disease-association profile). See §6.

## 3. Does this work at ~900 nodes, or does it need more data?

**Genuinely small-scale — this is a case where the literature's own benchmarks are the same
order of magnitude as your graph, so this isn't extrapolation.**

Concrete comparators, all from Nickel & Kiela's own experiments (Table 1, Table 2, Figure 2):

| Dataset | N (nodes) | E (edges) | Dim | Result |
|---|---|---|---|---|
| WordNet noun hierarchy (transitive closure) | 82,115 | 743,241 | 10 | Poincaré MAP 0.851 (reconstruction) |
| **WordNet mammals subtree** | **1,180–1,181** | **6,540–6,541** | **5** | **mean rank 1.26, MAP 0.927** — their own headline small-scale demo, visualized in 2-D in their Figure 2 |
| GrQc (collaboration network) | 5,242 | 14,496 | 10 | Poincaré reconstruction MAP 0.990 vs Euclidean 0.522 at the same dim |
| HepPh | 12,008 | 118,521 | 10 | Poincaré reconstruction MAP 0.811 vs Euclidean 0.434 |

Your graph (883 taxa + 40 diseases ≈ 923 nodes, 727 containment edges + 2,008 association edges)
sits almost exactly on top of the **mammals subtree** benchmark — the paper's own "this also
works at small scale" example, embedded and visualized at `d=2` and `d=5` with near-perfect
reconstruction (mean rank 1.26 out of ~1,180). That benchmark is not an ablation on the margins
of the paper; it's the figure they lead with. **The honest reading is that the method actually
prefers being small when the goal is a clean, low-dimensional taxonomy embedding — the paper's
selling point (MAP 0.927 at d=5) is at your scale, not at the 82K-node scale**, where the numbers
never get above MAP 0.87 even at d=200 because there's more genuine ambiguity in the full noun
hierarchy.

Where you *do* need real scale is elsewhere: knowledge-graph-flavored hyperbolic methods (MuRP,
AttH — §5) are usually validated on tens-of-thousands-of-triples KG benchmarks (WN18RR, FB15k-237
have 40K–15K entities), and *large* biological applications (below) run at 10⁴–10⁵ nodes. None of
that is a floor — it's just where those specific follow-on papers happened to test — but it means
if the deliverable grows to include free-text disease/taxon descriptions or literature co-mentions
as additional relation types, you'd be in territory those papers actually validated, whereas the
pure containment-tree embedding is validated at your exact size already.

One more concrete anchor for the small-scale question, since it's about clustering, not just
tree reconstruction: a Geomstats tutorial on hyperbolic embedding notes that for a **34-node**
graph (Zachary's karate club), a 2-D Poincaré disk is already sufficient for a faithful
representation — dimension only needs to grow into double digits once node counts move into the
tens of thousands. That's the opposite failure mode from what you'd worry about (needing *more*
data): the risk at 900 nodes is underfitting the optimizer's negative-sampling noise into
spurious structure, not lacking capacity.

## 4. Applications to NCBI taxonomy / microbial phylogeny specifically

No paper embeds NCBI's taxonomy tree directly with Poincaré embeddings that this search
surfaced, but there is a directly relevant cluster of work in **phylogenetics**, which is the
same combinatorial object (a tree, here with branch lengths) with an extra continuous-optimization
twist:

- **Matsumoto, Mimori, Fukunaga, *"Novel metric for hyperbolic phylogenetic tree embeddings,"***
  *Biology Methods and Protocols* 2021 — [Oxford Academic](https://academic.oup.com/biomethods/article/6/1/bpab006/6192799).
  Defines a hyperbolic-distance variant tuned so pairwise distances better match phylogenetic
  branch-length distances, not just topology.
- **Matsumoto et al., "Learning Hyperbolic Embedding for Phylogenetic Tree Placement and
  Updates,"** *Biology* 2022 — [MDPI](https://www.mdpi.com/2079-7737/11/9/1256),
  [PMC9495508](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9495508/). This is the H-DEPP line of
  work: embed a reference phylogenetic tree's leaves (from gene sequences) in hyperbolic space,
  then place new sequences by nearest-embedding search rather than re-running full tree inference.
  Reported to let a species tree be updated accurately "with only a handful of genes" — i.e. this
  family of methods is explicitly pitched as data-efficient, reinforcing §3.
- **Macaulay, Fourment et al., "Differentiable phylogenetics via hyperbolic embeddings with
  Dodonaphy,"** *Bioinformatics Advances* 2024 — [PubMed 39132286](https://pubmed.ncbi.nlm.nih.gov/39132286/),
  [PMC11310108](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11310108/). Uses hyperbolic
  embeddings plus a differentiable neighbour-joining relaxation ("soft-NJ") to make tree
  *inference* itself gradient-based, rather than embedding a tree that's already known. Less
  relevant to you since your containment tree (NCBI taxonomy) is already given, not something you
  need to infer.
- Closest thing to your actual object (a biological hierarchy queried by rank, at moderate scale,
  for downstream statistics) is **"Multiscale Hyperbolic Embedding for Cell Hierarchies in
  Large-Scale Bioinformatics Data"** — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.29.679407.full.pdf),
  [PMC13530193](https://pmc.ncbi.nlm.nih.gov/articles/PMC13530193/), 2025. Embeds cell-type
  hierarchies (Lorentz/hyperboloid coordinates for optimization, converted to geodesic-polar
  coordinates for interpretation) and explicitly states **"hierarchical depth can be represented
  naturally by radial distance from the origin"** and validates it against known developmental
  time in *C. elegans* (85,333 cells, ~1.1 hr on a single CPU) — i.e. this is the modern,
  actually-maintained-codebase version of the depth-via-radius idea in §2, demonstrated on a
  biological hierarchy at both a much larger scale than yours and (per their own robustness
  analysis) staying reliable down to small cluster counts. Worth reading before building anything,
  since it's the most recent and most on-topic implementation found.

There is no hit for embedding the **disease** side (MONDO, ICD, or similar ontologies) in
hyperbolic space specifically for a microbiome KG — that half of Sam's ask (disease hierarchy +
taxonomy hierarchy jointly) doesn't have a directly reusable paper; it would be combining two
known techniques (disease ontology is already a DAG, same math applies) rather than adapting a
published result.

## 5. Libraries that work today

| Library | Model(s) | Status (as of this search, Sept 2026) | Notes |
|---|---|---|---|
| **`geoopt`** ([github.com/geoopt/geoopt](https://github.com/geoopt/geoopt)) | Poincaré ball (`geoopt.PoincareBall`) + Lorentz/hyperboloid (`geoopt.Lorentz`), general Riemannian manifolds/optimizers | Active — requires `pytorch>=2.0.1`, README explicitly recommends installing from GitHub master over PyPI because "PyPI releases may lag." **This is the recommendation** if you want a general Riemannian-SGD toolkit to plug a custom loss into (e.g. the exact `L(Θ)` in Eq. 6 of Nickel & Kiela). |
| **`facebookresearch/poincare-embeddings`** | Poincaré ball, official Nickel & Kiela reference implementation | **Archived Oct 21, 2025, read-only.** Still usable as reference code / to reproduce their exact numbers, but not a foundation to build on — no custom-graph guidance in the README, only WordNet scripts (`train-mammals.sh`, `train-nouns.sh`). |
| **`dalab/hyperbolic_cones`** | Entailment cones, built on Gensim + autograd (not PyTorch) | Research code released alongside the ICML'18 paper, 7 commits total, no evidence of ongoing maintenance, no custom-taxonomy guidance in the README. Usable to reproduce the paper, not to build on. |
| **`hypll` / HypLL** ([github.com/maxvanspengler/hyperbolic_learning_library](https://github.com/maxvanspengler/hyperbolic_learning_library)) | Poincaré ball now; Lorentz/Klein "future work" per their own paper | Published ACM Multimedia 2023 ([arXiv:2306.06154](https://arxiv.org/pdf/2306.06154)), Python 3.10+/PyTorch 1.11+, `pip install hypll`, has an official tutorial that trains Poincaré embeddings on **exactly the WordNet mammals subset** — [tutorial link](https://hyperbolic-learning-library.readthedocs.io/en/latest/tutorials/poincare_embeddings_tutorial.html) — i.e. closest thing to a turnkey starting point for your scale. Designed explicitly for ease-of-use/onboarding rather than being a research-frontier library. |
| **Gensim's `PoincareModel`** | Poincaré ball | Exists (`rare-technologies.com/implementing-poincare-embeddings`), is what `dalab/hyperbolic_cones` was built from; gensim's broader project has been deprioritizing newer features in recent years — treat as legacy/stable rather than actively extended. |
| **Manify** ([arXiv:2503.09576](https://arxiv.org/pdf/2503.09576)) | Multiple non-Euclidean spaces (product manifolds — combine hyperbolic + spherical + Euclidean factors) | 2025 paper, positions itself as a general "non-Euclidean representations" library — flagged as worth a closer look if you want to jointly model tree-like taxonomy *and* non-tree-like association structure in one product space, but not evaluated in depth here. |

**Practical recommendation among these:** `geoopt` for anything custom (you write the loss,
matching Eq. 6 in Nickel & Kiela almost verbatim against your 727 containment edges), or `hypll`
if you want the fastest path to a working mammals-subtree-style demo before committing engineering
time — its tutorial is trained on data at essentially your scale already.

## 6. Go / no-go recommendation

**Go, but scope it precisely — for the taxonomy-rank part of Sam's ask, not for the "how many
studies support X in direction Y" part.**

- The rank-query ask (§2) is a well-validated, small-data-friendly technique. Your graph is
  the same order of magnitude as the paper's own flagship small-scale demo (1,180-node mammals
  subtree, MAP 0.927 at d=5). This is low-risk, well-understood, and the norm-as-depth /
  cone-as-containment mechanics are exactly what "ask at species vs. genus vs. family" needs.
- **But you already have the rank and the containment edges as ground truth (NCBI taxid rank
  field, 727 containment links).** A pure taxonomy embedding would be re-deriving something you
  can already query exactly with a dictionary lookup. The actual value-add is only there if the
  embedding also has to carry *other* information in the same space — specifically Sam's other
  ask, per-node direction-count statistics (how many papers support taxon X enriched vs. depleted
  in disease Y). That is **not** a hierarchy-embedding problem; it's closer to a **hyperbolic
  knowledge-graph embedding** problem (MuRP — Balažević, Allen, Hospedales, *"Multi-relational
  Poincaré Graph Embeddings,"* NeurIPS 2019; AttH — Chami et al., ACL 2020,
  [aclanthology.org/2020.acl-main.617](https://aclanthology.org/2020.acl-main.617.pdf)), where
  hierarchy is one relation type and the association edges (with their evidence counts as edge
  weights) are another. Those methods are typically validated at 15K–40K-entity scale, i.e. one
  to two orders of magnitude above your graph — the go/no-go there is weaker than for pure
  hierarchy embedding, and would be its own separate research question, not answered here.
- Net: **go on hyperbolic taxonomy embedding as a standalone, low-cost proof of concept.**
  **No** on treating it as already answering the joint "taxonomy + evidence-count" ask — that
  needs either a follow-up literature pass on hyperbolic KG embedding at small scale, or a
  simpler non-hyperbolic fallback (e.g. store direction counts as a node attribute alongside a
  hyperbolic position, don't try to make the geometry encode both).

### Smallest experiment (under a day, on a laptop)

1. `pip install hypll` (or `geoopt` if you want to write the loss by hand).
2. Build the input exactly like Nickel & Kiela's: the list of `(child, parent)` pairs from your
   existing 727 containment edges, **transitively closed** (i.e. also add (genus, family),
   (species, phylum), etc., not just direct parent-child — this is what "transitive closure of
   WordNet" means in their setup and is what let a single embedding answer "is A a descendant of
   B" for non-adjacent ranks).
3. Train a `d=5` and a `d=10` Poincaré embedding with their exact loss (Eq. 6: softmax over
   10 sampled negatives per positive edge) — HypLL's mammals tutorial is literally this recipe
   with your node count already in the right ballpark, so it should be closer to "adapt the
   tutorial's data-loading cell" than "write new training code."
   `PoincareBall` and `RiemannianSGD` are provided by both libraries.
4. Evaluate with their own reconstruction metric: mean rank / MAP of each true (child, ancestor)
   pair's distance among all-other-node negatives. **This is a cheap, deterministic pass/fail** —
   if MAP is anywhere near their 0.85–0.93 range at d=5–10, the hierarchy embeds cleanly and the
   norm-as-depth property (§2) should already be visible by eye in a 2-D training run (their
   Figure 2 pattern: root near center, leaves toward the boundary).
5. Sanity-check the depth signal directly: sort embedded nodes by `‖θ‖` and check that it
   correlates with known rank (phylum < class < order < family < genus < species in norm). This
   is the concrete, falsifiable test of "can we query at genus level by thresholding on radius" —
   if the correlation is weak or non-monotonic at your scale, that's the signal to fall back to
   entailment cones (§2) instead of a scalar-norm threshold.

No GPU, no cluster — this is a few hundred training iterations over ~1,600 (deduplicated
transitive-closure) edges, which the reference implementation runs on CPU in minutes even at
82K-node scale, so at your scale it's a small fraction of that.

---

### All sources

- Nickel, Kiela, *"Poincaré Embeddings for Learning Hierarchical Representations,"* NeurIPS 2017 — [arXiv:1705.08039](https://arxiv.org/pdf/1705.08039), [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html)
- Ganea, Bécigneul, Hofmann, *"Hyperbolic Entailment Cones for Learning Hierarchical Embeddings,"* ICML 2018 — [arXiv:1804.01882](https://arxiv.org/abs/1804.01882), [PMLR 80](https://proceedings.mlr.press/v80/ganea18a.html), code: [dalab/hyperbolic_cones](https://github.com/dalab/hyperbolic_cones)
- Sarkar, *"Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane,"* 2011 — [PDF](https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf)
- De Sa, Gu, Ré, Sala, *"Representation Tradeoffs for Hyperbolic Embeddings,"* ICML 2018 — [arXiv:1804.03329](https://arxiv.org/pdf/1804.03329)
- Matsumoto, Mimori, Fukunaga, *"Novel metric for hyperbolic phylogenetic tree embeddings,"* Biology Methods and Protocols 2021 — [Oxford Academic](https://academic.oup.com/biomethods/article/6/1/bpab006/6192799)
- Matsumoto et al., *"Learning Hyperbolic Embedding for Phylogenetic Tree Placement and Updates,"* Biology 2022 — [MDPI](https://www.mdpi.com/2079-7737/11/9/1256), [PMC9495508](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9495508/)
- Macaulay, Fourment et al., *"Differentiable phylogenetics via hyperbolic embeddings with Dodonaphy,"* Bioinformatics Advances 2024 — [PubMed 39132286](https://pubmed.ncbi.nlm.nih.gov/39132286/), [PMC11310108](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11310108/)
- *"Multiscale Hyperbolic Embedding for Cell Hierarchies in Large-Scale Bioinformatics Data,"* 2025 — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.29.679407.full.pdf), [PMC13530193](https://pmc.ncbi.nlm.nih.gov/articles/PMC13530193/)
- Kochurov, Karimov, Kozlukov, *"Geoopt: Riemannian Optimization in PyTorch,"* 2020 — [github.com/geoopt/geoopt](https://github.com/geoopt/geoopt)
- van Spengler et al., *"HypLL: The Hyperbolic Learning Library,"* ACM Multimedia 2023 — [arXiv:2306.06154](https://arxiv.org/pdf/2306.06154), [github.com/maxvanspengler/hyperbolic_learning_library](https://github.com/maxvanspengler/hyperbolic_learning_library)
- [facebookresearch/poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings) (archived Oct 21, 2025)
- Balažević, Allen, Hospedales, *"Multi-relational Poincaré Graph Embeddings"* (MuRP), NeurIPS 2019
- Chami et al., *"Low-Dimensional Hyperbolic Knowledge Graph Embeddings"* (AttH), ACL 2020 — [aclanthology.org/2020.acl-main.617](https://aclanthology.org/2020.acl-main.617.pdf)
- Manify, *"A Python Library for Learning Non-Euclidean Representations,"* 2025 — [arXiv:2503.09576](https://arxiv.org/pdf/2503.09576)
- Geomstats tutorial, *"Hyperbolic Embedding of Graphs and Clustering"* — [geomstats.github.io](https://geomstats.github.io/notebooks/13_real_world_applications__graph_embedding_and_clustering_in_hyperbolic_space.html)
