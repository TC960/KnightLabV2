# GNNs on small biomedical association graphs — are they appropriate at our scale?

**Scope of this note.** Our graph: 883 taxa, 40 diseases, 2,008 direction-only
microbe-disease association edges, 727 taxon-taxon containment edges, sourced from
271 papers. No node features beyond names/NCBI taxids — no abundance matrices,
sequences, or patient-level data. Question: is a GNN justified here, or is this a
"do not bother" situation.

---

## 1. Published microbe-disease association (MDA) prediction methods

The dominant benchmark in this literature is **HMDAD** (Human Microbe-Disease
Association Database): **292 microbes, 39 diseases, 450 associations** (Ma et al.
2017, cited via the 2021 survey below) — this is *within a factor of 2-4x of our
graph on every axis* (taxa 883 vs 292, diseases 40 vs 39, edges 2,008 vs 450).
**Disbiome**, the larger benchmark, has been reported at different sizes across
papers depending on filtering: **1,622 microbes / 374 diseases / 8,731
associations** in one accounting, or a filtered **1,052 microbes / 218 diseases /
4,351 associations** in another (Disbiome database papers, PMC5987391 and
follow-ups). Every MDA-prediction paper below trains and evaluates on graphs at
or below our scale — this is a small-graph subfield across the board, not a
"we happen to be small" situation.

| Method | Year | Graph / features | Eval protocol | Reported AUC |
|---|---|---|---|---|
| **KATZHMDA** (Chen et al.) | 2017 | HMDAD only; pure network topology (Katz path-counting index) + Gaussian interaction-profile kernel similarity, no external features | Global + local LOOCV | 0.8382–0.8644 global; ~0.699–0.7688 local (numbers vary slightly across papers re-running it) |
| **NinimHMDA** (Ma et al., *Bioinformatics* 2020) | 2020 | Multiplex heterogeneous network combining HMDAD + Disbiome + other DBs; GCN-based; first to predict *direction* (increase/decrease) as two association types | Large-scale k-fold CV + case studies | reported "highly competitive," specific AUC not surfaced in secondary sources — original paper needed for the number |
| **MNNMDA** (matrix nuclear-norm minimization) | 2022 | Three datasets scaled small/medium/large: HMDAD, Disbiome, Combined | 5-fold CV | 0.9536 (HMDAD), 0.9364 (Disbiome) |
| **GATMDA** (Long et al., *Briefings in Bioinformatics* 2021) | 2021 | HMDAD + Disbiome; graph attention network + inductive matrix completion; features built from gene-gene interaction, disease-gene association, GO-derived similarity — i.e. **external biological features we do not have** | Compared against 7 SOTA methods; case studies (asthma, IBD) | outperformed all 7 baselines (specific AUC not isolated in secondary source) |
| **GCNMDA** (Long et al., analogous microbe-*drug* variant, *Bioinformatics* 2020) | 2020 | GCN + conditional random field, MDAD dataset | 5-fold CV | **0.9423 ± 0.0105** |
| **MVGCNMDA** | 2022 | Multi-view graph augmentation GCN | 5-fold CV | AUC 0.9428, AUPR 0.9440 |
| **DBGCNMDA** (dual-branch GCN) | 2024 | — | 5-fold CV | AUC 0.9559, AUPR 0.9630 |
| **GCNATMDA** (GCN + attention) | ~2024 | — | 5-fold CV | AUC 0.9659, AUPR 0.9301 |
| **NRGCNMDA** (residual GCN + CRF) | 2025 | — | 5-fold CV | AUC 0.9516, AUPR 0.9302 |

**Pattern worth naming explicitly:** the *simplest* method in this table —
KATZHMDA, pure topology, no learned features, no GNN — gets AUC 0.84–0.86.
Every GNN variant published since clusters tightly in **0.94–0.97**, a gap of
roughly 0.08–0.13 AUC. That gap is real evidence that *something* about
learned representations helps on this data — but section 2 argues the
evaluation protocol producing all of these numbers is itself compromised, so
the gap's size cannot be taken at face value, and the gap between different
GNN variants (0.94 to 0.97, i.e. all bunched within 0.03 of each other despite
architectural differences from GCN to attention to dual-branch) is a classic
sign of a benchmark that has saturated, not of steadily-improving science.

## 2. Is the 0.90+ AUC credible? — critiques of evaluation in this subfield

This is the load-bearing section. Three independent lines of critique apply,
none specific to microbiome but all directly on point.

### 2a. "Unobserved" is not "negative" — and this survey says so about itself

The 2021 *Briefings in Bioinformatics* survey on MDA prediction (Peng et al.,
"A survey on predicting microbe-disease associations," DOI via
academic.oup.com/bib/article/22/3/bbaa157) states plainly: **"There is yet no
certain microbe-disease non-association dataset to date. The label '0' has
two possible interpretations, unknown association or non-association."**
Every AUC in the table above is computed against negatives drawn from
literature *silence*, not confirmed absence. In a literature-derived graph
like ours, silence means "no paper happened to test that pair" — exactly the
same ambiguity our own KG has for any taxon-disease pair we didn't extract.
A held-out true positive competing against a shuffled bag of "probably-just-
untested" pairs is an easy discrimination task almost by construction, and
inflates AUC independent of whether the model learned anything biologically
meaningful. Proposed fixes in this literature (ABHMDA's balanced sampling,
BMCMDA's probabilistic non-association modeling, PU-learning approaches like
RNMFMDA) are attempts to patch this, not evidence it's solved.

### 2b. Pair-input leakage — Park & Marcotte's C1/C2/C3 framework

**Park & Marcotte, "Flaws in evaluation schemes for pair-input computational
predictions," *Nature Methods* 2012** (pubmed 23223166) is the canonical
citation here, written about protein-protein interaction prediction but
mechanically identical to microbe-disease prediction: both are pair-input
problems on a bipartite-ish graph. Their finding: **random k-fold CV over
edges, not over nodes**, silently creates a test set dominated by pairs that
share one or both endpoints with the training set. They define three classes:
**C1** (test pair shares both entities with training), **C2** (shares one),
**C3** (shares neither — the true generalization test). Performance is
systematically inflated in C1 > C2 > C3, and most pair-input papers — this
includes every 5-fold-CV MDA paper in the table above, since none reports a
node-disjoint split — report an unweighted average dominated by C1/C2. **None
of the MDA papers surveyed here report C1/C2/C3-separated performance.** This
is precisely the "warm start masquerading as generalization" failure mode.

The DTI (drug-target interaction) literature — a close cousin, same pair-input
structure — has already run this experiment explicitly:
**Pahikkala et al., "Toward more realistic drug-target interaction
predictions," *Briefings in Bioinformatics* 2015** shows that switching from
random-pair CV to a cold-start split (a drug family or target family entirely
held out) causes reported performance to collapse from the 0.90s to
substantially lower — the whole point of the paper is that the "easy" setting
and the "realistic" (cold-start) setting are not the same experiment and the
gap between them is large. This is the single most transferable result for
us: **whatever headline AUC a microbe-disease GNN paper reports is the easy
number until proven otherwise**, because none of the papers in section 1
state they used anything but random/LOOCV splits.

### 2c. Negative-sampling difficulty is itself gameable, generically (not microbiome-specific, but directly applicable)

**Li et al., "Evaluating Graph Neural Networks for Link Prediction: Current
Pitfalls and New Benchmarking," NeurIPS 2023 (arXiv:2306.10453)** — general
link-prediction, but the mechanism is exactly what's missing from every MDA
paper above. Their core finding: **standard practice draws negative test
edges uniformly at random from all non-edges**, and on typical graphs "nearly
all randomly sampled negatives share zero common neighbors" with either
endpoint, making them trivially distinguishable from positives by any method,
GNN or not. They introduce HeaRT (Heuristic-Related Sampling Technique),
which restricts negatives to *hard* corruptions (one true endpoint kept,
selected via multiple heuristics), and re-benchmark. Result: **under HeaRT,
simple heuristics ranked in the top 3 methods 10 times, vs. only 5 times under
the standard easy-negative protocol** — e.g., on ogbl-collab, plain **Katz**
scored the best MRR (47.15) and Shortest Path second (46.71), beating GNN
baselines outright. Standard-deviation across repeated runs dropped 36–92%
under the harder protocol, meaning the *easy* protocol wasn't just inflating
scores, it was making the comparisons noisy and unreliable. Since every MDA
paper in section 1 uses the easy (random-negative) protocol Li et al. show
is gameable, the reported 0.94+ AUCs should be read as an upper bound on a
metric that is known, in the general case, to collapse toward simple
baselines once negatives are made realistic.

### 2d. AUC itself is the wrong metric under class imbalance

**Yang, Lichtenwalter & Chawla, "Evaluating Link Prediction Methods,"
*Knowledge and Information Systems* 2015 (arXiv:1505.04094)** — AUC "tends to
overrate algorithms that can rank many negative samples at the bottom,"
which is nearly free in a link-prediction setting with severe class
imbalance (almost all possible pairs are non-edges). They recommend
precision-recall curves instead. Every table entry in section 1 reports AUC
and nothing else — no precision-recall numbers appear in any secondary
source we could retrieve, which is itself informative.

### 2e. The general link-prediction embeddings literature agrees, from a different angle

**Menand & Seshadhri, "Link prediction using low-dimensional node
embeddings: The measurement problem," PNAS 2024** (10.1073/pnas.2312527121) —
not MDA-specific, but relevant because most MDA-GNN methods *are* low-
dimensional embedding methods (GCN/GAT encoders producing small microbe and
disease vectors, scored by dot product or inductive matrix completion, which
is dot-product-shaped). Menand & Seshadhri prove that **standard AUC-based
evaluation of low-dimensional embeddings is "based on faulty measurements,"**
and that low-dimensional dot-product embeddings mathematically **cannot
capture sparse ground-truth structure** the way AUC makes it appear they do.
Testing 12 embedding methods (DeepWalk, node2vec, NetMF, GraphSAGE, HOP-Rec)
with a stricter vertex-centric metric (VCMPR@k) instead of AUC, performance
was markedly worse than the AUC numbers suggested. This is an independent,
theoretical reason (not just a sampling-protocol critique) to distrust
dot-product-style AUC numbers from any of the MDA-GNN architectures above,
essentially all of which score edges by an inner product of learned microbe
and disease embeddings.

**Verdict on credibility:** every one of five independent critique lines —
(a) "unobserved ≠ negative" acknowledged in the field's own survey, (b)
pair-input leakage with no C1/C2/C3 reporting, (c) easy-negative sampling
shown in general link-prediction to erase the GNN-vs-heuristic gap, (d) wrong
metric for the imbalance regime, (e) a theoretical result that low-dim
embedding AUC is measuring the wrong thing — apply directly and none has been
addressed by any MDA-GNN paper surveyed. **The 0.90+ AUCs should not be taken
at face value; the credible floor is closer to KATZHMDA's 0.84, and even that
number is measured under the same compromised protocol.**

## 3. Bipartite / recommender framing

A microbe-disease matrix (rows = taxa, columns = diseases, cell = direction of
association or evidence count) is structurally a recommender-systems
interaction matrix, and several MDA methods are explicitly framed this way
(inductive matrix completion in GATMDA, matrix nuclear-norm minimization in
MNNMDA, graph-regularized non-negative matrix factorization). **LightGCN**
(He et al. 2020) is the relevant modern baseline: it strips GCN feature
transformation and nonlinearity entirely, keeping only weighted neighbor
aggregation, and beats NGCF (a heavier GCN-based recommender) by ~16%
relative — the lesson transfers directly: *on bipartite interaction data,
architectural complexity is not obviously buying anything, and the field's
own strongest recent baseline is the simplified one.* A "Just Propagate"
unifying framework (arXiv:2410.21325) makes the same point formally: matrix
factorization, network embedding, and LightGCN are shown to be special cases
of one propagation operator, meaning the apparent diversity of methods in
section 1's table is smaller than it looks — many are minor variations on
the same low-rank completion idea, which is exactly what KATZHMDA is doing
with a path-counting index instead of a learned embedding.

**For us specifically:** we don't have interaction *counts* in the
collaborative-filtering sense uniformly — we have per-edge paper counts and a
direction label, but the underlying signal is "does the literature report
this pair" rather than "how much does this user like this item." A pure
matrix-completion/LightGCN-style approach is a more honest baseline to try
before any message-passing GNN, precisely because it makes no claim to be
using graph structure beyond the bipartite adjacency itself.

## 4. How many edges before message passing beats a trivial baseline?

No paper in either the MDA literature or the general link-prediction
literature states a hard edge-count threshold ("GNNs need N edges"); the
better-supported finding is **structural, not just numerical**:

- **Revisiting Link Prediction: A Data Perspective** (arXiv:2310.00793,
  Guo et al.) frames it as **local vs. global structural proximity**: GNNs'
  advantage over heuristics shows up specifically when *local* proximity
  (common neighbors, Adamic-Adar) is *deficient* — i.e., in sparse regions of
  the graph — and even then, GNN-for-link-prediction methods
  **"consistently underperform on edges where the feature-proximity factor
  dominates."** We have no node features, so this failure mode doesn't apply
  to us directly, but it does mean the paper's own stated condition for GNN
  advantage (sparse local structure) needs checking against our graph before
  assuming it holds.
- A biomedical-specific data point: a BMC Bioinformatics multi-dimensional
  evaluation of graph-embedding approaches on realistic biomedical graphs
  found that on the smallest graph tested (~15,000 edges) and on graphs with
  ~14% disconnected nodes, **simple Common Neighbours was "a justifiable
  choice"** and neural methods did not clearly win; the neural advantage
  appeared only once node representations had enough data to be learned well
  and disconnection was low. **15,000 edges is 7.5x our 2,008**, and that
  paper is describing that scale as still a wash. We should read our own
  2,008 edges as sitting below, not at, the point where that literature
  starts to see daylight for GNNs.
- The clearest generic result, again from Li et al. (section 2c): under a
  fair (hard-negative) evaluation, **Katz and Shortest Path — both pure
  topology, zero learning — outrank GNN methods outright** on real benchmark
  graphs that are far larger than ours (ogbl-collab has ~1.28M edges). If
  simple topological heuristics can out-rank GNNs on a graph 600x our size
  under a fair test, there is no basis for expecting a GNN to earn its
  complexity on 2,008 edges.

**Bottom line for point 4: there is no published edge-count threshold, but
every comparison we could find that used a fair protocol and a graph within
one or two orders of magnitude of ours (15K edges, or even 1.28M edges under
hard negatives) had simple baselines competitive with or beating GNNs.** We
have not found a single fair (non-leaky, hard-negative) comparison anywhere
in this space where a GNN beat simple heuristics on a graph as small as ours.

## 5. Cold start — can any of them say anything about a taxon or disease with ONE paper?

This is where the literature is most honest about its own limits, and it
matters most for us because **most of our edges come from few papers** (per
the root CLAUDE.md: contested-edge and single-paper-edge structure is
pervasive in this graph).

- The MDA survey states directly: **"most scoring function-based models are
  not applicable to new diseases"** — i.e., a disease/microbe with zero prior
  associations breaks similarity-based methods (KATZHMDA, network-consistency
  projection, etc.) outright, because their similarity kernels are built from
  the association matrix itself. A one-paper node is not the zero-paper case
  these methods fail on, but it is the *adjacent* regime: with one edge, a
  node's row/column in the similarity matrix is nearly degenerate, and
  Gaussian-interaction-profile-style kernels (used by KATZHMDA, GATMDA, most
  of the table) are dominated by noise at that support level.
- **ABHMDA** is flagged as an exception explicitly built to handle new
  diseases with no known associations at all (via a boosting ensemble rather
  than pure similarity), but it is reported only at global/local LOOCV
  (0.8869/0.7910) — a genuinely harder split than the 5-fold random CV most
  others use, and its AUC drop relative to the GNN cluster (0.94+) is
  suggestive: **the one method explicitly designed for the cold case scores
  worse**, consistent with cold-start being a real, not just theoretical,
  drop in achievable performance, not an artifact of a weaker model.
- No method in this table reports a stratified breakdown by evidence count
  (1-paper edges vs. 10-paper edges) analogous to what our graph's provenance
  actually looks like. This is a genuine gap in the literature relative to
  our situation, not just an evaluation-protocol quibble: **nobody has
  published the number we'd actually need — accuracy on edges supported by a
  single paper** — because none of these benchmarks carry per-edge evidence
  counts as a graph attribute at all. Disbiome/HMDAD collapse each pair to a
  single binary or categorical label; our graph is unusual in the field for
  keeping per-edge paper-level provenance, and that is exactly the dimension
  none of this literature stress-tests.

## 6. Verdict

**A GNN is not justified for this graph, and the burden of proof should be
placed accordingly.**

Reasoning:

1. **Scale.** 2,008 edges / 883 taxa / 40 diseases is at or below the scale of
   the standard MDA benchmark (HMDAD: 450/292/39) where even the *published,
   optimistic* literature shows the simplest possible method (KATZHMDA, pure
   Katz-index topology, no learning) already gets AUC 0.84, within ~0.10 of
   every GNN variant published since. On a fair evaluation (section 2), that
   gap is not established to survive.
2. **No node features.** We have names and taxids, nothing else. Every MDA-
   GNN method that clears 0.95 AUC (GATMDA, GCNATMDA, etc.) does so by
   injecting external biological features (gene-gene networks, disease-gene
   associations, GO terms) we explicitly do not have. Without them we would
   be comparing a feature-less GNN against feature-less heuristics — the
   exact regime where section 2's critiques (leakage, easy negatives) explain
   *all* of the apparent GNN advantage, because there's no external signal
   left for the GNN to exploit that a heuristic couldn't.
3. **No credible negative-sampling story.** Our graph has the identical
   "unobserved ≠ negative" problem the field's own survey names, and treating
   an untested taxon-disease pair as a true negative is not more defensible
   for us than it is for HMDAD/Disbiome.
4. **Cold start is our actual regime**, and it's exactly where this
   literature is weakest — methods either fail outright on zero-association
   nodes, or (ABHMDA) score worse when adapted to handle them, and nobody
   benchmarks the one-paper-edge case that describes most of our graph.
5. **Every fair external comparison we found — general link prediction, not
   just MDA — has simple topological heuristics competitive with or beating
   GNNs at our scale or larger.**

**What would change the verdict:** a GNN (or embedding method) would need to
beat, on a **node-disjoint (cold-start, Park & Marcotte C3-style) split**,
all of:
   - **Node degree** (does the taxon or disease just appear in a lot of
     papers)
   - **Common neighbors / Adamic-Adar** on the taxon-taxon containment graph
     (does a sibling/parent taxon already have this disease association)
   - **Personalized PageRank** from the disease side
   - **A plain matrix-factorization / LightGCN-style baseline** on the
     taxon x disease matrix, no message passing

If a GNN cannot beat that baseline set under a cold-start split, on our
2,008-edge graph, with no node features — **it should not be built.** Given
everything above, our expectation should be that it will not, and the more
productive use of effort is almost certainly the corpus-scale GPU
re-extraction work already flagged elsewhere in this project (edge recall,
corpus screening) rather than a GNN layer on top of a graph this size.

---

### Sources

- Chen et al., KATZHMDA, *Bioinformatics* 2017 — HMDAD, LOOCV AUC 0.8382–0.8644 (via secondary citation)
- Peng et al., "A survey on predicting microbe-disease associations," *Briefings in Bioinformatics* 22(3), 2021 (academic.oup.com/bib/article/22/3/bbaa157/5881365) — HMDAD size, negative-label ambiguity quote, LOOCV table
- Ma et al. (NinimHMDA), *Bioinformatics* 36(24), 2020 (academic.oup.com/bioinformatics/article/36/24/5665/6069539)
- MNNMDA, *ScienceDirect* 2022 (S2001037022006080)
- Long et al. (GATMDA), *Briefings in Bioinformatics* 22(3) bbaa146, 2021
- Long et al. (GCNMDA, microbe-drug variant), *Bioinformatics* 36(19) 4918, 2020
- MVGCNMDA, PubMed 35428964, 2022
- DBGCNMDA, PMC11291560 / PMC11671253, 2024
- Disbiome database, PMC5987391 and follow-on usage papers (size varies by filtering: 1,622/374/8,731 vs. filtered 1,052/218/4,351)
- Park & Marcotte, "Flaws in evaluation schemes for pair-input computational predictions," *Nature Methods* 9, 1223–1225 (2012), PMID 23223166
- Pahikkala et al., "Toward more realistic drug–target interaction predictions," *Briefings in Bioinformatics* 16(2), 325 (2015)
- Li et al., "Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking," NeurIPS 2023, arXiv:2306.10453
- Yang, Lichtenwalter & Chawla, "Evaluating Link Prediction Methods," *Knowledge and Information Systems* 45, 751–782 (2015), arXiv:1505.04094
- Menand & Seshadhri, "Link prediction using low-dimensional node embeddings: The measurement problem," *PNAS* 121(8), 2024, DOI 10.1073/pnas.2312527121
- He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation," SIGIR 2020
- "Just Propagate: Unifying Matrix Factorization, Network Embedding, and LightGCN for Link Prediction," arXiv:2410.21325
- Guo et al., "Revisiting Link Prediction: A Data Perspective," arXiv:2310.00793
- BMC Bioinformatics, "Neural networks for link prediction in realistic biomedical graphs: a multi-dimensional evaluation of graph embedding-based approaches," 2018 (PMC5963080)
