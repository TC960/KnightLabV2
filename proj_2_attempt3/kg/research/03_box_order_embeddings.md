# Region/box embeddings for containment-native representation

Narrow question: for a graph with **883 taxa, 727 containment edges, 40 diseases, 2,008
direction-only association edges** where the hierarchy must stay legible (Lachnospiraceae
depleted while its child Hungatella is enriched), do region embeddings (order, box,
cone, ellipsoid/density) beat point embeddings for rank-level queries, and what is the
smallest published graph any of them has been shown to work on.

Status: research complete. All entries below are sourced; PDF-only sources that WebFetch
could not parse are marked and their claims cross-checked against secondary sources
(ADS/dblp abstracts, survey text, GitHub READMEs) rather than asserted from the PDF.

---

## 1. Order embeddings (Vendrov, Kiros, Fidler, Urtasun 2016)

**"Order-Embeddings of Images and Language"**, ICLR 2016. arXiv:1511.06361.
https://arxiv.org/abs/1511.06361 · code: https://github.com/ivendrov/order-embedding

- **Representation.** Points in R^N+ (non-negative orthant), but interpreted as defining
  an axis-parallel cone/region: x ⪯ y iff x_i ≤ y_i for all i. A point *is* the corner of
  its induced region, so this is the minimal region embedding — order embeddings are a
  degenerate case of the box/cone family (a box with one corner pinned at the point, the
  other at +∞), which is exactly why later work (Vilnis et al. below) generalizes it.
- **Scoring.** Violation penalty s(x,y) = ||max(0, y−x)||², zero iff x ⪯ y (parent
  "contains" child under coordinate-wise inequality — containment falls out of the
  ordering with no extra machinery).
- **Input.** Just the partial-order edge list (positive pairs); no textual/attribute
  features required, though the original paper also fuses this with an image/caption
  encoder for its multimodal task.
- **Negative sampling: yes, required.** Training is a margin loss over positive vs.
  corrupted (negative) pairs: L = s(x,y) for true edges + max(0, η − s(x⁺,y⁻)) for
  negatives, i.e. an edge-corruption scheme identical in spirit to standard KGE training.
- **Smallest graph shown to work on:** the WordNet noun hypernym task is the relevant
  one — |T| = 82,115 entities, |clo(T)| = 838,073 edges in the transitive closure (4,000
  held out for test, 4,000 for dev). That's a **large** graph, not a small one; order
  embeddings' original hypernym benchmark is two orders of magnitude bigger than our KG.
  Reported 0/1 accuracy: transitive-closure baseline 88.2%, Gaussian embeddings 86.6%,
  order embeddings 90.6%.
- Direct descendants worth flagging: **Athiwaratkun & Wilson, "Hierarchical Density
  Order Embeddings"** (ICLR 2018, arXiv:1804.09843) fuse this with Gaussian densities
  (see §5) — PDF was not machine-readable via WebFetch, cross-checked only by title/venue,
  flagged as unverified detail beyond that.

## 2. Box embeddings / Box Lattice, Gumbel boxes, and successors

**Foundational: Vilnis, Li, Murty, McCallum, "Probabilistic Embedding of Knowledge
Graphs with Box Lattice Measures"**, ACL 2018. arXiv:1805.06627.
https://aclanthology.org/P18-1025/

- **Representation.** Each concept is an axis-aligned hyperrectangle (box) instead of a
  point. Containment (box A ⊇ box B) *is* the is-a relation — this is the property that
  maps directly onto our containment links: Lachnospiraceae's box would contain
  Hungatella's box, while their disease-association *directions* live outside the box
  geometry entirely (association edges are a separate relation type, not encoded by
  containment).
- **Motivating result:** the paper proves a broad class of probability measures over
  Order Embeddings can never express *negative correlation* between concepts (disjoint
  concepts cannot be represented at all) — boxes fix this because two boxes can be
  contained, overlapping, or fully disjoint, covering all set relations.
- **Input / negative sampling.** Trained on edge lists (WordNet and Flickr entailment
  graphs, per the abstract; exact node/edge counts were not extractable — the arXiv PDF
  and HTML mirror both failed to parse as text under WebFetch, so this figure is a
  documented gap, not a claim). Cross-referencing dblp/ACL Anthology confirms venue,
  authors, page range (263–272) only.
- **Probabilistic / calibrated?** Yes — this is the paper's central claim: boxes support
  "rich joint and conditional queries over arbitrary sets of concepts" with **calibrated
  uncertainty**, both learned from and predicted as probabilities. This directly answers
  question 3 below (P(child|parent)-style queries).

**Gradient problem + fix: Li, Vilnis, Zhang, Boratko, McCallum, "Smoothing the Geometry
of Probabilistic Box Embeddings"**, ICLR 2019. https://openreview.net/forum?id=H1xSNiRcF7

- Hard-edged boxes give **zero gradient** when two boxes are (incorrectly) disjoint,
  because intersection volume and its derivative are both exactly zero — the model gets
  stuck and can never learn to correct a false "disjoint" prediction. Fix: convolve box
  indicator functions with a Gaussian kernel (soft/smoothed volume) so gradient flows
  even through disjoint configurations.

**Successor: Dasgupta, Boratko, Zhang, Vilnis, Li, McCallum, "Improving Local
Identifiability in Probabilistic Box Embeddings"** (introduces **Gumbel boxes**),
NeurIPS 2020. arXiv:2010.04831 (PDF not machine-readable via WebFetch; summary below is
from the earlier ar5iv/GitHub cross-reference on the Gumbel-box mechanism, not the raw
PDF).

- Models each box's min/max coordinates as **Gumbel-distributed random variables**
  instead of fixed numbers. Gumbel is chosen because it's *min/max-stable* — the set of
  Gumbel boxes is closed under intersection, so a learned "intersection box" is still a
  well-formed Gumbel box. The latent noise effectively ensembles over many boxes per
  concept, which is what fixes the local-identifiability/gradient-plateau problem instead
  of (or in addition to) kernel smoothing.

**Reference implementation:** Chheda, Goyal, Tran, Patel, Boratko, Dasgupta, McCallum,
**"Box Embeddings: An open-source library for representation learning using geometric
structures"**, EMNLP 2021 (system demo track). arXiv:2109.04997.
https://github.com/iesl/box-embeddings — PyTorch/TensorFlow, ships MinDeltaBoxTensor,
SigmoidBoxTensor, HardIntersection/GumbelIntersection, and Hard/Soft/BesselApprox volume
functions. This is the practical starting point if we prototype: it already implements
every box variant above as swappable modules.

**KG-completion-flavored successor: Query2box (Ren, Hu, Leskovec)**, ICLR 2020.
arXiv:2002.05969. https://github.com/hyren/query2box — embeds *queries* (not just single
concepts) as boxes so that conjunctions of constraints intersect naturally; evaluated on
FB15k / FB15k-237 / NELL995, all far larger than our graph (14.5k–63k+ entities). Useful
mainly as evidence the box formalism scales to compositional queries, not as a
small-graph precedent.

**Uncertain-KG-flavored successor: Chen, Boratko, Chen, Dasgupta, Li, McCallum, "BEUrRE:
Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning"**, NAACL 2021.
arXiv:2104.04597.

- Entities = Gumbel boxes, relations = affine transforms on head/tail boxes. Confidence
  of a fact = intersection volume of the transformed boxes divided by the tail box's own
  volume — an explicit, literal **P(tail | head, relation)** read off box volumes. This is
  the clearest existing answer to "do probabilistic boxes give calibrated
  P(child|parent)": yes, by construction, volume ratios are used as the conditional
  probability estimator and trained against human-annotated confidence scores.
- Datasets: CN15k (15,000 entities / 241,158 facts), NL27k (27,221 entities / 175,412
  facts) — again much larger than our graph, but the *mechanism* (volume-ratio
  conditional probability) is graph-size-agnostic and is exactly what we'd want for
  "what fraction of Lachnospiraceae-genus mass is Hungatella."
- Negative sampling: yes — 30 corrupted (head, tail) pairs per positive fact.

## 3. Do probabilistic boxes give calibrated P(child | parent)?

Yes, and it's the load-bearing selling point of the whole box line, not an incidental
feature — two independent designs both use box volume ratios as literal probability
estimators:
1. Vilnis et al. 2018 build the entire box-lattice measure around joint/conditional
   probability queries with *learned and predicted calibrated uncertainty* — the paper's
   stated motivation is exactly the failure mode order embeddings have (no negative
   correlation, no calibrated queries).
2. BEUrRE (Chen et al. 2021) makes this completely explicit:
   P(tail | head) := Vol(box_head ∩ box_tail) / Vol(box_tail), trained against real
   confidence-labeled facts (CN15k/NL27k have graded truth values, not just binary
   edges).

Caveat: none of these calibration results were run on anything close to a biological
taxonomy — they're calibrated against WordNet/Flickr entailment judgments or CN15k/NL27k
confidence scores, so "calibrated" here means "matches the *training* distribution's
notion of confidence," not "matches biological reality." For our graph, where containment
is asserted ground truth (from NCBI taxonomy, not a soft label), the calibration question
would really be about the never-observed cells — e.g., estimating P(disease association
generalizes from Hungatella up to Lachnospiraceae) from box volumes, which is closer to
Query2box-style compositional inference than to BEUrRE's confidence-fitting.

## 4. Cone embeddings (hyperbolic entailment cones)

**Ganea, Bécigneul, Hofmann, "Hyperbolic Entailment Cones for Learning Hierarchical
Embeddings"**, ICML 2018. arXiv:1804.01882. https://github.com/dalab/hyperbolic_cones

- Embeds DAGs by representing hierarchical relations as nested **geodesically convex
  cones** in hyperbolic space, with a closed-form optimal cone aperture (derived in both
  Euclidean and hyperbolic geometry). This is squarely a **hyperbolic** paper — flagging
  per your instruction that it overlaps whatever the hyperbolic-literature workstream
  (r1-hyperbolic in this session) already covers.
- **Order/region framing, isolated from the hyperbolic part:** the survey **"Geometric
  Relational Embeddings: A Survey"** (arXiv:2304.11949, ar5iv-readable) classifies plain
  **order embeddings as a special case of axis-parallel cones in Euclidean space**, and
  notes recent variants add an explicit angle parameter to get genuine (non-axis-parallel)
  cones — i.e., cones are the natural generalization that sits *between* order embeddings
  (axis-parallel, zero angle) and full boxes (bounded on both sides). **ConE** (cited in
  the same survey) applies angular cones to complex logical queries and models logical
  negation as the closure-complement of a cone region — a capability boxes don't cleanly
  have (box complement isn't a box).
- No dataset-size figures were extractable for the cone line specifically from the
  survey (see limitation note in §6); Ganea et al.'s own paper uses WordNet noun
  hypernymy (same large 82k/838k-edge setup as Vendrov above) plus a hypernymy-detection
  benchmark — not evidence for our scale.

## 5. Ellipsoid / density embeddings (Gaussian embeddings)

**Vilnis & McCallum, "Word Representations via Gaussian Embedding"**, ICLR 2015.
arXiv:1412.6623. https://arxiv.org/abs/1412.6623

- Each concept → a multivariate Gaussian N(μ, Σ): mean = point location (like a normal
  embedding), covariance = **learned specificity/uncertainty**. A broad covariance can
  represent a general concept (e.g., a phylum) whose distribution overlaps many narrower
  ones; a narrow covariance represents a specific concept (e.g., a species). This is the
  most direct ellipsoid analogue of "family contains genus contains species": containment
  is approximated by one Gaussian's probability mass being mostly inside another's,
  scored by asymmetric KL divergence rather than exact set inclusion.
- **Not exact containment.** Unlike boxes, Gaussian "containment" is soft/statistical —
  there's no hard guarantee that a child's mode lies inside the parent's high-density
  region, which is a real risk for a load-bearing invariant like ours (we need
  Lachnospiraceae to *provably* contain Hungatella, not just "usually overlap it").
- On the original WordNet hypernymy benchmark from §1, Gaussian embeddings scored 86.6%
  vs. order embeddings' 90.6% and the transitive-closure baseline's 88.2% — i.e. Gaussian
  embeddings actually **underperformed** hard order embeddings and even a no-learning
  baseline on this exact task, which is a caution against ellipsoid/density approaches
  for a task where exact partial-order correctness (not just similarity ranking) matters.
- Negative sampling: yes, margin-based ranking loss against corrupted pairs, same family
  as order embeddings' training regime.
- Successor combining both lines: **Athiwaratkun & Wilson, "Hierarchical Density Order
  Embeddings"** (ICLR 2018, arXiv:1804.09843) — puts an explicit order/containment
  constraint on top of Gaussian densities specifically to fix the "soft containment isn't
  guaranteed" problem above. PDF unreadable via WebFetch in this session; flagged as a
  promising but unverified-in-detail lead for whoever picks this up next.

## 6. Applied to biological taxonomy or ontologies (GO/MONDO/ChEBI/NCBI/GTDB/SILVA)?

**No hits for box/order/cone embeddings applied directly to NCBI, GTDB, or SILVA
taxonomy** in this search — this looks like a genuine gap, not a search failure (multiple
query phrasings returned nothing on-topic). The closest biological-taxonomy work found
uses **hyperbolic**, not box/order, geometry:

- **Jeong, Kim, Kim, Sohn, "GeOKG: geometry-aware knowledge graph embedding for Gene
  Ontology and genes"**, *Bioinformatics*, April 2025.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12036960/ — uses one Euclidean + two
  differently-curved **Poincaré-ball hyperbolic** spaces, not boxes/cones. Dataset: 42,950
  GO terms, 83,975 GO-GO edges, 18,137 human genes, 286,628 gene-GO annotations; 50
  negative triples per positive during training; is-a-only link-prediction MRR = 0.242
  for their best variant (GeOKG-H). Relevant as a scale reference (comparable order of
  magnitude to a "real" bio-ontology) but it's evidence *for* the hyperbolic track, not
  box/order.

**Box/order/cone-family work DOES exist on OWL description-logic ontologies**, which is
the right adjacent literature even without a live GO/MONDO/NCBI-taxonomy box paper:

- **Kulmanov, Liu-Wei, Yan, Hoehndorf, "EL Embeddings: Geometric construction of models
  for the Description Logic EL++"**, IJCAI 2019. arXiv:1902.10499. Maps each ontology
  class to an **open n-ball** (not a box) and each relation to a vector, so that the
  embedding is provably a *model* of the TBox axioms (existential restrictions,
  conjunction, bottom). Explicitly targeted at "large EL++ theories... used in the life
  sciences"; downstream-evaluated on protein-protein interaction prediction using GO
  annotations. Uses balls, but is the direct ancestor of the box-based ontology work
  below and is squarely biomedical.
- **Jackermeier et al., "Dual Box Embeddings for the Description Logic EL++"
  (BoxSquaredEL / Box²EL)**, WWW 2024. arXiv:2301.11118.
  https://github.com/KRR-Oxford/BoxSquaredEL — represents **both concepts and roles as
  boxes** (roles as regions rather than fixed translation vectors, with a "bumping"
  mechanism to handle role-inclusion axioms and many-to-many relations, which plain box
  containment can't express by itself). Evaluated on **GALEN, Gene Ontology (GO), and
  Uberon** — real biomedical ontologies — for subsumption prediction, role-assertion
  prediction, and deductive reasoning; state-of-the-art on all three per the abstract.
  Exact node/edge counts for GO/Uberon/GALEN splits were not extractable (dl.acm.org and
  the arXiv abstract page both blocked/insufficient for WebFetch) — this is the single
  most relevant paper for us and is worth a manual follow-up read rather than trusting
  the abstract alone.
- **Memariani et al. (ChEBI box embeddings)**, *Journal of Cheminformatics*, 2025.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12403937/ — boxes for **ChEBI** (chemical
  ontology) classes, 854 training classes (subclasses of "molecular entity"), n=16
  embedding dimension found optimal, is-a scored by literal box-containment
  (parent box ⊇ child box) exactly as our containment links would be, membership of new
  instances scored by point-in-box with a sigmoid-smoothed distance. No explicit negative
  sampling — treats unlabeled class pairs as negative under a closed-world assumption
  (different from the corruption-based sampling of the KG-embedding papers above).
  **This is the closest scale precedent found**: 854 classes is within 2x of our 883 taxa.
- **OWL2Vec\* (Chen, Hu, Jimenez-Ruiz et al.)**, *Machine Learning* journal, 2021.
  arXiv:2009.14654. Not box/order — a random-walk + skip-gram (word2vec-style) ontology
  embedding that folds in graph structure, lexical labels, and OWL logical constructors.
  Widely used in bioinformatics ontology embedding but produces plain point vectors, so
  it does **not** natively represent containment as a geometric region relation the way
  box/order/cone methods do — mentioning it only because it's the standard non-region
  baseline the box papers above compare against.

## 6b. Taxonomy-expansion box work (not biological, but the closest graph *sizes*)

This sub-literature (NLP taxonomy expansion/completion, e.g. WordNet/food/verb
taxonomies) is where the **smallest published graphs** for box embeddings actually live,
smaller than any KG-completion or ontology paper above:

- **Jiang, Song, Zhang, El-Kishky, Yu, "A Single Vector Is Not Enough: Taxonomy
  Expansion via Box Embeddings" (BoxTaxo)**, WWW 2023. doi:10.1145/3543507.3583310.
  https://github.com/songjiang0909/BoxTaxo — learns a box per taxonomy term; a new term
  is attached under whichever candidate parent's box **fully encloses** it, jointly
  optimizing a geometric (containment) and probabilistic (volume-based) objective.
  Evaluated on **SemEval-2016 Task 13** taxonomies: **Environment (209 nodes / 209
  edges)**, Science (344 nodes / 354 edges), Food (1,486 nodes / ~1,533–1,576 edges).
  **The Environment split (209/209) is smaller than our 883-taxon / 727-edge graph by
  ~4x on nodes** — direct proof the box-embedding machinery has been shown to work at a
  scale below ours, not just on 15k+-entity KGs.
- **Xu et al., "Insert or Attach: Taxonomy Completion via Box Embedding"**, ACL 2023.
  arXiv:2305.11004. Extends the same idea to taxonomy *completion* (inserting a node
  between an existing parent and child, not just attaching a leaf) — directly analogous
  to a future need if we ever want to insert an intermediate rank between two existing
  taxonomy nodes. PDF was not machine-readable via WebFetch in this session; only the
  title/venue/problem framing above is confirmed (via search-result summaries), dataset
  sizes not independently verified here.

## Limitations of this search

- Several primary-source PDFs (arXiv and OpenReview) came back as unparsed binary
  through WebFetch in this session (Vilnis et al. 2018 exact WordNet/Flickr sizes,
  Dasgupta et al. 2020 Gumbel-box dataset sizes, BoxSquaredEL's GO/Uberon/GALEN split
  sizes, "Hierarchical Density Order Embeddings," "Insert or Attach," "Representing Joint
  Hierarchies with Box Embeddings" — the last blocked by an OpenReview bot check
  entirely). Every claim above sourced only from a PDF that failed to parse is explicitly
  flagged inline; nothing is asserted from a document this session couldn't actually
  read.
- No box/order/cone paper was found evaluated on a graph with genus-vs-family
  *contested-direction* structure like ours (same taxon, opposite association direction
  at two ranks) — that specific stress test appears to be novel to our data, not
  something any cited paper validates against.

---

## Recommendation

**Region embeddings (boxes specifically) beat point embeddings for our rank-level query
requirement, and the case is strong enough to prototype rather than merely note.**

Reasoning:
1. **Containment is literally the primitive**, not a derived property. Box A ⊇ Box B
   *is* "A is-a ancestor of B" with zero extra parameters — this is exactly Vilnis et
   al.'s and BoxTaxo's design, and it is a structural match to our 727 containment edges
   in a way no point-embedding distance threshold can be (point embeddings need an
   arbitrary radius/threshold hyperparameter to approximate "contains," and that
   threshold cannot simultaneously fit a 4-level taxonomy with wildly different branching
   factors at each rank).
2. **It composes with, rather than fights, our sign-flip requirement.** The
   Lachnospiraceae/Hungatella case needs containment (genus box inside family box) and
   *independently* varying association-edge signs to disease nodes. Box containment only
   constrains the taxon-taxon geometry; disease-association edges can be modeled as a
   separate relation (as BEUrRE and Query2box both do — relations act as transforms
   *on* boxes, not by merging two taxon boxes together), so nothing forces the
   family's and genus's disease-facing edges to agree. A single point embedding with
   cosine/Euclidean similarity has no equivalent decoupling — pulling Hungatella close to
   "depleted-in-PD" pulls it away from its own parent unless a separate hierarchy loss is
   added, at which point you've reinvented a containment constraint anyway, just less
   directly.
3. **Calibrated conditional probability comes for free if we want it later.** If the PI
   eventually wants "what fraction of this family's papers implicate this specific genus"
   as a query, BEUrRE-style volume-ratio conditional probabilities give that directly;
   no comparable well-established point-embedding analogue exists for graded
   parent-to-child mass.
4. **Scale is not a blocker.** The smallest verified working example (BoxTaxo on
   SemEval Environment, 209 nodes) is *smaller* than our graph, and the closest
   biomedical-ontology precedent (ChEBI, 854 classes) is within 2x of our 883 taxa. This
   is a small-graph-friendly method family, not one that needs KG-completion-scale data
   (14k–60k+ entities) to work — the FB15k/CN15k/NL27k numbers above are upper bounds on
   difficulty, not lower bounds on required size.
5. **Ellipsoid/Gaussian embeddings are the one region family to actively avoid** for
   this specific requirement: on the one head-to-head benchmark found (WordNet hypernymy),
   Gaussian embeddings *underperformed both* hard order embeddings and a no-learning
   transitive-closure baseline. Soft/statistical "containment" is also the wrong
   guarantee for us — we need Lachnospiraceae to provably contain Hungatella given that
   it's asserted ground truth from NCBI taxonomy, not a fuzzy tendency to overlap.
6. Cone embeddings are worth keeping on the radar only if the hyperbolic workstream in
   this session (r1-hyperbolic) finds hyperbolic point embeddings compelling for other
   reasons — angular cones are the natural containment-native upgrade *on top of* a
   hyperbolic embedding, and Ganea et al.'s method is literally hyperbolic-space cones.
   Don't adopt cones as a separate third option; they're a refinement of whichever of
   {Euclidean boxes, hyperbolic points} wins first.

**Minimal experiment** (small enough to run before committing further):
- Take the existing 883-taxon / 727-edge containment DAG (already built, `graph.json`)
  and fit a Gumbel-box embedding using the `iesl/box-embeddings` library (EMNLP 2021,
  already cited above) with dimension ~16–64 (Memariani et al.'s ChEBI work found n=16
  sufficient at a comparable node count) and a margin/negative-sampling loss over the
  727 true is-a pairs vs. randomly corrupted non-pairs (standard KGE-style negative
  sampling, as every box paper above uses).
- **Success check, not accuracy metric**: verify the box-containment loss reaches ~0 on
  held-out containment pairs (i.e., the geometry can even *represent* our real 4-level
  hierarchy without contradiction) before touching the association edges at all. This is
  a pure structural sanity check, decoupled from any disease-direction signal.
- Only after that: attach the 2,008 association edges as a second relation (disease-taxon
  affine transform on the taxon's box, à la BEUrRE/Query2box) and test whether a
  genus-level query ("all taxa enriched in Parkinson's under family X") returns
  Hungatella and Lachnospiraceae as geometrically distinct answers with opposite
  predicted signs — i.e., the actual rank-query capability the PI asked for, tested on
  the one case in the corpus already known to require it.
- If that structural check fails at our exact scale (883/727), that alone would be a
  more useful negative result than anything in the literature above, since no cited
  paper tests a hierarchy with our specific contested-direction pathology.
