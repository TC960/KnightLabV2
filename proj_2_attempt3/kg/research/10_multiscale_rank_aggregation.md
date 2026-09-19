# Multi-scale / multi-rank representation and aggregation in microbiome analysis

Motivating constraint (measured in this project, see root `CLAUDE.md`): in Parkinson's,
the family *Lachnospiraceae* is depleted across 15 papers while the genus *Hungatella*
inside it is enriched across 7 (one study reports both directions itself). Corpus-wide,
related taxa (ancestor–descendant pairs within a single paper) agree on direction 89% of
the time vs. 54% for unrelated taxa — so containment is *mostly* redundant, and the ~11%
that disagree is exactly the case that a fixed-rank collapse would erase.

PI's request under investigation: *"we need a taxonomy mapping in the embedding space as
well, because we can ask questions like you are asking at species level or genus level or
family level"* — the same question must be answerable at multiple taxonomic ranks
**without silently merging them.**

This note surveys how the microbiome-statistics and graph-learning literatures handle
that tension, then proposes a concrete scheme for this project.

---

## 1. Agglomeration to a fixed rank (`tax_glom` / QIIME `collapse`) and its criticism

The standard move is to collapse an OTU/ASV table to a chosen rank before testing:
phyloseq's `tax_glom()` merges all taxa sharing the same label at a target rank (analogous
to `tip_glom()`, which does the same thing using branch-length distance on the tree instead
of the categorical taxonomy string) [phyloseq docs](https://www.rdocumentation.org/packages/phyloseq/versions/1.16.2/topics/tax_glom); QIIME's `collapse` action does the
equivalent on a feature table + taxonomy artifact.

**What is lost, documented in the tooling and the methods literature itself:**

- **Everything below the chosen rank is discarded, not summarized.** phyloseq's own docs
  note that ranks to the right of the agglomeration level become meaningless and are set to
  `NA` — there is no record left of *which* genus contributed how much to a family total, and no
  way to recover that a family's evidence was actually 6 genera unanimous and 1 genus opposite,
  vs. 15 papers all reporting the family directly.
- **Taxa unresolved to the target rank are silently dropped or merged into "unclassified".**
  Reported repeatedly as a practical failure mode in phyloseq issue threads (e.g.
  [#927](https://github.com/joey711/phyloseq/issues/927), [#1619](https://github.com/joey711/phyloseq/issues/1619)) — organisms that can't be
  classified to species vanish entirely from a species-level glom, which is a form of
  differential, taxon-dependent missingness, not random noise.
- **The choice of rank changes both power and the answer, not just resolution.** The
  multi-scale differential-abundance literature (below) states this plainly: results
  "depend largely on the choice of analysis unit," and "as the number of analyzed taxa
  becomes smaller [i.e., coarser rank], the compositional effect becomes stronger" —
  agglomeration doesn't just lose information, it changes the compositional-data
  statistics of what remains ([Xiao, Chen et al., *Bioinformatics* 2023, "Multiscale adaptive differential abundance analysis in microbial compositional data"](https://academic.oup.com/bioinformatics/article/39/4/btad178/7108773)).
- **Different DA tools and preprocessing choices produce drastically different taxon sets**
  even at a fixed rank ([Nearing et al., *Nature Communications* 2022, "Microbiome differential abundance methods produce different results across 38 datasets"](https://www.nature.com/articles/s41467-022-28034-z); [Nearing et al., *Microbiome* 2022, comprehensive DA-methods evaluation](https://link.springer.com/article/10.1186/s40168-022-01320-0)) — so rank-collapse
  error compounds with method-choice error rather than being a clean, separable step.

Net: `tax_glom`-style collapse is a one-way, irreversible projection. Any system that wants
to answer the *same* question at several ranks cannot be built on top of pre-collapsed
tables — it needs to keep the full tree and defer the rank choice to query time.

## 2. Hierarchy-aware multiple testing (hierarchical FDR / tree-based procedures)

A parallel statistics literature asks: rather than picking one rank and testing all nodes
at that rank, why not test **every node of the taxonomy at once**, with an error-control
guarantee that accounts for the tree structure?

- **Yekutieli, D. (2008), "Hierarchical False Discovery Rate–Controlling Methodology,"
  *JASA* 103(481):309–316** ([PDF](http://www.math.tau.ac.il/~yekutiel/papers/JASA%20FDR%20trees.pdf)). The foundational procedure: hypotheses are arranged in
  a tree; BH is applied at the root, and a child node is tested *only if its parent was
  rejected*. This top-down, conditional structure controls FDR among the "outer"
  (furthest-tested) nodes at level ≤ 2·D·q·δ*, D = tree depth. Directly transferable
  to a taxonomy: test "is phylum X associated?", only descend into its classes/orders/
  families/genera if the phylum-level test rejects. Implemented as `structSSI::hFDR.adjust`
  in R (Sankaran & Holmes, *J. Stat. Software*; [CRAN vignette](https://cran.r-project.org/web/packages/structSSI/structSSI.pdf)).
- **TreeFDR / StructFDR — Xiao, J., Cao, H., Chen, J. (2017), "False discovery rate control
  incorporating phylogenetic tree increases detection power in microbiome-wide multiple
  testing," *Bioinformatics* 33(18):2873–2881** ([paper](https://academic.oup.com/bioinformatics/article/33/18/2873/3824757)). An empirical-Bayes hierarchical model
  puts a structure-based prior (built from the tree/phylogenetic distances) over per-taxon
  effect sizes, so a taxon's test statistic is *moderated* by its neighbors' — "borrowing
  strength" across the tree — then uses permutation for FDR control. Generalizes beyond
  trees to `structFDR`, which accepts any feature-distance matrix. Important: when the tree
  is uninformative or misspecified, TreeFDR degrades gracefully to the un-moderated result
  (no power loss), which matters because our taxonomy is a hard containment tree, not a
  branch-length phylogeny, and containment ≠ functional similarity 11% of the time.
- **Countervailing null result — Sankaran, Holmes et al. (2020), "Incorporating
  Phylogenetic Information in Microbiome Differential Abundance Studies Has No Effect on
  Detection Power and FDR Control," *Frontiers in Microbiology*** ([paper](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2020.00649/full); also posted to bioRxiv). Using
  realistic simulations, this paper found phylogeny-aware DA methods (including tree-based
  moderation) gave **no measurable improvement** over phylogeny-naive methods in several
  benchmark settings. This is the load-bearing caveat for any tree-borrowing approach here:
  the gain is real in the original TreeFDR benchmarks but is not universal, and should be
  validated on our own taxon–disease data before being trusted, exactly the kind of
  "measure it, don't assume it" standard the rest of this project holds itself to.
- **treeclimbR — Huang, Soneson, Robinson et al. (2021), "treeclimbR pinpoints the
  data-dependent resolution of hierarchical hypotheses," *Genome Biology* 22:157**
  ([paper](https://link.springer.com/article/10.1186/s13059-021-02368-1)). Rather than fixing a rank *or* testing every node independently, treeclimbR
  searches the tree for the resolution (which may differ by branch!) at which a signal is
  best expressed, then reports candidate cut levels with FDR control across candidates.
  Applied to microbiome (infant gut, vaginal vs. C-section delivery) and single-cell data.
  This is the most direct prior art for "sometimes the true signal lives at genus, sometimes
  at family, and the tree itself should tell you which, per-branch" — i.e., a statistical
  answer to why *Lachnospiraceae* (family-resolved) and *Hungatella* (genus-resolved) are
  both "real" but at different cuts of the same tree.
- **Multiscale adaptive DA (MsRDB) — Xiao et al. (2023), *Bioinformatics* 39(4):btad178**
  ([paper](https://academic.oup.com/bioinformatics/article/39/4/btad178/7108773)). Embeds taxa into a metric space from the tree and adaptively smooths
  test statistics across scales, explicitly to find "the finest resolution offered by the
  data" while remaining robust to sparse/zero counts — another instance of not
  pre-committing to a rank.
- **Phylofactorization — Washburne, Silverman et al. (2017), "Phylogenetic factorization of
  compositional data yields lineage-level associations in microbiome datasets," *PeerJ* 5:e2969**
  ([paper](https://peerj.com/articles/2969/)), with the general method in Silverman et al., "Phylogenetic factorization of
  compositional data," *bioRxiv*. Iteratively identifies the **edge** of the phylogeny
  (not a fixed rank!) that best explains variation, builds an orthonormal ILR-style basis
  from that edge, removes its contribution, and recurses — producing a ranked list of
  "phylogenetic factors" each anchored at whatever clade depth the data supports. This is
  the clearest existing instance of "aggregate evidence at the scale the data indicates,
  and represent that scale explicitly" rather than picking rank first.

## 3. Is a family-level effect actually shared by every genus inside it? (empirical rank-specificity)

Several independent literatures converge on **no, not reliably**:

- The multiscale-DA papers state directly that **the appropriate rank is not knowable in
  advance** and must be searched for — treating rank as a hyperparameter of the analysis,
  not a fixed property of the biology ([Xiao et al. 2023](https://academic.oup.com/bioinformatics/article/39/4/btad178/7108773)).
- **TaxaHFE — Oliver et al. (2023), "TaxaHFE: a machine learning approach to collapse
  microbiome datasets using taxonomic structure," *Bioinformatics Advances* 3(1):vbad165**
  ([paper](https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad165/7453373), [GitHub](https://github.com/aoliver44/taxaHFE)). Built explicitly because
  the informative rank is trait-specific and not known a priori: it selects, feature by
  feature, whichever level of the taxonomy carries the most information for the outcome
  while discarding redundant levels — implying empirically that a single fixed collapse
  rank is suboptimal across almost any real dataset (90% feature reduction relative to
  using the full taxonomy, meaning most levels *are* redundant, but not the same 10% every
  time).
- The original **Hierarchical Feature Engineering** paper (Oudah & Henschel,
  *BMC Bioinformatics* 2018, "Taxonomy-aware feature engineering for microbiome
  classification") frames this as a search problem across taxonomic levels for exactly
  the same reason.
- Nearing et al.'s cross-dataset DA benchmarking (*Nat. Commun.* 2022) found that even
  within one rank, ~80% of genera were flagged by *at least one* method in a given
  dataset, but only 46% replicated across two datasets, 18% across a majority of methods,
  and <1% across all methods — evidence that "signal" in this literature is fragile even
  before crossing ranks, which raises the bar for what "shared across ranks" should mean:
  it needs a null distribution, not a raw disagreement count.
- **This project's own number is the cleanest positive evidence of true rank-specific
  disagreement in a fully independent regime**: 89% within-paper concordance for
  related taxa vs. 54% for unrelated ones, with *Lachnospiraceae*/*Hungatella* in Parkinson's
  as a concrete, high-n (15 vs. 7 papers) counter-example to "family effects propagate to
  every child genus." No external empirical paper found in this search reports a directly
  comparable ancestor–descendant concordance rate for microbiome taxa — this may be worth
  a short methods note of its own; the 89/54% split is more precise than the qualitative
  "rank matters" statements this literature otherwise offers.

## 4. Phylogeny-aware distances and representations: UniFrac, PhILR

Two established ways of turning "which taxa are related, and how closely" into numbers
that standard statistics can consume, both directly germane to an embedding-space
taxonomy mapping:

- **UniFrac** (Lozupone & Knight, *Applied and Environmental Microbiology* 2005; weighted
  variant Lozupone et al., *AEM* 2007) — a **distance between samples**, not taxa: the
  fraction of total branch length on the shared phylogenetic tree that leads to
  descendants unique to one sample vs. the other. *Unweighted* UniFrac uses only
  presence/absence; *weighted* UniFrac additionally weights branches by relative
  abundance and is a special case of the earth-mover's / Kantorovich–Rubinstein distance
  ([Evans & Matsen, *arXiv* 2010](https://arxiv.org/pdf/1005.1699); [Lozupone et al., generalized UniFrac, *Bioinformatics* 2012](https://academic.oup.com/bioinformatics/article/28/16/2106/324465)). UniFrac answers "how far apart are two
  communities, respecting relatedness" — useful for retrieval/ordination, but it does not
  by itself give a *per-node, per-rank* coordinate; it's a pairwise-sample metric, not an
  embedding of the taxonomy.
- **PhILR — Silverman, Washburne, Mukherjee, David (2017), "A phylogenetic transform
  enhances analysis of compositional microbiota data," *eLife* 6:e21887**
  ([paper](https://elifesciences.org/articles/21887); [Bioconductor vignette](https://bioconductor.org/packages/release/bioc/vignettes/philr/inst/doc/philr-intro.html)). This is the closest existing prior art to
  what the PI is describing. PhILR builds an isometric log-ratio (ILR) coordinate system
  **directly from the phylogenetic tree**: every **internal node** of the tree gets one
  orthonormal coordinate ("balance"), defined as the log-ratio of the geometric-mean
  abundance of the two child clades descending from that node. Consequences that map
  cleanly onto the PI's ask:
  - A query "is this **family** enriched" is literally reading off the balance at the
    internal node corresponding to that family's split from its sibling clades — a
    genus-level query reads a *different*, deeper coordinate. Ranks are different axes of
    the same space, not overwritten versions of one axis, by construction.
  - The transform is exact and invertible (orthonormal basis), so no information is
    thrown away the way `tax_glom` throws it away — the finer-grained coordinates
    (species/genus balances) still exist and can be inspected alongside the coarser
    (family/order) ones.
  - It was built to fix exactly the compositional-data pitfalls (spurious
    correlations from closure) that plague naive per-rank abundance comparisons, which
    is a second, independent reason to prefer it over relative-abundance collapse.
  - Caveat: PhILR balances are built for continuous relative-abundance data on a
    resolved bifurcating tree. Our data is discrete (per-paper directional calls,
    enriched/depleted/contested) on a **containment DAG with polytomies and unresolved
    placeholder ranks** (per this project's taxon-spelling and clade-label findings) — so
    PhILR's machinery would need to be adapted (Section 7), not applied verbatim.

## 5. Multi-resolution / hierarchical pooling in graph learning

A separate literature, in graph neural networks, builds embeddings at multiple
granularities of a graph simultaneously by learning *how* to coarsen it rather than fixing
the coarsening in advance:

- **DiffPool — Ying, You, Morris, Ren, Hamilton, Leskovec (2018), "Hierarchical Graph
  Representation Learning with Differentiable Pooling," *NeurIPS 2018*** ([arXiv:1806.08804](https://arxiv.org/abs/1806.08804)).
  Learns a soft, differentiable cluster assignment matrix at each GNN layer, mapping fine
  nodes to a smaller set of "super-nodes" that become the input to the next layer — so the
  network produces a **stack** of representations, one per coarsening level, all trained
  jointly and end-to-end, rather than one embedding at one fixed resolution. Gave 5–10%
  accuracy gains over flat/fixed pooling on graph classification benchmarks.
  Follow-ons in the same family (MinCutPool, gPool/SAGPool, MxPool — multiplex pooling)
  generalize the clustering mechanism but keep the core idea: learn the hierarchy jointly
  with the task instead of imposing a taxonomy-shaped one.
- **Relevance and a caution for this project**: DiffPool-style methods *learn* the
  hierarchy from data — useful if we wanted the model to discover which taxa cluster
  functionally regardless of formal taxonomy (e.g., grouping *Hungatella* with something
  other than *Lachnospiraceae* if the data supports it). But our taxonomy is not something
  to be learned — NCBI/GTDB containment is a known, correct structure, and the interesting
  question is exactly where *learned* functional similarity **diverges** from *given*
  taxonomic containment (this is arguably a more useful use of DiffPool-style pooling here:
  as a diagnostic — cluster taxa by embedding similarity and see where clusters cut across
  family/genus boundaries — rather than as the representation of record). Hyperbolic and
  box/order embeddings of the taxonomy itself (for representing containment losslessly in
  embedding space) are being covered separately in this research effort (see the
  hyperbolic- and box-embedding tracks) and are not re-derived here.

## 6. Aggregating evidence up a hierarchy: the gene-set-enrichment analogy

If several genera inside a family each show weak, individually-non-significant depletion,
is the family depleted? This is structurally identical to asking whether a gene set /
pathway is enriched from many individually-weak gene-level signals, and the microbiome
field has directly imported that machinery:

- **GSEA — Subramanian et al. (2005), *PNAS* 102(43):15545–15550**, and the modern
  fast implementation **fgsea** ([bioRxiv](https://www.biorxiv.org/content/10.1101/060012.full.pdf)). Walks a *ranked* list of all
  genes and asks whether members of a predefined set are concordantly enriched near the
  top or bottom, rather than thresholding first and counting — directly transferable
  pattern: rank all taxa by their (paper-count-weighted) direction score, then ask whether
  "children of family X" are concordantly enriched/depleted in that ranked list, without
  needing every child individually significant.
- **Microbe-set enrichment analysis (MSEA)** ([*Scientific Reports* 2020](https://www.nature.com/articles/s41598-020-78511-y)) — ports the
  GSEA machinery to microbe sets, including using external microbe–gene/pathway
  associations to borrow power from the human-genomics enrichment ecosystem.
- **CBEA — Nguyen, Hoen, Frost (2022), "CBEA: Competitive balances for taxonomic
  enrichment analysis," *PLOS Computational Biology* 18(5):e1010091** ([paper](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1010091)). The
  most directly applicable method found: for a predefined taxon set (e.g., "all genera in
  family X"), computes a **sample-level score** as the log-ratio of the geometric mean
  abundance of the in-set taxa vs. the geometric mean of everything else (a **competitive**
  null — is this set doing something different from the rest of the community, not just
  "is it non-zero"). An empirical null is built by column permutation rather than assumed
  parametrically, which matters given how irregular our per-paper evidence-count data
  would be. This is architecturally very close to "is family X depleted" phrased correctly:
  it tests the set as a set, while remaining fully aware of, and reportable at, the
  member level.
- **microBiomeGSM** ([*PMC* 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10703168/)) — a "grouping, scoring, modeling" framework for
  taxonomic-biomarker discovery, another instance of set-level scoring built for sparse
  compositional taxon data specifically (vs. importing bulk-RNA GSEA machinery unmodified).
- **Phylofactorization (Section 2) is also an up-aggregation method**, just phrased as
  factor extraction rather than enrichment testing — it explicitly answers "at what edge of
  the tree is the signal best summarized as one factor" and stops recursing exactly when a
  clade's children stop behaving as one unit, which is the up-aggregation analogue of
  treeclimbR's down-refinement.

**Caveat that applies to all of these**: GSEA-style competitive tests assume the *within-set*
members are behaving as one exchangeable unit under the null; the whole point of Section 3
is that this fails ~11% of the time for our taxa. So the right question for us is not
"aggregate," it's "test whether aggregation is licensed, and refuse it when it isn't" —
which is exactly what treeclimbR and phylofactorization do and naive `tax_glom` does not.

---

## 7. Concrete proposal for this project

**Goal:** answer "is taxon T enriched/depleted in disease D?" at a user-chosen rank
(species/genus/family/…), using evidence attached at *other* ranks, without ever silently
merging a contested split like *Lachnospiraceae*/*Hungatella* into one number.

**Design, in five parts:**

1. **Never pool onto a magnitude — keep the project's existing rule** (edge weight = evidence
   count, never effect size) and extend it: aggregation across ranks must also stay in
   count/direction space, not a blended score. This rules out simple weighted averaging of
   parent and child "confidence" the moment a real family/genus split like the motivating
   example exists.

2. **Report three numbers per queried node, always separately, never blended:**
   - *Direct evidence at T*: (papers-up, papers-down) reported **exactly** on T.
   - *Child-aggregate evidence*: a CBEA-style competitive score over T's immediate
     taxonomic children — "of T's children with any evidence, what fraction lean up vs.
     down, and is that distinguishable from the rest of the tree under a permutation
     null?" This produces a second, clearly-labeled number (e.g., "6 of 7 evidenced
     genera under this family lean depleted, p = …") that a user can see is *aggregate*,
     not *direct*, evidence for T.
   - *Homogeneity flag*: whether T passes a hierarchical-homogeneity test (below). If it
     fails, the UI answer must be "contested across children" with the split shown, full
     stop — mirroring the project's existing "contested edges are kept, never averaged"
     rule, just applied one level up the tree.

3. **License upward aggregation with a real test, not by default.** Run a Yekutieli-style
   top-down hierarchical test, or treeclimbR, over the taxonomy using the per-paper
   directional call as the leaf-level statistic (contested leaves count as their own
   split, not as a wash). A family node is only "resolved" as depleted/enriched using
   pooled child evidence if:
   - the homogeneity test does not reject "these children agree" at the node, **and**
   - the node clears the hierarchical FDR threshold for that level of the tree.
   *Lachnospiraceae* fails this by construction (89%→54% concordance boundary is
   *exactly* the discriminating statistic already measured for this corpus) and must
   render as contested; a family with 12 unanimous child genera and no contradicting
   direct evidence passes and can render one pooled number with the child-aggregate
   annotation still visible on click-through.

4. **Represent the taxonomy in embedding space as a PhILR-style balance tree, adapted for
   discrete evidence rather than continuous abundance.** Build one coordinate per internal
   node of the populated taxonomy (using the containment tree already resolved by
   `taxonomy.py`, not a phylogeny — ours is containment, and that's the right tree for
   this purpose since the question is inheritance of a category, not evolutionary
   relatedness). Each node's coordinate is the log-odds (or a Bayesian-smoothed log-ratio,
   to survive small counts and zero cells — cf. CBEA's empirical-null handling of sparsity)
   of "papers calling this clade enriched" vs. "papers calling it depleted," computed
   *only from evidence attached at or below that node*, exactly as PhILR's balance is
   computed only from the abundances below its node. This gives:
   - A rank query = read one coordinate. A species query and its containing genus query
     are *adjacent, independent* coordinates, never the same number — which structurally
     prevents the silent-merge failure mode by construction, the same guarantee PhILR
     gives for compositional abundance.
   - A natural place to attach the homogeneity flag from step 3: color/flag a node's
     coordinate as "resolved" vs. "contested" so the embedding space itself encodes where
     aggregation is licensed.
   - Compatibility with retrieval: nearby coordinates in this space are literally
     nearby clades in the same part of the taxonomy at a comparable rank, which is a
     usable multi-rank retrieval index without re-deriving one from scratch.

5. **When aggregation is explicitly wrong — build these as refusals, not edge cases:**
   - **Contested edges** (already a project rule) never enter the child-aggregate
     computation as a single averaged direction; both directions count separately in the
     permutation null.
   - **Nodes failing the homogeneity/hierarchical-FDR test** (step 3) — report split, not
     pooled, even if a naive average would look confident.
   - **Small-n nodes** where CBEA's permutation null can't be estimated stably (this
     project's per-disease sample sizes are already small enough — see the MCI/ASD
     n-limited findings — that this will bite often; state the minimum n up front rather
     than silently returning a wide-CI aggregate).
   - **Children whose only relationship is shared paper provenance, not independent
     replication** — i.e., if the "7 genera under this family" are actually reported by
     the same 2 papers, the competitive null is testing paper-level idiosyncrasy, not
     biology; this is the same non-independence trap the project already documented for
     the Disbiome/Peryton agreement figures, and it applies identically to within-family
     evidence pooling.
   - **Do not use this machinery to "fix" or override direct evidence at the queried rank
     itself.** If 15 papers directly call *Lachnospiraceae* depleted, that direct number
     is reported as-is regardless of what its genera show — the child-aggregate number is
     an annotation, never a correction.

This gives the PI's requested capability — the same query, answerable at species, genus,
or family, in one embedding space — while making the exact failure mode already measured
in this project (family says one thing, an interior genus says another) a first-class,
visible state rather than something an averaging step would quietly erase.
