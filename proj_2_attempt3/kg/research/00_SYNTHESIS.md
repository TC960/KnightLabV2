# Synthesis — embedding the KG for explainability

*2026-09-19. Eleven parallel literature reviews, ~3,800 lines, in `research/01`–`11`.
This file is the decision layer: what to do, what not to, and why.*

Sam's ask, verbatim: *"can we embed everything (all microbes and all disease) and
for each one we know how many studies support said microbe in X direction and how
many in Y"* plus *"we need a taxonomy mapping in the embedding space as well,
because we can ask questions like you are asking at species level or genus level
or family level"*.

That is **two** requests with very different risk profiles, and conflating them has
been costing us. Separating them is the main result here.

---

## 1. The blocker nobody would have found from the literature

**Our containment graph is a forest, not a tree.** Measured, not assumed:

| | |
|---|---|
| nodes in hierarchy | 747 |
| containment edges | 727 |
| **roots** | **20** |
| root ranks | 13 phylum, 2 order, 2 genus, 1 kingdom, 1 family, 1 no-rank |
| nodes with >1 parent | 0 |

Depth-within-fragment is therefore **not** taxonomic rank: a genus hanging
directly off a phylum root sits at depth 1, while another genus three levels into
a populated fragment sits at depth 3.

This was confirmed empirically. A Poincaré embedding trained on these 727 links
(`hyperbolic_taxonomy.py`, torch autograd, loss converging 2.46 → 1.91) gives
**Spearman(radius, true rank) = 0.0095, p = 0.81** — no relationship at all. The
geometry faithfully learned fragment depth, which is meaningless.

**Every hierarchy-geometry method in this review depends on a real tree**:
hyperbolic (01), box/order embeddings (03), hierarchical FDR (10). All are blocked
on the same one-line fix: rebuild containment from **full NCBI lineages** via
`taxonomy.py`, inserting the missing intermediate ancestors so every taxon hangs
off one root at its true depth.

**Do this before any embedding work.** It is cheap, `taxonomy.py` already reads
`nodes.dmp`, and it is a precondition for three separate recommendations.

## 2. Do not build a GNN (08)

The clearest negative result in the set, and it saves the most wasted effort.

The microbe-disease-association literature reports AUC 0.94–0.97 on HMDAD (292
microbes / 39 diseases / 450 edges — smaller than ours). Those numbers do not
survive scrutiny:

- **KATZHMDA, a pure topology index with zero learning, already scores 0.84–0.86**
  on the same data. The GNN "advance" is ~0.10 AUC over a method that does no
  learning.
- Every published method evaluates with random-negative 5-fold CV or LOOCV.
  **None uses a node-disjoint cold-start split.** Park & Marcotte (Nature Methods
  2012) showed pair-input random CV leaks through shared entities.
- "Unobserved" is not "negative" in a literature-derived graph — the field's own
  2021 survey concedes this.
- Li et al. (NeurIPS 2023, arXiv:2306.10453): under realistic hard-negative
  sampling, plain Katz and shortest-path **outrank GNNs** on graphs 600× ours.
- Menand & Seshadhri (PNAS 2024): low-dimensional dot-product embeddings provably
  cannot capture sparse ground truth the way AUC makes it appear they do — and
  most of these models score edges by exactly that mechanism.
- The high scorers (GATMDA etc.) inject external features — gene networks,
  disease-gene associations, GO terms — **that we do not have**.

**Bar for reconsidering**: beat node degree, Adamic-Adar on the containment graph,
personalised PageRank, and plain matrix factorisation, under a node-disjoint
cold-start split. Expect it won't.

## 3. The cheapest answer to Sam's rank question needs no embedding at all (07)

Expand the queried taxon to its descendants through the containment graph and
filter the chunk index on that set. `is-a` is a crisp relation we already hold at
100% precision — there is no reason to approximate it with geometry.

**Any embedding-space method must beat this baseline before it earns its place.**
State it that way when reporting, because "we used hyperbolic embeddings" sounds
more impressive than a set lookup and is not obviously better.

## 4. Where embedding genuinely helps, ranked

**(a) Box embeddings (03) — best fit for the rank query.** A box containing a box
*is* the is-a relation, so containment is native rather than approximated. Crucially
it **decouples hierarchy from association sign**, so *Lachnospiraceae* depleted and
*Hungatella* enriched can coexist without fighting the containment loss — the
exact pathology that makes rank-collapsing wrong for us. Precedent exists at our
scale: BoxTaxo (WWW'23) on 209 nodes/209 edges; a ChEBI box-embedding paper
(J. Cheminformatics 2025) at 854 classes, dim 16 — we have 883 taxa. Library:
`iesl/box-embeddings`. **Blocked on §1.**

**(b) Hyperbolic (01) — right method, wrong input today.** Nickel & Kiela's own
small-data showcase is the WordNet mammals subtree at 1,180 nodes / 6,540 edges,
MAP 0.927 at d=5 — our regime exactly. Depth falls out of radius by construction.
**Blocked on §1**; already tried and failed for that reason.

**(c) Retrofitting (07) — cheapest real embedding method.** Faruqui et al. 2015
post-processes existing vectors against a graph in closed form: no retraining, no
GPU. Apply to the 883 taxon-name embeddings, *not* the 20,905-chunk index. Evaluate
by held-out-pair AUC: does retrofitted cosine separate parent-child / sibling /
unrelated better than raw MiniLM? Note SapBERT and BioLORD train on **synonymy**,
not **hierarchy** — they would merge Firmicutes/Bacillota but would not pull
Blautia toward Lachnospiraceae.

**(d) RotatE via PyKEEN (02) — if we want the association half embedded.**
DistMult is structurally symmetric and therefore wrong for containment. Use
PyKEEN's per-triple loss weighting (`RelationLossWeighter`, v1.11) with
weight = log1p(paper count), which preserves contestedness while encoding evidence
strength. **Do not adopt UKGE**: it models one relation with graded confidence,
whereas we have two separately-true relations — collapsing them is precisely the
averaging this project already rejected.

## 5. A correction to something I reported as a null (05)

I reported the 34-concept contrast experiment as a null result. **It was not a
valid test.**

Real TCAV (Kim et al. 2018) requires a *trained and validated* concept activation
vector from ~50+ diverse examples per concept, ≥10 retrainings against random
negatives, and a t-test on the resulting directional derivatives. We used **one
hand-written probe sentence per concept with no separability check**. That is not
TCAV; it is cosine similarity to a sentence.

So the null is **part method failure, part genuine power failure** — not a correct
negative. The power ceiling is real and separate (`ceiling_direction_probe.py`:
only 1 of 226 edges could ever reach q<0.05 alone). But the concept test itself
deserves one proper attempt before we conclude concepts don't matter, and
`variable_sweep.py` already has the ≥50-example retrieval sets needed to build
real CAVs.

## 6. Direction-only is the binding constraint, and it is now quantified (06, 11)

- **Vote counting — which is exactly our edge weight — has power that *decreases*
  as studies accumulate** (Hedges & Olkin 1980, Psych Bull 88(2):359–369). The
  opposite of every other meta-analytic method.
- A validation study calibrates it: 2 supporting studies → ~50% confirmation (a
  coin flip), 3 → ~67–70%, 4+ → ~90%. **Most of our 43 dual-sided edges sit in the
  near-chance tier.**
- **Heterogeneity statistics (Q, τ², I²) mechanically require per-study effect size
  and variance.** They cannot be computed from direction counts. This forecloses
  "just run a heterogeneity test" permanently, not temporarily.
- Meta-regression needs ~10 studies per moderator (Cochrane). Our edges have 2–4
  per side. **The 24-variable null was guaranteed by sample size before any
  analysis ran.**
- Duvallet et al. 2017 (Nat Commun 8:1784) — the field's closest comparator — use
  vote counting themselves and state they *"did not find consistent bacterial
  associations for conditions with fewer than four data sets"*.

**Verdict: no reanalysis of direction-only counts at this n yields a more
defensible per-edge confidence than what we already have.** Either extract effect
sizes, or stop attaching per-edge statistics and report contested edges as a
qualitative flag.

## 7. External base rates for our disagreement numbers (06, 09)

Our contested rate is 11.7% (extraction) / 15.8% (human gold). Context:

- **Gibbons et al. 2018 (PLOS Comp Biol)**: 681 of 1,021 OTUs (67%) differed
  "significantly" between two studies' **healthy-control cohorts alone** — zero
  disease involved. Discordance at our magnitude is fully explicable by batch
  effect with **no true biological disagreement required**.
- **Tierney et al. 2022 (PLOS Biology)**: ~1 in 3 taxa show sign inconsistency
  across 581 associations / 15 cohorts; >90% of T1D/T2D associations are not
  robust to confounder choice.
- Sinha et al. 2017 (MBQC): DNA extraction protocol is a measured, substantial
  variance source — supporting "kit is real but we are underpowered to see it"
  rather than "kit does not matter".

This is the strongest available framing for the contested edges: **the field's own
numbers say disagreement at our rate is expected**, and our null on study-design
moderators is consistent with published work rather than an embarrassment.

## 8. Prior art, and what is actually novel (04)

- **MINERVA** (Briefings in Bioinformatics 2025) is the closest method sibling: a
  fine-tuned Biomistral-7B over 129,719 papers → 66,400 edges, relation-extraction
  F1 0.884. It resolves disagreement by **majority vote then impact-factor-weighted
  averaging** — the precise opposite of our design. Its independent critique paper
  (Nov 2025) attacks that averaging on statistical grounds. **That is external
  corroboration for keeping contested edges separate.**
- Disbiome and Peryton both record direction and use NCBI taxonomy at all ranks,
  and **neither describes any disagreement-resolution mechanism**.
- MicroPhenoDB merges HMDAD + Disbiome, so it **cannot serve as an independent
  third validator**.
- Correction: **NJS16 is a metabolic-interaction network, not a disease-association
  database** — it does not belong in this comparison.
- **No published direction-agreement rate between two independent curated
  microbe-disease databases could be found.** Everyone else uses HMDAD/Disbiome as
  *training targets* for link prediction. Our 73.0%/72.5% figures, and especially
  the shared-paper vs disjoint-literature decomposition (87.5%/96.8% vs
  58.1%/52.6%), may be the first published use of two curated DBs to cross-validate
  an extractor's direction calls. **This is the most likely publishable angle.**

## 8b. The closest existing answer to Sam's exact question is PhILR (10)

**PhILR** (Silverman, Washburne, Mukherjee & David, eLife 2017) builds one
orthonormal ILR "balance" coordinate **per internal tree node**. Rank is then
simply *which coordinate you read* — ranks never overwrite one another, which is
precisely the property we need and precisely what `tax_glom` destroys.

It is the closest prior art to *"ask at species level or genus level or family
level"* found anywhere in this review. Caveat: it is built for continuous
abundance on a fully resolved phylogeny, and we have discrete per-paper
directional calls on a containment graph with polytomies. Adaptation is real work,
not a library call.

**A countervailing null worth recording**: Sankaran & Holmes (Frontiers in
Microbiology 2020) found phylogeny-aware differential abundance gave **no power
gain** over flat testing in their benchmarks. So tree-aware methods are not a
guaranteed win, and any gain must be demonstrated rather than assumed.

Also relevant: `treeclimbR` (Huang, Soneson & Robinson, Genome Biology 2021) picks
the resolution **per branch** from the data instead of fixing one rank globally,
and phylofactorization (Washburne et al., PeerJ 2017) finds the tree *edge* that
explains variation rather than a rank. Both are closer to how this literature
actually behaves than "collapse to genus".

### The proposal worth adopting

Report **three separate numbers per node**, never one:

1. **direct evidence** — papers naming this exact taxon
2. **child-aggregate evidence** — CBEA-style competitive log-ratio test of the
   taxon's descendants against their complement, with a permutation null
   (Nguyen, Hoen & Frost, PLOS Comp Biol 2022)
3. **a homogeneity flag** — do the descendants actually agree?

And an explicit **refusal list**: do not aggregate when the edge is contested,
when a hierarchical homogeneity test fails, when n is small, or when the children
share provenance — and **never let aggregate evidence override direct evidence**.

*Lachnospiraceae* / *Hungatella* is the worked example throughout: it must render
as **contested**, not averaged.

### One number we have that the literature does not

Our measured within-paper concordance — **related taxa agree on direction 89% of
the time within a single paper versus 54% for unrelated taxa** — is more precise
than anything the review could find externally. That is a small, citable methods
result in its own right.

## 9. Representing contested edges properly (09)

Model each contested pair as a **Quantitative Bipolar Argumentation Framework**:
two poles (ENRICHED / DEPLETED), each paper an attack/support edge carrying
provenance plus a SemMedDB-style hedge tag, a gradual-semantics strength score
(uniform weights reproduce today's counts, extensible to covariate weighting), and
Shapley-style attribution for "why this side is stronger" — while keeping the full
bipartite structure queryable so nothing collapses to a scalar.

This is a *representation* upgrade, not a statistical one. It will not fix §6.

---

## Recommended order

| # | action | cost | blocked by |
|---|---|---|---|
| 1 | **Rebuild containment from full NCBI lineages** | hours | — |
| 2 | **Symbolic descendant-expansion rank query** — the baseline everything must beat | hours | 1 |
| 3 | **Re-run the Poincaré test** on the repaired tree — falsifiable in minutes | minutes | 1 |
| 4 | **Box embeddings** on the repaired tree | 1–2 days | 1 |
| 5 | **Proper TCAV** with real CAVs from existing retrieval sets | 1 day | — |
| 6 | **AnyBURL rule mining** over graph + paper + study-attribute nodes — human-readable Horn rules, no scale floor | 1–2 days | — |
| 7 | Effect-size extraction pilot on 20 papers — decides whether §6 is escapable | 2 days | — |
| 8 | Three-number-per-node reporting (direct / child-aggregate / homogeneity) with the refusal list | 2 days | 1 |
| — | ~~GNN~~ | — | **do not** |

Items 5 and 6 need no tree repair and are the only two that directly attack
*explainability* rather than representation. Item 6 is the one whose output a
biologist can read without an intermediary.

## Standing caution

Three of these reviews independently flagged that **published numbers in this
subfield are inflated by evaluation choices** — leaky CV in MDA-GNN work,
HPO-budget dominance in KGE benchmarks, attention-as-explanation. Whatever we
build, the comparison that matters is against a trivial baseline under a split
that does not leak, and the number to publish is the one with a confidence
interval, not a point estimate.
