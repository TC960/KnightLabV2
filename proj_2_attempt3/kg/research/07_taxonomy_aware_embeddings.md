# Injecting taxonomy into a text embedding space

**Question:** we have 20,905 chunks from 303 papers embedded with `all-MiniLM-L6-v2`
(384-dim, frozen, generic sentence encoder) and, separately, a KG with 883 taxa
resolved to NCBI taxids and 727 taxonomic containment edges
(`proj_2_attempt3/kg/taxonomy.py`, `graph.json`). The encoder has no idea `Blautia`
sits inside `Lachnospiraceae`, or that `Firmicutes` and `Bacillota` are the same
phylum. The PI wants "a taxonomy mapping in the embedding space... so we can ask
questions at species level or genus level." How do you get an external
taxonomy into a text embedding space, and what's the cheapest way to do it
for ~900 entities without a GPU?

Status: draft, filling in incrementally.

---

## 1. Knowledge-enhanced language models (retraining or adapter-training required)

These all fuse a KG into a transformer *during* pretraining/fine-tuning. Relevant
as background, but every one of them needs a training run — not applicable
as a bolt-on to a frozen MiniLM index.

**General-domain (entity-centric, not built for synonymy/hierarchy specifically):**
- **ERNIE (Zhang et al. 2019, ACL)** — aligns Wikipedia text spans to Wikidata
  entities via TAGME, masks the alignment, and trains BERT to predict the KG
  entity from context. Needs entity-linked text + a KG embedding table (TransE)
  pretrained separately. [arXiv](https://arxiv.org/abs/1905.07129)
- **KnowBERT (Peters et al. 2019, EMNLP)** — adds a "Knowledge Attention and
  Recontextualization" layer that pulls candidate entity embeddings from a KB
  (Wikipedia + WordNet) and re-contextualizes token representations with them.
  Full architecture change, joint retraining of BERT layers.
  [arXiv](https://arxiv.org/abs/1909.04164)
- **K-Adapter (Wang et al. 2021, ACL Findings)** — freezes the base LM and
  trains small adapter modules per knowledge source (factual KG, linguistic),
  injected between transformer layers. Cheaper than full retraining (adapters
  only, ~a few % of params) but still a gradient-based training run per
  knowledge source, needs a fact/relation dataset, not just a taxonomy tree.
  [arXiv](https://arxiv.org/abs/2002.01808)
- **KEPLER (Wang et al. 2021, TACL)** — literally the same PLM encodes both
  entity description text and does knowledge-embedding (TransE-style) link
  prediction on KG triples, jointly optimizing KE loss + MLM loss. Elegant
  ("no extra inference cost," the entity embedding *is* the text embedding of
  its description) but needs entity **descriptions** (Wikidata has these;
  NCBI taxonomy nodes mostly don't) and a full pretraining run.
  [arXiv](https://arxiv.org/abs/1911.06136)
- **CoLAKE (Sun et al. 2020, COLING) / JAKET (Yu et al. 2022, AAAI)** — go
  further and build one heterogeneous graph out of the sentence's word graph
  fused with KG neighbors, encoded with a single (R)GCN-augmented transformer.
  JAKET's selling point is adapting to a KG unseen at training time by
  jointly training a "knowledge module" and "language module" that bootstrap
  each other — closer to what we'd want (883 taxa is a small, unseen-at-Wikidata
  KG) but this is a research-grade joint-training setup, not a weekend job.
  [CoLAKE arXiv](https://arxiv.org/abs/2010.00309),
  [JAKET arXiv](https://arxiv.org/abs/2010.00796)

**Biomedical, synonym/hierarchy-focused — the directly relevant family:**
- **SapBERT (Liu et al. 2021, NAACL)** — self-alignment pretraining: takes
  UMLS's 4M+ concept names (many synonymous strings per CUI) and does metric
  learning (multi-similarity loss) so that different surface forms of the
  *same concept* land on top of each other in embedding space. Base model
  `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`; weights on
  HF as [`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext).
  Usage is trivial: feed a string, take the `[CLS]` embedding. This is
  **exactly our Firmicutes/Bacillota problem** — SapBERT is a synonym-collapsing
  embedding, not a hierarchy-encoding one. It would merge synonyms but does
  **not**, out of the box, pull `Blautia` toward `Lachnospiraceae` (parent-child
  is not the relation it was trained on — UMLS's SapBERT training pairs are
  same-CUI synonyms, not is-a pairs, though UMLS *does* have some hierarchical
  relation types (`isa`, `RB`/broader) it could be retrained on).
  [paper](https://aclanthology.org/2021.naacl-main.334/),
  [GitHub](https://github.com/cambridgeltl/sapbert)
- **CODER (Yuan et al. 2022, JBI)** — same idea as SapBERT but explicitly
  trains on UMLS *relation triples* too (not just synonym pairs), via
  contrastive learning where the positive pairs come from both synonymy and
  relational co-occurrence in the graph. Closer to what we want because it's
  not synonym-only, but still needs a UMLS-scale relational KG to contrast
  against; nobody has done this for NCBI Taxonomy/GTDB.
  [arXiv](https://arxiv.org/abs/2011.02947)
- **BioLORD-2023 (Remy et al. 2024, JAMIA)** — the most structurally-aware of
  this family. Instead of contrasting name-pairs directly, it grounds each
  KG concept in an LLM-*generated definition/description* (from a multi-relational
  biomedical KG) and trains the encoder (base: `sentence-transformers/all-mpnet-base-v2`)
  via contrastive learning + self-distillation + weight averaging so the
  concept embeddings "match the hierarchical structure of the ontology" —
  their own framing is almost verbatim the PI's ask, just for clinical
  terminology (SNOMED-CT/UMLS) rather than NCBI Taxonomy. HF:
  [`FremyCompany/BioLORD-2023`](https://huggingface.co/FremyCompany/BioLORD-2023).
  [paper](https://academic.oup.com/jamia/article/31/9/1844/7614965)

**Retraining cost for all of the above:** full pretraining runs need GPU-days
and a source KG with millions of triples/synonym pairs; even the "cheapest"
of these (K-Adapter's frozen-base adapters) still needs a gradient-based
training loop and a purpose-built dataset. None of them was built for a
900-node, 727-edge taxonomy — they're calibrated for UMLS/Wikidata scale.
**Not recommended as a first move for this project.**

---

## 2. Retrofitting — post-hoc, no retraining, this is the interesting option

Faruqui et al. 2015 (NAACL), **"Retrofitting Word Vectors to Semantic
Lexicons"** ([paper](https://aclanthology.org/N15-1184/)): given (a) a
pretrained embedding matrix and (b) a graph over the same vocabulary from an
external resource (WordNet, PPDB, FrameNet), retrofit each vector to be close
to both its original position *and* its graph neighbors' (post-hoc) vectors.
It's a convex quadratic objective with a **closed-form iterative solution**
(Jacobi-style weighted averaging with graph neighbors, ~10 iterations) —
seconds on a laptop, no gradient descent, no GPU, and it touches only the
vectors for words that appear in the lexicon. This is the one method in this
whole list that is genuinely "take embeddings you already have and bend them
toward a graph you already have," which is our exact situation (883 taxon
strings already embedded by MiniLM as part of chunk text; 727 containment
edges already in `graph.json`).

Successors, roughly in order of sophistication:
- **Counter-fitting (Mrkšić et al. 2016, NAACL)** — adds *repel* terms
  (antonyms should be pushed apart) alongside the *attract* terms retrofitting
  uses, so it optimizes attract + repel jointly rather than just averaging
  toward neighbors. Relevant if we ever want "these two taxa are known to be
  mutually exclusive/distinct genera" as a negative constraint, not just
  "these are related." [paper via ACL Anthology]
- **Attract-Repel (Mrkšić et al. 2017, TACL)** — generalizes counter-fitting:
  optimizes how an update to one pair's vectors affects *all* other vectors
  (not myopically per-pair), and explicitly supports cross-lingual constraint
  sets. More faithful to the true joint objective, at somewhat higher
  implementation cost (still no GPU, still closed-ish form / few epochs of
  a light SGD).
- **Post-Specialisation (Vulić et al. 2018, NAACL)** — the retrofitting
  methods above only touch vectors present in the lexicon graph. Post-spec
  learns a global transformation function *from* specialized vectors *to*
  unspecialized ones, then applies it to out-of-lexicon words too — relevant
  for us because plenty of taxon *mentions* in text won't exactly string-match
  our 883 canonical names (misspellings, `taxon_typos.py` already handles some
  of this), so a learned generalization function could extend retrofitting's
  benefit to those variants. [GitHub](https://github.com/cambridgeltl/post-specialisation)

**Caveat for our case:** all of the above were designed for **static word
vectors** (word2vec/GloVe, one vector per token type), not for **sentence
encoders** like MiniLM where the taxon name only exists contextually inside a
384-dim chunk embedding. The direct match to our setup is: embed each of the
883 **taxon name strings** in isolation (not full chunks) with MiniLM to get
an 883×384 concept table, then retrofit *that small table* against the
727-edge containment graph. This is cheap (883 forward passes + a closed-form
graph smoothing step) and produces a "taxonomy mapping in embedding space" as
a side table, without touching the 20,905-chunk index or retraining the
encoder at all. See the proposed experiment at the end.

---

## 3. Joint text + graph embedding (train together from scratch)

Distinguished from §1 by training the KG and text jointly from the start
rather than injecting a pretrained KG into a pretrained LM:
- **KEPLER**, **JAKET**, **CoLAKE** (already covered above) are the standard
  citations; all need a training run.
- **Taxonomy Enrichment with Text and Graph Vector Representations**
  (Nikishina et al. 2022) — directly on-topic: the task is literally "given
  a taxonomy (WordNet/hypernym graph) and new terms with defining glosses,
  place the new term correctly in the hierarchy," combining a text encoder
  over the gloss with a graph encoder (Poincaré / node2vec-style) over the
  existing taxonomy, then learning a joint scorer for "is-a" placement.
  This is the closest published task-formulation to "given a taxon
  string + NCBI Taxonomy, embed it so its position reflects rank" — worth
  reading in full if we go beyond retrofitting.
  [arXiv](https://arxiv.org/abs/2201.08598)

Cost/data: needs a training loop (moderate — these are much smaller models
than SapBERT-scale, e.g. GCN + small text encoder) and a **decent-sized
taxonomy with glosses/definitions** for the text side to attach to. NCBI
Taxonomy nodes mostly lack glosses; GTDB/SILVA are pure label trees. Feasible
at our 883-node scale, but it's a real modeling project (days), not a
post-hoc fix — positioned above retrofitting on cost, below full
knowledge-enhanced LM pretraining.

*(Hyperbolic/Poincaré geometry for hierarchy embedding, and KG-embedding
methods like TransE/RotatE/box embeddings, are being covered in depth by the
other research threads in this session — r1-hyperbolic, r2-kge, r3-box — so
noted here only as the graph-encoder half of the Nikishina et al. approach
above, not re-derived.)*

---

## 4. Taxonomy-specific prior work: NCBI/GTDB/SILVA embeddings, and DNA LMs

**Symbolic-taxonomy → vector, no text involved:**
- Generic approach seen in several papers: **node2vec over the NCBI
  Taxonomy tree** (biased random walks over parent/child edges) to get one
  vector per taxon that preserves tree neighborhood structure. Simple,
  well-understood, no GPU (node2vec is a CPU random-walk + skip-gram job),
  and directly usable for our 883-node containment graph — but it encodes
  **only** the tree structure, with no connection at all to the text/chunk
  embedding space unless explicitly fused (e.g. via retrofitting or
  concatenation).
- **species2vec (Angelov, 2018, bioRxiv)** — orders species names by
  co-occurrence in GBIF biodiversity survey records (not literature text) and
  runs fastText skip-gram over the resulting "corpus," producing 300-dim
  species vectors from ecological co-occurrence rather than phylogeny.
  Different *relation* being captured (co-occurs-in-a-survey vs. is-a-child-of)
  from what we want. [bioRxiv](https://www.biorxiv.org/content/10.1101/461996v2.full),
  [code](https://github.com/boyanangelov/species2vec)
- **TEPI (Aakur et al. 2024, IEEE JBHI)** — represents whole *genomes* as
  pseudo-images and maps them into a taxonomy-aware embedding space (encodes
  compositional + phylogenetic relationships) for zero-shot species
  classification from genomic input, not text. Confirms the taxonomy-aware
  embedding-space idea is a live research pattern, but the modality is genome
  sequence → pseudo-image → embedding, unrelated to our text pipeline.
  [arXiv](https://arxiv.org/abs/2401.13219)

**DNA/genome language models — embed *sequences*, not taxon names, important
distinction for us:**
- **DNABERT / DNABERT-2** — BERT-style models pretrained on genomic k-mers;
  give a `[CLS]` embedding per DNA sequence. Used for taxonomic classification
  by training a classifier head on top (phylum-of-origin prediction), but the
  embedding itself represents *a sequence's content*, not *a taxon concept*.
  Two sequences from the same species get similar embeddings because they're
  biologically similar, not because the model "knows" the taxonomy label.
- **DNABERT-S (Zhou et al. 2024)** — adds a species-discriminative contrastive
  objective (Manifold Instance Mixup + curriculum contrastive learning)
  specifically so that sequences *cluster by species* in embedding space —
  the closest thing in this family to "genus/species relations geometrically
  encoded," but again for raw sequence embeddings, useless for embedding the
  string `"Blautia"` that appears in a sentence.
  [arXiv](https://arxiv.org/abs/2402.08777)
- **Nucleotide Transformer** — same category (sequence embedding models);
  not evaluated here in as much depth since the sequence-vs-taxon-name
  distinction already rules this whole family out for our use case (we have
  no genomic sequence data — our corpus is literature text about taxa, not
  taxa's genomes).

**Bottom line for §4:** none of the dedicated NCBI/GTDB/SILVA embedding
efforts operate in a *text* embedding space at all — they're either
pure-graph (node2vec-style), ecological co-occurrence (species2vec), or
genomic-sequence (TEPI, DNABERT family). They confirm the taxonomy side is
easy to embed on its own; the actual gap — fusing that with our *text*
retrieval index — is not something any of them solve for us.

---

## 5. Microbial taxa embeddings with geometrically-encoded genus/species structure

Two papers directly answer "has this been done for microbial taxa
specifically, from a language-model angle" — both operate on **abundance
tables** (which taxa co-occur across samples), not free text, but both show
the geometry comes out right:

- **"Learning a deep language model for microbiomes" (2024, PLOS Comp Bio,
  bioRxiv 2023)** — treats each microbiome sample's taxa list as a
  "sentence" (taxa = tokens, presence/relative-abundance = context) and
  trains a transformer (mBERT-style) self-supervised over large-scale
  unlabeled microbiome sample data. Result they highlight: t-SNE of the
  learned **contextualized taxa embeddings clusters by phylum**, better than
  plain GloVe baselines, and separately recovers known metabolic-pathway and
  IBD-association structure the model was never told about. This is direct
  evidence that genus/family/phylum structure **does** fall out of
  co-occurrence-based training without being explicitly supervised —
  but the training signal is sample-level co-occurrence across many
  microbiome studies' abundance tables, which we don't have (we have
  papers' *prose about* taxa, not raw abundance matrices).
  [PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011353),
  [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.07.17.549267v2.full.pdf)
- **DeepPhylo (Wang et al. 2024, Advanced Science)** — explicitly
  phylogeny-aware: builds amplicon (OTU/ASV) embeddings directly from the
  16S phylogenetic tree, encoding evolutionary *distances* (branch length),
  not just topology, so two OTUs' embedding distance reflects how recently
  they diverged. Beats prior methods (which only used tree *topology*, e.g.
  UniFrac-style) on downstream clustering/prediction tasks. Again,
  abundance-table input, not text — but it's the strongest existing evidence
  that **phylogenetic branch length, not just parent/child topology, can be
  embedded geometrically for microbial taxa**, which is a strictly richer
  signal than what our 727 containment edges currently carry (those are
  unweighted is-a edges).
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615782/)
- Related, not text-embedding either: **MIOSTONE** and similar
  "taxonomy-adaptive neural networks" build a neural net whose *architecture*
  literally mirrors the GTDB tree (one hidden unit sub-tree per taxonomic
  rank) to predict host traits from abundance profiles — taxonomy is baked
  into network topology rather than into a vector space per se. Relevant as
  a design pattern (rank-aware structure > post-hoc vector fix) but doesn't
  produce reusable taxon embeddings.

**Bottom line for §5:** yes, genus/species/phylum structure has been shown to
fall out geometrically for microbial taxa — but every published instance
uses **abundance/co-occurrence data across many samples**, which is a
different data modality from our corpus (paper *prose*, no sample-level OTU
tables). Nothing published fuses literature-text embeddings with microbial
phylogeny directly; that gap is exactly what the retrofitting proposal below
would fill cheaply, and what the Nikishina et al. (§3) joint text+graph
approach would fill properly if we ever want to invest the training time.

---

## Recommendation: cheapest method + minimal experiment

**Cheapest method that answers the PI's actual question** ("ask questions at
species level or genus level") is not, in the end, an embedding-space method
at all — it's the **symbolic baseline that must be beaten first**: taxonomic
containment is a crisp is-a relation we already have with 100% precision
(`taxonomy.py` + `graph.json`'s 727 edges), so a query at "genus level" can be
answered by looking up the queried taxon's NCBI lineage, expanding to all its
descendant taxids via the containment graph, and filtering/boosting the
20,905-chunk index for chunks tagged with any taxon in that descendant set.
**Zero embedding-space work, zero GPU, correctness bounded only by how good
our existing taxon-tagging of chunks already is.** Any embedding-space fix
below should be evaluated against this baseline, not assumed to beat it.

**Cheapest genuine embedding-space method — retrofitting (§2), scoped to a
small side table, not the chunk index:**

1. Embed the 883 canonical taxon-name strings (+ any recorded synonyms from
   `species_synonyms.py`/`taxon_typos.py`) with the *same* `all-MiniLM-L6-v2`
   model already used for the chunk index → an 883×384 "concept table."
   (Minutes on CPU; no GPU needed.)
2. Build the retrofitting graph directly from the 727 containment edges in
   `graph.json` (child ↔ parent) — this is the Faruqui et al. 2015 setup
   exactly: pretrained vectors + an external relational graph over the same
   vocabulary.
3. Run closed-form retrofitting (a few lines of numpy, weighted iterative
   averaging with graph neighbors, ~10 iterations) to get 883 "taxonomy-aware"
   concept vectors. Optionally add Attract-Repel-style repel constraints
   between sibling taxa known to be *distinct* (e.g. different genera under
   the same family) if plain attract-only retrofitting collapses siblings
   too aggressively.
4. **Evaluation (the actual minimal experiment):** for a held-out set of
   taxon pairs at known rank distances (parent-child, same-genus siblings,
   cross-phylum unrelated), check that cosine similarity in the *retrofitted*
   table separates these three classes better than the *raw* MiniLM
   embeddings of the same strings do (raw MiniLM has no reason to know
   `Blautia` and `Lachnospiraceae` are related — that's the whole premise).
   A simple AUC or rank-correlation of (retrofitted similarity) vs.
   (taxonomic distance in the containment graph) on held-out pairs is
   sufficient to know if it worked, before touching the 20,905-chunk index
   at all.
5. Only if step 4 succeeds: use the retrofitted concept vector for a
   query-time rank filter/boost over the existing chunk index (e.g., boost
   chunks whose tagged taxon's retrofitted vector is close to the queried
   rank's retrofitted vector) — this still requires **no retraining of the
   encoder and no re-embedding of the 20,905 chunks**, since it's an
   additive re-ranking signal computed from the small 883-row table.

This ordering (symbolic baseline → 883-row retrofit → evaluate → only then
touch the big index) is the cheapest path from "disconnected" to "taxonomy
mapped in embedding space," and every step is CPU-only, closed-form or
near-closed-form, and reuses data already sitting in `proj_2_attempt3/kg/`
(`taxonomy.py`'s resolved taxids, `graph.json`'s 727 containment edges,
`species_synonyms.py`/`taxon_typos.py`'s synonym tables) rather than requiring
any new annotation or GPU time.
