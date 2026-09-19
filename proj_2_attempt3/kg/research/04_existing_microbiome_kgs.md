# Prior art: existing microbe–disease knowledge graphs and databases

Research question: what microbe-disease resources already exist, how do they handle
representation/taxonomy/disagreement, and what benchmark numbers exist to compare our
~73%/72.5% Disbiome/Peryton direction-agreement against? Compiled 2026-09-18 from
published papers (web search + fetch of primary sources). All numbers below are the
counts reported in the cited paper/version — most of these databases have grown since
their founding publication, so "size" figures are dated to the citation given.

---

## 1. Disbiome

- **Paper**: Janssens et al., "Disbiome database: linking the microbiome to disease,"
  *BMC Microbiology* 2018. [PubMed](https://pubmed.ncbi.nlm.nih.gov/29866037/) |
  [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5987391/)
- **Size / construction**: Manually curated. ~20,000 PubMed hits (2009–2018) screened
  down to ~500 publications at launch; over 800 organisms and 190+ diseases in the
  founding paper. (By the time other papers cite the live database circa 2021 it had
  grown to ~8,700 associations / 1,622 microbes / 374 diseases — e.g. the integration
  study in item 12 below.) Each source publication is scored against a 16-item
  reporting-quality questionnaire (participant demographics, analysis specifics, study
  design, methodology). Updated manually every ~3 months.
- **Direction**: **Yes.** Each entry records a qualitative outcome — "elevated" or
  "reduced" — for the microbe in disease vs. control.
- **Taxonomic rank**: Organisms classified via **both NCBI and SILVA** taxonomy,
  linked out to each; NCBI resolves to species level, SILVA is used for cross-checking.
  All ranks are kept (not normalized to genus).
- **Disagreement handling**: **Not addressed.** The paper does not describe any
  resolution, voting, or averaging step for conflicting studies — implicitly, Disbiome
  stores one row per (publication, microbe, disease, direction) observation and leaves
  reconciliation to the user.

## 2. Peryton

- **Paper**: Skoufos et al., "Peryton: a manual collection of experimentally supported
  microbe-disease associations," *Nucleic Acids Research* (Database issue) 2021.
  [Oxford Academic](https://academic.oup.com/nar/article/49/D1/D1328/5932864) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7779029/)
- **Size / construction**: Manually curated from **primary research articles only**
  (reviews excluded). ~2,000 PubMed hits refined by abstract/results screening to ~350,
  then 314 papers used. **7,977 associations, 1,396 microorganisms, 43 diseases**
  (23 cancer types, 10 GI disorders, 7 cardiovascular, 3 neurodegenerative). Minimum
  inclusion bar: valid NCBI taxid, reported statistical significance, and documented
  experimental design — notably close to our own extraction gate (significance
  required, main text implied).
- **Direction**: **Yes**, records increased/decreased abundance vs. comparison group.
  Also uniquely includes non-healthy-control comparisons (cancer grade, benign vs.
  malignant, symptomatic vs. asymptomatic) that Disbiome and gutMDisorder don't cover.
- **Taxonomic rank**: All ranks kept, standardized to **NCBI taxid**; genus is most
  common (46%) but species-or-below make up 21.5% of entries.
- **Disagreement handling**: **Not addressed** in the paper — same gap as Disbiome.

## 3. GMrepo (v1 2020, v2 2022, v3 2025)

- **Papers**: Wu et al. 2020 *NAR* [PubMed](https://pubmed.ncbi.nlm.nih.gov/31504765/);
  v2 2022 [Oxford Academic](https://academic.oup.com/nar/article/50/D1/D777/6426060); v3 2025
  [Oxford Academic](https://academic.oup.com/nar/article/54/D1/D734/8340991).
- **What it actually is**: Not a literature-curated association table. It's a
  **repository of reprocessed public metagenomic runs/samples** (v3: 890 projects,
  ~119,000 runs) with consistently-called taxonomic/functional profiles and manually
  curated sample metadata (disease phenotype, age, sex, country, BMI, antibiotic use).
  Disease coverage grew from 133 (v2) to 302 (v3) phenotypes.
- **Direction**: Not pre-computed as literature claims. Users run **on-the-fly
  differential-abundance / biomarker analysis** across selected case/control cohorts
  from the raw data; direction is whatever that live computation returns for the
  cohort the user picks — so "direction" is per-query, not a stored literature-derived
  fact.
- **Disagreement handling**: N/A in the same sense — GMrepo exposes cross-dataset
  comparison tooling explicitly so users can see cohort-to-cohort variance directly
  in the abundance data, but there's no single directional claim to reconcile.
- **Relevance to us**: A structurally different kind of resource (raw-data repository)
  from ours and from Disbiome/Peryton (literature-claim tables) — worth noting as a
  different tier of "microbiome database," not a competitor on direction agreement.

## 4. gutMDisorder (v1 2020, v2.0 2023)

- **Papers**: Cheng et al. 2020 *NAR* [Oxford Academic](https://academic.oup.com/nar/article/48/D1/D554/5580916);
  v2.0 2023 [Oxford Academic](https://academic.oup.com/nar/article/51/D1/D717/6754909).
- **Size / construction**: Manually curated. Human: **2,263 associations between 579
  gut microbes and 123 disorders or 77 interventions**; separate mouse data: 930
  associations / 273 microbes / 33 disorders / 151 interventions.
- **Direction**: Each entry has a free-text "brief description of the alteration"
  (i.e., increase/decrease is captured, but as descriptive text rather than a
  normalized categorical field per the source paper's own wording).
- **Taxonomic rank**: **All ranks kept** — phylum through species — with genus most
  common (48.19% of human microbes).
- **Disagreement handling**: **Not specified** in the paper.

## 5. MASI (v1 2021, v2.0 2025)

- **Papers**: Zhao et al. 2021 *NAR* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7779062/);
  MASI2.0 2025 [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022283625006606).
- **What it actually is**: Primarily a **microbiota–active-substance interaction**
  database (drugs, diet, herbs, probiotics, environmental chemicals), not a
  microbe-disease database per se. v1 does include a disease side-table: **784
  microbiota-disease associations, 806 microbiota species, 56 diseases**. v2.0 scaled
  substance interactions to 44,643 literature-curated entries plus 166,766
  microbiota-drug and 205,505 microbiota-food interactions derived computationally
  from genome-scale metabolic models (not literature-curated).
- **Direction / disagreement**: Not the focus of either paper; disease associations
  are a minor annex rather than the core schema, so neither is well documented.

## 6. "NJS16" — correction

Searched exactly as given; **NJS16 is not a microbe-disease association database.**
It is a literature-curated **metabolic interaction network** (Shoaie et al., *Sci
Rep* / related work) of ~567–570 gut bacterial/archaeal species, 3 human cell types,
and metabolites, connected by ~4,400–4,483 small-molecule transport/degradation
events — built for community metabolic modeling, not disease association mining. It
shows up in the microbe-disease-prediction literature only as a metabolite-interaction
feature source for some GNN methods, not as a disease-association benchmark. If the
intended target was a disease-association benchmark, the two standard ones in that
literature are **HMDAD** and **Disbiome** (below) — likely what "NJS16" was meant to
reference, or possibly a mis-transcription of a first-author-year label from one of
the matrix-completion papers we didn't independently confirm.

## 7. HMDAD

- **Paper**: Ma et al., "An analysis of human microbe-disease associations,"
  *Briefings in Bioinformatics* 2017 (database built 2016).
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/26883326/) | [site](http://www.cuilab.cn/hmdad)
- **Size / construction**: First-ever resource of its kind. Curated (described in
  different secondary sources as manual curation / text mining) from **61
  publications**, all pre-July-2014. **483 disease–microbe entries** (some sources
  say 450), **39 diseases, 292 microbes**.
- **Direction**: **Yes** — records increase/decrease, plus the body site sequenced.
- **Taxonomic rank**: Species/genus mixed, not normalized; no taxid resolution
  described.
- **Disagreement handling**: Not addressed.
- **Status**: Frozen since ~2014 literature; not updated. Despite this it remains
  the single most-used **benchmark** in the microbe-disease-*prediction* (not
  extraction) literature — nearly every graph-embedding/matrix-completion paper in
  our search (GCATCMDA, HGNNTMDA, MAGMDA, DuGEL, adversarial-autoencoder-GNN, etc.,
  2023–2025) validates against HMDAD and/or Disbiome via 5-fold cross-validation and
  AUC, not against ground truth in the literature. That is a fundamentally different
  evaluation question from ours (link-prediction AUC on a static, frozen, unaudited
  table vs. our direction-agreement against two *independent* curated sources).

## 8. Amadis

- **Paper**: Yang et al., "Amadis: A Comprehensive Database for Association Between
  Microbiota and Disease," *Frontiers in Physiology* 2021.
  [Frontiers](https://www.frontiersin.org/articles/10.3389/fphys.2021.697059/full) |
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/34335304/)
- **Size / construction**: Manually curated from **>1,000 articles**, supplemented by
  text-mining-assisted PubMed abstract retrieval for candidate new associations.
  Reported scale: **~20,167 associations, 221 diseases, 774 gut microbes** (secondary
  source lists "17 species" alongside this, which reads like a mis-parsed field —
  flagged as unverified rather than repeated as fact).
- **Direction / taxonomy / disagreement**: Not established from available abstracts;
  would need the full-text PDF to confirm. Its headline feature is a **dynamic
  network-construction tool** (build a custom microbe-disease subnetwork from
  user-selected modules) rather than a fixed direction/rank schema.

## 9. MicroPhenoDB

- **Paper**: Yao et al., "MicroPhenoDB Associates Metagenomic Data with Pathogenic
  Microbes, Microbial Core Genes, and Human Disease Phenotypes," *Genomics,
  Proteomics & Bioinformatics* 2020. [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8377004/)
- **Size / construction**: Built by **integrating existing resources** — IDSA
  clinical guidelines, NCI Thesaurus (OBO edition), plus **HMDAD and Disbiome
  themselves** — rather than fresh literature curation. **5,677 non-redundant
  associations, 1,781 microbes, 542 disease phenotypes**, across >22 body sites.
  Adds microbial core-gene links (via MetaPhlAn2) and a pathogenicity/virulence
  layer (antibiotic resistance genes) that none of the others has.
- **Direction / disagreement**: Since it's a downstream merge of HMDAD + Disbiome,
  it inherits (and presumably deduplicates) whatever direction/conflict information
  those two carry — the paper doesn't describe an independent reconciliation step.
  This makes MicroPhenoDB a *consumer* of Disbiome/HMDAD data quality, not an
  independent check on it — worth noting because it means it can't serve as a third
  independent validation source the way Peryton can.

## 10. mBodyMap

- **Paper**: Jin et al., "mBodyMap: a curated database for microbes across human
  body and their associations with health and diseases," *NAR* 2021.
  [Oxford Academic](https://academic.oup.com/nar/article/50/D1/D808/6413603)
- **What it actually is**: Like GMrepo, this is a **reprocessed-sample repository**
  (63,148 runs: 14,401 metagenomes + 48,747 amplicons) across 22 body sites and 56
  diseases, from 136 projects, with manually curated sample-level metadata — not a
  table of literature-asserted directional claims. Abundance/prevalence differences
  across body sites and disease states are computed dynamically from the reprocessed
  data, analogous to GMrepo.
- **Direction / disagreement**: Same caveat as GMrepo — computed per-query from raw
  abundance data, not a stored literature claim to reconcile.

## 11. LLM/text-mining microbiome KGs since 2023 — MINERVA

This is the closest prior art to our project in *method* (LLM-driven literature
extraction into a graph) and the most important comparison point.

- **Paper**: Langarica-Kim et al., "MINERVA — microbiome network research and
  visualization atlas: a scalable knowledge graph for mapping microbiome-disease
  associations," *Briefings in Bioinformatics* 26(5), September 2025.
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12454267/) |
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/40984703/) | live at
  [minervabio.org](https://minervabio.org/)
- **Size**: Mined **129,719 publications** (PubMed/PMC, 2014–2023) into **66,400
  direct microbe-disease associations** linking **2,941 microbes and 3,299 diseases**
  (entity tables report 3,429 microbe entities and 35,883 disease entities total
  after UMLS/ontology expansion — the 35,883 figure includes ontology nodes with no
  direct edge, not diseases actually observed in text). ~1,600x more edges than our
  graph, ~2 orders of magnitude more source papers — a genuinely different scale
  class (industrial text-mining vs. our curated closed-schema extraction).
- **Construction**: NER via SciBERT (diseases) + DistilBERT (microbes), then
  sentence-level relation extraction via a **fine-tuned Biomistral-7B-AUG** model,
  with a "multi-tiered validation requiring complete consistency."
- **Direction**: **Yes**, but semantically different from ours — MINERVA's
  positive/negative means "microbe **promotes**/**inhibits** the disease" (a causal/
  mechanistic framing), not "abundance is increased/decreased in the disease state"
  (our and Disbiome/Peryton's framing). Reported split: 60.2% positive, 36.3%
  negative, 3.6% undetermined.
- **Taxonomic rank**: Entities normalized to **UMLS Concept Unique Identifiers**
  (not NCBI taxid) with parent-child hierarchy links for cross-rank inference —
  a different backbone than the NCBI-taxid convention Disbiome/Peryton/our graph use.
- **Disagreement handling**: The one place MINERVA makes an **opposite design
  choice from ours**: it resolves conflicts by **(i) majority voting within a single
  paper, then (ii) impact-factor-weighted aggregation across papers**, collapsing
  every microbe-disease pair down to one net score. We explicitly do the reverse —
  keep all 220 contested edges as separate, un-averaged, provenance-tagged
  observations, on the finding that ~1 taxon in 3 flips sign between cohorts (repo
  root `CLAUDE.md`). MINERVA's own published critique (below) calls out exactly this
  choice as a statistical problem.
- **Validation numbers reported**: relation-extraction F1 = **0.884** (on
  full-confidence predictions, 81% coverage at that confidence level); a downstream
  **link-prediction** task (Node2vec embeddings over the graph) reaches only
  **F1 ≈ 0.71**.
- **Published critique**: Anonymous/independent commentary, "Methodological and
  statistical concerns in MINERVA microbiome-disease knowledge graph," *Briefings in
  Bioinformatics*, November 2025. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12780759/) |
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/41396813/). Flags five specific problems,
  each of which is a place our design differs and is worth citing defensively:
  1. **Open-access-only source bias** — MINERVA can only mine OA text, systematically
     under-representing paywalled venues/regions.
  2. **Sentence-level independence violation** — treating each extracted sentence as
     an independent Bernoulli trial ignores document-level negation/hedging context.
  3. **Impact-factor-as-quality-proxy** — weighting conflicting evidence by journal
     impact factor "conflates prestige with experimental validity" and imports
     publication bias.
  4. **Compositional-data violation** — treats microbial abundances as free-standing
     numbers when they're constrained to sum to a constant, without log-ratio
     correction.
  5. **No multiple-testing correction** — tens of thousands of edges scored without
     FDR control, inflating false-positive rate.
  The critique's bottom line: MINERVA's 71% link-prediction F1 means ~29%
  misclassification, "suitable only for low-stakes exploratory analysis, not
  clinical decision support."

## 12. Text-mining benchmark numbers for microbe-disease relation extraction (not just KG papers)

Two data points, forming a lineage on the **same task class** as our extractor
(classify a microbe-disease sentence/pair by direction), useful as an apples-to-
apples precision/recall reference distinct from the KG-level agreement numbers above:

- **Baseline (2021)**: Wang/co-authors, "Discovering microbe-disease associations
  from the literature using a hierarchical long short-term memory network and an
  ensemble parser model," *Scientific Reports* 2021.
  [Nature](https://www.nature.com/articles/s41598-021-83966-8). Reports F1 ≈ **0.74**
  on the classification task, cited as the field's benchmark to beat.
- **Follow-up (2023)**: "Leveraging pre-trained language models for mining
  microbiome-disease relationships," [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10357883/).
  Compares generative models (GPT-3, BioGPT, BioMedLM) against discriminative BERT
  variants (BioLinkBERT, PubMedBERT, BioMegatron) on a **corrected gold corpus of
  1,100 annotated sentences** (178 mislabeled "NA" instances re-annotated). Four-way
  label scheme: **positive** (microbe worsens/increases with disease), **negative**
  (treats/decreases), **relate** (associated, direction unknown), **NA** (unrelated)
  — notably this scheme has an explicit "direction unknown" bucket that neither
  Disbiome/Peryton/HMDAD nor our schema carries (we omit rather than tag-as-unknown).
  Best results: fine-tuned **BioLinkBERT F1 = 0.804 ± 0.036**, fine-tuned **GPT-3
  F1 = 0.810 ± 0.025** — both "> 0.8" vs. the 0.74 prior benchmark.
- **Read against our numbers**: these are sentence/pair-level classification F1 on
  held-out gold text, the same kind of quantity as our in-house-gold F1 (0.680,
  flagged in root `CLAUDE.md` as unreliable due to gold-standard defects) and our
  reading-fidelity figure (≥86.6% [81.7, 91.3], measured against papers' own
  sentences). Direct comparison is not apples-to-apples (different corpora, label
  schemes, and gold quality), but it puts published state-of-the-art for this task
  class at ~0.80–0.81 F1 on a corpus the authors themselves needed to re-annotate
  once already — i.e., published biomedical RE benchmarks in this exact niche also
  run into gold-standard reliability problems, which is a useful external
  corroboration of our own gold-standard audit finding.

## 13. Published database-vs-database agreement rates — the gap we're filling

Searched specifically for any prior published **direction-agreement rate** between
two independent curated microbe-disease databases (the exact quantity behind our
73.0%/72.5%). **Found none.** The one integration study that combines Disbiome and
Peryton — Zhu et al., "Prioritizing Disease-Related Microbes Based on the
Topological Properties of a Comprehensive Network," *Frontiers in Microbiology*
2021 [Frontiers](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.685549/full) —
takes the **union** of the two databases (7,810 Peryton + 7,378 Disbiome standardized
associations → 11,037 distinct associations after dedup, 2,106 microbes, 287
diseases) purely to build a bigger network for topological ranking. It does not
check whether the two sources **agree on direction** where they overlap — it only
removes exact duplicates. Every other multi-database paper we found (MicroPhenoDB,
GMRepo cross-dataset comparison, the GNN benchmark papers) treats the databases as
independent label sources to train/evaluate *predictors* on, not as instruments to
cross-validate *each other's* direction calls.

This means **our 73.0%/72.5% figures — and the caveat structure built around them
(the shared-paper vs. disjoint-literature split, the reading-fidelity number that
doesn't depend on either gold) — appear to be the first published instance of using
two independent curated microbiome databases as cross-validation for a literature
extractor's direction calls**, rather than as training/prediction targets. We
should say this plainly rather than hedge it: the field measures link-prediction
AUC against a single static table (HMDAD/Disbiome); it does not, as far as this
search found, measure whether two independently-curated tables agree with each
other, still less decompose that agreement by shared-source-paper status the way
`FINDINGS_independence.md` does.

## 14. Embeddings of these resources

Confirmed use case: **microbe-disease link prediction**, not disease-microbe
biomarker discovery or drug repurposing directly (though several papers frame link
prediction as a step toward the latter). This is a mature sub-field — dozens of
papers 2023–2025 — almost all trained/evaluated on **HMDAD and/or Disbiome** via
5-fold cross-validation and AUC:

- GCATCMDA (graph contrastive learning), HGNNTMDA (heterogeneous GNN + transformer),
  MAGMDA (multilayer-attention GCN, +1.87%/+1.44% AUC over prior best on HMDAD/
  Disbiome respectively), DuGEL (dual graph-embedded fusion, AUC 0.970/0.912 on
  HMDAD/Disbiome), an adversarial-regularized-autoencoder GNN, and others — all
  2023–2025, all framed as link-prediction/recommendation over the *static* HMDAD/
  Disbiome tables, none re-validating against independent literature.
- MINERVA (above) is the only one we found that runs Node2vec embeddings over an
  **LLM-extracted** (rather than manually curated) graph for link prediction,
  reaching F1 ≈ 0.71.
- We found **no published embedding work for drug-repurposing** specifically off a
  microbe-disease KG in this search (MASI's substance-interaction side is the
  closer analogue but wasn't embedding-based in what we found).
- We found **no published embedding/link-prediction work over Peryton** alone —
  it's used far less than HMDAD/Disbiome as a training target, plausibly because
  it's smaller and disease-scope-limited (43 diseases).

---

## Comparison table

| Resource | Year(s) | Size (assoc / taxa / diseases) | Built by | Direction? | Taxonomic rank handling | Disagreement handling |
|---|---|---|---|---|---|---|
| **Disbiome** | 2018– | ~8,700 / 1,622 / 374 (2021 snapshot) | Manual curation, ~500 papers | Yes (elevated/reduced) | NCBI + SILVA, all ranks | Not addressed — one row per observation |
| **Peryton** | 2021 | 7,977 / 1,396 / 43 | Manual curation, 314 primary papers | Yes (increased/decreased) | NCBI taxid, all ranks (46% genus) | Not addressed |
| **GMrepo** | 2020–2025 (v3) | 890 projects / ~119k runs / 302 diseases | Reprocessed public metagenomes + curated metadata | Computed per-query, not stored | N/A (raw taxonomic profiles) | N/A (live cohort comparison, not a claim table) |
| **gutMDisorder** | 2019/2023 | 2,263 / 579 / 123 (human) | Manual curation | Yes (free-text description) | All ranks (phylum–species) | Not specified |
| **MASI** | 2021/2025 | 784 disease-assoc / 806 / 56 (disease side, v1) | Manual curation (substance-interaction focus) | Not established (disease side is a minor annex) | Not established | Not established |
| **HMDAD** | 2016/2017 | 483 (or 450) / 292 / 39 | Curated/text-mined, 61 papers, frozen at 2014 | Yes (increase/decrease) | Not normalized | Not addressed |
| **Amadis** | 2021 | ~20,167 / 774 / 221 | Manual + text-mining-assisted | Unconfirmed | Unconfirmed | Unconfirmed |
| **MicroPhenoDB** | 2020 | 5,677 / 1,781 / 542 | Merge of HMDAD+Disbiome+IDSA+NCI Thesaurus | Inherited from sources | Inherited | Inherited (no independent step) |
| **mBodyMap** | 2021 | 63,148 runs / 22 sites / 56 diseases | Reprocessed metagenomes + curated metadata | Computed per-query | N/A | N/A |
| **MINERVA** | 2025 | 66,400 / 2,941 / 3,299 | Fine-tuned Biomistral-7B-AUG over 129,719 papers | Yes, but causal (promotes/inhibits), not abundance-direction | UMLS CUIs + hierarchy (not NCBI taxid) | **Collapses**: majority vote in-paper, then impact-factor-weighted average across papers |
| **Our KG** | 2026 | 2,008 / 883 / 40 | LLM (Qwopus3.5-27B) closed-schema extraction, 271 papers | Yes (enriched/depleted, abundance) | NCBI taxid, all ranks, 76% resolved, documented refusal log | **Keeps all 220 contested edges separate**, provenance-tagged per paper, never averaged |

---

## What is genuinely novel about our graph vs. what already exists

1. **We are the only resource, as far as this search found, that reports a
   direction-agreement rate between two independent curated databases and
   decomposes it by whether the compared papers are shared or disjoint between
   corpora.** Existing multi-database work (item 13) only takes unions/dedups
   across Disbiome and Peryton; none of it checks whether they agree, and MINERVA
   — the one other LLM-built KG we found — resolves disagreement by collapsing it
   away (majority vote + impact-factor weighting) rather than measuring it.
2. **We keep contested edges as first-class, provenance-tagged, un-averaged
   observations** (220 of them) instead of collapsing to a single net direction.
   This is the opposite of every other resource surveyed that addresses the
   question at all (only MINERVA does, and it averages). Our own corpus-discordance
   analysis (root `CLAUDE.md` / `FINDINGS_paper_discordance.md`) shows this matters:
   ~1 taxon in 3 flips sign between cohorts, and ~85% of that variance is edge
   structure rather than study-design confounds we can name — i.e., averaging it
   away would be discarding signal, not noise, which MINERVA's own published
   critique independently arrives at from a purely statistical argument (violating
   compositionality and independent-Bernoulli assumptions).
3. **We publish self-measured recall and fidelity audits against the papers' own
   source text**, independent of any gold standard or external database: reading
   fidelity ≥86.6% [81.7, 91.3], paper-level recall 96.1–99.6%, edge-level recall
   upper bound ~98.1% [96.4, 99.5]. None of the ten curated databases surveyed
   publishes a recall or fidelity audit against their own source papers at all
   (their QA, where described — e.g. Disbiome's 16-item questionnaire — scores the
   *source study's* reporting quality, not the *curator's* extraction accuracy).
   MINERVA reports extraction F1 (0.884) and link-prediction F1 (0.71) but not a
   recall-against-source-sentences audit, and its own critique paper argues its
   validation methodology (sentence-independence, no FDR correction) is unsound —
   an explicit gap we address by gating the extractor on stated significance and
   auditing both directions of error (misses and fidelity) against source text.
4. **Scale and scope**: we are two to three orders of magnitude smaller than
   MINERVA (2,008 vs. 66,400 edges; 271 vs. 129,719 papers) and comparable in kind
   to the classic manually-curated databases (Disbiome, Peryton, HMDAD, gutMDisorder)
   in size, but built by an LLM pipeline rather than manual reading — i.e., we sit
   in a scale/method niche between "hand-curated, small, high-trust" (Disbiome/
   Peryton, ~2018–2021) and "LLM-scaled, huge, lower-trust-per-edge" (MINERVA,
   2025). We are not attempting to replace either; the honest positioning is
   **a validated, audited, mid-scale extraction with an explicit uncertainty
   budget on every headline number**, which none of the ten curated databases
   states for itself and which MINERVA's own published critique argues it lacks.
5. **NCBI-taxid discipline with a documented refusal log.** Several of the curated
   databases (Disbiome, Peryton, gutMDisorder) say they use NCBI taxonomy, but none
   reports a resolution *rate* or documents specific cases they declined to fold
   (our `taxon_typos.py` / 13 recorded refusals, `species_synonyms.py` decisions).
   MINERVA uses UMLS CUIs, not NCBI taxids, so it isn't directly joinable to
   Disbiome/Peryton/our graph on taxonomy at all without a remapping step.

**One caution for our own write-ups going forward, drawn from this survey**: our
73.0%/72.5% agreement figures are not directly comparable to any other published
number in this space — there is no equivalent metric published elsewhere to cite as
a reference point, good or bad. The nearest comparable numbers (HMDAD/Disbiome
link-prediction AUC ~0.91–0.97, MINERVA relation-extraction F1 0.884, sentence-level
RE F1 ~0.80–0.81) are all **different quantities** (prediction accuracy vs.
extraction accuracy vs. cross-database concordance) and should not be quoted
side-by-side with ours as if they measured the same thing — the same discipline the
root `CLAUDE.md` already applies internally between our own fidelity/recall/
agreement numbers should extend to any external comparison too.
