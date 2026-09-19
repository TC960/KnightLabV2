# CLAUDE.md — KnightLabV2 (repo root)

> Knight Lab microbiome NLP project. This root file gives the **big picture, lineage, and repo map**.
> The live working code has its own, more detailed `proj_2_attempt3/CLAUDE.md` — read that too when working there.

## What this project is

**The end goal is a knowledge graph** of microbe–disease relationships built from the microbiome
literature — nodes for microbial taxa and diseases, edges for "taxon X is **enriched ↑** / **depleted ↓**
in disease Y."

Extraction is done and **the graph is built**: see `proj_2_attempt3/kg/`, published at
<https://www.mohakprakash.com/KnightLabV2/>. **883 taxa** (76% resolved to NCBI
taxids), **40 diseases**, **2,008 association edges** plus
**727 taxonomic-containment links**, from **271 contributing
papers** of a screened 325-paper corpus. It agrees with two hand-curated databases at
**73.0%** (Disbiome) and **72.5%** (Peryton) on edge direction — but **do not quote those two
numbers as independent replication**; see the caveat below.

**Caveat on the agreement figures (2026-09-09).** Those curations are *not* independent of our
corpus. 43 of our 272 papers are also cited by Disbiome and 24 by Peryton, and because the shared
ones are the heavily-reported papers they back **half** the decisive pairs. Agreement splits hard
on that line: **87.5% / 96.8%** where the two sides read the same paper, **58.1% / 52.6%** where
the literature is disjoint. Within Parkinson's — the only disease with both buckets full — both
databases independently land on the same disjoint rate (**59.0%** and **59.6%**, vs 100% and 95.8%
shared, p=0.0001 each). So 73% is a blend of ~90% *reading fidelity* and ~55% *cross-literature
reproducibility* and measures neither. The good half is real and is the cleanest evidence for the
extractor that does not depend on the in-house gold: where a single-paper edge's one paper **is**
the curated source, agreement is **85–94%**. Counter-example, logged: in Multiple sclerosis the gap
is absent (72.7 vs 70.6, n=39) at an MDE that could have seen it. Full write-up and the calibration
that came with it: `proj_2_attempt3/kg/FINDINGS_independence.md`.

*Numbers current as of 2026-09-03; the earlier "712 taxa / 1,398 edges / 77.5% / 75.6%, from 250
papers" line described a graph three corpus revisions ago. Agreement fell because the corpus grew
and the question set changed, NOT because the graph got worse — five structural corrections since
have each moved agreement by less than this corpus can resolve (~0.013). See
`proj_2_attempt3/kg/SESSION_LOG.md`.*

**The gold standard was replaced, and the F1 caveat with it (2026-09-17).** The old reference was
incomplete — 162 of 250 papers had blank taxa columns — so the extractor's long-quoted "F1 0.680"
was agreement with a flawed reference, not accuracy. A new hand-curated gold
(`high_confidence - final_constrained_override.csv`, 334 DOIs, no LLM assistance) replaces it:
**F1 0.739** taxonomy-aware (P 0.721 / R 0.759), 0.733 char-ngram, over the 260 scoreable papers,
permutation p = 0.001 against a null mean of 0.148. **Nothing about the model changed** — the same
cached extractions scored 0.639 against the old reference. Two corrections are folded in: the taxon
matcher **double-counted** (two predictions could each claim the same gold taxon; every F1 on record,
`leaderboard.csv` included, was inflated — pre-fix 0.755/0.780), and **LCA matching is worth +0.007,
not +0.025**. See `proj_2_attempt3/kg/FINDINGS_newgold.md`.

**The gold is a TEST SET, not graph content.** It measures the extractor and must never be merged
into the graph — a union build was made and reverted on 2026-09-17, because a partly hand-curated
graph can no longer support the claim it exists to make ("this is what a model extracted from the
literature"). The per-edge "human-backed vs model-only" quality tier that came with it is
**retracted**: the model-only bucket is *defined* as the edges the human did not confirm, so that
comparison measured corroborated vs uncorroborated, not human vs model.

**The replacement, and the cleanest extractor-quality number here, because it depends on neither the
in-house gold nor the model:** on the same 259 papers, human annotation and model output scored
*independently* against a third party — human 80.5% (Disbiome) / 80.9% (Peryton), model 74.8% /
73.5%. **Neither gap is significant** (Fisher p = 0.275 / 0.186). The model sits within ~6 points of
a human curator, and a human curator's own agreement with Disbiome is 80.5%, not 100% — that is the
ceiling, not 1.0.

**Best current fidelity number (2026-09-12): reading fidelity ≥ 86.6%**, 95% CI [81.7, 91.3], from
181/209 scoreable observations across 122 papers — measured against *the papers' own sentences*, so it
depends on neither the in-house gold nor the curated databases. All 28 residual disagreements were
adjudicated twice independently (25/28 exact agreement) and **none is an extraction error**, so this is
a lower bound. It is a *different quantity* from the 73%/72.5% agreement figures — reading fidelity,
not cross-literature reproducibility — and must not be quoted as moving them. See
`proj_2_attempt3/kg/FINDINGS_direction_audit.md`.

**Precision of SCOPE (2026-09-19): 93.3% of observations come from a confirmed
disease-vs-healthy-control contrast, and out-of-gate is bounded at 1.8%–2.7%.** Every other
instrument on this page scores whether an edge's *taxon and direction* are right; this one asks
whether the *comparison* was admissible at all, which `samgated-v1` restricts to disease vs healthy
control. All 271 contributing papers were read against their own sentences by adjudicators blinded
to the disease label and to everything graph-side. **Quote the range, not a point estimate.**
A **separate** number from the same pass, which must be quoted apart from that one and never added
to it: within papers that *are* in scope, observations whose taxon is never reported against a
control in any sentence naming it are **~1.3% of the graph, 95% CI [0.8%, 1.8%]** — 48.7% of an
84-observation flagged set adjudicate as genuinely out-of-gate, CI [29.0, 65.8] over a
paper-clustered bootstrap. It bounds contamination *inside the flagged set only*, so it is a floor,
not a ceiling. Getting there required **overruling 16 of 20 adjudicator verdicts** — they inferred
healthy-control comparators the text never states — with the re-read run in both directions (all 18
opposing verdicts survived) and every override reasoned in `contrast_candidate_override.json`.
Out-of-gate papers disagree with the leave-one-out literature **1.75×** as often (O/E 1.747 vs
0.989, p = 0.00015 over 20,000 paper-level permutations, MDE ±0.303, four attacks survived) — the
project's first *validated* quality flag, built to avoid the flaw that got the 2026-09-17 provenance
tier retracted. It is worth ~6 excess disagreements: **a flag, not an accuracy gain.** Two nulls with
power from the same pass — the free-text disease label does not predict out-of-gate design
(p = 1.000, MDE ±8.5 pts), and *mixed provenance*, the dominant residual risk (94 of 241 in-scope
papers also report a within-disease subgroup contrast), does not degrade agreement (p = 0.650,
MDE ±0.140). See `proj_2_attempt3/kg/FINDINGS_contrast_scope.md`.

**Two edge-content decisions are open and waiting on a human, both on the Alzheimer's node.** The
SILCODE amyloid paper contributes **13 edges** to `Alzheimer's disease` and **no subject in it has
Alzheimer's** — every result sentence contrasts cognitively normal amyloid-positive against
cognitively normal amyloid-negative, and 10 of the 13 land on contested pairs including
*Faecalibacterium* (16 papers). A second paper contributes **7 AD edges** from AD-with vs
AD-without neuropsychiatric symptoms, so AD is the background rather than the contrast. Neither was
changed: dropping a paper is a corpus-inclusion decision, and there is no correct node for
"amyloid-positive but cognitively normal". The tiered list ships opt-in as
`proj_2_attempt3/kg/contrast_out_of_gate.json`; `graph.json` is untouched.

**Best current recall number (2026-09-16): paper-level recall ≥96.1%**, and **99.6%**
[97.9, 99.9] counting only the one miss that is *confirmable*. Like the fidelity figure
this is measured against *the papers' own sentences*, so it depends on neither the
in-house gold nor the curated databases. Of 313 deduplicated screened papers, 271
contribute at least one edge (86.6% paper yield); of the 42 that contribute nothing, 9
have no relation-bearing sentence at all and the other 33 adjudicate to 14 correct
refusals on study design, 4 explicit negative results, 4 background-only, 7 unclear, and
**1 confirmed miss** (3 further papers state a direction but report no significance —
see below). **Quote the range, not a point estimate.**

**The reason that is a range is the most transferable thing here, and it caught a wrong
number in this very file within the hour.** The extraction prompt
(`eval-v2/run_eval.py`, `samgated-v1`) does not extract anything that merely *states a
direction*: it requires **reported statistical significance** ("if significance is
unclear or unreported for a taxon, **omit it**"), **main text only** (not tables,
figures or supplementary), and **disease vs healthy control only**. An audit that scores
the extractor without applying its own gate will manufacture misses — the first pass
here reported 4 and the true confirmable count is 1. Conversely
`relation_sentences_clean.json` holds only sentences with a taxon *and* a direction cue
(~10% of corpus text), so a sentence reporting significance without a direction word is
invisible to it. **Net: this instrument can confirm a miss but cannot refute one.** Any
future recall work must apply the gate and state that asymmetry.

It is not an accuracy gain — nothing in the graph was changed. A deterministic
side-result worth keeping: `build_kg.py` loses nothing, i.e. zero papers had extracted
taxa that failed to become an edge. See `proj_2_attempt3/kg/FINDINGS_zero_yield.md`.

**Edge-level recall (2026-09-16): at most ~98.1%**, 95% CI [96.4, 99.5] — roughly 60
missed observations [14, 116] against the 3,077 in the graph. This is the *separate,
harder* question the paper-level figure above does not answer, and it is the first
edge-level recall number the project has that does not depend on the flawed gold. From
378 candidate (paper, taxon) pairs across 94 papers, a random sample of 24 papers / 95
candidates adjudicated with the gate applied. **Quote it as an upper bound**: the
candidate generator only sees taxa in sentences the provenance screen keeps (~75%
paper-level recall), and the significance gate can be confirmed from visible text but
never refuted — both biases push recall down, not up. **The actionable part is the
clustering, not the average: 12 of the 15 confirmed misses come from 3 papers**, so a
short-list re-extraction recovers most of the loss without a corpus-scale GPU run
(candidates enumerated in `kg/edge_recall_packets.json`). A tempting false finding was
tested and rejected here — high-rank taxa are *not* missed more often (p=0.054); the
apparent 33%-vs-6.5% phylum skew is a property of the candidate generator, 27% of whose
candidates are already phylum/class. See `proj_2_attempt3/kg/FINDINGS_edge_recall.md`.

**The KG is broad-scoped** — all microbe–disease relationships, not a single disease area. The current
gold-standard/test set happens to skew neuro-adjacent (Parkinson's, MS, Alzheimer's, ALS, stroke,
dementia, SMA, epilepsy, and others) simply because that's the disease mix in the test papers we have
on hand — it is **not** a decision to narrow the domain.

## Where the live work is

**`proj_2_attempt3/`** is the only active directory. Everything else in the repo is archival
history (`mega_dump/`) or superseded sub-approaches. If you're doing real work, you're almost
certainly in `proj_2_attempt3/` — see its `CLAUDE.md` for the active pipeline, data sets, and schema.

## Repo map (top level)

```
KnightLabV2/
├── CLAUDE.md                 # this file — project overview + lineage
├── .gitignore
├── mega_dump/                # ARCHIVAL. Historical attempts, kept for reference only.
│   ├── dump/                 # corpus-cleaning tooling + data-provenance docs (Llama3-8B text cleaner)
│   ├── proj_1/               # SEPARATE early project: ENA sample-metadata harvesting (not literature IE)
│   ├── proj_2/               # SUBMODULE -> TC960/KG-knightlab. The MicrobioRel KG attempt (see lineage)
│   └── proj_2_attempt2/      # abandoned: supervised PubMedBERT NER (CoNLL/BIO tagging)
├── docs/                     # GitHub Pages site (generated; docs/index.html == proj_2_attempt3/kg/kg.html)
└── proj_2_attempt3/          # ACTIVE. LLM prompting + gold-standard eval. Has its own CLAUDE.md.
    └── kg/                   # knowledge-graph construction, validation, RAG
```

**`.gitmodules` matters.** `mega_dump/proj_2` is a real submodule pointing at
`https://github.com/TC960/KG-knightlab.git`. It sat in the tree as a gitlink with **no `.gitmodules`
entry at all** until 2026-08-31, so every `git submodule update --init` — including GitHub Pages'
checkout — died with `fatal: No url found for submodule path`. If a clone or CI job fails on
submodules, check that file first.

## Lineage / how we got here

The arc is **supervised NER → LLM prompting + rigorous eval**:

1. **proj_1 (ENA)** — early, tangential: harvest microbiome *sample* metadata from the ENA
   (European Nucleotide Archive) portal API. Same lab/domain, not the text-IE pipeline.
2. **Corpus construction** (`mega_dump/dump/`) — scrape ~2,000 papers from DOI links → clean with a
   quantized **Meta-Llama-3-8B** pipeline → canonical chunked corpus of **2,026 papers / 241,873 chunks**
   (`research_content_cleaned_20250824_222305.json`). This corpus survives today as `MAIN_DATA.json`.
   Provenance is documented in `mega_dump/dump/Research_Data_Processing_Documentation.txt`.
3. **proj_2_attempt2** — first IE approach: hand-tag tokens (BIO), convert to CoNLL, fine-tune
   **PubMedBERT** for token-classification NER. Abandoned (needs heavy manual annotation, rigid).
3b. **`mega_dump/proj_2` (submodule: TC960/KG-knightlab)** — a *previous KG build*, and the most
   instructive failure in the repo. It used **MicrobioRel**, a fine-tuned BioBERT relation extractor
   with an **open 22-label schema** (`config.json`: Associated_with, Interacts_with, Location_of,
   Marker/Mechanism, part_of, physically_related_to, causes, affects, increase, decrease, …) over
   many entity types. `PHASE2_FINAL_ANALYSIS_REPORT.md` records the outcome: **6% precision**, 88%
   false positives on a stratified sample of 50 from 6,363 extracted relations. The diagnosis is the
   useful part — **process–process relations were 0% precise** and accounted for 73% of errors, and
   the generic predicates (`part_of`, `physically_related_to`) another 50%; raising the confidence
   threshold *lowered* precision, i.e. the model's confidence was anti-calibrated.

   **This is the direct argument for attempt3's design.** attempt3 deliberately uses a *closed*
   schema — one relation type over one entity pair (microbial taxon -> disease), direction only
   (enriched/depleted), gated on disease-vs-healthy-control — and lands at ~77% agreement with
   curated databases. Open-schema biomedical RE was tried here and produced 6%. Do not reintroduce
   generic predicates or process entities without re-reading that report.
4. **proj_2_attempt3 (current)** — pivot to **LLM-based extraction**: use Claude (multi-agent) to
   build a gold standard, then prompt & benchmark open-weight models against it. Qwopus3.5-27B
   (q4_k_m GGUF) was chosen and run over all 250 usable papers; those relations feed
   `proj_2_attempt3/kg/`, which is built, validated and published.

`mega_dump/` is purely for reference — don't build on it. The one artifact that carries forward
is the chunked corpus (below).

## Core data format — `MAIN_DATA.json`

The canonical corpus. Tracked as `proj_2_attempt3/MAIN_DATA.json.zip` (~33 MB; the unzipped
`MAIN_DATA.json` ≈ 105 MB and is **gitignored**). It is a JSON **dict keyed by string paper ID**:

```json
{
  "1": {
    "name": "The Oral Microbiota May Have Influence on Oral Cancer",
    "url": "https://pmc.ncbi.nlm.nih.gov/articles/pmid/32010645/",
    "original_url": "…",
    "source": "pmc",
    "chunks": ["Abstract\nThe oral microbiota plays…", "Therefore, in this study…", "…"]
  },
  "2": { … }
}
```

`chunks` is an ordered list of cleaned section/paragraph-level text strings.
`concatenate_chunks.ipynb` merges a paper's `chunks[]` into one full-text string
(→ `MERGED_PAPERS.json`, also gitignored).

## Gitignored / untracked (don't expect these in git)

- `MAIN_DATA.json`, `MERGED_PAPERS.json` — large derived corpora (only the `.zip` is tracked).
- `*.env`, `.env.env` — API keys.
- `labenv/` — Python virtual env.
- `proj_2_attempt3/dsmlp_model_prompting/benchmark_results/` — eval output dumps.
- `.claude/` — Claude Code local settings.

## Conventions

- Work happens in Jupyter notebooks (`.ipynb`) plus a few Python scripts; there is no single
  app entrypoint or test suite.
- Treat `MAIN_DATA.json` and the corpus as read-only inputs.
- See `proj_2_attempt3/CLAUDE.md` for the active data sets, extraction schema, eval workflow,
  and the current cleanup TODOs.

## Knowledge graph (`proj_2_attempt3/kg/`)

Built from the 250-paper extraction. Published: <https://www.mohakprakash.com/KnightLabV2/>
(also `docs/index.html`; the artifact copy is the same file).

| script | what it does |
|---|---|
| `taxonomy.py` | resolves taxon strings to NCBI taxids by reading `names.dmp` directly (no taxonkit binary, so it runs on macOS). Folds synonyms/renames: Bacteroidetes+Bacteroidota -> 976, Firmicutes+Bacillota -> 1239 |
| `build_kg.py` | extraction -> `graph.json` (nodes, association edges, containment links, paper table) |
| `build_viz.py` + `viz_network.js` | `graph.json` -> `kg.html`: force-directed network + ranked bars + per-study metadata panel |
| `extract_metadata.py` | second LLM pass over the same papers for study design (country, cohort size, sequencing, body site, 16S region, medication/diet control) -> `metadata.jsonl`, 250/250, 0 parse errors |
| `analyze_contested.py` | do the papers that disagree differ by study design? |
| `validate_external.py` | joins the graph against Disbiome (live API) and Peryton (manual TSV export) |
| `build_rag.py` | `graph.json` -> `rag_corpus.jsonl`, one document per edge, + a hybrid BM25/entity retriever |

### Design decisions that are load-bearing

- **Edge weight is evidence count, not effect size.** The extractor returns direction only, and the
  source papers report incommensurable statistics (LEfSe LDA, fold-change, p-values). A pooled
  "magnitude" would be invented precision.
- **Contested edges are kept, never averaged.** 220 pairs are contested in the current `graph.json`
  (`meta.n_contested`; "217" and, before it, "174" were both stale). ~1 taxon in
  3 flips sign between cohorts in this literature, so disagreement is a finding, not noise.
- **Containment is modelled, not collapsed.** 2,384 ancestor-descendant pairs sit within the same
  disease. Merging ranks would destroy real signal: in Parkinson's, *Lachnospiraceae* (family) is
  depleted in 8 of the 9 papers reporting it while *Hungatella* (a genus inside it) is enriched in
  6 of 7, and one study reports both directions itself. (An earlier "15 papers / 7" here predated
  the 2026-09-03 deduplication.) Corpus-wide, related taxa agree on direction 89% of the time
  within a single paper vs 54% for unrelated taxa — so containment is mostly redundant, and the
  11% that disagree are exactly what this layer is for. Synonym
  folding (same rank, renamed) and containment (different ranks) are different operations.
- **Never join on another database's taxid.** Disbiome records "Prevotella" as taxid 59823
  (*Prevotella sp.*, a species) where the genus is 838. Joining on their stored id silently dropped
  *Prevotella*/Parkinson's — 16 papers, and an *agreement*. Both sides must pass through
  `taxonomy.py`. Doing so moved overlap 188 -> 238 and recall 41% -> 54%.

### Open questions

- **Contested edges are unexplained — but as of 2026-09-10 we know the variance is real and
  where it lives.** Disagreement with the rest of the literature is a property of the **paper**:
  scored against the leave-one-out majority, 377 of 1,367 decisive observations (27.6%) disagree,
  and which papers hold the minority direction is clustered far beyond a within-edge null
  (p = 0.0003; p = 0.0013 after dropping within-paper taxonomic relatives). No paper is
  systematically *inverted* — a well-powered null: 0 of 134 survive BH, and a fully inverted copy
  would have been caught for 81 of them. But nothing extracted explains the offset. Country,
  cohort size, sequencing platform, 16S region, medication and diet control and disease identity
  are all null at MDEs of 16–22% **once the exact within-edge expectation is used as the offset**;
  on raw disagreement rate three of them survive BH and all three are edge-depth artifacts. A
  second pass adding 15 wet-lab/bioinformatics variables (extraction kit, pipeline, OTU vs ASV,
  LEfSe vs DESeq2, rarefaction, platform, year) is also null — **24 variables, 24 nulls**.
  **The size is why:** the paper-level SD of discordance is only **3.4 percentage points** on a
  27.6% base (cluster-bootstrap CI [0.0, 6.0], including zero), against MDEs of ±4–7 points. So
  the defensible claim is that this corpus *cannot answer* whether kit or pipeline drives
  disagreement, not that they don't — and **~85% of the variance is edge structure, not paper
  identity**, which is quantitative support for keeping contested edges rather than averaging
  them. See `proj_2_attempt3/kg/FINDINGS_paper_discordance.md`. The earlier edge-level result
  stands too: study design at FDR 0.243, `country=China` splitting 45/44.
- **Next planned analysis: embeddings.** Embed the full texts and test, *within* each contested edge,
  whether the up-papers separate from the down-papers, with permutation testing. Note the naive
  framing "do papers producing contested edges differ from papers producing unanimous ones" is
  **ill-posed** — 145 of 211 contributing papers do both, and only 7 are contested-only. The
  comparison must be within a fixed taxon-disease pair.
- 11 pairs are contradicted by **both** Disbiome and Peryton — the highest-value review targets.
- ~~254 of 925 taxa never resolve to a taxid (16S clade labels)~~ — **that framing was wrong,
  2026-09-11.** It is now **212 of 883**, and the difference was not clade labels at all: 12
  concepts were split across two nodes by punctuation alone (the placeholder branch of
  `norm_taxon` returns before the separator collapse added on 2026-09-08), and 33 labels are
  **the papers' own misspellings** — `Fecalibacterium`, `Subdogranulum`, `Lachinospiracea` —
  all 33 verified to occur verbatim in their source paper's text, so none is an extraction
  error. Folded via a curated table with 13 recorded refusals, because edit distance would
  have merged `Oscillospirales` into `Oscillospira` and undone the placeholder split. See
  `proj_2_attempt3/kg/FINDINGS_taxon_spelling.md` and `taxon_typos.py`. What remains really
  is clade labels (`SMB53`, `cc115`, `PAC000195_g`) plus real taxa absent from the cached
  taxdump (`Anaerostignum`, `Mogibacteriaceae`).
- **Only 23 of the 271 contributing papers have ever been screened for study design
  (2026-09-11).** `maindata_screen.json` covers the 45 title-matched MAIN_DATA additions only;
  the ~250 datasheet papers were never put through it. An animal-study prefilter validated at
  **recall 15/15** against that gold set returns a null on the rest (13 flagged, all 13 genuine
  human case-control), so no animal-only study is in the graph — but animal studies were only
  15 of 22 drop reasons, and no-healthy-control / case-report / review remain unscreened across
  248 papers. See `proj_2_attempt3/kg/FINDINGS_corpus_screen.md`. **This is the cheapest
  unblocked lever left: no GPU, no taxdump, no new papers.**
- ~~54 named species folded into their genus~~ — **FIXED 2026-09-08**, and it was never blocked on
  the taxdump. It was **24** species, not 54; the other 91 child folds must *not* be split
  (`Escherichia / Shigella` names two taxa, `Clostridium_XlVa` is a cluster label). The mapping comes
  from joining Disbiome's pre-rename names to NCBI on the **stable taxid** — see
  `proj_2_attempt3/kg/species_synonyms.py` and `FINDINGS_species_split.md`.
- **Punctuation was fragmenting concepts across nodes — FIXED 2026-09-08.** `Escherichia-Shigella`
  (the standard SILVA label for two genera 16S cannot separate) was split over **four** nodes by
  hyphen/slash/en-dash/underscore alone, and the underscore spelling was being filed under
  *Escherichia* outright. 17 concepts were affected. See `multi_taxon.py` and
  `FINDINGS_species_split.md`.
- **What remains is a decision, not an analysis** — but there is now one fewer of them.
  Whether a joint two-genus 16S signal should be attributed to one genus, split, or held
  separate (it is now held separate, on its own node) is still a modelling call for a human.
  **Disease subtypes as containment is ANSWERED (2026-09-14).** The disease half of the graph
  now resolves against MONDO (`kg/mondo.py`; the MONDO GitHub release is reachable even though
  NCBI is not), which confirmed 2 of 2 checkable is-a claims and upheld 2 of 2 rejections — and
  the 71-paper cognitive-decline cluster **does not cohere microbially**: 0.592 directional
  agreement against a 0.672 background over 453 cross-cluster pairs, with every MCI pair at or
  below a coin flip while the MONDO-confirmed Alzheimer's/Dementia link runs 0.938.
  **Caveat added 2026-09-15: the pooled result is robust, but NO individual pair survives
  multiple-comparison correction** — exact binomial + BH over all 16 pairs at q=0.05 gives
  **0 of 16**, the AD/Dementia 0.938 link (p=0.030) included. Quote the cluster-level number,
  not the per-pair ones. The pooled conclusion is unmoved by dropping the HIV-cohort
  `Neurocognitive impairment` node (0.592 → 0.582, i.e. *further* from background). So: link
  those nodes for **retrieval**, do **not** pool their evidence, and do **not** fold MCI into
  Alzheimer's. The 2 MONDO is-a links ship as an opt-in `kg/disease_hierarchy_links.json`; they
  add **no** retrieval reach in the PPR retriever (measured — identical subgraphs), only
  explicit attribution. See `kg/FINDINGS_disease_ontology.md`. Everything else is
  limited by n=272, which needs a GPU.

**Disease identifiers were wrong on 209 edges until 2026-09-14.** `Mild cognitive impairment`
(13 papers, 154 edges) carried `MONDO:0005453`, which is *congenital heart disease*, and
`Autism spectrum disorder` carried the id for narrow *autism*, a child of ASD. Both were live
in `graph.json`, `rag_corpus.jsonl` and the published `kg.html`. The rule already in this file —
**never join on another database's identifier** — applies to the disease dimension too, and this
shipped because every fidelity instrument here scores the *taxon* half of an edge. `MCI` now
correctly has **no** MONDO id; do not add one.
