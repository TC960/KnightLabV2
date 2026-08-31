# CLAUDE.md — KnightLabV2 (repo root)

> Knight Lab microbiome NLP project. This root file gives the **big picture, lineage, and repo map**.
> The live working code has its own, more detailed `proj_2_attempt3/CLAUDE.md` — read that too when working there.

## What this project is

**The end goal is a knowledge graph** of microbe–disease relationships built from the microbiome
literature — nodes for microbial taxa and diseases, edges for "taxon X is **enriched ↑** / **depleted ↓**
in disease Y."

Extraction is done and **the graph is built**: see `proj_2_attempt3/kg/`, published at
<https://www.mohakprakash.com/KnightLabV2/>. 712 taxa (84% resolved to NCBI taxids), 27 diseases,
1,398 association edges plus 551 taxonomic-containment links, from 250 papers. It agrees with two
independent hand-curated databases at **77.5%** (Disbiome) and **75.6%** (Peryton) on edge direction.

**Caveat on the gold standard.** The human annotations are under audit and are turning out to be
unreliable; the annotator expects to report an error rate rather than a corrected set. So the
extractor's "F1 0.680" is *agreement with a flawed reference*, not accuracy. Three independent signs
of this: 162 of 250 papers have blank taxa columns, a thorough Opus 4.8 re-annotation found 72 taxa
the humans missed, and the 15-paper benchmark only rose 0.64 -> 0.84 once the gold was corrected.
Prefer the Disbiome/Peryton agreement figures — they do not depend on the in-house gold.

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
- **Contested edges are kept, never averaged.** 174 pairs have papers pointing both ways. ~1 taxon in
  3 flips sign between cohorts in this literature, so disagreement is a finding, not noise.
- **Containment is modelled, not collapsed.** 2,384 ancestor-descendant pairs sit within the same
  disease. Merging ranks would destroy real signal: in Parkinson's, *Lachnospiraceae* (family) is
  depleted across 15 papers while *Hungatella* (a genus inside it) is enriched across 7. Synonym
  folding (same rank, renamed) and containment (different ranks) are different operations.
- **Never join on another database's taxid.** Disbiome records "Prevotella" as taxid 59823
  (*Prevotella sp.*, a species) where the genus is 838. Joining on their stored id silently dropped
  *Prevotella*/Parkinson's — 16 papers, and an *agreement*. Both sides must pass through
  `taxonomy.py`. Doing so moved overlap 188 -> 238 and recall 41% -> 54%.

### Open questions

- **Contested edges are unexplained.** Study design does not account for them: after a cluster-robust
  permutation test (533 observations come from only 136 papers) and BH correction across 26
  categories, nothing survives — the best, `diet_controlled`, sits at FDR 0.243. `country=China`
  splits 45/44. This is "no effect visible at n=250", not "no effect".
- **Next planned analysis: embeddings.** Embed the full texts and test, *within* each contested edge,
  whether the up-papers separate from the down-papers, with permutation testing. Note the naive
  framing "do papers producing contested edges differ from papers producing unanimous ones" is
  **ill-posed** — 145 of 211 contributing papers do both, and only 7 are contested-only. The
  comparison must be within a fixed taxon-disease pair.
- 11 pairs are contradicted by **both** Disbiome and Peryton — the highest-value review targets.
- 113 of 712 taxa never resolve to a taxid (16S clade labels like `[Eubacterium] ventriosum group`).
