# CLAUDE.md — KnightLabV2 (repo root)

> Knight Lab microbiome NLP project. This root file gives the **big picture, lineage, and repo map**.
> The live working code has its own, more detailed `proj_2_attempt3/CLAUDE.md` — read that too when working there.

## What this project is

**The end goal is a knowledge graph** of microbe–disease relationships built from the microbiome
literature — nodes for microbial taxa and diseases, edges for "taxon X is **enriched ↑** / **depleted ↓**
in disease Y."

Everything currently in the repo is the **intermediate step**: reliably *extracting* those
relationships from full-text papers with LLMs, and *evaluating* the extractors against a hand-built
gold standard so we can trust the edges before assembling the graph. Relationship extraction is the
current bottleneck — the KG-construction stage hasn't started yet.

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
│   ├── proj_2/               # EMPTY (its big artifacts — cleaned corpus, model.safetensors — were relocated/removed)
│   └── proj_2_attempt2/      # abandoned: supervised PubMedBERT NER (CoNLL/BIO tagging)
└── proj_2_attempt3/          # ACTIVE. LLM prompting + gold-standard eval. Has its own CLAUDE.md.
```

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
4. **proj_2_attempt3 (current)** — pivot to **LLM-based extraction**: use Claude (multi-agent) to
   build a gold standard, then prompt & benchmark open-weight models (served on UCSD **DSMLP**) against it.
   Once extraction is trustworthy, the validated relations feed **knowledge-graph construction** (not yet built).

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
