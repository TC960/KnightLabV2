# CLAUDE.md

## Project

GraphRAG system for 2,026 gut/oral microbiome-disease research papers. Extract structured relations from papers, build a knowledge graph, use it for retrieval-augmented generation.

## Data

- `MAIN_DATA.json` — 2,026 papers as chunked text. Dict keyed by paper ID, each entry has `name` (paper title) and `chunks` (list of text strings).
- Papers cover gut and oral microbiome relationships with disease (e.g. Fusobacterium nucleatum in oral squamous cell carcinoma, gut dysbiosis in IBD, etc.)

## Environment

- Python 3.10, virtual env at `labenv/` (`source labenv/bin/activate`)
- LLM access via LAVA API (lava.so) — supports Sonnet 4.5, Opus 4.6, GPT 5.3, Kimi 2.5, Gemini 3 Pro
- Agent teams enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`)

## Pipeline

### Phase 1 — Data Quality Validation (Agent Team)

Spawn 8 workers + 1 queen. Workers operate in pairs (2 per category), cross-validate each other's findings. Queen resolves disagreements.

**Categories:**
1. **Text Integrity** — garbled text, encoding errors, PDF extraction artifacts, broken unicode, nonsensical character sequences
2. **Duplicate Detection** — near-duplicate papers, duplicate chunks within or across papers
3. **Chunk Completeness** — chunks cut mid-sentence/paragraph, empty chunks, chunks that are too short to be useful
4. **Content Quality** — chunks that are just references/bibliography, author bios, table headers without data, acknowledgments, or other non-extractable content vs chunks with actual microbiome-disease findings

**Output:** `phase1_data_quality_report.md` + flagged paper/chunk IDs per category.

### Phase 2 — Schema Debate (Agent Team)

Workers each sample ~20-30 papers from the cleaned set, independently propose a relation extraction schema (entity types, relation types, output format). Queen collects proposals, identifies disagreements, sends workers back to argue and refine. Loop until convergence. Present final schema to user for approval before proceeding.

**Output:** `curr_schema.md` with finalized entity types, relation types, directionality conventions, and example JSON output format.

### Phase 3 — LLM Benchmarking (Single Notebook)

One `.ipynb` notebook. Use the finalized schema from Phase 2 to write an extraction prompt. Run the same prompt on the same sample papers across all models via LAVA API. Compare outputs for precision, recall, schema adherence, and cost.

**Models to test:** Sonnet 4.5, Opus 4.6, GPT 5.3, Kimi 2.5, Gemini 3 Pro

**Output:** `benchmark.ipynb` with results and model recommendation.

## Agent Team Rules

- Queen runs on Opus. Workers run on Sonnet.
- Workers in the same category must cross-validate before reporting.
- All outputs go to the project root directory.
- Do not modify `MAIN_DATA.json`.