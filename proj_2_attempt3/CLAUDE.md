# CLAUDE.md

## Project

GraphRAG system for 2,026 gut/oral microbiome-disease research papers. Extract structured relations from papers, build a knowledge graph, use it for retrieval-augmented generation.

## Data

- `MAIN_DATA.json` — 2,026 papers as chunked text. Dict keyed by paper ID, each entry has `name` (paper title) and `chunks` (list of text strings).
- `MERGED_PAPERS.json` — Same papers with chunks concatenated into single strings per paper.
- Papers cover gut and oral microbiome relationships with disease (e.g. Fusobacterium nucleatum in oral squamous cell carcinoma, gut dysbiosis in IBD, etc.)

## Environment

- Python 3.10, virtual env at `labenv/` (`source labenv/bin/activate`)
- Agent teams enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`)

## Schema

Defined in `curr_schema.md`. Minimal schema focused on **microbe-disease associations**:
- **Entities:** Microbe (taxon_name, domain, taxonomic_level) and Disease (primary_disease)
- **Relation:** `microbe_disease_association` with fields: direction, change_context, sample_site, p_value, confidence
- Output format is JSON per paper with a `relationships` array.

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

Workers each sample ~20-30 papers from the cleaned set, independently propose a relation extraction schema. Queen collects proposals, identifies disagreements, sends workers back to argue and refine. Loop until convergence.

**Output:** `curr_schema.md` — finalized and approved.

### Phase 3 — Gold-Standard Annotation (Two-Pass)

Build a gold-standard annotation set from ~30 papers to evaluate open-source models.

**Pass 1 — Claude-generated extractions (high recall):**
Use Claude to extract relations from ~30 selected papers following `curr_schema.md`. Intentionally over-include — favor recall over precision. Output JSON annotations per paper following the format in `data_annotation/annotation_schema.json`.

**Pass 2 — Manual review via Label Studio:**
Import Pass 1 annotations into Label Studio for human review. Correct false positives, fix entity boundaries, validate relation fields. Label Studio config and setup in `data_annotation/`.

**Note:** Label Studio requires `DEBUG=true` env var (not `WARN`) to start — see `data_annotation/annotation_assistant.ipynb` for the known issue.

**Output:** Gold-standard annotated JSON for ~30 papers.

### Phase 4 — Open-Source Model Evaluation (Google Colab)

Run the same extraction prompt from Phase 3 on the ~30 annotated papers using open-source models on Google Colab. Compare against gold-standard annotations for precision, recall, and schema adherence.

**Models to evaluate:** Llama 3.1, Qwen 2.5

**Output:** Evaluation notebook with per-model metrics and recommendation.

## Agent Team Rules

- Queen runs on Opus. Workers run on Sonnet.
- Workers in the same category must cross-validate before reporting.
- All outputs go to the project root directory.
- Do not modify `MAIN_DATA.json`.