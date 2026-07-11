# CLAUDE.md — proj_2_attempt3 (active work)

> The live directory. **End goal: build a microbe–disease knowledge graph.** Current stage is the
> means to that end — extract **microbe–disease relations** (taxa enriched ↑ / depleted ↓ per disease)
> from full-text papers and **benchmark open-weight LLMs** against a hand-built gold standard, so the
> edges are trustworthy before the graph is assembled. KG construction hasn't started; extraction is
> the current bottleneck. Repo-wide context (lineage, `MAIN_DATA.json` format) is in the parent `../CLAUDE.md`.

## Directory status — active vs legacy

| Path | Role | Status |
|---|---|---|
| `EmilySong_GoldStandardPaper/` | Gold-standard dataset construction + the eval sets | **ACTIVE** |
| `dsmlp_model_prompting/eval-v1/` | LLM eval pipeline (notebooks + results) | **ACTIVE** |
| `dsmlp_model_prompting/eval-v2/` | Taxonomy-aware eval harness + method experiments on `test_set_v2` | **ACTIVE** |
| `MAIN_DATA.json.zip` | Canonical 2,026-paper chunked corpus (see `../CLAUDE.md`) | input, read-only |
| `frameworks.md` | Notes on candidate biomedical-IE frameworks (LLM-IE, RELATE, LangExtract) | research notes |
| `bibliographies.txt` | Reference-paper URLs | notes |
| `archive/` | Superseded work, kept for reference only (see below) | **LEGACY** |

## Data sets (`EmilySong_GoldStandardPaper/`)

Human annotations come from a labmate (Emily Song); master sheet
`ALL_EMILY_PAPERS_WITH_(inGoldStd)_COLUMN.csv` (~342 rows) with taxa columns.
`gold_standard_paper_extract_and_eda.ipynb` scrapes PMC full text and builds the sets.

| File | Records | Meaning |
|---|---|---|
| `all_usable_papers.json` | 250 | all scraped papers (source of truth input) |
| `test_set_v2.json` | 15 | **CURRENT held-out test set** — disease-stratified. Use for all eval. Do not modify. |
| `holdout_pool.json` | 57 | reserve pool (passed quality filter, not in the 15) |
| `gold_standard_final_15.json` | 15 | **OLD/stale** 15-paper set (built on a subjective Yes/No flag) |

**Test-set build logic** (in the EDA notebook): 250 scraped → keep 88 with non-empty taxa →
full-text completeness score 0–5 (length, p-value count, figure/table refs, methods vocab like
QIIME/DADA2/16S/LEfSe) → keep score ≥4 → **72** → disease-stratified split (`seed=42`) into
**15 test + 57 holdout**.

### Gold-standard record schema

```json
{
  "title": "…",
  "link": "https://doi.org/…",
  "disease": "Alzheimer's disease (AD)",
  "taxa_enriched": "Phyllobacterium; Lactobacillus salivarius; …",   // ';'/',' string, may be NaN
  "taxa_depleted": "Rhodospirillales; Bacteroides; …",               // may be NaN
  "text": "ORIGINAL RESEARCH article …",
  "char_len": 53661,
  "usable": true
  // test_set_v2 / holdout also carry filter columns:
  // has_enriched, has_depleted, has_annotation, chars, has_methods,
  // has_results, has_discussion, likely_full, full_score, score, disease_clean
}
```

## Eval pipeline (`dsmlp_model_prompting/eval-v1/`)

Models are served on UCSD **DSMLP** GPUs. `main_eval.ipynb` is the current driver; the
per-model notebooks are precursors.

| Notebook | Model | Serving stack |
|---|---|---|
| `main_eval.ipynb` | `Jackrong/Qwopus3.5-27B-v3-GGUF` (Q8_0) — newest | llama.cpp |
| `qwopus.ipynb` / `qwopus_with_updated_prompt.ipynb` | Qwopus 27B v1 GGUF (Q4_K_M) | llama.cpp |
| `LLama3.3-70B-FP8.ipynb` | `nvidia/Llama-3.3-70B-Instruct-FP8` | vLLM |

"**Qwopus**" = a Qwen-3.5-27B model distilled on Claude-Opus reasoning traces.

**Flow:** load a gold-standard JSON → `smart_truncate()` (cut at References) → format prompt →
`create_chat_completion(temperature=0, max_tokens=8192)` → `parse_output()` (strip `</think>` +
```json fences) → write `benchmark_results/{tag}.json` + `.csv`.

**Prompt generations** (all in `main_eval.ipynb`): (1) simple JSON prompt with `/no_think`;
(2) `Prompt_alt` — XML-tagged prompt with 3 few-shot examples + rules (use only
Abstract/Results/Discussion, copy taxon names verbatim); (3) **`LlamaGrammar`** GBNF grammar to
force the `{"disease", "taxa_enriched":[…], "taxa_depleted":[…]}` schema. Runs tagged
`w_cot` / `wo_cot`.

**Metric** (`main_eval.ipynb` cells 9–10): taxa split on `[,;]`, lowercased, p-values stripped;
predicted↔expected matched by **TF-IDF char n-gram cosine ≥ 0.5** (fuzzy). Report micro
**Precision / Recall / F1**. Result CSV columns: `title, disease, in_gold_standard,
expected_enriched, expected_depleted, predicted_enriched, predicted_depleted, predicted_disease,
time_seconds, parse_error`. `benchmark_results_pt2/` (Qwopus v3, 2026-04-09) is the latest run.

**Finding so far:** Qwopus v3 (with or without CoT) underperformed Qwopus v1 — see git log.

## eval-v2 — taxonomy-aware harness + method experiments (`dsmlp_model_prompting/eval-v2/`)

Active eval round on `test_set_v2` (15 papers). Built out 2026-07-10/11.

**Harness.** `run_eval.py --model <key>` runs one GGUF model; scoring now defaults to a
**taxonomy-aware (LCA) metric** (`taxonomy_match.py`): each predicted taxon matches an expected one if
their NCBI lineages are **nested** (one is an ancestor of the other — collapses genus/family/species
rank variants), else falls back to char-ngram cosine ≥0.5. `--metric char` reverts. Setup once:
`python taxonomy_match.py --setup` (fetches taxonkit + gnparser + NCBI taxdump to `/tmp`; **rebuild after
any pod reset — `/tmp` is ephemeral**). Taxonomy scoring lifted Qwopus3.5 from **.642 → .751** vs the same
original gold (fairer metric, not a better model).

**Two benchmarks.** `results/leaderboard.csv` = original human gold. `results/fable_gold_15.json` = a
**thorough Fable-5 (Claude) re-annotation** ("Fable Benchmark"). Scores diverge because the two disagree
on ~70 taxa — **UNRESOLVED whether the human gold is incomplete or Fable over-extracts** (SMA paper: Fable
was wrong; elsewhere its extras sit near significance cues). Needs human adjudication vs Sam's gates.

**Leaderboard (taxonomy metric).** vs original gold: Qwopus3.5 **.751**, Qwythos-9B .665, Qwen2.5-32B .606.
vs Fable gold: Qwopus3.5 **.806** (best local). Fable-5 tops both but is the reference/oracle, not a
contestant (scoring it on its own gold is circular). **Decision: Qwopus3.5 is the model for corpus-scale
extraction** (free/local, best local quality, ~20–22s/paper → ~35 min per 100 papers, within budget).

**Method experiments** (`experiments/`, real GGUF runs on the H100, logged in `experiments/logs/`):
- `run_gguf_experiments.py` runs RELATE + grounded per model; `score_experiments.py` → `results/matrix.json`;
  write-up in `experiments/RESULTS.md`, pilot design in `experiments/README.md`.
- **RELATE (retrieve→rerank-with-reject) BACKFIRED** on all local models (weak quantized models can't
  reject noisy regex candidates → FPs rise). Needs a strong reranker + real NER/dense retrieval.
- **Grounded** (verbatim-sentence-per-taxon) helped only the under-extractor (Qwen). **Normalize** = 0 F1
  change (it's a KG node-dedup tool, not a scoring lever). **No method beat plain single-shot** on the
  local models — **the benchmark's completeness is the ceiling, not the prompt.**
- **LLM-judge is a DATA LEAK as prototyped** (recovered "FPs" using the Fable gold as oracle → needs a
  gold, which won't exist at 30k-paper scale). Real version = **gold-free per-claim verification**
  ("does the paper support this claim? y/n → drop the no's"). Not yet built.

**Next:** adjudicate Fable's ~70 extra taxa vs Sam's gates (settle incomplete-vs-correct); then start KG
edges with Qwopus over the corpus + gold-free verification pass. Validate vs Peryton/Disbiome.

## Legacy reference — `archive/` (don't build on these)

Everything below was moved into `archive/` to keep the working tree clean. Kept only for
methodology reference.

- **`archive/claude_annotation_old/`** — how the gold standard was pre-annotated. A
  Queen(Opus)+9-workers(Sonnet) multi-agent Claude pipeline (extract → validate → adversarial-criticize)
  with a recall bias.
  - Extraction schema: `OLD_schema.md` (minimal microbe–disease: `taxon_name, domain,
    taxonomic_level, direction, change_context, sample_site, p_value, confidence`). `future_schema.md`
    is an aspirational expanded multi-entity schema (metabolites, immune markers). **Both `OLD_schema.md`
    and `future_schema.md` are slated for deletion** — schema is being re-scoped (TODO, not done yet).
  - Latest annotations: `LATEST_claude_first_pass_annotation/` (uses the v2 qualification gate);
    earlier run in `claude_first_pass_annotation_1/` (`summary.json`: 30 papers, 510 relations).
  - Prompts: `data_annotation/it2_prompt_for_annotating_data_with_claude.md`. Manual review UI:
    `data_annotation/annotation_tool.html` (standalone, no Label Studio).
- **`archive/prev_methods/`** — abandoned extraction approaches: `labelStudioBS/` (Label Studio
  review, gave up), `withLLMs/` (frontier-model API benchmark — "SCRAPPED: TOO EXPENSIVE"),
  `withLavaRouter/` (Lava proxy + local BioMistral cost-avoidance experiment).
- **`archive/concatenate_chunks.ipynb`** — one-off utility that merged corpus `chunks[]` →
  `MERGED_PAPERS.json`.

## Known issues / cleanup targets

- **Eval still points at the stale set.** The Qwopus notebooks load `gold_standard_final_15.json`,
  but the intended current set is `test_set_v2.json`. Migrating eval → `test_set_v2.json` is the
  pending work implied by the empty `eval-v2/` folder.
- **KG construction not started.** The whole point is a knowledge graph; we're still stuck getting
  extraction reliable enough to build edges from. That stage has no code yet.
- **Schema re-scope pending.** `archive/claude_annotation_old/{OLD_schema.md, future_schema.md}` are
  slated for deletion; the extraction schema needs to be re-scoped for KG edges.
- Two serving stacks coexist (vLLM for Llama-70B, llama.cpp GGUF for Qwopus) — no shared harness.
- `gold_standard_paper_extract_and_eda.ipynb` lists stale intermediate artifacts
  (`annotated_papers.json`, `annotated_fullpapers.json`, `usable_papers.json`) as deletable.
- No tests, no single entrypoint — work is notebook-driven.

## Conventions

- `test_set_v2.json` is the eval ground truth going forward — read-only.
- Treat `MAIN_DATA.json` / the corpus as read-only inputs.
- LLM eval runs on DSMLP GPUs (llama.cpp / vLLM), not local CPU.
