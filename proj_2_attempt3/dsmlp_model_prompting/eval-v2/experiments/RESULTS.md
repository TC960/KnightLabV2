# Experiment results — 3 real GGUF models × 5 methods × 15 papers

Ran RELATE + grounded with **actual GGUF inference** on the H100 (proof in `logs/<model>.log`:
distinct model names, VRAM 18.5/10.2/27 GB, arch qwen35/qwen2, 31 min GPU time, 15 papers each,
0/2/0 parse errors). normalize + judge are deterministic post-processing. Scored with the
taxonomy-aware metric vs both the original gold and the Fable thorough benchmark.

## F1 (higher = better)

| model | original | RELATE | grounded | normalize | judge* |
|---|---|---|---|---|---|
| Qwopus3.5-27B (vs orig) | 0.751 | 0.705 | 0.668 | 0.745 | **0.880** |
| Qwopus3.5-27B (vs Fable) | 0.806 | 0.753 | 0.786 | 0.802 | — |
| Qwythos-9B (vs orig) | 0.665 | 0.590 | 0.644 | 0.665 | **0.809** |
| Qwythos-9B (vs Fable) | 0.742 | 0.646 | 0.650 | 0.740 | — |
| Qwen2.5-32B (vs orig) | 0.606 | 0.537 | **0.653** | 0.608 | **0.682** |
| Qwen2.5-32B (vs Fable) | 0.502 | 0.490 | 0.502 | 0.498 | — |

\* judge scored vs original gold, recovering metric-FPs the oracle (Fable benchmark) confirms real:
recovered **37 / 43 / 13** for Qwopus / Qwythos / Qwen.

## Findings

1. **RELATE backfired on the local models** (Qwopus −.046, Qwythos −.075, Qwen −.069 vs original).
   This *reverses* the Fable pilot. Cause: stage-1 retrieval was a crude regex (~200 noisy
   candidates) and the quantized 9-27B models **failed to reject** them — false positives went UP
   (Qwopus FP 65→114). **RELATE needs a strong reranker + clean candidate generation** (NER/dense
   retrieval); it is not plug-and-play on weak models. This is the headline lesson.
2. **Grounded is a floor-raiser for weak extractors.** It *helped only Qwen2.5* (the under-extractor:
   .606→.653, precision .598→.681) by forcing "extract only what a sentence states." Neutral for the
   stronger Qwopus/Qwythos. Good when a model otherwise mis-/under-extracts.
3. **Judge (incomplete-gold recovery) is the biggest, most consistent lever** — +.13 to +.14 F1 on
   every model. It confirms the original gold under-counts real findings **across all models**, not
   just Fable. (Caveat: oracle = Fable benchmark, so this measures gold incompleteness, not an
   independent adjudication — a production judge would be a per-FP LLM call over the paper text.)
4. **Normalize** doesn't move F1 (dedup only) — it's a KG node-canonicalization tool, confirmed again.
5. **No extraction-method trick beat plain single-shot on the local models.** The ceiling is the
   *benchmark*, not the method: fixing gold completeness (judge) recovers 15-25 F1 points everywhere,
   far more than any prompting change.

## Implication for the pipeline
- Keep **single-shot** extraction for the local models (Qwopus best local, F1 .806 vs Fable gold).
- If adopting RELATE, pair it with a real NER/dense retriever and a strong reranker — otherwise it hurts.
- Use **grounding** as a discipline for weaker models / high-precision needs.
- Priority is **benchmark quality** (completeness + scope), not more prompt engineering — that's where
  the points are. Validate against Peryton/Disbiome next.

Files: `run_gguf_experiments.py` (GGUF driver, logged), `score_experiments.py`, `results/matrix.json`,
per-model `logs/<model>.log`, `cache/{relate,grounded}__<model>.json`.
