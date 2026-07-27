# Microbe–disease extraction — status & a couple of questions

**TL;DR.** We stress-tested our LLM extractors and the gold standard together. The low F1 scores we'd been seeing were **mostly the benchmark, not the models**. Scored fairly, a free local model (Qwopus) reaches **~0.84 F1**. We're going with **Qwopus** for scale-up. Two decisions below need your call.

## What we did
Used a frontier model (Claude / "Opus 4.8") two ways: (1) as an extractor, and (2) to build a **thorough re-annotation** of the 15-paper test set ("Opus 4.8 Benchmark") to compare against our original hand-annotated one. Then re-scored every model.

## What we found — the F1 was being held down by the *evaluation*, not the models
Qwopus3.5 (our best local model), same outputs throughout:

| Scored against | F1 |
|---|---|
| Original benchmark | 0.64 |
| Opus 4.8 benchmark (more complete) | 0.77 |
| + taxonomy-aware matching | 0.82 |
| + excluding 2 out-of-scope papers | **0.84** |

Three fixable measurement issues, none the model's fault:
1. **The original benchmark was incomplete** — it listed 119 taxa; a thorough pass found ~70 more real ones in the text. Extractions that were actually correct were being scored as false positives.
2. **Rank mismatches** — the old metric compared names like a spell-checker; it didn't know *Bacteroides* (genus) and *Bacteroidaceae* (family) are the same lineage. Now fixed with NCBI-taxonomy-aware matching (**baked into the eval harness**).
3. **Scope** — see question 1.

A useful sanity check that this isn't just the model flattering itself: the one model that *under-*extracts (Qwen2.5) got **worse** on the fuller benchmark, exactly as a real completeness fix should behave.

## Two questions for you
1. **Scope.** Two of the 15 papers have **no clean healthy-vs-disease control arm** (an HIV-cognition study and a memory-clinic study — they report symptom correlations / subgroup-vs-subgroup differences). Under the current annotation rule ("disease vs healthy control only"), the right output is *nothing*. Should such papers be **in-scope at all**? Right now they penalize every model no matter how good.
2. **Annotation rules — uniform?** Relatedly, we noticed at least one paper annotated as **group A vs group B** rather than control vs experimental. Should the guidelines be applied strictly & identically to every paper, or is per-paper judgment expected? This changes what "correct" means for the whole benchmark.

## Next
Move from evaluating extractors to **building the first knowledge-graph edges** — run Qwopus over the full corpus, normalize taxa to NCBI, emit microbe →↑/↓→ disease edges, and validate against curated DBs (Peryton, Disbiome, which cover our neurodegenerative disease mix).

*(Interactive write-up with the 15 papers annotated inline — green = original benchmark, yellow = the additions Opus 4.8 found — available on request.)*
