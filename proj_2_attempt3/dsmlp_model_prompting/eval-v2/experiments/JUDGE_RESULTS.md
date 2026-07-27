# Gold-free LLM-as-judge verification — results & scalability

Real, gold-free judge: each model re-reads the paper and verifies its OWN single-shot extractions
("is taxon X significantly enriched/depleted in disease vs HC? yes/no"), dropping the no's. No gold
used → scalable in principle. Ran two batch sizes (1 claim/call vs 3 claims/call) on all 3 GGUF models,
15 papers, scored vs both golds. Driver `run_judge_experiments.py`, scorer `judge_score.py`,
`results/judge_matrix.json`, logs `logs/judge_<model>.log`.

## F1 (higher = better)

| model | original | judge-b1 | judge-b3 |
|---|---|---|---|
| Qwopus3.5 (vs orig) | .751 | **.772** | .761 |
| Qwopus3.5 (vs Opus 4.8) | .806 | .808 | .810 |
| Qwythos-9B (vs orig) | .665 | .602 | .624 |
| Qwythos-9B (vs Opus 4.8) | .742 | .606 | .613 |
| Qwen2.5-32B (vs orig) | .606 | .621 | .621 |
| Qwen2.5-32B (vs Opus 4.8) | .502 | .457 | .488 |

## Timing — batch-1 vs batch-3 (15 papers)

| model | batch-1 total | /paper | calls | batch-3 total | /paper | calls | speedup |
|---|---|---|---|---|---|---|---|
| Qwopus3.5 | 1643s | 109.5s | 196 | 628s | 41.8s | 70 | 2.6× |
| Qwythos-9B | 621s | 41.4s | 230 | 251s | 16.7s | 83 | 2.5× |
| Qwen2.5-32B | 1735s | 115.7s | 127 | 640s | 42.7s | 47 | 2.7× |

## Findings
1. **The judge is a precision↔recall trade, not a free win.** It only drops taxa → precision up, recall
   down. Net F1 barely moves.
2. **Only the strongest model (Qwopus) nets a small gain (+.02);** the weak model (Qwythos) LOSES (−.06,
   recall .846→.60) because a weak reader is a weak judge — it rejects real findings. Qwen: precision
   jumps but recall tanks. **A model is only as good a judge as it is an extractor.**
3. **batch-3 dominates batch-1:** ~identical metrics, ~2.6× faster. Never use batch-1. (Endpoint: one call
   per paper = all taxa in one verify call, ~extraction cost.)
4. **Not scalable at the latency bar.** Even batch-3 ≈ 42s/paper (Qwopus) — alone busts the 30-40s/paper
   budget, and it's ON TOP of ~22s extraction (~64s/paper combined) for a +.02 F1 best case. Bad trade.

## Decision
**Drop the standalone judge pass.** Fold significance-verification into the single extraction prompt
(0 extra calls) instead of a second model pass. Keep single-shot as the corpus-scale extractor.
