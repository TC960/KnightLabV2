# Extraction-method experiments

Five run files, each building on the original extraction, scored with the taxonomy-aware metric
(`../taxonomy_match.py`). Pilot = 5 papers (idx 0,1,4,10,11), LLM stages generated with Fable and
cached in `cache/`. The scripts are backend-agnostic — swap `cache/*.json` for a GGUF/Qwopus batch
on DSMLP to run the same suite on the production model.

```
bash run_all.sh                 # runs 0→4 sequentially, vs the Fable benchmark (EXP_GOLD=fable)
EXP_GOLD=orig python run_1_relate.py   # score any run vs the original (incomplete) gold instead
```

| # | file | idea | what it does |
|---|---|---|---|
| 0 | `run_0_original.py` | baseline | single-shot samgated-v1 prompt |
| 1 | `run_1_relate.py` | RELATE (2509.19057) | recall-biased candidate retrieval (Python) → LLM re-rank with an explicit **reject** option |
| 2 | `run_2_grounded.py` | LangExtract | every taxon must carry a **verbatim supporting sentence**; ungrounded ones dropped |
| 3 | `run_3_normalize.py` | taxonomy normalization | canonicalize to NCBI taxid, **collapse same-node names** (KG dedup) |
| 4 | `run_4_judge.py` | LLM-judge eval | adjudicate each metric "false positive" vs an oracle; recover the real ones |

## Pilot results (5 papers)

**vs the Fable (thorough) benchmark — recall differentiator** (precision is ~1.0 here because these are
Fable extractions vs Fable's own gold; only recall separates them):

| method | P | R | F1 |
|---|---|---|---|
| 2_grounded | 1.00 | **0.96** | **0.98** |
| 0_original | 1.00 | 0.77 | 0.87 |
| 3_normalize | 1.00 | 0.75 | 0.86 |
| 1_relate | 1.00 | 0.69 | 0.82 |

**vs the original (incomplete) gold — precision differentiator:**

| method | P | R | F1 | FP |
|---|---|---|---|---|
| 1_relate | **0.91** | 0.79 | 0.84 | **5** |
| 0_original | 0.90 | 0.82 | 0.86 | 6 |
| 2_grounded | 0.75 | 0.84 | 0.79 | 19 |

**judge (vs original gold):** metric F1 0.855 → **judge-adjusted 0.908**; the judge recovered **6/6**
metric "false positives" as real findings (precision 0.90 → 1.00).

## Takeaways
- **RELATE = precision.** Its reject option gave the **fewest false positives / highest precision**
  against the original gold — the direct fix for over-extraction. Cost: recall (it correctly rejects
  correlation-based findings a *thorough* gold happens to include). NOTE: our stage-1 retrieval was a
  crude regex (681 noisy candidates); a real NER/dense retriever would raise RELATE's recall too.
- **Grounded = recall.** Forcing a verbatim sentence per taxon caught the most (recall 0.96 vs the
  thorough gold) but over-includes → most FPs vs a strict gold. Great for discovery / high-recall.
- **They're complementary** → the strong pipeline is **grounded (recall) → RELATE-style reject
  (precision)**, i.e. extract-everything-with-evidence, then adjudicate with a reject option.
- **Normalize** collapsed nodes without moving F1 — confirms it's a **KG dedup** tool, not a scoring
  lever (matches the earlier finding).
- **LLM-judge** recovered every incomplete-gold false positive — worth adopting as a scoring aid when
  the benchmark is known to be incomplete.

Caveat: 5-paper pilot, Fable backend (so precision-vs-Fable is circular). The real test is this suite
on **Qwopus over the full 15+**, where there's precision headroom — that's the next run.
