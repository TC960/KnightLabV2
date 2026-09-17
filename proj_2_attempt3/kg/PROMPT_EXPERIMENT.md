# Prompt-variant experiment — design

## Why not "run it three times"

The extractor runs at `temperature=0` (`run_eval.py:264`, `TEMP`). Repeated runs are
deterministic up to GPU non-determinism, so three identical reruns would measure
kernel jitter, not anything about extraction. To get a variance number you must
change something: either the sampling temperature or the prompt.

Prompt variants are the more valuable change, because there is a specific
hypothesis to test and a measured amount of headroom.

## The hypothesis

`samgated-v1` carries three hard gates:

```
SIGNIFICANCE     include a taxon ONLY if reported statistically significant
                 (p/FDR/q < 0.05, or a significant LEfSe/LDA result).
                 If significance is unclear or unreported, OMIT IT.
MAIN TEXT ONLY   ignore taxa appearing only in tables, figures, supplementary.
DISEASE vs HC    the comparison must be disease vs a healthy control group.
```

Emily's gold was curated from the whole paper without those constraints. So a
"false negative" may be the model correctly obeying its own instructions.

**Measured, before spending anything.** Of the 653 gold taxa the extractor missed:

| | count | share |
|---|---:|---:|
| not present in the paper text at all | 378 | 57.9% |
| present, but **no significance cue** in any sentence naming them | **116** | 17.8% |
| present **with** a significance cue — genuine candidates | **159** | 24.3% |

Example correct refusal: *"The phylum Bacteroidetes was typically the dominant
phylum in the gut microbiome (Figure S2)"* — descriptive, no test, and the figure
is supplementary. Omitting it is right under two separate gates.

So the real recall headroom is **~159 taxa**, not 653. That is what a prompt
change could plausibly recover, and it bounds the possible gain.

## Variants

| id | change | prediction |
|---|---|---|
| `A-baseline` | `samgated-v1` unchanged | F1 ≈ 0.739 (LCA), the number to beat |
| `B-softgate` | significance gate relaxed: accept an explicit directional claim about the disease-vs-control comparison even when the statistic is not restated in that sentence; still reject purely descriptive or correlational mentions | recall ↑, precision ↓ — **net F1 is the open question** |
| `C-tables` | drop MAIN TEXT ONLY; allow taxa from tables and captions | recall ↑ slightly. **Likely null**: measured across 50 papers, zero gold taxa appear in a table but not the body |

`C` is included as a control precisely because the local measurement predicts it
does nothing. If `C` moves F1, the local measurement was wrong and that is worth
knowing.

## Protocol

- **Stratified subset first.** 80 papers sampled across diseases, not the full 265.
  3 variants x 80 papers x ~22 s ≈ 88 min of compute versus ~5 h for the full set.
  A variant that cannot show an effect on 80 papers does not deserve 265.
- **Same scoring as the headline number**: `score_lca.py`, LCA metric, matcher
  guard in place (the double-counting fix), 260-paper gold, blanks excluded.
- **Paired comparison.** Every variant sees the identical 80 papers, so the
  per-paper differences are paired; use a paired test (Wilcoxon signed-rank over
  per-paper F1), not an unpaired one.
- **Pre-registered decision rule**, written before the run: promote a variant only
  if it beats baseline on **LCA F1** by more than the paired 95% CI excludes zero
  AND does not drop precision below 0.65. Recall bought at any precision is not a
  win — the KG's value is that its edges are trustworthy.
- **Report the null.** If no variant beats baseline, that is the result and the
  prompt stays as it is.

## Cost and guardrails

~88 min of compute plus ~15 min provisioning. The watchdog
(`~/.brev-watchdog/watchdog.sh`) is session-independent and enforces three
independent kill conditions — DONE marker, FAILED/silent for 20 consecutive
checks, and a hard deadline — and pulls results down every 60 s so partial output
survives. massedcompute has no working stop primitive, so the watchdog **deletes**
the instance; that is the only way to halt billing.

**GBNF hazard**: every grammar rule must be on ONE line. `LlamaGrammar.from_string()`
does not validate, so a multi-line `root` rule reports success and then segfaults
during sampling (rc=139, no traceback). Verify before shipping.
