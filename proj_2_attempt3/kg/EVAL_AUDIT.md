# EVAL_AUDIT — adversarial audit of `score_lca.py` / `score_newgold.py`

Auditor: independent agent. Everything below was re-derived with code written for this audit
(`scratchpad/audit*.py`), not by re-running the scripts under test. Shared infrastructure that
I did NOT rewrite: `taxonomy.py` (NCBI name→taxid resolution) and sklearn's TF-IDF — but the
lineage walk, the tokenizer, the matchers, the counting, the permutation and the bootstrap are
all my own. Audit scripts live in this session's scratchpad
(`audit1.py` reconciliation, `audit2.py` scoring/matching, `audit3.py` double-counting and rank
gaps, `audit4.py` deltas/leaderboard/leakage, `audit5.py` flips and circularity, `audit6.py`
match quality, `audit_perm.py` permutation); they are throwaway and not added to the repo, but
every number below is reproducible from them. Status: **COMPLETE.**

Files under test: `score_lca.py`, `run_lca_eval.py` (the driver — note `score_lca.py` is a
library with no `main()`; the claims actually come from `run_lca_eval.py` and `lca_checks.py`),
`score_newgold.py`; outputs `results_lca.json`, `lca_checks.json`, `score_newgold.json`.

---

**Bottom line up front.** Every number you asked me to check is arithmetically correct — I
reproduced all of them to 4 decimal places with independent code, including the ones you
suspected (the 36 vs 40 blanks is not a bug; both numbers are right about different sets). But
the **matcher itself is broken in a way that inflates every score in the project**: a gold taxon
can be claimed by several predictions at once, each scoring a true positive. Fixing that with
optimal bipartite matching takes char F1 0.7549 -> 0.7346 and LCA F1 0.7801 -> **0.7403**, which
collapses the headline "taxonomy-aware metric gains +0.025" to **+0.006**. The model ranking
survives; the metric's absolute numbers do not. Separately, the new gold's provenance cannot be
verified from this repo and 67% of it is verbatim our own model's output — that is a bigger
threat to the story than anything in the code.

## Claim table

| # | claim | reported | my value | verdict |
|---|---|---|---|---|
| 1 | char-ngram, 260 papers | P 0.7491 R 0.7608 F1 0.7549 | P 0.7491 R 0.7608 F1 0.7549 (TP 2156 / FP 722 / FN 678) | **CONFIRMED** as arithmetic — but see A/B: 0.7346 under sound matching |
| 1 | LCA, 260 papers | P 0.7880 R 0.7722 F1 0.7801 | P 0.7880 R 0.7722 F1 0.7801 (TP 2268 / FP 610 / FN 669) | **CONFIRMED** as arithmetic — 0.7403 under sound matching |
| 2 | permutation of LCA F1 | obs 0.7801, null mean 0.2076, p 0.001, 1000 draws | obs 0.7801, null mean **0.2069**, null sd 0.0104, null max 0.2443, 0/1000 >= obs, p 0.001 (own seed) | **CONFIRMED** (but the test is near-vacuous — see E) |
| 3 | 334 gold DOIs / 296 joined / 38 no-extraction / 36 blank / 260 scoreable | as stated | 334 / 296 / 38 / 36 / 260 | **CONFIRMED** |
| 3b | blank count 36 vs your 40 | file says 36 | **both right**: 40 blank in the full gold; 4 of them had no extraction and were dropped by the earlier filter, leaving 36 among the 296 joined | **CONFIRMED (no bug)** |
| 4 | LCA forgives 112; delta_tp +112, delta_fp −112, delta_fn −9 | as stated | +112 / −112 / −9, 0 superset violations | **CONFIRMED** — and the asymmetry is coherent, see below |
| 5 | testv2 re-rank: 0.807 / 0.742 / 0.739 / 0.718 / 0.593 | as stated | 0.8072 / 0.7423 / 0.7391 / 0.7182 / 0.5926, same order | **CONFIRMED** |
| 5 | bootstrap P(qwopus3.5 rank 1) = 0.8355 | as stated | 0.837 (B=2000, my own resampler and seed) | **CONFIRMED** |
| 6 | 19 direction flips, 8 rank mismatches | as stated | 19 flips, 8 rank mismatches, residual FP 695 / FN 653, 1402 charges | **CONFIRMED** |
| C | `min_rank_sweep` in results_lca.json | none .7801 / phylum .7796 / class .7684 / order .7655 / family .7628 / genus .7558 | identical to 4 dp | **CONFIRMED** |
| A | greedy vs optimal matching | not reported anywhere | char .7549 -> **.7346**; LCA .7801 -> **.7403**; LCA−char +.0252 -> **+.0057** | **WRONG / overstated** |
| C | rank gap unbounded | `--min-lca-rank` exists but defaults to off | 59 of 112 LCA matches involve a phylum-or-higher node; capping the gap at 1 rank gives F1 .7635 | **WRONG / overstated** |
| G2 | new gold is an independent reference | implied throughout | 67.0% of gold taxa are verbatim our own predictions; no build script exists in-repo | **CANNOT-VERIFY** |

### Claim 4 — why delta_fp and delta_tp are symmetric but delta_fn is not

It is coherent, and it is the fingerprint of the bug in B. TP + FP is fixed at the number of
predictions (every prediction is charged exactly once), so any rescue must move one prediction
from the FP column to the TP column: delta_tp = −delta_fp = 112 is forced by construction, not
evidence of anything. FN is a count of *gold* taxa left unclaimed, and it only falls when the
rescued prediction claims a gold taxon that nothing else had claimed. That happened **9 times out
of 112**. The other **103 rescues landed on a gold taxon another prediction had already matched**
— which my independent multiplicity count confirms exactly (204 duplicate LCA claims − 101
duplicate char claims = 103). So the comment in `run_lca_eval.py` ("do not assume 2 charges per
pair") is right, but the honest reading of the asymmetry is not "this is a precision correction",
it is "92% of what LCA forgives is redundant credit for a gold taxon we had already matched."

### Claim 3 detail — the 36 vs 40 discrepancy

The gold CSV is 668 rows = 334 DOIs x exactly 2 rows (Enriched/Depleted), no duplicate
(DOI, field) pairs, so the loader's last-write-wins dict cannot lose anything.

- **40** DOIs have both taxa cells blank. Identical under raw-string-empty and
  post-tokenizer definitions (0 cells contain text that tokenizes to nothing), so the
  tokenizer is not involved in the discrepancy.
- 4 of those 40 never joined to an extraction and were removed by the earlier
  `no_extraction` filter: `10.1016/j.eplepsyres.2018.09.013`,
  `10.1016/j.neurobiolaging.2017.09.023`, `10.1016/j.neuropharm.2023.109566`,
  `10.1097/htr.0000000000000615`. All four ARE in the datasheet; they fail on the
  title→extraction leg.
- So: 334 − 38 (no extraction) = 296 joined; of those 36 are blank; 296 − 36 = 260.

Your earlier "40" measured blanks in the whole gold; the file's "36" measures blanks among
the joined subset. Both are correct, the filter order just makes them different quantities.
The pipeline is not double-counting or losing papers. One cosmetic note: the exclusion in
`score_newgold.py` is done by **Title** (`scoreable = [m for m in matched if m[1]["Title"]
not in blank_titles]`) rather than by DOI — I checked, it gives the same 260 here, but it is
fragile: two DOIs sharing a title would drop a scoreable paper. `run_lca_eval.py` does it by
DOI and is fine.

---

## A. Is the matching greedy in a way that inflates scores? — **YES. This is the headline finding.**

Both scorers do one pass over predictions; each prediction takes its argmax gold and, if that
misses the 0.5 threshold, the **first** gold in document order whose lineage nests. A gold taxon
that has already been claimed is *not* removed from the pool, so several predictions can each
score a TP against the same gold cell.

Re-scored with maximum bipartite matching (`scipy.optimize.linear_sum_assignment`, edge allowed
iff char-cosine >= 0.5 **or** lineages nested — the same admissibility rule, only the assignment
changed):

| matcher | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| char, their greedy | 0.7491 | 0.7608 | **0.7549** | 2156 | 722 | 678 |
| char, optimal 1-to-1 | 0.7161 | 0.7541 | **0.7346** | 2061 | 817 | 672 |
| LCA, their greedy | 0.7880 | 0.7722 | **0.7801** | 2268 | 610 | 669 |
| LCA, optimal 1-to-1 | 0.7217 | 0.7600 | **0.7403** | 2077 | 801 | 656 |

(a greedy-but-one-to-one variant lands in the same place: char .7339, LCA .7389 — so this is not
an artefact of the assignment algorithm, it is the one-to-many licence itself.)

**Greedy inflates char F1 by +0.0203 and LCA F1 by +0.0398.** The damage is concentrated in
precision (LCA precision 0.7880 -> 0.7217, −0.066), which is exactly what you would expect: the
free extra TPs come with no extra FP.

The consequence that matters for the story being told about LCA:

> reported LCA − char = **+0.0252**.
> Under honest 1-to-1 matching, LCA − char = 0.7403 − 0.7346 = **+0.0057**.

**About four fifths of the "taxonomy-aware metric lifts us 0.755 -> 0.780" effect is
double-counting, not taxonomy.** The headline "LCA F1 0.7801" should not be quoted without this
caveat. Note this is an inherited defect: `eval-v2/run_eval.match_taxa` (the char metric behind
every number in `../CLAUDE.md`, including the .642 -> .751 claim) has the same one-to-many bug.

## B. Does LCA double-count many-to-one? — **YES, demonstrated directly.**

Constructed cases, run through the scorer under test:

| predicted | gold | their greedy (TP,FP,FN) | optimal (TP,FP,FN) |
|---|---|---|---|
| `blautia`, `lachnospiraceae` | `blautia` | **(2, 0, 0)** | (1, 1, 0) |
| `blautia`, `roseburia`, `dorea` | `lachnospiraceae` | **(3, 0, 0)** | (1, 2, 0) |
| `firmicutes`, `bacteroidetes` | `bacteria` | **(2, 0, 0)** | (1, 1, 0) |
| `blautia`, `blautia wexlerae` | `blautia` | **(2, 0, 0)** | (1, 1, 0) |
| `akkermansia`, `akkermansia muciniphila` | `akkermansia muciniphila` | **(2, 0, 0)** (char rule alone!) | (1, 1, 0) |

So the exact case you asked about — predict both `Blautia` and `Lachnospiraceae`, gold has only
`Blautia` — scores **two TPs and zero FPs**, not one TP and one FP. TP exceeds the number of gold
taxa in the cell, which is arithmetically impossible for a real matching.

This is not hypothetical on the real data:

- corpus totals: **2733 gold taxa**, 2878 predicted taxa.
- greedy char: TP+FN = 2834 -> **101 phantom gold slots** (gold taxa credited more than once).
- greedy LCA: TP+FN = 2937 -> **204 phantom gold slots**.
- multiplicity histogram, LCA rule: 101 gold taxa claimed twice, 25 three times, 6 four times,
  4 five times, 2 seven times, 1 eight times.
- real example: gold `enterobacteriales` is claimed simultaneously by predictions
  `proteobacteria`, `gammaproteobacteria`, `enterobacteriales`, `enterobacteriaceae` — one whole
  nested chain, four TPs for one gold taxon. Gold `lachnospiraceae` is claimed by
  `lachnospira`/`roseburia`/`blautia` at once.

**Of the 112 pairs the report calls "forgiven", 204 − 101 = 103 of the extra LCA credits are
duplicate claims on an already-matched gold taxon.** That is why delta_fn is only −9 (see claim 4).

## C. Is the rank gap bounded? — **No, it is unbounded, but empirically the damage is modest.**

There is a `--min-lca-rank` guard, but the default (and every number reported) uses `None`, i.e.
nesting at any depth. Nothing prevents `Bacteria` from matching everything; it simply did not
happen much here.

Profile of the shallower member of each of the 112 LCA-only matches (my count, independent):

`phylum 57, family 31, order 11, class 10, kingdom 2, genus 1` — rank-gap histogram
`{0:2, 1:34, 2:14, 3:30, 4:28, 5:3, 6:1}`. This reproduces `results_lca.json`'s `rank_gap_hist`
exactly.

- **59 of 112 matches (53%) have a phylum-or-above taxon on one side**, e.g. `dorea (genus) ~
  firmicutes (phylum)`, `proteobacteria (phylum) ~ acetobacter (genus)`, `firmicutes (phylum) ~
  faecalibacterium prausnitzii (species)` (gap 5). Calling a phylum-level prediction a correct
  hit for a species-level gold is exactly the degenerate case you were worried about; it is
  happening 59 times.
- 2 matches involve a kingdom-level node. No literal `Bacteria`-matches-everything event.

F1 if nesting is capped at a maximum rank gap (my own cap, applied on top of their greedy rule):

| cap | P | R | F1 | vs char (.7549) |
|---|---|---|---|---|
| gap <= 1 | 0.7627 | 0.7643 | 0.7635 | +0.0086 |
| gap <= 2 | 0.7679 | 0.7666 | 0.7672 | +0.0123 |
| gap <= 3 | 0.7794 | 0.7697 | 0.7745 | +0.0196 |
| uncapped (reported) | 0.7880 | 0.7722 | 0.7801 | +0.0252 |

So **two thirds of the LCA lift comes from matches spanning more than one taxonomic rank**, and
a third comes from gaps of 4+. Combine with A: a defensible metric (1-to-1 matching, gap <= 1)
would put LCA essentially on top of char.

The `min_rank_sweep` already in `results_lca.json` reproduces exactly under my code
(none .7801 / phylum .7796 / class .7684 / order .7655 / family .7628 / genus .7558) — that key
is correct. But note it answers a different question than yours: it rejects matches whose
*shallower member* is above a rank, not matches whose *gap* is large.

## D. Blank / empty handling — **CORRECT.**

- The 36 papers with both gold cells blank are excluded before scoring (verified independently:
  260 = 296 − 36, and the FP total does not contain their predictions). The "F1 0.390" failure
  mode is not present.
- Empty **prediction** vs non-empty gold IS charged: `match([], ['blautia','dorea'])` returns
  (0, 0, 2). On the real data, **175 FN charges come from a direction where the model predicted
  nothing**; 16 of the 260 scoreable papers have empty predictions in both directions and are
  fully charged. Not skipped.
- Non-empty prediction vs an empty gold *cell* (paper non-blank overall, but one direction blank)
  is charged as FP: 47 such FP charges, from 21 papers with an empty gold Enriched cell and 20
  with an empty gold Depleted cell. This is a defensible choice but worth stating: ~6.5% of all
  FP charges come from one-sided gold cells, where the annotator may simply not have recorded a
  direction rather than asserting it is empty.

## E. Is the permutation null constructed correctly? — **Yes, but it is a near-worthless test.**

Mechanics check (`run_lca_eval.py` L133-154): the shuffle permutes the **gold** list index while
predictions stay in place, both directions of a paper move together (so within-paper structure is
preserved, correct), and the null statistic is computed with the *same* `lca` matcher as the
observed statistic (`f1_only(..., lca)`) — that is right, and it is the thing most often got
wrong. `rng = random.Random(0)` makes it reproducible. A `random.shuffle` is not a derangement,
so ~1 of 260 papers keeps its own gold per draw (measured: 1.0 per draw in my run); that biases
the null *upward* by ~0.3% of one paper's contribution, i.e. it is conservative and irrelevant.

I re-ran the null with my own code and a different seed (see table). The numbers reproduce.

What the test does **not** support: the p-value answers "is our extraction better than pairing
each paper with a random other paper's gold?" The null mean is 0.2076 because microbiome papers
share a small vocabulary (`Bacteroides`, `Firmicutes`, `Prevotella` recur everywhere) — under LCA
that floor is **twice** the char floor (0.0967), which is itself a warning about how much credit
the LCA rule hands out for free. Clearing a 0.21 floor at p=0.001 is not evidence that 0.78 is a
good score; it is evidence the pipeline is not shuffled. Do not present p=0.001 as validation of
extraction quality.

## F. testv2 leakage — **Real, quantified, and it does not matter numerically.**

- All **15/15** testv2 papers are in the new gold, are non-blank, and are **among the 260 corpus
  papers scored in claim 1**. `results_lca.json`'s `testv2_overlap` block reports the first three
  facts but never states the fourth, which is the one that bears on independence.
- Size of the contamination: 15/260 = **5.8%** of papers.
- For 11 of the 15, the corpus prediction is byte-identical to the benchmark prediction; 4 differ
  (the corpus run is a separate generation).
- Dropping the 15: corpus LCA F1 **0.7790** on n=245 vs 0.7801 on n=260. Immaterial.

So the overlap is real but the corpus number is not meaningfully propped up by it. The
*conceptual* dependence is larger than the numeric one: qwopus3.5 was selected as the corpus
extractor **on testv2**, so the corpus score carries that selection, and the n=15 re-ranking is
not an out-of-sample confirmation of the corpus number. State them as one result, not two.

## G. Other findings

### G1. "LCA beats char" is a tautology, and the code says so — but the write-ups don't

`score_lca.py`'s own docstring is honest: rule (b) fires only after (a) fails, so LCA is a strict
superset of char and "any F1 delta is the metric forgiving, never punishing." I verified this
holds on every one of the 520 scored cells (0 superset violations). The consequence should be
stated wherever the +0.0252 is quoted: **the sign of the delta carries no information.** The only
question worth asking about LCA is whether the forgiveness is *justified*, and sections A–C say a
large part of it is not (103 of the 112 rescues land on a gold taxon that was already matched;
59 of 112 span a phylum-or-higher node).

### G2. The new gold's provenance cannot be verified from this repo — and 67% of it is verbatim our own output

`high_confidence - final_constrained_override.csv` lives only in `~/Downloads`, is not tracked in
git, and **no script in this repo produces it**. The filename ("constrained override") implies an
automated step. Measured overlap on the 260 scoreable papers:

| | count | share of the 2733 gold taxa |
|---|---:|---:|
| gold taxon string **identical** to one of our predictions for that paper+direction | 1831 | **67.0%** |
| gold taxon string identical to an OLD datasheet gold entry | 1363 | 49.9% |
| present in our prediction but absent from the old gold | 764 | **28.0%** |
| present in the old gold but absent from our prediction | 296 | 10.8% |

67% verbatim agreement with the system under test is not proof of circularity — a curator reading
the same paper copies the same names the extractor copied. But it is **consistent** with
circularity, and 764 taxa that the new gold added relative to the old are taxa we predicted.
Since the entire "the extractor was never as bad as we reported" result (F1 .639 -> .758) is
driven by that reference change, this is the single largest unresolved threat to the headline —
larger than anything in the scoring code. **Verdict: CANNOT-VERIFY.** What would settle it: the
build script/provenance for the CSV, and whether any model output was an input to it. If it was,
`FINDINGS_newgold.md` §1 needs a retraction, not a caveat.

(For what it's worth, the direction of the old->new change survives the matching fix: under
optimal 1-to-1 matching the like-for-like on 253 papers is OLD 0.6020 -> NEW 0.7378, +0.136, vs
+0.119 greedy. So *if* the gold is legitimate, that conclusion is robust.)

### G3. The greedy inflation is model-dependent, which makes it a leaderboard bias

Re-scoring the 5 testv2 models with optimal 1-to-1 matching under the same new-gold + LCA rule:

| model | reported (greedy) | optimal 1-to-1 | inflation |
|---|---:|---:|---:|
| qwopus3.5-27b-v3 | 0.8072 | 0.7222 | +0.085 |
| qwythos-9b | 0.7423 | 0.6497 | +0.093 |
| qwen3.6-35b-a3b | 0.7391 | 0.6139 | **+0.125** |
| qwopus3.6-35b-a3b-mtp | 0.7182 | 0.5963 | **+0.122** |
| qwen2.5-32b-instruct | 0.5926 | 0.5773 | +0.015 |

Models that dump whole nested chains (`proteobacteria`, `gammaproteobacteria`,
`enterobacteriaceae`, `serratia` for one gold taxon) are rewarded ~8x more than the conservative
under-extractor. **The rank order happens to be unchanged**, and qwopus3.5's lead actually widens
(P(rank 1) rises from 0.837 to 0.934 in my bootstrap), so the model-selection decision is safe.
But the *gaps* between models are not interpretable as reported, and a future model that
over-extracts nested chains could top this leaderboard purely on the bug.

### G4. Things I checked that are fine — no disagreement manufactured

- The char metric's true positives really are the same organism: of 2156 char TPs, **89.9%**
  resolve to the identical NCBI taxid, 3.6% to a nested pair, 5.5% have an unresolvable side, and
  only **22 (1.0%)** pair genuinely different non-nested taxa (`parabacteroides`~`bacteroides`,
  `coriobacteriaceae`~`enterobacteriaceae`, `prevotella 9`~`paraprevotella`). The 0.5 cosine
  threshold is not, as one might fear, matching on spelling alone.
- TF-IDF is re-fit per cell, so a similarity depends on which other taxa are in the same cell
  (`blautia wexlerae`~`blautia` scores 0.554 alone, 0.593 in a 3x3 cell). Fragile in principle;
  only **50 of 2831** decisions (1.8%) sit within +/-0.05 of the threshold, so it changes almost
  nothing. Noted, not a finding.
- `run_lca_eval.py` correctly restricts the OLD-gold leaderboard scoring to the same 15 papers
  used for the NEW gold (`old_same`), so that comparison is not a subset artefact.
- `lca_checks.py`'s superset check and its independent recount of the 112 are genuinely a second
  route to the same number, and they agree with mine.
- Results are flushed after every stage; the partial-crash story is real.

### G5. Smaller notes

- `score_lca.py` has no `main()`. Every claim attributed to it is actually produced by
  `run_lca_eval.py` (results_lca.json) or `lca_checks.py` (lca_checks.json). Cite those.
- `score_newgold.py`'s module docstring still says the taxonomy-aware metric "needs linux-only
  binaries and is not used for the headline score." `score_lca.py` obsoleted that; the two files
  now disagree in their own comments about what the headline metric is.
- `score_newgold.py` excludes blank papers by **Title**, `run_lca_eval.py` by DOI. Same answer
  today (260), but the title path breaks silently on a title collision.
- In the `min_rank` guard, a taxon whose NCBI rank is absent from `RANK_DEPTH` (e.g. `no rank`,
  `clade`) gets depth 99 = "maximally specific", so the guard never rejects it. Conservative in
  the wrong direction, but only 1 of 112 rescues has an unranked member.
- The error-analysis denominator (1402 charges) inherits the many-to-one bug: under 1-to-1
  matching the charge total is 1489, so the "direction flip = 2.7% of all error" share is
  slightly optimistic (it becomes 2.6%). Immaterial to that conclusion — direction really is
  nearly solved.

---

## What I would change, in priority order

1. **Make the matcher 1-to-1.** One line of the fix is already written above: build the
   admissibility matrix (cosine >= 0.5 OR nested), then
   `scipy.optimize.linear_sum_assignment`. Everything else in the harness can stay. Every F1 in
   this project — including the `.642 -> .751` and `77.5% / 75.6%` lineage in the CLAUDE.md files,
   which use the same `run_eval.match_taxa` — is inflated until this lands. Re-run, then restate.
2. **Bound the rank gap**, or stop calling a phylum-vs-species match a hit. `gap <= 1` is the
   defensible default; report the sweep alongside.
3. **Get the provenance of `high_confidence - final_constrained_override.csv`** and put the file
   (or its builder) under version control. If model output fed into it, `FINDINGS_newgold.md` §1
   is circular and must be retracted rather than caveated.
4. Stop quoting the permutation p-value as evidence of quality; quote the null mean (0.21 under
   LCA) as evidence of how much the metric gives away for free.
5. Say explicitly that the corpus number and the n=15 benchmark are the same experiment: the 15
   testv2 papers are inside the 260, and qwopus3.5 was chosen on them.
