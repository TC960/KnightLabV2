# GPU run plan — new gold standard (334 papers)

Prepared 2026-09-12. **Nothing here has been launched.** No GPU was provisioned, `brev`
was never invoked. Everything below is staged and dry-run locally; a human reviews, then
launches.

---

## TL;DR

| | |
|---|---|
| Gold papers | 334 |
| Have full text | **301** (296 already local + 5 newly fetched) |
| Staged for extraction (text **and** gold taxa) | **265** → `extract_input_gold.json` |
| Of those, already extracted by the same model+prompt | **260** |
| Genuinely never extracted | **5** |
| Existing extractions scored against the NEW gold | **P 0.747 / R 0.755 / F1 0.751** |

**Recommendation: do not re-extract the 260. Run the 5 new papers only — and run them on
DSMLP, not on a paid instance.** Reasoning in §3.

---

## 1. What is staged

| file | what it is |
|---|---|
| `gold_coverage.py` → `gold_coverage.json` | the DOI→title→corpus join; definitive have/missing lists |
| `fetch_gold_missing.py` → `gold_missing_papers.json` | the 5 papers recovered, in `new_papers.json` record shape |
| `gold_missing_report.json` | the 33 that could not be fetched, with a reason each |
| `build_extract_input_gold.py` → `extract_input_gold.json` | the 265-paper extraction set |
| `run_eval_gold.py` | the GPU-side runner (copy of `eval-v2/run_eval.py`, 3 deltas — see §5) |
| `rescore_vs_new_gold.py` → `rescore_vs_new_gold.json` | scores existing extractions against the new gold |
| `dryrun_gold_run.py` | 25 pre-flight checks, all passing |

Nothing existing was modified in place; `fetch_new_papers.py`, `run_eval.py` and both
watchdogs are untouched.

### Coverage

The gold CSV carries no titles and our corpora carry no DOIs, so the join runs
gold DOI → `Microbiota Signatures … Main Datasheet.csv` → corpora, matching on a
normalised title (lowercase, punctuation collapsed), with a DOI parsed out of the corpus
`link` field as a secondary key. All 334 gold DOIs are present in the datasheet, so the
join loses nothing.

```
334 gold papers
├── 301 have full text
│   ├── 247  all_usable_papers.json
│   ├──  49  new_papers.json
│   └──   5  gold_missing_papers.json   ← fetched in this task
│   └── of these, 36 have BLANK gold taxa → not extractable, dropped
│       = 265 staged
└──  33 no full text obtainable
```

### The 5 recovered papers

| DOI | source | chars |
|---|---|---|
| 10.1002/mds.26942 | PMC5469442 | 29,114 |
| 10.1016/j.cell.2022.08.021 | PMC10143502 | 102,055 |
| 10.1126/scitranslmed.abo2984 | PMC10680783 | 57,965 |
| 10.1177/1877718x251354931 | PMC13347513 | 17,428 |
| 10.3233/jpd-223500 | PMC9890728 | 31,267 |

All five are PMC **author manuscripts**. Europe PMC's `fullTextXML` returns 404 for these
— which is why `fetch_new_papers.py` gave up on them in August — while NCBI
`efetch(db=pmc)` serves them in full. `fetch_gold_missing.py` adds NCBI as a second
source behind EPMC at both steps (ID Converter for DOI→PMCID, efetch for the body).
That fallback was also load-bearing on the day: **the Europe PMC REST API was returning
nginx 503 on every request** while this ran (`ebi.ac.uk` root answered 200, so it was
their service, not our egress), and the EPMC-only path could not have run at all.

### The 33 that could not be fetched

| reason | n | meaning |
|---|---|---|
| `not_in_pmc` | 31 | subscription-only. Confirmed against **both** EPMC (`isOpenAccess:N, inPMC:N`) and the NCBI ID Converter (`Identifier not found in PMC`). No open route exists. |
| `no_oa_fulltext` | 1 | `pmid/31497202` → PMC6731442 exists, and efetch returns the record carrying `<!--The publisher of this article does not allow downloading of the full text in XML form.-->`. Readable as HTML on the PMC site; not in the OA bulk subset. |
| `too_short(1703)` | 1 | `10.1002/mds.26069` is in `MAIN_DATA.json` but as a 1,703-char stub (abstract only), below the 5,000-char threshold. |

Two gold rows carry `pmid/NNNNN` in the DOI column rather than a DOI; the fetcher
resolves those through EPMC `EXT_ID:` / the ID Converter's PMID form.

Getting the 31 would need institutional PDF access + a PDF→text step. That is a separate
piece of work and is **not** a blocker: they are 9% of the gold and their absence is
recorded per-DOI in `gold_missing_report.json`.

---

## 2. The result that changes the decision

`rescore_vs_new_gold.py` scores the extractions **we already have** against both
references. Same 260 papers, same extractor output, two different gold standards:

| reference | P | R | F1 |
|---|---|---|---|
| **new gold** (`high_confidence`) | **0.747** | **0.755** | **0.751** |
| old Main-Datasheet annotations | 0.358 | 0.711 | 0.476 |

Precision moves **+0.389** with no change to the model, the prompt, or the extractions.

This is the audit in the root `CLAUDE.md` landing. The new gold is 2.1× denser than the
sheet (10.5 vs 5.1 taxa/paper); 132 of the 265 papers had *blank* sheet taxa and now have
annotations; only 32% of the new gold traces back to a sheet entry. Where the two do
overlap they agree on direction **883 to 9**. So the new gold is a recall expansion of the
old, not a relabelling — and the extractor's long-standing "over-extraction" problem was
overwhelmingly the old reference being incomplete.

Metric caveat: this is the greedy char-ngram matcher (`run_eval.match_taxa`), not the
taxonomy-aware LCA metric, which needs the NCBI taxdump. LCA scores strictly higher, so
0.751 is a lower bound.

---

## 3. What to extract — recommendation

**Extract the 5 new papers. Do not re-run the 260.**

The brief's own criterion was that re-extracting the already-done papers is justified only
if the prompt changes. It does not need to change: §2 shows `samgated-v1` already sits at
F1 0.751 against the new gold, and its known weakness was an artefact of the old
reference. There is no hypothesis that a re-run would test.

**And 5 papers does not warrant a paid instance.** At the observed rate (3.33 s per 10k
chars) those 5 papers are **79 seconds** of GPU compute. The job would be ~99%
provisioning: ~15 min of apt + pip + a 16 GB model download. Per `proj_2_attempt3/CLAUDE.md`
the lab has free UCSD **DSMLP** GPUs and that is where eval-v2 ran. Run it there.
`run_eval_gold.py` is stack-agnostic — it is llama.cpp either way.

If a paid instance is used anyway, §4 has the commands and the guardrails are ready.

### If you disagree and want one clean pass over all 265

The honest case for it is provenance, not accuracy: the 260 come from three merge paths
(`all250` 250 + the `newset` GPU run 98 + corrections), at possibly mixed `n_ctx`, with one
recorded parse error. One pass over one input file collapses that. The cost is ~65 min of
extra GPU and a few dollars. Since temperature is 0, the output would be near-identical.
It is a defensible spend, just not one I would make.

### Wall-clock and cost

Measured from the 2026-08-31 run (98 papers, `gpu_results/extract_out/checkpoint.jsonl`):
mean **14.7 s/paper**, median 12.8, p90 23.7, max 43.9 — i.e. **3.33 s per 10k chars**.
The leaderboard's 22.5 s/paper came from the 15-paper testv2 set; the corpus-scale number
is lower. The 265-paper set totals 12.0M chars, mean 45,365/paper.

| scope | extraction | + provisioning | total wall-clock |
|---|---|---|---|
| 5 papers | ~1.5 min | ~15 min | **~17 min** |
| 265 papers (rate-scaled) | ~67 min | ~15 min | **~1 h 25 m** |
| 265 papers (at 22.5 s/paper, pessimistic) | ~99 min | ~15 min | **~1 h 55 m** |

Cost = (total wall-clock) × (instance hourly rate). **I cannot quote a dollar figure**: the
instance type for `kg-extract2` was not recorded anywhere in the repo, the log, or the
watchdog log, and I am not permitted to run `brev` to look it up. Set `MAX_HOURS` from the
table — **3** for the full 265, **1** for the 5 — and the deadline kill caps the spend at
rate × MAX_HOURS regardless of what goes wrong.

---

## 4. Launch procedure

Two terminals. Start the watchdog **first**, so no window exists where the instance is up
and unguarded.

### 4a. Terminal 1 — the guardrail, before anything else

```bash
# runs LOCALLY and independent of any Claude session; survives this session dying
~/.brev-watchdog/watchdog.sh <instance-name> 3      # 3 = MAX_HOURS; use 1 for the 5-paper run
tail -f ~/.brev-watchdog/<instance-name>.log
```

It is harmless to start before the instance exists — `brev ls` returns nothing and it
exits cleanly, so just restart it once the box is up. `kg/gpu_watchdog.sh` is a
byte-identical copy (verified); either path works.

### 4b. Terminal 2 — create the instance and ship the job

```bash
INST=kg-goldset
brev create "$INST" ...              # <- instance type is the human's call; see §3
brev shell "$INST"                   # wait until SSH answers

# ship the code + data (the model downloads on the box from HF)
scp "/Users/mohak/Desktop/Lab Work/proj_2_attempt3/kg/run_eval_gold.py" \
    "/Users/mohak/Desktop/Lab Work/proj_2_attempt3/kg/extract_input_gold.json" \
    "$INST:~/kg/repo/proj_2_attempt3/kg/"
```

Remote bootstrap — note the paths, the watchdog hard-codes
`REMOTE_DIR=~/kg/repo/proj_2_attempt3/kg`:

```bash
ssh "$INST" bash -s <<'REMOTE'
set -uo pipefail
cd ~/kg/repo/proj_2_attempt3/kg
python3 -m venv ~/kg/venv 2>/dev/null; . ~/kg/venv/bin/activate
pip install -q huggingface_hub
CMAKE_ARGS="-DGGML_CUDA=on" pip install -q llama-cpp-python

rm -f STATUS
mkdir -p extract_out
EVAL_OUT_DIR=$PWD/extract_out python run_eval_gold.py \
    --model qwopus3.5-27b-v3 --dataset goldset --n-ctx 32768 --resume \
    > ~/kg/run.log 2>&1
echo "rc=$?" >> ~/kg/run.log
REMOTE
```

To run only the 5 new papers, write a filtered copy locally — do **not** overwrite
`extract_input_gold.json`, it is the canonical 265-paper set:

```bash
python3 -c "import json; d=json.load(open('extract_input_gold.json')); \
n=[r for r in d if r['text_source']=='gold_missing_papers.json']; \
json.dump(n, open('extract_input_gold5.json','w')); print(len(n))"       # -> 5
```

then `scp extract_input_gold5.json` instead, and add it to `DATASETS` in
`run_eval_gold.py` (or point `goldset` at it on the box only).

### 4c. Results come down continuously, not at the end

The watchdog `scp`s `extract_out/` into `kg/gpu_results/` **on every 60 s tick**, before it
checks any stop condition, and again inside the kill path. `run_eval_gold.py` flushes
`checkpoint.jsonl` after every single paper. So a crash at paper 200 leaves 200 papers on
disk locally. Nothing depends on the job finishing.

### 4d. The three kill conditions

Both watchdog copies implement exactly these, any one of which is sufficient — no single
failure can leave the GPU billing:

1. **Job reported DONE** — `cat $REMOTE_DIR/STATUS` matches `DONE*` → final copy, delete.
2. **Job reported FAILED, or the box went silent** — `STATUS` matches `FAILED*`, **or**
   `ssh "echo alive"` fails on 20 consecutive checks (~20 min) → final copy, delete.
   Liveness is probed with `echo alive` and *not* by looking for `STATUS`, because
   `STATUS` does not exist during the ~15 min provisioning window and an earlier version
   of this script would have killed a healthy instance before the job ever started.
3. **Hard deadline** — `MAX_HOURS` elapsed → copy and delete regardless of state. This is
   the backstop that bounds the bill.

**Delete, not stop.** The provider has no working stop primitive, so delete is the only
way to halt billing. Note what the 2026-08-31 log actually shows: `brev stop` did *not*
report "does not support stop" — it printed `stop issued` and the instance was still
`RUNNING` 20 s later. Only the re-check-and-force-delete branch actually killed it. Do not
remove that branch; it is the line that saved the money, and the `grep -qi "does not
support stop"` test above it does not fire on this provider.

Manual override, if you ever need it: `brev delete <instance-name>`, then `brev ls` to
confirm it is gone. Do not trust `brev stop`.

---

## 5. Hazards

### GBNF grammar — every rule on ONE line

`LlamaGrammar.from_string()` does **not** validate. A `root` rule split across lines is
accepted, reports success, and then **segfaults during sampling — rc=139, no traceback**.

**Verified.** `dryrun_gold_run.py` check 1 parses `GRAMMAR_STR` as shipped in
`run_eval_gold.py`: 4 non-blank lines, 4 rules (`root`, `array`, `string`, `ws`), every
line contains exactly one `::=`, zero orphan continuation lines, `root` on a single line.
The check is a text check precisely because the library's own check is worthless. Re-run
`dryrun_gold_run.py` after **any** edit to the grammar.

### Scoring must not be able to kill a finished extraction

This is what actually went wrong last time. The 2026-08-31 run extracted all 98 papers
successfully and then died with `ModuleNotFoundError: No module named 'sklearn'` **in the
scoring step**. The watchdog read `FAILED`, correctly deleted the box — and the results
survived only because the checkpoint is flushed per paper. In `run_eval_gold.py`, scoring
is now opt-in (`--score`, off by default) and wrapped in try/except. Score locally off the
checkpoint; the laptop has sklearn 1.6.1.

Do not pass `--score` on the GPU unless you have also installed sklearn *and* run
`python taxonomy_match.py --setup` on the box.

### `run_eval_gold.py` — the three deltas from `eval-v2/run_eval.py`

Everything else (prompt, grammar, `smart_truncate`, checkpointing, the metric) is
byte-identical, so results stay comparable to the leaderboard.

1. `DATASETS` gains `goldset` → `kg/extract_input_gold.json`, and it is the default. The
   GPU copy last time had an equivalent `newset` entry that **was never committed** — the
   remote `run_eval.py` (339 lines) does not exist in this repo. This file replaces that
   untracked edit.
2. Scoring opt-in and non-fatal (above). A model **load** failure now exits 2 rather than
   returning, so it is recorded as FAILED, not DONE.
3. Writes `kg/STATUS` (`DONE` / `FAILED <exc>`) from Python, so a dead shell wrapper
   cannot leave the watchdog with no marker to read. A **zero** exit from argparse
   (`--help`) deliberately writes *nothing*: the watchdog deletes the instance on
   `DONE` **or** `FAILED`, so either marker would mean that typing
   `run_eval_gold.py --help` on the box kills the run. Verified both ways locally —
   `--help` leaves no STATUS; a real exception writes `FAILED <exc>`.

Also fixed: `_EMILY` pointed at `HERE/../../EmilySong_GoldStandardPaper`, correct from
`eval-v2/` but wrong from `kg/` where this copy lives. It only affects the `testv2` /
`all250` datasets, but it was silently broken on the GPU box last run.

### Context window

Launch with `--n-ctx 32768`, matching the previous corpus run. At that setting
`smart_truncate`'s budget is 123,472 chars and the longest staged paper is 105,172 — so
**zero of the 265 are truncated**. At the default 24,576 the budget drops to 90,704 and
papers would start getting cut. Do not leave it at the default.

(Minor: the reference-list strip fires on only 2 of 265, because most texts are
whitespace-collapsed by the XML stripper and no longer contain a `References\n` marker.
Irrelevant here since nothing truncates, but worth knowing before reusing the number.)

### Cache behaviour in the fetcher

`fetch_gold_missing.py` re-fetches cached `__ERR__` bodies **except** 404s, which are a
real and permanent "not in the OA subset" answer. One consequence to be aware of: the
EPMC search cache entries were written by `fetch_new_papers.py` at `pageSize=1`; the new
script requests `pageSize=5` but the cache key is the query string, so cached hits still
return the old single-result body. Resolution is unaffected because the NCBI ID Converter
runs as a second source. Delete `fetch_cache/search_*` to force a clean re-resolve.

---

## 6. Dry run — done

`python dryrun_gold_run.py` → **25/25 checks pass**. Covered: grammar single-line;
`extract_input_gold.json` shape, required keys, non-empty text, unique titles, every paper
has ≥1 gold taxon; `smart_truncate` at n_ctx 32768 (0 truncated); `PROMPT_TEMPLATE.format`
over all 265; `parse_output` on grammar-shaped output; checkpoint resume round-trip; both
watchdogs present, executable, identical, with all three kill conditions and the
incremental copy present, and `DEST` existing.

Also exercised locally: the full fetch (live network, 38 papers), the coverage join, the
extraction-set build, and the re-score of 260 papers.

**Not testable locally, and therefore the only untested step:** `llama_cpp` is not
installed on this laptop, so the grammar is verified as text but never compiled, and the
27B model is never loaded. Smoke-test it on the box before the full run:

```bash
EVAL_OUT_DIR=$PWD/extract_out python run_eval_gold.py \
    --model qwopus3.5-27b-v3 --dataset goldset --n-ctx 32768 --limit 2
```

Two papers, ~30 s after the model is resident. If that returns rc=0 with populated
`enr=`/`dep=` lines, the grammar compiled and sampled correctly and the full run is safe.

---

## 7. After the run

```bash
# results land in kg/gpu_results/extract_out/ continuously, via the watchdog
python rescore_vs_new_gold.py        # once the new rows are merged in
# then the existing path: merge_new_extractions.py -> build_kg.py -> build_viz.py
```

`merge_new_extractions.py` keys on title and marks provenance; it currently points at
`new_papers.json` for the gold flag and will need `extract_input_gold.json` added before
it classifies the new rows correctly.
