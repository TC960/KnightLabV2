# Handoff — embedding space, vector DB, contrast experiments

*Written 2026-08-31. Paste this into a fresh session in `/Users/mohak/Desktop/Lab Work`.
Read `CLAUDE.md` and `proj_2_attempt3/CLAUDE.md` first for project context.*

---

## The question this work serves

The KG has **226 contested edges** — taxon–disease pairs where some papers report
enrichment and others report depletion. Nobody knows why.

`analyze_contested.py` already tested the **26 study-design variables we have**
(country, sequencing, diet/medication control, …) against contested direction.
Nothing survived BH correction; best was `diet_controlled` at FDR 0.243, and
`country=China` split 45/44.

**Sam's proposal:** the explanatory variable may be one we never extracted
(recruitment setting, antibiotic washout, BMI…). Those facts are in the free
text. Embed the text and find what separates the camps.

**The user's framing, which is the right one:** this is not expected to *prove*
anything at n=303. It is a **hypothesis generator** — surface candidate variables
worth extracting properly, then confirm by extraction. Do not oversell results as
evidence.

---

## What was established this session

### 1. The variable-coverage survey was wrong, and is now corrected

A subagent surveyed 303 papers and reported 17 variables, `recruitment setting`
at **96.4%**. It counted **keyword hits anywhere in the text**. Its own top
example was `"Department of General Surgery, Shanghai Tenth People's Hospital"`
— an **author affiliation**. Another cited a **reference title**.

`audit_variable_coverage.py` re-measures at four tiers (keyword → outside
front-matter/refs → in an own-study sentence with no animal-model cue → with a
concrete value adjacent). **Every variable dropped 37–63 points.**

| variable | survey | audited (VALUED) |
|---|---:|---:|
| recruitment setting | 96.4% | **34.7%** |
| differential abundance method | 71.6% | 32.3% |
| disease severity | 51.5% | 25.7% |
| sample storage | 76.6% | 17.5% |
| antibiotic use | 65.7% | 15.5% |
| bmi | 63.0% | **7.3%** |
| dna extraction kit | 40.3% | 4.3% |
| probiotic use | 63.0% | **3.3%** |

`bmi` and `probiotic use` were the survey's #5 and #6 picks; they are the worst
two. And `differential abundance method` is **not new** — the correct datasheet
already has it hand-curated at **306/337**. Join to it, don't extract it.

VALUED is a lower bound (strict regex), NAIVE an upper bound. The ordering and
the size of the gap are the findings, not the exact percentages.

### 2. Predicting edge direction from a PAPER embedding is capped at 0.706

`ceiling_direction_probe.py`. This kills the literal version of the probe idea
with arithmetic, no GPU needed:

- 957 contested observations from 204 papers
- **132 of 204 papers (65%) report BOTH an enrichment and a depletion**
- 787 of 957 observations (82%) sit inside those papers
- majority-class floor **0.538**, paper-level-feature ceiling **0.706**

A paper reporting *Prevotella* up and *Lachnospiraceae* down contributes two rows
with an **identical feature vector and opposite labels**. Unseparable by any
model. The fix is the unit: the feature must be the **relation sentence** for a
specific (paper, taxon, disease), not the paper.

### 3. Semantic retrieval works — but the query must be prose

`chunk_search.py`. Querying a variable NAME fails; querying a SENTENCE works.

| query | top score | quality |
|---|---:|---|
| `hospital beds` | 0.347 | junk — editorial metadata, affiliations, consent boilerplate |
| `patients were recruited from the outpatient clinic` | 0.523 | 4/4 real |
| `participants had not taken antibiotics in the previous three months` | 0.707 | 4/4 real |

Nobody writes "hospital beds" in a methods section, so nearest-neighbour lands on
whatever else contains "hospital" — overwhelmingly affiliations.

**`rag_query.py` fixes this with HyDE**: an LLM writes N plausible passages, and
those are used as the queries. `hospital beds` went **0.347 → 0.699**.

### 4. Retrieval beats keywords, but only by 1–6×

`variable_sweep.py` scores retrieval against the independent regex labels. AUC
implementation **verified against sklearn to 1e-9** over 300 randomised trials
with ties. Analytic Mann-Whitney p and 2000-draw permutation p agree throughout.

| variable | AUC | P@10 | base | lift |
|---|---:|---:|---:|---:|
| differential abundance method | 0.838 | 0.80 | 0.32 | 2.5× |
| probiotic use | 0.748 | 0.20 | 0.03 | 6.1× |
| dna extraction kit | 0.668 | 0.20 | 0.04 | 4.7× |
| antibiotic use | 0.721 | 0.50 | 0.16 | 3.2× |
| bmi | 0.799 | 0.20 | 0.07 | 2.8× |
| disease severity | 0.719 | 0.50 | 0.26 | 1.9× |
| recruitment setting | 0.597 | 0.40 | 0.35 | 1.2× |
| sample storage | 0.695 | 0.20 | 0.17 | 1.1× |

**Read lift, not AUC.** Spot-checking `bmi` (AUC 0.799, 2nd best) found only 1 of
the top 4 chunks actually stated a BMI. AUC says positives outrank negatives *on
average*, which is compatible with a bad top-10 — and the top-10 is what a human
reads.

**Correction carried forward:** `recruitment setting` was called the best
surviving candidate on regex coverage. It has the **worst** AUC (0.597, lift
1.15×). Two independent methods both fail on it → the variable is genuinely
diffuse (hospital-vs-community has no distinctive vocabulary, unlike a named kit
or statistical test).

### 5. The disease direction has been deleted from the space — surgically

`nuisance_removal.py`. The contrast method (mean of up-papers − mean of
down-papers) otherwise just rediscovers *which disease the papers are about*.

Disease centroids → subspace → project every chunk onto the orthogonal
complement. **Nothing is trained.** The subspace is fitted on a 60% split of
papers and evaluated on held-out ones (fitting on all 303 then reporting disease
became unpredictable would be circular). 11 diseases ≥4 papers, rank 7, 96% of
between-disease variance.

| test | before | after |
|---|---:|---:|
| disease predictable (held-out, 11 classes, chance 0.275) | 0.793 | **0.532** |
| mean study-design AUC change | — | **+0.006** |
| bmi AUC | 0.799 | **0.821** |

Disease collapses, study-design signal survives, BMI *improves* — which is what
should happen when a dominant nuisance stops crowding a weaker axis. This is also
an anisotropy fix ("all-but-the-top"), which is likely why BMI rose.

**Caveat:** 0.532 is still above the 0.275 chance line. One projection removes
most of the disease signal, not all. Iterating (fit → project → refit on the
residual, i.e. INLP) would go further.

Outputs: `chunk_vecs_deconfounded.npy`, `disease_basis.npy` (both gitignored).
**Probes must be projected through the same basis** before comparison, or the
scores are meaningless. `nuisance_removal.py` does this; new code must too.

---

## Where things stand right now

### Files (all in `proj_2_attempt3/kg/`)

| file | what it does | state |
|---|---|---|
| `chunk_search.py` | build + query the chunk index | **rebuild in flight** |
| `rag_query.py` | HyDE: keyword → LLM passages → retrieve → optional grounded answer | works |
| `variable_sweep.py` | benchmark: AUC/lift per variable vs regex labels | works |
| `audit_variable_coverage.py` | 4-tier coverage audit, emits per-paper labels | works |
| `ceiling_direction_probe.py` | the 0.706 arithmetic | works |
| `nuisance_removal.py` | delete the disease subspace | works |
| `atlas_ingest.py` | push to MongoDB Atlas + `$vectorSearch` | **untested — cluster was empty** |
| `FINDINGS_variable_coverage.md` | survey + the audit that corrects it | done |

### The rebuild that is running

`chunk_search.py --build` was restarted with **symmetric overlap** (user's
request): 4 core sentences + 1 context sentence on each side, replacing the old
trailing-only overlap. Reason: a chunk opening on *"This was significantly higher
in the patient group (p = 0.01)"* is useless — the antecedent is in the
**preceding** sentence.

- 303 papers → **20,905 chunks**, mean 745 chars (was 19,483 / 829)
- **⚠ the 1000-char cap bound on 7,072 chunks (33.8%)** — a third lost a context
  sentence. MiniLM truncates at 256 tokens *silently*, so the cap is real
  protection, but 33.8% is high. **Consider `CORE_SENTS = 3`** so the context
  sentences survive more often. Worth an A/B on `variable_sweep.py`.

**IMPORTANT — verify before trusting any query.** `build()` writes
`chunks.jsonl` *before* embedding. An earlier interrupted run left 20,905 chunks
against 19,483 stale vectors, meaning chunk *N* in the file was not chunk *N* in
the array. `chunk_search.py` and `atlas_ingest.py` both now assert the counts
match. Check:

```bash
python3 -c "import numpy as np;print(sum(1 for _ in open('chunks.jsonl')), np.load('chunk_vecs.npy').shape)"
```

If they disagree, re-run `python3 chunk_search.py --build` (~12 min, CPU, local,
no network needed).

### MongoDB Atlas

- Cluster `test-08-31-26.zmmbmsq.mongodb.net` is live, free M0, **connection
  verified from Python** (pymongo 4.17.0 installed).
- `MONGODB_URI` is in `proj_2_attempt3/.env` (gitignored, untracked, verified).
  Code reads it via `os.environ`; it is never printed.
- **The cluster is EMPTY** — only `admin` and `local`. No `knightlab.chunks`, and
  **no vector index exists yet**.
- ⚠ **ROTATE THE DB PASSWORD.** It was pasted into a chat and onto a `mongosh`
  command line, so it is in `~/.zsh_history`. Then update `.env`.

**The index must be created after the collection exists.** pymongo 4.17 can do it
via `create_search_index`. Definition:

```json
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
    {"type": "filter", "path": "title"},
    {"type": "filter", "path": "disease"}
  ]
}
```

**384**, not 1536 — MiniLM's width. A dimension mismatch returns **empty results
at query time**, not an error at index time. The `filter` fields are the reason
Atlas earns its place: they allow vector search restricted to one disease, which
the flat numpy array can't do without a full rescan.

---

## Next steps, in order

### A. Finish the vector DB (30 min)

1. Confirm `chunks.jsonl` and `chunk_vecs.npy` counts match.
2. `python3 atlas_ingest.py --push` (full replace; safe to re-run).
3. Create the vector index (definition above), wait for `queryable: true`.
4. `python3 atlas_ingest.py --query "how were stool samples stored" --disease "Parkinson's disease (PD)"`
   — the filtered query is the thing worth proving.

### B. The contrast experiment — this is the actual payoff

For a contested edge, split papers into the **up-pile** and **down-pile**, then:

**Fast screen (user's idea, and it's the better first pass).** Compute
`d = mean(up) − mean(down)` on the **deconfounded** vectors, then score each of
~50 named probe sentences by `cos(probe, d)`. One number per probe, ranked. The
output is already in English because you wrote the probe names.

**Then test what survives.** Keep per-paper scores (paper score = **max** cosine
over its chunks — a paper states its storage protocol in *one* sentence out of
~69 chunks, so averaging drowns it). Shuffle pile labels **at the paper level**,
BH-correct across probes. Screening 50 probes over many edges will manufacture
false positives otherwise — this corpus has already produced two that survived
until tested.

**Build the probe bank first**: ~50 sentences, one named study-design concept
each, written in methods-section prose. Prefer concepts the audit shows are
actually stated (differential abundance method, disease severity, antibiotic use)
over ones it shows are not (probiotic use at 3.3%).

**Power warning, state it up front:** contested edges average ~4 papers per side.
Most per-edge tests cannot resolve anything. Prefer pooling across edges where
the design allows, and report power, not just p.

### C. Cross-encoder reranking — the highest-value retrieval upgrade

Residual noise in every query is author-contribution lists, IRB statements and
funding boilerplate eating top-k slots. A bi-encoder compares two independently
made vectors; a cross-encoder reads query and chunk **together** and is far more
accurate. Retrieve top-50, rerank, keep top-10.
`cross-encoder/ms-marco-MiniLM-L-6-v2` runs on CPU. **Measure it on
`variable_sweep.py`** — that benchmark exists precisely so this is a number, not
an opinion.

### D. Optional: embedding-model bake-off

`variable_sweep.py` is a real benchmark now, so "is a biomedical encoder better?"
is a 10-minute experiment. Use models trained with a **sentence/retrieval
objective** (S-PubMedBERT-MS-MARCO, NeuML/pubmedbert-base-embeddings, MedCPT).
**Do not** mean-pool raw PubMedBERT or BioBERT — they were trained on masked-token
prediction, never taught that two differently-worded sentences should sit close,
and typically *lose* to MiniLM here.

---

## Do not redo / do not repeat

- **Don't fine-tune the encoder on 303 papers.** It memorises 303 papers. If a
  learned metric is wanted, learn a 384×384 (or diagonal) projection and
  cross-validate **by paper** — orders of magnitude fewer parameters.
- **Cosine, dot product and L2 rank identically** on our unit-norm vectors
  (‖a−b‖² = 2 − 2cos). Swapping the metric buys nothing. Geometry (anisotropy)
  and reranking are the levers.
- **There is no decoder to remove.** Embedding models are encoder-only; the
  vector is the output.
- **Shuffle at the PAPER level**, always. Observations are not independent.
- **Don't trust a subagent's numbers without a script.** The variable survey
  shipped no code and its headline was 3× too high. Six-plus subagents have died
  mid-task in this project; check for partial output before relaunching.
- **Don't join on an external DB's taxid** (Disbiome records *Prevotella* as
  59823, a species, where the genus is 838). Both sides go through `taxonomy.py`.
- **No GPU** unless explicitly authorised — everything above is CPU/local.

## Cost note

Anthropic credits ~$200. HyDE expansion is a few hundred tokens per query and is
cached in `rag_cache/`. Nothing here is expensive. Atlas M0 is free and 20,905
chunks × 384 floats fits comfortably inside the 512 MB limit.
