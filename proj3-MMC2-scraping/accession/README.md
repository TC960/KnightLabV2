# Accession-code extraction pipeline

Python port of the retired Google Apps Script extractor. For ~13k microbiome papers, finds the
data **accession code(s)** (BioProject, SRA, GEO, EGA, dbGaP, …) and tags provenance
(`own` = deposited by the authors vs `reused`). Principle: **answer every row, flag ambiguity** —
nothing is silently dropped.

## Two design rules (both learned from the Apps Script version's bugs)

1. **One dictionary, never copied.** All repo patterns, guards, provenance and floor logic live in
   `dictionary.py`, imported by the pipeline *and* any eval. The old codebase had three divergent
   copies (one missing a repo pattern); that class of bug is now structurally impossible.
2. **Fetch once, extract infinitely.** `cache.py` is a content-addressed disk cache keyed by URL.
   The slow network phase runs once; extraction reads only from cache, so tweaking a regex re-runs
   over the whole corpus in seconds instead of re-fetching 13k papers.

## Run order

```
python3 -m accession.fetch_corpus         # Phase 1: resolve ids + fetch EPMC record/fulltext/annotations  (~50 min, once)
python3 -m accession.fetch_ncbi_fallback  # NCBI efetch for the EPMC-404 rows (preflight-gated)
python3 -m accession.extract              # Phase 2: cache-only extraction -> extracted.jsonl  (seconds, re-runnable)
python3 -m accession.provenance_pass      # offline: earliest-depositor + dbGaP dedup -> corrections
python3 -m accession.unpaywall_pass       # OA-outside-PMC recovery for the floor rows
python3 -m accession.writeback            # -> ../articles.out.csv  (never the live sheet)
```

## Modules

| module | role |
|---|---|
| `config.py` | all knobs (rate limits, GCA flag, emails). Nothing behavioural is a silent default. |
| `cache.py` | disk cache + per-host thread-safe rate limiters (EPMC 10/s, NCBI 3/s) |
| `sources.py` | thin clients: EPMC record/fulltext/annotations, NCBI efetch, Crossref, Unpaywall |
| `resolve.py` | DOI normalization (dirty DOIs → clean/None) + PMID/PMCID backfill via EPMC |
| `dictionary.py` | **single source of truth**: DICT, guards, OWN/REUSE, provenance, floor reasons |
| `extract.py` | cache-only extraction cascade: full text → annotations → abstract |
| `fetch_corpus.py` | concurrent resumable fetch (targets first) |
| `fetch_ncbi_fallback.py` | NCBI efetch for EPMC-404 rows; **preflight asserts HTTP 200 unauthenticated** |
| `provenance_pass.py` | offline corrections (no network) |
| `unpaywall_pass.py` | finds + fetches OA copies outside PMC, extracts from them |
| `writeback.py` | writes `articles.out.csv` under the never-overwrite policy |

## Channels and what each is worth (measured, not assumed)

| channel | what it recovers |
|---|---|
| EPMC full text (PMCID) | the strong channel — ~40% of PMCID rows yield a code |
| EPMC annotations (`MED:`/`PMC:`) | the only channel for pmid-only closed-access rows |
| NCBI efetch fallback | EPMC-404 rows; also mines non-OA front matter → `HUMAN_CAN_GET` |
| Crossref | weak — `relation[]` was empty on every probe; abstract-only fallback |
| Unpaywall | OA copies outside PMC (esp. repositories); reflags floor into `OA_AVAILABLE`/`PAYWALLED` |

## Flags in `articles.out.csv`

- `CLEAN` / `MULTI_CODE` / `MINED_NO_CONTEXT` — code(s) found
- `HUMAN_CAN_GET` — non-OA full text exists in PMC (restricted); a human can retrieve it
- `OA_AVAILABLE` — free full text exists **outside** PMC; the click-through URL is in `flag_detail`
- `PAYWALLED` — full text exists at publisher, retrievable with library access
- `SUPPLEMENT` / `ON_REQUEST` / `NO_DATA` / `NO_ACCESSION` — floor reasons

## Config notes

- `INCLUDE_GCA_IN_ACCESSION_COLUMN=False`: GCA/GCF genome assemblies are **captured** to
  `extracted.jsonl` (`repo=GCA`) but kept out of the accession column (they're reused reference
  genomes). Flip the flag to include them — no re-fetch needed.
- `NCBI_KEY=""`: the prod.gs key was invalid (HTTP 400). Deliberately not ported; runs unauthenticated.

Regenerable artifacts (`cache/`, `oa_cache/`, `*.jsonl`, logs) are gitignored.
