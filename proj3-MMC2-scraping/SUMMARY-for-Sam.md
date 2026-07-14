# Accession extraction — summary

## The finding that reframes the project

**Full-text availability is not the bottleneck. Accession presence is.**

We can now reach the full text for the large majority of the papers we previously wrote off as
"no full text." Of the ~3,300 unanswered rows with a DOI, **~55% have a free open-access copy
sitting outside PMC** (publisher gold/hybrid, institutional repositories, bioRxiv), and nearly all
of the remainder are paywalled papers a human with library access opens in seconds. Almost none are
"the paper doesn't exist / has no full text anywhere."

**And yet — when we fetched those complete papers and parsed them, only ~4% contained an accession
code.** Even with the entire paper in hand, most of these studies simply don't report a repository
accession: the data is in-article, "available on request," or not shared at all.

So the ceiling on this task is not our pipeline's reach — it's a property of the literature. A large
share of microbiome papers, especially in clinical/subscription journals, do not deposit sequence
data under a citable accession. That's a result about data-sharing practice in the field, and it's
the kind of thing that belongs in a discussion section, not just a spreadsheet cell.

## What the pipeline delivered

Deliverable: **`articles.out.csv`** (same schema as the source sheet; the live sheet is never
touched, and no human-entered code is ever overwritten — verified 0 changed).

| | before | after |
|---|---|---|
| rows with a data accession code | 4,405 | **5,092 (+687)** |

The 4,141 previously-blank target rows now break down as:

| status | rows | meaning |
|---|---|---|
| **coded** | **687** | 649 from PMC/EPMC + NCBI full text, 38 from OA copies outside PMC |
| `OA_AVAILABLE` | 1,796 | free full text exists outside PMC — **click-through URL is in the row**; no accession was in it |
| `PAYWALLED` | 1,390 | full text at publisher, retrievable with library access |
| `HUMAN_CAN_GET` | 114 | restricted PMC copy exists; human-retrievable |
| residual floor | 154 | genuinely nothing to fetch (no DOI / no record) |

The old "3,378 dead rows" is now a **triaged curator worklist**: 3,300 of them are retrievable,
most with a URL a curator can click straight through. Only 154 are true dead ends.

## What changed vs the old Apps Script version

- **Runtime:** ~5 hours across multiple days → **~1 hour once**, then seconds to re-run extraction
  after any rule change (fetch and extract are separated; every response is cached to disk).
- **New recovery channels the old version never used:** NCBI full-text fallback for papers EPMC
  lacks (closed 55 of 108 known gaps), and Unpaywall for open-access copies outside PMC.
- **Correctness fixes to known biases:** 66 provenance corrections (an accession can only be
  *deposited* by the earliest paper that cites it — later papers are reusing; concentrated in
  EGA/dbGaP exactly as expected) and 9 dbGaP version-collapses. All auditable.
- We deliberately do **not** reproduce the old version's mistakes — e.g. it mis-read grant IDs
  (`JUSRP51710A`) as SRA accessions; ours correctly rejects them.

## Assumptions in the old handoff doc that the data disproved

Each was tested against real papers, not assumed:

- "DOI-only rows are dead" → **false**: 86% resolve to a PMID, 68% to full text via EPMC.
- "Crossref is the obvious next lever" → **false**: its dataset-link field was empty on every probe.
- "~600–800 codes are sitting in supplementary files" → **false**: measured hit rate ~2–5%
  (~50 codes at most); the accessions aren't in the supplements, the *results* are.

## Honest limits

- The pmid-only bucket (56% of the target rows) is closed-access with no PMC full text; even when
  opened via other channels it rarely contains an accession. This is the bulk of the residual.
- The 1,796 `OA_AVAILABLE` and 1,390 `PAYWALLED` rows are where remaining accessions live, but they
  require a human (library access + judgement reading a data-availability statement). That's a
  curation task, not an extraction one.
