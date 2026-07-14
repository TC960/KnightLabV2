# Accession extraction — summary

## The finding that reframes the project

**When we can read a paper in full, a slight majority already contain an accession. The papers that
stay unanswered are the ones where reading doesn't help — and increasingly that's because they never
deposited data, not because we can't reach them.**

Two measurements together:

1. **Availability is largely solvable.** Of the ~3,300 unanswered rows with a DOI, **~55% have a
   free open-access copy outside PMC** (publisher gold/hybrid, institutional repositories, bioRxiv),
   and nearly all the rest are paywalled papers a human with library access opens in seconds. Almost
   none are "the paper doesn't exist / has no full text anywhere."

2. **But reaching the full text doesn't rescue the residual.** Across every channel, **55.5% of the
   papers we fully parsed contain an accession** — yet that rate collapses as we move down the
   channels toward the harder, still-unanswered papers:

   | channel (population it handles) | parsed full texts | contain an accession |
   |---|---|---|
   | EPMC (all PMC papers) | 7,781 | **58.7%** |
   | NCBI fallback (papers EPMC lacks) | 647 | 34.3% |
   | Unpaywall, first read (OA fallback for the floor) | 264 | **14.0%** |

The gradient 59% → 34% → 14% is not the open-access copies being worse (a repository PDF is the same
text as the publisher's) — it's **selection**. By the time a paper reaches the OA fallback it's one
no easier channel could answer, and those are disproportionately clinical/subscription studies that
never deposited sequence data under a citable accession (data in-article, "on request," or not
shared). So the ceiling on the remaining rows is two compounding things: some have no open copy to
machine-read (a curation task — open the paywalled ones), and the ones we *can* read mostly just
don't contain an accession (a fact about data-sharing practice in the field). The second is the
discussion-section point.

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
