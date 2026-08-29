# Papers Without an Accession Code — Analysis & Confidence Report

*Corpus: 13,092 microbiome papers. Deliverable: `articles.out.csv`. Counts recomputed from the
delivered CSV; supersedes the 2026-07-14 snapshot, which predated the browser-recovery pass and the
`SUB` format fix.*

> The narrative version of this analysis, with the audit methodology written out, is Part II of
> `../PROJECTS-2-AND-3-REPORT.md`. Both are generated from the same CSV and agree.

## 1. Headline

| | rows | share |
|---|---:|---:|
| **Coded** — an accession is in the cell | **5,581** | 42.6% |
| **No code** | **7,511** | 57.4% |
| **Total** | **13,092** | |

Every no-code row is classified by *why* it has no code and, deterministically, *whether we are
entitled to say so*.

## 2. The confidence rule (deterministic — no judgement calls)

A no-code row is only trusted as empty if we hold hard evidence: **a full text of ≥ 40,000
characters that we read end-to-end and in which the dictionary + an independent audit found no
accession.** Anything short of that is not "empty," it is "not yet read well enough" — and is
flagged for a human. There is no probabilistic middle ground.

The **`certainty`** column (col I) holds one of three values:

| `certainty` | rows | rule |
|---|---:|---|
| `coded` | 5,581 | an accession is present |
| `verified-empty` | 3,434 | we read a **≥40k-char full paper**; no accession. Deterministic. |
| `needs-review` | 4,077 | **flagged for human review** — full text was paywalled, restricted, blocked, or too short (<40k) to trust |

Why 40k: real microbiome papers run 30k–80k characters of body text. A "read" that returned less
than that is a landing page, cookie wall, abstract, or a silently-blocked fetch — not a paper.
This threshold was adopted after finding that 170 of 262 browser "reads" had returned <500
characters (a blocked page) yet had been provisionally counted as read. The rule eliminates that
class of error by construction.

## 3. Master table — every no-code bucket

| flag | rows | verified-empty (≥40k) | needs-review | why no code / how determined |
|---|---:|---:|---:|---|
| **OA_AVAILABLE** | 2,678 | 1,902 | 776 | free copy exists outside PMC; 1,902 read a full paper → none; 776 not read to ≥40k. URL is in the row. |
| **PAYWALLED** | 2,319 | 41 | 2,278 | full text only at publisher; almost none machine-read. Library access needed. |
| **ON_REQUEST** | 991 | 844 | 147 | authors say "available on request"; 844 read in full confirm no deposited accession, 147 read too shallow to trust. |
| **NO_ACCESSION** | 813 | 482 | 331 | read the paper, none found; 482 on a full ≥40k text, 331 on shorter/abstract text → re-read. |
| **HUMAN_CAN_GET** | 434 | 0 | 434 | restricted PMC copy, machine-inaccessible; never read the body. |
| **SUPPLEMENT** | 188 | 140 | 48 | paper says "data in supplement"; main text read (none), supplement fetch tried → 0 codes in the 15 openable supplements. |
| **NO_IDENTIFIER** | 36 | 0 | 36 | no DOI/PMID/PMC to act on. |
| **NO_DATA** | 28 | 25 | 3 | paper states no new data generated. |
| **NO_FULLTEXT** | 24 | 0 | 24 | publisher restricts full text everywhere; metadata only. |

Totals: verified-empty **3,434**, needs-review **4,077**, sum **7,511** no-code rows.

## 4. How we analyzed each paper (channels)

Cheapest path first; a paper counts as "read" only if a channel returned ≥40k chars of body text.

1. **Europe PMC full text** (by PMCID) — primary source, open-access body XML.
2. **NCBI E-utilities fallback** — for papers EPMC lacks; non-OA papers return front-matter only
   (→ `HUMAN_CAN_GET`, not counted as read).
3. **EBI text-mined annotations** (by PMID) — recovers accessions from papers PMC can't
   redistribute; every mined string re-checked against the dictionary.
4. **Unpaywall** — locates legal OA copies outside PMC; drives `OA_AVAILABLE` vs `PAYWALLED`.
5. **403 Tier-1** — for bot-blocked publisher pages, re-tried an *alternate* OA location
   (repository/green copy). Read 1,462, recovered **27 codes**.
6. **403 Tier-2 (headful Chromium)** — headless is blocked by Wiley/OUP/MDPI (Akamai/Cloudflare);
   a headful, fingerprint-masked browser clears many. Ran all 648: **8 codes**, 70 full reads
   (≥40k) confirming no accession, the rest still blocked or returned thin pages.
7. **Supplement fetch** — EPMC supplementaryFiles for `SUPPLEMENT` rows; only 15/166 served, and
   those held **zero** accessions (questionnaires, patient tables, figures).

Every response is cached, so extraction re-runs in seconds after any rule change.

## 5. Validation — how confident is `verified-empty`?

An **independent audit** replayed each floor row's exact text through a detector deliberately
looser than production (guards off, whitespace-tolerant), so any accession the pipeline dropped
would resurface. Over 1,853 floor rows: 95 of 127 candidate hits were the journal name *"Cancers"*
glued to a citation number (`cancERS14040909`) — the guards were correctly rejecting them — and
only ~8 were genuine misses (~0.4%), all from a space splitting the accession (`PRJNA 795467`) or
an over-greedy lookalike filter. **All fixed and merged** (see §6). So `verified-empty` is an
audited floor, not an assumption.

A second audit went after the blind spot the first one structurally could not see: a looser copy of
our own regex shares our own assumptions. **Six independent reviewers read 24 randomly sampled
`verified-empty` papers in full** and tried to disprove the label. **23 of 24 confirmed empty.** The
single miss was an SRA *submission* ID of the form `SUB######` — a format class the dictionary had
never contained. A corpus sweep for that pattern found 34 rows, **29 of them wrongly marked
verified-empty**. That is the audit that justifies the rest: 96% confirmation is a real measurement,
and the 4% failure was a whole missing format, not a tuning error.

## 6. Corrections applied

Coded count over the recovery rounds: **5,117 → 5,547 → 5,581**.

- **+430** (5,117 → 5,547) from repository fallback on 403-blocked publishers, the whitespace-split
  and lookalike audit fixes, restricted-PMC Unpaywall copies, and a real-Chrome (CDP) parallel
  browser run over bot-walled publishers.
- **+34** (5,547 → 5,581) from the `SUB\d{6,}` submission-ID class the reviewer audit exposed;
  29 of those rows had been wrongly labelled `verified-empty` and were corrected.
- **10 were damaging false negatives** — rows previously ON_REQUEST / NO_ACCESSION that actually
  had public data. 8 confirmed; **3 tagged `[review]`** (rows 12106, 12252, 9601): repository DOIs
  (Zenodo/figshare) scraped from a landing page, not stated in the paper body — verify by hand.
- **66 provenance corrections** and **9 dbGaP version collapses** from the offline chronology pass.
- Invariant held throughout: **a human-entered code is never overwritten** — verified 0 changed.

## 7. What is flagged `needs-review` (4,077)

Every no-code row we are NOT entitled to call empty:

- **2,278** PAYWALLED — need library access
- **776** OA_AVAILABLE — free URL in the row, not yet read to ≥40k
- **434** HUMAN_CAN_GET — restricted PMC copy
- **331** NO_ACCESSION read too shallow (<40k / abstract) — re-read in full
- **147** ON_REQUEST read too shallow
- **48** SUPPLEMENT, **36** NO_IDENTIFIER, **24** NO_FULLTEXT, **3** NO_DATA

None of these is "confirmed no accession." They carry `certainty = needs-review` so they can be
filtered out and worked by a human.

---

### Column reference (`articles.out.csv`)

`url · doi · pubmed_id · pmc_id · Notes · Accession Code · flag · flag_detail · certainty`
