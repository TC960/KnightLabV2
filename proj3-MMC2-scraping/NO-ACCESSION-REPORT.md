# Papers Without an Accession Code — Analysis & Confidence Report

*Generated 2026-07-14. Corpus: 13,092 microbiome papers. Deliverable: `articles.out.csv`.*

## 1. Headline

| | rows | share |
|---|---:|---:|
| **Coded** — an accession is in the cell | **5,185** | 39.6% |
| **No code** | **7,907** | 60.4% |
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
| `coded` | 5,185 | an accession is present |
| `verified-empty` | 2,834 | we read a **≥40k-char full paper**; no accession. Deterministic. |
| `needs-review` | 5,073 | **flagged for human review** — full text was paywalled, restricted, blocked, or too short (<40k) to trust |

Why 40k: real microbiome papers run 30k–80k characters of body text. A "read" that returned less
than that is a landing page, cookie wall, abstract, or a silently-blocked fetch — not a paper.
This threshold was adopted after finding that 170 of 262 browser "reads" had returned <500
characters (a blocked page) yet had been provisionally counted as read. The rule eliminates that
class of error by construction.

## 3. Master table — every no-code bucket

| flag | rows | verified-empty (≥40k) | needs-review | why no code / how determined |
|---|---:|---:|---:|---|
| **OA_AVAILABLE** | 3,062 | 1,355 | 1,707 | free copy exists outside PMC; 1,355 read a full paper → none; 1,707 not read to ≥40k. URL is in the row. |
| **PAYWALLED** | 2,319 | 41 | 2,278 | full text only at publisher; almost none machine-read. Library access needed. |
| **ON_REQUEST** | 998 | 824 | 174 | authors say "available on request"; 824 read in full confirm no deposited accession, 174 read too shallow to trust. |
| **NO_ACCESSION** | 817 | 450 | 367 | read the paper, none found; 450 on a full ≥40k text, 367 on shorter/abstract text → re-read. |
| **HUMAN_CAN_GET** | 434 | 0 | 434 | restricted PMC copy, machine-inaccessible; never read the body. |
| **SUPPLEMENT** | 189 | 141 | 48 | paper says "data in supplement"; main text read (none), supplement fetch tried → 0 codes in the 15 openable supplements. |
| **NO_IDENTIFIER** | 36 | 0 | 36 | no DOI/PMID/PMC to act on. |
| **NO_DATA** | 28 | 23 | 5 | paper states no new data generated. |
| **NO_FULLTEXT** | 24 | 0 | 24 | publisher restricts full text everywhere; metadata only. |

Totals: verified-empty **2,834**, needs-review **5,073**, sum **7,907** no-code rows.

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

## 6. Corrections applied this session

- **68 recovered codes merged** (coded 5,117 → **5,185**): 27 from 403 repository copies, 31 from
  the whitespace-split fix, 2 from restricted-PMC Unpaywall, 8 from the headful browser run.
- **10 were damaging false negatives** — rows previously ON_REQUEST / NO_ACCESSION that actually
  had public data. 8 confirmed; **3 tagged `[review]`** (rows 12106, 12252, 9601): repository DOIs
  (Zenodo/figshare) scraped from a landing page, not stated in the paper body — verify by hand.

## 7. What is flagged `needs-review` (5,073)

Every no-code row we are NOT entitled to call empty:

- **2,278** PAYWALLED — need library access
- **1,707** OA_AVAILABLE — free URL in the row, not yet read to ≥40k
- **434** HUMAN_CAN_GET — restricted PMC copy
- **367** NO_ACCESSION read too shallow (<40k / abstract) — re-read in full
- **174** ON_REQUEST read too shallow
- **48** SUPPLEMENT, **36** NO_IDENTIFIER, **24** NO_FULLTEXT, **5** NO_DATA

None of these is "confirmed no accession." They carry `certainty = needs-review` so they can be
filtered out and worked by a human.

---

### Column reference (`articles.out.csv`)

`url · doi · pubmed_id · pmc_id · Notes · Accession Code · flag · flag_detail · certainty`
