# Accession Code Extraction — Handoff to Claude Code

## Goal

For each of ~13,093 microbiome papers in a Google Sheet, find the **data accession code(s)**
(BioProject, SRA, GEO, EGA, dbGaP, etc.) and, where possible, tag whether the authors
**deposited** that data (`own`) or **reused** someone else's (`reused`).

Guiding principle, from the PI: **answer every row, flag ambiguity.** We are not trying to
be 100% deterministic. Every row gets a value. Anything uncertain gets flagged so a human
curator can target it. Flag-don't-drop. A flagged row is a working row, not a failure.

## Current state

The pipeline currently runs in **Google Apps Script**, which is the problem. 6-minute
execution cap, ~90 min/day trigger quota on personal Gmail, one-cell-at-a-time writes,
no debugger. A 5,712-row job takes ~5 hours across multiple days.

**Task: port to Python.** Same logic, running locally, with async requests and a real rate
limiter. Should complete the whole corpus in ~20 minutes.

### Numbers as of the last run

| | count |
|---|---|
| total rows | ~13,093 |
| answered | 7,351 |
| **unanswered (no `pmc_id`)** | **5,712** ← the main job |
| — of which have a PMID | 3,266 |
| — of which are DOI-only | 2,439 |
| — no identifier at all | 7 |

Flag breakdown of the 7,351 answered:

```
CLEAN             3957
SUPPLEMENT        1423   <- biggest recoverable opportunity, see below
ON_REQUEST         815   <- true floor
NO_ACCESSION       452
NO_FULLTEXT        321   <- publisher restricted, true floor
PROV_UNCLEAR       162
MULTI_CODE         105
MINED_NO_CONTEXT    75
NO_DATA             24
```

## Sheet schema

Tab: `articles.csv`

| col | header | notes |
|---|---|---|
| A | `url` | |
| B | `doi` | sometimes the only identifier |
| C | `pubmed_id` | often blank |
| D | `pmc_id` | **often blank — this is why 5,712 rows were skipped** |
| E | `notes` | pipeline writes `[auto] CODE=prov; CODE=prov` or `[auto] <floor reason>` |
| F | `accession code` | pipeline writes `CODE; CODE` or `N/A`. Blank or `ACCESSION_NOT_FOUND` = target row |
| G | `flag` | added by the flagging pass |
| H | `flag_detail` | human-readable why |

**Never overwrite a row where a human already put a real code.** Target rows are only
those where `accession code` is blank or `ACCESSION_NOT_FOUND`.

---

## Data sources — and everything we learned the hard way

### Europe PMC (EBI, UK) — the primary source. Reliable.

**Full text by PMCID:**
```
https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{digits}/fullTextXML
```
200 + `<body>` = full text. 404 = not in EPMC's open-access set (common for closed-access).

**Record lookup by DOI or PMID** — returns `pmcid`, `title`, **`abstractText`**:
```
https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:"{doi}"&format=json&resultType=core&pageSize=1
https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid} AND SRC:MED&format=json&resultType=core&pageSize=1
```
This replaces NCBI's `idconv` entirely. **We have no hard NCBI dependency anymore.**

**Text-mined annotations** — the sleeper hit. EBI mines papers it cannot redistribute
and exposes the extracted accessions:
```
https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds=PMC:{digits}&type=Accession Numbers&format=JSON
https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds=MED:{pmid}&type=Accession Numbers&format=JSON
```
- Accepts **up to 8 article IDs per request** (comma-separated). Batch this.
- Keyed by **PMID (`MED:`)** as well as PMCID, so it works on papers PMC has no copy of.
- **Recovered 78 codes from papers with zero downloadable full text.** Hit rate ~11%.
- **Returns non-deposit IDs too** (RefSNP `rs6667202`, etc). Do NOT trust `subType`.
  Instead, run every mined `exact` string through the DICT regexes — non-deposit IDs
  match nothing and drop out for free.

### NCBI E-utilities — fallback only. Treat with suspicion.

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={digits}&rettype=full&retmode=xml
```

**Findings that cost us hours:**

1. **The API key in `CFG.NCBI_KEY` was INVALID.** Every request carrying it returned
   **HTTP 400** `{"error":"API key status invalid"}`. This masqueraded as rate-limiting
   for hours. Verify any key before trusting it. If the 429 body echoes an **IPv6 address**
   in the `api-key` field, there is no valid key and you're on the shared-IP 3/sec limit.
2. **Non-OA papers return HTTP 200, not an error.** You get ~10-15KB of XML containing
   the front matter (title, authors, **abstract**) plus the comment
   `<!--The publisher of this article does not allow downloading of the full text-->`.
   Old code checked for `<body>`, found none, and threw all of it away. **Mine it** —
   it yielded real codes.
3. **`bioproject → pubmed` elink is EMPTY.** Confirmed: no `linksetdbs` at all. NCBI only
   populates it if a submitter registers a publication, and most never do. Dead end.
   `gds → pubmed` (GEO) *does* work, but there were only 3 such rows.

### Crossref — UNEXPLORED, and the obvious next lever

The DOI-only rows are closed-access papers (*Nature*, *Science*, *Nature Genetics*...).
PubMed has the abstract, PMC has nothing, EBI mined nothing. **Confirmed dead via
EPMC.** But Crossref stores publisher-deposited metadata, sometimes including data
availability statements and linked datasets:
```
https://api.crossref.org/works/{doi}
```
Free, no auth. **Probe this on a handful of the DOI-only leftovers before building on it.**
This is the most promising untried channel for the ~2,400 DOI-only rows.

### Supplementary files — the single biggest known opportunity

**1,423 rows** are flagged `SUPPLEMENT`: the paper says "data is in the supplementary
files," and we never fetched supplementary files. The accession is very likely sitting in
a supplementary table. EPMC has a supplementary-files endpoint. This is a scoped,
concrete project worth ~hundreds of codes.

---

## Extraction logic (port verbatim)

### Dictionary

```
BioProject     PRJ(?:EB|NA|DB|CA)\d{4,}       guard
BioSample      SAM(?:EA|N|D)\d{6,}            guard
SRA/ENA/DDBJ   [SED]R[APRSX]\d{6,}            guard
GEO            G(?:SE|SM|PL|DS)\d{4,}         guard
dbGaP          phs\d{6}(?:\.v\d+\.p\d+)?      guard
GSA            CR[ARX]\d{4,}                  guard
GSA-Human      HRA\d{4,}                      guard
GSA-KAP        KAP\d{4,}                      guard
NODE           OE[PXZS]\d{4,}                 guard
CNGB           CN[PSXR]\d{5,}                 guard
MG-RAST        mg[mp]\d+\.\d+                 guard
MetaboLights   MTBLS\d+                       guard
PRIDE          PXD\d{6}                       guard
EGA            EGA[SD]\d{6,}                  guard
ArrayExpress   E-[A-Z]{4}-\d+                 guard
BioStudies     S-[A-Z]{4}\d+                  guard
figshare       10\.6084/m9\.figshare\.\d+     no guard (DOI)
Zenodo         10\.5281/zenodo\.\d+           no guard (DOI)
Dryad          10\.506\d/dryad\.\w+           no guard (DOI)
```

The non-INSDC entries (GSA, CNGB, NODE, BioStudies, HRA, KAP) plus the DOI repos
accounted for **~450 codes, >10% of everything found**. They were added after the SOP
and are not busywork.

### Guards

- **Standalone-token guard** (where marked): reject if the char before is `[A-Za-z0-9]`
  or the char after is `[A-Za-z]`. Kills grant IDs like `JUSRP51710A`.
- **Lookalike exclusion**: reject if `±60` chars around the match matches
  `(refseq|primer|probe|taqman|assay|catalog|cat\.?\s*no|gene (id|expression)|mrna reference|\bNM_|\bNR_|\bXM_|\bNP_)`.

### Provenance (own / reused / unclear)

Currently regex on the text around the code: `OWN` matches deposit language
("deposited", "submitted", "we uploaded", "generated in this study", "data availability"),
`REUSE` matches ("downloaded from", "obtained from", "previously published",
"reanalyzed", "publicly available data"). Window widens ±200 → ±600 → whole paper.

**KNOWN SYSTEMATIC WEAKNESS — this is important:**
A cross-check against EPMC publication dates found **23 conflicts, and all 23 were in the
same direction: regex said `own`, evidence said `reused`.** They were almost entirely
**controlled-access repos: EGA (12), dbGaP (4), legacy SRA/ERA (3), HRA (1).**

Cause: in EGA/dbGaP, deposit and reuse are phrased identically —
*"data are available in the EGA under accession EGAS00001002702"* is what you write whether
you put it there or you're citing it. The regex cannot distinguish them.

**Free fix not yet implemented:** if an accession appears on **multiple rows in the corpus**,
only the earliest paper can be the depositor; the rest are reusing.
`EGAD00001004106` appears on rows 624, 1016, **and** 3441 — all labelled `own`. They can't
all be right. This is a pure offline pass over the sheet, no network, and it would catch
most of the EGA problem.

**Also unfixed:** dbGaP dedup. `PHS001768` (mined, bare) and `PHS001768.V1.P1`
(from front matter) are the same study but get written as two codes with contradictory
provenance. Collapse on the `phs` number, keep the versioned form.

### Floor reasons (when no code is found)

- `available on request` — true floor
- `no data generated` — true floor
- `data in article/supplement` — **NOT floor, recoverable** (the 1,423)
- `no accession in text` — full text read, genuinely nothing
- `no full text (publisher restricted)` — PMC has metadata only

---

## What to build

1. **Port to Python.** `gspread` or the Sheets API for I/O. Read the sheet once, process
   in memory, **write back in bulk** (the Apps Script version wrote one cell at a time,
   which is most of why it was slow). Async requests with a rate limiter.
2. **Finish the 5,712.** Per row, cheapest path first:
   `EPMC record (gives pmcid + abstract)` → check abstract → if pmcid, fetch full text →
   else EBI annotations by `MED:pmid` → else N/A with a specific reason.
3. **Probe Crossref** on the DOI-only leftovers before assuming they're floor.
4. **Then** the duplicate-accession provenance pass (offline, free, fixes the EGA bias).
5. **Then** supplementary-file fetching for the 1,423.

## Rate limits

- EPMC: forgiving. ~3-5 req/s is fine. Batch annotations 8 IDs per call.
- NCBI: 3/sec **per IP** unauthenticated, and the shared IP is contended. Get a **valid**
  API key (`account.ncbi.nlm.nih.gov/settings/` → API Key Management) for 10/sec, and
  **verify it returns 200 before trusting it.**

## Existing Apps Script files (paste these in for reference)

- `runExtraction.gs` — CFG, DICT, OWN/REUSE, `extractCodes_`, main loop
- `nonOA.gs` — non-OA handling, annotation mining
- `phase2.gs` — the 5,712 backfill
- `flags.gs` — the flagging pass and scorecard
