#!/usr/bin/env python3
"""Fetch full text for the new-gold papers we do not have locally.

Adapted from fetch_new_papers.py -- same Europe PMC path (DOI -> PMCID ->
fullTextXML), same on-disk cache (fetch_cache/), same polite 0.25 s delay, same
>=5000 char usability threshold. Differences, all forced by the new gold file:

  * input is the missing list from gold_coverage.py (DOI-keyed), not the sheet;
  * two gold rows carry `pmid/NNNNN` in the DOI column instead of a real DOI, so
    those resolve through EPMC's EXT_ID: (PMID) query;
  * the DOI search is retried as a PMID search and then a title search;
  * cached `__ERR__` bodies are re-fetched unless they are a 404 -- a 404 from
    fullTextXML is a real "not in the OA subset" answer and will never change,
    whereas a timeout is transient and should not be frozen into the cache;
  * NCBI is added as a second source behind Europe PMC, at both steps: the PMC ID
    Converter for DOI->PMCID and efetch(db=pmc) for the body. This is not
    redundancy for its own sake -- EPMC's fullTextXML 404s on PMC *author
    manuscripts* that efetch serves in full (PMC10143502, PMC10680783 and two
    others here), and EPMC's REST API was returning 503 across the board while
    this was written, so the EPMC-only script could not have run at all.

Output: gold_missing_papers.json, same record shape as new_papers.json.

    python fetch_gold_missing.py              # resume-safe
    python fetch_gold_missing.py --dry-run    # resolve from cache only, no network
"""
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
COVERAGE = os.path.join(HERE, "gold_coverage.json")
SHEET = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
MAIN = os.path.join(HERE, "..", "MAIN_DATA.json")
OUT = os.path.join(HERE, "gold_missing_papers.json")
REPORT = os.path.join(HERE, "gold_missing_report.json")
CACHE = os.path.join(HERE, "fetch_cache")
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_ID = "tool=KnightLabKG&email=map.960.20@gmail.com"   # NCBI asks callers to identify
MIN_CHARS = 5000          # matches the usable-threshold of the existing corpus

DRY = "--dry-run" in sys.argv


def get(url, tag):
    """Cached HTTP GET. A cached 404 is final; any other cached error is retried."""
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", tag)[:120] + ".txt")
    if os.path.exists(key):
        body = open(key, encoding="utf-8", errors="replace").read()
        if not body.startswith("__ERR__") or "404" in body[:80]:
            return body
        if DRY:
            return body
    if DRY:
        return "__ERR__ DryRun: not cached"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KnightLab-KG/1.0"})
        with urllib.request.urlopen(req, timeout=45) as r:
            body = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        body = f"__ERR__ {type(e).__name__}: {e}"
    open(key, "w", encoding="utf-8").write(body)
    time.sleep(0.25)                      # EPMC is forgiving but be polite
    return body


def strip_xml(x):
    x = re.sub(r"<ref-list.*?</ref-list>", " ", x, flags=re.S)
    x = re.sub(r"<(table-wrap|fig|back).*?</\1>", " ", x, flags=re.S)
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"&[a-z]+;|&#\d+;", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def norm_title(t):
    t = re.sub(r"[^a-z0-9]+", " ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def queries(doi, title):
    """Ordered EPMC query list. `pmid/NNN` in the DOI column becomes EXT_ID."""
    qs = []
    pm = re.match(r"^pmid[:/](\d+)$", (doi or "").strip(), re.I)
    if pm:
        qs.append(f"EXT_ID:{pm.group(1)} AND SRC:MED")
    elif doi:
        qs.append(f'DOI:"{doi}"')
    if title:
        qs.append(f'TITLE:"{title[:120]}"')
    return qs


def resolve_epmc(doi, title):
    """-> (pmcid, reason). reason is only meaningful when pmcid is None."""
    seen_any = False
    for q in queries(doi, title):
        raw = get(f"{EPMC}/search?query={urllib.parse.quote(q)}&format=json&resultType=core&pageSize=5",
                  "search_" + q)
        if raw.startswith("__ERR__"):
            continue
        try:
            res = json.loads(raw)["resultList"]["result"]
        except Exception:
            continue
        if res:
            seen_any = True
            # a DOI can return the MED record first and the PMC one second; scan all
            hit = next((r for r in res if r.get("pmcid")), None)
            if hit:
                return hit["pmcid"], None
    return None, ("indexed_but_no_pmcid" if seen_any else "not_found_in_epmc")


def resolve_ncbi(doi):
    """PMC ID Converter. Accepts a DOI or a bare PMID. -> (pmcid, reason)."""
    pm = re.match(r"^pmid[:/](\d+)$", (doi or "").strip(), re.I)
    ident = pm.group(1) if pm else doi
    raw = get(f"{IDCONV}?ids={urllib.parse.quote(ident, safe='')}&format=json&{NCBI_ID}",
              "idconv_" + ident)
    if raw.startswith("__ERR__"):
        return None, "idconv_error"
    try:
        rec = json.loads(raw).get("records", [{}])[0]
    except Exception:
        return None, "idconv_unparseable"
    if rec.get("pmcid"):
        return rec["pmcid"], None
    return None, "not_in_pmc"


def body_for(pmcid):
    """EPMC fullTextXML first, then NCBI efetch. -> (xml, reason)."""
    xml = get(f"{EPMC}/{pmcid}/fullTextXML", "ft_" + pmcid)
    if not xml.startswith("__ERR__") and "<body" in xml:
        return xml, None
    # EPMC 404s on PMC author manuscripts that efetch serves in full
    xml = get(f"{EFETCH}?db=pmc&id={pmcid}&rettype=xml&{NCBI_ID}", "efetch_" + pmcid)
    if xml.startswith("__ERR__"):
        return None, "fetch_error"
    if "<body" not in xml:
        return None, "no_oa_fulltext"       # record exists, publisher blocks the body
    return xml, None


def main():
    cov = json.load(open(COVERAGE))
    todo = cov["missing"]
    print(f"{len(todo)} gold papers lack local full text\n", flush=True)

    sheet = {(r.get("DOI") or "").strip().lower(): r
             for r in csv.DictReader(open(SHEET, encoding="utf-8-sig"))
             if (r.get("DOI") or "").strip()}

    # papers whose text already sits in MAIN_DATA need no network at all.
    # (fetch_new_papers.py matched raw-lowercase titles; normalised matching here
    # catches rows that differ only by punctuation or a trailing period.)
    by_title = {}
    if os.path.exists(MAIN):
        for v in json.load(open(MAIN)).values():
            if v.get("chunks"):
                by_title.setdefault(norm_title(v.get("name")), v)
    print(f"MAIN_DATA titles indexed: {len(by_title)}", flush=True)

    out, fails = [], []
    from_main = fetched = 0
    for i, m in enumerate(todo, 1):
        doi, title = m["doi"], m["title"]
        text, how, reason = None, None, None

        hit = by_title.get(norm_title(title))
        if hit:
            text, how = " ".join(hit["chunks"]), "MAIN_DATA"
        else:
            pmcid, reason = resolve_epmc(doi, title)
            if not pmcid:
                pmcid, r2 = resolve_ncbi(doi)
                reason = reason if pmcid else r2
            if pmcid:
                xml, reason = body_for(pmcid)
                if xml:
                    text, how = strip_xml(xml), "PMC:" + pmcid

        if text and len(text) < MIN_CHARS:
            reason, text = f"too_short({len(text)})", None
        if not text:
            fails.append({**m, "reason": reason or "unknown"})
            print(f"  [{i}/{len(todo)}] MISS {doi:<38} {reason}", flush=True)
            continue

        if how == "MAIN_DATA":
            from_main += 1
        else:
            fetched += 1
        r = sheet.get(doi, {})
        enr = next((c for c in r if "KeyTaxa_Enriched" in c), None)
        dep = next((c for c in r if "Depleted" in c), None)
        out.append({
            "title": title,
            "link": (r.get("Link (Use DOI or Title if missing)") or "").strip() or f"https://doi.org/{doi}",
            "disease": (r.get("Disease") or "").strip() or m.get("disorder", ""),
            "taxa_enriched": (r.get(enr) or "").strip() if enr else "",
            "taxa_depleted": (r.get(dep) or "").strip() if dep else "",
            "country": (r.get("Country") or "").strip(),
            "sequencing": (r.get("SequencingType") or "").strip(),
            "year": (r.get("Year") or "").strip(),
            "char_len": len(text), "usable": True, "text": text,
            "doi": doi, "fetched_via": how,
        })
        print(f"  [{i}/{len(todo)}] OK   {doi:<38} {how} {len(text)} chars", flush=True)

    json.dump(out, open(OUT, "w"))
    by_reason = {}
    for f in fails:
        by_reason[f["reason"]] = by_reason.get(f["reason"], 0) + 1
    json.dump({"fetched": len(out), "failed": len(fails),
               "by_reason": by_reason, "failures": fails}, open(REPORT, "w"), indent=1)

    print(f"\nwrote {OUT}: {len(out)}/{len(todo)} papers "
          f"({from_main} from MAIN_DATA, {fetched} from EPMC)")
    print(f"failed {len(fails)}: {by_reason}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
