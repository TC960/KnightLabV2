#!/usr/bin/env python3
"""Fetch full text for the sheet papers we have not extracted yet.

Reuses the Europe PMC path proven in proj3-MMC2-scraping: resolve DOI -> PMCID via
the EPMC search API, then pull the open-access body XML. Local, free, no GPU.

Output: new_papers.json, same shape as all_usable_papers.json so the existing
extraction harness consumes it unchanged.

    python fetch_new_papers.py            # resume-safe, caches every response
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
SHEET = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
HAVE = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
MAIN = os.path.join(HERE, "..", "MAIN_DATA.json")
OUT = os.path.join(HERE, "new_papers.json")
CACHE = os.path.join(HERE, "fetch_cache")
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MIN_CHARS = 5000          # matches the usable-threshold of the existing corpus


def get(url, tag):
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", tag)[:120] + ".txt")
    if os.path.exists(key):
        return open(key, encoding="utf-8", errors="replace").read()
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


def resolve_pmcid(doi, title):
    for q in ([f'DOI:"{doi}"'] if doi else []) + ([f'TITLE:"{title[:120]}"'] if title else []):
        raw = get(f"{EPMC}/search?query={urllib.parse.quote(q)}&format=json&resultType=core&pageSize=1",
                  "search_" + q)
        if raw.startswith("__ERR__"):
            continue
        try:
            res = json.loads(raw)["resultList"]["result"]
        except Exception:
            continue
        if res and res[0].get("pmcid"):
            return res[0]["pmcid"], res[0]
    return None, None


def main():
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8-sig")))
    have = {(p["title"] or "").strip().lower()
            for p in json.load(open(HAVE))}
    todo = [r for r in rows if (r.get("Title") or "").strip().lower() not in have]
    print(f"{len(rows)} sheet rows | {len(todo)} not yet extracted", flush=True)

    # papers whose text already sits in MAIN_DATA need no network at all
    main = json.load(open(MAIN))
    by_title = {(v.get("name") or "").strip().lower(): v for v in main.values()}

    out, from_main, fetched, failed = [], 0, 0, 0
    enr = [c for c in rows[0] if "KeyTaxa_Enriched" in c][0]
    dep = [c for c in rows[0] if "Depleted" in c][0]

    for i, r in enumerate(todo, 1):
        title = (r.get("Title") or "").strip()
        text = None
        m = by_title.get(title.lower())
        if m and m.get("chunks"):
            text = " ".join(m["chunks"])
            from_main += 1
        else:
            doi = (r.get("DOI") or "").strip()
            pmcid, _rec = resolve_pmcid(doi, title)
            if pmcid:
                xml = get(f"{EPMC}/{pmcid}/fullTextXML", "ft_" + pmcid)
                if not xml.startswith("__ERR__") and "<body" in xml:
                    text = strip_xml(xml)
                    fetched += 1
        if not text or len(text) < MIN_CHARS:
            failed += 1
            continue
        out.append({
            "title": title,
            "link": (r.get("Link (Use DOI or Title if missing)") or "").strip(),
            "disease": (r.get("Disease") or "").strip(),
            "taxa_enriched": (r.get(enr) or "").strip(),
            "taxa_depleted": (r.get(dep) or "").strip(),
            "country": (r.get("Country") or "").strip(),
            "sequencing": (r.get("SequencingType") or "").strip(),
            "year": (r.get("Year") or "").strip(),
            "char_len": len(text), "usable": True, "text": text,
        })
        if i % 20 == 0:
            print(f"  [{i}/{len(todo)}] kept={len(out)} main={from_main} "
                  f"epmc={fetched} failed={failed}", flush=True)

    json.dump(out, open(OUT, "w"))
    print(f"\nwrote {OUT}: {len(out)} papers with full text "
          f"({from_main} from MAIN_DATA, {fetched} from EPMC, {failed} unavailable)")


if __name__ == "__main__":
    main()
