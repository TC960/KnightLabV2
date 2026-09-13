#!/usr/bin/env python3
"""Which of the 334 new-gold papers do we already have full text for?

Join path (the gold CSV carries no titles, and our corpora carry no DOIs):

    gold CSV  --DOI-->  Main Datasheet  --Title-->  {all_usable_papers, new_papers}

Title is the only key shared with the corpora, so matching is done on a normalised
title (lowercase, punctuation and whitespace collapsed). DOI parsed out of the
corpus `link` field is used as a secondary key where the link happens to be a
doi.org URL -- it catches the handful of rows whose titles were re-typed.

    python gold_coverage.py            # prints the report, writes gold_coverage.json
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = "/Users/mohak/Downloads/high_confidence - final_constrained_override.csv"
SHEET = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
HAVE = [
    os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json"),
    os.path.join(HERE, "new_papers.json"),
    os.path.join(HERE, "gold_missing_papers.json"),      # may not exist yet
]
OUT = os.path.join(HERE, "gold_coverage.json")


def norm_title(t):
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def doi_of(link):
    m = re.search(r"(10\.\d{4,9}/\S+)", (link or "").strip(), re.I)
    return m.group(1).rstrip("./").lower() if m else None


def load_gold():
    """-> {doi: {"disorder":…, "enriched":[…], "depleted":[…]}} over 334 DOIs."""
    gold = {}
    for r in csv.DictReader(open(GOLD, encoding="utf-8-sig")):
        doi = (r["DOI"] or "").strip().lower()
        if not doi:
            continue
        g = gold.setdefault(doi, {"doi": doi, "disorder": (r["disorder"] or "").strip(),
                                  "enriched": [], "depleted": []})
        taxa = [t.strip() for t in re.split(r"[;,]", r["high_confidence_taxa"] or "") if t.strip()]
        key = "enriched" if (r["field"] or "").strip().lower().startswith("enrich") else "depleted"
        g[key].extend(taxa)
    return gold


def load_sheet():
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8-sig")))
    by_doi = {}
    for r in rows:
        d = (r.get("DOI") or "").strip().lower()
        if d:
            by_doi.setdefault(d, r)
    return rows, by_doi


def load_corpora():
    """-> ({norm_title: rec}, {doi: rec}) across every local full-text store."""
    by_title, by_doi, counts = {}, {}, {}
    for path in HAVE:
        if not os.path.exists(path):
            continue
        recs = json.load(open(path))
        counts[os.path.basename(path)] = len(recs)
        for p in recs:
            if not (p.get("text") or "").strip():
                continue
            p["_source"] = os.path.basename(path)
            by_title.setdefault(norm_title(p.get("title")), p)
            d = doi_of(p.get("link"))
            if d:
                by_doi.setdefault(d, p)
    return by_title, by_doi, counts


def main():
    gold = load_gold()
    rows, sheet_by_doi = load_sheet()
    by_title, by_doi, counts = load_corpora()

    have, missing = [], []
    for doi, g in gold.items():
        srow = sheet_by_doi.get(doi)
        title = (srow.get("Title") or "").strip() if srow else ""
        rec = by_doi.get(doi) or by_title.get(norm_title(title))
        item = {
            "doi": doi,
            "title": title,
            "disorder": g["disorder"],
            "disease_sheet": (srow.get("Disease") or "").strip() if srow else "",
            "link": (srow.get("Link (Use DOI or Title if missing)") or "").strip() if srow else "",
            "n_enriched": len(g["enriched"]),
            "n_depleted": len(g["depleted"]),
            "in_sheet": srow is not None,
        }
        if rec:
            item["source"] = rec["_source"]
            item["char_len"] = len(rec["text"])
            have.append(item)
        else:
            missing.append(item)

    print(f"gold papers (unique DOI)     : {len(gold)}")
    print(f"present in Main Datasheet    : {sum(1 for g in gold if g in sheet_by_doi)}")
    print(f"local corpora                : {counts}")
    print(f"HAVE full text               : {len(have)}")
    print(f"MISSING full text            : {len(missing)}")
    src = {}
    for h in have:
        src[h["source"]] = src.get(h["source"], 0) + 1
    print(f"  by source                  : {src}")
    print()
    print("MISSING:")
    for m in sorted(missing, key=lambda x: x["doi"]):
        print(f"  {m['doi']:<42} {m['title'][:88]}")

    json.dump({"have": have, "missing": missing}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
