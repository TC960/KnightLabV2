#!/usr/bin/env python3
"""Fold the human gold standard into the extraction rows, tagged with provenance.

WHY MERGE AT THE INPUT, NOT THE GRAPH. `build_kg.py` takes extraction ROWS and
derives ~14 enrichment fields per edge (confidence, methods_diversity, sites,
taxon_purity, rank_conflicts ...) that `graph_gold.json` does not have, because it
was built before those were added. Unioning two finished graphs would produce edges
with half the fields missing and would drift from `build_kg.py` every time the
scheduled routine improves it. Converting the gold into rows and letting the one
builder run over everything keeps a single code path.

WHAT THIS PRESERVES. Each row carries `provenance`:

    "model"  extraction only
    "human"  the gold standard only
    "both"   the same paper appears in each

and the gold's taxa go into the SAME `predicted_enriched` / `predicted_depleted`
fields the builder reads, so no builder change is required for the merge itself.
A separate small patch adds a per-edge `provenance` summary for the viz legend.

WHAT THIS IS NOT. It is not a claim that the two sources are equally reliable. The
gold is human-curated over 334 papers; the extraction is model output over a
screened corpus. Marking each edge's provenance is exactly so a reader can weight
them differently -- that is the point of merging rather than averaging.

Overlap measured before writing: 1,067 taxon-disease keys appear in both graphs,
941 only in the extraction, 673 only in the gold -> union 2,681 (+33% over either).

    python build_union_input.py            # writes extractions_union.json
    python build_union_input.py --dry-run
"""
import argparse
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = "/Users/mohak/Downloads/high_confidence - final_constrained_override.csv"
SHEET = os.path.join(HERE, "Microbiota Signatures Neurological Disorders "
                           "Sheet 2 - Main Datasheet.csv")
EXTRACT = os.path.join(HERE, "extractions_screened.json")
OUT = os.path.join(HERE, "extractions_union.json")


def doi_of(s):
    m = re.search(r"10\.\d{4,9}/[^\s\"<>,;\]]+", s or "")
    return m.group(0).rstrip(".").lower() if m else None


def sv(v):
    return "" if v is None or isinstance(v, float) else str(v).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--extract", default=EXTRACT)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if not os.path.exists(a.extract):
        raise SystemExit(f"missing {a.extract} -- run the extraction pipeline first")

    # ---- gold: DOI -> {enriched, depleted, disorder} -----------------------
    gold = {}
    for x in csv.DictReader(open(a.gold, encoding="utf-8-sig")):
        d = x["DOI"].strip().lower()
        g = gold.setdefault(d, {"e": "", "d": "", "disorder": x["disorder"].strip()})
        g["e" if x["field"] == "Enriched" else "d"] = x["high_confidence_taxa"].strip()

    # ---- DOI -> title, from the datasheet ---------------------------------
    d2t, d2meta = {}, {}
    for x in csv.DictReader(open(SHEET, encoding="utf-8-sig")):
        d = doi_of(sv(x.get("DOI")).lower()) or sv(x.get("DOI")).lower()
        t = sv(x.get("Title"))
        if d and t:
            d2t[d] = t
            d2meta[d] = {"country": sv(x.get("Country")),
                         "sequencing": sv(x.get("SequencingType")),
                         "year": sv(x.get("Year")),
                         "link": sv(x.get("Link (Use DOI or Title if missing)"))}

    rows = json.load(open(a.extract))
    by_title = {sv(r.get("title")).lower(): r for r in rows}

    for r in rows:
        r["provenance"] = "model"

    added, upgraded, skipped = 0, 0, 0
    for doi, g in gold.items():
        if not (g["e"] or g["d"]):
            skipped += 1                      # blank gold contributes nothing
            continue
        title = d2t.get(doi)
        if not title:
            skipped += 1
            continue
        key = title.lower()
        if key in by_title:
            # Same paper in both sources. UNION the taxa rather than keeping only
            # the extraction's: 259 of 325 extraction papers are also in the gold,
            # so if shared papers contributed nothing the merge would add only the
            # 35 genuinely new papers and discard the recall gain that motivated
            # it -- most of the 673 gold-only edges come from papers we already
            # have, where the human curator found taxa the model missed.
            #
            # Per-taxon origin is recorded so edge provenance stays traceable;
            # `human_only_*` is what the human found and the model did not.
            r = by_title[key]
            for fld, gval in (("predicted_enriched", g["e"]),
                              ("predicted_depleted", g["d"])):
                have = {t.strip().lower() for t in re.split(r"[;,]", sv(r.get(fld)))
                        if t.strip()}
                new_t = [t.strip() for t in re.split(r"[;,]", gval)
                         if t.strip() and t.strip().lower() not in have]
                if new_t:
                    r[fld] = (sv(r.get(fld)) + "; " + "; ".join(new_t)).strip("; ")
                r["human_only_" + fld.split("_")[1]] = "; ".join(new_t)
            r["provenance"] = "both"
            r["gold_enriched"] = g["e"]
            r["gold_depleted"] = g["d"]
            upgraded += 1
            continue
        meta = d2meta.get(doi, {})
        rows.append({
            "title": title,
            "doi": doi,
            "link": meta.get("link", ""),
            "disease": g["disorder"],
            "predicted_disease": g["disorder"],
            "predicted_enriched": g["e"],
            "predicted_depleted": g["d"],
            "expected_enriched": g["e"],
            "expected_depleted": g["d"],
            "country": meta.get("country", ""),
            "sequencing": meta.get("sequencing", ""),
            "year": meta.get("year", ""),
            "parse_error": False,
            "provenance": "human",
        })
        added += 1

    n = {"model": 0, "human": 0, "both": 0}
    for r in rows:
        n[r.get("provenance", "model")] += 1

    print(f"extraction rows in       : {len(rows) - added}")
    print(f"gold papers added        : {added}")
    print(f"gold papers already present (marked 'both'): {upgraded}")
    print(f"gold papers skipped (blank taxa or no title): {skipped}")
    print(f"UNION rows               : {len(rows)}")
    print(f"  provenance model       : {n['model']}")
    print(f"  provenance human       : {n['human']}")
    print(f"  provenance both        : {n['both']}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    json.dump(rows, open(a.out, "w"))
    print(f"\nwrote {a.out}")
    print("next: python build_kg.py --input extractions_union.json "
          "--out graph_union.json && python build_viz.py --graph graph_union.json "
          "--out kg_union.html")


if __name__ == "__main__":
    main()
