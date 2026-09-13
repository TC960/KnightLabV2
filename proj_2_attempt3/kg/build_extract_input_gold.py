#!/usr/bin/env python3
"""Stage the new-gold extraction set.

extract_input_gold.json = every paper in `high_confidence - final_constrained_override.csv`
that (a) has full text locally and (b) has at least one high-confidence gold taxon.
Records carry the run_eval.py field names, so the existing harness reads it with no
change beyond one DATASETS entry.

`taxa_enriched` / `taxa_depleted` hold the NEW gold (high_confidence_taxa), which is
what this run is meant to be scored against. The old Main-Datasheet annotations are
kept alongside as `sheet_enriched` / `sheet_depleted` so the two can be compared
without re-deriving the join; run_eval.py ignores the extra keys.

    python build_extract_input_gold.py
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = "/Users/mohak/Downloads/high_confidence - final_constrained_override.csv"
COVERAGE = os.path.join(HERE, "gold_coverage.json")
CORPORA = [
    os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json"),
    os.path.join(HERE, "new_papers.json"),
    os.path.join(HERE, "gold_missing_papers.json"),
]
OUT = os.path.join(HERE, "extract_input_gold.json")


def norm_title(t):
    t = re.sub(r"[^a-z0-9]+", " ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def doi_of(link):
    m = re.search(r"(10\.\d{4,9}/\S+)", (link or "").strip(), re.I)
    return m.group(1).rstrip("./").lower() if m else None


def main():
    gold = {}
    for r in csv.DictReader(open(GOLD, encoding="utf-8-sig")):
        doi = (r["DOI"] or "").strip().lower()
        g = gold.setdefault(doi, {"disorder": (r["disorder"] or "").strip(),
                                  "Enriched": [], "Depleted": []})
        taxa = [t.strip() for t in re.split(r"[;,]", r["high_confidence_taxa"] or "") if t.strip()]
        g[(r["field"] or "").strip()].extend(taxa)

    by_title, by_doi = {}, {}
    for path in CORPORA:
        if not os.path.exists(path):
            continue
        for p in json.load(open(path)):
            if not (p.get("text") or "").strip():
                continue
            p["_source"] = os.path.basename(path)
            by_title.setdefault(norm_title(p.get("title")), p)
            d = doi_of(p.get("link")) or (p.get("doi") or "").lower() or None
            if d:
                by_doi.setdefault(d, p)

    cov = json.load(open(COVERAGE))
    meta = {h["doi"]: h for h in cov["have"]}
    for m in cov["missing"]:
        meta.setdefault(m["doi"], m)

    out, no_text, no_taxa = [], 0, 0
    for doi, g in gold.items():
        m = meta.get(doi, {})
        rec = by_doi.get(doi) or by_title.get(norm_title(m.get("title")))
        if not rec:
            no_text += 1
            continue
        if not g["Enriched"] and not g["Depleted"]:
            no_taxa += 1
            continue
        out.append({
            "title": rec["title"],
            "link": m.get("link") or rec.get("link") or f"https://doi.org/{doi}",
            "doi": doi,
            "disease": g["disorder"] or m.get("disease_sheet") or rec.get("disease", ""),
            "taxa_enriched": "; ".join(g["Enriched"]),
            "taxa_depleted": "; ".join(g["Depleted"]),
            "sheet_enriched": rec.get("taxa_enriched", ""),
            "sheet_depleted": rec.get("taxa_depleted", ""),
            "in_gold_standard": "high_confidence",
            "text_source": rec["_source"],
            "char_len": len(rec["text"]),
            "usable": True,
            "text": rec["text"],
        })

    out.sort(key=lambda r: r["doi"])
    json.dump(out, open(OUT, "w"))

    chars = sorted(r["char_len"] for r in out)
    dis = {}
    for r in out:
        dis[r["disease"]] = dis.get(r["disease"], 0) + 1
    print(f"gold papers            : {len(gold)}")
    print(f"  dropped, no full text: {no_text}")
    print(f"  dropped, no gold taxa: {no_taxa}")
    print(f"EXTRACTION SET         : {len(out)}")
    print(f"chars  median {chars[len(chars)//2]:,}  p90 {chars[int(len(chars)*.9)]:,}  max {chars[-1]:,}")
    print(f"disorders: {len(dis)}")
    for d, n in sorted(dis.items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {d}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
