#!/usr/bin/env python3
"""Does restricting the graph to gut evidence move agreement with Disbiome/Peryton?

Disbiome and Peryton are gut-weighted, and 103 of our 1,985 edges (5.2%) carry
evidence from a saliva, nasal or blood study -- 58 of them MIXED with stool on the
same node. Scoring an oral finding against a gut record is a category error, and
`analyze_bodysite.py` puts a ceiling on how much it can matter: 10 of 167 decisive
Disbiome pairs and 24 of 136 Peryton pairs carry any non-gut evidence.

Unlike the three structural corrections before it, that ceiling is NOT ~zero, so
this is worth an actual test rather than an assumption either way.

Method is the same as analyze_filter_effect.py, for the same reason: comparing
headline percentages across variants is confounded, because dropping papers drops
whole diseases and moves the reference denominator. So compare PAIRWISE on the
pairs decisive in both variants, and use McNemar's exact test on the discordant
ones.

    python analyze_bodysite_effect.py
"""
import json
import os
import subprocess
import sys

from analyze_filter_effect import compare, per_pair
from validate_external import load_disbiome, load_peryton

HERE = os.path.dirname(os.path.abspath(__file__))
GUT = {"stool", "gut biopsy"}


def build_variant(rows, out_path):
    src = os.path.join(HERE, "_bodysite_rows.json")
    json.dump(rows, open(src, "w"))
    subprocess.run([sys.executable, os.path.join(HERE, "build_kg.py"),
                    "--input", src, "--out", out_path],
                   check=True, capture_output=True)
    os.remove(src)
    return json.load(open(out_path))["meta"]


def main():
    site = {t: v["site"] for t, v in
            json.load(open(os.path.join(HERE, "body_site.json")))["papers"].items()}
    rows = json.load(open(os.path.join(HERE, "extractions_screened.json")))

    # A paper with no label at all is KEPT: `unknown` is not evidence of non-gut,
    # and dropping it would confound this test with a coverage effect.
    gut_rows = [r for r in rows if site.get(r.get("title"), "unknown") in GUT
                or site.get(r.get("title")) is None]
    dropped = [r["title"] for r in rows if r not in gut_rows]
    print(f"rows: {len(rows)} -> gut-only {len(gut_rows)}  (dropped {len(dropped)})")
    for t in dropped:
        print(f"    [{site.get(t)}] {t[:78]}")

    m_all = build_variant(rows, os.path.join(HERE, "_graph_bs_all.json"))
    m_gut = build_variant(gut_rows, os.path.join(HERE, "_graph_bs_gut.json"))
    print(f"\n  all      : {m_all['n_edges']} edges, {m_all['n_contested']} contested, "
          f"{m_all['n_taxa']} taxa")
    print(f"  gut-only : {m_gut['n_edges']} edges, {m_gut['n_contested']} contested, "
          f"{m_gut['n_taxa']} taxa")

    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy()
    sources = [("Disbiome", load_disbiome()[0])]
    recs = load_peryton()[0]
    if recs:
        sources.append(("Peryton", recs))

    results = []
    for name, recs in sources:
        print("\n" + "=" * 78)
        print(name.upper())
        print("=" * 78)
        A = per_pair(os.path.join(HERE, "_graph_bs_all.json"), recs, tax)
        B = per_pair(os.path.join(HERE, "_graph_bs_gut.json"), recs, tax)
        results.append(compare(name, A, B, "all_sites", "gut_only"))

    json.dump(results, open(os.path.join(HERE, "bodysite_effect.json"), "w"), indent=1)
    print("\nwrote bodysite_effect.json")


if __name__ == "__main__":
    main()
