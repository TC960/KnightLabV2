#!/usr/bin/env python3
"""Do the papers that DISAGREE about a taxon differ systematically in study design?

For every contested edge (papers report both enriched and depleted for the same
taxon-disease pair), split the supporting papers into the "up" group and the
"down" group and compare their study metadata: country, cohort size, body site,
sequencing method, 16S region, medication/diet control.

This is the standard explanation for microbiome irreproducibility -- that
disagreement tracks study design rather than biology -- and it is untestable from
direction alone, which is why extract_metadata.py exists.

WHAT THIS CAN AND CANNOT SHOW. With a typical contested edge supported by 5-15
papers split maybe 5/7, no test has the power to establish anything. Fisher's
exact p-values are reported to RANK leads, not to claim findings; with ~174
contested edges tested, some small p-values are expected by chance alone, so a
Benjamini-Hochberg FDR is reported alongside. Treat output as "these pairs are
worth a human look", never as "study design explains this disagreement".

    python analyze_contested.py            # ranked leads
    python analyze_contested.py --edge "Bacteroides|Alzheimer's disease"
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
METADATA = os.path.join(HERE, "metadata.jsonl")

CATEGORICAL = ["country", "body_site", "sequencing", "region_16S"]
BOOLEAN = ["medication_controlled", "diet_controlled"]


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher's exact p for [[a,b],[c,d]]. Exact, no scipy dependency."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, r2, c1 = a + b, c + d, a + c

    def p_of(x):
        y, z, w = r1 - x, c1 - x, r2 - (c1 - x)
        if min(y, z, w) < 0:
            return 0.0
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p_obs = p_of(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs + 1e-12))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values, same order as input."""
    m = len(pvals)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        prev = min(prev, pvals[i] * m / k)
        adj[i] = prev
    return adj


def load():
    G = json.load(open(GRAPH))
    md = {}
    if os.path.exists(METADATA):
        for line in open(METADATA):
            try:
                r = json.loads(line)
                if r.get("meta") and not r.get("parse_error"):
                    md[r["title"]] = r["meta"]
            except Exception:
                continue
    return G, md


def split_groups(edge, md, rows_by_paper):
    """-> (up_meta, down_meta): study metadata for papers on each side."""
    up, down = [], []
    for p in edge.get("papers", []):
        d = rows_by_paper.get(p)
        m = md.get(p)
        if not d or not m:
            continue
        (up if d == "enriched" else down).append(m)
    return up, down


def compare(up, down):
    """Yield (field, value, up_with, up_tot, down_with, down_tot, p)."""
    out = []
    for f in CATEGORICAL:
        vals = {m.get(f) for m in up + down if m.get(f)}
        for v in vals:
            a = sum(1 for m in up if m.get(f) == v)
            b = len(up) - a
            c = sum(1 for m in down if m.get(f) == v)
            d = len(down) - c
            if a + c == 0 or (b + d) == 0:
                continue
            out.append((f, v, a, len(up), c, len(down), fisher_exact_2x2(a, b, c, d)))
    for f in BOOLEAN:
        a = sum(1 for m in up if m.get(f) is True)
        b = len(up) - a
        c = sum(1 for m in down if m.get(f) is True)
        d = len(down) - c
        if a + c == 0:
            continue
        out.append((f, "true", a, len(up), c, len(down), fisher_exact_2x2(a, b, c, d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", help='"Taxon|Disease" to inspect in detail')
    ap.add_argument("--min-papers", type=int, default=4,
                    help="only contested edges with at least this many papers (default 4)")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    G, md = load()
    if not md:
        print(f"No metadata found at {METADATA} — run extract_metadata.py first.")
        return
    print(f"study metadata available for {len(md)} papers\n")

    # which direction did each paper report, per edge
    contested = [e for e in G["edges"] if e["contested"] and e["n_papers"] >= a.min_papers]
    print(f"contested edges with >={a.min_papers} papers: {len(contested)}")

    # graph.json keeps the paper list but not per-paper direction, so recover it
    # from the raw extraction rows.
    src = os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2", "results",
                       "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")
    rows = json.load(open(src))
    import re
    from build_kg import parse_taxa, norm_taxon, norm_disease
    try:
        from taxonomy import Taxonomy
        tax = Taxonomy()
    except Exception:
        tax = None
    dir_by = defaultdict(dict)      # (taxon_key, disease) -> {paper: direction}
    for r in rows:
        dis, _ = norm_disease(r.get("predicted_disease") or r.get("disease") or "")
        for d, col in (("enriched", "predicted_enriched"), ("depleted", "predicted_depleted")):
            for raw in parse_taxa(r.get(col)):
                k, _disp, _rank, _how = norm_taxon(raw, tax)
                dir_by[(k, dis)][r["title"]] = d

    results = []
    for e in contested:
        rb = dir_by.get((e["taxon_key"], e["disease"]), {})
        up, down = split_groups(e, md, rb)
        if len(up) < 2 or len(down) < 2:
            continue
        for (f, v, ua, ut, da, dt, p) in compare(up, down):
            results.append({"edge": f"{e['taxon']}|{e['disease']}", "n_papers": e["n_papers"],
                            "field": f, "value": v, "up": f"{ua}/{ut}", "down": f"{da}/{dt}", "p": p})

    if not results:
        print("\nNo contested edge yet has >=2 papers WITH metadata on both sides.")
        print("That is a coverage limit, not a null result — many papers do not state these fields.")
        return

    ps = [r["p"] for r in results]
    for r, q in zip(results, bh_fdr(ps)):
        r["fdr"] = q
    results.sort(key=lambda r: r["p"])

    print(f"\n{len(results)} field comparisons across {len({r['edge'] for r in results})} edges")
    print("\nRanked leads (p ranks them; FDR is the honest multiple-testing correction):\n")
    print(f"{'edge':46} {'field':16} {'value':14} {'up':>7} {'down':>7} {'p':>7} {'FDR':>7}")
    print("-" * 108)
    for r in results[:a.top]:
        print(f"{r['edge'][:45]:46} {r['field'][:15]:16} {str(r['value'])[:13]:14} "
              f"{r['up']:>7} {r['down']:>7} {r['p']:>7.3f} {r['fdr']:>7.3f}")

    sig = [r for r in results if r["fdr"] < 0.1]
    print(f"\n{len(sig)} comparison(s) survive FDR < 0.10.")
    if not sig:
        print("Expected: contested edges have single-digit paper counts on each side, so no\n"
              "test here has the power to reach significance. These are leads to check by\n"
              "hand, not findings.")


if __name__ == "__main__":
    main()
