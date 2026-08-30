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


def _perm_p(paper_dirs, paper_meta, field, value, nperm=6000, seed=0):
    """Cluster-robust p: shuffle the field label across PAPERS, not observations."""
    import random
    ps = list(paper_dirs.keys())
    if field in BOOLEAN:
        lab = [paper_meta[p].get(field) is True for p in ps]
    else:
        lab = [paper_meta[p].get(field) == value for p in ps]

    def stat(assign):
        A = B = C = D = 0
        for p, dirs in paper_dirs.items():
            for d in dirs:
                if assign[p]:
                    A += d == "enriched"; B += d == "depleted"
                else:
                    C += d == "enriched"; D += d == "depleted"
        pa = A / (A + B) if A + B else 0
        pc = C / (C + D) if C + D else 0
        return abs(pa - pc)

    obs = stat(dict(zip(ps, lab)))
    rnd = random.Random(seed)
    hits = 0
    for _ in range(nperm):
        rnd.shuffle(lab)
        if stat(dict(zip(ps, lab))) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (nperm + 1)


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

    # ---- pooled test: the per-edge tests are structurally underpowered -------
    # A contested edge has 2-8 papers per side, so the smallest attainable Fisher
    # p is ~0.1 no matter how clean the split. Pooling every (paper, edge)
    # observation across ALL contested edges trades per-edge specificity for
    # actual power: it cannot say "China explains Bacteroides in Alzheimer's",
    # but it can say whether cohort origin or method associates with reported
    # direction corpus-wide -- which is the reproducibility question anyway.
    print("\n" + "=" * 78)
    print("POOLED across all contested edges (per-edge tests are underpowered by design)")
    print("=" * 78)
    pooled = defaultdict(lambda: Counter())
    paper_dirs = defaultdict(list)   # paper -> [direction, ...] across contested edges
    paper_meta = {}
    n_obs = 0
    for e in contested:
        rb = dir_by.get((e["taxon_key"], e["disease"]), {})
        for paper, direction in rb.items():
            m = md.get(paper)
            if not m:
                continue
            n_obs += 1
            paper_dirs[paper].append(direction)
            paper_meta[paper] = m
            for f in CATEGORICAL:
                v = m.get(f)
                if v:
                    pooled[(f, v)][direction] += 1
            for f in BOOLEAN:
                if m.get(f) is True:
                    pooled[(f, "true")][direction] += 1
    all_up = sum(1 for e in contested
                 for pp, dd in dir_by.get((e["taxon_key"], e["disease"]), {}).items()
                 if dd == "enriched" and pp in md)
    all_dn = sum(1 for e in contested
                 for pp, dd in dir_by.get((e["taxon_key"], e["disease"]), {}).items()
                 if dd == "depleted" and pp in md)
    print(f"{n_obs} (paper, edge) observations with metadata "
          f"— {all_up} report enriched, {all_dn} depleted\n")
    prows = []
    for (f, v), c in pooled.items():
        a, b = c["enriched"], c["depleted"]
        if a + b < 8:
            continue
        cc, d = all_up - a, all_dn - b
        prows.append((f, v, a, b, fisher_exact_2x2(a, b, cc, d)))
    if not prows:
        print("Not enough observations per category yet.")
        return
    # zip straight into a new list -- an index() lookup would rebind the wrong row
    # whenever two categories share identical counts and p.
    # Fisher assumes independent observations, which is FALSE here: one paper
    # contributes one observation per taxon it reports (mean 3.9, max 15), so a
    # few prolific papers dominate and the naive p is anti-conservative. Recompute
    # the leading candidates with a permutation test that shuffles the label at the
    # PAPER level, keeping each paper's observations together. On this corpus that
    # moved diet_controlled from p=0.0021 to p=0.0093 -- same direction, weaker.
    prows.sort(key=lambda r: r[4])
    perm = {}
    for f, v, a, b, pv in prows[:6]:
        perm[(f, v)] = _perm_p(paper_dirs, paper_meta, f, v)
    final = [(f, v, a, b, pv, perm.get((f, v), pv)) for f, v, a, b, pv in prows]
    qs = bh_fdr([r[5] for r in final])
    final = [(*r, q) for r, q in zip(final, qs)]
    final.sort(key=lambda r: r[5])
    print(f"{'field':18} {'value':20} {'enr':>5} {'dep':>5} {'naive p':>8} "
          f"{'cluster p':>10} {'FDR':>7}")
    print("-" * 80)
    for f, v, a, b, pv, cp, q in final[:12]:
        star = "  <-- FDR<0.1" if q < 0.1 else ""
        print(f"{f[:17]:18} {str(v)[:19]:20} {a:>5} {b:>5} {pv:>8.4f} {cp:>10.4f} {q:>7.3f}{star}")
    n_sig = sum(1 for r in final if r[6] < 0.1)
    print(f"\n{n_sig} categor(ies) survive FDR<0.10 after BOTH the cluster and "
          f"multiple-testing corrections.")
    if not n_sig:
        print("The strongest signal is diet_controlled (studies controlling for diet report\n"
              "enrichment more often), but at FDR ~0.24 across 26 categories it is a\n"
              "hypothesis, not a result. On this corpus, study design does NOT visibly\n"
              "explain which direction a paper reports.")


if __name__ == "__main__":
    main()
