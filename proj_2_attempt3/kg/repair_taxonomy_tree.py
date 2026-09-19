#!/usr/bin/env python3
"""Rebuild the containment hierarchy from FULL NCBI lineages, not just observed taxa.

THE PROBLEM, measured. `graph.json`'s hierarchy links only taxa that both appear
in our papers, so lineages have holes:

    nodes 747, containment edges 727, ROOTS 20
    root ranks: 13 phylum, 2 order, 2 genus, 1 kingdom, 1 family, 1 no rank

It is a forest of fragments, and depth-within-fragment is NOT taxonomic rank -- a
genus hanging straight off a phylum root sits at depth 1 while another genus three
levels into a populated fragment sits at depth 3.

WHY IT MATTERS. Every hierarchy-geometry method in the literature review needs a
real tree: hyperbolic embeddings (research/01), box and order embeddings (03),
hierarchical FDR and treeclimbR (10). It was confirmed empirically rather than
assumed -- a Poincare embedding on the broken hierarchy gives
Spearman(radius, true rank) = 0.0095, p = 0.81. The geometry learned fragment
depth faithfully; fragment depth means nothing.

THE FIX. Walk each observed taxon's full lineage in NCBI and insert the missing
intermediate ancestors as STEINER nodes -- present to hold the tree together, but
flagged so nothing downstream mistakes them for taxa we have evidence about.
Every observed taxon then hangs off one root at its true depth.

WHAT THIS DOES NOT DO. It does not merge ranks, and it does not touch the
association edges. Lachnospiraceae (depleted, 15 papers) and Hungatella inside it
(enriched, 7 papers) stay separate nodes with opposite signs -- that disagreement
is a finding, and collapsing it is exactly the error this project already refused.

    python repair_taxonomy_tree.py --dry-run
    python repair_taxonomy_tree.py
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "taxonomy_tree.json")

CANONICAL = ["superkingdom", "kingdom", "phylum", "class", "order",
             "family", "genus", "species"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, HERE)
    try:
        from taxonomy_cache import load_taxonomy
        tax = load_taxonomy()
    except Exception:
        from taxonomy import Taxonomy
        tax = Taxonomy()
    if not getattr(tax, "ok", False):
        raise SystemExit("no taxdump available -- taxonomy.py needs names.dmp/nodes.dmp")

    g = json.load(open(GRAPH))
    taxa = [n for n in g["nodes"] if n.get("type") == "taxon"]
    resolved = [n for n in taxa if n.get("taxid")]
    print(f"taxa in graph: {len(taxa)}   with an NCBI taxid: {len(resolved)}")

    # ---- walk every observed taxon up to the root ------------------------
    parent = {}                 # taxid -> parent taxid
    rank = {}
    observed = set()
    missing_lineage = 0
    for n in resolved:
        tid = str(n["taxid"]).replace("ncbi:", "")
        try:
            lin = tax.lineage(tid)
        except Exception:
            lin = None
        if not lin:
            missing_lineage += 1
            continue
        observed.add(tid)
        # lineage() returns root-ward; chain consecutive pairs
        chain = [str(x) for x in lin]
        for child, par in zip(chain, chain[1:]):
            parent[child] = par
            rank.setdefault(child, tax.rank.get(child, "no rank"))
            rank.setdefault(par, tax.rank.get(par, "no rank"))

    all_nodes = set(parent) | set(parent.values())
    roots = [t for t in all_nodes if t not in parent]
    steiner = all_nodes - observed

    print(f"could not resolve a lineage for: {missing_lineage}")
    print()
    print("AFTER REPAIR")
    print(f"  nodes          : {len(all_nodes)}  "
          f"({len(observed)} observed + {len(steiner)} inserted ancestors)")
    print(f"  edges          : {len(parent)}")
    print(f"  ROOTS          : {len(roots)}  ->  {[rank.get(r,'?') for r in roots][:6]}")

    # ---- the test that matters: does depth now equal rank? ---------------
    def depth(t):
        d, seen = 0, set()
        while t in parent and t not in seen:
            seen.add(t)
            t = parent[t]
            d += 1
        return d

    by_rank = collections.defaultdict(list)
    for t in all_nodes:
        r = rank.get(t, "no rank")
        if r in CANONICAL:
            by_rank[r].append(depth(t))

    print()
    print("  mean depth by canonical rank (must increase monotonically):")
    prev, mono = -1, True
    for r in CANONICAL:
        if r not in by_rank:
            continue
        m = sum(by_rank[r]) / len(by_rank[r])
        flag = "" if m >= prev else "   <-- OUT OF ORDER"
        if m < prev:
            mono = False
        prev = m
        print(f"    {r:<14} n={len(by_rank[r]):>5}  mean depth {m:5.2f}{flag}")
    print(f"\n  monotonic: {mono}   "
          f"{'-> depth now tracks rank; geometry methods are unblocked' if mono else '-> still broken'}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    json.dump({"parent": parent, "rank": rank,
               "observed": sorted(observed), "roots": roots,
               "n_steiner": len(steiner)}, open(OUT, "w"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
