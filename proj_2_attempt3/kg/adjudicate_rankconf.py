#!/usr/bin/env python3
"""Task B: how much of the graph's contested/contradictory structure is rank confusion?

Two questions, kept separate because they have different fixes:

  1. STRUCTURAL   of the contested edges, how many involve a taxon that also
                  appears at another rank in the same disease (via graph.json's
                  `hierarchy` links)? Of those, how many point opposite ways?
  2. MECHANICAL   how many edges exist ONLY because taxonomy.py silently promoted
                  a SILVA/greengenes placeholder genus label ("X UCG-003",
                  "X ND3007 group", "unclassified_f__X") to its parent family?
                  Those are conflation by construction, not biology.

Read-only. Writes nothing.
"""
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "graph.json")))
SRC = os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2", "results",
                   "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")

# SILVA / greengenes placeholder labels: a *sub*-family bin, not the family itself
PLACEHOLDER = re.compile(
    r"(UCG[-_ ]?\d+|_?group$|ND\d{3,}|R-\d+\b|incertae[ _]sedis|"
    r"^(un|nor)(classified|ank)[_ ]|_?[a-z]__|sensu[ _]stricto|\bAD\d{3,}\b|"
    r"\b[A-Z]{1,3}\d{2,}\b)", re.I)


def closure(links):
    """ancestor -> {descendants} over the hierarchy DAG."""
    kids = defaultdict(set)
    for h in links:
        kids[h["parent"]].add(h["child"])
    anc = defaultdict(set)
    for root in list(kids):
        stack, seen = list(kids[root]), set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(kids.get(n, ()))
        anc[root] = seen
    return anc


def main():
    anc = closure(G["hierarchy"])
    desc_of = defaultdict(set)          # node -> ancestors
    for a, ds in anc.items():
        for d in ds:
            desc_of[d].add(a)

    by_dis = defaultdict(dict)          # disease -> node -> edge
    for e in G["edges"]:
        by_dis[e["disease"]][e["source"]] = e

    # --- 1. structural -------------------------------------------------------
    rel_pairs, opp_pairs = 0, 0
    for dis, nodes in by_dis.items():
        for n, e in nodes.items():
            for d in anc.get(n, ()):
                if d in nodes:
                    rel_pairs += 1
                    o = nodes[d]
                    if (not e["contested"] and not o["contested"]
                            and e["direction"] != o["direction"]):
                        opp_pairs += 1
    print(f"ancestor-descendant pairs within one disease : {rel_pairs}")
    print(f"  ... pointing OPPOSITE ways (both decisive) : {opp_pairs}")

    contested = [e for e in G["edges"] if e["contested"]]
    with_rel, with_opp = [], []
    for e in contested:
        nodes = by_dis[e["disease"]]
        rel = [nodes[x] for x in (anc.get(e["source"], set()) | desc_of.get(e["source"], set()))
               if x in nodes]
        if rel:
            with_rel.append((e, rel))
            if any(r["direction"] != e["direction"] or r["contested"] for r in rel):
                with_opp.append((e, rel))
    print(f"\ncontested edges                              : {len(contested)}")
    print(f"  with a same-disease relative at other rank : {len(with_rel)} "
          f"({100*len(with_rel)/len(contested):.1f}%)")
    print(f"  ... where the relative disagrees/contested : {len(with_opp)}")

    # --- 2. mechanical -------------------------------------------------------
    from taxonomy import Taxonomy
    tax = Taxonomy()
    recs = json.load(open(SRC))
    # every raw predicted string -> the taxid it collapsed onto
    promoted = defaultdict(set)         # taxid -> {raw placeholder strings}
    plain = defaultdict(set)            # taxid -> {raw exact-name strings}
    for r in recs:
        for fld in ("predicted_enriched", "predicted_depleted"):
            for raw in re.split(r"[,;]", r.get(fld) or ""):
                raw = raw.strip()
                if not raw:
                    continue
                tid, sci, rank, _ = tax.resolve(raw)
                if not tid:
                    continue
                if raw.lower() != (sci or "").lower() and PLACEHOLDER.search(raw):
                    promoted[tid].add(raw)
                else:
                    plain[tid].add(raw)

    only_promoted, mixed = [], []
    for e in G["edges"]:
        if not e["taxon_key"].startswith("ncbi:"):
            continue
        tid = e["taxon_key"].split(":", 1)[1]
        if tid in promoted:
            (only_promoted if tid not in plain else mixed).append(e)
    print(f"\nplaceholder labels silently promoted to a parent taxon : "
          f"{sum(len(v) for v in promoted.values())} distinct raw strings "
          f"over {len(promoted)} taxids")
    print(f"  edges whose taxon is ONLY ever named by a placeholder : {len(only_promoted)}")
    print(f"  edges whose taxon is named both ways (mixed evidence) : {len(mixed)}")
    print(f"  contested edges among the mixed set                   : "
          f"{sum(1 for e in mixed if e['contested'])}")
    print("\n  worst offenders (taxid -> raw strings folded into it):")
    for tid, raws in sorted(promoted.items(), key=lambda kv: -len(kv[1]))[:12]:
        _, sci, rank, _ = tax.resolve(tid) if False else (None, None, None, None)
        name = next((n["label"] for n in G["nodes"] if n.get("taxid") == tid), tid)
        print(f"    {name:26} <- {sorted(raws)[:6]}")


if __name__ == "__main__":
    main()
