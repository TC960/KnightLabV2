#!/usr/bin/env python3
"""Independently check every taxid->taxid containment link against NCBI lineages.

WHY A SECOND AUDIT. `audit_containment.py` already checks the hierarchy and
reports 601/616 confirmed with 2 defects. This one exists because that count is
not reproduced here: the species split attaches renamed species to the genus
named by their OBSOLETE binomial (Segatella copri under Prevotella, Agathobacter
rectalis under Eubacterium), and those links are taxonomically false yet were not
counted as defects. Two audits that disagree is the point -- one of them is
wrong, and the disagreement localises where.

THE TEST is deliberately the crudest possible one, so that it cannot share an
assumption with the thing it audits: for a link child -> parent, ask NCBI whether
`parent` appears anywhere in `child`'s lineage. Nothing else. No notion of
"deliberate", no rank logic, no allowance for how the link was produced.

  confirmed   parent is a true ancestor of child.
  FALSE       parent is NOT an ancestor. The link asserts containment that NCBI
              does not have. This is the class the species split introduces.
  unknown     one side is not in the bundled NCBI snapshot, so no verdict.

Data source is the `ncbi-taxon-db` wheel (NCBI 2024-09 snapshot) -- scientific
names only, but this audit needs only taxid->lineage, which is complete there.

Also reports MULTI-PARENT children: a taxon with two containment parents. NCBI
taxonomy is a tree, so a second parent is always an artifact, and the split
produces them by keeping the old-genus link and adding the new-genus one.

Usage: python3 audit_containment_ncbi.py [graph.json]
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "graph.json")
    try:
        import taxoniq
    except ImportError:
        sys.exit("need taxoniq: pip install taxoniq")

    g = json.load(open(path))
    nodes = {n["id"]: n for n in g["nodes"]}

    lin_cache = {}

    def lineage(tid):
        if tid not in lin_cache:
            try:
                t = taxoniq.Taxon(int(tid))
                lin_cache[tid] = {str(a.tax_id) for a in t.lineage}
            except Exception:
                lin_cache[tid] = None
        return lin_cache[tid]

    def label(nid):
        return nodes.get(nid, {}).get("label", nid)

    confirmed, false_links, unknown = [], [], []
    parents_of = defaultdict(list)

    for h in g.get("hierarchy", []):
        c, p = str(h.get("child", "")), str(h.get("parent", ""))
        if not (c.startswith("t:ncbi:") and p.startswith("t:ncbi:")):
            continue
        parents_of[c].append(p)
        ct, pt = c[7:], p[7:]
        cl = lineage(ct)
        if cl is None or lineage(pt) is None:
            unknown.append((c, p))
            continue
        if pt in cl:
            confirmed.append((c, p))
        else:
            false_links.append((c, p))

    multi = {c: ps for c, ps in parents_of.items() if len(ps) > 1}

    total = len(confirmed) + len(false_links) + len(unknown)
    print(f"graph: {path}")
    print(f"taxid->taxid containment links: {total}")
    print(f"  confirmed by NCBI lineage : {len(confirmed)}")
    print(f"  FALSE (parent not ancestor): {len(false_links)}")
    print(f"  unknown (not in snapshot)  : {len(unknown)}")
    print(f"  children with >1 parent    : {len(multi)}")

    if false_links:
        print("\n--- FALSE containment links ---")
        for c, p in sorted(false_links, key=lambda x: label(x[0])):
            try:
                t = taxoniq.Taxon(int(c[7:]))
                cur = t.scientific_name
                gen = next((a.scientific_name for a in t.ranked_lineage
                            if a.rank and a.rank.name == "genus"), "?")
            except Exception:
                cur, gen = "?", "?"
            print(f"  {label(c):34} -> {label(p):22} | NCBI: {cur} "
                  f"(genus {gen})")

    if multi:
        print("\n--- children with more than one containment parent ---")
        for c, ps in sorted(multi.items(), key=lambda x: label(x[0])):
            print(f"  {label(c):34} -> {', '.join(label(p) for p in ps)}")

    out = os.path.join(HERE, "containment_audit_ncbi.json")
    json.dump({
        "graph": os.path.basename(path),
        "links_checked": total,
        "confirmed": len(confirmed),
        "false": [[label(c), label(p), c, p] for c, p in false_links],
        "unknown": [[label(c), label(p)] for c, p in unknown],
        "multi_parent": {label(c): [label(p) for p in ps] for c, ps in multi.items()},
    }, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    return 1 if false_links else 0


if __name__ == "__main__":
    sys.exit(main())
