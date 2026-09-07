#!/usr/bin/env python3
"""Add each split species' TRUE NCBI ancestor chain to named_child_taxids.json.

WHY. `named_child_taxids.json` records, for every named child split out of its
parent, the genus named by the surface string -- `parent_taxid` 838 (Prevotella)
for "Prevotella copri". NCBI has moved many of these: *Prevotella copri* is
*Segatella copri*, and 838 is not in its lineage at all. Parenting the node by
`parent_taxid` therefore asserts a containment NCBI contradicts, which
`audit_containment_ncbi.py` counts (11 such links) and which also produces nodes
with two different parents in what is supposed to be a tree.

The provenance those links were meant to carry is already kept: the surface
string stays in the node's `aliases`. So the containment link is free to say what
is taxonomically true.

This writes `ncbi_ancestors` -- every ancestor taxid, NEAREST FIRST -- so
`build_kg.py` can link each split species to the nearest ancestor that is
actually a node in the graph, without needing NCBI at build time. It also writes
`ncbi_genus` for display.

Idempotent: re-running overwrites the same fields. Needs `pip install taxoniq`
(the taxdump is unavailable here; see FINDINGS_containment_provenance.md).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "named_child_taxids.json")


def main():
    try:
        import taxoniq
    except ImportError:
        sys.exit("need taxoniq: pip install taxoniq")

    d = json.load(open(PATH))
    n_ok = n_moved = 0
    for r in d["resolutions"]:
        if r.get("verdict") != "species" or not r.get("taxid"):
            continue
        try:
            t = taxoniq.Taxon(int(r["taxid"]))
            anc = [str(a.tax_id) for a in t.lineage][1:]
        except Exception:
            print(f"  ! no NCBI lineage for {r['surface']} ({r.get('taxid')})")
            continue
        r["ncbi_ancestors"] = anc
        r["ncbi_genus"] = next(
            (a.scientific_name for a in t.ranked_lineage
             if a.rank and a.rank.name == "genus"), None)
        # Does the genus the surface string names actually contain this organism?
        r["parent_is_true_ancestor"] = bool(
            r.get("parent_taxid") and r["parent_taxid"] in anc)
        n_ok += 1
        if not r["parent_is_true_ancestor"]:
            n_moved += 1
            print(f"  moved: {r['surface']:32} -> {t.scientific_name:34} "
                  f"(genus {r['ncbi_genus']}); {r.get('parent')} is NOT an ancestor")

    d["note"] = (d.get("note", "") +
                 " | ncbi_ancestors/ncbi_genus/parent_is_true_ancestor added by "
                 "add_true_ancestors.py (NCBI 2024-09 snapshot via taxoniq)")
    json.dump(d, open(PATH, "w"), indent=1)
    print(f"\n{n_ok} species given a true ancestor chain; "
          f"{n_moved} are NOT contained by the genus their surface string names")


if __name__ == "__main__":
    main()
