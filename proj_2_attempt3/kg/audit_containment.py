#!/usr/bin/env python3
"""Check every containment link in the graph against NCBI's real lineages.

WHY THIS COULD NOT BE RUN BEFORE. `graph.json`'s containment links come from
whichever taxonomy the build had. On a machine with the taxdump that is a true
lineage walk; here it is `taxonomy_cache`, which by its own docstring stores
"nearest-present-ancestor links, not full NCBI lineages" and is explicitly NOT a
general taxonomy. So the hierarchy has never been checked against an independent
source -- there was not one available.

`taxoniq` is that source: it ships NCBI's 2024 taxon DB via PyPI, which is
reachable where `ftp.ncbi.nih.gov` is not. It has no synonyms, so it cannot
build the graph, but ancestry is exactly what it does have.

TWO DIFFERENT QUESTIONS, and only the second is a defect:

  1. Is the link the NEAREST ancestor? Disagreements here are mostly harmless --
     a coarser link (family instead of genus) states something true.
  2. Is the parent an ancestor AT ALL? A "no" means the graph asserts a
     containment that does not exist.

The deliberate exception, which this reports separately rather than counting as
an error: a node split out of its parent by `resolve_named_children.py` is
contained by the genus THE PAPER NAMED it under. NCBI has since moved several of
these (*Bacteroides dorei* -> *Phocaeicola*, *Prevotella copri* -> *Segatella*),
so the link is deliberately not an NCBI ancestry claim. It is a claim about what
the corpus said, which is the thing the graph is a record of.

Run: python audit_containment.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import taxoniq
except ImportError:
    sys.exit("taxoniq not installed:  pip install taxoniq")

_ANC = {}


def ancestors(tid):
    """Every ancestor taxid of `tid`, nearest first. None if NCBI lacks it."""
    if tid in _ANC:
        return _ANC[tid]
    out = []
    try:
        t = taxoniq.Taxon(int(tid))
    except Exception:
        _ANC[tid] = None
        return None
    for _ in range(40):
        p = t.parent
        if p is None or p.tax_id == t.tax_id:
            break
        out.append(str(p.tax_id))
        t = p
    _ANC[tid] = out
    return out


def main():
    G = json.load(open(os.path.join(HERE, "graph.json")))
    present = {n["taxid"]: n for n in G["nodes"]
               if n["type"] == "taxon" and n.get("taxid")}
    split = {n["taxid"] for n in G["nodes"] if n.get("split_from_parent")}
    lab = lambda t: present[t]["label"] if t in present else t

    confirmed, unknown, wrong, deliberate, coarse = 0, [], [], [], []
    for h in G["hierarchy"]:
        p, c = h["parent"], h["child"]
        if not (p.startswith("t:ncbi:") and c.startswith("t:ncbi:")):
            continue
        pt, ct = p.split(":")[-1], c.split(":")[-1]
        anc = ancestors(ct)
        if anc is None:
            unknown.append((pt, ct))
            continue
        if pt not in anc:
            (deliberate if ct in split else wrong).append((pt, ct))
            continue
        confirmed += 1
        nearest = next((a for a in anc if a in present), None)
        if nearest and nearest != pt:
            coarse.append((pt, ct, nearest))

    n = confirmed + len(unknown) + len(wrong) + len(deliberate)
    print(f"taxid-to-taxid containment links: {n}")
    print(f"  confirmed by NCBI ancestry     : {confirmed}")
    print(f"  deliberate split links         : {len(deliberate)}")
    print(f"  NOT an ancestor (DEFECT)       : {len(wrong)}")
    print(f"  taxid unknown to NCBI 2024     : {len(unknown)}")
    print(f"  true but not nearest           : {len(coarse)}")
    for pt, ct in wrong:
        print(f"    DEFECT  {lab(pt)[:32]:32} does not contain {lab(ct)}")
    for pt, ct, nr in coarse[:15]:
        print(f"    coarse  {lab(pt)[:26]:26} > {lab(ct)[:30]:30} "
              f"(nearer: {lab(nr)})")
    json.dump({"links": n, "confirmed": confirmed, "deliberate": len(deliberate),
               "defects": [[lab(a), lab(b)] for a, b in wrong],
               "unknown": len(unknown), "coarse": len(coarse)},
              open(os.path.join(HERE, "containment_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
