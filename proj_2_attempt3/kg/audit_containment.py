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

The "deliberate" bucket is RETIRED and should now report 0. It counted split
species parented by the genus in their obsolete binomial (*Bacteroides dorei*
under *Bacteroides*, *Prevotella copri* under *Prevotella*), on the reasoning
that the link recorded what the corpus asserted by naming the organism that way.

That was dropped on 2026-09-07. The reasoning was coherent but the build did not
implement it: the lineage walk sometimes added the true genus as well, so 4 nodes
ended up with two different parents, and which 4 depended on whether the corpus
also happened to use the current name. Split species are now parented by their
nearest TRUE ancestor present in the graph; the obsolete binomial stays in the
node's `aliases`, which carries the provenance without asserting an ancestry NCBI
contradicts. A non-zero count here now means that fix has regressed.
See FINDINGS_containment_provenance.md.

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
