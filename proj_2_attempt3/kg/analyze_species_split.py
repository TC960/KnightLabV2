#!/usr/bin/env python3
"""Decompose the agreement change from the named-child split. Do not report a rate.

WHY A DECOMPOSITION AND NOT A P-VALUE. `agreement_metric.py` exists for
corrections that REMOVE papers, and its null resamples papers accordingly. This
correction removes nothing: it re-keys 25 surface strings onto their own species
taxids. A drop-N-papers null does not describe it, and running one anyway would
be a p-value for a question nobody asked.

The question that matters is cheaper and sharper than a p-value. An agreement
RATE can rise three ways, and only one of them is the graph getting anything
right:

  1. pairs the two sources newly share, which happen to agree  -> COVERAGE
  2. pairs that were disagreeing and now agree                 -> CORRECTION
  3. pairs that were agreeing and drop out of the comparison   -> ATTRITION

The project's own rule is that a structural correction may never be cited as an
accuracy gain. So the useful output is which of these three moved, not whether
the rate moved -- and the only way to be honest about that is to name each pair
that changed and say why.

Run: python analyze_species_split.py --before <graph.json> --after <graph.json>
"""
import argparse
import json
import os
from collections import Counter, defaultdict

from validate_external import (DISEASE_MAP, OUTCOME, load_disbiome, load_peryton,
                               load_species_overrides)
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def reference_dirs(records, tax, override):
    """(taxid, our_disease) -> reference direction, from a curated database."""
    ref = defaultdict(Counter)
    for r in records:
        our = DISEASE_MAP.get(r["disease"].lower())
        out = OUTCOME.get(r["outcome"].lower())
        if not (our and out and r["microbe"]):
            continue
        mic = re.sub(r"\s+", " ", r["microbe"].strip())
        tid = override.get(mic.lower())
        if not tid:
            tid, _s, _r, _h = tax.resolve(mic) if tax.ok else (None, 0, 0, 0)
        if not tid:
            continue
        ref[(tid, our)][out] += 1
    out = {}
    for k, v in ref.items():
        if v["enriched"] != v["depleted"]:
            out[k] = "enriched" if v["enriched"] > v["depleted"] else "depleted"
    return out


def our_dirs(G):
    """(taxid, disease) -> (direction, n_papers) for taxid-keyed edges."""
    out = {}
    for e in G["edges"]:
        k = e["taxon_key"]
        if k.startswith("ncbi:"):
            out[(k.split(":", 1)[1], e["disease"])] = (
                "contested" if e["contested"] else e["direction"], e["n_papers"],
                e["taxon"])
    return out


def verdicts(G, ref):
    """Decisive pairs only: both sides commit to a direction."""
    ours = our_dirs(G)
    shared = {d for _, d in ref} & {d for _, d in ours}
    out = {}
    for k in set(ref) & set(ours):
        if k[1] not in shared:
            continue
        odir, n, label = ours[k]
        if odir == "contested":
            continue
        out[k] = (odir == ref[k], odir, ref[k], n, label)
    return out


def report(name, records, tax, override, GA, GB):
    ref = reference_dirs(records, tax, override)
    A, B = verdicts(GA, ref), verdicts(GB, ref)
    ra = sum(1 for v in A.values() if v[0]) / len(A) if A else 0
    rb = sum(1 for v in B.values() if v[0]) / len(B) if B else 0
    print(f"\n=== {name} ===")
    print(f"  decisive pairs : {len(A)} -> {len(B)}")
    print(f"  agreement rate : {ra:.3%} -> {rb:.3%}   ({rb - ra:+.3%})")

    added = set(B) - set(A)
    dropped = set(A) - set(B)
    common = set(A) & set(B)
    flipped = [k for k in common if A[k][0] != B[k][0]]

    add_ok = sum(1 for k in added if B[k][0])
    drop_ok = sum(1 for k in dropped if A[k][0])
    print(f"  COVERAGE   : {len(added)} pairs added   "
          f"({add_ok} agree, {len(added) - add_ok} disagree)")
    print(f"  ATTRITION  : {len(dropped)} pairs dropped "
          f"({drop_ok} agreed, {len(dropped) - drop_ok} disagreed)")
    print(f"  CORRECTION : {len(flipped)} pairs present in both changed verdict")

    for k in sorted(added, key=lambda k: -B[k][3]):
        ok, odir, rdir, n, label = B[k]
        print(f"     + {label[:34]:34} {k[1][:26]:26} ours={odir}({n}p) "
              f"ref={rdir}  {'AGREE' if ok else 'disagree'}")
    for k in sorted(dropped, key=lambda k: -A[k][3]):
        ok, odir, rdir, n, label = A[k]
        print(f"     - {label[:34]:34} {k[1][:26]:26} ours={odir}({n}p) "
              f"ref={rdir}  {'agreed' if ok else 'disagreed'}")
    for k in flipped:
        print(f"     ~ {A[k][4][:34]:34} {k[1][:26]:26} "
              f"{A[k][1]}({A[k][3]}p) -> {B[k][1]}({B[k][3]}p)  ref={A[k][2]}")

    # The comparison that is actually apples-to-apples.
    ca = sum(1 for k in common if A[k][0]) / len(common) if common else 0
    cb = sum(1 for k in common if B[k][0]) / len(common) if common else 0
    print(f"  ON THE {len(common)} PAIRS PRESENT IN BOTH: "
          f"{ca:.3%} -> {cb:.3%} ({cb - ca:+.3%})")
    return {"n_before": len(A), "n_after": len(B), "rate_before": ra,
            "rate_after": rb, "added": len(added), "added_agree": add_ok,
            "dropped": len(dropped), "dropped_agree": drop_ok,
            "flipped": len(flipped), "common": len(common),
            "common_rate_before": ca, "common_rate_after": cb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="/tmp/graph_before.json")
    ap.add_argument("--after", default=os.path.join(HERE, "graph.json"))
    a = ap.parse_args()
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy(a.after)
    override = load_species_overrides()
    GA, GB = json.load(open(a.before)), json.load(open(a.after))
    res = {}
    dis, _ = load_disbiome()
    res["disbiome"] = report("Disbiome", dis, tax, override, GA, GB)
    per, _ = load_peryton()
    if per:
        res["peryton"] = report("Peryton", per, tax, override, GA, GB)
    json.dump(res, open(os.path.join(HERE, "species_split_effect.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
