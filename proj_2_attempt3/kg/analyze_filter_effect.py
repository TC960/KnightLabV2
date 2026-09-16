#!/usr/bin/env python3
"""Did screening the 45 MAIN_DATA papers actually recover agreement?

The naive comparison -- run validate_external.py on each variant and read off the
headline percentage -- is CONFOUNDED, and visibly so: dropping all 45 papers moves
Disbiome's reference denominator from 506 to 364. That happens because
`compare_against` restricts the reference to diseases present in OUR graph, so
removing papers removes whole diseases and the two runs then score different
question sets. A rate computed over a different disease mix is not a recovery.

So this script compares PAIRWISE instead. For each variant it records, per
(taxid, disease), whether we agreed with the curated database. It then restricts
to the pairs that are decisive in BOTH variants being compared -- a fixed question
set -- and asks how many flipped, in which direction.

McNemar's exact test is the right test here: the pairs are matched (same taxon,
same disease, same reference record), and only the discordant ones carry
information about whether filtering helped. Reporting an unpaired difference of
proportions would overstate significance by ignoring that the two samples are
almost entirely the same pairs.
"""
import json
import os
from collections import defaultdict
from math import comb

from validate_external import DISEASE_MAP, OUTCOME, load_disbiome, load_peryton

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["all348", "screened", "no_maindata"]


def per_pair(graph_path, records, tax):
    """-> {(taxid, disease): True if we agree with the reference, else False}"""
    ref = defaultdict(lambda: defaultdict(int))
    for r in records:
        our = DISEASE_MAP.get(r["disease"].lower())
        out = OUTCOME.get(r["outcome"].lower())
        if not (our and out and r["microbe"]):
            continue
        tid, *_ = tax.resolve(r["microbe"])
        if tid:
            ref[(tid, our)][out] += 1

    G = json.load(open(graph_path))
    ours = {}
    for e in G["edges"]:
        k = e["taxon_key"]
        if k.startswith("ncbi:"):
            ours[(k.split(":", 1)[1], e["disease"])] = e

    out = {}
    for k in set(ref) & set(ours):
        e, rv = ours[k], ref[k]
        rdir = ("enriched" if rv["enriched"] > rv["depleted"]
                else "depleted" if rv["depleted"] > rv["enriched"] else "contested")
        odir = "contested" if e["contested"] else e["direction"]
        if odir == "contested" or rdir == "contested":
            continue          # not decisive -> carries no direction information
        out[k] = (odir == rdir)
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(name, A, B, labelA, labelB):
    common = set(A) & set(B)
    a_ok = sum(A[k] for k in common)
    b_ok = sum(B[k] for k in common)
    # discordant: agreed only after (gained) vs agreed only before (lost)
    gained = sum(1 for k in common if B[k] and not A[k])
    lost = sum(1 for k in common if A[k] and not B[k])
    p = mcnemar_exact(lost, gained)
    print(f"\n  {labelA}  ->  {labelB}")
    print(f"    pairs decisive in both : {len(common)}"
          f"   (only in {labelA}: {len(set(A)-set(B))}, only in {labelB}: {len(set(B)-set(A))})")
    print(f"    agreement {labelA:12}: {a_ok}/{len(common)} ({100*a_ok/max(len(common),1):.1f}%)")
    print(f"    agreement {labelB:12}: {b_ok}/{len(common)} ({100*b_ok/max(len(common),1):.1f}%)")
    print(f"    flips: {gained} gained, {lost} lost   McNemar exact p = {p:.3f}")
    if gained + lost == 0:
        print("    -> filtering changed NO decisive pair. Not underpowered: zero effect.")
    return {"source": name, "from": labelA, "to": labelB, "n_common": len(common),
            "agree_from": a_ok, "agree_to": b_ok, "gained": gained, "lost": lost,
            "mcnemar_p": round(p, 4)}


def main():
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy()

    sources = []
    recs, _ = load_disbiome()
    sources.append(("Disbiome", recs))
    recs, _ = load_peryton()
    if recs:
        sources.append(("Peryton", recs))

    results = []
    for name, recs in sources:
        print("=" * 78)
        print(name.upper())
        print("=" * 78)
        pp = {v: per_pair(os.path.join(HERE, f"_graph_{v}.json"), recs, tax)
              for v in VARIANTS}
        for v in VARIANTS:
            n = len(pp[v])
            print(f"  {v:12}: {sum(pp[v].values())}/{n} decisive pairs agree "
                  f"({100*sum(pp[v].values())/max(n,1):.1f}%)")
        results.append(compare(name, pp["all348"], pp["screened"], "all348", "screened"))
        results.append(compare(name, pp["all348"], pp["no_maindata"], "all348", "no_maindata"))
        results.append(compare(name, pp["screened"], pp["no_maindata"], "screened", "no_maindata"))
        print()

    json.dump(results, open(os.path.join(HERE, "filter_effect.json"), "w"), indent=1)
    print("wrote filter_effect.json")


if __name__ == "__main__":
    main()
