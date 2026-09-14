#!/usr/bin/env python3
"""Is the is-a agreement signal real, or is it two clusters?

`disease_hierarchy.py` reports is-a disease pairs agreeing on direction 34/38 =
89.5% against 69.2% for ontologically distant pairs. Taken at face value that is
a 20-point effect. It is not evidence, and this script is the arithmetic showing
why: those 38 observations come from exactly **2** disease pairs (Alzheimer's
under Dementia, intracerebral hemorrhage under Stroke). The unit of independence
is the PAIR, not the observation -- the same mistake the project has on record
three times over (pair-level shuffling produced three false positives; the
`diet_controlled` result went p=0.002 -> FDR 0.243 once clustering was handled).

So: cluster the test at the pair level, and report the minimum detectable effect
rather than a p-value dressed up as a finding.
"""
import json
import os
import random
from collections import defaultdict

from mondo import Mondo

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "disease_hierarchy_power.json")


def main():
    m = Mondo()
    g = json.load(open(GRAPH))
    res = json.load(open(os.path.join(HERE, "disease_hierarchy.json")))
    resolved = {lb: r["mondo"] for lb, r in res["resolution"].items() if r["mondo"]}

    prof, epapers = defaultdict(dict), defaultdict(dict)
    for e in g["edges"]:
        up, dn = e["n_up"], e["n_down"]
        prof[e["disease"]][e["taxon_key"]] = ("contested" if (up and dn)
                                              else "enriched" if up else "depleted")
        epapers[e["disease"]][e["taxon_key"]] = set(e["papers"])

    ds = [d for d in resolved if len(prof[d]) >= 5]

    def agree(a, b):
        pa, pb = prof[a], prof[b]
        n = k = 0
        for t in set(pa) & set(pb):
            if pa[t] == "contested" or pb[t] == "contested":
                continue
            n += 1
            k += (pa[t] == pb[t])
        return n, k

    # ---- every pair, bucketed, with the PAIR as the unit ----
    pairs = defaultdict(list)     # relation -> [(a,b,n,k)]
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            rel = m.relation(resolved[a], resolved[b])
            rel = {"ancestor": "is-a", "descendant": "is-a"}.get(rel, rel)
            n, k = agree(a, b)
            if n:
                pairs[rel].append((a, b, n, k))

    print("bucket      pairs  obs   pooled   mean-of-pair-rates")
    summary = {}
    for rel in ("is-a", "sibling", "cousin", "unrelated"):
        P = pairs.get(rel, [])
        if not P:
            print(f"{rel:11s} {0:5d}")
            summary[rel] = {"n_pairs": 0}
            continue
        n = sum(p[2] for p in P)
        k = sum(p[3] for p in P)
        rates = [p[3] / p[2] for p in P]
        mean_r = sum(rates) / len(rates)
        print(f"{rel:11s} {len(P):5d} {n:5d}  {k/n:.3f}    {mean_r:.3f}")
        summary[rel] = {"n_pairs": len(P), "n_obs": n, "n_agree": k,
                        "pooled_rate": round(k / n, 4),
                        "mean_pair_rate": round(mean_r, 4),
                        "pairs": [{"a": a, "b": b, "n": nn, "k": kk,
                                   "rate": round(kk / nn, 3)} for a, b, nn, kk in P]}

    print("\nthe is-a bucket in full:")
    for a, b, n, k in pairs.get("is-a", []):
        print(f"   {a} / {b}: {k}/{n} = {k/n:.3f}")

    # ---- pair-level permutation: draw as many random pairs as there are is-a
    # pairs, from the pairs that are NOT is-a, and pool them the same way. ----
    isa = pairs.get("is-a", [])
    pool = [p for rel, P in pairs.items() if rel != "is-a" for p in P]
    out = {"summary": summary}
    if isa and pool:
        n_true = sum(p[2] for p in isa)
        k_true = sum(p[3] for p in isa)
        rate_true = k_true / n_true
        rng = random.Random(29)
        N = 20000
        hits = 0
        rates = []
        for _ in range(N):
            draw = rng.sample(pool, len(isa))
            n = sum(p[2] for p in draw)
            k = sum(p[3] for p in draw)
            if not n:
                continue
            r = k / n
            rates.append(r)
            hits += (r >= rate_true)
        rates.sort()
        mean = sum(rates) / len(rates)
        sd = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5
        p = (hits + 1) / (len(rates) + 1)
        p95 = rates[int(0.95 * len(rates))]
        print(f"\npair-clustered null ({len(isa)} pairs drawn from {len(pool)} non-is-a pairs, "
              f"{N} draws):")
        print(f"  observed  {k_true}/{n_true} = {rate_true:.3f}")
        print(f"  null      {mean:.3f} +/- {sd:.3f}   95th pct {p95:.3f}")
        print(f"  p = {p:.4f}")
        print(f"  => a true is-a rate would have to exceed {p95:.3f} to clear the null "
              f"at this cluster count. MDE = {p95 - mean:+.3f} on a {mean:.3f} base.")
        out["pair_clustered_null"] = {
            "n_isa_pairs": len(isa), "n_pool_pairs": len(pool),
            "n_obs": n_true, "n_agree": k_true, "rate": round(rate_true, 4),
            "null_mean": round(mean, 4), "null_sd": round(sd, 4),
            "null_p95": round(p95, 4), "p": round(p, 4),
            "mde_points": round(p95 - mean, 4), "draws": len(rates),
        }

        # How many is-a pairs WOULD be needed to resolve a 20-point effect?
        need = None
        for npairs in range(2, len(pool) + 1):
            rs = []
            for _ in range(2000):
                draw = rng.sample(pool, npairs)
                n = sum(x[2] for x in draw)
                k = sum(x[3] for x in draw)
                if n:
                    rs.append(k / n)
            rs.sort()
            if rs and rs[int(0.95 * len(rs))] < mean + 0.20:
                need = npairs
                break
        if need:
            print(f"  => {need} is-a disease pairs would put a 20-point effect above "
                  f"the null's 95th percentile. The graph has {len(isa)}.")
            out["pair_clustered_null"]["pairs_needed_for_20pt_effect"] = need

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
