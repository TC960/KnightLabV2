#!/usr/bin/env python3
"""Which of the calibration signals survive controlling for the others?

`calibrate_agreement.py` found three candidate predictors of agreement with
curated databases. Two of them could easily be the same fact wearing different
hats:

  evidence count   1-paper edges agree ~64%, >=3-paper edges ~91%.
  specificity      "discriminating" taxa (purity <= 0.6 -- the taxon points
                   different ways in different diseases) agree ~38%, versus
                   ~80% for taxa with a consistent lean.
  rank             species-level edges agree ~91%, genus-level ~68%.

A discriminating taxon might simply be one measured by fewer papers per disease,
and a species-level edge might be one that only well-powered studies report. So
each is re-tested INSIDE strata of the others. A signal that vanishes when
stratified is a restatement of the variable it was confounded with, and this
project has shipped that mistake before.

Multiple testing is corrected across the whole family (Benjamini-Hochberg), and
every test is reported with its minimum detectable effect, because at 174 and
138 pairs most of these cannot resolve a small difference and saying so is the
result.

    python calibrate_confounds.py
"""
import argparse
import json
import os
import random
from collections import defaultdict

from calibrate_agreement import (build_pairs, cluster_bootstrap, rate)
from validate_external import GRAPH, load_disbiome, load_peryton

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "calibration_confounds.json")


def block_perm_two(pairs, hi_fn, iters, rng):
    """p and null sd for a two-group rate difference, shuffling TAXON blocks.

    One-sided on |difference| is wrong when the direction is the question, so
    the statistic is the signed difference and the p-value is two-sided.
    """
    hi = [hi_fn(p) for p in pairs]
    if len(set(hi)) < 2:
        return None
    labels = [p["agree"] for p in pairs]
    obs = _diff(hi, labels)

    blocks = defaultdict(list)
    for p in pairs:
        blocks[p["taxon_key"]].append(p["agree"])
    blk = list(blocks.values())

    null = []
    for _ in range(iters):
        rng.shuffle(blk)
        flat = [x for b in blk for x in b]
        d = _diff(hi, flat)
        if d == d:
            null.append(d)
    if not null:
        return None
    hits = sum(1 for d in null if abs(d) >= abs(obs))
    mean = sum(null) / len(null)
    sd = (sum((x - mean) ** 2 for x in null) / len(null)) ** 0.5
    return {"diff": obs, "p": (hits + 1) / (len(null) + 1), "null_sd": sd,
            "mde": 1.96 * sd,
            "n_hi": sum(hi), "n_lo": len(hi) - sum(hi)}


def _diff(hi, labels):
    a = [l for h, l in zip(hi, labels) if h]
    b = [l for h, l in zip(hi, labels) if not h]
    if not a or not b:
        return float("nan")
    return sum(a) / len(a) - sum(b) / len(b)


def bh(pvals):
    """Benjamini-Hochberg q-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n, q, prev = len(pvals), [0.0] * len(pvals), 1.0
    for rank, i in enumerate(reversed(idx), 1):
        j = n - rank + 1
        prev = min(prev, pvals[i] * n / j)
        q[i] = prev
    return q


HI = {
    "evidence >=2 papers": lambda p: p["our_n_papers"] >= 2,
    "not discriminating": lambda p: p["taxon_class"] != "discriminating",
    "species rank": lambda p: p["rank"] == "species",
}

STRATA = {
    "evidence": [("1 paper", lambda p: p["our_n_papers"] == 1),
                 (">=2 papers", lambda p: p["our_n_papers"] >= 2)],
    "specificity": [("discriminating", lambda p: p["taxon_class"] == "discriminating"),
                    ("not discriminating", lambda p: p["taxon_class"] != "discriminating")],
    "rank": [("species", lambda p: p["rank"] == "species"),
             ("not species", lambda p: p["rank"] != "species")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    G = json.load(open(GRAPH))
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy()

    sources = {}
    for loader in (load_disbiome, load_peryton):
        recs, nm = loader()
        if recs is not None:
            sources[nm] = build_pairs(G, recs, tax)

    out = {"iters": a.iters, "seed": a.seed, "marginal": [], "stratified": [],
           "composition": []}
    pv, tags = [], []

    # ---- 1. are the three predictors even distinguishable in this data? -----
    print("=" * 78)
    print("COMPOSITION -- is 'discriminating' just low evidence in disguise?")
    print("=" * 78)
    for nm, pairs in sources.items():
        for cls in ("discriminating", "not discriminating"):
            sel = [p for p in pairs if (p["taxon_class"] == "discriminating")
                   == (cls == "discriminating")]
            if not sel:
                continue
            mean_np = sum(p["our_n_papers"] for p in sel) / len(sel)
            frac1 = sum(p["our_n_papers"] == 1 for p in sel) / len(sel)
            print(f"  {nm:9} {cls:20} n={len(sel):<4} "
                  f"mean papers/edge {mean_np:.2f}   single-paper {100*frac1:.0f}%")
            out["composition"].append({"source": nm, "group": cls, "n": len(sel),
                                       "mean_n_papers": round(mean_np, 3),
                                       "frac_single_paper": round(frac1, 3)})
        sp = [p for p in pairs if p["rank"] == "species"]
        ns = [p for p in pairs if p["rank"] != "species"]
        print(f"  {nm:9} {'species':20} n={len(sp):<4} "
              f"mean papers/edge {sum(p['our_n_papers'] for p in sp)/max(len(sp),1):.2f}")
        print(f"  {nm:9} {'not species':20} n={len(ns):<4} "
              f"mean papers/edge {sum(p['our_n_papers'] for p in ns)/max(len(ns),1):.2f}")

    # ---- 2. marginal effects -----------------------------------------------
    print("\n" + "=" * 78)
    print("MARGINAL -- each predictor on its own")
    print("=" * 78)
    for nm, pairs in sources.items():
        for label, fn in HI.items():
            r = block_perm_two(pairs, fn, a.iters, rng)
            if not r:
                continue
            ci = cluster_bootstrap(pairs, fn, 4000, rng)
            row = {"source": nm, "predictor": label, "stratum": "(all)",
                   "n_hi": r["n_hi"], "n_lo": r["n_lo"],
                   "diff": round(r["diff"], 4), "p": round(r["p"], 4),
                   "mde": round(r["mde"], 4)}
            if ci:
                row["ci"] = [round(ci[0], 4), round(ci[1], 4)]
            print(f"  {nm:9} {label:22} {100*r['diff']:+6.1f} pts "
                  f"(n {r['n_hi']}/{r['n_lo']})  p={r['p']:.4f}  "
                  f"MDE {100*r['mde']:.1f} pts"
                  + (f"  CI [{100*ci[0]:+.1f},{100*ci[1]:+.1f}]" if ci else ""))
            out["marginal"].append(row)
            pv.append(r["p"])
            tags.append(row)

    # ---- 3. stratified ------------------------------------------------------
    print("\n" + "=" * 78)
    print("STRATIFIED -- each predictor inside levels of the others")
    print("=" * 78)
    for nm, pairs in sources.items():
        for label, fn in HI.items():
            key = ("evidence" if "evidence" in label
                   else "specificity" if "discriminating" in label else "rank")
            for sname, sfn in [(s, f) for k, v in STRATA.items() if k != key
                               for s, f in v]:
                sub = [p for p in pairs if sfn(p)]
                if len(sub) < 15:
                    continue
                r = block_perm_two(sub, fn, a.iters, rng)
                if not r or r["n_hi"] < 4 or r["n_lo"] < 4:
                    continue
                row = {"source": nm, "predictor": label, "stratum": sname,
                       "n_hi": r["n_hi"], "n_lo": r["n_lo"],
                       "diff": round(r["diff"], 4), "p": round(r["p"], 4),
                       "mde": round(r["mde"], 4)}
                print(f"  {nm:9} {label:22} within {sname:20} "
                      f"{100*r['diff']:+6.1f} pts (n {r['n_hi']}/{r['n_lo']})  "
                      f"p={r['p']:.4f}  MDE {100*r['mde']:.1f}")
                out["stratified"].append(row)
                pv.append(r["p"])
                tags.append(row)

    # ---- 4. BH across the whole family -------------------------------------
    qs = bh(pv)
    for row, q in zip(tags, qs):
        row["q_bh"] = round(q, 4)
    print("\n" + "=" * 78)
    print(f"BENJAMINI-HOCHBERG across all {len(pv)} tests -- q < 0.05 survives")
    print("=" * 78)
    for row, q in sorted(zip(tags, qs), key=lambda t: t[1]):
        mark = "SURVIVES" if q < 0.05 else "        "
        print(f"  {mark}  q={q:.4f}  {row['source']:9} {row['predictor']:22} "
              f"within {row['stratum']:20} {100*row['diff']:+6.1f} pts")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
