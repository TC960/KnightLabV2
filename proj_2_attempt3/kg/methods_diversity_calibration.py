#!/usr/bin/env python3
"""Does methodological diversity of an edge's evidence predict external agreement?

`build_kg.py` now labels every edge `single-method` (its supporting papers ran the
same pipeline and the same extraction kit) or `multi-method` (they did not). The
motivation is that six papers agreeing through one instrument are not six
independent measurements, and a reader judging an edge wants to know which they
are looking at.

That is a claim about evidential independence, and this repo does not ship claims
like that unmeasured -- `annotate_confidence` earns its tiers against Disbiome and
Peryton, and so should this. If multi-method edges agree with curated databases
more often than single-method edges of the SAME evidence weight, diversity is a
calibration signal on top of evidence count. If not, it is provenance only, and
this file says so.

THE CONFOUND, and it is fatal if ignored. `multi-method` requires at least two
papers with recoverable methods, and more papers means more chances to differ, so
diversity is correlated with evidence count -- which is already the strongest
known predictor of agreement (1 paper 65.8% -> >=3 papers 91.7%). Every comparison
here is therefore made WITHIN a fixed paper-count stratum, and the pooled figure is
reported only alongside the stratified one.

Null is `calibrate_agreement.block_permutation`: agreement labels shuffled in whole
TAXON blocks, so pairs sharing a taxon are not treated as independent.

Writes methods_diversity_calibration.json.
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict

from calibrate_agreement import GRAPH, block_permutation, build_pairs, rate
from validate_external import load_disbiome, load_peryton

ITERS = 10000
SEED = 42


def annotate(pairs, graph):
    idx = {(e["taxon_key"], e["disease"]): e for e in graph["edges"]}
    out = []
    for p in pairs:
        e = idx.get((p["taxon_key"], p["disease"]))
        if not e:
            continue
        out.append({**p,
                    "methods_diversity": e.get("methods_diversity", "unknown"),
                    "n_pipelines": e.get("n_pipelines", 0),
                    "n_kits": e.get("n_kits", 0),
                    "n_methods_papers": e.get("n_methods_papers", 0)})
    return out


def report(name, pairs, rng):
    print(f"\n=== {name} ({len(pairs)} decisive pairs) ===")
    c = Counter(p["methods_diversity"] for p in pairs)
    print(f"  composition: {dict(c)}")

    res = {"source": name, "n_pairs": len(pairs), "composition": dict(c)}

    known = [p for p in pairs if p["methods_diversity"] != "unknown"]
    if len(known) < 20:
        print("  too few edges with a diversity label to test")
        res["testable"] = False
        return res
    res["testable"] = True

    multi = [p for p in known if p["methods_diversity"] == "multi-method"]
    single = [p for p in known if p["methods_diversity"] == "single-method"]
    print(f"  POOLED  multi-method {rate(multi):.1%} (n={len(multi)})  vs  "
          f"single-method {rate(single):.1%} (n={len(single)})")
    p_pooled, _, obs = block_permutation(
        known, lambda x: x["methods_diversity"], ITERS, rng)
    print(f"          spread {obs:.3f}, taxon-block permutation p = {p_pooled:.4f}")
    res["pooled"] = {"multi_rate": round(rate(multi), 4), "n_multi": len(multi),
                     "single_rate": round(rate(single), 4), "n_single": len(single),
                     "p": round(p_pooled, 5)}

    # the confound, made visible
    print(f"  evidence weight: multi-method mean {sum(p['our_n_papers'] for p in multi)/len(multi):.2f} "
          f"papers vs single-method "
          f"{sum(p['our_n_papers'] for p in single)/len(single):.2f}")
    res["mean_papers"] = {
        "multi": round(sum(p["our_n_papers"] for p in multi) / len(multi), 3),
        "single": round(sum(p["our_n_papers"] for p in single) / len(single), 3)}

    # stratified: within each paper-count band
    print("  STRATIFIED on evidence count:")
    strata = {}
    for lo, hi, lab in ((2, 2, "2 papers"), (3, 4, "3-4 papers"),
                        (5, 99, ">=5 papers")):
        band = [p for p in known if lo <= p["our_n_papers"] <= hi]
        m = [p for p in band if p["methods_diversity"] == "multi-method"]
        s = [p for p in band if p["methods_diversity"] == "single-method"]
        if len(m) < 3 or len(s) < 3:
            print(f"    {lab:<12} multi n={len(m)}, single n={len(s)} "
                  f"-- too thin to test")
            strata[lab] = {"n_multi": len(m), "n_single": len(s),
                           "testable": False}
            continue
        pv, _, sp = block_permutation(band, lambda x: x["methods_diversity"],
                                      ITERS, rng)
        print(f"    {lab:<12} multi {rate(m):.1%} (n={len(m)})  vs  "
              f"single {rate(s):.1%} (n={len(s)})   p = {pv:.4f}")
        strata[lab] = {"multi_rate": round(rate(m), 4), "n_multi": len(m),
                       "single_rate": round(rate(s), 4), "n_single": len(s),
                       "p": round(pv, 5), "testable": True}
    res["strata"] = strata
    return res


def main():
    rng = random.Random(SEED)
    graph = json.load(open(GRAPH))
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy()
    if not tax.ok:
        print("FATAL: no taxonomy available.")
        return 1

    out = {"iters": ITERS, "seed": SEED, "sources": []}
    for loader in (load_disbiome, load_peryton):
        recs, nm = loader()
        if recs is None:
            continue
        pairs = annotate(build_pairs(graph, recs, tax), graph)
        out["sources"].append(report(nm, pairs, rng))

    with open("methods_diversity_calibration.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote methods_diversity_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
