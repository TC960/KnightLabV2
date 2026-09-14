#!/usr/bin/env python3
"""Do clinically adjacent disease nodes behave like one disease, or like several?

The design call this informs
----------------------------
MONDO settles the two containment questions it can see (Alzheimer's under
Dementia, intracerebral hemorrhage under Stroke -- both CONFIRMED) and is silent
exactly where the graph's problem is largest: it carries **no term named "mild
cognitive impairment"**, so the six-node / 71-paper cognitive-decline cluster
cannot be linked by ontology lookup at all. That leaves a genuine design
decision, and a PI making it deserves a number rather than an intuition.

The number this script produces: **within a clinical cluster, do two disease
nodes agree on the direction of a shared taxon more often than two unrelated
disease nodes do?** If they agree at the is-a rate, they are behaving like one
disease and containment (or folding) is recovering real replication. If they
agree at the background rate, they are behaving like distinct diseases and a
containment layer is bookkeeping for retrieval -- still defensible, but not a
signal gain, and it must not be sold as one.

Method notes that matter
------------------------
- **The unit is the disease PAIR, not the observation.** Pooling observations
  across pairs is how this project produced three false positives; the cognitive
  cluster's 15 pairs are 15 clusters, not 200 independent calls.
- **A negative control is included.** Multiple system atrophy and essential
  tremor were explicitly rejected as Parkinson's subtypes, and MONDO upheld both
  rejections. They must NOT cohere. If they do, the metric is measuring
  something other than disease identity.
- **Contested taxa are dropped on both sides**, as in every other agreement
  metric in this repo; only unambiguous calls count.
- Paper overlap was checked and is **zero** for every disease pair in the graph
  (`build_kg.py` files each paper under one predicted disease), so the
  shared-paper inflation that makes the 73% Disbiome figure a blend cannot
  operate here.
"""
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "disease_cluster_coherence.json")

CLUSTERS = {
    # The one the decision is actually about: 71 papers, 0 hierarchy links,
    # and MONDO cannot model it (no MCI term).
    "cognitive-decline": ["Alzheimer's disease", "Mild cognitive impairment",
                          "Dementia", "Cognitive impairment",
                          "Neurocognitive impairment", "Subjective cognitive decline",
                          "Sporadic Creutzfeldt-Jakob disease"],
    "cerebrovascular": ["Stroke", "Intracerebral hemorrhage",
                        "Hypertensive intracerebral hemorrhage",
                        "Poststroke aphasia", "Hemorrhagic transformation"],
    "spinal-cord-injury": ["Spinal cord injury",
                           "Chronic traumatic complete spinal cord injury",
                           "Traumatic thoracic spinal cord injury"],
    "hepatic-encephalopathy": ["Hepatic encephalopathy", "Minimal hepatic encephalopathy"],
}

# NEGATIVE CONTROL. Both rejected as Parkinson's subtypes by the earlier
# session, both rejections upheld by MONDO (cousin / sibling). These pairs must
# look like background; if they cohere, the metric is not measuring what it claims.
NEGATIVE_CONTROL = [
    ("Multiple system atrophy", "Parkinson's disease"),
    ("Essential tremor", "Parkinson's disease"),
    ("Essential tremor", "Multiple system atrophy"),
]


def main():
    g = json.load(open(GRAPH))
    prof = defaultdict(dict)
    dpapers = defaultdict(set)
    for e in g["edges"]:
        up, dn = e["n_up"], e["n_down"]
        prof[e["disease"]][e["taxon_key"]] = ("contested" if (up and dn)
                                              else "enriched" if up else "depleted")
        dpapers[e["disease"]].update(e["papers"])

    def agree(a, b):
        pa, pb = prof.get(a, {}), prof.get(b, {})
        n = k = 0
        for t in set(pa) & set(pb):
            if pa[t] == "contested" or pb[t] == "contested":
                continue
            n += 1
            k += (pa[t] == pb[t])
        return n, k

    in_cluster = {}
    for cname, members in CLUSTERS.items():
        for d in members:
            in_cluster[d] = cname

    all_d = sorted(prof)
    results = {"clusters": {}, "negative_control": [], "background": {}}

    # ---- background: pairs from DIFFERENT clusters (or neither) ----
    bg_pairs = []
    for i, a in enumerate(all_d):
        for b in all_d[i + 1:]:
            ca, cb = in_cluster.get(a), in_cluster.get(b)
            if ca and cb and ca == cb:
                continue                      # within-cluster; not background
            n, k = agree(a, b)
            if n:
                bg_pairs.append((a, b, n, k))
    bn = sum(p[2] for p in bg_pairs)
    bk = sum(p[3] for p in bg_pairs)
    results["background"] = {"n_pairs": len(bg_pairs), "n_obs": bn, "n_agree": bk,
                             "pooled_rate": round(bk / bn, 4)}
    print(f"background (cross-cluster disease pairs): {len(bg_pairs)} pairs, "
          f"{bk}/{bn} = {bk/bn:.3f}\n")

    rng = random.Random(31)

    def pair_null(obs_pairs, label):
        """Draw as many background pairs as the cluster has, pool identically."""
        if not obs_pairs:
            return None
        n_true = sum(p[2] for p in obs_pairs)
        k_true = sum(p[3] for p in obs_pairs)
        if n_true == 0:
            return None
        rate = k_true / n_true
        rates = []
        hits = 0
        for _ in range(20000):
            draw = rng.sample(bg_pairs, min(len(obs_pairs), len(bg_pairs)))
            n = sum(x[2] for x in draw)
            k = sum(x[3] for x in draw)
            if not n:
                continue
            r = k / n
            rates.append(r)
            hits += (r >= rate)
        rates.sort()
        mean = sum(rates) / len(rates)
        sd = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5
        p95 = rates[int(0.95 * len(rates))]
        p = (hits + 1) / (len(rates) + 1)
        print(f"  {label}: {k_true}/{n_true} = {rate:.3f}  vs null {mean:.3f} +/- {sd:.3f}"
              f"  p={p:.4f}  (MDE: needs > {p95:.3f}, i.e. {p95-mean:+.3f})")
        return {"n_pairs": len(obs_pairs), "n_obs": n_true, "n_agree": k_true,
                "rate": round(rate, 4), "null_mean": round(mean, 4),
                "null_sd": round(sd, 4), "null_p95": round(p95, 4),
                "p": round(p, 4), "mde_points": round(p95 - mean, 4)}

    for cname, members in CLUSTERS.items():
        present = [d for d in members if d in prof]
        pairs = []
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                n, k = agree(a, b)
                if n:
                    pairs.append((a, b, n, k))
        npapers = len(set().union(*(dpapers[d] for d in present))) if present else 0
        print(f"=== {cname}: {len(present)} nodes present, {npapers} papers, "
              f"{len(pairs)} pairs with a decisive shared taxon ===")
        for a, b, n, k in sorted(pairs, key=lambda x: -x[2]):
            print(f"    {a[:34]:36s} / {b[:30]:32s} {k:3d}/{n:<3d} = {k/n:.3f}")
        null = pair_null(pairs, "pooled")
        results["clusters"][cname] = {
            "nodes_present": present, "n_papers": npapers,
            "pairs": [{"a": a, "b": b, "n": n, "k": k, "rate": round(k / n, 3)}
                      for a, b, n, k in pairs],
            "null": null,
        }
        print()

    # ---- confound check: do pairs agree merely because they share FAMOUS taxa? ----
    # Two nodes with little overlap can only overlap on the widely-reported
    # genera, where any two neurological cohorts tend to agree. If agreement
    # tracked taxon ubiquity, the whole comparison would be measuring overlap
    # size rather than disease relatedness.
    import statistics as st
    breadth = defaultdict(set)
    for e in g["edges"]:
        breadth[e["taxon_key"]].add(e["disease"])
    bd = {k: len(v) for k, v in breadth.items()}
    pts = []
    ds5 = [d for d in all_d if len(prof[d]) >= 5]
    for i, a in enumerate(ds5):
        for b in ds5[i + 1:]:
            sh = [(t, prof[a][t] == prof[b][t], bd[t]) for t in set(prof[a]) & set(prof[b])
                  if prof[a][t] != "contested" and prof[b][t] != "contested"]
            if len(sh) >= 3:
                pts.append((sum(1 for x in sh if x[1]) / len(sh),
                            sum(x[2] for x in sh) / len(sh), len(sh)))

    def corr(xs, ys):
        mx, my = st.mean(xs), st.mean(ys)
        cv = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
        return cv / (st.pstdev(xs) * st.pstdev(ys))

    r_bd = corr([p[1] for p in pts], [p[0] for p in pts])
    r_n = corr([p[2] for p in pts], [p[0] for p in pts])
    pts.sort(key=lambda p: p[1])
    k3 = len(pts) // 3
    tert = [(nm, st.mean([p[1] for p in grp]), st.mean([p[0] for p in grp]), len(grp))
            for nm, grp in (("low", pts[:k3]), ("mid", pts[k3:2 * k3]), ("high", pts[2 * k3:]))]
    print(f"=== confound: taxon ubiquity ({len(pts)} pairs with >=3 decisive shared taxa) ===")
    print(f"  corr(mean shared-taxon breadth, agreement) = {r_bd:+.3f}")
    print(f"  corr(n shared decisive taxa, agreement)    = {r_n:+.3f}")
    for nm, mb, ma, ng in tert:
        print(f"    {nm:5s} breadth {mb:5.2f} -> agreement {ma:.3f}  (n={ng})")
    print("  => flat. Agreement is not an artifact of which taxa a pair happens to share.\n")
    results["ubiquity_confound"] = {
        "n_pairs": len(pts), "corr_breadth_agreement": round(r_bd, 4),
        "corr_noverlap_agreement": round(r_n, 4),
        "tertiles": [{"bucket": nm, "mean_breadth": round(mb, 3),
                      "mean_agreement": round(ma, 4), "n_pairs": ng}
                     for nm, mb, ma, ng in tert],
    }

    print("=== NEGATIVE CONTROL (rejected as subtypes; must look like background) ===")
    nc = []
    for a, b in NEGATIVE_CONTROL:
        n, k = agree(a, b)
        nc.append({"a": a, "b": b, "n": n, "k": k,
                   "rate": round(k / n, 3) if n else None})
        print(f"    {a[:34]:36s} / {b[:30]:32s} "
              + (f"{k}/{n} = {k/n:.3f}" if n else "no decisive shared taxon"))
    results["negative_control"] = nc
    ncp = [(x["a"], x["b"], x["n"], x["k"]) for x in nc if x["n"]]
    if ncp:
        results["negative_control_null"] = pair_null(ncp, "pooled")

    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
