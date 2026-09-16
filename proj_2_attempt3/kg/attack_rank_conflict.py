#!/usr/bin/env python3
"""Adversarial attacks on the Q2 result from analyze_rank_conflict.py.

Q2 came back at +0.354, z=15.5. On this corpus a z that size is a reason for
suspicion, not celebration: the +0.047 five-sigma co-occurrence result of
2026-09-03 was 12 duplicate papers, and two earlier "findings" died to a
permutation test. Four things could manufacture it.

A. PIPELINE SELF-AGREEMENT. 100 placeholder nodes ("Prevotella 9") were SPLIT
   out of their parents by this pipeline, and the containment link to the parent
   was created by the same operation. If a paper's single mention produced both
   nodes, the two "measurements" are one measurement and must agree. Attack:
   drop every pair touching a placeholder and re-test on named NCBI taxa only.

B. ONE LOUD PAPER. 474 related pairs come from only ~102 paper x disease units;
   a single paper reporting a family and six of its genera contributes 6 pairs
   that are not independent. Attack: cluster-robust statistic -- compute the gap
   per unit, average across units so every paper weighs the same, and permute.

C. DIRECTION SKEW. If related taxa happen to sit among a paper's majority
   direction, they agree for free. The within-paper shuffle already controls
   this, but report the marginal so the reader can see it.

D. SIBLINGS, NOT ANCESTRY. If agreement is really "taxa named in the same
   sentence agree", then unrelated pairs at the same rank should show it too.
   Attack: compare against same-rank unrelated pairs specifically.

Usage: python3 attack_rank_conflict.py [--iters 20000]
"""
import argparse
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def build(exclude_placeholder):
    G = json.load(open(os.path.join(HERE, "graph.json")))
    rank = {n["id"]: (n.get("rank") or "") for n in G["nodes"]}
    kin = set()
    for h in G["hierarchy"]:
        if exclude_placeholder and (h["parent"].startswith("t:ph:")
                                    or h["child"].startswith("t:ph:")):
            continue
        kin.add((h["parent"], h["child"]))
        kin.add((h["child"], h["parent"]))

    calls = defaultdict(lambda: defaultdict(dict))
    for e in G["edges"]:
        if exclude_placeholder and e["source"].startswith("t:ph:"):
            continue
        for ev in e["ev"]:
            calls[ev["i"]][e["disease"]][e["source"]] = ev["d"]

    units = []
    for pi, dmap in calls.items():
        for dis, nd in dmap.items():
            nodes = sorted(nd)
            if len(nodes) < 2:
                continue
            rel, unrel, unrel_same_rank = [], [], []
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    if (nodes[a], nodes[b]) in kin:
                        rel.append((a, b))
                    else:
                        unrel.append((a, b))
                        if rank[nodes[a]] and rank[nodes[a]] == rank[nodes[b]]:
                            unrel_same_rank.append((a, b))
            if rel:
                units.append({"dirs": [nd[n] for n in nodes], "rel": rel,
                              "unrel": unrel, "unrel_sr": unrel_same_rank, "paper": pi})
    return units


def pooled(units, key="unrel"):
    ra = rn = ua = un = 0
    for u in units:
        d = u["dirs"]
        for a, b in u["rel"]:
            rn += 1; ra += d[a] == d[b]
        for a, b in u[key]:
            un += 1; ua += d[a] == d[b]
    return (ra / rn if rn else 0) - (ua / un if un else 0), rn, un


def clustered(units, key="unrel"):
    """One number per paper x disease unit, then the unweighted mean of those."""
    gaps = []
    for u in units:
        d = u["dirs"]
        if not u["rel"] or not u[key]:
            continue
        r = sum(d[a] == d[b] for a, b in u["rel"]) / len(u["rel"])
        v = sum(d[a] == d[b] for a, b in u[key]) / len(u[key])
        gaps.append(r - v)
    return (sum(gaps) / len(gaps) if gaps else 0), len(gaps)


def permute(units, statfn, iters, key, seed=42):
    obs = statfn(units, key)[0]
    rng = random.Random(seed)
    null = []
    ge = 0
    for _ in range(iters):
        sh = []
        for u in units:
            d2 = u["dirs"][:]
            rng.shuffle(d2)                     # preserves this paper's up/down counts
            sh.append({**u, "dirs": d2})
        g = statfn(sh, key)[0]
        null.append(g)
        ge += g >= obs
    m = sum(null) / len(null)
    sd = (sum((x - m) ** 2 for x in null) / len(null)) ** 0.5
    return {"obs": round(obs, 4), "null_mean": round(m, 5), "null_sd": round(sd, 5),
            "z": round((obs - m) / sd, 2) if sd else None,
            "p": round((ge + 1) / (iters + 1), 5), "mde": round(1.96 * sd, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    a = ap.parse_args()
    res = {}

    for tag, excl in (("all_links", False), ("named_only__attack_A", True)):
        units = build(excl)
        pg, nrel, nunrel = pooled(units)
        cg, nclust = clustered(units)
        res[tag] = {
            "n_units": len(units), "n_related_pairs": nrel, "n_unrelated_pairs": nunrel,
            "pooled": permute(units, pooled, a.iters, "unrel"),
            "clustered__attack_B": {**permute(units, clustered, a.iters, "unrel"),
                                    "n_units_contributing": nclust},
            "vs_same_rank_unrelated__attack_D": permute(units, pooled, a.iters, "unrel_sr"),
        }
        # attack C -- how skewed is each paper's direction mix?
        skew = [max(u["dirs"].count("e"), u["dirs"].count("d")) / len(u["dirs"]) for u in units]
        res[tag]["attack_C_mean_direction_skew"] = round(sum(skew) / len(skew), 3)
        # how concentrated are the related pairs across papers?
        per_paper = defaultdict(int)
        for u in units:
            per_paper[u["paper"]] += len(u["rel"])
        top = sorted(per_paper.values(), reverse=True)
        res[tag]["pairs_from_top_paper"] = top[0]
        res[tag]["pairs_from_top_5_papers"] = sum(top[:5])
        res[tag]["n_papers_contributing"] = len(per_paper)

    print(json.dumps(res, indent=1))
    p = os.path.join(HERE, "rank_conflict_attacks.json")
    json.dump(res, open(p, "w"), indent=1, sort_keys=True)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
