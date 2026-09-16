#!/usr/bin/env python3
"""Is agreement with curated databases predictable? Calibrate the graph.

The headline validation numbers (Disbiome 73.0%, Peryton 72.5%) are a single
pooled rate over every decisive pair. That pooling hides the question a reader
actually has, which is not "how good is the graph" but "which edges can I
trust". This script asks whether any property of an edge -- evidence count,
rank, disease specificity, whether it restates a taxonomic parent -- predicts
whether it agrees with an independent curation. It also asks the mirror-image
question nobody has asked here: does the *reference's* evidence depth predict
agreement, i.e. how much of the 27% disagreement is our error versus a curated
entry resting on a single paper.

The join is the one from validate_external.py, imported rather than copied so
the two cannot drift: both sides pass through kg/taxonomy.py, never through a
stored taxid (see that file's header for why).

STATISTICS. Pairs are not independent -- one taxon appears in many diseases and
one paper backs many edges -- so a plain binomial test on 174 pairs would
overstate significance, which is exactly how this corpus has produced false
positives before. Two guards, both reported:

  block permutation   agreement labels are shuffled in whole TAXON blocks, so
                      within-taxon correlation is preserved while the link
                      between the predictor and agreement is broken.
  cluster bootstrap   taxa are resampled with replacement; the CI on the
                      difference is read off the resampled distribution.

A result is reported as real only if the permutation p is small AND the
bootstrap CI excludes zero. Everything else is reported as a null with its
minimum detectable effect, per the project's standing rule.

    python calibrate_agreement.py
    python calibrate_agreement.py --iters 20000
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

from validate_external import (DISEASE_MAP, OUTCOME, GRAPH, load_disbiome,
                               load_peryton, paper_keys)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "calibration.json")


def build_pairs(G, records, tax):
    """The validate_external join, but keeping per-pair detail instead of a count.

    Returns one record per pair that is DECISIVE ON BOTH SIDES -- the same set
    the headline agreement rate is computed over, so the rates here reproduce it.
    """
    ref = defaultdict(Counter)
    ref_pmids = defaultdict(set)
    for r in records:
        our = DISEASE_MAP.get(r["disease"].lower())
        out = OUTCOME.get(r["outcome"].lower())
        if not (our and out and r["microbe"]):
            continue
        tid, _sci, _rank, _how = tax.resolve(r["microbe"]) if tax.ok else (None, 0, 0, 0)
        if not tid:
            continue
        ref[(tid, our)][out] += 1
        ref_pmids[(tid, our)] |= paper_keys(pmid=r.get("pmid"), doi=r.get("doi"),
                                            title=r.get("ptitle"))

    ours = {}
    for e in G["edges"]:
        k = e["taxon_key"]
        if k.startswith("ncbi:"):
            ours[(k.split(":", 1)[1], e["disease"])] = e

    shared = {d for _, d in ref} & {d for _, d in ours}
    ref_s = {k: v for k, v in ref.items() if k[1] in shared}
    our_s = {k: v for k, v in ours.items() if k[1] in shared}

    pairs = []
    for k in set(ref_s) & set(our_s):
        e, rv = our_s[k], ref_s[k]
        rdir = ("enriched" if rv["enriched"] > rv["depleted"]
                else "depleted" if rv["depleted"] > rv["enriched"] else "contested")
        odir = "contested" if e["contested"] else e["direction"]
        if odir == "contested" or rdir == "contested":
            continue
        pairs.append({
            "taxon": e["taxon"],
            "taxon_key": e["taxon_key"],
            "disease": e["disease"],
            "agree": odir == rdir,
            "our_dir": odir,
            "ref_dir": rdir,
            "our_n_papers": e["n_papers"],
            "ref_n": rv["enriched"] + rv["depleted"],
            "ref_n_dominant": rv[rdir],
            "rank": e.get("rank") or "unranked",
            "taxon_class": e.get("taxon_class") or "unknown",
            "restates_prior": bool(e.get("restates_prior")),
            "rank_conflict": bool(e.get("has_within_paper_conflict")),
            "papers": e["papers"],
            "ref_keys": sorted(ref_pmids.get(k, ())),
        })
    return pairs


# ---------------------------------------------------------------- statistics

def rate(rows):
    return sum(r["agree"] for r in rows) / len(rows) if rows else float("nan")


def block_permutation(pairs, group_fn, iters, rng):
    """p-value for 'group membership predicts agreement', TAXON-block shuffled.

    The agreement labels are permuted in whole taxon blocks: blocks keep their
    internal label pattern and only their placement is randomised. That breaks
    the group/agreement association without pretending pairs sharing a taxon are
    independent observations.
    """
    groups = [group_fn(p) for p in pairs]
    obs = _spread(groups, [p["agree"] for p in pairs])
    if obs != obs:  # nan -- a group is empty
        return float("nan"), float("nan"), 0

    blocks = defaultdict(list)
    for p in pairs:
        blocks[p["taxon_key"]].append(p["agree"])
    block_labels = list(blocks.values())

    hits, null = 0, []
    for _ in range(iters):
        rng.shuffle(block_labels)
        flat = [lab for blk in block_labels for lab in blk]
        s = _spread(groups, flat)
        null.append(s)
        if s >= obs:
            hits += 1
    mean = sum(null) / len(null)
    sd = (sum((x - mean) ** 2 for x in null) / len(null)) ** 0.5
    return (hits + 1) / (iters + 1), sd, obs


def _spread(groups, labels):
    """Max minus min agreement rate across groups -- one statistic for any arity."""
    agg = defaultdict(lambda: [0, 0])
    for g, lab in zip(groups, labels):
        agg[g][0] += lab
        agg[g][1] += 1
    rates = [a / n for a, n in agg.values() if n]
    return max(rates) - min(rates) if len(rates) > 1 else float("nan")


def cluster_bootstrap(pairs, hi_fn, iters, rng):
    """CI on (rate among hi) - (rate among lo), resampling TAXA with replacement."""
    by_taxon = defaultdict(list)
    for p in pairs:
        by_taxon[p["taxon_key"]].append(p)
    taxa = list(by_taxon)

    diffs = []
    for _ in range(iters):
        boot = []
        for _ in range(len(taxa)):
            boot.extend(by_taxon[taxa[rng.randrange(len(taxa))]])
        hi = [p for p in boot if hi_fn(p)]
        lo = [p for p in boot if not hi_fn(p)]
        if hi and lo:
            diffs.append(rate(hi) - rate(lo))
    diffs.sort()
    if len(diffs) < 100:
        return None
    return (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))],
            sum(diffs) / len(diffs))


def report(name, pairs, iters, rng):
    print("=" * 78)
    print(f"{name}  --  {len(pairs)} decisive pairs, "
          f"{len({p['taxon_key'] for p in pairs})} distinct taxa, "
          f"pooled agreement {100*rate(pairs):.1f}%")
    print("=" * 78)

    res = {"source": name, "n_pairs": len(pairs),
           "n_taxa": len({p["taxon_key"] for p in pairs}),
           "pooled": round(rate(pairs), 4), "tests": []}

    tests = [
        ("our evidence (n_papers)", lambda p: "1" if p["our_n_papers"] == 1
         else "2" if p["our_n_papers"] == 2 else ">=3",
         lambda p: p["our_n_papers"] >= 3),
        ("their evidence (n_records)", lambda p: "1" if p["ref_n"] == 1
         else "2" if p["ref_n"] == 2 else ">=3",
         lambda p: p["ref_n"] >= 3),
        ("rank", lambda p: p["rank"], None),
        ("disease specificity", lambda p: p["taxon_class"], None),
        ("restates a parent", lambda p: str(p["restates_prior"]), None),
        ("within-paper rank conflict", lambda p: str(p["rank_conflict"]), None),
    ]

    for label, gfn, hifn in tests:
        agg = defaultdict(lambda: [0, 0])
        for p in pairs:
            agg[gfn(p)][0] += p["agree"]
            agg[gfn(p)][1] += 1
        cells = sorted(agg.items(), key=lambda kv: -kv[1][1])
        if len(cells) < 2:
            continue
        p_perm, null_sd, obs = block_permutation(pairs, gfn, iters, rng)
        print(f"\n  {label}")
        for g, (a, n) in cells:
            print(f"    {str(g)[:26]:27} {a:>3}/{n:<3}  {100*a/n:>5.1f}%")
        # min detectable spread: what the taxon-block null produces by chance
        mde = 1.96 * null_sd if null_sd == null_sd else float("nan")
        print(f"    spread {100*obs:.1f} pts   block-perm p = {p_perm:.4f}"
              f"   (chance spread ~{100*mde:.1f} pts)")

        entry = {"predictor": label, "p_block_perm": round(p_perm, 4),
                 "observed_spread": round(obs, 4),
                 "min_detectable_spread": round(mde, 4) if mde == mde else None,
                 "cells": {str(g): {"agree": a, "n": n, "rate": round(a / n, 4)}
                           for g, (a, n) in cells}}
        if hifn:
            ci = cluster_bootstrap(pairs, hifn, max(2000, iters // 5), rng)
            if ci:
                lo, hi, mean = ci
                entry["bootstrap_diff"] = {"mean": round(mean, 4),
                                           "ci_lo": round(lo, 4),
                                           "ci_hi": round(hi, 4)}
                excl = "EXCLUDES 0" if lo > 0 or hi < 0 else "includes 0"
                print(f"    cluster-bootstrap diff (hi-lo) = {100*mean:+.1f} pts, "
                      f"95% CI [{100*lo:+.1f}, {100*hi:+.1f}] -- {excl}")
        res["tests"].append(entry)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--graph", default=GRAPH)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    G = json.load(open(a.graph))
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy()
    if not tax.ok:
        print("FATAL: no taxonomy available.")
        return

    out = {"graph": os.path.basename(a.graph), "iters": a.iters,
           "seed": a.seed, "sources": []}
    all_pairs = {}
    for loader in (load_disbiome, load_peryton):
        recs, nm = loader()
        if recs is None:
            continue
        pairs = build_pairs(G, recs, tax)
        all_pairs[nm] = pairs
        out["sources"].append(report(nm, pairs, a.iters, rng))
        print()

    # Pairs both databases judge -- the strongest available reference.
    if len(all_pairs) == 2:
        (n1, p1), (n2, p2) = all_pairs.items()
        i1 = {(p["taxon_key"], p["disease"]): p for p in p1}
        i2 = {(p["taxon_key"], p["disease"]): p for p in p2}
        both = sorted(set(i1) & set(i2))
        concur = [k for k in both if i1[k]["ref_dir"] == i2[k]["ref_dir"]]
        print("=" * 78)
        print(f"PAIRS JUDGED BY BOTH  --  {len(both)} pairs, "
              f"{len(concur)} where the two curations agree with EACH OTHER")
        print("=" * 78)
        if concur:
            ag = sum(i1[k]["agree"] for k in concur)
            print(f"  our agreement where both curations concur: "
                  f"{ag}/{len(concur)} ({100*ag/len(concur):.1f}%)")
        disc = [k for k in both if i1[k]["ref_dir"] != i2[k]["ref_dir"]]
        print(f"  pairs where the two curations CONTRADICT each other: {len(disc)}")
        for k in disc:
            print(f"    {i1[k]['taxon'][:26]:27} {i1[k]['disease'][:24]:25} "
                  f"{n1}={i1[k]['ref_dir']}({i1[k]['ref_n']})  "
                  f"{n2}={i2[k]['ref_dir']}({i2[k]['ref_n']})  ours={i1[k]['our_dir']}")
        out["both"] = {"n_pairs": len(both), "n_concurring": len(concur),
                       "n_contradicting": len(disc),
                       "our_agreement_where_concur":
                           round(sum(i1[k]["agree"] for k in concur) / len(concur), 4)
                           if concur else None,
                       "contradicting": [
                           {"taxon": i1[k]["taxon"], "disease": i1[k]["disease"],
                            n1: i1[k]["ref_dir"], n2: i2[k]["ref_dir"],
                            "ours": i1[k]["our_dir"]} for k in disc]}

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
