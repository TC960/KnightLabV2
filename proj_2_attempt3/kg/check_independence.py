#!/usr/bin/env python3
"""Is the external validation actually independent of our corpus?

The project's most-cited numbers are agreement with Disbiome (73.0%) and Peryton
(72.5%), and they are trusted precisely because those curations are independent
of this pipeline. That is an ASSUMPTION, and it has never been checked. Both
databases curate the primary literature, and so do we. If Disbiome's entry for
Roseburia/Parkinson's was read out of the same paper our extractor read, then
agreeing with it measures whether two readers read one sentence the same way --
a fair test of the extractor, but NOT the independent replication the number is
presented as.

This is checkable: our extraction rows carry PubMed links, Disbiome's
publications table carries PubMed urls, and Peryton ships a PMID column. So:

  1. how much of our 272-paper corpus do the curations also cite,
  2. and -- the question that matters -- does agreement DIFFER between pairs
     where the curated evidence shares a paper with ours and pairs where the two
     sides rest on entirely disjoint literature.

If the disjoint subset agrees at the same rate, the headline is independent and
the assumption is earned. If it agrees much less, the headline is inflated by
shared sources and the honest number is the disjoint one.

Same clustering discipline as the rest of this directory: taxon-block
permutation, because pairs sharing a taxon are not independent observations.

    python check_independence.py
"""
import argparse
import json
import os
import random
from collections import defaultdict

from calibrate_agreement import build_pairs, cluster_bootstrap, rate
from calibrate_confounds import block_perm_two
from validate_external import GRAPH, load_disbiome, load_peryton, paper_keys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACTIONS = os.path.join(HERE, "extractions_screened.json")
OUT = os.path.join(HERE, "independence.json")


def our_keys():
    """paper title -> its identity keys (PMID / DOI / normalised title).

    The title is itself one of the keys, so every paper matches on at least one
    key and a paper with no link is still comparable against a curation that
    records titles. Both databases do.
    """
    by_title = {}
    for r in json.load(open(EXTRACTIONS)):
        by_title[r["title"]] = paper_keys(link=r.get("link"), title=r["title"])
    return by_title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    G = json.load(open(GRAPH))
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy()

    title2keys = our_keys()
    corpus_keys = set()
    for p in G["papers"]:
        corpus_keys |= title2keys.get(p["title"], set())
    print(f"our corpus: {len(G['papers'])} contributing papers, "
          f"{sum(1 for p in G['papers'] if any(k.startswith('pmid:') for k in title2keys.get(p['title'], ())))}"
          f" with a PMID, all {len(G['papers'])} matchable by title/DOI")

    out = {"corpus_papers": len(G["papers"]), "sources": []}

    for loader in (load_disbiome, load_peryton):
        recs, nm = loader()
        if recs is None:
            continue
        ref_papers = {frozenset(paper_keys(pmid=r.get("pmid"), doi=r.get("doi"),
                                           title=r.get("ptitle"))) for r in recs}
        ref_papers = {f for f in ref_papers if f}
        ref_all = set().union(*ref_papers) if ref_papers else set()
        shared_titles = [p["title"] for p in G["papers"]
                         if title2keys.get(p["title"], set()) & ref_all]
        print("\n" + "=" * 78)
        print(f"{nm.upper()}")
        print("=" * 78)
        print(f"  distinct publications cited by {nm:9}: {len(ref_papers)}")
        print(f"  of our {len(G['papers'])} corpus papers, also cited by {nm}: "
              f"{len(shared_titles)} "
              f"({100*len(shared_titles)/len(G['papers']):.1f}%)")

        pairs = build_pairs(G, recs, tax)
        # A pair is "shared-source" if ANY curated record backing it comes from a
        # paper our own edge also rests on. Matching on PMID *or* DOI *or* title
        # maximises the shared set, which is the conservative direction: it makes
        # the "disjoint" subset as clean as this data allows.
        for p in pairs:
            our_k = set()
            for t in p["papers"]:
                our_k |= title2keys.get(t, set())
            p["shared_source"] = bool(our_k & set(p["ref_keys"]))

        sh = [p for p in pairs if p["shared_source"]]
        dj = [p for p in pairs if not p["shared_source"]]
        print(f"\n  decisive pairs            : {len(pairs)}")
        print(f"    resting on a shared paper: {len(sh)} "
              f"({100*len(sh)/max(len(pairs),1):.1f}%)  agreement "
              f"{100*rate(sh):.1f}%" if sh else "    resting on a shared paper: 0")
        print(f"    fully disjoint sources   : {len(dj)} "
              f"({100*len(dj)/max(len(pairs),1):.1f}%)  agreement "
              f"{100*rate(dj):.1f}%")

        entry = {"source": nm, "ref_publications": len(ref_papers),
                 "shared_corpus_papers": len(shared_titles),
                 "n_pairs": len(pairs), "n_shared_source": len(sh),
                 "n_disjoint": len(dj),
                 "agreement_shared": round(rate(sh), 4) if sh else None,
                 "agreement_disjoint": round(rate(dj), 4) if dj else None,
                 "pooled": round(rate(pairs), 4)}

        if sh and dj:
            r = block_perm_two(pairs, lambda p: p["shared_source"], a.iters, rng)
            ci = cluster_bootstrap(pairs, lambda p: p["shared_source"], 4000, rng)
            print(f"    difference (shared - disjoint) = {100*r['diff']:+.1f} pts"
                  f"   block-perm p = {r['p']:.4f}"
                  f"   MDE {100*r['mde']:.1f} pts")
            if ci:
                excl = "EXCLUDES 0" if ci[0] > 0 or ci[1] < 0 else "includes 0"
                print(f"    cluster-bootstrap 95% CI "
                      f"[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}] -- {excl}")
                entry["ci"] = [round(ci[0], 4), round(ci[1], 4)]
            entry.update({"diff": round(r["diff"], 4), "p": round(r["p"], 4),
                          "mde": round(r["mde"], 4)})

        # DISEASE IS A CONFOUNDER AND MUST BE HELD FIXED. The buckets are not
        # comparable across diseases: every ALS pair here is shared-source (both
        # curations cite the ALS papers we used) and every autism pair is
        # disjoint, so a pooled shared-vs-disjoint difference partly measures
        # "ALS versus autism". Only a within-disease comparison is interpretable,
        # and only for diseases carrying enough of both.
        entry["by_disease"] = []
        print(f"    within a single disease (>=8 pairs on each side):")
        any_dis = False
        for d in sorted({p["disease"] for p in pairs}):
            sub = [p for p in pairs if p["disease"] == d]
            s_sh = [p for p in sub if p["shared_source"]]
            s_dj = [p for p in sub if not p["shared_source"]]
            if len(s_sh) < 8 or len(s_dj) < 8:
                continue
            any_dis = True
            r = block_perm_two(sub, lambda p: p["shared_source"], a.iters, rng)
            print(f"      {d[:26]:27} shared {100*rate(s_sh):5.1f}% (n={len(s_sh)})"
                  f"  disjoint {100*rate(s_dj):5.1f}% (n={len(s_dj)})"
                  f"  diff {100*r['diff']:+.1f}  p={r['p']:.4f}"
                  f"  MDE {100*r['mde']:.1f}")
            entry["by_disease"].append({
                "disease": d, "n_shared": len(s_sh), "n_disjoint": len(s_dj),
                "agreement_shared": round(rate(s_sh), 4),
                "agreement_disjoint": round(rate(s_dj), 4),
                "diff": round(r["diff"], 4), "p": round(r["p"], 4),
                "mde": round(r["mde"], 4)})
        if not any_dis:
            print("      (none -- no disease has >=8 pairs in both buckets)")

        # Shared-source pairs are the well-studied ones, and evidence count is
        # itself a predictor of agreement (calibrate_agreement.py). So the split
        # above could be that effect wearing a different hat. Re-test inside
        # evidence strata; if it survives both, it is not an evidence artifact.
        entry["strata"] = []
        for sname, sfn in [("1 paper", lambda p: p["our_n_papers"] == 1),
                           (">=2 papers", lambda p: p["our_n_papers"] >= 2)]:
            sub = [p for p in pairs if sfn(p)]
            s_sh = [p for p in sub if p["shared_source"]]
            s_dj = [p for p in sub if not p["shared_source"]]
            if len(s_sh) < 4 or len(s_dj) < 4:
                continue
            r = block_perm_two(sub, lambda p: p["shared_source"], a.iters, rng)
            print(f"    within {sname:11}: shared {100*rate(s_sh):5.1f}% "
                  f"(n={len(s_sh)})  disjoint {100*rate(s_dj):5.1f}% (n={len(s_dj)})"
                  f"  diff {100*r['diff']:+.1f}  p={r['p']:.4f}  "
                  f"MDE {100*r['mde']:.1f}")
            entry["strata"].append({
                "stratum": sname, "n_shared": len(s_sh), "n_disjoint": len(s_dj),
                "agreement_shared": round(rate(s_sh), 4),
                "agreement_disjoint": round(rate(s_dj), 4),
                "diff": round(r["diff"], 4), "p": round(r["p"], 4),
                "mde": round(r["mde"], 4)})

        out["sources"].append(entry)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
