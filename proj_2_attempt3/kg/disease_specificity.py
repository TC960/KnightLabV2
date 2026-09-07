#!/usr/bin/env python3
"""Does disease identity explain edge direction, or is direction a taxon property?

Motivation. `disease_containment.py` found that a clinical subtype's directional
profile does not resemble its parent's more than an unrelated disease's does --
but it also turned up the number that prompted this script: **two arbitrary
diseases already agree on 67.2% of the taxa they share**, against a marginal
chance rate of 51.4% (1,045 enriched / 747 depleted decisive edges). Something is
making directions agree across diseases that have nothing to do with each other.

The candidate explanation is a corpus-wide "generic dysbiosis" prior: this
literature reports the same short list of taxa going the same way in almost every
disease (butyrate producers down, Enterobacteriaceae up). If that is what the
67% is, then a large share of the graph's 2,011 edges are restatements of ~900
taxon-level facts, and "enriched in Parkinson's" carries much less
disease-specific information than the edge count suggests. That is a first-order
caveat on how the graph should be read, so it needs a test rather than a hunch.

Two questions, two different nulls.

QA -- does taxon identity predict direction at all?
    Null: within each paper, permute which of that paper's taxa got which
    direction. Preserves each paper's taxon set, its disease, and its
    enriched/depleted balance; destroys only the taxon->direction link.

QB -- does disease identity add anything beyond the taxon's global tendency?
    THIS IS THE CONSEQUENTIAL ONE. Statistic: among all pairs of papers
    reporting the same taxon, the agreement rate for pairs from the SAME disease
    minus the rate for pairs from DIFFERENT diseases. If disease matters,
    same-disease pairs agree more.
    Null: permute the disease label across the 272 papers. Each paper keeps all
    of its taxon-direction calls intact, so within-taxon structure is fully
    preserved and only the taxon x disease association is broken. This is a
    cluster-level (paper-level) permutation, as this project's rules require --
    pair-level shuffling has produced three false positives here on record.

A note on what the QB null does NOT do: it does not preserve how many papers
each disease has... it does, exactly, since it is a permutation of the label
vector. It does not preserve which diseases co-occur with which body site or
country; those are paper attributes that travel with the permuted label, so they
are handled the same way as disease and cannot leak.
"""
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "disease_specificity.json")

N_ITER = 5000
SEED = 20260905


def load_observations():
    """-> obs list of (paper_idx, taxon_key, direction), and paper_idx -> disease.

    Built from each edge's `ev`, which carries one entry per contributing paper
    with that paper's direction. 17 of 2,011 edges have n_obs > len(ev) because a
    single paper reported both directions for one taxon (the known
    self-contradiction set); those contribute one call here rather than two,
    which is conservative for every statistic below.
    """
    g = json.load(open(GRAPH))
    papers = g["papers"]
    obs = []
    pdisease = {}
    for e in g["edges"]:
        d = e["disease"]
        for x in e["ev"]:
            i = x["i"]
            obs.append((i, e["taxon_key"], "e" if x["d"] == "e" else "d"))
            # A paper can in principle appear under two diseases; record and
            # count that rather than silently overwriting.
            pdisease.setdefault(i, set()).add(d)
    multi = {i: s for i, s in pdisease.items() if len(s) > 1}
    disease_of = {i: sorted(s)[0] for i, s in pdisease.items()}
    return obs, disease_of, multi, papers


# ---------------------------------------------------------------------------
# QA: does taxon identity predict direction?
# ---------------------------------------------------------------------------
def taxon_purity(obs):
    """sum(majority calls) / sum(all calls), over taxa with >=2 calls."""
    by_t = defaultdict(lambda: [0, 0])
    for _, t, d in obs:
        by_t[t][0 if d == "e" else 1] += 1
    num = den = 0
    for t, (ne, nd) in by_t.items():
        if ne + nd >= 2:
            num += max(ne, nd)
            den += ne + nd
    return num / den, den


def qa_test(obs, n_iter=N_ITER, seed=SEED):
    rng = random.Random(seed)
    obs_stat, den = taxon_purity(obs)
    by_paper = defaultdict(list)
    for i, t, d in obs:
        by_paper[i].append((t, d))
    hits = 0
    null = []
    for _ in range(n_iter):
        shuffled = []
        for i, items in by_paper.items():
            dirs = [d for _, d in items]
            rng.shuffle(dirs)
            for (t, _), nd in zip(items, dirs):
                shuffled.append((i, t, nd))
        s, _ = taxon_purity(shuffled)
        null.append(s)
        if s >= obs_stat:
            hits += 1
    mean = sum(null) / len(null)
    sd = (sum((x - mean) ** 2 for x in null) / len(null)) ** 0.5
    return {
        "statistic": "taxon directional purity",
        "observed": round(obs_stat, 4),
        "n_calls": den,
        "null_mean": round(mean, 4),
        "null_sd": round(sd, 5),
        "z": round((obs_stat - mean) / sd, 2) if sd else None,
        "p": round((hits + 1) / (n_iter + 1), 5),
    }


# ---------------------------------------------------------------------------
# QB: does disease identity add anything beyond the taxon's global tendency?
# ---------------------------------------------------------------------------
def pair_stat(obs, disease_of):
    """(same-disease agreement rate) - (different-disease agreement rate).

    Over all unordered pairs of paper-calls on the same taxon. Returns the two
    rates and their counts as well, so a null result can be reported with the n
    that produced it.
    """
    by_t = defaultdict(list)
    for i, t, d in obs:
        by_t[t].append((i, d))
    same_n = same_k = diff_n = diff_k = 0
    for t, calls in by_t.items():
        if len(calls) < 2:
            continue
        for a in range(len(calls)):
            ia, da = calls[a]
            for b in range(a + 1, len(calls)):
                ib, db = calls[b]
                if ia == ib:
                    continue
                agree = (da == db)
                if disease_of[ia] == disease_of[ib]:
                    same_n += 1
                    same_k += agree
                else:
                    diff_n += 1
                    diff_k += agree
    same_r = same_k / same_n if same_n else None
    diff_r = diff_k / diff_n if diff_n else None
    delta = (same_r - diff_r) if (same_r is not None and diff_r is not None) else None
    return delta, same_r, diff_r, same_n, diff_n


def qb_test(obs, disease_of, n_iter=N_ITER, seed=SEED):
    rng = random.Random(seed)
    delta, same_r, diff_r, same_n, diff_n = pair_stat(obs, disease_of)
    idx = sorted(disease_of)
    labels = [disease_of[i] for i in idx]
    hits = 0
    null = []
    for _ in range(n_iter):
        perm = labels[:]
        rng.shuffle(perm)
        shuffled_map = dict(zip(idx, perm))
        d, _, _, _, _ = pair_stat(obs, shuffled_map)
        if d is None:
            continue
        null.append(d)
        if d >= delta:
            hits += 1
    mean = sum(null) / len(null)
    sd = (sum((x - mean) ** 2 for x in null) / len(null)) ** 0.5
    # Minimum detectable effect: what delta would have cleared p=0.05?
    mde = sorted(null)[int(0.95 * len(null))]
    return {
        "statistic": "same-disease minus different-disease agreement",
        "observed_delta": round(delta, 4),
        "same_disease_rate": round(same_r, 4),
        "different_disease_rate": round(diff_r, 4),
        "n_same_disease_pairs": same_n,
        "n_different_disease_pairs": diff_n,
        "null_mean": round(mean, 5),
        "null_sd": round(sd, 5),
        "z": round((delta - mean) / sd, 2) if sd else None,
        "p": round((hits + 1) / (len(null) + 1), 5),
        "min_detectable_delta_at_p05": round(mde, 4),
    }


# ---------------------------------------------------------------------------
# Descriptive: which taxa are generic, which are disease-specific?
# ---------------------------------------------------------------------------
def generic_vs_specific(obs, disease_of, min_diseases=3):
    by_t = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for i, t, d in obs:
        by_t[t][disease_of[i]][0 if d == "e" else 1] += 1
    rows = []
    for t, per_dis in by_t.items():
        # one vote per disease: that disease's majority call
        votes = []
        for dis, (ne, nd) in per_dis.items():
            if ne == nd:
                continue
            votes.append("e" if ne > nd else "d")
        if len(votes) < min_diseases:
            continue
        ne, nd = votes.count("e"), votes.count("d")
        rows.append({
            "taxon_key": t,
            "n_diseases_voting": len(votes),
            "n_enriched": ne,
            "n_depleted": nd,
            "purity": round(max(ne, nd) / len(votes), 3),
            "consensus": "enriched" if ne > nd else "depleted",
            "n_papers": sum(a + b for a, b in per_dis.values()),
        })
    rows.sort(key=lambda r: (-r["purity"], -r["n_diseases_voting"]))
    return rows


def label_map():
    g = json.load(open(GRAPH))
    return {n["id"].split("t:", 1)[-1]: n["label"]
            for n in g["nodes"] if n.get("type") == "taxon"}


def main():
    obs, disease_of, multi, papers = load_observations()
    print(f"{len(obs)} paper-level calls over {len(set(o[1] for o in obs))} taxa "
          f"and {len(disease_of)} papers")
    if multi:
        print(f"WARNING: {len(multi)} papers carry >1 disease label; using the "
              f"lexicographically first for each. Examples: "
              f"{list(multi.items())[:3]}")

    print("\n--- QA: does taxon identity predict direction? ---")
    qa = qa_test(obs)
    for k, v in qa.items():
        print(f"  {k}: {v}")

    print("\n--- QB: does disease identity add anything? ---")
    qb = qb_test(obs, disease_of)
    for k, v in qb.items():
        print(f"  {k}: {v}")

    lm = label_map()
    gvs = generic_vs_specific(obs, disease_of)
    print(f"\n--- taxa reported in >=3 diseases: {len(gvs)} ---")
    print("MOST GENERIC (same direction in every disease that reports them):")
    for r in [x for x in gvs if x["purity"] == 1.0][:15]:
        print(f"  {lm.get(r['taxon_key'], r['taxon_key']):34s} "
              f"{r['consensus']:9s} in all {r['n_diseases_voting']} diseases "
              f"({r['n_papers']} papers)")
    print("MOST DISEASE-SPECIFIC (direction flips between diseases):")
    for r in sorted(gvs, key=lambda r: (r["purity"], -r["n_diseases_voting"]))[:15]:
        print(f"  {lm.get(r['taxon_key'], r['taxon_key']):34s} "
              f"purity {r['purity']:.2f}  {r['n_enriched']}up/"
              f"{r['n_depleted']}down across {r['n_diseases_voting']} diseases "
              f"({r['n_papers']} papers)")

    n_pure = sum(1 for r in gvs if r["purity"] == 1.0)
    print(f"\n{n_pure}/{len(gvs)} taxa reported in >=3 diseases never flip "
          f"direction ({n_pure / len(gvs):.1%})")

    json.dump({"qa": qa, "qb": qb,
               "n_calls": len(obs), "n_papers": len(disease_of),
               "multi_disease_papers": len(multi),
               "taxa_in_3plus_diseases": len(gvs),
               "n_never_flip": n_pure,
               "generic_vs_specific": gvs},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
