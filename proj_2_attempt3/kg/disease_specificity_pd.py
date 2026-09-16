#!/usr/bin/env python3
"""Is disease specificity a corpus-wide property, or is it Parkinson's disease?

Leave-one-disease-out in `disease_specificity_confounds.py` showed the
same-disease agreement gap falling from +0.0591 to +0.0187 when Parkinson's 67
papers are dropped -- a 68% reduction. Removing any other disease leaves it
between +0.052 and +0.087. Two readings:

  (a) SUBSTANTIVE. Parkinson's really does have the most reproducible gut
      signature in this literature, so its papers agree with each other more than
      other diseases' papers do. That is the published consensus in the field and
      would be a genuine confirmation.
  (b) LEVERAGE. Parkinson's is simply the biggest disease in the corpus, so it
      supplies the most same-disease pairs and any gap it has dominates the pooled
      number, without its internal agreement being unusual.

These make different predictions and a per-disease internal agreement rate
separates them: under (a) Parkinson's rate is high relative to other diseases of
comparable n; under (b) it is ordinary and merely heavily weighted.

Then the question that actually matters for the graph: with Parkinson's removed
entirely, is there ANY disease specificity left? +0.0187 needs its own
permutation test and its own power statement -- a shrunken point estimate is not
evidence of absence.
"""
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "disease_specificity_pd.json")

N_ITER = 5000
SEED = 20260905
PD = "Parkinson's disease"


def load():
    g = json.load(open(GRAPH))
    obs = []
    pdis = {}
    for e in g["edges"]:
        for x in e["ev"]:
            obs.append((x["i"], e["taxon_key"], "e" if x["d"] == "e" else "d"))
            pdis.setdefault(x["i"], set()).add(e["disease"])
    return obs, {i: sorted(s)[0] for i, s in pdis.items()}


def taxon_pairs(obs, keep=None):
    by_t = defaultdict(list)
    for i, t, d in obs:
        if keep is not None and i not in keep:
            continue
        by_t[t].append((i, d))
    pairs = []
    for t, calls in by_t.items():
        if len(calls) < 2:
            continue
        for a in range(len(calls)):
            ia, da = calls[a]
            for b in range(a + 1, len(calls)):
                ib, db = calls[b]
                if ia != ib:
                    pairs.append((ia, ib, da == db))
    return pairs


def gap(pairs, label_of):
    sn = sk = dn = dk = 0
    for ia, ib, agree in pairs:
        la, lb = label_of.get(ia), label_of.get(ib)
        if la is None or lb is None:
            continue
        if la == lb:
            sn += 1
            sk += agree
        else:
            dn += 1
            dk += agree
    if not sn or not dn:
        return None
    return (sk / sn - dk / dn, sk / sn, dk / dn, sn, dn)


def permute(pairs, label_of, n_iter=N_ITER, seed=SEED):
    rng = random.Random(seed)
    o = gap(pairs, label_of)
    if o is None:
        return {"note": "no comparable pairs"}
    ids = sorted(label_of)
    null, hits = [], 0
    for _ in range(n_iter):
        labs = [label_of[i] for i in ids]
        rng.shuffle(labs)
        r = gap(pairs, dict(zip(ids, labs)))
        if r is None:
            continue
        null.append(r[0])
        if r[0] >= o[0]:
            hits += 1
    mean = sum(null) / len(null)
    sd = (sum((x - mean) ** 2 for x in null) / len(null)) ** 0.5
    return {
        "gap": round(o[0], 4), "same_rate": round(o[1], 4),
        "diff_rate": round(o[2], 4), "n_same_pairs": o[3], "n_diff_pairs": o[4],
        "null_mean": round(mean, 5), "null_sd": round(sd, 5),
        "z": round((o[0] - mean) / sd, 2) if sd else None,
        "p": round((hits + 1) / (len(null) + 1), 5),
        "min_detectable_gap_at_p05": round(sorted(null)[int(0.95 * len(null))], 4),
    }


def per_disease_internal(pairs, disease_of, min_pairs=30):
    """Within-disease agreement rate, per disease, with the cross-disease rate
    each is being compared against."""
    within = defaultdict(lambda: [0, 0])
    across = [0, 0]
    for ia, ib, agree in pairs:
        da, db = disease_of.get(ia), disease_of.get(ib)
        if da is None or db is None:
            continue
        if da == db:
            within[da][0] += 1
            within[da][1] += agree
        else:
            across[0] += 1
            across[1] += agree
    base = across[1] / across[0]
    rows = []
    for d, (n, k) in within.items():
        if n < min_pairs:
            continue
        rows.append({"disease": d, "n_pairs": n, "agree": k,
                     "rate": round(k / n, 4),
                     "lift_over_cross_disease": round(k / n - base, 4)})
    rows.sort(key=lambda r: -r["rate"])
    return rows, round(base, 4)


def main():
    obs, disease_of = load()
    all_pairs = taxon_pairs(obs)
    res = {}

    rows, base = per_disease_internal(all_pairs, disease_of)
    print(f"cross-disease agreement baseline: {base}\n")
    print("--- within-disease agreement, per disease (>=30 pairs) ---")
    for r in rows:
        print(f"  {r['disease']:34s} {r['rate']:.3f}  "
              f"({r['agree']}/{r['n_pairs']} pairs)  "
              f"lift {r['lift_over_cross_disease']:+.4f}")
    res["per_disease_internal"] = rows
    res["cross_disease_baseline"] = base

    # Does Parkinson's stand out, or is it just big?
    pd_row = next((r for r in rows if r["disease"] == PD), None)
    others = [r for r in rows if r["disease"] != PD]
    if pd_row and others:
        med = sorted(r["rate"] for r in others)[len(others) // 2]
        n_above = sum(1 for r in others if r["rate"] >= pd_row["rate"])
        print(f"\nParkinson's rate {pd_row['rate']:.3f} on "
              f"{pd_row['n_pairs']} pairs; median of the other "
              f"{len(others)} diseases {med:.3f}; "
              f"{n_above} of {len(others)} match or beat it")
        res["pd_vs_others"] = {"pd_rate": pd_row["rate"],
                               "pd_n_pairs": pd_row["n_pairs"],
                               "median_other_rate": med,
                               "n_others_at_or_above": n_above,
                               "n_others": len(others)}
        # share of same-disease pairs that are Parkinson's
        tot_same = sum(r["n_pairs"] for r in rows)
        print(f"Parkinson's supplies {pd_row['n_pairs']}/{tot_same} = "
              f"{pd_row['n_pairs'] / tot_same:.1%} of all same-disease pairs")
        res["pd_share_of_same_disease_pairs"] = round(pd_row["n_pairs"] / tot_same, 4)

    # The decisive test: drop Parkinson's papers entirely and re-test.
    print("\n--- full corpus ---")
    full = permute(all_pairs, disease_of)
    res["full_corpus"] = full
    print(f"  gap {full['gap']:+.4f}  null {full['null_mean']:+.5f}  "
          f"z={full['z']}  p={full['p']}  MDE {full['min_detectable_gap_at_p05']:+.4f}")

    keep = {i for i, d in disease_of.items() if d != PD}
    no_pd_pairs = taxon_pairs(obs, keep=keep)
    no_pd_labels = {i: d for i, d in disease_of.items() if i in keep}
    print(f"\n--- Parkinson's removed ({len(disease_of) - len(keep)} papers, "
          f"{len(all_pairs) - len(no_pd_pairs)} pairs) ---")
    nopd = permute(no_pd_pairs, no_pd_labels)
    res["parkinsons_removed"] = nopd
    print(f"  gap {nopd['gap']:+.4f} (same {nopd['same_rate']:.3f} / "
          f"diff {nopd['diff_rate']:.3f}, n={nopd['n_same_pairs']}/"
          f"{nopd['n_diff_pairs']})")
    print(f"  null {nopd['null_mean']:+.5f} sd {nopd['null_sd']:.5f}  "
          f"z={nopd['z']}  p={nopd['p']}  "
          f"MDE {nopd['min_detectable_gap_at_p05']:+.4f}")
    verdict = ("SURVIVES without Parkinson's" if nopd["p"] < 0.05
               else "does NOT survive without Parkinson's -- the pooled effect "
                    "is substantially carried by one disease")
    print(f"  => {verdict}")
    res["verdict_without_pd"] = verdict

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
