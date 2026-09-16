#!/usr/bin/env python3
"""Two follow-ups to `cooccur_diagnostics.py`.

1. A RANK-based statistic. Dropping pairs with cosine >= 0.8 killed the effect
   (+0.047 -> +0.014, p=0.113) on only 1.1% of pairs -- but that deletion is
   biased: the high-similarity tail is exactly where a real same-direction
   signal would sit, so removing it removes the signal by construction. The
   honest outlier-robust test keeps every pair and uses only their ORDER:
   per-edge AUC = P(a random same-direction pair is more similar than a random
   cross-direction pair). 0.5 under the null, insensitive to how extreme the
   top pair is.

2. WHO the near-duplicate pairs are. Two papers with near-identical relation
   vocabulary that agree on direction may be the same cohort published twice --
   pseudo-replication that inflates an edge's evidence count -- or just two
   short papers sharing their only two taxa.
"""
import json
import numpy as np
from cooccur_diagnostics import load, build_edges, stat_perm, permuted_sims

RNG = np.random.default_rng(7)
N_PERM = 2000


def auc_edge(sims, same):
    """Mann-Whitney AUC of same-direction pairs vs cross-direction pairs."""
    a, b = sims[same], sims[~same]
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    vals = np.concatenate([a, b])
    sv = vals[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_a = ranks[:len(a)].sum()
    return (r_a - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def stat_auc_simple(edges, sims_list):
    vals = []
    for e, sims in zip(edges, sims_list):
        if sims is None:
            continue
        lab = e["lab"]
        v = ~np.isnan(sims)
        same = (lab[e["iu"]] == lab[e["ju"]]) & v
        diff = (lab[e["iu"]] != lab[e["ju"]]) & v
        if same.sum() == 0 or diff.sum() == 0:
            continue
        sel = same | diff
        vals.append(auc_edge(sims[sel], same[sel]))
    return float(np.mean(vals)), len(vals)


def main():
    g, titles, X, vocab, disease = load()
    edges = build_edges(g, X, vocab, "cosine")
    size = X.sum(1)

    obs, n = stat_auc_simple(edges, [e["sims"] for e in edges])
    print(f"=== rank-based test (no pairs deleted) ===")
    print(f"observed mean per-edge AUC {obs:.4f} over {n} edges "
          f"(0.5 = no association)")

    groups = [np.where(disease == s)[0] for s in np.unique(disease)]
    null = np.zeros(N_PERM)
    for k in range(N_PERM):
        perm = np.arange(len(titles))
        for gp in groups:
            perm[gp] = RNG.permutation(gp)
        sims = [permuted_sims(e, X, perm, "cosine") for e in edges]
        null[k] = stat_auc_simple(edges, sims)[0]
    p = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (
        N_PERM + 1)
    print(f"null mean {null.mean():.4f} sd {null.std():.4f}  p={p:.4f}  "
          f"min-detectable |AUC-0.5| {1.96*null.std():.4f}")

    out = {"auc": dict(observed=obs, n_edges=n, null_mean=float(null.mean()),
                       null_sd=float(null.std()), p=float(p),
                       min_detectable=float(1.96 * null.std()),
                       n_perm=N_PERM)}

    # ---- who are the near-duplicate pairs -------------------------------
    print("\n=== near-duplicate paper pairs inside contested edges "
          "(cosine >= 0.8) ===")
    seen, rows = set(), []
    for e in edges:
        lab = e["lab"]
        for a, b, s in zip(e["iu"], e["ju"], e["sims"]):
            if s < 0.8:
                continue
            pa, pb = int(e["idx"][a]), int(e["idx"][b])
            key = (min(pa, pb), max(pa, pb))
            agree = bool(lab[a] == lab[b])
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(cosine=float(s), same_direction=agree,
                             paper_a=titles[pa], paper_b=titles[pb],
                             taxa_a=int(size[pa]), taxa_b=int(size[pb]),
                             country_a=g["papers"][pa].get("country"),
                             country_b=g["papers"][pb].get("country"),
                             edge=f"{e['key'][0]} / {e['key'][1]}"))
    rows.sort(key=lambda r: -r["cosine"])
    print(f"{len(rows)} distinct paper pairs")
    small = sum(1 for r in rows if min(r["taxa_a"], r["taxa_b"]) <= 3)
    print(f"  of which {small} involve a paper with <=3 taxa in its profile "
          f"(cosine is trivially high on tiny profiles)")
    for r in rows:
        print(f"  cos {r['cosine']:.2f} same_dir={r['same_direction']} "
              f"taxa {r['taxa_a']}/{r['taxa_b']}  [{r['edge']}]")
        print(f"      A: {r['paper_a'][:95]}")
        print(f"      B: {r['paper_b'][:95]}")
    out["near_duplicate_pairs"] = rows
    json.dump(out, open("cooccur_followup.json", "w"), indent=1)
    print("\nwrote cooccur_followup.json")


if __name__ == "__main__":
    main()
