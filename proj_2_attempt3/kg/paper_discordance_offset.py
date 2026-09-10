#!/usr/bin/env python3
"""The same predictors, against the EXACT expectation instead of a pooled rate.

`paper_discordance_predictors.py` returned three survivors -- Parkinson's papers,
large case counts, large cohorts -- and all three are suspect for one mechanical
reason. The disagreement rate is not comparable across edges of different depth:

  a 2-paper contested edge scores BOTH its papers as disagreeing (rate 1.00)
  a 10-paper 8/2 edge scores only the two minority papers (rate 0.20)

so a paper sitting on well-evidenced edges has a structurally lower rate no matter
how good it is. Parkinson's is the most-reported disease in this corpus and large
cohorts study well-studied diseases, so all three "findings" could be that artifact
wearing three hats.

The fix is not another covariate. The within-edge null already defines each
paper's expected disagreements exactly, and it can be written down in closed form.
For an edge with n_e up-votes and n_d down-votes out of N, a paper occupying one
slot draws its direction from that multiset, so

    P(decisive)  = 1                     if |n_e - n_d| != 1
                 = min(n_e, n_d) / N     otherwise
    P(disagree)  = 1                     if n_e == n_d
                 = min(n_e, n_d) / N     otherwise

Summing over a paper's edges gives E_dec(p) and E_dis(p), which absorb edge depth,
contestedness and the tie structure at once. The outcome becomes the
observed/expected ratio O/E, and a group difference in O/E is a difference the
edge structure does not already explain.

Verified against the permutation ensemble rather than trusted: the closed form is
checked to agree with the simulated per-paper means before anything is tested.

Null unchanged: shuffle predictor labels ACROSS PAPERS holding each paper's
(observed, expected) fixed. Unit of randomisation is the paper. BH across
predictors, MDE reported for every null.

Writes paper_discordance_offset.json.
"""

import json
import random
import sys
from collections import defaultdict

from paper_inversion import MIN_DECISIVE, build_observations, load_graph, score, \
    score_permuted
from paper_inversion_control import ancestor_sets, thin
from paper_discordance_predictors import paper_disease

N_PERM = 20000
SEED = 20260910


def expectations(edges):
    """Closed-form E[decisive] and E[disagree] per paper under the within-edge null."""
    e_dec = defaultdict(float)
    e_dis = defaultdict(float)
    for ed in edges:
        obs = ed["obs"]
        n = len(obs)
        ne = sum(1 for _, d in obs if d == "e")
        nd = n - ne
        lo = min(ne, nd)
        p_dec = 1.0 if abs(ne - nd) != 1 else lo / n
        p_dis = 1.0 if ne == nd else lo / n
        for pi, _ in obs:
            e_dec[pi] += p_dec
            e_dis[pi] += p_dis
    return e_dec, e_dis


def verify(edges, e_dec, e_dis, n_check=4000, seed=7):
    """Check the closed form against the permutation ensemble it claims to describe."""
    rng = random.Random(seed)
    acc_dis = defaultdict(float)
    acc_dec = defaultdict(float)
    for _ in range(n_check):
        dis, dec = score_permuted(edges, rng)
        for p, v in dis.items():
            acc_dis[p] += v
        for p, v in dec.items():
            acc_dec[p] += v
    keys = [p for p in e_dis if e_dis[p] > 0]
    worst_dis = max(abs(acc_dis[p] / n_check - e_dis[p]) for p in keys)
    worst_dec = max(abs(acc_dec[p] / n_check - e_dec[p]) for p in keys)
    print(f"closed form vs {n_check} permutations: max abs error "
          f"{worst_dis:.4f} (disagree), {worst_dec:.4f} (decisive)")
    return worst_dis, worst_dec


def oe(rows, mask):
    o = sum(r["dis"] for r, m in zip(rows, mask) if m)
    e = sum(r["e_dis"] for r, m in zip(rows, mask) if m)
    return (o / e if e else float("nan")), o, e


def test_binary(rows, labels, rng, n_perm=N_PERM):
    idx = [i for i, l in enumerate(labels) if l is not None]
    if len(idx) < 8:
        return None
    sub = [rows[i] for i in idx]
    lab = [bool(labels[i]) for i in idx]
    if len(set(lab)) < 2:
        return None
    a, o_a, e_a = oe(sub, lab)
    b, o_b, e_b = oe(sub, [not x for x in lab])
    obs = a - b
    null = []
    perm = list(lab)
    for _ in range(n_perm):
        rng.shuffle(perm)
        x, _, _ = oe(sub, perm)
        y, _, _ = oe(sub, [not v for v in perm])
        null.append(x - y)
    p = (sum(1 for d in null if abs(d) >= abs(obs)) + 1) / (n_perm + 1)
    null.sort()
    mde = max(abs(null[int(0.025 * n_perm)]), abs(null[int(0.975 * n_perm)]))
    return {"n_true": sum(lab), "n_false": len(lab) - sum(lab),
            "oe_true": round(a, 4), "oe_false": round(b, 4),
            "obs_true": o_a, "exp_true": round(e_a, 2),
            "obs_false": o_b, "exp_false": round(e_b, 2),
            "diff": round(obs, 4), "p": round(p, 5),
            "mde_abs_diff": round(mde, 4)}


def main():
    graph = load_graph()
    edges = build_observations(graph)
    rel = ancestor_sets(graph)
    thinned, _ = thin(edges, rel)
    per_paper, _ = score(thinned)
    e_dec, e_dis = expectations(thinned)
    verify(thinned, e_dec, e_dis)

    dis_of = paper_disease(graph, thinned)
    papers = graph["papers"]
    rows = []
    for p, v in sorted(per_paper.items()):
        if v[0] < MIN_DECISIVE or e_dis[p] <= 0:
            continue
        rows.append({"paper": p, "dec": v[0], "dis": v[1],
                     "e_dec": round(e_dec[p], 3), "e_dis": round(e_dis[p], 3),
                     "title": papers[p]["title"][:60],
                     "meta": papers[p].get("has_meta", False),
                     "disease": dis_of.get(p, "?"), "m": papers[p]})
    tot_o = sum(r["dis"] for r in rows)
    tot_e = sum(r["e_dis"] for r in rows)
    print(f"{len(rows)} testable papers; observed {tot_o} disagreements vs "
          f"{tot_e:.1f} expected under the within-edge null (O/E = "
          f"{tot_o/tot_e:.3f})\n")

    rng = random.Random(SEED)

    def meta(r, f):
        return r["m"].get(f) if r["meta"] else None

    def med_split(field):
        vals = [meta(r, field) for r in rows]
        nums = sorted(v for v in vals if isinstance(v, (int, float)) and v)
        cut = nums[len(nums) // 2]
        return [None if not isinstance(v, (int, float)) or not v else v >= cut
                for v in vals]

    coh = []
    for r in rows:
        if not r["meta"]:
            coh.append(None); continue
        a, b = r["m"].get("n_cases"), r["m"].get("n_controls")
        coh.append(((a or 0) + (b or 0)) or None)
    cn = sorted(v for v in coh if v)
    ccut = cn[len(cn) // 2]

    # Edge depth as a PURE structural property: the mean number of contributing
    # papers on the edges this paper sits on. It must not be built from the
    # paper's own decisive count -- decisiveness depends on the paper's drawn
    # direction, so e_dec/dec would put part of the outcome inside the predictor.
    depth_acc = defaultdict(list)
    for ed in thinned:
        for pi, _ in ed["obs"]:
            depth_acc[pi].append(len(ed["obs"]))
    edge_depth = [sum(depth_acc[r["paper"]]) / len(depth_acc[r["paper"]])
                  if depth_acc.get(r["paper"]) else None for r in rows]
    ed_sorted = sorted(v for v in edge_depth if v is not None)
    ed_cut = ed_sorted[len(ed_sorted) // 2]
    # and how contested those edges are, likewise structural
    cont_acc = defaultdict(list)
    for ed in thinned:
        ne = sum(1 for _, d in ed["obs"] if d == "e")
        nd = len(ed["obs"]) - ne
        for pi, _ in ed["obs"]:
            cont_acc[pi].append(min(ne, nd) / len(ed["obs"]))
    contested = [sum(cont_acc[r["paper"]]) / len(cont_acc[r["paper"]])
                 if cont_acc.get(r["paper"]) else None for r in rows]
    ct_sorted = sorted(v for v in contested if v is not None)
    ct_cut = ct_sorted[len(ct_sorted) // 2]

    tests = {
        "disease_is_Parkinsons":
            [r["disease"].lower().startswith("parkinson") for r in rows],
        "n_cases_above_median": med_split("n_cases"),
        "cohort_total_above_median": [None if not v else v >= ccut for v in coh],
        "n_controls_above_median": med_split("n_controls"),
        "diet_controlled": [meta(r, "diet") for r in rows],
        "medication_controlled": [meta(r, "med") for r in rows],
        "sequencing_16S_vs_shotgun":
            [None if not r["meta"] or not meta(r, "seq") else meta(r, "seq") == "16S"
             for r in rows],
        "country_China": [None if not r["meta"] or not meta(r, "country")
                          else meta(r, "country") == "China" for r in rows],
        "region_V3V4": [None if not r["meta"] or not meta(r, "region")
                        else meta(r, "region") == "V3-V4" for r in rows],
        # the confound itself, now as a predictor: if the offset works, these
        # should be null where the raw-rate version was strongly positive
        "sits_on_deep_edges":
            [None if v is None else v >= ed_cut for v in edge_depth],
        "sits_on_contested_edges":
            [None if v is None else v >= ct_cut for v in contested],
    }

    results = {}
    for name, labels in tests.items():
        res = test_binary(rows, labels, rng)
        results[name] = res
        if res is None:
            print(f"  {name:32s} skipped")
            continue
        print(f"  {name:32s} O/E {res['oe_true']:.3f} (n={res['n_true']:>2}) vs "
              f"{res['oe_false']:.3f} (n={res['n_false']:>2})  "
              f"diff {res['diff']:+.3f}  p={res['p']:.4f}  "
              f"MDE +-{res['mde_abs_diff']:.3f}")

    live = sorted(((k, v["p"]) for k, v in results.items() if v),
                  key=lambda kv: kv[1])
    m = len(live)
    q, prev = {}, 1.0
    for i in range(m - 1, -1, -1):
        k, p = live[i]
        prev = min(prev, p * m / (i + 1))
        q[k] = round(min(1.0, prev), 4)
    for k in q:
        results[k]["q"] = q[k]
    print(f"\nBH across {m} predictors:")
    for k, p in live:
        print(f"  {k:32s} p={p:.4f}  q={q[k]:.4f}"
              f"{'  <-- survives' if q[k] < 0.05 else ''}")
    survivors = [k for k in q if q[k] < 0.05]
    print(f"\nsurvivors: {survivors if survivors else 'NONE'}")

    out = {"n_testable": len(rows), "total_observed": tot_o,
           "total_expected": round(tot_e, 2), "overall_oe": round(tot_o / tot_e, 4),
           "n_perm": N_PERM, "seed": SEED, "tests": results,
           "survivors": survivors,
           "papers": [{k: r[k] for k in ("paper", "dec", "dis", "e_dec", "e_dis",
                                         "title", "disease", "meta")}
                      for r in rows]}
    with open("paper_discordance_offset.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote paper_discordance_offset.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
