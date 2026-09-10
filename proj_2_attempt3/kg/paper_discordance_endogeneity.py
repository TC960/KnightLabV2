#!/usr/bin/env python3
"""Why `sits_on_contested_edges` is not an explanatory variable.

`paper_discordance_offset.py` leaves exactly one predictor standing after BH:
papers sitting on contested edges disagree with consensus more than the within-edge
null expects (O/E 1.056 vs 0.830, p = 0.0029, q = 0.032). It is the only survivor,
so it is the one most likely to be written up as a finding -- and it should not be.
Two reasons, both checkable rather than arguable, and this script checks them.

REASON 1 -- it is very nearly the offset itself.
  The offset is E[disagree] = 1 if n_e == n_d else min(n_e, n_d)/n per edge. The
  predictor is the mean of min(n_e, n_d)/n over the same edges. Testing one
  against the other is closer to a calibration diagnostic than to a test of a
  study property. The specific worry is tied edges, where E is pinned at exactly
  1: on a 1e/1d edge BOTH papers disagree under every permutation, so those
  observations carry no null variance at all while scoring maximum contestedness.
  Check: drop tied edges and re-test.

REASON 2 -- the outcome causes the predictor.
  An edge is contested BECAUSE the papers on it disagreed. A paper with a high
  discordance propensity manufactures the contested edges it is then observed to
  sit on. The association is guaranteed by construction in a way no permutation
  of paper labels can undo, because the permutation shuffles the label and not the
  history that produced it.
  Check: recompute each edge's contestedness LEAVE-ONE-OUT -- from the other
  papers only -- so the paper's own vote cannot inflate its own predictor. If the
  effect is endogenous it should shrink sharply; if a paper genuinely seeks out
  ground that others already dispute, it should survive.

Writes paper_discordance_endogeneity.json.
"""

import json
import random
import sys
from collections import defaultdict

from paper_inversion import MIN_DECISIVE, build_observations, load_graph, score
from paper_inversion_control import ancestor_sets, thin
from paper_discordance_offset import expectations, test_binary

SEED = 20260910


def rows_for(edges):
    per_paper, _ = score(edges)
    _, e_dis = expectations(edges)
    out = []
    for p, v in sorted(per_paper.items()):
        if v[0] < MIN_DECISIVE or e_dis[p] <= 0:
            continue
        out.append({"paper": p, "dec": v[0], "dis": v[1], "e_dis": e_dis[p]})
    return out


def contestedness(edges, rows, leave_one_out):
    acc = defaultdict(list)
    for ed in edges:
        obs = ed["obs"]
        n = len(obs)
        ne = sum(1 for _, d in obs if d == "e")
        nd = n - ne
        for pi, di in obs:
            if leave_one_out:
                # others only: remove this paper's own vote from the counts
                oe = ne - (di == "e")
                od = nd - (di == "d")
                m = oe + od
                acc[pi].append(min(oe, od) / m if m else 0.0)
            else:
                acc[pi].append(min(ne, nd) / n)
    vals = [sum(acc[r["paper"]]) / len(acc[r["paper"]]) if acc.get(r["paper"])
            else None for r in rows]
    return vals


def run(edges, label, leave_one_out):
    rows = rows_for(edges)
    if len(rows) < 10:
        print(f"[{label}] only {len(rows)} papers -- skipped")
        return None
    vals = contestedness(edges, rows, leave_one_out)
    present = sorted(v for v in vals if v is not None)
    cut = present[len(present) // 2]
    res = test_binary(rows, [None if v is None else v >= cut for v in vals],
                      random.Random(SEED))
    if res is None:
        print(f"[{label}] degenerate split -- skipped")
        return None
    print(f"[{label}] {len(rows)} papers, expected mass "
          f"{res['exp_true']:.1f}/{res['exp_false']:.1f}")
    print(f"[{label}]   O/E {res['oe_true']:.3f} (n={res['n_true']}) vs "
          f"{res['oe_false']:.3f} (n={res['n_false']})  diff {res['diff']:+.3f}  "
          f"p = {res['p']:.4f}  MDE +-{res['mde_abs_diff']:.3f}")
    return {**res, "label": label, "n_papers": len(rows), "n_edges": len(edges),
            "leave_one_out": leave_one_out}


def main():
    graph = load_graph()
    edges = build_observations(graph)
    rel = ancestor_sets(graph)
    thinned, _ = thin(edges, rel)
    out = {}

    print("REASON 1 -- tied edges, where E[disagree] is pinned at 1\n")
    out["self_inclusive_all"] = run(thinned, "self-inclusive, all edges", False)
    notie = [e for e in thinned
             if sum(1 for _, d in e["obs"] if d == "e") * 2 != len(e["obs"])]
    print(f"\n  dropped {len(thinned) - len(notie)} tied edges of {len(thinned)}")
    out["self_inclusive_no_ties"] = run(notie, "self-inclusive, no ties", False)
    ge3 = [e for e in notie if len(e["obs"]) >= 3]
    print(f"\n  further restricted to {len(ge3)} edges with >=3 papers")
    out["self_inclusive_no_ties_ge3"] = run(ge3, "self-inclusive, no ties, >=3", False)

    print("\n" + "=" * 72)
    print("REASON 2 -- leave-one-out contestedness (the paper's own vote removed)\n")
    out["leave_one_out_all"] = run(thinned, "leave-one-out, all edges", True)
    out["leave_one_out_no_ties"] = run(notie, "leave-one-out, no ties", True)

    a = out.get("self_inclusive_all")
    b = out.get("leave_one_out_all")
    if a and b:
        print(f"\nself-inclusive diff {a['diff']:+.3f} (p={a['p']:.4f})  ->  "
              f"leave-one-out diff {b['diff']:+.3f} (p={b['p']:.4f})")
        shrink = 1 - abs(b["diff"]) / abs(a["diff"]) if a["diff"] else float("nan")
        print(f"effect shrinks by {shrink:.0%} once the paper's own vote is "
              f"removed from its own predictor")
        out["shrinkage_self_to_loo"] = round(shrink, 4)

    with open("paper_discordance_endogeneity.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote paper_discordance_endogeneity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
