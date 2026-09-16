#!/usr/bin/env python3
"""Two more ways the paper-level dispersion signal could be an artifact.

`paper_inversion_control.py` showed the excess dispersion survives dropping every
within-paper taxonomic relative (p = 0.0013 on the thinned set). Two structural
objections remain, and both are cheap to settle.

OBJECTION 1 -- two-paper edges are structurally forced.
  On an edge with exactly two papers, the leave-one-out majority IS the other
  paper. If they disagree, BOTH are scored as disagreeing, no matter how the
  within-edge shuffle assigns the directions: a 1e/1d edge yields two
  disagreements under every permutation, and a 2e/0d edge yields none. Those
  contributions carry zero null variance. They appear identically in the observed
  statistic and in the null, so the test stays valid -- but the effect should not
  be RESTING on them. Re-run using only edges with >= 3 papers, where the
  assignment of the minority direction is genuinely free.

OBJECTION 2 -- dispersion is two-tailed and the interesting tail is only one of them.
  sum (obs - exp)^2 / exp rises both when papers disagree MORE than expected and
  when they disagree LESS. "Some papers are unusually concordant" is a much weaker
  claim than "some papers are unusually discordant", and only the second is an
  error signal. Split the statistic by the sign of the residual and test each half
  against its own null.

Writes paper_inversion_decompose.json.
"""

import json
import random
import sys
from collections import defaultdict

from paper_inversion import (MIN_DECISIVE, N_PERM, SEED, build_observations,
                             load_graph, score, score_permuted)
from paper_inversion_control import ancestor_sets, thin


def split_dispersion(dis, dec, eligible, base):
    """-> (total, positive-residual part, negative-residual part)."""
    pos = neg = 0.0
    for p in eligible:
        e_p = dec.get(p, 0) * base
        r = dis.get(p, 0) - e_p
        c = r * r / max(e_p, 1.0)
        if r > 0:
            pos += c
        else:
            neg += c
    return pos + neg, pos, neg


def run(edges, label, n_perm=N_PERM):
    per_paper, _ = score(edges)
    n_dec = sum(v[0] for v in per_paper.values())
    n_dis = sum(v[1] for v in per_paper.values())
    if n_dec == 0:
        print(f"[{label}] no decisive observations")
        return None
    base = n_dis / n_dec
    eligible = sorted(p for p, v in per_paper.items() if v[0] >= MIN_DECISIVE)
    obs_dis = {p: per_paper[p][1] for p in eligible}
    obs_dec = {p: per_paper[p][0] for p in eligible}
    tot, pos, neg = split_dispersion(obs_dis, obs_dec, eligible, base)

    rng = random.Random(SEED)
    nt, npos, nneg = [], [], []
    for _ in range(n_perm):
        d, c = score_permuted(edges, rng)
        t, p_, n_ = split_dispersion(d, c, eligible, base)
        nt.append(t); npos.append(p_); nneg.append(n_)

    def emp(obs, null):
        return (sum(1 for s in null if s >= obs) + 1) / (n_perm + 1)

    res = {
        "label": label,
        "n_edges": len(edges),
        "n_obs": sum(len(e["obs"]) for e in edges),
        "n_decisive": n_dec, "n_disagree": n_dis,
        "base_rate": round(base, 4), "n_testable": len(eligible),
        "total": {"obs": round(tot, 2), "null_mean": round(sum(nt) / n_perm, 2),
                  "p": round(emp(tot, nt), 5)},
        "excess_disagreement": {"obs": round(pos, 2),
                                "null_mean": round(sum(npos) / n_perm, 2),
                                "p": round(emp(pos, npos), 5)},
        "excess_agreement": {"obs": round(neg, 2),
                             "null_mean": round(sum(nneg) / n_perm, 2),
                             "p": round(emp(neg, nneg), 5)},
    }
    print(f"\n[{label}] {len(edges)} edges, {res['n_obs']} obs, {n_dec} decisive, "
          f"{n_dis} disagree ({base:.1%}), {len(eligible)} testable papers")
    for k in ("total", "excess_disagreement", "excess_agreement"):
        r = res[k]
        print(f"  {k:22s} observed {r['obs']:8.1f}  null {r['null_mean']:8.1f}  "
              f"p = {r['p']:.4f}")
    return res


def main():
    graph = load_graph()
    edges = build_observations(graph)
    rel = ancestor_sets(graph)
    thinned, _ = thin(edges, rel)

    out = {}
    out["full"] = run(edges, "full")
    out["containment_controlled"] = run(thinned, "containment-controlled")

    ge3 = [e for e in edges if len(e["obs"]) >= 3]
    out["ge3_papers"] = run(ge3, ">=3 papers per edge")

    ge3_thin = [e for e in thinned if len(e["obs"]) >= 3]
    out["ge3_papers_containment_controlled"] = run(
        ge3_thin, ">=3 papers, containment-controlled")

    # who drives it, on the strictest set that still has power
    strict = ge3_thin
    per_paper, detail = score(strict)
    n_dec = sum(v[0] for v in per_paper.values())
    base = sum(v[1] for v in per_paper.values()) / n_dec if n_dec else 0
    rows = []
    for p, v in per_paper.items():
        if v[0] < MIN_DECISIVE:
            continue
        exp = v[0] * base
        rows.append({"paper": p, "title": graph["papers"][p]["title"][:70],
                     "n_decisive": v[0], "n_disagree": v[1],
                     "expected": round(exp, 2),
                     "residual": round(v[1] - exp, 2)})
    rows.sort(key=lambda r: -r["residual"])
    print("\nMost discordant papers on the strictest set "
          "(>=3 papers/edge, containment-controlled):")
    for r in rows[:8]:
        print(f"  +{r['residual']:5.2f}  {r['n_disagree']:>2}/{r['n_decisive']:<3} "
              f"exp {r['expected']:5.2f}  {r['title'][:58]}")
    print("Most concordant:")
    for r in rows[-5:]:
        print(f"  {r['residual']:6.2f}  {r['n_disagree']:>2}/{r['n_decisive']:<3} "
              f"exp {r['expected']:5.2f}  {r['title'][:58]}")
    out["strict_paper_residuals"] = rows

    with open("paper_inversion_decompose.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote paper_inversion_decompose.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
