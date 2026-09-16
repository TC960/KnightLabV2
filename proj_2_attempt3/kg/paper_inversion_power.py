#!/usr/bin/env python3
"""Is the strict-set null a refutation, or just 53% less data?

State of the question after `paper_inversion_decompose.py`:

  full set                              1367 decisive   p = 0.0003
  containment-controlled                 975 decisive   p = 0.0013
  >=3 papers per edge                    907 decisive   p = 0.0003
  BOTH controls at once                  637 decisive   p = 0.0582

Each control alone leaves the paper-level dispersion significant; both together
put it just outside 0.05. Two readings, and they lead to opposite claims:

  (a) the effect was riding on structure, and removing both artifacts kills it;
  (b) the effect is real and the strict set is simply too small to see it.

These are distinguishable without any new modelling assumption. Take the
containment-controlled set -- where the effect IS significant -- and randomly
throw away edges until only as many decisive observations remain as the strict
set has. Re-run the identical test on each subsample. The fraction that still
reaches p < 0.05 is exactly the power the strict test had. If that fraction is
low, reading (b) holds and p = 0.058 is what an underpowered look at a real effect
is supposed to produce. If it is high, reading (a) holds and the strict null means
something.

Subsampling by EDGE, not by observation, because dropping observations from an
edge would change that edge's up/down counts and therefore the very quantity the
within-edge null conditions on.

Writes paper_inversion_power.json.
"""

import json
import random
import sys

from paper_inversion import MIN_DECISIVE, build_observations, load_graph, score, \
    score_permuted
from paper_inversion_control import ancestor_sets, thin
from paper_inversion_decompose import split_dispersion

N_SUB = 200          # subsample replicates
N_PERM_SUB = 2000    # permutations per replicate (enough to resolve p ~ 0.05)
SEED = 20260910


def test_once(edges, n_perm, rng):
    """-> (p_total, p_excess_disagreement, n_decisive) or None if degenerate."""
    per_paper, _ = score(edges)
    n_dec = sum(v[0] for v in per_paper.values())
    if n_dec == 0:
        return None
    base = sum(v[1] for v in per_paper.values()) / n_dec
    eligible = sorted(p for p, v in per_paper.items() if v[0] >= MIN_DECISIVE)
    if not eligible:
        return None
    tot, pos, _ = split_dispersion({p: per_paper[p][1] for p in eligible},
                                   {p: per_paper[p][0] for p in eligible},
                                   eligible, base)
    nt, npos = [], []
    for _ in range(n_perm):
        d, c = score_permuted(edges, rng)
        t, p_, _ = split_dispersion(d, c, eligible, base)
        nt.append(t); npos.append(p_)
    pt = (sum(1 for s in nt if s >= tot) + 1) / (n_perm + 1)
    pp = (sum(1 for s in npos if s >= pos) + 1) / (n_perm + 1)
    return pt, pp, n_dec


def main():
    graph = load_graph()
    edges = build_observations(graph)
    rel = ancestor_sets(graph)
    thinned, _ = thin(edges, rel)
    strict = [e for e in thinned if len(e["obs"]) >= 3]

    per_strict, _ = score(strict)
    target_dec = sum(v[0] for v in per_strict.values())
    per_thin, _ = score(thinned)
    full_dec = sum(v[0] for v in per_thin.values())
    print(f"containment-controlled set: {len(thinned)} edges, {full_dec} decisive")
    print(f"strict set (also >=3 papers): {len(strict)} edges, {target_dec} decisive")
    print(f"subsampling the first down to ~{target_dec} decisive, "
          f"{N_SUB} replicates x {N_PERM_SUB} permutations\n")

    edge_dec = []
    for ed in thinned:
        pp, _ = score([ed])
        edge_dec.append(sum(v[0] for v in pp.values()))
    assert sum(edge_dec) == full_dec, (sum(edge_dec), full_dec)

    rng = random.Random(SEED)
    hits_t = hits_p = 0
    ps_t, ps_p = [], []
    done = 0
    for rep in range(N_SUB):
        # drop whole edges, in random order, until decisive count is at or below
        # the strict set's
        # each edge's decisive count is independent of the others (score() sums
        # over edges), so the running total can be kept without rescoring
        order = list(range(len(thinned)))
        rng.shuffle(order)
        keep, cur = set(order), full_dec
        for i in order:
            if cur <= target_dec:
                break
            keep.discard(i)
            cur -= edge_dec[i]
        sub = [thinned[j] for j in sorted(keep)]
        r = test_once(sub, N_PERM_SUB, rng)
        if r is None:
            continue
        pt, pex, ndec = r
        ps_t.append(pt); ps_p.append(pex)
        hits_t += pt < 0.05
        hits_p += pex < 0.05
        done += 1
        if (rep + 1) % 25 == 0:
            print(f"  {rep+1}/{N_SUB} replicates: "
                  f"total p<0.05 in {hits_t}/{done} ({hits_t/done:.0%}), "
                  f"excess-disagreement p<0.05 in {hits_p}/{done} "
                  f"({hits_p/done:.0%})")

    ps_t.sort(); ps_p.sort()
    med_t = ps_t[len(ps_t) // 2]
    med_p = ps_p[len(ps_p) // 2]
    print(f"\nPOWER of a strict-sized look at the containment-controlled effect:")
    print(f"  total dispersion       : {hits_t}/{done} = {hits_t/done:.1%} "
          f"(median p {med_t:.3f})")
    print(f"  excess disagreement    : {hits_p}/{done} = {hits_p/done:.1%} "
          f"(median p {med_p:.3f})")
    print(f"\nobserved strict-set p was 0.0582 (total) / 0.1145 (excess disagreement)")

    out = {
        "n_subsamples": done, "n_perm_per_subsample": N_PERM_SUB, "seed": SEED,
        "decisive_full_controlled": full_dec,
        "decisive_strict_target": target_dec,
        "power_total_dispersion": round(hits_t / done, 4),
        "power_excess_disagreement": round(hits_p / done, 4),
        "median_p_total": round(med_t, 4),
        "median_p_excess": round(med_p, 4),
        "observed_strict_p_total": 0.0582,
        "observed_strict_p_excess": 0.1145,
        "p_distribution_total": [round(x, 4) for x in ps_t],
        "p_distribution_excess": [round(x, 4) for x in ps_p],
    }
    with open("paper_inversion_power.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote paper_inversion_power.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
