#!/usr/bin/env python3
"""How big is the unexplained paper-level effect?

Twenty-four variables have now been tested against paper discordance and all
twenty-four are null (nine study-design in `paper_discordance_offset.py`, fifteen
wet-lab and bioinformatics in `methods_discordance.py`), while the paper-level
variance itself is solid at p = 0.0003. "Real but unexplained" is only actionable
with a magnitude attached, so this puts one on it.

MODEL. Paper p contributes dec_p decisive observations; under the within-edge null
its disagreements have mean E_p and a variance the permutation ensemble measures
directly. Under the alternative each paper carries a multiplicative discordance
propensity r_p with mean 1 and standard deviation sigma, so

    Var(dis_p) = Var_null(dis_p) + (E_p * sigma)^2

Summing over papers and solving for sigma gives a moment estimate of the spread of
paper propensities. Multiplying by the corpus disagreement rate turns it into
percentage points of discordance, which is the unit a reader thinks in.

A cluster bootstrap over PAPERS gives the interval -- resampling papers, not
observations, because the paper is the unit the effect lives at.
"""

import json
import random
import statistics
import sys
from collections import defaultdict

from paper_inversion import MIN_DECISIVE, N_PERM, build_observations, load_graph, \
    score, score_permuted
from paper_inversion_control import ancestor_sets, thin
from paper_discordance_offset import expectations

SEED = 20260910
N_BOOT = 2000


def main():
    graph = load_graph()
    edges = build_observations(graph)
    rel = ancestor_sets(graph)
    thinned, _ = thin(edges, rel)
    per_paper, _ = score(thinned)
    e_dec, e_dis = expectations(thinned)

    eligible = sorted(p for p, v in per_paper.items() if v[0] >= MIN_DECISIVE)
    print(f"{len(eligible)} testable papers on the containment-controlled set")

    # null variance of each paper's disagreement count, from the ensemble
    rng = random.Random(SEED)
    samples = defaultdict(list)
    n_perm = 8000
    for _ in range(n_perm):
        dis, _ = score_permuted(thinned, rng)
        for p in eligible:
            samples[p].append(dis.get(p, 0))
    null_var = {p: statistics.pvariance(samples[p]) for p in eligible}

    obs_sq = sum((per_paper[p][1] - e_dis[p]) ** 2 for p in eligible)
    null_sq = sum(null_var[p] for p in eligible)
    e_sq = sum(e_dis[p] ** 2 for p in eligible)
    excess = obs_sq - null_sq
    sigma = (excess / e_sq) ** 0.5 if excess > 0 else 0.0

    base = (sum(per_paper[p][1] for p in eligible)
            / sum(per_paper[p][0] for p in eligible))
    print(f"\nobserved sum of squared deviations : {obs_sq:.1f}")
    print(f"expected under the within-edge null: {null_sq:.1f}")
    print(f"excess                             : {excess:.1f}")
    print(f"\nsigma (SD of the paper propensity multiplier) = {sigma:.3f}")
    print(f"corpus disagreement rate = {base:.1%}")
    print(f"=> paper-level SD of discordance = {sigma * base * 100:.1f} "
          f"percentage points")
    print(f"=> a paper one SD above the mean disagrees "
          f"{base * (1 + sigma):.1%} of the time, one SD below "
          f"{base * (1 - sigma):.1%}")

    # cluster bootstrap over papers
    boots = []
    brng = random.Random(SEED + 1)
    for _ in range(N_BOOT):
        pick = [brng.choice(eligible) for _ in eligible]
        o = sum((per_paper[p][1] - e_dis[p]) ** 2 for p in pick)
        n = sum(null_var[p] for p in pick)
        e2 = sum(e_dis[p] ** 2 for p in pick)
        ex = o - n
        boots.append((ex / e2) ** 0.5 if ex > 0 and e2 else 0.0)
    boots.sort()
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]
    frac_zero = sum(1 for b in boots if b <= 0) / N_BOOT
    print(f"\ncluster bootstrap over papers ({N_BOOT} resamples):")
    print(f"  sigma 95% CI [{lo:.3f}, {hi:.3f}]  "
          f"(in points: [{lo*base*100:.1f}, {hi*base*100:.1f}])")
    print(f"  resamples with no excess variance at all: {frac_zero:.1%}")

    out = {"n_papers": len(eligible), "n_perm": n_perm, "seed": SEED,
           "observed_sumsq": round(obs_sq, 2), "null_sumsq": round(null_sq, 2),
           "excess": round(excess, 2), "sigma": round(sigma, 4),
           "base_rate": round(base, 4),
           "sd_points": round(sigma * base * 100, 2),
           "sigma_ci95": [round(lo, 4), round(hi, 4)],
           "sd_points_ci95": [round(lo * base * 100, 2), round(hi * base * 100, 2)],
           "bootstrap_frac_no_excess": round(frac_zero, 4),
           "n_boot": N_BOOT}
    with open("paper_effect_size.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote paper_effect_size.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
