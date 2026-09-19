#!/usr/bin/env python3
"""Does a bigger corpus make the paper-discordance question answerable?

THE QUESTION. `FINDINGS_paper_discordance.md` tested 24 study-design variables as
explanations for inter-study disagreement and got 24 nulls, all power-limited:
the between-paper SD of discordance is 3.4 points on a 27.6% base, with a
cluster-bootstrap CI of [0.0, 6.0] that includes zero. The standing note in
CLAUDE.md says everything here "is limited by n=272, which needs a GPU". This
script asks what a larger corpus would actually buy, and the answer is not the
obvious one in either direction.

WHY RAREFACTION AND NOT A LITERATURE REVIEW. The crux is arithmetic about how
edge structure scales with paper count, and we already hold the data to settle
it. Subsample k papers from the 271, rebuild the edge structure, and read the
scaling exponents off the result.

TWO EXPONENTS, AND THE SECOND ONE IS A TRAP.

    all edges            b = 0.840   sublinear
    decisive observations b = 1.525  SUPERlinear

Decisive means "on an edge with >=2 papers" -- the only observations the
discordance test can use. They grow superlinearly because an observation becomes
decisive only when its edge acquires a SECOND paper, which makes the pool a
coincidence count. This refutes the natural guess that new papers mostly mint new
single-paper edges: the 1-paper share falls monotonically 92.4% -> 77.7%.

THE TRAP: extrapolating b=1.525 to 10x predicts 33x more decisive observations.
That is impossible. An observation cannot be decisive more often than it is
observed, so decisive-per-paper is capped by observations-per-paper = 11.35. The
current 5.59 is already 49% of that ceiling, so the superlinear regime is half
spent and growth must bend to linear. Any power claim riding b=1.525 out to 10x
is fantasy, and this script exists partly to stop that claim being made.

THE POWER MODEL. Between-paper discordance is a VARIANCE COMPONENT, not a
difference in means, so it must be posed as one:

    signal    tau^2 = 0.034^2
    noise     sigma^2 = p(1-p)/n_decisive_per_paper,  p = 0.276
    SE(tau^2) ~ sigma^2 * sqrt(2/(K-1))

Result, with decisive-per-paper held to its ceiling:

    271 papers   z = 0.38     <- why 24 variables gave 24 nulls
    2x           z = 0.76
    5x           z = 1.55
    10x          z = 2.34     <- significant, not comfortable
    20x          z = 3.37

The z=0.38 at the current corpus independently reproduces the reported bootstrap
CI including zero, which is the check that the model is set up correctly.

READ IT AS: 10x is roughly the threshold, and it buys a marginal result about an
effect that stays small (I^2 only moves 3.1% -> 6.1% across the whole range).
Scaling buys the precision to measure a small effect; it does not make the effect
important.

    python3 corpus_scaling_power.py
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")

P_DISCORD = 0.276        # base discordance rate, FINDINGS_paper_discordance.md
SD_PAPER = 0.034         # between-paper SD of discordance (the signal)
N_DRAWS = 40
KS = [30, 60, 90, 120, 150, 180, 210, 240, 271]


def edge_structure():
    g = json.load(open(GRAPH))
    by_paper = collections.defaultdict(set)
    for ei, e in enumerate(g["edges"]):
        for ev in (e.get("ev") or []):
            by_paper[ev.get("i")].add(ei)
    return len(g["edges"]), by_paper


def rarefy(by_paper, rng):
    """edges / decisive-observations vs corpus size."""
    papers = sorted(by_paper)
    rows = []
    for k in KS:
        e_n, dec, single = [], [], []
        for _ in range(N_DRAWS):
            samp = rng.choice(papers, size=k, replace=False)
            cnt = collections.Counter()
            for p in samp:
                for ei in by_paper[p]:
                    cnt[ei] += 1
            e_n.append(len(cnt))
            dec.append(sum(v for v in cnt.values() if v >= 2))
            single.append(sum(1 for v in cnt.values() if v == 1) / len(cnt))
        rows.append((k, np.mean(e_n), np.mean(dec), np.mean(single)))
    return rows


def main():
    n_edges, by_paper = edge_structure()
    n_obs = sum(len(v) for v in by_paper.values())
    obs_per_paper = n_obs / len(by_paper)

    print(f"edges {n_edges}   observations {n_obs}   papers {len(by_paper)}")
    print(f"observations per paper {obs_per_paper:.2f}   "
          f"papers per edge {n_obs / n_edges:.3f}")

    rows = rarefy(by_paper, np.random.default_rng(0))
    print(f"\n{'k':>5} {'edges':>7} {'decisive':>9} {'dec/paper':>10} {'%1paper':>8}")
    for k, e, d, s in rows:
        print(f"{k:>5} {e:>7.0f} {d:>9.0f} {d / k:>10.2f} {100 * s:>7.1f}%")

    ks = np.array([r[0] for r in rows], float)
    b_edge = np.polyfit(np.log(ks), np.log([r[1] for r in rows]), 1)[0]
    b_dec = np.polyfit(np.log(ks), np.log([r[2] for r in rows]), 1)[0]
    print(f"\nscaling: edges ~ k^{b_edge:.3f} (sublinear)   "
          f"decisive ~ k^{b_dec:.3f} (SUPERlinear)")
    print(f"ceiling: dec/paper cannot exceed obs/paper = {obs_per_paper:.2f}; "
          f"now at {rows[-1][2] / 271:.2f} "
          f"({100 * rows[-1][2] / 271 / obs_per_paper:.0f}% of it)")

    # ---- power, with dec/paper held to the ceiling ----------------------
    tau2 = SD_PAPER ** 2
    print(f"\nvariance components: tau^2 = {tau2:.6f}, p = {P_DISCORD}")
    print(f"\n{'corpus':>8} {'K':>6} {'dec/pp':>7} {'sigma^2':>9} "
          f"{'I^2':>6} {'SE(tau2)':>10} {'z':>6}")
    for mult, dec_pp in [(1, 5.59), (2, 8.0), (5, 10.3), (10, 11.0), (20, 11.2)]:
        K = 271 * mult
        dec_pp = min(dec_pp, obs_per_paper)
        s2 = P_DISCORD * (1 - P_DISCORD) / dec_pp
        se = s2 * np.sqrt(2 / (K - 1))
        print(f"{K:>8} {K:>6} {dec_pp:>7.2f} {s2:>9.5f} "
              f"{100 * tau2 / (tau2 + s2):>5.1f}% {se:>10.6f} {tau2 / se:>6.2f}")

    print("\nz = 0.38 at the current corpus is the arithmetic explanation for")
    print("24 variables giving 24 nulls -- the design could not have found the")
    print("effect. 10x reaches z = 2.34: significant, not comfortable.")
    print("\nActionable consequence (203 S1.5): because edges grow at b=0.84")
    print("while decisive observations grow at b=1.53, a paper is worth much")
    print("more if it lands on an edge that already exists. Scale for DEPTH")
    print("(target known taxon-disease pairs), not for breadth.")


if __name__ == "__main__":
    main()
