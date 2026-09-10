#!/usr/bin/env python3
"""Is any single paper systematically INVERTED in our extractions?

The motivating case is external. `FINDINGS_db_conflicts.md` showed that all five
places where Disbiome and Peryton contradict each other trace to ONE paper
(PMID 27703453), and that Disbiome has that paper's case/control assignment
swapped -- every taxon flipped together, which is the fingerprint of a group-label
error rather than five independent curation mistakes.

That is a failure mode our own extractor can have too: an LLM reading a paper
whose group labels are unusual ("Group A" / "Group B", or a table whose control
column comes second) can invert the whole paper at once. Nobody has asked whether
it did. This asks, deterministically, against the graph's own structure.

METHOD
------
For every (taxon, disease) edge with >= 2 contributing papers, and every paper i
on that edge, compute the LEAVE-ONE-OUT majority direction of the *other* papers.
If the others are tied or absent the observation is not decisive and is dropped.
Otherwise the observation is scored agree / disagree.

Per paper: n_decisive, n_disagree, rate.

NULL (this is the part that matters)
------------------------------------
Papers differ in how many observations they contribute and in how contested the
edges they sit on are, so a raw rate comparison is meaningless. The null holds
BOTH fixed: within each edge, the multiset of directions is preserved exactly and
only the assignment of those directions to the papers reporting that edge is
shuffled. So each edge keeps its up/down counts, each paper keeps its exact set of
edges, and the only thing destroyed is any paper-level consistency in *which*
papers hold the minority direction -- which is precisely the alternative
hypothesis.

Two statistics, both from the same permutation ensemble:
  * GLOBAL: is minority-direction status clustered by paper at all?  Dispersion
    statistic sum_p (n_disagree_p - E_p)^2 / max(E_p, 1) over papers.
  * PER-PAPER: one-sided empirical p for each paper's n_disagree, BH-corrected
    across the papers eligible to be tested.

POWER
-----
Reported explicitly, by simulation: for each eligible paper, flip every one of its
observations and re-run the per-paper test to see whether a fully-inverted copy of
that paper would have been caught. "No inverted papers" is only worth saying
alongside the number of papers where an inversion would have been visible.

Writes paper_inversion.json.  CPU-only, no network, no MAIN_DATA.
"""

import json
import random
import sys
from collections import defaultdict

N_PERM = 20000
SEED = 20260910
MIN_DECISIVE = 4  # a paper needs this many decisive observations to be testable


def load_graph(path="graph.json"):
    with open(path) as fh:
        return json.load(fh)


def build_observations(graph):
    """-> list of edges, each a list of (paper_idx, direction) with >=2 papers."""
    edges = []
    for e in graph["edges"]:
        obs = [(o["i"], o["d"]) for o in e["ev"]]
        if len(obs) < 2:
            continue
        # ev is deduplicated to one entry per paper per edge (verified: 0 edges
        # carry two entries for the same paper), so no collapsing is needed here.
        edges.append({
            "key": (e["taxon_key"], e["disease"]),
            "taxon": e["taxon"],
            "disease": e["disease"],
            "source": e["source"],      # taxon node id, for containment lookups
            "obs": obs,
        })
    return edges


def score(edges):
    """Leave-one-out agreement. -> {paper: [n_decisive, n_disagree]}, detail list."""
    per_paper = defaultdict(lambda: [0, 0])
    detail = []
    for ed in edges:
        obs = ed["obs"]
        n_e = sum(1 for _, d in obs if d == "e")
        n_d = len(obs) - n_e
        for pi, di in obs:
            oe = n_e - (di == "e")
            od = n_d - (di == "d")
            if oe == od:            # others tied (or none) -> not decisive
                continue
            maj = "e" if oe > od else "d"
            agree = (di == maj)
            per_paper[pi][0] += 1
            if not agree:
                per_paper[pi][1] += 1
                detail.append({"paper": pi, "taxon": ed["taxon"],
                               "disease": ed["disease"], "said": di,
                               "others": f"{oe}e/{od}d"})
    return per_paper, detail


def score_permuted(edges, rng):
    """Same statistic under the within-edge direction shuffle."""
    per_paper = defaultdict(int)
    dec = defaultdict(int)
    for ed in edges:
        obs = ed["obs"]
        dirs = [d for _, d in obs]
        rng.shuffle(dirs)
        n_e = dirs.count("e")
        n_d = len(dirs) - n_e
        for (pi, _), di in zip(obs, dirs):
            oe = n_e - (di == "e")
            od = n_d - (di == "d")
            if oe == od:
                continue
            dec[pi] += 1
            if di != ("e" if oe > od else "d"):
                per_paper[pi] += 1
    return per_paper, dec


def main():
    graph = load_graph()
    papers = graph["papers"]
    edges = build_observations(graph)
    obs_total = sum(len(e["obs"]) for e in edges)
    print(f"{len(edges)} multi-paper edges, {obs_total} observations, "
          f"{len(papers)} papers in table")

    per_paper, detail = score(edges)
    n_dec_total = sum(v[0] for v in per_paper.values())
    n_dis_total = sum(v[1] for v in per_paper.values())
    base = n_dis_total / n_dec_total if n_dec_total else 0.0
    print(f"decisive observations: {n_dec_total}; disagreements with the "
          f"leave-one-out majority: {n_dis_total} ({base:.1%})")

    eligible = sorted(p for p, v in per_paper.items() if v[0] >= MIN_DECISIVE)
    print(f"papers with >= {MIN_DECISIVE} decisive observations: {len(eligible)}")

    rng = random.Random(SEED)
    # Permutation ensemble. Store per-paper null counts for the eligible papers
    # and the global dispersion statistic.
    null_counts = {p: [] for p in eligible}
    null_disp = []
    exp_dec = defaultdict(float)
    for _ in range(N_PERM):
        dis, dec = score_permuted(edges, rng)
        for p in eligible:
            null_counts[p].append(dis.get(p, 0))
        for p, v in dec.items():
            exp_dec[p] += v
        # dispersion over the same eligible set, expectation approximated by the
        # observed corpus base rate applied to each paper's null decisive count
        s = 0.0
        for p in eligible:
            e_p = max(dec.get(p, 0) * base, 1e-9)
            s += (dis.get(p, 0) - e_p) ** 2 / max(e_p, 1.0)
        null_disp.append(s)

    obs_disp = 0.0
    for p in eligible:
        e_p = max(per_paper[p][0] * base, 1e-9)
        obs_disp += (per_paper[p][1] - e_p) ** 2 / max(e_p, 1.0)
    p_global = (sum(1 for s in null_disp if s >= obs_disp) + 1) / (N_PERM + 1)
    print(f"\nGLOBAL dispersion  observed {obs_disp:.1f}  "
          f"null mean {sum(null_disp)/len(null_disp):.1f}  p = {p_global:.4f}")

    # per-paper one-sided p, then BH
    rows = []
    for p in eligible:
        obs_dis = per_paper[p][1]
        nc = null_counts[p]
        pv = (sum(1 for c in nc if c >= obs_dis) + 1) / (N_PERM + 1)
        rows.append({
            "paper": p,
            "title": papers[p]["title"] if p < len(papers) else "?",
            "n_decisive": per_paper[p][0],
            "n_disagree": obs_dis,
            "rate": round(obs_dis / per_paper[p][0], 3),
            "null_mean": round(sum(nc) / len(nc), 2),
            "p": round(pv, 5),
        })
    rows.sort(key=lambda r: r["p"])
    m = len(rows)
    for rank, r in enumerate(rows, 1):
        r["q"] = round(min(1.0, r["p"] * m / rank), 4)
    # enforce BH monotonicity
    for i in range(m - 2, -1, -1):
        rows[i]["q"] = min(rows[i]["q"], rows[i + 1]["q"])

    print(f"\nTop candidates by permutation p (BH across {m} testable papers):")
    print(f"{'p':>5} {'dec':>4} {'dis':>4} {'rate':>6} {'null':>6} {'pval':>8} "
          f"{'q':>7}  title")
    for r in rows[:12]:
        print(f"{r['paper']:>5} {r['n_decisive']:>4} {r['n_disagree']:>4} "
              f"{r['rate']:>6.2f} {r['null_mean']:>6.2f} {r['p']:>8.4f} "
              f"{r['q']:>7.4f}  {r['title'][:60]}")

    hits = [r for r in rows if r["q"] < 0.05]
    print(f"\npapers surviving BH q<0.05: {len(hits)}")

    # ---- POWER: would a fully inverted paper have been caught? ----
    print("\nPower: flipping each eligible paper's every observation and re-testing")
    power = []
    for p in eligible:
        # under full inversion the paper disagrees on every decisive observation
        # that its own vote does not itself define; recompute honestly by flipping
        flipped = []
        for ed in edges:
            obs = [(pi, ("d" if d == "e" else "e") if pi == p else d)
                   for pi, d in ed["obs"]]
            flipped.append({**ed, "obs": obs})
        fp, _ = score(flipped)
        f_dis = fp[p][1]
        f_dec = fp[p][0]
        nc = null_counts[p]
        pv = (sum(1 for c in nc if c >= f_dis) + 1) / (N_PERM + 1)
        power.append({"paper": p, "n_decisive": f_dec, "n_disagree_if_flipped": f_dis,
                      "p_if_flipped": round(pv, 5)})
    # BH over the flipped p-values, one paper at a time, is not the right frame;
    # report the raw detectability: how many would clear the BH threshold that the
    # real run's smallest p had to clear (p <= 0.05/m is the conservative bound).
    strict = 0.05 / m
    detectable_strict = sum(1 for w in power if w["p_if_flipped"] <= strict)
    detectable_raw = sum(1 for w in power if w["p_if_flipped"] <= 0.05)
    print(f"  fully-inverted copy would reach raw p<=0.05 for "
          f"{detectable_raw}/{m} eligible papers")
    print(f"  ... and clear the Bonferroni-strict bound {strict:.5f} for "
          f"{detectable_strict}/{m}")

    out = {
        "n_edges_multipaper": len(edges),
        "n_observations": obs_total,
        "n_decisive": n_dec_total,
        "n_disagree": n_dis_total,
        "base_disagreement_rate": round(base, 4),
        "min_decisive": MIN_DECISIVE,
        "n_testable_papers": m,
        "n_perm": N_PERM,
        "seed": SEED,
        "global_dispersion": {"observed": round(obs_disp, 2),
                              "null_mean": round(sum(null_disp) / len(null_disp), 2),
                              "p": round(p_global, 5)},
        "papers": rows,
        "n_significant_bh05": len(hits),
        "power": {
            "detectable_raw_p05": detectable_raw,
            "detectable_bonferroni": detectable_strict,
            "bonferroni_threshold": round(strict, 6),
            "per_paper": power,
        },
        "disagreement_detail": detail,
    }
    with open("paper_inversion.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote paper_inversion.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
