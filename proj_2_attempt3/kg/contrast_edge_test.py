#!/usr/bin/env python3
"""Does an observation with no visible control contrast disagree with the
literature more than one that has one?

`contrast_edge_probe.py` labels each of the 3,077 (paper, taxon) observations
CLEAN / CANDIDATE / UNRESOLVED from the sentences that name that taxon in that
paper. This tests the label against the graph.

THE NULL IS DIFFERENT FROM EVERY OTHER TEST IN THIS PROJECT, and it has to be.
The standing rule is "shuffle at the PAPER level" because observations are not
independent. That rule exists for PAPER-level predictors. This predictor varies
WITHIN a paper -- that is the entire point of it -- so a paper-level shuffle would
destroy nothing and test nothing. The exchangeable null here is:

    permute the labels WITHIN each paper, holding that paper's count of each
    label fixed,

which preserves every paper's size, its edge set, and its overall discordance, and
destroys only the association between WHICH of a paper's observations lack a
visible control contrast and which of them disagree. Papers contributing a single
observation, or all-one-label papers, contribute nothing to the null and nothing
to the signal; they are counted and reported.

Outcome is the same leave-one-out disagreement used throughout, with the
closed-form within-edge expectation as the offset so edge depth and contestedness
are absorbed (`paper_discordance_offset.py`).

Writes contrast_edge_test.json.
"""
import json, random
from collections import defaultdict
from pathlib import Path

from paper_inversion import build_observations, load_graph
from paper_inversion_control import ancestor_sets, thin

HERE = Path(__file__).parent
N_PERM = 20000
SEED = 20260919

def observations(graph):
    """-> list of dicts: paper idx, key, decisive, disagree, expected-disagree."""
    thinned, _ = thin(build_observations(graph), ancestor_sets(graph))
    out = []
    for ed in thinned:
        obs = ed["obs"]
        n = len(obs)
        ne = sum(1 for _, d in obs if d == "e")
        nd = n - ne
        lo = min(ne, nd)
        p_dis = 1.0 if ne == nd else (lo / n if abs(ne - nd) == 1 else lo / n)
        for pi, di in obs:
            oe, od = ne - (di == "e"), nd - (di == "d")
            if oe == od:
                dec, dis = 0, 0
            else:
                dec = 1
                dis = int(di != ("e" if oe > od else "d"))
            out.append({"paper": pi, "taxon_key": ed["key"][0],
                        "disease": ed["disease"], "dec": dec, "dis": dis,
                        "e_dis": p_dis})
    return out

def oe(rows, lab, want):
    o = sum(r["dis"] for r, l in zip(rows, lab) if l == want)
    e = sum(r["e_dis"] for r, l in zip(rows, lab) if l == want)
    return (o / e if e else float("nan")), o, e

def main():
    graph = load_graph()
    papers = graph["papers"]
    probe = {(r["paper"], r.get("taxon_key"), r["disease"]): r["verdict"]
             for r in json.load(open(HERE / "contrast_edge_probe.json"))["rows"]}

    rows, lab, by_paper = [], [], defaultdict(list)
    for o in observations(graph):
        v = probe.get((papers[o["paper"]]["title"], o["taxon_key"], o["disease"]))
        if v not in ("CLEAN", "CANDIDATE"):
            continue
        if o["e_dis"] <= 0:
            continue
        by_paper[o["paper"]].append(len(rows))
        rows.append(o)
        lab.append(v)

    informative = [p for p, ix in by_paper.items()
                   if len({lab[i] for i in ix}) > 1]
    a, oa, ea = oe(rows, lab, "CANDIDATE")
    b, ob, eb = oe(rows, lab, "CLEAN")
    obs_diff = a - b

    rng = random.Random(SEED)
    perm = list(lab)
    null = []
    groups = [ix for ix in by_paper.values() if len(ix) > 1]
    for _ in range(N_PERM):
        for ix in groups:
            vals = [perm[i] for i in ix]
            rng.shuffle(vals)
            for i, v in zip(ix, vals):
                perm[i] = v
        x, _, _ = oe(rows, perm, "CANDIDATE")
        y, _, _ = oe(rows, perm, "CLEAN")
        if x == x and y == y:
            null.append(x - y)
    p = (sum(1 for d in null if abs(d) >= abs(obs_diff)) + 1) / (len(null) + 1)
    null.sort()
    mde = max(abs(null[int(0.025 * len(null))]), abs(null[int(0.975 * len(null))]))

    res = {"n_observations_scored": len(rows),
           "n_candidate": lab.count("CANDIDATE"), "n_clean": lab.count("CLEAN"),
           "n_papers": len(by_paper),
           "n_papers_informative_for_null": len(informative),
           "oe_candidate": round(a, 4), "obs_candidate": oa, "exp_candidate": round(ea, 2),
           "oe_clean": round(b, 4), "obs_clean": ob, "exp_clean": round(eb, 2),
           "diff": round(obs_diff, 4), "p": round(p, 5),
           "mde_abs_diff": round(mde, 4), "n_perm": len(null), "seed": SEED,
           "null_type": "within-paper label shuffle (predictor varies within paper)"}
    json.dump(res, open(HERE / "contrast_edge_test.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
