#!/usr/bin/env python3
"""Task 3.2 -- consume the containment links. Are parent/child direction
conflicts real biology, or rank confusion?

The project's load-bearing design decision is that containment is modelled and
never collapsed, justified by one example: in Parkinson's, *Lachnospiraceae*
(family) is depleted in 8 of the 9 papers reporting it while *Hungatella* (a
genus inside it) is enriched in 6 of 7. GraphRAG later counted the general case -- 903 (taxon,
disease) edges have a parent edge in the same disease and 229 of those point the
opposite way -- but nobody asked whether those 229 are signal or artefact.

TWO QUESTIONS, deliberately separated because only one of them is inferential.

1. DETERMINISTIC. For each opposite-direction parent/child pair, is the conflict
   asserted INSIDE a single paper -- one study reporting the family down and the
   genus up -- or does it only appear when you pool papers that never measured
   both? A within-paper conflict cannot be rank confusion: the same authors, the
   same cohort, the same pipeline reported both numbers. A conflict visible only
   across papers is a weaker thing, consistent with cohort differences, with
   different resolution limits, or with genuine rank confusion. This is a string
   comparison, not a judgement call -- the kind of test this project trusts after
   an LLM adjudication of 18 self-contradictions got 4 of 6 verdicts wrong.

2. INFERENTIAL. Do papers agree on direction MORE between taxonomically related
   taxa than between unrelated ones? Null: shuffle each paper's direction labels
   across the taxa that paper reported, preserving that paper's own up/down
   counts. This is a PAPER-LEVEL null -- it destroys the kinship-direction
   association while keeping each paper's overall enrichment propensity, so a
   paper that simply reports mostly-depleted taxa cannot manufacture agreement.
   Pair-level shuffling has produced three false positives on record here.

Usage: python3 analyze_rank_conflict.py [--iters 20000]
"""
import argparse
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    G = json.load(open(os.path.join(HERE, "graph.json")))
    papers = G["papers"]
    # node id -> disease -> edge
    by_node = defaultdict(dict)
    for e in G["edges"]:
        by_node[e["source"]][e["disease"]] = e
    label = {n["id"]: n.get("label", n["id"]) for n in G["nodes"]}
    return G, papers, by_node, label


def majority(e):
    if e["n_up"] > e["n_down"]:
        return "e"
    if e["n_down"] > e["n_up"]:
        return "d"
    return None            # exact tie -- no majority to conflict with


def q1_deterministic(G, by_node, label):
    """Classify every parent/child pair sharing a disease."""
    out = {"same": 0, "opposite": 0, "tie": 0, "pairs": []}
    for h in G["hierarchy"]:
        P, C = h["parent"], h["child"]
        # A placeholder child ("t:ph:Prevotella 9") is a SILVA rank label split
        # out of its parent, not an independently named taxon. Kept separate:
        # a parent/placeholder disagreement is a different animal from a real
        # family-vs-genus disagreement and must not be pooled with it.
        kind = "placeholder" if C.startswith("t:ph:") or P.startswith("t:ph:") else "named"
        shared = set(by_node.get(P, {})) & set(by_node.get(C, {}))
        for dis in sorted(shared):
            pe, ce = by_node[P][dis], by_node[C][dis]
            pm, cm = majority(pe), majority(ce)
            if pm is None or cm is None:
                out["tie"] += 1
                continue
            if pm == cm:
                out["same"] += 1
                continue
            out["opposite"] += 1

            # --- the deterministic part: who asserts the conflict? ---
            pdir = {ev["i"]: ev["d"] for ev in pe["ev"]}
            cdir = {ev["i"]: ev["d"] for ev in ce["ev"]}
            both = sorted(set(pdir) & set(cdir))
            within = [i for i in both if pdir[i] != cdir[i]]
            agree_within = [i for i in both if pdir[i] == cdir[i]]
            if within:
                verdict = "within_paper"      # one study reports both directions
            elif both:
                verdict = "cross_paper_only"  # shared papers exist and all AGREE;
                                              # the conflict comes from papers that
                                              # measured only one of the two
            else:
                verdict = "no_shared_paper"   # no study ever measured both
            out["pairs"].append({
                "kind": kind, "disease": dis,
                "parent": label.get(P, P), "child": label.get(C, C),
                "parent_rank": h.get("parent_rank"), "child_rank": h.get("child_rank"),
                "parent_dir": pm, "child_dir": cm,
                "parent_papers": pe["n_papers"], "child_papers": ce["n_papers"],
                "n_shared": len(both), "n_within_conflict": len(within),
                "n_shared_agree": len(agree_within),
                "verdict": verdict,
                "witnesses": [G["papers"][i]["title"] for i in within[:3]],
            })
    return out


def q2_inferential(G, by_node, iters, seed=42):
    """Within a paper, do related taxa agree on direction more than unrelated ones?

    Unit of observation: (paper, disease, taxon-pair). Both arms are drawn from
    the SAME paper, so paper-level confounds -- cohort, country, pipeline, and
    that paper's overall tendency to report enrichment -- are differenced out by
    construction before any shuffling happens.
    """
    kin = set()
    for h in G["hierarchy"]:
        kin.add((h["parent"], h["child"]))
        kin.add((h["child"], h["parent"]))

    # paper -> disease -> {node_id: direction}
    calls = defaultdict(lambda: defaultdict(dict))
    for e in G["edges"]:
        for ev in e["ev"]:
            calls[ev["i"]][e["disease"]][e["source"]] = ev["d"]

    # Pre-extract the comparable units once; the permutation only relabels.
    units = []   # (list_of_nodes, list_of_dirs, list_of_related_pair_indices)
    for pi, dmap in calls.items():
        for dis, nd in dmap.items():
            nodes = sorted(nd)
            if len(nodes) < 2:
                continue
            rel = [(a, b) for a in range(len(nodes)) for b in range(a + 1, len(nodes))
                   if (nodes[a], nodes[b]) in kin]
            if not rel:
                continue          # no related pair -> contributes to neither arm
            unrel = [(a, b) for a in range(len(nodes)) for b in range(a + 1, len(nodes))
                     if (nodes[a], nodes[b]) not in kin]
            units.append(([nd[n] for n in nodes], rel, unrel))

    def stat(dirsets):
        ra = rn = ua = un = 0
        for dirs, rel, unrel in dirsets:
            for a, b in rel:
                rn += 1; ra += dirs[a] == dirs[b]
            for a, b in unrel:
                un += 1; ua += dirs[a] == dirs[b]
        return (ra / rn if rn else 0), (ua / un if un else 0), rn, un

    obs_r, obs_u, n_rel, n_unrel = stat(units)
    obs_gap = obs_r - obs_u

    rng = random.Random(seed)
    ge = 0
    null_gaps = []
    for _ in range(iters):
        shuffled = []
        for dirs, rel, unrel in units:
            d2 = dirs[:]                 # preserve this paper's up/down counts
            rng.shuffle(d2)
            shuffled.append((d2, rel, unrel))
        g = stat(shuffled)
        gap = g[0] - g[1]
        null_gaps.append(gap)
        if gap >= obs_gap:
            ge += 1
    mean = sum(null_gaps) / len(null_gaps)
    sd = (sum((x - mean) ** 2 for x in null_gaps) / len(null_gaps)) ** 0.5
    return {
        "n_units": len(units), "n_related_pairs": n_rel, "n_unrelated_pairs": n_unrel,
        "agree_related": round(obs_r, 4), "agree_unrelated": round(obs_u, 4),
        "gap": round(obs_gap, 4),
        "null_mean": round(mean, 5), "null_sd": round(sd, 5),
        "z": round((obs_gap - mean) / sd, 3) if sd else None,
        "p_one_sided": round((ge + 1) / (iters + 1), 5),
        "mde_1.96sd": round(1.96 * sd, 4),
        "iters": iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    a = ap.parse_args()
    G, papers, by_node, label = load()

    q1 = q1_deterministic(G, by_node, label)
    named = [p for p in q1["pairs"] if p["kind"] == "named"]
    ph = [p for p in q1["pairs"] if p["kind"] == "placeholder"]
    def tally(ps):
        t = defaultdict(int)
        for p in ps:
            t[p["verdict"]] += 1
        return dict(t)

    print("=== Q1  parent/child pairs sharing a disease ===")
    print(f"same direction   {q1['same']}")
    print(f"opposite         {q1['opposite']}")
    print(f"no majority(tie) {q1['tie']}")
    print(f"\nopposite, named parent+child ({len(named)}): {tally(named)}")
    print(f"opposite, placeholder        ({len(ph)}): {tally(ph)}")

    print(f"\n=== Q2  do related taxa agree more, WITHIN a paper? ({a.iters} perms) ===")
    q2 = q2_inferential(G, by_node, a.iters)
    for k, v in q2.items():
        print(f"  {k:20s} {v}")

    out = {"q1": {"same": q1["same"], "opposite": q1["opposite"], "tie": q1["tie"],
                  "named_verdicts": tally(named), "placeholder_verdicts": tally(ph),
                  "pairs": q1["pairs"]},
           "q2": q2}
    p = os.path.join(HERE, "rank_conflict.json")
    json.dump(out, open(p, "w"), indent=1, sort_keys=True)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
