#!/usr/bin/env python3
"""How much of the disagreement could a study-level variable explain, even in principle?

A study contributing several effect sizes is normal -- meta-analyses handle it
with multilevel models, not by dropping the study. So "papers appear in many
contested edges" is not itself a defect.

The real constraint is narrower and is a standard meta-analytic one:

    A STUDY-LEVEL MODERATOR CAN ONLY EXPLAIN BETWEEN-STUDY VARIATION.

If one paper reports taxon X enriched and taxon Y depleted in the same cohort,
no property of that paper -- recruitment setting, storage protocol, statistics --
can account for the difference. The paper is identical in both cases. The
contrast is WITHIN the study, and study-level variables are constant there.

Our design inherits this exactly: every edge from a paper carries the same 384-d
paper embedding, so a paper reporting both directions contributes +w and -w at an
identical score. Those contributions cancel.

This script partitions the comparable up-vs-down pairs by whether each paper has
a CONSISTENT stance across all its contested edges:

    both consistent  -> a genuine between-study contrast; addressable
    one mixed        -> partially addressable
    both mixed       -> not addressable by ANY paper-level feature

The addressable share is a hard ceiling on what contrast_experiment.py could ever
detect, independent of sample size, probe quality or embedding model. Report the
null against THAT denominator, not against all disagreement.

    python within_between_decomposition.py
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "within_between_decomposition.json")
MIN_SIDE = 2


def main():
    g = json.load(open(GRAPH))
    edges = []
    for e in g["edges"]:
        if not e.get("contested"):
            continue
        ev = e.get("ev") or e.get("evidence") or []
        up = {x["i"] for x in ev if x.get("d") == "e"}
        dn = {x["i"] for x in ev if x.get("d") == "d"}
        if len(up) >= MIN_SIDE and len(dn) >= MIN_SIDE:
            edges.append((up, dn))

    stance = collections.defaultdict(set)
    for up, dn in edges:
        for i in up:
            stance[i].add("e")
        for i in dn:
            stance[i].add("d")
    consistent = {i for i, v in stance.items() if len(v) == 1}
    mixed = {i for i, v in stance.items() if len(v) > 1}

    tot = cc = cm = mm = 0
    for up, dn in edges:
        for a in up:
            for b in dn:
                tot += 1
                k = (a in consistent) + (b in consistent)
                cc += k == 2
                cm += k == 1
                mm += k == 0

    clean = [(u, d) for u, d in edges if (u | d) <= consistent]

    print(f"contested edges with >={MIN_SIDE} papers per side : {len(edges)}")
    print(f"papers involved                                : {len(stance)}")
    print(f"  consistent stance across all their edges     : {len(consistent)} "
          f"({100*len(consistent)/len(stance):.0f}%)")
    print(f"  report BOTH directions somewhere             : {len(mixed)} "
          f"({100*len(mixed)/len(stance):.0f}%)")
    print()
    print(f"comparable up-vs-down pairs : {tot}")
    print(f"  both consistent  : {cc:>4} ({100*cc/tot:>4.1f}%)  ADDRESSABLE")
    print(f"  one mixed        : {cm:>4} ({100*cm/tot:>4.1f}%)  partial")
    print(f"  both mixed       : {mm:>4} ({100*mm/tot:>4.1f}%)  NOT addressable")
    print()
    print(f"edges where every paper is consistent : {len(clean)} of {len(edges)} "
          f"({sum(len(u)*len(d) for u, d in clean)} pairs)")
    print(f"""
INTERPRETATION

  {100*(1-cc/tot):.0f}% of the disagreement is within-paper. That is a CEILING, not a
  power problem -- more papers would not move it, because the limit is that a
  paper-level feature cannot vary within a paper.

  So the null result from contrast_experiment.py should be read as:
  "34 study-design concepts explain none of the {100*cc/tot:.0f}% of disagreement they
  could possibly explain", not "nothing explains the disagreement".

  The way past it is a finer unit of analysis: score the RELATION SENTENCE for a
  specific (paper, taxon, disease) instead of the whole paper. Then a paper
  reporting X up and Y down has two different feature vectors, each with the
  correct label, and the within-paper contrast becomes measurable.""")

    json.dump({
        "n_edges": len(edges), "n_papers": len(stance),
        "consistent": len(consistent), "mixed": len(mixed),
        "pairs_total": tot, "pairs_both_consistent": cc,
        "pairs_one_mixed": cm, "pairs_both_mixed": mm,
        "addressable_share": round(cc / tot, 4),
        "clean_edges": len(clean),
    }, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
