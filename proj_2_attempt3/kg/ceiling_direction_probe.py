#!/usr/bin/env python3
"""What is the best score ANY paper-level feature could get at predicting edge direction?

This is the feasibility question behind Sam's proposal: embed the free text of each
paper, then find which dimensions explain why papers disagree.

The proposal has a unit mismatch that is fatal before any model is fitted, and it
can be settled with arithmetic rather than a GPU:

  - the FEATURE is one vector per PAPER (a paper embedding)
  - the LABEL is one direction per (PAPER, TAXON, DISEASE) observation

A paper that reports Prevotella up and Lachnospiraceae down contributes two rows
with an IDENTICAL feature vector and OPPOSITE labels. No classifier can separate
them -- not logistic regression, not a transformer, not with more data. The best
any paper-level feature can do is predict each paper's own majority direction.

So the accuracy ceiling is fixed by the data:

    ceiling = sum over papers of (majority label count) / total observations

and the floor is the majority-class rate. If those two numbers are close, the
probe has almost no room to show anything, and a "significant" result inside that
band is measuring paper-level reporting bias, not the thing we care about.

    python ceiling_direction_probe.py
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "ceiling_direction_probe.json")


def observations(graph, contested_only=True):
    """One row per (paper, edge) with its reported direction."""
    rows = []
    for e in graph["edges"]:
        if contested_only and not e.get("contested"):
            continue
        for ev in (e.get("ev") or e.get("evidence") or []):
            i, d = ev.get("i"), ev.get("d")
            if i is None or d is None:
                continue
            rows.append((i, d))
    return rows


def report(rows, label):
    by_paper = collections.defaultdict(list)
    for i, d in rows:
        by_paper[i].append(d)

    counts = collections.Counter(d for _, d in rows)
    n = len(rows)
    floor = max(counts.values()) / n
    ceiling = sum(max(collections.Counter(v).values())
                  for v in by_paper.values()) / n
    mixed = [v for v in by_paper.values() if len(set(v)) > 1]
    mixed_rows = sum(len(v) for v in mixed)

    print(f"\n=== {label} ===")
    print(f"observations                     : {n}")
    print(f"unique papers (effective n)      : {len(by_paper)}")
    print(f"label balance                    : {dict(counts)}")
    print(f"papers reporting BOTH directions : {len(mixed)} "
          f"({100*len(mixed)/len(by_paper):.0f}%)")
    print(f"  observations inside them       : {mixed_rows} ({100*mixed_rows/n:.0f}%)")
    print(f"majority-class FLOOR             : {floor:.3f}")
    print(f"paper-level-feature CEILING      : {ceiling:.3f}")
    print(f"  -> total usable band           : {ceiling-floor:.3f} accuracy points")
    return {
        "label": label, "n_obs": n, "n_papers": len(by_paper),
        "counts": dict(counts), "mixed_papers": len(mixed),
        "mixed_obs": mixed_rows, "floor": round(floor, 4),
        "ceiling": round(ceiling, 4), "band": round(ceiling - floor, 4),
    }


def main():
    g = json.load(open(GRAPH))
    out = [
        report(observations(g, True), "contested edges only"),
        report(observations(g, False), "ALL edges"),
    ]

    c = out[0]
    print(f"""
INTERPRETATION

  {c['mixed_papers']} of {c['n_papers']} papers ({100*c['mixed_papers']/c['n_papers']:.0f}%) report
  both an enrichment and a depletion among their contested edges, covering
  {c['mixed_obs']} of {c['n_obs']} observations. Those rows carry the same paper embedding
  and opposite labels, so they are unseparable by construction.

  A paper-level probe can therefore only score between {c['floor']:.3f} and {c['ceiling']:.3f}.
  Reaching the top of that band would require an oracle that already knows each
  paper's majority direction -- which is not a finding, it is the label.

  This does NOT say embeddings are useless. It says the FEATURE MUST VARY WITH
  THE LABEL. The unit has to be the relation-bearing sentence for a specific
  (paper, taxon, disease), not the paper. That is Task 1 in
  NEXT_SESSION_PROMPT.md, and it is a prerequisite for Task 2, not an
  optimisation of it.

  Note also the effective sample size: {c['n_obs']} observations come from only
  {c['n_papers']} papers. Anything shuffled or bootstrapped must move whole papers.""")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
