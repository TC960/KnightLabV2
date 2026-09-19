#!/usr/bin/env python3
"""Stress-test the one positive result in the contrast census.

`contrast_census.py` finds that papers whose MAIN reported contrast is NOT
disease-vs-healthy-control disagree with the rest of the literature far more than
papers that are in scope: O/E 1.75 vs 0.99, paper-level permutation p = 0.0001
against an MDE of +-0.30.

This project has shipped false positives that survived until tested, and it
retracted a quality tier three days ago because its two buckets were not
independent. So before that number is quoted anywhere it gets four attacks:

1. LEAVE-ONE-PAPER-OUT. 11 papers carry the effect. Drop each in turn.
2. DROP THE CONFIRMED MISCALL. The blinded reader labelled one paper ANIMAL from
   its title; it is a human MS-twin-vs-healthy-co-twin study with a mouse transfer
   experiment alongside, so it is in scope. It must not be load-bearing.
3. DROP THE MIXED PAPERS. Five out-of-gate papers ALSO report a healthy-control
   arm, so some of their edges are legitimate. The effect should survive keeping
   only the cleanly out-of-scope ones.
4. INDEPENDENCE. The predictor is read from the paper's own sentences by an
   adjudicator blinded to the disease label, the label's provenance and the graph;
   the outcome is computed from the graph. Stated, not tested -- but it is the
   exact property the retracted tier lacked.

Writes contrast_robustness.json.
"""
import json, random
from collections import defaultdict
from pathlib import Path

from paper_inversion import build_observations, load_graph, score
from paper_inversion_control import ancestor_sets, thin
from paper_discordance_offset import expectations, test_binary

HERE = Path(__file__).parent
N_PERM = 20000
SEED = 20260919
IN_GATE = {"HC"}

# the one verdict this session read and overturned
MISCALL = "Multiple sclerosis and gut microbiota: Lachnospiraceae from the ileum"
# out-of-gate papers that nonetheless report a healthy-control arm (read in-session)
MIXED = [
    "Comparison of the effects of probiotics, rifaximin, and lactulose",
    "The ketogenic diet influences taxonomic and functional composition",
    "Effect of Probiotics Supplementation on REM Sleep Behavior Disorder",
    "Integrated Traditional Chinese Medicine Improves Functional Outcome",
    "Gut microbiota composition during a 12-week intervention with delayed-release",
]

def build():
    graph = load_graph()
    thinned, _ = thin(build_observations(graph), ancestor_sets(graph))
    per_paper, _ = score(thinned)
    _, e_dis = expectations(thinned)
    papers = graph["papers"]
    rows, titles = [], []
    for p, v in sorted(per_paper.items()):
        if e_dis[p] <= 0:
            continue
        rows.append({"dis": v[1], "e_dis": e_dis[p]})
        titles.append(papers[p]["title"])
    return rows, titles

def label(titles, verdicts, drop=()):
    by = {v["title"]: v for v in verdicts if v.get("supported")}
    out = []
    for t in titles:
        v = by.get(t)
        if v is None or v.get("contrast_type") == "UNCLEAR":
            out.append(None); continue
        if any(d in t for d in drop):
            out.append(None); continue
        out.append(v.get("contrast_type") not in IN_GATE)
    return out

def run(rows, labels, seed=SEED):
    return test_binary(rows, labels, random.Random(seed), N_PERM)

def main():
    verdicts = json.load(open(HERE / "contrast_census.json"))["verdicts"]
    rows, titles = build()
    res = {}

    base = run(rows, label(titles, verdicts))
    res["baseline"] = base
    print(f"baseline           O/E {base['oe_true']:.3f} (n={base['n_true']}) vs "
          f"{base['oe_false']:.3f}  diff {base['diff']:+.3f}  p={base['p']:.5f}  "
          f"MDE +-{base['mde_abs_diff']:.3f}")

    # 1. leave one out
    oog = [t for t, l in zip(titles, label(titles, verdicts)) if l]
    loo = {}
    for t in oog:
        r = run(rows, label(titles, verdicts, drop=(t,)))
        loo[t[:60]] = {"p": r["p"], "diff": r["diff"], "n_true": r["n_true"],
                       "mde": r["mde_abs_diff"]}
    res["leave_one_out"] = loo
    worst = max(loo.values(), key=lambda d: d["p"])
    print(f"\nleave-one-out over {len(oog)} papers: worst p={worst['p']:.5f} "
          f"(diff {worst['diff']:+.3f})")
    for k, v in sorted(loo.items(), key=lambda kv: -kv[1]["p"])[:4]:
        print(f"   drop {k[:55]:55} p={v['p']:.5f} diff {v['diff']:+.3f}")

    # 2. drop the confirmed miscall
    r2 = run(rows, label(titles, verdicts, drop=(MISCALL,)))
    res["drop_confirmed_miscall"] = r2
    print(f"\ndrop miscall       O/E {r2['oe_true']:.3f} (n={r2['n_true']}) vs "
          f"{r2['oe_false']:.3f}  diff {r2['diff']:+.3f}  p={r2['p']:.5f}")

    # 3. keep only the cleanly out-of-scope papers
    r3 = run(rows, label(titles, verdicts, drop=tuple([MISCALL] + MIXED)))
    res["clean_out_of_gate_only"] = r3
    print(f"clean-only         O/E {r3['oe_true']:.3f} (n={r3['n_true']}) vs "
          f"{r3['oe_false']:.3f}  diff {r3['diff']:+.3f}  p={r3['p']:.5f}  "
          f"MDE +-{r3['mde_abs_diff']:.3f}")

    # 4. seed sensitivity
    seeds = {s: run(rows, label(titles, verdicts), seed=s)["p"]
             for s in (1, 7, 12345)}
    res["seed_sensitivity"] = seeds
    print(f"\nseed sensitivity   {seeds}")

    json.dump(res, open(HERE / "contrast_robustness.json", "w"), indent=1)
    print("\nwrote contrast_robustness.json")

if __name__ == "__main__":
    main()
