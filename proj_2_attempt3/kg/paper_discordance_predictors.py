#!/usr/bin/env python3
"""What makes a paper discordant with the rest of the literature?

`paper_inversion.py` established that consensus-disagreement is a PAPER-level
property here: no single paper is inverted, but which papers hold the minority
direction is clustered far beyond the within-edge null (p = 0.0003 full,
0.0013 containment-controlled, 0.0003 on >=3-paper edges).

That reframes a question this project has failed to answer four times. Study
design, body site, ASD, and taxon co-occurrence were all tested EDGE-level --
"do the up-papers of a contested edge differ from its down-papers" -- with 130
contested edges averaging ~5 papers, and all four returned null. The unit here is
the paper, and every paper that reports anything decisive contributes: 112
testable papers, 89 of them carrying extracted study metadata. Same question,
better-conditioned denominator.

OUTCOME per paper: n_disagree / n_decisive against the leave-one-out majority, on
the containment-controlled set (within-paper taxonomic relatives already dropped,
so a paper is not credited twice for one measurement).

STATISTIC: the difference in POOLED disagreement rate between groups
(sum disagree / sum decisive, not the mean of per-paper rates, so a paper with 4
observations does not weigh as much as one with 23).

NULL: shuffle the predictor labels across papers, holding each paper's
(n_decisive, n_disagree) fixed. The unit of randomisation is the paper, which is
the standing requirement in this repo -- pair-level shuffling has produced three
false positives on record. Continuous variables are split at the median and tested
the same way, so there is one statistic and one null throughout.

BH across all predictors. Every null is reported with its minimum detectable
difference, obtained from the same permutation ensemble.

Writes paper_discordance_predictors.json.
"""

import json
import random
import sys
from collections import Counter, defaultdict

from paper_inversion import MIN_DECISIVE, build_observations, load_graph, score
from paper_inversion_control import ancestor_sets, thin

N_PERM = 20000
SEED = 20260910


def paper_disease(graph, edges):
    """Modal disease of each paper's observations."""
    seen = defaultdict(Counter)
    for ed in edges:
        for pi, _ in ed["obs"]:
            seen[pi][ed["disease"]] += 1
    return {p: c.most_common(1)[0][0] for p, c in seen.items()}


def pooled(rows, mask):
    dec = sum(r["dec"] for r, m in zip(rows, mask) if m)
    dis = sum(r["dis"] for r, m in zip(rows, mask) if m)
    return (dis / dec if dec else float("nan")), dec, dis


def test_binary(rows, labels, rng, n_perm=N_PERM):
    """labels: list of bool/None aligned with rows. None = excluded."""
    idx = [i for i, l in enumerate(labels) if l is not None]
    if len(idx) < 8:
        return None
    sub = [rows[i] for i in idx]
    lab = [bool(labels[i]) for i in idx]
    if len(set(lab)) < 2:
        return None
    a, dec_a, dis_a = pooled(sub, lab)
    b, dec_b, dis_b = pooled(sub, [not x for x in lab])
    obs = a - b
    null = []
    perm = list(lab)
    for _ in range(n_perm):
        rng.shuffle(perm)
        x, _, _ = pooled(sub, perm)
        y, _, _ = pooled(sub, [not v for v in perm])
        null.append(x - y)
    p = (sum(1 for d in null if abs(d) >= abs(obs)) + 1) / (n_perm + 1)
    null.sort()
    # two-sided MDE: the smallest |difference| that would have reached p<0.05
    mde = max(abs(null[int(0.025 * n_perm)]), abs(null[int(0.975 * n_perm)]))
    return {
        "n_true": sum(lab), "n_false": len(lab) - sum(lab),
        "rate_true": round(a, 4), "rate_false": round(b, 4),
        "dec_true": dec_a, "dec_false": dec_b,
        "diff": round(obs, 4), "p": round(p, 5),
        "mde_abs_diff": round(mde, 4),
    }


def main():
    graph = load_graph()
    edges = build_observations(graph)
    rel = ancestor_sets(graph)
    thinned, _ = thin(edges, rel)
    per_paper, _ = score(thinned)
    dis_of = paper_disease(graph, thinned)

    papers = graph["papers"]
    rows = []
    for p, v in sorted(per_paper.items()):
        if v[0] < MIN_DECISIVE:
            continue
        m = papers[p]
        rows.append({"paper": p, "dec": v[0], "dis": v[1],
                     "title": m["title"][:60], "meta": m.get("has_meta", False),
                     "disease": dis_of.get(p, "?"), "m": m})
    base = sum(r["dis"] for r in rows) / sum(r["dec"] for r in rows)
    print(f"{len(rows)} testable papers, "
          f"{sum(1 for r in rows if r['meta'])} with metadata; "
          f"pooled disagreement {base:.1%}")

    rng = random.Random(SEED)

    def meta(r, f):
        return r["m"].get(f) if r["meta"] else None

    def med_split(field):
        vals = [meta(r, field) for r in rows]
        nums = sorted(v for v in vals if isinstance(v, (int, float)) and v)
        if len(nums) < 12:
            return [None] * len(rows)
        cut = nums[len(nums) // 2]
        return [None if not isinstance(v, (int, float)) or not v else v >= cut
                for v in vals]

    def cohort(r):
        if not r["meta"]:
            return None
        a, b = r["m"].get("n_cases"), r["m"].get("n_controls")
        return (a or 0) + (b or 0) or None

    coh = [cohort(r) for r in rows]
    coh_nums = sorted(v for v in coh if v)
    coh_cut = coh_nums[len(coh_nums) // 2] if coh_nums else 0

    tests = {
        "sequencing_16S_vs_shotgun":
            [None if not r["meta"] or not meta(r, "seq") else meta(r, "seq") == "16S"
             for r in rows],
        "medication_controlled": [meta(r, "med") for r in rows],
        "diet_controlled": [meta(r, "diet") for r in rows],
        "country_China": [None if not r["meta"] or not meta(r, "country")
                          else meta(r, "country") == "China" for r in rows],
        "n_cases_above_median": med_split("n_cases"),
        "n_controls_above_median": med_split("n_controls"),
        "cohort_total_above_median":
            [None if not v else v >= coh_cut for v in coh],
        "region_V3V4": [None if not r["meta"] or not meta(r, "region")
                        else meta(r, "region") == "V3-V4" for r in rows],
        # structural controls, not study design -- included so a positive on a
        # metadata field can be read against them
        "reports_a_lot_above_median":
            [r["dec"] >= sorted(x["dec"] for x in rows)[len(rows) // 2]
             for r in rows],
        "disease_is_Parkinsons":
            [r["disease"].lower().startswith("parkinson") for r in rows],
    }

    results = {}
    for name, labels in tests.items():
        res = test_binary(rows, labels, rng)
        results[name] = res
        if res is None:
            print(f"  {name:32s} skipped (too few)")
            continue
        print(f"  {name:32s} {res['rate_true']:.3f} (n={res['n_true']:>2}) vs "
              f"{res['rate_false']:.3f} (n={res['n_false']:>2})  "
              f"diff {res['diff']:+.3f}  p={res['p']:.4f}  "
              f"MDE +-{res['mde_abs_diff']:.3f}")

    # BH
    live = [(k, v["p"]) for k, v in results.items() if v]
    live.sort(key=lambda kv: kv[1])
    m = len(live)
    q = {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        k, p = live[i]
        prev = min(prev, p * m / (i + 1))
        q[k] = round(min(1.0, prev), 4)
    for k in q:
        results[k]["q"] = q[k]
    print(f"\nBH across {m} predictors:")
    for k, p in live:
        flag = "  <-- survives" if q[k] < 0.05 else ""
        print(f"  {k:32s} p={p:.4f}  q={q[k]:.4f}{flag}")

    survivors = [k for k in q if q[k] < 0.05]
    print(f"\nsurvivors: {survivors if survivors else 'NONE'}")

    out = {"n_testable": len(rows), "n_with_metadata": sum(1 for r in rows if r["meta"]),
           "pooled_base_rate": round(base, 4), "n_perm": N_PERM, "seed": SEED,
           "tests": results, "survivors": survivors,
           "papers": [{k: r[k] for k in ("paper", "dec", "dis", "title",
                                         "disease", "meta")} for r in rows]}
    with open("paper_discordance_predictors.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote paper_discordance_predictors.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
