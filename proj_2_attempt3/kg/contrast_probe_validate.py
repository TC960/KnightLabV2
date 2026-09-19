#!/usr/bin/env python3
"""Do the deterministic edge probe and the blinded human-read verdicts agree?

They are independent by construction: `contrast_census.py`'s verdicts come from a
reader judging the study's ARMS from title + sentences, blinded to the graph;
`contrast_edge_probe.py` is a regex over the sentences that name a particular
TAXON, with no reader and no notion of a study design. If the regex flag is
measuring what the reader measured, CANDIDATE observations should concentrate in
the papers the reader called out of gate.

Unit of randomisation is the PAPER (the predictor is a paper property here), and
the statistic is the pooled CANDIDATE rate over scoreable observations.

Writes contrast_probe_validate.json.
"""
import json, random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
N_PERM = 20000
SEED = 20260919
SCOREABLE = {"CLEAN", "CANDIDATE", "UNRESOLVED"}

def pooled(items, keep):
    n = c = 0
    for cand, tot, k in items:
        if k in keep:
            c += cand; n += tot
    return (c / n if n else float("nan")), c, n

def test(items, a, b, rng, n_perm=N_PERM):
    sub = [(c, t, k) for c, t, k in items if k in (a, b)]
    ra, ca, na = pooled(sub, {a})
    rb, cb, nb = pooled(sub, {b})
    obs = ra - rb
    keys = [k for _, _, k in sub]
    null = []
    for _ in range(n_perm):
        rng.shuffle(keys)
        perm = [(c, t, k) for (c, t, _), k in zip(sub, keys)]
        x, _, _ = pooled(perm, {a}); y, _, _ = pooled(perm, {b})
        null.append(x - y)
    p = (sum(1 for d in null if abs(d) >= abs(obs)) + 1) / (n_perm + 1)
    null.sort()
    mde = max(abs(null[int(0.025 * n_perm)]), abs(null[int(0.975 * n_perm)]))
    return {"group_a": a, "group_b": b,
            "rate_a": round(ra, 4), "cand_a": ca, "n_a": na, "papers_a": keys.count(a),
            "rate_b": round(rb, 4), "cand_b": cb, "n_b": nb, "papers_b": keys.count(b),
            "diff": round(obs, 4), "p": round(p, 5), "mde_abs_diff": round(mde, 4)}

def main():
    rows = json.load(open(HERE / "contrast_edge_probe.json"))["rows"]
    per = defaultdict(lambda: [0, 0, None])
    for r in rows:
        if r["verdict"] not in SCOREABLE:
            continue
        p = per[r["paper"]]
        p[1] += 1
        p[0] += (r["verdict"] == "CANDIDATE")
        if p[2] is None:
            p[2] = ("out_of_gate" if r.get("paper_contrast") not in (None, "HC", "UNCLEAR")
                    else "in_gate_mixed" if r.get("paper_also_subgroup")
                    else "in_gate_clean")
    items = [(c, t, k) for c, t, k in per.values() if k]
    rng = random.Random(SEED)
    res = {
        "n_papers": len(items),
        "out_of_gate_vs_in_gate_clean": test(items, "out_of_gate", "in_gate_clean", rng),
        "in_gate_mixed_vs_in_gate_clean": test(items, "in_gate_mixed", "in_gate_clean", rng),
    }
    # BH over the two
    live = sorted(((k, v["p"]) for k, v in res.items() if isinstance(v, dict)),
                  key=lambda kv: kv[1])
    m = len(live); prev = 1.0
    for i in range(m - 1, -1, -1):
        k, pv = live[i]
        prev = min(prev, pv * m / (i + 1))
        res[k]["q_bh"] = round(min(1.0, prev), 5)
    json.dump(res, open(HERE / "contrast_probe_validate.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
