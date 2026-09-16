#!/usr/bin/env python3
"""Does an observation's TEXTUAL PROVENANCE predict whether it disagrees with the
rest of the literature?

FINDINGS_paper_discordance.md established that 27.6% of decisive observations
disagree with their edge's leave-one-out majority, that which papers hold the
minority direction is clustered well beyond a within-edge null (p=0.0003), and
that **24 paper-level variables in two passes returned 24 nulls** -- study design,
country, cohort size, platform, region, medication and diet control, then 15
wet-lab/bioinformatics variables. Nothing extracted explains the offset.

All 24 of those describe how the study was RUN. This tests something different:
what the paper's own text says about the taxon. `audit_direction_witness.py`
classifies every sentence naming a taxon as the paper's OWN RESULT (it carries a
statistical or measurement cue and no citation) or as BACKGROUND (it cites
another study, or is mechanism/speculation). Each observation then falls in one
of three provenance classes:

    own         at least one own-result sentence names this taxon
    background  sentences name the taxon, but every one of them is background
    silent      no kept sentence names the taxon at all

The hypothesis is that `background` observations are weaker evidence -- an
extractor that read "Chang et al. indicated an increased abundance of Blautia"
as this paper's finding would produce an edge with no cohort behind it -- and
should therefore disagree with the rest of the literature more often.

CAVEAT, stated up front because it cuts both ways: `relation_sentences.json`
keeps 7.5k of 106k sentences, and a taxon whose only appearance is a row in a
table or a bar in a figure is `silent` however well the paper supports it. So
`silent` mixes "unsupported" with "supported outside prose", and a null there is
uninformative. The `own` vs `background` contrast is the meaningful one, since
both classes required the taxon to be named in a cue-bearing sentence.

Discordance follows FINDINGS_paper_discordance.md exactly: an observation is
decisive if its edge has >= 2 papers with a non-tied leave-one-out majority, and
discordant if its direction differs from that majority.

Observations are NOT independent -- one paper contributes many -- so the null
permutes the provenance label at the PAPER level, holding each paper's whole
block of observations together. Pair-level shuffling has manufactured three false
positives in this project already.

Writes witness_discordance.json.
"""
import json
import os
import random
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NPERM = 10000
SEED = 20260912

import audit_direction_witness as A  # noqa: E402  (provenance + matching logic)


def build():
    g = json.load(open(os.path.join(HERE, "graph.json")))
    rs_path = os.path.join(HERE, "relation_sentences_clean.json")
    if not os.path.exists(rs_path):
        rs_path = os.path.join(HERE, "relation_sentences.json")
    rs = json.load(open(rs_path))["papers"]
    by_norm = {A.norm_title(k): v for k, v in rs.items()}
    papers, nodes = g["papers"], {n["id"]: n for n in g["nodes"]}

    obs = []
    for ei, e in enumerate(g["edges"]):
        node = nodes.get(e["source"])
        if node is None:
            continue
        taxid, forms = A.taxon_matchers(node)
        for ev in e.get("ev", []):
            rec = by_norm.get(A.norm_title(papers[ev["i"]]["title"]))
            klass = "silent"
            if rec is not None:
                ws = A.witnesses(rec, taxid, forms)
                if ws:
                    klass = ("own" if any(A.provenance(s["s"]) == "own" for s in ws)
                             else "background")
            obs.append({
                "edge": ei, "paper": ev["i"],
                "dir": "up" if ev["d"] == "e" else "down",
                "prov": klass,
                "taxon": e["taxon"], "disease": e["disease"],
            })
    return obs


def decisive(obs):
    """Keep observations whose edge has a non-tied leave-one-out majority."""
    by_edge = defaultdict(list)
    for o in obs:
        by_edge[o["edge"]].append(o)
    out = []
    for rows in by_edge.values():
        if len(rows) < 2:
            continue
        for r in rows:
            others = [x["dir"] for x in rows if x is not r]
            up = others.count("up")
            dn = others.count("down")
            if up == dn:
                continue
            maj = "up" if up > dn else "down"
            r = dict(r, discordant=int(r["dir"] != maj))
            out.append(r)
    return out


def rate(rows, klass):
    sel = [r for r in rows if r["prov"] == klass]
    if not sel:
        return None
    k = sum(r["discordant"] for r in sel)
    return k, len(sel), k / len(sel)


def main():
    obs = build()
    dec = decisive(obs)

    counts = {k: sum(1 for o in obs if o["prov"] == k)
              for k in ("own", "background", "silent")}
    res = {"n_observations": len(obs), "provenance_counts": counts,
           "n_decisive": len(dec),
           "overall_discordance": round(
               sum(r["discordant"] for r in dec) / len(dec), 4) if dec else None}

    for k in ("own", "background", "silent"):
        r = rate(dec, k)
        res[f"discordance_{k}"] = [r[0], r[1], round(r[2], 4)] if r else None

    # ---- own vs background --------------------------------------------------
    # The pooled difference is confounded: a paper that is both background-heavy
    # and discordance-heavy would produce one without any per-observation effect.
    # So the PRIMARY test is paired WITHIN paper -- for each paper carrying both
    # kinds of observation, the difference of its own two rates. Paper identity
    # cancels exactly, and the null is a sign flip of each paper's difference,
    # which is a paper-level randomisation by construction.
    import math
    sub = [r for r in dec if r["prov"] in ("own", "background")]
    ro, rb = rate(sub, "own"), rate(sub, "background")
    by_paper = defaultdict(list)
    for r in sub:
        by_paper[r["paper"]].append(r)

    diffs = []
    for pid, rows in by_paper.items():
        own = [r for r in rows if r["prov"] == "own"]
        bg = [r for r in rows if r["prov"] == "background"]
        if not own or not bg:
            continue
        diffs.append(sum(r["discordant"] for r in bg) / len(bg)
                     - sum(r["discordant"] for r in own) / len(own))

    res["own_vs_background"] = {
        "pooled_own": [ro[0], ro[1], round(ro[2], 4)] if ro else None,
        "pooled_background": [rb[0], rb[1], round(rb[2], 4)] if rb else None,
        "pooled_difference": round(rb[2] - ro[2], 4) if (ro and rb) else None,
        "n_papers_with_both": len(diffs),
    }
    if ro and rb:
        pbar = (ro[0] + rb[0]) / (ro[1] + rb[1])
        se = math.sqrt(pbar * (1 - pbar) * (1 / ro[1] + 1 / rb[1]))
        res["own_vs_background"]["mde_80pct_pooled"] = round(2.8 * se, 4)

    if diffs:
        obs_mean = sum(diffs) / len(diffs)
        rng = random.Random(SEED)
        ge = 0
        for _ in range(NPERM):
            m = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
            if abs(m) >= abs(obs_mean):
                ge += 1
        sd = (sum((d - obs_mean) ** 2 for d in diffs) / max(1, len(diffs) - 1)) ** 0.5
        res["own_vs_background"].update({
            "paired_mean_difference": round(obs_mean, 4),
            "paired_sd": round(sd, 4),
            "p_sign_flip": round((ge + 1) / (NPERM + 1), 5),
            "mde_80pct_paired": round(2.8 * sd / math.sqrt(len(diffs)), 4),
        })

    json.dump(res, open(os.path.join(HERE, "witness_discordance.json"), "w"), indent=1)

    print(f"observations           : {res['n_observations']}  {counts}")
    print(f"decisive               : {res['n_decisive']}  "
          f"overall discordance {res['overall_discordance']}")
    for k in ("own", "background", "silent"):
        print(f"  {k:11s} discordance: {res[f'discordance_{k}']}")
    ob = res.get("own_vs_background")
    if ob:
        print()
        print(f"pooled  background - own = {ob['pooled_difference']:+.4f}  "
              f"(MDE 80% = {ob.get('mde_80pct_pooled')})")
        if "paired_mean_difference" in ob:
            print(f"paired (within paper)    = {ob['paired_mean_difference']:+.4f}  "
                  f"p = {ob['p_sign_flip']:.4f}  "
                  f"({ob['n_papers_with_both']} papers carrying both kinds)")
            print(f"minimum detectable paired difference at 80% power: "
                  f"{ob['mde_80pct_paired']:.3f}")
    print("wrote witness_discordance.json")


if __name__ == "__main__":
    main()
