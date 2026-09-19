#!/usr/bin/env python3
"""Is MAIN_DATA.json's cleaned text as good a mention substrate as the git text?

`silent_edge_mentions.py` now falls back to MAIN_DATA.json for papers absent from
all_usable_papers.json, which closed a quarter of its not-scoreable hole. That is
only worth having if a MAIN_DATA paper answers "is this taxon named here?" the
same way the git copy of the SAME paper does.

MAIN_DATA text went through a quantized Llama-3-8B cleaning pipeline. If that
pipeline dropped tables -- exactly where a `silent` taxon tends to live -- then
MAIN_DATA would report ABSENT for taxa the paper does name, manufacturing
fabrication candidates. This script measures that instead of assuming it.

Design: the two sources overlap on a set of graph papers by title. For every
observation on an overlapping paper, run the identical matcher against both
texts and cross-tabulate. Disagreement is directional and the two directions
mean different things:

  git=mentioned, md=absent  -> MAIN_DATA lost text. Its ABSENT verdicts are unsafe.
  git=absent, md=mentioned  -> MAIN_DATA has text the git copy lacks.

A clean diagonal licenses the fallback. Anything else bounds how far to trust it.
"""
import json
import os
import re
from collections import Counter

import witness_discordance as W
from verify_taxon_mentions import classify, norm_text, squash

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
MAIN_DATA = os.path.join(HERE, "..", "MAIN_DATA.json")
OUT = os.path.join(HERE, "validate_maindata_text.json")


def key(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def load():
    git, md = {}, {}
    for p in json.load(open(PAPERS)):
        k = key(p.get("title"))
        if k and len(p.get("text") or "") > 500:
            git[k] = p["text"]
    for rec in json.load(open(MAIN_DATA)).values():
        k = key(rec.get("name"))
        body = "\n".join(rec.get("chunks") or [])
        if k and len(body) > 500:
            md[k] = body
    return git, md


def mention(text_pair, node):
    text, text_sq = text_pair
    for f in [s for s in [node.get("label")] + list(node.get("aliases") or []) if s]:
        tier, _ = classify(f, text, text_sq)
        if tier in ("exact", "variant", "abbrev"):
            return True
    return False


def main():
    g = json.load(open(os.path.join(HERE, "graph.json")))
    papers, edges = g["papers"], g["edges"]
    nodes = {n["id"]: n for n in g["nodes"]}

    git, md = load()
    overlap = sorted(set(git) & set(md))
    print(f"papers in git: {len(git)}  in MAIN_DATA: {len(md)}  overlap: {len(overlap)}")

    # Restrict to overlapping papers that actually back an observation.
    obs = W.build()
    prep = {}
    for k in overlap:
        tg = norm_text(git[k])
        tm = norm_text(md[k])
        prep[k] = ((tg, squash(tg)), (tm, squash(tm)))

    tab = Counter()
    disagree = []
    len_ratio = []
    seen_papers = set()
    for o in obs:
        pk = key(papers[o["paper"]]["title"])
        if pk not in prep:
            continue
        e = edges[o["edge"]]
        node = nodes.get(e["source"])
        if node is None:
            continue
        gpair, mpair = prep[pk]
        a, b = mention(gpair, node), mention(mpair, node)
        tab[(a, b)] += 1
        seen_papers.add(pk)
        if a != b:
            disagree.append({
                "taxon": e["taxon"], "disease": e["disease"], "prov": o["prov"],
                "paper": papers[o["paper"]]["title"],
                "git_mentioned": a, "main_data_mentioned": b,
                "node_label": node.get("label"),
                "aliases": node.get("aliases") or [],
            })

    for k in sorted(seen_papers):
        len_ratio.append(len(md[k]) / max(1, len(git[k])))

    n = sum(tab.values())
    agree = tab[(True, True)] + tab[(False, False)]
    out = {
        "n_overlap_papers_with_observations": len(seen_papers),
        "n_observations_compared": n,
        "both_mentioned": tab[(True, True)],
        "both_absent": tab[(False, False)],
        "git_mentioned_maindata_absent": tab[(True, False)],
        "git_absent_maindata_mentioned": tab[(False, True)],
        "agreement": round(agree / n, 4) if n else None,
        "main_data_len_over_git_len": {
            "min": round(min(len_ratio), 3) if len_ratio else None,
            "median": round(sorted(len_ratio)[len(len_ratio) // 2], 3) if len_ratio else None,
            "max": round(max(len_ratio), 3) if len_ratio else None,
        },
        "disagreements": disagree,
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))
    if disagree:
        print(f"\n--- {len(disagree)} disagreements ---")
        for d in disagree[:40]:
            print(f"  git={d['git_mentioned']!s:5s} md={d['main_data_mentioned']!s:5s} "
                  f"[{d['prov']:10s}] {d['taxon'][:40]:40s} {d['paper'][:50]}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
