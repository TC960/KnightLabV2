#!/usr/bin/env python3
"""Do the graph's four full-text sources answer "is this taxon named here?" alike?

Generalises validate_maindata_text.py to every source `silent_edge_mentions.py`
can draw on, because as of 2026-09-17 there are four and they do NOT all mean
the same thing:

  all_usable  all_usable_papers.json    tracked; the datasheet scrape.
  extract_in  extract_input.json        UNTRACKED (copyright, see below).
  new_papers  new_papers.json           UNTRACKED (same).
  main_data   MAIN_DATA.json            untracked, but its .zip IS tracked.

`extract_input.json` / `new_papers.json` are special: they are the text the
extractor was actually GIVEN. For a fabrication question that is the correct
substrate -- a taxon absent from the text the model read is a true fabrication,
whereas a taxon absent from some other rendering of the same paper may just mean
that rendering is lossy. So this script asks whether source choice can change a
verdict before any priority order is chosen for it.

COPYRIGHT, and it is not optional. `extract_input.json`, `new_papers.json`,
`extract_input_gold.json` and `gold_missing_papers.json` were untracked in
254b0a8 because THIS REPOSITORY IS PUBLIC and the papers we hold include
non-open-access articles. They are in .gitignore. Read them locally, never
`git add` them, and never copy their text into a file that is tracked (including
a findings doc). Restore them for local work with:

    git show 254b0a8^:proj_2_attempt3/kg/extract_input.json > extract_input.json
    git show 254b0a8^:proj_2_attempt3/kg/new_papers.json    > new_papers.json

Everything here degrades to whatever subset is present on disk.
"""
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import witness_discordance as W
from verify_taxon_mentions import classify, norm_text, squash

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validate_text_sources.json")


def key(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def load_sources():
    """name -> {paper_key: raw_text}. Missing files are skipped silently."""
    S = defaultdict(dict)
    p = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
    if os.path.exists(p):
        for r in json.load(open(p)):
            if len(r.get("text") or "") > 500:
                S["all_usable"][key(r.get("title"))] = r["text"]
    for name, fn in (("extract_in", "extract_input.json"),
                     ("new_papers", "new_papers.json")):
        fp = os.path.join(HERE, fn)
        if os.path.exists(fp):
            for r in json.load(open(fp)):
                if len(r.get("text") or "") > 500:
                    S[name][key(r.get("title"))] = r["text"]
    md = os.path.join(HERE, "..", "MAIN_DATA.json")
    if os.path.exists(md):
        for r in json.load(open(md)).values():
            body = "\n".join(r.get("chunks") or [])
            if len(body) > 500:
                S["main_data"][key(r.get("name"))] = body
    return S


def mention(pair, node):
    text, text_sq = pair
    for f in [s for s in [node.get("label")] + list(node.get("aliases") or []) if s]:
        tier, _ = classify(f, text, text_sq)
        if tier in ("exact", "variant", "abbrev"):
            return True
    return False


def main():
    g = json.load(open(os.path.join(HERE, "graph.json")))
    papers, edges = g["papers"], g["edges"]
    nodes = {n["id"]: n for n in g["nodes"]}
    graph_keys = {key(p["title"]) for p in papers}

    S = load_sources()
    print("sources on disk:")
    for n, d in S.items():
        print(f"  {n:12s} {len(d):5d} papers, {len(set(d) & graph_keys):3d} of "
              f"{len(graph_keys)} contributing papers")
    covered = set().union(*[set(d) for d in S.values()]) if S else set()
    print(f"\ncontributing papers with text: {len(graph_keys & covered)}/{len(graph_keys)}"
          f"   still missing: {len(graph_keys - covered)}")

    obs = W.build()
    prep = {n: {} for n in S}
    results = {}
    for name, d in S.items():
        for k, raw in d.items():
            if k in graph_keys:
                t = norm_text(raw)
                prep[name][k] = (t, squash(t))

    pairs = {}
    for a, b in combinations(sorted(S), 2):
        tab = Counter()
        disagree = []
        for o in obs:
            pk = key(papers[o["paper"]]["title"])
            if pk not in prep[a] or pk not in prep[b]:
                continue
            e = edges[o["edge"]]
            node = nodes.get(e["source"])
            if node is None:
                continue
            x, y = mention(prep[a][pk], node), mention(prep[b][pk], node)
            tab[(x, y)] += 1
            if x != y:
                disagree.append({
                    "taxon": e["taxon"], "prov": o["prov"],
                    "paper": papers[o["paper"]]["title"],
                    a: x, b: y,
                    "len_%s" % a: len(S[a][pk]), "len_%s" % b: len(S[b][pk]),
                })
        n = sum(tab.values())
        if not n:
            continue
        pairs[f"{a} vs {b}"] = {
            "n_observations_both_have": n,
            "agree": tab[(True, True)] + tab[(False, False)],
            f"{a}_only_mentions": tab[(True, False)],
            f"{b}_only_mentions": tab[(False, True)],
            "agreement": round((tab[(True, True)] + tab[(False, False)]) / n, 4),
            "disagreements": disagree,
        }

    out = {
        "sources_present": {n: len(d) for n, d in S.items()},
        "contributing_papers": len(graph_keys),
        "contributing_with_text": len(graph_keys & covered),
        "contributing_missing": sorted(graph_keys - covered),
        "pairwise": pairs,
    }
    json.dump(out, open(OUT, "w"), indent=1)

    print("\npairwise agreement (only papers both sources hold):")
    for k, v in pairs.items():
        print(f"  {k:26s} n={v['n_observations_both_have']:5d} "
              f"agree={v['agreement']:.4f}  disagreements={len(v['disagreements'])}")
        for d in v["disagreements"][:6]:
            lens = {kk: vv for kk, vv in d.items() if kk.startswith("len_")}
            print(f"      {d['taxon'][:34]:34s} {lens}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
