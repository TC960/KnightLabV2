#!/usr/bin/env python3
"""Are the `silent` observations unsupported, or merely invisible to the prose filter?

FINDINGS_direction_audit.md sized 580 edges (28.9% of the graph) that rest on a
single paper AND carry no own-result prose witness, and recorded that "prose
filtering cannot reach them". `silent` there means no sentence passed the
relation filter (taxon AND direction cue in the same sentence) -- NOT that the
taxon is absent from the paper. A taxon reported only as a row in a table is
silent to that instrument however well the paper supports it.

This script asks the weaker, answerable question those 580 edges deserve: is the
taxon named ANYWHERE in the paper's full text? That separates two things the
`silent` label conflates:

  silent + mentioned  -> reported outside prose (table/figure). Not a fabrication.
                         Direction still unverified by any instrument here.
  silent + absent     -> the taxon does not occur in the only paper backing it.
                         A candidate fabrication, and individually reviewable.

Reuses the repo's own provenance logic (`witness_discordance.build`) and its own
surface forms (`audit_direction_witness.taxon_matchers`) so the classification
matches the published one exactly rather than a re-derivation of it.

Full text is in git for the datasheet papers only; observations whose paper is
reachable only through the gitignored MAIN_DATA.json are reported as
not-scoreable, never as absent.
"""
import json
import os
import re
from collections import Counter, defaultdict

import audit_direction_witness as A
import witness_discordance as W
from verify_taxon_mentions import classify, norm_text, squash

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
OUT = os.path.join(HERE, "silent_edge_mentions.json")


def key(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def main():
    g = json.load(open(os.path.join(HERE, "graph.json")))
    papers = g["papers"]
    nodes = {n["id"]: n for n in g["nodes"]}
    edges = g["edges"]

    fulltext = {}
    for p in json.load(open(PAPERS)):
        k = key(p.get("title"))
        if k and len(p.get("text") or "") > 500:
            t = norm_text(p["text"])
            fulltext[k] = (t, squash(t))

    obs = W.build()  # the published per-observation provenance classification

    # Cache mention lookups: (paper_key, node_id) -> bool | None(not scoreable)
    cache = {}

    def mentioned(paper_key, node):
        ck = (paper_key, node["id"])
        if ck in cache:
            return cache[ck]
        pair = fulltext.get(paper_key)
        if pair is None:
            cache[ck] = None
            return None
        text, text_sq = pair
        # RAW label/aliases, not A.taxon_matchers(): that returns norm_surface'd
        # (whitespace-stripped) forms, which cannot match unstripped text.
        forms = [s for s in [node.get("label")] + list(node.get("aliases") or []) if s]
        hit = False
        for f in forms:
            tier, _ = classify(f, text, text_sq)
            if tier in ("exact", "variant", "abbrev"):
                hit = True
                break
        cache[ck] = hit
        return hit

    # cross-tab provenance x mention
    tab = defaultdict(Counter)
    absent_rows = []
    n_unscoreable = 0

    # edge-level: how many papers back the edge, for the "single-paper" cut
    for o in obs:
        e = edges[o["edge"]]
        node = nodes.get(e["source"])
        if node is None:
            continue
        pk = key(papers[o["paper"]]["title"])
        m = mentioned(pk, node)
        if m is None:
            n_unscoreable += 1
            tab[o["prov"]]["not_scoreable"] += 1
            continue
        tab[o["prov"]]["mentioned" if m else "absent"] += 1
        if not m:
            absent_rows.append({
                "taxon": e["taxon"], "disease": e["disease"],
                "direction": o["dir"], "prov": o["prov"],
                "paper": papers[o["paper"]]["title"],
                "n_papers_on_edge": e.get("n_papers"),
                "node_label": node.get("label"),
                "aliases": node.get("aliases") or [],
                "resolved": node.get("resolved"),
                "placeholder": node.get("placeholder"),
            })

    # The specific population FINDINGS_direction_audit.md called unreachable:
    # single-paper edges with no own-result witness.
    own_by_edge = defaultdict(set)
    for o in obs:
        own_by_edge[o["edge"]].add(o["prov"])

    unreachable = Counter()
    for ei, e in enumerate(edges):
        if e.get("n_papers") != 1:
            continue
        provs = own_by_edge.get(ei, set())
        if "own" in provs:
            continue
        node = nodes.get(e["source"])
        if node is None:
            continue
        rows = [o for o in obs if o["edge"] == ei]
        if not rows:
            continue
        pk = key(papers[rows[0]["paper"]]["title"])
        m = mentioned(pk, node)
        unreachable["total"] += 1
        if m is None:
            unreachable["not_scoreable"] += 1
        elif m:
            unreachable["mentioned"] += 1
        else:
            unreachable["absent"] += 1

    def rate(c):
        d = c["mentioned"] + c["absent"]
        return round(c["mentioned"] / d, 4) if d else None

    out = {
        "n_observations": len(obs),
        "n_not_scoreable_no_fulltext_in_git": n_unscoreable,
        "by_provenance": {k: dict(v) for k, v in tab.items()},
        "mention_rate_by_provenance": {k: rate(v) for k, v in tab.items()},
        "single_paper_no_own_witness_edges": dict(unreachable),
        "single_paper_no_own_witness_mention_rate": rate(unreachable),
        "absent_observations": sorted(
            absent_rows, key=lambda r: (r["prov"], r["taxon"]))[:400],
        "n_absent_total": len(absent_rows),
    }
    json.dump(out, open(OUT, "w"), indent=1)

    print(json.dumps({k: v for k, v in out.items()
                      if k != "absent_observations"}, indent=1))
    print(f"\nwrote {OUT}")
    if absent_rows:
        print(f"\n--- {len(absent_rows)} absent observations (first 30) ---")
        for r in absent_rows[:30]:
            print(f"  [{r['prov']:10s}] {r['taxon'][:38]:38s} {r['disease'][:22]:22s} "
                  f"npap={r['n_papers_on_edge']}")


if __name__ == "__main__":
    main()
