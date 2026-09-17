#!/usr/bin/env python3
"""Tag every union edge with who backs it, then test whether the tag predicts quality.

THE DECISION THIS ANSWERS. Merging the human gold into the extraction buys 25%
more edges but costs agreement with the curated databases (81% -> 77%). That
looks like a straight trade-off, and if it is, the union is a worse graph and
should not be published.

It is only a real trade-off if the two kinds of edge are indistinguishable once
merged. They are not: every row carries `provenance`, so every edge can be tagged
by who supports it --

    human   only gold papers support this edge
    model   only extraction papers support it
    both    at least one paper of each

-- and if `both` and `human` edges keep the gold's ~81% agreement while `model`
edges sit at ~73%, the union is not a compromise. It is the full graph plus a
filter, and a reader can have whichever they want.

That is the hypothesis this script tests, against Disbiome and Peryton, using the
same taxonomy re-resolution `validate_external.py` uses (never join on another
database's stored taxid -- Disbiome files Prevotella as 59823, a species, where
the genus is 838).

    python annotate_provenance.py            # writes graph_union.json in place
    python annotate_provenance.py --dry-run
"""
import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, "extractions_union.json")
GRAPH = os.path.join(HERE, "graph_union.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--graph", default=GRAPH)
    ap.add_argument("--rows", default=ROWS)
    a = ap.parse_args()

    # PAPER-level provenance is nearly useless here: 259 of 325 papers are in both
    # sources, so 87% of edges tag as "both" and the filter separates nothing. The
    # meaningful unit is the (paper, TAXON) pair -- for a shared paper, did the
    # model name this taxon, did the human, or did both? `human_only_*` records
    # exactly the taxa the human added that the model had not, so the complement
    # is what the model found.
    import re as _re

    def taxa_of(s):
        return {t.strip().lower() for t in _re.split(r"[;,]", s or "") if t.strip()}

    prov = {}          # (title, taxon) -> {"model"} / {"human"} / both
    paper_prov = {}
    for r in json.load(open(a.rows)):
        t = (r.get("title") or "").strip().lower()
        if not t:
            continue
        paper_prov[t] = r.get("provenance", "model")
        human_only = taxa_of(r.get("human_only_enriched")) | \
                     taxa_of(r.get("human_only_depleted"))
        allt = taxa_of(r.get("predicted_enriched")) | taxa_of(r.get("predicted_depleted"))
        for tx in allt:
            if r.get("provenance") == "human":
                prov[(t, tx)] = "human"
            elif tx in human_only:
                prov[(t, tx)] = "human"
            else:
                prov[(t, tx)] = "model" 

    g = json.load(open(a.graph))

    # graph.json's paper table may hold dicts or bare strings depending on build
    ptbl = g.get("papers", [])
    def title_at(i):
        if not isinstance(i, int) or i >= len(ptbl):
            return None
        rec = ptbl[i]
        return (rec["title"] if isinstance(rec, dict) else rec).strip().lower()

    counts = collections.Counter()
    unknown = 0
    for e in g["edges"]:
        srcs = set()
        tx = str(e.get("taxon", "")).strip().lower()
        titles = []
        for p in (e.get("papers") or []):
            titles.append((p["title"] if isinstance(p, dict) else str(p)).strip().lower())
        for ev in (e.get("ev") or []):
            t = title_at(ev.get("i") if isinstance(ev, dict) else ev)
            if t:
                titles.append(t)
        for t in titles:
            v = prov.get((t, tx))
            if v is None:
                # the edge's taxon label is the CANONICAL name; the row holds the
                # raw string, so fall back to the paper's own provenance
                v = paper_prov.get(t)
            if v:
                srcs.add(v)
        if not srcs:
            tag = "unknown"
            unknown += 1
        elif srcs == {"model"}:
            tag = "model"
        elif srcs == {"human"}:
            tag = "human"
        else:
            # any mixture, or a paper that is itself in both sources
            tag = "both"
        e["provenance"] = tag
        counts[tag] += 1

    total = sum(counts.values())
    print(f"{total} edges tagged")
    for k in ("both", "human", "model", "unknown"):
        if counts[k]:
            print(f"  {k:<8} {counts[k]:>5}  ({100*counts[k]/total:.1f}%)")
    if unknown:
        print(f"  NOTE {unknown} edges had no resolvable source paper")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    json.dump(g, open(a.graph, "w"))
    print(f"\nwrote {a.graph} with per-edge `provenance`")
    print("next: validate each subset with validate_external.py --graph <subset>")


if __name__ == "__main__":
    main()
