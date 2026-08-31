#!/usr/bin/env python3
"""Dump the Disbiome + Peryton records behind the 11 doubly-contradicted pairs.

Reuses validate_external.py's loaders and taxonomy.py's resolver so the join is
identical to the one that flagged the disagreement. Read-only.

    python adjudicate_dbrecords.py
"""
import json
import os
from collections import defaultdict

from validate_external import load_disbiome, load_peryton, DISEASE_MAP, OUTCOME
from taxonomy import Taxonomy

HERE = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    ("Erysipelotrichaceae", "Parkinson's disease"),
    ("Dorea", "Alzheimer's disease"),
    ("Paraprevotella", "Parkinson's disease"),
    ("Bulleidia", "Parkinson's disease"),
    ("Rothia", "Parkinson's disease"),
    ("Citrobacter", "Parkinson's disease"),
    ("Acidaminococcus", "Parkinson's disease"),
    ("Stenotrophomonas", "Parkinson's disease"),
    ("Halomonas", "Alzheimer's disease"),
    ("Roseburia inulinivorans", "Parkinson's disease"),
    ("Phascolarctobacterium", "Parkinson's disease"),
]


def main():
    G = json.load(open(os.path.join(HERE, "graph.json")))
    tax = Taxonomy()
    keys = {}
    for e in G["edges"]:
        keys[(e["taxon"], e["disease"])] = e

    want = {}
    for t, d in PAIRS:
        e = keys[(t, d)]
        want[(e["taxon_key"].split(":", 1)[1], d)] = (t, d, e["direction"], e["n_papers"])

    buckets = defaultdict(list)
    for loader in (load_disbiome, load_peryton):
        recs, name = loader()
        for r in recs or []:
            our = DISEASE_MAP.get(r["disease"].lower())
            out = OUTCOME.get(r["outcome"].lower())
            if not (our and out and r["microbe"]):
                continue
            tid, sci, rank, how = tax.resolve(r["microbe"])
            if not tid or (tid, our) not in want:
                continue
            buckets[(tid, our)].append((name, r["microbe"], sci, rank, out, r["pmid"]))

    for k, (t, d, odir, npap) in want.items():
        print("=" * 90)
        print(f"{t}  |  {d}   OURS={odir} ({npap} papers)   taxid={k[0]}")
        for name, raw, sci, rank, out, pmid in sorted(buckets.get(k, [])):
            print(f"   {name:9} {out:9} raw={raw!r:38} -> {sci} ({rank})  pmid={pmid}")


if __name__ == "__main__":
    main()
