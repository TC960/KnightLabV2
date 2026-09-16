#!/usr/bin/env python3
"""Assemble, per doubly-contradicted pair, the sentences our papers actually wrote.

14 (taxon, disease) pairs are contradicted by BOTH Disbiome and Peryton
(`doubly_contradicted.json`). Two independent curations agreeing against us is the
strongest error signal available, so each deserves reading rather than arbitration
by vote.

What this builds is the reading material, not the verdict: for every paper backing
one of those edges, the relation-bearing sentences that actually name the taxon,
pulled from `relation_sentences.json` (the filtered substrate validated at 93.9%
recall). A packet therefore contains the paper's own words about that organism and
nothing else -- which is what is needed to decide whether the extractor read the
paper correctly, and separately whether the paper supports the direction we stored.

Those are DIFFERENT questions and the packets keep them apart:
  - extraction error : the paper says the opposite of what we recorded.
  - genuine dispute  : the paper says what we recorded, and disagrees with the
                       curated databases. Then nobody is "wrong" and the edge is
                       real disagreement in the literature.

Sentences are matched on TAXID, not surface string, so "Ruminococcaceae" reaches
the node stored as Oscillospiraceae.
"""
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "_adjudication")


def norm(s):
    s = (s or "").replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    pairs = json.load(open(os.path.join(HERE, "doubly_contradicted.json")))
    G = json.load(open(os.path.join(HERE, "graph.json")))
    rs = json.load(open(os.path.join(HERE, "relation_sentences.json")))["papers"]
    rs_norm = {norm(t): v for t, v in rs.items()}

    edge_papers = {}
    for e in G["edges"]:
        if e["taxon_key"].startswith("ncbi:"):
            edge_papers[(e["taxon_key"].split(":", 1)[1], e["disease"])] = e

    os.makedirs(OUTDIR, exist_ok=True)
    manifest, no_text = [], []
    for i, p in enumerate(pairs, 1):
        key = (p["taxid"], p["disease"])
        e = edge_papers.get(key)
        if not e:
            continue
        blocks, found = [], 0
        for title in e.get("papers", []):
            rec = rs_norm.get(norm(title))
            if rec is None:
                blocks.append(f"### PAPER: {title}\n(no filtered text available — "
                              f"paper not in relation_sentences.json)\n")
                continue
            hits = [k for k in rec["kept"]
                    if any(t[1] == p["taxid"] for t in k["taxa"])]
            found += len(hits)
            body = "\n".join(f"  - {h['s']}" for h in hits[:14]) or "  (taxon not in any kept sentence)"
            blocks.append(f"### PAPER: {title}\nLINK: {rec.get('link','')}\n"
                          f"SENTENCES NAMING {p['taxon']}:\n{body}\n")
        if not found:
            no_text.append(p["taxon"] + " / " + p["disease"])
        fn = os.path.join(OUTDIR, f"pair{i:02d}.md")
        open(fn, "w").write(
            f"# PAIR {i}: {p['taxon']} in {p['disease']}\n\n"
            f"- OUR GRAPH says: **{p['ours']}** ({p['n_papers']} paper(s), "
            f"up={e['n_up']} down={e['n_down']})\n"
            f"- Disbiome says: **{p['disbiome']}** ({p['disbiome_n']} record(s))\n"
            f"- Peryton says: **{p['peryton']}** ({p['peryton_n']} record(s))\n\n"
            + "\n".join(blocks))
        manifest.append({"i": i, "taxon": p["taxon"], "disease": p["disease"],
                         "ours": p["ours"], "n_sentences": found, "file": fn})
        print(f"pair{i:02d}  {p['taxon'][:26]:27} {p['disease'][:22]:23} "
              f"{len(e.get('papers', [])):>2}p  {found:>2} sentences")

    json.dump(manifest, open(os.path.join(HERE, "_adjudication", "manifest.json"), "w"), indent=1)
    if no_text:
        print(f"\n{len(no_text)} pair(s) with NO retrievable sentence — cannot be "
              f"adjudicated from text here:")
        for n in no_text:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
