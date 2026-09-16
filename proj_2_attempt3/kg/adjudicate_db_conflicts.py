#!/usr/bin/env python3
"""Where Disbiome and Peryton contradict EACH OTHER, who is right?

Two independent curations disagreeing on the same association is the strongest
error signal available, stronger than either disagreeing with us: at least one of
them is wrong by construction, and neither can be the reference for the other.

`calibrate_agreement.py` turned up 5 such pairs. Before reading any paper, ask
the cheap structural question first -- the method that keeps working in this
repo: WHICH PUBLICATIONS do the contradicting records come from? If a cluster of
contradictions traces to one paper, it is one curation error, not N independent
ones, and the write-up should say so.

Adjudication then uses `relation_sentences.json` (the Task 1 filtered
relation-bearing sentences, which carry verbatim text and tagged taxa) rather
than MAIN_DATA, so it runs in an environment that has only the git checkout.

    python adjudicate_db_conflicts.py
"""
import argparse
import json
import os
import re
from collections import defaultdict

from calibrate_agreement import build_pairs
from validate_external import (DISEASE_MAP, GRAPH, OUTCOME, load_disbiome,
                               load_peryton)

HERE = os.path.dirname(os.path.abspath(__file__))
SENTENCES = os.path.join(HERE, "relation_sentences.json")
OUT = os.path.join(HERE, "db_conflicts.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", type=int, default=4,
                    help="verbatim sentences to print per implicated paper")
    a = ap.parse_args()

    G = json.load(open(GRAPH))
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy()

    src = {}
    for loader in (load_disbiome, load_peryton):
        recs, nm = loader()
        if recs is not None:
            src[nm] = (recs, build_pairs(G, recs, tax))
    if len(src) < 2:
        print("need both databases")
        return

    (n1, (r1, p1)), (n2, (r2, p2)) = src.items()
    i1 = {(p["taxon_key"], p["disease"]): p for p in p1}
    i2 = {(p["taxon_key"], p["disease"]): p for p in p2}
    conflicts = [k for k in set(i1) & set(i2)
                 if i1[k]["ref_dir"] != i2[k]["ref_dir"]]

    print("=" * 78)
    print(f"{len(conflicts)} pairs where {n1} and {n2} contradict EACH OTHER")
    print("=" * 78)
    for k in sorted(conflicts, key=lambda k: i1[k]["taxon"]):
        print(f"  {i1[k]['taxon'][:24]:25} {i1[k]['disease'][:26]:27} "
              f"{n1}={i1[k]['ref_dir']:8} {n2}={i2[k]['ref_dir']:8} "
              f"ours={i1[k]['our_dir']}")

    # ---- the structural question: how many PAPERS are behind them? ----------
    want = {k: (i1[k]["taxon"], i1[k]["disease"]) for k in conflicts}
    taxid_of = {}
    for k in conflicts:
        taxid_of[k] = k[0].split(":", 1)[1]

    src_pubs = defaultdict(lambda: defaultdict(set))  # db -> pair -> {(pmid,title)}
    for nm, (recs, _) in src.items():
        for r in recs:
            dis = DISEASE_MAP.get(r["disease"].lower())
            if not dis or not OUTCOME.get(r["outcome"].lower()):
                continue
            tid, _s, _r, _h = tax.resolve(r["microbe"]) if tax.ok else (None,)*4
            for k in conflicts:
                if tid == taxid_of[k] and dis == k[1]:
                    src_pubs[nm][k].add((str(r.get("pmid")),
                                         (r.get("ptitle") or "").strip()))

    allpubs = {p for nm in src_pubs for k in src_pubs[nm] for p in src_pubs[nm][k]}
    # The two databases spell the same title with different capitalisation, so
    # dedupe on the normalised form or the same paper is reported twice.
    by_pmid = defaultdict(dict)
    for pmid, title in allpubs:
        by_pmid[pmid].setdefault(title.lower().strip().rstrip("."), title)
    by_pmid = {p: sorted(v.values()) for p, v in by_pmid.items()}
    print(f"\n  publications behind all {len(conflicts)} contradictions: "
          f"{len(by_pmid)}")
    for pmid, titles in by_pmid.items():
        print(f"    PMID {pmid} — {titles[0][:64]}")
    if len(by_pmid) == 1:
        print("  => ONE paper, curated twice, read opposite ways. This is a single"
              "\n     curation error, not "
              f"{len(conflicts)} independent contradictions.")

    # ---- verbatim evidence, from the paper if we extracted it too ----------
    sent = json.load(open(SENTENCES)).get("papers", {}) \
        if os.path.exists(SENTENCES) else {}
    out = {"n_conflicts": len(conflicts), "n_publications": len(by_pmid),
           "conflicts": [{"taxon": i1[k]["taxon"], "disease": i1[k]["disease"],
                          n1: i1[k]["ref_dir"], n2: i2[k]["ref_dir"],
                          "ours": i1[k]["our_dir"]} for k in conflicts],
           "publications": by_pmid,
           "quotes": {}}

    taxa_re = re.compile("|".join(sorted({re.escape(i1[k]["taxon"])
                                          for k in conflicts})), re.I)
    for pmid, titles in by_pmid.items():
        for title in titles:
            norm = title.lower().strip().rstrip(".")
            hit = [t for t in sent
                   if t.lower().strip().rstrip(".") == norm]
            if not hit:
                continue
            print(f"\n  VERBATIM from our own extraction of PMID {pmid}:")
            # Rank by how many of the contradicted taxa a sentence names, then by
            # whether it carries an explicit direction word. Taking the first N in
            # document order surfaces background prose; the decisive sentences are
            # the summary ones that name several taxa and a direction at once.
            DIR = re.compile(r"higher|lower|increas|decreas|reduc|enrich|deplet"
                             r"|elevat|abundan", re.I)
            scored = sorted(
                sent[hit[0]]["kept"],
                key=lambda x: (-len(set(m.group(0).lower()
                                        for m in taxa_re.finditer(x["s"]))),
                               -bool(DIR.search(x["s"]))))
            quotes = [x["s"] for x in scored[:a.quotes]]
            for q in quotes:
                print(f"    · {q[:340]}")
            out["quotes"][pmid] = quotes

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
