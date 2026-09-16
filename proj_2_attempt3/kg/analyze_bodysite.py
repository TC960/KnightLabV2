#!/usr/bin/env python3
"""Does body site belong in the edge key? Measure it before changing anything.

THE CLAIM UNDER TEST (top lever out of FINDINGS_task3_adjudication.md): edges
pool papers that sampled different body sites onto one node, manufacturing false
contradictions -- Rothia/Parkinson's is two saliva studies colliding with gut
records in Disbiome and Peryton, not a disagreement -- so body site should be part
of the edge key, and doing so should make the external comparison honest.

That was diagnosed from two pairs. This script asks how big it actually is, using
the completed labelling in body_site.json (all 281 contributing papers), and
reports the CEILING on any possible effect before spending a rebuild on it.

    python analyze_bodysite.py
"""
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GUT = {"stool", "gut biopsy"}


def main():
    G = json.load(open(os.path.join(HERE, "graph.json")))
    site = {t: v["site"] for t, v in json.load(open(os.path.join(HERE, "body_site.json")))["papers"].items()}
    papers = [p["title"] for p in G["papers"]]

    print(f"papers: {len(papers)}   sites: {dict(Counter(site.get(t,'?') for t in papers))}")
    nonstool = {t for t in papers if site.get(t) not in GUT}
    print(f"non-gut papers: {len(nonstool)} ({100*len(nonstool)/len(papers):.1f}%)")
    for t in sorted(nonstool):
        print(f"    [{site[t]:11}] {t[:80]}")

    # --- edge-level exposure --------------------------------------------------
    pure_gut = mixed = pure_nongut = 0
    mixed_edges, nongut_edges = [], []
    for e in G["edges"]:
        sites = {site.get(papers[x["i"]], "unknown") for x in e["ev"]}
        g = bool(sites & GUT)
        n = bool(sites - GUT)
        if g and n:
            mixed += 1
            mixed_edges.append(e)
        elif n:
            pure_nongut += 1
            nongut_edges.append(e)
        else:
            pure_gut += 1
    tot = len(G["edges"])
    print(f"\nedges: {tot}")
    print(f"  gut-only        : {pure_gut} ({100*pure_gut/tot:.1f}%)")
    print(f"  MIXED sites     : {mixed} ({100*mixed/tot:.1f}%)   <- the false-contradiction mechanism")
    print(f"  non-gut only    : {pure_nongut} ({100*pure_nongut/tot:.1f}%)   <- wrongly scored against gut databases")

    print("\n  mixed-site edges (these are what an edge-key change would split):")
    for e in sorted(mixed_edges, key=lambda e: -e["n_papers"]):
        ss = Counter(site.get(papers[x["i"]], "?") for x in e["ev"])
        print(f"    {e['taxon'][:26]:27} {e['disease'][:24]:25} {e['direction']:9} "
              f"{e['n_papers']}p contested={e['contested']} {dict(ss)}")

    # --- ceiling on the external comparison ----------------------------------
    # Only edges that (a) resolve to a taxid and (b) carry non-gut evidence can
    # possibly change a decisive pair, because that is the whole join key.
    exposed = {(e["taxon_key"].split(":", 1)[1], e["disease"])
               for e in mixed_edges + nongut_edges if e["taxon_key"].startswith("ncbi:")}
    print(f"\n(taxid, disease) pairs carrying ANY non-gut evidence: {len(exposed)}")

    for src, path in (("Disbiome", "disbiome_experiments.json"), ("Peryton", "Peryton-results.tsv")):
        dec = decisive_pairs(G, src)
        if dec is None:
            print(f"  {src}: source unavailable")
            continue
        hit = dec & exposed
        print(f"  {src}: {len(dec)} decisive pairs, of which {len(hit)} carry non-gut evidence "
              f"-> CEILING on any change = {len(hit)} pair(s)"
              + (f"  {sorted(hit)}" if hit else ""))


def decisive_pairs(G, which):
    """The (taxid, disease) pairs that actually get scored -- reusing the validator."""
    import validate_external as V
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy(verbose=False)
    recs, _ = V.load_disbiome() if which == "Disbiome" else V.load_peryton()
    if recs is None:
        return None
    ref = defaultdict(Counter)
    for r in recs:
        our = V.DISEASE_MAP.get(r["disease"].lower())
        out = V.OUTCOME.get(r["outcome"].lower())
        if not (our and out and r["microbe"]):
            continue
        tid, *_ = tax.resolve(r["microbe"])
        if tid:
            ref[(tid, our)][out] += 1
    ours = {(e["taxon_key"].split(":", 1)[1], e["disease"]): e
            for e in G["edges"] if e["taxon_key"].startswith("ncbi:")}
    shared = {d for _, d in ref} & {d for _, d in ours}
    out = set()
    for k in set(ref) & set(ours):
        if k[1] not in shared:
            continue
        e, rv = ours[k], ref[k]
        rdir = ("enriched" if rv["enriched"] > rv["depleted"]
                else "depleted" if rv["depleted"] > rv["enriched"] else "contested")
        if e["contested"] or rdir == "contested":
            continue
        out.add(k)
    return out


if __name__ == "__main__":
    main()
