#!/usr/bin/env python3
"""Validate the knowledge graph against Disbiome, a hand-curated microbe-disease database.

Disbiome (https://disbiome.ugent.be) curates ~10.9k experiments from ~1.2k papers,
each recording a taxon that was Elevated or Reduced in a disease vs healthy
controls. Crucially it stores `organism_ncbi_id`, so the join is on NCBI taxid --
the same key this graph uses after taxonomy folding. That makes the comparison a
real entity-level join, not fuzzy string matching.

Three numbers matter, and they answer different questions:

  precision-like  of our edges on shared (taxon, disease) pairs, how many does
                  Disbiome agree with on DIRECTION?
  recall-like     of Disbiome's pairs for diseases we cover, how many did we find?
  disagreement    pairs where we and Disbiome assert opposite directions.

None is a pure accuracy figure. Disbiome curates a different, partly older paper
set, so a pair we have and they lack is not necessarily wrong -- it may be newer.
Expect substantial non-overlap in both directions; the literature on this puts
cross-study direction inconsistency around 1 in 3. Disagreements are the useful
output: they are specific, checkable claims.

    python validate_disbiome.py [--refresh]
"""
import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
CACHE = os.path.join(HERE, "disbiome_experiments.json")
API = "https://disbiome.ugent.be:8080/experiment"

# Disbiome disease names (MedDRA) -> our normalized labels. Only diseases present
# in our corpus are mapped; everything else is ignored rather than force-matched.
DISEASE_MAP = {
    "parkinson's disease": "Parkinson's disease",
    "alzheimer's disease": "Alzheimer's disease",
    "multiple sclerosis": "Multiple sclerosis",
    "amyotrophic lateral sclerosis": "Amyotrophic lateral sclerosis",
    "ischemic stroke": "Stroke",
    "huntington's disease": "Huntington's disease",
    "cognitive impairment": "Mild cognitive impairment",
    "mild cognitive impairment": "Mild cognitive impairment",
    "autism": "Autism spectrum disorder",
    "autism spectrum disorders": "Autism spectrum disorder",
    "schizophrenia": "Schizophrenia",
    "major depressive disorder": "Depressive disorder",
    "neuromyelitis optica": "Neuromyelitis optica",
    "myasthenia gravis": "Myasthenia gravis",
    "migraine": "Migraine",
    "drug-resistant epilepsy": "Epilepsy",
    "idiopathic focal epilepsy": "Epilepsy",
    "dementia": "Dementia",
}
OUTCOME = {"elevated": "enriched", "reduced": "depleted"}


def fetch(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return json.load(open(CACHE))
    print(f"fetching {API} ...")
    subprocess.run(["curl", "-sL", "--max-time", "180", API, "-o", CACHE], check=True)
    return json.load(open(CACHE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    G = json.load(open(GRAPH))
    dis = fetch(a.refresh)

    # Both sides MUST go through the same normalizer or the join silently misses.
    # Disbiome's stored organism_ncbi_id is not consistently at the rank the paper
    # reported: for a paper saying just "Prevotella" its curators recorded taxid
    # 59823 ("Prevotella sp.", a SPECIES) where the genus is 838. Joining on their
    # taxid therefore missed Prevotella/Parkinson's -- one of the most replicated
    # findings in the field, 16 papers in our graph. So we re-resolve their
    # organism_name with our own resolver and fall back to their taxid only when
    # the name will not resolve.
    from taxonomy import Taxonomy
    tax = Taxonomy()
    print(f"NCBI taxdump: {'loaded' if tax.ok else 'MISSING -> falling back to Disbiome taxids'}")

    # ---- Disbiome side: (taxid, disease) -> direction votes ----
    dref = defaultdict(Counter)
    dnames = {}
    kept = 0
    for r in dis:
        if r.get("host_type") != "Human":
            continue
        dn = (r.get("disease_name") or "").strip().lower()
        our = DISEASE_MAP.get(dn)
        name = (r.get("organism_name") or "").strip()
        out = OUTCOME.get((r.get("qualitative_outcome") or "").strip().lower())
        tid = None
        if tax.ok and name:
            rid, _sci, _rank, _how = tax.resolve(name)
            tid = rid
        if not tid:
            tid = r.get("organism_ncbi_id")          # fall back to their id
        if not (our and tid and out):
            continue
        dref[(str(tid), our)][out] += 1
        dnames[str(tid)] = name or str(tid)
        kept += 1

    # ---- our side: only taxid-resolved edges can be joined ----
    ours = {}
    unresolved = 0
    for e in G["edges"]:
        k = e["taxon_key"]
        if not k.startswith("ncbi:"):
            unresolved += 1
            continue
        ours[(k.split(":", 1)[1], e["disease"])] = e

    shared_dis = {d for _, d in dref} & {d for _, d in ours}
    dref_in_scope = {k: v for k, v in dref.items() if k[1] in shared_dis}
    ours_in_scope = {k: v for k, v in ours.items() if k[1] in shared_dis}

    print(f"Disbiome: {len(dis)} experiments -> {kept} usable "
          f"(human, mapped disease, NCBI id, clear outcome)")
    print(f"          {len(dref)} distinct (taxon, disease) pairs")
    print(f"Ours:     {len(G['edges'])} edges, {len(ours)} taxid-resolved "
          f"({unresolved} unresolved and unjoinable)")
    print(f"Diseases in both: {len(shared_dis)} -> {', '.join(sorted(shared_dis))}\n")

    overlap = set(dref_in_scope) & set(ours_in_scope)
    agree, disagree, examples = 0, [], []
    for k in overlap:
        e = ours_in_scope[k]
        dv = dref_in_scope[k]
        ddir = "enriched" if dv["enriched"] > dv["depleted"] else \
               "depleted" if dv["depleted"] > dv["enriched"] else "contested"
        odir = "contested" if e["contested"] else e["direction"]
        # a contested edge on either side is neither agreement nor contradiction
        if odir == "contested" or ddir == "contested":
            continue
        if odir == ddir:
            agree += 1
        else:
            disagree.append((k, e, dv, odir, ddir))

    comparable = agree + len(disagree)
    print("=" * 76)
    print(f"OVERLAP: {len(overlap)} (taxon, disease) pairs are in BOTH")
    print(f"  of ours in scope   : {len(overlap)}/{len(ours_in_scope)} "
          f"({100*len(overlap)/max(len(ours_in_scope),1):.1f}%)  <- how much of ours Disbiome corroborates")
    print(f"  of Disbiome's      : {len(overlap)}/{len(dref_in_scope)} "
          f"({100*len(overlap)/max(len(dref_in_scope),1):.1f}%)  <- how much of Disbiome we recovered")
    print(f"\nDIRECTION on the {comparable} pairs where both sides are decisive:")
    print(f"  agree    : {agree} ({100*agree/max(comparable,1):.1f}%)")
    print(f"  disagree : {len(disagree)} ({100*len(disagree)/max(comparable,1):.1f}%)")

    if disagree:
        print(f"\nDisagreements — specific, checkable claims (showing {min(a.show,len(disagree))}):")
        disagree.sort(key=lambda x: -x[1]["n_papers"])
        print(f"  {'taxon':26} {'disease':26} {'ours':>10} {'disbiome':>12}")
        print("  " + "-" * 76)
        for (tid, d), e, dv, odir, ddir in disagree[:a.show]:
            print(f"  {e['taxon'][:25]:26} {d[:25]:26} "
                  f"{odir+' ('+str(e['n_papers'])+'p)':>10} "
                  f"{ddir+' ('+str(dv[ddir])+')':>12}")

    only_ours = set(ours_in_scope) - set(dref_in_scope)
    only_dis = set(dref_in_scope) - set(ours_in_scope)
    print(f"\nNot in Disbiome: {len(only_ours)} of our pairs. Not necessarily wrong —")
    print(f"  Disbiome curates a different, partly older paper set.")
    top = sorted((ours_in_scope[k] for k in only_ours), key=lambda e: -e["n_papers"])[:6]
    for e in top:
        print(f"    {e['n_papers']:3}p  {e['taxon'][:28]:28} {e['direction']:9} {e['disease']}")
    print(f"\nIn Disbiome but not ours: {len(only_dis)} pairs (our extraction's recall gap).")


if __name__ == "__main__":
    main()
