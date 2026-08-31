#!/usr/bin/env python3
"""Validate the knowledge graph against curated microbe-disease databases.

Two sources, one comparison path (the shared core is `compare_against`; the only
per-source code is a loader, so the two can never drift apart):

  disbiome  ~10.9k curated experiments, 372 diseases. Open API, fetched and cached.
            11 diseases overlap our corpus.
  peryton   ~7.9k curated entries, 43 diseases. NO reachable API -- every endpoint
            probed returns 404 and the site is a client-side app -- so its TSV must
            be exported by hand from https://dianalab.e-ce.uth.gr/peryton/#/associations
            (leave Microorganism blank for all) and dropped in beside this script.
            Weighted toward GI/cancer, so only 3 diseases overlap our corpus, but it
            is DEEPER than Disbiome on those (Parkinson's 558 vs 287, Alzheimer's
            205 vs 54).

BOTH SIDES MUST PASS THROUGH THE SAME NORMALIZER. Curated taxids cannot be trusted
as join keys: for a paper reporting just "Prevotella", Disbiome records taxid 59823
("Prevotella sp.", a SPECIES) where the genus is 838. Joining on the stored id
dropped Prevotella/Parkinson's -- 16 papers here, and an AGREEMENT with Disbiome --
as a phantom gap. Re-resolving their organism NAME through kg/taxonomy.py lifted
overlap 188 -> 238 and recall 41.4% -> 53.6%.

    python validate_external.py                 # both sources
    python validate_external.py --source peryton
"""
import argparse
import csv
import json
import os
import subprocess
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
DISBIOME_CACHE = os.path.join(HERE, "disbiome_experiments.json")
DISBIOME_API = "https://disbiome.ugent.be:8080/experiment"
PERYTON_TSV = os.path.join(HERE, "Peryton-results.tsv")

# curated disease label -> our normalized label. Unmapped diseases are ignored
# rather than force-matched.
DISEASE_MAP = {
    # Disbiome (MedDRA)
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
    # Peryton (MeSH headings, no possessive)
    "parkinson disease": "Parkinson's disease",
    "alzheimer disease": "Alzheimer's disease",
}
OUTCOME = {"elevated": "enriched", "increased": "enriched",
           "reduced": "depleted", "decreased": "depleted"}


def load_disbiome(refresh=False):
    if not os.path.exists(DISBIOME_CACHE) or refresh:
        print(f"fetching {DISBIOME_API} ...")
        subprocess.run(["curl", "-sL", "--max-time", "180", DISBIOME_API,
                        "-o", DISBIOME_CACHE], check=True)
    out = []
    for r in json.load(open(DISBIOME_CACHE)):
        if r.get("host_type") != "Human":
            continue
        out.append({"disease": (r.get("disease_name") or "").strip(),
                    "microbe": (r.get("organism_name") or "").strip(),
                    "outcome": (r.get("qualitative_outcome") or "").strip(),
                    "pmid": r.get("publication_id")})
    return out, "Disbiome"


def load_peryton():
    if not os.path.exists(PERYTON_TSV):
        return None, "Peryton"
    out = []
    with open(PERYTON_TSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("host_species_name") != "Homo sapiens":
                continue
            # keep only true disease-vs-healthy contrasts, matching our extraction's
            # gate; "Normal adjacent" is tumour-adjacent tissue, a different question
            if r.get("group_two") != "Healthy Controls":
                continue
            out.append({"disease": (r.get("disease_name") or "").strip(),
                        "microbe": (r.get("microbe_scientific_name") or "").strip(),
                        "outcome": (r.get("relationship_name") or "").strip(),
                        "pmid": r.get("publication_PMID")})
    return out, "Peryton"


def compare_against(G, records, name, tax, show=10):
    ref = defaultdict(Counter)
    kept = 0
    for r in records:
        our = DISEASE_MAP.get(r["disease"].lower())
        out = OUTCOME.get(r["outcome"].lower())
        if not (our and out and r["microbe"]):
            continue
        tid, _sci, _rank, _how = tax.resolve(r["microbe"]) if tax.ok else (None, 0, 0, 0)
        if not tid:
            continue
        ref[(tid, our)][out] += 1
        kept += 1

    ours = {}
    for e in G["edges"]:
        k = e["taxon_key"]
        if k.startswith("ncbi:"):
            ours[(k.split(":", 1)[1], e["disease"])] = e

    shared = {d for _, d in ref} & {d for _, d in ours}
    ref_s = {k: v for k, v in ref.items() if k[1] in shared}
    our_s = {k: v for k, v in ours.items() if k[1] in shared}
    overlap = set(ref_s) & set(our_s)

    agree, disagree = 0, []
    for k in overlap:
        e, rv = our_s[k], ref_s[k]
        rdir = ("enriched" if rv["enriched"] > rv["depleted"]
                else "depleted" if rv["depleted"] > rv["enriched"] else "contested")
        odir = "contested" if e["contested"] else e["direction"]
        if odir == "contested" or rdir == "contested":
            continue
        if odir == rdir:
            agree += 1
        else:
            disagree.append((e, rv, odir, rdir))
    comparable = agree + len(disagree)

    print("=" * 78)
    print(f"{name.upper()}")
    print("=" * 78)
    print(f"  usable records          : {kept} (human, mapped disease, resolvable taxon)")
    print(f"  diseases in both        : {len(shared)} -> {', '.join(sorted(shared))}")
    print(f"  (taxon,disease) in both : {len(overlap)}")
    print(f"    of ours in scope      : {len(overlap)}/{len(our_s)} "
          f"({100*len(overlap)/max(len(our_s),1):.1f}%)")
    print(f"    of theirs             : {len(overlap)}/{len(ref_s)} "
          f"({100*len(overlap)/max(len(ref_s),1):.1f}%)  <- recall")
    if comparable:
        print(f"  direction, {comparable} decisive pairs:")
        print(f"    agree    : {agree} ({100*agree/comparable:.1f}%)")
        print(f"    disagree : {len(disagree)} ({100*len(disagree)/comparable:.1f}%)")
    if disagree:
        disagree.sort(key=lambda x: -x[0]["n_papers"])
        print(f"\n  disagreements (top {min(show,len(disagree))} by our evidence):")
        for e, rv, odir, rdir in disagree[:show]:
            print(f"    {e['taxon'][:26]:27} {e['disease'][:24]:25} "
                  f"ours={odir}({e['n_papers']}p)  {name.lower()}={rdir}({rv[rdir]})")
    return {"name": name, "overlap": len(overlap), "agree": agree,
            "disagree": len(disagree), "comparable": comparable,
            "recall": len(overlap) / max(len(ref_s), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["disbiome", "peryton", "both"], default="both")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()

    G = json.load(open(GRAPH))
    from taxonomy import Taxonomy
    tax = Taxonomy()
    if not tax.ok:
        print("FATAL: NCBI taxdump missing — the join needs it. See kg/taxonomy.py.")
        return
    print(f"graph: {len(G['edges'])} edges\n")

    summaries = []
    if a.source in ("disbiome", "both"):
        recs, nm = load_disbiome(a.refresh)
        summaries.append(compare_against(G, recs, nm, tax, a.show))
        print()
    if a.source in ("peryton", "both"):
        recs, nm = load_peryton()
        if recs is None:
            print("=" * 78)
            print("PERYTON: no TSV found.")
            print(f"  Peryton has no reachable API. Export from")
            print(f"  https://dianalab.e-ce.uth.gr/peryton/#/associations (leave")
            print(f"  Microorganism blank) and save as {os.path.basename(PERYTON_TSV)}")
        else:
            summaries.append(compare_against(G, recs, nm, tax, a.show))

    if len(summaries) > 1:
        print("\n" + "=" * 78)
        print(f"{'source':12} {'overlap':>8} {'recall':>8} {'agree':>8} {'disagree':>9}")
        print("-" * 78)
        for s in summaries:
            rate = f"{100*s['agree']/s['comparable']:.1f}%" if s["comparable"] else "n/a"
            print(f"{s['name']:12} {s['overlap']:>8} {100*s['recall']:>7.1f}% "
                  f"{rate:>8} {s['disagree']:>9}")


if __name__ == "__main__":
    main()
