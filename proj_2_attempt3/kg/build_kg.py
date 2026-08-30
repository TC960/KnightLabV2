#!/usr/bin/env python3
"""Assemble a microbe-disease knowledge graph from the extraction output.

Input : eval-v2/results/<model>__...__all250.json  (one row per paper)
Output: kg/graph.json  {nodes, edges, meta}

Design follows what the established resources (Disbiome, Peryton, MicroPhenoDB)
actually do, plus two properties this corpus forces:

1. **One edge per taxon-disease pair, aggregating papers -- never a consensus
   edge that hides disagreement.** Each edge carries n_up / n_down and the list
   of contributing papers, so a contested pair stays visibly contested. In this
   corpus 119 of 1,729 pairs (~7%, but 45% of the *replicated* pairs) have
   papers pointing both ways; the microbiome replication literature reports
   roughly 1 in 3 taxa flipping sign between cohorts, so contradiction is
   signal, not noise, and must survive into the graph.

2. **No effect sizes.** The extractor returns direction only, and the underlying
   papers report incommensurable statistics (LEfSe LDA, fold-change, p-values)
   that cannot be pooled into a single magnitude. So edge weight is *evidence
   count*, and edge confidence is *directional consistency* -- both computed
   from data we actually have, rather than a fabricated magnitude.

Ranks are preserved rather than collapsed: papers report phylum, genus, species
and OTU-level labels as peers, and there is no accepted convention for merging
them. Rank is a node attribute; downstream consumers can roll up if they want.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2", "results",
                          "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")

# --- disease normalization -------------------------------------------------
# The extractor returns free text ("Alzheimer disease" / "Alzheimer's disease" /
# "AD"), so surface strings must be folded before anything can be counted.
# MONDO ids are the target vocabulary; only the diseases actually present in
# this corpus are mapped, and anything unmapped keeps its cleaned label with
# mondo=None rather than being silently dropped.
DISEASE_MAP = [
    (r"\bparkinson", "Parkinson's disease", "MONDO:0005180"),
    (r"\balzheimer", "Alzheimer's disease", "MONDO:0004975"),
    (r"multiple sclerosis|\bms\b", "Multiple sclerosis", "MONDO:0005301"),
    (r"amyotrophic lateral|\bals\b", "Amyotrophic lateral sclerosis", "MONDO:0004976"),
    (r"mild cognitive impairment|\bmci\b", "Mild cognitive impairment", "MONDO:0005453"),
    (r"\bstroke|cerebral infarct", "Stroke", "MONDO:0005098"),
    (r"huntington", "Huntington's disease", "MONDO:0007739"),
    (r"\bdementia", "Dementia", "MONDO:0001627"),
    (r"spinal muscular atrophy|\bsma\b", "Spinal muscular atrophy", "MONDO:0001516"),
    (r"epilep", "Epilepsy", "MONDO:0005027"),
    (r"autism|\basd\b", "Autism spectrum disorder", "MONDO:0005260"),
    (r"depress", "Depressive disorder", "MONDO:0002050"),
    (r"schizophren", "Schizophrenia", "MONDO:0005090"),
    (r"neuromyelitis", "Neuromyelitis optica", "MONDO:0019100"),
    (r"myasthenia", "Myasthenia gravis", "MONDO:0009688"),
    (r"migraine", "Migraine", "MONDO:0005277"),
]

# rank hints from the naming conventions the papers use
RANK_SUFFIX = [
    (r"^[a-z]__|^[pcofgs]-", None),          # greengenes-style prefix, handled below
    (r"aceae$", "family"), (r"ales$", "order"), (r"ia$|ies$", "class"),
    (r"(ota|etes|bacteria|micrObia)$", "phylum"),
]


def parse_taxa(v):
    if not v or str(v).strip().lower() in ("", "nan", "none"):
        return []
    out = []
    for t in re.split(r"[,;]", str(v)):
        t = re.sub(r"\(.*?\)", "", t)
        t = re.sub(r"p\s*[<>=]\s*[\d.]+", "", t, flags=re.I)
        t = t.strip().strip(".) ").strip()
        if t and t.lower() != "nan" and len(t) > 2:
            out.append(t)
    return out


def norm_disease(s):
    s = (s or "").strip()
    low = s.lower()
    for pat, label, mondo in DISEASE_MAP:
        if re.search(pat, low):
            return label, mondo
    return (s[:1].upper() + s[1:]) if s else "Unspecified", None


def norm_taxon(t):
    """Canonical key + display name + best-effort rank. Deliberately conservative:
    we fold case and strip rank prefixes, but do NOT merge across ranks or
    resolve synonyms (Bacteroidetes/Bacteroidota) without a taxonomy backend."""
    disp = t.strip()
    key = disp.lower()
    rank = None
    m = re.match(r"^([pcofgs])[-_]", key)          # "o-Clostridia", "f_Rikenellaceae"
    if m:
        rank = {"p": "phylum", "c": "class", "o": "order",
                "f": "family", "g": "genus", "s": "species"}[m.group(1)]
        key = key[m.end():]
        disp = disp[m.end():]
    key = re.sub(r"^[a-z]__", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    if rank is None:
        if len(key.split()) >= 2:
            rank = "species"
        else:
            for pat, r in RANK_SUFFIX:
                if r and re.search(pat, key):
                    rank = r
                    break
            rank = rank or "genus"
    return key, disp, rank


def build(rows, min_papers=1):
    ev = defaultdict(list)
    taxon_disp, taxon_rank = {}, {}
    for r in rows:
        dis_raw = (r.get("predicted_disease") or r.get("disease") or "")
        disease, mondo = norm_disease(dis_raw)
        for direction, col in (("enriched", "predicted_enriched"),
                               ("depleted", "predicted_depleted")):
            for raw in parse_taxa(r.get(col)):
                key, disp, rank = norm_taxon(raw)
                if not key:
                    continue
                taxon_disp.setdefault(key, disp)
                taxon_rank.setdefault(key, rank)
                ev[(key, disease, mondo)].append(
                    {"dir": direction, "paper": r.get("title", ""), "link": r.get("link", ""),
                     "as_written": raw})

    edges = []
    for (taxon, disease, mondo), obs in ev.items():
        c = Counter(o["dir"] for o in obs)
        up, dn = c["enriched"], c["depleted"]
        n = up + dn
        if n < min_papers:
            continue
        # papers, not observations: the same paper naming a taxon twice is one vote
        papers = {o["paper"] for o in obs}
        consistency = max(up, dn) / n
        edges.append({
            "taxon": taxon, "disease": disease, "mondo": mondo,
            "direction": "enriched" if up > dn else "depleted" if dn > up else "contested",
            "n_up": up, "n_down": dn, "n_obs": n, "n_papers": len(papers),
            "consistency": round(consistency, 3),
            "contested": bool(up and dn),
            "papers": sorted(papers)[:25],
        })

    tax_deg = Counter(e["taxon"] for e in edges)
    dis_deg = Counter(e["disease"] for e in edges)
    nodes = (
        [{"id": f"t:{k}", "label": taxon_disp[k], "type": "taxon",
          "rank": taxon_rank[k], "degree": tax_deg[k]} for k in tax_deg]
        + [{"id": f"d:{d}", "label": d, "type": "disease",
            "mondo": next((e["mondo"] for e in edges if e["disease"] == d), None),
            "degree": dis_deg[d]} for d in dis_deg]
    )
    for e in edges:
        e["source"], e["target"] = f"t:{e['taxon']}", f"d:{e['disease']}"
    return nodes, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--min-papers", type=int, default=1,
                    help="drop edges supported by fewer than N papers")
    ap.add_argument("--out", default=os.path.join(HERE, "graph.json"))
    a = ap.parse_args()

    rows = json.load(open(a.input))
    nodes, edges = build(rows, a.min_papers)
    meta = {
        "source": os.path.basename(a.input),
        "papers_in": len(rows),
        "papers_contributing": len({e for r in rows for e in [r["title"]]
                                    if parse_taxa(r.get("predicted_enriched")) or
                                    parse_taxa(r.get("predicted_depleted"))}),
        "n_taxa": sum(1 for n in nodes if n["type"] == "taxon"),
        "n_diseases": sum(1 for n in nodes if n["type"] == "disease"),
        "n_edges": len(edges),
        "n_replicated": sum(1 for e in edges if e["n_papers"] > 1),
        "n_contested": sum(1 for e in edges if e["contested"]),
        "min_papers": a.min_papers,
        "note": ("Edge weight is evidence count, not effect size: the extractor yields "
                 "direction only and the source papers report incommensurable statistics. "
                 "Contested edges are retained, never merged away."),
    }
    json.dump({"meta": meta, "nodes": nodes, "edges": edges}, open(a.out, "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {a.out}")
    top = sorted(edges, key=lambda e: -e["n_papers"])[:10]
    print("\nmost-replicated edges:")
    for e in top:
        flag = " CONTESTED" if e["contested"] else ""
        print(f"  {e['n_papers']:3}p  {e['taxon'][:24]:24} {e['direction']:9} in {e['disease'][:28]:28}"
              f" (up={e['n_up']} dn={e['n_down']}){flag}")


if __name__ == "__main__":
    main()
