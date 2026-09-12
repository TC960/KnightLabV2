#!/usr/bin/env python3
"""Assemble the SAME knowledge graph, but from the human-curated gold standard.

Input : high_confidence - final_constrained_override.csv   (DOI, disorder, field,
        high_confidence_taxa) -- two rows per DOI, one Enriched + one Depleted
        + "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv"
        for DOI -> Title/Link (the gold CSV has no titles, and the extraction graph
        is keyed by title, so the join has to go through the datasheet).
Output: kg/graph_gold.json, identical schema to kg/graph.json
        {meta, nodes, edges, hierarchy, papers}

Deliberately NOT a second implementation. The gold rows are reshaped into the
column names build_kg.build() already expects (predicted_disease /
predicted_enriched / predicted_depleted) and handed to that function, so the two
graphs share one node/edge/hierarchy/paper-table code path and cannot drift. The
only gold-specific code is the loader below plus a disease-vocabulary extension.

Three judgement calls, all visible in the output:

1. **Compound disorders are split.** 17 DOIs carry a disorder like
   "Parkinson's Disease; Multiple System Atrophy". The gold does not say which
   taxon belongs to which disease, so the taxa are attributed to BOTH. That
   duplicates those papers' evidence across two diseases; `meta.n_multi_disease_papers`
   records how many. The alternative -- a compound "PD; MSA" disease node -- would
   join to nothing, internally or externally.
2. **Blank cells are absence of evidence, not evidence of absence.** 40 DOIs have
   both cells blank and contribute no edges at all; 48 more have exactly one
   direction filled. A blank Depleted cell is not read as "nothing was depleted".
3. **Non-taxon strings are dropped, 16S clade labels are kept.** "ambiguous",
   "unclassified genus", "Firmicutes/Bacteroidetes ratio" and "enterotype I" are
   not organisms. Things like "[Eubacterium] ventriosum group" or "UCG-005" ARE
   organisms that NCBI simply has no id for -- they stay in the graph as
   unresolved nodes, same as in the extraction graph.
"""
import argparse
import csv
import json
import os
import re

import build_kg
from build_kg import build, parse_taxa

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLD = os.path.expanduser(
    "~/Downloads/high_confidence - final_constrained_override.csv")
DEFAULT_SHEET = os.path.join(
    HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")

# Disorder labels the curators use that build_kg.DISEASE_MAP does not cover, or
# covers with different casing than the extraction graph settled on. Prepended so
# these exact readings win before the looser patterns get a chance; the goal is
# that a disease present in BOTH graphs carries the SAME string, otherwise the
# graph-vs-graph comparison would report phantom gold-only edges.
GOLD_DISEASE_EXTRA = [
    (r"cognitive decline|cognitive impairment", "Mild cognitive impairment", "MONDO:0005453"),
    (r"spinal cord injury", "Spinal cord injury", None),
    (r"traumatic brain injury", "Traumatic brain injury", None),
    (r"multiple system atrophy", "Multiple system atrophy", "MONDO:0007803"),
    (r"essential tremor", "Essential tremor", "MONDO:0008457"),
    (r"rett syndrome", "Rett syndrome", "MONDO:0010726"),
    (r"tuberous sclerosis", "Tuberous sclerosis complex", "MONDO:0001734"),
    (r"normal pressure hydrocephalus", "Idiopathic normal pressure hydrocephalus", None),
    (r"familial dysautonomia", "Familial dysautonomia", "MONDO:0009175"),
    (r"creutzfeldt", "Creutzfeldt-Jakob disease", "MONDO:0005611"),
    (r"cerebral palsy", "Cerebral palsy", "MONDO:0006497"),
    (r"encephalitis", "Encephalitis", None),
    (r"encephalopathy", "Encephalopathy", None),
]

# strings in the taxa cells that are not organisms
NON_TAXON = re.compile(
    r"^(ambiguous|none|n/?a|unclassified genus|enterotype\s+[ivx]+|"
    r".*\bratio\b.*)$", re.I)


def load_titles(sheet_path):
    """DOI (lowercased) -> (title, link). The gold CSV only carries DOIs."""
    out = {}
    with open(sheet_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            doi = (r.get("DOI") or "").strip().lower()
            if not doi:
                continue
            out.setdefault(doi, ((r.get("Title") or "").strip(),
                                 (r.get("Link (Use DOI or Title if missing)") or "").strip()))
    return out


def clean_cell(cell):
    """gold taxa cell -> list of surface strings, minus the non-organisms."""
    return [t for t in parse_taxa(cell) if not NON_TAXON.match(t.strip())]


def load_gold(gold_path, sheet_path):
    """-> (rows in build_kg's column shape, stats dict)"""
    doi2title = load_titles(sheet_path)
    by_doi = {}
    order = []
    with open(gold_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            doi = (r.get("DOI") or "").strip()
            if not doi:
                continue
            if doi not in by_doi:
                by_doi[doi] = {"disorder": (r.get("disorder") or "").strip(),
                               "Enriched": [], "Depleted": []}
                order.append(doi)
            fld = (r.get("field") or "").strip().title()
            if fld in ("Enriched", "Depleted"):
                by_doi[doi][fld] += clean_cell(r.get("high_confidence_taxa"))

    rows = []
    st = {"dois": len(by_doi), "no_title": 0, "blank_both": 0, "blank_one": 0,
          "multi_disease": 0, "dropped_non_taxon": 0}
    for doi in order:
        g = by_doi[doi]
        up, dn = g["Enriched"], g["Depleted"]
        if not up and not dn:
            st["blank_both"] += 1
            continue
        if not up or not dn:
            st["blank_one"] += 1
        title, link = doi2title.get(doi.lower(), ("", ""))
        if not title:
            st["no_title"] += 1
            title = doi                      # never drop a paper for a missing title
        parts = [p.strip() for p in g["disorder"].split(";") if p.strip()] or ["Unspecified"]
        if len(parts) > 1:
            st["multi_disease"] += 1
        for p in parts:
            rows.append({"title": title,
                         "link": link or f"https://doi.org/{doi}",
                         "doi": doi,
                         "predicted_disease": p,
                         "predicted_enriched": "; ".join(up),
                         "predicted_depleted": "; ".join(dn)})
    st["contributing_papers"] = len({r["title"] for r in rows})
    st["rows_emitted"] = len(rows)
    return rows, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--sheet", default=DEFAULT_SHEET)
    ap.add_argument("--min-papers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "graph_gold.json"))
    ap.add_argument("--no-taxonomy", action="store_true")
    a = ap.parse_args()

    # extend the shared disease vocabulary before build() reads it
    build_kg.DISEASE_MAP = GOLD_DISEASE_EXTRA + build_kg.DISEASE_MAP

    rows, st = load_gold(a.gold, a.sheet)
    tax = None
    if not a.no_taxonomy:
        from taxonomy import Taxonomy
        tax = Taxonomy()
        print(f"NCBI taxdump: {'loaded' if tax.ok else 'NOT FOUND -> string folding only'}")

    nodes, edges, hierarchy, papers_tbl = build(rows, a.min_papers, tax)
    meta = {
        "source": os.path.basename(a.gold),
        "provenance": "human-curated gold standard (high-confidence taxa)",
        "papers_in": st["dois"],
        "papers_contributing": st["contributing_papers"],
        "papers_blank_both_cells": st["blank_both"],
        "papers_blank_one_cell": st["blank_one"],
        "n_multi_disease_papers": st["multi_disease"],
        "n_taxa": sum(1 for n in nodes if n["type"] == "taxon"),
        "n_diseases": sum(1 for n in nodes if n["type"] == "disease"),
        "n_edges": len(edges),
        "n_replicated": sum(1 for e in edges if e["n_papers"] > 1),
        "n_contested": sum(1 for e in edges if e["contested"]),
        "n_taxa_resolved": sum(1 for n in nodes if n["type"] == "taxon" and n.get("resolved")),
        "n_hierarchy_links": len(hierarchy),
        "n_papers_table": len(papers_tbl),
        "n_papers_with_metadata": sum(1 for p in papers_tbl if p["has_meta"]),
        "min_papers": a.min_papers,
        "note": ("Built from human annotations, not LLM extraction. Same edge semantics "
                 "as graph.json: weight is evidence count, contested edges retained. "
                 "Compound disorders ('PD; MSA') attribute their taxa to every disease "
                 "named, so those papers' evidence appears under each."),
    }
    json.dump({"meta": meta, "nodes": nodes, "edges": edges,
               "hierarchy": hierarchy, "papers": papers_tbl}, open(a.out, "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {a.out}")
    print("\nmost-replicated edges:")
    for e in sorted(edges, key=lambda e: -e["n_papers"])[:12]:
        flag = " CONTESTED" if e["contested"] else ""
        print(f"  {e['n_papers']:3}p  {e['taxon'][:24]:24} {e['direction']:9} in "
              f"{e['disease'][:28]:28} (up={e['n_up']} dn={e['n_down']}){flag}")


if __name__ == "__main__":
    main()
