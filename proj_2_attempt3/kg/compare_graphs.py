#!/usr/bin/env python3
"""Diff the human-gold graph against the LLM-extraction graph.

  python compare_graphs.py            # edge-level diff + direction disagreements
  python compare_graphs.py --json out.json

Two comparisons, because they answer different questions:

**Edge level** -- (taxon, disease) pairs present in one graph, the other, or both,
and where both are decisive, whether they point the same way. This is the graph
consumer's view: if you look up Akkermansia/Parkinson's, do the two resources tell
you the same thing?

**Paper level** -- for the 296 papers BOTH sources annotated, the (paper, taxon,
disease) triples where gold and extraction disagree on direction. This is the
extractor's error signal: an edge-level flip can come from either graph covering a
different set of papers, but a paper-level flip is one source reading one paper
backwards. Recomputed from the source CSV/JSON rather than from the graphs,
because build_kg truncates each edge's per-paper evidence list at 25.

**Disease folding.** The extraction graph splits diseases the curators lump: it
carries `Anti-NMDAR encephalitis`, `NMDAR encephalitis` and `Anti-N-methyl-D-
aspartate receptor encephalitis` where gold has `Encephalitis`; likewise
intracerebral-haemorrhage variants under `Stroke` and several cognitive labels
under `Mild cognitive impairment`. Left alone these register as gold-only and
extraction-only edges that are really the same claim, so a comparison-time fold is
applied (FOLD below). The graphs themselves are untouched -- this is a view, not a
rewrite, and --no-fold reports the unfolded numbers.

Taxonomic ranks are NEVER folded. A family and a genus inside it are different
claims (Lachnospiraceae down 15 papers / Hungatella up 7 in Parkinson's), so
containment stays as it is in both graphs.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import build_kg
from build_kg import norm_disease, norm_taxon
from build_kg_gold import GOLD_DISEASE_EXTRA, clean_cell, load_titles

HERE = os.path.dirname(os.path.abspath(__file__))

# extraction-graph disease label -> the coarser label the curators use
FOLD = {
    "Cognitive impairment": "Mild cognitive impairment",
    "Neurocognitive impairment": "Mild cognitive impairment",
    "Subjective cognitive decline": "Mild cognitive impairment",
    "Anti-NMDAR encephalitis": "Encephalitis",
    "NMDAR encephalitis": "Encephalitis",
    "Anti-N-methyl-D-aspartate receptor encephalitis": "Encephalitis",
    "Neuroinfection": "Encephalitis",
    "Minimal hepatic encephalopathy": "Encephalopathy",
    "Hepatic encephalopathy": "Encephalopathy",
    "Intracerebral hemorrhage": "Stroke",
    "Hypertensive intracerebral hemorrhage": "Stroke",
    "Hemorrhagic transformation": "Stroke",
    "Poststroke aphasia": "Stroke",
    "Chronic traumatic complete spinal cord injury": "Spinal cord injury",
    "Traumatic thoracic spinal cord injury": "Spinal cord injury",
    "Sporadic Creutzfeldt-Jakob disease": "Creutzfeldt-Jakob disease",
}


def norm_title(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def edge_index(G, fold=True):
    """(taxon_key, disease) -> edge, merging any edges the fold collapses."""
    out = {}
    for e in G["edges"]:
        d = FOLD.get(e["disease"], e["disease"]) if fold else e["disease"]
        k = (e["taxon_key"], d)
        if k in out:                     # two labels folded onto one: pool the votes
            p = out[k]
            p["n_up"] += e["n_up"]
            p["n_down"] += e["n_down"]
            p["n_papers"] += e["n_papers"]
        else:
            out[k] = {"taxon": e["taxon"], "taxon_key": e["taxon_key"], "disease": d,
                      "rank": e["rank"], "resolved": e["resolved"],
                      "n_up": e["n_up"], "n_down": e["n_down"], "n_papers": e["n_papers"]}
    for e in out.values():
        e["direction"] = ("contested" if e["n_up"] and e["n_down"] and e["n_up"] == e["n_down"]
                          else "enriched" if e["n_up"] > e["n_down"] else "depleted")
        e["contested"] = bool(e["n_up"] and e["n_down"])
    return out


# ---------------------------------------------------------------- paper level
def gold_triples(gold_csv, sheet_csv, tax):
    doi2title = load_titles(sheet_csv)
    out = defaultdict(dict)              # norm_title -> {(taxon_key, disease): dir}
    disp = {}
    with open(gold_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        doi = (r.get("DOI") or "").strip()
        fld = (r.get("field") or "").strip().lower()
        if fld not in ("enriched", "depleted"):
            continue
        title = doi2title.get(doi.lower(), ("", ""))[0] or doi
        for part in [p.strip() for p in (r.get("disorder") or "").split(";") if p.strip()]:
            dis, _ = norm_disease(part)
            dis = FOLD.get(dis, dis)
            for raw in clean_cell(r.get("high_confidence_taxa")):
                key, d, _rank, _how = norm_taxon(raw, tax)
                disp.setdefault(key, d)
                out[norm_title(title)][(key, dis)] = fld
    return out, disp


def extraction_triples(path, tax):
    out = defaultdict(dict)
    disp = {}
    for r in json.load(open(path)):
        dis, _ = norm_disease(r.get("predicted_disease") or r.get("disease") or "")
        dis = FOLD.get(dis, dis)
        for direction, col in (("enriched", "predicted_enriched"),
                               ("depleted", "predicted_depleted")):
            for raw in build_kg.parse_taxa(r.get(col)):
                key, d, _rank, _how = norm_taxon(raw, tax)
                disp.setdefault(key, d)
                out[norm_title(r.get("title", ""))][(key, dis)] = direction
    return out, disp


def external_votes(tax):
    """(taxid, our disease label) -> Counter of curated directions.

    Disbiome + Peryton pooled, loaded through validate_external so the join rule
    is the shared one: BOTH SIDES RE-RESOLVED BY NAME through taxonomy.py, never
    on the database's own stored taxid (Disbiome files bare "Prevotella" under
    59823, a species, where the genus is 838).
    """
    import validate_external as V
    recs = []
    try:
        recs += V.load_disbiome()[0]
    except Exception as ex:
        print(f"  (disbiome unavailable: {ex.__class__.__name__})")
    p = V.load_peryton()[0]
    if p:
        recs += p
    votes = defaultdict(Counter)
    for r in recs:
        our = V.DISEASE_MAP.get(r["disease"].lower())
        out = V.OUTCOME.get(r["outcome"].lower())
        if not (our and out and r["microbe"]):
            continue
        tid = tax.resolve(r["microbe"])[0]
        if tid:
            votes[(tid, FOLD.get(our, our))][out] += 1
    return votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-graph", default=os.path.join(HERE, "graph_gold.json"))
    ap.add_argument("--ext-graph", default=os.path.join(HERE, "graph.json"))
    ap.add_argument("--gold-csv", default=os.path.expanduser(
        "~/Downloads/high_confidence - final_constrained_override.csv"))
    ap.add_argument("--sheet", default=os.path.join(
        HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv"))
    ap.add_argument("--extractions", default=os.path.join(HERE, "extractions_corrected.json"))
    ap.add_argument("--no-fold", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", default=os.path.join(HERE, "compare_graphs.json"))
    a = ap.parse_args()

    build_kg.DISEASE_MAP = GOLD_DISEASE_EXTRA + build_kg.DISEASE_MAP
    from taxonomy import Taxonomy
    tax = Taxonomy()
    if not tax.ok:
        print("FATAL: NCBI taxdump missing.")
        sys.exit(1)

    fold = not a.no_fold
    Gg, Ge = json.load(open(a.gold_graph)), json.load(open(a.ext_graph))
    gold, ext = edge_index(Gg, fold), edge_index(Ge, fold)

    both = set(gold) & set(ext)
    gonly, eonly = set(gold) - set(ext), set(ext) - set(gold)
    res = {"fold": fold, "gold_edges": len(gold), "ext_edges": len(ext),
           "both": len(both), "gold_only": len(gonly), "ext_only": len(eonly),
           "jaccard": round(len(both) / len(set(gold) | set(ext)), 3)}

    agree, dis, one_contested = 0, [], 0
    for k in both:
        g, e = gold[k], ext[k]
        if g["contested"] or e["contested"]:
            one_contested += 1
            if g["direction"] == "contested" or e["direction"] == "contested":
                continue
        if g["direction"] == e["direction"]:
            agree += 1
        else:
            dis.append((g, e))
    decisive = agree + len(dis)
    res.update({"decisive": decisive, "agree": agree, "disagree": len(dis),
                "agree_pct": round(100 * agree / max(decisive, 1), 1),
                "edges_with_a_contested_side": one_contested})

    print("=" * 78)
    print(f"EDGE-LEVEL DIFF   (disease folding: {'on' if fold else 'off'})")
    print("=" * 78)
    print(f"  gold edges              : {len(gold)}")
    print(f"  extraction edges        : {len(ext)}")
    print(f"  in both                 : {len(both)}   (Jaccard {res['jaccard']})")
    print(f"  gold only               : {len(gonly)}")
    print(f"  extraction only         : {len(eonly)}")
    print(f"  decisive on both sides  : {decisive}")
    print(f"    same direction        : {agree} ({res['agree_pct']}%)")
    print(f"    OPPOSITE direction    : {len(dis)} ({100-res['agree_pct']:.1f}%)")

    dis.sort(key=lambda x: -(x[0]["n_papers"] + x[1]["n_papers"]))
    print(f"\n  top {min(a.top,len(dis))} direction disagreements (by combined paper support):")
    print(f"    {'taxon':26} {'disease':26} {'gold':>18}  {'extraction':>18}")
    top = []
    for g, e in dis[:a.top]:
        print(f"    {g['taxon'][:25]:26} {g['disease'][:25]:26} "
              f"{g['direction']:>10} ({g['n_papers']:2}p)  {e['direction']:>10} ({e['n_papers']:2}p)")
        top.append({"taxon": g["taxon"], "taxon_key": g["taxon_key"], "disease": g["disease"],
                    "gold": g["direction"], "gold_papers": g["n_papers"],
                    "gold_up": g["n_up"], "gold_down": g["n_down"],
                    "ext": e["direction"], "ext_papers": e["n_papers"],
                    "ext_up": e["n_up"], "ext_down": e["n_down"]})
    res["top_disagreements"] = top

    # which diseases the two graphs do not share at all
    gd, ed = {d for _, d in gold}, {d for _, d in ext}
    res["diseases_gold_only"] = sorted(gd - ed)
    res["diseases_ext_only"] = sorted(ed - gd)
    print(f"\n  diseases only in gold       : {', '.join(res['diseases_gold_only']) or '-'}")
    print(f"  diseases only in extraction : {', '.join(res['diseases_ext_only']) or '-'}")

    # ------------------------------------------------------------ paper level
    gt, gdisp = gold_triples(a.gold_csv, a.sheet, tax)
    et, edisp = extraction_triples(a.extractions, tax)
    disp = {**edisp, **gdisp}
    shared = set(gt) & set(et)
    pagree, pdis, gold_missed, ext_missed = 0, [], 0, 0
    per_pair = Counter()
    for t in shared:
        for k, d in gt[t].items():
            if k not in et[t]:
                ext_missed += 1
            elif et[t][k] == d:
                pagree += 1
            else:
                pdis.append((t, k, d, et[t][k]))
                per_pair[k] += 1
        for k in et[t]:
            if k not in gt[t]:
                gold_missed += 1
    ptot = pagree + len(pdis)
    res.update({"papers_in_both": len(shared),
                "paper_triples_shared": ptot,
                "paper_triples_agree": pagree,
                "paper_triples_flip": len(pdis),
                "paper_flip_pct": round(100 * len(pdis) / max(ptot, 1), 2),
                "gold_taxa_extraction_missed": ext_missed,
                "extraction_taxa_not_in_gold": gold_missed})

    print("\n" + "=" * 78)
    print("PAPER-LEVEL DIFF  (same paper, same taxon, same disease)")
    print("=" * 78)
    print(f"  papers annotated by both   : {len(shared)}")
    print(f"  (taxon,disease) claims both sources make about the same paper: {ptot}")
    print(f"    same direction           : {pagree} ({100-res['paper_flip_pct']:.2f}%)")
    print(f"    OPPOSITE direction       : {len(pdis)} ({res['paper_flip_pct']}%)")
    print(f"  gold claims extraction has no taxon for : {ext_missed}")
    print(f"  extraction claims absent from gold      : {gold_missed}")
    if per_pair:
        print(f"\n  taxon/disease pairs flipped in the most papers:")
        for (k, d), n in per_pair.most_common(10):
            print(f"    {n}x  {disp.get(k,k)[:30]:31} {d}")
        res["paper_flip_hotspots"] = [
            {"taxon": disp.get(k, k), "disease": d, "n_papers": n}
            for (k, d), n in per_pair.most_common(20)]
    res["paper_flips"] = [{"paper": t, "taxon": disp.get(k, k), "disease": d,
                           "gold": gd_, "ext": ed_} for t, (k, d), gd_, ed_ in
                          ((t, k, g, e) for t, k, g, e in pdis)][:400]

    # Same thing keyed on taxon alone. 8 of the 244 shared papers get disjoint
    # disease labels from the two sources (gold says Stroke where the extractor
    # says CADASIL, etc.), which would otherwise book every taxon in those papers
    # as missed by both sides. Dropping the disease from the key removes that.
    tagree, tflip, tg_only, te_only = 0, 0, 0, 0
    for t in shared:
        g = {k[0]: v for k, v in gt[t].items()}
        e = {k[0]: v for k, v in et[t].items()}
        for k, v in g.items():
            if k not in e:
                tg_only += 1
            elif e[k] == v:
                tagree += 1
            else:
                tflip += 1
        te_only += sum(1 for k in e if k not in g)
    res.update({"taxon_only_agree": tagree, "taxon_only_flip": tflip,
                "taxon_only_gold_missed_by_ext": tg_only,
                "taxon_only_ext_not_in_gold": te_only,
                "taxon_only_jaccard": round((tagree + tflip) /
                                            max(tagree + tflip + tg_only + te_only, 1), 3)})
    print("\n  ignoring the disease label (8/244 papers get different disease labels):")
    print(f"    shared taxon mentions    : {tagree + tflip}  "
          f"(same dir {tagree}, flipped {tflip})")
    print(f"    gold only / extraction only : {tg_only} / {te_only}")
    print(f"    taxon-set Jaccard per paper : {res['taxon_only_jaccard']}")

    # --------------------------------------------- external adjudication
    # For the (taxon,disease) pairs where the two graphs point OPPOSITE ways, ask
    # a curated database which side it agrees with. This is the only umpire
    # available -- neither graph can adjudicate itself.
    ext_votes = external_votes(tax)
    if ext_votes:
        print("\n" + "=" * 78)
        print("WHO IS RIGHT? curated databases adjudicating the direction flips")
        print("=" * 78)
        adj = {"gold": 0, "ext": 0, "n": 0, "rows": []}
        for g, e in dis:
            v = ext_votes.get((g["taxon_key"].split(":", 1)[-1], g["disease"]))
            if not v:
                continue
            rdir = ("enriched" if v["enriched"] > v["depleted"]
                    else "depleted" if v["depleted"] > v["enriched"] else None)
            if not rdir:
                continue
            side = "gold" if rdir == g["direction"] else "ext"
            adj[side] += 1
            adj["n"] += 1
            adj["rows"].append({"taxon": g["taxon"], "disease": g["disease"],
                                "gold": g["direction"], "ext": e["direction"],
                                "curated": rdir, "supports": side,
                                "curated_n": v[rdir]})
        print(f"  flips a curated DB can rule on : {adj['n']} of {len(dis)}")
        if adj["n"]:
            print(f"    curated DB sides with GOLD       : {adj['gold']} "
                  f"({100*adj['gold']/adj['n']:.0f}%)")
            print(f"    curated DB sides with EXTRACTION : {adj['ext']} "
                  f"({100*adj['ext']/adj['n']:.0f}%)")
            for r in sorted(adj["rows"], key=lambda r: -r["curated_n"])[:15]:
                print(f"    {r['taxon'][:25]:26} {r['disease'][:24]:25} "
                      f"gold={r['gold']:8} ext={r['ext']:8} curated={r['curated']:8}"
                      f" ({r['curated_n']}) -> {r['supports']}")
        res["external_adjudication"] = adj

    json.dump(res, open(a.json, "w"), indent=2)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
