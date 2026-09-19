#!/usr/bin/env python3
"""Build adjudication packets for the 25 free-text ("fallback") disease nodes.

`norm_disease` in build_kg.py tries 17 DISEASE_MAP regexes and, on a miss, falls
through to the extractor's own `predicted_disease` string as a node label. 25 of
40 disease nodes (399 edges, 19.9% of the graph) are such strings; 11 of them
carry no MONDO id either. Nobody has ever checked those labels against the
cohort the source paper actually studied -- the one node that was checked
(`Neurocognitive impairment`, 2026-09-15) turned out CORRECT, but that is n=1.

This script emits, per fallback node, the source papers with the sentences from
`relation_sentences_clean.json` most likely to name the study contrast, plus the
human datasheet's own label for that paper. Judgement is done elsewhere; this is
a packet builder only.

Output: disease_label_packets.json
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).parent

CONTRAST = re.compile(
    r"\b(compar|versus|vs\.?|than|control|healthy|HC\b|patients?|participants?|"
    r"subjects?|cohort|group|case[- ]control|enrolled|recruit)", re.I)

def main():
    g = json.load(open(HERE / "graph.json"))
    prov = json.load(open(HERE / "disease_node_provenance.json"))
    rs = json.load(open(HERE / "relation_sentences_clean.json"))["papers"]

    fallback = {n["disease"]: n for n in prov["nodes"] if n["origin"] != "DISEASE_MAP"}

    # disease -> papers, from the edges (authoritative: what actually backs the node)
    by_disease = {}
    for e in g["edges"]:
        if e["disease"] in fallback:
            by_disease.setdefault(e["disease"], set()).update(e["papers"])

    packets = []
    unmatched = []
    for dis, meta in sorted(fallback.items(), key=lambda kv: -kv[1]["n_edges"]):
        papers = sorted(by_disease.get(dis, ()))
        pks = []
        for t in papers:
            rec = rs.get(t)
            if rec is None:
                # titles in graph.json may differ in trailing punctuation
                cand = [k for k in rs if k.rstrip(".").lower() == t.rstrip(".").lower()]
                rec = rs[cand[0]] if cand else None
            if rec is None:
                unmatched.append((dis, t))
                pks.append({"title": t, "datasheet_disease": None, "sentences": []})
                continue
            sents = [k["s"] for k in rec["kept"]]
            picked = [s for s in sents if CONTRAST.search(s)][:14]
            if len(picked) < 6:
                picked = (picked + [s for s in sents if s not in picked])[:10]
            pks.append({
                "title": t,
                "datasheet_disease": rec.get("disease"),
                "link": rec.get("link"),
                "n_sentences_available": len(sents),
                "sentences": picked,
            })
        packets.append({
            "node_label": dis,
            "mondo": meta["mondo"],
            "n_papers_graph": meta["n_papers"],
            "n_edges": meta["n_edges"],
            "papers": pks,
        })

    out = {"n_nodes": len(packets),
           "n_edges_covered": sum(p["n_edges"] for p in packets),
           "unmatched_titles": unmatched,
           "packets": packets}
    json.dump(out, open(HERE / "disease_label_packets.json", "w"), indent=1)
    print(f"nodes={out['n_nodes']} edges={out['n_edges_covered']} "
          f"papers={sum(len(p['papers']) for p in packets)} unmatched={len(unmatched)}")
    for d, t in unmatched:
        print("  UNMATCHED", d, "|", t[:70])

if __name__ == "__main__":
    main()
