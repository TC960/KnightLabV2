#!/usr/bin/env python3
"""Build per-paper packets for a corpus-wide STUDY-CONTRAST census.

The extraction prompt (`eval-v2/run_eval.py`, samgated-v1) admits only
**disease vs healthy control** comparisons. Every fidelity instrument in this
project scores whether an edge's taxon and direction are right; none has ever
asked whether the COMPARISON the edge came from was in scope. A taxon that is
genuinely higher in ICH-survivors than in ICH-deceased is a correct reading of
the paper and still not a disease-vs-control edge.

One packet per contributing paper: the disease node(s) its edges back, whether
that node came from DISEASE_MAP or the free-text fallback, and the sentences
from `relation_sentences_clean.json` most likely to name the study's arms.

Output: contrast_packets.json  (+ _ctbatch/batch*.json)
"""
import json, re
from pathlib import Path

HERE = Path(__file__).parent
NBATCH = 12

CONTRAST = re.compile(
    r"\b(compar|versus|vs\.?|than|control|healthy|\bHCs?\b|patients?|participants?|"
    r"subjects?|cohort|group|case[- ]control|enrolled|recruit|relative to)", re.I)

def main():
    g = json.load(open(HERE / "graph.json"))
    prov = {n["disease"]: n for n in json.load(open(HERE / "disease_node_provenance.json"))["nodes"]}
    rs = json.load(open(HERE / "relation_sentences_clean.json"))["papers"]
    lower = {k.rstrip(".").lower(): k for k in rs}

    per_paper = {}
    for e in g["edges"]:
        for t in e["papers"]:
            d = per_paper.setdefault(t, {"diseases": {}, "n_edges": 0})
            d["diseases"][e["disease"]] = d["diseases"].get(e["disease"], 0) + 1
            d["n_edges"] += 1

    packets, missing = [], []
    for t, info in sorted(per_paper.items()):
        rec = rs.get(t) or rs.get(lower.get(t.rstrip(".").lower(), ""))
        if rec is None:
            missing.append(t); continue
        sents = [k["s"] for k in rec["kept"]]
        picked = [s for s in sents if CONTRAST.search(s)][:14]
        if len(picked) < 6:
            picked = (picked + [s for s in sents if s not in picked])[:10]
        dis = sorted(info["diseases"], key=lambda d: -info["diseases"][d])
        packets.append({
            "title": t,
            "diseases": dis,
            "origin": ["DISEASE_MAP" if prov.get(d, {}).get("origin") == "DISEASE_MAP"
                       else "FALLBACK" for d in dis],
            "n_edges": info["n_edges"],
            "datasheet_disease": rec.get("disease"),
            "sentences": picked,
        })

    json.dump({"n_papers": len(packets), "missing": missing, "packets": packets},
              open(HERE / "contrast_packets.json", "w"), indent=1)

    out = HERE / "_ctbatch"; out.mkdir(exist_ok=True)
    packets.sort(key=lambda p: -len(p["sentences"]))
    for i in range(NBATCH):
        b = packets[i::NBATCH]
        json.dump({"packets": b}, open(out / f"batch{i+1}.json", "w"), indent=1)
    print(f"papers={len(packets)} missing={len(missing)} "
          f"edges={sum(p['n_edges'] for p in packets)} batches={NBATCH} "
          f"per_batch~{len(packets)//NBATCH}")
    for m in missing[:10]: print("  MISSING", m[:80])

if __name__ == "__main__":
    main()
