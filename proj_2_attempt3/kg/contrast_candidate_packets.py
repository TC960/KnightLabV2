#!/usr/bin/env python3
"""Sample the 114 CANDIDATE observations for adjudication, to turn an upper bound
into an estimate -- the same move `FINDINGS_edge_recall.md` made for recall.

`contrast_edge_probe.py` flags 114 of 3,077 observations as having no VISIBLE
healthy-control contrast in any sentence naming their taxon. That is an upper bound
on within-paper contamination: the regex can miss a control contrast stated in a
sentence the provenance filter never kept, and it can fire on a sentence that
mentions a subgroup incidentally.

Sampling is by PAPER, not by observation. 114 candidates come from only 47 papers
and one paper contributes 10, so an observation-level sample would over-weight a
handful of papers and the resulting CI would be wrong. A paper-level cluster sample
keeps the unit of independence right, exactly as the edge-recall audit did.

Each packet gives the adjudicator every sentence in that paper that names the taxon,
plus the paper's other sentences for context, and asks one question: does this
paper report that taxon as differing between a diseased group and a healthy/normal
control group? Nothing about the graph, the disease node, or the probe's verdict is
shown.

Writes contrast_candidate_packets.json (+ _ccbatch/batch*.json).
"""
import json, random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260919
N_PAPERS = 24          # matches the edge-recall audit's sample size
NBATCH = 6

def main():
    rows = json.load(open(HERE / "contrast_edge_probe.json"))["rows"]
    rs = json.load(open(HERE / "relation_sentences_clean.json"))["papers"]
    cands = [r for r in rows if r["verdict"] == "CANDIDATE"]
    by_paper = defaultdict(list)
    for r in cands:
        by_paper[r["paper"]].append(r)

    papers = sorted(by_paper)
    rng = random.Random(SEED)
    rng.shuffle(papers)
    picked = sorted(papers[:N_PAPERS])

    packets = []
    for t in picked:
        rec = rs[t]
        sents = [k["s"] for k in rec["kept"]]
        idx = defaultdict(list)
        for k in rec["kept"]:
            for surf, tid, name, rank in k["taxa"]:
                for key in (str(tid), str(name).lower(), str(surf).lower()):
                    idx[key].append(k["s"])
        for r in by_paper[t]:
            tid = (r.get("taxon_key") or "").split(":")[-1]
            own = idx.get(tid) or idx.get(str(r["taxon"]).lower()) or []
            others = [s for s in sents if s not in own][:8]
            packets.append({
                "id": f"{picked.index(t)}:{r['taxon']}",
                "title": t, "taxon": r["taxon"],
                "sentences_naming_taxon": own[:8],
                "other_sentences_from_the_paper": others,
            })

    json.dump({"n_papers_in_pool": len(papers), "n_candidates_in_pool": len(cands),
               "n_papers_sampled": len(picked), "n_packets": len(packets),
               "seed": SEED, "packets": packets},
              open(HERE / "contrast_candidate_packets.json", "w"), indent=1)

    out = HERE / "_ccbatch"; out.mkdir(exist_ok=True)
    for i in range(NBATCH):
        b = packets[i::NBATCH]
        json.dump({"observations": b}, open(out / f"batch{i+1}.json", "w"), indent=1)
    print(f"pool: {len(cands)} candidates / {len(papers)} papers; "
          f"sampled {len(picked)} papers -> {len(packets)} observations; "
          f"{NBATCH} batches")

if __name__ == "__main__":
    main()
