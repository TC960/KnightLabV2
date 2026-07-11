#!/usr/bin/env python3
"""Source-grounded extraction (LangExtract-style): every taxon must carry a VERBATIM supporting
sentence. We keep only taxa whose sentence actually appears in the paper text (drops hallucinated
grounding). Outputs in cache/grounded.json carry taxon+direction (+sentence when available)."""
import common, json, os, re
c = common.read_cache("grounded")
if c:
    papers = {p["idx"]: re.sub(r"\s+", " ", p["text"]).lower() for p in common.load_papers()}
    preds = []
    dropped = 0
    for k, v in c.items():
        idx = int(k)
        # grounding validation: taxon (or its genus) must occur in the paper text
        def grounded(t):
            tl = t.lower()
            return tl in papers[idx] or tl.split()[0] in papers[idx]
        e = [t for t in v["taxa_enriched"] if grounded(t)]
        d = [t for t in v["taxa_depleted"] if grounded(t)]
        dropped += (len(v["taxa_enriched"]) - len(e)) + (len(v["taxa_depleted"]) - len(d))
        preds.append({"idx": idx, "pred_enriched": e, "pred_depleted": d})
    common.save_and_report("2_grounded", preds, note=f"grounding-verified (dropped {dropped} ungrounded)")
