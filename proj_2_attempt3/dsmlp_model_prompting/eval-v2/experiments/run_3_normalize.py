#!/usr/bin/env python3
"""Idea-3 / normalization: build on the original run, then canonicalize every taxon to its NCBI
taxid and COLLAPSE names that resolve to the same node (Bacteroides vulgatus == Phocaeicola
vulgatus). Demonstrates the KG node-dedup value. Reports raw-vs-canonical node counts + score."""
import common
from taxonomy_match import TaxResolver
c = common.read_cache("original")
if c:
    R = TaxResolver()
    allnames = []
    for v in c.values():
        allnames += [x.lower() for x in v["taxa_enriched"] + v["taxa_depleted"]]
    R.warm(allnames)

    def canon_dedup(names):
        seen, out, collapsed = {}, [], 0
        for n in names:
            tid = R.cache.get(n.lower(), ("", set()))[0]
            key = tid if tid else n.lower()   # unresolved -> keep by name
            if key in seen:
                collapsed += 1
                continue
            seen[key] = n; out.append(n)
        return out, collapsed

    preds = []; raw = 0; nodes = 0; total_collapsed = 0
    for k, v in c.items():
        e, ce = canon_dedup(v["taxa_enriched"]); d, cd = canon_dedup(v["taxa_depleted"])
        raw += len(v["taxa_enriched"]) + len(v["taxa_depleted"])
        nodes += len(e) + len(d); total_collapsed += ce + cd
        preds.append({"idx": int(k), "pred_enriched": e, "pred_depleted": d})
    common.save_and_report("3_normalize", preds,
                           note=f"{raw} raw -> {nodes} canonical nodes (collapsed {total_collapsed})")
