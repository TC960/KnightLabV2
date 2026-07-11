#!/usr/bin/env python3
"""RELATE (arXiv 2509.19057): retrieve recall-biased candidates (common.retrieve_candidates,
deterministic) -> LLM re-ranks each with an explicit REJECT option -> keep survivors.
Stage-1 candidates were generated in common; stage-2 rerank outputs are in cache/relate.json."""
import common
c = common.read_cache("relate")
if c:
    # report retrieval recall for context
    papers = common.load_papers()
    tot_cand = sum(len(common.retrieve_candidates(p["text"])[0]) for p in papers)
    kept = sum(len(v["taxa_enriched"]) + len(v["taxa_depleted"]) for v in c.values())
    preds = [{"idx": int(k), "pred_enriched": v["taxa_enriched"], "pred_depleted": v["taxa_depleted"]}
             for k, v in c.items()]
    common.save_and_report("1_relate", preds, note=f"retrieve {tot_cand} cand -> rerank kept {kept}")
