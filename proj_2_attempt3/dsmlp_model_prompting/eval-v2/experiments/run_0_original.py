#!/usr/bin/env python3
"""Baseline: the original single-shot samgated-v1 extraction. Scored as-is."""
import common
c = common.read_cache("original")
if c:
    preds = [{"idx": int(k), "pred_enriched": v["taxa_enriched"], "pred_depleted": v["taxa_depleted"]}
             for k, v in c.items()]
    common.save_and_report("0_original", preds, note="single-shot samgated-v1")
