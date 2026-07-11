#!/usr/bin/env python3
"""Idea-3 / LLM-judge evaluation. Score the ORIGINAL extraction against the ORIGINAL (incomplete)
gold with the metric, then let a judge adjudicate each 'false positive': is it actually a real
significant finding? FPs the judge confirms are recovered as TP (they were penalised only because
the gold was incomplete). Here the judge oracle = the Fable thorough benchmark (an LLM adjudication
already produced); in production this is a per-FP LLM-judge call over the paper text.
Reports metric-F1 vs judge-adjusted-F1 against the ORIGINAL gold."""
import common, csv, os
from taxonomy_match import TaxResolver, match_taxa_lca

c = common.read_cache("original")
if c:
    papers = {p["idx"]: p for p in common.load_papers()}
    R = TaxResolver()
    names = []
    for p in papers.values():
        for col in ("orig_enriched", "orig_depleted", "fable_enriched", "fable_depleted"):
            names += common.parse_taxa(p[col])
    for v in c.values():
        names += [x.lower() for x in v["taxa_enriched"] + v["taxa_depleted"]]
    R.warm(names)

    def unmatched_preds(pred, exp):
        """return preds that DON'T match exp (the metric FPs)."""
        fps = []
        for t in pred:
            tp, fp, fn = match_taxa_lca([t], exp, R)
            if fp:  # this single pred matched nothing in exp
                fps.append(t)
        return fps

    mTP = mFP = mFN = 0        # metric, vs original gold
    jTP = jFP = jFN = 0        # judge-adjusted, vs original gold
    recovered = 0
    for k, v in c.items():
        p = papers[int(k)]
        for pk, ok, fk in (("taxa_enriched", "orig_enriched", "fable_enriched"),
                           ("taxa_depleted", "orig_depleted", "fable_depleted")):
            pred = [x.lower() for x in v[pk]]
            og = common.parse_taxa(p[ok]); fg = common.parse_taxa(p[fk])
            tp, fp, fn = match_taxa_lca(pred, og, R)
            mTP += tp; mFP += fp; mFN += fn
            # judge each metric-FP against the oracle (Fable benchmark)
            fps = unmatched_preds(pred, og)
            conf = 0
            for t in fps:
                jtp, jfp, jfn = match_taxa_lca([t], fg, R)
                if jtp:            # judge confirms it's a real finding
                    conf += 1
            recovered += conf
            jTP += tp + conf; jFP += fp - conf; jFN += fn

    def prf(tp, fp, fn):
        P = tp / (tp + fp) if tp + fp else 0; Rc = tp / (tp + fn) if tp + fn else 0
        return P, Rc, (2 * P * Rc / (P + Rc) if P + Rc else 0)
    mP, mR, mF = prf(mTP, mFP, mFN); jP, jR, jF = prf(jTP, jFP, jFN)
    os.makedirs(common.RESULTS, exist_ok=True)
    exists = os.path.exists(common.LEDGER)
    with open(common.LEDGER, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["method", "gold", "n", "precision", "recall", "f1", "TP", "FP", "FN", "note"])
        w.writerow(["4_judge_metric", "orig", len(c), round(mP,3), round(mR,3), round(mF,3), mTP, mFP, mFN, "metric vs original gold"])
        w.writerow(["4_judge_adjusted", "orig", len(c), round(jP,3), round(jR,3), round(jF,3), jTP, jFP, jFN, f"LLM-judge recovered {recovered} FPs"])
    print(f"[4_judge] vs ORIGINAL gold:")
    print(f"    metric        P={mP:.3f} R={mR:.3f} F1={mF:.3f}  (TP={mTP} FP={mFP} FN={mFN})")
    print(f"    judge-adjusted P={jP:.3f} R={jR:.3f} F1={jF:.3f}  (recovered {recovered} 'FPs' that are real findings)")
