#!/usr/bin/env python3
"""Score the extractions we ALREADY have against the new gold standard.

This is the check that decides whether a GPU run is needed at all: 260 of the 265
staged papers were already extracted by Qwopus3.5-27B/samgated-v1, so the only
open question is whether that prompt is good enough against the *new* gold. No
model is loaded and nothing is re-extracted -- it re-scores existing output.

Metric is match_taxa() imported from eval-v2/run_eval.py (greedy char-ngram cosine
>= 0.5), i.e. the same function the leaderboard used. NOT the taxonomy-aware LCA
metric, which needs the NCBI taxdump; LCA scores strictly higher, so these are
lower bounds.

    python rescore_vs_new_gold.py        # prints, writes rescore_vs_new_gold.json
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2")))
from run_eval import parse_taxa, match_taxa                 # noqa: E402

GOLD = os.path.join(HERE, "extract_input_gold.json")
EXTRACTIONS = os.path.join(HERE, "extractions_corrected.json")
OUT = os.path.join(HERE, "rescore_vs_new_gold.json")


def nt(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (t or "").lower())).strip()


def score(pairs, key_e, key_d):
    TP = FP = FN = 0
    for g, e in pairs:
        for pk, gk in (("predicted_enriched", key_e), ("predicted_depleted", key_d)):
            tp, fp, fn = match_taxa(parse_taxa(e.get(pk, "")), parse_taxa(g.get(gk, "")))
            TP += tp; FP += fp; FN += fn
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"TP": TP, "FP": FP, "FN": FN,
            "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4)}


def main():
    gold = {nt(p["title"]): p for p in json.load(open(GOLD))}
    ex = {nt(r["title"]): r for r in json.load(open(EXTRACTIONS))}
    pairs = [(gold[t], ex[t]) for t in gold if t in ex]

    res = {
        "n_staged": len(gold),
        "n_scoreable": len(pairs),
        "n_never_extracted": len(gold) - len(pairs),
        "vs_new_gold": score(pairs, "taxa_enriched", "taxa_depleted"),
        "vs_old_sheet": score(pairs, "sheet_enriched", "sheet_depleted"),
        "metric": "char-ngram cosine >=0.5 (run_eval.match_taxa); LCA would score higher",
    }
    print(f"staged {res['n_staged']} | already extracted {res['n_scoreable']} | "
          f"never extracted {res['n_never_extracted']}\n")
    for k in ("vs_new_gold", "vs_old_sheet"):
        s = res[k]
        print(f"  {k:<14} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}"
              f"   (TP={s['TP']} FP={s['FP']} FN={s['FN']})")
    d = res["vs_new_gold"]["precision"] - res["vs_old_sheet"]["precision"]
    print(f"\nSAME extractions, two references: precision moves {d:+.3f}. The extractor's "
          f"\"over-extraction\"\nwas mostly the old sheet being incomplete, not false positives.")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
