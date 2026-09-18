#!/usr/bin/env python3
"""Score the three prompt variants against the gold, paired by paper.

Applies the decision rule fixed in PROMPT_EXPERIMENT.md BEFORE the run:

    promote a variant only if it beats baseline on LCA F1 with a paired 95% CI
    excluding zero AND keeps precision >= 0.65

Paired is the operative word. All three variants saw the identical 80 papers in
one process on one GPU, so per-paper differences are paired and an unpaired test
would throw away most of the power. Wilcoxon signed-rank over per-paper F1, plus
a bootstrap CI on the mean difference.

Uses the SAME matcher as the headline number, including the guard that stops two
predictions claiming the same gold taxon (without it every score inflates).

    python score_prompt_exp.py
    python score_prompt_exp.py --results-dir gpu_results/extract_out
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = "/Users/mohak/Downloads/high_confidence - final_constrained_override.csv"
SHEET = os.path.join(HERE, "Microbiota Signatures Neurological Disorders "
                           "Sheet 2 - Main Datasheet.csv")
DEFAULT_RESULTS = os.path.join(HERE, "gpu_results", "extract_out")
OUT = os.path.join(HERE, "prompt_exp_results.json")
MIN_PRECISION = 0.65          # pre-registered floor


def parse_taxa(s):
    if not isinstance(s, str):
        return []
    s = re.sub(r"\(?p\s*[<>=]\s*0?\.\d+\)?", "", s, flags=re.I)
    return [t.strip() for t in re.split(r"[;,]", s) if len(t.strip()) > 2]


def doi_of(s):
    m = re.search(r"10\.\d{4,9}/[^\s\"<>,;\]]+", s or "")
    return m.group(0).rstrip(".").lower() if m else None


def load_gold():
    g = {}
    for x in csv.DictReader(open(GOLD, encoding="utf-8-sig")):
        d = x["DOI"].strip().lower()
        g.setdefault(d, {"e": [], "d": []})
        g[d]["e" if x["field"] == "Enriched" else "d"] = parse_taxa(
            x["high_confidence_taxa"])
    return g


def match(pred, exp, resolver=None):
    """Greedy char-ngram >= 0.5, optionally falling back to nested NCBI lineages.

    The `j not in used` guard is load-bearing: without it two predictions whose
    best match is the same gold taxon each score a true positive.
    """
    if not pred or not exp:
        return 0
    from sklearn.feature_extraction.text import TfidfVectorizer
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(pred + exp)
    S = (V.transform(pred) @ V.transform(exp).T).toarray()
    used, hits = set(), 0
    for i in range(len(pred)):
        j = int(np.argmax(S[i]))
        if S[i][j] >= 0.5 and j not in used:
            used.add(j)
            hits += 1
            continue
        if resolver is not None:
            for k in range(len(exp)):
                if k in used:
                    continue
                try:
                    if resolver.nested(pred[i], exp[k]):
                        used.add(k)
                        hits += 1
                        break
                except Exception:
                    pass
    return hits


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--no-lca", action="store_true")
    a = ap.parse_args()

    gold = load_gold()
    d2t = {}
    for x in csv.DictReader(open(SHEET, encoding="utf-8-sig")):
        d = doi_of((x.get("DOI") or "").strip().lower()) or ""
        t = (x.get("Title") or "").strip()
        if d and t:
            d2t[d] = t.lower()
    t2d = {v: k for k, v in d2t.items()}

    resolver = None
    if not a.no_lca:
        try:
            sys.path.insert(0, HERE)
            from taxonomy_cache import load_taxonomy
            tx = load_taxonomy()
            resolver = tx if getattr(tx, "ok", False) else None
        except Exception as e:
            print(f"(taxonomy unavailable: {type(e).__name__}; char metric only)")

    variants = {}
    for f in sorted(os.listdir(a.results_dir)):
        m = re.match(r"variant_([ABC])\.checkpoint\.jsonl$", f)
        if not m:
            continue
        rows = []
        for line in open(os.path.join(a.results_dir, f)):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        variants[m.group(1)] = rows
    if not variants:
        raise SystemExit(f"no variant_*.checkpoint.jsonl under {a.results_dir}")

    # score per paper so the comparison can be paired
    per_paper = {}
    agg = {}
    for vid, rows in sorted(variants.items()):
        TP = FP = FN = 0
        pp = {}
        nerr = 0
        for r in rows:
            d = (r.get("doi") or "").strip().lower() or t2d.get(
                (r.get("title") or "").strip().lower(), "")
            g = gold.get(d)
            if not g or not (g["e"] or g["d"]):
                continue
            nerr += bool(r.get("parse_error"))
            tp = fp = fn = 0
            for fld, gk in (("predicted_enriched", "e"), ("predicted_depleted", "d")):
                p_, e_ = parse_taxa(r.get(fld)), g[gk]
                if not p_ and not e_:
                    continue
                h = match(p_, e_, resolver)
                tp += h
                fp += len(p_) - h
                fn += len(e_) - h
            TP += tp
            FP += fp
            FN += fn
            pp[d] = prf(tp, fp, fn)[2]
        per_paper[vid] = pp
        P, R, F = prf(TP, FP, FN)
        agg[vid] = {"n_papers": len(pp), "TP": TP, "FP": FP, "FN": FN,
                    "precision": round(P, 4), "recall": round(R, 4),
                    "f1": round(F, 4), "parse_errors": nerr}

    print(f"metric: {'LCA (taxonomy-aware)' if resolver else 'char-ngram only'}\n")
    print(f"{'variant':<9}{'n':>5}{'P':>9}{'R':>9}{'F1':>9}{'parse_err':>11}")
    print("-" * 52)
    names = {"A": "A baseline", "B": "B softgate", "C": "C tables"}
    for vid in sorted(agg):
        v = agg[vid]
        print(f"{names.get(vid, vid):<9}{v['n_papers']:>5}{v['precision']:>9.4f}"
              f"{v['recall']:>9.4f}{v['f1']:>9.4f}{v['parse_errors']:>11}")

    # ---- paired comparison against baseline -------------------------------
    print("\npaired against baseline A (same papers, one run, one GPU):")
    verdicts = {}
    for vid in sorted(agg):
        if vid == "A":
            continue
        common = sorted(set(per_paper["A"]) & set(per_paper[vid]))
        d = np.array([per_paper[vid][k] - per_paper["A"][k] for k in common])
        mean = float(d.mean())
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(d, len(d), replace=True).mean()
                         for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        try:
            from scipy.stats import wilcoxon
            nz = d[d != 0]
            w = wilcoxon(nz).pvalue if len(nz) else 1.0
        except Exception:
            w = float("nan")
        beats = lo > 0
        prec_ok = agg[vid]["precision"] >= MIN_PRECISION
        verdicts[vid] = "PROMOTE" if (beats and prec_ok) else "keep baseline"
        print(f"  {names.get(vid, vid)}  n={len(common)}  "
              f"mean dF1 {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"wilcoxon p={w:.3f}")
        print(f"      CI excludes 0: {beats} | precision {agg[vid]['precision']:.3f} "
              f">= {MIN_PRECISION}: {prec_ok}  ->  {verdicts[vid]}")

    json.dump({"metric": "lca" if resolver else "char", "aggregate": agg,
               "verdicts": verdicts,
               "decision_rule": f"paired 95% CI on dF1 excludes 0 AND precision >= {MIN_PRECISION}"},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
