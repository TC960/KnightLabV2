#!/usr/bin/env python3
"""Score the full model x method matrix and print P/R/F1 vs both golds.
Methods: original (samgated single-shot), relate, grounded, normalize, judge.
Models:  the 3 GGUF models (qwopus3.5, qwythos-9b, qwen2.5) + opus (reference, ignore).
Reads GGUF outputs from cache/{relate,grounded}__<model>.json and originals from ../results/."""
import json, os, re, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); EVAL = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, EVAL); sys.path.insert(0, "/tmp/pylibs")
import common
from taxonomy_match import TaxResolver, match_taxa_lca

TS = json.load(open(os.path.join(EVAL, "..", "..", "EmilySong_GoldStandardPaper", "test_set_v2.json")))
FG = json.load(open(os.path.join(EVAL, "results", "opus_gold_15.json")))
N = len(TS)
def parse(v): return common.parse_taxa(v)

# gold per idx
ORIG = {i: (parse(TS[i].get("taxa_enriched")), parse(TS[i].get("taxa_depleted"))) for i in range(N)}
OPUS = {i: (parse(FG[i]["taxa_enriched"]), parse(FG[i]["taxa_depleted"])) for i in range(N)}
TEXT = {i: re.sub(r"\s+", " ", TS[i]["text"]).lower() for i in range(N)}

MODEL_FILES = {
    "qwopus3.5-27b-v3": "qwopus3.5-27b-v3__q4km__samgated-v1__testv2.json",
    "qwythos-9b": "qwythos-9b__q8__samgated-v1__testv2.json",
    "qwen2.5-32b-instruct": "qwen2.5-32b-instruct__q4km__samgated-v1__testv2.json",
}
OPUS_PRED = None  # loaded lazily if present


def original_pred(model):
    if model == "opus":
        # earlier Opus 4.8 single-shot (all 15), if cached in scratch; else skip
        return None
    f = os.path.join(EVAL, "results", MODEL_FILES[model])
    if not os.path.exists(f): return None
    d = json.load(open(f)); by_title = {r["title"]: r for r in d}
    out = {}
    for i in range(N):
        r = by_title.get(TS[i]["title"])
        if r: out[i] = (parse(r.get("predicted_enriched")), parse(r.get("predicted_depleted")))
    return out


def cache_pred(method, model):
    f = os.path.join(HERE, "cache", f"{method}__{model}.json")
    if not os.path.exists(f): return None
    d = json.load(open(f))
    return {int(k): ([x for x in v["taxa_enriched"]], [x for x in v["taxa_depleted"]]) for k, v in d.items()}


def grounded_validate(pred):
    """drop taxa not present (name or genus) in the paper text."""
    out = {}
    for i, (e, d) in pred.items():
        def ok(t): tl = t.lower(); return tl in TEXT[i] or tl.split()[0] in TEXT[i]
        out[i] = ([t for t in e if ok(t)], [t for t in d if ok(t)])
    return out


_R = None
def resolver(all_pred):
    global _R
    _R = TaxResolver()
    names = []
    for pred in all_pred:
        if not pred: continue
        for e, d in pred.values(): names += [x.lower() for x in e + d]
    for g in (ORIG, OPUS):
        for e, d in g.values(): names += e + d
    _R.warm(names); return _R


def score(pred, gold):
    TP = FP = FN = 0
    for i in range(N):
        if pred is None or i not in pred:
            e, d = [], []
        else:
            e, d = pred[i]
        ge, gd = gold[i]
        for pp, gg in (([x.lower() for x in e], ge), ([x.lower() for x in d], gd)):
            tp, fp, fn = match_taxa_lca(pp, gg, _R); TP += tp; FP += fp; FN += fn
    P = TP/(TP+FP) if TP+FP else 0; R = TP/(TP+FN) if TP+FN else 0
    return P, R, (2*P*R/(P+R) if P+R else 0), TP, FP, FN


def normalize_pred(pred):
    out = {}
    for i, (e, d) in pred.items():
        def dedup(names):
            seen, res = set(), []
            for n in names:
                tid = _R.cache.get(n.lower(), ("", set()))[0]; k = tid or n.lower()
                if k in seen: continue
                seen.add(k); res.append(n)
            return res
        out[i] = (dedup(e), dedup(d))
    return out


def judge_pred(pred):
    """recover metric-FPs (vs orig gold) that the oracle (opus gold) confirms real -> counts as TP."""
    TP = FP = FN = 0; rec = 0
    for i in range(N):
        e, d = pred[i] if pred and i in pred else ([], [])
        for pp, og, fg in (([x.lower() for x in e], ORIG[i][0], OPUS[i][0]),
                           ([x.lower() for x in d], ORIG[i][1], OPUS[i][1])):
            tp, fp, fn = match_taxa_lca(pp, og, _R); TP += tp; FN += fn
            for t in pp:
                if match_taxa_lca([t], og, _R)[1]:   # metric-FP vs orig
                    if match_taxa_lca([t], fg, _R)[0]:  # oracle confirms real
                        rec += 1; TP += 1
                    else:
                        FP += 1
    P = TP/(TP+FP) if TP+FP else 0; R = TP/(TP+FN) if TP+FN else 0
    return P, R, (2*P*R/(P+R) if P+R else 0), TP, FP, FN, rec


def main():
    models = ["qwopus3.5-27b-v3", "qwythos-9b", "qwen2.5-32b-instruct"]
    preds = {}
    for m in models:
        preds[(m, "original")] = original_pred(m)
        preds[(m, "relate")] = cache_pred("relate", m)
        gp = cache_pred("grounded", m)
        preds[(m, "grounded")] = grounded_validate(gp) if gp else None
    resolver([p for p in preds.values() if p])
    rows = []
    for m in models:
        orig = preds[(m, "original")]
        for method in ["original", "relate", "grounded", "normalize", "judge"]:
            if method == "normalize":
                p = normalize_pred(orig) if orig else None
                for gname, gold in (("orig", ORIG), ("opus", OPUS)):
                    if p is None: continue
                    P, R, F, TP, FP, FN = score(p, gold)
                    rows.append((m, method, gname, P, R, F, TP, FP, FN, ""))
                continue
            if method == "judge":
                if orig is None: continue
                P, R, F, TP, FP, FN, rec = judge_pred(orig)
                rows.append((m, method, "orig*", P, R, F, TP, FP, FN, f"recovered {rec}"))
                continue
            p = preds[(m, method)]
            for gname, gold in (("orig", ORIG), ("opus", OPUS)):
                if p is None:
                    rows.append((m, method, gname, "-", "-", "-", "-", "-", "-", "MISSING")); continue
                P, R, F, TP, FP, FN = score(p, gold)
                rows.append((m, method, gname, P, R, F, TP, FP, FN, ""))
    # print
    hdr = ["model", "method", "gold", "P", "R", "F1", "TP", "FP", "FN", "note"]
    def fmt(x): return f"{x:.3f}" if isinstance(x, float) else str(x)
    data = [hdr] + [[r[0], r[1], r[2]] + [fmt(x) for x in r[3:6]] + [str(x) for x in r[6:]] for r in rows]
    w = [max(len(row[i]) for row in data) for i in range(len(hdr))]
    for row in data:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(row)))
    json.dump([dict(zip(hdr, r)) for r in rows],
              open(os.path.join(HERE, "results", "matrix.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
