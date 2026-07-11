#!/usr/bin/env python3
"""Score original vs judge-b1 vs judge-b3 for each model, vs both golds, and pull the judge
timing from logs/judge_<model>.log. Print a compact table."""
import json, os, re, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); EVAL = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, EVAL); sys.path.insert(0, "/tmp/pylibs")
import common
from taxonomy_match import TaxResolver, match_taxa_lca, tools_available, setup
if not tools_available():
    setup()

TS = json.load(open(os.path.join(EVAL, "..", "..", "EmilySong_GoldStandardPaper", "test_set_v2.json")))
FG = json.load(open(os.path.join(EVAL, "results", "fable_gold_15.json")))
N = len(TS)
def parse(v): return common.parse_taxa(v)
ORIG = {i: (parse(TS[i].get("taxa_enriched")), parse(TS[i].get("taxa_depleted"))) for i in range(N)}
FABLE = {i: (parse(FG[i]["taxa_enriched"]), parse(FG[i]["taxa_depleted"])) for i in range(N)}
RESF = {"qwopus3.5-27b-v3": "qwopus3.5-27b-v3__q4km__samgated-v1__testv2.json",
        "qwythos-9b": "qwythos-9b__q8__samgated-v1__testv2.json",
        "qwen2.5-32b-instruct": "qwen2.5-32b-instruct__q4km__samgated-v1__testv2.json"}

def orig_pred(m):
    d = json.load(open(os.path.join(EVAL, "results", RESF[m]))); bt = {r["title"]: r for r in d}
    return {i: (parse(bt[TS[i]["title"]].get("predicted_enriched")),
                parse(bt[TS[i]["title"]].get("predicted_depleted"))) for i in range(N) if TS[i]["title"] in bt}

def cache_pred(name):
    f = os.path.join(HERE, "cache", name)
    if not os.path.exists(f): return None
    d = json.load(open(f)); return {int(k): (v["taxa_enriched"], v["taxa_depleted"]) for k, v in d.items()}

def judge_time(m, batch):
    lf = os.path.join(HERE, "logs", f"judge_{m}.log")
    if not os.path.exists(lf): return None
    mt = re.search(rf"BATCH={batch} DONE: (\d+) calls, (\d+)s total \(([\d.]+)s/paper\)", open(lf).read())
    return (int(mt.group(1)), int(mt.group(2)), float(mt.group(3))) if mt else None

_R = None
def warm(all_preds):
    global _R; _R = TaxResolver(); names = []
    for pr in all_preds:
        if not pr: continue
        for e, d in pr.values(): names += [x.lower() for x in e + d]
    for g in (ORIG, FABLE):
        for e, d in g.values(): names += e + d
    _R.warm(names)

def score(pred, gold):
    TP = FP = FN = 0
    for i in range(N):
        e, d = pred.get(i, ([], [])) if pred else ([], [])
        for pp, gg in (([x.lower() for x in e], gold[i][0]), ([x.lower() for x in d], gold[i][1])):
            tp, fp, fn = match_taxa_lca(pp, gg, _R); TP += tp; FP += fp; FN += fn
    P = TP/(TP+FP) if TP+FP else 0; R = TP/(TP+FN) if TP+FN else 0
    return P, R, (2*P*R/(P+R) if P+R else 0)

models = ["qwopus3.5-27b-v3", "qwythos-9b", "qwen2.5-32b-instruct"]
P = {}
for m in models:
    P[(m, "original")] = orig_pred(m)
    P[(m, "judge_b1")] = cache_pred(f"judged_b1__{m}.json")
    P[(m, "judge_b3")] = cache_pred(f"judged_b3__{m}.json")
warm([v for v in P.values() if v])

hdr = ["model", "variant", "gold", "P", "R", "F1", "judge_time"]
rows = [hdr]
for m in models:
    for var in ["original", "judge_b1", "judge_b3"]:
        pred = P[(m, var)]
        if pred is None:
            rows.append([m, var, "-", "-", "-", "-", "MISSING"]); continue
        batch = 1 if var == "judge_b1" else (3 if var == "judge_b3" else None)
        t = judge_time(m, batch) if batch else None
        tstr = f"{t[1]}s ({t[2]}s/pp, {t[0]}calls)" if t else ("-" if var == "original" else "?")
        for gname, gold in (("orig", ORIG), ("fable", FABLE)):
            p, r, f = score(pred, gold)
            rows.append([m, var, gname, f"{p:.3f}", f"{r:.3f}", f"{f:.3f}", tstr if gname == "orig" else ""])
w = [max(len(str(row[i])) for row in rows) for i in range(len(hdr))]
for row in rows:
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(row)))
json.dump([dict(zip(hdr, r)) for r in rows[1:]], open(os.path.join(HERE, "results", "judge_matrix.json"), "w"), indent=2)
