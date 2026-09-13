#!/usr/bin/env python3
"""Second-way checks on run_lca_eval.py, plus uncertainty for the n=15 re-ranking.

Every headline number gets verified by a route that does not reuse the code that
produced it:

  A. the 112 "forgiven" pairs must equal (LCA TP - char TP) -- two independent counts
  B. LCA must be a strict superset of char on EVERY row (TP up, FP down, FN down)
  C. paired bootstrap over the 15 testv2 papers -> is qwopus3.5's lead real at n=15?
  D. how much of the corpus-level lift would survive if gold/pred were both
     collapsed to genus (i.e. is LCA just doing what rank-normalisation would)?
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_lca import (  # noqa: E402
    HERE, EXTRACTIONS, EVAL_RESULTS, TESTV2,
    LCA, match_taxa_char, match_taxa_lca, score_rows, prf,
    load_new_gold, load_sheet, gold_is_blank, dedup_first, norm_title, parse_taxa,
)

OUT = os.path.join(HERE, "lca_checks.json")
RES = {}


def flush():
    json.dump(RES, open(OUT, "w"), indent=2)


R = LCA()
gold, _ = load_new_gold()
doi2row, title2doi = load_sheet()
by_title, _ = dedup_first(json.load(open(EXTRACTIONS)))

scoreable = []
for doi, cells in gold.items():
    row = doi2row.get(doi)
    if row is None:
        continue
    e = by_title.get(norm_title(row["Title"]))
    if e is not None and not gold_is_blank(cells):
        scoreable.append((doi, row, e, cells))
rows = [(e["predicted_enriched"], e["predicted_depleted"],
         c.get("Enriched", ""), c.get("Depleted", "")) for _, _, e, c in scoreable]
print("scoreable papers:", len(rows))

# ---------------------------------------------------------------- A + B
print("\n[A/B] per-row superset check + independent forgiven count")
d_tp = d_fp = d_fn = 0
viol = 0
for pe, pd_, ge, gd in rows:
    for p, g in ((pe, ge), (pd_, gd)):
        P, G = parse_taxa(p), parse_taxa(g)
        c = match_taxa_char(P, G)
        l = match_taxa_lca(P, G, R)
        if not (l[0] >= c[0] and l[1] <= c[1] and l[2] <= c[2]):
            viol += 1
        d_tp += l[0] - c[0]
        d_fp += l[1] - c[1]
        d_fn += l[2] - c[2]
print(f"  rows where LCA was NOT a superset of char: {viol}  (must be 0)")
print(f"  TP gained {d_tp:+d} | FP {d_fp:+d} | FN {d_fn:+d}")
print(f"  -> independent forgiven count = {d_tp} (run_lca_eval reported 112)")
RES["superset_violations"] = viol
RES["delta_tp"] = d_tp
RES["delta_fp"] = d_fp
RES["delta_fn"] = d_fn
flush()

# ---------------------------------------------------------------- C
print("\n[C] paired bootstrap over the 15 testv2 papers (2000 resamples)")
tv2 = json.load(open(TESTV2))
keys = [norm_title(p["title"]) for p in tv2]
k2doi = {k: title2doi[k] for k in keys if k in title2doi}

models = {}
for fn in sorted(os.listdir(EVAL_RESULTS)):
    if not fn.endswith("__samgated-v1__testv2.json"):
        continue
    m = fn.split("__")[0]
    recs = {norm_title(r["title"]): r for r in json.load(open(os.path.join(EVAL_RESULTS, fn)))}
    models[m] = recs

order = [k for k in keys if k in k2doi and not gold_is_blank(gold[k2doi[k]])]
print("  papers usable:", len(order))


def counts_per_paper(m, which):
    """-> [(tp,fp,fn)] one per paper, so the bootstrap resamples PAPERS not taxa."""
    out = []
    for k in order:
        r = models[m][k]
        if which.startswith("new"):
            g = gold[k2doi[k]]
            ge, gd = g.get("Enriched", ""), g.get("Depleted", "")
        else:
            ge, gd = r["expected_enriched"], r["expected_depleted"]
        match = (lambda p, e: match_taxa_lca(p, e, R)) if which.endswith("lca") else match_taxa_char
        t = f = n = 0
        for pp, gg in ((r["predicted_enriched"], ge), (r["predicted_depleted"], gd)):
            a, b, c = match(parse_taxa(pp), parse_taxa(gg))
            t += a
            f += b
            n += c
        out.append((t, f, n))
    return out


cache = {(m, w): counts_per_paper(m, w) for m in models
         for w in ("old_char", "old_lca", "new_char", "new_lca")}


def f1_from(cnts, idx):
    tp = sum(cnts[i][0] for i in idx)
    fp = sum(cnts[i][1] for i in idx)
    fn = sum(cnts[i][2] for i in idx)
    return prf(tp, fp, fn)[2]


rng = random.Random(11)
B = 2000
boots = {k: [] for k in cache}
wins = Counter()
top1 = Counter()
for _ in range(B):
    idx = [rng.randrange(len(order)) for _ in range(len(order))]
    for k, c in cache.items():
        boots[k].append(f1_from(c, idx))
    # under the fair metric+gold, who is #1 in this resample?
    best = max(models, key=lambda m: boots[(m, "new_lca")][-1])
    top1[best] += 1
    for m in models:
        if m == "qwopus3.5-27b-v3":
            continue
        if boots[("qwopus3.5-27b-v3", "new_lca")][-1] > boots[(m, "new_lca")][-1]:
            wins[m] += 1

ci = {}
for (m, w), v in boots.items():
    v = sorted(v)
    ci[f"{m}|{w}"] = [round(v[int(.025 * B)], 4), round(v[int(.975 * B)], 4)]
print("  95% bootstrap CI on F1 (new gold, LCA metric):")
for m in sorted(models, key=lambda m: -f1_from(cache[(m, "new_lca")], range(len(order)))):
    pt = f1_from(cache[(m, "new_lca")], range(len(order)))
    lo, hi = ci[f"{m}|new_lca"]
    print(f"    {m:24s} {pt:.4f}  [{lo:.4f}, {hi:.4f}]")
print("\n  P(qwopus3.5 beats X) over resamples, new gold + LCA:")
for m in sorted(wins, key=lambda m: -wins[m]):
    print(f"    vs {m:24s} {wins[m]/B:.3f}")
print("\n  P(model is rank #1) over resamples:")
for m, n in top1.most_common():
    print(f"    {m:24s} {n/B:.3f}")
RES["bootstrap"] = {"B": B, "n_papers": len(order), "ci": ci,
                    "p_qwopus35_beats": {m: round(wins[m] / B, 4) for m in wins},
                    "p_rank1": {m: round(n / B, 4) for m, n in top1.items()}}
flush()

# ---------------------------------------------------------------- D
print("\n[D] is LCA just genus-collapse? roll every taxon up to its genus, then char-match")


def rollup(s):
    out = []
    for t in parse_taxa(s):
        tid, sci, rank = R.info(t)
        if not tid:
            out.append(t)
            continue
        # walk up until genus or above
        cur = tid
        for _ in range(40):
            if R.tax.rank.get(cur) == "genus":
                break
            nxt = R.tax.parent.get(cur)
            if not nxt or nxt == cur:
                cur = tid
                break
            cur = nxt
        out.append(R.tax.sci.get(cur, sci).lower())
    return "; ".join(out)


rolled = [(rollup(a), rollup(b), rollup(c), rollup(d)) for a, b, c, d in rows]
s_roll = score_rows(rolled, match_taxa_char)["combined"]
print(f"  genus-collapsed + char : P={s_roll['precision']:.4f} R={s_roll['recall']:.4f} "
      f"F1={s_roll['f1']:.4f}  (TP={s_roll['TP']})")
print("  (compare: char 0.7549, LCA 0.7801)")
RES["genus_rollup_char"] = s_roll
flush()
print("\nwrote", OUT)
