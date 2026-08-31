#!/usr/bin/env python3
"""TASK 1 supplement -- the MECHANISM behind the old88/new139 gap.

analyze_drop.py showed the gap is in the papers, not the annotation, and that the
88 carry more gold taxa. This asks *why more gold taxa means higher F1*, and
whether gold size fully explains the group gap (stratum-matched test).

Hypothesis: the extractor emits a near-constant ~10 taxa per paper regardless of
how many the gold holds. If so, precision is close to a mechanical function of
gold size and the "difficulty" of a paper is mostly the annotator's verbosity.

Writes analyze_drop2.json.
"""
import csv, json, os, re, sys, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
from run_eval import parse_taxa, match_taxa  # noqa: E402

RESULTS = os.path.join(P3, "dsmlp_model_prompting", "eval-v2", "results",
                       "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")
NEW_CSV = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
ENR = "KeyTaxa_Enriched ↑ (Taxa1, Taxa2, etc.)"
DEP = "KeyTaxa_Depleted ↓"
NPERM = 10000


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def micro(recs):
    TP = sum(r["c"][0] for r in recs); FP = sum(r["c"][1] for r in recs); FN = sum(r["c"][2] for r in recs)
    p = TP / (TP + FP) if TP + FP else 0.0
    r_ = TP / (TP + FN) if TP + FN else 0.0
    return 2 * p * r_ / (p + r_) if p + r_ else 0.0


def pearson(x, y):
    n = len(x); mx = st.mean(x); my = st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** .5
    dy = sum((b - my) ** 2 for b in y) ** .5
    return num / (dx * dy) if dx and dy else 0.0


def perm_corr(x, y, nperm=NPERM, seed=0):
    """Permutation test on a correlation. Each (x,y) pair is one PAPER, so
    shuffling y across pairs is a paper-level shuffle."""
    obs = pearson(x, y)
    y = list(y); rng = random.Random(seed); ge = 0
    for _ in range(nperm):
        rng.shuffle(y)
        if abs(pearson(x, y)) >= abs(obs) - 1e-12:
            ge += 1
    return {"r": round(obs, 4), "p_two_sided": round((ge + 1) / (nperm + 1), 4), "n": len(x)}


def main():
    res = json.load(open(RESULTS))
    sheet = {norm_title(r["Title"]): r for r in csv.DictReader(open(NEW_CSV))}
    recs = []
    for r in res:
        g = sheet.get(norm_title(r["title"]))
        if not g:
            continue
        ge_, gd_ = parse_taxa(g[ENR]), parse_taxa(g[DEP])
        if not (ge_ or gd_):
            continue
        pe_, pd_ = parse_taxa(r["predicted_enriched"]), parse_taxa(r["predicted_depleted"])
        a = match_taxa(pe_, ge_); b = match_taxa(pd_, gd_)
        c = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
        old_n = len(parse_taxa(r["expected_enriched"])) + len(parse_taxa(r["expected_depleted"]))
        recs.append({"title": r["title"], "c": c,
                     "gold_n": len(ge_) + len(gd_), "pred_n": len(pe_) + len(pd_),
                     "group": "old88" if old_n > 0 else "new139"})

    out = {}
    gold_n = [r["gold_n"] for r in recs]
    pred_n = [r["pred_n"] for r in recs]

    # 1. Does the extractor scale its output to the paper's true taxa count?
    out["corr_gold_vs_pred_count"] = perm_corr(gold_n, pred_n, seed=1)
    print("=== does the extractor size its output to the gold? ===")
    print(f"  corr(gold taxa, predicted taxa) r={out['corr_gold_vs_pred_count']['r']:.3f} "
          f"p={out['corr_gold_vs_pred_count']['p_two_sided']:.4f}")
    print(f"  predicted taxa per paper: mean={st.mean(pred_n):.2f} sd={st.pstdev(pred_n):.2f} "
          f"median={st.median(pred_n)}  (gold: mean={st.mean(gold_n):.2f} sd={st.pstdev(gold_n):.2f})")
    out["pred_count_stats"] = {"mean": round(st.mean(pred_n), 2), "sd": round(st.pstdev(pred_n), 2),
                               "median": st.median(pred_n)}
    out["gold_count_stats"] = {"mean": round(st.mean(gold_n), 2), "sd": round(st.pstdev(gold_n), 2),
                               "median": st.median(gold_n)}

    # 2. per-paper precision / recall vs gold size
    pp = [(r["c"][0] / (r["c"][0] + r["c"][1])) for r in recs if r["c"][0] + r["c"][1]]
    pg = [r["gold_n"] for r in recs if r["c"][0] + r["c"][1]]
    out["corr_goldsize_vs_precision"] = perm_corr(pg, pp, seed=2)
    rr = [(r["c"][0] / (r["c"][0] + r["c"][2])) for r in recs if r["c"][0] + r["c"][2]]
    rg = [r["gold_n"] for r in recs if r["c"][0] + r["c"][2]]
    out["corr_goldsize_vs_recall"] = perm_corr(rg, rr, seed=3)
    print(f"  corr(gold size, per-paper precision) r={out['corr_goldsize_vs_precision']['r']:.3f} "
          f"p={out['corr_goldsize_vs_precision']['p_two_sided']:.4f}")
    print(f"  corr(gold size, per-paper recall)    r={out['corr_goldsize_vs_recall']['r']:.3f} "
          f"p={out['corr_goldsize_vs_recall']['p_two_sided']:.4f}")

    # 3. Stratum-matched group comparison: does the old88 advantage survive
    #    conditioning on gold size? Permute the group label WITHIN strata.
    print("\n=== stratum-matched old88 vs new139 (conditioning on gold size) ===")
    strata = [(1, 5), (6, 10), (11, 15), (16, 10 ** 6)]

    def strat_gap(assign):
        """assign: dict id->group. Returns weighted mean F1 gap across strata."""
        gaps, wts = [], []
        for lo, hi in strata:
            s = [r for r in recs if lo <= r["gold_n"] <= hi]
            a = [r for r in s if assign[id(r)] == "old88"]
            b = [r for r in s if assign[id(r)] == "new139"]
            if len(a) >= 3 and len(b) >= 3:
                gaps.append(micro(a) - micro(b)); wts.append(len(s))
        return sum(g * w for g, w in zip(gaps, wts)) / sum(wts) if wts else 0.0

    real = {id(r): r["group"] for r in recs}
    obs = strat_gap(real)
    rng = random.Random(4); ge = 0
    for _ in range(NPERM):
        perm = {}
        for lo, hi in strata:
            s = [r for r in recs if lo <= r["gold_n"] <= hi]
            labs = [r["group"] for r in s]
            rng.shuffle(labs)
            for r, l in zip(s, labs):
                perm[id(r)] = l
        if abs(strat_gap(perm)) >= abs(obs) - 1e-12:
            ge += 1
    out["stratum_matched_gap"] = {"observed_weighted_f1_gap": round(obs, 4),
                                  "p_two_sided": round((ge + 1) / (NPERM + 1), 4),
                                  "n_perm": NPERM,
                                  "note": "group label permuted WITHIN gold-size strata"}
    print(f"  weighted F1 gap (old88 - new139), gold-size matched: {obs:+.4f} "
          f"p={out['stratum_matched_gap']['p_two_sided']:.4f}")

    # 4. Power: what gap would we have detected at n=88 vs 139?
    #    Simulate by subsampling the observed per-paper triples.
    print("\n=== power check on the headline paper effect ===")
    rng = random.Random(5)
    allr = recs
    detect = {}
    for target in [0.05, 0.08, 0.10, 0.15]:
        # crude: bootstrap the null spread of the 88-vs-139 F1 gap
        pass
    null_gaps = []
    for _ in range(2000):
        pool = list(allr); rng.shuffle(pool)
        null_gaps.append(micro(pool[:88]) - micro(pool[88:]))
    sd = st.pstdev(null_gaps)
    crit = 1.96 * sd
    out["power"] = {"null_sd_of_f1_gap": round(sd, 4),
                    "min_detectable_gap_at_alpha.05": round(crit, 4),
                    "observed_gap": round(micro([r for r in allr if r['group'] == 'old88'])
                                          - micro([r for r in allr if r['group'] == 'new139']), 4)}
    print(f"  null SD of the 88-vs-139 F1 gap = {sd:.4f}; "
          f"smallest gap detectable at alpha=.05 is ~{crit:.3f}")
    print(f"  observed gap = {out['power']['observed_gap']:.3f} -> "
          f"{'above' if out['power']['observed_gap'] > crit else 'BELOW'} that threshold")

    json.dump(out, open(os.path.join(HERE, "analyze_drop2.json"), "w"), indent=2)
    print("\nwrote analyze_drop2.json")


if __name__ == "__main__":
    main()
