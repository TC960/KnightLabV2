#!/usr/bin/env python3
"""Rescore the 250-paper Qwopus3.5 extraction against the CORRECT datasheet.

All published numbers used ALL_EMILY_PAPERS_WITH_(inGoldStd)_COLUMN.csv, which is
64% blank on both taxa columns — every blank row turned every prediction into a
false positive. The correct sheet ("Microbiota Signatures Neurological Disorders
Sheet 2 - Main Datasheet.csv") is 1% blank.

Scoring reuses run_eval.py's parse_taxa/match_taxa (char-ngram cosine >= 0.5).
The taxonomy-aware (NCBI lineage) metric needs taxonkit/gnparser linux_amd64
binaries -- unavailable on this machine, so char is used throughout, for BOTH
the old and new gold, so the comparison is apples-to-apples.

Writes nothing except stdout + rescore_correct_sheet.json.
"""
import csv, json, os, re, sys, random

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
from run_eval import parse_taxa, match_taxa  # noqa: E402

RESULTS = os.path.join(P3, "dsmlp_model_prompting", "eval-v2", "results",
                       "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")
NEW_CSV = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
OLD_CSV = os.path.join(P3, "EmilySong_GoldStandardPaper", "ALL_EMILY_PAPERS_WITH_(inGoldStd)_COLUMN.csv")
TESTV2 = os.path.join(P3, "EmilySong_GoldStandardPaper", "test_set_v2.json")

ENR_NEW = "KeyTaxa_Enriched ↑ (Taxa1, Taxa2, etc.)"
DEP_NEW = "KeyTaxa_Depleted ↓"


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def load():
    res = json.load(open(RESULTS))
    new = list(csv.DictReader(open(NEW_CSV)))
    old = list(csv.DictReader(open(OLD_CSV)))
    testv2 = {norm_title(p["title"]) for p in json.load(open(TESTV2))}
    return res, new, old, testv2


def score(rows, gold_key):
    """rows: list of (pred_enr, pred_dep, gold_enr, gold_dep). Returns dict + per-paper."""
    TP = FP = FN = 0
    per = []
    for pe, pd, ge, gd in rows:
        a = match_taxa(parse_taxa(pe), parse_taxa(ge))
        b = match_taxa(parse_taxa(pd), parse_taxa(gd))
        tp, fp, fn = a[0] + b[0], a[1] + b[1], a[2] + b[2]
        TP += tp; FP += fp; FN += fn
        per.append((tp, fp, fn))
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"gold": gold_key, "n": len(rows), "TP": TP, "FP": FP, "FN": FN,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}, per


def main():
    res, new, old, testv2 = load()

    new_by = {}
    for r in new:
        new_by[norm_title(r["Title"])] = r
    old_by = {norm_title(r["Title"]): r for r in old}

    joined, unjoined = [], []
    for r in res:
        g = new_by.get(norm_title(r["title"]))
        (joined if g else unjoined).append((r, g))

    print(f"results papers: {len(res)}  joined to correct sheet: {len(joined)}  "
          f"unjoined: {len(unjoined)}")
    for r, _ in unjoined:
        print("   UNJOINED:", r["title"])

    # blankness of each sheet
    def blank_count(rows, ke, kd):
        return sum(1 for r in rows if not (r[ke].strip() or r[kd].strip()))
    print(f"\nOLD sheet: {len(old)} rows, {blank_count(old,'KeyTaxa_Enriched','KeyTaxa_Depleted')} blank on both")
    print(f"NEW sheet: {len(new)} rows, {blank_count(new,ENR_NEW,DEP_NEW)} blank on both")

    # ---- assemble scoring rows -------------------------------------------------
    # OLD gold = the expected_* already baked into the results json (from
    # all_usable_papers.json, which came from the old sheet).
    old_rows_all = [(r["predicted_enriched"], r["predicted_depleted"],
                     r["expected_enriched"], r["expected_depleted"]) for r in res]
    old_gold_nonblank = [r for r in res
                         if parse_taxa(r["expected_enriched"]) or parse_taxa(r["expected_depleted"])]
    old_rows_gold = [(r["predicted_enriched"], r["predicted_depleted"],
                      r["expected_enriched"], r["expected_depleted"]) for r in old_gold_nonblank]

    new_rows_all, new_rows_gold, new_blank_papers = [], [], []
    tv2_old, tv2_new = [], []
    for r, g in joined:
        row = (r["predicted_enriched"], r["predicted_depleted"], g[ENR_NEW], g[DEP_NEW])
        new_rows_all.append(row)
        if parse_taxa(g[ENR_NEW]) or parse_taxa(g[DEP_NEW]):
            new_rows_gold.append(row)
        else:
            new_blank_papers.append(r["title"])
        if norm_title(r["title"]) in testv2:
            tv2_new.append(row)
            tv2_old.append((r["predicted_enriched"], r["predicted_depleted"],
                            r["expected_enriched"], r["expected_depleted"]))

    out = {}
    tbl = []
    for label, rows, gold in [
        ("all papers", old_rows_all, "OLD (blank-ridden)"),
        ("all papers", new_rows_all, "NEW (correct)"),
        ("papers with non-blank gold", old_rows_gold, "OLD (blank-ridden)"),
        ("papers with non-blank gold", new_rows_gold, "NEW (correct)"),
        ("test_set_v2 (15)", tv2_old, "OLD (blank-ridden)"),
        ("test_set_v2 (15)", tv2_new, "NEW (correct)"),
    ]:
        s, _ = score(rows, gold)
        s["subset"] = label
        tbl.append(s)
        print(f"{label:28s} {gold:20s} n={s['n']:3d}  P={s['precision']:.3f} "
              f"R={s['recall']:.3f} F1={s['f1']:.3f}  (TP={s['TP']} FP={s['FP']} FN={s['FN']})")

    print(f"\npapers still blank in NEW gold among our joined set: {len(new_blank_papers)}")
    for t in new_blank_papers:
        print("   ", t)

    out["table"] = tbl
    out["unjoined"] = [r["title"] for r, _ in unjoined]
    out["blank_in_new_gold"] = new_blank_papers
    out["metric"] = "char-ngram cosine >= 0.5 (taxonomy/LCA metric unavailable: linux-only binaries)"

    # ---- paper-level permutation test: is the new-gold F1 better than chance
    # alignment between predictions and golds? Shuffle gold assignment across papers.
    obs, _ = score(new_rows_gold, "obs")
    rng = random.Random(0)
    preds = [(a, b) for a, b, _, _ in new_rows_gold]
    golds = [(c, d) for _, _, c, d in new_rows_gold]
    null = []
    for _ in range(1000):
        idx = list(range(len(golds)))
        rng.shuffle(idx)
        rows = [(preds[i][0], preds[i][1], golds[idx[i]][0], golds[idx[i]][1])
                for i in range(len(idx))]
        null.append(score(rows, "null")[0]["f1"])
    ge = sum(1 for x in null if x >= obs["f1"])
    print(f"\npermutation (paper-level gold shuffle, 1000x): observed F1={obs['f1']:.3f}, "
          f"null mean={sum(null)/len(null):.3f}, max={max(null):.3f}, p={(ge+1)/1001:.4f}")
    out["permutation_f1"] = {"observed": obs["f1"], "null_mean": round(sum(null) / len(null), 4),
                             "null_max": round(max(null), 4), "p": round((ge + 1) / 1001, 4),
                             "n_perm": 1000}

    json.dump(out, open(os.path.join(HERE, "rescore_correct_sheet.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
