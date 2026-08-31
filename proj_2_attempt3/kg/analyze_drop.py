#!/usr/bin/env python3
"""TASK 1 -- why did F1 fall from .647 (n=88) to .596 (n=227)?

The 88 papers that were annotated in the OLD export are a subset of the 227 that
are annotated in the NEW (correct) sheet. Two competing explanations:

  (A) THE PAPERS. The 88 are intrinsically easier -- fewer gold taxa, shorter,
      cleaner disease mix, better differential-abundance reporting.
  (B) THE ANNOTATION. The same papers score differently because the OLD gold and
      the NEW gold disagree about what the right answer is.

These are separable, because for the 88 papers we hold BOTH golds. Scoring the
same 88 predictions against both isolates the annotation effect; comparing
88-under-NEW vs 139-under-NEW isolates the paper effect.

All permutation tests shuffle the GROUP LABEL at the paper level (observations
within a paper are not independent).

Writes analyze_drop.json. Reads only; modifies nothing.
"""
import csv, json, os, re, sys, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
from run_eval import parse_taxa, match_taxa  # noqa: E402

RESULTS = os.path.join(P3, "dsmlp_model_prompting", "eval-v2", "results",
                       "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")
NEW_CSV = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
PAPERS = os.path.join(P3, "EmilySong_GoldStandardPaper", "all_usable_papers.json")

ENR = "KeyTaxa_Enriched ↑ (Taxa1, Taxa2, etc.)"
DEP = "KeyTaxa_Depleted ↓"
DAT = "Differential Abundance Test (try Ctrl F for these)"
SRC = "TaxaExtractionSource (Should be Main Text unless there is nothing listed)"

NPERM = 10000


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def prf(TP, FP, FN):
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def counts(pe, pd, ge, gd):
    a = match_taxa(parse_taxa(pe), parse_taxa(ge))
    b = match_taxa(parse_taxa(pd), parse_taxa(gd))
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def micro(recs, key):
    TP = sum(r[key][0] for r in recs)
    FP = sum(r[key][1] for r in recs)
    FN = sum(r[key][2] for r in recs)
    p, r, f = prf(TP, FP, FN)
    return {"n": len(recs), "TP": TP, "FP": FP, "FN": FN,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def perm_diff_micro(a, b, key, nperm=NPERM, seed=0):
    """Permutation test on the difference in micro-F1 between paper groups a and b.
    Shuffles group membership at the paper level (each paper's whole TP/FP/FN
    triple moves together), which is the right unit -- taxa within a paper share
    the paper's difficulty, annotator and text."""
    obs = micro(a, key)["f1"] - micro(b, key)["f1"]
    pool = a + b
    na = len(a)
    rng = random.Random(seed)
    ge = 0
    for _ in range(nperm):
        rng.shuffle(pool)
        d = micro(pool[:na], key)["f1"] - micro(pool[na:], key)["f1"]
        if abs(d) >= abs(obs) - 1e-12:
            ge += 1
    return {"observed_diff": round(obs, 4), "p_two_sided": round((ge + 1) / (nperm + 1), 4),
            "n_perm": nperm}


def perm_diff_mean(xa, xb, nperm=NPERM, seed=0):
    """Two-sided permutation test on a difference in means. Each element is one
    PAPER, so exchanging elements is already a paper-level shuffle."""
    if not xa or not xb:
        return None
    obs = st.mean(xa) - st.mean(xb)
    pool = list(xa) + list(xb)
    na = len(xa)
    rng = random.Random(seed)
    ge = 0
    for _ in range(nperm):
        rng.shuffle(pool)
        d = st.mean(pool[:na]) - st.mean(pool[na:])
        if abs(d) >= abs(obs) - 1e-12:
            ge += 1
    return {"mean_a": round(st.mean(xa), 3), "mean_b": round(st.mean(xb), 3),
            "median_a": round(st.median(xa), 3), "median_b": round(st.median(xb), 3),
            "observed_diff": round(obs, 3), "p_two_sided": round((ge + 1) / (nperm + 1), 4),
            "n_a": len(xa), "n_b": len(xb)}


def perm_prop(flags_a, flags_b, nperm=NPERM, seed=0):
    """Permutation test on a difference in proportions (paper-level booleans)."""
    return perm_diff_mean([float(x) for x in flags_a], [float(x) for x in flags_b],
                          nperm=nperm, seed=seed)


def main():
    res = json.load(open(RESULTS))
    sheet = {norm_title(r["Title"]): r for r in csv.DictReader(open(NEW_CSV))}
    papers = {norm_title(p["title"]): p for p in json.load(open(PAPERS))}

    recs = []
    for r in res:
        k = norm_title(r["title"])
        g = sheet.get(k)
        if not g:
            continue  # the 1 unjoined paper
        new_gold = (g[ENR], g[DEP])
        old_gold = (r["expected_enriched"], r["expected_depleted"])
        new_n = len(parse_taxa(g[ENR])) + len(parse_taxa(g[DEP]))
        old_n = len(parse_taxa(old_gold[0])) + len(parse_taxa(old_gold[1]))
        if new_n == 0:
            continue  # blank in the new gold -- unscoreable either way (5 papers)
        rec = {
            "title": r["title"],
            "group": "old88" if old_n > 0 else "new139",
            "gold_n_new": new_n,
            "gold_n_old": old_n,
            "pred_n": len(parse_taxa(r["predicted_enriched"])) + len(parse_taxa(r["predicted_depleted"])),
            "year": g["Year"].strip(),
            "country": g["Country"].strip(),
            "seq": g["SequencingType"].strip(),
            "disease": g["Disease"].strip(),
            "dat": g[DAT].strip(),
            "src": g[SRC].strip(),
            "chars": papers[k]["char_len"] if k in papers else None,
            "new": counts(r["predicted_enriched"], r["predicted_depleted"], *new_gold),
        }
        if old_n > 0:
            rec["old"] = counts(r["predicted_enriched"], r["predicted_depleted"], *old_gold)
        recs.append(rec)

    old88 = [r for r in recs if r["group"] == "old88"]
    new139 = [r for r in recs if r["group"] == "new139"]
    out = {"n_scoreable": len(recs), "n_old88": len(old88), "n_new139": len(new139)}
    print(f"scoreable under NEW gold: {len(recs)}  (old-annotated {len(old88)}, "
          f"newly-annotated {len(new139)})")

    # ---------------- DECOMPOSITION: papers vs annotation -------------------
    print("\n=== decomposition ===")
    dec = {
        "A_old88_under_OLD_gold": micro(old88, "old"),
        "B_old88_under_NEW_gold": micro(old88, "new"),
        "C_new139_under_NEW_gold": micro(new139, "new"),
        "D_all227_under_NEW_gold": micro(recs, "new"),
    }
    for k, v in dec.items():
        print(f"  {k:28s} n={v['n']:3d} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")
    out["decomposition"] = dec

    annot_effect = dec["B_old88_under_NEW_gold"]["f1"] - dec["A_old88_under_OLD_gold"]["f1"]
    paper_effect = dec["C_new139_under_NEW_gold"]["f1"] - dec["B_old88_under_NEW_gold"]["f1"]
    print(f"  annotation effect (same 88 papers, OLD->NEW gold): {annot_effect:+.4f}")
    print(f"  paper effect (NEW gold, 88 -> 139 papers):         {paper_effect:+.4f}")
    out["annotation_effect_f1"] = round(annot_effect, 4)
    out["paper_effect_f1"] = round(paper_effect, 4)

    # paired permutation for the annotation effect: same papers, two golds.
    # Flip which gold each paper contributes (paper-level sign flip).
    rng = random.Random(1)
    obs = annot_effect
    ge = 0
    for _ in range(NPERM):
        A, B = [], []
        for r in old88:
            if rng.random() < 0.5:
                A.append({"x": r["old"]}); B.append({"x": r["new"]})
            else:
                A.append({"x": r["new"]}); B.append({"x": r["old"]})
        d = micro(B, "x")["f1"] - micro(A, "x")["f1"]
        if abs(d) >= abs(obs) - 1e-12:
            ge += 1
    out["annotation_effect_perm"] = {"observed_diff": round(obs, 4),
                                     "p_two_sided": round((ge + 1) / (NPERM + 1), 4),
                                     "n_perm": NPERM, "test": "paired paper-level gold swap"}
    print(f"  annotation effect permutation p={out['annotation_effect_perm']['p_two_sided']:.4f}")

    out["paper_effect_perm"] = perm_diff_micro(new139, old88, "new")
    print(f"  paper effect permutation      p={out['paper_effect_perm']['p_two_sided']:.4f}")

    # ---------------- CHARACTERISE the two groups --------------------------
    print("\n=== group comparison (old88 vs new139) ===")
    comp = {}
    for field, getter in [
        ("gold_taxa_per_paper", lambda r: r["gold_n_new"]),
        ("pred_taxa_per_paper", lambda r: r["pred_n"]),
        ("paper_chars", lambda r: r["chars"]),
        ("year", lambda r: int(r["year"]) if r["year"].isdigit() else None),
    ]:
        a = [getter(r) for r in old88 if getter(r) is not None]
        b = [getter(r) for r in new139 if getter(r) is not None]
        comp[field] = perm_diff_mean(a, b, seed=hash(field) % 10000)
        c = comp[field]
        print(f"  {field:22s} old88={c['mean_a']:9.2f} new139={c['mean_b']:9.2f} "
              f"diff={c['observed_diff']:+9.2f} p={c['p_two_sided']:.4f}")

    for field, key, vals in [
        ("seq_16S", "seq", ["16S"]),
        ("seq_shotgun_or_both", "seq", ["Shotgun", "Both"]),
        ("dat_LEfSe", "dat", None),          # substring test below
        ("dat_blank", "dat", [""]),
        ("src_main_text", "src", ["Main Text"]),
        ("country_China", "country", ["China"]),
    ]:
        if field == "dat_LEfSe":
            fa = ["LEfSe" in r["dat"] for r in old88]
            fb = ["LEfSe" in r["dat"] for r in new139]
        else:
            fa = [r[key] in vals for r in old88]
            fb = [r[key] in vals for r in new139]
        comp[field] = perm_prop(fa, fb, seed=hash(field) % 10000)
        c = comp[field]
        print(f"  {field:22s} old88={c['mean_a']:9.3f} new139={c['mean_b']:9.3f} "
              f"diff={c['observed_diff']:+9.3f} p={c['p_two_sided']:.4f}")
    out["group_comparison"] = comp

    # disease mix
    import collections
    da = collections.Counter(r["disease"] for r in old88)
    db = collections.Counter(r["disease"] for r in new139)
    dis = {}
    for d in set(da) | set(db):
        dis[d] = {"old88": da[d], "new139": db[d],
                  "old88_pct": round(100 * da[d] / len(old88), 1),
                  "new139_pct": round(100 * db[d] / len(new139), 1)}
    out["disease_mix"] = dict(sorted(dis.items(), key=lambda kv: -(kv[1]["old88"] + kv[1]["new139"])))
    print("\n  disease mix (top):")
    for d, v in list(out["disease_mix"].items())[:8]:
        print(f"    {d[:44]:46s} old88 {v['old88']:3d} ({v['old88_pct']:4.1f}%)  "
              f"new139 {v['new139']:3d} ({v['new139_pct']:4.1f}%)")

    # per-disease F1 to see whether the mix or the within-disease difficulty moved
    perdis = {}
    for d in set(r["disease"] for r in recs):
        a = [r for r in old88 if r["disease"] == d]
        b = [r for r in new139 if r["disease"] == d]
        if len(a) >= 5 and len(b) >= 5:
            perdis[d] = {"old88": micro(a, "new"), "new139": micro(b, "new")}
    out["per_disease_f1_new_gold"] = perdis
    print("\n  per-disease F1 under the NEW gold (diseases with >=5 papers in both):")
    for d, v in perdis.items():
        print(f"    {d[:44]:46s} old88 n={v['old88']['n']:3d} F1={v['old88']['f1']:.3f}  "
              f"new139 n={v['new139']['n']:3d} F1={v['new139']['f1']:.3f}")

    # Is gold size the mediator? Stratify the NEW-gold F1 by gold taxa count.
    print("\n=== F1 by gold-taxa-count stratum (NEW gold, all 227) ===")
    strata = [(1, 5), (6, 10), (11, 15), (16, 25), (26, 10**6)]
    strat = {}
    for lo, hi in strata:
        s = [r for r in recs if lo <= r["gold_n_new"] <= hi]
        sa = [r for r in s if r["group"] == "old88"]
        sb = [r for r in s if r["group"] == "new139"]
        lbl = f"{lo}-{hi if hi < 10**6 else '+'}"
        strat[lbl] = {"all": micro(s, "new"),
                      "old88": micro(sa, "new") if sa else None,
                      "new139": micro(sb, "new") if sb else None}
        m = strat[lbl]["all"]
        print(f"  gold taxa {lbl:7s} n={m['n']:3d} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f}"
              + (f"   | old88 n={len(sa):3d} F1={strat[lbl]['old88']['f1']:.3f}" if sa else "")
              + (f"  new139 n={len(sb):3d} F1={strat[lbl]['new139']['f1']:.3f}" if sb else ""))
    out["f1_by_gold_size"] = strat

    json.dump(out, open(os.path.join(HERE, "analyze_drop.json"), "w"), indent=2)
    print("\nwrote analyze_drop.json")


if __name__ == "__main__":
    main()
