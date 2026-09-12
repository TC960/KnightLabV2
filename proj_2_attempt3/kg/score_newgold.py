#!/usr/bin/env python3
"""Score the Qwopus3.5 extractions against the NEW human-curated gold standard.

The new gold is `high_confidence - final_constrained_override.csv`: 334 DOIs, two
rows each (Enriched / Depleted), taxa ';'-separated.

Join path (all three legs verified, see reconcile()):
    gold DOI -> "Microbiota Signatures ... Main Datasheet.csv" (DOI + Title)
             -> Title -> extractions_corrected.json

Two rules carried over from FINDINGS_task0_rescore.md, both load-bearing:

1. **Papers whose gold is blank on BOTH taxa columns are EXCLUDED.** Scoring a
   prediction against an empty cell turns every extraction into a false positive.
   That is exactly what produced the bogus "F1 0.390" earlier in this project.
2. **Metric is char-ngram cosine >= 0.5**, imported from eval-v2/run_eval.py so
   the numbers are directly comparable with everything already reported. The
   taxonomy-aware (LCA) metric needs linux-only binaries and is not used for the
   headline score -- but NCBI lineages ARE loaded here (taxonomy.py reads
   names.dmp directly) to *diagnose* rank-mismatch errors the char metric misses.

Everything is CPU-local. No API calls. Writes score_newgold.json + stdout.
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
from run_eval import parse_taxa, match_taxa  # noqa: E402

GOLD_CSV = "/Users/mohak/Downloads/high_confidence - final_constrained_override.csv"
SHEET_CSV = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
EXTRACTIONS = os.path.join(HERE, "extractions_corrected.json")
TEXTS = [os.path.join(P3, "EmilySong_GoldStandardPaper", "all_usable_papers.json"),
         os.path.join(HERE, "new_papers.json")]
OUT = os.path.join(HERE, "score_newgold.json")

SHEET_ENR = "KeyTaxa_Enriched ↑ (Taxa1, Taxa2, etc.)"
SHEET_DEP = "KeyTaxa_Depleted ↓"


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


# --------------------------------------------------------------------------- load

def load():
    gold = defaultdict(dict)          # doi -> {"Enriched": str, "Depleted": str}
    disorder = {}
    for r in csv.DictReader(open(GOLD_CSV)):
        doi = r["DOI"].strip().lower()
        gold[doi][r["field"].strip()] = r["high_confidence_taxa"].strip()
        disorder[doi] = r["disorder"].strip()

    sheet = list(csv.DictReader(open(SHEET_CSV)))
    doi2row = {}
    for r in sheet:
        d = r["DOI"].strip().lower()
        if d and d not in doi2row:
            doi2row[d] = r

    ex = json.load(open(EXTRACTIONS))
    texts = {}
    for path in TEXTS:
        for p in json.load(open(path)):
            texts.setdefault(norm_title(p["title"]), p.get("text", ""))
    return gold, disorder, sheet, doi2row, ex, texts


def dedup_extractions(ex):
    """extractions_corrected.json has 348 rows over 335 distinct titles: 13 papers
    were fetched twice (once by pubmed id, once by PMC/DOI url) and extracted twice.
    Keep the FIRST occurrence in file order. 9 of the 13 pairs are byte-identical
    predictions anyway; sensitivity to this choice is measured in main()."""
    seen, keep, dups = {}, [], []
    for e in ex:
        k = norm_title(e["title"])
        if k in seen:
            dups.append(k)
            continue
        seen[k] = e
        keep.append(e)
    return seen, keep, dups


# ---------------------------------------------------------------------- alignment

def align(predicted, expected):
    """Same greedy rule as run_eval.match_taxa, but returns the alignment so
    failures can be bucketed: each predicted taxon takes its best expected by
    char-ngram cosine; a hit needs >= 0.5.

    -> (pairs, fp_list, fn_list) where pairs = [(pred, exp, sim)].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not predicted and not expected:
        return [], [], []
    if not predicted:
        return [], [], list(expected)
    if not expected:
        return [], list(predicted), []
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(predicted + expected)
    sim = cosine_similarity(tf[:len(predicted)], tf[len(predicted):])
    pairs, fp, matched = [], [], set()
    for i in range(len(predicted)):
        j = int(sim[i].argmax())
        s = float(sim[i][j])
        if s >= 0.5:
            pairs.append((predicted[i], expected[j], round(s, 3)))
            matched.add(j)
        else:
            fp.append(predicted[i])
    fn = [expected[j] for j in range(len(expected)) if j not in matched]
    return pairs, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def score_rows(rows):
    """rows: [(pred_enr, pred_dep, gold_enr, gold_dep)] -> per-direction + combined."""
    acc = {"enriched": [0, 0, 0], "depleted": [0, 0, 0]}
    for pe, pdp, ge, gd in rows:
        for key, p, g in (("enriched", pe, ge), ("depleted", pdp, gd)):
            tp, fp, fn = match_taxa(parse_taxa(p), parse_taxa(g))
            acc[key][0] += tp
            acc[key][1] += fp
            acc[key][2] += fn
    out = {}
    for k, (tp, fp, fn) in acc.items():
        p, r, f = prf(tp, fp, fn)
        out[k] = {"n": len(rows), "TP": tp, "FP": fp, "FN": fn,
                  "precision": p, "recall": r, "f1": f}
    tp = sum(acc[k][0] for k in acc)
    fp = sum(acc[k][1] for k in acc)
    fn = sum(acc[k][2] for k in acc)
    p, r, f = prf(tp, fp, fn)
    out["combined"] = {"n": len(rows), "TP": tp, "FP": fp, "FN": fn,
                       "precision": p, "recall": r, "f1": f}
    return out


def f1_only(rows):
    tp = fp = fn = 0
    for pe, pdp, ge, gd in rows:
        for p, g in ((pe, ge), (pdp, gd)):
            a, b, c = match_taxa(parse_taxa(p), parse_taxa(g))
            tp += a
            fp += b
            fn += c
    return prf(tp, fp, fn)[2]


# -------------------------------------------------------------------------- main

def main():
    gold, disorder, sheet, doi2row, ex_raw, texts = load()
    by_title, ex, dup_titles = dedup_extractions(ex_raw)
    res = {"metric": "TF-IDF char_wb n-gram(2,4) cosine >= 0.5, micro-averaged "
                     "(identical to eval-v1/eval-v2 --metric char)"}

    # ---------------- 1. reconciliation -------------------------------------
    matched, no_extraction = [], []
    for doi, cells in gold.items():
        row = doi2row.get(doi)
        if row is None:
            no_extraction.append((doi, None, "DOI absent from datasheet"))
            continue
        e = by_title.get(norm_title(row["Title"]))
        if e is None:
            why = ("no full text scraped" if norm_title(row["Title"]) not in texts
                   else "text present but not extracted")
            no_extraction.append((doi, row["Title"], why))
        else:
            matched.append((doi, row, e, cells))

    blank = [(d, r["Title"]) for d, r, e, c in matched
             if not (parse_taxa(c.get("Enriched", "")) or parse_taxa(c.get("Depleted", "")))]
    blank_titles = {t for _, t in blank}
    scoreable = [m for m in matched if m[1]["Title"] not in blank_titles]

    print(f"gold DOIs                         : {len(gold)}")
    print(f"  joined to datasheet (DOI)       : {len(gold) - sum(1 for x in no_extraction if x[2].startswith('DOI'))}")
    print(f"  joined to an extraction (Title) : {len(matched)}")
    print(f"  NO extraction                   : {len(no_extraction)}")
    print(f"  of matched, blank on both cells : {len(blank)}  (EXCLUDED)")
    print(f"  SCOREABLE                       : {len(scoreable)}")
    print(f"\nduplicate-title extraction rows dropped: {len(dup_titles)} "
          f"({len(ex_raw)} rows -> {len(ex)} papers)")
    print("\nsample of gold papers with no extraction:")
    for d, t, why in no_extraction[:12]:
        print(f"   {d:34s} {why:22s} {(t or '')[:70]}")

    res["reconciliation"] = {
        "gold_dois": len(gold),
        "joined_to_datasheet": len(gold) - sum(1 for x in no_extraction if x[2].startswith("DOI")),
        "joined_to_extraction": len(matched),
        "no_extraction": len(no_extraction),
        "blank_gold_excluded": len(blank),
        "scoreable": len(scoreable),
        "dup_extraction_rows_dropped": len(dup_titles),
        "no_extraction_detail": [{"doi": d, "title": t, "reason": w} for d, t, w in no_extraction],
        "blank_gold_papers": [{"doi": d, "title": t} for d, t in blank],
    }

    # ---------------- 2. scores ---------------------------------------------
    rows = [(e["predicted_enriched"], e["predicted_depleted"],
             c.get("Enriched", ""), c.get("Depleted", "")) for _, r, e, c in scoreable]
    s = score_rows(rows)
    res["scores_new_gold"] = s
    print("\n--- NEW GOLD, %d scoreable papers -------------------------" % len(rows))
    for k in ("enriched", "depleted", "combined"):
        v = s[k]
        print(f"  {k:9s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
              f"   (TP={v['TP']} FP={v['FP']} FN={v['FN']})")

    # sensitivity to the dedup choice: score with LAST occurrence instead
    last = {}
    for e in ex_raw:
        last[norm_title(e["title"])] = e
    rows_last = [(last[norm_title(r["Title"])]["predicted_enriched"],
                  last[norm_title(r["Title"])]["predicted_depleted"],
                  c.get("Enriched", ""), c.get("Depleted", ""))
                 for _, r, e, c in scoreable]
    f_last = f1_only(rows_last)
    print(f"  dedup sensitivity: keeping LAST duplicate instead of FIRST -> F1 {f_last:.3f}")
    res["dedup_sensitivity_f1_last"] = f_last
    json.dump(res, open(OUT, "w"), indent=2)

    # ---------------- 3. like-for-like vs the OLD gold ----------------------
    # OLD gold is baked into extractions_corrected.json as expected_*; it came from
    # the same Main Datasheet columns. Restrict BOTH to the papers scoreable under
    # BOTH golds so the comparison is not a subset artefact.
    both, old_rows, new_rows = [], [], []
    for _, r, e, c in scoreable:
        if parse_taxa(e["expected_enriched"]) or parse_taxa(e["expected_depleted"]):
            both.append((r, e, c))
            old_rows.append((e["predicted_enriched"], e["predicted_depleted"],
                             e["expected_enriched"], e["expected_depleted"]))
            new_rows.append((e["predicted_enriched"], e["predicted_depleted"],
                             c.get("Enriched", ""), c.get("Depleted", "")))
    s_old = score_rows(old_rows)
    s_new = score_rows(new_rows)
    print(f"\n--- LIKE-FOR-LIKE, {len(both)} papers scoreable under BOTH golds ---")
    for lbl, ss in (("OLD (datasheet cols)", s_old), ("NEW (high_confidence)", s_new)):
        v = ss["combined"]
        print(f"  {lbl:22s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
              f"  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")
    res["like_for_like"] = {"n": len(both), "old_gold": s_old, "new_gold": s_new,
                            "delta_f1": round(s_new["combined"]["f1"] - s_old["combined"]["f1"], 4)}

    # gold set sizes, to explain any delta
    old_sz = sum(len(parse_taxa(e["expected_enriched"])) + len(parse_taxa(e["expected_depleted"]))
                 for _, e, _ in both)
    new_sz = sum(len(parse_taxa(c.get("Enriched", ""))) + len(parse_taxa(c.get("Depleted", "")))
                 for _, _, c in both)
    print(f"  gold taxa on those {len(both)} papers: OLD {old_sz}, NEW {new_sz}")
    res["like_for_like"]["gold_taxa_old"] = old_sz
    res["like_for_like"]["gold_taxa_new"] = new_sz
    json.dump(res, open(OUT, "w"), indent=2)

    # ---------------- 4. permutation test -----------------------------------
    obs = s["combined"]["f1"]
    rng = random.Random(0)
    preds = [(a, b) for a, b, _, _ in rows]
    golds = [(c, d) for _, _, c, d in rows]
    null = []
    for _ in range(1000):
        idx = list(range(len(golds)))
        rng.shuffle(idx)
        null.append(f1_only([(preds[i][0], preds[i][1], golds[idx[i]][0], golds[idx[i]][1])
                             for i in range(len(idx))]))
    ge = sum(1 for x in null if x >= obs)
    nm = sum(null) / len(null)
    print(f"\n--- PERMUTATION (shuffle gold->paper, 1000 draws) ---")
    print(f"  observed F1 {obs:.3f} | null mean {nm:.3f} | null max {max(null):.3f} "
          f"| p = {(ge + 1) / 1001:.4f}")
    res["permutation"] = {"observed_f1": obs, "null_mean": round(nm, 4),
                          "null_sd": round((sum((x - nm) ** 2 for x in null) / len(null)) ** 0.5, 4),
                          "null_max": round(max(null), 4),
                          "p": round((ge + 1) / 1001, 4), "n_perm": 1000}
    json.dump(res, open(OUT, "w"), indent=2)

    # ---------------- 5. error analysis -------------------------------------
    err = error_analysis(scoreable, texts)
    res["error_analysis"] = err
    json.dump(res, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


# ------------------------------------------------------------------ error buckets

def error_analysis(scoreable, texts):
    """Bucket every failure. The combined metric charges a direction flip twice
    (once FP, once FN) and cannot see it; rank mismatches are charged the same way.
    Both are separated out here against the NCBI taxdump."""
    import taxonomy
    tax = taxonomy.shared()
    print(f"\n--- ERROR ANALYSIS (taxdump loaded: {tax.ok}) ---")

    buckets = {"direction_flip": [], "rank_mismatch": [], "true_fp": [], "true_fn": []}
    fn_all, fp_all = [], []

    for doi, row, e, cells in scoreable:
        title = row["Title"]
        text = texts.get(norm_title(title), "")
        pe, pdp = parse_taxa(e["predicted_enriched"]), parse_taxa(e["predicted_depleted"])
        ge, gd = parse_taxa(cells.get("Enriched", "")), parse_taxa(cells.get("Depleted", ""))

        _, fp_e, fn_e = align(pe, ge)
        _, fp_d, fn_d = align(pdp, gd)

        # (a) direction flips: an unmatched prediction in one direction that DOES
        #     match an unmatched gold taxon in the OTHER direction.
        flipped_fp, flipped_fn = set(), set()
        for pred_side, fp_list, gold_other, fn_other, pdir in (
                ("enriched", fp_e, gd, fn_d, "enriched"),
                ("depleted", fp_d, ge, fn_e, "depleted")):
            pr, _, _ = align(fp_list, fn_other)
            for p, g, sim in pr:
                buckets["direction_flip"].append(
                    {"doi": doi, "title": title, "taxon_pred": p, "taxon_gold": g,
                     "we_said": pdir, "gold_says": "depleted" if pdir == "enriched" else "enriched",
                     "sim": sim})
                flipped_fp.add((pdir, p))
                flipped_fn.add(("depleted" if pdir == "enriched" else "enriched", g))

        for d, lst in (("enriched", fp_e), ("depleted", fp_d)):
            for t in lst:
                if (d, t) not in flipped_fp:
                    fp_all.append({"doi": doi, "title": title, "taxon": t, "direction": d})
        for d, lst in (("enriched", fn_e), ("depleted", fn_d)):
            for t in lst:
                if (d, t) not in flipped_fn:
                    fn_all.append({"doi": doi, "title": title, "taxon": t, "direction": d,
                                   "in_text": bool(text) and t.lower() in text.lower()})

    # (b) rank mismatch: an FP and an FN, SAME paper SAME direction, whose NCBI
    #     lineages are nested (one is an ancestor of the other). The char metric
    #     scores "Lachnospiraceae" vs "Blautia" as two separate errors; taxonomically
    #     it is one rank error.
    def resolve(t):
        tid, sci, rank, how = tax.resolve(t)
        return tid, rank

    lin = {}

    def lineage(tid):
        if tid not in lin:
            lin[tid] = set(tax.lineage(tid))
        return lin[tid]

    fp_by = defaultdict(list)
    for i, f in enumerate(fp_all):
        fp_by[(f["doi"], f["direction"])].append(i)
    rank_pairs, used_fp, used_fn = [], set(), set()
    for j, g in enumerate(fn_all):
        gtid, grank = resolve(g["taxon"])
        if not gtid:
            continue
        for i in fp_by.get((g["doi"], g["direction"]), []):
            if i in used_fp:
                continue
            ptid, prank = resolve(fp_all[i]["taxon"])
            if not ptid or ptid == gtid:
                continue
            if gtid in lineage(ptid) or ptid in lineage(gtid):
                rank_pairs.append({"doi": g["doi"], "title": g["title"],
                                   "direction": g["direction"],
                                   "predicted": fp_all[i]["taxon"], "pred_rank": prank,
                                   "gold": g["taxon"], "gold_rank": grank})
                used_fp.add(i)
                used_fn.add(j)
                break
    buckets["rank_mismatch"] = rank_pairs
    buckets["true_fp"] = [f for i, f in enumerate(fp_all) if i not in used_fp]
    buckets["true_fn"] = [f for j, f in enumerate(fn_all) if j not in used_fn]

    n_flip = len(buckets["direction_flip"])
    n_rank = len(rank_pairs)
    n_fp = len(buckets["true_fp"])
    n_fn = len(buckets["true_fn"])
    # each flip and each rank mismatch consumed one FP + one FN charge
    total_charges = 2 * n_flip + 2 * n_rank + n_fp + n_fn
    print(f"  total error charges (FP+FN in the metric): {total_charges}")
    for lbl, n, ch in (("direction flip", n_flip, 2 * n_flip),
                       ("rank mismatch", n_rank, 2 * n_rank),
                       ("false positive (unexplained)", n_fp, n_fp),
                       ("false negative (unexplained)", n_fn, n_fn)):
        print(f"    {lbl:30s} {n:5d} events -> {ch:5d} charges ({ch / total_charges:5.1%})")

    # FN rank profile + text presence
    fn_rank = Counter()
    fn_unres = 0
    for g in buckets["true_fn"]:
        tid, rank = resolve(g["taxon"])
        if tid:
            fn_rank[rank] += 1
        else:
            fn_unres += 1
    in_text = sum(1 for g in buckets["true_fn"] if g["in_text"])
    print(f"\n  FN rank profile: {dict(fn_rank.most_common())} | unresolved {fn_unres}")
    print(f"  FN present verbatim in the paper text: {in_text}/{len(buckets['true_fn'])} "
          f"({in_text / max(1, len(buckets['true_fn'])):.1%})")

    # baseline: rank profile of the gold overall, so 'concentrated at a rank' means something
    gold_rank = Counter()
    for doi, row, e, cells in scoreable:
        for d in ("Enriched", "Depleted"):
            for t in parse_taxa(cells.get(d, "")):
                tid, rank = resolve(t)
                gold_rank[rank if tid else "unresolved"] += 1
    print(f"  gold rank profile (all gold taxa): {dict(gold_rank.most_common())}")

    # FP: does the taxon appear in the paper at all? (weak evidence, but it separates
    # hallucination from gold under-annotation)
    fp_in_text = 0
    for f in buckets["true_fp"]:
        text = texts.get(norm_title(f["title"]), "")
        f["in_text"] = bool(text) and f["taxon"].lower() in text.lower()
        fp_in_text += f["in_text"]
    print(f"  FP present verbatim in the paper text: {fp_in_text}/{n_fp} "
          f"({fp_in_text / max(1, n_fp):.1%})")

    fp_rank = Counter()
    for f in buckets["true_fp"]:
        tid, rank = resolve(f["taxon"])
        fp_rank[rank if tid else "unresolved"] += 1
    print(f"  FP rank profile: {dict(fp_rank.most_common())}")

    return {
        "counts": {"direction_flip": n_flip, "rank_mismatch": n_rank,
                   "false_positive": n_fp, "false_negative": n_fn,
                   "total_metric_charges": total_charges},
        "shares": {"direction_flip": round(2 * n_flip / total_charges, 4),
                   "rank_mismatch": round(2 * n_rank / total_charges, 4),
                   "false_positive": round(n_fp / total_charges, 4),
                   "false_negative": round(n_fn / total_charges, 4)},
        "fn_rank_profile": dict(fn_rank.most_common()),
        "fn_unresolved": fn_unres,
        "fn_in_text": in_text,
        "fn_total": n_fn,
        "gold_rank_profile": dict(gold_rank.most_common()),
        "fp_in_text": fp_in_text,
        "fp_rank_profile": dict(fp_rank.most_common()),
        "examples": {k: v[:40] for k, v in buckets.items()},
        "all_direction_flips": buckets["direction_flip"],
        "all_rank_mismatch": buckets["rank_mismatch"],
    }


if __name__ == "__main__":
    main()
