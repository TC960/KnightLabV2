#!/usr/bin/env python3
"""Follow-up checks on score_newgold.py. Kept separate because the main script
takes ~6 min (a TF-IDF fit per paper per direction).

Four things:
  A. Reproduce the previously-reported old-gold F1 (0.596 / 227 papers) with THIS
     harness, to prove the +0.12 jump is not a harness change.
  B. Circularity check: is the new gold drifting toward OUR predictions? If the
     taxa the new gold ADDS are ones we predicted and the ones it REMOVES are ones
     we didn't, the improvement is an artefact of how the gold was built.
  C. FN diagnosis: text presence under looser normalisation + truncation check.
  D. Verbatim examples for each error bucket.
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
from run_eval import parse_taxa, match_taxa  # noqa: E402
from score_newgold import (load, dedup_extractions, norm_title, align, prf,
                           score_rows, f1_only)  # noqa: E402

OUT = os.path.join(HERE, "score_newgold_checks.json")
res = {}


def loose(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    gold, disorder, sheet, doi2row, ex_raw, texts = load()
    by_title, ex, dups = dedup_extractions(ex_raw)

    matched = []
    for doi, cells in gold.items():
        row = doi2row.get(doi)
        if row is None:
            continue
        e = by_title.get(norm_title(row["Title"]))
        if e is not None:
            matched.append((doi, row, e, cells))
    scoreable = [m for m in matched
                 if parse_taxa(m[3].get("Enriched", "")) or parse_taxa(m[3].get("Depleted", ""))]

    # ---- A. reproduce the prior old-gold number ----------------------------
    # FINDINGS_task0_rescore.md: 227 papers with non-blank OLD gold, F1 0.596.
    # That run used the raw 348-row results file, no title dedup, and scored every
    # paper with non-blank expected_*, regardless of the new gold.
    old_all = [(e["predicted_enriched"], e["predicted_depleted"],
                e["expected_enriched"], e["expected_depleted"])
               for e in ex_raw
               if parse_taxa(e["expected_enriched"]) or parse_taxa(e["expected_depleted"])]
    s = score_rows(old_all)["combined"]
    print(f"A. OLD gold, all non-blank rows (no dedup): n={s['n']} "
          f"P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")
    old_dedup = [(e["predicted_enriched"], e["predicted_depleted"],
                  e["expected_enriched"], e["expected_depleted"])
                 for e in ex
                 if parse_taxa(e["expected_enriched"]) or parse_taxa(e["expected_depleted"])]
    s2 = score_rows(old_dedup)["combined"]
    print(f"   OLD gold, deduped papers            : n={s2['n']} "
          f"P={s2['precision']:.3f} R={s2['recall']:.3f} F1={s2['f1']:.3f}")
    res["A_old_gold_reproduction"] = {"raw_rows": s, "deduped": s2,
                                      "findings_md_reported": {"n": 227, "f1": 0.596}}

    # ---- B. circularity: where did the new gold move? ----------------------
    # Per paper per direction, compare the OLD gold cell to the NEW gold cell with
    # the same fuzzy matcher. Taxa in NEW but not OLD = "added"; OLD not NEW =
    # "removed". Then ask what share of each we predicted.
    added, removed, kept = [], [], []
    for doi, row, e, cells in scoreable:
        for d, oldcell, newcell, predcell in (
                ("enriched", e["expected_enriched"], cells.get("Enriched", ""), e["predicted_enriched"]),
                ("depleted", e["expected_depleted"], cells.get("Depleted", ""), e["predicted_depleted"])):
            o, n, p = parse_taxa(oldcell), parse_taxa(newcell), parse_taxa(predcell)
            pairs, only_new, only_old = align(n, o)   # predicted=new gold, expected=old gold
            for t in only_new:
                added.append({"doi": doi, "title": row["Title"], "direction": d, "taxon": t,
                              "we_predicted": bool(align([t], p)[0]) if p else False})
            for t in only_old:
                removed.append({"doi": doi, "title": row["Title"], "direction": d, "taxon": t,
                                "we_predicted": bool(align([t], p)[0]) if p else False})
            for a, b, _ in pairs:
                kept.append({"doi": doi, "direction": d, "taxon": a,
                             "we_predicted": bool(align([a], p)[0]) if p else False})

    def rate(lst):
        return sum(x["we_predicted"] for x in lst), len(lst)

    ra, na = rate(added)
    rr, nr = rate(removed)
    rk, nk = rate(kept)
    print(f"\nB. gold drift on the {len(scoreable)} scoreable papers")
    print(f"   kept    (in both golds) : {nk:5d}  we predicted {rk:5d} ({rk/max(1,nk):.1%})")
    print(f"   ADDED   (new gold only) : {na:5d}  we predicted {ra:5d} ({ra/max(1,na):.1%})")
    print(f"   REMOVED (old gold only) : {nr:5d}  we predicted {rr:5d} ({rr/max(1,nr):.1%})")
    res["B_gold_drift"] = {
        "kept": {"n": nk, "we_predicted": rk, "rate": round(rk / max(1, nk), 4)},
        "added": {"n": na, "we_predicted": ra, "rate": round(ra / max(1, na), 4)},
        "removed": {"n": nr, "we_predicted": rr, "rate": round(rr / max(1, nr), 4)},
        "added_examples": added[:25], "removed_examples": removed[:25],
    }

    # Counterfactual: what is the F1 if we score against OLD-gold-minus-removed, i.e.
    # isolate the effect of dropping taxa we never predicted?
    json.dump(res, open(OUT, "w"), indent=2)

    # ---- C. FN diagnosis ---------------------------------------------------
    fn_rows = []
    for doi, row, e, cells in scoreable:
        text = texts.get(norm_title(row["Title"]), "")
        lt, lloose = text.lower(), loose(text)
        for d, predcell, goldcell in (("enriched", e["predicted_enriched"], cells.get("Enriched", "")),
                                      ("depleted", e["predicted_depleted"], cells.get("Depleted", ""))):
            _, _, fn = align(parse_taxa(predcell), parse_taxa(goldcell))
            for t in fn:
                # genus of a binomial: "bacteroides fragilis" -> "bacteroides"
                head = t.split()[0] if " " in t else t
                fn_rows.append({
                    "doi": doi, "title": row["Title"], "direction": d, "taxon": t,
                    "exact_in_text": t in lt,
                    "loose_in_text": loose(t) in lloose,
                    "genus_in_text": loose(head) in lloose,
                    "text_chars": len(text),
                })
    n = len(fn_rows)
    print(f"\nC. false negatives (incl. flips/rank, before bucketing): {n}")
    for k in ("exact_in_text", "loose_in_text", "genus_in_text"):
        c = sum(r[k] for r in fn_rows)
        print(f"   {k:16s}: {c:5d} ({c/max(1,n):.1%})")
    # does FN rate track text length? (truncation hypothesis)
    bylen = defaultdict(lambda: [0, 0])
    for doi, row, e, cells in scoreable:
        text = texts.get(norm_title(row["Title"]), "")
        b = "<20k" if len(text) < 20000 else ("20-50k" if len(text) < 50000 else ">=50k")
        ng = len(parse_taxa(cells.get("Enriched", ""))) + len(parse_taxa(cells.get("Depleted", "")))
        nf = sum(1 for r in fn_rows if r["doi"] == doi)
        bylen[b][0] += nf
        bylen[b][1] += ng
    print("   FN rate by paper length:", {k: f"{v[0]}/{v[1]} = {v[0]/max(1,v[1]):.1%}"
                                          for k, v in sorted(bylen.items())})
    res["C_fn_text_presence"] = {
        "n": n,
        "exact_in_text": sum(r["exact_in_text"] for r in fn_rows),
        "loose_in_text": sum(r["loose_in_text"] for r in fn_rows),
        "genus_in_text": sum(r["genus_in_text"] for r in fn_rows),
        "fn_rate_by_text_length": {k: {"fn": v[0], "gold": v[1]} for k, v in sorted(bylen.items())},
        "examples_not_in_text": [r for r in fn_rows if not r["loose_in_text"]][:25],
    }
    json.dump(res, open(OUT, "w"), indent=2)

    # ---- D. FP sample with sentence context --------------------------------
    fp_rows = []
    for doi, row, e, cells in scoreable:
        text = texts.get(norm_title(row["Title"]), "")
        for d, predcell, goldcell in (("enriched", e["predicted_enriched"], cells.get("Enriched", "")),
                                      ("depleted", e["predicted_depleted"], cells.get("Depleted", ""))):
            _, fp, _ = align(parse_taxa(predcell), parse_taxa(goldcell))
            for t in fp:
                fp_rows.append({"doi": doi, "title": row["Title"], "direction": d, "taxon": t,
                                "in_text": t in text.lower()})
    import random
    rng = random.Random(7)
    samp = rng.sample(fp_rows, 15)
    SIG = re.compile(r"p\s*[<>=]|lda|lefse|fdr|q\s*=|significan|adjusted", re.I)
    for f in samp:
        text = texts.get(norm_title(f["title"]), "")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if f["taxon"].lower() in s.lower()]
        f["n_sentences"] = len(sents)
        f["has_stat_cue"] = any(SIG.search(s) for s in sents)
        f["sentence"] = (sents[0][:320] if sents else None)
    print("\nD. 15 sampled false positives")
    for f in samp:
        print(f"   [{f['direction'][:3]}] {f['taxon'][:38]:38s} sent={f['n_sentences']:2d} "
              f"stat={f['has_stat_cue']!s:5s} {f['title'][:44]}")
    res["D_fp_sample"] = samp
    res["D_fp_overall"] = {"n": len(fp_rows),
                           "in_text": sum(r["in_text"] for r in fp_rows)}
    json.dump(res, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
