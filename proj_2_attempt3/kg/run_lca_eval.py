#!/usr/bin/env python3
"""Driver: re-score everything with char AND LCA against the NEW gold.

  1. corpus run  (extractions_corrected.json, ~290 scoreable papers)
  2. permutation test on the LCA F1
  3. what LCA forgives: the rank-mismatch pairs char calls errors
  4. re-rank the 5 testv2 models, old gold vs new gold, char vs LCA

Results are flushed to results_lca.json after EVERY stage so a crash keeps the
partial work. CPU only, no network.
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_lca import (  # noqa: E402
    HERE, EXTRACTIONS, EVAL_RESULTS, TESTV2, RANK_DEPTH,
    LCA, align_lca, match_taxa_char, match_taxa_lca, score_rows, f1_only,
    load_new_gold, load_sheet, gold_is_blank, dedup_first, norm_title, parse_taxa, prf,
)

OUT = os.path.join(HERE, "results_lca.json")
RES = {}


def flush():
    json.dump(RES, open(OUT, "w"), indent=2)


def hdr(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def show(label, s):
    v = s["combined"]
    print(f"  {label:24s} P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f}"
          f"   (TP={v['TP']} FP={v['FP']} FN={v['FN']})")


def main():
    R = LCA()
    print("taxdump loaded:", R.ok)
    if not R.ok:
        sys.exit("no taxdump at ~/.ncbi-taxdump -- cannot run the LCA metric")
    char = lambda p, e: match_taxa_char(p, e)                      # noqa: E731
    lca = lambda p, e: match_taxa_lca(p, e, R)                     # noqa: E731
    lca_deep = lambda p, e: match_taxa_lca(p, e, R, deepest=True)  # noqa: E731

    gold, disorder = load_new_gold()
    doi2row, title2doi = load_sheet()
    by_title, _ = dedup_first(json.load(open(EXTRACTIONS)))

    # ------------------------------------------------ 1. build the scoreable set
    hdr("1. RECONCILIATION  (same join path as score_newgold.py)")
    matched, no_extraction = [], 0
    for doi, cells in gold.items():
        row = doi2row.get(doi)
        if row is None:
            no_extraction += 1
            continue
        e = by_title.get(norm_title(row["Title"]))
        if e is None:
            no_extraction += 1
        else:
            matched.append((doi, row, e, cells))
    blank = [m for m in matched if gold_is_blank(m[3])]
    scoreable = [m for m in matched if not gold_is_blank(m[3])]
    print(f"  gold DOIs {len(gold)} | joined to an extraction {len(matched)} | "
          f"no extraction {no_extraction}")
    print(f"  blank on BOTH gold cells: {len(blank)}  (EXCLUDED -- scoring against an "
          f"empty cell manufactures FPs)")
    print(f"  SCOREABLE: {len(scoreable)}")
    RES["reconciliation"] = {"gold_dois": len(gold), "joined": len(matched),
                             "no_extraction": no_extraction,
                             "blank_excluded": len(blank), "scoreable": len(scoreable)}
    flush()

    rows = [(e["predicted_enriched"], e["predicted_depleted"],
             c.get("Enriched", ""), c.get("Depleted", "")) for _, _, e, c in scoreable]

    # warm the resolver on every string that will ever be compared
    allnames = set()
    for pe, pd_, ge, gd in rows:
        for s in (pe, pd_, ge, gd):
            allnames.update(parse_taxa(s))
    R.warm(allnames)
    resolved = sum(1 for n in allnames if R.info(n)[0])
    print(f"  distinct taxon strings {len(allnames)} | resolve to an NCBI taxid "
          f"{resolved} ({resolved/len(allnames):.1%})")
    RES["reconciliation"]["distinct_taxon_strings"] = len(allnames)
    RES["reconciliation"]["resolved_to_taxid"] = resolved

    # ------------------------------------------------ 2. side-by-side scores
    hdr(f"2. NEW GOLD, {len(rows)} scoreable papers -- char vs LCA")
    s_char = score_rows(rows, char)
    RES["char"] = s_char
    flush()
    s_lca = score_rows(rows, lca)
    RES["lca"] = s_lca
    flush()
    s_deep = score_rows(rows, lca_deep)
    RES["lca_deepest_tiebreak"] = s_deep
    for k in ("enriched", "depleted", "combined"):
        a, b = s_char[k], s_lca[k]
        print(f"  {k:9s} char  P={a['precision']:.4f} R={a['recall']:.4f} F1={a['f1']:.4f}"
              f"  (TP={a['TP']} FP={a['FP']} FN={a['FN']})")
        print(f"  {'':9s} LCA   P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f}"
              f"  (TP={b['TP']} FP={b['FP']} FN={b['FN']})")
    print(f"\n  delta F1 = {s_lca['combined']['f1'] - s_char['combined']['f1']:+.4f}")
    print(f"  sensitivity: LCA with deepest-expected tie-break -> F1 "
          f"{s_deep['combined']['f1']:.4f} (vs {s_lca['combined']['f1']:.4f} document order)")
    RES["delta_f1"] = round(s_lca["combined"]["f1"] - s_char["combined"]["f1"], 4)
    flush()

    # guard sensitivity: refuse nested pairs shallower than <rank>
    hdr("2b. HOW MUCH OF THE LIFT IS HIGH-LEVEL NESTING?  (--min-lca-rank sweep)")
    sweep = {}
    for mr in (None, "phylum", "class", "order", "family", "genus"):
        Rg = LCA(min_rank=mr)
        Rg._info, Rg._lin = R._info, R._lin      # share the caches, skip the reload
        f = score_rows(rows, lambda p, e, _r=Rg: match_taxa_lca(p, e, _r))["combined"]
        sweep[mr or "none"] = f
        print(f"  reject nesting above {str(mr):7s} -> P={f['precision']:.4f} "
              f"R={f['recall']:.4f} F1={f['f1']:.4f}  (TP={f['TP']})")
    RES["min_rank_sweep"] = sweep
    flush()

    # ------------------------------------------------ 3. permutation test
    hdr("3. PERMUTATION TEST on the LCA F1 (shuffle gold -> paper)")
    obs = s_lca["combined"]["f1"]
    rng = random.Random(0)
    preds = [(a, b) for a, b, _, _ in rows]
    golds = [(c, d) for _, _, c, d in rows]
    null = []
    for it in range(1000):
        idx = list(range(len(golds)))
        rng.shuffle(idx)
        null.append(f1_only([(preds[i][0], preds[i][1], golds[idx[i]][0], golds[idx[i]][1])
                             for i in range(len(idx))], lca))
        if (it + 1) % 100 == 0:
            print(f"    {it+1}/1000 draws, running null mean {sum(null)/len(null):.4f}", flush=True)
            RES["permutation_lca"] = {"observed_f1": obs, "n_perm_so_far": len(null),
                                      "null_mean": round(sum(null) / len(null), 4)}
            flush()
    nm = sum(null) / len(null)
    sd = (sum((x - nm) ** 2 for x in null) / len(null)) ** 0.5
    ge = sum(1 for x in null if x >= obs)
    RES["permutation_lca"] = {"observed_f1": obs, "null_mean": round(nm, 4),
                              "null_sd": round(sd, 4), "null_max": round(max(null), 4),
                              "p": round((ge + 1) / 1001, 4), "n_perm": 1000,
                              "n_ge_observed": ge}
    print(f"  observed F1 {obs:.4f} | null mean {nm:.4f} (sd {sd:.4f}) | "
          f"null max {max(null):.4f} | p = {(ge+1)/1001:.4f}")
    flush()

    # ------------------------------------------------ 4. what LCA forgives
    hdr("4. WHAT LCA FORGIVES -- the pairs char scores as two separate errors")
    forgiven = []
    for doi, row, e, cells in scoreable:
        for direction, pk, gk in (("enriched", "predicted_enriched", "Enriched"),
                                  ("depleted", "predicted_depleted", "Depleted")):
            p, g = parse_taxa(e[pk]), parse_taxa(cells.get(gk, ""))
            pairs, _, _ = align_lca(p, g, R)
            for pred, exp, rule, sim in pairs:
                if rule != "lca":
                    continue
                ptid, psci, prank = R.info(pred)
                gtid, gsci, grank = R.info(exp)
                if ptid == gtid:
                    rel = "synonym"          # same taxid, different surface string
                else:
                    rel = "we_coarser" if ptid in R.lineage(gtid) else "we_finer"
                forgiven.append({"doi": doi, "title": row["Title"], "direction": direction,
                                 "predicted": pred, "pred_sci": psci, "pred_rank": prank,
                                 "gold": exp, "gold_sci": gsci, "gold_rank": grank,
                                 "relation": rel, "char_sim": sim,
                                 "rank_gap": abs(RANK_DEPTH.get(prank, 99) - RANK_DEPTH.get(grank, 99))
                                 if rel != "synonym" else 0})
    byrel = Counter(f["relation"] for f in forgiven)
    print(f"  total pairs rescued by LCA: {len(forgiven)}")
    # NB: a rescued prediction always removes one FP, but removes an FN only if no
    # OTHER prediction had already covered that gold taxon. Empirically most had
    # (FP -112 vs FN -9, see lca_checks.py), so this is a PRECISION correction, not
    # a recall one. Do not assume 2 charges per pair.
    print(f"  -> removes {len(forgiven)} FP charges; the FN removal is smaller and is "
          f"measured directly in lca_checks.py")
    for k, v in byrel.most_common():
        print(f"    {k:12s} {v:4d}  ({v/len(forgiven):5.1%})")
    pairsrank = Counter((f["pred_rank"], f["gold_rank"]) for f in forgiven
                        if f["relation"] != "synonym")
    print("\n  rank transitions (predicted_rank -> gold_rank), top 15:")
    for (pr, gr), n in pairsrank.most_common(15):
        print(f"    {str(pr):10s} -> {str(gr):10s} {n:4d}")
    RES["forgiven"] = {"total": len(forgiven), "fp_charges_removed": len(forgiven),
                       "by_relation": dict(byrel),
                       "rank_transitions": {f"{a}->{b}": n for (a, b), n in pairsrank.most_common()},
                       "rank_gap_hist": dict(Counter(f["rank_gap"] for f in forgiven)),
                       "all": forgiven}
    print("\n  --- 10 verbatim examples ---")
    ex_pool = [f for f in forgiven if f["relation"] != "synonym"]
    rng2 = random.Random(7)
    sample = rng2.sample(ex_pool, min(10, len(ex_pool)))
    for f in sample:
        print(f"    [{f['relation']:10s}] {f['direction']:8s} pred '{f['predicted']}' "
              f"({f['pred_rank']}) vs gold '{f['gold']}' ({f['gold_rank']})  "
              f"char_sim={f['char_sim']}  -- {f['title'][:52]}")
    RES["forgiven"]["examples_10"] = sample
    flush()

    # ------------------------------------------------ 5. re-rank the models
    hdr("5. MODEL RE-RANKING on testv2 (15 papers)")
    tv2 = json.load(open(TESTV2))
    tv2_dois = {}
    for p in tv2:
        d = title2doi.get(norm_title(p["title"]))
        if d:
            tv2_dois[norm_title(p["title"])] = d
    in_gold = {t: d for t, d in tv2_dois.items() if d in gold}
    nonblank = {t: d for t, d in in_gold.items() if not gold_is_blank(gold[d])}
    print(f"  testv2 papers: {len(tv2)} | matched to a DOI: {len(tv2_dois)} | "
          f"present in NEW gold: {len(in_gold)} | non-blank new gold: {len(nonblank)}")
    RES["testv2_overlap"] = {"testv2_n": len(tv2), "doi_matched": len(tv2_dois),
                             "in_new_gold": len(in_gold), "scoreable": len(nonblank)}
    flush()

    files = sorted(f for f in os.listdir(EVAL_RESULTS)
                   if f.endswith("__samgated-v1__testv2.json"))
    board = []
    for fn in files:
        model = fn.split("__")[0]
        recs = json.load(open(os.path.join(EVAL_RESULTS, fn)))
        old_rows, new_rows, n = [], [], 0
        for r in recs:
            k = norm_title(r["title"])
            old_rows.append((r["predicted_enriched"], r["predicted_depleted"],
                             r["expected_enriched"], r["expected_depleted"]))
            d = nonblank.get(k)
            if d:
                new_rows.append((r["predicted_enriched"], r["predicted_depleted"],
                                 gold[d].get("Enriched", ""), gold[d].get("Depleted", "")))
                n += 1
        # restrict OLD-gold scoring to the SAME papers, else it is a subset artefact
        old_same = [old_rows[i] for i, r in enumerate(recs)
                    if norm_title(r["title"]) in nonblank]
        entry = {"model": model, "n_papers_scored": n,
                 "old_gold_char": score_rows(old_same, char)["combined"],
                 "old_gold_lca": score_rows(old_same, lca)["combined"],
                 "new_gold_char": score_rows(new_rows, char)["combined"],
                 "new_gold_lca": score_rows(new_rows, lca)["combined"]}
        board.append(entry)
        print(f"\n  {model}   (n={n} papers)")
        for lbl in ("old_gold_char", "old_gold_lca", "new_gold_char", "new_gold_lca"):
            v = entry[lbl]
            print(f"     {lbl:15s} P={v['precision']:.4f} R={v['recall']:.4f} "
                  f"F1={v['f1']:.4f}  (TP={v['TP']} FP={v['FP']} FN={v['FN']})")
        RES["leaderboard"] = board
        flush()

    hdr("5b. RANK ORDER UNDER EACH (metric, gold) COMBINATION")
    for lbl in ("old_gold_char", "old_gold_lca", "new_gold_char", "new_gold_lca"):
        order = sorted(board, key=lambda e: -e[lbl]["f1"])
        print(f"  {lbl}:")
        for i, e in enumerate(order, 1):
            v = e[lbl]
            print(f"     {i}. {e['model']:24s} F1 {v['f1']:.4f}  "
                  f"(P {v['precision']:.4f} / R {v['recall']:.4f})")
        RES.setdefault("rank_orders", {})[lbl] = [e["model"] for e in order]
    flush()
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
