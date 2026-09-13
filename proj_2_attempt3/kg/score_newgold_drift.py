#!/usr/bin/env python3
"""Decompose the old-gold -> new-gold F1 jump, and test whether it is earned.

score_newgold_checks.py showed the new gold REMOVED 647 old-gold taxa of which we
had predicted only 29.7%, while our overall recall is ~76%. Removal is therefore
strongly non-random in the direction that flatters the extractor. Three questions:

  E. How much of the +0.12 F1 is removals vs additions? Score against the
     intersection gold (old minus removed) as the midpoint.
  F. Were the removals justified? If a removed taxon does not appear in the paper
     text, or appears with no statistical cue, dropping it was cleanup. If it
     appears next to a p-value/LDA score, the new gold may have discarded real
     findings -- which would make the higher F1 partly an easier-gold artefact.
  G. Abbreviation failures: "e. coli" vs "escherichia coli" scores below the char
     threshold, so one taxon becomes 1 FP + 1 FN. Count them.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
from run_eval import parse_taxa  # noqa: E402
from score_newgold import (load, dedup_extractions, norm_title, align,
                           score_rows, f1_only)  # noqa: E402

OUT = os.path.join(HERE, "score_newgold_drift.json")
SIG = re.compile(r"p\s*[<>=]\s*0|p\s*[<>=]\s*\.|lda|lefse|fdr|q\s*[<>=]|significantl|adjusted p", re.I)
res = {}


def main():
    gold, disorder, sheet, doi2row, ex_raw, texts = load()
    by_title, ex, dups = dedup_extractions(ex_raw)
    scoreable = []
    for doi, cells in gold.items():
        row = doi2row.get(doi)
        if not row:
            continue
        e = by_title.get(norm_title(row["Title"]))
        if e and (parse_taxa(cells.get("Enriched", "")) or parse_taxa(cells.get("Depleted", ""))):
            scoreable.append((doi, row, e, cells))

    # ---- E. three-gold decomposition ---------------------------------------
    rows_old, rows_mid, rows_new = [], [], []
    removed_detail, added_detail = [], []
    for doi, row, e, cells in scoreable:
        cell_mid = {}
        for d, oldcell, newcell in (("Enriched", e["expected_enriched"], cells.get("Enriched", "")),
                                    ("Depleted", e["expected_depleted"], cells.get("Depleted", ""))):
            o, n = parse_taxa(oldcell), parse_taxa(newcell)
            pairs, only_new, only_old = align(n, o)
            kept_old = [b for _, b, _ in pairs]
            cell_mid[d] = "; ".join(kept_old)
            for t in only_old:
                removed_detail.append({"doi": doi, "title": row["Title"], "direction": d, "taxon": t})
            for t in only_new:
                added_detail.append({"doi": doi, "title": row["Title"], "direction": d, "taxon": t})
        rows_old.append((e["predicted_enriched"], e["predicted_depleted"],
                         e["expected_enriched"], e["expected_depleted"]))
        rows_mid.append((e["predicted_enriched"], e["predicted_depleted"],
                         cell_mid["Enriched"], cell_mid["Depleted"]))
        rows_new.append((e["predicted_enriched"], e["predicted_depleted"],
                         cells.get("Enriched", ""), cells.get("Depleted", "")))

    print(f"E. gold decomposition on the same {len(rows_old)} papers")
    step = {}
    for lbl, rws in (("OLD gold", rows_old),
                     ("MID = old ∩ new (removals applied)", rows_mid),
                     ("NEW gold (removals + additions)", rows_new)):
        v = score_rows(rws)["combined"]
        step[lbl] = v
        print(f"   {lbl:36s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"(gold taxa {v['TP']+v['FN']})")
    f_old = step["OLD gold"]["f1"]
    f_mid = step["MID = old ∩ new (removals applied)"]["f1"]
    f_new = step["NEW gold (removals + additions)"]["f1"]
    print(f"   removals contribute {f_mid - f_old:+.3f}; additions contribute {f_new - f_mid:+.3f}; "
          f"total {f_new - f_old:+.3f}")
    res["E_decomposition"] = {"steps": step,
                              "removal_effect": round(f_mid - f_old, 4),
                              "addition_effect": round(f_new - f_mid, 4),
                              "total": round(f_new - f_old, 4)}
    json.dump(res, open(OUT, "w"), indent=2)

    # ---- F. were the removals justified? -----------------------------------
    def evidence(items):
        out = {"n": len(items), "in_text": 0, "stat_cue": 0, "no_mention": 0}
        for it in items:
            text = texts.get(norm_title(it["title"]), "")
            sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if it["taxon"].lower() in s.lower()]
            it["n_sentences"] = len(sents)
            it["stat_cue"] = any(SIG.search(s) for s in sents)
            if sents:
                out["in_text"] += 1
                it["sentence"] = sents[0][:300]
            else:
                out["no_mention"] += 1
            out["stat_cue"] += it["stat_cue"]
        return out

    ev_rm = evidence(removed_detail)
    ev_add = evidence(added_detail)
    print(f"\nF. evidence for the taxa the new gold REMOVED (n={ev_rm['n']})")
    print(f"   mentioned in our text : {ev_rm['in_text']} ({ev_rm['in_text']/max(1,ev_rm['n']):.1%})")
    print(f"   with a statistical cue: {ev_rm['stat_cue']} ({ev_rm['stat_cue']/max(1,ev_rm['n']):.1%})")
    print(f"   never mentioned       : {ev_rm['no_mention']} ({ev_rm['no_mention']/max(1,ev_rm['n']):.1%})")
    print(f"   for contrast, taxa ADDED (n={ev_add['n']}): in_text "
          f"{ev_add['in_text']/max(1,ev_add['n']):.1%}, stat cue {ev_add['stat_cue']/max(1,ev_add['n']):.1%}")
    print("   sample removals that DO carry a statistical cue:")
    shown = 0
    for it in removed_detail:
        if it.get("stat_cue") and shown < 6:
            print(f"     {it['taxon'][:30]:30s} [{it['direction'][:3]}] {it['title'][:40]}")
            print(f"       \"{it.get('sentence','')[:170]}\"")
            shown += 1
    res["F_removal_evidence"] = {"removed": ev_rm, "added": ev_add,
                                 "removed_examples": removed_detail[:40]}
    json.dump(res, open(OUT, "w"), indent=2)

    # ---- G. abbreviation / formatting FP+FN pairs --------------------------
    ABBR = re.compile(r"^([a-z])\.?\s+([a-z]{3,})$")
    pairs = []
    for doi, row, e, cells in scoreable:
        for d, predcell, goldcell in (("enriched", e["predicted_enriched"], cells.get("Enriched", "")),
                                      ("depleted", e["predicted_depleted"], cells.get("Depleted", ""))):
            _, fp, fn = align(parse_taxa(predcell), parse_taxa(goldcell))
            for p in fp:
                m = ABBR.match(p)
                if not m:
                    continue
                init, sp = m.group(1), m.group(2)
                for g in fn:
                    if " " in g and g.split()[0][0] == init and g.split()[-1] == sp:
                        pairs.append({"doi": doi, "title": row["Title"], "direction": d,
                                      "predicted": p, "gold": g})
                        break
    print(f"\nG. abbreviated-binomial FP/FN pairs the char metric splits: {len(pairs)}")
    for p in pairs[:12]:
        print(f"   pred '{p['predicted']}' vs gold '{p['gold']}'  [{p['direction'][:3]}] {p['title'][:40]}")
    # how many predictions are abbreviated at all
    n_abbr = sum(1 for doi, row, e, c in scoreable
                 for d in ("predicted_enriched", "predicted_depleted")
                 for t in parse_taxa(e[d]) if ABBR.match(t))
    print(f"   total abbreviated predictions emitted: {n_abbr}")
    res["G_abbreviation"] = {"recoverable_pairs": len(pairs), "abbrev_predictions": n_abbr,
                             "examples": pairs[:25]}

    # ---- viruses / phages: a whole class the gold does not annotate ---------
    VIR = re.compile(r"phage|virus|viridae|virales|siphovir|myovir|podovir", re.I)
    vfp = vgold = 0
    vpapers = Counter()
    for doi, row, e, cells in scoreable:
        for d, predcell, goldcell in (("enriched", e["predicted_enriched"], cells.get("Enriched", "")),
                                      ("depleted", e["predicted_depleted"], cells.get("Depleted", ""))):
            _, fp, _ = align(parse_taxa(predcell), parse_taxa(goldcell))
            k = sum(1 for t in fp if VIR.search(t))
            vfp += k
            if k:
                vpapers[row["Title"][:60]] += k
            vgold += sum(1 for t in parse_taxa(goldcell) if VIR.search(t))
    print(f"\n   virus/phage taxa: {vfp} false positives, {vgold} in the whole new gold")
    print("   concentrated in:", vpapers.most_common(5))
    res["G_viruses"] = {"fp": vfp, "in_gold": vgold, "top_papers": vpapers.most_common(5)}
    json.dump(res, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
