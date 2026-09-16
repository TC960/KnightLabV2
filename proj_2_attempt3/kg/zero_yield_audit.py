#!/usr/bin/env python3
"""Why do 42 screened papers contribute NOTHING to the graph?

Every fidelity instrument in this directory scores edges that EXIST:
reading fidelity (`audit_direction_witness.py`, >=86.6%), taxon mention rate
(`verify_taxon_mentions.py`, 99.57%), agreement with Disbiome/Peryton. The
mirror question has never been asked -- **are there relations stated in a
paper that produced no edge at all?** That is extractor *recall*, and the only
recall number the project has ever had (F1 ~0.59) is scored against the
in-house gold standard, which is itself under audit and known unreliable.

This scores recall against THE PAPERS' OWN SENTENCES instead, so it depends on
neither the gold nor the curated databases -- the same move that made the
direction audit the best fidelity number here.

The funnel, all deterministic:

    348 extraction rows (extractions_corrected.json)
 ->  325 survive the corpus screen (extractions_screened.json)
 ->  313 after build_kg's title deduplication
 ->  271 contribute at least one edge            = 86.6% paper yield
 ->   42 contribute nothing                      <- the subject of this file

The 42 split three ways, and only the third is a candidate error:

  A. 10 rows are dedup twins whose surviving copy IS in the graph. Their
     content is represented; they are not losses. (Counted before dedup.)
  B.  9 papers have no relation-bearing sentence at all -- `relation_sentences`
     found nothing naming a resolved taxon near a direction cue, so there was
     nothing for the extractor to find.
  C. 33 papers DO have relation-bearing sentences. These are the only ones
     where "returned nothing" could be a miss.

For C this script reports two independent things:

  1. A deterministic provenance screen, reusing the CITATION / THIRD_PARTY /
     RESULT_CUE / CONTROL_FRAME regexes already written and permutation-tested
     in `audit_direction_witness.py`. A sentence is an extraction CANDIDATE
     only if it carries a result cue AND frames a comparison against controls
     AND carries no citation or third-party attribution. Anything else cannot
     support a disease-vs-healthy-control edge, which is the only edge the
     schema admits.
  2. The per-disease yield rate, with a paper-level permutation test using a
     max-statistic so the multiple comparison across diseases is controlled.

The verdicts in `zero_yield_verdicts.json` come from LLM adjudication of the
sentences (four independent batches) and are a judgement, not a measurement --
they are reported alongside the deterministic screen, never in place of it.
Two papers entered the adjudication twice as dedup twins; their verdicts are
used as an inter-copy consistency check on the adjudicator.

Run:  python3 zero_yield_audit.py
"""
import collections
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_kg import norm_title, parse_taxa  # noqa: E402

NPERM = 20000
SEED = 20260916


def _witness():
    """Import the provenance regexes without running that script's main()."""
    spec = importlib.util.spec_from_file_location(
        "_wit", os.path.join(HERE, "audit_direction_witness.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wit"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def load():
    scr = json.load(open(os.path.join(HERE, "extractions_screened.json")))
    graph = json.load(open(os.path.join(HERE, "graph.json")))
    rs = json.load(open(os.path.join(HERE, "relation_sentences_clean.json")))["papers"]
    return scr, graph, rs


def dedup(scr):
    """Replay build_kg.dedup_rows' choice so the denominator matches the graph."""
    groups = collections.defaultdict(list)
    for r in scr:
        groups[norm_title(r["title"])].append(r)
    kept = []
    for k in sorted(groups):
        g = sorted(groups[k],
                   key=lambda r: (-(len(parse_taxa(r.get("predicted_enriched"))) +
                                    len(parse_taxa(r.get("predicted_depleted")))),
                                  r.get("title", "")))
        kept.append(g[0])
    return kept


def main():
    scr, graph, rs = load()
    wit = _witness()
    in_graph = {norm_title(p["title"]) for p in graph["papers"]}
    rs_by_key = {norm_title(t): v for t, v in rs.items()}

    rows = dedup(scr)
    contributing = [r for r in rows if norm_title(r["title"]) in in_graph]
    zero = [r for r in rows if norm_title(r["title"]) not in in_graph]

    print("=" * 72)
    print("PAPER-LEVEL YIELD")
    print("=" * 72)
    print(f"  extraction rows          : {len(scr)} screened")
    print(f"  after title dedup        : {len(rows)}")
    print(f"  contributing >=1 edge    : {len(contributing)}")
    print(f"  zero-yield               : {len(zero)}")
    print(f"  paper yield rate         : {100*len(contributing)/len(rows):.1f}%")

    # --- was anything LOST between extraction and the graph? -----------------
    lost = [r for r in zero
            if len(parse_taxa(r.get("predicted_enriched"))) +
               len(parse_taxa(r.get("predicted_depleted"))) > 0]
    print(f"\n  zero-yield papers whose extractor DID return taxa: {len(lost)}")
    print("  (non-zero here would mean build_kg silently drops extractions)")
    for r in lost:
        print("    LOST:", r["title"][:70])

    # --- provenance screen over the zero-yield papers' own sentences ---------
    print("\n" + "=" * 72)
    print("ZERO-YIELD PAPERS: WHAT DO THEIR OWN SENTENCES SAY?")
    print("=" * 72)
    nosent, detail = 0, []
    for r in zero:
        v = rs_by_key.get(norm_title(r["title"]))
        sents = [s["s"] for s in (v["kept"] if v else []) if s.get("taxa")]
        if not sents:
            nosent += 1
            continue
        cand = own = bg = 0
        for s in sents:
            if wit.CITATION.search(s) or wit.THIRD_PARTY.search(s):
                bg += 1
                continue
            res = bool(wit.RESULT_CUE.search(s))
            own += res
            cand += res and bool(wit.CONTROL_FRAME.search(s))
        detail.append({"title": r["title"], "disease": r.get("disease"),
                       "n": len(sents), "background": bg, "own_result": own,
                       "candidate": cand})
    detail.sort(key=lambda d: -d["candidate"])
    tot = sum(d["n"] for d in detail)
    print(f"  papers with NO relation-bearing sentence : {nosent} (nothing to extract)")
    print(f"  papers with relation-bearing sentences   : {len(detail)}")
    print(f"  their sentences                          : {tot}")
    print(f"    cited / third-party attributed         : {sum(d['background'] for d in detail)}")
    print(f"    own result (result cue, no citation)   : {sum(d['own_result'] for d in detail)}")
    print(f"    own result AND control-framed          : {sum(d['candidate'] for d in detail)}"
          "   <- only these could support an edge")
    print(f"  papers with >=1 candidate sentence       : "
          f"{sum(1 for d in detail if d['candidate'])} / {len(detail)}")
    print()
    for d in detail[:12]:
        print(f"    cand={d['candidate']:3d} own={d['own_result']:3d} "
              f"bg={d['background']:3d} n={d['n']:3d}  {d['title'][:58]}")

    # --- is zero-yield concentrated in a disease? ---------------------------
    print("\n" + "=" * 72)
    print("IS ZERO-YIELD CONCENTRATED BY DISEASE?")
    print("=" * 72)
    dis = [str(r.get("disease") or "UNLABELLED") for r in rows]
    yld = [norm_title(r["title"]) in in_graph for r in rows]
    idx = collections.defaultdict(list)
    for i, d in enumerate(dis):
        idx[d].append(i)
    big = [d for d in idx if len(idx[d]) >= 5]
    print(f"  {'disease':40s} {'n':>4s} {'edges':>6s} {'yield':>6s}")
    for d in sorted(idx, key=lambda d: -len(idx[d])):
        n = len(idx[d])
        if n < 3:
            continue
        y = sum(yld[i] for i in idx[d])
        print(f"  {d[:40]:40s} {n:4d} {y:6d} {100*y/n:5.0f}%")

    def stat(labels):
        return max(1 - sum(labels[i] for i in idx[d]) / len(idx[d]) for d in big)

    obs = stat(yld)
    worst = [d for d in big if 1 - sum(yld[i] for i in idx[d]) / len(idx[d]) == obs]
    rng = random.Random(SEED)
    lab, hits = list(yld), 0
    for _ in range(NPERM):
        rng.shuffle(lab)
        if stat(lab) >= obs:
            hits += 1
    p = (hits + 1) / (NPERM + 1)
    print(f"\n  worst per-disease zero-yield rate : {obs:.2f}  {worst}")
    print(f"  diseases tested (n>=5)            : {len(big)}")
    print(f"  paper-level permutation, max-stat : p = {p:.4f}"
          f"  ({'NULL' if p > 0.05 else 'SURVIVES'})")
    print("  Shuffling is at the PAPER level and the statistic is the maximum")
    print("  over diseases, so the multiple comparison is already controlled.")

    out = {"papers_screened": len(scr), "papers_after_dedup": len(rows),
           "papers_contributing": len(contributing), "papers_zero_yield": len(zero),
           "yield_rate": len(contributing) / len(rows),
           "zero_yield_extractions_lost_in_build": len(lost),
           "zero_yield_no_relation_sentence": nosent,
           "zero_yield_with_sentences": len(detail),
           "sentences_total": tot,
           "sentences_background": sum(d["background"] for d in detail),
           "sentences_own_result": sum(d["own_result"] for d in detail),
           "sentences_own_and_control_framed": sum(d["candidate"] for d in detail),
           "disease_permutation_p": p, "disease_worst": worst,
           "disease_worst_rate": obs, "per_paper": detail}
    with open(os.path.join(HERE, "zero_yield.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote zero_yield.json")


if __name__ == "__main__":
    main()
