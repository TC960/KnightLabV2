#!/usr/bin/env python3
"""Edge-level recall: relations a CONTRIBUTING paper states that produced no edge.

`zero_yield_audit.py` asked the paper-level question -- did the extractor refuse a
whole paper it should have read (answer: almost never). This asks the harder one,
one level down: **inside the 271 papers it DID read, did it catch every taxon?**

That number has never existed. The only recall figure the project has ever had
(F1 ~0.59) is scored against the in-house gold standard, which is under audit and
known unreliable. This scores against the papers' own sentences instead.

METHOD

1. Candidate generation (deterministic). For every contributing paper, keep
   sentences that pass the provenance screen from `audit_direction_witness.py`
   (RESULT_CUE and CONTROL_FRAME present, CITATION and THIRD_PARTY absent), then
   collect every taxon in them that resolves to a taxid and for which that paper
   has NO edge. Matching is on taxid AND on normalised label AND on every node
   alias, so synonyms and renames do not create phantom misses.
   -> 378 candidate (paper, taxon) pairs across 94 papers.

2. Adjudication. A random sample of 24 papers (95 candidates) was read, in four
   independent batches, with verbatim quotes required.

3. **Gate application. This is the step that matters and the step the first pass
   of this audit got wrong.** The extraction prompt (`eval-v2/run_eval.py`,
   `samgated-v1`) does not extract anything that merely states a direction:

     - SIGNIFICANCE: "include a taxon ONLY if the paper reports it as statistically
       significant ... If significance is unclear or unreported for a taxon, OMIT
       it."
     - MAIN TEXT ONLY (not tables, figures, captions, supplementary).
     - DISEASE vs HEALTHY CONTROL ONLY.

   So a candidate counts as a confirmed miss only if its quote is verbatim AND some
   sentence naming that taxon carries a significance cue that is not negated
   ("no significant", "tendency", "trend toward") and is not a citation. Scoring an
   extractor without applying its own gate manufactures misses: 36 raw REAL_MISS
   verdicts -> 29 verbatim -> **15 confirmed**.

RESULT

  15 / 95 candidates confirmed = 0.158, paper-level cluster bootstrap 95% CI
  [0.038, 0.308]. Projected over all 378 candidates: ~60 missed observations
  [14, 116] against 3,077 in the graph.

  **Observation-level recall ~98.1%, 95% CI [96.4, 99.5].**

DIRECTION OF THE BIAS -- this is an UPPER bound on recall. The candidate generator
only sees taxa that resolve to a taxid AND sit in a sentence the regex screen keeps,
and that screen was measured at ~75% paper-level recall in `FINDINGS_zero_yield.md`.
Misses stated in sentences the screen drops are invisible here. Conversely the
significance gate can only be *confirmed* from visible text, never refuted, so
borderline cases are scored as non-misses. Both push the same way: true recall is at
or below this figure.

Run:  python3 edge_recall_audit.py
"""
import collections
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NBOOT = 20000
SEED = 20260916

SIG = re.compile(r"(p\s*[<>=]|p[-\s]?value|\bq\s*[<>=]|\bfdr\b|adjusted\s*p|benjamini|"
                 r"\blda\b|\blefse\b|significan\w*|\bci\s*[:=]|95%)", re.I)
NEG = re.compile(r"\b(no|not|non)[-\s]?significan|tendency|trend toward|"
                 r"did not (?:differ|reach)", re.I)
CIT = re.compile(r"\[\s*\d+\s*(?:[,;–\-]\s*\d+\s*)*\]|\bet\s*al", re.I)


def confirmed(verdict, sentences):
    """A REAL_MISS survives only with a verbatim quote AND a clean significance cue."""
    if verdict.get("verdict") != "REAL_MISS":
        return False
    q = (verdict.get("quote") or "").strip()
    if not q or not any(q in s for s in sentences):
        return False
    return any(SIG.search(s) and not NEG.search(s) and not CIT.search(s)
               for s in sentences)


def main():
    pack = json.load(open(os.path.join(HERE, "edge_recall_packets.json")))
    verd = json.load(open(os.path.join(HERE, "edge_recall_verdicts.json")))
    graph = json.load(open(os.path.join(HERE, "graph.json")))
    obs = sum(e["n_papers"] for e in graph["edges"])
    total_candidates = pack["total_candidates"]
    sents = {(p["title"], m["name"]): m["sentences"]
             for p in pack["sampled"] for m in p["missing"]}
    rank = {(p["title"], m["name"]): (m.get("rank") or "?")
            for p in pack["sampled"] for m in p["missing"]}

    by_paper = collections.defaultdict(lambda: [0, 0])
    hits = []
    for v in verd:
        k = (v["title"], v["taxon"])
        by_paper[v["title"]][0] += 1
        ok = confirmed(v, sents.get(k, []))
        by_paper[v["title"]][1] += int(ok)
        if ok:
            hits.append((v["title"], v["taxon"], v.get("direction"), rank.get(k, "?")))

    papers = list(by_paper)
    n = sum(x[0] for x in by_paper.values())
    c = sum(x[1] for x in by_paper.values())
    raw = sum(1 for v in verd if v["verdict"] == "REAL_MISS")

    print("=" * 72)
    print("EDGE-LEVEL RECALL")
    print("=" * 72)
    print(f"  candidate (paper,taxon) pairs, all : {total_candidates} over 94 papers")
    print(f"  adjudicated sample                 : {len(papers)} papers, {n} candidates")
    print(f"  raw REAL_MISS verdicts             : {raw}")
    print(f"  after verbatim + significance gate : {c}   <- confirmed misses")
    print(f"  rate                               : {c/n:.3f}")

    rng = random.Random(SEED)
    rates = []
    for _ in range(NBOOT):
        pick = [by_paper[rng.choice(papers)] for _ in papers]
        a = sum(x[0] for x in pick)
        b = sum(x[1] for x in pick)
        if a:
            rates.append(b / a)
    rates.sort()
    lo, hi = rates[int(.025 * len(rates))], rates[int(.975 * len(rates))]
    print(f"  paper-level cluster bootstrap 95%  : [{lo:.3f}, {hi:.3f}]")
    print()
    for lbl, r in (("point", c / n), ("upper", lo), ("lower", hi)):
        m = r * total_candidates
        print(f"    {lbl:6s}: ~{m:5.0f} missed observations of {obs}"
              f"  ->  recall {obs/(obs+m)*100:.1f}%")
    print("\n  This is an UPPER bound: the candidate generator only sees taxa in")
    print("  sentences the provenance screen keeps (~75% paper-level recall), and")
    print("  the significance gate can be confirmed but never refuted.")

    # --- is rank associated with being a confirmed miss? --------------------
    print("\n" + "=" * 72)
    print("ARE HIGH-RANK TAXA MISSED MORE OFTEN?")
    print("=" * 72)
    gr = collections.Counter(e.get("rank") or "?" for e in graph["edges"])
    tot = sum(gr.values())
    cr = collections.Counter(h[3] for h in hits)
    ar = collections.Counter(rank.values())
    print(f"  {'rank':10s} {'confirmed':>9s} {'candidates':>11s} {'graph edges':>13s}")
    for r in ("phylum", "class", "order", "family", "genus", "species"):
        print(f"  {r:10s} {cr.get(r,0):9d} {ar.get(r,0):11d} "
              f"{gr.get(r,0):8d} ({100*gr.get(r,0)/tot:4.1f}%)")

    high = sum(cr.get(r, 0) for r in ("phylum", "class"))
    obs_stat = high / max(1, sum(cr.values()))
    rows = collections.defaultdict(list)
    for v in verd:
        k = (v["title"], v["taxon"])
        rows[v["title"]].append((confirmed(v, sents.get(k, [])),
                                 rank.get(k, "?") in ("phylum", "class")))
    rng = random.Random(SEED)
    hitsN = 0
    for _ in range(NBOOT):
        pick = []
        for p in rows:
            k = sum(1 for r in rows[p] if r[0])
            pick += [r[1] for r in rng.sample(rows[p], k)]
        if pick and sum(pick) / len(pick) >= obs_stat:
            hitsN += 1
    p = (hitsN + 1) / (NBOOT + 1)
    print(f"\n  confirmed misses at phylum/class : {high}/{sum(cr.values())} "
          f"= {100*obs_stat:.0f}%")
    print(f"  graph edges at phylum/class      : "
          f"{100*(gr.get('phylum',0)+gr.get('class',0))/tot:.1f}%")
    print(f"  BUT candidates at phylum/class   : "
          f"{100*(ar.get('phylum',0)+ar.get('class',0))/sum(ar.values()):.0f}%"
          "  <- the generator is itself rank-skewed")
    print(f"  permutation, paper-level, preserving each paper's confirmed count:")
    print(f"    p = {p:.4f}  ({'NULL' if p > 0.05 else 'survives'}) "
          "-- single uncorrected test, 15 events from 24 papers")

    out = {"total_candidates": total_candidates, "sample_papers": len(papers),
           "sample_candidates": n, "raw_real_miss": raw, "confirmed": c,
           "rate": c / n, "rate_ci": [lo, hi], "graph_observations": obs,
           "recall_point": obs / (obs + c / n * total_candidates),
           "rank_permutation_p": p,
           "confirmed_list": [{"title": h[0], "taxon": h[1],
                               "direction": h[2], "rank": h[3]} for h in hits]}
    with open(os.path.join(HERE, "edge_recall.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote edge_recall.json")


if __name__ == "__main__":
    main()
