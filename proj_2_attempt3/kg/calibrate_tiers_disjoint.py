#!/usr/bin/env python3
"""Recompute the per-tier agreement rates on DISJOINT sources only.

WHAT WAS WRONG. `build_kg.CONFIDENCE_RATES` carries a measured agreement rate
per confidence tier so the viewer can quote a number it did not invent. Those
rates pool two situations that are not comparable, and `FINDINGS_independence.md`
is the reason we know they are not:

  shared source    the curated database read THE SAME PAPER we did. We agree
                   ~90% of the time, but that measures reading fidelity -- both
                   sides are looking at one sentence.
  disjoint source  the database read DIFFERENT papers about the same taxon and
                   disease. That is the question a reader actually has: does
                   this finding hold up in somebody else's cohort?

Pooling them produces a number that describes neither. The comment that used to
sit above CONFIDENCE_RATES called Disbiome and Peryton "independent references",
which `FINDINGS_independence.md` disproves: they cite 43 and 24 of our own
papers, and because the shared ones are the heavily-reported papers they back
roughly half the decisive pairs.

WHY IT MATTERED MOST FOR THE WEAKEST TIER. The contamination is concentrated in
`provisional`, which is 1,574 of 2,008 edges -- 78% of the graph. A reader told
a single-paper edge agrees with curation ~66% of the time was being told a
number ~19 points too high for the only case that tests anything.

METHOD. Reuse `check_independence.py`'s shared/disjoint marking and
`calibrate_agreement.build_pairs`'s join verbatim, so this cannot drift from the
numbers those scripts report. A pair is shared-source if ANY curated record
backing it comes from a paper our edge also rests on, matched on PMID *or* DOI
*or* title -- which maximises the shared set and therefore makes the disjoint
subset as clean as this data allows.

CIs are Wilson intervals. They are reported per database rather than pooled
because the two databases overlap in which pairs they judge, so a pooled
interval would be overconfident.

    python3 calibrate_tiers_disjoint.py
    python3 calibrate_tiers_disjoint.py --write   # update build_kg.CONFIDENCE_RATES
"""
import argparse
import json
import math
import os
import re

from calibrate_agreement import build_pairs
from check_independence import our_keys
from validate_external import GRAPH, load_disbiome, load_peryton

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "confidence_rates_disjoint.json")
BUILD_KG = os.path.join(HERE, "build_kg.py")
TIERS = ["well-supported", "supported", "provisional"]


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="rewrite CONFIDENCE_RATES in build_kg.py")
    a = ap.parse_args()

    G = json.load(open(GRAPH))
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy()

    tier_of = {(e["taxon_key"], e["disease"]): e.get("confidence")
               for e in G["edges"]}
    title2keys = our_keys()

    results = {}
    for loader in (load_disbiome, load_peryton):
        recs, nm = loader()
        if recs is None:
            print(f"  {nm}: unavailable, skipped")
            continue

        pairs = build_pairs(G, recs, tax)
        for p in pairs:
            our_k = set()
            for t in p["papers"]:
                our_k |= title2keys.get(t, set())
            p["shared_source"] = bool(our_k & set(p["ref_keys"]))
            p["tier"] = tier_of.get((p["taxon_key"], p["disease"]))

        print(f"\n{nm.upper()}  ({len(pairs)} decisive pairs)")
        print(f"  {'tier':<16} {'pooled (shipped)':>18} "
              f"{'DISJOINT ONLY':>26} {'shared':>14}")
        for t in TIERS:
            sub = [p for p in pairs if p["tier"] == t]
            dj = [p for p in sub if not p["shared_source"]]
            sh = [p for p in sub if p["shared_source"]]
            if not sub:
                continue
            pool = sum(p["agree"] for p in sub) / len(sub)
            k = sum(p["agree"] for p in dj)
            d_rate = k / len(dj) if dj else float("nan")
            lo, hi = wilson(k, len(dj))
            s_rate = (sum(p["agree"] for p in sh) / len(sh)) if sh else float("nan")
            ci = f"[{lo:.3f}, {hi:.3f}]" if dj else "--"
            print(f"  {t:<16} {pool:>9.3f} (n={len(sub):>3}) "
                  f"{d_rate:>9.3f} (n={len(dj):>3}) {ci:>16} "
                  f"{s_rate:>7.3f} (n={len(sh):>3})")
            # lowercase: the loaders return "Disbiome"/"Peryton" but build_kg's
            # table is keyed lowercase. Writing without normalising here emitted
            # a table of Nones -- caught by reading build_kg.py back, which is
            # why --write is a separate flag from the measurement.
            results.setdefault(t, {})[nm.lower()] = {
                "pooled": round(pool, 4), "n_pooled": len(sub),
                "disjoint": round(d_rate, 4) if dj else None,
                "n_disjoint": len(dj),
                "ci_lo": round(lo, 4) if dj else None,
                "ci_hi": round(hi, 4) if dj else None,
                "shared": round(s_rate, 4) if sh else None,
                "n_shared": len(sh),
            }

    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

    if not a.write:
        print("\n(--write not given; build_kg.py untouched)")
        return

    # ---- rewrite the table in build_kg.py -------------------------------
    lines = ["CONFIDENCE_RATES = {"]
    for t in TIERS:
        r = results.get(t, {})
        d = r.get("disbiome", {})
        p = r.get("peryton", {})
        lines.append(
            f'    "{t}": {{"disbiome": {d.get("disjoint")}, '
            f'"peryton": {p.get("disjoint")},\n'
            f'{" " * (len(t) + 7)}"n_disbiome": {d.get("n_disjoint", 0)}, '
            f'"n_peryton": {p.get("n_disjoint", 0)}}},')
    lines.append('    "contested": {"disbiome": None, "peryton": None,')
    lines.append('                  "n_disbiome": 0, "n_peryton": 0},')
    lines.append("}")
    new = "\n".join(lines)

    # Refuse to write a table of Nones. The first run of this script did exactly
    # that -- a key-case mismatch silently produced an all-None table that would
    # have deleted every rate the viewer quotes while reporting success.
    if new.count("None") > 2:      # only `contested` may legitimately be None
        raise SystemExit("refusing to write: rates came out None, check key names")

    src = open(BUILD_KG).read()
    pat = re.compile(r"^CONFIDENCE_RATES = \{.*?^\}", re.S | re.M)
    if not pat.search(src):
        raise SystemExit("could not locate CONFIDENCE_RATES in build_kg.py")
    open(BUILD_KG, "w").write(pat.sub(new, src, count=1))
    print(f"rewrote CONFIDENCE_RATES in {BUILD_KG}")
    print("now rerun: python3 build_kg.py && python3 build_viz.py")


if __name__ == "__main__":
    main()
