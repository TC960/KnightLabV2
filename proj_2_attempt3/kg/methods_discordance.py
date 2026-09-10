#!/usr/bin/env python3
"""Do wet-lab and bioinformatics choices explain the paper-level direction offset?

`FINDINGS_paper_discordance.md` left the project here: disagreement with the rest
of the literature is a real paper-level property (within-edge null p = 0.0003),
and none of the nine variables in `metadata.jsonl` explain it (best q = 0.234,
MDE +-0.16 to +-0.22 in O/E). The variables the microbiome methods literature
actually blames for cohorts disagreeing were never extracted. `methods_metadata.py`
now extracts them from full text that was in the repo all along.

Same machinery as `paper_discordance_offset.py`, deliberately: outcome is each
paper's disagreements against the leave-one-out majority, offset is the exact
closed-form within-edge expectation (which absorbs edge depth -- the confound that
produced three false positives when raw rates were used), statistic is a
difference in O/E, null shuffles the predictor label ACROSS PAPERS holding
(observed, expected) fixed, BH across predictors, MDE reported for every null.

Predictors are chosen a priori from the batch-effect literature rather than by
scanning for whatever separates:
  * differential-abundance test    -- LEfSe is known to be liberal
  * multiple-testing correction    -- reporting uncorrected p-values
  * feature type                   -- OTU clustering vs exact ASVs
  * DNA extraction kit             -- the canonical microbiome batch effect
  * rarefaction, CLR, absolute quantification
  * pipeline and platform

Two planted controls that SHOULD be null if the detector is behaving:
  * `detector_found_methods_section` -- scope is a property of the text dump, not
    of the study, so it must not predict discordance
  * `sits_on_deep_edges`             -- the confound the offset exists to absorb

Writes methods_discordance.json.
"""

import json
import random
import sys
from collections import Counter, defaultdict

from paper_inversion import MIN_DECISIVE, build_observations, load_graph, score
from paper_inversion_control import ancestor_sets, thin
from paper_discordance_offset import expectations, test_binary

SEED = 20260910


def main():
    graph = load_graph()
    edges = build_observations(graph)
    rel = ancestor_sets(graph)
    thinned, _ = thin(edges, rel)
    per_paper, _ = score(thinned)
    _, e_dis = expectations(thinned)

    mm = json.load(open("methods_metadata.json"))
    rec = {r["paper"]: r for r in mm["records"]}

    rows = []
    for p, v in sorted(per_paper.items()):
        if v[0] < MIN_DECISIVE or e_dis[p] <= 0:
            continue
        r = rec.get(p, {})
        rows.append({"paper": p, "dec": v[0], "dis": v[1], "e_dis": e_dis[p],
                     "title": graph["papers"][p]["title"][:60], "mm": r})
    o = sum(r["dis"] for r in rows)
    e = sum(r["e_dis"] for r in rows)
    with_text = sum(1 for r in rows if r["mm"].get("have_text"))
    print(f"{len(rows)} testable papers ({with_text} with full text); "
          f"observed {o} vs expected {e:.1f} (O/E {o/e:.3f})\n")

    def fam(r, family):
        return set(r["mm"].get(family) or []) if r["mm"].get("have_text") else None

    def has(family, *names):
        """True if any of `names` present, False if the family fired but none of
        them, None if the family is empty (no evidence either way)."""
        out = []
        for r in rows:
            s = fam(r, family)
            if s is None or not s:
                out.append(None)
            else:
                out.append(bool(s & set(names)))
        return out

    def exclusive(family, a, b):
        """True for a-only, False for b-only, None otherwise."""
        out = []
        for r in rows:
            s = fam(r, family)
            if s is None:
                out.append(None); continue
            ia, ib = a in s, b in s
            out.append(True if ia and not ib else False if ib and not ia else None)
        return out

    # multiple-testing correction: absence is informative here, because the
    # family is only ever empty when no correction is named anywhere
    mt = []
    for r in rows:
        s = fam(r, "multiple_testing")
        mt.append(None if s is None else bool(s))

    depth = defaultdict(list)
    for ed in thinned:
        for pi, _ in ed["obs"]:
            depth[pi].append(len(ed["obs"]))
    dvals = [sum(depth[r["paper"]]) / len(depth[r["paper"]]) for r in rows]
    dcut = sorted(dvals)[len(dvals) // 2]

    # Publication year, parsed from the article header and validated against the
    # 22 testable papers that carry an explicit `year` field: 21 exact (95.5%).
    # Caveat kept in view -- those 22 come from the sources that HAVE the field,
    # so the check is not a random sample of the corpus.
    years = {}
    for i, y in json.load(open("paper_years.json")).items():
        years[int(i)] = y
    yvals = sorted(y for y in years.values() if y)
    ycut = yvals[len(yvals) // 2] if yvals else 0
    year_lbl = [None if years.get(r["paper"]) is None
                else years[r["paper"]] >= ycut for r in rows]

    # Cohort imbalance: a study with many more controls than cases (or the
    # reverse) has a different effective power profile per taxon.
    imbal = []
    for r in rows:
        m = graph["papers"][r["paper"]]
        a, b = m.get("n_cases"), m.get("n_controls")
        if not m.get("has_meta") or not a or not b:
            imbal.append(None)
        else:
            imbal.append(max(a / b, b / a) >= 1.5)

    # How many distinct taxa the paper reports at all -- a broad screen versus a
    # paper reporting only its top hits.
    nobs = [len(depth[r["paper"]]) for r in rows]
    ncut = sorted(nobs)[len(nobs) // 2]

    tests = {
        "uses_LEfSe": has("diff_abundance", "LEfSe", "LDA"),
        "uses_DESeq2_or_ANCOM": has("diff_abundance", "DESeq2", "ANCOM",
                                    "ALDEx2", "edgeR", "metagenomeSeq"),
        "nonparametric_only": [
            None if (s := fam(r, "diff_abundance")) is None or not s
            else bool(s & {"Wilcoxon", "KruskalWallis"})
            and not (s & {"LEfSe", "DESeq2", "ANCOM", "ALDEx2", "edgeR",
                          "metagenomeSeq", "MaAsLin"})
            for r in rows],
        "reports_multiple_testing_correction": mt,
        "feature_ASV_vs_OTU": exclusive("feature_type", "ASV", "OTU"),
        "kit_QIAamp_vs_other": [
            None if (s := fam(r, "extraction_kit")) is None or not s
            else "QIAamp" in s for r in rows],
        "kit_bead_beating": has("extraction_kit", "PowerSoil", "PowerFecal",
                                "FastDNA", "MoBio"),
        "pipeline_QIIME2_or_DADA2": has("pipeline", "QIIME2", "DADA2"),
        "pipeline_legacy": has("pipeline", "QIIME1", "mothur", "UPARSE",
                               "USEARCH"),
        "platform_MiSeq": has("platform", "MiSeq"),
        "rarefied": [None if (s := fam(r, "normalisation")) is None or not s
                     else "rarefied" in s for r in rows],
        "absolute_or_CLR": has("normalisation", "absolute_quant", "CLR"),
        "published_recently": year_lbl,
        "cohort_imbalanced_1.5x": imbal,
        "reports_many_taxa": [v >= ncut for v in nobs],
        # planted controls -- both should be null
        "CONTROL_found_methods_section": [
            None if not r["mm"].get("have_text")
            else r["mm"].get("scope") in ("methods_section",
                                          "methods_section_weak") for r in rows],
        "CONTROL_sits_on_deep_edges": [v >= dcut for v in dvals],
    }

    rng = random.Random(SEED)
    results = {}
    for name, labels in tests.items():
        res = test_binary(rows, labels, rng)
        results[name] = res
        if res is None:
            print(f"  {name:36s} skipped (too few informative papers)")
            continue
        print(f"  {name:36s} O/E {res['oe_true']:.3f} (n={res['n_true']:>2}) vs "
              f"{res['oe_false']:.3f} (n={res['n_false']:>2})  "
              f"diff {res['diff']:+.3f}  p={res['p']:.4f}  "
              f"MDE +-{res['mde_abs_diff']:.3f}")

    live = sorted(((k, v["p"]) for k, v in results.items() if v),
                  key=lambda kv: kv[1])
    m = len(live)
    q, prev = {}, 1.0
    for i in range(m - 1, -1, -1):
        k, p = live[i]
        prev = min(prev, p * m / (i + 1))
        q[k] = round(min(1.0, prev), 4)
    for k in q:
        results[k]["q"] = q[k]
    print(f"\nBH across {m} predictors:")
    for k, p in live:
        print(f"  {k:36s} p={p:.4f}  q={q[k]:.4f}"
              f"{'  <-- survives' if q[k] < 0.05 else ''}")
    survivors = [k for k in q if q[k] < 0.05]
    print(f"\nsurvivors: {survivors if survivors else 'NONE'}")

    out = {"n_testable": len(rows), "n_with_text": with_text,
           "total_observed": o, "total_expected": round(e, 2),
           "overall_oe": round(o / e, 4), "seed": SEED,
           "tests": results, "survivors": survivors}
    with open("methods_discordance.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote methods_discordance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
