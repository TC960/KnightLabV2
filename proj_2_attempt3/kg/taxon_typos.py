#!/usr/bin/env python3
"""Surface-spelling variants of taxa that the papers themselves misspell.

WHY THIS EXISTS
---------------
254 of 925 taxon nodes never resolve to an NCBI taxid. Most are legitimate 16S
clade labels (SMB53, cc115, PAC000195_g, Marine_Methylotrophic_Group_3) and must
stay unresolved. But a sweep of that set (`--sweep`) turns up two classes that
are not labels at all:

  A. Separator/bracket variants of ONE concept, split across two nodes by
     punctuation. 12 groups. These are a build_kg bug, not a data problem, and
     are fixed there -- see `taxon_typos_test()` and the two-line fix in
     `norm_taxon`. Listed here only so the sweep that finds them is reproducible.

  B. Misspellings. `Fecalibacterium`, `Subdogranulum`, `pesudobeautyrivibrio`,
     `Verruocomicrobiota`, `Enterobacteriaeae`. 33 of them.

The load-bearing fact about class B, established by `--verify` and NOT assumed:
**every one of the 33 appears verbatim in its own source paper's full text.**
These are the PAPERS' spelling errors, copied faithfully by the extractor. Not
one is an extraction error. That is why a curated table is the right instrument:
there is nothing upstream to fix.

WHY THE TABLE IS CURATED AND NOT AN EDIT-DISTANCE RULE
------------------------------------------------------
Because edit distance gets it wrong in both directions, and the near-misses are
exactly the cases that matter:

  * `Oscillospirales` (order) is 2 edits from `Oscillospira` (genus) and is a
    DIFFERENT REAL TAXON. So is `Thermoactinomycetales` vs
    `Thermoactinomycetaceae` -- and one paper names both in the same sentence.
  * `Enterococcus phage EFAP 1` is close to `Enterococcus phage EFRM31`. Two
    different phages.
  * `Prevotella_9`, `Ruminococcus_1`, `Coprococcus_2`, `Tyzzerella 4`,
    `Clostridiaceae 1` are all ~1 edit from their parent genus and are
    DELIBERATELY held separate (FINDINGS_rank_collapse.md). An edit-distance
    rule would silently undo the placeholder split.
  * `Corynebacteria` is a real misspelling but of WHAT is ambiguous -- the paper
    calls it "class Corynebacteria among phylum Actinobacteria", and there is no
    class by that name. Genus *Corynebacterium* and order *Corynebacteriales*
    are both plausible. Refused rather than guessed.
  * `lactic acid bacteria` is not a taxon at all; it is a physiological guild
    spanning several genera, measured in that paper with group-specific primers.
    Refused -- it needs a human decision about whether it belongs in a taxon
    vocabulary, not a spelling correction.
  * bare `UCG-002` is a real SILVA placeholder but does not say whose. In one of
    its two papers it sits beside Ruminococcus, suggesting
    `Ruminococcaceae UCG-002`; in the other it does not occur in the text at all.
    Refused.

So REFUSALS are recorded here with their reasons, alongside the accepted folds.
A future sweep re-derives the same candidates and must re-read this list rather
than re-litigate it.

WHAT IT IS NOT
--------------
Not an accuracy gain. Five structural corrections have now each moved agreement
with Disbiome/Peryton by less than this corpus can resolve (~0.013), and this one
touches 48 of 2,034 edges. It is justified on correctness of meaning:
`Fecalibacterium` and `Faecalibacterium` are one genus, and the graph said two.
Measure the agreement delta, report it, do NOT cite it as evidence of anything.

Usage:
    python3 taxon_typos.py --sweep     # re-derive candidates from graph.json
    python3 taxon_typos.py --verify    # check every entry against source text
    python3 taxon_typos.py             # write taxon_typos.json
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "taxon_typos.json")

TEXT_SOURCES = [
    os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json"),
    os.path.join(HERE, "extract_input.json"),
    os.path.join(HERE, "new_papers.json"),
]

# ---------------------------------------------------------------- the table --
# variant (exactly as it appears in the extraction) -> canonical spelling.
# Every entry is verified by --verify to occur verbatim in its source paper.
TYPOS = {
    # -- misspelled genus / family / order names -------------------------------
    "Fecalibacterium":              "Faecalibacterium",
    "Fecalibacterium prausnitzii":  "Faecalibacterium prausnitzii",
    "Subdogranulum":                "Subdoligranulum",
    "pesudobeautyrivibrio":         "Pseudobutyrivibrio",
    "Megamonus":                    "Megamonas",
    "Morganelia":                   "Morganella",
    "Massilla":                     "Massilia",
    "Parabacteroids":               "Parabacteroides",
    "Christensenea":                "Christensenella",
    "Cristensenellaceae":           "Christensenellaceae",
    "Christensellaceae":            "Christensenellaceae",
    "Peptococcacceae":              "Peptococcaceae",
    "Enterobacteriaeae":            "Enterobacteriaceae",
    "Verrumicrobiaceae":            "Verrucomicrobiaceae",
    "Verruocomicrobiota":           "Verrucomicrobiota",
    "Lachinospiracea":              "Lachnospiraceae",
    "Lachnospiracea_incertae_sedis": "Lachnospiraceae incertae sedis",
    "Tuzzerella":                   "Tyzzerella",
    "Diallister invisus":           "Dialister invisus",
    "Butyricoccus":                 "Butyricicoccus",
    "Butyricococcus":               "Butyricicoccus",
    "Butyricoccaceae":              "Butyricicoccaceae",
    "Candidatus Soleaferre":        "Candidatus Soleaferrea",
    "Lactobacillaes":               "Lactobacillales",
    "Blaut":                        "Blautia",
    "Egerthella sp":                "Eggerthella",

    # -- abbreviation of a binomial the same sentence spells out elsewhere -----
    "E. coli":                      "Escherichia coli",

    # -- rank word / prefix carried into the name ------------------------------
    "family Lachnospiraceae":       "Lachnospiraceae",
    "unclassified_g_Bacteroides":   "unclassified Bacteroides",

    # -- vernacular plural of a genus ------------------------------------------
    # Same operation as the sanctioned "X sp./spp./unclassified -> genus" fold:
    # an English plural of the genus name, not a distinct clade.
    "Bifidobacteria":               "Bifidobacterium",

    # -- whitespace injected into a SILVA placeholder --------------------------
    # These fold to the PLACEHOLDER node, never to the parent genus. The same
    # paper writes "Rumino coccus_1" and "Rumino coccus_torques_group", so the
    # split is an artifact of that paper's typesetting.
    "Rumino coccus_1":              "Ruminococcus_1",
    "Ruminococcus2":                "Ruminococcus 2",

    # -- "uncultured X sp." -> genus (explicitly sanctioned in CLAUDE.md) ------
    "uncultured Veillonella sp":    "Veillonella",
}

# Candidates the sweep surfaces and this table deliberately REFUSES, with the
# reason. Kept so a later sweep does not re-open them.
REFUSED = {
    "Oscillospirales":       "real order; near-match to genus Oscillospira is coincidence",
    "Thermoactinomycetales": "real order; the same paper separately names Thermoactinomycetaceae",
    "Anaerostignum":         "real genus (Lachnospiraceae, 2019); absent from the taxdump copy, not misspelled",
    "Corynebacteria":        "ambiguous: paper says 'class Corynebacteria', no such class; genus vs order undecidable",
    "lactic acid bacteria":  "not a taxon -- a physiological guild; needs a human call on whether it belongs at all",
    "UCG-002":               "bare SILVA placeholder, parent family unstated; absent from one of its two source texts",
    "Lachnospiraceae_Eubacterium": "a pipeline label naming a genus inside a family, not a misspelling",
    "SMB53":                 "Greengenes label, correct as written",
    "cc115":                 "Greengenes label, correct as written",
    "PAC000195_g":           "pipeline label, correct as written",
    "Marine_Methylotrophic_Group_3": "SILVA label, correct as written",
    "Mogibacteriaceae":      "real family name, unresolved for taxdump-coverage reasons",
    "Enterococcus phage EFAP 1": "a different phage from EFRM31",
}


def load():
    """variant -> canonical, as a plain dict."""
    if os.path.exists(OUT):
        return json.load(open(OUT))["typos"]
    return dict(TYPOS)


# ------------------------------------------------------------------ sweeping --
RANKWORDS = r"^(?:family|genus|order|class|phylum|species|the)\s+"


def _norm(s):
    s = s.lower().strip()
    s = re.sub(RANKWORDS, "", s)
    s = s.replace("[", "").replace("]", "")
    s = re.sub(r"[\s_\-‐-―/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sweep():
    """Re-derive the candidate list from graph.json. Prints, returns nothing."""
    g = json.load(open(GRAPH))
    tax = [n for n in g["nodes"] if n["type"] == "taxon"]
    res = [n for n in tax if n.get("resolved")]
    unres = [n for n in tax if not n.get("resolved")]

    print(f"{len(tax)} taxa, {len(unres)} unresolved")

    buckets = collections.defaultdict(list)
    for n in unres:
        buckets[_norm(n["label"])].append(n)
    groups = {k: v for k, v in buckets.items() if len(v) > 1}
    print(f"\nA. separator/bracket variants of one concept: {len(groups)} groups")
    for k, v in sorted(groups.items()):
        print("   " + " | ".join(f"{x['label']!r}({x['degree']})" for x in v))

    resmap = {}
    for n in res:
        for a in set([n["label"]] + list(n.get("aliases") or [])):
            resmap.setdefault(_norm(a), n)
    print("\nB. unresolved label that normalises onto a resolved node:")
    for n in unres:
        hit = resmap.get(_norm(n["label"]))
        if hit:
            print(f"   {n['label']!r}({n['degree']}) -> {hit['label']!r} taxid {hit['taxid']}")

    reslabels = sorted({_norm(n["label"]) for n in res})
    print("\nC. near-misses needing judgement (cutoff 0.87):")
    for n in sorted(unres, key=lambda x: -x["degree"]):
        t = _norm(n["label"])
        if len(t) < 5:
            continue
        close = [c for c in difflib.get_close_matches(t, reslabels, n=2, cutoff=0.87) if c != t]
        if not close:
            continue
        verdict = ("ACCEPTED -> " + TYPOS[n["label"]]) if n["label"] in TYPOS else \
                  ("REFUSED: " + REFUSED[n["label"]]) if n["label"] in REFUSED else "*** UNTRIAGED ***"
        print(f"   {n['label']!r}({n['degree']}) ~ {close}  [{verdict}]")


# ---------------------------------------------------------------- verifying --
def _tkey(t):
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(t)).lower())


def _load_texts():
    T = {}
    for path in TEXT_SOURCES:
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        recs = d if isinstance(d, list) else list(d.values())
        for r in recs:
            if not isinstance(r, dict):
                continue
            t = r.get("title") or r.get("name")
            body = r.get("text") or r.get("full_text") or r.get("body")
            if body is None and isinstance(r.get("chunks"), list):
                body = "\n".join(r["chunks"])
            if t and body:
                T.setdefault(_tkey(t), body)
    return T


def verify():
    """Every accepted variant must occur verbatim in a paper that reports it.

    This is the claim the whole table rests on -- that these are the papers'
    spellings and not the extractor's. If an entry cannot be found in its own
    source text it is an EXTRACTION error and belongs in a different fix, so
    this exits non-zero rather than shipping it quietly.
    """
    g = json.load(open(GRAPH))
    T = _load_texts()
    by_taxon = collections.defaultdict(list)
    for e in g["edges"]:
        by_taxon[e["taxon"]].append(e)

    ok, bad, notext = [], [], []
    for variant in TYPOS:
        edges = by_taxon.get(variant, [])
        if not edges:
            bad.append((variant, "no edge in graph"))
            continue
        found = False
        for e in edges:
            for title in e["papers"]:
                body = T.get(_tkey(title))
                if body is None:
                    continue
                if variant.lower() in body.lower():
                    found = True
                    break
            if found:
                break
        if found:
            ok.append(variant)
        elif not any(_tkey(t) in T for e in edges for t in e["papers"]):
            notext.append(variant)
        else:
            bad.append((variant, "not in any source paper's text -> EXTRACTION error, not a paper typo"))

    print(f"verified verbatim in source text : {len(ok)}/{len(TYPOS)}")
    if notext:
        print(f"source text unavailable          : {len(notext)}  {notext}")
    if bad:
        print(f"FAILED                           : {len(bad)}")
        for v, why in bad:
            print(f"   {v!r}: {why}")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="re-derive candidates from graph.json")
    ap.add_argument("--verify", action="store_true", help="check entries against source full text")
    a = ap.parse_args()
    if a.sweep:
        sweep()
        return 0
    if a.verify:
        return verify()
    payload = {
        "note": ("Surface spellings the SOURCE PAPERS get wrong, not the extractor: "
                 "every entry is verified to occur verbatim in its own paper's full "
                 "text by `taxon_typos.py --verify`. Curated, not edit-distance: see "
                 "REFUSED for the near-misses that are real distinct taxa."),
        "n_typos": len(TYPOS),
        "n_refused": len(REFUSED),
        "typos": TYPOS,
        "refused": REFUSED,
    }
    json.dump(payload, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"wrote {OUT}: {len(TYPOS)} folds, {len(REFUSED)} refusals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
