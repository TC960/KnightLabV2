#!/usr/bin/env python3
"""Push the contrast audit from the PAPER to the (paper, taxon) OBSERVATION.

`FINDINGS_contrast_scope.md` bounds out-of-gate observations at 1.8-2.7% and shows
out-of-gate papers disagree with the literature 1.75x as often. Its stated limit is
that it is a PAPER-level instrument, while the dominant residual risk is WITHIN a
paper: 94 of 241 in-scope papers also report a within-disease subgroup contrast, so
a paper can be entirely in scope and still contribute an edge read off the wrong
comparison. No paper-level verdict can see that.

This is the edge-level version, and it is DETERMINISTIC -- no adjudicator, so no
paraphrase risk and no blinding needed. For each (paper, taxon) observation behind
an edge, find the sentences in that paper that name that taxon (taxa are already
resolved per sentence in relation_sentences_clean.json, so this joins on TAXID, not
on strings) and ask what comparison those sentences describe:

  CLEAN        >=1 sentence naming the taxon states a healthy/normal-control contrast
  CANDIDATE    0 such sentences, but >=1 states a named within-disease subgroup or
               treatment-arm contrast -- the observation may have been read off the
               wrong comparison
  UNRESOLVED   neither; the filtered sentences do not name a comparison at all

ASYMMETRY, and it is the same one the 2026-09-16 recall audit had to state:
relation_sentences_clean keeps only sentences carrying a taxon AND a direction cue,
about 10% of corpus text. A paper can state its comparison in a sentence this file
never kept. So CANDIDATE means "no visible control contrast for this taxon", never
"this observation is wrong" -- this instrument can flag, it cannot convict. The
counts are an UPPER bound on contamination.

Writes contrast_edge_probe.json.
"""
import json, random, re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
N_PERM = 20000
SEED = 20260919

# Widened 2026-09-19 after adjudication exposed four real phrasings it missed:
# "compared with CON", "in contrast to both the control and ...", "the healthy
# MALE group", "their non-stroke counterparts". Justified on the regex being
# wrong, not on moving a number -- the effect on the sampled observations is
# reported separately from the effect on the 68 that were never adjudicated.
CONTROL = re.compile(
    r"health(y|ies)\b[ -]?\w*\s?(control|subject|volunteer|individual|participant|"
    r"donor|adult|elder|people|person|group|twin|male|female)|normal control|"
    r"\bHCs?\b|\bNCs?\b|\bCONs?\b|control group|\bthe controls?\b|"
    r"(than|versus|vs\.?|compared (with|to)|relative to|in contrast to)\s+"
    r"(both\s+)?(the\s+)?(healthy\s+)?controls?\b|cognitively normal|"
    r"non-?demented|unaffected|\bCTR\b|\bHV\b|"
    r"non-?\w+ (counterpart|control|subject|participant|group)s?", re.I)

SUBGROUP = re.compile(
    r"\bwith(out)? (and|or) without\b|\bthose without\b|non-?[A-Z]{2,}\b|"
    r"responders?\b|non-?responder|survivors?\b|deceased|"
    r"(mild|moderate|severe) (versus|vs\.?|compared)|"
    r"after treatment|post-?treatment|baseline (and|vs)|placebo|"
    r"\b(probiotic|rifaximin|lactulose|ketogenic|supplementation|intervention)\b|"
    r"\bCN\s?[+-]|amyloid-?(positive|negative)|\bsALS\b|\bbALS\b|"
    r"body-?first|brain-?first|cluster [AB]\b", re.I)

def taxid_of(key):
    return key.split(":", 1)[1] if key and key.startswith("ncbi:") else None

def main():
    g = json.load(open(HERE / "graph.json"))
    rs = json.load(open(HERE / "relation_sentences_clean.json"))["papers"]
    lower = {k.rstrip(".").lower(): k for k in rs}
    census = {v["title"]: v for v in
              json.load(open(HERE / "contrast_census.json"))["verdicts"]
              if v.get("supported")}

    # taxid -> sentences, and resolved-name -> sentences, per paper
    idx = {}
    for t, rec in rs.items():
        by_tax, by_name = defaultdict(list), defaultdict(list)
        for k in rec["kept"]:
            for surf, tid, name, rank in k["taxa"]:
                if tid:
                    by_tax[str(tid)].append(k["s"])
                if name:
                    by_name[name.lower()].append(k["s"])
                if surf:
                    by_name[str(surf).lower()].append(k["s"])
        idx[t] = (by_tax, by_name)

    rows = []
    for e in g["edges"]:
        tid = taxid_of(e.get("taxon_key"))
        tname = str(e.get("taxon", "")).lower()
        for t in e["papers"]:
            key = t if t in idx else lower.get(t.rstrip(".").lower())
            if key is None:
                rows.append({"paper": t, "taxon": e["taxon"],
                             "taxon_key": e.get("taxon_key"),
                             "disease": e["disease"], "verdict": "NO_PAPER"})
                continue
            by_tax, by_name = idx[key]
            sents = by_tax.get(tid, []) if tid else []
            if not sents:
                sents = by_name.get(tname, [])
            ctrl = [s for s in sents if CONTROL.search(s)]
            sub = [s for s in sents if SUBGROUP.search(s) and not CONTROL.search(s)]
            v = ("NO_SENTENCE" if not sents else
                 "CLEAN" if ctrl else
                 "CANDIDATE" if sub else "UNRESOLVED")
            rows.append({"paper": key, "taxon": e["taxon"],
                         "taxon_key": e.get("taxon_key"),
                         "disease": e["disease"],
                         "verdict": v, "n_sent": len(sents),
                         "paper_contrast": census.get(key, {}).get("contrast_type"),
                         "paper_also_subgroup": census.get(key, {}).get("also_subgroup"),
                         "example": (sub or ctrl or sents or [""])[0][:300]})

    c = Counter(r["verdict"] for r in rows)
    # does the flag concentrate in papers the paper-level audit already suspected?
    cross = defaultdict(Counter)
    for r in rows:
        if r["verdict"] in ("CLEAN", "CANDIDATE", "UNRESOLVED"):
            k = ("out_of_gate" if r.get("paper_contrast") not in (None, "HC", "UNCLEAR")
                 else "in_gate_mixed" if r.get("paper_also_subgroup")
                 else "in_gate_clean")
            cross[k][r["verdict"]] += 1

    out = {"n_observations": len(rows), "verdicts": dict(c),
           "by_paper_class": {k: dict(v) for k, v in cross.items()},
           "rows": rows}
    json.dump(out, open(HERE / "contrast_edge_probe.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in
                      ("n_observations", "verdicts", "by_paper_class")}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
