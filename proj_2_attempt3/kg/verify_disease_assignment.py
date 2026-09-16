#!/usr/bin/env python3
"""Is each paper filed under the disease it actually studies?

`build_kg.py:383` keys every edge's disease node on the LLM's `predicted_disease`,
falling back to the datasheet's human `disease` only when that is empty. So the
disease half of every edge in the graph rests on a model output that -- unlike
the taxon half -- has never been audited. A wrong disease does not corrupt one
edge; it misfiles an entire paper's edges onto the wrong node, and it is
invisible to every taxon-side check in this repo.

Two signals, both deterministic:

1. **Disagreement with the human datasheet.** Most differences are cosmetic
   (curly apostrophe, `(PD)` suffix, case) or granularity (`Stroke` ->
   `acute ischemic stroke`), which are not errors. Those are canonicalised away
   so that only genuine family-level conflicts survive -- `Alzheimer's` vs
   `multiple sclerosis` is a conflict, `Stroke` vs `ischemic stroke` is not.

2. **Adjudication against the paper's own text.** For each surviving conflict,
   count how often each candidate disease's vocabulary occurs in the full text.
   The disease a case-control study is about is named throughout it; the loser is
   typically named once, in a citation. This adjudicates the conflict without
   asking a model to re-read the paper.

Reports the LLM's granularity refinements separately from its disagreements,
because refining `Stroke` to `intracerebral hemorrhage` is the extractor doing
something USEFUL and should not be counted as an error.
"""
import json
import os
import re
from collections import Counter, defaultdict

from verify_taxon_mentions import norm_text

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACTIONS = os.path.join(HERE, "extractions_screened.json")
PAPERS = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
OUT = os.path.join(HERE, "disease_assignment.json")

# Disease families and the vocabulary that identifies them in a paper.
# Abbreviations are matched with word boundaries; they are ambiguous alone
# (`AD`, `MS`, `PD`) so the full name carries the weight and the abbreviation is
# counted separately.
FAMILIES = {
    "parkinson": {"full": [r"parkinson"], "abbr": [r"\bpd\b"]},
    "alzheimer": {"full": [r"alzheimer"], "abbr": [r"\bad\b"]},
    "multiple sclerosis": {"full": [r"multiple sclerosis"], "abbr": [r"\bms\b", r"\brrms\b"]},
    "als": {"full": [r"amyotrophic lateral sclerosis"], "abbr": [r"\bals\b"]},
    "mild cognitive impairment": {"full": [r"mild cognitive impairment"], "abbr": [r"\bmci\b"]},
    "stroke": {"full": [r"stroke", r"cerebral infarction", r"ischemi[ac]",
                        r"intracerebral h[ae]morrhage"], "abbr": [r"\bais\b", r"\bich\b"]},
    "autism": {"full": [r"autism", r"autistic"], "abbr": [r"\basd\b"]},
    "huntington": {"full": [r"huntington"], "abbr": [r"\bhd\b"]},
    "spinal cord injury": {"full": [r"spinal cord injury"], "abbr": [r"\bsci\b"]},
    "spinal muscular atrophy": {"full": [r"spinal muscular atrophy"], "abbr": [r"\bsma\b"]},
    "multiple system atrophy": {"full": [r"multiple system atrophy"], "abbr": [r"\bmsa\b"]},
    "epilepsy": {"full": [r"epilep"], "abbr": []},
    "migraine": {"full": [r"migraine"], "abbr": []},
    "dementia": {"full": [r"dementia"], "abbr": []},
    "neuromyelitis": {"full": [r"neuromyelitis"], "abbr": [r"\bnmosd\b"]},
    "encephalitis": {"full": [r"encephalitis"], "abbr": []},
    "encephalopathy": {"full": [r"encephalopathy"], "abbr": []},
    "cadasil": {"full": [r"cadasil"], "abbr": []},
    "essential tremor": {"full": [r"essential tremor"], "abbr": []},
    "depression": {"full": [r"depression", r"depressive"], "abbr": [r"\bmdd\b"]},
    "cerebral palsy": {"full": [r"cerebral palsy"], "abbr": []},
    "myasthenia": {"full": [r"myasthenia"], "abbr": []},
    "neuropathy": {"full": [r"neuropathy"], "abbr": []},
}


def canon(label: str):
    """Map a free-text disease label onto the family set it names."""
    s = re.sub(r"[‘’']", "'", (label or "").lower())
    s = re.sub(r"\(([^)]*)\)", r" \1 ", s)  # unwrap "(PD)"
    s = re.sub(r"\s+", " ", s).strip()
    fams = set()
    for fam, pats in FAMILIES.items():
        for p in pats["full"] + pats["abbr"]:
            if re.search(p, s):
                fams.add(fam)
                break
    return fams


def counts(text: str, fam: str):
    full = sum(len(re.findall(p, text)) for p in FAMILIES[fam]["full"])
    abbr = sum(len(re.findall(p, text)) for p in FAMILIES[fam]["abbr"])
    return full, abbr


def main():
    extractions = json.load(open(EXTRACTIONS))
    fulltext = {}
    for p in json.load(open(PAPERS)):
        k = re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower())
        if k and len(p.get("text") or "") > 500:
            fulltext[k] = norm_text(p["text"])

    tally = Counter()
    conflicts = []
    unresolved_family = []

    for rec in extractions:
        sheet = (rec.get("disease") or "").strip()
        pred = (rec.get("predicted_disease") or "").strip()
        if not pred:
            tally["no_prediction_falls_back_to_sheet"] += 1
            continue
        fs, fp = canon(sheet), canon(pred)

        if not fp:
            tally["prediction_unmapped"] += 1
            unresolved_family.append({"title": rec.get("title"), "sheet": sheet, "pred": pred})
            continue
        if not fs:
            # datasheet said "Other"/blank: nothing to disagree with
            tally["sheet_uninformative"] += 1
            continue
        if fp & fs:
            # overlap -> same family. Narrower prediction is a refinement, not an error.
            tally["agree" if fp == fs else "refinement"] += 1
            continue

        tally["family_conflict"] += 1
        k = re.sub(r"[^a-z0-9]", "", (rec.get("title") or "").lower())
        text = fulltext.get(k)
        row = {"title": rec.get("title"), "sheet": sheet, "pred": pred,
               "sheet_families": sorted(fs), "pred_families": sorted(fp),
               "link": rec.get("link", "")}
        if text is None:
            # No full text, but the title alone often settles it: a paper called
            # "...cerebral autosomal dominant arteriopathy with subcortical..."
            # IS the CADASIL study whatever the datasheet's coarse label says.
            tnorm = norm_text(rec.get("title") or "")
            pit = any(re.search(p, tnorm) for f in fp for p in FAMILIES[f]["full"])
            sit = any(re.search(p, tnorm) for f in fs for p in FAMILIES[f]["full"])
            row["pred_in_title"], row["sheet_in_title"] = pit, sit
            if pit and not sit:
                row["verdict"] = "prediction_correct_by_title_sheet_coarse"
                tally["conflict_prediction_correct_by_title"] += 1
            elif sit and not pit:
                row["verdict"] = "SHEET_IN_TITLE_PREDICTION_SUSPECT"
                tally["conflict_sheet_in_title"] += 1
            else:
                row["verdict"] = "not_scoreable_no_fulltext"
                tally["conflict_not_scoreable"] += 1
        else:
            ev = {}
            for fam in sorted(fs | fp):
                full, abbr = counts(text, fam)
                ev[fam] = {"full_name_hits": full, "abbrev_hits": abbr}
            row["evidence"] = ev
            best_pred = max((ev[f]["full_name_hits"] for f in fp), default=0)
            best_sheet = max((ev[f]["full_name_hits"] for f in fs), default=0)
            row["pred_fullname_hits"] = best_pred
            row["sheet_fullname_hits"] = best_sheet

            # The TITLE is near-decisive for a case-control paper and body counts
            # are not: an MCI study discusses Alzheimer's throughout because MCI
            # is its prodrome, so raw frequency favours the wrong answer. Title
            # evidence is therefore checked first.
            tnorm = norm_text(rec.get("title") or "")
            row["pred_in_title"] = any(
                re.search(p, tnorm) for f in fp for p in FAMILIES[f]["full"])
            row["sheet_in_title"] = any(
                re.search(p, tnorm) for f in fs for p in FAMILIES[f]["full"])

            if best_pred == 0 and best_sheet > 0:
                row["verdict"] = "PREDICTION_UNSUPPORTED"
                tally["conflict_prediction_unsupported"] += 1
            elif row["pred_in_title"] and not row["sheet_in_title"]:
                row["verdict"] = "prediction_correct_by_title_sheet_coarse"
                tally["conflict_prediction_correct_by_title"] += 1
            elif row["sheet_in_title"] and not row["pred_in_title"]:
                row["verdict"] = "SHEET_IN_TITLE_PREDICTION_SUSPECT"
                tally["conflict_sheet_in_title"] += 1
            elif row["pred_in_title"] and row["sheet_in_title"]:
                row["verdict"] = "comparative_study_both_in_title"
                tally["conflict_comparative"] += 1
            elif best_pred >= 3 * max(1, best_sheet):
                row["verdict"] = "prediction_dominant_sheet_likely_stale"
                tally["conflict_prediction_dominant"] += 1
            elif best_sheet >= 3 * max(1, best_pred):
                row["verdict"] = "SHEET_DOMINANT_PREDICTION_SUSPECT"
                tally["conflict_sheet_dominant"] += 1
            else:
                row["verdict"] = "ambiguous_both_present"
                tally["conflict_ambiguous"] += 1
        conflicts.append(row)

    out = {"n_records": len(extractions), "tally": dict(tally),
           "conflicts": conflicts, "unmapped_predictions": unresolved_family}
    json.dump(out, open(OUT, "w"), indent=1)

    print(json.dumps(dict(tally), indent=1))
    print(f"\n{len(conflicts)} family-level conflicts\n")
    for c in conflicts:
        print(f"[{c['verdict']}]")
        print(f"    sheet={c['sheet']!r} -> pred={c['pred']!r}")
        print(f"    {c['title'][:88]}")
        if "evidence" in c:
            ev = ", ".join(f"{f}: {v['full_name_hits']}full/{v['abbrev_hits']}abbr"
                           for f, v in c["evidence"].items())
            print(f"    {ev}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
