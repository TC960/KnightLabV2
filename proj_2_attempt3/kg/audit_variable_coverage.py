#!/usr/bin/env python3
"""Audit the variable-coverage survey: is a keyword hit the same as a stated variable?

The survey in FINDINGS_variable_coverage.md counted a paper as "covering" a
variable if a keyword appeared ANYWHERE in the full text. Its own top hit shows
why that is not enough:

    recruitment setting, 96.4% -- "Department of General Surgery, Shanghai Tenth
    People's Hospital, Tongji ..."

That is an author affiliation. It is the identical failure the `country` field
already made (recording an author's institution instead of the cohort's site).
`disease severity`'s example is a REFERENCE TITLE ("Probiotic VSL#3 reduces liver
disease severity ...").

So this script re-measures each variable at three tiers:

  NAIVE       keyword anywhere                       (what the survey reported)
  BODY        keyword outside front-matter and outside the reference block
  ATTRIBUTED  keyword in a sentence that also carries a study-design cue for
              THAT variable -- i.e. the sentence is plausibly about the study's
              own cohort/protocol rather than background, a mouse model, or a
              citation
  VALUED      a concrete value is adjacent (a number, a named kit/test/platform)
              -- the tier that actually matters, because an extractor needs a
              value, not a mention

Only VALUED coverage should drive a decision to extract a variable.

    python audit_variable_coverage.py            # table to stdout + JSON
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json"),
    os.path.join(HERE, "new_papers.json"),
]
OUT = os.path.join(HERE, "audit_variable_coverage.json")

# Front matter: title block + author list + affiliations. Affiliations are where
# "Hospital"/"University" cluster, and they say nothing about the cohort.
FRONT_CHARS = 2500

# The reference list. all_usable_papers came from a PMC scrape that keeps it;
# new_papers.json had <ref-list> stripped at fetch time. Cut at the last plausible
# heading so citation titles cannot be counted as study facts.
REF_HEAD = re.compile(
    r"\n\s*(?:References|REFERENCES|Bibliography|Literature Cited)\s*\n", re.M)

SENT = re.compile(r"(?<=[.!?])\s+")

# Cues that a sentence describes THIS study's own cohort/protocol, not background.
OWN_STUDY = re.compile(
    r"\b(we |our |this study|the present study|were recruited|was recruited|"
    r"were enrolled|was enrolled|participants|subjects were|patients were|"
    r"inclusion criteri|exclusion criteri|were collected|was performed|"
    r"were performed|in this (?:study|work)|the cohort)\b", re.I)

# Animal / in-vitro context -- a hit here is not a human study variable.
NON_HUMAN = re.compile(r"\b(mice|mouse|murine|rat|rats|germ-free|in vitro|"
                       r"animal model|zebrafish|C57BL)\b", re.I)

NUM = r"[-+]?\d+(?:\.\d+)?"

VARS = {
    "recruitment setting": {
        "kw": re.compile(r"\b(hospital|clinic|outpatient|inpatient|community|"
                         r"nursing home|ward|department of neurology)\b", re.I),
        # a value = an explicit statement of WHERE subjects came from
        "val": re.compile(r"\b(recruited|enrolled|admitted|referred|consecutive)\b"
                          r"[^.]{0,120}\b(hospital|clinic|outpatient|inpatient|"
                          r"community|ward|centre|center)\b"
                          r"|\b(hospital|clinic|outpatient|inpatient|community)\b"
                          r"[^.]{0,120}\b(recruited|enrolled|admitted|referred)\b", re.I),
    },
    "sample storage": {
        "kw": re.compile(r"\b(stored|storage|frozen|freezer|-80|−80|-20|−20|"
                         r"liquid nitrogen|dry ice)\b", re.I),
        "val": re.compile(r"(?:[-−]\s?(?:80|20|196)\s?°?\s?C)|liquid nitrogen|"
                          r"\bstored\b[^.]{0,60}\b(?:within|immediately|"
                          rf"{NUM}\s*(?:h|hours|min|minutes|days))\b", re.I),
    },
    "differential abundance method": {
        "kw": re.compile(r"\b(LEfSe|LDA|DESeq2|edgeR|ANCOM|ALDEx2|MaAsLin|"
                         r"Wilcoxon|Mann-?Whitney|Kruskal-?Wallis|"
                         r"differential abundance|metagenomeSeq|t-test)\b", re.I),
        # named test IS the value
        "val": re.compile(r"\b(LEfSe|DESeq2|edgeR|ANCOM(?:-BC)?|ALDEx2|MaAsLin2?|"
                          r"metagenomeSeq|Wilcoxon rank[- ]sum|Mann-?Whitney|"
                          r"Kruskal-?Wallis|linear discriminant analysis)\b", re.I),
    },
    "antibiotic use": {
        "kw": re.compile(r"\b(antibiotic|antibiotics|antimicrobial)\b", re.I),
        "val": re.compile(r"\b(antibiotic|antibiotics|antimicrobial)s?\b[^.]{0,140}"
                          rf"\b(?:within|in the (?:past|previous)|prior)\b[^.]{{0,40}}"
                          rf"{NUM}\s*(?:day|days|week|weeks|month|months|year|years)"
                          r"|\b(?:excluded|exclusion|inclusion|criteri\w+|free of|"
                          r"no (?:recent )?use of)\b[^.]{0,140}"
                          r"\b(antibiotic|antibiotics|antimicrobial)s?\b", re.I),
    },
    "bmi": {
        "kw": re.compile(r"\b(BMI|body mass index)\b", re.I),
        "val": re.compile(rf"\b(?:BMI|body mass index)\b[^.]{{0,80}}{NUM}"
                          rf"|{NUM}\s*(?:kg/m|kg m)", re.I),
    },
    "probiotic use": {
        "kw": re.compile(r"\b(probiotic|prebiotic|synbiotic|yogurt|fermented)\b", re.I),
        "val": re.compile(r"\b(?:excluded|exclusion|inclusion|criteri\w+|"
                          r"free of|no (?:recent )?use of|within)\b[^.]{0,140}"
                          r"\b(probiotic|prebiotic|synbiotic)s?\b"
                          r"|\b(probiotic|prebiotic|synbiotic)s?\b[^.]{0,140}"
                          rf"\bwithin\b[^.]{{0,40}}{NUM}\s*(?:day|week|month)", re.I),
    },
    "disease severity": {
        "kw": re.compile(r"\b(severity|UPDRS|Hoehn|Yahr|EDSS|MMSE|MoCA|NIHSS|"
                         r"CDR|ALSFRS|H&Y)\b", re.I),
        # value = a named scale with a score
        "val": re.compile(rf"\b(?:UPDRS|Hoehn[- ]?(?:and|&)?[- ]?Yahr|H&Y|EDSS|MMSE|"
                          rf"MoCA|NIHSS|CDR|ALSFRS(?:-R)?)\b[^.]{{0,80}}{NUM}", re.I),
    },
    "dna extraction kit": {
        "kw": re.compile(r"\b(DNA extraction|extraction kit|QIAamp|PowerSoil|"
                         r"MoBio|Qiagen|FastDNA|MagAttract)\b", re.I),
        "val": re.compile(r"\b(QIAamp|PowerSoil|MoBio|FastDNA|MagAttract|"
                          r"E\.?Z\.?N\.?A|TIANamp|Omega Bio-?tek)\b[^.]{0,60}"
                          r"|\b(?:DNA (?:was )?(?:extracted|isolated))\b[^.]{0,100}"
                          r"\b(?:kit|Qiagen|QIAamp|PowerSoil|MoBio)\b", re.I),
    },
}


def load():
    papers = []
    for src in SOURCES:
        if not os.path.exists(src):
            print(f"missing {src}", file=sys.stderr)
            continue
        for p in json.load(open(src)):
            t = p.get("text") or ""
            if t:
                papers.append({"title": (p.get("title") or "").strip(), "text": t})
    # de-dupe on title; the two sources are meant to be disjoint but verify
    seen, out = set(), []
    for p in papers:
        k = p["title"].lower()
        if k and k not in seen:
            seen.add(k)
            out.append(p)
    return out


def body_of(text):
    """Drop front matter and the reference list."""
    m = list(REF_HEAD.finditer(text))
    end = m[-1].start() if m and m[-1].start() > len(text) * 0.4 else len(text)
    return text[FRONT_CHARS:end]


def main():
    papers = load()
    print(f"{len(papers)} papers (deduped on title)\n")

    rows = []
    for name, spec in VARS.items():
        kw, val = spec["kw"], spec["val"]
        naive = body = attributed = valued = 0
        examples = []
        for p in papers:
            text = p["text"]
            if kw.search(text):
                naive += 1
            b = body_of(text)
            if not kw.search(b):
                continue
            body += 1
            sents = [s for s in SENT.split(b) if kw.search(s)]
            own = [s for s in sents
                   if OWN_STUDY.search(s) and not NON_HUMAN.search(s)]
            if own:
                attributed += 1
            hit = [s for s in own if val.search(s)]
            if hit:
                valued += 1
                if len(examples) < 3:
                    examples.append(re.sub(r"\s+", " ", hit[0])[:220])
        n = len(papers)
        rows.append({
            "variable": name,
            "naive": naive, "body": body,
            "attributed": attributed, "valued": valued,
            "naive_pct": round(100 * naive / n, 1),
            "valued_pct": round(100 * valued / n, 1),
            "examples": examples,
        })

    rows.sort(key=lambda r: -r["valued"])
    w = max(len(r["variable"]) for r in rows)
    print(f"{'variable':<{w}}  {'NAIVE':>13}  {'BODY':>6}  {'ATTRIB':>6}  {'VALUED':>13}  drop")
    print("-" * (w + 56))
    for r in rows:
        drop = r["naive_pct"] - r["valued_pct"]
        print(f"{r['variable']:<{w}}  {r['naive']:>4} ({r['naive_pct']:>5.1f}%)  "
              f"{r['body']:>6}  {r['attributed']:>6}  "
              f"{r['valued']:>4} ({r['valued_pct']:>5.1f}%)  -{drop:.1f}pt")

    print("\nVALUED examples (the tier an extractor could actually work from):")
    for r in rows[:5]:
        print(f"\n  {r['variable']}  [{r['valued_pct']}%]")
        for e in r["examples"][:2]:
            print(f"    - {e}")

    json.dump({"n_papers": len(papers), "rows": rows}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
