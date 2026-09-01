#!/usr/bin/env python3
"""Screen the 45 title-matched MAIN_DATA papers out of the extraction set.

Those 45 entered the corpus by matching neuro keywords against MAIN_DATA.json
titles. Unlike the 303 datasheet papers they were never screened for study
design, and agreement with Disbiome and Peryton fell when they arrived. Reading
them (kg/maindata_screen.json) shows 22 of 45 are not human case-control studies
at all: 15 report their microbiome from mice or rats, 3 have no healthy control
arm, 2 are n<=2 case reports, 2 report no primary cohort.

That matters because the extractor was never asked "is this a human case-control
study?" -- it was asked which taxa go up and down. Point it at a paper where
3xTgAD mice differ from wild-type littermates and it will faithfully return that
contrast as if it were a disease-vs-healthy human finding. The edges are not
extraction errors; they are correct readings of papers that should not be in a
human microbe-disease graph.

Emits three variants so the effect can be attributed rather than assumed:

  all348      baseline, everything
  screened    drop only the 22 that fail the screen  (the proposed corpus)
  no_maindata drop all 45 title-matched papers       (control)

`screened` vs `no_maindata` is the informative contrast. If screening recovers
agreement and dropping everything recovers no more, the problem is the unscreened
studies specifically, not the MAIN_DATA provenance.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "extractions_corrected.json")
SCREEN = os.path.join(HERE, "maindata_screen.json")


def norm(s):
    s = re.sub(r"^#+\s*PAPER_ID:\s*", "", (s or "").strip())
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    rows = json.load(open(SRC))
    screen = json.load(open(SCREEN))
    by_norm = {norm(t): v for t, v in screen.items()}
    assert len(by_norm) == 45, len(by_norm)

    drop_fail, drop_all = set(), set()
    seen = 0
    for r in rows:
        n = norm(r.get("title"))
        v = by_norm.get(n)
        if v is None:
            continue
        seen += 1
        drop_all.add(n)
        if v["category"] != "KEEP":
            drop_fail.add(n)
    assert seen == 45, f"only matched {seen} of the 45 screened papers in {SRC}"

    variants = {
        "all348": set(),
        "screened": drop_fail,
        "no_maindata": drop_all,
    }
    for name, drop in variants.items():
        kept = [r for r in rows if norm(r.get("title")) not in drop]
        out = os.path.join(HERE, f"_variant_{name}.json")
        json.dump(kept, open(out, "w"), indent=1)
        print(f"{name:12} kept {len(kept):3} papers  (dropped {len(rows)-len(kept)})  -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
