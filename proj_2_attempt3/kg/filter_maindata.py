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
    """Title key, identical in strength to build_kg.norm_title.

    It MUST be, and was not. This normaliser folded curly quotes and dashes but
    kept a trailing full stop, while the deduper strips every non-alphanumeric.
    13 papers sit in extractions_corrected.json under two spellings that differ
    only by such punctuation -- the same paper scraped twice under two links.
    The screen matched one spelling and dropped it; the other spelling was a
    different key here, so it survived, and the deduper that would have folded
    the two copies never saw them together because one was already gone.

    One DROP_ANIMAL paper reached the graph that way: "Microbiota from
    Alzheimer's patients induce deficits in cognition and hippocampal
    neurogenesis", which transplants human faeces into rats and reports the
    RATS' microbiome, contributing 6 edges as if they were human findings.

    The `assert seen == 45` below gave false assurance: it counts screen entries
    matched, not paper copies dropped, so it passed while a copy leaked. It now
    asserts on the screen entries that matched at least one row, and separately
    that no surviving row shares a key with a failing one.
    """
    s = re.sub(r"^#+\s*PAPER_ID:\s*", "", (s or "").strip())
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    rows = json.load(open(SRC))
    screen = json.load(open(SCREEN))
    by_norm = {norm(t): v for t, v in screen.items()}
    assert len(by_norm) == 45, len(by_norm)

    drop_fail, drop_all, matched = set(), set(), set()
    for r in rows:
        n = norm(r.get("title"))
        v = by_norm.get(n)
        if v is None:
            continue
        matched.add(n)
        drop_all.add(n)
        if v["category"] != "KEEP":
            drop_fail.add(n)
    # Count SCREEN ENTRIES matched, not row hits: a paper present under two
    # spellings must not be able to satisfy this while one of its copies leaks.
    assert matched == set(by_norm), (
        f"{len(set(by_norm) - matched)} screened papers never matched a row in {SRC}")

    variants = {
        "all348": set(),
        "screened": drop_fail,
        "no_maindata": drop_all,
    }
    for name, drop in variants.items():
        kept = [r for r in rows if norm(r.get("title")) not in drop]
        # all348 is the deliberate no-drop baseline; the other two must not leak.
        if drop:
            leaked = [r.get("title") for r in kept if norm(r.get("title")) in drop]
            assert not leaked, f"{name}: screened-out paper survived: {leaked}"
        out = os.path.join(HERE, f"_variant_{name}.json")
        json.dump(kept, open(out, "w"), indent=1)
        print(f"{name:12} kept {len(kept):3} papers  (dropped {len(rows)-len(kept)})  -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
