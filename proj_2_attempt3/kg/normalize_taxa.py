#!/usr/bin/env python3
"""Strip annotation-pipeline decoration from taxon names before matching.

WHY. 425 of ~700 gold taxa we "missed" do not appear in the paper text at all --
not because the text is missing, but because the curator recorded the string an
annotation PIPELINE produced rather than the name the paper prints:

    Firmicutes_A                GTDB splits a polyphyletic taxon with a letter
                                suffix; the paper says "Firmicutes"
    Prevotella 9                SILVA numbers ambiguous groups; the paper says
                                "Prevotella"
    unclassified Veillonella    what a classifier emits when it cannot resolve
                                below the named rank
    [Eubacterium] ventriosum    square brackets mark a name whose placement is
      group                     disputed; "group" is a 16S clade label, not a taxon
    g__Blautia                  QIIME/greengenes rank prefix
    Clostridium_P perfringens   GTDB suffix on the genus inside a binomial

None of these are different organisms. They are the same organism wearing a
database's internal notation, and exact/fuzzy string matching treats them as
unrelated to the plain name.

This module normalises both sides of a comparison so the decoration cannot cause
a spurious miss. It is deliberately CONSERVATIVE: it removes notation, never
changes rank and never maps one taxon onto another. `Firmicutes_A` -> `firmicutes`
is safe (same phylum, GTDB's internal split); `Blautia` -> `Lachnospiraceae` would
NOT be, and is not done here -- that is the LCA metric's job, and it belongs only
in scoring, never in graph construction.

    python normalize_taxa.py            # measure the recovery on the current gold
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# QIIME / greengenes rank prefixes: d__ k__ p__ c__ o__ f__ g__ s__
_PREFIX = re.compile(r"^[dkpcofgs]__", re.I)
# GTDB polyphyly suffix: Firmicutes_A, Clostridium_P. Letters only, 1-2 of them,
# and only at a word end, so "E_coli"-style typos are not caught by accident.
_GTDB = re.compile(r"_[A-Z]{1,2}\b")
# SILVA numbered clades: "Prevotella 9", "Ruminococcus 1", "UCG-005", "group 2"
_SILVA_NUM = re.compile(r"\s+\d{1,3}$")
_UCG = re.compile(r"\b(ucg|ncg|gca)[- ]?\d+\b", re.I)
# classifier placeholders
_UNCL = re.compile(r"^(unclassified|unidentified|uncultured|unknown|other|"
                   r"candidatus|incertae[ _]sedis)\s+", re.I)
_TRAIL = re.compile(r"\s+(group|clade|complex|sensu[ _]stricto|cluster|"
                    r"subgroup|lineage|type)(\s+\d+)?$", re.I)
# disputed-placement brackets: [Eubacterium], [Ruminococcus]
_BRACKET = re.compile(r"[\[\]]")


def normalize(name):
    """Taxon string -> comparison key. Conservative: strips notation only."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = _BRACKET.sub(" ", s)
    s = _PREFIX.sub("", s)
    s = _UNCL.sub("", s)
    s = _UCG.sub(" ", s)
    # GTDB suffixes are uppercase in the source; apply before lowering would be
    # ideal, so run it on the ORIGINAL casing too and union the results
    s = _GTDB.sub("", name.strip()).lower() if _GTDB.search(name.strip()) else s
    s = _BRACKET.sub(" ", s)
    s = _PREFIX.sub("", s)
    s = _UNCL.sub("", s)
    # repeat trailing/number strips until stable ("Prevotella 9 group" needs two)
    for _ in range(3):
        t = _TRAIL.sub("", s)
        t = _SILVA_NUM.sub("", t)
        if t == s:
            break
        s = t
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def variants(name):
    """The raw lowercase form plus the normalised key, deduped."""
    raw = (name or "").strip().lower()
    n = normalize(name)
    return [v for v in dict.fromkeys([raw, n]) if v]


def _demo():
    cases = [
        "Firmicutes_A", "Prevotella 9", "unclassified Veillonella",
        "[Eubacterium] ventriosum group", "g__Blautia", "Clostridium_P perfringens",
        "Ruminococcaceae UCG-005", "Clostridium sensu stricto 1",
        "uncultured Bacteroides", "Lachnospiraceae NK4A136 group",
        "Bacteroides",            # already clean -- must be untouched
        "Faecalibacterium prausnitzii",
    ]
    print(f"{'raw':<38} -> normalised")
    print("-" * 64)
    for c in cases:
        print(f"{c:<38} -> {normalize(c)}")


def _measure():
    """How many 'missing' gold taxa does normalisation recover?"""
    gold_csv = "/Users/mohak/Downloads/high_confidence - final_constrained_override.csv"
    if not os.path.exists(gold_csv):
        print(f"\n(gold not found at {gold_csv}; skipping measurement)")
        return

    def parse(s):
        if not isinstance(s, str):
            return []
        s = re.sub(r"\(?p\s*[<>=]\s*0?\.\d+\)?", "", s, flags=re.I)
        return [t.strip() for t in re.split(r"[;,]", s) if len(t.strip()) > 2]

    def doi_of(s):
        m = re.search(r"10\.\d{4,9}/[^\s\"<>,;\]]+", s or "")
        return m.group(0).rstrip(".").lower() if m else None

    g = {}
    for x in csv.DictReader(open(gold_csv, encoding="utf-8-sig")):
        d = x["DOI"].strip().lower()
        g.setdefault(d, [])
        g[d] += parse(x["high_confidence_taxa"])

    d2t = {}
    sheet = os.path.join(HERE, "Microbiota Signatures Neurological Disorders "
                               "Sheet 2 - Main Datasheet.csv")
    for x in csv.DictReader(open(sheet, encoding="utf-8-sig")):
        d = doi_of((x.get("DOI") or "").strip().lower()) or ""
        t = (x.get("Title") or "").strip()
        if d and t:
            d2t[d] = t.lower()

    txt = {}
    for src in ["../EmilySong_GoldStandardPaper/all_usable_papers.json",
                "new_papers.json"]:
        p = os.path.join(HERE, src)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            t = r.get("title")
            if isinstance(t, str) and r.get("text"):
                txt[t.strip().lower()] = r["text"].lower()

    raw_hit = norm_hit = total = 0
    recovered = []
    for doi, taxa in g.items():
        t = d2t.get(doi)
        if not t or t not in txt:
            continue
        body = txt[t]
        for gt in taxa:
            total += 1
            r_ok = gt.lower() in body
            nk = normalize(gt)
            n_ok = bool(r_ok or (nk and nk in body))
            raw_hit += r_ok
            norm_hit += n_ok
            if n_ok and not r_ok and len(recovered) < 12:
                recovered.append((gt, normalize(gt)))

    print(f"\ngold taxa checked against paper text : {total}")
    print(f"  found using the RAW string          : {raw_hit} ({100*raw_hit/total:.1f}%)")
    print(f"  found after normalisation           : {norm_hit} ({100*norm_hit/total:.1f}%)")
    print(f"  RECOVERED by normalisation          : {norm_hit-raw_hit}")
    if recovered:
        print("\n  examples recovered:")
        for a, b in recovered:
            print(f"    {a:<40} -> {b}")


if __name__ == "__main__":
    _demo()
    _measure()
