#!/usr/bin/env python3
"""A second metadata pass: the wet-lab and bioinformatics variables we never extracted.

`FINDINGS_paper_discordance.md` established that disagreement with the rest of the
literature is a real paper-level property (within-edge null p = 0.0003) and that
NOTHING we currently extract explains it -- country, cohort size, sequencing
platform, 16S region, medication and diet control and disease identity are all
null at MDEs of 16-22% once the exact within-edge expectation is used as offset.

The variables the microbiome methods literature actually blames for cohorts
disagreeing are not in `metadata.jsonl`: DNA extraction kit, primer set, the
bioinformatics pipeline, OTU vs ASV, rarefaction, and above all the
differential-abundance test. Every recent session has filed this under "needs a
GPU". It does not. Full text for **all 272 contributing papers** is already in the
repo, spread across three files, and the variables of interest are TOOL NAMES --
literal strings. A regex reads them deterministically and, unlike an LLM pass,
gives the same answer twice.

Sources, preferring the longest text where a paper appears in more than one:
  ../EmilySong_GoldStandardPaper/all_usable_papers.json   (250)
  extract_input.json                                      (98)
  new_papers.json                                         (53)

SCOPING. Tool names recur in reference lists and in discussion of other people's
work, so detection runs on the METHODS SECTION where one can be located (heading
match through to Results/Discussion) and on the reference-stripped body otherwise.
Which was used is recorded per paper.

VALIDATION, before any of this is used as a predictor. Two deterministic checks
that need no adjudication:
  1. Against the existing LLM-extracted `sequencing` field (16S vs shotgun) --
     an independent label on the one axis where we have one.
  2. Against the existing LLM-extracted `region_16S` field.
Agreement is reported; a detector that cannot reproduce a label we already trust
has no business predicting anything.

Writes methods_metadata.json.  CPU-only, no network, no MAIN_DATA, no GPU.
"""

import json
import re
import sys
from collections import Counter, defaultdict

SOURCES = [
    "../EmilySong_GoldStandardPaper/all_usable_papers.json",
    "extract_input.json",
    "new_papers.json",
]

# Each family maps a label to a regex. Word boundaries everywhere; version digits
# and punctuation are handled per pattern rather than globally.
PATTERNS = {
    "extraction_kit": {
        "QIAamp": r"\bqiaamp\b",
        "DNeasy": r"\bdneasy\b",
        "PowerSoil": r"\bpower\s?soil\b",
        "PowerFecal": r"\bpower\s?fecal\b",
        "MoBio": r"\bmo\s?bio\b",
        "FastDNA": r"\bfast\s?dna\b",
        "MagAttract": r"\bmagattract\b",
        "TIANamp": r"\btian(amp|gen)\b",
        "Omega": r"\bomega\s+bio-?tek\b|\be\.?z\.?n\.?a\.?\b",
        "phenol-chloroform": r"\bphenol[- ]chloroform\b",
        "CTAB": r"\bctab\b",
        "Zymo": r"\bzymo(biomics)?\b",
        "NucleoSpin": r"\bnucleospin\b",
    },
    "platform": {
        "MiSeq": r"\bmiseq\b",
        "HiSeq": r"\bhiseq\b",
        "NovaSeq": r"\bnovaseq\b",
        "NextSeq": r"\bnextseq\b",
        "IonTorrent": r"\bion\s+(torrent|pgm|s5)\b",
        "PacBio": r"\bpacbio\b|\bpacific\s+biosciences\b",
        "454": r"\b454\s+(pyrosequenc|gs\s|life\s+sciences)|\bpyrosequencing\b",
        "BGI": r"\bbgiseq\b|\bdnbseq\b",
        "Nanopore": r"\bnanopore\b|\bminion\b",
    },
    "pipeline": {
        "QIIME2": r"\bqiime\s?2\b|\bqiime2\b",
        "QIIME1": r"\bqiime\s?1\b|\bqiime\b(?!\s?2)",
        "DADA2": r"\bdada2\b",
        "mothur": r"\bmothur\b",
        "UPARSE": r"\buparse\b",
        "USEARCH": r"\busearch\b",
        "VSEARCH": r"\bvsearch\b",
        "Deblur": r"\bdeblur\b",
        "Kraken": r"\bkraken\b|\bbracken\b",
        "MetaPhlAn": r"\bmetaphlan\b",
        "HUMAnN": r"\bhumann\b",
    },
    "feature_type": {
        "ASV": r"\basvs?\b|\bamplicon\s+sequence\s+variants?\b",
        "OTU": r"\botus?\b|\boperational\s+taxonomic\s+units?\b",
    },
    "diff_abundance": {
        "LEfSe": r"\blefse\b|\blinear\s+discriminant\s+analysis\s+effect\s+size\b",
        "LDA": r"\blda\s+score\b|\blinear\s+discriminant\s+analysis\b",
        "DESeq2": r"\bdeseq\s?2\b",
        "edgeR": r"\bedger\b",
        "ANCOM": r"\bancom\b",
        "MaAsLin": r"\bmaaslin\b",
        "metagenomeSeq": r"\bmetagenomeseq\b",
        "ALDEx2": r"\baldex\s?2\b",
        "Wilcoxon": r"\bwilcoxon\b|\bmann[- ]whitney\b",
        "KruskalWallis": r"\bkruskal[- ]wallis\b",
        "t-test": r"\bt[- ]tests?\b|\bstudent'?s?\s+t\b",
        "RandomForest": r"\brandom\s+forests?\b",
        "PERMANOVA": r"\bpermanova\b|\badonis\b",
    },
    "normalisation": {
        "rarefied": r"\brarefi(ed|action)\b|\brarefy\b",
        "relative_abundance": r"\brelative\s+abundance\b",
        "absolute_quant": r"\bqpcr\b|\babsolute\s+(abundance|quantification)\b"
                          r"|\bspike[- ]in\b",
        "CLR": r"\bcentered?\s+log[- ]ratio\b|\bclr\s+transform",
    },
    "multiple_testing": {
        "FDR": r"\bfdr\b|\bfalse\s+discovery\s+rate\b|\bbenjamini\b",
        "Bonferroni": r"\bbonferroni\b",
    },
    "amplicon_region": {
        "V3-V4": r"\bv3\s?[-–]\s?v4\b|\bv3v4\b",
        "V4": r"\bv4\b",
        "V1-V2": r"\bv1\s?[-–]\s?v2\b",
        "V4-V5": r"\bv4\s?[-–]\s?v5\b",
        "V1-V3": r"\bv1\s?[-–]\s?v3\b",
        "V3": r"\bv3\b",
        "V6": r"\bv6\b",
    },
    "assay": {
        "16S": r"\b16s\b",
        "shotgun": r"\bshotgun\b|\bmetagenomic\s+sequencing\b|\bwgs\b",
        "ITS": r"\bits[12]?\s+region\b",
    },
}

# These texts are cleaned into ONE flowing string with no line breaks -- headings
# sit inline ("...are examined. Materials and methods Patients This was a..."), so
# an anchored ^heading$ pattern can never fire. It did not: the first run scoped
# 272 of 272 papers to the full body. Locating a heading inline instead means
# separating it from ordinary prose ("drug intervention methods for MHE"), which
# two cheap signals do: a heading is Capitalised and is followed by another
# capitalised word, and it is not preceded by a determiner.
_UNAMBIGUOUS = (r"(?:Materials?|Patients?|Subjects?|Participants?)\s+and\s+[Mm]ethods"
                r"|[Mm]ethods\s+and\s+[Mm]aterials"
                r"|Experimental\s+(?:Procedures|Section|Design)"
                r"|METHODS")
METHODS_START = re.compile(r"(?<![A-Za-z])(?:" + _UNAMBIGUOUS + r")(?![a-z])")
# bare "Methods" as a fallback: sentence-initial and followed by a capital
METHODS_START_WEAK = re.compile(r"(?:[.;]\s+|^)(Methods?)\s+(?=[A-Z])")
_NOT_AFTER = re.compile(r"(?:\b(?:the|The|these|These|our|Our|its|Its|whose|and|"
                        r"or|of|in|for|with|to)\s+)$")
METHODS_END = re.compile(r"(?<![A-Za-z])(Results?|Discussion|Conclusions?)"
                         r"(?![a-z])\s+(?=[A-Z0-9])")
REFS = re.compile(r"(?<![A-Za-z])(References|REFERENCES|Bibliography|"
                  r"Literature\s+Cited)(?![a-z])\s+(?=[A-Z0-9\d])")


YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def guess_year(text):
    """Publication year from the article header. Validated in main() against the
    papers that carry an explicit `year` field."""
    ys = [int(x) for x in YEAR.findall(text[:900])]
    return max(ys) if ys else None


def load_texts():
    best = {}
    provenance = {}
    for path in SOURCES:
        try:
            data = json.load(open(path))
        except FileNotFoundError:
            print(f"  (missing: {path})")
            continue
        recs = data if isinstance(data, list) else list(data.values())
        for r in recs:
            if not isinstance(r, dict):
                continue
            t = norm_title(r.get("title"))
            txt = r.get("text") or r.get("full_text") or ""
            if not t or not isinstance(txt, str) or len(txt) < 2000:
                continue
            if len(txt) > len(best.get(t, "")):
                best[t] = txt
                provenance[t] = path
    return best, provenance


def strip_refs(text):
    """Cut at the LAST reference heading that sits in the back half of the paper."""
    cut = None
    for m in REFS.finditer(text):
        if m.start() > len(text) * 0.4:
            cut = m.start()
    return text[:cut] if cut else text


def _heading_ok(body, pos):
    return not _NOT_AFTER.search(body[max(0, pos - 12):pos])


def methods_section(text):
    """-> (segment, scope_label). Ends at the first Results/Discussion heading."""
    body = strip_refs(text)
    start = None
    for m in METHODS_START.finditer(body):
        if m.start() < len(body) * 0.75 and _heading_ok(body, m.start()):
            start = m.end()
            break
    scope = "methods_section"
    if start is None:
        for m in METHODS_START_WEAK.finditer(body):
            if m.start(1) < len(body) * 0.6 and _heading_ok(body, m.start(1)):
                start = m.end(1)
                scope = "methods_section_weak"
                break
    if start is None:
        return body, "full_body"
    end = None
    for m in METHODS_END.finditer(body, start):
        if _heading_ok(body, m.start()):
            end = m.start()
            break
    seg = body[start:end] if end else body[start:]
    if len(seg) < 800:
        return body, "full_body"
    return seg, scope


def detect(segment):
    low = segment.lower()
    out = {}
    for family, pats in PATTERNS.items():
        hits = [name for name, rx in pats.items() if re.search(rx, low)]
        out[family] = hits
    # QIIME1 pattern deliberately also matches bare "qiime"; if QIIME2 fired,
    # a bare mention is not evidence of version 1
    if "QIIME2" in out["pipeline"] and "QIIME1" in out["pipeline"]:
        out["pipeline"].remove("QIIME1")
    # Amplicon regions nest: "V3-V4" contains both "V3" and "V4", so the first
    # run reported V4 in 162 papers and V3 in 142 with only 103 V3-V4. Keep the
    # most specific span that fired and drop the sub-regions it contains.
    regions = out["amplicon_region"]
    for span in ("V1-V3", "V1-V2", "V3-V4", "V4-V5"):
        if span in regions:
            a, b = span.split("-")
            for sub in (a, b):
                if sub in regions:
                    regions.remove(sub)
    out["amplicon_region"] = regions
    return out


def main():
    print("loading full texts")
    texts, prov = load_texts()
    print(f"  {len(texts)} distinct papers with full text")

    graph = json.load(open("graph.json"))
    gtitles = [norm_title(p["title"]) for p in graph["papers"]]
    covered = sum(1 for t in gtitles if t in texts)
    print(f"  covers {covered}/{len(gtitles)} graph papers")

    meta_by_title = {}
    for line in open("metadata.jsonl"):
        r = json.loads(line)
        meta_by_title[norm_title(r.get("title"))] = r.get("meta") or {}

    records = []
    scopes = Counter()
    for i, t in enumerate(gtitles):
        if t not in texts:
            records.append({"paper": i, "title": graph["papers"][i]["title"][:70],
                            "have_text": False})
            continue
        seg, scope = methods_section(texts[t])
        scopes[scope] += 1
        d = detect(seg)
        records.append({"paper": i, "title": graph["papers"][i]["title"][:70],
                        "have_text": True, "scope": scope,
                        "scope_chars": len(seg), "source": prov[t], **d})
    print(f"  scope: {dict(scopes)}")

    # ---- validation against labels we already trust ----
    print("\nVALIDATION against the existing LLM-extracted metadata")
    agree = tot = 0
    conf = Counter()
    for i, t in enumerate(gtitles):
        r = records[i]
        if not r.get("have_text"):
            continue
        m = meta_by_title.get(t)
        if not m or not m.get("sequencing"):
            continue
        llm = m["sequencing"].strip().lower()
        llm_is_16s = llm.startswith("16s")
        llm_is_shot = "shotgun" in llm or "metagenom" in llm
        if not (llm_is_16s or llm_is_shot):
            continue
        assay = r["assay"]
        # the detector's call: 16S unless shotgun is present without 16S
        det_16s = "16S" in assay
        det_shot = "shotgun" in assay
        if det_16s and not det_shot:
            call = "16S"
        elif det_shot and not det_16s:
            call = "shotgun"
        elif det_16s and det_shot:
            call = "both"
        else:
            call = "none"
        truth = "16S" if llm_is_16s else "shotgun"
        conf[(truth, call)] += 1
        tot += 1
        agree += (call == truth)
    print(f"  assay 16S/shotgun: {agree}/{tot} exact agreement "
          f"({agree/tot:.1%})" if tot else "  no comparable labels")
    for k, v in sorted(conf.items(), key=lambda kv: -kv[1]):
        print(f"    LLM={k[0]:<8} regex={k[1]:<8} {v}")

    ragree = rtot = 0
    for i, t in enumerate(gtitles):
        r = records[i]
        m = meta_by_title.get(t)
        if not r.get("have_text") or not m or not m.get("region_16S"):
            continue
        rtot += 1
        ragree += m["region_16S"].strip().upper().replace("–", "-") in \
            {x.upper() for x in r["amplicon_region"]}
    if rtot:
        print(f"  16S region: regex recovers the LLM's label for "
              f"{ragree}/{rtot} ({ragree/rtot:.1%})")

    print("\nCoverage of the new variables (papers with text, n="
          f"{sum(1 for r in records if r.get('have_text'))})")
    for family in PATTERNS:
        c = Counter()
        for r in records:
            if not r.get("have_text"):
                continue
            for h in r[family]:
                c[h] += 1
            if not r[family]:
                c["(none)"] += 1
        top = ", ".join(f"{k} {v}" for k, v in c.most_common(6))
        print(f"  {family:18s} {top}")

    # ---- publication year, with its own validation ----
    known = {}
    for path in SOURCES:
        try:
            data = json.load(open(path))
        except FileNotFoundError:
            continue
        for r in (data if isinstance(data, list) else list(data.values())):
            if not isinstance(r, dict):
                continue
            t, y = norm_title(r.get("title")), r.get("year")
            if t and y and str(y).strip()[:4].isdigit():
                known.setdefault(t, int(str(y)[:4]))
    years, exact, ytot = {}, 0, 0
    for i, t in enumerate(gtitles):
        if t not in texts:
            continue
        y = guess_year(texts[t])
        if y is None:
            continue
        years[i] = y
        if t in known:
            ytot += 1
            exact += (y == known[t])
    print(f"\n  publication year parsed for {len(years)}/{len(gtitles)} papers; "
          f"against the {ytot} carrying an explicit year field, "
          f"{exact} exact ({exact/ytot:.1%})" if ytot else "")
    print("  caveat: those come from the sources that HAVE the field, so the "
          "check is not a random sample of the corpus")
    with open("paper_years.json", "w") as fh:
        json.dump({str(k): v for k, v in sorted(years.items())}, fh,
                  indent=1, sort_keys=True)
    print("  wrote paper_years.json")

    out = {"n_papers": len(records),
           "year": {"n_parsed": len(years), "validated_against": ytot,
                    "exact": exact},
           "n_with_text": sum(1 for r in records if r.get("have_text")),
           "scopes": dict(scopes),
           "validation": {"assay_agree": agree, "assay_total": tot,
                          "assay_confusion": {f"{k[0]}|{k[1]}": v
                                              for k, v in conf.items()},
                          "region_agree": ragree, "region_total": rtot},
           "families": {k: sorted(v) for k, v in PATTERNS.items()},
           "records": records}
    with open("methods_metadata.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote methods_metadata.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
