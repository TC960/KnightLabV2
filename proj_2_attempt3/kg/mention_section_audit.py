#!/usr/bin/env python3
"""Adversarial check on the 100% mention rate: WHERE in the paper is the taxon?

`silent_edge_mentions.py` now reports that all 3,077 observations name their
taxon in their source paper. A rate that high is exactly when an instrument
deserves to be attacked rather than quoted, and this one has an obvious attack:

    "mentioned anywhere in the document" includes the REFERENCE LIST.

A cited article titled *"...Akkermansia muciniphila in type 2 diabetes"* puts
that taxon in our paper's text while supporting nothing whatever about our
paper's cohort. If a meaningful share of `silent` observations are carried by
reference-list hits, the 100% is partly an artifact and must be stated with that
caveat. If essentially none are, the number survives a real attempt to break it.

Method, deterministic throughout:
  - locate the reference section by heading (References / Bibliography / Literature
    Cited / Works Cited) taken as the LAST such heading in the document, so an
    in-text phrase like "see references" cannot truncate the body;
  - for every observation, find all match offsets of the taxon's surface forms;
  - classify: body_only / both / references_only / (absent).

`references_only` is the failure mode. `both` is fine -- the taxon is in the body
AND happens to appear in a citation title.

Reported per provenance class, because the classes differ in what they claim:
`own` and `background` observations are anchored to a sentence by construction
and so cannot be reference-only; that is a built-in positive control, and if it
fails the section finder is wrong rather than the graph.
"""
import json
import os
import re
from collections import Counter, defaultdict

import witness_discordance as W
from silent_edge_mentions import abbrev_map, key, load_fulltext_sources
from verify_taxon_mentions import norm_taxon, squash, variants

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mention_section_audit.json")

REF_HEADING = re.compile(
    r"\n\s*(?:\d+\s*[.)]?\s*)?(references|bibliography|literature cited|"
    r"works cited|reference list)\s*\n", re.I)
# Fallback: some cleaned texts lose the newlines around the heading.
REF_INLINE = re.compile(r"\breferences\b\s*:?\s*(?=\s*\[?\s*1\s*[.\]])", re.I)


def ref_start(text):
    """Character offset where the reference list begins, or None."""
    hits = list(REF_HEADING.finditer(text))
    if hits:
        cand = hits[-1].start()
        # A heading in the first half is almost certainly not the reference list.
        if cand > len(text) * 0.4:
            return cand
    hits = list(REF_INLINE.finditer(text))
    if hits and hits[-1].start() > len(text) * 0.4:
        return hits[-1].start()
    return None


def offsets(forms, text, amap):
    """Every offset at which any surface form occurs.

    MUST be at least as strong as the matcher this audit is checking, or it
    reports body mentions as reference-only. The first cut was weaker and did
    exactly that on both of its two hits: it missed `p. timonensis` (the body
    reports it with an adjusted p-value) and `es. coli`, because it searched
    only full spellings. Sixth time in this repo an instrument came out weaker
    than the thing it audited -- so the abbreviated forms are searched too:

      - `G. species` / `Ge. species` / `Gen. species` -- an abbreviation is a
        PREFIX of the genus, so 1-3 leading characters are searched. Papers use
        undeclared ones freely: the MCI paper writes `es. coli` and
        `su. sp apc924` having defined neither.
      - `ab. species` -- the abbreviation THIS paper defined for that genus.

    The epithet must follow immediately, which is what keeps a 1-3 character
    prefix from matching loosely.
    """
    out = []
    for f in forms:
        for v in {norm_taxon(f)} | set(variants(norm_taxon(f))):
            if len(v) >= 4:
                out += [m.start() for m in re.finditer(re.escape(v), text)]
        toks = [t for t in re.sub(r"[^A-Za-z0-9 ]", " ", f).split() if t]
        if len(toks) < 2:
            continue
        genus, rest = toks[0].lower(), " ".join(toks[1:]).lower()
        abbrevs = {genus[:n] for n in (1, 2, 3)}
        rev = {g: a for a, g in amap.items()}
        if genus in rev:
            abbrevs.add(rev[genus])
        for ab in abbrevs:
            pat = r"\b%s\.?\s*%s" % (re.escape(ab), re.escape(rest).replace(r"\ ", r"\s*"))
            out += [m.start() for m in re.finditer(pat, text)]
    return out


# Heading detection only reaches the 123 contributing papers that still carry a
# recognisable "References" heading -- roughly half. A DOI-density heuristic was
# tried to rescue the rest and REJECTED: these are PMC scrapes whose journal
# header carries a DOI at offset 53, so density measures boilerplate, not a
# bibliography.
#
# So the same question is asked a second way, with no dependence on headings or
# document structure at all: does the taxon mention SIT INSIDE A CITATION? A
# bibliography entry looks like "J Reprod Immunol. 2021;12:587696. doi:10...."
# -- a doi or a journal-style volume:page -- within a short window of the
# mention. If EVERY offset for a taxon is citation-like, the taxon is carried by
# a citation alone, whatever section it sits in. This covers all 271 papers and
# is the primary result; the heading-based table corroborates it.
#
# A zero result is reported with its own sensitivity, because a test that cannot
# fire proves nothing: see calibrate() below.
# Only BIBLIOGRAPHY-ENTRY markers: a doi, or a journal-style `YYYY;vol:page`.
# An earlier version also counted "Author et al., 2021" and bare journal names,
# and the built-in positive control caught it: it flagged 4 `own` observations,
# one of which reads "COMT inhibitor use was associated with overrepresentation
# of Bifidobacteriaceae ... IN OUR COHORT" -- a first-person result that merely
# has an in-text citation nearby. In-text citation is author+year; a reference
# entry carries a doi or volume:page. Keeping only the latter is what makes this
# a test of bibliography membership rather than of proximity to any citation.
CITE_CUE = re.compile(r"\b(?:doi\s*:|(?:19|20)\d{2}\s*;\s*\d+\s*[:(]\s*\d+)", re.I)
CITE_WINDOW = 200


def calibrate(fulltext, papers, n=40, seed=0):
    """A zero is only a result if the test could have fired. Sample random
    positions inside detected bibliographies (should look citation-like) and
    inside bodies (should not), and report both rates. Sensitivity converts an
    observed zero into an upper bound on the true rate."""
    import random
    rng = random.Random(seed)
    hr = tr = hb = tb = 0
    for p in papers:
        pair = fulltext.get(key(p["title"]))
        if not pair:
            continue
        t = pair[0]
        rs = ref_start(t)
        if rs is None or rs < 100:
            continue
        for _ in range(n):
            o = rng.randrange(rs, len(t))
            tr += 1
            hr += bool(CITE_CUE.search(t[max(0, o - CITE_WINDOW):o + CITE_WINDOW]))
            o = rng.randrange(0, int(rs * 0.8))
            tb += 1
            hb += bool(CITE_CUE.search(t[max(0, o - CITE_WINDOW):o + CITE_WINDOW]))
    return {"sensitivity": round(hr / tr, 4) if tr else None,
            "false_positive_rate_in_body": round(hb / tb, 4) if tb else None,
            "n_sampled_each": tr}


def citation_bound(offs, text):
    """True if EVERY offset sits in citation-looking context."""
    if not offs:
        return None
    for o in offs:
        w = text[max(0, o - CITE_WINDOW):o + CITE_WINDOW]
        if not CITE_CUE.search(w):
            return False
    return True


def main():
    g = json.load(open(os.path.join(HERE, "graph.json")))
    papers, edges = g["papers"], g["edges"]
    nodes = {n["id"]: n for n in g["nodes"]}

    fulltext, src = load_fulltext_sources()
    obs = W.build()

    # Precompute reference offsets per paper.
    refs, no_ref = {}, []
    for k, pair in fulltext.items():
        r = ref_start(pair[0])
        refs[k] = r
        if r is None:
            no_ref.append(k)

    tab = defaultdict(Counter)
    cite = defaultdict(Counter)
    ref_only_rows = []
    cite_rows = []
    cache = {}
    ccache = {}
    for o in obs:
        e = edges[o["edge"]]
        node = nodes.get(e["source"])
        if node is None:
            continue
        pk = key(papers[o["paper"]]["title"])
        pair = fulltext.get(pk)
        if pair is None:
            tab[o["prov"]]["no_text"] += 1
            continue
        ck = (pk, node["id"])
        if ck not in cache:
            forms = [s for s in [node.get("label")] + list(node.get("aliases") or []) if s]
            offs = offsets(forms, pair[0], pair[4])
            r = refs.get(pk)
            if not offs:
                v = "no_offset"          # matched only via a squashed/fuzzy tier
            elif r is None:
                v = "no_ref_section"     # cannot test this paper
            elif all(x >= r for x in offs):
                v = "references_only"
            elif any(x >= r for x in offs):
                v = "both"
            else:
                v = "body_only"
            cache[ck] = v
            ccache[ck] = citation_bound(offs, pair[0])
        v = cache[ck]
        cb = ccache[ck]
        tab[o["prov"]][v] += 1
        cite[o["prov"]]["no_offset" if cb is None
                        else ("citation_only" if cb else "has_real_mention")] += 1
        if cb:
            cite_rows.append({
                "taxon": e["taxon"], "disease": e["disease"], "prov": o["prov"],
                "paper": papers[o["paper"]]["title"],
                "n_papers_on_edge": e.get("n_papers"),
            })
        if v == "references_only":
            ref_only_rows.append({
                "taxon": e["taxon"], "disease": e["disease"], "prov": o["prov"],
                "paper": papers[o["paper"]]["title"],
                "n_papers_on_edge": e.get("n_papers"),
            })

    total = Counter()
    for c in tab.values():
        total.update(c)
    testable = total["body_only"] + total["both"] + total["references_only"]
    out = {
        "n_observations": len(obs),
        "papers_with_no_detectable_reference_section": len(no_ref),
        "by_provenance": {k: dict(v) for k, v in tab.items()},
        "totals": dict(total),
        "n_testable": testable,
        "references_only_rate": round(total["references_only"] / testable, 5) if testable else None,
        "references_only": ref_only_rows,
        "citation_context_by_provenance": {k: dict(v) for k, v in cite.items()},
        "citation_only": cite_rows,
        "citation_cue_calibration": None,  # filled below
    }
    cal = calibrate(fulltext, papers)
    out["citation_cue_calibration"] = cal
    den_c = sum(v.get("citation_only", 0) + v.get("has_real_mention", 0)
                for v in cite.values())
    obs_c = sum(v.get("citation_only", 0) for v in cite.values())
    if cal["sensitivity"] and den_c and obs_c == 0:
        # Rule of three: 0/N gives a 95% upper bound of 3/N on the DETECTED rate;
        # divide by sensitivity for the upper bound on the TRUE rate.
        out["citation_only_rate_95pct_upper_bound"] = round(
            3 / den_c / cal["sensitivity"], 6)
    json.dump(out, open(OUT, "w"), indent=1)
    skip = {"references_only", "citation_only"}
    print(json.dumps({k: v for k, v in out.items() if k not in skip}, indent=1))
    ctot = Counter()
    for c in cite.values():
        ctot.update(c)
    den = ctot["citation_only"] + ctot["has_real_mention"]
    print(f"\nCITATION-CONTEXT TEST (all 271 papers, no heading needed): "
          f"{ctot['citation_only']} of {den} observations are citation-only "
          f"({ctot['citation_only'] / den:.4%})" if den else "")
    for r in cite_rows[:25]:
        print(f"    [{r['prov']:10s}] {r['taxon'][:36]:36s} {r['disease'][:24]}")
    if ref_only_rows:
        print(f"\n--- {len(ref_only_rows)} REFERENCES-ONLY observations ---")
        for r in ref_only_rows[:40]:
            print(f"  [{r['prov']:10s}] {r['taxon'][:38]:38s} {r['disease'][:24]:24s} "
                  f"npap={r['n_papers_on_edge']}")
    else:
        print("\nNo observation is carried by a reference-list mention alone.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
