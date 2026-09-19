#!/usr/bin/env python3
"""Verify and summarise the free-text disease-node adjudication.

Two independent checks are applied to every adjudicated (node, paper) pair
before any verdict is believed:

1. **Verbatim check.** Every quote must be a byte-for-byte substring of the
   sentences that were actually supplied in the packet. The 2026-09-16 edge-recall
   audit found 7 of 36 adjudicator "verbatim" quotes were paraphrases, so this is
   not optional. A verdict whose quotes all fail is downgraded to UNSUPPORTED.

2. **Deterministic cue screen.** An independent regex pass over title+sentences
   for healthy-control / treatment / subgroup / animal language. This is TRIAGE,
   not adjudication -- the 2026-09-11 abstract screen and the 2026-09-16
   provenance screen both scored ~33% precision on this corpus -- so it is used
   only to flag pairs where the reader and the regexes disagree, for a second look.

Reads:  disease_label_packets.json, _dlbatch/verdicts*.json
Writes: disease_label_audit.json
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).parent

CUES = {
    "HC": re.compile(r"health(y|ies)[ -]?(control|subject|volunteer|individual|participant|donor|"
                     r"adult|elder|people|person)|normal control|\bHCs?\b|control group", re.I),
    "TREATMENT": re.compile(r"\bprobiotic|rifaximin|lactulose|\bplacebo\b|randomi[sz]ed|"
                            r"supplementation|\bFMT\b|faecal microbiota transplant|fecal microbiota transplant|"
                            r"after treatment|post-?treatment|responder|\btrial\b", re.I),
    "SUBGROUP": re.compile(r"with and without|with or without|those without|non-?[A-Z]{2,}\s+group|"
                           r"\bMHE\b.*\bNMHE\b|stratified by|severity group", re.I),
    "ANIMAL": re.compile(r"\bmice\b|\bmouse\b|\brats?\b|murine|germ-?free|animal model|\bC57BL", re.I),
}

def load_verdicts():
    out, bad = [], []
    for f in sorted(HERE.glob("_dlbatch/verdicts*.json")):
        try:
            v = json.load(open(f))
        except Exception as e:
            bad.append((f.name, str(e)))
            continue
        if isinstance(v, dict):
            v = v.get("verdicts") or v.get("results") or []
        out.extend(v)
    return out, bad

def main():
    pk = json.load(open(HERE / "disease_label_packets.json"))
    sent_by = {}          # (node, title) -> joined sentences
    edges_by_node = {}
    expected = set()
    for p in pk["packets"]:
        edges_by_node[p["node_label"]] = p["n_edges"]
        for pp in p["papers"]:
            key = (p["node_label"], pp["title"])
            expected.add(key)
            sent_by[key] = "\n".join(pp["sentences"]) + "\n" + pp["title"]

    verdicts, bad = load_verdicts()
    seen, rows = set(), []
    for v in verdicts:
        key = (v.get("node_label"), v.get("title"))
        if key not in sent_by:
            # tolerate truncated titles from the reader
            cand = [k for k in sent_by if k[0] == key[0]
                    and (str(key[1])[:40] in k[1] or k[1][:40] in str(key[1]))]
            if len(cand) == 1:
                key = cand[0]
            else:
                rows.append({**v, "quote_check": "UNMATCHED_PAPER"})
                continue
        if key in seen:
            continue
        seen.add(key)
        hay = sent_by[key]
        qs = v.get("quotes") or []
        ok = [q for q in qs if q and q.strip() in hay]
        v = dict(v)
        v["node_label"], v["title"] = key
        v["n_quotes"] = len(qs)
        v["n_quotes_verbatim"] = len(ok)
        v["quote_check"] = ("OK" if qs and len(ok) == len(qs)
                            else "PARTIAL" if ok else ("NO_QUOTES" if not qs else "ALL_PARAPHRASED"))
        if v["quote_check"] in ("ALL_PARAPHRASED", "NO_QUOTES"):
            v["supported"] = False
        else:
            v["supported"] = True
        v["cues"] = sorted(k for k, r in CUES.items() if r.search(hay))
        ct = v.get("contrast_type")
        v["cue_conflict"] = bool(
            (ct == "HC" and "HC" not in v["cues"]) or
            (ct != "HC" and ct != "UNCLEAR" and ct not in v["cues"] and "HC" in v["cues"])
        )
        v["n_edges_node"] = edges_by_node.get(key[0])
        rows.append(v)

    missing = sorted(expected - seen)
    from collections import Counter
    summary = {
        "expected_pairs": len(expected),
        "adjudicated": len(seen),
        "missing": [list(m) for m in missing],
        "unreadable_files": bad,
        "quote_check": dict(Counter(r["quote_check"] for r in rows)),
        "contrast_type": dict(Counter(r.get("contrast_type") for r in rows if r.get("supported"))),
        "label_verdict": dict(Counter(r.get("label_verdict") for r in rows if r.get("supported"))),
        "cue_conflicts": sum(1 for r in rows if r.get("cue_conflict")),
    }
    json.dump({"summary": summary, "rows": rows},
              open(HERE / "disease_label_audit.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
