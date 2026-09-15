"""Where does each disease node's LABEL come from?

The taxon half of this graph has two curation instruments -- `taxon_typos.py`
(the papers' own misspellings) and `multi_taxon.py` (punctuation fragmenting one
concept over several nodes). The disease half has none. This script measures the
gap rather than closing it, because closing it is a modelling call for a human
(see CLAUDE.md, "What remains is a decision, not an analysis").

`build_kg.norm_disease` tries 17 regexes in DISEASE_MAP and, on a miss, falls
through to:

    label = s[:1].upper() + s[1:]

i.e. the extractor's free-text `predicted_disease` string, title-cased, becomes a
disease node verbatim. That fallback is deliberate and defensible -- the comment
in build_kg.py says unmapped labels keep their cleaned label "rather than being
silently dropped" -- but it has never been measured, and it means a fifth of the
graph's edges hang off strings no vocabulary ever approved.

REPORT ONLY. This script folds nothing and writes nothing into the graph. The
fragmentation candidates it prints are candidates, not verdicts: "Minimal hepatic
encephalopathy" really is a distinct clinical entity from "Hepatic
encephalopathy", and deciding which of these pairs are subtypes, which are
synonyms and which are unrelated is exactly the human call this file refuses to
pre-empt.

Run:  python3 disease_node_provenance.py
"""

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_disease_map():
    """Read DISEASE_MAP *live* out of build_kg.py.

    Deliberately not a copy. A control holding its own copy of the table it
    checks cannot detect drift in that table -- which is how "174 contested"
    survived three sessions after the number had become 217, and how the
    2026-09-14 MONDO control was nearly written blind. If build_kg.py's table
    changes, this script changes with it or fails loudly.
    """
    src = open(os.path.join(HERE, "build_kg.py")).read()
    m = re.search(r"^DISEASE_MAP = \[.*?^\]", src, re.S | re.M)
    if not m:
        raise SystemExit("FATAL: could not find DISEASE_MAP in build_kg.py -- "
                         "the builder was restructured; fix this loader before "
                         "trusting any number below.")
    ns = {}
    exec(m.group(0), ns)
    return ns["DISEASE_MAP"]


def mapped_label(s, disease_map):
    """Replay norm_disease's regex arm. Returns the mapped label or None."""
    low = (s or "").strip().lower()
    for pat, label, _mondo in disease_map:
        if re.search(pat, low):
            return label
    return None


def self_test(disease_map):
    """Positive + negative controls on the replay.

    The point is that this script's claim -- "node X came from the fallback" --
    is only as good as its reimplementation of norm_disease's regex arm. So
    check the arm against labels whose routing is known from build_kg.py's own
    comments before reporting anything.
    """
    cases = [
        # (input, expected mapped label or None)
        ("parkinson's disease", "Parkinson's disease"),
        ("Alzheimer's disease (AD)", "Alzheimer's disease"),
        ("mild cognitive impairment", "Mild cognitive impairment"),
        ("MCI", "Mild cognitive impairment"),
        ("autism spectrum disorder", "Autism spectrum disorder"),
        ("NMDAR encephalitis", "Anti-NMDAR encephalitis"),
        # the fallback arm: these must NOT match any regex
        ("neurocognitive impairment", None),
        ("cognitive impairment", None),      # distinct from *mild* cognitive impairment
        ("spinal cord injury", None),
        ("Rett syndrome", None),
        ("hemorrhagic transformation", None),
    ]
    bad = []
    for s, want in cases:
        got = mapped_label(s, disease_map)
        if got != want:
            bad.append((s, want, got))
    if bad:
        print("SELF-TEST FAILED -- not reporting numbers derived from a broken replay:")
        for s, want, got in bad:
            print(f"   {s!r}: expected {want!r}, got {got!r}")
        raise SystemExit(1)
    print(f"self-test: {len(cases)}/{len(cases)} OK "
          f"({len(disease_map)} DISEASE_MAP entries read live from build_kg.py)")


HEAD_STOPWORDS = {
    "acute", "chronic", "sporadic", "familial", "idiopathic", "minimal",
    "mild", "severe", "early", "late", "complete", "incomplete", "traumatic",
    "thoracic", "hypertensive", "subjective", "primary", "secondary",
}


def head_key(label):
    """Crude head-noun key for spotting labels that may name one family.

    Strictly a *candidate generator* for human review. It is not a folding rule
    and must never become one: the taxon side already learned this the hard way
    -- edit distance would have merged `Oscillospirales` into `Oscillospira`,
    which is why `taxon_typos.py` ships a curated table with 13 recorded
    refusals instead of a similarity threshold.
    """
    words = [w for w in re.split(r"[^a-z]+", label.lower()) if w]
    words = [w for w in words if w not in HEAD_STOPWORDS]
    return " ".join(words[-2:]) if len(words) >= 2 else (words[-1] if words else "")


def main():
    disease_map = load_disease_map()
    self_test(disease_map)

    g = json.load(open(os.path.join(HERE, "graph.json")))
    edges = collections.Counter()
    papers = collections.defaultdict(set)
    for e in g["edges"]:
        edges[e["disease"]] += 1
        for p in e["papers"]:
            papers[e["disease"]].add(p)
    mondo = {n["label"]: n.get("mondo")
             for n in g["nodes"] if n.get("type") == "disease"}

    canon = {lab for _p, lab, _m in disease_map}
    nodes = sorted(edges)
    frm = {d: ("DISEASE_MAP" if d in canon else "fallback") for d in nodes}

    rows = []
    for d in nodes:
        rows.append({
            "disease": d,
            "origin": frm[d],
            "n_papers": len(papers[d]),
            "n_edges": edges[d],
            "mondo": mondo.get(d),
        })

    fb = [r for r in rows if r["origin"] == "fallback"]
    mp = [r for r in rows if r["origin"] == "DISEASE_MAP"]

    def agg(rs):
        return (len(rs), sum(r["n_edges"] for r in rs),
                len(set().union(*[papers[r["disease"]] for r in rs])) if rs else 0)

    n_mp, e_mp, p_mp = agg(mp)
    n_fb, e_fb, p_fb = agg(fb)
    tot_e = e_mp + e_fb

    print()
    print(f"{'origin':14s} {'nodes':>5s} {'edges':>6s} {'edge share':>11s} {'papers':>7s}")
    print(f"{'DISEASE_MAP':14s} {n_mp:5d} {e_mp:6d} {e_mp/tot_e:10.1%} {p_mp:7d}")
    print(f"{'fallback':14s} {n_fb:5d} {e_fb:6d} {e_fb/tot_e:10.1%} {p_fb:7d}")
    print()
    print("FALLBACK nodes -- label is the extractor's free text, title-cased, "
          "approved by nothing:")
    print(f"   {'label':50s} {'papers':>6s} {'edges':>5s}  mondo")
    for r in sorted(fb, key=lambda r: -r["n_edges"]):
        print(f"   {r['disease'][:50]:50s} {r['n_papers']:6d} {r['n_edges']:5d}  {r['mondo']}")

    no_id = [r for r in fb if not r["mondo"]]
    print(f"\n   of which carry NO MONDO id: {len(no_id)} nodes, "
          f"{sum(r['n_edges'] for r in no_id)} edges -- neither a regex nor an "
          f"ontology has ever seen these labels.")

    # candidate fragmentation: >1 node sharing a head key
    fam = collections.defaultdict(list)
    for r in rows:
        fam[head_key(r["disease"])].append(r)
    frag = {k: v for k, v in fam.items() if len(v) > 1}
    print(f"\nCANDIDATE label families (>1 node sharing a head noun) -- "
          f"{len(frag)} families, FOR HUMAN REVIEW, nothing folded:")
    for k, v in sorted(frag.items(), key=lambda kv: -sum(r["n_edges"] for r in kv[1])):
        tot = sum(r["n_edges"] for r in v)
        print(f"   [{k}]  {tot} edges across {len(v)} nodes")
        for r in sorted(v, key=lambda r: -r["n_edges"]):
            print(f"        {r['disease'][:48]:48s} {r['n_papers']:2d}p {r['n_edges']:3d}e "
                  f"{r['origin']:11s} {r['mondo']}")

    out = {
        "summary": {
            "nodes_total": len(rows),
            "nodes_disease_map": n_mp, "edges_disease_map": e_mp,
            "nodes_fallback": n_fb, "edges_fallback": e_fb,
            "edges_total": tot_e,
            "fallback_edge_share": round(e_fb / tot_e, 4),
            "fallback_nodes_without_mondo": len(no_id),
            "fallback_edges_without_mondo": sum(r["n_edges"] for r in no_id),
        },
        "nodes": rows,
        "candidate_families": {
            k: [r["disease"] for r in v] for k, v in frag.items()
        },
    }
    path = os.path.join(HERE, "disease_node_provenance.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {os.path.basename(path)}")


if __name__ == "__main__":
    main()
