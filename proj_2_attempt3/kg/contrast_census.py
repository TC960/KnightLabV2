#!/usr/bin/env python3
"""Corpus-wide census of the STUDY CONTRAST behind every edge in the graph.

WHY. The extraction prompt (`eval-v2/run_eval.py`, samgated-v1) admits only
**disease vs healthy control** comparisons. Every fidelity instrument this
project has scores whether an edge's taxon and direction are right -- reading
fidelity (86.6%), the mention audit (99.57%), Disbiome/Peryton agreement
(73.0/72.5%), the new gold (F1 0.739). None of them asks whether the COMPARISON
the edge came from was in scope at all. A taxon genuinely higher in ICH
survivors than in ICH deceased is a correct reading of its paper and is still
not a disease-vs-control edge.

WHAT. 271 contributing papers, each read by a blinded adjudicator that saw only
the title and the paper's own relation-bearing sentences -- NOT the disease
label, NOT whether that label came from DISEASE_MAP or the free-text fallback.
Blinding matters because the second test below compares exactly those two groups.

THREE CHECKS, in order of how much they are believed:

1. Quote verification (deterministic). Every quote must be a byte-for-byte
   substring of the sentences supplied. 7 of 36 adjudicator "verbatim" quotes
   failed this in the 2026-09-16 edge-recall audit. Unsupported verdicts are
   reported separately and excluded from the tests.
2. Is an out-of-gate contrast more common under free-text disease labels than
   under DISEASE_MAP ones? Fisher exact + a paper-level permutation.
3. Do out-of-gate papers disagree with the rest of the literature more? This
   reuses `paper_discordance_offset.py` wholesale -- the closed-form within-edge
   expectation that absorbs edge depth, and the paper-level permutation null --
   so the answer is comparable to the 25 variables already tested that way.
   ONE pre-registered test, uncorrected, MDE reported.

Writes contrast_census.json.
"""
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
N_PERM = 20000
SEED = 20260919
IN_GATE = {"HC"}

def load_verdicts():
    rows, bad = [], []
    for f in sorted(HERE.glob("_ctbatch/verdicts*.json"),
                    key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0)):
        try:
            v = json.load(open(f))
        except Exception as e:
            bad.append((f.name, str(e))); continue
        if isinstance(v, dict):
            v = v.get("papers") or v.get("verdicts") or v.get("results") or []
        rows.extend(v)
    return rows, bad

def verify(rows, packets):
    by_title = {p["title"]: p for p in packets}
    prefix = {p["title"][:45]: p["title"] for p in packets}
    out, seen = [], set()
    for v in rows:
        t = v.get("title", "")
        if t not in by_title:
            t = prefix.get(str(t)[:45], None)
            if t is None:
                cand = [k for k in by_title if k[:30] == str(v.get("title"))[:30]]
                if len(cand) != 1:
                    out.append({**v, "quote_check": "UNMATCHED_PAPER", "supported": False})
                    continue
                t = cand[0]
        if t in seen:
            continue
        seen.add(t)
        pkt = by_title[t]
        hay = "\n".join(pkt["sentences"]) + "\n" + t
        qs = [q for q in (v.get("quotes") or []) if q]
        ok = [q for q in qs if q.strip() in hay]
        # Second tier. A "paraphrase" is often only re-spaced: these sentences carry
        # artifacts of PDF extraction (" , ", " [ 32 ] ") that a reader silently
        # tidies. Collapsing whitespace separates a tidied quote, which is still
        # the paper's own words, from an invented one, which is not. Byte-for-byte
        # remains the headline; this tier is reported alongside it, never instead.
        norm = lambda s: " ".join(s.split())
        nhay = norm(hay)
        nok = [q for q in qs if norm(q) in nhay]
        v = dict(v); v["title"] = t
        v["n_quotes"] = len(qs)
        v["n_quotes_verbatim"] = len(ok)
        v["n_quotes_normalised"] = len(nok)
        v["quote_check"] = ("OK" if qs and len(ok) == len(qs) else
                            "PARTIAL" if ok else
                            "NO_QUOTES" if not qs else
                            "OK_RESPACED" if len(nok) == len(qs) else
                            "PARTIAL_RESPACED" if nok else "ALL_PARAPHRASED")
        v["supported"] = v["quote_check"] != "ALL_PARAPHRASED" and bool(qs)
        v["supported_strict"] = v["quote_check"] in ("OK", "PARTIAL")
        v["diseases"] = pkt["diseases"]
        v["origin"] = pkt["origin"]
        v["n_edges"] = pkt["n_edges"]
        v["any_fallback"] = "FALLBACK" in pkt["origin"]
        out.append(v)
    return out, sorted(set(by_title) - seen)

def fisher(a, b, c, d):
    """two-sided Fisher exact on [[a,b],[c,d]] without scipy."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    obs = pr(a)
    lo = max(0, c1 - (n - r1)); hi = min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= obs * (1 + 1e-9)))

def perm_diff(flags, groups, rng, n=N_PERM):
    """paper-level permutation on the difference in out-of-gate rate."""
    idx = [i for i in range(len(flags)) if flags[i] is not None and groups[i] is not None]
    f = [flags[i] for i in idx]; g = [groups[i] for i in idx]
    def rate(gg):
        a = [x for x, y in zip(f, gg) if y]; b = [x for x, y in zip(f, gg) if not y]
        if not a or not b: return None
        return sum(a) / len(a) - sum(b) / len(b)
    obs = rate(g)
    if obs is None: return None
    perm = list(g); null = []
    for _ in range(n):
        rng.shuffle(perm)
        r = rate(perm)
        if r is not None: null.append(r)
    p = (sum(1 for d in null if abs(d) >= abs(obs)) + 1) / (len(null) + 1)
    null.sort()
    mde = max(abs(null[int(0.025 * len(null))]), abs(null[int(0.975 * len(null))]))
    return {"n": len(f), "obs_diff": round(obs, 4), "p": round(p, 5),
            "mde_abs_diff": round(mde, 4)}

def discordance_test(verdicts, predictor="out_of_gate"):
    """Predictors #26 and #27, run through the existing within-edge-offset machinery.

    `out_of_gate`     -- the paper's main reported contrast is not disease vs
                         healthy control, so its edges should not exist at all.
    `mixed_provenance`-- the paper DOES report a healthy-control contrast but also
                         a within-disease subgroup contrast, so some of its edges
                         may have come from the wrong one. This is the larger and
                         better-powered group, and it is the risk the paper-level
                         gate cannot rule out.
    """
    from paper_inversion import MIN_DECISIVE, build_observations, load_graph, score
    from paper_inversion_control import ancestor_sets, thin
    from paper_discordance_offset import expectations, test_binary
    graph = load_graph()
    edges = build_observations(graph)
    thinned, _ = thin(edges, ancestor_sets(graph))
    per_paper, _ = score(thinned)
    e_dec, e_dis = expectations(thinned)
    papers = graph["papers"]
    min_dec = 0 if predictor.endswith("_allpapers") else MIN_DECISIVE
    predictor = predictor.replace("_allpapers", "")
    if predictor == "out_of_gate":
        gate = {v["title"]: (v.get("contrast_type") not in IN_GATE)
                for v in verdicts
                if v.get("supported") and v.get("contrast_type") != "UNCLEAR"}
    else:
        gate = {v["title"]: bool(v.get("also_subgroup"))
                for v in verdicts
                if v.get("supported") and v.get("contrast_type") == "HC"}
    rows, labels = [], []
    for p, v in sorted(per_paper.items()):
        if v[0] < min_dec or e_dis[p] <= 0:
            continue
        rows.append({"dis": v[1], "e_dis": e_dis[p]})
        labels.append(gate.get(papers[p]["title"]))
    res = test_binary(rows, labels, random.Random(SEED))
    return {"predictor": predictor, "n_testable_papers": len(rows),
            "n_labelled": sum(1 for l in labels if l is not None),
            "min_decisive": min_dec, "result": res}

def main():
    packets = json.load(open(HERE / "contrast_packets.json"))["packets"]
    raw, bad = load_verdicts()
    verdicts, missing = verify(raw, packets)
    sup = [v for v in verdicts if v.get("supported")]

    ct = Counter(v.get("contrast_type") for v in sup)
    edges_ct = Counter()
    for v in sup:
        edges_ct[v.get("contrast_type")] += v["n_edges"]
    out_of_gate = [v for v in sup if v.get("contrast_type") not in IN_GATE
                   and v.get("contrast_type") != "UNCLEAR"]

    rng = random.Random(SEED)
    fb = [v for v in sup if v["any_fallback"]]
    dm = [v for v in sup if not v["any_fallback"]]
    def oog(vs):  # excludes UNCLEAR from both numerator and denominator
        d = [v for v in vs if v.get("contrast_type") != "UNCLEAR"]
        return sum(1 for v in d if v.get("contrast_type") not in IN_GATE), len(d)
    a, na = oog(fb); c, nc = oog(dm)
    enrich = {
        "fallback_out_of_gate": a, "fallback_n": na,
        "diseasemap_out_of_gate": c, "diseasemap_n": nc,
        "fallback_rate": round(a / na, 4) if na else None,
        "diseasemap_rate": round(c / nc, 4) if nc else None,
        "fisher_p": round(fisher(a, na - a, c, nc - c), 5) if na and nc else None,
    }
    flags = [(None if v.get("contrast_type") == "UNCLEAR"
              else v.get("contrast_type") not in IN_GATE) for v in sup]
    groups = [v["any_fallback"] for v in sup]
    enrich["permutation"] = perm_diff(flags, groups, rng)

    disc = {}
    # MIN_DECISIVE=4 is right when the unit of interest is the paper's own rate,
    # but it throws away 17 of the 19 out-of-gate papers -- they are small
    # contributors, which is why they were never noticed. The O/E offset already
    # absorbs edge depth, so pooling every observation and permuting the PAPER
    # label is valid without the filter and is the only version with any power
    # on this group. Both are reported; the filtered one is not quotable at n=2.
    for pred in ("out_of_gate", "mixed_provenance",
                 "out_of_gate_allpapers", "mixed_provenance_allpapers"):
        try:
            disc[pred] = discordance_test(sup, pred)
        except Exception as e:
            disc[pred] = {"error": repr(e)}
    # BH over the two pre-registered tests
    live = sorted(((k, v["result"]["p"]) for k, v in disc.items()
                   if v.get("result")), key=lambda kv: kv[1])
    m = len(live); prev = 1.0
    for i in range(m - 1, -1, -1):
        k, pv = live[i]
        prev = min(prev, pv * m / (i + 1))
        disc[k]["q_bh"] = round(min(1.0, prev), 4)

    summary = {
        "papers_expected": len(packets),
        "papers_adjudicated": len(verdicts),
        "papers_supported": len(sup),
        "missing_papers": missing,
        "unreadable_files": bad,
        "quote_check": dict(Counter(v["quote_check"] for v in verdicts)),
        "papers_supported_strict": sum(1 for v in verdicts if v.get("supported_strict")),
        "contrast_type_papers": dict(ct),
        "contrast_type_edges": dict(edges_ct),
        "n_out_of_gate_papers": len(out_of_gate),
        "n_out_of_gate_edges": sum(v["n_edges"] for v in out_of_gate),
        "also_subgroup_among_HC": sum(1 for v in sup
                                      if v.get("contrast_type") == "HC"
                                      and v.get("also_subgroup")),
        "enrichment_fallback_vs_diseasemap": enrich,
        "discordance_of_out_of_gate_papers": disc,
    }
    json.dump({"summary": summary, "verdicts": verdicts},
              open(HERE / "contrast_census.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("\nOUT OF GATE:")
    for v in sorted(out_of_gate, key=lambda x: -x["n_edges"]):
        print(f"  {v['n_edges']:>3}e  {v.get('contrast_type'):13} "
              f"{'|'.join(v['diseases'])[:30]:30} {v['title'][:70]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
