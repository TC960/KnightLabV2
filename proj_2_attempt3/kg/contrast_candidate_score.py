#!/usr/bin/env python3
"""Score the CANDIDATE adjudication: turn the 114-observation upper bound into an
estimate, with a paper-level cluster bootstrap CI.

Every quote is machine-checked byte-for-byte, then again after collapsing
whitespace (the sentences carry PDF-extraction spacing a reader tidies silently).
Verdicts whose quotes are genuine paraphrases are excluded and reported.

The estimator is the fraction of adjudicated CANDIDATE observations that really do
report the taxon only against a within-disease subgroup, a treatment arm or another
disease. The CI resamples PAPERS with replacement, not observations: 114 candidates
come from 47 papers and one paper contributes 10, so an observation-level bootstrap
would understate the interval.

WHAT THIS DOES AND DOES NOT BOUND. It estimates contamination INSIDE the flagged
set. It says nothing about the 2,506 CLEAN and UNRESOLVED observations, which the
regex could also have got wrong in the other direction. So:

    total contamination  >=  (this rate) x 114 ... and is NOT bounded above by 114.

Writes contrast_candidate_score.json.
"""
import json, random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260919
N_BOOT = 20000
CONTAMINATED = {"SUBGROUP_ONLY"}

def load():
    rows = []
    for f in sorted(HERE.glob("_ccbatch/verdicts*.json")):
        v = json.load(open(f))
        if isinstance(v, dict):
            v = v.get("observations") or v.get("verdicts") or []
        rows.extend(v)
    return rows

def main():
    pk = json.load(open(HERE / "contrast_candidate_packets.json"))
    by_id = {p["id"]: p for p in pk["packets"]}
    norm = lambda s: " ".join(s.split())

    ov = json.load(open(HERE / "contrast_candidate_override.json"))["overrides"]
    probe_now = {(r["paper"], r["taxon"]): r["verdict"]
                 for r in json.load(open(HERE / "contrast_edge_probe.json"))["rows"]}

    seen, scored, unsupported, dropped = set(), [], [], []
    for v in load():
        i = v.get("id")
        if i not in by_id or i in seen:
            continue
        seen.add(i)
        p = by_id[i]
        hay = "\n".join(p["sentences_naming_taxon"])
        nhay = norm(hay)
        qs = [q for q in (v.get("quotes") or []) if q]
        exact = sum(1 for q in qs if q.strip() in hay)
        resp = sum(1 for q in qs if norm(q) in nhay)
        rec = {**v, "title": p["title"], "taxon": p["taxon"],
               "n_quotes": len(qs), "n_exact": exact, "n_respaced": resp}
        if qs and resp == 0:
            rec["quote_check"] = "ALL_PARAPHRASED"
            unsupported.append(rec)
            continue
        rec["quote_check"] = ("OK" if qs and exact == len(qs) else
                              "OK_RESPACED" if qs and resp == len(qs) else
                              "PARTIAL" if resp else "NO_QUOTES")
        # the widened regex may now see a control contrast this observation's
        # sentences do state; such an observation is no longer a candidate at all
        # and leaves the denominator rather than counting as a correct refusal
        if probe_now.get((p["title"], p["taxon"])) != "CANDIDATE":
            rec["left_pool"] = probe_now.get((p["title"], p["taxon"]))
            dropped.append(rec)
            continue
        if i in ov:
            rec["adjudicator_verdict"] = rec.get("verdict")
            rec["verdict"], rec["override_reason"] = ov[i]
            rec["overridden"] = rec["verdict"] != rec["adjudicator_verdict"]
        scored.append(rec)

    missing = sorted(set(by_id) - seen)
    from collections import Counter
    counts = Counter(r.get("verdict") for r in scored)

    by_paper = defaultdict(list)
    for r in scored:
        by_paper[r["title"]].append(r.get("verdict") in CONTAMINATED)
    papers = list(by_paper.values())
    n_obs = sum(len(v) for v in papers)
    n_bad = sum(sum(v) for v in papers)
    rate = n_bad / n_obs if n_obs else float("nan")

    rng = random.Random(SEED)
    boot = []
    for _ in range(N_BOOT):
        pick = [papers[rng.randrange(len(papers))] for _ in papers]
        o = sum(len(v) for v in pick)
        b = sum(sum(v) for v in pick)
        if o:
            boot.append(b / o)
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]

    pool = json.load(open(HERE / "contrast_edge_probe.json"))
    n_cand = sum(1 for r in pool["rows"] if r["verdict"] == "CANDIDATE")
    total = pool["n_observations"]

    out = {
        "packets": len(by_id), "adjudicated": len(seen),
        "missing": missing,
        "excluded_all_paraphrased": len(unsupported),
        "left_pool_after_regex_widening": len(dropped),
        "overridden_by_orchestrator": sum(1 for r in scored if r.get("overridden")),
        "adjudicator_verdicts_before_override":
            dict(Counter(r.get("adjudicator_verdict") or r.get("verdict") for r in scored)),
        "quote_check": dict(Counter(r["quote_check"] for r in scored)),
        "verdicts": dict(counts),
        "n_papers_scored": len(papers), "n_observations_scored": n_obs,
        "n_contaminated": n_bad,
        "rate_within_flagged": round(rate, 4),
        "ci95_cluster_bootstrap": [round(lo, 4), round(hi, 4)],
        "n_candidates_in_pool": n_cand,
        "estimated_contaminated_observations": round(rate * n_cand, 1),
        "estimated_range": [round(lo * n_cand, 1), round(hi * n_cand, 1)],
        "graph_observations": total,
        "estimated_share_of_graph_pct": round(100 * rate * n_cand / total, 3),
        "estimated_share_range_pct": [round(100 * lo * n_cand / total, 3),
                                      round(100 * hi * n_cand / total, 3)],
        "n_boot": len(boot), "seed": SEED,
        "rows": scored, "unsupported": unsupported,
    }
    out["dropped_rows"] = dropped
    json.dump(out, open(HERE / "contrast_candidate_score.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("rows", "unsupported", "dropped_rows")}, indent=1))

if __name__ == "__main__":
    main()
