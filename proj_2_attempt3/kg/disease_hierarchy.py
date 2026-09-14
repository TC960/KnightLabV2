#!/usr/bin/env python3
"""A disease containment layer derived from MONDO, and a test of whether it carries signal.

Context
-------
`disease_containment.py` (2026-09-10) sized a hand-written containment proposal:
11 (child, parent) pairs in three judgement tiers, every per-pair null test
non-significant, and the whole thing correctly logged as "a design decision
needing a PI, not a script" because there was no authority to appeal to. The
2026-09-13 disease-assignment audit then sized the stakes: six cognitive-decline
nodes carry 71 papers and the graph has 0 disease hierarchy links against 713 for
taxa.

`mondo.py` removes the missing authority. So this script does three things the
earlier one could not:

1. **Derives the hierarchy instead of asserting it.** Where both disease labels
   resolve to MONDO, the relation is a lookup in MONDO's is-a DAG. Judgement is
   confined to the residue MONDO does not carry, and that residue is now named.
2. **Grades the hand-written tiers against MONDO** -- an external check on a
   human judgement call, in both directions (confirmations AND refutations).
3. **Tests the signal with the shared-paper confound split out.** This is the
   correction that matters statistically. Two disease nodes can be fed by the
   SAME paper (a comparative study files under both), in which case their
   directional agreement is one observation counted twice, not replication. The
   2026-09-09 independence finding is the same trap one level up: agreement with
   Disbiome/Peryton ran 87.5% on shared papers and 58.1% on disjoint ones, and
   the pooled 73% measured neither. `disease_containment.py`'s null did not
   condition on paper overlap. This one does.

Null model
----------
Permuting observations would break the paper-level clustering the project's
rules forbid breaking. Instead the **ontology position is permuted**: the
MONDO id assignment is shuffled across disease nodes, leaving every node's
microbial profile, paper set and edge structure exactly as it is, and the
related-pair agreement rate is recomputed. That asks precisely the question --
does ontological proximity predict microbial agreement, or would any pairing do
as well -- while holding the data fixed.

Outputs `disease_hierarchy.json`. **Does not modify graph.json**; rebuilding on a
cloud checkout is how this repo lost work on 2026-09-11.
"""
import json
import os
import random
import sys
from collections import defaultdict

from mondo import ALIASES, Mondo, norm_label

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "disease_hierarchy.json")

# ---------------------------------------------------------------------------
# Curated aliases: graph label -> MONDO id, for labels whose MONDO term exists
# under a different surface form. Each needs a reason, and the REFUSALS are
# recorded alongside so a later session does not re-propose them. Same pattern
# as `taxon_typos.py`, for the same reason: edit distance would merge
# `Cognitive impairment` into `specific language impairment`.
# ---------------------------------------------------------------------------
# ALIASES now live in mondo.py -- resolution is the resolver's business, and a
# second copy here could drift from it.

REFUSED_ALIASES = {
    "Mild cognitive impairment":
        "MONDO contains NO term named 'mild cognitive impairment' -- 0 index keys "
        "match the phrase. MCI is a clinical STAGE, not a MONDO disease. The "
        "nearest term, 'cognitive disorder' (MONDO:0002039), is broader and would "
        "silently merge MCI with 13 other things. Left unresolved.",
    "Cognitive impairment":
        "same as above; 'cognitive disorder' is not the same concept and the graph "
        "label is a cohort descriptor.",
    "Neurocognitive impairment":
        "ditto, and it may be a synonym of the graph's own 'Cognitive impairment' "
        "node rather than a MONDO term -- a folding question, not a resolution one.",
    "Neuroinfection":
        "not a disease, a category of causes. No MONDO term, correctly.",
    "REM sleep behavior disorder-Lewy body disease continuum":
        "a prodromal continuum spanning two diseases; MONDO has each end but not "
        "the continuum, and it is contained by neither (already Tier-C rejected).",
    "Hemorrhagic transformation":
        "a complication of ischemic stroke, not a disease entity in MONDO.",
    "Poststroke aphasia":
        "MONDO has 'aphasia' (MONDO:0000598) but post-stroke aphasia is a "
        "stroke sequela; resolving it to plain aphasia would misfile the cohort.",
    "Chronic traumatic complete spinal cord injury":
        "severity/chronicity-specified SCI; MONDO carries no graded SCI terms.",
    "Traumatic thoracic spinal cord injury":
        "level-specified SCI; same reason.",
    "Hypertensive intracerebral hemorrhage":
        "cause-specified ICH; MONDO has ICH but not the hypertensive subtype.",
    "Minimal hepatic encephalopathy":
        "subclinical grade of HE; MONDO has HE but not the graded form.",
    "Hepatitis B virus-associated liver cirrhosis":
        "MONDO has both hepatitis B and liver cirrhosis but not this conjunction.",
}

# The hand-written tier claims from `disease_containment.py`, restated here so
# MONDO can grade them. (child, parent, tier).
TIER_CLAIMS = [
    ("Intracerebral hemorrhage", "Stroke", "A"),
    ("Hypertensive intracerebral hemorrhage", "Intracerebral hemorrhage", "A"),
    ("Poststroke aphasia", "Stroke", "A"),
    ("Hemorrhagic transformation", "Stroke", "A"),
    ("Chronic traumatic complete spinal cord injury", "Spinal cord injury", "A"),
    ("Traumatic thoracic spinal cord injury", "Spinal cord injury", "A"),
    ("Minimal hepatic encephalopathy", "Hepatic encephalopathy", "A"),
    ("Alzheimer's disease", "Dementia", "B"),
    ("Sporadic Creutzfeldt-Jakob disease", "Dementia", "B"),
    ("Mild cognitive impairment", "Cognitive impairment", "B"),
    ("Neurocognitive impairment", "Cognitive impairment", "B"),
    ("Subjective cognitive decline", "Cognitive impairment", "B"),
]
# Tier C: claims the earlier session explicitly REJECTED. MONDO grading these is
# the more informative half -- a rejection MONDO contradicts would be a real find.
TIER_C_CLAIMS = [
    ("Multiple system atrophy", "Parkinson's disease"),
    ("Essential tremor", "Parkinson's disease"),
    ("Mild cognitive impairment", "Alzheimer's disease"),
    ("Hepatitis B virus-associated liver cirrhosis", "Parkinson's disease"),
]


def resolve_all(m, labels):
    out = {}
    for lb in labels:
        if lb in ALIASES:
            mid, why = ALIASES[lb]
            out[lb] = {"mondo": mid, "how": "curated_alias", "why": why,
                       "mondo_name": m.name.get(mid)}
            continue
        mid, how = m.resolve(lb)
        row = {"mondo": mid, "how": how, "mondo_name": m.name.get(mid)}
        if not mid:
            # match refusal keys tolerantly (the graph uses an en dash in one label)
            for k, v in REFUSED_ALIASES.items():
                if norm_label(k) == norm_label(lb):
                    row["refusal"] = v
                    break
        out[lb] = row
    return out


def profiles(g):
    """disease -> taxon_key -> ('enriched'|'depleted'|'contested'), plus paper sets."""
    prof = defaultdict(dict)
    dpapers = defaultdict(set)
    epapers = defaultdict(dict)
    for e in g["edges"]:
        d = e["disease"]
        up, dn = e["n_up"], e["n_down"]
        prof[d][e["taxon_key"]] = "contested" if (up and dn) else ("enriched" if up else "depleted")
        dpapers[d].update(e["papers"])
        epapers[d][e["taxon_key"]] = set(e["papers"])
    return prof, dpapers, epapers


def pair_agreement(prof, epapers, a, b, require_disjoint=False):
    """(n_decisive, n_agree) on taxa where BOTH diseases are non-contested.

    require_disjoint drops any taxon whose evidence in the two diseases shares a
    source paper -- the same observation counted twice is not replication.
    """
    pa, pb = prof.get(a, {}), prof.get(b, {})
    n = k = 0
    for t in set(pa) & set(pb):
        if pa[t] == "contested" or pb[t] == "contested":
            continue
        if require_disjoint and (epapers[a][t] & epapers[b][t]):
            continue
        n += 1
        k += (pa[t] == pb[t])
    return n, k


def main():
    m = Mondo()
    g = json.load(open(GRAPH))
    labels = sorted(n["label"] for n in g["nodes"] if str(n.get("id", "")).startswith("d:"))
    prof, dpapers, epapers = profiles(g)
    res = resolve_all(m, labels)
    resolved = {lb: r["mondo"] for lb, r in res.items() if r["mondo"]}
    print(f"MONDO {m.version}; {len(resolved)}/{len(labels)} disease labels resolved "
          f"({sum(1 for r in res.values() if r['how'] == 'curated_alias')} by curated alias, "
          f"{len(labels) - len(resolved)} refused)")

    # ---------------------------------------------------------------- 1. grade
    print("\n=== MONDO's verdict on the hand-written tier claims ===")
    graded = []
    for child, parent, tier in TIER_CLAIMS:
        ci, pi = resolved.get(child), resolved.get(parent)
        if not (ci and pi):
            verdict = "unresolvable"
            rel = dist = None
        else:
            rel = m.relation(ci, pi)
            dist = m.distance(ci, pi)
            verdict = "CONFIRMED" if rel == "descendant" else f"not-is-a ({rel})"
        graded.append({"child": child, "parent": parent, "tier": tier,
                       "mondo_relation": rel, "mondo_distance": dist,
                       "verdict": verdict})
        print(f"  [{tier}] {child[:42]:44s} -> {parent[:26]:28s} {verdict}")

    print("\n=== MONDO's verdict on the Tier-C REJECTIONS (should NOT be is-a) ===")
    graded_c = []
    for child, parent in TIER_C_CLAIMS:
        ci, pi = resolved.get(child), resolved.get(parent)
        if not (ci and pi):
            verdict, rel, dist = "unresolvable", None, None
        else:
            rel = m.relation(ci, pi)
            dist = m.distance(ci, pi)
            verdict = ("REJECTION OVERTURNED - MONDO says is-a" if rel == "descendant"
                       else f"rejection upheld ({rel})")
        graded_c.append({"child": child, "parent": parent, "mondo_relation": rel,
                         "mondo_distance": dist, "verdict": verdict})
        print(f"  {child[:42]:44s} -> {parent[:26]:28s} {verdict}")

    # ------------------------------------------------- 2. the derived layer
    # Every ordered pair of resolved labels where MONDO asserts is-a.
    links = []
    for a in resolved:
        for b in resolved:
            if a == b:
                continue
            if m.relation(resolved[a], resolved[b]) == "descendant":
                links.append({"child": a, "parent": b,
                              "child_mondo": resolved[a], "parent_mondo": resolved[b],
                              "mondo_distance": m.distance(resolved[a], resolved[b]),
                              "source": "mondo",
                              "child_papers": len(dpapers[a]),
                              "parent_papers": len(dpapers[b])})
    print(f"\n=== derived layer: {len(links)} MONDO is-a links between graph disease nodes ===")
    for l in sorted(links, key=lambda x: -x["child_papers"]):
        print(f"  {l['child'][:40]:42s} -> {l['parent'][:30]:32s} "
              f"(dist {l['mondo_distance']}, {l['child_papers']}p under {l['parent_papers']}p)")

    # ------------------------------------------------- 3. the retrieval payoff
    # What does a query for a parent disease currently MISS?
    payoff = []
    kids = defaultdict(list)
    for l in links:
        kids[l["parent"]].append(l["child"])
    for parent, cs in sorted(kids.items()):
        own_p = dpapers[parent]
        own_t = set(prof[parent])
        gain_p = set().union(*(dpapers[c] for c in cs)) - own_p
        gain_t = set().union(*(set(prof[c]) for c in cs)) - own_t
        payoff.append({"parent": parent, "children": cs,
                       "own_papers": len(own_p), "extra_papers": len(gain_p),
                       "own_taxa": len(own_t), "extra_taxa": len(gain_t)})
        print(f"\n  query '{parent}' currently returns {len(own_p)} papers / "
              f"{len(own_t)} taxa;\n    + {len(gain_p)} papers and {len(gain_t)} taxa "
              f"sit on {cs}")

    # --------------------------------- 4. does proximity predict agreement?
    ds = [d for d in resolved if len(prof[d]) >= 5]
    buckets = defaultdict(lambda: [0, 0])          # relation -> [n, k]
    buckets_dj = defaultdict(lambda: [0, 0])       # paper-disjoint only
    shared_paper_pairs = 0
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            rel = m.relation(resolved[a], resolved[b])
            rel = {"ancestor": "is-a", "descendant": "is-a"}.get(rel, rel)
            n, k = pair_agreement(prof, epapers, a, b)
            buckets[rel][0] += n
            buckets[rel][1] += k
            nd, kd = pair_agreement(prof, epapers, a, b, require_disjoint=True)
            buckets_dj[rel][0] += nd
            buckets_dj[rel][1] += kd
            if dpapers[a] & dpapers[b]:
                shared_paper_pairs += 1

    print(f"\n=== directional agreement by MONDO relation "
          f"({len(ds)} resolved diseases with >=5 taxa, "
          f"{shared_paper_pairs} pairs share >=1 paper) ===")
    print(f"{'relation':12s} {'all observations':>22s} {'paper-disjoint only':>24s}")
    for rel in ("is-a", "sibling", "cousin", "unrelated"):
        n, k = buckets[rel]
        nd, kd = buckets_dj[rel]
        a1 = f"{k}/{n} = {k/n:.3f}" if n else "n=0"
        a2 = f"{kd}/{nd} = {kd/nd:.3f}" if nd else "n=0"
        print(f"{rel:12s} {a1:>22s} {a2:>24s}")

    # ---- permutation null: shuffle the ONTOLOGY POSITION, hold data fixed ----
    def related_rate(assign, disjoint):
        n = k = 0
        for i, a in enumerate(ds):
            for b in ds[i + 1:]:
                if m.relation(assign[a], assign[b]) in ("descendant", "ancestor", "sibling"):
                    nn, kk = pair_agreement(prof, epapers, a, b, require_disjoint=disjoint)
                    n += nn
                    k += kk
        return n, k

    perm = {}
    for disjoint in (False, True):
        n_true, k_true = related_rate(resolved, disjoint)
        if n_true == 0:
            perm["disjoint" if disjoint else "all"] = {"n": 0, "note": "no decisive observations"}
            continue
        rate_true = k_true / n_true
        rng = random.Random(23)
        ids = [resolved[d] for d in ds]
        hits = 0
        rates = []
        N = 10000
        for _ in range(N):
            sh = ids[:]
            rng.shuffle(sh)
            assign = dict(zip(ds, sh))
            n, k = related_rate(assign, disjoint)
            if n == 0:
                continue
            r = k / n
            rates.append(r)
            hits += (r >= rate_true)
        mean = sum(rates) / len(rates)
        sd = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5
        p = (hits + 1) / (len(rates) + 1)
        perm["disjoint" if disjoint else "all"] = {
            "n_decisive": n_true, "n_agree": k_true, "rate": round(rate_true, 4),
            "null_mean": round(mean, 4), "null_sd": round(sd, 4),
            "null_valid": len(rates), "p": round(p, 4),
            # MDE: the smallest true rate that would clear the 95th percentile
            "null_p95": round(sorted(rates)[int(0.95 * len(rates))], 4),
        }
        tag = "paper-disjoint" if disjoint else "all observations"
        print(f"\n  [{tag}] related (is-a or sibling) {k_true}/{n_true} = {rate_true:.3f}"
              f"  vs ontology-shuffled null {mean:.3f} +/- {sd:.3f}"
              f"  p={p:.4f}  (null 95th pct {perm['disjoint' if disjoint else 'all']['null_p95']:.3f})")

    json.dump({"mondo_version": m.version, "resolution": res,
               "refused": REFUSED_ALIASES, "aliases": ALIASES,
               "tier_grading": graded, "tier_c_grading": graded_c,
               "links": links, "retrieval_payoff": payoff,
               "agreement_by_relation": {k: {"n": v[0], "k": v[1]} for k, v in buckets.items()},
               "agreement_by_relation_disjoint": {k: {"n": v[0], "k": v[1]}
                                                  for k, v in buckets_dj.items()},
               "permutation": perm,
               "n_pairs_sharing_papers": shared_paper_pairs},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
