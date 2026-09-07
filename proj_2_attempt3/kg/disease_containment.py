#!/usr/bin/env python3
"""Disease-side fragmentation: do subtype nodes hide replication and contradiction?

The taxon dimension already models containment (708 links): *Lachnospiraceae* the
family and *Hungatella* the genus inside it are separate nodes, deliberately, and
their disagreement in Parkinson's is a finding rather than noise. The disease
dimension has no such structure at all. `Intracerebral hemorrhage` sits beside
`Stroke` with no link; `Chronic traumatic complete spinal cord injury` beside
`Spinal cord injury`; `Minimal hepatic encephalopathy` beside `Hepatic
encephalopathy`.

Two prior results say this is worth measuring rather than assuming:

- Folding three spellings of anti-NMDAR encephalitis (a pure *synonym* case)
  turned 4 edges that every view showed as single-paper into replicated ones,
  3 of them CONTESTED. Fragmentation hides both replication and contradiction.
- On the taxon side, collapsing ranks would have destroyed real signal. So
  subtypes must NOT be folded the way synonyms were; containment is the right
  model, and containment is additive -- it links nodes without merging evidence.

This script does not modify the graph. It measures what a containment layer
would connect, and tests whether a subtype's microbial profile actually
resembles its parent's more than an unrelated disease's does. If it does not,
the layer is bookkeeping (still defensible, but not a signal gain) and should be
reported as such.
"""
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "disease_containment.json")

# ---------------------------------------------------------------------------
# Candidate containment pairs, (child, parent), in tiers by how much judgement
# each needs. Tier A is a clinical is-a that the label strings also carry.
# Tier B is defensible but a design decision a human should sign off on.
# Tier C is explicitly REJECTED and recorded so the reasoning is not lost.
# ---------------------------------------------------------------------------
TIER_A = [
    # Stroke: ICH is hemorrhagic stroke; hypertensive ICH is a cause-specified ICH.
    ("Intracerebral hemorrhage", "Stroke", "ICH is hemorrhagic stroke"),
    ("Hypertensive intracerebral hemorrhage", "Intracerebral hemorrhage",
     "cause-specified ICH"),
    ("Poststroke aphasia", "Stroke", "stroke sequela; cohort is stroke patients"),
    ("Hemorrhagic transformation", "Stroke",
     "complication of ischemic stroke; cohort is stroke patients"),
    # Spinal cord injury: both children are severity/level-specified SCI.
    ("Chronic traumatic complete spinal cord injury", "Spinal cord injury",
     "severity/chronicity-specified SCI"),
    ("Traumatic thoracic spinal cord injury", "Spinal cord injury",
     "level-specified traumatic SCI"),
    # Hepatic encephalopathy: minimal HE is the subclinical grade of HE.
    ("Minimal hepatic encephalopathy", "Hepatic encephalopathy",
     "subclinical grade of HE"),
]

TIER_B = [
    # Dementia is the PARENT of AD, not a child -- direction matters.
    ("Alzheimer's disease", "Dementia", "AD is the commonest dementia"),
    ("Sporadic Creutzfeldt-Jakob disease", "Dementia", "sCJD presents as dementia"),
    # The cognitive-decline continuum. MCI is a STAGE, not a subtype of AD, so it
    # is NOT placed under AD. These three labels may be near-synonyms of each
    # other rather than a hierarchy -- flagged, not folded.
    ("Mild cognitive impairment", "Cognitive impairment", "MCI is graded CI"),
    ("Neurocognitive impairment", "Cognitive impairment", "possible synonym of CI"),
    ("Subjective cognitive decline", "Cognitive impairment",
     "earliest point on the same continuum"),
]

# Recorded so a later session does not re-propose them.
TIER_C_REJECTED = {
    "Multiple system atrophy": "atypical parkinsonism -- a SIBLING of PD under "
                               "parkinsonian syndromes, not a subtype of PD",
    "Essential tremor": "distinct movement disorder, not parkinsonism",
    "REM sleep behavior disorder-Lewy body disease continuum":
        "prodromal state spanning PD and DLB; not contained by either",
    "Hepatitis B virus-associated liver cirrhosis":
        "hepatic disease, not a neurological subtype",
    "Mild cognitive impairment -> Alzheimer's disease":
        "MCI is a clinical STAGE that may or may not convert; not an AD subtype",
}


def load():
    g = json.load(open(GRAPH))
    return g


def profiles(g):
    """disease -> {taxon_key: {'up': n_papers_up, 'down': n_papers_down}}

    Counts come from the edge's own n_up / n_down, which are paper counts (fixed
    in the 2026-09-03 dedup correction: count papers, not observations).
    """
    prof = defaultdict(dict)
    for e in g["edges"]:
        prof[e["disease"]][e["taxon_key"]] = {"up": e["n_up"], "down": e["n_down"]}
    return prof


def direction_of(slot):
    """-> 'enriched' | 'depleted' | 'contested'"""
    up, dn = slot["up"], slot["down"]
    if up and dn:
        return "contested"
    return "enriched" if up else "depleted"


def compare(prof, child, parent):
    """What a containment link between child and parent would connect."""
    cp, pp = prof.get(child, {}), prof.get(parent, {})
    shared = sorted(set(cp) & set(pp))
    agree, disagree, either_contested = [], [], []
    for t in shared:
        dc, dp = direction_of(cp[t]), direction_of(pp[t])
        if dc == "contested" or dp == "contested":
            either_contested.append((t, dc, dp))
        elif dc == dp:
            agree.append((t, dc))
        else:
            disagree.append((t, dc, dp))
    # A child edge is "orphaned" if the child has only one paper for it -- the
    # kind of edge the NMDAR fold turned into a replicated one.
    child_singletons = sum(
        1 for t in cp if cp[t]["up"] + cp[t]["down"] == 1
    )
    return {
        "child": child,
        "parent": parent,
        "n_child_taxa": len(cp),
        "n_parent_taxa": len(pp),
        "n_shared": len(shared),
        "n_agree": len(agree),
        "n_disagree": len(disagree),
        "n_either_contested": len(either_contested),
        "child_singleton_edges": child_singletons,
        "agree": agree,
        "disagree": disagree,
        "either_contested": either_contested,
    }


def decisive_agreement(prof, a, b):
    """Signed concordance on taxa where BOTH diseases are non-contested.

    Returns (n_decisive, n_agree). This is the same shape as the external
    validation metric: only unambiguous calls on both sides count.
    """
    pa, pb = prof.get(a, {}), prof.get(b, {})
    n = k = 0
    for t in set(pa) & set(pb):
        da, db = direction_of(pa[t]), direction_of(pb[t])
        if da == "contested" or db == "contested":
            continue
        n += 1
        k += (da == db)
    return n, k


def null_agreement(prof, child, parent, n_iter=10000, seed=17):
    """Is child-vs-parent concordance higher than child vs an unrelated disease?

    The comparison set is every other disease with at least as many taxa as the
    true parent has shared with the child, so a pseudo-parent is not rejected
    merely for being small. This permutes the PARENT IDENTITY, which is the unit
    the hypothesis is about; it deliberately does not shuffle observations within
    a disease, which would break the paper-level clustering the project's rules
    require be preserved.
    """
    rng = random.Random(seed)
    n_true, k_true = decisive_agreement(prof, child, parent)
    if n_true == 0:
        return {"n_decisive": 0, "note": "no decisive shared taxa -- untestable"}
    others = [d for d in prof
              if d not in (child, parent) and len(prof[d]) >= 5]
    hits = 0
    rates = []
    for _ in range(n_iter):
        pseudo = rng.choice(others)
        n, k = decisive_agreement(prof, child, pseudo)
        if n == 0:
            rates.append(None)
            continue
        r = k / n
        rates.append(r)
        if r >= k_true / n_true:
            hits += 1
    valid = [r for r in rates if r is not None]
    return {
        "n_decisive": n_true,
        "n_agree": k_true,
        "rate": round(k_true / n_true, 4),
        "null_mean": round(sum(valid) / len(valid), 4) if valid else None,
        "null_n": len(valid),
        "p": round((hits + 1) / (n_iter + 1), 4),
    }


def main():
    g = load()
    prof = profiles(g)
    present = set(prof)

    results = {"tiers": {}, "rejected": TIER_C_REJECTED, "missing_labels": []}
    for tier_name, pairs in (("A", TIER_A), ("B", TIER_B)):
        rows = []
        for child, parent, why in pairs:
            if child not in present or parent not in present:
                results["missing_labels"].append(
                    {"child": child, "parent": parent,
                     "child_present": child in present,
                     "parent_present": parent in present})
                continue
            r = compare(prof, child, parent)
            r["rationale"] = why
            r["null_test"] = null_agreement(prof, child, parent)
            rows.append(r)
        results["tiers"][tier_name] = rows

    # Corpus-wide baseline: what does agreement between two arbitrary diseases
    # look like? Without this the per-pair rates mean nothing.
    ds = [d for d in prof if len(prof[d]) >= 5]
    base_n = base_k = 0
    pairwise = []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            n, k = decisive_agreement(prof, a, b)
            if n >= 3:
                pairwise.append((a, b, n, k, round(k / n, 3)))
                base_n += n
                base_k += k
    results["baseline"] = {
        "n_disease_pairs_with_3plus_decisive": len(pairwise),
        "pooled_decisive": base_n,
        "pooled_agree": base_k,
        "pooled_rate": round(base_k / base_n, 4) if base_n else None,
    }

    json.dump(results, open(OUT, "w"), indent=1)

    # ---- report ----
    print(f"corpus baseline: {base_k}/{base_n} = "
          f"{base_k / base_n:.3f} directional agreement between arbitrary "
          f"disease pairs ({len(pairwise)} pairs with >=3 decisive shared taxa)\n")
    for tier_name in ("A", "B"):
        print(f"=== TIER {tier_name} ===")
        for r in results["tiers"][tier_name]:
            nt = r["null_test"]
            print(f"\n{r['child']}  ->  {r['parent']}")
            print(f"   ({r['rationale']})")
            print(f"   child taxa {r['n_child_taxa']}, parent taxa "
                  f"{r['n_parent_taxa']}, shared {r['n_shared']}")
            print(f"   agree {r['n_agree']}  disagree {r['n_disagree']}  "
                  f"one-side-contested {r['n_either_contested']}")
            print(f"   child single-paper edges: {r['child_singleton_edges']}")
            if nt.get("n_decisive"):
                print(f"   decisive {nt['n_agree']}/{nt['n_decisive']} = "
                      f"{nt['rate']}  vs null {nt['null_mean']}  p={nt['p']}")
            else:
                print(f"   {nt.get('note')}")
            if r["disagree"]:
                print("   DISAGREEMENTS (subtype specificity, or hidden "
                      "inter-study conflict):")
                for t, dc, dp in r["disagree"][:12]:
                    print(f"     {t:38s} child={dc:9s} parent={dp}")
        print()
    if results["missing_labels"]:
        print("labels not found in graph:", results["missing_labels"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
