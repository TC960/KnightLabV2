#!/usr/bin/env python3
"""Is the disease-specificity effect actually country, cohort or method?

`disease_specificity.py` found that two papers reporting the same taxon agree on
direction 71.6% of the time when they study the SAME disease and 65.7% when they
study different ones -- a 5.9-point gap, z=3.39, p=0.0014 under a paper-level
permutation of the disease label, against a minimum detectable gap of 3.0 points.

That test is valid as far as it goes: permuting the disease label across papers
correctly absorbs the pair-level non-independence (23,627 pairs come from only
272 papers). But it cannot tell disease apart from anything that travels WITH
disease. Two same-disease papers are also disproportionately likely to share a
country, a sequencing platform and a body site, because this corpus was assembled
disease by disease. If Chinese 16S stroke studies agree with each other because
they are Chinese 16S studies, the disease label gets the credit.

This project has been burned by exactly this shape twice: 198 "explanatory" terms
that a random split reproduced (p=0.41), and a `diet_controlled` effect that went
from p=0.002 to FDR 0.243 once clustering and multiple testing were handled. So
the effect is decomposed here before it is written up as disease specificity.

Three checks:

1. **Stratify.** Compute the same-disease minus different-disease gap separately
   within same-country pairs and within different-country pairs, and likewise for
   sequencing platform. If the gap survives inside both strata it is not that
   covariate. Repeat with the covariate as the "treatment" to see how large ITS
   own gap is.
2. **Permute within stratum.** Re-run the paper-level permutation restricted so
   that the disease label is only ever swapped between papers sharing a country.
   That is the honest test of "disease, holding country fixed".
3. **Leave-one-disease-out.** Parkinson's, Alzheimer's and MS supply most of the
   corpus. Drop each disease in turn and confirm the gap is not one disease's.

Metadata covers 206 of 272 papers, so checks 1-2 run on a subset and lose power;
the reduced n is reported with every number.
"""
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "disease_specificity_confounds.json")

N_ITER = 5000
SEED = 20260905


def load():
    g = json.load(open(GRAPH))
    obs = []
    pdis = {}
    for e in g["edges"]:
        for x in e["ev"]:
            obs.append((x["i"], e["taxon_key"], "e" if x["d"] == "e" else "d"))
            pdis.setdefault(x["i"], set()).add(e["disease"])
    disease_of = {i: sorted(s)[0] for i, s in pdis.items()}
    meta = {}
    for i, p in enumerate(g["papers"]):
        meta[i] = {
            "country": (p.get("country") or "").strip() or None,
            "seq": (p.get("seq") or "").strip() or None,
            "site": (p.get("site") or "").strip() or None,
        }
    return obs, disease_of, meta


def taxon_pairs(obs):
    """All unordered pairs of distinct papers calling the same taxon.

    Materialised once: every check below is a different partition of the SAME
    pair list, so the pairing work is not repeated.
    """
    by_t = defaultdict(list)
    for i, t, d in obs:
        by_t[t].append((i, d))
    pairs = []
    for t, calls in by_t.items():
        if len(calls) < 2:
            continue
        for a in range(len(calls)):
            ia, da = calls[a]
            for b in range(a + 1, len(calls)):
                ib, db = calls[b]
                if ia != ib:
                    pairs.append((ia, ib, da == db))
    return pairs


def gap(pairs, label_of, restrict=None):
    """(same-label rate) - (diff-label rate) over pairs where both labels exist.

    `restrict` optionally filters pairs by a predicate on (ia, ib).
    """
    sn = sk = dn = dk = 0
    for ia, ib, agree in pairs:
        if restrict and not restrict(ia, ib):
            continue
        la, lb = label_of.get(ia), label_of.get(ib)
        if la is None or lb is None:
            continue
        if la == lb:
            sn += 1
            sk += agree
        else:
            dn += 1
            dk += agree
    if not sn or not dn:
        return None
    return (sk / sn - dk / dn, sk / sn, dk / dn, sn, dn)


def permute_gap(pairs, label_of, blocks=None, n_iter=N_ITER, seed=SEED,
                restrict=None):
    """Paper-level permutation of the label. `blocks` maps paper -> block; when
    given, labels are only ever swapped between papers in the same block, which
    is what holds that covariate fixed."""
    rng = random.Random(seed)
    obsv = gap(pairs, label_of, restrict)
    if obsv is None:
        return {"note": "no comparable pairs"}
    ids = [i for i in label_of if label_of[i] is not None]
    if blocks:
        grouped = defaultdict(list)
        for i in ids:
            grouped[blocks.get(i)].append(i)
    null = []
    hits = 0
    for _ in range(n_iter):
        perm = {}
        if blocks:
            for _b, members in grouped.items():
                labs = [label_of[i] for i in members]
                rng.shuffle(labs)
                perm.update(dict(zip(members, labs)))
        else:
            labs = [label_of[i] for i in ids]
            rng.shuffle(labs)
            perm = dict(zip(ids, labs))
        r = gap(pairs, perm, restrict)
        if r is None:
            continue
        null.append(r[0])
        if r[0] >= obsv[0]:
            hits += 1
    mean = sum(null) / len(null)
    sd = (sum((x - mean) ** 2 for x in null) / len(null)) ** 0.5
    return {
        "gap": round(obsv[0], 4),
        "same_rate": round(obsv[1], 4),
        "diff_rate": round(obsv[2], 4),
        "n_same_pairs": obsv[3],
        "n_diff_pairs": obsv[4],
        "null_mean": round(mean, 5),
        "null_sd": round(sd, 5),
        "z": round((obsv[0] - mean) / sd, 2) if sd else None,
        "p": round((hits + 1) / (len(null) + 1), 5),
        "min_detectable_gap_at_p05": round(sorted(null)[int(0.95 * len(null))], 4),
    }


def main():
    obs, disease_of, meta = load()
    pairs = taxon_pairs(obs)
    country = {i: m["country"] for i, m in meta.items()}
    seq = {i: m["seq"] for i, m in meta.items()}
    n_country = sum(1 for i in disease_of if country.get(i))
    n_seq = sum(1 for i in disease_of if seq.get(i))
    print(f"{len(pairs)} same-taxon paper pairs over {len(disease_of)} papers")
    print(f"country known for {n_country}/{len(disease_of)} contributing papers, "
          f"sequencing for {n_seq}\n")

    res = {"n_pairs": len(pairs), "n_papers": len(disease_of),
           "n_papers_with_country": n_country, "n_papers_with_seq": n_seq}

    # ---- how big is each covariate's OWN gap, for scale? ----
    print("--- each label's own same-vs-different gap (unstratified) ---")
    for name, lab in (("disease", disease_of), ("country", country),
                      ("sequencing", seq)):
        r = permute_gap(pairs, lab)
        res[f"gap_{name}"] = r
        if "gap" in r:
            print(f"  {name:11s} gap {r['gap']:+.4f} "
                  f"(same {r['same_rate']:.3f} / diff {r['diff_rate']:.3f}, "
                  f"n={r['n_same_pairs']}/{r['n_diff_pairs']})  "
                  f"z={r['z']} p={r['p']}  MDE {r['min_detectable_gap_at_p05']:+.4f}")

    # ---- disease gap WITHIN country strata ----
    print("\n--- disease gap, holding country fixed ---")
    same_country = lambda ia, ib: (country.get(ia) is not None
                                   and country.get(ia) == country.get(ib))
    diff_country = lambda ia, ib: (country.get(ia) is not None
                                   and country.get(ib) is not None
                                   and country.get(ia) != country.get(ib))
    for nm, restrict in (("same-country pairs only", same_country),
                         ("different-country pairs only", diff_country)):
        r = gap(pairs, disease_of, restrict)
        if r is None:
            print(f"  {nm}: no comparable pairs")
            res[f"disease_gap_{nm}"] = None
            continue
        print(f"  {nm}: gap {r[0]:+.4f} (same {r[1]:.3f} / diff {r[2]:.3f}, "
              f"n={r[3]}/{r[4]})")
        res[f"disease_gap_{nm}"] = {"gap": round(r[0], 4),
                                    "same_rate": round(r[1], 4),
                                    "diff_rate": round(r[2], 4),
                                    "n_same": r[3], "n_diff": r[4]}

    # The real test: permute disease only within country, so the permutation
    # cannot borrow strength from country structure.
    print("\n--- disease gap, permuted WITHIN country blocks (the honest test) ---")
    blocks = {i: country.get(i) for i in disease_of}
    only_known = {i: d for i, d in disease_of.items() if country.get(i)}
    r = permute_gap(pairs, only_known, blocks=blocks)
    res["disease_gap_permuted_within_country"] = r
    if "gap" in r:
        print(f"  gap {r['gap']:+.4f} (same {r['same_rate']:.3f} / "
              f"diff {r['diff_rate']:.3f}, n={r['n_same_pairs']}/"
              f"{r['n_diff_pairs']})")
        print(f"  null {r['null_mean']:+.5f} sd {r['null_sd']:.5f}  "
              f"z={r['z']}  p={r['p']}  MDE {r['min_detectable_gap_at_p05']:+.4f}")

    # ---- leave-one-disease-out ----
    print("\n--- leave-one-disease-out (is it one disease?) ---")
    counts = defaultdict(int)
    for i, d in disease_of.items():
        counts[d] += 1
    big = sorted(counts, key=lambda d: -counts[d])[:6]
    lodo = {}
    for drop in big:
        keep = {i: d for i, d in disease_of.items() if d != drop}
        r = gap(pairs, keep)
        if r is None:
            continue
        lodo[drop] = {"gap": round(r[0], 4), "n_same": r[3], "n_diff": r[4]}
        print(f"  without {drop:28s} gap {r[0]:+.4f} "
              f"(n={r[3]}/{r[4]}, {counts[drop]} papers dropped)")
    res["leave_one_disease_out"] = lodo

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
