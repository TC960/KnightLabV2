#!/usr/bin/env python3
"""Does semantic retrieval find the papers that state a variable? Scored, not eyeballed.

`audit_variable_coverage.py` labels each paper for each variable by regex: does a
sentence about THIS study state a concrete value? Those labels are strict and will
miss real statements, so they are a lower bound -- but they are independent of the
embeddings, which makes them usable as a reference to score retrieval against.

For each variable this script:
  1. runs 2-3 queries phrased as a sentence a paper would actually contain
     (querying the variable NAME does not work -- "hospital beds" retrieves author
     affiliations; see the commit for chunk_search.py)
  2. scores each paper by its best-matching chunk
  3. computes AUC: how well that score ranks the regex-labelled papers above the rest

AUC is reported first because it needs no threshold, so there is no cutoff to tune
until the answer looks good. 0.5 is chance. The null is analytic (Mann-Whitney U),
with a permutation run alongside as a check that the analytic p is not lying.

BUT AUC ALONE IS MISLEADING HERE, so precision@10/@20 is reported too. A
spot-check of `bmi` (AUC 0.799, the second-best score in the table) found only 1
of the top 4 chunks actually stated a BMI; the other three were topically adjacent
and wrong. AUC says positives outrank negatives ON AVERAGE, which is compatible
with a bad top-10 -- and the top-10 is what a human actually reads. Judge a
variable by LIFT (precision@10 / base rate), not by AUC.

Both label directions are noisy: the regex is a lower bound, so a retrieval "false
positive" may be a regex under-count rather than a retrieval error. Treat these as
agreement measures between two imperfect methods, not as accuracy.

    python variable_sweep.py
"""
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "chunks.jsonl")
VECS = os.path.join(HERE, "chunk_vecs.npy")
AUDIT = os.path.join(HERE, "audit_variable_coverage.json")
OUT = os.path.join(HERE, "variable_sweep.json")
MODEL = "all-MiniLM-L6-v2"
N_PERM = 2000

# Queries are written as prose a methods section would contain.
QUERIES = {
    "recruitment setting": [
        "patients were recruited from the outpatient clinic of our hospital",
        "healthy controls were recruited from the community",
        "participants were consecutively enrolled from the inpatient ward",
    ],
    "antibiotic use": [
        "participants had not taken antibiotics in the previous three months",
        "subjects were excluded if they had used antibiotics before sampling",
    ],
    "sample storage": [
        "fecal samples were immediately frozen and stored at -80 degrees",
        "samples were transported on dry ice and stored until DNA extraction",
    ],
    "bmi": [
        "there was no significant difference in body mass index between groups",
        "the mean body mass index of the patients was 24.5 kg/m2",
    ],
    "disease severity": [
        "disease severity was assessed using the UPDRS and Hoehn and Yahr stage",
        "cognitive function was evaluated with the MMSE and MoCA scores",
    ],
    "probiotic use": [
        "subjects who had taken probiotics or prebiotics were excluded",
        "no participant consumed yogurt or fermented products before sampling",
    ],
    "dna extraction kit": [
        "genomic DNA was extracted using the QIAamp DNA stool mini kit",
        "total DNA was isolated with the PowerSoil DNA isolation kit",
    ],
    "differential abundance method": [
        "linear discriminant analysis effect size LEfSe was used to identify taxa",
        "differential abundance was tested with DESeq2 and Wilcoxon rank sum tests",
    ],
}


def auc_and_p(scores, labels):
    """AUC via Mann-Whitney U, with the normal-approximation p-value.

    AUC = P(score of a positive > score of a negative), ties counted as half.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    u = ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0
    auc = u / (n1 * n0)
    mu = n1 * n0 / 2.0
    sd = math.sqrt(n1 * n0 * (n1 + n0 + 1) / 12.0)
    z = (u - mu) / sd if sd else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))          # two-sided
    return auc, p


def perm_p(scores, labels, n=N_PERM, seed=0):
    """Check the analytic p by shuffling labels."""
    obs, _ = auc_and_p(scores, labels)
    rng = np.random.default_rng(seed)
    y = labels.copy()
    hits = 0
    for _ in range(n):
        rng.shuffle(y)
        a, _ = auc_and_p(scores, y)
        if abs(a - 0.5) >= abs(obs - 0.5):
            hits += 1
    return (hits + 1) / (n + 1)


def main():
    rows = [json.loads(l) for l in open(CHUNKS)]
    V = np.load(VECS)
    titles = np.array([r["title"] for r in rows])
    audit = {r["variable"]: r for r in json.load(open(AUDIT))["rows"]}

    uniq = sorted(set(titles))
    idx = {t: i for i, t in enumerate(uniq)}
    chunk_paper = np.array([idx[t] for t in titles])

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)

    out = []
    print(f"{len(uniq)} papers, {len(rows)} chunks\n")
    print(f"{'variable':<30} {'regex n':>7} {'AUC':>6} {'p(anal)':>9} "
          f"{'p(perm)':>8} {'P@10':>6} {'P@20':>6} {'base':>6}")
    print("-" * 84)

    for var, qs in QUERIES.items():
        if var not in audit:
            continue
        qv = m.encode(qs, normalize_embeddings=True)
        sims = V @ qv.T                       # (chunks, queries)
        best_per_chunk = sims.max(axis=1)

        # a paper scores as its single best-matching chunk
        paper_score = np.full(len(uniq), -1.0)
        np.maximum.at(paper_score, chunk_paper, best_per_chunk)

        pos = set(audit[var]["valued_titles"])
        labels = np.array([1 if t in pos else 0 for t in uniq])

        auc, pa = auc_and_p(paper_score, labels)
        pp = perm_p(paper_score, labels)

        order = np.argsort(-paper_score)
        # Precision@k matters more than AUC for the actual use case: you read the
        # top hits. A high AUC only says positives rank above negatives ON
        # AVERAGE, which a spot-check showed can coexist with a poor top-10.
        # NOTE both directions of error: regex labels are a lower bound, so a
        # "miss" here may be a regex under-count rather than a retrieval error.
        p10 = labels[order[:10]].mean()
        p20 = labels[order[:20]].mean()
        base = labels.mean()
        top = order[:50]
        new = [uniq[i] for i in top if labels[i] == 0]

        print(f"{var:<30} {labels.sum():>7} {auc:>6.3f} {pa:>9.2e} "
              f"{pp:>8.4f} {p10:>6.2f} {p20:>6.2f} {base:>6.2f}")

        out.append({
            "variable": var, "queries": qs, "n_regex_positive": int(labels.sum()),
            "auc": round(float(auc), 4), "p_analytic": pa, "p_perm": pp,
            "prec_at_10": round(float(p10), 3), "prec_at_20": round(float(p20), 3),
            "base_rate": round(float(base), 3),
            "lift_at_10": round(float(p10 / base), 2) if base else None,
            "unlabelled_in_top50": new[:12],
        })

    print("\nP@10 vs base rate = lift. Lift is the honest measure of whether")
    print("reading the top hits beats reading random papers.")
    for r in sorted(out, key=lambda x: -(x["lift_at_10"] or 0)):
        print(f"  {r['variable']:<30} P@10={r['prec_at_10']:.2f}  "
              f"base={r['base_rate']:.2f}  lift={r['lift_at_10']}x")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
