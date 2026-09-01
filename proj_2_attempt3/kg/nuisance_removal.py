#!/usr/bin/env python3
"""Delete the "which disease is this" direction from the embedding space.

WHY. The contrast method (mean of up-papers minus mean of down-papers) finds the
direction that most distinguishes two piles of papers. The problem is that the
single largest source of variation in this corpus is which DISEASE a paper is
about. If the up-pile skews Parkinson's and the down-pile skews Alzheimer's, the
difference vector is a disease direction, and reading its extremes rediscovers
something we already know and have in a column.

FIX. Disease is a nuisance variable, and a nuisance direction can simply be
deleted. Compute a centroid per disease, take the subspace those centroids span,
and project every chunk vector onto its orthogonal complement:

    P = I - U U^T          (U = orthonormal basis of the disease subspace)
    v' = normalise(P v)

Nothing is trained. No parameters, so nothing can overfit in the usual sense --
but the subspace IS estimated from data, so it is fitted on a TRAIN split of
papers and evaluated on held-out papers. Estimating it on all 303 and then
reporting that disease became unpredictable would be circular.

THE TEST THAT MATTERS. Removal must be surgical, not lobotomy:

  - disease predictability should COLLAPSE toward chance   (it worked)
  - study-design signal should SURVIVE                     (we did not nuke the space)

The second is the real check. A projection that destroys everything would ace
the first test and be useless. Study-design signal is measured with the same
probe queries as variable_sweep.py, scored against the same independent regex
labels, before vs after.

    python nuisance_removal.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "chunks.jsonl")
VECS = os.path.join(HERE, "chunk_vecs.npy")
AUDIT = os.path.join(HERE, "audit_variable_coverage.json")
OUT_VECS = os.path.join(HERE, "chunk_vecs_deconfounded.npy")
OUT = os.path.join(HERE, "nuisance_removal.json")
MODEL = "all-MiniLM-L6-v2"
MIN_PAPERS = 4          # a disease needs this many papers to get a centroid
SEED = 0

PROBES = {
    "recruitment setting": [
        "patients were recruited from the outpatient clinic of our hospital",
        "healthy controls were recruited from the community",
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
    "differential abundance method": [
        "linear discriminant analysis effect size LEfSe was used to identify taxa",
        "differential abundance was tested with DESeq2 and Wilcoxon rank sum tests",
    ],
}


def auc(scores, labels):
    """AUC via Mann-Whitney U with tie-averaged ranks (checked against sklearn)."""
    n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
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
    return u / (n1 * n0)


def sv(v):
    return "" if v is None or isinstance(v, float) else str(v).strip()


def disease_map():
    out = {}
    for src in ["../EmilySong_GoldStandardPaper/all_usable_papers.json",
                "new_papers.json"]:
        p = os.path.join(HERE, src)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            t, d = sv(r.get("title")), sv(r.get("disease"))
            if t and d:
                out.setdefault(t, d)
    return out


def main():
    rows = [json.loads(l) for l in open(CHUNKS)]
    V = np.load(VECS).astype(np.float64)
    titles = np.array([r["title"] for r in rows])
    dmap = disease_map()

    papers = sorted(set(titles))
    pidx = {t: i for i, t in enumerate(papers)}
    chunk_paper = np.array([pidx[t] for t in titles])
    pdisease = np.array([dmap.get(t, "") for t in papers])

    # paper vector = mean of its chunks, for the disease evaluation
    def paper_matrix(M):
        P = np.zeros((len(papers), M.shape[1]))
        np.add.at(P, chunk_paper, M)
        cnt = np.bincount(chunk_paper, minlength=len(papers))[:, None]
        P = P / np.maximum(cnt, 1)
        return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)

    labelled = np.array([i for i, d in enumerate(pdisease) if d])
    counts = {}
    for i in labelled:
        counts[pdisease[i]] = counts.get(pdisease[i], 0) + 1
    keep = {d for d, c in counts.items() if c >= MIN_PAPERS}
    usable = np.array([i for i in labelled if pdisease[i] in keep])
    print(f"{len(papers)} papers, {len(rows)} chunks")
    print(f"{len(keep)} diseases with >={MIN_PAPERS} papers, "
          f"covering {len(usable)} papers\n")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(usable)
    cut = int(0.6 * len(perm))
    train, test = perm[:cut], perm[cut:]

    # --- build the disease subspace on TRAIN papers only -------------------
    P_orig = paper_matrix(V)
    cents, names = [], []
    for d in sorted(keep):
        idx = [i for i in train if pdisease[i] == d]
        if len(idx) >= 2:
            cents.append(P_orig[idx].mean(axis=0))
            names.append(d)
    C = np.vstack(cents)
    C = C - C.mean(axis=0, keepdims=True)     # centre: remove the global mean first
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    # keep components explaining 95% of the between-disease variance
    ev = (S ** 2) / (S ** 2).sum()
    k = int(np.searchsorted(np.cumsum(ev), 0.95) + 1)
    # The RIGHT singular vectors are the principal directions in the 384-dim
    # embedding space, so the basis must come from Vt.
    #
    # This previously read `np.linalg.qr(C.T)[0][:, :k]`, which is Gram-Schmidt
    # over the centroids IN ALPHABETICAL ORDER OF DISEASE NAME -- an orthonormal
    # basis for a subspace, but not the top-variance one. Five of seven
    # directions happened to coincide; two were nearly orthogonal to the
    # intended ones (principal-angle cosines 0.161 and 0.007). The printed "96%
    # of variance" described the SVD subspace while the code removed a different
    # one capturing 83.3%, and held-out disease accuracy stalled at ~0.52
    # instead of 0.14. That was misread as a method limitation needing INLP
    # iteration; it was this line.
    B = Vt[:k].T                              # orthonormal basis, 384 x k
    print(f"disease subspace: {len(names)} centroids -> rank {k} "
          f"({100*ev[:k].sum():.0f}% of between-disease variance)\n")

    V2 = V - (V @ B) @ B.T
    V2 = V2 / np.maximum(np.linalg.norm(V2, axis=1, keepdims=True), 1e-9)

    # --- test 1: is disease still predictable on HELD-OUT papers? ----------
    def disease_acc(M):
        Pm = paper_matrix(M)
        cs, ns = [], []
        for d in sorted(keep):
            idx = [i for i in train if pdisease[i] == d]
            if idx:
                cs.append(Pm[idx].mean(axis=0))
                ns.append(d)
        Cm = np.vstack(cs)
        Cm = Cm / np.maximum(np.linalg.norm(Cm, axis=1, keepdims=True), 1e-9)
        pred = [ns[j] for j in np.argmax(Pm[test] @ Cm.T, axis=1)]
        return float(np.mean([p == pdisease[i] for p, i in zip(pred, test)]))

    big = max(counts[d] for d in keep)
    chance = big / sum(counts[d] for d in keep)
    a_before, a_after = disease_acc(V), disease_acc(V2)
    print("TEST 1 -- disease predictability on held-out papers "
          f"(n={len(test)}, {len(keep)} classes)")
    print(f"  majority-class baseline : {chance:.3f}")
    print(f"  before removal          : {a_before:.3f}")
    print(f"  after  removal          : {a_after:.3f}")
    print(f"  -> {'COLLAPSED (good)' if a_after < (a_before+chance)/2 else 'STILL PRESENT'}\n")

    # --- test 2: did study-design signal survive? --------------------------
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    audit = {r["variable"]: r for r in json.load(open(AUDIT))["rows"]}

    print("TEST 2 -- study-design signal, AUC vs the same regex labels")
    print(f"  {'variable':<30} {'before':>7} {'after':>7} {'delta':>7}")
    res = []
    for var, qs in PROBES.items():
        if var not in audit:
            continue
        qv = m.encode(qs, normalize_embeddings=True).astype(np.float64)
        pos = set(audit[var]["valued_titles"])
        y = np.array([1 if t in pos else 0 for t in papers])

        def var_auc(M, q):
            sims = (M @ q.T).max(axis=1)
            ps = np.full(len(papers), -1e9)
            np.maximum.at(ps, chunk_paper, sims)
            return auc(ps, y)

        # the probe must live in the SAME space as the chunks it is compared to
        qv2 = qv - (qv @ B) @ B.T
        qv2 = qv2 / np.maximum(np.linalg.norm(qv2, axis=1, keepdims=True), 1e-9)
        b, a = var_auc(V, qv), var_auc(V2, qv2)
        print(f"  {var:<30} {b:>7.3f} {a:>7.3f} {a-b:>+7.3f}")
        res.append({"variable": var, "auc_before": round(b, 4),
                    "auc_after": round(a, 4), "delta": round(a - b, 4)})

    mean_d = float(np.mean([r["delta"] for r in res]))
    print(f"\n  mean delta: {mean_d:+.3f}")
    print("  (near zero = surgical. strongly negative = we destroyed the space.)")

    np.save(OUT_VECS, V2.astype(np.float32))
    np.save(os.path.join(HERE, "disease_basis.npy"), B.astype(np.float32))
    json.dump({"rank": int(k), "diseases": names,
               "disease_acc_before": a_before, "disease_acc_after": a_after,
               "chance": chance, "variables": res, "mean_delta": mean_d},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT_VECS}, disease_basis.npy, {OUT}")


if __name__ == "__main__":
    main()
