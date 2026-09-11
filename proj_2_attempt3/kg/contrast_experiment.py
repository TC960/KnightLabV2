#!/usr/bin/env python3
"""Do the papers that disagree about an edge differ in study design? Stratified test.

THE DESIGN PROBLEM, AND WHY THIS IS THE ONLY SHAPE THAT WORKS
-------------------------------------------------------------
Two earlier attempts are dead for reasons already measured:

  * Per-edge testing is arithmetically impossible for all but one edge.
    A 4-vs-3 edge admits only C(7,3)=35 label splits, so the smallest
    permutation p is 0.029; after BH across ~34 probes the best achievable q
    is 1.0. No effect size can produce a result. Of 226 contested edges only
    ONE (Bacteroides/Alzheimer's, 7v7) could ever reach q<0.05 alone.

  * Naively pooling raw (paper, edge) observations reintroduces the fault that
    capped the earlier probe at 0.706: 65% of papers report BOTH directions, so
    the same paper vector carries opposite labels.

The fix turns on one fact: WITHIN A SINGLE EDGE, EACH PAPER HAS EXACTLY ONE
LABEL. So compute the contrast per edge, where the feature and label finally
agree, then AGGREGATE the per-edge statistics. Each edge is weak; 47 of them
together may not be.

THE STATISTIC
-------------
Per edge e and probe p, the rank-based effect is the Mann-Whitney AUC:

    A(e,p) = P(an up-paper scores higher than a down-paper), ties at 0.5

Rank-based because n is 2-7 per side, where means and SDs are meaningless.
Aggregate across edges weighted by n_up*n_dn (the number of comparable pairs),
which is the stratified Mann-Whitney / van Elteren test:

    T(p) = sum_e w_e * (A(e,p) - 0.5) / sum_e w_e

T > 0 means the concept is more present in up-papers, T < 0 in down-papers.

THE NULL
--------
Labels are shuffled WITHIN each edge, preserving each edge's up/down counts and
the paper composition. 10,000 draws, two-sided, BH across probes.

  KNOWN LIMITATION, stated rather than buried: edges share papers (211
  contributing papers across 226 edges), so edges are not independent and this
  null is mildly anti-conservative. A paper-level block permutation is the
  stricter alternative but is not well defined here, because a paper's label
  legitimately DIFFERS between edges -- that is the whole phenomenon. Read a
  surviving hit as a lead to check, not a settled result.

Paper score for a probe is the MAX cosine over that paper's chunks, never the
mean: a paper states its storage protocol in one sentence out of ~69, and
averaging drowns it.

Runs on the DISEASE-DECONFOUNDED vectors by default. How far that goes is an
empirical question, not a guarantee: check the held-out disease accuracy printed
by nuisance_removal.py. If it sits above chance, disease is ATTENUATED, not
eliminated, and a hit could still be partly a disease effect. Probes are
projected through the same basis -- comparing a raw probe against deconfounded
chunks is meaningless.

    python contrast_experiment.py
    python contrast_experiment.py --raw-space     # sanity: how much did disease drive it?
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "chunks.jsonl")
VECS = os.path.join(HERE, "chunk_vecs.npy")
DECONF = os.path.join(HERE, "chunk_vecs_deconfounded.npy")
BASIS = os.path.join(HERE, "disease_basis.npy")
GRAPH = os.path.join(HERE, "graph.json")
BANK = os.path.join(HERE, "probe_bank.json")
OUT = os.path.join(HERE, "contrast_experiment.json")
# --raw-space must NOT share the primary output path: running the sanity check
# silently overwrote the headline result, so the artifact on disk contradicted
# the numbers being quoted from it.
OUT_RAW = os.path.join(HERE, "contrast_experiment_rawspace.json")
OUT_CLEAN = os.path.join(HERE, "contrast_experiment_clean.json")
MODEL = "all-MiniLM-L6-v2"
N_PERM = 10000
MIN_SIDE = 2          # an edge needs this many papers on BOTH sides to contribute


def auc(up, dn):
    """P(up > dn) with ties at 0.5. Exact; n is tiny."""
    if len(up) == 0 or len(dn) == 0:
        return None
    g = sum((u > d) + 0.5 * (u == d) for u in up for d in dn)
    return g / (len(up) * len(dn))


def bh(ps):
    n = len(ps)
    order = np.argsort(ps)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), 1):
        val = ps[i] * n / (n - rank + 1)
        prev = min(prev, val)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-space", action="store_true",
                    help="use the original vectors instead of the deconfounded ones")
    ap.add_argument("--clean-only", action="store_true",
                    help="keep only papers with a CONSISTENT stance across all their "
                         "contested edges. within_between_decomposition.py shows only "
                         "17%% of pairs compare two such papers -- the rest pit a paper "
                         "against itself, since a paper reporting both directions "
                         "carries an identical feature vector on both sides. This is "
                         "the honest denominator: a study-level variable can only "
                         "explain BETWEEN-study contrasts.")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(CHUNKS)]
    use_deconf = (not a.raw_space) and os.path.exists(DECONF)
    V = np.load(DECONF if use_deconf else VECS).astype(np.float64)
    if len(rows) != V.shape[0]:
        raise SystemExit(f"{len(rows)} chunks vs {V.shape[0]} vectors -- rebuild")
    print(f"space: {'DECONFOUNDED (disease removed)' if use_deconf else 'RAW'}"
          f"  {V.shape}")

    titles = [r["title"] for r in rows]
    papers = sorted(set(titles))
    pidx = {t: i for i, t in enumerate(papers)}
    chunk_paper = np.array([pidx[t] for t in titles])

    bank = {k: v for k, v in json.load(open(BANK)).items()
            if not k.startswith("_") and isinstance(v, list)}
    names = sorted(bank)

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    B = np.load(BASIS).astype(np.float64) if use_deconf else None

    # paper x probe matrix of max-cosine scores
    S = np.zeros((len(papers), len(names)))
    for j, nm in enumerate(names):
        q = m.encode(bank[nm], normalize_embeddings=True).astype(np.float64)
        if B is not None:
            q = q - (q @ B) @ B.T
            q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        best = (V @ q.T).max(axis=1)
        col = np.full(len(papers), -1e9)
        np.maximum.at(col, chunk_paper, best)
        S[:, j] = col
    print(f"scored {len(papers)} papers x {len(names)} concepts")

    # contested edges -> (up paper indices, down paper indices)
    g = json.load(open(GRAPH))
    gpapers = g["papers"]
    edges = []
    for e in g["edges"]:
        if not e.get("contested"):
            continue
        up, dn = [], []
        for ev in (e.get("ev") or e.get("evidence") or []):
            i, d = ev.get("i"), ev.get("d")
            if i is None or d is None or i >= len(gpapers):
                continue
            # graph.json's paper table holds dicts (title + study metadata),
            # not bare title strings
            rec = gpapers[i]
            t = rec["title"] if isinstance(rec, dict) else rec
            if t not in pidx:
                continue
            (up if d == "e" else dn).append(pidx[t])
        up, dn = sorted(set(up)), sorted(set(dn))
        if len(up) >= MIN_SIDE and len(dn) >= MIN_SIDE:
            edges.append((up, dn))
    if a.clean_only:
        stance = {}
        for up, dn in edges:
            for i in up:
                stance.setdefault(i, set()).add("e")
            for i in dn:
                stance.setdefault(i, set()).add("d")
        keep = {i for i, v in stance.items() if len(v) == 1}
        filt = []
        for up, dn in edges:
            u2 = [i for i in up if i in keep]
            d2 = [i for i in dn if i in keep]
            if u2 and d2:
                filt.append((u2, d2))
        print(f"--clean-only: {len(keep)} of {len(stance)} papers have a consistent "
              f"stance; {len(filt)} of {len(edges)} edges retain both sides")
        edges = filt
    if not edges:
        raise SystemExit("no edges with enough papers on both sides")
    w = np.array([len(u) * len(d) for u, d in edges], float)
    print(f"{len(edges)} contested edges with >={MIN_SIDE} papers on both sides "
          f"({int(w.sum())} comparable pairs)\n")

    def stat(labelsets):
        """Weighted mean of (AUC - 0.5) across edges, per probe."""
        acc = np.zeros(len(names))
        for (up, dn), wi in zip(labelsets, w):
            su, sd = S[up], S[dn]
            for j in range(len(names)):
                acc[j] += wi * (auc(su[:, j], sd[:, j]) - 0.5)
        return acc / w.sum()

    obs = stat(edges)

    rng = np.random.default_rng(0)
    null = np.zeros((N_PERM, len(names)))
    pooled = [(np.array(u + d), len(u)) for u, d in edges]
    for k in range(N_PERM):
        shuf = []
        for arr, nu in pooled:
            p = rng.permutation(arr)
            shuf.append((p[:nu].tolist(), p[nu:].tolist()))
        null[k] = stat(shuf)
    p = (np.sum(np.abs(null) >= np.abs(obs), axis=0) + 1) / (N_PERM + 1)
    q = bh(p)

    order = np.argsort(-np.abs(obs))
    print(f"{'concept':<28} {'effect':>8} {'p':>8} {'q(BH)':>8}   direction")
    print("-" * 72)
    res = []
    for j in order:
        d = "up-papers" if obs[j] > 0 else "down-papers"
        star = " *" if q[j] < 0.05 else ""
        print(f"{names[j]:<28} {obs[j]:>+8.4f} {p[j]:>8.4f} {q[j]:>8.3f}   {d}{star}")
        res.append({"concept": names[j], "effect": round(float(obs[j]), 5),
                    "p": float(p[j]), "q": float(q[j]), "higher_in": d})

    sig = [r for r in res if r["q"] < 0.05]
    print(f"\n{len(sig)} concept(s) survive BH at q<0.05 out of {len(names)}")
    if not sig:
        sd = null.std(axis=0).mean()
        # 1.96*SD is the threshold for an UNCORRECTED single test at 50% power.
        # The actual decision rule is BH q<0.05 across all probes, which is far
        # stricter, so quoting 1.96*SD as "the minimum detectable effect"
        # overstates sensitivity by roughly 1.6-2x.
        from math import sqrt
        n_p = len(names)
        z_bh = 3.18 if n_p > 20 else 2.81      # ~BH-corrected z, best case
        print(f"  NULL RESULT. Mean null SD of the statistic is {sd:.4f}.")
        print(f"    uncorrected single test, 50% power : {1.96*sd:.3f}")
        print(f"    BH q<0.05 across {n_p} probes, 50% power : {z_bh*sd:.3f}")
        print(f"    BH q<0.05, 80% power                : {(z_bh+0.84)*sd:.3f}")
        print(f"    largest observed effect             : {np.abs(obs).max():.3f}")
        print("  The largest effect is below even the 50%-power BH threshold.")
        print("  Report as 'no study-design concept separates the camps at n=303',\n"
              "  with that power statement attached -- not as 'there is no effect'.")

    out_path = OUT_CLEAN if a.clean_only else (OUT if use_deconf else OUT_RAW)
    json.dump({"space": "deconfounded" if use_deconf else "raw",
               "n_edges": len(edges), "n_pairs": int(w.sum()),
               "n_perm": N_PERM, "results": res}, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
