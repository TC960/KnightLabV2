#!/usr/bin/env python3
"""Pooled test: do papers reporting ENRICHMENT differ from papers reporting
DEPLETION in their taxon co-occurrence profile?

Substrate: `relation_sentences.json` (the validated relation-bearing sentence
filter, 94.8% recall of a 97.3% ceiling). A paper's profile is the binary
incidence vector over the taxids named in its KEPT sentences.

The comparison is made WITHIN a fixed (taxon, disease) edge, so disease is
controlled by construction. Pairs of papers on the same edge are labelled
same-direction or different-direction; the statistic is the difference in mean
profile similarity between the two pair classes.

Observations are NOT independent -- one paper contributes to many edges -- so
both nulls randomise at the PAPER level:

  null A  profile permutation, WITHIN disease. Every edge keeps its exact
          up/down group sizes; only the paper->profile association is broken.
  null B  whole-paper sign flip. A paper's directions all flip together,
          preserving its internal structure and re-partitioning the edges.

Neither null shuffles individual observations, which would understate the null
variance and manufacture a false positive (as pair-level shuffling did for the
`diet_controlled` and ASD results already on record).
"""
import json
import re
import sys
import numpy as np

RNG = np.random.default_rng(20260903)
N_PERM = 2000


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load():
    g = json.load(open("graph.json"))
    rs = json.load(open("relation_sentences.json"))["papers"]
    rsn = {norm(k): k for k in rs}

    titles = [p["title"] for p in g["papers"]]
    missing = [t for t in titles if norm(t) not in rsn]
    if missing:
        sys.exit(f"FATAL: {len(missing)} graph papers absent from relation_sentences")

    # vocabulary over kept (relation-bearing) sentences only
    vocab = {}
    prof_sets = []
    disease = []
    for t in titles:
        rec = rs[rsn[norm(t)]]
        s = set()
        for sent in rec["kept"]:
            for tx in sent["taxa"]:
                s.add(tx[1])
        prof_sets.append(s)
        disease.append(rec.get("disease", "?"))
        for tid in s:
            vocab.setdefault(tid, len(vocab))

    X = np.zeros((len(titles), len(vocab)), dtype=np.float64)
    for i, s in enumerate(prof_sets):
        for tid in s:
            X[i, vocab[tid]] = 1.0
    return g, titles, X, vocab, np.array(disease), prof_sets


def edge_pairs(g, X, vocab):
    """Per contested edge, the within-edge paper pairs and their similarity.

    Returns a list of edge records; similarity is cosine on the binary profile
    with the edge's OWN focal taxon column removed (it is present in every
    paper of the edge by construction and would only add a constant).
    """
    out = []
    for e in g["edges"]:
        if not e["contested"]:
            continue
        # per-paper direction; drop papers that report both ways on this edge
        d = {}
        for ob in e["ev"]:
            d.setdefault(ob["i"], set()).add(ob["d"])
        members = [(i, list(v)[0]) for i, v in d.items() if len(v) == 1]
        if len(members) < 3:
            continue  # need >=1 same-direction pair AND >=1 cross pair
        ups = [i for i, s in members if s == "e"]
        dns = [i for i, s in members if s == "d"]
        if not ups or not dns:
            continue
        idx = [i for i, _ in members]
        col = vocab.get(e["taxon_key"].split(":")[-1])
        sub = X[idx].copy()
        if col is not None and col < sub.shape[1]:
            sub[:, col] = 0.0
        nrm = np.linalg.norm(sub, axis=1)
        keep = nrm > 0
        if keep.sum() < 3:
            continue
        idx = [i for i, k in zip(idx, keep) if k]
        sub = sub[keep]
        nrm = nrm[keep]
        S = (sub @ sub.T) / np.outer(nrm, nrm)
        lab = np.array([1 if i in ups else 0 for i in idx])
        iu, ju = np.triu_indices(len(idx), 1)
        same = lab[iu] == lab[ju]
        if same.sum() == 0 or (~same).sum() == 0:
            continue
        out.append(
            dict(
                key=(e["taxon"], e["disease"]),
                idx=np.array(idx),
                col=col,
                iu=iu,
                ju=ju,
                sims=S[iu, ju],
                _lab=lab,
                n_up=len(ups),
                n_dn=len(dns),
            )
        )
    return out


def statistic(edges, lab_of):
    """mean sim(same-direction pairs) - mean sim(different-direction pairs).

    Pooled over every within-edge pair (`pooled`), and averaged over edges so
    that a single well-papered edge cannot dominate (`per_edge`).
    """
    s_all, d_all, per_edge = [], [], []
    for e in edges:
        lab = lab_of(e)
        same = lab[e["iu"]] == lab[e["ju"]]
        if same.sum() == 0 or (~same).sum() == 0:
            continue
        ss, dd = e["sims"][same], e["sims"][~same]
        s_all.append(ss)
        d_all.append(dd)
        per_edge.append(ss.mean() - dd.mean())
    if not per_edge:
        return np.nan, np.nan
    pooled = np.concatenate(s_all).mean() - np.concatenate(d_all).mean()
    return pooled, float(np.mean(per_edge))


def stat_permuted_profiles(edges, X, perm):
    """Statistic recomputed after the paper->profile map is permuted."""
    s_all, d_all, per_edge = [], [], []
    for e in edges:
        sub = X[perm[e["idx"]]].copy()
        if e["col"] is not None:
            sub[:, e["col"]] = 0.0
        nrm = np.linalg.norm(sub, axis=1)
        ok = nrm > 0
        if ok.sum() < 3:
            continue
        S = np.zeros((len(nrm), len(nrm)))
        nz = np.where(ok)[0]
        S[np.ix_(nz, nz)] = (sub[nz] @ sub[nz].T) / np.outer(nrm[nz], nrm[nz])
        lab, iu, ju = e["_lab"], e["iu"], e["ju"]
        valid = ok[iu] & ok[ju]
        same = (lab[iu] == lab[ju]) & valid
        diff = (lab[iu] != lab[ju]) & valid
        if same.sum() == 0 or diff.sum() == 0:
            continue
        sims = S[iu, ju]
        s_all.append(sims[same])
        d_all.append(sims[diff])
        per_edge.append(sims[same].mean() - sims[diff].mean())
    if not per_edge:
        return np.nan, np.nan
    return (np.concatenate(s_all).mean() - np.concatenate(d_all).mean(),
            float(np.mean(per_edge)))


def main():
    g, titles, X, vocab, disease, prof_sets = load()
    edges = edge_pairs(g, X, vocab)
    n_pairs = sum(len(e["sims"]) for e in edges)
    n_papers_used = len(set(int(i) for e in edges for i in e["idx"]))
    print(f"contested edges usable: {len(edges)}  within-edge pairs: {n_pairs}")
    print(f"distinct papers involved: {n_papers_used} of {len(titles)}")
    print(f"vocabulary: {len(vocab)} taxids; median profile "
          f"{np.median(X.sum(1)):.0f} taxa")

    obs_pooled, obs_edge = statistic(edges, lambda e: e["_lab"])
    print(f"\nOBSERVED  pooled {obs_pooled:+.4f}   per-edge {obs_edge:+.4f}")
    print("(positive = same-direction papers have MORE similar taxon profiles)")

    res = {"n_edges": len(edges), "n_pairs": int(n_pairs),
           "n_papers": n_papers_used, "n_vocab": len(vocab),
           "n_perm": N_PERM,
           "observed": {"pooled": float(obs_pooled), "per_edge": float(obs_edge)}}

    for kind, name in (("A", "profile permutation within disease"),
                       ("B", "whole-paper sign flip")):
        vals = np.zeros((N_PERM, 2))
        for k in range(N_PERM):
            if kind == "A":
                perm = np.arange(len(titles))
                for dz in np.unique(disease):
                    m = np.where(disease == dz)[0]
                    perm[m] = RNG.permutation(m)
                vals[k] = stat_permuted_profiles(edges, X, perm)
            else:
                flip = RNG.random(len(titles)) < 0.5
                vals[k] = statistic(
                    edges,
                    lambda e, f=flip: np.where(f[e["idx"]], 1 - e["_lab"], e["_lab"]))
        print(f"\nnull {kind}: {name}")
        for j, sname in enumerate(("pooled", "per_edge")):
            o = obs_pooled if j == 0 else obs_edge
            null = vals[:, j]
            null = null[~np.isnan(null)]
            p = (np.sum(np.abs(null - null.mean()) >= abs(o - null.mean())) + 1) / (
                len(null) + 1)
            mde = 1.96 * null.std()
            print(f"  [{sname:8s}] null mean {null.mean():+.4f} sd {null.std():.4f}"
                  f"  p={p:.3f}  min-detectable |effect| {mde:.4f}")
            res.setdefault(f"null_{kind}", {})[sname] = dict(
                null_mean=float(null.mean()), null_sd=float(null.std()),
                p=float(p), min_detectable=float(mde), n_draws=int(len(null)))

    json.dump(res, open("cooccur_direction.json", "w"), indent=1)
    print("\nwrote cooccur_direction.json")


if __name__ == "__main__":
    main()
