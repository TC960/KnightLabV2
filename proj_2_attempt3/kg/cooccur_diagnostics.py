#!/usr/bin/env python3
"""Attack the co-occurrence result from `cooccur_direction.py`.

The headline (same-direction papers on a contested edge have more similar taxon
profiles, per-edge +0.047, p<0.001) has an obvious mechanical explanation that
the two paper-level nulls do NOT rule out:

  most contested edges are unbalanced (e.g. 5 up, 1 down), so every
  cross-direction pair contains the lone dissenter. If dissenting papers are
  merely ATYPICAL -- fewer taxa, so lower cosine against everything -- the
  effect appears with no relationship between direction and profile CONTENT.

Variants tested here:
  balanced   only edges with >=2 papers on BOTH sides (no lone dissenter)
  dedup      near-duplicate paper pairs (cosine >= 0.8) removed
  size       permutation restricted within profile-size quintile
  country    permutation restricted within country
  seq        permutation restricted within sequencing type
  jaccard    same test, Jaccard instead of cosine
  median     per-edge MEDIAN rather than mean (outlier-robust)

Self-check: the cosine/all-edges variant must reproduce cooccur_direction.json.
"""
import json
import re
import sys
import numpy as np

RNG = np.random.default_rng(20260903)
N_PERM = 1000


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load():
    g = json.load(open("graph.json"))
    rs = json.load(open("relation_sentences.json"))["papers"]
    rsn = {norm(k): k for k in rs}
    titles = [p["title"] for p in g["papers"]]
    vocab, prof, disease = {}, [], []
    for t in titles:
        rec = rs[rsn[norm(t)]]
        s = {tx[1] for sent in rec["kept"] for tx in sent["taxa"]}
        prof.append(s)
        disease.append(rec.get("disease", "?"))
        for tid in s:
            vocab.setdefault(tid, len(vocab))
    X = np.zeros((len(titles), len(vocab)))
    for i, s in enumerate(prof):
        for tid in s:
            X[i, vocab[tid]] = 1.0
    return g, titles, X, vocab, np.array(disease)


def simmat(sub, metric):
    """Pairwise similarity of binary rows; rows with no taxa give 0."""
    n = sub.sum(1)
    ok = n > 0
    S = np.zeros((len(n), len(n)))
    inter = sub @ sub.T
    if metric == "cosine":
        den = np.sqrt(np.outer(n, n))
    else:  # jaccard
        den = np.add.outer(n, n) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.where(den > 0, inter / np.where(den == 0, 1, den), 0.0)
    S[np.ix_(np.where(ok)[0], np.where(ok)[0])] = V[np.ix_(
        np.where(ok)[0], np.where(ok)[0])]
    return S, ok


def build_edges(g, X, vocab, metric="cosine"):
    out = []
    for e in g["edges"]:
        if not e["contested"]:
            continue
        d = {}
        for ob in e["ev"]:
            d.setdefault(ob["i"], set()).add(ob["d"])
        members = [(i, list(v)[0]) for i, v in d.items() if len(v) == 1]
        ups = [i for i, s in members if s == "e"]
        dns = [i for i, s in members if s == "d"]
        if len(members) < 3 or not ups or not dns:
            continue
        idx = [i for i, _ in members]
        col = vocab.get(e["taxon_key"].split(":")[-1])
        sub = X[idx].copy()
        if col is not None:
            sub[:, col] = 0.0
        S, ok = simmat(sub, metric)
        idx = [i for i, k in zip(idx, ok) if k]
        if len(idx) < 3:
            continue
        keep = np.where(ok)[0]
        S = S[np.ix_(keep, keep)]
        lab = np.array([1 if i in ups else 0 for i in idx])
        iu, ju = np.triu_indices(len(idx), 1)
        same = lab[iu] == lab[ju]
        if same.sum() == 0 or (~same).sum() == 0:
            continue
        out.append(dict(key=(e["taxon"], e["disease"]), idx=np.array(idx),
                        col=col, iu=iu, ju=ju, sims=S[iu, ju], lab=lab,
                        n_up=int(sum(lab)), n_dn=int(len(lab) - sum(lab))))
    return out


def stat(edges, sims_of, agg="mean"):
    s_all, d_all, per_edge = [], [], []
    for e in edges:
        sims = sims_of(e)
        if sims is None:
            continue
        lab = e["lab"]
        same = lab[e["iu"]] == lab[e["ju"]]
        if same.sum() == 0 or (~same).sum() == 0:
            continue
        ss, dd = sims[same], sims[~same]
        s_all.append(ss)
        d_all.append(dd)
        per_edge.append(ss.mean() - dd.mean())
    if not per_edge:
        return np.nan, np.nan
    pooled = np.concatenate(s_all).mean() - np.concatenate(d_all).mean()
    agg_fn = np.mean if agg == "mean" else np.median
    return float(pooled), float(agg_fn(per_edge))


def permuted_sims(e, X, perm, metric):
    sub = X[perm[e["idx"]]].copy()
    if e["col"] is not None:
        sub[:, e["col"]] = 0.0
    S, ok = simmat(sub, metric)
    if ok.sum() < 3:
        return None
    valid = ok[e["iu"]] & ok[e["ju"]]
    lab = e["lab"]
    same = (lab[e["iu"]] == lab[e["ju"]]) & valid
    diff = (lab[e["iu"]] != lab[e["ju"]]) & valid
    if same.sum() == 0 or diff.sum() == 0:
        return None
    sims = S[e["iu"], e["ju"]].copy()
    sims[~valid] = np.nan
    return sims


def stat_perm(edges, X, perm, metric, agg):
    s_all, d_all, per_edge = [], [], []
    for e in edges:
        sims = permuted_sims(e, X, perm, metric)
        if sims is None:
            continue
        lab = e["lab"]
        v = ~np.isnan(sims)
        same = (lab[e["iu"]] == lab[e["ju"]]) & v
        diff = (lab[e["iu"]] != lab[e["ju"]]) & v
        ss, dd = sims[same], sims[diff]
        s_all.append(ss)
        d_all.append(dd)
        per_edge.append(ss.mean() - dd.mean())
    if not per_edge:
        return np.nan, np.nan
    agg_fn = np.mean if agg == "mean" else np.median
    return (float(np.concatenate(s_all).mean() - np.concatenate(d_all).mean()),
            float(agg_fn(per_edge)))


def run(name, edges, X, strata, metric="cosine", agg="mean", n_perm=N_PERM):
    obs = stat(edges, lambda e: e["sims"], agg=agg)
    vals = np.zeros((n_perm, 2))
    n = X.shape[0]
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    singleton = sum(len(gp) for gp in groups if len(gp) == 1)
    for k in range(n_perm):
        perm = np.arange(n)
        for gp in groups:
            perm[gp] = RNG.permutation(gp)
        vals[k] = stat_perm(edges, X, perm, metric, agg)
    res = {"n_edges": len(edges),
           "n_pairs": int(sum(len(e["sims"]) for e in edges)),
           "singleton_stratum_papers": int(singleton)}
    for j, sname in enumerate(("pooled", "per_edge")):
        null = vals[:, j][~np.isnan(vals[:, j])]
        o = obs[j]
        p = (np.sum(np.abs(null - null.mean()) >= abs(o - null.mean())) + 1) / (
            len(null) + 1)
        res[sname] = dict(observed=o, null_mean=float(null.mean()),
                          null_sd=float(null.std()), p=float(p),
                          min_detectable=float(1.96 * null.std()))
    print(f"{name:28s} edges {res['n_edges']:3d} pairs {res['n_pairs']:5d} | "
          f"pooled {res['pooled']['observed']:+.4f} p={res['pooled']['p']:.3f} | "
          f"per-edge {res['per_edge']['observed']:+.4f} "
          f"p={res['per_edge']['p']:.3f} "
          f"(null sd {res['per_edge']['null_sd']:.4f}, "
          f"mde {res['per_edge']['min_detectable']:.4f})")
    return res


def main():
    g, titles, X, vocab, disease = load()
    papers = g["papers"]
    country = np.array([p.get("country") or "?" for p in papers])
    seq = np.array([p.get("seq") or "?" for p in papers])
    size = X.sum(1)
    qs = np.quantile(size[size > 0], [0.2, 0.4, 0.6, 0.8])
    size_str = np.digitize(size, qs).astype(str)

    edges = build_edges(g, X, vocab, "cosine")
    out = {}

    print("=== descriptive: is the dissenter simply an atypical paper? ===")
    maj, minor = [], []
    for e in edges:
        lab = e["lab"]
        mj = 1 if e["n_up"] >= e["n_dn"] else 0
        for i, l in zip(e["idx"], lab):
            (maj if l == mj else minor).append(size[i])
    print(f"profile size (taxa/paper): majority-side members n={len(maj)} "
          f"mean {np.mean(maj):.1f} median {np.median(maj):.0f} | "
          f"minority-side n={len(minor)} mean {np.mean(minor):.1f} "
          f"median {np.median(minor):.0f}")
    out["profile_size"] = dict(majority_mean=float(np.mean(maj)),
                               minority_mean=float(np.mean(minor)),
                               majority_n=len(maj), minority_n=len(minor))
    bal = [e for e in edges if min(e["n_up"], e["n_dn"]) >= 2]
    print(f"edges with >=2 papers on both sides: {len(bal)} of {len(edges)}")

    print("\n=== variants (null: profile permutation within stratum) ===")
    out["all_disease"] = run("all edges / disease", edges, X, disease)
    out["balanced"] = run("balanced edges only", bal, X, disease)

    # near-duplicate pairs removed
    ded = []
    ndrop = 0
    for e in edges:
        m = e["sims"] < 0.8
        ndrop += int((~m).sum())
        lab = e["lab"]
        same = lab[e["iu"]] == lab[e["ju"]]
        if (same & m).sum() == 0 or ((~same) & m).sum() == 0:
            continue
        f = dict(e)
        f["iu"], f["ju"], f["sims"] = e["iu"][m], e["ju"][m], e["sims"][m]
        ded.append(f)
    print(f"(near-duplicate pairs with cosine>=0.8 dropped: {ndrop})")
    out["dedup"] = run("no near-duplicate pairs", ded, X, disease)

    out["size_stratified"] = run("perm within size quintile", edges, X, size_str)
    out["country"] = run("perm within country", edges, X, country)
    out["seq"] = run("perm within sequencing", edges, X, seq)

    ej = build_edges(g, X, vocab, "jaccard")
    out["jaccard"] = run("jaccard metric", ej, X, disease, metric="jaccard")
    out["median"] = run("per-edge median", edges, X, disease, agg="median")

    # where does it come from: per-edge effect distribution
    de = []
    for e in edges:
        lab = e["lab"]
        same = lab[e["iu"]] == lab[e["ju"]]
        de.append((float(e["sims"][same].mean() - e["sims"][~same].mean()),
                   e["key"], e["n_up"], e["n_dn"]))
    de.sort(reverse=True)
    pos = sum(1 for d, *_ in de if d > 0)
    print(f"\nper-edge effect: {pos} of {len(de)} edges positive "
          f"({100*pos/len(de):.0f}%), median {np.median([d for d,*_ in de]):+.4f}")
    out["per_edge_positive"] = dict(positive=pos, total=len(de),
                                    median=float(np.median([d for d, *_ in de])))
    out["top_edges"] = [dict(effect=d, taxon=k[0], disease=k[1], n_up=u, n_dn=n)
                        for d, k, u, n in de[:10]]
    out["bottom_edges"] = [dict(effect=d, taxon=k[0], disease=k[1], n_up=u, n_dn=n)
                           for d, k, u, n in de[-5:]]
    print("strongest edges:")
    for d, k, u, n in de[:5]:
        print(f"   {d:+.3f}  {k[0]} / {k[1]}  ({u}up/{n}dn)")

    json.dump(out, open("cooccur_diagnostics.json", "w"), indent=1)
    print("\nwrote cooccur_diagnostics.json")


if __name__ == "__main__":
    main()
