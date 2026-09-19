#!/usr/bin/env python3
"""Embed the taxonomy in hyperbolic space, then test whether NORM recovers RANK.

WHY HYPERBOLIC. Trees grow exponentially with depth: the number of nodes at depth
d multiplies. Euclidean balls only grow polynomially in radius, so a tree embedded
in R^n is always cramped. Hyperbolic discs grow exponentially in circumference
(2*pi*sinh(r)), which matches. Sarkar (2011) proved any finite tree embeds in
hyperbolic space with (1+eps) distortion for any eps; no such result exists for
finite-dimensional Euclidean space.

WHY IT SUITS *THIS* GRAPH, which is the non-obvious part. Every embedding attempt
in this project so far has died on power -- 303 papers is too few to separate
contested edges. This one does not depend on paper count at all. It uses only the
727 containment links, and that part of the data is complete and clean:

    nodes with more than one parent : 0   (a strict tree, not a DAG)
    ranks present                   : phylum 20, class 22, order 33,
                                      family 89, genus 305, species 301

Scale is not a stretch either. Nickel & Kiela's own small-data demo is the WordNet
mammals subtree at 1,180 nodes / 6,540 edges, reaching MAP 0.927 at d=5. Our
taxonomy is ~900 nodes / 727 links -- the same regime, not an extrapolation.

THE FALSIFIABLE TEST. Depth should fall out of the geometry: the training
objective pulls the root toward the origin and pushes leaves outward, so
||embedding|| should increase monotonically phylum < class < order < family <
genus < species. We KNOW the true rank of every node, so if norm does not recover
that ordering, the approach is wrong for this data and we stop. Reported as
Spearman correlation between norm and rank depth, against a shuffled-rank null.

    python hyperbolic_taxonomy.py --dim 5
    python hyperbolic_taxonomy.py --dim 10 --epochs 300
"""
import argparse
import collections
import json
import math
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "hyperbolic_taxonomy.json")

# canonical depth; 'clade'/'no rank' are 16S placeholder labels with no true depth
RANK_DEPTH = {"kingdom": 0, "domain": 0, "phylum": 1, "class": 2, "order": 3,
              "family": 4, "genus": 5, "species": 6, "subspecies": 7, "strain": 7}


def load_tree():
    g = json.load(open(GRAPH))
    lab = {n["id"]: n for n in g["nodes"]}
    edges = []
    for e in (g.get("hierarchy") or []):
        p = e.get("parent") or e.get("source")
        c = e.get("child") or e.get("target")
        if p and c and p in lab and c in lab:
            edges.append((p, c))
    nodes = sorted({x for e in edges for x in e})
    return nodes, edges, lab


def transitive_pairs(edges):
    """(descendant, ancestor) for every ancestor, not just the direct parent.

    Nickel & Kiela train on the transitive closure: a node must be closer to ALL
    its ancestors, which is what forces depth onto the radius.
    """
    par = {c: p for p, c in edges}
    out = set()
    for c in par:
        seen, cur = set(), c
        while cur in par and par[cur] not in seen:
            seen.add(par[cur])
            out.add((c, par[cur]))
            cur = par[cur]
    return sorted(out)


def poincare_dist(u, v, eps=1e-7):
    su = np.sum(u * u, axis=-1)
    sv = np.sum(v * v, axis=-1)
    d2 = np.sum((u - v) ** 2, axis=-1)
    x = 1 + 2 * d2 / np.maximum((1 - su) * (1 - sv), eps)
    return np.arccosh(np.maximum(x, 1 + eps))


def project(x, eps=1e-5):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    f = np.minimum(1.0, (1 - eps) / np.maximum(n, eps))
    return x * f


def train(pairs, n_nodes, dim, epochs, n_neg, lr, seed=0):
    """Poincare embedding, Nickel & Kiela 2017, with torch autograd.

    A hand-rolled numeric gradient was tried first and FAILED in a specific,
    recognisable way: loss rose after epoch 50 and every rank saturated at radius
    ~0.99, i.e. the whole embedding collapsed onto the boundary of the disc and
    the radius stopped carrying depth. Two things fix it and both matter --
    autograd for a correct gradient, and clipping the max norm to 1-1e-5 BEFORE
    the arccosh so the distance never sees a degenerate denominator.
    """
    import torch

    torch.manual_seed(seed)
    P = torch.tensor(pairs, dtype=torch.long)
    emb = torch.nn.Parameter(torch.randn(n_nodes, dim) * 1e-3)
    opt = torch.optim.SGD([emb], lr=lr)

    def pdist(u, v, eps=1e-7):
        su = (u * u).sum(-1)
        sv = (v * v).sum(-1)
        d2 = ((u - v) ** 2).sum(-1)
        x = 1 + 2 * d2 / torch.clamp((1 - su) * (1 - sv), min=eps)
        return torch.acosh(torch.clamp(x, min=1 + eps))

    n_batch = max(1, len(P) // 256)
    for ep in range(epochs):
        # burn-in: a tenth of the lr for the first 20 epochs. Without it the
        # embedding rushes to the boundary in the first few steps and never
        # recovers -- that is exactly what the failed numeric version did.
        for gp in opt.param_groups:
            gp["lr"] = lr / 10 if ep < 20 else lr
        perm = torch.randperm(len(P))
        total = 0.0
        for start in range(0, len(P), 256):
            b = P[perm[start:start + 256]]
            u, v = b[:, 0], b[:, 1]
            neg = torch.randint(0, n_nodes, (len(b), n_neg))
            du = pdist(emb[u], emb[v])
            dn = pdist(emb[u].unsqueeze(1), emb[neg])
            # the true ancestor must be the nearest among {ancestor, negatives}
            logits = -torch.cat([du.unsqueeze(1), dn], dim=1)
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(b), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            # Riemannian rescaling of the Euclidean gradient
            with torch.no_grad():
                sc = ((1 - (emb ** 2).sum(-1, keepdim=True)) ** 2) / 4
                emb.grad *= sc
            opt.step()
            with torch.no_grad():                       # stay inside the disc
                n = emb.norm(dim=-1, keepdim=True)
                emb.data *= torch.clamp((1 - 1e-5) / n, max=1.0)
            total += float(loss)
        if (ep + 1) % 50 == 0:
            print(f"    epoch {ep+1:>4}  loss {total/n_batch:.4f}", flush=True)
    return emb.detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.3)
    a = ap.parse_args()

    nodes, edges, lab = load_tree()
    nid = {n: i for i, n in enumerate(nodes)}
    pairs = [(nid[c], nid[p]) for c, p in transitive_pairs(edges)]
    print(f"tree: {len(nodes)} nodes, {len(edges)} direct links, "
          f"{len(pairs)} transitive (descendant, ancestor) pairs")

    ranks = [lab[n].get("rank", "") for n in nodes]
    known = [i for i, r in enumerate(ranks) if r in RANK_DEPTH]
    print(f"nodes with a canonical rank: {len(known)} of {len(nodes)} "
          f"(the rest are 16S clade labels with no true depth)")
    print(f"training d={a.dim}, {a.epochs} epochs ...")

    emb = train(pairs, len(nodes), a.dim, a.epochs, a.neg, a.lr)
    norm = np.linalg.norm(emb, axis=1)

    # ---- THE TEST: does radius recover rank depth? -----------------------
    d = np.array([RANK_DEPTH[ranks[i]] for i in known])
    r = norm[known]
    from scipy.stats import spearmanr
    rho, p = spearmanr(r, d)
    rng = np.random.default_rng(0)
    null = [spearmanr(r, rng.permutation(d)).statistic for _ in range(1000)]
    null = np.array(null)

    print()
    print(f"Spearman(norm, rank depth) = {rho:+.4f}   p = {p:.2e}")
    print(f"  shuffled-rank null: mean {null.mean():+.4f}, "
          f"95th pct |rho| {np.percentile(np.abs(null), 95):.4f}")
    print(f"  beats null: {abs(rho) > np.percentile(np.abs(null), 95)}")
    print()
    print("mean radius by rank (should increase down the list):")
    by = collections.defaultdict(list)
    for i in known:
        by[ranks[i]].append(norm[i])
    ordered = sorted(by, key=lambda k: RANK_DEPTH[k])
    prev, mono = -1, True
    for k in ordered:
        m = float(np.mean(by[k]))
        flag = "" if m >= prev else "   <-- OUT OF ORDER"
        if m < prev:
            mono = False
        prev = m
        print(f"    {k:<12} n={len(by[k]):>4}  mean radius {m:.4f}{flag}")
    print(f"\n  monotonic across ranks: {mono}")

    json.dump({"dim": a.dim, "epochs": a.epochs, "n_nodes": len(nodes),
               "n_pairs": len(pairs), "spearman_rho": float(rho),
               "p": float(p), "null_95": float(np.percentile(np.abs(null), 95)),
               "monotonic": bool(mono),
               "mean_radius_by_rank": {k: float(np.mean(by[k])) for k in ordered}},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
