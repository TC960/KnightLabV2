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
contested edges, and four independent tests have now returned nulls on that
front. This one does not depend on paper count at all. It uses only the
containment links, and that part of the data is complete and clean:

    nodes with more than one parent : 0   (a strict tree, not a DAG)
    ranks present                   : phylum 30, class 44, order 70,
                                      family 127, genus 278, species 267

Scale is not a stretch either. Nickel & Kiela's own small-data demo is the WordNet
mammals subtree at 1,180 nodes / 6,540 edges, reaching MAP 0.927 at d=5. Our
taxonomy is 916 nodes / 915 links -- the same regime, not an extrapolation.

READS THE REPAIRED TREE, NOT graph.json. This is load-bearing and was the first
of three bugs here. `graph.json`'s hierarchy only links taxa that BOTH appear in
our papers, so it is a forest of 20 fragments with holes in every lineage, and
depth-within-fragment is not rank. Trained on that, the geometry gave
Spearman(radius, rank) = 0.0095, p = 0.81 -- it learned fragment depth faithfully,
and fragment depth means nothing. `repair_taxonomy_tree.py` walks full NCBI
lineages and inserts the missing ancestors as Steiner nodes; this script consumes
that output. Steiner nodes are kept in training (they hold the tree together) but
EXCLUDED from the rank test, since we have no evidence about them.

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
TREE = os.path.join(HERE, "taxonomy_tree.json")
OUT = os.path.join(HERE, "hyperbolic_taxonomy.json")

# canonical depth; 'clade'/'no rank' are 16S placeholder labels with no true depth
RANK_DEPTH = {"superkingdom": 0, "kingdom": 0, "domain": 0, "phylum": 1,
              "class": 2, "order": 3, "family": 4, "genus": 5, "species": 6,
              "subspecies": 7, "strain": 7}


def load_tree():
    """The repaired tree: taxid -> parent taxid, plus which taxids we observed."""
    t = json.load(open(TREE))
    parent, rank = t["parent"], t["rank"]
    observed = set(t["observed"])
    edges = [(p, c) for c, p in parent.items()]
    nodes = sorted(set(parent) | set(parent.values()))
    return nodes, edges, rank, observed


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

    THREE BUGS DIED HERE; each one produced a plausible-looking number.

    1. A hand-rolled numeric gradient made loss RISE after epoch 50 and every
       rank saturate at radius ~0.99 -- the embedding collapsed onto the
       boundary of the disc and the radius stopped carrying depth. Fixed by
       autograd plus clipping the max norm to 1-1e-5 BEFORE the arccosh, so the
       distance never sees a degenerate denominator.
    2. Training on graph.json's 20-fragment forest. See the module docstring.
    3. Negatives drawn uniformly over ALL nodes -- so a node's own grandparent
       could be sampled as a negative and pushed AWAY, directly fighting the
       positive term that pulls it close. The result was not merely weak, it was
       INVERTED: rho = -0.178. Nickel & Kiela define the negative set as
       N(u) = {v : (u,v) not in D}, i.e. explicitly the non-ancestors. `anc`
       below is that exclusion, applied by resampling collisions.

    The inversion is the instructive one: a sign flip is not noise, and a
    correlation that comes out backwards is a bug hypothesis, not a finding.
    """
    import torch

    torch.manual_seed(seed)
    P = torch.tensor(pairs, dtype=torch.long)

    # anc[u, v] = True iff v is a true ancestor of u (never a valid negative).
    # 916x916 bools is 839 KB -- cheaper than rejection-testing sets per batch.
    anc = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    anc[P[:, 0], P[:, 1]] = True
    anc[torch.arange(n_nodes), torch.arange(n_nodes)] = True   # nor itself
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
            # resample any negative that is actually an ancestor of its anchor;
            # ancestor sets are <=10 of 916 nodes, so this clears in a few passes
            for _ in range(10):
                bad = anc[u.unsqueeze(1).expand_as(neg), neg]
                if not bool(bad.any()):
                    break
                neg[bad] = torch.randint(0, n_nodes, (int(bad.sum()),))
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

    nodes, edges, rank, observed = load_tree()
    nid = {n: i for i, n in enumerate(nodes)}
    pairs = [(nid[c], nid[p]) for c, p in transitive_pairs(edges)]
    print(f"tree: {len(nodes)} nodes, {len(edges)} direct links, "
          f"{len(pairs)} transitive (descendant, ancestor) pairs")

    ranks = [rank.get(n, "") for n in nodes]
    # Steiner nodes trained on but not scored: we inserted them, so 'recovering'
    # their rank would be grading the geometry on our own scaffolding.
    known = [i for i, r in enumerate(ranks)
             if r in RANK_DEPTH and nodes[i] in observed]
    print(f"scoreable nodes: {len(known)} of {len(nodes)} "
          f"(observed taxa with a canonical rank; "
          f"{len(nodes) - len(observed)} inserted ancestors excluded)")
    print(f"training d={a.dim}, {a.epochs} epochs ...")

    emb = train(pairs, len(nodes), a.dim, a.epochs, a.neg, a.lr)
    norm = np.linalg.norm(emb, axis=1)

    # ---- THE TEST: does radius recover rank depth? -----------------------
    d = np.array([RANK_DEPTH[ranks[i]] for i in known])
    r = norm[known]
    from scipy.stats import rankdata, spearmanr
    rho, p = spearmanr(r, d)
    rng = np.random.default_rng(0)
    null = [spearmanr(r, rng.permutation(d)).statistic for _ in range(1000)]
    null = np.array(null)

    print()
    print(f"Spearman(norm, rank depth) = {rho:+.4f}   p = {p:.2e}")
    print(f"  shuffled-rank null: mean {null.mean():+.4f}, "
          f"95th pct |rho| {np.percentile(np.abs(null), 95):.4f}")
    print(f"  beats null: {abs(rho) > np.percentile(np.abs(null), 95)}")
    # THE DEFLATIONARY CHECK. A Poincare embedding is SUPPOSED to put depth on
    # the radius -- that is its training objective, not a discovery. And after
    # repair_taxonomy_tree.py, tree depth already tracks rank by construction.
    # So the honest question is whether radius knows anything BEYOND depth.
    # If rho(radius, rank) is fully explained by depth, this run is a sanity
    # check that the geometry trained correctly, and nothing more.
    # `edges` holds (parent, child) -- unpacking it the other way round silently
    # produced depth measured from the LEAVES, and a -0.70 depth/rank
    # correlation that should have been impossible. Sanity-checked below.
    par = {nid[c]: nid[p] for p, c in edges}

    def tree_depth(i):
        d, seen = 0, set()
        while i in par and i not in seen:
            seen.add(i)
            i = par[i]
            d += 1
        return d

    dep = np.array([tree_depth(i) for i in known])
    rho_dep = spearmanr(r, dep).statistic
    rho_rank_dep = spearmanr(dep, d).statistic
    if rho_rank_dep < 0:
        raise SystemExit("depth anti-correlates with rank -- the tree is upside down")

    # Spearman PARTIAL correlation: rank-transform all three, residualise both
    # radius and rank on depth, correlate what is left. If this is ~0 the run is
    # a sanity check on the training, not a result about taxonomy.
    def resid(y, x):
        rx, ry = rankdata(x), rankdata(y)
        return ry - np.polyval(np.polyfit(rx, ry, 1), rx)

    rho_partial = spearmanr(resid(r, dep), resid(d, dep)).statistic
    # printed to 6dp because at 4dp these two landed on an identical +0.7210
    # and an apparent identity between two statistics is a bug hypothesis
    print(f"Spearman(norm, TREE DEPTH)  = {rho_dep:+.6f}    "
          f"Spearman(depth, rank) = {rho_rank_dep:+.6f}")
    print(f"  radius vs rank, controlling for depth: {rho_partial:+.4f}")

    print()
    print("mean radius by rank (should increase down the list):")
    by = collections.defaultdict(list)
    for i in known:
        by[ranks[i]].append(norm[i])
    ordered = sorted(by, key=lambda k: RANK_DEPTH[k])
    # Monotonicity is judged only over ranks with >= MIN_N members. `subspecies`
    # has n=1 and `strain` n=5 in this corpus; letting a single taxon veto the
    # ordering reports False for a run whose six well-populated ranks are in
    # perfect order. The excluded ranks are still printed, marked (n too small).
    MIN_N = 10
    prev, mono = -1, True
    for k in ordered:
        m = float(np.mean(by[k]))
        if len(by[k]) < MIN_N:
            print(f"    {k:<12} n={len(by[k]):>4}  mean radius {m:.4f}"
                  f"   hyperbolic {2*math.atanh(min(m, 1-1e-12)):5.2f}"
                  f"   (n too small to score)")
            continue
        flag = "" if m >= prev else "   <-- OUT OF ORDER"
        if m < prev:
            mono = False
        prev = m
        # Euclidean radius saturates near 1 and LOOKS collapsed even when it is
        # not; hyperbolic distance from the origin, 2*artanh(r), is the honest
        # scale -- 0.990 and 0.9966 are 2.65 and 3.39 apart, not "both ~1".
        print(f"    {k:<12} n={len(by[k]):>4}  mean radius {m:.4f}"
              f"   hyperbolic {2*math.atanh(min(m, 1-1e-12)):5.2f}{flag}")
    print(f"\n  monotonic across ranks: {mono}")

    np.save(os.path.join(HERE, "hyperbolic_emb.npy"), emb)
    json.dump({"dim": a.dim, "epochs": a.epochs, "n_nodes": len(nodes),
               "n_pairs": len(pairs), "n_scored": len(known),
               "nodes": nodes, "spearman_rho": float(rho),
               "p": float(p), "null_95": float(np.percentile(np.abs(null), 95)),
               "rho_tree_depth": float(rho_dep),
               "rho_depth_rank": float(rho_rank_dep),
               "rho_partial_rank_given_depth": float(rho_partial),
               "monotonic": bool(mono),
               "mean_radius_by_rank": {k: float(np.mean(by[k])) for k in ordered}},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT} and hyperbolic_emb.npy")


if __name__ == "__main__":
    main()
